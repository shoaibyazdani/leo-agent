"""Phase 6: operator-owned pinning, fallback and fail-closed needs-you."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import ModelProviderConfig, Settings
from leo.core.app import LeoApp
from leo.events import EventBus
from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompileError, ScheduleCompiler
from leo.schedules.delivery import DeliveryContext, Origin
from leo.schedules.execution import ExecutionError, ScheduleExecution
from leo.schedules.store import ScheduleStore

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)
BASE = {"schedule": "every day at 9am", "prompt": "Brief", "deliver": "local"}


def _settings() -> Settings:
    settings = Settings()
    settings.models.providers = {
        "provider-x": ModelProviderConfig(
            kind="openai-compatible", model="model-y", base_url="https://example.test/v1"
        )
    }
    settings.models.default = "provider-x"
    return settings


def _compiler(settings: Settings, **overrides) -> ScheduleCompiler:
    values = {
        "pinned_provider": "provider-x",
        "pinned_model": "model-y",
        "delivery_context": DeliveryContext(origin=Origin("telegram", "1")),
    }
    values.update(overrides)
    return ScheduleCompiler(FrozenClock(NOW), default_timezone="UTC", **values)


def _execution(settings: Settings) -> ScheduleExecution:
    clock = FrozenClock(NOW)
    return ScheduleExecution(
        settings=settings,
        store=ScheduleStore(clock=clock),
        clock=clock,
    )


async def _prepare(execution: ScheduleExecution, execution_config: dict):
    task = SimpleNamespace(id="t1", objective="job", metadata={"schedule_id": "sch_1"})
    return await execution.prepare_agent(
        task=task,
        context={
            "schedule_payload": {"name": "job", "execution": execution_config},
        },
    )


def test_model_cannot_choose_provider_model_or_base_url() -> None:
    compiler = _compiler(_settings())
    for key, value in (
        ("provider", "other"),
        ("model", "other-model"),
        ("base_url", "https://evil.test/v1"),
        ("grace_minutes", 5),
        ("script_args", ["--x"]),
    ):
        with pytest.raises(ScheduleCompileError) as caught:
            compiler.compile_create({**BASE, key: value})
        assert caught.value.code == "operator_only_field"


def test_pinned_creation_freezes_the_current_operator_target() -> None:
    created = _compiler(_settings()).compile_create({**BASE, "pinned": True})
    assert created.spec.execution.pinned is True
    assert created.spec.execution.provider == "provider-x"
    assert created.spec.execution.model == "model-y"

    unpinned = _compiler(_settings()).compile_create({**BASE, "pinned": False})
    assert unpinned.spec.execution.pinned is False
    assert unpinned.spec.execution.provider is None
    assert unpinned.spec.execution.model is None


async def test_pinned_true_uses_stored_target_and_false_inherits() -> None:
    settings = _settings()
    execution = _execution(settings)
    pinned = await _prepare(
        execution,
        {"pinned": True, "provider": "provider-x", "model": "model-y"},
    )
    assert pinned.context["pinned_provider"] == "provider-x"
    assert pinned.context["pinned_model"] == "model-y"

    default = await _prepare(
        execution,
        {"pinned": False, "provider": None, "model": None},
    )
    assert "pinned_provider" not in default.context
    assert "pinned_model" not in default.context


async def test_missing_pinned_target_fails_closed_and_notices() -> None:
    settings = _settings()
    settings.models.providers = {}
    execution = _execution(settings)
    with pytest.raises(ExecutionError) as caught:
        await _prepare(
            execution,
            {"pinned": True, "provider": "provider-x", "model": "model-y"},
        )
    assert caught.value.code == "pinned_target_unavailable"

    with pytest.raises(ExecutionError) as caught:
        await _prepare(execution, {"pinned": True, "provider": None, "model": None})
    assert caught.value.code == "pinned_target_missing"


async def test_app_wires_scripts_root_roster_and_operator_target(tmp_path) -> None:
    from leo.config import ModelConfig

    settings = Settings()
    settings.models = ModelConfig(
        default="mock",
        fallback="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    settings.security.allowed_roots = [tmp_path]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    settings.automation.scripts_root = scripts
    settings.automation.allow_scripts = True

    app = LeoApp(settings)
    await app.initialize()
    try:
        assert app.schedule_service.scripts_root == scripts.resolve()
        assert app.schedule_service.pinned_provider == "mock"
        assert app.schedule_service.pinned_model == "mock-1"
        roster = set(app.schedule_service.available_toolsets or ())
        assert "filesystem" in roster
        assert app.scheduler.execution is app.schedule_execution
        assert app.schedule_execution.scripts.allow_scripts is True
    finally:
        await app.shutdown()


async def test_missing_pinned_target_emits_one_durable_needs_you() -> None:
    await db.initialize()
    events = EventBus()
    app = object.__new__(LeoApp)
    app.events = events
    task = SimpleNamespace(
        id="t1",
        metadata={"schedule": "job", "schedule_id": "sch_1"},
    )
    exc = ExecutionError(
        "pinned_target_unavailable", "The pinned provider is gone."
    )
    await LeoApp._schedule_worker_failed(app, task, {}, exc)
    history = await events.history("automation.needs_you", limit=10)
    assert len(history) == 1
    assert history[0].payload["reason"] == "pinned_target_unavailable"
    assert history[0].payload["schedule_id"] == "sch_1"
