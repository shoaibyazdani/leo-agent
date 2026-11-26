"""Phase 6: toolset intersection, frozen skill docs and wrap_response."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo.agents.engine import AgentEngine
from leo.contracts import AgentSpec, ToolSpec
from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompileError, ScheduleCompiler
from leo.schedules.delivery import DeliveryContext, Origin
from leo.schedules.execution import ScheduleExecution, apply_wrap_response
from leo.schedules.store import ScheduleStore
from leo.tools.registry import ToolRegistry

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)
BASE = {"schedule": "every day at 9am", "prompt": "Brief", "deliver": "local"}


async def _handler(args, ctx):  # pragma: no cover - never invoked here
    return {"ok": True}


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name, category in (
        ("read_file", "filesystem"),
        ("write_file", "filesystem"),
        ("shell", "development"),
        ("http_get", "network"),
    ):
        registry.register(
            ToolSpec(name=name, description=name, category=category), _handler
        )
    return registry


def _engine() -> AgentEngine:
    engine = object.__new__(AgentEngine)
    engine.tools = SimpleNamespace(registry=_registry())
    return engine


def _spec() -> AgentSpec:
    return AgentSpec(
        id="scheduler",
        name="Scheduler",
        purpose="Run scheduled work",
        tool_permissions=["filesystem", "development"],
    )


def test_enabled_toolsets_only_narrow_the_agent_scope() -> None:
    engine = _engine()
    spec = _spec()

    inherited = engine._scoped_tools(spec)
    assert {item.name for item in inherited.specs()} == {
        "read_file", "write_file", "shell"
    }

    empty = engine._scoped_tools(spec, [])
    assert empty.specs() == []
    assert empty.allow == frozenset()

    narrowed = engine._scoped_tools(spec, ["filesystem"])
    assert {item.name for item in narrowed.specs()} == {"read_file", "write_file"}

    # An enabled toolset outside the agent's declared scope cannot widen it.
    outside = engine._scoped_tools(spec, ["network"])
    assert outside.specs() == []
    assert "http_get" not in outside.allow


def test_unknown_toolset_is_rejected_at_compile() -> None:
    compiler = _compiler(available_toolsets={"filesystem", "development"})
    with pytest.raises(ScheduleCompileError) as caught:
        compiler.compile_create({**BASE, "enabled_toolsets": ["shell"]})
    assert caught.value.code == "unknown_toolset"
    accepted = compiler.compile_create(
        {**BASE, "enabled_toolsets": ["filesystem"]}
    )
    assert accepted.spec.execution.enabled_toolsets == ["filesystem"]


def _compiler(**overrides) -> ScheduleCompiler:
    values = {
        "skill_docs": {"maps": "MAP DOC V1", "brief": "BRIEF V1"},
        "delivery_context": DeliveryContext(origin=Origin("telegram", "1")),
    }
    values.update(overrides)
    return ScheduleCompiler(FrozenClock(NOW), default_timezone="UTC", **values)


def test_skill_documentation_is_frozen_at_creation() -> None:
    compiler = _compiler()
    created = compiler.compile_create({**BASE, "skills": ["maps"]})
    assert created.spec.execution.skills == ["maps"]
    assert created.config["skill_snapshot"] == {"maps": "MAP DOC V1"}

    base = created.spec.model_copy(update={"config": dict(created.config)})
    newer = _compiler(skill_docs={"maps": "MAP DOC V2", "brief": "BRIEF V2"})
    prompt_edit = newer.compile_update(base, {"prompt": "Changed"}, base_revision=1)
    assert prompt_edit.config["skill_snapshot"] == {"maps": "MAP DOC V1"}

    skill_edit = newer.compile_update(base, {"skills": ["maps"]}, base_revision=1)
    assert skill_edit.config["skill_snapshot"] == {"maps": "MAP DOC V2"}

    cleared = newer.compile_update(base, {"skills": []}, base_revision=1)
    assert not cleared.config.get("skill_snapshot")


async def test_skill_snapshot_is_injected_not_live_read() -> None:
    execution = ScheduleExecution(
        settings=SimpleNamespace(
            automation=SimpleNamespace(
                allow_scripts=False,
                script_timeout_seconds=60,
                script_max_output_bytes=1000,
                monitor_timeout_seconds=5,
                monitor_max_bytes=1000,
            ),
            models=SimpleNamespace(providers={}),
        ),
        store=ScheduleStore(clock=FrozenClock(NOW)),
        clock=FrozenClock(NOW),
    )
    task = SimpleNamespace(id="t1", objective="job", metadata={"schedule_id": "sch_1"})
    prepared = await execution.prepare_agent(
        task=task,
        context={
            "schedule_payload": {
                "name": "job",
                "execution": {"skills": ["maps"], "enabled_toolsets": []},
                "skill_snapshot": {"maps": "FROZEN DOC"},
            }
        },
    )
    assert "Frozen skill documentation" in prepared.objective
    assert "FROZEN DOC" in prepared.objective
    assert prepared.context["enabled_toolsets"] == []


def test_wrap_response_controls_text_and_fingerprint() -> None:
    compiler = _compiler()
    created = compiler.compile_create(dict(BASE))
    base = created.spec.model_copy(update={"config": dict(created.config)})
    on = compiler.compile_update(base, {"wrap_response": "on"}, base_revision=1)
    off = compiler.compile_update(base, {"wrap_response": "off"}, base_revision=1)
    inherit = compiler.compile_update(
        base, {"wrap_response": "inherit"}, base_revision=1
    )
    assert on.edit_fingerprint != off.edit_fingerprint
    assert inherit.edit_fingerprint != on.edit_fingerprint

    assert apply_wrap_response(
        "body", {"name": "job", "execution": {"wrap_response": "on"}}
    ) == "📅 job\n\nbody"
    assert apply_wrap_response(
        "body", {"name": "job", "execution": {"wrap_response": "off"}}
    ) == "body"
    assert apply_wrap_response(
        "body", {"name": "job", "execution": {"wrap_response": "inherit"}}
    ) == "body"
    assert apply_wrap_response("body", {"name": "job"}) == "body"
