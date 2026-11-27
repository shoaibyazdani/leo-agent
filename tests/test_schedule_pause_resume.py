"""Explicit pause/resume tools: consent, bounded reasons and stable cadence.

Both tools are thin wrappers over the one ScheduleService update path. A model
call needs an edit card; an operator call is direct. Neither changes the stored
instant or repeat limit, and a create grant never substitutes for edit consent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall, TurnContext, turn_context
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.schedules.clock import FrozenClock
from leo.schedules.service import ScheduleService
from leo.schedules.store import ScheduleStore
from leo.tasks import DurableTaskEngine, Scheduler
from leo.tools import ToolExecutorImpl, ToolRegistry, automation_tools

NOW = datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc)


class ServicesExecutor(ToolExecutorImpl):
    """Inject the integration-owned services mapping into each tool context."""

    def __init__(self, *args: object, services: dict | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.injected_services = services or {}

    def _context(self, call, task_key, dry_run):  # type: ignore[no-untyped-def]
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.injected_services
        return ctx


def _operator_turn() -> TurnContext:
    return TurnContext(
        principal="operator:cli", channel="cli", surface="cli", source="operator"
    )


def _chat_turn() -> TurnContext:
    return TurnContext(
        principal="chat:web",
        channel="tui",
        conversation_id="conv-pause",
        surface="tui",
        origin="tui",
        source="chat",
    )


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    clock = FrozenClock(NOW)
    permissions = PermissionManagerImpl(settings, events, clock=clock)
    engine = DurableTaskEngine(settings, events)
    scheduler = Scheduler(engine, settings, events, clock=clock)
    store = ScheduleStore(clock=clock)
    service = ScheduleService(
        store=store,
        scheduler=scheduler,
        engine=engine,
        permissions=permissions,
        events=events,
        settings=settings,
        clock=clock,
        scripts_root=tmp_path,
        allowed_workdir_roots=[tmp_path],
    )
    services = {
        "schedule_service": service,
        "scheduler": scheduler,
        "tasks": engine,
        "permissions": permissions,
        "settings": settings,
        "events": events,
    }
    registry = ToolRegistry()
    automation_tools.register(registry, services=services)
    executor = ServicesExecutor(
        registry, permissions, events, settings, services=services
    )
    return SimpleNamespace(
        permissions=permissions, service=service, executor=executor, registry=registry
    )


async def run(env, name: str, arguments: dict, *, turn=None, actor="chat:web"):
    call = ToolCall(name=name, arguments=arguments, actor=actor)
    if turn is None:
        return await env.executor.execute(call)
    with turn_context(turn):
        return await env.executor.execute(call)


async def run_operator(env, name: str, arguments: dict):
    return await run(env, name, arguments, turn=_operator_turn(), actor="operator:cli")


async def row(name: str = "pause-me"):
    return await db.fetchone("SELECT * FROM schedules WHERE name=?", (name,))


async def create_operator_job(env):
    created = await run_operator(env, "schedule_create", {
        "name": "pause-me", "schedule": "every 2h", "prompt": "check queue",
    })
    assert created.output["status"] == "created", created.error
    return await row()


async def resolve_card(env, result) -> dict:
    card = result.output["card"]
    assert card["kind"] == "schedule_edit"
    assert card["choices"] == ["once", "deny"]
    resolved = await env.permissions.resolve_schedule_card(
        card["approval_id"],
        choice="once",
        resolver="operator:local",
        confirm_proposal_digest=card["proposal_digest"],
        commit=env.service.commit,
    )
    assert resolved.status == "approved"
    return resolved.as_dict() if hasattr(resolved, "as_dict") else {}


async def test_operator_pause_resume_changes_enabled_state_without_reanchoring(env):
    before = await create_operator_job(env)
    assert before["enabled"] == 1
    original_next = before["next_run_at"]

    paused = await run_operator(env, "schedule_pause", {
        "name": "pause-me",
        "config_revision": before["config_revision"],
        "paused_reason": "operator hold",
    })
    assert paused.output["status"] == "updated", paused.error
    assert paused.output["applied"] is True
    after_pause = await row()
    assert after_pause["enabled"] == 0
    assert after_pause["status"] == "paused"
    assert after_pause["next_run_at"] == original_next
    assert after_pause["expression"] == before["expression"]
    assert after_pause["repeat_limit"] == before["repeat_limit"]
    paused_config = json.loads(after_pause["config"])
    assert paused_config["paused"] is True
    assert paused_config["paused_reason"] == "operator hold"

    resumed = await run_operator(env, "schedule_resume", {
        "name": "pause-me",
        "config_revision": after_pause["config_revision"],
    })
    assert resumed.output["status"] == "updated", resumed.error
    after_resume = await row()
    assert after_resume["enabled"] == 1
    assert after_resume["status"] == "active"
    assert after_resume["next_run_at"] == original_next
    assert after_resume["expression"] == before["expression"]
    assert json.loads(after_resume["config"])["paused"] is False


async def test_model_pause_and_resume_require_an_edit_card(env):
    before = await create_operator_job(env)

    paused = await run(
        env, "schedule_pause",
        {"name": "pause-me", "paused_reason": "awaiting consent"},
        turn=_chat_turn(),
    )
    assert paused.output["status"] == "approval_required", paused.error
    assert paused.output["card"]["kind"] == "schedule_edit"
    unchanged = await row()
    assert unchanged["enabled"] == 1
    assert unchanged["config_revision"] == before["config_revision"]
    await resolve_card(env, paused)
    paused_row = await row()
    assert paused_row["enabled"] == 0
    assert paused_row["status"] == "paused"

    resumed = await run(
        env, "schedule_resume", {"name": "pause-me"}, turn=_chat_turn()
    )
    assert resumed.output["status"] == "approval_required", resumed.error
    assert resumed.output["card"]["kind"] == "schedule_edit"
    assert (await row())["enabled"] == 0
    await resolve_card(env, resumed)
    assert (await row())["enabled"] == 1


async def test_create_grant_does_not_authorize_resume(env):
    created = await run(env, "schedule_create", {
        "name": "pause-me", "schedule": "every 2h", "prompt": "check queue",
    }, turn=_chat_turn())
    assert created.output["status"] == "approval_required", created.error
    card = created.output["card"]
    granted = await env.permissions.resolve_schedule_card(
        card["approval_id"],
        choice="session",
        resolver="operator:local",
        confirm_proposal_digest=card["proposal_digest"],
        commit=env.service.commit,
    )
    assert granted.grant_id

    paused = await run_operator(env, "schedule_pause", {
        "name": "pause-me", "paused_reason": "test hold",
    })
    assert paused.output["status"] == "updated", paused.error
    assert (await row())["enabled"] == 0

    resumed = await run(
        env, "schedule_resume", {"name": "pause-me"}, turn=_chat_turn()
    )
    assert resumed.output["status"] == "approval_required", resumed.error
    assert resumed.output["card"]["kind"] == "schedule_edit"
    assert (await row())["enabled"] == 0


async def test_pause_fields_are_bounded_and_stale_revisions_fail(env):
    await create_operator_job(env)
    too_long = await run(env, "schedule_pause", {
        "name": "pause-me", "paused_reason": "x" * 501,
    })
    assert too_long.success is False
    assert "at most" in too_long.error
    assert (await row())["enabled"] == 1

    stale = await run_operator(env, "schedule_pause", {
        "name": "pause-me", "config_revision": 99,
    })
    assert stale.success is False
    assert "revision" in stale.error.lower()
    assert (await row())["enabled"] == 1
