"""Intentional compatibility changes and preserved legacy behavior.

Every changed assertion in the scheduling port is classified here as an
intentional contract update (a new authority, roster or projection), never as a
waived regression. Behavior that the port promises to keep — command approval
wrappers, legacy schedule rows, old delete/run semantics and the legacy delivery
projection — has an explicit passing assertion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, TurnContext, utcnow
from leo.events import EventBus
from leo.schedules.clock import FrozenClock
from leo.schedules.service import ScheduleService
from leo.schedules.store import ScheduleStore
from leo.security.permissions import PermissionManagerImpl
from leo.tasks import DurableTaskEngine, Scheduler
from leo.tasks.engine import _delivery_targets
from leo.tools import automation_tools
from leo.tools.registry import ToolRegistry

NOW = datetime(2026, 11, 26, 7, 0, tzinfo=timezone.utc)


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
    service = ScheduleService(
        store=ScheduleStore(clock=clock), scheduler=scheduler, engine=engine,
        permissions=permissions, events=events, settings=settings, clock=clock,
        allowed_workdir_roots=[tmp_path],
    )
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        engine=engine, scheduler=scheduler, service=service,
    )


def _operator() -> TurnContext:
    return TurnContext(
        principal="operator:cli", channel="cli", surface="cli", source="operator"
    )


def test_tool_roster_change_is_an_intentional_contract_update():
    """Contract update: four legacy tools → seven final-head → nine with pause/resume.

    This is a deliberate roster replacement, not a regression waiver: aliases
    are not registered as separate callable tools and the action surface lives
    in the tool schemas instead.
    """
    assert automation_tools.register(ToolRegistry()) == [
        "schedule_create", "schedule_list", "schedule_get", "schedule_update",
        "schedule_pause", "schedule_resume", "schedule_delete",
        "schedule_run_now", "schedule_cancel",
    ]
    names = {spec.name for spec in ToolRegistry().list_specs()}
    assert "schedule_run" not in names and "cronjob_manage" not in names


async def test_legacy_command_approval_still_resolves_without_schedule_choices(env):
    """Preserved: command-kind cards keep the old approved/confirm_action path."""
    decision = await env.permissions.authorize(
        "legacy:actor", "legacy.action", None, RiskLevel.HIGH
    )
    assert decision.approval_id
    resolved = await env.permissions.resolve_approval(
        decision.approval_id, True, "operator:cli"
    )
    assert resolved.allowed is True
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (decision.approval_id,)
    )
    assert row["status"] == "approved"


async def test_legacy_schedule_row_is_readable_and_manual_run_keeps_repeat(env):
    """Preserved: old rows keep their cursor; manual runs never consume repeat."""
    await db.execute(
        "INSERT INTO schedules(id,name,kind,expression,payload,enabled,next_run_at,"
        "created_at,repeat_limit,claimed_count,completed_count,status) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "legacy-1", "legacy-job", "interval", "*/5 * * * *",
            '{"objective":"tick","delivery":"local"}', 1,
            NOW.isoformat(), utcnow(), None, 0, 0, "active",
        ),
    )
    listed = await env.scheduler.list_schedules()
    assert [item["name"] for item in listed] == ["legacy-job"]
    assert listed[0]["payload"]["objective"] == "tick"

    result = await env.service.run_now(
        schedule_id="legacy-1", actor="operator:cli", turn=_operator()
    )
    assert result["manual_run"] is True and result["triggered"] is True
    row = await db.fetchone(
        "SELECT claimed_count,completed_count FROM schedules WHERE id='legacy-1'"
    )
    assert (row["claimed_count"], row["completed_count"]) == (0, 0)
    task = await db.fetchone(
        "SELECT metadata FROM tasks WHERE id=?", (result["task_id"],)
    )
    assert '"manual_run":true' in task["metadata"]


async def test_legacy_remove_still_deletes_and_tombstones(env):
    """Contract update: removal remains a delete plus an inspectable tombstone."""
    created = await env.service.create(
        {"name": "removable", "schedule": "every day at 9am", "prompt": "tick",
         "deliver": "local"},
        actor="operator:cli", turn=_operator(),
    )
    schedule_id = created["schedule_id"]
    result = await env.service.remove(
        schedule_id=schedule_id, name="removable", actor="operator:cli", turn=_operator()
    )
    assert result["status"] == "removed" and result["name"] == "removable"
    assert await db.fetchone("SELECT id FROM schedules WHERE id=?", (schedule_id,)) is None
    tombstone = await db.fetchone(
        "SELECT reason FROM schedule_tombstones WHERE schedule_id=?", (schedule_id,)
    )
    assert tombstone["reason"] == "removed"


def test_legacy_delivery_projection_is_preserved():
    """Preserved: legacy `{channel: target}` callers keep their receipt shape."""
    legacy = SimpleNamespace(metadata={"notify_targets": {"ops": "room-1"}})
    assert _delivery_targets(legacy) == {"ops": "room-1"}
    modern = SimpleNamespace(metadata={"schedule_payload": {
        "delivery": [
            {"kind": "external", "platform": "telegram", "chat_id": "42"},
            {"kind": "local", "platform": "local"},
        ],
    }})
    assert _delivery_targets(modern) == {"telegram": "42", "local": "local"}
