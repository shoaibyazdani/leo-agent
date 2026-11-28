"""Automation tools exercise the one ScheduleService authority end to end.

The roster is the final-head model surface (create/list/get/update/pause/
resume/delete/run_now/cancel), schemas are closed, and a create/edit either
presents a real card on an answerable surface or fails closed with no write.
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

# Intentional roster update: explicit pause/resume tools join the final-head
# surface; both use ScheduleService.update and the same edit-consent path.
TOOL_NAMES = [
    "schedule_create",
    "schedule_list",
    "schedule_get",
    "schedule_update",
    "schedule_pause",
    "schedule_resume",
    "schedule_delete",
    "schedule_run_now",
    "schedule_cancel",
]
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
        principal="operator:cli",
        channel="cli",
        surface="cli",
        source="operator",
    )


def _chat_turn(surface: str = "tui") -> TurnContext:
    return TurnContext(
        principal="chat:web",
        channel=surface,
        conversation_id="conv-a",
        surface=surface,
        origin=surface,
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
    scripts_root = tmp_path / "scripts"
    scripts_root.mkdir()
    service = ScheduleService(
        store=store,
        scheduler=scheduler,
        engine=engine,
        permissions=permissions,
        events=events,
        settings=settings,
        clock=clock,
        scripts_root=scripts_root,
        allowed_workdir_roots=[tmp_path],
    )
    services = {
        "scheduler": scheduler,
        "tasks": engine,
        "schedule_service": service,
        "events": events,
        "permissions": permissions,
        "settings": settings,
    }
    registry = ToolRegistry()
    assert automation_tools.register(registry) == TOOL_NAMES
    executor = ServicesExecutor(
        registry, permissions, events, settings, services=services
    )
    return SimpleNamespace(
        tmp_path=tmp_path,
        settings=settings,
        events=events,
        permissions=permissions,
        engine=engine,
        scheduler=scheduler,
        store=store,
        service=service,
        services=services,
        registry=registry,
        executor=executor,
        scripts_root=scripts_root,
    )


async def run(env, name: str, arguments: dict | None = None, *, turn=None, actor="chat:web"):
    call = ToolCall(name=name, arguments=arguments or {}, actor=actor)
    if turn is None:
        return await env.executor.execute(call)
    with turn_context(turn):
        return await env.executor.execute(call)


async def run_operator(env, name: str, arguments: dict | None = None):
    return await run(env, name, arguments, turn=_operator_turn(), actor="operator:cli")


async def task_rows(name: str):
    return await db.fetchall(
        "SELECT * FROM tasks WHERE json_extract(metadata, '$.schedule')=?", (name,)
    )


# --------------------------------------------------------------------------- registration


async def test_registration_specs_and_closed_schema(env):
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert list(specs) == TOOL_NAMES
    for name, spec in specs.items():
        assert spec.category == "automation"
        assert spec.permissions
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""
        # Schedule consent is the authority, not a generic HIGH/MEDIUM card:
        # the tool itself must never insert an invisible command approval.
        assert spec.risk.value == "low"
    create_schema = specs["schedule_create"].input_schema
    assert create_schema["properties"]["schedule"]["type"] == "string"
    # repeat is an integer *or* a numeric string, matching the runtime parser.
    assert create_schema["properties"]["repeat"]["type"] == ["integer", "string"]
    for forbidden in ("provider", "model", "base_url", "grace_minutes", "reasoning_effort", "interpreter"):
        assert forbidden not in create_schema["properties"]

    missing = await run(env, "schedule_create", {"prompt": "x"})
    assert missing.success is False and missing.decision == "invalid_arguments"
    assert "schedule" in missing.error

    extra = await run(env, "schedule_list", {"unknown": True})
    assert extra.success is False and extra.decision == "invalid_arguments"
    assert "unknown field" in extra.error

    operator_only = await run(env, "schedule_create", {
        "schedule": "1h", "prompt": "x", "model": "gpt-x",
    })
    assert operator_only.success is False and operator_only.decision == "invalid_arguments"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='x'") is None


# --------------------------------------------------------------------------- operator direct


async def test_operator_direct_create_get_list_and_audit(env):
    created = await run_operator(env, "schedule_create", {
        "name": "digest",
        "schedule": "*/5 * * * *",
        "prompt": "Summarise new items",
    })
    assert created.success, created.error
    assert created.output["status"] == "created"
    row = await db.fetchone("SELECT * FROM schedules WHERE name='digest'")
    assert row is not None
    assert row["kind"] == "cron" and row["expression"] == "*/5 * * * *"
    assert row["config_revision"] == 1 and row["consent_digest"]
    assert json.loads(row["payload"])["objective"] == "Summarise new items"
    assert await db.fetchall("SELECT id FROM approvals") == []

    got = await run_operator(env, "schedule_get", {"name": "digest"})
    assert got.success and got.output["schedule"]["id"] == row["id"]
    listed = await run_operator(env, "schedule_list")
    assert listed.success and listed.output["count"] == 1
    assert listed.output["schedules"][0]["id"] == row["id"]

    audit = await db.fetchone(
        "SELECT actor, action, decision FROM audit_log WHERE action='schedule.created'"
    )
    assert dict(audit) == {
        "actor": "operator:cli", "action": "schedule.created", "decision": "allowed",
    }


async def test_operator_update_noop_cas_and_repeat_merge(env):
    created = await run_operator(env, "schedule_create", {
        "name": "digest", "schedule": "1h", "prompt": "first", "repeat": 5,
    })
    assert created.success, created.error
    row = await db.fetchone("SELECT * FROM schedules WHERE name='digest'")
    original_next = row["next_run_at"]
    await db.execute(
        "UPDATE schedules SET claimed_count=2, completed_count=1 WHERE id=?", (row["id"],)
    )

    noop = await run_operator(env, "schedule_update", {
        "name": "digest", "prompt": "first",
    })
    assert noop.success, noop.error
    assert noop.output["status"] == "no_change"
    unchanged = await db.fetchone("SELECT * FROM schedules WHERE id=?", (row["id"],))
    assert unchanged["config_revision"] == 1 and unchanged["next_run_at"] == original_next

    stale = await run_operator(env, "schedule_update", {
        "name": "digest", "prompt": "second", "config_revision": 99,
    })
    assert stale.success is False and "stale_update" in stale.error

    updated = await run_operator(env, "schedule_update", {
        "name": "digest", "prompt": "second", "repeat": 2, "config_revision": 1,
    })
    assert updated.success, updated.error
    assert updated.output["status"] == "updated"
    row2 = await db.fetchone("SELECT * FROM schedules WHERE id=?", (row["id"],))
    assert row2["config_revision"] == 2
    assert row2["repeat_limit"] == 2
    assert row2["claimed_count"] == 2 and row2["completed_count"] == 1
    assert json.loads(row2["payload"])["objective"] == "second"


async def test_operator_delete_records_a_tombstone(env):
    created = await run_operator(env, "schedule_create", {
        "name": "temp", "schedule": "1h", "prompt": "temporary",
    })
    assert created.success, created.error
    deleted = await run_operator(env, "schedule_delete", {"name": "temp"})
    assert deleted.success, deleted.error
    assert deleted.output["status"] == "removed"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='temp'") is None
    tomb = await db.fetchone(
        "SELECT schedule_id, name, reason FROM schedule_tombstones WHERE name='temp'"
    )
    assert tomb is not None and tomb["reason"] == "removed"
    again = await run_operator(env, "schedule_delete", {"name": "temp"})
    assert again.success is False and "unknown_schedule" in again.error


# --------------------------------------------------------------------------- model consent


async def test_model_create_on_answerable_surface_presents_one_card(env):
    created = await run(env, "schedule_create", {
        "name": "water", "schedule": "every day at 8am", "prompt": "Drink water.",
    }, turn=_chat_turn("tui"))
    assert created.success, created.error
    out = created.output
    assert out["status"] == "approval_required"
    assert out["applied"] is False and out["scheduled"] is False
    assert out["approval_id"] and out["proposal_id"] and out["card"]
    assert "Drink water." in out["card"]["body"]
    assert out["card"]["choices"] == ["once", "session", "always", "deny"]
    assert await db.fetchone("SELECT id FROM schedules WHERE name='water'") is None
    approval = await db.fetchone(
        "SELECT status, kind, shown_at, presented_surface, digest "
        "FROM approvals WHERE id=?", (out["approval_id"],)
    )
    assert approval["status"] == "pending"
    assert approval["kind"] == "schedule_create"
    assert approval["shown_at"] and approval["presented_surface"] == "tui"
    proposal = await db.fetchone(
        "SELECT status, active_name, schedule_id FROM schedule_proposals WHERE id=?",
        (out["proposal_id"],),
    )
    assert proposal["status"] == "pending" and proposal["active_name"] == "water"


async def test_model_create_on_incapable_surface_saves_at_once(env):
    """Hermes (Teknium PR #158289): where no card can be shown, the job
    saves at once, audited."""
    created = await run(env, "schedule_create", {
        "name": "cli-job", "schedule": "1h", "prompt": "Check.",
    }, turn=_chat_turn("cli"))
    assert created.success, created.error
    out = created.output
    assert out["status"] == "created"
    assert out["approval_id"] is None and out["card"] is None
    assert await db.fetchone("SELECT id FROM schedules WHERE name='cli-job'") is not None
    assert await db.fetchall("SELECT id FROM approvals") == []
    proposal = await db.fetchone(
        "SELECT status, active_name FROM schedule_proposals WHERE id=?",
        (out["proposal_id"],),
    )
    assert proposal["status"] == "committed" and proposal["active_name"] == "cli-job"


async def test_model_delete_requires_an_edit_card(env):
    created = await run_operator(env, "schedule_create", {
        "name": "remove-me", "schedule": "1h", "prompt": "x",
    })
    assert created.success, created.error
    removed = await run(env, "schedule_delete", {"name": "remove-me"}, turn=_chat_turn("web"))
    assert removed.success, removed.error
    assert removed.output["status"] == "approval_required"
    assert removed.output["card"]["kind"] == "schedule_edit"
    assert "remove this schedule" in removed.output["card"]["body"]
    assert await db.fetchone("SELECT id FROM schedules WHERE name='remove-me'") is not None


# --------------------------------------------------------------------------- fields


async def test_script_monitor_and_no_agent_validation(env):
    bad_no_agent = await run_operator(env, "schedule_create", {
        "name": "scriptless", "schedule": "1h", "prompt": "x", "no_agent": True,
    })
    assert bad_no_agent.success is False and "no_agent" in bad_no_agent.error

    missing_script = await run_operator(env, "schedule_create", {
        "name": "missing", "schedule": "1h", "script": "missing.py", "no_agent": True,
    })
    assert missing_script.success is False and "script file does not exist" in missing_script.error

    script = env.scripts_root / "check.py"
    script.write_text("print('ok')", encoding="utf-8")
    created = await run_operator(env, "schedule_create", {
        "name": "scripted",
        "schedule": "1h",
        "script": "check.py",
        "no_agent": True,
    })
    assert created.success, created.error
    row = await db.fetchone("SELECT config FROM schedules WHERE name='scripted'")
    stored = json.loads(row["config"])
    assert stored["execution"]["script"] == "check.py"
    assert stored["execution"]["no_agent"] is True

    bad_monitor = await run_operator(env, "schedule_create", {
        "name": "monitor", "schedule": "1h", "prompt": "x",
        "monitor": "file:///etc/passwd",
    })
    assert bad_monitor.success is False


# --------------------------------------------------------------------------- run / cancel


async def test_run_now_is_a_background_handle_with_no_cadence_consumption(env):
    created = await run_operator(env, "schedule_create", {
        "name": "nightly", "schedule": "1h", "prompt": "build report",
    })
    assert created.success, created.error
    before = await db.fetchone(
        "SELECT next_run_at, last_run_at, claimed_count FROM schedules WHERE name='nightly'"
    )

    result = await run_operator(env, "schedule_run_now", {"name": "nightly"})
    assert result.success, result.error
    out = result.output
    assert out["status"] == "running"
    assert out["manual_run"] is True and out["triggered"] is True
    assert out["executed"] is False and out["task_status"] == "created"
    rows = await task_rows("nightly")
    assert len(rows) == 1
    metadata = json.loads(rows[0]["metadata"])
    assert metadata["manual_run"] is True and metadata["schedule_id"]
    after = await db.fetchone(
        "SELECT next_run_at, last_run_at, claimed_count FROM schedules WHERE name='nightly'"
    )
    assert dict(after) == dict(before), "a manual run never moves cadence or counters"

    cancelled = await run_operator(env, "schedule_cancel", {"name": "nightly"})
    assert cancelled.success, cancelled.error
    assert cancelled.output["status"] == "cancelled"
    assert cancelled.output["cancelled"] is True
    assert cancelled.output["schedule_deleted"] is False
    task = await db.fetchone("SELECT status FROM tasks WHERE id=?", (rows[0]["id"],))
    assert task["status"] == "cancelled"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='nightly'") is not None

    noop = await run_operator(env, "schedule_cancel", {"name": "nightly"})
    assert noop.success, noop.error
    assert noop.output["status"] == "no_live_run"


async def test_run_and_cancel_copy_discloses_the_service_gate_and_it_holds(env):
    """R10: routine copy must name the real operator/approved-card gate."""
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    for name in ("schedule_run_now", "schedule_cancel"):
        description = specs[name].description.lower()
        # The ordinary operation stays plain; the gate is attributed to the
        # service instead of promising the model elevated rights.
        assert "schedule" in description and "task" in description
        assert "operator turn" in description and "approved card" in description
        assert "the schedule service enforces this gate" in description
        for misleading in ("admin", "privileg", "bypass"):
            assert misleading not in description, (name, misleading)

    await run_operator(env, "schedule_create", {
        "name": "gated", "schedule": "1h", "prompt": "build report",
    })
    fired = await run(env, "schedule_run_now", {"name": "gated"})
    assert fired.success is False
    assert await task_rows("gated") == []

    stopped = await run(env, "schedule_cancel", {"name": "gated"})
    assert stopped.success is False


async def test_missing_services_never_fake_success(env):
    registry = ToolRegistry()
    assert automation_tools.register(registry) == TOOL_NAMES
    empty = ServicesExecutor(
        registry, env.permissions, env.events, env.settings, services={}
    )
    result = await empty.execute(ToolCall(name="schedule_create", arguments={
        "name": "no-service", "schedule": "1h", "prompt": "x",
    }, actor="operator:cli"))
    assert result.success is False and "schedule service unavailable" in result.error
    assert await db.fetchone("SELECT id FROM schedules WHERE name='no-service'") is None

    # Registration-time services still work without executor injection.
    registry2 = ToolRegistry()
    automation_tools.register(registry2, services=env.services)
    executor = ToolExecutorImpl(registry2, env.permissions, env.events, env.settings)
    fallback = await executor.execute(ToolCall(
        name="schedule_create",
        arguments={"name": "via-register", "schedule": "5m", "prompt": "fallback"},
        actor="operator:cli",
    ))
    assert fallback.success, fallback.error
    assert await db.fetchone("SELECT id FROM schedules WHERE name='via-register'") is not None
