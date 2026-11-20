"""Automation tools execute through ToolExecutorImpl against the durable store.

Assertions are about real rows: a schedule is only reported created/deleted when
the SQLite row changed, and a manual trigger is only reported as triggered when
a task row exists with the read-back status.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tasks import DurableTaskEngine, Scheduler
from leo.tools import ToolExecutorImpl, ToolRegistry, automation_tools

TOOL_NAMES = [
    "schedule_create",
    "schedule_list",
    "schedule_delete",
    "schedule_run_now",
]


class ServicesExecutor(ToolExecutorImpl):
    """Inject the integration-owned services mapping into each tool context."""

    def __init__(self, *args: object, services: dict | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.injected_services = services or {}

    def _context(self, call, task_key, dry_run):  # type: ignore[no-untyped-def]
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.injected_services
        return ctx


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    engine = DurableTaskEngine(settings, events)
    scheduler = Scheduler(engine, settings, events)
    services = {"scheduler": scheduler, "tasks": engine}
    registry = ToolRegistry()
    assert automation_tools.register(registry) == TOOL_NAMES
    executor = ServicesExecutor(
        registry, permissions, events, settings, services=services
    )
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        engine=engine, scheduler=scheduler, services=services,
        registry=registry, executor=executor,
    )


async def run(env, name: str, arguments: dict | None = None):
    return await env.executor.execute(
        ToolCall(name=name, arguments=arguments or {})
    )


def task_rows(name: str):
    return db.fetchall(
        "SELECT * FROM tasks WHERE json_extract(metadata, '$.schedule')=?", (name,)
    )


# --------------------------------------------------------------------------- registration


async def test_registration_specs_and_schema_validation(env):
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert set(specs) == set(TOOL_NAMES)
    assert [spec.name for spec in env.registry.list_specs()] == TOOL_NAMES
    for name, spec in specs.items():
        assert spec.category == "automation"
        assert spec.permissions
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""  # identifiers are not paths
    assert specs["schedule_list"].risk.value == "low"
    for name in ("schedule_create", "schedule_delete", "schedule_run_now"):
        assert specs[name].risk.value == "medium"

    missing = await run(env, "schedule_create", {"name": "x", "expression": "* * * * *"})
    assert missing.success is False and missing.decision == "invalid_arguments"
    assert "objective" in missing.error

    extra = await run(env, "schedule_list", {"unknown": True})
    assert extra.success is False and extra.decision == "invalid_arguments"
    assert "unknown field" in extra.error

    wrong_type = await run(env, "schedule_run_now", {"name": "x", "run_at": 123})
    assert wrong_type.success is False and wrong_type.decision == "invalid_arguments"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='x'") is None


# --------------------------------------------------------------------------- create/list


async def test_create_persists_schedule_and_list_reads_it(env):
    created = await run(env, "schedule_create", {
        "name": "digest",
        "expression": "*/5 * * * *",
        "objective": "Summarise new items",
        "payload": {"owner": "ops"},
    })
    assert created.success, created.error
    row = await db.fetchone("SELECT * FROM schedules WHERE name='digest'")
    assert row is not None
    assert created.output["id"] == row["id"]
    assert row["kind"] == "cron" and row["expression"] == "*/5 * * * *"
    assert row["next_run_at"] and row["enabled"] == 1
    assert '"owner":"ops"' in row["payload"] and "Summarise new items" in row["payload"]
    assert created.output["next_run_at"] == row["next_run_at"]

    listed = await run(env, "schedule_list")
    assert listed.success, listed.error
    assert listed.output["count"] == 1
    assert listed.output["schedules"][0]["id"] == row["id"]
    assert listed.output["schedules"][0]["payload"]["owner"] == "ops"

    await env.scheduler.disable("digest")
    enabled_only = await run(env, "schedule_list", {"enabled_only": True})
    assert enabled_only.success and enabled_only.output["count"] == 0
    all_rows = await run(env, "schedule_list")
    assert all_rows.output["count"] == 1
    assert all_rows.output["schedules"][0]["enabled"] is False


async def test_create_accepts_interval_and_every_expressions(env):
    numeric = await run(env, "schedule_create", {
        "name": "beat", "expression": 60, "objective": "heartbeat",
    })
    assert numeric.success, numeric.error
    row = await db.fetchone("SELECT * FROM schedules WHERE name='beat'")
    assert row is not None and row["kind"] == "interval"
    assert row["expression"] == "60.0"

    every = await run(env, "schedule_create", {
        "name": "hourly", "expression": "@every 1h", "objective": "hourly job",
    })
    assert every.success, every.error
    row = await db.fetchone("SELECT * FROM schedules WHERE name='hourly'")
    assert row is not None and row["next_run_at"]


async def test_invalid_cron_interval_and_iso_are_rejected(env):
    bad_cron = await run(env, "schedule_create", {
        "name": "bad-cron", "expression": "not a cron", "objective": "x",
    })
    assert bad_cron.success is False and "cron" in bad_cron.error.lower()
    assert await db.fetchone("SELECT id FROM schedules WHERE name='bad-cron'") is None

    bad_every = await run(env, "schedule_create", {
        "name": "bad-every", "expression": "@every 0s", "objective": "x",
    })
    assert bad_every.success is False and "@every" in bad_every.error
    assert await db.fetchone("SELECT id FROM schedules WHERE name='bad-every'") is None

    assert (await run(env, "schedule_create", {
        "name": "bad-steps", "expression": "*/0 * * * *", "objective": "x",
    })).success is False

    created = await run(env, "schedule_create", {
        "name": "iso-check", "expression": "@every 1h", "objective": "x",
    })
    assert created.success, created.error
    bad_iso = await run(env, "schedule_run_now", {
        "name": "iso-check", "run_at": "yesterday",
    })
    assert bad_iso.success is False and "invalid ISO 8601 run_at" in bad_iso.error
    assert await task_rows("iso-check") == []


# --------------------------------------------------------------------------- delete


async def test_delete_removes_persisted_row_and_reports_truthfully(env):
    created = await run(env, "schedule_create", {
        "name": "temp", "expression": "*/5 * * * *", "objective": "temporary",
    })
    assert created.success, created.error
    assert await db.fetchone("SELECT id FROM schedules WHERE name='temp'") is not None

    deleted = await run(env, "schedule_delete", {"name": "temp"})
    assert deleted.success, deleted.error
    assert deleted.output == {
        "name": "temp", "schedule_id": created.output["id"], "deleted": True,
    }
    assert await db.fetchone("SELECT id FROM schedules WHERE name='temp'") is None

    again = await run(env, "schedule_delete", {"name": "temp"})
    assert again.success is False and "unknown schedule" in again.error


# --------------------------------------------------------------------------- run now


async def test_run_now_creates_persisted_task_and_reports_no_execution(env):
    created = await run(env, "schedule_create", {
        "name": "nightly", "expression": "@every 1h", "objective": "build report",
    })
    assert created.success, created.error
    before = await env.scheduler.get("nightly")

    result = await run(env, "schedule_run_now", {"name": "nightly"})
    assert result.success, result.error
    out = result.output
    assert out["triggered"] is True and out["task_count"] == 1
    assert out["executed"] is False and "no worker executed" in out["note"]
    # A manual occurrence is anchored to the request, not the future cursor.
    assert out["scheduled_for"] == out["run_at"]
    assert out["target_next_run_at"] == before["next_run_at"]

    rows = await task_rows("nightly")
    assert len(rows) == 1
    task = rows[0]
    assert task["status"] == "created"
    assert out["task_id"] == task["id"] and out["task_status"] == "created"
    assert "build report" in task["objective"]
    assert out["next_run_at"] != before["next_run_at"]
    assert out["last_run_at"] is not None
    assert (await env.engine.get(task["id"])).status.value == "created"

    history = await env.events.history("schedule.triggered")
    assert any(event.payload.get("name") == "nightly" for event in history)


async def test_run_now_honours_iso_run_at_and_does_not_run_disabled(env):
    created = await run(env, "schedule_create", {
        "name": "iso-run", "expression": "@every 1h", "objective": "iso job",
    })
    assert created.success, created.error
    row = await db.fetchone("SELECT next_run_at FROM schedules WHERE name='iso-run'")
    result = await run(env, "schedule_run_now", {
        "name": "iso-run", "run_at": row["next_run_at"],
    })
    assert result.success, result.error
    assert result.output["run_at"] == row["next_run_at"]
    assert result.output["requested_run_at"] == row["next_run_at"]
    assert result.output["triggered"] is True and result.output["executed"] is False
    assert len(await task_rows("iso-run")) == 1

    await env.scheduler.disable("iso-run")
    disabled = await run(env, "schedule_run_now", {"name": "iso-run"})
    assert disabled.success is False and "disabled" in disabled.error
    assert len(await task_rows("iso-run")) == 1  # no silent extra task


async def test_missing_services_never_fake_success(env):
    registry = ToolRegistry()
    assert automation_tools.register(registry) == TOOL_NAMES
    empty = ServicesExecutor(
        registry, env.permissions, env.events, env.settings, services={}
    )
    call = ToolCall(name="schedule_create", arguments={
        "name": "no-service", "expression": "*/5 * * * *", "objective": "x",
    })
    result = await empty.execute(call)
    assert result.success is False and "scheduler service unavailable" in result.error
    assert await db.fetchone("SELECT id FROM schedules WHERE name='no-service'") is None

    no_engine = ServicesExecutor(
        registry, env.permissions, env.events, env.settings,
        services={"scheduler": env.scheduler},
    )
    created = await run(env, "schedule_create", {
        "name": "no-engine", "expression": "@every 1h", "objective": "x",
    })
    assert created.success, created.error
    failed = await no_engine.execute(
        ToolCall(name="schedule_run_now", arguments={"name": "no-engine"})
    )
    assert failed.success is False and "tasks service unavailable" in failed.error
    assert await task_rows("no-engine") == []


async def test_registration_time_services_fallback_works(env):
    """The executor need not inject services when register() was given them."""
    registry = ToolRegistry()
    assert automation_tools.register(registry, services=env.services) == TOOL_NAMES
    executor = ToolExecutorImpl(registry, env.permissions, env.events, env.settings)
    result = await executor.execute(ToolCall(name="schedule_create", arguments={
        "name": "via-register", "expression": "@every 5m", "objective": "fallback",
    }))
    assert result.success, result.error
    assert await db.fetchone("SELECT id FROM schedules WHERE name='via-register'") is not None


async def test_run_now_triggers_only_named_schedule_at_requested_time(env):
    for name in ("manual", "bystander"):
        created = await run(env, "schedule_create", {
            "name": name, "expression": "@every 1h", "objective": f"{name} job",
        })
        assert created.success, created.error

    request_at = "2030-01-01T09:00:00+00:00"
    future_cursor = "2031-01-01T00:00:00+00:00"
    due_cursor = "2030-01-01T08:00:00+00:00"
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?", (future_cursor, "manual")
    )
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?", (due_cursor, "bystander")
    )

    result = await run(env, "schedule_run_now", {
        "name": "manual", "run_at": request_at,
    })
    assert result.success, result.error
    out = result.output
    assert out["triggered"] is True and out["task_count"] == 1
    assert out["run_at"] == out["scheduled_for"] == request_at
    assert out["target_next_run_at"] == future_cursor
    assert out["next_run_at"] == future_cursor  # never move the cursor backwards

    rows = await task_rows("manual")
    assert len(rows) == 1
    assert json.loads(rows[0]["metadata"])["scheduled_for"] == request_at
    assert await task_rows("bystander") == []
    bystander = await db.fetchone(
        "SELECT next_run_at FROM schedules WHERE name='bystander'"
    )
    assert bystander["next_run_at"] == due_cursor
    triggered = await env.events.history("schedule.triggered")
    assert [event.payload["name"] for event in triggered] == ["manual"]
