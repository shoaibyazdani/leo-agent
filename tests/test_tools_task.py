"""Executor-level tests for the durable task-control toolset."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, TaskStatus, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tasks.engine import DurableTaskEngine
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import task_tools

EXPECTED = ["task_create", "task_list", "task_status", "task_progress", "task_cancel", "task_result"]


@pytest.fixture
async def env(tmp_path):
    # The shared conftest autouse fixture isolates the DB too; re-point it here
    # so every executor call and every direct assertion uses the same store.
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    engine = DurableTaskEngine(settings, events)
    registry = ToolRegistry()
    task_tools.register(registry)  # no services: exercises the import fallback
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events,
        permissions=permissions, engine=engine, registry=registry, executor=executor,
    )


async def _run(env, name, arguments=None, **kwargs):
    call = ToolCall(name=name, arguments=arguments or {}, **kwargs)
    return await env.executor.execute(call)


async def _row_count():
    row = await db.fetchone("SELECT COUNT(*) AS n FROM tasks")
    return row["n"]


# ---------------------------------------------------------------- registry


async def test_register_typed_specs(env):
    assert task_tools.register(ToolRegistry()) == EXPECTED
    assert [spec.name for spec in env.registry.list_specs()] == EXPECTED

    for spec in env.registry.list_specs():
        assert spec.category == "tasks"
        assert spec.permissions and all(spec.permissions)
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""  # task ids are not filesystem resources

    risks = {spec.name: spec.risk for spec in env.registry.list_specs()}
    assert risks == {
        "task_create": RiskLevel.MEDIUM, "task_list": RiskLevel.LOW,
        "task_status": RiskLevel.LOW, "task_progress": RiskLevel.LOW,
        "task_cancel": RiskLevel.MEDIUM, "task_result": RiskLevel.LOW,
    }
    for name in ("task_status", "task_cancel", "task_result"):
        schema = env.registry.get(name).spec.input_schema
        assert schema["required"] == ["task_id"]
    assert env.registry.get("task_create").spec.input_schema["required"] == ["objective"]


# ---------------------------------------------------------------- create/list


async def test_create_and_list_persist_durable_rows_and_events(env):
    result = await _run(
        env, "task_create",
        {
            "objective": "ship task tools",
            "plan": ["implement", "test"],
            "owner_agent": "builder",
            "metadata": {"source": "test"},
            "idempotency_key": "create-key-1",
        },
    )
    assert result.success and result.error is None, result.error
    task_id = result.output["task_id"]
    assert result.output["status"] == "created"
    assert result.output["objective"] == "ship task tools"
    assert result.output["plan"] == ["implement", "test"]

    row = await db.fetchone("SELECT * FROM tasks WHERE id=?", (task_id,))
    assert row is not None
    assert row["objective"] == "ship task tools"
    assert row["status"] == "created"
    assert row["owner_agent"] == "builder"
    assert json.loads(row["plan"]) == ["implement", "test"]
    assert json.loads(row["metadata"]) == {"source": "test", "idempotency_key": "create-key-1"}

    events = await env.events.history("task.created")
    assert any(event.payload.get("task_id") == task_id for event in events)

    listed = await _run(env, "task_list", {"status": "created", "limit": 25})
    assert listed.success and listed.output["count"] >= 1
    assert task_id in [task["task_id"] for task in listed.output["tasks"]]

    none = await _run(env, "task_list", {"status": "completed"})
    assert none.success and none.output["count"] == 0

    # Idempotent replay returns the same durable row, not a second fabricated task.
    replay = await _run(
        env, "task_create",
        {"objective": "ship task tools", "idempotency_key": "create-key-1"},
    )
    assert replay.success and replay.output["task_id"] == task_id
    rows = await db.fetchall("SELECT id FROM tasks WHERE objective=?", ("ship task tools",))
    assert len(rows) == 1


async def test_task_list_objective_query_is_compact_newest_first_and_bounded(env):
    older = await env.engine.create("old tool execution audit")
    newer = await env.engine.create("NEW Tool Execution harness")
    unrelated = await env.engine.create("memory consolidation")
    # Distinct timestamps make the ordering assertion independent of clock speed.
    for task_id, stamp in (
        (older.id, "2026-01-01T00:00:00+00:00"),
        (newer.id, "2026-02-01T00:00:00+00:00"),
        (unrelated.id, "2026-03-01T00:00:00+00:00"),
    ):
        await db.execute("UPDATE tasks SET created_at=? WHERE id=?", (stamp, task_id))

    found = await _run(env, "task_list", {"query": "TOOL EXECUTION", "limit": 10})
    assert found.success, found.error
    assert [task["task_id"] for task in found.output["tasks"]] == [newer.id, older.id]
    assert found.output["query"] == "TOOL EXECUTION"
    assert found.output["newest_first"] is True
    assert found.output["matched"] == 2 and found.output["count"] == 2
    assert found.output["truncated"] is False
    # Compact rows: enough to identify and order the hit, not the whole record.
    assert set(found.output["tasks"][0]) == {
        "task_id", "objective", "status", "created_at", "updated_at",
    }
    assert "plan" not in found.output["tasks"][0]

    alias = await _run(env, "task_list", {"objective_contains": "tool execution"})
    assert alias.success
    assert alias.output["query"] == "tool execution"
    assert [t["task_id"] for t in alias.output["tasks"]] == [newer.id, older.id]

    oldest_first = await _run(env, "task_list", {"query": "tool execution", "newest_first": False})
    assert oldest_first.success
    assert [t["task_id"] for t in oldest_first.output["tasks"]] == [older.id, newer.id]

    limited = await _run(env, "task_list", {"query": "execution", "limit": 1})
    assert limited.success and limited.output["count"] == 1
    assert limited.output["tasks"][0]["task_id"] == newer.id
    assert limited.output["matched"] == 2 and limited.output["truncated"] is True

    filtered = await _run(env, "task_list", {"query": "execution", "status": "completed"})
    assert filtered.success and filtered.output["count"] == 0
    missing = await _run(env, "task_list", {"query": "no such objective"})
    assert missing.success and missing.output["tasks"] == []
    assert missing.output["matched"] == 0 and missing.output["query"] == "no such objective"


async def test_task_list_default_ordering_and_create_query_workflow(env):
    first = await env.engine.create("first task about tool execution")
    second = await env.engine.create("second task about tool execution")
    await db.execute(
        "UPDATE tasks SET created_at=? WHERE id=?", ("2026-01-01T00:00:00+00:00", first.id)
    )
    await db.execute(
        "UPDATE tasks SET created_at=? WHERE id=?", ("2026-04-01T00:00:00+00:00", second.id)
    )

    default = await _run(env, "task_list")
    assert default.success
    assert [t["task_id"] for t in default.output["tasks"]] == [second.id, first.id]
    assert default.output["newest_first"] is True and default.output["query"] is None

    created = await _run(env, "task_create", {"objective": "fresh tool execution triage"})
    assert created.success
    found = await _run(env, "task_list", {"query": "tool execution", "limit": 5})
    assert found.success
    assert found.output["tasks"][0]["task_id"] == created.output["task_id"]
    assert created.output["objective"] == "fresh tool execution triage"


async def test_task_list_query_schema_and_conflicts_are_actionable(env, monkeypatch):
    schema = env.registry.get("task_list").spec.input_schema
    for field in ("query", "objective_contains"):
        assert schema["properties"][field]["maxLength"] == task_tools.MAX_QUERY_CHARS
        assert schema["properties"][field]["minLength"] == 1
    assert schema["properties"]["newest_first"]["type"] == "boolean"

    for arguments in (
        {"query": 42}, {"query": ""}, {"objective_contains": None},
        {"newest_first": "yes"}, {"query": "x" * (task_tools.MAX_QUERY_CHARS + 1)},
    ):
        rejected = await _run(env, "task_list", arguments)
        assert not rejected.success and rejected.decision == "invalid_arguments", arguments

    conflict = await _run(env, "task_list", {"query": "alpha", "objective_contains": "beta"})
    assert not conflict.success and "must match" in (conflict.error or "")
    blank = await _run(env, "task_list", {"query": "   "})
    assert not blank.success and "non-whitespace" in (blank.error or "")

    # A query widens the engine read to a fixed ceiling, then truncates again.
    calls: list[int] = []

    class SpyEngine:
        def __init__(self, real):
            self.real = real

        def __getattr__(self, name):
            attribute = getattr(self.real, name)
            if not callable(attribute):
                return attribute

            async def wrapped(*args, **kwargs):
                if name == "list":
                    calls.append(kwargs.get("limit", 100))
                return await attribute(*args, **kwargs)

            return wrapped

    env.executor.services["tasks"] = SpyEngine(env.engine)
    scanned = await _run(env, "task_list", {"query": "anything", "limit": 3})
    assert scanned.success
    assert calls == [task_tools._SEARCH_SCAN_LIMIT]
    assert len(scanned.output["tasks"]) <= 3

    plain = await _run(env, "task_list", {"limit": 3})
    assert plain.success and calls[-1] == 3


# ---------------------------------------------------------------- status/result


async def test_status_and_result_report_only_persisted_truth(env):
    created = await env.engine.create("completed work", metadata={"k": "v"})
    await env.engine.transition(created.id, TaskStatus.READY)
    await env.engine.transition(created.id, TaskStatus.RUNNING)
    evidence = [{"kind": "test", "detail": "asserted"}]
    await env.engine.complete(created.id, {"value": 42}, evidence)

    status = await _run(env, "task_status", {"task_id": created.id})
    assert status.success
    assert status.output["status"] == "completed"
    assert status.output["terminal"] is True
    assert status.output["completed_at"] is not None
    assert status.output["evidence_count"] == 1

    result = await _run(env, "task_result", {"task_id": created.id})
    assert result.success
    assert result.output["status"] == "completed"
    assert result.output["completed"] is True
    assert result.output["result"] == {"value": 42}
    assert result.output["evidence"] == evidence

    pending = await env.engine.create("not finished")
    result2 = await _run(env, "task_result", {"task_id": pending.id})
    assert result2.success  # a lookup can succeed while the task is incomplete
    assert result2.output["completed"] is False
    assert result2.output["status"] == "created"
    assert result2.output["result"] is None
    assert result2.output["evidence"] == []

    for name in ("task_status", "task_result"):
        missing = await _run(env, name, {"task_id": "task-does-not-exist"})
        assert not missing.success
        assert "unknown task" in (missing.error or "")
        assert missing.output is None


# ---------------------------------------------------------------- cancel


async def test_cancel_reports_actual_state_and_rejects_invalid_targets(env):
    created = await env.engine.create("cancel me")
    cancelled = await _run(env, "task_cancel", {"task_id": created.id})
    assert cancelled.success, cancelled.error
    assert cancelled.output["status"] == "cancelled"
    assert cancelled.output["cancelled"] is True
    assert cancelled.output["cancellation_requested"] is True
    row = await db.fetchone("SELECT status,cancellation_requested FROM tasks WHERE id=?", (created.id,))
    assert row["status"] == "cancelled" and row["cancellation_requested"] == 1

    running = await env.engine.create("running work")
    await env.engine.transition(running.id, TaskStatus.READY)
    await env.engine.transition(running.id, TaskStatus.RUNNING)
    requested = await _run(env, "task_cancel", {"task_id": running.id})
    assert requested.success, requested.error
    # The tool must report exactly what the durable engine persisted. With no
    # live worker claim the engine settles the cancellation immediately; with a
    # claim it would remain running until the worker observes the flag.
    row = await db.fetchone("SELECT status,cancellation_requested FROM tasks WHERE id=?", (running.id,))
    assert requested.output["status"] == row["status"]
    assert requested.output["cancelled"] is (row["status"] == "cancelled")
    assert requested.output["cancellation_requested"] is True
    assert row["cancellation_requested"] == 1

    again = await _run(env, "task_cancel", {"task_id": created.id})
    assert not again.success and "cannot cancel terminal" in (again.error or "")
    missing = await _run(env, "task_cancel", {"task_id": "task-absent"})
    assert not missing.success and "unknown task" in (missing.error or "")


# ---------------------------------------------------------------- injection/failures


async def test_injected_tasks_service_is_used(env, monkeypatch):
    class SpyEngine:
        def __init__(self, real):
            self.real = real
            self.calls: list[str] = []

        def __getattr__(self, name):
            attribute = getattr(self.real, name)
            if not callable(attribute):
                return attribute

            async def wrapped(*args, **kwargs):
                self.calls.append(name)
                return await attribute(*args, **kwargs)

            return wrapped

    spy = SpyEngine(env.engine)
    original_context = env.executor._context

    def with_services(call, task_key, dry_run):
        ctx = original_context(call, task_key, dry_run)
        ctx["services"] = {"tasks": spy}
        return ctx

    monkeypatch.setattr(env.executor, "_context", with_services)
    result = await _run(env, "task_create", {"objective": "injected engine"})
    assert result.success, result.error
    assert "create" in spy.calls
    row = await db.fetchone("SELECT id FROM tasks WHERE id=?", (result.output["task_id"],))
    assert row is not None


async def test_schema_permission_and_service_failures(env, monkeypatch):
    before = await _row_count()

    invalid = await _run(env, "task_create", {"plan": ["no objective"]})
    assert not invalid.success and invalid.decision == "invalid_arguments"
    unknown_field = await _run(env, "task_status", {"task_id": "x", "bogus": True})
    assert not unknown_field.success and unknown_field.decision == "invalid_arguments"
    wrong_type = await _run(env, "task_list", {"limit": "ten"})
    assert not wrong_type.success and wrong_type.decision == "invalid_arguments"
    assert await _row_count() == before

    env.settings.security.allow_medium_risk = False
    denied = await _run(env, "task_create", {"objective": "must not be created"})
    assert not denied.success and denied.decision == "denied"
    assert await _row_count() == before

    out_of_scope = await env.executor.execute(
        ToolCall(name="task_list"), scope=frozenset({"utility"})
    )
    assert not out_of_scope.success and out_of_scope.decision == "out_of_scope"

    monkeypatch.setattr(task_tools.db, "database_path_configured", lambda: False)
    unavailable = await _run(env, "task_list")
    assert not unavailable.success
    assert "task service unavailable" in (unavailable.error or "")


async def test_task_create_bounds_plan_and_dependencies_items(env):
    schema = env.registry.get("task_create").spec.input_schema
    for field in ("plan", "dependencies"):
        assert schema["properties"][field]["maxItems"] == task_tools.MAX_PLAN_ITEMS
        assert schema["properties"][field]["items"]["maxLength"] == task_tools.MAX_PLAN_ITEM_CHARS

    before = await _row_count()
    too_many = await _run(env, "task_create", {
        "objective": "bounded plan", "plan": ["step"] * (task_tools.MAX_PLAN_ITEMS + 1),
    })
    assert too_many.success is False and too_many.decision == "invalid_arguments"
    assert "at most" in (too_many.error or "")

    too_long = await _run(env, "task_create", {
        "objective": "bounded plan",
        "dependencies": ["d" * (task_tools.MAX_PLAN_ITEM_CHARS + 1)],
    })
    assert too_long.success is False and too_long.decision == "invalid_arguments"
    assert "at most" in (too_long.error or "")
    assert await _row_count() == before


async def test_owned_toolsets_register_with_no_duplicate_names():
    from leo.tools import (
        approval_tools, automation_tools, config_tools, context_tools,
        run_tools, skill_tools, task_tools, webhook_tools,
    )

    registry = ToolRegistry()
    registrars = (
        approval_tools.register, webhook_tools.register, automation_tools.register,
        run_tools.register, skill_tools.register, context_tools.register,
        config_tools.register, task_tools.register,
    )
    seen: set[str] = set()
    for registrar in registrars:
        names = registrar(registry)
        assert len(names) == len(set(names))
        assert not seen.intersection(names), f"duplicate tool names: {seen.intersection(names)}"
        seen.update(names)
    assert len(registry.list_specs()) == len(seen)
    assert "approval_resolve" not in seen
    assert "webhook_sign" not in seen
