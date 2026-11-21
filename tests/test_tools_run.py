"""Executor-level tests for the durable-run toolset.

All tools are executed through ``ToolExecutorImpl`` against the shared SQLite
store. Runs, tasks and events are seeded as real durable rows; cancellation is
asserted against the persisted task row and engine events, never against a
claim made by the tool.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, TaskStatus, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tasks.engine import DurableTaskEngine
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import run_tools

EXPECTED = ["run_list", "run_status", "run_log", "run_cancel"]


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    engine = DurableTaskEngine(settings, events)
    registry = ToolRegistry()
    run_tools.register(registry)  # no services: exercise the real DB-backed fallback
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions, engine=engine,
        registry=registry, executor=executor,
    )


async def _run(env, name, arguments=None, **kwargs):
    return await env.executor.execute(ToolCall(name=name, arguments=arguments or {}, **kwargs))


async def _seed_run(
    run_id: str,
    *,
    agent_id: str = "builder",
    task_id: str | None = None,
    status: str = "running",
    started_at: str = "2026-11-19T00:00:00+00:00",
    finished_at: str | None = None,
    result=None,
    evaluations=None,
    metadata=None,
) -> None:
    await db.execute(
        "INSERT INTO agent_runs(id,agent_id,task_id,status,started_at,finished_at,result,evaluations,metadata) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (
            run_id, agent_id, task_id, status, started_at, finished_at,
            json.dumps(result) if result is not None else None,
            json.dumps(evaluations or []),
            json.dumps(metadata or {}),
        ),
    )


async def _task(task_id: str) -> Any:
    return await db.fetchone(
        "SELECT id,status,cancellation_requested,history,errors FROM tasks WHERE id=?",
        (task_id,),
    )


# ---------------------------------------------------------------- registry


async def test_register_typed_specs(env):
    assert run_tools.register(ToolRegistry()) == EXPECTED
    assert [spec.name for spec in env.registry.list_specs()] == EXPECTED

    for spec in env.registry.list_specs():
        assert spec.category == "runs"
        assert spec.permissions and all(spec.permissions)
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""  # run/task ids are not filesystem resources

    risks = {spec.name: spec.risk for spec in env.registry.list_specs()}
    assert risks == {
        "run_list": RiskLevel.LOW, "run_status": RiskLevel.LOW,
        "run_log": RiskLevel.LOW, "run_cancel": RiskLevel.MEDIUM,
    }
    for name in ("run_status", "run_log", "run_cancel"):
        assert env.registry.get(name).spec.input_schema["required"] == ["run_id"]
    assert "include_events" in env.registry.get("run_log").spec.input_schema["properties"]
    assert "runs:write" in env.registry.get("run_cancel").spec.permissions


# ---------------------------------------------------------------- run_list


async def test_run_list_reads_real_rows_filters_and_bounds_pages(env):
    task = await env.engine.create("linked work")
    await _seed_run("run-a", task_id=task.id)
    await _seed_run("run-b", agent_id="reviewer", status="completed", started_at="2026-11-19T01:00:00+00:00")
    await _seed_run("run-c", status="failed", started_at="2026-11-19T02:00:00+00:00")
    await _seed_run("run-d", status="completed", started_at="2026-11-19T03:00:00+00:00")

    result = await _run(env, "run_list")
    assert result.success and result.error is None, result.error
    assert result.output["total"] == 4 and result.output["count"] == 4
    ids = [item["run_id"] for item in result.output["runs"]]
    assert ids == ["run-d", "run-c", "run-b", "run-a"]  # newest first
    linked = next(item for item in result.output["runs"] if item["run_id"] == "run-a")
    assert linked["task_id"] == task.id
    assert linked["task_status"] == TaskStatus.CREATED.value
    assert linked["status"] == "running" and linked["terminal"] is False
    assert linked["cancellation_requested"] is False

    filtered = await _run(env, "run_list", {"agent_id": "reviewer", "status": "completed"})
    assert filtered.success and [r["run_id"] for r in filtered.output["runs"]] == ["run-b"]
    assert filtered.output["total"] == 1

    page = await _run(env, "run_list", {"limit": 2, "offset": 1})
    assert page.success and page.output["count"] == 2
    assert page.output["next_offset"] == 3
    assert [r["run_id"] for r in page.output["runs"]] == ["run-c", "run-b"]

    last = await _run(env, "run_list", {"limit": 2, "offset": 2})
    assert last.output["next_offset"] is None

    # Validation and defence in depth for direct handler calls.
    invalid = await _run(env, "run_list", {"limit": 1000})
    assert not invalid.success and invalid.decision == "invalid_arguments"
    direct = await run_tools.run_list({"limit": 100000}, {})
    assert direct["limit"] == run_tools.MAX_RUN_PAGE
    assert len(direct["runs"]) <= run_tools.MAX_RUN_PAGE


# ---------------------------------------------------------------- run_status


async def test_run_status_reports_persisted_run_task_and_bounded_evaluations(env):
    task = await env.engine.create("real work")
    await env.engine.transition(task.id, TaskStatus.READY)
    await env.engine.transition(task.id, TaskStatus.RUNNING)
    evaluations = [{"kind": f"eval-{i}", "score": i} for i in range(20)]
    await _seed_run(
        "run-live", task_id=task.id, result={"answer": "api_key=sk-abcdefghijklmnop"},
        evaluations=evaluations, metadata={"depth": 1},
    )

    result = await _run(env, "run_status", {"run_id": "run-live"})
    assert result.success and result.error is None, result.error
    output = result.output
    assert output["run_id"] == "run-live" and output["status"] == "running"
    assert output["terminal"] is False and output["agent_id"] == "builder"
    assert output["task"]["task_id"] == task.id
    assert output["task"]["status"] == TaskStatus.RUNNING.value
    assert output["task"]["terminal"] is False
    assert output["task"]["current_step"] == 0 and output["task"]["version"] >= 1
    assert output["evaluation_count"] == 20
    assert len(output["evaluations"]) == run_tools.MAX_EVALUATIONS
    assert "sk-abcdefghijklmnop" not in json.dumps(output)
    assert output["result"]["answer"] == "[REDACTED]" or "[REDACTED]" in output["result"]["answer"]

    # A finished run reports terminal state from its own persisted row.
    await _seed_run("run-done", status="completed", finished_at="2026-11-19T04:00:00+00:00")
    done = await _run(env, "run_status", {"run_id": "run-done"})
    assert done.success and done.output["terminal"] is True
    assert done.output["task"] is None and done.output["task_status"] is None

    missing = await _run(env, "run_status", {"run_id": "run-absent"})
    assert not missing.success and "unknown run" in (missing.error or "")


# ---------------------------------------------------------------- run_log


async def test_run_log_is_durable_bounded_and_secret_redacted(env):
    task = await env.engine.create("logged work")
    await env.engine.transition(task.id, TaskStatus.READY)
    await env.engine.transition(task.id, TaskStatus.RUNNING)
    await env.engine.cancel(task.id)
    await _seed_run("run-log", task_id=task.id)
    await db.execute(
        "UPDATE tasks SET errors=? WHERE id=?",
        (json.dumps(["password=hunter2secret"]), task.id),
    )

    # Durable events: one with a first-class correlation id, one whose payload
    # also identifies the run, and one belonging to an unrelated run.
    await env.events.emit(
        "task.log", run_id="run-log", task_id=task.id,
        correlation_id="run-log", line="api_key=sk-abcdefghijklmnop",
    )
    unrelated = json.dumps({"run_id": "run-other", "text": "not related"})
    await db.execute(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) VALUES(?,?,?,?,?,?)",
        ("event-unrelated", "run.output", "test", unrelated, None, "2026-11-19T01:00:00+00:00"),
    )
    await db.execute(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) VALUES(?,?,?,?,?,?)",
        (
            "event-secret", "run.output", "test",
            json.dumps({"run_id": "run-log", "text": "Authorization: Bearer abcdef1234567890"}),
            "run-log", "2026-11-19T02:00:00+00:00",
        ),
    )

    result = await _run(env, "run_log", {"run_id": "run-log"})
    assert result.success and result.error is None, result.error
    output = result.output
    events = [entry for entry in output["entries"] if entry["source"] == "event"]
    history = [entry for entry in output["entries"] if entry["source"] == "task_history"]
    assert {"created", "transition", "cancellation_requested"} <= {
        entry["event"] for entry in history
    }
    assert {entry["event"] for entry in events} >= {"task.log", "run.output"}
    assert output["total"] == len(output["entries"])
    assert output["next_offset"] is None and output["redacted"] is True
    assert [entry["seq"] for entry in output["entries"]] == list(range(output["total"]))
    dumped = json.dumps(output)
    for secret in (
        "sk-abcdefghijklmnop", "Bearer abcdef1234567890", "hunter2secret",
        "not related",
    ):
        assert secret not in dumped
    assert dumped.count("[REDACTED]") >= 3

    bounded = await _run(env, "run_log", {"run_id": "run-log", "limit": 3})
    assert bounded.success and len(bounded.output["entries"]) == 3
    assert bounded.output["total"] > 3 and bounded.output["next_offset"] == 3
    later = await _run(env, "run_log", {"run_id": "run-log", "limit": 3, "offset": 3})
    assert [entry["seq"] for entry in later.output["entries"]][0] == 3

    no_events = await _run(env, "run_log", {"run_id": "run-log", "include_events": False})
    assert no_events.success
    assert all(entry["source"] != "event" for entry in no_events.output["entries"])
    assert no_events.output["total"] < output["total"]

    direct = await run_tools.run_log({"run_id": "run-log", "limit": 100000}, {})
    assert len(direct["entries"]) <= run_tools.MAX_LOG_PAGE
    assert direct["limit"] == run_tools.MAX_LOG_PAGE

    missing = await _run(env, "run_log", {"run_id": "run-absent"})
    assert not missing.success and "unknown run" in (missing.error or "")


# ---------------------------------------------------------------- run_cancel


async def test_run_cancel_uses_engine_and_reports_only_persisted_state(env):
    # A non-running task is moved to cancelled by the real engine.
    task = await env.engine.create("cancel this run")
    await _seed_run("run-creatable", task_id=task.id)
    result = await _run(env, "run_cancel", {"run_id": "run-creatable"})
    assert result.success and result.error is None, result.error
    out = result.output
    assert out["applied"] is True and out["cancelled"] is True
    assert out["task_status"] == TaskStatus.CANCELLED.value and out["terminal"] is True
    assert out["cancellation_requested"] is True
    assert out["run_status"] == "running"  # the run row itself is never forged
    row = await _task(task.id)
    assert row["status"] == "cancelled" and row["cancellation_requested"] == 1

    # Replay on a terminal task is a no-op, not a fabricated second cancellation.
    replay = await _run(env, "run_cancel", {"run_id": "run-creatable"})
    assert replay.success and replay.output["applied"] is False
    assert "terminal" in replay.output["reason"]
    assert replay.output["task_status"] == "cancelled" and replay.output["cancelled"] is True

    # Report only the state the durable engine actually persisted: an unclaimed
    # running task is settled as cancelled, while a claimed one stays running.
    running = await env.engine.create("worker is busy")
    await env.engine.transition(running.id, TaskStatus.READY)
    await env.engine.transition(running.id, TaskStatus.RUNNING)
    await _seed_run("run-running", task_id=running.id)
    requested = await _run(env, "run_cancel", {"run_id": "run-running"})
    assert requested.success, requested.error
    assert requested.output["applied"] is True
    row = await _task(running.id)
    assert requested.output["task_status"] == row["status"]
    assert requested.output["cancelled"] is (row["status"] == TaskStatus.CANCELLED.value)
    assert requested.output["terminal"] is (row["status"] == TaskStatus.CANCELLED.value)
    assert requested.output["cancellation_requested"] is True
    assert row["cancellation_requested"] == 1
    assert any(event.topic == "task.cancellation_requested" for event in await env.events.history())

    # Completed tasks are terminal: never claim a cancellation was applied.
    done = await env.engine.create("already finished")
    await env.engine.transition(done.id, TaskStatus.READY)
    await env.engine.transition(done.id, TaskStatus.RUNNING)
    await env.engine.complete(done.id, {"ok": True}, [{"kind": "test"}])
    await _seed_run("run-complete", task_id=done.id, status="completed", finished_at="2026-11-19T05:00:00+00:00")
    terminal = await _run(env, "run_cancel", {"run_id": "run-complete"})
    assert terminal.success and terminal.output["applied"] is False
    assert terminal.output["task_status"] == "completed"
    assert terminal.output["cancelled"] is False and terminal.output["terminal"] is True
    assert (await _task(done.id))["cancellation_requested"] == 0

    # Runs without a durable task cannot be cancelled through this API.
    await _seed_run("run-unlinked")
    unlinked = await _run(env, "run_cancel", {"run_id": "run-unlinked"})
    assert unlinked.success and unlinked.output["applied"] is False
    assert "no linked durable task" in unlinked.output["reason"]

    missing = await _run(env, "run_cancel", {"run_id": "run-absent"})
    assert not missing.success and "unknown run" in (missing.error or "")


# ---------------------------------------------------------------- failures


async def test_cancel_injection_validation_policy_and_scope(env, monkeypatch):
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
    task = await env.engine.create("injected cancellation")
    await _seed_run("run-injected", task_id=task.id)
    result = await _run(env, "run_cancel", {"run_id": "run-injected"})
    assert result.success and result.output["applied"] is True
    assert "cancel" in spy.calls
    assert (await _task(task.id))["status"] == "cancelled"

    # With no injected engine and no configured database, cancellation fails
    # honestly instead of mutating a row itself.
    task2 = await env.engine.create("cannot reach service")
    await _seed_run("run-no-service", task_id=task2.id)
    monkeypatch.setattr(env.executor, "_context", original_context)
    monkeypatch.setattr(run_tools.db, "database_path_configured", lambda: False)
    unavailable = await _run(env, "run_cancel", {"run_id": "run-no-service"})
    assert not unavailable.success and "task service unavailable" in (unavailable.error or "")
    assert (await _task(task2.id))["status"] == "created"

    monkeypatch.setattr(run_tools.db, "database_path_configured", lambda: True)
    env.settings.security.allow_medium_risk = False
    denied = await _run(env, "run_cancel", {"run_id": "run-no-service"})
    assert not denied.success and denied.decision == "denied"
    assert (await _task(task2.id))["cancellation_requested"] == 0
    env.settings.security.allow_medium_risk = True

    for bad_args in ({}, {"run_id": 12}, {"run_id": "run-no-service", "extra": 1}):
        invalid = await _run(env, "run_status", bad_args)
        assert not invalid.success and invalid.decision == "invalid_arguments"
    bad_log = await _run(env, "run_log", {"run_id": "run-no-service", "limit": 9999})
    assert not bad_log.success and bad_log.decision == "invalid_arguments"
    out_of_scope = await env.executor.execute(
        ToolCall(name="run_list"), scope=frozenset({"utility"})
    )
    assert not out_of_scope.success and out_of_scope.decision == "out_of_scope"


# ---------------------------------------------------------------- run_log event scoping


async def test_run_log_excludes_unrelated_noisy_run_and_non_run_topics(env):
    mine = await env.engine.create("my run")
    theirs = await env.engine.create("another run")
    await _seed_run("run-scope-a", task_id=mine.id)
    await _seed_run("run-scope-b", task_id=theirs.id)
    rows = [
        ("event-mine", "run.output", json.dumps({"run_id": "run-scope-a", "text": "mine-only"}),
         "run-scope-a", "2026-11-19T01:00:00+00:00"),
        ("event-other", "run.output", json.dumps({"run_id": "run-scope-b", "text": "NOISY-OTHER-RUN"}),
         "run-scope-b", "2026-11-19T01:01:00+00:00"),
        ("event-task-other", "task.log", json.dumps({"task_id": theirs.id, "text": "NOISY-OTHER-TASK"}),
         theirs.id, "2026-11-19T01:02:00+00:00"),
        ("event-chat", "chat.done", json.dumps({"run_id": "run-scope-a", "text": "NOISY-CHAT"}),
         "run-scope-a", "2026-11-19T01:03:00+00:00"),
    ]
    await db.executemany(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) "
        "VALUES(?,?,'test',?,?,?)",
        rows,
    )

    result = await _run(env, "run_log", {"run_id": "run-scope-a"})
    assert result.success, result.error
    dumped = json.dumps(result.output)
    assert "mine-only" in dumped
    for noise in ("NOISY-OTHER-RUN", "NOISY-OTHER-TASK", "NOISY-CHAT"):
        assert noise not in dumped
    assert {entry["event"] for entry in result.output["entries"] if entry["source"] == "event"} == {
        "run.output",
    }


async def test_run_log_event_bound_applies_after_scoped_filter(env):
    await _seed_run("run-bounded", task_id=None)
    noise = [
        (
            f"noise-{index}", "chat.noise", "test",
            json.dumps({"run_id": "run-bounded", "text": f"noise-{index}"}),
            "run-bounded", "2026-11-19T01:00:00+00:00",
        )
        for index in range(run_tools.MAX_EVENT_SCAN + 10)
    ]
    noise.append((
        "event-needle", "run.output", "test",
        json.dumps({"run_id": "run-bounded", "text": "needle-after-the-bound"}),
        "run-bounded", "2026-11-19T02:00:00+00:00",
    ))
    await db.executemany(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) "
        "VALUES(?,?,?,?,?,?)",
        noise,
    )

    result = await _run(env, "run_log", {"run_id": "run-bounded"})
    assert result.success, result.error
    events = [entry for entry in result.output["entries"] if entry["source"] == "event"]
    assert [entry["event"] for entry in events] == ["run.output"]
    assert "needle-after-the-bound" in json.dumps(result.output)
