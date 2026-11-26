"""Phase 6: bounded monitor baseline, suppression, wake and failure lane."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.core.app import LeoApp
from leo.events import EventBus
from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompileError, ScheduleCompiler
from leo.schedules.delivery import DeliveryContext, Origin
from leo.schedules.execution import ScheduleExecution
from leo.schedules.monitor import Monitor, MonitorError, validate_monitor_url
from leo.schedules.script_runner import ScriptExecutionError, ScriptRunner
from leo.schedules.store import ScheduleStore
from leo.tasks import DurableTaskEngine, Scheduler

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


class FakeFetcher:
    def __init__(self, content: str = "first") -> None:
        self.content = content
        self.fail = False
        self.calls = 0

    async def __call__(self, args, ctx):
        self.calls += 1
        if self.fail:
            raise PermissionError("monitor target not allowed: private")
        return {
            "content": self.content,
            "status_code": 200,
            "truncated": False,
        }


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.automation.allow_scripts = True
    events = EventBus()
    clock = FrozenClock(NOW)
    engine = DurableTaskEngine(settings, events)
    scheduler = Scheduler(engine, settings, events, clock=clock)
    store = ScheduleStore(clock=clock)
    fetcher = FakeFetcher()
    execution = ScheduleExecution(
        settings=settings,
        store=store,
        clock=clock,
        scripts_root=tmp_path,
        allowed_workdir_roots=[tmp_path],
        http_fetcher=fetcher,
        display_chars=50,
    )
    scheduler.attach_execution(execution)
    return SimpleNamespace(
        settings=settings,
        events=events,
        clock=clock,
        engine=engine,
        scheduler=scheduler,
        store=store,
        execution=execution,
        fetcher=fetcher,
        tmp_path=tmp_path,
    )


async def _monitor_schedule(env, *, target: str = "https://example.test/feed") -> None:
    await env.scheduler.register_interval(
        "mon",
        60,
        "check the feed",
        {"name": "mon", "execution": {"monitor": target}},
    )
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name='mon'",
        (NOW.isoformat(),),
    )


async def test_first_run_baselines_and_unchanged_suppresses(env) -> None:
    await _monitor_schedule(env)
    first = await env.scheduler.tick(now=NOW)
    assert first == []
    assert await db.fetchone("SELECT COUNT(*) AS n FROM tasks") is not None
    count = await db.fetchone("SELECT COUNT(*) AS n FROM tasks")
    assert int(count["n"]) == 0
    baseline = await env.store.runtime_get("sch", "monitor_baseline")
    assert baseline is None
    row = await db.fetchone("SELECT id FROM schedules WHERE name='mon'")
    baseline = await env.store.runtime_get(row["id"], "monitor_baseline")
    assert baseline is not None
    events = await env.events.history("schedule.monitor_unchanged", limit=10)
    assert events and events[0].payload["first_run"] is True

    second = await env.scheduler.tick(now=NOW.replace(minute=1))
    assert second == []
    count = await db.fetchone("SELECT COUNT(*) AS n FROM tasks")
    assert int(count["n"]) == 0
    events = await env.events.history("schedule.monitor_unchanged", limit=10)
    assert any(not event.payload["first_run"] for event in events)


async def test_change_wakes_with_bounded_old_and_new_content(env) -> None:
    await _monitor_schedule(env)
    await env.scheduler.tick(now=NOW)
    env.fetcher.content = "changed " + ("x" * 500)
    created = await env.scheduler.tick(now=NOW.replace(minute=1))
    assert len(created) == 1
    task = await env.engine.get(created[0].id)
    payload = task.metadata["schedule_payload"]
    assert payload["worker"] == "schedule_monitor"
    change = payload["monitor_change"]
    assert change["old"] == "first"
    assert change["new"].startswith("changed ")
    assert len(change["new"]) <= 50 + len("\n[truncated]")
    events = await env.events.history("schedule.monitor_changed", limit=10)
    assert events and events[0].payload["new_chars"] > 0


async def test_baseline_and_task_commit_together_or_neither(env, monkeypatch) -> None:
    await _monitor_schedule(env)
    await env.scheduler.tick(now=NOW)  # establish the baseline
    row = await db.fetchone("SELECT id FROM schedules WHERE name='mon'")
    baseline_before = await env.store.runtime_get(row["id"], "monitor_baseline")
    assert baseline_before is not None

    env.fetcher.content = "changed"
    original_create = env.engine.create

    async def _explode(*args, **kwargs):
        raise RuntimeError("task insert failed")

    # A crash/failure between the deferred preflight baseline and task creation
    # must roll the baseline back, or the retry would compare against the new
    # content and silently suppress the change forever.
    monkeypatch.setattr(env.engine, "create", _explode)
    with pytest.raises(RuntimeError, match="task insert failed"):
        await env.scheduler.tick(now=NOW.replace(minute=1))
    assert await env.store.runtime_get(row["id"], "monitor_baseline") == baseline_before
    count = await db.fetchone("SELECT COUNT(*) AS n FROM tasks")
    assert int(count["n"]) == 0

    monkeypatch.setattr(env.engine, "create", original_create)
    created = await env.scheduler.tick(now=NOW.replace(minute=1))
    assert len(created) == 1
    task = await env.engine.get(created[0].id)
    assert task.metadata["schedule_payload"]["monitor_change"]["old"] == "first"
    persisted = json.loads(
        await env.store.runtime_get(row["id"], "monitor_baseline")
    )
    assert persisted["content"] == "changed"


async def test_failure_uses_failure_lane_and_preserves_baseline(env) -> None:
    await _monitor_schedule(env)
    await env.scheduler.tick(now=NOW)
    row = await db.fetchone("SELECT id FROM schedules WHERE name='mon'")
    good = await env.store.runtime_get(row["id"], "monitor_baseline")
    assert good is not None

    env.fetcher.fail = True
    created = await env.scheduler.tick(now=NOW.replace(minute=1))
    assert len(created) == 1
    task = await env.engine.get(created[0].id)
    payload = task.metadata["schedule_payload"]
    assert payload["worker"] == "schedule_monitor"
    assert "private" in payload["monitor_error"]
    assert await env.store.runtime_get(row["id"], "monitor_baseline") == good
    events = await env.events.history("schedule.monitor_failed", limit=10)
    assert events and "private" in events[0].payload["error"]

    # The monitor worker turns the prefetched failure into a failed task.
    with pytest.raises(Exception):
        await env.execution.run_monitor(task=task, context={"schedule_payload": payload})
    assert env.fetcher.calls == 2


def test_private_targets_and_credentials_are_rejected() -> None:
    for url in (
        "http://127.0.0.1/feed",
        "http://10.0.0.5/feed",
        "http://localhost/feed",
        "https://user:pass@example.test/feed",
        "ftp://example.test/feed",
    ):
        with pytest.raises(MonitorError):
            validate_monitor_url(url)
    assert validate_monitor_url("https://example.test/feed") == "https://example.test/feed"


async def test_script_monitor_uses_restricted_runner(env) -> None:
    script = env.tmp_path / "monitor.sh"
    script.write_text("#!/bin/sh\nprintf 'observed'\n", encoding="utf-8")
    script.chmod(0o755)
    runner = ScriptRunner(
        scripts_root=env.tmp_path,
        allow_scripts=True,
        timeout_seconds=5,
        max_output_bytes=1000,
        allowed_workdir_roots=[env.tmp_path],
    )
    store = env.store
    clock = env.clock
    monitor = Monitor(store=store, clock=clock, script_runner=runner)
    first = await monitor.check(schedule_id="sch_script", target="monitor.sh")
    assert first.suppressed and first.first_run
    second = await monitor.check(schedule_id="sch_script", target="monitor.sh")
    assert second.suppressed and not second.first_run
    assert second.new_content == "observed"

    with pytest.raises(ScriptExecutionError):
        runner.resolve("../escape.sh")


async def test_monitor_preflight_is_selected_from_persisted_config(env) -> None:
    await _monitor_schedule(env)
    decision = await env.execution.preflight(
        schedule_id="sch_1",
        name="mon",
        payload={"execution": {"monitor": "https://example.test/feed"}},
        scheduled_for=NOW,
    )
    assert decision is not None and decision.suppress is True
    assert decision.event_topic == "schedule.monitor_unchanged"


async def test_monitor_wake_runs_the_job_agent_when_prompt_present(env) -> None:
    async def _heartbeat(*_args, **_kwargs):
        return None

    async def _failed(*_args, **_kwargs):
        raise AssertionError("unexpected failure")

    captured: dict[str, object] = {}

    async def _agent(task, engine, context, *, objective=None, extra_objective=None):
        captured["extra"] = extra_objective
        return {"output": "agent result"}

    app = object.__new__(LeoApp)
    app.schedule_execution = env.execution
    app._start_heartbeat = _heartbeat
    app._schedule_worker_failed = _failed
    app._run_schedule_agent = _agent

    task = SimpleNamespace(
        id="t_monitor",
        objective="summarize the feed",
        metadata={"schedule_id": "sch_1", "schedule": "mon"},
    )
    context = {
        "objective": "summarize the feed",
        "schedule_payload": {
            "name": "mon",
            "prompt": "summarize the feed",
            "execution": {"monitor": "https://example.test/feed"},
            "monitor_change": {"old": "before", "new": "after", "source": "url"},
        },
    }

    class FakeEngine:
        async def complete(self, *args, **kwargs):
            raise AssertionError("the agent path should own completion")

        async def fail(self, *args, **kwargs):
            raise AssertionError("failure not expected")

    result = await LeoApp._schedule_monitor_worker(
        app, task, FakeEngine(), context
    )
    assert result["output"] == "agent result"
    assert "before" in str(captured["extra"])
    assert "after" in str(captured["extra"])


def test_monitor_and_script_are_mutually_exclusive(tmp_path) -> None:
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    compiler = ScheduleCompiler(
        FrozenClock(NOW),
        default_timezone="UTC",
        scripts_root=tmp_path,
        allowed_workdir_roots=[tmp_path],
        delivery_context=DeliveryContext(origin=Origin("telegram", "1")),
    )
    with pytest.raises(ScheduleCompileError) as caught:
        compiler.compile_create(
            {
                "schedule": "every day at 9am",
                "prompt": "Brief",
                "deliver": "local",
                "script": "job.sh",
                "monitor": "https://example.test/feed",
            }
        )
    assert caught.value.code == "monitor_incompatible_with_script"
