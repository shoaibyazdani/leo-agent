"""Phase 7: exactly-once recovery, cancellation and terminal replay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import TaskStatus, TurnContext
from leo.events import EventBus
from leo.schedules.clock import FrozenClock
from leo.schedules.service import ScheduleService
from leo.schedules.store import ScheduleStore
from leo.security.permissions import PermissionManagerImpl
from leo.tasks import DurableTaskEngine, Scheduler

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def _chat() -> TurnContext:
    return TurnContext(
        principal="chat:web",
        channel="tui",
        conversation_id="conv-a",
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
    settings.automation.max_concurrent_tasks = 0
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
        allowed_workdir_roots=[tmp_path],
    )
    return SimpleNamespace(
        settings=settings,
        events=events,
        clock=clock,
        engine=engine,
        scheduler=scheduler,
        store=store,
        service=service,
        permissions=permissions,
    )


async def _due(env, name: str, moment: datetime) -> None:
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?", (moment.isoformat(), name)
    )


async def _task_count() -> int:
    row = await db.fetchone("SELECT COUNT(*) AS n FROM tasks")
    return int(row["n"])


async def test_restart_during_claim_keeps_one_task_and_one_event(env):
    await env.scheduler.register_interval("heartbeat", 60, "send heartbeat")
    await _due(env, "heartbeat", NOW)
    first = await env.scheduler.tick(now=NOW)
    assert len(first) == 1
    # Simulate a crash after the occurrence transaction committed but before
    # anything else observed the cursor: a new process still sees it due.
    await _due(env, "heartbeat", NOW)

    restarted = Scheduler(
        DurableTaskEngine(env.settings, env.events), env.settings, env.events, clock=env.clock
    )
    again = await restarted.tick(now=NOW)
    assert len(again) == 1 and again[0].id == first[0].id
    assert await _task_count() == 1
    assert len(await env.events.history("schedule.triggered")) == 1
    row = await db.fetchone("SELECT next_run_at FROM schedules WHERE name='heartbeat'")
    assert row["next_run_at"] == "2026-11-26T10:01:00+00:00"


async def test_retry_adopts_a_task_claimed_under_the_legacy_name_key(env):
    await env.scheduler.register_interval("legacy", 60, "work")
    row = await db.fetchone("SELECT * FROM schedules WHERE name='legacy'")
    when = NOW.isoformat()
    await _due(env, "legacy", NOW)
    # A pre-Phase-7 occurrence: only the name-keyed idempotency claim exists,
    # and no occurrence marker.
    pre_upgrade = await env.engine.create(
        "work",
        metadata={
            "schedule": "legacy",
            "schedule_id": row["id"],
            "scheduled_for": when,
        },
        idempotency_key=f"schedule:legacy:{when}",
    )

    restarted = Scheduler(
        DurableTaskEngine(env.settings, env.events), env.settings, env.events, clock=env.clock
    )
    again = await restarted.tick(now=NOW)
    assert len(again) == 1 and again[0].id == pre_upgrade.id
    assert await _task_count() == 1
    alias = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        (f"task:idempotency:schedule:{row['id']}:{when}",),
    )
    assert alias is not None and alias["value"] == pre_upgrade.id
    marker = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        (f"schedule:occurrence:{row['id']}:{when}",),
    )
    assert marker is not None and marker["value"] == pre_upgrade.id


async def test_cancelled_run_stays_terminal_across_restart_and_replays(env):
    await env.scheduler.register_interval("cancel", 60, "work")
    await _due(env, "cancel", NOW)
    created = await env.scheduler.tick(now=NOW)
    task_id = created[0].id
    assert await env.engine.cancel(task_id) is True
    row = await db.fetchone("SELECT completed_count FROM schedules WHERE name='cancel'")
    assert row["completed_count"] == 1

    restarted = Scheduler(
        DurableTaskEngine(env.settings, env.events), env.settings, env.events, clock=env.clock
    )
    await restarted.drain()
    assert (await env.engine.get(task_id)).status is TaskStatus.CANCELLED
    # Cancellation is absorbing: claiming or running it again cannot resurrect
    # or replay the task.
    assert await env.engine._claim(task_id, note="recovery probe") is None
    replayed = await restarted.drain()
    terminal = await env.events.history("schedule.terminal", limit=20)
    cancelled = [event for event in terminal if event.payload["task_id"] == task_id]
    assert len(cancelled) == 1
    assert cancelled[0].payload["status"] == "cancelled"
    row = await db.fetchone("SELECT completed_count FROM schedules WHERE name='cancel'")
    assert row["completed_count"] == 1  # counted once, not per replay
    assert replayed["recovered_cancellations"] == []


async def test_restart_settles_a_cancellation_requested_but_never_observed(env):
    await env.scheduler.register_interval("orphan", 60, "work")
    await _due(env, "orphan", NOW)
    created = await env.scheduler.tick(now=NOW)
    task_id = created[0].id
    # A worker owned the task when the stop landed; the flag is durable but
    # the process died before settling it.
    await db.execute(
        "UPDATE tasks SET cancellation_requested=1 WHERE id=?", (task_id,)
    )
    restarted = Scheduler(
        DurableTaskEngine(env.settings, env.events), env.settings, env.events, clock=env.clock
    )
    outcome = await restarted.drain()
    assert outcome["recovered_cancellations"] == [task_id]
    assert (await env.engine.get(task_id)).status is TaskStatus.CANCELLED
    row = await db.fetchone("SELECT completed_count FROM schedules WHERE name='orphan'")
    assert row["completed_count"] == 1


async def test_deferred_backlog_keeps_the_cursor_and_retries(env):
    env.settings.automation.max_concurrent_tasks = 1
    scheduler = Scheduler(env.engine, env.settings, env.events, clock=env.clock)
    for name in ("first", "second"):
        await scheduler.register_interval(name, 60, f"work {name}")
        await _due(env, name, NOW)

    created = await scheduler.tick(now=NOW)
    assert len(created) == 1
    deferred = await env.events.history("schedule.deferred", limit=10)
    assert len(deferred) == 1
    rows = {
        row["name"]: row["next_run_at"]
        for row in await db.fetchall("SELECT name,next_run_at FROM schedules")
    }
    assert rows["second"] == NOW.isoformat()

    env.settings.automation.max_concurrent_tasks = 0
    again = await scheduler.tick(now=NOW)
    assert len(again) == 1
    assert await _task_count() == 2


async def test_catch_up_is_bounded_and_records_truncation(env):
    await env.scheduler.register_interval("minute", 60, "minute work")
    await _due(env, "minute", NOW - timedelta(hours=25))
    due = await env.scheduler.tick(now=NOW)
    assert len(due) == 10
    assert await env.events.history("schedule.catchup_truncated", limit=10)
    row = await db.fetchone("SELECT next_run_at FROM schedules WHERE name='minute'")
    assert datetime.fromisoformat(row["next_run_at"]) > NOW


async def test_terminal_replay_is_idempotent_after_restart(env):
    await env.scheduler.register_interval("report", 60, "report")
    await _due(env, "report", NOW)
    created = await env.scheduler.tick(now=NOW)
    await env.engine.complete(
        created[0].id, {"summary": "ok"}, [{"kind": "test", "claim": "done"}]
    )
    restarted = Scheduler(
        DurableTaskEngine(env.settings, env.events), env.settings, env.events, clock=env.clock
    )
    first = await restarted.drain()
    second = await restarted.drain()
    assert len(first["terminal_events"]) == 1
    assert second["terminal_events"] == []
    terminal = await env.events.history("schedule.terminal", limit=10)
    assert len(terminal) == 1
    assert terminal[0].payload["result"] == {"summary": "ok"}


async def test_stopping_a_turn_withdraws_its_pending_card_without_committing(env):
    out = await env.service.create(
        {"name": "water", "schedule": "1h", "prompt": "Drink water."},
        actor="chat:web",
        turn=_chat(),
        run_id="run-1",
    )
    assert out["status"] == "approval_required"
    assert out["card"] is not None
    await env.service.withdraw_turn(run_id="run-1", reason="turn stopped")

    proposal = await db.fetchone(
        "SELECT status FROM schedule_proposals WHERE id=?", (out["proposal_id"],)
    )
    approval = await db.fetchone(
        "SELECT status, deny_reason FROM approvals WHERE id=?",
        (out["approval_id"],),
    )
    assert proposal["status"] == "withdrawn"
    assert approval["status"] == "withdrawn"
    assert approval["deny_reason"] == "turn stopped"
    assert await db.fetchall("SELECT id FROM schedules") == []

    with pytest.raises(ValueError, match="already withdrawn"):
        await env.permissions.resolve_schedule_card(
            out["approval_id"],
            choice="once",
            resolver="operator:local",
            confirm_proposal_digest=out["card"]["proposal_digest"],
            commit=env.service.commit,
        )
    # The withdrawn proposal released the active name.
    retry = await env.service.create(
        {"name": "water", "schedule": "1h", "prompt": "Drink water."},
        actor="chat:web",
        turn=_chat(),
        run_id="run-2",
    )
    assert retry["status"] == "approval_required"
