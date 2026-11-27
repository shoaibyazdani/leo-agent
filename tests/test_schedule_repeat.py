"""Phase 7: atomic repeat admission, completion accounting and retirement."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import TurnContext
from leo.events import EventBus
from leo.security.approval_cards import ScheduleBodySpec, render_schedule_body
from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompiler
from leo.schedules.delivery import DeliveryContext
from leo.schedules.service import ScheduleService
from leo.schedules.store import ScheduleStore
from leo.security.permissions import PermissionManagerImpl
from leo.tasks import DurableTaskEngine, Scheduler

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def _operator() -> TurnContext:
    return TurnContext(
        principal="operator:cli", channel="cli", surface="cli", source="operator"
    )


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    # Repeat admission is the subject here; the global concurrency gate would
    # defer later occurrences behind earlier queued tasks.
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


async def _create(env, **arguments) -> dict:
    return await env.service.create(
        arguments, actor="operator:cli", turn=_operator()
    )


async def _schedule_row(env, name: str):
    return await db.fetchone("SELECT * FROM schedules WHERE name=?", (name,))


async def _due(env, name: str, moment: datetime) -> None:
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?",
        (moment.isoformat(), name),
    )


async def _complete(env, task_id: str) -> None:
    await env.engine.complete(task_id, {"ok": True}, [{"kind": "test", "claim": "done"}])


async def test_omitted_repeat_runs_forever_and_counts_each_terminal_once(env):
    out = await _create(env, name="forever", schedule="1h", prompt="work")
    assert out["status"] == "created"
    await _due(env, "forever", NOW)

    first = await env.scheduler.tick(now=NOW)
    assert len(first) == 1
    row = await _schedule_row(env, "forever")
    assert row["repeat_limit"] is None
    assert row["claimed_count"] == 1
    assert row["next_run_at"] == "2026-11-26T11:00:00+00:00"

    await _complete(env, first[0].id)
    view = await env.store.get(row["id"])
    assert view.claimed_count == 1 and view.completed_count == 1

    # Re-saving a terminal task and replaying terminal events must not count it
    # a second time: the durable marker is the authority.
    task = await env.engine.get(first[0].id)
    await env.engine._save(task)
    await env.scheduler.drain()
    await env.scheduler.drain()
    row = await _schedule_row(env, "forever")
    assert row["completed_count"] == 1

    second = await env.scheduler.tick(now=datetime(2026, 11, 26, 11, 0, tzinfo=timezone.utc))
    assert len(second) == 1 and second[0].id != first[0].id
    row = await _schedule_row(env, "forever")
    assert row["claimed_count"] == 2
    assert row["status"] == "active" and row["enabled"] == 1


async def test_numeric_string_repeat_three_admits_exactly_three_then_tombstones(env):
    out = await _create(env, name="three", schedule="1h", prompt="work", repeat="3")
    assert out["status"] == "created"
    row = await _schedule_row(env, "three")
    assert row["repeat_limit"] == 3
    schedule_id = row["id"]
    await _due(env, "three", NOW)

    created = []
    for hour in range(3):
        tasks = await env.scheduler.tick(
            now=datetime(2026, 11, 26, 10 + hour, 0, tzinfo=timezone.utc)
        )
        assert len(tasks) == 1
        created.append(tasks[0])
    row = await _schedule_row(env, "three")
    assert row["claimed_count"] == 3

    # The fourth due slot has no repeat slot left: the schedule retires with a
    # tombstone instead of firing again.
    again = await env.scheduler.tick(now=datetime(2026, 11, 26, 13, 0, tzinfo=timezone.utc))
    assert again == []
    assert await db.fetchone("SELECT id FROM schedules WHERE id=?", (schedule_id,)) is None
    tomb = await env.store.get_tombstone(schedule_id)
    assert tomb is not None and tomb["reason"] == "repeat limit reached"
    assert tomb["name"] == "three"
    events = await env.events.history("schedule.completed", limit=10)
    assert any(event.payload["schedule_id"] == schedule_id for event in events)

    # Terminal completion is still counted once per created task even though
    # the schedule row is gone (the durable markers remain inspectable).
    for task in created:
        await _complete(env, task.id)
    markers = await db.fetchall(
        "SELECT key FROM kv WHERE key LIKE 'schedule:completed:%' ORDER BY key"
    )
    assert len(markers) == 3


async def test_lowering_below_claimed_blocks_and_raising_reopens_admission(env):
    await _create(env, name="lower", schedule="1h", prompt="work", repeat=5)
    await _due(env, "lower", NOW)
    await env.scheduler.tick(now=NOW)
    await env.scheduler.tick(now=datetime(2026, 11, 26, 11, 0, tzinfo=timezone.utc))
    await env.scheduler.tick(now=datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    row = await _schedule_row(env, "lower")
    assert row["claimed_count"] == 3
    await db.execute("UPDATE schedules SET repeat_limit=2 WHERE id=?", (row["id"],))
    await env.scheduler.tick(now=datetime(2026, 11, 26, 13, 0, tzinfo=timezone.utc))
    assert await db.fetchone("SELECT id FROM schedules WHERE id=?", (row["id"],)) is None

    await _create(env, name="raise", schedule="1h", prompt="work", repeat=2)
    await _due(env, "raise", NOW)
    await env.scheduler.tick(now=NOW)
    await env.scheduler.tick(now=datetime(2026, 11, 26, 11, 0, tzinfo=timezone.utc))
    row = await _schedule_row(env, "raise")
    assert row["claimed_count"] == 2
    # Operator raises the limit above what was claimed: admission opens again
    # without resetting the existing counters.
    await db.execute("UPDATE schedules SET repeat_limit=4 WHERE id=?", (row["id"],))
    task = await env.scheduler.tick(now=datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    assert len(task) == 1
    row = await _schedule_row(env, "raise")
    assert row["claimed_count"] == 3 and row["repeat_limit"] == 4


async def test_manual_runs_never_consume_repeat_or_completion(env):
    await _create(env, name="manual", schedule="1h", prompt="work", repeat=1)
    row = await _schedule_row(env, "manual")

    run = await env.service.run_now(
        name="manual", actor="chat:web", turn=_operator()
    )
    await _complete(env, run["task_id"])
    row = await _schedule_row(env, "manual")
    assert row["claimed_count"] == 0 and row["completed_count"] == 0
    assert row["next_run_at"] == "2026-11-26T11:00:00+00:00"

    triggered = await env.scheduler.trigger(
        "manual", now=datetime(2026, 11, 26, 10, 15, tzinfo=timezone.utc)
    )
    assert len(triggered) == 1
    await _complete(env, triggered[0].id)
    row = await _schedule_row(env, "manual")
    assert row["claimed_count"] == 0 and row["completed_count"] == 0

    # The single scheduled occurrence is still available and does consume the
    # finite slot.
    await _due(env, "manual", NOW)
    scheduled = await env.scheduler.tick(now=NOW)
    assert len(scheduled) == 1
    row = await _schedule_row(env, "manual")
    assert row["claimed_count"] == 1


async def test_once_schedule_hard_deletes_and_records_a_tombstone(env):
    out = await _create(env, name="once", schedule="in 2h", prompt="remind")
    assert out["status"] == "created"
    row = await _schedule_row(env, "once")
    assert row["kind"] == "once" and row["repeat_limit"] == 1
    schedule_id = row["id"]
    await _due(env, "once", NOW)

    first = await env.scheduler.tick(now=NOW)
    assert len(first) == 1
    assert await db.fetchone("SELECT id FROM schedules WHERE id=?", (schedule_id,)) is None
    tomb = await env.store.get_tombstone(schedule_id)
    assert tomb is not None and tomb["reason"] == "one-shot occurrence created"
    assert json.loads(tomb["payload"]) is not None

    # A second tick never repeats the retired reminder, and completion cannot
    # resurrect it.
    assert await env.scheduler.tick(now=NOW) == []
    await _complete(env, first[0].id)
    assert await db.fetchone("SELECT id FROM schedules WHERE name='once'") is None


async def test_tombstones_are_queried_by_schedule_id_and_names_can_be_reused(env):
    await _create(env, name="reusable", schedule="1h", prompt="work", repeat="2")
    await _due(env, "reusable", NOW)
    await env.scheduler.tick(now=NOW)
    await env.scheduler.tick(now=datetime(2026, 11, 26, 11, 0, tzinfo=timezone.utc))
    row = await _schedule_row(env, "reusable")
    old_id = row["id"]
    await env.scheduler.tick(now=datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    assert await db.fetchone("SELECT id FROM schedules WHERE id=?", (old_id,)) is None

    tomb = await env.store.get_tombstone(old_id)
    assert tomb is not None and tomb["schedule_id"] == old_id
    # Explicit active-name reuse: the tombstone is history, not a reservation.
    fresh = await _create(env, name="reusable", schedule="1h", prompt="new work")
    assert fresh["status"] == "created"
    new_row = await _schedule_row(env, "reusable")
    assert new_row["id"] != old_id


def test_numeric_string_repeat_renders_as_three_on_the_card():
    clock = FrozenClock(NOW)
    compiler = ScheduleCompiler(
        clock,
        default_timezone="UTC",
        delivery_context=DeliveryContext(),
    )
    result = compiler.compile_create(
        {"name": "three", "schedule": "1h", "prompt": "work", "repeat": "3"},
        operator=True,
    )
    assert result.spec.repeat_limit == 3
    body = render_schedule_body(
        ScheduleBodySpec(
            name=result.spec.name,
            display=result.display,
            first_run_at=result.first_run_at,
            repeat_limit=result.spec.repeat_limit,
            prompt=result.spec.prompt,
        )
    )
    assert "3 runs" in body
