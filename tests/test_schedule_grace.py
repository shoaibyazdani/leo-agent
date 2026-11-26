"""Phase 6: operator-only grace_minutes late-run policy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.events import EventBus
from leo.schedules.clock import FrozenClock
from leo.schedules.store import ScheduleStore
from leo.tasks import DurableTaskEngine, Scheduler

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    clock = FrozenClock(NOW)
    engine = DurableTaskEngine(settings, events)
    scheduler = Scheduler(engine, settings, events, clock=clock)
    store = ScheduleStore(clock=clock)
    return SimpleNamespace(
        settings=settings,
        events=events,
        clock=clock,
        engine=engine,
        scheduler=scheduler,
        store=store,
    )


async def test_late_occurrence_outside_grace_is_skipped_and_cursor_advances(env) -> None:
    await env.scheduler.register_interval("job", 60, "do work", {})
    late = NOW - timedelta(minutes=10)
    await db.execute(
        "UPDATE schedules SET next_run_at=?, grace_minutes=1 WHERE name='job'",
        (late.isoformat(),),
    )
    created = await env.scheduler.tick(now=NOW)
    # Occurrences older than one minute are skipped. The bounded catch-up cap
    # is consumed by the nine skipped occurrences plus the 09:59 run; the 10:00
    # occurrence is truncated exactly like any other capped backlog.
    assert len(created) == 1
    assert created[0].metadata["scheduled_for"] == (NOW - timedelta(minutes=1)).isoformat()
    skipped = await env.events.history("schedule.skipped", limit=50)
    assert len(skipped) == 9
    assert all(event.payload["reason"] == "outside grace window" for event in skipped)
    truncated = await env.events.history("schedule.catchup_truncated", limit=10)
    assert truncated
    row = await db.fetchone("SELECT next_run_at FROM schedules WHERE name='job'")
    assert row["next_run_at"] is not None
    assert datetime.fromisoformat(row["next_run_at"]) > NOW


async def test_occurrence_inside_grace_runs(env) -> None:
    await env.scheduler.register_interval("job", 60, "do work", {})
    await db.execute(
        "UPDATE schedules SET next_run_at=?, grace_minutes=5 WHERE name='job'",
        ((NOW - timedelta(seconds=30)).isoformat(),),
    )
    created = await env.scheduler.tick(now=NOW)
    assert len(created) == 1
    assert await env.events.history("schedule.skipped", limit=10) == []


async def test_no_grace_column_keeps_existing_catchup_behavior(env) -> None:
    await env.scheduler.register_interval("job", 60, "do work", {})
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name='job'",
        ((NOW - timedelta(minutes=2)).isoformat(),),
    )
    created = await env.scheduler.tick(now=NOW)
    # Three missed minute occurrences are caught up (the bounded catch-up
    # loop), which is the pre-Phase-6 behavior when no grace is configured.
    assert len(created) == 3
