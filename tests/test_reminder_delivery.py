"""Reminder delivery: a declared message reaches its declared address, once.

Regression for the 2026-11-23 reminder failure, where a scheduled reminder
fired, its task completed, and the message never reached the operator:

* ``schedule_payload`` declared ``{"channel": "telegram", "target": "525083981"}``
  but target extraction read only the channel, so delivery addressed the
  literal string ``None``;
* the adapter's failure came back as ``{"success": False}`` and the receipt was
  still recorded as ``sent``;
* the payload carried ``one_shot: true`` that nothing honoured, so it would
  have fired again on the next due slot.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import TaskStatus
from leo.events import EventBus
from leo.tasks import DurableTaskEngine, Scheduler
from leo.tasks.engine import _delivery_targets


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    return SimpleNamespace(
        settings=settings, events=events, engine=DurableTaskEngine(settings, events)
    )


async def test_payload_channel_and_target_become_a_delivery_address(env):
    task = await env.engine.create(
        "remind",
        metadata={
            "schedule_payload": {
                "kind": "reminder",
                "channel": "telegram",
                "target": "525083981",
            }
        },
    )
    assert _delivery_targets(task) == {"telegram": "525083981"}


async def test_payload_channel_without_address_is_not_addressed_to_none(env):
    task = await env.engine.create(
        "no address", metadata={"schedule_payload": {"channel": "telegram"}}
    )
    assert _delivery_targets(task) == {"telegram": None}


async def test_reminder_payload_selects_the_reminder_worker(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval(
        "ping", 60, "say something",
        {"channel": "telegram", "target": "1", "text": "hello there"},
    )
    created = await scheduler.tick(now=_far_future())
    stored = await env.engine.get(created[0].id)
    assert stored.metadata["schedule_payload"]["worker"] == "reminder"


async def test_one_shot_schedule_is_retired_after_one_occurrence(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval(
        "once", 60, "say something",
        {"channel": "telegram", "target": "1", "text": "hi", "one_shot": True},
    )
    await scheduler.tick(now=_far_future())
    rows = await db.fetchall("SELECT name FROM schedules")
    assert rows == []
    assert any(
        event.topic == "schedule.completed" for event in await env.events.history()
    )


async def test_manual_trigger_also_consumes_a_one_shot(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval(
        "once-manual", 60, "say something",
        {"channel": "telegram", "target": "1", "text": "hi", "one_shot": True},
    )
    await scheduler.trigger("once-manual", now=_far_future())
    rows = await db.fetchall("SELECT name FROM schedules")
    assert rows == []


async def test_recurring_schedule_survives_a_tick(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval(
        "recurring", 60, "say something",
        {"channel": "telegram", "target": "1", "text": "hi"},
    )
    await scheduler.tick(now=_far_future())
    rows = await db.fetchall("SELECT name FROM schedules")
    assert [row["name"] for row in rows] == ["recurring"]


def _far_future() -> str:
    return "2035-01-01T00:00:00+00:00"
