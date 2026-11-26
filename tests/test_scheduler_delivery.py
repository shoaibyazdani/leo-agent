"""Scheduler dispatch/delivery and keyset event pagination regressions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import TaskStatus
from leo import events as events_module
from leo.events import Event, EventBus, EventPage
from leo.tasks import DurableTaskEngine, Scheduler


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    engine = DurableTaskEngine(settings, events)
    return SimpleNamespace(settings=settings, events=events, engine=engine)


class RecordingEngine:
    """Stands in for the engine's registered-worker dispatch API."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def dispatch_ready(self, limit=None):
        self.calls.append(("dispatch", limit))
        return {"started": []}

    async def deliver_terminal(self, limit=None):
        self.calls.append(("delivery", limit))
        return {"delivered": []}


async def test_drain_uses_registered_worker_and_delivery_apis(env):
    engine = RecordingEngine()
    scheduler = Scheduler(engine, env.settings, env.events)
    outcome = await scheduler.drain(limit=7)
    assert engine.calls == [("dispatch", 7), ("delivery", 7)]
    assert outcome["dispatch"] == {"started": []}
    assert outcome["delivery"] == {"delivered": []}


async def test_drain_without_worker_api_fails_loudly(env):
    scheduler = Scheduler(object(), env.settings, env.events)
    with pytest.raises(RuntimeError, match="dispatch_ready"):
        await scheduler.drain()


async def test_tick_only_queues_scheduled_tasks(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval("ops", 60, "write report", {"channel": "ops"})
    await db.execute("UPDATE schedules SET next_run_at=? WHERE name=?", ("2030-01-01T09:00:00+00:00", "ops"))
    created = await scheduler.tick(now="2030-01-01T09:00:00+00:00")
    assert len(created) == 1
    task = await env.engine.get(created[0].id)
    assert task is not None and task.status is TaskStatus.CREATED
    assert task.metadata["schedule_payload"]["channel"] == "ops"
    assert await env.events.history("task.completed") == []


async def test_restart_tick_does_not_duplicate_occurrence_or_event(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval("heartbeat", 60, "send heartbeat", {"channel": "ops"})
    await db.execute("UPDATE schedules SET next_run_at=? WHERE name=?", ("2030-01-01T09:01:00+00:00", "heartbeat"))
    first = await scheduler.tick(now="2030-01-01T09:01:00+00:00")
    # Simulate a crash after the occurrence/event were persisted but before
    # the schedule cursor was advanced: a fresh process retries the tick.
    await db.execute("UPDATE schedules SET next_run_at=? WHERE name=?", ("2030-01-01T09:01:00+00:00", "heartbeat"))
    restarted = Scheduler(DurableTaskEngine(env.settings, env.events), env.settings, env.events)
    again = await restarted.tick(now="2030-01-01T09:01:00+00:00")
    rows = await db.fetchall("SELECT id FROM tasks")
    triggered = await env.events.history("schedule.triggered")
    assert len(first) == len(again) == 1
    assert first[0].id == again[0].id
    assert len(rows) == 1
    assert len(triggered) == 1
    assert triggered[0].id.startswith("schedule-triggered:")
    assert triggered[0].payload["task_count"] == 1
    # Genuine next occurrence still fires.
    second = await restarted.tick(now="2030-01-01T09:02:00+00:00")
    assert len(second) == 1 and second[0].id != first[0].id
    assert len(await env.events.history("schedule.triggered")) == 2


async def test_terminal_events_replayed_after_restart_without_losing_results(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval("report", 60, "report", {"channel": "ops"})
    created = await scheduler.tick(now="2030-01-01T09:00:00+00:00")
    task_id = created[0].id
    await env.engine.transition(task_id, TaskStatus.READY)
    await env.engine.complete(task_id, {"summary": "ok"}, [{"kind": "test", "detail": "passed"}])

    calls: list[object] = []

    async def dispatch_ready(limit=None):
        calls.append(("dispatch", limit))

    env.engine.dispatch_ready = dispatch_ready
    restarted = Scheduler(env.engine, env.settings, env.events)
    outcome = await restarted.drain(limit=5)
    terminal = await env.events.history("schedule.terminal")
    assert calls == [("dispatch", 5)]
    assert len(outcome["terminal_events"]) == 1
    assert len(terminal) == 1
    assert terminal[0].payload["task_id"] == task_id
    assert terminal[0].payload["status"] == "completed"
    assert terminal[0].payload["result"] == {"summary": "ok"}
    assert terminal[0].payload["schedule_payload"]["channel"] == "ops"
    await restarted.drain()
    assert len(await env.events.history("schedule.terminal")) == 1


async def test_drain_delivers_completed_schedule_result_after_restart(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval("deliver", 60, "work", {"channel": "ops"})
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?", ("2030-01-01T09:00:00+00:00", "deliver")
    )
    created = await scheduler.tick(now="2030-01-01T09:00:00+00:00")
    task_id = created[0].id
    await env.engine.transition(task_id, TaskStatus.READY)
    await env.engine.complete(task_id, {"ok": True}, [{"kind": "test", "detail": "done"}])

    sent: list[tuple[object, object]] = []

    async def deliver(task, channel, target):
        sent.append((channel, target))
        return "msg-1"

    restarted_engine = DurableTaskEngine(env.settings, env.events)
    restarted_engine.register_delivery("ops", deliver)
    restarted = Scheduler(restarted_engine, env.settings, env.events)
    await restarted.drain()
    assert sent == [("ops", None)]
    stored = await restarted_engine.get(task_id)
    assert stored is not None and stored.metadata["task.delivery"]["ops"]["status"] == "sent"
    await restarted.drain()
    assert sent == [("ops", None)]


async def test_occurrence_keys_are_schedule_id_with_a_legacy_dual_alias(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval("keyed", 60, "work")
    row = await db.fetchone("SELECT * FROM schedules WHERE name='keyed'")
    when = "2030-01-01T09:00:00+00:00"
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?", (when, "keyed")
    )
    created = await scheduler.tick(now=when)
    assert len(created) == 1
    task_id = created[0].id
    id_key = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        (f"task:idempotency:schedule:{row['id']}:{when}",),
    )
    legacy_key = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        (f"task:idempotency:schedule:keyed:{when}",),
    )
    marker = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        (f"schedule:occurrence:{row['id']}:{when}",),
    )
    assert id_key["value"] == legacy_key["value"] == task_id
    assert marker["value"] == task_id

    # A retried tick finds both keys and creates neither a second task nor a
    # second durable trigger event.
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?", (when, "keyed")
    )
    restarted = Scheduler(DurableTaskEngine(env.settings, env.events), env.settings, env.events)
    again = await restarted.tick(now=when)
    assert len(again) == 1 and again[0].id == task_id
    rows = await db.fetchall("SELECT id FROM tasks")
    assert len(rows) == 1
    assert len(await env.events.history("schedule.triggered")) == 1


async def test_failed_occurrence_transaction_rolls_back_every_mutation(env, monkeypatch):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval("atomic", 60, "work")
    when = "2030-01-01T09:00:00+00:00"
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?", (when, "atomic")
    )
    row = await db.fetchone("SELECT * FROM schedules WHERE name='atomic'")

    async def _explode(event):
        raise RuntimeError("event persistence failed")

    monkeypatch.setattr(env.events, "publish", _explode)
    with pytest.raises(RuntimeError, match="event persistence failed"):
        await scheduler.tick(now=when)

    # The claim marker, task, cursor and counter all share the failed
    # transaction: none may survive it, and no phantom event is persisted.
    assert await db.fetchall("SELECT id FROM tasks") == []
    assert (
        await db.fetchone(
            "SELECT key FROM kv WHERE key=?",
            (f"schedule:occurrence:{row['id']}:{when}",),
        )
        is None
    )
    fresh = await db.fetchone("SELECT next_run_at,claimed_count FROM schedules WHERE id=?", (row["id"],))
    assert fresh["next_run_at"] == when and fresh["claimed_count"] == 0
    assert await env.events.history("schedule.triggered") == []
    assert await env.events.history("task.created") == []

    # Restoring the bus lets the same due occurrence run exactly once.
    monkeypatch.setattr(env.events, "publish", EventBus.publish.__get__(env.events, EventBus))
    created = await scheduler.tick(now=when)
    assert len(created) == 1
    fresh = await db.fetchone("SELECT claimed_count FROM schedules WHERE id=?", (row["id"],))
    assert fresh["claimed_count"] == 1
    assert len(await env.events.history("schedule.triggered")) == 1


async def test_event_page_keyset_paging_counts_and_filters(env):
    for index in range(25):
        await env.events.emit("task.done" if index % 2 else "chat.done", index=index)
    first = await env.events.page(limit=10)
    assert isinstance(first, EventPage)
    assert first.total == 25 and first.has_more and len(first.events) == 10
    seen: list[str] = [event.id for event in first.events]
    cursor = first.next_cursor
    while cursor:
        page = await env.events.page(limit=10, cursor=cursor)
        assert page.total == 25
        seen.extend(event.id for event in page.events)
        cursor = page.next_cursor
    assert len(seen) == len(set(seen)) == 25
    full = await env.events.history(limit=1000)
    assert seen == [event.id for event in full]

    filtered = await env.events.page(topic="task.done", limit=100)
    assert filtered.total == 12 and len(filtered.events) == 12
    assert all(event.topic == "task.done" for event in filtered.events)
    prefixed = await env.events.page(topic_prefix="task.", limit=100)
    assert prefixed.total == 12
    assert await env.events.count(source="leo") == 25
    with pytest.raises(ValueError, match="malformed event cursor"):
        await env.events.page(cursor="not-a-cursor")


async def test_event_page_stable_order_for_equal_timestamps(env):
    moment = datetime.now(timezone.utc).isoformat()
    for index in range(3):
        await db.execute(
            "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) VALUES(?,?,?,?,?,?)",
            (f"tie-{index}", "tie", "leo", Event(id=f"tie-{index}", topic="tie", created_at=moment).model_dump_json(), None, moment),
        )
    page = await env.events.page(topic="tie", limit=2)
    assert [event.id for event in page.events] == ["tie-2", "tie-1"]
    assert page.next_cursor is not None
    next_page = await env.events.page(topic="tie", limit=2, cursor=page.next_cursor)
    assert [event.id for event in next_page.events] == ["tie-0"]
    assert next_page.has_more is False
    ascending = await env.events.page(topic="tie", limit=2, direction="after")
    assert [event.id for event in ascending.events] == ["tie-0", "tie-1"]


async def test_trigger_only_fires_named_schedule_at_requested_time(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_interval("manual", 60, "manual work", {"channel": "ops"})
    await scheduler.register_interval("other", 60, "other work", {"channel": "ops"})
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?",
        ("2031-01-01T00:00:00+00:00", "manual"),
    )
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?",
        ("2030-01-01T08:00:00+00:00", "other"),
    )

    triggered = await scheduler.trigger("manual", now="2030-01-01T09:00:00+00:00")
    assert len(triggered) == 1
    task = await env.engine.get(triggered[0].id)
    assert task is not None and task.status is TaskStatus.CREATED
    assert task.metadata["scheduled_for"] == "2030-01-01T09:00:00+00:00"

    rows = await db.fetchall("SELECT id FROM tasks")
    assert len(rows) == 1  # no global tick; the due 'other' schedule stayed idle
    other = await db.fetchone("SELECT next_run_at FROM schedules WHERE name='other'")
    assert other["next_run_at"] == "2030-01-01T08:00:00+00:00"
    manual = await db.fetchone("SELECT next_run_at FROM schedules WHERE name='manual'")
    assert manual["next_run_at"] == "2031-01-01T00:00:00+00:00"

    events = await env.events.history("schedule.triggered")
    assert len(events) == 1 and events[0].payload["name"] == "manual"
    assert events[0].payload["scheduled_for"] == "2030-01-01T09:00:00+00:00"


async def test_failed_event_insert_is_queued_in_outbox_and_flushed(env, monkeypatch):
    real_execute = events_module.db.execute

    async def flaky(sql, params=()):
        if str(sql).startswith("INSERT OR IGNORE INTO events("):
            raise RuntimeError("database is locked")
        return await real_execute(sql, params)

    monkeypatch.setattr(events_module.db, "execute", flaky)
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("ops.alert", seen.append)
    event = await bus.emit("ops.alert", value=1)
    assert seen == [event]  # in-process delivery survives the DB failure
    stats = await bus.stats()
    assert stats["event_outbox.pending"] == 1

    monkeypatch.setattr(events_module.db, "execute", real_execute)
    outcome = await bus.drain_outbox()
    assert outcome["flushed"] == 1
    assert [item.id for item in await bus.history("ops.alert")] == [event.id]
    stats = await bus.stats()
    assert stats["event_outbox.processed"] == 1
    assert stats["event_outbox.pending"] == 0


async def test_durable_subscription_replays_events_emitted_while_down(env):
    first_seen: list[str] = []
    first = EventBus()
    await first.subscribe_durable("ops-job", "task.*", lambda event: _record(first_seen, event), replay=False)
    old = await first.emit("task.old", n=1)
    await _await_record(first_seen, old.id)

    # Simulate a restarted process: no local handler, but the durable cursor
    # and delivery rows live in SQLite.
    down = EventBus()
    missed = await down.emit("task.missed", n=2)
    second_seen: list[str] = []
    restarted = EventBus()
    await restarted.subscribe_durable("ops-job", "task.*", lambda event: _record(second_seen, event))
    await _await_record(second_seen, missed.id)
    assert second_seen == [missed.id]

    third_seen: list[str] = []
    again = EventBus()
    await again.subscribe_durable("ops-job", "task.*", lambda event: _record(third_seen, event))
    assert third_seen == []  # cursor already covers both events
    assert [item["name"] for item in await again.durable_subscriptions()] == ["ops-job"]


async def test_durable_delivery_failures_are_bounded_and_explicit(env):
    calls: list[str] = []

    async def failing(event: Event) -> None:
        calls.append(event.id)
        raise RuntimeError("consumer down")

    bus = EventBus()
    await bus.subscribe_durable("flaky", "boom", failing, replay=False)
    event = await bus.emit("boom", n=1)
    assert calls == [event.id]
    assert (await bus.stats())["event_deliveries.pending"] == 1
    for expected in ("pending", "failed"):
        outcome = await bus.drain_deliveries(subscription="flaky", max_attempts=3)
        assert outcome[expected] == 1
    stats = await bus.stats()
    assert stats["event_deliveries.failed"] == 1
    assert stats["event_deliveries.pending"] == 0
    before_final_drain = len(calls)
    await bus.drain_deliveries(subscription="flaky", max_attempts=3)
    assert len(calls) == before_final_drain == 3  # failed is terminal; no extra retries
    assert calls == [event.id] * 3
    assert (await bus.stats())["event_deliveries.failed"] == 1


async def test_durable_delivery_retry_succeeds_and_advances_cursor(env):
    attempts = 0
    seen: list[str] = []

    async def flaky_then_ok(event: Event) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        seen.append(event.id)

    bus = EventBus()
    await bus.subscribe_durable("retry", "retry.topic", flaky_then_ok, replay=False)
    event = await bus.emit("retry.topic", n=1)
    assert seen == []
    assert (await bus.drain_deliveries(subscription="retry"))["delivered"] == 1
    assert seen == [event.id]
    subscriptions = await bus.durable_subscriptions()
    assert subscriptions[0]["cursor_seq"] is not None
    replay = await bus.replay_subscriptions(names=("retry",))
    assert replay["delivered"] == 0
    assert seen == [event.id]  # dedupe key blocks a second side effect


async def test_public_drain_combines_outbox_replay_and_deliveries(env):
    bus = EventBus()
    seen: list[str] = []

    async def consume(event: Event) -> None:
        seen.append(event.id)

    await bus.subscribe_durable("combined", "combined.*", consume, replay=False)
    event = await bus.emit("combined.one", n=1)
    await _await_record(seen, event.id)
    outcome = await bus.drain()
    assert set(outcome) == {"outbox", "replay", "deliveries"}
    assert outcome["outbox"]["flushed"] == 0
    assert seen == [event.id]


def _record(target: list[str], event: Event) -> None:
    target.append(event.id)


async def _await_record(target: list[str], event_id: str, timeout: float = 1.0) -> None:
    import asyncio

    async def wait() -> None:
        while event_id not in target:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout)
