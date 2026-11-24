"""Tests for the durable task engine, scheduler and signed webhook intake."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import TaskStatus
from leo.events import EventBus
from leo.tasks import (
    DuplicateDelivery,
    DurableTaskEngine,
    InvalidSignature,
    Scheduler,
    TaskConflictError,
    WebhookError,
    WebhookHandler,
)
from leo.tasks.scheduler import _advance_past_due, every_interval, next_run, parse_cron


@pytest.fixture
async def env(tmp_path):
    # The shared conftest autouse fixture also isolates the DB per test.
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    engine = DurableTaskEngine(settings, events)
    return SimpleNamespace(settings=settings, events=events, engine=engine)


async def _running(env, objective="work", **kwargs):
    task = await env.engine.create(objective, **kwargs)
    await env.engine.transition(task.id, TaskStatus.READY)
    await env.engine.transition(task.id, TaskStatus.RUNNING)
    return task


# ----------------------------------------------------------------------
# Engine lifecycle, evidence and history
# ----------------------------------------------------------------------
async def test_full_lifecycle_and_progress(env):
    task = await env.engine.create(
        "ship feature",
        plan=["plan", "implement"],
        dependencies=["design"],
        owner_agent="builder",
        resource_limits={"max_retries": 1},
    )
    assert task.status is TaskStatus.CREATED
    assert task.plan == ["plan", "implement"]
    assert (await env.engine.get(task.id)).dependencies == ["design"]

    with pytest.raises(ValueError):
        await env.engine.transition(task.id, TaskStatus.COMPLETED)
    for status in (TaskStatus.PLANNING, TaskStatus.READY, TaskStatus.RUNNING):
        await env.engine.transition(task.id, status, note="step")

    await env.engine.checkpoint(task.id, {"step": 1, "value": "a"})
    await env.engine.progress(task.id, "halfway", step=2)
    finished = await env.engine.complete(
        task.id, {"summary": "done"}, [{"kind": "test", "detail": "unit tests passed"}]
    )
    assert finished.status is TaskStatus.COMPLETED
    assert finished.current_step == 2
    assert finished.completed_at is not None
    assert finished.result == {"summary": "done"}
    assert finished.evidence[0]["kind"] == "test"
    events = [h["event"] for h in finished.history]
    assert {"created", "transition", "checkpoint", "progress", "completed"} <= set(events)
    assert len(await env.engine.checkpoints(task.id)) == 1

    completed_events = await env.events.history("task.completed")
    assert len(completed_events) == 1
    assert completed_events[0].payload["task_id"] == task.id
    assert any(e.topic == "task.progress" for e in await env.events.history())

    with pytest.raises(ValueError):
        await env.engine.transition(task.id, TaskStatus.READY)


async def test_completion_requires_evidence(env):
    task = await _running(env, "no evidence")
    with pytest.raises(ValueError, match="evidence"):
        await env.engine.complete(task.id, {"ok": True}, [])
    with pytest.raises(ValueError, match="evidence"):
        await env.engine.complete(task.id, {"ok": True}, None)  # type: ignore[arg-type]
    assert (await env.engine.get(task.id)).status is TaskStatus.RUNNING
    assert (await env.engine.get(task.id)).evidence == []
    done = await env.engine.complete(task.id, {"ok": True}, [{"kind": "artifact", "path": "out.txt"}])
    assert done.status is TaskStatus.COMPLETED


async def test_direct_evidence_completion_and_cancel_return(env):
    task = await env.engine.create("direct")
    done = await env.engine.complete(task.id, "ok", [{"kind": "manual"}])
    assert done.status is TaskStatus.COMPLETED
    assert done.evidence == [{"kind": "manual"}]

    other = await env.engine.create("cancel via protocol")
    assert await env.engine.cancel(other.id) is True
    assert (await env.engine.get(other.id)).cancellation_requested is True


async def test_fail_records_error_and_events(env):
    task = await _running(env)
    failed = await env.engine.fail(task.id, "tool exploded")
    assert failed.status is TaskStatus.FAILED
    assert failed.errors == ["tool exploded"]
    assert failed.metadata["last_error"] == "tool exploded"
    assert any(e.topic == "task.failed" for e in await env.events.history())


# ----------------------------------------------------------------------
# Worker execution, retries, timeouts, cancellation, dead-letter
# ----------------------------------------------------------------------
async def test_run_success_wraps_result_and_respects_complete(env):
    task = await env.engine.create("verified work")

    async def worker(task_record, engine):
        await engine.progress(task_record.id, "step one", step=1)
        await engine.checkpoint(task_record.id, {"stage": "one"})
        await engine.complete(task_record.id, {"summary": "verified"}, [{"kind": "check"}])
        return "worker-return"

    result = await env.engine.run(task.id, worker)
    assert result.status is TaskStatus.COMPLETED
    assert result.result == {"summary": "verified", "_result": "worker-return"}
    assert result.evidence == [{"kind": "check"}]
    assert len(result.checkpoints) == 1


async def test_run_supports_single_argument_workers(env):
    task = await env.engine.create("single argument worker")

    async def worker(task_record):
        assert task_record.id == task.id
        return "one-arg"

    result = await env.engine.run(task.id, worker)
    assert result.status is TaskStatus.FAILED
    assert result.result["_result"] == "one-arg"


async def test_run_without_evidence_is_failed_and_result_wrapped(env):
    task = await env.engine.create("unverified work")

    async def worker(task_record, engine):
        return 42

    result = await env.engine.run(task.id, worker)
    assert result.status is TaskStatus.FAILED
    assert result.result["_result"] == 42
    assert any("evidence" in error for error in result.errors)
    assert any(h.get("to") == "verifying" for h in result.history)


async def test_bounded_retries_then_success(env):
    calls = 0

    async def flaky(task_record, engine):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient")
        await engine.complete(task_record.id, {"attempts": calls}, [{"kind": "retry-ok"}])
        return calls

    task = await env.engine.create("flaky", resource_limits={"max_retries": 2})
    result = await env.engine.run(task.id, flaky)
    assert calls == 3
    assert result.status is TaskStatus.COMPLETED
    assert result.retry_count == 2
    assert len([e for e in await env.events.history("task.retry")]) == 2


async def test_repeated_failure_dead_letters(env):
    calls = 0

    async def always_fails(task_record, engine):
        nonlocal calls
        calls += 1
        raise ValueError("broken")

    task = await env.engine.create("broken", resource_limits={"max_retries": 2})
    result = await env.engine.run(task.id, always_fails)
    assert calls == 3
    assert result.status is TaskStatus.FAILED
    assert result.retry_count == 2
    assert result.metadata["dead_letter"] is True
    assert len(result.errors) == 3
    assert any(h["event"] == "dead_letter" for h in result.history)
    assert any(e.topic == "task.dead_letter" for e in await env.events.history())


async def test_timeout_is_recorded(env):
    async def slow(task_record, engine):
        await asyncio.sleep(0.2)

    task = await env.engine.create("slow", resource_limits={"timeout_seconds": 0.01})
    result = await env.engine.run(task.id, slow)
    assert result.status is TaskStatus.FAILED
    assert any("timed out" in error for error in result.errors)


async def test_cancellation_is_observed_by_running_worker(env):
    started = asyncio.Event()

    async def long_worker(task_record, engine):
        started.set()
        while not (await engine.get(task_record.id)).cancellation_requested:
            await asyncio.sleep(0.005)
        await engine.transition(task_record.id, TaskStatus.CANCELLED, note="worker stopped")
        return "stopped"

    async def canceller():
        await started.wait()
        await env.engine.cancel(task.id)

    task = await env.engine.create("long running")
    canceller_task = asyncio.create_task(canceller())
    result = await env.engine.run(task.id, long_worker)
    await canceller_task
    assert result.status is TaskStatus.CANCELLED
    assert result.cancellation_requested is True
    assert result.result["_result"] == "stopped"
    assert any(e.topic == "task.cancellation_requested" for e in await env.events.history())


async def test_cancellation_before_run_skips_worker(env):
    called = False

    async def worker(task_record, engine):
        nonlocal called
        called = True

    task = await env.engine.create("cancel me")
    await env.engine.cancel(task.id)
    result = await env.engine.run(task.id, worker)
    assert called is False
    assert result.status is TaskStatus.CANCELLED


async def test_idempotency_dedupe(env):
    first = await env.engine.create("original", idempotency_key="stable-key")
    duplicate = await env.engine.create("different objective", idempotency_key="stable-key")
    assert first.id == duplicate.id
    assert duplicate.objective == "original"
    assert len(await env.engine.list()) == 1
    assert first.metadata["idempotency_key"] == "stable-key"


# ----------------------------------------------------------------------
# Recovery after a process restart
# ----------------------------------------------------------------------
async def test_recovery_marks_clean_tasks_ready_and_unsafe_tasks_blocked(env):
    clean = await _running(env, "clean")
    unsafe = await _running(env, "unsafe", idempotency_key="external-action")
    await env.engine.checkpoint(unsafe.id, {"sent": "email"})
    waiting = await env.engine.create("waiting task")
    await env.engine.transition(waiting.id, TaskStatus.READY)
    await env.engine.transition(waiting.id, TaskStatus.RUNNING)
    await env.engine.transition(waiting.id, TaskStatus.WAITING)

    # Simulate a restart: a brand-new engine over the same database.
    restarted = DurableTaskEngine(env.settings, EventBus())
    recovered = await restarted.recover_interrupted()
    by_id = {task.id: task for task in recovered}
    assert set(by_id) == {clean.id, unsafe.id, waiting.id}

    assert by_id[clean.id].status is TaskStatus.READY
    assert by_id[waiting.id].status is TaskStatus.READY
    assert by_id[unsafe.id].status is TaskStatus.BLOCKED
    assert "idempotency" in by_id[unsafe.id].metadata["recovery_reason"]
    for task in recovered:
        assert any(h["event"] == "recovered" for h in task.history)
        assert "replay_performed" not in task.metadata
    for task in recovered:
        assert (await restarted.get(task.id)).status is not TaskStatus.COMPLETED
    assert any(e.topic == "task.recovered" for e in await env.events.history())


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------
async def test_scheduler_due_jobs_and_persistence(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    start = datetime.now(timezone.utc)
    schedule = await scheduler.register_interval("heartbeat", 60, "send heartbeat", {"channel": "ops"})
    assert schedule["kind"] == "interval"
    assert schedule["enabled"] is True

    assert await scheduler.tick(now=start + timedelta(seconds=59)) == []
    created = await scheduler.tick(now=start + timedelta(seconds=61))
    assert len(created) == 1
    assert created[0].objective == "send heartbeat"
    assert created[0].metadata["schedule"] == "heartbeat"
    assert await scheduler.tick(now=start + timedelta(seconds=61)) == []
    assert len(await env.engine.list()) == 1

    updated = await scheduler.get("heartbeat")
    assert datetime.fromisoformat(updated["next_run_at"]) > start + timedelta(seconds=61)
    assert updated["last_run_at"] is not None

    await scheduler.disable("heartbeat")
    assert (await scheduler.get("heartbeat"))["enabled"] is False
    assert await scheduler.tick(now=start + timedelta(days=1)) == []
    await scheduler.enable("heartbeat")
    assert (await scheduler.get("heartbeat"))["enabled"] is True


async def test_scheduler_cron_and_event_triggers(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_cron("five-min", "*/5 * * * *", "cron work")
    schedule = await scheduler.get("five-min")
    due_at = datetime.fromisoformat(schedule["next_run_at"]) + timedelta(seconds=1)
    created = await scheduler.tick(now=due_at)
    assert any(task.metadata["schedule"] == "five-min" for task in created)

    every = await scheduler.register_cron("every-minute", "@every 60s", "@every work")
    assert every["expression"] == "@every 60s"

    scheduler.register_event("sensor.alert", "handle sensor alert")
    await env.events.emit("sensor.alert", value=7)
    tasks = await env.engine.list()
    triggered = [task for task in tasks if task.metadata.get("event_topic") == "sensor.alert"]
    assert len(triggered) == 1
    assert triggered[0].objective == "handle sensor alert"


# ----------------------------------------------------------------------
# Webhooks
# ----------------------------------------------------------------------
async def test_webhook_signature_verification_and_replay_rejection(env, monkeypatch):
    monkeypatch.setenv("TEST_WEBHOOK_SECRET", "super-secret-value")
    handler = WebhookHandler(env.engine, env.events, secret_env="TEST_WEBHOOK_SECRET")
    body = json.dumps({"delivery_id": "delivery-1", "objective": "process payment"}).encode()
    headers = {
        "X-Hub-Signature-256": WebhookHandler.signature(body, "super-secret-value"),
        "X-Delivery-ID": "delivery-1",
        "X-Webhook-Event": "payment.created",
    }
    accepted = await handler.handle({"body": body, "headers": headers})
    assert accepted["status"] == "accepted"
    assert accepted["delivery_id"] == "delivery-1"
    task = await env.engine.get(accepted["task_id"])
    assert task.objective == "process payment"
    assert task.metadata["delivery_id"] == "delivery-1"
    assert "super-secret-value" not in json.dumps(task.model_dump())

    with pytest.raises(DuplicateDelivery):
        await handler.handle({"body": body, "headers": headers})

    bad_headers = dict(headers)
    bad_headers["X-Hub-Signature-256"] = "sha256=" + "0" * 64
    with pytest.raises(InvalidSignature):
        await handler.handle({"body": body, "headers": bad_headers})
    assert len(await env.engine.list()) == 1
    assert any(e.topic == "webhook.accepted" for e in await env.events.history())


async def test_webhook_requires_secret_and_dedupes_by_delivery_id(env, monkeypatch):
    monkeypatch.delenv("MISSING_WEBHOOK_SECRET", raising=False)
    handler = WebhookHandler(env.engine, env.events, secret_env="MISSING_WEBHOOK_SECRET")
    body = b'{"objective":"x"}'
    with pytest.raises(Exception, match="not set"):
        await handler.handle({"body": body, "headers": {}})

    monkeypatch.setenv("MISSING_WEBHOOK_SECRET", "key")
    valid_headers = {"X-Signature": WebhookHandler.signature(body, "key"), "X-Delivery-ID": "d-2"}
    first = await handler.handle({"body": body, "headers": valid_headers})
    second = await env.engine.create("unrelated", idempotency_key="webhook:d-2")
    assert second.id == first["task_id"]


async def test_transition_cannot_complete_without_evidence(env):
    task = await _running(env, "no-evidence completion")
    with pytest.raises(ValueError, match="evidence"):
        await env.engine.transition(task.id, TaskStatus.COMPLETED)
    assert (await env.engine.get(task.id)).status is TaskStatus.RUNNING


async def test_cancelled_task_cannot_be_completed(env):
    task = await _running(env, "cancel then complete")
    assert await env.engine.cancel(task.id)
    with pytest.raises(ValueError, match="cancellation"):
        await env.engine.complete(task.id, {"ok": True}, [{"source": "test", "claim": "ok"}])
    current = await env.engine.get(task.id)
    # Running tasks stay RUNNING until the worker observes the cancellation flag.
    assert current.cancellation_requested


async def test_webhook_replay_with_new_delivery_id_is_rejected(env, monkeypatch):
    monkeypatch.setenv("TEST_WEBHOOK_SECRET", "replay-secret")
    handler = WebhookHandler(env.engine, env.events, secret_env="TEST_WEBHOOK_SECRET")
    body = json.dumps({"objective": "do work"}).encode()
    sig = WebhookHandler.signature(body, "replay-secret")
    headers = {"X-Signature": sig, "X-Delivery-ID": "original"}
    assert (await handler.handle({"body": body, "headers": headers}))["status"] == "accepted"
    replay = dict(headers, **{"X-Delivery-ID": "forged-new-id"})
    with pytest.raises(DuplicateDelivery):
        await handler.handle({"body": body, "headers": replay})
    assert len(await env.engine.list()) == 1

    stale = dict(headers, **{"X-Webhook-Request-Timestamp": "0"})
    fresh_body = json.dumps({"objective": "another task"}).encode()
    stale_headers = {"X-Signature": WebhookHandler.signature(fresh_body, "replay-secret"),
                     "X-Delivery-ID": "fresh", "X-Webhook-Request-Timestamp": "0"}
    with pytest.raises(Exception, match="timestamp"):
        await handler.handle({"body": fresh_body, "headers": stale_headers})


# ----------------------------------------------------------------------
# Atomic claims, idempotent creation and conflict detection (2026-11-18)
# ----------------------------------------------------------------------
async def test_second_run_while_running_does_not_double_execute(env):
    calls = 0
    release = asyncio.Event()

    async def worker(task_record, engine):
        nonlocal calls
        calls += 1
        await release.wait()
        await engine.complete(task_record.id, "done", [{"kind": "manual"}])
        return "worker-return"

    task = await env.engine.create("guarded execution")
    first = asyncio.create_task(env.engine.run(task.id, worker))
    # Wait until the first runner's worker has actually started; only the
    # claim holder ever reaches the worker, so calls==1 proves the claim.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while calls == 0 and loop.time() < deadline:
        await asyncio.sleep(0.005)
    assert calls == 1
    current = await env.engine.run(task.id, worker)  # second claim must lose
    assert calls == 1
    assert current.status is TaskStatus.RUNNING
    release.set()
    result = await first
    assert result.status is TaskStatus.COMPLETED
    assert calls == 1


async def test_concurrent_run_claims_execute_worker_once(env):
    calls = 0

    async def worker(task_record, engine):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        await engine.complete(task_record.id, {"n": calls}, [{"kind": "count"}])
        return calls

    task = await env.engine.create("single execution under concurrency")
    results = await asyncio.gather(
        env.engine.run(task.id, worker),
        env.engine.run(task.id, worker),
    )
    assert calls == 1
    assert any(result.status is TaskStatus.COMPLETED for result in results)


async def test_concurrent_idempotent_create_yields_one_task(env):
    async def make():
        return await env.engine.create("same objective", idempotency_key="same-key")

    first, second = await asyncio.gather(make(), make())
    assert first.id == second.id
    assert len(await env.engine.list()) == 1


async def test_worker_calling_fail_then_returning_is_safe(env):
    """F-18: a worker that fails the task itself must not crash run()."""

    async def worker(task_record, engine):
        await engine.fail(task_record.id, "self-reported failure")
        return "already failed"

    task = await env.engine.create("self fail")
    result = await env.engine.run(task.id, worker)
    assert result.status is TaskStatus.FAILED
    assert "self-reported failure" in result.errors


async def test_recovery_includes_verifying_and_planning(env):
    """F-17: tasks stranded mid-verification or planning are recovered too."""
    verifying = await _running(env, "stranded verifying")
    await env.engine.transition(verifying.id, TaskStatus.VERIFYING)
    planning = await env.engine.create("stranded planning")
    await env.engine.transition(planning.id, TaskStatus.PLANNING)

    restarted = DurableTaskEngine(env.settings, EventBus())
    recovered = await restarted.recover_interrupted()
    by_id = {task.id: task for task in recovered}
    assert set(by_id) == {verifying.id, planning.id}
    assert by_id[verifying.id].status is TaskStatus.READY
    assert by_id[planning.id].status is TaskStatus.READY


async def test_conflicting_save_raises_instead_of_silent_loss(env):
    task = await env.engine.create("conflict detection")
    stale = await env.engine._load(task.id)
    fresh = await env.engine._load(task.id)
    fresh.metadata["note"] = "written by fresh writer"
    await env.engine._save(fresh)
    stale.metadata["note"] = "written by stale writer"
    with pytest.raises(TaskConflictError):
        await env.engine._save(stale)
    stored = await env.engine.get(task.id)
    assert stored.metadata["note"] == "written by fresh writer"


# ----------------------------------------------------------------------
# v1 webhook signatures: timestamp, delivery ID and body in one MAC
# ----------------------------------------------------------------------
async def test_v1_signature_binds_timestamp_and_delivery_id(env, monkeypatch):
    monkeypatch.setenv("TEST_WEBHOOK_SECRET", "v1-secret-value")
    handler = WebhookHandler(env.engine, env.events, secret_env="TEST_WEBHOOK_SECRET")
    body = json.dumps({"objective": "v1 delivery", "delivery_id": "d-v1-1"}).encode()
    signed = WebhookHandler.sign_v1(body, "v1-secret-value", delivery_id="d-v1-1")
    accepted = await handler.handle({"body": body, "headers": signed})
    assert accepted["status"] == "accepted"
    assert accepted["delivery_id"] == "d-v1-1"

    # Freshness forgery: replay the captured request with a different (still
    # in-window) timestamp; the MAC over t=... must reject it.
    forged_ts = dict(signed, **{"X-Webhook-Request-Timestamp": str(int(time.time()) + 60)})
    with pytest.raises(InvalidSignature):
        await handler.handle({"body": body, "headers": forged_ts})

    # Delivery-ID forgery: keep the signature, change the delivery identity.
    forged_id = dict(signed, **{"X-Delivery-ID": "future-legitimate-id"})
    with pytest.raises(InvalidSignature):
        await handler.handle({"body": body, "headers": forged_id})

    # An exact byte replay is still a duplicate.
    with pytest.raises(DuplicateDelivery):
        await handler.handle({"body": body, "headers": signed})

    # A properly signed, distinct delivery of the same body is a new task.
    second = WebhookHandler.sign_v1(body, "v1-secret-value", delivery_id="d-v1-2")
    again = await handler.handle({"body": body, "headers": second})
    assert again["delivery_id"] == "d-v1-2"
    assert len(await env.engine.list()) == 2


async def test_v1_signature_requires_timestamp_and_window(env, monkeypatch):
    monkeypatch.setenv("TEST_WEBHOOK_SECRET", "v1-secret-value")
    handler = WebhookHandler(env.engine, env.events, secret_env="TEST_WEBHOOK_SECRET")
    body = b'{"objective":"timestamp required"}'
    signed = WebhookHandler.sign_v1(body, "v1-secret-value", delivery_id="d-ts")
    stripped = {k: v for k, v in signed.items() if k != "X-Webhook-Request-Timestamp"}
    with pytest.raises(WebhookError, match="imestamp"):
        await handler.handle({"body": body, "headers": stripped})

    stale = WebhookHandler.sign_v1(
        body, "v1-secret-value", delivery_id="d-stale", timestamp=int(time.time()) - 400
    )
    with pytest.raises(WebhookError, match="timestamp"):
        await handler.handle({"body": body, "headers": stale})


# ----------------------------------------------------------------------
# Durable update(): merged metadata, legal transitions (Batch A)
# ----------------------------------------------------------------------
async def test_durable_update_merges_metadata_and_rejects_illegal_status(env):
    task = await env.engine.create("update me", metadata={"origin": "test"})
    updated = await env.engine.update(
        task.id, result={"partial": True}, metadata={"extra": "merged"},
        status=TaskStatus.READY, note="bookkeeping",
    )
    assert updated.status is TaskStatus.READY
    assert updated.metadata["origin"] == "test"     # merge, never replace
    assert updated.metadata["extra"] == "merged"
    assert updated.completed_at is None             # only complete() stamps it

    with pytest.raises(ValueError, match="illegal task transition") as excinfo:
        await env.engine.update(task.id, status=TaskStatus.COMPLETED)
    assert "completed" in str(excinfo.value)
    # A rejected completion must not mutate the row on its way out.
    unchanged = await env.engine.get(task.id)
    assert unchanged.status is TaskStatus.READY
    assert unchanged.evidence == []
    assert unchanged.completed_at is None
    assert unchanged.result == {"partial": True}
    done = await env.engine.complete(task.id, "ok", [{"kind": "check"}])
    assert done.status is TaskStatus.COMPLETED


async def test_fallback_cancel_never_flips_terminal_tasks(tmp_path, monkeypatch):
    from leo.agents.engine import _DurableTasks

    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "fallback.db")
    await db.initialize()
    tasks = _DurableTasks()

    async def worker(task):
        return {"success": True, "evidence": [{"claim": "done"}]}

    task = await tasks.create("complete me")
    finished = await tasks.run(task.id, worker)
    assert finished.status is TaskStatus.COMPLETED
    assert await tasks.cancel(task.id) is False     # terminal stays terminal
    assert (await tasks.get(task.id)).status is TaskStatus.COMPLETED

    live = await tasks.create("cancel me")
    assert await tasks.cancel(live.id) is True
    reloaded = await tasks.get(live.id)
    assert reloaded.cancellation_requested is True  # the flag now persists
    assert reloaded.status is TaskStatus.CANCELLED


# ----------------------------------------------------------------------
# Scheduler fixes: F-15 cron steps, F-16 catch-up, F-23 concurrency gate
# ----------------------------------------------------------------------
def test_cron_scalar_step_expands_to_field_bounds():
    from leo.tasks.scheduler import parse_cron

    assert parse_cron("5/10 * * * *")[0] == {5, 15, 25, 35, 45, 55}
    assert parse_cron("0/15 * * * *")[0] == {0, 15, 30, 45}
    assert parse_cron("*/15 * * * *")[0] == {0, 15, 30, 45}
    assert parse_cron("5 * * * *")[0] == {5}  # no step: plain scalar


async def test_scheduler_catches_up_missed_occurrences_without_drift(env):
    env.settings.automation.max_concurrent_tasks = 0
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_cron("catchup", "0,30 * * * *", "quarterly work")
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?",
        ("2026-11-18T09:00:00+00:00", "catchup"),
    )
    due = await scheduler.tick(now="2026-11-18T09:31:00+00:00")
    assert len(due) == 2
    assert {task.metadata["scheduled_for"] for task in due} == {
        "2026-11-18T09:00:00+00:00", "2026-11-18T09:30:00+00:00",
    }
    # Anchored to the last occurrence, not to the late tick moment (F-16).
    schedule = await scheduler.get("catchup")
    assert schedule["next_run_at"] == "2026-11-18T10:00:00+00:00"


async def test_scheduler_catchup_is_bounded(env):
    env.settings.automation.max_concurrent_tasks = 0
    scheduler = Scheduler(env.engine, env.settings, env.events)
    await scheduler.register_cron("minute", "* * * * *", "minute work")
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name=?",
        ("2026-11-18T09:00:00+00:00", "minute"),
    )
    due = await scheduler.tick(now="2026-11-19T10:00:00+00:00")  # 25 hours late
    assert len(due) == 10
    assert any(e.topic == "schedule.catchup_truncated" for e in await env.events.history())
    schedule = await scheduler.get("minute")
    assert schedule["next_run_at"] > "2026-11-19T10:00:00+00:00"


async def test_scheduler_max_concurrent_tasks_defers_bursts(env):
    env.settings.automation.max_concurrent_tasks = 2
    scheduler = Scheduler(env.engine, env.settings, env.events)
    for name in ("a", "b", "c"):
        await scheduler.register_interval(name, 60, f"work {name}")
        await db.execute(
            "UPDATE schedules SET next_run_at=? WHERE name=?",
            ("2026-11-18T09:00:00+00:00", name),
        )
    due = await scheduler.tick(now="2026-11-18T09:00:30+00:00")
    assert len(due) == 2
    deferred = [e for e in await env.events.history("schedule.deferred")]
    assert len(deferred) == 1
    # The deferred occurrence keeps its due time so the next tick retries.
    assert (await scheduler.get("c"))["next_run_at"] == "2026-11-18T09:00:00+00:00"

    env.settings.automation.max_concurrent_tasks = 0
    again = await scheduler.tick(now="2026-11-18T09:00:30+00:00")
    assert len(again) == 1


# ----------------------------------------------------------------------
# Agent 05 hardening regressions (2026-11-19)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("via", ["running", "verifying"])
async def test_update_cannot_complete_without_evidence(env, via):
    """update() is a bookkeeping API, not a second completion path."""
    task = await env.engine.create("no evidence via update")
    await env.engine.transition(task.id, TaskStatus.PLANNING)
    await env.engine.transition(task.id, TaskStatus.READY)
    await env.engine.transition(task.id, TaskStatus.RUNNING)
    if via == "verifying":
        await env.engine.transition(task.id, TaskStatus.VERIFYING)

    with pytest.raises(ValueError, match="evidence"):
        await env.engine.update(task.id, status=TaskStatus.COMPLETED)
    current = await env.engine.get(task.id)
    assert current.status.value == via
    assert current.evidence == []
    assert current.completed_at is None

    done = await env.engine.complete(task.id, "ok", [{"kind": "check"}])
    assert done.status is TaskStatus.COMPLETED
    assert done.evidence == [{"kind": "check"}]


async def test_record_failure_after_completion_is_noop(env):
    task = await env.engine.create("late failure")
    await env.engine.complete(task.id, "done", [{"kind": "test"}])
    result = await env.engine._record_failure(task.id, "late timeout")
    assert result.status is TaskStatus.COMPLETED
    assert result.errors == []
    assert result.retry_count == 0
    assert result.result == "done"


async def test_timeout_after_worker_completes_does_not_fail_task(env):
    gate = asyncio.Event()

    async def worker(task_record, engine):
        await engine.complete(task_record.id, "done", [{"kind": "check"}])
        await gate.wait()

    task = await env.engine.create(
        "complete then stall", resource_limits={"timeout_seconds": 0.01}
    )
    try:
        result = await env.engine.run(task.id, worker)
    finally:
        gate.set()
    assert result.status is TaskStatus.COMPLETED
    assert result.errors == []
    assert result.result["value"] == "done"
    assert result.evidence == [{"kind": "check"}]


async def test_recovery_skips_concurrently_modified_task(env, monkeypatch):
    first = await _running(env, "first stranded")
    second = await _running(env, "second stranded")
    original_save = env.engine._save
    bumped = [False]

    async def conflicting_save(task):
        if task.id == first.id and not bumped[0]:
            bumped[0] = True
            # Simulate another writer saving between recovery's SELECT and save.
            await db.execute("UPDATE tasks SET version=version+1 WHERE id=?", (task.id,))
        return await original_save(task)

    monkeypatch.setattr(env.engine, "_save", conflicting_save)
    recovered = await env.engine.recover_interrupted()
    assert [task.id for task in recovered] == [second.id]
    assert (await env.engine.get(first.id)).status is TaskStatus.RUNNING
    assert (await env.engine.get(second.id)).status is TaskStatus.READY


def test_cron_accepts_month_and_weekday_names():
    """'mon-fri' used to die in int(); names are accepted case-insensitively."""
    assert parse_cron("0 9 * * mon-fri")[4] == {1, 2, 3, 4, 5}
    assert parse_cron("0 9 * * MON-FRI")[4] == {1, 2, 3, 4, 5}
    assert parse_cron("0 0 * * sun")[4] == {0}
    assert parse_cron("0 0 * * fri,1")[4] == {1, 5}
    assert parse_cron("0 0 1 jan,apr *")[3] == {1, 4}
    # Numbers and the 7==Sunday alias must keep working alongside names.
    assert parse_cron("0 0 * * 7")[4] == {0}
    assert parse_cron("0 0 * * 0-7")[4] == set(range(7))
    with pytest.raises(ValueError, match="invalid cron field value"):
        parse_cron("0 9 * * xyz")


def test_cron_weekday_name_range_schedules_next_workday():
    after = datetime(2026, 11, 28, 12, tzinfo=timezone.utc)  # Saturday
    assert next_run("cron", "0 9 * * mon-fri", after) == datetime(
        2026, 11, 30, 9, tzinfo=timezone.utc
    )


def test_every_accepts_day_and_week_units():
    assert every_interval("@every 1d") == timedelta(days=1)
    assert every_interval("@every 1w") == timedelta(days=7)
    assert every_interval("@every 2 days") == timedelta(days=2)
    # The minute/hour units must not be shadowed by the new day/week ones.
    assert every_interval("@every 90m") == timedelta(minutes=90)
    assert every_interval("@every 3h") == timedelta(hours=3)
    assert every_interval("@every 30s") == timedelta(seconds=30)
    assert next_run(
        "cron", "@every 1d", datetime(2026, 11, 24, 3, tzinfo=timezone.utc)
    ) == datetime(2026, 11, 25, 3, tzinfo=timezone.utc)


def test_cron_leap_day_is_schedulable():
    """A 366-day minute sweep could never reach Feb 29."""
    assert next_run("cron", "0 0 29 2 *", datetime(2026, 3, 1, tzinfo=timezone.utc)) == datetime(
        2028, 2, 29, tzinfo=timezone.utc
    )
    # Strictly after: an occurrence exactly at `after` rolls to the next cycle.
    assert next_run("cron", "0 0 29 2 *", datetime(2028, 2, 29, tzinfo=timezone.utc)) == datetime(
        2032, 2, 29, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError, match="no matching time found"):
        next_run("cron", "0 0 30 2 *", datetime(2026, 3, 1, tzinfo=timezone.utc))


def test_cron_next_run_unchanged_for_ordinary_expressions():
    base = datetime(2026, 11, 24, 3, 20, tzinfo=timezone.utc)
    assert next_run("cron", "0 * * * *", base) == datetime(2026, 11, 24, 4, tzinfo=timezone.utc)
    assert next_run("cron", "*/15 * * * *", base) == datetime(2026, 11, 24, 3, 30, tzinfo=timezone.utc)
    assert next_run("cron", "0 4 * * *", base) == datetime(2026, 11, 24, 4, tzinfo=timezone.utc)


def test_cron_sunday_alias_and_step():
    assert parse_cron("0 0 * * 7")[4] == {0}
    assert parse_cron("0 0 * * 7/2")[4] == {0, 2, 4, 6}
    assert parse_cron("0 0 * * 0-7")[4] == set(range(7))
    after = datetime(2026, 11, 19, tzinfo=timezone.utc)  # Thursday
    assert next_run("cron", "0 0 * * 7", after) == datetime(2026, 11, 22, tzinfo=timezone.utc)


def test_cron_rejects_out_of_range_fields():
    for expression in (
        "60 * * * *",
        "0,60 * * * *",
        "* 24 * * *",
        "0 0 32 * *",
        "0 0 * 13 *",
        "0 0 * * 8",
    ):
        with pytest.raises(ValueError):
            parse_cron(expression)


def test_non_positive_every_interval_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        every_interval("@every 0s")
    with pytest.raises(ValueError, match="positive"):
        next_run("interval", "@every 0s", datetime(2026, 11, 19, tzinfo=timezone.utc))


async def test_register_rejects_non_positive_and_submicrosecond_intervals(env):
    scheduler = Scheduler(env.engine, env.settings, env.events)
    with pytest.raises(ValueError, match="positive"):
        await scheduler.register_cron("zero", "@every 0s", "zero work")
    with pytest.raises(ValueError, match="positive"):
        await scheduler.register_interval("zero-num", 0, "zero work")
    with pytest.raises(ValueError, match="microsecond"):
        await scheduler.register_interval("submicro", 1e-9, "tiny work")
    assert await scheduler.list_schedules() == []


def test_advance_past_due_uses_arithmetic_for_short_periods():
    start = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)
    for expression in ("0.001", "@every 0.001s"):
        cursor, skipped = _advance_past_due("interval", expression, start, end)
        assert cursor > end
        assert skipped == 3_600_001
    cursor, skipped = _advance_past_due("cron", "0,30 * * * *", start, end)
    # next_run() is strictly after, so the occurrence exactly at `end` is
    # skipped too (09:00, 09:30, 10:00 -> next is 10:30).
    assert cursor == end + timedelta(minutes=30)
    assert skipped == 3


async def test_scheduler_triggered_task_count_is_per_schedule(env):
    env.settings.automation.max_concurrent_tasks = 0
    scheduler = Scheduler(env.engine, env.settings, env.events)
    schedules = {
        "a": "2026-11-18T09:00:00+00:00",
        "b": "2026-11-18T08:59:00+00:00",
        "c": "2026-11-18T09:00:00+00:00",
    }
    for name, due in schedules.items():
        await scheduler.register_interval(name, 60, f"work {name}")
        await db.execute("UPDATE schedules SET next_run_at=? WHERE name=?", (due, name))

    await scheduler.tick(now="2026-11-18T09:00:30+00:00")
    counts = {
        event.payload["name"]: event.payload["task_count"]
        for event in await env.events.history("schedule.triggered")
    }
    assert counts == {"a": 1, "b": 2, "c": 1}


# ----------------------------------------------------------------------
# Live worker registry, dispatch, delivery and claim leases (2026-11-19)
# ----------------------------------------------------------------------
async def test_dispatch_ready_executes_once_and_delivers(env):
    calls: list[Any] = []
    delivered: list[tuple[str, Any]] = []

    async def worker(task_record, engine, context):
        calls.append(context["notify_targets"])
        await engine.complete(task_record.id, {"ok": True}, [{"source": "worker", "claim": "done"}])
        return "worker-return"

    async def deliver(task_record, channel, target, context):
        delivered.append((channel, target))

    env.engine.register_worker("planner", worker)
    env.engine.register_delivery("ops", deliver)
    task = await env.engine.create(
        "dispatch me", owner_agent="planner", metadata={"notify_targets": {"ops": "room-1"}}
    )

    first = await env.engine.dispatch_ready()
    second = await env.engine.dispatch_ready()
    assert [item.id for item in first] == [task.id]
    assert second == []
    assert calls == [{"ops": "room-1"}]
    assert delivered == [("ops", "room-1")]

    finished = await env.engine.get(task.id)
    assert finished.status is TaskStatus.COMPLETED
    assert finished.result == {"ok": True, "_result": "worker-return"}
    assert finished.metadata["task.delivery"]["ops"]["status"] == "sent"
    assert len(await env.engine.checkpoints(task.id)) == 0
    assert any(e.topic == "task.delivered" for e in await env.events.history())


async def test_dispatch_ready_handles_schedule_and_notify_payloads(env):
    seen: list[tuple[str, Any]] = []

    async def schedule_worker(task_record, engine, context):
        seen.append(("schedule", context["schedule_payload"]))
        await engine.complete(task_record.id, "tick", [{"source": "schedule", "claim": "due"}])

    async def notify_worker(task_record, engine):
        seen.append(("notify", None))
        await engine.complete(task_record.id, "notice", [{"source": "notify", "claim": "sent"}])

    async def deliver(task_record, channel, target, context):
        seen.append((channel, target))

    scheduled = await env.engine.create("cron work", metadata={"schedule_payload": {"channel": "ops"}})
    notified = await env.engine.create("notice", metadata={"notify_targets": ["console"]})
    env.engine.register_worker("schedule", schedule_worker)
    env.engine.register_worker("notify", notify_worker)
    env.engine.register_delivery("ops", deliver)
    env.engine.register_default_delivery(deliver)

    dispatched = await env.engine.dispatch_ready()
    assert {item.id for item in dispatched} == {scheduled.id, notified.id}
    assert {kind for kind, _ in seen if kind in ("schedule", "notify")} == {"schedule", "notify"}
    assert ("ops", None) in seen
    assert ("console", None) in seen
    for task_id in (scheduled.id, notified.id):
        assert (await env.engine.get(task_id)).status is TaskStatus.COMPLETED


async def test_dispatch_skips_unregistered_owner_cancelled_and_terminal(env):
    calls = 0

    async def worker(task_record, engine):
        nonlocal calls
        calls += 1
        await engine.complete(task_record.id, "ok", [{"source": "test", "claim": "done"}])

    waiting = await env.engine.create("no worker yet", owner_agent="late")
    assert await env.engine.dispatch_ready() == []
    assert (await env.engine.get(waiting.id)).status is TaskStatus.CREATED

    cancelled = await env.engine.create("cancelled", owner_agent="late")
    await env.engine.cancel(cancelled.id)
    done = await env.engine.create("done", owner_agent="late")
    await env.engine.complete(done.id, "ok", [{"source": "test", "claim": "done"}])

    env.engine.register_worker("late", worker)
    dispatched = await env.engine.dispatch_ready()
    assert [item.id for item in dispatched] == [waiting.id]
    assert calls == 1
    assert (await env.engine.get(cancelled.id)).status is TaskStatus.CANCELLED
    assert (await env.engine.get(done.id)).status is TaskStatus.COMPLETED


async def test_concurrent_dispatch_ready_executes_worker_once(env):
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(task_record, engine):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        await engine.complete(task_record.id, "ok", [{"source": "test", "claim": "done"}])

    env.engine.register_worker("parallel", worker)
    task = await env.engine.create("parallel work", owner_agent="parallel")
    first = asyncio.create_task(env.engine.dispatch_ready())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    second = await env.engine.dispatch_ready()
    assert calls == 1
    assert second == []
    release.set()
    dispatched = await first
    assert [item.id for item in dispatched] == [task.id]
    assert calls == 1
    assert (await env.engine.get(task.id)).status is TaskStatus.COMPLETED


async def test_recovery_finalizes_cancelled_tasks_and_claim_refuses_them(env):
    flagged = await _running(env, "cancelled while running")
    await db.execute(
        "UPDATE tasks SET cancellation_requested=1 WHERE id=?", (flagged.id,)
    )
    assert await env.engine._claim(flagged.id, "late claim") is None
    stored = await env.engine.get(flagged.id)
    assert stored.status is TaskStatus.CANCELLED
    assert stored.completed_at is not None

    stranded = await _running(env, "stranded cancellation")
    await db.execute(
        "UPDATE tasks SET cancellation_requested=1 WHERE id=?", (stranded.id,)
    )
    finalized = await _running(env, "cancelled without completion stamp")
    await db.execute(
        "UPDATE tasks SET status='cancelled', completed_at=NULL WHERE id=?", (finalized.id,)
    )

    restarted = DurableTaskEngine(env.settings, EventBus())
    recovered = await restarted.recover_interrupted()
    by_id = {task.id: task for task in recovered}
    assert {stranded.id, finalized.id} <= set(by_id)
    for task_id in (stranded.id, finalized.id):
        assert by_id[task_id].status is TaskStatus.CANCELLED
        assert by_id[task_id].completed_at is not None
        assert "cancellation" in by_id[task_id].metadata["recovery_reason"]
        assert await restarted._claim(task_id, "retry") is None
    assert flagged.id not in by_id
    assert await restarted.dispatch_ready() == []


async def test_recovery_blocks_when_evidence_result_or_delivery_exists(env):
    with_evidence = await _running(env, "evidence signal")
    with_result = await _running(env, "result signal")
    with_delivery = await _running(env, "delivery signal")
    await db.execute(
        "UPDATE tasks SET evidence=? WHERE id=?",
        (json.dumps([{"source": "test", "claim": "external side effect"}]), with_evidence.id),
    )
    await db.execute(
        "UPDATE tasks SET result=? WHERE id=?", (json.dumps({"sent": True}), with_result.id)
    )
    await db.execute(
        "UPDATE tasks SET metadata=? WHERE id=?",
        (json.dumps({"task.delivery": {"ops": {"status": "sent"}}}), with_delivery.id),
    )

    restarted = DurableTaskEngine(env.settings, EventBus())
    by_id = {task.id: task for task in await restarted.recover_interrupted()}
    assert {with_evidence.id, with_result.id, with_delivery.id} <= set(by_id)
    for task_id, signal in (
        (with_evidence.id, "evidence"),
        (with_result.id, "result"),
        (with_delivery.id, "delivery"),
    ):
        assert by_id[task_id].status is TaskStatus.BLOCKED
        assert signal in by_id[task_id].metadata["recovery_reason"]
        assert (await restarted.get(task_id)).status is TaskStatus.BLOCKED


async def test_checkpoint_writes_are_atomic_monotonic_and_consistent(env):
    task = await env.engine.create("checkpointed")
    first = await env.engine.checkpoint(task.id, {"step": 1})
    second = await env.engine.checkpoint(task.id, {"step": 2})
    stored = await env.engine.checkpoints(task.id)
    assert [entry["seq"] for entry in stored] == [1, 2]
    assert [entry["state"]["step"] for entry in stored] == [1, 2]
    assert second.checkpoints == stored
    assert first.checkpoints != stored
    assert (await env.engine.get(task.id)).checkpoints == stored

    await db.execute(
        "CREATE TRIGGER reject_checkpoint BEFORE INSERT ON task_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'checkpoint failure'); END"
    )
    try:
        before = await env.engine.get(task.id)
        with pytest.raises(Exception, match="checkpoint failure"):
            await env.engine.checkpoint(task.id, {"step": 3})
        after = await env.engine.get(task.id)
        assert after.checkpoints == before.checkpoints
        assert after.version == before.version
        assert await env.engine.checkpoints(task.id) == stored
    finally:
        await db.execute("DROP TRIGGER reject_checkpoint")


async def test_retries_and_recovery_restore_checkpoint_state(env):
    calls = 0

    async def flaky(task_record, engine):
        nonlocal calls
        calls += 1
        if calls == 1:
            await engine.checkpoint(task_record.id, {"step": 7, "value": "keep"})
            raise RuntimeError("transient")
        assert task_record.current_step == 7
        await engine.complete(task_record.id, "ok", [{"source": "retry", "claim": "done"}])

    task = await env.engine.create("resumable", resource_limits={"max_retries": 1})
    result = await env.engine.run(task.id, flaky)
    assert result.status is TaskStatus.COMPLETED
    assert calls == 2
    assert result.current_step == 7
    assert any(h["event"] == "resumed_from_checkpoint" for h in result.history)

    stranded = await _running(env, "stranded checkpoint")
    await env.engine.checkpoint(stranded.id, {"step": 9})
    restarted = DurableTaskEngine(env.settings, EventBus())
    recovered = await restarted.recover_interrupted()
    by_id = {item.id: item for item in recovered}
    assert by_id[stranded.id].status is TaskStatus.BLOCKED
    assert by_id[stranded.id].current_step == 9
    assert "checkpoints" in by_id[stranded.id].metadata["recovery_reason"]
    assert any(h["event"] == "resumed_from_checkpoint" for h in by_id[stranded.id].history)

    resumed = await restarted.resume_from_checkpoint(stranded.id, seq=1)
    assert resumed is not None
    assert resumed.current_step == 9
    assert (await restarted.get(stranded.id)).status is TaskStatus.BLOCKED
    with pytest.raises(ValueError):
        await restarted.resume_from_checkpoint(task.id)  # completed rows stay absorbing


async def test_heartbeat_protects_active_claims_and_recover_stale_handles_the_rest(env):
    live = await env.engine.create("live claim")
    claim = await env.engine._claim(live.id, "worker started", lease_seconds=1.0)
    assert claim is not None
    token = claim[0].metadata["lease_owner"]
    assert await env.engine.heartbeat(live.id, owner="wrong-owner") is None
    assert await env.engine.heartbeat(live.id, owner=token) is not None

    observer = DurableTaskEngine(env.settings, EventBus())
    assert await observer.recover_stale(lease_seconds=0.1) == []
    assert (await env.engine.get(live.id)).status is TaskStatus.RUNNING

    stale = await env.engine.create("silent worker")
    await env.engine._claim(stale.id, "worker started", lease_seconds=60.0)
    current = await env.engine.get(stale.id)
    metadata = dict(current.metadata)
    metadata["lease_expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    await db.execute(
        "UPDATE tasks SET metadata=? WHERE id=?", (json.dumps(metadata), stale.id)
    )
    env.engine._active_claims.pop(stale.id, None)
    recovered = await observer.recover_stale()
    assert [item.id for item in recovered] == [stale.id]
    assert recovered[0].status is TaskStatus.READY
    assert (await env.engine.get(live.id)).status is TaskStatus.RUNNING


async def test_recover_stale_finalizes_cancellation_and_blocks_results(env):
    cancelled = await _running(env, "stale cancellation")
    with_result = await _running(env, "stale result")
    expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    await db.execute(
        "UPDATE tasks SET cancellation_requested=1, metadata=? WHERE id=?",
        (json.dumps({"lease_expires_at": expired}), cancelled.id),
    )
    await db.execute(
        "UPDATE tasks SET result=?, metadata=? WHERE id=?",
        (json.dumps({"sent": True}), json.dumps({"lease_expires_at": expired}), with_result.id),
    )

    observer = DurableTaskEngine(env.settings, EventBus())
    by_id = {item.id: item for item in await observer.recover_stale()}
    assert by_id[cancelled.id].status is TaskStatus.CANCELLED
    assert by_id[cancelled.id].completed_at is not None
    assert by_id[with_result.id].status is TaskStatus.BLOCKED
    assert "result" in by_id[with_result.id].metadata["recovery_reason"]


async def test_complete_rejects_empty_and_self_attested_evidence(env):
    task = await _running(env, "evidence rules")
    for bad in (
        [{}],
        [{"claim": "done"}],
        [{"source": "test", "verified": False, "claim": "ok"}],
        [{"source": "test", "status": "rejected", "claim": "ok"}],
    ):
        with pytest.raises(ValueError):
            await env.engine.complete(task.id, "ok", bad)
    stored = await env.engine.get(task.id)
    assert stored.status is TaskStatus.RUNNING
    assert stored.evidence == []

    done = await env.engine.complete(task.id, "ok", [{"source": "unit-test", "claim": "passed"}])
    assert done.status is TaskStatus.COMPLETED
    assert done.evidence == [{"source": "unit-test", "claim": "passed"}]


async def test_cancellation_wins_over_failure_and_dead_letter(env):
    failed = await _running(env, "cancel then fail")
    await db.execute("UPDATE tasks SET cancellation_requested=1 WHERE id=?", (failed.id,))
    result = await env.engine.fail(failed.id, "worker exploded")
    assert result.status is TaskStatus.CANCELLED
    assert "dead_letter" not in result.metadata

    dead_letter = await _running(env, "cancel before dead-letter")
    await db.execute(
        "UPDATE tasks SET cancellation_requested=1 WHERE id=?", (dead_letter.id,)
    )
    recorded = await env.engine._record_failure(dead_letter.id, "boom")
    assert recorded.status is TaskStatus.CANCELLED
    assert recorded.retry_count == 0
    assert "dead_letter" not in recorded.metadata


async def test_terminal_and_cancelled_rows_absorb_all_writes(env):
    done = await env.engine.create("finished")
    await env.engine.complete(done.id, "ok", [{"source": "test", "claim": "passed"}])
    cancelled = await env.engine.create("stopped")
    await env.engine.cancel(cancelled.id)

    for task_id in (done.id, cancelled.id):
        with pytest.raises(ValueError):
            await env.engine.progress(task_id, "late progress")
        with pytest.raises(ValueError):
            await env.engine.checkpoint(task_id, {"step": 99})
        with pytest.raises(ValueError):
            await env.engine.update(task_id, metadata={"late": True})
        with pytest.raises(ValueError):
            await env.engine.fail(task_id, "late failure")
        with pytest.raises(ValueError):
            await env.engine.transition(task_id, TaskStatus.READY)

    finished = await env.engine.get(done.id)
    assert finished.result == "ok"
    assert finished.checkpoints == []
    assert finished.errors == []
    assert finished.metadata == {}
    assert (await env.engine.get(cancelled.id)).status is TaskStatus.CANCELLED
