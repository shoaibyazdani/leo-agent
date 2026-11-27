"""Phase 8: delivery receipts are durable, keyed and lane-aware."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import TaskStatus
from leo.events import EventBus
from leo.tasks import DurableTaskEngine
from leo.tasks.engine import _delivery_targets


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    engine = DurableTaskEngine(settings, events)
    return SimpleNamespace(settings=settings, events=events, engine=engine)


async def _finished(env, *, delivery, failure_delivery=None, objective="scheduled work"):
    payload = {"name": "brief", "objective": objective, "delivery": delivery}
    if failure_delivery is not None:
        payload["failure_delivery"] = failure_delivery
    task = await env.engine.create(
        objective,
        metadata={"schedule": "brief", "schedule_id": "sch_r", "schedule_payload": payload},
    )
    await env.engine.transition(task.id, TaskStatus.READY)
    await env.engine.transition(task.id, TaskStatus.RUNNING)
    await env.engine.complete(task.id, {"ok": True}, [{"source": "test", "claim": "done"}])
    return await env.engine.get(task.id)


def _target(platform: str, chat: str, *, kind: str | None = None):
    return {"kind": kind or "external", "platform": platform, "chat_id": chat}


async def test_receipts_keyed_by_platform_chat_thread_and_no_double_send(env) -> None:
    calls: list[tuple[str, str]] = []

    async def deliver(task, channel, target, context):
        calls.append((channel, target))
        return "msg-1"

    env.engine.register_delivery("telegram", deliver)
    env.engine.register_delivery("local", deliver)
    finished = await _finished(env, delivery=[
        _target("telegram", "1"),
        _target("telegram", "1", kind="local"),  # same platform key, different kind
        _target("telegram", "2"),
    ])
    receipts = await env.engine._deliver_result(finished)
    assert len(receipts) == 2
    stored = await env.engine.get(finished.id)
    assert set(stored.metadata["task.delivery"]) == {"telegram:1:", "telegram:2:"}
    assert stored.metadata["task.delivery"]["telegram:1:"]["status"] == "sent"
    first = list(calls)
    await env.engine._deliver_result(stored)
    assert calls == first  # a durable sent receipt prevents a second send


async def test_partial_failure_records_failure_receipt_and_notifies_failure_lane(env) -> None:
    notices: list[tuple[str, str]] = []

    async def broken(task, channel, target, context):
        raise RuntimeError("adapter down")

    async def local(task, channel, target, context):
        notices.append((context.get("delivery_lane"), context.get("delivery_text_override") or ""))
        return "local:note"

    env.engine.register_delivery("telegram", broken)
    env.engine.register_delivery("local", local)
    finished = await _finished(
        env,
        delivery=[_target("telegram", "bad")],
        failure_delivery=[{"kind": "local", "platform": "local", "chat_id": ""}],
    )
    await env.engine._deliver_result(finished)
    stored = await env.engine.get(finished.id)
    receipts = stored.metadata["task.delivery"]
    assert receipts["telegram:bad:"]["status"] == "failed"
    assert "adapter down" in receipts["telegram:bad:"]["detail"]
    assert receipts["failure:local::"]["status"] == "sent"
    assert notices and notices[0][0] == "failure"
    assert "telegram:bad:" in notices[0][1]
    assert stored.metadata["task.delivery_failure_notices"]["telegram:bad:"] == "failure:local::"
    # A replay must not notify the failure lane twice.
    await env.engine._deliver_result(stored)
    assert len(notices) == 1


async def test_nowhere_is_a_durable_honest_receipt(env) -> None:
    called: list[str] = []

    async def deliver(task, channel, target, context):
        called.append(channel)
        return "sent"

    env.engine.register_default_delivery(deliver)
    finished = await _finished(env, delivery=[])
    await env.engine._deliver_result(finished)
    stored = await env.engine.get(finished.id)
    receipt = stored.metadata["task.delivery"]["nowhere"]
    assert receipt["status"] == "nowhere"
    assert receipt["target"] is None
    assert called == []


async def test_legacy_channel_target_projection_is_retained(env) -> None:
    task = await env.engine.create(
        "legacy",
        metadata={"schedule_payload": {
            "delivery": [_target("telegram", "42"), {"kind": "local", "platform": "local"}],
        }},
    )
    assert _delivery_targets(task) == {"telegram": "42", "local": "local"}
    legacy = await env.engine.create(
        "old", metadata={"notify_targets": {"ops": "room-1"}}
    )
    assert _delivery_targets(legacy) == {"ops": "room-1"}
