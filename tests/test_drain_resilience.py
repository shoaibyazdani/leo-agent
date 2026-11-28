"""The outbox consumer must survive a cancelled turn.

Live failure (2026-11-24): LEO stopped answering while the process was healthy —
`/chat` answered in 1.3 s, Telegram was reachable in 90 ms, yet five operator
messages sat in `processing` and every later message stayed `pending` with
attempts=0. The drain loop was gone: a turn that hit its lifetime cap raised
CancelledError, `_gateway_loop` re-raised it, and that loop is the ONLY consumer
of the channel outbox. Nothing re-armed it until the next restart.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leo import db
from leo.config import ChannelsConfig
from leo.contracts import Decision, RiskLevel, utcnow
from leo.integrations.messaging import NormalizedMessage
from leo.events import EventBus
from leo.integrations.admission import ChannelAdmission
from leo.integrations.gateway import Gateway


class AllowAll:
    async def authorize(self, actor, action, resource=None, risk=None, **context):
        return Decision(allowed=True, reason="test", risk=risk or RiskLevel.LOW)

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        return Decision(allowed=approved, reason="test", risk=RiskLevel.HIGH)


@pytest.fixture
async def outbox_db(tmp_path):
    db.set_db_path(tmp_path / "drain.db")
    await db.initialize()
    yield db


def make_gateway(**kwargs: Any) -> Gateway:
    return Gateway(
        kwargs.pop("adapters", {}),
        events=EventBus(persist=True),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
        **kwargs,
    )


def msg(text: str, message_id: str, *, conversation: str = "chat-1") -> NormalizedMessage:
    return NormalizedMessage(
        channel="fake", sender="Shoaib", sender_id="u1",
        conversation_ref=conversation, text=text, message_id=message_id,
    )


@pytest.mark.asyncio
async def test_a_cancelled_turn_does_not_stop_the_drain(outbox_db) -> None:
    gateway = make_gateway()
    await gateway.receive(msg("first", "m-1"))
    await gateway.receive(msg("second", "m-2"))

    seen: list[str] = []

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(str(payload.get("text")))
        if payload.get("text") == "first":
            raise asyncio.CancelledError()
        return {"status": "sent", "success": True, "delivered": True}

    summary = await gateway.drain_outbox(consumer, limit=10)

    assert seen == ["first", "second"], "the row behind a cancelled turn must still run"
    assert summary["processed"] == 1
    assert summary["failed"] == 1
    row = await db.fetchone(
        "SELECT status, coalesce(last_error,'') AS err FROM gateway_outbox "
        "WHERE coalesce(last_error,'') LIKE '%CancelledError%'"
    )
    assert row is not None, "the cancelled row must stay retryable, not vanish"
    assert str(row["status"]) in ("pending", "failed")


@pytest.mark.asyncio
async def test_stale_processing_rows_come_back(outbox_db) -> None:
    """A drainer that died mid-row leaves the row invisible to the claim query."""
    gateway = make_gateway()
    await gateway.receive(msg("stranded", "m-3"))
    await db.execute(
        "UPDATE gateway_outbox SET status='processing', updated_at=?",
        ("2000-01-01T00:00:00+00:00",),
    )
    assert (await gateway.outbox_stats())["processing"] == 1

    recovered = await gateway.recover_stale_outbox()

    assert recovered == 1
    stats = await gateway.outbox_stats()
    assert stats.get("pending") == 1 and stats.get("processing") == 0


@pytest.mark.asyncio
async def test_the_loop_helper_only_reports_real_shutdown() -> None:
    from leo.core.app import _task_is_shutting_down

    assert _task_is_shutting_down() is False, "a healthy task is not shutting down"

    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    assert _task_is_shutting_down() is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(0)
    await asyncio.sleep(0)  # swallow the pending cancellation for the fixture


@pytest.mark.asyncio
async def test_a_hung_turn_is_timed_out_instead_of_holding_the_queue(outbox_db) -> None:
    """The live five-hour stall: a turn that never returns, nothing else drains."""
    gateway = make_gateway()
    await gateway.receive(msg("hangs", "m-10"))
    await gateway.receive(msg("after", "m-11"))

    seen: list[str] = []

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(str(payload.get("text")))
        if payload.get("text") == "hangs":
            await asyncio.sleep(5)  # never returns in time
        return {"status": "sent", "success": True, "delivered": True}

    summary = await gateway.drain_outbox(consumer, limit=10, per_row_timeout=0.2)
    assert seen == ["hangs", "after"], "the row behind a hung turn must still run"
    assert summary["processed"] == 1
    row = await db.fetchone(
        "SELECT status, coalesce(last_error,'') AS err FROM gateway_outbox "
        "WHERE coalesce(last_error,'') LIKE '%Timeout%'"
    )
    assert row is not None, "a hung row must be recorded, not lost"


@pytest.mark.asyncio
async def test_the_sweep_never_reclaims_a_turn_that_is_still_running(outbox_db) -> None:
    """Sweeping at 300s would double-run a legitimate 20-minute turn."""
    gateway = make_gateway()
    await gateway.receive(msg("long turn", "m-12"))
    recent = utcnow()
    await db.execute(
        "UPDATE gateway_outbox SET status='processing', updated_at=?", (recent,)
    )

    assert await gateway.recover_stale_outbox(older_than_seconds=2400.0) == 0
    assert (await gateway.outbox_stats())["processing"] == 1


# ------------------------------------------------------- per-conversation drain
def _delivered() -> dict[str, Any]:
    return {"status": "sent", "success": True, "delivered": True}


async def _order_rows(*texts: str) -> None:
    """Pin claim order: created_at can tie when rows arrive in one event loop."""
    for index, text in enumerate(texts):
        await db.execute(
            "UPDATE gateway_outbox SET created_at=? WHERE text=?",
            (f"2026-01-01T00:00:{index:02d}+00:00", text),
        )


@pytest.mark.asyncio
async def test_distinct_conversations_run_in_parallel(outbox_db) -> None:
    """Different chats must overlap; the old single loop queued them."""
    gateway = make_gateway(concurrency=4)
    for index in range(3):
        await gateway.receive(
            msg(f"chat-{index}", f"m-parallel-{index}", conversation=f"conv-{index}")
        )
    await _order_rows("chat-0", "chat-1", "chat-2")

    started: list[str] = []
    all_started = asyncio.Event()

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        started.append(str(payload["text"]))
        if len(started) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1.0)
        return _delivered()

    summary = await gateway.drain_outbox(consumer, limit=10)

    assert sorted(started) == ["chat-0", "chat-1", "chat-2"]
    assert all_started.is_set(), "three chats did not start their turns together"
    assert summary["processed"] == 3


@pytest.mark.asyncio
async def test_ten_chats_can_all_overlap_when_configured(outbox_db) -> None:
    """Acceptance case: ten distinct chats, ten simultaneous turns."""
    gateway = make_gateway(concurrency=10)
    for index in range(10):
        await gateway.receive(
            msg(f"chat-{index}", f"m-ten-{index}", conversation=f"conv-{index}")
        )
    await _order_rows(*[f"chat-{index}" for index in range(10)])

    started: list[str] = []
    all_started = asyncio.Event()

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        started.append(str(payload["text"]))
        if len(started) == 10:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1.0)
        return _delivered()

    summary = await gateway.drain_outbox(consumer, limit=10)

    assert all_started.is_set(), "ten chats did not all start before any answered"
    assert summary["processed"] == 10


@pytest.mark.asyncio
async def test_one_conversation_stays_serial_and_ordered(outbox_db) -> None:
    """Bounded fan-out may not interleave two turns of the same chat."""
    gateway = make_gateway(concurrency=4)
    for index in range(3):
        await gateway.receive(msg(f"m{index}", f"m-serial-{index}", conversation="same-chat"))
    await _order_rows("m0", "m1", "m2")

    seen: list[str] = []
    active = 0
    peak = 0

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen.append(str(payload["text"]))
        await asyncio.sleep(0.02)
        active -= 1
        return _delivered()

    summary = await gateway.drain_outbox(consumer, limit=10)

    assert seen == ["m0", "m1", "m2"], "same-chat turns ran out of order"
    assert peak == 1, "two turns of one conversation overlapped"
    assert summary["processed"] == 3


@pytest.mark.asyncio
async def test_fan_out_is_bounded_by_configured_concurrency(outbox_db) -> None:
    """Two slots, five chats: all answered, never more than two at once."""
    gateway = make_gateway(concurrency=2)
    for index in range(5):
        await gateway.receive(
            msg(f"job-{index}", f"m-bound-{index}", conversation=f"chat-{index}")
        )
    await _order_rows(*[f"job-{index}" for index in range(5)])

    active = 0
    peak = 0

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return _delivered()

    summary = await gateway.drain_outbox(consumer, limit=10)

    assert summary["processed"] == 5
    assert peak == 2, f"expected exactly two slots in use, saw {peak}"


@pytest.mark.asyncio
async def test_a_per_call_concurrency_override_bounds_the_drain(outbox_db) -> None:
    """A caller can narrow the gateway's default without changing the service."""
    gateway = make_gateway(concurrency=4)
    for index in range(3):
        await gateway.receive(
            msg(f"one-{index}", f"m-override-{index}", conversation=f"chat-{index}")
        )
    await _order_rows("one-0", "one-1", "one-2")

    active = 0
    peak = 0

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return _delivered()

    summary = await gateway.drain_outbox(consumer, limit=10, concurrency=1)

    assert summary["processed"] == 3
    assert peak == 1


@pytest.mark.asyncio
async def test_a_hung_turn_does_not_delay_another_conversation(outbox_db) -> None:
    """Only the stuck chat waits out the per-row timeout."""
    gateway = make_gateway()
    await gateway.receive(msg("hangs", "m-isolate-1", conversation="slow-chat"))
    await gateway.receive(msg("fast", "m-isolate-2", conversation="fast-chat"))
    await _order_rows("hangs", "fast")

    events: list[tuple[str, str]] = []
    hang_ended = asyncio.Event()
    fast_started_while_hung = False

    async def consumer(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal fast_started_while_hung
        text = str(payload["text"])
        events.append(("start", text))
        if text == "fast":
            fast_started_while_hung = not hang_ended.is_set()
        try:
            if text == "hangs":
                await asyncio.sleep(1.0)
            events.append(("end", text))
        finally:
            if text == "hangs":
                hang_ended.set()
        return _delivered()

    summary = await gateway.drain_outbox(consumer, limit=10, per_row_timeout=0.2)

    assert ("start", "hangs") in events
    assert fast_started_while_hung, "the fast turn waited for the hung turn's cap"
    assert ("end", "hangs") not in events, "the hung turn was allowed to finish"
    assert ("end", "fast") in events
    assert summary["processed"] == 1 and summary["failed"] == 1
