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


def msg(text: str, message_id: str) -> NormalizedMessage:
    return NormalizedMessage(
        channel="fake", sender="Shoaib", sender_id="u1",
        conversation_ref="chat-1", text=text, message_id=message_id,
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
