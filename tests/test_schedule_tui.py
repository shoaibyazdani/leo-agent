"""Phase 9 terminal surface: full cards, ack and mid-turn decisions."""

from __future__ import annotations

import asyncio
import io

import pytest
from rich.console import Console

from leo import db
from leo.config import Settings
from leo.events import EventBus
from leo.interfaces import tui as tui_mod
from leo.security.permissions import PermissionManagerImpl


def _console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(file=stream, force_terminal=False, width=140, highlight=False), stream


class _App:
    """A stream that stays open until released, like a turn waiting on a card."""

    def __init__(self, permissions: PermissionManagerImpl, events: EventBus) -> None:
        self.permissions = permissions
        self.events = events
        self.gate = asyncio.Event()
        self.turn = None

    async def ask_stream(self, text, channel="cli", subject=None, conversation_id=None, **kwargs):
        self.turn = kwargs.get("turn")
        yield {"type": "run.started", "conversation_id": "conv-tui"}
        await self.gate.wait()
        yield {"type": "message.finished", "content": "done", "status": "completed"}


async def _card(body: str = 'Schedule "Water" (every day at 8am): Drink water.') -> tuple[PermissionManagerImpl, EventBus, object]:
    settings = Settings()
    settings.security.approval_poll_seconds = 0.01
    permissions = PermissionManagerImpl(settings)
    bus = EventBus(persist=False)
    card = await permissions.create_schedule_card(
        kind="schedule_create", body=body, proposal_digest="tui-phase9",
        requested_by="chat:web", conversation_id="conv-tui",
    )
    return permissions, bus, card


@pytest.mark.asyncio
async def test_presented_event_prints_literal_full_card_acknowledges_and_rewrites() -> None:
    permissions, bus, card = await _card()
    app = _App(permissions, bus)
    console, stream = _console()
    tui = tui_mod.ChatTUI(app, console=console, conversation_id="conv-tui")
    task = asyncio.create_task(tui.turn("schedule it"))
    try:
        for _ in range(20):
            if "approval.presented" in bus._handlers and bus._handlers["approval.presented"]:
                break
            await asyncio.sleep(0.01)
        await bus.emit("approval.presented", surface="tui",
                       **{**card.event_payload(), "presented_surface": "tui"})
        output = stream.getvalue()
        assert app.turn is not None and app.turn.surface == "tui"
        assert app.turn.capabilities.can_present_card is True
        assert card.body in output
        assert "Allow for this chat [session]" in output
        row = await db.fetchone(
            "SELECT shown_at, metadata FROM approvals WHERE id=?", (card.approval_id,)
        )
        assert row["shown_at"]
        assert '"acknowledged_at"' in (row["metadata"] or "")

        assert await tui.run_command("approve", [card.approval_id, "session"]) is True
        app.gate.set()
        await asyncio.wait_for(task, timeout=5)
        assert "approved" in stream.getvalue().lower()
        row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (card.approval_id,))
        assert row["status"] == "approved"
        grants = await db.fetchall("SELECT subject_type, subject_id FROM approval_grants")
        assert [dict(item) for item in grants] == [{
            "subject_type": "conversation", "subject_id": "conv-tui"
        }]
    finally:
        app.gate.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_mid_turn_approve_is_dispatched_while_other_input_is_queued() -> None:
    permissions, bus, card = await _card()
    app = _App(permissions, bus)
    console, stream = _console()
    tui = tui_mod.ChatTUI(app, console=console, conversation_id="conv-tui")
    tui._lines = asyncio.Queue()
    tui._queued = []
    tui._lines.put_nowait("an ordinary follow-up")
    tui._lines.put_nowait(f"/approve {card.approval_id} once")
    asyncio.get_running_loop().call_later(0.05, app.gate.set)
    await asyncio.wait_for(tui._turn_with_interrupts("schedule it"), timeout=5)
    assert tui._queued == ["an ordinary follow-up"]
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (card.approval_id,))
    assert row["status"] == "approved"
    assert "queued for the next turn" in stream.getvalue()


@pytest.mark.asyncio
async def test_mid_turn_deny_records_a_reason_and_never_commits() -> None:
    permissions, bus, card = await _card('Schedule "Brief": summarize inboxes.')
    app = _App(permissions, bus)
    console, stream = _console()
    tui = tui_mod.ChatTUI(app, console=console, conversation_id="conv-tui")
    tui._lines = asyncio.Queue()
    tui._queued = []
    tui._lines.put_nowait(f"/deny {card.approval_id} not this week")
    asyncio.get_running_loop().call_later(0.05, app.gate.set)
    await asyncio.wait_for(tui._turn_with_interrupts("schedule it"), timeout=5)
    row = await db.fetchone(
        "SELECT status, card, deny_reason FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "denied"
    assert row["deny_reason"] == "not this week"
    assert tui._queued == []
    assert "denied" in stream.getvalue().lower()


@pytest.mark.asyncio
async def test_always_choice_and_wrong_choice_are_both_bounded() -> None:
    permissions, bus, card = await _card()
    tui = tui_mod.ChatTUI(_App(permissions, bus), console=_console()[0])
    bad = await tui._resolve_schedule_card_tui(
        card.approval_id, approved=True, choice="bogus"
    )
    assert bad["success"] is False
    ok = await tui._resolve_schedule_card_tui(
        card.approval_id, approved=True, choice="always"
    )
    assert ok["success"] is True and ok["grant_id"]
    grant = await db.fetchone("SELECT subject_type FROM approval_grants")
    assert grant["subject_type"] == "profile"


def test_tui_paginates_long_bodies_without_losing_text() -> None:
    body = "line of approval text\n" * 600
    from leo.security.approval_cards import ApprovalCard

    permissions = PermissionManagerImpl(Settings())
    long_card = ApprovalCard(approval_id="apr_long", body=body, title="long")
    console, stream = _console()
    tui = tui_mod.ChatTUI(_App(permissions, EventBus(persist=False)), console=console)
    tui._print_approval_card(long_card)
    output = stream.getvalue()
    assert "[page 1/" in output
    assert "line of approval text" in output
