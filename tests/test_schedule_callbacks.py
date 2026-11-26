"""Schedule card taps: kind-preserving payloads, sender allowlist, threads.

A schedule tap is still authorized by the immutable sender id. The callback
carries a choice and a locator only; the digest comes from the durable row and
is re-checked by the manager. The terminal rewrite keeps the schedule kind and
the literal full body.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import db
from leo.config import ChannelsConfig, Settings
from leo.gateway.callbacks import handle_tap, resolve_tap_id
from leo.security.approval_buttons import (
    build_callback_data,
    build_resolution_ref,
    build_schedule_keyboard,
    parse_callback_data,
)
from leo.security.approval_cards import KIND_SCHEDULE_CREATE, KIND_SCHEDULE_EDIT
from leo.security.permissions import PermissionManagerImpl

OWNER = "100000001"
STRANGER = "999999999"
BODY = (
    'Schedule "Water" (every day at 8am, first run Wed Nov 25 08:00 UTC): '
    "Drink water.; results go to local history only"
)


class FakeAdapter:
    channel = "telegram"

    def __init__(self) -> None:
        self.answers: list[str] = []
        self.edits: list[tuple[str, str, str, dict]] = []

    def set_callback_sink(self, sink) -> None:
        self.sink = sink

    async def answer_callback_query(self, callback_id: str, text: str = "") -> bool:
        self.answers.append(str(text))
        return True

    async def edit_message_text(self, target, message_id, text, reply_markup=None) -> bool:
        self.edits.append((str(target), str(message_id), str(text), dict(reply_markup or {})))
        return True


@pytest.fixture
async def env():
    settings = Settings()
    settings.security.high_risk_requires_approval = True
    settings.channels = ChannelsConfig(
        policies={"telegram": {"mode": "allowlist", "allow": [OWNER]}}
    )
    permissions = PermissionManagerImpl(settings)
    adapter = FakeAdapter()
    return SimpleNamespace(
        settings=settings, permissions=permissions, adapter=adapter,
        adapters={"telegram": adapter},
    )


async def _card(
    env, *, kind: str = KIND_SCHEDULE_CREATE, digest: str = "tap-digest",
    present: bool = True,
):
    body = BODY if kind == KIND_SCHEDULE_CREATE else (
        'Change "Water": prompt: water -> tea.'
    )
    card = await env.permissions.create_schedule_card(
        kind=kind,
        body=body,
        proposal_digest=digest,
        requested_by="chat:web",
        conversation_id="conv-a",
    )
    if present:
        await env.permissions.present_schedule_card(
            card.approval_id, surface="telegram"
        )
    return card


def _tap(card, choice: str, *, sender: str = OWNER, thread_id: str = "77") -> dict:
    kind = "schedule_edit" if card.kind == KIND_SCHEDULE_EDIT else "schedule"
    return {
        "callback_id": "cb-1",
        "data": build_callback_data(kind, choice, card.approval_id),
        "sender_id": sender,
        "chat_id": OWNER,
        "message_id": "71",
        "thread_id": thread_id,
    }


def test_schedule_keyboard_round_trips_every_frozen_choice() -> None:
    keyboard = build_schedule_keyboard(
        "apr_1",
        approval_kind="schedule",
        choices=["once", "session", "always", "deny"],
        choice_labels=["Allow", "Allow for this chat", "Always allow new jobs", "Deny"],
    )
    buttons = [button for row in keyboard["inline_keyboard"] for button in row]
    assert [button["text"] for button in buttons] == [
        "Allow", "Allow for this chat", "Always allow new jobs", "Deny"
    ]
    parsed = [parse_callback_data(button["callback_data"]) for button in buttons]
    assert [item["decision"] for item in parsed] == [
        "once", "session", "always", "deny"
    ]
    assert {item["approval_kind"] for item in parsed} == {"schedule"}

    edit = build_schedule_keyboard(
        "apr_2", approval_kind="schedule_edit", choices=["once", "deny"],
    )
    parsed_edit = [
        parse_callback_data(button["callback_data"])
        for row in edit["inline_keyboard"] for button in row
    ]
    assert [item["decision"] for item in parsed_edit] == ["once", "deny"]
    assert {item["approval_kind"] for item in parsed_edit} == {"schedule_edit"}


async def test_schedule_tap_resolves_the_choice_and_preserves_the_thread(env) -> None:
    card = await _card(env)

    result = await handle_tap(
        "telegram", _tap(card, "session"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters,
        allowed=[OWNER],
    )

    assert result["ok"] is True
    assert result["choice"] == "session"
    assert result["grant_id"]
    assert result["thread_id"] == "77"
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "approved"
    grants = await db.fetchall(
        "SELECT principal, subject_type, subject_id FROM approval_grants"
    )
    assert [dict(item) for item in grants] == [{
        "principal": "chat:web",
        "subject_type": "conversation",
        "subject_id": "conv-a",
    }]
    # The terminal rewrite keeps the card kind and the literal full body.
    assert env.adapter.edits
    text = env.adapter.edits[-1][2]
    assert 'Schedule "Water"' in text
    assert "Drink water." in text
    assert "✅ approved" in text
    assert "🔐 Approval" not in text, "a schedule card is not a generic tool approval"


async def test_schedule_edit_tap_preserves_kind_and_creates_no_grant(env) -> None:
    card = await _card(env, kind=KIND_SCHEDULE_EDIT, digest="edit-tap")

    result = await handle_tap(
        "telegram", _tap(card, "once"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters, allowed=[OWNER],
    )

    assert result["ok"] is True and result["choice"] == "once"
    assert await db.fetchall("SELECT id FROM approval_grants") == []
    text = env.adapter.edits[-1][2]
    assert 'Change "Water": prompt: water -> tea.' in text
    assert "🔐 Approval" not in text


async def test_a_stranger_cannot_tap_a_schedule_card(env) -> None:
    card = await _card(env)

    result = await handle_tap(
        "telegram", _tap(card, "once", sender=STRANGER), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters,
    )

    assert result["ok"] is False and result["reason"] == "not authorized"
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "pending"
    assert env.adapter.edits == []


async def test_an_unpresented_schedule_card_refuses_the_tap(env) -> None:
    card = await _card(env, present=False)

    result = await handle_tap(
        "telegram", _tap(card, "once"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters, allowed=[OWNER],
    )

    assert result["ok"] is False and result["reason"] == "not actionable"
    assert "not been shown" in result["note"]
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "pending"


async def test_an_expired_schedule_card_refuses_the_tap(env) -> None:
    card = await _card(env)
    # Move the card into the past without waiting: the tap must read the row.
    await db.execute(
        "UPDATE approvals SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
        (card.approval_id,),
    )

    result = await handle_tap(
        "telegram", _tap(card, "once"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters, allowed=[OWNER],
    )

    assert result["ok"] is False and result["reason"] == "not actionable"
    assert "deadline passed" in result["note"]


async def test_a_long_schedule_id_still_resolves_through_its_locator(env) -> None:
    card = await _card(env)
    long_id = "apr_" + "a" * 120
    await db.execute(
        "UPDATE approvals SET id=? WHERE id=?", (long_id, card.approval_id)
    )
    locator = build_resolution_ref("schedule", long_id)
    resolved = await resolve_tap_id({
        "approval_id": locator,
        "approval_kind": "schedule",
        "is_ref": "1",
    })
    assert resolved == long_id


async def test_the_schedule_resolution_audit_names_the_choice(env) -> None:
    card = await _card(env)
    await handle_tap(
        "telegram", _tap(card, "deny"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters, allowed=[OWNER],
    )
    row = await db.fetchone(
        "SELECT decision, metadata FROM audit_log WHERE action='schedule:create' "
        "ORDER BY id DESC LIMIT 1"
    )
    assert row is not None
    import json

    metadata = json.loads(row["metadata"])
    assert metadata["choice"] == "deny"
    assert metadata["resolver"] == f"telegram-button:{OWNER}"
