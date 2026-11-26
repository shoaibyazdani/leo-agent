"""Phase 8: literal Telegram cards, lossless chunks and threaded callbacks."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from leo import db
from leo.config import Settings
from leo.events import Event
from leo.gateway.callbacks import handle_tap
from leo.integrations.messaging import (
    MAX_MESSAGE_CHUNKS,
    TELEGRAM_MAX_CHARS,
    AdapterCapabilities,
    MessagingError,
    TelegramAdapter,
    split_plain_text,
)
from leo.security.approval_buttons import parse_callback_data
from leo.security.approval_cards import card_from_row
from leo.security.notifier import ApprovalNotifier
from leo.security.permissions import PermissionManagerImpl


def test_plain_chunks_are_lossless_and_never_reassemble_fences() -> None:
    body = ("**literal** ```\n" * 20) + "tail"
    chunks = split_plain_text(body, limit=100, max_chunks=MAX_MESSAGE_CHUNKS)
    assert "".join(chunks) == body
    assert all(len(part) <= 100 for part in chunks)
    assert all("truncated" not in part for part in chunks)
    assert sum(part.count("```") for part in chunks) == body.count("```")
    with pytest.raises(MessagingError):
        split_plain_text("x" * 1000, limit=100, max_chunks=3)


async def test_plain_send_has_no_parse_mode(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})

    body = ("**verbatim** ```\n" * 400) + "end"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        adapter = TelegramAdapter(client=client, require_verification=False)
        result = await adapter.send("42", body, plain_text=True)
    assert result["success"] is True
    assert len(seen) > 1
    assert all("parse_mode" not in item for item in seen)
    assert "".join(item["text"] for item in seen) == body


async def test_webhook_callback_dispatch_keeps_thread_and_immutable_sender() -> None:
    seen: list[dict] = []

    async def sink(channel, descriptor):
        seen.append({"channel": channel, **dict(descriptor)})

    adapter = TelegramAdapter(require_verification=False)
    adapter.set_callback_sink(sink)
    update = {
        "update_id": 1,
        "callback_query": {
            "id": "cb-1",
            "data": "tga1:j:o:apr_1",
            "from": {"id": 111, "username": "mutable-name"},
            "message": {
                "message_id": 9,
                "message_thread_id": 77,
                "chat": {"id": 42},
                "text": "approval body",
            },
        },
    }
    messages = await adapter.handle_webhook(update)
    assert messages == []
    assert len(seen) == 1
    assert seen[0]["sender_id"] == "111"
    assert seen[0]["chat_id"] == "42"
    assert seen[0]["thread_id"] == "77"
    assert seen[0]["data"] == "tga1:j:o:apr_1"
    caps = AdapterCapabilities.for_adapter(adapter)
    assert caps.approval_card and caps.full_text and caps.terminal_edit


async def test_unauthorized_sender_cannot_decide_a_threaded_callback() -> None:
    settings = SimpleNamespace(
        channels=SimpleNamespace(policies={"telegram": SimpleNamespace(allow=["999"])}),
        security=SimpleNamespace(approval_timeout_seconds=900),
    )
    adapter = TelegramAdapter(require_verification=False)
    result = await handle_tap(
        "telegram",
        {
            "callback_id": "cb-1", "data": "tga1:j:o:apr_1", "sender_id": "111",
            "chat_id": "42", "message_id": "9", "thread_id": "77",
        },
        settings=settings, permissions=SimpleNamespace(), adapters={"telegram": adapter},
    )
    assert result["ok"] is False
    assert result["reason"] == "not authorized"


@pytest.mark.asyncio
async def test_oversized_card_is_deny_only_and_full_text_stays_durable(monkeypatch) -> None:
    monkeypatch.delenv("LEO_NO_NOTIFY", raising=False)
    settings = Settings()
    settings.security.notify.enabled = True
    settings.security.notify.channel = "telegram"
    settings.security.notify.target = "42"
    permissions = PermissionManagerImpl(settings)
    body = "FULL-CARD-" + ("x" * (TELEGRAM_MAX_CHARS * 2))
    card = await permissions.create_schedule_card(
        kind="schedule_create", body=body, proposal_digest="digest-long",
        requested_by="chat:web", conversation_id="conv-1",
    )
    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (card.approval_id,))
    assert row is not None

    class Adapter:
        def __init__(self):
            self.calls: list[tuple[str, str, dict]] = []

        async def send(self, target, text, **kwargs):
            self.calls.append((target, text, kwargs))
            return {"message_id": "1"}

    adapter = Adapter()
    gateway = SimpleNamespace(_adapter=lambda channel: adapter)
    notifier = ApprovalNotifier(gateway, settings, events=None, notify_only=False)
    await notifier._on_requested(Event(topic="approval.requested", payload={
        "approval_id": card.approval_id,
        "kind": "schedule_create",
        "title": card.title,
        "body": body,
        "choices": list(card.choices),
        "choice_labels": list(card.choice_labels),
    }))
    assert len(adapter.calls) == 1
    target, text, kwargs = adapter.calls[0]
    assert target == "42"
    assert "FULL-CARD-" not in text
    assert kwargs.get("plain_text") is True
    buttons = kwargs["reply_markup"]["inline_keyboard"]
    assert len(buttons) == 1 and len(buttons[0]) == 1
    parsed = parse_callback_data(buttons[0][0]["callback_data"])
    assert parsed is not None and parsed["decision"] == "deny"
    updated = card_from_row(
        await db.fetchone("SELECT * FROM approvals WHERE id=?", (card.approval_id,))
    )
    assert updated.deny_only is True
    assert updated.body == body


@pytest.mark.asyncio
async def test_short_card_is_literal_plain_text_with_every_choice(monkeypatch) -> None:
    monkeypatch.delenv("LEO_NO_NOTIFY", raising=False)
    settings = Settings()
    settings.security.notify.enabled = True
    settings.security.notify.channel = "telegram"
    settings.security.notify.target = "42"
    permissions = PermissionManagerImpl(settings)
    body = "**literal** ```card``` body\nline two"
    card = await permissions.create_schedule_card(
        kind="schedule_create", body=body, proposal_digest="digest-short",
        requested_by="chat:web", conversation_id="conv-2",
    )

    class Adapter:
        def __init__(self):
            self.calls: list[tuple[str, str, dict]] = []

        async def send(self, target, text, **kwargs):
            self.calls.append((target, text, kwargs))
            return {"message_id": "2"}

    adapter = Adapter()
    gateway = SimpleNamespace(_adapter=lambda channel: adapter)
    notifier = ApprovalNotifier(gateway, settings, events=None, notify_only=False)
    await notifier._on_requested(Event(topic="approval.requested", payload={
        "approval_id": card.approval_id,
        "kind": "schedule_create",
        "title": card.title,
        "body": body,
        "choices": list(card.choices),
        "choice_labels": list(card.choice_labels),
    }))
    assert len(adapter.calls) == 1
    _, text, kwargs = adapter.calls[0]
    assert body in text
    assert kwargs.get("plain_text") is True
    rows = kwargs["reply_markup"]["inline_keyboard"]
    assert len(rows) == 2 and sum(len(row) for row in rows) == 4
    assert {parse_callback_data(item["callback_data"])["decision"] for row in rows
            for item in row} == {"once", "session", "always", "deny"}


@pytest.mark.asyncio
async def test_native_callback_resolves_a_threaded_schedule_card_and_rewrites_it(monkeypatch) -> None:
    """Capability proof: native callbacks, immutable sender, kind-preserving rewrite."""
    from leo.security.approval_buttons import build_schedule_keyboard
    from leo.security.permissions import PermissionManagerImpl

    settings = Settings()
    permissions = PermissionManagerImpl(settings)
    card = await permissions.create_schedule_card(
        kind="schedule_create", body='Schedule "Water": drink water.',
        proposal_digest="tg-native", requested_by="chat:web",
        conversation_id="conv-tg", origin="gateway:telegram",
    )
    await permissions.present_schedule_card(card.approval_id, surface="telegram")
    keyboard = build_schedule_keyboard(
        card.approval_id, approval_kind=card.kind, choices=card.choices,
        choice_labels=card.choice_labels,
    )
    data = keyboard["inline_keyboard"][0][0]["callback_data"]

    class Adapter:
        def __init__(self):
            self.edits: list[tuple] = []
            self.answers: list[str] = []

        async def edit_message_text(self, target, message_id, text, markup=None, **extra):
            self.edits.append((target, message_id, text, markup, extra))
            return True

        async def answer_callback_query(self, callback_id, note):
            self.answers.append(note)

    adapter = Adapter()
    result = await handle_tap(
        "telegram",
        {
            "callback_id": "cb-native", "data": data, "sender_id": "111",
            "chat_id": "42", "message_id": "9", "thread_id": "77",
        },
        settings=settings, permissions=permissions,
        adapters={"telegram": adapter}, allowed=["111"],
    )
    assert result["ok"] is True
    assert result["choice"] == "once"
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (card.approval_id,))
    assert row["status"] == "approved"
    assert len(adapter.edits) == 1
    target, message_id, text, markup, extra = adapter.edits[0]
    assert (target, message_id) == ("42", "9")
    assert card.body in text and "approved" in text
    assert markup == {"inline_keyboard": []}
    assert extra.get("thread_id") == "77" and extra.get("plain_text") is True


@pytest.mark.asyncio
async def test_incapable_surface_gets_no_schedule_keyboard(monkeypatch) -> None:
    from leo.contracts import surface_capabilities

    monkeypatch.delenv("LEO_NO_NOTIFY", raising=False)
    monkeypatch.setenv("LEO_DB_PATH", str(db.get_db_path()))
    settings = Settings()
    settings.security.notify.enabled = True
    settings.security.notify.channel = "whatsapp"
    settings.security.notify.target = "42"
    permissions = PermissionManagerImpl(settings)
    card = await permissions.create_schedule_card(
        kind="schedule_create", body='Schedule "Water": drink water.',
        proposal_digest="tg-incapable", requested_by="chat:web",
    )

    class Adapter:
        def __init__(self):
            self.calls: list[dict] = []

        async def send(self, target, text, **kwargs):
            self.calls.append(kwargs)
            return {"message_id": "1"}

    adapter = Adapter()
    notifier = ApprovalNotifier(
        SimpleNamespace(_adapter=lambda channel: adapter), settings, events=None,
    )
    assert surface_capabilities("whatsapp").can_present_card is False
    await notifier._on_requested(Event(topic="approval.requested", payload={
        **card.event_payload(), "kind": "schedule_create",
    }))
    assert len(adapter.calls) == 1
    assert "reply_markup" not in adapter.calls[0]
    row = await db.fetchone(
        "SELECT shown_at FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["shown_at"] is None
