"""Offline tests for WhatsApp message transport and the reply runner.

No Node process, WhatsApp socket or model call is used.
"""

from __future__ import annotations

import pytest

from leo.integrations.whatsapp_bridge import manager as bridge_manager
from leo.integrations.whatsapp_msg import (
    MAX_REPLY_CHARS,
    _digits,
    _prefer_pn,
    _sender_key,
    run_message_agent,
    split_text,
)


def test_sender_helpers_strip_device_and_domain():
    assert _digits("+1 (555) 123-4567") == "15551234567"
    assert _sender_key("15551234567:7@s.whatsapp.net") == "15551234567"
    assert _sender_key("1203630@g.us") == "1203630"


def test_prefer_pn_resolves_a_lid_to_the_phone_number():
    lid = "100000000000001@lid"
    pn = "15551234567@s.whatsapp.net"
    assert _prefer_pn(lid, pn) == pn
    assert _prefer_pn(pn, lid) == pn
    assert _prefer_pn("", "") == ""


def test_split_text_prefers_paragraphs_and_respects_limit():
    first = "a" * 1200
    second = "b" * 1200
    chunks = split_text(f"{first}\n\n{second}", limit=1400)
    assert chunks == [first, second]
    assert all(len(chunk) <= 1400 for chunk in chunks)

    hard = "x" * (MAX_REPLY_CHARS * 2 + 5)
    chunks = split_text(hard)
    assert all(len(chunk) <= MAX_REPLY_CHARS for chunk in chunks)
    assert "".join(chunks) == hard


def test_public_message_rejects_malformed_payloads():
    good = {
        "id": "ABCDEF123",
        "chat": "15551234567@s.whatsapp.net",
        "sender": "15551234567@s.whatsapp.net",
        "chat_alt": "100000000000001@lid",
        "sender_alt": "15551234567@s.whatsapp.net",
        "text": "hello",
        "timestamp": 1_700_000_000,
        "is_group": False,
    }
    assert bridge_manager.WhatsAppBridgeManager._public_message(good) == good
    # A LID-only direct chat is valid on its own (the alt fields are optional).
    lid_only = {**good, "chat": "100000000000001@lid", "chat_alt": None}
    assert bridge_manager.WhatsAppBridgeManager._public_message(lid_only) is not None

    for mutation in (
        {"id": ""},
        {"chat": "bad id with spaces"},
        {"sender": ""},
        {"text": ""},
        {"text": "x" * (bridge_manager.MAX_MESSAGE_TEXT_CHARS + 1)},
    ):
        assert bridge_manager.WhatsAppBridgeManager._public_message({**good, **mutation}) is None


async def test_message_runner_refuses_without_an_allowlist(monkeypatch):
    from leo.integrations import whatsapp_msg

    monkeypatch.setattr(whatsapp_msg, "ensure_ready", lambda: None)
    monkeypatch.setattr(whatsapp_msg, "bridge_status", lambda _owner: {"linked": True})

    with pytest.raises(whatsapp_msg.LinkUnavailable):
        await run_message_agent("owner", allow=())


async def test_message_runner_refuses_an_unlinked_session(monkeypatch):
    from leo.integrations import whatsapp_msg

    monkeypatch.setattr(whatsapp_msg, "ensure_ready", lambda: None)
    monkeypatch.setattr(whatsapp_msg, "bridge_status", lambda _owner: {"linked": False})

    with pytest.raises(whatsapp_msg.LinkUnavailable):
        await run_message_agent("owner", allow=["15551234567"])
