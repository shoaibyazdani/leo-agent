"""Offline tests for WhatsApp message transport and the reply runner.

No Node process, WhatsApp socket or model call is used.
"""

from __future__ import annotations

import asyncio

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


async def test_typing_keepalive_repeats_then_pauses():
    """The keepalive refreshes "typing…" and always clears it afterwards."""
    from leo.integrations import whatsapp_msg

    class FakeManager:
        def __init__(self) -> None:
            self.states: list[str] = []
            self.fail = False

        async def typing(self, owner_id=None, *, to, state="composing", auth=None):
            self.states.append(state)
            if self.fail:
                raise RuntimeError("presence unavailable")

    manager = FakeManager()
    keepalive = whatsapp_msg._TypingKeepalive(
        manager, "owner", "auth", "15551234567@s.whatsapp.net",
    )
    keepalive.INTERVAL_SECONDS = 0.01
    await keepalive.start()
    assert manager.states == ["composing"]
    await asyncio.sleep(0.05)
    assert manager.states.count("composing") >= 2
    await keepalive.stop()
    assert manager.states[-1] == "paused"
    # A failing presence call must never raise into the reply path.
    manager.fail = True
    await keepalive.start()
    await keepalive.stop()


async def test_message_runner_sends_presence_around_a_slow_turn(monkeypatch):
    """A turn publishes composing and pauses before the reply is sent."""
    from leo.integrations import whatsapp_msg

    order: list[str] = []

    class FakeManager:
        session_auth = "auth"

        def __init__(self, _config) -> None:
            pass

        async def start(self, owner_id=None, auth=None):
            return {"state": "connected"}

        async def wait_for_connection(self, owner_id=None, timeout=None, auth=None):
            return {"state": "connected", "linked": True}

        async def messages(self, owner_id=None, after=0, timeout=0, auth=None):
            if order and "asked" in order:
                raise asyncio.CancelledError
            return {
                "sequence": 1,
                "messages": [{
                    "id": "ABC123",
                    "chat": "15551234567@s.whatsapp.net",
                    "sender": "15551234567@s.whatsapp.net",
                    "text": "hello",
                    "timestamp": 1_700_000_000,
                    "is_group": False,
                }],
            }

        async def typing(self, owner_id=None, *, to, state="composing", auth=None):
            order.append(f"typing:{state}")

        async def send(self, owner_id=None, *, to, text, auth=None):
            order.append("sent")

        async def stop(self, owner_id=None, auth=None):
            pass

        def diagnostics(self):
            return []

    class FakeApp:
        settings = None

        async def initialize(self):
            pass

        async def ask(self, text, **kwargs):
            order.append("asked")
            return {"content": "hi there"}

        async def shutdown(self):
            pass

    monkeypatch.setattr(whatsapp_msg, "ensure_ready", lambda: None)
    monkeypatch.setattr(whatsapp_msg, "bridge_status", lambda _owner: {"linked": True})
    monkeypatch.setattr(whatsapp_msg, "LeoApp", lambda _settings: FakeApp())
    monkeypatch.setattr(whatsapp_msg, "WhatsAppBridgeManager", FakeManager)
    monkeypatch.setattr(whatsapp_msg, "missing_model_env", lambda _app: [])
    monkeypatch.setattr(whatsapp_msg._TypingKeepalive, "INTERVAL_SECONDS", 3600.0)

    with pytest.raises(asyncio.CancelledError):
        await run_message_agent("owner", allow=["15551234567"])

    assert order[:3] == ["typing:composing", "asked", "typing:paused"]
    assert order[-1] == "sent"


async def test_message_runner_refuses_an_unlinked_session(monkeypatch):
    from leo.integrations import whatsapp_msg

    monkeypatch.setattr(whatsapp_msg, "ensure_ready", lambda: None)
    monkeypatch.setattr(whatsapp_msg, "bridge_status", lambda _owner: {"linked": False})

    with pytest.raises(whatsapp_msg.LinkUnavailable):
        await run_message_agent("owner", allow=["15551234567"])


def test_the_serve_process_cannot_restart_itself(monkeypatch) -> None:
    """A self-restart kills the WhatsApp turn before it can send its answer.

    Live 2026-11-28: "why not replying" → model asks shell to restart
    leo-whatsapp → systemd stops the process 5 s later → no reply, fresh listener.
    """
    from leo.config import Settings
    from leo.tools.shell_policy import preflight

    settings = Settings()
    monkeypatch.setenv("LEO_WHATSAPP_SERVE", "1")

    for command in (
        "systemctl restart leo-whatsapp.service",
        "sudo systemctl restart leo-agent.service leo-whatsapp.service; sleep 6",
        "service leo-whatsapp restart",
        "systemctl try-restart leo-whatsapp",
    ):
        reason = preflight("shell", {"command": command}, settings)
        assert reason and "inside its own turn" in reason, command

    # Diagnosis and the OTHER service stay available.
    assert preflight("shell", {"command": "systemctl is-active leo-whatsapp.service"}, settings) is None
    assert preflight("shell", {"command": "systemctl restart leo-agent.service"}, settings) is None


def test_outside_the_serve_process_the_guard_is_inactive(monkeypatch) -> None:
    from leo.config import Settings
    from leo.tools.shell_policy import preflight

    monkeypatch.delenv("LEO_WHATSAPP_SERVE", raising=False)
    assert preflight(
        "shell", {"command": "systemctl restart leo-whatsapp.service"}, Settings()
    ) is None
