"""Offline verification for the BlueBubbles channel adapter.

BlueBubbles is a community iMessage bridge whose webhooks are NOT signed by
upstream — verification here is LEO's own fail-closed HMAC convention. No
live network calls: HTTP goes through ``httpx.MockTransport`` and the bridge
password is a fake value that must never surface in errors or records.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from leo.integrations.channels.bluebubbles import BlueBubblesAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

ENV_NAMES = ("BLUEBUBBLES_URL", "BLUEBUBBLES_PASSWORD", "BLUEBUBBLES_WEBHOOK_SECRET")
FAKE_PASSWORD = "fake-bb-password-never-leak"
BB_URL = "https://bb.example.org"
SECRET = "bb-webhook-secret"


@pytest.fixture(autouse=True)
def clean_bb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _signature(secret: str, raw: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _raw(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _signed_headers(payload: Any, secret: str = SECRET) -> dict[str, str]:
    return {"X-Leo-Signature": _signature(secret, _raw(payload))}


def _configured(monkeypatch: pytest.MonkeyPatch, *, url: str = BB_URL) -> None:
    monkeypatch.setenv("BLUEBUBBLES_URL", url)
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", FAKE_PASSWORD)


# ------------------------------------------------------------------- verify


async def test_bluebubbles_verify_webhook_leo_hmac_convention(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_SECRET", SECRET)
    adapter = BlueBubblesAdapter()
    raw = b'{"type":"new-message","data":{"guid":"SQL-1"}}'
    good = {"X-Leo-Signature": _signature(SECRET, raw)}
    assert adapter.verify_webhook(raw, good, raw_body=raw) is True
    assert adapter.verify_webhook(raw, {"X-Leo-Signature": "sha256=" + "00" * 32}, raw_body=raw) is False
    assert adapter.verify_webhook(raw, {}, raw_body=raw) is False
    # Byte-exactness: a signature over different bytes must fail.
    assert adapter.verify_webhook(raw, {"X-Leo-Signature": _signature(SECRET, b"other")}, raw_body=raw) is False
    # Same convention when only a dict payload is available (compact JSON).
    payload = {"type": "ping"}
    assert adapter.verify_webhook(payload, {"x-leo-signature": _signature(SECRET, _raw(payload))}) is True


async def test_bluebubbles_verify_fails_closed_without_secret(monkeypatch):
    adapter = BlueBubblesAdapter()
    assert adapter.verify_webhook({}, {}) is False  # default: fail closed
    # The escape hatch is explicit and local to the adapter instance.
    assert BlueBubblesAdapter(require_verification=False).verify_webhook({}, {}) is True


# ------------------------------------------------------------------ webhook


async def test_bluebubbles_handle_webhook_normalizes_new_message(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_SECRET", SECRET)
    adapter = BlueBubblesAdapter()
    payload = {
        "type": "new-message",
        "data": {
            "guid": "SQL-chat-1",
            "text": "hello from imessage",
            "fromAddress": "+15551234567",
            "timestamp": 1700000000000,
            "message": {
                "guid": "SQL-msg-9",
                "text": "hello from imessage",
                "fromHandle": {"id": "+15551234567"},
            },
        },
    }
    raw = _raw(payload)
    headers = {"X-Leo-Signature": _signature(SECRET, raw)}
    messages = await adapter.handle_webhook(payload, headers, raw_body=raw)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == (
        "bluebubbles",
        "+15551234567",
        "hello from imessage",
    )
    assert message.message_id == "SQL-chat-1"  # data.guid wins over message.guid
    assert message.timestamp.startswith("2023-11-14")
    assert message.reply_ref is None

    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"X-Leo-Signature": "sha256=" + "00" * 32}, raw_body=raw)
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {}, raw_body=raw)


async def test_bluebubbles_handle_webhook_sender_fallbacks(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_SECRET", SECRET)
    adapter = BlueBubblesAdapter()
    cases = [
        ({"type": "new-message", "data": {"guid": "g2", "text": "hi", "handle": {"id": "+1555999"}}}, "+1555999"),
        (
            {"type": "new-message", "data": {"guid": "g3", "text": "yo", "message": {"fromHandle": {"id": "+1555777"}}}},
            "+1555777",
        ),
        ({"type": "new-message", "data": {"guid": "g4", "text": "anon"}}, "unknown"),
    ]
    for payload, expected_sender in cases:
        messages = await adapter.handle_webhook(payload, _signed_headers(payload))
        assert len(messages) == 1
        assert messages[0].sender == expected_sender


async def test_bluebubbles_handle_webhook_skips_malformed_without_raising(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_SECRET", SECRET)
    adapter = BlueBubblesAdapter()
    # Non-text (attachment/tapback) events and unknown event types yield nothing.
    for payload in (
        {"type": "new-message", "data": {"guid": "g5", "timestamp": 1700000000000}},  # no text
        {"type": "new-message", "data": "not-a-dict"},
        {"type": "typing-indicator", "data": {"guid": "g6", "text": "x"}},
        {"type": "new-message", "data": {"text": 42}},  # text is not a string
    ):
        assert await adapter.handle_webhook(payload, _signed_headers(payload)) == []
    # Unparseable bodies are data, not crashes.
    raw = b"not json at all"
    assert await adapter.handle_webhook(raw, {"X-Leo-Signature": _signature(SECRET, raw)}, raw_body=raw) == []
    raw = b'["a list"]'
    assert await adapter.handle_webhook(raw, {"X-Leo-Signature": _signature(SECRET, raw)}, raw_body=raw) == []


# --------------------------------------------------------------------- send


async def test_bluebubbles_send_uses_password_header_and_leo_user_agent(monkeypatch):
    _configured(monkeypatch)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"message": {"guid": "SQL-out-1"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await BlueBubblesAdapter(client=client).send("+15551234567", "hello bridge")
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/message/text"
    assert request.url.params["guid"] == "+15551234567"
    assert "%2B15551234567" in str(request.url)  # the '+' is safely encoded
    assert request.headers["password"] == FAKE_PASSWORD
    assert request.headers["user-agent"] == "LEO"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {"message": "hello bridge", "method": "POST", "guid": "+15551234567"}
    assert result == {
        "channel": "bluebubbles",
        "target": "+15551234567",
        "success": True,
        "message_id": "SQL-out-1",
    }


async def test_bluebubbles_send_blocked_without_envs(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", FAKE_PASSWORD)
    with pytest.raises(MessagingError, match="bluebubbles send blocked: BLUEBUBBLES_URL is not set"):
        await BlueBubblesAdapter().send("+15551234567", "hi")
    monkeypatch.setenv("BLUEBUBBLES_URL", BB_URL)
    monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)
    with pytest.raises(MessagingError, match="bluebubbles send blocked: BLUEBUBBLES_PASSWORD is not set"):
        await BlueBubblesAdapter().send("+15551234567", "hi")


async def test_bluebubbles_send_rejects_invalid_url_and_empty_target(monkeypatch):
    _configured(monkeypatch, url="ftp://bb.example.org")
    with pytest.raises(MessagingError, match="BLUEBUBBLES_URL is not a valid"):
        await BlueBubblesAdapter().send("+15551234567", "hi")
    _configured(monkeypatch, url="https://bb.example.org/../evil")
    with pytest.raises(MessagingError, match="BLUEBUBBLES_URL is not a valid"):
        await BlueBubblesAdapter().send("+15551234567", "hi")
    _configured(monkeypatch)
    with pytest.raises(MessagingError, match="empty chat guid"):
        await BlueBubblesAdapter().send("", "hi")


async def test_bluebubbles_send_failures_report_no_secrets(monkeypatch):
    _configured(monkeypatch)

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "no chat found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError, match="bluebubbles send failed: HTTP 400") as exc:
            await BlueBubblesAdapter(client=client).send("+15551234567", "hi")
    assert FAKE_PASSWORD not in str(exc.value)

    def exploding(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"dial {BB_URL} with {FAKE_PASSWORD}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(exploding)) as client:
        with pytest.raises(MessagingError, match="bluebubbles send failed: ConnectError") as exc:
            await BlueBubblesAdapter(client=client).send("+15551234567", "hi")
    assert FAKE_PASSWORD not in str(exc.value)
    assert "dial" not in str(exc.value)

    # A non-JSON success response still counts as sent, just without an id.
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"ok"))
    ) as client:
        result = await BlueBubblesAdapter(client=client).send("+15551234567", "hi")
    assert result == {"channel": "bluebubbles", "target": "+15551234567", "success": True, "message_id": None}


# ------------------------------------------------------------------- record


async def test_bluebubbles_record_blocked_configured_and_capabilities(monkeypatch):
    adapter = BlueBubblesAdapter()
    record = adapter.record()
    assert record.status == "blocked"
    assert record.capabilities == ["send", "webhook"]  # no pull transport
    assert record.verification == "mock-tested"
    assert record.metadata["credential_env"] == ["BLUEBUBBLES_URL", "BLUEBUBBLES_PASSWORD"]
    assert record.metadata["verification_env"] == ["BLUEBUBBLES_WEBHOOK_SECRET"]

    _configured(monkeypatch)
    monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_SECRET", SECRET)
    record = BlueBubblesAdapter().record()
    assert record.status == "configured"
    assert record.config_ref == "BLUEBUBBLES_URL,BLUEBUBBLES_PASSWORD,BLUEBUBBLES_WEBHOOK_SECRET"
    assert record.metadata["live_verified"] is False
    assert FAKE_PASSWORD not in json.dumps(record.model_dump())
