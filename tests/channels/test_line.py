"""Offline verification for the LINE channel adapter.

No live network calls: HTTP goes through httpx.MockTransport and webhook
signatures are computed exactly as LINE computes them (base64 of
HMAC-SHA256 over the raw body).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from leo.integrations.channels import ALL_ADAPTERS
from leo.integrations.channels.line import LineAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

FAKE_TOKEN = "fake-line-token-should-never-leak"


def _body(payload: Any) -> bytes:
    """The exact bytes the adapter verifies when no raw_body is supplied."""
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _line_signature(secret: str, body: bytes) -> str:
    """X-Line-Signature = base64(HMAC-SHA256(channel_secret, raw_body))."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


async def test_line_webhook_signature_verification(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "line-secret")
    adapter = LineAdapter()
    payload = {"events": []}
    headers = {"X-Line-Signature": _line_signature("line-secret", _body(payload))}
    assert adapter.verify_webhook(payload, headers) is True

    # Byte-exact verification when the framework supplies the raw body.
    raw = b'{"events":[{"type":"message"}]}'
    raw_headers = {"X-Line-Signature": _line_signature("line-secret", raw)}
    assert adapter.verify_webhook(raw, raw_headers, raw_body=raw) is True

    assert adapter.verify_webhook(payload, {"X-Line-Signature": _line_signature("nope", _body(payload))}) is False
    assert adapter.verify_webhook(payload, {"X-Line-Signature": "not-base64!!"}) is False
    assert adapter.verify_webhook(payload, {}) is False

    # Fail closed when the secret env is absent.
    monkeypatch.delenv("LINE_CHANNEL_SECRET")
    assert adapter.verify_webhook(payload, headers) is False


async def test_line_handle_webhook_normalizes_and_skips(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "line-secret")
    adapter = LineAdapter()
    payload = {
        "destination": "U-dest",
        "events": [
            {
                "type": "message",
                "replyToken": "rtok",
                "timestamp": 1700000000000,  # LINE sends milliseconds
                "source": {"type": "user", "userId": "U123"},
                "message": {"id": "msg-1", "type": "text", "text": "hello line"},
            },
            {"type": "follow", "source": {"type": "user", "userId": "U123"}},
            {
                "type": "message",
                "timestamp": 1700000000001,
                "source": {"type": "group", "groupId": "C456"},  # no userId → group id
                "message": {"id": "msg-2", "type": "text", "text": "group hi"},
            },
            {
                "type": "message",  # non-text message types are skipped
                "timestamp": 1700000000002,
                "source": {"type": "group", "groupId": "C456", "userId": "U789"},
                "message": {"id": "msg-3", "type": "image"},
            },
            {
                "type": "message",  # missing source normalizes to "unknown"
                "timestamp": 1700000000003,
                "message": {"id": "msg-4", "type": "text", "text": "anon"},
            },
        ],
    }
    headers = {"X-Line-Signature": _line_signature("line-secret", _body(payload))}
    messages = await adapter.handle_webhook(payload, headers)
    assert len(messages) == 3
    first, second, third = messages
    for message in messages:
        assert isinstance(message, NormalizedMessage)
        assert message.channel == "line" and message.reply_ref is None
    assert (first.sender, first.text, first.message_id) == ("U123", "hello line", "msg-1")
    assert first.timestamp == "2023-11-14T22:13:20+00:00"  # ms epoch converted to seconds
    assert (second.sender, second.text, second.message_id) == ("C456", "group hi", "msg-2")
    assert (third.sender, third.text, third.message_id) == ("unknown", "anon", "msg-4")

    # Unsigned delivery is rejected before parsing.
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {})
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"X-Line-Signature": _line_signature("nope", _body(payload))})


async def test_line_handle_webhook_malformed_payload_yields_nothing(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "line-secret")
    adapter = LineAdapter()
    cases: list[Any] = [
        {"events": "not-a-list"},
        {"nope": 1},
        {"events": [42, None, "oops"]},
        {"events": [{"type": "message", "message": "not-a-dict"}]},
        {"events": [{"type": "message", "message": {"type": "sticker", "id": "s1"}}]},
        {"events": [{"type": "postback", "postback": {"data": "x"}}]},
        "not json",
        b'{"events": [',
    ]
    for payload in cases:
        headers = {"X-Line-Signature": _line_signature("line-secret", _body(payload))}
        assert await adapter.handle_webhook(payload, headers) == []


async def test_line_send_uses_injected_client_and_hides_token(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", FAKE_TOKEN)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"messageId": "line-msg-1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = LineAdapter(client=client)
        result = await adapter.send("U123", "hello")
    assert result == {"channel": "line", "target": "U123", "success": True, "message_id": "line-msg-1"}
    request = requests[0]
    assert str(request.url) == "https://api.line.me/v2/bot/message/push"
    assert request.headers["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert json.loads(request.content) == {"to": "U123", "messages": [{"type": "text", "text": "hello"}]}

    # LINE's push endpoint answers with an empty body: still a success, no id.
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    async with httpx.AsyncClient(transport=httpx.MockTransport(empty)) as client:
        result = await LineAdapter(client=client).send("U123", "hi")
    assert result["success"] is True and result["message_id"] is None


async def test_line_send_blocked_without_token(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", FAKE_TOKEN)
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN")  # read at call time, not import time
    with pytest.raises(MessagingError) as exc:
        await LineAdapter().send("U123", "hi")
    assert "LINE_CHANNEL_ACCESS_TOKEN is not set" in str(exc.value)
    assert "line send blocked" in str(exc.value)
    assert FAKE_TOKEN not in str(exc.value)


async def test_line_send_http_and_network_errors_hide_token(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", FAKE_TOKEN)

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": f"bad request for {FAKE_TOKEN}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError) as exc:
            await LineAdapter(client=client).send("U123", "hi")
    assert "HTTP 400" in str(exc.value)
    assert FAKE_TOKEN not in str(exc.value)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused while using {FAKE_TOKEN}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
        with pytest.raises(MessagingError) as exc:
            await LineAdapter(client=client).send("U123", "hi")
    assert "ConnectError" in str(exc.value)
    assert FAKE_TOKEN not in str(exc.value)


def test_line_record_blocked_and_configured(monkeypatch):
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
    adapter = LineAdapter()
    assert ALL_ADAPTERS["line"] is LineAdapter  # auto-discovery registered it

    record = adapter.record()
    assert record.status == "blocked"
    assert record.kind == "messaging"
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.config_ref == "LINE_CHANNEL_ACCESS_TOKEN,LINE_CHANNEL_SECRET"
    assert record.metadata["credential_env"] == ["LINE_CHANNEL_ACCESS_TOKEN"]
    assert record.metadata["verification_env"] == ["LINE_CHANNEL_SECRET"]
    assert record.metadata["live_verified"] is False
    assert "missing credentials" in record.metadata["reason"]

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", FAKE_TOKEN)
    record = adapter.record()
    assert record.status == "configured"
    assert record.metadata["reason"] is None
    assert FAKE_TOKEN not in json.dumps(record.model_dump())
