"""Offline verification for the relay channel adapter (LEO's generic transport).

No live network calls: HTTP goes through httpx.MockTransport, and the shared
secret is a fake value that must never leak into records or errors. The
round-trip test proves the outbound signature and the inbound verifier agree
byte-for-byte over the exact wire content.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime

import httpx
import pytest

from leo.integrations.channels.relay import RelayAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

SECRET = "fake-relay-secret-should-never-leak"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def test_round_trip_signature_and_envelope_agree_byte_for_byte(monkeypatch):
    monkeypatch.setenv("RELAY_TARGET_URL", "https://relay.example/inbox")
    monkeypatch.setenv("RELAY_SECRET", SECRET)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    async with _client(handler) as client:
        sender = RelayAdapter(client=client)
        result = await sender.send("channel-9", "round trip hello")

    request = requests[0]
    assert str(request.url) == "https://relay.example/inbox"
    assert request.headers["content-type"] == "application/json"
    captured = request.content

    # Recompute the HMAC in the test over the CAPTURED bytes: the adapter's
    # signature must match an independent computation of the wire content.
    signature = _sign(SECRET, captured)
    assert request.headers["x-leo-signature"] == signature

    envelope = json.loads(captured)
    assert envelope["sender"] == "leo"
    assert envelope["text"] == "round trip hello"
    assert envelope["target"] == "channel-9"
    assert envelope["sent_at"]
    assert envelope["message_id"] == result["message_id"]

    # Feed the same bytes to a fresh receiving adapter sharing the secret.
    receiver = RelayAdapter()
    messages = await receiver.handle_webhook(envelope, {"X-Leo-Signature": signature}, raw_body=captured)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("relay", "leo", "round trip hello")
    assert message.message_id == result["message_id"]
    assert message.timestamp == envelope["sent_at"]


async def test_verify_webhook_rejects_wrong_and_missing_signatures(monkeypatch):
    monkeypatch.setenv("RELAY_SECRET", SECRET)
    adapter = RelayAdapter()
    body = b'{"sender":"leo","text":"hi"}'

    assert adapter.verify_webhook(body, {"X-Leo-Signature": _sign(SECRET, body)}, raw_body=body) is True
    assert adapter.verify_webhook(body, {"X-Leo-Signature": _sign("other-secret", body)}, raw_body=body) is False
    assert adapter.verify_webhook(body, {}, raw_body=body) is False
    assert adapter.verify_webhook(body, None, raw_body=body) is False

    # Fail closed: without the shared secret nothing verifies, even a
    # correctly computed signature for the missing secret's value.
    monkeypatch.delenv("RELAY_SECRET", raising=False)
    receiver = RelayAdapter(require_verification=True)
    assert receiver.verify_webhook(body, {"X-Leo-Signature": _sign(SECRET, body)}, raw_body=body) is False


async def test_handle_webhook_raises_on_bad_signature(monkeypatch):
    monkeypatch.setenv("RELAY_SECRET", SECRET)
    adapter = RelayAdapter()
    payload = {"sender": "leo", "text": "hi"}
    raw = b'{"sender":"leo","text":"hi"}'
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"X-Leo-Signature": "sha256=" + "0" * 64}, raw_body=raw)
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {}, raw_body=raw)


async def test_handle_webhook_skips_malformed_and_caps_fields(monkeypatch):
    monkeypatch.setenv("RELAY_SECRET", SECRET)
    adapter = RelayAdapter()

    async def deliver(raw: bytes, payload) -> list[NormalizedMessage]:
        headers = {"X-Leo-Signature": _sign(SECRET, raw)}
        return await adapter.handle_webhook(payload, headers, raw_body=raw)

    # Malformed but properly signed envelopes are skipped, never fatal.
    assert await deliver(b"[1,2,3]", [1, 2, 3]) == []
    assert await deliver(b"not-json{{{", "not-json{{{") == []
    assert await deliver(b'{"sender":"x"}', {"sender": "x"}) == []
    assert await deliver(b'{"text":{"nested":true}}', {"text": {"nested": True}}) == []
    assert await deliver(b'{"text":42}', {"text": 42}) == []

    # Oversized text is capped at 10,000; long sender names at 200.
    long_text = "T" * 12_000
    long_sender = "S" * 300
    raw = json.dumps({"sender": long_sender, "text": long_text}).encode()
    messages = await deliver(raw, json.loads(raw))
    assert len(messages) == 1
    assert messages[0].text == long_text[:10_000]
    assert messages[0].sender == long_sender[:200]

    # Missing sender falls back to "unknown"; refs pass through as strings.
    raw = json.dumps({"text": "ping", "message_id": "m-1", "reply_ref": 7}).encode()
    messages = await deliver(raw, json.loads(raw))
    assert len(messages) == 1
    assert (messages[0].sender, messages[0].text) == ("unknown", "ping")
    assert messages[0].reply_ref == "7" and messages[0].message_id == "m-1"

    # Timestamps: epoch and ISO values parse; garbage falls back to now.
    raw = json.dumps({"text": "t", "timestamp": 1700000000}).encode()
    messages = await deliver(raw, json.loads(raw))
    assert messages[0].timestamp.startswith("2023-11-14")

    raw = json.dumps({"text": "t", "sent_at": "2024-01-01T00:00:00Z"}).encode()
    messages = await deliver(raw, json.loads(raw))
    assert messages[0].timestamp.startswith("2024-01-01")

    raw = json.dumps({"text": "t", "sent_at": "not-a-date"}).encode()
    messages = await deliver(raw, json.loads(raw))
    datetime.fromisoformat(messages[0].timestamp)  # still a valid ISO timestamp


def test_record_capabilities_and_config_ref_without_secret_values(monkeypatch):
    monkeypatch.setenv("RELAY_TARGET_URL", "https://relay.example/inbox")
    monkeypatch.setenv("RELAY_SECRET", SECRET)
    configured = RelayAdapter().record()
    assert configured.status == "configured"
    assert configured.capabilities == ["send", "webhook"]
    assert "RELAY_TARGET_URL" in configured.config_ref
    assert "RELAY_SECRET" in configured.config_ref
    assert configured.metadata["verification_env"] == ["RELAY_SECRET"]
    assert SECRET not in json.dumps(configured.model_dump())

    monkeypatch.delenv("RELAY_TARGET_URL", raising=False)
    monkeypatch.delenv("RELAY_SECRET", raising=False)
    blocked = RelayAdapter().record()
    assert blocked.status == "blocked"
    assert blocked.capabilities == ["send", "webhook"]
    assert "RELAY_TARGET_URL" in blocked.config_ref
    assert blocked.metadata["reason"] == "missing credentials: RELAY_TARGET_URL, RELAY_SECRET"


async def test_send_failures_never_leak_the_secret(monkeypatch):
    monkeypatch.setenv("RELAY_TARGET_URL", "https://relay.example/inbox")
    monkeypatch.setenv("RELAY_SECRET", SECRET)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with _client(handler) as client:
        with pytest.raises(MessagingError) as exc:
            await RelayAdapter(client=client).send("chan", "hi")
    assert "HTTP 500" in str(exc.value)
    assert SECRET not in str(exc.value)


async def test_send_blocked_when_either_env_is_missing(monkeypatch):
    monkeypatch.delenv("RELAY_TARGET_URL", raising=False)
    monkeypatch.setenv("RELAY_SECRET", SECRET)
    with pytest.raises(MessagingError, match="RELAY_TARGET_URL is not set") as exc:
        await RelayAdapter().send("chan", "hi")
    assert SECRET not in str(exc.value)

    monkeypatch.setenv("RELAY_TARGET_URL", "https://relay.example/inbox")
    monkeypatch.delenv("RELAY_SECRET", raising=False)
    with pytest.raises(MessagingError, match="RELAY_SECRET is not set"):
        await RelayAdapter().send("chan", "hi")


async def test_send_rejects_non_http_target_urls(monkeypatch):
    monkeypatch.setenv("RELAY_SECRET", SECRET)
    for bad in ("ftp://relay.example", "file:///etc/passwd", "not a url"):
        monkeypatch.setenv("RELAY_TARGET_URL", bad)
        with pytest.raises(MessagingError, match="not a valid http"):
            await RelayAdapter().send("chan", "hi")


async def test_send_wraps_transport_errors_without_echoing_them(monkeypatch):
    monkeypatch.setenv("RELAY_TARGET_URL", "https://relay.example/inbox")
    monkeypatch.setenv("RELAY_SECRET", SECRET)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused by relay")

    async with _client(handler) as client:
        with pytest.raises(MessagingError, match="ConnectError") as exc:
            await RelayAdapter(client=client).send("chan", "hi")
    assert "connection refused by relay" not in str(exc.value)
