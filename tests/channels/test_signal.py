"""Offline verification for the Signal channel adapter.

signal-cli-rest-api is a community bridge whose webhooks are NOT signed by
default — verification here is LEO's own fail-closed HMAC convention. No
live network calls: HTTP goes through ``httpx.MockTransport`` and every
credential is a fake value that must never surface in errors or records.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from leo.integrations.channels.signal import SignalAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

ENV_NAMES = (
    "SIGNAL_CLI_URL",
    "SIGNAL_PHONE_NUMBER",
    "SIGNAL_CLI_USER",
    "SIGNAL_CLI_PASSWORD",
    "SIGNAL_WEBHOOK_SECRET",
)
FAKE_PASSWORD = "fake-signal-password-never-leak"
SIGNAL_URL = "https://signal.example.org"
SECRET = "signal-webhook-secret"


@pytest.fixture(autouse=True)
def clean_signal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _signature(secret: str, raw: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _raw(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _signed_headers(payload: Any, secret: str = SECRET) -> dict[str, str]:
    return {"X-Leo-Signature": _signature(secret, _raw(payload))}


def _configured(monkeypatch: pytest.MonkeyPatch, *, url: str = SIGNAL_URL) -> None:
    monkeypatch.setenv("SIGNAL_CLI_URL", url)
    monkeypatch.setenv("SIGNAL_PHONE_NUMBER", "+15550001")


# ------------------------------------------------------------------- verify


async def test_signal_verify_webhook_leo_hmac_convention(monkeypatch):
    monkeypatch.setenv("SIGNAL_WEBHOOK_SECRET", SECRET)
    adapter = SignalAdapter()
    raw = b'{"source":"+15551234567","message":"hi"}'
    assert adapter.verify_webhook(raw, {"X-Leo-Signature": _signature(SECRET, raw)}, raw_body=raw) is True
    assert adapter.verify_webhook(raw, {"X-Leo-Signature": "sha256=" + "00" * 32}, raw_body=raw) is False
    assert adapter.verify_webhook(raw, {}, raw_body=raw) is False
    payload = {"message": "hi"}
    assert adapter.verify_webhook(payload, {"x-leo-signature": _signature(SECRET, _raw(payload))}) is True


async def test_signal_verify_fails_closed_without_secret(monkeypatch):
    assert SignalAdapter().verify_webhook({}, {}) is False
    # The escape hatch is explicit and local to the adapter instance.
    assert SignalAdapter(require_verification=False).verify_webhook({}, {}) is True


# ------------------------------------------------------------------ webhook


async def test_signal_handle_webhook_normalizes_message(monkeypatch):
    monkeypatch.setenv("SIGNAL_WEBHOOK_SECRET", SECRET)
    adapter = SignalAdapter()
    payload = {
        "source": "+15551234567",
        "message": "hello signal",
        "timestamp": 1700000000000,
        "id": "sig-msg-1",
    }
    raw = _raw(payload)
    headers = {"X-Leo-Signature": _signature(SECRET, raw)}
    messages = await adapter.handle_webhook(payload, headers, raw_body=raw)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("signal", "+15551234567", "hello signal")
    assert message.message_id == "sig-msg-1"
    assert message.timestamp.startswith("2023-11-14")
    assert message.reply_ref is None

    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"X-Leo-Signature": "sha256=" + "00" * 32}, raw_body=raw)
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {}, raw_body=raw)


async def test_signal_handle_webhook_sender_and_group_forms(monkeypatch):
    monkeypatch.setenv("SIGNAL_WEBHOOK_SECRET", SECRET)
    adapter = SignalAdapter()
    cases = [
        # group message with an individual sender
        (
            {
                "sender": {"number": "+1555999"},
                "message": "group hi",
                "groupInfo": {"groupId": "group.g1="},
                "messageId": "sig-2",
                "timestamp": 1700000000000,
            },
            "+1555999",
            "sig-2",
        ),
        # source as an object
        ({"source": {"number": "+1555222"}, "message": "dict source", "id": "sig-3"}, "+1555222", "sig-3"),
        # relay stripped the individual sender: the group id keeps admission possible
        ({"groupInfo": {"groupId": "group.g7="}, "message": "anon group", "timestamp": 1700000000000}, "group.g7=", None),
    ]
    for payload, expected_sender, expected_id in cases:
        messages = await adapter.handle_webhook(payload, _signed_headers(payload))
        assert len(messages) == 1
        assert messages[0].sender == expected_sender
        assert messages[0].message_id == expected_id


async def test_signal_handle_webhook_reads_only_textual_fields(monkeypatch):
    monkeypatch.setenv("SIGNAL_WEBHOOK_SECRET", SECRET)
    adapter = SignalAdapter()
    # An attachment's caption is text and is used; its media is never touched.
    payload = {
        "source": "+1555333",
        "body": {"attachments": [{"contentType": "image/png", "text": "look at this"}]},
        "timestamp": 1700000000000,
        "id": "sig-4",
    }
    messages = await adapter.handle_webhook(payload, _signed_headers(payload))
    assert [(m.sender, m.text, m.message_id) for m in messages] == [("+1555333", "look at this", "sig-4")]

    for bad in (
        {"source": "+1555333", "timestamp": 1700000000000},  # no text at all
        {"source": "+1555333", "message": {"nested": "dict"}},  # message is not text
        {"type": "receipt", "source": "+1555333"},  # non-message envelope
    ):
        assert await adapter.handle_webhook(bad, _signed_headers(bad)) == []
    # Unparseable bodies are data, not crashes.
    raw = b"not json"
    assert await adapter.handle_webhook(raw, {"X-Leo-Signature": _signature(SECRET, raw)}, raw_body=raw) == []
    raw = b'{"message": "x", "timestamp": "not-a-number"}'  # bad timestamp falls back to now
    messages = await adapter.handle_webhook(raw, {"X-Leo-Signature": _signature(SECRET, raw)}, raw_body=raw)
    assert len(messages) == 1 and messages[0].timestamp.endswith("+00:00")


# --------------------------------------------------------------------- send


async def test_signal_send_posts_v2_body_with_optional_basic_auth(monkeypatch):
    _configured(monkeypatch)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"timestamp": "1700000000000"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = SignalAdapter(client=client)
        result = await adapter.send("+15559876", "hello signal")
        # Group ids are valid recipients too (defensive: non-empty, no spaces).
        group_result = await adapter.send("group.g123=", "hello group")
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{SIGNAL_URL}/v2/send"
    assert json.loads(request.content) == {
        "message": "hello signal",
        "recipients": ["+15559876"],
        "number": "+15550001",
    }
    assert "authorization" not in request.headers
    assert result == {"channel": "signal", "target": "+15559876", "success": True, "message_id": "1700000000000"}
    assert group_result["target"] == "group.g123="


async def test_signal_send_basic_auth_from_env(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("SIGNAL_CLI_USER", "signal-user")
    monkeypatch.setenv("SIGNAL_CLI_PASSWORD", FAKE_PASSWORD)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SignalAdapter(client=client).send("+15559876", "hi")
    expected = "Basic " + base64.b64encode(f"signal-user:{FAKE_PASSWORD}".encode()).decode("ascii")
    assert requests[0].headers["authorization"] == expected
    assert result == {"channel": "signal", "target": "+15559876", "success": True, "message_id": None}


async def test_signal_send_blocked_without_envs(monkeypatch):
    with pytest.raises(MessagingError, match="signal send blocked: SIGNAL_CLI_URL is not set"):
        await SignalAdapter().send("+15559876", "hi")
    monkeypatch.setenv("SIGNAL_CLI_URL", SIGNAL_URL)
    with pytest.raises(MessagingError, match="signal send blocked: SIGNAL_PHONE_NUMBER is not set"):
        await SignalAdapter().send("+15559876", "hi")


async def test_signal_send_rejects_invalid_url_and_recipient(monkeypatch):
    _configured(monkeypatch, url="not a url")
    with pytest.raises(MessagingError, match="SIGNAL_CLI_URL is not a valid"):
        await SignalAdapter().send("+15559876", "hi")
    _configured(monkeypatch)
    adapter = SignalAdapter()
    for bad in ("+1555 9876", ""):
        with pytest.raises(MessagingError, match="invalid Signal recipient"):
            await adapter.send(bad, "hi")


async def test_signal_send_failures_report_no_secrets(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("SIGNAL_CLI_USER", "signal-user")
    monkeypatch.setenv("SIGNAL_CLI_PASSWORD", FAKE_PASSWORD)

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "unregistered recipient"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError, match="signal send failed: HTTP 400") as exc:
            await SignalAdapter(client=client).send("+15559876", "hi")
    assert FAKE_PASSWORD not in str(exc.value)

    def exploding(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout(f"dial {SIGNAL_URL} as signal-user:{FAKE_PASSWORD}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(exploding)) as client:
        with pytest.raises(MessagingError, match="signal send failed: ConnectTimeout") as exc:
            await SignalAdapter(client=client).send("+15559876", "hi")
    assert FAKE_PASSWORD not in str(exc.value)
    assert "dial" not in str(exc.value)


# ------------------------------------------------------------------- record


async def test_signal_record_blocked_configured_and_capabilities(monkeypatch):
    adapter = SignalAdapter()
    record = adapter.record()
    assert record.status == "blocked"
    assert record.capabilities == ["send", "webhook"]  # no pull transport
    assert record.verification == "mock-tested"
    assert record.metadata["credential_env"] == ["SIGNAL_CLI_URL", "SIGNAL_PHONE_NUMBER"]
    assert record.metadata["verification_env"] == ["SIGNAL_WEBHOOK_SECRET"]

    _configured(monkeypatch)
    monkeypatch.setenv("SIGNAL_WEBHOOK_SECRET", SECRET)
    record = SignalAdapter().record()
    assert record.status == "configured"
    assert record.config_ref == "SIGNAL_CLI_URL,SIGNAL_PHONE_NUMBER,SIGNAL_WEBHOOK_SECRET"
    assert record.metadata["live_verified"] is False
    assert FAKE_PASSWORD not in json.dumps(record.model_dump())
