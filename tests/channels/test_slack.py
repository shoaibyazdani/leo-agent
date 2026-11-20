"""Offline Slack adapter tests: v0 request signing, event normalization, send, record.

No live network access: HTTP goes through ``httpx.MockTransport`` and the
signing secret / bot token are fake values whose absence from every error
message and record dump is explicitly asserted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx
import pytest

from leo.integrations.channels.slack import SlackAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

SIGNING_SECRET = "slack-signing-secret"
FAKE_TOKEN = "fake-slack-token-should-never-leak"


def _body(payload) -> bytes:
    """Byte-exact serialization matching the adapter's ``body_bytes`` for dicts."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _v0_signature(secret: str, timestamp: str, body: bytes) -> str:
    """Exactly Slack's documented scheme: ``v0=<hex hmac of v0:{ts}:{body}>``."""
    basestring = b"v0:" + timestamp.encode("utf-8") + b":" + body
    return "v0=" + hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()


def _signed_headers(payload, secret: str = SIGNING_SECRET, timestamp: str | None = None) -> dict[str, str]:
    ts = str(int(time.time())) if timestamp is None else timestamp
    return {
        "x-slack-signature": _v0_signature(secret, ts, _body(payload)),
        "x-slack-request-timestamp": ts,
    }


# ---------------------------------------------------------------------- verify


def test_slack_verify_accepts_only_correct_v0_signature(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    payload = {"type": "event_callback", "event": {"type": "message", "ts": "1"}}
    headers = _signed_headers(payload)
    assert adapter.verify_webhook(payload, headers, raw_body=_body(payload)) is True
    # Without a raw body the adapter verifies the canonical serialization.
    assert adapter.verify_webhook(payload, headers) is True
    assert adapter.verify_webhook(payload, _signed_headers(payload, secret="wrong-secret")) is False
    bogus = {
        "x-slack-signature": "v0=" + "00" * 32,
        "x-slack-request-timestamp": headers["x-slack-request-timestamp"],
    }
    assert adapter.verify_webhook(payload, bogus, raw_body=_body(payload)) is False


def test_slack_verify_rejects_missing_or_garbage_headers(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    payload = {"type": "event_callback"}
    body = _body(payload)
    ts = str(int(time.time()))
    signature = _v0_signature(SIGNING_SECRET, ts, body)
    assert adapter.verify_webhook(payload, {}, raw_body=body) is False
    assert adapter.verify_webhook(payload, None, raw_body=body) is False
    assert adapter.verify_webhook(payload, {"x-slack-signature": signature}, raw_body=body) is False
    assert adapter.verify_webhook(payload, {"x-slack-request-timestamp": ts}, raw_body=body) is False
    # A non-numeric timestamp can never be fresh.
    junk = {"x-slack-signature": signature, "x-slack-request-timestamp": "soon"}
    assert adapter.verify_webhook(payload, junk, raw_body=body) is False


def test_slack_verify_fails_closed_without_signing_secret(monkeypatch):
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    adapter = SlackAdapter()  # require_verification=True by default
    payload = {"type": "event_callback"}
    headers = _signed_headers(payload, secret="never-configured-secret")
    assert adapter.verify_webhook(payload, headers, raw_body=_body(payload)) is False


def test_slack_verify_rejects_stale_and_future_timestamps(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    payload = {"type": "event_callback"}
    body = _body(payload)
    stale = str(int(time.time()) - int(SlackAdapter.REPLAY_WINDOW) - 1)
    future = str(int(time.time()) + int(SlackAdapter.REPLAY_WINDOW) + 1)
    for ts in (stale, future):
        headers = _signed_headers(payload, timestamp=ts)
        # Correctly signed but outside the documented replay window.
        assert adapter.verify_webhook(payload, headers, raw_body=body) is False


# --------------------------------------------------------------- handle webhook


async def test_slack_event_callback_normalizes_threaded_reply(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    payload = {
        "type": "event_callback",
        "event_id": "Ev1",
        "event": {
            "type": "message",
            "user": "U123",
            "text": "hello from slack",
            "ts": "1700000000.000200",
            "thread_ts": "1700000000.000100",
            "client_msg_id": "client-1",
            "event_ts": "1700000000.000200",
        },
    }
    headers = _signed_headers(payload)
    messages = await adapter.handle_webhook(payload, headers, raw_body=_body(payload))
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("slack", "U123", "hello from slack")
    assert message.timestamp.startswith("2023-11-14")
    assert message.reply_ref == "1700000000.000100"
    assert message.message_id == "client-1"
    # The same envelope verifies over the canonical bytes when no raw body
    # reached the adapter.
    assert len(await adapter.handle_webhook(payload, headers)) == 1


async def test_slack_thread_root_has_no_reply_ref_and_falls_back_to_event_ts(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    payload = {
        "type": "event_callback",
        "event": {
            "type": "message",
            "user": "U9",
            "text": "thread root",
            "ts": "1700000000.000100",
            "thread_ts": "1700000000.000100",  # thread root: thread_ts == ts
            "event_ts": "1700000000.000100",
        },
    }
    headers = _signed_headers(payload)
    messages = await adapter.handle_webhook(payload, headers, raw_body=_body(payload))
    assert len(messages) == 1
    assert messages[0].reply_ref is None
    assert messages[0].message_id == "1700000000.000100"  # client_msg_id absent -> event_ts


async def test_slack_skips_bot_messages_url_verification_and_other_envelopes(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    envelopes = [
        {"type": "event_callback", "event": {"type": "message", "subtype": "bot_message", "bot_id": "B1", "text": "echo", "ts": "1"}},
        {"type": "event_callback", "event": {"type": "message", "bot_id": "B1", "text": "echo", "ts": "1"}},
        {"type": "event_callback", "event": {"type": "app_mention", "user": "U1", "text": "mention"}},
        {"type": "url_verification", "challenge": "must-be-echoed-by-the-route"},
        {"type": "event_callback", "event": "not-a-dict"},
        {"type": "event_callback"},
        {},
    ]
    for envelope in envelopes:
        headers = _signed_headers(envelope)
        assert await adapter.handle_webhook(envelope, headers, raw_body=_body(envelope)) == []


async def test_slack_malformed_bodies_never_raise(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    raw = b'["not", "an", "object"]'
    headers = {
        "x-slack-signature": _v0_signature(SIGNING_SECRET, str(int(time.time())), raw),
        "x-slack-request-timestamp": str(int(time.time())),
    }
    assert await adapter.handle_webhook(raw, headers, raw_body=raw) == []
    # A verified message event with hostile field types coerces instead of
    # raising; the timestamp header is fresh even though the event ts is junk.
    hostile = {
        "type": "event_callback",
        "event": {"type": "message", "ts": "garbage", "text": {"nested": "object"}, "user": None},
    }
    headers = _signed_headers(hostile)
    messages = await adapter.handle_webhook(hostile, headers, raw_body=_body(hostile))
    assert len(messages) == 1
    assert messages[0].sender == "unknown"
    assert messages[0].text == str({"nested": "object"})
    assert messages[0].timestamp  # from_epoch fell back to "now" for the junk ts


async def test_slack_handle_webhook_rejects_bad_signature(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    adapter = SlackAdapter()
    payload = {
        "type": "event_callback",
        "event": {"type": "message", "user": "U1", "text": "hi", "ts": "1700000000.000100"},
    }
    body = _body(payload)
    bad = {"x-slack-signature": "v0=" + "00" * 32, "x-slack-request-timestamp": str(int(time.time()))}
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, bad, raw_body=body)
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, None, raw_body=body)


# ------------------------------------------------------------------------ send


async def test_slack_send_posts_to_chat_postmessage(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", FAKE_TOKEN)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "ts": "1700000000.000100", "channel": "C123456"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SlackAdapter(client=client).send("C123456", "hello")
    assert result == {
        "channel": "slack",
        "target": "C123456",
        "success": True,
        "message_id": "1700000000.000100",
    }
    assert str(requests[0].url) == "https://slack.com/api/chat.postMessage"
    # Exact header assertion: any token leakage would be visible here on failure.
    assert requests[0].headers["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert json.loads(requests[0].content) == {"channel": "C123456", "text": "hello"}


async def test_slack_send_blocked_without_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(MessagingError, match="SLACK_BOT_TOKEN is not set") as exc:
        await SlackAdapter().send("C123456", "hi")
    assert FAKE_TOKEN not in str(exc.value)


async def test_slack_send_failures_hide_token_and_provider_text(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", FAKE_TOKEN)

    def http_400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"invalid_auth for {FAKE_TOKEN}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(http_400)) as client:
        with pytest.raises(MessagingError, match="slack send failed: HTTP 400") as exc:
            await SlackAdapter(client=client).send("C1", "hi")
    assert FAKE_TOKEN not in str(exc.value)

    def ok_false(request: httpx.Request) -> httpx.Response:
        # Slack answers HTTP 200 with ok=false; the "error" field is
        # provider data and must never reach an exception message.
        return httpx.Response(200, json={"ok": False, "error": f"not_authed {FAKE_TOKEN}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(ok_false)) as client:
        with pytest.raises(MessagingError, match="slack send failed: API reported failure") as exc:
            await SlackAdapter(client=client).send("C1", "hi")
    assert FAKE_TOKEN not in str(exc.value)

    def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(garbage)) as client:
        with pytest.raises(MessagingError, match="slack send failed: invalid response") as exc:
            await SlackAdapter(client=client).send("C1", "hi")
    assert FAKE_TOKEN not in str(exc.value)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection dropped near {FAKE_TOKEN}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
        with pytest.raises(MessagingError, match="slack send failed: ConnectError") as exc:
            await SlackAdapter(client=client).send("C1", "hi")
    assert FAKE_TOKEN not in str(exc.value)


# ---------------------------------------------------------------------- record


def test_slack_record_blocked_and_configured(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    record = SlackAdapter().record()
    assert (record.name, record.kind) == ("slack", "messaging")
    assert record.status == "blocked"
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.config_ref == "SLACK_BOT_TOKEN,SLACK_SIGNING_SECRET"
    assert record.metadata["credential_env"] == ["SLACK_BOT_TOKEN"]
    assert record.metadata["verification_env"] == ["SLACK_SIGNING_SECRET"]
    assert record.metadata["live_verified"] is False
    assert FAKE_TOKEN not in json.dumps(record.model_dump())

    monkeypatch.setenv("SLACK_BOT_TOKEN", FAKE_TOKEN)
    configured = SlackAdapter().record()
    assert configured.status == "configured"
    assert configured.config_ref == "SLACK_BOT_TOKEN,SLACK_SIGNING_SECRET"
    assert FAKE_TOKEN not in json.dumps(configured.model_dump())
