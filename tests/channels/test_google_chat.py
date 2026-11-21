"""Offline verification for the Google Chat channel adapter.

No live network calls: HTTP goes through httpx.MockTransport and webhook
verification uses the app verification token Google Chat presents.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from leo.integrations.channels import ALL_ADAPTERS
from leo.integrations.channels.google_chat import GoogleChatAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

FAKE_KEY = "fake-gchat-key-should-never-leak"
WEBHOOK_URL = f"https://chat.googleapis.com/v1/spaces/SPACE/messages?key={FAKE_KEY}"


async def test_google_chat_token_verification(monkeypatch):
    monkeypatch.setenv("GOOGLE_CHAT_VERIFICATION_TOKEN", "gchat-token")
    adapter = GoogleChatAdapter()
    payload = {"event": {}}
    assert adapter.verify_webhook(payload, {"Authorization": "Bearer gchat-token"}) is True
    assert adapter.verify_webhook(payload, {"X-Goog-Chat-Token": "gchat-token"}) is True
    assert adapter.verify_webhook(payload, {"Authorization": "Bearer wrong"}) is False
    assert adapter.verify_webhook(payload, {"X-Goog-Chat-Token": "wrong"}) is False
    # The value without the Bearer scheme is not the configured presentation.
    assert adapter.verify_webhook(payload, {"Authorization": "gchat-token"}) is False
    assert adapter.verify_webhook(payload, {}) is False

    # Fail closed when the verification token env is absent.
    monkeypatch.delenv("GOOGLE_CHAT_VERIFICATION_TOKEN")
    assert adapter.verify_webhook(payload, {"Authorization": "Bearer gchat-token"}) is False


async def test_google_chat_handle_webhook_event_and_body_shapes(monkeypatch):
    monkeypatch.setenv("GOOGLE_CHAT_VERIFICATION_TOKEN", "gchat-token")
    adapter = GoogleChatAdapter()
    headers = {"Authorization": "Bearer gchat-token"}

    event_payload = {
        "event": {
            "user": {"name": "users/123", "displayName": "Alice", "type": "HUMAN"},
            "message": {
                "name": "spaces/AAA/messages/MSG1",
                "sender": {"name": "users/123", "type": "HUMAN"},
                "createTime": "2024-06-01T10:00:00Z",
                "text": "hello chat",
                "thread": {"name": "spaces/AAA/threads/T1"},
            },
        }
    }
    messages = await adapter.handle_webhook(event_payload, headers)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("google_chat", "Alice", "hello chat")
    assert message.message_id == "spaces/AAA/messages/MSG1"
    assert message.timestamp == "2024-06-01T10:00:00Z"
    assert message.reply_ref is None

    # Bare message-body shape: thread.name serves as the message reference
    # when the body carries no message name.
    body_payload = {
        "sender": {"name": "users/456", "type": "HUMAN"},
        "text": "direct shape",
        "createTime": "2024-06-01T11:00:00Z",
        "thread": {"name": "spaces/BBB/threads/T2"},
    }
    messages = await adapter.handle_webhook(body_payload, headers)
    assert len(messages) == 1
    assert (messages[0].sender, messages[0].text) == ("users/456", "direct shape")
    assert messages[0].message_id == "spaces/BBB/threads/T2"
    assert messages[0].timestamp == "2024-06-01T11:00:00Z"

    # Bot-authored messages are skipped in both shapes.
    bot_event = {"event": {"user": {"type": "HUMAN"}, "message": {"text": "beep", "sender": {"type": "BOT"}}}}
    assert await adapter.handle_webhook(bot_event, headers) == []
    app_user_event = {"event": {"user": {"type": "BOT"}, "message": {"text": "boop", "sender": {"type": "HUMAN"}}}}
    assert await adapter.handle_webhook(app_user_event, headers) == []
    assert await adapter.handle_webhook({"sender": {"type": "BOT"}, "text": "bzzt"}, headers) == []

    # Unsigned deliveries never reach the parser.
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(event_payload, {})
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(event_payload, {"Authorization": "Bearer wrong"})


async def test_google_chat_handle_webhook_malformed_payload_yields_nothing(monkeypatch):
    monkeypatch.setenv("GOOGLE_CHAT_VERIFICATION_TOKEN", "gchat-token")
    adapter = GoogleChatAdapter()
    headers = {"Authorization": "Bearer gchat-token"}
    cases: list[Any] = [42, "not json", b"[", {"event": "nope"}, {"event": {"message": "nope"}}, {"event": {"user": "x"}}, {"nope": 1}]
    for payload in cases:
        assert await adapter.handle_webhook(payload, headers) == []


async def test_google_chat_send_posts_to_env_webhook(monkeypatch):
    monkeypatch.setenv("GOOGLE_CHAT_WEBHOOK_URL", WEBHOOK_URL)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"name": "spaces/SPACE/messages/MSG9"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = GoogleChatAdapter(client=client)
        # The URL is space-scoped: target does not change the destination.
        result = await adapter.send("ignored-target", "hello")
    assert result == {
        "channel": "google_chat",
        "target": "ignored-target",
        "success": True,
        "message_id": "spaces/SPACE/messages/MSG9",
    }
    request = requests[0]
    assert str(request.url) == WEBHOOK_URL
    assert json.loads(request.content) == {"text": "hello"}


async def test_google_chat_send_rejects_invalid_webhook_urls(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    bad_urls = (
        f"http://chat.googleapis.com/v1/spaces/S/messages?key={FAKE_KEY}",  # not https
        "https://evil.example/v1/spaces/S/messages",
        "https://chat.googleapis.com/v1/spaces/S/other",  # no /messages in path
        "https://chat.googleapis.com/webhook",
        f"https://user:pass@chat.googleapis.com/v1/spaces/S/messages?key={FAKE_KEY}",  # userinfo
    )
    for bad in bad_urls:
        monkeypatch.setenv("GOOGLE_CHAT_WEBHOOK_URL", bad)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(MessagingError, match="not a Google Chat webhook URL"):
                await GoogleChatAdapter(client=client).send("t", "hi")

    # The URL embeds its key parameter: the rejection never echoes it.
    monkeypatch.setenv("GOOGLE_CHAT_WEBHOOK_URL", f"https://evil.example/hook?key={FAKE_KEY}")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MessagingError) as exc:
            await GoogleChatAdapter(client=client).send("t", "hi")
    assert "not a Google Chat webhook URL" in str(exc.value)
    assert FAKE_KEY not in str(exc.value)


async def test_google_chat_send_blocked_and_http_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_CHAT_WEBHOOK_URL", raising=False)
    with pytest.raises(MessagingError) as exc:
        await GoogleChatAdapter().send("t", "hi")
    assert "GOOGLE_CHAT_WEBHOOK_URL is not set" in str(exc.value)
    assert "google_chat send blocked" in str(exc.value)

    monkeypatch.setenv("GOOGLE_CHAT_WEBHOOK_URL", WEBHOOK_URL)

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": f"bad {FAKE_KEY}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError) as exc:
            await GoogleChatAdapter(client=client).send("t", "hi")
    assert "HTTP 400" in str(exc.value)
    assert FAKE_KEY not in str(exc.value)


def test_google_chat_record_blocked_and_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_CHAT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("GOOGLE_CHAT_VERIFICATION_TOKEN", raising=False)
    adapter = GoogleChatAdapter()
    assert ALL_ADAPTERS["google_chat"] is GoogleChatAdapter  # auto-discovery registered it

    record = adapter.record()
    assert record.status == "blocked"
    assert record.kind == "messaging"
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.config_ref == "GOOGLE_CHAT_WEBHOOK_URL,GOOGLE_CHAT_VERIFICATION_TOKEN"
    assert record.metadata["verification_env"] == ["GOOGLE_CHAT_VERIFICATION_TOKEN"]
    assert record.metadata["live_verified"] is False
    assert "missing credentials" in record.metadata["reason"]

    monkeypatch.setenv("GOOGLE_CHAT_WEBHOOK_URL", WEBHOOK_URL)
    record = adapter.record()
    assert record.status == "configured"
    assert record.metadata["reason"] is None
    assert FAKE_KEY not in json.dumps(record.model_dump())
