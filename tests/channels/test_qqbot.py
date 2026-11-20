"""Offline verification for the QQ official bot channel adapter.

No live QQ endpoint is contacted: HTTP goes through httpx.MockTransport, the
Ed25519 signature check runs through an injected verifier (the same injectable
pattern as the Discord adapter's test), and the token-cache clock is an
injectable ``now_fn``. No secret value ever appears in an error or a record.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from leo.integrations.channels import adapter_classes, polling_channels, webhook_channels
from leo.integrations.channels.qqbot import QQBotAdapter
from leo.integrations.messaging import (
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
)

APP_ID = "fake-qq-app-id"
APP_SECRET = "fake-qq-app-secret"
ACCESS_TOKEN = "fake-qq-access-token"
PUBLIC_KEY = "ab" * 32
GOOD_SIGNATURE = "cd" * 32
TIMESTAMP = str(int(time.time()))  # fresh for the webhook freshness window


def fake_verifier(seen: list | None = None):
    def verifier(public_key: str, signature: str, timestamp: str, body: bytes) -> bool:
        if seen is not None:
            seen.append((public_key, signature, timestamp, body))
        return signature == GOOD_SIGNATURE

    return verifier


SIGNED_HEADERS = {"X-Signature-Ed25519": GOOD_SIGNATURE, "X-Signature-Timestamp": TIMESTAMP}

GROUP_EVENT = {
    "event_type": "GROUP_AT_MESSAGE_CREATE",
    "event": {
        "group_openid": "group-openid-1",
        "member_openid": "member-9",
        "message_id": "qq-msg-7",
        "content": "<@!robot> hello qq",
    },
}


async def test_verify_webhook_uses_injectable_verifier(monkeypatch):
    monkeypatch.setenv("QQBOT_PUBLIC_KEY", PUBLIC_KEY)
    seen: list[tuple[str, str, str, bytes]] = []
    adapter = QQBotAdapter(signature_verifier=fake_verifier(seen))
    payload = {"event_type": "GROUP_AT_MESSAGE_CREATE", "event": {"content": "hi"}}
    assert adapter.verify_webhook(payload, SIGNED_HEADERS) is True
    assert seen[0][0] == PUBLIC_KEY
    assert seen[0][3] == json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    bad = {"X-Signature-Ed25519": "ef" * 32, "X-Signature-Timestamp": "1"}
    assert adapter.verify_webhook(payload, bad) is False
    assert adapter.verify_webhook(payload, {"X-Signature-Ed25519": GOOD_SIGNATURE}) is False
    assert adapter.verify_webhook(payload, {}) is False
    assert adapter.verify_webhook(payload, None) is False


async def test_verify_webhook_fails_closed_without_key_env(monkeypatch):
    monkeypatch.delenv("QQBOT_PUBLIC_KEY", raising=False)
    assert QQBotAdapter(signature_verifier=fake_verifier()).verify_webhook(GROUP_EVENT, SIGNED_HEADERS) is False
    assert QQBotAdapter(signature_verifier=fake_verifier(), require_verification=False).verify_webhook(
        GROUP_EVENT, SIGNED_HEADERS
    ) is True


async def test_handle_webhook_normalizes_group_event(monkeypatch):
    monkeypatch.setenv("QQBOT_PUBLIC_KEY", PUBLIC_KEY)
    adapter = QQBotAdapter(signature_verifier=fake_verifier())
    messages = await adapter.handle_webhook(GROUP_EVENT, SIGNED_HEADERS)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("qqbot", "member-9", "hello qq")
    assert message.message_id == "qq-msg-7"
    assert message.reply_ref is None
    assert "T" in message.timestamp  # arrival time: QQ callbacks carry no uniform timestamp


async def test_handle_webhook_envelope_and_direct_shapes(monkeypatch):
    monkeypatch.setenv("QQBOT_PUBLIC_KEY", PUBLIC_KEY)
    adapter = QQBotAdapter(signature_verifier=fake_verifier())

    gateway_envelope = {
        "op": 0,
        "id": "event-id-1",
        "t": "GROUP_AT_MESSAGE_CREATE",
        "d": {"group_openid": "g", "member_openid": "member-d", "content": "@bot from gateway", "message_id": "qq-11"},
    }
    messages = await adapter.handle_webhook(gateway_envelope, SIGNED_HEADERS)
    assert (messages[0].sender, messages[0].text, messages[0].message_id) == ("member-d", "from gateway", "qq-11")

    direct = {"group_openid": "g", "member_openid": "member-direct", "content": "plain object", "id": "qq-12"}
    messages = await adapter.handle_webhook(direct, SIGNED_HEADERS)
    assert (messages[0].sender, messages[0].text, messages[0].message_id) == ("member-direct", "plain object", "qq-12")

    c2c = {
        "event_type": "C2C_MESSAGE_CREATE",
        "event": {"content": "direct hi", "author": {"member_openid": "member-c2c"}, "message_id": "qq-c2c-1"},
    }
    messages = await adapter.handle_webhook(c2c, SIGNED_HEADERS)
    assert (messages[0].sender, messages[0].text, messages[0].message_id) == ("member-c2c", "direct hi", "qq-c2c-1")


async def test_handle_webhook_strips_leading_bot_mention(monkeypatch):
    monkeypatch.setenv("QQBOT_PUBLIC_KEY", PUBLIC_KEY)
    adapter = QQBotAdapter(signature_verifier=fake_verifier())
    cases = {
        "<@!123> hi there": "hi there",  # angle-form mention
        "@bot hello": "hello",  # plain mention followed by text
        "@botname": "@botname",  # no trailing space: a literal word, not a mention
        "hello": "hello",  # untouched
    }
    for content, expected in cases.items():
        payload = {"event": {"group_openid": "g", "member_openid": "m", "content": content}}
        messages = await adapter.handle_webhook(payload, SIGNED_HEADERS)
        assert messages[0].text == expected


async def test_handle_webhook_skip_rules_and_malformed_payloads(monkeypatch):
    monkeypatch.setenv("QQBOT_PUBLIC_KEY", PUBLIC_KEY)
    adapter = QQBotAdapter(signature_verifier=fake_verifier())
    skipped = (
        {"event": {"group_openid": "g", "member_openid": "m", "content": "   "}},
        {"event": {"group_openid": "g", "member_openid": "m", "content": None}},
        {"event": "not-a-dict"},
        {"d": "not-a-dict"},
        {},
    )
    for payload in skipped:
        assert await adapter.handle_webhook(payload, SIGNED_HEADERS) == []
    for malformed in ([1, 2], "not json", None, 42):
        assert await adapter.handle_webhook(malformed, SIGNED_HEADERS) == []


async def test_handle_webhook_rejects_bad_signature(monkeypatch):
    monkeypatch.setenv("QQBOT_PUBLIC_KEY", PUBLIC_KEY)
    adapter = QQBotAdapter(signature_verifier=fake_verifier())
    bad = {"X-Signature-Ed25519": "ef" * 32, "X-Signature-Timestamp": "1"}
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(GROUP_EVENT, bad)
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(GROUP_EVENT, {})


async def test_send_group_and_c2c_with_cached_token(monkeypatch):
    monkeypatch.setenv("QQBOT_APP_ID", APP_ID)
    monkeypatch.setenv("QQBOT_APP_SECRET", APP_SECRET)
    requests: list[httpx.Request] = []
    token_hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/app/getAppAccessToken":
            token_hits.append(request)
            return httpx.Response(200, json={"access_token": ACCESS_TOKEN, "expires_in": 7200})
        return httpx.Response(200, json={"message_id": "qq-out-1"})

    clock = {"now": 500.0}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = QQBotAdapter(client=client, now_fn=lambda: clock["now"])
        result = await adapter.send("group:group-openid-1", "hi")
        assert result == {
            "channel": "qqbot",
            "target": "group:group-openid-1",
            "success": True,
            "message_id": "qq-out-1",
        }
        token_request = token_hits[0]
        assert token_request.method == "GET"
        assert token_request.url.host == "api.sgroup.qq.com"
        assert dict(token_request.url.params) == {"appId": APP_ID, "clientSecret": APP_SECRET}

        message_request = requests[1]
        assert message_request.url.path == "/v2/groups/group-openid-1/messages"
        assert message_request.headers["authorization"] == f"QQBot {ACCESS_TOKEN}"
        assert json.loads(message_request.content) == {"content": "hi", "msg_type": 0}

        await adapter.send("c2c:user-openid-2", "yo")
        assert requests[2].url.path == "/v2/c2c/user-openid-2/messages"
        assert len(token_hits) == 1  # token cached across both sends

        clock["now"] += 7200.0
        await adapter.send("group:group-openid-1", "again")
        assert len(token_hits) == 2  # past expiry: refetched exactly once


async def test_send_target_format_and_path_safety(monkeypatch):
    monkeypatch.setenv("QQBOT_APP_ID", APP_ID)
    monkeypatch.setenv("QQBOT_APP_SECRET", APP_SECRET)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/app/getAppAccessToken":
            return httpx.Response(200, json={"access_token": ACCESS_TOKEN, "expires_in": 7200})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = QQBotAdapter(client=client, now_fn=lambda: 1000.0)
        with pytest.raises(MessagingError, match="target must be"):
            await adapter.send("plain-openid", "hi")  # no group:/c2c: prefix
        with pytest.raises(MessagingError, match="target must be"):
            await adapter.send("group:", "hi")  # empty openid
        assert requests == []  # rejected before any network call
        await adapter.send("group:weird/openid", "hi")  # openid is path-quoted
        assert requests[0].url.path == "/app/getAppAccessToken"
        assert requests[1].url.raw_path == b"/v2/groups/weird%2Fopenid/messages"


async def test_send_failures_and_env_blocks(monkeypatch):
    monkeypatch.setenv("QQBOT_APP_ID", APP_ID)
    monkeypatch.setenv("QQBOT_APP_SECRET", APP_SECRET)

    def token_400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": 100016})

    async with httpx.AsyncClient(transport=httpx.MockTransport(token_400)) as client:
        with pytest.raises(MessagingError, match="HTTP 400") as exc:
            await QQBotAdapter(client=client, now_fn=lambda: 0.0).send("group:g", "hi")
    assert APP_SECRET not in str(exc.value) and ACCESS_TOKEN not in str(exc.value)

    def token_missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 100016, "message": "invalid appid or secret"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(token_missing)) as client:
        with pytest.raises(MessagingError, match="API reported failure"):
            await QQBotAdapter(client=client, now_fn=lambda: 0.0).send("group:g", "hi")

    def message_400(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/app/getAppAccessToken":
            return httpx.Response(200, json={"access_token": ACCESS_TOKEN, "expires_in": 7200})
        return httpx.Response(400, json={"code": 12000})

    async with httpx.AsyncClient(transport=httpx.MockTransport(message_400)) as client:
        with pytest.raises(MessagingError, match="HTTP 400"):
            await QQBotAdapter(client=client, now_fn=lambda: 0.0).send("group:g", "hi")

    monkeypatch.delenv("QQBOT_APP_ID", raising=False)
    with pytest.raises(MessagingError, match="QQBOT_APP_ID is not set") as exc:
        await QQBotAdapter().send("group:g", "hi")
    assert APP_SECRET not in str(exc.value)
    monkeypatch.setenv("QQBOT_APP_ID", APP_ID)
    monkeypatch.delenv("QQBOT_APP_SECRET", raising=False)
    with pytest.raises(MessagingError, match="QQBOT_APP_SECRET is not set"):
        await QQBotAdapter().send("group:g", "hi")


def test_record_reports_env_names_only(monkeypatch):
    for name in QQBotAdapter.credential_env + QQBotAdapter.verification_env:
        monkeypatch.delenv(name, raising=False)
    record = QQBotAdapter().record()
    assert record.status == "blocked" and record.kind == "messaging"
    assert record.capabilities == ["send", "webhook"]
    assert record.config_ref == "QQBOT_APP_ID,QQBOT_APP_SECRET,QQBOT_PUBLIC_KEY"
    assert record.metadata["reason"] == "missing credentials: QQBOT_APP_ID, QQBOT_APP_SECRET"

    monkeypatch.setenv("QQBOT_APP_ID", APP_ID)
    monkeypatch.setenv("QQBOT_APP_SECRET", APP_SECRET)
    configured = QQBotAdapter().record()
    assert configured.status == "configured"
    assert configured.verification == "mock-tested"
    assert configured.metadata["credential_env"] == ["QQBOT_APP_ID", "QQBOT_APP_SECRET"]
    assert configured.metadata["verification_env"] == ["QQBOT_PUBLIC_KEY"]
    assert configured.metadata["live_verified"] is False
    dumped = json.dumps(configured.model_dump())
    assert APP_ID not in dumped and APP_SECRET not in dumped and ACCESS_TOKEN not in dumped


def test_adapter_registers_via_channel_discovery():
    assert adapter_classes()["qqbot"] is QQBotAdapter
    assert "qqbot" in webhook_channels()
    assert "qqbot" not in polling_channels()


async def test_handle_webhook_rejects_stale_timestamp_and_replayed_message(monkeypatch):
    monkeypatch.setenv("QQBOT_PUBLIC_KEY", PUBLIC_KEY)
    adapter = QQBotAdapter(signature_verifier=fake_verifier())
    stale = {"X-Signature-Ed25519": GOOD_SIGNATURE, "X-Signature-Timestamp": str(int(time.time()) - 900)}
    with pytest.raises(WebhookVerificationError, match="stale"):
        await adapter.handle_webhook(GROUP_EVENT, stale)

    assert len(await adapter.handle_webhook(GROUP_EVENT, SIGNED_HEADERS)) == 1
    with pytest.raises(WebhookVerificationError, match="replay"):
        await adapter.handle_webhook(GROUP_EVENT, SIGNED_HEADERS)
