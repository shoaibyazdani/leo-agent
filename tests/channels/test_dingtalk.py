"""Offline verification for the DingTalk channel adapter.

No live DingTalk endpoint is contacted: HTTP goes through httpx.MockTransport
and callback signatures are computed in-test with the documented algorithms.
No secret value ever appears in an error or a record.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import httpx
import pytest

from leo.integrations.channels import adapter_classes, polling_channels, webhook_channels
from leo.integrations.channels.dingtalk import DingTalkAdapter
from leo.integrations.messaging import (
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
)

CALLBACK_SECRET = "dt-callback-secret"
TIMESTAMP = "1700000000000"
WEBHOOK_URL = "https://oapi.dingtalk.com/robot/send?access_token=fake-robot-token"
ROBOT_SECRET = "fake-robot-secret"


def callback_sign(secret: str, timestamp: str) -> str:
    """Documented HTTP-callback sign: base64(HMAC-SHA256(secret, f"{ts}\\n{secret}"))."""
    message = f"{timestamp}\n{secret}".encode("utf-8")
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    ).decode("utf-8")


def robot_sign(secret: str, timestamp_ms: str) -> str:
    """Documented custom-robot sign: urlsafe base64 of the same HMAC."""
    message = f"{timestamp_ms}\n{secret}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8")


SIGNED_HEADERS = {"timestamp": TIMESTAMP, "sign": callback_sign(CALLBACK_SECRET, TIMESTAMP)}


async def test_verify_webhook_callback_sign(monkeypatch):
    monkeypatch.setenv("DINGTALK_CALLBACK_SECRET", CALLBACK_SECRET)
    adapter = DingTalkAdapter()
    payload = {"text": {"content": "hello"}}
    assert adapter.verify_webhook(payload, SIGNED_HEADERS) is True
    wrong_key = {"timestamp": TIMESTAMP, "sign": callback_sign("wrong", TIMESTAMP)}
    assert adapter.verify_webhook(payload, wrong_key) is False
    assert adapter.verify_webhook(payload, {"timestamp": TIMESTAMP}) is False  # sign missing
    assert adapter.verify_webhook(payload, {"sign": callback_sign(CALLBACK_SECRET, TIMESTAMP)}) is False
    assert adapter.verify_webhook(payload, {}) is False
    assert adapter.verify_webhook(payload, None) is False


async def test_verify_webhook_fails_closed_without_secret_env(monkeypatch):
    monkeypatch.delenv("DINGTALK_CALLBACK_SECRET", raising=False)
    assert DingTalkAdapter().verify_webhook({}, SIGNED_HEADERS) is False
    assert DingTalkAdapter(require_verification=False).verify_webhook({}, SIGNED_HEADERS) is True


async def test_handle_webhook_normalizes_callback_text(monkeypatch):
    monkeypatch.setenv("DINGTALK_CALLBACK_SECRET", CALLBACK_SECRET)
    adapter = DingTalkAdapter()
    payload = {
        "conversationId": "cid-01",
        "conversationType": "2",
        "senderCorpId": "corp-1",
        "senderStaffId": "staff-77",
        "senderNick": "alice",
        "msgId": "msg-42",
        "createTime": 1700000000123,
        "msgtype": "text",
        "text": {"content": "hello dingtalk"},
    }
    messages = await adapter.handle_webhook(payload, SIGNED_HEADERS)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("dingtalk", "alice", "hello dingtalk")
    assert message.message_id == "msg-42"
    assert message.reply_ref is None
    assert message.timestamp.startswith("2023-11-14")  # createTime ms → epoch seconds


async def test_handle_webhook_sender_fallbacks_and_skip_rules(monkeypatch):
    monkeypatch.setenv("DINGTALK_CALLBACK_SECRET", CALLBACK_SECRET)
    adapter = DingTalkAdapter()
    payload = {"senderStaffId": "staff-77", "msgId": "m1", "text": {"content": "ping"}}
    assert (await adapter.handle_webhook(payload, SIGNED_HEADERS))[0].sender == "staff-77"
    stripped = {key: value for key, value in payload.items() if key != "senderStaffId"}
    assert (await adapter.handle_webhook(stripped, SIGNED_HEADERS))[0].sender == "unknown"
    # Skip rules: no well-formed text object → nothing normalized, no crash.
    assert await adapter.handle_webhook({"senderNick": "x"}, SIGNED_HEADERS) == []
    assert await adapter.handle_webhook({"text": "not-a-dict"}, SIGNED_HEADERS) == []
    assert await adapter.handle_webhook({"text": {}}, SIGNED_HEADERS) == []
    assert await adapter.handle_webhook({}, SIGNED_HEADERS) == []


async def test_handle_webhook_malformed_payload_returns_empty(monkeypatch):
    monkeypatch.setenv("DINGTALK_CALLBACK_SECRET", CALLBACK_SECRET)
    adapter = DingTalkAdapter()
    for malformed in ([1, 2, 3], "not json", None, 42, {"text": {"content": None}}):
        assert await adapter.handle_webhook(malformed, SIGNED_HEADERS) == []


async def test_handle_webhook_rejects_unsigned_callback(monkeypatch):
    monkeypatch.setenv("DINGTALK_CALLBACK_SECRET", CALLBACK_SECRET)
    adapter = DingTalkAdapter()
    payload = {"text": {"content": "hi"}, "msgId": "m"}
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"timestamp": TIMESTAMP, "sign": "forged"})
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {})


async def test_send_signs_webhook_url_and_captures_request(monkeypatch):
    monkeypatch.setenv("DINGTALK_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setenv("DINGTALK_SECRET", ROBOT_SECRET)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = DingTalkAdapter(client=client)
        result = await adapter.send("room-1", "hi")
    assert result == {"channel": "dingtalk", "target": "room-1", "success": True, "message_id": None}
    assert len(requests) == 1
    url = requests[0].url
    assert url.host == "oapi.dingtalk.com" and url.path == "/robot/send"
    assert url.params["access_token"] == "fake-robot-token"
    timestamp = url.params["timestamp"]
    assert timestamp.isdigit()  # millisecond epoch
    assert url.params["sign"] == robot_sign(ROBOT_SECRET, timestamp)  # exact documented construction
    assert json.loads(requests[0].content) == {"msgtype": "text", "text": {"content": "hi"}}


async def test_send_failures_never_leak_secrets(monkeypatch):
    monkeypatch.setenv("DINGTALK_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setenv("DINGTALK_SECRET", ROBOT_SECRET)

    def api_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errcode": 310000, "errmsg": "sign not match"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(api_failure)) as client:
        with pytest.raises(MessagingError, match="API reported failure") as exc:
            await DingTalkAdapter(client=client).send("room-1", "hi")
    assert ROBOT_SECRET not in str(exc.value)
    assert "fake-robot-token" not in str(exc.value)

    def http_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"errcode": 400})

    async with httpx.AsyncClient(transport=httpx.MockTransport(http_failure)) as client:
        with pytest.raises(MessagingError, match="HTTP 400") as exc:
            await DingTalkAdapter(client=client).send("room-1", "hi")
    assert ROBOT_SECRET not in str(exc.value)


async def test_send_rejects_non_dingtalk_webhook_urls(monkeypatch):
    monkeypatch.setenv("DINGTALK_SECRET", ROBOT_SECRET)
    for url in (
        "https://evil.example/robot/send?access_token=x",  # not a DingTalk host
        "https://oapi.dingtalk.com/api/chat/send",  # not the robot-send path
        "https://dingtalk.com/robot/send",  # bare registrable domain
        "http://oapi.dingtalk.com/robot/send",  # not HTTPS
    ):
        monkeypatch.setenv("DINGTALK_WEBHOOK_URL", url)
        with pytest.raises(MessagingError, match="not a DingTalk robot webhook URL"):
            await DingTalkAdapter().send("room", "hi")  # no client: must fail before any I/O


async def test_send_blocked_names_missing_env(monkeypatch):
    monkeypatch.delenv("DINGTALK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DINGTALK_SECRET", raising=False)
    with pytest.raises(MessagingError, match="DINGTALK_WEBHOOK_URL is not set") as exc:
        await DingTalkAdapter().send("room", "hi")
    assert ROBOT_SECRET not in str(exc.value)
    monkeypatch.setenv("DINGTALK_WEBHOOK_URL", WEBHOOK_URL)
    with pytest.raises(MessagingError, match="DINGTALK_SECRET is not set"):
        await DingTalkAdapter().send("room", "hi")


def test_record_reports_env_names_only(monkeypatch):
    for name in DingTalkAdapter.credential_env + DingTalkAdapter.verification_env:
        monkeypatch.delenv(name, raising=False)
    record = DingTalkAdapter().record()
    assert record.status == "blocked" and record.kind == "messaging"
    assert record.capabilities == ["send", "webhook"]
    assert record.config_ref == "DINGTALK_WEBHOOK_URL,DINGTALK_SECRET,DINGTALK_CALLBACK_SECRET"
    assert record.metadata["reason"] == "missing credentials: DINGTALK_WEBHOOK_URL, DINGTALK_SECRET"

    monkeypatch.setenv("DINGTALK_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setenv("DINGTALK_SECRET", ROBOT_SECRET)
    configured = DingTalkAdapter().record()
    assert configured.status == "configured"
    assert configured.verification == "mock-tested"
    assert configured.metadata["credential_env"] == ["DINGTALK_WEBHOOK_URL", "DINGTALK_SECRET"]
    assert configured.metadata["verification_env"] == ["DINGTALK_CALLBACK_SECRET"]
    assert configured.metadata["live_verified"] is False
    dumped = json.dumps(configured.model_dump())
    assert "fake-robot-token" not in dumped
    assert ROBOT_SECRET not in dumped


def test_adapter_registers_via_channel_discovery():
    assert adapter_classes()["dingtalk"] is DingTalkAdapter
    assert "dingtalk" in webhook_channels()
    assert "dingtalk" not in polling_channels()
