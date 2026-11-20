"""Offline verification for the Feishu channel adapter.

No live Feishu endpoint is contacted: HTTP goes through httpx.MockTransport,
event signatures are computed in-test with the documented v2 scheme, and the
token-cache clock is an injectable ``now_fn``. No secret value ever appears in
an error or a record.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid

import httpx
import pytest

from leo.integrations.channels import adapter_classes, polling_channels, webhook_channels
from leo.integrations.channels.feishu import FeishuAdapter
from leo.integrations.messaging import (
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
)

ENCRYPT_KEY = "fs-encrypt-key"
APP_ID = "fake-app-id"
APP_SECRET = "fake-app-secret"
TENANT_TOKEN = "fake-tenant-token"


def lark_signature(key: str, timestamp: str, nonce: str, body: bytes) -> str:
    """Documented v2 scheme: hex HMAC-SHA256(key, timestamp + nonce + raw body)."""
    message = timestamp.encode("utf-8") + nonce.encode("utf-8") + body
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def signed_headers(payload, *, key=ENCRYPT_KEY, timestamp=None, nonce=None, raw=None):
    """Fresh, unique v2 headers so the replay/freshness guard is exercised honestly."""
    timestamp = str(int(time.time())) if timestamp is None else str(timestamp)
    nonce = uuid.uuid4().hex if nonce is None else str(nonce)
    body = raw if raw is not None else json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    headers = {
        "X-Lark-Request-Timestamp": timestamp,
        "X-Lark-Request-Nonce": nonce,
        "X-Lark-Signature": lark_signature(key, timestamp, nonce, body),
    }
    return body, headers


EVENT_PAYLOAD = {
    "schema": "2.0",
    "header": {
        "event_id": "evt-1",
        "event_type": "im.message.receive_v1",
        "create_time": "1700000000000",
    },
    "event": {
        "message": {
            "message_id": "om_42",
            "chat_id": "oc_chat",
            "message_type": "text",
            "create_time": "1700000000",
            "content": json.dumps({"text": "hello feishu"}),
        },
        "sender": {"sender_id": {"union_id": "un_9", "open_id": "ou_8"}, "sender_type": "app"},
    },
}


async def test_verify_webhook_lark_v2_signature(monkeypatch):
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    adapter = FeishuAdapter()
    payload = {"type": "event_callback", "event": {"x": 1}}
    raw, headers = signed_headers(payload)
    assert adapter.verify_webhook(payload, headers, raw_body=raw) is True
    # Without raw_body the canonical dict serialization is signed instead.
    assert adapter.verify_webhook(payload, headers) is True
    wrong_raw, wrong_headers = signed_headers(payload, key="wrong")
    assert adapter.verify_webhook(payload, wrong_headers, raw_body=wrong_raw) is False
    tampered = {
        **headers,
        "X-Lark-Signature": lark_signature(ENCRYPT_KEY, "1700000000", "nonce-1", b"other body"),
    }
    assert adapter.verify_webhook(payload, tampered, raw_body=raw) is False
    assert adapter.verify_webhook(payload, {"X-Lark-Request-Timestamp": "1", "X-Lark-Request-Nonce": "n"}) is False
    assert adapter.verify_webhook(payload, {}) is False
    assert adapter.verify_webhook(payload, None, raw_body=raw) is False


async def test_verify_webhook_fails_closed_without_key_env(monkeypatch):
    monkeypatch.delenv("FEISHU_ENCRYPT_KEY", raising=False)
    payload = {"type": "event_callback"}
    raw, headers = signed_headers(payload)
    assert FeishuAdapter().verify_webhook(payload, headers, raw_body=raw) is False
    assert FeishuAdapter(require_verification=False).verify_webhook(
        payload, headers, raw_body=raw
    ) is True


async def test_handle_webhook_normalizes_text_event(monkeypatch):
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    adapter = FeishuAdapter()
    raw, headers = signed_headers(EVENT_PAYLOAD)
    messages = await adapter.handle_webhook(EVENT_PAYLOAD, headers, raw_body=raw)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("feishu", "un_9", "hello feishu")
    assert message.message_id == "om_42"
    assert message.reply_ref is None
    assert message.timestamp.startswith("2023-11-14")  # create_time seconds str → ISO

    # Sender fallback: union_id → open_id → "unknown".
    no_union = json.loads(json.dumps(EVENT_PAYLOAD))
    del no_union["event"]["sender"]["sender_id"]["union_id"]
    raw2, headers2 = signed_headers(no_union)
    assert (await adapter.handle_webhook(no_union, headers2, raw_body=raw2))[0].sender == "ou_8"
    no_sender = json.loads(json.dumps(EVENT_PAYLOAD))
    del no_sender["event"]["sender"]
    raw3, headers3 = signed_headers(no_sender)
    assert (await adapter.handle_webhook(no_sender, headers3, raw_body=raw3))[0].sender == "unknown"


async def test_handle_webhook_url_verification_returns_empty(monkeypatch):
    """Feishu's handshake needs the challenge echoed by the HTTP route; the
    adapter honestly returns no messages instead of pretending to answer it."""
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    adapter = FeishuAdapter()
    payload = {"type": "url_verification", "challenge": "fake-challenge-value"}
    raw, headers = signed_headers(payload)
    assert await adapter.handle_webhook(payload, headers, raw_body=raw) == []


async def test_handle_webhook_skip_rules(monkeypatch):
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    adapter = FeishuAdapter()
    non_text = json.loads(json.dumps(EVENT_PAYLOAD))
    non_text["event"]["message"]["message_type"] = "post"
    bad_content = json.loads(json.dumps(EVENT_PAYLOAD))
    bad_content["event"]["message"]["content"] = "{not json"
    no_event = {"schema": "2.0", "header": {"event_type": "im.message.receive_v1"}}
    other_event = {"header": {"event_type": "contact.user.updated_v3"}, "event": {"anything": True}}
    for payload in (non_text, bad_content, no_event, other_event):
        raw, headers = signed_headers(payload)
        assert await adapter.handle_webhook(payload, headers, raw_body=raw) == []


async def test_handle_webhook_malformed_payload_returns_empty(monkeypatch):
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    adapter = FeishuAdapter()
    for malformed in ([1, 2], "garbage", None, 42, {"event": "not-a-dict"}):
        raw, headers = signed_headers(malformed)
        assert await adapter.handle_webhook(malformed, headers, raw_body=raw) == []


async def test_handle_webhook_rejects_bad_signature(monkeypatch):
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    adapter = FeishuAdapter()
    bad_headers = {
        "X-Lark-Signature": "00" * 32,
        "X-Lark-Request-Timestamp": "1",
        "X-Lark-Request-Nonce": "n",
    }
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(EVENT_PAYLOAD, bad_headers)
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(EVENT_PAYLOAD, {})


async def test_send_fetches_token_then_posts_and_caches(monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", APP_ID)
    monkeypatch.setenv("FEISHU_APP_SECRET", APP_SECRET)
    requests: list[httpx.Request] = []
    token_hits: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            token_hits.append(request)
            return httpx.Response(200, json={"code": 0, "tenant_access_token": TENANT_TOKEN, "expire": 7200})
        return httpx.Response(200, json={"code": 0, "msg": "success", "data": {"message_id": "om_out_1"}})

    clock = {"now": 1000.0}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = FeishuAdapter(client=client, now_fn=lambda: clock["now"])
        result = await adapter.send("oc_chat", "hello")
        assert result == {"channel": "feishu", "target": "oc_chat", "success": True, "message_id": "om_out_1"}
        assert len(token_hits) == 1
        token_request = token_hits[0]
        assert token_request.url.host == "open.feishu.cn"
        assert token_request.url.path == "/open-apis/auth/v3/tenant_access_token/internal"
        assert json.loads(token_request.content) == {"app_id": APP_ID, "app_secret": APP_SECRET}

        message_request = requests[1]
        assert message_request.url.host == "open.feishu.cn"
        assert message_request.url.path == "/open-apis/im/v1/messages"
        assert message_request.url.params["receive_id_type"] == "chat_id"
        assert message_request.headers["authorization"] == f"Bearer {TENANT_TOKEN}"
        body = json.loads(message_request.content)
        assert body["receive_id"] == "oc_chat"
        assert body["msg_type"] == "text"
        assert json.loads(body["content"]) == {"text": "hello"}

        await adapter.send("oc_chat", "again")
        assert len(token_hits) == 1  # token cached: no second fetch

        clock["now"] += 7200.0
        await adapter.send("oc_chat", "third")
        assert len(token_hits) == 2  # past expiry: refetched exactly once


async def test_send_failures_and_env_blocks(monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", APP_ID)
    monkeypatch.setenv("FEISHU_APP_SECRET", APP_SECRET)

    def token_400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": 99991668})

    async with httpx.AsyncClient(transport=httpx.MockTransport(token_400)) as client:
        with pytest.raises(MessagingError, match="HTTP 400") as exc:
            await FeishuAdapter(client=client, now_fn=lambda: 0.0).send("oc_chat", "hi")
    assert APP_SECRET not in str(exc.value) and TENANT_TOKEN not in str(exc.value)

    def token_missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 99991661, "msg": "app access token error"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(token_missing)) as client:
        with pytest.raises(MessagingError, match="API reported failure"):
            await FeishuAdapter(client=client, now_fn=lambda: 0.0).send("oc_chat", "hi")

    def message_400(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(200, json={"tenant_access_token": TENANT_TOKEN, "expire": 7200})
        return httpx.Response(400, json={"code": 99991663})

    async with httpx.AsyncClient(transport=httpx.MockTransport(message_400)) as client:
        with pytest.raises(MessagingError, match="HTTP 400"):
            await FeishuAdapter(client=client, now_fn=lambda: 0.0).send("oc_chat", "hi")

    def api_code_failure(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(200, json={"tenant_access_token": TENANT_TOKEN, "expire": 7200})
        return httpx.Response(200, json={"code": 230002, "msg": "not have resource"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(api_code_failure)) as client:
        with pytest.raises(MessagingError, match="API reported failure") as exc:
            await FeishuAdapter(client=client, now_fn=lambda: 0.0).send("oc_chat", "hi")
    assert TENANT_TOKEN not in str(exc.value)

    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    with pytest.raises(MessagingError, match="FEISHU_APP_ID is not set") as exc:
        await FeishuAdapter().send("oc_chat", "hi")
    assert APP_SECRET not in str(exc.value)
    monkeypatch.setenv("FEISHU_APP_ID", APP_ID)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    with pytest.raises(MessagingError, match="FEISHU_APP_SECRET is not set"):
        await FeishuAdapter().send("oc_chat", "hi")


def test_record_reports_env_names_only(monkeypatch):
    for name in FeishuAdapter.credential_env + FeishuAdapter.verification_env:
        monkeypatch.delenv(name, raising=False)
    record = FeishuAdapter().record()
    assert record.status == "blocked" and record.kind == "messaging"
    assert record.capabilities == ["send", "webhook"]
    assert record.config_ref == "FEISHU_APP_ID,FEISHU_APP_SECRET,FEISHU_ENCRYPT_KEY"
    assert record.metadata["reason"] == "missing credentials: FEISHU_APP_ID, FEISHU_APP_SECRET"

    monkeypatch.setenv("FEISHU_APP_ID", APP_ID)
    monkeypatch.setenv("FEISHU_APP_SECRET", APP_SECRET)
    configured = FeishuAdapter().record()
    assert configured.status == "configured"
    assert configured.verification == "mock-tested"
    assert configured.metadata["credential_env"] == ["FEISHU_APP_ID", "FEISHU_APP_SECRET"]
    assert configured.metadata["verification_env"] == ["FEISHU_ENCRYPT_KEY"]
    assert configured.metadata["live_verified"] is False
    dumped = json.dumps(configured.model_dump())
    assert APP_ID not in dumped and APP_SECRET not in dumped and TENANT_TOKEN not in dumped


def test_adapter_registers_via_channel_discovery():
    assert adapter_classes()["feishu"] is FeishuAdapter
    assert "feishu" in webhook_channels()
    assert "feishu" not in polling_channels()


async def test_handle_webhook_decrypts_encrypted_event_with_configured_key(monkeypatch):
    """A signed {"encrypt": ...} body is decrypted and normalized, not discarded."""
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    seen: list[tuple[str, str]] = []

    def decryptor(encrypt_b64: str, key: str) -> str:
        seen.append((encrypt_b64, key))
        assert encrypt_b64 == "encrypted-event-blob"
        return json.dumps(EVENT_PAYLOAD)

    adapter = FeishuAdapter(decryptor=decryptor)
    encrypted = {"encrypt": "encrypted-event-blob"}
    raw, headers = signed_headers(encrypted)
    messages = await adapter.handle_webhook(encrypted, headers, raw_body=raw)
    assert len(messages) == 1
    assert (messages[0].channel, messages[0].sender, messages[0].text) == (
        "feishu", "un_9", "hello feishu"
    )
    assert seen == [("encrypted-event-blob", ENCRYPT_KEY)]


async def test_handle_webhook_encrypted_body_fails_closed(monkeypatch):
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    encrypted = {"encrypt": "opaque"}
    raw, headers = signed_headers(encrypted)

    def boom(encrypt_b64: str, key: str) -> str:
        raise ValueError("ciphertext is malformed")

    with pytest.raises(MessagingError, match="decryptor raised ValueError"):
        await FeishuAdapter(decryptor=boom).handle_webhook(encrypted, headers, raw_body=raw)
    with pytest.raises(MessagingError, match="decrypted body is not JSON"):
        await FeishuAdapter(decryptor=lambda *_: "not json").handle_webhook(
            encrypted, signed_headers(encrypted)[1], raw_body=raw
        )
    with pytest.raises(MessagingError, match="decrypted body is not an object"):
        await FeishuAdapter(decryptor=lambda *_: "[1,2]").handle_webhook(
            encrypted, signed_headers(encrypted)[1], raw_body=raw
        )

    monkeypatch.delenv("FEISHU_ENCRYPT_KEY", raising=False)
    adapter = FeishuAdapter(decryptor=lambda *_: "{}", require_verification=False)
    with pytest.raises(MessagingError, match="FEISHU_ENCRYPT_KEY is not set"):
        await adapter.handle_webhook(encrypted, headers, raw_body=raw)


async def test_handle_webhook_rejects_stale_and_replayed_signatures(monkeypatch):
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    adapter = FeishuAdapter()
    stale_raw, stale_headers = signed_headers(EVENT_PAYLOAD, timestamp=int(time.time()) - 900)
    with pytest.raises(WebhookVerificationError, match="stale"):
        await adapter.handle_webhook(EVENT_PAYLOAD, stale_headers, raw_body=stale_raw)

    raw, headers = signed_headers(EVENT_PAYLOAD, nonce="replay-nonce")
    assert len(await adapter.handle_webhook(EVENT_PAYLOAD, headers, raw_body=raw)) == 1
    with pytest.raises(WebhookVerificationError, match="replay"):
        await adapter.handle_webhook(EVENT_PAYLOAD, headers, raw_body=raw)
