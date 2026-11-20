"""Offline tests for the WeCom (WeChat Work / 企业微信) adapter.

No network access: HTTP goes through ``httpx.MockTransport`` and every
webhook payload is signed locally. Full-crypto round-trips use the documented
EncodingAESKey scheme and skip cleanly when the optional ``cryptography``
package is absent; parsing, fail-closed and send paths run everywhere.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import struct
import sys
import time

import httpx
import pytest

from leo.integrations.channels._wechat_crypto import wechat_msg_signature
from leo.integrations.channels.wecom import WeComAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

CRYPTOGRAPHY_AVAILABLE = importlib.util.find_spec("cryptography") is not None
requires_crypto = pytest.mark.skipif(
    not CRYPTOGRAPHY_AVAILABLE, reason="optional 'cryptography' package not installed"
)

ENV_VALUES = {
    "WECOM_CORP_ID": "fake-corp-id-should-not-leak",
    "WECOM_SECRET": "fake-corp-secret-should-not-leak",
    "WECOM_CALLBACK_TOKEN": "wecom-callback-token-should-not-leak",
    "WECOM_ENCODING_AES_KEY": "fake-aes-key-should-not-leak",
}
TOKEN = ENV_VALUES["WECOM_CALLBACK_TOKEN"]
ENCRYPT_FIELD = "ZmFrZS1lbmNyeXB0ZA=="


def set_wecom_env(monkeypatch: pytest.MonkeyPatch, *, drop: str | None = None) -> None:
    for name, value in ENV_VALUES.items():
        if name != drop:
            monkeypatch.setenv(name, value)
    if drop:
        monkeypatch.delenv(drop, raising=False)


def signed_payload(encrypt: str, token: str, timestamp: str | None = None, nonce: str = "nonce-1") -> dict:
    """The JSON-mode payload an operator's fronting shim builds for LEO."""
    timestamp = str(int(time.time())) if timestamp is None else str(timestamp)
    return {
        "encrypt": encrypt,
        "msg_signature": wechat_msg_signature(token, timestamp, nonce, encrypt),
        "timestamp": timestamp,
        "nonce": nonce,
    }


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def fake_decryptor(plaintext: str, calls: list | None = None):
    def decrypt(encrypt_b64: str, aes_key_b64: str, app_id: str) -> str:
        if calls is not None:
            calls.append((encrypt_b64, aes_key_b64, app_id))
        return plaintext

    return decrypt


def encrypt_wechat_plaintext(message: bytes, key: bytes, receive_id: str) -> str:
    """Encrypt with the documented scheme the default decryptor must undo."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    body = os.urandom(16) + struct.pack(">I", len(message)) + message + receive_id.encode("utf-8")
    pad = 32 - len(body) % 32
    padded = body + bytes([pad]) * pad
    encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
    return base64.b64encode(encryptor.update(padded) + encryptor.finalize()).decode("ascii")


def make_crypto_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in sys.modules if n == "cryptography" or n.startswith("cryptography.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "cryptography", None)


# --------------------------------------------------------------------- webhook


async def test_verify_webhook_signature_and_fail_closed(monkeypatch):
    set_wecom_env(monkeypatch)
    adapter = WeComAdapter()
    payload = signed_payload(ENCRYPT_FIELD, TOKEN)
    assert adapter.verify_webhook(payload) is True
    # constant-time compare tolerates case/whitespace, not wrong values
    assert adapter.verify_webhook(dict(payload, msg_signature=payload["msg_signature"].upper())) is True
    assert adapter.verify_webhook(dict(payload, msg_signature="0" * 40)) is False
    # missing or mistyped fields
    assert adapter.verify_webhook({"encrypt": ENCRYPT_FIELD}) is False
    assert adapter.verify_webhook({"encrypt": 1, "msg_signature": "2", "timestamp": "3", "nonce": "4"}) is False
    # a raw XML envelope is not the JSON-mode contract
    assert adapter.verify_webhook("<xml>legacy</xml>") is False
    # fail closed without the verification token
    monkeypatch.delenv("WECOM_CALLBACK_TOKEN")
    assert WeComAdapter(require_verification=True).verify_webhook(payload) is False
    assert WeComAdapter(require_verification=False).verify_webhook(payload) is True


async def test_handle_webhook_normalizes_json_message(monkeypatch):
    set_wecom_env(monkeypatch)
    plaintext = json.dumps(
        {
            "msgtype": "text",
            "text": {"content": "hello wecom"},
            "fromusername": "alice",
            "msgid": 123456,
            "createtime": 1700000000,
        }
    )
    calls: list = []
    adapter = WeComAdapter(decryptor=fake_decryptor(plaintext, calls))
    messages = await adapter.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("wecom", "alice", "hello wecom")
    assert message.timestamp == "2023-11-14T22:13:20+00:00"
    assert message.message_id == "123456"
    assert message.reply_ref is None
    assert calls == [(ENCRYPT_FIELD, ENV_VALUES["WECOM_ENCODING_AES_KEY"], ENV_VALUES["WECOM_CORP_ID"])]

    # capitalized keys (the classic callback casing) normalize the same way
    caps = WeComAdapter(
        decryptor=fake_decryptor(
            json.dumps({"FromUserName": "bob", "text": {"content": "caps"}, "MsgId": 7, "CreateTime": 1600000000})
        )
    )
    messages = await caps.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    assert (messages[0].sender, messages[0].text, messages[0].timestamp, messages[0].message_id) == (
        "bob",
        "caps",
        "2020-09-13T12:26:40+00:00",
        "7",
    )

    # a payload without sender defaults honestly instead of crashing
    bare = WeComAdapter(decryptor=fake_decryptor(json.dumps({"text": {"content": "anon"}})))
    messages = await bare.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    assert messages[0].sender == "unknown" and messages[0].text == "anon" and messages[0].message_id is None


async def test_handle_webhook_rejects_unverified_payload(monkeypatch):
    set_wecom_env(monkeypatch)
    adapter = WeComAdapter(decryptor=fake_decryptor("{}"))
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(dict(signed_payload(ENCRYPT_FIELD, TOKEN), msg_signature="0" * 40))
    # missing verification token with require_verification=True: fail closed
    set_wecom_env(monkeypatch, drop="WECOM_CALLBACK_TOKEN")
    with pytest.raises(WebhookVerificationError):
        await WeComAdapter(decryptor=fake_decryptor("{}")).handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))


async def test_handle_webhook_malformed_payloads_return_empty(monkeypatch):
    # Verification disabled to isolate the parser; the verified path is
    # covered above and by the round-trip tests below.
    adapter = WeComAdapter(require_verification=False)
    assert await adapter.handle_webhook({}) == []
    assert await adapter.handle_webhook({"encrypt": 123}) == []
    assert await adapter.handle_webhook({"encrypt": ""}) == []
    assert await adapter.handle_webhook(["not", "a", "dict"]) == []
    # the legacy XML envelope is documented as unsupported: skipped, not crashed
    assert await adapter.handle_webhook("<xml><Encrypt>legacy</Encrypt></xml>") == []

    # With verification on and a valid signature, untrusted plaintext is data:
    set_wecom_env(monkeypatch)
    not_json = WeComAdapter(decryptor=fake_decryptor("not json at all"))
    assert await not_json.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN)) == []
    mixed = WeComAdapter(
        decryptor=fake_decryptor(json.dumps(["nope", 42, {"text": {"content": "ok"}, "fromusername": "dan"}]))
    )
    messages = await mixed.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    assert len(messages) == 1 and messages[0].sender == "dan" and messages[0].text == "ok"


async def test_handle_webhook_missing_env_fails_closed(monkeypatch):
    payload = signed_payload(ENCRYPT_FIELD, TOKEN)
    set_wecom_env(monkeypatch, drop="WECOM_ENCODING_AES_KEY")
    with pytest.raises(MessagingError) as exc_info:
        await WeComAdapter(decryptor=fake_decryptor("{}")).handle_webhook(payload)
    assert "WECOM_ENCODING_AES_KEY" in str(exc_info.value)
    assert ENV_VALUES["WECOM_ENCODING_AES_KEY"] not in str(exc_info.value)

    set_wecom_env(monkeypatch, drop="WECOM_CORP_ID")
    with pytest.raises(MessagingError) as exc_info:
        await WeComAdapter(decryptor=fake_decryptor("{}")).handle_webhook(payload)
    assert "WECOM_CORP_ID" in str(exc_info.value)
    assert ENV_VALUES["WECOM_CORP_ID"] not in str(exc_info.value)


async def test_handle_webhook_fail_closed_on_decryptor_errors(monkeypatch):
    set_wecom_env(monkeypatch)

    def rejecting(encrypt_b64: str, aes_key_b64: str, app_id: str) -> str:
        raise ValueError("app_id mismatch, got some-foreign-id")

    adapter = WeComAdapter(decryptor=rejecting)
    with pytest.raises(MessagingError) as exc_info:
        await adapter.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    # the exception text of an untrusted-input failure is never echoed
    assert "some-foreign-id" not in str(exc_info.value)


async def test_handle_webhook_fail_closed_without_crypto_backend(monkeypatch):
    set_wecom_env(monkeypatch)
    make_crypto_unimportable(monkeypatch)
    adapter = WeComAdapter()  # default decryptor: lazily imports cryptography
    with pytest.raises(MessagingError) as exc_info:
        await adapter.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    message = str(exc_info.value)
    assert "cryptography" in message and "decryptor" in message
    for value in ENV_VALUES.values():
        assert value not in message


@requires_crypto
async def test_handle_webhook_real_encrypt_round_trip(monkeypatch):
    key = os.urandom(32)
    aes_key = base64.b64encode(key).decode("ascii")
    monkeypatch.setenv("WECOM_CALLBACK_TOKEN", TOKEN)
    monkeypatch.setenv("WECOM_ENCODING_AES_KEY", aes_key)
    monkeypatch.setenv("WECOM_CORP_ID", "ww-fake-corp")
    plaintext = json.dumps(
        {
            "msgtype": "text",
            "text": {"content": "real pipeline"},
            "fromusername": "alice",
            "msgid": 42,
            "createtime": 1700000000,
        }
    )
    encrypt = encrypt_wechat_plaintext(plaintext.encode("utf-8"), key, "ww-fake-corp")
    messages = await WeComAdapter().handle_webhook(signed_payload(encrypt, TOKEN))
    assert [
        (m.channel, m.sender, m.text, m.timestamp, m.message_id, m.reply_ref) for m in messages
    ] == [("wecom", "alice", "real pipeline", "2023-11-14T22:13:20+00:00", "42", None)]


@requires_crypto
async def test_handle_webhook_wrong_receive_id_rejected(monkeypatch):
    key = os.urandom(32)
    monkeypatch.setenv("WECOM_CALLBACK_TOKEN", TOKEN)
    monkeypatch.setenv("WECOM_ENCODING_AES_KEY", base64.b64encode(key).decode("ascii"))
    monkeypatch.setenv("WECOM_CORP_ID", "ww-fake-corp")
    # a payload minted for another corp must never normalize
    encrypt = encrypt_wechat_plaintext(b'{"text": {"content": "intruder"}}', key, "ww-other-corp")
    with pytest.raises(MessagingError):
        await WeComAdapter().handle_webhook(signed_payload(encrypt, TOKEN))


# ------------------------------------------------------------------------ send


async def test_send_token_cache_two_step_flow(monkeypatch):
    monkeypatch.setenv("WECOM_CORP_ID", "fake-corp-id")
    monkeypatch.setenv("WECOM_SECRET", "fake-corp-secret")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/gettoken":
            return httpx.Response(200, json={"errcode": 0, "access_token": "fake-access-token", "expires_in": 7200})
        return httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})

    clock = FakeClock(1000.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = WeComAdapter(client=client, now_fn=clock)
        result = await adapter.send("agent:100:alice", "hi")
        assert result == {"channel": "wecom", "target": "agent:100:alice", "success": True, "message_id": None}
        await adapter.send("agent:100:alice", "again")
        # token expired (expiry = 1000 + 7200): the third send re-fetches
        clock.now = 8200.5
        await adapter.send("agent:100:alice", "third")

    token_requests = [r for r in requests if r.url.path == "/gettoken"]
    send_requests = [r for r in requests if r.url.path == "/cgi-bin/message/send"]
    assert len(token_requests) == 2  # one cached reuse, one expiry-driven re-fetch
    assert len(send_requests) == 3
    assert str(token_requests[0].url).startswith("https://qyapi.weixin.qq.com/gettoken")
    assert token_requests[0].url.params["corpid"] == "fake-corp-id"
    assert token_requests[0].url.params["corpsecret"] == "fake-corp-secret"
    for request in send_requests:
        assert str(request.url).startswith("https://qyapi.weixin.qq.com/cgi-bin/message/send")
        assert request.url.params["access_token"] == "fake-access-token"
    assert json.loads(send_requests[0].content) == {
        "touser": "alice",
        "msgtype": "text",
        "agentid": 100,
        "text": {"content": "hi"},
    }


async def test_send_blocked_without_credentials(monkeypatch):
    monkeypatch.setenv("WECOM_SECRET", "fake-corp-secret-should-never-leak")
    monkeypatch.delenv("WECOM_CORP_ID", raising=False)
    with pytest.raises(MessagingError) as exc_info:
        await WeComAdapter().send("agent:100:alice", "hi")
    assert "WECOM_CORP_ID" in str(exc_info.value)
    assert "fake-corp-secret-should-never-leak" not in str(exc_info.value)

    monkeypatch.setenv("WECOM_CORP_ID", "fake-corp-id")
    monkeypatch.delenv("WECOM_SECRET", raising=False)
    with pytest.raises(MessagingError) as exc_info:
        await WeComAdapter().send("agent:100:alice", "hi")
    assert "WECOM_SECRET" in str(exc_info.value)


async def test_send_malformed_target(monkeypatch):
    monkeypatch.setenv("WECOM_CORP_ID", "fake-corp-id")
    monkeypatch.setenv("WECOM_SECRET", "fake-corp-secret")
    adapter = WeComAdapter()
    malformed = (
        "alice",
        "agent:100",
        "agent:abc:alice",
        "agent:100:",
        "user:100:alice",
        "agent:100:alice:extra",
        None,
        42,
    )
    for bad in malformed:
        with pytest.raises(MessagingError, match="malformed target"):
            await adapter.send(bad, "hi")
    # the (untrusted) target value is never echoed in the error
    with pytest.raises(MessagingError) as exc_info:
        await adapter.send("agent:100:sneaky-user", "hi")
    assert "sneaky-user" not in str(exc_info.value)


async def test_send_api_and_transport_failures(monkeypatch):
    monkeypatch.setenv("WECOM_CORP_ID", "fake-corp-id")
    monkeypatch.setenv("WECOM_SECRET", "fake-corp-secret")

    # gettoken itself reports failure (errcode, no access_token)
    failing_token = httpx.MockTransport(lambda r: httpx.Response(200, json={"errcode": 40013, "errmsg": "x"}))
    async with httpx.AsyncClient(transport=failing_token) as client:
        with pytest.raises(MessagingError, match="wecom send failed: API reported failure"):
            await WeComAdapter(client=client).send("agent:1:u", "hi")

    # token ok, message/send reports a nonzero errcode; the token never leaks
    def ok_token_then_errcode(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gettoken":
            return httpx.Response(200, json={"access_token": "fake-access-token", "expires_in": 7200})
        return httpx.Response(200, json={"errcode": 81013, "errmsg": "content is empty"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(ok_token_then_errcode)) as client:
        with pytest.raises(MessagingError) as exc_info:
            await WeComAdapter(client=client).send("agent:1:u", "hi")
    assert "API reported failure" in str(exc_info.value)
    assert "fake-access-token" not in str(exc_info.value)

    # transport-level and HTTP-status failures carry no URL or secret either
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        with pytest.raises(MessagingError, match="HTTP 500"):
            await WeComAdapter(client=client).send("agent:1:u", "hi")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to fake host")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        with pytest.raises(MessagingError, match="ConnectError"):
            await WeComAdapter(client=client).send("agent:1:u", "hi")

    not_json = httpx.MockTransport(lambda r: httpx.Response(200, text="not-json"))
    async with httpx.AsyncClient(transport=not_json) as client:
        with pytest.raises(MessagingError, match="invalid response"):
            await WeComAdapter(client=client).send("agent:1:u", "hi")


async def test_record_blocked_and_configured(monkeypatch):
    for name in WeComAdapter.credential_env + WeComAdapter.verification_env:
        monkeypatch.delenv(name, raising=False)
    record = WeComAdapter().record()
    assert record.status == "blocked"
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.metadata["live_verified"] is False
    assert record.config_ref == "WECOM_CORP_ID,WECOM_SECRET,WECOM_CALLBACK_TOKEN,WECOM_ENCODING_AES_KEY"
    assert "missing credentials" in record.metadata["reason"]

    set_wecom_env(monkeypatch)
    record = WeComAdapter().record()
    assert record.status == "configured"
    assert record.metadata["credential_env"] == ["WECOM_CORP_ID", "WECOM_SECRET"]
    assert record.metadata["credentials_present"] == ["WECOM_CORP_ID", "WECOM_SECRET"]
    dump = json.dumps(record.model_dump())
    for value in ENV_VALUES.values():
        assert value not in dump


# ------------------------------------------------ native XML / query / replay


def _native_xml(encrypt: str, timestamp: str, nonce: str, token: str) -> str:
    signature = wechat_msg_signature(token, timestamp, nonce, encrypt)
    return (
        "<xml>"
        f"<Encrypt><![CDATA[{encrypt}]]></Encrypt>"
        f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
        f"<TimeStamp>{timestamp}</TimeStamp>"
        f"<Nonce><![CDATA[{nonce}]]></Nonce>"
        "</xml>"
    )


async def test_handle_webhook_accepts_native_xml_envelope(monkeypatch):
    set_wecom_env(monkeypatch)
    timestamp = str(int(time.time()))
    plaintext = json.dumps({"text": {"content": "xml wecom"}, "fromusername": "alice"})
    adapter = WeComAdapter(decryptor=fake_decryptor(plaintext))
    envelope = _native_xml(ENCRYPT_FIELD, timestamp, "xml-nonce-wecom", TOKEN)
    assert adapter.verify_webhook(envelope) is True
    messages = await adapter.handle_webhook(envelope)
    assert [(m.sender, m.text) for m in messages] == [("alice", "xml wecom")]


async def test_handle_webhook_accepts_query_fields_from_headers(monkeypatch):
    set_wecom_env(monkeypatch)
    timestamp = str(int(time.time()))
    nonce = "header-query-nonce-wecom"
    signature = wechat_msg_signature(TOKEN, timestamp, nonce, ENCRYPT_FIELD)
    headers = {"MsgSignature": signature, "TimeStamp": timestamp, "Nonce": nonce}
    adapter = WeComAdapter(decryptor=fake_decryptor("<xml><FromUserName>bob</FromUserName><Content>inner xml</Content></xml>"))
    assert adapter.verify_webhook({"encrypt": ENCRYPT_FIELD}, headers) is True
    messages = await adapter.handle_webhook({"encrypt": ENCRYPT_FIELD}, headers)
    assert messages[0].text == "inner xml" and messages[0].sender == "bob"
    # JSON-mode plaintext still works through the same normalized route.
    json_adapter = WeComAdapter(decryptor=fake_decryptor(json.dumps({"text": {"content": "json"}, "fromusername": "dan"})))
    headers["Nonce"] = "header-query-nonce-wecom-2"
    signature = wechat_msg_signature(TOKEN, timestamp, headers["Nonce"], ENCRYPT_FIELD)
    headers["MsgSignature"] = signature
    assert [m.text for m in await json_adapter.handle_webhook({"encrypt": ENCRYPT_FIELD}, headers)] == ["json"]


async def test_handle_webhook_rejects_stale_timestamp_and_replay(monkeypatch):
    set_wecom_env(monkeypatch)
    adapter = WeComAdapter(decryptor=fake_decryptor("[]"))
    stale = signed_payload(ENCRYPT_FIELD, TOKEN, timestamp=str(int(time.time()) - 900), nonce="stale-wecom")
    with pytest.raises(WebhookVerificationError, match="stale"):
        await adapter.handle_webhook(stale)

    fresh = signed_payload(ENCRYPT_FIELD, TOKEN, nonce="replay-nonce-wecom")
    assert await adapter.handle_webhook(fresh) == []
    with pytest.raises(WebhookVerificationError, match="replay"):
        await adapter.handle_webhook(fresh)
