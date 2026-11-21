"""Offline tests for the WeChat Official Account adapter.

No network access: HTTP goes through ``httpx.MockTransport`` and every
webhook payload is signed locally. Full-crypto round-trips use the
documented EncodingAESKey scheme and skip cleanly when the optional
``cryptography`` package is absent; parsing, fail-closed and send paths run
everywhere. The inner message XML is stdlib-parsed and treated strictly as
untrusted data, including the external-entity probe below.
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
from leo.integrations.channels.weixin import WeChatAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

CRYPTOGRAPHY_AVAILABLE = importlib.util.find_spec("cryptography") is not None
requires_crypto = pytest.mark.skipif(
    not CRYPTOGRAPHY_AVAILABLE, reason="optional 'cryptography' package not installed"
)

ENV_VALUES = {
    "WEIXIN_APP_ID": "wx-fake-app-id-should-not-leak",
    "WEIXIN_APP_SECRET": "fake-app-secret-should-not-leak",
    "WEIXIN_TOKEN": "weixin-verify-token-should-not-leak",
    "WEIXIN_ENCODING_AES_KEY": "fake-aes-key-should-not-leak",
}
TOKEN = ENV_VALUES["WEIXIN_TOKEN"]
ENCRYPT_FIELD = "ZmFrZS1lbmNyeXB0ZA=="

INNER_TEXT_XML = (
    "<xml>"
    "<ToUserName><![CDATA[gh_fake_official]]></ToUserName>"
    "<FromUserName><![CDATA[openid-alice]]></FromUserName>"
    "<CreateTime>1700000000</CreateTime>"
    "<MsgType><![CDATA[text]]></MsgType>"
    "<Content><![CDATA[hello wechat]]></Content>"
    "<MsgId>789123456</MsgId>"
    "<AgentID>1</AgentID>"
    "<UnknownTag>ignored</UnknownTag>"
    "</xml>"
)


def set_weixin_env(monkeypatch: pytest.MonkeyPatch, *, drop: str | None = None) -> None:
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
    set_weixin_env(monkeypatch)
    adapter = WeChatAdapter()
    payload = signed_payload(ENCRYPT_FIELD, TOKEN)
    assert adapter.verify_webhook(payload) is True
    assert adapter.verify_webhook(dict(payload, msg_signature="0" * 40)) is False
    # the merged-query "MsgSignature" capitalization is also accepted
    caps = dict(payload)
    caps["MsgSignature"] = caps.pop("msg_signature")
    assert adapter.verify_webhook(caps) is True
    # missing or mistyped fields
    assert adapter.verify_webhook({"encrypt": ENCRYPT_FIELD}) is False
    assert adapter.verify_webhook({"encrypt": 1, "msg_signature": "2", "timestamp": "3", "nonce": "4"}) is False
    assert adapter.verify_webhook("<xml>raw envelope</xml>") is False
    # the dedicated raw-value helper for fronting shims
    assert adapter.verify_webhook_xml(
        ENCRYPT_FIELD, payload["msg_signature"], payload["timestamp"], payload["nonce"]
    ) is True
    assert adapter.verify_webhook_xml(ENCRYPT_FIELD, "0" * 40, payload["timestamp"], payload["nonce"]) is False
    assert adapter.verify_webhook_xml(None, "sig", payload["timestamp"], payload["nonce"]) is False
    # fail closed without the verification token
    monkeypatch.delenv("WEIXIN_TOKEN")
    assert WeChatAdapter(require_verification=True).verify_webhook(payload) is False
    assert WeChatAdapter(require_verification=False).verify_webhook(payload) is True
    assert WeChatAdapter(require_verification=True).verify_webhook_xml("e", "s", "t", "n") is False


async def test_handle_webhook_normalizes_inner_xml(monkeypatch):
    set_weixin_env(monkeypatch)
    calls: list = []
    adapter = WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML, calls))
    messages = await adapter.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("weixin", "openid-alice", "hello wechat")
    assert message.timestamp == "2023-11-14T22:13:20+00:00"
    assert message.message_id == "789123456"
    assert message.reply_ref is None
    assert calls == [(ENCRYPT_FIELD, ENV_VALUES["WEIXIN_ENCODING_AES_KEY"], ENV_VALUES["WEIXIN_APP_ID"])]

    # missing FromUserName defaults honestly; missing Content becomes empty text
    sparse = WeChatAdapter(decryptor=fake_decryptor("<xml><MsgType>text</MsgType></xml>"))
    messages = await sparse.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    assert messages[0].sender == "unknown" and messages[0].text == "" and messages[0].message_id is None
    assert messages[0].timestamp  # defaults to "now" rather than crashing


async def test_handle_webhook_rejects_unverified_payload(monkeypatch):
    set_weixin_env(monkeypatch)
    adapter = WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML))
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(dict(signed_payload(ENCRYPT_FIELD, TOKEN), msg_signature="0" * 40))
    # missing verification token with require_verification=True: fail closed
    set_weixin_env(monkeypatch, drop="WEIXIN_TOKEN")
    with pytest.raises(WebhookVerificationError):
        await WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML)).handle_webhook(
            signed_payload(ENCRYPT_FIELD, TOKEN)
        )


async def test_handle_webhook_skips_non_text_and_malformed(monkeypatch):
    # verification disabled to isolate the parser; the verified path is
    # covered above and by the round-trip tests below.
    adapter = WeChatAdapter(require_verification=False)
    assert await adapter.handle_webhook({}) == []
    assert await adapter.handle_webhook({"encrypt": 123}) == []
    assert await adapter.handle_webhook({"encrypt": ""}) == []
    # a raw XML envelope is not the JSON-mode contract: skipped, not crashed
    assert await adapter.handle_webhook("<xml><Encrypt>legacy</Encrypt></xml>") == []

    set_weixin_env(monkeypatch)
    signed = signed_payload(ENCRYPT_FIELD, TOKEN)
    # non-text message types are skipped
    image = WeChatAdapter(decryptor=fake_decryptor("<xml><FromUserName>u</FromUserName><MsgType>image</MsgType></xml>"))
    assert await image.handle_webhook(signed) == []
    event = WeChatAdapter(decryptor=fake_decryptor("<xml><MsgType>event</MsgType><Event>subscribe</Event></xml>"))
    assert await event.handle_webhook(signed) == []
    # malformed inner XML and non-XML plaintexts are data, not crashes
    for plaintext in ("not xml at all", "<xml><MsgType>text</MsgType>", "{not xml either}"):
        assert await WeChatAdapter(decryptor=fake_decryptor(plaintext)).handle_webhook(signed) == []
    # external entities are not resolved by stdlib ElementTree
    xxe = (
        '<!DOCTYPE xml [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        "<xml><MsgType>text</MsgType><Content>&xxe;</Content></xml>"
    )
    assert await WeChatAdapter(decryptor=fake_decryptor(xxe)).handle_webhook(signed) == []


async def test_handle_webhook_missing_env_fails_closed(monkeypatch):
    payload = signed_payload(ENCRYPT_FIELD, TOKEN)
    set_weixin_env(monkeypatch, drop="WEIXIN_ENCODING_AES_KEY")
    with pytest.raises(MessagingError) as exc_info:
        await WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML)).handle_webhook(payload)
    assert "WEIXIN_ENCODING_AES_KEY" in str(exc_info.value)
    assert ENV_VALUES["WEIXIN_ENCODING_AES_KEY"] not in str(exc_info.value)

    set_weixin_env(monkeypatch, drop="WEIXIN_APP_ID")
    with pytest.raises(MessagingError) as exc_info:
        await WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML)).handle_webhook(payload)
    assert "WEIXIN_APP_ID" in str(exc_info.value)
    assert ENV_VALUES["WEIXIN_APP_ID"] not in str(exc_info.value)


async def test_handle_webhook_fail_closed_on_decryptor_errors(monkeypatch):
    set_weixin_env(monkeypatch)

    def rejecting(encrypt_b64: str, aes_key_b64: str, app_id: str) -> str:
        raise ValueError("receive id mismatch, got some-foreign-appid")

    adapter = WeChatAdapter(decryptor=rejecting)
    with pytest.raises(MessagingError) as exc_info:
        await adapter.handle_webhook(signed_payload(ENCRYPT_FIELD, TOKEN))
    # the exception text of an untrusted-input failure is never echoed
    assert "some-foreign-appid" not in str(exc_info.value)


async def test_handle_webhook_fail_closed_without_crypto_backend(monkeypatch):
    set_weixin_env(monkeypatch)
    make_crypto_unimportable(monkeypatch)
    adapter = WeChatAdapter()  # default decryptor: lazily imports cryptography
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
    monkeypatch.setenv("WEIXIN_TOKEN", TOKEN)
    monkeypatch.setenv("WEIXIN_ENCODING_AES_KEY", aes_key)
    monkeypatch.setenv("WEIXIN_APP_ID", "wx-fake-app")
    encrypt = encrypt_wechat_plaintext(INNER_TEXT_XML.encode("utf-8"), key, "wx-fake-app")
    messages = await WeChatAdapter().handle_webhook(signed_payload(encrypt, TOKEN))
    assert [
        (m.channel, m.sender, m.text, m.timestamp, m.message_id, m.reply_ref) for m in messages
    ] == [("weixin", "openid-alice", "hello wechat", "2023-11-14T22:13:20+00:00", "789123456", None)]


@requires_crypto
async def test_handle_webhook_wrong_receive_id_rejected(monkeypatch):
    key = os.urandom(32)
    monkeypatch.setenv("WEIXIN_TOKEN", TOKEN)
    monkeypatch.setenv("WEIXIN_ENCODING_AES_KEY", base64.b64encode(key).decode("ascii"))
    monkeypatch.setenv("WEIXIN_APP_ID", "wx-fake-app")
    # a payload minted for another official account must never normalize
    encrypt = encrypt_wechat_plaintext(b"<xml><MsgType>text</MsgType></xml>", key, "wx-other-app")
    with pytest.raises(MessagingError):
        await WeChatAdapter().handle_webhook(signed_payload(encrypt, TOKEN))


# ------------------------------------------------------------------------ send


async def test_send_token_cache_two_step_flow(monkeypatch):
    monkeypatch.setenv("WEIXIN_APP_ID", "fake-app-id")
    monkeypatch.setenv("WEIXIN_APP_SECRET", "fake-app-secret")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/cgi-bin/token":
            return httpx.Response(200, json={"access_token": "fake-access-token", "expires_in": 7200})
        return httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})

    clock = FakeClock(1000.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = WeChatAdapter(client=client, now_fn=clock)
        result = await adapter.send("openid-alice", "hi")
        assert result == {"channel": "weixin", "target": "openid-alice", "success": True, "message_id": None}
        await adapter.send("openid-alice", "again")
        # token expired (expiry = 1000 + 7200): the third send re-fetches
        clock.now = 8200.5
        await adapter.send("openid-alice", "third")

    token_requests = [r for r in requests if r.url.path == "/cgi-bin/token"]
    send_requests = [r for r in requests if r.url.path == "/cgi-bin/message/custom/send"]
    assert len(token_requests) == 2  # one cached reuse, one expiry-driven re-fetch
    assert len(send_requests) == 3
    assert str(token_requests[0].url).startswith("https://api.weixin.qq.com/cgi-bin/token")
    assert token_requests[0].url.params["grant_type"] == "client_credential"
    assert token_requests[0].url.params["appid"] == "fake-app-id"
    assert token_requests[0].url.params["secret"] == "fake-app-secret"
    for request in send_requests:
        assert str(request.url).startswith("https://api.weixin.qq.com/cgi-bin/message/custom/send")
        assert request.url.params["access_token"] == "fake-access-token"
    assert json.loads(send_requests[0].content) == {
        "touser": "openid-alice",
        "msgtype": "text",
        "text": {"content": "hi"},
    }


async def test_send_blocked_without_credentials(monkeypatch):
    monkeypatch.setenv("WEIXIN_APP_SECRET", "fake-app-secret-should-never-leak")
    monkeypatch.delenv("WEIXIN_APP_ID", raising=False)
    with pytest.raises(MessagingError) as exc_info:
        await WeChatAdapter().send("openid-alice", "hi")
    assert "WEIXIN_APP_ID" in str(exc_info.value)
    assert "fake-app-secret-should-never-leak" not in str(exc_info.value)

    monkeypatch.setenv("WEIXIN_APP_ID", "fake-app-id")
    monkeypatch.delenv("WEIXIN_APP_SECRET", raising=False)
    with pytest.raises(MessagingError) as exc_info:
        await WeChatAdapter().send("openid-alice", "hi")
    assert "WEIXIN_APP_SECRET" in str(exc_info.value)


async def test_send_api_and_transport_failures(monkeypatch):
    monkeypatch.setenv("WEIXIN_APP_ID", "fake-app-id")
    monkeypatch.setenv("WEIXIN_APP_SECRET", "fake-app-secret")

    # the token endpoint itself reports failure (errcode, no access_token)
    failing_token = httpx.MockTransport(lambda r: httpx.Response(200, json={"errcode": 40013, "errmsg": "x"}))
    async with httpx.AsyncClient(transport=failing_token) as client:
        with pytest.raises(MessagingError, match="weixin send failed: API reported failure"):
            await WeChatAdapter(client=client).send("openid-alice", "hi")

    # token ok, custom/send reports a nonzero errcode; the token never leaks
    def ok_token_then_errcode(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cgi-bin/token":
            return httpx.Response(200, json={"access_token": "fake-access-token", "expires_in": 7200})
        return httpx.Response(200, json={"errcode": 45002, "errmsg": "api forbidden"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(ok_token_then_errcode)) as client:
        with pytest.raises(MessagingError) as exc_info:
            await WeChatAdapter(client=client).send("openid-alice", "hi")
    assert "API reported failure" in str(exc_info.value)
    assert "fake-access-token" not in str(exc_info.value)

    # transport-level and HTTP-status failures carry no URL or secret either
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        with pytest.raises(MessagingError, match="HTTP 500"):
            await WeChatAdapter(client=client).send("openid-alice", "hi")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to fake host")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        with pytest.raises(MessagingError, match="ConnectError"):
            await WeChatAdapter(client=client).send("openid-alice", "hi")

    not_json = httpx.MockTransport(lambda r: httpx.Response(200, text="not-json"))
    async with httpx.AsyncClient(transport=not_json) as client:
        with pytest.raises(MessagingError, match="invalid response"):
            await WeChatAdapter(client=client).send("openid-alice", "hi")


async def test_record_blocked_and_configured(monkeypatch):
    for name in WeChatAdapter.credential_env + WeChatAdapter.verification_env:
        monkeypatch.delenv(name, raising=False)
    record = WeChatAdapter().record()
    assert record.status == "blocked"
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.metadata["live_verified"] is False
    assert record.config_ref == "WEIXIN_APP_ID,WEIXIN_APP_SECRET,WEIXIN_TOKEN,WEIXIN_ENCODING_AES_KEY"
    assert "missing credentials" in record.metadata["reason"]

    set_weixin_env(monkeypatch)
    record = WeChatAdapter().record()
    assert record.status == "configured"
    assert record.metadata["credential_env"] == ["WEIXIN_APP_ID", "WEIXIN_APP_SECRET"]
    assert record.metadata["credentials_present"] == ["WEIXIN_APP_ID", "WEIXIN_APP_SECRET"]
    dump = json.dumps(record.model_dump())
    for value in ENV_VALUES.values():
        assert value not in dump


# ------------------------------------------------ native XML / query / replay


def _native_xml(encrypt: str, timestamp: str, nonce: str, token: str) -> str:
    signature = wechat_msg_signature(token, timestamp, nonce, encrypt)
    return (
        "<xml>"
        f"<ToUserName><![CDATA[gh_fake]]></ToUserName>"
        f"<Encrypt><![CDATA[{encrypt}]]></Encrypt>"
        f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
        f"<TimeStamp>{timestamp}</TimeStamp>"
        f"<Nonce><![CDATA[{nonce}]]></Nonce>"
        "</xml>"
    )


async def test_handle_webhook_accepts_native_xml_envelope(monkeypatch):
    set_weixin_env(monkeypatch)
    timestamp = str(int(time.time()))
    adapter = WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML))
    envelope = _native_xml(ENCRYPT_FIELD, timestamp, "xml-nonce-1", TOKEN)
    assert adapter.verify_webhook(envelope) is True
    messages = await adapter.handle_webhook(envelope)
    assert [(m.sender, m.text, m.message_id) for m in messages] == [
        ("openid-alice", "hello wechat", "789123456")
    ]


async def test_handle_webhook_accepts_query_fields_from_headers(monkeypatch):
    set_weixin_env(monkeypatch)
    timestamp = str(int(time.time()))
    nonce = "header-query-nonce"
    signature = wechat_msg_signature(TOKEN, timestamp, nonce, ENCRYPT_FIELD)
    headers = {
        "MsgSignature": signature,
        "TimeStamp": timestamp,
        "Nonce": nonce,
    }
    payload = {"encrypt": ENCRYPT_FIELD}  # bundled JSON-only route
    adapter = WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML))
    assert adapter.verify_webhook(payload, headers) is True
    messages = await adapter.handle_webhook(payload, headers)
    assert [m.text for m in messages] == ["hello wechat"]


async def test_handle_webhook_rejects_stale_timestamp_and_replay(monkeypatch):
    set_weixin_env(monkeypatch)
    adapter = WeChatAdapter(decryptor=fake_decryptor(INNER_TEXT_XML))
    stale = signed_payload(ENCRYPT_FIELD, TOKEN, timestamp=str(int(time.time()) - 900), nonce="stale-nonce")
    with pytest.raises(WebhookVerificationError, match="stale"):
        await adapter.handle_webhook(stale)

    fresh = signed_payload(ENCRYPT_FIELD, TOKEN, nonce="replay-nonce-wx")
    assert len(await adapter.handle_webhook(fresh)) == 1
    with pytest.raises(WebhookVerificationError, match="replay"):
        await adapter.handle_webhook(fresh)
