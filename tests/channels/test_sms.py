"""Offline tests for the SMS (Twilio-compatible) channel adapter.

No test touches a network: HTTP goes through ``httpx.MockTransport``
and X-Twilio-Signature values are computed locally with Twilio's
documented algorithm, independently of the adapter's implementation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from urllib.parse import parse_qsl, urlencode

import httpx
import pytest

from leo.integrations.channels.sms import SMSAdapter
from leo.integrations.messaging import (
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
)

SMS_ENV = {
    "TWILIO_ACCOUNT_SID": "ACfake-account-sid",
    "TWILIO_AUTH_TOKEN": "fake-auth-token-never-leak",
    "TWILIO_FROM_NUMBER": "+15550001111",
}
WEBHOOK_URL = "https://leo.example.test/webhooks/sms"
INBOUND = {
    "From": "+15552223333",
    "To": "+15554445555",
    "Body": "Hello LEO",
    "MessageSid": "SMfake42",
    "FromCity": "Boston",
    "ToCity": "Seattle",
    "SmsStatus": "received",
}


def twilio_signature(token: str, url: str, params: dict[str, str]) -> str:
    """Twilio's documented algorithm, implemented independently of the adapter:
    base64(HMAC-SHA1(token, url + params sorted by key, concatenated k+v))."""
    signed = url + "".join(f"{key}{params[key]}" for key in sorted(params))
    return base64.b64encode(hmac.new(token.encode("utf-8"), signed.encode("utf-8"), hashlib.sha1).digest()).decode(
        "ascii"
    )


def _inbound_raw(params: dict[str, str] | None = None) -> bytes:
    return urlencode(params if params is not None else INBOUND).encode("utf-8")


# --------------------------------------------------------------------------- verify


async def test_verify_webhook_accepts_documented_signature(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "the-token")
    monkeypatch.setenv("TWILIO_WEBHOOK_URL", WEBHOOK_URL)
    adapter = SMSAdapter()
    raw = _inbound_raw()
    signature = twilio_signature("the-token", WEBHOOK_URL, INBOUND)
    assert adapter.verify_webhook(raw, {"X-Twilio-Signature": signature}, raw_body=raw) is True
    assert adapter.verify_webhook(raw, {"X-Twilio-Signature": "AAAAwrongAAAA=="}, raw_body=raw) is False
    # a correctly formed signature from a different token is still a mismatch
    assert (
        adapter.verify_webhook(
            raw,
            {"X-Twilio-Signature": twilio_signature("other-token", WEBHOOK_URL, INBOUND)},
            raw_body=raw,
        )
        is False
    )


async def test_verify_webhook_payload_dict_fallback(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "the-token")
    monkeypatch.setenv("TWILIO_WEBHOOK_URL", WEBHOOK_URL)
    adapter = SMSAdapter()
    signature = twilio_signature("the-token", WEBHOOK_URL, INBOUND)
    # raw_body=None: the params come from the payload mapping
    assert adapter.verify_webhook(INBOUND, {"X-Twilio-Signature": signature}) is True
    # a JSON string is NOT Twilio's documented form encoding: fails closed
    assert adapter.verify_webhook(json.dumps(INBOUND), {"X-Twilio-Signature": signature}) is False


async def test_verify_webhook_fails_closed(monkeypatch):
    raw = _inbound_raw()
    signature = twilio_signature("the-token", WEBHOOK_URL, INBOUND)

    # token missing: nothing can be verified -> fail closed when required;
    # only an explicit require_verification=False opts out
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    required = SMSAdapter(require_verification=True)
    assert required.verify_webhook(raw, {"X-Twilio-Signature": signature}, raw_body=raw) is False
    assert SMSAdapter(require_verification=False).verify_webhook(raw, {}, raw_body=raw) is True

    # token present but the public callback URL is unknown: the signed string
    # cannot be constructed -> fail closed
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "the-token")
    monkeypatch.delenv("TWILIO_WEBHOOK_URL", raising=False)
    assert required.verify_webhook(raw, {"X-Twilio-Signature": signature}, raw_body=raw) is False

    # URL present but header missing: hard fail even with valid config
    monkeypatch.setenv("TWILIO_WEBHOOK_URL", WEBHOOK_URL)
    assert required.verify_webhook(raw, {}, raw_body=raw) is False


async def test_verify_webhook_tolerates_hostile_input(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "the-token")
    monkeypatch.setenv("TWILIO_WEBHOOK_URL", WEBHOOK_URL)
    adapter = SMSAdapter()
    raw = _inbound_raw()
    # non-ASCII header would make compare_digest raise: treated as a mismatch
    assert adapter.verify_webhook(raw, {"X-Twilio-Signature": "sígnature"}, raw_body=raw) is False
    # junk payload with no raw body: never crashes, never verifies
    assert adapter.verify_webhook(None, {"X-Twilio-Signature": "AAAA"}, raw_body=None) is False


# --------------------------------------------------------------------------- webhook


async def test_handle_webhook_normalizes_inbound_sms(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "the-token")
    monkeypatch.setenv("TWILIO_WEBHOOK_URL", WEBHOOK_URL)
    adapter = SMSAdapter()
    raw = _inbound_raw()
    headers = {"X-Twilio-Signature": twilio_signature("the-token", WEBHOOK_URL, INBOUND)}
    messages = await adapter.handle_webhook(raw, headers, raw_body=raw)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("sms", "+15552223333", "Hello LEO")
    assert message.message_id == "SMfake42"
    assert message.reply_ref is None
    assert message.timestamp  # inbound webhooks are stamped at receipt time

    # tampered body: the signature no longer matches the verified parameters
    tampered = _inbound_raw({**INBOUND, "Body": "tampered"})
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(tampered, headers, raw_body=tampered)

    # signed but empty payload: nothing SMS-like to normalize
    empty_headers = {"X-Twilio-Signature": twilio_signature("the-token", WEBHOOK_URL, {})}
    assert await adapter.handle_webhook(b"", empty_headers, raw_body=b"") == []

    # unsigned request never reaches normalization
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(raw, {}, raw_body=raw)


# --------------------------------------------------------------------------- send


async def test_send_posts_twilio_documented_form_with_basic_auth(monkeypatch):
    for name, value in SMS_ENV.items():
        monkeypatch.setenv(name, value)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"sid": "SMoutgoing1", "status": "queued"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SMSAdapter(client=client).send("+15552223333", "hi from LEO")
    assert result == {"channel": "sms", "target": "+15552223333", "success": True, "message_id": "SMoutgoing1"}

    request = requests[0]
    assert str(request.url) == "https://api.twilio.com/2010-04-01/Accounts/ACfake-account-sid/Messages.json"
    auth_header = request.headers["authorization"]
    assert auth_header.startswith("Basic ")
    decoded = base64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8")
    assert decoded == "ACfake-account-sid:fake-auth-token-never-leak"
    form = dict(parse_qsl(request.content.decode("utf-8"), keep_blank_values=True))
    assert form == {"From": "+15550001111", "To": "+15552223333", "Body": "hi from LEO"}


async def test_send_honours_twilio_base_url_override(monkeypatch):
    for name, value in SMS_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TWILIO_BASE_URL", "https://sms-proxy.example.test")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"sid": "SMproxy1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await SMSAdapter(client=client).send("+15552223333", "hi")
    assert str(requests[0].url).startswith("https://sms-proxy.example.test/2010-04-01/Accounts/ACfake-account-sid/")


async def test_send_blocked_per_missing_env_name(monkeypatch):
    for name, value in SMS_ENV.items():
        monkeypatch.setenv(name, value)
    for missing in SMS_ENV:
        monkeypatch.delenv(missing)
        with pytest.raises(MessagingError) as exc:
            # no client injected: the env check must fire before any client
            # is even constructed, so no network can be touched
            await SMSAdapter().send("+15552223333", "hi")
        assert missing in str(exc.value)
        assert "fake-auth-token-never-leak" not in str(exc.value)
        monkeypatch.setenv(missing, SMS_ENV[missing])


async def test_send_surfaces_http_error_code(monkeypatch):
    for name, value in SMS_ENV.items():
        monkeypatch.setenv(name, value)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": 21211, "message": "The 'To' number is not valid"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MessagingError) as exc:
            await SMSAdapter(client=client).send("not-a-number", "hi")
    assert "HTTP 400" in str(exc.value)
    assert "fake-auth-token-never-leak" not in str(exc.value)


async def test_send_wraps_network_errors_without_leaking(monkeypatch):
    for name, value in SMS_ENV.items():
        monkeypatch.setenv(name, value)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("api.twilio.com unreachable; fake-auth-token-never-leak")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MessagingError) as exc:
            await SMSAdapter(client=client).send("+15552223333", "hi")
    assert "ConnectError" in str(exc.value)
    assert "fake-auth-token-never-leak" not in str(exc.value)
    assert "api.twilio.com" not in str(exc.value)


# --------------------------------------------------------------------------- record


def test_record_blocked_without_credentials(monkeypatch):
    for name in SMS_ENV:
        monkeypatch.delenv(name, raising=False)
    record = SMSAdapter().record()
    assert (record.name, record.kind, record.status) == ("sms", "messaging", "blocked")
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.metadata["live_verified"] is False
    assert "TWILIO_ACCOUNT_SID" in record.metadata["reason"]


def test_record_configured_lists_env_names_and_hides_token(monkeypatch):
    for name, value in SMS_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TWILIO_WEBHOOK_URL", raising=False)
    record = SMSAdapter().record()
    assert record.status == "configured"
    # the auth token doubles as the inbound signing secret: listed once, by NAME
    assert record.config_ref == "TWILIO_ACCOUNT_SID,TWILIO_AUTH_TOKEN,TWILIO_FROM_NUMBER"
    assert record.metadata["verification_env"] == ["TWILIO_AUTH_TOKEN"]
    assert record.metadata["webhook_url_env"] == "TWILIO_WEBHOOK_URL"
    assert record.metadata["webhook_url_present"] is False
    dumped = json.dumps(record.model_dump())
    assert "fake-auth-token-never-leak" not in dumped


# --------------------------------------------------------------------------- registry


def test_registered_as_webhook_channel():
    from leo.integrations.channels import ALL_ADAPTERS, polling_channels, webhook_channels

    assert ALL_ADAPTERS["sms"] is SMSAdapter
    assert "sms" in webhook_channels()
    assert "sms" not in polling_channels()
