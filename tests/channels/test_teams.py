"""Offline verification for the Microsoft Teams channel adapter.

No live network calls: HTTP goes through httpx.MockTransport and webhook
verification uses an injected fake signature verifier, mirroring the
Discord Ed25519 test. The fail-closed story without an injected verifier
is asserted explicitly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from leo.integrations.channels import ALL_ADAPTERS
from leo.integrations.channels.teams import TeamsAdapter
from leo.integrations.messaging import (
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
    verification_health,
)

FAKE_WEBHOOK = "https://contoso.workflow.microsoft.com/workflows/fake-teams-url-should-never-leak"
JWT_SECRET = "fake-botframework-verification-secret"
BOT_APP_ID = "fake-bot-app-id"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_jwt(*, secret: str = JWT_SECRET, aud: str = BOT_APP_ID, iss: str = "https://api.botframework.com",
             exp: float | None = None, iat: float | None = None, alg: str = "HS256") -> str:
    now = time.time()
    header = _b64url(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    claims: dict[str, Any] = {"aud": aud, "iss": iss, "exp": now + 900 if exp is None else exp}
    if iat is not None:
        claims["iat"] = iat
    payload = _b64url(json.dumps(claims).encode())
    signed = f"{header}.{payload}"
    digest = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).digest()
    return f"{signed}.{_b64url(digest)}"


def _verifier(seen: list[tuple[str, str, str, bytes]], *, accept: str):
    """A fake JWT verifier shaped exactly like the Discord test's."""

    def verify(public_key: str, signature: str, timestamp: str, body: bytes) -> bool:
        seen.append((public_key, signature, timestamp, body))
        return signature == accept

    return verify


async def test_teams_verifier_injection_and_fail_closed(monkeypatch):
    monkeypatch.setenv("TEAMS_PUBLIC_KEY", "fake-teams-public-key")
    seen: list[tuple[str, str, str, bytes]] = []
    adapter = TeamsAdapter(signature_verifier=_verifier(seen, accept="good.jwt.token"))
    payload = {"type": "message", "text": "hi"}
    headers = {"X-Teams-Signature": "good.jwt.token"}

    assert adapter.verify_webhook(payload, headers) is True
    assert adapter.verify_webhook(payload, {"X-Teams-Signature": "bad.jwt.token"}) is False
    assert adapter.verify_webhook(payload, {}) is False  # missing signature header
    assert seen[0][0] == "fake-teams-public-key"
    assert seen[0][2] == ""  # the JWT carries its own iat/exp claims
    assert json.loads(seen[0][3]) == payload  # byte-exact body handed to the verifier

    # Fail closed, loudly, when the header is not a valid JWT: LEO never
    # pretends to have validated Microsoft's token.
    with pytest.raises(WebhookVerificationError, match="msteams jwt verification failed"):
        TeamsAdapter().verify_webhook(payload, headers)
    with pytest.raises(WebhookVerificationError):
        await TeamsAdapter().handle_webhook(payload, headers)

    # Fail closed when the key env is absent, even with a verifier.
    monkeypatch.delenv("TEAMS_PUBLIC_KEY")
    assert TeamsAdapter(signature_verifier=_verifier(seen, accept="good.jwt.token")).verify_webhook(
        payload, headers
    ) is False


async def test_teams_handle_webhook_normalizes_activities(monkeypatch):
    monkeypatch.setenv("TEAMS_PUBLIC_KEY", "fake-teams-public-key")
    adapter = TeamsAdapter(signature_verifier=lambda *args: True)
    headers = {"X-Teams-Signature": "any.jwt.value"}
    fresh = datetime.now(timezone.utc).isoformat()
    activity = {
        "type": "message",
        "id": "act-7",
        "timestamp": fresh,
        "replyToId": "act-2",
        "from": {"id": "29:user-1", "name": "Alice"},
        "text": "hello teams",
    }
    messages = await adapter.handle_webhook(activity, headers)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("msteams", "Alice", "hello teams")
    assert message.message_id == "act-7"
    assert message.reply_ref == "act-2"
    assert message.timestamp == fresh

    # Display name absent: sender falls back to from.id.
    no_name = {"type": "message", "id": "act-8", "from": {"id": "29:user-2", "role": "user"}, "text": "anon"}
    normalized = await adapter.handle_webhook(no_name, headers)
    assert normalized[0].sender == "29:user-2" and normalized[0].reply_ref is None

    # Non-message activities and the bot's own activities are skipped.
    assert await adapter.handle_webhook({"type": "conversationUpdate"}, headers) == []
    bot_activity = {"type": "message", "id": "act-9", "from": {"id": "bot-1", "name": "LEO", "role": "bot"}, "text": "self"}
    assert await adapter.handle_webhook(bot_activity, headers) == []

    # An unverified JWT never reaches the parser.
    seen: list[tuple[str, str, str, bytes]] = []
    strict = TeamsAdapter(signature_verifier=_verifier(seen, accept="good.jwt.token"))
    with pytest.raises(WebhookVerificationError):
        await strict.handle_webhook(activity, {"X-Teams-Signature": "bad.jwt.token"})


async def test_teams_handle_webhook_malformed_payload_yields_nothing(monkeypatch):
    monkeypatch.setenv("TEAMS_PUBLIC_KEY", "fake-teams-public-key")
    adapter = TeamsAdapter(signature_verifier=lambda *args: True)
    headers = {"X-Teams-Signature": "any.jwt.value"}
    cases: list[Any] = [None, 42, "not json", b"[", {"type": "typing"}, {"type": "message", "from": {"role": "bot"}}]
    for payload in cases:
        assert await adapter.handle_webhook(payload, headers) == []


async def test_teams_send_posts_incoming_webhook(monkeypatch):
    monkeypatch.setenv("TEAMS_INCOMING_WEBHOOK_URL", FAKE_WEBHOOK)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202)  # Microsoft answers 202 Accepted

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = TeamsAdapter(client=client)
        result = await adapter.send("ignored", "hello")
    assert result == {"channel": "msteams", "target": "ignored", "success": True, "message_id": None}
    request = requests[0]
    assert str(request.url) == FAKE_WEBHOOK
    assert json.loads(request.content) == {"type": "message", "text": "hello"}

    # The legacy office.com connector host shape is accepted too.
    monkeypatch.setenv("TEAMS_INCOMING_WEBHOOK_URL", "https://outlook.office.com/webhook/abc/def")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (await TeamsAdapter(client=client).send("t", "hi"))["success"] is True


async def test_teams_send_rejects_invalid_webhook_urls(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202)

    for bad in (
        "http://outlook.office.com/webhook/x",  # not https
        "https://evil.example/hook",
        f"http://evil.example/hook?webhook={FAKE_WEBHOOK}",
        "https://user:pass@outlook.office.com/webhook/x",
    ):
        monkeypatch.setenv("TEAMS_INCOMING_WEBHOOK_URL", bad)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(MessagingError, match="not a Teams webhook URL"):
                await TeamsAdapter(client=client).send("t", "hi")

    # The webhook URL is a credential: the rejection never echoes it.
    monkeypatch.setenv("TEAMS_INCOMING_WEBHOOK_URL", f"https://evil.example/hook?u={FAKE_WEBHOOK}")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MessagingError) as exc:
            await TeamsAdapter(client=client).send("t", "hi")
    assert "not a Teams webhook URL" in str(exc.value)
    assert FAKE_WEBHOOK not in str(exc.value)


async def test_teams_send_blocked_and_http_error(monkeypatch):
    monkeypatch.delenv("TEAMS_INCOMING_WEBHOOK_URL", raising=False)
    with pytest.raises(MessagingError) as exc:
        await TeamsAdapter().send("t", "hi")
    assert "TEAMS_INCOMING_WEBHOOK_URL is not set" in str(exc.value)
    assert "msteams send blocked" in str(exc.value)

    monkeypatch.setenv("TEAMS_INCOMING_WEBHOOK_URL", FAKE_WEBHOOK)

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad request for {FAKE_WEBHOOK}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError) as exc:
            await TeamsAdapter(client=client).send("t", "hi")
    assert "HTTP 400" in str(exc.value)
    assert FAKE_WEBHOOK not in str(exc.value)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused while using {FAKE_WEBHOOK}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
        with pytest.raises(MessagingError) as exc:
            await TeamsAdapter(client=client).send("t", "hi")
    assert "ConnectError" in str(exc.value)
    assert FAKE_WEBHOOK not in str(exc.value)


def test_teams_record_blocked_and_configured(monkeypatch):
    monkeypatch.delenv("TEAMS_INCOMING_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TEAMS_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("TEAMS_APP_ID", raising=False)
    adapter = TeamsAdapter()
    assert ALL_ADAPTERS["msteams"] is TeamsAdapter  # auto-discovery registered it

    record = adapter.record()
    assert record.status == "blocked"
    assert record.kind == "messaging"
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.config_ref == "TEAMS_INCOMING_WEBHOOK_URL,TEAMS_PUBLIC_KEY,TEAMS_APP_ID"
    assert record.metadata["verification_env"] == ["TEAMS_PUBLIC_KEY", "TEAMS_APP_ID"]
    assert record.metadata["live_verified"] is False
    assert "missing credentials" in record.metadata["reason"]
    assert record.metadata["verification_status"] == "degraded"
    assert "TEAMS_APP_ID" in record.metadata["missing_verification"]

    monkeypatch.setenv("TEAMS_INCOMING_WEBHOOK_URL", FAKE_WEBHOOK)
    record = adapter.record()
    assert record.status == "configured"  # audience material is not a send credential
    assert record.metadata["reason"] is None
    assert record.metadata["verification_status"] == "degraded"  # still missing JWT audience
    assert FAKE_WEBHOOK not in json.dumps(record.model_dump())


def test_teams_verification_health_degrades_without_app_id(monkeypatch):
    monkeypatch.setenv("TEAMS_PUBLIC_KEY", JWT_SECRET)
    monkeypatch.delenv("TEAMS_APP_ID", raising=False)
    health = verification_health(TeamsAdapter())
    assert health["status"] == "degraded"
    assert health["missing_verification"] == ["TEAMS_APP_ID"]
    assert "TEAMS_APP_ID" in health["reason"]
    assert JWT_SECRET not in str(health)

    monkeypatch.setenv("TEAMS_APP_ID", BOT_APP_ID)
    healthy = verification_health(TeamsAdapter())
    assert healthy["status"] == "ready"
    assert healthy["missing_verification"] == []
    assert healthy["reason"] is None


def _jwt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEAMS_PUBLIC_KEY", JWT_SECRET)
    monkeypatch.setenv("TEAMS_APP_ID", BOT_APP_ID)
    monkeypatch.delenv("TEAMS_JWT_ISSUER", raising=False)


def test_default_teams_verifier_validates_real_hs256_jwt(monkeypatch):
    """The default configuration accepts a genuine signed token, not nothing."""
    _jwt_env(monkeypatch)
    adapter = TeamsAdapter()
    good = make_jwt()
    assert adapter.verify_webhook({"type": "message"}, {"Authorization": f"Bearer {good}"}) is True
    # A fronting gateway's copied header is accepted too.
    assert adapter.verify_webhook({"type": "message"}, {"X-Teams-Signature": good}) is True
    assert adapter.verify_webhook({"type": "message"}, {}) is False


def test_default_teams_verifier_rejects_bad_claims_and_signature(monkeypatch):
    _jwt_env(monkeypatch)
    adapter = TeamsAdapter()
    bad_tokens = {
        "audience": make_jwt(aud="some-other-bot"),
        "issuer": make_jwt(iss="https://evil.example"),
        "expired": make_jwt(exp=time.time() - 10),
        "stale iat": make_jwt(iat=time.time() - 7200),
        "bad signature": make_jwt(secret="wrong-secret"),
        "rs256": make_jwt(alg="RS256"),
        "malformed": "not.a.jwt",
    }
    for reason, token in bad_tokens.items():
        with pytest.raises(WebhookVerificationError) as exc:
            adapter.verify_webhook({}, {"Authorization": f"Bearer {token}"})
        assert JWT_SECRET not in str(exc.value)
        assert token not in str(exc.value)

    # Missing audience configuration fails closed as well.
    monkeypatch.delenv("TEAMS_APP_ID")
    with pytest.raises(WebhookVerificationError, match="TEAMS_APP_ID"):
        TeamsAdapter().verify_webhook({}, {"Authorization": f"Bearer {make_jwt()}"})

    # A custom issuer is honoured when configured.
    monkeypatch.setenv("TEAMS_APP_ID", BOT_APP_ID)
    monkeypatch.setenv("TEAMS_JWT_ISSUER", "https://sts.windows.net/fake-tenant/")
    assert adapter.verify_webhook(
        {}, {"Authorization": f"Bearer {make_jwt(iss='https://sts.windows.net/fake-tenant/')}"}
    ) is True


async def test_default_teams_configuration_normalizes_verified_activity(monkeypatch):
    _jwt_env(monkeypatch)
    adapter = TeamsAdapter()  # no injected verifier
    token = make_jwt()
    activity = {
        "type": "message",
        "id": "act-default-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from": {"id": "29:u", "name": "Alice"},
        "text": "real default path",
    }
    messages = await adapter.handle_webhook(activity, {"Authorization": f"Bearer {token}"})
    assert [(m.sender, m.text, m.message_id) for m in messages] == [
        ("Alice", "real default path", "act-default-1")
    ]

    # The same activity id is a replay even with a freshly signed token.
    again = {**activity, "timestamp": datetime.now(timezone.utc).isoformat()}
    with pytest.raises(WebhookVerificationError, match="replay"):
        await adapter.handle_webhook(again, {"Authorization": f"Bearer {make_jwt()}"})
