"""Channel adapters must read credentials through LEO's runtime broker.

The dashboard's "enable channel" flow installs values into ``default_store``
at runtime without touching ``os.environ``. These tests prove that three
different channel modules (Slack, Relay, Email) actually consult that broker
first, still fall back to the environment, and never leak a broker value into
a record dump.

Values are installed directly into the process-local dict via
``default_store._memory`` (the autouse fixture in ``tests/conftest.py`` clears
it between tests), which keeps the tests independent of any event loop.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx

from leo.integrations.channels.email import EmailAdapter
from leo.integrations.channels.relay import RelayAdapter
from leo.integrations.channels.slack import SlackAdapter
from leo.integrations.secrets import credential, default_store

# ---------------------------------------------------------------- broker helpers


def _install(**creds: str) -> None:
    """Install broker values the way the enrollment flow does, no env vars."""
    for name, value in creds.items():
        default_store._memory[name] = value


def _relay_sig(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


# ------------------------------------------------------------------ 1. Slack send


async def test_slack_send_uses_broker_token_and_ignores_env(monkeypatch):
    """SLACK_BOT_TOKEN from the broker wins over the environment."""
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "env-token-should-lose")
    _install(SLACK_BOT_TOKEN="broker-fake")

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "ts": "1700000000.000100"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SlackAdapter(client=client).send("C123", "hello")

    assert result["success"] is True
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer broker-fake"
    assert "env-token-should-lose" not in requests[0].headers["authorization"]


# -------------------------------------------------- 2. Relay webhook verification


async def test_relay_webhook_verification_uses_broker_secret(monkeypatch):
    """RELAY_SECRET installed in the broker verifies, the env value does not."""
    monkeypatch.delenv("RELAY_SECRET", raising=False)
    _install(RELAY_SECRET="broker-relay-secret")

    payload = {"sender": "leo", "text": "hello"}
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    good = {"x-leo-signature": _relay_sig("broker-relay-secret", body)}
    wrong = {"x-leo-signature": _relay_sig("env-relay-secret", body)}
    adapter = RelayAdapter()

    assert adapter.verify_webhook(payload, good, raw_body=body) is True
    assert adapter.verify_webhook(payload, wrong, raw_body=body) is False
    # Even with a conflicting env value, the broker secret is the one used.
    monkeypatch.setenv("RELAY_SECRET", "env-relay-secret")
    assert adapter.verify_webhook(payload, good, raw_body=body) is True
    assert adapter.verify_webhook(payload, wrong, raw_body=body) is False


# ------------------------------------------------------------- 3. Email (poll)


_BROKER_EMAIL = {
    "SMTP_HOST": "broker.smtp.test",
    "SMTP_USER": "broker-smtp-user",
    "SMTP_PASSWORD": "broker-smtp-secret-never-leak",
    "EMAIL_FROM": "broker@leo.test",
    "IMAP_HOST": "broker.imap.test",
    "IMAP_USER": "broker-imap-user",
    "IMAP_PASSWORD": "broker-imap-secret-never-leak",
}


class _FakeSmtp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def send_message(self, from_addr: str, to_addr: str, subject: str, text: str) -> None:
        self.calls.append((from_addr, to_addr, text))


def test_email_record_reflects_broker_credentials(monkeypatch):
    """record() treats broker-installed SMTP/IMAP creds as configured."""
    for name in _BROKER_EMAIL:
        monkeypatch.delenv(name, raising=False)
    _install(**_BROKER_EMAIL)

    record = EmailAdapter().record()
    assert record.status == "configured"
    assert record.metadata["poll_credentials_present"] == ["IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD"]
    assert "SMTP_HOST" in record.metadata["credentials_present"]
    assert record.metadata["reason"] is None
    dumped = json.dumps(record.model_dump())
    assert "broker-smtp-secret-never-leak" not in dumped
    assert "broker-imap-secret-never-leak" not in dumped


async def test_email_send_reads_broker_credentials(monkeypatch):
    """The SMTP send path also consults the broker, with no env vars set."""
    for name in _BROKER_EMAIL:
        monkeypatch.delenv(name, raising=False)
    _install(**_BROKER_EMAIL)

    smtp = _FakeSmtp()
    result = await EmailAdapter(smtp_transport=smtp).send("friend@example.test", "hi")
    assert result["success"] is True
    assert smtp.calls == [("broker@leo.test", "friend@example.test", "hi")]


# ---------------------------------------------------------------- env fallback


async def test_env_fallback_when_broker_has_no_value(monkeypatch):
    """With no broker value, credential() still reads os.environ."""
    default_store._memory.pop("SLACK_BOT_TOKEN", None)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "env-only-token")

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "ts": "1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await SlackAdapter(client=client).send("C1", "hi")

    assert requests[0].headers["authorization"] == "Bearer env-only-token"


# ------------------------------------------------------ enrollment via set()/


def test_broker_set_enrollment_reaches_adapter(monkeypatch):
    """The synchronous enrollment API installs a value the broker returns."""
    monkeypatch.delenv("LEO_CREDENTIALS_KEY", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    mode = default_store.set("SLACK_BOT_TOKEN", "enrolled-by-dashboard")
    assert mode == "memory"
    assert credential("SLACK_BOT_TOKEN") == "enrolled-by-dashboard"


# ---------------------------------------------------------------- no leakage


def test_slack_record_never_embeds_broker_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    _install(SLACK_BOT_TOKEN="broker-token-never-leak")

    record = SlackAdapter().record()
    assert record.status == "configured"
    assert "broker-token-never-leak" not in json.dumps(record.model_dump())
