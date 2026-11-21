"""Offline tests for the email channel adapter.

No test touches a network: SMTP and IMAP are replaced by fakes, the
default transports fail fast on missing env before ever opening a
socket, and the adapter itself never creates an HTTP client.
"""

from __future__ import annotations

import json

import pytest

from leo.integrations.channels.email import EmailAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage

SMTP_ENV = {
    "SMTP_HOST": "smtp.example.test",
    "SMTP_USER": "smtp-user@example.test",
    "SMTP_PASSWORD": "fake-smtp-password-never-leak",
    "EMAIL_FROM": "leo@example.test",
}
IMAP_ENV = {
    "IMAP_HOST": "imap.example.test",
    "IMAP_USER": "imap-user@example.test",
    "IMAP_PASSWORD": "fake-imap-password-never-leak",
}
ALL_EMAIL_ENV = {**SMTP_ENV, **IMAP_ENV}

CANNED_MESSAGES = [
    {
        "uid": "101",
        "from": "Alice <alice@example.com>",
        "subject": "Hello from Alice",
        "body": "first unseen message",
        "date": "Wed, 18 Nov 2026 08:30:00 +0000",
        "authentication_results": "mx.example.test; spf=pass smtp.mailfrom=example.com; dkim=pass header.d=example.com",
        "received_spf": "pass (sender SPF authorized) client-ip=192.0.2.1; envelope-from=alice@example.com",
    },
    {
        "uid": "102",
        "from": "bob@example.com",
        "subject": "Quick question",
        "body": "second unseen message",
        "date": "not a real date",
        "authentication_results": "mx.example.test; dkim=pass header.from=example.com",
        "received_spf": "",
    },
]


def _set_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _clear_env(monkeypatch: pytest.MonkeyPatch, names: dict[str, str] | list[str]) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


class FakeSmtpTransport:
    """Records what the adapter asked it to send."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def send_message(self, from_addr: str, to_addr: str, subject: str, text: str) -> None:
        self.calls.append((from_addr, to_addr, subject, text))


class FakeImapTransport:
    """Returns canned UNSEEN messages; counts polls."""

    def __init__(self, messages: list[dict] | None = None) -> None:
        self.messages: list[dict] = list(messages or [])
        self.calls = 0

    def fetch_unseen(self) -> list[dict]:
        self.calls += 1
        return list(self.messages)


class ExplodingSmtpTransport:
    def send_message(self, from_addr: str, to_addr: str, subject: str, text: str) -> None:
        raise ConnectionRefusedError("smtp.example.test:9999 refused; fake-smtp-password-never-leak")


class ExplodingImapTransport:
    def fetch_unseen(self) -> list[dict]:
        raise RuntimeError("boom; fake-imap-password-never-leak")


# --------------------------------------------------------------------------- send


async def test_send_uses_injected_smtp_transport(monkeypatch):
    _set_env(monkeypatch, SMTP_ENV)
    smtp = FakeSmtpTransport()
    adapter = EmailAdapter(smtp_transport=smtp)
    result = await adapter.send("friend@example.test", "hello there")
    assert result == {"channel": "email", "target": "friend@example.test", "success": True, "message_id": None}
    assert smtp.calls == [("leo@example.test", "friend@example.test", "LEO message", "hello there")]


async def test_send_blocked_per_missing_env_name(monkeypatch):
    _set_env(monkeypatch, SMTP_ENV)
    smtp = FakeSmtpTransport()
    for missing in SMTP_ENV:
        monkeypatch.delenv(missing)
        with pytest.raises(MessagingError) as exc:
            await EmailAdapter(smtp_transport=smtp).send("friend@example.test", "hi")
        assert missing in str(exc.value)
        assert "fake-smtp-password-never-leak" not in str(exc.value)
        monkeypatch.setenv(missing, SMTP_ENV[missing])
    assert smtp.calls == []  # blocked before any transport call


async def test_send_blocked_with_no_env_at_all(monkeypatch):
    _clear_env(monkeypatch, ALL_EMAIL_ENV)
    with pytest.raises(MessagingError, match="SMTP_HOST"):
        await EmailAdapter(smtp_transport=FakeSmtpTransport()).send("friend@example.test", "hi")


async def test_send_rejects_invalid_email_addresses(monkeypatch):
    _set_env(monkeypatch, SMTP_ENV)
    smtp = FakeSmtpTransport()
    adapter = EmailAdapter(smtp_transport=smtp)
    for bad in ("no-at-sign", "a@b\nnewline", "", None, "@example.test", "user@", "two@ats@here"):
        with pytest.raises(MessagingError, match="invalid email address"):
            await adapter.send(bad, "hi")
    assert smtp.calls == []


async def test_send_wraps_transport_errors_without_leaking(monkeypatch):
    _set_env(monkeypatch, SMTP_ENV)
    adapter = EmailAdapter(smtp_transport=ExplodingSmtpTransport())
    with pytest.raises(MessagingError) as exc:
        await adapter.send("friend@example.test", "hi")
    assert "ConnectionRefusedError" in str(exc.value)
    assert "fake-smtp-password-never-leak" not in str(exc.value)
    assert "smtp.example.test" not in str(exc.value)


# --------------------------------------------------------------------------- poll


async def test_poll_normalizes_unseen_messages(monkeypatch):
    imap = FakeImapTransport(CANNED_MESSAGES)
    adapter = EmailAdapter(imap_transport=imap)
    first = await adapter.poll()
    assert len(first) == 2
    assert all(isinstance(message, NormalizedMessage) for message in first)
    alice, bob = first
    assert (alice.channel, alice.sender, alice.text) == (
        "email",
        "alice@example.com",  # parseaddr strips the display name
        "Hello from Alice\n\nfirst unseen message",
    )
    assert alice.message_id == "101" and alice.reply_ref is None
    assert alice.timestamp == "2026-11-18T08:30:00+00:00"  # parsed RFC 2822 date
    assert (bob.channel, bob.sender, bob.text) == (
        "email",
        "bob@example.com",
        "Quick question\n\nsecond unseen message",
    )
    assert bob.message_id == "102"
    assert bob.timestamp  # unparseable date falls back to utcnow()


async def test_poll_cursor_skips_already_seen_uids(monkeypatch):
    imap = FakeImapTransport(CANNED_MESSAGES)
    adapter = EmailAdapter(imap_transport=imap)
    assert len(await adapter.poll()) == 2
    assert await adapter.poll() == []  # cursor: same uids are never returned twice
    imap.messages = CANNED_MESSAGES + [
        {"uid": "103", "from": "carol@example.com", "subject": "New", "body": "third message", "date": ""}
    ]
    third = await adapter.poll()
    assert [message.message_id for message in third] == ["103"]
    assert imap.calls == 3
    # a fresh instance (new-process semantics) starts with an empty cursor
    # and re-reads everything the server still lists as UNSEEN
    fresh = await EmailAdapter(imap_transport=FakeImapTransport(CANNED_MESSAGES)).poll()
    assert len(fresh) == 2


async def test_poll_skips_and_coerces_malformed_items(monkeypatch):
    imap = FakeImapTransport(
        [
            "not-a-dict",
            {"uid": None, "from": "x@example.com", "subject": "s", "body": "b", "date": ""},
            {"uid": "  ", "from": "x@example.com", "subject": "s", "body": "b", "date": ""},
            {
                "uid": 201,  # int uid is coerced to its string form
                "from": 123,  # parseaddr leniently keeps a bare token as the address
                "subject": None,
                "body": None,
                "date": None,
            },
        ]
    )
    messages = await EmailAdapter(imap_transport=imap).poll()
    assert len(messages) == 1
    only = messages[0]
    assert only.message_id == "201"
    assert only.sender == "unverified:123"  # no SPF/DKIM evidence: degraded identity
    assert only.text == "\n\n"  # None subject and body coerce to empty strings


async def test_poll_caps_body_at_10000_chars(monkeypatch):
    imap = FakeImapTransport(
        [{"uid": "300", "from": "a@example.com", "subject": "s", "body": "x" * 15_000, "date": ""}]
    )
    message = (await EmailAdapter(imap_transport=imap).poll())[0]
    body = message.text.split("\n\n", 1)[1]
    assert len(body) == 10_000


async def test_poll_wraps_transport_errors_without_leaking(monkeypatch):
    adapter = EmailAdapter(imap_transport=ExplodingImapTransport())
    with pytest.raises(MessagingError) as exc:
        await adapter.poll()
    assert "RuntimeError" in str(exc.value)
    assert "fake-imap-password-never-leak" not in str(exc.value)


async def test_default_transports_fail_closed_without_env(monkeypatch):
    """Default smtplib/imaplib transports raise on missing env before any socket."""
    _clear_env(monkeypatch, ALL_EMAIL_ENV)
    adapter = EmailAdapter()  # no fakes injected: default transports
    with pytest.raises(MessagingError, match="SMTP_HOST"):
        await adapter.send("friend@example.test", "hi")
    with pytest.raises(MessagingError, match="IMAP_HOST"):
        await adapter.poll()


async def test_email_has_no_webhook_ingress(monkeypatch):
    adapter = EmailAdapter(smtp_transport=FakeSmtpTransport(), imap_transport=FakeImapTransport())
    assert EmailAdapter.supports_webhook is False
    assert EmailAdapter.supports_polling is True
    assert adapter.verify_webhook({}, {}) is False
    with pytest.raises(MessagingError, match="webhook"):
        await adapter.handle_webhook({}, {})


# --------------------------------------------------------------------------- record


def _record_adapter() -> EmailAdapter:
    return EmailAdapter(smtp_transport=FakeSmtpTransport(), imap_transport=FakeImapTransport())


def test_record_blocked_without_any_env(monkeypatch):
    _clear_env(monkeypatch, ALL_EMAIL_ENV)
    record = _record_adapter().record()
    assert (record.name, record.kind, record.status) == ("email", "messaging", "blocked")
    assert record.capabilities == ["send", "poll"]
    assert record.verification == "mock-tested"
    assert record.metadata["live_verified"] is False
    assert "SMTP_HOST" in record.metadata["reason"]
    assert "IMAP_HOST" in record.metadata["reason"]


def test_record_blocked_when_only_one_half_is_configured(monkeypatch):
    _set_env(monkeypatch, SMTP_ENV)
    _clear_env(monkeypatch, IMAP_ENV)
    record = _record_adapter().record()
    assert record.status == "blocked"  # "configured" only when BOTH halves exist
    assert record.metadata["reason"] == "missing poll credentials: IMAP_HOST, IMAP_USER, IMAP_PASSWORD"

    _clear_env(monkeypatch, SMTP_ENV)
    _set_env(monkeypatch, IMAP_ENV)
    record = _record_adapter().record()
    assert record.status == "blocked"
    assert record.metadata["reason"] == "missing send credentials: SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM"


def test_record_configured_lists_both_halves_and_hides_secrets(monkeypatch):
    _set_env(monkeypatch, ALL_EMAIL_ENV)
    record = _record_adapter().record()
    assert record.status == "configured"
    assert record.capabilities == ["send", "poll"]
    assert record.config_ref == "SMTP_HOST,SMTP_USER,SMTP_PASSWORD,EMAIL_FROM,IMAP_HOST,IMAP_USER,IMAP_PASSWORD"
    assert record.metadata["poll_env"] == ["IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD"]
    assert record.metadata["poll_credentials_present"] == ["IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD"]
    assert record.metadata["reason"] is None
    dumped = json.dumps(record.model_dump())
    assert "fake-smtp-password-never-leak" not in dumped
    assert "fake-imap-password-never-leak" not in dumped


# --------------------------------------------------------------------------- registry


def test_registered_as_polling_channel():
    from leo.integrations.channels import ALL_ADAPTERS, polling_channels, webhook_channels

    assert ALL_ADAPTERS["email"] is EmailAdapter
    assert "email" in polling_channels()
    assert "email" not in webhook_channels()


# ----------------------------------------------------------- transport security


class FakeSmtpBase:
    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
        self.starttls_calls: list = []
        self.ehlo_calls = 0
        self.login_calls: list[tuple[str, str]] = []
        self.has_starttls = True
        self.sent: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo_or_helo_if_needed(self):
        self.ehlo_calls += 1

    def ehlo(self):
        self.ehlo_calls += 1

    def has_extn(self, name):
        return name == "starttls" and self.has_starttls

    def starttls(self, context=None):
        self.starttls_calls.append(context)

    def login(self, user, password):
        self.login_calls.append((user, password))

    def send_message(self, message):
        self.sent.append(message)


class FakeImapBase:
    def __init__(self, host, port, timeout=None, ssl_context=None):
        self.host, self.port, self.timeout, self.ssl_context = host, port, timeout, ssl_context
        self.starttls_calls: list = []
        self.login_calls: list[tuple[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, ssl_context=None):
        self.starttls_calls.append(ssl_context)

    def login(self, user, password):
        self.login_calls.append((user, password))

    def select(self, mailbox):
        return "OK", [b"1"]

    def uid(self, command, *args):
        return ("OK", [b"1"]) if command == "SEARCH" else ("OK", [])


async def test_default_smtp_uses_verified_starttls_and_never_plaintext_login(monkeypatch):
    import ssl

    import leo.integrations.channels.email as email_mod

    _set_env(monkeypatch, SMTP_ENV)
    monkeypatch.setenv("SMTP_PORT", "587")
    instances: list[FakeSmtpBase] = []

    def factory(*args, **kwargs):
        instance = FakeSmtpBase(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(email_mod.smtplib, "SMTP", factory)
    email_mod._SmtplibTransport().send_message("from@example.test", "to@example.test", "s", "t")
    smtp = instances[0]
    assert len(smtp.starttls_calls) == 1
    context = smtp.starttls_calls[0]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    assert smtp.ehlo_calls >= 2  # EHLO before and after STARTTLS
    assert smtp.login_calls == [(SMTP_ENV["SMTP_USER"], SMTP_ENV["SMTP_PASSWORD"])]
    assert len(smtp.sent) == 1

    # A server without STARTTLS is refused before any AUTH command.
    instances.clear()

    def no_tls_factory(*args, **kwargs):
        instance = FakeSmtpBase(*args, **kwargs)
        instance.has_starttls = False
        instances.append(instance)
        return instance

    monkeypatch.setattr(email_mod.smtplib, "SMTP", no_tls_factory)
    with pytest.raises(MessagingError, match="STARTTLS"):
        email_mod._SmtplibTransport().send_message("from@example.test", "to@example.test", "s", "t")
    assert instances[0].login_calls == []


async def test_default_smtp_implicit_tls_on_465(monkeypatch):
    import ssl

    import leo.integrations.channels.email as email_mod

    _set_env(monkeypatch, SMTP_ENV)
    monkeypatch.setenv("SMTP_PORT", "465")
    instances: list[FakeSmtpBase] = []

    def factory(*args, **kwargs):
        instance = FakeSmtpBase(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(email_mod.smtplib, "SMTP", lambda *a, **kw: pytest.fail("cleartext SMTP used on 465"))
    monkeypatch.setattr(email_mod.smtplib, "SMTP_SSL", factory)
    email_mod._SmtplibTransport().send_message("from@example.test", "to@example.test", "s", "t")
    assert instances[0].port == 465
    assert isinstance(instances[0].context, ssl.SSLContext)
    assert instances[0].context.verify_mode == ssl.CERT_REQUIRED
    assert instances[0].starttls_calls == []
    assert instances[0].login_calls == [(SMTP_ENV["SMTP_USER"], SMTP_ENV["SMTP_PASSWORD"])]


async def test_default_imap_uses_implicit_tls_on_993_and_starttls_elsewhere(monkeypatch):
    import ssl

    import leo.integrations.channels.email as email_mod

    _set_env(monkeypatch, IMAP_ENV)
    monkeypatch.delenv("IMAP_PORT", raising=False)
    ssl_instances: list[FakeImapBase] = []
    plain_instances: list[FakeImapBase] = []

    def ssl_factory(*args, **kwargs):
        instance = FakeImapBase(*args, **kwargs)
        ssl_instances.append(instance)
        return instance

    def plain_factory(*args, **kwargs):
        instance = FakeImapBase(*args, **kwargs)
        plain_instances.append(instance)
        return instance

    monkeypatch.setattr(email_mod.imaplib, "IMAP4_SSL", ssl_factory)
    monkeypatch.setattr(email_mod.imaplib, "IMAP4", plain_factory)
    assert email_mod._ImaplibTransport().fetch_unseen() == []
    assert ssl_instances[0].port == 993
    assert isinstance(ssl_instances[0].ssl_context, ssl.SSLContext)
    assert ssl_instances[0].ssl_context.verify_mode == ssl.CERT_REQUIRED
    assert ssl_instances[0].starttls_calls == []
    assert ssl_instances[0].login_calls == [(IMAP_ENV["IMAP_USER"], IMAP_ENV["IMAP_PASSWORD"])]
    assert plain_instances == []

    monkeypatch.setenv("IMAP_PORT", "143")
    assert email_mod._ImaplibTransport().fetch_unseen() == []
    assert len(plain_instances) == 1 and len(ssl_instances) == 1
    assert isinstance(plain_instances[0].starttls_calls[0], ssl.SSLContext)
    assert plain_instances[0].starttls_calls[0].verify_mode == ssl.CERT_REQUIRED
    assert plain_instances[0].login_calls == [(IMAP_ENV["IMAP_USER"], IMAP_ENV["IMAP_PASSWORD"])]


async def test_default_imap_does_not_login_when_starttls_fails(monkeypatch):
    import leo.integrations.channels.email as email_mod

    _set_env(monkeypatch, IMAP_ENV)
    monkeypatch.setenv("IMAP_PORT", "143")
    instances: list[FakeImapBase] = []

    import imaplib as real_imaplib

    error_type = real_imaplib.IMAP4.error

    class NoStartTls(FakeImapBase):
        def starttls(self, ssl_context=None):
            raise error_type("STARTTLS unavailable")

    def factory(*args, **kwargs):
        instance = NoStartTls(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(email_mod.imaplib, "IMAP4", factory)
    adapter = EmailAdapter(imap_transport=email_mod._ImaplibTransport())
    with pytest.raises(MessagingError, match="email poll failed: error"):
        await adapter.poll()
    assert instances[0].login_calls == []  # no credentials sent before TLS


# --------------------------------------------------------- sender authentication


async def test_poll_sender_identity_requires_aligned_spf_or_dkim(monkeypatch):
    cases = [
        ({}, "unverified:alice@example.com"),
        ({"authentication_results": "mx.test; spf=fail smtp.mailfrom=example.com"}, "unverified:alice@example.com"),
        ({"authentication_results": "mx.test; dkim=pass header.d=evil.test"}, "unverified:alice@example.com"),
        ({"authentication_results": "mx.test; spf=pass smtp.mailfrom=example.com"}, "alice@example.com"),
        ({"authentication_results": "mx.test; dkim=pass header.from=example.com"}, "alice@example.com"),
        ({"received_spf": "pass (authorized) client-ip=192.0.2.1; envelope-from=alice@example.com"}, "alice@example.com"),
    ]
    for index, (extra, expected) in enumerate(cases):
        item = {
            "uid": str(900 + index),
            "from": "Alice <alice@example.com>",
            "subject": "s",
            "body": "b",
            "date": "",
            **extra,
        }
        message = (await EmailAdapter(imap_transport=FakeImapTransport([item])).poll())[0]
        assert message.sender == expected


async def test_parse_message_extracts_authentication_headers():
    import leo.integrations.channels.email as email_mod

    raw = (
        b"From: Alice <alice@example.com>\r\n"
        b"Subject: s\r\n"
        b"Authentication-Results: mx.test; spf=pass smtp.mailfrom=example.com\r\n"
        b"Received-SPF: pass (ok); envelope-from=alice@example.com\r\n"
        b"\r\nbody"
    )
    record = email_mod._parse_message(b"7", raw)
    assert "spf=pass" in record["authentication_results"]
    assert "envelope-from=alice@example.com" in record["received_spf"]
