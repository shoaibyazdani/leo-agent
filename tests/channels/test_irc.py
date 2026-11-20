"""Offline verification for the IRC channel adapter.

No socket is ever opened: the adapter is driven through a fake transport that
records every call. Credentials are fake values that must never surface in
errors or records.
"""

from __future__ import annotations

import json

import pytest

from leo.integrations.channels.irc import IRCAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage

ENV_NAMES = (
    "IRC_HOST", "IRC_PORT", "IRC_NICKNAME", "IRC_PASSWORD", "IRC_CHANNELS",
    "IRC_TLS", "IRC_ALLOW_INSECURE",
)
FAKE_PASSWORD = "fake-irc-password-never-leak"


class FakeIrcTransport:
    """Records transport calls; ``read_lines`` drains like the default one."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = list(lines or [])
        self.connect_calls: list[tuple[str, str, str, str | None, bool]] = []
        self.joined: list[list[str]] = []
        self.privmsgs: list[tuple[str, str]] = []

    async def connect(self, host, port, nickname, password=None, *, use_tls=True):
        self.connect_calls.append((str(host), str(port), str(nickname), password, use_tls))

    async def join(self, channels):
        self.joined.append(list(channels))

    async def privmsg(self, target, text):
        self.privmsgs.append((str(target), str(text)))

    async def read_lines(self):
        drained, self.lines = self.lines, []
        return drained


class ExplodingConnectTransport(FakeIrcTransport):
    async def connect(self, host, port, nickname, password=None, *, use_tls=True):
        raise ConnectionRefusedError(f"dial irc with {FAKE_PASSWORD} failed")


class BrokenPrivmsgTransport(FakeIrcTransport):
    async def privmsg(self, target, text):
        raise OSError("write boom after login")


@pytest.fixture(autouse=True)
def clean_irc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _configured(monkeypatch: pytest.MonkeyPatch, *, channels: str = "#leo, #ops", port: str = "6697") -> None:
    monkeypatch.setenv("IRC_HOST", "irc.example.org")
    monkeypatch.setenv("IRC_PORT", port)
    monkeypatch.setenv("IRC_NICKNAME", "leo-bot")
    if channels is not None:
        monkeypatch.setenv("IRC_CHANNELS", channels)


# ------------------------------------------------------------------------ send


async def test_irc_send_connects_lazily_joins_and_privmsgs(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("IRC_PASSWORD", FAKE_PASSWORD)
    fake = FakeIrcTransport()
    adapter = IRCAdapter(transport=fake)
    result = await adapter.send("#leo", "hello channel")
    assert result == {"channel": "irc", "target": "#leo", "success": True, "message_id": None}
    assert fake.connect_calls == [("irc.example.org", "6697", "leo-bot", FAKE_PASSWORD, True)]
    assert fake.joined == [["#leo", "#ops"]]
    assert fake.privmsgs == [("#leo", "hello channel")]
    # The session is reused: no second registration or join.
    await adapter.send("alice", "direct hello")
    assert len(fake.connect_calls) == 1 and len(fake.joined) == 1
    assert fake.privmsgs[-1] == ("alice", "direct hello")


async def test_irc_send_without_channels_env_skips_join(monkeypatch):
    _configured(monkeypatch, channels=None)
    fake = FakeIrcTransport()
    await IRCAdapter(transport=fake).send("#leo", "hi")
    assert len(fake.connect_calls) == 1
    assert fake.joined == []


async def test_irc_send_blocked_without_envs(monkeypatch):
    adapter = IRCAdapter(transport=FakeIrcTransport())
    with pytest.raises(MessagingError, match="irc send blocked: IRC_HOST is not set"):
        await adapter.send("#leo", "hi")
    monkeypatch.setenv("IRC_HOST", "irc.example.org")
    with pytest.raises(MessagingError, match="irc send blocked: IRC_NICKNAME is not set"):
        await adapter.send("#leo", "hi")
    # IRC_PORT is optional: the TLS default is 6697.
    monkeypatch.setenv("IRC_NICKNAME", "leo-bot")
    fake = FakeIrcTransport()
    await IRCAdapter(transport=fake).send("#leo", "hi")
    assert fake.connect_calls == [("irc.example.org", "6697", "leo-bot", None, True)]


async def test_irc_send_rejects_invalid_targets(monkeypatch):
    _configured(monkeypatch)
    adapter = IRCAdapter(transport=FakeIrcTransport())
    for bad in ("nick with space", "bob!user@host", "nick,list", ""):
        with pytest.raises(MessagingError, match="invalid IRC target"):
            await adapter.send(bad, "hi")


async def test_irc_transport_failure_reports_type_name_only(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("IRC_PASSWORD", FAKE_PASSWORD)

    adapter = IRCAdapter(transport=ExplodingConnectTransport())
    with pytest.raises(MessagingError, match="irc send failed: ConnectionRefusedError") as exc:
        await adapter.send("#leo", "hi")
    assert FAKE_PASSWORD not in str(exc.value)
    assert "dial" not in str(exc.value)

    # A broken write invalidates the session: the next send reconnects once
    # (the fake keeps failing at privmsg, but a fresh connect is attempted).
    fake = BrokenPrivmsgTransport()
    adapter = IRCAdapter(transport=fake)
    with pytest.raises(MessagingError, match="irc send failed: OSError") as exc:
        await adapter.send("#leo", "hi")
    assert "write boom" not in str(exc.value)
    assert len(fake.connect_calls) == 1
    with pytest.raises(MessagingError, match="irc send failed: OSError"):
        await adapter.send("#leo", "retry")
    assert len(fake.connect_calls) == 2  # fresh lazy connect after the failure


# ------------------------------------------------------------------------ poll


async def test_irc_poll_parses_privmsg_and_skips_self_and_noise(monkeypatch):
    monkeypatch.setenv("IRC_NICKNAME", "leo-bot")
    fake = FakeIrcTransport(
        lines=[
            "PING :irc.example.org",  # keepalive, not a message
            ":bob!user@example.org PRIVMSG #leo :hello irc",
            ":LEO-BOT!bot@example.org PRIVMSG #leo :self echo is case-insensitive",
            "NOTICE * :server chatter",
            ":eve!u@example.org PRIVMSG leo-bot :direct hello",
            ":garbage",
        ]
    )
    adapter = IRCAdapter(transport=fake)
    messages = await adapter.poll()
    assert [(m.sender, m.text) for m in messages] == [("bob", "hello irc"), ("eve", "direct hello")]
    assert all(isinstance(m, NormalizedMessage) for m in messages)
    assert all(m.channel == "irc" for m in messages)
    assert all(m.message_id is None for m in messages)  # IRC has no message ids
    assert all(m.timestamp.endswith("+00:00") for m in messages)
    # read_lines drains the buffer: the drain itself is the cursor.
    assert await adapter.poll() == []


async def test_irc_poll_without_transport_returns_empty(monkeypatch):
    assert await IRCAdapter().poll() == []


# ---------------------------------------------------------------------- record


async def test_irc_record_blocked_configured_and_capabilities(monkeypatch):
    adapter = IRCAdapter()
    record = adapter.record()
    assert record.status == "blocked"
    assert record.capabilities == ["send", "poll"]
    assert record.metadata["credential_env"] == ["IRC_HOST", "IRC_NICKNAME"]
    assert record.metadata["reason"] == "missing credentials: IRC_HOST, IRC_NICKNAME"

    _configured(monkeypatch)
    monkeypatch.setenv("IRC_PASSWORD", FAKE_PASSWORD)
    record = IRCAdapter().record()
    assert record.status == "configured"
    # Optional envs are not part of the required credential set.
    assert record.metadata["credentials_present"] == ["IRC_HOST", "IRC_NICKNAME"]
    assert record.metadata["live_verified"] is False
    assert FAKE_PASSWORD not in json.dumps(record.model_dump())


async def test_irc_refuses_plaintext_credentials_unless_explicitly_allowed(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("IRC_PASSWORD", FAKE_PASSWORD)

    # Default: TLS is used, so there is nothing plaintext to block.
    default_fake = FakeIrcTransport()
    await IRCAdapter(transport=default_fake).send("#leo", "hi")
    assert default_fake.connect_calls[0][4] is True

    # Opting out of TLS alone is not enough to send credentials in cleartext.
    monkeypatch.setenv("IRC_TLS", "0")
    fake = FakeIrcTransport()
    with pytest.raises(MessagingError, match="IRC_ALLOW_INSECURE=1"):
        await IRCAdapter(transport=fake).send("#leo", "hi")
    assert fake.connect_calls == []  # still no registration on a cleartext socket

    # Both explicit flags are required.
    monkeypatch.setenv("IRC_ALLOW_INSECURE", "1")
    allowed_fake = FakeIrcTransport()
    await IRCAdapter(transport=allowed_fake).send("#leo", "hi")
    assert allowed_fake.connect_calls == [("irc.example.org", "6697", "leo-bot", FAKE_PASSWORD, False)]


async def test_irc_uses_tls_on_nonstandard_port_by_default(monkeypatch):
    _configured(monkeypatch, port="6667")
    fake = FakeIrcTransport()
    await IRCAdapter(transport=fake).send("#leo", "hi")
    assert fake.connect_calls == [("irc.example.org", "6667", "leo-bot", None, True)]


async def test_default_irc_transport_verifies_tls_before_registration(monkeypatch):
    import ssl as ssl_module

    import leo.integrations.channels.irc as irc_mod

    calls: list[tuple] = []
    writes: list[bytes] = []

    class FakeWriter:
        def is_closing(self):
            return False

        def write(self, data):
            writes.append(data)

        async def drain(self):
            pass

        def close(self):
            pass

        async def wait_closed(self):
            pass

    async def fake_open_connection(host, port, *, ssl=None):
        calls.append((host, port, ssl))
        return object(), FakeWriter()

    monkeypatch.setattr(irc_mod.asyncio, "open_connection", fake_open_connection)
    transport = irc_mod.AsyncioIrcTransport()
    await transport.connect("irc.example.org", 6697, "leo-bot", "secret-password")
    assert calls[0][:2] == ("irc.example.org", 6697)
    context = calls[0][2]
    assert isinstance(context, ssl_module.SSLContext)
    assert context.verify_mode == ssl_module.CERT_REQUIRED
    assert context.check_hostname is True
    # The handshake completed before open_connection returned; no PASS/NICK/USER
    # was written to a cleartext stream.
    assert writes == [b"PASS secret-password\r\n", b"NICK leo-bot\r\n", b"USER leo-bot 0 * :LEO\r\n"]
