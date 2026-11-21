"""IRC adapter: a minimal RFC-1459-compatible client with injectable transport.

Honest status: this is a **simple RFC-1459-compatible IRC client**, not a full
IRC library — no SASL, no reconnect, no capability negotiation and no
nick-collision handling. Authentication is the classic server ``PASS`` command
only (``IRC_PASSWORD``, optional). Inbound polling drains lines the server has
already buffered, so no cursor is required; keepalives are answered whenever
``poll()`` runs, which means a session that is never polled may eventually be
disconnected by the server. A background reader loop was deliberately avoided.

TLS is the default: when ``IRC_PORT`` is unset the client connects to 6697,
and ``ssl.create_default_context()`` is used for the handshake (certificate
and hostname verified) *before* ``PASS``/``NICK``/``USER`` are written.
Cleartext is possible only when ``IRC_TLS=0`` **and** ``IRC_ALLOW_INSECURE=1``
are both explicitly configured; otherwise the adapter blocks before opening
a connection. No credential value is ever stored on the adapter or echoed in
errors. The transport is injectable so tests never open a socket.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import ssl
from typing import Any, Protocol

import httpx

from ...contracts import utcnow
from ..messaging import Ed25519Verifier, MessagingAdapter, MessagingError, NormalizedMessage
from ..secrets import credential

# ":nick!user@host PRIVMSG <target> :text" — the only inbound line shape this
# minimal client normalizes; everything else (PING, JOIN, notices, numeric
# replies, malformed lines) is ignored.
_PRIVMSG_LINE = re.compile(
    r"^:(?P<prefix>\S+)\s+PRIVMSG\s+(?P<target>\S+)\s+:(?P<text>.*)$", re.IGNORECASE
)

#: IRC-over-TLS port, used when IRC_PORT is not configured.
DEFAULT_IRC_PORT = "6697"
_FALSEY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = (credential(name) or "").strip().lower()
    if raw in _FALSEY:
        return False
    if raw in _TRUTHY:
        return True
    return default


class IrcTransport(Protocol):
    """Transport contract the adapter speaks; inject a fake in tests.

    Implementations are responsible for their own keepalives: the default
    transport answers server ``PING`` lines inside ``read_lines()``. An
    optional ``close()`` (async) is used by ``IRCAdapter.aclose()`` when the
    adapter owns the transport.
    """

    async def connect(
        self,
        host: str,
        port: str | int,
        nickname: str,
        password: str | None = None,
        *,
        use_tls: bool = True,
    ) -> None:
        """Open the session: TLS handshake (when enabled), then registration."""
        ...

    async def join(self, channels: list[str]) -> None:
        """Join the given channels."""
        ...

    async def privmsg(self, target: str, text: str) -> None:
        """Deliver one message to a channel or nick."""
        ...

    async def read_lines(self) -> list[str]:
        """Drain pending server lines (answering PINGs along the way)."""
        ...


class AsyncioIrcTransport:
    """Default transport over asyncio TCP streams (``asyncio.open_connection``).

    Deliberately minimal: registration only (PASS when a password is given,
    NICK, USER), no SASL, no reconnect, no capabilities. TLS is established by
    ``asyncio.open_connection(..., ssl=context)`` with a verified default
    context before any registration bytes are written. CR/LF inside a message
    body is collapsed to spaces so untrusted text can never smuggle a second
    IRC command onto the wire.
    """

    _MAX_LINES_PER_POLL = 200

    def __init__(self) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def _send(self, line: str) -> None:
        if self._writer is None or self._writer.is_closing():
            raise RuntimeError("IRC transport is not connected")
        self._writer.write((line + "\r\n").encode("utf-8"))
        await self._writer.drain()

    async def connect(
        self,
        host: str,
        port: str | int,
        nickname: str,
        password: str | None = None,
        *,
        use_tls: bool = True,
    ) -> None:
        context = ssl.create_default_context() if use_tls else None
        # open_connection performs the TLS handshake (certificate + hostname
        # verification) before returning, so registration is never sent in
        # cleartext when TLS is enabled.
        reader, writer = await asyncio.open_connection(str(host), int(port), ssl=context)
        self._reader, self._writer = reader, writer
        if password:
            await self._send(f"PASS {password}")
        await self._send(f"NICK {nickname}")
        await self._send(f"USER {nickname} 0 * :LEO")

    async def join(self, channels: list[str]) -> None:
        for channel in channels:
            name = str(channel).replace("\r", "").replace("\n", "").strip()
            if name:
                await self._send(f"JOIN {name}")

    async def privmsg(self, target: str, text: str) -> None:
        safe_target = str(target).replace("\r", "").replace("\n", "").strip()
        safe_text = str(text).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        await self._send(f"PRIVMSG {safe_target} :{safe_text}")

    async def read_lines(self) -> list[str]:
        if self._reader is None or self._writer is None:
            return []
        lines: list[str] = []
        while len(lines) < self._MAX_LINES_PER_POLL:
            try:
                # Bounded read: drain what has arrived, do not block forever.
                raw = await asyncio.wait_for(self._reader.readline(), timeout=0.05)
            except TimeoutError:  # asyncio.TimeoutError is this since 3.11
                break
            except OSError:
                break  # connection lost; reconnecting is out of scope
            if not raw:
                break  # EOF: the server closed the session
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if parts and parts[0].upper() == "PING":
                token = parts[1] if len(parts) > 1 else ""
                await self._send("PONG " + token if token else "PONG")
                continue
            lines.append(line)
        return lines

    async def close(self) -> None:
        writer, self._writer, self._reader = self._writer, None, None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


class IRCAdapter(MessagingAdapter):
    """IRC sender plus line-poll inbound over an injectable transport.

    Sending connects lazily (one attempt per call) and joins the channels
    named by ``IRC_CHANNELS`` right after registration. Polling drains lines
    the transport has buffered — the drain itself is the cursor, so a second
    poll never re-reads a line. Messages whose nick is our own nickname are
    self-echo (the server mirrors our sends back) and are skipped.
    """

    channel = "irc"
    # IRC_PORT is optional and defaults to the TLS port 6697.
    credential_env = ("IRC_HOST", "IRC_NICKNAME")
    optional_env = ("IRC_PORT", "IRC_PASSWORD", "IRC_CHANNELS", "IRC_TLS", "IRC_ALLOW_INSECURE")
    verification_env = ()
    supports_webhook = False
    supports_polling = True

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        require_verification: bool = True,
        signature_verifier: Ed25519Verifier | None = None,
        transport: IrcTransport | None = None,
    ) -> None:
        super().__init__(
            client=client,
            timeout=timeout,
            require_verification=require_verification,
            signature_verifier=signature_verifier,
        )
        self.transport: IrcTransport | None = transport
        self._owns_transport = transport is None
        self._connected = False
        self._nickname: str | None = None

    async def aclose(self) -> None:
        await super().aclose()
        if self._owns_transport and self.transport is not None:
            closer = getattr(self.transport, "close", None)
            if callable(closer):
                result = closer()
                if inspect.isawaitable(result):
                    await result
            self.transport = None
            self._connected = False

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _valid_target(target: str) -> bool:
        """Channels start with '#'; nicks are single words without host sigils."""
        if not target or any(ch.isspace() for ch in target):
            return False
        if target.startswith("#"):
            return True
        return not any(ch in target for ch in "!@,:*")

    @staticmethod
    def _channels_from_env() -> list[str]:
        channels: list[str] = []
        for item in (credential("IRC_CHANNELS") or "").split(","):
            name = item.strip()
            if name and not any(ch.isspace() for ch in name):
                channels.append(name)
        return channels

    async def _connect_once(self, host: str, port: str, nickname: str) -> IrcTransport:
        if self._connected and self.transport is not None:
            return self.transport
        use_tls = _env_flag("IRC_TLS", True)
        if not use_tls and not _env_flag("IRC_ALLOW_INSECURE", False):
            raise MessagingError(
                "irc send blocked: plaintext connections require IRC_TLS=0 and IRC_ALLOW_INSECURE=1"
            )
        transport: IrcTransport = self.transport or AsyncioIrcTransport()
        try:
            await transport.connect(
                host, port, nickname, credential("IRC_PASSWORD") or None, use_tls=use_tls
            )
            channels = self._channels_from_env()
            if channels:
                await transport.join(channels)
        except Exception as exc:
            # Never echo the transport's message: it can contain the host or
            # password. A half-open default transport is simply dropped.
            raise MessagingError(f"irc send failed: {type(exc).__name__}") from None
        self.transport = transport
        self._connected = True
        self._nickname = nickname
        return transport

    # --------------------------------------------------------------------- send

    async def send(self, target: str, text: str) -> dict[str, Any]:
        host = credential("IRC_HOST")
        port = credential("IRC_PORT")
        nickname = credential("IRC_NICKNAME")
        if not host:
            raise MessagingError("irc send blocked: IRC_HOST is not set")
        if not nickname:
            raise MessagingError("irc send blocked: IRC_NICKNAME is not set")
        port = port or DEFAULT_IRC_PORT
        target_str = str(target)
        if not self._valid_target(target_str):
            raise MessagingError("irc send blocked: invalid IRC target")
        transport = await self._connect_once(host, port, nickname)
        try:
            await transport.privmsg(target_str, str(text))
        except Exception as exc:
            # The session state is unknown after a write failure: reconnect
            # on the next call rather than trust the broken transport.
            self._connected = False
            raise MessagingError(f"irc send failed: {type(exc).__name__}") from None
        return {
            "channel": self.channel,
            "target": target_str,
            "success": True,
            "message_id": None,  # IRC carries no message ids
        }

    # --------------------------------------------------------------------- poll

    async def poll(self) -> list[NormalizedMessage]:
        if self.transport is None:
            # No session yet: send() connects lazily; a poll-only deployment
            # should inject an already-connected transport.
            return []
        try:
            lines = await self.transport.read_lines()
        except Exception as exc:
            raise MessagingError(f"irc poll failed: {type(exc).__name__}") from None
        if not isinstance(lines, list):
            return []
        own_nick = (self._nickname or credential("IRC_NICKNAME") or "").lower()
        messages: list[NormalizedMessage] = []
        for line in lines:
            if not isinstance(line, str):
                continue
            match = _PRIVMSG_LINE.match(line.strip())
            if match is None:
                continue  # PING keepalives, joins, notices, malformed input
            sender = match.group("prefix").split("!", 1)[0]
            if own_nick and sender.lower() == own_nick:
                continue  # self-echo of our own sends
            text = match.group("text")
            if not text:
                continue
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    sender=sender,
                    text=text,
                    timestamp=utcnow(),
                    reply_ref=None,
                    message_id=None,
                )
            )
        return messages


__all__ = ["AsyncioIrcTransport", "IRCAdapter", "IrcTransport"]
