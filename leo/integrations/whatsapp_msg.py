"""Run the linked WhatsApp session as a LEO chat surface.

This is the message loop behind ``leo whatsapp serve``. It reuses the pinned,
hardened device-link bridge: inbound text arrives over the authenticated
loopback API, each admitted sender gets a durable LEO conversation, and the
agent's reply is sent back through the same linked session.

Safety posture:

* Fail closed. Without an explicit ``--allow`` number (or an explicit
  ``--allow-all``) nothing is answered at all.
* Only direct chats are answered by default; groups need ``--groups``.
* Messages from the linked account itself are filtered by the bridge.
* Message text is untrusted external content and is handled by the normal LEO
  turn (including its high-risk approval gates). This module never executes
  instructions found in a message.
* Outbound text is split to the bridge's 2000-character bound.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Settings
from ..core import LeoApp
from .whatsapp_bridge import BridgeConfig, BridgeError, BridgeTimeout, WhatsAppBridgeManager
from .whatsapp_link import CONNECTED, LinkUnavailable, bridge_status, ensure_ready

CHANNEL = "whatsapp_web"
MAX_REPLY_CHARS = 1900
POLL_TIMEOUT = 5.0
CONNECT_TIMEOUT = 60.0


@dataclass
class MessageServeResult:
    state: str
    replied: int = 0
    ignored: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.state == CONNECTED,
            "state": self.state,
            "replied": self.replied,
            "ignored": self.ignored,
            "failed": self.failed,
        }


def _digits(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _sender_key(jid: str) -> str:
    # Drop the device suffix (``123:7@s.whatsapp.net``) and any domain.
    local = str(jid or "").split("@", 1)[0].split(":", 1)[0]
    return local


def _is_pn(jid: Any) -> bool:
    return isinstance(jid, str) and jid.endswith("@s.whatsapp.net")


def _prefer_pn(primary: Any, alt: Any) -> str:
    """Prefer the phone-number JID over a privacy LID when both are supplied."""
    if _is_pn(primary):
        return str(primary)
    if _is_pn(alt):
        return str(alt)
    return str(primary or alt or "")


class _TypingKeepalive:
    """Re-publish WhatsApp "typing…" for one chat while a turn is running.

    The composing state expires after a few seconds on the recipient side, so a
    single call is not enough for a slow model turn. Presence is cosmetic: every
    failure here is swallowed so a broken presence call can never break a reply.
    """

    INTERVAL_SECONDS = 4.0

    def __init__(self, manager: Any, owner: str, auth: str, jid: str) -> None:
        self._manager = manager
        self._owner = owner
        self._auth = auth
        self._jid = jid
        self._task: asyncio.Task[None] | None = None

    async def _send(self, state: str) -> None:
        try:
            await self._manager.typing(
                owner_id=self._owner, to=self._jid, state=state, auth=self._auth,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.INTERVAL_SECONDS)
            await self._send("composing")

    async def start(self) -> None:
        await self._send("composing")
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await self._send("paused")


def missing_model_env(app: LeoApp) -> list[str]:
    """Environment names the default provider declares but the process lacks.

    A linked WhatsApp session does not imply a working model. The runner still
    starts (so the owner can fix the environment), but it says up front which
    names are unset instead of only failing on the first inbound message.
    """
    providers = getattr(getattr(app, "settings", None), "models", None)
    default = getattr(providers, "default", None)
    provider = (getattr(providers, "providers", None) or {}).get(default)
    if provider is None:
        return []
    names = [provider.api_key_env, *(provider.extra_header_env or {}).values()]
    return [name for name in dict.fromkeys(names) if name and not os.getenv(name)]


def split_text(text: str, limit: int = MAX_REPLY_CHARS) -> list[str]:
    """Split a reply into bridge-sized chunks, preferring paragraph breaks."""
    text = str(text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunk = text[:cut].strip()
        if chunk:
            chunks.append(chunk)
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


async def run_message_agent(
    owner: str = "owner",
    *,
    allow: list[str] | tuple[str, ...] = (),
    allow_all: bool = False,
    include_groups: bool = False,
    debug: bool = False,
    notify: Callable[[str, Any], None] | None = None,
) -> MessageServeResult:
    """Serve the linked session until cancelled (Ctrl+C) or the link ends."""
    ensure_ready()
    # A dedicated serve process: its shell tool must refuse commands that kill
    # this very listener before the turn can deliver an answer.
    os.environ["LEO_WHATSAPP_SERVE"] = "1"
    if not allow_all and not [_digits(item) for item in allow if _digits(item)]:
        raise LinkUnavailable(
            "Refusing to answer WhatsApp messages without an allowlist. "
            "Pass --allow +15551234567 (repeatable) or --allow-all explicitly."
        )
    if not bridge_status(owner)["linked"]:
        raise LinkUnavailable(
            "No linked WhatsApp session. Run `leo whatsapp link` first, then retry."
        )

    emit = notify or (lambda _kind, _data: None)
    allowed = {_digits(item) for item in allow if _digits(item)}
    app = LeoApp(Settings.load())
    manager = WhatsAppBridgeManager(BridgeConfig(
        owner_id=owner,
        actor="default",
        enabled=True,
        # wait() bounds its timeout by this ceiling; leave headroom above the
        # connect window so a slow first sync is not rejected outright.
        max_wait_seconds=90.0,
    ))
    auth = manager.session_auth
    sequence = 0
    result = MessageServeResult(state="idle")

    try:
        await app.initialize()
        missing = missing_model_env(app)
        if missing:
            emit("warning", {"missing": missing})
        await manager.start(owner_id=owner, auth=auth)
        async def wait_connected() -> dict[str, Any]:
            return await manager.wait_for_connection(
                owner_id=owner, timeout=CONNECT_TIMEOUT, auth=auth
            )

        try:
            status = await wait_connected()
        except BridgeTimeout:
            # A stalled socket is recoverable without destroying the link.
            repaired = await manager.repair(owner_id=owner, forget=False, auth=auth)
            try:
                status = repaired if str(repaired.get("state")) == CONNECTED else await wait_connected()
            except BridgeTimeout:
                status = await manager.status(owner_id=owner, auth=auth)
                raise BridgeError(
                    "WhatsApp did not connect in time "
                    f"(state={status.get('state')}, error={status.get('error')}). Try again.",
                    code="connect_timeout",
                ) from None
        state = str(status.get("state") or "unknown")
        if state != CONNECTED:
            result = MessageServeResult(state=state)
            emit("status", status)
            return result
        emit("status", status)
        result = MessageServeResult(state=CONNECTED)
        seen_diagnostics: set[str] = set()

        def dump_diagnostics() -> None:
            if not debug:
                return
            for line in manager.diagnostics():
                if line not in seen_diagnostics:
                    seen_diagnostics.add(line)
                    emit("diagnostic", line)

        dump_diagnostics()
        while True:
            batch = await manager.messages(owner_id=owner, after=sequence, timeout=POLL_TIMEOUT, auth=auth)
            sequence = max(sequence, int(batch.get("sequence") or 0))
            dump_diagnostics()
            for message in batch.get("messages") or []:
                if message.get("is_group") and not include_groups:
                    result.ignored += 1
                    continue
                # Direct chats may be addressed by LID; the bridge supplies the
                # phone-number alternative. Match the allowlist against every
                # identity and reply to the PN JID when one is present.
                identities = [
                    message.get("sender"), message.get("sender_alt"),
                    message.get("chat"), message.get("chat_alt"),
                ]
                digits = {
                    _digits(_sender_key(jid))
                    for jid in identities if isinstance(jid, str) and jid
                }
                if not allow_all and not (digits & allowed):
                    result.ignored += 1
                    emit("ignored", {"sender": _sender_key(str(message.get("sender") or ""))})
                    continue
                sender_jid = _prefer_pn(message.get("sender"), message.get("sender_alt"))
                sender = _sender_key(sender_jid or str(message.get("sender") or ""))
                reply_to = _prefer_pn(message.get("chat"), message.get("chat_alt"))
                if not _is_pn(reply_to) and not message.get("is_group"):
                    # A direct chat addressed only by LID: the participant's PN
                    # alternative is the same account, so reply there.
                    reply_to = sender_jid or reply_to
                reply_to = reply_to or str(message.get("chat") or "")
                if not reply_to:
                    result.ignored += 1
                    continue
                emit("message", {"sender": sender, "chars": len(str(message.get("text") or ""))})
                # WhatsApp expires a composing state after a few seconds, so a
                # background task repeats it for as long as the turn is running.
                presence = _TypingKeepalive(manager, owner, auth, reply_to)
                await presence.start()
                try:
                    answer = await app.ask(
                        str(message.get("text") or ""),
                        channel=CHANNEL,
                        subject=reply_to if message.get("is_group") else sender,
                        conversation_id=f"wa-web:{reply_to if message.get('is_group') else sender}",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # one bad turn must not stop the loop
                    result.failed += 1
                    emit("failed", {"sender": sender, "error": type(exc).__name__})
                    continue
                finally:
                    await presence.stop()
                reply = str(answer.get("content") or "").strip()
                if not reply:
                    result.ignored += 1
                    continue
                for chunk in split_text(reply):
                    await manager.send(owner_id=owner, to=reply_to, text=chunk, auth=auth)
                result.replied += 1
                emit("replied", {"sender": sender, "chars": len(reply)})
        # Unreachable; the loop runs until cancelled.
    finally:
        try:
            await manager.stop(owner_id=owner, auth=auth)
        except Exception:
            pass
        try:
            await app.shutdown()
        except Exception:
            pass


__all__ = [
    "CHANNEL",
    "MAX_REPLY_CHARS",
    "MessageServeResult",
    "missing_model_env",
    "run_message_agent",
    "split_text",
]
