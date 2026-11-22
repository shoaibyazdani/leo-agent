"""Messaging channel adapters.

Secrets are read from the environment by *name* at call time; no token value is
ever stored in an object, a record, a config file or an error message. Network
access goes through an injectable ``httpx.AsyncClient`` so adapters can be
verified with ``httpx.MockTransport`` without contacting a live platform.

A constructed adapter is not proof of a working integration: ``record()`` reports
"configured" only when the relevant environment variables are present and
"blocked" otherwise. ``verification`` is always "mock-tested" because live
delivery is deliberately never exercised by this module's tests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Type

import httpx
from pydantic import Field

from ..contracts import BaseRecord, IntegrationRecord, utcnow
from .secrets import credential
from .transport_state import TransportState

_LOG = logging.getLogger(__name__)

# Telegram rejects a message body over 4096 characters outright. Leave room for the
# part marker so a split never lands exactly on the limit.
TELEGRAM_MAX_CHARS = 4000
MAX_MESSAGE_CHUNKS = 20
# Legacy Markdown: **bold**, _italic_, `code` and ``` blocks. Agent replies are
# written in markdown, so plain text showed the user their own markup as noise.
TELEGRAM_PARSE_MODE = "Markdown"

# Outbound media: only files under this directory may leave the box. A caller
# supplied absolute path would otherwise let a prompt-injected agent attach
# ~/.ssh/id_rsa or /etc/shadow to a chat.
OUTBOUND_MEDIA_DIRNAME = ".leo/media/outbound"
# Telegram's Bot API caps a bot upload at 50 MB for both sendPhoto and
# sendDocument. Refuse locally: a 413 from the API wastes the whole upload.
TELEGRAM_MAX_MEDIA_BYTES = 50 * 1024 * 1024
# Extensions Telegram renders inline; everything else goes as a document so the
# file arrives intact instead of being re-encoded as an image.
_MEDIA_PHOTO_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
})


def _is_entity_error(response: Any) -> bool:
    """True when Telegram rejected the text because its entities do not parse."""
    try:
        body = response.json()
    except ValueError:
        return False
    description = str((body or {}).get("description") or "").lower()
    return "parse" in description or "entit" in description


def split_message(text: str, limit: int = TELEGRAM_MAX_CHARS,
                  max_chunks: int = MAX_MESSAGE_CHUNKS) -> list[str]:
    """Split text into transport-sized parts, preferring natural boundaries.

    A single 400 is worth nothing to a user who asked a question: the answer either
    arrives in readable parts or it does not arrive at all. Paragraphs first, then
    lines, then words, and only as a last resort a hard cut.
    """
    text = text or ""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining and len(parts) < max_chunks:
        if len(remaining) <= limit:
            parts.append(remaining)
            remaining = ""      # consumed: a leftover here would add a false "truncated"
            break
        window = remaining[:limit]
        # Never split inside a fenced block: close the open fence at the cut and
        # reopen it in the next part, or the remainder renders as broken markup.
        fences = window.count("```")
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        head = remaining[:cut].rstrip()
        tail = remaining[cut:].lstrip("\n")
        if fences % 2 == 1 and cut < len(remaining):
            head = head + "\n```"
            tail = "```\n" + tail
        parts.append(head)
        remaining = tail
    if remaining:
        parts.append(remaining[:limit].rstrip() + "\n\n… (truncated)")
    # A source that never closed its own fence would leave the final part inside
    # an open block, and every later marker in the chat renders literally.
    if parts and parts[-1].count("```") % 2 == 1:
        parts[-1] = parts[-1].rstrip() + "\n```"
    return parts


class MessagingError(RuntimeError):
    """Adapter could not send or process a message."""


def outbound_media_root(root: str | Path | None = None) -> Path:
    """Directory that is allowed to contain outbound attachments.

    In production the root comes from ``settings.paths.outbound_media`` and is
    resolved once at the composition root, so callers never choose it. The
    explicit argument exists for tests and for the adapter's own keyword.
    """
    base = Path(root) if root is not None else Path.cwd() / OUTBOUND_MEDIA_DIRNAME
    return Path(base)


def resolve_outbound_media(
    media: str | Path, root: str | Path | None = None
) -> Path:
    """Return the real path of an outbound attachment, or raise ``MessagingError``.

    Confinement is checked on the *resolved* parent, not on the raw string: a
    symlink inside the whitelist that points at ``/root/.ssh/id_rsa`` must not
    be a way out. The file itself is resolved too, so a symlinked leaf fails the
    same containment test.
    """
    if media is None or str(media).strip() == "":
        raise MessagingError("outbound media blocked: empty path")
    allowed = outbound_media_root(root)
    try:
        allowed_real = allowed.resolve(strict=False)
    except OSError as exc:  # unreadable parent chain
        raise MessagingError(f"outbound media blocked: {type(exc).__name__}") from None
    candidate = Path(str(media))
    if not candidate.is_absolute():
        # A bare name means "the attachment I just wrote", not "relative to
        # whatever directory the worker happens to run in". Resolve it inside the
        # whitelist first, and only fall back to the cwd-relative path when that
        # file exists and is inside the root.
        inside = allowed / candidate
        if inside.exists() or not candidate.exists():
            candidate = inside
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise MessagingError("outbound media blocked: file does not exist") from None
    except OSError as exc:
        raise MessagingError(f"outbound media blocked: {type(exc).__name__}") from None
    if resolved != allowed_real and allowed_real not in resolved.parents:
        # The offending path is deliberately not echoed back: an error string
        # ends up in logs and in a chat message.
        raise MessagingError(
            f"outbound media blocked: path is not under {OUTBOUND_MEDIA_DIRNAME}"
        )
    if not resolved.is_file():
        raise MessagingError("outbound media blocked: not a regular file")
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise MessagingError(f"outbound media blocked: {type(exc).__name__}") from None
    if size <= 0:
        raise MessagingError("outbound media blocked: file is empty")
    return resolved


class WebhookVerificationError(MessagingError):
    """Inbound webhook failed signature verification or is missing a signature."""


class NormalizedMessage(BaseRecord):
    """Channel-agnostic inbound message contract used by the gateway.

    ``sender`` is a *display* name (a username/nickname that the provider lets
    users change). Admission identity, session routing and reply targets must
    use the stable ids below instead:

    * ``sender_id`` — immutable account/user id supplied by the provider.
    * ``conversation_ref`` — stable chat/channel id (group context survives
      sender renames); ``thread`` is the finer-grained topic when present.
    """

    channel: str
    sender: str
    text: str
    timestamp: str = Field(default_factory=utcnow)
    reply_ref: str | None = None
    message_id: str | None = None
    sender_id: str | None = None
    conversation_ref: str | None = None
    thread: str | None = None
    # Provider-side descriptors only (a file id, never bytes). The gateway hands
    # them to its attachment sink, which downloads and stores them — a channel
    # adapter must not own the operator's media store.
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def identity(self) -> str:
        """Stable identity used by admission (never the mutable display name)."""
        return self.sender_id or self.sender

    @property
    def session_ref(self) -> str:
        """Conversation-scoped routing key: group/thread over sender."""
        return self.conversation_ref or self.thread or self.sender_id or self.sender

    @property
    def reply_target(self) -> str:
        """Where a reply must be sent (provider chat/channel/user id)."""
        return self.conversation_ref or self.thread or self.sender_id or self.sender


HeaderMap = Mapping[str, str] | Mapping[str, bytes] | None
Ed25519Verifier = Callable[[str, str, str, bytes], bool]


def _header(headers: HeaderMap, name: str) -> str | None:
    if not headers:
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    return None


def _body_bytes(payload: Any, raw_body: bytes | None = None) -> bytes:
    """Bytes used for HMAC verification.

    Pass ``raw_body`` for byte-exact verification when the framework has it.
    A dict is serialized compactly in insertion order; a str/bytes payload is
    used verbatim.
    """
    if raw_body is not None:
        return bytes(raw_body)
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _parsed_payload(payload: Any) -> Any:
    """Decode a payload once, turning malformed JSON into a typed error.

    A raw JSONDecodeError escaping an adapter used to surface as an opaque
    ``Exception`` in the API layer; callers of the adapter need a stable,
    intentional failure mode instead.
    """
    if isinstance(payload, (bytes, bytearray, str)):
        try:
            return json.loads(payload)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise MessagingError("webhook payload is not valid JSON") from exc
    return payload


def _payload_dict(payload: Any) -> dict[str, Any]:
    parsed = _parsed_payload(payload)
    if not isinstance(parsed, dict):
        raise MessagingError("webhook payload must be a JSON object")
    return parsed


def _payload_dict_or_list(payload: Any) -> dict[str, Any] | list[Any]:
    """Like :func:`_payload_dict`, but also accepts a batch of updates.

    Telegram proxies may deliver a JSON array of updates. The normalization
    code already handled a list, but the shared parser rejected it first.
    """
    parsed = _parsed_payload(payload)
    if not isinstance(parsed, (dict, list)):
        raise MessagingError("webhook payload must be a JSON object or array")
    return parsed


def _hmac_hex(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _secure_equal(left: str, right: str) -> bool:
    """Constant-time comparison that cannot raise on malformed input.

    ``hmac.compare_digest`` raises ``TypeError`` for non-ASCII ``str``
    operands. A hostile signature/token header must produce a clean rejection,
    never an unhandled exception.
    """
    try:
        return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
    except (AttributeError, TypeError, UnicodeError):
        return False


def _signature_matches(provided: str | None, expected: str) -> bool:
    if not provided:
        return False
    value = provided.strip()
    if "=" in value:  # e.g. "sha256=<hex>"
        value = value.split("=", 1)[1].strip()
    # Hex signatures are case-insensitive by convention; compare in constant
    # time on bytes so a non-ASCII header returns False instead of raising.
    return _secure_equal(value.lower(), expected.lower())


def _from_epoch(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return utcnow()


def _ref(value: Any) -> str | None:
    return None if value is None else str(value)


def _default_ed25519_verifier(public_key: str, signature: str, timestamp: str, body: bytes) -> bool:
    """Verify Discord's Ed25519 interaction signature using an optional library.

    Ed25519 is not HMAC; Discord interactions sign ``timestamp + raw_body`` with
    the application's public key. The optional dependencies are declared by the
    integrator, never assumed by LEO.
    """
    try:
        signature_bytes = bytes.fromhex(signature)
        key_bytes = bytes.fromhex(public_key)
    except ValueError:
        return False
    message = timestamp.encode("utf-8") + body
    try:
        from nacl.signing import VerifyKey  # type: ignore[import-not-found]

        try:
            VerifyKey(key_bytes).verify(message, signature_bytes)
            return True
        except Exception:
            return False
    except ImportError:
        pass
    try:
        from cryptography.exceptions import InvalidSignature  # type: ignore[import-not-found]
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore[import-not-found]
            Ed25519PublicKey,
        )

        try:
            Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature_bytes, message)
            return True
        except (InvalidSignature, ValueError):
            return False
    except ImportError as exc:
        raise WebhookVerificationError(
            "Ed25519 verification unavailable: install 'cryptography' or 'PyNaCl', "
            "or pass signature_verifier= to the Discord adapter"
        ) from exc
    return False


def implements_webhook(cls_or_adapter: type[MessagingAdapter] | MessagingAdapter) -> bool:
    """True only when the adapter actually overrides ``handle_webhook``.

    ``supports_webhook`` is a declaration; the base implementation raises
    ``NotImplementedError``. Advertising the capability from the flag alone
    would make unimplemented channels look reachable in the dashboard/API.
    """
    impl = getattr(cls_or_adapter, "handle_webhook", None)
    base = MessagingAdapter.handle_webhook
    return bool(
        getattr(cls_or_adapter, "supports_webhook", False)
        and impl is not None
        and getattr(impl, "__func__", impl) is not base
    )


def implements_polling(cls_or_adapter: type[MessagingAdapter] | MessagingAdapter) -> bool:
    """True only when the adapter actually overrides ``poll``."""
    impl = getattr(cls_or_adapter, "poll", None)
    base = MessagingAdapter.poll
    return bool(
        getattr(cls_or_adapter, "supports_polling", False)
        and impl is not None
        and getattr(impl, "__func__", impl) is not base
    )


def verification_health(adapter: MessagingAdapter) -> dict[str, Any]:
    """Honest verification-material report; never contains secret values.

    A credential proves sending capability; ``verification_env`` material is
    what makes *signed inbound* webhooks verifiable. Missing it degrades the
    channel even when the outbound credential is present.
    """
    missing = [name for name in adapter.verification_env if not credential(name)]
    if missing:
        status = "degraded"
        reason = f"missing verification material: {', '.join(missing)}"
    elif adapter.verification_env:
        status = "ready"
        reason = None
    else:
        status = "not-required"
        reason = None
    return {
        "channel": adapter.channel,
        "status": status,
        "missing_verification": missing,
        "reason": reason,
        "source": "verification_env",
    }


class MessagingAdapter:
    """Base adapter with credential discovery, verification and record-keeping."""

    channel: str = "unknown"
    credential_env: tuple[str, ...] = ()
    verification_env: tuple[str, ...] = ()
    # Capability declarations used by the registry, API routes and UI:
    # webhook channels accept signed POST ingress, polling channels have a
    # pull transport (IMAP, IRC, Matrix sync, ntfy, ...). Most channels are
    # webhook-capable; the base poll() simply raises.
    supports_webhook: bool = True
    supports_polling: bool = False

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        require_verification: bool = True,
        signature_verifier: Ed25519Verifier | None = None,
    ) -> None:
        self.client = client
        self._owns_client = client is None
        self.timeout = max(0.1, min(float(timeout), 30.0))
        self.require_verification = require_verification
        self.signature_verifier = signature_verifier

    def _http(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=False)
        return self.client

    async def aclose(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()
            self.client = None

    async def __aenter__(self) -> "MessagingAdapter":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    def _check_verification(self, payload: Any, headers: HeaderMap, raw_body: bytes | None) -> None:
        if self.require_verification and not self.verify_webhook(payload, headers, raw_body=raw_body):
            raise WebhookVerificationError(f"{self.channel} webhook signature verification failed")

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        raise NotImplementedError

    async def typing(self, target: str) -> None:
        """Best-effort 'is typing' indicator. Channels without one do nothing."""
        return None

    async def react(self, target: str, message_id: str, emoji: str = "\u2764") -> bool:
        """Add one emoji reaction to a message. Channels without one do nothing.

        Returns True only when the provider accepted the reaction. A channel that
        cannot react returns False rather than pretending it did.
        """
        return False

    async def send(
        self, target: str, text: str, *, reply_markup: dict[str, Any] | None = None,
        silent: bool = False,
    ) -> dict[str, Any]:
        """Send one text message.

        ``silent`` asks the channel not to notify the human — a live progress line
        is an update, not an interruption.

        ``reply_markup`` is an optional keyboard. A channel that cannot render one
        ignores it rather than failing the send: the text still carries the
        approval id, so a human can always resolve from the CLI.
        """
        raise NotImplementedError

    # Inline-keyboard taps arrive on the same transport as messages. The
    # composition root installs a sink; without one a tap is acknowledged and
    # dropped rather than left spinning on the user's screen.
    callback_sink: Any = None

    def set_callback_sink(self, sink: Any) -> None:
        self.callback_sink = sink

    async def send_media(
        self,
        target: str,
        media: str | Path,
        *,
        caption: str = "",
        root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Send one local file as an attachment.

        Default is "unsupported", not "silently send the caption only": a
        channel without attachment support must make the caller see that the
        file did not arrive.
        """
        raise MessagingError(f"{self.channel} does not support outbound media")

    async def typing(self, target: str) -> None:
        """sendChatAction(typing). Cosmetic: failures are swallowed by the caller."""
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token:
            return None
        url = f"{self.API_BASE}/bot{token}/sendChatAction"
        response = await self._http().post(
            url, json={"chat_id": str(target), "action": "typing"}, timeout=self.timeout
        )
        if response.status_code >= 400:
            raise MessagingError(f"telegram typing failed: HTTP {response.status_code}")
        return None

    async def react(self, target: str, message_id: str, emoji: str = "\U0001F44D") -> bool:
        """setMessageReaction on one message. Best effort, never raises.

        Telegram scopes message ids per chat *per bot*: an id that a different
        bot saw in the same chat does not exist for this one, and the API
        answers "message to react not found". That is a real failure to report,
        not a success.
        """
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token or not message_id:
            return False
        url = f"{self.API_BASE}/bot{token}/setMessageReaction"
        payload: dict[str, Any] = {
            "chat_id": str(target),
            "message_id": int(message_id),
            "reaction": [{"type": "emoji", "emoji": str(emoji)}],
        }
        try:
            response = await self._http().post(url, json=payload, timeout=self.timeout)
        except (httpx.HTTPError, ValueError):
            return False
        if response.status_code < 400:
            return True
        # The token appears in the URL: log only the status and description.
        description = ""
        try:
            description = str((response.json() or {}).get("description") or "")
        except ValueError:
            description = ""
        _LOG.warning("telegram reaction failed", extra={
            "channel": self.channel, "status": response.status_code,
            "description": description[:120]})
        return False

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        raise NotImplementedError

    async def poll(self, state: TransportState | None = None) -> list[NormalizedMessage]:
        """Pull inbound messages from channels without webhook delivery.

        Adapters with a poll transport override this and set
        ``supports_polling = True``. When a :class:`TransportState` is passed,
        the poll position is persisted, so a restart resumes instead of
        replaying or dropping what arrived while the process was down.
        """
        raise MessagingError(f"{self.channel} does not support polling")

    def record(self) -> IntegrationRecord:
        names = self.credential_env + self.verification_env
        configured = self._is_configured()
        health = verification_health(self)
        capabilities = ["send"]
        if implements_webhook(self):
            capabilities.append("webhook")
        if implements_polling(self):
            capabilities.append("poll")
        return IntegrationRecord(
            name=self.channel,
            kind="messaging",
            status="configured" if configured else "blocked",
            capabilities=capabilities,
            verification="mock-tested",
            config_ref=",".join(names) or None,
            metadata={
                "credential_env": list(self.credential_env),
                "verification_env": list(self.verification_env),
                "credentials_present": [n for n in self.credential_env if credential(n)],
                # Separate from the outbound credential status: a channel can
                # be configured to send yet degraded for signed ingress.
                "verification_status": health["status"],
                "missing_verification": health["missing_verification"],
                "live_verified": False,
                "reason": None if configured else f"missing credentials: {', '.join(self.credential_env)}",
            },
        )

    def _is_configured(self) -> bool:
        return all(credential(name) for name in self.credential_env)


class TelegramAdapter(MessagingAdapter):
    """Telegram Bot API sender plus webhook update normalization.

    Verification: Telegram sends a shared secret in
    ``X-Telegram-Bot-Api-Secret-Token`` (``TELEGRAM_WEBHOOK_SECRET``); a proxy
    supplied ``X-Hub-Signature-256`` HMAC is also accepted. Both use
    constant-time comparison.
    """

    async def send_media(
        self,
        target: str,
        media: str | Path,
        *,
        caption: str = "",
        root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Send one whitelisted local file via ``sendPhoto`` or ``sendDocument``.

        The path is confined to the outbound media root (resolved, so a symlink
        cannot escape) and refused above 50 MB *before* any bytes leave the box.
        Every failure raises ``MessagingError``: an attachment that did not
        arrive must never be reported as sent.
        """
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token:
            raise MessagingError("telegram media blocked: TELEGRAM_BOT_TOKEN is not set")
        path = resolve_outbound_media(media, root)
        size = path.stat().st_size
        if size > TELEGRAM_MAX_MEDIA_BYTES:
            raise MessagingError(
                f"telegram media blocked: {size} bytes exceeds the "
                f"{TELEGRAM_MAX_MEDIA_BYTES} byte limit"
            )
        method = "sendPhoto" if path.suffix.lower() in _MEDIA_PHOTO_SUFFIXES else "sendDocument"
        field_name = "photo" if method == "sendPhoto" else "document"
        # The token appears in the URL: never echo the URL or httpx's exception text.
        url = f"{self.API_BASE}/bot{token}/{method}"
        form: dict[str, Any] = {"chat_id": str(target)}
        if caption:
            # 1024 is Telegram's caption limit; a longer one is rejected outright.
            form["caption"] = str(caption)[:1024]
            form["parse_mode"] = TELEGRAM_PARSE_MODE
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise MessagingError(f"telegram media failed: {type(exc).__name__}") from None
        files = {field_name: (path.name, content)}
        try:
            response = await self._http().post(url, data=form, files=files, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise MessagingError(f"telegram media failed: {type(exc).__name__}") from None
        if response.status_code == 400 and "parse_mode" in form and _is_entity_error(response):
            # Same rule as text: an unclosed `**` in the caption must not lose the file.
            form.pop("parse_mode", None)
            try:
                response = await self._http().post(
                    url, data=form, files=files, timeout=self.timeout
                )
            except httpx.HTTPError as exc:
                raise MessagingError(f"telegram media failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"telegram media failed: HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise MessagingError("telegram media failed: invalid response") from exc
        if not isinstance(body, dict) or not body.get("ok"):
            raise MessagingError("telegram media failed: API reported failure")
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": _ref(result.get("message_id")),
            "media": path.name,
            "media_method": method,
            "media_bytes": size,
        }

    channel = "telegram"
    credential_env = ("TELEGRAM_BOT_TOKEN",)
    verification_env = ("TELEGRAM_WEBHOOK_SECRET",)
    API_BASE = "https://api.telegram.org"
    # Both transports are real: signed webhook ingress, or getUpdates polling
    # for a box with no public HTTPS endpoint (allow-listed, fails closed).
    supports_webhook = True
    supports_polling = True

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        secret = credential("TELEGRAM_WEBHOOK_SECRET")
        if not secret:
            # Fail closed: an unsigned webhook must never be accepted just
            # because verification was never configured.
            return not self.require_verification
        token = _header(headers, "x-telegram-bot-api-secret-token")
        if token and _secure_equal(token, secret):
            return True
        signature = _header(headers, "x-hub-signature-256") or _header(headers, "x-telegram-signature")
        return _signature_matches(signature, _hmac_hex(secret, _body_bytes(payload, raw_body)))

    async def send(
        self, target: str, text: str, *, reply_markup: dict[str, Any] | None = None,
        silent: bool = False,
    ) -> dict[str, Any]:
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token:
            raise MessagingError("telegram send blocked: TELEGRAM_BOT_TOKEN is not set")
        # Telegram caps one message at 4096 characters and rejects the whole send
        # with 400 otherwise. A long answer must arrive in parts, not vanish.
        chunks = split_message(str(text), TELEGRAM_MAX_CHARS)
        last: dict[str, Any] = {}
        for index, chunk in enumerate(chunks):
            # The keyboard belongs to the message the human reads last, not to a
            # continuation that may arrive before it.
            keyboard = reply_markup if index + 1 == len(chunks) else None
            last = await self._send_one(token, target, chunk, reply_markup=keyboard, silent=silent)
            if index + 1 < len(chunks):
                _LOG.info("telegram long message split", extra={
                    "chunks": len(chunks), "part": index + 1})
        return last

    async def answer_callback_query(self, callback_id: str, text: str = "") -> bool:
        """Stop the tap spinner and optionally show a one-line result."""
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token or not callback_id:
            return False
        url = f"{self.API_BASE}/bot{token}/answerCallbackQuery"
        payload: dict[str, Any] = {"callback_query_id": str(callback_id)}
        if text:
            payload["text"] = str(text)[:200]
        try:
            response = await self._http().post(url, json=payload, timeout=self.timeout)
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    async def edit_message_text(
        self, target: str, message_id: str, text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        """Replace the text of a sent message (best effort)."""
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token or not message_id:
            return False
        url = f"{self.API_BASE}/bot{token}/editMessageText"
        payload: dict[str, Any] = {
            "chat_id": str(target), "message_id": str(message_id), "text": str(text),
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            response = await self._http().post(url, json=payload, timeout=self.timeout)
        except httpx.HTTPError:
            return False
        if response.status_code < 400:
            return True
        # "message is not modified" means the text is already there. That is the
        # outcome the caller wanted, not a failure: reporting it as one made the
        # progress reporter open a NEW message every few seconds.
        description = ""
        try:
            description = str((response.json() or {}).get("description") or "")
        except ValueError:
            description = ""
        if "not modified" in description.lower():
            return True
        _LOG.debug("edit failed", extra={
            "channel": self.channel, "status": response.status_code,
            "description": description[:120]})
        return False

    async def _send_one(
        self, token: str, target: str, text: str,
        reply_markup: dict[str, Any] | None = None,
        silent: bool = False,
    ) -> dict[str, Any]:
        # The token appears in the URL: never echo the URL or httpx's exception text.
        url = f"{self.API_BASE}/bot{token}/sendMessage"
        payload: dict[str, Any] = {"chat_id": str(target), "text": text,
                                   "parse_mode": TELEGRAM_PARSE_MODE}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if silent:
            # A live progress line is an update, not an interruption.
            payload["disable_notification"] = True
        try:
            response = await self._http().post(url, json=payload, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise MessagingError(f"telegram send failed: {type(exc).__name__}") from None
        if response.status_code == 400 and _is_entity_error(response):
            # The text is markdown-ish but not valid Telegram entities (an unclosed
            # `**` or a stray underscore). Plain text keeps the content; a 400 loses
            # the whole message, and a user would rather read raw markers than nothing.
            payload.pop("parse_mode", None)
            try:
                response = await self._http().post(url, json=payload, timeout=self.timeout)
            except httpx.HTTPError as exc:
                raise MessagingError(f"telegram send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"telegram send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("telegram send failed: invalid response") from exc
        if not isinstance(data, dict) or not data.get("ok"):
            raise MessagingError("telegram send failed: API reported failure")
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": _ref(result.get("message_id")),
        }

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        data = _payload_dict_or_list(payload)
        updates = (
            data.get("updates")
            if isinstance(data, dict) and isinstance(data.get("updates"), list)
            else None
        )
        if updates is None:
            updates = data if isinstance(data, list) else [data]
        return self._messages_from_updates(updates)

    def _messages_from_updates(self, updates: list[Any]) -> list[NormalizedMessage]:
        """Normalize raw Telegram updates. Shared by webhook and poll ingress."""
        messages: list[NormalizedMessage] = []
        for update in updates:
            if not isinstance(update, dict):
                continue
            message = update.get("message") or update.get("edited_message")
            if not message and isinstance(update.get("callback_query"), dict):
                message = update["callback_query"].get("message")
            if not isinstance(message, dict):
                continue
            sender_data = message.get("from") if isinstance(message.get("from"), dict) else {}
            # A bot's own traffic is not a human turn. Telegram hands an agent back
            # its own messages in some topologies, and the approval notifier writes
            # a notice into the same chat: ingesting either makes the agent answer
            # itself, which reads to the operator as the agent "mixing things up".
            if sender_data.get("is_bot"):
                _LOG.info("inbound from a bot ignored", extra={
                    "channel": self.channel, "sender_id": _ref(sender_data.get("id"))})
                continue
            _body = str(message.get("text") or message.get("caption") or "").lstrip()
            if _body.startswith("🔐 Approval needed"):
                # The approval itself is already a durable row with buttons; its
                # notification text is not something the operator said.
                _LOG.info("approval notice echoed back ignored", extra={"channel": self.channel})
                continue
            reply = message.get("reply_to_message") if isinstance(message.get("reply_to_message"), dict) else {}
            chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
            # The numeric id is the admission identity and the chat id is the
            # session key; the username is only a display label.
            sender_id = _ref(sender_data.get("id")) or _ref(sender_data.get("username")) or "unknown"
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    sender=str(sender_data.get("username") or sender_id),
                    text=str(message.get("text") or message.get("caption") or ""),
                    timestamp=_from_epoch(message.get("date")),
                    reply_ref=_ref(reply.get("message_id")),
                    message_id=_ref(message.get("message_id")),
                    sender_id=sender_id,
                    conversation_ref=_ref(chat.get("id")),
                    thread=_ref(message.get("message_thread_id")),
                    attachments=self._attachments_from_message(message),
                )
            )
        return messages

    @staticmethod
    def _attachments_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Provider descriptors for a photo/document — else a photo is an empty message."""
        out: list[dict[str, Any]] = []
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            best = max(
                (p for p in photos if isinstance(p, dict)),
                key=lambda p: int(p.get("file_size") or p.get("width") or 0),
                default=None,
            )
            if best and best.get("file_id"):
                out.append({
                    "kind": "photo",
                    "file_id": str(best["file_id"]),
                    "mime": "image/jpeg",
                    "name": "photo.jpg",
                    "size": int(best.get("file_size") or 0),
                })
        document = message.get("document")
        if isinstance(document, dict) and document.get("file_id"):
            out.append({
                "kind": "document",
                "file_id": str(document["file_id"]),
                "mime": str(document.get("mime_type") or "application/octet-stream"),
                "name": str(document.get("file_name") or "document"),
                "size": int(document.get("file_size") or 0),
            })
        return out

    async def download(self, descriptor: dict[str, Any]) -> tuple[bytes, str, str]:
        """Fetch one descriptor's bytes through getFile. Returns (raw, mime, name)."""
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token:
            raise MessagingError("telegram download blocked: TELEGRAM_BOT_TOKEN is not set")
        file_id = str(descriptor.get("file_id") or "")
        if not file_id:
            raise MessagingError("telegram download blocked: descriptor has no file_id")
        meta_url = f"{self.API_BASE}/bot{token}/getFile"
        response = await self._http().post(meta_url, json={"file_id": file_id}, timeout=self.timeout)
        if response.status_code >= 400:
            raise MessagingError(f"telegram getFile failed: HTTP {response.status_code}")
        try:
            meta = response.json()
        except ValueError as exc:
            raise MessagingError("telegram getFile failed: invalid response") from exc
        path = str(((meta or {}).get("result") or {}).get("file_path") or "")
        if not path:
            raise MessagingError("telegram getFile failed: no file_path")
        blob = await self._http().get(
            f"{self.API_BASE}/file/bot{token}/{path}", timeout=max(self.timeout, 30.0)
        )
        if blob.status_code >= 400:
            raise MessagingError(f"telegram download failed: HTTP {blob.status_code}")
        return (
            blob.content,
            str(descriptor.get("mime") or "application/octet-stream"),
            str(descriptor.get("name") or "attachment"),
        )

    # ------------------------------------------------------------------- poll

    # No allow-list here on purpose: ChannelAdmission is the single policy
    # resolver for inbound senders (see admission.py), so the transport only
    # fetches and normalizes. Duplicating the check would give two sources of
    # truth for who may talk to LEO.
    _cursor: int | None = None

    async def poll(self, state: TransportState | None = None) -> list[NormalizedMessage]:
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token:
            raise MessagingError("telegram poll blocked: TELEGRAM_BOT_TOKEN is not set")
        stored = await state.cursor() if state is not None else None
        offset = stored if stored is not None else self._cursor
        params: dict[str, str] = {"timeout": "0"}
        if offset is not None:
            params["offset"] = str(offset)
        latest: int | None = None
        # The token appears in the URL: never echo the URL or httpx's exception text.
        url = f"{self.API_BASE}/bot{token}/getUpdates"
        try:
            response = await self._http().get(url, params=params, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise MessagingError(f"telegram poll failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"telegram poll failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            return []
        if not isinstance(data, dict) or not data.get("ok"):
            return []
        updates = data.get("result")
        if not isinstance(updates, list):
            return []
        for update in updates:
            update_id = update.get("update_id") if isinstance(update, dict) else None
            if isinstance(update_id, int):
                if offset is not None and update_id > offset and latest is None:
                    # A confirmed offset followed by a higher id means the
                    # platform dropped the updates between them. Say so rather
                    # than presenting a silent hole as a clean poll.
                    _LOG.warning(
                        "poll gap: %s update(s) missing between %s and %s",
                        update_id - offset,
                        offset,
                        update_id,
                        extra={"channel": self.channel, "kind": "transport_gap"},
                    )
                latest = update_id + 1 if latest is None else max(latest, update_id + 1)
                self._cursor = update_id + 1
        if state is not None and latest is not None:
            await state.set_cursor(latest)
        await self._dispatch_callbacks(updates)
        return self._messages_from_updates(updates)

    async def _dispatch_callbacks(self, updates: list[Any]) -> None:
        """Hand inline-keyboard taps to the composition root.

        A tap is not a message: it must never be normalised into a chat turn, or
        every button press would read as user text.
        """
        for update in updates or []:
            if not isinstance(update, dict):
                continue
            tap = update.get("callback_query")
            if not isinstance(tap, dict):
                continue
            message = tap.get("message") if isinstance(tap.get("message"), dict) else {}
            chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
            sender = tap.get("from") if isinstance(tap.get("from"), dict) else {}
            descriptor = {
                "callback_id": _ref(tap.get("id")),
                "data": str(tap.get("data") or ""),
                "sender_id": _ref(sender.get("id")),
                "chat_id": _ref(chat.get("id")),
                "message_id": _ref(message.get("message_id")),
            }
            sink = self.callback_sink
            if sink is None:
                # No sink: acknowledge so the tap does not spin forever, and say
                # plainly that nothing was decided.
                await self.answer_callback_query(
                    descriptor["callback_id"], "No handler for this button"
                )
                continue
            try:
                await sink(self.channel, descriptor)
            except Exception as exc:  # noqa: BLE001 - a tap must not break polling
                _LOG.warning("callback dispatch failed", extra={
                    "channel": self.channel, "error": type(exc).__name__})


# Discord signs ``timestamp + raw_body`` but does not bind the request to a
# window itself; without a freshness check a captured signed interaction can be
# replayed indefinitely inside the gateway's body-dedupe window (or after it).
DISCORD_INTERACTION_MAX_AGE_SECONDS = 300.0
_DISCORD_PING_TYPES = frozenset({1, "1"})


def is_discord_ping(payload: Any) -> bool:
    """Detect an interaction PING (types 1 and "1") without trusting content."""
    try:
        data = _parsed_payload(payload)
    except MessagingError:
        return False
    if not isinstance(data, dict):
        return False
    value = data.get("type")
    if isinstance(value, bool):
        return False
    return value in _DISCORD_PING_TYPES


def discord_timestamp_is_fresh(
    timestamp: Any,
    *,
    now: float | None = None,
    max_age: float = DISCORD_INTERACTION_MAX_AGE_SECONDS,
) -> bool:
    """Whether a Discord signature timestamp is within the acceptance window."""
    try:
        seconds = float(timestamp)
    except (TypeError, ValueError):
        return False
    if seconds <= 0:
        return False
    age = (time.time() if now is None else now) - seconds
    # Small negative age is clock skew, not a replay.
    return -30.0 <= age <= max(0.0, float(max_age))


class DiscordAdapter(MessagingAdapter):
    """Discord webhook/incoming-webhook sender and minimal bot interaction gateway.

    Inbound interaction requests are signed with Ed25519 (not HMAC) using
    ``DISCORD_PUBLIC_KEY``; an injectable ``signature_verifier`` is supported so
    this works without adding a crypto dependency. Plain incoming webhooks have
    no platform signature, so do not expose them to untrusted callers.
    """

    channel = "discord"
    credential_env = ("DISCORD_BOT_TOKEN", "DISCORD_WEBHOOK_URL")
    verification_env = ("DISCORD_PUBLIC_KEY",)
    WEBHOOK_HOSTS = {"discord.com", "discordapp.com", "canary.discord.com"}
    API_BASE = "https://discord.com/api/v10"
    interaction_max_age = DISCORD_INTERACTION_MAX_AGE_SECONDS

    def record(self) -> IntegrationRecord:
        record = super().record()
        health = verification_health(self)
        record.metadata["verification_reason"] = health["reason"]
        if record.status == "configured":
            return record
        if credential("DISCORD_BOT_TOKEN") or credential("DISCORD_WEBHOOK_URL"):
            # Sending works, but signed interaction ingress does not without the
            # public key. The dedicated verification report calls this degraded.
            record.status = "configured"
            record.metadata["reason"] = None
        return record

    def _is_configured(self) -> bool:
        return bool(credential("DISCORD_BOT_TOKEN") and credential("DISCORD_WEBHOOK_URL"))

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        public_key = credential("DISCORD_PUBLIC_KEY")
        if not public_key:
            return not self.require_verification
        signature = _header(headers, "x-signature-ed25519")
        timestamp = _header(headers, "x-signature-timestamp")
        if not signature or not timestamp:
            return False
        if self.signature_verifier is None and not discord_timestamp_is_fresh(
            timestamp, max_age=self.interaction_max_age
        ):
            raise WebhookVerificationError("discord interaction timestamp is outside the freshness window")
        verifier = self.signature_verifier or _default_ed25519_verifier
        body = _body_bytes(payload, raw_body)
        try:
            return bool(verifier(public_key, signature, timestamp, body))
        except WebhookVerificationError:
            raise
        except Exception as exc:
            raise WebhookVerificationError(f"discord signature verifier failed: {type(exc).__name__}") from exc

    @classmethod
    def _valid_webhook_url(cls, url: str) -> bool:
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        host = (parsed.hostname or "").lower()
        return (
            parsed.scheme == "https"
            and host in cls.WEBHOOK_HOSTS
            and parsed.path.startswith("/api/webhooks/")
            and not parsed.username
            and not parsed.password
        )

    async def send(self, target: str, text: str) -> dict[str, Any]:
        target_str = str(target)
        webhook_url = target_str if target_str.startswith("https://") else credential("DISCORD_WEBHOOK_URL")
        if webhook_url and not self._valid_webhook_url(webhook_url):
            raise MessagingError("discord send blocked: webhook URL is not a Discord HTTPS webhook URL")
        try:
            if webhook_url:
                payload: dict[str, Any] = {"content": str(text)}
                if not target_str.startswith("https://") and target_str:
                    payload["thread_id"] = target_str
                response = await self._http().post(webhook_url, json=payload, timeout=self.timeout)
            else:
                token = credential("DISCORD_BOT_TOKEN")
                if not token:
                    raise MessagingError(
                        "discord send blocked: set DISCORD_WEBHOOK_URL or DISCORD_BOT_TOKEN"
                    )
                response = await self._http().post(
                    f"{self.API_BASE}/channels/{target_str}/messages",
                    headers={"Authorization": f"Bot {token}"},
                    json={"content": str(text)},
                    timeout=self.timeout,
                )
        except MessagingError:
            raise
        except httpx.HTTPError as exc:
            raise MessagingError(f"discord send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"discord send failed: HTTP {response.status_code}")
        message_id: str | None = None
        if response.content:
            try:
                data = response.json()
                if isinstance(data, dict):
                    message_id = _ref(data.get("id"))
            except ValueError:
                message_id = None
        return {
            "channel": self.channel,
            "target": target_str,
            "success": True,
            "message_id": message_id,
        }

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        data = _payload_dict(payload)
        messages: list[NormalizedMessage] = []
        if is_discord_ping(data):  # interaction PING (int or numeric string)
            return messages
        bodies: list[dict[str, Any]] = []
        if data.get("type") in (2, 3):  # application command / component interaction
            interaction_data = data.get("data") if isinstance(data.get("data"), dict) else {}
            options = interaction_data.get("options")
            text = str(interaction_data.get("name") or "interaction")
            if isinstance(options, list) and options and isinstance(options[0], dict):
                text = str(options[0].get("value") or text)
            user = data.get("user")
            if not isinstance(user, dict):
                member = data.get("member") if isinstance(data.get("member"), dict) else {}
                user = member.get("user") if isinstance(member.get("user"), dict) else {}
            bodies.append(
                {
                    "author": user,
                    "content": text,
                    "timestamp": data.get("timestamp"),
                    "id": data.get("id"),
                    "channel_id": data.get("channel_id"),
                }
            )
        elif isinstance(data.get("d"), dict) and data.get("t") == "MESSAGE_CREATE":
            bodies.append(data["d"])
        elif "content" in data or "author" in data:
            bodies.append(data)
        for body in bodies:
            author = body.get("author") if isinstance(body.get("author"), dict) else {}
            if author.get("bot"):
                continue
            reference = body.get("message_reference") if isinstance(body.get("message_reference"), dict) else {}
            reply_ref = reference.get("message_id")
            if reply_ref is None and isinstance(body.get("referenced_message"), dict):
                reply_ref = body["referenced_message"].get("id")
            sender_id = _ref(author.get("id")) or _ref(author.get("username")) or "unknown"
            conversation_ref = _ref(body.get("channel_id")) or _ref(data.get("channel_id")) or sender_id
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    sender=str(author.get("username") or sender_id),
                    text=str(body.get("content") or ""),
                    timestamp=str(body.get("timestamp") or utcnow()),
                    reply_ref=_ref(reply_ref),
                    message_id=_ref(body.get("id")),
                    sender_id=sender_id,
                    conversation_ref=conversation_ref,
                    thread=_ref(body.get("thread_id")) or conversation_ref,
                )
            )
        return messages


class WhatsAppAdapter(MessagingAdapter):
    """WhatsApp Cloud API sender plus webhook normalization.

    Verification uses ``X-Hub-Signature-256`` (HMAC-SHA256 of the raw request
    body with ``WHATSAPP_APP_SECRET``) and constant-time comparison.
    """

    channel = "whatsapp"
    credential_env = ("WHATSAPP_CLOUD_TOKEN", "WHATSAPP_PHONE_NUMBER_ID")
    verification_env = ("WHATSAPP_APP_SECRET",)
    API_BASE = "https://graph.facebook.com"

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        secret = credential("WHATSAPP_APP_SECRET")
        if not secret:
            return not self.require_verification
        signature = _header(headers, "x-hub-signature-256")
        return _signature_matches(signature, _hmac_hex(secret, _body_bytes(payload, raw_body)))

    async def send(self, target: str, text: str) -> dict[str, Any]:
        token = credential("WHATSAPP_CLOUD_TOKEN")
        phone_id = credential("WHATSAPP_PHONE_NUMBER_ID")
        if not token or not phone_id:
            raise MessagingError(
                "whatsapp send blocked: WHATSAPP_CLOUD_TOKEN and WHATSAPP_PHONE_NUMBER_ID are required"
            )
        version = credential("WHATSAPP_API_VERSION", "v20.0")
        url = f"{self.API_BASE}/{version}/{phone_id}/messages"
        try:
            response = await self._http().post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "messaging_product": "whatsapp",
                    "to": str(target),
                    "type": "text",
                    "text": {"body": str(text)},
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"whatsapp send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"whatsapp send failed: HTTP {response.status_code}")
        message_id: str | None = None
        try:
            data = response.json()
            messages = data.get("messages") if isinstance(data, dict) else None
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                message_id = _ref(messages[0].get("id"))
        except ValueError:
            message_id = None
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": message_id,
        }

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        data = _payload_dict(payload)
        messages: list[NormalizedMessage] = []
        for entry in data.get("entry", []) if isinstance(data.get("entry"), list) else []:
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes", []) if isinstance(entry.get("changes"), list) else []:
                if not isinstance(change, dict):
                    continue
                value = change.get("value") if isinstance(change.get("value"), dict) else {}
                contacts = {
                    str(c.get("wa_id")): c
                    for c in value.get("contacts", [])
                    if isinstance(c, dict)
                }
                for item in value.get("messages", []) if isinstance(value.get("messages"), list) else []:
                    if not isinstance(item, dict):
                        continue
                    sender = str(item.get("from") or "unknown")
                    body = item.get("text") if isinstance(item.get("text"), dict) else {}
                    if not body and isinstance(item.get("button"), dict):
                        body = item["button"]
                    if not body and isinstance(item.get("interactive"), dict):
                        body = item["interactive"]
                    text = body.get("body") or body.get("text") or body.get("title") or ""
                    context = item.get("context") if isinstance(item.get("context"), dict) else {}
                    contact = contacts.get(sender, {})
                    # WhatsApp's `from` is the immutable phone id; contacts[].name
                    # is user-editable display metadata.
                    display = str(contact.get("name") or sender)
                    messages.append(
                        NormalizedMessage(
                            channel=self.channel,
                            sender=display,
                            text=str(text),
                            timestamp=_from_epoch(value.get("timestamp") or item.get("timestamp")),
                            reply_ref=_ref(context.get("id")),
                            message_id=_ref(item.get("id")),
                            sender_id=sender,
                            conversation_ref=sender,
                            thread=_ref(context.get("id")),
                        )
                    )
        return messages


ADAPTERS: dict[str, type[MessagingAdapter]] = {
    "telegram": TelegramAdapter,
    "discord": DiscordAdapter,
    "whatsapp": WhatsAppAdapter,
}

# Public aliases for channel modules in leo.integrations.channels: those
# modules share the exact byte-for-byte HMAC/header/parsing semantics as the
# core adapters instead of re-deriving them.
header_value = _header
body_bytes = _body_bytes
payload_dict = _payload_dict
payload_value = _parsed_payload
payload_dict_or_list = _payload_dict_or_list
hmac_hex = _hmac_hex
signature_matches = _signature_matches
from_epoch = _from_epoch
ref_or_none = _ref

# Descriptive aliases for callers that prefer the longer names.
TelegramMessagingAdapter = TelegramAdapter
DiscordMessagingAdapter = DiscordAdapter
WhatsAppMessagingAdapter = WhatsAppAdapter


def integration_records() -> list[IntegrationRecord]:
    """Status of all messaging adapters without contacting any platform."""
    return [ADAPTERS[name]().record() for name in sorted(ADAPTERS)]
