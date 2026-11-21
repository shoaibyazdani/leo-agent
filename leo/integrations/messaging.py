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
from typing import Any, Callable, Mapping, Type

import httpx
from pydantic import Field

from ..contracts import BaseRecord, IntegrationRecord, utcnow
from .secrets import credential


class MessagingError(RuntimeError):
    """Adapter could not send or process a message."""


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

    async def send(self, target: str, text: str) -> dict[str, Any]:
        raise NotImplementedError

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        raise NotImplementedError

    async def poll(self) -> list[NormalizedMessage]:
        """Pull inbound messages from channels without webhook delivery.

        Adapters with a poll transport override this and set
        ``supports_polling = True``. Poll state (e.g. a ``since`` cursor) is
        the adapter's own responsibility and starts empty each process.
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

    async def send(self, target: str, text: str) -> dict[str, Any]:
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token:
            raise MessagingError("telegram send blocked: TELEGRAM_BOT_TOKEN is not set")
        # The token appears in the URL: never echo the URL or httpx's exception text.
        url = f"{self.API_BASE}/bot{token}/sendMessage"
        try:
            response = await self._http().post(
                url, json={"chat_id": str(target), "text": str(text)}, timeout=self.timeout
            )
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
                )
            )
        return messages

    # ------------------------------------------------------------------- poll

    # LEO has no per-user authorisation (see LEO_SECURITY.md): the owner/from
    # fields are data, not an auth boundary. A poll transport therefore has to
    # carry its own allow-list, and it fails CLOSED — an unset list means the
    # bot refuses every message rather than answering a stranger.
    allowlist_env = "TELEGRAM_ALLOWED_USER_IDS"
    _cursor: int | None = None

    def _allowed_senders(self) -> set[str]:
        raw = credential(self.allowlist_env) or ""
        return {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}

    async def poll(self) -> list[NormalizedMessage]:
        token = credential("TELEGRAM_BOT_TOKEN")
        if not token:
            raise MessagingError("telegram poll blocked: TELEGRAM_BOT_TOKEN is not set")
        allowed = self._allowed_senders()
        if not allowed:
            return []
        params: dict[str, str] = {"timeout": "0"}
        if self._cursor is not None:
            params["offset"] = str(self._cursor)
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
                # ponytail: cursor is in-process; a restart re-fetches recent
                # updates, which the gateway drops on its durable message-id
                # dedupe. Persist it in kv only if replay ever shows up twice.
                self._cursor = update_id + 1
        return [m for m in self._messages_from_updates(updates) if m.sender_id in allowed]


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
