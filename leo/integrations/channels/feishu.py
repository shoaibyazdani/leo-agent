"""Feishu / Lark (飞书) open-platform channel adapter.

Outbound sends text messages through the IM API
(``POST /open-apis/im/v1/messages?receive_id_type=chat_id``). Sending is
two-step: a ``tenant_access_token`` is fetched from
``/open-apis/auth/v3/tenant_access_token/internal`` using the app id and
secret, cached in memory until shortly before its reported ``expire`` (the
clock is injectable for offline tests) and presented as
``Authorization: Bearer <token>``. The app secret appears in the token request
body, so httpx exception text — which would repeat it — is never echoed into
an error message.

Inbound verification follows Feishu's event-subscription v2 scheme:
``X-Lark-Signature`` is the hex HMAC-**SHA256** of ``encrypt_key`` over
``timestamp + nonce + raw body``, using the ``X-Lark-Request-Timestamp`` and
``X-Lark-Request-Nonce`` headers, compared in constant time. This is not SHA1.

Encrypted event callbacks (``{"encrypt": "<base64>"}``) are decrypted in this
adapter with the configured ``FEISHU_ENCRYPT_KEY`` (AES-256-CBC, key =
SHA256(encrypt key), IV = first 16 key bytes, PKCS7). A body that claims to be
encrypted but has no key, no crypto backend or malformed ciphertext is
rejected — never silently accepted or silently discarded. ``cryptography`` is
an optional backend and may be replaced with an injected ``decryptor``.

Honest limitations, stated up front:
- ``url_verification`` handshakes expect the ``challenge`` value echoed back
  in the HTTP response body. ``handle_webhook`` returns normalized messages,
  not response bodies, so the calling webhook route must perform that echo
  itself; this adapter simply returns ``[]`` for handshakes.
- Only ``im.message.receive_v1`` text messages are normalized; other event
  types and non-text message types are skipped.

Shared signed-webhook guard
---------------------------
:class:`SignedWebhookReplayGuard` is the timestamp-freshness and nonce /
message-id replay helper shared by every signed channel webhook in this
package (Feishu, QQ Bot, WeCom, Weixin and Teams). It is defined here so all
adapters import one implementation; it carries no Feishu-specific state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from ..messaging import (
    Ed25519Verifier,
    HeaderMap,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
    body_bytes,
    from_epoch,
    header_value,
    hmac_hex,
    payload_dict,
    ref_or_none,
    signature_matches,
)
from ..secrets import credential

# A token is refreshed this many seconds before its reported expiry so a
# token expiring mid-flight never 401s.
TOKEN_REFRESH_MARGIN = 60.0


class SignedWebhookReplayGuard:
    """Freshness window plus nonce/message-id replay suppression, in memory.

    Every signed webhook adapter owns one instance. ``check`` is called only
    after signature verification, so an attacker cannot consume a legitimate
    delivery's nonce with an unsigned request. Each accepted key is remembered
    for ``max_age_seconds``; expired keys are pruned and the table is bounded.

    ``timestamp`` accepts epoch seconds (number or numeric string) or an ISO
    8601 string. A missing/malformed/stale timestamp is rejected when one is
    supplied; channels without a timestamp rely on their nonce/message id and
    (as a last resort) a SHA-256 fingerprint of the signed body.
    """

    DEFAULT_MAX_AGE_SECONDS = 300.0
    DEFAULT_MAX_ENTRIES = 4096

    def __init__(
        self,
        *,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        now_fn: Callable[[], float] = time.time,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.max_age_seconds = max(1.0, float(max_age_seconds))
        self._now_fn = now_fn
        self._max_entries = max(1, int(max_entries))
        self._seen: dict[str, float] = {}

    @staticmethod
    def _epoch(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    def check(
        self,
        *,
        timestamp: Any = None,
        nonce: Any = None,
        message_id: Any = None,
        body: bytes | None = None,
        channel: str = "webhook",
    ) -> None:
        """Raise :class:`WebhookVerificationError` on stale/replayed input."""
        now = float(self._now_fn())
        if timestamp is not None:
            moment = self._epoch(timestamp)
            if moment is None or abs(now - moment) > self.max_age_seconds:
                raise WebhookVerificationError(f"{channel} webhook timestamp is missing, malformed or stale")
        keys: list[str] = []
        for value in (nonce, message_id):
            if value is not None and str(value).strip():
                keys.append(f"{channel}:id:{str(value).strip()}")
        if not keys and body is not None:
            keys.append(f"{channel}:body:{hashlib.sha256(bytes(body)).hexdigest()}")
        if not keys:
            return
        for key, seen_at in list(self._seen.items()):
            if now - seen_at > self.max_age_seconds:
                del self._seen[key]
        for key in keys:
            if key in self._seen:
                raise WebhookVerificationError(f"{channel} webhook replay detected")
        for key in keys:
            self._seen[key] = now
        if len(self._seen) > self._max_entries:
            oldest = sorted(self._seen, key=self._seen.get)[: len(self._seen) - self._max_entries]
            for key in oldest:
                del self._seen[key]


FeishuDecryptor = Callable[[str, str], str]


def _default_feishu_decryptor(encrypt_b64: str, encrypt_key: str) -> str:
    """Decrypt one Feishu ``{"encrypt": ...}`` event body to JSON text.

    ``cryptography`` is imported lazily so it stays optional; without it the
    call fails closed with a clear, key-free error.
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MessagingError(
            "feishu webhook decryption unavailable: install 'cryptography' or pass decryptor= to FeishuAdapter"
        ) from exc
    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    try:
        ciphertext = base64.b64decode(encrypt_b64, validate=True)
    except (TypeError, ValueError):
        raise MessagingError("feishu webhook failed: invalid encrypt field") from None
    if not ciphertext or len(ciphertext) % 16:
        raise MessagingError("feishu webhook failed: invalid encrypt field length")
    decryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    pad = padded[-1]
    if not 1 <= pad <= 16 or padded[-pad:] != bytes([pad]) * pad:
        raise MessagingError("feishu webhook failed: invalid plaintext padding")
    try:
        return padded[:-pad].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MessagingError("feishu webhook failed: decrypted body is not UTF-8") from None


class FeishuAdapter(MessagingAdapter):
    """Feishu open-platform sender plus event-subscription-v2 webhook normalizer."""

    channel = "feishu"
    credential_env = ("FEISHU_APP_ID", "FEISHU_APP_SECRET")
    verification_env = ("FEISHU_ENCRYPT_KEY",)
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://open.feishu.cn"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        require_verification: bool = True,
        signature_verifier: Ed25519Verifier | None = None,
        now_fn: Callable[[], float] = time.time,
        decryptor: FeishuDecryptor | None = None,
        replay_max_age_seconds: float = SignedWebhookReplayGuard.DEFAULT_MAX_AGE_SECONDS,
    ) -> None:
        super().__init__(
            client=client,
            timeout=timeout,
            require_verification=require_verification,
            signature_verifier=signature_verifier,
        )
        # Injectable clock so the token cache's expiry is testable offline.
        self._now_fn = now_fn
        self._tenant_token: str | None = None
        self._token_expires_at: float = 0.0
        self._decryptor: FeishuDecryptor = decryptor or _default_feishu_decryptor
        self._replay = SignedWebhookReplayGuard(
            max_age_seconds=replay_max_age_seconds, now_fn=now_fn
        )

    async def _tenant_access_token(self) -> str:
        """Tenant access token, cached in memory until shortly before expiry.

        The token value is a short-lived API credential, not a configured
        secret: it lives only in instance state and never surfaces in an
        error, a record or a log line.
        """
        if self._tenant_token and self._now_fn() < self._token_expires_at:
            return self._tenant_token
        app_id = credential("FEISHU_APP_ID")
        app_secret = credential("FEISHU_APP_SECRET")
        if not app_id:
            raise MessagingError("feishu send blocked: FEISHU_APP_ID is not set")
        if not app_secret:
            raise MessagingError("feishu send blocked: FEISHU_APP_SECRET is not set")
        try:
            response = await self._http().post(
                f"{self.API_BASE}/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": app_id, "app_secret": app_secret},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"feishu send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"feishu send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("feishu send failed: invalid response") from exc
        if not isinstance(data, dict):
            raise MessagingError("feishu send failed: invalid response")
        token = data.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise MessagingError("feishu send failed: API reported failure")
        try:
            ttl = float(data["expire"]) if data.get("expire") is not None else 0.0
        except (TypeError, ValueError):
            ttl = 0.0
        self._tenant_token = token
        self._token_expires_at = (
            self._now_fn() + ttl - TOKEN_REFRESH_MARGIN if ttl > TOKEN_REFRESH_MARGIN else 0.0
        )
        return token

    def verify_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> bool:
        encrypt_key = credential("FEISHU_ENCRYPT_KEY")
        if not encrypt_key:
            # Fail closed: an unsigned webhook must never be accepted just
            # because verification was never configured.
            return not self.require_verification
        timestamp = header_value(headers, "x-lark-request-timestamp")
        nonce = header_value(headers, "x-lark-request-nonce")
        signature = header_value(headers, "x-lark-signature")
        if not timestamp or not nonce or not signature:
            return False
        body = body_bytes(payload, raw_body)
        expected = hmac_hex(encrypt_key, timestamp.encode("utf-8") + nonce.encode("utf-8") + body)
        return signature_matches(signature, expected)

    async def send(self, target: str, text: str) -> dict[str, Any]:
        if not credential("FEISHU_APP_ID"):
            raise MessagingError("feishu send blocked: FEISHU_APP_ID is not set")
        if not credential("FEISHU_APP_SECRET"):
            raise MessagingError("feishu send blocked: FEISHU_APP_SECRET is not set")
        token = await self._tenant_access_token()
        try:
            response = await self._http().post(
                f"{self.API_BASE}/open-apis/im/v1/messages?receive_id_type=chat_id",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "receive_id": str(target),
                    "msg_type": "text",
                    "content": json.dumps({"text": str(text)}, ensure_ascii=False),
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"feishu send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"feishu send failed: HTTP {response.status_code}")
        message_id: str | None = None
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            code = data.get("code")
            if code is not None:
                try:
                    failed = int(code) != 0
                except (TypeError, ValueError):
                    failed = True
                if failed:
                    raise MessagingError("feishu send failed: API reported failure")
            inner = data.get("data") if isinstance(data.get("data"), dict) else {}
            message_id = ref_or_none(inner.get("message_id"))
        return {"channel": self.channel, "target": str(target), "success": True, "message_id": message_id}

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        if self.require_verification:
            # Consume nonces only after the signature is known-good.
            body = body_bytes(payload, raw_body)
            self._replay.check(
                timestamp=header_value(headers, "x-lark-request-timestamp"),
                nonce=header_value(headers, "x-lark-request-nonce"),
                body=body,
                channel=self.channel,
            )
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError):
            # Untrusted body: non-object or non-JSON payloads normalize to nothing.
            return []
        if data.get("type") == "url_verification":
            # Feishu expects the HTTP layer to echo data["challenge"] back in
            # the response body. This method returns messages, not response
            # bodies, so the caller's webhook route must do the echo; there is
            # nothing to normalize here either way.
            return []
        encrypt = data.get("encrypt")
        if isinstance(encrypt, str) and encrypt:
            encrypt_key = credential("FEISHU_ENCRYPT_KEY")
            if not encrypt_key:
                raise MessagingError("feishu webhook failed: FEISHU_ENCRYPT_KEY is not set")
            try:
                decrypted = self._decryptor(encrypt, encrypt_key)
            except MessagingError:
                raise
            except Exception as exc:  # noqa: BLE001 - untrusted input never crashes
                raise MessagingError(
                    f"feishu webhook failed: decryptor raised {type(exc).__name__}"
                ) from None
            try:
                data = json.loads(decrypted)
            except (TypeError, ValueError):
                raise MessagingError("feishu webhook failed: decrypted body is not JSON") from None
            if not isinstance(data, dict):
                raise MessagingError("feishu webhook failed: decrypted body is not an object")
        event = data.get("event")
        if not isinstance(event, dict):
            return []
        header = data.get("header") if isinstance(data.get("header"), dict) else {}
        event_type = header.get("event_type")
        if event_type is not None and event_type != "im.message.receive_v1":
            return []
        message = event.get("message")
        if not isinstance(message, dict) or message.get("message_type") != "text":
            return []
        content = message.get("content")
        text_body: dict[str, Any] | None = content if isinstance(content, dict) else None
        if text_body is None and isinstance(content, str):
            try:
                parsed = json.loads(content)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                text_body = parsed
        if not isinstance(text_body, dict) or text_body.get("text") is None:
            return []
        sender_block = event.get("sender") if isinstance(event.get("sender"), dict) else {}
        sender_ids = sender_block.get("sender_id") if isinstance(sender_block.get("sender_id"), dict) else {}
        sender = sender_ids.get("union_id") or sender_ids.get("open_id")
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=str(sender or "unknown"),
                text=str(text_body["text"]),
                timestamp=from_epoch(message.get("create_time")),
                reply_ref=None,
                message_id=ref_or_none(message.get("message_id")),
            )
        ]
