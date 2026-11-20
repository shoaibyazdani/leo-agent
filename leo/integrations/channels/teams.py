"""Microsoft Teams adapter: incoming-webhook sending and Bot Framework ingress.

``TEAMS_INCOMING_WEBHOOK_URL`` is a Power Automate/workflows incoming
webhook URL. Inbound ``/api/messages`` traffic carries a Bot Framework JWT.
This module verifies that JWT itself with a stdlib-only HS256 path: the token
is read from ``Authorization: Bearer`` (or a gateway-copied
``X-Teams-Signature``), its signature is checked with HMAC-SHA256 using
``TEAMS_PUBLIC_KEY`` as the shared verification secret, and the ``aud``,
``iss``, ``exp``/``nbf`` and ``iat`` claims are enforced. ``TEAMS_APP_ID`` is
the expected audience and ``TEAMS_JWT_ISSUER`` optionally overrides the
default Bot Framework issuer.

RSA-signed tokens (``RS256``) from Microsoft's OpenId keys are *not* accepted
by the stdlib path: they fail closed with a clear error. Operators who need
them inject a ``signature_verifier`` that performs the RSA/JWKS validation,
or front ``/api/messages`` with a gateway that does. The injected callable
keeps the same ``(public_key, token, timestamp, body) -> bool`` shape used by
the other signed channels.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from ...contracts import utcnow
from ..messaging import (
    HeaderMap,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
    body_bytes,
    header_value,
    payload_dict,
    ref_or_none,
)
from ..secrets import credential
from .feishu import SignedWebhookReplayGuard

#: Bot Framework's production token issuer. Emulator/direct-line deployments
#: can override it with TEAMS_JWT_ISSUER.
DEFAULT_JWT_ISSUER = "https://api.botframework.com"


def _iso_or_now(value: Any) -> str:
    """Bot Framework timestamps are ISO strings; anything else is "now"."""
    return value if isinstance(value, str) and value else utcnow()


def _b64url_decode(segment: str) -> bytes:
    """Decode one JWT base64url segment, tolerating omitted padding."""
    if not isinstance(segment, str) or not segment:
        raise ValueError("empty JWT segment")
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _numeric_claim(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_teams_jwt_verifier(
    public_key: str,
    signature: str,
    timestamp: str,
    body: bytes,
    *,
    now_fn: Callable[[], float] = time.time,
) -> bool:
    """Verify a Bot Framework HS256 JWT: signature, audience, issuer and time.

    Every failure is a :class:`WebhookVerificationError` with no token or key
    material in its text. ``timestamp``/``body`` are unused by the JWT path
    (the claims carry time) but remain in the signature so injected verifiers
    and the other adapters share one contract.
    """
    parts = signature.strip().split(".")
    if len(parts) != 3:
        raise WebhookVerificationError("msteams jwt verification failed: malformed token")
    try:
        header = json.loads(_b64url_decode(parts[0]))
        claims = json.loads(_b64url_decode(parts[1]))
        supplied = _b64url_decode(parts[2])
    except (ValueError, TypeError, UnicodeDecodeError):
        raise WebhookVerificationError("msteams jwt verification failed: malformed token") from None
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise WebhookVerificationError("msteams jwt verification failed: unsupported algorithm")
    if not isinstance(claims, dict):
        raise WebhookVerificationError("msteams jwt verification failed: malformed claims")
    expected = hmac.new(
        public_key.encode("utf-8"), f"{parts[0]}.{parts[1]}".encode("utf-8"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, supplied):
        raise WebhookVerificationError("msteams jwt verification failed: bad signature")

    audience = credential("TEAMS_APP_ID")
    if not audience:
        raise WebhookVerificationError("msteams jwt verification requires TEAMS_APP_ID")
    raw_aud = claims.get("aud")
    aud = [raw_aud] if isinstance(raw_aud, str) else raw_aud if isinstance(raw_aud, list) else []
    if audience not in [str(value) for value in aud]:
        raise WebhookVerificationError("msteams jwt verification failed: audience mismatch")
    issuer = credential("TEAMS_JWT_ISSUER") or DEFAULT_JWT_ISSUER
    if claims.get("iss") != issuer:
        raise WebhookVerificationError("msteams jwt verification failed: issuer mismatch")

    now = float(now_fn())
    expires = _numeric_claim(claims.get("exp"))
    if expires is None or expires <= now:
        raise WebhookVerificationError("msteams jwt verification failed: expired token")
    not_before = _numeric_claim(claims.get("nbf"))
    if not_before is not None and not_before > now + 60:
        raise WebhookVerificationError("msteams jwt verification failed: token not yet valid")
    issued = _numeric_claim(claims.get("iat"))
    if issued is not None and now - issued > 3600:
        raise WebhookVerificationError("msteams jwt verification failed: stale token")
    return True


class TeamsAdapter(MessagingAdapter):
    """Microsoft Teams sender plus verified Bot Framework activity normalization.

    Verification is injectable (``signature_verifier``) and, by default,
    performed by the stdlib HS256 verifier above. Signed activities are also
    protected by a timestamp-freshness / message-id replay guard. Sending
    posts to the configured Power Automate/workflows incoming webhook;
    Microsoft answers 202 Accepted, which counts as success.
    """

    channel = "msteams"
    credential_env = ("TEAMS_INCOMING_WEBHOOK_URL",)
    # TEAMS_APP_ID is the JWT audience: it is verification material, not a
    # sending credential, so missing it degrades verification_health(). It
    # stays optional (required=False) in the enrollment field spec.
    verification_env = ("TEAMS_PUBLIC_KEY", "TEAMS_APP_ID")
    optional_env = ("TEAMS_JWT_ISSUER",)
    supports_webhook = True
    supports_polling = False
    WEBHOOK_HOST_SUFFIXES = (".office.com", ".microsoft.com")

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        require_verification: bool = True,
        signature_verifier: Callable[[str, str, str, bytes], bool] | None = None,
        now_fn: Callable[[], float] = time.time,
        replay_max_age_seconds: float = SignedWebhookReplayGuard.DEFAULT_MAX_AGE_SECONDS,
    ) -> None:
        super().__init__(
            client=client,
            timeout=timeout,
            require_verification=require_verification,
            signature_verifier=signature_verifier,
        )
        self._now_fn = now_fn
        self._replay = SignedWebhookReplayGuard(
            max_age_seconds=replay_max_age_seconds, now_fn=now_fn
        )

    @classmethod
    def _valid_webhook_url(cls, url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        host = (parsed.hostname or "").lower()
        return (
            parsed.scheme == "https"
            and (host.endswith(".office.com") or host.endswith(".microsoft.com"))
            and not parsed.username
            and not parsed.password
        )

    @staticmethod
    def _token_from_headers(headers: HeaderMap) -> str | None:
        authorization = header_value(headers, "authorization") or ""
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            if token:
                return token
        return header_value(headers, "x-teams-signature")

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        public_key = credential("TEAMS_PUBLIC_KEY")
        if not public_key:
            # Fail closed: without key material configured, no inbound
            # Bot Framework activity is trusted.
            return not self.require_verification
        token = self._token_from_headers(headers)
        if not token:
            return False
        body = body_bytes(payload, raw_body)
        if self.signature_verifier is not None:
            verifier = self.signature_verifier
        else:
            verifier = lambda key, jwt, ts, raw: _default_teams_jwt_verifier(  # noqa: E731
                key, jwt, ts, raw, now_fn=self._now_fn
            )
        try:
            # The timestamp slot is empty: the JWT carries its own claims.
            return bool(verifier(public_key, token, "", body))
        except WebhookVerificationError:
            raise
        except Exception as exc:
            raise WebhookVerificationError(f"msteams signature verifier failed: {type(exc).__name__}") from exc

    async def send(self, target: str, text: str) -> dict[str, Any]:
        url = credential("TEAMS_INCOMING_WEBHOOK_URL")
        if not url:
            raise MessagingError("msteams send blocked: TEAMS_INCOMING_WEBHOOK_URL is not set")
        if not self._valid_webhook_url(url):
            raise MessagingError("msteams send blocked: not a Teams webhook URL")
        try:
            response = await self._http().post(
                url, json={"type": "message", "text": str(text)}, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            # Never include httpx's message: it can echo the request.
            raise MessagingError(f"msteams send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"msteams send failed: HTTP {response.status_code}")
        # Incoming webhooks answer 202 Accepted with no message id.
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": None,
        }

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError):
            # Untrusted input: a malformed activity yields no messages.
            return []
        messages: list[NormalizedMessage] = []
        if data.get("type") != "message":
            return messages  # conversationUpdate, typing, installationUpdate, ...
        sender_obj = data.get("from") if isinstance(data.get("from"), dict) else {}
        if sender_obj.get("role") == "bot":
            return messages  # never react to the bot's own activity
        if self.require_verification:
            self._replay.check(
                timestamp=data.get("timestamp"),
                message_id=data.get("id"),
                body=body_bytes(payload, raw_body),
                channel=self.channel,
            )
        messages.append(
            NormalizedMessage(
                channel=self.channel,
                sender=str(sender_obj.get("name") or sender_obj.get("id") or "unknown"),
                text=str(data.get("text") or ""),
                timestamp=_iso_or_now(data.get("timestamp")),
                reply_ref=ref_or_none(data.get("replyToId")),
                message_id=ref_or_none(data.get("id")),
            )
        )
        return messages
