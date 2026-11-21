"""QQ official bot channel adapter (QQ 机器人开放平台, api.sgroup.qq.com).

Outbound uses the v2 group and c2c (single-chat) message endpoints with
``{"content": ..., "msg_type": 0}`` bodies (0 = plain text). Targets are
``"group:<group_openid>"`` → ``POST /v2/groups/{group_openid}/messages`` and
``"c2c:<openid>"`` → ``POST /v2/c2c/{openid}/messages``; anything else is
rejected before any network call.

Authentication: QQ's official docs (bot.qq.com, 接口调用与鉴权) specify the
header verbatim as ``Authorization: QQBot ACCESS_TOKEN`` — a plain
``QQBot <token>`` prefix, *not* ``Bearer``; the legacy ``Bot <token>`` scheme
is deprecated. The access token comes from
``GET {API_BASE}/app/getAppAccessToken?appId=..&clientSecret=..`` and is cached
in memory until shortly before its reported ``expires_in`` (clock injectable
for offline tests). QQ reports business errors in the response body
(``code``/``message``) even with HTTP 200; a token response without
``access_token`` is treated as a failed send. Honest caveat: QQ's current docs
also describe a POST+JSON-body form of the token endpoint on ``bots.qq.com``;
this adapter implements the GET-with-query form on the API base as specified
by LEO's platform contract — if QQ retires that form, ``_access_token`` is the
single method to update.

Inbound callbacks are Ed25519-signed exactly like Discord interactions:
``X-Signature-Ed25519`` over ``timestamp + raw body`` verified with
``QQBOT_PUBLIC_KEY``. Verification is fail-closed and runs through the base
adapter's injectable ``signature_verifier``; when no crypto backend
(cryptography/PyNaCl) is installed and none was injected, verification raises
``WebhookVerificationError`` instead of accepting.

Group and c2c callback payloads carry no uniform timestamp field, so
normalized messages are stamped honestly with arrival time (``utcnow``)
rather than a payload-provided instant.
"""

from __future__ import annotations

import time
from typing import Any, Callable
from urllib.parse import quote

import httpx

from ...contracts import utcnow
from ..messaging import (
    Ed25519Verifier,
    HeaderMap,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    WebhookVerificationError,
    _default_ed25519_verifier,
    body_bytes,
    header_value,
    payload_dict,
    ref_or_none,
)
from ..secrets import credential
from .feishu import SignedWebhookReplayGuard

# A token is refreshed this many seconds before its reported expiry so a
# token expiring mid-flight never 401s.
TOKEN_REFRESH_MARGIN = 60.0


def _strip_bot_mention(text: str) -> str:
    """Drop a leading @-mention of the bot from a group/c2c message, defensively.

    ``<@!id> hi`` and ``@name hi`` are mentions; ``@word`` (no space after a
    non-angle token) is left alone as a literal word.
    """
    stripped = text.lstrip()
    if stripped.startswith("<@"):
        end = stripped.find(">")
        if end != -1:
            return stripped[end + 1 :].lstrip()
        return stripped
    if stripped.startswith("@"):
        head, sep, tail = stripped[1:].partition(" ")
        if sep:
            return tail.lstrip()
        return stripped  # no trailing space: a literal "@word", not a mention
    return text


class QQBotAdapter(MessagingAdapter):
    """QQ official bot sender plus Ed25519-signed callback normalizer."""

    channel = "qqbot"
    credential_env = ("QQBOT_APP_ID", "QQBOT_APP_SECRET")
    verification_env = ("QQBOT_PUBLIC_KEY",)
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://api.sgroup.qq.com"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        require_verification: bool = True,
        signature_verifier: Ed25519Verifier | None = None,
        now_fn: Callable[[], float] = time.time,
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
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._replay = SignedWebhookReplayGuard(
            max_age_seconds=replay_max_age_seconds, now_fn=self._now_fn
        )

    async def _access_token(self) -> str:
        """App access token, cached in memory until shortly before expiry.

        The token value is a short-lived API credential, not a configured
        secret: it lives only in instance state and never surfaces in an
        error, a record or a log line.
        """
        if self._token and self._now_fn() < self._token_expires_at:
            return self._token
        app_id = credential("QQBOT_APP_ID")
        app_secret = credential("QQBOT_APP_SECRET")
        if not app_id:
            raise MessagingError("qqbot send blocked: QQBOT_APP_ID is not set")
        if not app_secret:
            raise MessagingError("qqbot send blocked: QQBOT_APP_SECRET is not set")
        try:
            response = await self._http().get(
                f"{self.API_BASE}/app/getAppAccessToken",
                params={"appId": app_id, "clientSecret": app_secret},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # The URL embeds the client secret: never echo httpx's text.
            raise MessagingError(f"qqbot send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"qqbot send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("qqbot send failed: invalid response") from exc
        if not isinstance(data, dict) or not isinstance(data.get("access_token"), str) or not data["access_token"]:
            raise MessagingError("qqbot send failed: API reported failure")
        token = data["access_token"]
        try:
            ttl = float(data["expires_in"]) if data.get("expires_in") is not None else 0.0
        except (TypeError, ValueError):
            ttl = 0.0
        self._token = token
        self._token_expires_at = (
            self._now_fn() + ttl - TOKEN_REFRESH_MARGIN if ttl > TOKEN_REFRESH_MARGIN else 0.0
        )
        return token

    def verify_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> bool:
        public_key = credential("QQBOT_PUBLIC_KEY")
        if not public_key:
            # Fail closed: an unsigned callback must never be accepted just
            # because verification was never configured.
            return not self.require_verification
        signature = header_value(headers, "x-signature-ed25519")
        timestamp = header_value(headers, "x-signature-timestamp")
        if not signature or not timestamp:
            return False
        verifier = self.signature_verifier or _default_ed25519_verifier
        body = body_bytes(payload, raw_body)
        try:
            return bool(verifier(public_key, signature, timestamp, body))
        except WebhookVerificationError:
            raise
        except Exception as exc:
            raise WebhookVerificationError(f"qqbot signature verifier failed: {type(exc).__name__}") from exc

    async def send(self, target: str, text: str) -> dict[str, Any]:
        if not credential("QQBOT_APP_ID"):
            raise MessagingError("qqbot send blocked: QQBOT_APP_ID is not set")
        if not credential("QQBOT_APP_SECRET"):
            raise MessagingError("qqbot send blocked: QQBOT_APP_SECRET is not set")
        target_str = str(target)
        if target_str.startswith("group:"):
            scope, openid = "groups", target_str[len("group:") :].strip()
        elif target_str.startswith("c2c:"):
            scope, openid = "c2c", target_str[len("c2c:") :].strip()
        else:
            raise MessagingError("qqbot send blocked: target must be 'group:<openid>' or 'c2c:<openid>'")
        if not openid:
            raise MessagingError("qqbot send blocked: target must be 'group:<openid>' or 'c2c:<openid>'")
        token = await self._access_token()
        url = f"{self.API_BASE}/v2/{scope}/{quote(openid, safe='')}/messages"
        try:
            response = await self._http().post(
                url,
                headers={"Authorization": f"QQBot {token}"},
                json={"content": str(text), "msg_type": 0},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"qqbot send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"qqbot send failed: HTTP {response.status_code}")
        message_id: str | None = None
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            message_id = ref_or_none(data.get("message_id") or data.get("id"))
        return {"channel": self.channel, "target": target_str, "success": True, "message_id": message_id}

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError):
            # Untrusted body: non-object or non-JSON payloads normalize to nothing.
            return []
        bodies: list[dict[str, Any]] = []
        if isinstance(data.get("event"), dict):
            # HTTP callback envelope: {"event_type": ..., "event": {...}}
            bodies.append(data["event"])
        elif isinstance(data.get("d"), dict):
            # Gateway/webhook envelope: {"op": 0, "t": ..., "d": {...}}
            bodies.append(data["d"])
        elif isinstance(data.get("content"), str) and (
            "group_openid" in data or "member_openid" in data or isinstance(data.get("author"), dict)
        ):
            bodies.append(data)  # direct message object
        messages: list[NormalizedMessage] = []
        for body in bodies:
            content = body.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            sender = body.get("member_openid")
            if not sender:
                author = body.get("author") if isinstance(body.get("author"), dict) else {}
                sender = author.get("member_openid") or author.get("open_id")
            message_id = ref_or_none(body.get("message_id") or body.get("id"))
            if self.require_verification:
                # Freshness comes from the signed header; replay is keyed on
                # the message id, with a body fingerprint as the fallback.
                self._replay.check(
                    timestamp=header_value(headers, "x-signature-timestamp"),
                    message_id=message_id,
                    body=body_bytes(payload, raw_body),
                    channel=self.channel,
                )
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    sender=str(sender or "unknown"),
                    text=_strip_bot_mention(content),
                    # QQ callback payloads carry no stable, uniform timestamp
                    # field, so arrival time is used honestly instead.
                    timestamp=utcnow(),
                    reply_ref=None,
                    message_id=message_id,
                )
            )
        return messages
