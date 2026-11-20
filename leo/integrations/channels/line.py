"""LINE Messaging API adapter: signed webhook ingress and push sending.

``LINE_CHANNEL_ACCESS_TOKEN`` is the channel access token used for outbound
calls; ``LINE_CHANNEL_SECRET`` is the channel secret used to verify inbound
webhook deliveries. Both are read from the environment by *name* at call
time — no token value is ever stored on the adapter or echoed in errors.

Outbound messages use the push endpoint rather than the reply endpoint:
LINE's ``replyToken`` expires shortly after an event is delivered, so a
delayed agent reply cannot rely on it. Push needs no token and delivers
directly to the user/group/room id that ``target`` names.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any

import httpx

from ...contracts import utcnow
from ..messaging import (
    HeaderMap,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    body_bytes,
    from_epoch,
    header_value,
    payload_dict,
    ref_or_none,
)
from ..secrets import credential


def _ms_to_iso(value: Any) -> str:
    """LINE event timestamps are milliseconds since the Unix epoch.

    :func:`from_epoch` expects seconds, so the value is divided by 1000
    first; anything non-numeric falls back to "now" like the core adapters.
    """
    try:
        seconds = float(value) / 1000.0
    except (TypeError, ValueError):
        return utcnow()
    return from_epoch(seconds)


class LineAdapter(MessagingAdapter):
    """LINE Messaging API sender plus webhook event normalization.

    Verification: LINE signs every webhook delivery with
    ``X-Line-Signature`` = base64(HMAC-SHA256(channel_secret, raw_body)).
    The signature is recomputed over the exact request bytes and compared
    with ``hmac.compare_digest``; a missing header or a missing secret
    fails closed.

    Sending uses the push endpoint (``/v2/bot/message/push``) rather than
    the reply endpoint: the ``replyToken`` attached to an inbound event
    expires quickly, so a delayed agent reply cannot rely on it. Push
    delivers to the user/group/room id named by ``target``.
    """

    channel = "line"
    credential_env = ("LINE_CHANNEL_ACCESS_TOKEN",)
    verification_env = ("LINE_CHANNEL_SECRET",)
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://api.line.me"

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        secret = credential("LINE_CHANNEL_SECRET")
        if not secret:
            # Fail closed: an unsigned webhook must never be accepted just
            # because verification was never configured.
            return not self.require_verification
        provided = header_value(headers, "x-line-signature")
        if not provided:
            return False
        digest = hmac.new(secret.encode("utf-8"), body_bytes(payload, raw_body), hashlib.sha256).digest()
        expected = base64.b64encode(digest).decode("ascii")
        # Compare over bytes so non-ASCII header values cannot raise.
        return hmac.compare_digest(provided.strip().encode("utf-8"), expected.encode("ascii"))

    async def send(self, target: str, text: str) -> dict[str, Any]:
        token = credential("LINE_CHANNEL_ACCESS_TOKEN")
        if not token:
            raise MessagingError("line send blocked: LINE_CHANNEL_ACCESS_TOKEN is not set")
        try:
            response = await self._http().post(
                f"{self.API_BASE}/v2/bot/message/push",
                headers={"Authorization": f"Bearer {token}"},
                json={"to": str(target), "messages": [{"type": "text", "text": str(text)}]},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # Never include httpx's message: it can echo the request.
            raise MessagingError(f"line send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"line send failed: HTTP {response.status_code}")
        message_id: str | None = None
        if response.content:
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                message_id = ref_or_none(data.get("messageId"))
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
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError):
            # Untrusted input: a malformed delivery yields no messages.
            return []
        messages: list[NormalizedMessage] = []
        events = data.get("events")
        if not isinstance(events, list):
            return messages
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "message":
                continue  # follow/join/leave/postback/member events carry no text
            message = event.get("message")
            if not isinstance(message, dict) or message.get("type") != "text":
                continue
            source = event.get("source") if isinstance(event.get("source"), dict) else {}
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    sender=str(source.get("userId") or source.get("groupId") or "unknown"),
                    text=str(message.get("text") or ""),
                    timestamp=_ms_to_iso(event.get("timestamp")),
                    reply_ref=None,
                    message_id=ref_or_none(message.get("id")),
                )
            )
        return messages
