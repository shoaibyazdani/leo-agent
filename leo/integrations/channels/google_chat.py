"""Google Chat adapter: incoming-webhook sending and event normalization.

``GOOGLE_CHAT_WEBHOOK_URL`` is the space-scoped incoming webhook URL. It
embeds its own key parameter, so it is treated as a credential: validated
against Google's host/path shape before use and never echoed in errors.
``GOOGLE_CHAT_VERIFICATION_TOKEN`` is the app's configured verification
token, presented by Google Chat on inbound event deliveries.
"""

from __future__ import annotations

import hmac
from typing import Any

import httpx

from ...contracts import utcnow
from ..messaging import (
    HeaderMap,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    header_value,
    payload_dict,
    ref_or_none,
)
from ..secrets import credential


def _iso_or_now(value: Any) -> str:
    """RFC 3339 ``createTime`` strings pass through; anything else is "now"."""
    return value if isinstance(value, str) and value else utcnow()


class GoogleChatAdapter(MessagingAdapter):
    """Google Chat webhook sender plus inbound event normalization.

    Verification: a Google Chat app can be configured with a verification
    token; inbound deliveries present it in the ``Authorization`` header
    as ``Bearer <token>`` (the same value in ``X-Goog-Chat-Token`` is also
    accepted). The presented value is compared with the configured token
    in constant time; a missing token, a missing header or a wrong value
    fails closed.

    Message payloads arrive in the two shapes Google Chat uses — an
    ``{"event": ...}`` envelope or a bare message body — and bot-authored
    messages are skipped so the agent never replies to itself.
    """

    channel = "google_chat"
    credential_env = ("GOOGLE_CHAT_WEBHOOK_URL",)
    verification_env = ("GOOGLE_CHAT_VERIFICATION_TOKEN",)
    supports_webhook = True
    supports_polling = False

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
            and host.endswith(".googleapis.com")
            and parsed.path.startswith("/v1/spaces/")
            and "/messages" in parsed.path
            and not parsed.username
            and not parsed.password
        )

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        token = credential("GOOGLE_CHAT_VERIFICATION_TOKEN")
        if not token:
            # Fail closed: without the configured verification token no
            # inbound delivery is trusted.
            return not self.require_verification
        expected = token.encode("utf-8")
        authorization = header_value(headers, "authorization")
        if authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer" and value.strip():
                if hmac.compare_digest(value.strip().encode("utf-8"), expected):
                    return True
        chat_token = header_value(headers, "x-goog-chat-token")
        if chat_token and hmac.compare_digest(chat_token.strip().encode("utf-8"), expected):
            return True
        return False

    async def send(self, target: str, text: str) -> dict[str, Any]:
        url = credential("GOOGLE_CHAT_WEBHOOK_URL")
        if not url:
            raise MessagingError("google_chat send blocked: GOOGLE_CHAT_WEBHOOK_URL is not set")
        if not self._valid_webhook_url(url):
            # The URL embeds a key parameter: never echo it in the error.
            raise MessagingError("google_chat send blocked: not a Google Chat webhook URL")
        try:
            # The webhook URL is space-scoped: ``target`` is kept for the
            # response record but does not change the destination.
            response = await self._http().post(url, json={"text": str(text)}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise MessagingError(f"google_chat send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"google_chat send failed: HTTP {response.status_code}")
        message_id: str | None = None
        if response.content:
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                message_id = ref_or_none(data.get("name"))
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
        body = data.get("event") if isinstance(data.get("event"), dict) else data
        message = body.get("message") if isinstance(body.get("message"), dict) else None
        if message is not None:
            message_sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
            user = body.get("user") if isinstance(body.get("user"), dict) else {}
            event_sender = body.get("sender") if isinstance(body.get("sender"), dict) else {}
            if (
                message_sender.get("type") == "BOT"
                or user.get("type") == "BOT"
                or event_sender.get("type") == "BOT"
            ):
                return []
            thread = message.get("thread") if isinstance(message.get("thread"), dict) else {}
            return [
                NormalizedMessage(
                    channel=self.channel,
                    sender=str(
                        user.get("displayName")
                        or message_sender.get("name")
                        or user.get("name")
                        or event_sender.get("name")
                        or "unknown"
                    ),
                    text=str(message.get("text") or ""),
                    timestamp=_iso_or_now(message.get("createTime")),
                    reply_ref=None,
                    message_id=ref_or_none(message.get("name") or thread.get("name")),
                )
            ]
        # Bare message-body shape: {"sender": {...}, "text": ...}.
        sender_obj = body.get("sender") if isinstance(body.get("sender"), dict) else {}
        if sender_obj.get("type") == "BOT":
            return []
        if "text" not in body and "sender" not in body:
            return []  # unrecognized shape: nothing to normalize
        thread = body.get("thread") if isinstance(body.get("thread"), dict) else {}
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=str(sender_obj.get("name") or "unknown"),
                text=str(body.get("text") or ""),
                timestamp=_iso_or_now(body.get("createTime")),
                reply_ref=None,
                message_id=ref_or_none(body.get("name") or thread.get("name")),
            )
        ]
