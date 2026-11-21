"""Signal adapter: COMMUNITY signal-cli-rest-api bridge (unofficial).

Honest status — read before trusting this channel: this adapter targets
**signal-cli-rest-api**, a community-built, self-hosted REST wrapper around
signal-cli (github.com/bbernhard/signal-cli-rest-api). It is NOT the official
Signal client or any Signal Messenger product — an unofficial bridge where
*you* register a number on a machine you control and the bridge keeps the
session. Nothing here is affiliated with or endorsed by Signal. It exists
because Signal has no official general-purpose bot API.

Sending uses the bridge's documented ``/v2/send`` endpoint (message body,
recipient list, registered sender number). Inbound webhooks are normalized
defensively and **media/attachments are out of scope**: only textual fields
are read, attachment payloads are never fetched or processed.

Webhook verification — important: **signal-cli-rest-api does not sign its
webhook POSTs by default.** An unauthenticated bridge URL on a public host
would let anyone inject messages, so LEO requires a fronting shared secret:
this adapter only accepts deliveries carrying ``X-Leo-Signature`` =
``"sha256=" + HMAC-SHA256(SIGNAL_WEBHOOK_SECRET, raw_body)``. That header is
LEO's own fail-closed convention, added by the reverse proxy/relay in front
of the bridge — with a missing secret or a missing header the webhook is
rejected.
"""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import urlparse

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
    hmac_hex,
    payload_dict,
    ref_or_none,
    signature_matches,
)
from ..secrets import credential


def _valid_url(url: str) -> bool:
    """http(s) URL with a host, no embedded credentials, no traversal tricks."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and ".." not in url
    )


def _ms_to_iso(value: Any) -> str:
    """signal-cli timestamps are milliseconds since the Unix epoch."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return from_epoch(value / 1000.0)
    return utcnow()


class SignalAdapter(MessagingAdapter):
    """signal-cli-rest-api sender plus signed-webhook normalization.

    The bridge password is basic auth (optional, only when
    ``SIGNAL_CLI_USER``/``SIGNAL_CLI_PASSWORD`` are both present) and never
    travels in an error. Inbound only textual content is normalized; a
    message with no text (attachment-only) is skipped because media is out
    of scope for this adapter.
    """

    channel = "signal"
    credential_env = ("SIGNAL_CLI_URL", "SIGNAL_PHONE_NUMBER")
    verification_env = ("SIGNAL_WEBHOOK_SECRET",)
    supports_webhook = True
    supports_polling = False

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        secret = credential("SIGNAL_WEBHOOK_SECRET")
        if not secret:
            # Fail closed: the bridge sends no signature by default, so
            # without LEO's fronting secret every delivery is untrusted.
            return not self.require_verification
        signature = header_value(headers, "x-leo-signature")
        return signature_matches(signature, hmac_hex(secret, body_bytes(payload, raw_body)))

    # --------------------------------------------------------------------- send

    async def send(self, target: str, text: str) -> dict[str, Any]:
        base = credential("SIGNAL_CLI_URL")
        number = credential("SIGNAL_PHONE_NUMBER")
        if not base:
            raise MessagingError("signal send blocked: SIGNAL_CLI_URL is not set")
        if not number:
            raise MessagingError("signal send blocked: SIGNAL_PHONE_NUMBER is not set")
        if not _valid_url(base):
            raise MessagingError("signal send blocked: SIGNAL_CLI_URL is not a valid http(s) URL")
        # Defensive recipient: phone number or group id, but never something
        # that could smuggle structure (whitespace is rejected outright).
        recipient = str(target)
        if not recipient or any(ch.isspace() for ch in recipient):
            raise MessagingError("signal send blocked: invalid Signal recipient")
        headers: dict[str, str] = {}
        user = credential("SIGNAL_CLI_USER")
        password = credential("SIGNAL_CLI_PASSWORD")
        if user and password:
            headers["Authorization"] = "Basic " + base64.b64encode(
                f"{user}:{password}".encode("utf-8")
            ).decode("ascii")
        url = f"{base.rstrip('/')}/v2/send"
        try:
            response = await self._http().post(
                url,
                headers=headers,
                json={"message": str(text), "recipients": [recipient], "number": number},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # Never echo httpx's message: it can contain the URL or password.
            raise MessagingError(f"signal send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"signal send failed: HTTP {response.status_code}")
        message_id: str | None = None
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            message_id = ref_or_none(data.get("timestamp"))
        return {
            "channel": self.channel,
            "target": recipient,
            "success": True,
            "message_id": message_id,
        }

    # ------------------------------------------------------------------ webhook

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError):
            # Untrusted input: a malformed delivery yields no messages.
            return []
        if not isinstance(data, dict):
            return []
        text = self._text(data)
        if text is None:
            return []  # attachment-only or empty: media is out of scope
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=self._sender(data),
                text=text,
                timestamp=_ms_to_iso(data.get("timestamp")),
                reply_ref=None,
                message_id=ref_or_none(data.get("id") or data.get("messageId")),
            )
        ]

    @staticmethod
    def _text(data: dict[str, Any]) -> str | None:
        """Textual content only — attachments' media payloads are never read."""
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
        # Some relay shapes wrap the text in a body object; only its textual
        # fields are read, never attachment media.
        body = data.get("body")
        if isinstance(body, str) and body:
            return body
        if isinstance(body, dict):
            text = body.get("text")
            if isinstance(text, str) and text:
                return text
            attachments = body.get("attachments")
            if isinstance(attachments, list):
                for attachment in attachments:
                    if isinstance(attachment, dict):
                        for key in ("text", "caption"):
                            value = attachment.get(key)
                            if isinstance(value, str) and value:
                                return value
        return None

    @staticmethod
    def _sender(data: dict[str, Any]) -> str:
        """Sender identity, defensively: number, object with a number, group
        id, else ``unknown`` (a group fallback keeps group admission possible
        when the relay strips the individual sender)."""
        for key in ("source", "sender"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict):
                for sub in ("number", "uuid", "id", "name"):
                    sub_value = value.get(sub)
                    if isinstance(sub_value, str) and sub_value:
                        return sub_value
        group_info = data.get("groupInfo")
        if isinstance(group_info, dict):
            group_id = group_info.get("groupId")
            if isinstance(group_id, str) and group_id:
                return group_id
        return "unknown"


__all__ = ["SignalAdapter"]
