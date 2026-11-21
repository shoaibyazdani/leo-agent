"""BlueBubbles adapter: COMMUNITY self-hosted iMessage bridge (unofficial).

Honest status — read before trusting this channel: BlueBubbles is a
**community-built, self-hosted iMessage bridge server** (bluebubbles.app),
not an Apple product and not an official API. It keeps an always-logged-in
macOS Messages session on a host you control and relays messages through its
own REST API. That means: macOS host required, the Apple ID used there is at
real risk of account action by Apple, and nothing here is affiliated with or
endorsed by Apple. ``BLUEBUBBLES_URL`` points at *your own* bridge instance.

Sending targets the bridge's documented ``/api/v1/message/text`` endpoint
with the bridge password in a ``password`` header (the documented auth
style). Inbound ``new-message`` webhooks are normalized defensively; media
and tapback payloads carry no plain text and are skipped.

Webhook verification — important: **BlueBubbles does not sign its webhook
POSTs.** An unauthenticated bridge URL on a public host would let anyone
inject messages, so LEO requires a fronting shared secret: this adapter only
accepts deliveries carrying ``X-Leo-Signature`` = ``"sha256=" +
HMAC-SHA256(BLUEBUBBLES_WEBHOOK_SECRET, raw_body)``. That header is LEO's
own fail-closed convention, added by the reverse proxy/relay in front of the
bridge — with a missing secret or a missing header the webhook is rejected.
"""

from __future__ import annotations

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
    """BlueBubbles timestamps are milliseconds since the Unix epoch."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return from_epoch(value / 1000.0)
    return utcnow()


class BlueBubblesAdapter(MessagingAdapter):
    """BlueBubbles bridge sender plus signed-webhook normalization.

    The bridge password travels in a ``password`` header (never the URL or
    an error). Message ids come from the nested ``message.guid`` of the send
    response when present. Inbound only the ``new-message`` event form is
    normalized; everything else yields no messages rather than guessing.
    """

    channel = "bluebubbles"
    credential_env = ("BLUEBUBBLES_URL", "BLUEBUBBLES_PASSWORD")
    verification_env = ("BLUEBUBBLES_WEBHOOK_SECRET",)
    supports_webhook = True
    supports_polling = False

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        secret = credential("BLUEBUBBLES_WEBHOOK_SECRET")
        if not secret:
            # Fail closed: BlueBubbles itself sends no signature, so without
            # LEO's fronting secret every delivery is untrusted and refused.
            return not self.require_verification
        signature = header_value(headers, "x-leo-signature")
        return signature_matches(signature, hmac_hex(secret, body_bytes(payload, raw_body)))

    # --------------------------------------------------------------------- send

    async def send(self, target: str, text: str) -> dict[str, Any]:
        base = credential("BLUEBUBBLES_URL")
        password = credential("BLUEBUBBLES_PASSWORD")
        if not base:
            raise MessagingError("bluebubbles send blocked: BLUEBUBBLES_URL is not set")
        if not password:
            raise MessagingError("bluebubbles send blocked: BLUEBUBBLES_PASSWORD is not set")
        if not _valid_url(base):
            raise MessagingError("bluebubbles send blocked: BLUEBUBBLES_URL is not a valid http(s) URL")
        guid = str(target)
        if not guid:
            raise MessagingError("bluebubbles send blocked: empty chat guid")
        url = f"{base.rstrip('/')}/api/v1/message/text"
        try:
            response = await self._http().post(
                url,
                params={"guid": guid},
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "LEO",
                    "password": password,
                },
                json={"message": str(text), "method": "POST", "guid": guid},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # Never echo httpx's message: it can contain the URL or password.
            raise MessagingError(f"bluebubbles send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"bluebubbles send failed: HTTP {response.status_code}")
        message_id: str | None = None
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("message"), dict):
            message_id = ref_or_none(data["message"].get("guid"))
        return {
            "channel": self.channel,
            "target": guid,
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
        if not isinstance(data, dict) or data.get("type") != "new-message":
            return []
        event_data = data.get("data")
        if not isinstance(event_data, dict):
            return []
        message = event_data.get("message") if isinstance(event_data.get("message"), dict) else {}
        text = event_data.get("text")
        if text is None:
            text = message.get("text")
        if not isinstance(text, str):
            # No plain text (e.g. attachment/tapback-only event): media is
            # out of scope for this adapter, so there is nothing to say.
            return []
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=self._sender(event_data, message),
                text=text,
                timestamp=_ms_to_iso(event_data.get("timestamp")),
                reply_ref=None,
                message_id=ref_or_none(event_data.get("guid") or message.get("guid")),
            )
        ]

    @staticmethod
    def _sender(event_data: dict[str, Any], message: dict[str, Any]) -> str:
        """iMessage sender, defensively: address string, handle id, else unknown."""
        from_address = event_data.get("fromAddress")
        if isinstance(from_address, str) and from_address:
            return from_address
        for source in (event_data.get("handle"), message.get("fromHandle")):
            if isinstance(source, dict) and isinstance(source.get("id"), str) and source["id"]:
                return source["id"]
            if isinstance(source, str) and source:
                return source
        return "unknown"


__all__ = ["BlueBubblesAdapter"]
