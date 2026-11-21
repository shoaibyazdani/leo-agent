"""LEO relay — LEO's own documented generic signed-JSON webhook transport.

This is *not* an official platform API: the relay channel is LEO's own
documented generic transport for LEO-to-LEO links and bridge integrations
where no official platform API exists. It exists so two LEO deployments — or
a LEO and a small bridge script — can exchange messages with nothing but a
shared secret and an HTTP endpoint.

One secret, ``RELAY_SECRET``, signs both directions:

* outbound ``send`` POSTs a compact JSON envelope to ``RELAY_TARGET_URL``
  with ``X-Leo-Signature: sha256=<hex>``, the HMAC-SHA256 of the *exact*
  bytes on the wire. The envelope is serialized once
  (``separators=(",", ":"), sort_keys=True``) and that exact string is sent
  via ``content=`` — not ``json=`` — so the signature matches the body
  byte-for-byte.
* inbound ``handle_webhook`` verifies the same header over the raw request
  bytes, fail-closed when the secret is unset, constant-time compare. The
  same secret being both credential and verification secret is by design:
  whoever can sign can also be trusted at the door.

Because the outbound envelope is exactly the shape ``handle_webhook``
normalizes (``sender``/``text``/``target``/``sent_at``/``message_id``), one
LEO can receive from another LEO with two adapters sharing one secret.

Webhook payloads are untrusted data: every field is type-checked, sender
names are truncated to 200 characters, text is capped at 10,000, and
malformed envelopes are skipped (``[]``), never fatal. No secret value is
ever stored on the adapter, logged, or echoed in an error message.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
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
    hmac_hex,
    header_value,
    payload_dict,
    ref_or_none,
    signature_matches,
)
from ..secrets import credential

# Inbound field caps: untrusted sender names and text bodies are bounded so a
# hostile peer cannot flood memory or logs through the relay envelope.
_SENDER_LIMIT = 200
_TEXT_LIMIT = 10_000


def _valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


def _parse_stamp(value: Any) -> str | None:
    """ISO string or epoch seconds to an ISO string; None when unparseable."""
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return from_epoch(value)
    return None


class RelayAdapter(MessagingAdapter):
    """Signed JSON envelope transport between LEO deployments (LEO's own)."""

    channel = "relay"
    # RELAY_SECRET is both a credential (it signs outbound) and the
    # verification secret (the same HMAC verifies inbound): one shared
    # secret, both directions.
    credential_env = ("RELAY_TARGET_URL", "RELAY_SECRET")
    verification_env = ("RELAY_SECRET",)
    supports_webhook = True
    supports_polling = False

    async def send(self, target: str, text: str) -> dict[str, Any]:
        url = credential("RELAY_TARGET_URL")
        secret = credential("RELAY_SECRET")
        if not url:
            raise MessagingError("relay send blocked: RELAY_TARGET_URL is not set")
        if not secret:
            raise MessagingError("relay send blocked: RELAY_SECRET is not set")
        if not _valid_http_url(url):
            raise MessagingError("relay send blocked: RELAY_TARGET_URL is not a valid http(s) URL")
        message_id = uuid.uuid4().hex
        envelope = {
            "sender": "leo",
            "text": str(text),
            "target": str(target),
            "sent_at": utcnow(),
            "message_id": message_id,
        }
        # Serialize once, sign those exact bytes, send those exact bytes: the
        # receiver verifies the HMAC over the raw body, so the signature must
        # match the wire content byte-for-byte (content=, not json=).
        body = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Leo-Signature": f"sha256={hmac_hex(secret, body)}",
        }
        try:
            response = await self._http().post(
                url, headers=headers, content=body, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            # Never echo httpx's exception text: it may repeat request data.
            raise MessagingError(f"relay send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"relay send failed: HTTP {response.status_code}")
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": message_id,
        }

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        secret = credential("RELAY_SECRET")
        if not secret:
            # Fail closed: without the shared secret nothing can be verified.
            return not self.require_verification
        signature = header_value(headers, "x-leo-signature")
        return signature_matches(signature, hmac_hex(secret, body_bytes(payload, raw_body)))

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError, TypeError):
            # Not a JSON object: untrusted input is skipped, never fatal.
            return []
        text = data.get("text")
        if not isinstance(text, str):
            return []  # an envelope without string text is not a relay message
        sender_value = data.get("sender")
        sender = "unknown" if sender_value is None else str(sender_value)
        sender = sender[:_SENDER_LIMIT] or "unknown"
        timestamp: str | None = None
        for key in ("sent_at", "timestamp"):
            timestamp = _parse_stamp(data.get(key))
            if timestamp is not None:
                break
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=sender,
                text=text[:_TEXT_LIMIT],
                timestamp=timestamp or utcnow(),
                reply_ref=ref_or_none(data.get("reply_ref")),
                message_id=ref_or_none(data.get("message_id")),
            )
        ]
