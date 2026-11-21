"""Mattermost channel adapter: webhook/bot-token sending plus outgoing-webhook ingress.

Secrets are read from the environment by *name* at call time; no token value
is ever stored on the adapter, logged, or embedded in an error message. All
network access flows through the injectable ``httpx.AsyncClient`` supplied by
``MessagingAdapter._http``, so this module is offline-verifiable with
``httpx.MockTransport``. The constructed adapter is not proof of a working
integration: ``record()`` stays ``blocked`` until both ``MATTERMOST_TOKEN``
and ``MATTERMOST_URL`` exist (per-call incoming-webhook targets are invisible
to that record and work without any environment credentials).
"""

from __future__ import annotations

import hmac
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import httpx

from ..messaging import (
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    from_epoch,
    ref_or_none,
)
from ..secrets import credential


class MattermostAdapter(MessagingAdapter):
    """Mattermost sender (incoming webhook or bot token) plus outgoing-webhook ingress.

    Verification: Mattermost's documented outgoing-webhook check is a *shared
    token*, not an HMAC signature. The server POSTs a form-encoded body whose
    ``token`` field holds the token configured on the outgoing webhook; this
    adapter parses that body (``urllib.parse.parse_qs`` over the raw bytes, or
    the already-parsed dict when no raw body is available) and compares it
    with ``hmac.compare_digest`` against ``MATTERMOST_OUTGOING_TOKEN``. A
    missing or empty token field fails verification, and if the environment
    variable is absent the adapter fails closed. Because the shared token
    travels inside the request body, TLS is the only confidentiality
    boundary — deploy behind HTTPS, exactly as Mattermost's own webhook
    documentation requires.

    Sending: a target starting with ``https://`` is treated as a Mattermost
    incoming-webhook URL and is validated (HTTPS scheme and a path starting
    with ``/hooks/``) before any network access — anything else is refused so
    a caller cannot repoint a webhook send at an arbitrary endpoint. Any
    other target is a channel ID posted through the official REST API v4
    (``{MATTERMOST_URL}/api/v4/posts``) with the bot token in the
    ``Authorization: Bearer`` header.
    """

    channel = "mattermost"
    credential_env = ("MATTERMOST_TOKEN", "MATTERMOST_URL")
    verification_env = ("MATTERMOST_OUTGOING_TOKEN",)
    supports_webhook = True
    supports_polling = False

    @staticmethod
    def _form_fields(payload: Any, raw_body: bytes | str | None = None) -> dict[str, Any]:
        """Outgoing-webhook fields from the raw form body, or a dict payload.

        Values coming from ``parse_qs`` are always strings; values taken
        from an already-parsed dict payload keep their original type so the
        adapter's defensive ``or``-fallbacks — not an eager ``str()`` —
        decide what is missing or hostile.
        """
        source: str
        if raw_body is not None:
            source = (
                raw_body.decode("utf-8", "replace")
                if isinstance(raw_body, (bytes, bytearray))
                else str(raw_body)
            )
        elif isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()}
        elif isinstance(payload, (bytes, bytearray)):
            source = bytes(payload).decode("utf-8", "replace")
        elif isinstance(payload, str):
            source = payload
        else:
            return {}
        fields = parse_qs(source, keep_blank_values=True)
        return {name: values[0] for name, values in fields.items() if values}

    @classmethod
    def _webhook_url(cls, url: str) -> bool:
        """True only for Mattermost incoming-webhook URLs (https + ``/hooks/`` path)."""
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        return parsed.scheme == "https" and (parsed.path or "").startswith("/hooks/")

    def verify_webhook(
        self,
        payload: Any,
        headers: Mapping[str, str] | Mapping[str, bytes] | None = None,
        *,
        raw_body: bytes | None = None,
    ) -> bool:
        expected = credential("MATTERMOST_OUTGOING_TOKEN")
        if not expected:
            # Fail closed: an unsigned webhook must never be accepted just
            # because verification was never configured.
            return not self.require_verification
        token = self._form_fields(payload, raw_body).get("token")
        if not token:
            return False
        return hmac.compare_digest(str(token), expected)

    async def send(self, target: str, text: str) -> dict[str, Any]:
        target_str = str(target)
        try:
            if target_str.startswith("https://"):
                if not self._webhook_url(target_str):
                    raise MessagingError("mattermost send blocked: not a Mattermost webhook URL")
                response = await self._http().post(target_str, json={"text": str(text)}, timeout=self.timeout)
            else:
                token = credential("MATTERMOST_TOKEN")
                if not token:
                    raise MessagingError("mattermost send blocked: MATTERMOST_TOKEN is not set")
                base_url = credential("MATTERMOST_URL")
                if not base_url:
                    raise MessagingError("mattermost send blocked: MATTERMOST_URL is not set")
                response = await self._http().post(
                    f"{base_url.rstrip('/')}/api/v4/posts",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel_id": target_str, "message": str(text)},
                    timeout=self.timeout,
                )
        except MessagingError:
            raise
        except httpx.HTTPError as exc:
            # httpx's message can embed request URLs: never let it surface.
            raise MessagingError(f"mattermost send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"mattermost send failed: HTTP {response.status_code}")
        message_id: str | None = None
        if response.content:
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                # Incoming webhooks answer plain-text "ok" (no JSON id);
                # REST posts answer 201 with the created post object.
                message_id = ref_or_none(data.get("id"))
        return {
            "channel": self.channel,
            "target": target_str,
            "success": True,
            "message_id": message_id,
        }

    async def handle_webhook(
        self,
        payload: Any,
        headers: Mapping[str, str] | Mapping[str, bytes] | None = None,
        *,
        raw_body: bytes | None = None,
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        fields = self._form_fields(payload, raw_body)
        text = str(fields.get("text") or "")
        if not text:
            return []
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=str(fields.get("user_name") or "unknown"),
                text=text,
                # Mattermost sends a unix timestamp; from_epoch falls back
                # to "now" when the field is absent or unparsable, so
                # ingress never crashes on hostile input.
                timestamp=from_epoch(fields.get("timestamp")),
                reply_ref=None,
                message_id=ref_or_none(fields.get("post_id")),
            )
        ]
