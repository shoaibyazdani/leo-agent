"""Slack channel adapter: bot-token sending plus signed Events-API ingress.

Secrets are read from the environment by *name* at call time; no token value
is ever stored on the adapter, logged, or embedded in an error message. All
network access flows through the injectable ``httpx.AsyncClient`` supplied by
``MessagingAdapter._http``, so this module is offline-verifiable with
``httpx.MockTransport``. The constructed adapter is not proof of a working
integration: ``record()`` stays ``blocked`` until ``SLACK_BOT_TOKEN`` exists.
"""

from __future__ import annotations

import hmac
import time
from typing import Any, Mapping

import httpx

from ..messaging import (
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    body_bytes,
    from_epoch,
    header_value,
    hmac_hex,
    payload_dict,
    ref_or_none,
)
from ..secrets import credential


class SlackAdapter(MessagingAdapter):
    """Slack ``chat.postMessage`` sender plus Events-API webhook normalizer.

    Verification: Slack signs every Events-API request with
    ``X-Slack-Signature = "v0=" + hex(HMAC-SHA256("v0:{X-Slack-Request-Timestamp}:{raw_body}"))``
    using ``SLACK_SIGNING_SECRET``. Both headers must be present, the digest is
    compared with ``hmac.compare_digest`` after stripping the ``v0=`` prefix,
    and requests whose timestamp is more than ``REPLAY_WINDOW`` seconds away
    from ``time.time()`` are rejected. Slack's own guidance is to refuse
    events older than five minutes; the same tolerance is applied to future
    timestamps so modest clock skew between Slack and this host cannot drop
    legitimate events (and non-numeric or non-finite timestamps are refused
    outright). If ``SLACK_SIGNING_SECRET`` is not set the adapter fails closed:
    no webhook is ever accepted just because verification was never
    configured.

    Known limitation (documented honestly): Slack's one-time
    ``url_verification`` handshake expects the raw ``challenge`` value echoed
    back in the HTTP response body. ``handle_webhook`` can only return
    normalized messages, so it returns ``[]`` for that envelope and the
    hosting webhook route must answer the handshake itself when the Events-API
    URL is first registered. Until such a route exists, event delivery cannot
    start — this adapter deliberately does not pretend otherwise.
    """

    channel = "slack"
    credential_env = ("SLACK_BOT_TOKEN",)
    verification_env = ("SLACK_SIGNING_SECRET",)
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://slack.com/api"
    REPLAY_WINDOW = 300.0  # seconds; Slack's documented five-minute freshness bound

    def verify_webhook(
        self,
        payload: Any,
        headers: Mapping[str, str] | Mapping[str, bytes] | None = None,
        *,
        raw_body: bytes | None = None,
    ) -> bool:
        secret = credential("SLACK_SIGNING_SECRET")
        if not secret:
            # Fail closed: an unsigned webhook must never be accepted just
            # because verification was never configured.
            return not self.require_verification
        signature = header_value(headers, "x-slack-signature")
        timestamp = header_value(headers, "x-slack-request-timestamp")
        if not signature or not timestamp:
            return False
        try:
            age = time.time() - float(timestamp)
        except (TypeError, ValueError):
            return False
        # ``not <=`` instead of ``>`` so NaN timestamps are refused too.
        if not abs(age) <= self.REPLAY_WINDOW:
            return False
        basestring = b"v0:" + timestamp.encode("utf-8") + b":" + body_bytes(payload, raw_body)
        expected = hmac_hex(secret, basestring)
        provided = signature.strip()
        if provided.startswith("v0="):
            provided = provided[len("v0=") :]
        return hmac.compare_digest(provided, expected)

    async def send(self, target: str, text: str) -> dict[str, Any]:
        token = credential("SLACK_BOT_TOKEN")
        if not token:
            raise MessagingError("slack send blocked: SLACK_BOT_TOKEN is not set")
        try:
            response = await self._http().post(
                f"{self.API_BASE}/chat.postMessage",
                headers={"Authorization": f"Bearer {token}"},
                json={"channel": str(target), "text": str(text)},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # httpx's message can embed request URLs: never let it surface.
            raise MessagingError(f"slack send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"slack send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("slack send failed: invalid response") from exc
        if not isinstance(data, dict) or not data.get("ok"):
            # Slack answers HTTP 200 with ok=false; the body's "error" field
            # is provider data and must never reach an exception message.
            raise MessagingError("slack send failed: API reported failure")
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": ref_or_none(data.get("ts")),
        }

    async def handle_webhook(
        self,
        payload: Any,
        headers: Mapping[str, str] | Mapping[str, bytes] | None = None,
        *,
        raw_body: bytes | None = None,
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError):
            # Untrusted input: a body that is not a JSON object yields no
            # messages rather than an exception.
            return []
        if data.get("type") == "url_verification":
            # One-time setup handshake; see the class docstring limitation.
            return []
        if data.get("type") != "event_callback":
            return []
        event = data.get("event")
        if not isinstance(event, dict) or event.get("type") != "message":
            return []
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            # Bot output (including this bot's own replies) must never loop
            # back through the gateway as a new inbound message.
            return []
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        # thread_ts equals ts on a thread root; only genuine replies carry a
        # different parent reference.
        reply_ref = ref_or_none(thread_ts) if thread_ts is not None and thread_ts != ts else None
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=str(event.get("user") or "unknown"),
                text=str(event.get("text") or ""),
                timestamp=from_epoch(ts),
                reply_ref=reply_ref,
                message_id=ref_or_none(event.get("client_msg_id") or event.get("event_ts")),
            )
        ]
