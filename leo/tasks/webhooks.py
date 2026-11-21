"""Signed webhook intake with idempotent delivery handling.

The HMAC secret is only ever read from the environment by *name*. Its value is
never stored in settings, task metadata, logs, events or the database.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Mapping

from .. import db
from ..contracts import TaskRecord, new_id, utcnow
from ..events import EventBus

_SIGNATURE_HEADERS = (
    "x-hub-signature-256",
    "x-leo-signature",
    "x-signature-256",
    "x-signature",
    "x-webhook-signature",
    "x-hook-signature",
)
# Native scheme: the MAC binds send time, delivery identity and the exact
# payload bytes, so a captured request cannot be re-labelled or refreshed.
_V1_SIGNATURE_HEADERS = (
    "x-leo-signature-v1",
    "x-webhook-signature-v1",
    "x-signature-v1",
)
_DELIVERY_HEADERS = ("x-delivery-id", "x-github-delivery", "x-webhook-id", "x-event-id")
_TIMESTAMP_WINDOW_SECONDS = 300


class WebhookError(Exception):
    """Base class for rejected webhook deliveries."""


class MissingSignature(WebhookError):
    pass


class InvalidSignature(WebhookError):
    pass


class MissingDeliveryId(WebhookError):
    pass


class DuplicateDelivery(WebhookError):
    """The delivery ID was already processed (safe replay rejection)."""


def header(headers: Mapping[str, Any] | None, name: str) -> str | None:
    if not headers:
        return None
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            return str(value) if value is not None else None
    return None


def body_bytes(body: Any) -> bytes:
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, Mapping):
        # Canonical JSON so callers signing a dict can verify deterministically.
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


async def extract_request(request: Any) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Accept a dict, an object with body/headers, or an ASGI/FastAPI-like request."""
    if isinstance(request, (bytes, bytearray, str)):
        raw, headers, payload = request, {}, {}
    elif isinstance(request, Mapping) and not any(
        key in request for key in ("body", "data", "payload", "headers", "json")
    ):
        # The mapping itself is the webhook payload.
        raw, headers, payload = request, {}, {}
    elif isinstance(request, Mapping):
        raw = request.get("body", request.get("data", request.get("payload")))
        headers = request.get("headers") or {}
        payload = request.get("json") or {}
    else:
        headers = getattr(request, "headers", {}) or {}
        raw = getattr(request, "body", None)
        if callable(raw):
            raw = await raw()
        if raw is None:
            json_method = getattr(request, "json", None)
            payload = await json_method() if callable(json_method) else (json_method or {})
            raw = payload
        else:
            payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    if not payload and isinstance(raw, (bytes, bytearray, str)):
        # Recover a JSON payload from the raw body for objective/delivery hints.
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, Mapping):
            payload = decoded
    return body_bytes(raw), dict(headers), dict(payload)


class WebhookHandler:
    """Verify HMAC signatures and create one deduplicated task per delivery."""

    def __init__(
        self,
        engine: Any,
        events: EventBus | None = None,
        *,
        secret_env: str = "LEO_WEBHOOK_SECRET",
        prefix: str = "webhook",
    ) -> None:
        self.engine = engine
        self.events = events
        self.secret_env = secret_env  # Name only; the secret value is never retained.
        self.prefix = prefix

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify(self, body: Any, headers: Mapping[str, Any]) -> bool:
        """Verify a legacy body-only HMAC signature.

        Kept for provider compatibility (GitHub/WhatsApp style): the MAC
        covers only the payload bytes, so replay defence rests entirely on
        the delivery dedupe. Callers able to use the v1 scheme should.
        """
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise WebhookError(f"webhook secret environment variable {self.secret_env} is not set")
        signature = None
        for candidate in _SIGNATURE_HEADERS:
            signature = header(headers, candidate)
            if signature:
                break
        if not signature:
            raise MissingSignature("webhook is missing an HMAC signature header")
        signature = signature.strip()
        if "=" in signature:
            signature = signature.split("=", 1)[1]
        digest = hmac.new(secret.encode("utf-8"), body_bytes(body), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature.lower(), digest):
            raise InvalidSignature("webhook HMAC signature does not match")
        return True

    @staticmethod
    def _v1_mac_input(timestamp: str, delivery_id: str, body: bytes) -> bytes:
        """Authenticated binding of send time, delivery identity and payload."""
        return b"t=" + timestamp.encode("utf-8") + b",d=" + delivery_id.encode("utf-8") + b"," + body

    @classmethod
    def sign_v1(
        cls,
        body: Any,
        secret: str,
        *,
        delivery_id: str,
        timestamp: int | None = None,
    ) -> dict[str, str]:
        """Helper for producers: headers authenticating a whole delivery.

        The returned headers bind the send time (unix seconds), the delivery
        ID and the exact payload bytes into one HMAC; altering any of the
        three invalidates the signature.
        """
        sent = str(int(timestamp if timestamp is not None else time.time()))
        digest = hmac.new(
            secret.encode("utf-8"),
            cls._v1_mac_input(sent, delivery_id, body_bytes(body)),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-LEO-Signature-V1": f"v1={digest}",
            "X-Delivery-ID": delivery_id,
            "X-Webhook-Request-Timestamp": sent,
        }

    def _verify_v1(self, body: bytes, headers: Mapping[str, Any], delivery_id: str) -> bool:
        """Verify the native v1 scheme, or return False if it is not in use.

        A v1 signature is mandatory-timestamp and binds the delivery ID, so
        captured requests cannot be replayed with a refreshed timestamp or a
        forged delivery identity.
        """
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise WebhookError(f"webhook secret environment variable {self.secret_env} is not set")
        signature = None
        for candidate in _V1_SIGNATURE_HEADERS:
            signature = header(headers, candidate)
            if signature:
                break
        if not signature:
            return False  # no v1 header present; caller falls back to legacy
        raw_ts = header(headers, "x-webhook-request-timestamp")
        if raw_ts is None:
            raise WebhookError("v1 webhook signature requires X-Webhook-Request-Timestamp")
        try:
            sent = int(str(raw_ts))
        except (TypeError, ValueError):
            raise WebhookError("invalid webhook timestamp") from None
        if abs(time.time() - sent) > _TIMESTAMP_WINDOW_SECONDS:
            raise WebhookError("webhook timestamp outside the allowed window")
        signature = signature.strip()
        if "=" in signature:
            signature = signature.split("=", 1)[1]
        digest = hmac.new(
            secret.encode("utf-8"), self._v1_mac_input(str(sent), delivery_id, body), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature.lower(), digest):
            raise InvalidSignature(
                "v1 webhook signature does not match timestamp, delivery ID and body"
            )
        return True

    @staticmethod
    def _check_timestamp(headers: Mapping[str, Any]) -> None:
        """Reject obviously stale deliveries when a timestamp is supplied."""
        raw = (
            header(headers, "x-webhook-request-timestamp")
            or header(headers, "x-github-request-timestamp")
            or header(headers, "x-request-timestamp")
        )
        if raw is None:
            return
        try:
            sent = int(str(raw))
        except (TypeError, ValueError):
            raise WebhookError("invalid webhook timestamp")
        if abs(time.time() - sent) > 300:
            raise WebhookError("webhook timestamp outside the allowed window")

    @staticmethod
    def signature(body: Any, secret: str) -> str:
        """Helper for producers/tests: return a ``sha256=<hex>`` signature."""
        digest = hmac.new(secret.encode("utf-8"), body_bytes(body), hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------
    async def handle(
        self, request: Any, extra_headers: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        body, headers, parsed = await extract_request(request)
        if extra_headers:
            headers.update(dict(extra_headers))
        if not any(
            header(headers, name)
            for name in _SIGNATURE_HEADERS + _V1_SIGNATURE_HEADERS
        ):
            for name in ("signature", "x-signature", "sig"):
                if parsed.get(name):
                    headers["X-Signature"] = str(parsed[name])
                    break

        delivery_id = None
        for candidate in _DELIVERY_HEADERS:
            delivery_id = header(headers, candidate)
            if delivery_id:
                break
        if not delivery_id:
            delivery_id = str(parsed.get("delivery_id") or parsed.get("id") or "") or None

        # Verification happens before the delivery-ID requirement so a
        # misconfigured (secret-less) or unsigned delivery always fails with
        # the verification error, never with a routing error.
        v1 = self._verify_v1(body, headers, delivery_id or "")
        if not v1:
            # Legacy provider signatures cover only the body; the timestamp
            # window and the dedupe below are the remaining replay defences.
            self.verify(body, headers)
            self._check_timestamp(headers)

        if not delivery_id:
            raise MissingDeliveryId("webhook is missing a delivery ID")
        if "\x00" in delivery_id or len(delivery_id) > 512:
            raise WebhookError("invalid delivery ID")

        # Dedupe fingerprint: under v1 the delivery identity is authenticated,
        # so distinct signed deliveries of an identical body are distinct
        # tasks. Under the legacy scheme the ID header is unsigned and the
        # payload bytes alone define identity.
        if v1:
            fingerprint = hashlib.sha256(
                b"d=" + delivery_id.encode("utf-8") + b"," + body
            ).hexdigest()
        else:
            fingerprint = hashlib.sha256(body).hexdigest()
        key = f"{self.prefix}:delivery:{fingerprint}"
        claim = new_id("wh")
        await db.execute(
            "INSERT OR IGNORE INTO kv(key,value,updated_at) VALUES(?,?,?)",
            (key, claim, utcnow()),
        )
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
        if row is not None and row["value"] != claim:
            if self.events is not None:
                await self.events.emit(
                    "webhook.rejected", delivery_id=delivery_id, reason="duplicate delivery"
                )
            raise DuplicateDelivery(f"webhook delivery already processed: {delivery_id}")

        objective = str(parsed.get("objective") or header(headers, "x-webhook-event") or f"{self.prefix} delivery {delivery_id}")
        metadata: dict[str, Any] = {
            "webhook": True,
            "delivery_id": delivery_id,
            "event": header(headers, "x-webhook-event") or parsed.get("event"),
        }
        try:
            task: TaskRecord = await self.engine.create(
                objective,
                metadata=metadata,
                idempotency_key=f"{self.prefix}:{delivery_id}",
            )
        except BaseException:
            # Allow a redelivery after a genuine processing failure.
            await db.execute("DELETE FROM kv WHERE key=? AND value=?", (key, claim))
            raise
        # Only the claim holder may retarget the dedupe key.
        await db.execute(
            "UPDATE kv SET value=? WHERE key=? AND value=?", (task.id, key, claim)
        )
        if self.events is not None:
            await self.events.emit(
                "webhook.accepted", delivery_id=delivery_id, task_id=task.id, prefix=self.prefix
            )
        return {"status": "accepted", "delivery_id": delivery_id, "task_id": task.id, "task": task}

    async def __call__(self, request: Any) -> dict[str, Any]:
        return await self.handle(request)
