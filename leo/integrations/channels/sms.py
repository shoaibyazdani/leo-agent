"""SMS channel adapter — Twilio-compatible Messaging API.

Outbound targets Twilio's documented REST API: ``POST
{TWILIO_BASE_URL}/2010-04-01/Accounts/{AccountSid}/Messages.json`` with
HTTP Basic auth (AccountSid, AuthToken) and a form-encoded body
(``From``, ``To``, ``Body``). ``TWILIO_BASE_URL`` (default
``https://api.twilio.com``) lets any Twilio-compatible provider be used
without code changes.

Inbound webhooks use Twilio's documented ``X-Twilio-Signature`` header:
base64(HMAC-SHA1(auth_token, full_callback_url + params sorted by key
and concatenated as ``key`` + ``value``)). The *same*
``TWILIO_AUTH_TOKEN`` used for outbound Basic auth signs inbound
requests — which is why it appears in both ``credential_env`` and
``verification_env``. Because the signature covers the full public
callback URL, LEO must know that exact URL: set
``TWILIO_WEBHOOK_URL`` to the URL Twilio calls. Without it (or without
the token) verification fails closed — an unsigned webhook is never
accepted just because verification was never configured.

Twilio only delivers webhooks over HTTPS; LEO relies on that platform
guarantee and does not re-check the scheme itself — the signature is
the actual control.

All network access goes through the injectable ``httpx.AsyncClient``.
Credentials are read from the environment by *name* at call time and
never appear in errors, records or logs (the Account SID is part of the
request URL, so URLs are never echoed either). Inbound fields are
untrusted data: type-checked, coerced, never executed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any, Mapping
from urllib.parse import parse_qsl

import httpx

from ...contracts import IntegrationRecord, utcnow
from ..messaging import (
    HeaderMap,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    header_value,
    ref_or_none,
)
from ..secrets import credential


def _form_params(payload: Any, raw_body: bytes | None) -> list[tuple[str, str]]:
    """The posted form parameters as text pairs — the exact set Twilio signs.

    Byte-exact from ``raw_body`` when the framework has it; otherwise a
    mapping payload's items, or a form-encoded str/bytes payload. Blank
    values are kept: Twilio signs every posted parameter.
    """
    if raw_body is not None:
        text = raw_body if isinstance(raw_body, str) else None
        if text is None:
            try:
                text = bytes(raw_body).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 - untrusted bytes must not crash verification
                return []
        return parse_qsl(text, keep_blank_values=True)
    if isinstance(payload, Mapping):
        return [(str(key), "" if value is None else str(value)) for key, value in payload.items()]
    if isinstance(payload, (bytes, bytearray)):
        return parse_qsl(bytes(payload).decode("utf-8", "replace"), keep_blank_values=True)
    if isinstance(payload, str):
        return parse_qsl(payload, keep_blank_values=True)
    return []


class SMSAdapter(MessagingAdapter):
    """Twilio-compatible SMS sender plus signed webhook normalization."""

    channel = "sms"
    credential_env = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")
    # The auth token doubles as the inbound signing secret.
    verification_env = ("TWILIO_AUTH_TOKEN",)
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://api.twilio.com"

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        token = credential("TWILIO_AUTH_TOKEN")
        if not token:
            # Fail closed: without the signing token nothing can be verified.
            return not self.require_verification
        url = credential("TWILIO_WEBHOOK_URL")
        if not url:
            # The signature covers the full public callback URL; without it
            # the expected signature cannot even be constructed. Fail closed.
            return not self.require_verification
        signature = header_value(headers, "x-twilio-signature")
        if not signature:
            return False
        items = _form_params(payload, raw_body)
        signed = url + "".join(f"{key}{value}" for key, value in sorted(items))
        expected = base64.b64encode(
            hmac.new(token.encode("utf-8"), signed.encode("utf-8"), hashlib.sha1).digest()
        ).decode("ascii")
        try:
            return hmac.compare_digest(signature.strip(), expected)
        except TypeError:
            # compare_digest rejects non-ASCII strings; a hostile header is
            # simply a non-match, never a crash.
            return False

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        params: dict[str, str] = {}
        for key, value in _form_params(payload, raw_body):
            params[key] = value
        # UNTRUSTED DATA: From/To/Body/MessageSid/FromCity etc. are plain
        # form fields. Anything that is not recognizably an inbound SMS
        # payload is skipped rather than guessed at.
        if not any(name in params for name in ("From", "Body", "MessageSid")):
            return []
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=str(params.get("From") or "unknown"),
                text=str(params.get("Body") or ""),
                timestamp=utcnow(),
                reply_ref=None,
                message_id=ref_or_none(params.get("MessageSid")),
            )
        ]

    async def send(self, target: str, text: str) -> dict[str, Any]:
        sid = credential("TWILIO_ACCOUNT_SID")
        token = credential("TWILIO_AUTH_TOKEN")
        from_number = credential("TWILIO_FROM_NUMBER")
        missing = next(
            (
                name
                for name, value in (
                    ("TWILIO_ACCOUNT_SID", sid),
                    ("TWILIO_AUTH_TOKEN", token),
                    ("TWILIO_FROM_NUMBER", from_number),
                )
                if not value
            ),
            None,
        )
        if missing:
            raise MessagingError(f"sms send blocked: {missing} is not set")
        base = credential("TWILIO_BASE_URL", self.API_BASE).rstrip("/")
        # The Account SID is part of the URL: never echo URLs or httpx's
        # exception text in errors.
        url = f"{base}/2010-04-01/Accounts/{sid}/Messages.json"
        try:
            response = await self._http().post(
                url,
                auth=(str(sid), str(token)),
                data={"From": str(from_number), "To": str(target), "Body": str(text)},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"sms send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"sms send failed: HTTP {response.status_code}")
        message_id: str | None = None
        if response.content:
            try:
                data = response.json()
                if isinstance(data, dict):
                    message_id = ref_or_none(data.get("sid"))
            except ValueError:
                message_id = None
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": message_id,
        }

    def record(self) -> IntegrationRecord:
        record = super().record()
        # The auth token is both an outbound credential and the inbound
        # signature secret; list each env NAME once in config_ref.
        names: list[str] = []
        for name in self.credential_env + self.verification_env:
            if name not in names:
                names.append(name)
        record.config_ref = ",".join(names)
        # Webhook verification additionally needs the public callback URL;
        # surface that requirement without affecting the status flip.
        record.metadata["webhook_url_env"] = "TWILIO_WEBHOOK_URL"
        record.metadata["webhook_url_present"] = bool(credential("TWILIO_WEBHOOK_URL"))
        return record


__all__ = ["SMSAdapter"]
