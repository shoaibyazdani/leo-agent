"""DingTalk (钉钉) robot channel adapter.

Outbound — "custom robot" webhooks. DingTalk signs every robot webhook call:
``sign = urlsafe-base64(HMAC-SHA256(secret, f"{timestamp_ms}\\n{secret}"))``
appended to the webhook URL as ``&timestamp=<ms>&sign=<percent-encoded>``.
The webhook URL itself embeds the robot access token, so the URL — and httpx
exception text, which would repeat it — is never echoed into an error.

Inbound — HTTP callback mode. DingTalk's newer robot platform defaults to
*stream mode* delivery; this adapter implements the documented HTTP callback
mode, where each POST carries ``timestamp`` and ``sign`` headers with
``sign = base64(HMAC-SHA256(DINGTALK_CALLBACK_SECRET, f"{timestamp}\\n{DINGTALK_CALLBACK_SECRET}"))``,
compared in constant time. The callback body is untrusted data: every field is
type-checked, malformed entries are skipped and nothing crashes the handler.

Credentials are referenced by environment-variable *name* and read at call
time; no token value is ever stored in an object, a record or an error.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from ..messaging import (
    HeaderMap,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    from_epoch,
    header_value,
    payload_dict,
    ref_or_none,
)
from ..secrets import credential


def _robot_sign(secret: str, timestamp_ms: str) -> str:
    """Custom-robot webhook sign: urlsafe base64 of HMAC-SHA256(secret, ts+"\\n"+secret)."""
    message = f"{timestamp_ms}\n{secret}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8")


def _callback_sign(secret: str, timestamp: str) -> str:
    """HTTP callback sign: standard base64 of HMAC-SHA256(secret, ts+"\\n"+secret)."""
    message = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


class DingTalkAdapter(MessagingAdapter):
    """Signed DingTalk custom-robot sender plus robot callback normalizer."""

    channel = "dingtalk"
    credential_env = ("DINGTALK_WEBHOOK_URL", "DINGTALK_SECRET")
    verification_env = ("DINGTALK_CALLBACK_SECRET",)
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://oapi.dingtalk.com"

    @classmethod
    def _valid_robot_url(cls, url: str) -> bool:
        """Only official DingTalk robot-send endpoints may be POSTed."""
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        host = (parsed.hostname or "").lower()
        return (
            parsed.scheme == "https"
            and (host == "oapi.dingtalk.com" or host.endswith(".dingtalk.com"))
            and "/robot/send" in parsed.path
            and not parsed.username
            and not parsed.password
        )

    def verify_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> bool:
        secret = credential("DINGTALK_CALLBACK_SECRET")
        if not secret:
            # Fail closed: an unsigned callback must never be accepted just
            # because verification was never configured.
            return not self.require_verification
        timestamp = header_value(headers, "timestamp")
        sign = header_value(headers, "sign")
        if not timestamp or not sign:
            return False
        # DingTalk signs the timestamp (not the body) with the shared secret.
        return hmac.compare_digest(sign.strip(), _callback_sign(secret, timestamp))

    async def send(self, target: str, text: str) -> dict[str, Any]:
        webhook_url = credential("DINGTALK_WEBHOOK_URL")
        if not webhook_url:
            raise MessagingError("dingtalk send blocked: DINGTALK_WEBHOOK_URL is not set")
        secret = credential("DINGTALK_SECRET")
        if not secret:
            raise MessagingError("dingtalk send blocked: DINGTALK_SECRET is not set")
        if not self._valid_robot_url(webhook_url):
            raise MessagingError("dingtalk send blocked: not a DingTalk robot webhook URL")
        timestamp_ms = str(int(time.time() * 1000))
        sign = _robot_sign(secret, timestamp_ms)
        url = f"{webhook_url}&timestamp={timestamp_ms}&sign={quote(sign, safe='')}"
        try:
            response = await self._http().post(
                url,
                json={"msgtype": "text", "text": {"content": str(text)}},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # The URL embeds the robot access token: never echo httpx's text.
            raise MessagingError(f"dingtalk send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"dingtalk send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise MessagingError("dingtalk send failed: invalid response") from None
        errcode = data.get("errcode") if isinstance(data, dict) else None
        try:
            ok = isinstance(data, dict) and int(errcode) == 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise MessagingError("dingtalk send failed: API reported failure")
        return {"channel": self.channel, "target": str(target), "success": True, "message_id": None}

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        self._check_verification(payload, headers, raw_body)
        try:
            data = payload_dict(payload)
        except (MessagingError, ValueError):
            # Untrusted body: non-object or non-JSON payloads normalize to nothing.
            return []
        text_field = data.get("text")
        if not isinstance(text_field, dict) or text_field.get("content") is None:
            return []
        create_time = data.get("createTime")
        try:
            seconds = float(create_time) / 1000.0  # DingTalk reports milliseconds
        except (TypeError, ValueError):
            seconds = None
        sender = data.get("senderNick") or data.get("senderStaffId")
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=str(sender or "unknown"),
                text=str(text_field["content"]),
                timestamp=from_epoch(seconds),
                reply_ref=None,
                message_id=ref_or_none(data.get("msgId")),
            )
        ]
