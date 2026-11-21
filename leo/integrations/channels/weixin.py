"""WeChat Official Account (微信公众平台) messaging adapter.

Outbound uses the documented platform API at ``https://api.weixin.qq.com``:
``GET /cgi-bin/token`` with ``grant_type=client_credential``, then ``POST
/cgi-bin/message/custom/send`` (the customer-service message API) with a text
message to an openid. The access token is cached in memory per adapter
instance with an injectable ``now_fn`` clock.

Inbound: official-account callbacks POST an XML envelope whose useful fields
are ``<Encrypt>``, ``<MsgSignature>``, ``<TimeStamp>`` and ``<Nonce>``, and
sign the request with the shared ``msg_signature`` scheme — ``SHA1`` over the
sorted ``[token, timestamp, nonce, encrypt]`` — carried in the URL **query**,
not in headers. This adapter parses that native XML envelope itself, and also
accepts the query fields from headers or from the JSON-mode normalized payload:

    {"encrypt": "<b64>", "msg_signature": "<sha1 hex>",
     "timestamp": "<str>", "nonce": "<str>"}

:func:`normalize_wechat_callback` performs that merge for both adapters, so
the bundled JSON-only webhook route works with an unmodified XML callback.
``verify_webhook_xml`` still takes the four raw values directly for operators
holding the unmodified request. Verified deliveries are additionally subject
to a timestamp-freshness / nonce replay guard.

The decrypted plaintext is the inner message XML
(``<xml><FromUserName/>...<MsgType>text</MsgType><Content/>...</xml>``),
parsed with stdlib ``xml.etree.ElementTree``. Modern stdlib ElementTree does
not resolve external entities, but payloads are still treated strictly as
untrusted data: unknown tags are ignored, every field is type-checked and
malformed XML yields ``[]`` without raising. Only ``MsgType`` ``text`` is
normalized; events and media callbacks are skipped.

Honest limitations:
- ``cryptography`` is optional: without it inbound decryption fails closed
  with :class:`~leo.integrations.messaging.MessagingError` unless a custom
  ``decryptor`` is injected.
- Customer-service sending requires the account to hold the customer-service
  interface permission; the API's IP allowlist must include this host.

Secrets are read from the environment by *name* at call time and never
appear in instances' records, logs or error messages.
"""

from __future__ import annotations

import hmac
import json
import time
import xml.etree.ElementTree as ElementTree
from typing import Any, Callable

import httpx

from ..messaging import (
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    body_bytes,
    from_epoch,
    header_value,
    ref_or_none,
)
from ..secrets import credential
from ._wechat_crypto import WeChatCryptoError, decrypt_wechat_payload, wechat_msg_signature
from .feishu import SignedWebhookReplayGuard

__all__ = ["WeChatAdapter", "normalize_wechat_callback", "parse_wechat_plaintext"]

Decryptor = Callable[[str, str, str], str]

#: Callback field aliases: native XML tags, JSON-mode keys and query-header names.
_ENCRYPT_KEYS = ("encrypt", "Encrypt")
_SIGNATURE_KEYS = ("msg_signature", "MsgSignature", "x-msg-signature", "x-wechat-msg-signature")
_TIMESTAMP_KEYS = ("timestamp", "TimeStamp", "x-timestamp")
_NONCE_KEYS = ("nonce", "Nonce", "x-nonce")


def _xml_fields(raw: Any) -> dict[str, str]:
    """Extract the four encrypted-callback fields from a native XML envelope."""
    if not isinstance(raw, (str, bytes, bytearray)) or not raw:
        return {}
    try:
        root = ElementTree.fromstring(raw)
    except (ElementTree.ParseError, ValueError, TypeError):
        return {}
    fields: dict[str, str] = {}
    for child in root:
        if isinstance(child.tag, str) and isinstance(child.text, str):
            fields[child.tag.lower()] = child.text.strip()
    return {
        "encrypt": fields.get("encrypt", ""),
        "msg_signature": fields.get("msgsignature", ""),
        "timestamp": fields.get("timestamp", ""),
        "nonce": fields.get("nonce", ""),
    }


def normalize_wechat_callback(
    payload: Any, headers: Any = None, *, raw_body: bytes | None = None
) -> dict[str, Any]:
    """Return the four signed callback values from XML, JSON or query headers.

    The bundled JSON-only route passes headers and a parsed body; a raw XML
    body may be delivered as ``payload`` (str/bytes) or as ``raw_body``. Query
    fields merged by a fronting shim can arrive under their native names in
    headers. Header values win over body values: the query is authoritative
    for the signature. Untrusted fields are coerced to text only when they
    are strings; a malformed body yields an empty dict.
    """
    data: dict[str, Any] = {}
    if isinstance(payload, dict):
        data = dict(payload)
    elif isinstance(payload, (str, bytes, bytearray)):
        data = _xml_fields(payload)
    if not data and raw_body is not None:
        data = _xml_fields(raw_body)
    if headers:
        for target, names in (
            ("encrypt", _ENCRYPT_KEYS),
            ("msg_signature", _SIGNATURE_KEYS),
            ("timestamp", _TIMESTAMP_KEYS),
            ("nonce", _NONCE_KEYS),
        ):
            for name in names:
                value = header_value(headers, name)
                if value:
                    data[target] = value
                    break
    normalized: dict[str, Any] = {}
    for target, names in (
        ("encrypt", _ENCRYPT_KEYS),
        ("msg_signature", _SIGNATURE_KEYS),
        ("timestamp", _TIMESTAMP_KEYS),
        ("nonce", _NONCE_KEYS),
    ):
        for name in names:
            value = data.get(name)
            if isinstance(value, str) and value:
                normalized[target] = value
                break
    return normalized


def parse_wechat_plaintext(plaintext: str) -> Any:
    """Parse a decrypted callback body as JSON, falling back to inner XML."""
    try:
        return json.loads(plaintext)
    except (TypeError, ValueError):
        pass
    try:
        root = ElementTree.fromstring(plaintext)
    except (ElementTree.ParseError, ValueError, TypeError):
        return None
    fields: dict[str, str] = {}
    for child in root:
        if isinstance(child.tag, str) and isinstance(child.text, str):
            fields[child.tag] = child.text
    return fields or None


class WeChatAdapter(MessagingAdapter):
    """WeChat Official Account sender plus encrypted-callback normalization."""

    channel = "weixin"
    credential_env = ("WEIXIN_APP_ID", "WEIXIN_APP_SECRET")
    verification_env = ("WEIXIN_TOKEN", "WEIXIN_ENCODING_AES_KEY")
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://api.weixin.qq.com"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        require_verification: bool = True,
        signature_verifier: Any = None,
        now_fn: Callable[[], float] = time.time,
        decryptor: Decryptor | None = None,
        replay_max_age_seconds: float = SignedWebhookReplayGuard.DEFAULT_MAX_AGE_SECONDS,
    ) -> None:
        super().__init__(
            client=client,
            timeout=timeout,
            require_verification=require_verification,
            signature_verifier=signature_verifier,
        )
        self.now_fn = now_fn or time.time
        self._decryptor: Decryptor = decryptor or decrypt_wechat_payload
        self._access_token: str | None = None
        self._token_expiry: float = 0.0
        self._replay = SignedWebhookReplayGuard(
            max_age_seconds=replay_max_age_seconds, now_fn=self.now_fn
        )

    # ------------------------------------------------------------------ webhook

    @staticmethod
    def _signature_matches(token: str, encrypt: Any, signature: Any, timestamp: Any, nonce: Any) -> bool:
        if not all(isinstance(value, str) and value for value in (encrypt, signature, timestamp, nonce)):
            return False
        expected = wechat_msg_signature(token, timestamp, nonce, encrypt)
        return hmac.compare_digest(signature.strip().lower().encode("utf-8"), expected.encode("utf-8"))

    def verify_webhook(
        self, payload: Any, headers: Any = None, *, raw_body: bytes | None = None
    ) -> bool:
        """Check ``msg_signature`` over XML/JSON/header callback fields.

        Fail closed: without ``WEIXIN_TOKEN`` this returns
        ``not require_verification``; missing or non-string fields return
        ``False``. Comparison is constant-time on bytes. Freshness and replay
        are enforced on the verified ``handle_webhook`` path, not here: this
        method is also used for URL handshakes where a nonce may be reused.
        """
        token = credential("WEIXIN_TOKEN")
        if not token:
            return not self.require_verification
        data = normalize_wechat_callback(payload, headers, raw_body=raw_body)
        return self._signature_matches(
            token, data.get("encrypt"), data.get("msg_signature"), data.get("timestamp"), data.get("nonce")
        )

    def verify_webhook_xml(
        self, encrypt: str, msg_signature: str, timestamp: str, nonce: str
    ) -> bool:
        """Standalone signature check for operators holding raw callback values.

        WeChat signs the callback with the same ``msg_signature`` scheme for
        XML and JSON-mode bodies alike, so a fronting shim can verify before
        building the JSON-mode payload for :meth:`handle_webhook`. The same
        fail-closed rule applies: without ``WEIXIN_TOKEN`` this returns
        ``not require_verification``.
        """
        token = credential("WEIXIN_TOKEN")
        if not token:
            return not self.require_verification
        return self._signature_matches(token, encrypt, msg_signature, timestamp, nonce)

    async def handle_webhook(
        self, payload: Any, headers: Any = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        """Verify, freshness-check, decrypt and normalize a WeChat callback.

        Native XML envelopes, JSON-mode bodies and query headers are all
        normalized by :func:`normalize_wechat_callback`. The decrypted
        plaintext is the inner message XML, parsed defensively: unknown tags
        are ignored, non-``text`` message types and malformed XML yield ``[]``
        without raising. Decryption failures (including a receive id minted
        for another app and a missing crypto backend) fail closed as
        :class:`MessagingError`.
        """
        self._check_verification(payload, headers, raw_body)
        data = normalize_wechat_callback(payload, headers, raw_body=raw_body)
        if not all(isinstance(data.get(key), str) and data[key] for key in ("encrypt", "msg_signature", "timestamp", "nonce")):
            # No verified signature fields: nothing to trust or replay-track.
            return []
        if self.require_verification:
            self._replay.check(
                timestamp=data["timestamp"],
                nonce=data["nonce"],
                message_id=data["encrypt"],
                body=body_bytes(payload, raw_body),
                channel=self.channel,
            )
        encrypt = data["encrypt"]
        aes_key = credential("WEIXIN_ENCODING_AES_KEY")
        if not aes_key:
            raise MessagingError("weixin webhook failed: WEIXIN_ENCODING_AES_KEY is not set")
        app_id = credential("WEIXIN_APP_ID")
        if not app_id:
            raise MessagingError("weixin webhook failed: WEIXIN_APP_ID is not set")
        try:
            plaintext = self._decryptor(encrypt, aes_key, app_id)
        except WeChatCryptoError as exc:
            raise MessagingError(f"weixin webhook failed: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - never crash on untrusted input
            raise MessagingError(f"weixin webhook failed: decryptor raised {type(exc).__name__}") from None
        return self._parse_inner_xml(plaintext)

    def _parse_inner_xml(self, plaintext: str) -> list[NormalizedMessage]:
        """Extract one text message from the decrypted inner XML, defensively."""
        try:
            root = ElementTree.fromstring(plaintext)
        except (ElementTree.ParseError, ValueError):
            return []
        fields: dict[str, str] = {}
        for child in root:
            # Comments and PIs carry a non-str tag; unknown tags are simply ignored.
            if isinstance(child.tag, str) and isinstance(child.text, str):
                fields[child.tag] = child.text
        msg_type = str(fields.get("MsgType") or "").strip().lower()
        if msg_type != "text":
            return []
        sender = fields.get("FromUserName") or ""
        content = fields.get("Content")
        return [
            NormalizedMessage(
                channel=self.channel,
                sender=str(sender) if sender else "unknown",
                text=str(content) if content is not None else "",
                timestamp=from_epoch(fields.get("CreateTime")),
                reply_ref=None,
                message_id=ref_or_none(fields.get("MsgId")),
            )
        ]

    # -------------------------------------------------------------------- send

    async def _fetch_access_token(self, app_id: str, secret: str) -> str:
        """Get (or reuse) the platform access token; expiry via ``now_fn``."""
        now = float(self.now_fn())
        if self._access_token and self._token_expiry > now:
            return self._access_token
        try:
            response = await self._http().get(
                f"{self.API_BASE}/cgi-bin/token",
                params={"grant_type": "client_credential", "appid": app_id, "secret": secret},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # The URL carries the secret: never echo it or httpx's message.
            raise MessagingError(f"weixin send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"weixin send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("weixin send failed: invalid response") from exc
        token = data.get("access_token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token:
            raise MessagingError("weixin send failed: API reported failure")
        try:
            expires_in = max(0.0, float(data.get("expires_in")))
        except (TypeError, ValueError):
            expires_in = 0.0
        self._access_token = token
        self._token_expiry = now + expires_in
        return token

    async def send(self, target: str, text: str) -> dict[str, Any]:
        app_id = credential("WEIXIN_APP_ID")
        if not app_id:
            raise MessagingError("weixin send blocked: WEIXIN_APP_ID is not set")
        secret = credential("WEIXIN_APP_SECRET")
        if not secret:
            raise MessagingError("weixin send blocked: WEIXIN_APP_SECRET is not set")
        target_str = target if isinstance(target, str) else str(target)
        access_token = await self._fetch_access_token(app_id, secret)
        try:
            response = await self._http().post(
                f"{self.API_BASE}/cgi-bin/message/custom/send",
                params={"access_token": access_token},
                json={"touser": target_str, "msgtype": "text", "text": {"content": str(text)}},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"weixin send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"weixin send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("weixin send failed: invalid response") from exc
        if not isinstance(data, dict) or data.get("errcode") != 0:
            raise MessagingError("weixin send failed: API reported failure")
        return {
            "channel": self.channel,
            "target": target_str,
            "success": True,
            "message_id": None,
        }
