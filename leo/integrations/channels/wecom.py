"""WeCom (WeChat Work / 企业微信) messaging adapter.

Outbound uses the documented server API at ``https://qyapi.weixin.qq.com``:
``GET /gettoken`` with the corp id/secret, then ``POST
/cgi-bin/message/send`` with an agent-scoped text message. The access token is
cached in memory per adapter instance with an injectable ``now_fn`` clock.

Inbound uses WeCom's callback protocol. WeCom signs every callback with the
shared ``msg_signature`` scheme — ``SHA1`` over the sorted ``[token,
timestamp, nonce, encrypt]`` — and delivers the message body AES-encrypted
with the account's ``EncodingAESKey``. This adapter parses the native XML
envelope (``<Encrypt>``/``<MsgSignature>``/``<TimeStamp>``/``<Nonce>``), a
JSON-mode body, or query fields merged into headers by a fronting shim:

    {"encrypt": "<b64>", "msg_signature": "<sha1 hex>",
     "timestamp": "<str>", "nonce": "<str>"}

:func:`~.weixin.normalize_wechat_callback` performs the merge for both WeChat
adapters. Verified deliveries are also subject to timestamp freshness and
nonce/message-id replay suppression.

Honest limitations:
- The first-time GET URL handshake (answering the decrypted ``echostr``) is a
  fronting-route concern; this adapter exposes the same signature and
  decryption primitives for it.
- ``cryptography`` is optional: without it inbound decryption fails closed
  with :class:`~leo.integrations.messaging.MessagingError` unless a custom
  ``decryptor`` is injected.

Secrets are read from the environment by *name* at call time and never
appear in instances' records, logs or error messages.
"""

from __future__ import annotations

import hmac
import time
from typing import Any, Callable

import httpx

from ..messaging import (
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    body_bytes,
    from_epoch,
    ref_or_none,
)
from ..secrets import credential
from ._wechat_crypto import WeChatCryptoError, decrypt_wechat_payload, wechat_msg_signature
from .feishu import SignedWebhookReplayGuard
from .weixin import normalize_wechat_callback, parse_wechat_plaintext

__all__ = ["WeComAdapter"]

Decryptor = Callable[[str, str, str], str]


class WeComAdapter(MessagingAdapter):
    """WeCom text sender plus JSON-mode callback normalization."""

    channel = "wecom"
    credential_env = ("WECOM_CORP_ID", "WECOM_SECRET")
    verification_env = ("WECOM_CALLBACK_TOKEN", "WECOM_ENCODING_AES_KEY")
    supports_webhook = True
    supports_polling = False
    API_BASE = "https://qyapi.weixin.qq.com"

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

    def verify_webhook(
        self, payload: Any, headers: Any = None, *, raw_body: bytes | None = None
    ) -> bool:
        """Check ``msg_signature`` over normalized XML/JSON/header fields.

        Fail closed: without ``WECOM_CALLBACK_TOKEN`` this returns
        ``not require_verification``; missing or non-string fields return
        ``False``. Comparison is constant-time on bytes. Freshness and replay
        are enforced on the verified ``handle_webhook`` path.
        """
        token = credential("WECOM_CALLBACK_TOKEN")
        if not token:
            return not self.require_verification
        data = normalize_wechat_callback(payload, headers, raw_body=raw_body)
        encrypt = data.get("encrypt")
        signature = data.get("msg_signature")
        timestamp = data.get("timestamp")
        nonce = data.get("nonce")
        if not all(isinstance(value, str) and value for value in (encrypt, signature, timestamp, nonce)):
            return False
        expected = wechat_msg_signature(token, timestamp, nonce, encrypt)
        return hmac.compare_digest(signature.strip().lower().encode("utf-8"), expected.encode("utf-8"))

    async def handle_webhook(
        self, payload: Any, headers: Any = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        """Verify, freshness-check, decrypt and normalize a WeCom callback.

        Native XML envelopes, JSON-mode bodies and query headers are all
        normalized by :func:`~.weixin.normalize_wechat_callback`. The decrypted
        plaintext is parsed defensively: provider payloads are untrusted data,
        every field is type-checked, malformed entries are skipped and a
        non-dict plaintext yields ``[]``. Decryption failures (including a
        receive id minted for another account and a missing crypto backend)
        fail closed as :class:`MessagingError`.
        """
        self._check_verification(payload, headers, raw_body)
        data = normalize_wechat_callback(payload, headers, raw_body=raw_body)
        if not all(isinstance(data.get(key), str) and data[key] for key in ("encrypt", "msg_signature", "timestamp", "nonce")):
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
        aes_key = credential("WECOM_ENCODING_AES_KEY")
        if not aes_key:
            raise MessagingError("wecom webhook failed: WECOM_ENCODING_AES_KEY is not set")
        corp_id = credential("WECOM_CORP_ID")
        if not corp_id:
            raise MessagingError("wecom webhook failed: WECOM_CORP_ID is not set")
        try:
            plaintext = self._decryptor(encrypt, aes_key, corp_id)
        except WeChatCryptoError as exc:
            raise MessagingError(f"wecom webhook failed: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - never crash on untrusted input
            raise MessagingError(f"wecom webhook failed: decryptor raised {type(exc).__name__}") from None
        decoded = parse_wechat_plaintext(plaintext)
        if decoded is None:
            return []
        entries = decoded if isinstance(decoded, list) else [decoded]
        messages: list[NormalizedMessage] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "text" not in entry and entry.get("Content") is not None:
                # Legacy XML mode carries the body as a <Content> child.
                entry = {**entry, "text": {"content": entry["Content"]}}
            sender = entry.get("fromusername") or entry.get("FromUserName")
            text_body = entry.get("text") if isinstance(entry.get("text"), dict) else {}
            content = text_body.get("content")
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    sender=str(sender) if sender else "unknown",
                    text=str(content) if content is not None else "",
                    timestamp=from_epoch(entry.get("createtime") or entry.get("CreateTime")),
                    reply_ref=None,
                    message_id=ref_or_none(entry.get("msgid") or entry.get("MsgId")),
                )
            )
        return messages

    # -------------------------------------------------------------------- send

    @classmethod
    def _parse_target(cls, target: Any) -> tuple[int, str]:
        """Parse ``agent:<agentid>:<userid>`` defensively."""
        target_str = target if isinstance(target, str) else str(target)
        parts = target_str.split(":")
        if len(parts) != 3 or parts[0] != "agent" or not parts[2]:
            raise MessagingError("wecom send blocked: malformed target, expected 'agent:<agentid>:<userid>'")
        try:
            agent_id = int(parts[1])
        except ValueError:
            raise MessagingError(
                "wecom send blocked: malformed target, expected 'agent:<agentid>:<userid>'"
            ) from None
        return agent_id, parts[2]

    async def _fetch_access_token(self, corp_id: str, secret: str) -> str:
        """Get (or reuse) the corp access token; expiry via ``now_fn``."""
        now = float(self.now_fn())
        if self._access_token and self._token_expiry > now:
            return self._access_token
        try:
            response = await self._http().get(
                f"{self.API_BASE}/gettoken",
                params={"corpid": corp_id, "corpsecret": secret},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # The URL carries the secret: never echo it or httpx's message.
            raise MessagingError(f"wecom send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"wecom send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("wecom send failed: invalid response") from exc
        token = data.get("access_token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token:
            raise MessagingError("wecom send failed: API reported failure")
        try:
            expires_in = max(0.0, float(data.get("expires_in")))
        except (TypeError, ValueError):
            expires_in = 0.0
        self._access_token = token
        self._token_expiry = now + expires_in
        return token

    async def send(self, target: str, text: str) -> dict[str, Any]:
        corp_id = credential("WECOM_CORP_ID")
        if not corp_id:
            raise MessagingError("wecom send blocked: WECOM_CORP_ID is not set")
        secret = credential("WECOM_SECRET")
        if not secret:
            raise MessagingError("wecom send blocked: WECOM_SECRET is not set")
        target_str = target if isinstance(target, str) else str(target)
        agent_id, user_id = self._parse_target(target_str)
        access_token = await self._fetch_access_token(corp_id, secret)
        try:
            response = await self._http().post(
                f"{self.API_BASE}/cgi-bin/message/send",
                params={"access_token": access_token},
                json={"touser": user_id, "msgtype": "text", "agentid": agent_id, "text": {"content": str(text)}},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"wecom send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"wecom send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MessagingError("wecom send failed: invalid response") from exc
        if not isinstance(data, dict) or data.get("errcode") != 0:
            raise MessagingError("wecom send failed: API reported failure")
        return {
            "channel": self.channel,
            "target": target_str,
            "success": True,
            "message_id": None,
        }
