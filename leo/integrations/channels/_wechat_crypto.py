"""Shared WeChat-family callback crypto: msg_signature and EncodingAESKey payloads.

Private helper module used by :mod:`leo.integrations.channels.wecom` and
:mod:`leo.integrations.channels.weixin`. It defines **no** adapter (a module
without adapter classes is harmless to the package's auto-discovery), reads no
environment variables and touches no network.

Two documented WeChat constructions live here:

``wechat_msg_signature``
    The callback signature shared by WeCom and the Official Account platform:
    ``SHA1`` of the lexicographically sorted ``[token, timestamp, nonce,
    encrypt]`` values joined with ``""``, as a lowercase hex digest.

``decrypt_wechat_payload``
    The ``EncodingAESKey`` payload scheme: the key is the 43-character (+ ``=``)
    base64 ``EncodingAESKey`` decoding to a 32-byte AES key; the ciphertext is
    base64; AES-CBC with the IV equal to the first 16 key bytes; the padded
    plaintext is ``random(16) + msg_len(4, network order) + msg +
    receive_id``. Decryption strips PKCS7 and *requires* the trailing
    ``receive_id`` (corp id / app id) to match, so a payload minted for a
    different account is rejected.

The optional-crypto rule mirrors the Discord Ed25519 verifier: ``cryptography``
is lazily imported *inside* the function, never at module import, and its
absence raises :class:`WeChatCryptoError` so integrators can either install
``cryptography`` or inject their own ``decryptor`` callable into the adapters.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import struct

__all__ = ["WeChatCryptoError", "decrypt_wechat_payload", "wechat_msg_signature"]


class WeChatCryptoError(RuntimeError):
    """WeChat payload decryption is unavailable (optional crypto backend missing)."""


def wechat_msg_signature(token: str, timestamp: str, nonce: str, encrypt: str) -> str:
    """Return the documented WeChat callback signature as a hex digest.

    The four values are sorted lexicographically and concatenated without a
    separator before SHA-1. Inputs are coerced with ``str()`` so untrusted
    query/payload values can never raise; the caller owns constant-time
    comparison of the result.
    """
    items = sorted((str(token), str(timestamp), str(nonce), str(encrypt)))
    return hashlib.sha1("".join(items).encode("utf-8")).hexdigest()


def _b64_decode(value: str, what: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {what}: not base64 text")
    try:
        return base64.b64decode(value)
    except (ValueError, TypeError) as exc:  # binascii.Error is a ValueError
        raise ValueError(f"invalid {what}: not base64 text") from exc


def _pkcs7_unpad(data: bytes) -> bytes:
    """Strip PKCS7 padding as used by the WeChat scheme (block boundary 32)."""
    if not data:
        raise ValueError("invalid plaintext: empty")
    pad = data[-1]
    if not 1 <= pad <= 32 or pad > len(data) or data[-pad:] != bytes([pad]) * pad:
        raise ValueError("invalid plaintext: bad PKCS7 padding")
    return data[:-pad]


def decrypt_wechat_payload(encrypt_b64: str, encoding_aes_key_b64: str, app_id: str) -> str:
    """Decrypt one WeChat/WeCom ``encrypt`` field to its plaintext text.

    Args:
        encrypt_b64: base64 ciphertext exactly as delivered in the callback.
        encoding_aes_key_b64: the account's 43-character (+ ``=``)
            ``EncodingAESKey``; it must decode to the 32-byte AES key.
        app_id: the expected trailing receive id (corp id for WeCom, app id
            for the Official Account). A mismatch raises ``ValueError``.

    Raises:
        WeChatCryptoError: ``cryptography`` is not importable. Integrators
            either install it or pass a custom ``decryptor`` to the adapter.
        ValueError: the key, ciphertext, padding, framing or receive id is
            invalid — every malformed-input path fails closed.
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching in tests
        raise WeChatCryptoError(
            "WeChat payload decryption unavailable: install 'cryptography' "
            "or pass a custom decryptor= to the WeCom/WeChat adapter"
        ) from exc

    key = _b64_decode(encoding_aes_key_b64, "EncodingAESKey")
    if len(key) != 32:
        raise ValueError("invalid EncodingAESKey: must decode to 32 bytes")
    ciphertext = _b64_decode(encrypt_b64, "encrypt field")
    if not ciphertext or len(ciphertext) % 16:
        raise ValueError("invalid encrypt field: bad ciphertext length")

    decryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    plain = _pkcs7_unpad(padded)
    if len(plain) < 20:
        raise ValueError("invalid plaintext: too short for the documented framing")
    msg_len = struct.unpack(">I", plain[16:20])[0]
    if msg_len > len(plain) - 20:
        raise ValueError("invalid plaintext: declared message length exceeds payload")
    message = plain[20 : 20 + msg_len]
    receive_id = plain[20 + msg_len :]
    if not hmac.compare_digest(receive_id, str(app_id).encode("utf-8")):
        raise ValueError("invalid plaintext: receive id does not match this account")
    try:
        return message.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid plaintext: not valid UTF-8") from exc
