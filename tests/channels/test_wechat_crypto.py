"""Known-vector and fail-closed tests for the shared WeChat crypto helpers.

The AES round-trip tests construct ciphertexts with the documented
EncodingAESKey scheme (random(16) + network-order length + message +
receive_id, PKCS7 to the 32-byte boundary, AES-CBC with the IV equal to the
first 16 key bytes) and therefore need the optional ``cryptography``
package; they skip cleanly when it is absent. The fail-closed tests run in
every environment and prove that a missing crypto backend can never silently
accept a payload.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import struct
import sys

import pytest

from leo.integrations.channels._wechat_crypto import (
    WeChatCryptoError,
    decrypt_wechat_payload,
    wechat_msg_signature,
)

CRYPTOGRAPHY_AVAILABLE = importlib.util.find_spec("cryptography") is not None
requires_crypto = pytest.mark.skipif(
    not CRYPTOGRAPHY_AVAILABLE, reason="optional 'cryptography' package not installed"
)

# A fixed test-only key so tamper outcomes are deterministic.
FIXED_KEY = bytes(range(32))
FIXED_AES_KEY = base64.b64encode(FIXED_KEY).decode("ascii")


def make_crypto_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an environment without the optional cryptography package."""
    for name in [n for n in sys.modules if n == "cryptography" or n.startswith("cryptography.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "cryptography", None)


def encrypt_wechat_plaintext(message: bytes, key: bytes, receive_id: str) -> str:
    """Encrypt with the documented scheme the decryptor must undo."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    body = os.urandom(16) + struct.pack(">I", len(message)) + message + receive_id.encode("utf-8")
    pad = 32 - len(body) % 32  # PKCS7 always pads to the 32-byte boundary
    padded = body + bytes([pad]) * pad
    encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
    return base64.b64encode(encryptor.update(padded) + encryptor.finalize()).decode("ascii")


def test_wechat_msg_signature_known_vector():
    # SHA1 of the lexicographically sorted [token, timestamp, nonce, encrypt]
    # joined with "" — precomputed so the documented semantics are pinned.
    expected = "dec636002b57c15ec4cf5e4095ae136dd1a4aaf5"
    assert wechat_msg_signature("LEOtestToken", "1700000000", "8f8c9a2b", "QUJDREVG") == expected
    recomputed = hashlib.sha1(
        "".join(sorted(["LEOtestToken", "1700000000", "8f8c9a2b", "QUJDREVG"])).encode("utf-8")
    ).hexdigest()
    assert recomputed == expected
    # Argument order is irrelevant (the scheme sorts); changing any value is not.
    assert wechat_msg_signature("LEOtestToken", "8f8c9a2b", "1700000000", "QUJDREVG") == expected
    assert wechat_msg_signature("LEOtestToken", "1700000001", "8f8c9a2b", "QUJDREVG") != expected


def test_wechat_msg_signature_never_crashes_on_bad_types():
    signature = wechat_msg_signature(None, 123, b"bytes", {"weird": True})
    coerced = hashlib.sha1(
        "".join(sorted(["None", "123", "b'bytes'", "{'weird': True}"])).encode("utf-8")
    ).hexdigest()
    assert signature == coerced


@requires_crypto
def test_decrypt_round_trip_json_and_xml():
    assert len(FIXED_AES_KEY) == 44 and FIXED_AES_KEY.endswith("=")  # 43 chars + "="
    for receive_id, message in [
        ("ww-fake-corp", json.dumps({"msgtype": "text", "text": {"content": "hi"}}).encode("utf-8")),
        ("wx-fake-app", b"<xml><MsgType>text</MsgType><Content>hello</Content></xml>"),
        ("wx-fake-app", b""),
    ]:
        encrypt = encrypt_wechat_plaintext(message, FIXED_KEY, receive_id)
        assert decrypt_wechat_payload(encrypt, FIXED_AES_KEY, receive_id) == message.decode("utf-8")


@requires_crypto
def test_decrypt_rejects_wrong_receive_id():
    encrypt = encrypt_wechat_plaintext(b'{"hello": 1}', FIXED_KEY, "ww-right-corp")
    with pytest.raises(ValueError):
        decrypt_wechat_payload(encrypt, FIXED_AES_KEY, "ww-wrong-corp")


@requires_crypto
def test_decrypt_rejects_malformed_inputs():
    good = encrypt_wechat_plaintext(b'{"ok": 1}', FIXED_KEY, "wx-fake-app")
    # Invalid / mistyped fields never crash and never decrypt.
    with pytest.raises(ValueError):
        decrypt_wechat_payload("!!! not base64 !!!", FIXED_AES_KEY, "wx-fake-app")
    with pytest.raises(ValueError):
        decrypt_wechat_payload("", FIXED_AES_KEY, "wx-fake-app")
    with pytest.raises(ValueError):
        decrypt_wechat_payload(123, FIXED_AES_KEY, "wx-fake-app")  # type: ignore[arg-type]
    # Keys that do not decode to the documented 32 bytes.
    with pytest.raises(ValueError):
        decrypt_wechat_payload(good, base64.b64encode(b"short-key").decode("ascii"), "wx-fake-app")
    # Ciphertext that is not a multiple of the AES block size.
    with pytest.raises(ValueError):
        decrypt_wechat_payload(base64.b64encode(b"0123456789abcde").decode("ascii"), FIXED_AES_KEY, "wx-fake-app")
    # Tampered ciphertext fails closed (padding / receive-id integrity).
    raw = bytearray(base64.b64decode(good))
    raw[-1] ^= 0xFF
    with pytest.raises(ValueError):
        decrypt_wechat_payload(base64.b64encode(bytes(raw)).decode("ascii"), FIXED_AES_KEY, "wx-fake-app")


def test_decrypt_fails_closed_without_crypto_backend(monkeypatch):
    """No cryptography package -> loud WeChatCryptoError, never silent acceptance."""
    make_crypto_unimportable(monkeypatch)
    with pytest.raises(WeChatCryptoError) as exc_info:
        decrypt_wechat_payload("ZmFrZQ==", FIXED_AES_KEY, "wx-fake-app")
    assert isinstance(exc_info.value, RuntimeError)
    message = str(exc_info.value)
    assert "cryptography" in message
    assert "decryptor" in message
