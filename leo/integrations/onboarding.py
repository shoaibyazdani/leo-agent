"""Guided, honest channel onboarding (no network, no secrets in payloads).

This module is the framework-free backend for the dashboard's "connect a
channel" wizard. It provides three things:

1. A guided field schema: labels, plain-language help, examples, defaults,
   required/optional semantics and per-field validation messages. No credential
   *value* is ever stored here or returned from here.
2. A local, QR-guided setup flow for WhatsApp. It issues a short-lived,
   single-use, opaque setup token, renders a real QR as a locally generated SVG
   data URI, and keeps an auditable state machine (started -> awaiting_scan ->
   paired, plus blocked/failed) in the durable ``kv`` store.
3. The security controls that such a flow needs: expiry, one-time redemption,
   per-client and per-code rate limits, a bound on outstanding sessions, strict
   loopback origin handling, and no secret material in URLs, QR payloads,
   responses, errors or logs.

Truthfulness is a hard requirement. LEO ships only the official Meta WhatsApp
Cloud API adapter; there is no WhatsApp Web/Baileys device-link bridge. The QR
below is therefore a *local setup* QR, never a WhatsApp device-link QR. The
status payload always reports ``linked: false`` and ``device_link.status ==
"blocked"`` until a real bridge registers itself through
:func:`device_link_backend`. Scanning the QR cannot link a personal WhatsApp
account and the UI copy says so.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

from .. import db
from ..logging import get_logger

_LOG = get_logger("integrations.onboarding")

# --- Flow policy -----------------------------------------------------------

STATE_STARTED = "started"
STATE_AWAITING_SCAN = "awaiting_scan"
STATE_PAIRED = "paired"
STATE_BLOCKED = "blocked"
STATE_FAILED = "failed"

#: Match the 60s stream-token convention; a multi-step scan may use at most 5m.
SETUP_TOKEN_TTL_SECONDS = 60.0
SETUP_TOKEN_MAX_SECONDS = 300.0
MAX_PENDING_SESSIONS = 16
ISSUE_LIMIT_PER_MINUTE = 5
REDEEM_LIMIT_PER_MINUTE = 10
REDEEM_LIMIT_PER_CODE = 5
FAILED_SESSION_RETENTION_SECONDS = 600.0

_SESSION_PREFIX = "onboarding.session."
_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{20,128}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{12,64}$")

BLOCKED_DEVICE_LINK_REASON = (
    "LEO has no WhatsApp Web (Baileys) bridge installed, so it cannot link a "
    "personal WhatsApp device. This QR is a local setup code, not a WhatsApp "
    "device-link QR. Use the official Meta Cloud API setup below."
)
_CLOUD_API_NOTE = (
    "Cloud credentials can be saved offline only. Sending and inbound replies "
    "are not tested by LEO; Meta app, a business phone number and a public "
    "HTTPS webhook are set up outside LEO."
)


class OnboardingError(ValueError):
    """An onboarding request was rejected. ``message`` is safe to show users."""

    def __init__(self, message: str, *, status_code: int = 400, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


# --- Durable session store -------------------------------------------------


def _code_hash(code: str) -> str:
    """Never persist or log the raw setup code, only its digest."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


async def _save_session(record: dict[str, Any]) -> None:
    now = _iso(time.time())
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (_SESSION_PREFIX + record["code_hash"], json.dumps(record, separators=(",", ":")), now),
    )


async def _load_session(code_hash: str) -> dict[str, Any] | None:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (_SESSION_PREFIX + code_hash,))
    if row is None:
        return None
    try:
        record = json.loads(row["value"])
    except (TypeError, ValueError):
        return None
    return record if isinstance(record, dict) else None


async def _all_sessions() -> list[tuple[str, dict[str, Any]]]:
    rows = await db.fetchall("SELECT key,value FROM kv WHERE key LIKE ?", (_SESSION_PREFIX + "%",))
    result: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            result.append((str(row["key"]), value))
    return result


async def _delete_session(code_hash: str) -> None:
    await db.execute("DELETE FROM kv WHERE key=?", (_SESSION_PREFIX + code_hash,))


def _transition(record: dict[str, Any], state: str, *, reason: str | None = None) -> None:
    entry: dict[str, Any] = {"state": state, "at": _iso(time.time())}
    if reason:
        entry["reason"] = reason
    record.setdefault("transitions", []).append(entry)
    record["state"] = state
    record["updated_at"] = entry["at"]
    if reason:
        record["failure"] = reason


async def _prune_sessions(now: float) -> None:
    """Mark expired sessions failed and eventually remove dead state."""
    for key, record in await _all_sessions():
        expires = float(record.get("expires_epoch", 0) or 0)
        if expires > now:
            continue
        if now - expires < FAILED_SESSION_RETENTION_SECONDS:
            if record.get("state") in (STATE_STARTED, STATE_AWAITING_SCAN):
                _transition(record, STATE_FAILED, reason="setup code expired")
                record["code_hash"] = key[len(_SESSION_PREFIX):]
                await _save_session(record)
        else:
            await db.execute("DELETE FROM kv WHERE key=?", (key,))


# --- Rate limiting ---------------------------------------------------------


class RateLimiter:
    """Fixed-window limiter keyed by an opaque identifier (never a secret)."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, *, message: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                raise OnboardingError(message, status_code=429, code="rate_limited")
            hits.append(now)
            if len(self._hits) > 1024:
                for name, values in tuple(self._hits.items()):
                    if not values or values[-1] <= now - self.window:
                        self._hits.pop(name, None)


_issue_limiter = RateLimiter(ISSUE_LIMIT_PER_MINUTE, 60.0)
_redeem_client_limiter = RateLimiter(REDEEM_LIMIT_PER_MINUTE, 60.0)
_redeem_code_limiter = RateLimiter(REDEEM_LIMIT_PER_CODE, 60.0)


# --- Loopback / origin helpers --------------------------------------------


def _configured_origin() -> str:
    from ..config import Settings

    configured = list(getattr(Settings.load().server, "cors_origins", None) or [])
    for item in configured:
        text = str(item).strip()
        if text and text != "*":
            return text.rstrip("/")
    return "http://localhost:5173"


def _normalized_origin(value: str) -> str:
    parts = urlsplit(str(value).strip())
    if not parts.scheme or not parts.hostname:
        return ""
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return ""
    if port is None:
        port = 443 if scheme == "https" else 80
    return f"{scheme}://{host}:{port}"


def safe_base_url(candidate: str | None = None, *, request_url: str | None = None) -> str:
    """Return a loopback-only origin for setup links.

    A hostile ``Host`` header can never steer the URL: non-loopback or
    credential-bearing candidates fall back to the configured dashboard origin.
    """
    raw = str(candidate or request_url or "").strip()
    parts = urlsplit(raw)
    host = parts.hostname or ""
    loopback = host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")
    if not loopback or parts.username or parts.password:
        parts = urlsplit(_configured_origin())
    scheme = parts.scheme.lower() if parts.scheme in ("http", "https") else "http"
    host = parts.hostname or "localhost"
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError:
        port = 443 if scheme == "https" else 80
    if ":" in host:
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def origin_allowed(origin: str | None, *, base_url: str) -> bool:
    """Exact origin match only; no wildcards, suffixes or ``null`` origins."""
    if not origin:
        return True  # non-browser clients use the custom header / bearer token
    candidate = _normalized_origin(origin)
    if not candidate:
        return False
    from ..config import Settings

    allowed = {_normalized_origin(safe_base_url(base_url))}
    for item in getattr(Settings.load().server, "cors_origins", None) or []:
        normalized = _normalized_origin(str(item))
        if normalized and normalized != "null":
            allowed.add(normalized)
    return candidate in allowed


# --- Local QR encoder (byte mode, ECC level M, versions 1-10) --------------

# ECC codewords per block and block counts for versions 1..10, columns L/M/Q/H.
_QR_ECC_CODEWORDS = (
    (7, 10, 15, 20), (10, 16, 26, 18), (15, 26, 18, 22), (20, 18, 26, 20),
    (26, 24, 18, 24), (18, 16, 24, 28), (20, 18, 22, 28), (24, 22, 28, 26),
    (30, 22, 26, 24), (18, 26, 24, 28),
)
_QR_BLOCKS = (
    (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 4), (1, 2, 4, 4),
    (2, 4, 4, 4), (2, 4, 4, 5), (2, 4, 5, 6), (2, 5, 5, 6), (4, 5, 5, 8),
)
_QR_ECL_INDEX = 1  # Medium: robust enough for a small SVG, compact enough for a URL.
_QR_FORMAT_BITS = (1, 0, 3, 2)  # L, M, Q, H


def _gf_mul(x: int, y: int) -> int:
    z = 0
    for i in range(7, -1, -1):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z & 0xFF


def _rs_divisor(degree: int) -> list[int]:
    result = [0] * (degree - 1) + [1]
    root = 1
    for _ in range(degree):
        for j in range(len(result)):
            result[j] = _gf_mul(result[j], root)
            if j + 1 < len(result):
                result[j] ^= result[j + 1]
        root = _gf_mul(root, 0x02)
    return result


def _rs_remainder(data: list[int], divisor: list[int]) -> list[int]:
    result = [0] * len(divisor)
    for value in data:
        factor = value ^ result[0]
        result = result[1:] + [0]
        for i, coefficient in enumerate(divisor):
            result[i] ^= _gf_mul(coefficient, factor)
    return result


def _raw_modules(version: int) -> int:
    result = (16 * version + 128) * version + 64
    if version >= 2:
        num_align = version // 7 + 2
        result -= (25 * num_align - 10) * num_align - 55
        if version >= 7:
            result -= 36
    return result


def _data_capacity(version: int) -> int:
    raw_codewords = _raw_modules(version) // 8
    return raw_codewords - _QR_ECC_CODEWORDS[version - 1][_QR_ECL_INDEX] * _QR_BLOCKS[version - 1][_QR_ECL_INDEX]


def _qr_version(payload: bytes) -> int:
    for version in range(1, len(_QR_ECC_CODEWORDS) + 1):
        count_bits = 8 if version <= 9 else 16
        needed = 4 + count_bits + len(payload) * 8
        if needed <= _data_capacity(version) * 8:
            return version
    raise ValueError("setup link is too long to render as a QR code")


def _make_codewords(payload: bytes, version: int) -> list[int]:
    capacity_bits = _data_capacity(version) * 8
    bits: list[int] = []

    def append(value: int, length: int) -> None:
        for shift in reversed(range(length)):
            bits.append((value >> shift) & 1)

    append(0b0100, 4)  # byte mode
    append(len(payload), 8 if version <= 9 else 16)
    for byte in payload:
        append(byte, 8)
    append(0, min(4, capacity_bits - len(bits)))
    while len(bits) % 8:
        bits.append(0)
    padding = 0xEC
    while len(bits) < capacity_bits:
        append(padding, 8)
        padding ^= 0xEC ^ 0x11

    codewords = [sum(bit << (7 - i) for i, bit in enumerate(bits[pos:pos + 8])) for pos in range(0, len(bits), 8)]
    num_blocks = _QR_BLOCKS[version - 1][_QR_ECL_INDEX]
    ecc_len = _QR_ECC_CODEWORDS[version - 1][_QR_ECL_INDEX]
    # Split the DATA codewords, not the raw (data + ECC) count: with multiple
    # blocks these differ and using the raw count leaves a trailing empty block
    # and invalid Reed-Solomon parity. Verified by decoding generated matrices.
    data_codewords = len(codewords)
    short_blocks = num_blocks - (data_codewords % num_blocks)
    short_len = data_codewords // num_blocks

    data_blocks: list[list[int]] = []
    ecc_blocks: list[list[int]] = []
    for index in range(num_blocks):
        # Short groups come first; longer groups (when the data count is not
        # divisible by the block count) start after all short groups.
        start = (
            index * short_len if index < short_blocks
            else short_blocks * short_len + (index - short_blocks) * (short_len + 1)
        )
        length = short_len + (0 if index < short_blocks else 1)
        block = codewords[start:start + length]
        data_blocks.append(block)
        ecc_blocks.append(_rs_remainder(block, _rs_divisor(ecc_len)))

    result: list[int] = []
    for i in range(short_len + 1):
        for block in data_blocks:
            if i < len(block):
                result.append(block[i])
    for i in range(ecc_len):
        for block in ecc_blocks:
            result.append(block[i])
    return result


def _mask(mask: int, x: int, y: int) -> bool:
    if mask == 0:
        return (x + y) % 2 == 0
    if mask == 1:
        return y % 2 == 0
    if mask == 2:
        return x % 3 == 0
    if mask == 3:
        return (x + y) % 3 == 0
    if mask == 4:
        return (y // 2 + x // 3) % 2 == 0
    if mask == 5:
        return x * y % 2 + x * y % 3 == 0
    if mask == 6:
        return (x * y % 2 + x * y % 3) % 2 == 0
    return ((x + y) % 2 + x * y % 3) % 2 == 0


def _penalty(modules: list[list[bool]]) -> int:
    size = len(modules)
    score = 0
    finder = (True, False, True, True, True, False, True)
    for line in list(modules) + [[modules[y][x] for y in range(size)] for x in range(size)]:
        run_value, run_length = line[0], 1
        for value in line[1:]:
            if value == run_value:
                run_length += 1
            else:
                if run_length >= 5:
                    score += 3 + run_length - 5
                run_value, run_length = value, 1
        if run_length >= 5:
            score += 3 + run_length - 5
        for i in range(size - 6):
            window = tuple(line[i:i + 7])
            before = tuple(line[max(0, i - 4):i])
            after = tuple(line[i + 7:i + 11])
            if window == finder and (before == (False,) * 4 or after == (False,) * 4):
                score += 40
    for y in range(size - 1):
        for x in range(size - 1):
            color = modules[y][x]
            if modules[y][x + 1] == color and modules[y + 1][x] == color and modules[y + 1][x + 1] == color:
                score += 3
    dark = sum(sum(row) for row in modules)
    score += abs(dark * 20 - size * size * 10) // (size * size) * 10
    return score


def qr_matrix(payload: str) -> list[list[bool]]:
    """Encode a string into a QR byte-mode matrix (useful for tests)."""
    raw = payload.encode("utf-8")
    version = _qr_version(raw)
    size = version * 4 + 17
    modules = [[False] * size for _ in range(size)]
    function = [[False] * size for _ in range(size)]

    def set_function(x: int, y: int, dark: bool) -> None:
        if 0 <= x < size and 0 <= y < size:
            modules[y][x] = dark
            function[y][x] = True

    for i in range(size):
        set_function(6, i, i % 2 == 0)
        set_function(i, 6, i % 2 == 0)
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                distance = max(abs(dx), abs(dy))
                set_function(cx + dx, cy + dy, distance != 2 and distance != 4)

    # Version 1 has no alignment pattern at all; versions 2+ place them along
    # the diagonal, skipping the three finder-pattern corners.
    if version >= 2:
        num_align = version // 7 + 2
        step = ((version * 4 + num_align * 2 + 1) // (num_align * 2 - 2)) * 2
        align_positions = [6] + [size - 7 - i * step for i in reversed(range(num_align - 1))]
        for x in align_positions:
            for y in align_positions:
                if (x == 6 and y == 6) or (x == 6 and y == size - 7) or (x == size - 7 and y == 6):
                    continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        set_function(x + dx, y + dy, max(abs(dx), abs(dy)) != 1)

    def draw_format(mask: int, target: list[list[bool]]) -> None:
        data = _QR_FORMAT_BITS[_QR_ECL_INDEX] << 3 | mask
        remainder = data
        for _ in range(10):
            remainder = (remainder << 1) ^ ((remainder >> 9) * 0x537)
        bits = (data << 10 | remainder) ^ 0x5412

        def put(x: int, y: int, dark: bool) -> None:
            if 0 <= x < size and 0 <= y < size:
                target[y][x] = dark
                function[y][x] = True

        for i in range(6):
            put(8, i, (bits >> i) & 1 == 1)
        put(8, 7, (bits >> 6) & 1 == 1)
        put(8, 8, (bits >> 7) & 1 == 1)
        put(7, 8, (bits >> 8) & 1 == 1)
        for i in range(9, 15):
            put(14 - i, 8, (bits >> i) & 1 == 1)
        for i in range(8):
            put(size - 1 - i, 8, (bits >> i) & 1 == 1)
        for i in range(8, 15):
            put(8, size - 15 + i, (bits >> i) & 1 == 1)
        put(8, size - 8, True)

    draw_format(0, modules)
    if version >= 7:
        remainder = version
        for _ in range(12):
            remainder = (remainder << 1) ^ ((remainder >> 11) * 0x1F25)
        bits = version << 12 | remainder
        for i in range(18):
            bit = (bits >> i) & 1 == 1
            a, b = size - 11 + i % 3, i // 3
            set_function(a, b, bit)
            set_function(b, a, bit)

    codewords = _make_codewords(raw, version)
    total_bits = len(codewords) * 8
    best: tuple[int, int, list[list[bool]]] | None = None
    for mask in range(8):
        candidate = [row[:] for row in modules]
        index = 0
        right = size - 1
        while right >= 1:
            if right == 6:
                right = 5
            for vert in range(size):
                for j in range(2):
                    x = right - j
                    upward = ((right + 1) & 2) == 0
                    y = size - 1 - vert if upward else vert
                    if function[y][x] or index >= total_bits:
                        continue
                    bit = (codewords[index >> 3] >> (7 - (index & 7))) & 1 == 1
                    if _mask(mask, x, y):
                        bit = not bit
                    candidate[y][x] = bit
                    index += 1
            right -= 2
        score = _penalty(candidate)
        if best is None or score < best[0]:
            best = (score, mask, candidate)
    assert best is not None
    final = best[2]
    # The scoring pass used placeholder format bits; write the winning mask.
    draw_format(best[1], final)
    return final


def qr_svg_data_uri(payload: str, *, border: int = 4, box: int = 4) -> tuple[str, str]:
    """Return ``(svg, data_uri)`` for a locally rendered QR (no dependencies)."""
    matrix = qr_matrix(payload)
    size = len(matrix)
    dimension = (size + 2 * border) * box
    segments = []
    for y, row in enumerate(matrix):
        for x, dark in enumerate(row):
            if dark:
                px, py = (x + border) * box, (y + border) * box
                segments.append(f"M{px} {py}h{box}v{box}h-{box}z")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{dimension}" height="{dimension}" '
        f'viewBox="0 0 {dimension} {dimension}" shape-rendering="crispEdges">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<path fill="#000000" d="{"".join(segments)}"/></svg>'
    )
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return svg, data_uri


# --- Guided field schema ---------------------------------------------------

_FIELD_GUIDANCE: dict[str, dict[str, str]] = {
    "WHATSAPP_CLOUD_TOKEN": {
        "example": "EAAG... (long Meta access token)",
        "placeholder": "Paste the Meta access token",
        "hint": "Meta Business Suite → Business settings → WhatsApp → API setup → temporary or permanent token.",
    },
    "WHATSAPP_PHONE_NUMBER_ID": {
        "example": "123456789012345",
        "placeholder": "13-18 digit number, not your phone number",
        "hint": "Meta Business Manager → WhatsApp → Phone numbers → Phone number ID. This is NOT your WhatsApp phone number.",
    },
    "WHATSAPP_APP_SECRET": {
        "example": "abcdef0123456789abcdef0123456789",
        "placeholder": "Paste the Meta app secret",
        "hint": "Meta for Developers → your app → Settings → Basic → App secret. Used to verify webhook signatures.",
    },
    "WHATSAPP_API_VERSION": {
        "example": "v20.0",
        "placeholder": "v20.0",
        "hint": "Graph API version shown in Meta's API setup page; leave blank for v20.0.",
    },
    "TELEGRAM_BOT_TOKEN": {
        "example": "123456789:AAExampleBotToken",
        "placeholder": "Paste the token from @BotFather",
        "hint": "Open Telegram, message @BotFather, send /newbot and copy the token it returns.",
    },
    "TELEGRAM_WEBHOOK_SECRET": {
        "example": "a-long-random-string-you-chose",
        "placeholder": "Choose a long random string",
        "hint": "Use the same value when you call Telegram setWebhook with X-Telegram-Bot-Api-Secret-Token.",
    },
}

_VALIDATION_RULES: dict[str, tuple[str, str]] = {
    "WHATSAPP_PHONE_NUMBER_ID": (
        r"^\d{13,18}$",
        "Enter the numeric Phone number ID from Meta Business Manager (13-18 digits), not your WhatsApp phone number.",
    ),
    "WHATSAPP_API_VERSION": (r"^v\d+\.\d+$", "Use a Graph API version such as v20.0."),
    "TWILIO_ACCOUNT_SID": (r"^AC[0-9a-fA-F]{32}$", "Twilio Account SIDs start with AC and have 34 characters."),
    "SMTP_PORT": (r"^\d{1,5}$", "Enter a port number between 1 and 65535."),
    "IMAP_PORT": (r"^\d{1,5}$", "Enter a port number between 1 and 65535."),
    "IRC_PORT": (r"^\d{1,5}$", "Enter a port number between 1 and 65535."),
}
_URL_FIELDS = frozenset({
    "DISCORD_WEBHOOK_URL", "MATTERMOST_URL", "GOOGLE_CHAT_WEBHOOK_URL",
    "TEAMS_INCOMING_WEBHOOK_URL", "DINGTALK_WEBHOOK_URL", "NTFY_URL",
    "BLUEBUBBLES_URL", "SIGNAL_CLI_URL", "A2A_AGENT_URL", "RELAY_TARGET_URL",
    "TWILIO_WEBHOOK_URL", "TWILIO_BASE_URL", "MATRIX_HOMESERVER",
})
_E164_CHANNELS = frozenset({"whatsapp", "sms", "signal"})
_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")


def field_guidance(name: str, *, kind: str = "text", label: str = "") -> dict[str, Any]:
    """Plain-language metadata for one field (never includes a value)."""
    guidance = _FIELD_GUIDANCE.get(name, {})
    example = guidance.get("example")
    if example is None:
        if kind == "secret":
            example = "Paste the value from your provider"
        elif kind == "number":
            example = "12345"
        elif kind == "url":
            example = "https://example.com"
        else:
            example = f"Example {label or name}".lower()
    validation: dict[str, str] = {}
    pattern = _VALIDATION_RULES.get(name)
    if pattern:
        validation = {"pattern": pattern[0], "message": pattern[1]}
    elif name in _URL_FIELDS:
        validation = {
            "pattern": r"^https?://[^\s]+$",
            "message": "Enter a full URL starting with http:// or https://.",
        }
    return {
        "example": example,
        "placeholder": guidance.get("placeholder", example),
        "hint": guidance.get("hint", ""),
        "validation": validation,
        "sensitive": kind == "secret",
    }


def validate_values(channel: str, values: dict[str, Any]) -> list[dict[str, str]]:
    """Return field-level validation errors. No submitted value is echoed."""
    errors: list[dict[str, str]] = []

    def add(name: str, message: str) -> None:
        errors.append({"field": name, "name": name, "message": message})

    known = _channel_field_names(channel)
    for name in sorted(set(values) - known):
        add(str(name), "This field is not part of this channel's setup. Remove it and try again.")

    for name, raw in values.items():
        name = str(name)
        text = "" if raw is None else str(raw).strip()
        if not text:
            continue
        if name not in known:
            continue
        if "\n" in text or "\r" in text:
            if name != "allowlist":
                add(name, f"{name} must not contain newlines; paste a single value.")
            continue
        if name == "mode":
            if text not in ("allowlist", "open", "deny"):
                add(name, "Choose one of: allowlist (only listed senders), open (anyone), deny (nobody).")
            continue
        if name == "allowlist":
            if channel in _E164_CHANNELS:
                targets = [item.strip() for item in text.split(",") if item.strip()]
                bad = [item for item in targets if not _E164_RE.fullmatch(item)]
                if bad:
                    add(name, "Use phone numbers with country code, e.g. +15551234567 (one per line or comma).")
            continue
        rule = _VALIDATION_RULES.get(name)
        if rule and not re.fullmatch(rule[0], text):
            add(name, rule[1])
            continue
        if name in _URL_FIELDS and not re.fullmatch(r"https?://[^\s]+", text):
            add(name, "Enter a full URL starting with http:// or https://.")
            continue
        if name in ("SMTP_PORT", "IMAP_PORT", "IRC_PORT") and not (1 <= int(text or 0) <= 65535):
            add(name, "Enter a port number between 1 and 65535.")
    return errors


def _channel_field_names(channel: str) -> set[str]:
    from .channels import ALL_ADAPTERS

    cls = ALL_ADAPTERS.get(channel)
    if cls is None:
        return set()
    from .channels.specs import spec_for

    spec = spec_for(channel)
    return {field.name for field in spec.fields}


def channel_guide(channel: str, *, fields: list[dict[str, Any]] | None = None,
                  docs_url: str = "") -> dict[str, Any]:
    """Explain the onboarding journey in plain language."""
    if channel == "whatsapp":
        steps = [
            {
                "title": "1. Pick a connection method",
                "detail": (
                    "Personal QR device linking is not available in LEO: no WhatsApp Web "
                    "bridge is installed. Use Meta's official Cloud API instead."
                ),
            },
            {
                "title": "2. Create a Meta app with WhatsApp",
                "detail": "Create an app in Meta for Developers, add the WhatsApp product, then open API setup.",
                "docs_url": docs_url or "https://developers.facebook.com/docs/whatsapp/cloud-api",
            },
            {
                "title": "3. Copy three values from Meta",
                "detail": "Access token, Phone number ID and App secret. The Phone number ID is not your phone number.",
            },
            {
                "title": "4. Decide who may message LEO",
                "detail": "Start with your own number in international format, e.g. +15551234567. An empty list blocks everyone.",
            },
            {
                "title": "5. Point Meta's webhook at LEO",
                "detail": (
                    "Use a public HTTPS URL ending in /webhooks/whatsapp and set the verify token "
                    "in the Meta dashboard. LEO verifies X-Hub-Signature-256 with the app secret."
                ),
                "callback_path": "/webhooks/whatsapp",
            },
            {
                "title": "6. Check, then enable",
                "detail": "Check my settings verifies saved fields offline only. It never sends a test message or links a device.",
            },
        ]
        notes = [BLOCKED_DEVICE_LINK_REASON, _CLOUD_API_NOTE]
        method = "qr-guided-local-setup"
    else:
        steps = [
            {"title": "1. Get the credentials from your provider", "detail": "Each field explains where to find its value."},
            {"title": "2. Fill in the fields and choose who may message LEO", "detail": "An empty allowed-sender list blocks all inbound messages."},
            {"title": "3. Check my settings, then enable", "detail": "The check is offline only; it does not send a test message."},
        ]
        notes = [_CLOUD_API_NOTE]
        method = "guided-credentials"
    return {
        "title": f"Connect {channel}",
        "method": method,
        "steps": steps,
        "notes": notes,
        "docs_url": docs_url,
        "field_count": len(fields or []),
        "required_count": sum(1 for field in (fields or []) if field.get("required")),
    }


# --- WhatsApp local-setup flow --------------------------------------------


def device_link_backend(channel: str) -> Callable[..., Any] | None:
    """Return a registered device-link backend, if a real bridge ever exists."""
    from .channels import ALL_ADAPTERS

    cls = ALL_ADAPTERS.get(channel)
    backend = getattr(cls, "device_link_backend", None)
    return backend if callable(backend) else None


def device_link_status(channel: str) -> dict[str, Any]:
    backend = device_link_backend(channel)
    if backend is not None:
        return {
            "status": "available",
            "linked": False,
            "reason": None,
            "backend": getattr(backend, "__name__", type(backend).__name__),
            "note": "A device-link bridge is registered; linking still requires a live QR from that bridge.",
        }
    reason = BLOCKED_DEVICE_LINK_REASON if channel == "whatsapp" else "This channel has no device-link transport."
    return {"status": "blocked", "linked": False, "reason": reason, "backend": None, "note": None}


def _public_status(record: dict[str, Any]) -> dict[str, Any]:
    expires = float(record.get("expires_epoch", 0) or 0)
    remaining = max(0, int(expires - time.time()))
    state = str(record.get("state", STATE_FAILED))
    messages = {
        STATE_STARTED: "Setup is starting.",
        STATE_AWAITING_SCAN: (
            "Setup QR is ready. Open the setup link on this computer and confirm the code. "
            "This QR does not link a WhatsApp device."
        ),
        STATE_PAIRED: (
            "Local setup code confirmed. No WhatsApp device is linked; finish with the "
            "official Cloud API steps below."
        ),
        STATE_BLOCKED: BLOCKED_DEVICE_LINK_REASON,
        STATE_FAILED: str(record.get("failure") or "This setup session is no longer valid. Start a new one."),
    }
    return {
        "channel": record.get("channel"),
        "session_id": record.get("session_id"),
        "state": state,
        "message": messages.get(state, "This setup session is no longer valid. Start a new one."),
        # Never claim a device link or a ready channel on the basis of a QR.
        "linked": False,
        "ready": False,
        "device_link": dict(record.get("device_link") or {}),
        "expires_at": record.get("expires_at"),
        "expires_in": remaining,
        "transitions": list(record.get("transitions") or []),
        "failure": record.get("failure"),
        "confirmed_at": record.get("confirmed_at"),
    }


async def _load_by_session_id(session_id: str) -> tuple[str, dict[str, Any]] | None:
    for key, record in await _all_sessions():
        if record.get("session_id") == session_id:
            return key[len(_SESSION_PREFIX):], record
    return None


async def session_status(
    session_id: str, *, refresh: bool = False
) -> dict[str, Any] | None:
    """Read status without writing; never includes the setup code.

    A plain GET must not mutate durable state: expiry housekeeping belongs to
    the explicit redemption endpoint (or to a caller that passes ``refresh``).
    An expired-but-unused session is reported as expired in the message and in
    ``expires_in`` while its stored state stays exactly as it was.
    """
    if refresh:
        await _prune_sessions(time.time())
    if not isinstance(session_id, str) or not _SESSION_RE.fullmatch(session_id):
        return None
    found = await _load_by_session_id(session_id)
    if found is None:
        return None
    record = found[1]
    status = _public_status(record)
    expires = float(record.get("expires_epoch", 0) or 0)
    if status["state"] in (STATE_STARTED, STATE_AWAITING_SCAN) and expires <= time.time():
        status["message"] = "This setup code expired. Start a new setup."
        status["expired"] = True
    return status


async def start_onboarding(
    channel: str,
    *,
    base_url: str,
    client: str = "local",
    ttl_seconds: float | None = SETUP_TOKEN_TTL_SECONDS,
) -> dict[str, Any]:
    """Issue one opaque setup token, its local QR and an honest status payload."""
    if channel != "whatsapp":
        raise OnboardingError(
            f"Guided QR setup is not available for '{channel}'. Use the guided credential form instead.",
            status_code=404,
            code="unsupported_channel",
        )
    ttl = min(max(float(SETUP_TOKEN_TTL_SECONDS if ttl_seconds is None else ttl_seconds), 1.0), SETUP_TOKEN_MAX_SECONDS)
    origin = safe_base_url(base_url)
    _issue_limiter.check(f"{client}:{channel}", message="Too many setup codes requested. Wait a minute and try again.")

    now = time.time()
    await _prune_sessions(now)
    active = sum(
        1 for _, record in await _all_sessions()
        if record.get("state") in (STATE_STARTED, STATE_AWAITING_SCAN)
        and float(record.get("expires_epoch", 0) or 0) > now
    )
    if active >= MAX_PENDING_SESSIONS:
        raise OnboardingError(
            "Too many setup codes are waiting. Try again in a minute.",
            status_code=429,
            code="too_many_sessions",
        )

    code = secrets.token_urlsafe(32)
    session_id = secrets.token_urlsafe(12)
    setup_url = f"{origin}/onboarding/pair#code={code}"
    try:
        svg, data_uri = qr_svg_data_uri(setup_url)
    except ValueError:
        # An unusually long Host header must not break onboarding: fall back to
        # the short configured dashboard origin and bind the code to that.
        fallback = safe_base_url(_configured_origin())
        setup_url = f"{fallback}/onboarding/pair#code={code}"
        origin = fallback
        try:
            svg, data_uri = qr_svg_data_uri(setup_url)
        except ValueError:
            raise OnboardingError(
                "The setup link is too long to display. Restart LEO with a shorter dashboard address.",
                status_code=500, code="qr_failed",
            ) from None

    record: dict[str, Any] = {
        "code_hash": _code_hash(code),
        "session_id": session_id,
        "channel": channel,
        "state": STATE_STARTED,
        "created_at": _iso(now),
        "updated_at": _iso(now),
        "expires_epoch": now + ttl,
        "expires_at": _iso(now + ttl),
        "setup_origin": origin,
        "client": str(client)[:64],
        "device_link": device_link_status(channel),
        "transitions": [],
        "attempts": 0,
        "failure": None,
        "confirmed_at": None,
    }
    _transition(record, STATE_STARTED)
    _transition(record, STATE_AWAITING_SCAN)
    await _save_session(record)

    result = _public_status(record)
    result.update({
        "qr": {
            "format": "svg-data-uri",
            "svg": svg,
            "data_uri": data_uri,
            "payload": setup_url,
        },
        "setup_url": setup_url,
        "status_url": f"/onboarding/sessions/{session_id}",
        "instructions": [
            "Open the setup link on this computer, or scan the QR with a device that can reach it.",
            "Confirm the one-time code shown there. It expires quickly and works once.",
            "The code is not a WhatsApp credential and does not link a WhatsApp device.",
        ],
    })
    return result


async def redeem_setup_code(
    code: str,
    *,
    client: str = "local",
    origin: str | None = None,
) -> dict[str, Any]:
    """Consume a setup code exactly once and record a local confirmation."""
    _redeem_client_limiter.check(str(client)[:64], message="Too many setup attempts. Wait a minute and try again.")
    if not isinstance(code, str) or not _CODE_RE.fullmatch(code):
        raise OnboardingError("That setup code is invalid. Start a new setup.", status_code=403, code="invalid_code")

    digest = _code_hash(code)
    _redeem_code_limiter.check(digest, message="That setup code has been tried too many times. Start a new setup.")

    failure: OnboardingError | None = None
    status: dict[str, Any] | None = None
    # BEGIN IMMEDIATE: two simultaneous redemptions cannot both observe the
    # awaiting-scan state. Exactly one of them can move the session to paired.
    async with db.transaction():
        record = await _load_session(digest)
        if record is None:
            failure = OnboardingError(
                "That setup code is invalid or has already been used.",
                status_code=403, code="invalid_code",
            )
        else:
            record["attempts"] = int(record.get("attempts", 0)) + 1
            expires = float(record.get("expires_epoch", 0) or 0)
            if expires <= time.time():
                if record.get("state") in (STATE_STARTED, STATE_AWAITING_SCAN):
                    _transition(record, STATE_FAILED, reason="setup code expired")
                failure = OnboardingError(
                    "This setup code expired. Start a new setup.",
                    status_code=403, code="expired",
                )
                await _save_session(record)
            elif record.get("state") != STATE_AWAITING_SCAN:
                failure = OnboardingError(
                    "That setup code has already been used. Start a new setup.",
                    status_code=403, code="replayed",
                )
                await _save_session(record)
            elif origin and not origin_allowed(origin, base_url=str(record.get("setup_origin") or "")):
                failure = OnboardingError(
                    "This setup code belongs to a different origin.",
                    status_code=403, code="origin_mismatch",
                )
                await _save_session(record)
            elif record.get("client") and str(client) != str(record["client"]):
                failure = OnboardingError(
                    "This setup code belongs to a different session. Start a new setup.",
                    status_code=403, code="client_mismatch",
                )
                await _save_session(record)
            else:
                _transition(record, STATE_PAIRED)
                record["confirmed_at"] = _iso(time.time())
                await _save_session(record)
                status = _public_status(record)
    if failure is not None:
        raise failure
    assert status is not None
    return status


__all__ = [
    "OnboardingError",
    "BLOCKED_DEVICE_LINK_REASON",
    "STATE_STARTED",
    "STATE_AWAITING_SCAN",
    "STATE_PAIRED",
    "STATE_BLOCKED",
    "STATE_FAILED",
    "SETUP_TOKEN_TTL_SECONDS",
    "SETUP_TOKEN_MAX_SECONDS",
    "MAX_PENDING_SESSIONS",
    "safe_base_url",
    "origin_allowed",
    "qr_matrix",
    "qr_svg_data_uri",
    "field_guidance",
    "validate_values",
    "channel_guide",
    "device_link_status",
    "start_onboarding",
    "session_status",
    "redeem_setup_code",
]
