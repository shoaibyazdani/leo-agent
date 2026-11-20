"""Durable conversation history tools: paginated reads and one-call export.

The chat loop used to answer "make a file of this chat" by scavenging source
files, because no tool could read its own durable transcript. These tools close
that gap against the *real* SQLite store (:mod:`leo.db`):

* ``conversation_messages`` returns compact, cursor-paginated turns (role, text
  excerpt, tool names and tool errors) so a model can inspect prior context
  without guessing at message ids or scanning the repository.
* ``conversation_export`` reads the same rows with bound SQL parameters,
  redacts token-shaped values with the same sanitizer used by unprivileged
  HTTP responses, writes ONE bounded file below ``settings.paths.exports``
  and reports the real path, byte count and count. Names are stable and
  versioned; both the version count and the directory's total size are capped.

Both tools fail truthfully: an unknown conversation, an invalid cursor or an
unsafe output name raises instead of inventing a transcript or a path. The
``/export`` slash command uses :func:`write_export` directly, so the tool and
the command share identical redaction, bounds and path-confinement logic.
"""

from __future__ import annotations

import errno
import fcntl
import functools
import html
import json
import math
import os
import re
import stat
import threading
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping
from html.entities import html5 as _HTML5_ENTITIES

from .. import db
from ..contracts import RiskLevel, ToolSpec, utcnow

FORMATS: tuple[str, ...] = ("md", "json", "txt")
DEFAULT_LIMIT = 100
MAX_MESSAGES = 500
MAX_EXPORT_BYTES = 1_000_000
MAX_TEXT_CHARS = 4_000
MAX_EXCERPT_CHARS = 280
MAX_ITEMS = 10
REDACTED = "[REDACTED]"

# Exports are stable, versioned files: repeated calls must not grow the
# directory without bound (versions and total bytes are both capped).
MAX_EXPORT_VERSIONS = 10
MAX_EXPORT_TOTAL_BYTES = 5_000_000
EXPORT_FILE_MODE = 0o600

# Intrinsic token shapes. Operator-configured redaction patterns are applied in
# addition to these, so an export is still safe if leo.yaml drops a pattern.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Known provider prefixes are credentials even when they are glued to
        # another word (``ref_sk_live_…``, ``x_ghp_…``). A word boundary here
        # would let those spellings through, and the snake_case guard in the
        # generic scanner cannot tell them apart from ordinary config keys.
        r"sk(?:-[A-Za-z0-9_-]{4,}){2,}",
        r"gh[pousr]_[A-Za-z0-9_]{4,}",
        r"github_pat_[A-Za-z0-9_]{4,}",
        r"xox[baprs]-[A-Za-z0-9-]{10,}",
        r"xox[baprs][_-][A-Za-z0-9_-]{4,}",
        r"sk_(?:live|test|proj)_[A-Za-z0-9_]{4,}",
        r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
        r"(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16}",
        r"AIza[0-9A-Za-z_-]{35}",
        r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}",
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?"
        r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
        r"\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]{6,}",
    )
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _output_schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    """Output shapes are descriptive: extra result fields must not fail a valid call."""
    return {"type": "object", "properties": properties, "required": required}


def _settings(ctx: Any) -> Any:
    settings = ctx.get("settings") if isinstance(ctx, Mapping) else None
    if settings is None:
        raise RuntimeError("conversation tools require settings in the tool context")
    return settings


def _patterns(ctx: Any) -> list[re.Pattern[str]]:
    settings = _settings(ctx)
    raw = getattr(getattr(settings, "security", None), "redact_patterns", None) or []
    compiled: list[re.Pattern[str]] = []
    for item in raw:
        try:
            compiled.append(re.compile(str(item)))
        except re.error:
            continue
    return compiled


def redact_text(text: Any, patterns: Any = (), *, apply_intrinsic: bool = True) -> str:
    """Redact token-shaped values in arbitrary text before it is persisted.

    ``apply_intrinsic=False`` is only used by an explicitly unredacted export;
    history reads always keep the intrinsic token patterns enabled.
    """
    value = str(text)
    if apply_intrinsic:
        for pattern in _TOKEN_PATTERNS:
            value = pattern.sub(REDACTED, value)
        # Same rule as the shared preview sanitizer: a ``sha256:`` marker in
        # front of raw data is not a digest, even when glued after a word.
        found: set[str] = set()
        _SHA256_PREFIX_IN_TEXT.sub(lambda match: _collect_fake_digest(match, found), value)
        if found:
            value = _replace_known_tokens(value, found)
    for pattern in patterns or ():
        try:
            value = pattern.sub(REDACTED, value)
        except (TypeError, re.error):
            continue
    return value


# Preview URLs and ``token_id`` fields are bearer credentials, not ordinary
# token-shaped strings. The sanitizer below is shared by durable export bytes
# and every unprivileged HTTP response: an export must be no more revealing
# than the API view of the same conversation. The digest replacement is
# idempotent, so applying it twice (turn text, then rendered document) cannot
# leak a token or silently change a previously redacted correlation id.
#
# Credentials reach browser-readable surfaces in many shapes: a plain URL, the
# same URL with JSON/percent/unicode-escaped separators, a loopback host written
# as ``[::1]``, a ``token_id`` assignment with backticks or escaped quotes, a
# nested list/dict value under a ``token_id`` key, or a raw token copied into a
# field with no URL/assignment syntax (an Event ``correlation_id``, for
# instance). Matching therefore runs on both the original text and a decoded
# view, and the whole-value pass scrubs every high-confidence urlsafe token it
# finds -- not just capabilities still registered by the preview tool.
# HTML5 named entities are another spelling of the same separators (and browser
# readable), so match ``&colon;``/``&sol;`` and their ``&amp;``-nested forms too.
# Escape runs are bounded: a JSON escape has at most a few leading
# backslashes, and an unbounded ``\\+`` rescans a long backslash run from
# every offset (quadratic). Source code and tool output contain such runs,
# so ``\\+`` on a 100 KB run took ~70 s of CPU and stalled the event loop.
_SLASH = r"(?:/|\\{1,8}(?:/|u002[fF]|%2[fF])|%2[fF]|&(?:amp;?)*sol;?)"
_COLON = r"(?::|%3[aA]|\\{1,8}u003[aA]|&(?:amp;?)*colon;?)"
_QUOTE = r"(?:[\"'`]|\\{1,8}[\"']|\\{1,8}u002[27]|%2[27]|&(?:amp;?)*(?:quot|apos);?)"
_TOKEN_ATOM = r"(?:[A-Za-z0-9_-]|%[0-9A-Fa-f]{2}|\\{1,8}u[0-9A-Fa-f]{4})"
_HOST = (
    r"(?:localhost|127(?:\.\d{1,3}){3}|\[::1\]|::1|"
    r"\[::ffff:127(?:\.\d{1,3}){3}\]|0:0:0:0:0:0:0:1)"
)
_PREVIEW_URL_IN_TEXT = re.compile(
    r"(?P<prefix>http" + _COLON + _SLASH + r"{2}" + _HOST + _COLON + r"\d{1,5}"
    + _SLASH + r"preview(?:" + _SLASH + r"|" + _COLON + r"))"
    + r"(?P<token>" + _TOKEN_ATOM + r"{1,64})",
    re.IGNORECASE,
)
_TOKEN_FIELD_IN_TEXT = re.compile(
    r"(?<![\w-])(?:\\{0,8}[\"'`]?[A-Za-z0-9_-]*token_id[A-Za-z0-9_-]*\\{0,8}[\"'`]?)"
    + r"\\{0,8}\\s*(?:[:=]|&(?:amp;?)*colon;?)\\s*\\{0,8}(?:" + _QUOTE + r")?"
    + r"(?P<token>sha256:" + _TOKEN_ATOM + r"{1,64}|" + _TOKEN_ATOM + r"{1,64})",
    re.IGNORECASE,
)
# HTML entities are another encoding of the same characters. A viewer would
# happily decode them back into a token, so match that spelling as well. There
# is no safe fixed nesting depth: decode until the text truly stops changing.
# Every decoding step shortens the string (except adding one missing semicolon
# to a numeric reference, which happens once per reference), so a fixed point
# is always reached; the generous cap is only a defensive guard.
_MAX_DECODE_ROUNDS = 4096
# Each round scans the whole string with several regexes. Deeply nested
# encodings on a large string could otherwise burn seconds of CPU per
# string (and this runs on the request path), so the total characters
# scanned per string are capped. Normal text stabilizes after two rounds;
# a 500 KB string still gets eight full rounds.
_MAX_DECODE_CHARS = 4_000_000
# A ``sha256:`` marker followed by raw data is not a digest, no matter which
# word is glued in front of it. Scan it independently of token-field or generic
# token syntax so short and embedded spellings cannot escape the second pass.
_SHA256_PREFIX_IN_TEXT = re.compile(
    r"sha256:(?!sha256:)(?P<token>" + _TOKEN_ATOM + r"{1,64})", re.IGNORECASE
)
# A token copied into an unrelated field has no URL or assignment syntax.
# Bearer values are not always long mixed-case strings: all-alpha, all-digit,
# all-hex and 12-16 character secrets occur too. Collect those candidates and
# let :func:`_is_high_confidence_token` separate them from ordinary ids,
# timestamps, versions and digests.
_MIN_RAW_TOKEN_CHARS = 8
_GENERIC_TOKEN_CHARS = 12
_GENERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<token>[A-Za-z0-9_-]{12,})(?![A-Za-z0-9_-])"
)
_HEX_ONLY_RE = re.compile(r"^[0-9A-Fa-f]+$")
# Only the exact shape emitted by ``token_audit_id`` is a digest. A bare
# ``sha256:`` prefix followed by a raw capability was being preserved -- and
# replayed -- on every browser-readable surface.
_DIGEST_RE = re.compile(r"^sha256:[0-9A-Fa-f]{16}$", re.IGNORECASE)
# A complete SHA hash is a content fingerprint rather than a capability; keep
# it intact even though it is not the 16-hex audit id shape.
_FULL_HASH_RE = re.compile(r"^sha(?:256|1|512):[0-9A-Fa-f]{32,128}$", re.IGNORECASE)
# Compact timestamps (``YYYYMMDD[HHMMSS[mmm]]``, epoch seconds/milliseconds)
# and dotted versions are data, not credentials, even when long.
_TIMESTAMP_RE = re.compile(
    r"^(?:(?:19|20)\d{6}(?:\d{2}){0,5}|1\d{9,12}|\d{1,4}(?:[-_.]\d{1,4}){2,})$"
)
# ``task_12345678``-style correlators are ids, not capabilities.
_BENIGN_COMPOUND_ID_RE = re.compile(r"^(?:[A-Za-z]{1,12}[-_]){1,3}\d{1,16}$")
# Lowercase/uppercase snake_case values are error codes, config keys and
# tool names (``invalid_attachment``), not bearer credentials.
_SNAKE_CASE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")
# A path component can be random-looking (pytest temp dirs, hex ids) and still
# be a location. Only unmistakable capability spellings override that context.
_KNOWN_TOKEN_PREFIX_RE = re.compile(
    r"^(?:sk-|sk_(?:live|test|proj)_|gh[pousr]_|github_pat_|xox[baprs][_-]|"
    r"AKIA|ASIA|AIDA|AROA|AIza|eyJ)",
    re.IGNORECASE,
)
# The same prefixes can appear *inside* an otherwise snake_case value
# (``ref_sk_live_…``, ``x_ghp_…``). Check those before the benign-id exemptions.
_KNOWN_TOKEN_PREFIX_EMBEDDED_RE = re.compile(
    _KNOWN_TOKEN_PREFIX_RE.pattern[1:], re.IGNORECASE
)
_UUID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
# Durable ids use ``<prefix>_<uuid4().hex>`` (see ``contracts.new_id``). They
# are correlators, not capabilities, and must survive in ids and file names.
_STRUCTURED_ID_RE = re.compile(
    r"^(?:conv|msg|task|run|agent|chat|session|skill|memory|audit|event|"
    r"attachment|call|user)_[0-9A-Fa-f]{16,}$",
    re.IGNORECASE,
)
_FILE_EXTENSION_RE = re.compile(r"^\.[A-Za-z0-9]{1,5}(?:\b|$)")
_UNICODE_ESCAPE_RE = re.compile(r"\\{1,8}u([0-9A-Fa-f]{4})")
_PERCENT_ESCAPE_RE = re.compile(r"%([0-9A-Fa-f]{2})")
_QUOTE_ESCAPE_RE = re.compile(r"\\{1,8}([\"'`])")
# Browsers also accept numeric character references without the trailing
# semicolon when the next character cannot extend the number. Normalize those
# before unescaping so ``&#105`` (or ``&#x69``) is decoded like ``&#105;``.
_SEMICOLONLESS_DEC_RE = re.compile(r"&#(\d{1,7})(?![0-9;])")
_SEMICOLONLESS_HEX_RE = re.compile(r"&#[xX]([0-9A-Fa-f]{1,8})(?![0-9A-Fa-f;])")


def _named_entity_forms() -> dict[str, tuple[str, ...]]:
    """Named HTML spellings of the characters our tokens/URLs use."""
    collected: dict[str, list[str]] = {}
    for name, character in _HTML5_ENTITIES.items():
        if len(character) != 1:
            continue
        forms = collected.setdefault(character, [])
        spelling = "&" + name
        if spelling not in forms:
            forms.append(spelling)
    return {
        character: tuple(sorted(forms, key=len)[:6])
        for character, forms in collected.items()
    }


_NAMED_ENTITY_FORMS = _named_entity_forms()

# A digest is only non-replayable if *this process* produced it (or it belongs
# to a live/recently seen token). A caller can otherwise label any 16-hex value
# ``sha256:`` and have it preserved as if it were an audit correlation id.
_AUDIT_DIGEST_LIMIT = 4096
_RECENT_TOKEN_LIMIT = 512
_AUDIT_DIGEST_LOCK = threading.Lock()
_AUDIT_DIGESTS: set[str] = set()
_AUDIT_DIGEST_ORDER: list[str] = []
_RECENT_TOKENS: list[str] = []


def _install_audit_digest_registry() -> None:
    """Register every audit id this app emits, not just sanitizer-generated ones.

    Preview audits and the browser grant call ``token_audit_id`` directly. A
    bounded process-local registry lets the sanitizer tell those genuine ids
    from a caller-supplied ``sha256:`` + 16 hex string. ``functools.wraps``
    keeps the public function's name and documentation intact.
    """
    try:
        from . import preview_tools
    except Exception:
        return
    original = preview_tools.token_audit_id
    if getattr(original, "_leo_registers_audit_digest", False):
        return

    @functools.wraps(original)
    def registering(token: str | None) -> str:
        digest = original(token)
        return _register_audit_digest(digest) if digest else digest

    registering._leo_registers_audit_digest = True  # type: ignore[attr-defined]
    preview_tools.token_audit_id = registering


def _register_audit_digest(digest: str) -> str:
    """Remember a digest produced by this app's own sanitizer."""
    key = str(digest).lower()
    if not _DIGEST_RE.fullmatch(key):
        return digest
    with _AUDIT_DIGEST_LOCK:
        if key not in _AUDIT_DIGESTS:
            _AUDIT_DIGESTS.add(key)
            _AUDIT_DIGEST_ORDER.append(key)
            while len(_AUDIT_DIGEST_ORDER) > _AUDIT_DIGEST_LIMIT:
                _AUDIT_DIGESTS.discard(_AUDIT_DIGEST_ORDER.pop(0))
    return digest


def _remember_token(token: str) -> None:
    """Keep a bounded set of raw tokens whose audit ids may appear later."""
    if not token:
        return
    with _AUDIT_DIGEST_LOCK:
        if token in _RECENT_TOKENS:
            return
        _RECENT_TOKENS.append(token)
        while len(_RECENT_TOKENS) > _RECENT_TOKEN_LIMIT:
            _RECENT_TOKENS.pop(0)
    _preview_token_digest(token)


def _preview_token_digest(value: str) -> str:
    from .preview_tools import token_audit_id  # local: avoids import cycle

    digest = token_audit_id(value)
    if digest and digest != value:
        _register_audit_digest(digest)
        return digest
    return value


def _is_digest(value: str) -> bool:
    """Only an audit id actually produced by this app is a non-replayable digest.

    The shape alone proves nothing: ``sha256:0123456789abcdef`` is easy to
    invent and must be scrubbed like any other planted credential. Digests are
    accepted when the sanitizer generated them from a token, when the token is
    live in the preview table, or when the token was seen recently in this
    process. Uppercase spellings of a genuine id compare case-insensitively.
    """
    if not _DIGEST_RE.fullmatch(value):
        return False
    key = value.lower()
    with _AUDIT_DIGEST_LOCK:
        if key in _AUDIT_DIGESTS:
            return True
        recent = tuple(_RECENT_TOKENS)
    if not recent:
        return False
    from .preview_tools import token_audit_id  # local: avoids import cycle

    return any(token_audit_id(token).lower() == key for token in recent)


def _decode_text(text: str) -> str:
    """Canonical view for matching: undo JSON/percent/quote/HTML escapes.

    Entities can be nested (``&amp;#102;``), and decoding one layer can expose
    another (``%26%23102%3B``), so repeat until the text stops changing. The
    canonical view is only used for matching; the original spelling is scrubbed
    through :func:`_token_variant_re` rather than silently rewritten.
    """

    def char(code: int) -> str:
        return chr(code) if 0 <= code <= 0x10FFFF else ""

    value = text
    spent = 0
    for _ in range(_MAX_DECODE_ROUNDS):
        spent += len(value)
        if spent > _MAX_DECODE_CHARS:
            # Bounded work: a pathological nesting depth must not stall the
            # caller. The raw-text token scanners still run on the original.
            break
        # Semicolon-less numeric references become well-formed references, then
        # html.unescape resolves named/numeric entities (one nested ``&amp;``
        # layer per round). Percent/JSON escapes can expose another entity, so
        # the whole normalization repeats until stable.
        decoded = _SEMICOLONLESS_DEC_RE.sub(r"&#\1;", value)
        decoded = _SEMICOLONLESS_HEX_RE.sub(r"&#x\1;", decoded)
        decoded = html.unescape(decoded)
        decoded = _UNICODE_ESCAPE_RE.sub(lambda match: char(int(match.group(1), 16)), decoded)
        decoded = _PERCENT_ESCAPE_RE.sub(lambda match: char(int(match.group(1), 16)), decoded)
        decoded = _QUOTE_ESCAPE_RE.sub(lambda match: match.group(1), decoded)
        if decoded == value:
            break
        value = decoded
    return value


@lru_cache(maxsize=1024)
def _token_variant_re(token: str, *, glued: bool = False) -> re.Pattern[str]:
    """Match a token written plainly or with common per-character escapes.

    ``glued`` drops the leading word guard for a known marker such as
    ``sha256:<raw>`` that a caller may append to another word.
    """
    guard = "" if glued else r"(?<![A-Za-z0-9_-])"
    atoms: list[str] = []
    for character in token:
        code = ord(character)
        forms = [re.escape(character)]
        if code <= 0xFF:
            forms.append("%" + f"{code:02X}")
        forms.extend((r"\\u" + f"{code:04x}", r"\\u" + f"{code:04X}"))
        if character.isalnum() or character in "_-":
            # Decimal/hex HTML entities, optionally double-encoded through
            # ``&amp;``. The whole compiled pattern is case-insensitive.
            # Semicolon-less references are legal in browsers when no digit
            # follows, so accept those spellings as well.
            forms.extend((
                r"&#0*" + str(code) + r";",
                r"&#[xX]0*" + format(code, "x") + r";",
                r"&#0*" + str(code) + r"(?![0-9;])",
                r"&#[xX]0*" + format(code, "x") + r"(?![0-9A-Fa-f;])",
                # Each extra encoding layer replaces the entity's leading
                # ``&`` with ``&amp;`` (``&amp;#101;``, ...).
                r"&(?:amp;?)+#0*" + str(code) + r";",
                r"&(?:amp;?)+#[xX]0*" + format(code, "x") + r";",
                r"&(?:amp;?)+#0*" + str(code) + r"(?![0-9;])",
                r"&(?:amp;?)+#[xX]0*" + format(code, "x") + r"(?![0-9A-Fa-f;])",
            ))
        for named in _NAMED_ENTITY_FORMS.get(character, ()):
            # e.g. ``&colon;``/``&sol;``/``&lowbar;`` and their nested forms.
            forms.append(re.escape(named))
            forms.append(r"&(?:amp;?)+" + re.escape(named[1:]))
        atoms.append("(?:" + "|".join(forms) + ")")
    return re.compile(guard + "".join(atoms) + r"(?![A-Za-z0-9_-])", re.IGNORECASE)


def _live_preview_tokens() -> set[str]:
    """Raw tokens still registered by the preview tool, if it is loaded.

    Live capabilities are scrubbed even when short; the generic pass below
    covers long tokens that have expired or were never registered at all.
    """
    try:
        from .preview_tools import _STATE  # local: avoids import cycle

        with _STATE.lock:
            tokens = {token for token in _STATE.previews if len(token) >= _MIN_RAW_TOKEN_CHARS}
        # A live token's audit id may already be stored in an event or export;
        # remember it as genuinely app-generated before the sanitizer runs.
        for token in tokens:
            _remember_token(token)
        return tokens
    except Exception:
        return set()


def _collect_preview_url(match: re.Match[str], found: set[str]) -> str:
    token = _decode_text(match.group("token"))
    if len(token) >= _MIN_RAW_TOKEN_CHARS:
        found.add(token)
    return f"{match.group('prefix')}{REDACTED}"


def _collect_token_field(match: re.Match[str], found: set[str]) -> str:
    token = _decode_text(match.group("token"))
    if _is_digest(token):
        return match.group(0)  # already a salted digest, not a capability
    if len(token) >= _MIN_RAW_TOKEN_CHARS:
        found.add(token)
    whole = match.group(0)
    start, end = match.span("token")
    offset = match.start()
    return whole[: start - offset] + _preview_token_digest(token) + whole[end - offset:]


def _collect_fake_digest(match: re.Match[str], found: set[str]) -> str:
    """Collect ``sha256:<raw>`` even when the marker is glued to a word."""
    whole = match.group(0)
    if _is_digest(whole) or _FULL_HASH_RE.fullmatch(whole):
        return whole  # a registered audit id or a complete hash stays stable
    found.add(whole)
    return whole  # replaced in the second pass with the glued-aware variant


def _entropy(value: str) -> float:
    """Shannon entropy in bits/character; a rough randomness signal."""
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _case_transitions(value: str) -> int:
    return sum(
        1 for first, second in zip(value, value[1:])
        if first.isupper() != second.isupper() or first.islower() != second.islower()
    )


def _is_high_confidence_token(token: str) -> bool:
    """Distinguish a bearer-shaped value from an id, timestamp or hash.

    Capabilities do not always mix letters and digits: all-alpha, all-digit,
    all-hex and 12-16 character values are all real shapes. Conversely, most
    benign values are structured (``conv_<hex>``, UUIDs, ISO timestamps, dotted
    versions) or low-entropy words and paths. Err on the side of scrubbing a
    random-looking value: over-redaction costs a little readability, while a
    missed raw capability can be replayed forever.
    """
    if len(token) < _GENERIC_TOKEN_CHARS:
        return False
    if _is_digest(token) or _UUID_RE.fullmatch(token) or _STRUCTURED_ID_RE.fullmatch(token):
        return False
    # Provider capabilities are deliberately snake_case (``ghp_…``,
    # ``github_pat_…``, ``sk_live_…``). The known-prefix check must come before
    # the generic snake_case/benign-id exemptions, or those credentials are
    # mistaken for config keys and leak into unprivileged HTTP responses.
    # Match embedded prefixes too: ``ref_ghp_…`` is a credential, not a
    # harmless snake_case identifier.
    if _KNOWN_TOKEN_PREFIX_RE.match(token) or _KNOWN_TOKEN_PREFIX_EMBEDDED_RE.search(token):
        return True
    if _TIMESTAMP_RE.fullmatch(token) or _BENIGN_COMPOUND_ID_RE.fullmatch(token):
        return False
    if "_" in token and (token.islower() or token.isupper()) and _SNAKE_CASE_ID_RE.fullmatch(token):
        return False

    upper = sum(1 for char in token if char.isupper())
    digits = sum(1 for char in token if char.isdigit())
    entropy = _entropy(token)
    transitions = _case_transitions(token)

    if _HEX_ONLY_RE.fullmatch(token):
        # Unstructured hex is not automatically a hash: short hex secrets are
        # common. Recognizable hash widths with low entropy stay intact; the
        # 16-hex digest body is excluded by its ``sha256:`` context in the
        # caller. Random hex (including 32 chars) is scrubbed.
        if len(token) in (32, 40, 64) and entropy < 3.6:
            return False
        return entropy >= 3.0
    if digits == len(token):
        # Long non-timestamp digit runs are bearer-shaped; repeated filler
        # digits (``000000000000``) are not.
        return entropy >= 2.5
    if digits == 0:
        if upper >= 3 and transitions >= 3:
            return entropy >= 3.2  # e.g. AbCdEfGhIjKl
        if len(token) >= 18:
            return entropy >= 3.4  # long random lowercase, e.g. 26 letters
        return len(token) >= 16 and entropy >= 3.7
    # A capital inside a digit-bearing value (``shortToken1234``) is a strong
    # capability signal even when there is only one of them.
    if upper and transitions >= 2:
        return entropy >= 3.1
    if len(token) >= 24:
        return entropy >= 3.4
    return len(token) >= 14 and entropy >= 3.5


def _path_component_is_credential(token: str) -> bool:
    """Whether a path segment is unmistakably a bearer capability."""
    if _KNOWN_TOKEN_PREFIX_RE.match(token) or _KNOWN_TOKEN_PREFIX_EMBEDDED_RE.search(token):
        return True
    upper = sum(1 for char in token if char.isupper())
    digits = sum(1 for char in token if char.isdigit())
    return bool(upper) and (bool(digits) or _case_transitions(token) >= 3)


def _collect_generic_token(match: re.Match[str], found: set[str], text: str) -> str:
    token = match.group("token")
    start, end = match.span("token")
    before = text[max(0, start - 24):start]
    after = text[end:end + 24]
    if start >= 7 and before.lower().endswith("sha256:"):
        whole = text[start - 7:end]
        if _is_digest(whole) or _FULL_HASH_RE.fullmatch(whole):
            return token  # an already-salted digest/hash must stay stable
        # ``sha256:`` in front of a raw capability is not a digest. Collect
        # the whole spelling so the second pass replaces it in one step.
        found.add(whole)
        return token
    if _UUID_RE.fullmatch(token) or _STRUCTURED_ID_RE.fullmatch(token):
        return token
    # A plain file-system component is a location, not a credential; temp dirs
    # and stable ids are often random-looking. A capability planted as a path
    # segment still has to go, so recognize unmistakable credential spellings
    # (known prefixes, or mixed-case high-entropy values) in that context.
    in_path = before.endswith(("/", "\\")) and (
        after.startswith(("/", "\\")) or _FILE_EXTENSION_RE.match(after)
    )
    if in_path and not _path_component_is_credential(token):
        return token
    if _is_high_confidence_token(token):
        found.add(token)
    return token


# The per-character variant pattern is huge (each byte gets ~15 encoding
# alternatives). Building and running one per candidate over every string is
# quadratic-ish in practice: a tool-heavy conversation contributed 103
# candidates and burned seconds per load. Plain spellings for every candidate
# go through one combined literal alternation (C-speed). The variant engine
# only runs for a candidate whose *decoded* form actually occurs in this
# string -- an encoded spelling cannot exist in a string with no escape
# markers, and a marker-bearing string is decoded once and checked per
# candidate. The guarantee is unchanged; the scanning is bounded.
_PLAIN_TOKEN_CACHE = 128
# Markers that make an encoded spelling possible: percent escapes, unicode
# escapes and HTML entities (named or nested via ``&amp;``). A plain
# backslash alone (JSON ``\"`` sequences) cannot hide a token -- JSON
# escaping is handled by ``_QUOTE_ESCAPE_RE`` only in combination with a
# decoded entity, and quoted JSON strings dominate tool output, so treating
# every backslash as a marker decoded megabytes of text for nothing.
_ESCAPE_MARKERS = ("%", "\\u", "&")


@lru_cache(maxsize=_PLAIN_TOKEN_CACHE)
def _plain_token_re(tokens: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted(set(tokens), key=len, reverse=True)
    return re.compile("|".join(re.escape(token) for token in ordered))


def _replace_known_tokens(text: str, found: set[str]) -> str:
    if not found:
        return text
    combined = _plain_token_re(tuple(found))
    text = combined.sub(lambda match: _preview_token_digest(match.group(0)), text)
    if not any(marker in text for marker in _ESCAPE_MARKERS):
        return text
    decoded = _decode_text(text)
    # One combined literal scan of the decoded view finds every candidate that
    # occurs here, however it was spelled. Checking candidates one by one made
    # pass two O(candidates x strings): a full conversation accumulated
    # thousands of candidates and the payload took over a minute to sanitize.
    present = {match.group(0) for match in combined.finditer(decoded)}
    for known in sorted(present, key=len, reverse=True):
        # ``sha256:`` is a known marker, not an identifier boundary: it may be
        # glued after a word (``refsha256:…``), so drop the leading guard for
        # those spellings while keeping genuine registered digests intact.
        glued = known.lower().startswith("sha256:")
        text = _token_variant_re(known, glued=glued).sub(_preview_token_digest(known), text)
    return text


def _redact_string(text: str, found: set[str], *, replace_known: bool) -> str:
    if replace_known:
        return _replace_known_tokens(text, found)
    current = _PREVIEW_URL_IN_TEXT.sub(lambda match: _collect_preview_url(match, found), text)
    current = _TOKEN_FIELD_IN_TEXT.sub(lambda match: _collect_token_field(match, found), current)
    current = _SHA256_PREFIX_IN_TEXT.sub(lambda match: _collect_fake_digest(match, found), current)
    _GENERIC_TOKEN_RE.sub(lambda match: _collect_generic_token(match, found, current), current)
    # The same credentials may be hidden behind percent/unicode escapes or
    # doubled quotes. Scan a decoded view as well so encoding cannot bypass
    # matching; the second pass scrubs the original spelling via variants.
    decoded = _decode_text(text)
    _PREVIEW_URL_IN_TEXT.sub(lambda match: _collect_preview_url(match, found), decoded)
    _TOKEN_FIELD_IN_TEXT.sub(lambda match: _collect_token_field(match, found), decoded)
    _SHA256_PREFIX_IN_TEXT.sub(lambda match: _collect_fake_digest(match, found), decoded)
    _GENERIC_TOKEN_RE.sub(lambda match: _collect_generic_token(match, found, decoded), decoded)
    return current


def _redact_token_id(item: Any, found: set[str], *, replace_known: bool) -> Any:
    """A ``token_id`` member is a credential at any depth or shape."""
    if isinstance(item, str):
        if replace_known:
            return _redact_string(item, found, replace_known=True)
        if _is_digest(item):
            return item
        token = _decode_text(item)
        if token and len(token) >= _MIN_RAW_TOKEN_CHARS:
            found.add(token)
        return _preview_token_digest(token or item)
    if isinstance(item, Mapping):
        return {
            _redact_value(key, found, replace_known=replace_known):
                _redact_token_id(value, found, replace_known=replace_known)
            for key, value in item.items()
        }
    if isinstance(item, (list, tuple, set)):
        return [_redact_token_id(value, found, replace_known=replace_known) for value in item]
    return item


def _redact_value(value: Any, found: set[str], *, replace_known: bool) -> Any:
    if isinstance(value, str):
        return _redact_string(value, found, replace_known=replace_known)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _redact_value(key, found, replace_known=replace_known) if isinstance(key, str) else key
            if isinstance(key, str) and "token_id" in str(safe_key).lower():
                result[safe_key] = _redact_token_id(item, found, replace_known=replace_known)
            else:
                result[safe_key] = _redact_value(item, found, replace_known=replace_known)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_redact_value(item, found, replace_known=replace_known) for item in value]
    return value


def _apply_intrinsic_patterns(value: Any) -> Any:
    """Apply the same intrinsic credential patterns used by durable exports.

    Doing this here (and not only in ``redact_text``) keeps browser-readable
    responses and exported bytes equally strict: a known-prefix capability is
    redacted on /conversations, /events, /status and SSE just as it is in a
    written export, even when it is too short for the generic heuristic.
    """
    if isinstance(value, str):
        result = value
        for pattern in _TOKEN_PATTERNS:
            result = pattern.sub(REDACTED, result)
        return result
    if isinstance(value, Mapping):
        return {
            (_apply_intrinsic_patterns(key) if isinstance(key, str) else key):
                _apply_intrinsic_patterns(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_apply_intrinsic_patterns(item) for item in value]
    return value


def redact_preview_content(value: Any) -> Any:
    r"""Copy any string/mapping/list shape with preview credentials removed.

    Preview URLs (including slash/quote/percent/unicode/HTML-entity escapes,
    mixed-case hosts and IPv6 loopback) become inert ``[REDACTED]`` paths;
    ``token_id`` values and keys are digested at any depth, including lists
    and maps; and any other high-confidence token is scrubbed in every string
    field, even when it is not registered by the preview tool. Intrinsic
    token shapes (including snake_case provider prefixes) are scrubbed with
    the same patterns used for exports. Audit digests produced by this app
    stay intact; an invented ``sha256:`` value does not. The input is never
    mutated: tools and the event bus keep the genuine capability.
    """
    found = _live_preview_tokens()
    once = _redact_value(_apply_intrinsic_patterns(value), found, replace_known=False)
    if not found:
        return once
    return _redact_value(once, found, replace_known=True)


def redact_preview_text(text: str) -> str:
    """String-shaped view of :func:`redact_preview_content`."""
    result = redact_preview_content(str(text))
    return result if isinstance(result, str) else str(result)


def redact_message_content(value: Any) -> Any:
    """Compatibility name used by the API feed sanitizer."""
    return redact_preview_content(value)


def _clip(text: Any, limit: int) -> tuple[str, bool]:
    value = str(text)
    if len(value) <= limit:
        return value, False
    return value[:limit] + "\n… [truncated]", True


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _add(items: list[str], value: Any, limit: int = 160) -> None:
    text, _ = _clip(str(value).strip(), limit)
    if text and text not in items:
        items.append(text)


def _tool_details(row: Any) -> tuple[list[str], list[str]]:
    """Extract bounded tool names and errors from a durable message row."""
    metadata = _object(row["metadata"])
    content = _object(row["content"])
    names: list[str] = []
    errors: list[str] = []
    for source in (metadata, content):
        for key in ("name", "tool"):
            value = source.get(key)
            if isinstance(value, str):
                _add(names, value, 120)
        calls = source.get("tool_calls")
        if isinstance(calls, list):
            for call in calls[:MAX_ITEMS]:
                if not isinstance(call, Mapping):
                    continue
                name = call.get("name") or _object(call.get("function")).get("name")
                if isinstance(name, str):
                    _add(names, name, 120)
        error = source.get("error")
        if isinstance(error, str):
            _add(errors, error)
        elif error is not None:
            _add(errors, error)
        reason = source.get("reason")
        if source.get("not_executed") and reason:
            _add(errors, f"not executed: {reason}")
        nested = _object(source.get("output"))
        nested_error = nested.get("error")
        if isinstance(nested_error, str):
            _add(errors, nested_error)
        if source.get("success") is False and not errors:
            _add(errors, "tool call reported failure")
    return names[:MAX_ITEMS], errors[:MAX_ITEMS]


def _turn(row: Any, patterns: Any, *, excerpt: bool, redact: bool = True) -> dict[str, Any]:
    text, truncated = _clip(row["content"], MAX_EXCERPT_CHARS if excerpt else MAX_TEXT_CHARS)
    names, errors = _tool_details(row)
    text = redact_text(text, patterns, apply_intrinsic=redact)
    clean_names = [redact_text(name, patterns, apply_intrinsic=redact) for name in names]
    clean_errors = [redact_text(error, patterns, apply_intrinsic=redact) for error in errors]
    if redact:
        # Preview URLs are not covered by the intrinsic token patterns. Use
        # the same sanitizer the HTTP responses use, not a look-alike.
        text = redact_preview_text(text)
        clean_names = [redact_preview_text(name) for name in clean_names]
        clean_errors = [redact_preview_text(error) for error in clean_errors]
    return {
        "id": int(row["id"]),
        "role": str(row["role"]),
        "created_at": str(row["created_at"]),
        "text": text,
        # Compatible spellings: callers reading compact history naturally look
        # for any of these, and all carry the same bounded, redacted value.
        "excerpt": text,
        "content": text,
        "truncated": truncated,
        "tools": clean_names,
        "tool_names": clean_names,
        "errors": clean_errors,
        "tool_errors": clean_errors,
    }


def _limit(value: Any, field: str, default: int, maximum: int) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer")
    if not 1 <= raw <= maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return raw


def _cursor(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("after_id must be a message id or null")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError("after_id must be a non-negative message id")


async def _load(conversation_id: Any, after_id: Any, limit: int) -> tuple[str, list[Any], bool, str | None]:
    if conversation_id is None:
        row = await db.fetchone(
            "SELECT id FROM conversations ORDER BY updated_at DESC, rowid DESC LIMIT 1"
        )
        if row is None:
            raise ValueError("no conversations exist; pass conversation_id")
        conversation_id = row["id"]
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string")
    conversation_id = conversation_id.strip()
    found = await db.fetchone("SELECT id FROM conversations WHERE id=?", (conversation_id,))
    if found is None:
        raise ValueError(f"unknown conversation: {conversation_id}")

    cursor = _cursor(after_id)
    rows = await db.fetchall(
        "SELECT id, role, content, metadata, created_at FROM messages "
        "WHERE conversation_id=? AND id>? ORDER BY id ASC LIMIT ?",
        (conversation_id, cursor, limit + 1),
    )
    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = str(page[-1]["id"]) if has_more and page else None
    return conversation_id, page, has_more, next_cursor


async def conversation_messages(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Read compact, redacted turns with a stable ``after_id`` cursor."""
    limit = _limit(args.get("limit"), "limit", DEFAULT_LIMIT, MAX_MESSAGES)
    conversation_id, rows, has_more, next_cursor = await _load(
        args.get("conversation_id"), args.get("after_id"), limit
    )
    patterns = _patterns(ctx)
    turns = [_turn(row, patterns, excerpt=True) for row in rows]
    return {
        "success": True,
        "conversation_id": conversation_id,
        "count": len(turns),
        "messages": turns,
        "turns": turns,
        "has_more": has_more,
        "next_cursor": next_cursor,
        "redacted": True,
    }


def _format(value: Any) -> str:
    fmt = "md" if value is None else str(value).strip().lower()
    if fmt not in FORMATS:
        raise ValueError(f"invalid format {value!r}; expected one of {list(FORMATS)}")
    return fmt


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value)[:40].strip("-")
    return slug or "conversation"


def _assert_no_symlink_chain(root: Path, anchor: Path) -> None:
    """Reject a link at the configured root or anywhere in its local chain.

    ``Path.resolve()`` silently follows a link planted before validation, so
    every component below the trusted anchor is checked with ``lstat`` first.
    System aliases above that anchor (for example ``/var`` on macOS) are not
    part of the configured chain and are deliberately out of scope.
    """
    try:
        anchor_info = anchor.lstat()
    except FileNotFoundError:
        return  # the whole chain will be re-checked after mkdir
    if stat.S_ISLNK(anchor_info.st_mode):
        raise PermissionError("exports root chain contains a symlink")
    try:
        relative = root.relative_to(anchor)
    except ValueError:
        relative = Path(root.name)
    cursor = anchor
    for part in relative.parts:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise PermissionError("exports root chain contains a symlink")


def _exports_root(ctx: Any) -> tuple[Path, tuple[int, int]]:
    """Validate the configured exports root *before* resolving or writing."""
    settings = _settings(ctx)
    configured = Path(settings.resolve(settings.paths.exports)).expanduser()
    try:
        anchor = Path(settings.root).expanduser().resolve()
        configured.relative_to(anchor)  # a workspace-relative path: check all of it
    except (ValueError, OSError, RuntimeError):
        anchor = configured.parent
    _assert_no_symlink_chain(configured, anchor)
    try:
        configured.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot create exports directory: {configured}") from exc
    _assert_no_symlink_chain(configured, anchor)
    expected = _root_identity(configured)
    try:
        resolved = configured.resolve(strict=True)
        if (os.lstat(resolved).st_dev, os.lstat(resolved).st_ino) != expected:
            raise PermissionError("exports root changed during resolution")
    except PermissionError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve exports directory: {configured}") from exc
    return resolved, expected


class ExportStorageError(RuntimeError):
    """An export cannot be written within the configured storage budget."""


def _target_path(root: Path, name: str | None, fmt: str, conversation_id: str) -> Path:
    """Pick a stable, versioned export name directly under ``root``.

    Re-exporting one conversation reuses its base name and advances a bounded
    version instead of minting unlimited fresh UUID files. The returned path is
    intentionally *not* resolved: it is a plain child name, so the writer can
    create it atomically relative to a directory descriptor.
    """
    if name is None:
        base = f"conversation-{_slug(conversation_id)}"
    else:
        text = str(name).strip()
        if not text or text in (".", ".."):
            raise ValueError("out must be a file name, not a path")
        if "/" in text or "\\" in text or "\x00" in text:
            raise ValueError("out must be a file name without path separators")
        if not _NAME_RE.match(text):
            raise ValueError("out must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}")
        suffix = f".{fmt}"
        base = text[: -len(suffix)] if text.lower().endswith(suffix) else text
    for version in range(1, MAX_EXPORT_VERSIONS + 1):
        version_suffix = "" if version == 1 else f"-{version}"
        candidate = root / f"{base}{version_suffix}.{fmt}"
        if candidate.is_symlink():
            raise PermissionError("refusing to write through a symlinked export target")
        if not candidate.exists():
            return candidate
    raise ExportStorageError(
        f"export version limit reached for {base!r} ({MAX_EXPORT_VERSIONS} files)"
    )


def _enforce_export_budget(root: Path, incoming: int) -> None:
    """Bound the whole exports directory, not just a single file."""
    used = 0
    for item in root.rglob("*"):
        try:
            if item.name == _EXPORT_LOCK_NAME or item.is_symlink() or not item.is_file():
                continue
            used += item.stat().st_size
        except OSError:
            continue
    if used + incoming > MAX_EXPORT_TOTAL_BYTES:
        raise ExportStorageError(
            f"export storage budget exceeded: {used} existing + {incoming} new > "
            f"{MAX_EXPORT_TOTAL_BYTES} bytes"
        )


# The thread lock avoids local races around lock-file creation; the OS file
# lock below is the real serialization point. It is held across target
# selection, aggregate budget accounting and creation, so *separate processes*
# cannot each observe the same free bytes and versions and multiply the
# configured budget by the number of writers.
_EXPORT_WRITE_LOCK = threading.RLock()
_EXPORT_LOCK_NAME = ".export.lock"


@contextmanager
def _export_lock(root: Path) -> Iterator[None]:
    """Serialize exports through an flock on a private file in the root."""
    lock_path = root / _EXPORT_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise PermissionError("refusing a symlinked exports lock") from exc
        raise
    try:
        opened = os.fstat(fd)
        named = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_ISLNK(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise PermissionError("exports lock is not a trusted private file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _root_identity(root: Path) -> tuple[int, int]:
    """Capture the validated exports root's inode without following a link."""
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError("exports root is not a directory")
    return (info.st_dev, info.st_ino)


def _open_exports_root(root: Path, expected: tuple[int, int]) -> int:
    """Hold a verified directory fd for the whole create/write sequence.

    ``O_DIRECTORY|O_NOFOLLOW`` refuses a symlinked root. Comparing the opened
    fd against both a fresh ``lstat`` of the path and the identity captured
    during validation closes both swap windows: before the open (identity
    changed) and after it (the held fd still points at the validated inode).
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(root, flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise PermissionError("exports root is not a trusted directory") from exc
        raise
    try:
        opened = os.fstat(fd)
        named = os.lstat(root)
        if not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(named.st_mode):
            raise PermissionError("exports root is not a directory")
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise PermissionError("exports root changed during open")
        if (opened.st_dev, opened.st_ino) != expected:
            raise PermissionError("exports root changed since validation")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _write_export_file(root_fd: int, target: Path, data: bytes) -> None:
    """Create the final path relative to the verified root fd.

    Validation in ``_target_path`` can be raced by a symlink swap. Opening by
    name beneath the held directory descriptor with O_EXCL|O_NOFOLLOW removes
    the gap: the descriptor refers to the freshly created regular file itself,
    and fstat/lstat must agree on the same single-link inode before any bytes
    are written.
    """
    name = target.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, EXPORT_FILE_MODE, dir_fd=root_fd)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ELOOP):
            existing = os.lstat(name, dir_fd=root_fd)
            if stat.S_ISLNK(existing.st_mode):
                raise PermissionError(
                    "refusing to write through a symlinked export target"
                ) from exc
        raise
    with os.fdopen(fd, "wb") as handle:
        created = os.fstat(handle.fileno())
        named = os.lstat(name, dir_fd=root_fd)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise PermissionError("refusing to write through a non-regular export target")
        if (
            stat.S_ISLNK(named.st_mode)
            or (created.st_dev, created.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise PermissionError("export target changed during creation")
        handle.write(data)


def _render(fmt: str, conversation_id: str, turns: list[dict[str, Any]], redacted: bool,
            truncated: bool) -> str:
    header = {
        "conversation_id": conversation_id,
        "exported_at": utcnow(),
        "format": fmt,
        "redacted": redacted,
        "count": len(turns),
        "truncated": truncated,
    }
    if fmt == "json":
        return json.dumps({**header, "messages": turns}, ensure_ascii=False, indent=2)
    if fmt == "txt":
        lines = [
            f"conversation_id: {conversation_id}",
            f"exported_at: {header['exported_at']}",
            f"redacted: {redacted}",
            f"messages: {len(turns)}" + (" (truncated)" if truncated else ""),
            "",
        ]
        for turn in turns:
            lines.append(f"[{turn['id']}] {turn['role']} {turn['created_at']}")
            if turn["tools"]:
                lines.append("tools: " + ", ".join(turn["tools"]))
            if turn["errors"]:
                lines.append("errors: " + " | ".join(turn["errors"]))
            lines.append(turn["text"])
            lines.append("")
        return "\n".join(lines)
    lines = [
        "# Conversation export",
        "",
        f"- conversation_id: `{conversation_id}`",
        f"- exported_at: {header['exported_at']}",
        f"- messages: {len(turns)}" + (" (truncated)" if truncated else ""),
        f"- redacted: {redacted}",
        "",
    ]
    for turn in turns:
        lines.append(f"## [{turn['id']}] {turn['role']} — {turn['created_at']}")
        if turn["tools"]:
            lines.append("")
            lines.append("tools: " + ", ".join(f"`{name}`" for name in turn["tools"]))
        if turn["errors"]:
            lines.append("")
            lines.append("errors: " + " | ".join(turn["errors"]))
        lines.append("")
        lines.append(turn["text"])
        lines.append("")
    return "\n".join(lines)


def _bounded_turns(turns: list[dict[str, Any]], fmt: str, budget: int) -> tuple[list[dict[str, Any]], bool]:
    """Keep whole turns inside the byte budget so `count` stays truthful."""
    kept: list[dict[str, Any]] = []
    used = 2048  # room for the document header
    for turn in turns:
        estimate = len(json.dumps(turn, ensure_ascii=False, default=str).encode("utf-8")) + 256
        if kept and used + estimate > budget:
            return kept, True
        kept.append(turn)
        used += estimate
    return kept, False


async def write_export(
    *,
    conversation_id: Any = None,
    fmt: Any = "md",
    redact: bool = True,
    after_id: Any = None,
    limit: Any = None,
    out: str | None = None,
    ctx: Any,
) -> dict[str, Any]:
    """Shared export implementation for the tool and the ``/export`` command."""
    format_name = _format(fmt)
    page_size = _limit(limit, "limit", DEFAULT_LIMIT, MAX_MESSAGES)
    conversation_id, rows, has_more, next_cursor = await _load(conversation_id, after_id, page_size)
    patterns = _patterns(ctx) if redact else ()
    turns = [_turn(row, patterns, excerpt=False, redact=bool(redact)) for row in rows]
    turns, budget_truncated = _bounded_turns(turns, format_name, MAX_EXPORT_BYTES)
    has_more = has_more or budget_truncated
    if budget_truncated:
        next_cursor = str(turns[-1]["id"]) if turns else (str(rows[0]["id"]) if rows else None)
    document = _render(format_name, conversation_id, turns, bool(redact), has_more)
    if redact:
        # The very same sanitizer protects unauthenticated API responses.
        # Applying it to the complete rendered document also covers headers,
        # metadata and any credential not already caught in _turn.
        document = redact_preview_text(document)
    data = document.encode("utf-8")
    if len(data) > MAX_EXPORT_BYTES:  # defensive: a huge id or metadata label
        data = data[:MAX_EXPORT_BYTES].decode("utf-8", errors="ignore").encode("utf-8")
        has_more = True
    # Selection, budget accounting and creation are one critical section: a
    # concurrent writer must see the bytes and versions created so far before
    # it picks its own target or checks the aggregate budget. The root fd is
    # held from validation to write, so a directory swap cannot redirect the
    # file outside the exports root.
    with _EXPORT_WRITE_LOCK:
        root, expected = _exports_root(ctx)
        with _export_lock(root):
            # The root can be swapped between validation and lock acquisition;
            # re-check its identity before any path is selected or written.
            if _root_identity(root) != expected:
                raise PermissionError("exports root changed since validation")
            target = _target_path(root, out, format_name, conversation_id)
            _enforce_export_budget(root, len(data))
            root_fd = _open_exports_root(root, expected)
            try:
                _write_export_file(root_fd, target, data)
            finally:
                os.close(root_fd)
    return {
        "success": True,
        "path": str(target),
        "bytes": len(data),
        "count": len(turns),
        "redacted": bool(redact),
        "format": format_name,
        "conversation_id": conversation_id,
        "has_more": has_more,
        "next_cursor": next_cursor,
        "truncated": has_more,
    }


async def conversation_export(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Write one bounded, redacted transcript file and report its real size."""
    redact = args.get("redact", True)
    if isinstance(redact, bool):
        pass
    elif isinstance(redact, str) and redact.strip().lower() in {"true", "false"}:
        redact = redact.strip().lower() == "true"
    else:
        raise ValueError("redact must be a boolean")
    return await write_export(
        conversation_id=args.get("conversation_id"),
        fmt=args.get("format", "md"),
        redact=redact,
        after_id=args.get("after_id"),
        limit=args.get("limit"),
        ctx=ctx,
    )


def register(registry: Any, services: Any = None) -> list[str]:
    """Register both conversation tools with typed, strict ToolSpecs.

    ``services`` is accepted for the standard registrar signature; these tools
    read the shared durable database directly from the tool context.
    """
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="conversation_messages",
                description=(
                    "Read compact, redacted turns from durable chat history with "
                    "after_id/next_cursor pagination; includes tool names and errors."
                ),
                category="conversation",
                risk=RiskLevel.LOW,
                permissions=["conversation:read"],
                input_schema=_schema(
                    {
                        "conversation_id": {"type": "string", "minLength": 1, "maxLength": 256},
                        "after_id": {"type": ["integer", "string", "null"], "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_MESSAGES},
                    },
                    [],
                ),
                output_schema=_output_schema(
                    {
                        "success": {"type": "boolean"},
                        "conversation_id": {"type": "string"},
                        "count": {"type": "integer", "minimum": 0},
                        "has_more": {"type": "boolean"},
                        "next_cursor": {"type": ["string", "null"]},
                        "redacted": {"type": "boolean"},
                    },
                    ["success", "count", "has_more"],
                ),
                resource_field="",
            ),
            conversation_messages,
        ),
        (
            ToolSpec(
                name="conversation_export",
                description=(
                    "Export durable chat history to one redacted Markdown, JSON or "
                    "text file under .leo/exports; returns the real path, byte size "
                    "and message count in a single call."
                ),
                category="conversation",
                risk=RiskLevel.MEDIUM,
                permissions=["conversation:read", "file:write"],
                input_schema=_schema(
                    {
                        "conversation_id": {"type": "string", "minLength": 1, "maxLength": 256},
                        "format": {"type": "string", "enum": list(FORMATS)},
                        "redact": {"type": "boolean"},
                        "after_id": {"type": ["integer", "string", "null"], "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_MESSAGES},
                    },
                    [],
                ),
                output_schema=_output_schema(
                    {
                        "success": {"type": "boolean"},
                        "path": {"type": "string", "minLength": 1},
                        "bytes": {"type": "integer", "minimum": 0},
                        "count": {"type": "integer", "minimum": 0},
                        "redacted": {"type": "boolean"},
                        "format": {"type": "string", "enum": list(FORMATS)},
                    },
                    ["success", "path", "bytes", "count", "redacted"],
                ),
                resource_field="",  # the path is derived and confined by the tool
            ),
            conversation_export,
        ),
    ]
    names: list[str] = []
    for spec, handler in definitions:
        try:
            registry.register(spec, handler)
        except ValueError:  # already present - keep the caller's version
            continue
        names.append(spec.name)
    return names


_install_audit_digest_registry()


__all__ = [
    "FORMATS",
    "DEFAULT_LIMIT",
    "MAX_MESSAGES",
    "MAX_EXPORT_BYTES",
    "MAX_EXPORT_VERSIONS",
    "MAX_EXPORT_TOTAL_BYTES",
    "EXPORT_FILE_MODE",
    "ExportStorageError",
    "redact_text",
    "redact_preview_content",
    "redact_preview_text",
    "redact_message_content",
    "conversation_messages",
    "conversation_export",
    "write_export",
    "register",
]
