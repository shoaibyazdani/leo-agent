"""Single-call, provenance-tagged HTTP JSON extraction.

The legacy pattern for "look up a number on the web" was two model turns:
``http_get`` returned a raw (and, in the chat loop, eventually context-trimmed)
body, then ``json_query`` required the model to re-supply the JSON as ``data``.
Nothing bound the second argument to the first response, so the value the model
finally quoted could be guessed rather than read. ``json_query`` is therefore
operator-only in the composed tool catalogue (see ``tools/__init__.py``).

The two tools here close that gap:

* ``http_json``      - one real GET, strict JSON parse, dotted/index path lookup,
                       returns only the value decoded from that response;
* ``http_json_paths`` - one real GET, returns the child paths available under a
                       prefix so the caller never has to guess a path. A prefix
                       that had to be masked or hashed yields no advertised
                       paths (with a ``reason``), because such a prefix cannot
                       be passed back to ``http_json``.

Both delegate the entire network operation to :func:`leo.tools.builtins.http_get`,
which owns the shared egress policy (public-address check, single-resolution DNS
pinning with Host/SNI preservation, no redirects, no proxy env, cumulative raw
byte cap with bounded decompression, cache-busting request). No second HTTP
implementation, no cached/static body, no ambient state and no second executor
call: one tool call means one real request, one tool_calls row, one executor
audit record and one tool event.

Provenance is deliberately named ``fetched-http``, not ``live-http``: a
successful GET proves only that a response was fetched at ``fetched_at``. A CDN
or forward proxy may answer from cache, and response Date/Age/Cache-Control/ETag
are echoed so the caller can judge freshness. Never describe these values as
real-time or authoritative without that evidence.

Output size is budgeted against the *final* chat serialization: the executor
wraps the result in a ToolResult (including a metadata mapping), the chat loop
calls ``json.dumps`` with ``ensure_ascii=True``, and the context optimizer trims
tool messages at 2000 characters. CJK text expands up to 6x when escaped, so the
value/source are emitted first and a *modeled* representative tool message
(envelope and representative metadata included) is capped at 900 characters.
The runtime envelope is only estimated: a longer call id or richer metadata on
an error path can make an individual message larger, and a caller that lowers
the optimizer's ``tool_output_chars`` below 900 gets no guarantee from this
budget. Echoed URL/path/prefix strings are bounded and hashed when cut. An
oversized or redaction-matching value is an explicit error, never a silently
truncated or ``[REDACTED]`` value paired with a live-data provenance.

Known limits (documentation only, F8/F9):

* Numeric normalization - values are decoded by Python's ``json`` parser using
  IEEE-754 binary floats, so a decimal in the response may be rounded (for
  example a long decimal or a large integer converted through ``float``). The
  tool returns the decoded number, not the literal response text; quote it with
  that precision caveat and do not claim the digits are byte-identical.
* Public reverse-proxy pivot - the egress check validates the one public IP
  this process connects to. A public reverse proxy can forward to a private
  origin that this process cannot see; these tools are not an internal-network
  boundary and must not be presented as one.
* Concatenated gzip streams (F2) - the shared GET enforces ``max_bytes`` on
  both wire bytes and bytes decoded by zlib from the *first* gzip/deflate
  member. Once that member reaches EOF, zlib leaves any concatenated member in
  ``unused_data``, which the shared handler does not decode. These tools tap
  that handler's decompressor for the duration of their own GET and refuse a
  response with trailing/concatenated members or an incomplete member, so a
  success cannot publish a partially decoded body. Callers that bypass these
  tools and invoke ``builtins.http_get`` directly still have that documented
  limit.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
import unicodedata
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlparse

from ..contracts import RiskLevel, ToolSpec
from . import builtins
from .data_tools import resolve_path
from .executor import redact_value

# A response was fetched; freshness is evidence, not a promise.
SOURCE = "fetched-http"

DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_BYTES = 64 * 1024
# The chat loop serializes a ToolResult with ensure_ascii=True and trims tool
# messages at 2000 characters. Budget the modeled FULL representative message
# (envelope and a representative metadata mapping included) at 900 so escaped
# non-ASCII values and provenance fit. This is a budget, not a promise about a
# lower optimizer trim setting or an unusually large runtime envelope.
MAX_TOOL_MESSAGE_CHARS = 900
MAX_ECHO_CHARS = 120
# A single unknown-shape path segment is capped before the whole token is
# capped, so one pathological segment cannot consume the entire echo budget.
MAX_ECHO_SEGMENT_CHARS = 64
MAX_KEY_CHARS = 200
DEFAULT_PATH_LIMIT = 25
MAX_PATH_LIMIT = 50
MAX_URL_CHARS = 4096

_URL_FIELD: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": MAX_URL_CHARS,
}
_TIMEOUT_FIELD: dict[str, Any] = {
    "type": "number",
    "minimum": 0.1,
    "maximum": MAX_TIMEOUT_SECONDS,
}
_MAX_BYTES_FIELD: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "maximum": builtins.MAX_HTTP_BYTES,
}

# Content types that are definitively not JSON. Everything else (including
# text/plain and a missing header) is left to the strict parser: plenty of real
# APIs serve JSON with an imprecise content type.
_NON_JSON_TYPES = ("text/html", "application/xhtml+xml", "application/xml", "text/xml")

# Query parameters whose values are always dropped from echoed URLs. Operators
# can broaden config.security.redact_patterns, but these tools never echo a
# query value even when the global redactor would miss it (e.g. ``key=``).
_SENSITIVE_QUERY_RE = re.compile(
    r"(?i)(?:api[_-]?key|apikey|token|secret|password|passwd|pwd|auth|session|"
    r"signature|sig|key|code|credential)"
)
# Independent of configuration: an assignment-shaped secret in a fetched value
# is refused rather than published with a provenance marker.
_INTRINSIC_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|apikey|token|secret|password|key)\s*[:=]\s*\S+"
)
# Bare token shapes are secrets too, even without an assignment and even when
# an operator has cleared every configured pattern. They are masked in echoed
# paths/keys and refused when found in a fetched value.
_INTRINSIC_TOKEN_RE = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bgh[pousr]_[A-Za-z0-9_]{20,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b|\bxox[baprs]-[A-Za-z0-9-]{10,}\b|"
    r"\b(?:AKIA|ASIA|AIDA|AROA|AGPA|AIPA|ANPA|ANVA)[A-Z0-9]{16}\b|"
    r"\bAIza[0-9A-Za-z_-]{35}\b|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
# Control characters in a model-supplied path token are never echoed raw.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
# Invisible characters split a token without changing how it reads in a chat
# message: Cf (soft hyphen, bidi overrides/isolates, ZWJ, BOM, ...), Cc
# (controls), Cs (lone surrogates) and combining marks (Mn/Mc/Me, including
# CGJ and variation selectors). This is a CATEGORY rule, not a fixed list: a
# new invisible codepoint must not silently re-open the secret split. These
# characters are removed before *every* secret match, and any echoed token
# containing one is masked as a whole rather than published raw.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc", "Cs", "Mn", "Mc", "Me"})


def _normalize_for_match(text: str) -> str:
    """Remove invisible/format characters before testing for secret shapes."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES
    )


# Representative worst case of ``json.dumps(result.model_dump(mode='json'),
# default=str)`` in the chat loop (long call id + longest tool name). The
# executor always attaches metadata (timeout_seconds for these tools, plus
# dry_run/approval/quota fields on other paths), so an empty mapping would
# understate the envelope. The measurement is still a representative estimate:
# unusually large runtime metadata or a smaller optimizer trim can exceed it.
_ENVELOPE: dict[str, Any] = {
    "call_id": "call_" + "x" * 64,
    "tool": "http_json_paths",
    "success": True,
    "output": None,
    "error": None,
    "duration_ms": 12345.6789,
    "risk": "medium",
    "decision": "allowed",
    "metadata": {"timeout_seconds": 10.0, "dry_run": False},
}
_ENVELOPE_OVERHEAD = len(
    json.dumps(_ENVELOPE, ensure_ascii=True, default=str)
) - len("null")

_FRESHNESS_FIELD: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "date": {"type": ["string", "null"]},
        "age": {"type": ["string", "null"]},
        "cache_control": {"type": ["string", "null"]},
        "etag": {"type": ["string", "null"]},
    },
    "required": ["date", "age", "cache_control", "etag"],
    "additionalProperties": False,
}
_OUTPUT_COMMON: dict[str, Any] = {
    "url": {"type": "string"},
    "url_hash": {"type": "string"},
    "status_code": {"type": "integer"},
    "fetched_at": {"type": "string"},
    "freshness": _FRESHNESS_FIELD,
    "source": {"type": "string", "const": SOURCE},
}


def _require_url(args: dict[str, Any]) -> str:
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    return url


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _bounded_echo(text: Any, limit: int = MAX_ECHO_CHARS) -> str:
    """Echo a bounded string; append a short hash when the original was cut."""
    value = str(text)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...#{_digest(value)}"


def _scrub_echo(text: str, patterns: list[re.Pattern[str]]) -> str:
    """Mask secret-shaped fragments in any echoed string before bounding it.

    Invisible Unicode (format controls, combining marks, controls, surrogates)
    is removed first: otherwise ``sk-<invisible>token`` is invisible to both
    the intrinsic regexes and a configured pattern. A token split that way is
    masked as a whole by ``_sanitize_token`` because the scrubbed text differs
    from the raw one.
    """
    scrubbed = _normalize_for_match(text)
    scrubbed = _INTRINSIC_SECRET_RE.sub("[masked]", scrubbed)
    scrubbed = _INTRINSIC_TOKEN_RE.sub("[masked]", scrubbed)
    scrubbed = _CONTROL_RE.sub("?", scrubbed)
    for pattern in patterns:
        scrubbed = pattern.sub("[masked]", scrubbed)
    return scrubbed


def _sanitize_token(
    text: Any, patterns: list[re.Pattern[str]], limit: int = MAX_ECHO_CHARS
) -> str:
    """Echo an arbitrary path token without raw secrets or unbounded segments.

    A token that is secret-shaped (or carries control characters) is replaced
    by ``[masked]#<digest>``: the digest is bounded and lets a caller correlate
    the token without publishing its raw value. Otherwise every dotted segment
    and then the whole token are bounded, with a hash appended when cut.
    """
    raw = str(text)
    cleaned = _scrub_echo(raw, patterns)
    if cleaned != raw:
        return f"[masked]#{_digest(raw)}"
    segments = [
        _bounded_echo(segment, MAX_ECHO_SEGMENT_CHARS)
        for segment in cleaned.split(".")
    ]
    return _bounded_echo(".".join(segments), limit)


def _sanitize_url(url: str, patterns: list[re.Pattern[str]]) -> str:
    """Return scheme://host[:port]/path with query values masked or dropped.

    Fragments and credentials are never echoed. Sensitive-looking parameter
    names are removed entirely (so the global redactor cannot accidentally
    turn a masked value into ``[REDACTED]``); the rest keep their key with a
    ``***`` placeholder. The path is scrubbed too: a token embedded in a URL
    path is still a secret, even though the global redactor misses ``key=``.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.rsplit("@", 1)[-1]  # credentials are rejected anyway
    path = _sanitize_token(parsed.path, patterns)
    base = f"{parsed.scheme}://{netloc}{path}"
    kept: list[str] = []
    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        if not key or _SENSITIVE_QUERY_RE.search(key):
            continue
        kept.append(f"{_sanitize_token(key, patterns, 32)}=***")
    query = "&".join(kept)
    return _bounded_echo(base + (f"?{query}" if query else ""))


def _redaction_patterns(ctx: dict[str, Any]) -> list[re.Pattern[str]]:
    settings = ctx.get("settings")
    raw = getattr(getattr(settings, "security", None), "redact_patterns", []) or []
    patterns: list[re.Pattern[str]] = []
    for pattern in raw:
        try:
            # Match the same normalized text the caller-facing secrets use: a
            # pattern that itself contains an invisible separator must still
            # catch the visible token.
            patterns.append(re.compile(_normalize_for_match(str(pattern))))
        except (re.error, TypeError):
            continue
    return patterns


def _matches_secret(text: str, patterns: list[re.Pattern[str]]) -> bool:
    normalized = _normalize_for_match(text)
    if _INTRINSIC_SECRET_RE.search(normalized) or _INTRINSIC_TOKEN_RE.search(normalized):
        return True
    return any(pattern.search(normalized) for pattern in patterns)


def _contains_secret(value: Any, patterns: list[re.Pattern[str]], seen: set[int] | None = None) -> bool:
    if isinstance(value, str):
        return _matches_secret(value, patterns)
    if isinstance(value, dict):
        return any(
            _contains_secret(k, patterns) or _contains_secret(v, patterns)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item, patterns) for item in value)
    return False


def _assert_publishable_value(value: Any, patterns: list[re.Pattern[str]]) -> None:
    """Fail closed when the executor's redactor would rewrite the evidence."""
    if _contains_secret(value, patterns):
        raise ValueError(
            "extracted value matches a secret/redaction pattern; refusing to publish "
            "a redacted value with an HTTP provenance marker"
        )


def _intrinsic_scrub(value: Any) -> Any:
    """Apply the intrinsic secret redaction to every emitted string (incl. keys).

    Invisible separators are stripped only when a secret is actually found; a
    benign value that happens to contain one is left byte-for-byte intact.
    """
    if isinstance(value, str):
        normalized = _normalize_for_match(value)
        text = _INTRINSIC_SECRET_RE.sub("[masked]", normalized)
        text = _INTRINSIC_TOKEN_RE.sub("[masked]", text)
        return text if text != normalized else value
    if isinstance(value, dict):
        return {_intrinsic_scrub(k): _intrinsic_scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_intrinsic_scrub(item) for item in value]
    return value


def _model_chat_output(
    output: dict[str, Any], patterns: list[re.Pattern[str]]
) -> dict[str, Any]:
    """The exact output the chat loop sees after the executor's redaction pass.

    ``redact_value`` is the executor's own function: configured patterns are
    applied recursively to every string, including dict keys, and replace a
    match with the 10-character ``[REDACTED]`` (which can *expand* a short
    match). Measuring before this pass is not the final message size.
    """
    modeled = redact_value(output, patterns)
    return _intrinsic_scrub(modeled)


def _final_message_size(
    output: dict[str, Any], patterns: list[re.Pattern[str]] | None = None
) -> int:
    """Length of the chat-loop tool message *after* redaction, envelope included."""
    payload = _model_chat_output(output, patterns or [])
    return _ENVELOPE_OVERHEAD + len(json.dumps(payload, ensure_ascii=True, default=str))


def _assert_message_budget(
    output: dict[str, Any],
    subject: str,
    patterns: list[re.Pattern[str]] | None = None,
) -> None:
    size = _final_message_size(output, patterns)
    if size > MAX_TOOL_MESSAGE_CHARS:
        suffix = " after executor redaction" if patterns else ""
        # The error is itself part of a chat message: never let the executor's
        # redactor rewrite it into a '[REDACTED]' marker on the way out.
        raise ValueError(_scrub_echo(
            f"{subject} is too large for the bounded tool message{suffix} "
            f"({size} chars > {MAX_TOOL_MESSAGE_CHARS}); choose a more specific "
            "path or a shorter query; a value that would be redacted is refused "
            "rather than truncated",
            patterns or [],
        ))


def _assert_redaction_stability(
    output: dict[str, Any], patterns: list[re.Pattern[str]], subject: str
) -> None:
    """Refuse any result the executor would silently rewrite.

    This covers non-value provenance (fetched_at, freshness, url_hash, source,
    echo fields) and every emitted key as well as the value itself. A
    ``[REDACTED]`` substitution in a provenance field is never published with a
    success marker.
    """
    modeled = _model_chat_output(output, patterns)
    if modeled == output:
        return
    if list(modeled) != list(output):
        changed = ["key names"]
    else:
        changed = [name for name in output if modeled[name] != output[name]] or ["payload"]
    raise ValueError(
        f"configured redaction would rewrite {subject} fields ({', '.join(changed)}); "
        "refusing to publish a redacted value or provenance with an HTTP source marker"
    )


def _path_lookup_failure(
    exc: Exception, path: str, patterns: list[re.Pattern[str]], subject: str
) -> ValueError:
    """Build a bounded, secret-free error for a failed path resolution.

    ``resolve_path`` quotes the rejected segment in some of its messages, so a
    raw ``str(exc)`` can publish the very secret the rest of this module masks.
    Every non-numeric segment of the caller's path is replaced by a short
    digest *before* the normal intrinsic/configured scrub runs, the detail is
    bounded, and the modeled error is checked for redaction stability and for
    the final chat-message budget. No raw segment and no ``[REDACTED]`` marker
    can reach the model through this path.
    """
    detail = str(exc)
    for segment in path.split("."):
        if not segment or segment.isdigit():
            continue  # list indexes carry no secret and keep the reason readable
        masked = f"[masked]#{_digest(segment)}"
        detail = detail.replace(repr(segment), masked)
        detail = detail.replace(segment, masked)
    detail = _bounded_echo(_scrub_echo(detail, patterns))
    # Scrub the fixed wording too: a configured pattern that happens to match
    # "path" or "lookup" must not turn the final error into '[REDACTED]'.
    message = _bounded_echo(
        _scrub_echo(f"{subject} could not be resolved: {detail}", patterns)
    )
    error_output = {"error": message}
    _assert_redaction_stability(error_output, patterns, subject)
    _assert_message_budget(error_output, subject, patterns)
    return ValueError(message)


def _content_type_failure(
    content_type: Any, patterns: list[re.Pattern[str]]
) -> ValueError:
    """Build a bounded, secret-free error for a server-supplied content type.

    The header is untrusted server data: a bare ``str(content_type)`` in the
    error would be unbounded and would carry any secret in the header either
    raw (empty redactor) or rewritten to ``[REDACTED]`` (configured redactor).
    Scrub and bound it, then check the modeled message for redaction stability
    and the chat-budget exactly as ``_path_lookup_failure`` does.
    """
    detail = _sanitize_token(str(content_type or "").lower(), patterns)
    message = _bounded_echo(
        _scrub_echo(f"HTTP response is not JSON (content-type: {detail!r})", patterns)
    )
    error_output = {"error": message}
    _assert_redaction_stability(error_output, patterns, "content type")
    _assert_message_budget(error_output, "content type error", patterns)
    return ValueError(message)


def _freshness(response: dict[str, Any]) -> dict[str, Any]:
    raw = response.get("freshness")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "date": _bounded_echo(raw["date"], 64) if raw.get("date") else None,
        "age": _bounded_echo(raw["age"], 32) if raw.get("age") else None,
        "cache_control": _bounded_echo(raw["cache_control"], 64)
        if raw.get("cache_control") else None,
        "etag": _bounded_echo(raw["etag"], 64) if raw.get("etag") else None,
    }


# asyncio tasks have their own context, so a concurrent GET from another tool
# never records its decompressor in this call's tap.
_DECOMPRESSOR_TAP = contextvars.ContextVar("leo_http_decompressor_tap", default=None)


@contextmanager
def _decompressor_tap() -> Iterator[list[Any]]:
    """Observe the decompressor created by this task's one shared GET.

    ``builtins.http_get`` owns the egress policy and the byte cap; this tap only
    records what its own decompressor did. Other tasks see the original factory
    and are never recorded, and the factory is restored on exit.
    """
    original = builtins._decompressor_for
    observed: list[Any] = []
    marker = object()

    def wrapped(encoding: str) -> Any:
        decoder = original(encoding)
        if decoder is not None and _DECOMPRESSOR_TAP.get() is marker:
            observed.append(decoder)
        return decoder

    builtins._decompressor_for = wrapped
    token = _DECOMPRESSOR_TAP.set(marker)
    try:
        yield observed
    finally:
        _DECOMPRESSOR_TAP.reset(token)
        if builtins._decompressor_for is wrapped:
            builtins._decompressor_for = original


def _assert_complete_decompression(decoders: list[Any]) -> None:
    """Refuse a compressed body the shared GET only partially decoded."""
    for decoder in decoders:
        tail = bytes(getattr(decoder, "unused_data", b"") or b"")
        tail += bytes(getattr(decoder, "unconsumed_tail", b"") or b"")
        if tail.strip(b"\x00"):
            # zlib stops at the end of the first gzip member; trailing members
            # are not decoded by builtins.http_get and must not be published.
            raise ValueError(
                "compressed HTTP response has concatenated/trailing members; "
                "refusing to publish a partially decoded body"
            )
        if not getattr(decoder, "eof", False):
            raise ValueError(
                "compressed HTTP response ended before the compressed member was "
                "complete; refusing to publish a partially decoded body"
            )


async def _fetch_document(
    args: dict[str, Any], ctx: dict[str, Any], patterns: list[re.Pattern[str]]
) -> dict[str, Any]:
    """Perform exactly one GET through the shared, SSRF-safe builtin path."""
    # Construct a fresh argument dict: the extraction fields must never reach
    # the HTTP handler, and the HTTP handler's raw body never escapes it.
    with _decompressor_tap() as decoders:
        response = await builtins.http_get(
            {
                "url": args.get("url"),
                "timeout": args.get("timeout", DEFAULT_TIMEOUT_SECONDS),
                "max_bytes": args.get("max_bytes", DEFAULT_MAX_BYTES),
            },
            ctx,
        )
    _assert_complete_decompression(decoders)
    if not isinstance(response, dict):
        raise ValueError("HTTP fetch returned an unexpected result")

    status = response.get("status_code")
    if not isinstance(status, int) or not 200 <= status < 300:
        # Never parse an error page or a redirect body as JSON.
        raise ValueError(
            f"HTTP request did not return 2xx (status {status!r}); refusing to parse response"
        )
    if response.get("truncated"):
        # A partial document must never yield a value.
        raise ValueError("HTTP response exceeded max_bytes; refusing to parse truncated JSON")

    content_type = str(response.get("content_type") or "").lower()
    # A server can split a non-JSON marker with invisible Unicode (e.g.
    # "text/ht<U+200B>ml") so it reads the same but evades a raw-substring test.
    # Match on the normalized header; keep the original for the bounded,
    # scrubbed error detail.
    normalized_type = _normalize_for_match(content_type)
    if any(marker in normalized_type for marker in _NON_JSON_TYPES):
        raise _content_type_failure(content_type, patterns)

    content = response.get("content")
    if not isinstance(content, str):
        raise ValueError("HTTP fetch returned no text content")
    fetched_at = response.get("fetched_at")
    if not isinstance(fetched_at, str):
        fetched_at = ""  # fail closed on provenance rather than inventing a time
    return {
        "content": content,
        "status_code": status,
        "fetched_at": fetched_at,
        "freshness": _freshness(response),
    }


async def _parse_document(content: str, ctx: dict[str, Any]) -> Any:
    """Strict decode: no NaN/Infinity, no non-finite floats, no deep recursion."""
    try:
        return await builtins.json_parse({"text": content}, ctx)
    except RecursionError as exc:
        raise ValueError("JSON document is too deeply nested") from exc


def _serialized_size(value: Any) -> int:
    """Size as the chat loop serializes it (ensure_ascii=True, not False)."""
    return len(json.dumps(value, ensure_ascii=True, allow_nan=False, default=str))


def _echo_fields(
    url: str, document: dict[str, Any], patterns: list[re.Pattern[str]]
) -> dict[str, Any]:
    """Untrusted/model-supplied echo fields, scrubbed, bounded and never raw."""
    return {
        "url": _sanitize_url(url, patterns),
        "url_hash": _digest(url),
        "status_code": document["status_code"],
        "fetched_at": _bounded_echo(document["fetched_at"], 64),
        "freshness": document["freshness"],
    }


async def http_json(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Fetch a URL once and return the exact decoded value at ``path``."""
    url = _require_url(args)
    path = args.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "path must be a non-empty dotted path; call http_json_paths to discover paths"
        )
    patterns = _redaction_patterns(ctx)

    document = await _fetch_document(args, ctx, patterns)
    decoded = await _parse_document(document["content"], ctx)
    try:
        found, value = resolve_path(decoded, path)
    except Exception as exc:
        # A malformed path is a model error, not a response to publish: the
        # exception itself may quote a raw path segment (F1).
        raise _path_lookup_failure(exc, path, patterns, "path lookup") from exc
    if not found:
        # Errors are part of the chat message too: never echo a raw (possibly
        # secret) path or let redaction expand a long error past the trim.
        raise ValueError(
            f"path {_sanitize_token(path, patterns)!r} is absent from the fetched "
            "response; call http_json_paths to list available paths "
            "(no default is substituted)"
        )
    try:
        size = _serialized_size(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"extracted value is not JSON serializable: {exc}") from exc
    # Clear errors beat a redacted value (F4) or a silently escaped CJK value
    # that the chat loop would cut in half (F1).
    _assert_publishable_value(value, patterns)
    output: dict[str, Any] = {
        "source": SOURCE,  # value and provenance first: trim cannot cut them
        "value": value,
        "found": True,
        "path": _sanitize_token(path, patterns),
    }
    output.update(_echo_fields(url, document, patterns))
    _assert_redaction_stability(output, patterns, "extracted value")
    _assert_message_budget(
        output, f"extracted value ({size} chars, ensure_ascii=True)", patterns
    )
    return output


async def http_json_paths(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Fetch a URL once and list the child paths under ``prefix``."""
    url = _require_url(args)
    prefix = args.get("prefix", "")
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    limit = args.get("limit", DEFAULT_PATH_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    limit = max(1, min(limit, MAX_PATH_LIMIT))
    patterns = _redaction_patterns(ctx)

    document = await _fetch_document(args, ctx, patterns)
    decoded = await _parse_document(document["content"], ctx)
    try:
        found, node = resolve_path(decoded, prefix)
    except Exception as exc:
        raise _path_lookup_failure(exc, prefix, patterns, "prefix lookup") from exc
    if not found:
        raise ValueError(
            f"prefix {_sanitize_token(prefix, patterns)!r} is absent from the fetched response"
        )

    if isinstance(node, dict):
        # Dotted keys cannot be expressed by the shared path grammar. Empty,
        # dotted and over-long keys are counted as omitted, never enumerated.
        keys = sorted(
            k for k in node
            if isinstance(k, str) and k and "." not in k and len(k) <= MAX_KEY_CHARS
        )
        omitted = len(node) - len(keys)
    elif isinstance(node, list):
        keys = [str(index) for index in range(len(node))]
        omitted = 0
    else:
        raise ValueError("http_json_paths prefix must address a JSON object or array")

    prefix_echo = _sanitize_token(prefix, patterns)
    # A masked/hashed prefix is only a correlation hint: feeding it back into
    # http_json resolves nothing. Never advertise paths built on one.
    usable_prefix = prefix_echo == prefix
    reason: str | None = None
    if not usable_prefix:
        reason = (
            "prefix was masked or hashed; child paths withheld because they "
            "cannot be passed back to http_json"
        )
    base = f"{prefix_echo}." if prefix_echo else ""
    base_output: dict[str, Any] = {
        "source": SOURCE,
        "paths": [],
        "omitted_paths": omitted,
        "has_more": True,
        "prefix": prefix_echo,
        "reason": reason,
    }
    base_output.update(_echo_fields(url, document, patterns))

    def _trial(candidates: list[str], omitted_count: int) -> dict[str, Any]:
        """Model a candidate message with the counters that would accompany it."""
        trial = dict(base_output)
        trial.update({
            "paths": candidates,
            "omitted_paths": omitted_count,
            "has_more": True,  # still enumerating while trials are measured
        })
        return trial

    paths: list[str] = []
    index = 0
    while usable_prefix and index < len(keys) and len(paths) < limit:
        key = keys[index]
        index += 1
        safe_key = _sanitize_token(key, patterns, MAX_KEY_CHARS)
        if safe_key != key:
            # A secret-looking or unknown-shape response key must not be echoed,
            # and a hashed path cannot be used for lookup: count it as omitted.
            omitted += 1
            continue
        candidate = base + safe_key
        if _contains_secret(candidate, patterns):
            omitted += 1
            continue
        if _final_message_size(_trial(paths + [candidate], omitted + 1), patterns) > MAX_TOOL_MESSAGE_CHARS:
            # Do not claim a complete enumeration when a key cannot fit after
            # the executor's redaction pass.
            omitted += 1
            continue
        paths.append(candidate)
    omitted += len(keys) - index  # keys never reached because of the limit
    base_output["paths"] = paths
    base_output["omitted_paths"] = omitted
    base_output["has_more"] = bool(len(keys) > len(paths) or omitted)
    # An omitted child key is a withheld branch. The tool description promises
    # an explanation for withheld keys, so reason must be truthful whenever
    # omitted_paths is non-zero, not only when the prefix itself was masked.
    withheld_reason = (
        "child keys withheld (secret-shaped/over-long/over budget/limit); "
        "see omitted_paths"
    )
    # A masked prefix already explains why its child paths are withheld; do not
    # spend the bounded-message budget on a second, redundant explanation.
    if omitted and not reason:
        base_output["reason"] = withheld_reason
    # The trials above used in-progress counters. Recompute with the final ones
    # (extra digits in omitted_paths/has_more can push the message over budget)
    # and drop tail paths until the final message really fits. The reason text
    # is fixed and already included here, so trimming cannot invalidate it.
    while paths and _final_message_size(base_output, patterns) > MAX_TOOL_MESSAGE_CHARS:
        paths.pop()
        omitted += 1
        base_output["paths"] = list(paths)
        base_output["omitted_paths"] = omitted
        base_output["has_more"] = True
        if base_output["reason"] is None:
            base_output["reason"] = withheld_reason
    _assert_redaction_stability(base_output, patterns, "path enumeration")
    _assert_message_budget(base_output, "path enumeration", patterns)
    return base_output


_OUTPUT_JSON = {
    "type": "object",
    "properties": {
        **_OUTPUT_COMMON,
        "path": {"type": "string"},
        "found": {"type": "boolean"},
        "value": {},  # any JSON value: proven by the strict decode above
    },
    "required": ["url", "status_code", "path", "found", "value", "source",
                 "url_hash", "fetched_at", "freshness"],
    "additionalProperties": False,
}

_OUTPUT_PATHS = {
    "type": "object",
    "properties": {
        **_OUTPUT_COMMON,
        "prefix": {"type": "string"},
        "paths": {"type": "array", "items": {"type": "string"}},
        "omitted_paths": {"type": "integer", "minimum": 0},
        "has_more": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["url", "status_code", "prefix", "paths", "omitted_paths",
                 "has_more", "reason", "source", "url_hash", "fetched_at", "freshness"],
    "additionalProperties": False,
}

_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="http_json",
            description=(
                "ONE HTTP GET plus a dotted/list-index lookup on the JSON response actually "
                "fetched. Returns {source:'fetched-http', value, path, url, fetched_at, "
                "freshness:{date,age,cache_control,etag}, ...}; the value is the exact decoded "
                "JSON value with no caller-supplied data and no default/fallback. The response "
                "may be cached/stale even though a fresh fetch was requested: check the freshness "
                "evidence and do not claim real-time data. Numbers are decoded as floats, so "
                "decimal precision may be normalized. Use http_json_paths first if the path is "
                "unknown. Keys containing '.' cannot be addressed by the shared path grammar. "
                "max_bytes counts wire bytes and decoded bytes of the first gzip/deflate member; "
                "concatenated/truncated compressed members are refused rather than partially "
                "decoded and published."
            ),
            category="research",
            risk=RiskLevel.MEDIUM,
            permissions=["network:read"],
            input_schema={
                "type": "object",
                "properties": {
                    "url": _URL_FIELD,
                    "path": {"type": "string", "minLength": 1, "maxLength": MAX_URL_CHARS},
                    "timeout": _TIMEOUT_FIELD,
                    "max_bytes": _MAX_BYTES_FIELD,
                },
                "required": ["url", "path"],
                "additionalProperties": False,
            },
            output_schema=_OUTPUT_JSON,
            resource_field="",  # a URL is not a filesystem resource
        ),
        http_json,
    ),
    (
        ToolSpec(
            name="http_json_paths",
            description=(
                "ONE HTTP GET that lists child paths (dict keys / list indexes) available under "
                "'prefix' in the fetched JSON response, for choosing a path for http_json. "
                "Returns {source:'fetched-http', paths, omitted_paths, has_more, prefix, reason, "
                "url, fetched_at, freshness:{date,age,cache_control,etag}, ...}. Every advertised "
                "path can be passed back to http_json: when prefix is masked/hashed or a child "
                "key is withheld (secret-shaped, over-long or too large for the bounded message), "
                "that branch is counted in omitted_paths and explained in reason. "
                "omitted_paths counts dotted/empty/over-long keys and paths that did "
                "not fit the bounded message; has_more is true whenever the enumeration is "
                "incomplete. Dotted keys cannot be addressed by the path grammar. The response "
                "may be cached/stale: check freshness "
                "evidence before implying the data is real-time. max_bytes counts wire bytes and "
                "decoded bytes of the first gzip/deflate member; concatenated/truncated compressed "
                "members are refused rather than partially decoded and published."
            ),
            category="research",
            risk=RiskLevel.MEDIUM,
            permissions=["network:read"],
            input_schema={
                "type": "object",
                "properties": {
                    "url": _URL_FIELD,
                    "prefix": {"type": "string", "maxLength": MAX_URL_CHARS},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_PATH_LIMIT},
                    "timeout": _TIMEOUT_FIELD,
                    "max_bytes": _MAX_BYTES_FIELD,
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            output_schema=_OUTPUT_PATHS,
            resource_field="",
        ),
        http_json_paths,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the fetched-JSON tools; ``services`` is accepted, not required."""
    registered: list[str] = []
    for spec, handler in _SPECS:
        try:
            registry.register(spec, handler)
        except ValueError:  # already present - keep the caller's version
            continue
        registered.append(spec.name)
    return registered


__all__ = ["register", "http_json", "http_json_paths", "SOURCE"]
