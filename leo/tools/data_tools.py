"""Deterministic, offline data-transformation tools.

This toolset exposes four pure local transformations to the chat tool loop:

* ``json_query``     - safe dotted-path / index lookup (no ``eval``, no JMESPath
                       expressions, no code execution);
* ``csv_parse``      - strict CSV parsing with configurable header and delimiter;
* ``text_transform`` - case conversion, plain replacement and a deliberately
                       restricted, linear-time-oriented regex replacement;
* ``base64_codec``   - base64 / urlsafe-base64 encode and decode.

Every handler is a pure function of its arguments. There is no network access,
no filesystem access, no ambient/durable state and no ``eval``/``exec``. All
inputs are size-bounded and all structures/row counts/pattern complexity are
capped so a tool call cannot run away. No service needs to be injected; the
``services`` argument of :func:`register` is accepted for interface
compatibility with the other toolsets.

Malformed input is reported as a plain ``ValueError`` with a descriptive
message; the executor converts it into ``success=False`` and an ``error``
string, which is an ordinary, auditable tool failure.

Known limit (F8, numeric normalization): ``json_query`` decodes numbers with
Python's ``json`` parser (IEEE-754 floats), so decimal literals can be rounded
and very large integers may lose precision. The returned values are the decoded
numbers, not the original lexical representation; do not present them as exact
copies of source text. Also note that ``json_query`` is operator/offline only
in the composed catalogue (``tools/MODEL_FORBIDDEN_TOOLS``) because its
caller-supplied ``data``/``default`` cannot prove provenance.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import re
from typing import Any

from ..contracts import RiskLevel, ToolSpec

# --------------------------------------------------------------------------- bounds

_MAX_INPUT_BYTES = 64 * 1024          # generic text/JSON input cap (64 KiB)
_MAX_JSON_PATH_SEGMENTS = 64
_MAX_JSON_PATH_SEGMENT_CHARS = 200
_MAX_CSV_ROWS = 1000
_MAX_CSV_COLUMNS = 200
_MAX_REGEX_INPUT_BYTES = 8192
_MAX_REGEX_PATTERN_CHARS = 120
# At most two quantifiers keeps the worst-case backtracking of the bounded
# input quadratic; three or more adjacent quantifiers (``a+a+a+``) can explode
# combinatorially even without a nested group, so they are rejected outright.
_MAX_REGEX_QUANTIFIERS = 2
_MAX_QUANTIFIER_BOUND = 64
_MAX_REPLACEMENTS = 10_000
_MAX_AUX_TEXT_BYTES = 4096

_CASE_OPERATIONS = ("upper", "lower", "title")
_TRANSFORM_OPERATIONS = _CASE_OPERATIONS + ("replace", "regex_replace")
_CODEC_OPERATIONS = ("encode", "decode")
_INDEX_RE = re.compile(r"[0-9]+\Z")
_BRACED_QUANTIFIER_RE = re.compile(r"\{([0-9]+)(?:,([0-9]*))?\}")


# --------------------------------------------------------------------------- helpers


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    return value


def _check_bytes(text: str, field: str, limit: int) -> str:
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} is not valid Unicode text") from exc
    if size > limit:
        raise ValueError(f"{field} exceeds {limit} byte limit ({size} bytes)")
    return text


def _bounded(value: Any, field: str, limit: int = _MAX_INPUT_BYTES) -> str:
    return _check_bytes(_require_str(value, field), field, limit)


def _require_bool(value: Any, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _parse_json_data(data: Any) -> Any:
    """Accept JSON text or an already-decoded JSON object/array, with bounds."""
    if isinstance(data, str):
        text = _bounded(data, "data")
        if not text.strip():
            raise ValueError("data is empty, not valid JSON")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"data is not valid JSON: {exc}") from exc
    if isinstance(data, (dict, list)):
        try:
            encoded = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"data is not JSON serializable: {exc}") from exc
        _check_bytes(encoded, "data", _MAX_INPUT_BYTES)
        return data
    raise ValueError("data must be a JSON string, object or array")


def resolve_path(root: Any, path: str) -> tuple[bool, Any]:
    """Resolve a dotted path against decoded JSON.

    Dict segments are literal keys; list segments must be non-negative decimal
    indexes. Returns ``(found, value)``. A missing key/index yields
    ``(False, None)``; trying to traverse a scalar and malformed paths are
    reported as errors because they indicate a bad query rather than absence.
    """
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if len(path.encode("utf-8", errors="replace")) > _MAX_INPUT_BYTES:
        raise ValueError("path exceeds byte limit")
    if not path:
        return True, root
    segments = path.split(".")
    if len(segments) > _MAX_JSON_PATH_SEGMENTS:
        raise ValueError(f"path exceeds {_MAX_JSON_PATH_SEGMENTS} segments")
    for position, segment in enumerate(segments):
        if not segment:
            raise ValueError(f"path segment {position} is empty")
        if len(segment) > _MAX_JSON_PATH_SEGMENT_CHARS:
            raise ValueError(
                f"path segment {position} exceeds {_MAX_JSON_PATH_SEGMENT_CHARS} characters"
            )

    current = root
    for position, segment in enumerate(segments):
        if isinstance(current, dict):
            if segment not in current:
                return False, None
            current = current[segment]
        elif isinstance(current, list):
            if not _INDEX_RE.fullmatch(segment):
                raise ValueError(
                    f"path segment {position} ({segment!r}) is not a valid list index"
                )
            index = int(segment)
            if index >= len(current):
                return False, None
            current = current[index]
        else:
            raise ValueError(
                f"path segment {position} cannot traverse a "
                f"{type(current).__name__} value"
            )
    return True, current


# --------------------------------------------------------------------------- safe regex


class _GroupFrame:
    __slots__ = ("quantified", "alternation")

    def __init__(self) -> None:
        self.quantified = False
        self.alternation = False


def _check_quantified_atom(atom: tuple[bool, bool] | None) -> None:
    if atom is None:
        raise ValueError("regex quantifier has no preceding atom")
    is_group, unsafe = atom
    if is_group and unsafe:
        raise ValueError(
            "regex pattern is rejected: nested quantifiers or alternation inside "
            "a quantified group can cause catastrophic backtracking"
        )


def validate_safe_regex(pattern: str) -> None:
    """Accept only a bounded regex subset that avoids catastrophic backtracking.

    Rejected: lookarounds/inline flags/special groups ``(?...)``, backreferences,
    quantifiers applied to a group that itself contains a quantifier or
    alternation (e.g. ``(a+)+`` or ``(a|aa)*``), more than two quantifiers in one
    pattern (which keeps worst-case backtracking quadratic on bounded input),
    and oversized repeat counts. Plain groups, alternation, character classes,
    anchors and at most two bounded quantifiers on single atoms remain
    available.
    """
    if not isinstance(pattern, str):
        raise ValueError("pattern must be a string")
    if not pattern:
        raise ValueError("pattern must not be empty")
    if len(pattern) > _MAX_REGEX_PATTERN_CHARS:
        raise ValueError(f"pattern exceeds {_MAX_REGEX_PATTERN_CHARS} characters")

    frames = [_GroupFrame()]
    # (is_group, unsafe) for the atom immediately preceding the cursor.
    last_atom: tuple[bool, bool] | None = None
    quantifiers = 0
    index = 0
    length = len(pattern)

    while index < length:
        char = pattern[index]

        if char == "\\":
            if index + 1 >= length:
                raise ValueError("regex pattern ends with a dangling escape")
            following = pattern[index + 1]
            if following.isdigit() or following in ("g", "k"):
                raise ValueError("backreferences are not allowed in regex pattern")
            last_atom = (False, False)
            index += 2
            continue

        if char == "(":
            if index + 1 < length and pattern[index + 1] == "?":
                raise ValueError(
                    "lookarounds, inline flags and special groups are not allowed "
                    "in regex pattern"
                )
            frames.append(_GroupFrame())
            last_atom = None
            index += 1
            continue

        if char == ")":
            if len(frames) == 1:
                raise ValueError("unbalanced parenthesis in regex pattern")
            frame = frames.pop()
            last_atom = (True, frame.quantified or frame.alternation)
            index += 1
            continue

        if char == "|":
            frames[-1].alternation = True
            last_atom = None
            index += 1
            continue

        if char in "*+?":
            _check_quantified_atom(last_atom)
            quantifiers += 1
            if quantifiers > _MAX_REGEX_QUANTIFIERS:
                raise ValueError(
                    f"regex pattern exceeds {_MAX_REGEX_QUANTIFIERS} quantifiers"
                )
            frames[-1].quantified = True
            last_atom = (False, False)
            index += 1
            continue

        if char == "{":
            match = _BRACED_QUANTIFIER_RE.match(pattern, index)
            if match:
                low = int(match.group(1))
                high = int(match.group(2)) if match.group(2) else low
                if high < low or high > _MAX_QUANTIFIER_BOUND:
                    raise ValueError("regex quantifier bound is invalid or too large")
                _check_quantified_atom(last_atom)
                quantifiers += 1
                if quantifiers > _MAX_REGEX_QUANTIFIERS:
                    raise ValueError(
                        f"regex pattern exceeds {_MAX_REGEX_QUANTIFIERS} quantifiers"
                    )
                frames[-1].quantified = True
                last_atom = (False, False)
                index = match.end()
                continue
            last_atom = (False, False)
            index += 1
            continue

        if char == "[":
            cursor = index + 1
            if cursor < length and pattern[cursor] == "^":
                cursor += 1
            while cursor < length and pattern[cursor] != "]":
                if pattern[cursor] == "\\":
                    cursor += 1
                cursor += 1
            if cursor >= length:
                raise ValueError("unterminated character class in regex pattern")
            last_atom = (False, False)
            index = cursor + 1
            continue

        last_atom = (False, False)
        index += 1

    if len(frames) != 1:
        raise ValueError("unbalanced parenthesis in regex pattern")


# --------------------------------------------------------------------------- handlers


async def json_query(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Resolve a safe dotted path in a JSON document."""
    if "data" not in args:
        raise ValueError("data is required")
    if "path" not in args:
        raise ValueError("path is required")
    document = _parse_json_data(args["data"])
    path = args["path"]
    found, value = resolve_path(document, path)
    if not found and "default" in args:
        value = args["default"]
    return {"found": found, "path": path, "value": value}


async def csv_parse(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Parse CSV text into header + row dictionaries (or plain rows)."""
    text = _bounded(args.get("text"), "text")
    delimiter = args.get("delimiter", ",")
    delimiter = _bounded(delimiter, "delimiter", 8)
    if len(delimiter) != 1:
        raise ValueError("delimiter must be exactly one character")
    if delimiter in ("\n", "\r", '"'):
        raise ValueError("delimiter must not be a newline, carriage return or quote")
    has_header = _require_bool(args.get("header"), "header", True)
    if not text.strip():
        raise ValueError("CSV input is empty")

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    rows: list[list[str]] = []
    try:
        for row in reader:
            if len(rows) >= _MAX_CSV_ROWS:
                raise ValueError(f"CSV input exceeds {_MAX_CSV_ROWS} row limit")
            if len(row) > _MAX_CSV_COLUMNS:
                raise ValueError(f"CSV row exceeds {_MAX_CSV_COLUMNS} column limit")
            rows.append(row)
    except csv.Error as exc:
        raise ValueError(f"malformed CSV input: {exc}") from exc
    if not rows:
        raise ValueError("CSV input is empty")

    if not has_header:
        return {"header": None, "rows": rows, "row_count": len(rows)}

    header = rows[0]
    if not header or any(not name for name in header):
        raise ValueError("CSV header row contains an empty column name")
    if len(set(header)) != len(header):
        raise ValueError("CSV header row contains duplicate column names")
    records: list[dict[str, str]] = []
    for position, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(
                f"CSV row {position} has {len(row)} field(s) but header has {len(header)}"
            )
        records.append(dict(zip(header, row)))
    return {"header": header, "rows": records, "row_count": len(records)}


async def text_transform(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Case-convert text, replace a literal substring, or apply a safe regex."""
    if "operation" not in args:
        raise ValueError("operation is required")
    operation = args["operation"]
    if operation not in _TRANSFORM_OPERATIONS:
        raise ValueError(f"unsupported text operation: {operation!r}")
    text = _bounded(args.get("text"), "text")

    if operation in _CASE_OPERATIONS:
        result = getattr(text, operation)()
        replacements = 0
    elif operation == "replace":
        old = _bounded(args.get("old"), "old", _MAX_AUX_TEXT_BYTES)
        new = _bounded(args.get("new"), "new", _MAX_AUX_TEXT_BYTES)
        if not old:
            raise ValueError("old must not be empty")
        replacements = text.count(old)
        if replacements > _MAX_REPLACEMENTS:
            raise ValueError(f"text exceeds {_MAX_REPLACEMENTS} replacement limit")
        result = text.replace(old, new)
    else:
        pattern = _bounded(
            args.get("pattern"), "pattern", _MAX_REGEX_PATTERN_CHARS
        )
        # Character count is the real regex complexity budget; byte length is
        # only a coarse pre-filter for non-ASCII patterns.
        if len(pattern) > _MAX_REGEX_PATTERN_CHARS:
            raise ValueError(f"pattern exceeds {_MAX_REGEX_PATTERN_CHARS} characters")
        replacement = _bounded(
            args.get("replacement"), "replacement", _MAX_AUX_TEXT_BYTES
        )
        _check_bytes(text, "text", _MAX_REGEX_INPUT_BYTES)
        validate_safe_regex(pattern)
        limit = args.get("max_replacements", 0)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= _MAX_REPLACEMENTS:
            raise ValueError(
                f"max_replacements must be an integer between 0 and {_MAX_REPLACEMENTS}"
            )
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex pattern: {exc}") from exc
        result, replacements = compiled.subn(replacement, text, count=limit)

    return {
        "operation": operation,
        "text": result,
        "length": len(result),
        "replacements": replacements,
    }


async def base64_codec(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Encode UTF-8 text to base64, or strictly decode base64 to UTF-8 text."""
    if "operation" not in args:
        raise ValueError("operation is required")
    operation = args["operation"]
    if operation not in _CODEC_OPERATIONS:
        raise ValueError(f"unsupported base64 operation: {operation!r}")
    text = _bounded(args.get("text"), "text")
    urlsafe = _require_bool(args.get("urlsafe"), "urlsafe", False)

    if operation == "encode":
        try:
            raw = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("text is not valid Unicode for UTF-8 encoding") from exc
        encoded = (
            base64.urlsafe_b64encode(raw) if urlsafe else base64.b64encode(raw)
        )
        result = encoded.decode("ascii")
    else:
        payload = text.strip()
        if not payload:
            raise ValueError("text is empty, not valid base64")
        if urlsafe:
            payload = payload.replace("-", "+").replace("_", "/")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"text is not valid base64: {exc}") from exc
        try:
            result = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("decoded base64 bytes are not valid UTF-8") from exc

    return {"operation": operation, "text": result, "length": len(result)}


# --------------------------------------------------------------------------- registration


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="json_query",
            description=(
                "Query a JSON document with a safe dotted path (e.g. 'users.0.name'). "
                "Dictionary keys and non-negative list indexes only; no expressions or code."
            ),
            category="data",
            risk=RiskLevel.LOW,
            permissions=["data:read"],
            input_schema=_schema(
                {
                    "data": {"type": ["string", "object", "array"]},
                    "path": {"type": "string", "maxLength": 4096},
                    "default": {},
                },
                ["data", "path"],
            ),
            resource_field="",  # JSON text/records are not filesystem paths
        ),
        json_query,
    ),
    (
        ToolSpec(
            name="csv_parse",
            description=(
                "Parse bounded CSV text into header and rows. Use header=false for "
                "positional rows; delimiter defaults to a comma."
            ),
            category="data",
            risk=RiskLevel.LOW,
            permissions=["data:read"],
            input_schema=_schema(
                {
                    "text": {"type": "string", "minLength": 1, "maxLength": 65536},
                    "delimiter": {"type": "string", "minLength": 1, "maxLength": 1, "default": ","},
                    "header": {"type": "boolean", "default": True},
                },
                ["text"],
            ),
            resource_field="",  # inline CSV text is not a filesystem path
        ),
        csv_parse,
    ),
    (
        ToolSpec(
            name="text_transform",
            description=(
                "Transform bounded text: upper/lower/title case, literal replace, or a "
                "restricted regex replacement that rejects catastrophic-backtracking patterns."
            ),
            category="data",
            risk=RiskLevel.LOW,
            permissions=["data:transform"],
            input_schema=_schema(
                {
                    "text": {"type": "string", "maxLength": 65536},
                    "operation": {"type": "string", "enum": list(_TRANSFORM_OPERATIONS)},
                    "old": {"type": "string", "maxLength": 4096},
                    "new": {"type": "string", "maxLength": 4096},
                    "pattern": {"type": "string", "minLength": 1, "maxLength": 120},
                    "replacement": {"type": "string", "maxLength": 4096},
                    "max_replacements": {
                        "type": "integer", "minimum": 0, "maximum": _MAX_REPLACEMENTS,
                    },
                },
                ["text", "operation"],
            ),
            resource_field="",  # inline text, not a filesystem path
        ),
        text_transform,
    ),
    (
        ToolSpec(
            name="base64_codec",
            description=(
                "Encode UTF-8 text as base64 or strictly decode base64 back to UTF-8 text. "
                "Set urlsafe=true for the URL/filename-safe alphabet."
            ),
            category="data",
            risk=RiskLevel.LOW,
            permissions=["data:transform"],
            input_schema=_schema(
                {
                    "operation": {"type": "string", "enum": list(_CODEC_OPERATIONS)},
                    "text": {"type": "string", "minLength": 1, "maxLength": 65536},
                    "urlsafe": {"type": "boolean", "default": False},
                },
                ["operation", "text"],
            ),
            resource_field="",  # inline text, not a filesystem path
        ),
        base64_codec,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the four data tools with ``registry`` and return their names.

    No service is required: these are pure offline transformations. ``services``
    is accepted so this function matches the common toolset registration
    interface used by the integration agent.
    """
    names: list[str] = []
    for spec, handler in _SPECS:
        registry.register(spec, handler)
        names.append(spec.name)
    return names


__all__ = [
    "register",
    "json_query",
    "csv_parse",
    "text_transform",
    "base64_codec",
    "resolve_path",
    "validate_safe_regex",
]
