"""Bounded, deterministic hygiene for agent-authored skill content.

Authored skills are derived from transcript text and tool output, so both are
untrusted data. This module applies the same class of checks the learning
engine applies to memory notes before a skill or one of its supporting files is
stored:

* invisible/bidirectional Unicode control characters are stripped (Trojan
  Source: text that reads one way to a human and another to the model),
* secret-shaped values are refused outright (a fragment in durable, always-
  available skill text is still a disclosure),
* instruction-shaped passages are reported as warnings, never silently copied
  into a "rule" without the author seeing them.

It performs no I/O, imports no service, and never evaluates content.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

MAX_DESCRIPTION_CHARS = 60
MAX_DOCUMENTATION_CHARS = 100_000
MAX_FILE_BYTES = 1_048_576
MAX_PATH_CHARS = 512
# Text-only bundle. Binary assets are deliberately not supported: the durable
# store is a TEXT column, and silently corrupting an image is worse than
# refusing it.
ALLOWED_SUBDIRS = frozenset({"references", "templates", "scripts"})

_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069\ufeff]"
)
_WHITESPACE_RE = re.compile(r"\s+")

# Strong, self-identifying credential shapes. These are refused even when the
# surrounding sentence looks harmless: a live key is never "just an example".
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsk(?:-[A-Za-z0-9_-]{4,}){2,}",
        r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b",
        r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
        r"\b(?:AKIA|ASIA|AIDA|AROA|AGPA|AIPA|ANPA|ANVA)[A-Z0-9]{16}\b",
        r"\bAIza[0-9A-Za-z_-]{35}\b",
        r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@",
    )
)

# Phrases that steer a model. A skill may legitimately discuss prompt
# injection, so this is advisory: the write succeeds, but the author is told
# that untrusted instruction text was copied into durable content.
_INJECTION_RE = re.compile(
    r"(?i)("
    r"\b(?:ignore|disregard|forget|override|bypass)\b[\W_]{0,12}"
    r"(?:(?:all|any|the|these|those|your|my)\b[\W_]{0,12})?"
    r"(?:previous|prior|above|earlier|preceding|system|developer)\b[\W_]{0,12}"
    r"(?:instructions?|rules?|prompts?|messages?|directions?|commands?)\b"
    r"|<\|\s*im_start\s*\|>|</?system>"
    r"|\byou[\W_]{0,3}are[\W_]{0,3}now\b"
    r")"
)


def strip_control(text: str) -> str:
    """Remove invisible/bidi control characters from stored text."""
    return _CONTROL_RE.sub("", str(text or ""))


def normalize_line(value: str, *, limit: int) -> str:
    """Collapse to one bounded line; used for the always-loaded skill index."""
    clean = strip_control(value).strip()
    clean = _WHITESPACE_RE.sub(" ", clean)
    return clean[: max(0, limit)].rstrip()


def safe_description(value: str, *, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """A single-line description that fits the system-prompt index.

    Longer text (legacy/native manifests) is cut at a word boundary and marked
    with an ellipsis inside the budget, so the index never shows a half word or
    silently implies the sentence ended there.
    """
    text = normalize_line(value, limit=10_000)
    if len(text) <= limit:
        return text
    head = text[: max(1, limit - 1)]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return (head.rstrip(" ,.;:-") + "…")[:limit]


def scan_text(value: str, *, field: str = "content") -> tuple[str, list[str], list[str]]:
    """Return ``(clean, errors, warnings)`` for one text value.

    ``clean`` has control characters removed; errors are hard refusals
    (credential-shaped content), warnings are advisory (instruction shapes).
    """
    text = strip_control(str(value or ""))
    errors: list[str] = []
    warnings: list[str] = []
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(
                f"{field} contains a credential-shaped value; secrets are never "
                "persisted in a skill"
            )
            break
    if _INJECTION_RE.search(text):
        warnings.append(
            f"{field} contains instruction-shaped text; confirm it is a rule you "
            "mean to keep, not copied source instructions"
        )
    return text, errors, warnings


def validate_description(value: str) -> tuple[str, list[str]]:
    """Validate a new/changed description against the index budget."""
    clean = safe_description(value)
    errors: list[str] = []
    if not clean:
        errors.append("description must be a non-empty single line")
    elif len(_WHITESPACE_RE.sub(" ", strip_control(value).strip())) > MAX_DESCRIPTION_CHARS:
        errors.append(
            f"description is longer than {MAX_DESCRIPTION_CHARS} characters; the "
            "system-prompt skill index clips it, so the trigger must fit in one "
            "short sentence"
        )
    return clean, errors


def validate_relative_path(value: str) -> tuple[str | None, str | None]:
    """Validate a supporting-file path and return ``(normalized, error)``.

    Only ``references/``, ``templates/`` and ``scripts/`` files are allowed.
    Paths are POSIX-normalized; absolute paths, drive prefixes, ``..``
    segments and empty names are refused.
    """
    raw = strip_control(str(value or "")).strip()
    if not raw:
        return None, "file_path must be a non-empty string"
    if "\x00" in raw:
        return None, "file_path contains a NUL byte"
    if len(raw) > MAX_PATH_CHARS:
        return None, f"file_path exceeds {MAX_PATH_CHARS} characters"
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return None, "file_path must be relative to the skill root"
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None, "file_path must not contain '..' segments"
    if len(parts) < 2:
        return None, "file_path must name a file inside references/, templates/ or scripts/"
    if parts[0].lower() not in ALLOWED_SUBDIRS:
        return None, (
            "file_path must start with references/, templates/ or scripts/"
        )
    if not parts[-1] or parts[-1] in {".", ".."}:
        return None, "file_path must name a file"
    return "/".join(parts), None


__all__ = [
    "ALLOWED_SUBDIRS",
    "MAX_DESCRIPTION_CHARS",
    "MAX_DOCUMENTATION_CHARS",
    "MAX_FILE_BYTES",
    "MAX_PATH_CHARS",
    "normalize_line",
    "safe_description",
    "scan_text",
    "strip_control",
    "validate_description",
    "validate_relative_path",
]
