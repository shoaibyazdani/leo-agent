"""Prompt/safety scan for schedule fields.

Two jobs, one module:

* **reject** unsafe model/operator text before any proposal or schedule write
  (hidden/bidi/C0 controls, prompt injection, exfiltration and detected
  secrets);
* **sanitize for display only** so a card/error can never render invisible
  control characters or echo a credential.

Rejected text is deliberately *not* persisted or echoed: errors carry a stable
code and rule id only. The scanner normalizes NFKC for matching so full-width
homographs cannot bypass a rule, while the returned display string is derived
from the original text with the unsafe characters removed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

_MAX_SCAN_CHARS = 65_536

# Unicode TAG block plus the usual invisible/bidi set.
_HIDDEN_CHARS = frozenset(
    "​‌‍⁠⁢⁣⁤﻿"
) | frozenset(chr(codepoint) for codepoint in range(0xE0000, 0xE0080))
_BIDI_CHARS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")
# RGI subdivision flags are the one legitimate TAG-character sequence.
_RGI_FLAGS = ("gbeng", "gbsct", "gbwls")
_RGI_RE = re.compile(
    "|".join(
        "\U0001F3F4" + "".join(chr(0xE0000 + ord(ch)) for ch in code) + "\U000E007F"
        for code in _RGI_FLAGS
    )
)
_VARIATION_SELECTOR = 0xFE0F
_EMOJI_RANGES = (
    (0x1F000, 0x1FFFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x1F1E6, 0x1F1FF),
    (0x20E3, 0x20E3),
)


class ScheduleSafetyError(ValueError):
    """A safe-to-display refusal with a stable code and rule id."""

    def __init__(self, code: str, rule: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.rule = rule
        self.message = message


@dataclass(frozen=True)
class ScanFinding:
    """One stable safety finding."""

    code: str
    rule: str
    detail: str = ""


@dataclass(frozen=True)
class ScanReport:
    """Findings plus the display-only sanitized form of the input."""

    findings: tuple[ScanFinding, ...]
    sanitized: str

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(finding.code for finding in self.findings))

    @property
    def rules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(finding.rule for finding in self.findings))

    def first(self) -> ScanFinding | None:
        return self.findings[0] if self.findings else None


# ---------------------------------------------------------------------------
# Control-character handling
# ---------------------------------------------------------------------------


def _zwj_is_emoji(text: str, index: int) -> bool:
    left = index - 1
    while left >= 0 and ord(text[left]) == _VARIATION_SELECTOR:
        left -= 1
    right = index + 1
    while right < len(text) and ord(text[right]) == _VARIATION_SELECTOR:
        right += 1
    if left < 0 or right >= len(text):
        return False
    return all(
        any(low <= ord(text[position]) <= high for low, high in _EMOJI_RANGES)
        for position in (left, right)
    )


def _strip_controls(text: str) -> tuple[str, list[ScanFinding]]:
    """Remove hidden/bidi/C0/C1 controls, keeping legitimate emoji joiners."""

    findings: list[ScanFinding] = []
    seen: set[tuple[str, str]] = set()
    flag_positions: set[int] = set()
    for match in _RGI_RE.finditer(text):
        flag_positions.update(range(match.start() + 1, match.end()))
    kept: list[str] = []
    for index, char in enumerate(text):
        if index in flag_positions:
            kept.append(char)
            continue
        codepoint = ord(char)
        if char in _BIDI_CHARS:
            finding = ScanFinding("bidi_control", "bidi_control", f"U+{codepoint:04X}")
            if (finding.code, finding.rule) not in seen:
                seen.add((finding.code, finding.rule))
                findings.append(finding)
            continue
        if char in _HIDDEN_CHARS:
            if char == "‍" and _zwj_is_emoji(text, index):
                kept.append(char)
                continue
            finding = ScanFinding("hidden_character", "invisible_unicode", f"U+{codepoint:04X}")
            if (finding.code, finding.rule) not in seen:
                seen.add((finding.code, finding.rule))
                findings.append(finding)
            continue
        if (codepoint < 0x20 and char not in "\n\r\t") or 0x7F <= codepoint <= 0x9F:
            finding = ScanFinding("control_character", "c0_control", f"U+{codepoint:04X}")
            if (finding.code, finding.rule) not in seen:
                seen.add((finding.code, finding.rule))
                findings.append(finding)
            continue
        kept.append(char)
    return "".join(kept), findings


# ---------------------------------------------------------------------------
# Threat patterns
# ---------------------------------------------------------------------------

_FILLER = r"(?:\w{1,32}\s{1,16}){0,6}"
_SECRET_VAR = r"\$\{?\w{0,128}(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b"

_PATTERNS: tuple[tuple[str, str, str], ...] = (
    # (regex, rule id, error code)
    (
        rf"ignore\s{{1,16}}{_FILLER}(previous|all|above|prior)\s{{1,16}}{_FILLER}instructions",
        "prompt_injection",
        "prompt_injection",
    ),
    (r"system\s{1,16}prompt\s{1,16}override", "sys_prompt_override", "prompt_injection"),
    (
        rf"disregard\s{{1,16}}{_FILLER}(your|all|any)\s{{1,16}}{_FILLER}"
        r"(instructions|rules|guidelines)",
        "disregard_rules",
        "prompt_injection",
    ),
    (
        rf"forget\s{{1,16}}(?:about\s{{1,16}})?(your|all|any|previous|prior|above|earlier)"
        r"\s{1,16}(?:\w{1,32}\s{1,16}){0,3}(instructions|rules|guidelines)",
        "forget_rules",
        "prompt_injection",
    ),
    (r"do\s{1,16}not\s{1,16}tell\s{1,16}the\s{1,16}user", "deception_hide", "deception"),
    (
        rf"act\s{{1,16}}as\s{{1,16}}(?:if|though)\s{{1,16}}{_FILLER}you\s{{1,16}}"
        r"(?:have\s{1,16}no|don'?t\s{1,16}have)\s{1,16}"
        r"(?:restrictions|limits|rules)",
        "bypass_restrictions",
        "prompt_injection",
    ),
    (
        r"<!--[^>]{0,256}(?:ignore|override|system|secret|hidden)[^>]{0,256}-->",
        "html_comment_injection",
        "prompt_injection",
    ),
    (
        r"<\s{0,16}div\s{1,16}style\s{0,16}=\s{0,16}[\"'][^>]{0,1024}"
        r"display\s{0,16}:\s{0,16}none",
        "hidden_div",
        "hidden_content",
    ),
    # Exfiltration commands.
    (rf"curl\s{{1,16}}[^\n]{{0,1024}}{_SECRET_VAR}", "exfil_curl", "exfiltration"),
    (rf"wget\s{{1,16}}[^\n]{{0,1024}}{_SECRET_VAR}", "exfil_wget", "exfiltration"),
    (r"cat\s{1,16}[^\n]{0,1024}(\.env|credentials|\.netrc|\.pgpass|id_rsa|id_ed25519)", "read_secrets", "secret_access"),
    (r"authorized_keys", "ssh_backdoor", "secret_access"),
    (r"/etc/sudoers|visudo", "sudoers_mod", "destructive"),
    (r"rm\s{1,16}-rf\s{1,16}/", "destructive_root_rm", "destructive"),
)

_COMPILED_PATTERNS = tuple(
    (re.compile(pattern, re.IGNORECASE), rule, code)
    for pattern, rule, code in _PATTERNS
)
_WS_RUN = re.compile(r"\s{2,}")


def _collapse_ws(match: re.Match[str]) -> str:
    return "\n" if "\n" in match.group() else " "


# ---------------------------------------------------------------------------
# Secret preflight
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private_key"),
    (r"\b(?:sk|rk)-[A-Za-z0-9_-]{16,}\b", "api_key"),
    (r"\bsk-ant-[A-Za-z0-9_-]{16,}\b", "anthropic_key"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "github_token"),
    (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "github_token"),
    (r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", "slack_token"),
    (r"\bAKIA[0-9A-Z]{16}\b", "aws_access_key"),
    (r"\bAIza[0-9A-Za-z_-]{30,}\b", "google_api_key"),
    (r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "jwt"),
    (r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*\b", "bearer_token"),
    (
        r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd)\s{0,8}[:=]\s{0,8}[\"']?"
        r"(?!(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)[\"']?)"
        r"[A-Za-z0-9+/=_~.-]{16,}",
        "secret_assignment",
    ),
)
_COMPILED_SECRETS = tuple(
    (re.compile(pattern, re.IGNORECASE), rule) for pattern, rule in _SECRET_PATTERNS
)

# Intrinsic shapes that have no vendor prefix to match. A truncated or header-
# only private key is still a credential, and an opaque mixed-case token with
# digits is credential-shaped even when no ``api_key=`` label accompanies it.
# Hex digests, UUIDs and all-one-case identifiers are excluded so ordinary
# content hashes are not mistaken for secrets.
_PRIVATE_KEY_BEGIN_RE = re.compile(
    r"(?:-----BEGIN(?: [A-Z0-9 ]*)?PRIVATE KEY-----|"
    r"\bBEGIN\s+(?:RSA|DSA|EC|OPENSSH|ENCRYPTED|PGP)?\s*PRIVATE KEY\b)",
    re.IGNORECASE,
)
_PRIVATE_KEY_END_RE = re.compile(
    r"-----END(?: [A-Z0-9 ]*)?PRIVATE KEY-----", re.IGNORECASE
)
_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL | re.IGNORECASE,
)
# A header without an end marker is still followed by key material; redact the
# rest of the value rather than leaving the base64 body in a durable sink.
_PRIVATE_KEY_TAIL_RE = re.compile(
    r"(?:-----BEGIN(?: [A-Z0-9 ]*)?PRIVATE KEY-----|"
    r"\bBEGIN\s+(?:RSA|DSA|EC|OPENSSH|ENCRYPTED|PGP)?\s*PRIVATE KEY\b)[\s\S]*",
    re.IGNORECASE,
)
_OPAQUE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_-]{36,}(?![A-Za-z0-9_+/=-])")
_HEXISH_RE = re.compile(r"[0-9A-Fa-f-]+\Z")


def _is_opaque_token(candidate: str) -> bool:
    """Return True for a vendor-less, high-shape credential token."""

    if len(candidate) < 36 or _HEXISH_RE.fullmatch(candidate):
        return False
    if candidate.lower() == candidate or candidate.upper() == candidate:
        return False
    return any(char.isdigit() for char in candidate)


def _redact_opaque_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "[redacted]" if _is_opaque_token(match.group(0)) else match.group(0)

    return _OPAQUE_TOKEN_RE.sub(replace, text)


def redact_secrets(text: str) -> str:
    """Mask detected credential-looking spans for display/logging."""

    cleaned = text or ""
    # Strip a whole PEM block before the generic credential patterns can erase
    # its header and leave the base64 body exposed.
    cleaned = _PEM_BLOCK_RE.sub("[redacted]", cleaned)
    for pattern, _rule in _COMPILED_SECRETS:
        cleaned = pattern.sub("[redacted]", cleaned)
    cleaned = _PRIVATE_KEY_TAIL_RE.sub("[redacted]", cleaned)
    return _redact_opaque_tokens(cleaned)


def _scan_secrets(text: str) -> list[ScanFinding]:
    if not text:
        return []
    folded = unicodedata.normalize("NFKC", text)
    findings: list[ScanFinding] = []
    for pattern, rule in _COMPILED_SECRETS:
        if pattern.search(folded):
            findings.append(ScanFinding("secret_detected", rule))
    if _PRIVATE_KEY_BEGIN_RE.search(folded) and not _PRIVATE_KEY_END_RE.search(folded):
        findings.append(ScanFinding("secret_detected", "unmatched_private_key"))
    if any(_is_opaque_token(match.group(0)) for match in _OPAQUE_TOKEN_RE.finditer(folded)):
        findings.append(ScanFinding("secret_detected", "unmatched_token"))
    return findings


def contains_secret(text: str) -> bool:
    if not text:
        return False
    return bool(_scan_secrets(text))


# ---------------------------------------------------------------------------
# Public scanning API
# ---------------------------------------------------------------------------


def _scan_patterns(text: str) -> list[ScanFinding]:
    if not text:
        return []
    folded = unicodedata.normalize("NFKC", text)
    normalized = _WS_RUN.sub(_collapse_ws, folded)
    findings: list[ScanFinding] = []
    for pattern, rule, code in _COMPILED_PATTERNS:
        if pattern.search(normalized):
            findings.append(ScanFinding(code, rule))
    return findings


def scan_text(text: str, *, scope: str = "prompt") -> ScanReport:
    """Scan text for controls/injection/exfiltration/secrets.

    ``scope="display"`` skips rejection-oriented patterns and only reports
    control characters, which is useful for labels that are already trusted but
    must not contain invisible directionality.
    """

    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) > _MAX_SCAN_CHARS:
        # Bound worst-case work. This is a rejection surface, not a content
        # renderer; refuse rather than scan an unbounded body.
        finding = ScanFinding("too_long", "max_scan_chars")
        return ScanReport((finding,), sanitize_for_display(text))
    stripped, findings = _strip_controls(text)
    if scope == "display":
        return ScanReport(tuple(findings), sanitize_for_display(text))
    findings = [*findings, *_scan_patterns(stripped), *_scan_secrets(stripped)]
    return ScanReport(tuple(findings), sanitize_for_display(text))


def scan_prompt(prompt: str) -> ScanReport:
    return scan_text(prompt, scope="prompt")


def sanitize_for_display(text: str) -> str:
    """Display-only sanitization: strip controls/hidden chars and redact secrets.

    Never use the result as the persisted value; unsafe input is rejected, not
    silently rewritten.
    """

    if not isinstance(text, str):
        text = "" if text is None else str(text)
    stripped, _ = _strip_controls(text)
    return redact_secrets(stripped)


def require_safe(text: str, *, field: str = "prompt", scope: str = "prompt") -> str:
    """Return ``text`` when safe, else raise :class:`ScheduleSafetyError`.

    The exception never contains the offending text, only the stable code, rule
    id and a generic message, so a caller cannot accidentally log a secret by
    reporting a failed preflight.
    """

    report = scan_text(text, scope=scope)
    first = report.first()
    if first is None:
        return text
    raise ScheduleSafetyError(
        first.code,
        first.rule,
        f"{field} rejected by schedule safety scan ({first.code}/{first.rule}).",
    )


def scan_fields(fields: Iterable[tuple[str, str]]) -> ScanReport:
    """Scan named fields and return one merged report.

    The sanitized string joins field labels, never the raw values, so a caller
    cannot persist rejected text from this object.
    """

    findings: list[ScanFinding] = []
    for label, value in fields:
        for finding in scan_text(value, scope="prompt").findings:
            findings.append(
                ScanFinding(finding.code, finding.rule, label or finding.detail)
            )
    return ScanReport(tuple(findings), "")


__all__ = [
    "ScanFinding",
    "ScanReport",
    "ScheduleSafetyError",
    "contains_secret",
    "redact_secrets",
    "require_safe",
    "sanitize_for_display",
    "scan_fields",
    "scan_prompt",
    "scan_text",
]
