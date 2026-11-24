"""Continuation intent and resume instructions — pure helpers, no I/O.

A user message such as "continue" is not a new objective. The durable transcript
already holds the objective and the last state; these helpers recognise the
continuation shape and build a bounded instruction that quotes that state as
data, so the model resumes instead of re-scanning history.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

CONTINUATION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"continue|keep going|go on|carry on|proceed|resume|"
    r"finish (?:it|that|the task|the work)|"
    r"as you were|where were you|do the needful"
    r")\s*[.!]?\s*$",
    re.IGNORECASE,
)
ACK_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|k|thanks|thank you|got it|yes|no|sure|cool|nice)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_SYNTHETIC_META_KEYS = ("slash", "command", "resume", "synthetic")


def is_continuation(text: str) -> bool:
    """Whether the whole message is a bare continuation prompt."""
    return bool(CONTINUATION_RE.match(str(text or "")))


def is_ack(text: str) -> bool:
    return bool(ACK_RE.match(str(text or "")))


def is_synthetic_row(metadata: Any) -> bool:
    """True for slash-command, resume-marker or other non-objective rows."""
    if not isinstance(metadata, Mapping):
        return False
    if any(metadata.get(key) for key in _SYNTHETIC_META_KEYS):
        return True
    return str(metadata.get("role_kind") or "") in {"command", "synthetic"}


def build_resume_instruction(
    objective: str, state: Mapping[str, Any] | None = None
) -> str:
    """Quote the earlier objective and last known state, then ask to continue.

    The objective is wrapped as data, never as a fresh instruction: it came from
    the transcript, which is untrusted relative to the system prompt.
    """
    text = str(objective or "").strip()[:4_000]
    lines = [
        "RESUME the earlier objective in this conversation.",
        "Do not restart, and do not scan conversation history to rediscover it.",
        "",
        "Original objective (quoted data):",
        text,
    ]
    state = state or {}
    status = str(state.get("run_status") or "").strip()
    pending = state.get("tool_calls_pending") or []
    if status or pending:
        lines.append("")
        lines.append("Last known state:")
        if status:
            lines.append(f"- run_status: {status}")
        if pending:
            names = ", ".join(str(item) for item in list(pending)[:8])
            lines.append(f"- tools that were requested but not executed: {names}")
    lines += [
        "",
        "Continue from that state. Verify anything you must, then finish the work.",
    ]
    return "\n".join(lines)


__all__ = [
    "ACK_RE",
    "CONTINUATION_RE",
    "build_resume_instruction",
    "is_ack",
    "is_continuation",
    "is_synthetic_row",
]
