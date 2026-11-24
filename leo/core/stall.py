"""Detect a turn that announced work and then stopped without acting.

Deliberately narrow: a normal answer, a long answer, or any response carrying
tool calls is never a stall. The caller adds one bounded nudge, then accepts the
next response as final.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

STALL_MAX_REPLY_CHARS = 400
STALL_TAIL_CHARS = 160

_LENGTH_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
_PLAN_TAIL = re.compile(
    r"(?is)(?:"
    r"\b(?:let me|i['’]ll|i will|i['’]m going to|i am going to|we['’]ll|we will)\b"
    r"|\b(?:next|then|plan|steps?|todo)\b\s*:"
    r"|\bfirst,? i['’]ll\b"
    r")"
)


def looks_like_stall(
    content: str,
    finish_reason: str | None,
    tool_calls: Sequence[Any],
    *,
    reasoning_tail: str = "",
    max_reply_chars: int = STALL_MAX_REPLY_CHARS,
    tail_chars: int = STALL_TAIL_CHARS,
) -> bool:
    """Whether this response announced an action but did not perform one."""
    if tool_calls:
        return False
    text = str(content or "").strip()
    if not text:
        return False
    if str(finish_reason or "").lower() in _LENGTH_FINISH_REASONS:
        # Truncated mid-answer: another bounded call can finish it.
        return True
    if len(text) > max_reply_chars:
        return False
    tail = (str(reasoning_tail or "") or text)[-max(1, int(tail_chars)):]
    if _PLAN_TAIL.search(tail):
        return True
    stripped = tail.rstrip()
    return stripped.endswith((":", "…", "...")) and bool(_PLAN_TAIL.search(stripped))


__all__ = ["STALL_MAX_REPLY_CHARS", "STALL_TAIL_CHARS", "looks_like_stall"]
