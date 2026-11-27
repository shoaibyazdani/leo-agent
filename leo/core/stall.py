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
# A short acknowledgement carries no information: only then can an internal
# plan justify another model call. A visible answer (including a one-line
# reaction to a photo) is never overruled by private reasoning that happened
# to mention future work.
_GENERIC_ACK = re.compile(
    r"(?is)^\s*(?:done|ok(?:ay)?|sure|yes|right|got it|thanks|thank you|"
    r"complete|finished|all done|that['’]s all|ready)\s*[.!]?\s*$"
)
_IN_PROGRESS = re.compile(
    r"(?is)\b(?:still (?:working|checking|running|looking|doing)|"
    r"currently (?:working|checking|running|looking|doing)|"
    r"in progress|working on it|hang on|hold on)\b"
)


def is_direct_reply(content: str) -> bool:
    """Whether visible text answers the user instead of announcing work.

    This is deliberately about the text the user can see. Private reasoning
    is not a reply and must never turn an answer into a stall.
    """
    text = str(content or "").strip()
    if not text or text.endswith((":", "…", "...")):
        return False
    if _GENERIC_ACK.match(text) or _PLAN_TAIL.search(text) or _IN_PROGRESS.search(text):
        return False
    return True


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
    # An answer in the visible channel ends the turn, even when the model's
    # private reasoning went on to plan unrelated work.
    if is_direct_reply(text):
        return False
    if _IN_PROGRESS.search(text):
        return True
    tail = (str(reasoning_tail or "") or text)[-max(1, int(tail_chars)):]
    # Check the visible text too: a reasoning tail must not hide an announced
    # action that never happened.
    if _PLAN_TAIL.search(tail) or _PLAN_TAIL.search(text[-max(1, int(tail_chars)):]):
        return True
    stripped = tail.rstrip()
    return stripped.endswith((":", "…", "...")) and bool(_PLAN_TAIL.search(stripped))


__all__ = [
    "STALL_MAX_REPLY_CHARS", "STALL_TAIL_CHARS", "is_direct_reply", "looks_like_stall",
]
