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
# A progress report is narration, not an answer: "Found the file. Let me now
# patch it." still promises work and must not become a delivered reply.
_WORK_NARRATION = re.compile(
    r"(?is)^\s*(?:(?:i|we)(?:'ve|'m|'ll| am| are| have)?\s+)?"
    r"(?:found|locat(?:ed|ing)|check(?:ed|ing)|look(?:ed|ing)|search(?:ed|ing)|"
    r"read(?:ing)?|inspect(?:ed|ing)|retriev(?:ed|ing)|download(?:ed|ing)|"
    r"pull(?:ed|ing)|ran|running|work(?:ed|ing)|patch(?:ed|ing)|"
    r"fetch(?:ed|ing)|open(?:ed|ing))\b"
)


def _answers_before(prefix: str) -> bool:
    """Whether text before a visible plan phrase is itself an answer."""
    text = prefix.strip(" \t\r\n.,!?;:…\"'“”‘’()[]-")
    if not text or len(text.split()) < 2:
        return False
    return not (
        _GENERIC_ACK.match(text)
        or _IN_PROGRESS.search(text)
        or _WORK_NARRATION.match(text)
    )


def is_direct_reply(content: str) -> bool:
    """Whether visible text answers the user instead of announcing work.

    This is deliberately about the text the user can see. Private reasoning
    is not a reply and must never turn an answer into a stall. A plan phrase
    only disqualifies the reply when nothing substantive precedes it: an
    answer followed by visible next steps is still a direct reply.
    """
    text = str(content or "").strip()
    if not text or text.endswith((":", "…", "...")):
        return False
    if _GENERIC_ACK.match(text) or _IN_PROGRESS.search(text):
        return False
    plan = _PLAN_TAIL.search(text)
    return plan is None or _answers_before(text[:plan.start()])


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
