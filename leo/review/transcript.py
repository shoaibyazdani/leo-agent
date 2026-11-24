"""Bounded, durable transcript loader for a post-turn review.

The review replays the conversation it is learning from. It must never trust
in-memory history (which may be a retried or partially streamed view) and it
must never replay an unbounded transcript into a model call. This module reads
the durable ``messages`` rows, keeps a bounded recent window, truncates tool
output, drops turns that did not finish, and returns a digest that can be
written into an audit artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .. import db
from ..contracts import ChatMessage
from ..skills.sanitize import strip_control

_UNFINISHED = frozenset({"error", "cancelled", "canceled", "incomplete", "failed"})
_TRAILER = "\n…[truncated]"


@dataclass(frozen=True)
class Transcript:
    """A bounded view of one conversation plus a content digest."""

    conversation_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    # Durable message ids parallel to ``messages``; used as evidence refs.
    ids: list[int] = field(default_factory=list)
    digest: str = ""
    chars: int = 0
    skipped: int = 0

    @property
    def empty(self) -> bool:
        return not self.messages

    def evidence_ids(self, limit: int = 8) -> list[int]:
        return self.ids[-max(1, limit):]


def _metadata(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _bounded(text: Any, limit: int) -> str:
    value = strip_control(str(text or ""))
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit].rstrip() + _TRAILER


def _digest(messages: list[ChatMessage]) -> str:
    payload = [{"role": message.role, "content": message.content} for message in messages]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()


async def load_transcript(
    conversation_id: str,
    *,
    max_messages: int = 80,
    max_chars: int = 60_000,
    tool_preview_chars: int = 500,
) -> Transcript:
    """Load the newest bounded window of a conversation.

    Rows are read newest-first so the character budget always keeps the end of
    the conversation (the part that contains the result and any correction),
    then reversed into chronological order. Turns that did not complete are
    not evidence and are skipped.
    """
    rows = await db.fetchall(
        "SELECT id, role, content, metadata FROM messages "
        "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
        (conversation_id, max(1, max_messages)),
    )
    selected: list[ChatMessage] = []
    ids: list[int] = []
    chars = 0
    skipped = 0
    for row in rows:
        role = str(row["role"] or "")
        meta = _metadata(row["metadata"])
        if role == "assistant" and str(meta.get("run_status") or "").lower() in _UNFINISHED:
            skipped += 1
            continue
        content = _bounded(
            row["content"], tool_preview_chars if role == "tool" else max_chars
        )
        if not content.strip():
            skipped += 1
            continue
        if selected and chars + len(content) > max_chars:
            break
        selected.append(ChatMessage(role=role, content=content))
        try:
            ids.append(int(row["id"]))
        except (TypeError, ValueError):
            ids.append(0)
        chars += len(content)
    selected.reverse()
    ids.reverse()
    return Transcript(
        conversation_id=str(conversation_id),
        messages=selected,
        ids=ids,
        digest=_digest(selected),
        chars=chars,
        skipped=skipped,
    )


__all__ = ["Transcript", "load_transcript"]
