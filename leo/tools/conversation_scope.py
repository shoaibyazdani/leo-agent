"""Shared conversation scoping for tools that read transcripts.

The current conversation is an execution-context fact, never a tool argument:
the executor derives it from the authenticated ``ToolCall``. Tools that accept an
optional ``conversation_id`` use these helpers so the binding rules cannot drift
between the context inspector and the history reader.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TRUSTED_CONVERSATION_KEYS = ("conversation_id", "current_conversation_id")
_QUOTA_PREFIXES = ("task:", "correlation:", "actor:")


def current_conversation(ctx: Any) -> str | None:
    """Return the conversation the authenticated call is bound to, if any."""
    if not isinstance(ctx, Mapping):
        return None
    current: str | None = None
    for key in _TRUSTED_CONVERSATION_KEYS:
        value = ctx.get(key)
        if isinstance(value, str) and value:
            current = value
            break
    if current is None:
        # A task-scoped chat call is the strongest binding; the executor derives
        # it from ToolCall.task_id, which the chat app sets to the conversation.
        value = ctx.get("task_id")
        if isinstance(value, str) and value:
            candidate = value[len("task:"):] if value.startswith("task:") else value
            if (candidate and candidate != "global"
                    and not candidate.startswith(_QUOTA_PREFIXES)):
                current = candidate
        if current is None:
            value = ctx.get("correlation_id")
            if isinstance(value, str) and value and not value.startswith(_QUOTA_PREFIXES):
                current = value
    if current is None:
        return None
    # Optional composition-root ownership metadata: if supplied, the actor must
    # own the conversation before any of its messages are readable.
    owners = ctx.get("conversation_actors") or ctx.get("conversation_owners")
    if isinstance(owners, Mapping):
        allowed = owners.get(current)
        actor = ctx.get("actor")
        if allowed is not None and actor is not None and str(allowed) != str(actor):
            return None
    return current


def assert_conversation_visible(ctx: Any, conversation_id: str) -> None:
    """Refuse a read when the execution context names a different owner.

    Absent ownership metadata is not a denial: a bound chat turn can read its own
    conversation, and a caller that explicitly names another conversation gets the
    tool's bounded cross-conversation window. Metadata, when present, wins.
    """
    if not isinstance(ctx, Mapping):
        return
    owners = ctx.get("conversation_actors") or ctx.get("conversation_owners")
    if not isinstance(owners, Mapping):
        return
    allowed = owners.get(conversation_id)
    actor = ctx.get("actor")
    if allowed is not None and actor is not None and str(allowed) != str(actor):
        raise ValueError("conversation is not visible to this actor")


__all__ = ["assert_conversation_visible", "current_conversation"]
