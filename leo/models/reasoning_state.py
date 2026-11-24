"""Durable per-conversation reasoning state for the /reasoning command.

Session overrides live in the shared ``kv`` table so they survive a process
restart and are visible to every surface (CLI, API, gateway) that shares the
database. Display toggles (show/hide, full/clamp) are session-scoped and stored
alongside the effort. Nothing here calls a model or mutates config.
"""

from __future__ import annotations

import json
from typing import Any

from .. import db
from ..contracts import utcnow
from ..logging import get_logger
from .reasoning import EFFORT_LADDER, parse_effort

log = get_logger("models.reasoning_state")

_PREFIX = "reasoning:session:"
_GLOBAL_KEY = "reasoning:global"


def _key(conversation_id: str) -> str:
    return f"{_PREFIX}{conversation_id}"


def _default_state() -> dict[str, Any]:
    return {"effort": None, "show": False, "full": False}


async def _read(key: str) -> dict[str, Any]:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
    if not row:
        return _default_state()
    try:
        value = json.loads(row["value"])
        return value if isinstance(value, dict) else _default_state()
    except (ValueError, TypeError):
        return _default_state()


async def _write(key: str, state: dict[str, Any]) -> None:
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, json.dumps(state), utcnow()),
    )


class ReasoningState:
    """Read/write reasoning overrides without touching live config."""

    async def get(self, conversation_id: str | None) -> dict[str, Any]:
        if not conversation_id:
            return _default_state()
        return await _read(_key(conversation_id))

    async def set_effort(self, conversation_id: str, effort: str | None) -> dict[str, Any]:
        state = await self.get(conversation_id)
        state["effort"] = effort
        await _write(_key(conversation_id), state)
        return state

    async def set_display(
        self, conversation_id: str, *, show: bool | None = None, full: bool | None = None
    ) -> dict[str, Any]:
        state = await self.get(conversation_id)
        if show is not None:
            state["show"] = bool(show)
        if full is not None:
            state["full"] = bool(full)
        await _write(_key(conversation_id), state)
        return state

    async def clear(self, conversation_id: str) -> dict[str, Any]:
        state = _default_state()
        await _write(_key(conversation_id), state)
        return state

    async def effort_for(self, conversation_id: str | None) -> str | None:
        """Return the validated session effort, or None when unset."""
        state = await self.get(conversation_id)
        return parse_effort(state.get("effort"))

    async def global_effort(self) -> str | None:
        state = await _read(_GLOBAL_KEY)
        return parse_effort(state.get("effort"))

    async def set_global(self, effort: str | None) -> str | None:
        state = await _read(_GLOBAL_KEY)
        state["effort"] = effort
        await _write(_GLOBAL_KEY, state)
        return parse_effort(effort)


__all__ = ["EFFORT_LADDER", "ReasoningState"]
