"""Durable reasoning settings and per-turn inheritance for ``/reasoning``.

Two scopes exist:

* **global** -- one persistent default for LEO, stored in the shared ``kv``
  table under ``reasoning:global`` so it survives a restart.
* **session** -- an override for one conversation, stored under
  ``reasoning:session:<conversation_id>``. It never leaks to another
  conversation or another user.

Each scope holds an effort and a display flag. A missing key means "inherit";
``False`` is a real value (hide), so the state distinguishes unset from
explicitly off. ``resolve()`` combines session > global > config and reports
where each effective value came from, for truthful status output.

The module also owns the per-turn context variable subagents inherit. A parent
chat turn sets it once; an agent worker reads it when it builds its own model
request. An explicit value in the agent's context always wins.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

from .. import db
from ..contracts import utcnow
from ..logging import get_logger
from .reasoning import EFFORT_LADDER, parse_effort

log = get_logger("models.reasoning_state")

_PREFIX = "reasoning:session:"
_GLOBAL_KEY = "reasoning:global"
SCOPES: tuple[str, ...] = ("session", "global")


def _key(conversation_id: str) -> str:
    return f"{_PREFIX}{conversation_id}"


def _default_state() -> dict[str, Any]:
    # ``None`` means inherit from the outer scope. ``False`` is explicit.
    return {"effort": None, "show": None, "full": None, "scope": "session"}


def _global_state() -> dict[str, Any]:
    return {"effort": None, "show": None}


def _merged(defaults: dict[str, Any], value: Any) -> dict[str, Any]:
    merged = dict(defaults)
    if isinstance(value, dict):
        merged.update(value)
    return merged


async def _read(key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
    if not row:
        return dict(defaults)
    try:
        return _merged(defaults, json.loads(row["value"]))
    except (ValueError, TypeError):
        return dict(defaults)


async def _write(key: str, state: dict[str, Any]) -> None:
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, json.dumps(state), utcnow()),
    )


def _scope(value: Any) -> str:
    scope = str(value or "").strip().lower()
    return scope if scope in SCOPES else "session"


def _bool_or_none(value: Any) -> bool | None:
    # bool is a subclass of int; only an actual bool is an explicit choice.
    return value if isinstance(value, bool) else None


class ReasoningState:
    """Read/write reasoning overrides without touching live config."""

    # ---- session scope -------------------------------------------------
    async def get(self, conversation_id: str | None) -> dict[str, Any]:
        if not conversation_id:
            return _default_state()
        return await _read(_key(conversation_id), _default_state())

    async def set_effort(self, conversation_id: str, effort: str | None) -> dict[str, Any]:
        state = await self.get(conversation_id)
        state["effort"] = parse_effort(effort)
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

    async def set_scope(self, conversation_id: str, scope: str) -> dict[str, Any]:
        state = await self.get(conversation_id)
        state["scope"] = _scope(scope)
        await _write(_key(conversation_id), state)
        return state

    async def clear(self, conversation_id: str) -> dict[str, Any]:
        """Drop this conversation's overrides, keeping its chosen edit scope."""
        current = await self.get(conversation_id)
        state = _default_state()
        state["scope"] = _scope(current.get("scope"))
        await _write(_key(conversation_id), state)
        return state

    async def effort_for(self, conversation_id: str | None) -> str | None:
        """Return the validated session effort, or None when unset.

        Kept for callers that ask specifically about the session override; use
        :meth:`resolve` for the effective value across both scopes.
        """
        state = await self.get(conversation_id)
        return parse_effort(state.get("effort"))

    # ---- global scope --------------------------------------------------
    async def global_state(self) -> dict[str, Any]:
        return await _read(_GLOBAL_KEY, _global_state())

    async def global_effort(self) -> str | None:
        state = await self.global_state()
        return parse_effort(state.get("effort"))

    async def set_global(self, effort: str | None) -> str | None:
        state = await self.global_state()
        state["effort"] = parse_effort(effort)
        await _write(_GLOBAL_KEY, state)
        return parse_effort(effort)

    async def set_global_display(self, show: bool | None) -> bool | None:
        state = await self.global_state()
        state["show"] = bool(show) if show is not None else None
        await _write(_GLOBAL_KEY, state)
        return state["show"]

    async def clear_global(self) -> None:
        await _write(_GLOBAL_KEY, _global_state())

    # ---- effective view ------------------------------------------------
    async def resolve(
        self,
        conversation_id: str | None,
        *,
        default_effort: str | None = None,
        default_show: bool = False,
    ) -> dict[str, Any]:
        """Combine session > global > config into one effective selection.

        ``effort_source``/``show_source`` are ``session``, ``global``,
        ``config`` or ``unset``. The returned ``scope`` is where a menu change
        is written next; it is per-conversation UI state, not a config value.
        """
        session = await self.get(conversation_id)
        global_state = await self.global_state()

        session_effort = parse_effort(session.get("effort"))
        global_effort = parse_effort(global_state.get("effort"))
        config_effort = parse_effort(default_effort)
        if session_effort:
            effort, effort_source = session_effort, "session"
        elif global_effort:
            effort, effort_source = global_effort, "global"
        elif config_effort:
            effort, effort_source = config_effort, "config"
        else:
            effort, effort_source = None, "unset"

        session_show = _bool_or_none(session.get("show"))
        global_show = _bool_or_none(global_state.get("show"))
        if session_show is not None:
            show, show_source = session_show, "session"
        elif global_show is not None:
            show, show_source = global_show, "global"
        else:
            show, show_source = bool(default_show), "config"

        return {
            "effort": effort,
            "effort_source": effort_source,
            "show": show,
            "show_source": show_source,
            "full": bool(session.get("full")),
            "scope": _scope(session.get("scope")),
            "session_effort": session_effort,
            "global_effort": global_effort,
            "config_effort": config_effort,
            "session_show": session_show,
            "global_show": global_show,
            "levels": list(EFFORT_LADDER),
        }


# ---------------------------------------------------------------------------
# Inheritance for subagents and background work


_CURRENT: ContextVar[dict[str, Any] | None] = ContextVar(
    "leo_reasoning_selection", default=None
)


def current_reasoning() -> dict[str, Any] | None:
    """The effective selection of the turn an agent worker is running under."""
    return _CURRENT.get()


def set_current_reasoning(selection: dict[str, Any] | None) -> Token:
    """Install a selection for this async context; return the reset token."""
    return _CURRENT.set(dict(selection) if selection else None)


def reset_current_reasoning(token: Token) -> None:
    _CURRENT.reset(token)


@contextmanager
def reasoning_scope(selection: dict[str, Any] | None) -> Iterator[None]:
    token = set_current_reasoning(selection)
    try:
        yield
    finally:
        reset_current_reasoning(token)


def inheritance_payload(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact, serializable form placed in an agent delegation context."""
    if not selection:
        return None
    return {
        "effort": selection.get("effort"),
        "show": bool(selection.get("show")),
        "source": selection.get("effort_source"),
        "full": bool(selection.get("full")),
    }


def selection_from_context(value: Any) -> dict[str, Any] | None:
    """Read an explicit reasoning override from a delegation context.

    Accepts the payload produced by :func:`inheritance_payload`, a bare effort
    string, or a mapping with ``effort``/``show``. Returns None when the value
    names no usable setting, so the caller can inherit instead.
    """
    if value is None:
        return None
    if isinstance(value, str):
        effort = parse_effort(value)
        return {"effort": effort, "show": False, "source": "context", "full": False} if effort else None
    if not isinstance(value, dict):
        return None
    effort = parse_effort(value.get("effort"))
    show = value.get("show")
    if effort is None and show is None:
        return None
    return {
        "effort": effort,
        "show": bool(show) if isinstance(show, bool) else False,
        "source": str(value.get("source") or "context"),
        "full": bool(value.get("full")),
    }


__all__ = [
    "EFFORT_LADDER", "ReasoningState", "SCOPES",
    "current_reasoning", "inheritance_payload", "reasoning_scope",
    "reset_current_reasoning", "selection_from_context", "set_current_reasoning",
]
