"""Per-context "this content was freshly read" marks for skill edits.

A patch must be based on what the caller actually loaded, not on text quoted
earlier in a transcript (which may be stale or fabricated). ``skill_read``
records the skill (and, for supporting files, the exact path) it returned;
``skill_manage`` refuses a write until the matching mark exists.

Marks live in a ContextVar so concurrent conversations, agents and background
reviews are isolated. A review resets the marks at the start of its run; a
foreground chat accumulates them for the session, which is the lifetime LEO's
durable transcript model gives a single conversation.

This is a guard, not an authorization: the caller still has to pass the tool
executor's permission and risk checks.
"""

from __future__ import annotations

from contextvars import ContextVar

SKILL_DOCUMENT = "SKILL.md"

_marks: ContextVar[frozenset[tuple[str, str]]] = ContextVar(
    "leo_skill_read_marks", default=frozenset()
)


def mark_skill_read(name: str, file_path: str | None = None) -> None:
    key = (str(name), str(file_path or SKILL_DOCUMENT))
    _marks.set(_marks.get() | {key})


def was_skill_read(name: str, file_path: str | None = None) -> bool:
    return (str(name), str(file_path or SKILL_DOCUMENT)) in _marks.get()


def reset_skill_read_marks() -> None:
    _marks.set(frozenset())


def read_marks() -> frozenset[tuple[str, str]]:
    """Copy of the current marks (diagnostics and tests only)."""
    return _marks.get()


__all__ = [
    "SKILL_DOCUMENT",
    "mark_skill_read",
    "read_marks",
    "reset_skill_read_marks",
    "was_skill_read",
]
