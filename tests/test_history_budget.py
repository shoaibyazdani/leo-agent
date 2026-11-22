"""Counting messages is not a context bound.

Fifty rows of shell output can be tens of megabytes. Without a character budget the
model is handed a flooded prompt: it answers an older topic, or returns nothing at
all (ModelEmptyOutputError). The budget keeps the recent exchange and drops the
oldest rows, never orphaning a tool result from the assistant row that opened it.
"""

from __future__ import annotations

from leo.core.app import LeoApp


class _Self:
    CONVERSATION_HISTORY_MAX_CHARS = 1_000


def _row(role: str, chars: int) -> dict:
    return {"id": 1, "role": role, "content": "x" * chars, "metadata": "{}"}


def test_a_giant_tool_dump_is_trimmed_away() -> None:
    rows = [_row("user", 50), _row("assistant", 50), _row("tool", 5_000), _row("user", 50)]
    kept = LeoApp._budget_history(_Self(), rows)

    assert len(kept) < len(rows)
    assert kept[-1]["content"] == "x" * 50, "the newest row must survive"


def test_the_result_never_starts_with_an_orphaned_tool_row() -> None:
    rows = [_row("assistant", 20), _row("tool", 4_000), _row("tool", 30), _row("user", 20)]
    kept = LeoApp._budget_history(_Self(), rows)

    assert kept, "something must be kept"
    assert kept[0]["role"] != "tool"


def test_a_short_history_is_untouched() -> None:
    rows = [_row("user", 10), _row("assistant", 10)]
    assert LeoApp._budget_history(_Self(), rows) == rows
