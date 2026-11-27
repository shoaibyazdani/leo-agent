"""Durable, small watcher state in the existing key-value store.

The state is a cursor plus pacing counters, never a to-do list the watcher
promises to finish: specific promises stay scheduled jobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .. import db

STATE_KEY = "watcher:state"
_MAX_SEEN = 500
_MAX_PENDING = 25
_MAX_GITHUB = 200


@dataclass
class WatcherState:
    last_run_at: str | None = None
    seen: list[str] = field(default_factory=list)
    day: str = ""
    ordinary_sent: int = 0
    pending: list[dict[str, Any]] = field(default_factory=list)
    github: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "WatcherState":
        if not isinstance(data, dict):
            return cls()
        raw_github = data.get("github")
        github: dict[str, dict[str, Any]] = {}
        if isinstance(raw_github, dict):
            for key, value in list(raw_github.items())[-_MAX_GITHUB:]:
                if isinstance(value, dict):
                    github[str(key)] = dict(value)
        return cls(
            last_run_at=str(data.get("last_run_at")) if data.get("last_run_at") else None,
            seen=[str(item) for item in (data.get("seen") or []) if str(item)],
            day=str(data.get("day") or ""),
            ordinary_sent=max(0, int(data.get("ordinary_sent") or 0)),
            pending=[item for item in (data.get("pending") or []) if isinstance(item, dict)][:_MAX_PENDING],
            github=github,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_run_at": self.last_run_at,
            "seen": self.seen[-_MAX_SEEN:],
            "day": self.day,
            "ordinary_sent": self.ordinary_sent,
            "pending": self.pending[-_MAX_PENDING:],
            "github": dict(list(self.github.items())[-_MAX_GITHUB:]),
        }

    def mark_seen(self, signal_ids: list[str]) -> None:
        for signal_id in signal_ids:
            if signal_id not in self.seen:
                self.seen.append(signal_id)
        self.seen = self.seen[-_MAX_SEEN:]


async def load_state() -> WatcherState:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (STATE_KEY,))
    if row is None:
        return WatcherState()
    try:
        return WatcherState.from_dict(json.loads(str(row["value"])))
    except (TypeError, ValueError):
        return WatcherState()


async def save_state(state: WatcherState) -> None:
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (
            STATE_KEY,
            json.dumps(state.to_dict(), ensure_ascii=False, default=str),
            datetime.now(UTC).isoformat(),
        ),
    )
