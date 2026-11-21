"""Durable transport bookkeeping: poll cursors and channel heartbeats.

A transport that keeps its position only in memory has to guess after a restart.
Telegram's ``getUpdates`` offset is the position: hold it in memory and a restart
either replays the last day of messages or drops everything that arrived while
the process was down. Two rows fix that and nothing else changes — the cursor
lives in SQLite, so a restart resumes exactly where the last confirmed update
ended.

This is deliberately not the gateway split. It is the part of it that users feel
on a single-process install.
"""

from __future__ import annotations

import os
from typing import Any

from .. import db
from ..contracts import utcnow


class TransportState:
    """Read/write the transport's durable position for a channel account."""

    def __init__(self, channel: str, account: str = "") -> None:
        self.channel = channel
        self.account = account

    async def cursor(self) -> int | None:
        """The last persisted position, or None when nothing was stored yet."""
        row = await db.fetchone(
            "SELECT value FROM transport_cursors WHERE channel=? AND account=?",
            (self.channel, self.account),
        )
        if row is None:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    async def set_cursor(self, value: int) -> None:
        await db.execute(
            "INSERT INTO transport_cursors(channel,account,value,updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(channel,account) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (self.channel, self.account, str(int(value)), utcnow()),
        )

    async def beat(self, note: str = "") -> None:
        """Mark the transport as alive; /gateway status reads these later."""
        await db.execute(
            "INSERT INTO transport_heartbeats(channel,account,pid,note,beat_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(channel,account) DO UPDATE SET "
            "pid=excluded.pid, note=excluded.note, beat_at=excluded.beat_at",
            (self.channel, self.account, os.getpid(), note[:200], utcnow()),
        )

    async def last_beat(self) -> dict[str, Any] | None:
        row = await db.fetchone(
            "SELECT pid, note, beat_at FROM transport_heartbeats WHERE channel=? AND account=?",
            (self.channel, self.account),
        )
        if row is None:
            return None
        return {"pid": row[0], "note": row[1], "beat_at": row[2]}
