"""SQLite backing store for the event outbox, subscriptions and deliveries.

The tables are created lazily (like the gateway outbox) so adding them does not
require a change to the shared ``db.py`` schema used by other work packages.
All operations are idempotent and safe to call from multiple EventBus instances.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import db

MAX_ATTEMPTS = 3

OUTBOX_TABLE = "event_outbox"
SUBSCRIPTIONS_TABLE = "event_subscriptions"
DELIVERIES_TABLE = "event_deliveries"

OUTBOX_SCHEMA: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {OUTBOX_TABLE} (
        id TEXT PRIMARY KEY, topic TEXT NOT NULL, source TEXT NOT NULL,
        payload TEXT NOT NULL, correlation_id TEXT, created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT, processed_at TEXT, updated_at TEXT NOT NULL, seq INTEGER)""",
    f"CREATE INDEX IF NOT EXISTS idx_event_outbox_status ON {OUTBOX_TABLE}(status, created_at, id)",
    f"""CREATE TABLE IF NOT EXISTS {SUBSCRIPTIONS_TABLE} (
        name TEXT PRIMARY KEY, topic TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
        cursor_created_at TEXT, cursor_seq INTEGER,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS {DELIVERIES_TABLE} (
        subscription TEXT NOT NULL, event_id TEXT NOT NULL, topic TEXT NOT NULL,
        payload TEXT NOT NULL, seq INTEGER, status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, delivered_at TEXT,
        PRIMARY KEY(subscription, event_id))""",
    f"CREATE INDEX IF NOT EXISTS idx_event_deliveries_status ON {DELIVERIES_TABLE}(status, subscription, created_at)",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventOutbox:
    """Low-level, durable helpers. Failures propagate to the caller."""

    def __init__(self) -> None:
        self._ready = False
        self._path: str | None = None
        self._lock = asyncio.Lock()

    async def ensure_schema(self) -> None:
        """Create the three tables once per database path."""
        path = str(db.get_db_path()) if db.database_path_configured() else None
        if self._ready and self._path == path:
            return
        async with self._lock:
            if self._ready and self._path == path:
                return
            for statement in OUTBOX_SCHEMA:
                await db.execute(statement)
            self._path = path
            self._ready = True

    # -- outbox ---------------------------------------------------------
    async def enqueue_event(self, event_id: str, topic: str, source: str, payload: str,
                            correlation_id: str | None, created_at: str, error: str) -> bool:
        await self.ensure_schema()
        now = utcnow()
        cursor = await db.execute(
            f"""INSERT OR IGNORE INTO {OUTBOX_TABLE}
                (id,topic,source,payload,correlation_id,created_at,status,attempts,last_error,updated_at,seq)
                VALUES(?,?,?,?,?,?,'pending',0,?,?,0)""",
            (event_id, topic, source, payload, correlation_id, created_at, error, now),
        )
        return bool(getattr(cursor, "rowcount", 0))

    async def claim_events(self, limit: int) -> list[Any]:
        await self.ensure_schema()
        return await db.fetchall(
            f"SELECT * FROM {OUTBOX_TABLE} WHERE status IN ('pending','failed') "
            "ORDER BY created_at, id LIMIT ?",
            (limit,),
        )

    async def mark_event(self, event_id: str, status: str, *, error: str | None = None,
                         attempts: int | None = None, seq: int | None = None) -> None:
        await self.ensure_schema()
        assignments = ["status=?", "updated_at=?"]
        params: list[Any] = [status, utcnow()]
        if error is not None:
            assignments.append("last_error=?")
            params.append(error)
        if attempts is not None:
            assignments.append("attempts=?")
            params.append(attempts)
        if seq is not None:
            assignments.append("seq=?")
            params.append(seq)
        if status == "processed":
            assignments.append("processed_at=?")
            params.append(utcnow())
        params.append(event_id)
        await db.execute(f"UPDATE {OUTBOX_TABLE} SET " + ",".join(assignments) + " WHERE id=?", params)

    # -- subscriptions --------------------------------------------------
    async def register_subscription(self, name: str, topic: str) -> None:
        await self.ensure_schema()
        now = utcnow()
        await db.execute(
            f"""INSERT INTO {SUBSCRIPTIONS_TABLE}(name,topic,enabled,cursor_created_at,cursor_seq,created_at,updated_at)
                VALUES(?,?,1,NULL,NULL,?,?)
                ON CONFLICT(name) DO UPDATE SET topic=excluded.topic, enabled=1, updated_at=excluded.updated_at""",
            (name, topic, now, now),
        )

    async def list_subscriptions(self, *, enabled_only: bool = False) -> list[Any]:
        await self.ensure_schema()
        sql = f"SELECT * FROM {SUBSCRIPTIONS_TABLE}"
        if enabled_only:
            sql += " WHERE enabled=1"
        return await db.fetchall(sql + " ORDER BY name")

    async def get_subscription(self, name: str) -> Any:
        await self.ensure_schema()
        return await db.fetchone(f"SELECT * FROM {SUBSCRIPTIONS_TABLE} WHERE name=?", (name,))

    async def set_subscription_enabled(self, name: str, enabled: bool) -> None:
        await self.ensure_schema()
        await db.execute(
            f"UPDATE {SUBSCRIPTIONS_TABLE} SET enabled=?, updated_at=? WHERE name=?",
            (int(bool(enabled)), utcnow(), name),
        )

    async def delete_subscription(self, name: str) -> None:
        await self.ensure_schema()
        await db.execute(f"DELETE FROM {SUBSCRIPTIONS_TABLE} WHERE name=?", (name,))
        await db.execute(f"DELETE FROM {DELIVERIES_TABLE} WHERE subscription=?", (name,))

    async def advance_cursor(self, name: str, created_at: str, seq: int) -> None:
        await self.ensure_schema()
        await db.execute(
            f"""UPDATE {SUBSCRIPTIONS_TABLE}
                SET cursor_created_at=?, cursor_seq=?, updated_at=?
                WHERE name=? AND (cursor_seq IS NULL OR cursor_seq < ?)""",
            (created_at, int(seq), utcnow(), name, int(seq)),
        )

    # -- deliveries -----------------------------------------------------
    async def enqueue_delivery(self, subscription: str, event_id: str, topic: str,
                               payload: str, seq: int = 0, created_at: str | None = None) -> bool:
        await self.ensure_schema()
        now = created_at or utcnow()
        cursor = await db.execute(
            f"""INSERT OR IGNORE INTO {DELIVERIES_TABLE}
                (subscription,event_id,topic,payload,seq,status,attempts,created_at,updated_at)
                VALUES(?,?,?,?,?,'pending',0,?,?)""",
            (subscription, event_id, topic, payload, int(seq), now, now),
        )
        return bool(getattr(cursor, "rowcount", 0))

    async def claim_delivery(self, subscription: str, event_id: str,
                             max_attempts: int = MAX_ATTEMPTS) -> bool:
        await self.ensure_schema()
        cursor = await db.execute(
            f"UPDATE {DELIVERIES_TABLE} SET status='processing', updated_at=? "
            "WHERE subscription=? AND event_id=? AND status IN ('pending','failed') "
            "AND attempts < ?",
            (utcnow(), subscription, event_id, max_attempts),
        )
        return bool(getattr(cursor, "rowcount", 0))

    async def mark_delivery(self, subscription: str, event_id: str, *, status: str,
                            attempts: int, error: str | None = None) -> None:
        await self.ensure_schema()
        now = utcnow()
        delivered_at = now if status == "delivered" else None
        await db.execute(
            f"""UPDATE {DELIVERIES_TABLE}
                SET status=?, attempts=?, last_error=?, updated_at=?, delivered_at=COALESCE(?,delivered_at)
                WHERE subscription=? AND event_id=?""",
            (status, int(attempts), error, now, delivered_at, subscription, event_id),
        )

    async def pending_deliveries(self, subscription: str | None, limit: int) -> list[Any]:
        await self.ensure_schema()
        sql = f"SELECT * FROM {DELIVERIES_TABLE} WHERE status='pending'"
        params: list[Any] = []
        if subscription is not None:
            sql += " AND subscription=?"
            params.append(subscription)
        sql += " ORDER BY created_at, subscription, event_id LIMIT ?"
        params.append(limit)
        return await db.fetchall(sql, params)

    async def delivered_without_seq(self, event_id: str, seq: int) -> list[Any]:
        await self.ensure_schema()
        return await db.fetchall(
            f"SELECT subscription FROM {DELIVERIES_TABLE} WHERE event_id=? AND status='delivered' AND seq=0",
            (event_id,),
        )

    async def update_delivery_seq(self, event_id: str, seq: int) -> None:
        await self.ensure_schema()
        await db.execute(
            f"UPDATE {DELIVERIES_TABLE} SET seq=? WHERE event_id=? AND seq=0", (seq, event_id)
        )

    async def stats(self) -> dict[str, int]:
        await self.ensure_schema()
        result: dict[str, int] = {}
        statuses = {
            OUTBOX_TABLE: ("pending", "processing", "processed", "failed"),
            DELIVERIES_TABLE: ("pending", "processing", "delivered", "failed"),
        }
        for table, known in statuses.items():
            for status in known:
                result[f"{table}.{status}"] = 0
            rows = await db.fetchall(f"SELECT status, COUNT(*) AS n FROM {table} GROUP BY status")
            for row in rows:
                result[f"{table}.{row['status']}"] = int(row["n"])
        total_row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {SUBSCRIPTIONS_TABLE}")
        result["event_subscriptions.total"] = int(total_row["n"]) if total_row else 0
        return result


def matches_topic(pattern: str, topic: str) -> bool:
    """Exact or ``prefix.*`` topic match (same rule as the in-process bus)."""
    if pattern == topic:
        return True
    return pattern.endswith(".*") and topic.startswith(pattern[:-1])


def normalize_subscriptions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "topic": row["topic"],
            "enabled": bool(row["enabled"]),
            "cursor_created_at": row["cursor_created_at"],
            "cursor_seq": int(row["cursor_seq"]) if row["cursor_seq"] is not None else None,
        }
        for row in rows
    ]
