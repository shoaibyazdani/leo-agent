"""Bounded before-state snapshots for authored skills.

Every mutating skill write records the content it is about to replace: a full
skill package (manifest + supporting files) before a version replace/delete,
or one file before a file write/removal. Snapshots live in ``kv`` under
``skillsnap:<id>`` and are pruned per skill. A rollback restores exactly that
before-state through the registry's normal write path, so validation and audit
still apply; nothing here writes SQLite rows for skills directly.

Snapshots are deliberately bounded: a package whose serialized form exceeds
``MAX_SNAPSHOT_BYTES`` is not stored, and the mutation proceeds. The operator
gets a truthful "not snapshotted" outcome rather than an unbounded ledger.
"""

from __future__ import annotations

import json
from typing import Any

from .. import db
from ..contracts import new_id, utcnow

PREFIX = "skillsnap:"
MAX_PER_SKILL = 10
MAX_SNAPSHOT_BYTES = 768 * 1024


def _key(snapshot_id: str) -> str:
    return f"{PREFIX}{snapshot_id}"


def _parse(value: Any) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def save(skill_name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Store one before-state; returns ``None`` when it exceeds the bound."""
    record = {
        **payload,
        "id": new_id("snap"),
        "skill_name": str(skill_name),
        "created_at": utcnow(),
    }
    encoded = json.dumps(record, ensure_ascii=False, default=str)
    if len(encoded.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
        return None
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)",
        (_key(record["id"]), encoded, utcnow()),
    )
    await _prune(skill_name)
    return record


async def load(snapshot_id: str) -> dict[str, Any] | None:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (_key(snapshot_id),))
    return _parse(row["value"]) if row is not None else None


async def list_for(skill_name: str, limit: int = MAX_PER_SKILL) -> list[dict[str, Any]]:
    # Snapshot rows are bounded per skill, so filtering the whole prefix is
    # cheaper (and correct) than a global LIMIT that could hide an older
    # skill's snapshots behind a busier one.
    rows = await db.fetchall(
        "SELECT value, updated_at FROM kv WHERE key LIKE ? ORDER BY updated_at DESC",
        (f"{PREFIX}%",),
    )
    items = []
    for row in rows:
        value = _parse(row["value"])
        if value is not None and value.get("skill_name") == skill_name:
            items.append(value)
        if len(items) >= limit:
            break
    return items


async def _prune(skill_name: str) -> None:
    rows = await db.fetchall(
        "SELECT key, value, updated_at FROM kv WHERE key LIKE ? ORDER BY updated_at DESC",
        (f"{PREFIX}%",),
    )
    matching = [
        row for row in rows
        if (_parse(row["value"]) or {}).get("skill_name") == skill_name
    ]
    for row in matching[MAX_PER_SKILL:]:
        await db.execute("DELETE FROM kv WHERE key=?", (row["key"],))


async def discard(skill_name: str) -> None:
    """Drop every snapshot for a skill (used only by an explicit hard purge)."""
    rows = await db.fetchall(
        "SELECT key, value FROM kv WHERE key LIKE ?", (f"{PREFIX}%",)
    )
    for row in rows:
        if (_parse(row["value"]) or {}).get("skill_name") == skill_name:
            await db.execute("DELETE FROM kv WHERE key=?", (row["key"],))


__all__ = [
    "MAX_PER_SKILL",
    "MAX_SNAPSHOT_BYTES",
    "PREFIX",
    "discard",
    "list_for",
    "load",
    "save",
]
