"""Durable staging for background skill writes.

An autonomous review may decide to author or patch a skill, but the operator
can require a human look first. A staged write is the exact ``skill_manage``
payload, stored under a ``kv`` key with the SHA-256 of those arguments. It is
inert until ``/skills approve <id>`` replays it through the same tool (with
staging bypassed for that one call), after re-verifying the digest. Rejection
and approval are terminal states; replay is one-shot via an atomic claim.

This mirrors the learning engine's proposal model (``kv`` + audit + explicit
resolution) and does not add a second approval mechanism.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .. import db
from ..contracts import new_id, utcnow
from ..security.permissions import arguments_digest

PREFIX = "skillproposal:"
MAX_PENDING = 50
MAX_OPS = 12
ACTIVE = frozenset({"pending", "applying"})
PLAN_OPS = frozenset({"read", "manage", "archive"})
_BYPASS: ContextVar[bool] = ContextVar("skill_stage_bypass", default=False)


@contextmanager
def bypass_staging() -> Iterator[None]:
    """Allow one already-approved replay to reach the tool without re-staging."""
    token = _BYPASS.set(True)
    try:
        yield
    finally:
        _BYPASS.reset(token)


def staging_bypassed() -> bool:
    return _BYPASS.get()


def _key(proposal_id: str) -> str:
    return f"{PREFIX}{proposal_id}"


def _load_sync(value: Any) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def load(proposal_id: str) -> dict[str, Any] | None:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (_key(proposal_id),))
    if row is None:
        return None
    return _load_sync(row["value"])


async def pending(limit: int = MAX_PENDING) -> list[dict[str, Any]]:
    rows = await db.fetchall(
        "SELECT key, value FROM kv WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
        (f"{PREFIX}%", max(1, limit)),
    )
    items = []
    for row in rows:
        value = _load_sync(row["value"])
        if value is not None and str(value.get("status")) in ACTIVE:
            items.append(value)
    return items


async def _store(record: dict[str, Any]) -> dict[str, Any]:
    """Insert one pending proposal, failing closed when the queue is full."""
    async with db.transaction() as conn:
        cursor = await conn.execute(
            "SELECT value FROM kv WHERE key LIKE ?", (f"{PREFIX}%",)
        )
        active = 0
        for row in await cursor.fetchall():
            value = _load_sync(row["value"])
            if value is not None and str(value.get("status")) in ACTIVE:
                active += 1
        if active >= MAX_PENDING:
            raise ValueError(
                f"{MAX_PENDING} skill writes are already awaiting review; "
                "approve or reject some before staging another"
            )
        await conn.execute(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)",
            (_key(str(record["id"])), json.dumps(record, ensure_ascii=False, default=str), utcnow()),
        )
    return record


async def stage(
    *,
    action: str,
    skill: str,
    payload: dict[str, Any],
    actor: str,
    reason: str = "",
) -> dict[str, Any]:
    """Stage one exact tool payload; fails closed when the queue is full."""
    digest = arguments_digest(payload)
    if not digest:
        raise ValueError("staged skill write cannot be bound to an arguments digest")
    return await _store({
        "id": new_id("sprop"),
        "created_at": utcnow(),
        "status": "pending",
        "kind": "write",
        "action": action,
        "skill": skill,
        "payload": dict(payload),
        "arguments_digest": digest,
        "actor": str(actor or "")[:120],
        "reason": str(reason or "")[:300],
    })


def validate_plan(operations: Any) -> list[dict[str, Any]]:
    """Validate a consolidation plan before it can be staged."""
    if not isinstance(operations, (list, tuple)) or not operations:
        raise ValueError("a consolidation plan needs at least one operation")
    if len(operations) > MAX_OPS:
        raise ValueError(f"a consolidation plan is limited to {MAX_OPS} operations")
    cleaned: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise TypeError(f"operation {index} must be an object")
        op = str(operation.get("op") or "").strip().lower()
        if op not in PLAN_OPS:
            raise ValueError(f"operation {index} has unsupported op {op!r}")
        if op == "read":
            name = str(operation.get("name") or "").strip()
            if not name:
                raise ValueError(f"operation {index} needs a skill name")
            item = {"op": "read", "name": name}
            file_path = operation.get("file_path")
            if isinstance(file_path, str) and file_path.strip():
                item["file_path"] = file_path.strip()
            cleaned.append(item)
        elif op == "archive":
            name = str(operation.get("name") or "").strip()
            if not name:
                raise ValueError(f"operation {index} needs a skill name")
            cleaned.append({"op": "archive", "name": name})
        else:
            arguments = operation.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError(f"operation {index} needs a tool arguments object")
            action = str(arguments.get("action") or "").strip().lower()
            if action not in {"create", "patch", "write_file", "remove_file"}:
                raise ValueError(f"operation {index} has unsupported action {action!r}")
            name = str(arguments.get("name") or "").strip()
            if not name:
                raise ValueError(f"operation {index} needs a skill name")
            cleaned.append({"op": "manage", "arguments": dict(arguments)})
    return cleaned


async def stage_plan(
    *,
    title: str,
    operations: list[dict[str, Any]],
    actor: str,
    reason: str = "",
) -> dict[str, Any]:
    """Stage one consolidated multi-operation plan (never applies it)."""
    cleaned = validate_plan(operations)
    digest = arguments_digest(cleaned)
    if not digest:
        raise ValueError("consolidation plan cannot be bound to an arguments digest")
    return await _store({
        "id": new_id("splan"),
        "created_at": utcnow(),
        "status": "pending",
        "kind": "consolidate",
        "title": str(title or "skill consolidation")[:160],
        "operations": cleaned,
        "arguments_digest": digest,
        "actor": str(actor or "")[:120],
        "reason": str(reason or "")[:300],
    })


async def claim(proposal_id: str, *, by: str) -> dict[str, Any] | None:
    """Atomically move pending -> applying and verify the stored digest."""
    async with db.transaction() as conn:
        cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (_key(proposal_id),))
        row = await cursor.fetchone()
        record = _load_sync(row["value"]) if row is not None else None
        if record is None or str(record.get("status")) != "pending":
            return None
        kind = str(record.get("kind") or "write")
        material = record.get("operations") if kind == "consolidate" else record.get("payload")
        if material is None or arguments_digest(material) != record.get("arguments_digest"):
            record["status"] = "failed"
            record["error"] = "stored payload no longer matches its arguments digest"
            record["resolved_by"] = str(by or "")[:120]
            record["resolved_at"] = utcnow()
            await conn.execute(
                "UPDATE kv SET value=?, updated_at=? WHERE key=?",
                (json.dumps(record, ensure_ascii=False, default=str), utcnow(), _key(proposal_id)),
            )
            return None
        record["status"] = "applying"
        record["claimed_by"] = str(by or "")[:120]
        record["claimed_at"] = utcnow()
        await conn.execute(
            "UPDATE kv SET value=?, updated_at=? WHERE key=?",
            (json.dumps(record, ensure_ascii=False, default=str), utcnow(), _key(proposal_id)),
        )
        return record


async def resolve(
    proposal_id: str,
    status: str,
    *,
    by: str,
    error: str | None = None,
    result: Any = None,
) -> dict[str, Any] | None:
    if status not in {"applied", "rejected", "failed"}:
        raise ValueError(f"unknown staging outcome: {status!r}")
    async with db.transaction() as conn:
        cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (_key(proposal_id),))
        row = await cursor.fetchone()
        record = _load_sync(row["value"]) if row is not None else None
        if record is None:
            return None
        if str(record.get("status")) not in {"pending", "applying"}:
            return record
        record["status"] = status
        record["resolved_by"] = str(by or "")[:120]
        record["resolved_at"] = utcnow()
        if error:
            record["error"] = str(error)[:400]
        if result is not None:
            record["result"] = result
        await conn.execute(
            "UPDATE kv SET value=?, updated_at=? WHERE key=?",
            (json.dumps(record, ensure_ascii=False, default=str), utcnow(), _key(proposal_id)),
        )
        return record


async def history(limit: int = 50) -> list[dict[str, Any]]:
    rows = await db.fetchall(
        "SELECT value FROM kv WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
        (f"{PREFIX}%", max(1, limit)),
    )
    return [item for item in (_load_sync(row["value"]) for row in rows) if item is not None]


def summarize(records: Sequence[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Bounded, transport-safe view for ``/skills pending``."""
    return [
        {
            "id": item.get("id"),
            "status": item.get("status"),
            "kind": item.get("kind") or "write",
            "title": item.get("title"),
            "action": item.get("action"),
            "skill": item.get("skill"),
            "operations": len(item.get("operations") or []) or None,
            "digest": str(item.get("arguments_digest") or "")[:16],
            "created_at": item.get("created_at"),
            "reason": item.get("reason"),
        }
        for item in list(records)[:limit]
    ]


__all__ = [
    "ACTIVE",
    "MAX_OPS",
    "MAX_PENDING",
    "PLAN_OPS",
    "PREFIX",
    "bypass_staging",
    "claim",
    "history",
    "load",
    "pending",
    "resolve",
    "stage",
    "stage_plan",
    "staging_bypassed",
    "summarize",
    "validate_plan",
]
