"""Durable, auditable memory storage backed by the shared SQLite database.

Only the ``memory_items``, ``memory_links`` and ``memory_versions`` tables
defined in :mod:`leo.db` are used; no extra schema is required.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .. import db
from ..contracts import MemoryCategory, MemoryKind, MemoryRecord, new_id, utcnow
from ..events import EventBus

MEMORY_CHANGED = "memory.changed"

#: Kinds that must never be silently re-labelled as facts.
NON_FACT_KINDS: frozenset[MemoryKind] = frozenset(
    {MemoryKind.ASSUMPTION, MemoryKind.UNVERIFIED, MemoryKind.TEMPORARY}
)

UPDATABLE_FIELDS = frozenset(
    {
        "category",
        "kind",
        "content",
        "metadata",
        "source",
        "confidence",
        "tags",
        "version",
        "expires_at",
        "embedding",
        "supersedes",
    }
)


def enum_value(value: Any) -> str:
    """Accept enum members or plain strings from callers."""
    return value.value if isinstance(value, (MemoryCategory, MemoryKind)) else str(value)


def _clamp_confidence(value: Any) -> float:
    """Coerce a confidence to a finite ``float`` within the documented [0, 1].

    Both :meth:`MemoryStore.add` and :meth:`MemoryStore.update` must agree on
    the stored range; otherwise retrieval (``min_confidence``) and audit events
    see out-of-range values. NaN is mapped to 0.0 instead of poisoning the DB.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number") from exc
    if math.isnan(result):
        return 0.0
    return max(0.0, min(1.0, result))


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_load(value: Any, fallback: Any) -> Any:
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record_from_row(row: Mapping[str, Any]) -> MemoryRecord:
    data = dict(row)
    data["metadata"] = _json_load(data.get("metadata"), {})
    data["tags"] = _json_load(data.get("tags"), [])
    embedding = _json_load(data.get("embedding"), None)
    if embedding is not None:
        data["embedding"] = embedding
    else:
        data.pop("embedding", None)
    supersedes = data.get("supersedes")
    if supersedes is None:
        data.pop("supersedes", None)
    else:
        data["supersedes"] = _json_load(supersedes, supersedes)
    data.pop("score", None)
    return MemoryRecord.model_validate(data)


def _link_specs(record: MemoryRecord) -> list[dict[str, Any]]:
    """Allow callers to declare links as an extra field on a new record."""
    raw = getattr(record, "links", None)
    if raw is None:
        raw = getattr(record, "relations", None)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        raw = [raw]
    specs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            specs.append({"target": item, "relation": "related", "weight": 1.0})
            continue
        if isinstance(item, MemoryRecord):
            item = {"target": item.id}
        if not isinstance(item, Mapping):
            continue
        target = item.get("target") or item.get("to") or item.get("id")
        if not target:
            continue
        specs.append(
            {
                "target": str(target),
                "relation": str(item.get("relation") or "related"),
                "weight": float(item.get("weight", 1.0)),
            }
        )
    return specs


class MemoryStore:
    """CRUD, versioning, links and retention for :class:`MemoryRecord`.

    ``events`` / ``event_bus`` may be supplied to share an EventBus with the
    rest of the runtime; otherwise a private bus is used. Every mutation is
    published on ``memory.changed`` (and therefore persisted in event history).
    """

    def __init__(
        self,
        events: EventBus | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self.bus = events or event_bus or EventBus()
        self.event_bus = self.bus
        self.events = self.bus

    # ------------------------------------------------------------------ CRUD
    async def add(self, record: MemoryRecord) -> MemoryRecord:
        if not isinstance(record, MemoryRecord):
            record = MemoryRecord.model_validate(record)
        record.category = MemoryCategory(enum_value(record.category))
        record.kind = MemoryKind(enum_value(record.kind))
        if not record.id:
            record.id = new_id("mem")
        record.version = max(1, int(record.version))
        record.confidence = _clamp_confidence(record.confidence)
        # Cap stored content so one hostile write cannot become an unbounded
        # durable prompt block in unrelated future conversations.
        record.content = str(record.content)[:4000]
        record.score = 0.0  # scores are assigned by retrieval, never persisted
        record.updated_at = record.updated_at or utcnow()
        record.created_at = record.created_at or record.updated_at

        embedding = getattr(record, "embedding", None)
        embedding_json = _json_dump(embedding) if embedding is not None else None
        supersedes = getattr(record, "supersedes", None)
        supersedes_json = _json_dump(supersedes) if supersedes is not None else None

        async with db.connection() as conn:
            await conn.execute(
                """INSERT INTO memory_items
                   (id, category, kind, content, metadata, source, confidence, tags,
                    version, embedding, created_at, updated_at, expires_at, supersedes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.id,
                    enum_value(record.category),
                    enum_value(record.kind),
                    record.content,
                    _json_dump(record.metadata),
                    record.source,
                    record.confidence,
                    _json_dump(record.tags),
                    record.version,
                    embedding_json,
                    record.created_at,
                    record.updated_at,
                    record.expires_at,
                    supersedes_json,
                ),
            )
        for spec in _link_specs(record):
            await self.link(record.id, spec["target"], spec["relation"], spec["weight"])
        await self.emit_change("created", record)
        return record

    async def get(self, id: str) -> MemoryRecord | None:
        row = await db.fetchone("SELECT * FROM memory_items WHERE id=?", (id,))
        return _record_from_row(row) if row else None

    async def update(self, id: str, **fields: Any) -> MemoryRecord:
        """Update fields, bump the version and append a durable version row.

        A record whose kind is ``assumption``, ``unverified`` or ``temporary``
        can only become a ``fact`` when the caller explicitly passes BOTH
        ``kind=MemoryKind.FACT`` and a ``confidence`` value. Any other attempt
        raises ``ValueError`` and leaves the stored record untouched.
        """
        row = await db.fetchone("SELECT * FROM memory_items WHERE id=?", (id,))
        if row is None:
            raise KeyError(f"memory not found: {id}")
        previous = _record_from_row(row)

        action = str(fields.pop("action", "updated"))
        reason = fields.pop("reason", None) or action
        actor = str(fields.pop("actor", "memory"))
        promotion = False

        target_kind = previous.kind
        if "kind" in fields:
            try:
                target_kind = MemoryKind(enum_value(fields["kind"]))
            except ValueError as exc:
                raise ValueError(f"invalid memory kind: {fields['kind']!r}") from exc
            fields["kind"] = target_kind
        if "category" in fields:
            try:
                fields["category"] = MemoryCategory(enum_value(fields["category"]))
            except ValueError as exc:
                raise ValueError(f"invalid memory category: {fields['category']!r}") from exc

        if "metadata" in fields and not isinstance(fields["metadata"], Mapping):
            raise ValueError("metadata must be a mapping")

        if previous.kind in NON_FACT_KINDS:
            metadata_kind = str((fields.get("metadata") or {}).get("kind", "")).lower()
            if metadata_kind == MemoryKind.FACT.value:
                raise ValueError("cannot promote uncertain memory to fact via metadata")
            if target_kind is MemoryKind.FACT:
                if "kind" not in fields or "confidence" not in fields:
                    raise ValueError(
                        "promoting an assumption/unverified/temporary memory to fact "
                        "requires explicit kind=MemoryKind.FACT and confidence"
                    )
                confidence = fields["confidence"]
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    raise ValueError("confidence must be a number for explicit fact promotion")
                promotion = True

        unknown = set(fields) - UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"unknown memory fields: {sorted(unknown)}")

        changes = {name: fields[name] for name in sorted(fields)}
        new_version = max(previous.version, int(fields.get("version", previous.version))) + 1
        if "version" in fields:
            fields.pop("version")

        values: dict[str, Any] = {
            "category": enum_value(fields.get("category", previous.category)),
            "kind": enum_value(fields.get("kind", previous.kind)),
            "content": fields.get("content", previous.content),
            "metadata": dict(fields.get("metadata", previous.metadata)),
            "source": fields.get("source", previous.source),
            "confidence": _clamp_confidence(fields.get("confidence", previous.confidence)),
            "tags": list(fields.get("tags", previous.tags)),
            "embedding": _json_load(
                fields.get("embedding", getattr(previous, "embedding", None)), None
            ),
            "expires_at": fields.get("expires_at", previous.expires_at),
            # The values here come from callers or from an already-decoded row,
            # never from a raw DB column. Passing them through _json_load()
            # discarded a plain id string (json.loads fails -> fallback None);
            # _json_dump() at write time handles serialization for us.
            "supersedes": fields.get("supersedes", getattr(previous, "supersedes", None)),
        }
        now = utcnow()

        async with db.connection() as conn:
            await conn.execute(
                """UPDATE memory_items SET category=?, kind=?, content=?, metadata=?,
                   source=?, confidence=?, tags=?, version=?, embedding=?, updated_at=?,
                   expires_at=?, supersedes=? WHERE id=?""",
                (
                    values["category"],
                    values["kind"],
                    values["content"],
                    _json_dump(values["metadata"]),
                    values["source"],
                    values["confidence"],
                    _json_dump(values["tags"]),
                    new_version,
                    _json_dump(values["embedding"]) if values["embedding"] is not None else None,
                    now,
                    values["expires_at"],
                    _json_dump(values["supersedes"]) if values["supersedes"] is not None else None,
                    id,
                ),
            )
            version_reason = str(reason)
            if promotion:
                version_reason += "; explicit promotion to fact"
            await conn.execute(
                """INSERT INTO memory_versions
                   (memory_id, version, content, metadata, changed_at, reason)
                   VALUES (?,?,?,?,?,?)""",
                (
                    id,
                    new_version,
                    values["content"],
                    _json_dump(values["metadata"]),
                    now,
                    version_reason,
                ),
            )
        updated = await self.get(id)
        assert updated is not None
        await self.emit_change(
            action,
            updated,
            actor=actor,
            reason=reason,
            changes=changes,
            promotion=promotion,
            previous={
                "content": previous.content,
                "kind": enum_value(previous.kind),
                "category": enum_value(previous.category),
                "confidence": previous.confidence,
                "version": previous.version,
            },
        )
        return updated

    async def delete(self, id: str) -> bool:
        """User-controlled deletion. Removes the record, links and versions."""
        row = await db.fetchone("SELECT * FROM memory_items WHERE id=?", (id,))
        if row is None:
            return False
        record = _record_from_row(row)
        async with db.connection() as conn:
            await conn.execute("DELETE FROM memory_links WHERE source=? OR target=?", (id, id))
            await conn.execute("DELETE FROM memory_versions WHERE memory_id=?", (id,))
            await conn.execute("DELETE FROM memory_items WHERE id=?", (id,))
        await self.emit_change("deleted", record, reason="user-controlled deletion")
        return True

    async def purge_expired(self) -> int:
        """Delete records whose retention window has elapsed. Returns the count."""
        rows = await db.fetchall(
            "SELECT id, expires_at FROM memory_items "
            "WHERE expires_at IS NOT NULL AND expires_at != ''"
        )
        now = datetime.now(timezone.utc)
        expired = [
            row["id"]
            for row in rows
            if (ts := parse_timestamp(row["expires_at"])) is not None and ts <= now
        ]
        for memory_id in expired:
            await self.delete(memory_id)
        return len(expired)

    # ---------------------------------------------------------------- listing
    async def list(
        self,
        category: Any = None,
        kind: Any = None,
        tags: Sequence[str] | None = None,
        limit: int = 100,
        *,
        include_expired: bool = False,
        metadata: Mapping[str, Any] | None = None,
        match_all_tags: bool = False,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memory_items WHERE 1=1"
        params: list[Any] = []
        if category is not None:
            sql += " AND category=?"
            params.append(enum_value(category))
        if kind is not None:
            sql += " AND kind=?"
            params.append(enum_value(kind))
        # NOTE: expiry is deliberately NOT pre-filtered in SQL. expires_at is a
        # free-form ISO string (Z suffix, non-UTC offsets, empty values, ...).
        # A lexicographic `expires_at > utcnow()` comparison silently drops live
        # records whose offset/format sorts before now (and keeps some expired
        # ones). is_expired() below parses the timestamp and is authoritative.
        rows = await db.fetchall(sql, params)

        wanted_tags = {str(t) for t in tags} if tags else set()
        records: list[MemoryRecord] = []
        for row in rows:
            record = _record_from_row(row)
            if not include_expired and is_expired(record):
                continue
            if wanted_tags:
                actual = {str(t) for t in record.tags}
                if match_all_tags:
                    if not wanted_tags <= actual:
                        continue
                elif not (wanted_tags & actual):
                    continue
            if metadata and not _metadata_matches(record.metadata, metadata):
                continue
            records.append(record)
        records.sort(key=lambda r: (r.updated_at, r.id), reverse=True)
        if limit is not None and limit >= 0:
            records = records[: int(limit)]
        return records

    # ------------------------------------------------------------------ links
    async def link(
        self,
        source: str,
        target: str,
        relation: str = "related",
        weight: float = 1.0,
    ) -> dict[str, Any]:
        if source == target:
            raise ValueError("a memory cannot link to itself")
        for memory_id in (source, target):
            if await db.fetchone("SELECT 1 FROM memory_items WHERE id=?", (memory_id,)) is None:
                raise KeyError(f"memory not found: {memory_id}")
        link = {
            "source": source,
            "target": target,
            "relation": relation,
            "weight": float(weight),
        }
        async with db.connection() as conn:
            await conn.execute(
                """INSERT INTO memory_links(source,target,relation,weight) VALUES(?,?,?,?)
                   ON CONFLICT(source,target,relation) DO UPDATE SET weight=excluded.weight""",
                (source, target, relation, link["weight"]),
            )
        await self.emit_change(
            "linked",
            await self.get(source),
            reason="link created",
            link=link,
        )
        return link

    # Convenience aliases for callers that prefer a verbier API.
    add_link = link

    async def unlink(self, source: str, target: str, relation: str | None = None) -> int:
        sql = "DELETE FROM memory_links WHERE source=? AND target=?"
        params: list[Any] = [source, target]
        if relation is not None:
            sql += " AND relation=?"
            params.append(relation)
        async with db.connection() as conn:
            cursor = await conn.execute(sql, params)
            removed = cursor.rowcount
        if removed:
            await self.emit_change(
                "unlinked",
                await self.get(source),
                reason="link removed",
                link={"source": source, "target": target, "relation": relation},
            )
        return int(removed)

    async def get_links(self, id: str) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT source,target,relation,weight FROM memory_links "
            "WHERE source=? OR target=? ORDER BY source,target,relation",
            (id, id),
        )
        return [dict(row) for row in rows]

    # Alias used by some retrieval code.
    links = get_links

    async def versions(self, id: str) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT id,memory_id,version,content,metadata,changed_at,reason "
            "FROM memory_versions WHERE memory_id=? ORDER BY version,id",
            (id,),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _json_load(item.get("metadata"), {})
            result.append(item)
        return result

    # ------------------------------------------------------------------ audit
    async def emit_change(
        self,
        action: str,
        record: MemoryRecord | None,
        *,
        actor: str = "memory",
        reason: str | None = None,
        **extra: Any,
    ) -> None:
        """Publish an auditable ``memory.changed`` event."""
        payload: dict[str, Any] = {
            "action": action,
            "actor": actor,
            "timestamp": utcnow(),
        }
        if record is not None:
            payload.update(
                {
                    "id": record.id,
                    "category": enum_value(record.category),
                    "kind": enum_value(record.kind),
                    "version": record.version,
                    "confidence": record.confidence,
                    "source": record.source,
                    "record": record.model_dump(mode="json"),
                }
            )
        if reason is not None:
            payload["reason"] = reason
        payload.update(extra)
        await self.bus.emit(MEMORY_CHANGED, **payload)


def _metadata_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        stored = actual[key]
        if stored == value:
            continue
        if isinstance(value, (list, tuple, set, frozenset)) and stored in value:
            continue
        if isinstance(stored, (list, tuple, set, frozenset)) and value in stored:
            continue
        return False
    return True


def is_expired(record: MemoryRecord, now: datetime | None = None) -> bool:
    if not record.expires_at:
        return False
    expires = parse_timestamp(record.expires_at)
    if expires is None:
        return False
    return expires <= (now or datetime.now(timezone.utc))


__all__ = [
    "MEMORY_CHANGED",
    "NON_FACT_KINDS",
    "MemoryStore",
    "enum_value",
    "is_expired",
    "parse_timestamp",
]
