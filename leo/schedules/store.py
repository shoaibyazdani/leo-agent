"""SQLite access for schedules, proposals, tombstones and schedule runtime kv.

This is a *reader* of the one existing ``schedules`` table plus persistence for
proposals/tombstones/runtime state. It never creates a second schedule store or
a second execution cursor; ``Scheduler`` remains the tick authority.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .. import db
from .clock import Clock, SystemClock
from .domain import (
    ScheduleProposal,
    ScheduleSpec,
    ScheduleView,
    canonical_json,
    new_schedule_id,
)

_RUNTIME_PREFIX = "schedule.runtime"


def _load_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class ScheduleStore:
    """The SQL seam for the schedule domain; one table, no parallel authority."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or SystemClock()

    def _now_iso(self) -> str:
        return self.clock.now().isoformat()

    # ------------------------------------------------------------------
    # Schedule reads (the same durable rows Scheduler executes)
    # ------------------------------------------------------------------
    async def get(self, schedule_id: str) -> ScheduleView | None:
        row = await db.fetchone("SELECT * FROM schedules WHERE id=?", (schedule_id,))
        return ScheduleView.from_row(row) if row is not None else None

    async def get_by_name(
        self, name: str, *, active_only: bool = False
    ) -> ScheduleView | None:
        sql = "SELECT * FROM schedules WHERE name=?"
        if active_only:
            sql += " AND status='active'"
        sql += " ORDER BY created_at LIMIT 1"
        row = await db.fetchone(sql, (name,))
        return ScheduleView.from_row(row) if row is not None else None

    async def list(
        self, *, enabled_only: bool = False, include_paused: bool = True
    ) -> list[ScheduleView]:
        sql = "SELECT * FROM schedules"
        clauses: list[str] = []
        if enabled_only:
            clauses.append("enabled=1")
        if not include_paused:
            clauses.append("status='active'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        rows = await db.fetchall(sql)
        return [ScheduleView.from_row(row) for row in rows]

    async def active_names(self) -> list[str]:
        rows = await db.fetchall(
            "SELECT name FROM schedules WHERE status='active' ORDER BY name"
        )
        return [str(row["name"]) for row in rows]

    async def name_is_active(self, name: str) -> bool:
        row = await db.fetchone(
            "SELECT 1 AS present FROM schedules WHERE name=? AND status='active'",
            (name,),
        )
        return row is not None

    # ------------------------------------------------------------------
    # Proposals (durable, non-executing reservation)
    # ------------------------------------------------------------------
    async def create_proposal(self, proposal: ScheduleProposal) -> ScheduleProposal:
        now = self._now_iso()
        created_at = proposal.created_at or now
        updated_at = proposal.updated_at or now
        digest = proposal.digest or proposal.spec.digest()
        stored = proposal.model_copy(
            update={"digest": digest, "created_at": created_at, "updated_at": updated_at}
        )
        await db.execute(
            "INSERT INTO schedule_proposals("
            "id,active_name,principal,operation,request_key,status,digest,spec,origin,"
            "conversation_id,run_id,tool_call_id,config_revision,schedule_id,error,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                stored.id,
                stored.active_name,
                stored.principal,
                stored.operation,
                stored.request_key,
                stored.status,
                stored.digest,
                canonical_json(stored.spec),
                stored.origin,
                stored.conversation_id,
                stored.run_id,
                stored.tool_call_id,
                stored.config_revision,
                stored.schedule_id,
                stored.error,
                stored.created_at,
                stored.updated_at,
            ),
        )
        return stored

    async def get_proposal(self, proposal_id: str) -> ScheduleProposal | None:
        row = await db.fetchone(
            "SELECT * FROM schedule_proposals WHERE id=?", (proposal_id,)
        )
        return self._proposal_from_row(row) if row is not None else None

    async def find_proposal_by_request(
        self, principal: str, operation: str, request_key: str
    ) -> ScheduleProposal | None:
        row = await db.fetchone(
            "SELECT * FROM schedule_proposals "
            "WHERE principal=? AND operation=? AND request_key=?",
            (principal, operation, request_key),
        )
        return self._proposal_from_row(row) if row is not None else None

    async def list_proposals(self, status: str | None = None) -> list[ScheduleProposal]:
        if status is None:
            rows = await db.fetchall(
                "SELECT * FROM schedule_proposals ORDER BY created_at"
            )
        else:
            rows = await db.fetchall(
                "SELECT * FROM schedule_proposals WHERE status=? ORDER BY created_at",
                (status,),
            )
        return [self._proposal_from_row(row) for row in rows]

    async def update_proposal(
        self,
        proposal_id: str,
        *,
        status: str | None = None,
        active_name: str | None = None,
        schedule_id: str | None = None,
        error: str | None = None,
        config_revision: int | None = None,
    ) -> ScheduleProposal | None:
        fields: list[str] = ["updated_at=?"]
        params: list[Any] = [self._now_iso()]
        if status is not None:
            fields.append("status=?")
            params.append(status)
        if active_name is not None:
            fields.append("active_name=?")
            params.append(active_name)
        if schedule_id is not None:
            fields.append("schedule_id=?")
            params.append(schedule_id)
        if error is not None:
            fields.append("error=?")
            params.append(error)
        if config_revision is not None:
            fields.append("config_revision=?")
            params.append(config_revision)
        params.append(proposal_id)
        await db.execute(
            f"UPDATE schedule_proposals SET {', '.join(fields)} WHERE id=?", params
        )
        return await self.get_proposal(proposal_id)

    @staticmethod
    def _proposal_from_row(row: Any) -> ScheduleProposal:
        data = dict(row)
        return ScheduleProposal(
            id=str(data["id"]),
            active_name=data.get("active_name"),
            principal=str(data["principal"]),
            operation=str(data["operation"]),
            request_key=data.get("request_key"),
            status=str(data.get("status") or "pending"),
            digest=str(data.get("digest") or ""),
            spec=ScheduleSpec.model_validate(_load_dict(data.get("spec"))),
            origin=data.get("origin"),
            conversation_id=data.get("conversation_id"),
            run_id=data.get("run_id"),
            tool_call_id=data.get("tool_call_id"),
            config_revision=data.get("config_revision"),
            schedule_id=data.get("schedule_id"),
            error=data.get("error"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    # ------------------------------------------------------------------
    # Tombstones (inspectable removal/retirement record)
    # ------------------------------------------------------------------
    async def add_tombstone(
        self,
        schedule_id: str,
        name: str,
        reason: str,
        *,
        payload: Mapping[str, Any] | None = None,
        origin: str | None = None,
        created_by: str | None = None,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = self._now_iso()
        tombstone_id = new_schedule_id("tomb")
        await db.execute(
            "INSERT INTO schedule_tombstones("
            "id,schedule_id,name,reason,payload,origin,created_by,created_at,"
            "removed_at,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                tombstone_id,
                schedule_id,
                name,
                reason,
                canonical_json(dict(payload or {})),
                origin,
                created_by,
                created_at or now,
                now,
                canonical_json(dict(metadata or {})),
            ),
        )
        return {
            "id": tombstone_id,
            "schedule_id": schedule_id,
            "name": name,
            "reason": reason,
            "removed_at": now,
        }

    async def get_tombstone(self, schedule_id: str) -> dict[str, Any] | None:
        """Return the terminal record for one immutable schedule ID.

        Runtime lookups are ID-only on purpose: a tombstone is history, not a
        name reservation, so it can never resolve a *name* back into a schedule
        or block explicit active-name reuse.
        """
        row = await db.fetchone(
            "SELECT * FROM schedule_tombstones WHERE schedule_id=? "
            "ORDER BY removed_at DESC LIMIT 1",
            (schedule_id,),
        )
        return dict(row) if row is not None else None

    async def list_tombstones(self, name: str | None = None) -> list[dict[str, Any]]:
        if name is None:
            rows = await db.fetchall(
                "SELECT * FROM schedule_tombstones ORDER BY removed_at DESC"
            )
        else:
            rows = await db.fetchall(
                "SELECT * FROM schedule_tombstones WHERE name=? ORDER BY removed_at DESC",
                (name,),
            )
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Schedule runtime kv (monitor baselines and other bounded state)
    # ------------------------------------------------------------------
    @staticmethod
    def _runtime_key(schedule_id: str, key: str) -> str:
        return f"{_RUNTIME_PREFIX}:{schedule_id}:{key}"

    async def runtime_get(
        self, schedule_id: str, key: str, default: str | None = None
    ) -> str | None:
        row = await db.fetchone(
            "SELECT value FROM kv WHERE key=?", (self._runtime_key(schedule_id, key),)
        )
        return str(row["value"]) if row is not None else default

    async def runtime_set(self, schedule_id: str, key: str, value: str) -> None:
        await db.execute(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (self._runtime_key(schedule_id, key), str(value), self._now_iso()),
        )

    async def runtime_delete(self, schedule_id: str, key: str) -> None:
        await db.execute(
            "DELETE FROM kv WHERE key=?", (self._runtime_key(schedule_id, key),)
        )


__all__ = ["ScheduleStore"]
