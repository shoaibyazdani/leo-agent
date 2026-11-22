"""Durable queue, admissions and turn leases for the gateway split.

Why these tables exist: with transport and brain in separate processes, the
queue between them is the only thing holding user messages, so it has to survive
both processes dying. Everything here is at-least-once: a claimed turn that is
never completed expires and becomes claimable again, and a completed turn is
terminal.

Invariants:

* one admission per (channel, payload digest) — a replayed update is dropped, so
  a restart can never double-answer;
* a claim is a lease with an expiry — a dead worker's turn comes back;
* attempts are counted, so a poison payload fails loudly instead of looping;
* routing pins a (channel, subject) to one conversation, so the brain keeps the
  thread across restarts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import db
from ..contracts import new_id

ADMISSION_QUEUED = "queued"
ADMISSION_CLAIMED = "claimed"
ADMISSION_DONE = "done"
ADMISSION_FAILED = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    return when.isoformat()


def payload_digest(payload: dict[str, Any]) -> str:
    """Stable digest of an inbound payload, used for at-least-once dedupe."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class GatewayStore:
    """The queue between transport and brain. One class, one owner."""

    # ---------------------------------------------------------------- admission
    async def admit(
        self,
        *,
        channel: str,
        subject: str,
        payload: dict[str, Any],
        principal: str = "",
        request_id: str = "",
    ) -> tuple[str, bool]:
        """Record an inbound payload. Returns (admission_id, created)."""
        digest = payload_digest(payload)
        existing = await db.fetchone(
            "SELECT admission_id FROM gateway_admissions WHERE channel=? AND payload_digest=?",
            (channel, digest),
        )
        if existing is not None:
            return str(existing[0]), False

        admission_id = new_id("adm")
        now = _iso(_now())
        encoded = json.dumps(payload, sort_keys=True, default=str)
        await db.execute(
            "INSERT INTO gateway_admissions(admission_id,request_id,channel,subject,principal,"
            "payload,payload_digest,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (admission_id, request_id, channel, subject, principal, encoded, digest,
             ADMISSION_QUEUED, now, now),
        )
        await db.execute(
            "INSERT INTO gateway_queue(admission_id,status,attempts,created_at,updated_at) "
            "VALUES(?,?,0,?,?)",
            (admission_id, ADMISSION_QUEUED, now, now),
        )
        return admission_id, True

    async def admission(self, admission_id: str) -> dict[str, Any] | None:
        row = await db.fetchone(
            "SELECT admission_id,channel,subject,principal,payload,status FROM gateway_admissions "
            "WHERE admission_id=?",
            (admission_id,),
        )
        if row is None:
            return None
        return {
            "admission_id": row[0], "channel": row[1], "subject": row[2],
            "principal": row[3], "payload": json.loads(row[4] or "{}"), "status": row[5],
        }

    async def admission_by_status(self, status: str) -> dict[str, Any] | None:
        """The oldest admission in a given state (handy for tests and ops)."""
        row = await db.fetchone(
            "SELECT admission_id FROM gateway_admissions WHERE status=? "
            "ORDER BY created_at ASC LIMIT 1",
            (status,),
        )
        if row is None:
            return None
        return await self.admission(str(row[0]))

    # --------------------------------------------------------------------- queue
    async def claim(self, worker: str, lease_seconds: int = 30) -> dict[str, Any] | None:
        """Lease the oldest claimable admission, or None when the queue is empty."""
        now = _now()
        async with db.transaction() as conn:
            cursor = await conn.execute(
                "SELECT q.admission_id FROM gateway_queue q "
                "WHERE q.status=? OR (q.status=? AND (q.claim_expires_at IS NULL "
                "OR q.claim_expires_at < ?)) "
                "ORDER BY q.created_at ASC LIMIT 1",
                (ADMISSION_QUEUED, ADMISSION_CLAIMED, _iso(now)),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            admission_id = str(row[0])
            await conn.execute(
                "UPDATE gateway_queue SET status=?, attempts=attempts+1, claimed_by=?, "
                "claimed_at=?, claim_expires_at=?, updated_at=? WHERE admission_id=?",
                (ADMISSION_CLAIMED, worker, _iso(now),
                 _iso(now + timedelta(seconds=lease_seconds)), _iso(now), admission_id),
            )
            await conn.execute(
                "UPDATE gateway_admissions SET status=?, updated_at=? WHERE admission_id=?",
                (ADMISSION_CLAIMED, _iso(now), admission_id),
            )
        record = await self.admission(admission_id)
        if record is not None:
            record["claimed_by"] = worker
            record["lease_expires_at"] = _iso(now + timedelta(seconds=lease_seconds))
        return record

    async def renew(self, admission_id: str, worker: str, lease_seconds: int = 30) -> bool:
        """Extend a live lease. False when the caller no longer holds it.

        A brain that is still working must be able to say so: without renewal the
        only signal a lease gives is the passage of time, so a slow-but-healthy
        turn competes with a dead one. Renewal separates "alive" from "expired".
        """
        now = _now()
        cursor = await db.execute(
            "UPDATE gateway_queue SET claim_expires_at=?, updated_at=? "
            "WHERE admission_id=? AND status=? AND claimed_by=? "
            "AND claim_expires_at IS NOT NULL AND claim_expires_at > ?",
            (_iso(now + timedelta(seconds=lease_seconds)), _iso(now), admission_id,
             ADMISSION_CLAIMED, worker, _iso(now)),
        )
        return bool(getattr(cursor, "rowcount", 0))

    async def complete(self, admission_id: str) -> None:
        now = _iso(_now())
        await db.execute(
            "UPDATE gateway_queue SET status=?, claim_expires_at=NULL, updated_at=? "
            "WHERE admission_id=?",
            (ADMISSION_DONE, now, admission_id),
        )
        await db.execute(
            "UPDATE gateway_admissions SET status=?, updated_at=? WHERE admission_id=?",
            (ADMISSION_DONE, now, admission_id),
        )

    async def fail(self, admission_id: str, error: str, *, retry: bool = True,
                   max_attempts: int = 5) -> str:
        """Give a turn back (retry) or bury it. Returns the resulting status."""
        now = _iso(_now())
        row = await db.fetchone(
            "SELECT attempts FROM gateway_queue WHERE admission_id=?", (admission_id,)
        )
        attempts = int(row[0]) if row else 0
        status = ADMISSION_QUEUED if retry and attempts < max_attempts else ADMISSION_FAILED
        await db.execute(
            "UPDATE gateway_queue SET status=?, claim_expires_at=NULL, claimed_by=NULL, "
            "last_error=?, updated_at=? WHERE admission_id=?",
            (status, error[:500], now, admission_id),
        )
        await db.execute(
            "UPDATE gateway_admissions SET status=?, updated_at=? WHERE admission_id=?",
            (status, now, admission_id),
        )
        return status

    async def reclaim_expired(self) -> int:
        """Return expired claims to the queue. Safe to call on a timer."""
        now = _iso(_now())
        cursor = await db.execute(
            "UPDATE gateway_queue SET status=?, claim_expires_at=NULL, claimed_by=NULL, "
            "updated_at=? WHERE status=? AND claim_expires_at IS NOT NULL AND claim_expires_at < ?",
            (ADMISSION_QUEUED, now, ADMISSION_CLAIMED, now),
        )
        return int(getattr(cursor, "rowcount", 0) or 0)

    async def stats(self) -> dict[str, Any]:
        queue = {
            row[0]: row[1]
            for row in await db.fetchall(
                "SELECT status, COUNT(*) FROM gateway_queue GROUP BY status"
            )
        }
        cursors = {
            row[0]: row[1]
            for row in await db.fetchall("SELECT channel, value FROM transport_cursors")
        }
        beats = {
            row[0]: {"pid": row[1], "beat_at": row[2]}
            for row in await db.fetchall(
                "SELECT channel, pid, beat_at FROM transport_heartbeats"
            )
        }
        return {"queue": queue, "cursors": cursors, "heartbeats": beats}

    # ------------------------------------------------------------------- routing
    async def route(self, channel: str, subject: str) -> str | None:
        row = await db.fetchone(
            "SELECT conversation_id FROM gateway_routes WHERE channel=? AND subject=?",
            (channel, subject),
        )
        return str(row[0]) if row else None

    async def save_route(self, channel: str, subject: str, conversation_id: str) -> None:
        now = _iso(_now())
        await db.execute(
            "INSERT INTO gateway_routes(channel,subject,conversation_id,updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(channel,subject) DO UPDATE SET "
            "conversation_id=excluded.conversation_id, updated_at=excluded.updated_at",
            (channel, subject, conversation_id, now),
        )

    # --------------------------------------------------------------- turn leases
    async def acquire_turn(self, conversation_id: str, holder: str,
                           ttl_seconds: int = 600) -> bool:
        """Take the conversation's turn lease; False when someone else holds it."""
        now = _now()
        async with db.transaction() as conn:
            cursor = await conn.execute(
                "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id=?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row is not None and row[1] and str(row[1]) > _iso(now):
                if str(row[0]) != holder:
                    return False
            await conn.execute(
                "INSERT INTO session_turn_leases(conversation_id,holder,acquired_at,expires_at) "
                "VALUES(?,?,?,?) ON CONFLICT(conversation_id) DO UPDATE SET "
                "holder=excluded.holder, acquired_at=excluded.acquired_at, "
                "expires_at=excluded.expires_at",
                (conversation_id, holder, _iso(now), _iso(now + timedelta(seconds=ttl_seconds))),
            )
        return True

    async def release_turn(self, conversation_id: str, holder: str) -> None:
        await db.execute(
            "DELETE FROM session_turn_leases WHERE conversation_id=? AND holder=?",
            (conversation_id, holder),
        )

    # ------------------------------------------------------------------- tickets
    async def issue_ticket(self, ttl_seconds: int = 300) -> str:
        ticket = new_id("tkt") + hashlib.sha256(new_id("salt").encode()).hexdigest()[:16]
        now = _now()
        await db.execute(
            "INSERT INTO gateway_attach_tickets(ticket,created_at,expires_at) VALUES(?,?,?)",
            (ticket, _iso(now), _iso(now + timedelta(seconds=ttl_seconds))),
        )
        return ticket

    async def redeem_ticket(self, ticket: str) -> bool:
        """Single use: a ticket that was already redeemed is not valid again."""
        now = _iso(_now())
        async with db.transaction() as conn:
            cursor = await conn.execute(
                "SELECT used_at, expires_at FROM gateway_attach_tickets WHERE ticket=?",
                (ticket,),
            )
            row = await cursor.fetchone()
            if row is None or row[0] or (row[1] and str(row[1]) < now):
                return False
            await conn.execute(
                "UPDATE gateway_attach_tickets SET used_at=? WHERE ticket=?", (now, ticket)
            )
        return True
