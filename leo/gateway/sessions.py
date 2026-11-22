"""Attach tickets and worker sessions: how a brain process gets in.

A worker must not hold the gateway's master token. The gateway mints a
single-use ticket, the worker exchanges it for a short-lived session token, and
only that token rides on claim/complete/fail. A leaked worker credential can
therefore be revoked (or simply expire) without rotating the master.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import db
from ..contracts import new_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SessionStore:
    """Tickets and sessions. Tokens are stored as digests, never in the clear."""

    async def issue_ticket(self, ttl_seconds: int = 300) -> tuple[str, str]:
        """Mint a single-use attach ticket. Returns (ticket, expires_at)."""
        ticket = secrets.token_urlsafe(32)
        now = _now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        await db.execute(
            "INSERT INTO gateway_attach_tickets(ticket,created_at,expires_at) VALUES(?,?,?)",
            (_hash(ticket), now.isoformat(), expires_at.isoformat()),
        )
        return ticket, expires_at.isoformat()

    async def redeem_ticket(self, ticket: str, ttl_seconds: int = 86400) -> str | None:
        """Exchange a ticket for a session token. A ticket works once."""
        if not ticket:
            return None
        now = _now()
        digest = _hash(ticket)
        async with db.transaction() as conn:
            cursor = await conn.execute(
                "SELECT used_at, expires_at FROM gateway_attach_tickets WHERE ticket=?",
                (digest,),
            )
            row = await cursor.fetchone()
            if row is None or row[0]:
                return None
            if row[1] and str(row[1]) < now.isoformat():
                return None
            await conn.execute(
                "UPDATE gateway_attach_tickets SET used_at=? WHERE ticket=?",
                (now.isoformat(), digest),
            )
            session_token = secrets.token_urlsafe(32)
            await conn.execute(
                "INSERT INTO gateway_sessions(session_id,token_hash,created_at,expires_at) "
                "VALUES(?,?,?,?)",
                (new_id("sess"), _hash(session_token), now.isoformat(),
                 (now + timedelta(seconds=ttl_seconds)).isoformat()),
            )
        return session_token

    async def session_valid(self, token: str) -> bool:
        """True when a session token is live. Touches last_seen_at."""
        if not token:
            return False
        now = _now()
        digest = _hash(token)
        row = await db.fetchone(
            "SELECT expires_at, revoked_at FROM gateway_sessions WHERE token_hash=?", (digest,)
        )
        if row is None:
            return False
        expires_at, revoked_at = row[0], row[1]
        if revoked_at:
            return False
        if expires_at and str(expires_at) < now.isoformat():
            return False
        await db.execute(
            "UPDATE gateway_sessions SET last_seen_at=? WHERE token_hash=?",
            (now.isoformat(), digest),
        )
        return True

    async def revoke(self, token: str) -> bool:
        digest = _hash(token)
        cursor = await db.execute(
            "UPDATE gateway_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
            (_now().isoformat(), digest),
        )
        return bool(getattr(cursor, "rowcount", 0))

    async def stats(self) -> dict[str, Any]:
        rows = await db.fetchall(
            "SELECT COUNT(*) FROM gateway_sessions WHERE revoked_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > ?)",
            (_now().isoformat(),),
        )
        live = int(rows[0][0]) if rows else 0
        tickets = await db.fetchall(
            "SELECT COUNT(*) FROM gateway_attach_tickets WHERE used_at IS NULL AND expires_at > ?",
            (_now().isoformat(),),
        )
        return {"live_sessions": live, "unused_tickets": int(tickets[0][0]) if tickets else 0}
