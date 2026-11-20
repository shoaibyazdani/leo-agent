"""Measurable self-evaluation.

LEO proposes improvements from recorded evidence; it never modifies its own
running core. Proposals are inert records until a human tests and approves them.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from .. import db
from ..contracts import new_id, utcnow


class Evaluation:
    def __init__(self, namespace: str = "leo") -> None:
        self.namespace = namespace

    async def record(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        number = float(value)
        # NaN is stored as NULL by SQLite and violates the NOT NULL constraint;
        # infinities would corrupt every aggregate computed in report().
        if not math.isfinite(number):
            raise ValueError(f"metric value must be finite: {name}")
        await db.execute(
            "INSERT INTO metrics(ts,name,value,labels) VALUES(?,?,?,?)",
            (utcnow(), f"{self.namespace}.{name}", number, json.dumps(labels or {}, default=str)),
        )

    async def report(self, since: str | None = None) -> dict[str, Any]:
        if since:
            rows = await db.fetchall(
                "SELECT name, value, labels, ts FROM metrics WHERE name LIKE ? AND ts>=? ORDER BY ts",
                (f"{self.namespace}.%", since),
            )
        else:
            rows = await db.fetchall(
                "SELECT name, value, labels, ts FROM metrics WHERE name LIKE ? ORDER BY ts", (f"{self.namespace}.%",)
            )
        by_name: dict[str, list[float]] = {}
        for row in rows:
            by_name.setdefault(row["name"], []).append(float(row["value"]))
        return {
            name: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1],
            }
            for name, values in by_name.items()
        }

    async def task_metrics(self) -> dict[str, Any]:
        rows = await db.fetchall("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
        counts = {r["status"]: r["n"] for r in rows}
        total = sum(counts.values())
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        durations = await db.fetchall(
            "SELECT created_at, completed_at FROM tasks WHERE completed_at IS NOT NULL AND status='completed'"
        )
        seconds = []
        for row in durations:
            try:
                start = datetime.fromisoformat(row["created_at"])
                end = datetime.fromisoformat(row["completed_at"])
                seconds.append((end - start).total_seconds())
            except (ValueError, TypeError):
                continue
        return {
            "total": total,
            "by_status": counts,
            "success_rate": completed / total if total else 0.0,
            "failure_rate": failed / total if total else 0.0,
            "mean_completion_seconds": sum(seconds) / len(seconds) if seconds else None,
        }

    async def tool_reliability(self) -> dict[str, Any]:
        rows = await db.fetchall(
            "SELECT tool, COUNT(*) AS calls, SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) AS ok, "
            "AVG(duration_ms) AS avg_ms FROM tool_calls GROUP BY tool"
        )
        return {
            r["tool"]: {
                "calls": r["calls"],
                "success_rate": r["ok"] / r["calls"] if r["calls"] else 0.0,
                "avg_duration_ms": r["avg_ms"],
            }
            for r in rows
        }

    async def detect_regressions(
        self,
        metric: str,
        window: int = 10,
        threshold: float = 0.2,
        higher_is_better: bool | None = None,
    ) -> list[dict[str, Any]]:
        if window < 1:
            return []
        rows = await db.fetchall(
            "SELECT value, ts FROM metrics WHERE name=? ORDER BY id DESC LIMIT ?", (f"{self.namespace}.{metric}", window * 2)
        )
        values = [float(r["value"]) for r in rows]
        if len(values) < window * 2:
            return []
        older = sum(values[window:]) / window
        recent = sum(values[:window]) / window
        if older == 0:
            return []
        change = (recent - older) / abs(older)
        if higher_is_better is None:
            higher_is_better = any(token in metric for token in ("success", "score", "accuracy", "reliability", "rate"))
            if any(token in metric for token in ("latency", "cost", "duration", "error", "failure", "time")):
                higher_is_better = False
        regressed = change < -threshold if higher_is_better else change > threshold
        if regressed:
            return [{"metric": metric, "change": change, "older_mean": older, "recent_mean": recent, "severity": "regression"}]
        return []

    async def propose(self, title: str, rationale: str, evidence: dict[str, Any]) -> dict[str, Any]:
        proposal = {
            "id": new_id("imp"),
            "title": title,
            "rationale": rationale,
            "evidence": evidence,
            "status": "proposed",
            "requires_tests": True,
            "requires_approval": True,
            "created_at": utcnow(),
        }
        await db.execute(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)",
            (f"improvement:{proposal['id']}", json.dumps(proposal, default=str), utcnow()),
        )
        await self.record("improvement.proposed", 1, {"id": proposal["id"]})
        return proposal

    async def proposals(self) -> list[dict[str, Any]]:
        rows = await db.fetchall("SELECT value FROM kv WHERE key LIKE 'improvement:%' ORDER BY updated_at DESC")
        return [json.loads(r["value"]) for r in rows]

    async def adopt(self, proposal_id: str, tests_passed: bool, approver: str) -> dict[str, Any]:
        """Record adoption only when tests passed and a human approved.

        Adoption is a one-way ``proposed -> adopted`` transition. The read and
        write happen in one write transaction so two concurrent adoptions
        cannot both observe ``proposed`` and overwrite each other's audit trail.
        """
        if not tests_passed or not str(approver or "").strip():
            raise ValueError("adoption requires passing tests and an approver")
        key = f"improvement:{proposal_id}"
        async with db.transaction() as conn:
            cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (key,))
            row = await cursor.fetchone()
            if not row:
                raise KeyError(proposal_id)
            proposal = json.loads(row["value"])
            if proposal.get("status") != "proposed":
                raise ValueError(f"proposal is no longer proposed: {proposal_id}")
            proposal.update(status="adopted", tests_passed=True, approver=approver, adopted_at=utcnow())
            await conn.execute(
                "UPDATE kv SET value=?, updated_at=? WHERE key=?",
                (json.dumps(proposal, default=str), utcnow(), key),
            )
        return proposal
