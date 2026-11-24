"""Deterministic lifecycle maintenance for agent-authored skills.

The curator only prunes; it never rewrites or deletes content. A skill that has
not been viewed, invoked or patched for ``stale_after_days`` becomes ``stale``;
after ``archive_after_days`` it is archived (still readable, restorable, and
excluded from the prompt index). Activity brings a stale skill back to active.

Exemptions are explicit:

* native shipped skills (``created_by != "agent"``),
* pinned skills,
* anything already archived (no repeated transitions).

The LLM consolidation pass is deliberately not part of this engine. It would
rewrite content at scale; when it exists it must be a proposal, not an
automatic mutation.
"""

from __future__ import annotations

import difflib
import json
import re
from datetime import UTC, datetime
from typing import Any

from .. import db
from ..contracts import utcnow
from ..events import EventBus
from ..logging import get_logger

_LOG = get_logger("curator")
LAST_RUN_KEY = "curator:last_run"


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class CuratorEngine:
    """Usage-driven stale/archive transitions for authored skills."""

    def __init__(self, settings: Any, skills: Any, events: EventBus | None = None) -> None:
        self.settings = settings
        self.skills = skills
        self.events = events

    @property
    def config(self) -> Any:
        return getattr(getattr(self.settings, "skills", None), "curator", None)

    @property
    def enabled(self) -> bool:
        cfg = self.config
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    async def last_run(self) -> dict[str, Any] | None:
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (LAST_RUN_KEY,))
        if row is None:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    async def _record_run(self, summary: dict[str, Any]) -> None:
        await db.execute(
            "INSERT OR REPLACE INTO kv(key,value,updated_at) VALUES(?,?,?)",
            (LAST_RUN_KEY, json.dumps(summary, default=str), utcnow()),
        )

    async def scan(
        self, *, now: datetime | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """Apply deterministic transitions; return a truthful count summary."""
        cfg = self.config
        if cfg is None or not getattr(cfg, "enabled", False):
            return {"status": "disabled", "scanned": 0, "staled": 0, "archived": 0, "reactivated": 0}
        now = now or datetime.now(UTC)
        stale_days = max(1, int(getattr(cfg, "stale_after_days", 14) or 14))
        archive_days = max(stale_days, int(getattr(cfg, "archive_after_days", 30) or 30))
        usage = await self.skills.usage_list()
        records = {record.name: record for record in await self.skills.list()}
        counts = {
            "status": "completed",
            "scanned": 0,
            "staled": 0,
            "archived": 0,
            "reactivated": 0,
            "skipped": 0,
            "errors": 0,
            "dry_run": bool(dry_run),
            "ran_at": now.isoformat(),
        }
        for row in usage:
            name = str(row.get("skill_name") or "")
            if not name:
                continue
            counts["scanned"] += 1
            record = records.get(name)
            if record is None:
                counts["skipped"] += 1
                continue
            if str(row.get("created_by") or record.created_by) != "agent":
                counts["skipped"] += 1
                continue
            extra = getattr(record, "model_extra", None) or {}
            if extra.get("pinned"):
                counts["skipped"] += 1
                continue
            state = str(row.get("state") or "active")
            if state == "archived":
                counts["skipped"] += 1
                continue
            last = max(
                (value for value in (
                    _parse(row.get("last_viewed_at")),
                    _parse(row.get("last_used_at")),
                    _parse(row.get("last_patched_at")),
                    _parse(row.get("updated_at")),
                ) if value is not None),
                default=None,
            )
            if last is None:
                counts["skipped"] += 1
                continue
            age_days = max(0.0, (now - last).total_seconds() / 86_400.0)
            try:
                if age_days >= archive_days:
                    if not dry_run:
                        await self.skills.set_state(name, "archived")
                    counts["archived"] += 1
                elif age_days >= stale_days:
                    if state != "stale":
                        if not dry_run:
                            await self.skills.set_state(name, "stale")
                        counts["staled"] += 1
                elif state == "stale":
                    if not dry_run:
                        await self.skills.set_state(name, "active")
                    counts["reactivated"] += 1
                else:
                    counts["skipped"] += 1
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop the pass
                counts["errors"] += 1
                _LOG.warning("curator transition failed for %s: %s", name, exc)
        if not dry_run:
            await self._record_run(counts)
            if self.events is not None:
                try:
                    await self.events.emit("skill.curator", **counts)
                except Exception as exc:  # noqa: BLE001 - telemetry must not fail the pass
                    _LOG.warning("could not emit curator event: %s", exc)
        return counts

    async def due(
        self, *, now: datetime | None = None, idle_since: datetime | None = None
    ) -> bool:
        """Whether a background pass is due (interval + idle window)."""
        cfg = self.config
        if cfg is None or not getattr(cfg, "enabled", False):
            return False
        now = now or datetime.now(UTC)
        last = await self.last_run()
        interval = max(1, int(getattr(cfg, "interval_hours", 168) or 168))
        last_at = _parse((last or {}).get("ran_at"))
        if last_at is None:
            # First observation seeds the clock and defers one full interval, so
            # a fresh install is never pruned before the operator can review it.
            await self._record_run({"status": "seeded", "ran_at": now.isoformat()})
            return False
        if (now - last_at).total_seconds() < interval * 3600:
            return False
        idle_hours = float(getattr(cfg, "min_idle_hours", 2.0) or 0.0)
        return idle_since is None or (
            (now - idle_since).total_seconds() >= idle_hours * 3600
        )

    async def candidates(
        self, *, threshold: float = 0.72, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Deterministic near-duplicate groups; a report, never a mutation."""
        records = [
            record for record in await self.skills.list()
            if record.created_by == "agent" and record.enabled
        ]

        def text(record: Any) -> str:
            return re.sub(
                r"[^a-z0-9]+", " ",
                f"{record.name} {record.description}".lower(),
            ).strip()

        parent = {record.name: record.name for record in records}

        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        groups: dict[str, dict[str, Any]] = {}
        for index, first in enumerate(records):
            for second in records[index + 1:]:
                ratio = difflib.SequenceMatcher(
                    None, text(first), text(second)
                ).ratio()
                if ratio < threshold:
                    continue
                left, right = find(first.name), find(second.name)
                if left != right:
                    parent[right] = left
                root = find(left)
                group = groups.setdefault(
                    root, {"skills": set(), "similarity": 0.0}
                )
                group["skills"].update({first.name, second.name})
                group["similarity"] = max(group["similarity"], ratio)
        items = [
            {
                "skills": sorted(group["skills"]),
                "similarity": round(float(group["similarity"]), 3),
                "reason": "near-duplicate class-level skills; consider one umbrella",
            }
            for group in groups.values() if len(group["skills"]) >= 2
        ]
        items.sort(key=lambda item: (-item["similarity"], item["skills"]))
        return items[: max(1, limit)]

    def status(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "enabled": self.enabled,
            "interval_hours": int(getattr(cfg, "interval_hours", 168) or 168),
            "min_idle_hours": float(getattr(cfg, "min_idle_hours", 2.0) or 0.0),
            "stale_after_days": int(getattr(cfg, "stale_after_days", 14) or 14),
            "archive_after_days": int(getattr(cfg, "archive_after_days", 30) or 30),
        }


__all__ = ["LAST_RUN_KEY", "CuratorEngine"]
