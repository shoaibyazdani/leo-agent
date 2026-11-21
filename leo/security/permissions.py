"""Tool/agent permission enforcement with durable audit log and approvals."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .. import db
from ..config import Settings
from ..contracts import Decision, RiskLevel, new_id, utcnow
from ..logging import get_logger
from ..events import Event, EventBus

_LOG = get_logger("security")


def arguments_digest(arguments: Any) -> str | None:
    """Canonical SHA-256 of an approval's tool arguments.

    Canonical JSON (sorted keys, compact separators) makes the digest stable
    across dict insertion order while binding every value — target, text,
    path, program — to the approval that was actually shown to a human.
    """
    if arguments is None:
        return None
    try:
        canonical = json.dumps(
            arguments, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), default=str,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_metadata_value(value: Any, depth: int = 0) -> Any:
    """Keep only JSON scalars/containers; never serialize live service objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if depth >= 4:
        return "[truncated]"
    if isinstance(value, Mapping):
        return {
            str(key): _safe_metadata_value(item, depth + 1)
            for key, item in value.items()
            if isinstance(key, (str, int, float, bool))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)
        return [_safe_metadata_value(item, depth + 1) for item in items[:50]]
    # Settings, EventBus, clients, locks, callbacks, ... must never be dumped.
    return None


def _safe_approval_metadata(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): clean
        for key, value in context.items()
        if (clean := _safe_metadata_value(value)) is not None
    }


class ApprovalRequired(Exception):
    def __init__(self, approval_id: str, reason: str) -> None:
        self.approval_id = approval_id
        self.reason = reason
        super().__init__(f"approval required ({approval_id}): {reason}")


class PermissionManagerImpl:
    """Default policy: low-risk allowed, medium-risk configurable, high-risk approved.

    Every decision is written to audit_log. Path-bearing resources must resolve
    inside an allowed root when workspace_only is enabled.
    """

    def __init__(self, settings: Settings, events: EventBus | None = None) -> None:
        self.settings = settings
        self.events = events
        self._roots = [settings.root.resolve()]
        for p in settings.security.allowed_roots:
            self._roots.append(settings.resolve(p).resolve())
        self._roots.append(settings.resolve(settings.paths.workspaces).resolve())

    def _path_violation(self, resource: str | None) -> str | None:
        if not self.settings.security.workspace_only or not resource:
            return None
        if "\x00" in str(resource):
            return "path contains NUL byte"
        candidate = resource if resource.startswith("/") else str(self.settings.root / resource)
        try:
            resolved = Path(candidate).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return "could not resolve resource path"
        for root in self._roots:
            try:
                resolved.relative_to(root)
                return None
            except ValueError:
                continue
        return f"path outside allowed roots: {resolved}"

    def _approval_is_valid(
        self,
        row: Any,
        actor: str,
        action: str,
        resource: str | None,
        risk: RiskLevel,
        arguments_digest_value: str | None = None,
    ) -> bool:
        """An approval ID grants authority only if every binding condition holds."""
        if not row or row["status"] != "approved":
            return False
        if row["action"] != action or row["resource"] != resource:
            return False
        if row["risk"] != risk.value or row["actor"] != actor:
            return False
        # A HIGH-risk approval is bound to the exact arguments that were
        # approved. Changing target/text/path/program yields a different digest
        # and must never consume the earlier decision.
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        expected = metadata.get("arguments_digest") if isinstance(metadata, Mapping) else None
        if expected:
            if not arguments_digest_value or not hmac.compare_digest(str(expected), str(arguments_digest_value)):
                return False
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - created > timedelta(seconds=self.settings.security.approval_timeout_seconds):
                return False
        except (TypeError, ValueError):
            return False
        return True

    def granted_capabilities(self, grants: Iterable[str] | None = None) -> frozenset[str] | None:
        """Return the effective grant set, or ``None`` when no policy is set.

        An explicit empty list is a real policy (deny everything); it is not
        the same as the unset default, which preserves risk-only enforcement.
        """
        if grants is None:
            configured = self.settings.security.grants
            if configured is None:
                return None
            grants = configured
        if isinstance(grants, str):
            grants = [grants]
        return frozenset(str(item).strip() for item in grants if str(item).strip())

    def missing_capabilities(
        self, required: Iterable[str] | None, grants: Iterable[str] | None = None
    ) -> list[str]:
        """Capabilities declared by a ToolSpec but not present in caller grants."""
        available = self.granted_capabilities(grants)
        if available is None:
            return []
        if "*" in available:
            return []
        needed = {str(item).strip() for item in (required or []) if str(item).strip()}
        return sorted(needed - available)

    async def authorize(
        self,
        actor: str,
        action: str,
        resource: str | None = None,
        risk: RiskLevel = RiskLevel.LOW,
        **context: Any,
    ) -> Decision:
        risk = RiskLevel(risk)
        reason = ""
        allowed = True
        requires_approval = False
        approval_id: str | None = None

        violation = self._path_violation(resource)
        if violation:
            allowed, reason = False, violation

        # Previously granted approval for this exact action/resource/actor/risk.
        # An unverified ID is never sufficient to clear a HIGH-risk gate.
        supplied_approval = context.get("approval_id")
        current_digest = context.get("arguments_digest")
        if not current_digest and context.get("arguments") is not None:
            current_digest = arguments_digest(context.get("arguments"))
        verified_approval = False
        if supplied_approval:
            row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (supplied_approval,))
            if self._approval_is_valid(row, actor, action, resource, risk, current_digest) and not violation:
                # Workspace/path restrictions are a policy floor: an approval
                # can authorize an action but cannot waive them.
                allowed, reason, approval_id = True, "prior approval verified", supplied_approval
                verified_approval = True
            else:
                if risk is RiskLevel.HIGH:
                    allowed = False
                    # A path/policy violation is the real reason for the denial;
                    # do not mask it with a message about the approval id.
                    if not violation:
                        if row is not None and row["status"] == "consumed":
                            reason = "approval already consumed"
                        else:
                            reason = "supplied approval id is not valid for this action"

        if allowed and risk is RiskLevel.MEDIUM and not self.settings.security.allow_medium_risk:
            allowed, reason = False, "medium-risk actions disabled by policy"

        if allowed and risk is RiskLevel.HIGH and self.settings.security.high_risk_requires_approval and not verified_approval:
            approval_id = new_id("apr")
            metadata = _safe_approval_metadata(context)
            if current_digest:
                metadata["arguments_digest"] = current_digest
            await db.execute(
                "INSERT INTO approvals(id,created_at,status,actor,action,resource,risk,reason,requested_by,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    approval_id, utcnow(), "pending", actor, action, resource, risk.value,
                    "high-risk action requires explicit approval", actor,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            allowed, requires_approval, reason = False, True, "approval required"
            if self.events:
                await self.events.emit("approval.requested", approval_id=approval_id, actor=actor, action=action, resource=resource, risk=risk.value)

        decision = Decision(
            allowed=allowed, reason=reason, risk=risk,
            requires_approval=requires_approval, approval_id=approval_id,
            verified_approval=verified_approval,
        )
        audit_metadata: dict[str, Any] = {}
        if verified_approval and approval_id:
            # Make the grant traceable: the audit row must name the exact
            # approval that authorized a HIGH-risk action.
            audit_metadata["approval_id"] = approval_id
        await self._audit(actor, action, resource, risk, decision, audit_metadata)
        return decision

    async def consume_approval(self, approval_id: str) -> bool:
        """Atomically move an approved row to consumed; one caller wins.

        ``db.execute`` reuses the caller's transaction connection when one is
        open, so this cannot deadlock behind a handler's own write transaction.
        Tools opt in via ``ToolSpec.consume_approval``; multi-step operations
        that manage their own consumption are unaffected.
        """
        cursor = await db.execute(
            "UPDATE approvals SET status='consumed', resolved_at=? "
            "WHERE id=? AND status='approved'",
            (utcnow(), approval_id),
        )
        return cursor.rowcount == 1

    async def resolve_approval(self, approval_id: str, approved: bool, resolver: str = "user") -> Decision:
        resolver_name = str(resolver).strip()
        if resolver_name.startswith("chat:"):
            # No self-approval, independently of any ToolSpec or channel scope:
            # a chat actor is the requesting principal, never the human gate.
            row = await db.fetchone("SELECT action,resource,risk FROM approvals WHERE id=?", (approval_id,))
            if not row:
                raise KeyError(f"unknown approval: {approval_id}")
            risk = RiskLevel(row["risk"])
            decision = Decision(
                allowed=False, reason="chat actors cannot resolve approvals",
                risk=risk, approval_id=approval_id,
            )
            await self._audit(
                resolver_name, row["action"], row["resource"], risk, decision,
                metadata={"approval_id": approval_id, "resolver": resolver_name,
                          "denied_reason": "self_approval"},
            )
            raise PermissionError(decision.reason)

        # The status check, the status change and the audit row used to live on
        # different connections, so a replay could resolve twice or leave an
        # approval committed without its audit trail. One serialized write
        # transaction plus a conditional UPDATE prevents both; an audit failure
        # rolls the state change back with it.
        async with db.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,))
            row = await cursor.fetchone()
            if not row:
                raise KeyError(f"unknown approval: {approval_id}")
            if row["status"] != "pending":
                raise ValueError(f"approval already {row['status']}")
            status = "approved" if approved else "denied"
            update = await conn.execute(
                "UPDATE approvals SET status=?, resolved_at=?, "
                "metadata=json_set(COALESCE(metadata,'{}'),'$.resolved_by',?) "
                "WHERE id=? AND status='pending'",
                (status, utcnow(), resolver_name, approval_id),
            )
            if update.rowcount != 1:
                raise ValueError(f"approval already {row['status']}")
            # Copy out of the row before the connection is closed.
            risk = RiskLevel(row["risk"])
            action = row["action"]
            resource = row["resource"]
            requested_by = row["requested_by"]
            decision = Decision(
                allowed=approved, reason=f"approval {status} by {resolver_name}",
                risk=risk, approval_id=approval_id,
            )
            # db.execute reuses this transaction's connection (see leo.db), so
            # the audit row and the approval decision commit or roll back together.
            await self._audit(
                resolver_name, action, resource, risk, decision,
                metadata={"approval_id": approval_id, "resolver": resolver_name,
                          "requested_by": requested_by},
            )

        if self.events:
            await self.events.emit("approval.resolved", approval_id=approval_id, status=status, resolver=resolver_name)
        return decision

    async def pending_approvals(self) -> list[dict[str, Any]]:
        rows = await db.fetchall("SELECT * FROM approvals WHERE status='pending' ORDER BY created_at")
        return [dict(r) for r in rows]

    async def cleanup_approvals(
        self,
        source: str | None = None,
        older_than_hours: float = 24.0,
        resolver: str = "operator",
        apply: bool = False,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Report stale pending approvals; with ``apply`` expire them.

        Expiry is a state change only — rows are never deleted, so what was
        requested stays inspectable. Default is a dry run that lists candidates
        without touching anything; ``apply`` marks each row ``expired`` with
        the resolver recorded in its metadata, one audit row per approval,
        inside a single transaction so a failure leaves nothing half-expired.
        ``source`` scopes the operation to one requesting actor (e.g.
        ``learning:engine``); it must never be ``chat:*``.
        """
        older_than_hours = float(older_than_hours)
        if older_than_hours < 0:
            raise ValueError("older_than_hours must be >= 0")
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be >= 1")
        resolver_name = str(resolver).strip() or "operator"
        if resolver_name.startswith("chat:"):
            raise ValueError("chat actors cannot resolve approvals")
        source_name = str(source).strip() if source else None
        if source_name and source_name.startswith("chat:"):
            raise ValueError("cleanup cannot be scoped to a chat actor")

        # utcnow() is an ISO string, so the cutoff stays the same shape and
        # the SQL comparison stays lexicographic-safe.
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).isoformat()
        where = "status='pending' AND created_at<=?"
        params: list[Any] = [cutoff]
        if source_name:
            where += " AND actor=?"
            params.append(source_name)
        params.append(min(limit, 5000))
        rows = await db.fetchall(
            f"SELECT id,created_at,actor,action,resource,risk FROM approvals "
            f"WHERE {where} ORDER BY created_at LIMIT ?",
            tuple(params),
        )
        candidates = [dict(r) for r in rows]
        report: dict[str, Any] = {
            "apply": bool(apply),
            "source": source_name,
            "older_than_hours": older_than_hours,
            "candidates": len(candidates),
        }
        if not apply:
            report["sample"] = candidates[:20]
            return report

        expired: list[str] = []
        async with db.transaction() as conn:
            for row in candidates:
                cursor = await conn.execute(
                    "UPDATE approvals SET status='expired', resolved_at=?, "
                    "metadata=json_set(COALESCE(metadata,'{}'),'$.resolved_by',?) "
                    "WHERE id=? AND status='pending'",
                    (utcnow(), resolver_name, row["id"]),
                )
                if cursor.rowcount != 1:
                    continue  # resolved concurrently; leave the row alone
                expired.append(row["id"])
                risk = RiskLevel(row["risk"])
                await self._audit(
                    resolver_name, row["action"], row["resource"], risk,
                    Decision(
                        allowed=False,
                        reason=f"pending approval expired by cleanup ({resolver_name})",
                        risk=risk, approval_id=row["id"],
                    ),
                    metadata={
                        "approval_id": row["id"], "resolver": resolver_name,
                        "cleanup": True, "requested_by": row["actor"],
                    },
                )
        if self.events and expired:
            await self.events.emit(
                "approval.resolved", status="expired",
                resolver=resolver_name, count=len(expired),
            )
        report["expired"] = len(expired)
        report["ids"] = expired[:50]
        return report

    async def _audit(
        self, actor: str, action: str, resource: str | None, risk: RiskLevel,
        decision: Decision, metadata: dict[str, Any] | None = None,
    ) -> None:
        await db.execute(
            "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) VALUES(?,?,?,?,?,?,?,?)",
            (utcnow(), actor, action, resource, risk.value, "allowed" if decision.allowed else ("pending" if decision.requires_approval else "denied"), decision.reason, json.dumps(metadata or {}, default=str)),
        )
