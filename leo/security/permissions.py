"""Tool/agent permission enforcement with durable audit log and approvals."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .. import db
from ..config import Settings
from ..contracts import Decision, RiskLevel, new_id, utcnow
from ..logging import get_logger
from ..events import Event, EventBus
from .approval_buttons import ALWAYS_ARG_SCOPED_ACTIONS

_LOG = get_logger("security")


def _grant_scope(action: str, arguments_digest: str) -> str:
    """Storage key for an "Always" grant scoped to one exact call payload."""
    return f"grant:{action}:{arguments_digest}"


# A workspace trust is for ordinary development commands. Destructive,
# credential, privilege, arbitrary-code, outbound and self-modifying work keeps
# asking even when the workspace is trusted.
_DANGEROUS_REASON = re.compile(
    r"(?i)("
    r"destruct|credential|privilege|escalat|power/unit|remote rewrite|force|"
    r"recursive permission|removes an installed|key/keystore|private key|"
    r"outside allowed roots|unparseable|arbitrary code|hardline|substitution|"
    r"shell policy manual|shell disabled"
    r")"
)


def reason_is_trustable(reason: str | None) -> bool:
    """True when a workspace trust may cover a HIGH call for this reason."""
    return not _DANGEROUS_REASON.search(str(reason or ""))


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
        self._reviewer: Callable[..., Awaitable[tuple[str, str]]] | None = None
        # (action, command, workspace) -> (expires_at, verdict, note)
        self._review_cache: dict[tuple[str, str, str], tuple[float, str, str]] = {}

    def set_reviewer(
        self, reviewer: Callable[..., Awaitable[tuple[str, str]]] | None
    ) -> None:
        """Install the smart-mode auto-reviewer (or clear it with None)."""
        self._reviewer = reviewer

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
        risk_reason = str(context.get("risk_reason") or "")
        command_preview = str(context.get("command_preview") or "")
        # The reviewer needs the real command, but it must not be persisted in
        # the approval metadata: pop it before that mapping is built.
        review_command = str(context.pop("review_command", "") or "")
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

        # YOLO mode: run HIGH-risk work without ordinary prompts. Path rules,
        # the hardline floor, deny globs and medium-risk policy still apply.
        if (
            allowed and not violation and risk is RiskLevel.HIGH and not verified_approval
            and self.settings.security.approval_mode == "off"
        ):
            allowed, reason = True, "approval mode off"

        # A session/workspace grant answers "the operator already said yes to
        # this operation (or this workspace's ordinary work)". It is checked
        # before a new approval is created and can never waive a path violation.
        grant_applied = False
        if (
            allowed and not violation and risk is RiskLevel.HIGH and not verified_approval
            and self.settings.security.high_risk_requires_approval
            and self.settings.security.approval_mode != "off"
            and await self._has_live_grant(action, resource, current_digest, risk_reason)
        ):
            allowed, reason, grant_applied = True, "approval grant", True

        # Smart mode asks a cheap model to review an otherwise-uncovered HIGH
        # shell call before a human is bothered. A destructive/credential/
        # arbitrary-code reason never reaches the reviewer.
        review_answered = False
        if (
            allowed and not grant_applied and risk is RiskLevel.HIGH
            and self.settings.security.high_risk_requires_approval and not verified_approval
            and self.settings.security.approval_mode == "smart"
            and self.settings.security.reviewer_enabled
            and self._reviewer is not None
            and action in {"tool:shell", "tool:terminal"}
            and reason_is_trustable(risk_reason)
            and review_command.strip()
        ):
            verdict, note = await self._review(
                action, review_command, resource, risk_reason
            )
            if verdict == "allow":
                allowed, reason, grant_applied = (
                    True, f"auto-reviewed: {note or 'allowed'}", True,
                )
                review_answered = True
            elif verdict == "deny":
                allowed, reason = False, f"auto-review denied: {note or 'denied'}"
                review_answered = True

        # A grant already answered this request; asking again would queue a
        # second approval for an action the operator just allowed.
        if (
            allowed and not grant_applied and not review_answered and risk is RiskLevel.HIGH
            and self.settings.security.high_risk_requires_approval and not verified_approval
            and self.settings.security.approval_mode != "off"
        ):
            approval_id = new_id("apr")
            metadata = _safe_approval_metadata(context)
            if current_digest:
                metadata["arguments_digest"] = current_digest
            if risk_reason:
                metadata["risk_reason"] = risk_reason
            if command_preview:
                metadata["command_preview"] = command_preview
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
                await self.events.emit(
                    "approval.requested",
                    approval_id=approval_id,
                    actor=actor,
                    action=action,
                    resource=resource,
                    risk=risk.value,
                    # A non-secret canonical hash: the notifier offers "Always"
                    # for shell/terminal only when it can scope the grant to the
                    # exact command shown.
                    **({"arguments_digest": current_digest} if current_digest else {}),
                    **({"risk_reason": risk_reason} if risk_reason else {}),
                    **({"command": command_preview} if command_preview else {}),
                )

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

    async def _review(
        self,
        action: str,
        command: str,
        resource: str | None,
        risk_reason: str,
    ) -> tuple[str, str]:
        """Run the smart-mode reviewer with a timeout and a short verdict cache."""
        key = (action, str(command or ""), str(resource or ""))
        loop = asyncio.get_running_loop()
        cached = self._review_cache.get(key)
        if cached is not None and loop.time() < cached[0]:
            return cached[1], cached[2]
        timeout = float(
            getattr(self.settings.security, "reviewer_timeout_seconds", 20) or 20
        )
        try:
            verdict, note = await asyncio.wait_for(
                self._reviewer(  # type: ignore[misc]
                    command=command,
                    resource=resource,
                    risk_reason=risk_reason,
                    action=action,
                ),
                timeout=max(1.0, timeout),
            )
        except Exception:  # noqa: BLE001 - a dead reviewer falls back to a human
            return "ask", "reviewer unavailable"
        verdict = str(verdict or "").strip().lower()
        note = str(note or "")[:300]
        if verdict not in {"allow", "deny", "ask"}:
            return "ask", "reviewer gave no verdict"
        if verdict in {"allow", "deny"}:
            self._review_cache[key] = (loop.time() + 600.0, verdict, note)
        return verdict, note

    def _workspace_contains(self, workspace: str, resource: str | None) -> bool:
        """True when the call's cwd (or the default root) is inside the grant."""
        try:
            boundary = Path(str(workspace)).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        call_path = Path(str(resource)).expanduser().resolve() if resource else Path(
            self.settings.root
        ).resolve()
        try:
            call_path.relative_to(boundary)
            return True
        except ValueError:
            return False

    async def _has_live_grant(
        self,
        action: str,
        resource: str | None,
        arguments_digest: str | None = None,
        risk_reason: str | None = None,
    ) -> bool:
        """True when an unexpired grant covers this call.

        Path tools are keyed on ``(action, resource)``; shell/terminal on the
        exact arguments digest. A third scope, a **workspace trust**, covers
        ordinary shell commands in one workspace for its lifetime — but never a
        destructive, credential, privilege, arbitrary-code or self-modifying
        command. ``manual`` mode ignores every grant.
        """
        if self.settings.security.approval_mode == "manual":
            return False
        try:
            if action in ALWAYS_ARG_SCOPED_ACTIONS:
                digest = str(arguments_digest or "").strip()
                if digest:
                    exact = await db.fetchone(
                        "SELECT id FROM approval_grants WHERE action=? AND resource=? "
                        "AND expires_at > ? LIMIT 1",
                        (action, _grant_scope(action, digest), utcnow()),
                    )
                    if exact is not None:
                        return True
                if not reason_is_trustable(risk_reason):
                    return False
                rows = await db.fetchall(
                    "SELECT resource FROM approval_grants WHERE action=? "
                    "AND scope='workspace' AND expires_at > ?",
                    (action, utcnow()),
                )
                for item in rows or []:
                    if self._workspace_contains(str(item["resource"]), resource):
                        return True
                return False
            if resource is None or str(resource).strip() == "":
                return False
            row = await db.fetchone(
                "SELECT id FROM approval_grants WHERE action=? AND resource IS ? "
                "AND expires_at > ? LIMIT 1",
                (action, resource, utcnow()),
            )
            return row is not None
        except Exception:  # noqa: BLE001 - a grant lookup must never crash a decision
            return False

    async def grant_session(
        self,
        action: str,
        resource: str | None,
        seconds: int,
        granted_by: str,
        scope: str = "session",
        arguments_digest: str | None = None,
    ) -> str:
        """Record a scoped, expiring grant for one operation.

        Path tools store their resource. Shell/terminal store the canonical
        arguments digest, so "Always" covers the exact command shown and no
        other shell command. A grant that names neither is refused.
        """
        from datetime import timedelta

        digest = str(arguments_digest or "").strip()
        if action in ALWAYS_ARG_SCOPED_ACTIONS:
            if not digest:
                raise ValueError(
                    "a shell/terminal session grant needs the arguments digest"
                )
            stored_resource = _grant_scope(action, digest)
        else:
            if resource is None or str(resource).strip() == "":
                raise ValueError("a session grant needs a resource to scope it")
            stored_resource = str(resource)
            digest = ""
        grant_id = new_id("grn")
        created = utcnow()
        expires = (datetime.fromisoformat(created) + timedelta(seconds=max(60, int(seconds)))).isoformat()
        metadata = json.dumps({"arguments_digest": digest}) if digest else "{}"
        await db.execute(
            "INSERT INTO approval_grants(id,created_at,expires_at,action,resource,granted_by,scope,metadata) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (grant_id, created, expires, action, stored_resource, granted_by, scope, metadata),
        )
        if self.events:
            await self.events.emit("approval.granted", grant_id=grant_id, action=action,
                                   resource=stored_resource, granted_by=granted_by, scope=scope)
        return grant_id

    async def grant_workspace(
        self,
        action: str,
        workspace: str | None,
        seconds: int,
        granted_by: str,
    ) -> str:
        """Trust ordinary shell/terminal work in one workspace for a bounded time.

        Destructive, credential, privilege, arbitrary-code and self-modifying
        commands are excluded by :func:`reason_is_trustable`; the hardline and
        deny-glob floors still apply before this is consulted.
        """
        from datetime import timedelta

        if action not in ALWAYS_ARG_SCOPED_ACTIONS:
            raise ValueError("a workspace trust applies to shell/terminal only")
        boundary = str(
            Path(str(workspace or self.settings.root)).expanduser().resolve()
        )
        grant_id = new_id("grn")
        created = utcnow()
        expires = (datetime.fromisoformat(created) + timedelta(seconds=max(60, int(seconds)))).isoformat()
        await db.execute(
            "INSERT INTO approval_grants(id,created_at,expires_at,action,resource,granted_by,scope,metadata) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (grant_id, created, expires, action, boundary, granted_by, "workspace", "{}"),
        )
        if self.events:
            await self.events.emit(
                "approval.granted", grant_id=grant_id, action=action,
                resource=boundary, granted_by=granted_by, scope="workspace",
            )
        return grant_id

    async def list_grants(self, *, include_expired: bool = False) -> list[dict[str, Any]]:
        """List live approval grants (a workspace trust is one of these)."""
        sql = (
            "SELECT id,created_at,expires_at,action,resource,granted_by,scope "
            "FROM approval_grants"
        )
        params: tuple[Any, ...] = ()
        if not include_expired:
            sql += " WHERE expires_at > ?"
            params = (utcnow(),)
        sql += " ORDER BY created_at DESC"
        rows = await db.fetchall(sql, params)
        return [dict(row) for row in rows or []]

    async def revoke_grant(self, grant_id: str, *, actor: str = "operator") -> bool:
        """Revoke one grant immediately; the audit row is never deleted."""
        cursor = await db.execute(
            "DELETE FROM approval_grants WHERE id=?", (grant_id,)
        )
        revoked = bool(getattr(cursor, "rowcount", 0))
        if revoked:
            await self._audit(
                actor, "approval.grant.revoke", grant_id, RiskLevel.LOW,
                Decision(
                    allowed=True, reason="approval grant revoked", risk=RiskLevel.LOW,
                ),
            )
        return revoked

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
