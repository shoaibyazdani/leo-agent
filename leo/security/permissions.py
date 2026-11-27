"""Tool/agent permission enforcement with durable audit log and approvals."""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import hmac
import json
import re
import shlex
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .. import db
from ..config import Settings
from ..contracts import Decision, RiskLevel, new_id, surface_capabilities
from ..logging import get_logger
from ..events import Event, EventBus
from ..schedules.clock import Clock, SystemClock
from .approval_buttons import ALWAYS_ARG_SCOPED_ACTIONS, SESSION_GRANT_SECONDS
from .approval_cards import (
    APPROVAL_CARD_PROTOCOL_VERSION,
    DECLINE_MESSAGES,
    KIND_SCHEDULE_CREATE,
    KIND_SCHEDULE_EDIT,
    ScheduleResolution,
    at_or_after,
    card_fits_surface,
    card_from_row,
    choice_labels_for,
    current_card_presenter,
    deadline_text,
    is_schedule_kind,
    kind_config,
    normalize_choice,
    parse_instant,
    proposal_digest_matches,
)

_LOG = get_logger("security")


def _grant_scope(action: str, arguments_digest: str) -> str:
    """Storage key for an "Always" grant scoped to one exact call payload."""
    return f"grant:{action}:{arguments_digest}"


# A schedule create grant is its own authority. It never covers edits, remove,
# run, cancel, a future tool, workspace trust or an ordinary tool call.
SCHEDULE_CREATE_ACTION = "schedule:create"
SCHEDULE_SESSION_GRANT_SECONDS = 8 * 60 * 60
SCHEDULE_ALWAYS_GRANT_DAYS = 30
_MAX_SCHEDULE_GRANT_DAYS = 365
_MAX_DENY_REASON_CHARS = 500
_SCHEDULE_ACTIONS: dict[str, str] = {
    "schedule_create": "schedule:create",
    "schedule_edit": "schedule:edit",
}


def _bounded_deny_reason(value: Any) -> str:
    text = str(value or "").strip().replace("\x00", "")
    return text[:_MAX_DENY_REASON_CHARS]


# A workspace trust, a smart-review verdict or a remembered grant may cover
# ordinary work only. The floor is deliberately keyed to the *specific* reason
# a classifier escalated a call, not to broad words like "code": "python runs
# arbitrary code" is the ordinary way LEO runs tests and scripts, so it must
# reach the reviewer and the trust button just like "unclassified command".
_HARDLINE_REASON = re.compile(
    r"(?i)("
    r"hardline|deny_commands|destruct|delet|wipe|credential|private key|"
    r"key/keystore|secret|password|privilege|escalat|sudo|doas|"
    r"remote rewrite|force|branch deletion|outside allowed roots|"
    r"redirection outside|unparseable|substitution|recursive permission|"
    r"removes an installed|power/unit|host power|awk can execute|"
    r"shell disabled|shell policy manual|self-modif|agent.s own|"
    r"pipe a remote|downloads or posts|outbound"
    r")"
)
_ARBITRARY_CODE_REASON = re.compile(r"(?i)\barbitrary code\b")
_ORDINARY_INTERPRETER_REASON = re.compile(
    r"^(?:python|python3|node|ruby|perl|php) runs arbitrary code$", re.I
)


def reason_is_trustable(reason: str | None) -> bool:
    """True when a smart reviewer, trust or grant may cover this call.

    Hardline reasons — destructive, credential, privilege, force/remote rewrite,
    substitution, unparsed input, out-of-root redirection, shell policy overrides
    and similar — always stay with the human. An interpreter such as ``python``
    or ``node`` classified as HIGH for "runs arbitrary code" is *ordinary* dev
    work: the reviewer must see it, and a trusted workspace must cover it.
    """
    text = str(reason or "").strip()
    if _ORDINARY_INTERPRETER_REASON.match(text):
        return True
    return not _HARDLINE_REASON.search(text) and not _ARBITRARY_CODE_REASON.search(text)


_GRANT_SCOPE_PATTERN = "pattern"
_GRANT_SCOPE_DIRECTORY = "directory"
_GRANT_SCOPE_STANDING = "standing"
_GRANT_SCOPE_CONVERSATION = "conversation"
_STANDING_EXPIRY = "9999-12-31T23:59:59+00:00"
_EXPIRY_GRACE_SECONDS = 2.0
_MAX_PATTERN_CHARS = 300


def _normalise_pattern(pattern: Any) -> str:
    """Validate an operator-supplied command pattern (``pytest *``, ``git *``)."""
    text = " ".join(str(pattern or "").split())
    if not text or len(text) > _MAX_PATTERN_CHARS or "\x00" in text:
        raise ValueError("a command pattern must be a bounded, non-empty string")
    if text.strip() in {"*", "**", "* *"}:
        raise ValueError("a command pattern must name a program, not match everything")
    return text


def _command_pattern(command: Any) -> str | None:
    """Derive the 'kind of work' grant from one approved command."""
    try:
        tokens = shlex.split(str(command or ""))
    except ValueError:
        return None
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens.pop(0)
    if not tokens:
        return None
    name = tokens[0]
    if not re.fullmatch(r"[A-Za-z0-9_./-]{1,80}", name):
        return None
    return f"{name} *"


def _pattern_matches(command: str, pattern: str) -> bool:
    """``pytest *`` covers both ``pytest -q`` and bare ``pytest``."""
    text = " ".join(str(command or "").split())
    if not text:
        return False
    if fnmatch.fnmatchcase(text, pattern):
        return True
    return pattern.endswith(" *") and text == pattern[:-2]


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

    def __init__(
        self,
        settings: Settings,
        events: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.settings = settings
        self.events = events
        # Expiry checks and audit timestamps share the injected clock so a
        # frozen clock can test deadlines without sleeping.
        self.clock = clock or SystemClock()
        self._roots = [settings.root.resolve()]
        for p in settings.security.allowed_roots:
            self._roots.append(settings.resolve(p).resolve())
        self._roots.append(settings.resolve(settings.paths.workspaces).resolve())
        self._reviewer: Callable[..., Awaitable[tuple[str, str]]] | None = None
        # (action, command, workspace) -> (expires_at, verdict, note)
        self._review_cache: dict[tuple[str, str, str], tuple[float, str, str]] = {}
        # Bounded expiry timers for command approvals. A timer is a loop handle,
        # not a long-lived task: closing a loop drops it silently.
        self._expiry_handles: dict[str, asyncio.TimerHandle] = {}
        self._expiry_tasks: set[asyncio.Task[Any]] = set()
        # Phase 5 registers the one ScheduleService commit callback here. It is
        # invoked inside the same transaction as the decision + grant + audit,
        # so a failed schedule write rolls the approval back with it.
        self._schedule_committer: Callable[..., Awaitable[Any]] | None = None

    def _now(self) -> datetime:
        return self.clock.now()

    def _now_iso(self) -> str:
        return self._now().isoformat()

    def set_reviewer(
        self, reviewer: Callable[..., Awaitable[tuple[str, str]]] | None
    ) -> None:
        """Install the smart-mode auto-reviewer (or clear it with None)."""
        self._reviewer = reviewer

    def set_schedule_committer(
        self, committer: Callable[..., Awaitable[Any]] | None
    ) -> None:
        """Install the one schedule commit authority (``ScheduleService``).

        The callback receives ``card``, ``choice`` and ``resolver`` keyword
        arguments and runs inside the decision transaction. It must not open a
        second transaction. Passing ``None`` removes it; a card then resolves
        without a schedule write, which only Phase 4 wiring and tests do.
        """
        self._schedule_committer = committer

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
            if self._now() - created > timedelta(seconds=self.settings.security.approval_timeout_seconds):
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
        grant_command = review_command or command_preview
        task_id = str(context.get("task_id") or context.get("conversation_id") or "").strip()
        conversation_id = str(context.get("conversation_id") or "").strip()
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
            and await self._has_live_grant(
                action, resource, current_digest, risk_reason,
                command=grant_command, task_id=task_id,
                conversation_id=conversation_id,
            )
        ):
            allowed, reason, grant_applied = True, "approval grant", True

        # Smart mode asks a cheap model to review an otherwise-uncovered HIGH
        # shell call before a human is bothered. Ordinary interpreters and
        # unclassified dev commands reach it; hardline reasons never do.
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
                    approval_id, self._now_iso(), "pending", actor, action, resource, risk.value,
                    "high-risk action requires explicit approval", actor,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            allowed, requires_approval, reason = False, True, "approval required"
            self._schedule_expiry(
                approval_id,
                float(getattr(self.settings.security, "approval_timeout_seconds", 900) or 900),
            )
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
            (self._now_iso(), approval_id),
        )
        return cursor.rowcount == 1

    # -- bounded approval expiry ------------------------------------------
    def _cancel_expiry(self, approval_id: str) -> None:
        handle = self._expiry_handles.pop(str(approval_id), None)
        if handle is not None:
            handle.cancel()

    def _schedule_expiry(self, approval_id: str, timeout: float) -> None:
        """Arm one timed expiry; later than the waiter so a timeout reads honestly.

        The executor's own bounded wait ends at the configured timeout. The
        expiry marker fires just after it, so the operator gets an explicit
        "what was waiting" notice without turning a pending card into an expired
        row underneath a still-waiting turn.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop: the caller has no wait to expire
            return
        delay = max(1.0, float(timeout)) + _EXPIRY_GRACE_SECONDS
        self._expiry_handles[str(approval_id)] = loop.call_later(
            delay, self._expiry_due, str(approval_id)
        )

    def _expiry_due(self, approval_id: str) -> None:
        self._expiry_handles.pop(approval_id, None)
        try:
            task = asyncio.create_task(self.expire_approval_if_stale(approval_id))
        except RuntimeError:  # loop closed between the timer and the callback
            return
        self._expiry_tasks.add(task)
        task.add_done_callback(self._expiry_tasks.discard)

    async def expire_approval_if_stale(self, approval_id: str) -> bool:
        """Mark a timed-out command approval expired and tell the operator.

        Only a still-pending row past its own deadline changes. The event is
        emitted with the exact command, action and risk reason so the notifier
        can say what was left waiting instead of letting the card vanish into a
        silent stall. Returns True when this call expired the card.
        """
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (str(approval_id),))
        if row is None or str(row["status"]) != "pending":
            self._cancel_expiry(approval_id)
            return False
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return False
        timeout = float(getattr(self.settings.security, "approval_timeout_seconds", 900) or 900)
        if self._now() < created + timedelta(seconds=max(0.0, timeout)):
            return False
        now = self._now_iso()
        async with db.transaction() as conn:
            cursor = await conn.execute(
                "UPDATE approvals SET status='expired', resolved_at=?, "
                "metadata=json_set(COALESCE(metadata,'{}'),'$.resolved_by',?,"
                "'$.outcome',?) WHERE id=? AND status='pending'",
                (now, "system:expiry", "timeout", str(approval_id)),
            )
            if cursor.rowcount != 1:
                return False
            risk = RiskLevel(str(row["risk"]))
            await self._audit(
                "system:expiry", str(row["action"]), row["resource"], risk,
                Decision(
                    allowed=False,
                    reason="approval expired: the operator did not decide in time",
                    risk=risk, approval_id=str(approval_id),
                ),
                metadata={"approval_id": str(approval_id), "outcome": "timeout"},
            )
        self._cancel_expiry(approval_id)
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        details = metadata if isinstance(metadata, Mapping) else {}
        if self.events:
            await self.events.emit(
                "approval.resolved", approval_id=str(approval_id),
                status="expired", resolver="system:expiry",
            )
            await self.events.emit(
                "approval.expired",
                approval_id=str(approval_id),
                action=str(row["action"]),
                resource=row["resource"],
                risk=str(row["risk"]),
                command=str(details.get("command_preview") or ""),
                risk_reason=str(details.get("risk_reason") or ""),
                reason="approval expired: the action was not run",
            )
        return True

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

    @staticmethod
    def _grant_metadata(row: Any) -> dict[str, Any]:
        try:
            value = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError, KeyError):
            return {}
        return value if isinstance(value, Mapping) else {}

    async def _live_workspace_trust(self, resource: str | None) -> bool:
        """Whether shell/terminal work in this workspace is currently trusted."""
        if self.settings.security.approval_mode == "manual":
            return False
        rows = await db.fetchall(
            "SELECT resource FROM approval_grants WHERE action IN (?,?) "
            "AND scope='workspace' AND revoked_at IS NULL AND expires_at > ?",
            ("tool:shell", "tool:terminal", self._now_iso()),
        )
        return any(self._workspace_contains(str(item["resource"]), resource) for item in rows or [])

    async def _has_live_grant(
        self,
        action: str,
        resource: str | None,
        arguments_digest: str | None = None,
        risk_reason: str | None = None,
        *,
        command: str = "",
        task_id: str = "",
        conversation_id: str = "",
    ) -> bool:
        """True when an unexpired grant covers this call.

        Scopes, narrowest first:

        * exact shell arguments digest (the original "This command" button),
        * command pattern/prefix (``pytest *``, ``git *``), optionally standing,
        * tool + directory (path tools), optionally standing,
        * workspace trust for ordinary shell/terminal work,
        * one remembered approval kind per task/conversation.

        Hardline reasons never consult any of them: destructive, credential,
        privilege, outbound and self-modifying work keeps asking every time.
        ``manual`` mode ignores every grant.
        """
        if self.settings.security.approval_mode == "manual":
            return False
        # The hardline floor lives here as well as in the reviewer gate: a
        # pattern or a trust must never silently absorb a dangerous command.
        if risk_reason and not reason_is_trustable(risk_reason):
            return False
        try:
            now = self._now_iso()
            if action in ALWAYS_ARG_SCOPED_ACTIONS:
                digest = str(arguments_digest or "").strip()
                if digest:
                    exact = await db.fetchone(
                        "SELECT id FROM approval_grants WHERE action=? AND resource=? "
                        "AND revoked_at IS NULL AND expires_at > ? LIMIT 1",
                        (action, _grant_scope(action, digest), now),
                    )
                    if exact is not None:
                        return True
                text = str(command or "").strip()
                rows = await db.fetchall(
                    "SELECT action,resource,metadata,conversation_id FROM approval_grants "
                    "WHERE action IN (?,?) AND scope IN (?,?,?) "
                    "AND revoked_at IS NULL AND expires_at > ?",
                    ("tool:shell", "tool:terminal", _GRANT_SCOPE_PATTERN,
                     _GRANT_SCOPE_STANDING, _GRANT_SCOPE_CONVERSATION, now),
                )
                for item in rows or []:
                    subject = str(item["conversation_id"] or "")
                    if subject and subject not in {task_id, conversation_id}:
                        continue
                    meta = self._grant_metadata(item)
                    if meta.get("kind") == "directory":
                        continue
                    pattern = " ".join(str(meta.get("pattern") or item["resource"] or "").split())
                    if not pattern or pattern in {"*", "**"} or len(pattern) > _MAX_PATTERN_CHARS:
                        continue
                    boundary = str(meta.get("workspace") or "")
                    if boundary and not self._workspace_contains(boundary, resource):
                        continue
                    if text and _pattern_matches(text, pattern):
                        return True
                return await self._live_workspace_trust(resource)
            if resource is None or str(resource).strip() == "":
                return False
            row = await db.fetchone(
                "SELECT id FROM approval_grants WHERE action=? AND resource IS ? "
                "AND revoked_at IS NULL AND expires_at > ? LIMIT 1",
                (action, resource, now),
            )
            if row is not None:
                return True
            rows = await db.fetchall(
                "SELECT resource,metadata,conversation_id FROM approval_grants "
                "WHERE action=? AND scope IN (?,?,?) "
                "AND revoked_at IS NULL AND expires_at > ?",
                (action, _GRANT_SCOPE_DIRECTORY, _GRANT_SCOPE_STANDING,
                 _GRANT_SCOPE_CONVERSATION, now),
            )
            for item in rows or []:
                subject = str(item["conversation_id"] or "")
                if subject and subject not in {task_id, conversation_id}:
                    continue
                meta = self._grant_metadata(item)
                if meta.get("kind") == "pattern":
                    continue
                directory = str(meta.get("directory") or item["resource"] or "")
                if directory and self._workspace_contains(directory, resource):
                    return True
            return False
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
        other shell command. A grant that names neither is refused. In a
        trusted workspace the lifetime is extended to the configured trust
        window instead of the short 8-hour session default.
        """
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
        lifetime = max(60, int(seconds))
        if await self._live_workspace_trust(resource):
            lifetime = max(lifetime, int(
                float(getattr(self.settings.security, "workspace_trust_days", 30) or 30) * 86400
            ))
        grant_id = new_id("grn")
        created = self._now_iso()
        expires = (datetime.fromisoformat(created) + timedelta(seconds=lifetime)).isoformat()
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
        created = self._now_iso()
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

    async def grant_command_pattern(
        self,
        pattern: str,
        *,
        granted_by: str,
        workspace: str | None = None,
        action: str = "tool:shell",
        seconds: int | None = None,
        standing: bool = False,
    ) -> str:
        """Authorize a command family (``pytest *``, ``git *``) in one workspace.

        Hardline reasons are still checked on every call, so a broad git grant
        cannot absorb a force push. ``standing=True`` creates a first-class
        authorisation with no expiry; every grant remains revocable by id
        through :meth:`revoke_grant`.
        """
        if str(action) not in ALWAYS_ARG_SCOPED_ACTIONS:
            raise ValueError("a command pattern applies to shell/terminal only")
        value = _normalise_pattern(pattern)
        boundary = str(
            Path(str(workspace or self.settings.root)).expanduser().resolve()
        )
        lifetime = int(seconds if seconds is not None else SESSION_GRANT_SECONDS)
        if not standing and await self._live_workspace_trust(workspace):
            lifetime = max(lifetime, int(
                float(getattr(self.settings.security, "workspace_trust_days", 30) or 30) * 86400
            ))
        created = self._now()
        expires = (
            _STANDING_EXPIRY if standing
            else (created + timedelta(seconds=max(60, lifetime))).isoformat()
        )
        grant_id = new_id("grn")
        scope = _GRANT_SCOPE_STANDING if standing else _GRANT_SCOPE_PATTERN
        metadata = json.dumps({
            "kind": "pattern", "pattern": value, "workspace": boundary,
        }, ensure_ascii=False)
        await db.execute(
            "INSERT INTO approval_grants(id,created_at,expires_at,action,resource,"
            "granted_by,scope,metadata) VALUES(?,?,?,?,?,?,?,?)",
            (grant_id, created.isoformat(), expires, str(action), value,
             str(granted_by), scope, metadata),
        )
        if self.events:
            await self.events.emit(
                "approval.granted", grant_id=grant_id, action=str(action),
                resource=value, granted_by=str(granted_by), scope=scope,
            )
        return grant_id

    async def grant_directory(
        self,
        action: str,
        directory: str | None,
        *,
        granted_by: str,
        seconds: int | None = None,
        standing: bool = False,
    ) -> str:
        """Authorize one tool for every resource inside a directory.

        Unlike an exact-resource session grant this covers future files in the
        same tree. It never crosses the directory boundary and never waives a
        path rule. Standing directory grants are revocable like any other.
        """
        action_name = str(action or "").strip()
        if not action_name:
            raise ValueError("a directory grant needs a tool action")
        if action_name in ALWAYS_ARG_SCOPED_ACTIONS:
            raise ValueError("use a command pattern for shell/terminal grants")
        if directory is None or not str(directory).strip():
            raise ValueError("a directory grant needs a directory to scope it")
        boundary = str(Path(str(directory)).expanduser().resolve())
        lifetime = int(seconds if seconds is not None else SESSION_GRANT_SECONDS)
        if not standing and await self._live_workspace_trust(boundary):
            lifetime = max(lifetime, int(
                float(getattr(self.settings.security, "workspace_trust_days", 30) or 30) * 86400
            ))
        created = self._now()
        expires = (
            _STANDING_EXPIRY if standing
            else (created + timedelta(seconds=max(60, lifetime))).isoformat()
        )
        grant_id = new_id("grn")
        scope = _GRANT_SCOPE_STANDING if standing else _GRANT_SCOPE_DIRECTORY
        metadata = json.dumps({"kind": "directory", "directory": boundary}, ensure_ascii=False)
        await db.execute(
            "INSERT INTO approval_grants(id,created_at,expires_at,action,resource,"
            "granted_by,scope,metadata) VALUES(?,?,?,?,?,?,?,?)",
            (grant_id, created.isoformat(), expires, action_name, boundary,
             str(granted_by), scope, metadata),
        )
        if self.events:
            await self.events.emit(
                "approval.granted", grant_id=grant_id, action=action_name,
                resource=boundary, granted_by=str(granted_by), scope=scope,
            )
        return grant_id

    async def grant_standing(
        self,
        kind: str,
        value: str,
        *,
        granted_by: str,
        action: str | None = None,
        workspace: str | None = None,
    ) -> str:
        """Grant a persistent authorisation: ``pattern`` or ``directory``.

        The operator-facing entry point for a standing allowance. It stores a
        revocable grant row with no expiry; :meth:`revoke_grant` withdraws it.
        """
        mode = str(kind or "").strip().lower()
        if mode == "pattern":
            return await self.grant_command_pattern(
                value, granted_by=granted_by, workspace=workspace,
                action=str(action or "tool:shell"), standing=True,
            )
        if mode == "directory":
            return await self.grant_directory(
                str(action or ""), value, granted_by=granted_by, standing=True,
            )
        raise ValueError("a standing grant must be a 'pattern' or 'directory'")

    async def list_grants(self, *, include_expired: bool = False) -> list[dict[str, Any]]:
        """List live approval grants (trust, patterns and standing grants too)."""
        sql = (
            "SELECT id,created_at,expires_at,action,resource,granted_by,scope,"
            "metadata,conversation_id FROM approval_grants"
        )
        params: tuple[Any, ...] = ()
        if not include_expired:
            sql += " WHERE expires_at > ? AND revoked_at IS NULL"
            params = (self._now_iso(),)
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

    # -- durable schedule consent cards -----------------------------------
    async def _schedule_row(self, approval_id: str) -> Any:
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None:
            raise KeyError(f"unknown approval: {approval_id}")
        if not is_schedule_kind(row["kind"]):
            raise ValueError(f"approval {approval_id} is not a schedule card")
        return row

    async def create_schedule_card(
        self,
        *,
        kind: str,
        body: str,
        proposal_digest: str,
        proposal_id: str | None = None,
        schedule_id: str | None = None,
        config_revision: int | None = None,
        card_revision: int = 1,
        title: str | None = None,
        choices: Iterable[str] | None = None,
        deny_only: bool = False,
        conversation_id: str | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        origin: str | None = None,
        requested_by: str | None = None,
        actor: str | None = None,
        resource: str | None = None,
        proposed_at: str | None = None,
        expires_at: str | None = None,
        shown_first_run: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Persist one pending schedule consent card.

        No schedule row is written here. The only write is the durable
        approval row (plus its audit entry); Phase 5 commits the proposal after
        a human chooses, inside :meth:`resolve_schedule_card`'s transaction.
        """
        from .approval_cards import ApprovalCard  # local import: keep module load light

        if not is_schedule_kind(kind):
            raise ValueError(f"not a schedule card kind: {kind!r}")
        text = str(body or "")
        if not text.strip():
            raise ValueError("a schedule card needs a literal body")
        if "\x00" in text:
            raise ValueError("a schedule card body cannot contain a NUL byte")
        digest = str(proposal_digest or "").strip()
        if not digest:
            raise ValueError("a schedule card needs a proposal digest")
        config = kind_config(kind)
        canonical = [str(item).strip().lower() for item in (choices or config.choices)]
        if deny_only:
            canonical = ["deny"]
        for choice in canonical:
            if normalize_choice(choice, kind) is None:
                raise ValueError(f"invalid schedule card choice: {choice!r}")
        labels = choice_labels_for(kind, canonical)
        now = self._now()
        created = proposed_at or now.isoformat()
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings.security, "approval_timeout_seconds", 900)
        )
        expiry = now + timedelta(seconds=max(1.0, timeout))
        bounded_first = parse_instant(shown_first_run)
        first_text = bounded_first.isoformat() if bounded_first is not None else None
        if bounded_first is not None:
            expiry = min(expiry, bounded_first)
        explicit_expiry = parse_instant(expires_at)
        if explicit_expiry is not None:
            expiry = min(expiry, explicit_expiry)
        approval_id = new_id("apr")
        requester = str(requested_by or actor or "schedule").strip() or "schedule"
        card = ApprovalCard(
            protocol_version=APPROVAL_CARD_PROTOCOL_VERSION,
            approval_id=approval_id,
            proposal_id=proposal_id,
            kind=kind,
            status="pending",
            title=str(title or config.header),
            body=text,
            choices=canonical,
            choice_labels=labels,
            deny_only=bool(deny_only),
            card_revision=max(1, int(card_revision)),
            proposal_digest=digest,
            schedule_id=schedule_id,
            config_revision=config_revision,
            proposed_at=created,
            deadline=deadline_text(kind, timeout),
            expires_at=expiry.isoformat(),
            shown_first_run=first_text,
            conversation_id=conversation_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            origin=origin,
            requested_by=requester,
        )
        metadata = {
            "proposal_id": proposal_id,
            "proposal_digest": digest,
            "schedule_id": schedule_id,
            "config_revision": config_revision,
            "deny_only": bool(deny_only),
            "origin": origin,
            "conversation_id": conversation_id,
            "run_id": run_id,
            "tool_call_id": tool_call_id,
        }
        await db.execute(
            "INSERT INTO approvals(id,created_at,status,actor,action,resource,risk,"
            "reason,requested_by,metadata,kind,protocol_version,proposal_id,digest,"
            "choices,expires_at,origin,conversation_id,run_id,tool_call_id,"
            "card_revision,presented_surface,deny_reason,card) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                approval_id, created, "pending", requester,
                _SCHEDULE_ACTIONS[kind], resource or schedule_id or requester,
                RiskLevel.HIGH.value, "schedule consent required", requester,
                json.dumps(metadata, ensure_ascii=False),
                kind, APPROVAL_CARD_PROTOCOL_VERSION, proposal_id, digest,
                json.dumps(canonical, ensure_ascii=False), expiry.isoformat(),
                origin, conversation_id, run_id, tool_call_id,
                max(1, int(card_revision)), None, None,
                json.dumps(card.model_dump(mode="json"), ensure_ascii=False),
            ),
        )
        await self._audit(
            requester, "schedule.consent.request", resource or schedule_id,
            RiskLevel.HIGH,
            Decision(
                allowed=False, reason="schedule consent card created",
                risk=RiskLevel.HIGH, requires_approval=True,
                approval_id=approval_id,
            ),
            metadata={"proposal_id": proposal_id, "digest": digest, "kind": kind},
        )
        if self.events:
            await self.events.emit("approval.requested", **card.event_payload())
        return card

    async def get_schedule_card(self, approval_id: str) -> Any | None:
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None or not is_schedule_kind(row["kind"]):
            return None
        return card_from_row(row)

    async def get_approval_card(self, approval_id: str) -> Any | None:
        """Project any approval kind (schedule or legacy command)."""
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None:
            return None
        return card_from_row(row)

    async def schedule_cards(
        self, *, status: str | None = "pending", include_command: bool = False
    ) -> list[Any]:
        clauses: list[str] = []
        params: list[Any] = []
        if include_command:
            clauses.append("(kind IN (?,?) OR kind IS NULL OR kind='' OR kind='command')")
            params += [KIND_SCHEDULE_CREATE, KIND_SCHEDULE_EDIT]
        else:
            clauses.append("kind IN (?,?)")
            params += [KIND_SCHEDULE_CREATE, KIND_SCHEDULE_EDIT]
        if status and str(status) != "all":
            clauses.append("status=?")
            params.append(str(status))
        rows = await db.fetchall(
            "SELECT * FROM approvals WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at, id",
            tuple(params),
        )
        cards: list[Any] = []
        for row in rows or []:
            try:
                cards.append(card_from_row(row))
            except ValueError:
                continue
        return cards

    async def present_schedule_card(
        self,
        approval_id: str,
        *,
        surface: str,
        capabilities: Any = None,
    ) -> Any:
        """Record that the full card reached a capable surface.

        ``shown_at`` is set once, on the first real presentation. Resolution
        before this point is refused, so a tap cannot decide a card no human
        could have seen. An incapable surface may not present a card at all.
        """
        row = await self._schedule_row(approval_id)
        if str(row["status"]) != "pending":
            raise ValueError(f"approval already {row['status']}")
        caps = capabilities if capabilities is not None else surface_capabilities(surface)
        if not caps.can_present_card or not caps.full_text:
            raise PermissionError(
                f"surface {surface!r} cannot present a full approval card"
            )
        card = card_from_row(row)
        if not card_fits_surface(card, surface):
            raise PermissionError(
                f"approval card does not fit the {surface!r} presentation budget"
            )
        presenter = current_card_presenter()
        if presenter is not None:
            # Split-gateway transport: do not record shown_at until the process
            # holding the adapter confirms the full card reached the surface.
            try:
                delivered = await presenter(approval_id, str(surface))
            except Exception as exc:  # noqa: BLE001 - no transport, no card
                raise PermissionError(
                    f"card presentation failed on {surface!r}: {type(exc).__name__}"
                ) from exc
            if not delivered:
                raise PermissionError(f"card was not delivered on {surface!r}")
        now = self._now_iso()
        # Two presenters can race on one card. Claim shown_at with a CAS on the
        # empty value instead of trusting the earlier read; a retry sees the
        # winner's row and only updates the surface. This is idempotent: a card
        # already shown keeps its original timestamp.
        for _attempt in range(3):
            current = await self._schedule_row(approval_id)
            if str(current["status"]) != "pending":
                break
            if current["shown_at"]:
                await db.execute(
                    "UPDATE approvals SET presented_surface=? WHERE id=?",
                    (str(surface), approval_id),
                )
                break
            cursor = await db.execute(
                "UPDATE approvals SET shown_at=?, presented_surface=? "
                "WHERE id=? AND status='pending' AND (shown_at IS NULL OR shown_at='')",
                (now, str(surface), approval_id),
            )
            if int(getattr(cursor, "rowcount", 0) or 0) > 0:
                break
        card = await self.get_schedule_card(approval_id)
        if self.events and card is not None:
            await self.events.emit(
                "approval.presented", surface=str(surface), **card.event_payload()
            )
        return card

    async def acknowledge_schedule_card(self, approval_id: str, *, surface: str) -> Any:
        """Record that a client received and displayed the card.

        An acknowledgement is not a decision. It doubles as the web client's
        presentation event: it records ``shown_at`` when the card was not
        already presented, so a subsequent resolve is bound to a card that
        reached a capable surface. A notification never calls this.
        """
        row = await self._schedule_row(approval_id)
        if str(row["status"]) != "pending":
            raise ValueError(f"approval already {row['status']}")
        now = self._now_iso()
        if not row["shown_at"]:
            await db.execute(
                "UPDATE approvals SET shown_at=?, presented_surface=?, "
                "metadata=json_set(COALESCE(metadata,'{}'),"
                "'$.acknowledged_at',?, '$.acknowledged_surface',?) WHERE id=?",
                (now, str(surface), now, str(surface), approval_id),
            )
        else:
            await db.execute(
                "UPDATE approvals SET metadata=json_set(COALESCE(metadata,'{}'),"
                "'$.acknowledged_at',?, '$.acknowledged_surface',?) WHERE id=?",
                (now, str(surface), approval_id),
            )
        card = await self.get_schedule_card(approval_id)
        if self.events and card is not None:
            await self.events.emit(
                "approval.acknowledged", surface=str(surface), **card.event_payload()
            )
        return card

    async def wait_for_schedule_card(
        self, approval_id: str, *, timeout: float | None = None
    ) -> Any:
        """Wait (bounded, poll-based) until the card leaves ``pending``.

        A card past its own ``expires_at`` is marked expired first, so the
        terminal copy is durable even if the waiting turn is interrupted.
        """
        bounded = float(
            timeout
            if timeout is not None
            else getattr(self.settings.security, "approval_timeout_seconds", 900)
        )
        poll = max(
            0.05,
            float(getattr(self.settings.security, "approval_poll_seconds", 2.0)),
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, bounded)
        while True:
            card = await self.get_schedule_card(approval_id)
            if card is None:
                raise KeyError(f"unknown approval: {approval_id}")
            if card.status != "pending":
                return card
            if at_or_after(card.expires_at, self._now()):
                return await self._expire_schedule_card(approval_id)
            if loop.time() >= deadline:
                raise TimeoutError(f"approval {approval_id} is still pending")
            await asyncio.sleep(min(poll, max(0.01, deadline - loop.time())))

    async def _expire_schedule_card(
        self, approval_id: str, *, stale: bool = False,
        decline_override: str | None = None,
    ) -> Any:
        row = await self._schedule_row(approval_id)
        card = card_from_row(row)
        if card.status != "pending":
            return card
        config = kind_config(card.kind)
        outcome = config.stale_outcome if stale else config.timeout_outcome
        decline = decline_override or (
            DECLINE_MESSAGES["stale_create"]
            if stale and card.kind == KIND_SCHEDULE_CREATE
            else DECLINE_MESSAGES["stale_edit"]
            if stale
            else DECLINE_MESSAGES["timeout"]
        )
        now = self._now_iso()
        await db.execute(
            "UPDATE approvals SET status='expired', answered_at=?, "
            "metadata=json_set(COALESCE(metadata,'{}'),'$.resolved_by',?,"
            "'$.outcome',?,'$.decline_message',?) "
            "WHERE id=? AND status='pending'",
            (now, "system:expiry", outcome, decline, approval_id),
        )
        await self._settle_schedule_proposal(row, "expired")
        risk = RiskLevel(str(row["risk"]))
        await self._audit(
            "system:expiry", str(row["action"]), row["resource"], risk,
            Decision(
                allowed=False, reason=outcome, risk=risk, approval_id=approval_id,
            ),
            metadata={"approval_id": approval_id, "outcome": outcome},
        )
        expired = await self.get_schedule_card(approval_id)
        if self.events and expired is not None:
            await self.events.emit(
                "approval.resolved",
                **{
                    **expired.event_payload(),
                    "status": "expired",
                    "resolver": "system:expiry",
                },
            )
            await self.events.emit(
                "schedule.expired", approval_id=approval_id, kind=expired.kind,
                outcome=outcome,
            )
        return expired

    @staticmethod
    def _identity_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text.startswith("operator:"):
            text = text[len("operator:"):]
        return text

    def _assert_resolver_not_requester(self, card: Any, resolver: str) -> None:
        resolver_text = str(resolver or "").strip().lower()
        if not resolver_text:
            raise PermissionError("an approval decision needs a resolver identity")
        if resolver_text.startswith("chat:"):
            # A chat actor is the requesting principal, never the human gate.
            raise PermissionError("chat actors cannot resolve approvals")
        requester = str(getattr(card, "requested_by", "") or "").strip()
        if requester and self._identity_key(resolver) == self._identity_key(requester):
            raise PermissionError("the requester cannot resolve its own approval")

    async def resolve_schedule_card(
        self,
        approval_id: str,
        *,
        choice: str,
        resolver: str,
        confirm_proposal_digest: str | None,
        deny_reason: str | None = None,
        commit: Callable[..., Awaitable[Any]] | None = None,
    ) -> ScheduleResolution:
        """Atomically decide one schedule card.

        The decision, a create grant (when the choice grants one) and the
        optional schedule commit share one write transaction. Any failure —
        wrong choice, stale digest, expired/stale/shown state, commit error —
        leaves the approval pending and writes no grant or schedule.
        """
        committer = commit if commit is not None else self._schedule_committer
        async with db.transaction() as conn:
            cursor = await conn.execute(
                "SELECT * FROM approvals WHERE id=?", (approval_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                raise KeyError(f"unknown approval: {approval_id}")
            if not is_schedule_kind(row["kind"]):
                raise ValueError(f"approval {approval_id} is not a schedule card")
            protocol = int(row["protocol_version"] or 0)
            if protocol != APPROVAL_CARD_PROTOCOL_VERSION:
                raise ValueError(
                    f"unsupported approval card protocol: {protocol or 'legacy'}"
                )
            status = str(row["status"] or "")
            if status != "pending":
                raise ValueError(f"approval already {status}")
            card = card_from_row(row)
            self._assert_resolver_not_requester(card, resolver)
            if not proposal_digest_matches(row, confirm_proposal_digest):
                raise ValueError("proposal digest mismatch")
            if card.proposal_digest and str(row["digest"]) != str(card.proposal_digest):
                raise ValueError("stored card digest does not match the approval row")
            if not row["shown_at"]:
                raise PermissionError("approval card has not been presented")
            now = self._now()
            first_run = parse_instant(card.shown_first_run)
            expires = parse_instant(row["expires_at"])
            # A passed first run is the honest reason when it is the effective
            # deadline; a shorter timeout still reports as a timeout.
            if (
                first_run is not None
                and now >= first_run
                and (expires is None or first_run <= expires)
            ):
                expired = await self._expire_schedule_card(approval_id, stale=True)
                decline = (
                    DECLINE_MESSAGES["stale_create"]
                    if card.kind == KIND_SCHEDULE_CREATE
                    else DECLINE_MESSAGES["stale_edit"]
                )
                return ScheduleResolution(
                    approval_id=approval_id, kind=expired.kind,
                    status="expired", allowed=False,
                    reason=str(expired.outcome or "stale"),
                    outcome=expired.outcome, decline_message=decline,
                )
            if at_or_after(row["expires_at"], now):
                expired = await self._expire_schedule_card(approval_id)
                return ScheduleResolution(
                    approval_id=approval_id, kind=expired.kind,
                    status="expired", allowed=False,
                    reason=str(expired.outcome or "expired"),
                    outcome=expired.outcome,
                    decline_message=DECLINE_MESSAGES["timeout"],
                )
            canonical = normalize_choice(choice, card.kind)
            if canonical is None:
                raise ValueError(
                    f"invalid choice {str(choice or '')!r} for a {card.kind} card"
                )
            if card.deny_only and canonical != "deny":
                raise ValueError("this card is deny-only")
            if canonical not in card.choices:
                raise ValueError(f"choice {canonical!r} is not offered on this card")
            config = kind_config(card.kind)
            now_text = now.isoformat()
            if canonical == "deny":
                reason = _bounded_deny_reason(deny_reason) or "denied by operator"
                await conn.execute(
                    "UPDATE approvals SET status='denied', answered_at=?, "
                    "deny_reason=?, metadata=json_set(COALESCE(metadata,'{}'),"
                    "'$.resolved_by',?,'$.choice',?,'$.outcome',?,"
                    "'$.decline_message',?) WHERE id=? AND status='pending'",
                    (
                        now_text, reason, str(resolver), canonical,
                        config.deny_outcome, DECLINE_MESSAGES["denied"], approval_id,
                    ),
                )
                await self._audit(
                    str(resolver), str(row["action"]), row["resource"],
                    RiskLevel(str(row["risk"])),
                    Decision(
                        allowed=False,
                        reason=f"schedule consent denied: {reason}",
                        risk=RiskLevel(str(row["risk"])), approval_id=approval_id,
                    ),
                    metadata={
                        "approval_id": approval_id, "resolver": str(resolver),
                        "choice": canonical, "kind": card.kind,
                    },
                )
                await self._settle_schedule_proposal(row, "denied")
                resolved = await self.get_schedule_card(approval_id)
                if self.events and resolved is not None:
                    await self.events.emit(
                        "approval.resolved",
                        **{
                            **resolved.event_payload(),
                            "status": "denied",
                            "resolver": str(resolver),
                            "choice": canonical,
                        },
                    )
                    await self.events.emit(
                        "schedule.denied", approval_id=approval_id, kind=card.kind,
                        choice=canonical,
                    )
                return ScheduleResolution(
                    approval_id=approval_id, kind=card.kind, status="denied",
                    allowed=False, reason=f"schedule consent denied: {reason}",
                    choice=canonical, outcome=config.deny_outcome,
                    decline_message=DECLINE_MESSAGES["denied"],
                )

            grant_id: str | None = None
            if card.kind == KIND_SCHEDULE_CREATE and canonical in {"session", "always"}:
                grant_id = await self._insert_schedule_create_grant(
                    card, canonical=canonical, granted_by=str(resolver)
                )
            commit_result: Any = None
            if committer is not None:
                # Must run inside this transaction; a failure rolls the
                # approval update, grant and audit back together.
                try:
                    commit_result = await committer(
                        card=card, choice=canonical, resolver=str(resolver)
                    )
                except ValueError as exc:
                    # A CAS loss/removal discovered at commit time is an expired
                    # edit, never an exception with a pending card. A row that
                    # is now paused/disabled gets its own honest decline instead
                    # of being blamed on a passed run time.
                    code = getattr(exc, "code", "")
                    if (
                        card.kind != KIND_SCHEDULE_EDIT
                        or code not in {"stale_update", "unknown_schedule", "disabled_update"}
                    ):
                        raise
                    decline = (
                        DECLINE_MESSAGES["disabled_edit"]
                        if code == "disabled_update"
                        else DECLINE_MESSAGES["stale_edit"]
                    )
                    expired = await self._expire_schedule_card(
                        approval_id, stale=True, decline_override=decline
                    )
                    return ScheduleResolution(
                        approval_id=approval_id, kind=expired.kind,
                        status="expired", allowed=False,
                        reason=str(exc) or str(expired.outcome or "stale"),
                        outcome=expired.outcome,
                        decline_message=decline,
                    )
            committed_schedule = (
                commit_result.get("schedule_id")
                if isinstance(commit_result, Mapping)
                else None
            ) or card.schedule_id
            await conn.execute(
                "UPDATE approvals SET status='approved', answered_at=?, "
                "metadata=json_set(COALESCE(metadata,'{}'),'$.resolved_by',?,"
                "'$.choice',?,'$.outcome',?,'$.grant_id',?,"
                "'$.schedule_id',?) WHERE id=? AND status='pending'",
                (
                    now_text, str(resolver), canonical, config.allow_outcome,
                    grant_id, committed_schedule, approval_id,
                ),
            )
            await self._audit(
                str(resolver), str(row["action"]), row["resource"],
                RiskLevel(str(row["risk"])),
                Decision(
                    allowed=True, reason=f"schedule consent approved: {canonical}",
                    risk=RiskLevel(str(row["risk"])), approval_id=approval_id,
                ),
                metadata={
                    "approval_id": approval_id, "resolver": str(resolver),
                    "choice": canonical, "kind": card.kind,
                    "grant_id": grant_id, "schedule_id": committed_schedule,
                },
            )
            resolved = await self.get_schedule_card(approval_id)
            if self.events and resolved is not None:
                await self.events.emit(
                    "approval.resolved",
                    **{
                        **resolved.event_payload(),
                        "status": "approved",
                        "resolver": str(resolver),
                        "choice": canonical,
                        "grant_id": grant_id,
                    },
                )
                await self.events.emit(
                    "schedule.approved", approval_id=approval_id, kind=card.kind,
                    choice=canonical, schedule_id=committed_schedule,
                )
            return ScheduleResolution(
                approval_id=approval_id, kind=card.kind, status="approved",
                allowed=True, reason=f"schedule consent approved: {canonical}",
                choice=canonical, outcome=config.allow_outcome,
                grant_id=grant_id, schedule_id=committed_schedule,
                commit_result=commit_result,
            )

    async def _insert_schedule_create_grant(
        self, card: Any, *, canonical: str, granted_by: str
    ) -> str:
        """Insert/renew one conversation- or profile-scoped create grant.

        Edit cards never reach this method: edits have no grant. The action is
        ``schedule:create``, so this row can never satisfy an ordinary tool,
        workspace trust or lifecycle check.
        """
        principal = str(getattr(card, "requested_by", "") or "").strip()
        if not principal:
            raise ValueError("a schedule create grant needs a requesting principal")
        if canonical == "session":
            conversation_id = str(getattr(card, "conversation_id", "") or "").strip()
            if not conversation_id:
                raise ValueError(
                    "a session schedule grant needs the conversation it covers"
                )
            subject_type, subject_id = "conversation", conversation_id
            seconds = SCHEDULE_SESSION_GRANT_SECONDS
        else:
            days = int(
                getattr(
                    self.settings.security,
                    "schedule_always_grant_days",
                    SCHEDULE_ALWAYS_GRANT_DAYS,
                )
                or SCHEDULE_ALWAYS_GRANT_DAYS
            )
            days = max(1, min(_MAX_SCHEDULE_GRANT_DAYS, days))
            subject_type, subject_id = "profile", principal
            seconds = days * 86400
        created = self._now()
        expires = (created + timedelta(seconds=seconds)).isoformat()
        existing = await db.fetchone(
            "SELECT id,expires_at FROM approval_grants WHERE principal=? "
            "AND subject_type=? AND subject_id=? AND action=? "
            "AND revoked_at IS NULL LIMIT 1",
            (principal, subject_type, subject_id, SCHEDULE_CREATE_ACTION),
        )
        if existing is not None:
            grant_id = str(existing["id"])
            if str(existing["expires_at"]) < expires:
                await db.execute(
                    "UPDATE approval_grants SET expires_at=? WHERE id=?",
                    (expires, grant_id),
                )
        else:
            grant_id = new_id("grn")
            resource = f"{subject_type}:{subject_id}"
            await db.execute(
                "INSERT INTO approval_grants(id,created_at,expires_at,action,"
                "resource,granted_by,scope,metadata,principal,subject_type,"
                "subject_id,conversation_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    grant_id, created.isoformat(), expires, SCHEDULE_CREATE_ACTION,
                    resource, str(granted_by), subject_type,
                    json.dumps(
                        {"kind": KIND_SCHEDULE_CREATE, "choice": canonical},
                        ensure_ascii=False,
                    ),
                    principal, subject_type, subject_id,
                    str(getattr(card, "conversation_id", "") or "") or None,
                ),
            )
        if self.events:
            await self.events.emit(
                "approval.granted", grant_id=grant_id, action=SCHEDULE_CREATE_ACTION,
                resource=f"{subject_type}:{subject_id}",
                granted_by=str(granted_by), scope=subject_type,
                principal=principal, subject_id=subject_id,
            )
        return grant_id

    async def has_live_schedule_create_grant(
        self,
        *,
        principal: str,
        conversation_id: str | None = None,
        profile: str | None = None,
    ) -> bool:
        """True when a live create grant covers this principal/conversation.

        Session grants are keyed by conversation; always grants by profile.
        Manual approval mode ignores grants, like ordinary approvals do.
        """
        if str(getattr(self.settings.security, "approval_mode", "smart")) == "manual":
            return False
        principal_text = str(principal or "").strip()
        if not principal_text:
            return False
        profile_text = str(profile or principal_text).strip()
        try:
            row = await db.fetchone(
                "SELECT id FROM approval_grants WHERE action=? "
                "AND principal=? AND revoked_at IS NULL AND expires_at > ? "
                "AND ((subject_type='conversation' AND subject_id=?) "
                "OR (subject_type='profile' AND subject_id=?)) LIMIT 1",
                (
                    SCHEDULE_CREATE_ACTION, principal_text, self._now_iso(),
                    str(conversation_id or ""), profile_text,
                ),
            )
        except Exception:  # noqa: BLE001 - a grant lookup must never crash a decision
            return False
        return row is not None

    async def revoke_schedule_grant(
        self, grant_id: str, *, actor: str = "operator"
    ) -> bool:
        """Revoke one schedule create grant; the row and audit are retained."""
        now = self._now_iso()
        cursor = await db.execute(
            "UPDATE approval_grants SET revoked_at=? WHERE id=? "
            "AND action=? AND revoked_at IS NULL",
            (now, grant_id, SCHEDULE_CREATE_ACTION),
        )
        revoked = bool(getattr(cursor, "rowcount", 0))
        if revoked:
            await self._audit(
                str(actor), "approval.grant.revoke", grant_id, RiskLevel.LOW,
                Decision(
                    allowed=True, reason="schedule create grant revoked",
                    risk=RiskLevel.LOW,
                ),
                metadata={"grant_id": grant_id, "action": SCHEDULE_CREATE_ACTION},
            )
            if self.events:
                await self.events.emit(
                    "approval.grant.revoke", grant_id=grant_id,
                    action=SCHEDULE_CREATE_ACTION, revoked_by=str(actor),
                )
        return revoked

    async def _remember_approval(self, row: Any, resolver: str) -> None:
        """Remember one approved kind of work for this task/conversation.

        The memory is a narrow, expiring grant keyed to the task id the executor
        passes on every call. A shell approval becomes a program-prefix pattern
        (``pytest *``), a path approval becomes a directory grant. Hardline
        approvals are never remembered, so "approve once" for a destructive
        command never turns into standing authority by accident.
        """
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, Mapping):
            return
        task_id = str(
            metadata.get("task_id") or metadata.get("conversation_id") or ""
        ).strip()
        if not task_id:
            return
        if not reason_is_trustable(str(metadata.get("risk_reason") or "")):
            return
        action = str(row["action"] or "")
        resource = str(row["resource"] or "")
        if action in ALWAYS_ARG_SCOPED_ACTIONS:
            pattern = _command_pattern(metadata.get("command_preview") or "")
            if pattern is None:
                return
            stored_resource, meta = pattern, {
                "kind": "pattern", "pattern": pattern,
                "workspace": resource or str(self.settings.root),
            }
        else:
            if not resource:
                return
            try:
                directory = str(Path(resource).expanduser().resolve().parent)
            except (OSError, RuntimeError, ValueError):
                return
            stored_resource, meta = directory, {
                "kind": "directory", "directory": directory,
            }
        seconds = int(
            getattr(self.settings.security, "conversation_grant_seconds", SESSION_GRANT_SECONDS)
            or SESSION_GRANT_SECONDS
        )
        created = self._now()
        grant_id = new_id("grn")
        await db.execute(
            "INSERT INTO approval_grants(id,created_at,expires_at,action,resource,"
            "granted_by,scope,metadata,conversation_id) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                grant_id, created.isoformat(),
                (created + timedelta(seconds=max(60, seconds))).isoformat(),
                action, stored_resource, str(resolver), _GRANT_SCOPE_CONVERSATION,
                json.dumps(meta, ensure_ascii=False), task_id,
            ),
        )
        if self.events:
            await self.events.emit(
                "approval.granted", grant_id=grant_id, action=action,
                resource=stored_resource, granted_by=str(resolver),
                scope=_GRANT_SCOPE_CONVERSATION, conversation_id=task_id,
            )

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
            remember_row = dict(row)
            status = "approved" if approved else "denied"
            update = await conn.execute(
                "UPDATE approvals SET status=?, resolved_at=?, "
                "metadata=json_set(COALESCE(metadata,'{}'),'$.resolved_by',?) "
                "WHERE id=? AND status='pending'",
                (status, self._now_iso(), resolver_name, approval_id),
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

        self._cancel_expiry(approval_id)
        if approved:
            # The row is still available after the transaction closed; the
            # conversation grant is written separately so a failure to remember
            # never rolls back the human's decision.
            try:
                await self._remember_approval(remember_row, resolver_name)
            except Exception:  # noqa: BLE001 - memory is an optimisation
                _LOG.warning("could not remember approval kind", extra={
                    "approval_id": approval_id})
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

        # self._now_iso() is an ISO string, so the cutoff stays the same shape and
        # the SQL comparison stays lexicographic-safe.
        cutoff = (self._now() - timedelta(hours=older_than_hours)).isoformat()
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
                    (self._now_iso(), resolver_name, row["id"]),
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
            (self._now_iso(), actor, action, resource, risk.value, "allowed" if decision.allowed else ("pending" if decision.requires_approval else "denied"), decision.reason, json.dumps(metadata or {}, default=str)),
        )

    async def record_audit(
        self, actor: str, action: str, resource: str | None, risk: RiskLevel,
        decision: Decision, metadata: dict[str, Any] | None = None,
    ) -> None:
        """Public seam for the schedule authority's direct operator writes.

        ScheduleService is the only writer of a direct/operator schedule
        mutation; it still records every such write through this one audit
        helper so the decision log has a single shape and timestamp source.
        """
        await self._audit(actor, action, resource, risk, decision, metadata)

    async def _settle_schedule_proposal(self, row: Any, status: str) -> None:
        """Release a proposal reservation after a deny/timeout.

        A pending proposal holds an active schedule name through a partial
        unique index. When a human denies or lets the card expire, the proposal
        must leave ``pending`` in the same durable step or that name stays
        blocked forever.
        """
        keys = row.keys() if hasattr(row, "keys") else ()
        proposal_id = row["proposal_id"] if "proposal_id" in keys else None
        if not proposal_id:
            return
        await db.execute(
            "UPDATE schedule_proposals SET status=?, updated_at=? "
            "WHERE id=? AND status='pending'",
            (str(status), self._now_iso(), str(proposal_id)),
        )
