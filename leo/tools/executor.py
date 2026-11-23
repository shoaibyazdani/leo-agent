"""Permission-enforcing tool execution with audit, events and redaction."""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from .. import db
from ..config import Settings
from ..contracts import Decision, RiskLevel, ToolCall, ToolResult, new_id, utcnow
from ..events import Event, EventBus
from ..logging import get_logger
from ..security.permissions import ApprovalRequired, PermissionManagerImpl, arguments_digest
from ..security.risk_policy import assess_call
from .registry import ToolRegistration, ToolRegistry, _validate_value
from .shell_policy import preflight as shell_preflight

_LOG = get_logger("tools")

# Fallback deadline for a tool that declares nothing and has no configured
# value. Tool-specific and operator-configured timeouts always win over it.
DEFAULT_TOOL_TIMEOUT_SECONDS = 180.0
# Tools whose deadline comes from security.shell_timeout_seconds. Before this
# mapping existed the shell tool declared no timeout, so this 30s fallback
# capped every command no matter what the operator configured.
_CONFIGURED_TIMEOUTS: dict[str, str] = {
    "shell": "shell_timeout_seconds",
    "terminal": "shell_timeout_seconds",
}
_RESOURCE_KEYS = ("path", "cwd", "resource", "file", "directory", "root", "target")

# No-progress guard, tool side: after this many failures of the SAME class in
# one turn, emit one consolidated hint telling the model to stop retrying
# variants. This is advisory text only - no permission or quota decision is
# relaxed, and core/app.py is untouched.
POLICY_HINT_THRESHOLD = 3
_HINT_HISTORY_TTL_SECONDS = 300.0
_HINT_HISTORY_MAX_KEYS = 128
# An idle gap on the same conversation key means a new user turn began.
_TURN_IDLE_SECONDS = 60.0
_ERROR_FIELD_RE = re.compile(r"field '([^']+)'")
# A timeout on a binary payload is a tool-choice problem, not a reason to retry
# the same command with a bigger timeout. The 30s deadline itself is untouched.
_BINARY_TARGET_RE = re.compile(
    r"(?i)\.(?:db|sqlite3?|bin|dat|pyc|so|dylib|dll|exe|zip|gz|bz2|xz|tar|7z|"
    r"jar|class|wasm|pdf|png|jpe?g|gif|webp|ico|mp[34]|mov|avi|woff2?|ttf|otf)\b"
)
_TIMEOUT_BINARY_HINT = (
    "Binary or very large targets cannot be searched this way; switch to a structured "
    "tool (task_list/task_status for task data, search_files/read_file for text), "
    "narrow the path/pattern and cap output instead of retrying."
)
_TIMEOUT_GENERIC_HINT = (
    "Reduce the scope of the request or switch to a structured tool instead of repeating it."
)

_POLICY_HINTS: dict[str, str] = {
    "terminal.executable_not_allowed": (
        "Unlisted executables (including python/python3/node/sh) are never allowed; "
        "use read_file/list_files/search_files to inspect files or an allowlisted command."
    ),
    "terminal.metacharacters": (
        "Shell pipes, redirects and chaining are forbidden; run one allowlisted "
        "executable with separate arguments, or use the filesystem tools."
    ),
    "terminal.dangerous_argument": (
        "That executable flag can execute code or mutate state and is denied; "
        "use the read-only filesystem tools instead."
    ),
    "path.outside_roots": (
        "Paths outside the permitted roots are denied; use a path inside the workspace."
    ),
    "invalid_arguments": (
        "Each tool has its own field list: read the 'valid fields' and 'did you mean' "
        "names in the validation error instead of guessing another field."
    ),
    "unknown_tool": (
        "That tool is not registered; check the tool catalogue before calling again."
    ),
    "rate_limited": (
        "The call quota is exhausted for this window; state the blocker rather than "
        "retrying immediately."
    ),
    "timeout": (
        "The tool timed out; reduce the scope of the request or switch tools instead "
        "of repeating the same call."
    ),
    "permission": (
        "Permission was denied; request approval or choose a different tool rather "
        "than repeating the same action."
    ),
    "handler": (
        "The tool keeps reporting failure; state the blocker or switch approach "
        "instead of retrying the same call."
    ),
}


def _failure_class(decision: str, error: str) -> str:
    """Stable, coarse class used to detect a no-progress retry loop."""
    text = str(error or "")
    if decision == "invalid_arguments":
        fields = sorted({
            match.group(1).split(".")[0].split("[")[0]
            for match in _ERROR_FIELD_RE.finditer(text)
        })
        return "invalid_arguments:" + ",".join(fields)
    markers = (
        ("executable not allowed", "terminal.executable_not_allowed"),
        ("shell metacharacters", "terminal.metacharacters"),
        ("dangerous argument", "terminal.dangerous_argument"),
        ("git subcommand not allowed", "terminal.dangerous_argument"),
        ("unsafe git option", "terminal.dangerous_argument"),
        ("outside allowed roots", "path.outside_roots"),
    )
    for marker, label in markers:
        if marker in text:
            return label
    if decision == "unknown_tool":
        return "unknown_tool"
    if decision == "rate_limited":
        return "rate_limited"
    if decision == "timeout":
        return "timeout"
    if decision in ("denied", "permission_denied", "out_of_scope"):
        return "permission"
    return "handler"


def _iter_strings(value: Any) -> Iterator[str]:
    """Yield model-supplied strings without assuming a flat argument shape."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_strings(item)


class _QuotaExceeded(Exception):
    """Internal signal carrying durable quota status for audit/metadata."""

    def __init__(self, status: dict[str, Any]) -> None:
        self.status = status
        super().__init__("tool call limit exceeded")


def redact_value(value: Any, patterns: list[re.Pattern[str]]) -> Any:
    """Recursively apply redaction patterns to strings in a tool result."""
    if isinstance(value, str):
        text = value
        for pattern in patterns:
            text = pattern.sub("[REDACTED]", text)
        return text
    if isinstance(value, dict):
        return {redact_value(k, patterns): redact_value(v, patterns) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, patterns) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v, patterns) for v in value)
    if isinstance(value, set):
        return {redact_value(v, patterns) for v in value}
    return value


class ToolExecutorImpl:
    """Execute registered tools behind permissions, limits and redaction."""

    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionManagerImpl,
        events: EventBus,
        settings: Settings,
        *,
        timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        services: Mapping[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.permissions = permissions
        self.events = events
        self.settings = settings
        # The composition root owns this mutable mapping and fills it in as
        # engines are constructed. Keeping the same object means handlers that
        # capture ``services`` at registration time see the live instances.
        self.services: MutableMapping[str, Any] = (
            services if isinstance(services, MutableMapping) else dict(services or {})
        )
        self.timeout_seconds = timeout_seconds
        self._patterns = self._compile_patterns(settings.security.redact_patterns)
        # Per-turn retry counters for advisory hints only (never a policy gate).
        self._failure_history: dict[tuple[str, str], dict[str, Any]] = {}
        self._history_lock = threading.Lock()
        # Quotas live in SQLite (tool_quotas), not in this process: restarts and
        # multiple workers see the same counters. Only actually dispatched
        # calls are charged; invalid, denied and dry-run calls stay free.

    QUOTA_WINDOW_SECONDS = 60.0

    def list_specs(self) -> list[Any]:
        """Tool catalogue facade: ``services['tools']`` is a tools service."""
        return self.registry.list_specs()

    @staticmethod
    def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns or []:
            try:
                compiled.append(re.compile(pattern))
            except re.error:
                _LOG.warning("ignoring invalid redaction pattern", extra={"pattern": pattern})
        return compiled

    def _redact(self, value: Any) -> Any:
        return redact_value(value, self._patterns)

    @staticmethod
    def _task_key(call: ToolCall) -> str:
        explicit = getattr(call, "task_id", None)
        if isinstance(explicit, str) and explicit:
            return f"task:{explicit}"
        if call.correlation_id:
            return f"correlation:{call.correlation_id}"
        return "global"

    def _quota_key(self, call: ToolCall) -> str:
        """Per-task/correlation bucket; HTTP callers fall back to their actor.

        There is deliberately no shared 'global' bucket: one noisy caller (or
        a burst of denied calls) must never lock out every other client.
        """
        key = self._task_key(call)
        return key if key != "global" else f"actor:{call.actor or 'anonymous'}"

    @staticmethod
    def _bucket_matches(bucket: str, key: str) -> bool:
        """A legacy key matches an actor-scoped bucket with that suffix."""
        return bucket == key or bucket.endswith("|" + key)

    def _quota_bucket(self, call: ToolCall) -> str:
        """Actor-scoped bucket: same actor/task shares limits across processes."""
        actor = str(getattr(call, "actor", None) or "anonymous")
        base = self._quota_key(call)
        return f"actor:{actor}" if base == "global" else f"actor:{actor}|{base}"

    def _window(self) -> tuple[float, float]:
        seconds = max(0.001, float(self.QUOTA_WINDOW_SECONDS or 60.0))
        now = time.time()
        start = math.floor(now / seconds) * seconds
        return start, start + seconds

    @staticmethod
    def _connect_sync() -> sqlite3.Connection:
        conn = sqlite3.connect(str(db.get_db_path()), timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _quota_counts(self, key: str, window_start: float) -> int:
        try:
            conn = self._connect_sync()
        except (RuntimeError, sqlite3.Error):
            return 0
        try:
            rows = conn.execute(
                "SELECT bucket, count FROM tool_quotas WHERE window_start=?",
                (window_start,),
            ).fetchall()
        except sqlite3.Error:
            return 0
        finally:
            conn.close()
        return sum(
            int(count) for bucket, count in rows if self._bucket_matches(str(bucket), key)
        )

    def _status(self, key: str, limit: int, used: int, window_seconds: float,
                reset_at: float) -> dict[str, Any]:
        return {
            "key": key, "limit": limit, "used": used,
            "remaining": max(0, limit - used),
            "reset_at": round(reset_at, 3),
            "window_seconds": window_seconds,
            "durable": True,
        }

    def quota_status(self, key: str, limit: int | None = None) -> dict[str, Any]:
        """Return durable limit/usage/reset metadata for one bucket."""
        start, reset_at = self._window()
        cap = int(limit) if limit is not None else self._limit_for_hint()
        cap = max(1, min(cap, self._ceiling()))
        used = self._quota_counts(key, start)
        return self._status(key, cap, used, float(self.QUOTA_WINDOW_SECONDS), reset_at)

    def reset_quota(self, key: str | None = None) -> int:
        """Clear one bucket (or all durable buckets); return charges removed."""
        try:
            conn = self._connect_sync()
        except (RuntimeError, sqlite3.Error):
            return 0
        try:
            rows = conn.execute("SELECT bucket, window_start, count FROM tool_quotas").fetchall()
            targets = [
                (bucket, window, count) for bucket, window, count in rows
                if key is None or self._bucket_matches(str(bucket), key)
            ]
            conn.executemany(
                "DELETE FROM tool_quotas WHERE bucket=? AND window_start=?",
                [(bucket, window) for bucket, window, _ in targets],
            )
            conn.commit()
            return sum(int(count) for _, _, count in targets)
        except sqlite3.Error:
            _LOG.exception("failed to reset durable quota", extra={"key": key})
            return 0
        finally:
            conn.close()

    async def _charge_quota(self, bucket: str, limit: int) -> dict[str, Any] | None:
        """Atomically increment one durable bucket; return status when denied."""
        seconds = max(0.001, float(self.QUOTA_WINDOW_SECONDS or 60.0))
        start, reset_at = self._window()
        status = self._status(bucket, limit, limit, seconds, reset_at)
        try:
            async with db.transaction() as conn:
                await conn.execute("DELETE FROM tool_quotas WHERE window_start < ?", (start - seconds,))
                cursor = await conn.execute(
                    "SELECT count FROM tool_quotas WHERE bucket=? AND window_start=?",
                    (bucket, start),
                )
                row = await cursor.fetchone()
                used = int(row["count"]) if row else 0
                if used >= limit:
                    status["used"] = used
                    status["remaining"] = 0
                    return status
                await conn.execute(
                    "INSERT INTO tool_quotas(bucket,window_start,count) VALUES(?,?,?) "
                    "ON CONFLICT(bucket,window_start) DO UPDATE SET count=excluded.count",
                    (bucket, start, used + 1),
                )
            return None
        except Exception as exc:  # quotas are a safety control: fail closed
            _LOG.error("durable quota unavailable", extra={"bucket": bucket, "error": str(exc)[:200]})
            return status

    def _ceiling(self) -> int:
        value = getattr(self.settings.security, "max_tool_calls_ceiling", 200)
        try:
            ceiling = int(value)
        except (TypeError, ValueError):
            ceiling = 200
        return max(1, ceiling)

    def _limit_for_hint(self) -> int:
        value = self.settings.security.max_tool_calls_per_task
        cap = int(value) if isinstance(value, int) and not isinstance(value, bool) else 200
        return max(1, min(cap, self._ceiling()))

    def _limit_for(self, call: ToolCall) -> int:
        explicit = getattr(call, "max_tool_calls", None)
        if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
            # The hint is advisory only: the server ceiling always wins.
            return max(1, min(explicit, self._ceiling()))
        return self._limit_for_hint()

    def _configured_timeout(self, registration: Any) -> float | None:
        """The operator's own deadline for this tool, when it has one."""
        name = str(getattr(getattr(registration, "spec", None), "name", "") or "")
        field = _CONFIGURED_TIMEOUTS.get(name)
        if not field:
            return None
        value = getattr(self.settings.security, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
        return None

    def _timeout_for(self, registration: Any, call: ToolCall) -> float:
        for source in (getattr(call, "timeout_seconds", None), getattr(registration.spec, "timeout_seconds", None),
                       getattr(registration.spec, "timeout", None),
                       self._configured_timeout(registration), self.timeout_seconds):
            if isinstance(source, (int, float)) and not isinstance(source, bool) and source > 0:
                return float(source)
        return DEFAULT_TOOL_TIMEOUT_SECONDS

    @staticmethod
    def _timeout_error(registration: Any, call: ToolCall, timeout: float) -> str:
        """Keep the deadline unchanged; make the failure tell the model what to do."""
        base = f"tool timed out after {timeout}s"
        text = " ".join(_iter_strings(call.arguments))
        binary_target = bool(_BINARY_TARGET_RE.search(text)) or "sqlite" in text.casefold()
        if binary_target:
            return f"{base}; {_TIMEOUT_BINARY_HINT}"
        return f"{base}; {_TIMEOUT_GENERIC_HINT}"

    @staticmethod
    def _resource_for(registration: Any, call: ToolCall) -> str | None:
        """Derive the path-like resource a permission decision should be made on."""
        field = getattr(registration.spec, "resource_field", None)
        if field == "":
            return None  # explicitly non-path tool (e.g. a network read)
        keys: list[str] = []
        if isinstance(field, str):
            keys.append(field)
        keys.extend(k for k in _RESOURCE_KEYS if k not in keys)
        for key in keys:
            value = call.arguments.get(key)
            if value is None:
                continue
            if not isinstance(value, (str, Path)):
                return str(value)
            text = str(value)
            if "\x00" in text:
                return text  # let the tool deny it cleanly; never call lstat on NUL
            if text.startswith("~"):
                text = str(Path(text).expanduser())
            try:
                return str(Path(text).resolve())
            except (OSError, RuntimeError):
                return text
        return None

    async def _approval_denied(self, approval_id: str | None) -> bool:
        if not approval_id:
            return False
        try:
            row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,))
        except Exception:  # pragma: no cover - defensive; audit store must not break safety
            return False
        return bool(row and row["status"] == "denied")

    async def _await_approval(
        self,
        *,
        actor: str,
        action: str,
        resource: str | None,
        risk: RiskLevel,
        approval_id: str,
        current_digest: str | None,
    ) -> Decision:
        """Wait for a human to resolve this approval, bounded by its timeout.

        Approval is in the same database for both the single-process app and a
        split worker/gateway, so polling the row is enough: the transport that
        handles the button tap writes the decision. Approved -> re-authorize
        with the approval id so the normal one-shot/verified path continues.
        Denied, expired, missing or timed out -> an ordinary denial the model
        can answer, never another ApprovalRequired (which would end the turn).
        """
        timeout = max(1.0, float(getattr(self.settings.security, "approval_timeout_seconds", 900)))
        poll = max(0.05, float(getattr(self.settings.security, "approval_poll_seconds", 2.0)))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        _LOG.info("waiting for approval", extra={
            "approval_id": approval_id, "action": action, "timeout_seconds": timeout,
        })
        while True:
            try:
                row = await db.fetchone(
                    "SELECT status FROM approvals WHERE id=?", (approval_id,)
                )
            except Exception as exc:  # noqa: BLE001 - a lookup blip must not end the wait
                _LOG.warning("approval wait lookup failed", extra={
                    "approval_id": approval_id, "error": type(exc).__name__})
                row = None
            status = str(row["status"]) if row is not None else ""
            if status == "approved":
                decision = await self.permissions.authorize(
                    actor, action, resource, risk,
                    approval_id=approval_id, arguments_digest=current_digest,
                )
                if decision.allowed:
                    return decision
                return Decision(
                    allowed=False,
                    reason=decision.reason or "approval is no longer valid",
                    risk=risk,
                )
            if status in {"denied", "expired", "consumed"}:
                return Decision(allowed=False, reason=f"approval {status}", risk=risk)
            if row is None:
                # A vanished row cannot be resolved by waiting: wrong/cleaned
                # database or a deleted approval. Fall back to a denial.
                return Decision(allowed=False, reason="approval not found", risk=risk)
            if loop.time() >= deadline:
                minutes = max(1, int(timeout // 60))
                return Decision(
                    allowed=False,
                    reason=f"approval timed out after {minutes} min",
                    risk=risk,
                )
            await asyncio.sleep(poll)

    def _context(self, call: ToolCall, task_key: str, dry_run: bool) -> dict[str, Any]:
        try:
            db_path: str | None = str(db.get_db_path())
        except RuntimeError:
            db_path = str(self.settings.db_path)
        ctx: dict[str, Any] = {
            "settings": self.settings,
            "db_path": db_path,
            "actor": call.actor,
            "permissions": self.permissions,
            "events": self.events,
            "task_id": task_key,
            "correlation_id": call.correlation_id,
            "call_id": call.id,
            "dry_run": dry_run,
        }
        # Real service objects, never monkeypatched stand-ins: toolsets resolve
        # their dependencies from ``ctx['services']`` on every call. An empty
        # mapping is omitted so registration-time service injection still works.
        if self.services:
            ctx["services"] = dict(self.services)
            # The origin chat of the turn in flight, when one is known: a
            # background task created in this turn reports its progress there.
            # Read once here so a tool can never mutate the app's copy.
            app = self.services.get("app")
            origin = getattr(app, "_turn_origin", None)
            if isinstance(origin, dict) and origin:
                ctx["origin_context"] = dict(origin)
        return ctx

    @staticmethod
    def _in_scope(registration: ToolRegistration, scope: frozenset[str]) -> bool:
        """A scope entry matches a tool when it equals its name or category."""
        return registration.spec.name in scope or str(registration.spec.category) in scope

    @staticmethod
    def _caller_grants(
        call: ToolCall,
        grants: Any,
        granted_permissions: Any,
    ) -> Any:
        """Resolve caller capability grants without trusting handler-supplied data."""
        if granted_permissions is not None:
            return granted_permissions
        if grants is not None:
            return grants
        for name in ("grants", "granted_permissions", "permissions"):
            value = getattr(call, name, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _turn_key(call: ToolCall) -> tuple[str, str]:
        """Scope retry counters to one actor/turn without trusting the handler."""
        turn = (
            getattr(call, "turn_id", None)
            or getattr(call, "task_id", None)
            or call.correlation_id
            or "default"
        )
        return (str(getattr(call, "actor", None) or "anonymous"), str(turn))

    def _policy_hint(
        self, call: ToolCall, decision: str, error: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Return ONE consolidated hint on the Nth repeat of an error class."""
        if not error or decision in ("allowed", "pending", "not_evaluated"):
            return None
        key = self._turn_key(call)
        cls = _failure_class(decision, error)
        now = time.monotonic()
        with self._history_lock:
            history = self._failure_history
            if len(history) >= _HINT_HISTORY_MAX_KEYS:
                for stale in [k for k, v in history.items()
                              if now - v["seen_at"] > _HINT_HISTORY_TTL_SECONDS]:
                    history.pop(stale, None)
                while len(history) >= _HINT_HISTORY_MAX_KEYS:
                    oldest = min(history, key=lambda k: history[k]["seen_at"])
                    history.pop(oldest, None)
            turn = history.setdefault(key, {"classes": {}, "seen_at": now})
            if now - turn["seen_at"] > _TURN_IDLE_SECONDS:
                turn["classes"] = {}
            turn["seen_at"] = now
            state = turn["classes"].setdefault(cls, {"count": 0, "hinted": False})
            state["count"] += 1
            if state["count"] < POLICY_HINT_THRESHOLD or state["hinted"]:
                return None
            state["hinted"] = True
            count = state["count"]
        label = cls.split(":", 1)[0]
        hint = _POLICY_HINTS.get(
            cls, _POLICY_HINTS.get(label, _POLICY_HINTS["handler"])
        )
        text = (
            f"Policy hint ({count} repeated {label} failures this turn): {hint} "
            "Stop retrying variants; report the blocker or switch tools."
        )
        payload = {
            "class": cls, "count": count, "threshold": POLICY_HINT_THRESHOLD,
            "hint": hint,
        }
        return text, payload

    async def execute(
        self,
        call: ToolCall,
        approval_id: str | None = None,
        *,
        scope: frozenset[str] | None = None,
        grants: Any = None,
        granted_permissions: Any = None,
    ) -> ToolResult:
        started = time.perf_counter()
        # Some callers attach the approval id to the call record instead of
        # passing it as an explicit argument; both forms are accepted.
        approval_id = approval_id or getattr(call, "approval_id", None)
        effective_grants = self._caller_grants(call, grants, granted_permissions)
        registration = self.registry.get(call.name)
        risk = registration.spec.risk if registration else RiskLevel.LOW
        risk_reason = "registered tool tier"
        if registration is not None:
            # Per-call policy: read-only work and bounded writes do not need a
            # human tap; destructive, credential and self-modifying work does.
            risk, risk_reason = assess_call(call.name, call.arguments, risk, self.settings)
        task_key = self._quota_key(call)
        decision = "not_evaluated"
        error: str | None = None
        output: Any = None
        metadata: dict[str, Any] = {"risk_reason": risk_reason}
        success = False
        raised: BaseException | None = None
        resource: str | None = None

        try:
            if registration is None:
                decision = "unknown_tool"
                error = f"unknown tool: {call.name!r}"
            elif scope is not None and not self._in_scope(registration, scope):
                # Per-agent least privilege: the caller declared a tool scope,
                # and this tool is outside it. Denied (and audited) before
                # validation, permissions or handler execution.
                decision = "out_of_scope"
                error = f"tool {call.name!r} is outside the caller's declared tool scope"
            else:
                errors = self.registry.validate(call.name, call.arguments)
                if errors:
                    decision = "invalid_arguments"
                    error = "invalid arguments: " + "; ".join(errors)
                elif (
                    policy_error := shell_preflight(
                        call.name, call.arguments, self.settings
                    )
                ) is not None:
                    # Hardline/deny-glob/disabled denials happen before any
                    # approval request or handler dispatch.
                    decision = "denied"
                    error = policy_error
                    metadata["policy"] = "shell"
                elif (
                    missing := self.permissions.missing_capabilities(
                        registration.spec.permissions, effective_grants
                    )
                ):
                    # ToolSpec.permissions is a declaration of authority, not
                    # documentation: a missing or partial grant denies before
                    # authorize(), the handler, or any success event.
                    decision = "permission_denied"
                    error = "permission denied: missing capability " + ", ".join(missing)
                    metadata["missing_capabilities"] = missing
                else:
                    resource = self._resource_for(registration, call)
                    # A shell command without an explicit cwd runs in the
                    # checkout, so the checkout is its workspace resource. That
                    # makes the approval card name the scope a workspace trust
                    # covers, instead of showing no resource at all.
                    if call.name in {"shell", "terminal"} and (
                        resource is None or str(resource).strip() == ""
                    ):
                        resource = str(self.settings.root.resolve())
                    current_digest = arguments_digest(call.arguments)
                    command_preview = ""
                    if call.name in {"shell", "terminal"}:
                        try:
                            command_preview = str(
                                self._redact(str(call.arguments.get("command") or ""))
                            )[:300]
                        except Exception:  # noqa: BLE001 - preview is cosmetic
                            command_preview = ""
                    permission_decision = await self.permissions.authorize(
                        call.actor,
                        "tool:" + call.name,
                        resource,
                        risk,
                        approval_id=approval_id,
                        task_id=task_key,
                        tool_call_id=call.id,
                        arguments_digest=current_digest,
                        risk_reason=risk_reason,
                        command_preview=command_preview,
                        review_command=(
                            str(call.arguments.get("command") or "")
                            if call.name in {"shell", "terminal"} else ""
                        ),
                    )
                    # Hold the turn while a human decides. Without this the turn
                    # ended at ApprovalRequired, so an approval tapped seconds
                    # later had nothing left to authorize: the operator had to
                    # tell LEO again that they had approved. The wait is bounded
                    # by approval_timeout_seconds and cancellation-aware.
                    if (
                        permission_decision.requires_approval
                        and not permission_decision.allowed
                        and permission_decision.approval_id
                        and getattr(self.settings.security, "wait_for_approval", False)
                    ):
                        approval_id = permission_decision.approval_id
                        metadata["approval_id"] = approval_id
                        metadata["waited_for_approval"] = True
                        permission_decision = await self._await_approval(
                            actor=call.actor,
                            action="tool:" + call.name,
                            resource=resource,
                            risk=risk,
                            approval_id=approval_id,
                            current_digest=current_digest,
                        )
                    # One-shot tools (shell) consume the verified approval before
                    # dispatch, so a replay cannot run the same command twice.
                    if (
                        permission_decision.allowed
                        and permission_decision.verified_approval
                        and approval_id
                        and not call.dry_run
                        and getattr(registration.spec, "consume_approval", False)
                    ):
                        if not await self.permissions.consume_approval(approval_id):
                            permission_decision.allowed = False
                            permission_decision.reason = "approval already consumed"
                    decision = "pending" if permission_decision.requires_approval else (
                        "allowed" if permission_decision.allowed else "denied"
                    )
                    metadata["reason"] = permission_decision.reason
                    if permission_decision.requires_approval and not permission_decision.allowed:
                        approval_id = permission_decision.approval_id or approval_id
                        metadata["approval_id"] = approval_id
                        raise ApprovalRequired(
                            approval_id or "", permission_decision.reason
                        )
                    if not permission_decision.allowed:
                        error = f"permission denied: {permission_decision.reason}"
                    elif call.dry_run and not getattr(registration.spec, "supports_dry_run", False):
                        metadata["dry_run"] = True
                        success = True
                    else:
                        # Charge only a real dispatch that passed validation
                        # and permissions; denied/dry-run calls are free. The
                        # durable increment is the atomic quota gate, so a
                        # concurrent caller cannot overshoot the server ceiling.
                        if not call.dry_run:
                            exceeded = await self._charge_quota(
                                self._quota_bucket(call), self._limit_for(call)
                            )
                            if exceeded is not None:
                                raise _QuotaExceeded(exceeded)
                        timeout = self._timeout_for(registration, call)
                        metadata["timeout_seconds"] = timeout
                        ctx = self._context(call, task_key, dry_run=bool(call.dry_run))
                        raw_output = await asyncio.wait_for(
                            registration.handler(dict(call.arguments), ctx), timeout=timeout
                        )
                        output = self._redact(raw_output)
                        output_errors: list[str] = []
                        if registration.spec.output_schema:
                            _validate_value(output, registration.spec.output_schema, "", output_errors)
                        if output_errors:
                            decision = "invalid_output"
                            error = "output schema violation: " + "; ".join(output_errors)
                            metadata["output_schema_errors"] = output_errors
                            output = None
                        elif isinstance(output, Mapping) and output.get("success") is False:
                            # A handler that reports its own failure (including
                            # dry-run previews that executed nothing) must not be
                            # published to the model as a successful tool call.
                            success = False
                            detail = output.get("error") or output.get("message")
                            error = f"handler reported failure: {detail or 'success=false'}"
                            metadata["handler_success"] = False
                        else:
                            success = True
                        if call.dry_run:
                            metadata["dry_run"] = True
        except ApprovalRequired as exc:
            raised = exc
            error = str(exc)
        except _QuotaExceeded as exc:
            decision = "rate_limited"
            metadata["quota"] = exc.status
            error = (
                f"tool call limit exceeded for {exc.status['key']}: "
                f"{exc.status['limit']} calls per {int(self.QUOTA_WINDOW_SECONDS)}s"
            )
        except asyncio.TimeoutError:
            decision = "timeout"
            error = self._timeout_error(
                registration, call,
                self._timeout_for(registration, call) if registration else self.timeout_seconds,
            )
        except asyncio.CancelledError as exc:
            raised = exc
            decision = "cancelled"
            error = "tool execution cancelled"
        except Exception as exc:
            decision = "denied" if isinstance(exc, PermissionError) else "error"
            error = f"{type(exc).__name__}: {exc}"

        duration_ms = (time.perf_counter() - started) * 1000.0
        if error:
            hint = self._policy_hint(call, decision, error)
            if hint:
                hint_text, metadata["policy_hint"] = hint
                error = f"{error} | {hint_text}"
            error = self._redact(error)
        await self._record(
            call, risk.value, decision, success, duration_ms, error, output,
            metadata, resource, task_key,
        )
        if raised is not None:
            raise raised
        return ToolResult(
            call_id=call.id,
            tool=call.name,
            success=success,
            output=output,
            error=error,
            duration_ms=duration_ms,
            risk=risk,
            decision=decision,
            metadata=metadata,
        )

    async def _record(
        self,
        call: ToolCall,
        risk: str,
        decision: str,
        success: bool,
        duration_ms: float,
        error: str | None,
        output: Any,
        metadata: dict[str, Any],
        resource: str | None,
        task_key: str,
    ) -> None:
        call_id = call.id or new_id("call")
        args_json = self._redact(
            json.dumps(call.arguments, ensure_ascii=False, default=str)
        )
        try:
            await db.execute(
                "INSERT OR REPLACE INTO tool_calls"
                "(id,tool,actor,args,risk,decision,success,duration_ms,error,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id, call.name, call.actor, args_json, risk, decision,
                    int(success), duration_ms, error, utcnow(),
                ),
            )
        except Exception:
            _LOG.exception("failed to persist tool call", extra={"tool": call.name})

        # One audit row per execution carries the FINAL decision (validated,
        # rate-limited, denied, errored, allowed), not just the pre-flight
        # permission verdict, so /audit can be trusted for incident review.
        try:
            await db.execute(
                "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,correlation_id) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (utcnow(), call.actor, "tool.execute", resource, risk, decision,
                 error, json.dumps(metadata, default=str)[:1000], task_key),
            )
        except Exception:
            _LOG.exception("failed to persist tool audit", extra={"tool": call.name})

        payload = {
            "call_id": call_id,
            "tool": call.name,
            "actor": call.actor,
            "task": task_key,
            "risk": risk,
            "decision": decision,
            "success": success,
            "dry_run": bool(call.dry_run or metadata.get("dry_run")),
            "duration_ms": duration_ms,
            "resource": self._redact(resource) if resource else None,
            "approval_id": metadata.get("approval_id"),
        }
        if success:
            payload["output"] = output
            topic = "tool.called"
        else:
            payload["error"] = error
            topic = "tool.failed"
        try:
            await self.events.publish(
                Event(
                    topic=topic,
                    source="tools",
                    payload=self._redact(payload),
                    correlation_id=call.correlation_id,
                    created_at=utcnow(),
                )
            )
        except Exception:
            _LOG.exception("failed to publish tool event", extra={"tool": call.name})
