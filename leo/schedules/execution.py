"""Native execution adapters for durable schedules.

Every surviving execution field has one real consumer here:

* ``agent`` (default) delegates the prompt to the agent engine;
* ``reminder``/``no_agent`` text delivers a fixed message without a model;
* ``script`` runs the operator-provisioned executable, and its stdout either
  becomes the result (``no_agent``) or is handed to the agent;
* ``monitor`` observes a URL/script before task creation and suppresses an
  unchanged occurrence entirely.

``context_from``, ``continuity``, ``skills``, ``workdir``, ``enabled_toolsets``,
``attach_to_session``, ``pinned`` and ``grace_minutes`` are consumed at this
boundary too. Nothing here creates a second schedule authority; the durable
row remains the scheduler's input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import db
from ..logging import get_logger
from .clock import Clock, SystemClock
from .monitor import BASELINE_KEY, Monitor, MonitorError, MonitorResult
from .scan import scan_text
from .script_runner import ScriptExecutionError, ScriptResult, ScriptRunner
from .store import ScheduleStore

WORKER_SCRIPT = "schedule_script"
WORKER_MONITOR = "schedule_monitor"

_CONTEXT_BLOCK_CHARS = 2_000
_CONTEXT_TOTAL_CHARS = 8_000
_SKILL_TOTAL_CHARS = 12_000
_NOTE_CHARS = 8_000

_LOG = get_logger("schedules.execution")


class ExecutionError(RuntimeError):
    """A stable execution refusal that fails the scheduled task closed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class OccurrenceDecision:
    """What the scheduler should do with one due occurrence."""

    suppress: bool = False
    reason: str = ""
    worker: str | None = None
    objective: str | None = None
    payload_updates: dict[str, Any] = field(default_factory=dict)
    event_topic: str | None = None
    event_payload: dict[str, Any] = field(default_factory=dict)
    # Durable runtime kv writes the scheduler must commit atomically with the
    # occurrence (monitor baselines). Values are JSON strings.
    runtime_updates: dict[str, str] = field(default_factory=dict)
    # The in-memory baseline to use for a following occurrence in the same tick.
    baseline_candidate: dict[str, Any] | None = None


@dataclass
class PreparedAgent:
    """The objective and trusted execution context for an agent run."""

    objective: str
    context: dict[str, Any] = field(default_factory=dict)


def _execution(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(payload or {}).get("execution")
    return dict(value) if isinstance(value, Mapping) else {}


def select_adapter(payload: Mapping[str, Any] | None) -> str:
    """Select the adapter from the persisted execution config."""
    payload = dict(payload or {})
    execution = _execution(payload)
    if execution.get("monitor"):
        return "monitor"
    if execution.get("script"):
        return "script"
    for key in ("text", "message", "reminder"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return "reminder"
    return "agent"


def _bounded(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)] + "\n...[truncated]"


def _result_text(result: Any) -> str:
    """Readable text from a stored task result without inventing content."""
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        for key in ("output", "content", "summary", "result", "answer", "reminder"):
            if key not in result:
                continue
            value = result[key]
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, Mapping):
                nested = _result_text(value)
                if nested.strip():
                    return nested
        try:
            return json.dumps(dict(result), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)
    return str(result or "")


def _loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return None


def apply_wrap_response(text: str, payload: Mapping[str, Any] | None) -> str:
    """Apply the stored ``wrap_response`` framing to delivered text."""
    payload = dict(payload or {})
    mode = _execution(payload).get("wrap_response")
    if str(mode or "").lower() != "on":
        # "off" and "inherit" both leave the underlying message untouched; the
        # inherited default is LEO's existing delivery format.
        return text
    name = str(payload.get("name") or "Scheduled job").strip() or "Scheduled job"
    return f"📅 {name}\n\n{text}"


class ScheduleExecution:
    """Adapter registry plus the durable execution helpers."""

    def __init__(
        self,
        *,
        settings: Any,
        store: ScheduleStore,
        clock: Clock | None = None,
        scripts_root: str | Path | None = None,
        allowed_workdir_roots: Sequence[str | Path] | None = None,
        http_fetcher: Any = None,
        display_chars: int = 4_000,
    ) -> None:
        self.settings = settings
        self.store = store
        self.clock = clock or getattr(store, "clock", None) or SystemClock()
        automation = getattr(settings, "automation", None)
        self.allowed_workdir_roots = tuple(
            Path(root).expanduser().resolve()
            for root in (allowed_workdir_roots or ())
        )
        self.scripts = ScriptRunner(
            scripts_root=scripts_root,
            allow_scripts=bool(getattr(automation, "allow_scripts", False)),
            timeout_seconds=int(getattr(automation, "script_timeout_seconds", 60)),
            max_output_bytes=int(
                getattr(automation, "script_max_output_bytes", 200_000)
            ),
            allowed_workdir_roots=self.allowed_workdir_roots,
        )
        self.monitor = Monitor(
            store=store,
            clock=self.clock,
            script_runner=self.scripts,
            http_fetcher=http_fetcher,
            timeout_seconds=float(
                getattr(automation, "monitor_timeout_seconds", 20.0)
            ),
            max_bytes=int(getattr(automation, "monitor_max_bytes", 200_000)),
            display_chars=display_chars,
        )
        self.display_chars = max(1, int(display_chars))

    # ------------------------------------------------------------------
    # Scheduler preflight (monitor suppression, worker selection)
    # ------------------------------------------------------------------
    async def preflight(
        self,
        *,
        schedule_id: str,
        name: str,
        payload: Mapping[str, Any],
        scheduled_for: datetime | str | None = None,
        baseline_override: dict[str, Any] | None = None,
    ) -> OccurrenceDecision | None:
        adapter = select_adapter(payload)
        if adapter == "script":
            return OccurrenceDecision(worker=WORKER_SCRIPT)
        if adapter != "monitor":
            return None

        execution = _execution(payload)
        target = str(execution.get("monitor") or "")
        try:
            workdir = self._workdir(execution, required=False)
        except ExecutionError as exc:
            return OccurrenceDecision(
                worker=WORKER_MONITOR,
                objective=f"Check monitor for {name}",
                payload_updates={"monitor_error": exc.message, "monitor_source": "error"},
                event_topic="schedule.monitor_failed",
                event_payload={
                    "name": name,
                    "schedule_id": schedule_id,
                    "scheduled_for": scheduled_for,
                    "source": "error",
                    "error": exc.message,
                },
            )
        try:
            result = await self.monitor.check(
                schedule_id=schedule_id,
                target=target,
                workdir=workdir,
                baseline_override=baseline_override,
                persist=False,
            )
        except MonitorError as exc:  # pragma: no cover - check() reports errors
            result = MonitorResult(
                changed=False,
                suppressed=False,
                first_run=True,
                source="error",
                error=exc.message,
                evidence={"reason": exc.code},
            )
        except Exception as exc:  # noqa: BLE001 - fail closed, never run blind
            result = MonitorResult(
                changed=False,
                suppressed=False,
                first_run=True,
                source="error",
                error=f"{type(exc).__name__}: {exc}",
                evidence={"reason": "monitor_error"},
            )

        when = (
            scheduled_for.isoformat()
            if isinstance(scheduled_for, datetime)
            else scheduled_for
        )
        base_event = {
            "name": name,
            "schedule_id": schedule_id,
            "scheduled_for": when,
            "source": result.source,
        }
        runtime_updates: dict[str, str] = {}
        baseline_candidate: dict[str, Any] | None = None
        if result.baseline_value is not None:
            baseline_candidate = dict(result.baseline_value)
            runtime_updates[BASELINE_KEY] = json.dumps(
                baseline_candidate, ensure_ascii=False
            )
        if result.error:
            # The failure lane is a task that fails, so the existing failure
            # delivery path handles it and the good baseline is untouched.
            return OccurrenceDecision(
                worker=WORKER_MONITOR,
                objective=f"Check monitor for {name}",
                payload_updates={
                    "monitor_error": result.error,
                    "monitor_source": result.source,
                },
                event_topic="schedule.monitor_failed",
                event_payload={**base_event, "error": result.error},
            )
        if result.suppressed:
            return OccurrenceDecision(
                suppress=True,
                reason="monitor baseline" if result.first_run else "monitor unchanged",
                event_topic="schedule.monitor_unchanged",
                event_payload={
                    **base_event,
                    "first_run": result.first_run,
                    "truncated": result.truncated,
                },
                runtime_updates=runtime_updates,
                baseline_candidate=baseline_candidate,
            )
        return OccurrenceDecision(
            worker=WORKER_MONITOR,
            objective=f"Monitor changed for {name}",
            payload_updates={
                "monitor_change": {
                    "old": _bounded(result.old_content, self.display_chars),
                    "new": _bounded(result.new_content, self.display_chars),
                    "source": result.source,
                    "truncated": result.truncated,
                }
            },
            event_topic="schedule.monitor_changed",
            event_payload={
                **base_event,
                "old_chars": len(result.old_content),
                "new_chars": len(result.new_content),
                "truncated": result.truncated,
            },
            runtime_updates=runtime_updates,
            baseline_candidate=baseline_candidate,
        )

    # ------------------------------------------------------------------
    # Worker-side adapters
    # ------------------------------------------------------------------
    async def run_script(self, *, task: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        payload, execution = self._task_payload(task, context)
        script = execution.get("script")
        result = await self._run_script_target(
            script,
            execution.get("script_args") or [],
            self._workdir(execution, required=False),
        )
        return {
            "objective": self._task_objective(task, context),
            "script": result.as_dict(),
            "execution": execution,
            "payload": payload,
        }

    async def run_monitor(self, *, task: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        payload, execution = self._task_payload(task, context)
        error = payload.get("monitor_error")
        if error:
            raise ExecutionError("monitor_failed", str(error))
        change = payload.get("monitor_change")
        if not isinstance(change, Mapping):
            # A monitor worker without a prefetched observation cannot claim a
            # wake; fail closed instead of delivering an empty "change".
            raise ExecutionError(
                "monitor_missing_observation",
                "monitor task has no prefetched observation.",
            )
        old = _bounded(str(change.get("old") or ""), self.display_chars)
        new = _bounded(str(change.get("new") or ""), self.display_chars)
        output = (
            "Monitor changed.\n\n"
            f"Previous observation:\n{old or '(empty)'}\n\n"
            f"Current observation:\n{new or '(empty)'}"
        )
        return {
            "objective": self._task_objective(task, context),
            "output": output,
            "monitor": {
                "changed": True,
                "old": old,
                "new": new,
                "source": str(change.get("source") or ""),
                "truncated": bool(change.get("truncated")),
            },
            "execution": execution,
            "payload": payload,
        }

    async def prepare_agent(
        self,
        *,
        task: Any,
        context: Mapping[str, Any],
        extra_objective: str | None = None,
    ) -> PreparedAgent:
        payload, execution = self._task_payload(task, context)
        pin = self._pinned_context(execution)
        workdir = self._workdir(execution, required=True)
        objective = self._task_objective(task, context)
        # Compile-time preflight is not enough: a stored prompt can be changed
        # between proposal and run (or by a direct SQLite operator edit). Re-scan
        # here, before the agent sees the text, and carry only the stable finding
        # code/rule into the refusal so the unsafe text is never echoed to a log.
        for label, text in (
            ("prompt", objective),
            ("extra_objective", str(extra_objective or "")),
        ):
            finding = scan_text(text, scope="prompt").first()
            if finding is not None:
                raise ExecutionError(
                    "prompt_unsafe_at_execution",
                    f"scheduled {label} refused by execution-time safety scan"
                    f" ({finding.code}/{finding.rule})",
                )
        parts: list[str] = []
        if extra_objective:
            parts.append(str(extra_objective).strip())
        if objective.strip():
            parts.append(objective.strip())
        brief = await self._build_brief(payload, execution)
        if brief:
            finding = scan_text(brief, scope="prompt").first()
            if finding is not None:
                raise ExecutionError(
                    "prompt_unsafe_at_execution",
                    "scheduled context refused by execution-time safety scan"
                    f" ({finding.code}/{finding.rule})",
                )
            parts.append(brief)

        agent_context: dict[str, Any] = {}
        if workdir:
            agent_context["workdir"] = workdir
        enabled = execution.get("enabled_toolsets")
        if enabled is not None:
            # [] is an explicit zero-tool allowlist; None inherits the agent's
            # declared scope.
            agent_context["enabled_toolsets"] = list(enabled) if isinstance(
                enabled, (list, tuple)
            ) else []
        agent_context.update(pin)
        return PreparedAgent(objective="\n\n".join(parts), context=agent_context)

    # ------------------------------------------------------------------
    # Script helpers
    # ------------------------------------------------------------------
    async def _run_script_target(
        self, script: Any, args: Any, workdir: str | None
    ) -> ScriptResult:
        if not script:
            raise ExecutionError("missing_script", "script execution has no script.")
        try:
            result = await self.scripts.run(script, args, workdir=workdir)
        except ScriptExecutionError as exc:
            raise ExecutionError(exc.code, exc.message) from None
        if result.exit_code != 0:
            detail = _bounded(result.stderr.strip() or result.stdout.strip(), 400)
            raise ExecutionError(
                "script_failed",
                f"script exited {result.exit_code}"
                + (f": {detail}" if detail else ""),
            )
        return result

    # ------------------------------------------------------------------
    # Context / continuity / skills
    # ------------------------------------------------------------------
    async def _build_brief(
        self, payload: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> str:
        schedule_id = str(payload.get("schedule_id_hint") or "")
        # The caller stores the schedule id in task metadata; execution helpers
        # accept it through the payload carry-over below.
        schedule_id = str(payload.get("__schedule_id") or schedule_id)
        name = str(payload.get("name") or "")
        blocks: list[str] = []
        total = 0
        refs = execution.get("context_from") or []
        if isinstance(refs, (list, tuple)):
            for ref in refs:
                ref = str(ref)
                is_self = ref.lower() == "self"
                target_id = schedule_id if is_self else ref
                label, text = await self._context_block(
                    target_id, is_self=is_self, own_name=name
                )
                remaining = _CONTEXT_TOTAL_CHARS - total
                if remaining <= 0:
                    blocks.append("[context truncated]")
                    break
                bounded = _bounded(text, min(_CONTEXT_BLOCK_CHARS, remaining))
                total += len(bounded)
                blocks.append(f"[{label}]\n{bounded}")
        if blocks:
            heading = "Context from previous scheduled runs:"
            body = "\n\n".join(blocks)
        else:
            heading = ""
            body = ""

        skill_blocks: list[str] = []
        snapshot = payload.get("skill_snapshot")
        snapshot = dict(snapshot) if isinstance(snapshot, Mapping) else {}
        skills = execution.get("skills") or []
        skill_total = 0
        for skill in skills if isinstance(skills, (list, tuple)) else []:
            skill = str(skill)
            doc = str(snapshot.get(skill) or "").strip()
            remaining = _SKILL_TOTAL_CHARS - skill_total
            if remaining <= 0:
                break
            if doc:
                bounded = _bounded(doc, remaining)
                skill_total += len(bounded)
                skill_blocks.append(f"### {skill}\n{bounded}")
            else:
                skill_blocks.append(f"### {skill}\n(documentation snapshot unavailable)")
        skill_heading = "Frozen skill documentation (captured when the schedule was created):"

        parts: list[str] = []
        if heading:
            parts.append(f"{heading}\n\n{body}")
        if skill_blocks:
            parts.append(f"{skill_heading}\n\n" + "\n\n".join(skill_blocks))
        # Nightly self-review is written without a card; the next scheduled run
        # carries a bounded copy so the finding is actually surfaced to the user.
        try:
            from ..learning.self_improvement import recent_review_brief

            review = await recent_review_brief()
        except Exception as exc:  # noqa: BLE001 - context must not break a run
            _LOG.warning("self-review brief unavailable: %s", exc)
            review = ""
        if review:
            parts.append("Recent self-review (learned from failures):\n" + review)
        return "\n\n".join(parts)

    async def _context_block(
        self, schedule_id: str, *, is_self: bool, own_name: str
    ) -> tuple[str, str]:
        if not schedule_id:
            return "context", "No stable schedule id was recorded for this run."
        view = await self.store.get(schedule_id)
        name = view.name if view is not None else ""
        row = await db.fetchone(
            "SELECT id,result,completed_at FROM tasks "
            "WHERE status='completed' AND ("
            "json_extract(metadata,'$.schedule_id')=? "
            "OR (?<>'' AND json_extract(metadata,'$.schedule')=?)"
            ") ORDER BY completed_at DESC, rowid DESC LIMIT 1",
            (schedule_id, name, name),
        )
        if is_self:
            label = "continuity: previous run of this job"
        else:
            label = f"context_from: {name or schedule_id}"
        if row is None:
            if is_self:
                return label, "No previous completed run yet; this is the first run."
            if view is None:
                return label, f"Source schedule {schedule_id} is missing or was removed."
            return label, f"Source schedule {name or schedule_id} has no completed output yet."
        text = _result_text(_loads(row["result"]))
        completed = str(row["completed_at"] or "")
        header = f"latest completed task {row['id']}"
        if completed:
            header += f" at {completed}"
        return label, f"{header}\n{_bounded(text, _CONTEXT_BLOCK_CHARS)}"

    # ------------------------------------------------------------------
    # Pinning / workdir
    # ------------------------------------------------------------------
    def _pinned_context(self, execution: Mapping[str, Any]) -> dict[str, Any]:
        if not execution.get("pinned"):
            return {}
        provider = str(execution.get("provider") or "").strip()
        model = str(execution.get("model") or "").strip()
        if not provider or not model:
            raise ExecutionError(
                "pinned_target_missing",
                "This schedule is pinned, but its operator provider/model is missing.",
            )
        providers = getattr(getattr(self.settings, "models", None), "providers", {}) or {}
        if provider not in providers:
            raise ExecutionError(
                "pinned_target_unavailable",
                f"The pinned provider {provider!r} is no longer configured.",
            )
        return {"pinned_provider": provider, "pinned_model": model}

    def _workdir(
        self, execution: Mapping[str, Any], *, required: bool
    ) -> str | None:
        raw = execution.get("workdir")
        if raw in (None, ""):
            return None
        try:
            candidate = Path(str(raw)).expanduser()
            if not candidate.is_absolute():
                raise ValueError("workdir must be absolute")
            resolved = candidate.resolve(strict=False)
        except (OSError, ValueError) as exc:
            if required:
                raise ExecutionError(
                    "invalid_workdir", f"workdir is invalid: {exc}"
                ) from None
            return None
        if not resolved.is_dir():
            if required:
                raise ExecutionError(
                    "invalid_workdir", "workdir does not exist or is not a directory."
                )
            return None
        if self.allowed_workdir_roots and not any(
            resolved == root or root in resolved.parents
            for root in self.allowed_workdir_roots
        ):
            raise ExecutionError(
                "workdir_outside_allowed_roots",
                "workdir is outside LEO's allowed roots.",
            )
        return str(resolved)

    # ------------------------------------------------------------------
    # Durable session note
    # ------------------------------------------------------------------
    async def record_outcome(
        self,
        *,
        task: Any,
        context: Mapping[str, Any],
        status: str,
        text: str,
    ) -> None:
        payload, execution = self._task_payload(task, context)
        if not execution.get("attach_to_session"):
            return
        schedule_id = str(
            payload.get("__schedule_id")
            or (getattr(task, "metadata", {}) or {}).get("schedule_id")
            or ""
        )
        name = str(payload.get("name") or (getattr(task, "metadata", {}) or {}).get("schedule") or "")
        if not schedule_id:
            return
        await record_session_note(
            schedule_id=schedule_id,
            name=name,
            task_id=str(getattr(task, "id", "") or ""),
            status=status,
            text=text,
            clock=self.clock,
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _task_payload(
        task: Any, context: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = dict(context.get("schedule_payload") or {})
        metadata = getattr(task, "metadata", None) or {}
        if not payload and isinstance(metadata, Mapping):
            payload = dict(metadata.get("schedule_payload") or {})
        # Carry the durable schedule id so context/notes never have to guess.
        schedule_id = (
            context.get("schedule_id")
            or metadata.get("schedule_id")
            or (payload.get("schedule") or {}).get("id")
        )
        if schedule_id:
            payload.setdefault("__schedule_id", str(schedule_id))
        return payload, _execution(payload)

    @staticmethod
    def _task_objective(task: Any, context: Mapping[str, Any]) -> str:
        return str(context.get("objective") or getattr(task, "objective", "") or "")


async def record_session_note(
    *,
    schedule_id: str,
    name: str,
    task_id: str,
    status: str,
    text: str,
    clock: Clock | None = None,
    conversation_id: str | None = None,
    channel: str | None = None,
    note_meta: Mapping[str, Any] | None = None,
) -> str:
    """Append one durable assistant note to the job's own conversation.

    This is the minimum honest local session path: a real ``conversations`` row
    plus an assistant ``messages`` row. The next turn that uses the same
    conversation id rehydrates it through the normal history path. There is no
    fake two-way chat and no inbound reply is invented.
    """
    clock = clock or SystemClock()
    now = clock.now().isoformat()
    conversation_id = str(conversation_id or f"schedule-{schedule_id}")
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at,metadata) "
        "VALUES(?,?,?,?,?,?)",
        (
            conversation_id,
            str(channel or "schedule"),
            name or "Scheduled job",
            now,
            now,
            json.dumps({"schedule_id": schedule_id, "kind": "schedule_session"}),
        ),
    )
    note = _bounded(
        f"{str(status).upper()} · {name or 'Scheduled job'}\n\n{str(text or '').strip()}",
        _NOTE_CHARS,
    )
    metadata = {
        "schedule_id": schedule_id,
        "schedule_name": name,
        "task_id": task_id,
        "run_status": status,
        "schedule_note": True,
        "tool_calls": [],
    }
    for key, value in dict(note_meta or {}).items():
        if key not in metadata:
            metadata[key] = value
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
        "VALUES(?,?,?,?,?)",
        (
            conversation_id,
            "assistant",
            note,
            json.dumps(metadata, default=str, ensure_ascii=False),
            now,
        ),
    )
    await db.execute(
        "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id)
    )
    return conversation_id


async def latest_session_note(schedule_id: str, *, limit_chars: int = 1_200) -> str:
    """Compact brief from the job's own conversation, for diagnostics/tests."""
    row = await db.fetchone(
        "SELECT content FROM messages WHERE conversation_id=? AND role='assistant' "
        "ORDER BY id DESC LIMIT 1",
        (f"schedule-{schedule_id}",),
    )
    if row is None:
        return ""
    return _bounded(str(row["content"] or ""), max(1, int(limit_chars)))


__all__ = [
    "ExecutionError",
    "OccurrenceDecision",
    "PreparedAgent",
    "ScheduleExecution",
    "WORKER_MONITOR",
    "WORKER_SCRIPT",
    "apply_wrap_response",
    "latest_session_note",
    "record_session_note",
    "select_adapter",
]
