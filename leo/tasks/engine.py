"""Durable task engine.

Tasks are state machines backed by the shared SQLite database. Every mutation is
persisted before the corresponding ``task.*`` event is emitted. Completion is
only possible through :meth:`DurableTaskEngine.complete` and always requires
recorded evidence; a worker's return value is data, never proof of success.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping

from .. import db
from ..config import Settings
from ..contracts import TaskRecord, TaskStatus, new_id, utcnow
from ..events import EventBus
from ..logging import get_logger

_LOG = get_logger("tasks")

Worker = Callable[..., Awaitable[Any]]
DeliveryHandler = Callable[..., Awaitable[Any]]

#: Registry keys used when a task has no explicit owner_agent.
SCHEDULE_WORKER = "schedule"
NOTIFY_WORKER = "notify"
DEFAULT_WORKER = "*"
DEFAULT_DELIVERY = "*"

#: A claim is considered stale after this long without a heartbeat. Workers
#: can override it per task via ``resource_limits['lease_seconds']`` and renew
#: it at any time with :meth:`DurableTaskEngine.heartbeat`.
DEFAULT_LEASE_SECONDS = 300.0

# An evidence item must identify an external source or artifact. A bare claim
# ("the worker says it succeeded") is self-attestation, not evidence. This is
# the task-store equivalent of the verifier's ``evidence_valid`` check.
_EVIDENCE_IDENTITY_KEYS = (
    "source", "kind", "type", "path", "data", "output", "content", "detail",
    "artifact", "file", "ref", "id", "name", "tool", "step", "test", "check",
    "result",
)
_EMPTY_VALUES = (None, "", [], {}, (), False, 0)
_REJECTED_STATUSES = frozenset({"rejected", "invalid", "failed", "contradictory"})
_REPLAY_MARKER_KEYS = ("delivery", "delivery_status", "notified", "notify_status", "task.delivery")
_TARGET_KEYS = ("notify_targets", "notify", "delivery_targets", "delivery_target", "delivery")

# Legal state transitions. Terminal states are absorbing: a failed or cancelled
# task must be re-created (with a new idempotency key) to run again.
ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset(
        {TaskStatus.PLANNING, TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.PLANNING: frozenset(
        {TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.READY: frozenset(
        {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.READY,
            TaskStatus.WAITING,
            TaskStatus.BLOCKED,
            TaskStatus.VERIFYING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.WAITING: frozenset(
        {
            TaskStatus.READY,
            TaskStatus.RUNNING,
            TaskStatus.BLOCKED,
            TaskStatus.VERIFYING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.BLOCKED: frozenset(
        {
            TaskStatus.READY,
            TaskStatus.RUNNING,
            TaskStatus.VERIFYING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.VERIFYING: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.READY,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))


def _loads(value: Any, fallback: Any) -> Any:
    if value is None or value == "":
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _history(task: TaskRecord, event: str, **details: Any) -> None:
    entry: dict[str, Any] = {"event": event, "at": utcnow()}
    entry.update(details)
    task.history.append(entry)


def _validate_evidence(evidence: Any) -> list[dict[str, Any]]:
    """Apply the verifier's evidence rules before a task can be COMPLETED.

    Evidence must be non-empty, each item must identify a source or artifact,
    and no item may explicitly contradict or disprove the claim. A model's own
    assertion ("done") is deliberately not sufficient.
    """
    if not isinstance(evidence, (list, tuple)) or not evidence:
        raise ValueError("completion requires non-empty evidence")
    cleaned: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("evidence entries must be dicts")
        record = dict(item)
        if not record:
            raise ValueError("self-attested evidence rejected: evidence entries must identify a source or artifact")
        if (
            any(record.get(key) is False for key in ("supports", "verified", "valid"))
            or str(record.get("status", "")).lower() in _REJECTED_STATUSES
        ):
            raise ValueError("contradictory evidence cannot complete a task")
        if not any(record.get(key) not in _EMPTY_VALUES for key in _EVIDENCE_IDENTITY_KEYS):
            raise ValueError(
                "self-attested evidence rejected: evidence needs a source, artifact, kind or data"
            )
        cleaned.append(record)
    return cleaned


def _replay_signals(task: TaskRecord) -> list[str]:
    """Name every durable side-effect signal present on a stranded task."""
    signals: list[str] = []
    if task.metadata.get("idempotency_key"):
        signals.append("idempotency_key")
    if task.checkpoints:
        signals.append("checkpoints")
    if task.evidence:
        signals.append("evidence")
    if task.result is not None:
        signals.append("result")
    for key in _REPLAY_MARKER_KEYS:
        if task.metadata.get(key) is not None:
            signals.append(key)
    return signals


def _add_targets(value: Any, out: dict[str, Any]) -> None:
    """Normalize a notify/delivery declaration into ``{channel: target}``."""
    if value is None:
        return
    if isinstance(value, Mapping):
        data = dict(value)
        if "channel" in data and any(k in data for k in ("target", "to", "recipient")):
            channel = data.get("channel")
            target = next(data[k] for k in ("target", "to", "recipient") if k in data)
            if channel:
                out.setdefault(str(channel), target)
            return
        for key, item in data.items():
            if key in ("status", "at", "detail", "error", "attempts", "seq"):
                continue  # a delivery receipt, not a declaration
            if key in ("channels", "targets") and isinstance(item, (Mapping, list, tuple)):
                _add_targets(item, out)
            elif item is None or isinstance(item, (str, int)):
                if item is not None:
                    out.setdefault(str(key), item)
            else:
                _add_targets(item, out)
        return
    if isinstance(value, str):
        if value:
            out.setdefault(value, None)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _add_targets(item, out)


def _delivery_targets(task: TaskRecord) -> dict[str, Any]:
    """Return the task's delivery targets from metadata and schedule payload."""
    out: dict[str, Any] = {}
    metadata = task.metadata or {}
    for key in _TARGET_KEYS:
        value = metadata.get(key)
        if isinstance(value, Mapping) and ("status" in value or "at" in value):
            continue  # persisted delivery receipt, never a target declaration
        _add_targets(value, out)
    payload = metadata.get("schedule_payload")
    if isinstance(payload, Mapping):
        for key in (*_TARGET_KEYS, "channel", "channels"):
            if key in payload:
                _add_targets(payload[key], out)
    return out


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _merge_checkpoints(
    stored: Any, rows: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """One authoritative checkpoint view: table rows plus any legacy JSON rows."""
    merged: dict[int, dict[str, Any]] = {}
    for entry in stored if isinstance(stored, list) else []:
        if not isinstance(entry, Mapping):
            continue
        try:
            seq = int(entry.get("seq", 0))
        except (TypeError, ValueError):
            continue
        if seq > 0:
            merged[seq] = dict(entry)
    for row in rows:
        merged[int(row["seq"])] = {
            "seq": int(row["seq"]),
            "state": _loads(row["state"], {}),
            "created_at": row["created_at"],
        }
    return [merged[seq] for seq in sorted(merged)]


def _utc_after(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class TaskConflictError(RuntimeError):
    """A task row was modified concurrently; reload and re-apply the change."""


def _row_to_task(row: Mapping[str, Any]) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        objective=row["objective"],
        status=TaskStatus(row["status"]),
        plan=_loads(row["plan"], []),
        current_step=row["current_step"],
        history=_loads(row["history"], []),
        dependencies=_loads(row["dependencies"], []),
        checkpoints=_loads(row["checkpoints"], []),
        result=_loads(row["result"], None),
        errors=_loads(row["errors"], []),
        retry_count=row["retry_count"],
        resource_limits=_loads(row["resource_limits"], {}),
        cancellation_requested=bool(row["cancellation_requested"]),
        owner_agent=row["owner_agent"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        evidence=_loads(row["evidence"], []),
        metadata=_loads(row["metadata"], {}),
        version=int(row["version"] or 0),
    )


def _max_retries(limits: Mapping[str, Any]) -> int:
    for key in ("max_retries", "retries", "retry_count", "max_retry_count", "retry_limit"):
        if key in limits:
            try:
                return max(0, int(limits[key]))
            except (TypeError, ValueError):
                return 0
    if "max_attempts" in limits:
        try:
            return max(0, int(limits["max_attempts"]) - 1)
        except (TypeError, ValueError):
            return 0
    return 0


def _worker_arity(worker: Worker) -> int:
    """Accept ``worker(task)``, ``worker(task, engine)`` and dispatch's 3-arg form."""
    try:
        parameters = list(inspect.signature(worker).parameters.values())
    except (TypeError, ValueError):
        return 2
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        return 3
    return sum(
        1 for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


async def _call_worker(
    worker: Worker,
    task: TaskRecord,
    engine: "DurableTaskEngine",
    context: Mapping[str, Any],
) -> Any:
    arity = _worker_arity(worker)
    if arity <= 1:
        return await worker(task)
    if arity == 2:
        return await worker(task, engine)
    return await worker(task, engine, context)


def _timeout(limits: Mapping[str, Any], default: float) -> float | None:
    value = limits.get("timeout_seconds", limits.get("timeout", limits.get("task_timeout_seconds", default)))
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = float(default)
    return seconds if seconds > 0 else None


class DurableTaskEngine:
    """Create, persist and execute tasks with evidence-backed completion."""

    def __init__(self, settings: Settings, events: EventBus) -> None:
        self.settings = settings
        self.events = events
        self._log = get_logger("tasks.engine")
        # Live worker registry. No process restart is needed: register a handler
        # and call dispatch_ready() at any time.
        self._workers: dict[str, Worker] = {}
        self._default_worker: Worker | None = None
        self._delivery_handlers: dict[str, DeliveryHandler] = {}
        self._default_delivery: DeliveryHandler | None = None
        # task id -> lease token for claims held by this engine instance; a
        # watchdog never recovers work that is still executing in-process.
        self._active_claims: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    async def _reconcile(self, task: TaskRecord) -> TaskRecord:
        """Make ``TaskRecord.checkpoints`` agree with ``checkpoints()``.

        The checkpoint table is authoritative; legacy JSON-only entries are
        kept so no history is lost by a read.
        """
        rows = await db.fetchall(
            "SELECT seq,state,created_at FROM task_checkpoints WHERE task_id=? ORDER BY seq",
            (task.id,),
        )
        if rows or task.checkpoints:
            task.checkpoints = _merge_checkpoints(task.checkpoints, rows)
        return task

    async def _load(self, task_id: str) -> TaskRecord:
        row = await db.fetchone("SELECT * FROM tasks WHERE id=?", (task_id,))
        if row is None:
            raise KeyError(f"unknown task: {task_id}")
        return await self._reconcile(_row_to_task(row))

    async def _write_task(
        self, conn: Any, task: TaskRecord, *, require_status: str | None = None
    ) -> TaskRecord:
        """Write one task row on ``conn`` with compare-and-swap on its revision.

        Callers own the transaction; a zero-row update aborts their whole unit
        of work (important for atomic checkpoints and claims).
        """
        task.updated_at = utcnow()
        previous_version = task.version
        task.version += 1
        sql = """UPDATE tasks SET objective=?, status=?, plan=?, current_step=?, history=?,
                 dependencies=?, checkpoints=?, result=?, errors=?, retry_count=?,
                 resource_limits=?, cancellation_requested=?, owner_agent=?, completed_at=?,
                 evidence=?, metadata=?, version=? WHERE id=? AND version=?"""
        params: list[Any] = [
            task.objective,
            task.status.value,
            _dumps(task.plan),
            task.current_step,
            _dumps(task.history),
            _dumps(task.dependencies),
            _dumps(task.checkpoints),
            _dumps(task.result),
            _dumps(task.errors),
            task.retry_count,
            _dumps(task.resource_limits),
            int(task.cancellation_requested),
            task.owner_agent,
            task.completed_at,
            _dumps(task.evidence),
            _dumps(task.metadata),
            task.version,
            task.id,
            previous_version,
        ]
        if require_status is not None:
            sql += " AND status=?"
            params.append(require_status)
        cursor = await conn.execute(sql, params)
        if cursor.rowcount != 1:
            task.version = previous_version
            raise TaskConflictError(
                f"task {task.id} was modified concurrently; reload and re-apply"
            )
        return task

    async def _save(self, task: TaskRecord) -> TaskRecord:
        """Persist ``task`` with compare-and-swap on the row revision.

        A zero-row update means another writer saved this task after it was
        loaded; the change is refused instead of silently overwriting it
        (a lost update could, for example, erase a cancellation flag).
        """
        async with db.transaction() as conn:
            return await self._write_task(conn, task)

    async def _emit(self, topic: str, task: TaskRecord | Mapping[str, Any] | str, **payload: Any) -> None:
        if self.events is None:
            return
        if isinstance(task, str):
            task_id, status_value = task, None
        elif isinstance(task, Mapping):
            task_id = task.get("id")
            status = task.get("status")
            status_value = status.value if isinstance(status, TaskStatus) else status
        else:
            task_id, status_value = task.id, task.status.value
        data: dict[str, Any] = {"task_id": task_id}
        if status_value is not None:
            data["status"] = status_value
        data.update(payload)
        await self.events.emit(topic, **data)

    @staticmethod
    def _apply_transition(task: TaskRecord, status: TaskStatus | str, note: str | None) -> str:
        target = TaskStatus(status)
        previous = task.status.value
        if target is not task.status and target not in ALLOWED_TRANSITIONS[task.status]:
            raise ValueError(f"illegal task transition: {task.status.value} -> {target.value}")
        if target is not task.status:
            task.status = target
            _history(task, "transition", **{"from": previous}, to=target.value, note=note)
        else:
            _history(task, "status", **{"from": previous}, to=target.value, note=note or "unchanged")
        if target is TaskStatus.COMPLETED:
            task.completed_at = utcnow()
        return previous

    async def _emit_transition(
        self, task: TaskRecord, previous: str, note: str | None, *, changed: bool = True
    ) -> None:
        await self._emit(
            "task.status_changed", task, from_status=previous, to_status=task.status.value, note=note
        )
        await self._emit(
            "task.state_changed", task, from_status=previous, to_status=task.status.value, note=note
        )
        if changed:
            await self._emit(f"task.{task.status.value}", task, note=note)
            if task.status is TaskStatus.RUNNING:
                await self._emit("task.started", task, note=note)

    # ------------------------------------------------------------------
    # Public task API
    # ------------------------------------------------------------------
    async def create(
        self,
        objective: str,
        plan: list[str] | None = None,
        dependencies: list[str] | None = None,
        owner_agent: str | None = None,
        resource_limits: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> TaskRecord:
        """Create a task, deduplicating on ``idempotency_key`` via the kv table.

        The kv claim and the task insert share one ``BEGIN IMMEDIATE``
        transaction, so a concurrent creator with the same key can never
        observe a claimed-but-absent task and never "reclaims" a live claim.
        """
        if not objective or not str(objective).strip():
            raise ValueError("objective must not be empty")

        task_id = new_id("task")
        meta = dict(metadata or {})
        if idempotency_key is not None and str(idempotency_key).strip():
            meta["idempotency_key"] = str(idempotency_key)
        elif idempotency_key is not None:
            raise ValueError("idempotency_key must not be empty")

        now = utcnow()
        task = TaskRecord(
            id=task_id,
            objective=str(objective),
            plan=list(plan or []),
            dependencies=list(dependencies or []),
            owner_agent=owner_agent,
            resource_limits=dict(resource_limits or {}),
            metadata=meta,
            created_at=now,
            updated_at=now,
        )
        _history(task, "created", note="task created", idempotency_key=idempotency_key)

        duplicate: TaskRecord | None = None
        async with db.transaction() as conn:
            if idempotency_key is not None:
                key = f"task:idempotency:{idempotency_key}"
                await conn.execute(
                    "INSERT OR IGNORE INTO kv(key,value,updated_at) VALUES(?,?,?)",
                    (key, task_id, now),
                )
                row = await (await conn.execute("SELECT value FROM kv WHERE key=?", (key,))).fetchone()
                winner = str(row["value"]) if row is not None else None
                if winner != task_id:
                    existing_row = await (
                        await conn.execute("SELECT * FROM tasks WHERE id=?", (winner,))
                    ).fetchone()
                    if existing_row is not None:
                        duplicate = _row_to_task(existing_row)
                    else:
                        # Stale claim whose task row is gone; reclaim atomically.
                        await conn.execute("DELETE FROM kv WHERE key=? AND value=?", (key, winner))
                        await conn.execute(
                            "INSERT OR IGNORE INTO kv(key,value,updated_at) VALUES(?,?,?)",
                            (key, task_id, now),
                        )
            if duplicate is None:
                await conn.execute(
                    """INSERT INTO tasks(id,objective,status,plan,current_step,history,dependencies,
                       checkpoints,result,errors,retry_count,resource_limits,cancellation_requested,
                       owner_agent,created_at,updated_at,completed_at,evidence,metadata)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        task.id,
                        task.objective,
                        task.status.value,
                        _dumps(task.plan),
                        task.current_step,
                        _dumps(task.history),
                        _dumps(task.dependencies),
                        _dumps(task.checkpoints),
                        _dumps(task.result),
                        _dumps(task.errors),
                        task.retry_count,
                        _dumps(task.resource_limits),
                        0,
                        task.owner_agent,
                        task.created_at,
                        task.updated_at,
                        None,
                        _dumps(task.evidence),
                        _dumps(task.metadata),
                    ),
                )

        if duplicate is not None:
            self._log.debug("duplicate task creation suppressed", extra={"task_id": duplicate.id})
            await self._emit("task.duplicate", duplicate, idempotency_key=idempotency_key)
            return duplicate
        await self._emit("task.created", task, idempotency_key=idempotency_key)
        return task

    async def get(self, task_id: str) -> TaskRecord | None:
        row = await db.fetchone("SELECT * FROM tasks WHERE id=?", (task_id,))
        return await self._reconcile(_row_to_task(row)) if row is not None else None

    async def list(self, status: TaskStatus | str | None = None, limit: int = 100) -> list[TaskRecord]:
        if status is None:
            rows = await db.fetchall("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            rows = await db.fetchall(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (TaskStatus(status).value, limit),
            )
        return [await self._reconcile(_row_to_task(row)) for row in rows]

    @staticmethod
    def _ensure_mutable(task: TaskRecord, action: str) -> None:
        """Terminal and cancelled rows are absorbing for all public writes."""
        if task.status.terminal:
            suffix = " (cancellation was finalized)" if task.status is TaskStatus.CANCELLED else ""
            raise ValueError(f"cannot {action} terminal task: {task.status.value}{suffix}")

    @staticmethod
    def _settle_cancelled(task: TaskRecord, reason: str) -> str:
        """Finalize CANCELLED (never leave a cancelled task claimable)."""
        previous = task.status.value
        task.cancellation_requested = True
        if task.status is not TaskStatus.CANCELLED:
            DurableTaskEngine._apply_transition(task, TaskStatus.CANCELLED, reason)
        task.completed_at = task.completed_at or utcnow()
        _history(task, "cancelled", note=reason)
        return previous

    async def cancel(self, task_id: str) -> bool:
        """Request cancellation; an in-process worker still observes the flag."""
        task = await self._load(task_id)
        if task.status.terminal:
            raise ValueError(f"cannot cancel terminal task: {task.status.value}")
        previous = task.status.value
        task.cancellation_requested = True
        _history(task, "cancellation_requested", note="cancellation requested by caller")
        if task.id not in self._active_claims:
            # No live worker here to observe the flag: settle it now so
            # recovery or dispatch can never resurrect the task.
            self._settle_cancelled(task, "cancelled by caller")
        await self._save(task)
        if task.status is TaskStatus.CANCELLED:
            self._active_claims.pop(task.id, None)
        await self._emit("task.cancellation_requested", task)
        if task.status is TaskStatus.CANCELLED:
            await self._emit_transition(task, previous, "cancelled by caller")
        return True

    async def transition(self, task_id: str, status: TaskStatus | str, note: str | None = None) -> TaskRecord:
        task = await self._load(task_id)
        target = TaskStatus(status)
        if target is TaskStatus.COMPLETED:
            raise ValueError("completion requires evidence: use complete(task_id, result, evidence)")
        if task.cancellation_requested and target is TaskStatus.FAILED:
            target = TaskStatus.CANCELLED  # cancellation wins over failure/dead-letter
        if task.cancellation_requested and target is not TaskStatus.CANCELLED:
            raise ValueError("task cancellation was requested; only cancellation is allowed")
        self._ensure_mutable(task, "transition")
        changed = target is not task.status
        previous = self._apply_transition(task, target, note)
        await self._save(task)
        await self._emit_transition(task, previous, note, changed=changed)
        return task

    async def progress(self, task_id: str, note: str, **details: Any) -> TaskRecord:
        """Record progress in task history and emit ``task.progress``.

        Iteration facts (``iteration``, ``max_iterations``) travel on the event
        too, not only into history: the chat bridge renders `iter 3/10` from
        them. Unknown detail keys stay history-only — the event stays small and
        the renderer never invents a place for them.
        """
        task = await self._load(task_id)
        self._ensure_mutable(task, "record progress on")
        step = details.pop("step", details.pop("current_step", None))
        if step is not None:
            task.current_step = int(step)
        _history(task, "progress", note=note, **details)
        await self._save(task)
        event_details = {
            key: details[key]
            for key in ("iteration", "max_iterations")
            if details.get(key) is not None
        }
        await self._emit(
            "task.progress", task, note=note, current_step=task.current_step, **event_details
        )
        return task

    async def checkpoint(self, task_id: str, state: dict[str, Any]) -> TaskRecord:
        """Atomically persist a checkpoint in the task row and checkpoint table.

        The sequence comes from ``MAX(seq)`` inside the same ``BEGIN IMMEDIATE``
        transaction, so concurrent checkpoints cannot reuse or skip a sequence
        number and a failed insert rolls back the task write with it.
        """
        if not isinstance(state, dict):
            raise ValueError("checkpoint state must be a dict")
        async with db.transaction() as conn:
            row = await (await conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,))).fetchone()
            if row is None:
                raise KeyError(f"unknown task: {task_id}")
            task = _row_to_task(row)
            self._ensure_mutable(task, "checkpoint")
            seq_row = await (
                await conn.execute(
                    "SELECT COALESCE(MAX(seq),0) AS highest FROM task_checkpoints WHERE task_id=?",
                    (task_id,),
                )
            ).fetchone()
            table_seq = int(seq_row["highest"] or 0)
            json_seq = max(
                (int(e.get("seq", 0)) for e in task.checkpoints if isinstance(e, Mapping)),
                default=0,
            )
            seq = max(table_seq, json_seq) + 1
            entry = {"seq": seq, "state": _dumps(state), "created_at": utcnow()}
            task.checkpoints = _merge_checkpoints(task.checkpoints, [entry])
            _history(task, "checkpoint", seq=seq, note="state checkpoint recorded")
            await self._write_task(conn, task)
            await conn.execute(
                "INSERT INTO task_checkpoints(task_id,seq,state,created_at) VALUES(?,?,?,?)",
                (task_id, seq, _dumps(state), entry["created_at"]),
            )
        await self._emit("task.checkpoint", task, seq=seq)
        return task

    async def checkpoints(self, task_id: str) -> list[dict[str, Any]]:
        row = await db.fetchone("SELECT checkpoints FROM tasks WHERE id=?", (task_id,))
        rows = await db.fetchall(
            "SELECT seq,state,created_at FROM task_checkpoints WHERE task_id=? ORDER BY seq", (task_id,)
        )
        return _merge_checkpoints(_loads(row["checkpoints"], []) if row is not None else [], rows)

    async def resume_from_checkpoint(
        self, task_id: str, seq: int | None = None
    ) -> TaskRecord | None:
        """Restore recorded checkpoint state into a live task.

        Used by retries and recovery; it never resumes a terminal or cancelled
        row and never replays a side effect by itself.
        """
        task = await self._load(task_id)
        self._ensure_mutable(task, "resume")
        if task.cancellation_requested:
            return None
        entries = await self.checkpoints(task_id)
        if not entries:
            return None
        entry = entries[-1] if seq is None else next(
            (item for item in entries if item["seq"] == int(seq)), None
        )
        if entry is None:
            raise KeyError(f"no checkpoint {seq} for task {task_id}")
        self._restore_checkpoint(task, entry)
        try:
            await self._save(task)
        except TaskConflictError:
            return None
        await self._emit("task.resumed", task, seq=entry["seq"])
        return task

    @staticmethod
    def _restore_checkpoint(task: TaskRecord, entry: Mapping[str, Any]) -> None:
        state = entry.get("state")
        if not isinstance(state, Mapping):
            return
        step = state.get("current_step", state.get("step"))
        if isinstance(step, int) and not isinstance(step, bool):
            task.current_step = step
        task.metadata["resume_checkpoint_seq"] = int(entry.get("seq", 0))
        _history(task, "resumed_from_checkpoint", seq=entry.get("seq"), note="checkpoint state restored")

    async def complete(
        self,
        task_id: str,
        result: Any,
        evidence: list[dict[str, Any]],
        note: str | None = None,
    ) -> TaskRecord:
        """Mark a task COMPLETED. Evidence is mandatory and cannot be asserted by a model.

        Completion is not cancellable once it starts. A worker that calls
        ``complete()`` and then stalls can hit the task deadline mid-write; if the
        cancellation aborted the write, a finished task would be recorded FAILED.
        The evidence-backed write is allowed to land first, then the cancellation
        is re-raised so shutdown semantics are unchanged.
        """
        pending = asyncio.ensure_future(self._complete_impl(task_id, result, evidence, note))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            await pending            # land the completion, then unwind
            raise

    async def _complete_impl(
        self,
        task_id: str,
        result: Any,
        evidence: list[dict[str, Any]],
        note: str | None = None,
    ) -> TaskRecord:
        evidence = _validate_evidence(evidence)
        task = await self._load(task_id)
        self._ensure_mutable(task, "complete")
        if task.cancellation_requested:
            raise ValueError("cannot complete a task whose cancellation was requested")
        # Evidence-backed completion is allowed from any active state; advance
        # through the legal intermediate states so the state machine stays intact.
        if task.status in (TaskStatus.CREATED, TaskStatus.PLANNING):
            await self.transition(task_id, TaskStatus.READY, note="prepared for completion")
        if task.status in (TaskStatus.CREATED, TaskStatus.PLANNING, TaskStatus.READY):
            await self.transition(task_id, TaskStatus.RUNNING, note="verifying evidence")
            task = await self._load(task_id)
        previous = self._apply_transition(task, TaskStatus.COMPLETED, note or "completed with evidence")
        task.result = result
        task.evidence = evidence
        _history(
            task,
            "completed",
            evidence_count=len(task.evidence),
            note=note or "evidence-backed completion recorded",
        )
        await self._save(task)
        await self._emit_transition(task, previous, note)
        await self._emit("task.evidence", task, evidence_count=len(task.evidence))
        return task

    async def fail(self, task_id: str, error: str) -> TaskRecord:
        task = await self._load(task_id)
        self._ensure_mutable(task, "fail")
        message = str(error) or "task failed"
        if task.cancellation_requested:
            # Cancellation is an operator instruction and outranks a worker's
            # failure report and the dead-letter path.
            previous = self._settle_cancelled(task, f"cancelled before failure: {message}")
            await self._save(task)
            self._active_claims.pop(task.id, None)
            await self._emit_transition(task, previous, message)
            await self._emit("task.cancelled", task, reason=message)
            return task
        previous = self._apply_transition(task, TaskStatus.FAILED, message)
        task.errors.append(message)
        task.metadata["last_error"] = message
        _history(task, "error", error=message)
        await self._save(task)
        await self._emit_transition(task, previous, message)
        await self._emit("task.failed", task, error=message)
        return task

    async def update(
        self,
        task_id: str,
        *,
        result: Any = None,
        metadata: dict[str, Any] | None = None,
        status: TaskStatus | str | None = None,
        note: str | None = None,
    ) -> TaskRecord:
        """Merge metadata/result and optionally move the state machine.

        A public, audited alternative to writing around the engine: metadata
        is **merged** (never replaced), status changes pass through the legal
        transition table, and the save is compare-and-swapped. ``completed_at``
        is never stamped here — only :meth:`complete` does that.
        """
        task = await self._load(task_id)
        self._ensure_mutable(task, "update")
        if status is not None and TaskStatus(status) is TaskStatus.COMPLETED:
            # update() is a bookkeeping API, not an evidence channel. Without
            # this guard a RUNNING/WAITING/BLOCKED/VERIFYING task could be
            # completed with an empty evidence list, bypassing complete().
            raise ValueError(
                f"illegal task transition without evidence: {task.status.value} -> completed; "
                "use complete(task_id, result, evidence)"
            )
        if metadata:
            task.metadata.update(metadata)
        if result is not None:
            task.result = result
        changed = False
        previous = task.status.value
        if status is not None and TaskStatus(status) is not task.status:
            self._apply_transition(task, TaskStatus(status), note or "updated by caller")
            changed = True
        await self._save(task)
        if changed:
            await self._emit_transition(task, previous, note or "updated by caller")
        else:
            await self._emit("task.updated", task, note=note)
        return task

    # ------------------------------------------------------------------
    # Worker execution
    # ------------------------------------------------------------------
    #: Statuses a runner may atomically claim; CREATED/PLANNING hop through READY.
    _CLAIMABLE = frozenset(
        {
            TaskStatus.CREATED,
            TaskStatus.PLANNING,
            TaskStatus.READY,
            TaskStatus.WAITING,
            TaskStatus.BLOCKED,
            TaskStatus.VERIFYING,
        }
    )

    @staticmethod
    def _lease_seconds(task: TaskRecord, override: float | None = None) -> float:
        value = override if override is not None else task.resource_limits.get(
            "lease_seconds", DEFAULT_LEASE_SECONDS
        )
        try:
            return max(1.0, float(value))
        except (TypeError, ValueError):
            return DEFAULT_LEASE_SECONDS

    async def _claim(
        self,
        task_id: str,
        note: str,
        *,
        owner: str | None = None,
        lease_seconds: float | None = None,
    ) -> tuple[TaskRecord, str] | None:
        """Atomically move a claimable task to RUNNING with a lease.

        The read, the state-machine hops and the conditional write share one
        ``BEGIN IMMEDIATE`` transaction, so exactly one caller can claim a
        task. Cancelled tasks are refused and finalized, never reclaimed.
        """
        token = str(owner or new_id("lease"))
        result: tuple[TaskRecord, str] | None = None
        cancelled = False
        cancelled_from = ""
        async with db.transaction() as conn:
            row = await (await conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,))).fetchone()
            if row is None:
                raise KeyError(f"unknown task: {task_id}")
            task = _row_to_task(row)
            previous = task.status.value
            if task.cancellation_requested:
                if task.status is not TaskStatus.CANCELLED:
                    cancelled_from = self._settle_cancelled(task, "cancelled before claim")
                    await self._write_task(conn, task)
                    cancelled = True
                self._active_claims.pop(task_id, None)
            elif task.status not in self._CLAIMABLE:
                return None
            else:
                now = utcnow()
                if task.status in (TaskStatus.CREATED, TaskStatus.PLANNING):
                    task.status = TaskStatus.READY
                    _history(task, "transition", **{"from": previous}, to="ready", note="prepared for execution")
                final_from = task.status.value
                task.status = TaskStatus.RUNNING
                task.updated_at = now
                _history(task, "transition", **{"from": final_from}, to="running", note=note)
                seconds = self._lease_seconds(task, lease_seconds)
                metadata = dict(task.metadata or {})
                metadata["lease_owner"] = token
                metadata["lease_seconds"] = seconds
                metadata["lease_heartbeat_at"] = now
                metadata["lease_expires_at"] = _utc_after(seconds)
                task.metadata = metadata
                _history(task, "claimed", lease_owner=token, lease_seconds=seconds)
                await self._write_task(conn, task, require_status=previous)
                self._active_claims[task_id] = token
                result = (task, previous)
        if cancelled:
            await self._emit_transition(task, cancelled_from, "cancelled before claim")
            await self._emit("task.cancelled", task, reason="cancellation requested before claim")
            return None
        return result

    async def _merge_worker_result(self, task_id: str, returned: Any) -> TaskRecord:
        """Store the worker's return value under ``_result`` without losing result data."""
        task = await self._load(task_id)
        if isinstance(task.result, dict):
            wrapped = dict(task.result)
        elif task.result is None:
            wrapped = {}
        else:
            wrapped = {"value": task.result}
        wrapped["_result"] = returned
        task.result = wrapped
        await self._save(task)
        return task

    async def _record_failure(self, task_id: str, error: str) -> TaskRecord:
        """Increment retries or dead-letter the task when retries are exhausted."""
        task = await self._load(task_id)
        if task.status.terminal:
            # A timeout/exception can surface after a worker has already
            # completed or cancelled the task. Terminal states are absorbing:
            # do not append errors or attempt an illegal transition to FAILED.
            return task
        if task.cancellation_requested:
            previous = self._settle_cancelled(task, f"cancelled before failure: {error}")
            await self._save(task)
            self._active_claims.pop(task.id, None)
            await self._emit_transition(task, previous, error)
            await self._emit("task.cancelled", task, reason=error)
            return task
        task.errors.append(error)
        task.metadata["last_error"] = error
        _history(task, "error", error=error)
        max_retries = _max_retries(task.resource_limits)
        if task.retry_count < max_retries:
            task.retry_count += 1
            task.metadata.pop("dead_letter", None)
            _history(task, "retry", attempt=task.retry_count, max_retries=max_retries, error=error)
            await self._save(task)
            await self._emit("task.retry", task, attempt=task.retry_count, error=error)
            return task
        previous = self._apply_transition(task, TaskStatus.FAILED, error)
        task.metadata["dead_letter"] = True
        task.metadata["dead_letter_reason"] = error
        _history(task, "dead_letter", reason=error, retries=task.retry_count)
        await self._save(task)
        await self._emit_transition(task, previous, error)
        await self._emit("task.failed", task, error=error, dead_letter=True)
        await self._emit("task.dead_letter", task, error=error, retries=task.retry_count)
        return task

    async def run(self, task_id: str, worker: Worker) -> TaskRecord:
        """Execute ``worker(task, engine)`` with timeout, retries and cancellation.

        A worker may produce intermediate results, checkpoints and progress, but
        only evidence-backed completion (explicit ``complete()`` or a returned
        payload with ``evidence``) can COMPLETE the task.
        """
        task = await self._load(task_id)
        if task.status.terminal:
            # Idempotent re-invocation: never replay a cancelled/completed task.
            return task
        limits = task.resource_limits or {}
        timeout = _timeout(limits, self.settings.automation.default_task_timeout_seconds)
        claimed = await self._claim(task_id, note="worker started")
        if claimed is None:
            # Another runner already holds the claim (or the task moved on):
            # never start a second worker for the same task.
            self._log.debug("task claim lost; worker not started", extra={"task_id": task_id})
            return await self.get(task_id)  # type: ignore[return-value]
        claimed_task, previous = claimed
        await self._emit_transition(claimed_task, previous, "worker started")
        try:
            return await self._execute_claimed(
                task_id, worker, timeout=timeout, context=self._worker_context(claimed_task)
            )
        finally:
            self._active_claims.pop(task_id, None)

    async def _execute_claimed(
        self,
        task_id: str,
        worker: Worker,
        *,
        timeout: float | None,
        context: Mapping[str, Any],
    ) -> TaskRecord:
        """Run an already-claimed worker; the caller owns claim release."""
        returned: Any = None
        while True:
            # Fresh state per attempt: workers see checkpoints, retries and cancellations.
            attempt_task = await self._load(task_id)
            if attempt_task.status.terminal:
                break
            if attempt_task.cancellation_requested:
                if not attempt_task.status.terminal:
                    await self.transition(task_id, TaskStatus.CANCELLED, note="cancellation requested before run")
                break
            try:
                call = _call_worker(worker, attempt_task, self, context)
                returned = await asyncio.wait_for(call, timeout=timeout)
            except asyncio.TimeoutError:
                await self._record_failure(task_id, f"task timed out after {timeout} seconds")
            except asyncio.CancelledError:
                current = await self.get(task_id)
                if current is not None and not current.status.terminal:
                    await self.transition(task_id, TaskStatus.CANCELLED, note="worker cancelled")
                break
            except Exception as exc:  # noqa: BLE001 - worker errors are task errors
                await self._record_failure(task_id, f"{type(exc).__name__}: {exc}")
            else:
                finished = await self.get(task_id)
                if finished is not None and finished.status is TaskStatus.COMPLETED:
                    await self._merge_worker_result(task_id, returned)
                    await self._emit("task.progress", task_id, note="worker returned")
                    return await self.get(task_id)  # type: ignore[return-value]
                if finished is not None and (
                    finished.cancellation_requested or finished.status.terminal
                ):
                    # The worker observed the cancellation flag, or it finished
                    # the task itself (complete/fail/cancel). Never attempt a
                    # VERIFYING transition from a terminal state.
                    if not finished.status.terminal:
                        await self.transition(
                            task_id, TaskStatus.CANCELLED, note="worker observed cancellation"
                        )
                    break
                evidence = self._returned_evidence(returned)
                if evidence is not None and not (
                    isinstance(returned, Mapping) and returned.get("success") is False
                ):
                    result = returned.get("result", returned.get("output"))
                    try:
                        await self.complete(task_id, result, evidence, note="worker evidence")
                    except ValueError as exc:
                        await self.transition(task_id, TaskStatus.VERIFYING, note=str(exc))
                        await self._record_failure(task_id, str(exc))
                    else:
                        await self._merge_worker_result(task_id, returned)
                        await self._emit("task.progress", task_id, note="worker evidence accepted")
                        return await self.get(task_id)  # type: ignore[return-value]
                    continue
                error = (
                    "worker returned without evidence-backed completion; "
                    "call complete(task_id, result, evidence) before returning"
                )
                if isinstance(returned, Mapping) and returned.get("success") is False:
                    error = f"worker reported failure: {returned.get('error', 'no reason given')}"
                    await self.fail(task_id, error)
                    continue
                await self.transition(task_id, TaskStatus.VERIFYING, note=error)
                await self._record_failure(task_id, error)

            current = await self.get(task_id)
            if current is None or current.status.terminal:
                # A late timeout/error may surface after the worker already
                # completed or cancelled the task; never retry an absorbing
                # state (COMPLETED -> READY would be an illegal transition).
                break
            if current.cancellation_requested:
                if not current.status.terminal:
                    await self.transition(task_id, TaskStatus.CANCELLED, note="cancelled during retry")
                break
            await self.resume_from_checkpoint(task_id)  # retries resume recorded state
            await self.transition(task_id, TaskStatus.READY, note="retry scheduled")
            claimed = await self._claim(task_id, note=f"retry attempt {current.retry_count}")
            if claimed is None:
                break  # another runner took over this attempt
            claimed_task, previous = claimed
            await self._emit_transition(claimed_task, previous, f"retry attempt {current.retry_count}")

        await self._merge_worker_result(task_id, returned)
        return await self.get(task_id)  # type: ignore[return-value]

    @staticmethod
    def _returned_evidence(returned: Any) -> list[Any] | None:
        if isinstance(returned, Mapping):
            evidence = returned.get("evidence")
            if isinstance(evidence, (list, tuple)) and evidence:
                return list(evidence)
        return None

    # ------------------------------------------------------------------
    # Worker registry, live dispatch and delivery
    # ------------------------------------------------------------------
    def register_worker(self, name: str, worker: Worker | None = None) -> Worker:
        """Register a dispatch worker for ``owner_agent`` (or a named key).

        Can also be used as a decorator when ``worker`` is omitted.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("worker name must not be empty")
        if worker is None:
            return lambda implementation: self.register_worker(name, implementation)  # type: ignore[return-value]
        if not callable(worker):
            raise TypeError("worker must be callable")
        self._workers[name] = worker
        return worker

    def unregister_worker(self, name: str) -> bool:
        return self._workers.pop(name, None) is not None

    def register_default_worker(self, worker: Worker) -> Worker:
        if not callable(worker):
            raise TypeError("worker must be callable")
        self._default_worker = worker
        return worker

    def register_delivery(self, name: str, handler: DeliveryHandler) -> DeliveryHandler:
        if not isinstance(name, str) or not name.strip() or not callable(handler):
            raise ValueError("delivery handlers need a non-empty name and a callable")
        self._delivery_handlers[name] = handler
        return handler

    def register_default_delivery(self, handler: DeliveryHandler) -> DeliveryHandler:
        if not callable(handler):
            raise TypeError("delivery handler must be callable")
        self._default_delivery = handler
        return handler

    def _worker_context(self, task: TaskRecord) -> dict[str, Any]:
        payload = task.metadata.get("schedule_payload")
        targets = _delivery_targets(task)
        return {
            "task_id": task.id,
            "objective": task.objective,
            "owner_agent": task.owner_agent,
            "status": task.status.value,
            "schedule_payload": dict(payload) if isinstance(payload, Mapping) else None,
            "notify_targets": dict(targets),
            "delivery_targets": dict(targets),
            "metadata": dict(task.metadata),
        }

    def _resolve_worker(self, task: TaskRecord) -> Worker | None:
        candidates: list[str] = []
        if task.owner_agent:
            candidates.extend((task.owner_agent, f"agent:{task.owner_agent}", f"owner:{task.owner_agent}"))
        payload = task.metadata.get("schedule_payload")
        if isinstance(payload, Mapping):
            for key in ("worker", "handler", "owner_agent", "agent"):
                value = payload.get(key)
                if isinstance(value, str):
                    candidates.append(value)
            candidates.append(SCHEDULE_WORKER)
        if _delivery_targets(task):
            candidates.append(NOTIFY_WORKER)
        candidates.append(DEFAULT_WORKER)
        for name in candidates:
            worker = self._workers.get(name)
            if worker is not None:
                return worker
            if name == DEFAULT_WORKER:
                return self._default_worker
        return None

    @staticmethod
    def _is_dispatchable(task: TaskRecord) -> bool:
        if task.owner_agent or task.metadata.get("schedule_payload") is not None:
            return True
        return bool(_delivery_targets(task))

    async def dispatch_ready(
        self, limit: int = 50, *, lease_seconds: float | None = None
    ) -> list[TaskRecord]:
        """Claim and execute ready tasks that declare an owner, schedule or delivery.

        Selection is restricted to CREATED/READY, and each task is moved to
        RUNNING through the atomic claim before its worker starts. Concurrent
        or repeated calls can therefore never execute the same task twice.
        Registered delivery targets are invoked once the result is terminal.
        """
        if limit <= 0:
            return []
        rows = await db.fetchall(
            "SELECT * FROM tasks WHERE status IN ('created','ready') AND cancellation_requested=0 "
            "ORDER BY created_at LIMIT ?",
            (max(limit * 5, limit),),
        )
        dispatched: list[TaskRecord] = []
        for row in rows:
            if len(dispatched) >= limit:
                break
            task = await self._reconcile(_row_to_task(row))
            if not self._is_dispatchable(task):
                continue
            worker = self._resolve_worker(task)
            if worker is None:
                await self._emit("task.dispatch_skipped", task, reason="no worker registered")
                continue
            claim = await self._claim(
                task.id, note="dispatch", owner="dispatch", lease_seconds=lease_seconds
            )
            if claim is None:
                continue
            claimed_task, previous = claim
            await self._emit_transition(claimed_task, previous, "dispatch claimed")
            timeout = _timeout(
                claimed_task.resource_limits or {},
                self.settings.automation.default_task_timeout_seconds,
            )
            try:
                finished = await self._execute_claimed(
                    task.id, worker, timeout=timeout, context=self._worker_context(claimed_task)
                )
            finally:
                self._active_claims.pop(task.id, None)
            if finished.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                await self._deliver_result(finished)
            dispatched.append(finished)
        return dispatched

    @staticmethod
    async def _call_delivery(
        handler: DeliveryHandler,
        task: TaskRecord,
        channel: str,
        target: Any,
        context: Mapping[str, Any],
    ) -> Any:
        arity = _worker_arity(handler)
        if arity <= 1:
            outcome = handler(task)
        elif arity == 2:
            outcome = handler(task, target)
        elif arity == 3:
            outcome = handler(task, channel, target)
        else:
            outcome = handler(task, channel, target, context)
        return await outcome if inspect.isawaitable(outcome) else outcome

    async def _deliver_result(self, task: TaskRecord) -> list[dict[str, Any]]:
        targets = _delivery_targets(task)
        if not targets:
            return []
        receipts = dict(task.metadata.get("task.delivery") or {})
        delivered: list[dict[str, Any]] = []
        context = self._worker_context(task)
        for channel, target in targets.items():
            prior = receipts.get(channel)
            if isinstance(prior, Mapping) and prior.get("status") == "sent":
                continue  # already delivered exactly once
            handler = self._delivery_handlers.get(channel) or self._default_delivery
            if handler is None:
                receipt = await self._record_delivery(
                    task.id, channel, "skipped", "no delivery handler registered", target
                )
            else:
                try:
                    outcome = await self._call_delivery(handler, task, channel, target, context)
                    detail = outcome if isinstance(outcome, (str, int, float, bool)) else None
                    receipt = await self._record_delivery(
                        task.id, channel, "sent", detail, target
                    )
                    await self._emit("task.delivered", task, channel=channel, target=target)
                except Exception as exc:  # noqa: BLE001 - delivery failures are durable
                    receipt = await self._record_delivery(
                        task.id, channel, "failed", f"{type(exc).__name__}: {exc}", target
                    )
                    await self._emit(
                        "task.delivery_failed", task, channel=channel, target=target, error=receipt["detail"]
                    )
            delivered.append(receipt)
        return delivered

    async def _record_delivery(
        self, task_id: str, channel: str, status: str, detail: Any, target: Any = None
    ) -> dict[str, Any]:
        """Persist a delivery receipt, including on terminal (absorbing) rows."""
        receipt = {"status": status, "target": target, "detail": detail, "at": utcnow()}
        task = await self._load(task_id)
        receipts = dict(task.metadata.get("task.delivery") or {})
        receipts[channel] = receipt
        task.metadata["task.delivery"] = receipts
        _history(task, "delivery", channel=channel, status=status, detail=detail)
        try:
            await self._save(task)
        except TaskConflictError:
            self._log.debug("delivery receipt lost a concurrent save", extra={"task_id": task_id})
        return receipt

    # ------------------------------------------------------------------
    # Claim lease, heartbeat and watchdog recovery
    # ------------------------------------------------------------------
    async def heartbeat(self, task_id: str, *, owner: str | None = None) -> TaskRecord | None:
        """Renew a RUNNING claim's lease; refuse leases held by another worker."""
        task = await self._load(task_id)
        if task.status is not TaskStatus.RUNNING or task.cancellation_requested:
            return None
        lease_owner = task.metadata.get("lease_owner")
        token = owner or self._active_claims.get(task_id)
        if lease_owner and token and lease_owner != token:
            return None
        seconds = self._lease_seconds(task)
        now = utcnow()
        metadata = dict(task.metadata)
        metadata["lease_owner"] = token or lease_owner or new_id("lease")
        metadata["lease_heartbeat_at"] = now
        metadata["lease_expires_at"] = _utc_after(seconds)
        task.metadata = metadata
        _history(task, "heartbeat", lease_owner=metadata["lease_owner"])
        try:
            await self._save(task)
        except TaskConflictError:
            return None
        await self._emit("task.heartbeat", task, lease_owner=metadata["lease_owner"])
        return task

    async def recover_stale(
        self, *, lease_seconds: float | None = None, limit: int | None = None
    ) -> list[TaskRecord]:
        """Recover RUNNING tasks whose claim lease expired (live watchdog path).

        Tasks with a live heartbeat (or a claim held by this engine instance)
        are never touched. Call this from a watchdog loop together with
        :meth:`dispatch_ready`.
        """
        fallback = DEFAULT_LEASE_SECONDS if lease_seconds is None else max(1.0, float(lease_seconds))
        now = datetime.now(timezone.utc)
        rows = await db.fetchall("SELECT * FROM tasks WHERE status='running' ORDER BY created_at")
        stale: list[Any] = []
        for row in rows:
            task = _row_to_task(row)
            if task.id in self._active_claims:
                continue
            expiry = _parse_timestamp(task.metadata.get("lease_expires_at"))
            if expiry is None:
                basis = _parse_timestamp(task.metadata.get("lease_heartbeat_at")) or _parse_timestamp(
                    task.updated_at
                )
                if basis is None:
                    continue
                expiry = basis + timedelta(seconds=fallback)
            if expiry <= now:
                stale.append(row)
                if limit is not None and len(stale) >= limit:
                    break
        return await self._recover_rows(stale, source="stale lease")

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    async def recover_interrupted(self) -> list[TaskRecord]:
        """Recover tasks left RUNNING/WAITING/VERIFYING/PLANNING by a crash.

        Cancellation is finalized first. Any recorded side-effect signal
        (idempotency key, checkpoints, evidence, result or delivery receipt)
        forces BLOCKED with the signals named in the reason; only clean tasks
        return to READY. Recovery never invokes a worker by itself.
        """
        rows = await db.fetchall(
            "SELECT * FROM tasks WHERE status IN ('running','waiting','verifying','planning') "
            "OR (status='cancelled' AND completed_at IS NULL) ORDER BY created_at"
        )
        return await self._recover_rows(rows, source="interruption")

    async def _recover_rows(
        self, rows: list[Any], *, source: str
    ) -> list[TaskRecord]:
        recovered: list[TaskRecord] = []
        for row in rows:
            task = await self._reconcile(_row_to_task(row))
            previous = task.status.value
            signals = _replay_signals(task)
            if task.cancellation_requested or task.status is TaskStatus.CANCELLED:
                target = TaskStatus.CANCELLED
                reason = (
                    "cancellation requested; replay signals: "
                    + (", ".join(signals) if signals else "none")
                )
            elif signals:
                target = TaskStatus.BLOCKED
                reason = (
                    f"recovered after interruption: {', '.join(signals)} present; "
                    "verify side effects before replaying"
                )
            else:
                target = TaskStatus.READY
                reason = "recovered after interruption: no recorded side effects, safe to reschedule"
            if target is TaskStatus.CANCELLED:
                self._settle_cancelled(task, reason)
            else:
                self._apply_transition(task, target, reason)
            if target is TaskStatus.BLOCKED and task.checkpoints:
                # A real resume: recorded step state is restored for the
                # inspector/retry path instead of being silently discarded.
                self._restore_checkpoint(task, task.checkpoints[-1])
            for key in ("lease_owner", "lease_expires_at", "lease_heartbeat_at", "lease_seconds"):
                task.metadata.pop(key, None)
            task.metadata["recovery_reason"] = reason
            task.metadata["recovered_at"] = utcnow()
            task.metadata["recovered_from"] = previous
            task.metadata["recovery_source"] = source
            _history(task, "recovered", **{"from": previous}, to=target.value, reason=reason)
            self._active_claims.pop(task.id, None)
            try:
                await self._save(task)
            except TaskConflictError:
                # A live worker or another recoverer moved this row after the
                # scan. Skip it instead of aborting recovery of the others.
                self._log.debug(
                    "recovery skipped a concurrently modified task", extra={"task_id": task.id}
                )
                continue
            await self._emit_transition(
                task, previous, reason, changed=(target is not TaskStatus.CANCELLED)
            )
            await self._emit("task.recovered", task, reason=reason, replayed=False, source=source)
            if target is TaskStatus.CANCELLED:
                await self._emit("task.cancelled", task, reason=reason)
            if target is TaskStatus.BLOCKED and task.checkpoints:
                await self._emit("task.resumed", task, seq=task.checkpoints[-1]["seq"])
            recovered.append(task)
        if recovered:
            self._log.info("recovered stranded tasks", extra={"count": len(recovered), "source": source})
        return recovered
