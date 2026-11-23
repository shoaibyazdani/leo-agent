"""Task progress reporting: bridge task-engine events to the origin chat.

LEO already has everything except the connection:

* ``DurableTaskEngine`` emits ``task.progress`` / ``task.recovered`` /
  ``task.resumed`` and persists state, leases, checkpoints.
* ``ProgressReporter`` owns the one-edited-message presentation on a channel.
* ``EventBus`` delivers events in-process.

What was missing: nothing subscribed ``task.*`` with a *channel* in mind, so a
task could run for twenty minutes in total silence while the operator stared at
a typing indicator. This module is that subscriber.

Design rules, mirroring the engine it serves:

* One hub instance, registered once. It is not a second task engine and holds
  no task state of its own beyond an id → reporter mapping.
* A reporter is opened lazily, on the first event that says real work started
  (``task.running``), and only when the task declares an origin
  (``metadata.origin`` = {channel, target}). Tasks without an origin — scheduler
  internals, webhook jobs — stay silent by design.
* Every reporter is independent: two concurrent tasks in one chat are two
  edited messages, each with its own elapsed time and steps.
* The hub never raises into the event bus. A chat that cannot be written to
  must not take down task execution.
* Heartbeats come from the reporter's own ticker (``idle_seconds``), which
  refreshes elapsed time without inventing progress — the same property the
  chat turn reporter already has.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from ..integrations.progress import ProgressReporter
from ..logging import get_logger

_LOG = get_logger("tasks.reporting")

# The origin declaration lives in task.metadata under this key. It is written
# by task_create when the operator asks for background execution, and by
# schedule payloads that already know their channel.
ORIGIN_KEY = "origin"

# Details the renderer knows how to show. Anything else is kept in history but
# does not get a dedicated line.
_ITERATION_KEYS = ("iteration", "max_iterations")


def _origin_of(task: Any) -> tuple[str, str] | None:
    """Return (channel, target) a task should report to, or None.

    Deliberately strict: an origin without both a channel and a target is not
    an origin, and a missing origin means silence, not a guess.
    """
    metadata = getattr(task, "metadata", None) or {}
    if isinstance(metadata, Mapping):
        origin = metadata.get(ORIGIN_KEY)
        if isinstance(origin, Mapping):
            channel = str(origin.get("channel") or "").strip()
            target = str(origin.get("target") or "").strip()
            if channel and target:
                return channel, target
    return None


def _note_of(event: Any) -> str:
    """A one-line human note from an event payload."""
    payload = getattr(event, "payload", None) or {}
    if isinstance(payload, Mapping):
        note = str(payload.get("note") or "").strip()
        if note:
            return note
    return ""


class TaskProgressHub:
    """Owns one ``ProgressReporter`` per running task with a chat origin."""

    def __init__(self, gateway: Any, settings: Any, events: Any) -> None:
        self.gateway = gateway
        self.settings = settings
        self.events = events
        self._reporters: dict[str, ProgressReporter] = {}
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Subscribe to task lifecycle events. Idempotent."""
        topics = ("task.running", "task.progress", "task.recovered", "task.resumed")
        for topic in topics:
            self.events.subscribe(topic, self._on_event)

    async def stop(self) -> None:
        """Close every open reporter. Call at shutdown so no ⏳ outlives us."""
        for task_id in list(self._reporters):
            await self._close(task_id, "done")

    # -- event bridge ------------------------------------------------------
    async def _on_event(self, event: Any) -> None:
        """Route one task event to its reporter. Never raises."""
        try:
            await self._handle(event)
        except Exception as exc:  # noqa: BLE001 - reporting must not break tasks
            _LOG.debug(
                "task progress event dropped",
                extra={"error": type(exc).__name__, "detail": str(exc)[:160]},
            )

    async def _handle(self, event: Any) -> None:
        topic = str(getattr(event, "topic", "") or "")
        payload = getattr(event, "payload", None) or {}
        if not isinstance(payload, Mapping):
            payload = {}
        task_id = str(payload.get("task_id") or getattr(event, "task_id", "") or "")
        if not task_id:
            return
        if topic == "task.running":
            task = await self._load_task(task_id)
            await self._open(task_id, task)
            return
        if topic == "task.progress":
            reporter = self._reporters.get(task_id)
            if reporter is None:
                # Progress for a task that never reported (no origin, or the
                # RUNNING event arrived after this one) is history, not chat.
                return
            note = _note_of(event)
            if note:
                reporter.note("task_note", {"query": note}, True)
            iteration = payload.get("iteration")
            if iteration is not None:
                maximum = payload.get("max_iterations")
                reporter.set_iteration(
                    int(iteration), max_iterations=int(maximum) if maximum is not None else None
                )
            # A progress event is a meaningful milestone: flush it now rather
            # than waiting for the next tick, or the operator sees the step only
            # after the interval has passed.
            await reporter._flush(force=True)  # noqa: SLF001 - same package
            return
        if topic in ("task.recovered", "task.resumed"):
            # A restart or a checkpoint resume is a fact the operator needs: the
            # work they thought was running was not, and now it is again.
            reporter = self._reporters.get(task_id)
            reason = _note_of(event) or topic.split(".", 1)[1]
            if reporter is not None:
                reporter.note("task_note", {"query": reason}, True)
            return

    async def _load_task(self, task_id: str) -> Any:
        """Fetch the task row; a missing task is silence, never an error."""
        from .. import db

        try:
            row = await db.fetchone("SELECT * FROM tasks WHERE id=?", (task_id,))
        except Exception:  # noqa: BLE001 - reporting is best effort
            return None
        if row is None:
            return None
        metadata: dict[str, Any] = {}
        try:
            import json as _json

            metadata = _json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        return type(
            "TaskRow",
            (),
            {"id": task_id, "status": row["status"], "metadata": metadata},
        )()

    async def _open(self, task_id: str, task: Any) -> None:
        origin = _origin_of(task)
        if origin is None:
            return  # no chat to report to: silence is correct, not a failure
        channel, target = origin
        async with self._lock:
            if task_id in self._reporters:
                return  # recovered tasks re-enter RUNNING; keep the open report
            reporter = ProgressReporter(self.gateway, self.settings)
            # A task earns its message immediately: it *is* long work by
            # definition, unlike a chat turn that may answer in one call.
            reporter.min_steps = 1
            cfg = getattr(self.settings, "progress", None)
            # Heartbeat ceiling and step window come from config so a deployment
            # can widen them without touching code.
            heartbeat = float(getattr(cfg, "task_heartbeat_seconds", 45.0) or 45.0)
            reporter.idle_seconds = heartbeat
            reporter.max_steps = int(getattr(cfg, "task_max_steps", 6) or 6)
            reporter._significant_seen = True  # noqa: SLF001 - same package, explicit
            opened = await reporter.begin(channel, target)
            if not opened:
                return
            self._reporters[task_id] = reporter
            # begin() alone sends nothing: the chat-turn reporter waits for its
            # first tool call and then for the ticker. A task that just entered
            # RUNNING is already the fact worth showing, so open the message
            # now — an explicit flush, not a wait for the next tick.
            reporter.note("task_note", {"query": "started"}, True)
            await reporter._flush(force=True)  # noqa: SLF001 - same package
            _LOG.info(
                "task progress opened",
                extra={"task_id": task_id, "channel": channel},
            )

    async def _close(self, task_id: str, verdict: str) -> None:
        reporter = self._reporters.pop(task_id, None)
        if reporter is not None:
            await reporter.end(verdict)

    # -- engine hooks ------------------------------------------------------
    async def on_terminal(self, task: Any, status: str) -> None:
        """Close the reporter when a task ends. Wired by the delivery path.

        ``status`` is the engine's terminal status; anything not completed is
        reported honestly (``stopped`` for cancelled, ``failed`` otherwise).
        """
        task_id = str(getattr(task, "id", "") or "")
        if not task_id:
            return
        if status == "completed":
            verdict = "done"
        elif status == "cancelled":
            verdict = "stopped"
        else:
            verdict = "failed"
        await self._close(task_id, verdict)

    def stats(self) -> dict[str, Any]:
        return {
            "open_reports": len(self._reporters),
            "task_ids": sorted(self._reporters),
        }
