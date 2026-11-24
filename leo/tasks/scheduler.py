"""Durable scheduler: interval jobs, cron-lite expressions and event triggers."""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .. import db
from ..config import Settings
from ..contracts import TaskRecord, new_id
from ..events import Event, EventBus

_EVERY = re.compile(
    r"^@every\s+(\d+(?:\.\d+)?)\s*"
    r"(s|sec|secs|second|seconds|m|min|mins|minute|minutes"
    r"|h|hour|hours|d|day|days|w|week|weeks)?$",
    re.IGNORECASE,
)

_FIELD_LIMITS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
# Three-letter aliases accepted in the month and weekday fields, so "mon-fri"
# and "jan-mar" work instead of dying on int(). Sunday is 0, matching cron.
_MONTH_NAMES = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_WEEKDAY_NAMES = {
    name: index
    for index, name in enumerate(("sun", "mon", "tue", "wed", "thu", "fri", "sat"))
}


def _field_int(token: str, index: int, expression: str) -> int:
    """Parse one cron field token as a number, or a month/weekday name."""
    text = token.strip().lower()
    if not text:
        raise ValueError(f"empty cron field in {expression!r}")
    if index == 3 and text[:3] in _MONTH_NAMES:
        return _MONTH_NAMES[text[:3]]
    if index == 4 and text[:3] in _WEEKDAY_NAMES:
        return _WEEKDAY_NAMES[text[:3]]
    try:
        return int(text)
    except ValueError:
        raise ValueError(
            f"invalid cron field value in {expression!r}: {token!r} "
            "(use a number, or a three-letter name such as mon or jan)"
        ) from None
# A tick that is late by more than this many occurrences records a truncation
# event instead of creating an unbounded backlog of historical tasks.
_MAX_CATCHUP = 10


def as_utc(value: datetime | str | None) -> datetime:
    """Normalize a datetime/ISO string to an aware UTC datetime."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def every_interval(expression: str) -> timedelta | None:
    match = _EVERY.match(expression.strip())
    if not match:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    if unit.startswith("w"):
        multiplier = 604800.0
    elif unit.startswith("d"):
        multiplier = 86400.0
    elif unit.startswith("m"):
        multiplier = 60.0
    elif unit.startswith("h"):
        multiplier = 3600.0
    else:
        multiplier = 1.0
    delta = timedelta(seconds=amount * multiplier)
    if delta <= timedelta(0):
        # A zero-length period would make next_run() return `after` itself.
        # During catch-up that yields an endless (un-cancellable) loop and
        # schedules that never advance their next_run_at.
        raise ValueError(f"@every interval must be positive: {expression!r}")
    return delta


def parse_cron(expression: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    """Parse a five-field cron-lite expression into minute/hour/day/month/weekday sets."""
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError(f"cron expression must have 5 fields: {expression!r}")
    parsed: list[set[int]] = []
    for index, (field, (low, high)) in enumerate(zip(fields, _FIELD_LIMITS)):
        values: set[int] = set()
        for part in field.split(","):
            base, _, step_text = part.partition("/")
            step = int(step_text) if step_text else 1
            if step <= 0:
                raise ValueError(f"invalid cron step in {expression!r}")
            if base == "*":
                start, end = low, high
            elif "-" in base:
                first, last = base.split("-", 1)
                start, end = _field_int(first, index, expression), _field_int(
                    last, index, expression
                )
            else:
                start = _field_int(base, index, expression)
                if step_text and index == 4 and start == 7:
                    # Weekday 7 is Sunday (0); "7/2" must expand from 0.
                    start = 0
                # "5/10" means 5, 15, 25, ...; without this the step yielded
                # a single value and the job ran at 1/10th of its frequency.
                end = high if step_text else start
            if start > end:
                raise ValueError(f"invalid cron range in {expression!r}")
            values.update(range(start, end + 1, step))
        if index == 4:
            # Cron allows 7 as an alias for Sunday. Do this *before* the
            # bounds check: filtering 7 out first made "* * * * 7" an empty
            # field that could never match.
            values = {0 if value == 7 else value for value in values}
        invalid = sorted(value for value in values if not low <= value <= high)
        if invalid:
            # Silently dropping out-of-range values would turn "0,60 * * * *"
            # into a job that runs only at minute 0.
            raise ValueError(f"cron field out of range in {expression!r}: {invalid}")
        parsed.append(values)
    return tuple(parsed)  # type: ignore[return-value]


def _cron_matches_day(moment: datetime, days: set[int], weekdays: set[int]) -> bool:
    """Whether the date part of ``moment`` satisfies the day/weekday fields."""
    dow = (moment.weekday() + 1) % 7
    days_restricted = days != set(range(1, 32))
    dow_restricted = weekdays != set(range(7))
    if days_restricted and dow_restricted:
        return moment.day in days or dow in weekdays
    if days_restricted:
        return moment.day in days
    if dow_restricted:
        return dow in weekdays
    return True


def next_run(kind: str, expression: str, after: datetime) -> datetime:
    """Return the first run strictly after ``after`` for a schedule definition."""
    if expression.strip().lower().startswith("@every"):
        delta = every_interval(expression)
        if delta is None:
            raise ValueError(f"invalid @every expression: {expression!r}")
        return after + delta
    if kind == "interval":
        try:
            seconds = float(expression)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid interval: {expression!r}") from exc
        if seconds <= 0:
            raise ValueError("interval seconds must be positive")
        delta = timedelta(seconds=seconds)
        if delta <= timedelta(0):
            # timedelta has microsecond resolution; a smaller period would
            # round to zero and stall the schedule forever.
            raise ValueError("interval seconds must be at least one microsecond")
        return after + delta
    fields = parse_cron(expression)
    minutes, hours, days, months, weekdays = fields
    start = after.replace(second=0, microsecond=0)
    day = start.replace(hour=0, minute=0)
    ordered_hours = sorted(hours)
    ordered_minutes = sorted(minutes)
    # Walk by day, not by minute: a leap-day expression (0 0 29 2 *) can only
    # match once every four years, so the old 366-day minute sweep made it
    # permanently unschedulable. The window must span a full leap cycle.
    for _ in range(366 * 4 + 1):
        if day.month in months and _cron_matches_day(day, days, weekdays):
            for hour in ordered_hours:
                for minute in ordered_minutes:
                    candidate = day.replace(hour=hour, minute=minute)
                    if candidate > after:
                        return candidate
        day += timedelta(days=1)
    raise ValueError(f"no matching time found for cron expression: {expression!r}")


def _advance_past_due(
    kind: str, expression: str, cursor: datetime, moment: datetime
) -> tuple[datetime, int]:
    """Return the first occurrence after ``moment`` and the number skipped.

    Fixed-period schedules advance arithmetically: iterating next_run() for a
    very short period (e.g. ``@every 0.001s``) could require millions of steps
    to skip a backlog. Cron expressions have no fixed period and use next_run().
    """
    delta: timedelta | None = None
    if expression.strip().lower().startswith("@every"):
        delta = every_interval(expression)
    elif kind == "interval":
        try:
            seconds = float(expression)
        except (TypeError, ValueError):
            seconds = 0.0
        delta = timedelta(seconds=seconds) if seconds > 0 else None
    if delta is not None and delta > timedelta(0):
        count = int((moment - cursor) / delta) + 1
        # Correct possible float rounding at the boundaries.
        while cursor + count * delta <= moment:
            count += 1
        while count > 0 and cursor + (count - 1) * delta > moment:
            count -= 1
        return cursor + count * delta, count
    skipped = 0
    while cursor <= moment:
        cursor = next_run(kind, expression, cursor)
        skipped += 1
    return cursor, skipped


class Scheduler:
    """Create tasks from durable schedules and event subscriptions."""

    def __init__(self, engine: Any, settings: Settings, events: EventBus) -> None:
        self.engine = engine
        self.settings = settings
        self.events = events
        self._event_handlers: list[tuple[str, Any]] = []
        # Per-process guard against re-emitting terminal events to in-process
        # subscribers. Durable history is deduplicated by deterministic ids.
        self._terminal_events_emitted: set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    async def _register(self, name: str, kind: str, expression: str, objective: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        if not name or not objective:
            raise ValueError("schedule name and objective are required")
        stored = dict(payload or {})
        stored["objective"] = objective
        now = datetime.now(timezone.utc)
        next_at = next_run(kind, expression, now)
        row = await db.fetchone("SELECT * FROM schedules WHERE name=?", (name,))
        if row is None:
            schedule_id = new_id("sch")
            await db.execute(
                """INSERT INTO schedules(id,name,kind,expression,payload,enabled,next_run_at,last_run_at,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (schedule_id, name, kind, expression, _dump(stored), 1, next_at.isoformat(), None, now.isoformat()),
            )
        else:
            schedule_id = row["id"]
            await db.execute(
                """UPDATE schedules SET kind=?,expression=?,payload=?,enabled=1,next_run_at=?
                   WHERE id=?""",
                (kind, expression, _dump(stored), next_at.isoformat(), schedule_id),
            )
        return await self.get(name)

    async def register_interval(
        self, name: str, seconds: float, objective: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("interval seconds must be positive")
        return await self._register(name, "interval", str(seconds), objective, payload)

    async def register_cron(
        self, name: str, expression: str, objective: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Register a cron-lite job such as ``*/5 * * * *`` or ``@every 60s``."""
        if every_interval(expression) is not None:
            return await self._register(name, "interval", expression, objective, payload)
        parse_cron(expression)  # validate early
        return await self._register(name, "cron", expression, objective, payload)

    async def register(
        self, name: str, expression: str | float, objective: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Auto-detect a numeric interval, ``@every`` expression or five-field cron."""
        if isinstance(expression, (int, float)):
            return await self.register_interval(name, float(expression), objective, payload)
        text = str(expression)
        if text.strip().lower().startswith("@every") or every_interval(text) is not None:
            return await self.register_cron(name, text, objective, payload)
        try:
            seconds = float(text)
        except ValueError:
            return await self.register_cron(name, text, objective, payload)
        return await self.register_interval(name, seconds, objective, payload)

    async def get(self, name: str) -> dict[str, Any]:
        row = await db.fetchone("SELECT * FROM schedules WHERE name=?", (name,))
        if row is None:
            raise KeyError(f"unknown schedule: {name}")
        return _row_to_schedule(row)

    async def list_schedules(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM schedules"
        if enabled_only:
            sql += " WHERE enabled=1"
        sql += " ORDER BY created_at"
        return [_row_to_schedule(row) for row in await db.fetchall(sql)]

    async def set_enabled(self, name: str, enabled: bool) -> dict[str, Any]:
        await self.get(name)
        await db.execute("UPDATE schedules SET enabled=? WHERE name=?", (int(bool(enabled)), name))
        updated = await self.get(name)
        await self.events.emit(
            "schedule.enabled" if enabled else "schedule.disabled",
            name=name,
            enabled=bool(enabled),
        )
        return updated

    async def enable(self, name: str) -> dict[str, Any]:
        return await self.set_enabled(name, True)

    async def disable(self, name: str) -> dict[str, Any]:
        return await self.set_enabled(name, False)

    # ------------------------------------------------------------------
    # Ticking
    # ------------------------------------------------------------------
    async def _scheduled_backlog(self) -> int:
        """Non-terminal tasks created by schedules (durable concurrency gate)."""
        row = await db.fetchone(
            """SELECT COUNT(*) AS n FROM tasks
               WHERE status IN ('created','planning','ready','running','waiting','blocked','verifying')
                 AND json_extract(metadata, '$.schedule') IS NOT NULL"""
        )
        return int(row["n"]) if row else 0

    def _concurrency_limit(self) -> int:
        automation = getattr(self.settings, "automation", None)
        try:
            return max(0, int(getattr(automation, "max_concurrent_tasks", 0) or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _occurrence_payload(row: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Extract the objective and engine-create kwargs from a schedule row."""
        payload = _load(row["payload"])
        objective = payload.pop("objective", row["name"])
        create_kwargs: dict[str, Any] = {}
        for key in ("plan", "dependencies", "owner_agent", "resource_limits"):
            if key in payload:
                create_kwargs[key] = payload.pop(key)
        return objective, payload, create_kwargs

    async def tick(self, now: datetime | str | None = None) -> list[TaskRecord]:
        """Create tasks for every enabled schedule due at ``now`` and persist next_run_at.

        Occurrences are anchored to ``scheduled_for`` (not the late tick
        moment), bounded missed occurrences are caught up, and a global
        ``max_concurrent_tasks`` gate defers bursts without losing the due
        occurrence (next_run_at is left untouched when deferred).
        """
        moment = as_utc(now)
        due: list[TaskRecord] = []
        limit = self._concurrency_limit()
        backlog = await self._scheduled_backlog() if limit else 0
        rows = await db.fetchall(
            "SELECT * FROM schedules WHERE enabled=1 AND next_run_at IS NOT NULL ORDER BY created_at"
        )
        for row in rows:
            scheduled_for = as_utc(row["next_run_at"])
            if scheduled_for > moment:
                continue
            if limit and backlog >= limit:
                await self.events.emit(
                    "schedule.deferred", name=row["name"], scheduled_for=scheduled_for.isoformat(),
                    reason="max_concurrent_tasks reached",
                )
                continue  # leave next_run_at in place; retry on the next tick
            objective, payload, create_kwargs = self._occurrence_payload(row)

            # Catch up bounded missed occurrences; each one keeps its own
            # scheduled_for identity (and idempotency key), so a repeated or
            # late tick can never duplicate or silently drop an occurrence.
            cursor = scheduled_for
            skipped = 0
            first_task: TaskRecord | None = None
            start_index = len(due)
            for _ in range(_MAX_CATCHUP):
                if cursor > moment:
                    break
                if limit and backlog >= limit:
                    await self.events.emit(
                        "schedule.deferred", name=row["name"], scheduled_for=cursor.isoformat(),
                        reason="max_concurrent_tasks reached",
                    )
                    cursor = None  # type: ignore[assignment]
                    break
                metadata = {
                    "schedule": row["name"],
                    "schedule_id": row["id"],
                    "scheduled_for": cursor.isoformat(),
                    "schedule_payload": payload,
                }
                task = await self.engine.create(
                    objective,
                    metadata=metadata,
                    idempotency_key=f"schedule:{row['name']}:{cursor.isoformat()}",
                    **create_kwargs,
                )
                due.append(task)
                if first_task is None:
                    first_task = task
                backlog += 1
                cursor = next_run(row["kind"], row["expression"], cursor)
            if cursor is None:
                # Deferred: keep next_run_at so the occurrence is retried.
                continue
            if cursor <= moment:
                # More occurrences were missed than the cap: skip the rest
                # and record the truncation instead of replaying history.
                cursor, skipped = _advance_past_due(
                    row["kind"], row["expression"], cursor, moment
                )
                await self.events.emit(
                    "schedule.catchup_truncated", name=row["name"],
                    skipped=skipped, next_run_at=cursor.isoformat(),
                )
            # Publish the occurrence event *before* advancing next_run_at.
            # A crash between the two leaves the durable event in history; on
            # restart the tick repeats the same (idempotent) publish with the
            # same deterministic id, so history has exactly one occurrence and
            # no second task is created (engine.create is idempotent by key).
            await self.events.publish(
                Event(
                    id=f"schedule-triggered:{row['id']}:{scheduled_for.isoformat()}",
                    topic="schedule.triggered",
                    payload={
                        "name": row["name"],
                        "task_id": first_task.id if first_task else None,
                        "task_count": len(due) - start_index,
                        "scheduled_for": scheduled_for.isoformat(),
                        "next_run_at": cursor.isoformat(),
                        "skipped": skipped,
                    },
                )
            )
            await db.execute(
                "UPDATE schedules SET last_run_at=?, next_run_at=? WHERE id=?",
                (moment.isoformat(), cursor.isoformat(), row["id"]),
            )
        return due

    async def trigger(
        self, name: str, now: datetime | str | None = None
    ) -> list[TaskRecord]:
        """Create exactly one occurrence of the named schedule at ``now``.

        This API exists for manual, per-schedule triggers. It never runs a
        global tick, so no other schedule can fire as a side effect, and the
        occurrence is anchored to the requested time rather than to a future
        ``next_run_at``. The schedule cursor never moves backwards; normal
        occurrences keep their original cadence after a manual trigger.
        """
        row = await db.fetchone("SELECT * FROM schedules WHERE name=?", (name,))
        if row is None:
            raise KeyError(f"unknown schedule: {name}")
        if not row["enabled"]:
            raise ValueError(f"schedule is disabled: {name}")

        moment = as_utc(now)
        limit = self._concurrency_limit()
        if limit and await self._scheduled_backlog() >= limit:
            await self.events.emit(
                "schedule.deferred", name=name, scheduled_for=moment.isoformat(),
                reason="max_concurrent_tasks reached",
            )
            return []

        objective, payload, create_kwargs = self._occurrence_payload(row)
        scheduled_for = moment.isoformat()
        metadata = {
            "schedule": row["name"],
            "schedule_id": row["id"],
            "scheduled_for": scheduled_for,
            "schedule_payload": payload,
        }
        task = await self.engine.create(
            objective,
            metadata=metadata,
            idempotency_key=f"schedule:{row['name']}:{scheduled_for}",
            **create_kwargs,
        )
        following = next_run(row["kind"], row["expression"], moment)
        cursor = as_utc(row["next_run_at"]) if row["next_run_at"] else following
        if following > cursor:
            cursor = following
        await self.events.publish(
            Event(
                id=f"schedule-triggered:{row['id']}:{scheduled_for}",
                topic="schedule.triggered",
                payload={
                    "name": row["name"],
                    "task_id": task.id,
                    "task_count": 1,
                    "scheduled_for": scheduled_for,
                    "next_run_at": cursor.isoformat(),
                    "skipped": 0,
                },
            )
        )
        await db.execute(
            "UPDATE schedules SET last_run_at=?, next_run_at=? WHERE id=?",
            (moment.isoformat(), cursor.isoformat(), row["id"]),
        )
        return [task]

    # ------------------------------------------------------------------
    # Dispatch / delivery drain
    # ------------------------------------------------------------------
    @staticmethod
    def _call_hook(hook: Any, limit: int | None) -> Any:
        """Call an engine hook, passing ``limit`` only when it accepts it."""
        try:
            parameters = inspect.signature(hook).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_limit = "limit" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        return hook(limit=limit) if accepts_limit and limit is not None else hook()

    async def _replay_terminal_events(self, limit: int) -> list[Event]:
        """Emit replay-safe terminal events for scheduled tasks.

        Task lifecycle events are emitted when a worker finishes, but a
        delivery subscriber may be attached later (or the process may restart
        between completion and delivery). These durable, deterministic events
        let the delivery path reconstruct terminal results without replaying
        the task itself. No task state is changed here.
        """
        rows = await db.fetchall(
            """SELECT id,status,metadata,result,completed_at,updated_at FROM tasks
               WHERE status IN ('completed','failed','cancelled')
                 AND json_extract(metadata, '$.schedule') IS NOT NULL
               ORDER BY updated_at DESC LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )
        replayed: list[Event] = []
        for row in rows:
            event_id = f"schedule-terminal:{row['id']}:{row['status']}"
            if event_id in self._terminal_events_emitted:
                continue
            metadata = _load(row["metadata"])
            payload = {
                "task_id": row["id"],
                "status": row["status"],
                "schedule": metadata.get("schedule"),
                "scheduled_for": metadata.get("scheduled_for"),
                "schedule_payload": metadata.get("schedule_payload", {}),
                "result": _load(row["result"]),
                "completed_at": row["completed_at"],
            }
            event = Event(id=event_id, topic="schedule.terminal", payload=payload)
            await self.events.publish(event)
            self._terminal_events_emitted.add(event_id)
            replayed.append(event)
        return replayed

    async def _redeliver_terminal(self, limit: int) -> list[str]:
        """Retry delivery for terminal scheduled tasks that still need it.

        ``dispatch_ready`` delivers results immediately after a worker finishes;
        a restart between completion and delivery (or a delivery handler that
        was not registered yet) would otherwise lose the result. Engine
        delivery APIs persist their own receipts, so repeated drains are safe.
        """
        deliver = getattr(self.engine, "deliver_terminal", None) or getattr(
            self.engine, "_deliver_result", None
        )
        if not callable(deliver):
            return []
        rows = await db.fetchall(
            """SELECT id FROM tasks
               WHERE status IN ('completed','failed')
                 AND json_extract(metadata, '$.schedule') IS NOT NULL
               ORDER BY updated_at DESC LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )
        redelivered: list[str] = []
        for row in rows:
            task = await self.engine.get(row["id"])
            if task is None:
                continue
            receipts = task.metadata.get("task.delivery") or {}
            if receipts and all(
                isinstance(receipt, Mapping) and receipt.get("status") == "sent"
                for receipt in receipts.values()
            ):
                continue
            try:
                parameters = inspect.signature(deliver).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "task" in parameters or (parameters and "limit" not in parameters):
                await deliver(task)
            else:
                await self._call_hook(deliver, limit)
            redelivered.append(task.id)
        return redelivered

    async def drain(self, limit: int | None = None) -> dict[str, Any]:
        """Dispatch ready tasks and deliver terminal results via the engine.

        This is the public hook for the scheduler loop and manual run-now
        endpoints. It delegates all task mutation and delivery receipts to the
        task engine's registered worker/delivery APIs: creating or queueing a
        task NEVER completes it, and this method never calls ``complete``.
        """
        dispatch = getattr(self.engine, "dispatch_ready", None)
        if not callable(dispatch):
            raise RuntimeError("task engine has no registered-worker dispatch API (dispatch_ready)")
        batch = limit or 100
        outcome: dict[str, Any] = {
            "dispatch": await self._call_hook(dispatch, limit),
            "delivery": None,
            "redelivered": [],
            "terminal_events": [],
        }
        deliver = getattr(self.engine, "deliver_terminal", None)
        if callable(deliver) and deliver is not dispatch:
            outcome["delivery"] = await self._call_hook(deliver, limit)
        else:
            outcome["redelivered"] = await self._redeliver_terminal(batch)
        outcome["terminal_events"] = await self._replay_terminal_events(batch)
        return outcome

    # ------------------------------------------------------------------
    # Event triggers
    # ------------------------------------------------------------------
    def register_event(
        self, topic: str, objective: str, payload: dict[str, Any] | None = None, name: str | None = None
    ) -> Any:
        """Subscribe to a bus topic; each event creates one deduplicated task."""
        trigger_name = name or f"event:{topic}"

        async def handler(event: Event) -> TaskRecord:
            metadata = {
                "event_topic": topic,
                "event_id": event.id,
                "trigger": trigger_name,
                "event_payload": event.payload,
            }
            create_kwargs: dict[str, Any] = {}
            for key in ("owner_agent", "resource_limits"):
                if payload and key in payload:
                    create_kwargs[key] = payload[key]
            task = await self.engine.create(
                objective,
                metadata=metadata,
                idempotency_key=f"event:{topic}:{event.id}",
                **create_kwargs,
            )
            await self.events.emit("schedule.event_triggered", name=trigger_name, topic=topic, task_id=task.id)
            return task

        self.events.subscribe(topic, handler)
        self._event_handlers.append((topic, handler))
        return handler

    # Aliases for callers that prefer the shorter names.
    on_event = register_event
    subscribe_event = register_event

    def unregister_event(self, handler: Any, topic: str | None = None) -> None:
        for registered_topic, registered_handler in list(self._event_handlers):
            if registered_handler is handler and (topic is None or topic == registered_topic):
                self.events.unsubscribe(registered_topic, handler)
                self._event_handlers.remove((registered_topic, handler))


def _dump(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))


def _load(value: Any) -> dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _row_to_schedule(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "expression": row["expression"],
        "payload": _load(row["payload"]),
        "enabled": bool(row["enabled"]),
        "next_run_at": row["next_run_at"],
        "last_run_at": row["last_run_at"],
        "created_at": row["created_at"],
    }
