"""Conversation-callable scheduling and automation tools.

Schedules are stored through the real ``Scheduler`` API and tasks through the
real ``DurableTaskEngine``/scheduler tick path. A manual trigger never claims
that work executed: it reports the persisted task and its actual status, and
leaves completion to a worker with evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .. import db
from ..contracts import RiskLevel, ToolSpec
from ..tasks.scheduler import as_utc, every_interval, parse_cron

_SCHEDULER_KEY = "scheduler"
_ENGINE_KEYS = ("tasks", "engine", "task_engine")


def _schema(
    properties: dict[str, dict[str, Any]], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": False,
    }


def _service(ctx: dict[str, Any], registered: Any, keys: tuple[str, ...]) -> Any:
    """Resolve a real service from the executor context, then registration.

    Context services win so the integration owner can swap/replace them at
    runtime. A missing dependency raises instead of producing a canned result.
    """
    for source in (ctx.get("services"), registered):
        if not source:
            continue
        if isinstance(source, Mapping):
            for key in keys:
                value = source.get(key)
                if value is not None:
                    return value
        else:
            for key in keys:
                value = getattr(source, key, None)
                if value is not None:
                    return value
    raise RuntimeError(
        f"{keys[0]} service unavailable: inject {keys[0]!r} via ctx['services']"
    )


def _scheduler(ctx: dict[str, Any], registered: Any) -> Any:
    return _service(ctx, registered, (_SCHEDULER_KEY,))


def _engine(ctx: dict[str, Any], registered: Any) -> Any:
    return _service(ctx, registered, _ENGINE_KEYS)


def _seconds(text: str) -> str:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("interval seconds must be positive and finite")
    return str(value)


def _validate_expression(expression: Any) -> str:
    """Validate with the scheduler's real parsers before persisting anything."""
    if isinstance(expression, bool):
        raise ValueError("expression must be a cron string, '@every' or positive seconds")
    if isinstance(expression, (int, float)):
        return _seconds(str(expression))
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string or positive number")
    text = expression.strip()
    if text.lower().startswith("@every"):
        if every_interval(text) is None:
            raise ValueError(f"invalid @every expression: {expression!r}")
        return text
    try:
        float(text)
    except ValueError:
        parse_cron(text)  # real cron parser; raises with precise field errors
        return text
    return _seconds(text)


def _view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a persisted schedule row without inventing fields."""
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "expression": row["expression"],
        "payload": dict(row["payload"] or {}),
        "enabled": bool(row["enabled"]),
        "next_run_at": row["next_run_at"],
        "last_run_at": row["last_run_at"],
        "created_at": row["created_at"],
    }


async def _create(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    scheduler = _scheduler(ctx, registered)
    name = str(args["name"]).strip()
    objective = str(args["objective"]).strip()
    if not name or not objective:
        raise ValueError("schedule name and objective are required")
    expression = _validate_expression(args["expression"])
    row = await scheduler.register(
        name, expression, objective, dict(args.get("payload") or {})
    )
    return _view(row)


async def _list(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    scheduler = _scheduler(ctx, registered)
    enabled_only = bool(args.get("enabled_only", False))
    rows = await scheduler.list_schedules(enabled_only=enabled_only)
    views = [_view(row) for row in rows]
    return {"schedules": views, "count": len(views), "enabled_only": enabled_only}


async def _delete(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    scheduler = _scheduler(ctx, registered)
    name = args["name"]
    existing = await scheduler.get(name)  # KeyError for an unknown schedule
    cursor = await db.execute("DELETE FROM schedules WHERE name=?", (name,))
    if int(getattr(cursor, "rowcount", 0) or 0) < 1:
        raise RuntimeError(f"failed to delete schedule: {name}")
    if await db.fetchone("SELECT id FROM schedules WHERE name=?", (name,)) is not None:
        raise RuntimeError(f"schedule still present after delete: {name}")
    events = ctx.get("events")
    if events is not None and hasattr(events, "emit"):
        await events.emit("schedule.deleted", name=name, schedule_id=existing["id"])
    return {"name": name, "schedule_id": existing["id"], "deleted": True}


async def _run_now(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    """Trigger one occurrence of *only* the named schedule.

    ``Scheduler.trigger`` anchors the occurrence to the requested instant and
    never runs a global tick, so another schedule cannot fire as a side effect.
    The task's status is read back from the durable engine. Nothing here runs a
    worker, so ``executed`` is always false and no completion is claimed.
    """
    scheduler = _scheduler(ctx, registered)
    engine = _engine(ctx, registered)
    name = args["name"]
    schedule = await scheduler.get(name)
    if not schedule["enabled"]:
        raise ValueError(f"schedule is disabled: {name}")

    requested = args.get("run_at")
    if requested is not None:
        if not isinstance(requested, str):
            raise ValueError("run_at must be an ISO 8601 string")
        try:
            requested_at = as_utc(requested)  # real ISO parser
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid ISO 8601 run_at: {requested!r} ({exc})"
            ) from exc
    else:
        requested_at = datetime.now(timezone.utc)

    scheduled_for = as_utc(schedule["next_run_at"])
    # The request decides when this one occurrence happens. Clamping to the
    # target's future next_run_at made a manual trigger fire at a time the
    # caller never asked for, and a global tick fired unrelated schedules.
    moment = requested_at
    due = await scheduler.trigger(name, now=moment.isoformat())
    ours = [
        task for task in due
        if (getattr(task, "metadata", None) or {}).get("schedule") == name
    ]
    tasks: list[dict[str, Any]] = []
    for task in ours:
        persisted = await engine.get(task.id)
        if persisted is None:
            continue
        status = getattr(persisted.status, "value", persisted.status)
        tasks.append({
            "task_id": persisted.id,
            "status": str(status),
            "objective": persisted.objective,
        })
    updated = await scheduler.get(name)
    return {
        "schedule": name,
        "schedule_id": schedule["id"],
        "run_at": moment.isoformat(),
        "requested_run_at": requested,
        "scheduled_for": moment.isoformat(),  # the actual requested occurrence time
        "target_next_run_at": scheduled_for.isoformat(),  # schedule cursor before the trigger
        "triggered": bool(tasks),
        "task_count": len(tasks),
        "task_id": tasks[0]["task_id"] if tasks else None,
        "task_status": tasks[0]["status"] if tasks else None,
        "tasks": tasks,
        "executed": False,
        "note": "Durable task(s) created; no worker executed and no completion claimed.",
        "next_run_at": updated["next_run_at"],
        "last_run_at": updated["last_run_at"],
    }


_TOOL_DEFS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="schedule_create",
            description=(
                "Persist a schedule from a 5-field cron expression, '@every' expression "
                "or positive interval seconds; syntax is validated before writing."
            ),
            category="automation",
            risk=RiskLevel.MEDIUM,
            permissions=["schedules:write", "tasks:write"],
            input_schema=_schema(
                {
                    "name": {"type": "string", "minLength": 1, "maxLength": 200},
                    "expression": {"type": ["string", "number"]},
                    "objective": {"type": "string", "minLength": 1},
                    "payload": {"type": "object"},
                },
                ["name", "expression", "objective"],
            ),
            resource_field="",  # identifiers, not filesystem paths
        ),
        _create,
    ),
    (
        ToolSpec(
            name="schedule_list",
            description="List persisted schedules with their real next/last run timestamps.",
            category="automation",
            risk=RiskLevel.LOW,
            permissions=["schedules:read"],
            input_schema=_schema({"enabled_only": {"type": "boolean"}}),
            resource_field="",
        ),
        _list,
    ),
    (
        ToolSpec(
            name="schedule_delete",
            description="Delete a persisted schedule by name; reports deletion only after the row is gone.",
            category="automation",
            risk=RiskLevel.MEDIUM,
            permissions=["schedules:write"],
            input_schema=_schema(
                {"name": {"type": "string", "minLength": 1, "maxLength": 200}},
                ["name"],
            ),
            resource_field="",
        ),
        _delete,
    ),
    (
        ToolSpec(
            name="schedule_run_now",
            description=(
                "Trigger one occurrence of the named schedule now (optionally at an ISO "
                "8601 instant) and return the persisted task status; no other schedule is "
                "fired and no worker is executed."
            ),
            category="automation",
            risk=RiskLevel.MEDIUM,
            permissions=["schedules:write", "tasks:write"],
            input_schema=_schema(
                {
                    "name": {"type": "string", "minLength": 1, "maxLength": 200},
                    "run_at": {"type": "string", "minLength": 1},
                },
                ["name"],
            ),
            resource_field="",
        ),
        _run_now,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register all automation tools and return their names in order."""
    registered: list[str] = []
    for spec, implementation in _TOOL_DEFS:

        async def handler(
            args: dict[str, Any], ctx: dict[str, Any], _impl=implementation
        ) -> Any:
            return await _impl(args, ctx, services)

        registry.register(spec, handler)
        registered.append(spec.name)
    return registered


__all__ = ["register"]
