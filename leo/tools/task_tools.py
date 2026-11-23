"""Conversation-callable control tools for the durable task engine.

Every handler resolves a real ``DurableTaskEngine`` and reads/writes the shared
SQLite task store. Results are projections of durable rows: a tool never claims
a task was created, cancelled or completed unless the engine's persisted state
says so. The injected ``services['tasks']`` engine is preferred; an import
fallback is only permitted when the host has already initialized the database.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import db
from ..contracts import RiskLevel, TaskStatus, ToolSpec

_STATUSES = [status.value for status in TaskStatus]


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": False,
    }


# Task ids are opaque strings, not filesystem paths: no permission resource is
# derived from them (see ToolExecutorImpl._resource_for).
_TASK_ID = {"type": "string", "minLength": 1, "maxLength": 200}
_OPTIONAL_LIMIT = {"type": "integer", "minimum": 1, "maximum": 1000}
# Objective search is a bounded substring scan over persisted rows: no regex,
# no SQL from the model and no unbounded walk of the task table.
MAX_QUERY_CHARS = 500
_SEARCH_SCAN_LIMIT = 1000
OBJECTIVE_QUERY = {
    "type": "string", "minLength": 1, "maxLength": MAX_QUERY_CHARS,
}
# Bound model-supplied task structure before it can reach the durable store.
MAX_PLAN_ITEMS = 50
MAX_PLAN_ITEM_CHARS = 500
_PLAN_ITEM = {"type": "string", "minLength": 1, "maxLength": MAX_PLAN_ITEM_CHARS}
_PLAN_ARRAY = {"type": "array", "maxItems": MAX_PLAN_ITEMS, "items": _PLAN_ITEM}


def _status_value(task: Any) -> str:
    value = getattr(task.status, "value", task.status)
    return str(value)


def _task_view(task: Any) -> dict[str, Any]:
    """Project a persisted TaskRecord without inventing fields."""
    status = _status_value(task)
    return {
        "task_id": task.id,
        "objective": task.objective,
        "status": status,
        "terminal": status in {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value,
                               TaskStatus.CANCELLED.value},
        "plan": list(task.plan or []),
        "current_step": task.current_step,
        "dependencies": list(task.dependencies or []),
        "owner_agent": task.owner_agent,
        "cancellation_requested": bool(task.cancellation_requested),
        "retry_count": task.retry_count,
        "error_count": len(task.errors or []),
        "evidence_count": len(task.evidence or []),
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "completed_at": task.completed_at,
        "version": task.version,
    }


def _compact_task_view(task: Any) -> dict[str, Any]:
    """A one-line, newest-first list projection; details live in task_status."""
    return {
        "task_id": task.id,
        "objective": task.objective,
        "status": _status_value(task),
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _objective_query(args: dict[str, Any]) -> str | None:
    """Accept both spellings of the objective filter without guessing silently."""
    query = args.get("query")
    alias = args.get("objective_contains")
    if query is not None and not isinstance(query, str):
        raise ValueError("query must be a string")
    if alias is not None and not isinstance(alias, str):
        raise ValueError("objective_contains must be a string")
    if query is not None and alias is not None and query.casefold() != alias.casefold():
        raise ValueError("query and objective_contains must match when both are supplied")
    value = query if query is not None else alias
    if value is None:
        return None
    # Schema already enforces a bounded, non-empty string; normalization makes
    # whitespace-only input honest instead of matching every objective.
    normalized = value.strip()
    if not normalized:
        raise ValueError("query must contain at least one non-whitespace character")
    return normalized


def _engine(ctx: dict[str, Any], registered_services: Any = None) -> Any:
    """Resolve the task engine from context services, then registration, then import.

    Importing/constructing an engine is only allowed when a host has already
    configured a database; otherwise an unavailable service is reported instead
    of silently operating on a different (or nonexistent) store.
    """
    for source in (ctx.get("services"), registered_services):
        if isinstance(source, Mapping):
            engine = source.get("tasks")
            if engine is not None:
                return engine
    if not db.database_path_configured():
        raise RuntimeError(
            "task service unavailable: no tasks service injected and no database initialized"
        )
    from ..tasks.engine import DurableTaskEngine  # lazy: only with a real DB

    settings = ctx.get("settings")
    events = ctx.get("events")
    if settings is None or events is None:
        raise RuntimeError(
            "task service unavailable: settings/events missing from tool context"
        )
    return DurableTaskEngine(settings, events)


async def _create(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    engine = _engine(ctx, services)
    metadata = dict(args.get("metadata") or {})
    if args.get("background"):
        # The chat that asked for background work is where progress belongs.
        # The executor injects the live conversation/channel under
        # ``origin_context``; writing it into metadata gives the progress hub,
        # the delivery path and post-restart recovery the same origin fact.
        origin = dict(ctx.get("origin_context") or {})
        if origin:
            metadata.setdefault("origin", {
                "channel": str(origin.get("channel") or ""),
                "target": str(origin.get("target") or ""),
                "conversation_id": str(origin.get("conversation_id") or ""),
            })
    task = await engine.create(
        objective=args["objective"],
        plan=list(args.get("plan") or []),
        dependencies=list(args.get("dependencies") or []),
        owner_agent=args.get("owner_agent"),
        resource_limits=dict(args.get("resource_limits") or {}),
        metadata=metadata,
        idempotency_key=args.get("idempotency_key"),
    )
    return _task_view(task)


async def _task_progress(
    args: dict[str, Any], ctx: dict[str, Any], services: Any = None
) -> Any:
    """Record a progress note from inside a running task.

    The model (or a worker) uses this to say what it is about to do; the hub
    relays it to the origin chat and the engine keeps it in durable history.
    Iteration facts must be real: pass ``iteration`` only when the work is
    actually iterative, and ``max_iterations`` only when the bound is known.
    """
    engine = _engine(ctx, services)
    note = str(args["note"]).strip()
    details: dict[str, Any] = {}
    iteration = args.get("iteration")
    if iteration is not None:
        details["iteration"] = max(0, int(iteration))
    maximum = args.get("max_iterations")
    if maximum is not None:
        details["max_iterations"] = max(0, int(maximum))
    step = args.get("step")
    if step is not None:
        details["step"] = int(step)
    task = await engine.progress(args["task_id"], note, **details)
    return _task_view(task)


async def _list(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    engine = _engine(ctx, services)
    status = args.get("status")
    limit = args.get("limit", 100)
    query = _objective_query(args)
    newest_first = bool(args.get("newest_first", True))

    # The engine already reads newest-first with a SQL LIMIT. When filtering by
    # objective we widen that read to a fixed ceiling, filter in memory, then
    # truncate again: a query never scans the whole table and never returns
    # more than the caller's bounded page.
    scan_limit = _SEARCH_SCAN_LIMIT if query else limit
    tasks = await engine.list(status=status, limit=scan_limit)
    if query:
        needle = query.casefold()
        tasks = [
            task for task in tasks
            if needle in str(getattr(task, "objective", None) or "").casefold()
        ]
    tasks = sorted(
        tasks,
        key=lambda task: (str(task.created_at or ""), str(task.id or "")),
        reverse=newest_first,
    )
    matched = len(tasks)
    views = [_compact_task_view(task) for task in tasks[:limit]]
    return {
        "tasks": views,
        "count": len(views),
        "status_filter": status,
        "limit": limit,
        "query": query,
        "newest_first": newest_first,
        "matched": matched,
        "truncated": matched > limit,
    }


async def _status(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    engine = _engine(ctx, services)
    task_id = args["task_id"]
    task = await engine.get(task_id)
    if task is None:
        raise KeyError(f"unknown task: {task_id}")
    return _task_view(task)


async def _cancel(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    engine = _engine(ctx, services)
    task_id = args["task_id"]
    if not await engine.cancel(task_id):
        raise RuntimeError(f"cancellation was not applied to task {task_id}")
    # cancel() only flags RUNNING/VERIFYING tasks; the post-write read keeps the
    # reported status honest instead of assuming "cancelled".
    task = await engine.get(task_id)
    if task is None:
        raise KeyError(f"unknown task after cancellation: {task_id}")
    view = _task_view(task)
    view["cancelled"] = view["status"] == TaskStatus.CANCELLED.value
    return view


async def _result(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    engine = _engine(ctx, services)
    task_id = args["task_id"]
    task = await engine.get(task_id)
    if task is None:
        raise KeyError(f"unknown task: {task_id}")
    status = _status_value(task)
    return {
        "task_id": task.id,
        "status": status,
        "completed": status == TaskStatus.COMPLETED.value,
        "result": task.result,
        "evidence": [dict(item) for item in (task.evidence or [])],
        "errors": list(task.errors or []),
        "completed_at": task.completed_at,
    }


_TOOL_DEFS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="task_create",
            description=(
                "Create a durable task in the shared task store. An idempotency key reuses an "
                "existing task. To check for prior work first, call task_list with query/"
                "objective_contains; do not inspect .db files directly."
            ),
            category="tasks",
            risk=RiskLevel.MEDIUM,
            permissions=["tasks:write"],
            input_schema=_schema(
                {
                    "objective": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "background": {
                        "type": "boolean",
                        "description": (
                            "Run in the background and report live progress to this chat."
                        ),
                    },
                    "plan": _PLAN_ARRAY,
                    "dependencies": _PLAN_ARRAY,
                    "owner_agent": {"type": "string", "minLength": 1},
                    "resource_limits": {"type": "object"},
                    "metadata": {"type": "object"},
                    "idempotency_key": {"type": "string", "minLength": 1},
                },
                ["objective"],
            ),
            resource_field="",
        ),
        _create,
    ),
    (
        ToolSpec(
            name="task_list",
            description=(
                "List durable tasks newest-first in one bounded call. Filter by status and/or "
                "query (alias: objective_contains) for a case-insensitive substring of the "
                "objective, e.g. {'query': 'tool execution', 'limit': 10}. Results are compact "
                "(task_id, objective, status, timestamps); set newest_first=false for oldest "
                "first and call task_status for full details. Never grep the SQLite file."
            ),
            category="tasks",
            risk=RiskLevel.LOW,
            permissions=["tasks:read"],
            input_schema=_schema(
                {
                    "status": {"type": "string", "enum": _STATUSES},
                    "limit": _OPTIONAL_LIMIT,
                    "query": OBJECTIVE_QUERY,
                    "objective_contains": OBJECTIVE_QUERY,
                    "newest_first": {"type": "boolean", "default": True},
                }
            ),
            resource_field="",
        ),
        _list,
    ),
    (
        ToolSpec(
            name="task_status",
            description="Read the persisted status and lifecycle metadata of one task by id.",
            category="tasks",
            risk=RiskLevel.LOW,
            permissions=["tasks:read"],
            input_schema=_schema({"task_id": _TASK_ID}, ["task_id"]),
            resource_field="",
        ),
        _status,
    ),
    (
        ToolSpec(
            name="task_progress",
            description=(
                "Record a progress note on a running task: what is about to happen, or the "
                "iteration reached. Pass iteration/max_iterations only for genuinely "
                "iterative work; never invent a percentage."
            ),
            category="tasks",
            risk=RiskLevel.LOW,
            permissions=["tasks:write"],
            input_schema=_schema(
                {
                    "task_id": _TASK_ID,
                    "note": {"type": "string", "minLength": 1, "maxLength": 500},
                    "iteration": {"type": "integer", "minimum": 0},
                    "max_iterations": {"type": "integer", "minimum": 1},
                    "step": {"type": "integer", "minimum": 0},
                },
                ["task_id", "note"],
            ),
            resource_field="",
        ),
        _task_progress,
    ),
    (
        ToolSpec(
            name="task_cancel",
            description="Request cancellation of a task; running tasks remain running until the worker observes the flag.",
            category="tasks",
            risk=RiskLevel.MEDIUM,
            permissions=["tasks:write"],
            input_schema=_schema({"task_id": _TASK_ID}, ["task_id"]),
            resource_field="",
        ),
        _cancel,
    ),
    (
        ToolSpec(
            name="task_result",
            description="Read a task's persisted result and evidence; reports completed=false unless the store says completed.",
            category="tasks",
            risk=RiskLevel.LOW,
            permissions=["tasks:read"],
            input_schema=_schema({"task_id": _TASK_ID}, ["task_id"]),
            resource_field="",
        ),
        _result,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register all task-control tools; return their names in registration order."""
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
