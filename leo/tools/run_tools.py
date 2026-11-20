"""Conversation-callable tools for *durable* agent runs.

A run is an ``agent_runs`` row plus (when present) the ``tasks`` row it was
created for. Every field returned here is projected from real SQLite rows via
``leo.db`` helpers, and cancellation goes through the real
``DurableTaskEngine.cancel`` API. Nothing in this module starts a subprocess,
mutates a run row to claim success, or reports a terminal state as live.

Reads work whenever a database has been configured. Writes (``run_cancel``)
prefer ``ctx['services']['tasks']`` and fall back to a real engine import only
when the host has already configured a database. Log entries are bounded and
secret-redacted before they leave the handler.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .. import db
from ..contracts import RiskLevel, TaskStatus, ToolSpec

TOOL_NAMES = ("run_list", "run_status", "run_log", "run_cancel")

MAX_RUN_PAGE = 100
DEFAULT_RUN_PAGE = 50
MAX_LOG_PAGE = 200
DEFAULT_LOG_PAGE = 100
MAX_EVENT_SCAN = 500
MAX_NESTED_ITEMS = 20
MAX_NESTED_KEYS = 50
MAX_TEXT_CHARS = 2000
MAX_EVALUATIONS = 10
MAX_ERRORS = 20

_RUN_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_ID_KEYS = frozenset({"run_id", "task_id", "id", "parent_task_id", "root_task_id"})

# Settings redaction patterns are honoured too; these cover bearer credentials
# and JWTs that the default configuration does not describe.
_BUILTIN_REDACTIONS = (
    r"(?i)\bauthorization\s*:\s*bearer\s+\S+",
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}",
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
)


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_RUN_ID = {"type": "string", "minLength": 1, "maxLength": 200}
_PAGE = {"type": "integer", "minimum": 1, "maximum": MAX_RUN_PAGE}
_LOG_PAGE = {"type": "integer", "minimum": 1, "maximum": MAX_LOG_PAGE}
_OFFSET = {"type": "integer", "minimum": 0, "maximum": 10**9}


def _bounded(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(number, maximum))


def _page(args: Mapping[str, Any]) -> tuple[int, int]:
    """Stay bounded even when a handler is called directly, bypassing validation."""
    return (
        _bounded(args.get("limit", DEFAULT_RUN_PAGE), DEFAULT_RUN_PAGE, 1, MAX_RUN_PAGE),
        _bounded(args.get("offset", 0), 0, 0, 10**9),
    )


def _log_page(args: Mapping[str, Any]) -> tuple[int, int]:
    return (
        _bounded(args.get("limit", DEFAULT_LOG_PAGE), DEFAULT_LOG_PAGE, 1, MAX_LOG_PAGE),
        _bounded(args.get("offset", 0), 0, 0, 10**9),
    )


def _loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return default


def _terminal(value: Any) -> bool:
    try:
        return TaskStatus(str(value)).terminal
    except ValueError:
        return False


def _require_db() -> None:
    if not db.database_path_configured():
        raise RuntimeError("durable run tools require an initialized database: call db.initialize() first")


def _services(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    value = ctx.get("services")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError("ctx['services'] must be a mapping of service names to services")
    return value


def _patterns(ctx: Mapping[str, Any], fallback: Any) -> list[re.Pattern[str]]:
    patterns = list(_BUILTIN_REDACTIONS)
    sources: list[Any] = [_services(ctx), fallback if isinstance(fallback, Mapping) else {}]
    for source in sources:
        candidate = source.get("settings") or ctx.get("settings")
        configured = getattr(getattr(candidate, "security", None), "redact_patterns", None)
        if isinstance(configured, (list, tuple)):
            patterns.extend(str(pattern) for pattern in configured)
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            continue
    return compiled


def _sanitize(value: Any, patterns: list[re.Pattern[str]]) -> tuple[Any, bool]:
    """Redact secrets and cap every string/list/dict in a result."""
    if isinstance(value, str):
        text = value
        changed = False
        for pattern in patterns:
            replaced = pattern.sub("[REDACTED]", text)
            if replaced != text:
                changed = True
                text = replaced
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "...[TRUNCATED]"
            changed = True
        return text, changed
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        changed = False
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_NESTED_KEYS:
                result["__truncated__"] = True
                changed = True
                break
            clean_key, key_changed = _sanitize(str(key), patterns)
            clean_value, value_changed = _sanitize(item, patterns)
            result[clean_key] = clean_value
            changed = changed or key_changed or value_changed
        return result, changed
    if isinstance(value, (list, tuple)):
        result_list: list[Any] = []
        changed = False
        for index, item in enumerate(value):
            if index >= MAX_NESTED_ITEMS:
                result_list.append("...[TRUNCATED]")
                changed = True
                break
            clean, item_changed = _sanitize(item, patterns)
            result_list.append(clean)
            changed = changed or item_changed
        return result_list, changed
    if isinstance(value, (bool, int, float)) or value is None:
        return value, False
    return _sanitize(str(value), patterns)


def _clean(value: Any, patterns: list[re.Pattern[str]]) -> Any:
    return _sanitize(value, patterns)[0]


def _engine(ctx: Mapping[str, Any], fallback: Any) -> Any:
    """Resolve the real task engine; only import one when a DB is configured."""
    for source in (ctx.get("services"), fallback):
        if isinstance(source, Mapping):
            engine = source.get("tasks")
            if engine is not None:
                return engine
    if not db.database_path_configured():
        raise RuntimeError(
            "task service unavailable: no tasks service injected and no database initialized"
        )
    settings = ctx.get("settings")
    events = ctx.get("events")
    if settings is None or events is None:
        raise RuntimeError("task service unavailable: settings/events missing from tool context")
    from ..tasks.engine import DurableTaskEngine  # lazy: only with a real DB

    return DurableTaskEngine(settings, events)


async def _run_row(run_id: str) -> Any:
    row = await db.fetchone(
        "SELECT id,agent_id,task_id,status,started_at,finished_at,result,evaluations,metadata "
        "FROM agent_runs WHERE id=?",
        (run_id,),
    )
    if row is None:
        raise LookupError(f"unknown run: {run_id}")
    return row


async def _task_row(task_id: str | None) -> Any:
    if not task_id:
        return None
    return await db.fetchone(
        "SELECT id,objective,status,current_step,owner_agent,cancellation_requested,"
        "history,errors,evidence,result,created_at,updated_at,completed_at,version "
        "FROM tasks WHERE id=?",
        (task_id,),
    )


def _run_view(row: Any, task: Any = None) -> dict[str, Any]:
    status = str(row["status"])
    return {
        "run_id": row["id"],
        "agent_id": row["agent_id"],
        "task_id": row["task_id"],
        "status": status,
        "terminal": status in _RUN_TERMINAL,
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "task_status": str(task["status"]) if task is not None else None,
        "task_terminal": _terminal(task["status"]) if task is not None else False,
        "cancellation_requested": bool(task["cancellation_requested"]) if task is not None else False,
    }


def _next(offset: int, count: int, total: int) -> int | None:
    consumed = offset + count
    return consumed if consumed < total else None


async def run_list(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """Return a bounded page of persisted agent runs (not in-process chat state)."""
    _require_db()
    limit, offset = _page(args)
    clauses: list[str] = []
    params: list[Any] = []
    for key, column in (("agent_id", "r.agent_id"), ("status", "r.status"), ("task_id", "r.task_id")):
        value = args.get(key)
        if isinstance(value, str) and value:
            clauses.append(f"{column}=?")
            params.append(value)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    total = int((await db.fetchone(
        "SELECT COUNT(*) AS n FROM agent_runs r" + where, params
    ))["n"])
    rows = await db.fetchall(
        "SELECT r.id,r.agent_id,r.task_id,r.status,r.started_at,r.finished_at,"
        "t.status AS task_status,t.cancellation_requested AS cancellation_requested "
        "FROM agent_runs r LEFT JOIN tasks t ON t.id=r.task_id" + where +
        " ORDER BY r.started_at DESC, r.id DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    patterns = _patterns(ctx, fallback)
    views = []
    for row in rows:
        linked = None
        if row["task_status"] is not None:
            linked = {
                "status": row["task_status"],
                "cancellation_requested": row["cancellation_requested"],
            }
        views.append(_clean(_run_view(row, linked), patterns))
    return {
        "runs": views,
        "count": len(views),
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": _next(offset, len(views), total),
    }


async def run_status(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """Report the actual persisted run and linked-task state."""
    _require_db()
    run = await _run_row(args["run_id"])
    task = await _task_row(run["task_id"])
    patterns = _patterns(ctx, fallback)
    evaluations = _loads(run["evaluations"], []) or []
    if not isinstance(evaluations, list):
        evaluations = []
    view = _run_view(run, task)
    view.update({
        "result": _clean(_loads(run["result"], None), patterns),
        "evaluations": _clean(evaluations[:MAX_EVALUATIONS], patterns),
        "evaluation_count": len(evaluations),
        "metadata": _clean(_loads(run["metadata"], {}), patterns),
        "task": None if task is None else {
            "task_id": task["id"],
            "status": str(task["status"]),
            "terminal": _terminal(task["status"]),
            "cancellation_requested": bool(task["cancellation_requested"]),
            "current_step": task["current_step"],
            "owner_agent": task["owner_agent"],
            "error_count": len(_loads(task["errors"], []) or []),
            "evidence_count": len(_loads(task["evidence"], []) or []),
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
            "completed_at": task["completed_at"],
            "version": task["version"],
        },
    })
    return _clean(view, patterns)


def _mentions(value: Any, identifiers: frozenset[str], depth: int = 0) -> bool:
    if depth > 3:
        return False
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _ID_KEYS and isinstance(item, str) and item in identifiers:
                return True
            if isinstance(item, (dict, list)) and _mentions(item, identifiers, depth + 1):
                return True
    elif isinstance(value, list):
        return any(_mentions(item, identifiers, depth + 1) for item in value[:MAX_NESTED_ITEMS])
    return False


async def run_log(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """Build a bounded, redacted log from task history, run rows and event rows."""
    _require_db()
    limit, offset = _log_page(args)
    include_events = args.get("include_events", True) is not False
    run = await _run_row(args["run_id"])
    task = await _task_row(run["task_id"])
    identifiers = frozenset(
        str(value) for value in (run["id"], run["task_id"]) if value
    )

    entries: list[dict[str, Any]] = []
    if task is not None:
        history = _loads(task["history"], []) or []
        for item in history if isinstance(history, list) else []:
            detail = dict(item) if isinstance(item, Mapping) else {"value": item}
            entries.append({
                "at": detail.get("at"),
                "source": "task_history",
                "event": str(detail.get("event", "history")),
                "detail": detail,
            })
        errors = _loads(task["errors"], []) or []
        for error in (errors if isinstance(errors, list) else [])[:MAX_ERRORS]:
            entries.append({
                "at": task["updated_at"],
                "source": "task_error",
                "event": "task.error",
                "detail": {"error": error},
            })

    entries.append({
        "at": run["started_at"],
        "source": "run",
        "event": "run.started",
        "detail": {"agent_id": run["agent_id"], "task_id": run["task_id"]},
    })
    if run["finished_at"]:
        entries.append({
            "at": run["finished_at"],
            "source": "run",
            "event": "run.finished",
            "detail": {"status": run["status"], "agent_id": run["agent_id"]},
        })

    if include_events and identifiers:
        placeholders = ",".join("?" for _ in identifiers)
        # Both predicates are mandatory: a run.% topic from a different run is
        # not this run's log just because it happens to be recent, and a chat.%
        # event is not a run event merely because its correlation id matches.
        sql = (
            "SELECT id,topic,source,payload,correlation_id,created_at FROM events "
            f"WHERE correlation_id IN ({placeholders}) "
            "AND (topic LIKE 'task.%' OR topic LIKE 'run.%') "
            "ORDER BY created_at ASC, rowid ASC"
        )
        rows = await db.fetchall(sql, list(identifiers))
        scoped_events: list[dict[str, Any]] = []
        for row in rows:
            payload = _loads(row["payload"], {})
            if row["correlation_id"] not in identifiers and not _mentions(payload, identifiers):
                continue
            scoped_events.append({
                "at": row["created_at"],
                "source": "event",
                "event": row["topic"],
                "detail": {
                    "id": row["id"],
                    "source": row["source"],
                    "correlation_id": row["correlation_id"],
                    "payload": payload,
                },
            })
        scoped_events.sort(
            key=lambda item: (str(item.get("at") or ""), str(item["detail"]["id"] or ""))
        )
        # Cap only after the scope filter and ordering: unrelated rows must not
        # consume the event budget and hide this run's own entries.
        entries.extend(scoped_events[:MAX_EVENT_SCAN])

    entries.sort(key=lambda item: str(item.get("at") or ""))
    total = len(entries)
    patterns = _patterns(ctx, fallback)
    page: list[dict[str, Any]] = []
    redacted = False
    for index, entry in enumerate(entries[offset:offset + limit]):
        clean, changed = _sanitize(entry, patterns)
        page.append({"seq": offset + index, **clean})
        redacted = redacted or changed
    return {
        "run_id": run["id"],
        "task_id": run["task_id"],
        "entries": page,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": _next(offset, len(page), total),
        "redacted": redacted,
        "include_events": include_events,
    }


def _cancel_view(run: Any, task: Any | None, *, applied: bool, reason: str) -> dict[str, Any]:
    return {
        "run_id": run["id"],
        "agent_id": run["agent_id"],
        "task_id": run["task_id"],
        "run_status": str(run["status"]),
        "applied": applied,
        "cancelled": task is not None and str(task["status"]) == TaskStatus.CANCELLED.value,
        "terminal": _terminal(task["status"]) if task is not None else False,
        "task_status": str(task["status"]) if task is not None else None,
        "cancellation_requested": bool(task["cancellation_requested"]) if task is not None else False,
        "reason": reason,
    }


async def run_cancel(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """Request cancellation through the durable engine; report only persisted state."""
    # Resolve the engine first so a missing service is reported accurately,
    # rather than being masked as a generic database error.
    engine = _engine(ctx, fallback)
    _require_db()
    run = await _run_row(args["run_id"])
    task = await _task_row(run["task_id"])
    if task is None:
        reason = (
            "run has no linked durable task" if not run["task_id"]
            else "linked durable task no longer exists"
        )
        return _cancel_view(run, task, applied=False, reason=reason)
    status = str(task["status"])
    if _terminal(status):
        return _cancel_view(run, task, applied=False, reason=f"task is in terminal state '{status}'")
    if task["cancellation_requested"]:
        return _cancel_view(run, task, applied=False, reason="cancellation already requested")

    try:
        applied = bool(await engine.cancel(task["id"]))
    except ValueError as exc:
        # A concurrent worker may have finished the task between our read and
        # the cancel call: re-read instead of assuming the failure is fatal.
        current = await _task_row(task["id"])
        if current is not None and _terminal(current["status"]):
            return _cancel_view(
                run, current, applied=False,
                reason=f"task is in terminal state '{current['status']}' ({exc})",
            )
        raise
    except LookupError:
        current = await _task_row(task["id"])
        if current is None:
            return _cancel_view(
                run, None, applied=False, reason="linked durable task no longer exists"
            )
        raise

    # cancel() only flags RUNNING/VERIFYING tasks; the post-write read keeps the
    # reported status honest instead of claiming a terminal cancellation.
    current = await _task_row(task["id"])
    if current is None:
        raise LookupError(f"unknown task after cancellation: {task['id']}")
    reason = (
        "cancellation requested; worker must observe the flag"
        if applied and not _terminal(current["status"])
        else "cancellation applied by durable task engine" if applied
        else "engine did not apply cancellation"
    )
    return _cancel_view(run, current, applied=applied, reason=reason)


_SPECS: tuple[tuple[ToolSpec, Any], ...] = (
    (
        ToolSpec(
            name="run_list",
            description="List persisted agent runs from agent_runs with bounded, filterable pages.",
            category="runs",
            risk=RiskLevel.LOW,
            permissions=["runs:read"],
            input_schema=_schema({
                "agent_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "status": {"type": "string", "minLength": 1, "maxLength": 64},
                "task_id": _RUN_ID,
                "limit": _PAGE,
                "offset": _OFFSET,
            }),
            resource_field="",  # database rows, not filesystem resources
        ),
        run_list,
    ),
    (
        ToolSpec(
            name="run_status",
            description="Read the real persisted agent run, its linked durable task and evaluation count.",
            category="runs",
            risk=RiskLevel.LOW,
            permissions=["runs:read"],
            input_schema=_schema({"run_id": _RUN_ID}, ("run_id",)),
            resource_field="",
        ),
        run_status,
    ),
    (
        ToolSpec(
            name="run_log",
            description="Return a bounded, secret-redacted log assembled from durable task history, run rows and event rows.",
            category="runs",
            risk=RiskLevel.LOW,
            permissions=["runs:read"],
            input_schema=_schema({
                "run_id": _RUN_ID,
                "limit": _LOG_PAGE,
                "offset": _OFFSET,
                "include_events": {"type": "boolean"},
            }, ("run_id",)),
            resource_field="",
        ),
        run_log,
    ),
    (
        ToolSpec(
            name="run_cancel",
            description="Request cancellation of a run's durable task through the task engine; never fabricates a cancelled state.",
            category="runs",
            risk=RiskLevel.MEDIUM,
            permissions=["runs:write", "tasks:write"],
            input_schema=_schema({"run_id": _RUN_ID}, ("run_id",)),
            resource_field="",
        ),
        run_cancel,
    ),
)


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the durable-run toolset and return its names in registration order."""
    registered: list[str] = []
    for spec, implementation in _SPECS:
        async def handler(args: dict[str, Any], ctx: dict[str, Any], _impl=implementation) -> Any:
            return await _impl(args, ctx, services)

        try:
            registry.register(spec, handler)
        except ValueError:
            # Another owner registered this name first; keep their version.
            continue
        registered.append(spec.name)
    return registered


__all__ = ["register", "TOOL_NAMES"]
