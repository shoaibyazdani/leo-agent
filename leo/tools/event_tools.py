"""Observability tools backed by durable event/audit state.

``event_query`` and ``audit_query`` read real rows from the shared SQLite
database (never an in-memory cache), with bounded pages and offset cursors.
``metrics_snapshot`` computes real counters from those rows, persists them in
the ``metrics`` table and publishes an ``observability.snapshot`` event.

The snapshot counters describe the database *before* the snapshot event and
this tool call's own audit/tool rows are written; that keeps the numbers
reproducible and self-consistent.
"""

from __future__ import annotations

import json
from typing import Any

from .. import db
from ..contracts import RiskLevel, ToolSpec, utcnow
from ..events import Event

MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 50
MAX_BREAKDOWN = 10
TOOL_NAMES = ("event_query", "audit_query", "metrics_snapshot")


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_PAGE_PROPERTIES: dict[str, Any] = {
    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_PAGE_SIZE},
    "offset": {"type": "integer", "minimum": 0},
}


def _bounded(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(number, maximum))


def _page(args: dict[str, Any]) -> tuple[int, int]:
    """Defence in depth: handlers stay bounded even when called directly."""
    limit = _bounded(args.get("limit", DEFAULT_PAGE_SIZE), DEFAULT_PAGE_SIZE, 1, MAX_PAGE_SIZE)
    offset = _bounded(args.get("offset", 0), 0, 0, 10**9)
    return limit, offset


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {"raw": raw}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _require_db() -> None:
    if not db.database_path_configured():
        raise RuntimeError(
            "observability tools require an initialized database: call db.initialize() first"
        )


async def event_query(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, newest-first page of durable event rows."""
    _require_db()
    limit, offset = _page(args)
    where: list[str] = []
    params: list[Any] = []
    topic = args.get("topic")
    correlation_id = args.get("correlation_id")
    if isinstance(topic, str) and topic:
        where.append("topic=?")
        params.append(topic)
    if isinstance(correlation_id, str) and correlation_id:
        where.append("correlation_id=?")
        params.append(correlation_id)
    clause = " WHERE " + " AND ".join(where) if where else ""

    total = int((await db.fetchone("SELECT COUNT(*) AS n FROM events" + clause, params))["n"])
    rows = await db.fetchall(
        "SELECT id,topic,source,payload,correlation_id,created_at FROM events"
        + clause
        + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    events = [
        {
            "id": row["id"],
            "topic": row["topic"],
            "source": row["source"],
            "payload": _json_object(row["payload"]),
            "correlation_id": row["correlation_id"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    consumed = offset + len(events)
    return {
        "success": True,
        "events": events,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": consumed if consumed < total else None,
    }


async def audit_query(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, newest-first page of durable permission/executor audit rows."""
    _require_db()
    limit, offset = _page(args)
    where: list[str] = []
    params: list[Any] = []
    for key, column in (("actor", "actor"), ("action", "action"),
                        ("decision", "decision"), ("risk", "risk")):
        value = args.get(key)
        if isinstance(value, str) and value:
            where.append(f"{column}=?")
            params.append(value)
    clause = " WHERE " + " AND ".join(where) if where else ""

    total = int((await db.fetchone("SELECT COUNT(*) AS n FROM audit_log" + clause, params))["n"])
    rows = await db.fetchall(
        "SELECT id,ts,actor,action,resource,risk,decision,reason,metadata,correlation_id "
        "FROM audit_log" + clause + " ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    entries = [
        {
            "id": row["id"],
            "ts": row["ts"],
            "actor": row["actor"],
            "action": row["action"],
            "resource": row["resource"],
            "risk": row["risk"],
            "decision": row["decision"],
            "reason": row["reason"],
            "metadata": _json_object(row["metadata"]),
            "correlation_id": row["correlation_id"],
        }
        for row in rows
    ]
    consumed = offset + len(entries)
    return {
        "success": True,
        "entries": entries,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": consumed if consumed < total else None,
    }


async def metrics_snapshot(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Compute real counters, persist them as metric rows and publish a snapshot event."""
    _require_db()
    services = dict(ctx.get("services") or {})
    bus = services.get("events") or ctx.get("events")
    if bus is None or not hasattr(bus, "publish"):
        raise RuntimeError(
            "metrics_snapshot requires an events service in ctx['services']['events'] "
            "or ctx['events'] to publish the snapshot"
        )
    breakdown = _bounded(args.get("topic_limit", MAX_BREAKDOWN), MAX_BREAKDOWN, 1, MAX_PAGE_SIZE)

    events_total = int((await db.fetchone("SELECT COUNT(*) AS n FROM events"))["n"])
    audit_total = int((await db.fetchone("SELECT COUNT(*) AS n FROM audit_log"))["n"])
    audit_denied = int((
        await db.fetchone("SELECT COUNT(*) AS n FROM audit_log WHERE decision='denied'")
    )["n"])
    calls_total = int((await db.fetchone("SELECT COUNT(*) AS n FROM tool_calls"))["n"])
    calls_failed = int((
        await db.fetchone("SELECT COUNT(*) AS n FROM tool_calls WHERE success=0")
    )["n"])
    topic_rows = await db.fetchall(
        "SELECT topic, COUNT(*) AS n FROM events GROUP BY topic "
        "ORDER BY n DESC, topic ASC LIMIT ?", [breakdown],
    )
    decision_rows = await db.fetchall(
        "SELECT decision, COUNT(*) AS n FROM audit_log GROUP BY decision "
        "ORDER BY n DESC, decision ASC LIMIT ?", [breakdown],
    )
    counters = {
        "events_total": events_total,
        "audit_total": audit_total,
        "audit_denied": audit_denied,
        "tool_calls_total": calls_total,
        "tool_calls_failed": calls_failed,
    }
    metric_names = {
        "events_total": "observability.events.total",
        "audit_total": "observability.audit.total",
        "audit_denied": "observability.audit.denied",
        "tool_calls_total": "observability.tool_calls.total",
        "tool_calls_failed": "observability.tool_calls.failed",
    }
    generated_at = utcnow()
    labels = json.dumps({"source": "event_tools", "tool": "metrics_snapshot"})
    for key, name in metric_names.items():
        await db.execute(
            "INSERT INTO metrics(ts,name,value,labels) VALUES(?,?,?,?)",
            (generated_at, name, float(counters[key]), labels),
        )
    snapshot = {
        "generated_at": generated_at,
        "counters": counters,
        "top_topics": [{"topic": row["topic"], "count": int(row["n"])} for row in topic_rows],
        "decisions": [
            {"decision": row["decision"], "count": int(row["n"])} for row in decision_rows
        ],
        "metrics_written": len(metric_names),
    }
    await bus.publish(
        Event(
            topic="observability.snapshot",
            source="observability",
            payload=snapshot,
            correlation_id=ctx.get("correlation_id"),
            created_at=generated_at,
        )
    )
    return {"success": True, **snapshot}


_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="event_query",
            description="Query durable event history with topic/correlation filters and bounded offset pages.",
            category="observability",
            risk=RiskLevel.LOW,
            permissions=["observability:read"],
            input_schema=_schema(
                {
                    "topic": {"type": "string", "minLength": 1},
                    "correlation_id": {"type": "string", "minLength": 1},
                    **_PAGE_PROPERTIES,
                }
            ),
            resource_field="",  # database rows, not a filesystem resource
        ),
        event_query,
    ),
    (
        ToolSpec(
            name="audit_query",
            description="Query durable permission/executor audit rows with bounded offset pages.",
            category="observability",
            risk=RiskLevel.LOW,
            permissions=["observability:read"],
            input_schema=_schema(
                {
                    "actor": {"type": "string", "minLength": 1},
                    "action": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "minLength": 1},
                    "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    **_PAGE_PROPERTIES,
                }
            ),
            resource_field="",
        ),
        audit_query,
    ),
    (
        ToolSpec(
            name="metrics_snapshot",
            description="Compute real event/audit/tool-call counters, persist them as metric rows and publish a snapshot event.",
            category="observability",
            risk=RiskLevel.MEDIUM,
            permissions=["observability:read", "metrics:write"],
            input_schema=_schema(
                {"topic_limit": {"type": "integer", "minimum": 1, "maximum": MAX_PAGE_SIZE}}
            ),
            resource_field="",
        ),
        metrics_snapshot,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the observability toolset; return newly registered tool names."""
    injected = dict(services or {})
    registered: list[str] = []
    for spec, handler in _SPECS:
        async def bound(args: dict[str, Any], ctx: dict[str, Any], _handler=handler) -> Any:
            merged = dict(ctx)
            merged.setdefault("services", injected)
            return await _handler(args, merged)

        try:
            registry.register(spec, bound)
        except ValueError:  # already present - preserve the caller's version
            continue
        registered.append(spec.name)
    return registered


__all__ = ["register", "event_query", "audit_query", "metrics_snapshot", "TOOL_NAMES"]
