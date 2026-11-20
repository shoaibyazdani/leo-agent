"""Incident-triage specialist: classify real durable signals, never remediate.

The handler reads bounded pages of ``events`` and non-allowed ``audit_log``
rows from the shared database, classifies severity from those observed rows and
reports candidate causes with counts. When an ``events`` service is injected it
also emits one ``incident.triaged`` event as an auditable side effect. It never
resolves an approval, cancels a task or changes any of the signals it reads.

Services are resolved only by ``make_handler(services)`` at bind time; the call
context supplies incident filters (topic, correlation id, limit), never service
objects. This is a bespoke handler (``handler_kind = bespoke:make_handler``).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from ... import db

SEED: dict[str, Any] = {
    "id": "incident-triage",
    "name": "Incident Triage",
    "purpose": (
        "Triage incidents from durable events and permission audit rows, assign "
        "severity from observed signals and report evidence-backed candidate causes "
        "without approving, cancelling or otherwise changing the system."
    ),
    "capabilities": [
        "signal_collection",
        "severity_classification",
        "root_cause_candidates",
        "evidence_reporting",
    ],
    # Exact live read-only tool names from event/audit/tool observability sets.
    "tool_permissions": ["event_query", "audit_query", "metrics_snapshot"],
    "model_preferences": ["reasoning", "structured"],
    "context_budget": 20000,
    "task_limit": 2,
    "metadata": {"handler_kind": "bespoke:make_handler"},
}

_ERROR_TOKENS = ("error", "failed", "failure", "exception", "fatal", "critical")
_WARN_TOKENS = ("warn", "blocked", "degraded", "retry", "denied", "rejected")


def _limit(context: Mapping[str, Any]) -> int:
    try:
        value = int(context.get("limit", 100))
    except (TypeError, ValueError):
        value = 100
    return max(1, min(value, 500))


def _decoded(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _classify(topic: str, payload: Any) -> str:
    lowered = topic.lower()
    if any(token in lowered for token in _ERROR_TOKENS):
        level = "error"
    elif any(token in lowered for token in _WARN_TOKENS):
        level = "warning"
    else:
        level = "info"
    if isinstance(payload, Mapping):
        status = str(payload.get("status") or "").lower()
        if payload.get("error") or payload.get("success") is False or status in ("failed", "error"):
            level = "error"
        elif level == "info" and (payload.get("warning") or status in ("blocked", "degraded")):
            level = "warning"
    return level


def _severity(signals: list[dict[str, Any]], denials: list[dict[str, Any]]) -> str:
    if any(item["decision"] == "denied" and str(item["risk"]).lower() == "high" for item in denials):
        return "critical"
    if any(item["level"] == "error" for item in signals) or denials:
        return "high"
    if any(item["level"] == "warning" for item in signals):
        return "medium"
    if signals:
        return "low"
    return "none"


def _candidates(signals: list[dict[str, Any]], denials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flagged = [item for item in signals if item["level"] in ("error", "warning")]
    causes: list[dict[str, Any]] = []
    for topic, count in Counter(item["topic"] for item in flagged).most_common(5):
        causes.append({"kind": "event_topic", "value": topic, "count": count})
    for action, count in Counter(item["action"] for item in denials).most_common(5):
        causes.append({"kind": "denied_action", "value": action, "count": count})
    for reason, count in Counter(item["reason"] for item in denials if item["reason"]).most_common(5):
        causes.append({"kind": "denial_reason", "value": reason, "count": count})
    return causes


def make_handler(services: Mapping[str, Any] | None = None) -> Callable[..., Any]:
    """Bind the triage handler to injected services (only ``events`` is used)."""

    services = dict(services or {})
    events = services.get("events")

    async def handler(objective: Any = None, context: Any = None, **_: Any) -> dict[str, Any]:
        ctx = dict(context or {})
        limit = _limit(ctx)
        where = ""
        params: list[Any] = []
        topic = ctx.get("topic")
        correlation_id = ctx.get("correlation_id")
        clauses: list[str] = []
        if isinstance(topic, str) and topic.strip():
            clauses.append("topic=?")
            params.append(topic.strip())
        if isinstance(correlation_id, str) and correlation_id.strip():
            clauses.append("correlation_id=?")
            params.append(correlation_id.strip())
        if clauses:
            where = " WHERE " + " AND ".join(clauses)

        rows = await db.fetchall(
            "SELECT topic,source,payload,correlation_id,created_at FROM events"
            + where
            + " ORDER BY created_at DESC, rowid DESC LIMIT ?",
            [*params, limit],
        )
        signals: list[dict[str, Any]] = []
        for row in rows:
            payload = _decoded(row["payload"])
            signals.append({
                "topic": str(row["topic"]),
                "source": str(row["source"]),
                "level": _classify(str(row["topic"]), payload),
                "correlation_id": row["correlation_id"],
                "created_at": row["created_at"],
            })

        audit_rows = await db.fetchall(
            "SELECT action,resource,risk,decision,reason,ts FROM audit_log "
            "WHERE decision<>'allowed' ORDER BY ts DESC, id DESC LIMIT ?",
            (limit,),
        )
        denials: list[dict[str, Any]] = [
            {
                "action": str(row["action"]),
                "resource": str(row["resource"] or ""),
                "risk": str(row["risk"] or ""),
                "decision": str(row["decision"]),
                "reason": str(row["reason"] or ""),
                "ts": str(row["ts"]),
            }
            for row in audit_rows
        ]

        severity = _severity(signals, denials)
        causes = _candidates(signals, denials)
        emitted = False
        if events is not None and callable(getattr(events, "emit", None)):
            await events.emit(
                "incident.triaged",
                severity=severity,
                signal_count=len(signals),
                non_allowed_count=len(denials),
            )
            emitted = True

        summary = {
            "severity": severity,
            "signal_count": len(signals),
            "non_allowed_count": len(denials),
            "error_count": sum(1 for item in signals if item["level"] == "error"),
            "warning_count": sum(1 for item in signals if item["level"] == "warning"),
            "candidate_causes": causes,
            "status": "no-signals" if not signals and not denials else "signals-found",
        }
        evidence = [
            {
                "source": "incident-triage",
                "claim": (
                    f"Triaged {len(signals)} event rows and {len(denials)} non-allowed "
                    f"audit rows into severity {severity}"
                ),
                "signal_count": len(signals),
                "non_allowed_count": len(denials),
            },
            {
                "source": "incident-triage",
                "claim": f"triage event emitted: {emitted}",
                "emitted": emitted,
            },
        ]
        return {
            "success": True,
            "output": {
                "severity": severity,
                "status": summary["status"],
                "summary": summary,
                "signals": signals,
                "non_allowed": denials,
                "emitted_triage_event": emitted,
                "actions_taken": [],
            },
            "evidence": evidence,
        }

    return handler
