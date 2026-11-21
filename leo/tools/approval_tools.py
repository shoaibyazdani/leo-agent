"""Durable approval and audit tools for the conversation tool loop.

These tools are a thin, typed surface over the *existing* security layer:
``approval_resolve`` records a human decision only through
``PermissionManagerImpl.resolve_approval`` (never by writing approval rows),
and uses the authenticated executor actor as the resolver. Callers cannot
choose the principal that is recorded for a decision. Binding (action,
resource, risk) and expiry are checked before a decision is recorded, and the
PermissionManager re-checks actor/action/resource/risk/expiry again when the
approved id is used by ``authorize``.

``audit_tail`` reads durable ``audit_log`` rows with an enforced upper bound so
an incident review cannot accidentally pull the whole table into a prompt.

A model must never decide an approval: ``register()`` exposes only the read-only
tools. The state transition is registered separately by
``register_operator()`` for a trusted, human/operator-controlled surface, and
its handler additionally refuses to let a requester resolve its own approval or
let a chat-sourced actor resolve any approval.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import db
from ..contracts import RiskLevel, ToolSpec

APPROVAL_LIST = "approval_list"
APPROVAL_RESOLVE = "approval_resolve"
AUDIT_TAIL = "audit_tail"

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
_STATUSES = ("pending", "approved", "denied", "all")
_FORBIDDEN_PRINCIPAL_KEYS = ("resolver", "actor", "principal", "requested_by", "user")
_CHAT_ACTOR_PREFIX = "chat:"


def _model_specs() -> list[ToolSpec]:
    """Fresh specs for the model-callable, read-only approval surface."""
    return [
        ToolSpec(
            name=APPROVAL_LIST,
            description=(
                "List approval requests from durable state. Pending rows come from the "
                "PermissionManager approval API; other/all statuses are read-only queries."
            ),
            category="approval",
            risk=RiskLevel.LOW,
            permissions=["approval", "audit"],
            resource_field="",  # arguments are identifiers, not filesystem paths
            input_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": list(_STATUSES)},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
                },
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name=AUDIT_TAIL,
            description=(
                "Read the most recent durable audit_log rows (bounded), optionally filtered "
                "by action/decision or an id cursor."
            ),
            category="audit",
            risk=RiskLevel.LOW,
            permissions=["audit"],
            resource_field="",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
                    "action": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "minLength": 1},
                    "after_id": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
        ),
    ]


def _operator_specs() -> list[ToolSpec]:
    """Fresh spec for the trusted operator-only approval decision surface."""
    return [
        ToolSpec(
            name=APPROVAL_RESOLVE,
            description=(
                "Operator-only: approve or deny a pending approval through the "
                "PermissionManager. The resolver is the authenticated executor actor; a "
                "caller-supplied principal is rejected. Action/resource/risk binding and "
                "expiry are verified first, and the requester cannot resolve its own "
                "approval."
            ),
            category="approval",
            # The human decision itself is the authorization: making this HIGH
            # would require an approval to resolve an approval. It is a
            # privileged but non-destructive state transition.
            risk=RiskLevel.MEDIUM,
            permissions=["approval"],
            resource_field="",
            input_schema={
                "type": "object",
                "properties": {
                    "approval_id": {"type": "string", "minLength": 1},
                    "approved": {"type": "boolean"},
                    "action": {"type": "string", "minLength": 1},
                    "resource": {"type": "string"},
                    "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["approval_id", "approved"],
                "additionalProperties": False,
            },
        ),
    ]


def _services(ctx: dict[str, Any]) -> dict[str, Any]:
    value = ctx.get("services")
    return value if isinstance(value, dict) else {}


def _permissions(ctx: dict[str, Any]) -> Any:
    """Prefer the injected service, then the executor-supplied permission manager."""
    manager = _services(ctx).get("permissions") or ctx.get("permissions")
    if manager is None:
        raise RuntimeError(
            "approval tools require a PermissionManager service "
            "(ctx['services']['permissions'] or ctx['permissions'])"
        )
    return manager


def _settings(ctx: dict[str, Any]) -> Any:
    settings = _services(ctx).get("settings") or ctx.get("settings")
    if settings is None:
        raise RuntimeError("approval tools require Settings (ctx['services']['settings'] or ctx['settings'])")
    return settings


def _bounded_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, value))


def _decoded(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _expired(row: dict[str, Any], settings: Any) -> bool:
    """Mirror the PermissionManager approval expiry window (read-only check)."""
    try:
        created = datetime.fromisoformat(str(row.get("created_at")))
    except (TypeError, ValueError):
        return True
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    security = getattr(settings, "security", None)
    timeout = getattr(security, "approval_timeout_seconds", 900)
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = 900
    return (datetime.now(timezone.utc) - created) > timedelta(seconds=timeout)


def _approval_view(row: Any) -> dict[str, Any]:
    data = dict(row)
    return {
        "id": data.get("id"),
        "status": data.get("status"),
        "actor": data.get("actor"),
        "action": data.get("action"),
        "resource": data.get("resource"),
        "risk": data.get("risk"),
        "reason": data.get("reason"),
        "requested_by": data.get("requested_by"),
        "created_at": data.get("created_at"),
        "resolved_at": data.get("resolved_at"),
        "metadata": _decoded(data.get("metadata")),
    }


async def _approval_list(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    permissions = _permissions(ctx)
    status = args.get("status") or "pending"
    limit = _bounded_limit(args.get("limit"))
    if status == "pending":
        rows = [dict(r) for r in await permissions.pending_approvals()]
        rows = rows[:limit]
    else:
        where = "" if status == "all" else " WHERE status=?"
        params: tuple[Any, ...] = () if status == "all" else (status,)
        rows = [
            dict(r)
            for r in await db.fetchall(
                f"SELECT * FROM approvals{where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            )
        ]
    return {
        "status": status,
        "limit": limit,
        "count": len(rows),
        "approvals": [_approval_view(r) for r in rows],
    }


async def _approval_resolve(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    permissions = _permissions(ctx)
    settings = _settings(ctx)
    forbidden = _FORBIDDEN_PRINCIPAL_KEYS & args.keys()
    if forbidden:
        raise ValueError("caller-chosen principal is not allowed: " + ", ".join(sorted(forbidden)))

    approval_id = str(args["approval_id"]).strip()
    approved = args["approved"]
    if not isinstance(approved, bool) or not approval_id:
        raise ValueError("approval_id and a boolean approved flag are required")
    # Executor-supplied authenticated actor, never a value from args.
    resolver = str(ctx.get("actor") or "local").strip() or "local"
    resolver_key = resolver.casefold()
    if resolver_key.startswith(_CHAT_ACTOR_PREFIX):
        raise ValueError("a chat actor cannot resolve an approval")

    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    if not row:
        raise ValueError(f"unknown approval: {approval_id}")
    record = dict(row)
    if record.get("status") != "pending":
        raise ValueError(f"approval already {record.get('status')}")
    requester = str(record.get("actor") or "").strip().casefold()
    if requester and requester == resolver_key:
        raise ValueError("the requester cannot resolve its own approval")

    for field in ("action", "resource", "risk"):
        if field in args and str(record.get(field)) != str(args[field]):
            raise ValueError(
                f"approval binding mismatch on {field}: "
                f"expected {str(record.get(field))!r}, got {str(args[field])!r}"
            )
    if _expired(record, settings):
        raise ValueError("approval has expired")

    # The only write path for the decision is the PermissionManager API; it
    # serializes the pending check and records the resolver in audit metadata.
    decision = await permissions.resolve_approval(approval_id, approved, resolver)
    return {
        "approval_id": approval_id,
        "status": "approved" if approved else "denied",
        "resolver": resolver,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "actor": record.get("actor"),
        "action": record.get("action"),
        "resource": record.get("resource"),
        "risk": record.get("risk"),
    }


async def _audit_tail(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    # Reading audit rows needs no service: audit_log is durable LEO state.
    limit = _bounded_limit(args.get("limit"))
    clauses: list[str] = []
    params: list[Any] = []
    for arg, column in (("action", "action"), ("decision", "decision")):
        if args.get(arg):
            clauses.append(f"{column}=?")
            params.append(str(args[arg]))
    if args.get("after_id") is not None and not isinstance(args["after_id"], bool):
        clauses.append("id>?")
        params.append(int(args["after_id"]))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = await db.fetchall(
        "SELECT id,ts,actor,action,resource,risk,decision,reason,metadata,correlation_id "
        f"FROM audit_log{where} ORDER BY id DESC LIMIT ?",
        (*params, limit),
    )
    items = [
        {
            "id": r["id"],
            "ts": r["ts"],
            "actor": r["actor"],
            "action": r["action"],
            "resource": r["resource"],
            "risk": r["risk"],
            "decision": r["decision"],
            "reason": r["reason"],
            "metadata": _decoded(r["metadata"]),
            "correlation_id": r["correlation_id"],
        }
        for r in rows
    ]
    return {"limit": limit, "count": len(items), "rows": items}


_HANDLERS = {
    APPROVAL_LIST: _approval_list,
    APPROVAL_RESOLVE: _approval_resolve,
    AUDIT_TAIL: _audit_tail,
}


def _register(registry: Any, specs: list[ToolSpec]) -> list[str]:
    names: list[str] = []
    for spec in specs:
        registry.register(spec, _HANDLERS[spec.name])
        names.append(spec.name)
    return names


def register(registry: Any, services: Any = None) -> list[str]:
    """Register only model-safe, read-only approval tools.

    ``approval_resolve`` is deliberately excluded: an LLM driving the tool loop
    must never be able to approve its own pending action. Trusted operator
    surfaces must call :func:`register_operator` explicitly.
    """
    return _register(registry, _model_specs())


def register_operator(registry: Any, services: Any = None) -> list[str]:
    """Register the human/operator-only approval decision tool."""
    return _register(registry, _operator_specs())


__all__ = [
    "register", "register_operator", "APPROVAL_LIST", "APPROVAL_RESOLVE", "AUDIT_TAIL",
]
