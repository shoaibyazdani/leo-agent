"""Conversation-callable agent delegation tools.

These tools expose the existing durable ``AgentRegistry`` / ``AgentEngine`` and
the shared task database to the chat tool loop. Delegation always goes through
``AgentEngine.delegate`` (real task + ``agent_runs`` rows); status and result
read those durable rows back instead of trusting an in-process reply.

Services are injected by the composition root under ``ctx["services"]``. A
``services`` mapping passed to :func:`register` is used as a fallback for hosts
that register the tools before the executor is wired up; per-call context always
wins. Missing services are an explicit error, never a canned success.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .. import db
from ..contracts import RiskLevel, TaskStatus, ToolSpec

_REGISTRY_KEYS = ("agent_registry", "agents")
_ENGINE_KEYS = ("agent_engine", "agents_engine")

TOOL_NAMES = ("agent_list", "agent_delegate", "agent_status", "agent_result")


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _service(ctx: dict[str, Any], fallback: Any, keys: tuple[str, ...]) -> Any:
    """Resolve one injected service, preferring the per-call context."""
    per_call = ctx.get("services")
    if per_call is not None and not isinstance(per_call, Mapping):
        raise RuntimeError("ctx['services'] must be a mapping of service names to services")
    for services in (per_call, fallback):
        if not isinstance(services, Mapping):
            continue
        for key in keys:
            value = services.get(key)
            if value is not None:
                return value
    raise RuntimeError(
        f"missing required service {keys[0]!r} in ctx['services'] "
        f"(accepted keys: {', '.join(keys)})"
    )


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return default


def _spec_view(spec: Any) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "purpose": spec.purpose,
        "capabilities": list(getattr(spec, "capabilities", None) or []),
        "tool_permissions": list(getattr(spec, "tool_permissions", None) or []),
        "enabled": bool(getattr(spec, "enabled", True)),
        "context_budget": int(getattr(spec, "context_budget", 0) or 0),
        "task_limit": int(getattr(spec, "task_limit", 0) or 0),
    }


async def _runs(task_id: str) -> list[dict[str, Any]]:
    rows = await db.fetchall(
        "SELECT id, agent_id, status, started_at, finished_at FROM agent_runs "
        "WHERE task_id=? ORDER BY started_at DESC",
        (task_id,),
    )
    return [dict(row) for row in rows]


async def _task_row(task_id: str) -> Any:
    row = await db.fetchone(
        "SELECT id, objective, status, owner_agent, result, errors, evidence, "
        "created_at, updated_at, completed_at FROM tasks WHERE id=?",
        (task_id,),
    )
    if row is None:
        raise LookupError(f"unknown task: {task_id}")
    return row


async def agent_list(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """List persisted agent specs from the injected registry."""
    registry = _service(ctx, fallback, _REGISTRY_KEYS)
    specs = await registry.list(enabled_only=bool(args.get("enabled_only", False)))
    views = [_spec_view(spec) for spec in specs]
    return {"agents": views, "count": len(views)}


async def agent_delegate(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """Delegate one objective through the real engine and return durable IDs."""
    registry = _service(ctx, fallback, _REGISTRY_KEYS)
    engine = _service(ctx, fallback, _ENGINE_KEYS)
    agent_id = args.get("agent_id")
    if agent_id:
        spec = await registry.get(agent_id)
        if spec is None:
            raise LookupError(f"unknown agent: {agent_id}")
        if not getattr(spec, "enabled", False):
            raise PermissionError(f"agent is disabled: {agent_id}")

    context = dict(args.get("context") or {})
    correlation_id = ctx.get("correlation_id")
    if correlation_id and not context.get("correlation_id"):
        context["correlation_id"] = str(correlation_id)

    result = await engine.delegate(
        args["objective"],
        agent_id=agent_id,
        context=context,
        verify=bool(args.get("verify", True)),
    )
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    task_id = getattr(result, "task_id", None)
    if not task_id:
        raise RuntimeError("delegation did not return a task id")

    task = await _task_row(task_id)
    runs = await _runs(task_id)
    evidence: list[dict[str, Any]] = []
    for item in getattr(result, "evidence", None) or []:
        evidence.append(dict(item) if isinstance(item, Mapping) else {"claim": str(item)})
    return {
        "task_id": task["id"],
        "run_id": runs[0]["id"] if runs else None,
        "agent_id": getattr(result, "agent_id", None) or agent_id,
        "status": task["status"],
        "success": bool(getattr(result, "success", False)),
        "objective": task["objective"],
        "output": getattr(result, "output", None),
        "error": getattr(result, "error", None),
        "evidence": evidence,
        "run_count": len(runs),
    }


async def agent_status(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """Read the persisted state of a task and its agent runs."""
    _service(ctx, fallback, _REGISTRY_KEYS)
    task = await _task_row(args["task_id"])
    runs = await _runs(args["task_id"])
    return {
        "task_id": task["id"],
        "objective": task["objective"],
        "status": task["status"],
        "agent_id": task["owner_agent"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "completed_at": task["completed_at"],
        "runs": runs,
        "run_count": len(runs),
    }


async def agent_result(args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None) -> Any:
    """Read a finished delegation's persisted result, never a cached reply."""
    _service(ctx, fallback, _REGISTRY_KEYS)
    task = await _task_row(args["task_id"])
    result = _loads(task["result"], None)
    if result is None:
        run = await db.fetchone(
            "SELECT result FROM agent_runs WHERE task_id=? AND result IS NOT NULL "
            "ORDER BY started_at DESC LIMIT 1",
            (args["task_id"],),
        )
        result = _loads(run["result"], None) if run else None
    errors = _loads(task["errors"], []) or []
    try:
        ready = TaskStatus(task["status"]).terminal
    except ValueError:
        ready = False
    return {
        "task_id": task["id"],
        "agent_id": task["owner_agent"],
        "status": task["status"],
        "ready": bool(ready),
        "result": result,
        "error": errors[-1] if errors else None,
        "evidence": _loads(task["evidence"], []) or [],
    }


_SPECS: tuple[tuple[ToolSpec, Any], ...] = (
    (
        ToolSpec(
            name="agent_list",
            description="List registered agent identities and their declared capabilities from durable storage.",
            category="agents",
            risk=RiskLevel.LOW,
            permissions=["agent:read"],
            input_schema=_schema({"enabled_only": {"type": "boolean"}}, []),
            resource_field="",
        ),
        agent_list,
    ),
    (
        ToolSpec(
            name="agent_delegate",
            description="Delegate an objective to a real agent; creates a durable task and agent run through the agent engine.",
            category="agents",
            risk=RiskLevel.MEDIUM,
            permissions=["agent:delegate"],
            input_schema=_schema(
                {
                    "objective": {"type": "string", "minLength": 1},
                    "agent_id": {"type": "string", "minLength": 1},
                    # Only inert coordination metadata crosses the delegation
                    # boundary. Approval flags, service objects and tool
                    # facades are reserved for the engine, never model input.
                    "context": {
                        "type": "object",
                        "properties": {
                            "correlation_id": {"type": "string", "minLength": 1},
                            "task_id": {"type": "string", "minLength": 1},
                            "objective": {"type": "string", "minLength": 1},
                            "notes": {"type": "string"},
                            "parameters": {"type": "object"},
                            "input": {"type": "object"},
                            "evidence": {"type": "array", "items": {"type": "object"}},
                        },
                        "additionalProperties": False,
                    },
                    "verify": {"type": "boolean"},
                },
                ["objective"],
            ),
            resource_field="",
        ),
        agent_delegate,
    ),
    (
        ToolSpec(
            name="agent_status",
            description="Read the durable status of an agent task and its execution runs.",
            category="agents",
            risk=RiskLevel.LOW,
            permissions=["agent:read"],
            input_schema=_schema({"task_id": {"type": "string", "minLength": 1}}, ["task_id"]),
            resource_field="",
        ),
        agent_status,
    ),
    (
        ToolSpec(
            name="agent_result",
            description="Read the persisted result and evidence of an agent task by task id.",
            category="agents",
            risk=RiskLevel.LOW,
            permissions=["agent:read"],
            input_schema=_schema({"task_id": {"type": "string", "minLength": 1}}, ["task_id"]),
            resource_field="",
        ),
        agent_result,
    ),
)


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the four agent tools; return their names.

    ``services`` is an optional fallback injection source. The executor's
    per-call ``ctx['services']`` is always preferred.
    """
    names: list[str] = []
    for spec, handler in _SPECS:
        async def call(args: dict[str, Any], ctx: dict[str, Any], _handler=handler) -> Any:
            return await _handler(args, ctx, services)

        try:
            registry.register(spec, call)
        except ValueError:
            # Another owner registered this name first; keep that version.
            pass
        names.append(spec.name)
    return names


__all__ = ["register", "TOOL_NAMES"]
