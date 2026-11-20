"""Automation-operator specialist: durable schedules, webhook triggers and queued work.

This spec uses a bespoke handler (``make_handler``), not the default model
delegation worker. Every action is executed through the agent's *scoped* tool
view against the real automation/task/webhook toolsets; the handler never
writes schedule or task rows itself and never calls a provider or the network.

Two honesty rules are enforced here:

* Queued is not executed. ``trigger_schedule`` and ``enqueue_task`` report the
  task id and the persisted status read back with ``task_status``, with
  ``executed=False`` and ``completed=False``. Only a task whose durable status
  says ``completed`` (via ``task_status``/``task_result``) is reported as
  completed. A webhook delivery is reported as an acceptance event, never as
  proof that a worker ran.
* Mutations need a real durable approval. ``create_schedule``,
  ``delete_schedule``, ``trigger_schedule`` and ``enqueue_task`` require an
  ``approval_id`` naming an approved row in the PermissionManager approval
  store, bound to this agent, action, resource and exact arguments digest. A
  free-text ``approved``/``confirm``/``force`` context flag is deliberately
  ignored; the decision is re-verified through ``permissions.authorize`` so it
  is written to the durable audit log with the approving id.

Trust boundary: settings, permissions, tools, scheduler, tasks and webhooks are
resolved only from the injected ``services`` mapping (or the engine's scoped
tool façade), never from request/handler context. ``context['actor']`` cannot
change the fixed ``automation-operator`` actor recorded on approvals, tool
calls and audit rows.

Evidence is not self-attested: every evidence item is a tool-call anchor that
names a real, successful ``ToolResult.call_id``, its exact arguments and its
observed output with a SHA-256 content digest. A final projection anchor binds
the reported output digest to that same audited call. No bare ``claim`` /
``verified`` / ``supports`` assertion is emitted.

The module exposes ``tool_arguments(action, request)`` and
``approval_target(tool, arguments)`` so an operator (or a test) can request
precisely the approval a mutation will demand. Metadata
carries only ``handler_kind``; a string-valued ``handler`` key would be treated
as a callable by ``AgentEngine`` and is never present here.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ... import db
from ...contracts import RiskLevel, TaskStatus, ToolCall
from ...security.permissions import arguments_digest
from ..verification import digest as evidence_digest

AGENT_ID = "automation-operator"
APPROVAL_ACTION = "automation:mutate"
APPROVAL_RISK = RiskLevel.HIGH

# Exact ToolSpec names, not categories. A category grant (e.g. "tasks" or
# "approval") would also admit tools this specialist must never call, and would
# widen the model-worker fallback if no handler were installed.
TOOL_GRANTS: tuple[str, ...] = (
    "schedule_create",
    "schedule_list",
    "schedule_delete",
    "schedule_run_now",
    "task_create",
    "task_status",
    "task_result",
    "webhook_routes",
    "webhook_verify",
    "webhook_deliveries",
)

SEED: dict[str, Any] = {
    "id": AGENT_ID,
    "name": "Automation Operator",
    "purpose": (
        "Manage cron/interval schedules, signed-webhook triggers and durable "
        "automations through the real task engine, and report queued, delivered "
        "and completed outcomes exactly as durable state records them."
    ),
    "capabilities": [
        "schedule_management",
        "cron_and_interval_triggers",
        "webhook_trigger_reporting",
        "durable_task_queuing",
        "truthful_delivery_reporting",
    ],
    "tool_permissions": list(TOOL_GRANTS),
    "model_preferences": ["tools", "structured"],
    "context_budget": 16000,
    "task_limit": 2,
    "metadata": {
        "tier": "operations",
        "risk": "medium",
        "handler_kind": "bespoke:make_handler",
        "approval_authority": "human_only",
        "never_self_approve": True,
        "mutation_gate": "durable_approval",
        "approval_action": APPROVAL_ACTION,
        "delivery_honesty": "queued_not_executed",
    },
}

# action -> (tool used, needs a durable approval)
_ACTIONS: dict[str, tuple[str, bool]] = {
    "list_schedules": ("schedule_list", False),
    "create_schedule": ("schedule_create", True),
    "delete_schedule": ("schedule_delete", True),
    "trigger_schedule": ("schedule_run_now", True),
    "enqueue_task": ("task_create", True),
    "task_status": ("task_status", False),
    "task_result": ("task_result", False),
    "webhook_routes": ("webhook_routes", False),
    "verify_webhook": ("webhook_verify", False),
    "delivery_status": ("webhook_deliveries", False),
}
_ALIASES: dict[str, str] = {
    "list": "list_schedules",
    "schedule_list": "list_schedules",
    "create": "create_schedule",
    "schedule_create": "create_schedule",
    "delete": "delete_schedule",
    "schedule_delete": "delete_schedule",
    "trigger": "trigger_schedule",
    "run_now": "trigger_schedule",
    "schedule_run_now": "trigger_schedule",
    "enqueue": "enqueue_task",
    "task_create": "enqueue_task",
    "status": "task_status",
    "result": "task_result",
    "routes": "webhook_routes",
    "verify": "verify_webhook",
    "webhook_verify": "verify_webhook",
    "deliveries": "delivery_status",
    "webhook_deliveries": "delivery_status",
}
_REQUEST_KEYS: tuple[str, ...] = (
    "action", "name", "expression", "objective", "task_objective", "payload",
    "enabled_only", "run_at", "task_id", "headers", "body", "delivery_id",
    "limit", "offset", "channel", "approval_id", "idempotency_key",
)
# These are never authority. Their presence is reported, not honoured.
_APPROVAL_FLAGS: tuple[str, ...] = (
    "approved", "approve", "confirm", "confirmed", "force", "allow",
    "authorize", "authorized", "bypass_approval",
)
_TERMINAL = frozenset(status.value for status in TaskStatus if status.terminal)


def approval_target(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Binding an operator must approve before this specialist may mutate.

    The returned values are the exact action, resource, risk and arguments
    digest that :func:`make_handler` re-verifies. ``arguments_digest`` is the
    canonical hash from the PermissionManager, so changing a schedule name,
    expression, objective or task id invalidates the earlier decision.
    """
    if tool_name not in {tool for tool, mutating in _ACTIONS.values() if mutating}:
        raise ValueError(f"tool {tool_name!r} is not a mutating automation action")
    subject = ""
    for key in ("name", "task_id", "objective"):
        if arguments.get(key):
            subject = str(arguments[key])
            break
    return {
        "action": APPROVAL_ACTION,
        "resource": f"{tool_name}:{subject[:160]}",
        "risk": APPROVAL_RISK.value,
        "arguments_digest": arguments_digest(
            {"tool": tool_name, "arguments": dict(arguments)}
        ),
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        try:
            return dict(zip(value.keys(), value))
        except (TypeError, ValueError):
            pass
    return {}


def _service(services: Mapping[str, Any], *names: str) -> Any:
    """Resolve a runtime service from the trusted injection mapping only.

    Request context is caller-influenced: it must never be able to substitute
    permissions, settings, tools, scheduler, tasks or webhook services. Those
    objects come exclusively from ``make_handler(services)``; tools themselves
    read scheduler/tasks/webhooks from the executor's injected service mapping.
    """
    for name in names:
        value = services.get(name)
        if value is not None:
            return value
    return None


def _parse_request(objective: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {}
    if isinstance(objective, Mapping):
        request.update(dict(objective))
    elif isinstance(objective, str):
        text = objective.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, Mapping):
                request.update(dict(parsed))
    # Engine context carries its *own* task_id/correlation id. Never let it
    # shadow the automation target explicitly supplied in the objective.
    for key in ("request", "automation", "automation_request"):
        nested = context.get(key)
        if isinstance(nested, Mapping):
            for name, value in nested.items():
                request.setdefault(name, value)
    for key in _REQUEST_KEYS:
        if key in context:
            request.setdefault(key, context[key])
    return request


def _normalize_action(value: Any) -> str:
    action = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(action, action)


def _required(request: Mapping[str, Any], key: str) -> Any:
    value = request.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"requires {key!r}")
    return value


def _tool_arguments(action: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Build exact ToolSpec arguments; unknown request keys are never forwarded."""
    if action == "list_schedules":
        return {"enabled_only": bool(request.get("enabled_only", False))}
    if action == "create_schedule":
        objective = request.get("objective") or request.get("task_objective")
        if objective is None or not str(objective).strip():
            raise ValueError("requires 'objective'")
        args = {
            "name": str(_required(request, "name")),
            "expression": _required(request, "expression"),
            "objective": str(objective),
        }
        payload = request.get("payload")
        if payload is not None:
            if not isinstance(payload, Mapping):
                raise ValueError("payload must be an object")
            args["payload"] = dict(payload)
        return args
    if action == "delete_schedule":
        return {"name": str(_required(request, "name"))}
    if action == "trigger_schedule":
        args = {"name": str(_required(request, "name"))}
        if request.get("run_at") is not None:
            args["run_at"] = request["run_at"]
        return args
    if action == "enqueue_task":
        objective = request.get("objective") or request.get("task_objective")
        if objective is None or not str(objective).strip():
            raise ValueError("requires 'objective'")
        args: dict[str, Any] = {
            "objective": str(objective),
            "metadata": {"source": AGENT_ID, "action": action},
        }
        if request.get("idempotency_key") is not None:
            args["idempotency_key"] = request["idempotency_key"]
        return args
    if action in ("task_status", "task_result"):
        return {"task_id": str(_required(request, "task_id"))}
    if action == "webhook_routes":
        if request.get("channel") is not None:
            return {"channel": str(request["channel"])}
        return {}
    if action == "verify_webhook":
        headers = request.get("headers")
        if not isinstance(headers, Mapping):
            raise ValueError("requires an object-valued 'headers'")
        args = {"body": _required(request, "body"), "headers": dict(headers)}
        if request.get("delivery_id") is not None:
            args["delivery_id"] = request["delivery_id"]
        return args
    if action == "delivery_status":
        args = {}
        for key, default in (("limit", 50), ("offset", 0)):
            value = request.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key!r} must be an integer")
            args[key] = value
        if request.get("delivery_id") is not None:
            args["delivery_id"] = request["delivery_id"]
        return args
    raise ValueError(f"unsupported action {action!r}")


def tool_arguments(action: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Exact ToolSpec arguments this action will use.

    Operators use this with :func:`approval_target` so the human-approved
    digest is byte-for-byte the one the handler re-verifies.
    """
    return _tool_arguments(action, request)


def _tool_value(result: Any) -> Any:
    if hasattr(result, "success") and not getattr(result, "success"):
        raise RuntimeError(str(getattr(result, "error", None) or "tool call failed"))
    return getattr(result, "output", result)


async def _call(
    tool_view: Any, name: str, arguments: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Execute one scoped tool call and return its output plus an audit anchor.

    The anchor uses the executor's own call id and exact arguments so the
    independent verifier can match it to the durable ``tool_calls`` row; its
    digest is computed over the actually observed output.
    """
    call = ToolCall(
        name=name,
        arguments=dict(arguments),
        actor=AGENT_ID,
        correlation_id=context.get("correlation_id"),
    )
    result = await tool_view.execute(call)
    output = _tool_value(result)
    anchor = {
        "id": str(getattr(result, "call_id", None) or call.id),
        "tool": name,
        "arguments": dict(arguments),
        "output": output,
        "digest": evidence_digest(output),
    }
    return output, anchor


def _anchor_item(anchor: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a call anchor in the verifier's recognised, auditable form."""
    return {"source": "tool_call", "tool_call": dict(anchor)}


def _blocked(action: str, tool_name: str, approval: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "agent_id": AGENT_ID,
        "action": action,
        "tool": tool_name,
        "status": "blocked_approval_required",
        "applied": False,
        "executed": False,
        "completed": False,
        "approval": dict(approval),
        "delivery": {
            "outcome": "blocked",
            "acknowledged": False,
            "executed": False,
            "completed": False,
            "source": "durable_approval_gate",
        },
        "objective": "",
    }
    return {
        "success": False,
        "agent_id": AGENT_ID,
        "error": str(approval.get("reason") or "durable approval required"),
        "output": output,
        # No tool ran, so there is no tool-call anchor to claim. Report the
        # gate outcome as data only, never as a self-attested verdict.
        "evidence": [{
            "source": "approval_gate",
            "gate": "durable_approval",
            "action": action,
            "reason": str(approval.get("reason") or ""),
            "approval": dict(approval),
        }],
    }


def _failed(action: str, tool_name: str, error: str, approval: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "success": False,
        "agent_id": AGENT_ID,
        "error": error,
        "output": {
            "agent_id": AGENT_ID,
            "action": action,
            "tool": tool_name,
            "status": "failed",
            "applied": False,
            "executed": False,
            "completed": False,
            "approval": dict(approval),
            "delivery": {"outcome": "failed", "executed": False, "completed": False},
        },
        "evidence": [{
            "source": "tool_error",
            "tool": tool_name,
            "error": error,
            "approval": dict(approval),
        }],
    }


async def _verify_approval(
    request: Mapping[str, Any],
    context: Mapping[str, Any],
    services: Mapping[str, Any],
    tool_name: str,
    arguments: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Require an approved durable row bound to this exact mutation.

    Permissions and settings are read only from the trusted injected services;
    neither the request nor the handler context can supply substitutes. The
    actor is the fixed specialist id, never ``context['actor']``.
    """
    target = approval_target(tool_name, arguments)
    raw_id = request.get("approval_id") or context.get("approval_id")
    flags = sorted(
        str(key) for key in _APPROVAL_FLAGS if key in request or key in context
    )
    info: dict[str, Any] = {
        "required": True,
        "approved": False,
        "approval_id": raw_id.strip() if isinstance(raw_id, str) else None,
        "action": target["action"],
        "resource": target["resource"],
        "risk": target["risk"],
        "arguments_digest": target["arguments_digest"],
        "ignored_context_flags": flags,
        # Diagnostic only: never used for authorization or for the actor passed
        # to authorize()/ToolCall, which is always the fixed specialist id.
        "actor": AGENT_ID,
        "context_actor_ignored": bool(
            context.get("actor") and str(context.get("actor")) != AGENT_ID
        ),
        "granted": False,
        "reason": "",
    }

    def deny(reason: str) -> tuple[bool, dict[str, Any]]:
        return False, {**info, "reason": reason}

    if not info["approval_id"]:
        return deny(
            "mutation requires a durable approval id; free-text context flags "
            "are not approval"
        )
    approval_id = info["approval_id"]
    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    if row is None:
        return deny(f"unknown approval: {approval_id}")
    if row["status"] != "approved":
        return deny(f"approval {approval_id} is {row['status']}, not approved")

    for field in ("action", "resource", "risk", "actor"):
        expected = {
            "action": target["action"],
            "resource": target["resource"],
            "risk": target["risk"],
            "actor": AGENT_ID,
        }[field]
        if str(row[field]) != str(expected):
            return deny(f"approval binding mismatch on {field}")

    metadata: Mapping[str, Any] = {}
    try:
        decoded = json.loads(row["metadata"] or "{}")
    except (TypeError, ValueError):
        decoded = {}
    if isinstance(decoded, Mapping):
        metadata = decoded
    expected_digest = metadata.get("arguments_digest")
    if expected_digest and not hmac.compare_digest(
        str(expected_digest), str(target["arguments_digest"])
    ):
        return deny("approval arguments digest does not match this mutation")

    # Trust boundary: never read settings from request context or from a
    # caller-supplied tool object; an attacker could otherwise disable expiry.
    settings = _service(services, "settings")
    if settings is None:
        return deny("settings service unavailable for approval expiry verification")
    security = getattr(settings, "security", None)
    try:
        timeout = int(getattr(security, "approval_timeout_seconds", 900))
    except (TypeError, ValueError):
        timeout = 900
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        expired = (datetime.now(timezone.utc) - created).total_seconds() > timeout
    except (TypeError, ValueError):
        expired = True
    if expired:
        return deny(f"approval {approval_id} has expired")

    # Trust boundary: only the injected PermissionManager can verify and audit
    # a decision. A forged context['permissions'] must never approve itself.
    permissions = _service(services, "permissions")
    if permissions is None or not hasattr(permissions, "authorize"):
        return deny("permissions service unavailable for approval verification")
    decision = await permissions.authorize(
        AGENT_ID,
        target["action"],
        target["resource"],
        APPROVAL_RISK,
        approval_id=approval_id,
        arguments_digest=target["arguments_digest"],
        task_id=context.get("task_id"),
        tool=tool_name,
    )
    if not decision.allowed or str(getattr(decision, "approval_id", None)) != approval_id:
        return deny(f"PermissionManager did not verify approval {approval_id}")

    info.update({
        "approved": True,
        "granted": True,
        "reason": decision.reason,
        "resolver": metadata.get("resolved_by"),
        "audit_decision": "allowed",
    })
    return True, info


async def _shape(
    action: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    tool_output: Any,
    approval: Mapping[str, Any],
    anchors: list[dict[str, Any]],
    tool_view: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only what durable tool output actually confirms."""
    data = _mapping(tool_output)
    result: dict[str, Any] = {"tool_output": dict(data)}
    status = "observed"
    applied = False
    executed = False
    completed = False
    delivery: dict[str, Any] = {
        "outcome": "observed", "executed": False, "completed": False,
        "source": tool_name,
    }

    if action in ("create_schedule", "delete_schedule"):
        schedules, list_anchor = await _call(
            tool_view, "schedule_list", {"enabled_only": False}, context
        )
        anchors.append(list_anchor)
        rows = [s for s in (_mapping(item) for item in _mapping(schedules).get("schedules") or [])
                if s.get("name") == arguments.get("name")]
        applied = bool(rows) if action == "create_schedule" else not rows
        status = "applied" if applied else "unverified"
        result["schedule"] = rows[0] if rows else None
        result["verified_by"] = "schedule_list"
        result["schedule_readback"] = {"name": arguments.get("name"), "present": bool(rows)}
        delivery["outcome"] = "applied" if applied else "unverified"

    elif action in ("trigger_schedule", "enqueue_task"):
        task_id = data.get("task_id")
        task: dict[str, Any] | None = None
        if task_id:
            task_raw, status_anchor = await _call(
                tool_view, "task_status", {"task_id": task_id}, context
            )
            anchors.append(status_anchor)
            task = _mapping(task_raw)
        task_status = task.get("status") if task else data.get("task_status")
        queued = bool(task_id) and bool(data.get("triggered", True))
        status = "queued" if queued else "not_queued"
        executed = False
        completed = str(task_status) == TaskStatus.COMPLETED.value
        result.update({
            "schedule": data.get("schedule"),
            "task_id": task_id,
            "task_status": task_status,
            "triggered": bool(data.get("triggered", True)),
            "task_count": data.get("task_count"),
            "next_run_at": data.get("next_run_at"),
            "last_run_at": data.get("last_run_at"),
            "verified_by": "task_status",
            "task_readback": {"task_id": task_id, "status": task_status},
        })
        delivery.update({
            "outcome": status,
            "acknowledged": bool(queued),
            "executed": False,
            "completed": completed,
            "note": "Durable task created; no worker execution is claimed.",
        })

    elif action == "task_status":
        task_status = str(data.get("status") or "unknown")
        completed = task_status == TaskStatus.COMPLETED.value or data.get("completed") is True
        status = "completed" if completed else task_status
        result.update({
            "task_id": arguments.get("task_id"),
            "status": task_status,
            "terminal": task_status in _TERMINAL,
            "verified_by": "task_status",
        })
        delivery.update({
            "outcome": task_status, "completed": completed,
            "note": "Status is a durable read-back, not an execution claim.",
        })

    elif action == "task_result":
        task_status = str(data.get("status") or "unknown")
        completed = data.get("completed") is True or task_status == TaskStatus.COMPLETED.value
        status = "completed" if completed else "not_completed"
        result.update({
            "task_id": arguments.get("task_id"),
            "status": task_status,
            "completed": completed,
            "result": data.get("result"),
            "evidence": list(data.get("evidence") or []),
            "verified_by": "task_result",
        })
        delivery.update({"outcome": status, "completed": completed})

    elif action == "webhook_routes":
        routes = list(data.get("routes") or [])
        result.update({
            "routes": routes, "count": len(routes),
            "signature": data.get("signature"),
            "task_intake": data.get("task_intake"),
            "verified_by": "webhook_routes",
        })
        delivery.update({
            "outcome": "observed", "route_count": len(routes),
            "note": "Route configuration is not evidence of a delivery.",
        })

    elif action == "verify_webhook":
        verified = data.get("verified") is True
        status = "verified" if verified else "unverified"
        result.update({
            "verified": verified,
            "delivery_id": data.get("delivery_id"),
            "signature_scheme": data.get("signature_scheme"),
            "verified_by": "webhook_verify",
        })
        delivery.update({
            "outcome": status, "verified": verified, "executed": False,
            "completed": False,
            "note": "Signature verification is not delivery or execution.",
        })

    elif action == "delivery_status":
        deliveries = [_mapping(item) for item in (data.get("deliveries") or [])]
        accepted = sum(1 for item in deliveries if item.get("status") == "accepted")
        rejected = sum(1 for item in deliveries if item.get("status") == "rejected")
        completed = any(
            str(item.get("task_status")) == TaskStatus.COMPLETED.value for item in deliveries
        )
        status = "accepted" if accepted else "rejected" if rejected else "no_deliveries"
        result.update({
            "deliveries": deliveries,
            "total": data.get("total"),
            "next_offset": data.get("next_offset"),
            "verified_by": "webhook_deliveries",
        })
        delivery.update({
            "outcome": status,
            "accepted": accepted,
            "rejected": rejected,
            "executed": False,
            "completed": completed,
            "note": "An accepted webhook event is not proof a worker executed.",
        })

    success = status not in ("unverified", "not_queued", "failed")
    final_output = {
        "agent_id": AGENT_ID,
        "action": action,
        "tool": tool_name,
        "status": status,
        "applied": applied,
        "executed": executed,
        "completed": completed,
        "approval": dict(approval),
        "delivery": delivery,
        "objective": "",
        **result,
    }

    # Every item is a concrete, auditable tool-call anchor. The last anchor is
    # a projection of the reported output onto the primary audited call: its
    # digest binds the whole result while its id/arguments still point at the
    # real successful ToolResult. Raw tool output stays visible alongside it.
    evidence_items = [_anchor_item(anchor) for anchor in anchors]
    primary = dict(anchors[0])
    evidence_items.append(_anchor_item({
        **primary,
        "output": final_output,
        "digest": evidence_digest(final_output),
        "projection": True,
        "raw_output": primary.get("output"),
        "approval": dict(approval),
    }))
    return {
        "success": success,
        "agent_id": AGENT_ID,
        "output": final_output,
        "evidence": evidence_items,
        "evaluations": [{
            "kind": "automation",
            "action": action,
            "tool": tool_name,
            "status": status,
            "executed": executed,
            "completed": completed,
        }],
    }


def make_handler(services: Any = None) -> Callable[..., Any]:
    """Build the automation operator's async worker.

    ``services`` is the trusted injection mapping: it supplies ``scheduler``,
    ``tasks``, ``webhooks``, ``events``, ``settings`` and ``permissions`` to the
    real tools, and is the *only* source of the approval gate's permission
    manager and settings. Request context is never consulted for services or
    for the actor. The scoped ``tools`` facade injected by ``AgentEngine`` is
    preferred over any raw executor in ``services``, so the handler cannot
    escape its declared tool grants.
    """

    injected = services if isinstance(services, Mapping) else {}

    async def handler(
        objective: Any = "",
        context: Any = None,
        tools: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ctx = _mapping(context)
        request = _parse_request(objective, ctx)
        action = _normalize_action(request.get("action"))
        if action not in _ACTIONS:
            raise ValueError(
                "automation action must be one of: " + ", ".join(sorted(_ACTIONS))
            )
        tool_name, mutating = _ACTIONS[action]
        # The engine passes its scoped façade as `tools`; never accept a tool
        # object from caller-influenced context.
        tool_view = tools or _service(injected, "tools", "executor")
        if tool_view is None or not hasattr(tool_view, "execute"):
            raise RuntimeError("automation operator requires a scoped tool executor")
        arguments = _tool_arguments(action, request)

        approval: dict[str, Any] = {
            "required": bool(mutating), "approved": False, "granted": False,
            "approval_id": None, "reason": "no approval required for read-only action",
        }
        if mutating:
            verified, approval = await _verify_approval(
                request, ctx, injected, tool_name, arguments
            )
            if not verified:
                return _blocked(action, tool_name, approval)

        try:
            output, primary_anchor = await _call(tool_view, tool_name, arguments, ctx)
        except Exception as exc:
            return _failed(action, tool_name, f"{type(exc).__name__}: {exc}", approval)

        try:
            return await _shape(
                action, tool_name, arguments, output, approval,
                [primary_anchor], tool_view, ctx,
            )
        except Exception as exc:
            return _failed(action, tool_name, f"read-back failed: {type(exc).__name__}: {exc}", approval)

    return handler


__all__ = [
    "SEED", "TOOL_GRANTS", "AGENT_ID", "APPROVAL_ACTION", "APPROVAL_RISK",
    "approval_target", "tool_arguments", "make_handler",
]
