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
from ...contracts import LIVE_AUTH_CLAIM_TTL_SECONDS, RiskLevel, TaskStatus, ToolCall
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
    "schedule_get",
    "schedule_update",
    "schedule_pause",
    "schedule_resume",
    "schedule_delete",
    "schedule_run_now",
    "schedule_cancel",
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

# action -> (tool used, needs the specialist's durable automation:mutate gate)
#
# create/update/pause/resume are deliberately ungated here: they converge on
# the same canonical ScheduleService proposal/edit card as a direct model call.
# Doubling the gate would require a human to approve an automation:mutate row
# *and* then the schedule card. Delete/trigger/cancel/enqueue keep the exact
# existing gate because they are lifecycle/execution mutations with no schedule
# card of their own, and the declared grant must match the enforced policy.
_ACTIONS: dict[str, tuple[str, bool]] = {
    "list_schedules": ("schedule_list", False),
    "get_schedule": ("schedule_get", False),
    "create_schedule": ("schedule_create", False),
    "update_schedule": ("schedule_update", False),
    "pause_schedule": ("schedule_pause", False),
    "resume_schedule": ("schedule_resume", False),
    "delete_schedule": ("schedule_delete", True),
    "trigger_schedule": ("schedule_run_now", True),
    "cancel_schedule": ("schedule_cancel", True),
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
    "get": "get_schedule",
    "schedule_get": "get_schedule",
    "create": "create_schedule",
    "schedule_create": "create_schedule",
    "update": "update_schedule",
    "schedule_update": "update_schedule",
    "pause": "pause_schedule",
    "schedule_pause": "pause_schedule",
    "resume": "resume_schedule",
    "schedule_resume": "resume_schedule",
    "delete": "delete_schedule",
    "remove": "delete_schedule",
    "schedule_delete": "delete_schedule",
    "trigger": "trigger_schedule",
    "run_now": "trigger_schedule",
    "schedule_run_now": "trigger_schedule",
    "cancel": "cancel_schedule",
    "schedule_cancel": "cancel_schedule",
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
    "action", "name", "schedule", "expression", "prompt", "objective",
    "task_objective", "payload", "repeat", "deliver", "failure_deliver",
    "skills", "skill", "catch_up", "wrap_response", "script", "monitor",
    "no_agent", "context_from", "continuity", "enabled_toolsets", "workdir",
    "timezone", "attach_to_session", "pinned", "paused", "paused_reason",
    "schedule_id", "job_id", "config_revision", "request_key",
    "enabled_only", "include_paused", "run_at", "headers", "body",
    "delivery_id", "limit", "offset", "channel", "approval_id",
    "idempotency_key",
)
# ``task_id`` is deliberately absent above: the engine context carries the
# specialist's OWN task id, which must never be mistaken for an automation
# target. A real target task_id arrives in the objective JSON instead.
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


_SCHEDULE_ARG_KEYS: tuple[str, ...] = (
    "schedule", "prompt", "name", "repeat", "deliver", "failure_deliver",
    "skills", "skill", "catch_up", "wrap_response", "script", "monitor",
    "no_agent", "context_from", "continuity", "enabled_toolsets", "workdir",
    "timezone", "attach_to_session", "pinned", "paused", "paused_reason",
    "config_revision", "request_key",
)


def _target_args(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("schedule_id") is not None:
        return {"schedule_id": str(request["schedule_id"])}
    if request.get("job_id") is not None:
        return {"schedule_id": str(request["job_id"])}
    return {"name": str(_required(request, "name"))}


def _tool_arguments(action: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Build exact ToolSpec arguments; unknown request keys are never forwarded."""
    if action == "list_schedules":
        args = {"enabled_only": bool(request.get("enabled_only", False))}
        if request.get("include_paused") is not None:
            args["include_paused"] = bool(request["include_paused"])
        return args
    if action == "get_schedule":
        return _target_args(request)
    if action in ("create_schedule", "update_schedule"):
        schedule = request.get("schedule", request.get("expression"))
        prompt = request.get("prompt", request.get("objective"))
        if action == "create_schedule":
            if schedule is None or not str(schedule).strip():
                raise ValueError("requires 'schedule'")
            if prompt is None or not str(prompt).strip():
                raise ValueError("requires 'prompt'")
        args: dict[str, Any] = {}
        for key in _SCHEDULE_ARG_KEYS:
            if key in request and request[key] is not None:
                args[key] = request[key]
        if schedule is not None:
            args["schedule"] = schedule
        if prompt is not None:
            args["prompt"] = prompt
        if action == "update_schedule":
            # The specialist identifies the job by id/name; it does not use the
            # field as an implicit rename. Renaming stays a direct card edit.
            args.pop("name", None)
            args.update(_target_args(request))
        return args
    if action in ("pause_schedule", "resume_schedule"):
        args = _target_args(request)
        for key in ("config_revision", "request_key"):
            if request.get(key) is not None:
                args[key] = request[key]
        if action == "pause_schedule" and request.get("paused_reason") is not None:
            args["paused_reason"] = request["paused_reason"]
        return args
    if action == "delete_schedule":
        return _target_args(request)
    if action == "trigger_schedule":
        args = _target_args(request)
        if request.get("run_at") is not None:
            args["run_at"] = request["run_at"]
        if request.get("prompt") is not None:
            args["prompt"] = request["prompt"]
        if request.get("request_key") is not None:
            args["request_key"] = request["request_key"]
        return args
    if action == "cancel_schedule":
        args = _target_args(request)
        if request.get("task_id") is not None:
            args["task_id"] = request["task_id"]
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
    tool_view: Any,
    name: str,
    arguments: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    authorization: str | None = None,
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
        authorization=authorization,
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

    metadata: Mapping[str, Any] = {}
    try:
        decoded = json.loads(row["metadata"] or "{}")
    except (TypeError, ValueError):
        decoded = {}
    if isinstance(decoded, Mapping):
        metadata = decoded
    # The schedule service settles a live run/cancel lease on this same row.
    # Refuse a card that is already consumed or currently leased, so a second
    # specialist call cannot authorize another mutation with the same approval.
    if metadata.get("live_authorization_used_at"):
        return deny(f"approval {approval_id} was already used for a live action")
    lease = metadata.get("live_authorization_claim")
    if isinstance(lease, Mapping) and lease.get("token"):
        # A lease left by a crashed worker is reclaimable after the shared
        # bound; an active or unreadable lease stays fail-closed.
        leased_at: datetime | None = None
        try:
            leased_at = datetime.fromisoformat(str(lease.get("at") or ""))
            if leased_at.tzinfo is None:
                leased_at = leased_at.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            leased_at = None
        stale = (
            leased_at is not None
            and (datetime.now(timezone.utc) - leased_at).total_seconds()
            >= LIVE_AUTH_CLAIM_TTL_SECONDS
        )
        if not stale:
            return deny(f"approval {approval_id} is already in use for a live action")

    for field in ("action", "resource", "risk", "actor"):
        expected = {
            "action": target["action"],
            "resource": target["resource"],
            "risk": target["risk"],
            "actor": AGENT_ID,
        }[field]
        if str(row[field]) != str(expected):
            return deny(f"approval binding mismatch on {field}")

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
    now = datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        expired = (now - created).total_seconds() > timeout
        # A card's own deadline can be shorter than the configured age limit.
        if "expires_at" in row.keys() and str(row["expires_at"] or "").strip():
            deadline = datetime.fromisoformat(str(row["expires_at"]))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            expired = expired or now >= deadline
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

    if action in (
        "create_schedule", "update_schedule", "delete_schedule", "get_schedule",
        "pause_schedule", "resume_schedule",
    ):
        # The canonical ScheduleService owns consent: a model create/update may
        # be waiting on a card and must never be shaped as applied. Only durable
        # read-back after the service reports success can do that.
        reported = str(data.get("status") or "")
        if action == "get_schedule":
            result["schedule"] = data.get("schedule")
            result["verified_by"] = "schedule_get"
            status = "observed" if data.get("schedule") else "unverified"
            delivery["outcome"] = status
        elif reported == "approval_required":
            status = "approval_required"
            applied = False
            result.update({
                "approval_id": data.get("approval_id"),
                "proposal_id": data.get("proposal_id"),
                "schedule_id": data.get("schedule_id"),
                "name": data.get("name"),
                "card": data.get("card"),
                "verified_by": "schedule_service",
            })
            delivery.update({
                "outcome": status,
                "acknowledged": bool(data.get("approval_id") or data.get("proposal_id")),
                "note": "A card or durable proposal is waiting; nothing was saved.",
            })
        elif action == "update_schedule" and data.get("status") == "no_change":
            status = "no_change"
            result["verified_by"] = "schedule_service"
            delivery["outcome"] = status
        else:
            target_name = data.get("name") or arguments.get("name")
            schedules, list_anchor = await _call(
                tool_view, "schedule_list", {"enabled_only": False}, context
            )
            anchors.append(list_anchor)
            rows = [
                s for s in (
                    _mapping(item)
                    for item in _mapping(schedules).get("schedules") or []
                )
                if s.get("name") == target_name
            ]
            if action == "delete_schedule":
                applied = not rows and reported in ("removed", "deleted")
            elif action in ("pause_schedule", "resume_schedule"):
                # Pause/resume must be proven by the durable enabled state, not
                # by the tool's own wording.
                expected_enabled = action == "resume_schedule"
                applied = bool(rows) and bool(rows[0].get("enabled")) is expected_enabled
            else:
                applied = bool(rows) and reported in ("created", "updated")
            status = "applied" if applied else "unverified"
            result["schedule"] = rows[0] if rows else None
            result["verified_by"] = "schedule_list"
            result["schedule_readback"] = {
                "name": target_name, "present": bool(rows)
            }
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
            "schedule": data.get("schedule") or data.get("name"),
            "task_id": task_id,
            "task_status": task_status,
            "triggered": bool(data.get("triggered", True)),
            "manual_run": bool(data.get("manual_run", action == "trigger_schedule")),
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

    elif action == "cancel_schedule":
        reported = str(data.get("status") or "")
        cancelled = bool(data.get("cancelled"))
        status = "cancelled" if cancelled else "no_live_run"
        result.update({
            "schedule_id": data.get("schedule_id"),
            "name": data.get("name"),
            "cancelled": cancelled,
            "tasks": list(data.get("tasks") or []),
            "schedule_deleted": bool(data.get("schedule_deleted")),
            "verified_by": "schedule_cancel",
        })
        delivery.update({
            "outcome": status,
            "executed": False,
            "completed": False,
            "note": (
                "Live tasks were stopped; the schedule itself was not deleted."
                if cancelled
                else "No live task was in flight."
            ),
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

    success = status not in (
        "unverified", "not_queued", "failed", "approval_required"
    )
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
            "approval_id": None,
            "actor": AGENT_ID,
            "context_actor_ignored": bool(
                ctx.get("actor") and str(ctx.get("actor")) != AGENT_ID
            ),
            "reason": (
                "canonical schedule consent"
                if action in (
                    "create_schedule", "update_schedule", "pause_schedule",
                    "resume_schedule",
                )
                else "no approval required for read-only action"
            ),
        }
        if mutating:
            verified, approval = await _verify_approval(
                request, ctx, injected, tool_name, arguments
            )
            if not verified:
                return _blocked(action, tool_name, approval)

        try:
            output, primary_anchor = await _call(
                tool_view,
                tool_name,
                arguments,
                ctx,
                authorization=(
                    approval.get("approval_id") if mutating else None
                ),
            )
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
