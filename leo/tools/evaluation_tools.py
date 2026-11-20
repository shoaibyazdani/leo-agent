"""Evaluation tools backed by real ``Evaluation`` APIs and durable run rows.

``evaluate_run`` scores one persisted ``agent_runs`` row from its recorded
evaluations, records the score through ``Evaluation.record`` and publishes an
event. ``evaluation_proposals`` lists real proposals from ``kv`` or creates one
from measured reports/task metrics/tool reliability/regressions via
``Evaluation.propose``. ``apply_improvement`` adopts a proposal only through
``Evaluation.adopt`` after the *existing* PermissionManager high-risk approval
gate has been cleared by an operator-resolved approval bound to the exact tool
arguments (the gateway binds a canonical arguments digest into the approval).
The approval id travels with the executor call, never as a tool argument, so
request and retry carry identical arguments and the binding stays intact.

The tool handler never resolves an approval, never supplies its own approver and
never marks tests as passed on its own behalf. Missing, rebound, self-resolved
or untested adoption is reported as a blocked/denied result, not a success.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import db
from ..contracts import RiskLevel, ToolSpec, utcnow
from ..security.permissions import arguments_digest

EVALUATE_RUN = "evaluate_run"
EVALUATION_PROPOSALS = "evaluation_proposals"
APPLY_IMPROVEMENT = "apply_improvement"
TOOL_NAMES = (EVALUATE_RUN, EVALUATION_PROPOSALS, APPLY_IMPROVEMENT)

CATEGORY = "evaluation"
# Must match the action the executor passes to PermissionManager.authorize.
_TOOL_ACTION = "tool:" + APPLY_IMPROVEMENT
_SCORE_METRIC = "run.score"
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 100


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _specs() -> list[tuple[ToolSpec, Any]]:
    """Build fresh specs on every registration (no shared mutable schemas)."""
    return [
        (
            ToolSpec(
                name=EVALUATE_RUN,
                description=(
                    "Score one durable agent run from its recorded verification and execution "
                    "evaluations, persist the score as a real evaluation metric and publish it."
                ),
                category=CATEGORY,
                risk=RiskLevel.LOW,
                permissions=["evaluation:read", "metrics:write"],
                resource_field="",  # run ids are identifiers, not filesystem paths
                input_schema=_schema({"run_id": {"type": "string", "minLength": 1}}, ["run_id"]),
            ),
            _evaluate_run,
        ),
        (
            ToolSpec(
                name=EVALUATION_PROPOSALS,
                description=(
                    "List real improvement proposals, or create one from measured evaluation "
                    "reports, task metrics, tool reliability and detected regressions."
                ),
                category=CATEGORY,
                risk=RiskLevel.LOW,
                permissions=["evaluation:read", "evaluation:write"],
                resource_field="",
                input_schema=_schema({
                    "action": {"type": "string", "enum": ["list", "propose"]},
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "metric": {"type": "string", "minLength": 1, "maxLength": 200},
                    "window": {"type": "integer", "minimum": 1, "maximum": 100},
                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT},
                }),
            ),
            _evaluation_proposals,
        ),
        (
            ToolSpec(
                name=APPLY_IMPROVEMENT,
                description=(
                    "Adopt a measured improvement proposal through Evaluation.adopt. HIGH risk: "
                    "the executor requires an operator-resolved approval bound to these exact "
                    "arguments (pass its id to the executor, not in this tool's arguments). "
                    "Missing, rebound, self-resolved or untested adoption is blocked."
                ),
                category=CATEGORY,
                risk=RiskLevel.HIGH,
                permissions=["evaluation:adopt", "approval:use"],
                resource_field="",
                input_schema=_schema({
                    "proposal_id": {"type": "string", "minLength": 1},
                    "tests_passed": {"type": "boolean"},
                }, ["proposal_id", "tests_passed"]),
            ),
            _apply_improvement,
        ),
    ]


def _services(ctx: dict[str, Any]) -> dict[str, Any]:
    value = ctx.get("services")
    return dict(value) if isinstance(value, dict) else {}


def _evaluation(ctx: dict[str, Any]) -> Any:
    services = _services(ctx)
    service = services.get("evaluation") or services.get("evaluations")
    if service is None:
        raise RuntimeError(
            "evaluation tools require an Evaluation service at ctx['services']['evaluation']"
        )
    return service


def _events(ctx: dict[str, Any]) -> Any:
    bus = _services(ctx).get("events") or ctx.get("events")
    if bus is None or not hasattr(bus, "emit"):
        raise RuntimeError(
            "evaluation tools require an events service at ctx['services']['events'] "
            "or ctx['events'] to publish evaluation results"
        )
    return bus


def _permissions(ctx: dict[str, Any]) -> Any:
    manager = _services(ctx).get("permissions") or ctx.get("permissions")
    if manager is None:
        raise RuntimeError(
            "apply_improvement requires a PermissionManager at "
            "ctx['services']['permissions'] or ctx['permissions']"
        )
    return manager


def _settings(ctx: dict[str, Any]) -> Any:
    settings = _services(ctx).get("settings") or ctx.get("settings")
    if settings is None:
        raise RuntimeError("apply_improvement requires Settings in the tool context")
    return settings


def _bounded(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return max(0.0, min(1.0, number))


def _expired(row: Any, settings: Any) -> bool:
    """Mirror the PermissionManager approval expiry window (read-only check)."""
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
    except (TypeError, ValueError, KeyError):
        return True
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    timeout = getattr(getattr(settings, "security", None), "approval_timeout_seconds", 900)
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = 900
    return (datetime.now(timezone.utc) - created) > timedelta(seconds=timeout)


async def _evaluate_run(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation(ctx)
    bus = _events(ctx)
    run_id = str(args["run_id"]).strip()
    row = await db.fetchone(
        "SELECT id,agent_id,task_id,status,started_at,finished_at,evaluations "
        "FROM agent_runs WHERE id=?",
        (run_id,),
    )
    if row is None:
        raise ValueError(f"unknown agent run: {run_id}")
    try:
        evaluations = json.loads(row["evaluations"] or "[]")
    except (TypeError, ValueError):
        evaluations = []
    if not isinstance(evaluations, list):
        evaluations = []

    verification_scores = [
        score
        for item in evaluations
        if isinstance(item, dict) and item.get("kind") != "execution"
        for score in [_finite_score(item.get("score"))]
        if score is not None
    ]
    execution_items = [
        item for item in evaluations
        if isinstance(item, dict) and item.get("kind") == "execution"
    ]
    components: dict[str, float] = {}
    if verification_scores:
        components["verification"] = round(
            sum(verification_scores) / len(verification_scores), 6
        )
    if execution_items:
        components["execution"] = 1.0 if any(
            item.get("success") is True for item in execution_items
        ) else 0.0
    # Runs without stored evaluations are scored on their terminal state only;
    # this is a proxy, and is reported as such in the components.
    components["status"] = 1.0 if row["status"] == "completed" else 0.0
    score = round(sum(components.values()) / len(components), 4)

    labels = {"run_id": row["id"], "agent_id": row["agent_id"]}
    await evaluation.record(_SCORE_METRIC, score, labels)
    await bus.emit(
        "evaluation.run_scored",
        run_id=row["id"],
        agent_id=row["agent_id"],
        task_id=row["task_id"],
        status=row["status"],
        score=score,
        components=components,
        evaluation_count=len(evaluations),
        correlation_id=ctx.get("correlation_id"),
        created_at=utcnow(),
    )
    return {
        "success": True,
        "run_id": row["id"],
        "agent_id": row["agent_id"],
        "task_id": row["task_id"],
        "status": row["status"],
        "score": score,
        "components": components,
        "evaluation_count": len(evaluations),
        "metric": f"{evaluation.namespace}.{_SCORE_METRIC}" if hasattr(evaluation, "namespace") else _SCORE_METRIC,
    }


async def _evaluation_proposals(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation(ctx)
    bus = _events(ctx)
    action = args.get("action") or "list"

    if action == "list":
        proposals = await evaluation.proposals()
        limit = _bounded(args.get("limit", _DEFAULT_LIMIT), _DEFAULT_LIMIT, 1, _MAX_LIMIT)
        page = proposals[:limit]
        return {
            "success": True,
            "action": "list",
            "count": len(page),
            "total": len(proposals),
            "limit": limit,
            "proposals": page,
        }

    title = str(args.get("title") or "").strip()
    rationale = str(args.get("rationale") or "").strip()
    if not title or not rationale:
        raise ValueError("action='propose' requires a non-empty title and rationale")

    evidence: dict[str, Any] = {
        "report": await evaluation.report(),
        "tasks": await evaluation.task_metrics(),
        "tool_reliability": await evaluation.tool_reliability(),
    }
    metric = str(args.get("metric") or "").strip()
    if metric:
        window = _bounded(args.get("window", 10), 10, 1, 100)
        threshold = args.get("threshold", 0.2)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            threshold = 0.2
        evidence["regressions"] = await evaluation.detect_regressions(
            metric, window=window, threshold=float(threshold)
        )
        evidence["metric"] = metric
        evidence["window"] = window
        evidence["threshold"] = float(threshold)

    proposal = await evaluation.propose(title, rationale, evidence)
    await bus.emit(
        "evaluation.proposal_created",
        proposal_id=proposal["id"],
        title=title,
        requires_approval=True,
        correlation_id=ctx.get("correlation_id"),
        created_at=utcnow(),
    )
    return {
        "success": True,
        "action": "propose",
        "proposal": proposal,
        "total": len(await evaluation.proposals()),
    }


async def _bound_approval(actor: str, digest: str) -> Any | None:
    """Find an approved, argument-bound HIGH-risk approval for this actor/action."""
    rows = await db.fetchall(
        "SELECT * FROM approvals WHERE actor=? AND action=? AND risk=? AND status='approved' "
        "ORDER BY created_at DESC, rowid DESC LIMIT 50",
        (actor, _TOOL_ACTION, RiskLevel.HIGH.value),
    )
    for row in rows:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(metadata, dict) and metadata.get("arguments_digest") == digest:
            return row
    return None


async def _apply_improvement(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation(ctx)
    permissions = _permissions(ctx)
    settings = _settings(ctx)
    bus = _events(ctx)
    proposal_id = str(args.get("proposal_id") or "").strip()
    if not proposal_id:
        raise ValueError("proposal_id is required")
    actor = str(ctx.get("actor") or "").strip() or "assistant"
    tests_passed = args.get("tests_passed") is True
    digest = arguments_digest(dict(args))

    if digest is None:
        raise PermissionError(
            "improvement adoption blocked: tool arguments cannot be bound to an approval"
        )
    if not tests_passed:
        raise PermissionError(
            "improvement adoption blocked: passing tests were not reported for this adoption"
        )
    # The approval must have been requested for these exact arguments. Looking
    # it up by its durable digest prevents reusing a decision for another
    # proposal or another test claim, even if an approval id is replayed.
    row = await _bound_approval(actor, digest)
    if row is None:
        raise PermissionError(
            "improvement adoption blocked: no operator-resolved approval bound to these arguments"
        )
    approval_id = row["id"]
    if _expired(row, settings):
        raise PermissionError("improvement adoption blocked: approval has expired")
    try:
        metadata = json.loads(row["metadata"] or "{}")
    except (TypeError, ValueError):
        metadata = {}
    resolver = str(metadata.get("resolved_by") or "").strip() if isinstance(metadata, dict) else ""
    if not resolver:
        raise PermissionError(
            "improvement adoption blocked: no recorded operator resolver on the approval"
        )
    if resolver.casefold().startswith("chat:") or resolver.casefold() == actor.casefold():
        raise PermissionError(
            "improvement adoption blocked: self-approval is not allowed "
            "(an approval must be resolved by an operator, not the requesting actor)"
        )

    # Re-check the same gate the executor uses, including the argument digest.
    # With a bad id this creates no new approval; with a good id it records the
    # exact approval that authorized the adoption in the audit log.
    decision = await permissions.authorize(
        actor, _TOOL_ACTION, None, RiskLevel.HIGH,
        approval_id=approval_id,
        task_id=ctx.get("task_id"),
        tool_call_id=ctx.get("call_id"),
        arguments_digest=digest,
    )
    if not decision.allowed:
        raise PermissionError(f"improvement adoption blocked: {decision.reason}")

    proposal = await evaluation.adopt(proposal_id, True, resolver)
    await bus.emit(
        "evaluation.improvement_applied",
        proposal_id=proposal_id,
        approval_id=approval_id,
        approver=resolver,
        requested_by=actor,
        correlation_id=ctx.get("correlation_id"),
        created_at=utcnow(),
    )
    return {
        "success": True,
        "status": "adopted",
        "proposal": proposal,
        "proposal_id": proposal_id,
        "approval_id": approval_id,
        "approver": resolver,
        "tests_passed_reported": True,
    }


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the evaluation toolset and return the names actually registered.

    ``services`` is an optional fallback injection source; per-call
    ``ctx['services']`` values always take precedence.
    """
    injected = dict(services) if isinstance(services, dict) else {}
    registered: list[str] = []
    for spec, handler in _specs():
        async def bound(args: dict[str, Any], ctx: dict[str, Any], _handler=handler) -> Any:
            merged = dict(ctx)
            merged["services"] = {**injected, **dict(ctx.get("services") or {})}
            return await _handler(args, merged)

        try:
            registry.register(spec, bound)
        except ValueError:  # already registered - preserve the caller's version
            continue
        registered.append(spec.name)
    return registered


__all__ = [
    "register", "TOOL_NAMES", "EVALUATE_RUN", "EVALUATION_PROPOSALS", "APPLY_IMPROVEMENT",
]
