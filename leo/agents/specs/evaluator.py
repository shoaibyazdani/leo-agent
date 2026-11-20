"""Evaluator specialist: evidence-backed measurement of completed runs.

This spec uses a bespoke handler (``make_handler``), not the default delegation
worker. The handler reads durable ``agent_runs`` rows and records routing,
retrieval, verification and cost scores through the real
:class:`leo.core.evaluation.Evaluation` API. Every score carries the rows it
was derived from. The evaluator is strictly a measurer: it never approves,
adopts or mutates the work it scores. Adoption remains a human-gated core
operation (``Evaluation.adopt``), and this module never calls it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from ... import db
from ...core.evaluation import Evaluation

_AGENT_ID = "evaluator"
_METRIC_NAMESPACE = "evaluator"
_DIMENSIONS: tuple[str, ...] = ("routing", "retrieval", "verification", "cost")

# Every grant is a *tool name* verified against a live ToolSpec category:
# event_tools (observability), approval_tools (audit), agent_tools (agents),
# task_tools (tasks), model_tools (models). Categories are deliberately not
# granted: a category would also admit delegate/cancel/write tools.
TOOL_GRANTS: tuple[str, ...] = (
    "event_query",
    "audit_query",
    "audit_tail",
    "agent_status",
    "agent_result",
    "task_status",
    "task_result",
    "model_usage",
)

SEED: dict[str, Any] = {
    "id": _AGENT_ID,
    "name": "Evaluator",
    "purpose": (
        "Measure completed run quality across routing, retrieval, verification and "
        "cost from durable evidence, and emit evidence-backed scores without "
        "approving or changing the work under review."
    ),
    "capabilities": [
        "run_quality_measurement",
        "evidence_scoring",
        "metric_emission",
        "regression_observation",
    ],
    "tool_permissions": list(TOOL_GRANTS),
    "model_preferences": ["reasoning", "structured", "tools"],
    "context_budget": 16000,
    "task_limit": 2,
    "metadata": {
        "tier": "assurance",
        "risk": "low",
        "read_only": True,
        "never_self_approve": True,
        "approval_authority": "human_only",
        "handler_kind": "bespoke:make_handler",
        "metric_namespace": _METRIC_NAMESPACE,
        "measured_dimensions": list(_DIMENSIONS),
    },
}


def _service(services: Any, *names: str) -> Any:
    """Resolve an injected service from a mapping or an attribute-style container."""
    for name in names:
        if isinstance(services, Mapping):
            value = services.get(name)
        else:
            value = getattr(services, name, None)
        if value is not None:
            return value
    return None


def _decoded(raw: Any, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str) or not raw:
        return default
    import json

    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN must never reach the metrics table
        return 0.0
    return max(0.0, min(1.0, number))


async def _load_run(
    run_id: Any, context: Mapping[str, Any], dependencies: Sequence[Any]
) -> tuple[Any | None, str]:
    """Resolve the run under review from durable state, with an explicit source."""
    if run_id:
        row = await db.fetchone("SELECT * FROM agent_runs WHERE id=?", (str(run_id),))
        if row is not None:
            return row, "run_id"
    references: list[str] = []
    for key in ("target_task_id", "evaluated_task_id", "reviewed_task_id"):
        if context.get(key):
            references.append(str(context[key]))
    for item in dependencies or []:
        if isinstance(item, Mapping) and item.get("task_id"):
            references.append(str(item["task_id"]))
        elif isinstance(item, str) and item.startswith("task_"):
            references.append(item)
    for reference in references:
        row = await db.fetchone(
            "SELECT * FROM agent_runs WHERE task_id=? ORDER BY rowid DESC LIMIT 1", (reference,)
        )
        if row is not None:
            return row, "task_id"
    if not run_id and not references:
        row = await db.fetchone(
            "SELECT * FROM agent_runs WHERE agent_id<>? ORDER BY rowid DESC LIMIT 1", (_AGENT_ID,)
        )
        if row is not None:
            return row, "latest_run"
    return None, "none"


def _score_routing(row: Any, context: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    actual = str(row["agent_id"] or "")
    expected = context.get("expected_agent") or context.get("expected_agent_id") or metadata.get("expected_agent")
    measured = bool(expected)
    score = 1.0 if str(expected) == actual else 0.0 if measured else 0.5
    return {
        "score": score,
        "measured": measured,
        "evidence": [
            {"source": "agent_runs.agent_id", "claim": f"run executed by {actual}", "value": actual},
            {
                "source": "context.expected_agent" if context.get("expected_agent") else "agent_runs.metadata",
                "claim": f"expected agent {expected or 'not recorded'}",
                "value": expected,
            },
        ],
    }


def _evidence_items(result: Any) -> list[Any]:
    if not isinstance(result, Mapping):
        return []
    direct = result.get("evidence")
    if isinstance(direct, list):
        return direct
    output = result.get("output")
    if isinstance(output, Mapping) and isinstance(output.get("evidence"), list):
        return output["evidence"]
    return []


def _score_retrieval(result: Any) -> dict[str, Any]:
    items = _evidence_items(result)
    verified = sum(
        1 for item in items if isinstance(item, Mapping) and item.get("verified") is True
    )
    sources = sorted({
        str(item.get("source") or "unattributed")
        for item in items if isinstance(item, Mapping)
    })
    score = verified / len(items) if items else 0.0
    return {
        "score": _clamp01(score),
        "measured": True,
        "evidence": [
            {
                "source": "agent_runs.result.evidence",
                "claim": f"{verified}/{len(items)} recorded evidence items are verified",
                "total": len(items),
                "verified": verified,
                "sources": sources,
            }
        ],
    }


def _score_verification(evaluations: Sequence[Any], status: str) -> dict[str, Any]:
    verification = next(
        (item for item in evaluations if isinstance(item, Mapping) and item.get("kind") == "verification"),
        None,
    )
    if verification is not None:
        raw_score = verification.get("score")
        score = (
            _clamp01(raw_score) if raw_score is not None
            else (1.0 if verification.get("verified") is True else 0.0)
        )
        return {
            "score": score,
            "measured": True,
            "evidence": [
                {
                    "source": "agent_runs.evaluations[kind=verification]",
                    "claim": f"verification record reported score {raw_score!r}, verified={verification.get('verified')!r}",
                    "value": dict(verification),
                }
            ],
        }
    return {
        "score": 0.0,
        "measured": False,
        "evidence": [
            {
                "source": "agent_runs.evaluations",
                "claim": f"no verification evaluation recorded; run status was {status}",
                "status": status,
            }
        ],
    }


def _duration_ms(row: Any, evaluations: Sequence[Any]) -> float | None:
    execution = next(
        (item for item in evaluations if isinstance(item, Mapping) and item.get("kind") == "execution"),
        None,
    )
    if execution is not None and execution.get("duration_ms") is not None:
        try:
            return max(0.0, float(execution["duration_ms"]))
        except (TypeError, ValueError):
            pass
    try:
        start = datetime.fromisoformat(str(row["started_at"]))
        end = datetime.fromisoformat(str(row["finished_at"]))
    except (TypeError, ValueError):
        return None
    return max(0.0, (end - start).total_seconds() * 1000)


def _score_cost(
    row: Any, result: Any, metadata: Mapping[str, Any], evaluations: Sequence[Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    duration = _duration_ms(row, evaluations)
    usage = metadata.get("usage") or (result.get("usage") if isinstance(result, Mapping) else None)
    tokens = usage.get("total_tokens") if isinstance(usage, Mapping) else None
    reported_cost = usage.get("cost") if isinstance(usage, Mapping) else None
    try:
        budget_ms = max(1.0, float(context.get("cost_budget_ms") or 60000))
        token_budget = max(1.0, float(context.get("token_budget") or 20000))
        duration_value = float(duration) if duration is not None else None
        token_value = float(tokens) if tokens is not None else None
    except (TypeError, ValueError):
        budget_ms, token_budget = 60000.0, 20000.0
        duration_value, token_value = None, None
    scores: list[float] = []
    if duration_value is not None:
        scores.append(1.0 - min(1.0, duration_value / budget_ms))
    if token_value is not None:
        scores.append(1.0 - min(1.0, token_value / token_budget))
    score = min(scores) if scores else 0.0
    return {
        "score": _clamp01(score),
        "measured": bool(scores),
        "evidence": [
            {
                "source": "agent_runs.evaluations[kind=execution]/timestamps",
                "claim": f"measured duration {duration} ms against budget {budget_ms} ms",
                "duration_ms": duration,
                "budget_ms": budget_ms,
            },
            {
                "source": "agent_runs.result/metadata.usage",
                "claim": f"reported tokens {tokens!r}, cost {reported_cost!r} against budget {token_budget}",
                "tokens": tokens,
                "cost": reported_cost,
                "token_budget": token_budget,
            },
        ],
    }


def make_handler(services: Any = None) -> Callable[..., Any]:
    """Build the evaluator's async handler.

    ``services['evaluation']`` may be a real :class:`Evaluation` instance; when
    absent the handler constructs one over the same durable database, so it is
    never inert. ``services['events']`` (an EventBus) is optional and enables
    the ``evaluation.scored`` event. Target runs are resolved from context,
    dependencies or the latest non-evaluator run.
    """

    async def handler(
        objective: Any = "",
        context: dict[str, Any] | None = None,
        *,
        agent_id: str | None = None,
        dependencies: Sequence[Any] = (),
        **kwargs: Any,
    ) -> dict[str, Any]:
        ctx = dict(context or {})
        evaluation = _service(services, "evaluation", "evaluations")
        if not isinstance(evaluation, Evaluation):
            evaluation = Evaluation(namespace=_METRIC_NAMESPACE)
        bus = _service(services, "events", "event_bus")

        run, source = await _load_run(
            ctx.get("run_id") or ctx.get("target_run_id") or ctx.get("evaluated_run_id"),
            ctx,
            dependencies,
        )
        if run is None:
            return {
                "success": False,
                "error": "evaluator needs a run_id, target_task_id or a prior run in dependencies",
                "output": {"status": "blocked", "run_source": source},
                "evidence": [{"source": "agent_runs", "claim": "no target run found", "verified": False}],
            }

        result = _decoded(run["result"], {})
        metadata = _decoded(run["metadata"], {}) or {}
        evaluations = _decoded(run["evaluations"], []) or []
        run_id = str(run["id"])

        scores = {
            "routing": _score_routing(run, ctx, metadata),
            "retrieval": _score_retrieval(result),
            "verification": _score_verification(evaluations, str(run["status"])),
            "cost": _score_cost(run, result, metadata, evaluations, ctx),
        }
        total = sum(item["score"] for item in scores.values()) / len(scores)
        labels = {
            "run_id": run_id,
            "target_agent": str(run["agent_id"]),
            "evaluator": _AGENT_ID,
            "correlation_id": str(ctx.get("correlation_id") or ""),
        }
        for dimension, data in scores.items():
            await evaluation.record(
                dimension,
                data["score"],
                {**labels, "measured": bool(data["measured"])},
            )
        await evaluation.record("score_total", total, labels)
        report = await evaluation.report()
        regressions = await evaluation.detect_regressions("score_total", window=1)

        evidence = [
            {"source": f"agent_runs:{run_id}", "claim": f"run resolved via {source}", "verified": True}
        ]
        for dimension, data in scores.items():
            evidence.extend(
                {"dimension": dimension, **item} for item in data["evidence"]
            )

        approval = {
            "required": True,
            "granted": False,
            "approver": None,
            "reason": "evaluator emits measurements only; approval/adoption requires a human",
        }
        output = {
            "status": "measured",
            "run_id": run_id,
            "run_source": source,
            "target_agent_id": str(run["agent_id"]),
            "scores": scores,
            "total_score": total,
            "metric_namespace": _METRIC_NAMESPACE,
            "persisted_report": {
                name: report[name]
                for name in (f"{_METRIC_NAMESPACE}.{dim}" for dim in (*_DIMENSIONS, "score_total"))
                if name in report
            },
            "regressions": regressions,
            "approval": approval,
            "objective": str(objective),
        }
        if bus is not None and hasattr(bus, "emit"):
            await bus.emit(
                "evaluation.scored",
                run_id=run_id,
                target_agent_id=str(run["agent_id"]),
                total_score=total,
                dimensions={name: data["score"] for name, data in scores.items()},
                evaluator_id=_AGENT_ID,
                approval=approval,
                correlation_id=ctx.get("correlation_id"),
            )
        return {
            "success": True,
            "output": output,
            "evidence": evidence,
            "evaluations": [
                {
                    "kind": "evaluation",
                    "run_id": run_id,
                    "total_score": total,
                    "dimensions": {name: data["score"] for name, data in scores.items()},
                }
            ],
        }

    return handler
