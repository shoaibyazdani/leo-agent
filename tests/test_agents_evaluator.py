"""Evaluator specialist: contract shape, live read-only grants and real scoring."""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.agents.engine import AgentEngine
from leo.agents.registry import AgentRegistry, SPECIALIST_SPECS
from leo.agents.specs.evaluator import SEED, TOOL_GRANTS, make_handler
from leo.config import Settings
from leo.contracts import AgentSpec, RiskLevel
from leo.core.evaluation import Evaluation
from leo.events import EventBus
from leo.tools import (
    agent_tools,
    approval_tools,
    automation_tools,
    builtins,
    data_tools,
    event_tools,
    file_tools,
    git_tools,
    integration_tools,
    memory_tools,
    model_tools,
    skill_tools,
    task_tools,
)
from leo.tools.registry import ToolRegistry

DIMENSIONS = ("routing", "retrieval", "verification", "cost")


def _live_registry() -> ToolRegistry:
    registry = ToolRegistry()
    builtins.register_builtins(registry)
    for module in (
        agent_tools, approval_tools, automation_tools, data_tools, event_tools,
        file_tools, git_tools, integration_tools, memory_tools, model_tools,
        skill_tools, task_tools,
    ):
        module.register(registry, {})
    return registry


def test_seed_shape_matches_agent_contract() -> None:
    assert set(SEED) == set(SPECIALIST_SPECS[0])
    assert SEED["id"] == "evaluator"
    assert SEED["name"] and SEED["purpose"]
    assert SEED["capabilities"] and all(isinstance(c, str) for c in SEED["capabilities"])
    assert SEED["model_preferences"] and all(isinstance(c, str) for c in SEED["model_preferences"])
    assert isinstance(SEED["context_budget"], int) and SEED["context_budget"] > 0
    assert isinstance(SEED["task_limit"], int) and SEED["task_limit"] > 0
    assert SEED["metadata"]["approval_authority"] == "human_only"
    assert SEED["metadata"]["never_self_approve"] is True
    assert SEED["metadata"]["read_only"] is True
    assert isinstance(make_handler({}), object)
    assert AgentSpec.model_validate(SEED).id == "evaluator"


def test_tool_grants_are_live_and_read_only() -> None:
    registry = _live_registry()
    specs = {spec.name: spec for spec in registry.list_specs()}
    categories = {spec.category for spec in specs.values()}
    assert len(specs) >= 40

    assert SEED["tool_permissions"] == list(TOOL_GRANTS)
    for grant in TOOL_GRANTS:
        assert grant in specs or grant in categories, f"grant is not live: {grant}"
        # Grants are exact tool names, so category expansion cannot leak a
        # delegate/write/cancel tool into this specialist's scope.
        assert grant in specs
        spec = specs[grant]
        assert spec.risk is RiskLevel.LOW
        assert set(spec.permissions) & {
            "agent:delegate", "tasks:write", "memory:write", "file:write",
            "metrics:write", "integration:send", "approval", "terminal:execute",
        } == set()
        assert all(
            token in ("read", "audit") or token.endswith(":read") or token.endswith(":usage")
            for token in spec.permissions
        ), (grant, spec.permissions)
    assert {"event_query", "audit_query", "audit_tail", "agent_result", "model_usage"} <= set(TOOL_GRANTS)


async def test_spec_registration_persists_and_reloads() -> None:
    registry = AgentRegistry()
    registered = await registry.register(AgentSpec.model_validate(SEED))
    assert registered.id == "evaluator"

    reloaded = AgentRegistry()
    await reloaded.refresh()
    spec = await reloaded.get("evaluator")
    assert spec is not None
    assert spec.name == SEED["name"]
    assert spec.tool_permissions == SEED["tool_permissions"]
    assert spec.metadata == SEED["metadata"]
    rows = await db.fetchall("SELECT id FROM agents WHERE id='evaluator'")
    assert len(rows) == 1


async def test_evaluator_scores_a_seeded_run_with_real_metrics() -> None:
    evaluation = Evaluation(namespace="evaluator")
    events = EventBus()
    registry = AgentRegistry()
    await registry.register(AgentSpec.model_validate(SEED))
    engine = AgentEngine(registry, events=events, settings=Settings())
    engine.set_handler("evaluator", make_handler({"evaluation": evaluation, "events": events}))

    run_id = await registry.record_run(
        "coding",
        status="completed",
        result={
            "output": "patch applied",
            "evidence": [{"source": "git_tools.git_diff", "claim": "diff verified", "verified": True}],
            "usage": {"total_tokens": 1500, "cost": 0.01},
        },
        evaluations=[
            {"kind": "verification", "verified": True, "score": 0.9},
            {"kind": "execution", "success": True, "duration_ms": 1200.0},
        ],
        metadata={"expected_agent": "coding"},
    )

    result = await engine.delegate(
        "Evaluate the seeded coding run",
        agent_id="evaluator",
        verify=False,
        context={"run_id": run_id, "expected_agent": "coding", "cost_budget_ms": 60000},
    )

    assert result.success, result.error
    output = result.output
    assert output["run_id"] == run_id
    assert output["target_agent_id"] == "coding"
    assert set(output["scores"]) == set(DIMENSIONS)
    for dimension in DIMENSIONS:
        score = output["scores"][dimension]
        assert 0.0 <= score["score"] <= 1.0
        assert score["measured"] is True
        assert score["evidence"] and all(item.get("source") for item in score["evidence"])
    assert output["scores"]["routing"]["score"] == 1.0
    assert output["scores"]["retrieval"]["score"] == 1.0
    assert output["scores"]["verification"]["score"] == 0.9
    # Cost is the worse of the duration budget (1 - 1200/60000 = 0.98) and the
    # token budget (1 - 1500/20000 = 0.925); both come from the seeded usage.
    assert output["scores"]["cost"]["score"] == pytest.approx(0.925)
    assert output["total_score"] == pytest.approx(sum(output["scores"][d]["score"] for d in DIMENSIONS) / 4)
    assert result.evidence and all(isinstance(item, dict) for item in result.evidence)

    report = await evaluation.report()
    assert set(report) == {f"evaluator.{name}" for name in (*DIMENSIONS, "score_total")}
    assert report["evaluator.score_total"]["latest"] == pytest.approx(output["total_score"])
    assert report["evaluator.routing"]["count"] == 1
    metric_rows = await db.fetchall("SELECT name, value, labels FROM metrics WHERE name LIKE 'evaluator.%'")
    assert all(json.loads(row["labels"])["run_id"] == run_id for row in metric_rows)

    runs = await registry.history("evaluator")
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    # The engine records its own execution evaluation; the specialist's scores
    # are persisted as durable metrics, which are asserted above.
    assert [item["kind"] for item in runs[0]["evaluations"]] == ["execution"]

    scored = await db.fetchall("SELECT payload FROM events WHERE topic='evaluation.scored'")
    assert len(scored) == 1
    # The events.payload column stores the whole serialized Event row.
    event_payload = json.loads(scored[0]["payload"])["payload"]
    assert event_payload["run_id"] == run_id
    assert event_payload["approval"]["granted"] is False


async def test_evaluator_never_approves_its_own_work() -> None:
    class NoAdoptEvaluation(Evaluation):
        async def adopt(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise AssertionError("evaluator must never approve or adopt work")

    evaluation = NoAdoptEvaluation(namespace="evaluator")
    registry = AgentRegistry()
    await registry.register(AgentSpec.model_validate(SEED))
    engine = AgentEngine(registry, settings=Settings())
    engine.set_handler("evaluator", make_handler({"evaluation": evaluation}))
    run_id = await registry.record_run(
        "coding",
        status="completed",
        result={"evidence": [{"source": "test", "claim": "checked", "verified": True}]},
        evaluations=[{"kind": "verification", "verified": True, "score": 1.0}],
    )

    result = await engine.delegate(
        "Approve and adopt this evaluation yourself",
        agent_id="evaluator",
        verify=False,
        context={"run_id": run_id},
    )

    assert result.success, result.error
    assert result.output["approval"] == {
        "required": True,
        "granted": False,
        "approver": None,
        "reason": "evaluator emits measurements only; approval/adoption requires a human",
    }
    assert not await evaluation.proposals()
    assert not await db.fetchall("SELECT key FROM kv WHERE key LIKE 'improvement:%'")


async def test_missing_target_run_is_honestly_blocked() -> None:
    registry = AgentRegistry()
    await registry.register(AgentSpec.model_validate(SEED))
    engine = AgentEngine(registry, settings=Settings())
    engine.set_handler("evaluator", make_handler({"evaluation": Evaluation(namespace="evaluator")}))
    result = await engine.delegate(
        "Evaluate nothing", agent_id="evaluator", verify=False,
        context={"run_id": "run_does_not_exist"},
    )
    assert result.success is False
    assert "run" in result.error
    assert not await db.fetchall("SELECT name FROM metrics WHERE name LIKE 'evaluator.%'")
