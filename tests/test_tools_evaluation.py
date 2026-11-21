"""Evaluation toolset: real scores, real proposals and the gated adoption path."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.core.evaluation import Evaluation
from leo.events import EventBus
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl, arguments_digest
from leo.tools import approval_tools
from leo.tools.evaluation_tools import (
    APPLY_IMPROVEMENT, EVALUATE_RUN, EVALUATION_PROPOSALS, TOOL_NAMES, register,
)
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry

NAMES = list(TOOL_NAMES)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    evaluation = Evaluation()
    registry = ToolRegistry()
    registered = register(registry, services={"evaluation": evaluation, "events": events})
    approval_tools.register(registry)  # model-safe read-only surface
    approval_tools.register_operator(registry)  # trusted human decision surface
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions, evaluation=evaluation,
        registry=registry, executor=executor, registered=registered,
    )


async def run(env, name, args=None, actor="agent-a", approval_id=None):
    call = ToolCall(name=name, arguments=args or {}, actor=actor)
    return await env.executor.execute(call, approval_id=approval_id)


async def seed_run(run_id, evaluations, status="completed", agent_id="reviewer", task_id="task-1"):
    await db.execute(
        "INSERT INTO agent_runs(id,agent_id,task_id,status,started_at,finished_at,result,evaluations,metadata) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (run_id, agent_id, task_id, status, now(), now(), None,
         json.dumps(evaluations), "{}"),
    )


async def test_registration_typed_specs_schema_and_missing_service(env):
    assert env.registered == NAMES
    specs = {s.name: s for s in env.registry.list_specs()}
    assert [specs[name].category for name in NAMES] == ["evaluation"] * 3
    assert specs[EVALUATE_RUN].risk is RiskLevel.LOW
    assert specs[EVALUATION_PROPOSALS].risk is RiskLevel.LOW
    assert specs[APPLY_IMPROVEMENT].risk is RiskLevel.HIGH
    assert specs[EVALUATE_RUN].permissions == ["evaluation:read", "metrics:write"]
    assert specs[EVALUATION_PROPOSALS].permissions == ["evaluation:read", "evaluation:write"]
    assert specs[APPLY_IMPROVEMENT].permissions == ["evaluation:adopt", "approval:use"]
    for name in NAMES:
        schema = specs[name].input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert specs[name].resource_field == ""
    apply_props = specs[APPLY_IMPROVEMENT].input_schema["properties"]
    assert specs[APPLY_IMPROVEMENT].input_schema["required"] == ["proposal_id", "tests_passed"]
    assert set(apply_props) == {"proposal_id", "tests_passed"}
    assert not {"approver", "resolver", "resolved_by", "approval_id"} & set(apply_props)

    missing_run = await run(env, EVALUATE_RUN, {}, actor="operator")
    assert not missing_run.success and "missing required field" in missing_run.error
    bad_action = await run(env, EVALUATION_PROPOSALS, {"action": "delete"}, actor="operator")
    assert not bad_action.success and "must be one of" in bad_action.error
    forged = await run(env, APPLY_IMPROVEMENT, {
        "proposal_id": "imp_x", "tests_passed": True, "approver": "admin",
    })
    assert not forged.success and "unknown field" in forged.error
    missing_tests = await run(env, APPLY_IMPROVEMENT, {"proposal_id": "imp_x"})
    assert not missing_tests.success and "missing required field" in missing_tests.error

    # The model-facing approval toolset must not expose the decision tool.
    model_registry = ToolRegistry()
    approval_tools.register(model_registry)
    assert model_registry.get("approval_resolve") is None
    assert env.registry.get("approval_resolve") is not None  # operator surface only

    bare = ToolRegistry()
    assert register(bare) == NAMES  # registration also works without injection
    with pytest.raises(RuntimeError, match="Evaluation service"):
        await bare.get(EVALUATE_RUN).handler({"run_id": "run_x"}, {"events": env.events})
    with pytest.raises(RuntimeError, match="Evaluation service"):
        await bare.get(APPLY_IMPROVEMENT).handler(
            {"proposal_id": "imp_x", "tests_passed": True},
            {"actor": "agent-a", "settings": env.settings, "events": env.events},
        )


async def test_evaluate_run_scores_seeded_rows_and_records_real_metrics(env):
    await seed_run("run_good", [
        {"kind": "verification", "score": 0.8, "verified": True},
        {"kind": "execution", "success": True, "duration_ms": 12},
    ])
    await seed_run("run_bad", [
        {"kind": "verification", "score": 0.2, "verified": False},
        {"kind": "execution", "success": False},
    ], status="failed", agent_id="drafter")

    good = await run(env, EVALUATE_RUN, {"run_id": "run_good"}, actor="operator")
    assert good.success, good.error
    assert good.output["score"] == pytest.approx(0.9333)
    assert good.output["components"] == {
        "verification": pytest.approx(0.8), "execution": 1.0, "status": 1.0,
    }
    assert good.output["status"] == "completed"
    assert good.output["evaluation_count"] == 2
    assert good.output["metric"] == "leo.run.score"

    bad = await run(env, EVALUATE_RUN, {"run_id": "run_bad"}, actor="operator")
    assert bad.success, bad.error
    assert bad.output["score"] == pytest.approx(0.0667)
    assert bad.output["components"]["execution"] == 0.0

    metric_rows = await db.fetchall(
        "SELECT value, labels FROM metrics WHERE name='leo.run.score' ORDER BY id"
    )
    assert [round(row["value"], 4) for row in metric_rows] == [0.9333, 0.0667]
    assert [json.loads(row["labels"])["run_id"] for row in metric_rows] == ["run_good", "run_bad"]
    report = await env.evaluation.report()
    assert report["leo.run.score"]["count"] == 2
    assert report["leo.run.score"]["mean"] == pytest.approx(0.5)

    events = await db.fetchall(
        "SELECT payload FROM events WHERE topic='evaluation.run_scored' ORDER BY rowid"
    )
    assert len(events) == 2
    assert json.loads(events[0]["payload"])["payload"]["score"] == pytest.approx(0.9333)
    assert json.loads(events[1]["payload"])["payload"]["run_id"] == "run_bad"

    unknown = await run(env, EVALUATE_RUN, {"run_id": "run_missing"}, actor="operator")
    assert not unknown.success and "unknown agent run" in unknown.error


async def test_evaluation_proposals_uses_measured_evidence_and_lists_rows(env):
    for task_id, status in (("t1", "completed"), ("t2", "completed"), ("t3", "failed")):
        await db.execute(
            "INSERT INTO tasks(id,objective,status,created_at,updated_at) VALUES(?,?,?,?,?)",
            (task_id, "objective", status, now(), now()),
        )
    for index in range(3):
        await db.execute(
            "INSERT INTO tool_calls(id,tool,actor,args,risk,decision,success,duration_ms,error,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (f"c_ok_{index}", "seeded_lookup", "agent-a", "{}", "low", "allowed", 1, 5.0, None, now()),
        )
    await db.execute(
        "INSERT INTO tool_calls(id,tool,actor,args,risk,decision,success,duration_ms,error,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        ("c_fail", "seeded_shell", "agent-a", "{}", "medium", "allowed", 0, 9.0, "boom", now()),
    )
    # Older latency 4, newer latency 10: a real regression for the detector.
    for value in ([4.0] * 5) + ([10.0] * 5):
        await env.evaluation.record("latency", value)

    proposed = await run(env, EVALUATION_PROPOSALS, {
        "action": "propose",
        "title": "Investigate latency regression",
        "rationale": "measured latency tripled",
        "metric": "latency",
        "window": 5,
        "threshold": 0.2,
    }, actor="operator")
    assert proposed.success, proposed.error
    proposal = proposed.output["proposal"]
    assert proposal["status"] == "proposed"
    assert proposal["requires_approval"] is True
    assert proposal["rationale"] == "measured latency tripled"
    evidence = proposal["evidence"]
    assert evidence["metric"] == "latency"
    assert evidence["regressions"][0]["metric"] == "latency"
    assert evidence["regressions"][0]["change"] == pytest.approx(1.5)
    assert evidence["tasks"]["by_status"] == {"completed": 2, "failed": 1}
    assert evidence["tool_reliability"]["seeded_lookup"]["success_rate"] == 1.0
    assert evidence["tool_reliability"]["seeded_shell"]["success_rate"] == 0.0

    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (f"improvement:{proposal['id']}",))
    assert json.loads(row["value"])["id"] == proposal["id"]

    listed = await run(env, EVALUATION_PROPOSALS, {"action": "list"}, actor="operator")
    assert listed.success and listed.output["total"] == 1
    assert [p["id"] for p in listed.output["proposals"]] == [proposal["id"]]
    events = await db.fetchall(
        "SELECT payload FROM events WHERE topic='evaluation.proposal_created'"
    )
    assert json.loads(events[0]["payload"])["payload"]["proposal_id"] == proposal["id"]

    oversized = await run(env, EVALUATION_PROPOSALS, {"limit": 1000}, actor="operator")
    assert not oversized.success and "must be <= 100" in oversized.error
    clamped = await env.registry.get(EVALUATION_PROPOSALS).handler(
        {"action": "list", "limit": 1000},
        {"services": {"evaluation": env.evaluation, "events": env.events}},
    )
    assert clamped["limit"] == 100 and clamped["count"] == 1
    no_title = await run(env, EVALUATION_PROPOSALS, {"action": "propose"}, actor="operator")
    assert not no_title.success and "title and rationale" in no_title.error


async def test_apply_improvement_requires_operator_resolved_argument_bound_approval(env):
    proposal = await env.evaluation.propose("cache tool results", "measured", {"calls": 20})
    pid = proposal["id"]
    args = {"proposal_id": pid, "tests_passed": True}
    digest = arguments_digest(args)

    with pytest.raises(ApprovalRequired) as caught:
        await run(env, APPLY_IMPROVEMENT, args, actor="agent-a")
    approval_id = caught.value.approval_id
    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    assert row["status"] == "pending"
    assert row["action"] == "tool:apply_improvement" and row["risk"] == "high"
    assert row["actor"] == "agent-a" and row["requested_by"] == "agent-a"
    assert json.loads(row["metadata"])["arguments_digest"] == digest
    assert (await env.evaluation.proposals())[0]["status"] == "proposed"

    forged = await run(env, APPLY_IMPROVEMENT, args, actor="agent-a", approval_id="apr_not_real")
    assert not forged.success and forged.decision == "denied"
    assert "supplied approval id is not valid" in forged.error

    # The requester cannot decide its own approval, not even via the operator
    # tool surface; the operator tool is registered separately from the model one.
    self_resolved = await run(env, "approval_resolve", {
        "approval_id": approval_id, "approved": True,
    }, actor="agent-a")
    assert not self_resolved.success
    assert "requester cannot resolve its own approval" in self_resolved.error
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,)))["status"] == "pending"

    # Even if a trusted boundary resolves it as the requester, the tool refuses
    # to record that principal as the approver.
    await env.permissions.resolve_approval(approval_id, True, "agent-a")
    self_applied = await run(env, APPLY_IMPROVEMENT, args, actor="agent-a", approval_id=approval_id)
    assert not self_applied.success and "self-approval is not allowed" in self_applied.error
    assert self_applied.decision == "denied"
    assert (await env.evaluation.proposals())[0]["status"] == "proposed"
    denied_audit = await db.fetchone(
        "SELECT reason FROM audit_log WHERE action='tool.execute' "
        "AND reason LIKE '%self-approval is not allowed%' ORDER BY id DESC LIMIT 1"
    )
    assert denied_audit is not None

    with pytest.raises(ApprovalRequired) as second:
        await run(env, APPLY_IMPROVEMENT, args, actor="agent-a")
    human_approval = second.value.approval_id
    assert human_approval != approval_id
    human_row = await db.fetchone("SELECT metadata FROM approvals WHERE id=?", (human_approval,))
    assert json.loads(human_row["metadata"])["arguments_digest"] == digest

    resolved = await run(env, "approval_resolve", {
        "approval_id": human_approval, "approved": True,
        "action": "tool:apply_improvement", "risk": "high",
    }, actor="operator")
    assert resolved.success and resolved.output["resolver"] == "operator"
    approval_row = await db.fetchone("SELECT metadata FROM approvals WHERE id=?", (human_approval,))
    assert json.loads(approval_row["metadata"])["resolved_by"] == "operator"

    # The approval is bound to the exact arguments: a different proposal or a
    # different test claim cannot consume it.
    rebound = await run(env, APPLY_IMPROVEMENT, {
        "proposal_id": "imp_other", "tests_passed": True,
    }, actor="agent-a", approval_id=human_approval)
    assert not rebound.success and rebound.decision == "denied"
    assert "supplied approval id is not valid" in rebound.error

    untested = env.registry.get(APPLY_IMPROVEMENT).handler(
        {"proposal_id": pid, "tests_passed": False},
        {"actor": "agent-a", "settings": env.settings, "events": env.events,
         "services": {"evaluation": env.evaluation, "events": env.events,
                      "permissions": env.permissions}},
    )
    with pytest.raises(PermissionError, match="passing tests were not reported"):
        await untested
    assert (await env.evaluation.proposals())[0]["status"] == "proposed"

    applied = await run(env, APPLY_IMPROVEMENT, args, actor="agent-a", approval_id=human_approval)
    assert applied.success, applied.error
    assert applied.output["status"] == "adopted"
    assert applied.output["approver"] == "operator"
    assert applied.output["approval_id"] == human_approval
    adopted = (await env.evaluation.proposals())[0]
    assert adopted["status"] == "adopted"
    assert adopted["approver"] == "operator" and adopted["tests_passed"] is True

    audit = await db.fetchall(
        "SELECT actor,action,decision,reason,metadata FROM audit_log "
        "WHERE action='tool:apply_improvement' ORDER BY id"
    )
    assert any(r["decision"] == "pending" for r in audit)
    assert any(
        r["decision"] == "allowed" and json.loads(r["metadata"] or "{}").get("approval_id") == human_approval
        for r in audit
    )
    calls = await db.fetchall(
        "SELECT tool,decision,success FROM tool_calls WHERE tool=? ORDER BY id", (APPLY_IMPROVEMENT,)
    )
    assert any(c["decision"] == "allowed" and c["success"] == 1 for c in calls)
    assert any(c["decision"] == "denied" and c["success"] == 0 for c in calls)

    events = await db.fetchall(
        "SELECT payload FROM events WHERE topic='evaluation.improvement_applied'"
    )
    assert len(events) == 1
    event = json.loads(events[0]["payload"])["payload"]
    assert event["proposal_id"] == pid and event["approver"] == "operator"

    replay = await run(env, APPLY_IMPROVEMENT, args, actor="agent-a", approval_id=human_approval)
    assert not replay.success and "no longer proposed" in replay.error
    assert (await env.evaluation.proposals())[0]["approver"] == "operator"
