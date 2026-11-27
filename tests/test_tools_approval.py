"""Approval toolset: every path runs through ToolExecutorImpl and durable state."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, RiskReason, ToolCall, ToolSpec
from leo.events import EventBus
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl
from leo.tools.approval_tools import register, register_operator
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry

NAMES = ["approval_list", "audit_tail"]
OPERATOR_NAMES = ["approval_resolve"]
DANGER = "approval_test_danger"


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    registered = register(registry, services={"permissions": permissions})
    operator_registered = register_operator(registry, services={"permissions": permissions})

    executed: list[dict] = []

    async def danger(args: dict, ctx: dict):
        executed.append({"args": dict(args), "actor": ctx.get("actor")})
        return {"executed": True, "actor": ctx.get("actor")}

    registry.register(
        ToolSpec(
            name=DANGER,
            description="high-risk fixture action requiring an approval",
            category="test",
            risk=RiskLevel.HIGH,
            resource_field="",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        danger,
    )
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        settings=settings, permissions=permissions, registry=registry,
        executor=executor, executed=executed, registered=registered,
        operator_registered=operator_registered,
    )


async def run(env, name, args=None, actor="agent-a", approval_id=None):
    call = ToolCall(name=name, arguments=args or {}, actor=actor)
    return await env.executor.execute(call, approval_id=approval_id)


async def request_approval(env, actor="agent-a"):
    with pytest.raises(ApprovalRequired) as caught:
        await run(env, DANGER, actor=actor)
    return caught.value.approval_id


async def test_registration_typed_specs_and_schema_guards(env):
    assert env.registered == NAMES
    assert env.operator_registered == OPERATOR_NAMES
    assert [s.name for s in env.registry.list_specs()][:2] == NAMES
    model_only = ToolRegistry()
    assert register(model_only) == NAMES
    assert "approval_resolve" not in {s.name for s in model_only.list_specs()}
    specs = {s.name: s for s in env.registry.list_specs()}
    assert specs["approval_list"].category == "approval"
    assert specs["approval_list"].risk is RiskLevel.LOW
    assert specs["approval_list"].permissions == ["approval", "audit"]
    assert specs["approval_resolve"].risk is RiskLevel.MEDIUM
    assert specs["approval_resolve"].permissions == ["approval"]
    assert specs["audit_tail"].category == "audit"
    assert specs["audit_tail"].risk is RiskLevel.LOW
    for name in NAMES:
        spec = specs[name]
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""
    resolve_props = specs["approval_resolve"].input_schema["properties"]
    assert not {"resolver", "actor", "principal", "requested_by"} & resolve_props.keys()

    forged_principal = await run(
        env, "approval_resolve",
        {"approval_id": "apr_x", "approved": True, "resolver": "attacker"},
    )
    assert not forged_principal.success
    assert "unknown field" in forged_principal.error

    bad_limit = await run(env, "audit_tail", {"limit": 5000})
    assert not bad_limit.success
    assert "must be <= 200" in bad_limit.error


async def test_legitimate_resolve_grants_retry_and_audit_tail_reads_durable_rows(env):
    approval_id = await request_approval(env)

    listed = await run(env, "approval_list", actor="operator")
    assert listed.success
    assert any(a["id"] == approval_id and a["status"] == "pending" for a in listed.output["approvals"])

    resolved = await run(
        env, "approval_resolve", {"approval_id": approval_id, "approved": True},
        actor="operator",
    )
    assert resolved.success
    assert resolved.output == {
        "approval_id": approval_id, "status": "approved", "resolver": "operator",
        "allowed": True, "reason": "approval approved by operator",
        "actor": "agent-a", "action": "tool:" + DANGER,
        "resource": None, "risk": "high",
    }
    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    assert row["status"] == "approved"
    assert json.loads(row["metadata"])["resolved_by"] == "operator"

    approved_list = await run(env, "approval_list", {"status": "approved"})
    assert approved_list.success
    assert any(a["id"] == approval_id for a in approved_list.output["approvals"])

    retried = await run(env, DANGER, actor="agent-a", approval_id=approval_id)
    assert retried.success and retried.output["executed"] is True
    assert env.executed == [{"args": {}, "actor": "agent-a"}]

    audit = await run(env, "audit_tail", {"action": "tool:" + DANGER, "limit": 200})
    assert audit.success and audit.output["count"] >= 3
    assert any(
        r["metadata"].get("approval_id") == approval_id
        and "approved by operator" in (r["reason"] or "")
        for r in audit.output["rows"]
    )
    assert any(r["reason"] == "prior approval verified" for r in audit.output["rows"])

    executed_audit = await run(env, "audit_tail", {"decision": "allowed", "limit": 5})
    assert executed_audit.success
    assert executed_audit.output["count"] <= 5
    assert any(r["action"] == "tool.execute" for r in executed_audit.output["rows"])


async def test_forged_and_wrong_actor_approvals_fail(env):
    forged = await run(
        env, "approval_resolve",
        {"approval_id": "apr_forged_000", "approved": True}, actor="operator",
    )
    assert not forged.success
    assert "unknown approval" in forged.error

    approval_id = await request_approval(env, actor="agent-a")
    resolved = await run(
        env, "approval_resolve",
        {"approval_id": approval_id, "approved": True}, actor="operator",
    )
    assert resolved.success

    # The approval is bound to agent-a: another actor cannot consume it.
    decision = await env.permissions.authorize(
        "mallory", "tool:" + DANGER, None, RiskLevel.HIGH, approval_id=approval_id
    )
    assert not decision.allowed
    denied = await run(env, DANGER, actor="mallory", approval_id=approval_id)
    assert not denied.success
    assert "permission denied" in denied.error
    assert env.executed == []

    # A replayed decision cannot flip a resolved approval.
    replay = await run(
        env, "approval_resolve", {"approval_id": approval_id, "approved": False},
        actor="operator",
    )
    assert not replay.success
    assert "already approved" in replay.error


async def test_expired_approval_is_not_resolvable_or_usable(env):
    approval_id = await request_approval(env)
    stale = (datetime.now(timezone.utc) - timedelta(seconds=env.settings.security.approval_timeout_seconds + 60)).isoformat()
    await db.execute("UPDATE approvals SET created_at=? WHERE id=?", (stale, approval_id))

    resolved = await run(
        env, "approval_resolve", {"approval_id": approval_id, "approved": True},
        actor="operator",
    )
    assert not resolved.success
    assert "expired" in resolved.error.lower()
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,))
    assert row["status"] == "pending"

    decision = await env.permissions.authorize(
        "agent-a", "tool:" + DANGER, None, RiskLevel.HIGH, approval_id=approval_id
    )
    assert not decision.allowed
    denied = await run(env, DANGER, actor="agent-a", approval_id=approval_id)
    assert not denied.success
    assert "permission denied" in denied.error
    assert env.executed == []


async def test_resolve_enforces_action_resource_and_risk_binding(env):
    for binding in ({"action": "tool:other"}, {"resource": "/etc/passwd"}, {"risk": "low"}):
        approval_id = await request_approval(env)
        result = await run(
            env, "approval_resolve",
            {"approval_id": approval_id, "approved": True, **binding}, actor="operator",
        )
        assert not result.success
        assert "binding mismatch" in result.error
        row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,))
        assert row["status"] == "pending"

    approval_id = await request_approval(env)
    matched = await run(
        env, "approval_resolve",
        {"approval_id": approval_id, "approved": False,
         "action": "tool:" + DANGER, "risk": "high"}, actor="operator",
    )
    assert matched.success and matched.output["status"] == "denied"
    denied = await run(env, DANGER, actor="agent-a", approval_id=approval_id)
    assert not denied.success
    assert "permission denied" in denied.error
    assert env.executed == []


async def test_audit_tail_is_bounded_and_filterable(env):
    for i in range(12):
        await db.execute(
            "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), "actor-" + str(i), "custom.action",
             None, "low", "allowed", "row-" + str(i), "{}"),
        )
    tail = await run(env, "audit_tail", {"action": "custom.action", "limit": 7})
    assert tail.success and tail.output["count"] == 7
    assert tail.output["limit"] == 7
    assert all(r["action"] == "custom.action" for r in tail.output["rows"])
    assert [r["reason"] for r in tail.output["rows"]] == ["row-11", "row-10", "row-9", "row-8", "row-7", "row-6", "row-5"]

    oldest_id = (await db.fetchone(
        "SELECT MIN(id) AS id FROM audit_log WHERE action='custom.action'"
    ))["id"]
    cursor = await run(env, "audit_tail", {"action": "custom.action", "after_id": oldest_id, "limit": 200})
    assert cursor.success and cursor.output["count"] == 11

    # Direct handler calls clamp oversized limits too (defence in depth).
    clamped = await env.registry.get("audit_tail").handler({"limit": 10**9}, {})
    assert clamped["limit"] == 200 and clamped["count"] <= 200


async def test_missing_permission_service_raises_clear_error(env):
    handler = env.registry.get("approval_list").handler
    with pytest.raises(RuntimeError, match="PermissionManager"):
        await handler({}, {"settings": env.settings, "services": {}})
    resolve = env.registry.get("approval_resolve").handler
    with pytest.raises(RuntimeError, match="PermissionManager"):
        await resolve({"approval_id": "apr_x", "approved": True}, {"settings": env.settings})


# --------------------------------------------------------------------------- decision isolation


async def test_resolve_is_not_model_callable(env):
    """A model-facing registry has no approval decision tool at all."""
    model_registry = ToolRegistry()
    assert register(model_registry) == NAMES
    model_executor = ToolExecutorImpl(
        model_registry, env.permissions, EventBus(), env.settings
    )
    approval_id = await request_approval(env)
    forged = await model_executor.execute(
        ToolCall(
            name="approval_resolve",
            arguments={"approval_id": approval_id, "approved": True},
            actor="operator",
        )
    )
    assert forged.success is False
    assert "unknown tool" in (forged.error or "")
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,))
    assert row["status"] == "pending"


async def test_requester_cannot_resolve_own_approval(env):
    approval_id = await request_approval(env, actor="agent-a")
    self_resolved = await run(
        env, "approval_resolve",
        {"approval_id": approval_id, "approved": True}, actor="agent-a",
    )
    assert self_resolved.success is False
    assert "requester" in (self_resolved.error or "").lower()
    row = await db.fetchone("SELECT status,metadata FROM approvals WHERE id=?", (approval_id,))
    assert row["status"] == "pending"
    assert "resolved_by" not in json.loads(row["metadata"])


async def test_chat_actor_cannot_resolve_any_approval(env):
    approval_id = await request_approval(env, actor="agent-a")
    for actor in ("chat:cli", "CHAT:API"):
        result = await run(
            env, "approval_resolve",
            {"approval_id": approval_id, "approved": True}, actor=actor,
        )
        assert result.success is False
        assert "chat actor" in (result.error or "").lower()
    row = await db.fetchone("SELECT status,metadata FROM approvals WHERE id=?", (approval_id,))
    assert row["status"] == "pending"
    assert "resolved_by" not in json.loads(row["metadata"])


# ---------------------------------------------------------- waiting for a human

async def _pending_approval(action: str, timeout: float = 3.0) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        row = await db.fetchone(
            "SELECT id FROM approvals WHERE action=? AND status='pending' "
            "ORDER BY rowid DESC LIMIT 1",
            (action,),
        )
        if row is not None:
            return str(row["id"])
        await asyncio.sleep(0.01)
    raise AssertionError(f"no pending approval for {action}")


async def test_wait_for_approval_resumes_the_same_call(env):
    """Approving after the request lands resumes the waiting tool call."""
    env.settings.security.wait_for_approval = True
    env.settings.security.approval_poll_seconds = 0.02
    env.settings.security.approval_timeout_seconds = 5

    task = asyncio.create_task(run(env, DANGER))
    approval_id = await _pending_approval("tool:" + DANGER)
    await env.permissions.resolve_approval(approval_id, True, resolver="operator")

    result = await asyncio.wait_for(task, 5)
    assert result.decision == "allowed"
    assert result.success is True
    assert env.executed, "the waiting call must dispatch after approval"


async def test_wait_for_approval_denied_returns_a_denial(env):
    env.settings.security.wait_for_approval = True
    env.settings.security.approval_poll_seconds = 0.02
    env.settings.security.approval_timeout_seconds = 5

    task = asyncio.create_task(run(env, DANGER))
    approval_id = await _pending_approval("tool:" + DANGER)
    await env.permissions.resolve_approval(approval_id, False, resolver="operator")

    result = await asyncio.wait_for(task, 5)
    assert result.decision == "denied"
    assert result.success is False
    assert "approval denied" in (result.error or "")
    assert not env.executed


async def test_wait_for_approval_times_out_without_killing_the_turn(env):
    env.settings.security.wait_for_approval = True
    env.settings.security.approval_poll_seconds = 0.05
    env.settings.security.approval_timeout_seconds = 1

    result = await run(env, DANGER)

    assert result.decision == "denied"
    assert "timed out" in (result.error or "")
    assert not env.executed


async def _shell_decision(
    env, reviewer,
    reason=RiskReason("unclassified command (rsvg-convert)", trustable=True),
):
    env.permissions.set_reviewer(reviewer)
    return await env.permissions.authorize(
        "operator", "tool:shell", str(env.settings.root.resolve()), RiskLevel.HIGH,
        risk_reason=reason,
        review_command="rsvg-convert -z 2 -o girl.png girl.svg",
    )


async def test_auto_reviewer_allow_runs_without_an_approval(env):
    calls = []

    async def reviewer(**kwargs):
        calls.append(kwargs)
        return "allow", "ordinary image conversion"

    decision = await _shell_decision(env, reviewer)

    assert decision.allowed is True and decision.requires_approval is False
    assert "auto-reviewed" in decision.reason
    assert calls and calls[0]["command"].startswith("rsvg-convert")
    row = await db.fetchone("SELECT COUNT(*) AS n FROM approvals")
    assert row["n"] == 0


async def test_auto_reviewer_deny_does_not_create_an_approval(env):
    async def reviewer(**kwargs):
        return "deny", "writes outside the workspace"

    decision = await _shell_decision(env, reviewer)

    assert decision.allowed is False and decision.requires_approval is False
    assert "auto-review denied" in decision.reason
    row = await db.fetchone("SELECT COUNT(*) AS n FROM approvals")
    assert row["n"] == 0


async def test_auto_reviewer_ask_creates_a_human_approval(env):
    async def reviewer(**kwargs):
        return "ask", "uncertain"

    decision = await _shell_decision(env, reviewer)

    assert decision.requires_approval is True and decision.approval_id
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (decision.approval_id,)
    )
    assert row["status"] == "pending"


async def test_auto_reviewer_is_not_used_for_a_dangerous_reason(env):
    calls = []

    async def reviewer(**kwargs):
        calls.append(kwargs)
        return "allow", ""

    decision = await _shell_decision(env, reviewer, reason="rm is destructive")

    assert decision.requires_approval is True
    assert calls == [], "destructive calls must never reach the reviewer"


async def test_auto_reviewer_failure_falls_back_to_a_human(env):
    async def reviewer(**kwargs):
        raise RuntimeError("reviewer down")

    decision = await _shell_decision(env, reviewer)

    assert decision.requires_approval is True and decision.approval_id


async def test_auto_reviewer_never_reaches_a_high_classifier_reason(env):
    """An interpreter flagged HIGH by the classifier stays with the human."""
    calls = []

    async def reviewer(**kwargs):
        calls.append(kwargs)
        return "allow", "local test run"

    env.permissions.set_reviewer(reviewer)
    decision = await env.permissions.authorize(
        "operator", "tool:shell", str(env.settings.root.resolve()), RiskLevel.HIGH,
        risk_reason=RiskReason("python runs arbitrary code", trustable=False),
        review_command="python -c 'print(1)'",
    )

    assert decision.requires_approval is True
    assert calls == []


@pytest.mark.parametrize("reason", [
    "git history or remote rewrite",
    "hardline blocklist: filesystem root wipe (rm -rf /)",
    "sudo escalates privileges",
    "command substitution",
    "redirection outside allowed roots",
    "unparseable command",
])
async def test_auto_reviewer_keeps_hardline_reasons_with_the_human(env, reason):
    calls = []

    async def reviewer(**kwargs):
        calls.append(kwargs)
        return "allow", ""

    decision = await _shell_decision(env, reviewer, reason=reason)

    assert decision.requires_approval is True
    assert calls == []


async def test_expiry_is_durable_and_emits_an_explicit_event(env):
    seen = []

    async def on_expired(event):
        seen.append(event.payload)

    env.permissions.events.subscribe("approval.expired", on_expired)
    decision = await env.permissions.authorize(
        "operator", "tool:shell", str(env.settings.root.resolve()), RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (rsvg-convert)", trustable=True),
        command_preview="rsvg-convert -z 2 -o girl.png girl.svg",
        task_id="conv-1",
    )
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await db.execute(
        "UPDATE approvals SET created_at=?, expires_at=? WHERE id=?",
        (old, old, decision.approval_id),
    )

    assert await env.permissions.expire_approval_if_stale(decision.approval_id) is True
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (decision.approval_id,))
    assert row["status"] == "expired"
    assert seen and seen[0]["command"].startswith("rsvg-convert")
    assert seen[0]["reason"] and "not run" in seen[0]["reason"]
    assert await env.permissions.expire_approval_if_stale(decision.approval_id) is False


async def test_startup_recovers_a_card_whose_timer_died_with_the_loop(env):
    """A restart re-evaluates stored deadlines and still shows the notice."""
    seen = []
    env.permissions.events.subscribe("approval.expired", lambda event: seen.append(event.payload))
    decision = await env.permissions.authorize(
        "operator", "tool:shell", str(env.settings.root.resolve()), RiskLevel.HIGH,
        risk_reason=RiskReason("ordinary command", trustable=True),
        command_preview="rsvg-convert in.png out.png", task_id="restart-1",
    )
    assert decision.requires_approval is True
    stored = await db.fetchone("SELECT expires_at FROM approvals WHERE id=?", (decision.approval_id,))
    assert stored["expires_at"], "the deadline must be durable, not only a loop timer"
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await db.execute(
        "UPDATE approvals SET created_at=?, expires_at=? WHERE id=?",
        (old, old, decision.approval_id),
    )
    # A fresh manager has no in-memory timer: only the durable row survives.
    restarted = PermissionManagerImpl(env.settings, env.permissions.events)

    assert await restarted.recover_pending_expiries() == 1
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (decision.approval_id,))
    assert row["status"] == "expired"
    assert seen and seen[0]["approval_id"] == decision.approval_id
    assert seen[0]["command"] == "rsvg-convert in.png out.png"
    assert seen[0]["reason"] and "not run" in seen[0]["reason"]


async def test_startup_rearms_a_card_that_is_still_live(env):
    decision = await env.permissions.authorize(
        "operator", "tool:shell", str(env.settings.root.resolve()), RiskLevel.HIGH,
        risk_reason=RiskReason("ordinary command", trustable=True),
        command_preview="pytest -q", task_id="restart-2",
    )
    restarted = PermissionManagerImpl(env.settings, env.permissions.events)

    assert await restarted.recover_pending_expiries() == 0
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (decision.approval_id,))
    assert row["status"] == "pending"
    assert decision.approval_id in restarted._expiry_handles

    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await db.execute(
        "UPDATE approvals SET expires_at=? WHERE id=?", (old, decision.approval_id)
    )
    assert await restarted.expire_approval_if_stale(decision.approval_id) is True
