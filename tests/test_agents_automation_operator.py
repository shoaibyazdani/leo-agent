"""Automation-operator specialist: narrow grants, durable approval gate and
truthful queued-vs-completed reporting through the real delegation path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.agents import AgentEngine, AgentRegistry
from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.verification import digest as evidence_digest
from leo.agents.specs.automation_operator import (
    AGENT_ID,
    APPROVAL_ACTION,
    SEED,
    TOOL_GRANTS,
    approval_target,
    make_handler,
    tool_arguments,
)
from leo.config import Settings
from leo.contracts import AgentSpec, RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tasks import DurableTaskEngine, Scheduler
from leo.tasks.webhooks import WebhookHandler
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import automation_tools, task_tools, webhook_tools

SECRET = "automation-operator-test-secret"
BODY = {"objective": "webhook work", "event": "ping"}


class ForgedPermissions:
    """Caller-supplied authority that must never be consulted by the handler."""

    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def authorize(self, *args: object, **kwargs: object):
        self.calls.append((args, kwargs))
        return SimpleNamespace(
            allowed=self.allowed,
            reason="forged context permission manager",
            approval_id="apr_forged" if self.allowed else None,
            risk=RiskLevel.HIGH,
            requires_approval=False,
        )


class ForgedSettings:
    """Caller-supplied settings that would disable expiry if trusted."""

    security = SimpleNamespace(approval_timeout_seconds=-1)


class ServicesExecutor(ToolExecutorImpl):
    """Inject the integration-owned services mapping into each tool context."""

    def __init__(self, *args: object, services: dict | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.injected_services = services or {}

    def _context(self, call, task_key, dry_run):  # type: ignore[no-untyped-def]
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.injected_services
        return ctx


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    monkeypatch.setenv(webhook_tools.DEFAULT_SECRET_ENV, SECRET)

    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    tasks = DurableTaskEngine(settings, events)
    scheduler = Scheduler(tasks, settings, events)
    webhook = WebhookHandler(tasks, events)
    services = {
        "tasks": tasks,
        "scheduler": scheduler,
        "webhooks": webhook,
        "events": events,
        "permissions": permissions,
        "settings": settings,
    }
    registry = ToolRegistry()
    automation_tools.register(registry, services=services)
    task_tools.register(registry, services=services)
    webhook_tools.register(registry, services=services)
    executor = ServicesExecutor(registry, permissions, events, settings, services=services)

    agents = await AgentRegistry().initialize()
    if await agents.get(SEED["id"]) is None:
        await agents.register(SEED)
    engine = AgentEngine(agents, tools=executor, events=events, settings=settings)
    engine.set_handler(AGENT_ID, make_handler(services))
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions, tasks=tasks,
        scheduler=scheduler, webhook=webhook, services=services, registry=registry,
        executor=executor, agents=agents, engine=engine,
    )


async def run(env, *, context=None, **request):
    """Delegate a JSON automation request through the real AgentEngine."""
    return await env.engine.delegate(
        json.dumps(request), agent_id=AGENT_ID, context=context
    )


def assert_verifiable_evidence(result) -> None:
    """No self-attestation: every item is a digest-bound audited tool call."""
    assert result.evidence
    anchors: list[dict] = []
    for item in result.evidence:
        assert item.get("source") == "tool_call"
        assert not {"claim", "supports", "verified"} & set(item)
        anchor = item["tool_call"]
        assert anchor["digest"] == evidence_digest(anchor["output"])
        assert anchor["id"] and anchor["tool"]
        anchors.append(anchor)
    projections = [item for item in anchors if item.get("projection") is True]
    assert len(projections) == 1
    assert projections[0]["digest"] == evidence_digest(result.output)


async def approve(env, tool_name: str, arguments: dict) -> str:
    """Request and resolve a real durable approval bound to one mutation."""
    target = approval_target(tool_name, arguments)
    decision = await env.permissions.authorize(
        AGENT_ID,
        target["action"],
        target["resource"],
        RiskLevel.HIGH,
        arguments_digest=target["arguments_digest"],
        tool=tool_name,
    )
    assert decision.requires_approval and decision.approval_id
    resolved = await env.permissions.resolve_approval(
        decision.approval_id, True, "operator@leo.local"
    )
    assert resolved.allowed
    return decision.approval_id


def test_seed_shape_and_no_string_handler_metadata() -> None:
    expected_keys = set(SPECIALIST_SPECS[0])
    assert expected_keys == {
        "id", "name", "purpose", "capabilities", "tool_permissions",
        "model_preferences", "context_budget", "task_limit", "metadata",
    }
    assert set(SEED) == expected_keys
    assert SEED["id"] == AGENT_ID
    assert SEED["name"] and SEED["purpose"] and SEED["capabilities"]
    assert SEED["model_preferences"] and SEED["tool_permissions"]
    assert SEED["context_budget"] > 0 and SEED["task_limit"] > 0
    assert isinstance(SEED["metadata"], dict)

    metadata = SEED["metadata"]
    # AgentEngine treats metadata['handler'] as a callable: a string there
    # would be an inert (and broken) execution path.
    assert "handler" not in metadata
    assert not any(
        key == "handler" and isinstance(value, str) for key, value in metadata.items()
    )
    assert metadata["handler_kind"] == "bespoke:make_handler"
    assert AgentSpec.model_validate(SEED).metadata.get("handler") is None
    assert callable(make_handler({}))


async def test_narrow_tool_grants_are_exact_registered_names(env) -> None:
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    grants = set(SEED["tool_permissions"])
    assert grants == set(TOOL_GRANTS)
    assert len(grants) == 10
    assert grants <= set(specs), f"unknown grants: {grants - set(specs)}"

    categories = {spec.category for spec in specs.values()}
    assert not grants & categories, "category grants would widen the tool scope"
    assert "task_cancel" not in grants  # not used by this specialist
    assert {"schedule_create", "schedule_run_now", "webhook_verify"} <= grants

    # Truthful webhook surface: the server must not expose a signing oracle.
    assert {name for name in grants if name.startswith("webhook_")} == set(
        webhook_tools.TOOL_NAMES
    ) == {"webhook_routes", "webhook_verify", "webhook_deliveries"}
    assert "webhook_sign" not in specs and "webhook_sign" not in grants
    # Approval resolution is operator-only, never a model-callable grant.
    assert "approval_resolve" not in specs and "approval_resolve" not in grants

    for name in grants:
        assert ToolExecutorImpl._in_scope(env.registry.get(name), frozenset(grants))
    # These are registered tools, but deliberately absent from this grant set.
    for name in ("task_cancel", "task_list"):
        assert name in specs
        assert not ToolExecutorImpl._in_scope(env.registry.get(name), frozenset(grants))

    # A forged out-of-scope call is denied by the real executor façade before
    # its handler, permission check or audit of a successful call.
    scoped = env.engine._scoped_tools(AgentSpec.model_validate(SEED))
    assert scoped is not None
    denied = await scoped.execute(
        ToolCall(name="task_cancel", arguments={"task_id": "task_foreign"})
    )
    assert denied.success is False and denied.decision == "out_of_scope"


async def test_spec_persists_and_reloads_without_handler_string(env) -> None:
    spec = await env.agents.get(AGENT_ID)
    assert spec is not None
    assert spec.metadata["handler_kind"] == "bespoke:make_handler"
    assert "handler" not in spec.metadata

    await env.agents.refresh()
    reloaded = await env.agents.get(AGENT_ID)
    assert reloaded is not None and reloaded.id == AGENT_ID
    assert set(reloaded.tool_permissions) == set(TOOL_GRANTS)
    row = await db.fetchone("SELECT metadata FROM agents WHERE id=?", (AGENT_ID,))
    stored = json.loads(row["metadata"])
    assert "handler" not in stored
    assert stored["handler_kind"] == "bespoke:make_handler"
    assert stored["approval_action"] == APPROVAL_ACTION


async def test_trigger_reports_queued_task_not_executed_run(env) -> None:
    await env.scheduler.register_interval("nightly", 60, "build report", {"channel": "ops"})
    args = {"name": "nightly"}
    approval_id = await approve(env, "schedule_run_now", args)

    result = await run(env, action="trigger_schedule", name="nightly", approval_id=approval_id)
    assert result.success, result.error
    out = result.output
    assert out["status"] == "queued"
    assert out["triggered"] is True
    assert out["executed"] is False
    assert out["completed"] is False
    assert out["task_status"] == "created"
    assert out["delivery"]["outcome"] == "queued"
    assert out["delivery"]["acknowledged"] is True
    assert out["delivery"]["executed"] is False
    assert out["delivery"]["completed"] is False
    assert out["approval"]["approved"] is True
    assert out["approval"]["approval_id"] == approval_id

    task_id = out["task_id"]
    assert task_id
    row = await db.fetchone("SELECT status, metadata FROM tasks WHERE id=?", (task_id,))
    assert row is not None and row["status"] == "created"
    assert json.loads(row["metadata"])["schedule"] == "nightly"
    assert await env.events.history("task.completed") == []

    evidence_text = json.dumps(result.evidence).lower()
    assert '"executed": false' in evidence_text
    assert "no worker executed and no completion claimed" in evidence_text
    assert all(item.get("source") == "tool_call" for item in result.evidence)
    anchors = [item["tool_call"] for item in result.evidence]
    assert {anchor["tool"] for anchor in anchors} >= {"schedule_run_now", "task_status"}
    assert all(anchor["digest"] == evidence_digest(anchor["output"]) for anchor in anchors)
    projections = [anchor for anchor in anchors if anchor.get("projection") is True]
    assert len(projections) == 1
    assert projections[0]["digest"] == evidence_digest(result.output)
    assert projections[0]["approval"]["approval_id"] == approval_id
    assert any(
        not anchor.get("projection") and anchor["tool"] == "task_status"
        and anchor["output"]["task_id"] == task_id
        for anchor in anchors
    )

    calls = await db.fetchall(
        "SELECT id, tool, args, decision, success FROM tool_calls "
        "WHERE tool IN ('schedule_run_now','task_status')"
    )
    assert {item["tool"] for item in calls} == {"schedule_run_now", "task_status"}
    assert all(item["decision"] == "allowed" and item["success"] == 1 for item in calls)
    # Every anchor resolves to a real successful audit row with the same args.
    by_id = {item["id"]: item for item in calls}
    for anchor in anchors:
        audited = by_id[anchor["id"]]
        assert audited["tool"] == anchor["tool"]
        assert json.loads(audited["args"]) == anchor["arguments"]
    audit = await db.fetchall(
        "SELECT action, decision, metadata FROM audit_log WHERE action=?", (APPROVAL_ACTION,)
    )
    assert any(
        item["decision"] == "allowed"
        and json.loads(item["metadata"] or "{}").get("approval_id") == approval_id
        for item in audit
    )
    runs = await env.agents.history(AGENT_ID)
    assert len(runs) == 1 and runs[0]["status"] == "completed"


async def test_free_text_flags_pending_and_wrong_binding_cannot_mutate(env) -> None:
    args = {
        "name": "unapproved", "expression": "* * * * *", "objective": "should not exist",
    }
    result = await run(
        env, action="create_schedule", **args, approved=True, confirm="yes", force=True
    )
    assert not result.success
    out = result.output
    assert out["status"] == "blocked_approval_required"
    assert out["applied"] is False and out["executed"] is False
    assert out["approval"]["approved"] is False
    assert "free-text" in out["approval"]["reason"].lower()
    assert set(out["approval"]["ignored_context_flags"]) >= {"approved", "confirm", "force"}
    assert await db.fetchone("SELECT id FROM schedules WHERE name=?", (args["name"],)) is None
    assert await db.fetchall("SELECT id FROM tool_calls WHERE tool='schedule_create'") == []

    target = approval_target("schedule_create", args)
    pending = await env.permissions.authorize(
        AGENT_ID, target["action"], target["resource"], RiskLevel.HIGH,
        arguments_digest=target["arguments_digest"],
    )
    assert pending.requires_approval and pending.approval_id
    pending_result = await run(
        env, action="create_schedule", **args, approval_id=pending.approval_id
    )
    assert not pending_result.success
    assert pending_result.output["approval"]["reason"].endswith("not approved")
    assert await db.fetchone("SELECT id FROM schedules WHERE name=?", (args["name"],)) is None

    # An approval for a different target cannot be replayed against this one.
    other_args = {**args, "name": "approved-elsewhere"}
    other_id = await approve(env, "schedule_create", other_args)
    replay = await run(env, action="create_schedule", **args, approval_id=other_id)
    assert not replay.success
    assert "mismatch" in replay.output["approval"]["reason"].lower()
    assert await db.fetchone("SELECT id FROM schedules WHERE name=?", (args["name"],)) is None
    assert await db.fetchall("SELECT id FROM tool_calls WHERE tool='schedule_create'") == []


async def test_create_and_delete_schedule_only_after_verified_approval(env) -> None:
    args = {
        "name": "hourly", "expression": "@every 60s", "objective": "check queue",
        "payload": {"channel": "ops"},
    }
    created = await run(
        env, action="create_schedule", **args,
        approval_id=await approve(env, "schedule_create", args),
    )
    assert created.success, created.error
    assert_verifiable_evidence(created)
    assert created.output["status"] == "applied"
    assert created.output["applied"] is True
    assert created.output["executed"] is False  # creating a schedule is not a run
    row = await db.fetchone("SELECT * FROM schedules WHERE name='hourly'")
    assert row is not None and row["expression"] == "@every 60s"
    assert json.loads(row["payload"])["channel"] == "ops"

    delete_args = {"name": "hourly"}
    deleted = await run(
        env, action="delete_schedule", **delete_args,
        approval_id=await approve(env, "schedule_delete", delete_args),
    )
    assert deleted.success, deleted.error
    assert deleted.output["status"] == "applied"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='hourly'") is None
    tools = {
        item["tool"] for item in await db.fetchall(
            "SELECT tool FROM tool_calls WHERE tool LIKE 'schedule_%'"
        )
    }
    assert {"schedule_create", "schedule_list", "schedule_delete"} <= tools


async def test_queued_task_becomes_completed_only_when_engine_records_it(env) -> None:
    args = {"objective": "durable work"}
    mutation_args = tool_arguments("enqueue_task", args)
    queued = await run(
        env, action="enqueue_task", **args,
        approval_id=await approve(env, "task_create", mutation_args),
    )
    assert queued.success, queued.error
    assert_verifiable_evidence(queued)
    assert queued.output["status"] == "queued"
    assert queued.output["executed"] is False
    assert queued.output["completed"] is False
    task_id = queued.output["task_id"]
    assert task_id

    before_status = await run(env, action="task_status", task_id=task_id)
    assert before_status.success, before_status.error
    assert before_status.output["status"] == "created"
    assert before_status.output["completed"] is False

    before_result = await run(env, action="task_result", task_id=task_id)
    assert before_result.success, before_result.error
    assert before_result.output["completed"] is False
    assert before_result.output["result"] is None

    await env.tasks.complete(
        task_id, {"ok": True}, [{"kind": "test", "detail": "evidence-backed"}]
    )
    after_status = await run(env, action="task_status", task_id=task_id)
    assert after_status.success and after_status.output["completed"] is True
    assert after_status.output["status"] == "completed"
    after_result = await run(env, action="task_result", task_id=task_id)
    assert after_result.success and after_result.output["completed"] is True
    assert after_result.output["result"] == {"ok": True}


async def test_webhook_delivery_reporting_is_truthful(env) -> None:
    routes = await run(env, action="webhook_routes")
    assert routes.success, routes.error
    assert_verifiable_evidence(routes)
    assert routes.output["status"] == "observed"
    assert isinstance(routes.output["routes"], list)
    assert SECRET not in json.dumps(routes.output)

    # Sign locally for fixture setup only: webhook_sign is intentionally not a
    # registered, model-callable tool (it would be an HMAC signing oracle).
    signed = WebhookHandler.sign_v1(BODY, SECRET, delivery_id="d-auto-1")
    accepted = await env.webhook.handle({"body": BODY, "headers": signed})
    assert accepted["status"] == "accepted"

    verified = await run(
        env, action="verify_webhook", body=BODY, headers=signed, delivery_id="d-auto-1"
    )
    assert verified.success, verified.error
    assert verified.output["verified"] is True
    assert verified.output["delivery"]["executed"] is False
    assert verified.output["delivery"]["completed"] is False

    deliveries = await run(env, action="delivery_status", delivery_id="d-auto-1")
    assert deliveries.success, deliveries.error
    assert deliveries.output["delivery"]["outcome"] == "accepted"
    assert deliveries.output["delivery"]["accepted"] == 1
    assert deliveries.output["delivery"]["executed"] is False
    assert deliveries.output["delivery"]["completed"] is False
    row = deliveries.output["deliveries"][0]
    assert row["status"] == "accepted"
    assert row["task_id"] == accepted["task_id"]
    assert row["task_status"] == "created"
    task_row = await db.fetchone("SELECT status FROM tasks WHERE id=?", (accepted["task_id"],))
    assert task_row is not None and task_row["status"] == "created"

    tampered = await run(
        env, action="verify_webhook", body={"objective": "changed"},
        headers=signed, delivery_id="d-auto-1",
    )
    assert not tampered.success
    assert tampered.output["status"] == "failed"
    assert "verified" not in tampered.output  # no success claim on failure

    tools = {
        item["tool"] for item in await db.fetchall(
            "SELECT tool FROM tool_calls WHERE tool LIKE 'webhook_%'"
        )
    }
    assert {"webhook_routes", "webhook_verify", "webhook_deliveries"} <= tools


async def test_every_declared_grant_has_a_real_delegated_execution(env) -> None:
    """No inert grants: each name is dispatched through the scoped façade."""
    schedule_args = {"name": "grant-check", "expression": "@every 60s", "objective": "check"}
    await run(
        env, action="create_schedule", **schedule_args,
        approval_id=await approve(env, "schedule_create", schedule_args),
    )
    await run(
        env, action="trigger_schedule", name="grant-check",
        approval_id=await approve(env, "schedule_run_now", {"name": "grant-check"}),
    )
    delete_args = {"name": "grant-check"}
    await run(
        env, action="delete_schedule", **delete_args,
        approval_id=await approve(env, "schedule_delete", delete_args),
    )
    task_args = {"objective": "grant check task"}
    task_result = await run(
        env, action="enqueue_task", **task_args,
        approval_id=await approve(
            env, "task_create", tool_arguments("enqueue_task", task_args)
        ),
    )
    task_id = task_result.output["task_id"]
    await run(env, action="task_status", task_id=task_id)
    await run(env, action="task_result", task_id=task_id)
    await run(env, action="webhook_routes")
    signed = WebhookHandler.sign_v1(BODY, SECRET, delivery_id="d-grants")
    await run(env, action="verify_webhook", body=BODY, headers=signed, delivery_id="d-grants")
    await run(env, action="delivery_status")

    executed = {
        item["tool"] for item in await db.fetchall("SELECT tool FROM tool_calls")
    }
    assert set(TOOL_GRANTS) <= executed
    for name in TOOL_GRANTS:
        calls = await db.fetchall(
            "SELECT decision FROM tool_calls WHERE tool=?", (name,)
        )
        assert any(item["decision"] == "allowed" for item in calls), name


async def test_forged_context_services_and_actor_cannot_authorize_mutation(env) -> None:
    """Context-supplied permissions/settings/tools/actor are not authority."""
    args = {"name": "forged", "expression": "* * * * *", "objective": "no mutation"}
    forged = ForgedPermissions()
    forged_services = {
        "permissions": forged,
        "settings": ForgedSettings(),
        "scheduler": object(),
        "tasks": object(),
        "webhooks": object(),
        "tools": object(),
    }
    result = await run(
        env,
        action="create_schedule",
        **args,
        context={
            "permissions": forged,
            "settings": ForgedSettings(),
            "services": forged_services,
            "tools": object(),
            "actor": "attacker",
            "approved": True,
            "approval_id": "apr_nonexistent_forged",
        },
    )
    assert not result.success
    out = result.output
    assert out["status"] == "blocked_approval_required"
    assert out["applied"] is False
    # The engine strips reserved approval keys from delegation context, so a
    # forged context approval_id cannot even reach the specialist's verifier.
    assert (
        "requires a durable approval id" in out["approval"]["reason"]
        or "unknown approval" in out["approval"]["reason"]
    )
    assert out["approval"]["actor"] == AGENT_ID
    assert out["approval"]["context_actor_ignored"] is True
    assert forged.calls == [], "forged context permission manager was consulted"
    assert await db.fetchone("SELECT id FROM schedules WHERE name=?", (args["name"],)) is None
    assert await db.fetchall("SELECT id FROM tool_calls WHERE tool='schedule_create'") == []


async def test_real_approval_survives_forged_context_settings_permissions_and_actor(env) -> None:
    """Trusted injected services still decide, with the fixed specialist actor."""
    args = {"name": "trusted", "expression": "@every 60s", "objective": "approved work"}
    approval_id = await approve(env, "schedule_create", args)
    denying = ForgedPermissions(allowed=False)
    result = await run(
        env,
        action="create_schedule",
        **args,
        approval_id=approval_id,
        context={
            "permissions": denying,
            "settings": ForgedSettings(),  # would mark the approval expired
            "services": {"permissions": denying, "settings": ForgedSettings()},
            "actor": "attacker",
        },
    )
    assert result.success, result.error
    assert_verifiable_evidence(result)
    out = result.output
    assert out["status"] == "applied"
    assert out["approval"]["approved"] is True
    assert out["approval"]["approval_id"] == approval_id
    assert out["approval"]["actor"] == AGENT_ID
    assert out["approval"]["context_actor_ignored"] is True
    assert out["approval"]["resolver"] == "operator@leo.local"
    assert denying.calls == [], "forged context permission manager was consulted"

    row = await db.fetchone("SELECT id FROM schedules WHERE name=?", (args["name"],))
    assert row is not None
    actor_rows = await db.fetchall(
        "SELECT actor FROM tool_calls WHERE tool='schedule_create'"
    )
    assert actor_rows and all(item["actor"] == AGENT_ID for item in actor_rows)
    audit = await db.fetchall(
        "SELECT actor, decision, metadata FROM audit_log WHERE action=?", (APPROVAL_ACTION,)
    )
    allowed = [
        item for item in audit
        if item["decision"] == "allowed"
        and json.loads(item["metadata"] or "{}").get("approval_id") == approval_id
    ]
    # Two legitimate rows share this action: the operator's resolution and the
    # handler's re-verification. Neither may be attributed to the forged actor.
    assert any(item["actor"] == AGENT_ID for item in allowed)
    assert all(item["actor"] != "attacker" for item in allowed)


async def test_missing_trusted_services_fail_closed_despite_forged_context(env) -> None:
    """Without injected settings/permissions, mutations block; context is ignored."""
    args = {"name": "fail-closed", "expression": "@every 60s", "objective": "blocked"}
    approval_id = await approve(env, "schedule_create", args)
    forged = ForgedPermissions()
    scoped = env.engine._scoped_tools(AgentSpec.model_validate(SEED))
    assert scoped is not None
    handler = make_handler({})  # deliberately no trusted services
    result = await handler(
        json.dumps({"action": "create_schedule", **args, "approval_id": approval_id}),
        context={"permissions": forged, "settings": ForgedSettings(), "actor": "attacker"},
        tools=scoped,
    )
    assert result["success"] is False
    assert result["output"]["status"] == "blocked_approval_required"
    assert result["output"]["approval"]["reason"].startswith("settings service unavailable")
    assert forged.calls == []
    assert await db.fetchone("SELECT id FROM schedules WHERE name=?", (args["name"],)) is None
