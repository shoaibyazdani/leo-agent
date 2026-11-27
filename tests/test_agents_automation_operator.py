"""Automation-operator specialist: canonical schedule consent plus exact gates.

Create/update converge on the same ScheduleService proposal/card as a direct
model call, so there is exactly one consent gate. Delete, manual trigger,
cancel and task enqueue keep the specialist's exact durable
``automation:mutate`` gate. Queued is never reported as executed or completed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.agents import AgentEngine, AgentRegistry
from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.verification import digest as evidence_digest
from leo.agents.specs.automation_operator import (
    _ACTIONS,
    AGENT_ID,
    APPROVAL_ACTION,
    SEED,
    TOOL_GRANTS,
    approval_target,
    make_handler,
    tool_arguments,
)
from leo.config import Settings
from leo.contracts import AgentSpec, RiskLevel, ToolCall, TurnContext, turn_context
from leo.events import EventBus
from leo.schedules.service import ScheduleService
from leo.schedules.store import ScheduleStore
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
    store = ScheduleStore()
    service = ScheduleService(
        store=store,
        scheduler=scheduler,
        engine=tasks,
        permissions=permissions,
        events=events,
        settings=settings,
    )
    webhook = WebhookHandler(tasks, events)
    services = {
        "tasks": tasks,
        "scheduler": scheduler,
        "schedule_service": service,
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
        executor=executor, agents=agents, engine=engine, store=store, service=service,
    )


async def run(env, *, context=None, **request):
    """Delegate a JSON automation request through the real AgentEngine."""
    return await env.engine.delegate(
        json.dumps(request), agent_id=AGENT_ID, context=context
    )


def _tui_turn() -> TurnContext:
    return TurnContext(
        principal="chat:web",
        channel="tui",
        conversation_id="conv-a",
        surface="tui",
        origin="tui",
        source="chat",
    )


def _operator_turn() -> TurnContext:
    return TurnContext(
        principal="operator:cli", channel="cli", surface="cli", source="operator"
    )


async def _service_create(env, name: str = "hourly", prompt: str = "check queue"):
    return await env.service.create(
        {"name": name, "schedule": "1m", "prompt": prompt},
        actor="operator:cli",
        turn=_operator_turn(),
    )


async def _resolve_card(env, out: dict, *, choice: str = "once"):
    card = out["card"]
    return await env.permissions.resolve_schedule_card(
        card["approval_id"],
        choice=choice,
        resolver="operator:local",
        confirm_proposal_digest=card["proposal_digest"],
        commit=env.service.commit,
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


# --------------------------------------------------------------------------- shape / grants


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
    # Intentional roster update: explicit pause/resume join the specialist.
    assert len(grants) == 15
    assert grants <= set(specs), f"unknown grants: {grants - set(specs)}"
    categories = {spec.category for spec in specs.values()}
    assert not grants & categories, "category grants would widen the tool scope"
    assert {
        "schedule_create", "schedule_get", "schedule_update",
        "schedule_pause", "schedule_resume",
        "schedule_delete", "schedule_run_now", "schedule_cancel",
    } <= grants
    assert "task_cancel" not in grants

    assert {name for name in grants if name.startswith("webhook_")} == set(
        webhook_tools.TOOL_NAMES
    ) == {"webhook_routes", "webhook_verify", "webhook_deliveries"}
    assert "webhook_sign" not in specs and "webhook_sign" not in grants
    assert "approval_resolve" not in specs and "approval_resolve" not in grants

    for name in grants:
        assert ToolExecutorImpl._in_scope(env.registry.get(name), frozenset(grants))
    for name in ("task_cancel", "task_list"):
        assert name in specs
        assert not ToolExecutorImpl._in_scope(env.registry.get(name), frozenset(grants))

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


# --------------------------------------------------------------------------- canonical consent


async def test_create_uses_one_canonical_card_and_no_double_gate(env) -> None:
    with turn_context(_tui_turn()):
        result = await run(
            env,
            action="create_schedule",
            name="hourly",
            schedule="1m",
            prompt="check queue",
        )
    assert not result.success, "a waiting card is not an applied creation"
    out = result.output
    assert out["status"] == "approval_required"
    assert out["applied"] is False
    assert out["approval_id"] and out["proposal_id"]
    assert out["card"]["kind"] == "schedule_create"
    assert out["card"]["choices"] == ["once", "session", "always", "deny"]
    assert out["card"]["requested_by"] == AGENT_ID
    assert out["approval"]["required"] is False, "the old specialist gate must not double-card"

    # Exactly one card, and it is the schedule card — no automation:mutate row.
    rows = await db.fetchall("SELECT id, action, kind FROM approvals")
    assert len(rows) == 1
    assert rows[0]["kind"] == "schedule_create"
    assert await db.fetchall(
        "SELECT id FROM approvals WHERE action=?", (APPROVAL_ACTION,)
    ) == []
    assert await db.fetchone("SELECT id FROM schedules WHERE name='hourly'") is None

    resolved = await _resolve_card(env, out)
    assert resolved.status == "approved" and resolved.schedule_id
    row = await db.fetchone("SELECT * FROM schedules WHERE name='hourly'")
    assert row is not None and row["expression"] == "60"
    assert row["created_by"] == AGENT_ID
    assert_verifiable_evidence(result)


async def test_update_uses_one_canonical_card_and_no_double_gate(env) -> None:
    created = await _service_create(env, name="hourly", prompt="first")
    assert created["status"] == "created"
    with turn_context(_tui_turn()):
        result = await run(
            env, action="update_schedule", name="hourly", prompt="second",
        )
    assert not result.success
    out = result.output
    assert out["status"] == "approval_required"
    assert out["card"]["kind"] == "schedule_edit"
    assert out["card"]["choices"] == ["once", "deny"]
    assert "prompt changed" in out["card"]["body"]
    assert await db.fetchall(
        "SELECT id FROM approvals WHERE action=?", (APPROVAL_ACTION,)
    ) == []

    resolved = await _resolve_card(env, out)
    assert resolved.status == "approved"
    row = await db.fetchone("SELECT * FROM schedules WHERE name='hourly'")
    assert row["config_revision"] == 2
    assert json.loads(row["payload"])["objective"] == "second"


async def test_forged_context_cannot_change_actor_or_supply_schedule_authority(env) -> None:
    forged = ForgedPermissions()
    with turn_context(_tui_turn()):
        result = await run(
            env,
            action="create_schedule",
            name="forged",
            schedule="1m",
            prompt="no mutation",
            context={
                "permissions": forged,
                "settings": ForgedSettings(),
                "services": {"permissions": forged, "settings": ForgedSettings()},
                "actor": "attacker",
                "approved": True,
            },
        )
    out = result.output
    assert out["status"] == "approval_required"
    assert out["approval"]["actor"] == AGENT_ID
    assert out["approval"]["context_actor_ignored"] is True
    assert forged.calls == [], "forged context permission manager was consulted"
    assert await db.fetchall(
        "SELECT id FROM approvals WHERE action=?", (APPROVAL_ACTION,)
    ) == []
    assert await db.fetchone("SELECT id FROM schedules WHERE name='forged'") is None


# --------------------------------------------------------------------------- preserved gates


async def test_delete_keeps_the_exact_durable_mutation_gate(env) -> None:
    created = await _service_create(env, name="hourly")
    assert created["status"] == "created"
    args = {"name": "hourly"}
    blocked = await run(env, action="delete_schedule", **args)
    assert not blocked.success
    assert blocked.output["status"] == "blocked_approval_required"
    assert blocked.output["applied"] is False
    assert await db.fetchone("SELECT id FROM schedules WHERE name='hourly'") is not None

    approval_id = await approve(env, "schedule_delete", args)
    deleted = await run(env, action="delete_schedule", **args, approval_id=approval_id)
    assert deleted.success, deleted.error
    assert deleted.output["status"] == "applied"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='hourly'") is None
    assert_verifiable_evidence(deleted)
    audit = await db.fetchall(
        "SELECT decision, metadata FROM audit_log WHERE action=?", (APPROVAL_ACTION,)
    )
    assert any(
        item["decision"] == "allowed"
        and json.loads(item["metadata"] or "{}").get("approval_id") == approval_id
        for item in audit
    )


async def test_trigger_keeps_the_exact_gate_and_reports_queued(env) -> None:
    await _service_create(env, name="nightly")
    args = {"name": "nightly"}
    blocked = await run(env, action="trigger_schedule", **args)
    assert blocked.output["status"] == "blocked_approval_required"

    approval_id = await approve(env, "schedule_run_now", args)
    result = await run(env, action="trigger_schedule", **args, approval_id=approval_id)
    assert result.success, result.error
    out = result.output
    assert out["status"] == "queued"
    assert out["triggered"] is True and out["manual_run"] is True
    assert out["executed"] is False and out["completed"] is False
    assert out["approval"]["approval_id"] == approval_id
    row = await db.fetchone("SELECT status, metadata FROM tasks WHERE id=?", (out["task_id"],))
    assert row is not None and row["status"] == "created"
    assert json.loads(row["metadata"])["manual_run"] is True
    assert await env.events.history("task.completed") == []
    assert_verifiable_evidence(result)


async def test_specialist_live_approval_is_single_use(env) -> None:
    """N2: one exact-action card cannot authorize unlimited run/cancel calls."""
    await _service_create(env, name="once-only")
    args = {"name": "once-only"}
    approval_id = await approve(env, "schedule_run_now", args)

    first = await run(env, action="trigger_schedule", **args, approval_id=approval_id)
    assert first.success, first.error
    assert first.output["status"] == "queued"

    second = await run(env, action="trigger_schedule", **args, approval_id=approval_id)
    assert second.success is False
    assert second.output["status"] == "blocked_approval_required"
    assert "already used" in second.output["approval"]["reason"]
    tasks = await db.fetchall(
        "SELECT id FROM tasks WHERE json_extract(metadata,'$.manual_run')=1"
    )
    assert len(tasks) == 1


async def test_stale_live_lease_is_reclaimable_by_the_specialist(env) -> None:
    """N3(a): a crashed worker's expired lease cannot lock a card forever."""
    await _service_create(env, name="stale-lease")
    args = {"name": "stale-lease"}
    approval_id = await approve(env, "schedule_run_now", args)
    stale_at = (datetime.now(timezone.utc) - timedelta(seconds=301)).isoformat()
    await db.execute(
        "UPDATE approvals SET metadata=json_set(COALESCE(metadata,'{}'),"
        "'$.live_authorization_claim', json_object('token',?,'at',?)) WHERE id=?",
        ("stale-lease-token", stale_at, approval_id),
    )

    result = await run(env, action="trigger_schedule", **args, approval_id=approval_id)
    assert result.success, result.error
    assert result.output["status"] == "queued"
    stored = await db.fetchone(
        "SELECT json_extract(metadata,'$.live_authorization_used_at') AS used_at "
        "FROM approvals WHERE id=?", (approval_id,),
    )
    assert stored["used_at"]


async def test_specialist_noop_cancel_does_not_burn_its_approval(env) -> None:
    """N3: a no-op cancel releases the lease instead of consuming the card."""
    await _service_create(env, name="cancel-now")
    args = {"name": "cancel-now"}
    approval_id = await approve(env, "schedule_cancel", args)

    noop = await run(env, action="cancel_schedule", **args, approval_id=approval_id)
    assert noop.success, noop.error
    assert noop.output["status"] == "no_live_run"

    await env.service.run_now(name="cancel-now", actor="operator:cli", turn=_operator_turn())
    cancelled = await run(env, action="cancel_schedule", **args, approval_id=approval_id)
    assert cancelled.success, cancelled.error
    assert cancelled.output["status"] == "cancelled"

    again = await run(env, action="cancel_schedule", **args, approval_id=approval_id)
    assert again.success is False
    assert again.output["status"] == "blocked_approval_required"


async def test_cancel_declares_and_enforces_the_same_exact_gate(env) -> None:
    """The grant table and the specialist's policy both gate cancellation."""
    assert _ACTIONS["cancel_schedule"] == ("schedule_cancel", True)
    await _service_create(env, name="cancel-me")
    args = {"name": "cancel-me"}

    blocked = await run(env, action="cancel_schedule", **args)
    assert not blocked.success
    assert blocked.output["status"] == "blocked_approval_required"
    with turn_context(_operator_turn()):
        blocked_operator = await run(env, action="cancel_schedule", **args)
    assert not blocked_operator.success
    assert blocked_operator.output["status"] == "blocked_approval_required"

    wrong = await approve(env, "schedule_run_now", args)
    mismatched = await run(env, action="cancel_schedule", **args, approval_id=wrong)
    assert not mismatched.success
    assert mismatched.output["status"] == "blocked_approval_required"

    approval_id = await approve(env, "schedule_cancel", args)
    result = await run(env, action="cancel_schedule", **args, approval_id=approval_id)
    assert result.success, result.error
    assert result.output["approval"]["approval_id"] == approval_id
    assert result.output["status"] in ("cancelled", "no_live_run")
    assert_verifiable_evidence(result)


async def test_missing_trusted_services_fail_closed_for_gated_actions(env) -> None:
    await _service_create(env, name="fail-closed")
    args = {"name": "fail-closed"}
    approval_id = await approve(env, "schedule_delete", args)
    scoped = env.engine._scoped_tools(AgentSpec.model_validate(SEED))
    assert scoped is not None
    handler = make_handler({})  # deliberately no trusted services
    result = await handler(
        json.dumps({"action": "delete_schedule", **args, "approval_id": approval_id}),
        context={"permissions": ForgedPermissions(), "settings": ForgedSettings(), "actor": "attacker"},
        tools=scoped,
    )
    assert result["success"] is False
    assert result["output"]["status"] == "blocked_approval_required"
    assert result["output"]["approval"]["reason"].startswith("settings service unavailable")
    assert await db.fetchone("SELECT id FROM schedules WHERE name='fail-closed'") is not None


# --------------------------------------------------------------------------- tasks / webhooks


async def test_queued_task_becomes_completed_only_when_engine_records_it(env) -> None:
    args = {"objective": "durable work"}
    mutation_args = tool_arguments("enqueue_task", args)
    queued = await run(
        env, action="enqueue_task", **args,
        approval_id=await approve(env, "task_create", mutation_args),
    )
    assert queued.success, queued.error
    assert queued.output["status"] == "queued"
    assert queued.output["executed"] is False
    assert queued.output["completed"] is False
    task_id = queued.output["task_id"]

    before_status = await run(env, action="task_status", task_id=task_id)
    assert before_status.success and before_status.output["completed"] is False
    before_result = await run(env, action="task_result", task_id=task_id)
    assert before_result.success and before_result.output["completed"] is False

    await env.tasks.complete(
        task_id, {"ok": True}, [{"kind": "test", "detail": "evidence-backed"}]
    )
    after_status = await run(env, action="task_status", task_id=task_id)
    assert after_status.success and after_status.output["completed"] is True
    after_result = await run(env, action="task_result", task_id=task_id)
    assert after_result.success and after_result.output["completed"] is True
    assert after_result.output["result"] == {"ok": True}


async def test_webhook_delivery_reporting_is_truthful(env) -> None:
    routes = await run(env, action="webhook_routes")
    assert routes.success, routes.error
    assert_verifiable_evidence(routes)
    assert SECRET not in json.dumps(routes.output)

    signed = WebhookHandler.sign_v1(BODY, SECRET, delivery_id="d-auto-1")
    accepted = await env.webhook.handle({"body": BODY, "headers": signed})
    assert accepted["status"] == "accepted"

    verified = await run(
        env, action="verify_webhook", body=BODY, headers=signed, delivery_id="d-auto-1"
    )
    assert verified.success, verified.error
    assert verified.output["verified"] is True
    assert verified.output["delivery"]["executed"] is False

    deliveries = await run(env, action="delivery_status", delivery_id="d-auto-1")
    assert deliveries.success, deliveries.error
    assert deliveries.output["delivery"]["outcome"] == "accepted"
    assert deliveries.output["delivery"]["executed"] is False
    assert deliveries.output["deliveries"][0]["status"] == "accepted"
    tampered = await run(
        env, action="verify_webhook", body={"objective": "changed"},
        headers=signed, delivery_id="d-auto-1",
    )
    assert not tampered.success
    assert tampered.output["status"] == "failed"
    assert "verified" not in tampered.output


async def test_every_declared_grant_has_a_real_delegated_execution(env) -> None:
    """No inert grants: each name is dispatched through the scoped façade."""
    with turn_context(_tui_turn()):
        created = await run(
            env, action="create_schedule", name="grant-check",
            schedule="1m", prompt="check",
        )
    assert created.output["status"] == "approval_required"
    await _resolve_card(env, created.output)

    with turn_context(_tui_turn()):
        updated = await run(
            env, action="update_schedule", name="grant-check", prompt="check again",
        )
    await _resolve_card(env, updated.output)

    with turn_context(_tui_turn()):
        paused = await run(
            env, action="pause_schedule", name="grant-check",
            paused_reason="grant exercise",
        )
    assert paused.output["status"] == "approval_required"
    await _resolve_card(env, paused.output)

    with turn_context(_tui_turn()):
        resumed = await run(env, action="resume_schedule", name="grant-check")
    assert resumed.output["status"] == "approval_required"
    await _resolve_card(env, resumed.output)

    await run(env, action="get_schedule", name="grant-check")
    await run(env, action="list_schedules")
    trigger_args = {"name": "grant-check"}
    await run(
        env, action="trigger_schedule", **trigger_args,
        approval_id=await approve(env, "schedule_run_now", trigger_args),
    )
    # Stopping a live run keeps the exact durable mutation gate even when the
    # caller is an operator turn: declared grant and enforced policy agree.
    cancel_args = {"name": "grant-check"}
    with turn_context(_operator_turn()):
        await run(
            env, action="cancel_schedule", **cancel_args,
            approval_id=await approve(env, "schedule_cancel", cancel_args),
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
    await run(env, action="task_status", task_id=task_result.output["task_id"])
    await run(env, action="task_result", task_id=task_result.output["task_id"])
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
