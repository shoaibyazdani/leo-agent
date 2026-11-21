"""Agent toolset tests: real delegation, durable records and denial paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import db
from leo.agents import AgentEngine, AgentRegistry
from leo.config import Settings
from leo.contracts import Decision, RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tasks.engine import DurableTaskEngine
from leo.tools.agent_tools import TOOL_NAMES, register
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry


class _ServicesExecutor(ToolExecutorImpl):
    """ToolExecutorImpl with the composition-root service injection wired in."""

    def __init__(self, *args: object, services: object = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.injected_services = services

    def _context(self, call: ToolCall, task_key: str, dry_run: bool) -> dict:
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.injected_services
        return ctx


class _DenyPermissions(PermissionManagerImpl):
    async def authorize(self, actor, action, resource=None, risk=RiskLevel.LOW, **context):
        return Decision(allowed=False, reason="policy denies agent delegation", risk=risk)


@pytest.fixture
async def env():
    registry = await AgentRegistry().initialize()
    settings = Settings()
    events = EventBus()
    tasks = DurableTaskEngine(settings, events)
    engine = AgentEngine(registry, tasks, None, None, events, settings)
    return SimpleNamespace(
        registry=registry, engine=engine, tasks=tasks, settings=settings, events=events,
        services={"agent_registry": registry, "agent_engine": engine, "tasks": tasks},
        seen=[],
    )


def _install_worker(env, objective_text: str = "Inspect the fixture"):
    def worker(objective, context, **kwargs):
        env.seen.append(objective)
        return {
            "output": {"answer": objective},
            "evidence": [{"source": "fake-worker", "claim": objective}],
        }

    env.engine.set_handler("general", worker)
    return worker


def _executor(env, *, services=None, permissions=None):
    registry = ToolRegistry()
    names = register(registry)
    assert names == list(TOOL_NAMES)
    permissions = permissions or PermissionManagerImpl(env.settings, env.events)
    executor = _ServicesExecutor(
        registry, permissions, env.events, env.settings,
        services=env.services if services is None else services,
    )
    return executor, registry


async def _call(executor, name: str, arguments: dict, **kwargs):
    call = ToolCall(name=name, arguments=arguments, actor="tester", correlation_id="corr-agent-test", **kwargs)
    return await executor.execute(call)


async def _delegate(env, executor, objective: str = "Inspect the fixture"):
    # These tests cover durable task/run persistence, not evidence verification.
    outcome = await _call(executor, "agent_delegate", {
        "objective": objective, "agent_id": "general", "verify": False,
    })
    assert outcome.success, outcome.error
    return outcome.output


async def _tasks():
    return await db.fetchall("SELECT * FROM tasks")


def test_register_defines_typed_specs():
    registry = ToolRegistry()
    assert register(registry) == list(TOOL_NAMES)
    specs = {spec.name: spec for spec in registry.list_specs()}
    assert all(spec.category == "agents" for spec in specs.values())
    assert all(spec.permissions for spec in specs.values())
    assert all(spec.resource_field == "" for spec in specs.values())
    assert all(spec.input_schema["type"] == "object" for spec in specs.values())
    assert specs["agent_delegate"].risk is RiskLevel.MEDIUM
    assert specs["agent_delegate"].permissions == ["agent:delegate"]
    assert all(specs[name].risk is RiskLevel.LOW for name in ("agent_list", "agent_status", "agent_result"))
    assert registry.validate("agent_delegate", {"objective": "Check"}) == []
    assert registry.validate("agent_delegate", {"agent_id": "qa"})


def test_delegate_context_schema_rejects_reserved_keys():
    """Approval/service/tool keys must never enter agent delegation context."""
    registry = ToolRegistry()
    register(registry)
    context_schema = registry.get("agent_delegate").spec.input_schema["properties"]["context"]
    assert context_schema["additionalProperties"] is False
    assert "correlation_id" in context_schema["properties"]
    for reserved in (
        "apply", "confirm", "approved_by", "approval_id", "apply_expiry",
        "services", "permissions", "settings", "tools",
    ):
        assert reserved not in context_schema["properties"]
        assert registry.validate("agent_delegate", {
            "objective": "Check", "context": {reserved: "forged"},
        })
    assert registry.validate("agent_delegate", {
        "objective": "Check", "context": {"correlation_id": "safe"},
    }) == []


async def test_agent_list_reads_persisted_registry(env):
    executor, _ = _executor(env)
    outcome = await _call(executor, "agent_list", {"enabled_only": True})
    assert outcome.success, outcome.error
    assert outcome.output["count"] >= 20
    assert all(agent["enabled"] for agent in outcome.output["agents"])
    assert "general" in {agent["id"] for agent in outcome.output["agents"]}

    await env.registry.register({
        "id": "temp-list-agent", "name": "Temp List Agent", "purpose": "Transient listing check",
    })
    await env.registry.enable("temp-list-agent", False)
    again = await _call(executor, "agent_list", {"enabled_only": False})
    assert again.output["count"] == outcome.output["count"] + 1  # temporary agent
    disabled = next(a for a in again.output["agents"] if a["id"] == "temp-list-agent")
    assert disabled["enabled"] is False


async def test_agent_delegate_creates_real_task_and_run(env):
    _install_worker(env)
    executor, _ = _executor(env)
    objective = "Inspect the fixture"
    payload = await _delegate(env, executor, objective)

    assert payload["success"] is True
    assert payload["status"] == "completed"
    assert payload["output"] == {"answer": objective}
    assert payload["objective"] == objective
    assert payload["agent_id"] == "general"
    assert env.seen == [objective]

    tasks = await _tasks()
    task = next(row for row in tasks if row["id"] == payload["task_id"])
    assert task["objective"] == objective
    assert task["owner_agent"] == "general"
    assert task["status"] == "completed"
    assert task["evidence"]

    runs = await env.registry.history("general")
    assert len(runs) == 1
    assert runs[0]["task_id"] == payload["task_id"]
    assert runs[0]["status"] == "completed"
    assert payload["run_id"] == runs[0]["id"]
    assert payload["run_count"] == 1

    audit = await db.fetchall("SELECT tool, decision FROM tool_calls WHERE tool='agent_delegate'")
    assert audit and audit[0]["decision"] == "allowed"


async def test_agent_status_reads_durable_state(env):
    _install_worker(env)
    executor, _ = _executor(env)
    payload = await _delegate(env, executor)

    status = await _call(executor, "agent_status", {"task_id": payload["task_id"]})
    assert status.success, status.error
    assert status.output["status"] == "completed"
    assert status.output["agent_id"] == "general"
    assert status.output["objective"] == "Inspect the fixture"
    assert status.output["run_count"] == 1
    assert status.output["runs"][0]["id"] == payload["run_id"]

    # A durable change is visible immediately; status is not a cached reply.
    await db.execute("UPDATE tasks SET status='running' WHERE id=?", (payload["task_id"],))
    changed = await _call(executor, "agent_status", {"task_id": payload["task_id"]})
    assert changed.output["status"] == "running"


async def test_agent_result_reads_durable_result(env):
    _install_worker(env)
    executor, _ = _executor(env)
    payload = await _delegate(env, executor)

    outcome = await _call(executor, "agent_result", {"task_id": payload["task_id"]})
    assert outcome.success, outcome.error
    assert outcome.output["ready"] is True
    assert outcome.output["status"] == "completed"
    assert outcome.output["result"]["output"] == {"answer": "Inspect the fixture"}
    assert outcome.output["evidence"]
    assert outcome.output["error"] is None

    pending = await env.tasks.create("Never started")
    unfinished = await _call(executor, "agent_result", {"task_id": pending.id})
    assert unfinished.success, unfinished.error
    assert unfinished.output["ready"] is False
    assert unfinished.output["result"] is None


@pytest.mark.parametrize("name,arguments", [
    ("agent_list", {}),
    ("agent_delegate", {"objective": "Check"}),
    ("agent_status", {"task_id": "task_unknown"}),
    ("agent_result", {"task_id": "task_unknown"}),
])
async def test_tools_deny_missing_services(env, name, arguments):
    executor, _ = _executor(env, services={})
    outcome = await _call(executor, name, arguments)
    assert not outcome.success
    assert outcome.decision == "error"
    assert "missing required service 'agent_registry'" in outcome.error
    assert env.seen == []
    assert await _tasks() == []


async def test_delegate_denies_missing_engine(env):
    executor, _ = _executor(env, services={"agent_registry": env.registry})
    outcome = await _call(executor, "agent_delegate", {"objective": "Check"})
    assert not outcome.success
    assert "missing required service 'agent_engine'" in outcome.error
    assert await _tasks() == []


@pytest.mark.parametrize("name,arguments", [
    ("agent_status", {"task_id": "task_unknown"}),
    ("agent_result", {"task_id": "task_unknown"}),
])
async def test_status_and_result_deny_unknown_task(env, name, arguments):
    executor, _ = _executor(env)
    outcome = await _call(executor, name, arguments)
    assert not outcome.success
    assert "unknown task: task_unknown" in outcome.error


async def test_delegate_denies_unknown_or_disabled_agent(env):
    _install_worker(env)
    executor, _ = _executor(env)
    unknown = await _call(executor, "agent_delegate", {"objective": "Check", "agent_id": "no-such-agent"})
    assert not unknown.success
    assert "unknown agent: no-such-agent" in unknown.error

    await env.registry.enable("general", False)
    disabled = await _call(executor, "agent_delegate", {"objective": "Check", "agent_id": "general"})
    assert not disabled.success
    assert "agent is disabled: general" in disabled.error
    assert env.seen == []
    assert await _tasks() == []


@pytest.mark.parametrize("name,arguments,reason", [
    ("agent_delegate", {}, "objective"),
    ("agent_delegate", {"objective": "Check", "bogus": 1}, "unknown field"),
    ("agent_status", {"task_id": 7}, "expected str"),
    ("agent_list", {"enabled_only": "yes"}, "expected bool"),
])
async def test_schema_validation_prevents_execution(env, name, arguments, reason):
    _install_worker(env)
    executor, _ = _executor(env)
    outcome = await _call(executor, name, arguments)
    assert not outcome.success
    assert outcome.decision == "invalid_arguments"
    assert reason in outcome.error
    assert env.seen == []
    assert await _tasks() == []


async def test_out_of_scope_denial_prevents_delegation(env):
    _install_worker(env)
    executor, _ = _executor(env)
    outcome = await executor.execute(
        ToolCall(name="agent_delegate", arguments={"objective": "Check"}, actor="tester"),
        scope=frozenset({"utility"}),
    )
    assert not outcome.success
    assert outcome.decision == "out_of_scope"
    assert env.seen == []
    assert await _tasks() == []


async def test_permission_denial_prevents_delegation(env):
    _install_worker(env)
    executor, _ = _executor(env, permissions=_DenyPermissions(env.settings, env.events))
    outcome = await _call(executor, "agent_delegate", {"objective": "Check"})
    assert not outcome.success
    assert outcome.decision == "denied"
    assert "policy denies agent delegation" in outcome.error
    assert env.seen == []
    assert await _tasks() == []


async def test_registration_time_services_are_a_fallback(env):
    tool_registry = ToolRegistry()
    register(tool_registry, services=env.services)
    permissions = PermissionManagerImpl(env.settings, env.events)
    executor = _ServicesExecutor(
        tool_registry, permissions, env.events, env.settings, services={}
    )
    outcome = await _call(executor, "agent_list", {})
    assert outcome.success, outcome.error
    assert outcome.output["count"] >= 20
