"""Planner specialist: seed shape, persistence, tool grants and real delegation."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from leo import db
from leo.agents import AgentEngine, AgentRegistry
from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.specs.planner import MAX_NODES, SEED, make_handler, node_evidence
from leo.config import Settings
from leo.contracts import AgentSpec, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, register_builtins
from leo.tools.operator_tools import register_operator_tools

# Every toolset module wired by the composition root (builtins and the operator
# tools use their dedicated registration helpers).
_TOOLSET_MODULES = (
    "agent_tools", "approval_tools", "automation_tools", "data_tools",
    "event_tools", "file_tools", "git_tools", "integration_tools",
    "memory_tools", "skill_tools", "task_tools",
)


def _live_registry() -> ToolRegistry:
    registry = ToolRegistry()
    settings = Settings()
    events = EventBus()
    register_builtins(registry)
    register_operator_tools(registry, settings, PermissionManagerImpl(settings, events))
    for module_name in _TOOLSET_MODULES:
        module = importlib.import_module(f"leo.tools.{module_name}")
        module.register(registry)
    return registry


def _evidence(claim: str) -> list[dict[str, Any]]:
    """Delegated nodes emit digest-backed artifacts, never bare assertions."""
    return node_evidence(claim)


def test_planner_seed_shape():
    assert set(SEED) == set(SPECIALIST_SPECS[0])
    assert SEED["id"] == "planner"
    assert SEED["id"] not in {row["id"] for row in SPECIALIST_SPECS}
    spec = AgentSpec.model_validate(SEED)
    assert spec.id == "planner" and spec.name and spec.purpose
    assert spec.capabilities and spec.tool_permissions and spec.model_preferences
    assert spec.context_budget > 0 and spec.task_limit > 0
    assert spec.enabled is True
    assert spec.metadata["delegation_mode"] == "explicit_scope"
    assert int(spec.metadata["max_nodes"]) == MAX_NODES
    assert "planner" not in spec.metadata["targets"]
    assert callable(make_handler)


async def test_planner_persists_and_reloads_through_agent_registry():
    registry = await AgentRegistry().initialize()
    stored = await registry.register(SEED)

    row = await db.fetchone("SELECT * FROM agents WHERE id='planner'")
    assert row is not None
    assert row["name"] == stored.name
    assert row["tool_permissions"] and "agents" in row["tool_permissions"]

    reloaded = AgentRegistry()
    await reloaded.initialize()
    copy = await reloaded.get("planner")
    assert copy is not None
    assert (copy.name, copy.purpose, copy.capabilities) == (
        stored.name, stored.purpose, stored.capabilities
    )
    assert (copy.tool_permissions, copy.model_preferences) == (
        stored.tool_permissions, stored.model_preferences
    )
    assert (copy.context_budget, copy.task_limit, copy.metadata) == (
        stored.context_budget, stored.task_limit, stored.metadata
    )

    assert await reloaded.unregister("planner")
    third = AgentRegistry()
    await third.refresh()
    assert await third.get("planner") is None


async def test_planner_tool_grants_are_real_and_executable():
    tool_registry = _live_registry()
    live = {spec.name for spec in tool_registry.list_specs()}
    categories = {spec.category for spec in tool_registry.list_specs()}
    assert SEED["tool_permissions"]
    for grant in SEED["tool_permissions"]:
        assert grant in live or grant in categories, f"unknown tool grant: {grant}"
    assert {"agent_delegate", "task_create", "calculator"} <= live
    assert {"agents", "tasks", "utility"} <= categories

    settings = Settings()
    events = EventBus()
    executor = ToolExecutorImpl(tool_registry, PermissionManagerImpl(settings, events), events, settings)
    registry = await AgentRegistry().initialize()
    await registry.register(SEED)
    spec = await registry.get("planner")
    engine = AgentEngine(registry, tools=executor, settings=settings)
    scoped = engine._scoped_tools(spec)
    advertised = {s.name for s in scoped.specs()}
    assert {"agent_delegate", "task_create", "calculator"} <= advertised
    assert "terminal" not in advertised and "write_file" not in advertised

    result = await scoped.execute(ToolCall(name="calculator", arguments={"expression": "2+2"}, actor="test"))
    assert result.success and result.output == 4


async def test_planner_delegates_bounded_graph_with_explicit_scope():
    registry = await AgentRegistry().initialize()
    await registry.register(SEED)
    engine = AgentEngine(registry)
    seen: list[dict] = []

    def child(objective, context, **kwargs):
        seen.append({"objective": objective, "context": context})
        return {"output": f"completed {objective}", "evidence": _evidence(f"completed {objective}")}

    engine.set_handler("research", child)
    engine.set_handler("qa", child)
    services = {"agent_registry": registry, "agent_engine": engine}
    engine.set_handler("planner", make_handler(services))

    objective = "Research the topic and verify the findings"
    result = await engine.delegate(objective, agent_id="planner")

    assert result.success
    checks = result.metadata["verification"]["checks"]
    assert checks["evidence_valid"] is True
    assert checks["evidence_verifiable"] is True
    assert checks["artifact_digests_valid"] is True
    assert checks["evidence_binds_output"] is True
    assert all(item["source"] == "artifact" and item["digest"] for item in result.evidence)
    nodes = result.output["graph"]["nodes"]
    edges = result.output["graph"]["edges"]
    assert [node["status"] for node in nodes] == ["completed", "completed"]
    assert [node["agent_id"] for node in nodes] == ["research", "qa"]
    assert [node["step"] for node in nodes] == ["Research the topic", "verify the findings"]
    assert all(node["task_id"] for node in nodes)
    assert edges == [{"from": nodes[0]["task_id"], "to": nodes[1]["task_id"]}]

    assert len(seen) == 2
    for index, entry in enumerate(seen):
        scope = entry["context"]["scope"]
        assert scope["explicit"] is True
        assert scope["node"] == index
        assert scope["target"] == nodes[index]["agent_id"]
        assert scope["parent_objective"] == objective
        assert entry["context"]["agent_id"] != "planner"  # never self-delegates
        assert entry["context"]["depth"] == 1
    assert len(seen[1]["context"]["dependencies"]) == 1
    assert seen[1]["context"]["dependencies"][0]["task_id"] == nodes[0]["task_id"]

    assert len(await registry.history("planner")) == 1
    assert len(await registry.history("research")) == 1
    assert len(await registry.history("qa")) == 1
    tasks = await db.fetchall("SELECT owner_agent FROM tasks")
    assert {row["owner_agent"] for row in tasks} >= {"planner", "research", "qa", "orchestrator"}


async def test_planner_enforces_node_bound():
    registry = await AgentRegistry().initialize()
    await registry.register(SEED)
    engine = AgentEngine(registry)

    def child(objective, context, **kwargs):
        return {"output": objective, "evidence": _evidence(objective)}

    for agent_id in ("general", "research", "qa", "documentation", "coding"):
        engine.set_handler(agent_id, child)
    engine.set_handler("planner", make_handler({"agent_registry": registry, "agent_engine": engine}))

    objective = "; ".join(f"step {index}" for index in range(MAX_NODES + 3))
    result = await engine.delegate(objective, agent_id="planner")

    assert result.success
    assert len(result.output["graph"]["nodes"]) == MAX_NODES
    assert result.output["omitted"] == 3
    assert result.output["completed"] == MAX_NODES
    assert result.output["blocked"] == 0
    assert len(await registry.history("general")) == MAX_NODES


async def test_planner_reports_missing_services_without_fake_success():
    registry = await AgentRegistry().initialize()
    await registry.register(SEED)
    engine = AgentEngine(registry, handlers={"planner": make_handler({})})
    result = await engine.delegate("Research something", agent_id="planner", verify=False)
    assert not result.success
    assert "agent_engine" in (result.error or "")
