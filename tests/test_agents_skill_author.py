"""Tests for the skill-author specialist: spec shape, live grants and a real
draft-then-register path against the isolated SQLite database.
"""

from __future__ import annotations

import pytest

from leo import db
from leo.agents.engine import AgentEngine
from leo.agents.registry import SPECIALIST_SPECS, AgentRegistry
from leo.agents.specs.skill_author import SEED, draft_manifest, make_handler
from leo.config import Settings
from leo.events import EventBus
from leo.skills.registry import SkillRegistry
from leo.tasks.engine import DurableTaskEngine
from leo.tools.registry import ToolRegistry

EXPECTED_KEYS = set(SPECIALIST_SPECS[0])
TEST_FILE = "tests/test_agents_skill_author.py"


def _live_registry() -> ToolRegistry:
    """Build a real ToolRegistry from the actual registered toolsets."""
    from leo.tools.builtins import register_builtins
    from leo.tools.memory_tools import register as register_memory_tools
    from leo.tools.skill_tools import register as register_skill_tools
    from leo.tools.task_tools import register as register_task_tools

    registry = ToolRegistry()
    register_builtins(registry)
    register_skill_tools(registry)
    register_task_tools(registry)
    register_memory_tools(registry)
    return registry


async def _verified_task(
    tasks: DurableTaskEngine, objective: str = "Draft a documented retry helper"
):
    task = await tasks.create(objective)
    await tasks.complete(
        task.id,
        {"procedure": ["inspect", "implement", "verify"]},
        [{"source": "test-fixture", "claim": "outcome verified locally"}],
    )
    return task


def test_seed_shape_matches_specialist_contract():
    assert set(SEED) == EXPECTED_KEYS
    assert SEED["id"] == "skill-author"
    assert SEED["name"] and SEED["purpose"]
    assert SEED["capabilities"] and all(isinstance(c, str) and c for c in SEED["capabilities"])
    assert SEED["model_preferences"] and all(
        isinstance(m, str) and m for m in SEED["model_preferences"]
    )
    assert isinstance(SEED["context_budget"], int) and SEED["context_budget"] > 0
    assert isinstance(SEED["task_limit"], int) and SEED["task_limit"] > 0
    assert isinstance(SEED["metadata"], dict) and SEED["metadata"]
    assert callable(make_handler)


def test_tool_grants_name_live_categories_or_tools():
    specs = _live_registry().list_specs()
    known = {spec.name for spec in specs} | {spec.category for spec in specs}
    assert SEED["tool_permissions"]
    assert all(grant in known for grant in SEED["tool_permissions"])
    # The check itself is real: a fabricated grant is not in the live registry.
    assert "not_a_real_tool" not in known


async def test_spec_registers_persists_and_reloads():
    registry = await AgentRegistry().initialize()
    registered = await registry.register(dict(SEED))
    assert registered.id == "skill-author"

    reloaded = await AgentRegistry().initialize()
    spec = await reloaded.get("skill-author")
    assert spec is not None
    assert spec.name == SEED["name"]
    assert spec.purpose == SEED["purpose"]
    assert spec.capabilities == SEED["capabilities"]
    assert spec.tool_permissions == SEED["tool_permissions"]
    assert spec.model_preferences == SEED["model_preferences"]

    rows = await db.fetchall("SELECT * FROM agents WHERE id=?", ("skill-author",))
    assert len(rows) == 1
    assert rows[0]["name"] == SEED["name"]


async def test_draft_then_register_against_temp_db():
    settings = Settings()
    events = EventBus()
    tasks = DurableTaskEngine(settings, events)
    task = await _verified_task(tasks)
    skills = SkillRegistry(events=events)

    handler = make_handler({"tasks": tasks, "skills": skills})
    result = await handler(
        objective=f"Author a skill from task {task.id}",
        context={"source_task_id": task.id},
        task_id=task.id,
    )

    assert result["success"] is True
    assert result["output"]["registered"] is True
    assert result["output"]["action"] == "registered"
    assert result["output"]["source_task_id"] == task.id
    name = result["output"]["skill"]

    record = await skills.get(name)
    assert record is not None
    assert record.name == name
    assert record.source == "native"
    assert record.documentation and "Verified source" in record.documentation
    assert record.tests == [TEST_FILE]
    assert record.permissions == ["utility"]
    validation = skills.validate(record)
    assert validation.valid is True, validation.errors

    rows = await db.fetchall("SELECT * FROM skills WHERE name=?", (name,))
    assert len(rows) == 1
    assert rows[0]["status"] == "registered"
    history = await db.fetchall(
        "SELECT * FROM skill_history WHERE skill_name=? ORDER BY changed_at", (name,)
    )
    assert any(row["action"] == "registered" for row in history)
    assert result["evidence"] and result["evidence"][0]["task_id"] == task.id


async def test_unverified_task_is_refused_without_registration():
    settings = Settings()
    events = EventBus()
    tasks = DurableTaskEngine(settings, events)
    pending = await tasks.create("Unfinished work")
    skills = SkillRegistry(events=events)
    baseline = len(await skills.list())

    handler = make_handler({"tasks": tasks, "skills": skills})
    with pytest.raises(ValueError, match="verified"):
        await handler(objective="Draft from unfinished work", context={"source_task_id": pending.id})

    assert len(await skills.list()) == baseline


async def test_delegation_path_registers_skill():
    settings = Settings()
    events = EventBus()
    tasks = DurableTaskEngine(settings, events)
    task = await _verified_task(tasks, "Delegate a documented helper")
    skills = SkillRegistry(events=events)

    registry = await AgentRegistry().initialize()
    await registry.register(dict(SEED))
    engine = AgentEngine(registry, events=events, settings=settings)
    engine.set_handler("skill-author", make_handler({"tasks": tasks, "skills": skills}))

    result = await engine.delegate(
        f"Author a skill from task {task.id}",
        agent_id="skill-author",
        context={"source_task_id": task.id},
    )

    assert result.success is True
    assert result.output["registered"] is True
    assert await skills.get(result.output["skill"]) is not None


def test_draft_is_pure_and_valid_before_registration():
    task = type("Task", (), {
        "id": "task_abc123def456",
        "objective": "Normalize CSV columns",
        "status": "completed",
        "evidence": [{"source": "fixture", "claim": "verified"}],
        "result": {"columns": 3},
    })()
    manifest = draft_manifest(task)
    assert manifest.name.startswith("authored_")
    assert manifest.permissions == ["utility"]
    assert manifest.documentation and manifest.tests == [TEST_FILE]
    assert SkillRegistry(events=EventBus(), auto_seed=False).validate(manifest).valid is True
