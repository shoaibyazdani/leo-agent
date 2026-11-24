"""The /journey timeline: authored skills + memory notes, oldest first."""

from __future__ import annotations

from types import SimpleNamespace

from leo import db
from leo.commands.builtins import journey_command
from leo.commands.registry import CommandContext
from leo.contracts import MemoryRecord, SkillManifest
from leo.events import EventBus
from leo.memory.store import MemoryStore
from leo.skills.registry import SkillRegistry


def _ctx(app) -> CommandContext:
    return CommandContext(
        app=app, principal="user", surface="cli", channel="cli",
        argv=[], raw="/journey", registry=None, conversation_id="conv-journey",
    )


async def test_journey_lists_authored_skills_and_memories_oldest_first():
    events = EventBus()
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    await skills.ready()
    await skills.register(SkillManifest(
        name="journey-skill", description="Use when testing the journey.",
        category="test", documentation="# Journey\n\nA learned skill.\n",
        created_by="agent",
    ))
    await skills.register(SkillManifest(
        name="native-journey", description="Use when testing native exclusion.",
        category="test", documentation="# Native\n\nShipped.\n", created_by="native",
    ))
    await db.execute(
        "UPDATE skills SET created_at=? WHERE name=?",
        ("2026-01-01T00:00:00+00:00", "journey-skill"),
    )
    memory = MemoryStore(events)
    await memory.add(MemoryRecord(
        content="User prefers staging before production.",
        category="personal", source="test",
        created_at="2026-02-01T00:00:00+00:00",
        updated_at="2026-02-01T00:00:00+00:00",
    ))
    app = SimpleNamespace(skills=skills, _redact=lambda text: str(text))
    result = await journey_command(_ctx(app))
    assert result.success is True
    items = result.data["items"]
    assert [item["kind"] for item in items] == ["skill", "memory"]
    assert items[0]["id"] == "journey-skill"
    assert items[1]["kind"] == "memory"
    assert "staging" in items[1]["detail"]
    assert all(item["id"] != "native-journey" for item in items)


async def test_journey_is_truthful_when_empty():
    events = EventBus()
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    await skills.ready()
    app = SimpleNamespace(skills=skills, _redact=lambda text: str(text))
    result = await journey_command(_ctx(app))
    assert result.success is True
    assert result.data["items"] == []
    assert "no learned items" in result.message
