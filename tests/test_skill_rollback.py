"""Bounded before-state snapshots, single-mutation rollback, consolidation report."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo.commands.builtins import skills_command
from leo.commands.registry import CommandContext
from leo.config import Settings
from leo.contracts import SkillManifest
from leo.events import EventBus
from leo.skills.curator import CuratorEngine
from leo.skills.registry import SkillRegistry


class _Registry:
    operator_principals = frozenset({"operator"})
    operator_surfaces = frozenset({"cli", "chat", "gateway", "api", "desktop"})


@pytest.fixture
async def env():
    settings = Settings()
    events = EventBus()
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    await skills.ready()
    curator = CuratorEngine(settings, skills, events)
    app = SimpleNamespace(skills=skills, curator=curator, _redact=lambda text: str(text))
    return SimpleNamespace(
        settings=settings, events=events, skills=skills, curator=curator, app=app,
    )


def _manifest(name: str, body: str, *, created_by: str = "agent", description: str | None = None):
    return SkillManifest(
        name=name,
        description=description or f"Use when testing {name}.",
        category="test",
        documentation=f"# {name}\n\n{body}\n",
        created_by=created_by,
    )


def _ctx(env, argv: list[str], *, principal: str = "operator") -> CommandContext:
    return CommandContext(
        app=env.app, principal=principal, surface="cli", channel="cli",
        argv=argv, raw="/skills " + " ".join(argv), registry=_Registry(),
        conversation_id="conv-rollback",
    )


async def test_patch_snapshot_rolls_back_the_previous_body(env):
    await env.skills.register(_manifest("roll-skill", "original body"))
    await env.skills.register(_manifest("roll-skill", "changed body"), replace=True)
    record = await env.skills.get("roll-skill")
    assert "changed body" in record.documentation

    snapshots = await env.skills.snapshots("roll-skill")
    assert len(snapshots) == 1
    assert snapshots[0]["kind"] == "skill"
    restored = await env.skills.rollback(snapshots[0]["id"])
    assert restored["restored"] is True
    record = await env.skills.get("roll-skill")
    assert "original body" in record.documentation
    assert "changed body" not in record.documentation


async def test_file_write_and_removal_snapshots_restore_the_file(env):
    await env.skills.register(_manifest("file-roll", "body"))
    await env.skills.write_file("file-roll", "references/api.md", "version one")
    await env.skills.write_file("file-roll", "references/api.md", "version two")
    snapshots = await env.skills.snapshots("file-roll")
    file_snap = next(item for item in snapshots if item["kind"] == "file")
    await env.skills.rollback(file_snap["id"])
    assert await env.skills.read_file("file-roll", "references/api.md") == "version one"

    await env.skills.remove_file("file-roll", "references/api.md")
    assert await env.skills.read_file("file-roll", "references/api.md") is None
    snapshots = await env.skills.snapshots("file-roll")
    removal = next(
        item for item in snapshots
        if item["kind"] == "file" and item["action"] == "before.remove"
    )
    await env.skills.rollback(removal["id"])
    assert await env.skills.read_file("file-roll", "references/api.md") == "version one"


async def test_delete_snapshot_restores_the_whole_package(env):
    await env.skills.register(_manifest("delete-roll", "body"))
    await env.skills.write_file("delete-roll", "references/notes.md", "kept depth")
    await env.skills.unregister("delete-roll")
    assert await env.skills.get("delete-roll") is None

    snapshots = await env.skills.snapshots("delete-roll")
    assert snapshots and snapshots[0]["kind"] == "skill"
    await env.skills.rollback(snapshots[0]["id"])
    record = await env.skills.get("delete-roll")
    assert record is not None
    assert await env.skills.read_file("delete-roll", "references/notes.md") == "kept depth"


async def test_snapshot_retention_is_bounded(env):
    await env.skills.register(_manifest("many-snaps", "v0"))
    for index in range(1, 14):
        await env.skills.register(_manifest("many-snaps", f"v{index}"), replace=True)
    assert len(await env.skills.snapshots("many-snaps")) <= 10


async def test_native_skills_are_not_snapshotted(env):
    await env.skills.register(_manifest("native-roll", "shipped", created_by="native"))
    await env.skills.register(
        _manifest("native-roll", "changed", created_by="native"), replace=True
    )
    assert await env.skills.snapshots("native-roll") == []


async def test_rollback_command_is_operator_only(env):
    await env.skills.register(_manifest("cmd-roll", "old"))
    await env.skills.register(_manifest("cmd-roll", "new"), replace=True)
    snapshots = await env.skills.snapshots("cmd-roll")
    snapshot_id = snapshots[0]["id"]

    denied = await skills_command(_ctx(env, ["rollback", snapshot_id], principal="guest"))
    assert denied.success is False
    assert denied.status.value == "denied"
    assert "new" in (await env.skills.get("cmd-roll")).documentation

    applied = await skills_command(_ctx(env, ["rollback", snapshot_id]))
    assert applied.success is True, applied.message
    assert "old" in (await env.skills.get("cmd-roll")).documentation

    listed = await skills_command(_ctx(env, ["snapshots", "cmd-roll"]))
    assert listed.success is True
    assert listed.data["snapshots"]


async def test_consolidation_candidates_group_similar_authored_skills(env):
    await env.skills.register(_manifest(
        "python-tests", "Run and debug Python test suites.",
        description="Use when running Python test suites and fixing failures.",
    ))
    await env.skills.register(_manifest(
        "python-testing", "Run and debug Python test suites.",
        description="Use when running Python test suites and fixing failures.",
    ))
    await env.skills.register(_manifest(
        "deploy-staging", "Deploy the staging server.", description="Use when deploying staging.",
    ))
    await env.skills.register(_manifest(
        "native-python-tests", "Run Python tests.", created_by="native",
        description="Use when running Python test suites and fixing failures.",
    ))

    groups = await env.curator.candidates()
    assert len(groups) == 1
    assert groups[0]["skills"] == ["python-testing", "python-tests"]
    assert groups[0]["similarity"] >= 0.72
    # Native skills never appear as candidates.
    assert all("native-python-tests" not in group["skills"] for group in groups)

    result = await skills_command(_ctx(env, ["curator", "candidates"]))
    assert result.success is True
    assert result.data["candidates"][0]["skills"] == ["python-testing", "python-tests"]
