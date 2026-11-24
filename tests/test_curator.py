"""Deterministic skill curator: stale/archive/reactivate and operator controls."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from leo import db
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
async def env(tmp_path):
    settings = Settings()
    settings.skills.curator.stale_after_days = 14
    settings.skills.curator.archive_after_days = 30
    events = EventBus()
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    await skills.ready()
    curator = CuratorEngine(settings, skills, events)
    app = SimpleNamespace(skills=skills, curator=curator, _redact=lambda text: str(text))
    return SimpleNamespace(
        settings=settings, events=events, skills=skills, curator=curator, app=app,
    )


async def _authored(env, name: str, *, description: str | None = None) -> None:
    await env.skills.register(SkillManifest(
        name=name,
        description=description or f"Use when testing {name}.",
        category="test",
        documentation=f"# {name}\n\nA test skill.\n",
        created_by="agent",
    ))


async def _backdate(env, name: str, days: float, *, state: str = "active") -> None:
    stamp = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    row = env.skills._usage[name]
    row["updated_at"] = stamp
    row["last_viewed_at"] = stamp
    row["last_used_at"] = stamp
    row["last_patched_at"] = stamp
    row["state"] = state
    await db.execute(
        "UPDATE skill_usage SET updated_at=?, last_viewed_at=?, last_used_at=?, "
        "last_patched_at=?, state=? WHERE skill_name=?",
        (stamp, stamp, stamp, stamp, state, name),
    )


def _ctx(env, argv: list[str], *, principal: str = "operator") -> CommandContext:
    return CommandContext(
        app=env.app, principal=principal, surface="cli", channel="cli",
        argv=argv, raw="/skills " + " ".join(argv), registry=_Registry(),
        conversation_id="conv-curator",
    )


async def test_idle_skill_goes_stale_then_archived(env):
    await _authored(env, "stale-skill")
    await _backdate(env, "stale-skill", 15)
    stale = await env.curator.scan()
    assert stale["staled"] == 1
    assert (await env.skills.usage("stale-skill"))["state"] == "stale"
    assert (await env.skills.get("stale-skill")).enabled is True

    await _backdate(env, "stale-skill", 31, state="stale")
    archived = await env.curator.scan()
    assert archived["archived"] == 1
    record = await env.skills.get("stale-skill")
    assert record.enabled is False
    assert record.status == "archived"
    assert all(entry["name"] != "stale-skill" for entry in env.skills.index_entries())


async def test_recent_activity_keeps_a_skill_active(env):
    await _authored(env, "fresh-skill")
    await _backdate(env, "fresh-skill", 2)
    counts = await env.curator.scan()
    assert counts["staled"] == 0 and counts["archived"] == 0
    assert (await env.skills.usage("fresh-skill"))["state"] == "active"


async def test_stale_skill_with_new_activity_is_reactivated(env):
    await _authored(env, "revived-skill")
    await _backdate(env, "revived-skill", 1, state="stale")
    counts = await env.curator.scan()
    assert counts["reactivated"] == 1
    assert (await env.skills.usage("revived-skill"))["state"] == "active"


async def test_pinned_and_native_skills_are_exempt(env):
    await _authored(env, "pinned-skill")
    await _backdate(env, "pinned-skill", 45)
    await env.skills.set_pinned("pinned-skill", True)
    await env.skills.register(SkillManifest(
        name="native-skill", description="Use when testing native exemption.",
        category="test", documentation="# Native\n\nShipped.\n",
    ))
    await _backdate(env, "native-skill", 45)
    counts = await env.curator.scan()
    assert counts["archived"] == 0
    assert (await env.skills.usage("pinned-skill"))["state"] == "active"
    assert (await env.skills.usage("native-skill"))["state"] == "active"


async def test_dry_run_reports_without_mutating(env):
    await _authored(env, "dry-skill")
    await _backdate(env, "dry-skill", 31)
    counts = await env.curator.scan(dry_run=True)
    assert counts["archived"] == 1
    assert (await env.skills.usage("dry-skill"))["state"] == "active"
    assert await env.curator.last_run() is None


async def test_due_requires_both_the_interval_and_idle_window(env):
    from leo.skills.curator import LAST_RUN_KEY

    old = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
    await db.execute(
        "INSERT OR REPLACE INTO kv(key,value,updated_at) VALUES(?,?,?)",
        (LAST_RUN_KEY, json.dumps({"ran_at": old}), old),
    )
    now = datetime.now(UTC)
    assert await env.curator.due(
        now=now, idle_since=now - timedelta(minutes=5)
    ) is False
    assert await env.curator.due(
        now=now, idle_since=now - timedelta(hours=5)
    ) is True


async def test_first_check_seeds_the_clock_and_defers(env):
    assert await env.curator.due() is False
    last = await env.curator.last_run()
    assert last is not None and last["status"] == "seeded"


async def test_operator_can_pin_archive_restore_and_run(env):
    await _authored(env, "managed-skill")
    pinned = await skills_command(_ctx(env, ["pin", "managed-skill"]))
    assert pinned.success is True, pinned.message
    assert (await env.skills.get("managed-skill")).model_extra["pinned"] is True

    archived = await skills_command(_ctx(env, ["archive", "managed-skill"]))
    assert archived.success is True, archived.message
    assert (await env.skills.get("managed-skill")).status == "archived"

    restored = await skills_command(_ctx(env, ["restore", "managed-skill"]))
    assert restored.success is True, restored.message
    assert (await env.skills.get("managed-skill")).status == "registered"

    status = await skills_command(_ctx(env, ["curator", "status"]))
    assert status.success is True
    assert "stale_after_days" in status.data

    run = await skills_command(_ctx(env, ["curator", "run"]))
    assert run.success is True
    assert "scanned" in run.data


async def test_non_operator_cannot_change_lifecycle(env):
    await _authored(env, "protected-skill")
    for argv in (
        ["pin", "protected-skill"], ["lock", "protected-skill"],
        ["archive", "protected-skill"], ["curator", "run"],
    ):
        result = await skills_command(_ctx(env, argv, principal="guest"))
        assert result.success is False, argv
        assert result.status.value == "denied", argv
    assert (await env.skills.get("protected-skill")).status == "registered"
