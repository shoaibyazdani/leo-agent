"""Staged background skill writes: exact payload, digest, one-shot replay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.commands.builtins import skills_command
from leo.commands.registry import CommandContext
from leo.config import Settings
from leo.contracts import ToolCall
from leo.events import EventBus
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl
from leo.skills import staging
from leo.skills.registry import SkillRegistry
from leo.tools import ToolExecutorImpl, ToolRegistry, skill_manage_tools, skill_tools


@pytest.fixture
async def env(tmp_path, monkeypatch):
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.learning.review.write_approval = True
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = skill_tools.register(registry) + skill_manage_tools.register(registry)
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    await skills.ready()
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    services = {"skills": skills, "settings": settings, "permissions": permissions, "events": events}

    original_context = executor._context

    def with_services(call, task_key, dry_run):
        ctx = original_context(call, task_key, dry_run)
        ctx["services"] = services
        return ctx

    monkeypatch.setattr(executor, "_context", with_services)
    app = SimpleNamespace(skills=skills, executor=executor, _redact=lambda text: str(text))
    return SimpleNamespace(
        settings=settings, events=events, registry=registry, executor=executor,
        skills=skills, names=names, app=app,
    )


def _content(body: str = "Run the procedure in order.") -> str:
    return (
        "# Staged Skill\n\nA staged authored skill.\n\n"
        "## When to Use\n\n- A test asks for it.\n\n"
        "## Procedure\n\n1. " + body + "\n"
    )


async def _run(env, action, *, actor="review:engine", **arguments):
    call = ToolCall(
        name="skill_manage",
        arguments={"action": action, **arguments},
        actor=actor,
    )
    return await env.executor.execute(call)


class _Registry:
    operator_principals = frozenset({"operator"})
    operator_surfaces = frozenset({"cli", "chat", "gateway", "api", "desktop"})


def _ctx(env, argv: list[str], *, principal: str = "operator") -> CommandContext:
    return CommandContext(
        app=env.app, principal=principal, surface="cli", channel="cli",
        argv=argv, raw="/skills " + " ".join(argv), registry=_Registry(),
        conversation_id="conv-stage",
    )


async def test_review_write_is_staged_not_applied(env):
    result = await _run(
        env, "create", name="staged-skill",
        description="Use when staging a skill.",
        category="development",
        content=_content(),
    )
    assert result.success is True, result.error
    assert result.output["staged"] is True
    assert result.output["category"] == "development"
    assert result.output["description"] == "Use when staging a skill."
    assert result.output["proposal_id"].startswith("sprop_")
    assert await env.skills.get("staged-skill") is None

    pending = await staging.pending()
    assert len(pending) == 1
    assert pending[0]["action"] == "create"
    assert pending[0]["skill"] == "staged-skill"
    assert pending[0]["payload"]["content"] == _content()


async def test_background_writes_apply_directly_by_default(env):
    env.settings.learning.review.write_approval = False
    result = await _run(
        env, "create", name="direct-skill",
        description="Use when testing the direct-write default.",
        content=_content(),
    )
    assert result.success is True, result.error
    assert "staged" not in result.output
    assert await env.skills.get("direct-skill") is not None
    assert await staging.pending() == []


async def test_foreground_writes_are_not_staged(env):
    result = await _run(
        env, "create", actor="chat:user", name="foreground-skill",
        description="Use when writing in the foreground.",
        content=_content(),
    )
    assert result.success is True, result.error
    assert "staged" not in result.output
    assert await env.skills.get("foreground-skill") is not None
    assert await staging.pending() == []


async def test_approve_applies_exactly_once_and_verifies_the_digest(env):
    staged = await _run(
        env, "create", name="approved-skill",
        description="Use when approving a staged skill.",
        content=_content(),
    )
    proposal_id = staged.output["proposal_id"]

    first = await skills_command(_ctx(env, ["approve", proposal_id]))
    assert first.success is True, first.message
    record = await env.skills.get("approved-skill")
    assert record is not None
    assert record.created_by == "agent"
    assert (await staging.load(proposal_id))["status"] == "applied"

    second = await skills_command(_ctx(env, ["approve", proposal_id]))
    assert second.success is False
    assert second.code == "not_pending"


async def test_tampered_payload_is_refused(env):
    staged = await _run(
        env, "create", name="tampered-skill",
        description="Use when tampering with the payload.",
        content=_content(),
    )
    proposal_id = staged.output["proposal_id"]
    row = await db.fetchone(
        "SELECT value FROM kv WHERE key=?", (f"{staging.PREFIX}{proposal_id}",)
    )
    record = json.loads(row["value"])
    record["payload"]["name"] = "different-skill"
    await db.execute(
        "UPDATE kv SET value=? WHERE key=?",
        (json.dumps(record), f"{staging.PREFIX}{proposal_id}"),
    )

    result = await skills_command(_ctx(env, ["approve", proposal_id]))
    assert result.success is False
    assert (await staging.load(proposal_id))["status"] == "failed"
    assert await env.skills.get("tampered-skill") is None
    assert await env.skills.get("different-skill") is None


async def test_reject_is_terminal_and_writes_nothing(env):
    staged = await _run(
        env, "create", name="rejected-skill",
        description="Use when rejecting a staged skill.",
        content=_content(),
    )
    proposal_id = staged.output["proposal_id"]
    result = await skills_command(_ctx(env, ["reject", proposal_id]))
    assert result.success is True
    assert (await staging.load(proposal_id))["status"] == "rejected"
    assert await env.skills.get("rejected-skill") is None
    again = await skills_command(_ctx(env, ["approve", proposal_id]))
    assert again.success is False


async def test_pending_queue_is_bounded(env, monkeypatch):
    monkeypatch.setattr(staging, "MAX_PENDING", 1)
    first = await _run(
        env, "create", name="first-staged",
        description="Use when filling the queue.",
        content=_content(),
    )
    assert first.success is True
    second = await _run(
        env, "create", name="second-staged",
        description="Use when the queue is full.",
        content=_content(),
    )
    assert second.success is False
    assert "awaiting review" in (second.error or "")


async def test_delete_is_not_staged_and_keeps_the_high_risk_gate(env):
    # Seed an authored skill directly (a foreground write), then ask the
    # review to delete it: deletion has no staging path.
    await _run(
        env, "create", actor="chat:user", name="delete-target",
        description="Use when testing delete staging.",
        content=_content(),
    )
    with pytest.raises(ApprovalRequired):
        await _run(env, "delete", name="delete-target")
    assert await env.skills.get("delete-target") is not None
    assert await staging.pending() == []


async def test_only_an_operator_may_apply_a_staged_write(env):
    staged = await _run(
        env, "create", name="operator-only",
        description="Use when testing approval authority.",
        content=_content(),
    )
    proposal_id = staged.output["proposal_id"]
    denied = await skills_command(_ctx(env, ["approve", proposal_id], principal="guest"))
    assert denied.success is False
    assert denied.status.value == "denied"
    assert await env.skills.get("operator-only") is None
    assert (await staging.load(proposal_id))["status"] == "pending"
