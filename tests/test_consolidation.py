"""Opt-in LLM consolidation: staged plans, operator approval, no auto-rewrite."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo.commands.builtins import skills_command
from leo.commands.registry import CommandContext
from leo.config import Settings
from leo.contracts import ModelResponse, SkillManifest
from leo.events import EventBus
from leo.review.consolidate import ConsolidationEngine
from leo.security.permissions import PermissionManagerImpl
from leo.skills import staging
from leo.skills.curator import CuratorEngine
from leo.skills.registry import SkillRegistry
from leo.tools import ToolExecutorImpl, ToolRegistry, skill_manage_tools, skill_tools


class _Registry:
    operator_principals = frozenset({"operator"})
    operator_surfaces = frozenset({"cli", "chat", "gateway", "api", "desktop"})


class JsonRouter:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        value = self.responses.pop(0) if self.responses else {"action": "keep"}
        return ModelResponse(
            content=json.dumps(value), model="scripted", provider="script",
        )


@pytest.fixture
async def env(tmp_path, monkeypatch):
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.skills.curator.consolidate = True
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    skill_tools.register(registry)
    skill_manage_tools.register(registry)
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    await skills.ready()
    curator = CuratorEngine(settings, skills, events)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    services = {"skills": skills, "settings": settings, "permissions": permissions, "events": events}

    original_context = executor._context

    def with_services(call, task_key, dry_run):
        ctx = original_context(call, task_key, dry_run)
        ctx["services"] = services
        return ctx

    monkeypatch.setattr(executor, "_context", with_services)
    app = SimpleNamespace(
        skills=skills, executor=executor, _redact=lambda text: str(text),
        curator=curator,
        consolidation=ConsolidationEngine(settings, JsonRouter([]), skills, curator, events),
    )
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events, skills=skills,
        curator=curator, executor=executor, app=app,
    )


async def _seed_group(env):
    for name in ("python-tests", "python-testing"):
        await env.skills.register(SkillManifest(
            name=name,
            description="Use when running Python tests.",
            category="test",
            documentation="# Python Tests\n\nRun the test suite and fix failures.\n",
            created_by="agent",
        ))
    # A native look-alike must never enter a candidate group.
    await env.skills.register(SkillManifest(
        name="native-python-tests",
        description="Use when running Python tests.",
        category="test",
        documentation="# Native\n\nShipped.\n",
        created_by="native",
    ))


def _merge_plan(umbrella: str = "python-testing-workflow", absorbed=None) -> dict:
    return {
        "action": "merge",
        "umbrella": {
            "name": umbrella,
            "description": "Use when running Python tests.",
            "content": (
                "# Python Testing\n\nRun the suite, read failures, fix the cause.\n\n"
                "## Procedure\n\n1. Run the suite.\n2. Fix failures.\n"
            ),
        },
        "absorbed": absorbed or ["python-tests", "python-testing"],
        "reason": "same class-level workflow",
    }


def _engine(env, router) -> ConsolidationEngine:
    return ConsolidationEngine(env.settings, router, env.skills, env.curator, env.events)


def _ctx(env, argv: list[str], *, principal: str = "operator") -> CommandContext:
    return CommandContext(
        app=env.app, principal=principal, surface="cli", channel="cli",
        argv=argv, raw="/skills " + " ".join(argv), registry=_Registry(),
        conversation_id="conv-consolidate",
    )


async def test_dry_run_reports_without_staging(env):
    await _seed_group(env)
    router = JsonRouter([_merge_plan()])
    result = await _engine(env, router).propose(dry_run=True)
    assert result["status"] == "completed"
    assert result["proposed"] == 1
    assert result["proposal_ids"] == []
    assert result["plans"][0]["umbrella"] == "python-testing-workflow"
    assert await staging.pending() == []
    assert await env.skills.get("python-testing-workflow") is None


async def test_plan_is_staged_and_approval_applies_it(env):
    await _seed_group(env)
    router = JsonRouter([_merge_plan()])
    result = await _engine(env, router).propose()
    assert result["proposed"] == 1
    proposal_id = result["proposal_ids"][0]

    pending = await staging.pending()
    assert len(pending) == 1
    assert pending[0]["kind"] == "consolidate"
    assert pending[0]["title"].startswith("Consolidate")
    assert await env.skills.get("python-testing-workflow") is None

    applied = await skills_command(_ctx(env, ["approve", proposal_id]))
    assert applied.success is True, applied.message
    umbrella = await env.skills.get("python-testing-workflow")
    assert umbrella is not None
    assert umbrella.created_by == "agent"
    for sibling in ("python-tests", "python-testing"):
        record = await env.skills.get(sibling)
        assert record.status == "archived"
        assert record.enabled is False
    assert (await staging.load(proposal_id))["status"] == "applied"


async def test_keep_decision_stages_nothing(env):
    await _seed_group(env)
    router = JsonRouter([{"action": "keep", "reason": "genuinely distinct"}])
    result = await _engine(env, router).propose()
    assert result["kept"] == 1
    assert result["proposed"] == 0
    assert await staging.pending() == []


async def test_invalid_plan_fails_closed(env):
    await _seed_group(env)
    router = JsonRouter([_merge_plan(absorbed=["python-tests", "not-in-the-group"])])
    result = await _engine(env, router).propose()
    assert result["failed"] == 1
    assert result["proposed"] == 0
    assert await staging.pending() == []


async def test_disabled_consolidation_is_truthful(env):
    env.settings.skills.curator.consolidate = False
    result = await _engine(env, JsonRouter([])).propose()
    assert result["status"] == "disabled"
    denied = await skills_command(_ctx(env, ["curator", "consolidate"]))
    assert denied.success is False
    assert denied.code == "consolidation_disabled"


async def test_non_operator_cannot_propose(env):
    denied = await skills_command(
        _ctx(env, ["curator", "consolidate", "--dry-run"], principal="guest")
    )
    assert denied.success is False
    assert denied.status.value == "denied"
