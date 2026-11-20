"""Model/tooling insight tools: real routers, registries and usage rows.

Every assertion goes through ``ToolExecutorImpl`` (with the composition-root
service injection wired in a test subclass) so validation, permissions, audit
and event paths are exercised together with the synthesized views.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import ModelProviderConfig, Settings
from leo.contracts import RiskLevel, SkillManifest, ToolCall, ToolSpec
from leo.events import EventBus
from leo.models.mock import MockProvider
from leo.models.router import ModelRouter
from leo.security.permissions import PermissionManagerImpl
from leo.skills.registry import SkillRegistry
from leo.tools.executor import ToolExecutorImpl
from leo.tools.model_tools import TOOL_NAMES, register
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


@pytest.fixture
async def env():
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    router = ModelRouter(
        settings,
        providers={"mock": MockProvider(ModelProviderConfig(kind="mock", model="mock-1"), name="mock")},
    )
    registry = ToolRegistry()
    names = register(registry)
    assert names == list(TOOL_NAMES)
    return SimpleNamespace(
        settings=settings,
        events=events,
        permissions=permissions,
        router=router,
        registry=registry,
        services={"router": router, "tools": registry, "skills": None},
    )


def _executor(env, *, registry=None, services=None):
    return _ServicesExecutor(
        registry if registry is not None else env.registry,
        env.permissions,
        env.events,
        env.settings,
        services=env.services if services is None else services,
    )


async def _run(env, name: str, args: dict | None = None, *, services=None, registry=None, **kwargs):
    call = ToolCall(
        name=name,
        arguments=args or {},
        actor="tester",
        correlation_id="corr-model-tools",
        **kwargs,
    )
    return await _executor(env, registry=registry, services=services).execute(call)


async def _seed_usage() -> None:
    rows = [
        (
            "2026-01-01T00:00:01+00:00", "other", "m2", None, None,
            10, 5, 5.0, 0.0, 0, json.dumps({"task_category": "chat"}),
        ),
        (
            "2026-01-01T00:00:02+00:00", "mock", "m1", "task-1", "agent-1",
            80, 20, 10.0, 0.002, 1,
            json.dumps({"estimated": False, "task_category": "coding", "attempt": 1, "stream": True}),
        ),
        (
            "2026-01-01T00:00:03+00:00", "mock", "m1", "task-1", "agent-1",
            120, 30, 15.5, 0.003, 1,
            json.dumps({"estimated": True, "task_category": "coding", "attempt": 0, "stream": False}),
        ),
    ]
    for row in rows:
        await db.execute(
            "INSERT INTO model_usage(ts,provider,model,task_id,agent_id,input_tokens,"
            "output_tokens,latency_ms,cost,success,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )


def _usage_row(row_id: int, *, ts: str, provider: str, model: str, task_id, agent_id,
               input_tokens: int, output_tokens: int, token_source: str,
               latency_ms: float, cost: float, success: bool,
               task_category: str, attempt: int | None, stream: bool | None) -> dict:
    return {
        "id": row_id,
        "timestamp": ts,
        "provider": provider,
        "model": model,
        "task_id": task_id,
        "agent_id": agent_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_source": token_source,
        "estimated": token_source == "estimated",
        "latency_ms": latency_ms,
        "cost": cost,
        "success": success,
        "task_category": task_category,
        "attempt": attempt,
        "stream": stream,
    }


# --------------------------------------------------------------------------- registration


def test_register_exposes_typed_read_only_specs(env) -> None:
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert set(specs) == set(TOOL_NAMES)
    assert {name: spec.category for name, spec in specs.items()} == {
        "model_providers": "models",
        "model_usage": "models",
        "tool_registry_list": "tools",
        "skill_registry_list": "skills",
    }
    expected_permissions = {
        "model_providers": ["model:read"],
        "model_usage": ["model:usage"],
        "tool_registry_list": ["tool:read"],
        "skill_registry_list": ["skill:read"],
    }
    for name, spec in specs.items():
        assert spec.risk is RiskLevel.LOW
        assert spec.permissions == expected_permissions[name]
        assert spec.description and spec.version
        assert spec.resource_field == ""
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.input_schema["properties"]["limit"]["maximum"] <= 100

    with pytest.raises(ValueError, match="already registered"):
        register(env.registry)


# --------------------------------------------------------------------------- providers


async def test_model_providers_synthesizes_real_router_state(env) -> None:
    env.settings.models.default = "z-primary"
    env.settings.models.fallback = "mock"
    env.settings.models.providers = {
        "z-primary": ModelProviderConfig(
            kind="mock", model="m1", capabilities=["usage", "tools", "text"],
            context_window=8192, cost_per_million_input=1.5,
            cost_per_million_output=2.0, api_key_env="TEST_KEY",
        ),
        "a-disabled": ModelProviderConfig(
            kind="mock", model="m0", enabled=False, capabilities=["text"]
        ),
        "m-live": ModelProviderConfig(
            kind="mock", model="m2", capabilities=["text", "usage"]
        ),
    }
    env.services["router"] = ModelRouter(
        env.settings,
        providers={
            name: MockProvider(config, name=name)
            for name, config in env.settings.models.providers.items()
            if config.enabled
        },
    )

    result = await _run(env, "model_providers")
    assert result.success, result.error
    provider = lambda name: {
        "name": name,
        "model": {"a-disabled": "m0", "m-live": "m2", "z-primary": "m1"}[name],
        "kind": "mock",
        "active": name != "a-disabled",
        "enabled": name != "a-disabled",
        "capabilities": {
            "a-disabled": ["text"],
            "m-live": ["text", "usage"],
            "z-primary": ["text", "tools", "usage"],
        }[name],
        "context_window": 8192 if name == "z-primary" else 128000,
        "base_url_configured": False,
        "credential_env_configured": name == "z-primary",
        "cost_per_million_input": 1.5 if name == "z-primary" else 0.0,
        "cost_per_million_output": 2.0 if name == "z-primary" else 0.0,
        "usage_supported": name != "a-disabled",
    }
    assert result.output == {
        "source": "router.providers",
        "default": "z-primary",
        "fallback": "mock",
        "count": 3,
        "returned": 3,
        "providers": [provider("a-disabled"), provider("m-live"), provider("z-primary")],
        "token_accounting": (
            "provider token counts count as measurements only when the provider reports them; "
            "derived counts are labeled estimates"
        ),
    }

    filtered = await _run(env, "model_providers", {"provider": "m-live"})
    assert filtered.success and filtered.output["providers"] == [provider("m-live")]
    assert filtered.output["count"] == 1

    limited = await _run(env, "model_providers", {"limit": 1})
    assert limited.success
    assert limited.output["count"] == 3 and limited.output["returned"] == 1
    assert limited.output["providers"] == [provider("a-disabled")]


# --------------------------------------------------------------------------- usage


async def test_model_usage_labels_provenance_and_aggregates_exactly(env) -> None:
    await _seed_usage()
    result = await _run(env, "model_usage")
    assert result.success, result.error
    bucket = lambda calls, i, o, cost: {
        "calls": calls, "input_tokens": i, "output_tokens": o, "cost": cost,
    }
    assert result.output == {
        "source": "router.usage_report",
        "measurement_filter": "all",
        "count": 3,
        "returned": 3,
        "totals": {
            "calls": 3,
            "input_tokens": 210,
            "output_tokens": 55,
            "cost": 0.005,
            "by_source": {
                "estimated": bucket(1, 120, 30, 0.003),
                "provider_reported": bucket(1, 80, 20, 0.002),
                "unverified": bucket(1, 10, 5, 0.0),
            },
        },
        "note": "estimated/unverified token counts are derived locally, not provider measurements",
        "usage": [
            _usage_row(
                3, ts="2026-01-01T00:00:03+00:00", provider="mock", model="m1",
                task_id="task-1", agent_id="agent-1", input_tokens=120,
                output_tokens=30, token_source="estimated", latency_ms=15.5,
                cost=0.003, success=True, task_category="coding", attempt=0,
                stream=False,
            ),
            _usage_row(
                2, ts="2026-01-01T00:00:02+00:00", provider="mock", model="m1",
                task_id="task-1", agent_id="agent-1", input_tokens=80,
                output_tokens=20, token_source="provider_reported", latency_ms=10.0,
                cost=0.002, success=True, task_category="coding", attempt=1,
                stream=True,
            ),
            _usage_row(
                1, ts="2026-01-01T00:00:01+00:00", provider="other", model="m2",
                task_id=None, agent_id=None, input_tokens=10, output_tokens=5,
                token_source="unverified", latency_ms=5.0, cost=0.0, success=False,
                task_category="chat", attempt=None, stream=None,
            ),
        ],
    }

    estimated = await _run(env, "model_usage", {"measurement": "estimated"})
    assert estimated.success
    assert estimated.output["count"] == 1
    assert estimated.output["usage"][0]["token_source"] == "estimated"
    assert estimated.output["usage"][0]["estimated"] is True
    assert estimated.output["totals"]["by_source"]["provider_reported"]["calls"] == 0
    assert "estimated" in estimated.output["note"]

    bounded = await _run(env, "model_usage", {"limit": 1, "provider": "mock"})
    assert bounded.success
    assert bounded.output["count"] == 2 and bounded.output["returned"] == 1
    assert bounded.output["usage"][0]["id"] == 3


# --------------------------------------------------------------------------- tool registry


async def test_tool_registry_list_reads_seeded_specs(env) -> None:
    seeded = ToolRegistry()
    seeded.register(
        ToolSpec(
            name="zeta", description="Z tool", category="utility",
            risk=RiskLevel.MEDIUM, permissions=["filesystem"],
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "mode": {"type": "string"}},
                "required": ["path"], "additionalProperties": False,
            },
            resource_field="path",
        ),
        lambda args, ctx: None,
    )
    seeded.register(
        ToolSpec(
            name="alpha", description="A tool", category="models",
            risk=RiskLevel.LOW, permissions=["model:read"],
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [], "additionalProperties": False,
            },
            resource_field="",
        ),
        lambda args, ctx: None,
    )
    env.services["tools"] = seeded

    alpha = {
        "name": "alpha", "description": "A tool", "category": "models", "risk": "low",
        "permissions": ["model:read"], "resource_field": "",
        "input_fields": ["limit"], "required_fields": [],
    }
    zeta = {
        "name": "zeta", "description": "Z tool", "category": "utility", "risk": "medium",
        "permissions": ["filesystem"], "resource_field": "path",
        "input_fields": ["mode", "path"], "required_fields": ["path"],
    }
    result = await _run(env, "tool_registry_list")
    assert result.success, result.error
    assert result.output == {
        "source": "tool_registry",
        "count": 2,
        "returned": 2,
        "categories": {"models": 1, "utility": 1},
        "risks": {"low": 1, "medium": 1, "high": 0},
        "tools": [alpha, zeta],
    }

    filtered = await _run(env, "tool_registry_list", {"category": "utility"})
    assert filtered.success
    assert filtered.output["count"] == 1
    assert filtered.output["categories"] == {"utility": 1}
    assert filtered.output["tools"] == [zeta]

    bounded = await _run(env, "tool_registry_list", {"limit": 1})
    assert bounded.success
    assert bounded.output["count"] == 2
    assert bounded.output["returned"] == 1
    assert bounded.output["tools"] == [alpha]


# --------------------------------------------------------------------------- skill registry


async def test_skill_registry_list_reads_seeded_manifests(env) -> None:
    skills = SkillRegistry(env.events, env.permissions, auto_seed=False)
    common = dict(
        category="insight-fixture", permissions=["skill:read"],
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        tests=["tests/test_tools_model.py"],
    )
    await skills.register(
        SkillManifest(
            name="beta-skill", description="Beta fixture skill", version="2.0.0",
            source_name="fixture", **common,
        ),
        lambda args, ctx: {"ok": True},
    )
    await skills.register(
        SkillManifest(
            name="alpha-skill", description="Alpha fixture skill", version="1.0.0",
            **common,
        ),
        None,
    )
    env.services["skills"] = skills

    result = await _run(env, "skill_registry_list", {"category": "insight-fixture"})
    assert result.success, result.error
    assert result.output == {
        "source": "skill_registry",
        "count": 2,
        "returned": 2,
        "categories": {"insight-fixture": 2},
        "executable": 1,
        "skills": [
            {
                "name": "alpha-skill", "version": "1.0.0",
                "category": "insight-fixture", "description": "Alpha fixture skill",
                "status": "registered", "health": "available", "enabled": True,
                "has_handler": False, "permissions": ["skill:read"],
                "dependencies": [], "input_fields": ["query"],
            },
            {
                "name": "beta-skill", "version": "2.0.0",
                "category": "insight-fixture", "description": "Beta fixture skill",
                "status": "registered", "health": "available", "enabled": True,
                "has_handler": True, "permissions": ["skill:read"],
                "dependencies": [], "input_fields": ["query"],
            },
        ],
    }

    bounded = await _run(
        env, "skill_registry_list",
        {"category": "insight-fixture", "limit": 1, "enabled_only": True},
    )
    assert bounded.success
    assert bounded.output["count"] == 2 and bounded.output["returned"] == 1
    assert bounded.output["skills"][0]["name"] == "alpha-skill"


# --------------------------------------------------------------------------- failure paths


async def test_missing_services_fail_closed(env) -> None:
    expected = {
        "model_providers": "router",
        "model_usage": "router",
        "tool_registry_list": "tools",
        "skill_registry_list": "skills",
    }
    for name, service in expected.items():
        result = await _run(env, name, services={})
        assert not result.success
        assert result.decision == "error"
        assert service in result.error
        assert result.output is None
        assert f"missing required service '{service}'" in result.error


async def test_invalid_arguments_and_scope_are_rejected(env) -> None:
    bad = await _run(env, "model_usage", {"limit": 101})
    assert bad.decision == "invalid_arguments" and not bad.success
    assert "maximum" in bad.error or "<= 100" in bad.error

    unknown = await _run(env, "tool_registry_list", {"unexpected": True})
    assert unknown.decision == "invalid_arguments" and not unknown.success
    assert "unknown field" in unknown.error

    scoped = await ToolExecutorImpl(
        env.registry, env.permissions, env.events, env.settings
    ).execute(
        ToolCall(
            name="model_providers", arguments={}, actor="tester",
            correlation_id="corr-model-tools",
        ),
        scope=frozenset({"browser"}),
    )
    assert not scoped.success and scoped.decision == "out_of_scope"
    assert scoped.output is None


async def test_registration_services_fallback_and_audit(env) -> None:
    registry = ToolRegistry()
    register(registry, services=env.services)
    executor = _ServicesExecutor(
        registry, env.permissions, env.events, env.settings, services={}
    )
    result = await executor.execute(
        ToolCall(name="model_providers", arguments={}, actor="tester")
    )
    assert result.success, result.error
    assert result.output["source"] == "router.providers"

    row = await db.fetchone(
        "SELECT tool, actor, decision, success FROM tool_calls WHERE tool=?",
        ("model_providers",),
    )
    assert row is not None
    assert (row["tool"], row["actor"], row["decision"], row["success"]) == (
        "model_providers", "tester", "allowed", 1,
    )

    events = await db.fetchall(
        "SELECT topic FROM events WHERE topic IN ('tool.called','tool.failed')"
    )
    assert any(event["topic"] == "tool.called" for event in events)
