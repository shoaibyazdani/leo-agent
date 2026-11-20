"""Release-auditor specialist: seed shape, read-only grants and a real audit.

All evidence is local: seeded ``integrations`` rows, repository code/test files
and the live tool/agent registries. No provider calls, credentials or network.
"""

from __future__ import annotations

from importlib import import_module

import pytest

from leo import db
from leo.agents import AgentEngine, AgentRegistry
from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.specs import release_auditor as subject
from leo.contracts import AgentSpec, RiskLevel, utcnow
from leo.tools.builtins import register_builtins
from leo.tools.registry import ToolRegistry

SEED_KEYS = set(SPECIALIST_SPECS[0])


def _live_registry() -> ToolRegistry:
    """Build the same tool universe the integration layer registers."""
    registry = ToolRegistry()
    register_builtins(registry)
    for module_name in (
        "event_tools", "approval_tools", "agent_tools", "integration_tools",
        "skill_tools", "task_tools", "model_tools", "file_tools", "memory_tools",
        "data_tools", "git_tools", "automation_tools", "context_tools",
    ):
        import_module(f"leo.tools.{module_name}").register(registry, None)
    return registry


def test_seed_shape_matches_registry_rows():
    assert set(subject.SEED) == SEED_KEYS
    assert subject.SEED["id"] == "release-auditor"
    assert subject.SEED["capabilities"]
    assert subject.SEED["model_preferences"]
    assert subject.SEED["tool_permissions"] == list(subject.TOOL_GRANTS)
    assert subject.SEED["metadata"]["read_only"] is True
    assert subject.SEED["metadata"]["handler"] == "make_handler(services)"
    assert callable(subject.make_handler)

    spec = AgentSpec.model_validate(subject.SEED)
    assert spec.id == "release-auditor"
    assert spec.enabled is True


async def _register_auditor(registry: AgentRegistry) -> AgentSpec:
    """Idempotent setup: any pre-existing row is replaced by this exact SEED."""
    await registry.unregister(subject.SEED["id"])
    return await registry.register(subject.SEED, replace=True)


@pytest.mark.asyncio
async def test_seed_registers_and_survives_reload():
    registry = await AgentRegistry().initialize()
    await registry.unregister(subject.SEED["id"])
    assert await registry.get("release-auditor") is None
    await _register_auditor(registry)
    reloaded = await AgentRegistry().initialize()
    stored = await reloaded.get("release-auditor")
    assert stored is not None
    assert stored.tool_permissions == list(subject.TOOL_GRANTS)
    assert stored.metadata["handler_kind"] == "bespoke:make_handler"
    assert "release-auditor" in {s.id for s in await reloaded.list(enabled_only=True)}


def test_tool_grants_are_live_and_read_only():
    specs = {spec.name: spec for spec in _live_registry().list_specs()}
    grants = subject.SEED["tool_permissions"]
    assert grants and len(grants) == len(set(grants))
    # Grants are individual tools, not broad categories that would admit writes.
    assert all(grant in specs for grant in grants)
    assert not set(grants) & {spec.category for spec in specs.values()}
    assert all(spec.risk == RiskLevel.LOW for spec in (specs[grant] for grant in grants))
    mutating = (":write", ":send", ":run", ":execute", "delegate", "cancel", "resolve", "terminal")
    for grant in grants:
        assert not any(token in permission.lower() for permission in specs[grant].permissions for token in mutating), grant
    assert not set(grants) & {
        "write_file", "terminal", "http_get", "channel_send", "agent_delegate",
        "task_cancel", "approval_resolve", "skill_invoke", "metrics_snapshot",
    }


async def _seed_status_records() -> None:
    now = utcnow()
    for name, status, capabilities, verification in (
        ("slack", "live", '["send"]', "mock-tested"),
        ("relay", "blocked", '["send"]', "mock-tested"),
        ("local-runner", "live", '["read"]', "live-verified"),
    ):
        await db.execute(
            "INSERT INTO integrations(name,kind,status,config_ref,capabilities,verification,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "kind=excluded.kind, status=excluded.status, capabilities=excluded.capabilities, "
            "verification=excluded.verification, updated_at=excluded.updated_at",
            (name, "messaging", status, None, capabilities, verification, now, now),
        )


@pytest.mark.asyncio
async def test_audit_path_over_seeded_status_records():
    await _seed_status_records()
    registry = await AgentRegistry().initialize()
    await _register_auditor(registry)
    registry_before = await db.fetchall("SELECT id FROM agents")
    integrations_before = await db.fetchall("SELECT name,status,verification FROM integrations ORDER BY name")
    tool_calls_before = await db.fetchall("SELECT id FROM tool_calls")

    claims = [
        {
            "claim": "Release audit is implemented and documented",
            # Publishable docs only: the release tree removes the planning
            # documents, so evidence must point at files that ship.
            "source": "STATUS.md",
            "doc": "LEO_SECURITY.md",
            "quote": "Atomic decision + audit commit",
            "code": ["leo/agents/specs/release_auditor.py"],
            "tests": ["tests/test_agents_release_auditor.py"],
        },
        {
            "claim": "The auditor can read real registry evidence",
            "tools": ["audit_query", "integration_status"],
            "agents": ["release-auditor"],
            "executed": {"table": "agent_runs", "min_rows": 0},
        },
        {"claim": "Slack delivery is live", "status": "live", "integration": "slack"},
        {"claim": "Local runner is live", "status": "live", "integration": "local-runner"},
        {"claim": "Everything is production ready", "source": "docs/LEO_TEST_REPORT.md"},
    ]
    services = {"agent_registry": registry, "tool_registry": _live_registry()}
    engine = AgentEngine(registry)
    engine.set_handler("release-auditor", subject.make_handler(services))
    result = await engine.delegate(
        "Audit release readiness",
        agent_id="release-auditor",
        context={"claims": claims},
        verify=False,
    )

    assert result.success and result.agent_id == "release-auditor"
    output = result.output
    assert output["readiness"] == "blocked" and output["ready"] is False
    assert output["counts"]["claims"] == 5
    assert output["counts"]["verified"] == 3
    assert output["counts"]["mock_only"] == 1
    assert output["counts"]["unverified"] == 1
    assert output["counts"]["findings"] == 1
    verdicts = {item["claim"]: item["verdict"] for item in output["claims"]}
    assert verdicts["Slack delivery is live"] == "mock_only"
    assert verdicts["Local runner is live"] == "verified"
    assert verdicts["Everything is production ready"] == "unverified"
    assert verdicts["The auditor can read real registry evidence"] == "verified"
    assert verdicts["Release audit is implemented and documented"] == "verified"
    finding = output["findings"][0]
    assert finding["kind"] == "mock_only_presented_as_live"
    assert finding["integration"] == "slack" and finding["severity"] == "high"
    assert "mock-tested" in finding["reason"]
    sources = {item["source"] for item in result.evidence}
    assert "code:leo/agents/specs/release_auditor.py" in sources
    assert "test:tests/test_agents_release_auditor.py" in sources
    assert "doc:LEO_SECURITY.md" in sources
    assert "tool_registry" in sources and "agents_table" in sources
    assert "db:integrations:slack" in sources

    # The audit itself is read-only: no tool calls, agents or status rows mutated.
    assert await db.fetchall("SELECT id FROM agents") == registry_before
    assert await db.fetchall("SELECT name,status,verification FROM integrations ORDER BY name") == integrations_before
    assert await db.fetchall("SELECT id FROM tool_calls") == tool_calls_before


@pytest.mark.asyncio
async def test_handler_without_services_never_fabricates_success():
    handler = subject.make_handler()
    result = await handler("All channel integrations are live", context={})
    assert result["success"] is True
    output = result["output"]
    assert output["readiness"] == "not_ready" and output["ready"] is False
    assert output["counts"]["unverified"] == 1
    assert output["claims"][0]["reasons"] == ["no evidence supplied; claim cannot be verified"]
    assert result["evidence"] == []
