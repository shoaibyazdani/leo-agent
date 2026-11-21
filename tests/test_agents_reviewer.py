"""Regression tests for the read-only reviewer specialist."""

from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.agents import AgentEngine, AgentRegistry
from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.specs.reviewer import SEED, make_handler
from leo.agents.verification import digest as evidence_digest
from leo.config import Settings
from leo.contracts import AgentSpec, RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, register_builtins
from leo import tools as tools_package
from leo.tools import operator_tools

_WRITE_PERMISSIONS = {
    "file:write", "terminal:execute", "network:write", "git:write",
    "process:execute", "computer:control", "channel:send",
}


@pytest.fixture
async def reviewer_registry():
    registry = AgentRegistry()
    await registry.initialize()
    await registry.register(AgentSpec.model_validate(SEED), handler=make_handler({}))
    return registry


@pytest.fixture
def tool_env(tmp_path: Path):
    """A real tool registry/executor wired exactly like the composition root."""
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    tools = ToolRegistry()
    register_builtins(tools)
    operator_tools.register_operator_tools(tools, settings, permissions)
    # Mirror the composition root: import every registered toolset module so
    # grant checks and the specialist loader see the live tool inventory.
    for info in sorted(pkgutil.iter_modules(tools_package.__path__), key=lambda item: item.name):
        module = importlib.import_module(f"{tools_package.__name__}.{info.name}")
        register = getattr(module, "register", None)
        if callable(register):
            register(tools)
    executor = ToolExecutorImpl(tools, permissions, events, settings)
    return SimpleNamespace(
        tools=tools, executor=executor, events=events, settings=settings,
        permissions=permissions, root=tmp_path,
    )


def test_seed_shape_and_identity():
    assert set(SEED) == set(SPECIALIST_SPECS[0])  # same keys as the registry rows
    assert SEED["id"] == "reviewer"
    assert SEED["capabilities"] and SEED["model_preferences"]
    assert SEED["context_budget"] > 0 and SEED["task_limit"] > 0
    assert SEED["metadata"]["read_only"] is True
    assert SEED["metadata"]["no_edit_rights"] is True
    spec = AgentSpec.model_validate(SEED)
    assert spec.id == "reviewer" and spec.enabled is True
    assert spec.tool_permissions == SEED["tool_permissions"]


async def test_registry_loader_imports_spec_and_binds_handler(tool_env):
    registry = AgentRegistry()
    initialized = await registry.initialize(services={}, tool_registry=tool_env.tools)
    assert initialized is registry
    spec = await registry.get("reviewer")
    assert spec is not None and spec.tool_permissions == SEED["tool_permissions"]
    assert "reviewer" in registry.loaded_specialist_ids
    assert callable(registry.get_handler("reviewer"))


def test_tool_grants_resolve_to_live_read_only_tools(tool_env):
    specs = tool_env.tools.list_specs()
    by_name = {spec.name: spec for spec in specs}
    grants = SEED["tool_permissions"]
    assert len(grants) == len(set(grants))
    for grant in grants:
        matching = [spec for spec in specs if spec.name == grant or spec.category == grant]
        assert matching, f"grant {grant!r} is not registered"
        # Exact tool names, not broad categories: a category would admit writers.
        assert grant in by_name, f"grant {grant!r} is a category, not a tool"
        for spec in matching:
            assert spec.risk == RiskLevel.LOW
            assert not _WRITE_PERMISSIONS.intersection(spec.permissions or [])
    assert {"write_file", "terminal", "file_copy", "file_move", "http_get"}.isdisjoint(grants)


async def test_reviewer_registers_and_survives_reload(reviewer_registry):
    spec = await reviewer_registry.get("reviewer")
    assert spec is not None and spec.name == SEED["name"]
    assert spec.tool_permissions == SEED["tool_permissions"]
    rows = await db.fetchall("SELECT * FROM agents WHERE id='reviewer'")
    assert len(rows) == 1
    assert json.loads(rows[0]["tool_permissions"]) == SEED["tool_permissions"]

    second = AgentRegistry()
    await second.initialize()
    reloaded = await second.get("reviewer")
    assert reloaded is not None
    assert reloaded.purpose == SEED["purpose"]
    assert reloaded.tool_permissions == SEED["tool_permissions"]
    assert reloaded.metadata["read_only"] is True


async def test_scoped_facade_grants_reads_and_denies_writes(reviewer_registry, tool_env):
    engine = AgentEngine(
        reviewer_registry, tools=tool_env.executor,
        events=tool_env.events, settings=tool_env.settings,
    )
    spec = await reviewer_registry.get("reviewer")
    scoped = engine._scoped_tools(spec)
    assert {item.name for item in scoped.specs()} == set(SEED["tool_permissions"])

    denied = await scoped.execute(ToolCall(
        name="write_file",
        arguments={"path": str(tool_env.root / "never.txt"), "content": "x"},
        actor="reviewer",
    ))
    assert not denied.success
    assert denied.decision == "out_of_scope"
    rows = await db.fetchall("SELECT tool, decision FROM tool_calls WHERE tool='write_file'")
    assert any(row["decision"] == "out_of_scope" for row in rows)
    assert not (tool_env.root / "never.txt").exists()


async def test_delegated_review_returns_evidence_backed_findings(reviewer_registry, tool_env):
    target = tool_env.root / "candidate.py"
    target.write_text(
        "API_KEY = 'supersecretvalue'\n# TODO: clean this up\nprint('debug')\n",
        encoding="utf-8",
    )
    engine = AgentEngine(
        reviewer_registry, tools=tool_env.executor,
        events=tool_env.events, settings=tool_env.settings,
    )
    engine.set_handler("reviewer", make_handler({}))

    result = await engine.delegate(
        "Review candidate.py without editing it",
        agent_id="reviewer",
        context={"paths": [str(target)]},
        verify=True,
    )

    assert result.success, result.error
    assert result.metadata["verification"]["verified"] is True
    output = result.output
    assert output["status"] == "reviewed" and output["read_only"] is True
    # The executor redacts secret-like lines from read_file output, so those
    # two markers are what a real file review can legitimately report.
    assert {item["category"] for item in output["findings"]} >= {"todo", "debug_print"}
    assert all(
        Path(item["source"].split("read_file:", 1)[1]).resolve() == target.resolve()
        for item in output["findings"]
    )
    assert all(item["line"] >= 1 and item["excerpt"] for item in output["findings"])

    # Every item is either an audited tool call with a matching content digest
    # or a content-addressed artifact; no self-attested claim is accepted.
    assert result.evidence
    for item in result.evidence:
        if item.get("source") == "tool":
            anchor = item["tool_call"]
            assert anchor["tool"] == "read_file"
            row = await db.fetchone("SELECT args FROM tool_calls WHERE id=?", (anchor["id"],))
            assert row is not None
            assert json.loads(row["args"]) == anchor["arguments"]
            assert evidence_digest(anchor["output"]) == anchor["digest"]
            assert item["claim"] in output["evidence_claims"]
        else:
            assert item["source"] == "artifact"
            assert item["claim"] == item["digest"]
            assert item["digest"] in output["artifact_digests"]
    assert any(item.get("type") == "code_review" for item in result.evidence)
    assert any(item.get("source") == "artifact" for item in result.evidence)
    checks = result.metadata["verification"]["checks"]
    assert checks["tool_calls_confirmed"] is True
    assert checks["evidence_digest_valid"] is True
    assert checks["artifact_digests_valid"] is True
    assert checks["evidence_binds_output"] is True

    calls = await db.fetchall("SELECT tool, decision FROM tool_calls WHERE tool='read_file'")
    assert calls and all(row["decision"] == "allowed" for row in calls)
    assert target.read_text(encoding="utf-8").startswith("API_KEY")  # reviewed, not edited


async def test_file_backed_diff_review_and_honest_blocked_status(tool_env):
    handler = make_handler({})
    # read_file output is redacted for assignment-shaped secrets, so use
    # markers that survive the executor's redaction pass.
    diff = "@@ -1,2 +1,3 @@\n def f():\n+    return eval(user_input)\n+    # FIXME: temporary\n"
    patch = tool_env.root / "candidate.diff"
    patch.write_text(diff, encoding="utf-8")

    raw = await handler("review patch", {
        "diff_path": str(patch), "tools": tool_env.executor,
    })
    assert raw["output"]["status"] == "reviewed" and raw["output"]["read_only"] is True
    assert {item["category"] for item in raw["output"]["findings"]} >= {"dynamic_exec", "todo"}
    assert all(item["source"] == f"diff:{patch}" for item in raw["output"]["findings"])
    assert any(item.get("source") == "tool" for item in raw["evidence"])
    assert any(item.get("source") == "artifact" for item in raw["evidence"])

    # Inline text without a file or an audited read cannot be verified.
    inline = await handler("review patch", {"diff": diff, "tools": None})
    assert inline["success"] is False
    assert "content-addressed artifact" in inline["error"]
    assert inline["evidence"] == []

    blocked = await handler("review nothing", {})
    assert blocked["success"] is False
    assert blocked["output"]["status"] == "blocked"
    assert blocked["output"]["findings"] == []
    assert blocked["evidence"] == []
