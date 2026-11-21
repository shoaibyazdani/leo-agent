"""Browser-tester specialist: seed shape, real browser tool grants and a
delegated UI check driven through the real tool executor with a fake backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leo import db
from leo.agents import AgentEngine, AgentRegistry
from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.specs.browser_tester import SEED, make_handler
from leo.agents.verification import digest
from leo.config import Settings
from leo.contracts import AgentSpec
from leo.events import EventBus
from leo.integrations.browser import BrowserOperator, FakeBrowserPage
from leo.security import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import operator_tools

URL = "https://93.184.216.34/check"
PAGE_TEXT = "Login successful. Welcome, tester."


def test_seed_shape_matches_registry_rows() -> None:
    expected_keys = set(SPECIALIST_SPECS[0])
    assert expected_keys == {
        "id", "name", "purpose", "capabilities", "tool_permissions",
        "model_preferences", "context_budget", "task_limit", "metadata",
    }
    assert set(SEED) == expected_keys
    assert SEED["id"] == "browser-tester"
    assert SEED["name"] and SEED["purpose"] and SEED["capabilities"]
    assert SEED["model_preferences"] and SEED["tool_permissions"]
    assert SEED["context_budget"] > 0 and SEED["task_limit"] > 0
    assert isinstance(SEED["metadata"], dict)
    assert AgentSpec.model_validate(SEED).id == "browser-tester"


def test_tool_grants_are_real_browser_scoped_tools() -> None:
    settings = Settings()
    tool_registry = ToolRegistry()
    operator_tools.register_operator_tools(
        tool_registry, settings, PermissionManagerImpl(settings, EventBus()),
        include_computer=False,
    )
    specs = {spec.name: spec for spec in tool_registry.list_specs()}
    assert {spec.category for spec in specs.values()} == {"browser"}

    grants = set(SEED["tool_permissions"])
    assert grants, "browser tester must declare executable tool grants"
    assert grants <= set(specs), f"unknown tool grants: {grants - set(specs)}"
    assert {"browser_navigate", "browser_extract_text"} <= grants
    assert all(specs[name].category == "browser" for name in grants)
    assert not any(name.startswith("computer_") for name in grants)

    # The declared scope is exactly what the agent engine's tool façade uses.
    scope = frozenset(grants)
    assert ToolExecutorImpl._in_scope(tool_registry.get("browser_navigate"), scope)
    assert ToolExecutorImpl._in_scope(tool_registry.get("browser_extract_text"), scope)
    assert not ToolExecutorImpl._in_scope(tool_registry.get("browser_click"), scope)
    assert not ToolExecutorImpl._in_scope(tool_registry.get("browser_screenshot"), scope)


@pytest.mark.asyncio
async def test_delegated_browser_check_with_fake_backend(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    settings.security.allow_medium_risk = True
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    tool_registry = ToolRegistry()
    operator_tools.register_operator_tools(
        tool_registry, settings, permissions, include_computer=False
    )
    executor = ToolExecutorImpl(tool_registry, permissions, events, settings)

    page = FakeBrowserPage(pages={URL: PAGE_TEXT})
    operator = BrowserOperator(
        permissions, actor="browser-tester", backend=page,
        profile_dir=tmp_path / "profile", base_dir=tmp_path, timeout=1.0,
    )
    monkeypatch.setattr(operator_tools, "_browser_operator", lambda ctx: operator)

    # Register this seed directly: the registry-wide spec loader imports every
    # concurrent specialist module, which is outside this file's ownership.
    agents = AgentRegistry()
    await agents.refresh()
    await agents.seed()
    if await agents.get(SEED["id"]) is None:
        await agents.register(SEED)
    engine = AgentEngine(agents, tools=executor, events=events, settings=settings)
    engine.set_handler("browser-tester", make_handler({"browser_operator": operator, "tools": executor}))

    objective = json.dumps({
        "url": URL, "selector": "body", "assert_contains": "Login successful",
    })
    result = await engine.delegate(objective, agent_id="browser-tester")

    assert result.success, result.error
    assert result.metadata["verification"]["verified"] is True
    assert result.output["url"] == URL
    assert result.output["snapshot"] == PAGE_TEXT
    assert result.output["assertion"] == {
        "kind": "assert_contains", "expected": "Login successful", "passed": True,
    }
    assert result.output["check_passed"] is True
    assert result.output["backend"] == "FakeBrowserPage"
    assert result.output["live_browser"] is False

    assert len(result.evidence) == 3
    assert any(e.get("url") == URL for e in result.evidence)
    assert any(e.get("snapshot") == PAGE_TEXT for e in result.evidence)
    assert any(e.get("passed") is True and e.get("assertion", {}).get("passed") is True
               for e in result.evidence)
    assert all(e["live"] is False and e["backend"] == "FakeBrowserPage" for e in result.evidence)
    assert "live browser" not in json.dumps(result.evidence).lower()

    # Every claim is anchored to a successful, durably audited tool call and
    # to the digest of that call's output; no bare self-attestation is emitted.
    seen_calls: set[str] = set()
    for item in result.evidence:
        assert item["source"] == "tool"
        anchor = item["tool_call"]
        assert anchor["tool"] in {"browser_navigate", "browser_extract_text"}
        assert anchor["digest"] == digest(anchor["output"])
        assert item["digest"] == anchor["digest"]
        row = await db.fetchone("SELECT tool, args, success, decision FROM tool_calls WHERE id=?",
                                (anchor["id"],))
        assert row is not None
        assert row["tool"] == anchor["tool"]
        assert json.loads(row["args"]) == anchor["arguments"]
        assert row["success"] == 1 and row["decision"] == "allowed"
        assert item["claim"] in json.dumps(result.output, ensure_ascii=False)
        seen_calls.add(anchor["id"])
    assert len(seen_calls) == 2

    # The fake is only the page backend: real operator tools, executor and
    # delegation ran, and their audit trail is durable.
    assert [call["action"] for call in page.calls] == ["navigate", "extract_text"]
    calls = await db.fetchall("SELECT tool, decision FROM tool_calls WHERE tool LIKE 'browser_%'")
    assert {row["tool"] for row in calls} == {"browser_navigate", "browser_extract_text"}
    assert all(row["decision"] == "allowed" for row in calls)
    runs = await agents.history("browser-tester")
    assert len(runs) == 1 and runs[0]["status"] == "completed"

    # A deterministic mismatch must fail instead of being reported as a pass.
    failed = await engine.delegate(json.dumps({
        "url": URL, "selector": "body", "assert_contains": "absent text",
    }), agent_id="browser-tester")
    assert not failed.success
    assert failed.output["assertion"]["passed"] is False
    assert failed.output["live_browser"] is False
