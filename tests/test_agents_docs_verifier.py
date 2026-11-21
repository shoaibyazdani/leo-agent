"""Docs-verifier spec: shape, exact grants and real claim verification."""

from __future__ import annotations

from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.specs.docs_verifier import SEED, make_handler
from leo.events import EventBus
from leo.skills.registry import SkillRegistry
from leo.tools.registry import ToolRegistry

EXPECTED_KEYS = set(SPECIALIST_SPECS[0])


def _live_registry() -> ToolRegistry:
    from leo.tools.builtins import register_builtins
    from leo.tools.file_tools import register as register_file_tools

    registry = ToolRegistry()
    register_builtins(registry)
    register_file_tools(registry)
    return registry


def test_seed_shape_and_handler_metadata():
    assert set(SEED) == EXPECTED_KEYS
    assert SEED["id"] == "docs-verifier"
    assert SEED["name"] and SEED["purpose"]
    assert all(isinstance(item, str) and item for item in SEED["capabilities"])
    assert SEED["model_preferences"]
    assert SEED["context_budget"] > 0 and SEED["task_limit"] > 0
    assert SEED["metadata"] == {"handler_kind": "bespoke:make_handler"}
    assert "handler" not in SEED["metadata"]
    assert callable(make_handler)


def test_grants_are_narrow_exact_read_only_tool_names():
    specs = _live_registry().list_specs()
    names = {spec.name for spec in specs}
    categories = {spec.category for spec in specs}
    assert 0 < len(SEED["tool_permissions"]) <= 3
    assert all(grant in names for grant in SEED["tool_permissions"])
    assert not any(grant in categories for grant in SEED["tool_permissions"])
    assert "write_file" not in SEED["tool_permissions"]


async def test_claims_checked_against_real_filesystem_and_registry(tmp_path):
    (tmp_path / "README.md").write_text("# real file\n", encoding="utf-8")
    skills = SkillRegistry(events=EventBus())
    await skills.get("summarize_text")  # seed a real registered native skill

    handler = make_handler({"skills": skills, "repo_root": tmp_path})
    result = await handler(objective="verify claims", context={"claims": [
        {"kind": "file", "value": "README.md"},
        {"kind": "file", "value": "MISSING.md"},
        {"kind": "skill", "value": "summarize_text"},
        {"kind": "skill", "value": "no_such_skill"},
        {"kind": "url", "value": "https://example.invalid"},
        {"kind": "file", "value": "../outside.md"},
    ]})

    output = result["output"]
    statuses = [claim["status"] for claim in output["claims"]]
    assert statuses == ["verified", "stale", "verified", "stale", "blocked", "blocked"]
    assert output["summary"] == {"count": 6, "verified": 2, "stale": 2, "blocked": 2, "all_verified": False}
    assert output["edited"] is False
    assert output["repo_root"] == str(tmp_path.resolve())
    assert result["evidence"][1]["edited"] is False
    assert "no documentation was changed" in result["evidence"][1]["claim"]


async def test_missing_service_is_blocked_not_verified(tmp_path):
    (tmp_path / "SPEC.md").write_text("spec", encoding="utf-8")
    handler = make_handler({"repo_root": tmp_path, "events": EventBus()})
    result = await handler(objective="verify", context={"claims": [
        {"kind": "file", "value": "SPEC.md"},
        {"kind": "skill", "value": "summarize_text"},
    ]})
    statuses = [claim["status"] for claim in result["output"]["claims"]]
    assert statuses == ["verified", "blocked"]
    assert "not injected" in result["output"]["claims"][1]["detail"]
