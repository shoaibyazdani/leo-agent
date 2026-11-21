"""Data-analyst spec: shape, exact grants and real local analysis."""

from __future__ import annotations

import pytest

from leo.agents.registry import SPECIALIST_SPECS, AgentRegistry
from leo.agents.specs.data_analyst import SEED, make_handler
from leo.tools.registry import ToolRegistry

EXPECTED_KEYS = set(SPECIALIST_SPECS[0])


def _live_registry() -> ToolRegistry:
    from leo.tools.builtins import register_builtins
    from leo.tools.data_tools import register as register_data_tools

    registry = ToolRegistry()
    register_builtins(registry)
    register_data_tools(registry)
    return registry


def test_seed_shape_and_handler_metadata():
    assert set(SEED) == EXPECTED_KEYS
    assert SEED["id"] == "data-analyst"
    assert SEED["name"] and SEED["purpose"]
    assert all(isinstance(item, str) and item for item in SEED["capabilities"])
    assert all(isinstance(item, str) and item for item in SEED["model_preferences"])
    assert SEED["context_budget"] > 0 and SEED["task_limit"] > 0
    assert SEED["metadata"] == {"handler_kind": "bespoke:make_handler"}
    assert "handler" not in SEED["metadata"]
    assert callable(make_handler)


def test_grants_are_narrow_exact_live_tool_names():
    specs = _live_registry().list_specs()
    names = {spec.name for spec in specs}
    categories = {spec.category for spec in specs}
    assert 0 < len(SEED["tool_permissions"]) <= 3
    assert all(grant in names for grant in SEED["tool_permissions"])
    assert not any(grant in categories for grant in SEED["tool_permissions"])
    assert "not_a_real_tool" not in names


async def test_real_profiling_of_records_and_csv():
    handler = make_handler({})
    result = await handler(
        objective="profile the rows",
        context={"data": [
            {"n": 1, "tag": "a"},
            {"n": 3, "tag": "b"},
            {"n": None, "tag": "a"},
        ]},
    )
    output = result["output"]
    assert output["row_count"] == 3
    assert output["field_count"] == 2
    numbers = output["report"]["fields"]["n"]
    assert numbers["numeric_count"] == 2
    assert numbers["min"] == 1 and numbers["max"] == 3 and numbers["mean"] == 2.0
    assert numbers["non_empty"] == 2 and numbers["missing"] == 0
    assert result["evidence"] and result["evidence"][0]["row_count"] == 3

    csv_result = await handler(
        objective="profile",
        context={"data": "n,tag\n1,a\n2,b\n", "format": "csv"},
    )
    assert csv_result["output"]["row_count"] == 2
    assert csv_result["output"]["report"]["fields"]["n"]["mean"] == 1.5


async def test_missing_or_malformed_data_fails_closed():
    with pytest.raises(ValueError, match="data"):
        await make_handler({})(objective="profile nothing")
    with pytest.raises(ValueError, match="data"):
        await make_handler({})(objective="profile", context={"data": []})


def _full_registry() -> ToolRegistry:
    """Every shipped toolset, as the app composes it, for grant validation."""
    from leo.config import Settings
    from leo.events import EventBus
    from leo.security.permissions import PermissionManagerImpl
    from leo.tools import register_all_toolsets
    from leo.tools.builtins import register_builtins

    registry = ToolRegistry()
    register_builtins(registry)
    settings = Settings()
    register_all_toolsets(
        registry, {}, settings=settings,
        permissions=PermissionManagerImpl(settings, EventBus()),
    )
    return registry


async def test_registry_autoload_binds_callable_handler():
    registry = await AgentRegistry().initialize(services={}, tool_registry=_full_registry())
    spec = await registry.get("data-analyst")
    assert spec is not None
    assert spec.tool_permissions == SEED["tool_permissions"]
    assert spec.metadata == SEED["metadata"]
    assert callable(registry.get_handler("data-analyst"))
