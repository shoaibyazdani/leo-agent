"""API-integrator spec: shape, exact grants and truthful posture reporting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.specs.api_integrator import SEED, make_handler
from leo.tools.registry import ToolRegistry

EXPECTED_KEYS = set(SPECIALIST_SPECS[0])


def _live_registry() -> ToolRegistry:
    from leo.tools.integration_tools import register as register_integration_tools

    registry = ToolRegistry()
    register_integration_tools(registry)
    return registry


def test_seed_shape_and_handler_metadata():
    assert set(SEED) == EXPECTED_KEYS
    assert SEED["id"] == "api-integrator"
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
    assert 0 < len(SEED["tool_permissions"]) <= 2
    assert all(grant in names for grant in SEED["tool_permissions"])
    assert not any(grant in categories for grant in SEED["tool_permissions"])
    assert "channel_send" not in SEED["tool_permissions"]


async def test_real_posture_from_local_adapter_records():
    result = await make_handler({})(objective="integration posture")
    output = result["output"]
    assert output["sent"] is False
    assert output["adapter_count"] >= 20
    assert output["status_counts"] and output["verification_counts"]
    assert len(output["adapters"]) == output["adapter_count"]
    assert all(set(("name", "status", "verification", "live_verified")) <= set(item) for item in output["adapters"])
    assert output["outbox_stats"] is None
    assert output["outbox_note"] and "not injected" in output["outbox_note"]
    assert result["evidence"][0]["adapter_count"] == output["adapter_count"]


async def test_channel_filter_and_injected_gateway_outbox():
    gateway = SimpleNamespace(outbox_stats=lambda: {"pending": 2, "failed": 0})
    handler = make_handler({"gateway": gateway})

    filtered = await handler(objective="posture for channel a2a")
    assert filtered["output"]["channel_filter"] == "a2a"
    assert len(filtered["output"]["adapters"]) == 1
    assert filtered["output"]["adapters"][0]["name"] == "a2a"

    full = await handler(objective="posture")
    assert full["output"]["outbox_stats"] == {"pending": 2, "failed": 0}
    assert full["output"]["outbox_note"] is None

    with pytest.raises(ValueError, match="unknown channel"):
        await handler(objective="posture", context={"channel": "not_a_channel"})


async def test_blocked_channels_are_not_reported_as_live():
    output = (await make_handler({})(objective="posture"))["output"]
    blocked = set(output["blocked"])
    assert blocked
    for item in output["adapters"]:
        if item["name"] in blocked:
            assert item["status"] == "blocked"
            assert item["live_verified"] is False
