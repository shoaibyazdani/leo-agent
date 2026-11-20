"""Incident-triage spec: shape, exact grants and real durable-signal triage."""

from __future__ import annotations

from leo import db
from leo.agents.registry import SPECIALIST_SPECS
from leo.agents.specs.incident_triage import SEED, make_handler
from leo.events import EventBus
from leo.tools.registry import ToolRegistry

EXPECTED_KEYS = set(SPECIALIST_SPECS[0])


def _live_registry() -> ToolRegistry:
    from leo.tools.event_tools import register as register_event_tools

    registry = ToolRegistry()
    register_event_tools(registry)
    return registry


def test_seed_shape_and_handler_metadata():
    assert set(SEED) == EXPECTED_KEYS
    assert SEED["id"] == "incident-triage"
    assert SEED["name"] and SEED["purpose"]
    assert all(isinstance(item, str) and item for item in SEED["capabilities"])
    assert SEED["model_preferences"]
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
    assert "approval_resolve" not in names


async def test_no_signals_is_reported_truthfully():
    result = await make_handler({})(objective="triage")
    output = result["output"]
    assert output["severity"] == "none"
    assert output["status"] == "no-signals"
    assert output["summary"]["signal_count"] == 0
    assert output["actions_taken"] == []
    assert output["emitted_triage_event"] is False


async def test_triage_reads_real_events_and_audit_rows():
    events = EventBus()
    await events.emit("terminal.failed", source="test-fixture", error="exit 2")
    await events.emit("task.completed", source="test-fixture", success=True)
    await db.execute(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
        "VALUES(?,?,?,?,?,?,?,?)",
        ("2026-01-01T00:00:00+00:00", "test", "terminal", "cmd", "high", "denied", "not allowed", "{}"),
    )

    result = await make_handler({"events": events})(objective="triage")
    output = result["output"]
    assert output["severity"] == "critical"
    assert output["status"] == "signals-found"
    assert output["summary"]["signal_count"] == 2
    assert output["summary"]["error_count"] == 1
    assert output["summary"]["warning_count"] == 0
    assert output["summary"]["non_allowed_count"] == 1
    assert any(cause["kind"] == "event_topic" and cause["value"] == "terminal.failed"
               for cause in output["summary"]["candidate_causes"])
    assert any(cause["kind"] == "denied_action" for cause in output["summary"]["candidate_causes"])
    assert output["emitted_triage_event"] is True
    assert output["actions_taken"] == []
    rows = await db.fetchall("SELECT topic FROM events WHERE topic='incident.triaged'")
    assert len(rows) == 1
    assert result["evidence"][0]["signal_count"] == 2


async def test_topic_filter_is_applied_to_durable_rows():
    events = EventBus()
    await events.emit("tool.denied", source="test-fixture", reason="scope")
    await events.emit("something.else", source="test-fixture")
    result = await make_handler({})(objective="triage", context={"topic": "tool.denied"})
    assert result["output"]["summary"]["signal_count"] == 1
    assert result["output"]["signals"][0]["level"] == "warning"
