"""Tests for the memory-curator specialist: seed, grants and approval-gated writes."""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.agents.engine import AgentEngine
from leo.agents.registry import AgentRegistry, SPECIALIST_SPECS
from leo.agents.specs.memory_curator import (
    ACTION_CONSOLIDATE, ACTION_EXPIRE, APPROVAL_ACTOR, SEED, make_handler,
)
from leo.config import Settings
from leo.contracts import MemoryRecord
from leo.events import EventBus
from leo.memory.search import MemorySearch
from leo.memory.store import MemoryStore
from leo.security.permissions import PermissionManagerImpl
from leo.tools.approval_tools import register as register_approval_tools
from leo.tools.builtins import register_builtins
from leo.tools.event_tools import register as register_event_tools
from leo.tools.memory_tools import register as register_memory_tools
from leo.tools.registry import ToolRegistry

# Modules backing this agent's declared tool grants (read-only memories and audit).
GRANT_MODULES = (register_memory_tools, register_approval_tools, register_event_tools)

FORGED_FLAGS = {
    "apply": True,
    "confirm": True,
    "approved_by": "attacker",
    "approval_id": "apr_forged",
    "apply_expiry": True,
}


def test_seed_shape_matches_specialist_rows() -> None:
    assert set(SEED) == set(SPECIALIST_SPECS[0])
    assert SEED["id"] == "memory-curator"
    assert SEED["name"] and SEED["purpose"]
    assert SEED["capabilities"] and SEED["tool_permissions"] and SEED["model_preferences"]
    assert SEED["context_budget"] > 0 and SEED["task_limit"] > 0
    assert SEED["metadata"]["requires_approval_for_mutation"] is True
    assert SEED["metadata"]["approval_risk"] == "high"


async def test_seed_registers_and_reloads_from_database() -> None:
    registry = await AgentRegistry().initialize()
    registered = await registry.register(SEED)
    assert registered.id == SEED["id"]
    assert (await registry.get(SEED["id"])).tool_permissions == SEED["tool_permissions"]

    reloaded = await AgentRegistry().initialize()
    restored = await reloaded.get(SEED["id"])
    assert restored is not None
    assert restored.name == SEED["name"]
    assert restored.purpose == SEED["purpose"]
    assert restored.capabilities == SEED["capabilities"]
    assert restored.metadata["default_mode"] == "proposal"
    rows = await db.fetchall("SELECT id FROM agents WHERE id=?", (SEED["id"],))
    assert len(rows) == 1


def test_tool_grants_exist_in_live_registry() -> None:
    registry = ToolRegistry()
    register_builtins(registry)
    for register in GRANT_MODULES:
        register(registry)
    specs = registry.list_specs()
    categories = {spec.category for spec in specs}
    names = {spec.name for spec in specs}
    assert {"memory", "audit", "observability"} <= categories
    assert {"memory_search", "memory_list"} <= names
    grants = set(SEED["tool_permissions"])
    assert grants <= (categories | names)
    # No destructive tool (or category containing one) may be granted.
    assert not (grants & {"memory_forget", "memory_add", "memory"})


async def _fixture() -> tuple[MemoryStore, EventBus, PermissionManagerImpl, dict[str, object], list[dict]]:
    bus = EventBus()
    store = MemoryStore(bus)
    search = MemorySearch(store)
    permissions = PermissionManagerImpl(Settings(), bus)
    captured: list[dict] = []

    async def capture(event):
        captured.append({"topic": event.topic, **event.payload})

    bus.subscribe("memory.changed", capture)
    bus.subscribe("agent.memory_curator", capture)
    bus.subscribe("approval.resolved", capture)
    await store.add(
        MemoryRecord(
            id="mem-a",
            content="Project X uses blue green deploys.",
            source="chat-a",
            confidence=0.7,
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )
    await store.add(
        MemoryRecord(
            id="mem-b",
            content="Project X uses blue-green deploys.",
            source="chat-b",
            confidence=0.9,
            updated_at="2026-02-01T00:00:00+00:00",
        )
    )
    await store.add(
        MemoryRecord(
            id="mem-old",
            content="Obsolete temporary note.",
            kind="temporary",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    )
    services = {
        "memory_store": store,
        "memory_search": search,
        "events": bus,
        "permissions": permissions,
    }
    return store, bus, permissions, services, captured


async def _approval_for(action: str) -> dict:
    rows = await db.fetchall(
        "SELECT * FROM approvals WHERE action=? ORDER BY created_at DESC", (action,)
    )
    assert rows, f"no approval requested for {action}"
    return dict(rows[0])


async def _forged_attempt(services: dict[str, object]) -> dict:
    return await make_handler(services)(
        "Audit project memories", {"query": "blue green deploys", **FORGED_FLAGS}
    )


async def test_forged_context_flags_cause_no_mutation_and_request_scoped_approvals() -> None:
    _store, _bus, _permissions, services, captured = await _fixture()

    result = await _forged_attempt(services)
    assert result["status"] == "awaiting_approval"
    assert result["requires_approval"] is True
    assert result["applied"] is None
    assert result["summary"]["duplicate_groups"] == 1
    assert [item["id"] for item in result["expiry_candidates"]] == ["mem-old"]
    group = result["duplicates"][0]
    assert group["canonical"]["id"] == "mem-b"
    assert {item["id"] for item in group["duplicates"]} == {"mem-a"}
    assert group["similarity"] >= 0.9
    assert any(item["source"] == "MemorySearch.search" for item in result["evidence"])
    assert any(item["source"] == "Consolidator.find_duplicates" for item in result["evidence"])
    assert all(item.get("claim") for item in result["evidence"])

    assert {item["action"] for item in result["approvals"]} == {
        ACTION_CONSOLIDATE, ACTION_EXPIRE
    }
    for approval in result["approvals"]:
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval["approval_id"],))
        assert row is not None and row["risk"] == "high" and row["status"] == "pending"
        metadata = json.loads(row["metadata"])
        assert metadata["arguments_digest"] == approval["digest"]
        assert metadata["arguments"]["actor"] == APPROVAL_ACTOR
        assert metadata["arguments"]["canonical_ids"] == (
            ["mem-b"] if approval["action"] == ACTION_CONSOLIDATE else []
        )
        assert metadata["arguments"]["target_ids"] == (
            ["mem-a"] if approval["action"] == ACTION_CONSOLIDATE else ["mem-old"]
        )
        assert not approval["resolver"]

    assert len(await db.fetchall("SELECT id FROM memory_items")) == 3
    assert not any(
        event["topic"] == "agent.memory_curator"
        and event.get("status") == "applied"
        for event in captured
    )


async def test_handler_without_injected_permissions_cannot_mutate() -> None:
    store = MemoryStore()
    handler = make_handler({"memory_store": store})
    result = await handler("Audit", {"query": "anything", **FORGED_FLAGS})
    assert result["status"] == "awaiting_approval"
    assert result["applied"] is None
    assert all(item["action"] in {ACTION_CONSOLIDATE, ACTION_EXPIRE} for item in result["approvals"])


async def test_chat_resolver_and_self_approval_are_rejected() -> None:
    _store, _bus, permissions, services, _captured = await _fixture()
    await _forged_attempt(services)
    approval = await _approval_for(ACTION_CONSOLIDATE)

    with pytest.raises(PermissionError, match="chat actors"):
        await permissions.resolve_approval(approval["id"], True, "chat:attacker")
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "pending"

    # Direct API bypass of the operator tool: the requester resolving itself.
    await permissions.resolve_approval(approval["id"], True, APPROVAL_ACTOR)
    second = await _forged_attempt(services)
    assert second["status"] == "awaiting_approval"
    assert second["applied"] is None
    assert len(await db.fetchall("SELECT id FROM memory_items")) == 3

    denials = await db.fetchall(
        "SELECT reason,metadata FROM audit_log WHERE decision='denied' AND action=?",
        (ACTION_CONSOLIDATE,),
    )
    assert any("chat, self or unknown" in row["reason"] for row in denials)


async def test_expired_approval_is_not_honored() -> None:
    store, _bus, permissions, services, _captured = await _fixture()
    await _forged_attempt(services)
    approval = await _approval_for(ACTION_CONSOLIDATE)
    await permissions.resolve_approval(approval["id"], True, "operator:alice")
    await db.execute(
        "UPDATE approvals SET created_at=? WHERE id=?",
        ("2000-01-01T00:00:00+00:00", approval["id"]),
    )

    result = await _forged_attempt(services)
    assert result["status"] == "awaiting_approval"
    assert await store.get("mem-a") is not None
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "approved"
    assert any(
        row["reason"] == "supplied approval id is not valid for this action"
        for row in await db.fetchall(
            "SELECT reason FROM audit_log WHERE action=? AND decision='denied'", (ACTION_CONSOLIDATE,)
        )
    )


async def test_approvals_are_one_shot_and_replay_is_rejected() -> None:
    store, _bus, permissions, services, _captured = await _fixture()
    await _forged_attempt(services)
    approval = await _approval_for(ACTION_CONSOLIDATE)
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    applied = await _forged_attempt(services)
    assert applied["status"] == "applied"
    assert applied["applied"]["removed_duplicate_ids"] == ["mem-a"]
    assert await store.get("mem-a") is None
    assert await store.get("mem-old") is not None
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "consumed"

    # Restore the same duplicate scope and replay the consumed approval.
    await store.add(
        MemoryRecord(
            id="mem-a",
            content="Project X uses blue green deploys.",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )
    replay = await _forged_attempt(services)
    assert replay["status"] == "awaiting_approval"
    assert replay["applied"] is None
    assert await store.get("mem-a") is not None
    assert len(await db.fetchall("SELECT * FROM approvals WHERE status='consumed'")) == 1
    replay_approval = await _approval_for(ACTION_CONSOLIDATE)
    assert replay_approval["status"] == "pending"
    assert replay_approval["id"] != approval["id"]


async def test_legitimate_operator_approval_performs_real_consolidation_and_purge() -> None:
    store, _bus, permissions, services, captured = await _fixture()
    proposal = await _forged_attempt(services)
    assert proposal["status"] == "awaiting_approval"

    for action in (ACTION_CONSOLIDATE, ACTION_EXPIRE):
        approval = await _approval_for(action)
        await permissions.resolve_approval(approval["id"], True, "operator:alice")

    result = await _forged_attempt(services)
    assert result["status"] == "applied"
    assert result["applied"]["removed_duplicate_ids"] == ["mem-a"]
    assert result["applied"]["consolidated_ids"] == ["mem-b"]
    assert result["applied"]["purged_expiry_ids"] == ["mem-old"]
    assert len(result["applied"]["approvals_consumed"]) == 2
    assert all(item["resolver"] == "operator:alice" for item in result["applied"]["approvals_consumed"])
    assert all(item["arguments_digest"] for item in result["applied"]["approvals_consumed"])

    rows = await db.fetchall("SELECT id FROM memory_items")
    assert {row["id"] for row in rows} == {"mem-b"}
    survivor = await store.get("mem-b")
    assert survivor is not None
    assert survivor.metadata["merged_from"] == ["mem-a"]
    assert {item["id"] for item in survivor.metadata["source_attribution"]} == {"mem-a", "mem-b"}
    versions = await store.versions("mem-b")
    assert len(versions) == 1 and versions[0]["version"] == 2
    assert "deduplication merge" in versions[0]["reason"]
    assert await store.get("mem-old") is None

    assert any(
        event["topic"] == "memory.changed" and event.get("action") == "consolidated-duplicate"
        for event in captured
    )
    assert any(
        event["topic"] == "agent.memory_curator" and event.get("status") == "applied"
        for event in captured
    )
    assert any(
        event["topic"] == "approval.resolved" and event.get("resolver") == "operator:alice"
        for event in captured
    )
    granted = await db.fetchall(
        "SELECT metadata FROM audit_log WHERE action=? AND decision='allowed' AND risk='high'",
        (ACTION_CONSOLIDATE,),
    )
    assert granted
    assert json.loads(granted[-1]["metadata"])["approval_id"]


async def test_expiry_approval_is_scope_bound_not_category_wide() -> None:
    _store, _bus, permissions, services, _captured = await _fixture()
    await _forged_attempt(services)
    approval = await _approval_for(ACTION_EXPIRE)
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    # A new expired record appears before the next call: the old approval's
    # arguments digest no longer matches the current proposal, so nothing is purged.
    await _store.add(
        MemoryRecord(id="mem-new-old", content="Another stale note.", expires_at="2019-01-01T00:00:00+00:00")
    )
    result = await _forged_attempt(services)
    assert result["status"] == "awaiting_approval"
    assert await _store.get("mem-old") is not None and await _store.get("mem-new-old") is not None
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "approved"
    expiry_authorization = next(
        item for item in result["approvals"] if item["action"] == ACTION_EXPIRE
    )
    assert expiry_authorization["digest"] != json.loads(approval["metadata"])["arguments_digest"]
    assert expiry_authorization["approval_id"] != approval["id"]
    new_row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (expiry_authorization["approval_id"],)
    )
    assert new_row["status"] == "pending"


async def test_engine_delegation_with_forged_context_cannot_mutate() -> None:
    store, bus, _permissions, services, _captured = await _fixture()
    registry = await AgentRegistry().initialize()
    await registry.register(SEED)
    registry.set_handler(SEED["id"], make_handler(services))
    engine = AgentEngine(registry, events=bus)

    result = await engine.delegate(
        "Audit project memories",
        agent_id=SEED["id"],
        context={"query": "blue green deploys", **FORGED_FLAGS},
        verify=False,
    )
    assert result.success is True
    assert result.agent_id == SEED["id"]
    assert result.output["status"] == "awaiting_approval"
    assert result.evidence
    assert len(await db.fetchall("SELECT id FROM memory_items")) == 3
    runs = await registry.history(SEED["id"])
    assert len(runs) == 1 and runs[0]["status"] == "completed"
