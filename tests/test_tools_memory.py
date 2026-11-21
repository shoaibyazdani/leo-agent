"""Memory toolset tests: real SQLite persistence, recall, deletion and validation.

Every tool is executed through ``ToolExecutorImpl`` on the isolated temp DB
supplied by the shared ``isolated_db`` fixture. No memory behaviour is mocked;
the only stub is a counting proxy used to prove injected services are honoured.
Deletion is HIGH risk: the regression tests exercise argument-bound approval
and the operator-only resolution path rather than bypassing permissions.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall
from leo.events import EventBus
from leo.memory.search import MemorySearch
from leo.memory.store import MemoryStore
from leo.security.permissions import (
    ApprovalRequired,
    PermissionManagerImpl,
    arguments_digest,
)
from leo.tools import ToolExecutorImpl, ToolRegistry, memory_tools

EXPECTED = {"memory_add", "memory_search", "memory_list", "memory_forget"}


def _env() -> SimpleNamespace:
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = memory_tools.register(registry)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        registry=registry, executor=executor, names=names,
        permissions=permissions, events=events, settings=settings,
    )


async def _run(env: SimpleNamespace, name: str, arguments: dict, **kwargs) -> dict:
    result = await env.executor.execute(ToolCall(name=name, arguments=arguments, **kwargs))
    assert result.success, f"{name} failed: {result.error} (decision={result.decision})"
    assert result.decision in {"allowed", "dry_run"}
    return result.output


async def _request_forget(env: SimpleNamespace, memory_id: str) -> str:
    """A model-chosen forget must stop at approval and never reach the handler."""
    with pytest.raises(ApprovalRequired) as exc:
        await env.executor.execute(ToolCall(
            name="memory_forget", arguments={"id": memory_id}
        ))
    assert exc.value.approval_id
    return exc.value.approval_id


async def _approved_forget(env: SimpleNamespace, memory_id: str, approval_id: str) -> dict:
    result = await env.executor.execute(
        ToolCall(name="memory_forget", arguments={"id": memory_id}),
        approval_id=approval_id,
    )
    assert result.success, f"approved forget failed: {result.error} ({result.decision})"
    return result.output


async def test_register_exposes_four_typed_specs() -> None:
    env = _env()
    assert set(env.names) == EXPECTED
    assert len(env.names) == len(set(env.names))
    assert {spec.name for spec in env.registry.list_specs()} == EXPECTED
    risks = {
        "memory_add": "medium",
        "memory_search": "low",
        "memory_list": "low",
        "memory_forget": "high",  # destructive: operator approval required
    }
    for spec in env.registry.list_specs():
        assert spec.category == "memory"
        assert spec.risk.value == risks[spec.name]
        assert spec.permissions and all(perm.startswith("memory:") for perm in spec.permissions)
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert set(spec.input_schema["properties"])
        assert spec.input_schema["required"] == (
            ["content"] if spec.name == "memory_add" else
            ["query"] if spec.name == "memory_search" else
            ["id"] if spec.name == "memory_forget" else []
        )
        assert spec.resource_field == ""
    # Repeated registration is idempotent and never clobbers an existing tool.
    assert memory_tools.register(env.registry) == []


async def test_add_persists_and_list_returns_real_rows() -> None:
    env = _env()
    first = await _run(env, "memory_add", {
        "content": "Deployment runbook: rotate the staging token",
        "category": "procedural", "kind": "fact", "tags": ["ops", "runbook"],
        "confidence": 0.8,
    })
    second = await _run(env, "memory_add", {
        "content": "Personal note about hiking in Kyoto",
        "category": "personal", "tags": ["travel"],
    })
    memory_id = first["id"]
    assert memory_id and first["record"]["content"].startswith("Deployment")

    row = await db.fetchone("SELECT content, category, tags FROM memory_items WHERE id=?", (memory_id,))
    assert row is not None
    assert row["content"] == "Deployment runbook: rotate the staging token"
    assert row["category"] == "procedural"
    assert "runbook" in row["tags"]

    listed = await _run(env, "memory_list", {"category": "procedural"})
    assert [item["id"] for item in listed["records"]] == [memory_id]
    tagged = await _run(env, "memory_list", {"tags": ["travel"], "limit": 5})
    assert [item["id"] for item in tagged["records"]] == [second["id"]]
    all_rows = await _run(env, "memory_list", {})
    assert {item["id"] for item in all_rows["records"]} == {memory_id, second["id"]}


async def test_search_recalls_stored_content() -> None:
    env = _env()
    target = await _run(env, "memory_add", {
        "content": "Project Apollo launch checklist depends on kubernetes staging",
        "category": "project", "tags": ["apollo", "launch"],
    })
    await _run(env, "memory_add", {
        "content": "Dinner recipe: tomato pasta with basil",
        "category": "personal", "tags": ["cooking"],
    })
    found = await _run(env, "memory_search", {
        "query": "apollo launch checklist", "category": "project", "limit": 5,
    })
    assert found["count"] == 1
    assert found["results"][0]["id"] == target["id"]
    assert "Apollo" in found["results"][0]["content"]
    assert found["results"][0]["score"] > 0

    unrestricted = await _run(env, "memory_search", {"query": "kubernetes staging"})
    assert unrestricted["results"][0]["id"] == target["id"]
    # The hash-embedding fallback can give unrelated records a small positive
    # semantic score, so assert the category filter is applied, not that a
    # filtered low-score query is necessarily empty.
    filtered = await _run(env, "memory_search", {"query": "kubernetes", "category": "personal"})
    assert all(item["category"] == "personal" for item in filtered["results"])
    assert target["id"] not in [item["id"] for item in filtered["results"]]
    tagged = await _run(env, "memory_search", {"query": "launch", "tags": ["apollo"]})
    assert [item["id"] for item in tagged["results"]] == [target["id"]]


async def test_forget_deletes_only_the_requested_record_after_approval() -> None:
    env = _env()
    keep = await _run(env, "memory_add", {"content": "Keep this memory"})
    target = await _run(env, "memory_add", {"content": "Forget this memory"})

    approval_id = await _request_forget(env, target["id"])
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (target["id"],)) is not None
    decision = await env.permissions.resolve_approval(approval_id, True, resolver="operator:cli")
    assert decision.allowed is True
    removed = await _approved_forget(env, target["id"], approval_id)
    assert removed == {"success": True, "id": target["id"], "deleted": True}

    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (target["id"],)) is None
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (keep["id"],)) is not None
    listed = await _run(env, "memory_list", {})
    assert {item["id"] for item in listed["records"]} == {keep["id"]}

    # An already-deleted id still needs its own fresh operator approval.
    second = await _request_forget(env, target["id"])
    await env.permissions.resolve_approval(second, True, resolver="operator:cli")
    again = await _approved_forget(env, target["id"], second)
    assert again["deleted"] is False


async def test_forget_approval_is_argument_bound_and_operator_only() -> None:
    """A model-chosen delete cannot self-approve or reuse another id's approval."""
    env = _env()
    first = await _run(env, "memory_add", {"content": "First memory to delete"})
    second = await _run(env, "memory_add", {"content": "Second memory to delete"})

    approval_id = await _request_forget(env, first["id"])
    pending = {row["id"]: row for row in await env.permissions.pending_approvals()}
    row = pending[approval_id]
    assert row["action"] == "tool:memory_forget"
    assert row["risk"] == "high"
    assert row["resource"] is None
    assert row["status"] == "pending"
    metadata = json.loads(row["metadata"] or "{}")
    assert metadata["arguments_digest"] == arguments_digest({"id": first["id"]})

    # Chat actors are the requesting principal, never the human gate.
    with pytest.raises(PermissionError, match="chat actors cannot resolve"):
        await env.permissions.resolve_approval(approval_id, True, resolver="chat:assistant")
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (first["id"],)) is not None

    # Only the operator path can grant it.
    await env.permissions.resolve_approval(approval_id, True, resolver="operator:cli")

    # The approval is bound to the exact id: replaying it for another memory
    # is denied outright, even though the approval itself is "approved".
    replay = await env.executor.execute(
        ToolCall(name="memory_forget", arguments={"id": second["id"]}),
        approval_id=approval_id,
    )
    assert replay.success is False and replay.decision == "denied"
    assert "approval id is not valid" in (replay.error or "")
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (second["id"],)) is not None
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (first["id"],)) is not None

    # The approved call for the exact bound arguments does delete.
    assert (await _approved_forget(env, first["id"], approval_id))["deleted"] is True
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (first["id"],)) is None
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (second["id"],)) is not None


async def test_validation_errors_are_reported_and_do_not_write() -> None:
    env = _env()
    cases = [
        ("memory_add", {"content": ""}, "content"),
        ("memory_add", {"content": "   "}, "content must be a non-empty string"),
        ("memory_add", {"content": "x", "category": "bogus"}, "category"),
        ("memory_add", {"content": "x", "kind": "bogus"}, "kind"),
        ("memory_add", {"content": 42}, "content"),
        ("memory_search", {"query": "  "}, "query must be a non-empty string"),
        ("memory_search", {"query": "x", "category": "bogus"}, "category"),
        ("memory_list", {"category": "bogus"}, "category"),
        ("memory_forget", {}, "id"),
    ]
    for name, arguments, fragment in cases:
        result = await env.executor.execute(ToolCall(name=name, arguments=arguments))
        assert result.success is False, (name, arguments, result.output)
        assert result.decision == "invalid_arguments" or "ValueError" in (result.error or ""), result.error
        assert fragment in (result.error or ""), (name, result.error)

    rows = await db.fetchall("SELECT id, content FROM memory_items")
    assert rows == []

    # Handler-level guards also hold for callers that bypass schema validation.
    ctx = {"services": {}}
    with pytest.raises(ValueError, match="content must be a non-empty string"):
        await memory_tools.memory_add({"content": "  "}, ctx)
    with pytest.raises(ValueError, match="invalid category"):
        await memory_tools.memory_add({"content": "x", "category": "bogus"}, ctx)
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        await memory_tools.memory_forget({"id": "  "}, ctx)


async def test_injected_services_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the executor injects services, handlers use them (not a fallback)."""
    env = _env()
    real_store = MemoryStore()
    real_search = MemorySearch(real_store)
    calls: dict[str, int] = {"add": 0, "list": 0, "delete": 0, "search": 0}

    class CountingStore:
        async def add(self, record):
            calls["add"] += 1
            return await real_store.add(record)

        async def list(self, **kwargs):
            calls["list"] += 1
            return await real_store.list(**kwargs)

        async def delete(self, memory_id):
            calls["delete"] += 1
            return await real_store.delete(memory_id)

    class CountingSearch:
        async def search(self, query, **kwargs):
            calls["search"] += 1
            return await real_search.search(query, **kwargs)

    injected = {
        "services": {
            "memory_store": CountingStore(),
            "memory_search": CountingSearch(),
        }
    }
    original_context = ToolExecutorImpl._context

    def with_services(self, call, task_key, dry_run):
        return {**original_context(self, call, task_key, dry_run), **injected}

    monkeypatch.setattr(ToolExecutorImpl, "_context", with_services)

    stored = await _run(env, "memory_add", {"content": "Injected backend record"})
    assert calls["add"] == 1
    assert await db.fetchone("SELECT id FROM memory_items WHERE id=?", (stored["id"],)) is not None
    found = await _run(env, "memory_search", {"query": "Injected backend"})
    assert calls["search"] == 1 and found["count"] == 1
    listed = await _run(env, "memory_list", {})
    assert calls["list"] == 1 and listed["count"] == 1
    approval_id = await _request_forget(env, stored["id"])
    await env.permissions.resolve_approval(approval_id, True, resolver="operator:cli")
    removed = await _approved_forget(env, stored["id"], approval_id)
    assert calls["delete"] == 1 and removed["deleted"] is True
