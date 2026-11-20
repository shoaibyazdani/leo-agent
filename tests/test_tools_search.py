"""Unified-search toolset tests: real durable rows through ToolExecutorImpl.

Every result is produced by a real MemorySearch / ToolRegistry / SkillRegistry /
DurableTaskEngine / EventBus; the tests only seed the shared temp database and
assert the exact source-labelled, bounded output plus unavailable-service errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import MemoryRecord, RiskLevel, SkillManifest, ToolCall, ToolSpec
from leo.events import Event, EventBus
from leo.memory.search import MemorySearch
from leo.memory.store import MemoryStore
from leo.security.permissions import PermissionManagerImpl
from leo.skills.registry import SkillRegistry
from leo.tasks.engine import DurableTaskEngine
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import search_tools

EXPECTED = ("search_all", "search_recent_events", "find_related")
QUERY = "qzephyr"


async def _noop(args: dict, ctx: dict) -> None:
    return None


def _tool_spec(name: str, description: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        category="search",
        risk=RiskLevel.LOW,
        permissions=["search:read"],
        input_schema={"type": "object", "properties": {}},
        resource_field="",
    )


async def _seed_memory(store: MemoryStore) -> None:
    await store.add(
        MemoryRecord(
            id="mem-alpha", content="qzephyr memory note",
            tags=["project-x", "alpha"], confidence=0.9,
        )
    )
    for memory_id, tag in (
        ("mem-beta", "beta"),
        ("mem-gamma", "gamma"),
        ("mem-delta", "delta"),
    ):
        await store.add(
            MemoryRecord(
                id=memory_id, content=f"{tag} memory note",
                tags=["project-x", tag], confidence=0.6,
            )
        )
    await store.add(MemoryRecord(id="mem-other", content="tango memory note", tags=["tango"]))
    await store.link("mem-gamma", "mem-alpha", "supports", 0.9)
    await store.link("mem-alpha", "mem-beta", "relates", 0.4)


async def _seed_skills(skills: SkillRegistry) -> None:
    await skills.register(
        SkillManifest(
            name="qzephyr-skill",
            description="qzephyr skill manifest",
            category="search",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            documentation="qzephyr documentation",
        )
    )


async def _seed_tasks() -> None:
    for task_id, objective in (
        ("task-a", "qzephyr alpha task"),
        ("task-b", "qzephyr beta task"),
        ("task-other", "tango unrelated task"),
    ):
        await db.execute(
            "INSERT INTO tasks(id,objective,status,created_at,updated_at) VALUES(?,?,?,?,?)",
            (task_id, objective, "created", "2026-01-01T00:00:01+00:00", "2026-01-01T00:00:01+00:00"),
        )


async def _seed_events() -> None:
    rows = [
        ("seed-1", "search.seed", {"needle": "alpha"}, "2026-01-01T00:00:01+00:00", None),
        ("seed-2", "search.seed", {"needle": "beta"}, "2026-01-01T00:00:02+00:00", None),
        ("seed-3", "search.seed", {"needle": "alpha"}, "2026-01-01T00:00:03+00:00", "corr-x"),
        ("seed-4", "search.other", {"needle": "alpha"}, "2026-01-01T00:00:04+00:00", None),
    ]
    await db.executemany(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) VALUES(?,?,?,?,?,?)",
        [
            (
                event.id, event.topic, event.source, event.model_dump_json(),
                event.correlation_id, event.created_at,
            )
            for event in (
                Event(
                    id=i, topic=topic, source="seed", payload=payload,
                    correlation_id=corr, created_at=ts,
                )
                for i, topic, payload, ts, corr in rows
            )
        ],
    )


async def _build(tmp_path, *, with_services: bool) -> SimpleNamespace:
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    store = MemoryStore(events)
    search = MemorySearch(store)
    skills = SkillRegistry(events=events)
    engine = DurableTaskEngine(settings, events)
    registry = ToolRegistry()
    registry.register(_tool_spec("qzephyr-tool-a", "qzephyr alpha tool"), _noop)
    registry.register(_tool_spec("qzephyr-tool-b", "qzephyr beta tool"), _noop)

    await _seed_memory(store)
    await _seed_skills(skills)
    await _seed_tasks()
    await _seed_events()

    services = {
        "events": events,
        "memory": {"store": store, "search": search},
        "memory_store": store,
        "memory_search": search,
        "skills": skills,
        "tasks": engine,
        "tools": registry,
    }
    names = search_tools.register(registry, services if with_services else None)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        store=store, search=search, skills=skills, engine=engine,
        registry=registry, executor=executor, services=services, names=names,
    )


@pytest.fixture
async def env(tmp_path):
    return await _build(tmp_path, with_services=True)


@pytest.fixture
async def bare_env(tmp_path):
    return await _build(tmp_path, with_services=False)


async def _run(env, name: str, arguments: dict | None = None, **kwargs):
    return await env.executor.execute(
        ToolCall(name=name, arguments=arguments or {}, **kwargs)
    )


async def _ok(env, name: str, arguments: dict | None = None, **kwargs) -> dict:
    result = await _run(env, name, arguments, **kwargs)
    assert result.success, f"{name} failed: {result.error} (decision={result.decision})"
    assert result.decision == "allowed", result.decision
    return result.output


# ------------------------------------------------------------------ registry


async def test_register_typed_specs(env) -> None:
    assert env.names == list(EXPECTED)
    assert [spec.name for spec in env.registry.list_specs()][-3:] == list(EXPECTED)
    assert search_tools.register(env.registry) == []  # idempotent, never clobbers

    for name in EXPECTED:
        spec = env.registry.get(name).spec
        assert spec.category == "search"
        assert spec.risk is RiskLevel.LOW
        assert "search:read" in spec.permissions
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.input_schema["properties"]
        assert spec.resource_field == ""  # logical records, not file paths

    assert env.registry.get("search_all").spec.input_schema["required"] == ["query"]
    assert env.registry.get("find_related").spec.input_schema["required"] == ["id"]
    assert env.registry.get("search_all").spec.input_schema["properties"]["sources"][
        "items"
    ]["enum"] == list(search_tools.SOURCE_ORDER)


# ----------------------------------------------------------------- search_all


async def test_search_all_merges_exact_source_labelled_results(env) -> None:
    out = await _ok(env, "search_all", {"query": QUERY, "limit": 6})
    assert out["source"] == "unified"
    assert out["query"] == QUERY
    assert out["sources"] == list(search_tools.SOURCE_ORDER)
    assert out["limit"] == 6
    assert out["count"] == 6
    assert out["by_source"] == {"memory": 1, "tool": 2, "skill": 1, "task": 2}
    assert [(row["source"], row["id"]) for row in out["results"]] == [
        ("memory", "mem-alpha"),
        ("tool", "qzephyr-tool-a"),
        ("tool", "qzephyr-tool-b"),
        ("skill", "qzephyr-skill"),
        ("task", "task-a"),
        ("task", "task-b"),
    ]
    memory_hit = out["results"][0]
    assert memory_hit["title"] == "qzephyr memory note"
    assert "project-x" in memory_hit["snippet"]
    assert memory_hit["score"] > 0
    assert out["results"][1]["snippet"] == "qzephyr alpha tool"

    # The memory hit comes from an actual MemorySearch over stored rows.
    row = await db.fetchone("SELECT content FROM memory_items WHERE id='mem-alpha'")
    assert row["content"] == memory_hit["title"]


async def test_search_all_bounds_and_source_filter(env) -> None:
    bounded = await _ok(env, "search_all", {"query": QUERY, "limit": 3})
    assert bounded["count"] == 3
    assert len(bounded["results"]) == 3
    assert bounded["by_source"] == {"memory": 1, "tool": 2, "skill": 0, "task": 0}
    assert [row["id"] for row in bounded["results"]] == [
        "mem-alpha", "qzephyr-tool-a", "qzephyr-tool-b",
    ]

    only_tasks = await _ok(env, "search_all", {"query": QUERY, "sources": ["task"], "limit": 9})
    assert only_tasks["sources"] == ["task"]
    assert only_tasks["by_source"] == {"task": 2}
    assert [(row["source"], row["id"]) for row in only_tasks["results"]] == [
        ("task", "task-a"), ("task", "task-b"),
    ]

    none = await _ok(env, "search_all", {"query": "no-such-token-anywhere", "limit": 9})
    assert none["results"] == [] and none["count"] == 0


# ---------------------------------------------------------- recent events


async def test_search_recent_events_reads_durable_history(env) -> None:
    page = await _ok(env, "search_recent_events", {"topic": "search.seed", "limit": 2})
    assert page["source"] == "events"
    assert page["limit"] == 2 and page["count"] == 2
    assert [row["source"] for row in page["events"]] == ["event", "event"]
    assert [row["id"] for row in page["events"]] == ["seed-3", "seed-2"]
    assert page["events"][0]["topic"] == "search.seed"
    assert page["events"][0]["payload"] == {"needle": "alpha"}

    filtered = await _ok(
        env,
        "search_recent_events",
        {"topic": "search.seed", "query": "alpha", "limit": 10},
    )
    assert [row["id"] for row in filtered["events"]] == ["seed-3", "seed-1"]
    assert filtered["filters"] == {
        "topic": "search.seed", "query": "alpha", "correlation_id": None,
    }

    correlated = await _ok(
        env, "search_recent_events", {"topic": "search.seed", "correlation_id": "corr-x"}
    )
    assert [row["id"] for row in correlated["events"]] == ["seed-3"]

    rows = await db.fetchall("SELECT id FROM events WHERE topic='search.seed'")
    assert {row["id"] for row in rows} == {"seed-1", "seed-2", "seed-3"}


# ------------------------------------------------------------ find_related


async def test_find_related_uses_real_links_and_tags(env) -> None:
    out = await _ok(env, "find_related", {"id": "mem-alpha", "limit": 3})
    assert out["source"] == "memory_links"
    assert out["id"] == "mem-alpha"
    assert out["count"] == 3 and out["returned"] == 3
    assert [(row["source"], row["id"]) for row in out["related"]] == [
        ("link", "mem-gamma"), ("link", "mem-beta"), ("tag", "mem-delta"),
    ]
    assert out["related"][0] == {
        "source": "link", "id": "mem-gamma", "title": "gamma memory note",
        "relation": "supports", "direction": "incoming", "weight": 0.9,
        "shared_tags": [],
    }
    assert out["related"][1]["relation"] == "relates"
    assert out["related"][1]["direction"] == "outgoing"
    assert out["related"][2]["shared_tags"] == ["project-x"]
    assert "mem-other" not in [row["id"] for row in out["related"]]

    bounded = await _ok(env, "find_related", {"id": "mem-alpha", "limit": 2})
    assert bounded["count"] == 3 and bounded["returned"] == 2
    assert [row["id"] for row in bounded["related"]] == ["mem-gamma", "mem-beta"]

    links_only = await _ok(
        env, "find_related", {"id": "mem-alpha", "include_tags": False}
    )
    assert [row["source"] for row in links_only["related"]] == ["link", "link"]

    # The link evidence is the durable row, not a computed score.
    row = await db.fetchone(
        "SELECT relation,weight FROM memory_links WHERE source='mem-gamma' AND target='mem-alpha'"
    )
    assert (row["relation"], row["weight"]) == ("supports", 0.9)


async def test_find_related_unknown_id_errors(env) -> None:
    result = await _run(env, "find_related", {"id": "mem-missing"})
    assert result.success is False
    assert "unknown memory: mem-missing" in (result.error or "")


# ---------------------------------------------------------- missing services


async def test_missing_services_report_clear_errors(
    bare_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search_tools.db, "database_path_configured", lambda: False)
    cases = [
        ("search_all", {"query": QUERY}, ("memory", "initialized database")),
        ("search_recent_events", {}, ("events", "initialized database")),
        ("find_related", {"id": "mem-alpha"}, ("memory store", "initialized database")),
    ]
    for name, arguments, fragments in cases:
        result = await _run(bare_env, name, arguments)
        assert result.success is False, (name, result.output)
        for fragment in fragments:
            assert fragment in (result.error or ""), (name, result.error)


async def test_context_services_take_precedence(
    bare_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternate = ToolRegistry()
    alternate.register(_tool_spec("qzephyr-ctx-tool", "qzephyr context tool"), _noop)
    injected = dict(bare_env.services)
    injected["tools"] = alternate

    original = ToolExecutorImpl._context

    def with_services(self, call, task_key, dry_run):
        return {**original(self, call, task_key, dry_run), "services": injected}

    monkeypatch.setattr(ToolExecutorImpl, "_context", with_services)

    out = await _ok(bare_env, "search_all", {"query": QUERY, "limit": 6})
    tools = [row["id"] for row in out["results"] if row["source"] == "tool"]
    assert tools == ["qzephyr-ctx-tool"]
    assert "qzephyr-tool-a" not in [row["id"] for row in out["results"]]

    related = await _ok(bare_env, "find_related", {"id": "mem-alpha", "limit": 1})
    assert [row["id"] for row in related["related"]] == ["mem-gamma"]


# ------------------------------------------------------------- validation


async def test_schema_validation_blocks_bad_arguments(env) -> None:
    cases = [
        ("search_all", {"query": ""}, "query"),
        ("search_all", {"query": QUERY, "limit": 0}, "limit"),
        ("search_all", {"query": QUERY, "limit": 51}, "limit"),
        ("search_all", {"query": QUERY, "sources": ["bogus"]}, "sources"),
        ("search_recent_events", {"limit": 101}, "limit"),
        ("search_recent_events", {"topic": ""}, "topic"),
        ("find_related", {}, "id"),
        ("find_related", {"id": "mem-alpha", "include_tags": "yes"}, "include_tags"),
    ]
    for name, arguments, fragment in cases:
        result = await _run(env, name, arguments)
        assert result.success is False, (name, result.output)
        assert result.decision == "invalid_arguments", (name, result.decision)
        assert fragment in (result.error or ""), (name, result.error)

    # Handler-level guards also hold when a caller bypasses the registry schema.
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        await search_tools.search_all({"query": "  "}, {})
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        await search_tools.search_recent_events({"limit": 101}, {})
