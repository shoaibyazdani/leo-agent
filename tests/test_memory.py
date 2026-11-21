"""Tests for the LEO memory engine (store, retrieval, consolidation, eval)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from leo import db
from leo.contracts import MemoryCategory, MemoryKind, MemoryRecord
from leo.events import EventBus
from leo.memory import Consolidator, MemoryEvaluation, MemorySearch, MemoryStore


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    bus = EventBus()
    store = MemoryStore(bus)
    return store, MemorySearch(store), bus


def _record(**kwargs) -> MemoryRecord:
    return MemoryRecord(**kwargs)


# --------------------------------------------------------------- persistence
async def test_persistence_and_relevance_across_restart(env):
    store, search, _ = env
    saved = await store.add(
        _record(
            category=MemoryCategory.PROJECT,
            kind=MemoryKind.FACT,
            content="LEO keeps its durable state in SQLite",
            source="architecture.md",
            confidence=0.9,
            tags=["leo", "sqlite"],
        )
    )

    restarted = MemoryStore()
    loaded = await restarted.get(saved.id)
    assert loaded is not None
    assert loaded.id == saved.id
    assert loaded.category is MemoryCategory.PROJECT
    assert loaded.kind is MemoryKind.FACT
    assert loaded.content == saved.content
    assert loaded.metadata == {}
    assert loaded.tags == ["leo", "sqlite"]
    assert loaded.source == "architecture.md"
    assert loaded.version == 1
    assert [r.id for r in await restarted.list()] == [saved.id]

    results = await MemorySearch(restarted).search("sqlite")
    assert results[0].id == saved.id
    assert results[0].score > 0


# ------------------------------------------------------------------ filtering
async def test_category_kind_tag_and_metadata_filters(env):
    store, search, _ = env
    all_categories = list(MemoryCategory)
    for index, category in enumerate(all_categories):
        await store.add(
            _record(
                category=category,
                kind=MemoryKind.FACT if index % 2 == 0 else MemoryKind.ASSUMPTION,
                content=f"unique content for {category.value} category",
                tags=["group", category.value],
                metadata={"project": "leo" if index == 0 else "other", "rank": index},
            )
        )

    assert len(all_categories) == 7
    assert [r.category for r in await store.list(category=MemoryCategory.PERSONAL)] == [
        MemoryCategory.PERSONAL
    ]
    assert len(await store.list(category="project")) == 1
    assumptions = await store.list(kind=MemoryKind.ASSUMPTION)
    assert len(assumptions) == 3
    assert all(r.kind is MemoryKind.ASSUMPTION for r in assumptions)
    assert len(await store.list(kind="temporary")) == 0
    # Tags use any-match by default; match_all_tags opts into all-match.
    assert len(await store.list(tags=["semantic"])) == 1
    assert len(await store.list(tags=["group", "semantic"])) == 7
    assert len(await store.list(tags=["group", "semantic"], match_all_tags=True)) == 1
    assert len(await store.list(tags=["semantic", "project"])) == 2

    results = await search.search(
        "category", metadata={"project": "leo"}, category=MemoryCategory.WORKING
    )
    assert len(results) == 1
    assert results[0].metadata["project"] == "leo"
    assert results[0].score > 0


# -------------------------------------------------------------------- ranking
async def test_keyword_search_ranking(env):
    store, search, _ = env
    exact = await store.add(
        _record(content="python async sqlite cache implementation", confidence=0.6)
    )
    partial = await store.add(_record(content="python notes", confidence=0.95))
    unrelated = await store.add(_record(content="tomato soup recipe"))

    results = await search.search("python async sqlite cache")
    ids = [r.id for r in results]
    assert ids[0] == exact.id
    assert partial.id in ids
    assert unrelated.id not in ids
    assert all(0 < r.score <= 1 for r in results)

    assert search.last_stats["candidates_scanned"] == 3
    assert search.last_stats["returned"] == len(results)
    assert search.last_stats["latency_ms"] >= 0


async def test_custom_embedding_callable_is_pluggable(env):
    store, _, _ = env
    python = await store.add(_record(content="python embeddings"))
    other = await store.add(_record(content="woodworking bench"))

    calls: list[str] = []

    def embedding(text: str) -> list[float]:
        calls.append(text)
        return [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]

    results = await MemorySearch(store, embedding_fn=embedding).search("python")
    assert calls, "custom embedding callable was never used"
    assert results[0].id == python.id
    assert other.id not in [r.id for r in results]


# ---------------------------------------------------------------------- links
async def test_relationship_aware_retrieval(env):
    store, search, _ = env
    origin = await store.add(_record(content="deployment runbook"))
    related = await store.add(
        _record(content="postgres backup settings", tags=["ops"], confidence=0.1)
    )
    unrelated = await store.add(_record(content="banana bread recipe", confidence=0.1))

    link = await store.link(origin.id, related.id, "references", weight=1.0)
    assert link["relation"] == "references"
    assert await store.get_links(origin.id) == [link]

    results = await search.search("deployment runbook", links_from=origin.id)
    scores = {r.id: r.score for r in results}
    assert related.id in scores
    assert scores[related.id] > 0
    if unrelated.id in scores:
        assert scores[related.id] > scores[unrelated.id]

    assert await store.unlink(origin.id, related.id, "references") == 1
    assert await store.get_links(origin.id) == []


# ------------------------------------------------------- versions, audit, delete
async def test_versioning_events_and_user_controlled_deletion(env):
    store, _, bus = env
    record = await store.add(_record(content="original", source="user"))
    await store.link(record.id, (await store.add(_record(content="neighbour"))).id)

    updated = await store.update(
        record.id, content="revised", source="verified-report", reason="correction"
    )
    assert updated.version == 2
    assert updated.content == "revised"
    assert updated.updated_at != updated.created_at or updated.version > 1

    versions = await store.versions(record.id)
    assert len(versions) == 1
    assert versions[0]["version"] == 2
    assert versions[0]["content"] == "revised"
    assert versions[0]["metadata"] == {}

    events = await bus.history("memory.changed")
    actions = [event.payload["action"] for event in events]
    assert "created" in actions
    assert "updated" in actions
    update_event = next(e for e in events if e.payload["action"] == "updated")
    assert update_event.payload["id"] == record.id
    assert update_event.payload["changes"]["content"] == "revised"
    assert update_event.payload["reason"] == "correction"

    assert await store.delete(record.id) is True
    assert await store.get(record.id) is None
    assert await store.get_links(record.id) == []
    assert await store.versions(record.id) == []
    assert await store.delete(record.id) is False
    assert any(e.payload["action"] == "deleted" for e in await bus.history("memory.changed"))


# -------------------------------------------------------- dedup / consolidation
async def test_deduplication_and_consolidation_preserve_provenance(env):
    store, search, bus = env
    first = await store.add(
        _record(
            content="LEO uses SQLite for durable storage.",
            source="docs/a.md",
            tags=["storage"],
        )
    )
    second = await store.add(
        _record(
            content="LEO uses SQLite for durable storage!",
            source="docs/b.md",
            tags=["sqlite"],
        )
    )
    third = await store.add(
        _record(
            content="LEO  uses SQLite for durable storage",
            source="docs/c.md",
        )
    )
    other_kind = await store.add(
        _record(
            content="LEO uses SQLite for durable storage.",
            kind=MemoryKind.ASSUMPTION,
            source="docs/unverified.md",
        )
    )
    await store.update(second.id, tags=["sqlite", "storage"], reason="tag cleanup")
    external = await store.add(_record(content="external reference"))
    await store.link(second.id, external.id, "related")
    await store.link(external.id, third.id, "supports", weight=0.5)

    consolidator = Consolidator(store, threshold=0.9)
    duplicates = await consolidator.find_duplicates(category=MemoryCategory.SEMANTIC)
    assert len(duplicates) == 1
    assert {r.id for r in duplicates[0]} == {first.id, second.id, third.id}

    result = await consolidator.consolidate(category=MemoryCategory.SEMANTIC)
    assert len(result) == 1
    assert result.merged == 1
    assert result.removed == 2
    survivor = result.records[0]
    originals = {first.id, second.id, third.id}
    assert survivor.id in originals
    assert set(result.removed_ids) == originals - {survivor.id}

    persisted = await store.get(survivor.id)
    assert persisted is not None
    assert persisted.version >= 2
    assert persisted.source in {"docs/a.md", "docs/b.md", "docs/c.md"}
    assert persisted.metadata["merged_count"] == 2
    assert set(persisted.metadata["sources"]) == {
        "docs/a.md",
        "docs/b.md",
        "docs/c.md",
    }
    assert set(persisted.metadata["merged_from"]) == originals - {survivor.id}
    assert set(persisted.tags) == {"storage", "sqlite"}
    assert set(persisted.supersedes) == originals - {survivor.id}

    remaining = await store.list(category=MemoryCategory.SEMANTIC)
    assert survivor.id in [r.id for r in remaining]
    assert not set(result.removed_ids) & {r.id for r in remaining}
    assert other_kind.id in [r.id for r in remaining]

    links = await store.get_links(survivor.id)
    assert any(l["target"] == external.id for l in links)
    assert any(l["source"] == external.id and l["relation"] == "supports" for l in links)
    for removed_id in result.removed_ids:
        assert not await store.get_links(removed_id)
    assert len(await store.versions(survivor.id)) >= 2
    assert any(
        e.payload["action"] == "consolidated" for e in await bus.history("memory.changed")
    )
    assert survivor.id in [
        r.id for r in await search.search("SQLite", category=MemoryCategory.SEMANTIC)
    ]


# ----------------------------------------------------------- retention / delete
async def test_retention_expires_records_and_purge_removes_them(env):
    store, search, _ = env
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    expired = await store.add(_record(content="old temporary note", expires_at=past))
    live = await store.add(_record(content="fresh temporary note", expires_at=future))

    assert [r.id for r in await store.list()] == [live.id]
    assert len(await store.list(include_expired=True)) == 2
    assert [r.id for r in await search.search("temporary note")] == [live.id]
    assert [r.id for r in await search.search("temporary", include_expired=True)] == [
        expired.id,
        live.id,
    ]

    assert await store.purge_expired() == 1
    assert await store.get(expired.id) is None
    assert await store.purge_expired() == 0
    assert await store.get(live.id) is not None


# ------------------------------------------------------------- kind integrity
async def test_no_silent_fact_promotion(env):
    store, _, bus = env
    uncertain = await store.add(
        _record(
            content="the build might be broken",
            kind=MemoryKind.UNVERIFIED,
            confidence=0.3,
        )
    )
    with pytest.raises(ValueError, match="explicit"):
        await store.update(uncertain.id, kind=MemoryKind.FACT)
    with pytest.raises(ValueError, match="confidence must be a number"):
        await store.update(uncertain.id, kind=MemoryKind.FACT, confidence="high")
    assert (await store.get(uncertain.id)).kind is MemoryKind.UNVERIFIED

    promoted = await store.update(
        uncertain.id,
        kind=MemoryKind.FACT,
        confidence=0.85,
        reason="verified with CI logs",
        actor="test",
    )
    assert promoted.kind is MemoryKind.FACT
    assert promoted.confidence == 0.85
    assert promoted.version == 2
    promotion_events = [
        e for e in await bus.history("memory.changed") if e.payload.get("promotion")
    ]
    assert len(promotion_events) == 1
    assert promotion_events[0].payload["reason"] == "verified with CI logs"

    assumption = await store.add(
        _record(content="assumed preference", kind=MemoryKind.ASSUMPTION)
    )
    still_assumption = await store.update(assumption.id, confidence=0.99)
    assert still_assumption.kind is MemoryKind.ASSUMPTION
    with pytest.raises(ValueError, match="metadata"):
        await store.update(assumption.id, metadata={"kind": "fact"})

    temporary = await store.add(
        _record(content="temporary cache note", kind=MemoryKind.TEMPORARY)
    )
    changed = await store.update(temporary.id, kind=MemoryKind.UNVERIFIED)
    assert changed.kind is MemoryKind.UNVERIFIED


# ------------------------------------------------------------------ evaluation
async def test_evaluation_precision_and_stats_are_measured(env):
    store, search, _ = env
    good_a = await store.add(_record(content="kubernetes rollback procedure"))
    good_b = await store.add(_record(content="kubernetes rollback verification"))
    noise = await store.add(_record(content="sourdough starter"))

    evaluation = MemoryEvaluation(search)
    metrics = await evaluation.evaluate("kubernetes rollback", [good_a.id, good_b.id, noise.id], k=2)
    assert len(metrics.retrieved_ids) == 2
    assert set(metrics.retrieved_ids) == {good_a.id, good_b.id}
    assert metrics.hits == 2
    assert metrics.precision_at_k == 1.0
    assert metrics.precision == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert 0 < metrics.recall_at_k <= 1
    assert metrics.latency_ms >= 0
    assert metrics.stats["returned"] == 2
    assert metrics.stats["candidates_scanned"] == 3
    assert metrics.mean_score > 0
    assert metrics.missing_ids == [noise.id]

    report = await evaluation.evaluate_many(
        [
            {"query": "kubernetes rollback", "relevant_ids": [good_a.id], "k": 1},
            {"query": "sourdough starter", "relevant_ids": [noise.id], "k": 1},
        ]
    )
    assert report.total_queries == 2
    assert report.total_retrieved == 2
    assert report.total_hits == 2
    assert report.mean_precision_at_k == 1.0
    assert report.elapsed_ms >= 0
    assert MemoryEvaluation.precision_at_k([good_a.id], [good_b.id], 1) == 0.0


# ----------------------------------------------------------- confidence bounds
async def test_confidence_bounds_are_enforced_on_add_and_update(env):
    store, search, _ = env
    high = await store.add(_record(content="high confidence", confidence=5.0))
    low = await store.add(_record(content="low confidence", confidence=-3.0))
    assert (await store.get(high.id)).confidence == 1.0
    assert (await store.get(low.id)).confidence == 0.0

    nan = await store.add(_record(content="nan confidence", confidence=float("nan")))
    assert (await store.get(nan.id)).confidence == 0.0

    updated = await store.update(high.id, confidence=9.0)
    assert updated.confidence == 1.0
    with pytest.raises(ValueError, match="confidence must be a number"):
        await store.update(high.id, confidence="high")

    # min_confidence filtering sees the bounded value, never the raw 5.0.
    result = await search.search("confidence", min_confidence=1.0)
    assert [r.id for r in result] == [high.id]


# --------------------------------------------------------------------- expiry
async def test_list_keeps_live_records_with_unusual_expiry_formats(env):
    store, search, _ = env
    from leo.memory.store import is_expired

    now = datetime.now(timezone.utc)
    future = now + timedelta(hours=1)
    # Same future instant, expressed with a large negative UTC offset. The ISO
    # string sorts BEFORE `now`, which used to make the SQL pre-filter drop a
    # record that is actually still live.
    future_str = future.astimezone(timezone(timedelta(hours=-14))).isoformat()
    assert future_str < now.isoformat()
    live = await store.add(_record(content="live future expiry", expires_at=future_str))
    assert not is_expired(await store.get(live.id))
    assert live.id in [r.id for r in await store.list()]
    assert live.id in [r.id for r in await search.search("live future expiry")]

    # An empty expiry means "never expires", not "expired".
    empty = await store.add(_record(content="empty expiry marker", expires_at=""))
    assert empty.id in [r.id for r in await store.list()]
    assert empty.id in [r.id for r in await search.search("empty expiry marker")]


# ------------------------------------------------------------ supersede links
async def test_supersedes_survives_update_as_string_or_list(env):
    store, _, _ = env
    old = await store.add(_record(content="old fact"))
    survivor = await store.add(_record(content="new fact"))

    updated = await store.update(survivor.id, supersedes=old.id)
    assert updated.supersedes == old.id
    assert (await store.get(survivor.id)).supersedes == old.id

    older = await store.add(_record(content="older fact"))
    updated = await store.update(survivor.id, supersedes=[old.id, older.id])
    assert set(updated.supersedes) == {old.id, older.id}
    assert updated.version == 3
    assert [v["version"] for v in await store.versions(survivor.id)] == [2, 3]


# --------------------------------------------------------- unicode / injection
async def test_unicode_queries_tokenize_and_do_not_return_everything(env):
    store, search, _ = env
    from leo.memory.search import tokenize

    assert tokenize("café") == ["café"]
    assert tokenize("résumé") == ["résumé"]
    assert tokenize("数据库连接池") == ["数据库连接池"]

    cjk = await store.add(_record(content="数据库连接池配置说明"))
    unrelated = await store.add(_record(content="completely unrelated english note"))
    accented = await store.add(_record(content="café review notes"))

    results = await search.search("数据库连接池")
    ids = [r.id for r in results]
    assert cjk.id in ids
    assert unrelated.id not in ids

    assert accented.id in [r.id for r in await search.search("café")]

    # Empty/whitespace queries remain a deterministic "browse" mode.
    empty = await search.search("   ")
    assert len(empty) == 3


async def test_query_strings_are_never_executed_as_sql(env):
    store, search, _ = env
    secret = await store.add(_record(content="top secret payload"))
    for payload in ("' OR '1'='1", "'; DROP TABLE memory_items; --", '" OR 1=1 --'):
        results = await search.search(payload)
        # The payload is treated as literal query text, never interpolated into
        # SQL, so no error is raised and the exact string is reported in stats.
        assert isinstance(results, list)
        assert search.last_stats["query"] == payload
    # The table is intact and an ordinary query still works.
    assert (await store.get(secret.id)) is not None
    assert len(await store.list()) == 1
    assert [r.id for r in await search.search("top secret payload")] == [secret.id]


# ------------------------------------------------------ validation & ordering
async def test_invalid_category_and_kind_are_rejected(env):
    store, _, _ = env
    good = await store.add(_record(content="valid record"))
    with pytest.raises(ValueError):
        await store.update(good.id, category="not-a-category")
    with pytest.raises(ValueError):
        await store.update(good.id, kind="not-a-kind")
    assert (await store.get(good.id)).category is MemoryCategory.SEMANTIC
    assert (await store.get(good.id)).kind is MemoryKind.FACT
    with pytest.raises(ValueError):
        await store.add(_record(content="bad category", category="not-a-category"))


async def test_result_ordering_is_deterministic_and_scores_bounded(env):
    store, search, _ = env
    for index in range(5):
        await store.add(_record(content=f"alpha beta gamma {index}", confidence=0.5))

    first = await search.search("alpha beta gamma", limit=3)
    second = await search.search("alpha beta gamma", limit=3)
    assert len(first) == 3
    assert [r.id for r in first] == [r.id for r in second]
    assert all(0 <= r.score <= 1 for r in first)
    assert all(first[i].score >= first[i + 1].score for i in range(len(first) - 1))
