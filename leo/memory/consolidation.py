"""Deduplication, retention-aware consolidation and source-preserving merges."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .. import db
from ..contracts import MemoryCategory, MemoryRecord, utcnow
from .search import cosine_similarity
from .store import enum_value, is_expired

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .search import EmbeddingFn
    from .store import MemoryStore

DEFAULT_THRESHOLD = 0.9
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_content(content: str) -> str:
    """Canonical form used for exact and near-duplicate comparison."""
    text = str(content or "").lower()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def content_similarity(left: str | MemoryRecord, right: str | MemoryRecord) -> float:
    """Similarity in [0, 1] from normalized text (sequence + token overlap)."""
    a = normalize_content(left.content if isinstance(left, MemoryRecord) else left)
    b = normalize_content(right.content if isinstance(right, MemoryRecord) else right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sequence = SequenceMatcher(None, a, b).ratio()
    a_tokens, b_tokens = set(a.split()), set(b.split())
    union = a_tokens | b_tokens
    jaccard = len(a_tokens & b_tokens) / len(union) if union else 0.0
    return max(sequence, jaccard)


class ConsolidationResult(list[MemoryRecord]):
    """List of consolidated records, plus counts and provenance details."""

    def __init__(
        self,
        records: Iterable[MemoryRecord] = (),
        groups: Sequence[Sequence[MemoryRecord]] = (),
        removed_ids: Iterable[str] = (),
    ) -> None:
        super().__init__(records)
        self.groups: list[list[MemoryRecord]] = [list(group) for group in groups]
        self.removed_ids: list[str] = list(removed_ids)

    @property
    def merged(self) -> int:
        """Number of duplicate groups that were merged."""
        return len(self)

    @property
    def records(self) -> list[MemoryRecord]:
        return list(self)

    @property
    def removed(self) -> int:
        return len(self.removed_ids)


class Consolidator:
    """Find near-duplicates in the same category/kind and merge them.

    The newest, most confident member of each cluster becomes the surviving
    version. Its version is bumped through ``MemoryStore.update`` (which writes
    ``memory_versions`` and emits ``memory.changed``); source attribution, tags
    and links from all duplicates are carried over before the duplicates are
    removed.
    """

    def __init__(
        self,
        store: MemoryStore,
        threshold: float = DEFAULT_THRESHOLD,
        *,
        embedding_fn: EmbeddingFn | None = None,
        similarity_threshold: float | None = None,
        dedupe_threshold: float | None = None,
    ) -> None:
        if similarity_threshold is not None:
            threshold = similarity_threshold
        if dedupe_threshold is not None:
            threshold = dedupe_threshold
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.store = store
        self.threshold = float(threshold)
        self.embedding_fn = embedding_fn

    async def similarity(self, left: str | MemoryRecord, right: str | MemoryRecord) -> float:
        score = content_similarity(left, right)
        if self.embedding_fn is not None and score < self.threshold:
            from inspect import isawaitable

            def text(value: str | MemoryRecord) -> str:
                if isinstance(value, MemoryRecord):
                    return " ".join([value.content, *[str(t) for t in value.tags]])
                return str(value)

            a = self.embedding_fn(text(left))
            b = self.embedding_fn(text(right))
            if isawaitable(a):
                a = await a
            if isawaitable(b):
                b = await b
            score = max(score, cosine_similarity(a, b))
        return score

    async def find_duplicates(
        self, category: MemoryCategory | str | None = None
    ) -> list[list[MemoryRecord]]:
        """Cluster near-duplicate records; only clusters with 2+ members."""
        records = await self.store.list(
            category=category, limit=100_000, include_expired=True
        )
        live = [record for record in records if not is_expired(record)]
        if live:
            records = live
        buckets: dict[tuple[str, str], list[MemoryRecord]] = {}
        for record in records:
            key = (enum_value(record.category), enum_value(record.kind))
            buckets.setdefault(key, []).append(record)

        clusters: list[list[MemoryRecord]] = []
        for members in buckets.values():
            parents = list(range(len(members)))

            def find(index: int) -> int:
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if await self.similarity(members[i], members[j]) >= self.threshold:
                        parents[find(j)] = find(i)
            grouped: dict[int, list[MemoryRecord]] = {}
            for index, member in enumerate(members):
                grouped.setdefault(find(index), []).append(member)
            clusters.extend(group for group in grouped.values() if len(group) > 1)
        return clusters

    # A short alias used by callers that just want the duplicate count/groups.
    duplicates = find_duplicates

    async def consolidate(
        self, category: MemoryCategory | str | None = None
    ) -> ConsolidationResult:
        clusters = await self.find_duplicates(category)
        survivors: list[MemoryRecord] = []
        removed_ids: list[str] = []

        for cluster in clusters:
            canonical = max(
                cluster,
                key=lambda r: (r.updated_at, r.confidence, r.id),
            )
            duplicates = [r for r in cluster if r.id != canonical.id]
            if not duplicates:
                continue

            metadata = dict(canonical.metadata)
            sources: list[str] = []
            if canonical.source:
                sources.append(canonical.source)
            attribution: list[dict[str, Any]] = [
                {
                    "id": canonical.id,
                    "source": canonical.source,
                    "category": enum_value(canonical.category),
                    "version": canonical.version,
                }
            ]
            tags = list(canonical.tags)
            for duplicate in duplicates:
                if duplicate.source and duplicate.source not in sources:
                    sources.append(duplicate.source)
                attribution.append(
                    {
                        "id": duplicate.id,
                        "source": duplicate.source,
                        "category": enum_value(duplicate.category),
                        "version": duplicate.version,
                    }
                )
                for tag in duplicate.tags:
                    if tag not in tags:
                        tags.append(tag)

            metadata["sources"] = sources
            metadata["source_attribution"] = attribution
            metadata["merged_from"] = [r.id for r in duplicates]
            metadata["merged_count"] = len(duplicates)

            for duplicate in duplicates:
                await self._transfer_links(duplicate.id, canonical.id)
                async with db.connection() as conn:
                    await conn.execute(
                        """UPDATE memory_versions
                           SET memory_id=?, reason=COALESCE(reason,'') || ?
                           WHERE memory_id=?""",
                        (canonical.id, f"; merged-from:{duplicate.id}", duplicate.id),
                    )
                    await conn.execute(
                        "DELETE FROM memory_links WHERE source=? OR target=?",
                        (duplicate.id, duplicate.id),
                    )
                    await conn.execute("DELETE FROM memory_items WHERE id=?", (duplicate.id,))
                await self.store.emit_change(
                    "consolidated-duplicate",
                    duplicate,
                    reason="merged into canonical memory",
                    canonical_id=canonical.id,
                    merged_at=utcnow(),
                )
                removed_ids.append(duplicate.id)

            survivor = await self.store.update(
                canonical.id,
                metadata=metadata,
                tags=tags,
                supersedes=[r.id for r in duplicates],
                action="consolidated",
                reason=f"deduplication merge (threshold={self.threshold})",
            )
            survivors.append(survivor)

        return ConsolidationResult(survivors, clusters, removed_ids)

    async def _transfer_links(self, old_id: str, canonical_id: str) -> None:
        links = await self.store.get_links(old_id)
        for link in links:
            source = canonical_id if link["source"] == old_id else link["source"]
            target = canonical_id if link["target"] == old_id else link["target"]
            if source == target:
                continue
            await self.store.link(source, target, link["relation"], link["weight"])


async def purge_expired(store: MemoryStore) -> int:
    """Convenience wrapper around :meth:`MemoryStore.purge_expired`."""
    return await store.purge_expired()


__all__ = [
    "DEFAULT_THRESHOLD",
    "ConsolidationResult",
    "Consolidator",
    "content_similarity",
    "normalize_content",
    "purge_expired",
]
