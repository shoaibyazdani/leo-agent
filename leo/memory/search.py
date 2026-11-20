"""Retrieval for LEO memory: keyword, semantic, filtered and link-aware.

The keyword scorer is a small BM25-style token scorer implemented in Python;
the default semantic scorer is a deterministic hashing embedding compared with
cosine similarity. Any callable ``text -> vector`` can be injected instead.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ..contracts import MemoryRecord
from .store import enum_value, is_expired

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import MemoryStore

EmbeddingFn = Callable[[str], Sequence[float]] | Callable[[str], Any]

# ``[^\W_]`` matches any Unicode letter/digit but not underscores. The old
# ``[a-z0-9]+`` pattern silently discarded non-ASCII text (accented words were
# split mid-token, CJK text produced no tokens at all), which made a unicode
# query indistinguishable from an empty one and returned every candidate.
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
DEFAULT_DIMENSION = 128


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def stem(token: str) -> str:
    """Conservative suffix stripping so 'running' matches 'run'."""
    if len(token) <= 4:
        return token
    for suffix, minimum in (("ingly", 7), ("edly", 6), ("ing", 6), ("ies", 5), ("ed", 5), ("es", 5)):
        if token.endswith(suffix) and len(token) > minimum:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def hash_embedding(text: str, dimension: int = DEFAULT_DIMENSION) -> list[float]:
    """Deterministic, dependency-free embedding (signed feature hashing).

    Word unigrams/stems and character trigrams are hashed with blake2b so the
    vector is stable across processes (unlike Python's salted ``hash``).
    """
    vector = [0.0] * max(1, int(dimension))
    tokens = tokenize(text)
    features: set[str] = set()
    for token in tokens:
        features.add(token)
        if len(token) > 3:
            features.add("~" + stem(token))
    compact = "".join(tokens)
    features.update(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % len(vector)
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity in [0, 1]; mismatched dimensions use the overlap."""
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[i]) * float(right[i]) for i in range(size))
    left_norm = math.sqrt(sum(float(v) * float(v) for v in left))
    right_norm = math.sqrt(sum(float(v) * float(v) for v in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _document_text(record: MemoryRecord) -> str:
    return " ".join([record.content, *[str(tag) for tag in record.tags]])


def _timestamp_score(value: str | None, half_life_days: float = 30.0) -> float:
    from datetime import datetime

    from .store import parse_timestamp

    parsed = parse_timestamp(value)
    if parsed is None:
        return 0.5
    age_days = max(0.0, (datetime.now(parsed.tzinfo) - parsed).total_seconds() / 86400.0)
    if half_life_days <= 0:
        return 1.0 if age_days == 0 else 0.0
    return math.exp(-math.log(2) * age_days / half_life_days)


class MemorySearch:
    """Rank stored memories for a query.

    Final score = keyword + semantic + confidence + recency (+ link boost when
    ``links_from`` is used), with weights set in the constructor. ``.score`` is
    set on every returned record; ``last_stats`` holds measured retrieval
    statistics for the most recent call (never estimates).
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_fn: EmbeddingFn | None = None,
        *,
        dimension: int = DEFAULT_DIMENSION,
        keyword_weight: float = 0.45,
        semantic_weight: float = 0.25,
        confidence_weight: float = 0.15,
        recency_weight: float = 0.15,
        link_weight: float = 0.6,
        half_life_days: float = 30.0,
    ) -> None:
        self.store = store
        self.embedding_fn = embedding_fn
        self.dimension = dimension
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.confidence_weight = confidence_weight
        self.recency_weight = recency_weight
        self.link_weight = link_weight
        self.half_life_days = half_life_days
        self.last_stats: dict[str, Any] = {}

    async def _embed(self, text: str) -> Sequence[float]:
        if self.embedding_fn is None:
            return hash_embedding(text, self.dimension)
        value = self.embedding_fn(text)
        if inspect.isawaitable(value):
            value = await value
        return [float(component) for component in value]

    async def search(
        self,
        query: str,
        category: Any = None,
        kind: Any = None,
        tags: Sequence[str] | None = None,
        links_from: str | None = None,
        limit: int = 10,
        *,
        metadata: Mapping[str, Any] | None = None,
        filters: Mapping[str, Any] | None = None,
        include_expired: bool = False,
        min_confidence: float | None = None,
        match_all_tags: bool = False,
    ) -> list[MemoryRecord]:
        started = time.perf_counter()
        metadata_filters: dict[str, Any] = {}
        if metadata:
            metadata_filters.update(metadata)
        if filters:
            metadata_filters.update(filters)

        candidates = await self.store.list(
            category=category,
            kind=kind,
            tags=tags,
            limit=10_000,
            include_expired=include_expired,
            metadata=metadata_filters or None,
            match_all_tags=match_all_tags,
        )
        if min_confidence is not None:
            candidates = [r for r in candidates if r.confidence >= min_confidence]
        if not include_expired:
            candidates = [r for r in candidates if not is_expired(r)]

        link_boost: dict[str, float] = {}
        if links_from:
            links = await self._links_for(links_from)
            for link in links:
                weight = float(link.get("weight", 1.0))
                if link.get("source") == links_from:
                    target = str(link.get("target"))
                    link_boost[target] = max(link_boost.get(target, 0.0), self.link_weight * weight)
                elif link.get("target") == links_from:
                    source = str(link.get("source"))
                    link_boost[source] = max(
                        link_boost.get(source, 0.0), self.link_weight * 0.7 * weight
                    )

        keyword_scores = self._keyword_scores(query, candidates)
        query_vector = await self._embed(query)
        vector_cache: dict[str, Sequence[float]] = {}
        semantic_scores = [
            cosine_similarity(query_vector, await self._record_vector(record, vector_cache))
            for record in candidates
        ]

        empty_query = not tokenize(query)
        scored: list[tuple[float, MemoryRecord]] = []
        for record, keyword, semantic in zip(candidates, keyword_scores, semantic_scores):
            related = link_boost.get(record.id, 0.0)
            if not empty_query and keyword <= 0 and semantic <= 0 and related <= 0:
                continue
            base = (
                self.keyword_weight * keyword
                + self.semantic_weight * semantic
                + self.confidence_weight * max(0.0, min(1.0, record.confidence))
                + self.recency_weight * _timestamp_score(record.updated_at, self.half_life_days)
            )
            raw = base + related
            record.score = round(max(0.0, min(1.0, raw)), 6)
            scored.append((raw, record))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        results = [record for _, record in scored[: max(0, int(limit))]]

        self.last_stats = {
            "query": query,
            "candidates_scanned": len(candidates),
            "candidates_scored": len(scored),
            "returned": len(results),
            "links_from": links_from,
            "linked_candidates": len(link_boost),
            "filters": {
                "category": enum_value(category) if category is not None else None,
                "kind": enum_value(kind) if kind is not None else None,
                "tags": list(tags) if tags else None,
                "metadata": dict(metadata_filters),
                "min_confidence": min_confidence,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return results

    # Keep an alias for callers that think of the operation as a query.
    query = search

    async def _links_for(self, memory_id: str) -> list[dict[str, Any]]:
        getter = getattr(self.store, "get_links", None)
        if callable(getter):
            return list(await getter(memory_id))
        from .. import db

        rows = await db.fetchall(
            "SELECT source,target,relation,weight FROM memory_links "
            "WHERE source=? OR target=?",
            (memory_id, memory_id),
        )
        return [dict(row) for row in rows]

    async def _record_vector(
        self, record: MemoryRecord, cache: dict[str, Sequence[float]]
    ) -> Sequence[float]:
        if record.id in cache:
            return cache[record.id]
        stored = getattr(record, "embedding", None)
        if self.embedding_fn is None and stored:
            # Persisted vectors are authoritative for the default scorer.
            vector: Sequence[float] = [float(v) for v in stored]
        else:
            vector = await self._embed(_document_text(record))
        cache[record.id] = vector
        return vector

    def _keyword_scores(self, query: str, records: Sequence[MemoryRecord]) -> list[float]:
        if not records:
            return []
        documents = [Counter(stem(t) for t in tokenize(record.content)) for record in records]
        tag_documents = [
            Counter(stem(t) for t in tokenize(" ".join(str(tag) for tag in record.tags)))
            for record in records
        ]
        document_frequency: Counter[str] = Counter()
        for content_terms, tag_terms in zip(documents, tag_documents):
            for term in set(content_terms) | set(tag_terms):
                document_frequency[term] += 1
        total = len(records)
        idf = {
            term: math.log(1.0 + total / (1.0 + count))
            for term, count in document_frequency.items()
        }
        query_terms = list(dict.fromkeys(stem(t) for t in tokenize(query)))
        denominator = sum(idf.get(term, 1.0) for term in query_terms) or 1.0
        query_phrase = " ".join(tokenize(query))

        scores: list[float] = []
        for content_terms, tag_terms, record in zip(documents, tag_documents, records):
            if not query_terms:
                scores.append(0.0)
                continue
            raw = 0.0
            for term in query_terms:
                tf = content_terms.get(term, 0) + tag_terms.get(term, 0)
                if tf:
                    raw += idf.get(term, 1.0) * (1.0 + math.log(tf))
            score = min(1.0, raw / denominator)
            if query_phrase and query_phrase in " ".join(tokenize(record.content)):
                score = min(1.0, score + 0.15)
            scores.append(score)
        return scores


__all__ = [
    "EmbeddingFn",
    "MemorySearch",
    "cosine_similarity",
    "hash_embedding",
    "stem",
    "tokenize",
]
