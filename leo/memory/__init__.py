"""LEO memory engine: durable storage, retrieval, consolidation, evaluation.

All components communicate through :mod:`leo.contracts` types and the shared
``leo.db`` persistence layer; no module outside this package is modified.
"""

from __future__ import annotations

from .consolidation import (
    ConsolidationResult,
    Consolidator,
    content_similarity,
    normalize_content,
    purge_expired,
)
from .evaluation import (
    EvaluationReport,
    LabelledQuery,
    MemoryEvaluation,
    RetrievalMetrics,
)
from .search import (
    MemorySearch,
    cosine_similarity,
    hash_embedding,
    tokenize,
)
from .store import (
    MEMORY_CHANGED,
    NON_FACT_KINDS,
    MemoryStore,
    is_expired,
    parse_timestamp,
)

__all__ = [
    "MEMORY_CHANGED",
    "NON_FACT_KINDS",
    "ConsolidationResult",
    "Consolidator",
    "EvaluationReport",
    "LabelledQuery",
    "MemoryEvaluation",
    "MemorySearch",
    "MemoryStore",
    "RetrievalMetrics",
    "content_similarity",
    "cosine_similarity",
    "hash_embedding",
    "is_expired",
    "normalize_content",
    "parse_timestamp",
    "purge_expired",
    "tokenize",
]
