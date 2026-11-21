"""Measured retrieval evaluation. No metric is estimated or invented.

The caller supplies the labelled relevant ids; precision/recall/MRR are
computed solely from the ids actually returned by :class:`MemorySearch`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .search import MemorySearch


class LabelledQuery(BaseModel):
    """One evaluation case: a query and the caller's known relevant ids."""

    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("labels", "expected", "relevant", "relevant_memories"):
            if alias in values and "relevant_ids" not in values:
                values["relevant_ids"] = values.pop(alias)
        if "kwargs" in values and "filters" not in values:
            values["filters"] = values.pop("kwargs")
        return values


@dataclass
class RetrievalMetrics:
    """Per-query measurements for a single labelled retrieval."""

    query: str
    k: int
    relevant_ids: list[str]
    retrieved_ids: list[str]
    hits: int
    precision_at_k: float
    precision: float
    recall_at_k: float
    reciprocal_rank: float
    mean_score: float
    latency_ms: float
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def missing_ids(self) -> list[str]:
        retrieved = set(self.retrieved_ids)
        return [memory_id for memory_id in self.relevant_ids if memory_id not in retrieved]

    def to_dict(self) -> dict[str, Any]:
        result = {
            "query": self.query,
            "k": self.k,
            "relevant_ids": list(self.relevant_ids),
            "retrieved_ids": list(self.retrieved_ids),
            "hits": self.hits,
            "precision_at_k": self.precision_at_k,
            "precision": self.precision,
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "mean_score": self.mean_score,
            "latency_ms": self.latency_ms,
            "stats": dict(self.stats),
        }
        result["missing_ids"] = self.missing_ids
        return result


@dataclass
class EvaluationReport:
    """Aggregate over measured cases (all values derived from those cases)."""

    cases: list[RetrievalMetrics]
    total_queries: int
    total_relevant: int
    total_retrieved: int
    total_hits: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "total_queries": self.total_queries,
            "total_relevant": self.total_relevant,
            "total_retrieved": self.total_retrieved,
            "total_hits": self.total_hits,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "elapsed_ms": self.elapsed_ms,
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class MemoryEvaluation:
    """Evaluate a memory retriever against labels provided by the caller."""

    def __init__(self, searcher: MemorySearch) -> None:
        self.searcher = searcher

    @staticmethod
    def precision_at_k(
        retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
    ) -> float:
        """Fraction of the top k slots filled by labelled relevant results.

        Uses k as the denominator (standard precision@k); passing fewer than k
        results therefore lowers the score rather than inflating it.
        """
        if k <= 0:
            return 0.0
        relevant = set(relevant_ids)
        hits = sum(1 for memory_id in list(retrieved_ids)[:k] if memory_id in relevant)
        return hits / k

    async def evaluate(
        self,
        query: str,
        relevant_ids: Iterable[str],
        k: int = 10,
        **search_kwargs: Any,
    ) -> RetrievalMetrics:
        labelled = list(dict.fromkeys(str(memory_id) for memory_id in relevant_ids))
        relevant = set(labelled)
        started = time.perf_counter()
        results = await self.searcher.search(query, limit=k, **search_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        retrieved = [record.id for record in results]
        hits = sum(1 for memory_id in retrieved if memory_id in relevant)
        rank = next(
            (index for index, memory_id in enumerate(retrieved, start=1) if memory_id in relevant),
            None,
        )
        return RetrievalMetrics(
            query=query,
            k=k,
            relevant_ids=labelled,
            retrieved_ids=retrieved,
            hits=hits,
            precision_at_k=self.precision_at_k(retrieved, labelled, k),
            precision=(hits / len(retrieved)) if retrieved else 0.0,
            recall_at_k=(hits / len(relevant)) if relevant else 0.0,
            reciprocal_rank=(1.0 / rank) if rank else 0.0,
            mean_score=_mean([float(record.score) for record in results]),
            latency_ms=round(latency_ms, 3),
            stats=dict(self.searcher.last_stats),
        )

    async def evaluate_many(
        self,
        cases: Iterable[LabelledQuery | Mapping[str, Any] | tuple[str, Iterable[str]]],
        *,
        k: int = 10,
    ) -> EvaluationReport:
        prepared: list[LabelledQuery] = []
        for case in cases:
            if isinstance(case, LabelledQuery):
                prepared.append(case)
            elif isinstance(case, tuple) and len(case) == 2:
                prepared.append(LabelledQuery(query=case[0], relevant_ids=list(case[1]), k=k))
            else:
                item = LabelledQuery.model_validate(case)
                if item.k == 10 and "k" not in case:
                    item.k = k
                prepared.append(item)

        started = time.perf_counter()
        metrics: list[RetrievalMetrics] = []
        for case in prepared:
            metrics.append(
                await self.evaluate(
                    case.query,
                    case.relevant_ids,
                    k=case.k,
                    **case.filters,
                )
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return EvaluationReport(
            cases=metrics,
            total_queries=len(metrics),
            total_relevant=sum(len(case.relevant_ids) for case in metrics),
            total_retrieved=sum(len(case.retrieved_ids) for case in metrics),
            total_hits=sum(case.hits for case in metrics),
            mean_precision_at_k=_mean([case.precision_at_k for case in metrics]),
            mean_recall_at_k=_mean([case.recall_at_k for case in metrics]),
            mean_reciprocal_rank=_mean([case.reciprocal_rank for case in metrics]),
            elapsed_ms=round(elapsed_ms, 3),
        )


__all__ = [
    "EvaluationReport",
    "LabelledQuery",
    "MemoryEvaluation",
    "RetrievalMetrics",
]
