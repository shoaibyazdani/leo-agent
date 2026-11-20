"""Self-evaluation: metrics, regression detection, gated adoption."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from leo.core import evaluation as evaluation_module
from leo.core.evaluation import Evaluation


@pytest.mark.asyncio
async def test_metrics_and_task_reliability() -> None:
    ev = Evaluation()
    for value in (10, 10, 10, 10, 4, 4, 4, 4, 4, 4):
        await ev.record("latency", value)
    report = await ev.report()
    assert report["leo.latency"]["count"] == 10
    assert report["leo.latency"]["latest"] == 4
    regressions = await ev.detect_regressions("latency", window=5, threshold=0.2)
    # Lower latency is an improvement, not a regression.
    assert regressions == []
    for value in (4, 4, 4, 4, 4, 10, 10, 10, 10, 10):
        await ev.record("errors", value)
    assert await ev.detect_regressions("errors", window=5, threshold=0.2)


@pytest.mark.asyncio
async def test_proposal_adoption_is_gated() -> None:
    ev = Evaluation()
    proposal = await ev.propose("cache tools", "measured repeated calls", {"calls": 20})
    assert proposal["status"] == "proposed"
    assert len(await ev.proposals()) == 1
    with pytest.raises(ValueError):
        await ev.adopt(proposal["id"], tests_passed=False, approver="user")
    adopted = await ev.adopt(proposal["id"], tests_passed=True, approver="user")
    assert adopted["status"] == "adopted"


@pytest.mark.asyncio
async def test_report_only_includes_own_namespace() -> None:
    """The report query must match ``namespace.*`` exactly, not a bare prefix.

    A ``LIKE 'leo%'`` pattern also matches metrics from ``leo2`` or ``leochat``
    namespaces, silently mixing other components' data into this report.
    """
    ev = Evaluation(namespace="leo")
    other = Evaluation(namespace="leo2")
    await ev.record("latency", 5)
    await other.record("latency", 99)
    await other.record("quality", 0.5)

    report = await ev.report()
    assert set(report) == {"leo.latency"}
    assert report["leo.latency"]["latest"] == 5

    # The `since` branch uses the same pattern and must be scoped too.
    scoped = await ev.report(since="1970-01-01T00:00:00+00:00")
    assert set(scoped) == {"leo.latency"}


@pytest.mark.asyncio
async def test_empty_and_zero_data_is_safe() -> None:
    """Empty tables, zero denominators and zero baselines must not raise."""
    ev = Evaluation()
    assert await ev.report() == {}
    assert await ev.tool_reliability() == {}
    assert await ev.detect_regressions("latency") == []

    metrics = await ev.task_metrics()
    assert metrics["total"] == 0
    assert metrics["success_rate"] == 0.0
    assert metrics["failure_rate"] == 0.0
    assert metrics["mean_completion_seconds"] is None

    for _ in range(20):
        await ev.record("zero_baseline", 0)
    assert await ev.detect_regressions("zero_baseline", window=5) == []


@pytest.mark.asyncio
async def test_detect_regressions_invalid_window_is_safe() -> None:
    """A non-positive window must not trigger a division by zero."""
    ev = Evaluation()
    await ev.record("latency", 1)
    assert await ev.detect_regressions("latency", window=0) == []
    assert await ev.detect_regressions("latency", window=-1) == []


@pytest.mark.asyncio
async def test_record_rejects_non_finite_values() -> None:
    """NaN/Inf must be rejected up front instead of failing at the DB layer.

    SQLite stores NaN as NULL, which violates metrics.value NOT NULL and
    surfaces as an opaque IntegrityError rather than a clear validation error.
    """
    ev = Evaluation()
    with pytest.raises(ValueError):
        await ev.record("nan", float("nan"))
    with pytest.raises(ValueError):
        await ev.record("inf", float("inf"))
    with pytest.raises(ValueError):
        await ev.record("neg_inf", float("-inf"))
    assert await ev.report() == {}


@pytest.mark.asyncio
async def test_adopt_only_transitions_proposed_to_adopted() -> None:
    """Adoption is a one-way transition; re-adopting must not rewrite the audit trail."""
    ev = Evaluation()
    proposal = await ev.propose("cache tools", "measured", {"calls": 20})
    first = await ev.adopt(proposal["id"], tests_passed=True, approver="alice")
    assert first["approver"] == "alice"

    with pytest.raises(ValueError):
        await ev.adopt(proposal["id"], tests_passed=True, approver="bob")

    stored = (await ev.proposals())[0]
    assert stored["status"] == "adopted"
    assert stored["approver"] == "alice"


@pytest.mark.asyncio
async def test_adopt_rejects_blank_approver() -> None:
    """A whitespace-only approver is not a human approval and must be rejected."""
    ev = Evaluation()
    proposal = await ev.propose("cache tools", "measured", {"calls": 20})
    with pytest.raises(ValueError):
        await ev.adopt(proposal["id"], tests_passed=True, approver="   ")
    stored = (await ev.proposals())[0]
    assert stored["status"] == "proposed"


@pytest.mark.asyncio
async def test_propose_ids_unique_under_timestamp_collision(monkeypatch) -> None:
    """Proposal IDs must stay unique even if two calls share a wall-clock tick.

    The old timestamp-only id collided when a second proposal was created in
    the same microsecond; ``INSERT OR REPLACE`` then silently dropped one.
    """
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(evaluation_module, "datetime", FrozenDateTime)
    ev = Evaluation()
    a = await ev.propose("a", "rationale", {"x": 1})
    b = await ev.propose("b", "rationale", {"y": 2})

    assert a["id"] != b["id"]
    assert {p["id"] for p in await ev.proposals()} == {a["id"], b["id"]}
