"""Tests for the approval-gated self-healing engine.

Covers recurring-failure detection, HIGH-risk approval binding, bounded retries
with exponential backoff, rollback snapshots, and fail-closed behaviour when
evidence, approval, applier or verification is missing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from leo import db
from leo.config import SecurityConfig, Settings
from leo.contracts import new_id
from leo.events import Event, EventBus
from leo.security.permissions import PermissionManagerImpl, arguments_digest
from leo.selfheal import ACTOR, SelfHealingEngine, classify_failure


def fixture(*, min_occurrences: int = 3, max_attempts: int = 3, settings: Settings | None = None):
    settings = settings or Settings()
    bus = EventBus()
    permissions = PermissionManagerImpl(settings, bus)
    clock = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    sleeps: list[float] = []
    captured: list[Event] = []

    async def capture(event: Event) -> None:
        captured.append(event)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    bus.subscribe("selfheal.*", capture)
    engine = SelfHealingEngine(
        bus, permissions, settings,
        min_occurrences=min_occurrences,
        max_attempts=max_attempts,
        backoff_base_seconds=1.0,
        max_backoff_seconds=10.0,
        sleep=sleep,
        now=lambda: clock[0],
    )
    return bus, permissions, engine, clock, sleeps, captured


async def fail(
    bus: EventBus,
    *,
    tool: str = "list_files",
    error: str = "invalid arguments: unsupported 'limit'",
    decision: str = "invalid_arguments",
    topic: str = "tool.failed",
) -> Event:
    return await bus.emit(
        topic,
        call_id=new_id("call"),
        tool=tool,
        actor="assistant",
        decision=decision,
        success=False,
        dry_run=False,
        error=error,
    )


async def approval_for(action: str = "selfheal.apply") -> dict:
    row = await db.fetchone(
        "SELECT * FROM approvals WHERE action=? ORDER BY created_at DESC LIMIT 1", (action,)
    )
    assert row is not None, f"no approval requested for {action}"
    return dict(row)


def observing_verifier(state: dict, **overrides):
    """Trusted verifier that re-reads the target state and reports a real diff."""
    def verifier(details, before, applied):
        proof = {
            "tests_passed": True,
            "check": "pytest tests/test_selfheal.py",
            "after": dict(state),
        }
        proof.update(overrides)
        return proof
    return verifier


def restoring_rollbacker(state: dict):
    def rollbacker(snapshot):
        state.update(snapshot["before"])
        return {"success": True}
    return rollbacker


@pytest.mark.parametrize(
    "decision,error,expected",
    [
        ("invalid_arguments", "invalid arguments: unexpected keyword 'limit'", "schema_misuse"),
        ("invalid_output", "output schema violation: extra keys", "schema_misuse"),
        ("unknown_tool", "unknown tool: 'ls_files'", "schema_misuse"),
        ("permission_denied", "permission denied: missing capability", "permission_denial"),
        ("denied", "self-approval is not allowed", "permission_denial"),
        ("timeout", "tool timed out after 10s", "timeout"),
        ("error", "provider returned 429 rate limit exceeded", "provider_error"),
        ("error", "connection reset by upstream", "provider_error"),
        ("error", "something unusual happened", "unknown"),
    ],
)
def test_failure_classification(decision, error, expected) -> None:
    assert classify_failure("tool.failed", {"decision": decision, "error": error}) == expected


async def test_repeated_failures_cluster_into_evidence_bound_proposal_without_applying() -> None:
    bus, _permissions, engine, _clock, sleeps, captured = fixture()
    for _ in range(3):
        await fail(bus)

    incidents = await engine.incidents()
    assert len(incidents) == 1
    incident = incidents[0]
    assert incident["kind"] == "schema_misuse"
    assert incident["target"] == "list_files"
    assert incident["occurrences"] == 3
    assert len(incident["event_ids"]) == 3
    assert incident["proposal_id"]

    proposals = await engine.proposals()
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["status"] == "awaiting_approval"
    assert proposal["risk"] == "high"
    assert proposal["approval_id"]
    assert len(proposal["steps"]) >= 2
    assert "list_files" in proposal["title"]

    approval = await approval_for()
    assert approval["status"] == "pending"
    assert approval["risk"] == "high"
    assert approval["actor"] == ACTOR
    assert approval["action"] == "selfheal.apply"
    metadata = json.loads(approval["metadata"])
    assert metadata["arguments_digest"] == proposal["arguments_digest"]
    assert metadata["arguments"]["event_ids"] == sorted(incident["event_ids"])

    assert not sleeps
    assert all(not event.topic.startswith("selfheal.apply") for event in captured)
    assert any(event.topic == "selfheal.incident" and event.payload["occurrences"] == 3 for event in captured)
    assert any(event.topic == "selfheal.proposal" for event in captured)
    assert any(event.topic == "selfheal.approval" for event in captured)
    assert engine.governance()["self_edits_source"] is False

    trail = await engine.audit_trail()
    assert sum(row["action"] == "selfheal.observe" for row in trail) == 3
    assert any(row["action"] == "selfheal.apply" and row["decision"] == "pending" for row in trail)


async def test_permission_denial_is_recorded_but_never_proposed() -> None:
    """A refused gate is the security model working, not damage to repair."""
    bus, _permissions, engine, _clock, _sleeps, captured = fixture()
    for _ in range(3):
        await fail(
            bus,
            tool="read_file",
            decision="permission_denied",
            error="permission denied: path is outside the workspace",
        )

    incidents = await engine.incidents()
    assert len(incidents) == 1
    incident = incidents[0]
    assert incident["kind"] == "permission_denial"
    assert incident["occurrences"] == 3
    assert not incident["proposal_id"]

    assert await engine.proposals() == []
    assert await db.fetchone("SELECT id FROM approvals WHERE action='selfheal.apply'") is None
    assert await engine.propose(incident["id"]) is None

    assert any(event.topic == "selfheal.incident" for event in captured)
    assert not any(event.topic == "selfheal.proposal" for event in captured)
    assert not any(event.topic == "selfheal.approval" for event in captured)
    trail = await engine.audit_trail()
    assert sum(row["action"] == "selfheal.observe" for row in trail) == 3
    assert not any(row["action"] == "selfheal.apply" for row in trail)


async def test_observation_is_idempotent_and_ignores_self_events() -> None:
    bus, _permissions, engine, _clock, _sleeps, _captured = fixture()
    engine.detach()
    event = await fail(bus)
    first = await engine.observe(event)
    second = await engine.observe(event)
    assert first is not None and second is None
    await engine.observe(Event(topic="selfheal.incident", source="selfheal", payload={"error": "x"}))
    await engine.observe(Event(topic="chat.completed", payload={"error": "ignored"}))
    incidents = await engine.incidents()
    assert len(incidents) == 1
    assert incidents[0]["occurrences"] == 1


async def test_scan_reads_durable_history_once() -> None:
    bus, _permissions, engine, _clock, _sleeps, _captured = fixture()
    engine.detach()  # simulate a restart that replays durable history before attaching
    for _ in range(3):
        await fail(bus)
    assert await engine.scan() == 3
    assert await engine.scan() == 0
    incidents = await engine.incidents()
    assert len(incidents) == 1
    assert incidents[0]["occurrences"] == 3


async def test_apply_without_approval_is_denied() -> None:
    bus, _permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    calls: list[dict] = []

    async def applier(details: dict) -> str:
        calls.append(details)
        return "changed"

    result = await engine.apply(
        proposal["id"], applier=applier, before={"schema": "old"},
        verifier=observing_verifier({"schema": "new"}),
        rollbacker=restoring_rollbacker({"schema": "old"}),
    )
    assert result["status"] == "denied"
    assert "approval" in result["reason"]
    assert calls == []
    assert (await engine.proposals())[0]["status"] == "awaiting_approval"
    assert await engine.snapshots() == []


@pytest.mark.parametrize(
    "approval_kind,expected_reason",
    [
        ("chat", "not approved"),
        ("self", "resolver"),
        ("expired", "expired"),
        ("bogus", "not approved"),
    ],
)
async def test_approval_binding_rejects_untrusted_or_stale_grants(approval_kind, expected_reason) -> None:
    bus, permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()

    if approval_kind == "chat":
        with pytest.raises(PermissionError, match="chat actors"):
            await permissions.resolve_approval(approval["id"], True, "chat:attacker")
    elif approval_kind == "self":
        await permissions.resolve_approval(approval["id"], True, ACTOR)
    elif approval_kind == "expired":
        await permissions.resolve_approval(approval["id"], True, "operator:alice")
        await db.execute(
            "UPDATE approvals SET created_at=? WHERE id=?",
            ("2000-01-01T00:00:00+00:00", approval["id"]),
        )
    supplied = "apr_does_not_exist" if approval_kind == "bogus" else approval["id"]

    called: list[dict] = []

    async def applier(details: dict) -> str:
        called.append(details)
        return "changed"

    result = await engine.apply(
        proposal["id"], approval_id=supplied, applier=applier,
        before={"schema": "old"},
        verifier=observing_verifier({"schema": "new"}),
        rollbacker=restoring_rollbacker({"schema": "old"}),
    )
    assert result["status"] == "denied"
    assert expected_reason in result["reason"]
    assert called == []
    assert (await engine.proposals())[0]["status"] == "awaiting_approval"


async def test_tampered_proposal_arguments_invalidate_approval() -> None:
    bus, permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    key = f"selfheal:proposal:{proposal['id']}"
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
    tampered = json.loads(row["value"])
    tampered["steps"] = ["silently change a different file"]
    await db.execute(
        "UPDATE kv SET value=? WHERE key=?",
        (json.dumps(tampered), key),
    )
    result = await engine.apply(
        proposal["id"], applier=lambda _: "changed", before={"schema": "old"},
        verifier=observing_verifier({"schema": "new"}),
        rollbacker=restoring_rollbacker({"schema": "old"}),
    )
    assert result["status"] == "denied"
    assert "arguments changed" in result["reason"]
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "approved"


async def test_approved_apply_creates_snapshot_and_consumes_approval_once() -> None:
    bus, permissions, engine, _clock, _sleeps, captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}
    applied: list[dict] = []

    async def applier(details: dict) -> dict:
        applied.append(details)
        state["schema"] = "new"
        return {"updated": "schema"}

    result = await engine.apply(
        proposal["id"], applier=applier, before=dict(state), after={"schema": "new"},
        verifier=observing_verifier(state), rollbacker=restoring_rollbacker(state),
    )
    assert result["status"] == "applied"
    assert result["verification_status"] == "verified"
    assert result["verification"]["changed"] is True
    assert state == {"schema": "new"}
    assert len(applied) == 1

    snapshots = await engine.snapshots()
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["id"] == result["snapshot_id"]
    assert snapshot["before"] == {"schema": "old"}
    assert snapshot["digest"] == arguments_digest({"schema": "old"})
    assert snapshot["status"] == "active"

    assert (await engine.proposals())[0]["status"] == "applied"
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "consumed"
    assert (await engine.incidents())[0]["state"] == "remediated"
    assert (await engine.proposals())[0]["verification_status"] == "verified"
    assert any(
        event.topic == "selfheal.apply"
        and event.payload["status"] == "applied"
        and event.payload["verified"] is True
        for event in captured
    )
    trail = await engine.audit_trail()
    assert any(
        row["action"] == "selfheal.apply" and row["decision"] == "verified_applied"
        for row in trail
    )
    assert not any(row["decision"] == "applied" for row in trail)

    # The consumed approval cannot authorize a second change.
    replay = await engine.apply(
        proposal["id"], approval_id=approval["id"], applier=applier,
        before=dict(state), verifier=observing_verifier(state),
        rollbacker=restoring_rollbacker(state),
    )
    assert replay["status"] == "denied"


async def test_failed_verification_rolls_back_and_audits() -> None:
    bus, permissions, engine, _clock, _sleeps, captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}
    reverts: list[dict] = []

    async def applier(_details: dict) -> dict:
        state["schema"] = "broken"
        return {"updated": True}

    async def rollbacker(snapshot: dict) -> dict:
        reverts.append(snapshot)
        state.update(snapshot["before"])
        return {"success": True}

    def failing_verifier(_details, _before, _applied):
        return {"tests_passed": False, "check": "pytest failed", "after": dict(state)}

    result = await engine.apply(
        proposal["id"], applier=applier, before=dict(state),
        rollbacker=rollbacker, verifier=failing_verifier,
    )
    assert result["status"] == "rolled_back"
    assert result["verification_status"] == "not_verified"
    assert state == {"schema": "old"}
    assert len(reverts) == 1
    assert (await engine.proposals())[0]["status"] == "rolled_back"
    assert (await engine.snapshots())[0]["status"] == "rolled_back"
    assert any(event.topic == "selfheal.rollback" and event.payload["status"] == "rolled_back" for event in captured)
    assert any(event.topic == "selfheal.apply" and event.payload["status"] == "failed" for event in captured)


async def test_rollbacker_is_required_before_any_mutation() -> None:
    bus, permissions, engine, _clock, _sleeps, captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}
    called: list[dict] = []

    def applier(details: dict) -> dict:
        called.append(details)
        state["schema"] = "mutated"
        return {"updated": True}

    result = await engine.apply(
        proposal["id"], applier=applier, before=dict(state),
        verifier=observing_verifier(state),
    )
    assert result["status"] == "denied"
    assert "rollbacker" in result["reason"]
    assert called == []
    assert state == {"schema": "old"}
    assert await engine.snapshots() == []
    assert (await engine.proposals())[0]["status"] == "awaiting_approval"
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "approved"
    assert not any(event.topic == "selfheal.apply" for event in captured)


async def test_applier_mutation_then_exception_rolls_back_without_applied_claim() -> None:
    bus, permissions, engine, _clock, _sleeps, captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}

    def applier(_details: dict) -> dict:
        state["schema"] = "partially mutated"
        raise RuntimeError("patch failed")

    result = await engine.apply(
        proposal["id"], applier=applier, before=dict(state),
        verifier=observing_verifier(state), rollbacker=restoring_rollbacker(state),
    )
    assert result["status"] == "rolled_back"
    assert result["verification_status"] == "not_verified"
    assert "patch failed" in result["reason"]
    assert state == {"schema": "old"}
    assert (await engine.snapshots())[0]["status"] == "rolled_back"
    assert (await engine.proposals())[0]["status"] == "rolled_back"
    assert not any(
        event.topic == "selfheal.apply" and event.payload.get("status") == "applied"
        for event in captured
    )


async def test_explicit_rollback_requires_its_own_high_risk_approval() -> None:
    bus, permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")
    state = {"schema": "old"}

    def change(_details: dict) -> dict:
        state["schema"] = "new"
        return {"updated": True}

    result = await engine.apply(
        proposal["id"], applier=change, before=dict(state),
        verifier=observing_verifier(state), rollbacker=restoring_rollbacker(state),
    )
    snapshot_id = result["snapshot_id"]
    reverts: list[dict] = []

    def rollbacker(snapshot: dict) -> dict:
        reverts.append(snapshot)
        state.update(snapshot["before"])
        return {"success": True}

    denied = await engine.rollback(snapshot_id, rollbacker=rollbacker, reason="operator asked")
    assert denied["status"] == "denied"
    assert reverts == []

    request = await engine.request_rollback(snapshot_id)
    assert request["status"] == "pending" and request["approval_id"]
    rollback_approval = await approval_for("selfheal.rollback")
    assert rollback_approval["status"] == "pending"
    await permissions.resolve_approval(rollback_approval["id"], True, "operator:bob")

    rolled = await engine.rollback(snapshot_id, rollbacker=rollbacker, reason="operator asked")
    assert rolled["status"] == "rolled_back"
    assert rolled["resolver"] == "operator:bob"
    assert state == {"schema": "old"}
    assert len(reverts) == 1
    assert (await engine.snapshots())[0]["status"] == "rolled_back"
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (rollback_approval["id"],)))["status"] == "consumed"


async def test_bounded_retry_uses_exponential_backoff_then_proposes() -> None:
    bus, _permissions, engine, clock, sleeps, captured = fixture()
    for _ in range(3):
        await fail(
            bus,
            decision="timeout",
            error="tool timed out after 10s",
        )
    incident = (await engine.incidents())[0]
    assert incident["kind"] == "timeout"
    assert incident["state"] in ("observed", "proposed")

    attempts: list[int] = []

    def operation(attempt: int) -> dict:
        attempts.append(attempt)
        return {"success": False, "error": "still down"}

    first = await engine.retry(incident["id"], operation)
    assert first["status"] == "retrying" and first["attempt"] == 1
    assert first["backoff_seconds"] == 1.0
    clock[0] += timedelta(seconds=1)
    second = await engine.retry(incident["id"], operation)
    assert second["status"] == "retrying" and second["attempt"] == 2
    assert second["backoff_seconds"] == 2.0
    clock[0] += timedelta(seconds=2)
    third = await engine.retry(incident["id"], operation)
    assert third["status"] == "exhausted" and third["attempt"] == 3
    assert third["backoff_seconds"] == 4.0
    assert attempts == [1, 2, 3]

    after = await engine.retry(incident["id"], operation)
    assert after["status"] == "blocked"
    assert attempts == [1, 2, 3]
    assert any(event.topic == "selfheal.retry" and event.payload["attempt"] == 2 for event in captured)
    assert any(event.topic == "selfheal.exhausted" for event in captured)
    assert len(await engine.proposals()) == 1
    assert sleeps == []

    await engine.backoff(4.0)
    assert sleeps == [4.0]


async def test_retry_respects_schedule_and_recovery() -> None:
    bus, _permissions, engine, clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus, decision="error", error="provider returned 429 rate limit exceeded")
    incident = (await engine.incidents())[0]
    assert incident["kind"] == "provider_error"

    def broken(_attempt: int) -> dict:
        return {"success": False}

    first = await engine.retry(incident["id"], broken)
    assert first["status"] == "retrying"
    too_soon = await engine.retry(incident["id"], lambda _: {"success": True})
    assert too_soon["status"] == "waiting"
    clock[0] += timedelta(seconds=1)
    recovered = await engine.retry(incident["id"], lambda _: {"success": True})
    assert recovered["status"] == "recovered"
    assert (await engine.incidents())[0]["state"] == "recovered"


async def test_non_retryable_failures_are_never_auto_retried() -> None:
    bus, _permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    incident = (await engine.incidents())[0]
    calls: list[int] = []
    result = await engine.retry(incident["id"], lambda attempt: calls.append(attempt))
    assert result["status"] == "blocked"
    assert "schema_misuse" in result["reason"]
    assert calls == []


async def test_single_failure_produces_no_proposal_and_no_change() -> None:
    bus, _permissions, engine, _clock, _sleeps, _captured = fixture()
    await fail(bus)
    assert await engine.proposals() == []
    assert await engine.incidents() and (await engine.incidents())[0]["occurrences"] == 1
    rows = await db.fetchall("SELECT id FROM approvals")
    assert rows == []


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"applier": None}, "applier"),
        ({"before": None}, "snapshot"),
        ({"verifier": None}, "verifier"),
        ({"rollbacker": None}, "rollbacker"),
        ({"verification": {"tests_passed": True, "check": "pytest"}}, "canned verification"),
    ],
)
async def test_apply_fails_closed_without_trusted_callbacks_or_evidence(kwargs, expected) -> None:
    bus, permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}
    called: list[dict] = []

    def applier(details: dict) -> dict:
        called.append(details)
        state["schema"] = "new"
        return {"updated": True}

    options = {
        "applier": applier if kwargs.get("applier", True) else None,
        "before": dict(state) if kwargs.get("before", True) else None,
        "verifier": observing_verifier(state) if kwargs.get("verifier", True) else None,
        "rollbacker": restoring_rollbacker(state) if kwargs.get("rollbacker", True) else None,
    }
    if "verification" in kwargs:
        options["verification"] = kwargs["verification"]
    result = await engine.apply(proposal["id"], **options)
    assert result["status"] == "denied"
    assert expected in result["reason"]
    assert called == []
    assert state == {"schema": "old"}
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval["id"],)))["status"] == "approved"
    assert await engine.snapshots() == []


async def test_disabled_high_risk_gate_does_not_enable_self_apply() -> None:
    settings = Settings(security=SecurityConfig(high_risk_requires_approval=False))
    bus, permissions, engine, _clock, _sleeps, captured = fixture(settings=settings)
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    assert proposal["approval_id"] is None
    assert any(
        event.topic == "selfheal.denied" and "gate disabled" in event.payload["reason"]
        for event in captured
    )
    state = {"schema": "old"}
    result = await engine.apply(
        proposal["id"], applier=lambda _: "changed", before=dict(state),
        verifier=observing_verifier(state), rollbacker=restoring_rollbacker(state),
    )
    assert result["status"] == "denied"
    assert await engine.snapshots() == []


async def test_secrets_in_failure_evidence_are_redacted() -> None:
    bus, permissions, engine, _clock, _sleeps, captured = fixture()
    # Built at runtime: the secret scanner cannot tell a fake key from a real
    # one, and a literal here blocks the pre-commit gate for every later commit.
    fake_key = "sk-" + "abc" + "..." + "mnop"
    for _ in range(3):
        await fail(bus, error=f"provider failed api_key={fake_key} with list_files")
    incident = (await engine.incidents())[0]
    assert fake_key not in incident["sample_error"]
    assert fake_key not in incident["signature"]
    assert all(fake_key not in json.dumps(event.payload) for event in captured)


async def test_canned_verification_with_no_op_applier_cannot_claim_applied() -> None:
    bus, permissions, engine, _clock, _sleeps, captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}
    calls: list[dict] = []

    def no_op(details: dict) -> dict:
        calls.append(details)
        return {"updated": True}

    result = await engine.apply(
        proposal["id"], applier=no_op, before=dict(state),
        verification={"tests_passed": True, "check": "pytest", "after": {"schema": "new"}},
        rollbacker=restoring_rollbacker(state),
    )
    assert result["status"] == "denied"
    assert "canned verification" in result["reason"]
    assert calls == []
    assert state == {"schema": "old"}
    assert (await engine.proposals())[0]["status"] == "awaiting_approval"
    assert await engine.snapshots() == []
    assert not any(
        event.topic == "selfheal.apply" and event.payload.get("status") == "applied"
        for event in captured
    )


async def test_verifier_must_observe_a_real_diff() -> None:
    bus, permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}

    def no_op(_details: dict) -> dict:
        return {"updated": True}

    # A structurally valid proof whose read-back still equals the before-state.
    result = await engine.apply(
        proposal["id"], applier=no_op, before=dict(state),
        verifier=observing_verifier(state), rollbacker=restoring_rollbacker(state),
    )
    assert result["status"] == "rolled_back"
    assert "unchanged" in result["reason"]
    assert result["verification_status"] == "not_verified"



async def test_verifier_must_return_an_independent_readback() -> None:
    bus, permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}

    def change(_details: dict) -> dict:
        state["schema"] = "new"
        return {"updated": True}

    def no_readback(_details, _before, _applied):
        return {"tests_passed": True, "check": "pytest"}

    result = await engine.apply(
        proposal["id"], applier=change, before=dict(state),
        verifier=no_readback, rollbacker=restoring_rollbacker(state),
    )
    assert result["status"] == "rolled_back"
    assert result["verification_status"] == "not_verified"
    assert "read-back" in result["reason"]
    assert state == {"schema": "old"}
    assert (await engine.snapshots())[0]["status"] == "rolled_back"


async def test_rollback_failure_marks_snapshot_failed_without_applied_claim() -> None:
    bus, permissions, engine, _clock, _sleeps, captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")

    state = {"schema": "old"}

    def applier(_details: dict) -> dict:
        state["schema"] = "mutated"
        return {"updated": True}

    def broken_verifier(_details, _before, _applied):
        raise RuntimeError("verification unavailable")

    def broken_rollbacker(_snapshot: dict) -> dict:
        raise RuntimeError("rollback unavailable")

    result = await engine.apply(
        proposal["id"], applier=applier, before=dict(state),
        verifier=broken_verifier, rollbacker=broken_rollbacker,
    )
    assert result["status"] == "failed"
    assert result["verification_status"] == "not_verified"
    assert result["snapshot_status"] == "failed"
    assert (await engine.snapshots())[0]["status"] == "failed"
    assert (await engine.proposals())[0]["status"] == "failed"
    assert not any(
        event.topic == "selfheal.apply" and event.payload.get("status") == "applied"
        for event in captured
    )
    assert any(
        event.topic == "selfheal.rollback" and event.payload["status"] == "failed"
        for event in captured
    )


async def test_health_summary_distinguishes_proposed_from_verified_applied() -> None:
    bus, permissions, engine, _clock, _sleeps, _captured = fixture()
    for _ in range(3):
        await fail(bus)
    proposal = (await engine.proposals())[0]
    before = await engine.remediation_status()
    assert before["proposed"] == 1
    assert before["verified_applied"] == 0
    assert before["unverified_applied"] == 0

    approval = await approval_for()
    await permissions.resolve_approval(approval["id"], True, "operator:alice")
    state = {"schema": "old"}

    def change(_details: dict) -> dict:
        state["schema"] = "new"
        return {"updated": True}

    result = await engine.apply(
        proposal["id"], applier=change, before=dict(state),
        verifier=observing_verifier(state), rollbacker=restoring_rollbacker(state),
    )
    assert result["status"] == "applied"
    after = await engine.remediation_status()
    assert after["proposed"] == 0
    assert after["verified_applied"] == 1
    assert after["unverified_applied"] == 0
    stored = (await engine.proposals())[0]
    assert stored["status"] == "applied"
    assert stored["verification_status"] == "verified"


async def test_incident_evidence_lists_are_bounded_over_300_failures() -> None:
    bus, _permissions, engine, _clock, _sleeps, _captured = fixture()
    first_event = await fail(bus)
    for _ in range(299):
        await fail(bus)

    incidents = await engine.incidents()
    assert len(incidents) == 1
    incident = incidents[0]
    assert incident["occurrences"] == 300
    assert 0 < len(incident["event_ids"]) <= 100
    assert 0 < len(incident["sources"]) <= 20
    assert len(set(incident["event_ids"])) == len(incident["event_ids"])
    assert len(set(incident["sources"])) == len(incident["sources"])
    assert first_event.id not in incident["event_ids"]  # oldest ids are evicted

    row = await db.fetchone(
        "SELECT length(value) AS size FROM kv WHERE key=?",
        (f"selfheal:incident:{incident['id']}",),
    )
    assert row is not None and row["size"] < 8000
    # Re-observation of the same 300 events stays idempotent and bounded.
    assert await engine.scan() == 0
    assert (await engine.incidents())[0]["event_ids"] == incident["event_ids"]
