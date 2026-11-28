"""Canonical domain DTOs, digests and the schedule SQL seam."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel
from leo.events import EventBus
from leo.schedules.clock import FrozenClock
from leo.security import PermissionManagerImpl
from leo.tasks import DurableTaskEngine, Scheduler
from leo.schedules.domain import (
    DeliveryTarget,
    ExecutionSpec,
    ScheduleExpression,
    ScheduleProposal,
    ScheduleSpec,
    ScheduleView,
    canonical_json,
    sha256_digest,
)
from leo.schedules.store import ScheduleStore


def _spec(**overrides) -> ScheduleSpec:
    values = {
        "name": "Water",
        "expression": ScheduleExpression(
            kind="interval", expression="60", timezone="UTC"
        ),
        "prompt": "Remind the user to drink water.",
        "repeat_limit": 3,
        "delivery": [
            DeliveryTarget(
                kind="external", platform="Telegram", chat_id="123", thread_id="5"
            )
        ],
    }
    values.update(overrides)
    return ScheduleSpec(**values)


def test_changes_between_lists_old_and_new() -> None:
    """Teknium (Hermes PR #158289): a 'Change this job?' card lists each
    change as old -> new, never just 'changed'."""
    from leo.schedules.proposals import changes_between

    changes = changes_between(
        _spec(),
        _spec(
            prompt="Drink tea instead.",
            delivery=[DeliveryTarget(kind="local", platform="local", chat_id="cli")],
            execution=ExecutionSpec(script="check.py", script_args=["--fast"]),
            catch_up=False,
        ),
    )
    assert "prompt: Remind the user to drink water. -> Drink tea instead." in changes
    assert "delivery: telegram:123 -> local:cli" in changes
    assert "script: none -> check.py" in changes
    assert "catch-up: none -> false" in changes
    assert not any("changed" in item and "->" not in item for item in changes)


def test_canonical_json_is_order_independent_and_compact() -> None:
    left = {"b": [2, 1], "a": {"y": None, "x": "é"}}
    right = {"a": {"x": "é", "y": None}, "b": [2, 1]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":{"x":"é","y":null},"b":[2,1]}'
    assert sha256_digest(left) == sha256_digest(right)
    assert sha256_digest(left) == sha256_digest(dict(reversed(list(left.items()))))


def test_digest_binds_every_semantic_field() -> None:
    base = _spec()
    baseline = base.digest()
    assert baseline == base.content_digest()
    assert len(baseline) == 64

    assert _spec(prompt="Drink tea instead.").digest() != baseline
    assert _spec(name="Coffee").digest() != baseline
    assert _spec(repeat_limit=4).digest() != baseline
    assert _spec(
        delivery=[
            DeliveryTarget(kind="local", platform="local", chat_id="cli")
        ]
    ).digest() != baseline
    assert _spec(
        execution=ExecutionSpec(script="check.py", script_args=["--fast"])
    ).digest() != baseline
    # Delivery target identity is case-insensitive, so its *rendered* form is
    # not part of identity; the target still participates in the spec digest.
    assert _spec(
        delivery=[
            DeliveryTarget(
                kind="external", platform="telegram", chat_id="123", thread_id="5"
            )
        ]
    ).digest() == baseline


def test_dto_round_trips_preserve_content_and_digest() -> None:
    spec = _spec(
        execution=ExecutionSpec(
            script="check.py",
            script_args=["--fast"],
            enabled_toolsets=[],
            skills=["maps"],
            wrap_response="off",
            workdir="/workspace/brief",
            attach_to_session=True,
            pinned=True,
            provider="openai",
            model="gpt-x",
            grace_minutes=15,
        ),
        paused=True,
        paused_reason="operator hold",
        catch_up=False,
        config={"note": "keep"},
    )
    encoded = spec.canonical_json()
    restored = ScheduleSpec.model_validate_json(encoded)
    assert restored.digest() == spec.digest()
    assert restored.execution.enabled_toolsets == []
    assert restored.execution.script_args == ["--fast"]

    proposal = ScheduleProposal(
        principal="model:chat",
        operation="create",
        spec=spec,
        digest=spec.digest(),
    )
    proposal_round = ScheduleProposal.model_validate_json(proposal.canonical_json())
    assert proposal_round.spec.digest() == spec.digest()
    assert proposal_round.status == "pending"


def test_schedule_expression_and_execution_validation() -> None:
    with pytest.raises(ValueError, match="once schedule requires run_at"):
        ScheduleExpression(kind="once")
    with pytest.raises(ValueError, match="interval schedule requires an expression"):
        ScheduleExpression(kind="interval")
    assert ScheduleExpression(kind="once", run_at="2026-11-27T09:00:00+00:00").run_at

    with pytest.raises(ValueError, match="no_agent requires a script"):
        ExecutionSpec(no_agent=True)
    with pytest.raises(ValueError, match="incompatible with no_agent"):
        ExecutionSpec(no_agent=True, script="x.py", monitor="https://example.com")
    continuity = ExecutionSpec(continuity=True)
    assert continuity.context_from == ["self"]


def test_delivery_target_identity_is_normalized() -> None:
    target = DeliveryTarget(
        kind="external", platform="Telegram", chat_id="123", thread_id="7"
    )
    assert target.identity() == ("telegram", "123", "7")
    assert target.identity() == DeliveryTarget(
        kind="external", platform="telegram", chat_id="123", thread_id="7"
    ).identity()


def test_frozen_clock_is_deterministic_and_forward_only() -> None:
    clock = FrozenClock("2026-11-26T10:00:00+00:00")
    assert clock.now().isoformat() == "2026-11-26T10:00:00+00:00"
    clock.advance(seconds=90)
    assert clock.now().isoformat() == "2026-11-26T10:01:30+00:00"
    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.advance(seconds=-1)
    naive = FrozenClock(datetime(2026, 11, 26, 12, 0))
    assert naive.now().tzinfo is timezone.utc


@pytest.mark.asyncio
async def test_store_reads_the_existing_schedule_table_and_owns_proposals() -> None:
    now = "2026-11-26T10:00:00+00:00"
    await db.execute(
        "INSERT INTO schedules(id,name,kind,expression,payload,enabled,next_run_at,"
        "last_run_at,created_at,schedule_type,run_at,timezone,repeat_limit,"
        "claimed_count,completed_count,config_revision,grace_minutes,config,status,"
        "created_by,origin,consent_digest,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "sch_store", "stored", "interval", "60", json.dumps({"objective": "work"}),
            1, "2026-11-26T10:01:00+00:00", None, now, "interval", None,
            "America/New_York", 5, 2, 1, 3, 10, json.dumps({"a": 1}), "active",
            "operator:cli", "cli:1", "consent-digest", now,
        ),
    )
    store = ScheduleStore(FrozenClock(now))
    view = await store.get("sch_store")
    assert isinstance(view, ScheduleView)
    assert view.name == "stored" and view.payload["objective"] == "work"
    assert view.schedule_type == "interval" and view.timezone == "America/New_York"
    assert view.repeat_limit == 5
    assert view.claimed_count == 2 and view.completed_count == 1
    assert view.config_revision == 3 and view.grace_minutes == 10
    assert view.config == {"a": 1}
    assert view.created_by == "operator:cli" and view.origin == "cli:1"
    assert view.consent_digest == "consent-digest" and view.updated_at == now
    assert (await store.get_by_name("stored", active_only=True)).id == "sch_store"
    assert await store.name_is_active("stored") is True
    assert "stored" in await store.active_names()
    assert [item.id for item in await store.list(include_paused=False)] == ["sch_store"]

    spec = _spec()
    proposal = await store.create_proposal(
        ScheduleProposal(
            id="prop_1",
            active_name="Water",
            principal="model:chat:1",
            operation="create",
            request_key="req-1",
            spec=spec,
        )
    )
    assert proposal.digest == spec.digest()
    assert proposal.created_at == now
    loaded = await store.get_proposal("prop_1")
    assert loaded is not None and loaded.spec.digest() == spec.digest()
    by_request = await store.find_proposal_by_request("model:chat:1", "create", "req-1")
    assert by_request is not None and by_request.id == "prop_1"
    updated = await store.update_proposal(
        "prop_1", status="approved", schedule_id="sch_new", config_revision=2
    )
    assert updated is not None and updated.status == "approved"
    assert updated.schedule_id == "sch_new" and updated.config_revision == 2

    await store.add_tombstone(
        "sch_old", "stored", "one-shot occurred", origin="operator:cli"
    )
    tombstone = await store.get_tombstone("sch_old")
    assert tombstone is not None and tombstone["reason"] == "one-shot occurred"
    assert [item["schedule_id"] for item in await store.list_tombstones("stored")] == [
        "sch_old"
    ]

    await store.runtime_set("sch_store", "monitor_baseline", "abc")
    assert await store.runtime_get("sch_store", "monitor_baseline") == "abc"
    await store.runtime_delete("sch_store", "monitor_baseline")
    assert await store.runtime_get("sch_store", "monitor_baseline") is None


@pytest.mark.asyncio
async def test_scheduler_registration_and_ticks_follow_the_injected_clock() -> None:
    clock = FrozenClock("2026-11-26T10:00:00+00:00")
    events = EventBus()
    scheduler = Scheduler(
        DurableTaskEngine(Settings(), events), Settings(), events, clock=clock
    )
    registered = await scheduler.register_interval("frozen", 60, "work")
    assert registered["next_run_at"] == "2026-11-26T10:01:00+00:00"
    assert await scheduler.tick() == []
    clock.advance(seconds=61)
    due = await scheduler.tick()
    assert len(due) == 1 and due[0].metadata["schedule"] == "frozen"


@pytest.mark.asyncio
async def test_approval_expiry_uses_the_injected_clock() -> None:
    clock = FrozenClock("2026-11-26T10:00:00+00:00")
    manager = PermissionManagerImpl(Settings(), EventBus(), clock=clock)
    decision = await manager.authorize(
        "agent-a", "tool:send", None, RiskLevel.HIGH
    )
    assert decision.requires_approval and decision.approval_id
    resolved = await manager.resolve_approval(
        decision.approval_id, True, resolver="operator"
    )
    assert resolved.allowed

    # Within the timeout the approval is still valid.
    clock.advance(seconds=899)
    valid = await manager.authorize(
        "agent-a", "tool:send", None, RiskLevel.HIGH, approval_id=decision.approval_id
    )
    assert valid.allowed and valid.verified_approval is True

    # Past the configured approval timeout it must not authorize anything.
    clock.advance(seconds=2)
    expired = await manager.authorize(
        "agent-a", "tool:send", None, RiskLevel.HIGH, approval_id=decision.approval_id
    )
    assert not expired.allowed and "not valid" in expired.reason
