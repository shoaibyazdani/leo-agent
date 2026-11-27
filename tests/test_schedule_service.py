"""ScheduleService: the one proposal/consent/commit authority.

Covers direct operator writes, answerable-surface cards, fail-closed no-card
proposals, request replay, concurrent name reservation, semantic no-op edits,
CAS stale edits, repeat merge and the grant path.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import LIVE_AUTH_CLAIM_TTL_SECONDS, TurnContext
from leo.events import EventBus
from leo.security.approval_cards import (
    DECLINE_MESSAGES,
    KIND_SCHEDULE_CREATE,
    KIND_SCHEDULE_EDIT,
)
from leo.security.permissions import PermissionManagerImpl
from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompileError
from leo.schedules.service import ScheduleService, ScheduleServiceError
from leo.schedules.store import ScheduleStore
from leo.tasks import DurableTaskEngine, Scheduler

NOW = datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc)


def _operator() -> TurnContext:
    return TurnContext(
        principal="operator:cli", channel="cli", surface="cli", source="operator"
    )


def _chat(surface: str = "tui", *, conversation_id: str = "conv-a", principal: str = "chat:web") -> TurnContext:
    return TurnContext(
        principal=principal,
        channel=surface,
        conversation_id=conversation_id,
        surface=surface,
        origin=surface,
        source="chat",
    )


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    clock = FrozenClock(NOW)
    permissions = PermissionManagerImpl(settings, events, clock=clock)
    engine = DurableTaskEngine(settings, events)
    scheduler = Scheduler(engine, settings, events, clock=clock)
    store = ScheduleStore(clock=clock)
    service = ScheduleService(
        store=store,
        scheduler=scheduler,
        engine=engine,
        permissions=permissions,
        events=events,
        settings=settings,
        clock=clock,
        allowed_workdir_roots=[tmp_path],
    )
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        engine=engine, scheduler=scheduler, store=store, service=service,
        clock=clock,
    )


async def _create_operator(env, **arguments):
    return await env.service.create(
        arguments, actor="operator:cli", turn=_operator()
    )


async def _resolve(env, approval_id: str, *, choice: str = "once", digest: str, resolver: str = "operator:local"):
    return await env.permissions.resolve_schedule_card(
        approval_id,
        choice=choice,
        resolver=resolver,
        confirm_proposal_digest=digest,
        commit=env.service.commit,
    )


# --------------------------------------------------------------------------- direct operator


async def test_operator_create_is_unasked_audited_and_has_no_card(env):
    out = await _create_operator(env, name="ops", schedule="1h", prompt="check")
    assert out["status"] == "created"
    row = await db.fetchone("SELECT * FROM schedules WHERE name='ops'")
    assert row is not None and row["origin"] == "cli"
    assert row["created_by"] == "operator:cli"
    assert await db.fetchall("SELECT id FROM approvals") == []
    audit = await db.fetchone(
        "SELECT actor, decision FROM audit_log WHERE action='schedule.created'"
    )
    assert dict(audit) == {"actor": "operator:cli", "decision": "allowed"}


async def test_answerable_surface_presents_one_create_card(env):
    out = await env.service.create(
        {"name": "water", "schedule": "every day at 8am", "prompt": "Drink water."},
        actor="chat:web",
        turn=_chat("tui"),
    )
    assert out["status"] == "approval_required"
    card = out["card"]
    assert card["kind"] == KIND_SCHEDULE_CREATE
    assert card["choices"] == ["once", "session", "always", "deny"]
    assert "Drink water." in card["body"]
    assert card["shown_at"] and card["presented_surface"] == "tui"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='water'") is None
    assert await db.fetchone("SELECT id FROM approvals WHERE id=?", (card["approval_id"],)) is not None


async def test_incapable_surface_fails_closed_with_zero_schedule_and_approval_rows(env):
    out = await env.service.create(
        {"name": "cli-job", "schedule": "1h", "prompt": "Check."},
        actor="chat:cli",
        turn=_chat("cli"),
    )
    assert out["status"] == "approval_required"
    assert out["approval_id"] is None and out["card"] is None
    assert await db.fetchall("SELECT id FROM schedules") == []
    assert await db.fetchall("SELECT id FROM approvals") == []
    assert len(await db.fetchall("SELECT id FROM schedule_proposals")) == 1


async def test_concurrent_same_name_reservation_allows_exactly_one(env):
    results = await asyncio.gather(
        env.service.create(
            {"name": "race", "schedule": "1h", "prompt": "a"},
            actor="chat:one", turn=_chat("cli", conversation_id="c1", principal="chat:one"),
        ),
        env.service.create(
            {"name": "race", "schedule": "1h", "prompt": "b"},
            actor="chat:two", turn=_chat("cli", conversation_id="c2", principal="chat:two"),
        ),
        return_exceptions=True,
    )
    successes = [item for item in results if isinstance(item, dict)]
    failures = [item for item in results if isinstance(item, BaseException)]
    assert len(successes) == 1 and len(failures) == 1
    assert isinstance(failures[0], ScheduleServiceError)
    assert failures[0].code == "duplicate_name"
    proposals = await db.fetchall("SELECT id FROM schedule_proposals WHERE active_name='race'")
    assert len(proposals) == 1


async def test_request_key_replays_the_same_proposal(env):
    first = await env.service.create(
        {"name": "replay", "schedule": "1h", "prompt": "a"},
        actor="chat:web", turn=_chat("cli"), request_key="req-1",
    )
    second = await env.service.create(
        {"name": "replay", "schedule": "1h", "prompt": "different"},
        actor="chat:web", turn=_chat("cli"), request_key="req-1",
    )
    assert second["replayed"] is True
    assert second["proposal_id"] == first["proposal_id"]
    assert len(await db.fetchall("SELECT id FROM schedule_proposals")) == 1
    assert await db.fetchall("SELECT id FROM schedules") == []


async def test_committed_request_key_replays_without_a_second_write(env):
    first = await env.service.create(
        {"name": "committed", "schedule": "1h", "prompt": "a"},
        actor="operator:cli", turn=_operator(), request_key="req-op",
    )
    second = await env.service.create(
        {"name": "committed", "schedule": "1h", "prompt": "a"},
        actor="operator:cli", turn=_operator(), request_key="req-op",
    )
    assert first["status"] == second["status"] == "created"
    assert second["replayed"] is True
    assert len(await db.fetchall("SELECT id FROM schedules")) == 1


# --------------------------------------------------------------------------- card commit


async def test_once_card_commit_creates_the_schedule_atomically(env):
    out = await env.service.create(
        {"name": "water", "schedule": "1h", "prompt": "Drink water."},
        actor="chat:web", turn=_chat("tui"),
    )
    resolved = await _resolve(
        env, out["approval_id"], choice="once", digest=out["card"]["proposal_digest"]
    )
    assert resolved.status == "approved" and resolved.schedule_id
    row = await db.fetchone("SELECT * FROM schedules WHERE id=?", (resolved.schedule_id,))
    assert row is not None and row["name"] == "water"
    assert row["consent_digest"] == out["card"]["proposal_digest"]
    approval = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (out["approval_id"],)
    )
    proposal = await db.fetchone(
        "SELECT status FROM schedule_proposals WHERE id=?", (out["proposal_id"],)
    )
    assert approval["status"] == "approved"
    assert proposal["status"] == "committed"
    assert await db.fetchall("SELECT id FROM approval_grants") == []


async def test_session_choice_grants_cardless_create_in_the_same_conversation(env):
    turn = _chat("tui", conversation_id="conv-a")
    first = await env.service.create(
        {"name": "one", "schedule": "1h", "prompt": "a"},
        actor="chat:web", turn=turn,
    )
    resolved = await _resolve(
        env, first["approval_id"], choice="session",
        digest=first["card"]["proposal_digest"],
    )
    assert resolved.grant_id

    second = await env.service.create(
        {"name": "two", "schedule": "1h", "prompt": "b"},
        actor="chat:web", turn=_chat("cli", conversation_id="conv-a"),
    )
    assert second["status"] == "created"
    assert second["card"] is None
    assert await db.fetchone("SELECT id FROM schedules WHERE name='two'") is not None
    # Only the first card exists: the grant path did not leave an invisible
    # approval row behind.
    assert len(await db.fetchall("SELECT id FROM approvals")) == 1


async def test_always_grant_is_profile_scoped_and_expires(env):
    first = await env.service.create(
        {"name": "one", "schedule": "1h", "prompt": "a"},
        actor="chat:web", turn=_chat("tui", conversation_id="conv-a"),
    )
    resolved = await _resolve(
        env, first["approval_id"], choice="always",
        digest=first["card"]["proposal_digest"],
    )
    assert resolved.grant_id
    grant = await db.fetchone(
        "SELECT subject_type, subject_id FROM approval_grants WHERE id=?",
        (resolved.grant_id,),
    )
    assert dict(grant) == {"subject_type": "profile", "subject_id": "chat:web"}

    other = await env.service.create(
        {"name": "two", "schedule": "1h", "prompt": "b"},
        actor="chat:other", turn=_chat("cli", principal="chat:other", conversation_id="conv-b"),
    )
    assert other["status"] == "approval_required"
    assert other["approval_id"] is None


async def test_edit_card_commit_cas_and_revision_bump(env):
    created = await _create_operator(env, name="job", schedule="1h", prompt="a")
    assert created["status"] == "created"
    out = await env.service.update(
        name="job", arguments={"prompt": "b"},
        actor="chat:web", turn=_chat("tui"),
    )
    assert out["status"] == "approval_required"
    card = out["card"]
    assert card["kind"] == KIND_SCHEDULE_EDIT
    assert card["choices"] == ["once", "deny"]
    assert "prompt changed" in card["body"]

    resolved = await _resolve(
        env, card["approval_id"], choice="once", digest=card["proposal_digest"]
    )
    assert resolved.status == "approved"
    row = await db.fetchone("SELECT * FROM schedules WHERE name='job'")
    assert row["config_revision"] == 2
    assert json.loads(row["payload"])["objective"] == "b"
    assert await db.fetchall("SELECT id FROM approval_grants") == []


async def test_stale_edit_card_commit_expires_as_a_stale_edit(env):
    created = await _create_operator(env, name="job", schedule="1h", prompt="a")
    assert created["status"] == "created"
    pending = await env.service.update(
        name="job", arguments={"prompt": "b"},
        actor="chat:web", turn=_chat("tui"),
    )
    card = pending["card"]

    # Another writer advances the row before the operator answers.
    await env.service.update(
        name="job", arguments={"prompt": "c"},
        actor="operator:cli", turn=_operator(),
    )
    resolved = await _resolve(
        env, card["approval_id"], choice="once",
        digest=card["proposal_digest"],
    )
    # A CAS loss at commit time is an expired edit, not an exception that
    # leaves the card pending forever.
    assert resolved.status == "expired" and resolved.allowed is False
    assert resolved.decline_message == DECLINE_MESSAGES["stale_edit"]
    approval = await db.fetchone(
        "SELECT status, answered_at FROM approvals WHERE id=?", (card["approval_id"],)
    )
    assert approval["status"] == "expired" and approval["answered_at"] is not None
    proposal = await db.fetchone(
        "SELECT status FROM schedule_proposals WHERE id=?", (card["proposal_id"],)
    )
    assert proposal["status"] == "expired"
    row = await db.fetchone("SELECT config_revision FROM schedules WHERE name='job'")
    assert row["config_revision"] == 2  # only the direct operator edit committed


async def test_edit_card_for_removed_schedule_expires_as_a_stale_edit(env):
    """unknown_schedule at commit time must not leave a pending card either."""
    await _create_operator(env, name="job", schedule="1h", prompt="a")
    pending = await env.service.update(
        name="job", arguments={"prompt": "b"},
        actor="chat:web", turn=_chat("tui"),
    )
    await env.service.remove(
        name="job", actor="operator:cli", turn=_operator()
    )
    resolved = await _resolve(
        env, pending["card"]["approval_id"], choice="once",
        digest=pending["card"]["proposal_digest"],
    )
    assert resolved.status == "expired"
    assert resolved.decline_message == DECLINE_MESSAGES["stale_edit"]
    approval = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (pending["approval_id"],)
    )
    assert approval["status"] == "expired"


async def test_noop_update_does_not_reanchor_or_reserve(env):
    created = await _create_operator(
        env, name="job", schedule="1h", prompt="a", repeat=5,
    )
    assert created["status"] == "created"
    before = await db.fetchone(
        "SELECT next_run_at, config_revision FROM schedules WHERE name='job'"
    )
    out = await env.service.update(
        name="job", arguments={"prompt": "a"},
        actor="chat:web", turn=_chat("tui"),
    )
    assert out["status"] == "no_change"
    after = await db.fetchone(
        "SELECT next_run_at, config_revision FROM schedules WHERE name='job'"
    )
    assert dict(after) == dict(before)
    assert len(await db.fetchall("SELECT id FROM schedule_proposals")) == 1


async def test_repeat_merge_preserves_claimed_and_completed_counts(env):
    created = await _create_operator(
        env, name="job", schedule="1h", prompt="a", repeat=5,
    )
    assert created["status"] == "created"
    row = await db.fetchone("SELECT id FROM schedules WHERE name='job'")
    await db.execute(
        "UPDATE schedules SET claimed_count=3, completed_count=2 WHERE id=?",
        (row["id"],),
    )
    out = await env.service.update(
        name="job", arguments={"repeat": 2},
        actor="operator:cli", turn=_operator(),
    )
    assert out["status"] == "updated"
    merged = await db.fetchone("SELECT * FROM schedules WHERE id=?", (row["id"],))
    assert merged["repeat_limit"] == 2
    assert merged["claimed_count"] == 3 and merged["completed_count"] == 2


async def test_denied_card_releases_the_active_name(env):
    first = await env.service.create(
        {"name": "name", "schedule": "1h", "prompt": "a"},
        actor="chat:web", turn=_chat("tui"),
    )
    resolved = await _resolve(
        env, first["approval_id"], choice="deny",
        digest=first["card"]["proposal_digest"],
    )
    assert resolved.status == "denied"
    proposal = await db.fetchone(
        "SELECT status FROM schedule_proposals WHERE id=?", (first["proposal_id"],)
    )
    assert proposal["status"] == "denied"
    # The reservation no longer blocks a fresh proposal with the same name.
    second = await env.service.create(
        {"name": "name", "schedule": "1h", "prompt": "b"},
        actor="chat:web", turn=_chat("cli"),
    )
    assert second["status"] == "approval_required"


# --------------------------------------------------------------------------- remove / run / cancel


async def test_model_remove_is_a_card_and_operator_remove_is_direct(env):
    created = await _create_operator(env, name="job", schedule="1h", prompt="a")
    assert created["status"] == "created"
    pending = await env.service.remove(name="job", actor="chat:web", turn=_chat("tui"))
    assert pending["status"] == "approval_required"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='job'") is not None
    resolved = await _resolve(
        env, pending["card"]["approval_id"], choice="once",
        digest=pending["card"]["proposal_digest"],
    )
    assert resolved.status == "approved"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='job'") is None
    assert await db.fetchone(
        "SELECT id FROM schedule_tombstones WHERE name='job'"
    ) is not None


async def test_id_only_delete_card_cannot_delete_a_different_schedule(env):
    """M7: an empty delete subject binds to the resolved id, not every row."""
    await _create_operator(env, name="first", schedule="1h", prompt="a")
    first_id = (await db.fetchone(
        "SELECT id FROM schedules WHERE name='first'"
    ))["id"]
    await _create_operator(env, name="second", schedule="1h", prompt="b")
    second_id = (await db.fetchone(
        "SELECT id FROM schedules WHERE name='second'"
    ))["id"]

    card = await _approved_specialist_card(
        env, "schedule_delete", {"schedule_id": first_id}
    )
    stored = await db.fetchone(
        "SELECT resource FROM approvals WHERE id=?", (card,)
    )
    assert stored["resource"] == "schedule_delete:"

    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.remove(
            schedule_id=second_id, actor="automation-operator",
            authorized_by=card,
        )
    assert caught.value.code == "invalid_authorization"
    assert await db.fetchone(
        "SELECT id FROM schedules WHERE id=?", (second_id,)
    ) is not None
    assert await db.fetchone(
        "SELECT id FROM schedules WHERE id=?", (first_id,)
    ) is not None

    # The refused card is not burned by the wrong-target attempt: the same
    # card still deletes the schedule it was issued for.
    removed = await env.service.remove(
        schedule_id=first_id, actor="automation-operator",
        authorized_by=card,
    )
    assert removed["status"] == "removed"
    assert await db.fetchone(
        "SELECT id FROM schedules WHERE id=?", (first_id,)
    ) is None


@pytest.mark.parametrize("tamper", ["missing", "wrong"])
async def test_name_subject_delete_card_is_digest_bound(env, tamper):
    """Q4: a name subject alone does not authorize a service-layer delete."""
    await _create_operator(env, name="job", schedule="1h", prompt="a")
    approval_id = await _approved_specialist_card(
        env, "schedule_delete", {"name": "job"}
    )
    if tamper == "missing":
        await db.execute(
            "UPDATE approvals SET metadata=json_remove(metadata,'$.arguments_digest') "
            "WHERE id=?", (approval_id,),
        )
    else:
        await db.execute(
            "UPDATE approvals SET metadata=json_set(metadata,'$.arguments_digest',?) "
            "WHERE id=?", ("wrong-digest", approval_id),
        )

    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.remove(
            name="job", actor="automation-operator", authorized_by=approval_id,
        )
    assert caught.value.code == "invalid_authorization"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='job'") is not None
    assert (await _live_authorization_markers(approval_id))["claim"] is None


async def test_failed_delete_audit_releases_card_and_retry_succeeds(env, monkeypatch):
    """M8: a rolled-back delete releases its lease instead of burning it."""
    await _create_operator(env, name="job", schedule="1h", prompt="a")
    approval_id = await _approved_specialist_card(
        env, "schedule_delete", {"name": "job"}
    )
    original_audit = env.service._audit

    async def broken_audit(*args, **kwargs):
        raise RuntimeError("audit write failed")

    monkeypatch.setattr(env.service, "_audit", broken_audit)
    with pytest.raises(RuntimeError, match="audit write failed"):
        await env.service.remove(
            name="job", actor="automation-operator", authorized_by=approval_id,
        )
    assert await db.fetchone("SELECT id FROM schedules WHERE name='job'") is not None
    markers = await _live_authorization_markers(approval_id)
    assert markers["used_at"] is None and markers["claim"] is None

    monkeypatch.setattr(env.service, "_audit", original_audit)
    retry = await env.service.remove(
        name="job", actor="automation-operator", authorized_by=approval_id,
    )
    assert retry["status"] == "removed"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='job'") is None
    assert (await _live_authorization_markers(approval_id))["used_at"]


async def test_claim_snapshot_guard_refuses_card_widened_before_update(env, monkeypatch):
    """Q2: the conditional claim UPDATE refuses a card widened after its SELECT."""
    await _create_operator(env, name="job", schedule="1h", prompt="a")
    card = await _approved_edit_card(env)
    approval_id = card["approval_id"]
    original_execute = db.execute
    widened = False

    async def widen_before_claim(sql, params=()):
        nonlocal widened
        if (
            not widened
            and "UPDATE approvals" in sql
            and "live_authorization_claim" in sql
        ):
            widened = True
            await original_execute(
                "UPDATE approvals SET resource=? WHERE id=?",
                ("schedule_delete:someone-else", approval_id),
            )
        return await original_execute(sql, params)

    monkeypatch.setattr(db, "execute", widen_before_claim)
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="job", actor="chat:web", turn=_chat("tui"),
            authorized_by=approval_id,
        )
    assert caught.value.code == "authorization_consumed"
    assert widened is True
    assert await db.fetchall("SELECT id FROM tasks") == []
    markers = await _live_authorization_markers(approval_id)
    assert markers["used_at"] is None and markers["claim"] is None


async def test_run_now_creates_a_manual_task_and_cancel_stops_it(env):
    created = await _create_operator(env, name="job", schedule="1h", prompt="run")
    assert created["status"] == "created"
    before = await db.fetchone("SELECT next_run_at, claimed_count FROM schedules WHERE name='job'")
    run = await env.service.run_now(name="job", actor="operator:cli", turn=_operator())
    assert run["status"] == "running" and run["manual_run"] is True
    task = await env.engine.get(run["task_id"])
    assert task is not None and task.metadata.get("manual_run") is True
    after = await db.fetchone("SELECT next_run_at, claimed_count FROM schedules WHERE name='job'")
    assert dict(after) == dict(before)

    cancelled = await env.service.cancel(name="job", actor="operator:cli", turn=_operator())
    assert cancelled["status"] == "cancelled"
    assert (await env.engine.get(run["task_id"])).status.value == "cancelled"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='job'") is not None


@pytest.mark.parametrize("source", ["scheduled", "unattended", "agent"])
async def test_create_grant_cannot_write_silently_in_unattended_turns(env, source):
    """A session grant must not be inherited by a scheduled/agent run."""
    first = await env.service.create(
        {"name": "one", "schedule": "1h", "prompt": "a"},
        actor="chat:web", turn=_chat("tui"),
    )
    resolved = await _resolve(
        env, first["approval_id"], choice="session",
        digest=first["card"]["proposal_digest"],
    )
    assert resolved.grant_id

    unattended = TurnContext(
        principal="chat:web", channel="scheduled", conversation_id="conv-a",
        surface="scheduled", origin="scheduled", source=source,
    )
    second = await env.service.create(
        {"name": "two", "schedule": "1h", "prompt": "b"},
        actor="chat:web", turn=unattended,
    )
    assert second["status"] == "approval_required"
    assert second["card"] is None
    assert await db.fetchone("SELECT id FROM schedules WHERE name='two'") is None
    assert len(await db.fetchall("SELECT id FROM approvals")) == 1


async def test_stay_paused_edit_commits_without_changing_enabled_state(env):
    """An unrelated edit to a paused row does not re-enable it."""
    await _create_operator(env, name="job", schedule="1h", prompt="a")
    await env.service.update(
        name="job", arguments={"paused": True, "paused_reason": "hold"},
        actor="operator:cli", turn=_operator(),
    )
    pending = await env.service.update(
        name="job", arguments={"prompt": "b"}, actor="chat:web", turn=_chat("tui")
    )
    resolved = await _resolve(
        env, pending["card"]["approval_id"], choice="once",
        digest=pending["card"]["proposal_digest"],
    )
    assert resolved.status == "approved"
    row = await db.fetchone(
        "SELECT payload, enabled, status FROM schedules WHERE name='job'"
    )
    assert json.loads(row["payload"])["objective"] == "b"
    assert row["enabled"] == 0 and row["status"] == "paused"


async def test_edit_card_for_retired_row_declines_as_paused_or_disabled(env):
    """The decline names the disabled row; it must not blame a passed run time."""
    await _create_operator(env, name="job", schedule="1h", prompt="a")
    pending = await env.service.update(
        name="job", arguments={"prompt": "b"}, actor="chat:web", turn=_chat("tui")
    )
    # The row is retired while the operator's card is still waiting.
    await db.execute(
        "UPDATE schedules SET enabled=0, status='retired' WHERE name='job'"
    )
    resolved = await _resolve(
        env, pending["card"]["approval_id"], choice="once",
        digest=pending["card"]["proposal_digest"],
    )
    assert resolved.status == "expired" and resolved.allowed is False
    assert resolved.decline_message == DECLINE_MESSAGES["disabled_edit"]
    assert "paused or disabled" in resolved.decline_message
    assert "run time" not in resolved.decline_message
    stored = await db.fetchone(
        "SELECT status, json_extract(metadata,'$.decline_message') AS decline "
        "FROM approvals WHERE id=?", (pending["card"]["approval_id"],),
    )
    assert stored["status"] == "expired"
    assert stored["decline"] == DECLINE_MESSAGES["disabled_edit"]


async def test_model_turns_cannot_fire_or_kill_runs(env):
    await _create_operator(env, name="job", schedule="1h", prompt="run")
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="job", prompt="unasked", actor="chat:web", turn=_chat("tui")
        )
    assert caught.value.code == "operator_required"
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.cancel(name="job", actor="chat:web", turn=_chat("tui"))
    assert caught.value.code == "operator_required"
    assert await db.fetchall("SELECT id FROM tasks") == []


async def _approved_edit_card(env, name: str = "job") -> dict:
    """Create and approve one fresh edit card for the named job."""
    seq = int(getattr(env, "card_seq", 0)) + 1
    env.card_seq = seq
    pending = await env.service.update(
        name=name, arguments={"prompt": f"revised-{name}-{seq}"},
        actor="chat:web", turn=_chat("tui"),
    )
    card = pending["card"]
    resolved = await _resolve(
        env, card["approval_id"], choice="once", digest=card["proposal_digest"]
    )
    assert resolved.status == "approved"
    return card


async def _approved_specialist_card(env, tool_name: str, arguments: dict) -> str:
    """Create and resolve one exact-action automation:mutate approval."""
    from leo.agents.specs.automation_operator import approval_target
    from leo.contracts import RiskLevel

    target = approval_target(tool_name, arguments)
    decision = await env.permissions.authorize(
        "automation-operator", target["action"], target["resource"],
        RiskLevel.HIGH, arguments_digest=target["arguments_digest"],
        tool=tool_name,
    )
    assert decision.requires_approval and decision.approval_id
    resolved = await env.permissions.resolve_approval(
        decision.approval_id, True, "operator:local"
    )
    assert resolved.allowed
    return str(decision.approval_id)


async def _live_authorization_markers(approval_id: str) -> dict:
    row = await db.fetchone(
        "SELECT json_extract(metadata,'$.live_authorization_used_at') AS used_at,"
        "json_extract(metadata,'$.live_authorization_claim') AS claim "
        "FROM approvals WHERE id=?", (approval_id,),
    )
    assert row is not None
    return {"used_at": row["used_at"], "claim": json.loads(row["claim"] or "null")}


async def test_approved_edit_card_authorizes_one_manual_run_and_one_cancel(env):
    await _create_operator(env, name="job", schedule="1h", prompt="run")
    run_card = await _approved_edit_card(env)

    run = await env.service.run_now(
        name="job", actor="chat:web", turn=_chat("tui"),
        authorized_by=run_card["approval_id"],
    )
    assert run["status"] == "running"
    # One card is one capability: cancel needs its own approved edit.
    cancel_card = await _approved_edit_card(env)
    cancelled = await env.service.cancel(
        name="job", actor="chat:web", turn=_chat("tui"),
        authorized_by=cancel_card["approval_id"],
    )
    assert cancelled["status"] == "cancelled"

    other = await env.service.create(
        {"name": "other", "schedule": "1h", "prompt": "x"},
        actor="chat:web", turn=_chat("tui"),
    )
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="job", actor="chat:web", turn=_chat("tui"),
            authorized_by=other["approval_id"],
        )
    assert caught.value.code == "invalid_authorization"


async def test_approved_edit_card_is_single_use_for_live_actions(env):
    await _create_operator(env, name="job", schedule="1h", prompt="run")
    card = await _approved_edit_card(env)

    run = await env.service.run_now(
        name="job", actor="chat:web", turn=_chat("tui"),
        authorized_by=card["approval_id"],
    )
    assert run["status"] == "running"
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="job", actor="chat:web", turn=_chat("tui"),
            authorized_by=card["approval_id"],
        )
    assert caught.value.code == "authorization_consumed"
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.cancel(
            name="job", actor="chat:web", turn=_chat("tui"),
            authorized_by=card["approval_id"],
        )
    assert caught.value.code == "authorization_consumed"
    row = await db.fetchone(
        "SELECT json_extract(metadata,'$.live_authorization_used_at') AS used_at "
        "FROM approvals WHERE id=?", (card["approval_id"],),
    )
    assert row["used_at"]
    manual_tasks = await db.fetchone(
        "SELECT COUNT(*) AS n FROM tasks WHERE metadata LIKE '%manual_run%'"
    )
    assert manual_tasks["n"] == 1


async def test_approved_edit_card_live_authorization_has_one_concurrent_winner(env):
    await _create_operator(env, name="job", schedule="1h", prompt="run")
    card = await _approved_edit_card(env)
    results = await asyncio.gather(
        env.service.run_now(
            name="job", actor="chat:web", turn=_chat("tui"),
            authorized_by=card["approval_id"], request_key="race-a",
        ),
        env.service.run_now(
            name="job", actor="chat:web", turn=_chat("tui"),
            authorized_by=card["approval_id"], request_key="race-b",
        ),
        return_exceptions=True,
    )
    successes = [item for item in results if isinstance(item, dict)]
    failures = [item for item in results if isinstance(item, ScheduleServiceError)]
    assert len(successes) == 1 and len(failures) == 1
    assert failures[0].code == "authorization_consumed"


async def test_specialist_run_card_is_single_use_and_time_bounded(env):
    """N2: one automation:mutate approval may authorize exactly one live run."""
    await _create_operator(env, name="job", schedule="1h", prompt="run")
    approval_id = await _approved_specialist_card(
        env, "schedule_run_now", {"name": "job"}
    )
    first = await env.service.run_now(
        name="job", actor="automation-operator", authorized_by=approval_id,
    )
    assert first["status"] == "running"
    markers = await _live_authorization_markers(approval_id)
    assert markers["used_at"] and markers["claim"] is None

    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="job", actor="automation-operator", authorized_by=approval_id,
        )
    assert caught.value.code == "authorization_consumed"
    assert dict(await db.fetchone(
        "SELECT COUNT(*) AS n FROM tasks WHERE metadata LIKE '%manual_run%'"
    ))["n"] == 1

    # A fresh specialist approval outside its durable window is refused before
    # any task is created.
    stale = await _approved_specialist_card(
        env, "schedule_run_now", {"name": "job"}
    )
    env.clock.advance(seconds=16 * 60)
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="job", actor="automation-operator", authorized_by=stale,
        )
    assert caught.value.code == "invalid_authorization"
    assert dict(await db.fetchone(
        "SELECT COUNT(*) AS n FROM tasks WHERE metadata LIKE '%manual_run%'"
    ))["n"] == 1


async def test_stale_live_authorization_lease_can_be_reclaimed(env):
    """N3(a): the claim itself is time bounded, not a permanent lock."""
    await _create_operator(env, name="job", schedule="1h", prompt="run")
    card = await _approved_edit_card(env)
    approval_id = card["approval_id"]
    stale_at = (NOW - timedelta(seconds=301)).isoformat()
    await db.execute(
        "UPDATE approvals SET metadata=json_set(COALESCE(metadata,'{}'),"
        "'$.live_authorization_claim', json_object('token',?,'at',?)) WHERE id=?",
        ("old-lease", stale_at, approval_id),
    )
    run = await env.service.run_now(
        name="job", actor="chat:web", turn=_chat("tui"),
        authorized_by=approval_id,
    )
    assert run["status"] == "running"
    markers = await _live_authorization_markers(approval_id)
    assert markers["used_at"] and markers["claim"] is None


async def test_live_action_beyond_lease_ttl_cannot_be_reclaimed(env, monkeypatch):
    """M4: a long live action keeps its lease; only a crashed worker's is stale."""
    await _create_operator(env, name="slow-job", schedule="1h", prompt="run")
    card = await _approved_edit_card(env, name="slow-job")
    approval_id = card["approval_id"]
    started, release = asyncio.Event(), asyncio.Event()
    original_create = env.engine.create

    async def slow_create(*args, **kwargs):
        started.set()
        await release.wait()
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(env.engine, "create", slow_create)
    first = asyncio.create_task(env.service.run_now(
        name="slow-job", actor="chat:web", turn=_chat("tui"),
        authorized_by=approval_id, request_key="slow-a",
    ))
    await started.wait()
    env.clock.advance(seconds=LIVE_AUTH_CLAIM_TTL_SECONDS + 1)

    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="slow-job", actor="chat:web", turn=_chat("tui"),
            authorized_by=approval_id, request_key="slow-b",
        )
    assert caught.value.code == "authorization_consumed"

    release.set()
    result = await first
    assert result["status"] == "running"
    markers = await _live_authorization_markers(approval_id)
    assert markers["used_at"] and markers["claim"] is None


async def test_lease_staleness_uses_numeric_expiry_epoch(env):
    """M5: SQL never compares ISO timestamp strings to decide staleness."""
    await _create_operator(env, name="epoch-job", schedule="1h", prompt="run")
    card = await _approved_edit_card(env, name="epoch-job")
    approval_id = card["approval_id"]
    await db.execute(
        "UPDATE approvals SET metadata=json_set(COALESCE(metadata,'{}'),"
        "'$.live_authorization_claim', json_object('token',?,'at',?,'expires_at',?)) "
        "WHERE id=?",
        (
            "future-iso-past-expiry", (NOW + timedelta(minutes=5)).isoformat(),
            (NOW - timedelta(seconds=1)).timestamp(), approval_id,
        ),
    )
    claim_type = await db.fetchone(
        "SELECT typeof(json_extract(metadata,'$.live_authorization_claim.expires_at')) AS kind "
        "FROM approvals WHERE id=?", (approval_id,),
    )
    assert claim_type["kind"] == "real"
    # The numeric expiry wins even though the ISO 'at' still looks fresh.
    assert (await env.service.run_now(
        name="epoch-job", actor="chat:web", turn=_chat("tui"),
        authorized_by=approval_id,
    ))["status"] == "running"

    second = await _approved_edit_card(env, name="epoch-job")
    second_id = second["approval_id"]
    await db.execute(
        "UPDATE approvals SET metadata=json_set(COALESCE(metadata,'{}'),"
        "'$.live_authorization_claim', json_object('token',?,'at',?,'expires_at',?)) "
        "WHERE id=?",
        (
            "past-iso-future-expiry", (NOW - timedelta(minutes=5)).isoformat(),
            (NOW + timedelta(minutes=5)).timestamp(), second_id,
        ),
    )
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="epoch-job", actor="chat:web", turn=_chat("tui"),
            authorized_by=second_id,
        )
    assert caught.value.code == "authorization_consumed"


async def test_claim_and_settle_use_the_injected_clock(env):
    """M6: no wall-clock reads in the lease path."""
    await _create_operator(env, name="clock-job", schedule="1h", prompt="run")
    card = await _approved_edit_card(env, name="clock-job")
    approval_id = card["approval_id"]
    assert (await env.service.run_now(
        name="clock-job", actor="chat:web", turn=_chat("tui"),
        authorized_by=approval_id,
    ))["status"] == "running"
    row = await db.fetchone(
        "SELECT json_extract(metadata,'$.live_authorization_used_at') AS used_at "
        "FROM approvals WHERE id=?", (approval_id,),
    )
    assert row["used_at"] == NOW.isoformat()


async def test_failed_run_and_noop_cancel_do_not_burn_single_use_cards(env):
    """N3(b): consume only after a durable effect, never on a no-op."""
    await _create_operator(env, name="job", schedule="1h", prompt="run")
    run_card = await _approved_edit_card(env)
    approval_id = run_card["approval_id"]

    await db.execute(
        "UPDATE schedules SET enabled=0, status='retired' WHERE name='job'"
    )
    with pytest.raises(ScheduleServiceError) as caught:
        await env.service.run_now(
            name="job", actor="chat:web", turn=_chat("tui"),
            authorized_by=approval_id,
        )
    assert caught.value.code == "schedule_disabled"
    markers = await _live_authorization_markers(approval_id)
    assert markers["used_at"] is None and markers["claim"] is None

    await db.execute(
        "UPDATE schedules SET enabled=1, status='active' WHERE name='job'"
    )
    run = await env.service.run_now(
        name="job", actor="chat:web", turn=_chat("tui"),
        authorized_by=approval_id,
    )
    assert run["status"] == "running"
    assert (await _live_authorization_markers(approval_id))["used_at"]

    # A cancel with no live task reports the no-op and leaves its card intact.
    await _create_operator(env, name="job2", schedule="1h", prompt="run")
    cancel_card = await _approved_edit_card(env, name="job2")
    cancel_id = cancel_card["approval_id"]
    noop = await env.service.cancel(
        name="job2", actor="chat:web", turn=_chat("tui"),
        authorized_by=cancel_id,
    )
    assert noop["status"] == "no_live_run"
    markers = await _live_authorization_markers(cancel_id)
    assert markers["used_at"] is None and markers["claim"] is None

    await env.service.run_now(name="job2", actor="operator:cli", turn=_operator())
    cancelled = await env.service.cancel(
        name="job2", actor="chat:web", turn=_chat("tui"),
        authorized_by=cancel_id,
    )
    assert cancelled["status"] == "cancelled"
    assert (await _live_authorization_markers(cancel_id))["used_at"]


async def test_disabled_row_guard_names_retired_state_not_paused(env):
    """N7: a retired/disabled row is not described as a paused row."""
    await _create_operator(env, name="job", schedule="1h", prompt="a")
    pending = await env.service.update(
        name="job", arguments={"prompt": "b"}, actor="chat:web", turn=_chat("tui")
    )
    await db.execute(
        "UPDATE schedules SET enabled=0, status='retired' WHERE name='job'"
    )
    resolved = await _resolve(
        env, pending["card"]["approval_id"], choice="once",
        digest=pending["card"]["proposal_digest"],
    )
    assert resolved.status == "expired" and resolved.allowed is False
    assert "disabled or retired" in resolved.reason
    assert "paused or disabled" not in resolved.reason


async def test_recurring_card_can_be_answered_after_its_first_tick(env):
    pending = await env.service.create(
        {"name": "water", "schedule": "every day at 7:01am", "prompt": "Drink."},
        actor="chat:web", turn=_chat("tui"),
    )
    card = pending["card"]
    # The shown first instant has passed, but the recurring job still has
    # future occurrences and the approval timeout has not elapsed.
    env.clock.advance(seconds=120)
    resolved = await _resolve(
        env, card["approval_id"], choice="once", digest=card["proposal_digest"]
    )
    assert resolved.status == "approved"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='water'") is not None


async def test_one_shot_card_still_stales_after_its_instant(env):
    pending = await env.service.create(
        {"name": "dentist", "schedule": "in 3m", "prompt": "Call."},
        actor="chat:web", turn=_chat("tui"),
    )
    card = pending["card"]
    env.clock.advance(seconds=240)
    resolved = await _resolve(
        env, card["approval_id"], choice="once", digest=card["proposal_digest"]
    )
    assert resolved.status == "expired"
    assert resolved.decline_message == DECLINE_MESSAGES["stale_create"]
    assert await db.fetchone("SELECT id FROM schedules WHERE name='dentist'") is None


async def test_replayed_denial_tells_the_model_the_human_reason(env):
    arguments = {"name": "water", "schedule": "in 2h", "prompt": "Drink water."}
    pending = await env.service.create(
        arguments, actor="chat:web", turn=_chat("tui"),
        request_key="req-denied-with-reason",
    )
    card = pending["card"]
    await env.permissions.resolve_schedule_card(
        card["approval_id"], choice="deny", resolver="operator:local",
        confirm_proposal_digest=card["proposal_digest"],
        commit=env.service.commit, deny_reason="owner said no",
    )
    replay = await env.service.create(
        arguments, actor="chat:web", turn=_chat("tui"),
        request_key="req-denied-with-reason",
    )
    assert replay["status"] == "denied"
    assert replay["deny_reason"] == "owner said no"


async def test_create_card_discloses_catch_up_pinned_target_and_unlimited_runs(env):
    env.service.pinned_provider = "provider-x"
    env.service.pinned_model = "model-y"
    out = await env.service.create(
        {
            "name": "brief", "schedule": "every day at 8am", "prompt": "Brief.",
            "catch_up": False, "pinned": True,
        },
        actor="chat:web", turn=_chat("tui"),
    )
    body = out["card"]["body"]
    assert "runs forever" in body
    assert "skip missed runs (at most one run after downtime)" in body
    assert "pinned to provider-x/model-y" in body


async def test_service_rejects_a_skill_the_live_registry_does_not_know(env):
    async def _skills():
        return [SimpleNamespace(name="maps", documentation="Map documentation.")]

    env.service.skills = SimpleNamespace(list=_skills)
    known = await env.service.create(
        {"name": "known", "schedule": "in 2h", "prompt": "Brief.", "skills": ["maps"]},
        actor="chat:web", turn=_chat("tui"),
    )
    assert known["status"] == "approval_required"

    with pytest.raises(ScheduleCompileError) as caught:
        await env.service.create(
            {"name": "ghost", "schedule": "in 2h", "prompt": "Brief.", "skills": ["ghost"]},
            actor="chat:web", turn=_chat("tui"),
        )
    assert caught.value.code == "unknown_skill"
    assert await db.fetchone("SELECT id FROM schedules WHERE name='ghost'") is None
