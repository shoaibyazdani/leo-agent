"""Frozen ApprovalCard v1 protocol and atomic schedule-card resolution.

The card is a projection of one durable ``approvals`` row. These tests pin the
contract a human saw (literal body, choices, deadline), the shown/digest/expiry
binding, replay refusal, restart persistence, and the all-or-nothing decision +
grant + commit + audit transaction.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from leo import db
from leo.config import Settings
from leo.events import EventBus
from leo.security.approval_buttons import (
    build_callback_data,
    parse_callback_data,
    resolve_locator,
)
from leo.security.approval_cards import (
    APPROVAL_CARD_PROTOCOL_VERSION,
    DECLINE_MESSAGES,
    KIND_SCHEDULE_CREATE,
    KIND_SCHEDULE_EDIT,
    ApprovalCard,
    ApprovalService,
    ScheduleBodySpec,
    card_from_row,
    format_first_run,
    render_edit_body,
    render_schedule_body,
    schedule_terminal_text,
)
from leo.security.permissions import (
    SCHEDULE_CREATE_ACTION,
    PermissionManagerImpl,
)
from leo.schedules.clock import FrozenClock

NOW = datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc)


def _settings() -> Settings:
    settings = Settings()
    settings.security.approval_timeout_seconds = 900
    return settings


def _permissions(
    clock: FrozenClock | None = None, events: EventBus | None = None
) -> PermissionManagerImpl:
    return PermissionManagerImpl(
        _settings(), events=events, clock=clock or FrozenClock(NOW)
    )


async def _create(
    permissions: PermissionManagerImpl,
    *,
    kind: str = KIND_SCHEDULE_CREATE,
    digest: str = "digest-1",
    requested_by: str = "chat:web",
    conversation_id: str = "conv-a",
    **overrides,
):
    params = {
        "kind": kind,
        "body": 'Schedule "Water" (every day at 8am): Remind the user to drink water.',
        "proposal_digest": digest,
        "proposal_id": "prop-1",
        "schedule_id": "sch-1",
        "config_revision": 1,
        "requested_by": requested_by,
        "conversation_id": conversation_id,
        "origin": "telegram",
    }
    params.update(overrides)
    return await permissions.create_schedule_card(**params)


async def _approve(
    permissions: PermissionManagerImpl,
    card: ApprovalCard,
    *,
    choice: str = "once",
    resolver: str = "operator:local",
    digest: str = "digest-1",
    commit=None,
):
    return await permissions.resolve_schedule_card(
        card.approval_id,
        choice=choice,
        resolver=resolver,
        confirm_proposal_digest=digest,
        commit=commit,
    )


async def test_card_round_trip_preserves_the_literal_body_and_choices() -> None:
    permissions = _permissions()
    body = (
        'Schedule "Water" (every day at 8am, first run Wed Nov 25 08:00 UTC): '
        'Remind the user to drink water.; results go to local history only'
    )
    card = await _create(permissions, body=body)

    assert card.protocol_version == APPROVAL_CARD_PROTOCOL_VERSION
    assert card.kind == KIND_SCHEDULE_CREATE
    assert card.body == body, "the card stores literal text, not a summary"
    assert card.choices == ["once", "session", "always", "deny"]
    assert card.choice_labels == [
        "Allow", "Allow for this chat", "Always allow new jobs", "Deny"
    ]
    assert card.proposal_digest == "digest-1"
    assert card.schedule_id == "sch-1"

    view = card.view()
    assert view.body == body
    assert [choice.value for choice in view.choices] == card.choices
    assert [choice.label for choice in view.choices] == card.choice_labels

    # DTO round-trip is exact and rejects unknown fields.
    assert ApprovalCard.model_validate(card.model_dump(mode="json")) == card
    with pytest.raises(ValidationError):
        ApprovalCard.model_validate({**card.model_dump(mode="json"), "injected": True})


async def test_presentation_sets_shown_at_once_and_incapable_surfaces_are_refused() -> None:
    clock = FrozenClock(NOW)
    permissions = _permissions(clock)
    card = await _create(permissions)

    # A blocking CLI cannot present the full card and must not mark it shown.
    with pytest.raises(PermissionError):
        await permissions.present_schedule_card(card.approval_id, surface="cli")
    row = await db.fetchone(
        "SELECT shown_at, presented_surface FROM approvals WHERE id=?",
        (card.approval_id,),
    )
    assert row["shown_at"] is None

    first = await permissions.present_schedule_card(card.approval_id, surface="tui")
    assert first.shown_at == NOW.isoformat()
    assert first.presented_at == NOW.isoformat()
    assert first.presented_surface == "tui"

    clock.advance(seconds=60)
    again = await permissions.present_schedule_card(
        card.approval_id, surface="web"
    )
    assert again.shown_at == NOW.isoformat(), "the first presentation is authoritative"
    assert again.presented_surface == "web"


async def test_resolution_is_refused_before_presentation() -> None:
    permissions = _permissions()
    card = await _create(permissions)

    with pytest.raises(PermissionError, match="not been presented"):
        await _approve(permissions, card)
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "pending"


async def test_exact_deadline_equality_expires_without_a_decision() -> None:
    clock = FrozenClock(NOW)
    permissions = _permissions(clock)
    card = await _create(permissions, expires_at=NOW.isoformat())
    await permissions.present_schedule_card(card.approval_id, surface="web")

    result = await _approve(permissions, card)
    assert result.status == "expired"
    assert result.allowed is False
    assert result.outcome == "timed out — the job was not created"
    assert result.decline_message == DECLINE_MESSAGES["timeout"]
    row = await db.fetchone(
        "SELECT status, answered_at FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "expired" and row["answered_at"]


async def test_a_passed_first_run_makes_the_card_stale() -> None:
    clock = FrozenClock(NOW)
    permissions = _permissions(clock)
    card = await _create(permissions, shown_first_run=NOW.isoformat())
    await permissions.present_schedule_card(card.approval_id, surface="tui")

    result = await _approve(permissions, card)
    assert result.status == "expired"
    assert result.outcome == "stale — the job was not created"
    assert result.decline_message == DECLINE_MESSAGES["stale_create"]


async def test_wrong_digest_wrong_choice_and_replay_never_change_state() -> None:
    permissions = _permissions()
    card = await _create(permissions)
    await permissions.present_schedule_card(card.approval_id, surface="web")

    with pytest.raises(ValueError, match="digest mismatch"):
        await _approve(permissions, card, digest="stale-digest")
    with pytest.raises(ValueError, match="invalid choice"):
        await _approve(permissions, card, choice="banana")
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "pending"

    approved = await _approve(permissions, card, choice="once")
    assert approved.status == "approved"
    with pytest.raises(ValueError, match="already approved"):
        await _approve(permissions, card, choice="once")


async def test_edit_cards_offer_only_allow_or_deny() -> None:
    permissions = _permissions()
    card = await _create(
        permissions, kind=KIND_SCHEDULE_EDIT,
        body='Change "Water": prompt: water -> tea.', digest="edit-digest",
    )
    assert card.choices == ["once", "deny"]
    assert card.choice_labels == ["Allow", "Deny"]
    await permissions.present_schedule_card(card.approval_id, surface="web")

    with pytest.raises(ValueError, match="invalid choice"):
        await _approve(
            permissions, card, choice="session", digest="edit-digest"
        )
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "pending"


async def test_card_survives_restart_with_shown_state_and_digest() -> None:
    permissions = _permissions()
    card = await _create(permissions)
    await permissions.present_schedule_card(card.approval_id, surface="tui")

    # A fresh manager (new process) reads the same durable row.
    restarted = _permissions()
    restored = await restarted.get_schedule_card(card.approval_id)
    assert restored is not None
    assert restored.body == card.body
    assert restored.shown_at == NOW.isoformat()
    assert restored.proposal_digest == "digest-1"

    result = await _approve(restarted, restored)
    assert result.status == "approved"


async def test_decision_grant_and_audit_are_one_atomic_unit() -> None:
    permissions = _permissions()
    card = await _create(permissions)
    await permissions.present_schedule_card(card.approval_id, surface="telegram")

    result = await _approve(permissions, card, choice="session")
    assert result.status == "approved"
    assert result.grant_id

    grant = await db.fetchone(
        "SELECT action, principal, subject_type, subject_id, revoked_at "
        "FROM approval_grants WHERE id=?",
        (result.grant_id,),
    )
    assert dict(grant) == {
        "action": SCHEDULE_CREATE_ACTION,
        "principal": "chat:web",
        "subject_type": "conversation",
        "subject_id": "conv-a",
        "revoked_at": None,
    }
    audits = await db.fetchall(
        "SELECT action, decision, metadata FROM audit_log "
        "WHERE action IN ('schedule.consent.request','schedule:create') ORDER BY id"
    )
    assert [row["action"] for row in audits] == [
        "schedule.consent.request", "schedule:create",
    ]
    assert json.loads(audits[-1]["metadata"])["approval_id"] == card.approval_id


async def test_a_failed_commit_rolls_back_decision_grant_and_audit(monkeypatch) -> None:
    bus = EventBus(persist=False)
    published: list[str] = []

    async def capture(event):
        published.append(event.topic)

    bus.subscribe("approval.resolved", capture)
    permissions = _permissions(events=bus)
    card = await _create(permissions)
    await permissions.present_schedule_card(card.approval_id, surface="web")

    async def explode(**kwargs):
        raise RuntimeError("schedule write failed")

    with pytest.raises(RuntimeError, match="schedule write failed"):
        await _approve(permissions, card, choice="session", commit=explode)

    row = await db.fetchone(
        "SELECT status, answered_at FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "pending" and row["answered_at"] is None
    assert await db.fetchall(
        "SELECT id FROM approval_grants WHERE action=?", (SCHEDULE_CREATE_ACTION,)
    ) == []
    audits = await db.fetchall(
        "SELECT decision, reason FROM audit_log WHERE reason LIKE 'schedule consent approved%'"
    )
    assert audits == [], "the approval audit rolls back with the commit"
    assert "approval.resolved" not in published, "no event may escape a rollback"


async def test_approval_service_composes_the_one_commit_callback() -> None:
    permissions = _permissions()
    seen: dict = {}

    async def commit(**kwargs):
        seen.update(kwargs)
        return {"schedule_id": "sch-service"}

    service = ApprovalService(permissions, committer=commit)
    card = await service.create(
        kind=KIND_SCHEDULE_CREATE,
        body='Schedule "Water": drink water.',
        proposal_digest="service-digest",
        requested_by="chat:web",
        conversation_id="conv-a",
    )
    await service.present(card.approval_id, surface="web")
    result = await service.resolve(
        card.approval_id,
        choice="once",
        resolver="operator:local",
        confirm_proposal_digest="service-digest",
    )
    assert result.status == "approved"
    assert result.schedule_id == "sch-service"
    assert seen["choice"] == "once" and seen["resolver"] == "operator:local"
    assert seen["card"].approval_id == card.approval_id


async def test_wait_returns_the_terminal_card_after_a_decision() -> None:
    import asyncio

    permissions = _permissions()
    card = await _create(permissions)
    await permissions.present_schedule_card(card.approval_id, surface="web")

    async def decide_later() -> None:
        await asyncio.sleep(0.05)
        await _approve(permissions, card)

    task = asyncio.create_task(decide_later())
    try:
        waited = await permissions.wait_for_schedule_card(
            card.approval_id, timeout=2.0
        )
    finally:
        await task
    assert waited.status == "approved"
    assert waited.answered_at


async def test_a_commit_result_supplies_the_committed_schedule_id() -> None:
    permissions = _permissions()
    card = await _create(permissions, schedule_id=None)
    await permissions.present_schedule_card(card.approval_id, surface="web")

    async def commit(**kwargs):
        assert kwargs["choice"] == "once"
        return {"schedule_id": "sch-real", "name": "Water"}

    result = await _approve(permissions, card, commit=commit)
    assert result.commit_result["schedule_id"] == "sch-real"
    assert result.schedule_id == "sch-real"
    restored = await permissions.get_schedule_card(card.approval_id)
    assert restored.schedule_id == "sch-real"


async def test_deny_records_a_bounded_reason_and_no_grant() -> None:
    permissions = _permissions()
    card = await _create(permissions)
    await permissions.present_schedule_card(card.approval_id, surface="web")

    result = await permissions.resolve_schedule_card(
        card.approval_id,
        choice="deny",
        resolver="operator:local",
        confirm_proposal_digest="digest-1",
        deny_reason="no " + "x" * 2000,
    )
    assert result.status == "denied" and result.grant_id is None
    row = await db.fetchone(
        "SELECT deny_reason FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["deny_reason"].startswith("no ")
    assert len(row["deny_reason"]) == 500
    assert await db.fetchall("SELECT id FROM approval_grants") == []


async def test_a_chat_actor_cannot_resolve_its_own_card() -> None:
    permissions = _permissions()
    card = await _create(permissions, requested_by="chat:web-a")
    await permissions.present_schedule_card(card.approval_id, surface="web")

    with pytest.raises(PermissionError, match="chat actors"):
        await _approve(permissions, card, resolver="chat:web-a")
    with pytest.raises(PermissionError, match="requester cannot resolve"):
        await _approve(permissions, card, resolver="operator:chat:web-a")
    row = await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["status"] == "pending"


async def test_deny_only_card_refuses_every_allow_choice() -> None:
    permissions = _permissions()
    card = await _create(permissions, deny_only=True)
    assert card.choices == ["deny"]
    assert card.choice_labels == ["Deny"]
    await permissions.present_schedule_card(card.approval_id, surface="web")

    with pytest.raises(ValueError, match="deny-only"):
        await _approve(permissions, card, choice="once")
    result = await _approve(permissions, card, choice="deny")
    assert result.status == "denied"


def test_schedule_body_renderer_is_literal_complete_and_honest() -> None:
    body = render_schedule_body(
        ScheduleBodySpec(
            name="Brief",
            display="every day at 8am",
            first_run_at="2026-11-25T08:00:00+00:00",
            repeat_limit=None,
            prompt="Summarize updates.",
            skills=("maps",),
            script="collect.py",
            monitor="https://example.com/price",
            workdir="/workspace/brief",
            context_from=("source-job",),
            enabled_toolsets=(),
            attach_to_session=True,
            delivery="nowhere (saved to the job's history only)",
            failure_delivery="no destination (saved to the job's history only)",
        )
    )
    for fragment in (
        "every day at 8am — first run Wed Nov 25 08:00 UTC, runs forever",
        '"Summarize updates."',
        "(skills: maps)",
        "(after running script collect.py)",
        "Checks https://example.com/price first each run.",
        "Runs in /workspace/brief.",
        "Reads output of source-job.",
        "Toolsets: none.",
        "Copied into the chat's history.",
        "Results go to nowhere (saved to the job's history only).",
        "Failure notices go to no destination",
    ):
        assert fragment in body
    # Defaults stay out of the text; the job name is an internal handle.
    for absent in (
        'Schedule "Brief"', "missed runs are caught up",
        "uses the default model",
    ):
        assert absent not in body, absent

    pinned = render_schedule_body(
        ScheduleBodySpec(
            name="Brief", display="every day at 8am", repeat_limit=5,
            prompt="Summarize.", catch_up=False, pinned=True,
            pinned_provider="provider-x", pinned_model="model-y",
        )
    )
    assert "5 runs" in pinned
    assert "runs forever" not in pinned
    assert "Skips missed runs (at most one run after downtime)" in pinned
    assert "Always uses provider-x/model-y" in pinned

    paused = render_schedule_body(
        ScheduleBodySpec(
            name="Water", display="every day at 8am", paused=True,
            repeat_limit=3, prompt="Drink water.",
        )
    )
    assert "created paused: it will not run until resumed" in paused
    assert "first run" not in paused

    script_only = render_schedule_body(
        ScheduleBodySpec(
            name="Check", display="every 2h", script="check.py",
            no_agent=True, first_run_at="2026-11-25T09:00:00+00:00",
        )
    )
    assert 'Schedule "Check"' not in script_only
    assert "Runs script check.py." in script_only

    assert render_edit_body("Water", ["prompt: water -> tea"]) == (
        'Change "Water": prompt: water -> tea'
    )


def test_first_run_renders_in_the_schedule_timezone() -> None:
    """Regression: 28 Nov 2026 — a job set for 8am in Asia/Karachi showed its
    first run as 03:00 UTC on the approval card."""
    assert format_first_run("2026-11-29T03:00:00+00:00", "Asia/Karachi") == (
        "Sun Nov 29 08:00 PKT"
    )
    assert format_first_run("2026-11-29T03:00:00+00:00") == "Sun Nov 29 03:00 UTC"
    # An unknown timezone never breaks the card; it falls back to UTC.
    assert format_first_run("2026-11-29T03:00:00+00:00", "No/Such") == (
        "Sun Nov 29 03:00 UTC"
    )
    assert format_first_run(None, "Asia/Karachi") is None


def test_card_body_reads_like_a_message_to_a_human() -> None:
    """Regression: 28 Nov 2026 — the approval card was one dense parenthetical
    of internal fields ("default bounded catch-up after downtime",
    "not pinned to a provider/model", "telegram chat 525083981"). The same
    job now reads as three short lines."""
    body = render_schedule_body(
        ScheduleBodySpec(
            name="oneshot-0900-early-office-1129", display="once at 9am",
            first_run_at="2026-11-29T03:00:00+00:00",
            timezone="Asia/Karachi", repeat_limit=1,
            prompt="Aaj 9:00 AM tak pahunchna hai.",
            delivery="telegram chat",
        )
    )
    assert body == (
        "Once, Sun Nov 29 08:00 PKT\n"
        '"Aaj 9:00 AM tak pahunchna hai."\n'
        "Results go to telegram chat."
    )


@pytest.mark.parametrize(
    "sentinel",
    [
        "no reason supplied",
        "denied by operator",
        "  Denied By Operator  ",
    ],
)
def test_terminal_text_normalizes_placeholder_deny_reasons(sentinel: str) -> None:
    """M3: the Telegram/TUI rewrite never speaks a system fallback as a reason."""
    card = ApprovalCard(
        approval_id="apr-terminal",
        kind=KIND_SCHEDULE_CREATE,
        title="Schedule Water",
        body="Schedule \"Water\" (first run Wed Nov 25 08:00 UTC)",
        status="denied",
        outcome="denied",
        deny_reason=sentinel,
    )
    text = schedule_terminal_text(card, status="denied")
    assert "reason: no reason given" in text
    assert sentinel.strip().lower() not in text.lower()
    # Only the rendered copy changes: audit/model projections stay literal.
    assert card.deny_reason == sentinel
    assert card.view().deny_reason == sentinel
    assert "no reason given" not in str(card.model_dump())


def test_terminal_text_keeps_a_human_authored_deny_reason() -> None:
    card = ApprovalCard(
        approval_id="apr-terminal-human",
        kind=KIND_SCHEDULE_EDIT,
        title="Change Water",
        body='Change "Water": prompt: water -> tea',
        status="denied",
        outcome="denied",
        deny_reason="too expensive",
    )
    text = schedule_terminal_text(card, status="denied")
    assert "reason: too expensive" in text


def test_legacy_command_rows_project_into_the_same_frozen_shape() -> None:
    row = {
        "id": "apr_command",
        "created_at": NOW.isoformat(),
        "status": "pending",
        "actor": "operator",
        "action": "tool:shell",
        "resource": "/tmp/workspace",
        "risk": "high",
        "reason": "high-risk action requires explicit approval",
        "requested_by": "operator",
        "metadata": json.dumps({"command_preview": "ls -la"}),
        "kind": "command",
        "expires_at": (NOW + timedelta(minutes=15)).isoformat(),
    }
    card = card_from_row(row)
    assert card.kind == "command"
    assert card.title == "Approval needed"
    assert "command: ls -la" in card.body
    assert card.choices == ["once", "deny"]
    assert card.choice_labels == ["Approve", "Deny"]


def test_command_card_shows_exact_command_and_specific_risk() -> None:
    """R7: a human must see what will run and why, not the generic row copy."""
    row = {
        "id": "apr_why",
        "status": "pending",
        "action": "tool:shell",
        "resource": "/tmp/workspace",
        "risk": "high",
        "reason": "high-risk action requires explicit approval",
        "metadata": json.dumps({
            "command_preview": "rm -rf build/cache",
            "risk_reason": "rm is destructive",
        }),
        "kind": "command",
    }
    card = card_from_row(row)
    assert "command: rm -rf build/cache" in card.body
    assert "why: rm is destructive" in card.body
    assert "high-risk action requires explicit approval" not in card.body
    assert card.view().body == card.body
    assert card.event_payload()["body"] == card.body


def test_command_card_derives_a_specific_reason_for_legacy_rows() -> None:
    row = {
        "id": "apr_legacy",
        "status": "pending",
        "action": "tool:shell",
        "resource": "/tmp/workspace",
        "risk": "high",
        "reason": "high-risk action requires explicit approval",
        "metadata": json.dumps({"command_preview": "git push --force origin main"}),
        "kind": "command",
    }
    card = card_from_row(row)
    assert "command: git push --force origin main" in card.body
    assert "why: " in card.body
    why = next(line for line in card.body.splitlines() if line.startswith("why: "))
    assert "high-risk action requires explicit approval" not in why
    assert len(why) > len("why: ")


def test_schedule_callback_payloads_keep_kind_and_checksummed_locators() -> None:
    short = build_callback_data("schedule", "session", "apr_short")
    parsed = parse_callback_data(short)
    assert parsed is not None
    assert parsed["approval_kind"] == "schedule"
    assert parsed["decision"] == "session"
    assert parsed["is_ref"] == "0"

    long_id = "apr_" + "a" * 120
    encoded = build_callback_data("schedule_edit", "once", long_id)
    parsed_long = parse_callback_data(encoded)
    assert parsed_long is not None
    assert parsed_long["approval_kind"] == "schedule_edit"
    assert parsed_long["decision"] == "once"
    assert parsed_long["is_ref"] == "1"
    assert resolve_locator(
        "schedule_edit", parsed_long["approval_id"], [long_id]
    ) == long_id
