"""Scoped schedule create grants and the absence of edit grants.

A schedule create grant is its own authority: conversation-scoped for
"session", profile-scoped for "always", expiring, revocable and audited. It
never covers an edit, a lifecycle action, a tool call or workspace trust.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel
from leo.security.approval_cards import KIND_SCHEDULE_CREATE, KIND_SCHEDULE_EDIT
from leo.security.permissions import (
    SCHEDULE_ALWAYS_GRANT_DAYS,
    SCHEDULE_CREATE_ACTION,
    SCHEDULE_SESSION_GRANT_SECONDS,
    PermissionManagerImpl,
)
from leo.schedules.clock import FrozenClock

NOW = datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc)


def _settings() -> Settings:
    settings = Settings()
    settings.security.approval_timeout_seconds = 900
    return settings


def _manager(
    *, clock: FrozenClock | None = None, settings: Settings | None = None
) -> PermissionManagerImpl:
    return PermissionManagerImpl(
        settings or _settings(), clock=clock or FrozenClock(NOW)
    )


async def _card(
    permissions: PermissionManagerImpl,
    *,
    kind: str = KIND_SCHEDULE_CREATE,
    requested_by: str = "chat:web-a",
    conversation_id: str = "conv-a",
    digest: str = "digest-consent",
    body: str = 'Schedule "Water" (every day at 8am): Drink water.',
):
    card = await permissions.create_schedule_card(
        kind=kind,
        body=body,
        proposal_digest=digest,
        requested_by=requested_by,
        conversation_id=conversation_id,
    )
    await permissions.present_schedule_card(card.approval_id, surface="web")
    return card


async def _resolve(
    permissions: PermissionManagerImpl,
    card,
    *,
    choice: str,
    requested_by: str = "chat:web-a",
):
    return await permissions.resolve_schedule_card(
        card.approval_id,
        choice=choice,
        resolver="operator:local",
        confirm_proposal_digest=card.proposal_digest,
    )


async def test_allow_once_creates_no_standing_grant() -> None:
    permissions = _manager()
    card = await _card(permissions)

    result = await _resolve(permissions, card, choice="once")
    assert result.status == "approved" and result.grant_id is None
    assert await db.fetchall("SELECT id FROM approval_grants") == []
    assert await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    ) is False


async def test_session_grant_is_conversation_scoped() -> None:
    permissions = _manager()
    card = await _card(permissions, conversation_id="conv-a")

    result = await _resolve(permissions, card, choice="session")
    assert result.grant_id
    grant = await db.fetchone(
        "SELECT principal, subject_type, subject_id, scope, action "
        "FROM approval_grants WHERE id=?",
        (result.grant_id,),
    )
    assert dict(grant) == {
        "principal": "chat:web-a",
        "subject_type": "conversation",
        "subject_id": "conv-a",
        "scope": "conversation",
        "action": SCHEDULE_CREATE_ACTION,
    }
    assert await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    )
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-b"
    )
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-b", conversation_id="conv-a"
    )


async def test_always_grant_is_profile_scoped_and_cross_profile_isolated() -> None:
    permissions = _manager()
    card = await _card(permissions, requested_by="chat:web-a")

    result = await _resolve(permissions, card, choice="always")
    grant = await db.fetchone(
        "SELECT subject_type, subject_id, expires_at FROM approval_grants WHERE id=?",
        (result.grant_id,),
    )
    assert grant["subject_type"] == "profile"
    assert grant["subject_id"] == "chat:web-a"
    assert grant["expires_at"] > NOW.isoformat()
    # A profile grant follows the principal into another conversation.
    assert await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-other"
    )
    # It says nothing about a different profile.
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-b", conversation_id="conv-other"
    )
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-b", profile="chat:web-a"
    )


async def test_edit_cards_never_create_a_grant() -> None:
    permissions = _manager()
    card = await _card(
        permissions, kind=KIND_SCHEDULE_EDIT,
        body='Change "Water": prompt: water -> tea.', digest="edit-consent",
    )
    assert card.choices == ["once", "deny"]

    with pytest.raises(ValueError, match="invalid choice"):
        await _resolve(permissions, card, choice="session")
    result = await _resolve(permissions, card, choice="once")
    assert result.status == "approved"
    assert result.grant_id is None
    assert await db.fetchall("SELECT id FROM approval_grants") == []
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    )


async def test_grants_expire_on_the_injected_clock() -> None:
    clock = FrozenClock(NOW)
    permissions = _manager(clock=clock)
    card = await _card(permissions)
    result = await _resolve(permissions, card, choice="session")
    assert result.grant_id

    clock.advance(seconds=SCHEDULE_SESSION_GRANT_SECONDS - 1)
    assert await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    )
    clock.advance(seconds=2)
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    )


async def test_revocation_is_immediate_durable_and_audited() -> None:
    permissions = _manager()
    card = await _card(permissions)
    result = await _resolve(permissions, card, choice="session")
    assert result.grant_id

    assert await permissions.revoke_schedule_grant(
        result.grant_id, actor="operator:local"
    )
    row = await db.fetchone(
        "SELECT revoked_at FROM approval_grants WHERE id=?", (result.grant_id,)
    )
    assert row["revoked_at"] is not None, "revocation is recorded, not deleted"
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    )
    audit = await db.fetchone(
        "SELECT action, decision FROM audit_log WHERE action='approval.grant.revoke' "
        "AND resource=?",
        (result.grant_id,),
    )
    assert audit is not None and audit["decision"] == "allowed"
    # Re-revoking an already revoked grant is a no-op.
    assert not await permissions.revoke_schedule_grant(result.grant_id)


async def test_manual_mode_ignores_a_standing_create_grant() -> None:
    settings = _settings()
    permissions = _manager(settings=settings)
    card = await _card(permissions)
    result = await _resolve(permissions, card, choice="always")
    assert result.grant_id
    assert await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    )

    settings.security.approval_mode = "manual"
    assert not await permissions.has_live_schedule_create_grant(
        principal="chat:web-a", conversation_id="conv-a"
    )


async def test_a_schedule_grant_does_not_authorize_tools_or_workspace_trust() -> None:
    permissions = _manager()
    card = await _card(permissions)
    result = await _resolve(permissions, card, choice="always")
    assert result.grant_id

    # The action is schedule:create, so ordinary HIGH shell work is still gated.
    decision = await permissions.authorize(
        "operator", "tool:shell", None, RiskLevel.HIGH,
        arguments_digest="an-unrelated-command-digest",
    )
    assert decision.allowed is False and decision.requires_approval is True
    # And no workspace trust was written under a tool action.
    rows = await db.fetchall(
        "SELECT action FROM approval_grants WHERE action IN ('tool:shell','tool:terminal')"
    )
    assert rows == []


def test_grant_bounds_are_sane() -> None:
    assert SCHEDULE_SESSION_GRANT_SECONDS == 8 * 60 * 60
    assert 1 <= SCHEDULE_ALWAYS_GRANT_DAYS <= 365


async def test_fallback_deny_reason_is_audited_but_normalized_to_the_model() -> None:
    """N1: a web/API denial with no free text must not invent a human reason."""
    from leo.schedules.outcome import ScheduleOutcome, resolve_outcome_truth

    permissions = _manager()
    card = await _card(permissions)
    decision = await _resolve(permissions, card, choice="deny")
    assert decision.status == "denied"

    stored = await db.fetchone(
        "SELECT deny_reason FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert stored["deny_reason"] == "denied by operator"

    resolved = await resolve_outcome_truth(
        ScheduleOutcome(
            operation="create", status="approval_required", name="Water",
            approval_id=card.approval_id,
        )
    )
    projected = resolved.as_dict()
    assert projected["status"] == "denied"
    assert projected["deny_reason"] == "no reason given"
    assert "denied by operator" not in json.dumps(projected).lower()
