"""Per-turn schedule outcome registry and truthful final-answer enforcement.

A model that says "created it" after a denied, timed-out, failed or
approval-required schedule must have that claim replaced by durable store
truth — except when the pending job is already on a card in front of the
human: then the card is the message and the answer is silenced instead.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ModelResponse, Usage
from leo.core import LeoApp
from leo.schedules.outcome import (
    ScheduleOutcome,
    ScheduleOutcomeRegistry,
    enforce_truthful_answer,
    resolve_outcome_truth,
    summary_line,
    truthful_summary,
)

NOW = datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc)


def test_truthful_answer_replaces_a_false_success_claim() -> None:
    outcomes = [
        ScheduleOutcome(
            operation="create",
            status="approval_required",
            name="Water",
            proposal_id="prop-1",
            message="waiting",
        )
    ]
    answer = enforce_truthful_answer("Done — I created the Water schedule.", outcomes)
    assert "not created yet" in answer
    assert "Water" in answer
    assert "Done" not in answer


def test_a_shown_card_silences_the_answer() -> None:
    """Regression: 28 Nov 2026 — after a schedule card was shown, the model
    still sent a second message repeating the store summary ("not created
    yet — waiting for approval (approval apr_...)"). The card already states
    the job, the time, and that nothing is created unless answered, so when
    a card was presented the answer is silenced, even for a truthful draft.
    """
    outcomes = [
        ScheduleOutcome(
            operation="create", status="approval_required", name="Water",
            approval_id="apr-1",
        )
    ]
    answer = "The Water schedule is waiting for your approval; nothing was created yet."
    assert enforce_truthful_answer(answer, outcomes) == ""


def test_false_success_claim_with_a_shown_card_is_silenced() -> None:
    outcomes = [
        ScheduleOutcome(
            operation="create", status="approval_required", name="Water",
            approval_id="apr-1",
        )
    ]
    assert enforce_truthful_answer("Done — I created the Water schedule.", outcomes) == ""


def test_pending_without_a_card_still_gets_the_summary() -> None:
    """No card (CLI/API/WhatsApp) means the proposal is invisible without
    text, so the store summary still speaks for it."""
    outcomes = [
        ScheduleOutcome(
            operation="create", status="approval_required", name="Water",
            proposal_id="prop-1",
        )
    ]
    answer = enforce_truthful_answer("Here is the schedule I made for you.", outcomes)
    assert "not created yet" in answer
    assert "Water" in answer


def test_mixed_pending_and_denied_outcomes_still_get_the_summary() -> None:
    outcomes = [
        ScheduleOutcome(
            operation="create", status="approval_required", name="Water",
            approval_id="apr-1",
        ),
        ScheduleOutcome(operation="create", status="denied", name="Tea"),
    ]
    answer = enforce_truthful_answer("Done — I created both schedules.", outcomes)
    assert "not created yet" in answer
    assert "not created — denied" in answer
    assert "Tell them in one line what you wanted to schedule, and do not retry it." in answer


def test_successful_outcomes_never_rewrite_the_answer() -> None:
    outcomes = [ScheduleOutcome(operation="create", status="created", name="Water")]
    answer = "I created the Water schedule."
    assert enforce_truthful_answer(answer, outcomes) == answer


async def test_denied_approval_resolves_to_not_created() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db.set_db_path(Path(tmp) / "test.db")
        await db.initialize()
        await db.execute(
            "INSERT INTO approvals(id,created_at,status,actor,action,resource,risk,"
            "reason,requested_by,metadata,kind,protocol_version,proposal_id,digest,"
            "choices,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "apr-denied", NOW.isoformat(), "denied", "chat:web",
                "schedule.create", "Water", "high", "denied", "chat:web", "{}",
                "schedule_create", 1, "prop-1", "digest", '["deny"]',
                NOW.isoformat(),
            ),
        )
        outcome = ScheduleOutcome(
            operation="create",
            status="approval_required",
            name="Water",
            approval_id="apr-denied",
            proposal_id="prop-1",
        )
        resolved = await resolve_outcome_truth(outcome)
        assert resolved.status == "denied"
        answer = enforce_truthful_answer("Done, the job was created.", [resolved])
        assert "not created" in answer and "Done" not in answer


async def test_a_denied_outcome_exposes_the_stored_reason_to_the_model() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db.set_db_path(Path(tmp) / "test.db")
        await db.initialize()
        await db.execute(
            "INSERT INTO approvals(id,created_at,status,actor,action,resource,risk,"
            "reason,requested_by,metadata,kind,protocol_version,proposal_id,digest,"
            "choices,expires_at,deny_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "apr-deny-reason", NOW.isoformat(), "denied", "chat:web",
                "schedule.create", "Water", "high", "denied", "chat:web", "{}",
                "schedule_create", 1, "prop-reason", "digest", '["deny"]',
                NOW.isoformat(), "the reminder is too expensive",
            ),
        )
        resolved = await resolve_outcome_truth(
            ScheduleOutcome(
                operation="create", status="approval_required", name="Water",
                approval_id="apr-deny-reason", proposal_id="prop-reason",
            )
        )
        assert resolved.status == "denied"
        # Both the tool-result dict and the terminal answer carry the human's
        # words; the model may not substitute a reason of its own.
        assert resolved.as_dict()["deny_reason"] == "the reminder is too expensive"
        assert "reason: the reminder is too expensive" in truthful_summary([resolved])


def test_placeholder_deny_reason_is_normalized_for_the_model_but_audited() -> None:
    """Tap/web denials have no free text; never present their sentinel as a reason."""
    outcome = ScheduleOutcome(
        operation="create",
        status="denied",
        name="Water",
        deny_reason="no reason supplied",
    )
    projected = outcome.as_dict()
    assert projected["deny_reason"] == "no reason given"
    # Audit marker only: no raw sentinel may be replayed to the model in detail.
    assert projected["detail"]["deny_reason_placeholder"] is True
    assert "no reason supplied" not in json.dumps(projected)
    # The durable record itself is untouched.
    assert outcome.deny_reason == "no reason supplied"

    summary = truthful_summary([outcome])
    assert "reason: no reason given" in summary
    assert "no reason supplied" not in summary


def test_operator_fallback_deny_reason_is_also_normalized() -> None:
    """N1: the PermissionManager fallback is not a human-authored reason."""
    for sentinel in ("denied by operator", "Denied By Operator"):
        outcome = ScheduleOutcome(
            operation="update", status="denied", name="Water",
            deny_reason=sentinel,
        )
        projected = outcome.as_dict()
        assert projected["deny_reason"] == "no reason given"
        assert projected["detail"]["deny_reason_placeholder"] is True
        assert "denied by operator" not in json.dumps(projected).lower()
        assert "reason: no reason given" in summary_line(outcome)
        assert outcome.deny_reason == sentinel

    # A real human reason stays verbatim and gets no placeholder marker.
    human = ScheduleOutcome(
        operation="create", status="denied", name="Water",
        deny_reason="too expensive",
    )
    projected = human.as_dict()
    assert projected["deny_reason"] == "too expensive"
    assert "deny_reason_placeholder" not in projected.get("detail", {})


async def test_resolved_placeholder_deny_reason_reaches_the_model_normalized() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db.set_db_path(Path(tmp) / "test.db")
        await db.initialize()
        await db.execute(
            "INSERT INTO approvals(id,created_at,status,actor,action,resource,risk,"
            "reason,requested_by,metadata,kind,protocol_version,proposal_id,digest,"
            "choices,expires_at,deny_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "apr-placeholder", NOW.isoformat(), "denied", "chat:web",
                "schedule.create", "Water", "high", "denied", "chat:web", "{}",
                "schedule_create", 1, "prop-tap", "digest", '["deny"]',
                NOW.isoformat(), "no reason supplied",
            ),
        )
        resolved = await resolve_outcome_truth(
            ScheduleOutcome(
                operation="create", status="approval_required", name="Water",
                approval_id="apr-placeholder", proposal_id="prop-tap",
            )
        )
        assert resolved.as_dict()["deny_reason"] == "no reason given"
        answer = enforce_truthful_answer("Done — created it.", [resolved])
        assert "no reason given" in answer
        assert "no reason supplied" not in answer
        assert resolved.deny_reason == "no reason supplied"


async def test_a_created_outcome_without_a_row_becomes_failed() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db.set_db_path(Path(tmp) / "test.db")
        await db.initialize()
        resolved = await resolve_outcome_truth(
            ScheduleOutcome(
                operation="create", status="created", name="Ghost",
                schedule_id="sch_missing",
            )
        )
        assert resolved.status == "failed"
        assert "not created" in truthful_summary([resolved])


# --------------------------------------------------------------------------- app integration


class _ClaimingScheduleRouter:
    """First model call asks to create a schedule; the second claims success."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request) -> ModelResponse:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="Setting that up.",
                model="mock-1",
                provider="mock",
                tool_calls=[
                    {
                        "id": "call-schedule-1",
                        "name": "schedule_create",
                        "arguments": {
                            "name": "Water",
                            "schedule": "in 2h",
                            "prompt": "Remind the user to drink water.",
                        },
                    }
                ],
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            content="Done — I created the Water schedule and it will run in two hours.",
            model="mock-1",
            provider="mock",
            usage=Usage(input_tokens=10, output_tokens=8),
        )


async def test_app_final_answer_is_corrected_after_an_approval_required_create() -> None:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    app = LeoApp(settings)
    await app.initialize()
    app.router = _ClaimingScheduleRouter()  # type: ignore[assignment]
    try:
        answer = await app.ask("remind me to drink water in two hours", channel="cli")
    finally:
        await app.shutdown()

    content = answer["content"]
    assert "Done" not in content
    assert "not created yet" in content
    assert "Water" in content
    # The store is the truth: nothing was written.
    assert await db.fetchone("SELECT id FROM schedules WHERE name='Water'") is None
    assert await db.fetchall("SELECT id FROM approvals") == []
    assert len(await db.fetchall("SELECT id FROM schedule_proposals")) == 1
