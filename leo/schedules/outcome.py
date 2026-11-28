"""Per-turn schedule outcome registry and truthful final-answer enforcement.

A schedule tool can return ``approval_required`` while the model still writes a
confident "created it" answer. This registry is task-local: every schedule
operation records what really happened, and the terminal answer is corrected
from the durable store before it is persisted or streamed.

It is a reporting/authority check, not a second schedule store. Every truth
lookup reads the same ``approvals``/``schedules`` rows the rest of LEO uses.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

# Statuses that are never "created/updated/removed". A final answer may not
# claim success for any of them.
UNSUCCESSFUL_STATUSES = frozenset(
    {
        "approval_required",
        "pending",
        "denied",
        "timed_out",
        "expired",
        "stale",
        "failed",
        "no_change",
        "no_live_run",
    }
)
_SUCCESS_STATUSES = frozenset(
    {"created", "updated", "removed", "deleted", "running", "cancelled"}
)
# Affirmative phrases, not bare verbs: "nothing was created yet" is already a
# truthful gate report and must not be rewritten.
_CLAIM_TOKENS = (
    "i created",
    "i've created",
    "i have created",
    "created the",
    "created it",
    "created your",
    "i scheduled",
    "i've scheduled",
    "scheduled the",
    "scheduled it",
    "added the",
    "added it",
    "set up the",
    "set up your",
    "saved the",
    "updated the",
    "updated it",
    "changed the",
    "changed it",
    "removed the",
    "removed it",
    "deleted the",
    "deleted it",
    "will run",
    "is all set",
    "all set",
    "successfully",
    "done —",
    "done -",
    "done,",
    "confirmed",
)
_TRUTH_MARKERS = (
    "not created",
    "not changed",
    "not removed",
    "approval",
    "pending",
    "denied",
    "awaiting",
    "could not",
    "failed",
    "no schedule",
    "cancelled",
    "no live run",
)

def _bounded_stored_reason(value: Any) -> str | None:
    """Keep a durable deny reason bounded when it is projected to the model."""
    text = str(value or "").strip()
    return text[:500] if text else None


# Surfaces that cannot capture free text store an explicit sentinel instead of
# pretending the human gave a reason. Gateway taps use "no reason supplied";
# PermissionManager uses "denied by operator" when web/TUI/API denials carry no
# free text. Both are system fallbacks, never human-authored reasons: keep the
# durable record for audit, but normalize both to "no reason given" for the
# model. Adding a new fallback at a denial source requires adding it here.
_PLACEHOLDER_DENY_REASONS = frozenset(
    {"no reason supplied", "denied by operator"}
)


def _is_placeholder_deny_reason(reason: str | None) -> bool:
    return bool(reason) and reason.strip().lower() in _PLACEHOLDER_DENY_REASONS


def _model_deny_reason(value: Any) -> str | None:
    """Project a stored reason to the model without leaking a sentinel string."""
    reason = _bounded_stored_reason(value)
    if reason is None:
        return None
    return "no reason given" if _is_placeholder_deny_reason(reason) else reason


_STATUS_COPY: dict[str, str] = {
    "created": "created",
    "updated": "updated",
    "removed": "removed",
    "deleted": "removed",
    "running": "started as a background task",
    "cancelled": "live run cancelled",
    "approval_required": "not created yet — waiting for approval",
    "pending": "not created yet — waiting for approval",
    "denied": "not created — denied",
    "timed_out": "not created — approval timed out",
    "expired": "not created — approval expired",
    "stale": "not created — the proposal went stale",
    "failed": "not created — the schedule operation failed",
    "no_change": "unchanged",
    "no_live_run": "no live run to cancel",
}


@dataclass
class ScheduleOutcome:
    """What one schedule tool call actually did."""

    operation: str
    status: str
    name: str | None = None
    schedule_id: str | None = None
    approval_id: str | None = None
    proposal_id: str | None = None
    message: str = ""
    # The reason a human gave when denying the card. It is read from the
    # approval row at resolution time, so it is never taken from model output.
    deny_reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.status in _SUCCESS_STATUSES

    def as_dict(self) -> dict[str, Any]:
        data = {
            "operation": self.operation,
            "status": self.status,
            "name": self.name,
            "schedule_id": self.schedule_id,
            "approval_id": self.approval_id,
            "proposal_id": self.proposal_id,
            "message": self.message,
        }
        reason = _bounded_stored_reason(self.deny_reason)
        detail = dict(self.detail)
        if reason:
            data["deny_reason"] = _model_deny_reason(reason)
            if _is_placeholder_deny_reason(reason):
                # Audit-only marker: the raw sentinel stays in the durable
                # approval row, but must not be replayed to the model in this
                # tool result detail.
                detail.setdefault("deny_reason_placeholder", True)
        if detail:
            data["detail"] = detail
        return data


class ScheduleOutcomeRegistry:
    """Append-only, task-local record of schedule outcomes in one turn."""

    def __init__(self) -> None:
        self._items: list[ScheduleOutcome] = []

    def record(self, outcome: ScheduleOutcome) -> ScheduleOutcome:
        self._items.append(outcome)
        return outcome

    def outcomes(self) -> list[ScheduleOutcome]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()


_CURRENT_OUTCOMES: contextvars.ContextVar[ScheduleOutcomeRegistry | None] = (
    contextvars.ContextVar("leo_schedule_outcomes", default=None)
)


def set_schedule_outcomes(
    registry: ScheduleOutcomeRegistry | None,
) -> contextvars.Token:
    return _CURRENT_OUTCOMES.set(registry)


def reset_schedule_outcomes(token: contextvars.Token) -> None:
    _CURRENT_OUTCOMES.reset(token)


def current_schedule_outcomes() -> ScheduleOutcomeRegistry | None:
    return _CURRENT_OUTCOMES.get()


def record_schedule_outcome(outcome: ScheduleOutcome) -> ScheduleOutcome:
    registry = current_schedule_outcomes()
    if registry is not None:
        registry.record(outcome)
    return outcome


async def resolve_outcome_truth(outcome: ScheduleOutcome) -> ScheduleOutcome:
    """Return the outcome with status verified against durable rows.

    An ``approval_required`` record may have been decided after the tool call
    returned. A created record may name a row that was never written. Both are
    resolved here so the final answer cannot outrun the store.
    """
    from .. import db  # local import keeps this module import-light

    status = outcome.status
    operation = outcome.operation
    deny_reason = _bounded_stored_reason(outcome.deny_reason)

    if outcome.approval_id:
        row = await db.fetchone(
            "SELECT status, deny_reason FROM approvals WHERE id=?", (outcome.approval_id,)
        )
        if row is not None:
            approval_status = str(row["status"] or "")
            if approval_status == "approved":
                status = _success_status(operation)
            elif approval_status == "denied":
                status = "denied"
                # The human's reason is durable truth, not model narration.
                deny_reason = _bounded_stored_reason(row["deny_reason"]) or deny_reason
            elif approval_status in ("expired", "stale"):
                status = "timed_out" if approval_status == "expired" else "stale"
            elif approval_status == "pending":
                status = "approval_required"

    if status in _SUCCESS_STATUSES and status != "cancelled":
        schedule_id = outcome.schedule_id
        if schedule_id:
            row = await db.fetchone("SELECT id FROM schedules WHERE id=?", (schedule_id,))
            if row is None and status != "removed":
                status = "failed"
        elif status in ("created", "updated"):
            # No durable schedule id was ever reported: that is not success.
            if outcome.name:
                row = await db.fetchone(
                    "SELECT id FROM schedules WHERE name=? ORDER BY created_at DESC LIMIT 1",
                    (outcome.name,),
                )
                if row is None:
                    status = "failed"
            else:
                status = "failed"

    if status == outcome.status and deny_reason == outcome.deny_reason:
        return outcome
    return ScheduleOutcome(
        operation=operation,
        status=status,
        name=outcome.name,
        schedule_id=outcome.schedule_id,
        approval_id=outcome.approval_id,
        proposal_id=outcome.proposal_id,
        message=outcome.message,
        deny_reason=deny_reason,
        detail=dict(outcome.detail),
    )


def _success_status(operation: str) -> str:
    if operation in ("remove", "delete"):
        return "removed"
    if operation == "update":
        return "updated"
    if operation in ("run_now", "run"):
        return "running"
    if operation == "cancel":
        return "cancelled"
    return "created"


async def resolve_outcomes(
    outcomes: list[ScheduleOutcome],
) -> list[ScheduleOutcome]:
    return [await resolve_outcome_truth(item) for item in outcomes]


def summary_line(outcome: ScheduleOutcome) -> str:
    label = f'Schedule "{outcome.name}"' if outcome.name else "Schedule"
    copy = _STATUS_COPY.get(outcome.status, outcome.status)
    line = f"{label}: {copy}"
    if outcome.status == "approval_required" and outcome.approval_id:
        line += f" (approval {outcome.approval_id})"
    elif outcome.message:
        line += f" ({outcome.message})"
    if outcome.status == "denied":
        reason = _model_deny_reason(outcome.deny_reason)
        if reason:
            line += f" (reason: {reason})"
    return line


def truthful_summary(outcomes: list[ScheduleOutcome]) -> str:
    lines = [summary_line(item) for item in outcomes]
    text = "\n".join(lines) + "\n\nNo schedule was created or changed unless the line above says so."
    # Teknium's decline/timeout copy (Hermes PR #158289): a decline is an
    # answer — the model says in one line what it wanted and never retries.
    if any(item.status == "denied" for item in outcomes):
        text += "\nTell them in one line what you wanted to schedule, and do not retry it."
    elif any(item.status in ("timed_out", "expired") for item in outcomes):
        text += "\nDo not retry it."
    return text


def enforce_truthful_answer(content: str, outcomes: list[ScheduleOutcome]) -> str:
    """Replace a false success claim with durable truth.

    Detection is deliberately conservative in the safe direction: when a turn
    contains any non-success schedule outcome and the answer either claims
    success or says nothing recognisably truthful about the gate, the answer is
    replaced by the per-outcome store summary.

    Exception: when every unfinished outcome is a card already shown to the
    human (approval_required/pending with an approval id), the card IS the
    message — it states the job, the time, and that nothing is created unless
    answered — so the answer is silenced instead of replaced. The gateway
    sends nothing for an empty answer, the durable record still holds the
    pending state, and a false success claim is still swallowed, never shown.
    When no card was shown (no approval id), the proposal is invisible
    without text, so the store summary still speaks for it.
    """
    bad = [item for item in outcomes if not item.successful]
    if not bad:
        return content
    if all(
        item.status in ("approval_required", "pending") and item.approval_id
        for item in bad
    ):
        return ""
    text = str(content or "")
    lowered = text.lower()
    claims_success = any(token in lowered for token in _CLAIM_TOKENS)
    states_truth = any(marker in lowered for marker in _TRUTH_MARKERS)
    if claims_success or not states_truth:
        return truthful_summary(outcomes)
    return content


__all__ = [
    "ScheduleOutcome",
    "ScheduleOutcomeRegistry",
    "UNSUCCESSFUL_STATUSES",
    "current_schedule_outcomes",
    "enforce_truthful_answer",
    "record_schedule_outcome",
    "reset_schedule_outcomes",
    "resolve_outcome_truth",
    "resolve_outcomes",
    "set_schedule_outcomes",
    "summary_line",
    "truthful_summary",
]
