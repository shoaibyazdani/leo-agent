"""Centralized-copy contract tests replacing locale-catalog/prompt-cache checks.

LEO has no locale subsystem and rebuilds its system prompt every turn. The
stable contract is instead the frozen card copy, its DTO/event projections, the
API projection and the shared UI renderer.
"""

from __future__ import annotations

from pathlib import Path

from leo import db
from leo.config import Settings
from leo.events import EventBus
from leo.security import approval_cards as cards
from leo.security.permissions import PermissionManagerImpl

ROOT = Path(__file__).resolve().parents[1]
SCHEDULES_DOC = (ROOT / "docs" / "SCHEDULES.md").read_text(encoding="utf-8")
UI_RENDERER = (ROOT / "leo" / "interfaces" / "static" / "ui" / "approvals.js").read_text(
    encoding="utf-8"
)


def test_copy_is_one_frozen_english_catalog_not_a_locale_subsystem() -> None:
    assert set(cards.KIND_CONFIG) == {
        cards.KIND_SCHEDULE_CREATE, cards.KIND_SCHEDULE_EDIT, cards.KIND_COMMAND
    }
    create = cards.kind_config(cards.KIND_SCHEDULE_CREATE)
    edit = cards.kind_config(cards.KIND_SCHEDULE_EDIT)
    assert create.choices == ("once", "session", "always", "deny")
    assert edit.choices == ("once", "deny")
    assert create.header == "LEO wants to schedule a job"
    assert create.labels == ("Allow", "Allow for this chat", "Always allow new jobs", "Deny")
    assert "{window}" in create.deadline_template
    assert not (ROOT / "locales").exists()
    lower = SCHEDULES_DOC.lower()
    assert "no locale subsystem" in lower
    assert "no system-prompt cache" in lower
    assert "locale catalog" in lower and "cache invalidation" in lower


async def test_dto_event_api_and_ui_share_one_card_body_and_choices(tmp_path) -> None:
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    permissions = PermissionManagerImpl(Settings(), EventBus())
    body = 'Schedule "Water" (every day at 8am): Drink water.'
    card = await permissions.create_schedule_card(
        kind=cards.KIND_SCHEDULE_CREATE,
        body=body,
        proposal_digest="central-digest",
        requested_by="chat:web",
        conversation_id="conv-copy",
    )

    view = card.view().model_dump(mode="json")
    event = card.event_payload()
    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (card.approval_id,))
    projected = cards.project_cards([row])[0].view().model_dump(mode="json")

    for projection in (view, event, projected):
        assert projection["body"] == body
        assert projection["proposal_digest"] == "central-digest"
    assert [item["value"] for item in view["choices"]] == list(
        cards.SCHEDULE_CREATE_CHOICES
    )
    assert event["choices"] == list(cards.SCHEDULE_CREATE_CHOICES)
    assert [item["label"] for item in view["choices"]] == list(
        cards.kind_config(cards.KIND_SCHEDULE_CREATE).labels
    )
    assert projected == view

    for field in ("view.body", "view.choices", "view.proposal_digest", "view.deadline",
                  "view.deny_reason", "view.approval_id"):
        assert field in UI_RENDERER, f"UI renderer dropped {field}"
    # The only mention is a comment prohibiting the API; no assignment exists.
    assert ".innerHTML =" not in UI_RENDERER
    assert ".innerHTML=" not in UI_RENDERER
