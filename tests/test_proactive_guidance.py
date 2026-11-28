"""Conditional scheduling guidance in the conversation system prompt.

The hint is present only when proactive mode is enabled and the trusted turn
can show, answer and rewrite a schedule card. It never grants authority to
save a job and is absent from unattended and incapable contexts.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from leo.config import AutomationConfig, Settings
from leo.contracts import TurnContext, turn_context
from leo.core import LeoApp

GUIDANCE_START = "SCHEDULING:"


def _prompt(surface: str | None = None, *, mode: str = "offer", source: str = "chat") -> str:
    settings = Settings()
    settings.automation.proactive_mode = mode
    app = LeoApp(settings)
    if surface is None:
        return app._conversation_system_prompt()
    turn = TurnContext(
        principal="chat:test",
        channel=surface,
        conversation_id="conv-guidance",
        surface=surface,
        origin=surface,
        source=source,
    )
    with turn_context(turn):
        return app._conversation_system_prompt()


def _has_guidance(prompt: str) -> bool:
    return GUIDANCE_START in prompt


@pytest.mark.parametrize("surface", ["tui", "web", "electron", "telegram"])
def test_capable_interactive_surfaces_get_guidance(surface: str) -> None:
    prompt = _prompt(surface)
    assert _has_guidance(prompt)
    assert "schedule_list" in prompt
    assert "schedule_create" in prompt


@pytest.mark.parametrize("surface", ["cli", "api", "scheduled"])
def test_incapable_and_unattended_surfaces_get_no_guidance(surface: str) -> None:
    assert not _has_guidance(_prompt(surface))


def test_no_installed_turn_context_gets_no_guidance() -> None:
    assert not _has_guidance(_prompt())


def test_off_suppresses_guidance_and_create_mode_is_supported() -> None:
    assert not _has_guidance(_prompt("tui", mode="off"))
    assert _has_guidance(_prompt("tui", mode="create"))


def test_agent_source_does_not_inherit_a_capable_surface() -> None:
    assert not _has_guidance(_prompt("tui", source="agent"))


@pytest.mark.parametrize("source", ["scheduled", "schedule", "unattended", "agent"])
def test_unattended_sources_get_no_guidance_on_a_capable_surface(source: str) -> None:
    """R9 pin: both scheduled labels are denied even with a card-capable UI."""
    assert not _has_guidance(_prompt("tui", source=source))


def test_unattended_source_set_is_shared_across_entry_points() -> None:
    """R9/Q1 pin: guidance and grants withhold "agent" too; the tool gate does not.

    App and service share one label set. The executor is intentionally narrower:
    "agent" is a nested agent turn that inherits an attended parent's authority,
    so it must not be treated as an unattended tool-turn.
    """
    from leo.core.app import _UNATTENDED_TURN_SOURCES as APP_SOURCES
    from leo.schedules.service import _UNATTENDED_TURN_SOURCES as SERVICE_SOURCES
    from leo.tools.executor import _UNATTENDED_SOURCES as EXECUTOR_SOURCES

    assert APP_SOURCES == SERVICE_SOURCES
    assert "agent" in APP_SOURCES
    assert EXECUTOR_SOURCES == {"scheduled", "schedule", "unattended"}


def test_guidance_does_not_claim_an_autonomous_save() -> None:
    line = next(
        part for part in _prompt("tui").split("\n\n") if part.startswith(GUIDANCE_START)
    )
    lowered = line.lower()
    for token in (
        "schedule_list first", "one-shot", "finite run", "user-authorized",
        "pending until the human answers", "never retry after a decline",
    ):
        assert token in lowered, token
    for forbidden in (
        "saved automatically", "save it without", "already saved",
        "no approval needed", "autonomously",
    ):
        assert forbidden not in lowered, forbidden


def test_guidance_pins_exact_time_and_clean_prompt() -> None:
    """Regression: 28 Nov 2026 — a schedule moved 9am to 8am unasked, and the
    prompt leaked style instructions ("Keep it to 1-2 short lines, Roman
    Urdu mixed with English.") into the delivered reminder."""
    line = next(
        part for part in _prompt("telegram").split("\n\n") if part.startswith(GUIDANCE_START)
    )
    lowered = line.lower()
    for token in (
        "exact time the user stated",
        "never move it earlier or later without asking",
        "only the message to deliver",
        "never instructions about style",
    ):
        assert token in lowered, token


def test_granted_guidance_pins_exact_time_and_clean_prompt() -> None:
    """The live-grant variant creates schedules too, so it carries the same
    two rules; a grant never widens what may be scheduled."""
    from leo.core.app import _GRANTED_SCHEDULE_GUIDANCE

    lowered = _GRANTED_SCHEDULE_GUIDANCE.lower()
    for token in (
        "exact time the user stated",
        "never move it earlier or later without asking",
        "only the message to deliver",
        "never instructions about style",
    ):
        assert token in lowered, token


def test_schedule_create_prompt_field_forbids_meta_instructions() -> None:
    """The prompt field is what the model fills: it must say the field holds
    only the message to deliver, never style/language/length/channel
    instructions or a restated delivery target."""
    from leo.tools.automation_tools import _tool_defs

    spec = next(spec for spec, _ in _tool_defs() if spec.name == "schedule_create")
    description = spec.input_schema["properties"]["prompt"]["description"].lower()
    for token in (
        "only the exact message to deliver",
        "never instructions about style",
        "delivery belongs in 'deliver'",
    ):
        assert token in description, token


def test_proactive_mode_validates_like_other_automation_settings() -> None:
    assert AutomationConfig().proactive_mode == "offer"
    assert Settings.model_validate(
        {"automation": {"proactive_mode": "off"}}
    ).automation.proactive_mode == "off"
    with pytest.raises(ValidationError):
        AutomationConfig(proactive_mode="yolo")


async def test_live_create_grant_rewords_guidance_instead_of_promising_a_card(tmp_path) -> None:
    from leo import db
    from leo.security.approval_cards import KIND_SCHEDULE_CREATE

    db.set_db_path(tmp_path / "test.db")
    settings = Settings()
    app = LeoApp(settings)
    await app.initialize()
    try:
        card = await app.permissions.create_schedule_card(
            kind=KIND_SCHEDULE_CREATE,
            body='Schedule "Water" (in 2h): Drink water.',
            proposal_digest="grant-digest",
            requested_by="chat:grant",
            conversation_id="conv-guidance",
        )
        await app.permissions.present_schedule_card(card.approval_id, surface="web")
        decision = await app.permissions.resolve_schedule_card(
            card.approval_id, choice="session", resolver="operator:local",
            confirm_proposal_digest=card.proposal_digest,
        )
        assert decision.grant_id

        turn = TurnContext(
            principal="chat:grant", channel="tui", surface="tui",
            conversation_id="conv-guidance", origin="tui", source="chat",
        )
        assert await app._turn_has_create_grant(turn) is True
        with turn_context(turn):
            prompt = app._conversation_system_prompt(create_grant=True)
        line = next(
            part for part in prompt.split("\n\n") if part.startswith(GUIDANCE_START)
        )
        lowered = line.lower()
        assert "saves immediately without showing a card" in lowered
        assert "user-authorized" in lowered
        assert "pending until the human answers" not in lowered

        other = turn.model_copy(update={"conversation_id": "conv-other"})
        assert await app._turn_has_create_grant(other) is False
        for source in ("scheduled", "schedule", "unattended", "agent"):
            unattended = turn.model_copy(update={"source": source})
            assert await app._turn_has_create_grant(unattended) is False
            with turn_context(unattended):
                assert not _has_guidance(
                    app._conversation_system_prompt(create_grant=True)
                )
    finally:
        await app.shutdown()
