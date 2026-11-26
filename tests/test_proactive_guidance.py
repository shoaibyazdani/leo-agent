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


def test_proactive_mode_validates_like_other_automation_settings() -> None:
    assert AutomationConfig().proactive_mode == "offer"
    assert Settings.model_validate(
        {"automation": {"proactive_mode": "off"}}
    ).automation.proactive_mode == "off"
    with pytest.raises(ValidationError):
        AutomationConfig(proactive_mode="yolo")
