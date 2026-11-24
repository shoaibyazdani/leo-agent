"""Tests for the /reasoning port: ladder, clamp, wire params and command."""

from __future__ import annotations

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ChatMessage, ModelRequest
from leo.models.provider import AnthropicProvider, OpenAIProvider
from leo.models.reasoning import (
    EFFORT_LADDER,
    clamp_effort,
    parse_effort,
    thinking_budget,
)


# --- vocabulary / clamp ----------------------------------------------------

def test_parse_effort_normalizes_and_rejects():
    assert parse_effort("HIGH") == "high"
    assert parse_effort(" off ") == "none"
    assert parse_effort("min") == "minimal"
    assert parse_effort("extreme") == "max"
    assert parse_effort("nonsense") is None
    assert parse_effort("") is None


def test_clamp_never_escalates_cost():
    # ultra is internal; widest wire stops at max -> nearest weaker is max.
    assert clamp_effort("ultra", ("none", "low", "medium", "high", "max")) == "max"
    # high unsupported -> nearest weaker supported level.
    assert clamp_effort("high", ("none", "low", "medium")) == "medium"
    # nothing weaker -> weakest supported level.
    assert clamp_effort("minimal", ("low", "medium", "high")) == "low"
    # already supported / unknown set / unrecognized -> pass through.
    assert clamp_effort("medium", ("low", "medium", "high")) == "medium"
    assert clamp_effort("ultra", None) == "ultra"
    assert clamp_effort("custom", ("low", "high")) == "custom"
    assert clamp_effort(None, ("low",)) is None


def test_thinking_budget_bounds():
    assert thinking_budget("none", 32_000) is None
    assert thinking_budget(None, 32_000) is None
    assert thinking_budget("high", 32_000) == 8_192
    # Never exceeds the provider ceiling.
    assert thinking_budget("max", 4_000) <= 4_000 - 1_024


# --- wire params -----------------------------------------------------------

def _request(effort: str | None) -> ModelRequest:
    return ModelRequest(
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort=effort,
    )


def _openai_cfg(**kwargs) -> ModelProviderConfig:
    return ModelProviderConfig(
        kind="openai-compatible", base_url="https://example.test/v1",
        model="m", api_key_env=None, **kwargs,
    )


def test_openai_body_adds_reasoning_effort_only_when_set():
    provider = OpenAIProvider(_openai_cfg(), name="m")
    assert "reasoning_effort" not in provider._body(_request(None), stream=False)
    assert provider._body(_request("high"), stream=False)["reasoning_effort"] == "high"
    assert provider._body(_request("off"), stream=False)["reasoning_effort"] == "none"


def test_openai_body_clamps_to_declared_wire():
    cfg = _openai_cfg(reasoning_efforts=["none", "low", "medium", "high"])
    provider = OpenAIProvider(cfg, name="m")
    assert provider._body(_request("ultra"), stream=False)["reasoning_effort"] == "high"


def test_anthropic_body_adds_thinking_block():
    provider = AnthropicProvider(
        ModelProviderConfig(kind="anthropic-compatible", base_url="https://example.test", model="m"),
        name="m",
    )
    provider.max_tokens = 32_000
    assert "thinking" not in provider._body(_request(None), stream=False)
    body = provider._body(_request("high"), stream=False)
    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] == 8_192


def test_precedence_request_over_provider_over_global():
    cfg = _openai_cfg(reasoning_effort="low")
    provider = OpenAIProvider(cfg, name="m")
    provider.default_effort = "medium"
    # request wins
    assert provider._body(_request("high"), stream=False)["reasoning_effort"] == "high"
    # provider config wins over global default
    assert provider._body(_request(None), stream=False)["reasoning_effort"] == "low"
    # global default used only when nothing else set
    provider.config.reasoning_effort = None
    assert provider._body(_request(None), stream=False)["reasoning_effort"] == "medium"


# --- command ---------------------------------------------------------------

@pytest.fixture
async def app(monkeypatch):
    from leo.core import LeoApp

    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    instance = LeoApp(settings)
    await instance.initialize()
    yield instance
    await instance.shutdown()


async def _call(app, text):
    return await app.commands.dispatch(
        text, app=app, principal="local", surface="cli", channel="cli",
        conversation_id="conv-reasoning",
    )


async def test_reasoning_command_is_available(app):
    spec = app.commands.get("reasoning")
    assert spec is not None
    assert spec.available is True


async def test_reasoning_command_set_and_show(app):
    result = await _call(app, "/reasoning high")
    assert result.success is True
    assert "high" in result.message

    shown = await _call(app, "/reasoning")
    assert shown.success is True
    assert "high" in shown.message
    assert shown.data["effort"] == "high"

    # The session effort is applied to chat requests.
    assert await app.reasoning_state.effort_for("conv-reasoning") == "high"


async def test_reasoning_command_display_toggles(app):
    assert (await _call(app, "/reasoning show")).data["show"] is True
    assert (await _call(app, "/reasoning full")).data["full"] is True
    assert (await _call(app, "/reasoning clamp")).data["full"] is False
    assert (await _call(app, "/reasoning hide")).data["show"] is False


async def test_reasoning_command_rejects_unknown_level(app):
    result = await _call(app, "/reasoning banana")
    assert result.success is False
    assert "unknown reasoning level" in result.message


async def test_reasoning_command_reset(app):
    await _call(app, "/reasoning high")
    result = await _call(app, "/reasoning reset")
    assert result.success is True
    assert await app.reasoning_state.effort_for("conv-reasoning") is None
