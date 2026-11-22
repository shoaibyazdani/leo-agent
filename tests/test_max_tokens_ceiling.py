"""A reasoning model needs output budget left over after it thinks.

With max_tokens 4096 a long turn came back as stop_reason=max_tokens with no text
and no tool call — an empty response the router rightly refuses, which the operator
saw as "The model could not produce an answer". The ceiling is now a configured
value with real room, and both provider kinds send it.
"""

from __future__ import annotations

from leo.config import ModelProviderConfig
from leo.contracts import ChatMessage, ModelRequest
from leo.models.provider import build_provider


def _request() -> ModelRequest:
    return ModelRequest(messages=[ChatMessage(role="user", content="hi")])


def test_the_configured_ceiling_is_used() -> None:
    config = ModelProviderConfig(kind="anthropic-compatible", model="m", base_url="https://example.test/v1", max_tokens=48_000)
    provider = build_provider("p", config)

    assert provider._body(_request(), False)["max_tokens"] == 48_000


def test_the_default_leaves_room_after_thinking() -> None:
    config = ModelProviderConfig(kind="anthropic-compatible", model="m", base_url="https://example.test/v1")
    provider = build_provider("p", config)

    assert provider._body(_request(), False)["max_tokens"] >= 16_000


def test_a_caller_supplied_ceiling_still_wins() -> None:
    config = ModelProviderConfig(kind="anthropic-compatible", model="m", base_url="https://example.test/v1", max_tokens=32_000)
    provider = build_provider("p", config)

    assert provider._body(_request(), False)["max_tokens"] == 32_000


def test_the_openai_path_carries_it_too() -> None:
    config = ModelProviderConfig(kind="openai-compatible", model="m", base_url="https://example.test/v1", max_tokens=40_000)
    provider = build_provider("p", config)

    assert provider._body(_request(), False)["max_tokens"] == 40_000
