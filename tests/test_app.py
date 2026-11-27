"""End-to-end smoke tests for the composed LEO application."""

from __future__ import annotations

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.core import LeoApp


@pytest.fixture
def mock_settings() -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock",
        fallback="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    return settings


@pytest.mark.asyncio
async def test_app_initializes_and_all_engines_attached(mock_settings: Settings) -> None:
    app = LeoApp(mock_settings)
    await app.initialize()
    health = await app.health()
    assert health["status"] == "ok"
    assert health["tools"] >= 9
    assert health["skills"] >= 12
    expected_handlers = sum(
        1 for spec in await app.agent_registry.list(enabled_only=True)
        if callable(app.agent_registry.get_handler(spec.id))
    )
    assert health["agents"] == expected_handlers
    assert all(
        getattr(app, name) is not None
        for name in ("router", "registry", "executor", "memory", "search", "tasks", "scheduler", "agent_registry", "agent_engine", "skills", "factory", "gateway")
    )
    await app.shutdown()


@pytest.mark.asyncio
async def test_chat_records_conversation_and_usage(mock_settings: Settings) -> None:
    app = LeoApp(mock_settings)
    await app.initialize()
    result = await app.ask("Hello LEO", channel="test")
    assert result["content"]
    assert result["provider"] == "mock"
    assert result["conversation_id"]
    assert (await app.router.usage_report())
    await app.shutdown()


@pytest.mark.asyncio
async def test_memory_persists_across_app_restarts(mock_settings: Settings) -> None:
    first = LeoApp(mock_settings)
    await first.initialize()
    record = await first.remember("The user prefers dark mode", tags=["preference"])
    await first.shutdown()

    second = LeoApp(mock_settings)
    await second.initialize()
    found = await second.search.search("dark mode", limit=3)
    assert any(m.id == record.id for m in found)
    await second.shutdown()


@pytest.mark.asyncio
async def test_safe_tool_calls_run_before_a_call_that_may_wait(mock_settings: Settings) -> None:
    """A pending card must not hold up unrelated steps in the same batch."""
    mock_settings.security.wait_for_approval = True
    app = LeoApp(mock_settings)
    await app.initialize()
    calls = [
        {"id": "gated", "name": "shell", "arguments": {"command": "git push --force origin main"}},
        {"id": "safe", "name": "read_file", "arguments": {"path": "README.md"}},
        {"id": "also-gated", "name": "shell", "arguments": {"command": "rm -rf build"}},
    ]

    ordered = app._safe_calls_before_wait(calls)

    assert [call["id"] for call in ordered] == ["safe", "gated", "also-gated"]
    # The original batch is untouched: provider pairing still follows the ids.
    assert [call["id"] for call in calls] == ["gated", "safe", "also-gated"]
    await app.shutdown()
