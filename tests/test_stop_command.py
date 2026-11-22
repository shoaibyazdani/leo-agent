"""/stop must actually stop the turn that is running, from another process.

The command can be handled by one worker loop while a second loop is busy with a
long turn, so the request travels through SQLite and the live turn observes it
between steps. These tests pin both halves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leo import db
from leo.commands.builtins import stop_command
from leo.commands.registry import CommandContext, CommandRegistry
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ModelResponse, Usage
from leo.core import LeoApp


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    settings.max_chat_tool_rounds = 4
    settings.security.allowed_roots = [tmp_path]
    return settings


class _StoppingRouter:
    """Asks for one tool, and requests a stop while the turn is in flight."""

    def __init__(self, app: LeoApp, conversation_id: str) -> None:
        self.app = app
        self.conversation_id = conversation_id
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            await self.app.request_turn_stop(self.conversation_id, requested_by="test")
            tool_call = {
                "id": "call_1", "type": "function",
                "function": {"name": "current_time", "arguments": "{}"},
            }
            return ModelResponse(
                content="working", model="fake", provider="fake",
                usage=Usage(input_tokens=1, output_tokens=1), tool_calls=[tool_call],
            )
        return ModelResponse(
            content="should never be reached", model="fake", provider="fake",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


@pytest.fixture
async def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "stop.db")
    await db.initialize()
    application = LeoApp(_settings(tmp_path))
    await application.initialize()
    yield application
    await application.shutdown()


async def test_a_stop_request_is_scoped_to_the_turn_window(app) -> None:
    before = await db.fetchone("SELECT requested_at FROM turn_cancels LIMIT 1")
    assert before is None

    await app.request_turn_stop("conv-1", requested_by="test")

    # same turn (started before the request) → must stop
    assert await app._turn_stop_requested("conv-1", "2000-01-01T00:00:00+00:00") is True
    # a later turn (started after the request) → must not
    assert await app._turn_stop_requested("conv-1", "2999-01-01T00:00:00+00:00") is False
    # another conversation is untouched
    assert await app._turn_stop_requested("conv-2", "2000-01-01T00:00:00+00:00") is False

    await app._clear_turn_stop("conv-1")
    assert await app._turn_stop_requested("conv-1", "2000-01-01T00:00:00+00:00") is False


async def test_a_mid_turn_stop_ends_the_turn_as_cancelled(app) -> None:
    conversation_id = "conv-stop"
    router = _StoppingRouter(app, conversation_id)
    app.router = router

    result = await app.ask("do something long", channel="cli", conversation_id=conversation_id)

    assert result["status"] == "cancelled", result
    assert "Stopped" in result["content"]
    assert router.calls == 1, "the stop must win before the next model call"
    row = await db.fetchone(
        "SELECT content FROM messages WHERE conversation_id=? AND role='assistant' "
        "ORDER BY id DESC LIMIT 1", (conversation_id,)
    )
    assert row is not None and "Stopped" in row["content"]
    # the request is consumed, so the next turn is not cancelled by it
    assert await db.fetchone("SELECT requested_at FROM turn_cancels LIMIT 1") is None


async def test_slash_stop_without_an_id_stops_the_turn(app) -> None:
    ctx = CommandContext(
        app=app, principal="operator", surface="chat", channel="telegram",
        argv=[], raw="/stop", registry=CommandRegistry(), conversation_id="conv-slash",
    )

    result = await stop_command(ctx)

    assert result is not None
    assert "Stopping" in str(getattr(result, "message", "") or getattr(result, "text", "") or result)
    assert await app._turn_stop_requested("conv-slash", "2000-01-01T00:00:00+00:00") is True
