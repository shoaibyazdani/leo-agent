"""Contract tests for the terminal chat (`leo chat` / `leo tui`)."""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import RiskLevel
from leo.core import LeoApp
from leo.interfaces import cli as cli_mod
from leo.interfaces import tui as tui_mod


def _console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(file=stream, force_terminal=False, width=120, highlight=False), stream


async def _mock_app() -> LeoApp:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    app = LeoApp(settings)
    await app.initialize()
    return app


def test_parse_command_recognizes_slash_forms() -> None:
    assert tui_mod.parse_command("  /approve apr_123 ") == ("approve", ["apr_123"])
    assert tui_mod.parse_command("/DENY apr_9") == ("deny", ["apr_9"])
    assert tui_mod.parse_command("/quit") == ("quit", [])
    assert tui_mod.parse_command("hello there") is None
    assert tui_mod.parse_command("/") is None
    assert tui_mod.parse_command("   ") is None


def test_tool_line_is_one_bounded_line_for_any_shape() -> None:
    assert tui_mod.tool_line("read_file", '{"path": "a.txt"}') == 'read_file({"path": "a.txt"})'
    assert tui_mod.tool_line("t", {"a": 1}) == 't({"a": 1})'
    assert tui_mod.tool_line("t", None) == "t()"
    long = tui_mod.tool_line("t", "x" * 500)
    assert len(long) <= 135 and "…" in long


@pytest.mark.asyncio
async def test_turn_streams_a_complete_answer_and_tracks_conversation() -> None:
    app = await _mock_app()
    try:
        console, stream = _console()
        tui = tui_mod.ChatTUI(app, console=console)
        result = await tui.turn("hello")
        output = stream.getvalue()
        assert result.get("status") == "completed"
        assert "mock: hello" in output
        assert tui.conversation_id, "a conversation id must be assigned by run.started"
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_approve_command_resolves_as_operator_principal() -> None:
    app = await _mock_app()
    try:
        decision = await app.permissions.authorize(
            "learning:engine", "learning.note", None, RiskLevel.HIGH
        )
        assert decision.approval_id
        console, stream = _console()
        tui = tui_mod.ChatTUI(app, console=console)
        assert await tui.run_command("approve", [decision.approval_id]) is True
        rows = await app.permissions.pending_approvals()
        assert all(row["id"] != decision.approval_id for row in rows)
        assert "approved" in stream.getvalue().lower()
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_unknown_command_returns_false_and_approve_needs_an_id() -> None:
    app = await _mock_app()
    try:
        console, stream = _console()
        tui = tui_mod.ChatTUI(app, console=console)
        assert await tui.run_command("bogus", []) is False
        assert await tui.run_command("approve", []) is True
        assert "usage" in stream.getvalue()
    finally:
        await app.shutdown()


def test_cli_exposes_chat_and_tui_commands() -> None:
    import inspect

    commands = {command.name: command for command in cli_mod.cli.registered_commands}
    assert {"chat", "tui"} <= set(commands)
    # Both commands accept --conversation so a session can be resumed.
    for name in ("chat", "tui"):
        signature = inspect.signature(commands[name].callback)
        assert "conversation" in signature.parameters
    runner = CliRunner()
    help_result = runner.invoke(cli_mod.cli, ["chat", "--help"])
    assert help_result.exit_code == 0
    assert "--conversation" in help_result.stdout
