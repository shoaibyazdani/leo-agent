"""Turn budget and per-tool deadline: both come from settings, neither is a hard wall.

Two hardcoded numbers used to make a self-edit impossible on a real turn: eight
tool rounds (so read -> edit -> test could not finish) and a 30-second ceiling
that silently overrode the operator's configured shell timeout. These tests pin
the settings wiring, because a constant that is not reachable from config is
indistinguishable from a bug when it blocks work.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, builtins
from leo.tools.executor import DEFAULT_TOOL_TIMEOUT_SECONDS


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.security.workspace_only = True
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    builtins.register_builtins(registry)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events,
        permissions=permissions, registry=registry, executor=executor,
    )


def test_budget_defaults_are_end_to_end_not_a_wall() -> None:
    settings = Settings()
    assert settings.max_chat_tool_rounds == 250
    assert DEFAULT_TOOL_TIMEOUT_SECONDS == 180.0
    assert settings.security.shell_timeout_seconds == 180


def test_budget_is_operator_configurable() -> None:
    settings = Settings.model_validate({"max_chat_tool_rounds": 40})
    assert settings.max_chat_tool_rounds == 40


def test_budget_is_bounded_but_not_tiny() -> None:
    with pytest.raises(Exception):
        Settings.model_validate({"max_chat_tool_rounds": 0})
    with pytest.raises(Exception):
        Settings.model_validate({"max_chat_tool_rounds": 10_000})


async def test_app_reads_the_budget_from_settings() -> None:
    from leo.core.app import LeoApp

    settings = Settings.model_validate({"max_chat_tool_rounds": 12})
    settings.security.allowed_roots = [Path.cwd()]
    assert LeoApp(settings).round_budget == 12


async def test_app_falls_back_to_the_constant_when_settings_lack_it() -> None:
    from leo.core import app as app_module

    settings = Settings()
    settings.security.allowed_roots = [Path.cwd()]
    app = app_module.LeoApp(settings)
    assert app.round_budget == app_module.MAX_CHAT_TOOL_ROUNDS == 250


async def test_shell_deadline_comes_from_the_operator_setting(env) -> None:
    env.settings.security.shell_timeout_seconds = 180
    registration = env.registry.get("shell")

    assert env.executor._configured_timeout(registration) == 180.0


async def test_a_per_call_timeout_cannot_exceed_the_configured_one(env) -> None:
    """The shell handler clamps its own argument; the executor honors the setting."""
    env.settings.security.shell_timeout_seconds = 60
    call = ToolCall(name="shell", arguments={"command": "true"})

    resolved = env.executor._timeout_for(env.registry.get("shell"), call)

    assert resolved == 60.0, "a registered tool must not fall back to the 30s constant"


async def test_a_tool_without_a_configured_deadline_uses_the_default(env) -> None:
    call = ToolCall(name="read_file", arguments={"path": str(env.tmp_path / "x")})
    registration = env.registry.get("read_file")

    assert env.executor._timeout_for(registration, call) == DEFAULT_TOOL_TIMEOUT_SECONDS
