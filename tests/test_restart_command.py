"""``/restart`` and ``leo service`` — restarting LEO without a shell.

The service could only be restarted by remembering systemd and the unit name.
Two surfaces now exist: the ``/restart`` slash command (delivery-safe: the
restart happens after the reply leaves) and ``leo service`` on the CLI.

These tests pin the two things that matter: the command is registered and
dispatches a DETACHED child (so the restart survives this process dying), and
it refuses cleanly when systemctl is not available.
"""

from __future__ import annotations

import pytest

from leo.commands import builtins
from leo.commands.registry import CommandContext, CommandRegistry


def _context(registry: CommandRegistry) -> CommandContext:
    return CommandContext(
        app=None,
        principal="owner",
        surface="telegram",
        channel="telegram",
        argv=[],
        raw="/restart",
        registry=registry,
    )


def test_restart_is_registered():
    registry = CommandRegistry()
    builtins.register_builtins(registry)

    spec = registry.get("restart")

    assert spec is not None
    assert "reboot" in spec.aliases
    assert spec.handler is builtins.restart_command


async def test_restart_spawns_a_detached_child(monkeypatch):
    spawned: list[tuple[list[str], dict]] = []

    class _FakePopen:
        def __init__(self, argv, **kwargs):
            spawned.append((argv, kwargs))

    monkeypatch.setattr(builtins.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(builtins.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setenv("LEO_SERVICE_UNIT", "leo-agent.service")
    monkeypatch.setenv("LEO_RESTART_DELAY_SECONDS", "3")

    registry = CommandRegistry()
    result = await builtins.restart_command(_context(registry))

    assert result.ok
    assert result.data["scheduled"] is True
    assert result.data["delay_seconds"] == 3
    argv, kwargs = spawned[0]
    assert argv[0] == "/bin/sh"
    assert "systemctl restart leo-agent.service" in argv[2]
    # The restart must outlive the process that asked for it.
    assert kwargs["start_new_session"] is True


async def test_restart_reports_honestly_without_systemctl(monkeypatch):
    monkeypatch.setattr(builtins.shutil, "which", lambda _name: None)

    registry = CommandRegistry()
    result = await builtins.restart_command(_context(registry))

    assert result.data["scheduled"] is False
    assert "systemctl is not available" in result.message
