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
    # Hermetic: LEO_SERVICE_UNITS outranks LEO_SERVICE_UNIT, so an ambient copy
    # (the service env file sets it) would silently override the single unit
    # this test is about.
    monkeypatch.delenv("LEO_SERVICE_UNITS", raising=False)
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


def test_restart_targets_prefers_explicit_multi_unit(monkeypatch):
    monkeypatch.setenv("LEO_SERVICE_UNITS", "leo-gateway.service, leo-ui.service ,leo-brain.service")
    monkeypatch.delenv("LEO_SERVICE_UNIT", raising=False)

    units, source = builtins._restart_targets()

    assert units == ["leo-gateway.service", "leo-ui.service", "leo-brain.service"]
    assert source == "LEO_SERVICE_UNITS"


def test_own_units_reads_this_process_cgroup(monkeypatch):
    monkeypatch.setattr(
        builtins.Path, "read_text",
        lambda self: "0::/system.slice/leo-brain.service\n",
        raising=False,
    )

    assert builtins._own_units() == ["leo-brain.service"]


def test_restart_never_restarts_the_disabled_leftover_by_accident(monkeypatch):
    """The regression: an unset env used to hardcode leo-agent.service.

    On this host that unit is disabled and cannot bind 8765, so restarting it
    restarted nothing and started a crash loop instead. Without an explicit
    setting the targets must come from the units we can actually see.
    """
    monkeypatch.delenv("LEO_SERVICE_UNIT", raising=False)
    monkeypatch.delenv("LEO_SERVICE_UNITS", raising=False)
    monkeypatch.setattr(builtins, "_active_leo_units", lambda: ["leo-ui.service"])
    monkeypatch.setattr(builtins, "_own_units", lambda: ["leo-brain.service"])

    units, source = builtins._restart_targets()

    assert source == "auto-detected"
    assert units == ["leo-brain.service", "leo-ui.service"]


async def test_restart_reports_honestly_without_systemctl(monkeypatch):
    monkeypatch.setattr(builtins.shutil, "which", lambda _name: None)

    registry = CommandRegistry()
    result = await builtins.restart_command(_context(registry))

    assert result.data["scheduled"] is False
    assert "systemctl is not available" in result.message
