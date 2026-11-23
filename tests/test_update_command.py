"""``/update`` — check first, install only on request, never leave a half state.

An updater that can rewrite the running checkout has to be boring: report
without touching anything, refuse every uncertain precondition, and fast-forward
only. These tests pin the four refusals plus the two happy paths, with git
mocked so nothing here can touch a real repository.
"""

from __future__ import annotations

import pytest

from leo.commands import builtins
from leo.commands.registry import CommandContext, CommandRegistry


def _context(registry: CommandRegistry, argv: list[str] | None = None) -> CommandContext:
    return CommandContext(
        app=None,
        principal="owner",
        surface="telegram",
        channel="telegram",
        argv=argv or [],
        raw="/update",
        registry=registry,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("LEO_INSTALL_ROOT", str(tmp_path))
    return tmp_path


def test_update_is_registered():
    registry = builtins.register_builtins(CommandRegistry())

    spec = registry.get("update")

    assert spec is not None
    assert spec.handler is builtins.update_command


async def test_bare_update_reports_without_installing(repo, monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_git(*args, cwd):
        calls.append(args)
        return {
            ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
            ("status", "--porcelain"): (0, ""),
            ("fetch", "--quiet", "origin", "main"): (0, ""),
            ("rev-list", "--count", "HEAD..origin/main"): (0, "2"),
            ("rev-parse", "--short", "HEAD"): (0, "abc1234"),
            ("log", "--oneline", "-n", "8", "HEAD..origin/main"): (0, "aaa one fix\nbbb two fix"),
        }.get(args, (0, ""))

    monkeypatch.setattr(builtins, "_git", fake_git)
    registry = CommandRegistry()

    result = await builtins.update_command(_context(registry))

    assert result.data["behind"] == 2
    assert result.data["updated"] is False
    assert "update(s) available" in result.message
    # read-only: no merge, no dependency install
    assert not any(args[0] == "merge" for args in calls)


async def test_dirty_tree_refuses(repo, monkeypatch):
    monkeypatch.setattr(builtins, "_git", lambda *a, cwd: (
        (0, "main") if a[:2] == ("rev-parse", "--abbrev-ref")
        else (0, " M leo/core/app.py\n?? notes.md")
    ))
    registry = CommandRegistry()

    result = await builtins.update_command(_context(registry, ["apply"]))

    assert result.data["updated"] is False
    assert result.data["dirty_files"] == 2
    assert "uncommitted change" in result.message


async def test_not_a_checkout_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("LEO_INSTALL_ROOT", str(tmp_path))  # no .git
    registry = CommandRegistry()

    result = await builtins.update_command(_context(registry))

    assert result.data["updated"] is False
    assert "Not a git checkout" in result.message


async def test_already_up_to_date(repo, monkeypatch):
    monkeypatch.setattr(builtins, "_git", lambda *a, cwd: {
        ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
        ("status", "--porcelain"): (0, ""),
        ("fetch", "--quiet", "origin", "main"): (0, ""),
        ("rev-list", "--count", "HEAD..origin/main"): (0, "0"),
        ("rev-parse", "--short", "HEAD"): (0, "abc1234"),
    }.get(a, (0, "")))
    registry = CommandRegistry()

    result = await builtins.update_command(_context(registry, ["apply"]))

    assert result.data["updated"] is False
    assert "Already up to date" in result.message


async def test_apply_fast_forwards_and_restarts(repo, monkeypatch):
    git_calls: list[tuple[str, ...]] = []

    def fake_git(*args, cwd):
        git_calls.append(args)
        return {
            ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
            ("status", "--porcelain"): (0, ""),
            ("fetch", "--quiet", "origin", "main"): (0, ""),
            ("rev-list", "--count", "HEAD..origin/main"): (0, "1"),
            ("log", "--oneline", "-n", "8", "HEAD..origin/main"): (0, "ccc new feature"),
            ("diff", "--name-only", "HEAD..origin/main"): (0, "leo/core/app.py"),
            ("merge", "--ff-only", "origin/main"): (0, "Fast-forward"),
            ("rev-parse", "--short", "HEAD"): (0, "def5678"),
        }.get(args, (0, ""))

    monkeypatch.setattr(builtins, "_git", fake_git)
    monkeypatch.setattr(builtins.shutil, "which", lambda _n: "/usr/bin/systemctl")
    # Name the unit explicitly: auto-detection shells out to systemctl, which
    # this test does not mock (and which does not exist on every host).
    monkeypatch.setenv("LEO_SERVICE_UNITS", "leo-brain.service")
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        builtins.subprocess, "Popen", lambda argv, **kwargs: spawned.append(argv)
    )
    registry = CommandRegistry()

    result = await builtins.update_command(_context(registry, ["apply"]))

    assert result.data["updated"] is True
    assert result.data["head"] == "def5678"
    assert result.data["restarted"] is True
    assert ("merge", "--ff-only", "origin/main") in git_calls
    assert "Restarting LEO" in result.message
    assert spawned and "systemctl restart" in spawned[0][2]


async def test_diverged_branch_changes_nothing(repo, monkeypatch):
    def fake_git(*args, cwd):
        if args == ("merge", "--ff-only", "origin/main"):
            return (1, "fatal: Not possible to fast-forward, aborting.")
        return {
            ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
            ("status", "--porcelain"): (0, ""),
            ("fetch", "--quiet", "origin", "main"): (0, ""),
            ("rev-list", "--count", "HEAD..origin/main"): (0, "3"),
            ("log", "--oneline", "-n", "8", "HEAD..origin/main"): (0, "ddd away"),
            ("diff", "--name-only", "HEAD..origin/main"): (0, "leo/core/app.py"),
        }.get(args, (0, ""))

    monkeypatch.setattr(builtins, "_git", fake_git)
    registry = CommandRegistry()

    result = await builtins.update_command(_context(registry, ["apply"]))

    assert result.data["updated"] is False
    assert "Nothing was changed" in result.message
