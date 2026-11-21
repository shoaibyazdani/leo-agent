"""Read-only git tools: real repository output, permission path checks and
rejection of mutating commands through the normal ToolExecutorImpl pipeline.

The temporary repository is created inside pytest's tmp_path; no LEO working
tree is created, committed to or modified by these tests.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import git_tools

TOOL_NAMES = {"git_status", "git_diff", "git_log", "git_show"}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )
    return completed.stdout


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("hello\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "initial commit",
    )

    db.set_db_path(tmp_path / "git-tools-test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    assert set(git_tools.register(registry)) == TOOL_NAMES
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        root=tmp_path, settings=settings, events=events,
        permissions=permissions, registry=registry, executor=executor,
    )


async def _call(env, name: str, args: dict | None = None, **kwargs):
    return await env.executor.execute(ToolCall(name=name, arguments=args or {}, **kwargs))


def test_registration_contract() -> None:
    registry = ToolRegistry()
    assert set(git_tools.register(registry)) == TOOL_NAMES
    specs = {spec.name: spec for spec in registry.list_specs()}
    for spec in specs.values():
        assert spec.category == "development"
        assert spec.risk is RiskLevel.LOW
        assert spec.permissions == ["git:read"]
        assert spec.resource_field == "path"
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.timeout_seconds == git_tools.TIMEOUT_SECONDS
    assert specs["git_show"].input_schema["required"] == ["revision"]
    assert registry.validate("git_diff", {"staged": "yes"})
    assert registry.validate("git_status", {"path": "--force"})
    assert registry.validate("git_log", {"max_count": 101})
    assert git_tools.register(registry) == []  # idempotent: keeps existing handlers


async def test_status_reports_real_changes_and_audits(env) -> None:
    (env.root / "tracked.txt").write_text("hello\nworld\n", encoding="utf-8")
    (env.root / "new.txt").write_text("new\n", encoding="utf-8")

    result = await _call(env, "git_status", {"branch": True})

    assert result.success and result.decision == "allowed"
    assert result.output["root"] == str(env.root)
    assert result.output["branch"]
    entries = {(item["path"], item["status"]) for item in result.output["entries"]}
    assert ("tracked.txt", " M") in entries
    assert ("new.txt", "??") in entries
    audit = await db.fetchall("SELECT * FROM audit_log WHERE action='tool:git_status'")
    assert audit and audit[0]["decision"] == "allowed"


async def test_diff_reports_unstaged_and_staged_content(env) -> None:
    (env.root / "tracked.txt").write_text("hello\nworld\n", encoding="utf-8")

    unstaged = await _call(env, "git_diff")
    assert unstaged.success and "+world" in unstaged.output["diff"]
    assert unstaged.output["changed"] is True

    _git(env.root, "add", "tracked.txt")
    staged = await _call(env, "git_diff", {"staged": True, "stat": True})
    assert staged.success and "+world" in staged.output["diff"]
    assert "tracked.txt" in staged.output["diff"]

    clean = await _call(env, "git_diff")
    assert clean.success and clean.output["diff"].strip() == ""


async def test_log_and_show_return_committed_history(env) -> None:
    log = await _call(env, "git_log", {"max_count": 1})
    assert log.success and log.output["count"] == 1
    assert "initial commit" in log.output["text"]

    show = await _call(env, "git_show", {"revision": "HEAD"})
    assert show.success and "initial commit" in show.output["text"]
    assert "+hello" in show.output["text"]

    blob = await _call(env, "git_show", {"revision": "HEAD", "path": "tracked.txt"})
    assert blob.success and "+hello" in blob.output["text"]

    missing = await _call(env, "git_show", {"revision": "no-such-revision"})
    assert not missing.success and "failed" in missing.error


async def test_destructive_flags_and_subcommands_are_rejected(env) -> None:
    flag = await _call(env, "git_show", {"revision": "--output=/tmp/evil"})
    assert not flag.success and flag.decision == "invalid_arguments"

    path_flag = await _call(env, "git_status", {"path": "--force"})
    assert not path_flag.success and path_flag.decision == "invalid_arguments"

    subcommand = await _call(env, "git_status", {"subcommand": "push"})
    assert not subcommand.success and "unknown field" in subcommand.error

    ctx = {"settings": env.settings}
    with pytest.raises(PermissionError, match="state-changing"):
        await git_tools.git_show({"revision": "push"}, ctx)
    with pytest.raises(PermissionError):
        await git_tools.git_show({"revision": "HEAD; rm -rf /"}, ctx)
    with pytest.raises(PermissionError):
        await git_tools.git_status({"path": "-c core.hooksPath=/tmp"}, ctx)


async def test_path_escapes_and_symlinks_are_denied(env) -> None:
    outside = env.root.parent / "outside.txt"
    outside.write_text("not in the workspace\n", encoding="utf-8")
    cases = (
        ("git_status", {"path": str(outside)}),
        ("git_diff", {"path": "../outside.txt"}),
        ("git_log", {"path": str(outside)}),
        ("git_show", {"revision": "HEAD", "path": str(outside)}),
    )
    for name, arguments in cases:
        result = await _call(env, name, arguments)
        assert not result.success
        assert result.decision == "denied"
        assert "outside" in result.error.lower()

    link = env.root / "escape-link"
    link.symlink_to(outside)
    escaped = await _call(env, "git_status", {"path": "escape-link"})
    assert not escaped.success and escaped.decision == "denied"


async def test_output_is_bounded_and_timeout_is_declared(env, monkeypatch) -> None:
    monkeypatch.setattr(git_tools, "MAX_OUTPUT_BYTES", 64)
    result = await _call(env, "git_show", {"revision": "HEAD"})
    assert result.success
    assert result.output["truncated"] is True
    assert len(result.output["text"]) <= 64
    assert result.metadata["timeout_seconds"] == git_tools.TIMEOUT_SECONDS


def test_subprocess_argv_is_allowlisted_and_never_uses_a_shell(env, monkeypatch) -> None:
    seen: dict = {}
    real_popen = subprocess.Popen

    def spy(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["kwargs"] = kwargs
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(git_tools.subprocess, "Popen", spy)
    asyncio.run(git_tools.git_status({"branch": True}, {"settings": env.settings}))

    argv = seen["argv"]
    assert argv[0] == "git"
    assert "--no-pager" in argv and "--no-optional-locks" in argv
    assert argv[argv.index("-C") + 1] == str(env.root)
    assert "status" in argv and "--porcelain=v1" in argv and "--branch" in argv
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["cwd"] == str(env.root)
    assert not ({"push", "commit", "reset", "--force", "--output=/tmp/evil"} & set(argv))
    assert "GIT_DIR" not in seen["kwargs"]["env"]


def test_subprocess_timeout_raises(env, monkeypatch) -> None:
    class TimedOutProcess:
        def __init__(self) -> None:
            self.calls = 0
            self.killed = False

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["git"], timeout)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

    process = TimedOutProcess()
    monkeypatch.setattr(git_tools.subprocess, "Popen", lambda *a, **kw: process)
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(git_tools.git_status({}, {"settings": env.settings}))
    assert process.killed is True
