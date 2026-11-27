"""Phase 6: workdir anchors tool paths and shell cwd inside allowed roots."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo.config import Settings
from leo.contracts import ToolCall
from leo.tools import builtins as builtins_module
from leo.tools import file_tools
from leo.tools.executor import ToolExecutorImpl


@pytest.fixture
def workspace(tmp_path):
    allowed = tmp_path / "allowed"
    workdir = allowed / "work"
    (workdir / "sub").mkdir(parents=True)
    (workdir / "note.txt").write_text("inside", encoding="utf-8")
    (workdir / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    settings = Settings()
    settings.security.allowed_roots = [allowed]
    ctx = {"settings": settings, "workdir": str(workdir)}
    return SimpleNamespace(
        settings=settings,
        ctx=ctx,
        workdir=workdir,
        allowed=allowed,
        tmp_path=tmp_path,
    )


async def test_relative_reads_resolve_against_workdir(workspace) -> None:
    out = await builtins_module.read_file({"path": "note.txt"}, workspace.ctx)
    assert out["content"] == "inside"
    out = await builtins_module.read_file({"path": "sub/deep.txt"}, workspace.ctx)
    assert out["content"] == "deep"


async def test_path_escape_is_denied(workspace) -> None:
    with pytest.raises(PermissionError):
        await builtins_module.read_file({"path": "../../outside.txt"}, workspace.ctx)
    with pytest.raises(PermissionError):
        await builtins_module.read_file(
            {"path": str(workspace.tmp_path / "outside.txt")}, workspace.ctx
        )
    with pytest.raises(PermissionError):
        await file_tools.file_hash({"path": "../../outside.txt"}, workspace.ctx)


async def test_shell_cwd_defaults_to_workdir(workspace) -> None:
    # ``terminal`` runs an allowlisted executable directly (no shell); the
    # default cwd is the scheduled job's workdir.
    out = await builtins_module.terminal({"command": "pwd"}, workspace.ctx)
    assert out["exit_code"] == 0
    assert out["stdout"].strip() == str(workspace.workdir.resolve())

    sub = await builtins_module.terminal(
        {"command": "pwd", "cwd": "sub"}, workspace.ctx
    )
    assert sub["stdout"].strip() == str((workspace.workdir / "sub").resolve())


async def test_file_tools_resolve_relative_against_workdir(workspace) -> None:
    digest = await file_tools.file_hash({"path": "note.txt"}, workspace.ctx)
    assert digest["path"].endswith("/work/note.txt")


def test_tool_call_and_executor_context_carry_workdir(workspace) -> None:
    call = ToolCall(name="read_file", arguments={}, workdir=str(workspace.workdir))
    executor = object.__new__(ToolExecutorImpl)
    executor.settings = workspace.settings
    executor.services = {}
    executor.permissions = None
    executor.events = None
    ctx = executor._context(call, "task", dry_run=False)
    assert ctx["workdir"] == str(workspace.workdir)
