"""Per-call risk tiers: LOW runs, MEDIUM runs and is audited, HIGH asks.

The tier is a decision about consequence, not about which tool was called: an
edit of a file the agent just read is not the same act as writing a credential,
and ``grep`` is not ``rm -rf``.
"""

from __future__ import annotations

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl
from leo.security.risk_policy import (
    BOUNDED_WRITE_TOOLS,
    READ_TOOLS,
    SELF_MODIFY_TOOLS,
    assess_call,
)
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import edit_tools, memory_tools


def _settings() -> Settings:
    return Settings()


@pytest.mark.parametrize(
    "tool,arguments,expected",
    [
        # read-only shell work: no tap
        ("shell", {"command": "ls -la"}, RiskLevel.LOW),
        ("shell", {"command": "journalctl -u leo-brain -n 50"}, RiskLevel.LOW),
        ("shell", {"command": "systemctl is-active leo-brain"}, RiskLevel.LOW),
        ("shell", {"command": "sqlite3 leo.db 'SELECT count(*) FROM messages'"}, RiskLevel.LOW),
        ("shell", {"command": "curl -s http://127.0.0.1:8765/health"}, RiskLevel.LOW),
        ("shell", {"command": "git log --oneline -5"}, RiskLevel.LOW),
        # bounded writes: run, recorded
        ("shell", {"command": "mkdir -p build"}, RiskLevel.MEDIUM),
        ("shell", {"command": "git commit -m 'x'"}, RiskLevel.MEDIUM),
        ("shell", {"command": "systemctl restart leo-brain"}, RiskLevel.MEDIUM),
        ("shell", {"command": "sqlite3 leo.db 'DELETE FROM kv'"}, RiskLevel.MEDIUM),
        # destructive, arbitrary-code, unreversible: still ask
        ("shell", {"command": "rm -rf build"}, RiskLevel.HIGH),
        ("shell", {"command": "git reset --hard origin/main"}, RiskLevel.HIGH),
        ("shell", {"command": "python -c 'print(1)'"}, RiskLevel.HIGH),
        ("shell", {"command": "rm -rf /"}, RiskLevel.HIGH),
        ("shell", {"command": "echo x > out.txt"}, RiskLevel.HIGH),
        # edits: the workspace is not a minefield, credentials are
        ("write_file", {"path": "leo/core/app.py"}, RiskLevel.MEDIUM),
        ("patch", {"path": "tests/test_selfheal.py"}, RiskLevel.MEDIUM),
        ("write_file", {"path": ".env"}, RiskLevel.HIGH),
        ("write_file", {"path": "/root/.ssh/id_rsa"}, RiskLevel.HIGH),
        ("patch", {"path": "deploy/prod.pem"}, RiskLevel.HIGH),
        ("patch", {"path": ".git/config"}, RiskLevel.HIGH),
        ("write_file", {"path": "leo.yaml"}, RiskLevel.HIGH),
        # outbound: a reply is free; any send is audited but needs no tap
        ("channel_send", {"target": "525083981", "subject": "525083981", "text": "hi"}, RiskLevel.LOW),
        ("channel_send", {"target": "999000111", "subject": "525083981", "text": "hi"}, RiskLevel.MEDIUM),
        # memory: adding is bounded, deleting is not (no undelete on the surface)
        ("memory_add", {"text": "x"}, RiskLevel.MEDIUM),
        ("memory_forget", {"id": "mem_1"}, RiskLevel.HIGH),
        # self-modification stays behind the gate
        ("model_add", {"model": "x"}, RiskLevel.HIGH),
        ("apply_improvement", {"id": "x"}, RiskLevel.HIGH),
    ],
)
def test_tier_per_call(tool: str, arguments: dict, expected: RiskLevel) -> None:
    risk, reason = assess_call(tool, arguments, RiskLevel.LOW, _settings())
    assert risk is expected, f"{tool} {arguments} -> {risk} ({reason})"
    assert reason, "every tier carries a reason for the audit row"


async def _registered_tool_names() -> set[str]:
    from leo.tools import register_all_toolsets

    registry = ToolRegistry()
    register_all_toolsets(registry, {"settings": _settings()})
    return {spec.name for spec in registry.list_specs()}


async def test_every_policy_name_is_a_registered_tool() -> None:
    """A tier written for a tool that does not exist is a claim nobody can test.

    The first version of this module was written from another agent's tool
    vocabulary: nine read-tool names, a ``learning.note`` branch and four skill
    names matched nothing — and ``memory_forget`` was downgraded while its spec,
    docstring and tests still promised an approval.
    """
    names = await _registered_tool_names()
    for label, group in (
        ("READ_TOOLS", READ_TOOLS),
        ("BOUNDED_WRITE_TOOLS", BOUNDED_WRITE_TOOLS),
        ("SELF_MODIFY_TOOLS", SELF_MODIFY_TOOLS),
    ):
        unknown = sorted(name for name in group if name not in names)
        assert unknown == [], f"{label} names tools that are not registered: {unknown}"


async def test_memory_forget_asks_before_it_deletes(tmp_path, monkeypatch) -> None:
    """The policy must not remove the gate the tool's own contract promises."""
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = _settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    memory_tools.register(registry, {"settings": settings})
    executor = ToolExecutorImpl(registry, permissions, events, settings)

    added = await executor.execute(ToolCall(
        name="memory_add", arguments={"content": "a fact worth keeping"}, actor="tester",
    ))
    assert added.success
    row = await db.fetchone("SELECT id FROM memory_items ORDER BY rowid DESC LIMIT 1")
    memory_id = str(row["id"])
    assert memory_id

    with pytest.raises(ApprovalRequired):
        await executor.execute(ToolCall(
            name="memory_forget", arguments={"id": memory_id}, actor="tester",
        ))
    row = await db.fetchone("SELECT id FROM memory_items WHERE id=?", (memory_id,))
    assert row is not None, "the memory was deleted without an approval"


def test_unknown_tool_keeps_its_registered_tier() -> None:
    risk, reason = assess_call("some_future_tool", {}, RiskLevel.MEDIUM, _settings())
    assert risk is RiskLevel.MEDIUM
    assert "registered tier" in reason


async def test_an_in_workspace_edit_raises_no_approval(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = _settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    edit_tools.register(registry, {"settings": settings})
    executor = ToolExecutorImpl(registry, permissions, events, settings)

    result = await executor.execute(ToolCall(
        name="write_file",
        arguments={"path": str(tmp_path / "notes.md"), "content": "hello\n"},
        actor="tester",
    ))
    assert result.success
    assert result.risk is RiskLevel.MEDIUM
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "hello\n"
    assert await db.fetchone("SELECT id FROM approvals WHERE action='tool:write_file'") is None


async def test_a_credential_write_still_asks(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = _settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    edit_tools.register(registry, {"settings": settings})
    executor = ToolExecutorImpl(registry, permissions, events, settings)

    with pytest.raises(ApprovalRequired):
        await executor.execute(ToolCall(
            name="write_file",
            arguments={"path": str(tmp_path / ".env"), "content": "TOKEN=1\n"},
            actor="tester",
        ))
    assert not (tmp_path / ".env").exists()
