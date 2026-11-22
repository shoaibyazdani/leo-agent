"""Edit tools: a real write and an exact-snippet patch, with a diff and a hash.

Before these existed the only way to change a file was a shell command, so an
edit could not be reviewed, could not report what changed, and died with the
command. These tests pin the behaviour that makes an edit trustworthy.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, edit_tools

TOOL_NAMES = ["write_file", "patch"]


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.security.workspace_only = True
    # The point of these tests is the edit behaviour, not the approval gate.
    settings.security.high_risk_requires_approval = False
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = edit_tools.register(registry)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events,
        permissions=permissions, registry=registry, executor=executor, names=names,
    )


async def _run(env, name: str, arguments: dict | None = None, **kwargs):
    call = ToolCall(name=name, arguments=arguments or {}, **kwargs)
    return await env.executor.execute(call)


async def test_registration_is_idempotent_and_high_risk(env) -> None:
    assert env.names == TOOL_NAMES
    assert edit_tools.register(env.registry, services={}) == []
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert set(specs) == set(TOOL_NAMES)
    for name in TOOL_NAMES:
        assert specs[name].risk == RiskLevel.HIGH, name
        assert specs[name].supports_dry_run is True, name
        assert specs[name].resource_field == "path", name


async def test_write_creates_a_file_and_reports_its_hash(env) -> None:
    target = env.tmp_path / "notes.txt"
    result = await _run(env, "write_file", {"path": str(target), "content": "hello\n"})

    assert result.success, result.error
    output = result.output
    assert target.read_text() == "hello\n"
    assert output["created"] is True
    assert output["bytes"] == 6
    assert output["sha256"] == hashlib.sha256(b"hello\n").hexdigest()


async def test_rewrite_keeps_the_previous_hash_so_a_change_is_auditable(env) -> None:
    target = env.tmp_path / "notes.txt"
    await _run(env, "write_file", {"path": str(target), "content": "one\n"})
    first = hashlib.sha256(b"one\n").hexdigest()

    result = await _run(env, "write_file", {"path": str(target), "content": "two\n"})

    assert result.success, result.error
    assert result.output["created"] is False
    assert result.output["previous_sha256"] == first
    assert target.read_text() == "two\n"


async def test_overwrite_false_refuses_without_touching_the_file(env) -> None:
    target = env.tmp_path / "keep.txt"
    target.write_text("original")

    result = await _run(env, "write_file",
                        {"path": str(target), "content": "replacement", "overwrite": False})

    assert result.success is False
    assert target.read_text() == "original", "a refused write must not change content"


async def test_write_outside_the_allowed_roots_is_denied(env) -> None:
    outside = env.tmp_path.parent / "escape.txt"
    if outside.exists():
        outside.unlink()

    result = await _run(env, "write_file", {"path": str(outside), "content": "nope"})

    assert result.success is False
    assert not outside.exists()


async def test_dry_run_reports_without_writing(env) -> None:
    target = env.tmp_path / "dry.txt"

    result = await _run(env, "write_file",
                        {"path": str(target), "content": "x"}, dry_run=True)

    assert result.success, result.error
    assert result.output["dry_run"] is True
    assert not target.exists()


async def test_patch_replaces_exactly_once_and_shows_the_diff(env) -> None:
    target = env.tmp_path / "code.py"
    target.write_text("def f():\n    return 1\n")

    result = await _run(env, "patch", {
        "path": str(target),
        "old_string": "    return 1\n",
        "new_string": "    return 2\n",
    })

    assert result.success, result.error
    assert target.read_text() == "def f():\n    return 2\n"
    assert result.output["replacements"] == 1
    assert "-    return 1" in result.output["diff"]
    assert "+    return 2" in result.output["diff"]


async def test_patch_refuses_a_missing_snippet(env) -> None:
    target = env.tmp_path / "code.py"
    target.write_text("value = 1\n")

    result = await _run(env, "patch", {
        "path": str(target), "old_string": "value = 9", "new_string": "value = 2",
    })

    assert result.success is False
    assert "does not occur" in str(result.error)
    assert target.read_text() == "value = 1\n"


async def test_patch_refuses_an_ambiguous_snippet(env) -> None:
    target = env.tmp_path / "code.py"
    target.write_text("x = 1\nx = 1\n")

    result = await _run(env, "patch", {
        "path": str(target), "old_string": "x = 1", "new_string": "x = 2",
    })

    assert result.success is False
    assert "occurs 2 times" in str(result.error)
    assert target.read_text() == "x = 1\nx = 1\n"


async def test_replace_all_takes_every_occurrence(env) -> None:
    target = env.tmp_path / "code.py"
    target.write_text("x = 1\nx = 1\n")

    result = await _run(env, "patch", {
        "path": str(target), "old_string": "x = 1", "new_string": "x = 2",
        "replace_all": True,
    })

    assert result.success, result.error
    assert result.output["replacements"] == 2
    assert target.read_text() == "x = 2\nx = 2\n"


async def test_patch_deletes_when_new_string_is_empty(env) -> None:
    target = env.tmp_path / "code.py"
    target.write_text("keep\ndrop me\nkeep\n")

    result = await _run(env, "patch", {
        "path": str(target), "old_string": "drop me\n", "new_string": "",
    })

    assert result.success, result.error
    assert target.read_text() == "keep\nkeep\n"


async def test_patch_refuses_an_identical_replacement(env) -> None:
    target = env.tmp_path / "code.py"
    target.write_text("same\n")

    result = await _run(env, "patch", {
        "path": str(target), "old_string": "same", "new_string": "same",
    })

    assert result.success is False
    assert "identical" in str(result.error)


async def test_patch_dry_run_does_not_write(env) -> None:
    target = env.tmp_path / "code.py"
    target.write_text("a\n")

    result = await _run(env, "patch", {
        "path": str(target), "old_string": "a", "new_string": "b",
    }, dry_run=True)

    assert result.success, result.error
    assert result.output["dry_run"] is True
    assert target.read_text() == "a\n"
