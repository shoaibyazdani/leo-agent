"""File utility tools: real files, real effects, executor-mediated denials."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall, RiskLevel
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, file_tools

TOOL_NAMES = ["file_copy", "file_move", "file_hash", "file_diff", "dir_tree"]


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = file_tools.register(registry)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events,
        permissions=permissions, registry=registry, executor=executor,
        names=names,
    )


async def _run(env, name: str, arguments: dict | None = None, **kwargs):
    call = ToolCall(name=name, arguments=arguments or {}, **kwargs)
    return await env.executor.execute(call)


async def test_registration_schemas_and_validation(env) -> None:
    assert env.names == TOOL_NAMES
    # Idempotent registration keeps the first handler and reports no new names.
    assert file_tools.register(env.registry, services={}) == []
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert set(specs) == set(TOOL_NAMES)
    expected_risk = {
        "file_copy": RiskLevel.MEDIUM,
        "file_move": RiskLevel.MEDIUM,
        "file_hash": RiskLevel.LOW,
        "file_diff": RiskLevel.LOW,
        "dir_tree": RiskLevel.LOW,
    }
    for name, spec in specs.items():
        assert spec.category == "filesystem"
        assert spec.risk is expected_risk[name]
        assert spec.permissions
        assert spec.resource_field in spec.input_schema["properties"]
        assert spec.input_schema["additionalProperties"] is False
        assert env.registry.get(name).handler is not None
    assert specs["file_copy"].permissions == ["file:read", "file:write"]
    assert specs["file_hash"].permissions == ["file:read"]

    errors = env.registry.validate("file_copy", {"source": "a"})
    assert any("destination" in error for error in errors)
    assert env.registry.validate("file_diff", {
        "left": "a", "right": "b", "bogus": True,
    })
    assert env.registry.validate("file_hash", {"path": "a", "algorithm": "nope"})


async def test_file_copy_real_effect_no_overwrite_and_audit(env) -> None:
    source = env.tmp_path / "source.txt"
    source.write_text("alpha\n")
    destination = env.tmp_path / "nested" / "copy.txt"

    result = await _run(env, "file_copy", {
        "source": str(source), "destination": str(destination),
    })
    assert result.success, result.error
    assert destination.read_text() == "alpha\n"
    assert source.read_text() == "alpha\n"
    assert result.output["bytes"] == 6
    rows = await db.fetchall("SELECT * FROM audit_log WHERE action='tool:file_copy'")
    assert rows and rows[-1]["decision"] == "allowed"

    source.write_text("beta\n")
    refused = await _run(env, "file_copy", {
        "source": str(source), "destination": str(destination),
    })
    assert not refused.success and "already exists" in refused.error
    assert destination.read_text() == "alpha\n"

    replaced = await _run(env, "file_copy", {
        "source": str(source), "destination": str(destination), "overwrite": True,
    })
    assert replaced.success and destination.read_text() == "beta\n"


async def test_file_move_real_effect(env) -> None:
    source = env.tmp_path / "to-move.txt"
    source.write_text("moved content")
    destination = env.tmp_path / "deep" / "moved.txt"

    result = await _run(env, "file_move", {
        "source": str(source), "destination": str(destination),
    })
    assert result.success, result.error
    assert not source.exists()
    assert destination.read_text() == "moved content"
    assert result.output["bytes"] == len("moved content")


async def test_file_hash_computes_real_digests(env) -> None:
    path = env.tmp_path / "data.bin"
    payload = b"hello world" * 100
    path.write_bytes(payload)

    result = await _run(env, "file_hash", {"path": str(path)})
    assert result.success, result.error
    assert result.output["hash"] == hashlib.sha256(payload).hexdigest()
    assert result.output["algorithm"] == "sha256"
    assert result.output["size"] == len(payload)

    other = await _run(env, "file_hash", {"path": str(path), "algorithm": "sha512"})
    assert other.success and other.output["hash"] == hashlib.sha512(payload).hexdigest()


async def test_file_diff_reports_real_changes(env) -> None:
    left = env.tmp_path / "old.txt"
    right = env.tmp_path / "new.txt"
    left.write_text("a\nb\nc\n")
    right.write_text("a\nB\nc\n")

    changed = await _run(env, "file_diff", {"left": str(left), "right": str(right)})
    assert changed.success, changed.error
    assert changed.output["changed"] is True
    assert "-b" in changed.output["unified"] and "+B" in changed.output["unified"]
    assert "---" in changed.output["unified"] and "+++" in changed.output["unified"]

    same = await _run(env, "file_diff", {"left": str(left), "right": str(left)})
    assert same.success and same.output["changed"] is False
    assert same.output["unified"] == ""

    invalid = await _run(env, "file_diff", {"left": str(left)})
    assert not invalid.success and invalid.decision == "invalid_arguments"


async def test_dir_tree_real_entries_and_depth_limit(env) -> None:
    root = env.tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("a")
    (root / "sub" / "b.txt").write_text("b")

    result = await _run(env, "dir_tree", {"path": str(root)})
    assert result.success, result.error
    paths = {entry["path"] for entry in result.output["entries"]}
    assert {"a.txt", "sub", "sub/b.txt"} <= paths
    assert result.output["count"] == 3
    assert result.output["truncated"] is False
    assert "  b.txt" in result.output["text"]

    shallow = await _run(env, "dir_tree", {"path": str(root), "max_depth": 1})
    shallow_paths = {entry["path"] for entry in shallow.output["entries"]}
    assert "sub" in shallow_paths and "sub/b.txt" not in shallow_paths


async def test_path_traversal_and_symlink_escapes_are_denied(
    env, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.txt"
    secret.write_text("secret")
    inside = env.tmp_path / "inside.txt"
    inside.write_text("ok")

    direct = await _run(env, "file_copy", {
        "source": str(secret), "destination": str(env.tmp_path / "copied.txt"),
    })
    assert not direct.success and direct.decision == "denied"
    assert "outside allowed roots" in direct.error

    destination = outside / "new.txt"
    write_escape = await _run(env, "file_copy", {
        "source": str(inside), "destination": str(destination),
    })
    assert not write_escape.success and "outside allowed roots" in write_escape.error
    assert not destination.exists()

    dotdot = env.tmp_path / ".." / outside.name / "secret.txt"
    escaped = await _run(env, "file_hash", {"path": str(dotdot)})
    assert not escaped.success and "outside allowed roots" in escaped.error

    link = env.tmp_path / "secret-link.txt"
    link.symlink_to(secret)
    symlink_read = await _run(env, "file_hash", {"path": str(link)})
    assert not symlink_read.success and "outside allowed roots" in symlink_read.error

    target = outside / "target.txt"
    link_out = env.tmp_path / "link-out.txt"
    link_out.symlink_to(target)
    symlink_write = await _run(env, "file_copy", {
        "source": str(inside), "destination": str(link_out),
    })
    assert not symlink_write.success and "outside allowed roots" in symlink_write.error
    assert not target.exists()

    (env.tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    tree = await _run(env, "dir_tree", {"path": str(env.tmp_path)})
    assert tree.success, tree.error
    paths = {entry["path"] for entry in tree.output["entries"]}
    assert not any("escape" in item or "secret.txt" in item for item in paths)


async def test_overwriting_hard_link_is_refused(env) -> None:
    inside = env.tmp_path / "inside.txt"
    inside.write_text("ok")
    original = env.tmp_path / "original.txt"
    original.write_text("keep")
    linked = env.tmp_path / "linked.txt"
    os.link(original, linked)

    result = await _run(env, "file_copy", {
        "source": str(inside), "destination": str(linked), "overwrite": True,
    })
    assert not result.success and "hard-linked" in result.error
    assert original.read_text() == "keep"
    assert inside.read_text() == "ok"


async def test_dry_run_does_not_touch_files(env) -> None:
    source = env.tmp_path / "source.txt"
    source.write_text("data")
    destination = env.tmp_path / "dry.txt"

    result = await _run(env, "file_copy", {
        "source": str(source), "destination": str(destination),
    }, dry_run=True)
    assert result.success and result.output is None
    assert result.metadata["dry_run"] is True
    assert not destination.exists()
    assert source.read_text() == "data"
