"""Conversation history tools: real SQLite reads, redaction and bounded export.

Every assertion runs against the shared isolated database from conftest and a
real ToolExecutorImpl. No response is canned: exported files are read back from
disk and their byte counts compared with the tool's reported result.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall, utcnow
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import (
    ToolExecutorImpl,
    ToolRegistry,
    conversation_tools,
    register_all_toolsets,
)

EXPECTED = {"conversation_messages", "conversation_export"}
SECRET = "sk-abcdefghijklmnopqrstuvwxyz"


def _export_files(root: Path) -> list[Path]:
    """Export payloads only; the private flock file is not an export."""
    return [
        path for path in root.iterdir()
        if path.name != conversation_tools._EXPORT_LOCK_NAME and not path.is_symlink()
    ]


def _env(tmp_path: Path) -> SimpleNamespace:
    settings = Settings()
    settings.paths.exports = tmp_path / "exports"
    registry = ToolRegistry()
    names = conversation_tools.register(registry)
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        settings=settings, registry=registry, executor=executor, names=names,
        permissions=permissions, events=events, exports=settings.paths.exports,
    )


async def _run(env: SimpleNamespace, name: str, arguments: dict, **kwargs) -> dict:
    result = await env.executor.execute(ToolCall(name=name, arguments=arguments, **kwargs))
    assert result.success, f"{name} failed: {result.error} (decision={result.decision})"
    assert result.decision in {"allowed", "dry_run"}
    return result.output


async def _seed(conversation_id: str = "conv-1", count: int = 1, text: str = "hello",
                *, updated_at: str | None = None) -> list[int]:
    stamp = updated_at or utcnow()
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at) "
        "VALUES(?,?,?,?,?)",
        (conversation_id, "cli", "history test", stamp, stamp),
    )
    ids: list[int] = []
    for index in range(count):
        cursor = await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, "user" if index % 2 == 0 else "assistant",
             f"{text} #{index}", "{}", stamp),
        )
        ids.append(cursor.lastrowid or 0)
    return ids


def test_register_exposes_two_typed_strict_specs(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert set(env.names) == EXPECTED
    assert len(env.names) == len(set(env.names))
    assert {spec.name for spec in env.registry.list_specs()} == EXPECTED
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert specs["conversation_messages"].risk.value == "low"
    assert specs["conversation_export"].risk.value == "medium"
    for spec in specs.values():
        assert spec.category == "conversation"
        assert spec.permissions
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.input_schema["required"] == []
        assert spec.output_schema["type"] == "object"
        assert spec.resource_field == ""
    assert set(specs["conversation_export"].input_schema["properties"]) == {
        "conversation_id", "format", "redact", "after_id", "limit",
    }
    assert set(specs["conversation_messages"].input_schema["properties"]) == {
        "conversation_id", "after_id", "limit",
    }
    assert specs["conversation_export"].input_schema["properties"]["format"]["enum"] == [
        "md", "json", "txt",
    ]
    assert conversation_tools.register(env.registry) == []  # idempotent


def test_register_all_toolsets_auto_discovers_the_module(tmp_path: Path) -> None:
    registry = ToolRegistry()
    roster = register_all_toolsets(registry, services={}, settings=_env(tmp_path).settings)
    assert "conversation_tools" in roster
    assert EXPECTED <= set(roster["conversation_tools"])
    assert EXPECTED <= {spec.name for spec in registry.list_specs()}


async def test_messages_return_compact_redacted_turns_with_tool_details(tmp_path: Path) -> None:
    env = _env(tmp_path)
    ids = await _seed("conv-1", count=2)
    tool_cursor = await db.execute(
        "INSERT INTO messages(conversation_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
        ("conv-1", "tool",
         json.dumps({"success": False, "error": "unknown field 'limit' on list_files",
                     "tool": "list_files"}),
         json.dumps({"name": "list_files", "tool_call_id": "call_1",
                     "not_executed": True, "reason": "validation failed"}),
         utcnow()),
    )
    tool_id = tool_cursor.lastrowid
    long_text = f"remember api_key={SECRET} " + ("z" * 1000)
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
        ("conv-1", "user", long_text, "{}", utcnow()),
    )

    page = await _run(env, "conversation_messages", {"conversation_id": "conv-1", "limit": 3})
    assert page["count"] == 3 and page["has_more"] is True
    assert [turn["id"] for turn in page["messages"]] == ids + [tool_id]
    assert page["next_cursor"] == str(tool_id)
    assert page["redacted"] is True
    assert page["turns"] == page["messages"]
    assert page["messages"][0]["role"] == "user"
    assert len(page["messages"][0]["text"]) <= 320
    assert page["messages"][0]["truncated"] is False

    tool_turn = page["messages"][2]
    assert "list_files" in tool_turn["tools"]
    assert any("unknown field 'limit'" in error for error in tool_turn["errors"])
    assert any("validation failed" in error for error in tool_turn["errors"])

    second = await _run(env, "conversation_messages", {
        "conversation_id": "conv-1", "after_id": page["next_cursor"], "limit": 3,
    })
    assert second["count"] == 1 and second["has_more"] is False
    assert second["next_cursor"] is None
    secret_turn = second["messages"][0]
    assert SECRET not in secret_turn["text"]
    assert SECRET not in secret_turn["excerpt"]
    assert "[REDACTED]" in secret_turn["text"]
    assert secret_turn["truncated"] is True


async def test_messages_default_to_latest_conversation_and_fail_truthfully(tmp_path: Path) -> None:
    env = _env(tmp_path)
    await _seed("older", text="older note", updated_at="2020-01-01T00:00:00+00:00")
    await _seed("newest", text="newest note", updated_at="2026-01-01T00:00:00+00:00")
    latest = await _run(env, "conversation_messages", {"limit": 5})
    assert latest["conversation_id"] == "newest"
    assert "newest note" in latest["messages"][0]["text"]

    failed = await env.executor.execute(ToolCall(
        name="conversation_messages", arguments={"conversation_id": "missing"}
    ))
    assert not failed.success and "unknown conversation" in (failed.error or "")


async def test_no_conversations_is_an_explicit_failure(tmp_path: Path) -> None:
    env = _env(tmp_path)
    failed = await env.executor.execute(ToolCall(name="conversation_messages", arguments={}))
    assert not failed.success and "no conversations exist" in (failed.error or "")
    assert not list(env.exports.glob("*"))


async def test_export_writes_redacted_markdown_with_real_path_and_size(tmp_path: Path) -> None:
    env = _env(tmp_path)
    await _seed("conv-export", count=2, text=f"do not leak api_key={SECRET}")
    output = await _run(env, "conversation_export", {
        "conversation_id": "conv-export", "format": "md",
    })
    path = Path(output["path"])
    assert path.is_file() and path.parent == env.exports.resolve()
    assert path.suffix == ".md" and path.name == "conversation-conv-export.md"
    body = path.read_text(encoding="utf-8")
    assert "# Conversation export" in body and "do not leak" in body
    assert SECRET not in body and "[REDACTED]" in body
    assert output["bytes"] == path.stat().st_size
    assert output["count"] == 2 and output["redacted"] is True
    assert output["format"] == "md" and output["has_more"] is False


async def test_export_json_and_txt_formats_and_explicit_redact_toggle(tmp_path: Path) -> None:
    env = _env(tmp_path)
    await _seed("conv-export", text=f"secret={SECRET}")
    for fmt, marker in (("json", '"messages"'), ("txt", "redacted: True")):
        output = await _run(env, "conversation_export", {
            "conversation_id": "conv-export", "format": fmt,
        })
        path = Path(output["path"])
        body = path.read_text(encoding="utf-8")
        assert marker in body and output["format"] == fmt
        assert SECRET not in body
        if fmt == "json":
            assert json.loads(body)["count"] == 1

    raw = await _run(env, "conversation_export", {
        "conversation_id": "conv-export", "format": "txt", "redact": False,
    })
    raw_body = Path(raw["path"]).read_text(encoding="utf-8")
    assert raw["redacted"] is False and SECRET in raw_body
    assert "redacted: False" in raw_body


def test_shared_preview_sanitizer_is_idempotent() -> None:
    token = "idempotentToken_1234567890"
    once = conversation_tools.redact_preview_text(
        f"http://127.0.0.1:18765/preview/{token} "
        + json.dumps({"token_id": token})
    )
    twice = conversation_tools.redact_preview_text(once)
    assert once == twice
    assert token not in twice


async def test_export_bytes_never_contain_raw_preview_credentials(tmp_path: Path) -> None:
    """The file gets the same preview sanitizer as an unprivileged API reply."""
    env = _env(tmp_path)
    token = "previewBearer_AbCdEf1234567890"
    url = f"http://127.0.0.1:18765/preview/{token}"
    field = json.dumps({"token_id": token})
    await _seed("conv-preview", text=f"open {url} then {field}")
    output = await _run(env, "conversation_export", {
        "conversation_id": "conv-preview", "format": "json",
    })
    raw = Path(output["path"]).read_bytes()
    text = raw.decode("utf-8")
    assert token not in text
    assert token.encode() not in raw
    assert "http://127.0.0.1:18765/preview/[REDACTED]" in text
    assert "sha256:" in text  # token_id becomes a non-replayable digest

    # The explicit unredacted toggle is still truthful: it is the caller's
    # opt-out, and the response says so.
    raw_output = await _run(env, "conversation_export", {
        "conversation_id": "conv-preview", "format": "txt", "redact": False,
    })
    assert token.encode() in Path(raw_output["path"]).read_bytes()
    assert raw_output["redacted"] is False


async def test_export_uses_stable_bounded_versions_and_private_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    await _seed("conv-stable")
    first = await _run(env, "conversation_export", {"conversation_id": "conv-stable"})
    second = await _run(env, "conversation_export", {"conversation_id": "conv-stable"})
    assert Path(first["path"]).name == "conversation-conv-stable.md"
    assert Path(second["path"]).name == "conversation-conv-stable-2.md"
    assert Path(first["path"]).stat().st_mode & 0o077 == 0  # no group/other access

    monkeypatch.setattr(conversation_tools, "MAX_EXPORT_VERSIONS", 2)
    with pytest.raises(conversation_tools.ExportStorageError, match="version limit"):
        await conversation_tools.write_export(
            conversation_id="conv-stable", ctx={"settings": env.settings}
        )
    assert len(_export_files(env.exports)) == 2


async def test_export_enforces_aggregate_directory_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    await _seed("conv-budget", count=3, text="some transcript text")
    first = await _run(env, "conversation_export", {"conversation_id": "conv-budget"})
    assert Path(first["path"]).is_file()
    monkeypatch.setattr(conversation_tools, "MAX_EXPORT_TOTAL_BYTES", 1)
    before = sorted(path.name for path in env.exports.iterdir())
    with pytest.raises(conversation_tools.ExportStorageError, match="budget"):
        await conversation_tools.write_export(
            conversation_id="conv-budget", ctx={"settings": env.settings}
        )
    assert sorted(path.name for path in env.exports.iterdir()) == before


async def test_export_closes_symlink_swap_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink planted after validation cannot redirect the write."""
    env = _env(tmp_path)
    env.exports.mkdir(parents=True, exist_ok=True)
    await _seed("conv-race")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me", encoding="utf-8")
    original = conversation_tools._target_path

    def swap_after_validation(*args: Any, **kwargs: Any) -> Path:
        target = original(*args, **kwargs)
        target.symlink_to(outside)
        return target

    monkeypatch.setattr(conversation_tools, "_target_path", swap_after_validation)
    with pytest.raises(PermissionError, match="symlink"):
        await conversation_tools.write_export(
            conversation_id="conv-race", ctx={"settings": env.settings}
        )
    assert outside.read_text(encoding="utf-8") == "keep me"
    assert not _export_files(env.exports)


async def test_export_is_bounded_and_cursor_paginates(tmp_path: Path) -> None:
    env = _env(tmp_path)
    ids = await _seed("conv-export", count=20, text="x" * 5000)
    first = await _run(env, "conversation_export", {
        "conversation_id": "conv-export", "limit": 5,
    })
    path = Path(first["path"])
    assert path.stat().st_size <= conversation_tools.MAX_EXPORT_BYTES
    assert path.stat().st_size < 40_000  # each row is clipped before rendering
    assert first["count"] == 5 and first["has_more"] is True
    assert first["next_cursor"] == str(ids[4])
    assert "[truncated]" in path.read_text(encoding="utf-8")

    rest = await _run(env, "conversation_export", {
        "conversation_id": "conv-export", "limit": 20, "after_id": first["next_cursor"],
    })
    assert rest["count"] == 15 and rest["has_more"] is False
    assert f"[{ids[4]}]" not in Path(rest["path"]).read_text(encoding="utf-8")


async def test_export_rejects_invalid_arguments_without_writing_files(tmp_path: Path) -> None:
    env = _env(tmp_path)
    await _seed("conv-export")
    bad_calls = [
        {"conversation_id": "conv-export", "format": "html"},
        {"conversation_id": "conv-export", "limit": 0},
        {"conversation_id": "conv-export", "after_id": -1},
        {"conversation_id": "conv-export", "redact": "maybe"},
        {"conversation_id": "missing"},
    ]
    for arguments in bad_calls:
        result = await env.executor.execute(ToolCall(
            name="conversation_export", arguments=arguments
        ))
        assert not result.success, arguments
        assert result.error, arguments
    assert list(env.exports.glob("*")) == []


async def test_export_never_overwrites_or_follows_a_symlink(tmp_path: Path) -> None:
    env = _env(tmp_path)
    await _seed("conv-export")
    first = await conversation_tools.write_export(
        conversation_id="conv-export", out="report", ctx={"settings": env.settings}
    )
    second = await conversation_tools.write_export(
        conversation_id="conv-export", out="report", ctx={"settings": env.settings}
    )
    assert first["path"] != second["path"]
    assert Path(first["path"]).is_file() and Path(second["path"]).is_file()

    outside = tmp_path / "outside.txt"
    outside.write_text("do not overwrite", encoding="utf-8")
    link = env.exports / "linked.md"
    link.symlink_to(outside)
    with pytest.raises(PermissionError, match="symlink"):
        await conversation_tools.write_export(
            conversation_id="conv-export", out="linked", ctx={"settings": env.settings}
        )
    assert outside.read_text(encoding="utf-8") == "do not overwrite"
    for bad in ("../escape", "/etc/passwd", "a/b", ".", "", "x" * 81):
        with pytest.raises(ValueError):
            await conversation_tools.write_export(
                conversation_id="conv-export", out=bad, ctx={"settings": env.settings}
            )


# --- adversarial re-review: shape-independent credential redaction ---------
def test_preview_sanitizer_covers_json_escapes_bare_fields_and_host_case() -> None:
    token = "shapedToken_AbCdEf1234567890"
    variants: list[Any] = [
        f"http:\\/\\/127.0.0.1:18765\\/preview\\/{token}",
        f"http:\\\\/\\\\/127.0.0.1:18765\\\\/preview\\\\/{token}",  # nested JSON escaping
        f"HTTP://LOCALHOST:18765/PREVIEW/{token}",
        f"token_id: {token}",
        f"{{token_id='{token}'}}",
        json.dumps({"token_id": token}),
        {"token_id": token, "nested": [f"see http:\\/\\/localhost:42\\/preview\\/{token}"]},
    ]
    for value in variants:
        result = conversation_tools.redact_preview_content(value)
        assert token not in json.dumps(result, ensure_ascii=False), value
        assert conversation_tools.redact_preview_content(result) == result  # idempotent


def test_preview_sanitizer_scrubs_a_token_seen_in_another_field() -> None:
    token = "crossFieldToken_AbCdEf1234567890"
    value = {
        "url": f"http://127.0.0.1:18765/preview/{token}",
        "correlation_id": f"corr:{token}",
        "note": token,
    }
    result = conversation_tools.redact_preview_content(value)
    assert token not in json.dumps(result, ensure_ascii=False)
    assert "sha256:" in result["correlation_id"]


def test_live_preview_tokens_are_scrubbed_without_url_syntax() -> None:
    """A bare token in an unrelated field is still a live capability."""
    from leo.tools import preview_tools

    token = "liveOnlyToken_AbCdEf1234567890"
    fake = preview_tools._Preview(
        token=token, path=Path("/tmp/leo-preview-test"), roots=(), actor="test",
        call_id=None, port=1, expires_at=float("inf"), ttl_seconds=30,
        device=0, inode=0, size=0, content_type="text/plain",
    )
    with preview_tools._STATE.lock:
        preview_tools._STATE.previews[token] = fake
    try:
        result = conversation_tools.redact_preview_content({"correlation_id": token})
        assert token not in json.dumps(result, ensure_ascii=False)
        assert result["correlation_id"].startswith("sha256:")
    finally:
        with preview_tools._STATE.lock:
            preview_tools._STATE.previews.pop(token, None)


async def test_export_bytes_redact_escaped_urls_and_bare_token_ids(tmp_path: Path) -> None:
    env = _env(tmp_path)
    token = "escapedToken_AbCdEf1234567890"
    url = f"http:\\/\\/127.0.0.1:18765\\/preview\\/{token}"
    await _seed("conv-shapes", text=f"open {url} then token_id: {token}")
    output = await _run(env, "conversation_export", {
        "conversation_id": "conv-shapes", "format": "json",
    })
    raw = Path(output["path"]).read_bytes()
    assert token.encode() not in raw
    assert b"[REDACTED]" in raw and b"sha256:" in raw

    # A URL that was escaped once by a tool and again by JSON rendering must
    # still lose its token (no bare token_id in this message).
    nested_token = "doubleEscapedToken_AbCdEf1234567890"
    await _seed(
        "conv-shapes-nested",
        text=f"http:\\\\/\\\\/127.0.0.1:18765\\\\/preview\\\\/{nested_token}",
    )
    nested = await _run(env, "conversation_export", {
        "conversation_id": "conv-shapes-nested", "format": "json",
    })
    nested_raw = Path(nested["path"]).read_bytes()
    assert nested_token.encode() not in nested_raw
    assert b"[REDACTED]" in nested_raw


# --- adversarial re-review: atomic budget and verified root fd -------------
async def test_export_budget_is_atomic_across_concurrent_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    await _seed("conv-concurrent", count=2)
    first = await conversation_tools.write_export(
        conversation_id="conv-concurrent", ctx={"settings": env.settings}
    )
    size = Path(first["path"]).stat().st_size
    # Room for exactly one more file, however many writers arrive at once.
    monkeypatch.setattr(conversation_tools, "MAX_EXPORT_TOTAL_BYTES", size * 2 + 10)
    original = conversation_tools._enforce_export_budget

    def slow_check(root: Path, incoming: int) -> None:
        original(root, incoming)
        time.sleep(0.02)  # widen the check/write window a racy writer would exploit

    monkeypatch.setattr(conversation_tools, "_enforce_export_budget", slow_check)
    results = await asyncio.gather(*[
        conversation_tools.write_export(
            conversation_id="conv-concurrent", out=f"race{index}",
            ctx={"settings": env.settings},
        )
        for index in range(6)
    ], return_exceptions=True)
    successes = [item for item in results if isinstance(item, dict)]
    failures = [item for item in results if isinstance(item, BaseException)]
    assert len(successes) == 1, results
    assert len(failures) == 5
    assert all(isinstance(item, conversation_tools.ExportStorageError) for item in failures)
    files = list(env.exports.iterdir())
    assert sum(item.stat().st_size for item in files) <= size * 2 + 10


@pytest.mark.parametrize("as_symlink", [True, False])
async def test_export_refuses_exports_root_swapped_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, as_symlink: bool
) -> None:
    env = _env(tmp_path)
    root = env.exports
    root.mkdir(parents=True, exist_ok=True)
    await _seed("conv-root-swap")
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = tmp_path / "moved-root"
    original = conversation_tools._enforce_export_budget
    swapped = False

    def swap_after_validation(folder: Path, incoming: int) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            folder.rename(moved)
            if as_symlink:
                folder.symlink_to(outside, target_is_directory=True)
            else:
                folder.mkdir()
        original(folder, incoming)

    monkeypatch.setattr(conversation_tools, "_enforce_export_budget", swap_after_validation)
    with pytest.raises(PermissionError, match="root"):
        await conversation_tools.write_export(
            conversation_id="conv-root-swap", ctx={"settings": env.settings}
        )
    assert not _export_files(outside)  # no file escaped the validated root
    assert not _export_files(moved)
    assert swapped


# --- adversarial re-review: raw capabilities and encoded shapes ------------
def test_raw_token_shape_is_scrubbed_without_url_or_field_syntax() -> None:
    """A raw bearer value in correlation_id is still a bearer value."""
    from leo.tools.preview_tools import token_audit_id

    token = "bareRawCapability_AbCdEf1234567890"
    value = {
        "correlation_id": token,
        "source": f"worker:{token}",
        "nested": [{"note": f"incident {token}"}],
    }
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert token not in dumped
    assert result["correlation_id"] == token_audit_id(token)
    assert token_audit_id(token) in result["source"]
    assert conversation_tools.redact_preview_content(result) == result

    # Short ids, timestamps and already-salted digests are not credentials.
    intact = {
        "id": "abc123",
        "created_at": "2026-11-20T12:34:56.123456+00:00",
        "sha": "sha256:" + "a" * 64,
    }
    assert conversation_tools.redact_preview_content(intact) == intact


@pytest.mark.parametrize("shape", [
    "unicode-slash", "percent-slash", "percent-scheme", "ipv6-bracket",
    "ipv6-bare", "escaped-quotes", "backticks", "list-value",
    "dict-value", "map-key", "deep-map",
])
def test_preview_sanitizer_shape_matrix(shape: str) -> None:
    """Encodings, wrappers and nested shapes cannot carry a token through."""
    token = "matrixCapability_AbCdEf1234567890"
    second = "matrixSecondToken_AbCdEf1234567890"
    variants: dict[str, Any] = {
        "unicode-slash": f"http:\\u002f\\u002f127.0.0.1:18765\\u002fpreview\\u002f{token}",
        "percent-slash": f"http://127.0.0.1:18765%2Fpreview%2F{token}",
        "percent-scheme": f"http%3A%2F%2Flocalhost:18765%2fpreview%2f{token}",
        "ipv6-bracket": f"http://[::1]:18765/preview/{token}",
        "ipv6-bare": f"http://::1:18765/preview/{token}",
        "escaped-quotes": f'\\"token_id\\": \\"{token}\\" and {second}',
        "backticks": f"`token_id`: `{token}` plus `{second}`",
        "list-value": {"token_id": [token, f"also:{second}"]},
        "dict-value": {"token_id": {"a": token, "b": [second]}},
        "map-key": {"preview_token_id": token, second: "value"},
        "deep-map": [{"outer": {"token_id": {"inner": [token, {"k": second}]}}}],
    }
    result = conversation_tools.redact_preview_content(variants[shape])
    dumped = json.dumps(result, ensure_ascii=False)
    assert token not in dumped, shape
    assert second not in dumped, shape
    assert conversation_tools.redact_preview_content(result) == result, shape
    if shape in {"unicode-slash", "percent-slash", "percent-scheme",
                 "ipv6-bracket", "ipv6-bare"}:
        assert "[REDACTED]" in dumped, shape
    else:
        assert "sha256:" in dumped, shape


async def test_export_bytes_redact_a_bare_raw_token(tmp_path: Path) -> None:
    env = _env(tmp_path)
    token = "exportBareCapability_AbCdEf1234567890"
    await _seed("conv-bare", text=f"correlation_id was {token}")
    output = await conversation_tools.write_export(
        conversation_id="conv-bare", ctx={"settings": env.settings}
    )
    raw = Path(output["path"]).read_bytes()
    assert token.encode() not in raw
    assert b"sha256:" in raw


@pytest.mark.parametrize("planted", ["root", "parent", "lock"])
async def test_export_refuses_symlinks_planted_before_validation(
    tmp_path: Path, planted: str
) -> None:
    """resolve() must never follow a link planted before validation."""
    env = _env(tmp_path)
    await _seed("conv-prelink")
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.md"
    victim.write_text("keep me", encoding="utf-8")

    if planted == "root":
        link = tmp_path / "linked-exports"
        link.symlink_to(outside, target_is_directory=True)
        env.settings.paths.exports = link
    elif planted == "parent":
        link = tmp_path / "linked-parent"
        link.symlink_to(outside, target_is_directory=True)
        env.settings.paths.exports = link / "exports"
    else:
        env.exports.mkdir(parents=True, exist_ok=True)
        (env.exports / conversation_tools._EXPORT_LOCK_NAME).symlink_to(victim)

    with pytest.raises(PermissionError, match="link|symlink|lock"):
        await conversation_tools.write_export(
            conversation_id="conv-prelink", ctx={"settings": env.settings}
        )
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert not (outside / "conversation-conv-prelink.md").exists()


_MULTIPROCESS_EXPORT_WORKER = r"""
import asyncio, sys, time
from pathlib import Path
from leo import db
from leo.config import Settings
from leo.tools import conversation_tools as ct

db_path, exports, out, budget = sys.argv[1:5]
db.set_db_path(db_path)
settings = Settings()
settings.paths.exports = Path(exports)
ct.MAX_EXPORT_TOTAL_BYTES = int(budget)
original = ct._enforce_export_budget

def slow_check(root, incoming):
    original(root, incoming)
    time.sleep(0.2)  # widen the window a process-local-only lock would leave

ct._enforce_export_budget = slow_check
try:
    result = asyncio.run(ct.write_export(
        conversation_id="conv-mp", out=out, ctx={"settings": settings},
    ))
except Exception as exc:
    print(type(exc).__name__)
else:
    print("OK " + result["path"])
"""


async def test_export_budget_is_shared_across_processes(tmp_path: Path) -> None:
    """flock serializes target selection and budget accounting process-wide."""
    env = _env(tmp_path)
    await _seed("conv-mp", count=2)
    first = await conversation_tools.write_export(
        conversation_id="conv-mp", out="seed", ctx={"settings": env.settings}
    )
    size = Path(first["path"]).stat().st_size
    budget = size * 2 + 100  # room for exactly one more file
    repo = Path(__file__).resolve().parents[1]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _MULTIPROCESS_EXPORT_WORKER,
             str(db.get_db_path()), str(env.exports), f"proc{index}", str(budget)],
            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(repo)},
        )
        for index in range(4)
    ]
    outputs = [process.communicate(timeout=60) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    lines = [stdout.strip().splitlines()[-1] for stdout, _ in outputs]
    assert sum(line.startswith("OK ") for line in lines) == 1, lines
    assert sum("ExportStorageError" in line for line in lines) == 3, lines
    assert sum(item.stat().st_size for item in _export_files(env.exports)) <= budget


# --- sanitizer re-review: digest shape, naked shapes and encoded tokens -----
@pytest.mark.parametrize("spelling", [
    "sha256:rawSecret_AbCdEf1234567890",
    "SHA256:rawSecret_AbCdEf1234567890",
    "sha256:rawallalphapayload",
    "sha256:" + "1" * 15,  # digest shape requires exactly 16 hex chars
])
def test_fake_sha256_prefix_is_scrubbed_on_every_surface(spelling: str) -> None:
    """A prefix is not proof of hashing: the raw capability must not survive."""
    raw = spelling.split(":", 1)[1]
    value = {
        "correlation_id": f"ref {spelling}",
        "token_id": spelling,
        "nested": [{"note": spelling}],
    }
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert spelling not in dumped
    assert raw not in dumped
    assert "sha256:" in dumped  # replaced with a non-replayable audit id
    assert conversation_tools.redact_preview_content(result) == result


def test_valid_audit_digest_shape_survives_in_every_context() -> None:
    from leo.tools.preview_tools import token_audit_id

    digest = token_audit_id("anything")
    benign = {
        "token_id": digest,
        "correlation_id": f"ref {digest.upper()}",
        "nested": [digest],
        "full_hash": "sha256:" + "a" * 64,
    }
    assert conversation_tools.redact_preview_content(benign) == benign


@pytest.mark.parametrize("shape,token", [
    ("all-alpha", "NakedAlphaBearerAbCdEfGh"),
    ("all-digit", "8273640192837465019283"),
    ("all-hex", "0123456789abcdef0123"),
    ("short", "Ab3xDf9Qw2Zv"),
    ("camel-with-digits", "shortToken1234"),
    ("lower-alnum", "abc123def456abc"),
    ("path-segment", "Ab3xDf9Qw2ZvZv9"),
    ("camel-path-segment", "pathShapeToken_123456"),
])
def test_naked_non_preview_token_shapes_are_scrubbed(shape: str, token: str) -> None:
    """No URL or field syntax is needed to recognize a bearer-shaped value."""
    text = f"/srv/data/{token}/report.txt" if shape == "path-segment" else token
    value = {"correlation_id": f"ref {text}", "nested": [text]}
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert token not in dumped, shape
    assert "sha256:" in dumped, shape
    assert conversation_tools.redact_preview_content(result) == result, shape


def test_broadened_heuristic_preserves_benign_ids_timestamps_and_paths() -> None:
    from leo.tools.preview_tools import token_audit_id

    benign = {
        "id": "abc123",
        "short_id": "task_12345678",
        "created_at": "2026-11-20T12:34:56.123456+00:00",
        "compact_timestamp": "20261120123456",
        "epoch_ms": "1700000000000",
        "version": "1.2.34.5678",
        "conversation_id": "conv_a1b2c3d4e5f6a7b8c9d0e1f2",
        "message_id": "msg_0123456789abcdef",
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
        "path": "/Users/macbook/LEO/leo/tools/conversation_tools.py",
        "temp_path": (
            "/private/var/folders/gt/2yv68xpj1hq092w9vxdw25nh0000gn/T/"
            "pytest-of-macbook/pytest-718/exports/"
            "conversation-conv_a1b2c3d4e5f6a7b8c9d0e1f2.md"
        ),
        "digest": token_audit_id("benign"),
    }
    assert conversation_tools.redact_preview_content(benign) == benign


def test_html_entity_encoded_tokens_cannot_be_recovered() -> None:
    token = "entityToken_AbCdEf1234567890"

    def encode(value: str, *, double: bool = False) -> str:
        atoms: list[str] = []
        for index, character in enumerate(value):
            code = ord(character)
            atom = f"&#{code};" if index % 2 == 0 else f"&#x{code:x};"
            atoms.append("&amp;" + atom[1:] if double and index % 7 == 0 else atom)
        return "".join(atoms)

    for spelling in (encode(token), encode(token, double=True)):
        value = {
            "token_id": spelling,
            "correlation_id": f"ref {spelling}",
            "nested": [spelling],
        }
        result = conversation_tools.redact_preview_content(value)
        dumped = json.dumps(result, ensure_ascii=False)
        assert token not in dumped
        assert token not in html.unescape(html.unescape(dumped))
        assert "sha256:" in dumped
        assert conversation_tools.redact_preview_content(result) == result


async def test_export_bytes_never_recover_encoded_or_fake_digest_tokens(
    tmp_path: Path,
) -> None:
    env = _env(tmp_path)
    token = "exportEntityToken_AbCdEf1234567890"
    encoded = "".join(f"&#{ord(character)};" for character in token)
    await _seed("conv-entity", text=f"token_id: {encoded}; also sha256:{token}")
    output = await conversation_tools.write_export(
        conversation_id="conv-entity", ctx={"settings": env.settings}
    )
    body = Path(output["path"]).read_bytes().decode("utf-8")
    assert token not in body
    assert token not in html.unescape(body)
    assert "sha256:" in body


# --- residual review: known prefixes, entity decoding, digest spoofing -------
@pytest.mark.parametrize("prefix", [
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_",
    "sk_live_", "sk_test_", "xoxb_",
])
def test_known_prefix_capabilities_are_scrubbed_by_both_sanitizers(prefix: str) -> None:
    """The snake_case id guard must never exempt a provider credential."""
    # All-lowercase bodies look exactly like benign snake_case identifiers to
    # the generic heuristic; the intrinsic pass must still remove them.
    token = prefix + "abcdef1234567890"
    value = {
        "correlation_id": f"ref {token}",
        "token_id": token,
        "nested": [{"note": token}],
    }
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert token not in dumped, prefix
    assert "sha256:" in dumped, prefix
    assert conversation_tools.redact_preview_content(result) == result, prefix
    assert token not in conversation_tools.redact_text(f"ref {token}"), prefix
    assert token not in conversation_tools.redact_preview_text(
        token.replace("_", "&#95;")
    ), prefix


async def test_export_bytes_scrub_known_prefix_capabilities(tmp_path: Path) -> None:
    """Exports and unprivileged API replies apply the same intrinsic shapes."""
    env = _env(tmp_path)
    token = "sk_live_abcdef1234567890"
    second = "gho_abcdef1234567890"
    await _seed("conv-known-prefix", text=f"correlation_id={token}; token_id={second}")
    output = await conversation_tools.write_export(
        conversation_id="conv-known-prefix", ctx={"settings": env.settings}
    )
    body = Path(output["path"]).read_bytes()
    assert token.encode() not in body and second.encode() not in body
    assert b"sha256:" in body or b"[REDACTED]" in body


def _fully_unescaped(text: str, rounds: int = 16) -> str:
    """What a browser can still recover by repeatedly decoding entities."""
    value = text
    for _ in range(rounds):
        decoded = html.unescape(value)
        if decoded == value:
            break
        value = decoded
    return value


@pytest.mark.parametrize("style", [
    "named-separators", "semicolonless-dec", "semicolonless-hex",
    "nested-double", "nested-triple", "percent-entity", "amp-nosemi",
])
def test_entity_encoding_cannot_recover_a_token(style: str) -> None:
    """Named, semicolon-less and nested HTML references are all decodable."""
    token = "entityResidualToken_AbCdEf1234567890"
    if style == "named-separators":
        spelling = (
            f"http&colon;//127.0.0.1&colon;18765&sol;preview&sol;{token}"
        )
    elif style == "semicolonless-dec":
        spelling = "".join(f"&#{ord(character)}" for character in token)
    elif style == "semicolonless-hex":
        spelling = "".join(f"&#x{ord(character):x}" for character in token)
    elif style in {"nested-double", "nested-triple"}:
        layers = 2 if style == "nested-double" else 3
        spelling = "".join(
            "&" + "amp;" * layers + f"#{ord(character)};" for character in token
        )
    elif style == "amp-nosemi":
        # Browsers also accept ``&amp`` without its semicolon as a nested
        # encoding layer (``&amp#101;`` -> ``&#101;`` -> ``e``).
        spelling = "".join("&amp#" + str(ord(character)) + ";" for character in token)
    else:
        spelling = "".join(
            "%26%23" + str(ord(character)) + "%3B" for character in token
        )
    value = {"token_id": spelling, "correlation_id": f"ref {spelling}"}
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert token not in dumped, style
    assert token not in _fully_unescaped(dumped), style
    assert "sha256:" in dumped, style
    assert conversation_tools.redact_preview_content(result) == result, style


async def test_export_bytes_cannot_recover_named_or_nested_entity_tokens(
    tmp_path: Path,
) -> None:
    """The decoded bytes of an export must not restore a hidden capability."""
    env = _env(tmp_path)
    token = "exportResidualToken_AbCdEf1234567890"
    named = (
        f"http&colon;//127.0.0.1&colon;18765&sol;preview&sol;{token}"
    )
    nested = "".join("&amp;amp;#" + str(ord(char)) + ";" for char in token)
    await _seed("conv-residual-entity", text=f"token_id: {named} {nested}")
    output = await conversation_tools.write_export(
        conversation_id="conv-residual-entity", ctx={"settings": env.settings}
    )
    body = Path(output["path"]).read_bytes().decode("utf-8")
    assert token not in body
    assert token not in _fully_unescaped(body)


def test_invented_sixteen_hex_digest_is_scrubbed_but_real_digest_survives() -> None:
    """A digest shape is not proof that the value came from this app."""
    from leo.tools.preview_tools import token_audit_id

    spoof = "sha256:0123456789abcdef"
    value = {
        "correlation_id": f"ref {spoof}",
        "token_id": spoof,
        "nested": [spoof],
    }
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert spoof not in dumped and "0123456789abcdef" not in dumped
    assert "sha256:" in dumped
    assert conversation_tools.redact_preview_content(result) == result

    genuine = token_audit_id("genuine-token-for-digest-registry")
    authentic = {"token_id": genuine, "nested": [genuine.upper()]}
    assert conversation_tools.redact_preview_content(authentic) == authentic


# --- adversarial follow-up: glued prefixes, glued digests, deeper entities ----
@pytest.mark.parametrize("token", [
    "ref_sk_live_abcdef1234567890",
    "x_ghp_abcdef1234567890",
    "ref_ghp_abcdef1234567890",
    "prefixgithub_pat_abcdef1234567890",
])
def test_embedded_known_prefix_capabilities_cannot_be_glued_past_redaction(
    token: str,
) -> None:
    """A known provider prefix is a credential at any position in a word."""
    value = {
        "token_id": token,
        "correlation_id": f"ref {token}; done",
        "nested": [{"note": token}],
    }
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert token not in dumped
    assert "sha256:" in dumped or "[REDACTED]" in dumped
    assert conversation_tools.redact_preview_content(result) == result
    # Both durable-export and text-only sanitizers apply the same intrinsic
    # shapes; a word character in front of the prefix is not an exemption.
    assert token not in conversation_tools.redact_text(f"ref {token}")
    assert token not in conversation_tools.redact_preview_text(f"ref {token}")
    # Entity-encoded spellings are decoded before matching, then scrubbed in
    # their original spelling.
    encoded = token.replace("_", "&amp;" * 18 + "#95;")
    encoded_result = conversation_tools.redact_preview_text(f"ref {encoded}")
    assert token not in encoded_result
    assert token not in _fully_unescaped(encoded_result, rounds=24)


@pytest.mark.parametrize("fake", [
    "refsha256:rawSecret_AbCdEf1234567890",
    "xxsha256:rawallalphapayload",
    "refsha256:0123456789abcdef",
    "refsha256:" + "1" * 15,
])
def test_sha256_marker_glued_after_a_word_is_scrubbed(fake: str) -> None:
    """A ``sha256:`` marker is not proof of hashing, glued or not."""
    raw = fake.split("sha256:", 1)[1]
    value = {
        "token_id": fake,
        "correlation_id": f"ref {fake}; done",
        "nested": [fake],
    }
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert fake not in dumped and raw not in dumped
    assert "sha256:" in dumped  # replaced with a non-replayable audit id
    assert conversation_tools.redact_preview_content(result) == result
    assert fake not in conversation_tools.redact_preview_text(f"ref {fake}")
    assert fake not in conversation_tools.redact_text(f"ref {fake}")


def test_glued_genuine_digest_and_full_hash_stay_stable() -> None:
    from leo.tools.preview_tools import token_audit_id

    genuine = token_audit_id("genuine glued digest")
    spelled = "ref" + genuine
    full_hash = "refsha256:" + "a" * 64
    value = {"correlation_id": spelled, "nested": [full_hash]}
    assert conversation_tools.redact_preview_content(value) == value


def test_long_backslash_runs_cannot_stall_redaction() -> None:
    """Escape-run patterns must stay linear.

    ``\\\\+u`` rescanned a 100 KB backslash run from every offset, which took
    ~70 s of CPU on the request path and froze the whole event loop (chat
    loads took minutes). Source files and tool output contain such runs, so
    every escape pattern is bounded and this must stay fast.
    """
    for size in (10_000, 100_000):
        started = time.perf_counter()
        out = conversation_tools.redact_preview_content("\\" * size + "x")
        elapsed = time.perf_counter() - started
        assert isinstance(out, str)
        assert elapsed < 5.0, f"redacting {size} backslashes took {elapsed:.1f}s"


def test_wide_candidate_sets_do_not_recompile_per_string() -> None:
    """Many digest-shaped candidates must not cost a variant pass per string.

    A tool-heavy conversation yielded 103 candidates; running each candidate's
    per-character variant regex over every string burned seconds per load.
    Plain spellings go through one combined alternation, and the variant
    engine only runs when a decoded form is actually present.
    """
    digests = [f"sha256:{index:016x}" for index in range(120)]
    body = "\n".join(f"row {i} ok " + "x" * 40 for i in range(300))
    value = {"rows": [body + " " + digest for digest in digests]}
    started = time.perf_counter()
    result = conversation_tools.redact_preview_content(value)
    elapsed = time.perf_counter() - started
    dumped = json.dumps(result)
    for digest in digests:
        assert digest not in dumped
    assert elapsed < 5.0, f"wide candidate redaction took {elapsed:.1f}s"


@pytest.mark.parametrize("layers", [13, 20, 48])
def test_entity_nesting_beyond_any_fixed_bound_is_scrubbed(layers: int) -> None:
    """A browser decoder has no 12-layer limit; neither may our matcher."""
    token = "deepEntityToken_AbCdEf1234567890"
    nested = "".join("&" + "amp;" * layers + f"#{ord(char)};" for char in token)
    value = {"token_id": nested, "correlation_id": f"ref {nested}"}
    result = conversation_tools.redact_preview_content(value)
    dumped = json.dumps(result, ensure_ascii=False)
    assert token not in dumped
    assert token not in _fully_unescaped(dumped, rounds=layers + 8)
    assert conversation_tools.redact_preview_content(result) == result


def test_deeply_nested_named_entities_cannot_recover_a_token() -> None:
    token = "deepNamedEntityToken_AbCdEf1234567890"
    spelling = token.replace("_", "&" + "amp;" * 24 + "lowbar;")
    result = conversation_tools.redact_preview_text(f"token_id: {spelling}")
    assert token not in result
    assert token not in _fully_unescaped(result, rounds=32)


async def test_export_bytes_cannot_recover_deeply_nested_entities(
    tmp_path: Path,
) -> None:
    env = _env(tmp_path)
    token = "deepExportEntityToken_AbCdEf1234567890"
    nested = "".join("&" + "amp;" * 21 + f"#{ord(char)};" for char in token)
    await _seed("conv-deep-entity", text=f"token_id: {nested}")
    output = await conversation_tools.write_export(
        conversation_id="conv-deep-entity", ctx={"settings": env.settings}
    )
    body = Path(output["path"]).read_text(encoding="utf-8")
    assert token not in body
    assert token not in _fully_unescaped(body, rounds=32)
