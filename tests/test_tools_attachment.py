"""Attachment toolset tests: real store persistence, safe reads and deletion.

Every tool is executed through ``ToolExecutorImpl`` on the isolated temp DB from
the shared ``isolated_db`` fixture, with a real ``AttachmentStore`` rooted at a
temp workspace. No attachment behaviour is mocked; the only stub is a counting
proxy that proves an injected store is honoured.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall
from leo.events import EventBus
from leo.integrations.attachments import MAX_BYTES, AttachmentStore
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, attachment_tools

EXPECTED = {"attachment_list", "attachment_info", "attachment_add", "attachment_remove"}
PAYLOAD = b"Release notes: rotate the staging token before deploy.\n"
ENCODED = base64.b64encode(PAYLOAD).decode("ascii")


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, inject: bool = True,
         injected: object | None = None) -> SimpleNamespace:
    settings = Settings()
    settings.paths.workspaces = tmp_path / "workspaces"  # absolute -> real temp root
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = attachment_tools.register(registry)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    store = AttachmentStore(settings)
    if inject:
        original_context = ToolExecutorImpl._context
        services = {"attachments": injected if injected is not None else store}

        def with_services(self, call, task_key, dry_run):
            return {**original_context(self, call, task_key, dry_run), "services": services}

        monkeypatch.setattr(ToolExecutorImpl, "_context", with_services)
    return SimpleNamespace(
        registry=registry, executor=executor, names=names, store=store,
        settings=settings, events=events, permissions=permissions,
    )


async def _run(env: SimpleNamespace, name: str, arguments: dict, **kwargs) -> dict:
    result = await env.executor.execute(ToolCall(name=name, arguments=arguments, **kwargs))
    assert result.success, f"{name} failed: {result.error} (decision={result.decision})"
    assert result.decision in {"allowed", "dry_run"}
    return result.output


async def test_register_exposes_four_typed_specs(monkeypatch, tmp_path) -> None:
    env = _env(monkeypatch, tmp_path)
    assert set(env.names) == EXPECTED
    assert len(env.names) == len(set(env.names))
    assert {spec.name for spec in env.registry.list_specs()} == EXPECTED
    for spec in env.registry.list_specs():
        assert spec.category == "attachments"
        assert spec.risk.value in {"low", "medium"}
        assert spec.permissions and all(p.startswith("attachment:") for p in spec.permissions)
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert set(spec.input_schema["properties"])
        assert spec.resource_field == ("path" if spec.name == "attachment_add" else "")
        assert spec.input_schema["required"] == (
            ["id"] if spec.name in {"attachment_info", "attachment_remove"} else
            ["name", "mime"] if spec.name == "attachment_add" else []
        )
    assert attachment_tools.register(env.registry) == []  # idempotent


async def test_add_persists_real_row_and_file_then_list_info_remove(monkeypatch, tmp_path) -> None:
    env = _env(monkeypatch, tmp_path)
    stored = await _run(env, "attachment_add", {
        "name": "notes.txt", "mime": "text/plain", "data_base64": ENCODED,
        "conversation_id": "conv-1",
    })
    assert stored["name"] == "notes.txt"
    assert stored["mime"] == "text/plain"
    assert stored["size"] == len(PAYLOAD)
    assert "path" not in stored and "text_preview" not in stored

    row = await db.fetchone("SELECT * FROM attachments WHERE id=?", (stored["id"],))
    assert row is not None
    assert row["sha256"] == stored["sha256"]
    assert row["conversation_id"] == "conv-1"
    stored_path = Path(row["path"])
    assert stored_path.is_relative_to(env.store.root)
    assert stored_path.is_file() and stored_path.read_bytes() == PAYLOAD

    listed = await _run(env, "attachment_list", {})
    assert [a["id"] for a in listed["attachments"]] == [stored["id"]]
    assert listed["attachments"][0]["available"] is True
    assert [a["id"] for a in (await _run(env, "attachment_list", {"conversation_id": "conv-1"}))["attachments"]] == [stored["id"]]
    assert (await _run(env, "attachment_list", {"conversation_id": "other"}))["attachments"] == []

    info = await _run(env, "attachment_info", {"id": stored["id"]})
    assert info["verified"] is True and info["sha256"] == stored["sha256"]
    assert "Release notes" in info["text_preview"]

    removed = await _run(env, "attachment_remove", {"id": stored["id"]})
    assert removed["deleted"] is True and removed["file_removed"] is True
    assert await db.fetchone("SELECT id FROM attachments WHERE id=?", (stored["id"],)) is None
    assert not stored_path.exists()
    assert (await _run(env, "attachment_list", {}))["attachments"] == []

    again = await _run(env, "attachment_remove", {"id": stored["id"]})
    assert again == {"id": stored["id"], "deleted": False, "found": False}
    missing = await env.executor.execute(
        ToolCall(name="attachment_info", arguments={"id": stored["id"]})
    )
    assert missing.success is False and "unknown attachment" in (missing.error or "")


async def test_add_from_local_path_and_store_mime_size_rules(monkeypatch, tmp_path) -> None:
    env = _env(monkeypatch, tmp_path)
    env.store.root.mkdir(parents=True, exist_ok=True)
    source = env.store.root / "source.txt"
    source.write_bytes(PAYLOAD)
    stored = await _run(env, "attachment_add", {
        "name": "from-disk.txt", "mime": "text/plain", "path": str(source),
    })
    assert stored["size"] == len(PAYLOAD)
    assert await db.fetchone("SELECT id FROM attachments WHERE id=?", (stored["id"],)) is not None

    bad_cases = [
        ({"name": "x", "mime": "application/x-evil", "data_base64": ENCODED}, "unsupported attachment type"),
        ({"name": "x", "mime": "text/plain"}, "base64"),
        ({"name": "x", "mime": "text/plain", "data_base64": "!!!not-base64!!!"}, "invalid base64"),
        ({"name": "x", "mime": "text/plain", "data_base64": "A" * (int(MAX_BYTES * 4 / 3) + 32)}, "too large"),
        ({"name": "x", "mime": "text/plain", "data_base64": ENCODED, "path": str(source)}, "either"),
        ({"name": "x", "mime": "image/png", "data_base64": ENCODED}, "unsupported file content"),
        ({"mime": "text/plain", "data_base64": ENCODED}, "name"),
    ]
    for arguments, fragment in bad_cases:
        result = await env.executor.execute(ToolCall(name="attachment_add", arguments=arguments))
        assert result.success is False, (arguments.keys(), result.output)
        assert fragment in (result.error or ""), (arguments.keys(), result.error)
    rows = await db.fetchall("SELECT id FROM attachments")
    assert [row["id"] for row in rows] == [stored["id"]]  # failed adds wrote nothing



async def test_path_traversal_is_denied(monkeypatch, tmp_path) -> None:
    env = _env(monkeypatch, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("SENSITIVE")
    arguments = {"name": "outside.txt", "mime": "text/plain", "path": str(outside)}
    result = await env.executor.execute(ToolCall(name="attachment_add", arguments=arguments))
    assert result.success is False and result.decision == "denied"
    assert "outside allowed roots" in (result.error or "")
    # Handler-level guard also holds when callers bypass the executor's resource check.
    with pytest.raises(PermissionError, match="outside allowed workspace"):
        await attachment_tools.attachment_add(
            arguments, {"settings": env.settings, "services": {"attachments": env.store}}
        )
    relative = await env.executor.execute(ToolCall(name="attachment_add", arguments={
        "name": "passwd", "mime": "text/plain", "path": "../../etc/passwd",
    }))
    assert relative.success is False
    assert await db.fetchall("SELECT id FROM attachments") == []


async def test_symlink_swap_is_refused_at_read_time(monkeypatch, tmp_path) -> None:
    env = _env(monkeypatch, tmp_path)
    stored = await _run(env, "attachment_add", {
        "name": "notes.txt", "mime": "text/plain", "data_base64": ENCODED,
    })
    row = await db.fetchone("SELECT path FROM attachments WHERE id=?", (stored["id"],))
    path = Path(row["path"])
    payload = path.read_bytes()

    # Swap the recorded file for a symlink to a sensitive file OUTSIDE the store root.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET OUTSIDE")
    path.unlink()
    path.symlink_to(secret)
    denied = await env.executor.execute(
        ToolCall(name="attachment_info", arguments={"id": stored["id"]})
    )
    assert denied.success is False
    assert "TOP SECRET" not in str(denied.output) and "TOP SECRET" not in (denied.error or "")
    item = next(a for a in (await _run(env, "attachment_list", {}))["attachments"]
                if a["id"] == stored["id"])
    assert item["available"] is False

    # A symlink back INSIDE the workspace is also refused (O_NOFOLLOW on the
    # final component), so only the original regular file is ever accepted.
    inside = env.store.root / "other.txt"
    inside.write_text("INSIDE OTHER")
    path.unlink()
    path.symlink_to(inside)
    denied_inside = await env.executor.execute(
        ToolCall(name="attachment_info", arguments={"id": stored["id"]})
    )
    assert denied_inside.success is False
    assert "INSIDE OTHER" not in str(denied_inside.output)

    path.unlink()
    path.write_bytes(payload)
    restored = await _run(env, "attachment_info", {"id": stored["id"]})
    assert restored["verified"] is True
    removed = await _run(env, "attachment_remove", {"id": stored["id"]})
    assert removed["deleted"] is True


async def test_lazy_store_fallback_and_missing_service(monkeypatch, tmp_path) -> None:
    # No ctx service: the tool must construct a real store over the configured DB.
    lazy_env = _env(monkeypatch, tmp_path, inject=False)
    stored = await _run(lazy_env, "attachment_add", {
        "name": "lazy.txt", "mime": "text/plain", "data_base64": ENCODED,
    })
    assert await db.fetchone("SELECT id FROM attachments WHERE id=?", (stored["id"],)) is not None

    missing_env = _env(monkeypatch, tmp_path, inject=False)
    monkeypatch.setattr(db, "database_path_configured", lambda: False)
    failed = await missing_env.executor.execute(ToolCall(name="attachment_add", arguments={
        "name": "x", "mime": "text/plain", "data_base64": ENCODED,
    }))
    assert failed.success is False and "attachment service unavailable" in (failed.error or "")


async def test_injected_store_is_honoured(monkeypatch, tmp_path) -> None:
    injected_settings = Settings()
    injected_settings.paths.workspaces = tmp_path / "workspaces"
    real = AttachmentStore(injected_settings)

    class CountingStore:
        def __init__(self, inner: AttachmentStore) -> None:
            self._inner = inner
            self.root = inner.root
            self.calls: list[str] = []

        async def save(self, *args, **kwargs):
            self.calls.append("save")
            return await self._inner.save(*args, **kwargs)

        async def get(self, *args, **kwargs):
            self.calls.append("get")
            return await self._inner.get(*args, **kwargs)

        def public(self, record):
            return self._inner.public(record)

    proxy = CountingStore(real)
    env = _env(monkeypatch, tmp_path, injected=proxy)
    stored = await _run(env, "attachment_add", {
        "name": "injected.txt", "mime": "text/plain", "data_base64": ENCODED,
    })
    await _run(env, "attachment_info", {"id": stored["id"]})
    assert "save" in proxy.calls and "get" in proxy.calls
    assert await db.fetchone("SELECT id FROM attachments WHERE id=?", (stored["id"],)) is not None
