"""Regressions for verified security findings (audit F-01/F-02/F-03/F-08 and later)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import Decision, RiskLevel, TaskStatus, ToolCall, ToolSpec
from leo.events import EventBus
from leo.integrations.messaging import (
    DiscordAdapter,
    TelegramAdapter,
    WebhookVerificationError,
    WhatsAppAdapter,
)
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl, arguments_digest
from leo.tools import ToolExecutorImpl, ToolRegistry, register_builtins


@pytest.fixture
async def sec_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events,
        permissions=permissions, registry=registry, executor=executor,
    )


async def test_unverified_approval_id_cannot_execute_high_risk_tool(sec_env):
    calls: list[dict] = []

    async def handler(args: dict, ctx: dict):
        calls.append(args)
        return {"executed": True}

    sec_env.registry.register(
        ToolSpec(name="high_risk", description="dangerous", risk=RiskLevel.HIGH,
                 input_schema={"type": "object", "properties": {}}),
        handler,
    )
    call = ToolCall(name="high_risk", arguments={}, actor="tester")

    with pytest.raises(ApprovalRequired) as first:
        await sec_env.executor.execute(call)
    approval_id = first.value.approval_id
    assert approval_id
    for bogus in ("totally-made-up-id", approval_id):
        try:
            result = await sec_env.executor.execute(call, approval_id=bogus)
        except ApprovalRequired:
            pass  # a fresh approval request is also a safe outcome
        else:
            assert not result.success
    assert calls == []


async def test_pending_approval_cannot_be_reused_before_resolution(sec_env):
    calls: list[dict] = []

    async def handler(args: dict, ctx: dict):
        calls.append(args)
        return {"executed": True}

    sec_env.registry.register(
        ToolSpec(name="high_risk", description="dangerous", risk=RiskLevel.HIGH,
                 input_schema={"type": "object", "properties": {}}),
        handler,
    )
    call = ToolCall(name="high_risk", arguments={}, actor="tester")
    with pytest.raises(ApprovalRequired) as first:
        await sec_env.executor.execute(call)
    approval_id = first.value.approval_id
    decision = await sec_env.permissions.authorize(
        "tester", "tool:high_risk", None, RiskLevel.HIGH, approval_id=approval_id
    )
    assert not decision.allowed
    assert not calls


async def test_concurrent_resolutions_cannot_double_resolve(sec_env, monkeypatch):
    """A single pending approval must be resolvable exactly once, even when two
    resolvers race. The check and the write used to be on separate connections."""
    pending = await sec_env.permissions.authorize("alice", "tool:danger", None, RiskLevel.HIGH)
    assert pending.approval_id

    # Force the stale-read race deterministically: every approval read yields
    # control before returning. The fixed implementation reads inside a single
    # write transaction and no longer calls db.fetchone here.
    real_fetchone = db.fetchone

    async def racing_fetchone(sql, params=()):
        row = await real_fetchone(sql, params)
        if "FROM approvals" in str(sql):
            await asyncio.sleep(0.01)
        return row

    monkeypatch.setattr(db, "fetchone", racing_fetchone)
    try:
        results = await asyncio.gather(
            sec_env.permissions.resolve_approval(pending.approval_id, True, "approver-a"),
            sec_env.permissions.resolve_approval(pending.approval_id, False, "approver-b"),
            return_exceptions=True,
        )
    finally:
        monkeypatch.setattr(db, "fetchone", real_fetchone)

    successes = [r for r in results if isinstance(r, Decision)]
    assert len(successes) == 1, results
    row = await db.fetchone("SELECT status, metadata FROM approvals WHERE id=?", (pending.approval_id,))
    assert row["status"] in ("approved", "denied")
    assert successes[0].allowed is (row["status"] == "approved")
    meta = json.loads(row["metadata"])
    assert meta["resolved_by"] in ("approver-a", "approver-b")


async def test_verified_approval_is_traceable_in_audit(sec_env):
    """Audit rows must name the approval that authorized/resolved a HIGH risk."""
    pending = await sec_env.permissions.authorize("alice", "tool:danger", None, RiskLevel.HIGH)
    await sec_env.permissions.resolve_approval(pending.approval_id, True, "approver")
    await sec_env.permissions.authorize(
        "alice", "tool:danger", None, RiskLevel.HIGH, approval_id=pending.approval_id
    )
    rows = await db.fetchall(
        "SELECT metadata, reason FROM audit_log WHERE action='tool:danger' AND decision='allowed'"
    )
    assert any(pending.approval_id in (r["metadata"] or "") for r in rows)
    resolution = next(r for r in rows if "approval approved by approver" in (r["reason"] or ""))
    assert "approver" in (resolution["metadata"] or "")
    assert pending.approval_id in (resolution["metadata"] or "")


async def test_path_violation_reason_not_masked_by_bogus_approval(sec_env):
    outside = str(sec_env.tmp_path.parent / "outside-secret.txt")
    decision = await sec_env.permissions.authorize(
        "alice", "tool:read", outside, RiskLevel.HIGH, approval_id="bogus-id"
    )
    assert not decision.allowed
    assert "outside allowed roots" in decision.reason


async def test_webhooks_fail_closed_without_verification_material(monkeypatch):
    for name in ("TELEGRAM_WEBHOOK_SECRET", "DISCORD_PUBLIC_KEY", "WHATSAPP_APP_SECRET"):
        monkeypatch.delenv(name, raising=False)
    for adapter in (TelegramAdapter(), DiscordAdapter(), WhatsAppAdapter()):
        assert adapter.verify_webhook({}, {}) is False
        with pytest.raises(WebhookVerificationError):
            await adapter.handle_webhook({}, {})


async def test_hmac_uses_exact_raw_body_bytes(monkeypatch):
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "s3cr3t")
    raw = b'{\n  "object": "whatsapp_business_account",\n  "entry": []\n}'
    signature = "sha256=" + hmac.new(b"s3cr3t", raw, hashlib.sha256).hexdigest()
    payload = json.loads(raw)
    adapter = WhatsAppAdapter()
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": signature}) is False
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": signature}, raw_body=raw) is True


async def test_api_requires_configured_token(monkeypatch):
    import httpx

    from leo.interfaces import api as api_module

    async def pending():
        return []

    async def health():
        return {"status": "ok"}

    monkeypatch.setenv("LEO_API_TOKEN", "test-token")
    monkeypatch.setattr(
        api_module, "_app",
        SimpleNamespace(
            permissions=SimpleNamespace(pending_approvals=pending),
            health=health,
        ),
    )
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/approvals")).status_code == 401
        ok = await client.get("/approvals", headers={"authorization": "Bearer test-token"})
        assert ok.status_code == 200
        assert ok.json() == {"approvals": []}


async def test_terminal_denies_interpreters_and_dangerous_flags(sec_env):
    register_builtins(sec_env.registry)
    call = ToolCall(name="terminal", arguments={}, actor="tester")
    for command in (
        "python -c 'print(1)'", "python3 -c 'print(1)'",
        "find . -delete", "sed -i s/a/b/ file.txt", "git push origin main",
        "cat /etc/hostname",
    ):
        result = await sec_env.executor.execute(
            ToolCall(name="terminal", arguments={"command": command}, actor="tester")
        )
        assert not result.success, command
    assert call.name == "terminal"


async def test_repeated_error_hint_is_consolidated_and_never_permits(sec_env):
    """Three same-class denials yield ONE hint; enforcement stays unchanged."""
    register_builtins(sec_env.registry)
    results = []
    for _ in range(4):
        results.append(await sec_env.executor.execute(ToolCall(
            name="terminal", arguments={"command": "python3 -c 'print(1)'"},
            actor="tester", task_id="no-progress-turn",
        )))
    for result in results:
        assert not result.success
        assert result.decision == "denied"
        assert "executable not allowed" in (result.error or "")

    assert "Policy hint" not in (results[0].error or "")
    assert "Policy hint" not in (results[1].error or "")
    assert "Policy hint" in (results[2].error or "")
    assert results[2].metadata["policy_hint"]["class"] == "terminal.executable_not_allowed"
    assert results[2].metadata["policy_hint"]["count"] == 3
    # Emitted once: the fourth failure reports no second hint.
    assert "Policy hint" not in (results[3].error or "")
    assert "policy_hint" not in results[3].metadata

    # A different turn starts with a clean counter but cannot bypass policy.
    fresh = await sec_env.executor.execute(ToolCall(
        name="terminal", arguments={"command": "python3 -c 'print(1)'"},
        actor="tester", task_id="another-turn",
    ))
    assert not fresh.success and "Policy hint" not in (fresh.error or "")


def test_extra_header_env_secrets_are_scrubbed(monkeypatch):
    from leo.config import ModelProviderConfig
    from leo.models.router import ModelRouter

    monkeypatch.setenv("LEO_TEST_API_KEY_VALUE", "super-secret-key-value")
    monkeypatch.setenv("LEO_TEST_SESSION_VALUE", "super-secret-session-value")
    settings = Settings()
    settings.models.providers["custom"] = ModelProviderConfig(
        api_key_env="LEO_TEST_API_KEY_VALUE",
        extra_header_env={"x-opencode-session": "LEO_TEST_SESSION_VALUE"},
    )
    router = ModelRouter(settings, providers={})
    scrubbed = router._scrub("key=super-secret-key-value session=super-secret-session-value")
    assert "super-secret-key-value" not in scrubbed
    assert "super-secret-session-value" not in scrubbed
    assert scrubbed.count("[REDACTED]") == 2


def test_serve_guard_uses_configured_host(tmp_path, monkeypatch):
    import subprocess
    import sys

    config = tmp_path / "leo.yaml"
    config.write_text("server:\n  host: 0.0.0.0\n  port: 8899\n", encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
        "LEO_CONFIG": str(config), "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    env.pop("LEO_API_TOKEN", None)
    result = subprocess.run(
        [sys.executable, "-m", "leo.interfaces.cli", "serve"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "non-loopback bind" in result.stderr


async def test_terminal_flag_path_values_are_confined(sec_env):
    register_builtins(sec_env.registry)
    outside = sec_env.tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret-value\n", encoding="utf-8")
    for command in (
        f"grep --file={outside} probe.txt",
        f"git --git-dir={outside} log",
        "sed -i.bak s/a/b/ file.txt",
    ):
        result = await sec_env.executor.execute(
            ToolCall(name="terminal", arguments={"command": command}, actor="tester")
        )
        assert not result.success, command


async def test_fallback_task_engine_enforces_evidence_and_cancellation(tmp_path, monkeypatch):
    from leo import db
    from leo.agents.engine import _DurableTasks

    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "fallback.db")
    await db.initialize()
    tasks = _DurableTasks()

    async def worker(task):
        return {"output": "ok"}

    task = await tasks.create("fallback work")
    finished = await tasks.run(task.id, worker)
    assert finished.status is TaskStatus.FAILED
    assert any("evidence" in str(err) for err in finished.errors)
    assert await tasks.run(task.id, worker) is not None  # terminal task is not replayed

    second = await tasks.create("cancelled work")
    assert await tasks.cancel(second.id)
    cancelled = await tasks.run(second.id, worker)
    assert cancelled.status is TaskStatus.CANCELLED


@pytest.fixture
async def offline_app(monkeypatch):
    from leo.config import ModelConfig, ModelProviderConfig
    from leo.core import LeoApp
    from leo.interfaces import api as api_module

    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    app = LeoApp(settings)
    await app.initialize()
    monkeypatch.setattr(api_module, "_app", app)
    yield app
    await app.shutdown()


async def test_chat_stream_rejects_raw_attachment_path_dicts(offline_app, tmp_path):
    """A {path, mime} dict must never be readable as an attachment record."""
    import httpx

    from leo.interfaces import api as api_module

    secret = tmp_path / "id_rsa.png"
    secret.write_bytes(b"PRIVATE KEY MATERIAL")
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST", "/chat/stream",
            json={"message": "review this", "attachments": [
                {"path": str(secret), "mime": "image/png"},
            ]},
        ) as response:
            assert response.status_code == 200
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
    types = [event["type"] for event in events]
    assert "error" in types and types[-1] == "run.finished"
    assert events[-1]["status"] == "failed"
    error = next(event for event in events if event["type"] == "error")
    assert error["code"] == "invalid_attachment"
    # The model was never called: no delta ever streamed.
    assert "message.delta" not in types


async def test_image_block_refuses_paths_outside_workspace_root(tmp_path, monkeypatch):
    """Even a readable image outside the store root is never exfiltrated."""
    import base64 as b64

    from leo.integrations.attachments import AttachmentStore

    monkeypatch.chdir(tmp_path)
    store = AttachmentStore(Settings())
    png = b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    )
    outside = tmp_path.parent / "exfil-target.png"
    outside.write_bytes(png)
    assert store.image_block({"mime": "image/png", "path": str(outside)}) is None
    assert store.image_block({"mime": "image/png", "path": "/etc/hostname"}) is None
    # Positive control: a stored attachment still produces a block.
    record = await store.save("pixel.png", "image/png", b64.b64encode(png).decode())
    block = store.image_block(record)
    assert block and block["source"]["media_type"] == "image/png"


# ------------------------------------------------- approval binding / atomicity
async def test_high_risk_approval_is_bound_to_arguments_digest(sec_env):
    executed: list[dict] = []

    async def handler(args, ctx):
        executed.append(args)
        return {"ok": True}

    sec_env.registry.register(ToolSpec(
        name="dangerous_write", description="dangerous", risk=RiskLevel.HIGH,
        input_schema={"type": "object", "properties": {
            "path": {"type": "string"}, "text": {"type": "string"},
        }},
    ), handler)
    original = ToolCall(name="dangerous_write", actor="alice", arguments={
        "path": str(sec_env.tmp_path / "allowed.txt"), "text": "hello",
    })
    with pytest.raises(ApprovalRequired) as first:
        await sec_env.executor.execute(original)
    approval_id = first.value.approval_id
    row = await db.fetchone("SELECT metadata FROM approvals WHERE id=?", (approval_id,))
    digest = arguments_digest(original.arguments)
    assert json.loads(row["metadata"])["arguments_digest"] == digest

    await sec_env.permissions.resolve_approval(approval_id, True, "approver")
    assert (await sec_env.executor.execute(original, approval_id=approval_id)).success

    for changed in (
        {"path": str(sec_env.tmp_path / "allowed.txt"), "text": "goodbye"},
        {"path": str(sec_env.tmp_path / "other.txt"), "text": "hello"},
    ):
        call = ToolCall(name="dangerous_write", actor="alice", arguments=changed)
        try:
            outcome = await sec_env.executor.execute(call, approval_id=approval_id)
        except ApprovalRequired as raised:
            assert raised.value.approval_id != approval_id
        else:
            assert not outcome.success  # invalid binding fails closed
    assert len(executed) == 1  # changed arguments never consumed the old grant
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,)))["status"] == "approved"


async def test_approval_metadata_never_serializes_live_services(sec_env):
    decision = await sec_env.permissions.authorize(
        "alice", "tool:danger", None, RiskLevel.HIGH,
        settings=sec_env.settings, events=sec_env.events,
        permissions=sec_env.permissions, tool_call_id="call-1", safe_value=7,
    )
    row = await db.fetchone("SELECT metadata FROM approvals WHERE id=?", (decision.approval_id,))
    metadata = json.loads(row["metadata"])
    assert metadata == {"tool_call_id": "call-1", "safe_value": 7}
    assert "settings" not in metadata and "events" not in metadata and "permissions" not in metadata


async def test_chat_actor_cannot_resolve_an_approval(sec_env):
    pending = await sec_env.permissions.authorize("alice", "tool:danger", None, RiskLevel.HIGH)
    with pytest.raises(PermissionError, match="chat actors"):
        await sec_env.permissions.resolve_approval(pending.approval_id, True, "chat:web")
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (pending.approval_id,))
    assert row["status"] == "pending"
    audit = await db.fetchall("SELECT decision, actor FROM audit_log WHERE action='tool:danger'")
    assert any(r["decision"] == "denied" and r["actor"] == "chat:web" for r in audit)


async def test_audit_failure_rolls_back_approval_resolution(sec_env, monkeypatch):
    pending = await sec_env.permissions.authorize("alice", "tool:danger", None, RiskLevel.HIGH)
    real_execute = db.execute

    async def broken_audit(sql, params=()):
        if "INSERT INTO audit_log" in str(sql):
            raise RuntimeError("audit sink down")
        return await real_execute(sql, params)

    monkeypatch.setattr(db, "execute", broken_audit)
    with pytest.raises(RuntimeError, match="audit sink down"):
        await sec_env.permissions.resolve_approval(pending.approval_id, True, "approver")
    row = await db.fetchone("SELECT status, resolved_at FROM approvals WHERE id=?", (pending.approval_id,))
    assert row["status"] == "pending" and row["resolved_at"] is None
    audit = await db.fetchall("SELECT id FROM audit_log WHERE reason LIKE '%approval approved by approver%'")
    assert audit == []
