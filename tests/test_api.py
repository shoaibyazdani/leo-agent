"""HTTP API contract tests using an injected offline app."""

from __future__ import annotations

import asyncio
import json

import base64
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.core import LeoApp
from leo.models.provider import ModelEmptyOutputError
from leo.interfaces import api as api_module


@pytest.fixture
async def offline_app(monkeypatch: pytest.MonkeyPatch) -> LeoApp:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    app = LeoApp(settings)
    await app.initialize()
    monkeypatch.setattr(api_module, "_app", app)
    yield app
    await app.shutdown()


@pytest.mark.asyncio
async def test_conversation_feed_bounds_huge_tool_payloads(offline_app: LeoApp) -> None:
    """A long session must not ship megabytes of tool output to the feed.

    A 2,688-message live conversation stored 8.8 MB of tool results; loading
    it took minutes (redaction scales with payload size). Tool rows are
    bounded in the chat feed while user/assistant text and the database row
    stay intact.
    """
    from leo import db
    from leo.contracts import utcnow

    conv_id = "conv_feed_bounds"
    await db.execute(
        "INSERT INTO conversations(id, channel, subject, created_at, updated_at) VALUES(?,?,?,?,?)",
        (conv_id, "api", None, utcnow(), utcnow()),
    )
    huge = "x" * 500_000
    tool_content = json.dumps({
        "call_id": "call_1",
        "tool": "read_file",
        "success": True,
        "output": {"path": "big.txt", "text": huge},
    })
    await db.execute(
        "INSERT INTO messages(conversation_id, role, content, metadata, created_at) VALUES(?,?,?,?,?)",
        (conv_id, "user", "please read the big file", "{}", utcnow()),
    )
    await db.execute(
        "INSERT INTO messages(conversation_id, role, content, metadata, created_at) VALUES(?,?,?,?,?)",
        (conv_id, "tool", tool_content, json.dumps({"name": "read_file"}), utcnow()),
    )
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/conversations/{conv_id}")
    assert response.status_code == 200
    assert len(response.content) < 100_000, len(response.content)
    body = response.json()
    tool_row = next(m for m in body["messages"] if m["role"] == "tool")
    parsed = json.loads(tool_row["content"])
    assert parsed["truncated"] is True
    assert isinstance(parsed["output"], dict) and parsed["output"]["truncated"] is True
    assert len(parsed["output"]["preview"]) <= api_module.MAX_FEED_TOOL_RESULT_CHARS + 20
    user_row = next(m for m in body["messages"] if m["role"] == "user")
    assert user_row["content"] == "please read the big file"
    # The durable row keeps the full result for export.
    stored = await db.fetchone(
        "SELECT content FROM messages WHERE conversation_id=? AND role='tool'", (conv_id,)
    )
    assert len(stored["content"]) > 400_000


@pytest.mark.asyncio
async def test_approval_resolve_rejects_cross_site_and_blind_clients(
    offline_app: LeoApp,
) -> None:
    """A page on another origin, or a stale poller, cannot decide an approval.

    A dashboard tab left open auto-POSTed resolves for every pending approval;
    any web page can reach the loopback bind (CORS only hides the response),
    so the operator surface checks the request origin and requires the row's
    own action as an explicit confirmation.
    """
    from leo import db
    from leo.contracts import RiskLevel

    decision = await offline_app.permissions.authorize(
        "learning:engine", "learning.note", None, RiskLevel.HIGH
    )
    approval_id = decision.approval_id
    assert approval_id
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cross = await client.post(
            f"/approvals/{approval_id}/resolve",
            json={"approved": True, "confirm_action": "learning.note"},
            headers={"origin": "http://evil.example"},
        )
        assert cross.status_code == 403
        site = await client.post(
            f"/approvals/{approval_id}/resolve",
            json={"approved": True, "confirm_action": "learning.note"},
            headers={"sec-fetch-site": "cross-site"},
        )
        assert site.status_code == 403
        blind = await client.post(
            f"/approvals/{approval_id}/resolve", json={"approved": True}
        )
        assert blind.status_code == 409
        assert "confirm_action" in blind.json()["detail"]
        status = await db.fetchone(
            "SELECT status FROM approvals WHERE id=?", (approval_id,)
        )
        assert status["status"] == "pending", "rejected calls must not touch state"
        ok = await client.post(
            f"/approvals/{approval_id}/resolve",
            json={"approved": True, "confirm_action": "learning.note"},
            headers={"origin": "http://test"},
        )
        assert ok.status_code == 200
        assert ok.json()["allowed"] is True


@pytest.mark.asyncio
async def test_schedule_card_projection_ack_and_digest_bound_resolve(
    offline_app: LeoApp,
) -> None:
    """The card endpoints never decide without a shown, digest-bound choice."""
    from leo import db

    permissions = offline_app.permissions
    card = await permissions.create_schedule_card(
        kind="schedule_create",
        body='Schedule "Water" (every day at 8am): Drink water.',
        proposal_digest="api-digest",
        requested_by="chat:web",
        conversation_id="conv-api",
    )

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/approvals")
        projected = listing.json()["cards"]
        assert [item["approval_id"] for item in projected] == [card.approval_id]
        assert projected[0]["body"] == card.body
        assert [item["value"] for item in projected[0]["choices"]] == card.choices
        # Raw rows stay available for existing callers.
        assert listing.json()["approvals"][0]["id"] == card.approval_id

        # A blind resolve before any presentation is refused.
        unshown = await client.post(
            f"/approvals/{card.approval_id}/resolve",
            json={"choice": "once", "confirm_proposal_digest": "api-digest"},
            headers={"origin": "http://test"},
        )
        assert unshown.status_code == 403

        cross = await client.post(
            f"/approvals/{card.approval_id}/ack", json={},
            headers={"origin": "http://evil.example"},
        )
        assert cross.status_code == 403
        ack = await client.post(
            f"/approvals/{card.approval_id}/ack", json={},
            headers={"origin": "http://test"},
        )
        assert ack.status_code == 200
        assert ack.json()["card"]["approval_id"] == card.approval_id
        assert ack.json()["card"]["acknowledged_at"]
        assert ack.json()["card"]["acknowledged_surface"] == "web"
        assert ack.json()["card"]["shown_at"], "ack records the client presentation"

        no_digest = await client.post(
            f"/approvals/{card.approval_id}/resolve", json={"choice": "once"}
        )
        assert no_digest.status_code == 409
        stale = await client.post(
            f"/approvals/{card.approval_id}/resolve",
            json={"choice": "once", "confirm_proposal_digest": "wrong"},
            headers={"origin": "http://test"},
        )
        assert stale.status_code == 409
        row = await db.fetchone(
            "SELECT status FROM approvals WHERE id=?", (card.approval_id,)
        )
        assert row["status"] == "pending"

        approved = await client.post(
            f"/approvals/{card.approval_id}/resolve",
            json={
                "choice": "session",
                "confirm_proposal_digest": "api-digest",
            },
            headers={"origin": "http://test"},
        )
        assert approved.status_code == 200
        body = approved.json()
        assert body["status"] == "approved" and body["grant_id"]
        assert body["card"]["status"] == "approved"


@pytest.mark.asyncio
async def test_api_refuses_a_schedule_card_the_route_principal_requested(
    offline_app: LeoApp,
) -> None:
    """The unauthenticated local API principal cannot approve its own card."""
    permissions = offline_app.permissions
    card = await permissions.create_schedule_card(
        kind="schedule_create",
        body='Schedule "Water": drink water.',
        proposal_digest="self-digest",
        requested_by="local",
        conversation_id="conv-self",
    )
    await permissions.present_schedule_card(card.approval_id, surface="web")

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/approvals/{card.approval_id}/resolve",
            json={"choice": "once", "confirm_proposal_digest": "self-digest"},
            headers={"origin": "http://test"},
        )
    assert response.status_code == 403
    assert "requester cannot resolve" in response.json()["detail"]


@pytest.mark.asyncio
async def test_api_health_chat_tasks_and_audit(offline_app: LeoApp) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"

        chat = await client.post("/chat", json={"message": "hello"})
        assert chat.status_code == 200 and chat.json()["provider"] == "mock"

        created = await client.post("/tasks", json={"objective": "write docs"})
        assert created.status_code == 200 and created.json()["status"] == "created"
        task_id = created.json()["id"]
        assert (await client.get(f"/tasks/{task_id}")).status_code == 200

        assert (await client.get("/skills")).json()["skills"]
        assert (await client.get("/events")).json()["events"]
        assert (await client.get("/metrics")).status_code == 200
        assert (await client.get("/evaluation")).status_code == 200
        tool = await client.post("/tools/calculator/execute", json={"arguments": {"expression": "2+3"}})
        assert tool.status_code == 200 and tool.json()["success"]
        assert (await client.get("/audit")).json()["audit"]
        # unknown tool/skill are audited outcomes but must map to HTTP 404
        assert (await client.post("/tools/no_such_tool/execute", json={"arguments": {}})).status_code == 404
        assert (await client.post("/skills/no_such_skill/run", json={"arguments": {}})).status_code == 404
        assert (await client.get("/approvals")).json()["approvals"] == []
        assert (await client.get("/")).status_code == 200  # dashboard


async def test_api_agents_and_memory_endpoints(offline_app: LeoApp) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        agents = await client.get("/agents")
        assert agents.status_code == 200
        assert len(agents.json()["agents"]) >= 20
        await client.post("/memory", json={"content": "dashboard endpoint check"})
        memories = await client.get("/memory?limit=5")
        assert memories.status_code == 200
        assert any(m["content"] == "dashboard endpoint check" for m in memories.json()["memories"])


async def test_chat_stream_emits_ordered_events(offline_app: LeoApp) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/chat/stream", json={"message": "hello stream"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
    types = [event["type"] for event in events]
    assert "run.started" in types and "message.delta" in types
    assert "message.finished" in types and "run.finished" in types
    assert types[-1] == "run.finished"
    assert all("seq" in event and "run_id" in event for event in events)
    assert [event["seq"] for event in events] == sorted(event["seq"] for event in events)


async def test_chat_empty_model_output_returns_chat_response_error(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated model turn is a normal, sanitized error turn, not a 500."""
    from leo import db

    class EmptyRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            raise ModelEmptyOutputError("fake", "max_tokens")

    router = EmptyRouter()
    monkeypatch.setattr(offline_app, "router", router)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "answer this"})

    assert response.status_code == 200
    body = response.json()
    assert set(api_module.ChatResponse.model_fields) <= set(body)
    assert body["status"] == "error"
    assert body["error"]["code"] == "model_empty_output"
    assert body["content"] and "answer" in body["content"].lower()
    assert body["tool_calls_pending"] == [] and body["tool_results"] == []
    assert router.calls == 1

    rows = await db.fetchall(
        "SELECT content, metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    )
    meta = json.loads(rows[0]["metadata"])
    assert meta["run_status"] == "error"
    assert meta["error"]["code"] == "model_empty_output"
    assert rows[0]["content"].strip()


async def test_chat_unexpected_failure_still_returns_chat_response_error(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("unexpected failure with api_key = sk-abcdefghijklmnop")

    monkeypatch.setattr(offline_app, "ask", boom)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "answer this"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error" and body["error"]["code"] == "RuntimeError"
    assert "sk-abcdefghijklmnop" not in json.dumps(body)


async def test_chat_stream_empty_model_output_persists_error_turn(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leo import db

    class EmptyStreamRouter:
        async def stream(self, request):
            raise ModelEmptyOutputError("fake", "max_tokens")
            yield ""  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(offline_app, "router", EmptyStreamRouter())
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/chat/stream", json={"message": "answer this"}) as response:
            assert response.status_code == 200
            events = [
                json.loads(line[5:].strip())
                async for line in response.aiter_lines()
                if line.startswith("data:")
            ]

    error = next(event for event in events if event["type"] == "error")
    assert error["code"] == "model_empty_output"
    assert error["status"] == "error"
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "error"

    rows = await db.fetchall(
        "SELECT metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    )
    meta = json.loads(rows[0]["metadata"])
    assert meta["run_status"] == "error"
    assert meta["error"]["code"] == "model_empty_output"


async def test_attachment_upload_validation_and_no_host_path(offline_app: LeoApp) -> None:

    transport = httpx.ASGITransport(app=api_module.api)
    png = base64.b64encode(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==")
    ).decode()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/attachments", json={
            "name": "pixel.png", "mime": "image/png", "data_base64": png,
        })
        assert created.status_code == 200, created.text
        record = created.json()
        assert "path" not in record and "text_preview" not in record
        assert record["kind"] == "image" and record["mime"] == "image/png"

        preview = await client.get(f"/attachments/{record['id']}/content")
        assert preview.status_code == 200

        rejected = await client.post("/attachments", json={
            "name": "fake.png", "mime": "image/png",
            "data_base64": base64.b64encode(b"not a png").decode(),
        })
        assert rejected.status_code == 400

        # Unknown runs are rejected truthfully (no more unconditional true).
        cancelled = await client.post("/runs/run_test/cancel")
        assert cancelled.status_code == 404


async def test_memory_invalid_category_kind_are_client_errors(offline_app: LeoApp) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        bad_category = await client.post("/memory", json={
            "content": "invalid category", "category": "not_a_category", "kind": "fact",
        })
        assert bad_category.status_code == 400
        bad_kind = await client.post("/memory", json={
            "content": "invalid kind", "category": "semantic", "kind": "not_a_kind",
        })
        assert bad_kind.status_code == 400
        empty = await client.post("/memory", json={"content": "   ", "category": "semantic"})
        assert empty.status_code == 400
        good = await client.post("/memory", json={"content": "valid memory"})
        assert good.status_code == 200


async def test_conversation_delete_cleans_attachments(offline_app: LeoApp) -> None:

    from leo import db

    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/chat", json={"message": "with an attachment"})
        conv_id = created.json()["conversation_id"]
        upload = await client.post("/attachments", json={
            "name": "pixel.png", "mime": "image/png", "data_base64": png,
            "conversation_id": conv_id,
        })
        assert upload.status_code == 200
        att_id = upload.json()["id"]
        assert (await db.fetchone("SELECT id FROM attachments WHERE id=?", (att_id,))) is not None

        assert (await client.delete(f"/conversations/{conv_id}")).status_code == 200
        assert (await db.fetchone("SELECT id FROM attachments WHERE id=?", (att_id,))) is None
        audit = await db.fetchone(
            "SELECT metadata FROM audit_log WHERE action='conversation.delete' AND resource=?", (conv_id,)
        )
        assert json.loads(audit["metadata"])["attachments_removed"] == 1


async def test_conversation_list_detail_and_delete(offline_app: LeoApp) -> None:
    """Chat sidebar endpoints: listing, history with metadata, audited delete."""
    from leo import db
    from leo.contracts import utcnow

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/chat", json={"message": "first question about deployment"})
        assert first.status_code == 200
        first_id = first.json()["conversation_id"]
        second = await client.post("/chat", json={"message": "second question"})
        second_id = second.json()["conversation_id"]
        assert first_id != second_id

        listed = await client.get("/conversations")
        assert listed.status_code == 200
        rows = listed.json()["conversations"]
        assert [row["id"] for row in rows[:2]] == [second_id, first_id]  # newest first
        row = next(r for r in rows if r["id"] == first_id)
        assert row["title"].startswith("first question")
        assert row["preview_role"] in {"assistant", "user"}
        assert row["message_count"] >= 2
        assert row["channel"] == "api"

        # a streamed turn reuses an existing conversation when its id is passed
        async with client.stream(
            "POST", "/chat/stream", json={"message": "continue here", "conversation_id": first_id}
        ) as response:
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
        started = next(e for e in events if e["type"] == "run.started")
        assert started["conversation_id"] == first_id

        detail = await client.get(f"/conversations/{first_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["conversation_id"] == first_id
        roles = [m["role"] for m in body["messages"]]
        assert roles[0] == "user" and "assistant" in roles
        assert all(isinstance(m["metadata"], dict) for m in body["messages"])
        streamed_user = next(
            m for m in body["messages"] if m["role"] == "user" and m["content"] == "continue here"
        )
        assert "attachments" in streamed_user["metadata"]

        missing = await client.get("/conversations/conv_missing")
        assert missing.status_code == 404

        deleted = await client.delete(f"/conversations/{first_id}")
        assert deleted.status_code == 200 and deleted.json()["deleted"] == first_id
        assert (await client.get(f"/conversations/{first_id}")).status_code == 404
        assert (await client.delete(f"/conversations/{first_id}")).status_code == 404
        leftovers = await db.fetchone(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id=?", (first_id,)
        )
        assert leftovers["n"] == 0
        audit = await db.fetchone(
            "SELECT COUNT(*) AS n FROM audit_log WHERE action='conversation.delete' AND resource=?",
            (first_id,),
        )
        assert audit["n"] == 1

        relisted = await client.get("/conversations")
        assert first_id not in [r["id"] for r in relisted.json()["conversations"]]
        assert second_id in [r["id"] for r in relisted.json()["conversations"]]


# --- HTTP conversation export: shared core, bounded and confined ---
async def _export_fixture(client: httpx.AsyncClient, secret: str) -> str:
    from leo import db
    from leo.contracts import utcnow

    created = await client.post("/chat", json={"message": "export fixture question"})
    assert created.status_code == 200, created.text
    conversation_id = created.json()["conversation_id"]
    await db.execute(
        "INSERT INTO messages(conversation_id, role, content, metadata, created_at) "
        "VALUES(?,?,?,?,?)",
        (conversation_id, "user", secret, "{}", utcnow()),
    )
    return conversation_id


@pytest.mark.parametrize("fmt", ["md", "json", "txt"])
async def test_conversation_export_writes_real_bounded_file_in_all_formats(
    offline_app: LeoApp, tmp_path: Path, fmt: str
) -> None:
    from leo import db

    offline_app.settings.paths.exports = tmp_path / "exports"
    transport = httpx.ASGITransport(app=api_module.api)
    secret = "note api_key=sk-live-DONOTLEAK1234567890"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = await _export_fixture(client, secret)
        response = await client.post(
            f"/conversations/{conversation_id}/export?format={fmt}&redact=true"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True and body["format"] == fmt
        assert body["conversation_id"] == conversation_id and body["redacted"] is True
        assert body["has_more"] is False

        path = Path(body["path"])
        assert path.is_file() and path.suffix == f".{fmt}"
        assert path.resolve().parent == (tmp_path / "exports").resolve()
        data = path.read_bytes()
        assert body["bytes"] == len(data) and len(data) <= 1_000_000
        text = data.decode("utf-8")
        assert conversation_id in text and "DONOTLEAK" not in text and "[REDACTED]" in text
        expected = (await db.fetchone(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id=?", (conversation_id,)
        ))["n"]
        assert body["count"] == expected >= 3
        if fmt == "json":
            document = json.loads(text)
            assert document["count"] == expected and len(document["messages"]) == expected
        elif fmt == "md":
            assert text.startswith("# Conversation export")
        else:
            assert "conversation_id:" in text and "redacted: True" in text


async def test_conversation_export_redact_false_keeps_content_and_reports_it(
    offline_app: LeoApp, tmp_path: Path
) -> None:
    offline_app.settings.paths.exports = tmp_path / "exports"
    transport = httpx.ASGITransport(app=api_module.api)
    secret = "api_key=sk-live-PLAINTEXT1234567890"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = await _export_fixture(client, secret)
        response = await client.post(
            f"/conversations/{conversation_id}/export?format=txt&redact=false"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["redacted"] is False
        path = Path(body["path"])
        assert path.is_file()
        assert secret in path.read_text(encoding="utf-8")


async def test_conversation_export_rejects_symlink_target_truthfully(
    offline_app: LeoApp, tmp_path: Path
) -> None:
    from leo.tools import conversation_tools

    exports = (tmp_path / "exports").resolve()
    exports.mkdir()
    offline_app.settings.paths.exports = exports
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me", encoding="utf-8")
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = await _export_fixture(client, "safe message")
        # Stable naming lets a hostile symlink sit at exactly the path the
        # shared write core will choose.
        target = conversation_tools._target_path(exports, None, "md", conversation_id)
        target.symlink_to(outside)
        blocked = await client.post(f"/conversations/{conversation_id}/export?format=md")
        assert blocked.status_code == 400
        assert "symlink" in blocked.json()["detail"].lower()
        assert outside.read_text(encoding="utf-8") == "keep me"


async def test_conversation_export_errors_and_token_auth(
    offline_app: LeoApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_app.settings.paths.exports = tmp_path / "exports"
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = await _export_fixture(client, "auth fixture message")
        missing = await client.post("/conversations/conv_does_not_exist/export")
        assert missing.status_code == 404
        assert missing.json()["detail"] == "conversation not found"
        bad_format = await client.post(f"/conversations/{conversation_id}/export?format=csv")
        assert bad_format.status_code == 400 and "invalid format" in bad_format.text

        monkeypatch.setenv("LEO_API_TOKEN", "export-token")
        url = f"/conversations/{conversation_id}/export?format=json&redact=true"
        # The write route is a POST; GET stays a read-only manifest.
        assert (await client.get(url)).status_code == 401
        assert (await client.post(url)).status_code == 401
        assert (await client.post(url, headers={"authorization": "Bearer wrong"})).status_code == 401
        denied = await client.post(url, headers={"authorization": "Bearer export-token"})
        assert denied.status_code == 200, denied.text
        assert Path(denied.json()["path"]).is_file()
        # The alternate token header used by the other conversation routes works too.
        alternate = await client.post(url, headers={"x-leo-token": "export-token"})
        assert alternate.status_code == 200


async def test_export_get_is_read_only_manifest_and_post_checks_origin(
    offline_app: LeoApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exports = (tmp_path / "exports").resolve()
    offline_app.settings.paths.exports = exports
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = await _export_fixture(client, "manifest fixture")

        manifest = await client.get(f"/conversations/{conversation_id}/export")
        assert manifest.status_code == 200
        body = manifest.json()
        assert body["method"] == "POST" and body["exports"] == []
        assert body["limit"]["max"] == 500
        assert not exports.exists()  # a GET never writes a file

        # A cross-origin browser form POST must not be able to create files.
        foreign = await client.post(
            f"/conversations/{conversation_id}/export",
            headers={"origin": "https://evil.example"},
        )
        assert foreign.status_code == 403 and not exports.exists()

        allowed = await client.post(
            f"/conversations/{conversation_id}/export",
            headers={"origin": "http://localhost:5173"},
        )
        assert allowed.status_code == 200, allowed.text
        path = Path(allowed.json()["path"])
        listed = await client.get(f"/conversations/{conversation_id}/export")
        assert [item["path"] for item in listed.json()["exports"]] == [str(path)]
        assert listed.json()["exports"][0]["bytes"] == path.stat().st_size


async def test_export_route_is_rate_limited(
    offline_app: LeoApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leo.tools import conversation_tools

    offline_app.settings.paths.exports = tmp_path / "exports"
    monkeypatch.setattr(api_module, "EXPORT_RATE_LIMIT", 2)
    api_module._export_requests.clear()
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = await _export_fixture(client, "rate limit fixture")
        url = f"/conversations/{conversation_id}/export?format=json"
        assert (await client.post(url)).status_code == 200
        assert (await client.post(url)).status_code == 200
        limited = await client.post(url)
        assert limited.status_code == 429 and "rate limit" in limited.text
        assert len([
            path for path in (tmp_path / "exports").glob("*")
            if path.name != conversation_tools._EXPORT_LOCK_NAME
        ]) == 2


async def test_export_route_can_write_a_complete_150_message_transcript(
    offline_app: LeoApp, tmp_path: Path
) -> None:
    from leo import db
    from leo.contracts import utcnow

    offline_app.settings.paths.exports = tmp_path / "exports"
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/chat", json={"message": "long transcript"})
        conversation_id = created.json()["conversation_id"]
        stamp = utcnow()
        for index in range(148):
            await db.execute(
                "INSERT INTO messages(conversation_id, role, content, metadata, created_at) "
                "VALUES(?,?,?,?,?)",
                (conversation_id, "user" if index % 2 else "assistant",
                 f"message {index}: plain transcript text", "{}", stamp),
            )
        total = (await db.fetchone(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id=?", (conversation_id,)
        ))["n"]
        assert total == 150

        response = await client.post(
            f"/conversations/{conversation_id}/export?format=json&limit=150"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["count"] == total and body["has_more"] is False
        document = json.loads(Path(body["path"]).read_text(encoding="utf-8"))
        assert document["count"] == total and len(document["messages"]) == total

        # Bounded pages with after_id cover the same transcript.
        first = await client.post(f"/conversations/{conversation_id}/export?limit=100")
        second = await client.post(
            f"/conversations/{conversation_id}/export?limit=100"
            f"&after_id={first.json()['next_cursor']}"
        )
        assert first.json()["count"] == 100 and first.json()["has_more"] is True
        assert second.json()["count"] == 50 and second.json()["has_more"] is False
        assert (await client.post(
            f"/conversations/{conversation_id}/export?limit=501"
        )).status_code == 400


async def test_sse_query_token_never_exposes_preview_credentials(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A query token is browser-readable, even when minted by an API principal."""
    import asyncio

    from starlette.requests import Request as StarletteRequest

    from leo.tools.preview_tools import token_audit_id

    token = "ssePreview_AbCdEf1234567890"
    url = f"http://127.0.0.1:18765/preview/{token}"
    digest = token_audit_id(token)
    payload = {
        "token_id": token,
        "content": json.dumps({"success": True, "url": url}),
        "text": f"open {url}",
    }
    monkeypatch.setenv("LEO_API_TOKEN", "sse-owner-token")
    headers = {"Authorization": "Bearer sse-owner-token"}
    transport = httpx.ASGITransport(app=api_module.api)

    async def _connected(self: Any) -> bool:
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", _connected)

    async def _publish() -> None:
        for _ in range(40):
            await asyncio.sleep(0.05)
            await offline_app.events.emit("agent.message", **payload)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        grant = await client.post("/events/stream-token", headers=headers)
        assert grant.status_code == 200, grant.text
        stream_token = grant.json()["token"]

        publisher = asyncio.create_task(_publish())
        try:
            sse = await client.get(
                "/events/stream",
                params={"token": stream_token, "max_events": 1, "max_seconds": 3},
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert sse.status_code == 200
        assert '"topic":"agent.message"' in sse.text
        assert token not in sse.text
        assert digest in sse.text
        assert "http://127.0.0.1:18765/preview/[REDACTED]" in sse.text

        # The Bearer-authenticated feed still has the genuine value: only the
        # browser-readable query-token surface is redacted.
        private = await client.get("/events", params={"topic": "agent.message"}, headers=headers)
        assert private.status_code == 200 and token in private.text


async def test_unauthenticated_conversation_reads_redact_every_token_shape(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leo import db
    from leo.contracts import utcnow

    token = "apiReadToken_AbCdEf1234567890"
    nested = "apiNestedToken_AbCdEf1234567890"
    message = (
        f"open http:\\/\\/127.0.0.1:18765\\/preview\\/{token} "
        f"then token_id: {token} "
        f"or http:\\\\/\\\\/127.0.0.1:18765\\\\/preview\\\\/{nested}"
    )
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post("/chat", json={"message": "read fixture"})).json()[
            "conversation_id"
        ]
        await db.execute(
            "INSERT INTO messages(conversation_id, role, content, metadata, created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, "user", message, "{}", utcnow()),
        )
        public = await client.get(f"/conversations/{conversation_id}")
        assert public.status_code == 200
        assert token not in public.text and nested not in public.text
        assert "[REDACTED]" in public.text and "sha256:" in public.text

        monkeypatch.setenv("LEO_API_TOKEN", "reader-token")
        privileged = await client.get(
            f"/conversations/{conversation_id}",
            headers={"authorization": "Bearer reader-token"},
        )
        assert privileged.status_code == 200 and token in privileged.text


async def test_sse_redacts_entire_event_in_every_visible_topic(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token in correlation_id or a non-agent topic must not reach the feed."""
    import asyncio

    from starlette.requests import Request as StarletteRequest

    from leo.tools.preview_tools import token_audit_id

    token = "wholeEventToken_AbCdEf1234567890"
    url = f"http:\\/\\/127.0.0.1:18765\\/preview\\/{token}"
    digest = token_audit_id(token)
    monkeypatch.setenv("LEO_API_TOKEN", "whole-event-owner")
    headers = {"Authorization": "Bearer whole-event-owner"}

    async def _connected(self: Any) -> bool:
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", _connected)

    async def _publish() -> None:
        for _ in range(50):
            await asyncio.sleep(0.05)
            await offline_app.events.emit(
                "run.finished", detail=f"see {url}", correlation_id=token
            )
            await offline_app.events.emit("agent.message", text="plain answer")
            await offline_app.events.emit(
                "schedule.deleted", name="cleanup", correlation_id=f"corr:{url}"
            )

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        grant = await client.post("/events/stream-token", headers=headers)
        assert grant.status_code == 200
        publisher = asyncio.create_task(_publish())
        try:
            sse = await client.get(
                "/events/stream",
                params={
                    "token": grant.json()["token"], "max_events": 3, "max_seconds": 3,
                },
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert sse.status_code == 200
        for topic in ("run.finished", "agent.message", "schedule.deleted"):
            assert f'"topic":"{topic}"' in sse.text
        assert token not in sse.text
        assert digest in sse.text  # correlation_id is redacted, not dropped
        assert "[REDACTED]" in sse.text

        # The bus and privileged reads keep the genuine capability.
        stored = await offline_app.events.history(limit=20)
        assert any(event.correlation_id == token for event in stored)
        private = await client.get(
            "/events", params={"topic": "run.finished"}, headers=headers
        )
        assert private.status_code == 200 and token in private.text


async def test_agent_activity_get_sync_requires_an_explicit_signal(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-site browser GETs read the projection without writing SQLite."""
    from leo.agents import activity

    calls: list[int] = []

    async def fake_sync() -> dict[str, Any]:
        calls.append(1)
        return {}

    monkeypatch.setattr(activity, "sync_agent_activity", fake_sync)
    cross_site = {
        "origin": "https://evil.example",
        "sec-fetch-site": "cross-site",
        "sec-fetch-mode": "no-cors",
    }
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/agent-activity", headers=cross_site)
        assert listed.status_code == 200 and calls == []
        summary = await client.get("/agent-activity/summary", headers=cross_site)
        assert summary.status_code == 200 and calls == []

        explicit = await client.get("/agent-activity", headers={"x-leo-refresh": "1"})
        assert explicit.status_code == 200 and len(calls) == 1
        explicit_summary = await client.get(
            "/agent-activity/summary", headers={"x-leo-refresh": "1"}
        )
        assert explicit_summary.status_code == 200 and len(calls) == 2

        monkeypatch.setenv("LEO_API_TOKEN", "activity-token")
        privileged = await client.get(
            "/agent-activity",
            headers={"authorization": "Bearer activity-token"},
        )
        assert privileged.status_code == 200 and len(calls) == 3


# --- go-live robustness: tool identity and quota (report items 1-2) ---
async def test_tool_identity_comes_from_path_not_body(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        rejected = await client.post("/tools/json_parse/execute", json={
            "name": "terminal", "arguments": {"command": "pwd"},
        })
        assert rejected.status_code == 400
        assert "does not match" in rejected.text
        ok = await client.post("/tools/json_parse/execute", json={
            "arguments": {"text": '{"a":1}'},
        })
        assert ok.status_code == 200 and ok.json()["tool"] == "json_parse"


async def test_http_tool_quota_is_scoped_and_rate_limit_is_429(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    payload = {"arguments": {"expression": "1+1"}, "max_tool_calls": 1, "task_id": "quota-task"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/tools/calculator/execute", json=payload)
        assert first.status_code == 200, first.text
        limited = await client.post("/tools/calculator/execute", json=payload)
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) >= 1
        assert limited.json()["decision"] == "rate_limited"
        # A different task has its own bucket.
        other = await client.post("/tools/calculator/execute", json={
            **payload, "task_id": "other-task",
        })
        assert other.status_code == 200


# --- go-live robustness: truthful cancellation (report item 3) ---
async def test_run_cancel_is_truthful_and_audited(offline_app) -> None:
    from leo import db

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unknown = await client.post("/runs/no-such-run/cancel")
        assert unknown.status_code == 404

        offline_app._start_run("run_live_test", "conv_live")
        ok = await client.post("/runs/run_live_test/cancel")
        assert ok.status_code == 200
        assert ok.json() == {"cancelled": True, "status": "cancelled"}
        repeat = await client.post("/runs/run_live_test/cancel")
        assert repeat.status_code == 409

    assert any(e.topic == "run.cancelled" for e in await offline_app.events.history())
    rows = await db.fetchall("SELECT action, resource FROM audit_log WHERE action='run.cancel'")
    assert any(row["resource"] == "run_live_test" for row in rows)


async def test_task_cancel_returns_404_and_409(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/tasks", json={"objective": "cancel me"})
        task_id = created.json()["id"]
        first = await client.post(f"/tasks/{task_id}/cancel")
        assert first.status_code == 200 and first.json()["cancelled"] is True
        repeat = await client.post(f"/tasks/{task_id}/cancel")
        assert repeat.status_code == 409
        unknown = await client.post("/tasks/task_does_not_exist/cancel")
        assert unknown.status_code == 404


# --- go-live robustness: validation and audit integrity (items 7-8, 10) ---
async def test_status_enum_and_limits_are_validated(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/tasks?status=bogus")).status_code == 400
        assert (await client.get("/audit?limit=-1")).status_code == 400
        assert (await client.get("/events?limit=0")).status_code == 400
        assert (await client.get("/memory?limit=-5")).status_code == 400
        assert (await client.get("/tasks?status=ready")).status_code == 200


async def test_invalid_unicode_and_controls_rejected(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        bad = await client.post(
            "/chat", content=b'{"message": "lone surrogate \\ud800 here"}',
            headers={"content-type": "application/json"},
        )
        assert bad.status_code == 422
        nul = await client.post("/chat", json={"message": "has \u0000 byte"})
        assert nul.status_code == 422


async def test_integrations_evaluate_returns_400_for_malformed_candidate(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        bad = await client.post("/integrations/evaluate", json={"candidate": {"description": "no name"}})
        assert bad.status_code == 400
        good = await client.post("/integrations/evaluate", json={"candidate": {"name": "safe_notes"}})
        assert good.status_code == 200, good.text


async def test_denied_tool_call_is_recorded_in_audit(offline_app) -> None:
    from leo import db

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.post("/tools/terminal/execute", json={
            "arguments": {"command": "python -c 'print(1)'"},
        })
        assert denied.status_code == 200 and not denied.json()["success"]
    rows = await db.fetchall(
        "SELECT action, decision, reason FROM audit_log WHERE action='tool.execute' ORDER BY id DESC LIMIT 3"
    )
    assert rows and any(row["decision"] != "allowed" for row in rows)


# --- go-live robustness: memory caps and per-conversation runs (items 6, 9) ---
async def test_memory_content_is_capped(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        too_long = await client.post("/memory", json={"content": "x" * 5000})
        assert too_long.status_code == 422
        ok = await client.post("/memory", json={"content": "y" * 4000})
        assert ok.status_code == 200


async def test_overlapping_stream_on_same_conversation_is_409(offline_app) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    offline_app._active_conversations.add("conv_busy")
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            conflict = await client.post("/chat/stream", json={
                "message": "hello", "conversation_id": "conv_busy",
            })
            assert conflict.status_code == 409
    finally:
        offline_app._active_conversations.discard("conv_busy")


async def test_messages_conversation_index_exists(offline_app) -> None:
    from leo import db

    rows = await db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_conversation'"
    )
    assert rows


async def test_raw_token_shape_is_scrubbed_on_sse_reads_and_exports(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One raw capability cannot survive on any browser-readable surface."""
    import asyncio

    from starlette.requests import Request as StarletteRequest

    from leo import db
    from leo.contracts import utcnow
    from leo.tools.preview_tools import token_audit_id

    token = "crossSurfaceToken_AbCdEf1234567890"
    digest = token_audit_id(token)
    offline_app.settings.paths.exports = tmp_path / "exports"

    async def _connected(self: Any) -> bool:
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", _connected)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post(
            "/chat", json={"message": "cross surface fixture"}
        )).json()["conversation_id"]
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, "user", f"correlation_id={token}", "{}", utcnow()),
        )
        public = await client.get(f"/conversations/{conversation_id}")
        assert public.status_code == 200
        assert token not in public.text
        assert digest in public.text

        monkeypatch.setenv("LEO_API_TOKEN", "surface-owner")
        headers = {"Authorization": "Bearer surface-owner"}
        grant = await client.post("/events/stream-token", headers=headers)
        assert grant.status_code == 200, grant.text

        async def _publish() -> None:
            for _ in range(40):
                await asyncio.sleep(0.05)
                await offline_app.events.emit(
                    "run.finished", correlation_id=token, detail=f"ref {token}"
                )

        publisher = asyncio.create_task(_publish())
        try:
            sse = await client.get(
                "/events/stream",
                params={"token": grant.json()["token"], "max_events": 1, "max_seconds": 3},
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert sse.status_code == 200
        assert '"topic":"run.finished"' in sse.text
        assert token not in sse.text
        assert digest in sse.text

        exported = await client.post(
            f"/conversations/{conversation_id}/export?format=json&redact=true",
            headers=headers,
        )
        assert exported.status_code == 200, exported.text
        raw = Path(exported.json()["path"]).read_bytes()
        assert token.encode() not in raw
        assert digest.encode() in raw


async def test_export_rate_limit_is_durable_and_shared_across_workers(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing in-process state must not reset a kv-backed export window."""
    from fastapi import HTTPException

    from leo import db

    monkeypatch.setattr(api_module, "EXPORT_RATE_LIMIT", 2)
    api_module._export_requests.clear()  # the legacy cache is not authoritative
    principal = "durable-principal"
    await api_module._check_export_rate(principal)
    await api_module._check_export_rate(principal)
    with pytest.raises(HTTPException) as denied:
        await api_module._check_export_rate(principal)
    assert denied.value.status_code == 429

    kv_key = api_module._export_rate_kv_key(principal)
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (kv_key,))
    assert row is not None and len(json.loads(row["value"])) == 2

    # A genuinely different worker (separate interpreter and module state)
    # reads the same SQLite counters and must refuse the third request.
    repo = Path(__file__).resolve().parents[1]
    worker = (
        "import asyncio, sys\n"
        "from leo import db\n"
        "from leo.interfaces import api\n"
        "db.set_db_path(sys.argv[1])\n"
        "api.EXPORT_RATE_LIMIT = 2\n"
        "asyncio.run(api._check_export_rate(sys.argv[2]))\n"
    )
    # A transient cross-process SQLite lock is environmental, not a rate-limit
    # result. Retry the whole worker on timeout/lock so the assertion still
    # pins the real durable denial without accepting a slow hang as success.
    result = None
    for attempt in range(3):
        try:
            candidate = subprocess.run(
                [sys.executable, "-c", worker, str(db.get_db_path()), principal],
                cwd=str(repo), env={**os.environ, "PYTHONPATH": str(repo)},
                capture_output=True, text=True, timeout=20,
            )
        except subprocess.TimeoutExpired:
            candidate = None
        if candidate is not None and "rate limit" in candidate.stderr:
            result = candidate
            break
        if candidate is not None and "locked" not in candidate.stderr.lower():
            result = candidate  # real failure: let the assertion report it
            break
        await asyncio.sleep(0.2 * (attempt + 1))
    assert result is not None
    assert result.returncode != 0 and "rate limit" in result.stderr

    # Buckets are per principal, and expired kv counters are pruned.
    await api_module._check_export_rate("another-principal")
    stale_key = api_module._export_rate_kv_key("stale-principal")
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)",
        (stale_key, "[]", "2000-01-01T00:00:00+00:00"),
    )
    with pytest.raises(HTTPException):
        await api_module._check_export_rate(principal)
    assert await db.fetchone("SELECT 1 FROM kv WHERE key=?", (stale_key,)) is None


async def test_fake_digest_prefix_is_scrubbed_on_every_browser_surface(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``sha256:<raw token>`` must not become a replayable pseudo-digest."""
    import asyncio

    from starlette.requests import Request as StarletteRequest

    from leo import db
    from leo.contracts import utcnow

    fake = "sha256:rawReplayToken_AbCdEf1234567890"
    raw = fake.split(":", 1)[1]
    offline_app.settings.paths.exports = tmp_path / "exports"

    async def _connected(self: Any) -> bool:
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", _connected)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post(
            "/chat", json={"message": "fake digest fixture"}
        )).json()["conversation_id"]
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, "user", f"correlation_id={fake}", "{}", utcnow()),
        )
        public = await client.get(f"/conversations/{conversation_id}")
        assert public.status_code == 200
        assert fake not in public.text and raw not in public.text
        assert "sha256:" in public.text

        await offline_app.events.emit(
            "run.finished", correlation_id=fake, detail=f"see {fake}"
        )
        feed = await client.get("/events?topic=run.finished")
        assert feed.status_code == 200
        assert fake not in feed.text and raw not in feed.text

        grant = await client.post("/events/stream-token")
        assert grant.status_code == 200, grant.text

        async def _publish() -> None:
            for _ in range(30):
                await asyncio.sleep(0.05)
                await offline_app.events.emit(
                    "run.finished", correlation_id=fake, detail=f"see {fake}"
                )

        publisher = asyncio.create_task(_publish())
        try:
            sse = await client.get(
                "/events/stream",
                params={"token": grant.json()["token"], "max_events": 1, "max_seconds": 3},
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert sse.status_code == 200
        assert '"topic":"run.finished"' in sse.text
        assert fake not in sse.text and raw not in sse.text

        exported = await client.post(
            f"/conversations/{conversation_id}/export?format=json&redact=true"
        )
        assert exported.status_code == 200, exported.text
        body = Path(exported.json()["path"]).read_bytes()
        assert fake.encode() not in body and raw.encode() not in body


# --- residual review: cross-surface redaction and concurrent rate limits -----
def _browser_unescape(text: str, rounds: int = 16) -> str:
    """Decode entities repeatedly, as a browser viewer could."""
    import html

    value = text
    for _ in range(rounds):
        decoded = html.unescape(value)
        if decoded == value:
            break
        value = decoded
    return value


async def test_known_prefix_tokens_are_scrubbed_on_every_browser_surface(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A snake_case provider credential must not survive on any browser feed."""
    import asyncio

    from starlette.requests import Request as StarletteRequest

    from leo import db
    from leo.contracts import utcnow

    token = "ghp_abcdef1234567890"
    encoded = token.replace("_", "&#95;")  # named/numeric spelling of the same token
    offline_app.settings.paths.exports = tmp_path / "exports"

    async def _connected(self: Any) -> bool:
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", _connected)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post(
            "/chat", json={"message": "known prefix fixture"}
        )).json()["conversation_id"]
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, "user", f"correlation_id={token} and {encoded}", "{}", utcnow()),
        )
        public = await client.get(f"/conversations/{conversation_id}")
        assert public.status_code == 200
        assert token not in public.text and encoded not in public.text
        assert token not in _browser_unescape(public.text)

        await offline_app.events.emit(
            "run.finished", correlation_id=token, detail=f"see {encoded}"
        )
        feed = await client.get("/events?topic=run.finished")
        status = await client.get("/status")
        assert feed.status_code == status.status_code == 200
        for label, text in (("events", feed.text), ("status", status.text)):
            assert token not in text and encoded not in text, label
            assert token not in _browser_unescape(text), label

        grant = await client.post("/events/stream-token")
        assert grant.status_code == 200, grant.text

        async def _publish() -> None:
            for _ in range(40):
                await asyncio.sleep(0.05)
                await offline_app.events.emit(
                    "run.finished", correlation_id=token, detail=f"see {encoded}"
                )

        publisher = asyncio.create_task(_publish())
        try:
            sse = await client.get(
                "/events/stream",
                params={"token": grant.json()["token"], "max_events": 1, "max_seconds": 3},
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert sse.status_code == 200
        assert '"topic":"run.finished"' in sse.text
        assert token not in sse.text and encoded not in sse.text
        assert token not in _browser_unescape(sse.text)

        exported = await client.post(f"/conversations/{conversation_id}/export?format=json")
        assert exported.status_code == 200, exported.text
        body = Path(exported.json()["path"]).read_bytes().decode("utf-8")
        assert token not in body and encoded not in body
        assert token not in _browser_unescape(body)


async def test_entity_encoded_and_spoofed_digest_tokens_are_scrubbed_on_surfaces(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hidden HTML spellings and invented 16-hex digests are not safe data."""
    import asyncio

    from starlette.requests import Request as StarletteRequest

    from leo import db
    from leo.contracts import utcnow

    token = "surfaceEntityToken_AbCdEf1234567890"
    nested = "".join("&amp;amp;#" + str(ord(char)) + ";" for char in token)
    spoof = "sha256:0123456789abcdef"
    offline_app.settings.paths.exports = tmp_path / "exports"

    async def _connected(self: Any) -> bool:
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", _connected)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post(
            "/chat", json={"message": "entity surface fixture"}
        )).json()["conversation_id"]
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, "user", f"token_id: {nested}; ref {spoof}", "{}", utcnow()),
        )
        public = await client.get(f"/conversations/{conversation_id}")
        assert public.status_code == 200
        for label, text in (("conversations", public.text),):
            assert token not in text and token not in _browser_unescape(text), label
            assert spoof not in text, label

        await offline_app.events.emit("run.finished", correlation_id=nested, detail=spoof)
        feed = await client.get("/events?topic=run.finished")
        status = await client.get("/status")
        for label, text in (("events", feed.text), ("status", status.text)):
            assert token not in text and token not in _browser_unescape(text), label
            assert spoof not in text, label

        grant = await client.post("/events/stream-token")
        assert grant.status_code == 200, grant.text

        async def _publish() -> None:
            for _ in range(40):
                await asyncio.sleep(0.05)
                await offline_app.events.emit("run.finished", correlation_id=nested, detail=spoof)

        publisher = asyncio.create_task(_publish())
        try:
            sse = await client.get(
                "/events/stream",
                params={"token": grant.json()["token"], "max_events": 1, "max_seconds": 3},
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert sse.status_code == 200
        assert token not in sse.text and token not in _browser_unescape(sse.text)
        assert spoof not in sse.text

        exported = await client.post(f"/conversations/{conversation_id}/export?format=json")
        assert exported.status_code == 200, exported.text
        body = Path(exported.json()["path"]).read_bytes().decode("utf-8")
        assert token not in body and token not in _browser_unescape(body)
        assert spoof not in body


async def test_export_rate_cap_is_fixed_under_concurrent_processes(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A burst spanning separate interpreters gets one shared fixed cap."""
    from leo import db

    limit = 3
    monkeypatch.setattr(api_module, "EXPORT_RATE_LIMIT", limit)
    principal = "concurrent-rate-" + uuid.uuid4().hex
    repo = Path(__file__).resolve().parents[1]
    worker = (
        "import asyncio, sys\n"
        "from leo import db\n"
        "from leo.interfaces import api\n"
        "db.set_db_path(sys.argv[1])\n"
        "api.EXPORT_RATE_LIMIT = 3\n"
        "try:\n"
        "    asyncio.run(api._check_export_rate(sys.argv[2]))\n"
        "    print('OK')\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(db.get_db_path()), principal],
            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(repo)},
        )
        for _ in range(8)
    ]
    outputs = [process.communicate(timeout=60) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    lines = [stdout.strip().splitlines()[-1] for stdout, _ in outputs]
    assert lines.count("OK") == limit, lines
    assert lines.count("HTTPException") == len(processes) - limit, lines
    row = await db.fetchone(
        "SELECT value FROM kv WHERE key=?", (api_module._export_rate_kv_key(principal),)
    )
    assert row is not None and len(json.loads(row["value"])) == limit


# --- adversarial follow-up: blocking chat, tool-execute quota, audit fields ----
async def test_blocking_chat_sanitizes_model_output_for_browser_principals(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /chat is as browser-readable as /chat/stream and must redact too."""
    from leo.contracts import ModelResponse

    token = "ghp_abcdef1234567890"

    async def complete(request: Any) -> ModelResponse:
        return ModelResponse(
            content=f"the capability is {token}; keep it safe",
            model="mock", provider="mock",
        )

    monkeypatch.setattr(offline_app.router, "complete", complete)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        public = await client.post("/chat", json={"message": "show me the token"})
        assert public.status_code == 200, public.text
        assert token not in public.text
        assert "REDACTED" in public.text or "sha256:" in public.text
        assert public.json()["status"] == "completed"

        # An authenticated API-token principal keeps the unsanitized read.
        monkeypatch.setenv("LEO_API_TOKEN", "chat-read-token")
        privileged = await client.post(
            "/chat", json={"message": "show me the token"},
            headers={"authorization": "Bearer chat-read-token"},
        )
        assert privileged.status_code == 200, privileged.text
        assert token in privileged.text


async def test_export_via_tool_execute_shares_the_durable_rate_limit(
    offline_app: LeoApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool-execute route cannot multiply the HTTP export quota."""
    from leo import db
    from leo.contracts import utcnow

    offline_app.settings.paths.exports = tmp_path / "exports"
    monkeypatch.setattr(api_module, "EXPORT_RATE_LIMIT", 2)
    conversation_id = "conv-tool-export-rate"
    stamp = utcnow()
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at) "
        "VALUES(?,?,?,?,?)", (conversation_id, "api", "rate fixture", stamp, stamp),
    )
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
        "VALUES(?,?,?,?,?)", (conversation_id, "user", "rate fixture text", "{}", stamp),
    )

    transport = httpx.ASGITransport(app=api_module.api)
    payload = {"arguments": {"conversation_id": conversation_id}}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for index in range(2):
            accepted = await client.post("/tools/conversation_export/execute", json=payload)
            assert accepted.status_code == 200, (index, accepted.text)
            assert accepted.json()["success"] is True
        limited = await client.post("/tools/conversation_export/execute", json=payload)
        assert limited.status_code == 429 and "rate limit" in limited.text
        # The HTTP export path reads the same durable bucket, not a second one.
        http_limited = await client.post(f"/conversations/{conversation_id}/export")
        assert http_limited.status_code == 429

    row = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        (api_module._export_rate_kv_key("local"),),
    )
    assert row is not None and len(json.loads(row["value"])) == 2
    files = [
        path for path in (tmp_path / "exports").iterdir()
        if path.name != api_module.conversation_tools._EXPORT_LOCK_NAME
    ]
    assert len(files) == 2


async def test_tool_execute_errors_and_results_are_sanitized_for_browsers(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither HTTP error details nor audited result reasons may leak tokens."""
    token = "ref_ghp_abcdef1234567890"
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        mismatch = await client.post(
            "/tools/conversation_messages/execute",
            json={"name": token, "arguments": {}},
        )
        assert mismatch.status_code == 400
        assert token not in mismatch.text
        assert "does not match" in mismatch.text

        failed = await client.post(
            "/tools/conversation_messages/execute",
            json={"arguments": {"conversation_id": token}},
        )
        assert failed.status_code == 200
        assert failed.json()["success"] is False
        assert token not in failed.text

        monkeypatch.setenv("LEO_API_TOKEN", "tool-read-token")
        headers = {"authorization": "Bearer tool-read-token"}
        raw_mismatch = await client.post(
            "/tools/conversation_messages/execute",
            json={"name": token, "arguments": {}}, headers=headers,
        )
        assert raw_mismatch.status_code == 400 and token in raw_mismatch.text
        raw_failed = await client.post(
            "/tools/conversation_messages/execute",
            json={"arguments": {"conversation_id": token}}, headers=headers,
        )
        assert raw_failed.status_code == 200 and token in raw_failed.text


async def test_unauthenticated_audit_sanitizes_every_string_field(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capability planted in reason/resource/metadata is not a safe audit row."""
    from leo import db
    from leo.contracts import utcnow

    token = "ref_ghp_abcdef1234567890"
    reason = f"denied {token}; see http://127.0.0.1:18765/preview/{token}"
    metadata = json.dumps({"token_id": token, "note": f"also {token}"})
    await db.execute(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (utcnow(), token, "tool.execute", token, "high", "denied", reason, metadata),
    )

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        public = await client.get("/audit")
        assert public.status_code == 200
        assert token not in public.text
        row = public.json()["audit"][0]
        assert "token_id" not in json.loads(row["metadata"])
        assert row["reason"] != reason and row["resource"] != token

        monkeypatch.setenv("LEO_API_TOKEN", "audit-read-token")
        privileged = await client.get(
            "/audit", headers={"authorization": "Bearer audit-read-token"}
        )
        assert privileged.status_code == 200 and token in privileged.text


@pytest.mark.asyncio
async def test_model_default_switch_is_live_and_persisted(
    offline_app: LeoApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clicking a model in the dashboard must switch the live default and stick.

    The config rewrite is scalar-only: comments and YAML anchors elsewhere in
    the catalog must survive a selection.
    """
    config = tmp_path / "leo.yaml"
    config.write_text(
        "# keep this comment\n"
        "models:\n"
        "  default: mock\n"
        "  providers:\n"
        "    mock:\n"
        "      kind: mock\n"
        "      model: mock-1\n"
        "    mock-2:\n"
        "      kind: mock\n"
        "      model: mock-2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LEO_CONFIG", str(config))
    offline_app.settings.models.providers["mock-2"] = ModelProviderConfig(
        kind="mock", model="mock-2"
    )

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        switched = await client.post("/models/default", json={"provider": "mock-2"})
        assert switched.status_code == 200, switched.text
        body = switched.json()
        assert body["default_provider"] == "mock-2"
        assert body["model"] == "mock-2"
        assert body["persisted"] is True
        assert offline_app.settings.models.default == "mock-2"
        assert (await client.get("/health")).json()["default_provider"] == "mock-2"

        unknown = await client.post("/models/default", json={"provider": "no-such-model"})
        assert unknown.status_code == 404

    text = config.read_text(encoding="utf-8")
    assert "  default: mock-2" in text
    assert "# keep this comment" in text
    assert config.with_name("leo.yaml.bak").exists()


async def test_command_catalog_uses_the_chat_dispatch_surface(offline_app: LeoApp) -> None:
    """The composer palette must offer exactly what /chat can dispatch."""
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/commands")
    assert response.status_code == 200
    rows = response.json()["commands"]
    by_name = {row["name"]: row for row in rows}
    assert {"help", "commands", "model", "tools"} <= set(by_name)
    # /chat dispatches as principal "chat" on surface "chat"; the catalog must
    # answer the same question rather than trust the HTTP caller's principal.
    assert by_name["help"]["allowed_for_caller"] is True
    assert by_name["approve"]["allowed_for_caller"] is False
    assert by_name["approve"]["operator_only"] is True


@pytest.mark.asyncio
async def test_stop_while_a_run_is_active_is_not_rejected(offline_app: LeoApp) -> None:
    """Typed /stop on the dashboard must reach the turn it is meant to stop.

    The normal stream guard owns one run per conversation; an interrupt-class
    command is dispatched out of band and must not be refused with 409.
    """
    conversation_id = "conv-busy"
    offline_app._active_conversations.add(conversation_id)
    transport = httpx.ASGITransport(app=api_module.api)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            stopped = await client.post(f"/conversations/{conversation_id}/stop")
            assert stopped.status_code == 200
            assert stopped.json()["stopping"] is True
            assert await offline_app._turn_stop_requested(
                conversation_id, "2000-01-01T00:00:00+00:00"
            )

            ordinary = await client.post(
                "/chat/stream", json={"message": "hello", "conversation_id": conversation_id}
            )
            assert ordinary.status_code == 409

            interrupt = await client.post(
                "/chat/stream", json={"message": "/stop", "conversation_id": conversation_id}
            )
            assert interrupt.status_code == 200
            assert offline_app._active_conversations == {conversation_id}
    finally:
        offline_app._active_conversations.discard(conversation_id)


@pytest.mark.asyncio
async def test_chat_route_identity_wins_over_body_fields(offline_app: LeoApp) -> None:
    """A request body cannot choose the principal or surface of a chat turn."""
    captured: dict = {}

    async def fake_ask(text, channel="cli", subject=None, conversation_id=None,
                       attachments=None, *, turn=None):
        captured["turn"] = turn
        return {
            "conversation_id": "conv_identity",
            "content": "ok",
            "status": "completed",
            "provider": "", "model": "",
        }

    offline_app.ask = fake_ask  # type: ignore[method-assign]
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={
            "message": "hello",
            # Routing hints and forged authority fields; none may become authority.
            "channel": "telegram",
            "subject": "body-subject",
            "principal": "operator",
            "surface": "tui",
        })
    assert response.status_code == 200
    turn = captured["turn"]
    assert turn is not None
    assert turn.principal == "local"          # authenticated route identity
    assert turn.surface == "api"              # blocking route is incapable
    assert turn.source == "api"
    assert turn.channel == "telegram"         # display/routing hint only
    assert turn.capabilities.can_present_card is False
    assert turn.capabilities.can_wait_for_decision is False


@pytest.mark.asyncio
async def test_api_token_principal_wins_over_body_fields(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_ask(text, channel="cli", subject=None, conversation_id=None,
                       attachments=None, *, turn=None):
        captured["turn"] = turn
        return {
            "conversation_id": "conv_token",
            "content": "ok",
            "status": "completed",
            "provider": "", "model": "",
        }

    offline_app.ask = fake_ask  # type: ignore[method-assign]
    monkeypatch.setenv("LEO_API_TOKEN", "operator-token")
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test",
        headers={"Authorization": "Bearer operator-token"},
    ) as client:
        response = await client.post("/chat", json={
            "message": "hello", "principal": "forged", "surface": "web",
        })
    assert response.status_code == 200
    assert captured["turn"].principal == "api-token"
    assert captured["turn"].surface == "api"


@pytest.mark.asyncio
async def test_tool_route_identity_wins_over_body_fields(offline_app: LeoApp) -> None:
    """The HTTP executor route records its own actor/principal, not body values."""
    captured: dict = {}

    async def fake_execute(call, approval_id=None):
        captured["call"] = call
        from leo.contracts import ToolResult

        return ToolResult(call_id=call.id, tool=call.name, success=True, output={})

    offline_app.executor.execute = fake_execute  # type: ignore[method-assign]
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/tools/calculator/execute", json={
            "arguments": {"expression": "1+1"},
            "principal": "operator",
            "actor": "operator",
            "source": "chat",
        })
    assert response.status_code == 200
    call = captured["call"]
    assert call.actor == "api"
    assert call.principal == "local"
    assert call.source == "api"
    assert call.capabilities["can_present_card"] is False


@pytest.mark.asyncio
async def test_stream_route_uses_the_authenticated_web_surface(offline_app: LeoApp) -> None:
    """The authenticated web/Electron stream is the capable dashboard surface,
    while a body channel remains only a routing hint."""
    captured: dict = {}

    async def fake_stream(*args, **kwargs):
        captured["turn"] = kwargs.get("turn")
        yield {"type": "run.finished", "status": "completed", "seq": 0}

    offline_app.ask_stream = fake_stream  # type: ignore[method-assign]
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/chat/stream", json={
            "message": "hello", "channel": "telegram",
        }) as response:
            assert response.status_code == 200
            async for _line in response.aiter_lines():
                pass
    turn = captured["turn"]
    assert turn is not None
    assert turn.principal == "local"
    assert turn.surface == "web"
    assert turn.channel == "telegram"
    assert turn.capabilities.can_present_card is True
