"""App-level compaction wiring: auto threshold, /compact command, replay view."""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ModelResponse, Usage
from leo.core import LeoApp
from leo.models import compaction


class StubRouter:
    """Answers conversation turns and compaction requests with fixed text."""

    def __init__(self, summary: str = "earlier: the operator chose postgres") -> None:
        self.summary = summary
        self.requests: list[object] = []

    async def complete(self, request):
        self.requests.append(request)
        system = " ".join(
            m.content for m in request.messages if getattr(m, "role", "") == "system"
        )
        content = self.summary if "compress a conversation transcript" in system else "answer"
        return ModelResponse(
            content=content, model="stub", provider="stub", usage=Usage(estimated=True)
        )


def _settings(**compaction_kwargs) -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    for key, value in compaction_kwargs.items():
        setattr(settings.compaction, key, value)
    return settings


@pytest.fixture
async def app(tmp_path, monkeypatch):
    monkeypatch.setenv("LEO_HOME", str(tmp_path / "home"))
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    instance = LeoApp(_settings())
    await instance.initialize()
    yield instance
    await instance.shutdown()


async def _history(app: LeoApp, conversation_id: str, turns: int, filler: str = "detail " * 30):
    for index in range(turns):
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at)"
            " VALUES(?,?,?,?,?)",
            (conversation_id, "user", f"question {index}: {filler}", "{}", "2026-01-01T00:00:00+00:00"),
        )
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at)"
            " VALUES(?,?,?,?,?)",
            (conversation_id, "assistant", f"answer {index}", "{}", "2026-01-01T00:00:00+00:00"),
        )


async def _conversation(app: LeoApp, conversation_id: str) -> None:
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at)"
        " VALUES(?,?,?,?,?)",
        (conversation_id, "cli", "t", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )


@pytest.mark.asyncio
async def test_compact_conversation_persists_and_replays_summary(app):
    conversation_id = "conv-app-1"
    await _conversation(app, conversation_id)
    await _history(app, conversation_id, 12)
    stub = StubRouter()
    app.router = stub
    app.compaction.config = compaction.CompactionConfig(
        keep_recent_tokens=50, memory_flush=False
    )

    data = await app.compact_conversation(conversation_id)

    assert data["ok"], data.get("reason")
    boundary = await compaction.latest(conversation_id)
    assert boundary is not None
    # Replay view starts with the durable summary, then post-boundary history.
    replay = await app._conversation_history(conversation_id)
    assert replay[0].role == "system"
    assert replay[0].name == compaction.SUMMARY_NAME
    assert "postgres" in replay[0].content


@pytest.mark.asyncio
async def test_compact_preview_does_not_commit(app):
    conversation_id = "conv-app-preview"
    await _conversation(app, conversation_id)
    await _history(app, conversation_id, 10)
    app.router = StubRouter()
    app.compaction.config = compaction.CompactionConfig(
        keep_recent_tokens=50, memory_flush=False
    )

    data = await app.compact_conversation(conversation_id, preview=True)

    assert data["ok"]
    assert data.get("summary")
    assert await compaction.latest(conversation_id) is None


@pytest.mark.asyncio
async def test_compact_conversation_reports_empty_history(app):
    data = await app.compact_conversation("conv-empty")
    assert data["ok"] is False
    assert "no messages" in data["reason"]


@pytest.mark.asyncio
async def test_auto_compaction_triggers_above_threshold(app):
    conversation_id = "conv-app-auto"
    await _conversation(app, conversation_id)
    await _history(app, conversation_id, 20)
    app.router = StubRouter()
    app.compaction.config = compaction.CompactionConfig(
        keep_recent_tokens=50, memory_flush=False, threshold=0.0001
    )

    messages = await app._conversation_history(conversation_id)
    result = await app._maybe_auto_compact(conversation_id, messages, budget=10_000)

    assert await compaction.latest(conversation_id) is not None
    assert result and result[0].name == compaction.SUMMARY_NAME


@pytest.mark.asyncio
async def test_auto_compaction_is_skipped_when_disabled(app):
    conversation_id = "conv-app-off"
    await _conversation(app, conversation_id)
    await _history(app, conversation_id, 20)
    app.router = StubRouter()
    app.compaction.config = compaction.CompactionConfig(
        keep_recent_tokens=50, enabled=False
    )

    messages = await app._conversation_history(conversation_id)
    result = await app._maybe_auto_compact(conversation_id, messages, budget=10_000)

    assert await compaction.latest(conversation_id) is None
    assert result == messages


@pytest.mark.asyncio
async def test_auto_compaction_failure_does_not_break_the_turn(app):
    conversation_id = "conv-app-broken"
    await _conversation(app, conversation_id)
    await _history(app, conversation_id, 20)

    class BrokenRouter:
        async def complete(self, request):
            raise RuntimeError("provider down")

    app.router = BrokenRouter()
    app.compaction.config = compaction.CompactionConfig(
        keep_recent_tokens=50, memory_flush=False, threshold=0.0001, on_timeout="keep_history"
    )

    messages = await app._conversation_history(conversation_id)
    result = await app._maybe_auto_compact(conversation_id, messages, budget=10_000)

    # History is preserved verbatim and the turn still has messages to send.
    assert await compaction.latest(conversation_id) is None
    assert result == messages


@pytest.mark.asyncio
async def test_slash_compact_command_runs_and_audits(app):
    conversation_id = "conv-app-cmd"
    await _conversation(app, conversation_id)
    await _history(app, conversation_id, 12)
    app.router = StubRouter()
    app.compaction.config = compaction.CompactionConfig(
        keep_recent_tokens=50, memory_flush=False
    )

    result = await app.commands.dispatch(
        "/compact",
        app=app,
        principal="user",
        surface="cli",
        channel="cli",
        conversation_id=conversation_id,
    )

    assert result.success, result.message
    assert "Compacted" in result.message
    assert await compaction.latest(conversation_id) is not None
    audit = await db.fetchone(
        "SELECT action FROM audit_log WHERE action='slash.compact' ORDER BY id DESC LIMIT 1"
    )
    assert audit is not None


@pytest.mark.asyncio
async def test_slash_compact_clear_drops_boundary(app):
    conversation_id = "conv-app-clear"
    await _conversation(app, conversation_id)
    await _history(app, conversation_id, 12)
    app.router = StubRouter()
    app.compaction.config = compaction.CompactionConfig(
        keep_recent_tokens=50, memory_flush=False
    )
    await app.compact_conversation(conversation_id)
    assert await compaction.latest(conversation_id) is not None

    result = await app.commands.dispatch(
        "/compact --clear",
        app=app,
        principal="user",
        surface="cli",
        channel="cli",
        conversation_id=conversation_id,
    )

    assert result.success
    assert await compaction.latest(conversation_id) is None
    # The full history replays again.
    replay = await app._conversation_history(conversation_id)
    assert all(m.name != compaction.SUMMARY_NAME for m in replay)


@pytest.mark.asyncio
async def test_compact_is_listed_as_available_not_unavailable(app):
    catalog = app.commands.catalog(principal="user", surface="chat")
    entry = next(item for item in catalog if item.get("name") == "compact")
    assert entry["available"] is True
    assert "/compact" in entry["help"] or "Summarize" in entry["help"]
