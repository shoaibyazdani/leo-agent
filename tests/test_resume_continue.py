"""A continuation prompt resumes the last objective instead of scanning history."""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ModelResponse, Usage
from leo.core import LeoApp


class _CaptureRouter:
    def __init__(self, content: str = "done") -> None:
        self.content = content
        self.requests: list[object] = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            content=self.content, model="fake", provider="fake", usage=Usage(estimated=True)
        )


def _settings() -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    return settings


@pytest.fixture
async def app():
    instance = LeoApp(_settings())
    await instance.initialize()
    yield instance
    await instance.shutdown()


async def _seed(conversation_id: str, rows: list[tuple[str, str, dict]]) -> list[int]:
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at)"
        " VALUES(?,?,?,?,?)",
        (conversation_id, "telegram", "525", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    ids: list[int] = []
    for role, content, metadata in rows:
        cursor = await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at)"
            " VALUES(?,?,?,?,?)",
            (conversation_id, role, content, json.dumps(metadata), "2026-01-01T00:00:00+00:00"),
        )
        ids.append(int(cursor.lastrowid))
    return ids


@pytest.mark.asyncio
async def test_continue_resumes_the_last_objective(app):
    conversation_id = "conv-resume"
    await _seed(
        conversation_id,
        [
            ("user", "implement the whole parser end to end", {}),
            ("assistant", "working", {"run_status": "incomplete", "tool_calls_pending": ["shell"]}),
        ],
    )
    router = _CaptureRouter()
    app.router = router

    result = await app.ask(
        "continue", channel="telegram", subject="525", conversation_id=conversation_id
    )

    assert result["content"] == "done"
    sent = router.requests[0].messages
    last_user = next(m for m in reversed(sent) if m.role == "user")
    assert "RESUME" in last_user.content
    assert "implement the whole parser end to end" in last_user.content
    assert "tool_calls_pending" not in last_user.content
    assert "shell" in last_user.content

    row = await db.fetchone(
        "SELECT content, metadata FROM messages WHERE conversation_id=? AND role='user' "
        "ORDER BY id DESC LIMIT 1",
        (conversation_id,),
    )
    assert row["content"] == "continue", "the operator's literal text stays durable"
    metadata = json.loads(row["metadata"])
    assert metadata["resume"] is True and metadata["objective_message_id"]


@pytest.mark.asyncio
async def test_continue_without_an_objective_does_not_invent_one(app):
    conversation_id = "conv-no-objective"
    await _seed(conversation_id, [])
    router = _CaptureRouter()
    app.router = router

    await app.ask(
        "continue", channel="telegram", subject="525", conversation_id=conversation_id
    )

    last_user = next(m for m in reversed(router.requests[0].messages) if m.role == "user")
    assert last_user.content == "continue"


@pytest.mark.asyncio
async def test_slash_rows_are_not_resumed(app):
    conversation_id = "conv-slash-skip"
    await _seed(
        conversation_id,
        [
            ("user", "an old objective", {}),
            ("assistant", "ok", {"run_status": "completed"}),
            ("user", "/model", {"slash": True}),
        ],
    )
    router = _CaptureRouter()
    app.router = router

    await app.ask(
        "continue", channel="telegram", subject="525", conversation_id=conversation_id
    )

    last_user = next(m for m in reversed(router.requests[0].messages) if m.role == "user")
    assert "an old objective" in last_user.content, "the slash row is skipped"
