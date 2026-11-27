"""One bounded nudge when a turn announces work and stops."""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ModelResponse, TurnContext, Usage, utcnow
from leo.core import LeoApp


class _StallRouter:
    """First reply: a truncated plan. Second: the real answer."""

    def __init__(self, *, finish_reason: str = "length") -> None:
        self.calls = 0
        self.finish_reason = finish_reason

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="I'll now run the tests.",
                model="fake", provider="fake", finish_reason=self.finish_reason,
                usage=Usage(estimated=True),
            )
        return ModelResponse(
            content="Tests pass; the fix is live.",
            model="fake", provider="fake", usage=Usage(estimated=True),
        )


def _settings(**stall) -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    for key, value in stall.items():
        setattr(settings.stall_guard, key, value)
    return settings


async def _app(settings: Settings) -> LeoApp:
    instance = LeoApp(settings)
    await instance.initialize()
    return instance


@pytest.mark.asyncio
async def test_one_nudge_then_the_real_answer():
    app = await _app(_settings())
    router = _StallRouter()
    app.router = router
    try:
        result = await app.ask("fix it", channel="test")
    finally:
        await app.shutdown()

    assert router.calls == 2, "exactly one extra model call"
    assert result["content"] == "Tests pass; the fix is live."
    rows = await db.fetchall(
        "SELECT content, metadata FROM messages WHERE role='assistant' ORDER BY id"
    )
    interim = [row for row in rows if json.loads(row["metadata"] or "{}").get("stall_nudge")]
    assert len(interim) == 1
    assert interim[0]["content"].startswith("I'll now")
    assert rows[-1]["content"] == "Tests pass; the fix is live."


@pytest.mark.asyncio
async def test_disabled_stall_guard_takes_the_first_answer():
    app = await _app(_settings(enabled=False))
    router = _StallRouter()
    app.router = router
    try:
        result = await app.ask("fix it", channel="test")
    finally:
        await app.shutdown()

    assert router.calls == 1
    assert result["content"].startswith("I'll now")


@pytest.mark.asyncio
async def test_a_normal_answer_is_not_nudged():
    class Normal:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            return ModelResponse(
                content="The fix is live and verified.",
                model="fake", provider="fake", usage=Usage(estimated=True),
            )

    app = await _app(_settings())
    router = Normal()
    app.router = router
    try:
        await app.ask("status?", channel="test")
    finally:
        await app.shutdown()

    assert router.calls == 1


@pytest.mark.asyncio
async def test_streaming_turn_is_nudged_once():
    from leo.contracts import StreamChunk

    class StreamStall:
        def __init__(self) -> None:
            self.calls = 0
            self.last_stream_content = ""
            self.last_stream_tool_calls = []
            self.last_stream_provider = "fake"
            self.last_stream_usage = Usage(estimated=True)
            self.last_stream_finish_reason = "length"

        async def stream_events(self, request):
            self.calls += 1
            if self.calls == 1:
                self.last_stream_content = "I'll now run the tests."
                self.last_stream_finish_reason = "length"
                yield StreamChunk(kind="text", text="I'll now run the tests.")
            else:
                self.last_stream_content = "Tests pass; the fix is live."
                self.last_stream_finish_reason = "stop"
                yield StreamChunk(kind="text", text="Tests pass; the fix is live.")

    app = await _app(_settings())
    router = StreamStall()
    app.router = router
    try:
        events = [
            event async for event in app.ask_stream("fix it", channel="test")
        ]
    finally:
        await app.shutdown()

    assert router.calls == 2
    finished = [event for event in events if event.get("type") == "message.finished"]
    assert finished and finished[-1]["content"] == "Tests pass; the fix is live."


# --- issue #77: a direct answer reaches the channel; parked work stays parked ---

_PHOTO_REACTION = (
    "Cute — fluffy gray cat in a purple hedgehog cave bed, going for the pom-pom."
)


def _channel_turn() -> TurnContext:
    return TurnContext(
        principal="operator", channel="telegram", target="user-1",
        surface="telegram", origin="gateway:telegram", source="gateway",
    )


@pytest.mark.asyncio
async def test_photo_reaction_is_the_delivered_final_answer(monkeypatch):
    """A reaction with planning reasoning must not become a stall nudge."""

    class PhotoRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            assert self.calls == 1, "the reply ended the turn; no GitHub follow-up"
            return ModelResponse(
                content=_PHOTO_REACTION,
                reasoning="Let me now check the open PRs and issues.",
                model="fake", provider="fake", usage=Usage(estimated=True),
            )

    app = await _app(_settings())
    sends: list[dict] = []

    async def fake_send(channel, target, text, **kwargs):
        sends.append({"channel": channel, "target": target, "text": text, **kwargs})
        return {"success": True, "message_id": "photo-3105"}

    monkeypatch.setattr(app.gateway, "send", fake_send)
    router = PhotoRouter()
    app.router = router
    try:
        result = await app._gateway_reply({
            "channel": "telegram", "reply_target": "user-1", "text": "",
            "conversation_id": "conv-photo-77", "outbox_id": "outbox-77",
            "idempotency_key": "idem-77",
        })
    finally:
        await app.shutdown()

    assert result["delivered"] is True
    assert router.calls == 1
    assert len(sends) == 1 and sends[0]["text"] == _PHOTO_REACTION
    rows = await db.fetchall(
        "SELECT role, content, metadata FROM messages WHERE conversation_id='conv-photo-77'"
    )
    assert not [r for r in rows if r["role"] == "tool"]
    answers = [
        r for r in rows
        if r["role"] == "assistant" and not json.loads(r["metadata"] or "{}").get("stall_nudge")
    ]
    assert answers and answers[-1]["content"] == _PHOTO_REACTION


@pytest.mark.asyncio
async def test_answer_with_visible_next_steps_is_not_a_stall(monkeypatch):
    """The plan may be in the visible text; it still cannot swallow the answer."""

    class PhotoRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            return ModelResponse(
                content="Cute cat! First, I'll check the PRs.",
                model="fake", provider="fake", usage=Usage(estimated=True),
            )

    app = await _app(_settings())
    sends: list[str] = []

    async def fake_send(channel, target, text, **kwargs):
        sends.append(text)
        return {"success": True, "message_id": "visible-plan-77"}

    monkeypatch.setattr(app.gateway, "send", fake_send)
    router = PhotoRouter()
    app.router = router
    try:
        result = await app._gateway_reply({
            "channel": "telegram", "reply_target": "user-1", "text": "",
            "conversation_id": "conv-visible-plan-77", "outbox_id": "outbox-77",
            "idempotency_key": "idem-visible-plan-77",
        })
    finally:
        await app.shutdown()

    assert router.calls == 1, "no stall nudge after a delivered answer"
    assert result["delivered"] is True
    assert sends == ["Cute cat! First, I'll check the PRs."]
    rows = await db.fetchall(
        "SELECT content, metadata FROM messages "
        "WHERE conversation_id='conv-visible-plan-77' AND role='assistant'"
    )
    assert rows and rows[-1]["content"] == "Cute cat! First, I'll check the PRs."
    assert not json.loads(rows[-1]["metadata"] or "{}").get("stall_nudge")


@pytest.mark.asyncio
async def test_substantive_interim_reply_is_delivered_with_provider_id(monkeypatch):
    """If the turn must continue, the reply goes to chat before the nudge."""

    class TruncatedRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    content="Cute cat in the purple hedgehog bed",
                    model="fake", provider="fake", finish_reason="length",
                    usage=Usage(estimated=True),
                )
            return ModelResponse(
                content="Checked the tracker; #76 is the open item.",
                model="fake", provider="fake", usage=Usage(estimated=True),
            )

    app = await _app(_settings())
    sends: list[dict] = []
    conversation_id = "conv-interim-77"

    async def fake_send(channel, target, text, **kwargs):
        sends.append({"channel": channel, "target": target, "text": text, **kwargs})
        # Mirror the durable row real Gateway.send writes for a confirmed send.
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, "assistant", text, json.dumps({
                "channel": channel,
                "provider_message_id": "mid-77",
                "idempotency_key": kwargs["idempotency_key"],
            }), utcnow()),
        )
        return {"success": True, "message_id": "mid-77"}

    monkeypatch.setattr(app.gateway, "send", fake_send)
    router = TruncatedRouter()
    app.router = router
    try:
        result = await app.ask(
            "photo", channel="telegram", conversation_id=conversation_id,
            turn=_channel_turn(),
        )
    finally:
        await app.shutdown()

    assert router.calls == 2, "a genuine truncated answer still gets one nudge"
    assert len(sends) == 1
    assert sends[0]["text"] == "Cute cat in the purple hedgehog bed"
    interim = await db.fetchone(
        "SELECT metadata FROM messages WHERE conversation_id=? AND content=?",
        (conversation_id, sends[0]["text"]),
    )
    assert interim is not None
    meta = json.loads(interim["metadata"])
    assert meta["provider_message_id"] == "mid-77"
    assert meta["interim"] is True and meta["stall_nudge"] is True
    assert result["content"] == "Checked the tracker; #76 is the open item."


@pytest.mark.asyncio
async def test_undelivered_interim_answer_is_carried_into_final_text():
    """Without a channel destination, the answer is never persisted only."""

    class TruncatedRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    content="Cute cat in the purple hedgehog bed",
                    model="fake", provider="fake", finish_reason="length",
                    usage=Usage(estimated=True),
                )
            return ModelResponse(
                content="Tests pass; the fix is live.",
                model="fake", provider="fake", usage=Usage(estimated=True),
            )

    app = await _app(_settings())
    router = TruncatedRouter()
    app.router = router
    try:
        result = await app.ask("photo", channel="cli")
    finally:
        await app.shutdown()

    assert router.calls == 2
    assert "Cute cat in the purple hedgehog bed" in result["content"]
    assert "Tests pass; the fix is live." in result["content"]


@pytest.mark.asyncio
async def test_truncated_answer_with_visible_plan_is_carried_into_final_text():
    """A visible plan after the answer must not strand the undelivered reply."""

    class TruncatedRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    content="Cute cat! First, I'll check the PRs.",
                    model="fake", provider="fake", finish_reason="length",
                    usage=Usage(estimated=True),
                )
            return ModelResponse(
                content="Checked the tracker; #76 is the open item.",
                model="fake", provider="fake", usage=Usage(estimated=True),
            )

    app = await _app(_settings())
    router = TruncatedRouter()
    app.router = router
    try:
        result = await app.ask("photo", channel="cli")
    finally:
        await app.shutdown()

    assert router.calls == 2, "a genuine truncation still gets one nudge"
    assert "Cute cat! First, I'll check the PRs." in result["content"]
    assert "Checked the tracker; #76 is the open item." in result["content"]


@pytest.mark.asyncio
async def test_unrelated_message_does_not_resume_the_parked_task():
    conversation_id = "conv-parked-77"
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at) "
        "VALUES(?,?,?,?,?)",
        (conversation_id, "telegram", "user-1", utcnow(), utcnow()),
    )
    for role, content, metadata in (
        ("user", "pick up the open PR reading", {}),
        ("assistant", "Want me to pick back up on the open PR reading?",
         {"run_status": "incomplete"}),
    ):
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, role, content, json.dumps(metadata), utcnow()),
        )

    class CaptureRouter:
        def __init__(self) -> None:
            self.requests: list = []
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            self.requests.append(request)
            return ModelResponse(
                content=_PHOTO_REACTION, model="fake", provider="fake",
                usage=Usage(estimated=True),
            )

    app = await _app(_settings())
    router = CaptureRouter()
    app.router = router
    try:
        await app.ask(
            "here is my cat", channel="telegram",
            conversation_id=conversation_id, turn=_channel_turn(),
        )
    finally:
        await app.shutdown()

    assert router.calls == 1
    last_user = next(
        m for m in reversed(router.requests[0].messages) if m.role == "user"
    )
    assert "RESUME" not in last_user.content
    assert "open PR reading" not in last_user.content
    row = await db.fetchone(
        "SELECT metadata FROM messages WHERE conversation_id=? AND role='user' "
        "ORDER BY id DESC LIMIT 1", (conversation_id,),
    )
    assert json.loads(row["metadata"] or "{}").get("resume") is not True
