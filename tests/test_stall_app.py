"""One bounded nudge when a turn announces work and stops."""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ModelResponse, Usage
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
