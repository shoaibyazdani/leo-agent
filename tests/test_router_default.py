"""Configured default provider wins over unrelated cost ties."""

from __future__ import annotations

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ChatMessage, ModelRequest
from leo.models.context import ContextOptimizer
from leo.models.mock import MockProvider
from leo.models.provider import ModelError
from leo.models.router import ModelRouter


@pytest.mark.asyncio
async def test_default_provider_is_selected_first() -> None:
    settings = Settings(
        models=ModelConfig(
            default="preferred-default",
            providers={
                "preferred-default": ModelProviderConfig(kind="mock", model="p1", cost_per_million_input=5),
                "other": ModelProviderConfig(kind="mock", model="o1", cost_per_million_input=1),
            },
        )
    )
    router = ModelRouter(settings=settings)
    request = ModelRequest(messages=[ChatMessage(role="user", content="hello")])
    assert (await router.select(request))[0] == "preferred-default"


@pytest.mark.asyncio
async def test_configured_gateway_headers_are_sent(monkeypatch) -> None:
    import httpx

    from leo.models.provider import AnthropicProvider

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(
            200,
            json={"id": "msg_1", "model": "m", "content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    monkeypatch.setenv("LEO_TEST_SESSION_HEADER", "session-value")
    config = ModelProviderConfig(
        kind="anthropic-compatible",
        base_url="https://provider.example/v1",
        api_key_env="LEO_TEST_TOKEN",
        model="m",
        extra_headers={"x-gateway": "custom"},
        extra_header_env={"x-opencode-session": "LEO_TEST_SESSION_HEADER"},
    )
    monkeypatch.setenv("LEO_TEST_TOKEN", "secret-value")
    transport = httpx.MockTransport(handler)
    provider = AnthropicProvider(config, client=httpx.AsyncClient(transport=transport))
    response = await provider.complete(ModelRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert response.content == "ok"
    assert seen["x-gateway"] == "custom"
    assert seen["x-opencode-session"] == "session-value"


class _ToggleProvider(MockProvider):
    """Streams a fixed answer with tool calls until ``fail`` is flipped."""

    def __init__(self, name: str = "p") -> None:
        super().__init__(ModelProviderConfig(kind="mock", model="m"), name=name)
        self.fail = False
        self.last_tool_calls: list[dict[str, Any]] = []
        self.last_content = ""

    async def stream(self, request: ModelRequest):  # type: ignore[no-untyped-def]
        if self.fail:
            raise ModelError("provider down")
        self.last_content = "answer"
        self.last_tool_calls = [{"id": "c1", "name": "old_tool", "arguments": {}}]
        self.last_usage = None
        for chunk in ("an", "swer"):
            yield chunk


@pytest.mark.asyncio
async def test_stream_failure_clears_stale_last_stream_state() -> None:
    provider = _ToggleProvider()
    router = ModelRouter(
        settings=Settings(
            models=ModelConfig(
                default="p",
                providers={"p": ModelProviderConfig(kind="mock", model="m")},
                escalation_limit=0,
            )
        ),
        providers={"p": provider},
    )
    request = ModelRequest(messages=[ChatMessage(role="user", content="hi")])
    assert "".join([chunk async for chunk in router.stream(request)]) == "answer"
    assert router.last_stream_provider == "p"
    assert router.last_stream_content == "answer"
    assert router.last_stream_tool_calls == [{"id": "c1", "name": "old_tool", "arguments": {}}]

    provider.fail = True
    with pytest.raises(ModelError):
        [chunk async for chunk in router.stream(request)]
    assert router.last_stream_provider is None
    assert router.last_stream_tool_calls == []
    assert router.last_stream_content == ""
    assert router.last_stream_usage is None


@pytest.mark.asyncio
async def test_stream_does_not_select_provider_without_streaming_capability() -> None:
    settings = Settings(
        models=ModelConfig(
            default="text_only",
            providers={
                "text_only": ModelProviderConfig(kind="mock", model="t", capabilities=["text", "tools"]),
                "streamer": ModelProviderConfig(kind="mock", model="s", capabilities=["text", "streaming"]),
            },
            escalation_limit=0,
        )
    )
    router = ModelRouter(settings=settings)
    request = ModelRequest(messages=[ChatMessage(role="user", content="hi")])
    assert (await router.select(request))[0] == "text_only"  # no metadata hint set

    chunks = [chunk async for chunk in router.stream(request)]
    assert "".join(chunks).startswith("mock:")
    assert router.last_stream_provider == "streamer"
    report = await router.usage_report()
    assert report[0]["provider"] == "streamer"


@pytest.mark.asyncio
async def test_escalation_limit_zero_uses_only_first_provider() -> None:
    calls: list[str] = []

    class Counting(MockProvider):
        async def complete(self, request: ModelRequest):  # type: ignore[no-untyped-def]
            calls.append(self.name)
            raise ModelError("boom")

        async def stream(self, request: ModelRequest):  # type: ignore[no-untyped-def]
            calls.append(self.name)
            raise ModelError("boom")
            yield ""

    settings = Settings(
        models=ModelConfig(
            default="a",
            providers={n: ModelProviderConfig(kind="mock", model=n) for n in ("a", "b", "c")},
            escalation_limit=0,
        )
    )
    router = ModelRouter(settings=settings, providers={n: Counting(name=n) for n in ("a", "b", "c")})
    request = ModelRequest(messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ModelError):
        await router.complete(request)
    assert calls == ["a"]
    calls.clear()
    with pytest.raises(ModelError):
        [chunk async for chunk in router.stream(request)]
    assert calls == ["a"]


def test_deduplicate_preserves_distinct_image_attachments() -> None:
    optimizer = ContextOptimizer()
    messages = [
        ChatMessage(
            role="user",
            content="what is this?",
            images=[{"type": "image", "source": {"data": "AAA"}}],
        ),
        ChatMessage(
            role="user",
            content="what is this?",
            images=[{"type": "image", "source": {"data": "BBB"}}],
        ),
    ]
    result = optimizer.deduplicate(messages)
    assert len(result) == 2
    assert result[0].images[0]["source"]["data"] == "AAA"
    assert result[1].images[0]["source"]["data"] == "BBB"


def test_compaction_keeps_last_two_turns_verbatim() -> None:
    provider = MockProvider()
    optimizer = ContextOptimizer()
    messages = [ChatMessage(role="user", content=f"turn {i}: " + "x" * 80) for i in range(6)]
    compacted = optimizer._compact(messages, budget=100, provider=provider)
    assert compacted is not None
    assert any(m.name == "compaction" for m in compacted)
    assert any(m.content.startswith("turn 4:") for m in compacted)
    assert any(m.content.startswith("turn 5:") for m in compacted)


def test_compaction_skipped_when_tool_group_blocks_safe_boundary() -> None:
    provider = MockProvider()
    optimizer = ContextOptimizer()
    messages = [
        ChatMessage(role="assistant", content="", tool_calls=[{"id": "c1", "name": "t", "arguments": {}}]),
        ChatMessage(role="user", content="middle " + "x" * 80),
        ChatMessage(role="tool", content="result " + "y" * 80, tool_call_id="c1"),
    ]
    assert optimizer._compact(messages, budget=100, provider=provider) is None


def _blocked_request() -> ModelRequest:
    return ModelRequest(messages=[ChatMessage(role="user", content="hi")])


def _clear_mock_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("LEO_ALLOW_MOCK_FALLBACK", raising=False)
    monkeypatch.delenv("LEO_MODELS_ALLOW_MOCK_FALLBACK", raising=False)


@pytest.mark.asyncio
async def test_empty_provider_set_fails_closed_without_opt_in(monkeypatch) -> None:
    _clear_mock_opt_in(monkeypatch)
    router = ModelRouter(Settings(models=ModelConfig(default="opencode-go")), providers={})
    status = router.status()
    assert status["status"] == "blocked" and status["simulated"] is False
    assert status["routable_providers"] == []
    assert router.simulated is False

    request = _blocked_request()
    with pytest.raises(ModelError, match="no live provider is routable"):
        await router.select(request)
    with pytest.raises(ModelError, match="model calls are blocked"):
        await router.complete(request)
    with pytest.raises(ModelError, match="model calls are blocked"):
        [chunk async for chunk in router.stream(request)]


@pytest.mark.parametrize(
    "config",
    [
        ModelProviderConfig(kind="anthropic-compatible", model="m"),  # no base_url
        ModelProviderConfig(kind="bogus", base_url="http://127.0.0.1:1", model="m"),
        ModelProviderConfig(
            kind="openai-compatible",
            base_url="http://127.0.0.1:1",
            api_key_env="LEO_TEST_ABSENT_TOKEN",
            model="m",
        ),
    ],
    ids=["missing-base-url", "unsupported-kind", "missing-credential"],
)
@pytest.mark.asyncio
async def test_live_build_failure_is_blocked_not_silently_mocked(config, monkeypatch) -> None:
    _clear_mock_opt_in(monkeypatch)
    monkeypatch.delenv("LEO_TEST_ABSENT_TOKEN", raising=False)
    settings = Settings(models=ModelConfig(default="gateway", providers={"gateway": config}))
    router = ModelRouter(settings)

    status = router.status()
    assert status["status"] == "blocked"
    assert status["simulated"] is False
    assert status["routable_providers"] == []
    assert len(status["build_errors"]) == 1
    assert status["build_errors"][0]["provider"] == "gateway"

    request = _blocked_request()
    with pytest.raises(ModelError) as caught:
        await router.complete(request)
    text = str(caught.value)
    assert "gateway" in text and "model calls are blocked" in text
    assert "mock:" not in text
    with pytest.raises(ModelError, match="gateway"):
        [chunk async for chunk in router.stream(request)]


@pytest.mark.asyncio
async def test_broken_live_provider_blocks_even_with_explicit_mock(monkeypatch) -> None:
    _clear_mock_opt_in(monkeypatch)
    broken = ModelProviderConfig(kind="anthropic-compatible", model="m")  # no base_url
    settings = Settings(
        models=ModelConfig(
            default="gateway",
            fallback="offline",
            providers={
                "gateway": broken,
                "offline": ModelProviderConfig(kind="mock", model="mock-1"),
            },
            escalation_limit=1,
        )
    )
    router = ModelRouter(settings)
    assert router.status()["status"] == "blocked"
    with pytest.raises(ModelError, match="failed to build"):
        await router.complete(_blocked_request())

    # The explicit opt-in turns the same configuration into a visibly
    # degraded, simulated offline mode instead of a hidden downgrade.
    settings.models.allow_mock_fallback = True
    opted_in = ModelRouter(settings)
    status = opted_in.status()
    assert status["status"] == "degraded"
    assert status["simulated"] is True
    assert status["build_errors"][0]["provider"] == "gateway"
    response = await opted_in.complete(_blocked_request())
    assert response.content.startswith("mock:")


@pytest.mark.asyncio
async def test_mock_fallback_opt_in_enables_offline_answers(monkeypatch) -> None:
    _clear_mock_opt_in(monkeypatch)
    monkeypatch.setenv("LEO_ALLOW_MOCK_FALLBACK", "true")
    router = ModelRouter(Settings(models=ModelConfig(default="opencode-go", escalation_limit=0)))
    status = router.status()
    assert status["status"] == "simulated" and status["simulated"] is True
    assert status["routable_providers"] == ["mock"]

    request = _blocked_request()
    response = await router.complete(request)
    assert response.content.startswith("mock:")
    report = await router.usage_report()
    assert report[0]["metadata"]["simulated"] is True
    assert "".join([chunk async for chunk in router.stream(request)]).startswith("mock:")
