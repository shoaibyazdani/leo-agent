"""models.fallback is a real tier; retryable failures cool a provider down."""

from __future__ import annotations

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ChatMessage, ModelRequest, ModelResponse, Usage
from leo.models.provider import ModelHTTPError
from leo.models.router import ModelRouter


class FakeProvider:
    def __init__(self, name: str, behavior, config: ModelProviderConfig) -> None:
        self.name = name
        self.model = name
        self.config = config
        self.capabilities = frozenset({"text", "streaming", "tools", "structured", "usage"})
        self.context_window = 128_000
        self.behavior = behavior
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if isinstance(self.behavior, BaseException):
            raise self.behavior
        return ModelResponse(
            content=f"answer from {self.name}", model=self.name,
            provider=self.name, usage=Usage(estimated=True),
        )


def _rate_limited(retry_after: float | None = 0.0) -> ModelHTTPError:
    return ModelHTTPError(429, "rate limited", retryable=True, retry_after=retry_after)


def _settings(fallback: str | None = "backup", **models) -> Settings:
    settings = Settings()
    configs = {
        name: ModelProviderConfig(kind="mock", model=name)
        for name in ("primary", "backup")
    }
    settings.models = ModelConfig(
        default="primary", fallback=fallback, providers=configs, **models
    )
    return settings


def _request() -> ModelRequest:
    return ModelRequest(messages=[ChatMessage(role="user", content="hi")])


def _router(settings: Settings, primary, backup, now) -> ModelRouter:
    return ModelRouter(
        settings,
        providers={"primary": primary, "backup": backup},
        now=now,
    )


@pytest.mark.asyncio
async def test_rate_limited_primary_fails_over_with_escalation_zero() -> None:
    settings = _settings(escalation_limit=0)
    primary = FakeProvider("primary", _rate_limited(), settings.models.providers["primary"])
    backup = FakeProvider("backup", None, settings.models.providers["backup"])
    router = _router(settings, primary, backup, lambda: 0.0)

    response = await router.complete(_request())

    assert primary.calls == 1 and backup.calls == 1
    assert response.provider == "backup"


@pytest.mark.asyncio
async def test_cooldown_skips_the_primary_on_the_next_call() -> None:
    now = {"value": 0.0}
    settings = _settings(escalation_limit=0)
    primary = FakeProvider("primary", _rate_limited(), settings.models.providers["primary"])
    backup = FakeProvider("backup", None, settings.models.providers["backup"])
    router = _router(settings, primary, backup, lambda: now["value"])

    await router.complete(_request())
    await router.complete(_request())
    assert primary.calls == 1, "the cooling primary is skipped"
    assert backup.calls == 2

    now["value"] = 10_000.0
    await router.complete(_request())
    assert primary.calls == 2, "after the cooldown the primary is tried again"


@pytest.mark.asyncio
async def test_every_tier_rate_limited_surfaces_one_truthful_code() -> None:
    settings = _settings(escalation_limit=0)
    primary = FakeProvider("primary", _rate_limited(7), settings.models.providers["primary"])
    backup = FakeProvider("backup", _rate_limited(3), settings.models.providers["backup"])
    router = _router(settings, primary, backup, lambda: 0.0)

    with pytest.raises(Exception) as caught:
        await router.complete(_request())

    assert caught.value.code == "model_rate_limited"
    assert caught.value.retry_after == 7


@pytest.mark.asyncio
async def test_a_mixed_failure_surfaces_the_last_real_error() -> None:
    settings = _settings(escalation_limit=0)
    primary = FakeProvider("primary", _rate_limited(), settings.models.providers["primary"])
    backup = FakeProvider(
        "backup",
        ModelHTTPError(503, "overloaded", retryable=True),
        settings.models.providers["backup"],
    )
    router = _router(settings, primary, backup, lambda: 0.0)

    with pytest.raises(ModelHTTPError) as caught:
        await router.complete(_request())

    assert caught.value.code == "model_upstream_error"


@pytest.mark.asyncio
async def test_fallback_equal_to_default_keeps_one_attempt() -> None:
    settings = _settings(fallback="primary", escalation_limit=0)
    primary = FakeProvider("primary", _rate_limited(), settings.models.providers["primary"])
    backup = FakeProvider("backup", None, settings.models.providers["backup"])
    router = _router(settings, primary, backup, lambda: 0.0)

    with pytest.raises(Exception) as caught:
        await router.complete(_request())

    assert caught.value.code == "model_rate_limited"
    assert primary.calls == 1 and backup.calls == 0


@pytest.mark.asyncio
async def test_stream_finish_reason_is_exposed_on_the_router() -> None:
    from leo.contracts import StreamChunk

    settings = _settings(escalation_limit=0)
    config = settings.models.providers["primary"]

    class Streaming:
        name = "primary"
        model = "primary"
        config = settings.models.providers["primary"]
        capabilities = frozenset({"text", "streaming", "tools"})
        context_window = 128_000
        last_finish_reason = "length"
        last_content = "partial answer"
        last_tool_calls = []
        last_usage = Usage(estimated=True)

        async def stream_events(self, request):
            yield StreamChunk(kind="text", text="partial answer")

    router = ModelRouter(settings, providers={"primary": Streaming()}, now=lambda: 0.0)
    chunks = [chunk async for chunk in router.stream_events(_request())]

    assert chunks and router.last_stream_finish_reason == "length"
