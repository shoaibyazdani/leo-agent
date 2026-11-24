"""Retry-After, jitter and truthful 429 codes at the provider boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from leo.config import ModelProviderConfig
from leo.contracts import ChatMessage, ModelRequest
from leo.models.provider import ModelHTTPError, OpenAIProvider


def _config(**overrides) -> ModelProviderConfig:
    values = dict(
        kind="openai",
        base_url="https://example.invalid/v1",
        model="m",
        max_retries=2,
        timeout_seconds=5,
    )
    values.update(overrides)
    return ModelProviderConfig(**values)


def _request() -> ModelRequest:
    return ModelRequest(messages=[ChatMessage(role="user", content="hi")])


def _response(status: int = 429, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers or {},
        json={"error": {"message": "slow down"}},
        request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
    )


def test_retry_after_parses_seconds_and_http_dates() -> None:
    assert OpenAIProvider._retry_after_seconds(_response(headers={"retry-after": "7"})) == 7.0
    later = format_datetime(
        datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True
    )
    parsed = OpenAIProvider._retry_after_seconds(_response(headers={"retry-after": later}))
    assert parsed is not None and 25 <= parsed <= 31
    assert (
        OpenAIProvider._retry_after_seconds(
            _response(headers={"x-ratelimit-reset-after": "4"})
        )
        == 4.0
    )


@pytest.mark.asyncio
async def test_backoff_honours_retry_after_and_jitter() -> None:
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    provider = OpenAIProvider(
        _config(), client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _response())),
        sleep=sleep, rand=lambda: 0.5, backoff_base=1.0, backoff_cap=10.0,
    )
    await provider._backoff(1)
    await provider._backoff(0, 3.0)

    assert sleeps == [1.0, 3.0], "jittered exponential, then the provider's own wait"
    await provider.aclose()


@pytest.mark.asyncio
async def test_429_is_typed_and_retried_then_raised() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(headers={"retry-after": "0"})

    provider = OpenAIProvider(
        _config(max_retries=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelHTTPError) as caught:
        await provider.complete(_request())

    assert calls == 2
    assert caught.value.code == "model_rate_limited"
    assert caught.value.retry_after == 0.0
    assert caught.value.throttle_scope == "key"
    await provider.aclose()


@pytest.mark.asyncio
async def test_a_long_retry_after_fails_over_immediately() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(headers={"retry-after": "120"})

    provider = OpenAIProvider(
        _config(max_retries=5, max_retry_after_seconds=60),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelHTTPError) as caught:
        await provider.complete(_request())

    assert calls == 1, "waiting past the cap would block the turn"
    assert caught.value.retry_after == 120.0
    await provider.aclose()


@pytest.mark.asyncio
async def test_x_should_retry_false_suppresses_retry_without_retry_after() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(status=500, headers={"x-should-retry": "false"})

    provider = OpenAIProvider(
        _config(max_retries=3),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelHTTPError):
        await provider.complete(_request())
    assert calls == 1
    await provider.aclose()
