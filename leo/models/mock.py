"""Deterministic, offline model provider used by tests and local-only runs."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import AsyncIterator

from ..config import ModelProviderConfig
from ..contracts import ModelRequest, ModelResponse, Usage
from ..logging import get_logger
from .provider import estimate_request_tokens, usage_from_counts

log = get_logger("models.mock")


class MockProvider:
    """Offline provider: identical requests always produce identical output.

    Optional ``responses`` are consumed in order for tests that need distinct
    turns; after the sequence is exhausted the deterministic fallback applies.
    """

    style = "mock"

    def __init__(
        self,
        config: ModelProviderConfig | None = None,
        name: str = "mock",
        responses: Sequence[str] | None = None,
    ) -> None:
        self.config = config or ModelProviderConfig(kind="mock", model="mock-1")
        self.name = name
        self.model = self.config.model or "mock-1"
        self.capabilities: frozenset[str] = frozenset(self.config.capabilities)
        self.context_window = self.config.context_window
        self.last_usage: Usage | None = None
        self.last_finish_reason: str | None = "stop"
        self._responses = list(responses or [])
        self._response_index = 0

    def __repr__(self) -> str:
        return f"MockProvider(name={self.name!r}, model={self.model!r})"

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def _content(self, request: ModelRequest) -> str:
        if self._response_index < len(self._responses):
            content = self._responses[self._response_index]
        else:
            prompt = next(
                (m.content for m in reversed(request.messages) if m.role == "user"),
                "",
            )
            content = f"mock: {prompt[:200]}"
        self._response_index += 1
        if request.response_format is not None:
            # Deterministic JSON for structured-output tests.
            return json.dumps({"result": content}, ensure_ascii=False)
        return content

    def _usage(self, request: ModelRequest, content: str, latency_ms: float) -> Usage:
        # No provider reports tokens here: both counts are explicit estimates.
        usage = usage_from_counts(
            estimate_request_tokens(request),
            None,
            self.config,
            request,
            content,
            latency_ms,
        )
        usage.output_tokens = max(1, (len(content) + 3) // 4)
        usage.total_tokens = usage.input_tokens + usage.output_tokens
        usage.cost = (
            usage.input_tokens / 1_000_000 * self.config.cost_per_million_input
            + usage.output_tokens / 1_000_000 * self.config.cost_per_million_output
        )
        usage.estimated = True
        return usage

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a deterministic response with estimated usage."""
        started = time.perf_counter()
        content = self._content(request)
        usage = self._usage(request, content, (time.perf_counter() - started) * 1000)
        self.last_usage = usage
        log.debug("mock complete", extra={"provider": self.name, "task_id": request.task_id})
        return ModelResponse(
            content=content,
            model=request.model or self.model,
            provider=self.name,
            tool_calls=[],
            usage=usage,
            finish_reason="stop",
            raw={"mock": True},
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        """Yield the deterministic response in fixed-size chunks."""
        started = time.perf_counter()
        content = self._content(request)
        for offset in range(0, len(content), 16):
            yield content[offset : offset + 16]
        self.last_usage = self._usage(request, content, (time.perf_counter() - started) * 1000)
        self.last_finish_reason = "stop"

    async def aclose(self) -> None:
        """No resources to release."""


__all__ = ["MockProvider"]
