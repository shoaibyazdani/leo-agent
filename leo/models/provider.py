"""Provider-independent model adapters for Anthropic-compatible and OpenAI-compatible APIs.

Endpoints, model identifiers, credentials (by environment-variable *name*) and
pricing come exclusively from :class:`leo.config.ModelProviderConfig`. No model
ID or endpoint is invented here, and credential values are never logged,
returned in responses, or stored in error messages.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Any, AsyncIterator, Iterable, Mapping

import httpx

from ..config import ModelProviderConfig
from ..contracts import ModelRequest, ModelResponse, StreamChunk, Usage, new_id
from ..logging import get_logger

log = get_logger("models.provider")

#: Character-based token approximation. This is an ESTIMATE, not a tokenizer:
#: it is used only when a provider does not report token counts.
CHARS_PER_TOKEN = 4
ANTHROPIC_VERSION = "2023-06-01"
RETRYABLE_STATUSES = frozenset({408, 409, 429})

# Provider fields that carry chain-of-thought / thinking text. Naming varies
# across OpenAI-compatible gateways; all of them are untrusted *data*.
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")
THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning"})


def reasoning_text(value: Any) -> list[str]:
    """Collect provider-supplied thinking strings from a field or content parts.

    Accepts plain strings, structured ``[{"type": "thinking", ...}]`` parts and
    nested mappings. Opaque payloads (Anthropic ``signature_delta`` signatures
    and ``redacted_thinking.data``) are deliberately ignored: they are not
    human-readable reasoning and must never be rendered as such.
    """
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        keys = (
            ("thinking", "reasoning", "reasoning_content", "text", "content")
            if value.get("type") in THINKING_BLOCK_TYPES
            else REASONING_KEYS
        )
        pieces: list[str] = []
        for key in keys:
            if key in value:
                pieces.extend(reasoning_text(value[key]))
        return pieces
    if isinstance(value, (list, tuple)):
        pieces = []
        for item in value:
            pieces.extend(reasoning_text(item))
        return pieces
    return []


def _openai_reasoning_parts(source: Mapping[str, Any]) -> list[str]:
    """Extract reasoning from one OpenAI-compatible delta/message mapping."""
    pieces: list[str] = []
    for key in REASONING_KEYS:
        if key in source:
            pieces.extend(reasoning_text(source[key]))
    if pieces:
        return pieces
    # Some gateways put structured thinking parts inside `content` instead of
    # a dedicated field. A plain-string `content` is answer text, never
    # reasoning, so only list-shaped content is inspected here.
    content = source.get("content")
    if isinstance(content, (list, tuple)):
        return reasoning_text(content)
    return []


def _openai_text(value: Any) -> str:
    """Normalize OpenAI-compatible content to answer text only.

    Plain strings (the common case) are returned unchanged. Structured content
    parts contribute only their ``text`` parts; thinking parts are handled by
    :func:`_openai_reasoning_parts` and never become answer text.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(
            str(part.get("text") or "")
            for part in value
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return str(value) if value else ""


def _anthropic_reasoning_parts(event: Mapping[str, Any]) -> list[str]:
    """Extract thinking text from one Anthropic-compatible stream event."""
    kind = event.get("type")
    if kind == "content_block_start":
        block = event.get("content_block")
        if isinstance(block, Mapping) and block.get("type") == "thinking":
            return reasoning_text(block)
        return []
    if kind == "content_block_delta":
        delta = event.get("delta")
        if isinstance(delta, Mapping) and delta.get("type") == "thinking_delta":
            return reasoning_text(delta.get("thinking"))
    return []


class ModelError(RuntimeError):
    """Base class for all model-provider failures.

    ``code`` is a stable, provider-independent machine-readable reason. It is
    safe to surface to API clients; the human-readable text is redacted by the
    provider and again by the composition root.
    """

    code = "model_error"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class ModelEmptyOutputError(ModelError):
    """A provider ended on the token limit without emitting any usable output."""

    code = "model_empty_output"

    def __init__(self, provider: str = "model", finish_reason: str | None = None) -> None:
        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        super().__init__(
            f"{provider}: reached the output token limit with no answer or tool call{reason}"
        )


class ModelValidationError(ModelError):
    """A provider response failed local validation."""


class ModelUnsupportedError(ModelValidationError):
    """The request needs a capability the configured provider does not advertise."""


class ModelHTTPError(ModelError):
    """An HTTP error from a configured provider endpoint."""

    def __init__(self, status_code: int, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ModelTimeoutError(ModelError):
    """A request timed out."""


# Gateways spell "ran out of output tokens" differently. These are all the
# same failure when neither answer text nor a tool call was produced.
MAX_TOKENS_REASONS: frozenset[str] = frozenset({"max_tokens", "max_output_tokens", "length"})


def empty_output_reason(finish_reason: Any, content: str, tool_calls: Any) -> str | None:
    """Return the finish reason when a response is an unusable max_tokens turn."""
    reason = str(finish_reason or "").strip().lower()
    if reason not in MAX_TOKENS_REASONS:
        return None
    if str(content or "").strip() or tool_calls:
        return None
    return reason


def estimate_tokens(text: str) -> int:
    """Estimated token count for text (characters / 4, rounded up)."""
    return math.ceil(len(text or "") / CHARS_PER_TOKEN)


def estimate_request_tokens(request: ModelRequest) -> int:
    """Estimated input tokens for a request, including tool schemas (estimate only)."""
    parts = [m.content for m in request.messages]
    parts.extend(m.get("description", "") for m in request.tools if isinstance(m, Mapping))
    parts.append(json.dumps(request.tools, ensure_ascii=False, default=str))
    return estimate_tokens("\n".join(parts))


def _safe_count(value: Any) -> int | None:
    """Coerce a provider-reported token count to a non-negative int.

    Compatible gateways sometimes report counts as strings, floats, or junk.
    Junk must degrade to an estimate rather than crash a whole response.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def usage_from_counts(
    input_tokens: int | None,
    output_tokens: int | None,
    config: ModelProviderConfig,
    request: ModelRequest | None = None,
    output_text: str = "",
    latency_ms: float = 0.0,
) -> Usage:
    """Build Usage from provider-reported counts, falling back to estimates."""
    input_tokens = _safe_count(input_tokens)
    output_tokens = _safe_count(output_tokens)
    estimated = input_tokens is None or output_tokens is None
    in_tokens = input_tokens if input_tokens is not None else (
        estimate_request_tokens(request) if request is not None else estimate_tokens(output_text)
    )
    out_tokens = output_tokens if output_tokens is not None else estimate_tokens(output_text)
    in_tokens, out_tokens = max(0, int(in_tokens)), max(0, int(out_tokens))
    cost = (
        in_tokens / 1_000_000 * config.cost_per_million_input
        + out_tokens / 1_000_000 * config.cost_per_million_output
    )
    return Usage(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_tokens=in_tokens + out_tokens,
        cost=cost,
        latency_ms=max(0.0, float(latency_ms)),
        estimated=estimated,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    """Parse a JSON object string or return a mapping as a plain dict."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {"raw": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"raw": value}
    return {}


def normalize_tool_calls(raw: Iterable[Any] | None, style: str) -> list[dict[str, Any]]:
    """Normalize provider tool calls to the canonical LEO dict shape.

    Canonical shape: ``{"id", "name", "arguments", "actor"}`` where ``arguments``
    is always a dict. Accepts Anthropic ``tool_use`` blocks and OpenAI
    ``function`` calls (including accumulated streaming fragments).
    """
    calls: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        if style == "anthropic":
            name = item.get("name")
            arguments = _as_dict(item.get("input"))
            call_id = item.get("id") or new_id("call")
        else:
            function = item.get("function")
            # Native OpenAI tool calls nest details under "function", but
            # accumulated streaming fragments (and some compatible gateways)
            # use a flat {"id", "name", "arguments"} shape. Accept both, and
            # prefer the nested values when they are present.
            flat_arguments = item.get("arguments")
            if isinstance(function, Mapping):
                arguments_raw = function.get("arguments")
                if arguments_raw is None:
                    arguments_raw = flat_arguments
            else:
                function = {}
                arguments_raw = flat_arguments
            name = function.get("name") or item.get("name")
            arguments = _as_dict(arguments_raw)
            call_id = item.get("id") or new_id("call")
        if not name:
            continue
        calls.append(
            {
                "id": str(call_id),
                "name": str(name),
                "arguments": arguments,
                "actor": "assistant",
            }
        )
    return calls


def normalize_tool_spec(spec: Mapping[str, Any], style: str) -> dict[str, Any]:
    """Convert a LEO tool definition into the provider's tool schema shape."""
    if spec.get("type") == "function" and isinstance(spec.get("function"), Mapping):
        function = spec["function"]
        if style == "openai":
            return dict(spec)
        return {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
        }
    name = str(spec.get("name", ""))
    description = str(spec.get("description", ""))
    schema = spec.get("input_schema") or spec.get("parameters") or {"type": "object", "properties": {}}
    if style == "anthropic":
        return {"name": name, "description": description, "input_schema": dict(schema)}
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": dict(schema)},
    }


def _decode_sse_data(lines: list[str]) -> list[dict[str, Any]]:
    """Decode the ``data:`` lines collected for one SSE event.

    A spec-compliant event may put one JSON value across several ``data:``
    lines; those are joined with newlines before parsing. Some compatible
    gateways instead emit one complete JSON object per line *without* blank
    event separators. Joining those produces invalid JSON, so fall back to
    decoding each line independently instead of dropping every event.
    """
    if not lines:
        return []
    payload = "\n".join(lines)
    if payload.strip() == "[DONE]":
        return []
    try:
        decoded = json.loads(payload)
    except ValueError:
        results: list[dict[str, Any]] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped == "[DONE]":
                continue
            try:
                item = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(item, dict):
                results.append(item)
        if not results:
            log.warning("ignored malformed SSE payload")
        return results
    return [decoded] if isinstance(decoded, dict) else []


async def iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield JSON payloads from an SSE response, handling split data lines."""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            for payload in _decode_sse_data(data_lines):
                yield payload
            data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        # event:, id:, retry: and comments are metadata; JSON lives in data:.
    # Some streams end without a trailing blank line; do not drop the last event.
    for payload in _decode_sse_data(data_lines):
        yield payload


class BaseHTTPProvider:
    """Shared HTTP, retry, usage, and streaming machinery for adapters.

    Subclasses provide the endpoint path, auth headers, request body, response
    parser, and SSE event handler. Retries apply only before any streamed text
    has been emitted, so a partially consumed stream is never replayed.
    """

    style = "base"
    path = ""

    def __init__(
        self,
        config: ModelProviderConfig,
        name: str | None = None,
        client: httpx.AsyncClient | None = None,
        backoff_base: float = 0.25,
        backoff_cap: float = 2.0,
    ) -> None:
        self.config = config
        self.name = name or config.model or self.style
        self.model = config.model
        self.capabilities: frozenset[str] = frozenset(config.capabilities)
        self.context_window = config.context_window
        self.max_retries = max(0, config.max_retries)
        self.timeout = config.timeout_seconds
        self.backoff_base = max(0.0, backoff_base)
        self.backoff_cap = max(0.0, backoff_cap)
        self.last_usage: Usage | None = None
        self.last_finish_reason: str | None = None
        self.last_tool_calls: list[dict[str, Any]] = []
        self.last_content: str = ""
        # Provider-supplied reasoning only. Never logged or persisted here.
        self.last_reasoning: str = ""
        self._token: str | None = None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=self.timeout)
        if not config.base_url:
            raise ModelError(f"provider {self.name!r} has no configured base_url")
        self._url = config.base_url.rstrip("/") + self.path

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    @property
    def base_url(self) -> str:
        return str(self.config.base_url or "")

    def _read_token(self) -> str | None:
        env_name = self.config.api_key_env
        if not env_name:
            return None
        token = os.environ.get(env_name)
        if not token:
            raise ModelError(f"provider {self.name!r}: credential environment variable {env_name!r} is not set")
        return token

    def _headers(self) -> dict[str, str]:
        """Build auth headers; the token never appears in logs or error text."""
        raise NotImplementedError

    def _with_extra_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Add configured gateway headers; secret values come from env by name."""
        result = dict(headers)
        for key, value in self.config.extra_headers.items():
            result[str(key)] = str(value)
        for key, env_name in self.config.extra_header_env.items():
            value = os.environ.get(str(env_name))
            if value:
                result[str(key)] = value
        return result

    def _body(self, request: ModelRequest, stream: bool) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, payload: Mapping[str, Any], request: ModelRequest, latency_ms: float) -> ModelResponse:
        raise NotImplementedError

    def _new_state(self, request: ModelRequest) -> dict[str, Any]:
        return {"request": request}

    def _event_chunks(self, event: Mapping[str, Any], state: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def _event_reasoning(self, event: Mapping[str, Any], state: dict[str, Any]) -> list[str]:
        """Provider-supplied thinking deltas for this event; empty by default."""
        return []

    def _stream_usage(self, state: dict[str, Any], request: ModelRequest, latency_ms: float) -> Usage:
        raise NotImplementedError

    def _record_stream_state(
        self, state: dict[str, Any], request: ModelRequest, latency_ms: float
    ) -> None:
        """Publish the in-memory 'last ...' view, including partial streams."""
        self.last_usage = self._stream_usage(state, request, latency_ms)
        self.last_tool_calls = list(state.get("tool_calls") or [])
        self.last_content = "".join(state.get("text") or [])
        self.last_reasoning = "".join(state.get("reasoning") or [])

    def _redact(self, text: str) -> str:
        if self._token:
            text = text.replace(self._token, "[REDACTED]")
        # Gateway headers from extra_header_env can carry credentials (the
        # documented x-opencode-session is one). Re-read the env values at
        # call time so rotation is covered and no error or warning path can
        # echo them (F-11 residual in the provider layer).
        for env_name in getattr(self.config, "extra_header_env", {}).values():
            value = os.environ.get(str(env_name))
            if value:
                text = text.replace(value, "[REDACTED]")
        return text

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, Mapping):
            return {k: self._redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        return value

    def _error_message(self, response: httpx.Response) -> str:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, Mapping):
                error = payload.get("error")
                if isinstance(error, Mapping):
                    detail = str(error.get("message") or error.get("type") or "")
                detail = detail or str(payload.get("message") or payload.get("detail") or "")
        except ValueError:
            pass
        detail = detail or response.text[:200]
        return self._redact(f"{self.name} HTTP {response.status_code} for {self.path}: {detail[:300]}")

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        retry_header = response.headers.get("x-should-retry", "").lower() == "true"
        retryable = response.status_code in RETRYABLE_STATUSES or response.status_code >= 500 or retry_header
        raise ModelHTTPError(response.status_code, self._error_message(response), retryable=retryable)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, ModelTimeoutError):
            return True
        if isinstance(exc, ModelHTTPError):
            return exc.retryable
        return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))

    async def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_base * (2**attempt), self.backoff_cap)
        if delay > 0:
            await asyncio.sleep(delay)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one completion request with bounded retries and backoff."""
        body = self._body(request, stream=False)
        headers = self._with_extra_headers(self._headers())
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(self._url, headers=headers, json=body, timeout=self.timeout)
                self._raise_for_status(response)
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ModelValidationError(f"{self.name}: non-JSON response from {self.path}") from exc
                if not isinstance(payload, Mapping):
                    raise ModelValidationError(f"{self.name}: unexpected response shape from {self.path}")
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = self._parse(payload, request, elapsed_ms)
                # A truncated turn with neither answer nor tool calls is not a
                # response: it must fail with a typed, retryable-by-router code
                # instead of being persisted as a blank "completed" answer.
                blank_reason = empty_output_reason(
                    result.finish_reason, result.content, result.tool_calls
                )
                if blank_reason:
                    raise ModelEmptyOutputError(self.name, blank_reason)
                self.last_usage = result.usage
                self.last_finish_reason = result.finish_reason
                # Keep the "last ..." view consistent between complete() and
                # stream(); callers may read these after either call.
                self.last_tool_calls = list(result.tool_calls)
                self.last_content = result.content
                self.last_reasoning = result.reasoning
                log.info(
                    "model complete",
                    extra={
                        "provider": self.name,
                        "task_id": request.task_id,
                        "agent_id": request.agent_id,
                        "attempt": attempt + 1,
                        "latency_ms": round(elapsed_ms, 2),
                    },
                )
                return result
            except (httpx.TimeoutException) as exc:
                error: ModelError = ModelTimeoutError(f"{self.name}: request to {self.path} timed out")
                error.__cause__ = exc
            except (httpx.HTTPError, ModelHTTPError, ModelTimeoutError) as exc:
                error = exc if isinstance(exc, ModelError) else ModelError(self._redact(str(exc)))
            if not self._is_retryable(error) or attempt >= self.max_retries:
                raise error
            log.warning(
                "model request retry",
                extra={"provider": self.name, "attempt": attempt + 1, "error": self._redact(str(error))[:200]},
            )
            await self._backoff(attempt)
        raise ModelError(f"{self.name}: retries exhausted")

    async def stream_events(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """Stream typed text and reasoning deltas; retries stop after any output.

        Reasoning chunks are emitted only when the provider actually supplied
        thinking text. They are separate from text chunks so consumers can show
        them in their own channel without contaminating answer content.
        """
        body = self._body(request, stream=True)
        headers = self._with_extra_headers(self._headers())
        for attempt in range(self.max_retries + 1):
            state = self._new_state(request)
            emitted = False
            started = time.perf_counter()
            try:
                async with self._client.stream(
                    "POST", self._url, headers=headers, json=body, timeout=self.timeout
                ) as response:
                    if response.status_code >= 400:
                        # Error bodies must be buffered before they can be parsed.
                        await response.aread()
                    self._raise_for_status(response)
                    async for event in iter_sse(response):
                        for text in self._event_chunks(event, state):
                            emitted = True
                            yield StreamChunk(kind="text", text=text)
                        for reasoning in self._event_reasoning(event, state):
                            emitted = True
                            yield StreamChunk(kind="reasoning", text=reasoning)
                        # Tool fragments and reasoning are observable output too.
                        # Counting only yielded text would let a retry replay a
                        # stream that already announced tools or thoughts.
                        if state.get("text") or state.get("tools") or state.get("reasoning"):
                            emitted = True
            except (httpx.TimeoutException) as exc:
                error: BaseException = ModelTimeoutError(f"{self.name}: stream to {self.path} timed out")
                error.__cause__ = exc
            except (httpx.HTTPError, ModelError) as exc:
                # ModelError includes SSE error events raised by _event_chunks;
                # catching it lets the bookkeeping below record partial output
                # before the error is re-raised.
                error = exc if isinstance(exc, ModelError) else ModelError(self._redact(str(exc)))
            else:
                self._record_stream_state(state, request, (time.perf_counter() - started) * 1000)
                blank_reason = empty_output_reason(
                    state.get("finish_reason"), self.last_content, self.last_tool_calls
                )
                if blank_reason:
                    raise ModelEmptyOutputError(self.name, blank_reason)
                log.info(
                    "model stream complete",
                    extra={"provider": self.name, "task_id": request.task_id, "attempt": attempt + 1},
                )
                return
            if emitted or not self._is_retryable(error) or attempt >= self.max_retries:
                self._record_stream_state(state, request, (time.perf_counter() - started) * 1000)
                raise error
            log.warning(
                "model stream retry",
                extra={"provider": self.name, "attempt": attempt + 1, "error": self._redact(str(error))[:200]},
            )
            await self._backoff(attempt)

    async def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream answer text only, as plain strings (backward compatible)."""
        async for chunk in self.stream_events(request):
            if chunk.kind == "text":
                yield chunk.text

    async def aclose(self) -> None:
        """Close only clients owned by this provider."""
        if self._owns_client:
            await self._client.aclose()


class AnthropicProvider(BaseHTTPProvider):
    """Adapter for ``POST {base_url}/v1/messages`` (Messages API shape)."""

    style = "anthropic-compatible"
    path = "/v1/messages"

    def _headers(self) -> dict[str, str]:
        self._token = self._read_token()
        headers = {"content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if self._token:
            headers["x-api-key"] = self._token
        return headers

    def _messages(self, request: ModelRequest) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        declared: set[str] = set()
        for message in request.messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue
            if message.role == "assistant":
                declared = _declared_tool_ids(message)
            if message.role == "assistant" and message.tool_calls:
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": str(call.get("id") or new_id("call")),
                            "name": str(call.get("name") or ""),
                            "input": _as_dict(call.get("arguments")),
                        }
                    )
                messages.append({"role": "assistant", "content": content})
            elif message.role == "tool":
                tool_id = str(message.tool_call_id or "")
                if tool_id and tool_id in declared:
                    declared.discard(tool_id)  # one result per declaration
                    messages.append({"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": message.content,
                    }]})
                else:
                    # A strict upstream rejects the whole request when a
                    # tool_result has no tool_use above it: "messages[9] is a
                    # tool result, but it does not follow an assistant message
                    # with 'tool_calls'". That surfaced as an empty turn. The
                    # evidence still matters, so it travels as plain text.
                    messages.append({"role": "user", "content": [
                        {"type": "text",
                         "text": UNPAIRED_TOOL_PREFIX + str(message.content or "")},
                    ]})
            else:
                if message.role == "user" and message.images:
                    content: list[dict[str, Any]] = []
                    for image in message.images:
                        if isinstance(image, Mapping):
                            content.append(dict(image))
                    content.append({"type": "text", "text": message.content})
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": message.role, "content": message.content})
        return ("\n".join(system_parts) or None), messages

    def _body(self, request: ModelRequest, stream: bool) -> dict[str, Any]:
        if request.response_format is not None and not self.supports("structured"):
            raise ModelUnsupportedError(f"{self.name}: structured output is not a configured capability")
        if request.tools and not self.supports("tools"):
            raise ModelUnsupportedError(f"{self.name}: tools are not a configured capability")
        system, messages = self._messages(request)
        body: dict[str, Any] = {
            "model": request.model or self.model,
            # A reasoning model spends output tokens on thinking before it writes
            # a single character. 4096 was enough to leave nothing behind: the turn
            # came back with stop_reason=max_tokens, no text and no tool call, which
            # the router correctly refuses as an empty response. Providers may pin
            # their own ceiling; otherwise leave real room for the answer.
            "max_tokens": request.max_tokens or getattr(self, "max_tokens", None) or 32_000,
            "messages": messages,
        }
        if system:
            body["system"] = system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.tools:
            body["tools"] = [normalize_tool_spec(spec, "anthropic") for spec in request.tools]
        if request.response_format is not None:
            # Forward the caller-supplied format unchanged; no schema is invented.
            body["output_config"] = {"format": dict(request.response_format)}
        if stream:
            body["stream"] = True
        return body

    def _parse(self, payload: Mapping[str, Any], request: ModelRequest, latency_ms: float) -> ModelResponse:
        blocks = payload.get("content") or []
        if not isinstance(blocks, list):
            raise ModelValidationError(f"{self.name}: invalid content blocks")
        texts = [str(b.get("text", "")) for b in blocks if isinstance(b, Mapping) and b.get("type") == "text"]
        tools = [b for b in blocks if isinstance(b, Mapping) and b.get("type") == "tool_use"]
        reasoning = "".join(
            piece
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") in THINKING_BLOCK_TYPES
            for piece in reasoning_text(block)
        )
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        measured = usage.get("input_tokens"), usage.get("output_tokens")
        result_usage = usage_from_counts(
            measured[0], measured[1], self.config, request, "".join(texts), latency_ms
        )
        return ModelResponse(
            content="".join(texts),
            model=str(payload.get("model") or request.model or self.model),
            provider=self.name,
            reasoning=reasoning,
            tool_calls=normalize_tool_calls(tools, "anthropic"),
            usage=result_usage,
            finish_reason=payload.get("stop_reason"),
            raw=self._redact_value(dict(payload)),
        )

    def _new_state(self, request: ModelRequest) -> dict[str, Any]:
        return {
            "request": request,
            "text": [],
            "reasoning": [],
            "tools": {},
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
        }

    def _event_chunks(self, event: Mapping[str, Any], state: dict[str, Any]) -> list[str]:
        kind = event.get("type")
        if kind == "error" or event.get("error"):
            detail = event.get("error") or {}
            message = detail.get("message") if isinstance(detail, Mapping) else str(detail)
            if message is None and isinstance(detail, Mapping):
                message = detail.get("type")
            raise ModelError(self._redact(f"{self.name}: stream error: {message}"))
        if kind == "message_start":
            message = event.get("message") if isinstance(event.get("message"), Mapping) else {}
            usage = message.get("usage") if isinstance(message.get("usage"), Mapping) else {}
            state["input_tokens"] = usage.get("input_tokens")
        elif kind == "content_block_start":
            block = event.get("content_block") if isinstance(event.get("content_block"), Mapping) else {}
            if block.get("type") == "tool_use":
                index = event.get("index", 0)
                # Most servers send the arguments as input_json_delta fragments
                # and leave this empty, but some compatible gateways send the
                # complete input inline. Keep it so those arguments are not
                # silently dropped.
                raw_input = block.get("input")
                if isinstance(raw_input, Mapping) and raw_input:
                    initial_input = json.dumps(raw_input, ensure_ascii=False)
                elif isinstance(raw_input, str) and raw_input.strip():
                    initial_input = raw_input
                else:
                    initial_input = ""
                state["tools"][index] = {
                    "id": block.get("id") or new_id("call"),
                    "name": block.get("name", ""),
                    "input": initial_input,
                    "_inline": bool(initial_input),
                }
        elif kind == "content_block_delta":
            delta = event.get("delta") if isinstance(event.get("delta"), Mapping) else {}
            if delta.get("type") == "text_delta":
                text = str(delta.get("text") or "")
                state["text"].append(text)
                return [text]
            if delta.get("type") == "input_json_delta":
                index = event.get("index", 0)
                tool = state["tools"].get(index)
                if tool is not None:
                    # If a gateway sent both an inline input and fragments,
                    # the fragments are authoritative: drop the inline value
                    # once rather than concatenating two JSON documents.
                    if tool.pop("_inline", False):
                        tool["input"] = ""
                    tool["input"] += str(delta.get("partial_json") or "")
        elif kind == "message_delta":
            delta = event.get("delta") if isinstance(event.get("delta"), Mapping) else {}
            state["finish_reason"] = delta.get("stop_reason")
            usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
            if "output_tokens" in usage:
                state["output_tokens"] = usage["output_tokens"]
            # Some compatible gateways only report input_tokens here.
            if "input_tokens" in usage:
                state["input_tokens"] = usage["input_tokens"]
        return []

    def _event_reasoning(self, event: Mapping[str, Any], state: dict[str, Any]) -> list[str]:
        # Thinking deltas arrive keyed by content-block index; concatenation in
        # arrival order preserves the provider's sequence. Signatures and
        # redacted data are intentionally left out by _anthropic_reasoning_parts.
        pieces = _anthropic_reasoning_parts(event)
        if pieces:
            state.setdefault("reasoning", []).extend(pieces)
        return pieces

    def _stream_usage(self, state: dict[str, Any], request: ModelRequest, latency_ms: float) -> Usage:
        text = "".join(state.get("text") or [])
        tools = []
        for tool in (state.get("tools") or {}).values():
            entry = dict(tool)
            entry.pop("_inline", None)
            entry["input"] = _as_dict(entry.get("input"))
            tools.append(entry)
        state["tool_calls"] = normalize_tool_calls(tools, "anthropic")
        self.last_finish_reason = state.get("finish_reason")
        return usage_from_counts(
            state.get("input_tokens"), state.get("output_tokens"), self.config, request, text, latency_ms
        )


UNPAIRED_TOOL_PREFIX = "[Unpaired tool result; kept as data, not as a tool result] "


def _declared_tool_ids(message: Any) -> set[str]:
    """Ids an assistant turn actually declared, so its results can be matched."""
    return {str(call.get("id") or "") for call in (getattr(message, "tool_calls", None) or [])}


class OpenAIProvider(BaseHTTPProvider):
    """Adapter for ``POST {base_url}/v1/chat/completions`` (Chat Completions shape)."""

    style = "openai-compatible"
    path = "/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        self._token = self._read_token()
        headers = {"content-type": "application/json"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    def _messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        declared: set[str] = set()
        for message in request.messages:
            if message.role == "assistant":
                declared = _declared_tool_ids(message)
            if message.role == "tool":
                tool_id = str(message.tool_call_id or "")
                if not tool_id or tool_id not in declared:
                    messages.append({
                        "role": "user",
                        "content": UNPAIRED_TOOL_PREFIX + str(message.content or ""),
                    })
                    continue
                declared.discard(tool_id)
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.name:
                item["name"] = message.name
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": str(call.get("id") or new_id("call")),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name") or ""),
                            "arguments": json.dumps(_as_dict(call.get("arguments")), ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(item)
        return messages

    def _body(self, request: ModelRequest, stream: bool) -> dict[str, Any]:
        if request.tools and not self.supports("tools"):
            raise ModelUnsupportedError(f"{self.name}: tools are not a configured capability")
        body: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": self._messages(request),
        }
        # Same reasoning as the Anthropic path: a ceiling that vanishes when the
        # caller says nothing lets a gateway pick a tiny default.
        body["max_tokens"] = request.max_tokens or getattr(self, "max_tokens", None) or 32_000
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.tools:
            body["tools"] = [normalize_tool_spec(spec, "openai") for spec in request.tools]
        if request.response_format is not None:
            if not self.supports("structured"):
                raise ModelUnsupportedError(f"{self.name}: structured output is not a configured capability")
            body["response_format"] = dict(request.response_format)
        if stream:
            body["stream"] = True
            if self.supports("usage"):
                body["stream_options"] = {"include_usage": True}
        return body

    def _parse(self, payload: Mapping[str, Any], request: ModelRequest, latency_ms: float) -> ModelResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelValidationError(f"{self.name}: response has no choices")
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
        content = _openai_text(message.get("content"))
        reasoning = "".join(_openai_reasoning_parts(message))
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        result_usage = usage_from_counts(
            usage.get("prompt_tokens"), usage.get("completion_tokens"), self.config, request, content, latency_ms
        )
        return ModelResponse(
            content=content,
            model=str(payload.get("model") or request.model or self.model),
            provider=self.name,
            reasoning=reasoning,
            tool_calls=normalize_tool_calls(message.get("tool_calls"), "openai"),
            usage=result_usage,
            finish_reason=choice.get("finish_reason"),
            raw=self._redact_value(dict(payload)),
        )

    def _new_state(self, request: ModelRequest) -> dict[str, Any]:
        return {
            "request": request,
            "text": [],
            "reasoning": [],
            "tools": {},
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
        }

    def _event_chunks(self, event: Mapping[str, Any], state: dict[str, Any]) -> list[str]:
        if event.get("error"):
            error = event["error"]
            detail = error.get("message") if isinstance(error, Mapping) else str(error)
            raise ModelError(self._redact(f"{self.name}: stream error: {detail}"))
        usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
        if "prompt_tokens" in usage:
            state["input_tokens"] = usage["prompt_tokens"]
        if "completion_tokens" in usage:
            state["output_tokens"] = usage["completion_tokens"]
        choices = event.get("choices")
        if not isinstance(choices, list):
            return []
        chunks: list[str] = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            if choice.get("finish_reason"):
                state["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
            text = _openai_text(delta.get("content"))
            if text:
                state["text"].append(text)
                chunks.append(text)
            calls = delta.get("tool_calls") or []
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                # ``index`` is standard but some compatible gateways omit it.
                # Fall back to the call id (present in the first fragment) so
                # parallel tool calls are not merged into a single one.
                index = call.get("index")
                if index is None:
                    index = call.get("id") or f"auto-{len(state['tools'])}"
                tool = state["tools"].setdefault(
                    index, {"id": "", "name": "", "arguments": ""}
                )
                if call.get("id"):
                    tool["id"] = call["id"]
                function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                if function.get("name"):
                    tool["name"] += str(function["name"])
                if function.get("arguments"):
                    tool["arguments"] += str(function["arguments"])
        return chunks

    def _event_reasoning(self, event: Mapping[str, Any], state: dict[str, Any]) -> list[str]:
        pieces: list[str] = []
        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                source = choice.get("delta")
                if not isinstance(source, Mapping):
                    source = choice.get("message")
                if not isinstance(source, Mapping):
                    source = choice
                pieces.extend(_openai_reasoning_parts(source))
        if not pieces:
            # Some compatible gateways emit reasoning at the top level.
            pieces.extend(_openai_reasoning_parts(event))
        if pieces:
            state.setdefault("reasoning", []).extend(pieces)
        return pieces

    def _stream_usage(self, state: dict[str, Any], request: ModelRequest, latency_ms: float) -> Usage:
        text = "".join(state.get("text") or [])
        state["tool_calls"] = normalize_tool_calls(list((state.get("tools") or {}).values()), "openai")
        self.last_finish_reason = state.get("finish_reason")
        return usage_from_counts(
            state.get("input_tokens"), state.get("output_tokens"), self.config, request, text, latency_ms
        )


def build_provider(name: str, config: ModelProviderConfig, **kwargs: Any) -> BaseHTTPProvider:
    """Create a configured provider adapter by ``config.kind``."""
    kind = (config.kind or "anthropic-compatible").lower()
    if kind == "anthropic-compatible":
        provider = AnthropicProvider(config, name=name, **kwargs)
        provider.max_tokens = int(getattr(config, "max_tokens", 32_000) or 32_000)
        return provider
    if kind == "openai-compatible":
        provider = OpenAIProvider(config, name=name, **kwargs)
        provider.max_tokens = int(getattr(config, "max_tokens", 32_000) or 32_000)
        return provider
    raise ModelError(f"provider {name!r}: unsupported kind {config.kind!r}")


__all__ = [
    "ANTHROPIC_VERSION",
    "CHARS_PER_TOKEN",
    "AnthropicProvider",
    "BaseHTTPProvider",
    "MAX_TOKENS_REASONS",
    "ModelEmptyOutputError",
    "ModelError",
    "ModelHTTPError",
    "ModelTimeoutError",
    "ModelUnsupportedError",
    "ModelValidationError",
    "OpenAIProvider",
    "StreamChunk",
    "build_provider",
    "empty_output_reason",
    "estimate_tokens",
    "normalize_tool_calls",
    "usage_from_counts",
]
