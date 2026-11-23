"""Task classification, provider selection, bounded escalation, and usage accounting."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from .. import db
from ..config import ModelProviderConfig, Settings, _mock_fallback_env_override
from ..contracts import (
    ChatMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    StreamChunk,
    Usage,
    utcnow,
)
from ..logging import get_logger
from .mock import MockProvider
from .provider import (
    ModelEmptyOutputError,
    ModelError,
    ModelValidationError,
    build_provider,
    estimate_request_tokens,
    usage_from_counts,
)

log = get_logger("models.router")

# Explicit, final-turn synthesis instruction. Shared with the chat loop so a
# tool-heavy turn ends with one authoritative "stop calling tools" request,
# both in the normal N+1 call and in the shortened empty-output retry.
FINAL_ANSWER_ONLY_INSTRUCTION = (
    "SYSTEM DIRECTIVE (final answer only): No more tool calls — answer now from "
    "gathered results. Do not request or execute any further tool; if the evidence "
    "is insufficient, say precisely what is missing."
)

TASK_CATEGORIES: frozenset[str] = frozenset(
    {
        "simple_conversation",
        "summarization",
        "classification",
        "research",
        "coding",
        "debugging",
        "complex_reasoning",
        "planning",
        "tool_execution",
        "verification",
    }
)

# Ordered by tie-break priority. Scoring combines whole-phrase and word hits.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("debugging", ("debug", "bug", "error", "traceback", "exception", "crash", "broken", "failing", "failure")),
    ("verification", ("verify", "validate", "confirm", "assert", "review", "check that", "test result", "evidence")),
    ("classification", ("classify", "categorize", "label", "sentiment", "category", "tag this", "yes or no")),
    ("summarization", ("summarize", "summary", "condense", "tldr", "recap", "brief overview", "key points")),
    ("planning", ("plan", "planning", "roadmap", "steps", "strategy", "design", "schedule", "milestones")),
    ("research", ("research", "investigate", "search", "sources", "documentation", "compare", "literature", "find out")),
    ("complex_reasoning", ("reason", "reasoning", "prove", "derive", "analyze", "analyse", "tradeoff", "solve", "calculate", "why")),
    ("coding", ("code", "coding", "function", "python", "implement", "refactor", "script", "class", "compile", "repository", "bugfix")),
    ("tool_execution", ("execute", "run tool", "call tool", "use tool", "invoke", "perform")),
    ("simple_conversation", ("hello", "hey", "thanks", "thank you", "how are you", "good morning", "chat")),
)


def classify_task(request: ModelRequest) -> str:
    """Heuristically classify a request into one of :data:`TASK_CATEGORIES`.

    An explicit ``request.task_category`` wins. Otherwise the last user message
    is weighted more heavily than earlier context and tool presence is a signal.
    No provider call is made by classification.
    """
    explicit = getattr(request, "task_category", None)
    if explicit:
        normalized = re.sub(r"[\s-]+", "_", str(explicit).strip().lower())
        if normalized in TASK_CATEGORIES:
            return normalized
    texts = [m.content for m in request.messages if m.role == "user"]
    if not texts:
        texts = [m.content for m in request.messages]
    prompt = " ".join(texts).lower()
    recent = (texts[-1] if texts else "").lower()
    words = set(re.findall(r"[a-z]+", prompt))
    recent_words = set(re.findall(r"[a-z]+", recent))
    scores: dict[str, float] = {name: 0.0 for name, _ in _KEYWORDS}
    for name, keywords in _KEYWORDS:
        for keyword in keywords:
            if " " in keyword:
                if keyword in prompt:
                    scores[name] += 2.0
                if keyword in recent:
                    scores[name] += 1.0
            else:
                if keyword in words:
                    scores[name] += 1.0
                if keyword in recent_words:
                    scores[name] += 1.0
    if request.tools:
        scores["tool_execution"] += 2.0
        if any(word in recent_words for word in ("execute", "call", "run", "invoke", "use")):
            scores["tool_execution"] += 2.0
    best = max(scores.values())
    if best <= 0:
        return "simple_conversation"
    for name, _ in _KEYWORDS:
        if scores[name] == best:
            return name
    return "simple_conversation"


def _default_validator(request: ModelRequest, response: ModelResponse) -> bool:
    """Reject empty output as a validation failure that warrants escalation."""
    return bool(response.content.strip() or response.tool_calls)


class ModelRouter:
    """Selects providers, escalates on failure, and records every call.

    ``escalation_limit`` is the maximum number of *additional* provider attempts
    after the initial selection. Each attempt gets one ``model_usage`` row with
    wall-clock latency measured here and token counts reported by the provider
    (or marked estimated when the provider reports none).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        providers: dict[str, Any] | None = None,
        validator: Callable[[ModelRequest, ModelResponse], bool] | None = None,
        backoff_base: float = 0.25,
    ) -> None:
        self.settings = settings or Settings.load()
        self.validator = validator or _default_validator
        self._backoff_base = backoff_base
        # Build failures are recorded, never downgraded to a warning followed
        # by a mock answer. ``status()`` reports them as a blocked model state
        # and every model call raises a clear error until they are fixed.
        self._build_errors: list[dict[str, str]] = []
        self.providers: dict[str, Any] = providers if providers is not None else self._build_providers()
        self.last_stream_provider: str | None = None
        self.last_stream_tool_calls: list[dict[str, Any]] = []
        self.last_stream_content: str = ""
        # Provider-supplied thinking from the most recent stream_events() turn.
        # It is deliberately separate from last_stream_content: callers must
        # never fold it into an answer or a tool round.
        self.last_stream_reasoning: str = ""
        self.last_stream_usage: Any | None = None
        if not self.providers and self._mock_fallback_allowed:
            cfg = self.settings.models.providers.get("mock") or ModelProviderConfig(kind="mock", model="mock-1")
            self.providers["mock"] = MockProvider(cfg, name="mock")
        if not self.providers:
            log.warning("no routable model provider; model calls are blocked")

    @property
    def _mock_fallback_allowed(self) -> bool:
        """Whether automatic mock injection was explicitly opted into."""
        if self.settings.models.allow_mock_fallback:
            return True
        # Settings built directly in code (without load()) still honour the
        # env opt-in; invalid tokens raise and therefore fail closed.
        return bool(_mock_fallback_env_override())

    @property
    def simulated(self) -> bool:
        """True only when the routable providers are explicitly mock ones.

        An empty provider set is *blocked*, not simulated: no mock reply can be
        produced. Callers must surface this to users: mock output is simulated,
        never a live model answer.
        """
        if not self.providers:
            return False
        return all(
            str(getattr(self._config(provider), "kind", "")).lower() == "mock"
            for provider in self.providers.values()
        )

    def _is_blocked(self) -> bool:
        """Blocked only when NOTHING could be built.

        One unbuildable entry is a degraded configuration, not an outage: the
        broken provider is reported in ``build_errors`` and the healthy ones keep
        answering. Previously any build error blocked every model, so a single
        stale or half-configured entry took the whole agent offline.
        """
        return not self.providers

    def _blocked_reason(self) -> str:
        """Scrubbed, operator-facing explanation of a blocked model state."""
        parts: list[str] = []
        if self._build_errors:
            details = "; ".join(
                f"{item['provider']}: {item['error']}" for item in self._build_errors
            )
            parts.append(f"configured live provider failed to build: {details}")
        if not self.providers:
            parts.append("no live provider is routable")
        if not self._build_errors and not self.providers:
            configured_reason = self.settings.model_status().get("reason")
            if configured_reason:
                parts.append(str(configured_reason))
        parts.append(
            "model calls are blocked; configure a working live provider or set "
            "models.allow_mock_fallback=true (LEO_ALLOW_MOCK_FALLBACK=true) for "
            "explicit offline mock use"
        )
        return ". ".join(parts)

    def _no_route_error(self) -> ModelError:
        """Raise a clear, scrubbed error instead of silently using mock."""
        if self._is_blocked():
            return ModelError(self._blocked_reason())
        return ModelError("no enabled provider satisfies the request requirements")

    @staticmethod
    def final_answer_only_request(request: ModelRequest) -> ModelRequest:
        """Build a compact, tool-free retry request after an empty max_tokens turn.

        Only the user's latest message and the most recent tool results survive;
        tool results are demoted to user-role data so no assistant(tool_use)
        block can be left unanswered. Tool specs are removed so the retry cannot
        start another tool round.
        """
        messages: list[ChatMessage] = []
        user_messages = [message for message in request.messages if message.role == "user"]
        if user_messages:
            messages.append(user_messages[-1])
        tool_results = [message for message in request.messages if message.role == "tool"][-6:]
        if tool_results:
            gathered = "\n".join(
                f"[{getattr(message, 'name', None) or 'tool'} result] "
                f"{str(message.content)[:2000]}"
                for message in tool_results
            )
            messages.append(ChatMessage(
                role="user", content="Gathered tool results (data only):\n" + gathered[:12000],
            ))
        messages.append(ChatMessage(role="user", content=FINAL_ANSWER_ONLY_INSTRUCTION))
        return ModelRequest(
            messages=messages,
            model=request.model,
            task_category=request.task_category,
            task_id=request.task_id,
            agent_id=request.agent_id,
            temperature=request.temperature,
            tools=[],
            response_format=None,
        )

    def status(self) -> dict[str, Any]:
        """Router status: live, simulated (explicit mock), or blocked."""
        configured = self.settings.model_status()
        routable = sorted(self.providers)
        live = sorted(
            name for name in routable
            if str(getattr(self._config(self.providers[name]), "kind", "")).lower() != "mock"
        )
        mocks = sorted(set(routable) - set(live))
        simulated = self.simulated
        if self._is_blocked():
            status = "blocked"
            reason = self._blocked_reason()
        elif self._build_errors:
            # Some live providers built, or the operator opted into mock after
            # a build failure: never claim a fully healthy live configuration.
            status = "degraded"
            reason = self._blocked_reason()
        elif live:
            status = "live"
            reason = configured.get("reason")
        elif simulated:
            status = "simulated"
            reason = "only explicit mock providers are routable; output is simulated"
        else:
            status = "blocked"
            reason = self._blocked_reason()
        return {
            **configured,
            "simulated": simulated,
            "status": status,
            "reason": reason,
            "routable_providers": routable,
            "routable_live_providers": live,
            "routable_mock_providers": mocks,
            "build_errors": [dict(item) for item in self._build_errors],
        }

    def _build_providers(self) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        for name, cfg in self.settings.models.providers.items():
            if not cfg.enabled:
                continue
            if cfg.kind == "mock":
                # Explicitly configured mocks stay legal without the opt-in.
                providers[name] = MockProvider(cfg, name=name)
            elif cfg.kind in {"anthropic-compatible", "openai-compatible"}:
                if not cfg.base_url:
                    reason = "missing base_url"
                    self._build_errors.append({"provider": name, "error": reason})
                    log.warning("live provider cannot be built", extra={"provider": name, "error": reason})
                    continue
                # Credentials and gateway headers are checked at build time:
                # otherwise a live provider looks healthy until the first
                # request fails and gets mistaken for an escalation trigger.
                missing: list[str] = []
                if cfg.api_key_env and not os.getenv(cfg.api_key_env):
                    missing.append(f"api_key_env {cfg.api_key_env} is unset")
                for header, env_name in sorted(cfg.extra_header_env.items()):
                    if not os.getenv(env_name):
                        missing.append(f"extra_header_env[{header!r}] {env_name} is unset")
                if missing:
                    reason = f"missing required environment value(s): {', '.join(missing)}"
                    self._build_errors.append({"provider": name, "error": reason})
                    log.warning("live provider cannot be built", extra={"provider": name, "error": reason})
                    continue
                try:
                    providers[name] = build_provider(name, cfg, backoff_base=self._backoff_base)
                except Exception as exc:  # broken live config must never become a mock answer
                    reason = self._scrub(str(exc)) or type(exc).__name__
                    self._build_errors.append({"provider": name, "error": reason})
                    log.warning("live provider cannot be built", extra={"provider": name, "error": reason})
            else:
                reason = f"unsupported provider kind {cfg.kind!r}"
                self._build_errors.append({"provider": name, "error": reason})
                log.warning("live provider cannot be built", extra={"provider": name, "error": reason})
        return providers

    def _scrub(self, text: str) -> str:
        """Remove any configured credential value from free-form error text."""
        for cfg in self.settings.models.providers.values():
            env_names = [cfg.api_key_env] if cfg.api_key_env else []
            env_names.extend(cfg.extra_header_env.values())
            for env_name in env_names:
                value = os.environ.get(env_name) if env_name else None
                if value:
                    text = text.replace(value, "[REDACTED]")
        return text

    async def _measured_stats(self) -> dict[str, tuple[float, float]]:
        try:
            rows = await db.fetchall(
                "SELECT provider, AVG(latency_ms) AS avg_latency, "
                "AVG(CASE WHEN success=1 THEN 1.0 ELSE 0.0 END) AS reliability "
                "FROM model_usage GROUP BY provider"
            )
        except Exception as exc:  # table missing / db not initialized
            log.debug("usage stats unavailable", extra={"error": str(exc)[:120]})
            return {}
        stats: dict[str, tuple[float, float]] = {}
        for row in rows:
            stats[str(row["provider"])] = (
                float(row["avg_latency"] or 0.0),
                float(row["reliability"] if row["reliability"] is not None else 1.0),
            )
        return stats

    def _apply_request_defaults(self, request: ModelRequest) -> ModelRequest:
        """Apply configured request defaults without overwriting caller values."""
        if request.temperature is None:
            configured = getattr(self.settings.models, "temperature", None)
            if isinstance(configured, (int, float)) and not isinstance(configured, bool):
                request.temperature = float(configured)
        return request

    def _required_capabilities(self, request: ModelRequest) -> set[str]:
        metadata = getattr(request, "metadata", None) or {}
        required = {"text"}
        if request.tools:
            required.add("tools")
        if request.response_format is not None:
            required.add("structured")
        if metadata.get("require_streaming"):
            required.add("streaming")
        if any(getattr(message, "images", None) for message in request.messages):
            # Never send image blocks to a provider that did not declare vision.
            required.add("vision")
        return required

    @staticmethod
    def _preference_tokens(metadata: Mapping[str, Any]) -> list[str]:
        """Flatten provider names and agent capability tags deterministically."""
        raw: Any = None
        for key in (
            "preferred_providers", "preferred_provider", "model_preference",
            "preference", "provider", "agent_capabilities",
            "agent_capability_tags", "capabilities",
        ):
            if metadata.get(key):
                raw = metadata[key]
                break
        if isinstance(raw, Mapping):
            raw = [value for _, value in sorted(raw.items(), key=lambda item: str(item[0]))]
        if isinstance(raw, (set, frozenset)):
            raw = sorted(raw, key=str)
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        tokens: list[str] = []
        for item in raw:
            token = str(item).strip()
            if token and token not in tokens:
                tokens.append(token)
        return tokens

    # Capability tags used by specialist agent seeds. A tag never overrides an
    # explicit provider name; it only ranks providers that advertise the tagged
    # capability above those that do not.
    _CAPABILITY_TAGS: dict[str, str] = {
        "vision": "vision", "image": "vision", "images": "vision",
        "text": "text", "stream": "streaming", "streaming": "streaming",
        "tool": "tools", "tools": "tools", "tool_use": "tools",
        "structured": "structured", "json": "structured", "usage": "usage",
        "coding": "tools", "code": "tools", "research": "tools",
        "reasoning": "structured", "planning": "structured",
    }

    def _provider_tags(self, name: str) -> frozenset[str]:
        provider = self.providers.get(name)
        declared = getattr(provider, "capabilities", None)
        if not declared:
            config = self.settings.models.providers.get(name)
            declared = config.capabilities if config else ()
        return frozenset(str(item).strip().lower() for item in declared)

    def _preference_index(self, name: str, request: ModelRequest) -> int:
        metadata = getattr(request, "metadata", None) or {}
        preferred = self._preference_tokens(metadata)
        count = len(preferred)
        for index, item in enumerate(preferred):
            if item == name:
                # Scale the offset with the list length: every explicit
                # preference must rank strictly above capability tags and the
                # configured default, even when >10 providers are listed.
                return index - count - 1
        tags = self._provider_tags(name)
        matched = [
            index for index, item in enumerate(preferred)
            if self._CAPABILITY_TAGS.get(item.lower()) in tags
        ]
        if matched:
            return min(matched)
        if name == self.settings.models.default:
            return count + 1
        if name == self.settings.models.fallback:
            return count + 2
        return count + 3

    @staticmethod
    def _config(provider: Any) -> ModelProviderConfig:
        return getattr(provider, "config", None) or ModelProviderConfig()

    def _rank(
        self,
        request: ModelRequest,
        stats: Mapping[str, tuple[float, float]],
        *,
        require_streaming: bool = False,
    ) -> list[tuple[str, Any]]:
        required = self._required_capabilities(request)
        if require_streaming:
            # ``stream()`` calls ``provider.stream()`` on every candidate, so a
            # provider that does not advertise streaming must never be ranked
            # here — regardless of whether the caller set the metadata hint.
            required = required | {"streaming"}
        needed_tokens = estimate_request_tokens(request) + max(0, request.max_tokens or 0)
        candidates: list[tuple[int, float, str, str, Any]] = []
        for name, provider in self.providers.items():
            config = self._config(provider)
            if not config.enabled:
                continue
            declared = getattr(provider, "capabilities", None)
            capabilities = frozenset(declared) if declared is not None else required
            if not required.issubset(capabilities):
                continue
            context_window = getattr(provider, "context_window", None)
            if context_window is not None and needed_tokens > int(context_window):
                continue
            avg_latency, reliability = stats.get(name, (0.0, 1.0))
            cost = config.cost_per_million_input + config.cost_per_million_output
            score = cost + avg_latency / 1_000_000 + (1.0 - reliability) * 0.5
            preference = self._preference_index(name, request)
            # Preference dominates; cost/latency/reliability break ties thereafter.
            candidates.append((preference, score, name, request.task_category or "", provider))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [(item[2], item[4]) for item in candidates]

    async def select(self, request: ModelRequest) -> tuple[str, ModelProvider]:
        """Return the best capable ``(provider_name, provider)`` for a request."""
        self._apply_request_defaults(request)
        if self._is_blocked():
            raise self._no_route_error()
        stats = await self._measured_stats()
        ranked = self._rank(request, stats)
        if not ranked:
            raise self._no_route_error()
        return ranked[0]

    async def _record(
        self,
        name: str,
        provider: Any,
        request: ModelRequest,
        usage: Usage | None,
        latency_ms: float,
        success: bool,
        *,
        attempt: int,
        stream: bool,
        error: BaseException | None = None,
    ) -> None:
        config = self._config(provider)
        measured = usage or usage_from_counts(None, None, config, request, "", latency_ms)
        measured.latency_ms = max(0.0, float(latency_ms))
        metadata: dict[str, Any] = {
            "estimated": bool(measured.estimated),
            "simulated": self.simulated,
            "task_category": classify_task(request),
            "attempt": attempt,
            "stream": stream,
        }
        if error is not None:
            metadata["error"] = self._scrub(str(error))[:200]
        try:
            await db.execute(
                "INSERT INTO model_usage(ts,provider,model,task_id,agent_id,input_tokens,output_tokens,"
                "latency_ms,cost,success,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    utcnow(),
                    name,
                    str(getattr(provider, "model", None) or request.model or name),
                    request.task_id,
                    request.agent_id,
                    measured.input_tokens,
                    measured.output_tokens,
                    measured.latency_ms,
                    measured.cost,
                    int(success),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        except Exception as exc:
            log.error("could not record model usage", extra={"provider": name, "error": str(exc)[:200]})

    async def complete(
        self, request: ModelRequest, *, _empty_output_retried: bool = False
    ) -> ModelResponse:
        """Complete a request, escalating through ranked providers on failure.

        An empty max_tokens turn is retried exactly once as a shortened,
        tool-free final-answer request before the error reaches the caller.
        """
        self._apply_request_defaults(request)
        if self._is_blocked():
            raise self._no_route_error()
        stats = await self._measured_stats()
        ranked = self._rank(request, stats)
        if not ranked:
            raise self._no_route_error()
        max_attempts = min(len(ranked), max(0, self.settings.models.escalation_limit) + 1)
        last_error: BaseException | None = None
        for attempt, (name, provider) in enumerate(ranked[:max_attempts], start=1):
            started = time.perf_counter()
            response: ModelResponse | None = None
            attempt_error: BaseException | None = None
            try:
                response = await provider.complete(request)
                if not self.validator(request, response):
                    raise ModelValidationError(f"{name}: output failed validation")
                elapsed_ms = (time.perf_counter() - started) * 1000
                response.usage.latency_ms = elapsed_ms
                return response
            except Exception as exc:
                attempt_error = exc
                last_error = exc
                log.warning(
                    "model attempt failed",
                    extra={"provider": name, "attempt": attempt, "error": self._scrub(str(exc))[:200]},
                )
            finally:
                await self._record(
                    name,
                    provider,
                    request,
                    response.usage if response else None,
                    (time.perf_counter() - started) * 1000,
                    attempt_error is None,
                    attempt=attempt,
                    stream=False,
                    error=attempt_error,
                )
        if (
            isinstance(last_error, ModelEmptyOutputError)
            and not _empty_output_retried
        ):
            log.warning("model empty output; retrying with final-answer-only request")
            retry = self.final_answer_only_request(request)
            try:
                return await self.complete(retry, _empty_output_retried=True)
            except Exception as exc:
                raise exc from last_error
        raise last_error or ModelError("all model attempts failed")

    async def stream(
        self, request: ModelRequest, *, _empty_output_retried: bool = False
    ) -> AsyncIterator[str]:
        """Stream text with bounded escalation before the first emitted chunk."""
        # These attributes are read by the chat loop *after* iteration to learn
        # which provider answered and which tool calls it emitted. Clear them
        # for this turn first: a failed stream (all attempts exhausted, or a
        # partial stream that cannot be replayed) would otherwise leave the
        # previous turn's provider/content/tool calls in place and cause stale
        # tool calls to be executed again on the next iteration.
        self.last_stream_provider = None
        self.last_stream_tool_calls = []
        self.last_stream_content = ""
        self.last_stream_usage = None
        self._apply_request_defaults(request)
        if self._is_blocked():
            raise self._no_route_error()
        stats = await self._measured_stats()
        ranked = self._rank(request, stats, require_streaming=True)
        if not ranked:
            raise self._no_route_error()
        max_attempts = min(len(ranked), max(0, self.settings.models.escalation_limit) + 1)
        last_error: BaseException | None = None
        for attempt, (name, provider) in enumerate(ranked[:max_attempts], start=1):
            started = time.perf_counter()
            emitted = False
            attempt_error: BaseException | None = None
            try:
                async for chunk in provider.stream(request):
                    emitted = True
                    yield chunk
                self.last_stream_provider = name
                self.last_stream_tool_calls = list(getattr(provider, "last_tool_calls", []) or [])
                self.last_stream_content = str(getattr(provider, "last_content", "") or "")
                self.last_stream_usage = getattr(provider, "last_usage", None)
                return
            except Exception as exc:
                attempt_error = exc
                last_error = exc
                log.warning(
                    "model stream attempt failed",
                    extra={"provider": name, "attempt": attempt, "error": self._scrub(str(exc))[:200]},
                )
                if emitted:
                    # Output already delivered to the caller: replay would duplicate it.
                    raise
            finally:
                await self._record(
                    name,
                    provider,
                    request,
                    getattr(provider, "last_usage", None),
                    (time.perf_counter() - started) * 1000,
                    attempt_error is None,
                    attempt=attempt,
                    stream=True,
                    error=attempt_error,
                )
        if (
            isinstance(last_error, ModelEmptyOutputError)
            and not _empty_output_retried
        ):
            log.warning("model empty output in text stream; retrying with final-answer-only request")
            retry = self.final_answer_only_request(request)
            async for text in self.stream(retry, _empty_output_retried=True):
                yield text
            return
        raise last_error or ModelError("all model stream attempts failed")

    def _stream_hook_overridden(self) -> bool:
        """Whether a caller replaced this instance's legacy ``stream()`` hook.

        Subclasses and tests can legitimately override ``stream()`` (for
        example, to inject a deterministic tool round). Typed streaming must
        not silently bypass that override: the text hook is the contract such
        callers implement, and it can never invent a reasoning channel.
        """
        current = getattr(self, "stream", None)
        original = getattr(ModelRouter, "stream", None)
        if not callable(current) or not callable(original):
            return False
        return current != original.__get__(self, type(self))

    async def stream_events(
        self, request: ModelRequest, *, _empty_output_retried: bool = False
    ) -> AsyncIterator[StreamChunk]:
        """Stream typed ``text``/``reasoning`` chunks with bounded escalation.

        This mirrors :meth:`stream` (same ranking, attempt limit, output replay
        guard and usage accounting) but iterates the provider's typed hook.
        ``stream()`` stays text-only for its existing callers and fakes; this
        method is the opt-in channel for consumers that can render provider
        thinking separately. Reasoning is only ever forwarded when the provider
        actually emitted it -- it is never derived from answer text.
        """
        # Reset every per-turn view before the first await, including reasoning:
        # a failed/escalated turn must not expose a previous turn's thoughts.
        self.last_stream_provider = None
        self.last_stream_tool_calls = []
        self.last_stream_content = ""
        self.last_stream_reasoning = ""
        self.last_stream_usage = None
        if self._stream_hook_overridden():
            # Respect an injected text-only stream implementation. It already
            # publishes last_stream_provider/content/tool_calls/usage itself;
            # no reasoning is claimed for a hook that cannot emit it.
            async for text in self.stream(request):
                yield StreamChunk(kind="text", text=text)
            return
        self._apply_request_defaults(request)
        if self._is_blocked():
            raise self._no_route_error()
        stats = await self._measured_stats()
        ranked = self._rank(request, stats, require_streaming=True)
        if not ranked:
            raise self._no_route_error()
        max_attempts = min(len(ranked), max(0, self.settings.models.escalation_limit) + 1)
        last_error: BaseException | None = None
        for attempt, (name, provider) in enumerate(ranked[:max_attempts], start=1):
            started = time.perf_counter()
            emitted = False
            typed_text: list[str] = []
            typed_reasoning: list[str] = []
            attempt_error: BaseException | None = None
            try:
                typed_stream = getattr(provider, "stream_events", None)
                if callable(typed_stream):
                    async for raw_chunk in typed_stream(request):
                        emitted = True
                        if isinstance(raw_chunk, StreamChunk):
                            chunk = raw_chunk
                        elif isinstance(raw_chunk, str):
                            # Tolerate simple test doubles and third-party
                            # providers that still yield plain text deltas.
                            chunk = StreamChunk(kind="text", text=raw_chunk)
                        elif isinstance(raw_chunk, Mapping):
                            chunk = StreamChunk(
                                kind="reasoning" if raw_chunk.get("kind") == "reasoning" else "text",
                                text=str(raw_chunk.get("text") or ""),
                            )
                        else:
                            continue
                        if chunk.kind == "reasoning":
                            typed_reasoning.append(chunk.text)
                        else:
                            typed_text.append(chunk.text)
                        yield chunk
                else:
                    # Legacy providers only implement the text hook. Forwarding
                    # their deltas as text keeps them valid without inventing a
                    # reasoning channel they never supplied.
                    async for text in provider.stream(request):
                        emitted = True
                        typed_text.append(text)
                        yield StreamChunk(kind="text", text=text)
                self.last_stream_provider = name
                self.last_stream_tool_calls = list(getattr(provider, "last_tool_calls", []) or [])
                self.last_stream_content = "".join(typed_text) or str(
                    getattr(provider, "last_content", "") or ""
                )
                self.last_stream_reasoning = "".join(typed_reasoning) or str(
                    getattr(provider, "last_reasoning", "") or ""
                )
                self.last_stream_usage = getattr(provider, "last_usage", None)
                return
            except Exception as exc:
                attempt_error = exc
                last_error = exc
                log.warning(
                    "model typed stream attempt failed",
                    extra={"provider": name, "attempt": attempt, "error": self._scrub(str(exc))[:200]},
                )
                if emitted:
                    # Partial typed output was already delivered; replaying it
                    # through the next provider would duplicate text/thoughts.
                    raise
            finally:
                await self._record(
                    name,
                    provider,
                    request,
                    getattr(provider, "last_usage", None),
                    (time.perf_counter() - started) * 1000,
                    attempt_error is None,
                    attempt=attempt,
                    stream=True,
                    error=attempt_error,
                )
        if (
            isinstance(last_error, ModelEmptyOutputError)
            and not _empty_output_retried
        ):
            log.warning("model empty output in stream; retrying with final-answer-only request")
            retry = self.final_answer_only_request(request)
            async for chunk in self.stream_events(retry, _empty_output_retried=True):
                yield chunk
            return
        raise last_error or ModelError("all model typed stream attempts failed")

    async def usage_report(self) -> list[dict[str, Any]]:
        """Return recorded model calls, newest first, with parsed metadata."""
        rows = await db.fetchall("SELECT * FROM model_usage ORDER BY id DESC")
        report: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.get("metadata") or "{}")
            except (TypeError, ValueError):
                item["metadata"] = {}
            report.append(item)
        return report

    async def aclose(self) -> None:
        """Release provider-owned HTTP clients."""
        for provider in self.providers.values():
            close = getattr(provider, "aclose", None)
            if close:
                await close()


__all__ = ["ModelRouter", "TASK_CATEGORIES", "classify_task"]
