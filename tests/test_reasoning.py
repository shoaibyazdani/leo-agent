"""Tests for the /reasoning port: ladder, clamp, wire params and command."""

from __future__ import annotations

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ChatMessage, ModelRequest
from leo.models.provider import AnthropicProvider, OpenAIProvider
from leo.models.reasoning import (
    EFFORT_LADDER,
    clamp_effort,
    parse_effort,
    thinking_budget,
)


# --- vocabulary / clamp ----------------------------------------------------

def test_parse_effort_normalizes_and_rejects():
    assert parse_effort("HIGH") == "high"
    assert parse_effort(" off ") == "none"
    assert parse_effort("min") == "minimal"
    assert parse_effort("extreme") == "max"
    assert parse_effort("nonsense") is None
    assert parse_effort("") is None


def test_clamp_never_escalates_cost():
    # ultra is internal; widest wire stops at max -> nearest weaker is max.
    assert clamp_effort("ultra", ("none", "low", "medium", "high", "max")) == "max"
    # high unsupported -> nearest weaker supported level.
    assert clamp_effort("high", ("none", "low", "medium")) == "medium"
    # nothing weaker -> weakest supported level.
    assert clamp_effort("minimal", ("low", "medium", "high")) == "low"
    # already supported / unknown set / unrecognized -> pass through.
    assert clamp_effort("medium", ("low", "medium", "high")) == "medium"
    assert clamp_effort("ultra", None) == "ultra"
    assert clamp_effort("custom", ("low", "high")) == "custom"
    assert clamp_effort(None, ("low",)) is None
    # An enabled ask must never silently become "no reasoning".
    assert clamp_effort("minimal", ("none", "low")) == "low"
    assert clamp_effort("low", ("none",)) == "low"
    # An explicit disable is not an effort to clamp.
    assert clamp_effort("none", ("low", "medium")) == "none"
    # A declared vendor mapping wins over nearest-level math.
    assert clamp_effort("medium", ("low", "high", "max"), {"medium": "high"}) == "high"
    # A mapping to a level the wire does not publish is ignored.
    assert clamp_effort("medium", ("low", "high"), {"medium": "xhigh"}) == "low"


def test_thinking_budget_bounds():
    assert thinking_budget("none", 32_000) is None
    assert thinking_budget(None, 32_000) is None
    assert thinking_budget("high", 32_000) == 8_192
    # Never exceeds the provider ceiling.
    assert thinking_budget("max", 4_000) <= 4_000 - 1_024
    # No legal budget exists under the API's 1024-token minimum.
    assert thinking_budget("high", 1_024) is None
    assert thinking_budget("high", 256) is None


# --- wire params -----------------------------------------------------------

def _request(effort: str | None) -> ModelRequest:
    return ModelRequest(
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort=effort,
    )


def _openai_cfg(**kwargs) -> ModelProviderConfig:
    return ModelProviderConfig(
        kind="openai-compatible", base_url="https://example.test/v1",
        model="m", api_key_env=None, **kwargs,
    )


def test_openai_body_adds_reasoning_effort_only_when_set():
    provider = OpenAIProvider(_openai_cfg(), name="m")
    assert "reasoning_effort" not in provider._body(_request(None), stream=False)
    assert provider._body(_request("high"), stream=False)["reasoning_effort"] == "high"
    assert provider._body(_request("off"), stream=False)["reasoning_effort"] == "none"


def test_openai_body_clamps_to_declared_wire():
    cfg = _openai_cfg(reasoning_efforts=["none", "low", "medium", "high"])
    provider = OpenAIProvider(cfg, name="m")
    assert provider._body(_request("ultra"), stream=False)["reasoning_effort"] == "high"


def test_anthropic_body_adds_thinking_block():
    provider = AnthropicProvider(
        ModelProviderConfig(kind="anthropic-compatible", base_url="https://example.test", model="m"),
        name="m",
    )
    provider.max_tokens = 32_000
    assert "thinking" not in provider._body(_request(None), stream=False)
    body = provider._body(_request("high"), stream=False)
    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] == 8_192


def test_precedence_request_over_provider_over_global():
    cfg = _openai_cfg(reasoning_effort="low")
    provider = OpenAIProvider(cfg, name="m")
    provider.default_effort = "medium"
    # request wins
    assert provider._body(_request("high"), stream=False)["reasoning_effort"] == "high"
    # provider config wins over global default
    assert provider._body(_request(None), stream=False)["reasoning_effort"] == "low"
    # global default used only when nothing else set
    provider.config.reasoning_effort = None
    assert provider._body(_request(None), stream=False)["reasoning_effort"] == "medium"


# --- command ---------------------------------------------------------------

@pytest.fixture
async def app(monkeypatch):
    from leo.core import LeoApp

    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    instance = LeoApp(settings)
    await instance.initialize()
    yield instance
    await instance.shutdown()


async def _call(app, text):
    return await app.commands.dispatch(
        text, app=app, principal="local", surface="cli", channel="cli",
        conversation_id="conv-reasoning",
    )


async def test_reasoning_command_is_available(app):
    spec = app.commands.get("reasoning")
    assert spec is not None
    assert spec.available is True


async def test_reasoning_command_set_and_show(app):
    result = await _call(app, "/reasoning high")
    assert result.success is True
    assert "high" in result.message

    shown = await _call(app, "/reasoning")
    assert shown.success is True
    assert "high" in shown.message
    assert shown.data["effort"] == "high"

    # The session effort is applied to chat requests.
    assert await app.reasoning_state.effort_for("conv-reasoning") == "high"


async def test_reasoning_command_display_toggles(app):
    assert (await _call(app, "/reasoning show")).data["show"] is True
    assert (await _call(app, "/reasoning full")).data["full"] is True
    assert (await _call(app, "/reasoning clamp")).data["full"] is False
    assert (await _call(app, "/reasoning hide")).data["show"] is False


async def test_reasoning_command_rejects_unknown_level(app):
    result = await _call(app, "/reasoning banana")
    assert result.success is False
    assert "unknown reasoning level" in result.message


async def test_reasoning_command_reset(app):
    await _call(app, "/reasoning high")
    result = await _call(app, "/reasoning reset")
    assert result.success is True
    assert await app.reasoning_state.effort_for("conv-reasoning") is None


# --- install / display / scope ---------------------------------------------


def _level_buttons(markup: dict) -> dict[str, str]:
    buttons: dict[str, str] = {}
    for row in markup.get("inline_keyboard") or []:
        for button in row:
            buttons[str(button.get("callback_data") or "")] = str(button.get("text") or "")
    return buttons


async def _call_conv(app, text, conversation_id="conv-reasoning"):
    return await app.commands.dispatch(
        text, app=app, principal="local", surface="cli", channel="cli",
        conversation_id=conversation_id,
    )


async def test_reasoning_menu_is_interactive_and_shows_effective_state(app):
    result = await _call(app, "/reasoning")
    assert result.success is True
    assert "Active effort" in result.message
    assert "Edit scope" in result.message
    buttons = _level_buttons(result.data["reply_markup"])
    for level in EFFORT_LADDER:
        assert f"rsn:level:{level}" in buttons, level
    assert "rsn:scope:global" in buttons or "rsn:scope:session" in buttons
    assert "rsn:show" in buttons
    assert "rsn:reset" in buttons and "rsn:reset-global" in buttons


async def test_reasoning_scope_switches_where_edits_land(app):
    await _call(app, "/reasoning scope global")
    result = await _call(app, "/reasoning high")
    assert result.data["scope"] == "global"
    assert await app.reasoning_state.global_effort() == "high"
    # Another conversation with no session override resolves the global value.
    other = await app.reasoning_selection("conv-other")
    assert other["effort"] == "high"
    assert other["effort_source"] == "global"

    await _call(app, "/reasoning scope session")
    await _call(app, "/reasoning low")
    assert await app.reasoning_state.effort_for("conv-reasoning") == "low"
    other = await app.reasoning_selection("conv-other")
    assert other["effort"] == "high"  # no leakage into the other conversation


async def test_reasoning_global_survives_a_new_state_object(app):
    from leo.models.reasoning_state import ReasoningState

    await _call(app, "/reasoning global")
    await _call(app, "/reasoning xhigh")
    fresh = ReasoningState()
    assert await fresh.global_effort() == "xhigh"
    assert (await fresh.resolve("never-seen"))["effort_source"] == "global"


async def test_reasoning_display_is_scope_aware_and_isolated(app):
    await _call(app, "/reasoning global")
    shown = await _call(app, "/reasoning show")
    assert shown.data["show"] is True
    assert (await app.reasoning_state.global_state())["show"] is True

    # A session override can hide it for one conversation only.
    await _call(app, "/reasoning session")
    await _call(app, "/reasoning hide")
    mine = await app.reasoning_selection("conv-reasoning")
    other = await app.reasoning_selection("conv-other")
    assert mine["show"] is False and mine["show_source"] == "session"
    assert other["show"] is True and other["show_source"] == "global"


async def test_reasoning_reset_clears_only_session_overrides(app):
    await _call(app, "/reasoning global")
    await _call(app, "/reasoning high")
    await _call(app, "/reasoning session")
    await _call(app, "/reasoning low")
    await _call(app, "/reasoning show")
    result = await _call(app, "/reasoning reset")
    assert result.success is True
    state = await app.reasoning_state.get("conv-reasoning")
    assert state["effort"] is None and state["show"] is None
    assert await app.reasoning_state.global_effort() == "high"


# --- real requests ---------------------------------------------------------


class _RecordingProvider:
    """Deterministic provider that records every request it receives."""

    def __init__(self):
        from leo.models.mock import MockProvider

        self._inner = MockProvider(
            ModelProviderConfig(kind="mock", model="recording-1",
                                capabilities=["text", "streaming", "tools", "structured", "usage"]),
            name="recording",
        )
        self.config = self._inner.config
        self.name = self.inner_name = "recording"
        self.model = "recording-1"
        self.capabilities = self._inner.capabilities
        self.context_window = self._inner.context_window
        self.requests: list[ModelRequest] = []
        self.summary = ""
        self.raw = ""
        self.last_usage = None
        self.last_tool_calls = []
        self.last_content = ""
        self.last_reasoning = ""
        self.last_reasoning_summary = ""

    async def complete(self, request):
        from leo.contracts import ModelResponse, Usage

        self.requests.append(request)
        self.last_content = "answer"
        self.last_reasoning = self.raw
        self.last_reasoning_summary = self.summary if request.reasoning_show else ""
        self.last_usage = Usage(input_tokens=1, output_tokens=1, total_tokens=2)
        return ModelResponse(
            content="answer", model="recording-1", provider="recording",
            reasoning=self.raw, reasoning_summary=self.last_reasoning_summary,
            usage=self.last_usage,
        )

    async def stream_events(self, request):
        from leo.contracts import StreamChunk

        self.requests.append(request)
        if request.reasoning_show and self.summary:
            yield StreamChunk(kind="reasoning", text=self.summary)
        yield StreamChunk(kind="text", text="answer")
        self.last_content = "answer"
        self.last_reasoning_summary = self.summary if request.reasoning_show else ""

    def __repr__(self):
        return "RecordingProvider()"


async def _attach_recording(app, *, summary="", raw="RAW CHAIN OF THOUGHT"):
    provider = _RecordingProvider()
    provider.summary = summary
    provider.raw = raw
    app.router.providers = {"recording": provider}
    app.settings.models.providers["recording"] = provider.config
    app.settings.models.default = "recording"
    app.settings.models.fallback = "recording"
    return provider


async def test_reasoning_session_effort_reaches_the_actual_request(app):
    provider = await _attach_recording(app)
    await _call(app, "/reasoning high")
    await app.ask("hello", channel="cli", conversation_id="conv-reasoning")
    request = provider.requests[-1]
    assert request.reasoning_effort == "high"
    assert request.reasoning_source == "session"
    assert request.reasoning_show is False


async def test_reasoning_global_reaches_another_conversation(app):
    provider = await _attach_recording(app)
    await _call(app, "/reasoning global")
    await _call(app, "/reasoning medium")
    await app.ask("hello", channel="cli", conversation_id="another-conv")
    request = provider.requests[-1]
    assert request.reasoning_effort == "medium"
    assert request.reasoning_source == "global"


async def test_reasoning_summary_is_shown_but_raw_thought_never_is(app):
    provider = await _attach_recording(app, summary="Checked the inputs.",
                                       raw="RAW CHAIN OF THOUGHT")
    await _call(app, "/reasoning high")
    await _call(app, "/reasoning show")
    result = await app.ask("hello", channel="cli", conversation_id="conv-reasoning")
    assert "Checked the inputs." in result["content"]
    assert "RAW CHAIN OF THOUGHT" not in result["content"]
    assert provider.requests[-1].reasoning_show is True


async def test_reasoning_display_off_suppresses_summary_in_reply(app):
    await _attach_recording(app, summary="Checked the inputs.")
    await _call(app, "/reasoning high")
    await _call(app, "/reasoning hide")
    result = await app.ask("hello", channel="cli", conversation_id="conv-reasoning")
    assert "Checked the inputs." not in result["content"]
    assert "🧠 Reasoning" not in result["content"]


# --- provider wire params --------------------------------------------------


def test_adaptive_provider_sends_effort_and_summary_display():
    from leo.contracts import ChatMessage, ModelRequest

    provider = AnthropicProvider(
        ModelProviderConfig(kind="anthropic-compatible", base_url="https://example.test",
                            model="claude-opus-5-5"),
        name="claude",
    )
    provider.max_tokens = 32_000
    request = ModelRequest(
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="high", reasoning_show=True,
    )
    body = provider._body(request, stream=False)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "high"

    hidden = provider._body(
        ModelRequest(messages=[ChatMessage(role="user", content="hi")],
                     reasoning_effort="high", reasoning_show=False),
        stream=False,
    )
    assert hidden["thinking"]["display"] == "omitted"


def test_reasoning_levels_are_clamped_downward_and_reported():
    from leo.contracts import ChatMessage, ModelRequest

    provider = AnthropicProvider(
        ModelProviderConfig(kind="anthropic-compatible", base_url="https://example.test",
                            model="claude-opus-5-5"),
        name="claude",
    )
    request = ModelRequest(
        messages=[ChatMessage(role="user", content="hi")], reasoning_effort="ultra"
    )
    body = provider._body(request, stream=False)
    assert body["output_config"]["effort"] == "max"
    assert provider.last_reasoning_requested == "ultra"
    assert provider.last_reasoning_applied == "max"
    assert "mapped to max" in provider.last_reasoning_note


def test_reasoning_none_disables_adaptive_thinking():
    from leo.contracts import ChatMessage, ModelRequest

    provider = AnthropicProvider(
        ModelProviderConfig(kind="anthropic-compatible", base_url="https://example.test",
                            model="claude-opus-5-5"),
        name="claude",
    )
    request = ModelRequest(
        messages=[ChatMessage(role="user", content="hi")], reasoning_effort="none"
    )
    body = provider._body(request, stream=False)
    assert body["thinking"] == {"type": "disabled"}
    assert "output_config" not in body


@pytest.mark.asyncio
async def test_provider_rejection_falls_back_once_and_is_remembered(monkeypatch):
    """A gateway that rejects thinking still answers, and is not retried forever."""
    import httpx

    from leo.contracts import ChatMessage, ModelRequest

    monkeypatch.setenv("LEO_TEST_ANTHROPIC_KEY", "test-token")
    bodies: list[dict] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(http_request.content)
        bodies.append(body)
        if "thinking" in body:
            return httpx.Response(
                400, json={"error": {"message": "unsupported parameter: thinking"}},
                request=http_request,
            )
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "ok"}],
                       "usage": {"input_tokens": 1, "output_tokens": 1}},
            request=http_request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            ModelProviderConfig(
                kind="anthropic-compatible", base_url="https://example.test",
                model="claude-opus-5-5", api_key_env="LEO_TEST_ANTHROPIC_KEY", max_retries=0,
            ),
            name="claude", client=client,
        )
        request = ModelRequest(
            messages=[ChatMessage(role="user", content="hi")],
            reasoning_effort="high", reasoning_show=True,
        )
        response = await provider.complete(request)
        assert response.content == "ok"
        assert "thinking" not in bodies[-1]
        assert "rejected reasoning" in provider.last_reasoning_note

        # The rejection is remembered: no third request pays for it again.
        await provider.complete(request)
        assert "thinking" not in bodies[-1]
        assert len(bodies) == 3


# --- subagent inheritance --------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_inherits_parent_selection_and_allows_override():
    from types import SimpleNamespace

    from leo.agents.engine import AgentEngine, AgentRequest
    from leo.contracts import AgentSpec, ModelResponse
    from leo.models.reasoning_state import reasoning_scope

    class RecordingRouter:
        providers: dict = {}

        def __init__(self):
            self.requests: list[ModelRequest] = []

        async def complete(self, request):
            self.requests.append(request)
            return ModelResponse(content="ok", model="m", provider="p")

    router = RecordingRouter()
    engine = AgentEngine(registry=SimpleNamespace(), router=router,
                         events=SimpleNamespace(), settings=Settings())
    spec = AgentSpec(id="scout", name="Scout", purpose="look", tool_permissions=[])

    inherited = {"effort": "xhigh", "show": True, "source": "session", "full": False}
    with reasoning_scope(inherited):
        await engine._model_worker(spec, AgentRequest(objective="look"), {"task_id": "t1"})
    assert router.requests[-1].reasoning_effort == "xhigh"
    assert router.requests[-1].reasoning_show is True
    assert router.requests[-1].reasoning_source == "session"

    explicit = AgentRequest(
        objective="look",
        context={"reasoning": {"effort": "low", "show": False, "source": "context"}},
    )
    with reasoning_scope(inherited):
        await engine._model_worker(spec, explicit, {"task_id": "t2"})
    assert router.requests[-1].reasoning_effort == "low"
    assert router.requests[-1].reasoning_show is False


# --- callback tap grammar --------------------------------------------------


def test_reasoning_tap_payloads_are_allowlisted():
    from leo.gateway.callbacks import reasoning_tap_command

    assert reasoning_tap_command("level:high") == "/reasoning high"
    assert reasoning_tap_command("scope:global") == "/reasoning scope global"
    assert reasoning_tap_command("show") == "/reasoning show"
    assert reasoning_tap_command("reset") == "/reasoning reset --session"
    assert reasoning_tap_command("reset-global") == "/reasoning reset --global"
    # Anything not explicitly allowed is rejected, not relayed.
    assert reasoning_tap_command("level:root") is None
    assert reasoning_tap_command("scope:admin") is None
    assert reasoning_tap_command("../../etc/passwd") is None


@pytest.mark.asyncio
async def test_reasoning_tap_relays_through_the_command_queue():
    from leo.gateway.callbacks import handle_tap

    class Adapter:
        channel = "telegram"

        def __init__(self):
            self.answers: list[str] = []

        async def answer_callback_query(self, callback_id, text=""):
            self.answers.append(str(text))
            return True

    adapter = Adapter()
    admitted: list[tuple[str, str, str]] = []

    async def admit(text, channel, subject):
        admitted.append((text, channel, subject))

    result = await handle_tap(
        "telegram",
        {"callback_id": "cb-1", "data": "rsn:level:high", "sender_id": "42",
         "chat_id": "84", "message_id": "9"},
        settings=Settings(), permissions=None, adapters={"telegram": adapter},
        allowed=["42"], admit=admit,
    )
    assert result["ok"] is True
    assert admitted == [("/reasoning high", "telegram", "84")]
    assert adapter.answers and "Applying" in adapter.answers[-1]


@pytest.mark.asyncio
async def test_reasoning_tap_rejects_unknown_payload_without_relaying():
    from leo.gateway.callbacks import handle_tap

    class Adapter:
        channel = "telegram"

        def __init__(self):
            self.answers: list[str] = []

        async def answer_callback_query(self, callback_id, text=""):
            self.answers.append(str(text))
            return True

    adapter = Adapter()
    admitted: list[str] = []

    async def admit(text, channel, subject):
        admitted.append(text)

    result = await handle_tap(
        "telegram",
        {"callback_id": "cb-1", "data": "rsn:level:root", "sender_id": "42",
         "chat_id": "84", "message_id": "9"},
        settings=Settings(), permissions=None, adapters={"telegram": adapter},
        allowed=["42"], admit=admit,
    )
    assert result["ok"] is False
    assert admitted == []
    assert "Unrecognised" in adapter.answers[-1]


@pytest.mark.asyncio
async def test_streaming_reply_shows_only_a_summary(app):
    await _attach_recording(app, summary="Checked the inputs.", raw="RAW CHAIN OF THOUGHT")
    await _call(app, "/reasoning high")
    await _call(app, "/reasoning show")
    events = [event async for event in app.ask_stream(
        "hello", channel="cli", conversation_id="conv-reasoning"
    )]
    thinking = "".join(
        str(event.get("delta") or "") for event in events
        if event.get("type") == "thinking.delta"
    )
    finished = next(event for event in events if event.get("type") == "message.finished")
    assert thinking == "Checked the inputs."
    # The streaming answer content stays clean; the summary lives in its own
    # thinking channel, never in the answer text.
    assert finished["content"] == "answer"
    assert "RAW CHAIN OF THOUGHT" not in thinking
    assert finished["reasoning_effort"] == "high"
    assert finished["reasoning_display"] == "summary"


@pytest.mark.asyncio
async def test_streaming_reply_hides_reasoning_when_display_is_off(app):
    await _attach_recording(app, summary="Checked the inputs.")
    await _call(app, "/reasoning high")
    await _call(app, "/reasoning hide")
    events = [event async for event in app.ask_stream(
        "hello", channel="cli", conversation_id="conv-reasoning"
    )]
    assert not any(event.get("type") == "thinking.delta" for event in events)
    finished = next(event for event in events if event.get("type") == "message.finished")
    assert "Checked the inputs." not in finished["content"]
    assert finished["reasoning_display"] == "hidden"


@pytest.mark.asyncio
async def test_clearing_global_restores_the_configured_default(app):
    app.settings.models.reasoning_effort = "low"
    app._configured_reasoning_effort = "low"
    await _call(app, "/reasoning global")
    await _call(app, "/reasoning high")
    assert (await app.reasoning_selection("conv-reasoning"))["effort"] == "high"

    result = await _call(app, "/reasoning reset --global")
    assert result.success is True
    selection = await app.reasoning_selection("conv-reasoning")
    assert selection["effort"] == "low"
    assert selection["effort_source"] == "config"
    assert app.settings.models.reasoning_effort == "low"


@pytest.mark.asyncio
async def test_session_write_without_a_conversation_is_refused(app):
    result = await _call_conv(app, "/reasoning high", conversation_id=None)
    assert result.success is False
    assert "--global" in result.message
    assert await app.reasoning_state.effort_for("conv-reasoning") is None
    assert await app.reasoning_state.effort_for(None) is None


@pytest.mark.asyncio
async def test_single_process_telegram_tap_changes_the_session_and_returns_a_menu(app):
    """End to end: button -> slash command -> state -> reply with a keyboard."""
    from leo import db
    from leo.config import ChannelPolicyConfig

    class Adapter:
        channel = "telegram"

        def __init__(self):
            self.sent: list[dict] = []
            self.answers: list[str] = []

        async def answer_callback_query(self, callback_id, text=""):
            self.answers.append(str(text))
            return True

        async def send(self, target, text, **kwargs):
            self.sent.append({"target": str(target), "text": str(text), **kwargs})
            return {"ok": True}

    adapter = Adapter()
    app.gateway.adapters["telegram"] = adapter
    app.settings.channels.policies["telegram"] = ChannelPolicyConfig(
        mode="allowlist", allow=["100000001"]
    )
    await db.execute(
        "INSERT INTO conversations(id,channel,subject,created_at,updated_at) "
        "VALUES(?,?,?,?,?)",
        ("conv-tap", "telegram", "100000001", "2026-01-01T00:00:00+00:00",
         "2026-01-01T00:00:00+00:00"),
    )

    await app.handle_channel_callback(
        "telegram",
        {"callback_id": "cb-1", "data": "rsn:level:high", "sender_id": "100000001",
         "chat_id": "100000001", "message_id": "9"},
    )

    assert await app.reasoning_state.effort_for("conv-tap") == "high"
    assert adapter.sent
    assert "Active effort: high" in adapter.sent[-1]["text"]
    markup = adapter.sent[-1].get("reply_markup") or {}
    callbacks = {
        button.get("callback_data")
        for row in markup.get("inline_keyboard") or []
        for button in row
    }
    assert "rsn:level:medium" in callbacks


def test_route_preview_reports_clamping_before_a_request_is_sent():
    provider = AnthropicProvider(
        ModelProviderConfig(kind="anthropic-compatible", base_url="https://example.test",
                            model="claude-opus-5-5"),
        name="claude",
    )
    preview = provider.describe_reasoning("ultra")
    assert preview["requested"] == "ultra"
    assert preview["applied"] == "max"
    assert preview["style"] == "effort"
    assert "mapped to max" in preview["note"]
    # Previewing must not overwrite the last real request's reported state.
    assert provider.last_reasoning_requested is None


def test_glued_reasoning_summary_parts_get_a_paragraph_break():
    from leo.models.provider import _separate_glued_summary

    assert _separate_glued_summary("**First step**", "**Second step**") == (
        "\n\n**Second step**"
    )
    assert _separate_glued_summary("**First step** ", "**Second**") == "**Second**"
    assert _separate_glued_summary("plain text", "next word") == "next word"


@pytest.mark.asyncio
async def test_a_refused_disable_steps_up_to_the_lowest_effort(monkeypatch):
    """A provider that cannot turn thinking off gets the floor, not its own default."""
    import httpx

    from leo.contracts import ChatMessage, ModelRequest

    monkeypatch.setenv("LEO_TEST_OPENAI_KEY", "test-token")
    bodies: list[dict] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(http_request.content)
        bodies.append(body)
        if body.get("reasoning_effort") == "none":
            return httpx.Response(
                400,
                json={"error": {"message": "reasoning cannot be disabled on this model"}},
                request=http_request,
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"},
                                    "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            request=http_request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(_openai_cfg(max_retries=0), name="openai", client=client)
        request = ModelRequest(
            messages=[ChatMessage(role="user", content="hi")], reasoning_effort="none"
        )
        response = await provider.complete(request)
        assert response.content == "ok"
        assert bodies[-1]["reasoning_effort"] == "low"
        assert "cannot disable reasoning" in provider.last_reasoning_note

        # The floor is remembered: the next disabled request does not pay the 400 again.
        await provider.complete(request)
        assert bodies[-1]["reasoning_effort"] == "low"
        assert len(bodies) == 3


@pytest.mark.asyncio
async def test_global_display_survives_a_new_state_object(app):
    from leo.models.reasoning_state import ReasoningState

    await _call(app, "/reasoning scope global")
    await _call(app, "/reasoning show")
    fresh = ReasoningState()
    selection = await fresh.resolve("never-seen")
    assert selection["show"] is True
    assert selection["show_source"] == "global"


@pytest.mark.asyncio
async def test_status_reports_effective_reasoning(app):
    await _call(app, "/reasoning high")
    await _call(app, "/reasoning show")
    result = await _call(app, "/status")
    assert result.success is True
    assert result.data["reasoning"]["effort"] == "high"
    assert result.data["reasoning"]["effort_source"] == "session"
    assert result.data["reasoning"]["show"] is True
    assert "reasoning=high" in result.message


@pytest.mark.asyncio
async def test_reasoning_tap_queues_a_durable_admission_for_the_brain():
    """Split deployment: the gateway queues work instead of writing settings itself."""
    from leo.gateway import GatewayStore
    from leo.gateway.callbacks import handle_tap

    class Adapter:
        channel = "telegram"

        async def answer_callback_query(self, callback_id, text=""):
            return True

    store = GatewayStore()

    async def admit(text, channel, subject):
        await store.admit(
            channel=channel, subject=subject,
            payload={"text": text, "message_id": f"cb-{text}", "sender_id": subject,
                     "conversation_ref": subject},
            principal=subject, request_id=f"cb-{text}",
        )

    result = await handle_tap(
        "telegram",
        {"callback_id": "cb-queue", "data": "rsn:level:high", "sender_id": "42",
         "chat_id": "84", "message_id": "9"},
        settings=Settings(), permissions=None, adapters={"telegram": Adapter()},
        allowed=["42"], admit=admit,
    )

    assert result["ok"] is True
    claimed = await store.claim("worker-a")
    assert claimed is not None
    assert claimed["channel"] == "telegram"
    assert claimed["subject"] == "84"
    assert claimed["payload"]["text"] == "/reasoning high"
