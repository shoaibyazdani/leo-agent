"""Trusted, task-local turn context, origin and provenance.

These tests pin the Phase 3 authority boundary: principal/surface/source come
from a trusted entry point and live in a ContextVar, so concurrent turns cannot
inherit or overwrite each other's facts, and the app-global ``_turn_origin``
dict is only a read-only compatibility snapshot.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import (
    AgentSpec,
    ModelResponse,
    SurfaceCapabilities,
    ToolCall,
    ToolResult,
    TurnContext,
    current_turn_context,
    set_turn_context,
    turn_context,
)
from leo.core import LeoApp
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry


@pytest.fixture
async def app(monkeypatch: pytest.MonkeyPatch) -> LeoApp:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    application = LeoApp(settings)
    await application.initialize()
    yield application
    await application.shutdown()


def _reply(content: str = "ok") -> ModelResponse:
    return ModelResponse(content=content, model="mock-1", provider="mock")


async def test_concurrent_turns_keep_their_own_context(app: LeoApp) -> None:
    """Two overlapping turns never see each other's principal/surface/capability."""
    observed: dict[str, TurnContext] = {}
    started = 0
    barrier = asyncio.Event()

    async def fake_complete(request):  # type: ignore[no-untyped-def]
        nonlocal started
        started += 1
        if started >= 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), timeout=5)
        user = next(
            (message.content for message in reversed(request.messages)
             if message.role == "user"),
            "",
        )
        context = current_turn_context()
        assert context is not None
        observed[user] = context
        return _reply()

    app.router.complete = fake_complete  # type: ignore[method-assign]

    first = TurnContext(
        principal="user-one",
        channel="telegram",
        conversation_id="conv_one",
        thread_id="thread_one",
        target="chat_one",
        surface="telegram",
        origin="telegram",
        source="chat",
    )
    second = TurnContext(
        principal="user-two",
        channel="whatsapp_web",
        conversation_id="conv_two",
        surface="whatsapp_web",
        origin="whatsapp_web",
        source="chat",
    )

    await asyncio.gather(
        app.ask("first question", channel="telegram", turn=first),
        app.ask("second question", channel="whatsapp_web", turn=second),
    )

    first_seen = observed["first question"]
    second_seen = observed["second question"]
    assert first_seen.principal == "user-one"
    assert first_seen.surface == "telegram"
    assert first_seen.thread_id == "thread_one"
    assert first_seen.capabilities.can_present_card is True
    assert second_seen.principal == "user-two"
    assert second_seen.surface == "whatsapp_web"
    assert second_seen.capabilities.can_present_card is False
    assert second_seen.capabilities.can_rewrite is False


async def test_contextvar_isolation_between_tasks() -> None:
    """A bare ContextVar scope is task-local even when tasks interleave."""
    seen: dict[str, str] = {}
    hold = asyncio.Event()
    ready = 0

    async def worker(name: str, principal: str) -> None:
        nonlocal ready
        turn = TurnContext(
            principal=principal, channel="cli", surface="cli", source="chat",
        )
        token = set_turn_context(turn)
        try:
            ready += 1
            if ready >= 2:
                hold.set()
            await asyncio.wait_for(hold.wait(), timeout=5)
            seen[name] = current_turn_context().principal  # type: ignore[union-attr]
        finally:
            from leo.contracts import reset_turn_context

            reset_turn_context(token)

    await asyncio.gather(worker("a", "principal-a"), worker("b", "principal-b"))

    assert seen == {"a": "principal-a", "b": "principal-b"}
    assert current_turn_context() is None


async def test_chat_tool_calls_carry_trusted_provenance(app: LeoApp) -> None:
    """A chat ToolCall carries source/principal/thread/capabilities and the
    app-global origin remains only a compatibility snapshot."""
    captured: list[ToolCall] = []
    calls = 0

    async def fake_complete(request):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                content="working",
                model="mock-1",
                provider="mock",
                tool_calls=[{
                    "id": "call-1",
                    "name": "calculator",
                    "arguments": {"expression": "1+1"},
                }],
            )
        return _reply("done")

    async def fake_execute(call: ToolCall, approval_id=None):  # type: ignore[no-untyped-def]
        captured.append(call)
        return ToolResult(
            call_id=call.id, tool=call.name, success=True, output={"value": 2},
        )

    app.router.complete = fake_complete  # type: ignore[method-assign]
    app.executor.execute = fake_execute  # type: ignore[method-assign]

    turn = TurnContext(
        principal="operator-9",
        channel="telegram",
        conversation_id="conv_provenance",
        thread_id="topic-7",
        target="chat-9",
        surface="telegram",
        origin="telegram",
        source="chat",
    )
    result = await app.ask(
        "please calculate", channel="telegram",
        conversation_id="conv_provenance", turn=turn,
    )

    assert result["status"] == "completed"
    assert len(captured) == 1
    call = captured[0]
    assert call.source == "chat"
    assert call.origin == "telegram"
    assert call.principal == "operator-9"
    assert call.conversation_id == "conv_provenance"
    assert call.thread_id == "topic-7"
    assert call.capabilities["can_present_card"] is True
    # Compatibility snapshot only; it agrees with the trusted context here.
    assert app._turn_origin == {
        "channel": "telegram",
        "target": "chat-9",
        "conversation_id": "conv_provenance",
    }
    assert _has_context(current_turn_context()) is False  # reset after the turn


def _has_context(value) -> bool:  # type: ignore[no-untyped-def]
    """True when a context is still installed (the assertion expects False)."""
    return value is not None


async def test_executor_reads_turn_context_not_the_stale_app_snapshot() -> None:
    """The racy app-global origin is not consulted when a trusted turn exists."""
    registry = ToolRegistry()
    observed: dict = {}

    async def handler(args: dict, ctx: dict):  # type: ignore[no-untyped-def]
        observed.update(ctx)
        return {"ok": True}

    from leo.contracts import ToolSpec
    from leo.contracts import RiskLevel

    registry.register(
        ToolSpec(
            name="context_probe",
            description="read the trusted context",
            category="test",
            risk=RiskLevel.LOW,
            resource_field="",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        handler,
    )
    settings = Settings()
    events = EventBus()
    executor = ToolExecutorImpl(
        registry, PermissionManagerImpl(settings, events), events, settings,
        services={"app": SimpleNamespace(_turn_origin={
            "channel": "stale-channel", "target": "stale-target",
            "conversation_id": "stale-conversation",
        })},
    )
    trusted = TurnContext(
        principal="trusted-principal",
        channel="telegram",
        conversation_id="trusted-conversation",
        target="trusted-target",
        surface="telegram",
        origin="gateway:telegram",
        source="gateway",
    )
    with turn_context(trusted):
        result = await executor.execute(ToolCall(name="context_probe", actor="agent"))

    assert result.success
    assert observed["principal"] == "trusted-principal"
    assert observed["source"] == "gateway"
    assert observed["surface"] == "telegram"
    assert observed["origin_context"] == {
        "channel": "telegram",
        "target": "trusted-target",
        "conversation_id": "trusted-conversation",
    }
    assert observed["turn_context"]["principal"] == "trusted-principal"


async def test_agent_delegation_keeps_parent_identity_but_not_card_capability() -> None:
    """A delegated ToolCall inherits identity; it never inherits card authority."""
    from leo.agents.engine import AgentEngine, AgentRequest
    from leo.contracts import ToolSpec

    captured: list[ToolCall] = []
    calls = 0

    class FakeRouter:
        providers: dict = {}

        async def complete(self, request):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    content="working",
                    model="mock-1",
                    provider="mock",
                    tool_calls=[{
                        "id": "agent-call-1",
                        "name": "calculator",
                        "arguments": {"expression": "2+2"},
                    }],
                )
            return _reply(json.dumps({"output": "done"}))

    class FakeRegistry:
        @staticmethod
        def list_specs():  # type: ignore[no-untyped-def]
            return [ToolSpec(
                name="calculator", description="math", category="utility",
            )]

    class FakeTools:
        registry = FakeRegistry()

        async def execute(self, call, approval_id=None, scope=None):  # type: ignore[no-untyped-def]
            captured.append(call)
            return ToolResult(call_id=call.id, tool=call.name, success=True, output=4)

    spec = AgentSpec(
        id="calc-worker",
        name="Calc Worker",
        purpose="do math",
        tool_permissions=["calculator"],
        model_preferences=[],
        context_budget=8000,
    )
    engine = AgentEngine(
        registry=SimpleNamespace(),
        router=FakeRouter(),
        tools=FakeTools(),
        settings=Settings(),
    )
    request = AgentRequest(objective="add", context={"correlation_id": "corr-1"})

    parent = TurnContext(
        principal="parent-principal",
        channel="telegram",
        conversation_id="parent-conv",
        thread_id="parent-thread",
        surface="telegram",
        origin="telegram",
        source="chat",
    )
    with turn_context(parent):
        await engine._model_worker(spec, request, request.context)

    assert len(captured) == 1
    call = captured[0]
    assert call.source == "agent"  # attended parent -> nested side agent
    assert call.origin == "agent:calc-worker"
    assert call.principal == "parent-principal"
    assert call.conversation_id == "parent-conv"
    assert call.thread_id == "parent-thread"
    # Delegated work can never present or answer a card.
    assert call.capabilities["can_present_card"] is False
    assert call.capabilities["can_wait_for_decision"] is False

    # A scheduled parent keeps the unattended source so the executor gate sees it.
    captured.clear()
    calls = 0
    scheduled = TurnContext(
        principal="scheduled",
        channel="scheduled",
        surface="scheduled",
        origin="scheduled",
        source="scheduled",
    )
    with turn_context(scheduled):
        await engine._model_worker(spec, request, request.context)
    assert captured[0].source == "scheduled"


async def test_ask_stream_installs_and_resets_the_context(app: LeoApp) -> None:
    """The streaming wrapper keeps the trusted context across yields, then resets."""
    seen: list[TurnContext | None] = []

    async def fake_impl(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(current_turn_context())
        yield {"type": "run.started"}
        seen.append(current_turn_context())
        yield {"type": "run.finished"}

    app._ask_stream_turn = fake_impl  # type: ignore[method-assign]
    turn = TurnContext(
        principal="stream-user",
        channel="web",
        conversation_id="conv_stream",
        surface="web",
        origin="web",
        source="api",
    )
    events = [event async for event in app.ask_stream("hello", turn=turn)]

    assert [event["type"] for event in events] == ["run.started", "run.finished"]
    assert seen and all(item is not None and item.principal == "stream-user" for item in seen)
    assert current_turn_context() is None


def test_surface_manifest_defaults_to_incapable() -> None:
    from leo.contracts import surface_capabilities

    unknown = surface_capabilities("brand-new-adapter")
    assert unknown.can_present_card is False
    assert unknown.can_wait_for_decision is False
    assert unknown.full_text is False
    assert unknown.can_rewrite is False
    assert unknown.max_card_chars is None
    # The manifest text cannot be altered through a shared instance.
    assert surface_capabilities("a2a").can_present_card is False
    assert surface_capabilities("whatsapp_web").can_present_card is False
    assert surface_capabilities("cli").can_present_card is False
    assert surface_capabilities("api").can_present_card is False
    assert SurfaceCapabilities().model_dump()["can_present_card"] is False
