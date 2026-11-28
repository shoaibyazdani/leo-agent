"""Issue #83: per-task tool scoping, skill recommendation and cap handoffs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import (
    ChatMessage,
    ModelRequest,
    ModelResponse,
    ToolResult,
    Usage,
)
from leo.core import LeoApp
from leo.models import ModelRouter
from leo.task_scope import (
    CHAT,
    GENERAL,
    READ,
    SHELL,
    cap_report,
    classify_tool_scope,
    estimate_tool_spec_tokens,
    match_skill,
    select_tool_specs,
)
from leo.tools import register_all_toolsets
from leo.tools.builtins import register_builtins
from leo.tools.registry import ToolRegistry
from leo.tools.skill_read_marks import reset_skill_read_marks, was_skill_read


# --------------------------------------------------------------------------- R1


def test_classifier_gives_no_tools_to_plain_chat_and_scopes_work() -> None:
    assert classify_tool_scope("Hello there!") == CHAT
    assert classify_tool_scope("") == CHAT
    assert classify_tool_scope("Read the file and tell me what it says") == READ
    assert classify_tool_scope("Run the shell command ls -la") == SHELL
    assert classify_tool_scope("Search the web for the current docs") != SHELL


def _specs() -> list[Any]:
    registry = ToolRegistry()
    register_builtins(registry)
    register_all_toolsets(registry, services={})
    return list(registry.list_specs())


@pytest.mark.parametrize(
    ("scope", "included", "excluded"),
    [
        (CHAT, (), ("read_file", "shell", "skill_manage")),
        (READ, ("read_file", "search_files"), ("write_file", "shell", "skill_manage")),
        (SHELL, ("shell", "terminal", "read_file", "write_file"),
         ("skill_manage", "agent_delegate", "channel_send")),
    ],
)
def test_scope_sends_only_relevant_specs(
    scope: str, included: tuple[str, ...], excluded: tuple[str, ...]
) -> None:
    specs = _specs()
    selected = select_tool_specs(specs, scope)
    names = {spec["name"] for spec in selected}
    assert set(included) <= names
    assert not (set(excluded) & names)
    assert len(selected) < len(specs), "the full catalogue is never the default"
    assert [spec["name"] for spec in selected] == [
        spec.name for spec in specs if spec.name in names
    ], "registry order is preserved"


class _CapturingRouter:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content="Done.", model="fake", provider="fake",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


async def _app_with_router(tmp_path: Path) -> tuple[LeoApp, _CapturingRouter]:
    settings = Settings()
    settings.max_chat_tool_rounds = 3
    settings.models = ModelConfig(
        default="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    app = LeoApp(settings)
    await app.initialize()
    router = _CapturingRouter()
    app.router = router
    return app, router


@pytest.mark.asyncio
async def test_chat_turn_advertises_no_tools_and_shell_turn_only_shell_and_files() -> None:
    app, router = await _app_with_router(Path())
    try:
        await app.ask("Hello there", channel="test")
        assert router.requests[-1].tools == []

        await app.ask("Run the shell command ls", channel="test")
        names = {spec["name"] for spec in router.requests[-1].tools}
        assert {"shell", "terminal", "read_file", "list_files"} <= names
        assert not (names & {"skill_manage", "agent_delegate", "channel_send"})

        await app.ask("Read the file notes.txt", channel="test")
        names = {spec["name"] for spec in router.requests[-1].tools}
        assert "read_file" in names and "write_file" not in names and "shell" not in names
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- R2


def test_skill_match_uses_index_entries_and_rejects_plain_chat() -> None:
    entries = [
        {"name": "summarize_text", "description": "Extractive summary of a text using local word-frequency scoring.", "category": "research"},
        {"name": "system_health", "description": "Report local runtime health information for the current process.", "category": "infrastructure"},
    ]
    match = match_skill("Please summarize this text", entries)
    assert match and match["name"] == "summarize_text"
    assert match_skill("Hello there", entries) is None
    assert match_skill("", entries) is None


@pytest.mark.asyncio
async def test_matched_skill_procedure_is_loaded_before_work_starts() -> None:
    app, router = await _app_with_router(Path())
    reset_skill_read_marks()
    try:
        await app.ask("Summarize this text for me", channel="test")
        system = next(
            m.content for m in router.requests[-1].messages if m.role == "system"
        )
        assert "RECOMMENDED SKILL: summarize_text" in system
        # The full documented procedure (not just the 60-character trigger) is
        # in the very first request, before any model/tool round has happened.
        assert "Pure Python. Does not call a model or network service." in system
        # The automatic read goes through skill_read but is not a deliberate
        # pre-write read, so it cannot unlock a later skill patch by itself.
        assert not was_skill_read("summarize_text")
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_no_skill_match_proceeds_without_a_procedure() -> None:
    app, router = await _app_with_router(Path())
    try:
        await app.ask("Hello there", channel="test")
        system = next(
            m.content for m in router.requests[-1].messages if m.role == "system"
        )
        assert "RECOMMENDED SKILL:" not in system
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- R3


def test_cap_report_covers_done_unfinished_and_next_options() -> None:
    text = cap_report(
        reason="tool-round cap reached",
        done=["read_file", "search_files"],
        unfinished=["patch"],
    )
    assert "Tool budget exhausted" in text
    assert "Done so far: 2 operation(s): read_file, search_files" in text
    assert "Not finished: pending tool calls: patch" in text
    assert "Next options:" in text and "continue" in text and "narrow" in text and "split" in text


class _EndlessRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content="still working", model="fake", provider="fake",
            tool_calls=[{"id": f"c{self.calls}", "name": "calculator",
                         "arguments": {"expression": "1+1"}}],
            usage=Usage(input_tokens=1, output_tokens=1),
        )


@pytest.mark.asyncio
async def test_round_cap_ends_with_a_progress_report() -> None:
    app, _ = await _app_with_router(Path())
    try:
        app.router = _EndlessRouter()
        response = await app.ask("Keep calculating", channel="test")
        assert response["status"] == "incomplete"
        for phrase in ("Progress report:", "Done so far:", "Not finished:", "Next options:"):
            assert phrase in response["content"]
        assert "calculator" in response["content"]
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_stream_round_cap_ends_with_a_progress_report() -> None:
    app, _ = await _app_with_router(Path())

    class _EndlessStreamRouter:
        def __init__(self) -> None:
            self.calls = 0
            self.last_stream_provider = "fake"
            self.last_stream_content = ""
            self.last_stream_tool_calls: list[dict[str, Any]] = []
            self.last_stream_usage = Usage()

        async def stream(self, request: ModelRequest):
            self.calls += 1
            self.last_stream_content = "still working"
            self.last_stream_tool_calls = [
                {"id": f"c{self.calls}", "name": "calculator",
                 "arguments": {"expression": "1+1"}}
            ]
            self.last_stream_usage = Usage()
            if False:  # pragma: no cover - keep this an async generator
                yield ""

    try:
        app.router = _EndlessStreamRouter()
        events = [event async for event in app.ask_stream("Keep calculating")]
        finished = next(e for e in events if e["type"] == "message.finished")
        assert finished["status"] == "incomplete"
        assert "Progress report:" in finished["content"]
        assert "Not finished: pending tool calls: calculator" in finished["content"]
        assert "Next options:" in finished["content"]

        rows = await db.fetchall(
            "SELECT metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1"
        )
        metadata = json.loads(rows[0]["metadata"])
        assert metadata["tool_scope"] == "utility"
        assert 0 < metadata["tools_sent"] < len(_specs())
        assert metadata["tool_spec_tokens"] > 0
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_per_task_call_cap_ends_with_a_progress_report() -> None:
    app, _ = await _app_with_router(Path())

    async def capped(*args: Any, **kwargs: Any) -> ToolResult:
        return ToolResult(
            call_id="c1", tool="calculator", success=False,
            error="tool call limit exceeded for chat:test: 1 calls per 60s",
            decision="rate_limited",
            metadata={"quota": {"key": "chat:test", "limit": 1}},
        )

    class _TwoStepRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    content="working", model="fake", provider="fake",
                    tool_calls=[{"id": "c1", "name": "calculator",
                                 "arguments": {"expression": "1+1"}}],
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
            return ModelResponse(
                content="Here is a partial answer.", model="fake", provider="fake",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    try:
        app.executor.execute = capped  # type: ignore[assignment]
        app.router = _TwoStepRouter()
        response = await app.ask("Calculate 2 plus 2", channel="test")
        assert response["status"] == "incomplete"
        for phrase in ("Progress report:", "per-task tool-call cap reached",
                       "Done so far:", "Not finished:", "Next options:"):
            assert phrase in response["content"]
        assert "1 calls per 60s" in response["content"]
        assert "Here is a partial answer." in response["content"]
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- R4


def test_tool_spec_token_estimate_is_comparable_before_and_after() -> None:
    specs = _specs()
    full = [spec.model_dump(mode="json") for spec in specs]
    chat = select_tool_specs(specs, CHAT)
    shell = select_tool_specs(specs, SHELL)
    general = select_tool_specs(specs, GENERAL)
    assert estimate_tool_spec_tokens(chat) == 0
    assert estimate_tool_spec_tokens(shell) < estimate_tool_spec_tokens(full)
    assert estimate_tool_spec_tokens(general) < estimate_tool_spec_tokens(full)
    # The estimate is an explicit chars/4 measure, not a provider measurement.
    payload = json.dumps(full, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert estimate_tool_spec_tokens(full) == -(-len(payload) // 4)


@pytest.mark.asyncio
async def test_usage_records_tool_spec_tokens_per_call() -> None:
    settings = Settings(models=ModelConfig(
        default="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    ))
    router = ModelRouter(settings=settings)
    tools = [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}]
    request = ModelRequest(
        messages=[ChatMessage(role="user", content="Read the file")],
        tools=tools,
        task_id="scope-test",
    )
    await router.complete(request)
    report = await router.usage_report()
    assert len(report) == 1
    assert report[0]["metadata"]["tool_spec_tokens"] == estimate_tool_spec_tokens(tools)
    assert report[0]["metadata"]["tools_sent"] == 1
