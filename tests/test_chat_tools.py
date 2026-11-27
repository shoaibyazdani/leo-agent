"""Chat turns can invoke tools autonomously under the normal permission gate."""

from __future__ import annotations

import pytest

from leo.commands import CommandRegistry, create_registry
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import ModelResponse, StreamChunk, Usage
from leo.core import LeoApp

# The turn budget is settings-driven now (settings.max_chat_tool_rounds).
# These tests pin a small one so an exhaust-the-budget case costs seconds
# instead of minutes; the shipped default is pinned in
# tests/test_hermes_parity_limits.py.
TEST_ROUNDS = 8


class _ToolCallingRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="",
                model="fake-tools",
                provider="fake",
                tool_calls=[{"id": "c1", "name": "calculator", "arguments": {"expression": "21*2"}}],
                usage=Usage(input_tokens=10, output_tokens=2),
            )
        assert any(m.role == "tool" and "42" in m.content for m in request.messages)
        return ModelResponse(
            content="The answer is 42.",
            model="fake-tools",
            provider="fake",
            usage=Usage(input_tokens=20, output_tokens=5),
        )


@pytest.mark.asyncio
async def test_chat_executes_tool_then_answers() -> None:
    settings = Settings()
    settings.max_chat_tool_rounds = TEST_ROUNDS
    settings.models = ModelConfig(default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")})
    app = LeoApp(settings)
    await app.initialize()
    app.router = _ToolCallingRouter()
    response = await app.ask("What is 21 times 2?", channel="test")
    assert response["content"] == "The answer is 42."
    assert response["tool_results"][0]["name"] == "calculator"
    import json

    assert json.loads(response["tool_results"][0]["result"])["output"] == 42
    from leo import db

    tool_rows = await db.fetchall("SELECT tool, success FROM tool_calls WHERE tool='calculator'")
    assert tool_rows and tool_rows[0]["success"] == 1
    await app.shutdown()


async def test_chat_stream_marks_real_tool_success_correctly() -> None:
    """A ToolResult serializes `error: null` on success; that must not be read
    as a failure in the `tool.finished` stream event."""
    settings = Settings()
    settings.max_chat_tool_rounds = TEST_ROUNDS
    settings.models = ModelConfig(default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")})
    app = LeoApp(settings)
    await app.initialize()

    class StreamRouter(_ToolCallingRouter):
        def __init__(self) -> None:
            super().__init__()
            self.last_stream_provider = "fake"
            self.last_stream_content = ""
            self.last_stream_tool_calls: list[dict] = []
            self.last_stream_usage = Usage()

        async def stream(self, request):
            response = await self.complete(request)
            self.last_stream_content = response.content
            self.last_stream_tool_calls = response.tool_calls
            self.last_stream_usage = response.usage
            for piece in (response.content[:4], response.content[4:]):
                if piece:
                    yield piece

    app.router = StreamRouter()
    events = [event async for event in app.ask_stream("What is 21 times 2?")]
    finished = next(event for event in events if event["type"] == "tool.finished")
    assert finished["name"] == "calculator"
    assert finished["success"] is True
    assert finished["is_error"] is False
    await app.shutdown()


# ---------------------------------------------------------------------- F-25
class _EndlessToolsRouter:
    """Always requests another tool; TEST_ROUNDS is exhausted."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content="still working",
            model="fake-tools",
            provider="fake",
            tool_calls=[{"id": f"c{self.calls}", "name": "calculator",
                         "arguments": {"expression": f"{self.calls}+{self.calls}"}}],
            usage=Usage(input_tokens=10, output_tokens=2),
        )


class _EndlessStreamRouter:
    providers: dict = {}

    def __init__(self) -> None:
        self.calls = 0
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        self.calls += 1
        self.last_stream_content = "still working"
        self.last_stream_tool_calls = [
            {"id": f"c{self.calls}", "name": "calculator",
             "arguments": {"expression": f"{self.calls}+{self.calls}"}}
        ]
        self.last_stream_usage = Usage()
        for piece in ("still ", "working"):
            yield piece


async def test_chat_reports_unresolved_tool_turn_after_budget() -> None:
    import json

    from leo import db

    settings = Settings()

    settings.max_chat_tool_rounds = TEST_ROUNDS
    settings.models = ModelConfig(default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")})
    app = LeoApp(settings)
    await app.initialize()
    router = _EndlessToolsRouter()
    app.router = router
    response = await app.ask("Keep calculating", channel="test")
    assert router.calls == TEST_ROUNDS + 1  # N tool rounds + answer attempt
    assert response["tool_calls_pending"] == ["calculator"]
    # Tool-turn narration ("still working") is never presented as an answer.
    assert "Tool budget exhausted" in response["content"]
    rows = await db.fetchall(
        "SELECT metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    )
    assert json.loads(rows[0]["metadata"]).get("tool_calls_pending") == ["calculator"]
    assert json.loads(rows[0]["metadata"]).get("tool_calls") == []
    await app.shutdown()


async def test_chat_stream_reports_unresolved_tool_turn_after_budget() -> None:
    import json

    from leo import db

    settings = Settings()

    settings.max_chat_tool_rounds = TEST_ROUNDS
    settings.models = ModelConfig(default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")})
    app = LeoApp(settings)
    await app.initialize()
    router = _EndlessStreamRouter()
    app.router = router
    events = [event async for event in app.ask_stream("Keep calculating")]
    assert router.calls == TEST_ROUNDS + 1  # N tool rounds + answer attempt
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["tool_calls_pending"] == ["calculator"]
    assert "Tool budget exhausted" in finished["content"]
    assert events[-1]["type"] == "run.finished"
    rows = await db.fetchall(
        "SELECT metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    )
    assert json.loads(rows[0]["metadata"]).get("tool_calls_pending") == ["calculator"]
    assert json.loads(rows[0]["metadata"]).get("tool_calls") == []
    await app.shutdown()


# --- go-live robustness: no blank completed turns (report item 4) ---
class _EmptyFinalRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content="", model="fake", provider="fake",
            tool_calls=[{"id": f"c{self.calls}", "name": "calculator", "arguments": {"expression": "1+1"}}],
            usage=Usage(input_tokens=5, output_tokens=0),
        )


class _EmptyFinalStreamRouter:
    providers: dict = {}

    def __init__(self) -> None:
        self.calls = 0
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        self.calls += 1
        self.last_stream_tool_calls = [
            {"id": f"c{self.calls}", "name": "calculator", "arguments": {"expression": "1+1"}}
        ]
        self.last_stream_usage = Usage()
        if False:  # pragma: no cover - keeps this an async generator
            yield ""
        return


async def test_blank_final_tool_turn_is_incomplete_not_completed() -> None:
    import json

    from leo import db

    settings = Settings()

    settings.max_chat_tool_rounds = TEST_ROUNDS
    settings.models = ModelConfig(default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")})
    app = LeoApp(settings)
    await app.initialize()
    app.router = _EmptyFinalRouter()
    response = await app.ask("Calculate", channel="test")
    assert response["status"] == "incomplete"
    assert response["content"].strip()  # never a blank answer
    assert response["tool_calls_pending"] == ["calculator"]
    rows = await db.fetchall("SELECT content, metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1")
    assert json.loads(rows[0]["metadata"]).get("run_status") == "incomplete"
    await app.shutdown()


async def test_blank_final_stream_turn_is_incomplete_not_completed() -> None:
    import json

    from leo import db

    settings = Settings()

    settings.max_chat_tool_rounds = TEST_ROUNDS
    settings.models = ModelConfig(default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")})
    app = LeoApp(settings)
    await app.initialize()
    app.router = _EmptyFinalStreamRouter()
    events = [event async for event in app.ask_stream("Calculate")]
    done = next(event for event in events if event["type"] == "message.finished")
    assert done["status"] == "incomplete" and done["content"].strip()
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "incomplete"
    rows = await db.fetchall("SELECT metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1")
    assert json.loads(rows[0]["metadata"]).get("run_status") == "incomplete"
    await app.shutdown()


# --- repeated identical calls: execute once, then force a no-tool answer -----
class _EightIdenticalCallsRouter:
    """Eight terminal calls with different ids but one identical operation."""

    def __init__(self, *, answer: bool = True, repeats: bool = False) -> None:
        self.calls = 0
        self.answer = answer
        self.repeats = repeats
        self.final_request = None

    def _calls(self) -> list[dict]:
        return [
            {"id": f"repeat-{self.calls}-{index}", "name": "calculator",
             "arguments": {"expression": "7*8"}}
            for index in range(8)
        ]

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        _assert_provider_pairing(request.messages)
        if self.calls == 1 or self.repeats:
            return ModelResponse(
                content="working", model="fake", provider="fake",
                tool_calls=self._calls(), usage=Usage(input_tokens=10, output_tokens=2),
            )
        self.final_request = request
        return ModelResponse(
            content="The answer is 56.", model="fake", provider="fake",
            usage=Usage(input_tokens=20, output_tokens=5),
        )


class _EightIdenticalStreamRouter(_EightIdenticalCallsRouter):
    def __init__(self, *, repeats: bool = False) -> None:
        super().__init__(repeats=repeats)
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        response = await self.complete(request)
        self.last_stream_content = response.content
        self.last_stream_tool_calls = response.tool_calls
        self.last_stream_usage = response.usage
        yield response.content


async def _count_executions(app, monkeypatch) -> list[dict]:
    """Wrap the real executor so one operation may be executed at most once."""
    import json

    from leo import db

    original = app.executor.execute
    executions: list[dict] = []

    async def counted(call):
        executions.append({"name": call.name, "arguments": dict(call.arguments)})
        result = await original(call)
        # The real result must be reused verbatim by every duplicate call.
        executions[-1]["result"] = json.loads(result.model_dump_json())
        return result

    monkeypatch.setattr(app.executor, "execute", counted)
    return executions


async def test_ask_executes_identical_calls_once_then_requests_no_tools(monkeypatch) -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    router = _EightIdenticalCallsRouter()
    app.router = router
    executions = await _count_executions(app, monkeypatch)

    response = await app.ask("Calculate 7 times 8", channel="test")

    assert len(executions) == 1
    assert executions[0]["arguments"] == {"expression": "7*8"}
    assert router.calls == 2  # one tool round, one reserved synthesis call
    assert router.final_request is not None
    assert "no more tool calls" in router.final_request.messages[-1].content.lower()
    assert "answer now" in router.final_request.messages[-1].content.lower()
    assert response["status"] == "completed" and response["content"] == "The answer is 56."
    # Only the one real execution is reported, though all eight calls are
    # answered in the provider payload from that single cached result.
    assert len(response["tool_results"]) == 1
    assert response["tool_results"][0]["result"] == json.dumps(executions[0]["result"])

    rows = await db.fetchall(
        "SELECT metadata FROM messages WHERE role='tool' ORDER BY id"
    )
    duplicates = [json.loads(row["metadata"]).get("duplicate") for row in rows]
    assert duplicates == [False] + [True] * 7
    await app.shutdown()


async def test_stream_executes_identical_calls_once_then_requests_no_tools(monkeypatch) -> None:
    app = LeoApp(_settings())
    await app.initialize()
    router = _EightIdenticalStreamRouter()
    app.router = router
    executions = await _count_executions(app, monkeypatch)

    events = [event async for event in app.ask_stream("Calculate 7 times 8")]

    assert len(executions) == 1
    assert router.calls == 2
    assert router.final_request is not None
    assert "no more tool calls" in router.final_request.messages[-1].content.lower()
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["status"] == "completed" and finished["content"] == "The answer is 56."
    assert len([event for event in events if event["type"] == "tool.finished"]) == 8
    assert [event["duplicate"] for event in events if event["type"] == "tool.finished"] == (
        [False] + [True] * 7
    )
    await app.shutdown()


async def test_model_still_asking_for_tools_after_synthesis_stays_incomplete(monkeypatch) -> None:
    app = LeoApp(_settings())
    await app.initialize()
    router = _EightIdenticalCallsRouter(repeats=True)
    app.router = router
    executions = await _count_executions(app, monkeypatch)

    response = await app.ask("Calculate 7 times 8", channel="test")

    assert len(executions) == 1
    assert router.calls == 2
    assert response["status"] == "incomplete"
    assert "Tool budget exhausted" in response["content"]
    assert len(response["tool_calls_pending"]) == 8
    await app.shutdown()


def test_system_prompt_points_to_task_store_not_binary_database() -> None:
    prompt = LeoApp.__new__(LeoApp)._conversation_system_prompt()
    assert "task_list" in prompt and "task_status" in prompt
    assert ".leo/data/leo.db" in prompt
    assert "SQLite" in prompt and "do not grep" in prompt.lower()


def test_system_prompt_requires_real_issue_numbers_and_parked_tasks() -> None:
    prompt = LeoApp.__new__(LeoApp)._conversation_system_prompt()
    assert "`number` field" in prompt
    assert "list position" in prompt
    assert "An offer is not permission" in prompt
    assert "Never run tools for a parked task" in prompt


def test_system_prompt_asks_for_a_readable_answer() -> None:
    """The operator reads on a phone: the answer first, the audit never.

    A technical answer is not a good answer. These lines are load-bearing: without
    them a capable model opens with a heading, lists every check it ran and buries
    the result — which reads as noise to the person waiting.
    """
    prompt = LeoApp.__new__(LeoApp)._conversation_system_prompt()
    assert "Lead with the answer in the first line" in prompt
    assert "2-6 short lines" in prompt
    assert "code block only when the content IS code" in prompt
    assert "Do not narrate your process" in prompt
    assert "Match the user's language" in prompt
    assert "one clear recommendation over a menu of options" in prompt


# --- N+1 invariant: a finite tool sequence must reach a real model answer ---
class _SequentialToolsRouter:
    """Requests `rounds` distinct tool calls, then answers with a real string."""

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        self.calls = 0
        self.answer = f"Final answer after {rounds} tool rounds."

    def _next_response(self, request) -> ModelResponse:
        if self.calls <= self.rounds:
            index = self.calls
            return ModelResponse(
                content="",
                model="fake-tools",
                provider="fake",
                tool_calls=[{
                    "id": f"seq-{index}",
                    "name": "calculator",
                    "arguments": {"expression": f"{index}+{index}"},
                }],
                usage=Usage(input_tokens=10, output_tokens=2),
            )
        # The model must see every tool result before it is allowed to answer.
        seen_ids = [
            message.tool_call_id for message in request.messages
            if message.role == "tool" and message.tool_call_id
        ]
        assert seen_ids == [f"seq-{index}" for index in range(1, self.rounds + 1)]
        return ModelResponse(
            content=self.answer, model="fake-tools", provider="fake",
            usage=Usage(input_tokens=20, output_tokens=5),
        )

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        return self._next_response(request)


class _SequentialStreamRouter(_SequentialToolsRouter):
    def __init__(self, rounds: int) -> None:
        super().__init__(rounds)
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        response = await self.complete(request)
        self.last_stream_content = response.content
        self.last_stream_tool_calls = response.tool_calls
        self.last_stream_usage = response.usage
        for piece in (response.content[:8], response.content[8:]):
            if piece:
                yield piece


def _settings() -> Settings:
    settings = Settings()
    settings.max_chat_tool_rounds = TEST_ROUNDS
    settings.models = ModelConfig(default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")})
    settings.max_chat_tool_rounds = TEST_ROUNDS
    return settings


async def _assert_tool_history(executed_ids: list[str], pending_names: list[str]) -> dict:
    """Executed and pending calls must never be confused in durable history."""
    import json

    from leo import db

    rows = await db.fetchall(
        "SELECT role, metadata FROM messages WHERE role IN ('assistant','tool') ORDER BY id"
    )
    # The final row is excluded: when genuinely truncated it reports pending
    # calls by name and carries NO assistant tool_use. Unexecuted calls there
    # would rehydrate without a tool_result and break the next provider call.
    assistant_ids = [
        call["id"]
        for row in rows[:-1] if row["role"] == "assistant"
        for call in (json.loads(row["metadata"] or "{}").get("tool_calls") or [])
        if isinstance(call, dict)
    ]
    tool_ids = [
        json.loads(row["metadata"] or "{}")["tool_call_id"]
        for row in rows if row["role"] == "tool"
    ]
    # Exactly one assistant(tool_calls) row per executed call, each paired
    # with one tool row; the final row never duplicates an executed call.
    assert assistant_ids == executed_ids
    assert tool_ids == executed_ids
    final_meta = json.loads(rows[-1]["metadata"] or "{}")
    assert final_meta.get("tool_calls") == []
    assert final_meta.get("tool_calls_pending") == pending_names
    assert final_meta.get("run_status") == ("incomplete" if pending_names else "completed")
    return final_meta


async def test_chat_four_tool_rounds_then_real_answer() -> None:
    import json

    app = LeoApp(_settings())
    await app.initialize()
    router = _SequentialToolsRouter(4)
    app.router = router
    response = await app.ask("Create the page", channel="test")
    assert router.calls == 5  # four tool rounds + one consuming answer call
    assert response["status"] == "completed"
    assert response["tool_calls_pending"] == []
    assert response["content"] == router.answer
    assert len(response["tool_results"]) == 4
    assert [json.loads(item["result"])["output"] for item in response["tool_results"]] == [2, 4, 6, 8]
    final_meta = await _assert_tool_history([f"seq-{i}" for i in range(1, 5)], [])
    assert final_meta.get("tool_calls_pending") == []
    assert final_meta.get("run_status") == "completed"
    await app.shutdown()


async def test_chat_stream_four_tool_rounds_then_real_answer() -> None:
    import json

    app = LeoApp(_settings())
    await app.initialize()
    router = _SequentialStreamRouter(4)
    app.router = router
    events = [event async for event in app.ask_stream("Create the page")]
    assert router.calls == 5
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["content"] == router.answer
    assert finished["status"] == "completed"
    assert finished["tool_calls_pending"] == []
    assert len(finished["tool_results"]) == 4
    assert [json.loads(item["result"])["output"] for item in finished["tool_results"]] == [2, 4, 6, 8]
    assert [event["tool_call_id"] for event in events if event["type"] == "tool.finished"] == [
        f"seq-{i}" for i in range(1, 5)
    ]
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "completed"
    final_meta = await _assert_tool_history([f"seq-{i}" for i in range(1, 5)], [])
    assert final_meta.get("tool_calls_pending") == []
    assert final_meta.get("run_status") == "completed"
    await app.shutdown()


@pytest.mark.parametrize("tool_rounds,completes", [
    (TEST_ROUNDS, True),
    (TEST_ROUNDS + 1, False),
])
async def test_chat_boundary_between_answer_and_exhaustion(tool_rounds: int, completes: bool) -> None:
    app = LeoApp(_settings())
    await app.initialize()
    router = _SequentialToolsRouter(tool_rounds)
    app.router = router
    response = await app.ask("Boundary", channel="test")
    assert router.calls == TEST_ROUNDS + 1
    assert response["status"] == ("completed" if completes else "incomplete")
    assert response["tool_calls_pending"] == ([] if completes else ["calculator"])
    assert len(response["tool_results"]) == TEST_ROUNDS  # never a 9th execution
    pending_names = [] if completes else ["calculator"]
    final_meta = await _assert_tool_history(
        [f"seq-{i}" for i in range(1, TEST_ROUNDS + 1)], pending_names
    )
    assert final_meta.get("tool_calls_pending") == ([] if completes else ["calculator"])
    await app.shutdown()


@pytest.mark.parametrize("tool_rounds,completes", [
    (TEST_ROUNDS, True),
    (TEST_ROUNDS + 1, False),
])
async def test_chat_stream_boundary_between_answer_and_exhaustion(tool_rounds: int, completes: bool) -> None:
    app = LeoApp(_settings())
    await app.initialize()
    router = _SequentialStreamRouter(tool_rounds)
    app.router = router
    events = [event async for event in app.ask_stream("Boundary")]
    assert router.calls == TEST_ROUNDS + 1
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["status"] == ("completed" if completes else "incomplete")
    assert finished["tool_calls_pending"] == ([] if completes else ["calculator"])
    assert len(finished["tool_results"]) == TEST_ROUNDS
    assert events[-1]["type"] == "run.finished"
    assert events[-1]["status"] == finished["status"]
    pending_names = [] if completes else ["calculator"]
    final_meta = await _assert_tool_history(
        [f"seq-{i}" for i in range(1, TEST_ROUNDS + 1)], pending_names
    )
    assert final_meta.get("tool_calls_pending") == ([] if completes else ["calculator"])
    await app.shutdown()


# --- second-turn provider payload: no unanswered assistant tool_use --------
def _assert_provider_pairing(messages) -> None:
    """Strict pairing validator for the messages sent to a provider."""
    missing: list[str] = []
    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_calls:
            assert not missing, f"nested tool group while {missing} unanswered"
            missing = [
                str(call.get("id") or call.get("tool_call_id") or "")
                for call in message.tool_calls
            ]
            assert all(missing), f"tool call without an id at message {index}"
        elif message.role == "tool":
            call_id = str(message.tool_call_id)
            assert call_id in missing, f"orphan tool result {call_id} at message {index}"
            missing.remove(call_id)
        elif missing:
            raise AssertionError(f"unanswered tool calls {missing} at message {index}")
    assert not missing, f"unanswered tool calls at end of payload: {missing}"


class _EndlessThenValidateRouter:
    """Exhaust the budget, then validate the next turn's rehydrated history."""

    def __init__(self) -> None:
        self.calls = 0

    def _respond(self, request) -> ModelResponse:
        self.calls += 1
        _assert_provider_pairing(request.messages)
        if self.calls == 1:
            return self._tool_response(0)
        if self.calls <= TEST_ROUNDS + 1:
            return self._tool_response(self.calls)
        return ModelResponse(
            content="Validated follow-up answer.", model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=5),
        )

    @staticmethod
    def _tool_response(index: int) -> ModelResponse:
        return ModelResponse(
            content="still working", model="fake", provider="fake",
            tool_calls=[{
                "id": f"endless-{index}", "name": "calculator",
                # Distinct operations: this fixture exercises the round budget,
                # not the same-call deduplication path.
                "arguments": {"expression": f"{index}+{index}"},
            }],
            usage=Usage(input_tokens=5, output_tokens=2),
        )

    async def complete(self, request) -> ModelResponse:
        return self._respond(request)


class _EndlessThenValidateStreamRouter(_EndlessThenValidateRouter):
    def __init__(self) -> None:
        super().__init__()
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        response = self._respond(request)
        self.last_stream_content = response.content
        self.last_stream_tool_calls = response.tool_calls
        self.last_stream_usage = response.usage
        yield response.content


async def test_second_turn_after_exhaustion_sends_valid_provider_payload() -> None:
    app = LeoApp(_settings())
    await app.initialize()
    router = _EndlessThenValidateRouter()
    app.router = router
    conversation_id = "conv-exhaust-ask"
    first = await app.ask("Keep calculating", channel="test", conversation_id=conversation_id)
    assert first["status"] == "incomplete"
    second = await app.ask("Continue now", channel="test", conversation_id=conversation_id)
    assert second["content"] == "Validated follow-up answer."
    assert router.calls == TEST_ROUNDS + 2
    await app.shutdown()


async def test_second_stream_turn_after_exhaustion_sends_valid_provider_payload() -> None:
    app = LeoApp(_settings())
    await app.initialize()
    router = _EndlessThenValidateStreamRouter()
    app.router = router
    conversation_id = "conv-exhaust-stream"
    first = [event async for event in app.ask_stream("Keep calculating", conversation_id=conversation_id)]
    assert next(e for e in first if e["type"] == "message.finished")["status"] == "incomplete"
    second = [event async for event in app.ask_stream("Continue now", conversation_id=conversation_id)]
    finished = next(e for e in second if e["type"] == "message.finished")
    assert finished["content"] == "Validated follow-up answer."
    assert router.calls == TEST_ROUNDS + 2
    await app.shutdown()


# --- atomic history selection with tool-heavy turns ------------------------
class _MultiToolRouter:
    """Three tool calls per turn in a two-call (tool round + answer) turn."""

    def __init__(self, tools_per_round: int = 3) -> None:
        self.calls = 0
        self.tools_per_round = tools_per_round

    def _respond(self, request) -> ModelResponse:
        self.calls += 1
        turn = (self.calls - 1) // 2 + 1
        if self.calls % 2:
            return ModelResponse(
                content="", model="fake", provider="fake",
                tool_calls=[{
                    "id": f"multi-{turn}-{index}", "name": "calculator",
                    "arguments": {"expression": f"{turn}+{index}"},
                } for index in range(self.tools_per_round)],
                usage=Usage(input_tokens=10, output_tokens=2),
            )
        return ModelResponse(
            content=f"Turn {turn} done.", model="fake", provider="fake",
            usage=Usage(input_tokens=10, output_tokens=4),
        )

    async def complete(self, request) -> ModelResponse:
        return self._respond(request)


class _MultiToolStreamRouter(_MultiToolRouter):
    def __init__(self, tools_per_round: int = 3) -> None:
        super().__init__(tools_per_round)
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        response = self._respond(request)
        self.last_stream_content = response.content
        self.last_stream_tool_calls = response.tool_calls
        self.last_stream_usage = response.usage
        yield response.content


class _HistoryValidatorRouter:
    def __init__(self, expected_tool_ids: set[str]) -> None:
        self.expected_tool_ids = expected_tool_ids
        self.seen_tool_ids: list[str] = []
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        _assert_provider_pairing(request.messages)
        self.seen_tool_ids = [
            str(message.tool_call_id) for message in request.messages
            if message.role == "tool" and message.tool_call_id
        ]
        assert set(self.seen_tool_ids) == self.expected_tool_ids
        return ModelResponse(
            content="validated", model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=2),
        )


class _HistoryValidatorStreamRouter(_HistoryValidatorRouter):
    def __init__(self, expected_tool_ids: set[str]) -> None:
        super().__init__(expected_tool_ids)
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        response = await self.complete(request)
        self.last_stream_content = response.content
        self.last_stream_tool_calls = response.tool_calls
        self.last_stream_usage = response.usage
        yield response.content


_EXPECTED_HEAVY_IDS = {
    f"multi-{turn}-{index}" for turn in (2, 3, 4) for index in range(3)
}


async def test_history_limit_expands_backwards_over_tool_heavy_turns() -> None:
    app = LeoApp(_settings())
    await app.initialize()
    # 24 rows after four turns; a 14-row window would start inside turn 2's
    # tool group and produce an orphan tool result without atomic selection.
    app.CONVERSATION_HISTORY_LIMIT = 14
    app.router = _MultiToolRouter()
    conversation_id = "conv-heavy-ask"
    for turn in range(4):
        result = await app.ask(f"Heavy turn {turn}", channel="test", conversation_id=conversation_id)
        assert result["status"] == "completed" and len(result["tool_results"]) == 3
    validator = _HistoryValidatorRouter(_EXPECTED_HEAVY_IDS)
    app.router = validator
    followup = await app.ask("Validate history", channel="test", conversation_id=conversation_id)
    assert followup["content"] == "validated"
    assert validator.calls == 1 and len(validator.seen_tool_ids) == 9
    await app.shutdown()


async def test_stream_history_limit_expands_backwards_over_tool_heavy_turns() -> None:
    app = LeoApp(_settings())
    await app.initialize()
    app.CONVERSATION_HISTORY_LIMIT = 14
    app.router = _MultiToolStreamRouter()
    conversation_id = "conv-heavy-stream"
    for turn in range(4):
        events = [event async for event in app.ask_stream(f"Heavy turn {turn}", conversation_id=conversation_id)]
        finished = next(event for event in events if event["type"] == "message.finished")
        assert finished["status"] == "completed" and len(finished["tool_results"]) == 3
    validator = _HistoryValidatorStreamRouter(_EXPECTED_HEAVY_IDS)
    app.router = validator
    events = [event async for event in app.ask_stream("Validate history", conversation_id=conversation_id)]
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["content"] == "validated"
    assert validator.calls == 1 and len(validator.seen_tool_ids) == 9
    await app.shutdown()


# --- blank final answer without tools --------------------------------------
class _BlankAnswerRouter:
    async def complete(self, request) -> ModelResponse:
        return ModelResponse(
            content="", model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=0),
        )


class _BlankAnswerStreamRouter:
    def __init__(self) -> None:
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        if False:  # pragma: no cover - keeps this an async generator
            yield ""
        return


async def test_blank_answer_without_tools_is_not_completed() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _BlankAnswerRouter()
    response = await app.ask("Say nothing", channel="test")
    assert response["status"] == "incomplete"
    assert "empty answer" in response["content"].lower()
    rows = await db.fetchall("SELECT content, metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1")
    assert "empty answer" in rows[0]["content"].lower()
    assert json.loads(rows[0]["metadata"])["run_status"] == "incomplete"
    await app.shutdown()


async def test_blank_stream_answer_without_tools_is_not_completed() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _BlankAnswerStreamRouter()
    events = [event async for event in app.ask_stream("Say nothing")]
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["status"] == "incomplete"
    assert "empty answer" in finished["content"].lower()
    assert events[-1]["status"] == "incomplete"
    rows = await db.fetchall("SELECT metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1")
    assert json.loads(rows[0]["metadata"])["run_status"] == "incomplete"
    await app.shutdown()


# --- setup failures always close the run -----------------------------------
async def test_stream_setup_failure_emits_terminal_events_and_clears_run() -> None:
    app = LeoApp(_settings())
    await app.initialize()

    def boom(*args, **kwargs):
        raise RuntimeError("setup exploded")

    app.context.optimize = boom
    events = [event async for event in app.ask_stream("This will fail")]
    assert [event["type"] for event in events] == ["run.started", "error", "run.finished"]
    assert events[1]["code"] == "RuntimeError"
    assert events[-1]["status"] == "failed"
    assert events[-1]["run_id"] not in app._runs
    await app.shutdown()


async def test_stream_history_failure_emits_terminal_events_and_clears_run() -> None:
    app = LeoApp(_settings())
    await app.initialize()

    async def boom(*args, **kwargs):
        raise RuntimeError("history exploded")

    app._conversation_history = boom
    events = [event async for event in app.ask_stream("This will fail")]
    assert [event["type"] for event in events] == ["error", "run.finished"]
    assert events[-1]["status"] == "failed"
    assert events[-1]["run_id"] not in app._runs
    await app.shutdown()


# --- cancellation mid-tool keeps assistant/tool groups pairable -------------
class _TwoToolStreamRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.last_stream_provider = "fake"
        self.last_stream_content = "working"
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        self.calls += 1
        self.last_stream_tool_calls = [{
            "id": f"cancel-{index}", "name": "calculator",
            "arguments": {"expression": f"{index}+{index}"},
        } for index in (1, 2)]
        self.last_stream_content = "working"
        yield "working"


async def test_cancel_mid_tool_persists_explicit_not_executed_result() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _TwoToolStreamRouter()
    conversation_id = "conv-cancel-mid-tool"
    stream = app.ask_stream("Do two calculations", conversation_id=conversation_id)
    run_id = ""
    events: list[dict] = []
    async for event in stream:
        events.append(event)
        if event["type"] == "run.started":
            run_id = event["run_id"]
        if event["type"] == "tool.finished" and event["tool_call_id"] == "cancel-1":
            assert await app.cancel_run(run_id) == "cancelled"
            break
    events.extend([event async for event in stream])
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "cancelled"
    assert run_id not in app._runs

    rows = await db.fetchall(
        "SELECT role, content, metadata FROM messages WHERE conversation_id=? "
        "AND role IN ('assistant','tool') ORDER BY id", (conversation_id,)
    )
    assistant_calls = [
        str(call["id"]) for row in rows if row["role"] == "assistant"
        for call in (json.loads(row["metadata"] or "{}").get("tool_calls") or [])
    ]
    tool_rows = [row for row in rows if row["role"] == "tool"]
    assert assistant_calls == ["cancel-1", "cancel-2"]
    assert [json.loads(row["metadata"])["tool_call_id"] for row in tool_rows] == assistant_calls
    skipped = [json.loads(row["metadata"]) for row in tool_rows
               if json.loads(row["metadata"]).get("not_executed")]
    assert [row["tool_call_id"] for row in skipped] == ["cancel-2"]
    assert "not_executed" in next(
        row["content"] for row in tool_rows
        if json.loads(row["metadata"])["tool_call_id"] == "cancel-2"
    )

    # The following turn must still receive a valid assistant/tool payload.
    class Validator:
        async def complete(self, request):
            _assert_provider_pairing(request.messages)
            return ModelResponse(
                content="after cancellation", model="fake", provider="fake",
                usage=Usage(input_tokens=5, output_tokens=2),
            )

    app.router = Validator()
    followup = await app.ask("Continue", channel="test", conversation_id=conversation_id)
    assert followup["content"] == "after cancellation"
    await app.shutdown()


# --- F1: cancellation during the FINAL model stream ------------------------
class _FinalStreamCancelRouter:
    """Cancel the live run from inside the final (tool-free) stream call."""

    def __init__(self, app: LeoApp) -> None:
        self.app = app
        self.calls = 0
        self.last_stream_provider = "fake"
        self.last_stream_content = ""
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        self.calls += 1
        self.last_stream_provider = "fake"
        self.last_stream_usage = Usage()
        if self.calls == 1:
            self.last_stream_tool_calls = [{
                "id": "final-cancel-1", "name": "calculator",
                "arguments": {"expression": "1+1"},
            }]
            self.last_stream_content = "working"
            yield "working"
            return
        # The final model call streams a complete-looking answer, but the run
        # is cancelled while it is being produced.
        run_id = next(iter(self.app._runs))
        assert await self.app.cancel_run(run_id) == "cancelled"
        self.last_stream_tool_calls = []
        self.last_stream_content = "A complete-looking final answer."
        yield "A complete-looking "
        yield "final answer."


async def test_cancel_during_final_stream_does_not_emit_or_persist_completed() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    router = _FinalStreamCancelRouter(app)
    app.router = router
    conversation_id = "conv-final-cancel"
    events = [event async for event in app.ask_stream(
        "Finish this", conversation_id=conversation_id
    )]

    assert router.calls == 2
    assert not any(
        event["type"] == "message.finished" and event.get("status") == "completed"
        for event in events
    )
    assert not any(event["type"] == "message.finished" for event in events)
    assert any(event["type"] == "status" and event.get("status") == "cancelled" for event in events)
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "cancelled"

    rows = await db.fetchall(
        "SELECT role, metadata FROM messages WHERE conversation_id=? AND role='assistant' ORDER BY id",
        (conversation_id,),
    )
    # Only the earlier assistant(tool_calls) row exists: no final completed
    # answer was persisted after the cancellation.
    assert len(rows) == 1
    assert all(json.loads(row["metadata"]).get("run_status") != "completed" for row in rows)
    assert not any("complete-looking final answer" in str(row) for row in rows)
    await app.shutdown()


class _FinalAnswerStreamRouter:
    """A no-tool turn whose answer is already complete when it is streamed."""

    def __init__(self) -> None:
        self.last_stream_provider = "fake"
        self.last_stream_content = "A final answer."
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        self.last_stream_content = "A final answer."
        self.last_stream_tool_calls = []
        self.last_stream_usage = Usage()
        yield self.last_stream_content


async def test_cancel_before_final_persist_suppresses_completed_answer(monkeypatch) -> None:
    """Guard (c) is the last checkpoint before the terminal row is written."""
    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _FinalAnswerStreamRouter()
    original_execute = db.execute

    async def cancel_on_final_insert(sql, params=()):
        if (
            isinstance(sql, str)
            and "INSERT INTO messages" in sql
            and len(params) > 1
            and params[1] == "assistant"
            and app._runs
        ):
            await app.cancel_run(next(iter(app._runs)))
        return await original_execute(sql, params)

    monkeypatch.setattr(db, "execute", cancel_on_final_insert)
    events = [event async for event in app.ask_stream("Answer now")]
    assert not any(
        event["type"] == "message.finished" and event.get("status") == "completed"
        for event in events
    )
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "cancelled"
    rows = await db.fetchall("SELECT role, content FROM messages WHERE role='assistant'")
    assert rows == []  # no completed answer was made durable
    await app.shutdown()


# --- F2: blocking ask() normalizes legacy tool-call ids before persisting ---
class _LegacyIdAskRouter:
    """First turn asks for a call with no id, or only a ``tool_call_id``."""

    def __init__(self, shape: str) -> None:
        self.shape = shape
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            call: dict[str, object] = {
                "name": "calculator", "arguments": {"expression": "2+2"},
            }
            if self.shape == "tool_call_id":
                call["tool_call_id"] = "legacy-tcid-1"
            return ModelResponse(
                content="", model="fake", provider="fake", tool_calls=[call],
                usage=Usage(input_tokens=5, output_tokens=2),
            )
        return ModelResponse(
            content=f"Answer for {self.shape}.",
            model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=3),
        )


@pytest.mark.parametrize("shape", ["idless", "tool_call_id"])
async def test_blocking_ask_second_turn_pairs_legacy_tool_calls(shape: str) -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _LegacyIdAskRouter(shape)
    conversation_id = f"conv-legacy-{shape}"
    first = await app.ask("Calculate", channel="test", conversation_id=conversation_id)
    assert first["status"] == "completed"

    rows = await db.fetchall(
        "SELECT role, metadata FROM messages WHERE conversation_id=? ORDER BY id",
        (conversation_id,),
    )
    assistant_ids = [
        str(call["id"])
        for row in rows if row["role"] == "assistant"
        for call in (json.loads(row["metadata"] or "{}").get("tool_calls") or [])
        if isinstance(call, dict)
    ]
    tool_ids = [
        str(json.loads(row["metadata"] or "{}")["tool_call_id"])
        for row in rows if row["role"] == "tool"
    ]
    assert len(assistant_ids) == 1 and assistant_ids == tool_ids
    assert assistant_ids[0]  # stable id, not empty
    if shape == "tool_call_id":
        assert assistant_ids[0] == "legacy-tcid-1"

    class Validator:
        async def complete(self, request):
            _assert_provider_pairing(request.messages)
            seen = [
                str(message.tool_call_id) for message in request.messages
                if message.role == "tool"
            ]
            assert seen == assistant_ids
            return ModelResponse(
                content="validated", model="fake", provider="fake",
                usage=Usage(input_tokens=5, output_tokens=2),
            )

    app.router = Validator()
    second = await app.ask("Continue", channel="test", conversation_id=conversation_id)
    assert second["content"] == "validated"
    await app.shutdown()


# --- F3: identical tool calls repeated across two turns ----------------------
class _RepeatingToolRouter:
    """Every odd model call makes the same tool call; even calls answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        if self.calls % 2:
            return ModelResponse(
                content="", model="fake", provider="fake",
                tool_calls=[{
                    "id": "repeat-call", "name": "calculator",
                    "arguments": {"expression": "1+1"},
                }],
                usage=Usage(input_tokens=5, output_tokens=2),
            )
        return ModelResponse(
            content=f"Turn {self.calls // 2} answer.",
            model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=3),
        )


async def test_identical_tool_calls_across_turns_survive_dedup() -> None:
    app = LeoApp(_settings())
    await app.initialize()
    app.router = _RepeatingToolRouter()
    conversation_id = "conv-repeat-call"
    for turn in range(2):
        result = await app.ask(
            f"Repeat turn {turn}", channel="test", conversation_id=conversation_id
        )
        assert result["status"] == "completed" and len(result["tool_results"]) == 1

    class Validator:
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def complete(self, request):
            _assert_provider_pairing(request.messages)
            self.seen = [
                str(message.tool_call_id) for message in request.messages
                if message.role == "tool" and message.tool_call_id == "repeat-call"
            ]
            return ModelResponse(
                content="validated", model="fake", provider="fake",
                usage=Usage(input_tokens=5, output_tokens=2),
            )

    validator = Validator()
    app.router = validator
    followup = await app.ask("Validate", channel="test", conversation_id=conversation_id)
    assert followup["content"] == "validated"
    assert validator.seen == ["repeat-call", "repeat-call"]
    await app.shutdown()


# --- F4: a single tool group larger than 4x the history limit ---------------
async def test_history_fetch_walks_back_over_group_larger_than_four_times_limit() -> None:
    app = LeoApp(_settings())
    await app.initialize()
    app.CONVERSATION_HISTORY_LIMIT = 3
    tools_per_round = 13  # > 4 * CONVERSATION_HISTORY_LIMIT
    app.router = _MultiToolRouter(tools_per_round)
    conversation_id = "conv-large-group"
    first = await app.ask("Large group", channel="test", conversation_id=conversation_id)
    assert first["status"] == "completed" and len(first["tool_results"]) == tools_per_round

    expected = {f"multi-1-{index}" for index in range(tools_per_round)}
    validator = _HistoryValidatorRouter(expected)
    app.router = validator
    followup = await app.ask("Validate history", channel="test", conversation_id=conversation_id)
    assert followup["content"] == "validated"
    assert len(validator.seen_tool_ids) == tools_per_round
    assert set(validator.seen_tool_ids) == expected
    await app.shutdown()


async def test_unmatched_tool_group_is_demoted_not_dropped() -> None:
    """When the opener is beyond the fetch cap, results survive as data text."""
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.CONVERSATION_HISTORY_LIMIT = 1  # hard cap = 1 + 8 * 1 = 9 rows
    conversation_id = "conv-demote-group"
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at) VALUES(?,?,?,?,?)",
        (conversation_id, "test", None, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    )
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
        (conversation_id, "user", "start", "2026-01-01T00:00:00Z"),
    )
    calls = [{"id": f"big-{index}", "name": "calculator", "arguments": {}}
             for index in range(12)]
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
        (conversation_id, "assistant", "working",
         json.dumps({"tool_calls": calls}), "2026-01-01T00:00:00Z"),
    )
    for index in range(12):
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
            (conversation_id, "tool", json.dumps({"marker": f"marker-{index}"}),
             json.dumps({"name": "calculator", "tool_call_id": f"big-{index}"}),
             "2026-01-01T00:00:00Z"),
        )

    history = await app._conversation_history(conversation_id)
    assert not any(message.role == "tool" for message in history)
    assert any("marker-11" in message.content for message in history)
    assert any("data only" in message.content for message in history)
    await app.shutdown()


# --- blocking ask() redacts model output before returning or persisting ------
# Built at runtime: the secret scanner cannot tell a fake key from a real one.
_SECRET_ROUTER_KEY = "sk-" + "abcdefghijklmnop"


class _SecretAnswerRouter:
    async def complete(self, request) -> ModelResponse:
        return ModelResponse(
            content=f"The key is api_key = {_SECRET_ROUTER_KEY}, okay?",
            model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=5),
        )


async def test_blocking_ask_redacts_model_output_everywhere() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _SecretAnswerRouter()
    response = await app.ask("Reveal", channel="test")
    assert _SECRET_ROUTER_KEY not in response["content"]
    assert "[REDACTED]" in response["content"]
    rows = await db.fetchall("SELECT content, metadata FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1")
    assert _SECRET_ROUTER_KEY not in rows[0]["content"]
    assert _SECRET_ROUTER_KEY not in json.dumps(json.loads(rows[0]["metadata"]))
    await app.shutdown()


# --- terminal status is closed before message.finished reaches the client ----
async def test_cancel_at_message_finished_is_rejected_as_terminal() -> None:
    """A client cancellation arriving at message.finished cannot contradict it.

    The run is ended before that event is yielded, so cancel_run rejects it as
    no longer live; run.finished and the durable row remain truthful.
    """
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _FinalAnswerStreamRouter()
    events: list[dict] = []
    cancel_error: Exception | None = None
    async for event in app.ask_stream("Answer now"):
        events.append(event)
        if event["type"] == "message.finished":
            assert event["status"] == "completed"
            try:
                await app.cancel_run(event["run_id"])
            except (LookupError, ValueError) as exc:
                cancel_error = exc

    assert isinstance(cancel_error, (LookupError, ValueError))
    assert [event["type"] for event in events[-2:]] == ["message.finished", "run.finished"]
    assert events[-1]["status"] == "completed"
    assert events[0]["run_id"] not in app._runs

    rows = await db.fetchall(
        "SELECT content, metadata FROM messages WHERE role='assistant' ORDER BY id"
    )
    assert len(rows) == 1
    assert json.loads(rows[0]["metadata"]).get("run_status") == "completed"
    assert rows[0]["content"] == "A final answer."
    await app.shutdown()


# --- two tool calls in one round reusing an id must both get results ---------
class _DuplicateIdAskRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        _assert_provider_pairing(request.messages)
        if self.calls == 1:
            return ModelResponse(
                content="", model="fake", provider="fake",
                tool_calls=[
                    {"id": "dup-call", "name": "calculator", "arguments": {"expression": "1+1"}},
                    {"id": "dup-call", "name": "calculator", "arguments": {"expression": "2+2"}},
                ],
                usage=Usage(input_tokens=10, output_tokens=2),
            )
        return ModelResponse(
            content="Both calculations done.", model="fake", provider="fake",
            usage=Usage(input_tokens=20, output_tokens=5),
        )


class _DuplicateIdStreamRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.last_stream_provider = "fake"
        self.last_stream_content = "working"
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_usage = Usage()

    async def stream(self, request):
        self.calls += 1
        _assert_provider_pairing(request.messages)
        if self.calls == 1:
            self.last_stream_tool_calls = [
                {"id": "dup-call", "name": "calculator", "arguments": {"expression": "1+1"}},
                {"id": "dup-call", "name": "calculator", "arguments": {"expression": "2+2"}},
            ]
            self.last_stream_content = "working"
        else:
            self.last_stream_tool_calls = []
            self.last_stream_content = "Both calculations done."
        yield self.last_stream_content


class _PairingValidatorRouter:
    """Final-turn validator: injected per test after the tool round."""

    async def complete(self, request):
        _assert_provider_pairing(request.messages)
        return ModelResponse(
            content="validated", model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=2),
        )


async def test_blocking_ask_rewrites_duplicate_tool_call_ids() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _DuplicateIdAskRouter()
    conversation_id = "conv-dup-ids-ask"
    response = await app.ask("Two calculations", channel="test", conversation_id=conversation_id)
    assert response["status"] == "completed"
    assert len(response["tool_results"]) == 2
    assert app.router.calls == 2

    rows = await db.fetchall(
        "SELECT role, metadata FROM messages WHERE conversation_id=? AND role IN ('assistant','tool') ORDER BY id",
        (conversation_id,),
    )
    assistant_ids = [
        str(call["id"]) for row in rows if row["role"] == "assistant"
        for call in (json.loads(row["metadata"] or "{}").get("tool_calls") or [])
    ]
    tool_ids = [
        str(json.loads(row["metadata"] or "{}")["tool_call_id"])
        for row in rows if row["role"] == "tool"
    ]
    assert len(assistant_ids) == len(tool_ids) == 2
    assert len(set(assistant_ids)) == 2  # duplicate model ids were rewritten
    assert tool_ids == assistant_ids
    await _assert_tool_history(assistant_ids, [])
    await app.shutdown()


async def test_stream_rewrites_duplicate_tool_call_ids() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    router = _DuplicateIdStreamRouter()
    app.router = router
    conversation_id = "conv-dup-ids-stream"
    events = [event async for event in app.ask_stream(
        "Two calculations", conversation_id=conversation_id
    )]
    assert router.calls == 2
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["status"] == "completed"
    assert len(finished["tool_results"]) == 2
    assert len({event["tool_call_id"] for event in events if event["type"] == "tool.finished"}) == 2
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "completed"

    rows = await db.fetchall(
        "SELECT role, metadata FROM messages WHERE conversation_id=? AND role IN ('assistant','tool') ORDER BY id",
        (conversation_id,),
    )
    assistant_ids = [
        str(call["id"]) for row in rows if row["role"] == "assistant"
        for call in (json.loads(row["metadata"] or "{}").get("tool_calls") or [])
    ]
    tool_ids = [
        str(json.loads(row["metadata"] or "{}")["tool_call_id"])
        for row in rows if row["role"] == "tool"
    ]
    assert len(assistant_ids) == len(tool_ids) == 2
    assert len(set(assistant_ids)) == 2
    assert tool_ids == assistant_ids

    app.router = _PairingValidatorRouter()
    followup = await app.ask("Validate", channel="test", conversation_id=conversation_id)
    assert followup["content"] == "validated"
    await app.shutdown()


async def test_cancel_mid_duplicate_group_records_distinct_not_executed_id() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    app.router = _DuplicateIdStreamRouter()
    conversation_id = "conv-cancel-dup-group"
    stream = app.ask_stream("Two calculations", conversation_id=conversation_id)
    run_id = ""
    events: list[dict] = []
    async for event in stream:
        events.append(event)
        if event["type"] == "run.started":
            run_id = event["run_id"]
        if event["type"] == "tool.finished":
            assert await app.cancel_run(run_id) == "cancelled"
            break
    events.extend([event async for event in stream])
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "cancelled"

    rows = await db.fetchall(
        "SELECT role, content, metadata FROM messages WHERE conversation_id=? "
        "AND role IN ('assistant','tool') ORDER BY id", (conversation_id,),
    )
    assistant_ids = [
        str(call["id"]) for row in rows if row["role"] == "assistant"
        for call in (json.loads(row["metadata"] or "{}").get("tool_calls") or [])
    ]
    tool_rows = [row for row in rows if row["role"] == "tool"]
    assert len(assistant_ids) == 2 and len(set(assistant_ids)) == 2
    assert [str(json.loads(row["metadata"])["tool_call_id"]) for row in tool_rows] == assistant_ids
    skipped = [row for row in tool_rows if json.loads(row["metadata"]).get("not_executed")]
    assert len(skipped) == 1
    assert json.loads(skipped[0]["metadata"])["tool_call_id"] in assistant_ids
    assert "not_executed" in skipped[0]["content"]

    app.router = _PairingValidatorRouter()
    followup = await app.ask("Validate", channel="test", conversation_id=conversation_id)
    assert followup["content"] == "validated"
    await app.shutdown()


# --- 16 turns reusing one id: eviction must never orphan a tool_use ----------
class _SixteenTurnRepeatingRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.validated = 0
        self.turn_calls = 0

    async def complete(self, request):
        self.calls += 1
        _assert_provider_pairing(request.messages)
        self.validated += 1
        if any(
            message.role == "system" and message.content.startswith("Summarize the conversation")
            for message in request.messages
        ):
            return ModelResponse(
                content="Older turns summarized.", model="fake", provider="fake",
                usage=Usage(input_tokens=5, output_tokens=3),
            )
        self.turn_calls += 1
        if self.turn_calls % 2:
            return ModelResponse(
                content="", model="fake", provider="fake",
                tool_calls=[{
                    "id": "turn-call", "name": "calculator",
                    "arguments": {"expression": "6*7"},
                }],
                usage=Usage(input_tokens=5, output_tokens=2),
            )
        return ModelResponse(
            content=f"Answer {self.turn_calls // 2}.", model="fake", provider="fake",
            usage=Usage(input_tokens=5, output_tokens=3),
        )


async def test_sixteen_turns_same_tool_id_leave_no_unanswered_tool_use() -> None:
    settings = _settings()
    settings.models.providers["mock"].context_window = 1024
    app = LeoApp(settings)
    await app.initialize()
    router = _SixteenTurnRepeatingRouter()
    app.router = router
    conversation_id = "conv-sixteen-same-id"
    for turn in range(16):
        response = await app.ask(
            f"Turn {turn} question", channel="test", conversation_id=conversation_id
        )
        assert response["status"] == "completed", f"turn {turn}: {response}"
        assert len(response["tool_results"]) == 1

    assert router.turn_calls == 32  # two model calls per turn
    assert router.calls == router.validated
    app.router = _PairingValidatorRouter()
    followup = await app.ask("Validate", channel="test", conversation_id=conversation_id)
    assert followup["content"] == "validated"
    await app.shutdown()


# ------------------------------------------------------------------ thinking stream
class _ReasoningStreamRouter:
    """A router whose provider emits a real reasoning delta plus an answer."""

    def __init__(self, *, reasoning: str = "private thought", answer: str = "The answer.") -> None:
        self.calls = 0
        self.reasoning = reasoning
        self.answer = answer
        self.last_stream_provider = "fake"
        self.last_stream_content = answer
        self.last_stream_tool_calls: list[dict] = []
        self.last_stream_reasoning = reasoning
        self.last_stream_usage = Usage()

    async def stream_events(self, request):
        self.calls += 1
        if self.reasoning:
            yield StreamChunk(kind="reasoning", text=self.reasoning)
        yield StreamChunk(kind="text", text=self.answer)


async def test_stream_emits_reasoning_without_mixing_it_into_the_answer() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    router = _ReasoningStreamRouter()
    app.router = router
    conversation_id = "conv-reasoning-test"
    # Reasoning display is opt-in; the separation test turns it on explicitly.
    await app.reasoning_state.set_display(conversation_id, show=True)
    events = [event async for event in app.ask_stream("Think then answer", conversation_id=conversation_id)]

    thoughts = [event["delta"] for event in events if event["type"] == "thinking.delta"]
    assert "".join(thoughts) == "private thought"
    assert all(event["type"] != "message.delta" or "private thought" not in event["delta"] for event in events)
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["content"] == "The answer."
    assert finished["reasoning_seen"] is True
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "completed"
    assert any(
        event["type"] == "thinking.state" and event["seen"] is True and event["delta"] == ""
        for event in events
    )

    rows = await db.fetchall(
        "SELECT content, metadata FROM messages WHERE conversation_id=? AND role='assistant'",
        (conversation_id,),
    )
    assert len(rows) == 1
    assert "private thought" not in rows[0]["content"]
    assert "private thought" not in rows[0]["metadata"]
    assert json.loads(rows[0]["metadata"]).get("run_status") == "completed"
    await app.shutdown()


async def test_stream_without_provider_reasoning_reports_empty_state_truthfully() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    router = _ReasoningStreamRouter(reasoning="", answer="Plain answer.")
    app.router = router
    conversation_id = "conv-no-reasoning"
    events = [event async for event in app.ask_stream("Just answer", conversation_id=conversation_id)]

    assert not any(event["type"] == "thinking.delta" for event in events)
    assert any(
        event["type"] == "thinking.state" and event["empty"] is True and event["delta"] == ""
        for event in events
    )
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["content"] == "Plain answer."
    assert finished["reasoning_seen"] is False
    rows = await db.fetchall(
        "SELECT metadata FROM messages WHERE conversation_id=? AND role='assistant'", (conversation_id,)
    )
    assert len(rows) == 1 and "reasoning" not in json.loads(rows[0]["metadata"])
    await app.shutdown()


async def test_blocking_ask_never_persists_model_reasoning_or_raw_thinking() -> None:
    import json

    from leo import db

    class ReasoningAnswerRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            return ModelResponse(
                content="Visible answer",
                model="fake", provider="fake",
                reasoning="SECRET THOUGHT",
                raw={"reasoning": "SECRET THOUGHT", "nested": {"reasoning_content": "SECRET THOUGHT"}},
                usage=Usage(input_tokens=5, output_tokens=3),
            )

    app = LeoApp(_settings())
    await app.initialize()
    app.router = ReasoningAnswerRouter()
    conversation_id = "conv-no-persisted-thought"
    response = await app.ask("Explain", channel="test", conversation_id=conversation_id)

    assert response["content"] == "Visible answer"
    rows = await db.fetchall(
        "SELECT content, metadata FROM messages WHERE conversation_id=? AND role='assistant'",
        (conversation_id,),
    )
    assert len(rows) == 1
    assert "SECRET THOUGHT" not in rows[0]["content"]
    assert "SECRET THOUGHT" not in rows[0]["metadata"]
    meta = json.loads(rows[0]["metadata"])
    assert meta["reasoning"] == ""
    assert "reasoning" not in meta["raw"]
    assert "reasoning_content" not in meta["raw"]["nested"]
    await app.shutdown()


# ------------------------------------------------------------------- slash commands
class _ModelMustNotRun:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        raise AssertionError("slash input must never call the model")

    async def stream(self, request):
        self.calls += 1
        raise AssertionError("slash input must never call the model")
        yield "unreachable"

    async def stream_events(self, request):
        self.calls += 1
        raise AssertionError("slash input must never call the model")
        yield StreamChunk(kind="text", text="unreachable")


async def test_app_wires_finalized_slash_registry() -> None:
    # The composition root owns one authoritative catalog with real handlers.
    assert isinstance(create_registry(), CommandRegistry)
    app = LeoApp(_settings())
    assert app.services == {}
    await app.initialize()
    try:
        assert isinstance(app.commands, CommandRegistry)
        assert app.services["commands"] is app.commands
        available = {spec.name: spec for spec in app.commands.all() if spec.available}
        # /export (and its aliases) is the latest shipped slash command: it must
        # be live in the composed catalog with a callable handler and chat surface.
        # A floor, not an equality: an exact count turns every newly implemented
        # command into a false failure, while a floor still catches a command
        # silently disappearing from the composed catalog.
        assert len(available) >= 38
        export = available["export"]
        assert callable(export.handler)
        assert "chat" in export.surfaces
        compact = available["compact"]
        assert callable(compact.handler) and "chat" in compact.surfaces
        assert len(app.commands.all()) > 100  # unavailable commands stay visible
    finally:
        await app.shutdown()


async def test_slash_help_works_in_both_transports_without_a_model_call() -> None:
    import json

    from leo import db

    app = LeoApp(_settings())
    await app.initialize()
    router = _ModelMustNotRun()
    app.router = router

    blocking = await app.ask("/help", channel="chat-test", conversation_id="conv-slash-help")
    assert "LEO slash commands" in blocking["content"]
    assert blocking["command"] == "help"
    assert blocking["command_status"] == "ok"
    assert blocking["status"] == "completed"
    assert blocking["provider"] == "slash"

    conversation_id = "conv-slash-help-stream"
    events = [event async for event in app.ask_stream("/help", channel="chat-test", conversation_id=conversation_id)]
    assert router.calls == 0
    command_events = [event for event in events if event["type"] == "command.result"]
    assert len(command_events) == 1
    assert command_events[0]["command"] == "help"
    assert command_events[0]["success"] is True
    assert "LEO slash commands" in command_events[0]["content"]
    finished = next(event for event in events if event["type"] == "message.finished")
    assert finished["status"] == "completed" and "LEO slash commands" in finished["content"]
    assert finished["reasoning_seen"] is False
    assert events[-1]["type"] == "run.finished"
    assert events[-1]["status"] == "completed"
    assert events[0]["type"] == "run.started"
    assert events[-1]["run_id"] not in app._runs

    for conv in (blocking["conversation_id"], conversation_id):
        rows = await db.fetchall(
            "SELECT role, content, metadata FROM messages WHERE conversation_id=? ORDER BY id", (conv,)
        )
        assert [row["role"] for row in rows] == ["user", "assistant"]
        assert all(json.loads(row["metadata"]).get("slash") is True for row in rows)
        assert json.loads(rows[1]["metadata"])["command"] == "help"
    assert any(
        event["topic"] == "chat.completed" for event in app._durable_events["chat-lifecycle"]
    )
    await app.shutdown()


async def test_slash_unknown_command_is_honest_and_denied_operator_command_never_runs() -> None:
    app = LeoApp(_settings())
    await app.initialize()
    router = _ModelMustNotRun()
    app.router = router

    unknown = await app.ask("/definitely_not_a_command", channel="chat-test")
    assert unknown["status"] == "failed"
    assert unknown["command_status"] == "error"
    assert unknown["command_success"] is False
    assert unknown["command_result"]["code"] == "unknown_command"
    assert "Unknown command" in unknown["content"]

    events = [event async for event in app.ask_stream(
        "/approve apr_123", channel="chat-test", conversation_id="conv-slash-approve"
    )]
    denied = next(event for event in events if event["type"] == "command.result")
    assert denied["status"] == "denied"
    assert denied["success"] is False
    assert denied["command_status"] == "denied"
    assert "chat surface" in denied["content"].lower()
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "denied"
    assert router.calls == 0
    await app.shutdown()
