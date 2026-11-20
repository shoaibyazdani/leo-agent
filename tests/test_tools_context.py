"""Executor-level coverage for the read-only context inspection toolset.

Messages are seeded into the real SQLite tables and every call goes through
ToolExecutorImpl, so schema validation, permissions, audit, events and redaction
are exercised together with the handlers.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.models.context import ContextOptimizer
from leo.security.permissions import PermissionManagerImpl
from leo.tools.context_tools import TOOL_NAMES, register
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry


class ServicesExecutor(ToolExecutorImpl):
    """ToolExecutorImpl with the composition-root service injection wired in."""

    def __init__(self, *args: object, services: object = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.injected_services = services

    def _context(self, call: ToolCall, task_key: str, dry_run: bool) -> dict:
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.injected_services
        return ctx


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    assert register(registry, services=None) == list(TOOL_NAMES)
    executor = ServicesExecutor(
        registry, permissions, events, settings, services={}, timeout_seconds=5.0
    )
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        registry=registry, executor=executor,
    )


_UNSET = object()


async def _run(
    env, name: str, arguments: dict | None = None, *, services: object = None,
    correlation_id: str | None | object = _UNSET,
    actor: str = "context-tester", task_id=None,
):
    env.executor.injected_services = {} if services is None else services
    args = arguments or {}
    # A real chat call carries its active conversation id as the correlation id.
    current = (
        args.get("conversation_id") or "corr-context-tools"
        if correlation_id is _UNSET else correlation_id
    )
    call = ToolCall(
        name=name, arguments=args,
        actor=actor, correlation_id=current, task_id=task_id,
    )
    return await env.executor.execute(call)


SEED_ROWS = [
    ("system", "You are LEO."),
    ("user", "hello world"),
    ("user", "hello world"),  # exact duplicate; must be counted and removed
    ("assistant", "reply"),
    ("tool", "result"),
]
SECRET = (
    "Authorization: Bearer S3CR3TVALUE123456 password=hunter2 "
    "sk-abcdefghijklmnop " + "q" * 200
)


async def _seed(conversation_id: str, rows: list[tuple[str, str]]) -> None:
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,created_at,updated_at) VALUES(?,?,?,?)",
        (conversation_id, "test", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    await db.executemany(
        "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
        [
            (conversation_id, role, content, f"2026-01-01T00:00:{index:02d}+00:00")
            for index, (role, content) in enumerate(rows)
        ],
    )


# --------------------------------------------------------------------------- specs


async def test_register_typed_specs_and_validation(env):
    assert list(TOOL_NAMES) == ["context_stats", "token_estimate", "context_preview"]
    assert [spec.name for spec in env.registry.list_specs()] == list(TOOL_NAMES)
    for name in TOOL_NAMES:
        spec = env.registry.get(name).spec
        assert spec.category == "context"
        assert spec.risk is RiskLevel.LOW
        assert spec.permissions == ["context:read"]
        assert spec.resource_field == ""
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert isinstance(spec.input_schema["properties"], dict)
        assert callable(env.registry.get(name).handler)

    assert register(ToolRegistry(), services={}) == list(TOOL_NAMES)
    assert env.registry.validate("context_stats", {"conversation_id": "c", "budget": 100}) == []
    assert env.registry.validate("token_estimate", {"text": "abcd"}) == []
    assert env.registry.validate("context_preview", {"max_messages": 2, "max_chars": 80}) == []

    bad_calls = [
        ("context_stats", {"budget": 0}),
        ("token_estimate", {"text": 7}),
        ("context_preview", {"max_chars": 0}),
        ("context_preview", {"max_messages": 21}),
        ("context_preview", {"unexpected": 1}),
    ]
    for name, arguments in bad_calls:
        result = await _run(env, name, arguments)
        assert result.success is False
        assert result.decision == "invalid_arguments"
        assert "invalid arguments" in (result.error or "")


# --------------------------------------------------------------------------- stats


async def test_context_stats_counts_real_seeded_messages(env):
    await _seed("conv-stats", SEED_ROWS)
    result = await _run(env, "context_stats", {"conversation_id": "conv-stats"})
    assert result.success and result.decision == "allowed"
    output = result.output
    assert output["source"] == "db:conversation"
    assert output["message_count"] == len(SEED_ROWS) == 5
    assert output["unique_count"] == 4
    assert output["duplicate_count"] == 1
    assert output["role_counts"] == {"system": 1, "user": 2, "assistant": 1, "tool": 1}
    assert output["total_chars"] == sum(len(content) for _, content in SEED_ROWS)
    assert output["estimated_tokens"] == sum(math.ceil(len(content) / 4) for _, content in SEED_ROWS)
    assert "characters/4" in output["token_method"]
    assert output["estimated"] is True
    assert "budget" not in output

    fitted = await _run(env, "context_stats", {"conversation_id": "conv-stats", "budget": 5})
    assert fitted.success
    assert fitted.output["budget"] == 5
    assert 1 <= fitted.output["optimized_message_count"] < fitted.output["unique_count"]
    assert fitted.output["optimized_estimated_tokens"] <= output["estimated_tokens"]
    assert fitted.output["bytes_dropped_by_budget"] > 0

    rows = await db.fetchall("SELECT tool, decision, success FROM tool_calls WHERE tool='context_stats'")
    assert len(rows) == 2
    assert all(row["decision"] == "allowed" and row["success"] == 1 for row in rows)


async def test_context_stats_limit_and_empty_conversation(env):
    await _seed("conv-limit", [("user", f"message-{index}") for index in range(8)])
    result = await _run(env, "context_stats", {"conversation_id": "conv-limit", "limit": 3})
    assert result.success
    assert result.output["message_count"] == 3  # newest three rows, no full-table scan
    empty = await _run(env, "context_stats", {"conversation_id": "missing"})
    assert empty.success
    assert empty.output["message_count"] == 0
    assert empty.output["role_counts"] == {}
    assert empty.output["estimated_tokens"] == 0


# --------------------------------------------------------------------------- tokens


async def test_token_estimate_is_explicitly_a_characters_over_four_estimate(env):
    text = "abcd" * 3 + "a"  # 13 characters -> 4 estimated tokens
    result = await _run(env, "token_estimate", {"text": text})
    assert result.success and result.decision == "allowed"
    output = result.output
    assert output["source"] == "text"
    assert output["chars_per_token"] == 4
    assert output["text_chars"] == 13
    assert output["estimated_tokens"] == 4
    assert output["estimated"] is True
    assert "characters/4" in output["method"]
    assert "provider" in output["method"] and "not a provider" in output["method"]

    await _seed("conv-tokens", SEED_ROWS)
    stored = await _run(env, "token_estimate", {"conversation_id": "conv-tokens"})
    assert stored.success
    assert stored.output["source"] == "db:conversation"
    # The estimate is computed on the optimizer's deduplicated view (one of the
    # two identical "hello world" turns is dropped).
    assert stored.output["message_count"] == len(SEED_ROWS) - 1
    unique_contents = [content for _, content in dict.fromkeys(SEED_ROWS)]
    assert stored.output["text_chars"] == sum(len(content) for content in unique_contents)
    assert stored.output["estimated_tokens"] == sum(
        math.ceil(len(content) / 4) for content in unique_contents
    )
    assert sum(item["estimated_tokens"] for item in stored.output["by_role"]) == stored.output["estimated_tokens"]
    assert {item["role"] for item in stored.output["by_role"]} == {
        "system", "user", "assistant", "tool"
    }

    conflict = await _run(env, "token_estimate", {"text": "x", "conversation_id": "conv-tokens"})
    assert conflict.success is False
    assert "either 'text' or 'conversation_id'" in (conflict.error or "")


# --------------------------------------------------------------------------- preview


async def test_context_preview_redacts_secrets_and_is_bounded(env):
    await _seed("conv-secret", [("system", "safe"), ("user", SECRET)])
    result = await _run(env, "context_preview", {
        "conversation_id": "conv-secret", "max_messages": 2, "max_chars": 80,
    })
    assert result.success and result.decision == "allowed"
    output = result.output
    assert output["count"] == 2
    assert output["max_messages"] == 2 and output["max_chars"] == 80
    assert output["total_chars"] <= 80
    assert output["messages"][1]["truncated"] is True
    assert output["redacted"] is True
    serialized = str(output)
    assert "[REDACTED]" in serialized
    for leaked in ("S3CR3TVALUE123456", "hunter2", "abcdefghijklmnop"):
        assert leaked not in serialized
    assert output["estimated_tokens"] == math.ceil(output["total_chars"] / 4)

    # Even when the preview budget is small, no secret from the clipped tail leaks.
    tight = await _run(env, "context_preview", {
        "conversation_id": "conv-secret", "max_chars": 12,
    })
    assert tight.success
    assert tight.output["total_chars"] <= 12
    assert "S3CR3TVALUE123456" not in str(tight.output)

    row = await db.fetchone("SELECT COUNT(*) AS n FROM tool_calls WHERE tool='context_preview'")
    assert row["n"] == 2  # two audited preview calls


async def test_context_preview_caps_message_count(env):
    await _seed("conv-many", [("user", f"row-{index}-" + "z" * 40) for index in range(7)])
    result = await _run(env, "context_preview", {
        "conversation_id": "conv-many", "max_messages": 2, "max_chars": 60,
    })
    assert result.success
    assert result.output["count"] == 2
    assert [item["index"] for item in result.output["messages"]] == [0, 1]
    assert result.output["messages"][1]["truncated"] is True
    assert result.output["total_chars"] <= 60
    assert result.output["total_chars"] == sum(item["chars"] for item in result.output["messages"])


# --------------------------------------------------------------------------- services


class SpyOptimizer(ContextOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.deduplicate_calls = 0
        self.optimize_calls = 0

    def deduplicate(self, messages):
        self.deduplicate_calls += 1
        return super().deduplicate(messages)

    def optimize(self, messages, budget, memories=None, provider=None):
        self.optimize_calls += 1
        return super().optimize(messages, budget, memories=memories, provider=provider)


async def test_injected_optimizer_and_messages_are_used(env):
    await _seed("conv-ignored", SEED_ROWS)
    spy = SpyOptimizer()
    injected_messages = [
        {"role": "user", "content": "injected alpha"},
        {"role": "user", "content": "injected alpha"},
        {"role": "assistant", "content": "injected beta"},
    ]
    services = {"context": spy, "messages": injected_messages}
    stats = await _run(env, "context_stats", {"conversation_id": "conv-ignored", "budget": 1000}, services=services)
    assert stats.success
    assert stats.output["source"] == "services"
    assert stats.output["message_count"] == 3
    assert stats.output["duplicate_count"] == 1
    assert spy.deduplicate_calls >= 1 and spy.optimize_calls == 1

    tokens = await _run(env, "token_estimate", {"limit": 2}, services=services)
    assert tokens.success
    assert tokens.output["source"] == "services"
    assert tokens.output["message_count"] == 2

    preview = await _run(env, "context_preview", {"max_chars": 100}, services=services)
    assert preview.success
    assert preview.output["source"] == "services"
    assert preview.output["count"] == 2
    assert spy.deduplicate_calls >= 3  # stats, its optimize() path, tokens and preview


async def test_invalid_context_service_is_reported_not_ignored(env):
    for name in TOOL_NAMES:
        result = await _run(env, name, {"text": "abc"} if name == "token_estimate" else {},
                            services={"context": object()})
        assert result.success is False
        assert "ContextOptimizer-like" in (result.error or "")

    bad_mapping = await _run(env, "context_stats", {}, services=["not", "a", "mapping"])
    assert bad_mapping.success is False
    assert "ctx['services'] must be a mapping" in (bad_mapping.error or "")


# --------------------------------------------------------------------------- conversation binding


async def test_conversation_id_must_match_current_conversation(env):
    await _seed("conv-current", [("user", "current conversation content")])
    await _seed("conv-victim", [("user", "VICTIM-ONLY-SECRET")])
    calls = [
        ("context_stats", {"conversation_id": "conv-victim"}),
        ("token_estimate", {"conversation_id": "conv-victim"}),
        ("context_preview", {"conversation_id": "conv-victim"}),
    ]
    for name, arguments in calls:
        result = await _run(
            env, name, arguments, correlation_id="conv-current", actor="chat:cli",
        )
        assert result.success is False
        assert "does not match" in (result.error or "")
        assert "VICTIM-ONLY-SECRET" not in json.dumps(result.model_dump(mode="json"))


async def test_messages_are_always_scoped_to_bound_conversation(env):
    await _seed("conv-current", SEED_ROWS)
    await _seed("conv-other", [("user", "OTHER-CONVERSATION-SECRET")])

    implicit = await _run(env, "context_stats", {}, correlation_id="conv-current")
    assert implicit.success, implicit.error
    assert implicit.output["source"] == "db:conversation"
    assert implicit.output["message_count"] == len(SEED_ROWS)

    via_task = await _run(
        env, "context_preview", {"conversation_id": "conv-current"},
        correlation_id="corr-unrelated", actor="chat:api", task_id="conv-current",
    )
    assert via_task.success, via_task.error
    assert "OTHER-CONVERSATION-SECRET" not in json.dumps(via_task.output)

    unbound = await _run(env, "context_stats", {}, correlation_id=None)
    assert unbound.success is False
    assert "bound conversation" in (unbound.error or "").lower()
