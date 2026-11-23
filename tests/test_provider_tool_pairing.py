"""A tool result may only travel as a tool result when its call was declared.

Live failure this pins (2026-11-23, beam-free):

    beam-free HTTP 400 for /v1/chat/completions: Upstream request failed:
    [invalid_value] messages[9] is a tool result, but it does not follow an
    assistant message with 'tool_calls' or its results.

LEO surfaced that as "The model could not produce an answer for this turn
(code: model_error)". deepseek tolerated the same history, which is why the
error looked like a flaky model instead of a malformed request.
"""

from __future__ import annotations

from typing import Any

from leo.contracts import ChatMessage, ModelRequest
from leo.models.provider import (
    UNPAIRED_TOOL_PREFIX,
    AnthropicProvider,
    OpenAIProvider,
)

ANTHROPIC_CONFIG = {
    "kind": "anthropic-compatible",
    "base_url": "https://anthropic.example.test",
    "api_key_env": "LEO_TEST_ANTHROPIC_KEY",
    "model": "beam-free",
}
OPENAI_CONFIG = {
    "kind": "openai-compatible",
    "base_url": "https://openai.example.test",
    "api_key_env": "LEO_TEST_OPENAI_KEY",
    "model": "deepseek-v4.1-flash",
}


def _provider(kind: str):
    from leo.config import ModelProviderConfig

    if kind == "anthropic":
        return AnthropicProvider(ModelProviderConfig(**ANTHROPIC_CONFIG), name="beam-free")
    return OpenAIProvider(ModelProviderConfig(**OPENAI_CONFIG), name="deepseek")


def _history() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="fix the failing test"),
        ChatMessage(role="assistant", content="", tool_calls=[
            {"id": "call_ok", "name": "read_file", "arguments": {"path": "a.py"}},
        ]),
        ChatMessage(role="tool", content="file contents", tool_call_id="call_ok", name="read_file"),
        # The orphan: its opener was dropped (compaction, retry, a trimmed window).
        ChatMessage(role="tool", content="stale output", tool_call_id="call_gone", name="shell"),
        ChatMessage(role="user", content="now continue"),
    ]


def _blocks(body: Any) -> list[dict[str, Any]]:
    """Flatten every message's content into a list of blocks, keeping roles."""
    out: list[dict[str, Any]] = []
    for message in body["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    out.append({"role": message["role"], **block})
        else:
            out.append({"role": message["role"], "text": content})
    return out


def test_anthropic_body_never_carries_an_unpaired_tool_result() -> None:
    provider = _provider("anthropic")
    body = provider._body(ModelRequest(messages=_history()), stream=False)
    blocks = _blocks(body)

    results = [b for b in blocks if b.get("type") == "tool_result"]
    assert [b["tool_use_id"] for b in results] == ["call_ok"], "only the declared call may carry a result"
    assert not any(b.get("tool_use_id") == "call_gone" for b in blocks)

    demoted = [b for b in blocks if b.get("type") == "text" and b["text"].startswith(UNPAIRED_TOOL_PREFIX)]
    assert len(demoted) == 1, "the orphan's evidence is kept, as data"
    assert "stale output" in demoted[0]["text"]


def test_openai_body_never_carries_an_unpaired_tool_role() -> None:
    provider = _provider("openai")
    body = provider._body(ModelRequest(messages=_history()), stream=False)

    tool_roles = [m for m in body["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_roles] == ["call_ok"]
    demoted = [
        m for m in body["messages"]
        if m["role"] == "user" and str(m["content"]).startswith(UNPAIRED_TOOL_PREFIX)
    ]
    assert len(demoted) == 1
    assert "stale output" in str(demoted[0]["content"])


def test_a_tool_result_before_any_assistant_turn_is_demoted_on_both_wires() -> None:
    history = [
        ChatMessage(role="tool", content="orphan at the very start", tool_call_id="call_x"),
        ChatMessage(role="user", content="hi"),
    ]
    for kind in ("anthropic", "openai"):
        provider = _provider(kind)
        body = provider._body(ModelRequest(messages=history), stream=False)
        wire = body["messages"]
        if kind == "anthropic":
            blocks = _blocks(body)
            assert not any(b.get("type") == "tool_result" for b in blocks), kind
            assert any(UNPAIRED_TOOL_PREFIX in str(b.get("text", "")) for b in blocks), kind
        else:
            assert not any(m["role"] == "tool" for m in wire), kind
            assert any(UNPAIRED_TOOL_PREFIX in str(m.get("content", "")) for m in wire), kind


def test_one_declaration_pairs_with_one_result_only() -> None:
    """A duplicated result for the same call is the same offence as an orphan."""
    history = [
        ChatMessage(role="assistant", content="", tool_calls=[
            {"id": "call_dup", "name": "shell", "arguments": {"command": "ls"}},
        ]),
        ChatMessage(role="tool", content="first", tool_call_id="call_dup"),
        ChatMessage(role="tool", content="second", tool_call_id="call_dup"),
    ]
    body = _provider("anthropic")._body(ModelRequest(messages=history), stream=False)
    results = [b for b in _blocks(body) if b.get("type") == "tool_result"]
    assert len(results) == 1, "a tool_use id can only be answered once"
    demoted = [b for b in _blocks(body) if str(b.get("text", "")).startswith(UNPAIRED_TOOL_PREFIX)]
    assert len(demoted) == 1 and "second" in demoted[0]["text"]
