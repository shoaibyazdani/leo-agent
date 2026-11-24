"""Shared conversation binding: one precedence rule for every transcript tool."""

from __future__ import annotations

import pytest

from leo.tools.conversation_scope import assert_conversation_visible, current_conversation


def test_trusted_context_keys_win_in_order() -> None:
    assert current_conversation({"conversation_id": "conv-a", "correlation_id": "conv-b"}) == "conv-a"
    assert current_conversation({"current_conversation_id": "conv-c"}) == "conv-c"


def test_task_and_correlation_bindings_are_derived() -> None:
    assert current_conversation({"task_id": "task:conv-task"}) == "conv-task"
    assert current_conversation({"task_id": "conv-task"}) == "conv-task"
    assert current_conversation({"correlation_id": "conv-corr"}) == "conv-corr"


def test_global_and_quota_scopes_are_not_conversations() -> None:
    assert current_conversation({"task_id": "global"}) is None
    assert current_conversation({"correlation_id": "correlation:abc"}) is None
    assert current_conversation({"correlation_id": "actor:x"}) is None
    assert current_conversation(None) is None


def test_ownership_metadata_hides_a_foreign_conversation() -> None:
    ctx = {
        "conversation_id": "conv-a",
        "actor": "agent-a",
        "conversation_actors": {"conv-a": "agent-b"},
    }
    assert current_conversation(ctx) is None

    ctx["conversation_actors"] = {"conv-a": "agent-a"}
    assert current_conversation(ctx) == "conv-a"


def test_assert_visible_allows_absent_metadata_and_refuses_a_foreign_owner() -> None:
    assert_conversation_visible({}, "conv-x")
    assert_conversation_visible({"actor": "a", "conversation_owners": {"conv-y": "b"}}, "conv-x")
    with pytest.raises(ValueError, match="not visible"):
        assert_conversation_visible(
            {"actor": "a", "conversation_owners": {"conv-x": "b"}}, "conv-x"
        )
