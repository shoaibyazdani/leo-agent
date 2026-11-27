"""Narrow stall detector: announced work, no tool call, no real answer."""

from __future__ import annotations

from leo.core.stall import looks_like_stall


def test_a_normal_answer_is_not_a_stall() -> None:
    assert not looks_like_stall("The tests pass and the deploy is live.", "stop", [])
    assert not looks_like_stall("", "stop", [])
    assert not looks_like_stall("x" * 900 + " I'll now check.", "stop", [])


def test_tool_calls_are_never_a_stall() -> None:
    assert not looks_like_stall("I'll now run it.", "tool_calls", [{"id": "c1"}])


def test_a_short_plan_ending_reply_is_a_stall() -> None:
    assert looks_like_stall("Found the file. Let me now patch it.", "stop", [])
    assert looks_like_stall("Next: I will run the tests", "stop", [])
    assert looks_like_stall("Here is the plan:\n", "stop", [])


def test_length_finish_reason_with_content_is_a_stall() -> None:
    assert looks_like_stall("A long answer that was cut", "length", [])
    assert looks_like_stall("A long answer that was cut", "max_tokens", [])


def test_promoted_reasoning_tail_can_trigger() -> None:
    assert looks_like_stall(
        "Done.", "stop", [], reasoning_tail="Let me now check the parser."
    )


def test_a_direct_answer_is_not_a_stall_even_with_planning_reasoning() -> None:
    """A visible answer ends the turn; private reasoning is not a reply."""
    reaction = (
        "Cute — fluffy gray cat in a purple hedgehog cave bed, going for the pom-pom."
    )
    assert not looks_like_stall(
        reaction, "stop", [],
        reasoning_tail="Let me now check the open PRs and report back.",
    )
    assert not looks_like_stall(
        "Cute cat.", "stop", [], reasoning_tail="Next: I will run a GitHub check"
    )
    assert not looks_like_stall("The fix is live and verified.", "stop", [])


def test_short_acknowledgements_can_still_be_stalls() -> None:
    assert looks_like_stall("Done.", "stop", [], reasoning_tail="I'll now run tests")
    assert looks_like_stall("OK", "stop", [], reasoning_tail="First, I'll check git")
    assert not looks_like_stall("Done.", "stop", [])


def test_announced_work_is_still_a_stall_with_distracting_reasoning() -> None:
    assert looks_like_stall(
        "I'll now run the tests.", "stop", [], reasoning_tail="nothing to add"
    )
    assert looks_like_stall("Still working on it…", "stop", [])
    assert looks_like_stall("Still checking the PR list", "stop", [])
