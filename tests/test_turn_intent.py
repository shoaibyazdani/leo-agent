"""Continuation intent and resume instruction helpers."""

from __future__ import annotations

from leo.core.turn_intent import (
    build_resume_instruction,
    is_ack,
    is_continuation,
    is_synthetic_row,
)


def test_continuation_prompts_are_recognised() -> None:
    for text in (
        "continue", "Continue.", "please continue", "keep going", "go on",
        "carry on", "proceed", "resume", "finish it", "do the needful",
    ):
        assert is_continuation(text), text
    for text in ("continue building the parser", "yes please continue", "hello"):
        assert not is_continuation(text), text


def test_acknowledgements_are_not_objectives() -> None:
    assert is_ack("ok")
    assert is_ack("Thanks!")
    assert not is_ack("ok now implement it")


def test_synthetic_rows_are_skipped() -> None:
    assert is_synthetic_row({"slash": True})
    assert is_synthetic_row({"command": "compact"})
    assert is_synthetic_row({"resume": True})
    assert is_synthetic_row({"role_kind": "command"})
    assert not is_synthetic_row({"attachments": []})
    assert not is_synthetic_row(None)


def test_resume_instruction_quotes_objective_and_state() -> None:
    instruction = build_resume_instruction(
        "implement the parser", {"run_status": "incomplete", "tool_calls_pending": ["shell"]}
    )
    assert "RESUME" in instruction
    assert "implement the parser" in instruction
    assert "run_status: incomplete" in instruction
    assert "shell" in instruction
    assert "Do not restart" in instruction


def test_resume_instruction_bounds_the_objective() -> None:
    instruction = build_resume_instruction("x" * 10_000)
    assert len(instruction) < 5_000
