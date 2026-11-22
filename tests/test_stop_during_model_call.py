"""Stop must land while the model is thinking, not only between rounds.

Reported live: five /stop messages in a row and LEO kept typing. The turn was
inside a model call, and the only places that observed the flag were the round head
and just before a tool. A model call can run for minutes (the router retries across
providers), so the stop sat in the table untouched.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from leo.core.app import LeoApp, TurnStopped


class _StopFlag:
    """Stands in for the durable turn_cancels lookup."""

    def __init__(self, stops: list[bool]) -> None:
        self.stops = list(stops)
        self.cleared = False
        self.polls = 0

    async def _turn_stop_requested(self, conversation_id: str, since: str) -> bool:
        self.polls += 1
        return self.stops.pop(0) if self.stops else False

    async def _clear_turn_stop(self, conversation_id: str) -> None:
        self.cleared = True


@pytest.mark.asyncio
async def test_a_stop_interrupts_a_long_model_call() -> None:
    flag = _StopFlag([False, True])          # second poll sees the request

    async def slow_model_call() -> str:
        await asyncio.sleep(30)
        return "answer nobody waited for"

    started = time.perf_counter()
    with pytest.raises(TurnStopped):
        await LeoApp._await_with_stop(flag, slow_model_call(), "conv-1", "t0")
    elapsed = time.perf_counter() - started

    assert elapsed < 5, "the wait must end promptly, not when the model finishes"
    assert flag.cleared, "the consumed request must be cleared"


@pytest.mark.asyncio
async def test_a_finished_call_returns_its_value() -> None:
    flag = _StopFlag([])

    async def quick() -> str:
        return "fine"

    assert await LeoApp._await_with_stop(flag, quick(), "conv-1", "t0") == "fine"
    assert flag.cleared is False


@pytest.mark.asyncio
async def test_an_immediate_stop_cuts_the_call_short() -> None:
    flag = _StopFlag([True])

    async def slow() -> str:
        await asyncio.sleep(30)
        return "nope"

    with pytest.raises(TurnStopped):
        await LeoApp._await_with_stop(flag, slow(), "conv-1", "t0")
