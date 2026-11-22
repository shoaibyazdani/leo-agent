"""Live progress: one edited message per working turn, and nothing else.

The report exists so a long turn is visible instead of silent, which means the
things worth pinning are: it is one message, it is throttled, it never leaks a
secret from a command line, and it can never break the turn it describes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from leo.integrations.progress import ProgressReporter


class FakeAdapter:
    channel = "telegram"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.edits: list[tuple[str, str, str]] = []
        self.fail_send = False

    async def send(self, target: str, text: str, *, reply_markup=None):
        if self.fail_send:
            raise RuntimeError("network down")
        self.sent.append((str(target), text))
        return {"message_id": str(len(self.sent))}

    async def edit_message_text(self, target: str, message_id: str, text: str, reply_markup=None):
        self.edits.append((str(target), str(message_id), text))
        return True


class FakeGateway:
    def __init__(self, adapters: dict) -> None:
        self.adapters = adapters

    def _adapter(self, channel: str):
        if channel not in self.adapters:
            raise KeyError(channel)
        return self.adapters[channel]


@pytest.fixture
def env():
    adapter = FakeAdapter()
    gateway = FakeGateway({"telegram": adapter})
    settings = SimpleNamespace(progress=SimpleNamespace(
        enabled=True, min_interval_seconds=0.01, max_steps=4, idle_seconds=30.0,
    ))
    return SimpleNamespace(reporter=ProgressReporter(gateway, settings), adapter=adapter,
                           settings=settings, gateway=gateway)


def test_disabled_reports_nothing(env) -> None:
    env.settings.progress.enabled = False
    reporter = ProgressReporter(env.gateway, env.settings)

    assert reporter.begin("telegram", "100000001") is False
    reporter.note("shell", {"command": "ls"})
    assert env.adapter.sent == []


def test_a_channel_without_an_adapter_is_not_reported(env) -> None:
    assert env.reporter.begin("cli", "local") is False


async def test_the_first_step_sends_one_message_and_later_steps_edit_it(env) -> None:
    reporter = env.reporter
    assert reporter.begin("telegram", "100000001", budget=250) is True

    reporter.note("shell", {"command": "cd /root && pytest -q"})
    await reporter._flush(force=True)
    reporter.note("write_file", {"path": "leo/app.py"})
    await reporter._flush(force=True)

    assert len(env.adapter.sent) == 1, "progress must be one message, not a stream"
    assert len(env.adapter.edits) == 1
    target, _message_id, edited = env.adapter.edits[0]
    assert target == "100000001"
    assert "Working" in edited
    assert "✍️ Writing leo/app.py" in edited
    assert "iteration 2/250" in edited


async def test_a_burst_of_steps_is_throttled_into_the_latest_state(env) -> None:
    reporter = env.reporter
    env.settings.progress.min_interval_seconds = 30.0
    reporter.begin("telegram", "100000001")

    for index in range(5):
        reporter.note("shell", {"command": f"step {index}"})
        await reporter._flush()          # no force: only the first one gets out

    assert len(env.adapter.sent) == 1, "a burst is one message, not five"
    # The next flush (the heartbeat, or a forced one) shows the newest state only.
    reporter._last_flush = 0.0
    await reporter._flush()
    text = env.adapter.edits[-1][2] if env.adapter.edits else env.adapter.sent[-1][1]
    assert "step 4" in text, "the newest state wins"
    assert "step 0" not in text, "old steps are dropped past max_steps"


async def test_commands_are_redacted_and_truncated(env) -> None:
    reporter = env.reporter
    reporter.begin("telegram", "100000001")

    # Built at runtime so the fixture exercises the redaction path without ever
    # existing as a literal credential in the repo (a scanner cannot tell a fake
    # from a real one, and neither can a human skimming the diff).
    token = "sk-" + "x" * 24
    reporter.note("shell", {"command": f"curl -H 'Authorization: Bearer {token}' x"})
    reporter.note("shell", {"command": "echo " + "x" * 400})

    lines = reporter.render().splitlines()[1:]
    assert any("[redacted]" in line for line in lines)
    assert all(len(line) <= 120 for line in lines), "a step line stays one readable line"


async def test_end_leaves_a_verdict_in_the_same_message(env) -> None:
    reporter = env.reporter
    reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})
    await reporter._flush(force=True)

    await reporter.end("done")

    final = env.adapter.edits[-1][2]
    assert "✅" in final and "done in" in final
    assert len(env.adapter.sent) == 1


async def test_a_failed_send_never_raises(env) -> None:
    env.adapter.fail_send = True
    reporter = env.reporter
    reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})

    await reporter._flush(force=True)   # must not raise

    assert reporter.stats()["failed"] == 1


async def test_the_idle_ticker_finishes_an_abandoned_turn(env) -> None:
    """A crashed or abandoned turn must not leave "Working" on screen forever."""
    reporter = env.reporter
    reporter.interval = 0.01
    reporter.idle_seconds = 5.0
    reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})
    reporter._last_step = reporter._started - 10  # pretend the turn died ten seconds ago
    reporter._started -= 10

    await asyncio.sleep(0.05)          # let one tick run

    assert reporter.stats()["active"] is False
    closing = [text for _t, text in env.adapter.sent] + [edit[2] for edit in env.adapter.edits]
    assert any("✅" in text for text in closing), closing
