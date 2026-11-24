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
        self.silent: list[bool] = []
        self.edits: list[tuple[str, str, str]] = []
        self.fail_send = False

    async def send(self, target: str, text: str, *, reply_markup=None, silent=False):
        self.silent.append(bool(silent))
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


async def test_disabled_reports_nothing(env) -> None:
    env.settings.progress.enabled = False
    reporter = ProgressReporter(env.gateway, env.settings)

    assert await reporter.begin("telegram", "100000001") is False
    reporter.note("shell", {"command": "ls"})
    assert env.adapter.sent == []


async def test_a_channel_without_an_adapter_is_not_reported(env) -> None:
    assert await env.reporter.begin("cli", "local") is False


async def test_the_first_step_sends_one_message_and_later_steps_edit_it(env) -> None:
    reporter = env.reporter
    assert await reporter.begin("telegram", "100000001", budget=250) is True

    reporter.note("shell", {"command": "cd /root && pytest -q"})
    await reporter._flush(force=True)
    reporter.note("write_file", {"path": "leo/app.py"})
    await reporter._flush(force=True)

    assert len(env.adapter.sent) == 1, "progress must be one message, not a stream"
    assert len(env.adapter.edits) == 1
    target, _message_id, edited = env.adapter.edits[0]
    assert target == "100000001"
    assert edited.startswith("🛠"), edited
    assert "2 steps" in edited, edited
    # Each row is glyph + monospace label + monospace detail, so the columns
    # read as a set rather than as prose.
    assert "✓ `shell`" in edited, edited
    assert "✓ `write_file`" in edited, edited
    assert "`leo/app.py`" in edited, edited


async def test_a_burst_of_steps_is_throttled_into_the_latest_state(env) -> None:
    reporter = env.reporter
    env.settings.progress.min_interval_seconds = 30.0
    await reporter.begin("telegram", "100000001")

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
    await reporter.begin("telegram", "100000001")

    # Built at runtime so the fixture exercises the redaction path without ever
    # existing as a literal credential in the repo (a scanner cannot tell a fake
    # from a real one, and neither can a human skimming the diff).
    token = "sk-" + "x" * 24
    reporter.note("shell", {"command": f"curl -H 'Authorization: Bearer {token}' x"})
    reporter.note("shell", {"command": "echo " + "x" * 400})

    # Redaction happens when the step is recorded; the renderer may then coalesce
    # a run of the same tool down to its latest detail.
    raw = list(reporter._steps)
    assert any("[redacted]" in line for line in raw), raw
    assert all("sk-" not in line for line in raw), raw
    assert all(len(line) <= 80 for line in raw), raw
    rendered = reporter.render()
    assert len(rendered.splitlines()) <= reporter.max_steps + 1


async def test_end_leaves_a_verdict_in_the_same_message(env) -> None:
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})
    await reporter._flush(force=True)

    await reporter.end("done")

    final = env.adapter.edits[-1][2]
    assert "✅" in final and "done in" in final
    assert len(env.adapter.sent) == 1


async def test_a_failed_send_never_raises(env) -> None:
    env.adapter.fail_send = True
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})

    await reporter._flush(force=True)   # must not raise

    assert reporter.stats()["failed"] == 1


async def test_the_idle_ticker_finishes_an_abandoned_turn(env) -> None:
    """A crashed or abandoned turn must not leave "Working" on screen forever."""
    reporter = env.reporter
    reporter.interval = 0.01
    reporter.idle_seconds = 5.0
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})
    reporter._last_step = reporter._started - 10  # pretend the turn died ten seconds ago
    reporter._started -= 10

    await asyncio.sleep(0.05)          # let one tick run

    assert reporter.stats()["active"] is False
    closing = [text for _t, text in env.adapter.sent] + [edit[2] for edit in env.adapter.edits]
    assert any("✅" in text for text in closing), closing


async def test_an_unchanged_report_is_not_re_edited(env) -> None:
    """Re-editing identical text is what Telegram rejects as "not modified".

    The retry that followed used to open a brand-new message every few seconds,
    which is exactly how a long turn's log fell apart.
    """
    reporter = env.reporter
    reporter.interval = 0.0
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})

    await reporter._flush(force=True)
    edits_after_first = len(env.adapter.edits)
    await reporter._flush()          # same content, no force

    assert len(env.adapter.edits) == edits_after_first, "identical text must not be re-edited"
    assert len(env.adapter.sent) == 1


async def test_a_failed_edit_reopens_the_message_only_once(env) -> None:
    class Refusing(FakeAdapter):
        async def edit_message_text(self, *args, **kwargs):
            return False

    adapter = Refusing()
    reporter = ProgressReporter(FakeGateway({"telegram": adapter}), env.settings)
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})
    await reporter._flush(force=True)
    reporter.note("shell", {"command": "pwd"})
    await reporter._flush(force=True)

    assert len(adapter.sent) == 2, "a genuinely failed edit falls back to a new message"


async def test_a_new_turn_closes_the_previous_report(env) -> None:
    """Two open "Working" messages in one chat is what a broken log looks like."""
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})
    await reporter._flush(force=True)

    await reporter.begin("telegram", "100000001")     # the next turn starts

    closing = [edit[2] for edit in env.adapter.edits]
    assert any("✅" in text for text in closing), "the previous report must be closed"
    assert reporter.stats()["active"] is True
    await reporter.end("done")


async def test_every_progress_line_is_silent(env) -> None:
    """The reporter must never ring the phone — the answer itself still notifies."""
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "ls"})
    await reporter._flush(force=True)
    await reporter.end("done")

    assert env.adapter.silent, "the reporter sent nothing"
    assert all(env.adapter.silent), "every progress line must be silent"


async def test_a_turn_with_no_tool_work_reports_nothing(env) -> None:
    """A one-line answer must not carry a "done in 0s" footer.

    The screenshot bug was "done in 12s — 0 step(s)" printed twice; the deeper
    bug was that the footer existed at all under a reply that did no work.
    """
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")

    await reporter.end("done")
    await reporter._flush(force=True)

    assert env.adapter.sent == [], "a turn with no work must not open a report"
    assert env.adapter.edits == []


async def test_a_lookup_only_turn_stays_silent(env) -> None:
    """Reading one file or searching memory is an answer, not a task."""
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")

    reporter.note("read_file", {"path": "leo/config.py"})
    reporter.note("memory_search", {"query": "concise replies"})
    await reporter._flush(force=True)
    await reporter.end("done")

    assert env.adapter.sent == [], "lookups alone must not open a report"
    assert env.adapter.edits == []


async def test_enough_steps_earn_a_report_even_without_a_shell_call(env) -> None:
    """Past `min_steps` the turn is real work, whatever tools it used."""
    reporter = env.reporter
    reporter.min_steps = 3
    await reporter.begin("telegram", "100000001")

    for index in range(3):
        reporter.note("read_file", {"path": f"leo/module_{index}.py"})
    await reporter._flush(force=True)

    assert len(env.adapter.sent) == 1, "a long read-only turn is still work"


async def test_the_outcome_is_not_repeated_after_real_work(env) -> None:
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "pytest -q"})
    await reporter._flush(force=True)

    await reporter.end("done")
    await reporter._flush(force=True)

    text = env.adapter.edits[-1][2] if env.adapter.edits else env.adapter.sent[-1][1]
    assert text.count("done") == 1, text
    assert text.count("step") == 1, text


async def test_a_stopped_turn_says_stopped_not_done(env) -> None:
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "pytest -q"})

    await reporter.end("stopped")
    await reporter._flush(force=True)

    text = env.adapter.edits[-1][2] if env.adapter.edits else env.adapter.sent[-1][1]
    assert "stopped" in text, text
    assert "done" not in text, text


def test_a_shell_command_shows_the_work_not_its_preamble(env) -> None:
    """`cd /root/x && pytest -q` reads as `pytest -q`; the cd is noise."""
    reporter = env.reporter
    reporter.begin_sync = None  # not part of this assertion
    from leo.integrations.progress import _step_line

    assert _step_line("shell", {"command": "cd /root/ws && pytest -q"}, None) == "✓ `shell` `pytest -q`"
    # `set`/`source`/`cd` chains reduce to the last clause, which is the real work.
    assert _step_line("shell", {"command": "cd /w && set -a && pytest -q"}, None) == "✓ `shell` `pytest -q`"
    # A single command with no chain keeps its own text.
    assert _step_line("shell", {"command": "git status"}, None) == "✓ `shell` `git status`"


def test_a_step_with_no_detail_does_not_dangle(env) -> None:
    from leo.integrations.progress import _step_line

    assert _step_line("list_files", {}, None) == "✓ `list_files`"


async def test_an_unchanged_edit_does_not_open_a_second_message(env) -> None:
    """The "sent: 2" bug: a re-edit of identical text must not become a new message.

    Telegram answers an identical edit with HTTP 400 "message is not modified".
    Treating that as a failure made the reporter fall back to `send`, so one turn
    appeared in the chat twice — the duplicate the screenshot shows.
    """
    class IdempotentAdapter:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.edits = 0

        async def send(self, target: str, text: str, *, silent: bool = False):
            self.sent.append(text)
            return {"message_id": "1"}

        async def edit_message_text(self, target, message_id, text, extra=None):
            self.edits += 1
            # Identical text is "not modified": the adapter reports success.
            return True

    adapter = IdempotentAdapter()

    class _Gateway:
        def _adapter(self, channel):
            return adapter

    reporter = ProgressReporter(_Gateway(), env.settings)
    reporter.enabled = True
    reporter.interval = 0.01
    await reporter.begin("telegram", "100000001")
    reporter.note("shell", {"command": "pytest -q"})
    await reporter._flush(force=True)
    await reporter.end("done")
    await reporter._flush(force=True)
    # A second end/flush with the same content must not add a message.
    await reporter._flush(force=True)

    assert len(adapter.sent) == 1, f"one turn opened {len(adapter.sent)} messages"


async def test_the_models_status_sentence_sits_above_the_work(env) -> None:
    """The Claude-style half: what it found and is doing, not only tool rows."""
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.narrate("Found the quoted-operator bug; patching the splitter now.")
    reporter.note("shell", {"command": "pytest -q"})
    await reporter._flush(force=True)

    text = env.adapter.sent[0][1]
    assert "📍 Found the quoted-operator bug; patching the splitter now." in text, text
    assert text.index("📍") < text.index("✓ `shell`"), "the story comes before the log"


async def test_only_the_latest_narration_is_shown(env) -> None:
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    reporter.narrate("Reading the config loader.")
    reporter.narrate("Found it; patching the parser now.")
    reporter.note("shell", {"command": "pytest -q"})
    await reporter._flush(force=True)

    text = env.adapter.sent[0][1]
    assert "Found it; patching the parser now." in text, text
    assert "Reading the config loader." not in text, "one line, latest wins"


async def test_narration_is_one_clean_redacted_line(env) -> None:
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    token = "sk-" + "y" * 24
    reporter.narrate(f"**Found it** — token: {token}\nthis second line must not appear")
    reporter.note("shell", {"command": "ls"})
    await reporter._flush(force=True)

    text = env.adapter.sent[0][1]
    narration = next(line for line in text.splitlines() if line.startswith("📍"))
    assert "📍 Found it" in narration, text
    assert "**" not in narration, "Telegram legacy Markdown markers are stripped"
    assert "sk-" not in text, "a secret must never reach the chat"
    assert "[redacted]" in text, text
    assert "second line" not in text, "the card has one narration line"


async def test_repeated_calls_collapse_into_one_row(env) -> None:
    """`shell ×3 · last` — twenty shell calls are one fact, not twenty rows.

    The step deque must be longer than the rendered window: capping it at
    max_steps evicted the earlier calls before they could be counted, so the
    collapse silently did nothing.
    """
    reporter = env.reporter
    await reporter.begin("telegram", "100000001")
    for command in ("pytest -q", "git status", "python -m leo serve"):
        reporter.note("shell", {"command": command})
    await reporter._flush(force=True)

    text = reporter.render()
    assert "×3" in text, text
    assert "python -m leo serve" in text, "the newest call is the one shown"
    assert text.count("`shell`") == 1, "collapsed calls must be one row"
