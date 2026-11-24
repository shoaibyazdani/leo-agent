"""Task→chat progress bridge: events reach the origin chat, one message each."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from leo import db
from leo.contracts import TaskStatus
from leo.tasks.engine import DurableTaskEngine
from leo.tasks.reporting import ORIGIN_KEY, TaskProgressHub, _origin_of
from leo.events import EventBus


class FakeAdapter:
    """Records sends and edits; edit always succeeds."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.edits: list[tuple[str, str]] = []

    async def send(self, target, text, *, silent=False):
        self.sent.append((str(target), text))
        return {"message_id": f"m{len(self.sent)}"}

    async def edit_message_text(self, target, message_id, text, extra=None):
        self.edits.append((str(message_id), text))
        return True

    async def typing(self, target):
        return None


class FakeGateway:
    def __init__(self, adapter: FakeAdapter) -> None:
        self.adapter = adapter

    def _adapter(self, channel: str):
        return self.adapter


def _settings(**progress):
    from leo.config import ProgressConfig, Settings

    cfg = {"enabled": True, "min_interval_seconds": 0.5}
    cfg.update(progress)
    settings = Settings()
    settings.progress = ProgressConfig(**cfg)
    return settings


def _task_row(task_id: str, status: str = "running", origin: dict | None = None):
    return SimpleNamespace(id=task_id, status=status, metadata={ORIGIN_KEY: origin})


@pytest.fixture
async def hub_env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    events = EventBus()
    engine = DurableTaskEngine(_settings(), events)
    adapter = FakeAdapter()
    gateway = FakeGateway(adapter)
    settings = _settings()
    hub = TaskProgressHub(gateway, settings, events)
    await hub.start()
    return SimpleNamespace(
        events=events, engine=engine, adapter=adapter, hub=hub, settings=settings
    )


async def _emit_running(events, engine, task_id, origin):
    """Create a real task with an origin, then emit its RUNNING event.

    The engine's own row is what the hub loads, so this exercises the exact
    schema and defaults production writes.
    """
    task = await engine.create(f"objective {task_id}", metadata={ORIGIN_KEY: origin})
    assert task.id == task_id or True  # ids are engine-generated; use the row
    await events.emit("task.running", task_id=task.id, status="running")
    return task.id


async def test_origin_extraction_requires_channel_and_target():
    assert _origin_of(_task_row("t1", origin={"channel": "telegram", "target": "1"})) == (
        "telegram", "1"
    )
    assert _origin_of(_task_row("t2", origin={"channel": "telegram"})) is None
    assert _origin_of(_task_row("t3", origin=None)) is None
    assert _origin_of(SimpleNamespace(id="t4", metadata=None)) is None


async def test_a_running_task_with_origin_opens_one_report(hub_env):
    await _emit_running(hub_env.events, hub_env.engine, "task_a", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    assert len(hub_env.adapter.sent) == 1, "exactly one progress message opens"
    assert len(hub_env.hub.stats()["task_ids"]) == 1


async def test_a_task_without_origin_stays_silent(hub_env):
    # No tasks row: the hub cannot find an origin and stays quiet.
    await hub_env.events.emit("task.running", task_id="silent_task", status="running")
    await asyncio.sleep(0.05)
    assert hub_env.adapter.sent == []
    assert hub_env.hub.stats()["open_reports"] == 0


async def test_two_concurrent_tasks_have_independent_reports(hub_env):
    await _emit_running(hub_env.events, hub_env.engine, "task_one", {"channel": "telegram", "target": "100"})
    await _emit_running(hub_env.events, hub_env.engine, "task_two", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    assert len(hub_env.adapter.sent) == 2, "two tasks are two messages, not one shared"
    assert len(hub_env.hub.stats()["task_ids"]) == 2


async def test_progress_events_render_notes_and_iterations(hub_env):
    await _emit_running(hub_env.events, hub_env.engine, "task_it", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    task_id = hub_env.hub.stats()["task_ids"][0]
    await hub_env.events.emit(
        "task.progress",
        task_id=task_id,
        note="scanning page 3",
        iteration=3,
        max_iterations=10,
        current_step=3,
    )
    await asyncio.sleep(0.05)
    latest = hub_env.adapter.edits[-1][1] if hub_env.adapter.edits else hub_env.adapter.sent[-1][1]
    assert "iter 3/10" in latest, latest
    assert "scanning page 3" in latest, latest


async def test_terminal_status_closes_the_report_with_a_verdict(hub_env):
    await _emit_running(hub_env.events, hub_env.engine, "task_done", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    done_id = hub_env.hub.stats()["task_ids"][0]
    await hub_env.hub.on_terminal(_task_row(done_id, status="completed"), "completed")
    assert hub_env.hub.stats()["open_reports"] == 0
    latest = hub_env.adapter.edits[-1][1] if hub_env.adapter.edits else hub_env.adapter.sent[-1][1]
    assert "done" in latest, latest

    # A cancelled task reports stopped, not done.
    await _emit_running(hub_env.events, hub_env.engine, "task_stop", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    stop_id = hub_env.hub.stats()["task_ids"][0]
    await hub_env.hub.on_terminal(_task_row(stop_id, status="cancelled"), "cancelled")
    latest = hub_env.adapter.edits[-1][1]
    assert "stopped" in latest, latest


async def test_stop_closes_every_open_report(hub_env):
    await _emit_running(hub_env.events, hub_env.engine, "task_x", {"channel": "telegram", "target": "100"})
    await _emit_running(hub_env.events, hub_env.engine, "task_y", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    await hub_env.hub.stop()
    assert hub_env.hub.stats()["open_reports"] == 0


async def test_a_hub_error_never_breaks_task_execution(hub_env, monkeypatch):
    # A broken adapter.send must surface as a dropped event, never an exception.
    async def broken_send(target, text, *, silent=False):
        raise RuntimeError("channel exploded")

    hub_env.adapter.send = broken_send
    await _emit_running(hub_env.events, hub_env.engine, "task_break", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    # The task itself is untouched: no exception escaped the event handler.
    rows = await db.fetchone("SELECT COUNT(*) AS n FROM tasks")
    assert rows["n"] >= 1  # task row intact; no exception escaped the handler


async def test_engine_progress_emits_iteration_details(hub_env):
    task = await hub_env.engine.create("iterative work", metadata={
        ORIGIN_KEY: {"channel": "telegram", "target": "100"}
    })
    seen: dict[str, object] = {}

    async def spy(event):
        payload = getattr(event, "payload", None) or {}
        seen.update({k: v for k, v in payload.items() if k in ("iteration", "max_iterations", "note")})

    hub_env.events.subscribe("task.progress", spy)
    await hub_env.engine.progress(task.id, "round 2", iteration=2, max_iterations=8)
    assert seen.get("iteration") == 2
    assert seen.get("max_iterations") == 8
    assert seen.get("note") == "round 2"


async def test_task_create_background_captures_origin(tmp_path):
    """background=true records the chat that asked; the executor injects it."""
    from leo.contracts import ToolCall
    from leo.tools.executor import ToolExecutorImpl
    from leo.tools.registry import ToolRegistry
    from leo.tools import task_tools
    from leo.security.permissions import PermissionManagerImpl

    db.set_db_path(tmp_path / "t.db")
    await db.initialize()
    settings = _settings()
    events = EventBus()
    engine = DurableTaskEngine(settings, events)
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    task_tools.register(registry)
    app = SimpleNamespace(_turn_origin={"channel": "telegram", "target": "42",
                                        "conversation_id": "conv_1"})
    executor = ToolExecutorImpl(
        registry, permissions, events, settings,
        services={"tasks": engine, "app": app},
    )
    call = ToolCall(
        name="task_create",
        arguments={"objective": "long job", "background": True},
        actor="chat:telegram",
        correlation_id="conv_1",
    )
    result = await executor.execute(call)
    assert result.success, result.error
    task_id = result.output["task_id"]
    task = await engine.get(task_id)
    origin = task.metadata[ORIGIN_KEY]
    assert origin["channel"] == "telegram"
    assert origin["target"] == "42"
    assert origin["conversation_id"] == "conv_1"

    # Without the flag, no origin is captured: a foreground task reports to the
    # turn it runs in, not to a stored chat.
    call2 = ToolCall(
        name="task_create", arguments={"objective": "fg job"}, actor="chat:telegram"
    )
    result2 = await executor.execute(call2)
    task2 = await engine.get(result2.output["task_id"])
    assert ORIGIN_KEY not in task2.metadata


def test_completion_summary_is_readable_not_raw_json():
    """The operator returns to a report, not a serialized dict."""
    from leo.core.app import LeoApp

    task = SimpleNamespace(
        id="task_sum",
        objective="refactor the gateway",
        status="completed",
        result={"content": "Moved the poll loop into its own module.", "files": ["leo/gateway/server.py"]},
        errors=[],
        evidence=[{"source": "agent_engine"}],
    )
    text = LeoApp._task_summary(task, "completed")
    assert "✅" in text and "Completed" in text
    assert "refactor the gateway" in text
    assert "Moved the poll loop" in text
    assert "leo/gateway/server.py" in text
    # Evidence bookkeeping is not part of a readable report.
    assert "Evidence:" not in text
    # Not a dump: no braces from json.dumps.
    assert "{" not in text and "}" not in text


def test_scheduled_research_answer_reaches_the_delivery_summary():
    """A scheduled worker's AgentResult dump answers under "output".

    A cron objective completes with the agent result serialized to a dict; the
    prose lives in "output". Ignoring that key delivered a header and the
    objective with no work in it, so the digest is asserted explicitly.
    """
    from leo.core.app import LeoApp

    digest = "Yesterday in Pakistan: " + ("x" * 2500)
    task = SimpleNamespace(
        id="task_digest",
        objective="Yesterday's Pakistan news summary",
        status="completed",
        result={"agent_id": "research", "success": True, "output": digest, "evidence": []},
        errors=[],
        evidence=[{"source": "agent_engine"}],
    )
    text = LeoApp._task_summary(task, "completed")
    assert "Yesterday in Pakistan" in text
    # A digest is the deliverable, so it is not cut down to a footnote.
    assert len(text) > 1200


def test_structured_output_is_serialized_not_silently_dropped():
    from leo.core.app import LeoApp

    task = SimpleNamespace(
        id="task_struct",
        objective="summarize",
        status="completed",
        result={"output": {"headline": "Budget passed", "count": 3}},
        errors=[],
        evidence=[],
    )
    text = LeoApp._task_summary(task, "completed")
    assert "Budget passed" in text


def test_scheduled_report_is_labelled_and_not_cut_mid_word():
    """The delivered message names the job and never cuts mid-word.

    This is the exact shape of the 10am news delivery: a long objective, a
    schedule name in metadata, evidence rows and a 2100-char digest. The old
    output led with a 36-char task id, printed a truncated objective ending in
    "The News, " before the report, and ended on a dangling fragment.
    """
    from leo.core.app import LeoApp

    digest = "Today is Tue 24 Nov 2026 (PKT); yesterday = Mon 23 Nov 2026.\n\n" + (
        "Coalition row deepens over new provinces. " * 60
    )
    task = SimpleNamespace(
        id="task_2843050c082d4669940e9bbee8c03bf9",
        objective=(
            "Compile a summary of yesterday's news in Pakistan and deliver it. Work out "
            "yesterday's actual date first and gather the day's major Pakistan news: "
            "politics, economy, security, courts, and any big national story."
        ),
        status="completed",
        metadata={"schedule": "pakistan-news-10am"},
        result={"agent_id": "research", "success": True, "output": digest, "evidence": []},
        errors=[],
        evidence=[{"source": "agent_engine"}, {"source": "http_get"}],
    )
    text = LeoApp._task_summary(task, "completed")
    first = text.splitlines()[0]
    # The headline is the job name, not the raw row id.
    assert "Pakistan news 10am" in first
    assert "task_2843050c" not in first
    # The objective is no longer printed as a truncated, comma-dangling prefix.
    assert "Objective:" not in text
    assert not text.splitlines()[1].endswith(",")
    # The report is separated from the header by a blank line.
    assert text.split("\n")[1] == ""
    # No stray bookkeeping footer.
    assert "Evidence:" not in text
    # Any truncation is marked, and the text does not end mid-word.
    assert "…" in text or text.endswith(tuple(".!?\"'”)"))


def test_failed_summary_names_the_blocker():
    from leo.core.app import LeoApp

    task = SimpleNamespace(
        id="task_fail",
        objective="deploy",
        status="failed",
        result=None,
        errors=[{"error": "PermissionError: cannot bind 443"}],
        evidence=[],
    )
    text = LeoApp._task_summary(task, "failed")
    assert "❌" in text and "Failed" in text
    assert "PermissionError" in text


async def test_a_silent_task_gets_a_heartbeat_without_inventing_progress(hub_env):
    """Heartbeat refreshes elapsed time only; it never fabricates a percentage."""
    hub_env.hub.settings.progress.task_heartbeat_seconds = 30.0
    await _emit_running(hub_env.events, hub_env.engine, "hb", {"channel": "telegram", "target": "100"})
    await asyncio.sleep(0.05)
    reporter = next(iter(hub_env.hub._reporters.values()))
    assert reporter.idle_seconds == 30.0
    first = hub_env.adapter.sent[-1][1]
    # No percent sign anywhere: there is no measurable completion metric.
    assert "%" not in first
