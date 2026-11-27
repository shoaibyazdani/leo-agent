"""Judgment watcher: scheduling, quiet hours, signals, pacing and delivery."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings, WatcherConfig
from leo.contracts import ModelRequest
from leo.events import EventBus
from leo.schedules.clock import FrozenClock
from leo.tasks import DurableTaskEngine, Scheduler
from leo.watchers import (
    JudgmentWatcher,
    WatcherState,
    gather_signals,
    is_quiet,
    load_preferences,
    load_state,
    mentions_topic,
    register_watcher_schedule,
)

NOW = datetime(2030, 1, 2, 10, 0, tzinfo=timezone.utc)  # 15:00 PKT, awake
QUIET = datetime(2030, 1, 2, 2, 0, tzinfo=timezone.utc)  # 07:00 PKT, quiet


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    return SimpleNamespace(
        settings=settings,
        events=events,
        engine=DurableTaskEngine(settings, events),
        tmp_path=tmp_path,
    )


def config(**overrides) -> WatcherConfig:
    path = overrides.pop("preferences_path", None)
    cfg = WatcherConfig(**overrides)
    # Tests never read the operator's real preferences file unless they point
    # at one explicitly.
    cfg.preferences_path = path
    return cfg


class FakeModel:
    """Strict-JSON model stub; responses are keyed by signal id."""

    def __init__(self, decisions: dict[str, tuple[bool, str]] | None = None, *, tool_calls=None):
        self.decisions = decisions or {}
        self.tool_calls = tool_calls or []
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest):
        self.requests.append(request)
        decisions = [
            {"id": key, "worthy": worthy, "message": message, "reason": "test"}
            for key, (worthy, message) in self.decisions.items()
        ]
        return SimpleNamespace(content=json.dumps({"decisions": decisions}), tool_calls=self.tool_calls)


class Recorder:
    def __init__(self, fail_first: int = 0, *, fail_always: bool = False):
        self.calls: list[str] = []
        self.fail_first = fail_first
        self.fail_always = fail_always

    async def send(self, text: str) -> str:
        self.calls.append(text)
        if self.fail_always or len(self.calls) <= self.fail_first:
            raise RuntimeError("transport down")
        return "telegram:42"


def watcher(env, model, send, *, cfg=None, now=NOW, preferences=None, fetcher=None):
    path = None
    if preferences is not None:
        path = env.tmp_path / "preferences.txt"
        path.write_text(preferences, encoding="utf-8")
    return JudgmentWatcher(
        cfg or config(),
        model_client=model.complete if model else None,
        send=send.send,
        preferences_path=path,
        clock=lambda: now,
        github_fetcher=fetcher,
    )


async def add_task(
    task_id: str,
    status: str = "completed",
    result: str = "Summary worth reporting",
    metadata: str = "{}",
    updated_at: str = NOW.isoformat(),
) -> None:
    await db.execute(
        "INSERT INTO tasks(id,objective,status,plan,history,dependencies,checkpoints,result,"
        "errors,retry_count,resource_limits,cancellation_requested,created_at,updated_at,"
        "evidence,metadata,version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            task_id, f"objective {task_id}", status, "[]", "[]", "[]", "[]", result,
            "[]", 0, "{}", 0, updated_at, updated_at, "[]", metadata, 0,
        ),
    )


async def add_event(topic: str, payload: dict, *, event_id: str, at=NOW) -> None:
    await db.execute(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) "
        "VALUES(?,?,?,?,?,?)",
        (event_id, topic, "test", json.dumps(payload), None, at.isoformat()),
    )


async def add_waiting_conversation(conversation_id: str, *, subject="Project chat") -> None:
    old = (NOW - timedelta(hours=1)).isoformat()
    await db.execute(
        "INSERT INTO conversations(id,channel,subject,created_at,updated_at,metadata) "
        "VALUES(?,?,?,?,?,?)",
        (conversation_id, "telegram", subject, old, old, "{}"),
    )
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
        (conversation_id, "user", "Can you check the build?", "{}", old),
    )


def json_decisions(*items: tuple[str, bool, str]) -> dict:
    return {key: (worthy, message) for key, worthy, message in items}


# --- cadence and quiet hours -------------------------------------------------

async def test_watcher_job_registers_on_scheduler_with_defaults(env):
    clock = FrozenClock(NOW)
    scheduler = Scheduler(env.engine, env.settings, env.events, clock=clock)
    cfg = config()

    assert cfg.cadence_seconds == 1800
    assert (cfg.waking_start, cfg.waking_end, cfg.timezone) == ("09:00", "23:00", "Asia/Karachi")

    row = await register_watcher_schedule(scheduler, cfg)
    assert row["name"] == "judgment-watcher"
    assert row["expression"] == "1800.0"
    assert row["payload"]["worker"] == "watcher"

    # A restart must not reset the due cursor.
    await db.execute(
        "UPDATE schedules SET next_run_at=? WHERE name='judgment-watcher'", ("2031-01-01T00:00:00+00:00",)
    )
    await register_watcher_schedule(scheduler, cfg)
    stored = await db.fetchone("SELECT next_run_at FROM schedules WHERE name='judgment-watcher'")
    assert stored["next_run_at"] == "2031-01-01T00:00:00+00:00"

    disabled = config(enabled=False)
    await register_watcher_schedule(scheduler, disabled)
    stored = await db.fetchone("SELECT enabled FROM schedules WHERE name='judgment-watcher'")
    assert stored["enabled"] == 0


async def test_quiet_hours_suppresses_everything():
    cfg = config()
    prefs = load_preferences(None, default_quiet_hours=("23:00", "09:00"))
    assert is_quiet(QUIET, prefs, cfg)
    assert not is_quiet(NOW, prefs, cfg)
    # An overnight explicit range excludes 23:00 and includes 08:00.
    custom = load_preferences(None, default_quiet_hours=("22:00", "08:00"))
    assert is_quiet(datetime(2030, 1, 2, 21, 0, tzinfo=timezone.utc), custom, cfg)
    assert not is_quiet(datetime(2030, 1, 2, 3, 0, tzinfo=timezone.utc), custom, cfg)


async def test_quiet_run_does_not_gather_judge_send_or_write_state(env):
    await add_event("task.dead_letter", {"task_id": "t1", "error": "boom"}, event_id="e1")
    model = FakeModel(json_decisions(("delivery:e1", True, "urgent")))
    send = Recorder()
    result = await watcher(env, model, send, now=QUIET).run()

    assert result.suppressed and result.reason == "quiet hours"
    assert model.requests == [] and send.calls == []
    assert await db.fetchone("SELECT value FROM kv WHERE key='watcher:state'") is None


# --- signal gathering --------------------------------------------------------

async def test_gathers_task_outcomes_dead_letters_conversations_and_github(env):
    await add_task("done", result="Release notes are ready")
    await add_task("routine", result="")
    await add_task("dead", status="failed", result="crashed", metadata='{"dead_letter": true}')
    await add_event("task.dead_letter", {"task_id": "dead", "error": "crashed"}, event_id="e1")
    await add_event(
        "task.delivery_failed",
        {"task_id": "done", "channel": "telegram", "error": "adapter down"},
        event_id="e2",
    )
    await add_waiting_conversation("c1")
    state = WatcherState()

    async def fetch(repo):
        items = [{"number": 7, "title": "Fix build", "comments": 0, "state": "open"}]
        if fetch.calls:
            items.append({"number": 8, "title": "Fresh issue", "comments": 0, "state": "open"})
        fetch.calls.append(None)
        return items
    fetch.calls = []

    signals = await gather_signals(
        state=state, now=NOW, config=config(github_repos=["owner/repo"]), github_fetcher=fetch
    )
    kinds = {item.kind for item in signals}
    assert {"task_outcome", "dead_letter", "delivery_failure", "waiting_conversation"} <= kinds
    assert "github_open" not in kinds  # the first sight of a repo seeds silently
    assert not any(item.id == "task:routine:completed" for item in signals)
    assert all(item.urgent for item in signals if item.kind in ("dead_letter", "delivery_failure"))
    assert state.github["owner/repo#7"]["comments"] == 0

    # Only genuinely new items, not the seeded ones, are reported afterward.
    again = await gather_signals(
        state=state, now=NOW, config=config(github_repos=["owner/repo"]), github_fetcher=fetch
    )
    assert [item.id for item in again if item.kind == "github_open"] == ["github:owner/repo#8:opened"]


async def test_first_github_fetch_and_lost_state_seed_silently(env):
    async def fetch(repo):
        return [
            {"number": number, "title": f"Old issue {number}", "comments": 2, "state": "open"}
            for number in range(1, 21)
        ]

    state = WatcherState()
    signals = await gather_signals(
        state=state, now=NOW, config=config(github_repos=["owner/repo"]), github_fetcher=fetch
    )
    assert not any(item.source == "github" for item in signals)
    assert len(state.github) == 20

    # A repository added later is seeded the same way, even if other repos
    # already have entries in the snapshot.
    state.github["other/repo#1"] = {"comments": 0, "state": "open", "due": ""}
    signals = await gather_signals(
        state=state, now=NOW, config=config(github_repos=["new/repo"]), github_fetcher=fetch
    )
    assert not any(item.source == "github" for item in signals)


# --- preferences, filtering and judgment -------------------------------------

async def test_preferences_are_read_every_run_and_excluded_topics_never_send(env):
    path = env.tmp_path / "preferences.txt"
    path.write_text(
        "# topics to tell\n- github\n# never mention\n- health\n# max items per day\n2\n",
        encoding="utf-8",
    )
    send = Recorder()
    await add_task("health", result="Health check result")
    first_model = FakeModel(json_decisions(("task:health:completed", True, "health update")))
    first = JudgmentWatcher(
        config(), model_client=first_model.complete, send=send.send,
        preferences_path=path, clock=lambda: NOW,
    ).run
    result = await first()
    assert result.sent == [] and send.calls == []
    assert '"never_topics": ["health"]' in first_model.requests[0].messages[1].content

    # Edit the file between runs; the next run picks up the new policy.
    path.write_text("# topics to tell\n- github\n# max items per day\n2\n", encoding="utf-8")
    await add_task("github-task", result="GitHub PR is ready")
    second_model = FakeModel(json_decisions(("task:github-task:completed", True, "GitHub PR is ready")))
    second = JudgmentWatcher(
        config(), model_client=second_model.complete, send=send.send,
        preferences_path=path, clock=lambda: NOW,
    )
    result = await second.run()
    assert result.sent == ["task:github-task:completed"]
    assert "GitHub PR is ready" in send.calls[-1]
    assert '"never_topics": []' in second_model.requests[0].messages[1].content


async def test_model_judgment_filters_routine_and_ignores_tool_calls(env):
    await add_task("worthy", result="Deployment finished")
    await add_task("noise", result="Routine heartbeat")
    model = FakeModel(
        json_decisions(
            ("task:worthy:completed", True, "Deployment finished, all checks green"),
            ("task:noise:completed", False, "Heartbeat"),
        ),
        tool_calls=[{"name": "shell", "arguments": "{}"}],
    )
    send = Recorder()
    result = await watcher(env, model, send).run()

    assert result.sent == ["task:worthy:completed"]
    assert len(send.calls) == 1 and "Heartbeat" not in send.calls[0]
    assert model.requests[0].tools == []
    assert await db.fetchone("SELECT COUNT(*) AS n FROM approvals") is not None
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM approvals"))["n"] == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM tasks"))["n"] == 2
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM schedules"))["n"] == 0


async def test_no_worthy_signals_sends_nothing(env):
    await add_task("routine", result="Nothing new")
    model = FakeModel(json_decisions(("task:routine:completed", False, "")))
    send = Recorder()
    result = await watcher(env, model, send).run()
    assert result.sent == [] and send.calls == []


def test_never_topics_match_whole_terms():
    assert mentions_topic("Health check failed", ("health",))
    assert mentions_topic("health-check failed", ("health",))
    assert not mentions_topic("healthy build", ("health",))
    assert not mentions_topic("healthcare update", ("health",))
    assert mentions_topic("C++ build finished", ("C++",))
    assert not mentions_topic("C# build finished", ("C++",))
    assert not mentions_topic("nothing to report", ("",))


async def test_excluded_topic_does_not_hide_other_whole_term_updates(env):
    await add_task("healthy", result="Healthy update published")
    model = FakeModel(json_decisions(("task:healthy:completed", True, "Healthy update published")))
    send = Recorder()
    result = await watcher(
        env, model, send, preferences="# never mention\n- health\n"
    ).run()
    assert result.sent == ["task:healthy:completed"]
    assert send.calls == ["Healthy update published"]


# --- pacing, urgent bypass, delivery -----------------------------------------

async def test_daily_cap_limits_ordinary_items_but_not_urgent(env):
    await add_task("one", result="First update")
    await add_task("two", result="Second update")
    await add_event("task.dead_letter", {"task_id": "x", "error": "crashed"}, event_id="e1")
    model = FakeModel(json_decisions(
        ("task:one:completed", True, "First update"),
        ("task:two:completed", True, "Second update"),
        ("delivery:e1", True, "Delivery failed"),
    ))
    send = Recorder()
    result = await watcher(env, model, send, preferences="# max items per day\n1\n").run()

    assert set(result.sent) == {"delivery:e1", "task:one:completed"}
    assert result.ordinary_sent == 1
    assert len(send.calls) == 2

    await add_task("three", result="Third update")
    second_model = FakeModel(json_decisions(("task:three:completed", True, "Third update")))
    second = await watcher(env, second_model, send, preferences="# max items per day\n1\n").run()
    assert second.sent == [] and len(send.calls) == 2


async def test_delivery_is_retried_once(env):
    await add_task("one", result="First update")
    model = FakeModel(json_decisions(("task:one:completed", True, "First update")))
    send = Recorder(fail_first=1)
    result = await watcher(env, model, send).run()
    assert result.sent == ["task:one:completed"]
    assert send.calls == ["First update", "First update"]


async def test_failed_delivery_is_reported_on_next_successful_contact(env):
    await add_task("one", result="First update")
    first_send = Recorder(fail_always=True)
    model = FakeModel(json_decisions(("task:one:completed", True, "First update")))
    first = await watcher(env, model, first_send).run()
    assert first.sent == [] and first.pending == 1
    state = await load_state()
    assert state.pending[0]["text"] == "First update"

    await add_task("two", result="Second update")
    second_send = Recorder()
    second_model = FakeModel(json_decisions(("task:two:completed", True, "Second update")))
    second = await watcher(env, second_model, second_send).run()
    assert second.sent == ["task:two:completed"]
    assert second.pending_flushed is True
    assert len(second_send.calls) == 2
    assert "Earlier updates could not be delivered" in second_send.calls[0]
    assert "First update" in second_send.calls[0] and "Second update" not in second_send.calls[0]
    assert "Second update" in second_send.calls[1] and "First update" not in second_send.calls[1]
    assert (await load_state()).pending == []


async def test_pending_delivery_is_flushed_even_without_new_signals(env):
    await add_task("one", result="First update")
    model = FakeModel(json_decisions(("task:one:completed", True, "First update")))
    first = await watcher(env, model, Recorder(fail_always=True)).run()
    assert first.sent == [] and first.pending == 1

    # No new signals at all: the carried item still goes out on its own.
    second_send = Recorder()
    second = await watcher(env, FakeModel(), second_send).run()
    assert second.sent == [] and second.pending_flushed is True
    assert len(second_send.calls) == 1
    assert "Earlier updates could not be delivered" in second_send.calls[0]
    assert "First update" in second_send.calls[0]
    assert (await load_state()).pending == []

    # Genuine silence stays silent once the backlog is clear.
    third_send = Recorder()
    third = await watcher(env, FakeModel(), third_send).run()
    assert third.sent == [] and third.pending_flushed is False
    assert third_send.calls == []


async def test_pending_delivery_is_flushed_when_new_signal_is_unworthy(env):
    await add_task("one", result="First update")
    first_model = FakeModel(json_decisions(("task:one:completed", True, "First update")))
    await watcher(env, first_model, Recorder(fail_always=True)).run()

    await add_task("noise", result="Routine heartbeat")
    unworthy = FakeModel(json_decisions(("task:noise:completed", False, "")))
    send = Recorder()
    result = await watcher(env, unworthy, send).run()
    assert result.sent == [] and result.worthy == []
    assert result.pending_flushed is True
    assert len(send.calls) == 1 and "First update" in send.calls[0]


async def test_dead_letter_reaches_user_in_one_run_without_a_model(env):
    await add_event("task.dead_letter", {"task_id": "dead", "error": "crashed"}, event_id="e1")
    send = Recorder()
    result = await watcher(env, None, send).run()
    assert result.sent == ["delivery:e1"]
    assert "crashed" in send.calls[0]


async def test_github_reply_and_deadline_bypass_judgment(env):
    async def fetch(repo):
        first_round = len(fetch.calls) == 0
        comments = 0 if first_round else 2
        due = "" if first_round else "2030-02-01T00:00:00Z"
        fetch.calls.append(None)
        return [{"number": 9, "title": "Fix tests", "comments": comments, "state": "open",
                 "milestone": {"due_on": due}}]
    fetch.calls = []

    send = Recorder()
    first = await watcher(env, None, send, cfg=config(github_repos=["owner/repo"]), fetcher=fetch).run()
    assert first.sent == []  # first sight is ordinary; no model approval

    second = await watcher(env, None, send, cfg=config(github_repos=["owner/repo"]), fetcher=fetch).run()
    assert len(second.sent) == 2
    assert any("Reply arrived" in text for text in send.calls)
    assert any("Deadline moved" in text for text in send.calls)


async def test_app_worker_sends_plain_text_through_existing_channel_delivery(env):
    from leo.core.app import LeoApp

    await add_task("integration", result="Deployment finished")
    model = FakeModel(json_decisions(("task:integration:completed", True, "Deployment finished")))

    class Gateway:
        def __init__(self):
            self.adapters = {"telegram": SimpleNamespace(supports_rooms=False)}
            self.sent = []

        async def send(self, channel, address, text, **kwargs):
            self.sent.append((channel, address, text))
            return {"message_id": "m1"}

    class Engine:
        def __init__(self):
            self.completed = []

        async def complete(self, task_id, result, evidence, note=None):
            self.completed.append((task_id, result, evidence, note))

    async def no_progress(task, status):
        return None

    # This test uses the real clock, so keep the window open in every zone.
    always_awake = env.tmp_path / "always-awake.txt"
    always_awake.write_text("# quiet hours\n00:00-00:00\n", encoding="utf-8")
    app = object.__new__(LeoApp)
    app.settings = Settings()
    app.settings.automation.watcher = config(
        delivery=["telegram:123"], preferences_path=always_awake
    )
    app.router = SimpleNamespace(complete=model.complete)
    app.gateway = Gateway()
    app.task_reporting = SimpleNamespace(on_terminal=no_progress)
    app.engine = Engine()

    task = SimpleNamespace(id="watcher-task", metadata={}, objective="watch")
    payload = await app._watcher_worker(
        task, app.engine, {"schedule_payload": {"worker": "watcher"}}
    )
    assert payload["sent"] == ["task:integration:completed"], payload
    assert app.gateway.sent == [("telegram", "123", "Deployment finished")]
    assert app.engine.completed[0][0] == "watcher-task"
