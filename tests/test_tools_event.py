"""Observability tools: real durable rows, bounded pages and emitted counters.

Every call goes through ToolExecutorImpl, so permission/audit/event plumbing is
exercised as well as the handlers themselves.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools.event_tools import TOOL_NAMES, register
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    assert register(registry) == list(TOOL_NAMES)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        registry=registry, executor=executor,
    )


async def run(env, name: str, **arguments):
    return await env.executor.execute(ToolCall(name=name, arguments=arguments))


EVENTS = [
    ("evt-1", "seed.event", "test", {"n": 1}, "corr-1", "2026-01-01T00:00:01+00:00"),
    ("evt-2", "seed.event", "test", {"n": 2}, "corr-2", "2026-01-01T00:00:02+00:00"),
    ("evt-3", "seed.event", "test", {"n": 3}, "corr-1", "2026-01-01T00:00:03+00:00"),
    ("evt-x", "seed.other", "test", {"n": 9}, None, "2026-01-01T00:00:04+00:00"),
]


async def seed_events() -> None:
    await db.executemany(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) VALUES(?,?,?,?,?,?)",
        [(i, t, s, json.dumps(p), c, ts) for i, t, s, p, c, ts in EVENTS],
    )


AUDIT = [
    ("2026-01-01T00:00:01+00:00", "alice", "seed.audit", "res-a", "low", "allowed", "ok", '{"k":1}', "corr-a"),
    ("2026-01-01T00:00:02+00:00", "bob", "seed.audit", "res-b", "medium", "denied", "blocked", "{}", "corr-b"),
    ("2026-01-01T00:00:03+00:00", "alice", "seed.audit", "res-c", "low", "denied", "blocked2", '{"k":3}', "corr-a"),
    ("2026-01-01T00:00:04+00:00", "eve", "seed.other", "res-d", "low", "allowed", "ok", "{}", None),
]


async def seed_audit() -> None:
    await db.executemany(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,correlation_id) "
        "VALUES(?,?,?,?,?,?,?,?,?)", AUDIT,
    )


def event_dict(row) -> dict:
    return {
        "id": row[0], "topic": row[1], "source": row[2],
        "payload": row[3], "correlation_id": row[4], "created_at": row[5],
    }


def audit_dict(id_: int, row) -> dict:
    return {
        "id": id_, "ts": row[0], "actor": row[1], "action": row[2],
        "resource": row[3], "risk": row[4], "decision": row[5],
        "reason": row[6], "metadata": json.loads(row[7]), "correlation_id": row[8],
    }


def test_register_typed_specs_and_idempotency(env) -> None:
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert set(specs) == set(TOOL_NAMES)
    for name in TOOL_NAMES:
        spec = specs[name]
        assert spec.category == "observability"
        assert spec.risk is (RiskLevel.MEDIUM if name == "metrics_snapshot" else RiskLevel.LOW)
        assert spec.permissions and all(spec.permissions)
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""
    assert specs["event_query"].input_schema["properties"]["limit"]["maximum"] == 100
    assert specs["audit_query"].input_schema["properties"]["offset"]["minimum"] == 0
    assert specs["metrics_snapshot"].input_schema["properties"]["topic_limit"]["maximum"] == 100
    assert register(env.registry) == []


async def test_event_query_exact_results_and_cursor(env) -> None:
    await seed_events()
    first = await run(env, "event_query", topic="seed.event", limit=2)
    assert first.success and first.decision == "allowed"
    assert first.output == {
        "success": True,
        "events": [event_dict(EVENTS[2]), event_dict(EVENTS[1])],
        "total": 3, "limit": 2, "offset": 0, "next_offset": 2,
    }
    second = await run(env, "event_query", topic="seed.event", limit=2, offset=2)
    assert second.output == {
        "success": True,
        "events": [event_dict(EVENTS[0])],
        "total": 3, "limit": 2, "offset": 2, "next_offset": None,
    }
    correlated = await run(env, "event_query", correlation_id="corr-2")
    assert correlated.output["events"] == [event_dict(EVENTS[1])]
    assert correlated.output["total"] == 1
    past_end = await run(env, "event_query", topic="seed.event", offset=99)
    assert past_end.output["events"] == []
    assert past_end.output["next_offset"] is None


async def test_audit_query_exact_results_and_cursor(env) -> None:
    await seed_audit()
    first = await run(env, "audit_query", action="seed.audit", limit=2)
    assert first.success
    assert first.output == {
        "success": True,
        "entries": [audit_dict(3, AUDIT[2]), audit_dict(2, AUDIT[1])],
        "total": 3, "limit": 2, "offset": 0, "next_offset": 2,
    }
    second = await run(env, "audit_query", action="seed.audit", limit=2, offset=2)
    assert second.output["entries"] == [audit_dict(1, AUDIT[0])]
    assert second.output["next_offset"] is None
    filtered = await run(env, "audit_query", action="seed.audit", decision="denied", risk="low")
    assert filtered.output["entries"] == [audit_dict(3, AUDIT[2])]
    assert filtered.output["total"] == 1
    actor = await run(env, "audit_query", actor="bob")
    assert actor.output["entries"] == [audit_dict(2, AUDIT[1])]


async def test_limits_are_enforced_and_handlers_stay_bounded(env) -> None:
    await seed_events()
    for name, arguments, fragment in (
        ("event_query", {"limit": 0}, "must be >= 1"),
        ("event_query", {"limit": 101}, "must be <= 100"),
        ("event_query", {"offset": -1}, "must be >= 0"),
        ("audit_query", {"limit": 0}, "must be >= 1"),
        ("metrics_snapshot", {"topic_limit": 101}, "must be <= 100"),
    ):
        result = await run(env, name, **arguments)
        assert not result.success, arguments
        assert result.decision == "invalid_arguments"
        assert fragment in result.error
        assert result.output is None
    handler = env.registry.get("event_query").handler
    direct = await handler({"limit": 100000}, {"events": env.events})
    assert direct["limit"] == 100 and len(direct["events"]) <= 100
    assert {row[0] for row in EVENTS} <= {event["id"] for event in direct["events"]}


async def test_metrics_snapshot_emits_real_counters(env) -> None:
    await seed_events()
    await seed_audit()
    await db.executemany(
        "INSERT INTO tool_calls(id,tool,actor,args,risk,decision,success,duration_ms,error,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        [
            ("c-1", "read_file", "alice", "{}", "low", "allowed", 1, 1.0, None, "2026-01-01T00:00:01+00:00"),
            ("c-2", "terminal", "bob", "{}", "medium", "allowed", 1, 2.0, None, "2026-01-01T00:00:02+00:00"),
            ("c-3", "http_get", "eve", "{}", "medium", "error", 0, 3.0, "boom", "2026-01-01T00:00:03+00:00"),
        ],
    )
    result = await run(env, "metrics_snapshot")
    assert result.success and result.decision == "allowed"
    output = result.output
    assert output["success"] is True
    assert isinstance(output["generated_at"], str)
    # The snapshot is the first tool call here: only its permission audit row
    # is added before counters are computed.
    assert output["counters"] == {
        "events_total": 4, "audit_total": 5, "audit_denied": 2,
        "tool_calls_total": 3, "tool_calls_failed": 1,
    }
    assert output["top_topics"] == [
        {"topic": "seed.event", "count": 3}, {"topic": "seed.other", "count": 1},
    ]
    assert output["decisions"] == [
        {"decision": "allowed", "count": 3}, {"decision": "denied", "count": 2},
    ]
    assert output["metrics_written"] == 5

    rows = await db.fetchall("SELECT name,value,labels,ts FROM metrics ORDER BY name")
    assert [(row["name"], row["value"]) for row in rows] == [
        ("observability.audit.denied", 2.0),
        ("observability.audit.total", 5.0),
        ("observability.events.total", 4.0),
        ("observability.tool_calls.failed", 1.0),
        ("observability.tool_calls.total", 3.0),
    ]
    assert all(
        json.loads(row["labels"]) == {"source": "event_tools", "tool": "metrics_snapshot"}
        and row["ts"] == output["generated_at"] for row in rows
    )
    snapshots = await db.fetchall(
        "SELECT topic,source,payload FROM events WHERE topic='observability.snapshot'"
    )
    assert len(snapshots) == 1 and snapshots[0]["source"] == "observability"
    # The events.payload column stores the whole serialized Event row.
    stored_event = json.loads(snapshots[0]["payload"])
    assert stored_event["topic"] == "observability.snapshot"
    payload = stored_event["payload"]
    assert payload["counters"] == output["counters"]
    assert payload["top_topics"] == output["top_topics"]


async def test_metrics_snapshot_topic_limit(env) -> None:
    await seed_events()
    result = await run(env, "metrics_snapshot", topic_limit=1)
    assert result.success
    assert result.output["top_topics"] == [{"topic": "seed.event", "count": 3}]


async def test_metrics_snapshot_uses_injected_event_service(env, monkeypatch) -> None:
    class Recorder:
        def __init__(self):
            self.events = []

        async def publish(self, event):
            self.events.append(event)
            return event

    recorder = Recorder()
    original = ToolExecutorImpl._context

    def with_services(self, call, task_key, dry_run):
        ctx = original(self, call, task_key, dry_run)
        ctx["services"] = {"events": recorder}
        return ctx

    monkeypatch.setattr(ToolExecutorImpl, "_context", with_services)
    result = await run(env, "metrics_snapshot")
    assert result.success
    assert len(recorder.events) == 1
    assert recorder.events[0].topic == "observability.snapshot"
    assert recorder.events[0].payload["counters"] == result.output["counters"]
    stored = await db.fetchall("SELECT id FROM events WHERE topic='observability.snapshot'")
    assert stored == []


async def test_metrics_snapshot_requires_event_service(env, monkeypatch) -> None:
    original = ToolExecutorImpl._context

    def without_events(self, call, task_key, dry_run):
        ctx = original(self, call, task_key, dry_run)
        ctx.pop("events", None)
        ctx["services"] = {}
        return ctx

    monkeypatch.setattr(ToolExecutorImpl, "_context", without_events)
    result = await run(env, "metrics_snapshot")
    assert not result.success
    assert result.decision == "error"
    assert "requires an events service" in result.error
    metrics = await db.fetchall("SELECT id FROM metrics")
    assert metrics == []
