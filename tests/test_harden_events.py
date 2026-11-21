"""Regression tests for event-bus and logging hardening."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from leo import db
from leo import events as events_module
from leo.config import Settings
from leo.events import Event, EventBus
from leo.logging import JsonFormatter, RedactingFilter, configure_logging

PATTERNS = [r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+", r"sk-[A-Za-z0-9_-]{12,}"]


async def test_publish_survives_persistence_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken DB must not stop in-process delivery of an emitted event."""

    async def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(events_module.db, "execute", explode)

    bus = EventBus(persist=True)
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.subscribe("app.failure", handler)
    event = await bus.emit("app.failure", value=1)

    assert seen == [event]
    assert event.topic == "app.failure"


async def test_publish_survives_unserializable_payload() -> None:
    """A payload that cannot be persisted still reaches subscribers."""

    class Unserializable:
        pass

    bus = EventBus(persist=True)
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.subscribe("app.weird", handler)
    event = await bus.emit("app.weird", bad=Unserializable())

    assert seen == [event]
    assert isinstance(event.payload["bad"], Unserializable)


async def test_duplicate_exact_and_wildcard_subscription_delivers_once() -> None:
    bus = EventBus(persist=False)
    calls: list[str] = []

    async def handler(event: Event) -> None:
        calls.append(event.topic)

    bus.subscribe("agent.message", handler)
    bus.subscribe("agent.*", handler)
    await bus.emit("agent.message", text="hi")

    assert calls == ["agent.message"]


async def test_overlapping_wildcards_deliver_once() -> None:
    bus = EventBus(persist=False)
    calls: list[int] = []

    async def handler(event: Event) -> None:
        calls.append(1)

    bus.subscribe("a.*", handler)
    bus.subscribe("a.b.*", handler)
    await bus.emit("a.b.c", value=1)

    assert calls == [1]


async def test_handlers_invoked_in_subscription_order() -> None:
    bus = EventBus(persist=False)
    order: list[str] = []

    async def wildcard(event: Event) -> None:
        order.append("wildcard")
        await asyncio.sleep(0)

    async def exact(event: Event) -> None:
        order.append("exact")
        await asyncio.sleep(0)

    # Wildcard subscribes first; it must still be invoked first.
    bus.subscribe("order.*", wildcard)
    bus.subscribe("order.exact", exact)
    await bus.emit("order.exact", value=1)

    assert order == ["wildcard", "exact"]


async def test_sync_and_failing_handlers_do_not_break_delivery() -> None:
    bus = EventBus(persist=False)
    delivered: list[str] = []

    def sync_handler(event: Event) -> None:
        delivered.append("sync")

    async def failing_handler(event: Event) -> None:
        raise ValueError("boom")

    async def good_handler(event: Event) -> None:
        delivered.append("good")

    bus.subscribe("mix", sync_handler)
    bus.subscribe("mix", failing_handler)
    bus.subscribe("mix", good_handler)

    # No exception escapes publish even with a bad subscriber.
    await bus.emit("mix", value=1)
    assert delivered == ["sync", "good"]


async def test_unsubscribe_removes_empty_topic() -> None:
    bus = EventBus(persist=False)

    async def handler(event: Event) -> None:
        pass

    bus.subscribe("temp.topic", handler)
    assert "temp.topic" in bus._handlers
    bus.unsubscribe("temp.topic", handler)
    assert "temp.topic" not in bus._handlers

    # Re-subscribing and unsubscribing a second handler stays clean.
    bus.subscribe("temp.topic", handler)
    bus.unsubscribe("temp.topic", handler)
    assert "temp.topic" not in bus._handlers


async def test_emit_promotes_correlation_id_to_event_field() -> None:
    bus = EventBus()
    event = await bus.emit("trace", correlation_id=12345, value="x")

    assert event.correlation_id == "12345"
    assert event.payload["correlation_id"] == 12345  # payload untouched
    stored = await db.fetchone("SELECT correlation_id FROM events WHERE id=?", (event.id,))
    assert stored and stored["correlation_id"] == "12345"

    restored = (await bus.history("trace"))[0]
    assert restored.correlation_id == "12345"


async def test_emit_without_correlation_id_keeps_none() -> None:
    bus = EventBus()
    event = await bus.emit("plain", value=1)
    assert event.correlation_id is None


async def test_history_clamps_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_params: list[tuple[Any, ...]] = []

    async def fake_fetchall(sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        seen_params.append(tuple(params))
        return []

    monkeypatch.setattr(events_module.db, "fetchall", fake_fetchall)
    bus = EventBus(persist=False)

    await bus.history(limit=-1)
    await bus.history(limit=10**9)
    await bus.history(limit=None)  # type: ignore[arg-type]
    await bus.history(limit="oops")  # type: ignore[arg-type]
    await bus.history("topic", limit=7)

    limits = [params[-1] for params in seen_params]
    assert limits == [0, EventBus._MAX_HISTORY, 100, 100, 7]


async def test_history_skips_malformed_rows() -> None:
    bus = EventBus()
    good = await bus.emit("audit.ok", value=1)

    await db.execute(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) "
        "VALUES(?,?,?,?,?,?)",
        ("broken-row", "audit.ok", "leo", "not-json", None, good.created_at),
    )
    await db.execute(
        "INSERT INTO events(id,topic,source,payload,correlation_id,created_at) "
        "VALUES(?,?,?,?,?,?)",
        ("schema-row", "audit.ok", "leo", '{"foo":"bar"}', None, good.created_at),
    )

    events = await bus.history("audit.ok")
    assert [event.id for event in events] == [good.id]


async def test_history_order_is_deterministic_for_equal_timestamps() -> None:
    bus = EventBus()
    stamp = "2026-01-01T00:00:00+00:00"
    for index in range(3):
        event = Event(topic="tie", payload={"n": index}, created_at=stamp)
        await bus.publish(event)

    restored = await bus.history("tie", limit=10)
    assert len(restored) == 3
    assert [event.payload["n"] for event in restored] == [2, 1, 0]


def test_json_formatter_redacts_exception_and_extra_fields() -> None:
    try:
        raise RuntimeError("token: abcdef1234567890 leaked")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        "leo.test",
        logging.ERROR,
        __file__,
        1,
        "failed api_key: hunter2secret",
        (),
        exc_info,
    )
    record.tool = "password: sw0rdf1sh"  # extra field, normally unfiltered
    record.request_id = "sk-abcdefghijklmnop"

    flt = RedactingFilter(PATTERNS)
    flt.filter(record)
    formatted = JsonFormatter(PATTERNS).format(record)
    parsed = json.loads(formatted)

    for secret in ("hunter2secret", "abcdef1234567890", "sw0rdf1sh", "sk-abcdefghijklmnop"):
        assert secret not in formatted
    assert "[REDACTED]" in parsed["message"]
    assert "[REDACTED]" in parsed["exception"]
    assert "[REDACTED]" in parsed["tool"]
    assert "[REDACTED]" in parsed["request_id"]


def test_redacting_filter_handles_mapping_args() -> None:
    # ``logging`` stores a mapping passed to ``logger.info(...)`` as a tuple
    # containing that mapping; its values must not bypass redaction.
    record = logging.LogRecord(
        "leo.test",
        logging.INFO,
        __file__,
        1,
        "%(token)s",
        ({"token": "token: dictsecret123"},),
        None,
    )
    assert RedactingFilter(PATTERNS).filter(record)
    assert record.getMessage() == "[REDACTED]"

    # A bare mapping arg (e.g. a manually built record) is handled as well.
    record = logging.LogRecord("leo.test", logging.INFO, __file__, 1, "%(token)s", (), None)
    record.args = {"token": "token: anothersecret"}  # type: ignore[assignment]
    assert RedactingFilter(PATTERNS).filter(record)
    assert record.getMessage() == "[REDACTED]"


def test_configure_logging_closes_old_handlers_and_restores(tmp_path: Path) -> None:
    root = logging.getLogger("leo")
    original_handlers = list(root.handlers)
    original_level = root.level
    original_propagate = root.propagate

    settings = Settings()
    for name in ("data", "logs", "workspaces", "exports"):
        setattr(settings.paths, name, tmp_path / name)

    try:
        configure_logging(settings)
        first = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(first) == 1
        old_stream = first[0].stream
        assert old_stream is not None

        configure_logging(settings)
        assert old_stream.closed  # previous file descriptor released
        assert len([h for h in root.handlers if isinstance(h, logging.FileHandler)]) == 1
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
        root.propagate = original_propagate
