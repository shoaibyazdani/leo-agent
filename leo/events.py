"""In-process event bus with durable event history."""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from . import db
from .events_outbox import (
    MAX_ATTEMPTS,
    EventOutbox,
    matches_topic,
    normalize_subscriptions,
)
from .logging import get_logger

Handler = Callable[["Event"], Awaitable[None]]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    source: str = "leo"
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    created_at: str = Field(default_factory=utcnow)


class EventPage(BaseModel):
    """One bounded page of durable event history plus its paging tokens."""

    events: list[Event] = Field(default_factory=list)
    total: int = 0
    next_cursor: str | None = None
    prev_cursor: str | None = None
    has_more: bool = False


class EventBus:
    """Deliver events to in-process subscribers, optionally keeping history.

    Persistence is best-effort: a broken/unavailable database must not stop
    subscribers from receiving an event that was already emitted in-process.
    """

    # A negative SQLite LIMIT means "unbounded"; never let a caller dump the
    # entire events table through ``/events?limit=-1`` and friends.
    _MAX_HISTORY = 1000

    def __init__(self, persist: bool = True) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._persist = persist
        self._log = get_logger("events")
        self._outbox = EventOutbox()
        # Durable subscriptions are keyed by a stable operator-supplied name;
        # handlers are reattached after a restart, cursors stay in SQLite.
        self._durable_topics: dict[str, str] = {}
        self._durable_handlers: dict[str, Handler] = {}
        self._draining_depth = 0
        self._outbox_write_failures = 0
        self._durability_errors = 0

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        handlers = self._handlers.get(topic)
        if not handlers:
            return
        if handler in handlers:
            handlers.remove(handler)
        if not handlers:
            # Do not retain empty subscription lists forever: topics are
            # frequently dynamic (task ids, rule names) and ``publish`` walks
            # every registered pattern on each delivery.
            self._handlers.pop(topic, None)

    def _matching_handlers(self, topic: str) -> list[Handler]:
        """Handlers for *topic* in subscription order, delivered once each.

        A handler registered under both an exact topic and a matching wildcard
        pattern used to be invoked twice for the same event. Walking all
        patterns once also makes delivery order follow subscription order
        instead of always putting exact matches first.
        """
        matched: list[Handler] = []
        seen: set[int] = set()
        for pattern, handlers in self._handlers.items():
            wildcard = pattern.endswith(".*") and topic.startswith(pattern[:-1])
            if pattern != topic and not wildcard:
                continue
            for handler in handlers:
                marker = id(handler)
                if marker in seen:
                    continue
                seen.add(marker)
                matched.append(handler)
        return matched

    @staticmethod
    async def _invoke(handler: Handler, event: Event) -> None:
        """Tolerate a mistakenly synchronous subscriber without losing the event."""
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _serialize(event: Event) -> str:
        """Serialize an event for durable storage, tolerating odd payloads."""
        try:
            return event.model_dump_json()
        except Exception:  # noqa: BLE001 - unserializable values must not lose the event
            return json.dumps(event.model_dump(), default=str, ensure_ascii=False)

    async def _persist_event(self, event: Event) -> int:
        """Write history, falling back to the outbox on a transient DB failure.

        Returns the events.rowid sequence (0 when the row is not in history).
        In-process delivery continues even when neither write succeeds; the
        failure is logged and counted, never silently swallowed.
        """
        if not self._persist:
            return 0
        payload = self._serialize(event)
        try:
            await db.execute(
                "INSERT OR IGNORE INTO events(id,topic,source,payload,correlation_id,created_at) VALUES(?,?,?,?,?,?)",
                (event.id, event.topic, event.source, payload, event.correlation_id, event.created_at),
            )
            row = await db.fetchone("SELECT rowid AS seq FROM events WHERE id=?", (event.id,))
            return int(row["seq"]) if row else 0
        except Exception as exc:  # noqa: BLE001 - in-process delivery must survive
            self._log.warning(
                "event persistence failed; queuing in outbox",
                extra={"event": event.topic, "error": str(exc)},
            )
            try:
                await self._outbox.enqueue_event(
                    event.id, event.topic, event.source, payload,
                    event.correlation_id, event.created_at, f"{type(exc).__name__}: {exc}",
                )
            except Exception as outbox_exc:  # noqa: BLE001 - count a total durability failure
                self._outbox_write_failures += 1
                self._durability_errors += 1
                self._log.warning(
                    "event outbox write failed",
                    extra={"event": event.topic, "error": str(outbox_exc)},
                )
            return 0

    async def _queue_durable(self, event: Event, seq: int) -> int:
        """Queue one durable delivery per matching registered subscription."""
        if not self._persist:
            return 0
        payload = self._serialize(event)
        try:
            rows = await self._outbox.list_subscriptions(enabled_only=True)
        except Exception as exc:  # noqa: BLE001 - durability is best-effort at emit time
            self._durability_errors += 1
            self._log.warning("event subscription lookup failed", extra={"error": str(exc)})
            return 0
        queued = 0
        for row in rows:
            if not matches_topic(row["topic"], event.topic):
                continue
            try:
                queued += int(
                    await self._outbox.enqueue_delivery(
                        row["name"], event.id, event.topic, payload, seq, event.created_at
                    )
                )
            except Exception as exc:  # noqa: BLE001 - other subscriptions still proceed
                self._durability_errors += 1
                self._log.warning(
                    "event delivery queue failed",
                    extra={"event": event.topic, "subscription": row["name"], "error": str(exc)},
                )
        return queued

    async def publish(self, event: Event) -> Event:
        seq = await self._persist_event(event)
        await self._queue_durable(event, seq)

        extra: dict[str, Any] = {"event": event.topic}
        if event.correlation_id is not None:
            extra["correlation_id"] = event.correlation_id
        self._log.debug("event published", extra=extra)

        matched = self._matching_handlers(event.topic)
        results = await asyncio.gather(
            *(self._invoke(handler, event) for handler in matched),
            return_exceptions=True,
        )
        for handler, result in zip(matched, results):
            if isinstance(result, Exception):
                self._log.warning(
                    "event handler failed",
                    extra={"event": event.topic, "error": str(result)},
                )
        if self._persist and not self._draining_depth:
            # Durable subscribers are recorded and attempted from their durable
            # row, so a handler error has a retryable state instead of only a log.
            # The depth guard stops a handler that emits another event from
            # recursing through immediate attempts; the next drain handles it.
            self._draining_depth += 1
            try:
                for name, topic in list(self._durable_topics.items()):
                    handler = self._durable_handlers.get(name)
                    if handler is not None and matches_topic(topic, event.topic):
                        await self._attempt_delivery(name, event.id, handler)
            finally:
                self._draining_depth -= 1
        return event

    async def emit(self, topic: str, **payload: Any) -> Event:
        """Publish a payload as an event.

        A ``correlation_id`` in the payload is also promoted to the Event's
        first-class correlation field so tracing/history do not have to dig it
        out of the payload. The payload itself is left untouched for existing
        subscribers that read it there.
        """
        raw_correlation_id = payload.get("correlation_id")
        correlation_id = (
            str(raw_correlation_id) if raw_correlation_id is not None else None
        )
        return await self.publish(
            Event(topic=topic, payload=payload, correlation_id=correlation_id)
        )

    async def history(self, topic: str | None = None, limit: int = 100) -> list[Event]:
        try:
            bounded = int(limit)
        except (TypeError, ValueError):
            bounded = 100
        bounded = max(0, min(bounded, self._MAX_HISTORY))

        sql = "SELECT payload FROM events"
        params: list[Any] = []
        if topic:
            sql += " WHERE topic=?"
            params.append(topic)
        # rowid is a stable tie-breaker for events created in the same tick.
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(bounded)

        rows = await db.fetchall(sql, params)
        events: list[Event] = []
        for row in rows:
            try:
                events.append(Event.model_validate_json(row["payload"]))
            except Exception as exc:  # noqa: BLE001 - one bad row must not hide the rest
                self._log.warning(
                    "skipping malformed event history row",
                    extra={"error": str(exc)},
                )
        return events

    # ------------------------------------------------------------------
    # Keyset pagination over durable history
    # ------------------------------------------------------------------
    @staticmethod
    def encode_cursor(created_at: str, seq: int) -> str:
        """Encode a ``(created_at, rowid)`` keyset as an opaque URL-safe token."""
        raw = json.dumps({"t": created_at, "s": int(seq)}, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def decode_cursor(cursor: str) -> tuple[str, int]:
        """Decode a cursor, raising ``ValueError`` for malformed/foreign tokens."""
        if not isinstance(cursor, str) or not cursor or len(cursor) > 512:
            raise ValueError("malformed event cursor")
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            raw = base64.b64decode(padded, altchars=b"-_", validate=True)
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or set(value) != {"t", "s"}:
                raise ValueError("malformed event cursor")
            created_at, seq = value["t"], value["s"]
            if not isinstance(created_at, str) or not created_at:
                raise ValueError("malformed event cursor")
            if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
                raise ValueError("malformed event cursor")
            return created_at, seq
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("malformed event cursor") from exc

    @staticmethod
    def _page_filters(
        topic: str | None,
        topic_prefix: str | None,
        source: str | None,
        correlation_id: str | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if topic:
            clauses.append("topic=?")
            params.append(topic)
        if topic_prefix:
            clauses.append("topic LIKE ? ESCAPE '\\'")
            params.append(topic_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if source:
            clauses.append("source=?")
            params.append(source)
        if correlation_id is not None:
            clauses.append("correlation_id=?")
            params.append(correlation_id)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    async def count(
        self,
        topic: str | None = None,
        topic_prefix: str | None = None,
        source: str | None = None,
        correlation_id: str | None = None,
    ) -> int:
        """Count durable events matching the filters (independent of any cursor)."""
        where, params = self._page_filters(topic, topic_prefix, source, correlation_id)
        row = await db.fetchone("SELECT COUNT(*) AS n FROM events" + where, params)
        return int(row["n"]) if row else 0

    async def page(
        self,
        topic: str | None = None,
        topic_prefix: str | None = None,
        source: str | None = None,
        correlation_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        direction: str = "before",
    ) -> EventPage:
        """Return one bounded, keyset-paginated page of durable events.

        ``before`` walks newest-first (for "load older"); ``after`` walks
        oldest-first. Ordering is by ``(created_at, rowid)`` so events created in
        the same instant keep insertion order. The total is computed over the
        filters, never over the page, and cursors are opaque and validated.
        """
        if direction not in ("before", "after"):
            raise ValueError("direction must be 'before' or 'after'")
        try:
            bounded = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer") from exc
        bounded = max(1, min(bounded, self._MAX_HISTORY))
        where, params = self._page_filters(topic, topic_prefix, source, correlation_id)
        keyset = self.decode_cursor(cursor) if cursor is not None else None
        sql = "SELECT rowid AS seq, created_at, payload FROM events" + where
        if keyset is not None:
            comparator = "<" if direction == "before" else ">"
            sql += (" WHERE " if not where else " AND ")
            sql += f"(created_at {comparator} ? OR (created_at = ? AND rowid {comparator} ?))"
            params.extend([keyset[0], keyset[0], keyset[1]])
        sql += (
            " ORDER BY created_at DESC, rowid DESC LIMIT ?"
            if direction == "before"
            else " ORDER BY created_at ASC, rowid ASC LIMIT ?"
        )
        params.append(bounded + 1)
        rows = await db.fetchall(sql, params)
        has_more = len(rows) > bounded
        rows = rows[:bounded]
        events: list[Event] = []
        for row in rows:
            try:
                events.append(Event.model_validate_json(row["payload"]))
            except Exception as exc:  # noqa: BLE001 - one bad row must not hide the rest
                self._log.warning(
                    "skipping malformed event history row",
                    extra={"error": str(exc)},
                )
        total = await self.count(topic, topic_prefix, source, correlation_id)
        return EventPage(
            events=events,
            total=total,
            next_cursor=self.encode_cursor(rows[-1]["created_at"], rows[-1]["seq"]) if has_more and rows else None,
            prev_cursor=self.encode_cursor(rows[0]["created_at"], rows[0]["seq"]) if rows else None,
            has_more=has_more,
        )

    async def wait_for(
        self,
        topic: str,
        timeout: float = 30.0,
        predicate: Callable[[Event], bool] | None = None,
    ) -> Event:
        """Wait for one matching event. Useful for tests and coordination."""
        future: asyncio.Future[Event] = asyncio.get_running_loop().create_future()

        async def handler(event: Event) -> None:
            if predicate is None or predicate(event):
                if not future.done():
                    future.set_result(event)

        self.subscribe(topic, handler)
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self.unsubscribe(topic, handler)

    # ------------------------------------------------------------------
    # Durable subscriptions, outbox flush and delivery drain
    # ------------------------------------------------------------------
    async def subscribe_durable(
        self, name: str, topic: str, handler: Handler | None = None, *, replay: bool = True
    ) -> Handler | None:
        """Attach a named, restart-safe subscriber and register it in SQLite.

        The name is the durable identity: after a restart the application calls
        this again with the same name and a new handler; the persisted cursor
        causes events emitted while the process was down to be replayed once.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("durable subscription name is required")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("durable subscription topic is required")
        if handler is not None and not callable(handler):
            raise TypeError("handler must be callable")
        self._durable_topics[name] = topic
        if handler is None:
            self._durable_handlers.pop(name, None)
        else:
            self._durable_handlers[name] = handler
        if self._persist:
            try:
                await self._outbox.register_subscription(name, topic)
            except Exception as exc:  # noqa: BLE001 - in-process registration still works
                self._durability_errors += 1
                self._log.warning("durable subscription not persisted", extra={"error": str(exc)})
            if replay:
                await self.replay_subscriptions(names=(name,))
        return handler

    def unsubscribe_durable(self, name: str) -> None:
        """Detach the local handler without discarding its durable cursor."""
        self._durable_handlers.pop(name, None)
        self._durable_topics.pop(name, None)

    async def delete_durable_subscription(self, name: str) -> None:
        """Remove a durable registration and all of its delivery records."""
        self.unsubscribe_durable(name)
        if not self._persist:
            return
        try:
            await self._outbox.delete_subscription(name)
        except Exception as exc:  # noqa: BLE001 - best effort, logged explicitly
            self._durability_errors += 1
            self._log.warning("durable subscription removal failed", extra={"error": str(exc)})

    async def set_subscription_enabled(self, name: str, enabled: bool) -> None:
        if self._persist:
            await self._outbox.set_subscription_enabled(name, enabled)

    async def durable_subscriptions(self) -> list[dict[str, Any]]:
        if not self._persist:
            return [
                {"name": name, "topic": topic, "enabled": True,
                 "cursor_created_at": None, "cursor_seq": None}
                for name, topic in self._durable_topics.items()
            ]
        try:
            return normalize_subscriptions(await self._outbox.list_subscriptions())
        except Exception as exc:  # noqa: BLE001
            self._durability_errors += 1
            self._log.warning("subscription listing failed", extra={"error": str(exc)})
            return []

    async def _attempt_delivery(
        self, name: str, event_id: str, handler: Handler,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> str:
        """Claim and run one durable delivery, persisting its terminal state."""
        try:
            if not await self._outbox.claim_delivery(name, event_id, max_attempts):
                return "skipped"  # already delivered, in flight or exhausted
            row = await db.fetchone(
                "SELECT * FROM event_deliveries WHERE subscription=? AND event_id=?",
                (name, event_id),
            )
        except Exception as exc:  # noqa: BLE001
            self._durability_errors += 1
            self._log.warning("delivery claim failed", extra={"error": str(exc)})
            return "storage_error"
        if row is None:
            return "missing"
        attempts = int(row["attempts"]) + 1
        error: str | None = None
        try:
            await self._invoke(handler, Event.model_validate_json(row["payload"]))
        except Exception as exc:  # noqa: BLE001 - failures are durable state, not just logs
            error = f"{type(exc).__name__}: {exc}"
            status = "failed" if attempts >= max_attempts else "pending"
            self._log.warning(
                "event delivery failed",
                extra={"event": row["topic"], "subscription": name, "error": error},
            )
        else:
            status = "delivered"
        try:
            await self._outbox.mark_delivery(
                name, event_id, status=status, attempts=attempts, error=error
            )
        except Exception as exc:  # noqa: BLE001
            self._durability_errors += 1
            self._log.warning("delivery status write failed", extra={"error": str(exc)})
            return "storage_error"
        if status == "delivered" and int(row["seq"] or 0) > 0:
            try:
                await self._outbox.advance_cursor(name, row["created_at"], int(row["seq"]))
            except Exception as exc:  # noqa: BLE001 - dedupe key still prevents duplicates
                self._durability_errors += 1
                self._log.warning("subscription cursor write failed", extra={"error": str(exc)})
        return status

    async def replay_subscriptions(
        self, *, names: tuple[str, ...] | None = None, limit: int = 100
    ) -> dict[str, int]:
        """Queue and attempt persisted events after each subscription cursor."""
        stats = {"subscriptions": 0, "enqueued": 0, "delivered": 0, "pending": 0,
                 "failed": 0, "skipped": 0, "no_handler": 0, "errors": 0}
        if not self._persist:
            return stats
        self._draining_depth += 1
        try:
            try:
                rows = await self._outbox.list_subscriptions(enabled_only=True)
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                self._log.warning("replay listing failed", extra={"error": str(exc)})
                return stats
            for row in rows:
                name = str(row["name"])
                if names is not None and name not in names:
                    continue
                handler = self._durable_handlers.get(name)
                if handler is None:
                    stats["no_handler"] += 1
                    continue
                stats["subscriptions"] += 1
                pattern = str(row["topic"])
                sql = "SELECT rowid AS seq, id, topic, payload, created_at FROM events WHERE 1=1"
                params: list[Any] = []
                if pattern.endswith(".*"):
                    prefix = pattern[:-1].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    sql += " AND topic LIKE ? ESCAPE '\\'"
                    params.append(prefix + "%")
                else:
                    sql += " AND topic=?"
                    params.append(pattern)
                if row["cursor_seq"] is not None:
                    cursor_at = str(row["cursor_created_at"])
                    sql += " AND (created_at > ? OR (created_at = ? AND rowid > ?))"
                    params.extend([cursor_at, cursor_at, int(row["cursor_seq"])])
                sql += " ORDER BY created_at ASC, rowid ASC LIMIT ?"
                params.append(max(1, min(int(limit), self._MAX_HISTORY)))
                try:
                    events = await db.fetchall(sql, params)
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    self._log.warning("replay query failed", extra={"error": str(exc)})
                    continue
                for item in events:
                    try:
                        stats["enqueued"] += int(
                            await self._outbox.enqueue_delivery(
                                name, str(item["id"]), str(item["topic"]),
                                str(item["payload"]), int(item["seq"]), str(item["created_at"]),
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats["errors"] += 1
                        self._log.warning("replay enqueue failed", extra={"error": str(exc)})
                        continue
                    status = await self._attempt_delivery(name, str(item["id"]), handler)
                    stats[status] = stats.get(status, 0) + 1
        finally:
            self._draining_depth -= 1
        return stats

    async def drain_outbox(
        self, *, limit: int = 50, max_attempts: int = MAX_ATTEMPTS
    ) -> dict[str, int]:
        """Flush outboxed events into durable history with bounded retries."""
        stats = {"claimed": 0, "flushed": 0, "retryable": 0, "failed": 0, "errors": 0}
        if not self._persist:
            return stats
        self._draining_depth += 1
        try:
            try:
                rows = await self._outbox.claim_events(max(1, min(int(limit), 500)))
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                self._log.warning("outbox claim failed", extra={"error": str(exc)})
                return stats
            stats["claimed"] = len(rows)
            for row in rows:
                try:
                    await db.execute(
                        "INSERT OR IGNORE INTO events(id,topic,source,payload,correlation_id,created_at) "
                        "VALUES(?,?,?,?,?,?)",
                        (row["id"], row["topic"], row["source"], row["payload"],
                         row["correlation_id"], row["created_at"]),
                    )
                    seq_row = await db.fetchone(
                        "SELECT rowid AS seq FROM events WHERE id=?", (row["id"],)
                    )
                    seq = int(seq_row["seq"]) if seq_row else 0
                    await self._outbox.mark_event(row["id"], "processed", seq=seq)
                    if seq > 0:
                        for delivery in await self._outbox.delivered_without_seq(row["id"], seq):
                            name = str(delivery["subscription"])
                            await self._outbox.update_delivery_seq(row["id"], seq)
                            await self._outbox.advance_cursor(name, row["created_at"], seq)
                    stats["flushed"] += 1
                except Exception as exc:  # noqa: BLE001 - keep the row retryable
                    attempts = int(row["attempts"]) + 1
                    status = "failed" if attempts >= max_attempts else "pending"
                    try:
                        await self._outbox.mark_event(
                            row["id"], status, error=f"{type(exc).__name__}: {exc}",
                            attempts=attempts,
                        )
                    except Exception as mark_exc:  # noqa: BLE001
                        stats["errors"] += 1
                        self._log.warning("outbox status write failed", extra={"error": str(mark_exc)})
                    stats["failed" if status == "failed" else "retryable"] += 1
        finally:
            self._draining_depth -= 1
        return stats

    async def drain_deliveries(
        self, *, limit: int = 20, subscription: str | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> dict[str, int]:
        """Retry pending durable deliveries; failures dead-letter after the cap."""
        stats: dict[str, int] = {"claimed": 0, "no_handler": 0, "errors": 0}
        if not self._persist:
            return stats
        self._draining_depth += 1
        try:
            try:
                rows = await self._outbox.pending_deliveries(
                    subscription, max(1, min(int(limit), 500))
                )
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                self._log.warning("delivery listing failed", extra={"error": str(exc)})
                return stats
            stats["claimed"] = len(rows)
            for row in rows:
                handler = self._durable_handlers.get(str(row["subscription"]))
                if handler is None:
                    stats["no_handler"] += 1
                    continue
                status = await self._attempt_delivery(
                    str(row["subscription"]), str(row["event_id"]), handler, max_attempts
                )
                stats[status] = stats.get(status, 0) + 1
        finally:
            self._draining_depth -= 1
        return stats

    async def drain(
        self, *, limit: int = 50, max_attempts: int = MAX_ATTEMPTS
    ) -> dict[str, Any]:
        """Single public hook for the app loop: flush, replay, then retry."""
        return {
            "outbox": await self.drain_outbox(limit=limit, max_attempts=max_attempts),
            "replay": await self.replay_subscriptions(limit=limit),
            "deliveries": await self.drain_deliveries(limit=limit, max_attempts=max_attempts),
        }

    async def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "outbox_write_failures": self._outbox_write_failures,
            "durability_errors": self._durability_errors,
            "attached_subscriptions": len(self._durable_handlers),
        }
        if self._persist:
            try:
                result.update(await self._outbox.stats())
            except Exception as exc:  # noqa: BLE001
                result["stats_error"] = str(exc)
        return result
