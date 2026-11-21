"""Channel-agnostic messaging gateway.

Inbound messages from any adapter are normalized, stored in the shared
``conversations``/``messages`` tables and published to the durable
``gateway.message`` event topic. Outbound messages fan out through the adapter
registered for each channel and require explicit permission (external messages
are high risk). Sessions are isolated by channel plus subject; history can be
queried across channels, subject to permission checks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping, NamedTuple

from .. import db
from ..config import Settings
from ..contracts import IntegrationRecord, PermissionManager, RiskLevel, new_id, utcnow
from ..events import Event, EventBus
from ..security.permissions import PermissionManagerImpl, arguments_digest
from .admission import ChannelAdmission
from .messaging import (
    MessagingError,
    MessagingAdapter,
    NormalizedMessage,
    implements_polling,
    implements_webhook,
)
from .transport_state import TransportState

GATEWAY_MESSAGE_TOPIC = "gateway.message"
GATEWAY_SENT_TOPIC = "gateway.sent"
GATEWAY_OUTBOX_TOPIC = "gateway.outbox"
GATEWAY_OUTBOX_TABLE = "gateway_outbox"
_MESSAGE_DEDUPE_PREFIX = "gateway:message:"
# Delivery markers survive a failed bookkeeping transaction: if the adapter
# accepted a send but the outbox row could not be marked processed, the next
# drain sees the marker and suppresses the duplicate external side effect.
_OUTBOX_DELIVERY_PREFIX = "gateway:outbox:delivery:"
_SEND_IDEMPOTENCY_PREFIX = "gateway:outbox:send:"

_OUTBOX_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS gateway_outbox ("
    "id TEXT PRIMARY KEY, channel TEXT NOT NULL, subject TEXT, conversation_id TEXT NOT NULL, "
    "source_message_id TEXT, sender_id TEXT, sender TEXT, reply_ref TEXT, reply_target TEXT, "
    "text TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', "
    "attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
    "last_error TEXT, processed_at TEXT, last_result TEXT)",
    "CREATE INDEX IF NOT EXISTS idx_gateway_outbox_status ON gateway_outbox(status, created_at, id)",
)

# An application-level subscriber receives this payload, runs one agent turn
# and sends the reply through ``Gateway.send``. The record is durable, so a
# restart never loses an admitted message that had no live subscriber.
OutboxHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class _Prepared(NamedTuple):
    message: NormalizedMessage
    original_channel: str | None
    subject: str
    identity: str
    dedupe_key: str
    attachment_ids: tuple[str, ...] = ()

# Webhook signatures cover only the request body (or body+timestamp for
# Discord), so a captured delivery could otherwise be replayed forever. A
# durable body fingerprint suppresses redeliveries inside this window.
WEBHOOK_REPLAY_WINDOW_SECONDS = 300.0
_WEBHOOK_DEDUPE_PREFIX = "gateway:webhook:delivery:"


class GatewayError(RuntimeError):
    """Gateway operation failed."""


class GatewayAdmissionError(GatewayError):
    """Inbound message was rejected by the central admission policy."""


class GatewayReplayError(GatewayError):
    """Signed webhook delivery was already processed inside the replay window."""


class GatewayPermissionError(GatewayError):
    """Gateway operation was denied by the permission manager."""

    def __init__(self, reason: str, approval_id: str | None = None) -> None:
        self.approval_id = approval_id
        super().__init__(reason)


def _dump(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _delivery_confirmed(result: Any) -> bool:
    """Only an explicit success may mark an outbox record processed."""
    if result is None or result is False:
        return False
    if isinstance(result, Mapping):
        if result.get("success") is False or result.get("delivered") is False:
            return False
        if str(result.get("status", "")).lower() in {"failed", "error", "incomplete", "cancelled"}:
            return False
    return True


class Gateway:
    """Unified conversation store and inbound/outbound channel router."""

    def __init__(
        self,
        adapters: Mapping[str, MessagingAdapter] | None = None,
        *,
        events: EventBus | None = None,
        permissions: PermissionManager | None = None,
        actor: str = "gateway",
        admission: ChannelAdmission | None = None,
        webhook_replay_window: float | None = WEBHOOK_REPLAY_WINDOW_SECONDS,
    ) -> None:
        self.adapters: dict[str, MessagingAdapter] = dict(adapters or {})
        self.attachment_sink: Callable[[str, dict[str, Any]], Awaitable[str | None]] | None = None
        self.events = events or EventBus()
        self.permissions = permissions or PermissionManagerImpl(Settings.load())
        self.actor = actor
        # Central, fail-closed inbound policy. One resolver, no per-adapter
        # authorization: adapters supply identity facts, this decides.
        self.admission = admission or ChannelAdmission()
        # None/0 disables dedupe (useful for explicit replay-style integrations).
        self.webhook_replay_window = webhook_replay_window
        # Optional app-level hook used by drain_outbox(). Set from the
        # composition root after the agent engine exists.
        self._reply_handler: OutboxHandler | None = None
        if getattr(self.admission, "events", None) is None:
            self.admission.events = self.events

    def set_attachment_sink(
        self, sink: Callable[[str, dict[str, Any]], Awaitable[str | None]] | None
    ) -> None:
        """Register the store that turns a channel file descriptor into an id.

        The adapter supplies a provider file reference; only the app owns the
        attachment store, so the gateway routes bytes through this hook and
        never keeps media itself.
        """
        self.attachment_sink = sink

    def set_reply_handler(self, handler: OutboxHandler | None) -> None:
        """Register (or clear with ``None``) the durable outbox consumer hook."""
        self._reply_handler = handler

    def register(self, adapter: MessagingAdapter) -> None:
        self.adapters[adapter.channel] = adapter

    def _adapter(self, channel: str) -> MessagingAdapter:
        adapter = self.adapters.get(channel)
        if adapter is None:
            raise GatewayError(f"no adapter registered for channel: {channel}")
        return adapter

    async def _authorize(
        self,
        action: str,
        risk: RiskLevel,
        *,
        resource: str | None = None,
        **context: Any,
    ) -> None:
        decision = await self.permissions.authorize(self.actor, action, resource, risk, **context)
        if not decision.allowed or decision.requires_approval:
            raise GatewayPermissionError(
                f"{action} not permitted: {decision.reason}"
                + (f" (approval_id={decision.approval_id})" if decision.approval_id else ""),
                decision.approval_id,
            )

    async def _conversation(self, channel: str, subject: str) -> str:
        # A read-then-insert on two plain connections let concurrent messages
        # from the same (channel, subject) create two sessions, splitting a
        # conversation across ids. BEGIN IMMEDIATE serializes the whole
        # get-or-create so the session key stays deterministic under load.
        async with db.transaction() as conn:
            return await self._conversation_in(conn, channel, subject)

    @staticmethod
    async def _conversation_in(conn: Any, channel: str, subject: str) -> str:
        """Get-or-create inside an already-open write transaction.

        Newest-first: ``/new`` appends a conversation for the same (channel,
        subject), and the channel has to follow it. Oldest-first pinned the
        first conversation forever, so ``/new`` created a conversation nothing
        ever used.
        """
        cursor = await conn.execute(
            "SELECT id FROM conversations WHERE channel=? AND subject=? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (channel, subject),
        )
        row = await cursor.fetchone()
        if row is not None:
            return str(row["id"])
        conversation_id = new_id("conv")
        now = utcnow()
        await conn.execute(
            "INSERT INTO conversations(id,channel,subject,created_at,updated_at,metadata) VALUES(?,?,?,?,?,?)",
            (conversation_id, channel, subject, now, now, _dump({"channel": channel, "subject": subject})),
        )
        return conversation_id

    async def _ensure_outbox(self) -> None:
        """Create the durable outbox lazily (db.py is shared with other agents)."""
        for statement in _OUTBOX_SCHEMA:
            await db.execute(statement)

    @staticmethod
    def _subject_for(message: NormalizedMessage, subject: str | None = None) -> str:
        # Group/channel context wins over the (mutable) sender display name.
        return subject or message.session_ref

    @staticmethod
    def _message_dedupe_key(message: NormalizedMessage) -> str:
        """Durable (channel, message_id) key; also used by the poll path."""
        message_id = message.message_id
        if not message_id:
            basis = "|".join((message.channel, message.identity, message.text, message.timestamp))
            message_id = "auto:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
        digest = hashlib.sha256(str(message_id).encode("utf-8")).hexdigest()
        return f"{_MESSAGE_DEDUPE_PREFIX}{message.channel}:{digest}"

    async def _prepare(
        self,
        messages: list[NormalizedMessage],
        *,
        route_channel: str | None = None,
        subject: str | None = None,
    ) -> list[_Prepared]:
        """Preflight admission/authorization for a whole batch before any write.

        An adapter cannot choose the surface it arrives on: when a route names
        a channel, that channel is pinned over any ``channel`` in the payload,
        so external adapters cannot impersonate ``cli``/``api``/``dashboard``.
        """
        prepared: list[_Prepared] = []
        for raw in messages:
            message = (
                raw if isinstance(raw, NormalizedMessage) else NormalizedMessage.model_validate(raw)
            )
            original_channel: str | None = None
            if route_channel and message.channel != route_channel:
                original_channel = message.channel
                message = message.model_copy(update={"channel": route_channel})
            identity = message.identity
            target_subject = self._subject_for(message, subject)
            decision = self.admission.decide(message.channel, identity)
            if not decision.allowed:
                await self.admission.record_denial(
                    message.channel, identity, decision, correlation_id=message.message_id
                )
                raise GatewayAdmissionError(f"{decision.reason} [policy={decision.source}]")
            await self._authorize(
                "gateway.receive", RiskLevel.LOW, channel=message.channel, subject=target_subject
            )
            attachment_ids: list[str] = []
            if message.attachments and self.attachment_sink is not None:
                # Download + store BEFORE the write transaction: a network
                # round-trip must never hold the database lock.
                for descriptor in message.attachments:
                    try:
                        stored = await self.attachment_sink(message.channel, descriptor)
                    except Exception:
                        stored = None
                    if stored:
                        attachment_ids.append(str(stored))
            prepared.append(
                _Prepared(
                    message=message,
                    original_channel=original_channel,
                    subject=target_subject,
                    identity=identity,
                    dedupe_key=self._message_dedupe_key(message),
                    attachment_ids=tuple(attachment_ids),
                )
            )
        return prepared

    async def _persist_prepared(self, prepared: list[_Prepared]) -> list[dict[str, Any]]:
        """Atomically write a whole batch (messages + outbox + dedupe keys)."""
        if not prepared:
            return []
        await self._ensure_outbox()
        results: list[dict[str, Any]] = []
        to_publish: list[tuple[dict[str, Any], NormalizedMessage]] = []
        async with db.transaction() as conn:
            for item in prepared:
                message = item.message
                cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (item.dedupe_key,))
                row = await cursor.fetchone()
                if row is not None:
                    try:
                        prior = json.loads(str(row["value"]))
                    except (TypeError, ValueError):
                        prior = None
                    if isinstance(prior, dict):
                        results.append({**prior, "duplicate": True})
                        continue
                conversation_id = await self._conversation_in(conn, message.channel, item.subject)
                reply_target = message.reply_target
                metadata = {
                    "channel": message.channel,
                    "subject": item.subject,
                    # Display name stays metadata; routing uses stable ids.
                    "sender": message.sender,
                    "sender_id": message.sender_id,
                    "conversation_ref": message.conversation_ref,
                    "thread": message.thread,
                    "reply_ref": message.reply_ref,
                    "reply_target": reply_target,
                    "source_timestamp": message.timestamp,
                    "source_message_id": message.message_id,
                }
                if item.original_channel:
                    metadata["adapter_channel"] = item.original_channel
                if item.attachment_ids:
                    metadata["attachments"] = list(item.attachment_ids)
                now = utcnow()
                cursor = await conn.execute(
                    "INSERT INTO messages(conversation_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
                    (conversation_id, "user", message.text, _dump(metadata), now),
                )
                db_message_id = str(cursor.lastrowid)
                await conn.execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?", (utcnow(), conversation_id)
                )
                outbox_id = new_id("outbox")
                payload = {
                    "channel": message.channel,
                    "subject": item.subject,
                    "conversation_id": conversation_id,
                    "message_id": db_message_id,
                    "sender": message.sender,
                    "sender_id": message.sender_id,
                    "conversation_ref": message.conversation_ref,
                    "thread": message.thread,
                    "reply_ref": message.reply_ref,
                    "reply_target": reply_target,
                    "outbox_id": outbox_id,
                    "idempotency_key": outbox_id,
                    "text": message.text,
                    "timestamp": message.timestamp,
                    "metadata": metadata,
                    "attachments": list(item.attachment_ids),
                    "duplicate": False,
                }
                encoded = _dump(payload)
                await conn.execute(
                    "INSERT INTO gateway_outbox("
                    "id,channel,subject,conversation_id,source_message_id,sender_id,sender,"
                    "reply_ref,reply_target,text,payload,status,attempts,created_at,updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        outbox_id, message.channel, item.subject, conversation_id, message.message_id,
                        message.sender_id, message.sender, message.reply_ref, reply_target,
                        message.text, encoded, "pending", 0, now, now,
                    ),
                )
                await conn.execute(
                    "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (item.dedupe_key, encoded, now),
                )
                results.append(payload)
                to_publish.append((payload, message))
        for payload, message in to_publish:
            await self.events.publish(
                Event(
                    topic=GATEWAY_MESSAGE_TOPIC,
                    source="gateway",
                    payload=payload,
                    correlation_id=message.message_id,
                )
            )
        return results

    @staticmethod
    def _webhook_fingerprint(channel: str, payload: Any, raw_body: bytes | None) -> str:
        """Stable identity of one delivery, independent of JSON key order."""
        if raw_body is not None:
            body = bytes(raw_body)
        elif isinstance(payload, (bytes, bytearray)):
            body = bytes(payload)
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
            ).encode("utf-8")
        return hashlib.sha256(channel.encode("utf-8") + b"\x00" + body).hexdigest()

    async def _claim_webhook(self, channel: str, payload: Any, raw_body: bytes | None) -> str:
        """Atomically claim a delivery; raise on a replay inside the window."""
        fingerprint = self._webhook_fingerprint(channel, payload, raw_body)
        key = f"{_WEBHOOK_DEDUPE_PREFIX}{fingerprint}"
        now = time.time()
        async with db.transaction() as conn:
            cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (key,))
            row = await cursor.fetchone()
            if row is not None:
                try:
                    first_seen = float(row["value"])
                except (TypeError, ValueError):
                    first_seen = now
                if 0 <= now - first_seen < self.webhook_replay_window:
                    raise GatewayReplayError(
                        f"duplicate {channel} webhook delivery inside the replay window"
                    )
                await conn.execute(
                    "UPDATE kv SET value=?, updated_at=? WHERE key=?", (key, str(now), utcnow())
                )
            else:
                await conn.execute(
                    "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)", (key, str(now), utcnow())
                )
        return key

    async def _release_webhook_claim(self, key: str) -> None:
        """Let a genuine processing failure be redelivered by the provider."""
        async with db.transaction() as conn:
            await conn.execute("DELETE FROM kv WHERE key=?", (key,))

    async def receive(
        self, message: NormalizedMessage | dict[str, Any], *, subject: str | None = None
    ) -> dict[str, Any]:
        """Persist one inbound message, durable outbox record and event.

        Admission is decided centrally *before* anything is persisted or
        emitted: an unauthorized sender never reaches the conversation store,
        the event bus, the outbox or a model.
        """
        return (await self.receive_batch([message], subject=subject))[0]

    async def receive_batch(
        self,
        messages: list[NormalizedMessage] | list[dict[str, Any]],
        *,
        route_channel: str | None = None,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        """Admit, dedupe and persist a whole batch in one atomic write."""
        prepared = await self._prepare(messages, route_channel=route_channel, subject=subject)
        return await self._persist_prepared(prepared)

    async def receive_webhook(
        self, channel: str, payload: Any, headers: Mapping[str, str] | None = None, *,
        subject: str | None = None, raw_body: bytes | None = None,
    ) -> list[dict[str, Any]]:
        """Verify, normalize and route an adapter's webhook payload.

        ``raw_body`` must be the exact bytes received by the HTTP layer so HMAC
        verification matches the provider's signature.
        """
        adapter = self._adapter(channel)
        if not implements_webhook(adapter):
            raise GatewayError(f"channel does not implement webhook ingress: {channel}")
        # Verify and normalize first: an unverified request must never be able
        # to claim (and thereby suppress) a legitimate delivery's fingerprint.
        messages = await adapter.handle_webhook(payload, headers, raw_body=raw_body)
        # Preflight *every* message before claiming the delivery or writing a
        # single row: a batch with one unadmitted sender commits nothing.
        prepared = await self._prepare(messages, route_channel=channel, subject=subject)
        claim_key: str | None = None
        if self.webhook_replay_window:
            claim_key = await self._claim_webhook(channel, payload, raw_body)
        try:
            return await self._persist_prepared(prepared)
        except BaseException:
            if claim_key is not None:
                await self._release_webhook_claim(claim_key)
            raise

    async def poll(self, channel: str) -> list[dict[str, Any]]:
        """Pull inbound messages from a poll-capable channel and route them.

        Channels without a pull transport (most webhook platforms) raise
        ``GatewayError``; use :func:`leo.integrations.channels.polling_channels`.
        """
        adapter = self._adapter(channel)
        if not implements_polling(adapter):
            raise GatewayError(f"channel does not support polling: {channel}")
        # Durable position + liveness, owned by the gateway rather than the
        # adapter: a transport must not have to know where state is kept.
        state = TransportState(channel)
        messages = await adapter.poll(state)
        await state.beat(note=f"{len(messages)} inbound")
        # Same route pinning and durable message-id dedupe as the webhook path.
        return await self.receive_batch(messages, route_channel=channel)

    # ---- durable outbox ----------------------------------------------------

    @staticmethod
    async def _delivery_state(outbox_id: str) -> dict[str, Any]:
        row = await db.fetchone(
            "SELECT value FROM kv WHERE key=?", (_OUTBOX_DELIVERY_PREFIX + outbox_id,)
        )
        if row is None:
            return {}
        try:
            value = json.loads(str(row["value"]))
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    async def _write_delivery_marker(outbox_id: str, state: str, result: Any = None) -> None:
        marker: dict[str, Any] = {"state": state, "outbox_id": outbox_id}
        if result is not None:
            try:
                marker["result"] = json.loads(_dump(result))
            except (TypeError, ValueError):
                marker["result"] = str(result)[:2000]
        async with db.transaction() as conn:
            await conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (_OUTBOX_DELIVERY_PREFIX + outbox_id, _dump(marker), utcnow()),
            )

    async def _claim_outbox(self, limit: int, channel: str | None) -> list[Any]:
        """Atomically move retryable rows to ``processing`` for one drainer."""
        sql = (
            "SELECT id, payload, channel, attempts FROM " + GATEWAY_OUTBOX_TABLE
            + " WHERE status IN ('pending','failed')"
        )
        params: list[Any] = []
        if channel is not None:
            sql += " AND channel=?"
            params.append(channel)
        sql += " ORDER BY created_at ASC, id ASC LIMIT ?"
        params.append(limit)
        async with db.transaction() as conn:
            cursor = await conn.execute(sql, tuple(params))
            rows = await cursor.fetchall()
            for row in rows:
                await conn.execute(
                    "UPDATE " + GATEWAY_OUTBOX_TABLE
                    + " SET status='processing', attempts=attempts+1, updated_at=? "
                    "WHERE id=? AND status IN ('pending','failed')",
                    (utcnow(), row["id"]),
                )
                # DO NOTHING preserves a `delivered` marker from a previous
                # attempt whose bookkeeping transaction failed.
                await conn.execute(
                    "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO NOTHING",
                    (
                        _OUTBOX_DELIVERY_PREFIX + str(row["id"]),
                        _dump({"state": "claimed", "outbox_id": str(row["id"])}),
                        utcnow(),
                    ),
                )
        return list(rows)

    async def _finish_outbox(
        self, outbox_id: str, status: str, *, error: str | None = None, result: Any = None
    ) -> None:
        processed_at = utcnow() if status == "processed" else None
        encoded: str | None = None
        if result is not None:
            try:
                encoded = _dump(result)[:2000]
            except (TypeError, ValueError):
                encoded = str(result)[:2000]
        async with db.transaction() as conn:
            await conn.execute(
                "UPDATE " + GATEWAY_OUTBOX_TABLE
                + " SET status=?, last_error=?, last_result=?, processed_at=?, updated_at=? WHERE id=?",
                (status, error, encoded, processed_at, utcnow(), outbox_id),
            )

    async def pending_outbox(self, *, channel: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Read-only view of admitted messages still awaiting a reply turn."""
        await self._ensure_outbox()
        sql = "SELECT id, channel, subject, reply_target, text, status, attempts, created_at, last_error FROM " + GATEWAY_OUTBOX_TABLE + " WHERE status IN ('pending','failed')"
        params: list[Any] = []
        if channel is not None:
            sql += " AND channel=?"
            params.append(channel)
        sql += " ORDER BY created_at ASC, id ASC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        rows = await db.fetchall(sql, tuple(params))
        return [
            {
                "outbox_id": row["id"], "channel": row["channel"], "subject": row["subject"],
                "reply_target": row["reply_target"], "text": row["text"], "status": row["status"],
                "attempts": row["attempts"], "created_at": row["created_at"],
                "last_error": row["last_error"],
            }
            for row in rows
        ]

    async def recover_stale_outbox(self, *, older_than_seconds: float = 300.0) -> int:
        """Requeue rows stranded in ``processing`` by a crashed drainer.

        Call once at startup (before the drain loop) so an admitted message is
        never lost because its worker died after claiming it.
        """
        await self._ensure_outbox()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=max(0.0, float(older_than_seconds)))
        ).isoformat()
        async with db.transaction() as conn:
            cursor = await conn.execute(
                "UPDATE " + GATEWAY_OUTBOX_TABLE
                + " SET status='pending', updated_at=? WHERE status='processing' AND updated_at < ?",
                (utcnow(), cutoff),
            )
            return int(cursor.rowcount or 0)

    async def outbox_stats(self) -> dict[str, int]:
        """Per-status counts; useful for health checks and drain loops."""
        await self._ensure_outbox()
        rows = await db.fetchall(
            "SELECT status, COUNT(*) AS count FROM " + GATEWAY_OUTBOX_TABLE + " GROUP BY status"
        )
        stats = {"pending": 0, "processing": 0, "failed": 0, "processed": 0}
        for row in rows:
            stats[str(row["status"])] = int(row["count"])
        return stats

    # Telegram clears the typing indicator after ~5 s, so one call is not
    # enough for a turn that takes 20 s. Ping on a short interval until the
    # handler returns; a failure is cosmetic and must never break delivery.
    TYPING_INTERVAL_SECONDS = 4.0

    async def _typing_heartbeat(self, payload: dict[str, Any]) -> None:
        channel = str(payload.get("channel") or "")
        target = str(payload.get("reply_target") or payload.get("subject") or "")
        adapter = self.adapters.get(channel)
        if not target or adapter is None:
            return
        while True:
            try:
                await adapter.typing(target)
            except Exception:
                return
            await asyncio.sleep(self.TYPING_INTERVAL_SECONDS)

    async def drain_outbox(
        self,
        handler: OutboxHandler | None = None,
        *,
        limit: int = 10,
        channel: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, int]:
        """Public draining API: run one reply turn per claimed outbox record.

        The handler receives the full admitted-message payload (including
        ``conversation_id``, ``reply_target`` and ``text``) and is responsible
        for starting an agent turn and calling :meth:`send`. Failures keep the
        record retryable with an error note; repeated failures dead-letter it
        as ``failed``. No handler means no silent success.
        """
        await self._ensure_outbox()
        consumer = handler or self._reply_handler
        if consumer is None:
            raise GatewayError(
                "no outbox reply handler: call gateway.set_reply_handler() before drain_outbox()"
            )
        claimed = await self._claim_outbox(max(1, min(int(limit), 100)), channel)
        summary = {"claimed": len(claimed), "processed": 0, "suppressed": 0, "failed": 0, "remaining": 0}
        for row in claimed:
            outbox_id = str(row["id"])
            try:
                payload = json.loads(str(row["payload"]))
            except (TypeError, ValueError):
                payload = {}
            # The outbox id is the end-to-end idempotency key: an app handler
            # must forward it to gateway.send()/the adapter for real suppression.
            payload["idempotency_key"] = outbox_id
            try:
                marker = await self._delivery_state(outbox_id)
                if marker.get("state") == "delivered":
                    # The side effect already happened; only bookkeeping failed.
                    await self._finish_outbox(
                        outbox_id, "processed", result=marker.get("result")
                    )
                    await self.events.emit(
                        GATEWAY_OUTBOX_TOPIC + ".duplicate_suppressed",
                        outbox_id=outbox_id, channel=payload.get("channel"),
                        conversation_id=payload.get("conversation_id"),
                    )
                    summary["suppressed"] += 1
                    summary["processed"] += 1
                    continue
                typing_task = asyncio.create_task(self._typing_heartbeat(payload))
                try:
                    result = await consumer(payload)
                finally:
                    typing_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await typing_task
                if not _delivery_confirmed(result):
                    raise GatewayError("reply handler did not confirm delivery")
                # Persist the delivery marker BEFORE the processed status: if
                # either transaction fails, the next drain suppresses the send.
                await self._write_delivery_marker(outbox_id, "delivered", result)
                await self._finish_outbox(outbox_id, "processed", result=result)
                await self.events.emit(
                    GATEWAY_OUTBOX_TOPIC + ".processed",
                    outbox_id=outbox_id, channel=payload.get("channel"),
                    conversation_id=payload.get("conversation_id"),
                )
                summary["processed"] += 1
            except Exception as exc:  # noqa: BLE001 - one bad turn must not stall the queue
                marker = await self._delivery_state(outbox_id)
                error = f"{type(exc).__name__}: {exc}"
                if marker.get("state") == "delivered":
                    # External delivery is confirmed; never replay it.
                    await self.events.emit(
                        GATEWAY_OUTBOX_TOPIC + ".bookkeeping.failed",
                        outbox_id=outbox_id, channel=payload.get("channel"), error=error,
                    )
                else:
                    await self._write_delivery_marker(outbox_id, "claimed")
                    status = "pending" if (int(row["attempts"]) + 1) < max(1, int(max_attempts)) else "failed"
                    await self._finish_outbox(outbox_id, status, error=error)
                    await self.events.emit(
                        GATEWAY_OUTBOX_TOPIC + ".failed",
                        outbox_id=outbox_id, channel=payload.get("channel"),
                        conversation_id=payload.get("conversation_id"), error=error,
                    )
                summary["failed"] += 1
        stats = await self.outbox_stats()
        summary["remaining"] = stats["pending"] + stats["failed"]
        return summary

    async def policy_version(self) -> int:
        """Current durable channel-policy version."""
        return await self.admission.policy_version()

    async def refresh_policies(self, *, expected_version: int | None = None) -> dict[str, Any]:
        """Versioned policy refresh (emits an audited, structured event)."""
        return await self.admission.refresh_policies(expected_version=expected_version)

    async def send(
        self,
        channel: str,
        target: str,
        text: str,
        *,
        subject: str | None = None,
        approval_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Send one outbound message through the channel's adapter.

        HIGH-risk approvals bind channel, immutable target, text and subject via
        ``arguments_digest``; replaying an approval with changed arguments is
        rejected by the permission manager. ``idempotency_key`` (the outbox id
        in the delivery path) makes an accepted external send replay-safe.
        """
        adapter = self._adapter(channel)
        subject = subject or str(target)
        target_str = str(target)
        # The resource is the immutable destination, never a caller-chosen
        # subject: changing only the subject must not forge a new resource.
        resource = f"{channel}:{target_str}"
        send_arguments = {
            "channel": channel, "target": target_str, "text": str(text), "subject": subject,
        }
        digest = arguments_digest(send_arguments)
        # Authorize before any state change: a denied send must not leave an
        # empty conversation behind (receive() follows the same ordering).
        await self._authorize(
            "gateway.send",
            RiskLevel.HIGH,
            resource=resource,
            approval_id=approval_id,
            arguments=send_arguments,
            arguments_digest=digest,
            channel=channel,
            target=target_str,
        )
        if idempotency_key:
            key = _SEND_IDEMPOTENCY_PREFIX + str(idempotency_key)
            row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
            if row is not None:
                try:
                    prior = json.loads(str(row["value"]))
                except (TypeError, ValueError):
                    prior = None
                if isinstance(prior, dict):
                    return {**prior, "duplicate": True, "idempotency_key": str(idempotency_key)}
        conversation_id = await self._conversation(channel, subject)
        try:
            result = await adapter.send(target, text)
            if not isinstance(result, Mapping):
                raise MessagingError(f"{channel} adapter returned an invalid send result")
        except Exception as exc:
            return {
                "channel": channel,
                "target": target_str,
                "conversation_id": conversation_id,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        metadata = {
            "channel": channel,
            "subject": subject,
            "target": target_str,
            "provider_message_id": result.get("message_id"),
            "idempotency_key": idempotency_key,
        }
        response = {
            **result, "conversation_id": conversation_id, "subject": subject,
            "duplicate": False, "idempotency_key": idempotency_key,
        }
        now = utcnow()
        async with db.transaction() as conn:
            await conn.execute(
                "INSERT INTO messages(conversation_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
                (conversation_id, "assistant", str(text), _dump(metadata), now),
            )
            await conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id)
            )
            if idempotency_key:
                await conn.execute(
                    "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (key, _dump(response), now),
                )
        await self.events.emit(
            GATEWAY_SENT_TOPIC,
            channel=channel,
            subject=subject,
            conversation_id=conversation_id,
            target=target_str,
            provider_message_id=result.get("message_id"),
            idempotency_key=idempotency_key,
        )
        return response

    async def fanout(
        self, targets: Mapping[str, str], text: str, *, approval_id: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Send to one target per channel; one channel's failure does not stop others."""
        results: dict[str, dict[str, Any]] = {}
        for channel, target in targets.items():
            try:
                results[channel] = await self.send(
                    channel, target, text, approval_id=approval_id
                )
            except GatewayError as exc:
                results[channel] = {
                    "channel": channel,
                    "target": str(target),
                    "success": False,
                    "error": str(exc),
                }
        return results

    async def history(
        self, channel: str | None = None, subject: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Chronological unified history, optionally filtered by channel/subject."""
        await self._authorize(
            "gateway.history", RiskLevel.LOW, channel=channel, subject=subject, limit=limit
        )
        bounded = max(1, min(int(limit), 500))
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT m.id, m.conversation_id, m.role, m.content, m.metadata, m.created_at, "
                "c.channel, c.subject FROM messages m JOIN conversations c ON c.id=m.conversation_id "
                "WHERE (? IS NULL OR c.channel=?) AND (? IS NULL OR c.subject=?) "
                "ORDER BY m.created_at ASC, m.id ASC LIMIT ?",
                (channel, channel, subject, subject, bounded),
            )
            rows = await cursor.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
            except (TypeError, ValueError):
                metadata = {}
            result.append(
                {
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "channel": row["channel"],
                    "subject": row["subject"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": metadata,
                    "created_at": row["created_at"],
                }
            )
        return result

    async def sessions(self, channel: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List isolated sessions (channel + subject) with message counts."""
        await self._authorize("gateway.sessions", RiskLevel.LOW, channel=channel)
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count "
                "FROM conversations c WHERE (? IS NULL OR c.channel=?) "
                "ORDER BY c.updated_at DESC LIMIT ?",
                (channel, channel, max(1, min(int(limit), 500))),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    def integration_records(self) -> list[IntegrationRecord]:
        return [adapter.record() for adapter in self.adapters.values()]


__all__ = [
    "GATEWAY_MESSAGE_TOPIC",
    "GATEWAY_SENT_TOPIC",
    "GATEWAY_OUTBOX_TOPIC",
    "GATEWAY_OUTBOX_TABLE",
    "WEBHOOK_REPLAY_WINDOW_SECONDS",
    "Gateway",
    "GatewayAdmissionError",
    "GatewayReplayError",
    "GatewayPermissionError",
    "NormalizedMessage",
]
