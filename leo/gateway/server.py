"""The gateway process: transport in, durable queue, local API out.

Same shape as a dedicated agent gateway: one long-lived supervised process owns
the channel adapters, writes every inbound message to a durable queue, and serves
a small local API that a brain process (``leo serve --worker``) claims from. The
gateway also owns delivery, because only it holds the adapters.

The API is HTTP on loopback rather than WebSocket: this project ships no
WebSocket client and adding one for a single local hop buys nothing. The
guarantees the split exists for — durable queue, leases, at-least-once, a brain
that can restart without dropping transport — do not depend on the framing.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .. import db
from ..contracts import surface_capabilities

# One send attempt may not outlive this: a stuck platform call is a failed send,
# and an unbounded one keeps the worker's lease (and its typing pulse) alive.
DELIVERY_TIMEOUT_SECONDS = 45.0
from ..integrations.channels import default_adapters, polling_channels
from ..integrations.messaging import AdapterCapabilities, implements_polling
from ..integrations.secrets import credential
from ..integrations.transcription import VOICE_UNAVAILABLE_NOTE
from ..integrations.transport_state import TransportState
from ..security.approval_buttons import build_schedule_keyboard
from ..security.approval_cards import (
    APPROVAL_CARD_PROTOCOL_VERSION,
    card_fits_surface,
    card_from_row,
    is_schedule_kind,
    proposal_digest_matches,
    schedule_card_text,
)
from .sessions import SessionStore
from .store import GatewayStore

_LOG = logging.getLogger(__name__)

WORKER_IDLE_SLICE = 0.5

# The worker only sends a schedule card through a gateway that advertises the
# same card protocol and a channel with native inbound identity + terminal
# rewrite. An older gateway advertises nothing: it receives no card at all.
APPROVAL_RPC_PROTOCOL = APPROVAL_CARD_PROTOCOL_VERSION

# A tap is handled in the gateway process, so it already rewrites its own card.
# Marking it here prevents the worker's later lease-settle rewrite from editing
# the same message twice. This is display state, not approval authority.
_TERMINAL_REWRITES: set[str] = set()


def mark_terminal_rewrite(approval_id: str) -> None:
    _TERMINAL_REWRITES.add(str(approval_id))


def terminal_rewrite_done(approval_id: str) -> bool:
    return str(approval_id) in _TERMINAL_REWRITES


class ClaimRequest(BaseModel):
    worker: str = Field(min_length=1)
    lease_seconds: int = Field(default=30, ge=1, le=900)
    wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)


class RenewRequest(BaseModel):
    admission_id: str
    worker: str = Field(min_length=1)
    lease_seconds: int = Field(default=30, ge=1, le=900)


class CompleteRequest(BaseModel):
    admission_id: str
    answer: str = ""
    conversation_id: str = ""
    # An inline keyboard the answer carries (the model picker). Optional: a
    # channel that cannot render one simply ignores it.
    reply_markup: dict[str, Any] | None = None
    # When set, replace this already-sent message (the tapped menu card) with
    # the answer instead of posting a duplicate below it.
    edit_message_id: str | None = None
    # Identity the worker was admitted with. When supplied it must match the
    # durable admission; delivery always uses the admission's own reply target,
    # so a model or body cannot redirect the answer to another chat/thread.
    principal: str | None = None
    thread_id: str | None = None
    reply_target: str | None = None


class FailRequest(BaseModel):
    admission_id: str
    error: str = ""
    retry: bool = True


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApprovalPresentRequest(_StrictRequest):
    admission_id: str
    approval_id: str
    confirm_proposal_digest: str
    principal: str | None = None
    thread_id: str | None = None


class ApprovalAckRequest(_StrictRequest):
    admission_id: str
    approval_id: str


class ApprovalUpdateRequest(_StrictRequest):
    admission_id: str
    approval_id: str
    confirm_proposal_digest: str
    text: str = Field(min_length=1)


def gateway_approval_channels(health: Any) -> frozenset[str]:
    """Channels a worker may use for schedule cards, from advertised health only.

    Missing or mismatched protocol means an old gateway: no card, no guess.
    """
    if not isinstance(health, dict):
        return frozenset()
    if health.get("approval_protocol") != APPROVAL_RPC_PROTOCOL:
        return frozenset()
    advertised = health.get("approval_cards")
    if not isinstance(advertised, dict):
        return frozenset()
    capable: set[str] = set()
    for name, caps in advertised.items():
        if not isinstance(caps, dict):
            continue
        if all(caps.get(key) is True for key in (
            "can_present_card", "can_wait_for_decision", "full_text", "terminal_edit",
        )):
            capable.add(str(name))
    return frozenset(capable)


def approval_channel_capable(surface: str, adapter: Any) -> bool:
    """Full manifest proof, not a platform-name guess."""
    caps = surface_capabilities(surface)
    adapter_caps = AdapterCapabilities.for_adapter(adapter)
    return bool(
        caps.can_present_card
        and caps.can_wait_for_decision
        and caps.full_text
        and caps.can_rewrite
        and adapter_caps.approval_card
        and adapter_caps.full_text
        and adapter_caps.terminal_edit
    )


def _presented(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    value = authorization.removeprefix("Bearer ").strip()
    if not value:
        raise HTTPException(status_code=401, detail="empty bearer token")
    return value


def create_api(
    store: GatewayStore,
    *,
    token: str,
    adapters: dict[str, Any] | None = None,
    approval_protocol: int | None = APPROVAL_RPC_PROTOCOL,
) -> FastAPI:
    """Build the gateway's local API. Auth is required on every route."""
    api = FastAPI(title="LEO gateway", version="1")
    channels = adapters if adapters is not None else default_adapters()

    sessions = SessionStore()
    typing_tasks: dict[str, asyncio.Task[Any]] = {}
    # Transport state of cards this gateway actually delivered. Not an approval
    # authority: the shared approvals row stays the one durable authority.
    presented_cards: dict[str, dict[str, Any]] = {}

    def _advertised_cards() -> dict[str, dict[str, Any]]:
        if approval_protocol is None:
            return {}
        advertised: dict[str, dict[str, Any]] = {}
        for name, adapter in channels.items():
            if approval_channel_capable(name, adapter):
                caps = surface_capabilities(name)
                advertised[name] = {
                    **caps.model_dump(mode="json"),
                    "terminal_edit": True,
                }
        return advertised

    async def _typing_until(adapter: Any, subject: str, interval: float = 4.0) -> None:
        """Repeat the channel's typing action until the turn finishes.

        In the split the brain answers over the API, so nothing in the reply path
        ever touches an adapter - which is why the indicator disappeared when the
        processes were separated. Transport hints belong to the transport.
        """
        while True:
            try:
                await adapter.typing(subject)
            except Exception:  # cosmetic only: never fail a turn for a typing hint
                pass
            await asyncio.sleep(interval)

    def _start_typing(admission_id: str, channel: str, subject: str) -> None:
        adapter = channels.get(channel)
        if adapter is None or not hasattr(adapter, "typing"):
            return
        previous = typing_tasks.pop(admission_id, None)
        if previous is not None:
            previous.cancel()
        typing_tasks[admission_id] = asyncio.create_task(
            _typing_until(adapter, subject), name=f"typing-{admission_id}"
        )

    def _stop_typing(admission_id: str) -> None:
        task = typing_tasks.pop(admission_id, None)
        if task is not None:
            task.cancel()

    async def auth(authorization: str | None = Header(default=None)) -> None:
        """A worker's session token is valid for the turn routes; the master
        token stays with the operator and is the only thing that can mint one."""
        presented = _presented(authorization)
        if presented == token:
            return
        if await sessions.session_valid(presented):
            return
        raise HTTPException(status_code=401, detail="bad token")

    @api.get("/health")
    async def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await auth(authorization)
        stats = await store.stats()
        body: dict[str, Any] = {
            "ok": True, "pid": os.getpid(), "channels": sorted(channels),
            "sessions": await sessions.stats(), **stats,
        }
        if approval_protocol is not None:
            body["approval_protocol"] = approval_protocol
            body["approval_cards"] = _advertised_cards()
        return body

    @api.post("/v1/claim")
    async def claim(request: ClaimRequest,
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await auth(authorization)
        deadline = asyncio.get_running_loop().time() + request.wait_seconds
        while True:
            admission = await store.claim(request.worker, request.lease_seconds)
            if admission is not None:
                admission["conversation_id"] = await store.route(
                    admission["channel"], admission["subject"]
                )
                _start_typing(admission["admission_id"], admission["channel"],
                              admission["subject"])
                return {"admission": admission}
            if asyncio.get_running_loop().time() >= deadline:
                return {"admission": None}
            await asyncio.sleep(WORKER_IDLE_SLICE)
            # A worker that died mid-turn never calls /v1/complete or /v1/fail,
            # so its typing task would pulse on this channel forever. Reclaim
            # reports the ids precisely so those pulses can be stopped here —
            # the gesture that started them lives in this process.
            for reclaimed in await store.reclaim_expired():
                _stop_typing(reclaimed)

    @api.post("/v1/renew")
    async def renew(request: RenewRequest,
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await auth(authorization)
        held = await store.renew(request.admission_id, request.worker, request.lease_seconds)
        return {"renewed": held}

    @api.post("/v1/complete")
    async def complete(request: CompleteRequest,
                       authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await auth(authorization)
        admission = await store.admission(request.admission_id)
        if admission is None:
            raise HTTPException(status_code=404, detail="unknown admission")
        # Sender identity is the admission's immutable principal/thread, never
        # the worker's body. An echoed mismatch means a different conversation
        # (or a forged reply) is trying to answer this admission.
        if request.principal and admission.get("principal") and (
            request.principal != admission["principal"]
        ):
            raise HTTPException(status_code=409, detail="principal does not own this admission")
        if request.thread_id and admission.get("thread_id") and (
            str(request.thread_id) != str(admission["thread_id"])
        ):
            raise HTTPException(status_code=409, detail="thread does not match admission")
        delivered, error = True, ""
        target = str(admission.get("reply_target") or admission["subject"])
        if request.answer.strip():
            adapter = channels.get(admission["channel"])
            if adapter is None:
                delivered, error = False, f"no adapter for {admission['channel']}"
            else:
                edited = False
                if request.edit_message_id and callable(
                    getattr(adapter, "edit_message_text", None)
                ):
                    # A menu tap replaces the card the human tapped. Posting a
                    # second full menu below it made every tap look like a
                    # duplicated card.
                    try:
                        edited = bool(await asyncio.wait_for(
                            adapter.edit_message_text(
                                target, request.edit_message_id,
                                # The tap consumed the choice: keep the result
                                # text, remove the buttons.
                                request.answer, {"inline_keyboard": []},
                            ),
                            timeout=DELIVERY_TIMEOUT_SECONDS,
                        ))
                    except Exception:  # noqa: BLE001 - a failed edit falls back to send
                        edited = False
                if not edited:
                    try:
                        # A hung platform call must not keep the turn (and its typing
                        # pulse) alive forever: the worker renews the lease until this
                        # returns, so an unbounded send is an endless "still typing".
                        # A failed tap edit falls back to text-only, never a second
                        # live menu.
                        extra = (
                            {"reply_markup": request.reply_markup}
                            if request.reply_markup and not request.edit_message_id
                            else {}
                        )
                        await asyncio.wait_for(
                            adapter.send(target, request.answer, **extra),
                            timeout=DELIVERY_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        delivered, error = False, "delivery timed out"
                        _LOG.warning(
                            "gateway delivery timed out for %s after %ss",
                            admission["channel"], DELIVERY_TIMEOUT_SECONDS,
                            extra={"channel": admission["channel"], "kind": "delivery_timeout"},
                        )
                    except Exception as exc:  # delivery failure must not lose the turn
                        delivered, error = False, f"{type(exc).__name__}: {exc}"[:300]
                        # The reason belongs in the message, not only in an `extra` field
                        # that the default formatter drops: a silent "delivery failed"
                        # line is what made a 400 from the chat platform look like a
                        # healthy turn that simply never arrived.
                        _LOG.warning(
                            "gateway delivery failed for %s: %s",
                            admission["channel"], error,
                            extra={"channel": admission["channel"], "error": error},
                        )
        if request.conversation_id:
            await store.save_route(admission["channel"], admission["subject"],
                                   request.conversation_id)
        _stop_typing(request.admission_id)
        await store.complete(request.admission_id)
        return {"delivered": delivered, "error": error}

    @api.post("/v1/fail")
    async def fail(request: FailRequest,
                   authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await auth(authorization)
        _stop_typing(request.admission_id)
        status = await store.fail(request.admission_id, request.error, retry=request.retry)
        return {"status": status}

    @api.get("/v1/admission/{admission_id}")
    async def admission_route(admission_id: str,
                              authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await auth(authorization)
        admission = await store.admission(admission_id)
        if admission is None:
            raise HTTPException(status_code=404, detail="unknown admission")
        return admission

    @api.post("/v1/attach")
    async def attach(ttl_seconds: int = 300,
                     authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """Mint a single-use attach ticket (master token only)."""
        presented = _presented(authorization)
        if presented != token:
            raise HTTPException(status_code=401, detail="attach requires the master token")
        ticket, expires_at = await sessions.issue_ticket(ttl_seconds)
        return {"ticket": ticket, "expires_at": expires_at}

    @api.post("/v1/session")
    async def session(payload: dict[str, Any]) -> dict[str, Any]:
        """Exchange an attach ticket for a session token. A ticket works once."""
        ticket = str(payload.get("ticket") or "")
        ttl = int(payload.get("ttl_seconds") or 86400)
        session_token = await sessions.redeem_ticket(ticket, ttl)
        if session_token is None:
            raise HTTPException(status_code=401, detail="ticket is invalid, used or expired")
        return {"session_token": session_token}

    def _verify_identity(
        admission: Mapping[str, Any], principal: str | None, thread_id: str | None
    ) -> None:
        if principal and admission.get("principal") and principal != admission["principal"]:
            raise HTTPException(409, "principal does not own this admission")
        if thread_id and admission.get("thread_id") and str(thread_id) != str(admission["thread_id"]):
            raise HTTPException(409, "thread does not match admission")

    async def _card_for_transport(approval_id: str, digest: str, channel: str) -> Any:
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None:
            raise HTTPException(404, "unknown approval")
        if not is_schedule_kind(str(row["kind"] or "")):
            raise HTTPException(409, "not a schedule approval card")
        if str(row["status"] or "") != "pending":
            raise HTTPException(409, f"approval already {row['status']}")
        if not proposal_digest_matches(row, digest):
            raise HTTPException(409, "proposal digest does not match the durable card")
        try:
            card = card_from_row(row)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if not card_fits_surface(card, channel):
            raise HTTPException(409, "card exceeds this channel's presentation budget")
        return card

    if approval_protocol is not None:

        @api.post("/v1/approval/present")
        async def approval_present(
            request: ApprovalPresentRequest,
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            """Send a full literal card to a capable channel the gateway owns."""
            await auth(authorization)
            admission = await store.admission(request.admission_id)
            if admission is None:
                raise HTTPException(404, "unknown admission")
            _verify_identity(admission, request.principal, request.thread_id)
            channel = str(admission.get("channel") or "")
            adapter = channels.get(channel)
            if adapter is None or not approval_channel_capable(channel, adapter):
                raise HTTPException(409, "channel cannot present a schedule card")
            card = await _card_for_transport(
                request.approval_id, request.confirm_proposal_digest, channel
            )
            text = schedule_card_text(card)
            if len(text) > (surface_capabilities(channel).max_card_chars or len(text)):
                raise HTTPException(409, "card exceeds this channel's size limit")
            keyboard = build_schedule_keyboard(
                card.approval_id, approval_kind=card.kind, choices=card.choices,
                choice_labels=card.choice_labels, deny_only=card.deny_only,
            )
            extra: dict[str, Any] = {"reply_markup": keyboard}
            if channel == "telegram":
                extra["plain_text"] = True
            target = str(admission.get("reply_target") or admission.get("subject") or "")
            try:
                result = await asyncio.wait_for(
                    adapter.send(target, text, **extra),
                    timeout=DELIVERY_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # noqa: BLE001 - transport failure is fail-closed
                raise HTTPException(502, f"card delivery failed: {type(exc).__name__}") from exc
            message_id = ""
            if isinstance(result, Mapping):
                message_id = str(result.get("message_id") or "")
            presented_cards[card.approval_id] = {
                "admission_id": request.admission_id,
                "message_id": message_id,
                "channel": channel,
                "target": target,
                "digest": card.proposal_digest,
                "acked_at": None,
            }
            return {
                "presented": True, "approval_id": card.approval_id,
                "message_id": message_id, "channel": channel,
                "protocol": APPROVAL_RPC_PROTOCOL,
            }

        @api.post("/v1/approval/ack")
        async def approval_ack(
            request: ApprovalAckRequest,
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            """Transport receipt of a presented card; not a decision or shown proof."""
            await auth(authorization)
            admission = await store.admission(request.admission_id)
            if admission is None:
                raise HTTPException(404, "unknown admission")
            record = presented_cards.get(request.approval_id)
            if record is None or record["admission_id"] != request.admission_id:
                raise HTTPException(409, "this gateway did not present that card")
            record["acked_at"] = datetime.now(timezone.utc).isoformat()
            return {"acked": True, "approval_id": request.approval_id}

        @api.post("/v1/approval/update")
        async def approval_update(
            request: ApprovalUpdateRequest,
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            """Replace the tapped card with its kind-preserving terminal text."""
            await auth(authorization)
            admission = await store.admission(request.admission_id)
            if admission is None:
                raise HTTPException(404, "unknown admission")
            record = presented_cards.get(request.approval_id)
            if record is None or record["admission_id"] != request.admission_id:
                raise HTTPException(409, "this gateway did not present that card")
            if not record["acked_at"]:
                raise HTTPException(409, "card was not acknowledged before rewrite")
            if terminal_rewrite_done(request.approval_id):
                _TERMINAL_REWRITES.discard(request.approval_id)
                presented_cards.pop(request.approval_id, None)
                return {"updated": False, "reason": "tap already rewrote this card",
                        "approval_id": request.approval_id}
            if not proposal_digest_matches({"digest": record["digest"]}, request.confirm_proposal_digest):
                raise HTTPException(409, "proposal digest does not match the presented card")
            adapter = channels.get(str(record["channel"]))
            edit = getattr(adapter, "edit_message_text", None)
            if not callable(edit) or not str(record["message_id"]):
                raise HTTPException(409, "channel cannot rewrite a presented card")
            try:
                updated = bool(await asyncio.wait_for(
                    edit(record["target"], record["message_id"], request.text,
                         {"inline_keyboard": []}),
                    timeout=DELIVERY_TIMEOUT_SECONDS,
                ))
            except Exception as exc:  # noqa: BLE001 - a rewrite failure is reported
                raise HTTPException(502, f"card rewrite failed: {type(exc).__name__}") from exc
            if updated:
                presented_cards.pop(request.approval_id, None)
            return {"updated": updated, "approval_id": request.approval_id}

    @api.post("/v1/admit")
    async def admit(payload: dict[str, Any],
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """Explicit ingress for pushes the gateway did not poll itself."""
        await auth(authorization)
        admission_id, created = await store.admit(
            channel=str(payload.get("channel") or "unknown"),
            subject=str(payload.get("subject") or ""),
            principal=str(payload.get("principal") or payload.get("sender_id") or ""),
            payload=payload,
            request_id=str(payload.get("request_id") or ""),
            thread_id=(
                str(payload.get("thread_id") or payload.get("thread"))
                if payload.get("thread_id") or payload.get("thread") else None
            ),
            reply_target=(
                str(payload.get("reply_target"))
                if payload.get("reply_target") else None
            ),
        )
        return {"admission_id": admission_id, "created": created}

    return api


async def store_inbound_attachments(
    channel: str,
    descriptors: list[dict[str, Any]] | None,
    sink: Any | None,
) -> list[str]:
    """Turn provider descriptors into store-issued attachment ids.

    In a split deployment the gateway polls the transport but the worker owns
    the attachment store. A descriptor (a bare file_id) cannot cross that gap:
    the worker rejects it as "attachments must reference a stored attachment
    id" and the photo becomes an empty turn. Download + store here, in the
    process that holds the adapters, and admit only store-issued ids. A failed
    download drops the attachment, never the message.
    """
    ids: list[str] = []
    if not descriptors or sink is None:
        return ids
    for descriptor in descriptors:
        try:
            stored = await sink(channel, descriptor)
        except Exception:  # one bad attachment must not drop the message
            stored = None
        if stored:
            ids.append(str(stored))
    return ids


async def transcribe_inbound_voice(
    channel: str,
    descriptors: list[dict[str, Any]] | None,
    adapters: dict[str, Any],
) -> str:
    """Turn voice/audio descriptors into transcribed text for the turn.

    Voice notes carry no text of their own. They are downloaded through the
    channel adapter and transcribed through the model gateway, then appended to
    the message text so the brain sees a real turn instead of an empty one. A
    failed transcription yields no text but never drops the message.
    """
    from ..integrations.transcription import transcribe_channel_voice

    chunks: list[str] = []
    for descriptor in descriptors or []:
        if descriptor.get("kind") != "voice":
            continue
        text = await transcribe_channel_voice(channel, descriptor, adapters)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


async def poll_loop(store: GatewayStore, adapters: dict[str, Any],
                    interval: float = 2.0, attachment_sink: Any | None = None) -> None:
    """Pull from every credentialed poll channel and queue what arrives."""
    while True:
        for channel in polling_channels():
            adapter = adapters.get(channel)
            if adapter is None or not implements_polling(adapter):
                continue
            if not all(credential(name) for name in adapter.credential_env):
                continue
            state = TransportState(channel)
            try:
                messages = await adapter.poll(state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one broken channel must not stop the rest
                _LOG.warning("gateway poll failed",
                             extra={"channel": channel, "error": str(exc)})
                continue
            await state.beat(note=f"{len(messages)} inbound")
            for message in messages:
                # Voice notes carry no text: transcribe them into the turn here,
                # in the polling process that owns the adapters. Other media is
                # stored as attachments; the worker only understands stored ids.
                voice_text = await transcribe_inbound_voice(
                    message.channel, message.attachments, adapters
                )
                turn_text = message.text
                if voice_text:
                    turn_text = f"{turn_text}\n\n{voice_text}".strip() if turn_text else voice_text
                elif any(
                    d.get("kind") == "voice" for d in (message.attachments or [])
                ):
                    # A voice note that could not be transcribed must still be a
                    # real turn: silence here reads to the user as a dead bot.
                    note = VOICE_UNAVAILABLE_NOTE
                    turn_text = f"{turn_text}\n\n{note}".strip() if turn_text else note
                non_voice = [
                    d for d in (message.attachments or []) if d.get("kind") != "voice"
                ]
                attachment_ids = await store_inbound_attachments(
                    message.channel, non_voice, attachment_sink
                )
                await store.admit(
                    channel=message.channel,
                    subject=message.conversation_ref or message.sender_id,
                    principal=message.sender_id or "",
                    thread_id=message.thread,
                    reply_target=message.reply_target,
                    payload={
                        "text": turn_text,
                        "sender": message.sender,
                        "sender_id": message.sender_id,
                        "conversation_ref": message.conversation_ref,
                        "thread": message.thread,
                        "reply_target": message.reply_target,
                        "message_id": message.message_id,
                        "attachments": attachment_ids,
                    },
                    request_id=message.message_id,
                )
        await asyncio.sleep(interval)


def run_gateway(host: str = "127.0.0.1", port: int = 8766) -> None:
    """Start the gateway: adapters + durable queue + local API.

    The token is generated on first run and stored beside the config, so a
    worker can attach without the operator handling a secret by hand.
    """
    import uvicorn

    token = _ensure_token()
    db_path = os.getenv("LEO_DB_PATH")
    if db_path:
        db.set_db_path(db_path)
    store = GatewayStore()
    adapters = default_adapters()
    # Approval buttons are tapped on the transport that polls Telegram, which is
    # THIS process in a split deployment. The app sets a sink too, but it lives in
    # the worker and never sees an update - so without this every tap answered
    # "No handler for this button".
    try:
        from ..config import Settings
        from ..security.permissions import PermissionManagerImpl
        from .callbacks import install as install_callbacks

        _settings = Settings.load()
        _permissions = PermissionManagerImpl(_settings)

        async def _admit_command(
            text: str, channel: str, subject: str, source_message_id: str | None = None,
        ) -> None:
            """Queue a slash command as if the operator had typed it.

            A model or reasoning tap must not edit state from this process: the
            brain owns settings and the current conversation route, so the tap
            travels as normal work. The nonce keeps two taps of the same value
            distinct — admissions are de-duplicated by payload digest. The
            source message id lets the brain's reply replace the tapped card
            instead of posting a duplicate menu below it.
            """
            from ..contracts import new_id

            nonce = new_id("cb")
            payload = {"text": text, "sender_id": "operator-button",
                       "message_id": nonce, "conversation_ref": subject}
            if source_message_id:
                payload["source_message_id"] = str(source_message_id)
            await store.admit(
                channel=channel, subject=subject, payload=payload,
                principal=subject, request_id=nonce, reply_target=subject,
            )

        install_callbacks(_settings, _permissions, adapters, admit=_admit_command)
    except Exception as exc:  # noqa: BLE001 - a gateway must start even if taps cannot
        _LOG.warning("callback handler not installed", extra={"error": type(exc).__name__})
    api = create_api(store, token=token, adapters=adapters)

    # Inbound photos arrive on THIS polling process, but the attachment store
    # lives with the worker. Build a store here and download+store descriptors
    # before admitting, so a photo reaches the vision path as a stored id
    # instead of a bare file_id the worker refuses. Mirrors the app's
    # _store_channel_attachment; self-contained so a failed callback install
    # cannot leave it undefined.
    attachment_sink = None
    try:
        from ..config import Settings
        from ..integrations.attachments import AttachmentStore

        _media_settings = Settings.load()
        _attachments = AttachmentStore(_media_settings)

        async def _store_attachment(channel: str, descriptor: dict[str, Any]) -> str | None:
            adapter = adapters.get(channel)
            download = getattr(adapter, "download", None)
            if download is None:
                return None
            try:
                raw, mime, name = await download(descriptor)
            except Exception as exc:
                _LOG.warning("channel attachment download failed",
                             extra={"channel": channel, "error": str(exc)})
                return None
            try:
                record = await _attachments.save(name, mime, base64.b64encode(raw).decode("ascii"))
            except Exception as exc:
                _LOG.warning("channel attachment store failed",
                             extra={"channel": channel, "error": str(exc)})
                return None
            attachment_id = record.get("id") if isinstance(record, dict) else None
            return str(attachment_id) if attachment_id else None

        attachment_sink = _store_attachment
    except Exception as exc:  # noqa: BLE001 - a gateway must start without media
        _LOG.warning("attachment sink not installed", extra={"error": type(exc).__name__})

    config = uvicorn.Config(api, host=host, port=port, log_level="info", lifespan="on")
    server = uvicorn.Server(config)

    async def _serve() -> None:
        await db.initialize()
        loop_task = asyncio.create_task(
            poll_loop(store, adapters, attachment_sink=attachment_sink), name="gateway-poll"
        )
        try:
            await server.serve()
        finally:
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)

    asyncio.run(_serve())


def token_path() -> str:
    home = os.getenv("LEO_HOME") or os.path.expanduser("~/.config/leo")
    return os.path.join(home, "gateway_token")


def _ensure_token() -> str:
    existing = os.getenv("LEO_GATEWAY_TOKEN")
    if existing:
        return existing
    path = token_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            stored = handle.read().strip()
        if stored:
            return stored
    except FileNotFoundError:
        pass
    import secrets

    fresh = secrets.token_urlsafe(32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(fresh)
    return fresh
