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
import logging
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..integrations.channels import default_adapters, polling_channels
from ..integrations.messaging import implements_polling
from ..integrations.secrets import credential
from ..integrations.transport_state import TransportState
from .sessions import SessionStore
from .store import GatewayStore

_LOG = logging.getLogger(__name__)

WORKER_IDLE_SLICE = 0.5


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


class FailRequest(BaseModel):
    admission_id: str
    error: str = ""
    retry: bool = True


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
) -> FastAPI:
    """Build the gateway's local API. Auth is required on every route."""
    api = FastAPI(title="LEO gateway", version="1")
    channels = adapters if adapters is not None else default_adapters()

    sessions = SessionStore()
    typing_tasks: dict[str, asyncio.Task[Any]] = {}

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
        return {"ok": True, "pid": os.getpid(), "channels": sorted(channels),
                "sessions": await sessions.stats(), **stats}

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
            await store.reclaim_expired()

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
        delivered, error = True, ""
        if request.answer.strip():
            adapter = channels.get(admission["channel"])
            if adapter is None:
                delivered, error = False, f"no adapter for {admission['channel']}"
            else:
                try:
                    await adapter.send(admission["subject"], request.answer)
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

    @api.post("/v1/admit")
    async def admit(payload: dict[str, Any],
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """Explicit ingress for pushes the gateway did not poll itself."""
        await auth(authorization)
        admission_id, created = await store.admit(
            channel=str(payload.get("channel") or "unknown"),
            subject=str(payload.get("subject") or ""),
            principal=str(payload.get("principal") or ""),
            payload=payload,
            request_id=str(payload.get("request_id") or ""),
        )
        return {"admission_id": admission_id, "created": created}

    return api


async def poll_loop(store: GatewayStore, adapters: dict[str, Any],
                    interval: float = 2.0) -> None:
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
                await store.admit(
                    channel=message.channel,
                    subject=message.conversation_ref or message.sender_id,
                    principal=message.sender_id,
                    payload={
                        "text": message.text,
                        "sender": message.sender,
                        "sender_id": message.sender_id,
                        "conversation_ref": message.conversation_ref,
                        "message_id": message.message_id,
                        "attachments": message.attachments,
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
    api = create_api(store, token=token, adapters=adapters)

    config = uvicorn.Config(api, host=host, port=port, log_level="info", lifespan="on")
    server = uvicorn.Server(config)

    async def _serve() -> None:
        await db.initialize()
        loop_task = asyncio.create_task(poll_loop(store, adapters), name="gateway-poll")
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
