"""FastAPI HTTP interface. Binds to localhost by default; no public exposure."""

from __future__ import annotations

import asyncio
import copy
import hmac
import json
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .. import db
from ..contracts import new_id, utcnow
from ..core import LeoApp
from ..integrations.attachments import AttachmentError
from ..integrations.gateway import GatewayAdmissionError
from ..tools import conversation_tools
from ..tools.conversation_tools import (
    ExportStorageError,
    redact_message_content,
    redact_preview_content,
    redact_preview_text,
)

_app: LeoApp | None = None


def get_app() -> LeoApp:
    if _app is None:
        raise RuntimeError("LEO app not initialized")
    return _app


# EventSource cannot send an Authorization header. A short-lived, single-use
# query token is issued by an authenticated POST and consumed on connect; it
# never grants access to any other route and cannot be replayed.
STREAM_TOKEN_TTL_SECONDS = 60.0
_stream_tokens: dict[str, tuple[float, str]] = {}


def issue_stream_token(principal: str) -> dict[str, Any]:
    now = time.monotonic()
    for token, (expires, _) in tuple(_stream_tokens.items()):
        if expires <= now:
            _stream_tokens.pop(token, None)
    if len(_stream_tokens) >= 256:
        oldest = min(_stream_tokens, key=lambda key: _stream_tokens[key][0])
        _stream_tokens.pop(oldest, None)
    token = secrets.token_urlsafe(32)
    _stream_tokens[token] = (now + STREAM_TOKEN_TTL_SECONDS, str(principal))
    return {"token": token, "expires_in": int(STREAM_TOKEN_TTL_SECONDS)}


def consume_stream_token(token: str | None) -> str | None:
    """Return the issuing principal once, or None for an invalid/replayed token."""
    if not token or not isinstance(token, str) or len(token) > 256:
        return None
    record = _stream_tokens.pop(token, None)
    if record is None or record[0] <= time.monotonic():
        return None
    return record[1]


# Tool output, gateway payloads and preview-serve metadata contain message
# text, results or preview credentials. None may appear in a feed a browser
# can read without a Bearer token. Every other visible topic is serialized in
# full, so its whole Event (payload, correlation_id and all other strings) is
# sanitized with the same preview-credential redaction as message content.
SENSITIVE_EVENT_PREFIXES = ("tool.", "gateway.", "preview.")


def is_sensitive_event(event: Any) -> bool:
    topic = str(getattr(event, "topic", "") or "")
    return topic.startswith(SENSITIVE_EVENT_PREFIXES)


def _redact_event(event: Any) -> Any:
    """Copy one visible event, sanitizing *every* browser-readable field.

    The SSE frame is the whole serialized Event, not just ``payload``: a raw
    preview token can sit in ``correlation_id``, ``source``, ``id`` or a nested
    value of any visible topic. Redacting only ``agent.message.payload`` left
    those copies readable. Dump the model, run the shared shape-independent
    sanitizer over the complete dump, and rebuild a copy. The stored event and
    the in-process bus are never mutated: a tool call still sees the genuine
    capability.
    """
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            original = dump(mode="json")
            redacted = redact_message_content(original)
            if redacted == original:
                return event
            copy_method = getattr(event, "model_copy", None)
            if callable(copy_method):
                return copy_method(update=redacted)
            return type(event).model_validate(redacted)
        except Exception:
            return None  # fail closed: an unsanitizable event is not visible
    redacted = redact_message_content(getattr(event, "__dict__", {}))
    if not redacted:
        return event
    try:
        clone = copy.copy(event)
        for key, value in redacted.items():
            setattr(clone, key, value)
        return clone
    except Exception:
        return None  # fail closed: an unsanitizable event is not visible


def visible_events(
    events: list[Any], *, authenticated: bool, redact: bool | None = None
) -> list[Any]:
    """Filter sensitive topics and sanitize every visible event for browsers.

    ``authenticated`` controls the sensitive-topic filter (SSE never exposes
    tool/gateway/preview topics, even to a stream token). ``redact`` controls
    whole-event preview-credential sanitization separately: the query-token
    SSE route always redacts, even when the token was issued by an API-token
    principal. The stored events on the bus are left untouched.
    """
    if authenticated and redact is not True:
        return events
    should_redact = not authenticated if redact is None else redact
    visible: list[Any] = []
    for event in events:
        if is_sensitive_event(event):
            continue
        if should_redact:
            # Every visible topic is serialized in full on the SSE feed, so
            # every visible topic needs the whole-event sanitizer.
            event = _redact_event(event)
            if event is None:
                continue
        visible.append(event)
    return visible


def _redact_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Sanitize every unauthenticated audit row string field.

    Audit rows store metadata as a JSON string. A token id is only a
    correlation digest now, but the key is still dropped entirely. Every other
    string (reason, resource, actor, nested metadata) is attacker-influenced,
    so the shared sanitizer rewrites those too.
    """
    clean = dict(row)
    metadata = clean.get("metadata")
    if isinstance(metadata, str) and "token_id" in metadata:
        try:
            parsed = json.loads(metadata)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and "token_id" in parsed:
            parsed.pop("token_id", None)
            clean["metadata"] = json.dumps(parsed, ensure_ascii=False)
    # Audit reasons/actors/resources are attacker-influenced strings: a raw
    # capability can be planted in any of them, not just token_id. Apply the
    # same shape-independent sanitizer as conversation reads and exports.
    sanitized = redact_preview_content(clean)
    return sanitized if isinstance(sanitized, dict) else clean


# Message content is the durable transcript. A tool result (or an old row)
# can embed the one-time preview URL and/or a raw ``token_id``; neither may be
# returned to a browser that has not authenticated with the API token. The
# sanitizer (imported from conversation_tools) copies and rewrites only the
# API response: the database and the model context keep the original value so
# subsequent tool calls still work. Exports call the same function, so a file
# on disk can never reveal more than this feed does.


def sanitize_message_payload(value: Any, request: Request) -> Any:
    """Only an authenticated API token may read unsanitized message content."""
    principal = str(getattr(getattr(request, "state", None), "principal", "") or "")
    if principal == "api-token":
        return value
    return redact_preview_content(value)


def sanitize_conversation_feed(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Redact a chat feed one message at a time.

    The redactor collects candidate credentials per call and scrubs every
    candidate from every string it is given. Handing it a whole conversation
    made that quadratic: a 2,688-message session accumulated hundreds of
    candidates and the request took over a minute. Each message is
    self-contained for credential purposes (a token surfaced in one turn is
    detected, and scrubbed, within that turn's strings; live preview tokens
    are registered process-wide), so redacting per message keeps the
    guarantee and the cost linear.
    """
    principal = str(getattr(getattr(request, "state", None), "principal", "") or "")
    if principal == "api-token":
        return payload
    messages = payload.get("messages")
    if isinstance(messages, list):
        payload = {
            **payload,
            "messages": [
                redact_preview_content(message) if isinstance(message, dict) else message
                for message in messages
            ],
        }
    meta = {key: value for key, value in payload.items() if key != "messages"}
    cleaned = redact_preview_content(meta)
    if isinstance(messages, list):
        cleaned = {**cleaned, "messages": payload["messages"]}
    return cleaned


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    global _app
    _app = LeoApp()
    await _app.start_background()
    yield
    await _app.shutdown()


api = FastAPI(title="LEO API", version="0.1.0", lifespan=lifespan)

# Browser dashboards are served from a configured dev origin; allow only those
# explicit origins and the headers the API actually needs.
from ..config import Settings as _Settings  # noqa: E402  (middleware needs settings at import)


def cors_origins(configured: Any) -> list[str]:
    """Never combine credentialed CORS with a wildcard origin."""
    origins = [str(item) for item in (configured or []) if str(item).strip() and str(item) != "*"]
    return origins


api.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(_Settings.load().server.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Leo-Token"],
    expose_headers=["Content-Type"],
)


# Export writes a file, so it is a POST (GET only lists existing files) with
# an explicit Origin check and a small per-principal quota. The counters live
# in durable kv: a second worker (or a restart) must see the same window, so a
# process-local dict would let a caller multiply the quota by rotating workers.
EXPORT_RATE_LIMIT = 12
EXPORT_RATE_WINDOW_SECONDS = 60.0
EXPORT_RATE_KV_PREFIX = "export-rate:v1:"
EXPORT_RATE_MAX_KEYS = 512
# Kept only so older callers/tests that cleared the old cache still work; it
# is deliberately not consulted by the rate limiter.
_export_requests: dict[str, list[float]] = {}


def _export_rate_kv_key(key: str) -> str:
    return EXPORT_RATE_KV_PREFIX + re.sub(r"[^A-Za-z0-9_.:-]", "_", str(key))[:120]


async def _check_export_rate(key: str) -> None:
    """Sliding-window rate limit backed by kv and shared across workers.

    Read, prune, decide and write happen in one ``BEGIN IMMEDIATE`` transaction
    so concurrent workers cannot both accept the last slot. Wall-clock stamps
    (not ``monotonic``) are required because workers have different clocks'
    origins. The clock is read *after* the write lock is held: a stamp captured
    before blocking on SQLite can be older than stamps already committed by
    faster workers, which makes the ``stamp <= now`` filter drop those slots
    and lets a concurrent burst multiply the quota. Stale keys and per-key
    stamp lists are pruned to stay bounded.
    """
    kv_key = _export_rate_kv_key(key)
    limited = False
    async with db.transaction() as conn:
        # BEGIN IMMEDIATE is already held here; only now is the window boundary
        # meaningful for every writer that queued for this lock.
        now = time.time()
        cutoff = now - EXPORT_RATE_WINDOW_SECONDS
        cutoff_at = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
        cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (kv_key,))
        row = await cursor.fetchone()
        stamps: list[float] = []
        if row is not None:
            try:
                stored = json.loads(row["value"])
                if isinstance(stored, list):
                    stamps = [float(item) for item in stored]
            except (TypeError, ValueError):
                stamps = []
        fresh = [stamp for stamp in stamps if cutoff < stamp <= now]
        if len(fresh) >= EXPORT_RATE_LIMIT:
            limited = True
        else:
            fresh.append(now)
        fresh = fresh[-max(1, EXPORT_RATE_LIMIT):]
        await conn.execute(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (kv_key, json.dumps(fresh, separators=(",", ":")), utcnow()),
        )
        cursor = await conn.execute(
            "SELECT key, updated_at FROM kv WHERE key LIKE ?", (EXPORT_RATE_KV_PREFIX + "%",)
        )
        rows = await cursor.fetchall()
        stale = {
            item["key"] for item in rows
            if item["key"] != kv_key and str(item["updated_at"]) < cutoff_at
        }
        for name in stale:
            await conn.execute("DELETE FROM kv WHERE key=?", (name,))
        remaining = [item for item in rows if item["key"] not in stale]
        if len(remaining) > EXPORT_RATE_MAX_KEYS:
            excess = len(remaining) - EXPORT_RATE_MAX_KEYS
            oldest = sorted(remaining, key=lambda item: str(item["updated_at"]))[:excess]
            for item in oldest:
                if item["key"] != kv_key:
                    await conn.execute("DELETE FROM kv WHERE key=?", (item["key"],))
    if limited:
        raise HTTPException(429, "conversation export rate limit exceeded")


def _require_safe_export_origin(request: Request) -> None:
    """Reject a browser POST from an origin the dashboard does not use."""
    origin = (request.headers.get("origin") or "").strip()
    if not origin:
        return  # non-browser clients rely on the bearer-token middleware
    allowed = {
        item.rstrip("/").lower()
        for item in cors_origins(_Settings.load().server.cors_origins)
    }
    if origin.rstrip("/").lower() not in allowed:
        raise HTTPException(403, "cross-origin export request rejected")


def _safe_detail(value: Any) -> Any:
    """Make validation error details encodable (lone surrogates must not 500)."""
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, dict):
        return {str(key): _safe_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_detail(item) for item in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value).encode("utf-8", "backslashreplace").decode("utf-8")


@api.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _safe_detail(exc.errors())})


def _validated_text(value: str, field: str) -> str:
    """Reject NUL, control characters and lone surrogates at the boundary."""
    if "\x00" in value:
        raise ValueError(f"{field} contains a NUL byte")
    if any(ord(char) < 32 and char not in "\n\t\r" for char in value):
        raise ValueError(f"{field} contains control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{field} contains invalid unicode") from None
    return value


def _positive_limit(value: int, *, cap: int, default: int) -> int:
    """Bound every list limit; SQLite LIMIT -1 would dump the whole table."""
    if value < 1:
        raise HTTPException(400, "limit must be >= 1")
    return min(value, cap)


class ChatRequest(BaseModel):
    message: str
    channel: str = "api"
    subject: str | None = None
    conversation_id: str | None = None
    attachments: list[Any] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _check_message(cls, value: str) -> str:
        return _validated_text(value, "message")

    @field_validator("conversation_id")
    @classmethod
    def _check_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) > 256 or any(ord(char) < 33 for char in value):
            raise ValueError("conversation_id must be a short, control-free identifier")
        return value


class ChatResponse(BaseModel):
    """Stable shape of POST /chat, including its error responses.

    A failed model turn is a normal HTTP response with ``status="error"``; a
    browser must never receive an opaque 500 (or a raw provider exception) for
    an empty/truncated model answer.
    """

    conversation_id: str
    content: str
    status: str
    provider: str = ""
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    memories_used: list[str] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls_pending: list[str] = Field(default_factory=list)
    simulated: bool = False
    # Router status is a structured report, not a plain status string.
    model_status: Any = None
    error: dict[str, Any] | None = None


class DelegateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=8000)
    agent_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    verify: bool = True

    @field_validator("objective")
    @classmethod
    def _check_objective(cls, value: str) -> str:
        return _validated_text(value, "objective")


class AttachmentUploadRequest(BaseModel):
    name: str
    mime: str
    data_base64: str
    conversation_id: str | None = None


class ChatStreamRequest(BaseModel):
    message: str
    channel: str = "api"
    subject: str | None = None
    conversation_id: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _check_message(cls, value: str) -> str:
        return _validated_text(value, "message")

    @field_validator("conversation_id")
    @classmethod
    def _check_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) > 256 or any(ord(char) < 33 for char in value):
            raise ValueError("conversation_id must be a short, control-free identifier")
        return value


class TaskRequest(BaseModel):
    objective: str
    plan: list[str] | None = None
    dependencies: list[str] | None = None
    owner_agent: str | None = None


class MemoryRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    category: str = "semantic"
    kind: str = "fact"
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.7


class ApprovalRequest(BaseModel):
    approved: bool
    resolver: str = "user"
    # The approval's own action string must be echoed back. A blind or stale
    # client that POSTs ``{approved: true}`` for whatever id it last polled
    # cannot drive a HIGH-risk decision; the dashboard sends the row's action.
    confirm_action: str | None = None


class OperatorToolRequest(BaseModel):
    # Only approval decisions are exposed on the privileged surface.
    name: str = "approval_resolve"
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolRunRequest(BaseModel):
    # Kept only so a mismatched value can be rejected explicitly: the URL
    # path is the sole tool identity. task_id/correlation_id scope the quota.
    name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None
    dry_run: bool = False
    task_id: str | None = None
    correlation_id: str | None = None
    max_tool_calls: int | None = Field(default=None, ge=1, le=1000)


class SkillRunRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChannelSendRequest(BaseModel):
    target: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=8000)
    subject: str | None = None
    idempotency_key: str | None = None

    @field_validator("text")
    @classmethod
    def _check_text(cls, value: str) -> str:
        return _validated_text(value, "text")


class ChannelEnableRequest(BaseModel):
    """Runtime enrollment payload: credential values by field name, admission
    allowlist and optional admission mode. Values are never echoed back."""
    values: dict[str, str] = Field(default_factory=dict)
    allowlist: list[str] = Field(default_factory=list)
    mode: str | None = None


class ChannelValidateRequest(BaseModel):
    """Offline, value-free field check used by the guided onboarding form."""
    values: dict[str, str] = Field(default_factory=dict)
    allowlist: list[str] = Field(default_factory=list)
    mode: str | None = None


class OnboardingStartRequest(BaseModel):
    """Optional TTL override; the server, never the client, builds setup URLs."""
    ttl_seconds: float | None = Field(default=None, ge=1.0, le=300.0)


class OnboardingRedeemRequest(BaseModel):
    code: str = Field(min_length=20, max_length=128)

    @field_validator("code")
    @classmethod
    def _check_code(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("setup code must be an opaque URL-safe token")
        return value


@api.middleware("http")
async def api_auth(request: Request, call_next):
    """Optional bearer-token auth.

    Loopback, local-only use needs no token. When LEO_API_TOKEN is configured,
    every interactive route requires it; provider webhooks authenticate with
    their own HMAC signatures instead.
    """
    token = os.getenv("LEO_API_TOKEN")
    path = request.url.path
    if path.startswith("/webhooks/") or path == "/health":
        request.state.principal = "webhook" if path.startswith("/webhooks/") else "health"
        return await call_next(request)
    # EventSource cannot set a Bearer header. The only exception is a valid,
    # unconsumed stream token; it is exchanged here exactly once and is valid
    # for this SSE request alone.
    if path == "/events/stream":
        principal = consume_stream_token(request.query_params.get("token"))
        if principal is None:
            return JSONResponse({"detail": "stream token required or expired"}, status_code=401)
        request.state.principal = principal
        return await call_next(request)
    if token:
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-leo-token", "")
        if not supplied or not hmac.compare_digest(str(supplied), token):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        request.state.principal = "api-token"
    else:
        request.state.principal = "local"
    return await call_next(request)


@api.get("/health")
async def health() -> dict[str, Any]:
    return await get_app().health()


@api.get("/status")
async def status(request: Request) -> dict[str, Any]:
    app = get_app()
    authenticated = getattr(request.state, "principal", "") in ("api-token", "event-stream")
    return {
        "health": await app.health(),
        "pending_approvals": await app.permissions.pending_approvals(),
        "recent_events": [
            e.model_dump() for e in visible_events(
                await app.events.history(limit=20), authenticated=authenticated
            )
        ],
        # Degraded channels have send credentials but cannot verify signed
        # ingress; never let the dashboard imply they are fully verified.
        "channel_verification": [
            item for item in _channel_verification() if item["status"] == "degraded"
        ],
    }


def _channel_verification() -> list[dict[str, Any]]:
    from ..integrations.channels import verification_report

    return verification_report()


def _channel_verification_for(channel: str) -> dict[str, Any] | None:
    return next(
        (item for item in _channel_verification() if item["channel"] == channel), None
    )


@api.post("/attachments")
async def upload_attachment(req: AttachmentUploadRequest) -> dict[str, Any]:
    """Store a validated local attachment and return its metadata."""
    try:
        record = await get_app().attachments.save(
            req.name, req.mime, req.data_base64, req.conversation_id
        )
    except AttachmentError as exc:
        raise HTTPException(400, str(exc))
    await db.execute(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (utcnow(), "api", "attachment.upload", record["id"], "low", "allowed",
         "validated attachment stored", json.dumps({"mime": record["mime"], "size": record["size"]})),
    )
    return get_app().attachments.public(record)


@api.get("/attachments/{attachment_id}/content")
async def attachment_content(attachment_id: str) -> FileResponse:
    store = get_app().attachments
    record = await store.get(attachment_id)
    if not record:
        raise HTTPException(404, "attachment not found")
    # Re-check containment, regular-file status, MIME and size at request time:
    # the file may have been swapped for a symlink or the row tampered with
    # since the upload passed validation.
    path = store.resolve_content(record)
    if path is None:
        raise HTTPException(404, "attachment content unavailable")
    return FileResponse(
        path, media_type=record["mime"], filename=str(record.get("name") or "attachment")
    )


@api.post("/chat")
async def chat(request: Request, req: ChatRequest) -> dict[str, Any]:
    if not req.message.strip() and not req.attachments and not req.attachment_ids:
        raise HTTPException(400, "empty message")
    try:
        combined = list(req.attachments) + [{"id": item} for item in req.attachment_ids]
        result = await get_app().ask(
            req.message, req.channel, req.subject, req.conversation_id, combined
        )
    except Exception as exc:  # noqa: BLE001 - boundary must never leak a 500/raw traceback
        code = str(getattr(exc, "code", None) or type(exc).__name__)[:80]
        result = ChatResponse(
            conversation_id=req.conversation_id or new_id("conv"),
            content=f"The model could not produce an answer for this turn. (code: {code})",
            status="error",
            error={"code": code, "message": f"model request failed ({code})"},
        ).model_dump(mode="json")
    if result.get("status") == "error":
        # Even on failure the client gets the same ChatResponse shape as a
        # successful turn, with preview URLs stripped for browser principals.
        return sanitize_message_payload(
            ChatResponse(**result).model_dump(mode="json", exclude_none=True), request
        )
    # The blocking transport carries the same model/tool output as /chat/stream
    # and /conversations. A browser without the API token must not be able to
    # lift a preview URL or raw capability out of `content`/`tool_results`.
    return sanitize_message_payload(result, request)


@api.post("/chat/stream")
async def chat_stream(request: Request, req: ChatStreamRequest) -> StreamingResponse:
    """Server-sent events for a live chat turn.

    Each event is a JSON object in an SSE ``data:`` line. The stream closes
    with ``run.finished`` (or ``error`` followed by ``run.finished``).
    """
    if not req.message.strip() and not req.attachments and not req.attachment_ids:
        raise HTTPException(400, "empty message")
    app = get_app()
    conversation_id = req.conversation_id
    if conversation_id and conversation_id in app._active_conversations:
        raise HTTPException(409, "a run is already active for this conversation")
    if conversation_id:
        app._active_conversations.add(conversation_id)

    async def generate():
        try:
            combined = list(req.attachments) + [{"id": item} for item in req.attachment_ids]
            async for event in app.ask_stream(
                req.message, req.channel, req.subject, req.conversation_id, combined
            ):
                # tool.finished echoes tool results, which can contain a live
                # preview URL. A browser without the API token must not be able
                # to lift the bearer token out of the live stream either.
                payload = sanitize_message_payload(event, request)
                yield "data: " + json.dumps(payload, default=str, ensure_ascii=False) + "\n\n"
        finally:
            if conversation_id:
                app._active_conversations.discard(conversation_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    try:
        outcome = await get_app().cancel_run(run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"cancelled": outcome == "cancelled", "status": outcome}


@api.get("/conversations")
async def list_conversations(request: Request, limit: int = 50) -> dict[str, Any]:
    """Conversation switcher rows for the chat sidebar.

    Only conversations with at least one message are listed (an unsent
    "new chat" never lands here), newest activity first.
    """
    from .. import db

    capped = max(1, min(limit, 200))
    rows = await db.fetchall(
        """
        SELECT c.id, c.channel, c.subject, c.created_at, c.updated_at,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
               (SELECT m.content FROM messages m WHERE m.conversation_id = c.id
                 AND m.role = 'user' ORDER BY m.id LIMIT 1) AS first_user,
               (SELECT m.content FROM messages m WHERE m.conversation_id = c.id
                 ORDER BY m.id DESC LIMIT 1) AS preview,
               (SELECT m.role FROM messages m WHERE m.conversation_id = c.id
                 ORDER BY m.id DESC LIMIT 1) AS preview_role
        FROM conversations c
        WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)
        ORDER BY c.updated_at DESC, c.rowid DESC
        LIMIT ?
        """,
        (capped,),
    )
    conversations = []
    for row in rows:
        record = dict(row)
        title = (record.pop("first_user") or "").strip() or "Untitled conversation"
        record["title"] = title[:80] + ("…" if len(title) > 80 else "")
        preview = (record.get("preview") or "").strip()
        record["preview"] = preview[:140] + ("…" if len(preview) > 140 else "")
        conversations.append(record)
    return sanitize_message_payload({"conversations": conversations}, request)


# The chat feed only renders a short excerpt per tool card, but a long
# session stores multi-hundred-KB tool payloads -- a 2,688-message
# conversation carried 8.8 MB of tool output, which made the endpoint (and
# the redaction pass over it) take minutes. Tool rows are bounded here,
# keeping the JSON shape the UI parses; the database and the export
# endpoints retain the full result.
MAX_FEED_TOOL_RESULT_CHARS = 16_000
_BOUNDED_TOOL_FIELDS = ("result", "output", "text", "content", "data")


def _bound_tool_feed_content(raw: Any) -> Any:
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    if isinstance(raw, str) and len(text) <= MAX_FEED_TOOL_RESULT_CHARS:
        return raw  # small rows ship verbatim; no parse/serialize round-trip
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text[:MAX_FEED_TOOL_RESULT_CHARS] + "… [truncated]" if len(text) > MAX_FEED_TOOL_RESULT_CHARS else raw
    if not isinstance(parsed, dict):
        if len(text) <= MAX_FEED_TOOL_RESULT_CHARS:
            return raw
        return json.dumps({
            "truncated": True,
            "preview": text[:MAX_FEED_TOOL_RESULT_CHARS] + "… [truncated]",
        }, ensure_ascii=False)
    changed = False
    for key in _BOUNDED_TOOL_FIELDS:
        value = parsed.get(key)
        if isinstance(value, str):
            if len(value) > MAX_FEED_TOOL_RESULT_CHARS:
                parsed[key] = value[:MAX_FEED_TOOL_RESULT_CHARS] + "… [truncated]"
                changed = True
            continue
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            if len(serialized) > MAX_FEED_TOOL_RESULT_CHARS:
                parsed[key] = {
                    "truncated": True,
                    "preview": serialized[:MAX_FEED_TOOL_RESULT_CHARS] + "… [truncated]",
                }
                changed = True
    if not changed:
        return raw
    parsed["truncated"] = True
    bounded = json.dumps(parsed, ensure_ascii=False)
    if len(bounded) > MAX_FEED_TOOL_RESULT_CHARS * 4:
        # Unknown shapes (lists of records, nested dumps) still must not
        # ship unbounded data to the feed.
        return json.dumps({
            "name": parsed.get("name") or parsed.get("tool"),
            "truncated": True,
            "preview": bounded[:MAX_FEED_TOOL_RESULT_CHARS] + "… [truncated]",
        }, ensure_ascii=False)
    return bounded


@api.get("/conversations/{conversation_id}")
async def conversation(
    request: Request, conversation_id: str, after_id: int = 0
) -> dict[str, Any]:
    """Chat feed for one conversation, optionally only rows after ``after_id``.

    ``after_id`` powers live appends: an open dashboard asks for newer rows
    when the busy event feed reports activity in the conversation, instead of
    re-downloading (and re-redacting) the whole history. ``latest_id`` is the
    highest row id at this instant, so an empty incremental page still tells
    the client where to resume.

    Tool rows are bounded (see ``MAX_FEED_TOOL_RESULT_CHARS``); user and
    assistant text is returned in full. Use the export endpoints when the
    complete tool payloads are needed.
    """
    from .. import db

    meta = await db.fetchone(
        "SELECT id, channel, subject, created_at, updated_at FROM conversations WHERE id=?",
        (conversation_id,),
    )
    if not meta:
        raise HTTPException(404, "conversation not found")
    rows = await db.fetchall(
        "SELECT id, role, content, created_at, metadata FROM messages "
        "WHERE conversation_id=? AND id>? ORDER BY id",
        (conversation_id, max(0, int(after_id))),
    )
    latest = await db.fetchone(
        "SELECT COALESCE(MAX(id), 0) AS latest_id FROM messages WHERE conversation_id=?",
        (conversation_id,),
    )
    messages = []
    for row in rows:
        record = dict(row)
        try:
            record["metadata"] = json.loads(record.get("metadata") or "{}")
        except (TypeError, ValueError):
            record["metadata"] = {}
        if record.get("role") == "tool":
            record["content"] = _bound_tool_feed_content(record.get("content"))
        messages.append(record)
    return sanitize_conversation_feed({
        "conversation_id": conversation_id,
        "channel": meta["channel"],
        "subject": meta["subject"],
        "created_at": meta["created_at"],
        "updated_at": meta["updated_at"],
        "latest_id": int(latest["latest_id"] if latest else 0),
        "messages": messages,
    }, request)


@api.get("/conversations/{conversation_id}/export")
async def export_conversation_manifest(request: Request, conversation_id: str) -> dict[str, Any]:
    """Read-only manifest: list existing exports; GET never writes a file.

    A caller discovers the stable filenames here, then uses POST with bounded
    ``limit``/``after_id`` to create or advance one of them.
    """
    from .. import db

    if not await db.fetchone("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)):
        raise HTTPException(404, "conversation not found")
    settings = get_app().settings
    root = Path(settings.resolve(settings.paths.exports)).resolve()
    prefix = f"conversation-{conversation_tools._slug(conversation_id)}"
    exports: list[dict[str, Any]] = []
    if root.is_dir():
        for item in sorted(root.glob(f"{prefix}.*")):
            if item.is_symlink() or not item.is_file():
                continue
            exports.append({
                "path": str(item),
                "bytes": item.stat().st_size,
                "format": item.suffix.lstrip("."),
            })
    return sanitize_message_payload({
        "conversation_id": conversation_id,
        "method": "POST",
        "exports": exports,
        "limit": {"default": conversation_tools.DEFAULT_LIMIT,
                  "max": conversation_tools.MAX_MESSAGES},
        "after_id": "message id cursor; responses report next_cursor",
    }, request)


@api.post("/conversations/{conversation_id}/export")
async def export_conversation(
    request: Request,
    conversation_id: str,
    format: str = "md",
    redact: bool = True,
    limit: int = conversation_tools.DEFAULT_LIMIT,
    after_id: str | None = None,
) -> dict[str, Any]:
    """Write one bounded transcript below ``settings.paths.exports``.

    Writing is a POST guarded by the configured Origin and a rate limit; GET
    is a read-only manifest. The HTTP route is a thin adapter over the same
    shared redaction, size-bound and atomic path-confinement core used by
    ``conversation_export`` and ``/export``. It never accepts an output path
    from the caller and reports the real file, byte count and message count;
    failures are surfaced, not simulated. ``limit``/``after_id`` allow a full
    transcript to be exported in bounded pages.
    """
    from .. import db

    _require_safe_export_origin(request)
    # Match the sibling conversation routes: a missing conversation is a 404,
    # while an invalid export option or an unsafe target is a client error.
    if not await db.fetchone("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)):
        raise HTTPException(404, "conversation not found")
    principal = str(getattr(request.state, "principal", "local") or "local")
    await _check_export_rate(principal)
    try:
        result = await conversation_tools.write_export(
            conversation_id=conversation_id,
            fmt=format,
            redact=redact,
            after_id=after_id,
            limit=limit,
            ctx={"settings": get_app().settings},
        )
    except ExportStorageError as exc:
        raise HTTPException(429, str(exc))
    except (ValueError, PermissionError, IsADirectoryError) as exc:
        raise HTTPException(400, str(exc))
    except OSError as exc:
        raise HTTPException(500, f"conversation export could not be written: {exc}")
    return sanitize_message_payload(result, request)


@api.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, Any]:
    """Remove a conversation and its messages. Audit-recorded, irreversible."""
    from .. import db

    existing = await db.fetchone("SELECT id FROM conversations WHERE id=?", (conversation_id,))
    if not existing:
        raise HTTPException(404, "conversation not found")
    attachments = await db.fetchall(
        "SELECT id, path FROM attachments WHERE conversation_id=?", (conversation_id,)
    )
    async with db.transaction() as conn:
        await conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        await conn.execute("DELETE FROM attachments WHERE conversation_id=?", (conversation_id,))
        await conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    # Files are outside the transaction: remove them best-effort, but only if
    # the stored path is still contained by the workspace root (a tampered DB
    # row must never turn a delete into arbitrary file removal).
    root = get_app().settings.root.resolve()
    for record in attachments:
        try:
            path = Path(record["path"]).resolve()
            if path.is_relative_to(root) and path.is_file():
                path.unlink()
        except (OSError, ValueError):
            pass
    await db.execute(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (utcnow(), "api", "conversation.delete", conversation_id, "low", "allowed",
         "dashboard conversation removal",
         json.dumps({"attachments_removed": len(attachments)})),
    )
    return {"deleted": conversation_id}


@api.get("/tasks")
async def list_tasks(status: str | None = None) -> dict[str, Any]:
    from ..contracts import TaskStatus

    if status is not None:
        try:
            status = TaskStatus(status).value
        except ValueError:
            raise HTTPException(400, f"unknown task status: {status!r}")
    return {"tasks": [t.model_dump(mode="json") for t in await get_app().tasks.list(status=status)]}


@api.post("/tasks")
async def create_task(req: TaskRequest) -> dict[str, Any]:
    task = await get_app().tasks.create(
        req.objective, plan=req.plan, dependencies=req.dependencies, owner_agent=req.owner_agent
    )
    return task.model_dump(mode="json")


@api.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = await get_app().tasks.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    return task.model_dump(mode="json")


@api.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    try:
        return {"cancelled": await get_app().tasks.cancel(task_id)}
    except KeyError:
        raise HTTPException(404, "task not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@api.post("/tasks/{task_id}/run")
async def run_task(task_id: str) -> dict[str, Any]:
    """Run one durable task through the registered default worker."""
    from ..contracts import TaskStatus

    app = get_app()
    task = await app.tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "task not found")
    finished = await app.tasks.run(task_id, app._default_task_worker)
    succeeded = finished.status is TaskStatus.COMPLETED
    return {
        "success": succeeded,
        "task": finished.model_dump(mode="json"),
    }


@api.get("/approvals")
async def approvals() -> dict[str, Any]:
    return {"approvals": await get_app().permissions.pending_approvals()}


def _require_decision_origin(request: Request) -> None:
    """Approval decisions are an operator surface; reject drive-by browsers.

    Any page open in a local browser can POST to the loopback bind; CORS only
    blocks reading the response, not the side effect. Cross-origin fetches are
    rejected via ``Sec-Fetch-Site`` and the ``Origin`` allowlist. Non-browser
    clients (CLI, tests, agents) send neither header and keep working.
    """
    site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if site and site not in ("same-origin", "none"):
        raise HTTPException(403, "cross-site approval decision rejected")
    origin = (request.headers.get("origin") or "").strip().rstrip("/").lower()
    if not origin:
        return
    allowed = {
        item.rstrip("/").lower()
        for item in cors_origins(_Settings.load().server.cors_origins)
    }
    if origin in allowed:
        return
    host = (request.headers.get("host") or "").strip().lower()
    if host and origin in (f"http://{host}", f"https://{host}"):
        return
    raise HTTPException(403, "cross-origin approval decision rejected")


@api.post("/approvals/{approval_id}/resolve")
async def resolve_approval(approval_id: str, req: ApprovalRequest, request: Request) -> dict[str, Any]:
    from .. import db

    _require_decision_origin(request)
    row = await db.fetchone("SELECT action FROM approvals WHERE id=?", (approval_id,))
    if not row:
        raise HTTPException(404, "approval not found")
    if not req.confirm_action or str(req.confirm_action) != str(row["action"]):
        raise HTTPException(
            409,
            f"approval confirmation required: resend with confirm_action={row['action']!r}",
        )
    # The requester does not get to choose the identity recorded as resolver.
    try:
        decision = await get_app().permissions.resolve_approval(
            approval_id, req.approved, getattr(request.state, "principal", "local")
        )
    except KeyError:
        raise HTTPException(404, "approval not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return decision.model_dump(mode="json")



@api.post("/operator/tools/{name}/execute")
async def execute_operator_tool(
    name: str, req: OperatorToolRequest, request: Request
) -> dict[str, Any]:
    """Trusted human/operator surface (never reachable by chat or agents)."""
    from ..contracts import ToolCall
    from ..security import ApprovalRequired

    principal = str(getattr(request.state, "principal", "") or "")
    if principal not in ("local", "api-token"):
        raise HTTPException(403, "operator tools are not available to this principal")
    if name != "approval_resolve":
        raise HTTPException(404, "unknown operator tool")
    app = get_app()
    # The executor's authenticated actor is the API principal; callers cannot
    # choose who resolves an approval in the request body.
    call = ToolCall(
        name=name, arguments=req.arguments,
        actor=f"operator:{principal}", task_id=req.arguments.get("task_id"),
        correlation_id=req.arguments.get("correlation_id"),
    )
    try:
        result = await app.operator_executor.execute(call)
    except ApprovalRequired as exc:
        raise HTTPException(409, f"approval required: {exc.approval_id}")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return result.model_dump(mode="json")


@api.post("/agents/delegate")
async def delegate_agent(req: DelegateRequest) -> dict[str, Any]:
    """Delegate one objective with the same engine used by agent tools."""
    app = get_app()
    try:
        result = await app.agent_engine.delegate(
            req.objective, agent_id=req.agent_id, context=dict(req.context),
            verify=req.verify,
        )
    except (KeyError, LookupError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    outcome = result[0] if isinstance(result, (list, tuple)) else result
    success = bool(getattr(outcome, "success", False))
    return {
        "success": success,
        "task_id": str(getattr(outcome, "task_id", "") or ""),
        "error": getattr(outcome, "error", None) if not success else None,
    }


@api.get("/agents")
async def list_agents() -> dict[str, Any]:
    return {"agents": [s.model_dump(mode="json") for s in await get_app().agent_registry.list()]}


@api.get("/memory")
async def list_memories(limit: int = 20) -> dict[str, Any]:
    records = await get_app().memory.list(limit=_positive_limit(limit, cap=100, default=20))
    return {"memories": [m.model_dump(mode="json") for m in records]}


@api.get("/tools")
async def list_tools() -> dict[str, Any]:
    return {"tools": [s.model_dump(mode="json") for s in get_app().registry.list_specs()]}


@api.post("/tools/{name}/execute")
async def execute_tool(name: str, req: ToolRunRequest, request: Request) -> dict[str, Any]:
    from ..contracts import ToolCall
    from ..security import ApprovalRequired

    def safe(value: Any) -> Any:
        """Errors and results are browser-readable unless the caller has the token."""
        return sanitize_message_payload(value, request)

    app = get_app()
    if req.name is not None and req.name != name:
        raise HTTPException(
            400, safe(f"body tool name {req.name!r} does not match path {name!r}")
        )
    # The HTTP export route is not the only way to write an export: the tool
    # executor is reachable directly through this endpoint. Charge the same
    # durable, per-principal quota here so the two paths share one budget.
    if name == "conversation_export":
        principal = str(getattr(request.state, "principal", "local") or "local")
        await _check_export_rate(principal)
    call = ToolCall(
        name=name,  # path is the only identity used for risk, permissions and audit
        arguments=req.arguments, actor="api", dry_run=req.dry_run,
        task_id=req.task_id, correlation_id=req.correlation_id,
        max_tool_calls=req.max_tool_calls,
    )
    try:
        result = await app.executor.execute(call, approval_id=req.approval_id)
    except ApprovalRequired as exc:
        raise HTTPException(409, safe(f"approval required: {exc.approval_id}"))
    except KeyError:
        raise HTTPException(404, "unknown tool")
    except ValueError as exc:
        raise HTTPException(400, safe(str(exc)))
    # The executor records and returns an unknown-tool result rather than
    # raising; mirror that audited outcome as a proper HTTP 404.
    if result.decision == "unknown_tool":
        raise HTTPException(404, "unknown tool")
    payload = safe(result.model_dump(mode="json"))
    if result.decision == "rate_limited":
        quota = result.metadata.get("quota") or {}
        retry_after = max(1, int(float(quota.get("reset_at", time.time())) - time.time()))
        return JSONResponse(
            payload, status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    return payload

@api.get("/skills")
async def list_skills() -> dict[str, Any]:
    return {"skills": [s.model_dump(mode="json") for s in await get_app().skills.list()]}


@api.post("/skills/{name}/run")
async def run_skill(name: str, req: SkillRunRequest) -> dict[str, Any]:
    try:
        result = await get_app().skills.run(name, req.arguments)
    except KeyError:
        raise HTTPException(404, "skill not found")
    except Exception as exc:
        raise HTTPException(400, str(exc))
    if getattr(result, "error", None) == "unknown skill" or (
        isinstance(result, dict) and result.get("error") == "unknown skill"
    ):
        raise HTTPException(404, "skill not found")
    return {"result": result}


@api.post("/memory")
async def add_memory(req: MemoryRequest) -> dict[str, Any]:
    from ..contracts import MemoryCategory, MemoryKind

    if not req.content.strip():
        raise HTTPException(400, "memory content must not be empty")
    try:
        category = MemoryCategory(req.category)
        kind = MemoryKind(req.kind)
    except ValueError as exc:
        # Enum conversion errors are client mistakes; without this guard
        # FastAPI turns them into an opaque HTTP 500.
        raise HTTPException(400, f"invalid memory category or kind: {exc}")
    record = await get_app().remember(
        req.content,
        category=category,
        kind=kind,
        tags=req.tags,
        confidence=req.confidence,
    )
    return record.model_dump(mode="json")


@api.get("/memory/search")
async def search_memory(q: str, limit: int = 10) -> dict[str, Any]:
    results = await get_app().search.search(q, limit=_positive_limit(limit, cap=100, default=10))
    return {"results": [m.model_dump(mode="json") for m in results]}


@api.get("/models/usage")
async def model_usage() -> dict[str, Any]:
    return {"usage": await get_app().router.usage_report()}


@api.get("/events/durability")
async def event_durability() -> dict[str, Any]:
    """Durable event outbox/delivery statistics."""
    return await get_app().events.stats()


@api.get("/events/subscriptions")
async def event_subscriptions() -> dict[str, Any]:
    """Named, restart-safe event subscriptions and their cursors."""
    return {"subscriptions": await get_app().events.durable_subscriptions()}


@api.post("/events/durability/drain")
async def drain_event_durability() -> dict[str, Any]:
    """Flush the outbox, replay missed events and retry failed deliveries."""
    return await get_app().events.drain()


@api.post("/events/stream-token")
async def event_stream_token(request: Request) -> dict[str, Any]:
    """Issue a 60s single-use token for browser EventSource connections."""
    principal = str(getattr(request.state, "principal", "local") or "local")
    if principal not in ("local", "api-token"):
        raise HTTPException(403, "stream tokens require an authenticated API principal")
    return issue_stream_token(principal)


@api.get("/events")
async def events(
    request: Request,
    topic: str | None = None,
    topic_prefix: str | None = None,
    source: str | None = None,
    correlation_id: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    direction: str = "before",
) -> dict[str, Any]:
    """Keyset-paginated durable events with a real, filter-independent total."""
    if direction not in ("before", "after"):
        raise HTTPException(400, "direction must be 'before' or 'after'")
    try:
        page = await get_app().events.page(
            topic=topic, topic_prefix=topic_prefix, source=source,
            correlation_id=correlation_id,
            limit=_positive_limit(limit, cap=1000, default=50),
            cursor=cursor, direction=direction,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    authenticated = getattr(request.state, "principal", "") in ("api-token", "event-stream")
    return {
        "events": [
            event.model_dump(mode="json")
            for event in visible_events(page.events, authenticated=authenticated)
        ],
        "total": page.total,
        "next_cursor": page.next_cursor,
        "prev_cursor": page.prev_cursor,
        "has_more": page.has_more,
    }


@api.get("/events/stream")
async def events_stream(
    request: Request, max_events: int = 0, max_seconds: float = 3600.0
) -> StreamingResponse:
    """Live SSE feed of durable events, gated by a single-use stream token.

    Frames are ``data: <Event JSON>``; an empty comment every ~15s keeps proxy
    connections alive. Tool-output and gateway-payload topics are never sent
    on this feed, and every visible event has preview credentials redacted in
    all serialized fields (payload, correlation_id, ids and nested values),
    regardless of the principal that minted the query token. The bus keeps the
    unredacted event for in-process tool calls.
    ``max_events``/``max_seconds`` are bounded client limits.
    """
    if getattr(request.state, "principal", "") not in ("local", "api-token", "event-stream"):
        raise HTTPException(401, "stream token required")
    bus = get_app().events
    # A stream token travels in a URL query string: browsers, proxies and
    # history can all read it, and it is not a Bearer header. Redact preview
    # credentials even when the token was minted by an api-token principal;
    # that principal's private view remains available on /status and /events.
    event_budget = max(0, min(int(max_events), 10000))
    deadline = time.monotonic() + max(1.0, min(float(max_seconds), 86400.0))

    async def generate():
        # Start at the current tail: a fresh dashboard is not flooded by the
        # entire historical table, but nothing emitted after connect is lost.
        latest = await bus.page(limit=1, direction="before")
        cursor = latest.prev_cursor
        seen: set[str] = set()
        emitted = 0
        keepalive_at = time.monotonic() + 15.0
        while True:
            if await request.is_disconnected() or time.monotonic() >= deadline:
                return
            page = await bus.page(limit=100, cursor=cursor, direction="after")
            for event in visible_events(page.events, authenticated=False, redact=True):
                if event.id in seen:
                    continue
                seen.add(event.id)
                if len(seen) > 10000:
                    seen.clear()
                yield "data: " + event.model_dump_json() + "\n\n"
                emitted += 1
                keepalive_at = time.monotonic() + 15.0
                if event_budget and emitted >= event_budget:
                    return
            if page.events:
                cursor = page.prev_cursor
            now = time.monotonic()
            if now >= keepalive_at:
                yield ": keep-alive\n\n"
                keepalive_at = now + 15.0
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api.get("/metrics")
async def metrics() -> dict[str, Any]:
    return await get_app().metrics()


@api.get("/integrations")
async def integrations() -> dict[str, Any]:
    from .. import db

    rows = await db.fetchall("SELECT * FROM integrations ORDER BY name")
    return {"integrations": [dict(r) for r in rows]}


@api.get("/integrations/policies")
async def integration_policies() -> dict[str, Any]:
    """Current inbound channel policies plus their durable version."""
    app = get_app()
    return {
        "version": await app.gateway.policy_version(),
        "policies": app.gateway.admission.snapshot(),
    }


class PolicyRefreshRequest(BaseModel):
    expected_version: int | None = None


@api.post("/integrations/policies/refresh")
async def refresh_integration_policies(req: PolicyRefreshRequest | None = None) -> dict[str, Any]:
    """Reload durable policies (optimistically versioned) and audit the change."""
    from ..integrations.admission import PolicyVersionError

    expected = req.expected_version if req is not None else None
    try:
        return await get_app().gateway.refresh_policies(expected_version=expected)
    except PolicyVersionError as exc:
        raise HTTPException(409, str(exc))


def _channel_credential_fields(details: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the credential-field shapes a channel spec may expose.

    Only metadata (names, labels, required/configured flags) is read here —
    never a credential value. Kept defensive so the offline verify endpoint
    tolerates either a list of field dicts or a name→spec mapping.
    """
    for key in ("credentials", "fields", "required_fields", "config_fields"):
        value = details.get(key)
        if isinstance(value, list):
            return [field for field in value if isinstance(field, dict)]
        if isinstance(value, dict):
            normalized: list[dict[str, Any]] = []
            for name, spec in value.items():
                if isinstance(spec, dict):
                    normalized.append({"name": name, **spec})
                else:
                    normalized.append({"name": name, "configured": bool(spec)})
            return normalized
    return []


@api.get("/integrations/catalog")
async def integration_catalog() -> dict[str, Any]:
    from ..integrations.enrollment import list_channels

    return {"channels": await list_channels(gateway=get_app().gateway)}


@api.get("/integrations/catalog/{channel}")
async def integration_catalog_channel(channel: str) -> dict[str, Any]:
    from ..integrations.enrollment import EnrollmentError, channel_details

    try:
        return await channel_details(channel, gateway=get_app().gateway)
    except EnrollmentError as exc:
        raise HTTPException(404, str(exc))


@api.post("/integrations/{channel}/enable")
async def enable_integration_channel(channel: str, req: ChannelEnableRequest) -> dict[str, Any]:
    from ..integrations.enrollment import EnrollmentError, enable_channel

    try:
        return await enable_channel(
            channel,
            values=req.values,
            allowlist=req.allowlist,
            mode=req.mode,
            gateway=get_app().gateway,
        )
    except EnrollmentError as exc:
        # Field-level guidance lets the wizard attach the problem to the input;
        # neither the message nor the errors ever contain a submitted value.
        raise HTTPException(
            400,
            {"message": exc.message, "errors": exc.field_errors} if exc.field_errors else exc.message,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@api.post("/integrations/{channel}/disable")
async def disable_integration_channel(channel: str) -> dict[str, Any]:
    from ..integrations.enrollment import EnrollmentError, disable_channel

    try:
        return await disable_channel(channel, gateway=get_app().gateway)
    except EnrollmentError as exc:
        raise HTTPException(400, str(exc))


@api.post("/integrations/{channel}/verify")
async def verify_integration_channel(channel: str) -> dict[str, Any]:
    """Offline-only configuration check. Never attempts live delivery."""
    from ..integrations.enrollment import EnrollmentError, channel_details

    try:
        details = await channel_details(channel, gateway=get_app().gateway)
    except EnrollmentError as exc:
        raise HTTPException(404, str(exc))

    from ..integrations.enrollment import ALTERNATIVE_GROUPS

    fields = _channel_credential_fields(details)
    by_name = {str(field.get("name")): field for field in fields}
    missing: list[str] = []
    for field in fields:
        if field.get("required", True) and not field.get("configured", False):
            missing.append(field.get("label") or field.get("name") or "credential")
    # Some channels (Discord) accept any one of several send credentials.
    # Requiring every field would be wrong; accepting none would report
    # ready=true for a channel with zero credentials.
    for group in ALTERNATIVE_GROUPS.get(channel, ()):
        if not any(by_name.get(name, {}).get("configured") for name in group):
            missing.append("one of: " + ", ".join(group))
    verification = _channel_verification_for(channel)
    degraded = bool(verification and verification.get("status") == "degraded")
    return {
        "channel": channel,
        "ready": not missing,
        "missing": missing,
        "webhook": bool(details.get("webhook") or details.get("supports_webhook"))
        and not degraded,
        "poll": bool(details.get("poll") or details.get("supports_polling")),
        "policy": details.get("policy"),
        "verification": verification,
        "degraded": degraded,
    }


@api.post("/integrations/{channel}/validate")
async def validate_integration_channel(channel: str, req: ChannelValidateRequest) -> dict[str, Any]:
    """Field-level, offline validation for the guided enrollment form.

    The request values are checked and discarded; only field names and safe
    remediation messages can appear in the response.
    """
    from ..integrations.channels import ALL_ADAPTERS
    from ..integrations.enrollment import EnrollmentError, channel_details
    from ..integrations.onboarding import validate_values

    if channel not in ALL_ADAPTERS:
        raise HTTPException(404, "unknown channel")
    values: dict[str, Any] = dict(req.values)
    if req.allowlist:
        values["allowlist"] = ", ".join(str(item) for item in req.allowlist)
    if req.mode is not None:
        values["mode"] = req.mode
    errors = validate_values(channel, values)
    try:
        details = await channel_details(channel, gateway=get_app().gateway)
    except EnrollmentError as exc:
        raise HTTPException(404, exc.message)
    missing = [
        field["label"] or field["name"]
        for field in details["fields"]
        if field["required"]
        and not field["configured"]
        and not str(values.get(field["name"], "") or "").strip()
    ]
    return {
        "channel": channel,
        "valid": not errors and not missing,
        "errors": errors,
        "missing": missing,
        "offline": True,
        "note": "This check validates fields only. It never links a device or sends a test message.",
    }


# --- Guided onboarding (local setup QR, never a WhatsApp device link) -------

_ONBOARDING_HEADER = "x-leo-onboarding"
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _onboarding_client(request: Request) -> str:
    """CSRF, origin and loopback guard for state-changing onboarding calls.

    A custom header cannot be sent cross-origin without a permitted CORS
    preflight, the Origin must match exactly, and (without a bearer token) the
    request must arrive on a loopback host. The returned key is a rate-limit
    identifier derived from the peer address, never from a submitted secret.
    """
    from ..integrations.onboarding import origin_allowed, safe_base_url

    principal = str(getattr(request.state, "principal", "local"))
    if principal != "api-token" and request.headers.get(_ONBOARDING_HEADER) != "1":
        raise HTTPException(403, "onboarding writes require the X-Leo-Onboarding header")
    base = safe_base_url(request_url=str(request.url))
    if not origin_allowed(request.headers.get("origin"), base_url=base):
        raise HTTPException(403, "cross-origin onboarding request rejected")
    host = (urlsplit(f"http://{request.headers.get('host', '')}").hostname or "").lower()
    if principal != "api-token" and host not in _LOCAL_HOSTS:
        raise HTTPException(403, "onboarding is available only from this computer")
    client = request.client.host if request.client else "local"
    return f"{principal}:{client[:64]}"


@api.get("/onboarding/{channel}/guide")
async def onboarding_guide(channel: str, request: Request) -> dict[str, Any]:
    """Plain-language steps, guided fields and honest transport status."""
    from ..integrations.enrollment import EnrollmentError, channel_details
    from ..integrations.onboarding import safe_base_url

    try:
        details = await channel_details(channel, gateway=get_app().gateway)
    except EnrollmentError as exc:
        raise HTTPException(404, exc.message)
    result = dict(details)
    result["qr_setup"] = {
        "available": channel == "whatsapp",
        "method": "local-setup" if channel == "whatsapp" else None,
        "start_path": f"/onboarding/{channel}/start" if channel == "whatsapp" else None,
        "device_link": details["device_link"],
        "note": (
            details["device_link"].get("reason")
            or "This channel has no local pairing flow."
        ),
    }
    if details["webhook"]:
        result["callback_url"] = (
            f"{safe_base_url(request_url=str(request.url))}/webhooks/{channel}"
        )
    return result


@api.post("/onboarding/{channel}/start")
async def start_channel_onboarding(
    channel: str, request: Request, req: OnboardingStartRequest | None = None
) -> dict[str, Any]:
    """Issue a one-time setup code and render its QR entirely on this host."""
    from ..integrations.onboarding import (
        SETUP_TOKEN_TTL_SECONDS,
        OnboardingError,
        safe_base_url,
        start_onboarding,
    )

    client = _onboarding_client(request)
    try:
        return await start_onboarding(
            channel,
            base_url=safe_base_url(request_url=str(request.url)),
            client=client,
            ttl_seconds=req.ttl_seconds if req is not None else SETUP_TOKEN_TTL_SECONDS,
        )
    except OnboardingError as exc:
        raise HTTPException(exc.status_code, exc.message)


@api.get("/onboarding/sessions/{session_id}")
async def onboarding_session_status(session_id: str) -> dict[str, Any]:
    """Read-only status: expiry, state transitions and the linked=false truth."""
    from ..integrations.onboarding import session_status

    result = await session_status(session_id)
    if result is None:
        raise HTTPException(404, "no such setup session")
    return result


@api.post("/onboarding/redeem")
async def redeem_onboarding_code(
    req: OnboardingRedeemRequest, request: Request
) -> dict[str, Any]:
    """Confirm a setup code once. This confirms local setup, not a device link."""
    from ..integrations.onboarding import OnboardingError, redeem_setup_code

    client = _onboarding_client(request)
    try:
        return await redeem_setup_code(
            req.code, client=client, origin=request.headers.get("origin")
        )
    except OnboardingError as exc:
        raise HTTPException(exc.status_code, exc.message)


# Self-contained confirmation page for the locally rendered QR. It never sees
# a credential: the QR carries only a one-time opaque code in its URL fragment,
# which the browser exchanges in a single POST back to this host.
_PAIRING_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; form-action 'none'">
<title>LEO local setup</title>
<style>
body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; background: #f5f5f7; color: #1d1d1f; }
main { max-width: 34rem; margin: 4rem auto; padding: 2rem; background: #fff;
       border-radius: 14px; box-shadow: 0 2px 16px rgba(0,0,0,.08); }
h1 { font-size: 1.4rem; margin-top: 0; }
button { background: #0071e3; color: #fff; border: 0; border-radius: 8px;
         padding: .7rem 1.2rem; font-size: 1rem; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
input { width: 100%; box-sizing: border-box; padding: .5rem; margin: .5rem 0 1rem; }
#status { margin-top: 1rem; white-space: pre-wrap; }
.note { color: #6e6e73; }
</style>
</head>
<body>
<main>
<h1>LEO local setup</h1>
<p class="note">This code confirms setup on this computer. It is not a WhatsApp
login, does not link a device, and expires quickly. Do not share it.</p>
<label for="token">API token (only if this LEO requires one):</label>
<input id="token" type="password" autocomplete="off" spellcheck="false">
<button id="confirm" type="button" disabled>Confirm setup code</button>
<p id="status" role="status"></p>
</main>
<script>
(function () {
  var status = document.getElementById('status');
  var button = document.getElementById('confirm');
  var token = document.getElementById('token');
  var match = /(?:^|&)code=([A-Za-z0-9_-]{20,128})(?:&|$)/.exec(location.hash.replace(/^#/, ''));
  if (!match) {
    status.textContent = 'No setup code found in this link. Start a new setup in LEO.';
    return;
  }
  button.disabled = false;
  button.addEventListener('click', function () {
    button.disabled = true;
    status.textContent = 'Confirming…';
    var headers = { 'Content-Type': 'application/json', 'X-Leo-Onboarding': '1' };
    if (token.value) { headers.Authorization = 'Bearer ' + token.value; }
    fetch('/onboarding/redeem', {
      method: 'POST', headers: headers, body: JSON.stringify({ code: match[1] })
    }).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) { throw new Error(body.detail || 'Setup could not be confirmed.'); }
        status.textContent = 'Local setup confirmed. No WhatsApp device is linked. ' +
          'Continue with the Cloud API steps in LEO.';
      });
    }).catch(function (error) {
      status.textContent = error.message;
      button.disabled = false;
    });
  });
})();
</script>
</body>
</html>
"""


@api.get("/onboarding/pair")
async def onboarding_pair_page() -> HTMLResponse:
    return HTMLResponse(_PAIRING_PAGE)


@api.post("/integrations/{channel}/send")
async def send_integration_channel(channel: str, req: ChannelSendRequest) -> dict[str, Any]:
    """Send through the gateway, preserving its permission and audit gates."""
    from ..integrations.gateway import GatewayPermissionError

    app = get_app()
    try:
        outcome = await app.gateway.send(
            channel, req.target, req.text, subject=req.subject,
            idempotency_key=req.idempotency_key,
        )
    except GatewayPermissionError as exc:
        raise HTTPException(409, f"send requires approval: {exc.approval_id or exc}")
    except Exception as exc:  # unknown channel and adapter errors are client errors
        raise HTTPException(400, f"channel send failed: {type(exc).__name__}: {exc}")
    delivered = bool(outcome.get("success", False)) if isinstance(outcome, dict) else False
    return {
        "status": "sent" if delivered else "blocked",
        "delivered": delivered,
        "error": None if delivered else (outcome.get("error") or "delivery unconfirmed"),
        "result": outcome,
    }


@api.post("/webhooks/{channel}")
async def webhook(channel: str, request: Request) -> JSONResponse:
    from ..integrations.channels import webhook_channels
    from ..integrations.messaging import WebhookVerificationError

    if channel not in webhook_channels():
        raise HTTPException(404, "unknown channel")
    # Verify the signature over the exact bytes sent by the provider:
    # re-serializing parsed JSON breaks real HMAC/Ed25519 signatures.
    raw_body = await request.body()
    if len(raw_body) > 1_000_000:
        raise HTTPException(413, "webhook body too large")
    try:
        payload = json.loads(raw_body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "invalid webhook JSON")

    # Slack/Feishu-style subscription handshakes expect the challenge echoed in
    # the raw HTTP response. Verify the signature FIRST, then answer before the
    # gateway normalization path (a challenge is not a user message).
    if (
        isinstance(payload, dict)
        and payload.get("type") == "url_verification"
        and isinstance(payload.get("challenge"), str)
    ):
        adapter = get_app().gateway.adapters.get(channel)
        if adapter is None or not adapter.verify_webhook(
            payload, dict(request.headers), raw_body=raw_body
        ):
            raise HTTPException(401, "webhook verification failed")
        return JSONResponse({"challenge": payload["challenge"][:512]})

    try:
        result = await get_app().gateway.receive_webhook(
            channel, payload, dict(request.headers), raw_body=raw_body
        )
    except WebhookVerificationError as exc:
        raise HTTPException(401, str(exc))
    except GatewayAdmissionError as exc:
        # Signature valid, sender not admitted: do not persist or forward.
        raise HTTPException(403, "sender not admitted by channel policy")
    except Exception as exc:
        raise HTTPException(400, f"webhook rejected: {type(exc).__name__}")
    return JSONResponse({"status": "accepted", "messages": len(result)})


@api.post("/channels/{channel}/poll")
async def poll_channel(channel: str) -> dict[str, Any]:
    """Pull inbound messages from a poll-capable channel (IMAP, IRC, ...)."""
    from ..integrations.channels import polling_channels
    from ..integrations.gateway import GatewayError

    if channel not in polling_channels():
        raise HTTPException(404, "channel does not support polling")
    try:
        messages = await get_app().gateway.poll(channel)
    except GatewayError as exc:
        raise HTTPException(400, str(exc))
    return {"status": "polled", "messages": len(messages)}



class EvaluationRequest(BaseModel):
    title: str
    rationale: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class AdoptionRequest(BaseModel):
    tests_passed: bool
    approver: str


class ScheduleRequest(BaseModel):
    name: str
    seconds: float
    objective: str


class IntegrationEvaluateRequest(BaseModel):
    candidate: dict[str, Any]


@api.get("/evaluation")
async def evaluation() -> dict[str, Any]:
    app = get_app()
    return {
        "metrics": await app.evaluation.report(),
        "tasks": await app.evaluation.task_metrics(),
        "tools": await app.evaluation.tool_reliability(),
        "proposals": await app.evaluation.proposals(),
    }


@api.post("/evaluation/proposals")
async def propose_improvement(req: EvaluationRequest) -> dict[str, Any]:
    return await get_app().evaluation.propose(req.title, req.rationale, req.evidence)


@api.post("/evaluation/proposals/{proposal_id}/adopt")
async def adopt_improvement(proposal_id: str, req: AdoptionRequest) -> dict[str, Any]:
    try:
        return await get_app().evaluation.adopt(proposal_id, req.tests_passed, req.approver)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@api.get("/schedules")
async def schedules() -> dict[str, Any]:
    return {"schedules": await get_app().scheduler.list_schedules()}


@api.post("/schedules")
async def create_schedule(req: ScheduleRequest) -> dict[str, Any]:
    return await get_app().scheduler.register_interval(req.name, req.seconds, req.objective)


@api.delete("/schedules/{name}")
async def delete_schedule(name: str) -> dict[str, Any]:
    """Remove a schedule after verifying it with the scheduler."""
    from .. import db
    from ..contracts import utcnow

    app = get_app()
    try:
        await app.scheduler.get(name)
    except KeyError:
        raise HTTPException(404, "schedule not found")
    await db.execute("DELETE FROM schedules WHERE name=?", (name,))
    await db.execute(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (utcnow(), "api", "schedule.delete", name, "medium", "allowed",
         "dashboard schedule removal", json.dumps({"name": name})),
    )
    await app.events.emit("schedule.deleted", name=name)
    return {"deleted": name}


@api.get("/gateway/sessions")
async def gateway_sessions() -> dict[str, Any]:
    return {"sessions": await get_app().gateway.sessions()}


@api.get("/audit")
async def audit(request: Request, limit: int = 100) -> dict[str, Any]:
    from .. import db

    rows = await db.fetchall(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
        (_positive_limit(limit, cap=1000, default=100),),
    )
    authenticated = getattr(request.state, "principal", "") in ("api-token", "event-stream")
    return {
        "audit": [
            row if authenticated else _redact_audit_row(row)
            for row in (dict(item) for item in rows)
        ]
    }


@api.post("/integrations/evaluate")
async def evaluate_integration(req: IntegrationEvaluateRequest) -> dict[str, Any]:
    try:
        review = await get_app().factory.evaluate(req.candidate)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, f"invalid integration candidate: {exc}")
    return review.model_dump(mode="json")


# --- Live orchestration-agent activity (read-only projection of local transcripts) ---


def _allow_activity_refresh(request: Request) -> bool:
    """Decide whether this GET may trigger the durable transcript sync.

    ``list_activity``/``activity_summary`` only read the projection table.
    Refreshing that table parses local transcripts and writes SQLite, so a
    plain cross-site browser GET must not trigger it. We keep unauthenticated
    non-browser reads usable (the local CLI/dashboard fixtures use them), and
    require one explicit signal for a sync: the API token, an ``X-Leo-Refresh:
    1`` header, or a browser request whose ``Sec-Fetch-Site`` is same-origin
    (or absent) and whose Origin is on the dashboard allowlist. Without that
    signal the route still answers from the last durable projection, with no
    write. This is the documented CSRF guard for sync-triggering reads.
    """
    principal = str(getattr(request.state, "principal", "") or "")
    if principal == "api-token":
        return True
    if request.headers.get("x-leo-refresh") == "1":
        return True
    fetch_site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if fetch_site and fetch_site not in ("none", "same-origin"):
        return False
    origin = (request.headers.get("origin") or "").strip()
    if origin:
        allowed = {
            item.rstrip("/").lower()
            for item in cors_origins(_Settings.load().server.cors_origins)
        }
        if origin.rstrip("/").lower() not in allowed:
            return False
    return True


@api.get("/agent-activity")
async def agent_activity(
    request: Request,
    status: str | None = None,
    role_contains: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    from ..agents import activity

    # The dashboard's default "All statuses" choice is sent as `status=all`;
    # treat it as the absence of a filter instead of rejecting the initial load.
    if status is not None and status.lower() == "all":
        status = None
    if status == "all":
        status = None  # the dashboard's default "no filter" selection
    elif status is not None and status not in activity.STATUSES:
        raise HTTPException(400, f"status must be one of: all, {', '.join(activity.STATUSES)}")
    if limit < 1:
        raise HTTPException(400, "limit must be >= 1")
    if offset < 0 or offset > 100_000:
        raise HTTPException(400, "offset must be between 0 and 100000")
    for name, value in (("role_contains", role_contains), ("q", q)):
        if value is not None and len(value) > 200:
            raise HTTPException(400, f"{name} must be at most 200 characters")
    if _allow_activity_refresh(request):
        await activity.sync_agent_activity()
    return await activity.list_activity(
        status=status,
        role_contains=role_contains,
        q=q,
        limit=min(limit, 200),
        offset=offset,
    )


@api.get("/agent-activity/summary")
async def agent_activity_summary(request: Request) -> dict[str, Any]:
    from ..agents import activity

    if _allow_activity_refresh(request):
        await activity.sync_agent_activity()
    return await activity.activity_summary()

_static = Path(__file__).parent / "static"
if _static.exists():
    api.mount("/assets", StaticFiles(directory=_static), name="assets")

    @api.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(_static / "index.html")

    @api.get("/agent-activity/live")
    async def agent_activity_live() -> FileResponse:
        return FileResponse(_static / "agents-live.html")
