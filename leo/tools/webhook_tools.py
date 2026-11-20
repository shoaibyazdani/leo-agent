"""Conversation-callable webhook administration tools.

Routes come from the real channel-adapter registry. Verification uses the native
``WebhookHandler`` v1 scheme: one HMAC-SHA256 over the send timestamp, the
delivery ID and the exact payload bytes. The secret is only read from the
environment *by name* and is never returned in a tool result. There is no
model-callable signing tool: signing with the server secret is restricted to the
webhook sender, so a tool caller cannot mint usable signatures or headers.

Delivery history is a bounded, read-only projection of durable webhook events
written by ``WebhookHandler.handle``; the tools never create, mutate or replay a
delivery. No network calls and no third-party credentials are used here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from .. import db
from ..contracts import RiskLevel, ToolSpec
from ..tasks import webhooks
from ..tasks.webhooks import WebhookError, WebhookHandler

DEFAULT_SECRET_ENV = "LEO_WEBHOOK_SECRET"
MAX_DELIVERIES = 100
DEFAULT_DELIVERIES = 50
_DELIVERY_HEADERS = (
    "x-delivery-id",
    "x-github-delivery",
    "x-webhook-id",
    "x-event-id",
)
_WEBHOOK_TOPICS = ("webhook.accepted", "webhook.rejected")
_HANDLER_KEYS = ("webhooks", "webhook", "webhook_handler")
_ENGINE_KEYS = ("tasks", "engine", "task_engine")
_CHANNEL_KEYS = ("channels", "channel_registry")
# ``webhook_sign`` is intentionally absent: exposing it would turn the server
# HMAC secret into a signing oracle for any model that can call tools.
TOOL_NAMES = (
    "webhook_routes",
    "webhook_verify",
    "webhook_deliveries",
)


def _schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _services(ctx: dict[str, Any], registered: Any) -> Mapping[str, Any]:
    """Read the integration-owned services mapping, then the register-time one."""
    sources: list[Any] = []
    if isinstance(ctx, Mapping):
        sources.append(ctx.get("services"))
    sources.append(registered)
    for source in sources:
        if isinstance(source, Mapping):
            return source
    return {}


def _injected_handler(services: Mapping[str, Any]) -> WebhookHandler | None:
    for key in _HANDLER_KEYS:
        candidate = services.get(key)
        if candidate is not None and callable(getattr(candidate, "_verify_v1", None)):
            return candidate  # type: ignore[return-value]
    return None


def _handler(ctx: dict[str, Any], registered: Any) -> WebhookHandler:
    """Resolve a real WebhookHandler, or fail closed with a precise error."""
    services = _services(ctx, registered)
    injected = _injected_handler(services)
    if injected is not None:
        return injected
    engine = next(
        (services[key] for key in _ENGINE_KEYS if services.get(key) is not None), None
    )
    if engine is None:
        raise RuntimeError(
            "webhook tools require a 'webhooks' handler or a 'tasks' engine "
            "service in ctx['services']"
        )
    events = None
    if isinstance(ctx, Mapping):
        events = ctx.get("events")
    if events is None:
        events = services.get("events")
    return WebhookHandler(engine, events, secret_env=DEFAULT_SECRET_ENV)


def _secret_env(handler: WebhookHandler) -> str:
    name = getattr(handler, "secret_env", None)
    return name if isinstance(name, str) and name else DEFAULT_SECRET_ENV


def _secret(handler: WebhookHandler) -> str:
    """Read the configured secret by environment-variable name; never return it in results."""
    name = _secret_env(handler)
    secret = os.environ.get(name)
    if not secret:
        raise WebhookError(f"webhook secret environment variable {name} is not set")
    return secret


def _bounded(value: Any, default: int, minimum: int, maximum: int) -> int:
    """Defence in depth: handlers stay bounded even when called directly."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(number, maximum))


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _record_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump(mode="json"))
    return dict(record) if isinstance(record, Mapping) else {}


def _channel_records(services: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Project real integration records; env-var *names* only, never values."""
    from ..integrations import channels

    for key in _CHANNEL_KEYS:
        registry = services.get(key)
        func = getattr(registry, "integration_records", None)
        if callable(func):
            records = func()
            break
    else:
        records = channels.integration_records()
    return {
        record["name"]: record
        for record in (_record_dict(item) for item in records)
        if isinstance(record.get("name"), str)
    }


def _webhook_channel_names(services: Mapping[str, Any]) -> set[str]:
    from ..integrations import channels

    for key in _CHANNEL_KEYS:
        registry = services.get(key)
        func = getattr(registry, "webhook_channels", None)
        if callable(func):
            return {str(name) for name in func()}
    return channels.webhook_channels()


async def _routes(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    services = _services(ctx, registered)
    names = _webhook_channel_names(services)
    requested = args.get("channel")
    if requested is not None:
        requested = str(requested)
        names = {name for name in names if name == requested}
    records = _channel_records(services)
    routes: list[dict[str, Any]] = []
    for name in sorted(names):
        record = records.get(name, {})
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        routes.append(
            {
                "path": f"/webhooks/{name}",
                "method": "POST",
                "channel": name,
                "transport": "signed-webhook",
                "status": record.get("status") or "unknown",
                "verification": record.get("verification") or "not-verified",
                "config_ref": record.get("config_ref"),
                "credentials_present": list(metadata.get("credentials_present") or []),
                "live_verified": bool(metadata.get("live_verified", False)),
            }
        )
    handler = _injected_handler(services)
    secret_env = _secret_env(handler) if handler is not None else DEFAULT_SECRET_ENV
    return {
        "status": "ok",
        "routes": routes,
        "count": len(routes),
        "channel": requested,
        "signature": {
            "scheme": "v1",
            "binding": "timestamp,delivery_id,body",
            "secret_env": secret_env,  # name only
            "secret_configured": bool(os.environ.get(secret_env)),
        },
        "task_intake": {
            "available": handler is not None,
            "http_route": None,
            "note": (
                "Signed task intake is served by WebhookHandler; no generic HTTP "
                "route is registered for it."
            ),
        },
    }


async def _verify(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    handler = _handler(ctx, registered)
    _secret(handler)  # fail closed even when the incoming headers lack a v1 MAC
    headers = args["headers"]
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be an object")
    from_header = None
    for name in _DELIVERY_HEADERS:
        from_header = webhooks.header(headers, name)
        if from_header:
            break
    supplied = args.get("delivery_id")
    if supplied is not None and from_header is not None and str(supplied) != from_header:
        raise WebhookError("delivery_id does not match the signed delivery header")
    delivery_id = str(supplied) if supplied is not None else from_header
    if not delivery_id:
        raise WebhookError("webhook is missing a delivery ID")
    body = webhooks.body_bytes(args["body"])
    verified = handler._verify_v1(body, headers, delivery_id)
    if not verified:
        raise WebhookError("webhook is missing a v1 signature header")
    raw_timestamp = webhooks.header(headers, "x-webhook-request-timestamp")
    return {
        "status": "verified",
        "verified": True,
        "delivery_id": delivery_id,
        "timestamp": int(raw_timestamp) if raw_timestamp is not None else None,
        "signature_scheme": "v1",
        "secret_env": _secret_env(handler),  # name only
    }


async def _deliveries(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    if not db.database_path_configured():
        raise RuntimeError(
            "webhook_deliveries requires an initialized database: call db.initialize() first"
        )
    limit = _bounded(args.get("limit", DEFAULT_DELIVERIES), DEFAULT_DELIVERIES, 1, MAX_DELIVERIES)
    offset = _bounded(args.get("offset", 0), 0, 0, 10**9)
    where = "topic IN (?,?)"
    params: list[Any] = list(_WEBHOOK_TOPICS)
    delivery_id = args.get("delivery_id")
    if delivery_id is not None:
        where += " AND json_extract(payload, '$.payload.delivery_id')=?"
        params.append(str(delivery_id))
    clause = " WHERE " + where
    total = int(
        (await db.fetchone("SELECT COUNT(*) AS n FROM events" + clause, params))["n"]
    )
    rows = await db.fetchall(
        "SELECT id,topic,payload,created_at FROM events" + clause
        + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    deliveries: list[dict[str, Any]] = []
    for row in rows:
        event = _json_object(row["payload"])
        inner = _json_object(event.get("payload"))
        task_id = inner.get("task_id")
        task_status = None
        if task_id:
            task = await db.fetchone("SELECT status FROM tasks WHERE id=?", (str(task_id),))
            task_status = task["status"] if task is not None else None
        deliveries.append(
            {
                "id": row["id"],
                "topic": row["topic"],
                "status": "accepted" if row["topic"] == "webhook.accepted" else "rejected",
                "delivery_id": inner.get("delivery_id"),
                "task_id": task_id,
                "task_status": task_status,
                "reason": inner.get("reason"),
                "prefix": inner.get("prefix"),
                "created_at": row["created_at"],
            }
        )
    consumed = offset + len(deliveries)
    return {
        "status": "ok",
        "deliveries": deliveries,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": consumed if consumed < total else None,
    }


_BODY = {"type": ["string", "object"]}
_HEADERS = {"type": "object"}
_DELIVERY_ID = {"type": "string", "minLength": 1, "maxLength": 512}

_TOOL_DEFS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="webhook_routes",
            description=(
                "List the real signed webhook ingress routes derived from the channel "
                "adapter registry, with verification/config names only (no secrets)."
            ),
            category="webhook",
            risk=RiskLevel.LOW,
            permissions=["webhooks:read"],
            input_schema=_schema({"channel": {"type": "string", "minLength": 1}}),
            resource_field="",  # route/channel identifiers are not file paths
        ),
        _routes,
    ),
    (
        ToolSpec(
            name="webhook_verify",
            description=(
                "Verify a v1 webhook signature against the configured secret, checking "
                "the timestamp window and the delivery-ID/body binding; no task is created."
            ),
            category="webhook",
            risk=RiskLevel.MEDIUM,
            permissions=["webhooks:verify"],
            input_schema=_schema(
                {
                    "body": _BODY,
                    "headers": _HEADERS,
                    "delivery_id": _DELIVERY_ID,
                },
                ["body", "headers"],
            ),
            resource_field="",
        ),
        _verify,
    ),
    (
        ToolSpec(
            name="webhook_deliveries",
            description=(
                "Read a bounded, newest-first page of accepted/rejected webhook deliveries "
                "from durable event history; read-only and filterable by delivery ID."
            ),
            category="webhook",
            risk=RiskLevel.LOW,
            permissions=["webhooks:read"],
            input_schema=_schema(
                {
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_DELIVERIES},
                    "offset": {"type": "integer", "minimum": 0},
                    "delivery_id": _DELIVERY_ID,
                }
            ),
            resource_field="",
        ),
        _deliveries,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register all webhook tools and return their names in declaration order."""
    registered: list[str] = []
    for spec, implementation in _TOOL_DEFS:

        async def handler(
            args: dict[str, Any], ctx: dict[str, Any], _impl=implementation
        ) -> Any:
            return await _impl(args, ctx, services)

        registry.register(spec, handler)
        registered.append(spec.name)
    return registered


__all__ = ["register", "TOOL_NAMES", "DEFAULT_SECRET_ENV"]
