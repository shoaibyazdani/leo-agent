"""Conversation-callable integration/channel tools.

Every handler reads live objects from ``ctx['services']`` (injected by the
executor integration layer): ``gateway`` is the channel router, ``admission``
is the central policy resolver, ``channels``/``channel_registry`` is the
adapter registry and ``enrollment`` exposes the enrollment views. Records come
from those services' own ``integration_records()``/``snapshot()`` methods; no
status is invented here.

``channel_send`` routes exclusively through ``Gateway.send`` and preflights the
real integration record plus the real channel policy. Missing credentials or a
missing/denying policy produce an explicit ``blocked`` result with a reason —
never a canned success. Failures from the gateway itself are reported the same
way, so callers cannot mistake an un-delivered message for a delivered one.

These tools have no path-like arguments; their specs set ``resource_field=''``
so the executor does not treat a target/channel name as a filesystem resource.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import RiskLevel, ToolSpec
from ..integrations.gateway import GatewayError
from .registry import ToolRegistry

_CHANNEL_SERVICE_KEYS = ("channels", "channel_registry", "channel_adapters", "registry")
_ENROLLMENT_SERVICE_KEYS = ("enrollment", "enrollments")


# --------------------------------------------------------------------------- service access


def _services(ctx: dict[str, Any]) -> Mapping[str, Any]:
    services = ctx.get("services") if isinstance(ctx, Mapping) else None
    if not isinstance(services, Mapping):
        raise RuntimeError(
            "integration tools require a services mapping in ctx['services'] "
            "(gateway, admission, channels, enrollment)"
        )
    return services


def _gateway(services: Mapping[str, Any], *, require_send: bool = False) -> Any:
    gateway = services.get("gateway")
    if gateway is None or (require_send and not callable(getattr(gateway, "send", None))):
        raise RuntimeError(
            "integration tools require a configured 'gateway' service in ctx['services']"
        )
    return gateway


def _admission(services: Mapping[str, Any], gateway: Any) -> Any:
    admission = services.get("admission")
    return admission if admission is not None else getattr(gateway, "admission", None)


def _snapshot(admission: Any) -> dict[str, Any]:
    if admission is None:
        return {}
    try:
        snapshot = admission.snapshot() if callable(getattr(admission, "snapshot", None)) else None
    except Exception:  # a broken policy service must not fabricate an "open" policy
        return {}
    return dict(snapshot) if isinstance(snapshot, Mapping) else {}


def _record_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump(mode="json"))
    if isinstance(record, Mapping):
        return dict(record)
    return {}


def _records_from(value: Any) -> list[dict[str, Any]] | None:
    """Read adapter records from any supported registry service shape."""
    if value is None:
        return None
    if callable(getattr(value, "integration_records", None)):
        return [_record_dict(item) for item in value.integration_records()]
    if callable(getattr(value, "adapter_classes", None)):
        classes = value.adapter_classes()
    elif hasattr(value, "ALL_ADAPTERS"):
        classes = value.ALL_ADAPTERS
    elif isinstance(value, Mapping):
        classes = value
    else:
        return None
    if not isinstance(classes, Mapping):
        return None
    result: list[dict[str, Any]] = []
    for candidate in classes.values():
        adapter = candidate() if isinstance(candidate, type) else candidate
        if hasattr(adapter, "record"):
            result.append(_record_dict(adapter.record()))
    return result


def _records(services: Mapping[str, Any], gateway: Any) -> list[dict[str, Any]]:
    """Collect real records from the injected registry and gateway services."""
    merged: dict[str, dict[str, Any]] = {}
    sources: list[list[dict[str, Any]]] = []
    for key in _CHANNEL_SERVICE_KEYS:
        found = _records_from(services.get(key))
        if found is not None:
            sources.append(found)
    found = _records_from(gateway)
    if found is not None:
        sources.append(found)
    if not sources:
        from ..integrations.channels import integration_records

        sources.append([_record_dict(item) for item in integration_records()])
    for batch in sources:
        for record in batch:
            name = record.get("name")
            if isinstance(name, str) and name:
                merged[name] = record
    return [merged[name] for name in sorted(merged)]


def _call_service(service: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    func = getattr(service, method, None)
    if not callable(func):
        raise RuntimeError(
            f"integration enrollment service does not implement {method}()"
        )
    result = func(*args, **kwargs)
    return result


async def _await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _enrollment_service(services: Mapping[str, Any]) -> Any:
    for key in _ENROLLMENT_SERVICE_KEYS:
        service = services.get(key)
        if service is not None:
            return service
    from ..integrations import enrollment

    return enrollment


async def _blocked(ctx: dict[str, Any], channel: str, target: str, reason: str) -> dict[str, Any]:
    payload = {
        "status": "blocked",
        "success": False,
        "delivered": False,
        "channel": channel,
        "target": target,
        "reason": reason,
    }
    events = ctx.get("events") if isinstance(ctx, Mapping) else None
    if events is not None and callable(getattr(events, "emit", None)):
        try:
            await events.emit("integration.blocked", channel=channel, target=target, reason=reason)
        except Exception:  # an audit-event failure must not turn blocked into sent
            pass
    return payload


# --------------------------------------------------------------------------- handlers


async def _integration_status(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    services = _services(ctx)
    gateway = _gateway(services)
    admission = _admission(services, gateway)
    channel = args.get("channel")
    records = _records(services, gateway)
    if channel is not None:
        records = [record for record in records if record.get("name") == channel]
    policies = _snapshot(admission)
    if channel is not None:
        policies = {key: value for key, value in policies.items() if key == channel}
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "status": "ok",
        "queried": True,
        "live_verified": False,
        "channel": channel,
        "integrations": records,
        "policies": policies,
        "summary": {"total": len(records), "by_status": counts},
    }


async def _channel_send(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    services = _services(ctx)
    gateway = _gateway(services, require_send=True)
    channel = args["channel"]
    target = args["target"]
    text = args["text"]

    record = next(
        (item for item in _records(services, gateway) if item.get("name") == channel), None
    )
    if record is None:
        return await _blocked(
            ctx, channel, target,
            f"no integration record for channel '{channel}': credentials not configured",
        )
    status = str(record.get("status") or "unknown")
    if status != "configured":
        metadata = record.get("metadata") or {}
        reason = metadata.get("reason") if isinstance(metadata, Mapping) else None
        return await _blocked(
            ctx, channel, target,
            str(reason or f"channel '{channel}' is {status}: credentials not configured"),
        )

    admission = _admission(services, gateway)
    policy = _snapshot(admission).get(channel)
    if policy is None:
        return await _blocked(
            ctx, channel, target, f"no channel policy configured for '{channel}'"
        )
    mode = policy.get("mode") if isinstance(policy, Mapping) else getattr(policy, "mode", None)
    if str(mode) == "deny":
        return await _blocked(
            ctx, channel, target, f"channel '{channel}' is denied by policy"
        )

    try:
        result = await gateway.send(
            channel, target, text, subject=args.get("subject")
        )
    except GatewayError as exc:
        return await _blocked(ctx, channel, target, f"gateway refused send: {exc}")
    if not isinstance(result, Mapping):
        return await _blocked(
            ctx, channel, target, "gateway returned an invalid send result"
        )
    if result.get("success") is False:
        return await _blocked(
            ctx, channel, target,
            str(result.get("error") or f"gateway could not deliver to {channel}"),
        )
    return {
        "status": "sent",
        "success": True,
        "delivered": True,
        "channel": channel,
        "target": target,
        "conversation_id": result.get("conversation_id"),
        "provider_message_id": result.get("provider_message_id") or result.get("message_id"),
        "subject": result.get("subject"),
    }


async def _enrollment_status(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    services = _services(ctx)
    gateway = _gateway(services)
    admission = _admission(services, gateway)
    enrollment = _enrollment_service(services)
    channel = args.get("channel")
    if channel is not None:
        detail = await _await(
            _call_service(enrollment, "channel_details", channel, gateway=gateway)
        )
        details = [dict(detail)] if isinstance(detail, Mapping) else [detail]
    else:
        details = await _await(_call_service(enrollment, "list_channels", gateway=gateway))
        details = list(details) if isinstance(details, (list, tuple)) else []
    details = [dict(item) if isinstance(item, Mapping) else item for item in details]
    policies = _snapshot(admission)
    for detail in details:
        if not isinstance(detail, dict):
            continue
        name = detail.get("channel")
        if name and not detail.get("policy") and name in policies:
            detail["policy"] = policies[name]
    return {
        "status": "ok",
        "queried": True,
        "channel": channel,
        "channels": details,
        "admission": policies,
        "summary": {
            "total": len(details),
            "configured": sum(
                1 for item in details
                if isinstance(item, Mapping) and item.get("status") == "configured"
            ),
        },
    }


# --------------------------------------------------------------------------- registration


_SPECS: tuple[tuple[ToolSpec, Any], ...] = (
    (
        ToolSpec(
            name="integration_status",
            description=(
                "Read the live integration records and channel admission policies from the "
                "injected gateway/registry services; optionally filter by channel. No platform "
                "probe or live verification is performed."
            ),
            category="integrations",
            risk=RiskLevel.LOW,
            permissions=["integration:read"],
            resource_field="",
            input_schema={
                "type": "object",
                "properties": {"channel": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
        ),
        _integration_status,
    ),
    (
        ToolSpec(
            name="channel_send",
            description=(
                "Send one outbound message through the real channel Gateway. Returns an explicit "
                "blocked result with a reason when credentials or channel policy are absent or "
                "when the gateway refuses delivery; never reports an unsent message as sent."
            ),
            category="integrations",
            risk=RiskLevel.HIGH,
            permissions=["integration:send", "network:write"],
            resource_field="",
            input_schema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "minLength": 1},
                    "target": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "subject": {"type": "string", "minLength": 1},
                },
                "required": ["channel", "target", "text"],
                "additionalProperties": False,
            },
        ),
        _channel_send,
    ),
    (
        ToolSpec(
            name="enrollment_status",
            description=(
                "Describe the real runtime enrollment of every channel (or one channel) from the "
                "enrollment service, gateway records and central admission policy. Field names "
                "and configured flags are returned; secret values never are."
            ),
            category="integrations",
            risk=RiskLevel.LOW,
            permissions=["integration:read"],
            resource_field="",
            input_schema={
                "type": "object",
                "properties": {"channel": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
        ),
        _enrollment_status,
    ),
)


def register(registry: ToolRegistry, services: Any = None) -> list[str]:
    """Register the integration toolset and return its tool names.

    ``services`` is accepted for signatures symmetry with other toolsets; the
    handlers intentionally read the executor-injected per-call ``ctx['services']``
    mapping so no service instance is captured at registration time.
    """
    names: list[str] = []
    for spec, handler in _SPECS:
        registry.register(spec, handler)
        names.append(spec.name)
    return names


__all__ = ["register"]
