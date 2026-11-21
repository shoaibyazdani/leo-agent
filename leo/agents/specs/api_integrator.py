"""API-integrator specialist: report the real local integration posture.

The handler reads the shipped channel adapters' own ``IntegrationRecord``
values and, when a gateway is injected, its durable outbox statistics. It does
not send messages, resolve approvals or touch credentials: channels whose
credentials are absent are reported as ``blocked`` with their declared reason.
Services come only from ``make_handler(services)``; nothing is read from a
service object in the handler context.

This is a bespoke handler (``handler_kind = bespoke:make_handler``). Its two
grants are exact, read-only tool names from the live integration toolset.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

SEED: dict[str, Any] = {
    "id": "api-integrator",
    "name": "API Integrator",
    "purpose": (
        "Inspect the real channel and adapter posture from local records and "
        "durable outbox state, reporting verification status and blockers truthfully "
        "without sending traffic or resolving approvals."
    ),
    "capabilities": [
        "integration_inventory",
        "credential_posture_reporting",
        "blocker_diagnosis",
        "outbox_observation",
    ],
    # Exact live read-only tool names. 'channel_send' is deliberately excluded.
    "tool_permissions": ["integration_status", "enrollment_status"],
    "model_preferences": ["tools", "structured", "text"],
    "context_budget": 20000,
    "task_limit": 2,
    "metadata": {"handler_kind": "bespoke:make_handler"},
}

_CHANNEL_RE = re.compile(r"\bchannel\s*[=:]?\s*([a-z0-9][a-z0-9_-]{0,63})\b", re.IGNORECASE)


def _channel_filter(objective: Any, context: Mapping[str, Any]) -> str | None:
    requested = context.get("channel")
    if isinstance(requested, str) and requested.strip():
        return requested.strip().lower()
    if objective:
        match = _CHANNEL_RE.search(str(objective))
        if match:
            return match.group(1).lower()
    return None


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump(mode="json"))
    return dict(record)


async def _resolved(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


def make_handler(services: Mapping[str, Any] | None = None) -> Callable[..., Any]:
    """Bind the integration-posture handler to injected services.

    ``services['gateway']`` is optional and is read only here, at bind time.
    Absence is reported as ``outbox_stats: None`` with a reason, never as sent.
    """

    services = dict(services or {})
    gateway = services.get("gateway")

    async def handler(objective: Any = None, context: Any = None, **_: Any) -> dict[str, Any]:
        from ...integrations.channels import integration_records

        ctx = dict(context or {})
        records = [_as_dict(record) for record in integration_records()]
        channel = _channel_filter(objective, ctx)
        if channel:
            records = [record for record in records if str(record.get("name", "")).lower() == channel]
            if not records:
                raise ValueError(f"api-integrator: unknown channel {channel!r}")

        posture: list[dict[str, Any]] = []
        for record in records:
            metadata = record.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            posture.append({
                "name": record.get("name"),
                "kind": record.get("kind"),
                "status": str(record.get("status") or "unknown"),
                "verification": str(record.get("verification") or "not-verified"),
                "config_ref": record.get("config_ref"),
                "live_verified": bool(metadata.get("live_verified")),
                "reason": str(metadata.get("reason") or ""),
            })

        outbox: dict[str, Any] | None = None
        outbox_note: str | None = None
        if gateway is not None and callable(getattr(gateway, "outbox_stats", None)):
            result = await _resolved(gateway.outbox_stats())
            if isinstance(result, Mapping):
                outbox = {str(key): value for key, value in result.items()}
            else:
                outbox_note = "gateway.outbox_stats() returned a non-mapping value"
        else:
            outbox_note = "gateway service with outbox_stats() was not injected"

        statuses = dict(sorted(Counter(item["status"] for item in posture).items()))
        verifications = dict(sorted(Counter(item["verification"] for item in posture).items()))
        live_ready = [item["name"] for item in posture if item["live_verified"] and item["status"] == "ready"]
        blocked = [item["name"] for item in posture if item["status"] == "blocked"]

        return {
            "success": True,
            "output": {
                "mode": "local-posture",
                "sent": False,
                "channel_filter": channel,
                "adapter_count": len(posture),
                "status_counts": statuses,
                "verification_counts": verifications,
                "live_ready": live_ready,
                "blocked": blocked,
                "adapters": posture,
                "outbox_stats": outbox,
                "outbox_note": outbox_note,
            },
            "evidence": [
                {
                    "source": "api-integrator",
                    "claim": (
                        f"Read {len(posture)} real adapter records; "
                        f"{len(blocked)} blocked, {len(live_ready)} live-ready"
                    ),
                    "adapter_count": len(posture),
                    "channel_filter": channel,
                },
                {
                    "source": "api-integrator",
                    "claim": f"Outbox stats {'read from gateway' if outbox is not None else 'unavailable: ' + (outbox_note or 'unknown')}",
                    "outbox_available": outbox is not None,
                },
            ],
        }

    return handler
