"""Auto-discovered channel adapters — LEO's full channel matrix.

Hermes and OpenClaw both normalize many messaging platforms behind one gateway.
LEO does the same with three structural differences, both taken from LEO's own
upstream research:

1. Admission is decided centrally (see :mod:`leo.integrations.admission`),
   never per-adapter. Adapters supply verified identity facts only.
2. Every adapter is offline-verifiable: network access only through an
   injectable ``httpx.AsyncClient``, webhook verification fail-closed, and
   credentials referenced by environment-variable *name*. A record reports
   ``configured``/``blocked`` honestly — nothing here claims live delivery.

Any module in this package that defines ``MessagingAdapter`` subclasses with a
concrete ``channel`` is registered automatically (the core telegram/discord/
whatsapp adapters live in :mod:`leo.integrations.messaging` and are merged
in). A module that fails to import surfaces as an ``error`` record instead of
taking the gateway down — loudly, never silently.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from typing import Any

from ...contracts import IntegrationRecord, utcnow
from ...logging import get_logger
from ..messaging import ADAPTERS as _CORE_ADAPTERS
from ..messaging import MessagingAdapter, implements_polling, implements_webhook, verification_health

_LOG = get_logger("integrations.channels")


def _discover() -> tuple[dict[str, type[MessagingAdapter]], dict[str, str]]:
    found: dict[str, type[MessagingAdapter]] = {}
    broken: dict[str, str] = {}
    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        module_name = f"{__name__}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - one broken module must not kill the registry
            broken[module_info.name] = f"{type(exc).__name__}: {exc}"
            continue
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, MessagingAdapter)
                and value.__module__ == module.__name__
                and value is not MessagingAdapter
                and getattr(value, "channel", "unknown") not in ("", "unknown")
            ):
                found[value.channel] = value
    return found, broken


_DISCOVERED, BROKEN_MODULES = _discover()

ALL_ADAPTERS: dict[str, type[MessagingAdapter]] = {**_CORE_ADAPTERS, **_DISCOVERED}


def adapter_classes() -> dict[str, type[MessagingAdapter]]:
    """Every registered adapter class keyed by channel name."""
    return dict(ALL_ADAPTERS)


def default_adapters() -> dict[str, MessagingAdapter]:
    """Instantiate every channel adapter with default (offline) settings."""
    return {name: cls() for name, cls in ALL_ADAPTERS.items()}


def webhook_channels() -> set[str]:
    """Channels whose adapter really implements signed webhook ingress."""
    return {name for name, cls in ALL_ADAPTERS.items() if implements_webhook(cls)}


def polling_channels() -> set[str]:
    """Channels whose adapter really has a pull transport."""
    return {name for name, cls in ALL_ADAPTERS.items() if implements_polling(cls)}


def verification_report() -> list[dict[str, Any]]:
    """Per-channel verification material status, with no secret values.

    ``degraded`` means sending credentials exist but the material required to
    verify signed ingress is missing. Such a channel must not be described as
    a working webhook integration in the dashboard or API.
    """
    return [verification_health(cls()) for _, cls in sorted(ALL_ADAPTERS.items())]


def integration_records() -> list[IntegrationRecord]:
    """Status of every channel without contacting any platform.

    Broken package modules are reported as ``error`` records so an import
    failure is visible in the dashboard instead of hidden.
    """
    records = [cls().record() for _, cls in sorted(ALL_ADAPTERS.items())]
    for module, reason in sorted(BROKEN_MODULES.items()):
        records.append(
            IntegrationRecord(
                name=f"channel-module:{module}",
                kind="messaging",
                status="error",
                capabilities=[],
                verification="import-failed",
                metadata={"reason": reason, "live_verified": False},
            )
        )
    return records


async def sync_integration_records() -> dict[str, Any]:
    """Upsert every channel record into the ``integrations`` table.

    This is what makes channels appear in the dashboard's Connected Surfaces
    view. Statuses stay honest: ``blocked`` until the named environment
    variables exist, ``configured`` afterwards. Live verification is never
    implied. Returns a summary; no secret values are written.
    """
    from ... import db

    synced: list[str] = []
    for record in integration_records():
        now = utcnow()
        await db.execute(
            "INSERT INTO integrations(name,kind,status,config_ref,capabilities,verification,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "kind=excluded.kind, status=excluded.status, config_ref=excluded.config_ref, "
            "capabilities=excluded.capabilities, verification=excluded.verification, updated_at=excluded.updated_at",
            (
                record.name,
                record.kind,
                record.status,
                record.config_ref,
                json.dumps(record.capabilities),
                record.verification,
                now,
                now,
            ),
        )
        synced.append(record.name)
    _LOG.info("channel records synced", extra={"count": len(synced)})
    return {"synced": len(synced), "names": synced}


__all__ = [
    "ALL_ADAPTERS",
    "BROKEN_MODULES",
    "adapter_classes",
    "default_adapters",
    "integration_records",
    "polling_channels",
    "sync_integration_records",
    "verification_report",
    "webhook_channels",
]
