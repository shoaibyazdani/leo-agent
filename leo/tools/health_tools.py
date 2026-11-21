"""Runtime-status tools built from live app, integration and capability state.

Nothing here initializes the app, probes a platform or invents readiness:
``app_health`` mirrors ``LeoApp.health()`` (the payload served at ``GET /health``)
from the injected app or from the same live engines, ``integration_matrix``
classifies real integration records, and ``capability_status`` separates
executable native capabilities from mock-tested and credential-blocked ones.

Handlers read ``ctx['services']``; ``register(..., services=...)`` additionally
binds services at registration time so the tools also work before the executor
gains service injection. No credential values are ever read or emitted.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Mapping

from .. import __version__, db
from ..contracts import RiskLevel, ToolSpec
from .registry import ToolRegistry

TOOL_NAMES = ("app_health", "integration_matrix", "capability_status")

_APP_KEYS = ("app", "application", "leo_app")
_TOOL_KEYS = ("tools", "registry", "tool_registry")
_RECORD_KEYS = ("integrations", "channels", "channel_registry", "channel_adapters")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CONFIGURED = frozenset({"configured", "ready", "available", "enabled", "ok"})
_DEGRADED = frozenset({"degraded", "partial", "partial-configured"})
_BLOCKED = frozenset({"blocked", "not-configured", "unconfigured", "unavailable", "error", "failed"})
_TIERS = ("all", "native", "mock-tested", "credential-blocked")


# --------------------------------------------------------------------------- helpers


async def _await(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


def _services(ctx: dict[str, Any]) -> Mapping[str, Any]:
    services = ctx.get("services") if isinstance(ctx, Mapping) else None
    if not isinstance(services, Mapping):
        raise RuntimeError(
            "health tools require a services mapping in ctx['services'] "
            "(app, tools, skills, agents, gateway/admission)"
        )
    return services


def _find(services: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = services.get(key)
        if value is not None:
            return value
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_dict(value: Any, source: str) -> dict[str, Any] | None:
    """Normalize an IntegrationRecord, mapping or SQLite row; never invent fields."""
    if hasattr(value, "model_dump"):
        data = dict(value.model_dump(mode="json"))
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        try:
            data = dict(value)
        except (TypeError, ValueError):
            return None
    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None
    capabilities = data.get("capabilities")
    if isinstance(capabilities, str):
        try:
            capabilities = json.loads(capabilities)
        except (TypeError, ValueError):
            capabilities = []
    if not isinstance(capabilities, (list, tuple)):
        capabilities = []
    data["capabilities"] = sorted(str(item) for item in capabilities)
    metadata = data.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    data["metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}
    data["_source"] = source
    return data


async def _integration_records(services: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Merge real records from injected services with durable DB rows (service wins)."""
    merged: dict[str, dict[str, Any]] = {}
    for key in ("integrations", "gateway", *_RECORD_KEYS):
        service = services.get(key)
        if service is None or not callable(getattr(service, "integration_records", None)):
            continue
        for candidate in await _await(service.integration_records()):
            record = _as_dict(candidate, key)
            if record is not None:
                merged.setdefault(record["name"], record)
    if db.database_path_configured():
        rows = await db.fetchall(
            "SELECT name,kind,status,config_ref,capabilities,verification FROM integrations ORDER BY name"
        )
        for candidate in rows:
            record = _as_dict(candidate, "database")
            if record is not None:
                merged.setdefault(record["name"], record)
    return [merged[name] for name in sorted(merged)]


def _admission(services: Mapping[str, Any]) -> Any:
    admission = services.get("admission")
    if admission is not None:
        return admission
    gateway = services.get("gateway")
    return _field(gateway, "admission")


async def _policies(services: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    admission = _admission(services)
    if admission is None or not callable(getattr(admission, "snapshot", None)):
        return {}, False
    try:
        snapshot = await _await(admission.snapshot())
    except Exception:  # a broken policy service must never be treated as permissive
        return {}, True
    return (dict(snapshot) if isinstance(snapshot, Mapping) else {}), True


def _mode(policy: Any) -> str | None:
    value = _field(policy, "mode")
    return str(value) if value is not None else None


def _credential_names(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    required = metadata.get("credential_env")
    configured = metadata.get("credentials_present")
    if not isinstance(required, list) or not required:
        config_ref = str(record.get("config_ref") or "")
        required = [part.strip() for part in config_ref.split(",") if _ENV_NAME.match(part.strip())]
    if not isinstance(configured, list):
        configured = []
    return [str(name) for name in required], [str(name) for name in configured]


def _classify(
    record: Mapping[str, Any], policies: Mapping[str, Any], policies_available: bool
) -> tuple[str, str | None, str | None]:
    """Return (effective status, reason, admission mode) from recorded facts only."""
    name = str(record.get("name"))
    kind = str(record.get("kind") or "")
    raw = str(record.get("status") or "unknown").strip().lower()
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    recorded_reason = metadata.get("reason")
    recorded_reason = str(recorded_reason) if recorded_reason else None
    required, present = _credential_names(record)
    mode = _mode(policies.get(name))
    reason: str | None
    if raw in _CONFIGURED:
        if mode == "deny":
            return "degraded", f"channel '{name}' is denied by admission policy", mode
        if policies_available and kind == "messaging" and name not in policies:
            return "degraded", f"no admission policy for channel '{name}' (fail-closed)", mode
        return "configured", None, mode
    if raw in _DEGRADED:
        return "degraded", recorded_reason or "integration reports partial configuration", mode
    if raw in _BLOCKED:
        if present:
            return "degraded", recorded_reason or "only some credentials are configured", mode
        reason = recorded_reason
        if reason is None:
            reason = (
                f"missing credentials: {', '.join(required)}" if required
                else f"integration is {raw}; no reason recorded"
            )
        return "blocked", reason, mode
    return "blocked", f"unknown integration status {raw!r}; not treated as ready", mode


def _capability_tier(record: Mapping[str, Any], effective: str) -> str:
    if effective == "blocked":
        required, _ = _credential_names(record)
        return "credential-blocked" if required else "environment-blocked"
    verification = str(record.get("verification") or "").strip().lower()
    if "mock" in verification:
        return "mock-tested"
    if verification in ("native", "native-handler", "tested"):
        return "native"
    return "unverified"


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "additionalProperties": False}


# --------------------------------------------------------------------------- app_health


async def _app_health(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    services = _services(ctx)
    app = _find(services, _APP_KEYS)
    if app is not None and callable(getattr(app, "health", None)):
        payload = await _await(app.health())
        if not isinstance(payload, Mapping):
            raise RuntimeError("app.health() did not return a health mapping")
        report = dict(payload)
        report["source"] = "app.health()"
        return report

    # No app facade: compose the exact same fields from the live engines.
    settings = _find(services, ("settings",)) or ctx.get("settings")
    if settings is None:
        raise RuntimeError("app_health requires services['app'] with health() or services['settings']")
    models = _field(settings, "models")
    providers = _field(models, "providers") or {}
    report: dict[str, Any] = {
        "status": "ok" if _is_initialized(app, services) else "not-initialized",
        "version": str(_field(app, "version", __version__) or __version__),
        "db": str(ctx.get("db_path") or _field(settings, "db_path")),
        "providers": {
            str(name): str(_field(provider, "model", "")) for name, provider in providers.items()
        },
        "default_provider": str(_field(models, "default", "")),
        "source": "composed",
    }
    if report["status"] == "ok":
        tools = _find(services, _TOOL_KEYS) or _field(app, "registry")
        skills = _find(services, ("skills",)) or _field(app, "skills")
        agents = _find(services, ("agent_registry", "agents")) or _field(app, "agent_registry")
        permissions = _find(services, ("permissions",)) or _field(app, "permissions")
        if not all((tools, skills, agents, permissions)):
            raise RuntimeError(
                "an initialized app_health requires tools, skills, agents and permissions services"
            )
        if not callable(getattr(tools, "list_specs", None)):
            raise RuntimeError("app_health requires a tool registry with list_specs()")
        skill_list = await _await(skills.list())
        agent_list = await _await(agents.list())
        pending = await _await(permissions.pending_approvals())
        report.update(
            {
                "tools": len(tools.list_specs()),
                "skills": len(skill_list),
                "agents": len(agent_list),
                "pending_approvals": len(pending),
            }
        )
    return report


def _is_initialized(app: Any, services: Mapping[str, Any]) -> bool:
    for candidate in (app, services.get("initialized"), _field(app, "state")):
        if candidate is not None:
            value = _field(candidate, "initialized", candidate)
            return value is True
    return False


# --------------------------------------------------------------------------- integration_matrix


async def _matrix(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    services = _services(ctx)
    records = await _integration_records(services)
    if not records:
        raise RuntimeError(
            "integration_matrix requires real integration records via "
            "services['gateway']/['channels'] or the integrations database"
        )
    policies, policies_available = await _policies(services)
    channel = args.get("channel")
    status_filter = args.get("status")
    if channel is not None:
        records = [record for record in records if record["name"] == channel]
    integrations: list[dict[str, Any]] = []
    for record in records:
        effective, reason, mode = _classify(record, policies, policies_available)
        if status_filter is not None and effective != status_filter:
            continue
        metadata = record.get("metadata")
        integrations.append(
            {
                "name": record["name"],
                "kind": str(record.get("kind") or "unknown"),
                "status": effective,
                "record_status": str(record.get("status") or "unknown"),
                "verification": str(record.get("verification") or "not-verified"),
                "live_verified": metadata.get("live_verified") is True,
                "capabilities": record["capabilities"],
                "reason": reason,
                "policy": mode,
                "source": record["_source"],
            }
        )
    counts = Counter(item["status"] for item in integrations)
    return {
        "status": "ok",
        "queried": True,
        "live_verified": False,  # this tool never probes a platform
        "channel": channel,
        "status_filter": status_filter,
        "integrations": integrations,
        "summary": {
            "total": len(integrations),
            "configured": counts["configured"],
            "degraded": counts["degraded"],
            "blocked": counts["blocked"],
        },
    }


# --------------------------------------------------------------------------- capability_status


async def _capabilities(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    services = _services(ctx)
    tool_service = _find(services, _TOOL_KEYS)
    skill_service = services.get("skills")
    records = await _integration_records(services)
    has_tools = tool_service is not None and callable(getattr(tool_service, "list_specs", None))
    has_skills = skill_service is not None and callable(getattr(skill_service, "list", None))
    if not (has_tools or has_skills or records):
        raise RuntimeError(
            "capability_status requires services['tools'], services['skills'] or real integration records"
        )
    policies, policies_available = await _policies(services)
    capabilities: list[dict[str, Any]] = []

    if has_tools:
        for spec in sorted(tool_service.list_specs(), key=lambda item: str(item.name)):
            registration = (
                tool_service.get(spec.name) if callable(getattr(tool_service, "get", None)) else None
            )
            executable = callable(getattr(registration, "handler", None))
            capabilities.append(
                {
                    "name": f"tool:{spec.name}",
                    "tier": "native" if executable else "unverified",
                    "status": "available" if executable else "unavailable",
                    "verification": "native" if executable else "not-verified",
                    "reason": None if executable else "tool has no executable handler",
                    "source": "tool-registry",
                }
            )

    if has_skills:
        for skill in await _await(skill_service.list()):
            name = str(_field(skill, "name", ""))
            source = str(_field(skill, "source", "unknown"))
            handler = bool(_field(skill, "has_handler", False))
            native = source == "native" and handler
            if native:
                reason = None
                status = "available"
                verification = "native-handler"
            else:
                status = "unverified"
                verification = source
                reason = (
                    "no executable handler" if not handler
                    else f"non-native source {source!r} requires independent verification"
                )
            capabilities.append(
                {
                    "name": f"skill:{name}",
                    "tier": "native" if native else "unverified",
                    "status": status,
                    "verification": verification,
                    "reason": reason,
                    "source": "skill-registry",
                }
            )

    for record in records:
        effective, reason, _ = _classify(record, policies, policies_available)
        capabilities.append(
            {
                "name": f"integration:{record['name']}",
                "tier": _capability_tier(record, effective),
                "status": effective,
                "verification": str(record.get("verification") or "not-verified"),
                "reason": reason,
                "source": "integration-record",
            }
        )

    tier = args.get("tier", "all")
    if tier != "all":
        capabilities = [item for item in capabilities if item["tier"] == tier]
    capabilities.sort(key=lambda item: item["name"])
    counts = Counter(item["tier"] for item in capabilities)
    return {
        "status": "ok",
        "queried": True,
        "live_verified": False,
        "tier": tier,
        "capabilities": capabilities,
        "summary": {
            "total": len(capabilities),
            "native": counts["native"],
            "mock_tested": counts["mock-tested"],
            "credential_blocked": counts["credential-blocked"],
            "environment_blocked": counts["environment-blocked"],
            "unverified": counts["unverified"],
        },
    }


# --------------------------------------------------------------------------- registration


_SPECS: tuple[tuple[ToolSpec, Any], ...] = (
    (
        ToolSpec(
            name="app_health",
            description=(
                "Return the same live health payload as GET /health: initialization state, version, "
                "database path, provider names, default provider and (only when initialized) real "
                "tool/skill/agent/approval counts. Never initializes the app or fabricates readiness."
            ),
            category="health",
            risk=RiskLevel.LOW,
            permissions=["health:read"],
            resource_field="",
            input_schema=_schema({}),
        ),
        _app_health,
    ),
    (
        ToolSpec(
            name="integration_matrix",
            description=(
                "Classify real integration records into configured/degraded/blocked with recorded "
                "reasons, credential state and admission policy; supports channel/status filters. "
                "No platform probe or live verification is performed."
            ),
            category="health",
            risk=RiskLevel.LOW,
            permissions=["health:read", "integration:read"],
            resource_field="",
            input_schema=_schema(
                {
                    "channel": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "enum": ["configured", "degraded", "blocked"]},
                }
            ),
        ),
        _matrix,
    ),
    (
        ToolSpec(
            name="capability_status",
            description=(
                "Report executable native tools/skills separately from mock-tested integrations and "
                "credential-blocked or environment-blocked capabilities, with evidence and reasons; "
                "optionally filter by tier."
            ),
            category="health",
            risk=RiskLevel.LOW,
            permissions=["health:read", "capability:read"],
            resource_field="",
            input_schema=_schema(
                {"tier": {"type": "string", "enum": list(_TIERS)}}
            ),
        ),
        _capabilities,
    ),
)


def register(registry: ToolRegistry, services: Any = None) -> list[str]:
    """Register the runtime-status toolset and return its tool names."""
    injected: Mapping[str, Any] = dict(services) if services else {}
    registered: list[str] = []
    for spec, handler in _SPECS:
        async def bound(args: dict[str, Any], ctx: dict[str, Any], _handler=handler) -> Any:
            merged = dict(ctx)
            if not isinstance(merged.get("services"), Mapping):
                merged["services"] = injected
            return await _handler(args, merged)

        try:
            registry.register(spec, bound)
        except ValueError:  # already present - preserve an earlier registration
            continue
        registered.append(spec.name)
    return registered


__all__ = ["TOOL_NAMES", "register"]
