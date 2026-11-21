"""Read-only model, tool and skill registry insight tools.

Every view is synthesized from real LEO state: the injected ``ModelRouter``
(``router.providers`` and ``router.usage_report()``), the injected
``ToolRegistry`` (``list_specs()``) and the injected ``SkillRegistry``
(``list()``). No canned rows, no provider calls and no credentials are exposed.

Token accounting is deliberately explicit: a usage row is reported as
``provider_reported`` only when the recorded metadata says the provider supplied
its counts. Locally derived counts are labeled ``estimated`` (and legacy rows
with no provenance flag are labeled ``unverified``), so an estimate is never
presented as a provider measurement.

Services are injected by the composition root under ``ctx["services"]``; the
mapping accepted by :func:`register` is a fallback for hosts that wire tools
before the executor. A missing service raises a clear error instead of
returning an empty, misleading success.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import RiskLevel, ToolSpec

TOOL_NAMES = (
    "model_providers",
    "model_usage",
    "tool_registry_list",
    "skill_registry_list",
)

_ROUTER_KEYS = ("router", "model_router")
_TOOL_REGISTRY_KEYS = ("tools", "tool_registry", "registry")
_SKILL_REGISTRY_KEYS = ("skills", "skill_registry")

_MAX_PROVIDERS = 50
_MAX_USAGE_ROWS = 100
_MAX_TOOLS = 100
_MAX_SKILLS = 100
_TOKEN_SOURCES = ("estimated", "provider_reported", "unverified")

_USAGE_NOTE = (
    "estimated/unverified token counts are derived locally, not provider measurements"
)
_PROVIDER_NOTE = (
    "provider token counts count as measurements only when the provider reports them; "
    "derived counts are labeled estimates"
)


def _schema(
    properties: dict[str, dict[str, Any]], required: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_LIMIT = {"type": "integer", "minimum": 1, "maximum": _MAX_USAGE_ROWS, "default": 20}
_CATEGORY = {"type": "string", "minLength": 1, "maxLength": 100}
_PROVIDER = {"type": "string", "minLength": 1, "maxLength": 100}


def _service(ctx: dict[str, Any], fallback: Any, keys: tuple[str, ...]) -> Any:
    """Resolve one injected service, preferring the per-call context."""
    per_call = ctx.get("services")
    if per_call is not None and not isinstance(per_call, Mapping):
        raise RuntimeError("ctx['services'] must be a mapping of service names to services")
    for services in (per_call, fallback):
        if not isinstance(services, Mapping):
            continue
        for key in keys:
            value = services.get(key)
            if value is not None:
                return value
    raise RuntimeError(
        f"missing required service {keys[0]!r} in ctx['services'] "
        f"(accepted keys: {', '.join(keys)})"
    )


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump(mode="json"))
    if isinstance(record, Mapping):
        return dict(record)
    return {}


def _risk(value: Any) -> str:
    return str(getattr(value, "value", value) or "low")


def _number(value: Any) -> float:
    try:
        return round(float(value or 0.0), 6)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _input_fields(schema: Any) -> list[str]:
    if not isinstance(schema, Mapping):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return []
    return sorted(str(key) for key in properties)


def _required_fields(schema: Any) -> list[str]:
    if not isinstance(schema, Mapping):
        return []
    required = schema.get("required")
    if not isinstance(required, (list, tuple)):
        return []
    return [str(key) for key in required]


# --------------------------------------------------------------------------- handlers


async def model_providers(
    args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None
) -> Any:
    """Describe the router's real provider set without exposing credentials."""
    router = _service(ctx, fallback, _ROUTER_KEYS)
    settings = getattr(router, "settings", None) or ctx.get("settings")
    model_config = getattr(settings, "models", None)
    configs: Mapping[str, Any] = getattr(model_config, "providers", None) or {}
    active: Mapping[str, Any] = getattr(router, "providers", None) or {}
    name_filter = str(args.get("provider") or "").strip() or None

    rows: list[dict[str, Any]] = []
    for name in sorted(set(map(str, configs)) | set(map(str, active))):
        if name_filter and name != name_filter:
            continue
        config = configs.get(name)
        provider = active.get(name)
        effective = getattr(provider, "config", None) or config
        capabilities = sorted(
            str(capability)
            for capability in (
                getattr(provider, "capabilities", None)
                or getattr(effective, "capabilities", None)
                or []
            )
        )
        rows.append(
            {
                "name": name,
                "model": str(
                    getattr(provider, "model", None)
                    or getattr(effective, "model", None)
                    or ""
                ),
                "kind": str(
                    getattr(effective, "kind", None)
                    or getattr(provider, "style", None)
                    or "unknown"
                ),
                "active": provider is not None,
                "enabled": bool(getattr(config, "enabled", True)),
                "capabilities": capabilities,
                "context_window": _count(getattr(effective, "context_window", 0)),
                "base_url_configured": bool(getattr(config, "base_url", None)),
                "credential_env_configured": bool(getattr(config, "api_key_env", None)),
                "cost_per_million_input": _number(
                    getattr(effective, "cost_per_million_input", 0.0)
                ),
                "cost_per_million_output": _number(
                    getattr(effective, "cost_per_million_output", 0.0)
                ),
                "usage_supported": "usage" in capabilities,
            }
        )

    returned = rows[: int(args.get("limit", _MAX_PROVIDERS))]
    return {
        "source": "router.providers",
        "default": str(getattr(model_config, "default", None) or ""),
        "fallback": str(getattr(model_config, "fallback", None) or ""),
        "count": len(rows),
        "returned": len(returned),
        "providers": returned,
        "token_accounting": _PROVIDER_NOTE,
    }


def _token_source(metadata: Mapping[str, Any]) -> str:
    flag = metadata.get("estimated")
    if flag is True:
        return "estimated"
    if flag is False:
        return "provider_reported"
    return "unverified"


def _bucket() -> dict[str, Any]:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}


async def model_usage(
    args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None
) -> Any:
    """Synthesize recorded model calls, split by token provenance."""
    router = _service(ctx, fallback, _ROUTER_KEYS)
    report = await router.usage_report()
    provider_filter = str(args.get("provider") or "").strip() or None
    task_filter = str(args.get("task_id") or "").strip() or None
    measurement = str(args.get("measurement", "all"))

    rows: list[dict[str, Any]] = []
    totals: dict[str, Any] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                              "cost": 0.0, "by_source": {s: _bucket() for s in _TOKEN_SOURCES}}
    for raw in report:
        item = _as_dict(raw)
        if provider_filter and str(item.get("provider") or "") != provider_filter:
            continue
        if task_filter and str(item.get("task_id") or "") != task_filter:
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        source = _token_source(metadata)
        if measurement != "all" and source != measurement:
            continue

        input_tokens = _count(item.get("input_tokens"))
        output_tokens = _count(item.get("output_tokens"))
        cost = _number(item.get("cost"))
        totals["calls"] += 1
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["cost"] = _number(totals["cost"] + cost)
        bucket = totals["by_source"][source]
        bucket["calls"] += 1
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["cost"] = _number(bucket["cost"] + cost)
        rows.append(
            {
                "id": _count(item.get("id")),
                "timestamp": str(item.get("ts") or ""),
                "provider": str(item.get("provider") or ""),
                "model": str(item.get("model") or ""),
                "task_id": item.get("task_id"),
                "agent_id": item.get("agent_id"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "token_source": source,
                "estimated": source == "estimated",
                "latency_ms": _number(item.get("latency_ms")),
                "cost": cost,
                "success": bool(item.get("success")),
                "task_category": metadata.get("task_category"),
                "attempt": metadata.get("attempt"),
                "stream": metadata.get("stream"),
            }
        )

    totals["cost"] = _number(totals["cost"])
    returned = rows[: int(args.get("limit", 20))]
    return {
        "source": "router.usage_report",
        "measurement_filter": measurement,
        "count": len(rows),
        "returned": len(returned),
        "totals": totals,
        "note": _USAGE_NOTE,
        "usage": returned,
    }


async def tool_registry_list(
    args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None
) -> Any:
    """Describe the live ToolRegistry's typed specs (bounded, sorted view)."""
    tools = _service(ctx, fallback, _TOOL_REGISTRY_KEYS)
    specs = list(tools.list_specs())
    category_filter = str(args.get("category") or "").strip() or None

    rows: list[dict[str, Any]] = []
    for spec in specs:
        data = _as_dict(spec)
        if category_filter and str(data.get("category") or "") != category_filter:
            continue
        schema = data.get("input_schema")
        rows.append(
            {
                "name": str(data.get("name") or ""),
                "description": str(data.get("description") or ""),
                "category": str(data.get("category") or "general"),
                "risk": _risk(data.get("risk")),
                "permissions": [str(p) for p in (data.get("permissions") or [])],
                "resource_field": data.get("resource_field"),
                "input_fields": _input_fields(schema),
                "required_fields": _required_fields(schema),
            }
        )
    rows.sort(key=lambda row: row["name"])

    categories: dict[str, int] = {}
    risks = {"low": 0, "medium": 0, "high": 0}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
        if row["risk"] in risks:
            risks[row["risk"]] += 1

    returned = rows[: int(args.get("limit", _MAX_TOOLS))]
    return {
        "source": "tool_registry",
        "count": len(rows),
        "returned": len(returned),
        "categories": dict(sorted(categories.items())),
        "risks": risks,
        "tools": returned,
    }


async def skill_registry_list(
    args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None
) -> Any:
    """Describe the live SkillRegistry's manifests (bounded, sorted view)."""
    skills = _service(ctx, fallback, _SKILL_REGISTRY_KEYS)
    records = await skills.list(format="dicts")
    category_filter = str(args.get("category") or "").strip() or None
    enabled_only = bool(args.get("enabled_only", False))

    rows: list[dict[str, Any]] = []
    for raw in records:
        data = _as_dict(raw)
        if category_filter and str(data.get("category") or "") != category_filter:
            continue
        if enabled_only and not bool(data.get("enabled", True)):
            continue
        has_handler = bool(data.get("has_handler")) or bool(
            getattr(raw, "has_handler", False)
        )
        rows.append(
            {
                "name": str(data.get("name") or ""),
                "version": str(data.get("version") or ""),
                "category": str(data.get("category") or "general"),
                "description": str(data.get("description") or ""),
                "status": str(data.get("status") or ""),
                "health": str(data.get("health") or ""),
                "enabled": bool(data.get("enabled", True)),
                "has_handler": has_handler,
                "permissions": [str(p) for p in (data.get("permissions") or [])],
                "dependencies": [str(d) for d in (data.get("dependencies") or [])],
                "input_fields": _input_fields(data.get("input_schema")),
            }
        )
    rows.sort(key=lambda row: row["name"])

    categories: dict[str, int] = {}
    executable = 0
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
        executable += int(row["has_handler"])
    returned = rows[: int(args.get("limit", _MAX_SKILLS))]
    return {
        "source": "skill_registry",
        "count": len(rows),
        "returned": len(returned),
        "categories": dict(sorted(categories.items())),
        "executable": executable,
        "skills": returned,
    }


# --------------------------------------------------------------------------- specs


_SPECS: tuple[tuple[ToolSpec, Any], ...] = (
    (
        ToolSpec(
            name="model_providers",
            description=(
                "Describe configured and active model providers from the live router: "
                "models, capabilities, context windows and cost rates, without credentials."
            ),
            category="models",
            risk=RiskLevel.LOW,
            permissions=["model:read"],
            input_schema=_schema(
                {"limit": _LIMIT, "provider": _PROVIDER}
            ),
            resource_field="",  # provider names are not filesystem paths
        ),
        model_providers,
    ),
    (
        ToolSpec(
            name="model_usage",
            description=(
                "Summarize recorded model calls and token/cost totals, labeling derived "
                "token counts as estimates and provider-reported counts as measurements."
            ),
            category="models",
            risk=RiskLevel.LOW,
            permissions=["model:usage"],
            input_schema=_schema(
                {
                    "limit": _LIMIT,
                    "provider": _PROVIDER,
                    "task_id": _PROVIDER,
                    "measurement": {
                        "type": "string",
                        "enum": ["all", "estimated", "provider_reported", "unverified"],
                        "default": "all",
                    },
                }
            ),
            resource_field="",
        ),
        model_usage,
    ),
    (
        ToolSpec(
            name="tool_registry_list",
            description=(
                "Inspect the live tool registry: typed specs, risk, permissions, input "
                "fields and category/risk counts."
            ),
            category="tools",
            risk=RiskLevel.LOW,
            permissions=["tool:read"],
            input_schema=_schema(
                {"limit": _LIMIT, "category": _CATEGORY}
            ),
            resource_field="",
        ),
        tool_registry_list,
    ),
    (
        ToolSpec(
            name="skill_registry_list",
            description=(
                "Inspect live skill manifests: versions, lifecycle status, permissions "
                "and which skills have real local handlers."
            ),
            category="skills",
            risk=RiskLevel.LOW,
            permissions=["skill:read"],
            input_schema=_schema(
                {
                    "limit": _LIMIT,
                    "category": _CATEGORY,
                    "enabled_only": {"type": "boolean", "default": False},
                }
            ),
            resource_field="",
        ),
        skill_registry_list,
    ),
)


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the four insight tools and return their names.

    ``services`` is an optional fallback injection source; the per-call executor
    context ``ctx['services']`` always wins.
    """
    fallback = services if isinstance(services, Mapping) else None
    names: list[str] = []
    for spec, handler in _SPECS:

        async def call(
            args: dict[str, Any], ctx: dict[str, Any], _handler=handler, _fallback=fallback
        ) -> Any:
            return await _handler(args, ctx, _fallback)

        registry.register(spec, call)
        names.append(spec.name)
    return names


__all__ = [
    "register",
    "TOOL_NAMES",
    "model_providers",
    "model_usage",
    "tool_registry_list",
    "skill_registry_list",
]
