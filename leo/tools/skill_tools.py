"""Conversation-callable tools backed by the real native ``SkillRegistry``.

Every handler in this module reads the injected ``skills`` service from the
executor-supplied context (``ctx["services"]["skills"]``) and performs a real
registry read or real ``SkillRegistry.run()`` invocation. Nothing here returns
canned data.

Execution is reported honestly: a declarative/imported manifest with no local
handler is explicitly flagged as ``inert_manifest`` (``executed=False``) instead
of being represented as a successful execution. The integration agent injects
the ``services`` mapping into every tool context; until then a missing service
raises a clear error rather than silently succeeding.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ..contracts import RiskLevel, ToolSpec
from .skill_read_marks import mark_skill_read

# Gate inputs must come from the registry's trusted inventories (its
# ``available_tools``/``available_toolsets``/``gate_context`` configuration) and
# the local machine, never from a model-supplied ``context`` object. Otherwise a
# skill's availability gates are just a prompt-level suggestion.
_UNTRUSTED_GATE_KEYS = frozenset({
    "available_bins", "bins", "commands",
    "env", "environment",
    "config", "available_config",
    "prerequisites", "available_prerequisites",
    "platform", "os",
    "tools", "available_tools",
    "toolsets", "available_toolsets",
})


def _trusted_run_context(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    supplied = args.get("context") or {}
    if not isinstance(supplied, Mapping):
        raise ValueError("context must be an object")
    run_context = {
        str(key): value for key, value in supplied.items()
        if str(key).casefold() not in _UNTRUSTED_GATE_KEYS
    }
    # Always override the actor with the executor's authenticated identity.
    run_context["actor"] = str(ctx.get("actor") or "assistant")
    return run_context


def _services(ctx: dict[str, Any]) -> Any:
    """Return the skill registry injected into the tool context.

    Skills are addressed by name, not by filesystem path, so no path-like
    resource is derived for permission decisions (``resource_field=""``).
    """
    services = ctx.get("services")
    if not isinstance(services, dict) or services.get("skills") is None:
        raise RuntimeError(
            "skills service is not available in the tool context; "
            "the SkillRegistry must be injected as ctx['services']['skills']"
        )
    return services["skills"]


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    return dict(record)


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    """Stable, JSON-safe projection of a skill record."""
    return {
        "name": record.get("name"),
        "version": record.get("version"),
        "category": record.get("category"),
        "description": record.get("description"),
        "source": record.get("source"),
        "created_by": record.get("created_by", "native"),
        "status": record.get("status"),
        "state": record.get("state", "active"),
        "health": record.get("health"),
        "enabled": bool(record.get("enabled", True)),
        "pinned": bool(record.get("pinned", False)),
        "locked": bool(record.get("locked", False)),
        "has_handler": bool(record.get("has_handler")),
    }


# --------------------------------------------------------------------------- handlers


async def skill_list(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """List skills from the live registry (real ``SkillRegistry.list``)."""
    skills = _services(ctx)
    detail = args.get("detail", "summary")
    records = await skills.list(format="dicts")
    items = [_as_dict(record) for record in records]
    if detail == "summary":
        items = [_summary(item) for item in items]
    return {
        "detail": detail,
        "count": len(items),
        "skills": items,
    }


async def skill_search(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Full-text search over the live registry's manifests (local, deterministic)."""
    skills = _services(ctx)
    query = args["query"].strip().lower()
    category = str(args.get("category") or "").strip().lower()
    limit = int(args.get("limit", 20))

    records = await skills.list(format="dicts")
    matches: list[dict[str, Any]] = []
    for raw in records:
        record = _as_dict(raw)
        if category and str(record.get("category") or "").lower() != category:
            continue
        name = str(record.get("name") or "").lower()
        cat = str(record.get("category") or "").lower()
        description = str(record.get("description") or "").lower()
        documentation = str(record.get("documentation") or "").lower()
        weights = ((name, 4), (cat, 2), (description, 1), (documentation, 1))
        score = sum(text.count(query) * weight for text, weight in weights)
        if score <= 0:
            continue
        item = _summary(record)
        item["score"] = score
        matches.append(item)

    matches.sort(key=lambda item: (-item["score"], str(item["name"])))
    returned = matches[:limit]
    return {
        "query": query,
        "category": category or None,
        "count": len(matches),
        "returned": len(returned),
        "matches": returned,
    }


async def skill_read(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Read one skill manifest, or one supporting file, through the registry.

    A successful read records a per-context mark. ``skill_manage`` refuses to
    patch or overwrite that exact target until the mark exists, so a write is
    always based on the text this call returned rather than an earlier quote.
    """
    skills = _services(ctx)
    name = args["name"]
    file_path = args.get("file_path")
    record = await skills.get(name)
    if record is None:
        raise ValueError(f"unknown skill: {name!r}")
    data = _as_dict(record)
    has_handler = bool(data.get("has_handler") or getattr(record, "has_handler", False))
    result: dict[str, Any] = {
        "found": True,
        "name": name,
        "has_handler": has_handler,
        "executable": has_handler,
        "skill": data,
    }
    try:
        result["files"] = await skills.files(name)
    except Exception:  # noqa: BLE001 - an older registry has no file bundle
        result["files"] = []
    if file_path is not None:
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        content = await skills.read_file(name, file_path)
        if content is None:
            raise ValueError(f"file not found in skill {name!r}: {file_path}")
        result["file_path"] = file_path
        result["content"] = content
        mark_skill_read(name, file_path)
    else:
        mark_skill_read(name)
    try:
        await skills.bump_usage(name, "views")
    except Exception:  # noqa: BLE001, S110 - telemetry must never fail a read
        pass
    return result


async def skill_invoke(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Execute a skill via ``SkillRegistry.run`` and report the true outcome.

    ``success`` mirrors the skill's own ``SkillRunResult``; ``executed`` is only
    True when a real local handler actually ran. An imported/documentation-only
    manifest is reported as ``inert_manifest`` with an explicit message.
    """
    skills = _services(ctx)
    name = args["name"]
    payload = dict(args.get("arguments") or {})
    run_context = _trusted_run_context(args, ctx)
    timeout = float(args.get("timeout_seconds", 30.0))

    try:
        result = await asyncio.wait_for(skills.run(name, payload, run_context), timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "skill": name,
            "invoked": True,
            "success": False,
            "executed": False,
            "inert_manifest": False,
            "status": "timeout",
            "message": f"Skill {name!r} did not finish within {timeout}s; no execution is claimed.",
            "output": None,
            "error": f"skill run timed out after {timeout}s",
            "decision": "timeout",
            "approval_id": None,
            "validation_errors": [],
            "duration_ms": timeout * 1000.0,
        }

    data = _as_dict(result)
    record = await skills.get(name)
    has_handler = False
    if record is not None:
        has_handler = bool(_as_dict(record).get("has_handler") or getattr(record, "has_handler", False))
    success = bool(data.get("success"))
    executed = bool(success and has_handler)

    if record is None:
        status = "unknown_skill"
        message = f"Unknown skill {name!r}; no handler was executed."
    elif not has_handler:
        status = "inert_manifest"
        message = (
            f"Skill {name!r} is an imported/declarative manifest with no local executable "
            "handler; nothing was executed. " + (data.get("error") or "")
        ).strip()
    elif success:
        status = "executed"
        message = f"Executed skill {name!r} through the SkillRegistry handler."
    else:
        status = "failed"
        message = f"Skill {name!r} handler ran but reported failure: {data.get('error')}"

    return {
        "skill": name,
        "invoked": True,
        "success": success,
        "executed": executed,
        "inert_manifest": bool(record is not None and not has_handler),
        "status": status,
        "message": message,
        "output": data.get("output"),
        "error": data.get("error"),
        "decision": data.get("decision"),
        "approval_id": data.get("approval_id"),
        "validation_errors": data.get("validation_errors") or [],
        "duration_ms": data.get("duration_ms") or 0.0,
    }


# --------------------------------------------------------------------------- specs


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="skill_list",
            description="List all skills registered in LEO's native SkillRegistry (with health and handler status).",
            category="skills",
            risk=RiskLevel.LOW,
            permissions=["skill:read"],
            input_schema=_schema({
                "detail": {"type": "string", "enum": ["summary", "full"], "default": "summary"},
            }),
            resource_field="",  # skill names are not filesystem paths
        ),
        skill_list,
    ),
    (
        ToolSpec(
            name="skill_search",
            description="Search registered skill manifests by name, category, description or documentation.",
            category="skills",
            risk=RiskLevel.LOW,
            permissions=["skill:read"],
            input_schema=_schema({
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "category": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            }, ["query"]),
            resource_field="",
        ),
        skill_search,
    ),
    (
        ToolSpec(
            name="skill_read",
            description=(
                "Read one registered skill's full manifest and file index, or one "
                "supporting file with file_path (e.g. 'references/api.md'). A successful "
                "read authorizes a subsequent patch of that exact target."
            ),
            category="skills",
            risk=RiskLevel.LOW,
            permissions=["skill:read"],
            input_schema=_schema({
                "name": {"type": "string", "minLength": 1},
                "file_path": {
                    "type": "string",
                    "description": "Optional supporting file path under references/, templates/ or scripts/.",
                },
            }, ["name"]),
            resource_field="",
        ),
        skill_read,
    ),
    (
        ToolSpec(
            name="skill_invoke",
            description=(
                "Invoke a skill's real local handler through the SkillRegistry and report truthful "
                "success/failure; declarative manifests with no handler are reported as inert."
            ),
            category="skills",
            risk=RiskLevel.MEDIUM,
            permissions=["skill:run"],
            input_schema=_schema({
                "name": {"type": "string", "minLength": 1},
                "arguments": {"type": "object"},
                "context": {"type": "object"},
                "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 120},
            }, ["name"]),
            resource_field="",
        ),
        skill_invoke,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register all four skill tools and return their names.

    ``services`` is accepted for the common ``register(registry, services)``
    interface; handlers deliberately resolve the SkillRegistry from the
    per-call executor context so they stay instance-independent.
    """
    names: list[str] = []
    for spec, handler in _SPECS:
        registry.register(spec, handler)
        names.append(spec.name)
    return names


__all__ = ["register", "skill_list", "skill_search", "skill_read", "skill_invoke"]
