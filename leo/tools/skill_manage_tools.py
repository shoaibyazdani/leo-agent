"""Agent-authored skill management: create, patch, files and delete.

This is the only write surface for a skill that LEO authored itself. It is
deliberately narrower than a filesystem tool:

* an authored skill is **documentation-only** (``handler=None``); nothing in a
  manifest written here is ever executed,
* a write must be based on a fresh ``skill_read`` of the exact target
  (read-before-write, enforced through :mod:`.skill_read_marks`),
* supporting files are confined to ``references/``, ``templates/`` and
  ``scripts/`` inside the skill's durable, text-only bundle,
* credential-shaped content is refused and invisible/bidi control characters
  are stripped before storage,
* native skills shipped with LEO cannot be deleted or have their handlers,
  permissions or entrypoint changed by this tool.

Every mutation is recorded in ``skill_history`` and published on the event bus;
the executor's audit row records the actor, risk and argument digest.
"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import RiskLevel, SkillManifest, ToolSpec
from ..skills import staging
from ..skills.sanitize import (
    MAX_DOCUMENTATION_CHARS,
    scan_text,
    validate_description,
    validate_relative_path,
)
from .skill_read_marks import was_skill_read

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

_ACTIONS = ("create", "patch", "write_file", "remove_file", "delete")


def _skills(ctx: dict[str, Any]) -> Any:
    services = ctx.get("services")
    if not isinstance(services, dict) or services.get("skills") is None:
        raise RuntimeError(
            "skills service is not available in the tool context; "
            "the SkillRegistry must be injected as ctx['services']['skills']"
        )
    return services["skills"]


def _required_text(args: dict[str, Any], key: str, label: str | None = None) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label or key} must be a non-empty string")
    return value


def _optional_text(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(
            "skill name must be 1-64 lowercase letters, digits, '_' or '-', "
            "and must start with a letter or digit"
        )


def _validate_category(category: str) -> None:
    if not _NAME_RE.match(category):
        raise ValueError(
            "category must be 1-64 lowercase letters, digits, '_' or '-', "
            "and must start with a letter or digit"
        )


def _manifest_from_args(args: dict[str, Any], ctx: dict[str, Any]) -> tuple[SkillManifest, list[str]]:
    name = _required_text(args, "name").strip()
    _validate_name(name)
    description, errors = validate_description(_required_text(args, "description", "description"))
    if errors:
        raise ValueError("; ".join(errors))
    # The description is rendered into the always-loaded system-prompt index,
    # so instruction-shaped text is not merely advisory here: it would ride in
    # every turn before any skill_read. Refuse it at the write boundary.
    _, description_errors, description_warnings = scan_text(
        description, field="description"
    )
    if description_errors or description_warnings:
        raise ValueError("; ".join(description_errors + description_warnings))
    category = (_optional_text(args, "category") or "general").strip()
    _validate_category(category)
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty SKILL.md body")
    clean, scan_errors, warnings = scan_text(content, field="documentation")
    if scan_errors:
        raise ValueError("; ".join(scan_errors))
    if len(clean) > MAX_DOCUMENTATION_CHARS:
        raise ValueError(
            f"documentation is {len(clean):,} characters "
            f"(limit: {MAX_DOCUMENTATION_CHARS:,}); move depth into references/"
        )
    origin: dict[str, Any] = {}
    for key in ("conversation_id", "review_id"):
        value = _optional_text(args, key) or _optional_text(ctx, key)
        if value:
            origin[key] = value[:120]
    manifest = SkillManifest(
        name=name,
        description=description,
        version="0.1.0",
        author="leo-agent",
        category=category,
        permissions=[],
        input_schema={},
        output_schema={},
        source="native",
        documentation=clean,
        tests=[],
        created_by="agent",
        origin=origin,
    )
    return manifest, warnings


def _read_guard(name: str, action: str, file_path: str | None = None) -> None:
    if was_skill_read(name, file_path):
        return
    target = f"skill_read(name={name!r})" if not file_path else (
        f"skill_read(name={name!r}, file_path={file_path!r})"
    )
    raise PermissionError(
        f"read-before-write: call {target} immediately before {action}. "
        "Content quoted earlier in the conversation does not count."
    )


def _write_guard(record: Any, name: str) -> None:
    if str(getattr(record, "created_by", "native")) == "native":
        raise PermissionError(
            "native skills are part of LEO's shipped capability library; the agent "
            "may not change their manifest or supporting files. Create or edit an "
            "authored skill instead."
        )
    extra = getattr(record, "model_extra", None) or {}
    if extra.get("locked"):
        raise PermissionError(
            f"skill '{name}' is locked; the agent may not change it. "
            "An operator must unlock it first."
        )
    if str(extra.get("state") or "active") == "archived":
        raise PermissionError(
            f"skill '{name}' is archived; restore it before editing."
        )


async def _require_skill(skills: Any, name: str) -> Any:
    record = await skills.get(name)
    if record is None:
        raise ValueError(f"unknown skill: {name!r}")
    return record


def _record_summary(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        data = record.model_dump(mode="json")
    else:  # pragma: no cover - defensive
        data = dict(record)
    return {
        key: data.get(key)
        for key in (
            "name", "description", "version", "category", "source", "created_by",
            "status", "enabled", "health",
        )
    }


async def _create(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    skills = _skills(ctx)
    manifest, warnings = _manifest_from_args(args, ctx)
    existing = await skills.get(manifest.name)
    if existing is not None:
        raise ValueError(
            f"a skill named '{manifest.name}' already exists; use action='patch' to "
            "update it instead of creating a duplicate"
        )
    record = await skills.register(manifest)
    return {
        "success": True,
        "action": "create",
        "skill": manifest.name,
        "message": f"Skill '{manifest.name}' created (documentation-only).",
        "record": _record_summary(record),
        "warnings": warnings,
        "hint": (
            "Add depth with action='write_file' and file_path='references/<topic>.md'. "
            "Load it later with skill_read(name, file_path=...)."
        ),
    }


async def _patch(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    skills = _skills(ctx)
    name = _required_text(args, "name").strip()
    _validate_name(name)
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not old:
        raise ValueError("old_string must be a non-empty string")
    if not isinstance(new, str):
        raise ValueError("new_string must be a string (use '' to delete text)")
    record = await _require_skill(skills, name)
    if record.created_by == "native":
        raise PermissionError(
            "native skills are part of LEO's shipped capability library; the agent may "
            "not rewrite their manifest. Create or patch an authored skill instead."
        )
    _write_guard(record, name)
    _read_guard(name, "patch")
    current = str(record.documentation or "")
    count = current.count(old)
    if count == 0:
        raise ValueError(
            "old_string does not occur in the skill; call skill_read(name) and base the "
            "patch on the text it returns"
        )
    if count > 1 and not args.get("replace_all"):
        raise ValueError(
            f"old_string occurs {count} times; include more surrounding text or pass "
            "replace_all=true to replace every occurrence"
        )
    updated = current.replace(old, new) if args.get("replace_all") else current.replace(old, new, 1)
    clean, scan_errors, warnings = scan_text(updated, field="documentation")
    if scan_errors:
        raise ValueError("; ".join(scan_errors))
    if not clean.strip():
        raise ValueError("patch would leave the skill body empty")
    if len(clean) > MAX_DOCUMENTATION_CHARS:
        raise ValueError(
            f"patched documentation would be {len(clean):,} characters "
            f"(limit: {MAX_DOCUMENTATION_CHARS:,}); move depth into references/"
        )
    # Re-register through the real validator. Strip lifecycle fields first:
    # ``SkillRecord`` carries them, but ``register`` owns those columns.
    payload = record.model_dump(include=set(SkillManifest.model_fields))
    payload["documentation"] = clean
    saved = await skills.register(SkillManifest(**payload), replace=True)
    return {
        "success": True,
        "action": "patch",
        "skill": name,
        "message": f"Patched skill '{name}'.",
        "record": _record_summary(saved),
        "warnings": warnings,
        "_change": {"old": old[:200], "new": new[:200]},
    }


async def _write_file(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    skills = _skills(ctx)
    name = _required_text(args, "name").strip()
    _validate_name(name)
    file_path = _required_text(args, "file_path", "file_path")
    normalized, error = validate_relative_path(file_path)
    if error:
        raise ValueError(error)
    content = args.get("file_content")
    if not isinstance(content, str):
        raise ValueError("file_content must be a string")
    record = await _require_skill(skills, name)
    _write_guard(record, name)
    existing = await skills.read_file(name, normalized)
    if existing is not None:
        # Overwriting an existing supporting file needs a fresh read.
        _read_guard(name, "write_file", normalized)
    result = await skills.write_file(name, normalized, content)
    return {
        "success": True,
        "action": "write_file",
        "skill": name,
        "message": f"Wrote {normalized} ({result['bytes']:,} bytes).",
        "file": result,
        "warnings": result.get("warnings") or [],
    }


async def _remove_file(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    skills = _skills(ctx)
    name = _required_text(args, "name").strip()
    _validate_name(name)
    file_path = _required_text(args, "file_path", "file_path")
    normalized, error = validate_relative_path(file_path)
    if error:
        raise ValueError(error)
    record = await _require_skill(skills, name)
    _write_guard(record, name)
    existing = await skills.read_file(name, normalized)
    if existing is None:
        raise ValueError(f"file not found in skill '{name}': {normalized}")
    _read_guard(name, "remove_file", normalized)
    removed = await skills.remove_file(name, normalized)
    return {
        "success": True,
        "action": "remove_file",
        "skill": name,
        "file_path": normalized,
        "removed": removed,
        "message": f"Removed {normalized}.",
    }


async def _delete(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    skills = _skills(ctx)
    name = _required_text(args, "name").strip()
    _validate_name(name)
    record = await skills.get(name)
    if record is None:
        raise ValueError(f"unknown skill: {name!r}")
    if record.created_by == "native":
        raise PermissionError(
            "native skills are part of LEO's shipped capability library and cannot be "
            "deleted by the agent; disable or archive is not exposed for them either"
        )
    extra = record.model_extra or {}
    if extra.get("pinned"):
        raise PermissionError(
            f"skill '{name}' is pinned; unpin it before deleting"
        )
    if extra.get("locked"):
        raise PermissionError(f"skill '{name}' is locked and cannot be changed")
    await skills.unregister(name)
    return {
        "success": True,
        "action": "delete",
        "skill": name,
        "deleted": True,
        "message": f"Skill '{name}' deleted. Its history remains in skill_history.",
    }


_HANDLERS = {
    "create": _create,
    "patch": _patch,
    "write_file": _write_file,
    "remove_file": _remove_file,
    "delete": _delete,
}


def _should_stage(ctx: dict[str, Any], action: str) -> bool:
    """Background reviews stage writes; foreground turns stay interactive."""
    if staging.staging_bypassed():
        return False
    if action not in {"create", "patch", "write_file", "remove_file"}:
        return False
    actor = str(ctx.get("actor") or "")
    if not actor.startswith("review:"):
        return False
    services = ctx.get("services")
    settings = services.get("settings") if isinstance(services, dict) else None
    cfg = getattr(getattr(settings, "learning", None), "review", None)
    return bool(cfg is not None and getattr(cfg, "write_approval", False))


async def skill_manage(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Dispatch one validated skill mutation (or stage a background one)."""
    action = str(args.get("action") or "").strip().lower()
    if action not in _HANDLERS:
        raise ValueError(f"action must be one of: {', '.join(_ACTIONS)}")
    if _should_stage(ctx, action):
        name = str(args.get("name") or "").strip()
        payload = {"action": action, **{key: value for key, value in args.items() if key != "action"}}
        record = await staging.stage(
            action=action,
            skill=name,
            payload=payload,
            actor=str(ctx.get("actor") or ""),
            reason="post-turn review",
        )
        return {
            "success": True,
            "action": action,
            "skill": name,
            "staged": True,
            "proposal_id": record["id"],
            "digest": str(record["arguments_digest"])[:16],
            "message": (
                f"Skill write staged for review as {record['id']}. "
                "An operator can inspect it with /skills pending and apply it "
                "with /skills approve."
            ),
        }
    return await _HANDLERS[action](args, ctx)


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="skill_manage",
            description=(
                "Create, patch, or delete an agent-authored skill and add/remove its "
                "supporting files. Read the target with skill_read first: writes are "
                "refused until the exact content was loaded. Authored skills are "
                "documentation-only and never execute; native skills are read-only here. "
                "When review write-approval is enabled, a background review stages the "
                "write for /skills approve instead of applying it."
            ),
            category="skills",
            risk=RiskLevel.MEDIUM,
            permissions=["skill:write"],
            input_schema=_schema(
                {
                    "action": {"type": "string", "enum": list(_ACTIONS)},
                    "name": {"type": "string", "minLength": 1, "maxLength": 64},
                    "description": {"type": "string", "maxLength": 200},
                    "category": {"type": "string", "maxLength": 64},
                    "content": {
                        "type": "string",
                        "description": "Full SKILL.md body for action='create'.",
                    },
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                    "file_path": {"type": "string"},
                    "file_content": {"type": "string"},
                    "conversation_id": {"type": "string", "maxLength": 120},
                    "review_id": {"type": "string", "maxLength": 120},
                },
                ["action", "name"],
            ),
            resource_field="",  # skills are named records, not filesystem paths
        ),
        skill_manage,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the skill_manage tool and return its name."""
    names: list[str] = []
    for spec, handler in _SPECS:
        registry.register(spec, handler)
        names.append(spec.name)
    return names


__all__ = ["register", "skill_manage"]
