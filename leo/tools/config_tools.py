"""Read-only configuration inspection tools: config_get, config_sources, config_validate.

The effective settings come from ``ctx['services']['settings']`` when the
integration layer injected them, otherwise from the real ``Settings.load()``
path (including ``LEO_CONFIG`` and all environment overrides). Nothing here
invents configuration values.

Provenance is computed from three real inputs: Pydantic field defaults
(``Settings()``), the parsed YAML config file, and the environment variables
that ``Settings.load()`` actually honours. Precedence is env > file > default,
matching ``Settings.load()``. Values introduced by the loader itself (the
``mock`` provider, repaired ``models.default``/``models.fallback``) are labelled
``default`` and carry ``derived=True`` so they are not mistaken for user input.

Redaction is performed by this module before a result leaves the handler: any
field whose name looks secret-like is redacted wholesale, raw values of
secret-like environment variables are removed from every emitted string, and
common token shapes are masked. The executor applies its own redaction too, but
these tools remain safe when called directly. Environment variable *names* may
be reported (they are identifiers, not credentials); their values never are.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..contracts import RiskLevel, ToolSpec
from .registry import ToolRegistry

REDACTED = "[REDACTED]"

# Bounds for attacker-controlled YAML. The executor's workspace_only switch is
# not a substitute: config tools confine their own paths and cap their own
# input regardless of that setting.
MAX_YAML_BYTES = 256 * 1024
MAX_YAML_KEYS = 2000

_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|credential|authorization|private[_-]?key|session|auth)"
)
_SECRET_ENV_NAME_RE = re.compile(
    r"(?i)(token|key|secret|password|passwd|credential|authorization|session|auth)"
)
_SECRET_VALUE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)gh[pousr]_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
)
_ASSIGNMENT_RE = re.compile(
    r"(?i)(api[_-]?key(?:_env)?|auth[_-]?token|token|secret|password|passwd|credential|authorization)\s*[:=]\s*[^\s,;}\]]+"
)

# Exactly the environment overrides implemented by Settings.load().
_ENV_KEY_MAP: dict[str, str] = {
    "LEO_ENVIRONMENT": "environment",
    "LEO_PATHS_DATA": "paths.data",
    "LEO_PATHS_LOGS": "paths.logs",
    "LEO_PATHS_WORKSPACES": "paths.workspaces",
    "LEO_PATHS_EXPORTS": "paths.exports",
    "LEO_SECURITY_WORKSPACE_ONLY": "security.workspace_only",
    "LEO_SECURITY_ALLOW_MEDIUM_RISK": "security.allow_medium_risk",
    "LEO_SECURITY_HIGH_RISK_REQUIRES_APPROVAL": "security.high_risk_requires_approval",
    "LEO_AUTOMATION_MAX_CONCURRENT_TASKS": "automation.max_concurrent_tasks",
    "LEO_AUTOMATION_DEFAULT_TASK_TIMEOUT_SECONDS": "automation.default_task_timeout_seconds",
}
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "ANTHROPIC_BASE_URL": "base_url",
    "ANTHROPIC_MODEL": "model",
    "ANTHROPIC_AUTH_TOKEN": "api_key_env",
}


# --------------------------------------------------------------------------- redaction


def _is_secret_key(key: Any) -> bool:
    return isinstance(key, str) and bool(_SECRET_KEY_RE.search(key))


def _secret_env_values() -> tuple[str, ...]:
    """Credential-like env VALUES currently present; names stay reportable."""
    values: set[str] = set()
    for name, value in os.environ.items():
        if not value or len(value) < 8 or not _SECRET_ENV_NAME_RE.search(name):
            continue  # short/boolean env values would redact unrelated text
        if value != REDACTED:
            values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_text(text: str, secrets: tuple[str, ...]) -> str:
    result = text
    for value in secrets:
        if value in result:
            result = result.replace(value, REDACTED)
    for pattern in _SECRET_VALUE_RES:
        result = pattern.sub(REDACTED, result)
    result = _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}={REDACTED}", result)
    return result


def _redact_entry(key: str, value: Any, secrets: tuple[str, ...]) -> Any:
    """Redact a value whose field name is itself secret-like (scalar or nested)."""
    return REDACTED if _is_secret_key(key) else _redact(value, secrets)


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_secret_key(key) else _redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_redact(item, secrets) for item in value]
        if isinstance(value, tuple):
            return tuple(items)
        if isinstance(value, (set, frozenset)):
            return set(items)
        return items
    if isinstance(value, BaseModel):
        return _redact(value.model_dump(), secrets)
    if isinstance(value, (str, Path)):
        return _redact_text(str(value), secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value), secrets)


# --------------------------------------------------------------------------- config plumbing


def _config_roots(ctx: Mapping[str, Any]) -> list[Path]:
    """Independently derived roots, whether or not workspace_only is enabled."""
    services = ctx.get("services") if isinstance(ctx, Mapping) else None
    candidate = services.get("settings") if isinstance(services, Mapping) else None
    if isinstance(candidate, Settings):
        settings = candidate
    else:
        try:
            settings = Settings.load()  # honours LEO_PATHS_* and LEO_SECURITY_*
        except (TypeError, ValueError):
            settings = Settings()
    roots = [settings.root.resolve()]
    for path in settings.security.allowed_roots:
        roots.append(settings.resolve(path).resolve())
    for path in (
        settings.paths.workspaces, settings.paths.data,
        settings.paths.logs, settings.paths.exports,
    ):
        roots.append(settings.resolve(path).resolve())
    return roots


def _config_path(args: Mapping[str, Any], ctx: Mapping[str, Any] | None = None) -> Path:
    explicit = args.get("config_path")
    if isinstance(explicit, str) and explicit.strip():
        candidate = Path(explicit).expanduser()
    else:
        candidate = Path(os.getenv("LEO_CONFIG", "leo.yaml")).expanduser()
    resolved = candidate.resolve()
    for root in _config_roots(ctx or {}):
        if resolved == root or root in resolved.parents:
            return candidate  # preserve the caller's display form
    raise PermissionError(f"config path outside allowed roots: {resolved}")


def _key_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value) + sum(_key_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_key_count(item) for item in value)
    return 0


def _check_yaml_size(text: str, path: Any = None) -> None:
    if len(text.encode("utf-8", errors="replace")) > MAX_YAML_BYTES:
        raise ValueError(f"Config exceeds {MAX_YAML_BYTES} bytes: {path or 'data'}")


def _check_key_count(value: Any, path: Any = None) -> None:
    count = _key_count(value)
    if count > MAX_YAML_KEYS:
        raise ValueError(
            f"Config exceeds {MAX_YAML_KEYS} keys ({count}): {path or 'data'}"
        )


def _read_raw(path: Path, *, strict: bool = False) -> dict[str, Any]:
    """Read the YAML mapping used by ``Settings.load`` (empty if absent)."""
    try:
        if path.is_file() and path.stat().st_size > MAX_YAML_BYTES:
            raise ValueError(f"Config exceeds {MAX_YAML_BYTES} bytes: {path}")
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        if strict:
            raise ValueError(f"Cannot read config {path}: {exc}") from exc
        return {}
    _check_yaml_size(text, path)
    try:
        loaded = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        if strict:
            raise ValueError(f"Invalid YAML in config {path}: {exc}") from exc
        return {}
    if not isinstance(loaded, dict):
        if strict:
            raise ValueError(f"Config root must be a mapping: {path}")
        return {}
    _check_key_count(loaded, path)
    return loaded


def _settings(ctx: Mapping[str, Any], path: Path) -> tuple[Settings, str]:
    services = ctx.get("services") if isinstance(ctx, Mapping) else None
    candidate = services.get("settings") if isinstance(services, Mapping) else None
    if candidate is not None:
        if not isinstance(candidate, Settings):
            raise RuntimeError("ctx['services']['settings'] must be a Settings instance")
        return candidate, "services"
    return Settings.load(path), "Settings.load()"


def _flatten(value: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten Settings/raw YAML into dotted leaf keys (containers are recursed)."""
    result: dict[str, Any] = {} if out is None else out
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        if not value and prefix:
            result[prefix] = value
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (BaseModel, Mapping)):
                _flatten(item, child, result)
            else:
                result[child] = item
    else:
        result[prefix] = value
    return result


def _env_source(key: str, raw: Mapping[str, Any]) -> str | None:
    """Name of the environment variable that supplies ``key``, if any."""
    for env_name, mapped_key in _ENV_KEY_MAP.items():
        if mapped_key == key and env_name in os.environ:
            return env_name
    providers = raw.get("models", {})
    providers = providers.get("providers", {}) if isinstance(providers, Mapping) else {}
    if isinstance(providers, Mapping) and "opencode-go" not in providers:
        # Settings.load only creates this provider from env when the file does
        # not declare one and at least one of these variables is present.
        prefix = "models.providers.opencode-go."
        for env_name, field in _PROVIDER_ENV_KEYS.items():
            if key == prefix + field and os.getenv(env_name):
                return env_name
    return None


def _plain(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


def _entries(
    settings: Settings, path: Path, secrets: tuple[str, ...]
) -> list[dict[str, Any]]:
    raw = _read_raw(path)
    file_flat = _flatten(raw)
    default_flat = _flatten(Settings())
    effective = _flatten(settings)
    rows: list[dict[str, Any]] = []
    for key in sorted(effective):
        env_name = _env_source(key, raw)
        if env_name:
            source = "env"
        elif key in file_flat:
            source = "file"
            env_name = None
        else:
            source = "default"
            env_name = None
        derived = bool(
            source == "default"
            and (
                key not in default_flat
                or _plain(effective[key]) != _plain(default_flat[key])
            )
        )
        rows.append(
            {
                "key": key,
                "value": _redact_entry(key, effective[key], secrets),
                "source": source,
                "env_var": env_name,
                "derived": derived,
            }
        )
    return rows


def _lookup(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["key"] == key), None)


def _structured_error(error: Mapping[str, Any], secrets: tuple[str, ...]) -> dict[str, Any]:
    loc = [part for part in error.get("loc", ()) if part != "__root__"]
    # Messages produced by field validators can echo the rejected input; when
    # the field itself is secret-like, disclose neither the message nor input.
    secret_loc = any(_is_secret_key(part) for part in loc)
    context = error.get("ctx") or {}
    return {
        "loc": loc,
        "msg": REDACTED if secret_loc else _redact_text(str(error.get("msg", "")), secrets),
        "type": str(error.get("type", "validation_error")),
        "input": REDACTED if secret_loc else _redact(error.get("input"), secrets),
        "context": REDACTED if secret_loc else _redact(context, secrets),
    }


def _summary(settings: Settings, secrets: tuple[str, ...]) -> dict[str, Any]:
    return {
        "app_name": _redact(settings.app_name, secrets),
        "environment": _redact(settings.environment, secrets),
        "model_default": _redact(settings.models.default, secrets),
        "model_fallback": _redact(settings.models.fallback, secrets),
        "providers": sorted(settings.models.providers),
    }


# --------------------------------------------------------------------------- handlers


async def _config_get(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    path = _config_path(args, ctx)
    settings, origin = _settings(ctx, path)
    secrets = _secret_env_values()
    key = str(args["key"])
    row = _lookup(_entries(settings, path, secrets), key)
    return {
        "status": "ok",
        "key": key,
        "found": row is not None,
        "value": row["value"] if row else None,
        "source": row["source"] if row else None,
        "env_var": row["env_var"] if row else None,
        "derived": bool(row["derived"]) if row else False,
        "config_path": _redact_text(str(path), secrets),
        "settings_from": origin,
    }


async def _config_sources(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    path = _config_path(args, ctx)
    settings, origin = _settings(ctx, path)
    secrets = _secret_env_values()
    rows = _entries(settings, path, secrets)
    prefix = args.get("key_prefix")
    if isinstance(prefix, str) and prefix:
        rows = [row for row in rows if row["key"] == prefix or row["key"].startswith(prefix + ".")]
    counts = {name: sum(row["source"] == name for row in rows) for name in ("default", "file", "env")}
    return {
        "status": "ok",
        "config_path": _redact_text(str(path), secrets),
        "config_exists": path.is_file(),
        "settings_from": origin,
        "key_prefix": prefix,
        "count": len(rows),
        "sources": {row["key"]: row["source"] for row in rows},
        "keys": rows,
        "summary": counts,
    }


async def _config_validate(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    secrets = _secret_env_values()
    path = _config_path(args, ctx)
    display_path = _redact_text(str(path), secrets)
    data = args.get("data")
    errors: list[dict[str, Any]] = []
    validated: Settings | None = None
    input_kind = "data" if isinstance(data, Mapping) else "file"
    limit_error: ValueError | None = None
    if isinstance(data, Mapping):
        try:
            _check_key_count(data, "data")
            _check_yaml_size(json.dumps(dict(data), default=str), "data")
        except ValueError as exc:
            limit_error = exc

    try:
        if limit_error is not None:
            raise limit_error
        if isinstance(data, Mapping):
            # The same Pydantic path Settings.load() uses, applied to a caller
            # supplied candidate mapping (no environment overrides).
            validated = Settings.model_validate(dict(data))
        else:
            raw = _read_raw(path, strict=True)
            validated = Settings.model_validate(raw)
    except ValidationError as exc:
        errors = [_structured_error(error, secrets) for error in exc.errors()]
        validated = None
    except (ValueError, TypeError) as exc:
        errors = [
            {
                "loc": ["config", display_path] if input_kind == "file" else ["data"],
                "msg": _redact_text(str(exc), secrets),
                "type": "config_error",
                "input": REDACTED,
                "context": {},
            }
        ]

    if validated is not None and input_kind == "file":
        # Exercise the whole real load path too: env parsing can fail even
        # though the YAML itself validates against the Pydantic models.
        try:
            validated = Settings.load(path)
        except ValidationError as exc:
            errors = [_structured_error(error, secrets) for error in exc.errors()]
            validated = None
        except (ValueError, TypeError) as exc:
            text = _redact_text(str(exc), secrets)
            loc = ["env", str(exc).split(" ", 1)[0]] if str(exc).startswith("LEO_") else ["env"]
            errors = [
                {"loc": loc, "msg": text, "type": "env_config_error", "input": REDACTED, "context": {}}
            ]
            validated = None

    return {
        "status": "valid" if not errors else "invalid",
        "valid": not errors,
        "errors": errors,
        "error_count": len(errors),
        "config_path": display_path,
        "config_exists": path.is_file(),
        "input": input_kind,
        "validated": _summary(validated, secrets) if validated is not None else None,
    }


# --------------------------------------------------------------------------- registration


_PATH_PROPERTY: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "description": "Config file path; defaults to LEO_CONFIG or ./leo.yaml.",
}

_SPECS: tuple[tuple[ToolSpec, Any], ...] = (
    (
        ToolSpec(
            name="config_get",
            description=(
                "Read one effective configuration value by dotted key from the injected "
                "settings service or the real Settings.load() path. Reports whether the "
                "value came from defaults, the YAML file or an environment override; "
                "secret-like values are redacted."
            ),
            category="config",
            risk=RiskLevel.LOW,
            permissions=["config:read"],
            resource_field="config_path",
            supports_dry_run=True,
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Dotted leaf key, e.g. server.port.",
                    },
                    "config_path": _PATH_PROPERTY,
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        ),
        _config_get,
    ),
    (
        ToolSpec(
            name="config_sources",
            description=(
                "List effective configuration keys with their provenance (default/file/env), "
                "real redacted values and the responsible environment-variable NAME. "
                "Optionally filter by key_prefix. Never emits credential values."
            ),
            category="config",
            risk=RiskLevel.LOW,
            permissions=["config:read"],
            resource_field="config_path",
            supports_dry_run=True,
            input_schema={
                "type": "object",
                "properties": {
                    "key_prefix": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Only report this key and its dotted descendants.",
                    },
                    "config_path": _PATH_PROPERTY,
                },
                "additionalProperties": False,
            },
        ),
        _config_sources,
    ),
    (
        ToolSpec(
            name="config_validate",
            description=(
                "Validate a candidate mapping or the real config file through Pydantic "
                "(Settings.model_validate) and the full Settings.load() path. Returns "
                "structured errors with loc/type/msg/redacted input instead of a bare "
                "boolean; environment parsing failures are reported the same way."
            ),
            category="config",
            risk=RiskLevel.LOW,
            permissions=["config:read", "config:validate"],
            resource_field="config_path",
            supports_dry_run=True,
            input_schema={
                "type": "object",
                "properties": {
                    "config_path": _PATH_PROPERTY,
                    "data": {
                        "type": "object",
                        "description": "Candidate config mapping; validates without env overrides.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        _config_validate,
    ),
)


def register(registry: ToolRegistry, services: Any = None) -> list[str]:
    """Register the configuration toolset and return its tool names.

    ``services`` is accepted for signature symmetry; handlers read the
    executor-injected per-call ``ctx['services']`` so no instance is captured at
    registration time. With no injected settings they call ``Settings.load()``.
    """
    names: list[str] = []
    for spec, handler in _SPECS:
        registry.register(spec, handler)
        names.append(spec.name)
    return names


__all__ = ["register"]
