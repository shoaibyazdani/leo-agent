"""Model provider management for the agent (operator-gated).

LEO lets its agent extend the model roster, but a config write is a HIGH-risk
action: it produces an approval gate, accepts only a preset + model id +
environment-variable NAME (never a secret value), validates the whole candidate
configuration with ``Settings`` before writing, and keeps a timestamped backup
of the previous file.
"""

from __future__ import annotations

from typing import Any

from ..contracts import RiskLevel, ToolSpec
from ..models.presets import add_provider, build_provider_entry

_NAME_PROPERTY = {
    "type": "string",
    "minLength": 1,
    "description": "Provider name used in models.providers, e.g. 'openrouter'.",
}
_PATH_PROPERTY = {
    "type": "string",
    "minLength": 1,
    "description": "Config file path; defaults to LEO_CONFIG or ./leo.yaml.",
}


async def _model_add(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    entry = build_provider_entry(
        str(args.get("preset") or "openai-compatible"),
        str(args.get("model") or ""),
        api_key_env=args.get("api_key_env"),
        base_url=args.get("base_url"),
        kind=args.get("kind"),
        capabilities=args.get("capabilities"),
    )
    name = str(args.get("name") or "")
    if ctx.get("dry_run"):
        return {
            "success": True,
            "dry_run": True,
            "provider": name,
            "entry": entry,
            "note": "no file was written (dry run)",
        }
    return add_provider(
        name,
        entry,
        config_path=args.get("config_path") or "leo.yaml",
        make_default=bool(args.get("make_default")),
    )


_SPECS: tuple[tuple[ToolSpec, Any], ...] = (
    (
        ToolSpec(
            name="model_add",
            description=(
                "Add a model provider to leo.yaml. Supply a preset (openrouter, groq, "
                "deepseek, mistral, xai, together, moonshot, qwen, gemini, ollama, "
                "lmstudio, anthropic, openai), the provider's exact model id, and the "
                "NAME of the environment variable holding the key (never the key "
                "itself). The candidate config is validated before writing and the "
                "previous file is backed up. HIGH risk: requires approval."
            ),
            category="config",
            risk=RiskLevel.HIGH,
            permissions=["config:write", "filesystem"],
            resource_field="config_path",
            supports_dry_run=True,
            input_schema={
                "type": "object",
                "properties": {
                    "name": _NAME_PROPERTY,
                    "preset": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Provider preset, e.g. openrouter or ollama.",
                    },
                    "model": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Exact model id the provider serves.",
                    },
                    "api_key_env": {
                        "type": "string",
                        "description": "Environment-variable NAME holding the key.",
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Override the preset endpoint (http/https).",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["anthropic-compatible", "openai-compatible"],
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Override capabilities; include 'vision' for image models.",
                    },
                    "make_default": {
                        "type": "boolean",
                        "description": "Also set models.default to this provider.",
                    },
                    "config_path": _PATH_PROPERTY,
                },
                "required": ["name", "preset", "model"],
                "additionalProperties": False,
            },
        ),
        _model_add,
    ),
)


def register(registry: Any, services: Any = None) -> list[str]:
    names: list[str] = []
    for spec, handler in _SPECS:
        registry.register(spec, handler)
        names.append(spec.name)
    return names
