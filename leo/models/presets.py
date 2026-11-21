"""Built-in model provider presets and the validated config writer.

``leo model add`` (CLI) and the ``model_add`` tool share this module: a preset
fills the transport details, the caller supplies the model id and the NAME of
the environment variable that holds the key — a secret value is never accepted
or written. The candidate configuration is validated with ``Settings`` before
the file is replaced, and the previous file is kept as a timestamped backup.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from ..config import Settings

_BASE_CAPS = ["text", "streaming", "tools", "structured", "usage"]
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_SECRET_SHAPE_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|sk_(?:live|test|proj)_[A-Za-z0-9]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[A-Z0-9]{12,}"
    r"|AIza[A-Za-z0-9_-]{30,}|eyJ[A-Za-z0-9_-]{8,}\.)"
)

# base_url values are the provider's own endpoint; the api_key_env field is
# the NAME of an environment variable, never a value.
PRESETS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "kind": "anthropic-compatible",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "capabilities": [*_BASE_CAPS, "vision"],
    },
    "openai": {
        "kind": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "capabilities": [*_BASE_CAPS, "vision"],
    },
    "openrouter": {
        "kind": "openai-compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "capabilities": [*_BASE_CAPS, "vision"],
    },
    "groq": {
        "kind": "openai-compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "deepseek": {
        "kind": "openai-compatible",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "mistral": {
        "kind": "openai-compatible",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "xai": {
        "kind": "openai-compatible",
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "capabilities": [*_BASE_CAPS, "vision"],
    },
    "together": {
        "kind": "openai-compatible",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "moonshot": {
        "kind": "openai-compatible",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "qwen": {
        "kind": "openai-compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "capabilities": [*_BASE_CAPS, "vision"],
    },
    "gemini": {
        "kind": "openai-compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "capabilities": [*_BASE_CAPS, "vision"],
    },
    "perplexity": {
        "kind": "openai-compatible",
        "base_url": "https://api.perplexity.ai",
        "api_key_env": "PERPLEXITY_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "fireworks": {
        "kind": "openai-compatible",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "cerebras": {
        "kind": "openai-compatible",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "cohere": {
        "kind": "openai-compatible",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "api_key_env": "COHERE_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "sambanova": {
        "kind": "openai-compatible",
        "base_url": "https://api.sambanova.ai/v1",
        "api_key_env": "SAMBANOVA_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "hyperbolic": {
        "kind": "openai-compatible",
        "base_url": "https://api.hyperbolic.xyz/v1",
        "api_key_env": "HYPERBOLIC_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "novita": {
        "kind": "openai-compatible",
        "base_url": "https://api.novita.ai/v3/openai",
        "api_key_env": "NOVITA_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "nebius": {
        "kind": "openai-compatible",
        "base_url": "https://api.studio.nebius.ai/v1",
        "api_key_env": "NEBIUS_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "lambda": {
        "kind": "openai-compatible",
        "base_url": "https://api.lambdalabs.com/v1",
        "api_key_env": "LAMBDA_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "nvidia": {
        "kind": "openai-compatible",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "zhipu": {
        "kind": "openai-compatible",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "minimax": {
        "kind": "openai-compatible",
        "base_url": "https://api.minimax.chat/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "yi": {
        "kind": "openai-compatible",
        "base_url": "https://api.lingyiwanwu.com/v1",
        "api_key_env": "YI_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "github-models": {
        "kind": "openai-compatible",
        "base_url": "https://models.inference.ai.azure.com",
        "api_key_env": "GITHUB_TOKEN",
        "capabilities": list(_BASE_CAPS),
    },
    "vllm": {
        "kind": "openai-compatible",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": None,
        "capabilities": list(_BASE_CAPS),
    },
    "llamacpp": {
        "kind": "openai-compatible",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key_env": None,
        "capabilities": list(_BASE_CAPS),
    },
    "ollama": {
        "kind": "openai-compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key_env": None,
        "capabilities": list(_BASE_CAPS),
    },
    "lmstudio": {
        "kind": "openai-compatible",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key_env": None,
        "capabilities": list(_BASE_CAPS),
    },
    "openai-compatible": {
        "kind": "openai-compatible",
        "base_url": "",
        "api_key_env": "MODEL_API_KEY",
        "capabilities": list(_BASE_CAPS),
    },
    "anthropic-compatible": {
        "kind": "anthropic-compatible",
        "base_url": "",
        "api_key_env": "MODEL_API_KEY",
        "capabilities": [*_BASE_CAPS, "vision"],
    },
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def build_provider_entry(
    preset: str,
    model: str,
    *,
    api_key_env: str | None = None,
    base_url: str | None = None,
    kind: str | None = None,
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    """Return a validated provider mapping; raises ValueError on bad input."""
    name = str(preset or "").strip().lower()
    if name not in PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; choose one of: {', '.join(preset_names())}"
        )
    chosen_model = str(model or "").strip()
    if not chosen_model:
        raise ValueError("model id is required")
    template = dict(PRESETS[name])
    entry: dict[str, Any] = {
        "kind": kind or template["kind"],
        "model": chosen_model,
        "capabilities": list(capabilities or template["capabilities"]),
    }
    resolved_url = (base_url or template.get("base_url") or "").strip()
    if resolved_url:
        if not re.match(r"^https?://", resolved_url):
            raise ValueError("base_url must start with http:// or https://")
        entry["base_url"] = resolved_url
    env_name = (api_key_env if api_key_env is not None else template.get("api_key_env"))
    if env_name:
        env_name = str(env_name).strip()
        if not _ENV_RE.match(env_name):
            raise ValueError(
                "api_key_env must be an environment-variable NAME like OPENAI_API_KEY"
            )
        entry["api_key_env"] = env_name
    return entry


def _assert_no_secret_values(entry: dict[str, Any], name: str) -> None:
    for key, value in entry.items():
        text = str(value)
        if _SECRET_SHAPE_RE.search(text):
            raise ValueError(
                f"refusing to write a credential value in {name}.{key}; "
                "use api_key_env with the variable NAME instead"
            )


def add_provider(
    name: str,
    entry: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    make_default: bool = False,
) -> dict[str, Any]:
    """Merge one provider into the YAML config after full validation."""
    provider_name = str(name or "").strip()
    if not _NAME_RE.match(provider_name):
        raise ValueError("provider name must be 1-40 chars: letters, digits, . _ -")
    if not isinstance(entry, dict):
        raise ValueError("provider entry must be a mapping")
    _assert_no_secret_values(entry, provider_name)
    path = Path(config_path or os.getenv("LEO_CONFIG") or "leo.yaml")
    data: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = loaded if isinstance(loaded, dict) else {}
    models = data.get("models")
    if not isinstance(models, dict):
        models = {}
        data["models"] = models
    providers = models.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        models["providers"] = providers
    providers[provider_name] = dict(entry)
    if make_default:
        models["default"] = provider_name
    # Full pydantic validation before anything touches the disk.
    Settings.model_validate(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(path, backup)
    rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return {
        "success": True,
        "provider": provider_name,
        "model": entry.get("model"),
        "kind": entry.get("kind"),
        "default": bool(make_default),
        "config_path": str(path),
        "backup": str(backup) if backup else None,
        "note": "restart any running LEO server to load the new provider",
    }
