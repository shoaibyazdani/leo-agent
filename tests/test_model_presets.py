"""Provider presets, the validated config writer, and the model_add tool."""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from leo.interfaces import cli as cli_mod
from leo.models.presets import PRESETS, add_provider, build_provider_entry


def test_presets_cover_major_providers() -> None:
    for name in ("anthropic", "openai", "openrouter", "groq", "deepseek",
                 "mistral", "xai", "together", "moonshot", "qwen", "gemini",
                 "ollama", "lmstudio"):
        assert name in PRESETS, name
    assert PRESETS["anthropic"]["kind"] == "anthropic-compatible"
    assert PRESETS["openrouter"]["kind"] == "openai-compatible"
    assert "vision" in PRESETS["openai"]["capabilities"]
    assert PRESETS["ollama"]["api_key_env"] is None


def test_build_entry_requires_model_and_valid_env_name() -> None:
    with pytest.raises(ValueError, match="model id is required"):
        build_provider_entry("openrouter", "  ")
    with pytest.raises(ValueError, match="unknown preset"):
        build_provider_entry("nope", "m")
    with pytest.raises(ValueError, match="environment-variable NAME"):
        build_provider_entry("openrouter", "m", api_key_env="sk-live-secret-value-123456")
    entry = build_provider_entry("groq", "llama-3.3-70b", capabilities=["text", "tools"])
    assert entry["model"] == "llama-3.3-70b"
    assert entry["capabilities"] == ["text", "tools"]
    assert entry["base_url"].startswith("https://")


def test_add_provider_writes_validated_yaml_with_backup(tmp_path: Any) -> None:
    config = tmp_path / "leo.yaml"
    config.write_text("models:\n  default: mock\n  providers:\n    mock:\n      kind: mock\n")
    entry = build_provider_entry("openrouter", "openrouter/auto")
    result = add_provider("or", entry, config_path=config, make_default=True)
    assert result["success"] is True and result["default"] is True
    data = yaml.safe_load(config.read_text())
    assert data["models"]["default"] == "or"
    assert data["models"]["providers"]["or"]["model"] == "openrouter/auto"
    # the pre-existing provider survived the merge
    assert data["models"]["providers"]["mock"]["kind"] == "mock"
    # a second write keeps a timestamped backup of the first
    add_provider("or2", build_provider_entry("groq", "m"), config_path=config)
    backups = list(tmp_path.glob("leo.yaml.bak-*"))
    assert backups, "the previous config must be backed up"


def test_add_provider_never_accepts_secret_values(tmp_path: Any) -> None:
    config = tmp_path / "leo.yaml"
    with pytest.raises(ValueError, match="credential value"):
        add_provider(
            "leaky",
            {"kind": "openai-compatible", "model": "m",
             "base_url": "https://x.example/v1",
             "extra": "sk-proj-abcdefghijklmnopqrstuvwx"},
            config_path=config,
        )
    assert not config.exists()


def test_model_add_cli_and_presets_listing(tmp_path: Any) -> None:
    runner = CliRunner()
    presets = runner.invoke(cli_mod.cli, ["model", "presets"])
    assert presets.exit_code == 0
    assert "openrouter" in presets.stdout
    config = tmp_path / "leo.yaml"
    added = runner.invoke(cli_mod.cli, [
        "model", "add", "or",
        "--preset", "openrouter",
        "--model", "openrouter/auto",
        "--api-key-env", "OPENROUTER_API_KEY",
        "--config", str(config),
    ])
    assert added.exit_code == 0, added.output
    assert json.loads(added.stdout)["success"] is True
    assert yaml.safe_load(config.read_text())["models"]["providers"]["or"]["api_key_env"] == "OPENROUTER_API_KEY"


def test_bare_leo_prints_help_under_a_pipe() -> None:
    """"Bare leo opens the TUI only for a TTY; piped runs get help."""
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


@pytest.mark.asyncio
async def test_model_add_tool_dry_run_previews_without_writing(tmp_path: Any) -> None:
    from leo.tools import model_config_tools

    class Registry:
        def __init__(self) -> None:
            self.specs: list[Any] = []

        def register(self, spec: Any, handler: Any) -> None:
            self.specs.append(spec)

    registry = Registry()
    names = model_config_tools.register(registry)
    assert names == ["model_add"]
    spec = registry.specs[0]
    assert spec.risk.value == "high"
    handler = model_config_tools._model_add
    preview = await handler(
        {"name": "or", "preset": "openrouter", "model": "openrouter/auto",
         "config_path": str(tmp_path / "leo.yaml")},
        {"dry_run": True},
    )
    assert preview["dry_run"] is True and "entry" in preview
    assert not (tmp_path / "leo.yaml").exists()
