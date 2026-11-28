"""Onboarding wizard: validation order, re-runs and the no-keys-in-YAML rule."""

from __future__ import annotations

import json
import stat
from typing import Any

import yaml
from typer.testing import CliRunner

from leo.interfaces import cli as cli_mod
from leo.models.catalog import ProbeResult
from leo.onboarding_wizard import WizardAnswers, load_env_file, run_wizard

FAKE_KEY = "unit-key-123"


def _ok_probe(*args: Any, **kwargs: Any) -> ProbeResult:
    return ProbeResult(True, "key accepted", ("model-a", "model-b"))


def _installer(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True, "detail": "unit service installed"}


def _answers(**overrides: Any) -> WizardAnswers:
    values: dict[str, Any] = {
        "preset": "openrouter",
        "api_key": FAKE_KEY,
        "model": "model-a",
        "install_service": True,
    }
    values.update(overrides)
    return WizardAnswers(**values)


def test_wizard_writes_config_env_and_channels_without_secrets_in_yaml(tmp_path: Any) -> None:
    config = tmp_path / "leo.yaml"
    env_file = tmp_path / "env"
    result = run_wizard(
        _answers(
            channels=["both"],
            channel_owners={"telegram": ["12345"], "whatsapp": ["+15551234567"]},
            telegram_token="telegram-unit-token",
            whatsapp_token="whatsapp-unit-token",
            whatsapp_phone_id="phone-id-123",
        ),
        config_path=config,
        env_file=env_file,
        probe=_ok_probe,
        service_installer=_installer,
    )
    assert result.ok, result.error
    data = yaml.safe_load(config.read_text())
    assert data["models"]["default"] == "openrouter"
    assert data["models"]["fallback"] == "openrouter-fallback-1"
    assert data["models"]["providers"]["openrouter"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert data["models"]["providers"]["openrouter-fallback-1"]["model"] == "model-b"
    assert data["models"]["vision"]["provider"] == "openrouter-vision"
    assert data["channels"]["policies"]["telegram"]["allow"] == ["12345"]
    assert data["channels"]["policies"]["whatsapp"]["allow"] == ["+15551234567"]
    for secret in (FAKE_KEY, "telegram-unit-token", "whatsapp-unit-token"):
        assert secret not in config.read_text()
    env = load_env_file(env_file)
    assert env["OPENROUTER_API_KEY"] == FAKE_KEY
    assert env["TELEGRAM_BOT_TOKEN"] == "telegram-unit-token"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_bad_key_is_rejected_before_any_write(tmp_path: Any) -> None:
    config = tmp_path / "leo.yaml"
    env_file = tmp_path / "env"
    result = run_wizard(
        _answers(api_key="unit-bad-key"),
        config_path=config,
        env_file=env_file,
        probe=lambda *a, **k: ProbeResult(False, "authentication rejected (HTTP 401)"),
        service_installer=_installer,
    )
    assert not result.ok
    assert "authentication rejected" in result.error
    assert not config.exists() and not env_file.exists()


def test_rerun_updates_existing_setup_and_preserves_unrelated_data(tmp_path: Any) -> None:
    config = tmp_path / "leo.yaml"
    config.write_text(yaml.safe_dump({
        "app_name": "Custom LEO",
        "models": {
            "default": "mock",
            "providers": {"mock": {"kind": "mock", "model": "mock-1"}},
        },
    }))
    env_file = tmp_path / "env"
    env_file.write_text("UNRELATED=keep-me\nOPENROUTER_API_KEY=old-value\n")
    first = run_wizard(
        _answers(), config_path=config, env_file=env_file,
        probe=_ok_probe, service_installer=_installer,
    )
    assert first.ok, first.error
    second = run_wizard(
        _answers(model="model-b"),
        config_path=config, env_file=env_file,
        probe=_ok_probe, service_installer=_installer,
    )
    assert second.ok, second.error
    data = yaml.safe_load(config.read_text())
    assert data["app_name"] == "Custom LEO"
    assert data["models"]["providers"]["mock"]["kind"] == "mock"
    assert data["models"]["providers"]["openrouter"]["model"] == "model-b"
    assert len(data["models"]["providers"]) == 4  # mock + primary + fallback + vision
    env = load_env_file(env_file)
    assert env["UNRELATED"] == "keep-me"
    assert env["OPENROUTER_API_KEY"] == FAKE_KEY
    assert config.with_name("leo.yaml.bak-onboard").is_file()


def test_channel_without_token_is_rejected_before_writes(tmp_path: Any) -> None:
    config = tmp_path / "leo.yaml"
    result = run_wizard(
        _answers(channels=["telegram"]),
        config_path=config, env_file=tmp_path / "env",
        probe=_ok_probe, service_installer=_installer,
    )
    assert not result.ok and "bot token" in result.error
    assert not config.exists()


def test_advanced_mode_honours_explicit_vision_and_fallback(tmp_path: Any) -> None:
    config = tmp_path / "leo.yaml"
    result = run_wizard(
        _answers(
            mode="advanced", model="model-a",
            fallback_models=["model-b", "model-a"], vision_model="vision-1",
        ),
        config_path=config, env_file=tmp_path / "env",
        probe=_ok_probe, service_installer=_installer,
    )
    assert result.ok, result.error
    data = yaml.safe_load(config.read_text())
    assert data["models"]["providers"]["openrouter-vision"]["model"] == "vision-1"
    assert "vision" in data["models"]["providers"]["openrouter-vision"]["capabilities"]
    assert data["models"]["fallback"] == "openrouter-fallback-1"


def test_onboard_cli_noninteractive_flow(tmp_path: Any, monkeypatch: Any) -> None:
    config = tmp_path / "leo.yaml"
    env_file = tmp_path / "env"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "cli-unit-token")
    monkeypatch.setattr("leo.onboarding_wizard.probe_provider", _ok_probe)
    cli_mod._state.clear()
    result = CliRunner().invoke(cli_mod.cli, [
        "onboard", "--yes", "--provider", "openrouter", "--model", "model-a",
        "--key", FAKE_KEY, "--config", str(config), "--env-file", str(env_file),
        "--channel", "telegram", "--telegram-owner", "12345",
        "--no-service", "--skip-doctor",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert yaml.safe_load(config.read_text())["channels"]["policies"]["telegram"]["allow"] == ["12345"]
    assert FAKE_KEY not in config.read_text()
    assert FAKE_KEY not in result.stdout
    assert load_env_file(env_file)["OPENROUTER_API_KEY"] == FAKE_KEY


def test_onboard_cli_rejects_bad_key_without_writing(tmp_path: Any, monkeypatch: Any) -> None:
    config = tmp_path / "leo.yaml"
    monkeypatch.setattr(
        "leo.onboarding_wizard.probe_provider",
        lambda *a, **k: ProbeResult(False, "authentication rejected (HTTP 401)"),
    )
    cli_mod._state.clear()
    result = CliRunner().invoke(cli_mod.cli, [
        "onboard", "--yes", "--provider", "openrouter", "--model", "model-a",
        "--key", "bad-unit-key", "--config", str(config),
        "--env-file", str(tmp_path / "env"), "--no-service", "--skip-doctor",
    ])
    assert result.exit_code == 1, result.output
    assert "authentication rejected" in result.stdout
    assert "bad-unit-key" not in result.stdout
    assert not config.exists()
