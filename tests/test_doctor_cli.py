"""Doctor diagnostics: pass/fail rows, exit codes, --fix and --json."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import yaml
from typer.testing import CliRunner

from leo import db
from leo.doctor import run_diagnostics
from leo.interfaces import cli as cli_mod
from leo.models.catalog import ProbeResult

KEY_NAME = "UNIT_DOCTOR_KEY"


def _config(tmp_path: Any, *, data_dir: Any | None = None) -> Any:
    path = tmp_path / "leo.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"data": str(data_dir or (tmp_path / "data"))},
        "models": {
            "default": "or",
            "fallback": "mock",
            "providers": {
                "or": {
                    "kind": "openai-compatible",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": KEY_NAME,
                    "model": "model-a",
                },
                "mock": {"kind": "mock", "model": "mock-1"},
            },
        },
        "channels": {"policies": {"telegram": {"mode": "allowlist", "allow": ["12345"]}}},
    }))
    return path


def _clean_env(monkeypatch: Any) -> None:
    # An operator shell may point config at its own inference gateway; doctor
    # tests must stay deterministic.
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _healthy_environment(tmp_path: Any, monkeypatch: Any) -> tuple[Any, Any]:
    _clean_env(monkeypatch)
    path = _config(tmp_path)
    monkeypatch.setenv(KEY_NAME, "unit-doctor-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "unit-telegram-token")
    env_file = tmp_path / "env"
    env_file.write_text(
        f"{KEY_NAME}=unit-doctor-key\nTELEGRAM_BOT_TOKEN=unit-telegram-token\n",
        encoding="utf-8",
    )
    asyncio.run(db.initialize(tmp_path / "data" / "leo.db"))
    return path, env_file


def test_doctor_reports_one_pass_row_per_check(tmp_path: Any, monkeypatch: Any) -> None:
    config, env_file = _healthy_environment(tmp_path, monkeypatch)
    monkeypatch.setattr("leo.doctor.service_status", lambda: {"ok": True, "detail": "unit service active"})
    monkeypatch.setattr(
        "leo.doctor.probe_provider",
        lambda *a, **k: ProbeResult(True, "key accepted"),
    )
    cli_mod._state.clear()
    result = CliRunner().invoke(cli_mod.cli, ["doctor", "--config", str(config)])
    assert result.exit_code == 0, result.output
    for name in ("config", "keys", "provider", "models", "database", "service", "channels"):
        assert f"PASS {name}:" in result.output
    assert "7/7 checks passed" in result.output
    assert "unit-doctor-key" not in result.output


def test_doctor_names_failures_and_exits_nonzero(tmp_path: Any, monkeypatch: Any) -> None:
    _clean_env(monkeypatch)
    config = _config(tmp_path)
    env_file = tmp_path / "missing-env"
    monkeypatch.delenv(KEY_NAME, raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr("leo.doctor.service_status", lambda: {"ok": False, "detail": "unit service stopped"})
    monkeypatch.setattr(
        "leo.doctor.probe_provider",
        lambda *a, **k: ProbeResult(False, "authentication rejected (HTTP 401)"),
    )
    cli_mod._state.clear()
    result = CliRunner().invoke(cli_mod.cli, ["doctor", "--config", str(config)])
    assert result.exit_code == 1, result.output
    for name in ("keys", "provider", "database", "service", "channels"):
        assert f"FAIL {name}:" in result.output
    assert "authentication rejected" in result.output
    assert re.search(r"\d+/7 checks passed", result.output)


def test_doctor_fix_repairs_directories_and_restarts_service(tmp_path: Any, monkeypatch: Any) -> None:
    config, env_file = _healthy_environment(tmp_path, monkeypatch)
    # Start with no database and a stopped service; both are safe to repair.
    (tmp_path / "data" / "leo.db").unlink()
    calls = {"status": 0, "restart": 0}

    def status() -> dict[str, Any]:
        calls["status"] += 1
        active = calls["status"] > 1
        return {"ok": active, "detail": "unit service active" if active else "unit service stopped"}

    def restart() -> dict[str, Any]:
        calls["restart"] += 1
        return {"ok": True, "detail": "unit service restarted"}

    monkeypatch.setattr("leo.doctor.service_status", status)
    monkeypatch.setattr("leo.doctor.restart_service", restart)
    monkeypatch.setattr(
        "leo.doctor.probe_provider",
        lambda *a, **k: ProbeResult(True, "key accepted"),
    )
    cli_mod._state.clear()
    result = CliRunner().invoke(cli_mod.cli, ["doctor", "--config", str(config), "--fix"])
    assert result.exit_code == 0, result.output
    assert "PASS database:" in result.output and "(fixed)" in result.output
    assert "PASS service:" in result.output
    assert calls["restart"] == 1
    assert (tmp_path / "data" / "leo.db").is_file()


def test_doctor_reports_stale_default_that_runtime_repair_would_hide(
    tmp_path: Any, monkeypatch: Any
) -> None:
    config, env_file = _healthy_environment(tmp_path, monkeypatch)
    data = yaml.safe_load(config.read_text())
    data["models"]["default"] = "ghost-provider"
    config.write_text(yaml.safe_dump(data))
    results = run_diagnostics(
        cli_mod.Settings.load(str(config)),
        config_path=config, env_file=env_file,
        probe=lambda *a, **k: ProbeResult(True, "key accepted"),
        service=lambda: {"ok": True, "detail": "unit service active"},
    )
    models = next(item for item in results if item.name == "models")
    assert not models.ok and "ghost-provider" in models.detail


def test_doctor_json_keeps_the_health_snapshot_shape(tmp_path: Any, monkeypatch: Any) -> None:
    payload = {
        "status": "ok", "version": "0.1.0", "db": str(tmp_path / "db"),
        "providers": {}, "default_provider": "mock", "catalog": [],
    }

    class FakeApp:
        async def health(self) -> dict[str, Any]:
            return payload

    monkeypatch.setattr(cli_mod, "ready", lambda: FakeApp())
    result = CliRunner().invoke(cli_mod.cli, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload
    assert "PASS" not in result.output


def test_doctor_checks_never_report_key_values(tmp_path: Any, monkeypatch: Any) -> None:
    config, _ = _healthy_environment(tmp_path, monkeypatch)
    results = run_diagnostics(
        cli_mod.Settings.load(str(config)),
        config_path=config, env_file=tmp_path / "missing-env",
        probe=lambda *a, **k: ProbeResult(True, "key accepted"),
        service=lambda: {"ok": True, "detail": "unit service active"},
    )
    rendered = json.dumps([item.as_dict() for item in results])
    assert "unit-doctor-key" not in rendered
    assert "unit-telegram-token" not in rendered
