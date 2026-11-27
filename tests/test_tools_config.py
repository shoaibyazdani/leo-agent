"""Configuration toolset tests executed through the real ToolExecutorImpl.

All settings come from a monkeypatched LEO_CONFIG file plus monkeypatched
environment variables; no network and no real credentials are used. Assertions
check real Pydantic values/provenance, executor audit rows and events, tool
redaction and structured validation failures.
"""

from __future__ import annotations

import json
import re

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import config_tools
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry

NAMES = ["config_get", "config_sources", "config_validate"]
FAKE_TOKEN = "cred-env-value-9f8e7d6c5b4a3210"
# The env shim only adopts LEO's inference gateway (not the IDE endpoint).
BASE_URL_VALUE = "https://opencode.ai/inference/go/anthropic"
MODEL_VALUE = "env-model-1"
REDACTED = "[REDACTED]"

CONFIG_YAML = """\
app_name: LEO-Configured
server:
  host: 127.0.0.1
  port: 9911
security:
  workspace_only: true
  max_tool_calls_per_task: 55
automation:
  max_concurrent_tasks: 3
models:
  default: opencode-go
  fallback: mock
  providers:
    opencode-go:
      kind: anthropic-compatible
      base_url: https://file-config.invalid/v1
      api_key_env: ANTHROPIC_AUTH_TOKEN
      model: file-model
"""

INVALID_YAML = """\
server:
  port: not-a-number
security:
  max_tool_calls_per_task: many
models:
  temperature: hot
"""

SECRETISH_KEY = re.compile(r"(?i)(api[_-]?key|token|secret|password|credential|session|auth)")


class ServicesExecutor(ToolExecutorImpl):
    """Real executor; only the service-injection boundary is overridden."""

    def __init__(self, *args: object, services: dict | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.services = services or {}

    def _context(self, call, task_key, dry_run):
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.services
        return ctx


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "leo.yaml"
    config.write_text(CONFIG_YAML, encoding="utf-8")
    for name in (
        "LEO_PATHS_DATA", "LEO_PATHS_LOGS", "LEO_PATHS_WORKSPACES", "LEO_PATHS_EXPORTS",
        "LEO_SECURITY_ALLOW_MEDIUM_RISK", "LEO_SECURITY_HIGH_RISK_REQUIRES_APPROVAL",
        "LEO_AUTOMATION_DEFAULT_TASK_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LEO_CONFIG", str(config))
    monkeypatch.setenv("LEO_ENVIRONMENT", "staging")
    monkeypatch.setenv("LEO_SECURITY_WORKSPACE_ONLY", "false")
    monkeypatch.setenv("LEO_AUTOMATION_MAX_CONCURRENT_TASKS", "7")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", BASE_URL_VALUE)
    monkeypatch.setenv("ANTHROPIC_MODEL", MODEL_VALUE)

    settings = Settings.load(config)
    executor_settings = Settings()
    executor_settings.security.high_risk_requires_approval = False
    events = EventBus()
    permissions = PermissionManagerImpl(executor_settings, events)
    registry = ToolRegistry()
    assert config_tools.register(registry) == NAMES
    executor = ServicesExecutor(
        registry, permissions, events, executor_settings,
        services={"settings": settings}, timeout_seconds=5.0,
    )
    return tmp_path, config, settings, registry, executor, events


async def run(env, name: str, arguments: dict | None = None, **kwargs):
    _, _, _, _, executor, _ = env
    return await executor.execute(ToolCall(name=name, arguments=arguments or {}, **kwargs))


async def tool_rows(name: str) -> list[dict]:
    return [dict(row) for row in await db.fetchall("SELECT * FROM tool_calls WHERE tool=?", (name,))]


# --------------------------------------------------------------------------- registration


def test_registration_and_typed_specs():
    registry = ToolRegistry()
    assert config_tools.register(registry, services=None) == NAMES
    specs = {spec.name: spec for spec in registry.list_specs()}
    assert set(specs) == set(NAMES)
    for spec in specs.values():
        assert spec.category == "config"
        assert spec.risk is RiskLevel.LOW
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == "config_path"
        assert spec.description and "TODO" not in spec.description
    assert specs["config_get"].permissions == ["config:read"]
    assert specs["config_sources"].permissions == ["config:read"]
    assert set(specs["config_validate"].permissions) == {"config:read", "config:validate"}
    assert registry.validate("config_get", {"key": "server.port"}) == []
    assert registry.validate("config_get", {})
    assert registry.validate("config_get", {"key": "server.port", "extra": 1})
    assert registry.validate("config_sources", {"key_prefix": "server"}) == []
    assert registry.validate("config_validate", {"data": {"app_name": "X"}}) == []
    assert registry.validate("config_validate", {"config_path": 42})


# --------------------------------------------------------------------------- config_get


async def test_config_get_reads_real_effective_values_and_sources(env):
    seen = []
    env[5].subscribe("tool.called", seen.append)
    cases = {
        "server.port": (9911, "file"),
        "app_name": ("LEO-Configured", "file"),
        "security.allow_medium_risk": (True, "default"),
        "environment": ("staging", "env"),
        "security.workspace_only": (False, "env"),
        "security.max_tool_calls_per_task": (55, "file"),
        "automation.max_concurrent_tasks": (7, "env"),
    }
    for key, (expected, source) in cases.items():
        result = await run(env, "config_get", {"key": key})
        assert result.success is True, (key, result.error)
        assert result.output["found"] is True
        assert result.output["value"] == expected, key
        assert result.output["source"] == source, key
        assert result.output["settings_from"] == "services"
        assert result.output["config_path"] == str(env[1])

    env_result = await run(env, "config_get", {"key": "environment"})
    assert env_result.output["env_var"] == "LEO_ENVIRONMENT"

    missing = await run(env, "config_get", {"key": "no.such.key"})
    assert missing.success is True, missing.error
    assert missing.output["found"] is False
    assert missing.output["value"] is None and missing.output["source"] is None

    explicit = await run(env, "config_get", {"key": "server.port", "config_path": str(env[1])})
    assert explicit.success is True, explicit.error
    assert explicit.output["value"] == 9911

    assert [event.payload["tool"] for event in seen] == ["config_get"] * 10
    rows = await tool_rows("config_get")
    assert len(rows) == 10
    assert all(row["decision"] == "allowed" and row["success"] == 1 for row in rows)


async def test_config_get_falls_back_to_settings_load_without_service(env):
    env[4].services = {}
    result = await run(env, "config_get", {"key": "server.port"})
    assert result.success is True, result.error
    assert result.output["value"] == 9911
    assert result.output["source"] == "file"
    assert result.output["settings_from"] == "Settings.load()"


# --------------------------------------------------------------------------- config_sources


async def test_config_sources_reports_real_provenance_and_redaction(env):
    result = await run(env, "config_sources")
    assert result.success is True, result.error
    out = result.output
    assert out["status"] == "ok"
    assert out["config_exists"] is True
    assert out["settings_from"] == "services"
    assert out["sources"]["server.port"] == "file"
    assert out["sources"]["app_name"] == "file"
    assert out["sources"]["security.allow_medium_risk"] == "default"
    assert out["sources"]["environment"] == "env"
    assert out["sources"]["security.workspace_only"] == "env"
    assert out["sources"]["security.max_tool_calls_per_task"] == "file"
    assert set(out["sources"].values()) <= {"default", "file", "env"}
    assert out["count"] == len(out["keys"]) == sum(out["summary"].values())
    rows = {row["key"]: row for row in out["keys"]}
    assert rows["models.providers.opencode-go.api_key_env"]["value"] == REDACTED
    assert rows["environment"]["env_var"] == "LEO_ENVIRONMENT"
    for key, row in rows.items():
        if SECRETISH_KEY.search(key):
            assert row["value"] == REDACTED, key
    assert FAKE_TOKEN not in json.dumps(out)
    # The file declares this provider, so its base_url wins over ANTHROPIC_BASE_URL.
    assert rows["models.providers.opencode-go.base_url"]["value"] == "https://file-config.invalid/v1"
    assert rows["models.providers.opencode-go.base_url"]["source"] == "file"

    filtered = await run(env, "config_sources", {"key_prefix": "server"})
    assert filtered.success is True, filtered.error
    assert set(filtered.output["sources"]) == {
        "server.host", "server.port", "server.cors_origins", "server.concurrency",
    }
    assert filtered.output["count"] == 4
    assert (await tool_rows("config_sources"))[0]["decision"] == "allowed"


async def test_provider_env_overrides_are_env_sourced_and_redacted(env, monkeypatch):
    minimal = env[0] / "minimal.yaml"
    minimal.write_text("server:\n  port: 1234\n", encoding="utf-8")
    monkeypatch.setenv("LEO_CONFIG", str(minimal))
    env[4].services = {"settings": Settings.load(minimal)}

    result = await run(env, "config_sources")
    assert result.success is True, result.error
    out = result.output
    sources = out["sources"]
    prefix = "models.providers.opencode-go."
    assert sources[prefix + "base_url"] == "env"
    assert sources[prefix + "model"] == "env"
    assert sources[prefix + "api_key_env"] == "env"
    assert sources["server.port"] == "file"
    rows = {row["key"]: row for row in out["keys"]}
    assert rows[prefix + "base_url"]["value"] == BASE_URL_VALUE
    assert rows[prefix + "base_url"]["env_var"] == "ANTHROPIC_BASE_URL"
    assert rows[prefix + "model"]["value"] == MODEL_VALUE
    assert rows[prefix + "api_key_env"]["value"] == REDACTED
    assert rows[prefix + "api_key_env"]["env_var"] == "ANTHROPIC_AUTH_TOKEN"
    assert FAKE_TOKEN not in json.dumps(out)


async def test_env_credential_contents_never_appear_in_output(env, monkeypatch):
    leaky = env[0] / "leaky.yaml"
    leaky.write_text(f"app_name: 'prefix {FAKE_TOKEN} suffix'\nserver:\n  port: 4321\n", encoding="utf-8")
    monkeypatch.setenv("LEO_CONFIG", str(leaky))
    settings = Settings.load(leaky)
    assert FAKE_TOKEN in settings.app_name  # the real effective value contains it
    env[4].services = {"settings": settings}

    result = await run(env, "config_get", {"key": "app_name"})
    assert result.success is True, result.error
    assert result.output["value"] == f"prefix {REDACTED} suffix"
    assert FAKE_TOKEN not in json.dumps(result.output)
    assert FAKE_TOKEN not in json.dumps([dict(row) for row in await tool_rows("config_get")])
    all_keys = await run(env, "config_sources")
    assert all_keys.success is True, all_keys.error
    assert FAKE_TOKEN not in json.dumps(all_keys.output)


# --------------------------------------------------------------------------- config_validate


async def test_config_validate_valid_file_uses_real_settings(env):
    result = await run(env, "config_validate")
    assert result.success is True, result.error
    out = result.output
    assert out["status"] == "valid" and out["valid"] is True
    assert out["errors"] == [] and out["error_count"] == 0
    assert out["input"] == "file"
    assert out["config_exists"] is True
    assert out["validated"]["environment"] == "staging"
    assert out["validated"]["model_default"] == "opencode-go"
    assert "opencode-go" in out["validated"]["providers"]
    assert FAKE_TOKEN not in json.dumps(out)
    assert (await tool_rows("config_validate"))[0]["decision"] == "allowed"


async def test_config_validate_reports_structured_pydantic_errors(env):
    invalid = env[0] / "invalid.yaml"
    invalid.write_text(INVALID_YAML, encoding="utf-8")
    result = await run(env, "config_validate", {"config_path": str(invalid)})
    assert result.success is True, result.error
    out = result.output
    assert out["status"] == "invalid" and out["valid"] is False
    assert out["error_count"] == len(out["errors"]) >= 3
    assert out["validated"] is None
    by_loc = {tuple(error["loc"]): error for error in out["errors"]}
    for loc in (("server", "port"), ("security", "max_tool_calls_per_task"), ("models", "temperature")):
        assert loc in by_loc, loc
        assert set(("loc", "msg", "type", "input", "context")) <= set(by_loc[loc])
    assert by_loc[("server", "port")]["type"] == "int_parsing"
    assert "valid integer" in by_loc[("server", "port")]["msg"]
    assert by_loc[("models", "temperature")]["type"] == "float_parsing"
    assert (await tool_rows("config_validate"))[0]["success"] == 1


async def test_config_validate_data_path_and_secret_error_redaction(env):
    candidate = {
        "models": {
            "temperature": "hot",
            "providers": {"x": {"api_key_env": "NOT AN ENV NAME"}},
        }
    }
    result = await run(env, "config_validate", {"data": candidate})
    assert result.success is True, result.error
    out = result.output
    assert out["input"] == "data"
    assert out["valid"] is False
    by_loc = {tuple(error["loc"]): error for error in out["errors"]}
    assert by_loc[("models", "temperature")]["type"] == "float_parsing"
    secret_error = by_loc[("models", "providers", "x", "api_key_env")]
    assert secret_error["msg"] == REDACTED
    assert secret_error["input"] == REDACTED
    assert "NOT AN ENV NAME" not in json.dumps(out)

    valid = await run(env, "config_validate", {"data": {"app_name": "Candidate", "server": {"port": 9000}}})
    assert valid.success is True, valid.error
    assert valid.output["valid"] is True
    assert valid.output["validated"]["app_name"] == "Candidate"
    assert valid.output["validated"]["environment"] == "local"  # data path ignores env


async def test_config_validate_reports_invalid_env_override(env, monkeypatch):
    monkeypatch.setenv("LEO_SECURITY_WORKSPACE_ONLY", "perhaps")
    result = await run(env, "config_validate")
    assert result.success is True, result.error
    out = result.output
    assert out["status"] == "invalid" and out["valid"] is False
    assert out["error_count"] == 1
    assert out["errors"][0]["loc"] == ["env", "LEO_SECURITY_WORKSPACE_ONLY"]
    assert out["errors"][0]["type"] == "env_config_error"
    assert "LEO_SECURITY_WORKSPACE_ONLY" in out["errors"][0]["msg"]
    assert "must be one of" in out["errors"][0]["msg"]

    env[4].services = {}  # force the real Settings.load() env path
    get = await run(env, "config_get", {"key": "server.port"})
    assert get.success is False
    assert "LEO_SECURITY_WORKSPACE_ONLY" in (get.error or "")


async def test_config_validate_reports_bad_yaml(env):
    bad = env[0] / "bad.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    result = await run(env, "config_validate", {"config_path": str(bad)})
    assert result.success is True, result.error
    out = result.output
    assert out["valid"] is False
    assert out["errors"][0]["type"] == "config_error"
    assert out["errors"][0]["loc"] == ["config", str(bad)]
    assert "Config root must be a mapping" in out["errors"][0]["msg"]


# --------------------------------------------------------------------------- executor gates


async def test_invalid_arguments_never_reach_handler(env):
    missing_key = await run(env, "config_get", {})
    assert missing_key.success is False
    assert missing_key.decision == "invalid_arguments"
    assert "key" in (missing_key.error or "")
    extra = await run(env, "config_sources", {"unknown": 1})
    assert extra.decision == "invalid_arguments"
    assert "unknown" in (extra.error or "")
    rows = await tool_rows("config_get")
    assert rows[0]["decision"] == "invalid_arguments" and rows[0]["success"] == 0


async def test_config_path_outside_allowed_roots_is_denied(env, tmp_path):
    outside = tmp_path.parent / "outside-config-dir" / "config.yaml"
    result = await run(env, "config_get", {"key": "server.port", "config_path": str(outside)})
    assert result.success is False
    assert result.decision == "denied"
    assert "path outside allowed roots" in (result.error or "")
    rows = await tool_rows("config_get")
    assert rows[0]["decision"] == "denied"


# --------------------------------------------------------------------------- independent path and input caps


async def test_config_path_confined_even_with_workspace_only_disabled(env, tmp_path):
    executor = env[4]
    executor.permissions.settings.security.workspace_only = False
    executor.services = {}  # force the real Settings.load(config_path) path
    inside = tmp_path / "inside.yaml"
    inside.write_text("server:\n  port: 8123\n", encoding="utf-8")

    ok = await run(env, "config_get", {"key": "server.port", "config_path": str(inside)})
    assert ok.success, ok.error
    assert ok.output["value"] == 8123

    outside = tmp_path.parent / "outside-config-dir" / "config.yaml"
    denied = await run(env, "config_get", {"key": "server.port", "config_path": str(outside)})
    assert denied.success is False
    assert "config path outside allowed roots" in (denied.error or "")

    symlink = tmp_path / "escaping.yaml"
    symlink.symlink_to(outside)
    via_link = await run(env, "config_get", {"key": "server.port", "config_path": str(symlink)})
    assert via_link.success is False
    assert "config path outside allowed roots" in (via_link.error or "")

    # The implicit path honours LEO_CONFIG but is confined by the same rule.
    env[4].services = {}
    implicit_fallback = await run(env, "config_get", {"key": "server.port"})
    assert implicit_fallback.success, implicit_fallback.error  # LEO_CONFIG lives in tmp_path


async def test_config_yaml_byte_and_key_caps(env):
    oversized = env[0] / "oversized.yaml"
    oversized.write_text("#" * (config_tools.MAX_YAML_BYTES + 1), encoding="utf-8")
    huge = await run(env, "config_get", {
        "key": "server.port", "config_path": str(oversized),
    })
    assert huge.success is False
    assert "exceeds" in (huge.error or "") and "bytes" in (huge.error or "")

    many_keys = env[0] / "many-keys.yaml"
    lines = ["top:"]
    lines.extend(f"  key-{index}: {index}" for index in range(config_tools.MAX_YAML_KEYS + 1))
    many_keys.write_text("\n".join(lines) + "\n", encoding="utf-8")
    capped = await run(env, "config_get", {
        "key": "server.port", "config_path": str(many_keys),
    })
    assert capped.success is False
    assert "exceeds" in (capped.error or "") and "keys" in (capped.error or "")

    data_capped = await run(env, "config_validate", {
        "data": {f"key-{index}": index for index in range(config_tools.MAX_YAML_KEYS + 1)},
    })
    assert data_capped.success is True
    assert data_capped.output["valid"] is False
    assert any("exceeds" in error["msg"] for error in data_capped.output["errors"])
