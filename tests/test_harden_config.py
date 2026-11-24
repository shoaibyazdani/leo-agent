"""Regression tests for configuration loading and data-contract hardening.

Covers the defects found during the config/contracts audit:
  * boolean env overrides failed open on unrecognised tokens;
  * default/fallback providers could name a missing or disabled provider;
  * malformed YAML surfaced as an opaque parser error;
  * ``*_env`` fields accepted literal secrets instead of variable names;
  * path resolution / enum boundaries / id generation sanity.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from leo.config import (
    ModelConfig,
    ModelProviderConfig,
    Settings,
    _parse_env_bool,
    upstream_provider,
)
from leo.contracts import (
    MemoryCategory,
    MemoryKind,
    RiskLevel,
    TaskStatus,
    ToolCall,
    new_id,
    utcnow,
)


_CONFIG_ENV_PREFIXES = ("ANTHROPIC_", "LEO_")


def _load(tmp_path: Path, text: str = "", env: dict[str, str] | None = None, monkeypatch=None):
    """Load a config from an isolated file with a sanitised environment."""
    cfg = tmp_path / "leo.yaml"
    cfg.write_text(text, encoding="utf-8")
    for key in list(os.environ):
        if key.startswith(_CONFIG_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LEO_CONFIG", str(cfg))
    return Settings.load(cfg)


# --------------------------------------------------------------- env overrides
@pytest.mark.parametrize("token", ["true", "TRUE", "1", "yes", "On", " true "])
def test_bool_env_true_tokens(token):
    assert _parse_env_bool("LEO_X", token) is True


@pytest.mark.parametrize("token", ["false", "FALSE", "0", "no", "OFF"])
def test_bool_env_false_tokens(token):
    assert _parse_env_bool("LEO_X", token) is False


@pytest.mark.parametrize("token", ["garbage", "2", "truthy", "", "enabled"])
def test_bool_env_unknown_token_is_rejected(token):
    """A typo must never silently disable a security control (fail-open)."""
    with pytest.raises(ValueError, match="LEO_X"):
        _parse_env_bool("LEO_X", token)


@pytest.mark.parametrize(
    "env_name",
    [
        "LEO_SECURITY_WORKSPACE_ONLY",
        "LEO_SECURITY_ALLOW_MEDIUM_RISK",
        "LEO_SECURITY_HIGH_RISK_REQUIRES_APPROVAL",
    ],
)
def test_security_bool_env_garbage_fails_closed(tmp_path, monkeypatch, env_name):
    with pytest.raises(ValueError, match=env_name):
        _load(tmp_path, env={env_name: "definitely-not-a-bool"}, monkeypatch=monkeypatch)


def test_security_bool_env_valid_false_applies(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        env={"LEO_SECURITY_WORKSPACE_ONLY": "false"},
        monkeypatch=monkeypatch,
    )
    assert settings.security.workspace_only is False


def test_int_env_bad_value_has_clear_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="LEO_AUTOMATION_MAX_CONCURRENT_TASKS"):
        _load(
            tmp_path,
            env={"LEO_AUTOMATION_MAX_CONCURRENT_TASKS": "not-a-number"},
            monkeypatch=monkeypatch,
        )


def test_path_env_override_and_resolution(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        env={"LEO_PATHS_DATA": "~/leo-test-data"},
        monkeypatch=monkeypatch,
    )
    assert settings.paths.data == Path("~/leo-test-data")
    # expanduser + cwd-relative resolution
    resolved = settings.resolve(settings.paths.data)
    assert resolved.is_absolute()
    assert str(resolved).endswith("leo-test-data")
    # already-absolute paths are preserved
    absolute = tmp_path / "abs"
    assert settings.resolve(absolute) == absolute
    # db_path derives from the (resolved) data directory
    settings.paths.data = tmp_path / "data"
    assert settings.db_path == (tmp_path / "data" / "leo.db")


# ------------------------------------------------------------ model defaults
def test_missing_default_provider_is_not_silently_repaired_to_mock(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        "models:\n  default: nonexistent\n",
        monkeypatch=monkeypatch,
    )
    # The loader no longer invents a mock provider: the dead default is left
    # visible and the router reports a blocked model state on use.
    assert "mock" not in settings.models.providers
    assert settings.models.default == "nonexistent"
    assert settings.model_status()["status"] == "blocked"


def test_mock_fallback_env_opt_in_injects_mock(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        "models:\n  default: nonexistent\n",
        env={"LEO_ALLOW_MOCK_FALLBACK": "true"},
        monkeypatch=monkeypatch,
    )
    assert settings.models.allow_mock_fallback is True
    assert "mock" in settings.models.providers
    assert settings.models.default == "mock"
    assert settings.model_status()["status"] == "simulated"


def test_mock_fallback_yaml_opt_in_injects_mock(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        "models:\n  default: nonexistent\n  allow_mock_fallback: true\n",
        monkeypatch=monkeypatch,
    )
    assert "mock" in settings.models.providers
    assert settings.models.default == "mock"


def test_mock_fallback_env_opt_in_is_strict(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="LEO_ALLOW_MOCK_FALLBACK"):
        _load(
            tmp_path,
            env={"LEO_ALLOW_MOCK_FALLBACK": "definitely-not-a-bool"},
            monkeypatch=monkeypatch,
        )
    with pytest.raises(ValueError, match="LEO_ALLOW_MOCK_FALLBACK"):
        _load(
            tmp_path,
            env={
                "LEO_ALLOW_MOCK_FALLBACK": "true",
                "LEO_MODELS_ALLOW_MOCK_FALLBACK": "false",
            },
            monkeypatch=monkeypatch,
        )
    disabled = _load(
        tmp_path,
        env={"LEO_ALLOW_MOCK_FALLBACK": "false"},
        monkeypatch=monkeypatch,
    )
    assert disabled.models.allow_mock_fallback is False
    assert "mock" not in disabled.models.providers


def test_explicit_mock_provider_needs_no_opt_in(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        """
models:
  default: offline
  providers:
    offline:
      kind: mock
      model: mock-1
""",
        monkeypatch=monkeypatch,
    )
    assert settings.models.allow_mock_fallback is False
    assert "offline" in settings.models.providers
    status = settings.model_status()
    assert status["status"] == "simulated" and status["simulated"] is True
    assert status["mock_providers"] == ["offline"]


def test_disabled_default_provider_repairs_to_enabled(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        """
models:
  default: custom
  fallback: custom
  providers:
    custom:
      kind: mock
      enabled: false
    offline:
      kind: mock
      model: mock-1
""",
        monkeypatch=monkeypatch,
    )
    assert settings.models.default == "offline"
    assert settings.models.default in settings.models.providers
    assert settings.models.providers[settings.models.default].enabled is True


def test_enabled_default_is_respected(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        """
models:
  default: mine
  providers:
    mine:
      kind: mock
      model: m1
""",
        monkeypatch=monkeypatch,
    )
    assert settings.models.default == "mine"


def test_missing_fallback_repairs_to_distinct_enabled_provider(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        """
models:
  default: mine
  fallback: ghost
  providers:
    mine:
      kind: mock
    backup:
      kind: mock
""",
        monkeypatch=monkeypatch,
    )
    assert settings.models.fallback in settings.models.providers
    assert settings.models.providers[settings.models.fallback].enabled is True
    assert settings.models.fallback == "backup"


def test_dead_fallback_never_silently_points_at_mock(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        """
models:
  default: gateway
  fallback: mock
  providers:
    gateway:
      kind: anthropic-compatible
      base_url: http://127.0.0.1:1
      model: local
""",
        monkeypatch=monkeypatch,
    )
    assert "mock" not in settings.models.providers
    assert settings.models.fallback is None


# --------------------------------------------------------- secret-by-name
@pytest.mark.parametrize("bad", ["sk-ant-1234567890abcdef", "my-token", "abc.def", ""])
def test_api_key_env_rejects_non_names(bad):
    with pytest.raises(ValueError, match="environment-variable NAME"):
        ModelProviderConfig(api_key_env=bad)


def test_api_key_env_accepts_name_and_none():
    assert ModelProviderConfig(api_key_env="ANTHROPIC_AUTH_TOKEN").api_key_env == "ANTHROPIC_AUTH_TOKEN"
    assert ModelProviderConfig(api_key_env=None).api_key_env is None


@pytest.mark.parametrize("bad", ["sk-1234567890abcdef", "a-b", "value with space"])
def test_extra_header_env_rejects_non_names(bad):
    with pytest.raises(ValueError, match="environment-variable NAME"):
        ModelProviderConfig(extra_header_env={"x-opencode-session": bad})


def test_extra_header_env_accepts_names():
    cfg = ModelProviderConfig(extra_header_env={"x-opencode-session": "LEO_OPENCODE_SESSION"})
    assert cfg.extra_header_env["x-opencode-session"] == "LEO_OPENCODE_SESSION"


# ------------------------------------------------------- YAML / pydantic
def test_malformed_yaml_reports_config_path(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Invalid YAML") as exc:
        _load(tmp_path, "models: [unclosed\n", monkeypatch=monkeypatch)
    assert "leo.yaml" in str(exc.value)


def test_non_mapping_yaml_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="must be a mapping"):
        _load(tmp_path, "- a\n- b\n", monkeypatch=monkeypatch)


def test_missing_config_file_uses_defaults(tmp_path, monkeypatch):
    missing = tmp_path / "nope.yaml"
    monkeypatch.setenv("LEO_CONFIG", str(missing))
    for key in list(os.environ):
        if key.startswith(_CONFIG_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    settings = Settings.load(missing)
    assert "mock" not in settings.models.providers  # no implicit injection
    assert settings.models.default == "opencode-go"
    assert settings.models.fallback is None
    assert settings.model_status()["status"] == "blocked"


def test_invalid_config_type_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(default=123)  # type: ignore[arg-type]


# ------------------------------------------------------ contracts sanity
def test_new_id_is_unique_and_prefixed():
    a = new_id("task")
    b = new_id("task")
    assert a != b
    assert a.startswith("task_")
    assert len(a.split("_", 1)[1]) == 32  # uuid4 hex
    assert "_" not in new_id()


def test_tool_call_ids_are_unique():
    assert ToolCall(name="x").id != ToolCall(name="x").id


def test_enum_boundaries():
    assert TaskStatus("completed") is TaskStatus.COMPLETED
    assert TaskStatus.COMPLETED.terminal is True
    assert TaskStatus.RUNNING.terminal is False
    assert RiskLevel("high") is RiskLevel.HIGH
    assert MemoryCategory("episodic") is MemoryCategory.EPISODIC
    assert MemoryKind("fact") is MemoryKind.FACT
    with pytest.raises(ValueError):
        TaskStatus("not-a-status")


def test_utcnow_is_iso_and_timezone_aware():
    value = utcnow()
    assert "T" in value and value.endswith("+00:00")


# ------------------------------------------------------- computer live actions
def test_computer_live_actions_is_a_real_field_defaulting_off(tmp_path, monkeypatch):
    settings = _load(tmp_path, monkeypatch=monkeypatch)
    assert settings.computer.live_actions is False

    enabled = _load(
        tmp_path,
        "computer:\n  live_actions: true\n",
        monkeypatch=monkeypatch,
    )
    assert enabled.computer.live_actions is True


def test_allowed_executables_is_a_real_security_field(tmp_path, monkeypatch):
    settings = _load(
        tmp_path,
        "security:\n  allowed_executables:\n    - mytool\n",
        monkeypatch=monkeypatch,
    )
    assert settings.security.allowed_executables == ["mytool"]
    assert Settings().security.allowed_executables == []


def test_model_status_marks_blocked_simulated_and_live_configurations(tmp_path, monkeypatch):
    blocked = _load(tmp_path, monkeypatch=monkeypatch)
    status = blocked.model_status()
    assert status["simulated"] is False and status["status"] == "blocked"
    assert status["mock_providers"] == [] and status["live_providers"] == []
    assert "no enabled model provider" in status["reason"]

    simulated = _load(
        tmp_path,
        """
models:
  default: offline
  providers:
    offline:
      kind: mock
      model: mock-1
""",
        monkeypatch=monkeypatch,
    )
    status = simulated.model_status()
    assert status["simulated"] is True and status["status"] == "simulated"
    assert status["mock_providers"] == ["offline"] and status["live_providers"] == []

    live = _load(
        tmp_path,
        """
models:
  default: gateway
  providers:
    gateway:
      kind: anthropic-compatible
      base_url: http://127.0.0.1:1
      model: local
""",
        monkeypatch=monkeypatch,
    )
    status = live.model_status()
    assert status["simulated"] is False and status["status"] == "live"
    assert status["live_providers"] == ["gateway"]


# ------------------------------------------------------------ secret redaction
def test_default_redaction_patterns_cover_credential_families():
    patterns = [re.compile(pattern) for pattern in Settings().security.redact_patterns]
    # Provider-shaped values are assembled at runtime so the repository text
    # never contains a literal that hosting secret scanners flag (GitHub push
    # protection and secret scanning match formats even for synthetic values).
    # The runtime strings are unchanged, so redaction coverage is identical.
    samples = {
        "assignment": "api_key: " + "sk-legacyvalue1234567890",
        "openai": "sk-" + "proj-abcdefghijklmnopqrstuvwx",
        "github": "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789",
        "github_fine_grained": "github_" + "pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz",
        "aws_access_key": "AKIA" + "IOSFODNN7EXAMPLE",
        "aws_secret_assignment": "aws_secret_access_key=" + "abcd1234abcd1234abcd1234abcd1234abcd1234",
        "google_api_key": "AIza" + "SyD-1234567890abcdefghijklmnopqrstu",
        "stripe": "sk_" + "live_abcdefghijklmnop",
        "slack_bot": "xox" + "b-abcdefghij",
        "slack_user": "xox" + "p-abcdefghij",
        "jwt": (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        ),
        "basic_auth_url": "https://admin:s3cr3t-password@console.example.internal/v1/path",
        "pem_key": (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n"
            "-----END RSA PRIVATE KEY-----"
        ),
    }
    for label, secret in samples.items():
        redacted = secret
        for pattern in patterns:
            redacted = pattern.sub("[REDACTED]", redacted)
        assert "[REDACTED]" in redacted, label
        assert secret not in redacted, label


def test_redaction_leaves_ordinary_text_and_urls_untouched():
    from leo.tools.executor import redact_value

    patterns = [re.compile(p) for p in Settings().security.redact_patterns]
    ordinary = "Deploy finished at 10:00; see https://example.com/docs/page for details."
    assert redact_value(ordinary, patterns) == ordinary
    nested = {"url": "https://admin:hunter2@example.internal/", "count": 2}
    cleaned = redact_value(nested, patterns)
    assert "hunter2" not in cleaned["url"]
    assert "[REDACTED]" in cleaned["url"]
    assert cleaned["count"] == 2


# --------------------------------------------------------------------------- provider identity


def test_upstream_provider_names_the_service_not_the_model_slot():
    """A slot keyed by a model id must not be reported as a provider.

    The live install keys `models.providers` by model id (one gateway serves
    them all), so `health.providers` used to advertise 31 "providers" that were
    really 31 models on opencode.ai. Deriving the host is what makes status
    truthful without rewriting leo.yaml.
    """
    cfg = ModelProviderConfig(
        kind="openai-compatible", model="deepseek-v4.1-flash",
        base_url="https://opencode.ai/zen/go",
    )
    assert upstream_provider(cfg) == "opencode.ai"


def test_upstream_provider_prefers_an_explicit_provider_name():
    cfg = ModelProviderConfig(
        kind="openai-compatible", model="deepseek-v4.1-flash",
        base_url="https://opencode.ai/zen/go", provider="opencode-go",
    )
    assert upstream_provider(cfg) == "opencode-go"


def test_upstream_provider_strips_api_prefix_and_reports_mock():
    assert upstream_provider(
        ModelProviderConfig(base_url="https://api.groq.com/openai/v1")
    ) == "groq.com"
    assert upstream_provider(ModelProviderConfig(kind="mock")) == "mock"


def test_upstream_provider_returns_none_rather_than_inventing_a_name():
    assert upstream_provider(None) is None
    assert upstream_provider(ModelProviderConfig(model="orphan")) is None


def test_health_catalog_reports_one_provider_for_a_shared_gateway(tmp_path):
    """Slots on one gateway resolve to one provider, not N model-named providers."""
    from leo.config import PathConfig

    settings = Settings()
    settings.paths = PathConfig(data=tmp_path / "data", logs=tmp_path / "logs")
    settings.models.providers = {
        "deepseek-v4.1-flash": ModelProviderConfig(
            kind="openai-compatible", model="deepseek-v4.1-flash",
            base_url="https://opencode.ai/zen/go",
        ),
        "glm-5.3": ModelProviderConfig(
            kind="openai-compatible", model="glm-5.3",
            base_url="https://opencode.ai/zen/go",
        ),
    }
    settings.models.default = "deepseek-v4.1-flash"

    entries = [
        {"name": name, "model": cfg.model, "provider": upstream_provider(cfg), "enabled": cfg.enabled}
        for name, cfg in sorted(settings.models.providers.items())
    ]
    assert {entry["provider"] for entry in entries} == {"opencode.ai"}
    # The default slot is a model, so the reported model must be the model id.
    default_cfg = settings.models.providers[settings.models.default]
    assert upstream_provider(default_cfg) == "opencode.ai"
    assert str(default_cfg.model) == "deepseek-v4.1-flash"
