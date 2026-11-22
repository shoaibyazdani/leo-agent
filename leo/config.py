"""Configuration loading. Secrets are referenced by environment-variable name, never stored."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# Environment-variable names are identifiers, never secret values. Keeping the
# strict grammar here prevents a credential accidentally being pasted into a
# "*_env" field (which would then be committed inside the config file).
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def _parse_env_bool(env_name: str, raw: str) -> bool:
    """Parse a boolean env override strictly (never fail open on a typo)."""
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    allowed = ", ".join(sorted(_TRUE_TOKENS | _FALSE_TOKENS))
    raise ValueError(f"{env_name} must be one of {allowed}; got {raw!r}")


_MOCK_FALLBACK_ENV = "LEO_ALLOW_MOCK_FALLBACK"
_MOCK_FALLBACK_MODELS_ENV = "LEO_MODELS_ALLOW_MOCK_FALLBACK"


def _mock_fallback_env_override() -> bool | None:
    """Return the strict env opt-in for automatic mock injection, if set.

    Both the short form and the generic ``LEO_MODELS_<FIELD>`` form are
    accepted. An invalid token or a conflict between the two fails closed.
    """
    values: list[bool] = []
    for env_name in (_MOCK_FALLBACK_ENV, _MOCK_FALLBACK_MODELS_ENV):
        if env_name in os.environ:
            values.append(_parse_env_bool(env_name, os.environ[env_name]))
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{_MOCK_FALLBACK_ENV} conflicts with {_MOCK_FALLBACK_MODELS_ENV}")
    return values[0]


def _validate_env_name(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not _ENV_NAME_RE.match(value):
        raise ValueError(
            f"{field} must be an environment-variable NAME, not a value; got {value!r}"
        )
    return value


class PathConfig(BaseModel):
    data: Path = Path(".leo/data")
    logs: Path = Path(".leo/logs")
    workspaces: Path = Path(".leo/workspaces")
    exports: Path = Path(".leo/exports")
    # Whitelist root for outbound attachments. Configured here, resolved by the
    # composition root and handed to the Gateway once: no per-call argument can
    # widen it, so a tool call cannot turn the whitelist into "anywhere".
    outbound_media: Path = Path(".leo/media/outbound")


class SecurityConfig(BaseModel):
    workspace_only: bool = True
    allow_medium_risk: bool = True
    high_risk_requires_approval: bool = True
    # Smart shell policy. The hardline blocklist is always enforced.
    # smart: read-only/test commands run, risky/unknown commands ask, hardline
    # denies. manual: every shell command asks. off: risky commands run (yolo),
    # hardline still denies. deny_commands are glob patterns that always deny.
    shell_enabled: bool = True
    shell_mode: Literal["smart", "manual", "off"] = "smart"
    # One command, one deadline. 180s is the default an operator expects for a
    # real command; a hardcoded 30s ceiling elsewhere silently overrode this.
    shell_timeout_seconds: int = Field(default=180, ge=1, le=3600)
    shell_max_output_bytes: int = Field(default=200_000, ge=1_000, le=2_000_000)
    deny_commands: list[str] = Field(default_factory=list)
    approval_timeout_seconds: int = 900
    max_tool_calls_per_task: int = 200
    # Server-side ceiling: no caller hint (or configured per-task cap) may
    # exceed this value. 200 matches the legacy default cap.
    max_tool_calls_ceiling: int = 200
    max_agent_depth: int = 3
    redact_patterns: list[str] = Field(
        default_factory=lambda: [
            # Assignment-shaped secrets (keep first: legacy behaviour).
            r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+",
            r"sk-[A-Za-z0-9_-]{12,}",
            r"gh[pousr]_[A-Za-z0-9]{20,}",
            # Cloud/provider key families.
            r"\b(?:AKIA|ASIA|AIDA|AROA|AGPA|AIPA|ANPA|ANVA)[A-Z0-9]{16}\b",
            r"(?i)\baws[_-]?(?:secret[_-]?)?(?:access[_-]?)?key\s*[:=]\s*\S+",
            r"\bAIza[0-9A-Za-z_-]{35}\b",
            r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
            r"github_pat_[A-Za-z0-9_]{20,}",
            r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b",
            # Slack workspace/bot/user tokens.
            r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
            # JSON Web Tokens (base64url header.payload.signature).
            r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
            # Credentials embedded in URLs (basic auth).
            r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@",
            # PEM private-key blocks, including the enclosing markers.
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
        ]
    )
    allowed_roots: list[Path] = Field(default_factory=list)
    # Operator opt-in for additional terminal executables (basenames). The
    # builtin allowlist stays in force; this only extends it.
    allowed_executables: list[str] = Field(default_factory=list)
    # Capability grants available to tool callers. ``None`` means "no explicit
    # grant policy was configured" (legacy risk-only enforcement); an explicit
    # list — including an empty one — makes ToolSpec.permissions mandatory and
    # fails closed on any missing/partial grant.
    grants: list[str] | None = None


class ModelProviderConfig(BaseModel):
    kind: str = "anthropic-compatible"  # anthropic-compatible | openai-compatible | mock
    base_url: str | None = None
    api_key_env: str | None = None
    model: str = ""
    capabilities: list[str] = Field(
        default_factory=lambda: ["text", "streaming", "tools", "structured", "usage"]
    )
    context_window: int = 128_000
    enabled: bool = True
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0
    timeout_seconds: float = 120.0
    max_retries: int = 2
    # Non-secret headers required by some gateways, e.g. x-opencode-session.
    # Secret header VALUES are never stored: use extra_header_env instead.
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_header_env: dict[str, str] = Field(default_factory=dict)

    @field_validator("api_key_env")
    @classmethod
    def _check_api_key_env(cls, value: str | None) -> str | None:
        return _validate_env_name("api_key_env", value)

    @field_validator("extra_header_env")
    @classmethod
    def _check_extra_header_env(cls, values: dict[str, str]) -> dict[str, str]:
        for key, env_name in values.items():
            _validate_env_name(f"extra_header_env[{key!r}]", env_name)
        return values


class ModelConfig(BaseModel):
    default: str = "opencode-go"
    # No implicit mock fallback: an unset fallback means exactly that. Mock is
    # only usable when explicitly configured or when the opt-in below is set.
    fallback: str | None = None
    providers: dict[str, ModelProviderConfig] = Field(default_factory=dict)
    escalation_limit: int = 2
    temperature: float = 0.2
    # Opt-in for *automatic* mock injection when nothing live is routable.
    # An explicit ``kind: mock`` provider in config is itself an opt-in and
    # remains legal without this flag.
    allow_mock_fallback: bool = False


class AutomationConfig(BaseModel):
    tick_seconds: float = 1.0
    recovery_on_startup: bool = True
    max_concurrent_tasks: int = 4
    default_task_timeout_seconds: int = 3600


class ComputerConfig(BaseModel):
    """Explicit opt-in for live desktop actions.

    The default is dry-run only: no platform probe, environment variable or
    caller-supplied ``dry_run=False`` may turn on live actions by itself.
    """

    model_config = {"extra": "allow"}

    live_actions: bool = False


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class ChannelPolicyConfig(BaseModel):
    """Per-channel inbound admission policy. Fail closed: an external channel
    with an empty allowlist denies everyone rather than falling open."""

    mode: Literal["open", "allowlist", "deny"] = "allowlist"
    allow: list[str] = Field(default_factory=list)  # sender ids; "*" allows anyone


class ChannelsConfig(BaseModel):
    """Single source of truth for who may reach LEO through which surface.

    ``internal`` surfaces are the local user's own interfaces (CLI, API,
    dashboard) and are always allowed. Every external channel resolves through
    exactly one policy — no layered fallbacks.
    """

    internal: list[str] = Field(default_factory=lambda: ["cli", "api", "dashboard", "local"])
    default_mode: Literal["open", "allowlist", "deny"] = "allowlist"
    policies: dict[str, ChannelPolicyConfig] = Field(default_factory=dict)


class Settings(BaseModel):
    app_name: str = "LEO"
    environment: str = "local"
    paths: PathConfig = Field(default_factory=PathConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    computer: ComputerConfig = Field(default_factory=ComputerConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    # Tool-executing rounds one chat turn may use before it must answer. Bounded
    # so a runaway loop cannot burn a budget, but large enough for the normal
    # read -> edit -> test -> retry cycle to finish inside one turn.
    max_chat_tool_rounds: int = Field(default=250, ge=1, le=2000)

    @property
    def root(self) -> Path:
        return Path.cwd()

    def resolve(self, p: Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def db_path(self) -> Path:
        return self.resolve(self.paths.data) / "leo.db"

    def model_status(self) -> dict[str, Any]:
        """Explicit, deterministic live / simulated / blocked config status."""
        enabled = {
            name: cfg for name, cfg in self.models.providers.items() if cfg.enabled
        }
        live = sorted(name for name, cfg in enabled.items() if cfg.kind != "mock")
        mocks = sorted(name for name, cfg in enabled.items() if cfg.kind == "mock")
        # "blocked" is distinct from "simulated": with no provider at all there
        # is no mock output to serve, and callers must not claim either live or
        # simulated operation.
        simulated = bool(mocks) and not live
        blocked = not live and not mocks
        if blocked:
            status = "blocked"
            reason = (
                "no enabled model provider is configured; configure a live provider "
                "or set models.allow_mock_fallback=true (LEO_ALLOW_MOCK_FALLBACK=true) "
                "for deterministic offline mock output"
            )
        elif simulated:
            status = "simulated"
            reason = "only explicitly configured mock providers are enabled; output is simulated"
        else:
            status = "live"
            reason = None
        return {
            "simulated": simulated,
            "status": status,
            "reason": reason,
            "default": self.models.default,
            "fallback": self.models.fallback,
            "live_providers": live,
            "mock_providers": mocks,
            "allow_mock_fallback": self.models.allow_mock_fallback,
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        """Load YAML config with environment overrides.

        Provider endpoint/model are sourced from the user's actual environment when
        present. LEO never invents endpoints, credentials or model identifiers.
        """
        raw: dict[str, Any] = {}
        cfg_path = Path(path) if path else Path(os.getenv("LEO_CONFIG", "leo.yaml"))
        if cfg_path.exists():
            try:
                loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML in config {cfg_path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"Config root must be a mapping: {cfg_path}")
            raw = loaded
        settings = cls.model_validate(raw)

        # Explicit opt-in for automatic mock injection. Strict parsing: a typo
        # must never silently enable (or disable) mock fallback.
        mock_override = _mock_fallback_env_override()
        if mock_override is not None:
            settings.models.allow_mock_fallback = mock_override

        # Environment-sourced provider configuration (values, not secret values).
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        model = os.getenv("ANTHROPIC_MODEL")
        token_env = "ANTHROPIC_AUTH_TOKEN" if os.getenv("ANTHROPIC_AUTH_TOKEN") else None
        if "opencode-go" not in settings.models.providers and (base_url or token_env):
            settings.models.providers["opencode-go"] = ModelProviderConfig(
                base_url=base_url,
                api_key_env=token_env,
                model=model or "hy4-flash-free",
                # Anthropic-compatible gateways accept image content blocks;
                # `vision` lets image-bearing turns route here instead of
                # failing with "no provider satisfies the request".
                capabilities=["text", "streaming", "tools", "structured", "usage", "vision"],
            )
        # Automatic mock injection is opt-in only. An explicit ``kind: mock``
        # provider needs no flag; without either, a missing or broken live
        # provider stays blocked instead of silently answering with mock.
        if settings.models.allow_mock_fallback:
            settings.models.providers.setdefault(
                "mock",
                ModelProviderConfig(
                    kind="mock",
                    model="mock-1",
                    capabilities=["text", "streaming", "tools", "structured", "usage", "vision"],
                ),
            )

        def _usable(name: str | None) -> bool:
            provider = settings.models.providers.get(name or "")
            return provider is not None and provider.enabled

        enabled = [name for name, p in settings.models.providers.items() if p.enabled]
        # Repair dead default/fallback only to providers that exist in the
        # configuration; never synthesize a mock and never point at a dead name.
        # When no provider is usable, leave the values as declared: the router
        # reports a blocked model state and fails closed when called.
        if not _usable(settings.models.default):
            live = [name for name in enabled if settings.models.providers[name].kind != "mock"]
            if "opencode-go" in live:
                settings.models.default = "opencode-go"
            elif live:
                settings.models.default = live[0]
            elif enabled:
                settings.models.default = enabled[0]
        if not _usable(settings.models.fallback):
            alternatives = [name for name in enabled if name != settings.models.default]
            # Prefer a live provider; a mock is only usable here when the
            # operator explicitly configured (or opted into) one.
            alternatives.sort(key=lambda name: settings.models.providers[name].kind == "mock")
            settings.models.fallback = alternatives[0] if alternatives else None
        if os.getenv("LEO_ENVIRONMENT"):
            settings.environment = os.environ["LEO_ENVIRONMENT"]
        for group, fields in {
            "paths": ["data", "logs", "workspaces", "exports", "outbound_media"],
            "security": [
                "workspace_only", "allow_medium_risk", "high_risk_requires_approval",
                "max_tool_calls_per_task", "max_tool_calls_ceiling",
            ],
            "automation": ["max_concurrent_tasks", "default_task_timeout_seconds"],
        }.items():
            target = getattr(settings, group)
            for field in fields:
                env_name = f"LEO_{group.upper()}_{field.upper()}"
                if env_name in os.environ:
                    value: Any = os.environ[env_name]
                    current = getattr(target, field)
                    if isinstance(current, bool):
                        value = _parse_env_bool(env_name, value)
                    elif isinstance(current, int):
                        try:
                            value = int(value)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"{env_name} must be an integer; got {value!r}") from exc
                    elif isinstance(current, Path):
                        value = Path(value)
                    setattr(target, field, value)
        return settings

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.data, self.paths.logs, self.paths.workspaces,
            self.paths.exports, self.paths.outbound_media,
        ):
            self.resolve(p).mkdir(parents=True, exist_ok=True)
