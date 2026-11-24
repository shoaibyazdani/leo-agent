"""Configuration loading. Secrets are referenced by environment-variable name, never stored."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

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


def _is_opencode_inference_gateway(url: str | None) -> bool:
    """True only for the OpenCode Go inference endpoint LEO is built against.

    OpenCode exposes several surfaces. The IDE's ``/zen/go`` endpoint is a
    different product path, and a shared shell often exports it (or another
    Anthropic-compatible gateway) as ``ANTHROPIC_BASE_URL``. LEO must not adopt
    that value as its own opencode-go provider: this project's gateway is the
    ``/inference/go/anthropic`` endpoint, and only that one gets the session
    header this provider needs.
    """
    if not url:
        return False
    try:
        parsed = urlparse(str(url))
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")
    return host == "opencode.ai" and path == "/inference/go/anthropic"


def upstream_provider(config: Any) -> str | None:
    """Name the service that actually serves a model slot's requests.

    Explicit ``provider:`` wins. Otherwise the base_url host is the evidence we
    have: a slot keyed ``deepseek-v4.1-flash`` pointed at
    ``https://opencode.ai/zen/go`` is a model ON opencode.ai, not a provider
    called "deepseek-v4.1-flash". Returns ``None`` when neither exists rather
    than inventing a name.
    """
    if config is None:
        return None
    explicit = getattr(config, "provider", None)
    if not explicit and isinstance(config, dict):
        explicit = config.get("provider")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    kind = getattr(config, "kind", None)
    if not kind and isinstance(config, dict):
        kind = config.get("kind")
    if str(kind or "").strip().lower() == "mock":
        return "mock"
    base_url = getattr(config, "base_url", None)
    if not base_url and isinstance(config, dict):
        base_url = config.get("base_url")
    if not base_url:
        return None
    try:
        host = (urlparse(str(base_url)).hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    for prefix in ("api.", "www."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host or None


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


class ApprovalNotifyConfig(BaseModel):
    """Where a pending HIGH-risk approval is announced to the operator.

    Without this, a request only lands in SQLite and ``GET /approvals``, so a
    gated action stalls silently until it expires. ``enabled`` is False by
    default: the operator opts in by naming a channel and a target they own.
    Notification never grants authority — it only tells a human that a decision
    is waiting.
    """

    enabled: bool = False
    channel: str = ""
    # The operator's own address on that channel. Never a value taken from a
    # model or a chat message.
    target: str = ""


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
    # Hold a chat turn while a HIGH-risk approval is pending, bounded by
    # approval_timeout_seconds. Approved -> the same tool call resumes;
    # denied/expired/timed out -> a denial the model can answer, instead of
    # ending the turn and forgetting that the human already approved. Off by
    # default so CLI/API callers keep the immediate ApprovalRequired contract;
    # the shipped leo.yaml enables it for chat surfaces.
    wait_for_approval: bool = False
    approval_poll_seconds: float = Field(default=2.0, ge=0.05, le=60.0)
    # How HIGH-risk calls are handled:
    #   smart  - allow normal workspace shell under a workspace trust; prompt for
    #            the rest with once / this-command / trust-workspace options
    #   manual - always prompt (approval grants are ignored)
    #   off    - run HIGH-risk calls without ordinary prompts (hardline/deny and
    #            path rules still apply)
    approval_mode: Literal["smart", "manual", "off"] = "smart"
    # How long the "Trust workspace" button authorizes normal shell in one
    # workspace. The reference tools default to "until revoked"; 30 days is the
    # shipped middle ground and is configurable.
    workspace_trust_days: float = Field(default=30.0, ge=0.0, le=3650.0)
    # In smart mode, ask a cheap model to review a HIGH shell call that no
    # grant covers before bothering a human: allow once, deny with a reason, or
    # ask. Destructive/credential/privilege/arbitrary-code reasons are never
    # sent to the reviewer; they still require a human.
    reviewer_enabled: bool = True
    # Empty means the configured default chat model; set a provider key or model
    # id to pin a cheaper reviewer.
    reviewer_model: str = ""
    reviewer_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
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
    notify: ApprovalNotifyConfig = Field(default_factory=ApprovalNotifyConfig)


class ModelProviderConfig(BaseModel):
    """One model slot on one upstream service.

    The dict key in ``models.providers`` names the SLOT, which in a
    single-gateway install is the model id itself (``deepseek-v4.1-flash``).
    It is not a provider: one gateway (``base_url``) can serve every slot.
    ``provider`` names that upstream service explicitly; when it is omitted,
    :func:`upstream_provider` falls back to the base_url host so status output
    can still name the real provider instead of echoing a model id.
    """

    kind: str = "anthropic-compatible"  # anthropic-compatible | openai-compatible | mock
    base_url: str | None = None
    api_key_env: str | None = None
    model: str = ""
    # Upstream provider name, e.g. "opencode-go". Optional: derived from
    # base_url when absent, so no config has to be rewritten to be truthful.
    provider: str | None = None
    capabilities: list[str] = Field(
        default_factory=lambda: ["text", "streaming", "tools", "structured", "usage"]
    )
    context_window: int = 128_000
    enabled: bool = True
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0
    timeout_seconds: float = 120.0
    max_retries: int = 2
    # A provider's Retry-After above this cap fails over immediately instead of
    # blocking an interactive turn.
    max_retry_after_seconds: float = Field(default=60.0, ge=0.0, le=600.0)
    retry_jitter: bool = True
    # Output-token ceiling per turn. A reasoning model spends this budget on
    # thinking first; too small a ceiling produces a turn with no text and no tool
    # call, which is refused as an empty response.
    max_tokens: int = 32_000
    # Non-secret headers required by some gateways, e.g. x-opencode-session.
    # Secret header VALUES are never stored: use extra_header_env instead.
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_header_env: dict[str, str] = Field(default_factory=dict)
    # Optional per-model reasoning effort (none|minimal|low|medium|high|xhigh|
    # max|ultra). Unset means the global models.reasoning_effort applies. This
    # is a config default; /reasoning overrides it live for the session.
    reasoning_effort: str | None = None
    # Optional narrowing of the wire vocabulary this model accepts. When unset,
    # the widest wire for the adapter is assumed and the level is clamped onto
    # it. Declare the real set when a gateway rejects a level with HTTP 400.
    reasoning_efforts: list[str] = Field(default_factory=list)
    # Optional vendor mapping applied before nearest-level clamping, for wires
    # where a level's positional meaning differs (e.g. {"medium": "high"}).
    reasoning_overrides: dict[str, str] = Field(default_factory=dict)
    # How effort is expressed on the wire:
    #   auto   - adaptive effort for modern Claude models, thinking budget for
    #            other Anthropic-compatible models, reasoning_effort for OpenAI
    #   effort - adaptive effort / output_config.effort
    #   budget - thinking.budget_tokens (legacy Anthropic models)
    #   off    - never send a reasoning parameter for this provider
    reasoning_style: Literal["auto", "effort", "budget", "off"] = "auto"
    # Whether the provider may return a readable reasoning SUMMARY. Raw
    # chain-of-thought is never forwarded to a reply.
    reasoning_summaries: bool = True

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
    # A rate-limited or transiently failing provider is skipped for this long,
    # so the next call fails over instead of hammering it.
    failover_on_transient: bool = True
    rate_limit_cooldown_seconds: float = Field(default=30.0, ge=0.0, le=3_600.0)
    rate_limit_cooldown_max_seconds: float = Field(default=3_600.0, ge=0.0, le=86_400.0)
    transient_cooldown_seconds: float = Field(default=10.0, ge=0.0, le=3_600.0)
    temperature: float = 0.2
    # Global default reasoning effort. Unset means no reasoning parameter is
    # sent; a per-provider reasoning_effort or a live /reasoning overrides it.
    reasoning_effort: str | None = None
    # Global default for concise reasoning summaries in replies. Never exposes
    # raw chain-of-thought: only provider summaries and execution notes.
    reasoning_show: bool = False
    # Opt-in for *automatic* mock injection when nothing live is routable.
    # An explicit ``kind: mock`` provider in config is itself an opt-in and
    # remains legal without this flag.
    allow_mock_fallback: bool = False


class CompactionConfig(BaseModel):
    """Durable conversation compaction policy.

    Only names and numbers live here; the summarizer is whichever model the
    router already routes to unless ``model`` pins one.
    """

    enabled: bool = True
    # Fraction of the context budget that triggers automatic compaction.
    threshold: float = 0.9
    # Estimated tokens kept verbatim after the compacted boundary.
    keep_recent_tokens: int = 20_000
    # Character cap for the summarizer input, so one huge session cannot
    # build an unbounded prompt.
    summary_input_chars: int = 160_000
    # Corrective attempts after a failed quality audit.
    corrective_attempts: int = 2
    # ``strict`` requires opaque identifiers to survive the summary.
    identifier_policy: str = "strict"
    # Persist durable facts from the span leaving model context.
    memory_flush: bool = True
    # Automatic behaviour when summarization times out or returns nothing.
    on_timeout: str = "deterministic"
    # Optional summarizer override; empty means the active session model.
    model: str = ""
    # Carry the first real user ask through the durable summary verbatim, so an
    # interrupted/resumed turn never loses its objective.
    protect_objective: bool = True
    objective_chars: int = Field(default=4_000, ge=200, le=20_000)


class ContextConfig(BaseModel):
    """Per-turn context shaping, separate from durable compaction."""

    max_message_chars: int = Field(default=12_000, ge=0)
    tool_output_chars: int = Field(default=2_000, ge=0)
    max_memories: int = Field(default=5, ge=0)
    compact: bool = True
    min_memory_score: float = Field(default=0.0, ge=0.0)
    protect_head_messages: int = Field(default=2, ge=0, le=50)
    protect_head_tokens: int = Field(default=4_000, ge=0)
    protect_tail_messages: int = Field(default=2, ge=0, le=50)
    protect_tail_tokens: int = Field(default=6_000, ge=0)


class AutoNoteConfig(BaseModel):
    """Autonomous memory-note learning.

    LEO derives notes from its own durable history (the operator's statements,
    bounded tool/run trajectories) and, by default, writes the safe ones
    without asking. The sanitizer still drops secrets and instruction-shaped
    text, every note carries evidence hashes, dedupe runs before the write, and
    a supersede of an existing memory stays approval-bound.

    Set ``enabled: false`` to restore the approval gate for every note.
    """

    enabled: bool = True
    # Source kinds the engine may write unattended. ``external`` is refused by
    # the policy validator, so web/tool-injected text cannot become a note.
    trusted_sources: list[str] = Field(
        default_factory=lambda: ["user", "tool", "run"]
    )
    # Categories eligible for an unattended write. Personal/project facts are
    # included because they are exactly what the operator expects LEO to
    # remember; remove them to keep those approval-bound.
    allowed_categories: list[str] = Field(
        default_factory=lambda: [
            "working", "episodic", "semantic", "procedural",
            "personal", "project", "organizational",
        ]
    )
    max_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    promote_facts: bool = False
    # Required for the personal/organizational/project categories above.
    allow_high_risk_categories: bool = True
    allow_unattended_writes: bool = True


class LearningConfig(BaseModel):
    auto_notes: AutoNoteConfig = Field(default_factory=AutoNoteConfig)


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


class StallGuardConfig(BaseModel):
    """Bounded re-prompt when a turn announces work and then stops."""

    enabled: bool = True
    max_nudges: int = Field(default=1, ge=0, le=2)
    max_reply_chars: int = Field(default=400, ge=100, le=2_000)
    tail_chars: int = Field(default=160, ge=40, le=1_000)


class ProgressConfig(BaseModel):
    """Live "what am I doing right now" report for a working turn.

    One message per turn, edited in place. Off by default: it is a chat surface,
    and a deployment that never talks to a human should not open one.
    """

    enabled: bool = False
    min_interval_seconds: float = Field(default=4.0, ge=0.5, le=60.0)
    max_steps: int = Field(default=6, ge=1, le=20)
    idle_seconds: float = Field(default=45.0, ge=5.0, le=600.0)
    # A report is only worth opening for real work. A turn that answers from
    # memory in one call must stay silent: a "done" footer under every reply
    # reads as noise, not as progress.
    min_steps: int = Field(default=6, ge=1, le=20)
    significant_tools: list[str] = Field(
        default_factory=lambda: [
            "shell", "terminal", "patch", "write_file", "file_move", "file_copy",
            "git_*", "browser_navigate",
        ]
    )
    # A background task is silent by nature: the operator walked away. This is
    # the ceiling on that silence, measured from the last real step. It is a
    # heartbeat, not fake progress: the message refreshes elapsed time and the
    # steps already done, and never invents a percentage.
    task_heartbeat_seconds: float = Field(default=45.0, ge=30.0, le=60.0)
    # How many finished steps a task report keeps on screen. A task can run for
    # hours; the tail is what the reader wants, not the whole history.
    task_max_steps: int = Field(default=6, ge=1, le=20)


class Settings(BaseModel):
    app_name: str = "LEO"
    environment: str = "local"
    paths: PathConfig = Field(default_factory=PathConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    stall_guard: StallGuardConfig = Field(default_factory=StallGuardConfig)
    progress: ProgressConfig = Field(default_factory=ProgressConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
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
        # The shim exists for env-only deployments of LEO's own OpenCode Go
        # gateway. It activates only when ANTHROPIC_BASE_URL names that exact
        # inference endpoint: the IDE's Anthropic surface and any other shared
        # shell value are ignored, and a token alone never invents an endpoint.
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        model = os.getenv("ANTHROPIC_MODEL")
        token_env = "ANTHROPIC_AUTH_TOKEN" if os.getenv("ANTHROPIC_AUTH_TOKEN") else None
        if (
            "opencode-go" not in settings.models.providers
            and _is_opencode_inference_gateway(base_url)
        ):
            settings.models.providers["opencode-go"] = ModelProviderConfig(
                base_url=base_url,
                api_key_env=token_env,
                model=model or "deepseek-v4.1-flash",
                # Anthropic-compatible gateways accept image content blocks;
                # `vision` lets image-bearing turns route here instead of
                # failing with "no provider satisfies the request".
                capabilities=["text", "streaming", "tools", "structured", "usage", "vision"],
                # Mirrors the shipped leo.yaml: this gateway requires a stable
                # session id and rejects generic library user agents.
                extra_header_env={"x-opencode-session": "LEO_OPENCODE_SESSION"},
                extra_headers={"User-Agent": "leo-agent-os/0.1.0"},
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
                "wait_for_approval", "approval_poll_seconds",
                "approval_mode", "workspace_trust_days",
                "reviewer_enabled", "reviewer_timeout_seconds",
                "max_tool_calls_per_task", "max_tool_calls_ceiling",
            ],
            "automation": ["max_concurrent_tasks", "default_task_timeout_seconds"],
            "compaction": [
                "enabled", "threshold", "keep_recent_tokens",
                "summary_input_chars", "corrective_attempts",
                "identifier_policy", "memory_flush", "on_timeout",
                "protect_objective", "objective_chars",
            ],
            "context": [
                "max_message_chars", "tool_output_chars", "max_memories", "compact",
                "min_memory_score", "protect_head_messages", "protect_head_tokens",
                "protect_tail_messages", "protect_tail_tokens",
            ],
            "stall_guard": ["enabled", "max_nudges", "max_reply_chars", "tail_chars"],
            "models": [
                "escalation_limit", "failover_on_transient",
                "rate_limit_cooldown_seconds", "rate_limit_cooldown_max_seconds",
                "transient_cooldown_seconds",
            ],
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
                    elif isinstance(current, float):
                        try:
                            value = float(value)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"{env_name} must be a number; got {value!r}") from exc
                    elif isinstance(current, Path):
                        value = Path(value)
                    if field == "approval_mode":
                        value = str(value).strip().lower()
                        if value not in {"smart", "manual", "off"}:
                            raise ValueError(
                                f"{env_name} must be smart, manual or off; got {value!r}"
                            )
                    setattr(target, field, value)
        return settings

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.data, self.paths.logs, self.paths.workspaces,
            self.paths.exports, self.paths.outbound_media,
        ):
            self.resolve(p).mkdir(parents=True, exist_ok=True)
