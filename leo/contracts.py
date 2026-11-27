"""Shared, documented data contracts between LEO modules.

Modules depend on these types and on the public methods documented here; they
must not import one another's internals. This keeps the modular monolith free of
circular dependencies.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Iterator, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex
    return f"{prefix}_{value}" if prefix else value


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# Shared lease bound for a one-shot live schedule run/cancel authorization.
# The schedule service and the automation specialist must agree on when an
# unfinished claim becomes stale and can be reclaimed.
LIVE_AUTH_CLAIM_TTL_SECONDS = 300


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class MemoryCategory(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PERSONAL = "personal"
    PROJECT = "project"
    ORGANIZATIONAL = "organizational"


class MemoryKind(str, Enum):
    FACT = "fact"
    ASSUMPTION = "assumption"
    TEMPORARY = "temporary"
    UNVERIFIED = "unverified"


class ModelCapability(str, Enum):
    TEXT = "text"
    STREAMING = "streaming"
    TOOLS = "tools"
    STRUCTURED = "structured"
    USAGE = "usage"
    VISION = "vision"


class BaseRecord(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolSpec(BaseRecord):
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "leo-native"
    category: str = "general"
    risk: RiskLevel = RiskLevel.LOW
    permissions: list[str] = Field(default_factory=list)
    # Dry-run normally short-circuits before the handler. Tools that can
    # validate fully without side effects (e.g. terminal allowlist checks)
    # opt in so a dry-run reports the real decision.
    supports_dry_run: bool = False
    # Opt-in one-shot approval consumption on successful dispatch (shell).
    consume_approval: bool = False
    # Opt-in pre-persistence secret/unsafe-text preflight. The executor scans
    # the call arguments before authorization, quota charging, handler dispatch
    # and durable recording; a refusal persists only a digest and category.
    secret_scan: bool = False
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    output_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseRecord):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    # Trusted per-turn identity, copied from TurnContext.tool_fields() at the
    # entry point. Unlike correlation_id (a routing value often shared by every
    # call in one conversation), this is unique to one turn. Never derive it
    # from tool arguments: handlers and model output cannot widen provenance.
    turn_id: str | None = None
    actor: str = "assistant"
    dry_run: bool = False
    # Trusted provenance, populated at trusted construction sites (chat, API,
    # gateway and agent delegation). ``source`` is the provenance class
    # (chat/api/gateway/agent/operator/scheduled/unattended); ``origin`` names
    # the concrete origin (channel, agent id, api principal). A handler cannot
    # set these to widen authority.
    source: str = ""
    origin: str | None = None
    principal: str | None = None
    conversation_id: str | None = None
    thread_id: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    # Optional trusted working directory for relative tool paths and shell cwd.
    # It is set only by an internal execution adapter, never from tool
    # arguments, and every consumer still confines the result to allowed roots.
    workdir: str | None = None
    # A pre-verified durable approval id set only by a trusted internal caller
    # (the automation specialist after its own gate). It is never read from
    # model arguments, so it cannot be forged through a tool payload.
    authorization: str | None = None


class ToolResult(BaseRecord):
    call_id: str
    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    risk: RiskLevel = RiskLevel.LOW
    decision: str = "allowed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SurfaceCapabilities(BaseModel):
    """What one surface can actually do with an approval card.

    Everything defaults to incapable. A surface may carry a card only when it
    can present the full literal body, wait for a decision and rewrite the
    result. Send-only, notifier-blind and unverified adapters stay incapable;
    Phase 4/9 opt capable surfaces in explicitly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    can_present_card: bool = False
    can_wait_for_decision: bool = False
    full_text: bool = False
    can_rewrite: bool = False
    notify_only: bool = False
    max_card_chars: int | None = None


INCAPABLE_SURFACE = SurfaceCapabilities()

# The one surface capability manifest, keyed by surface name. Unknown surfaces
# resolve to ``INCAPABLE_SURFACE`` so a new adapter cannot accidentally receive
# a card just by existing.
SURFACE_CAPABILITIES: dict[str, SurfaceCapabilities] = {
    # Interactive card surfaces. Rollout is phased; the manifest records the
    # contract, not an entitlement.
    "tui": SurfaceCapabilities(
        can_present_card=True,
        can_wait_for_decision=True,
        full_text=True,
        # The TUI rewrites a decided card by printing its terminal text in the
        # same transcript; it cannot edit an already-scrolled line.
        can_rewrite=True,
        max_card_chars=8000,
    ),
    "web": SurfaceCapabilities(
        can_present_card=True,
        can_wait_for_decision=True,
        full_text=True,
        can_rewrite=True,
        max_card_chars=20000,
    ),
    "electron": SurfaceCapabilities(
        can_present_card=True,
        can_wait_for_decision=True,
        full_text=True,
        can_rewrite=True,
        max_card_chars=20000,
    ),
    "telegram": SurfaceCapabilities(
        can_present_card=True,
        can_wait_for_decision=True,
        full_text=True,
        can_rewrite=True,
        max_card_chars=4096,
    ),
    # Blocking surfaces: a human may see a notice, but the turn cannot wait
    # mid-flight for a card decision.
    "cli": SurfaceCapabilities(
        can_present_card=False,
        can_wait_for_decision=False,
        full_text=True,
        can_rewrite=False,
        notify_only=True,
    ),
    "api": SurfaceCapabilities(notify_only=True),
    # Explicitly incapable: send-only / notifier-blind adapters and delegated
    # or scheduled work with no human surface at all.
    "a2a": INCAPABLE_SURFACE,
    "whatsapp": INCAPABLE_SURFACE,
    "whatsapp_web": INCAPABLE_SURFACE,
    "agent": INCAPABLE_SURFACE,
    "scheduled": INCAPABLE_SURFACE,
    "chat": INCAPABLE_SURFACE,
}


def surface_capabilities(surface: str | None) -> SurfaceCapabilities:
    """Capabilities for a surface name; unknown surfaces are incapable."""
    key = str(surface or "").strip().lower()
    return SURFACE_CAPABILITIES.get(key, INCAPABLE_SURFACE)


class TurnContext(BaseModel):
    """Trusted, task-local facts about one turn.

    Built at a trusted entry point (chat, API, gateway, agent delegation,
    scheduled execution) and installed in a ``ContextVar`` for the duration of
    the turn. Authority fields (``principal``, ``surface``, ``source``) never
    come from a request body or model output. ``origin``/``target`` describe
    where the turn came from and where a reply belongs; they are routing facts,
    not grants.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal: str = "chat"
    channel: str = "cli"
    # Fresh, trusted identity for this turn. It is minted when the context is
    # built at the entry point and copied onto every ToolCall via tool_fields(),
    # so dedupe can be per turn rather than per conversation/time window.
    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    conversation_id: str | None = None
    thread_id: str | None = None
    target: str | None = None
    surface: str = "chat"
    capabilities: SurfaceCapabilities
    origin: str | None = None
    # Provenance class: chat/api/gateway/agent/operator/scheduled/unattended.
    source: str = "chat"

    @model_validator(mode="before")
    @classmethod
    def _fill_capabilities(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("capabilities") is not None:
            return data
        surface = str(data.get("surface") or "chat")
        capabilities = surface_capabilities(surface)
        return {**data, "capabilities": capabilities.model_dump(mode="json")}

    def tool_fields(self) -> dict[str, Any]:
        """The provenance fields copied onto a :class:`ToolCall`."""
        return {
            "source": self.source,
            "origin": self.origin or self.channel or None,
            "principal": self.principal,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "thread_id": self.thread_id,
            "capabilities": self.capabilities.model_dump(mode="json"),
        }


_CURRENT_TURN_CONTEXT: contextvars.ContextVar[TurnContext | None] = (
    contextvars.ContextVar("leo_turn_context", default=None)
)


def current_turn_context() -> TurnContext | None:
    """The trusted context of the turn running on this task, if any."""
    return _CURRENT_TURN_CONTEXT.get()


def set_turn_context(turn: TurnContext | None) -> contextvars.Token:
    """Install a trusted turn context; the returned token restores the prior one."""
    return _CURRENT_TURN_CONTEXT.set(turn)


def reset_turn_context(token: contextvars.Token) -> None:
    """Restore the context that was active before :func:`set_turn_context`."""
    _CURRENT_TURN_CONTEXT.reset(token)


@contextmanager
def turn_context(turn: TurnContext | None) -> Iterator[TurnContext | None]:
    """Scope a trusted turn context to a block (sync or async callers)."""
    token = set_turn_context(turn)
    try:
        yield turn
    finally:
        reset_turn_context(token)


class Decision(BaseRecord):
    allowed: bool
    reason: str
    risk: RiskLevel
    requires_approval: bool = False
    approval_id: str | None = None
    # True only when the supplied approval was verified for this exact action.
    verified_approval: bool = False


class ChatMessage(BaseRecord):
    role: str  # system | user | assistant | tool
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str | None = None
    images: list[dict[str, Any]] = Field(default_factory=list)


class Usage(BaseRecord):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    estimated: bool = True


class ModelRequest(BaseRecord):
    messages: list[ChatMessage]
    model: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    task_category: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    # Reasoning effort for this request (none|minimal|low|medium|high|xhigh|
    # max|ultra). None means no reasoning parameter is sent to the provider.
    reasoning_effort: str | None = None
    # Ask the provider for a readable reasoning SUMMARY. Raw chain-of-thought
    # is never forwarded: providers that cannot summarize send no text.
    reasoning_show: bool = False
    # Where the effective effort came from: session | global | config | context.
    reasoning_source: str | None = None


class StreamChunk(BaseRecord):
    """One normalized streaming delta.

    ``text`` carries exactly one kind of provider output. Reasoning is only ever
    present when the provider actually sent it; it is never derived from answer
    text, and it must never be appended to message content by consumers.
    """

    kind: Literal["text", "reasoning"] = "text"
    text: str


class ModelResponse(BaseRecord):
    content: str
    model: str
    provider: str
    # Provider-supplied thinking/reasoning, kept separate from `content` so it
    # cannot leak into tool rounds or durable assistant messages. This may be
    # raw chain-of-thought and must never be rendered to a user.
    reasoning: str = ""
    # A readable summary the provider explicitly returned (Anthropic adaptive
    # thinking with display=summarized, or an OpenAI-compatible summary field).
    # Empty when the provider supplied only raw or no reasoning.
    reasoning_summary: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseRecord):
    id: str = Field(default_factory=lambda: new_id("mem"))
    category: MemoryCategory = MemoryCategory.SEMANTIC
    kind: MemoryKind = MemoryKind.FACT
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    confidence: float = 0.7
    tags: list[str] = Field(default_factory=list)
    version: int = 1
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    expires_at: str | None = None
    score: float = 0.0


class TaskRecord(BaseRecord):
    id: str = Field(default_factory=lambda: new_id("task"))
    objective: str
    status: TaskStatus = TaskStatus.CREATED
    plan: list[str] = Field(default_factory=list)
    current_step: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    result: Any = None
    errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    cancellation_requested: bool = False
    owner_agent: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    completed_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Row revision used by the durable engine for compare-and-swap saves.
    version: int = 0


class AgentSpec(BaseRecord):
    id: str
    name: str
    purpose: str
    capabilities: list[str] = Field(default_factory=list)
    tool_permissions: list[str] = Field(default_factory=list)
    model_preferences: list[str] = Field(default_factory=list)
    context_budget: int = 16000
    task_limit: int = 4
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseRecord):
    agent_id: str
    task_id: str
    success: bool
    output: Any = None
    error: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = 0.0


class SkillManifest(BaseRecord):
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "leo-native"
    category: str = "general"
    permissions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    # Availability gates are conditions for a skill to be usable, never grants of
    # authority. They must not be folded into `permissions` (authorization) or
    # `dependencies` (registry dependencies).
    gates: dict[str, Any] = Field(default_factory=dict)
    ecosystem: str | None = None
    source_name: str | None = None
    version_source: str = "declared"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    entrypoint: str | None = None
    source: str = "native"
    documentation: str = ""
    tests: list[str] = Field(default_factory=list)
    # Who produced this manifest: a shipped native skill, LEO's post-turn
    # reviewer, or an explicit user request. The write path is the only place
    # that may set this; ``native`` skills are never agent-authored.
    created_by: Literal["native", "agent", "user"] = "native"
    # Bounded, reviewable provenance for an authored manifest (review id,
    # conversation id, evidence digests). Never an execution grant.
    origin: dict[str, Any] = Field(default_factory=dict)
    # Import provenance is metadata for review, never an execution grant. It is
    # persisted with the manifest so a DB reload cannot silently lose it.
    provenance: dict[str, Any] = Field(default_factory=dict)
    bundle: dict[str, Any] = Field(default_factory=dict)
    # An external manifest is executable only when both flags are set by an
    # explicit, audited approval. Native skills are unaffected.
    provenance_verified: bool = False
    provenance_approval_id: str | None = None


class IntegrationRecord(BaseRecord):
    name: str
    kind: str
    status: str
    capabilities: list[str] = Field(default_factory=list)
    verification: str = "not-verified"
    config_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PermissionManager(Protocol):
    async def authorize(
        self, actor: str, action: str, resource: str | None = None, risk: RiskLevel = RiskLevel.LOW, **context: Any
    ) -> Decision: ...

    async def resolve_approval(self, approval_id: str, approved: bool, resolver: str = "user") -> Decision: ...


class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall) -> ToolResult: ...


class ModelProvider(Protocol):
    name: str

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    async def stream(self, request: ModelRequest):
        ...

    async def stream_events(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """Typed streaming hook for callers that need reasoning deltas.

        The default keeps providers that only implement ``stream()`` valid: it
        forwards each text delta as a ``text`` chunk and never fabricates a
        reasoning chunk.
        """
        async for text in self.stream(request):
            yield StreamChunk(kind="text", text=text)


class TaskEngine(Protocol):
    async def create(self, objective: str, **kwargs: Any) -> TaskRecord: ...

    async def run(self, task_id: str, worker: Any) -> TaskRecord: ...

    async def get(self, task_id: str) -> TaskRecord | None: ...

    async def cancel(self, task_id: str) -> bool: ...
