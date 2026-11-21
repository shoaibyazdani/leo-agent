"""Shared, documented data contracts between LEO modules.

Modules depend on these types and on the public methods documented here; they
must not import one another's internals. This keeps the modular monolith free of
circular dependencies.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex
    return f"{prefix}_{value}" if prefix else value


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    output_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseRecord):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    actor: str = "assistant"
    dry_run: bool = False


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
    # cannot leak into tool rounds or durable assistant messages.
    reasoning: str = ""
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
