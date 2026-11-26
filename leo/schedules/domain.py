"""Durable schedule domain vocabulary and canonical digest helpers.

One domain type set is shared by proposals, persisted rows, approval cards and
views. Canonical JSON is sorted/compact so a digest binds semantic content, not
dict insertion order or object identity.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ScheduleKind = Literal["cron", "interval", "once"]
DeliveryKind = Literal["external", "local", "bot_chat", "room", "nowhere"]
WrapResponse = Literal["on", "off", "inherit"]


def new_schedule_id(prefix: str = "sch") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_safe(value: Any) -> Any:
    """Normalize pydantic models/containers into JSON-native values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Stable canonical JSON: sorted keys, compact separators, UTF-8 text."""
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_digest(value: Any) -> str:
    """SHA-256 over :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class CanonicalModel(BaseModel):
    """Base DTO that rejects unknown fields and can digest its own content."""

    model_config = ConfigDict(extra="forbid")

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_payload())

    def content_digest(self) -> str:
        return sha256_digest(self.canonical_payload())


class ScheduleExpression(CanonicalModel):
    """A canonical schedule definition.

    ``kind`` plus either ``expression`` (cron/interval) or ``run_at`` (ISO
    instant for ``once``) is the semantic identity used for equality and edits.
    """

    kind: ScheduleKind
    expression: str | None = None
    run_at: str | None = None
    timezone: str | None = None

    @model_validator(mode="after")
    def _require_definition(self) -> "ScheduleExpression":
        if self.kind == "once":
            if not self.run_at:
                raise ValueError("a once schedule requires run_at")
        elif not self.expression:
            raise ValueError(f"a {self.kind} schedule requires an expression")
        return self


class DeliveryTarget(CanonicalModel):
    """One resolved delivery destination.

    Identity is ``(platform.lower(), chat_id, thread_id)``; label/copy fields do
    not participate so two spellings of the same target deduplicate.
    """

    kind: DeliveryKind = "nowhere"
    platform: str = "local"
    chat_id: str | None = None
    thread_id: str | None = None
    profile: str | None = None
    room_id: str | None = None
    label: str | None = None

    @model_validator(mode="after")
    def _normalize_platform(self) -> "DeliveryTarget":
        # Platform identity is case-insensitive; normalize it in the DTO so a
        # spelling difference cannot create two targets (or two digests).
        self.platform = self.platform.strip().lower()
        return self

    def identity(self) -> tuple[str, str | None, str | None]:
        return (self.platform, self.chat_id, self.thread_id)


class ExecutionSpec(CanonicalModel):
    """Runtime execution fields. Every field here needs a runtime consumer.

    ``provider``/``model``/``base_url``/``grace_minutes`` are persisted so
    operator/programmatic callers can use them, but later phases must never
    expose them to the model surface.
    """

    script: str | None = None
    script_args: list[str] = Field(default_factory=list)
    no_agent: bool = False
    monitor: str | None = None
    context_from: list[str] = Field(default_factory=list)
    continuity: bool = False
    # None = inherit the agent default; [] = an explicit zero-tool allowlist.
    enabled_toolsets: list[str] | None = None
    skills: list[str] = Field(default_factory=list)
    wrap_response: WrapResponse | None = None
    workdir: str | None = None
    attach_to_session: bool = False
    pinned: bool = False
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    grace_minutes: int | None = Field(default=None, ge=0, le=1440)

    @model_validator(mode="after")
    def _script_only_requirements(self) -> "ExecutionSpec":
        if self.no_agent and not self.script:
            raise ValueError("no_agent requires a script")
        if self.monitor and self.no_agent:
            raise ValueError("monitor is incompatible with no_agent")
        if self.continuity and "self" not in self.context_from:
            self.context_from = [*self.context_from, "self"]
        return self


class ScheduleSpec(CanonicalModel):
    """A complete validated schedule definition, before it is persisted."""

    name: str
    expression: ScheduleExpression
    prompt: str = ""
    repeat_limit: int | None = Field(default=None, ge=1)
    delivery: list[DeliveryTarget] = Field(default_factory=list)
    failure_delivery: list[DeliveryTarget] = Field(default_factory=list)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    paused: bool = False
    paused_reason: str | None = None
    catch_up: bool | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    def delivery_identities(self) -> list[tuple[str, str | None, str | None]]:
        return [target.identity() for target in self.delivery]

    def digest(self) -> str:
        """Digest of the complete validated spec (the consent binding)."""
        return self.content_digest()


class ScheduleProposal(CanonicalModel):
    """A durable, not-yet-committed create/edit request."""

    id: str = Field(default_factory=lambda: new_schedule_id("prop"))
    active_name: str | None = None
    principal: str
    operation: str
    request_key: str | None = None
    status: str = "pending"
    digest: str = ""
    spec: ScheduleSpec
    origin: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None
    config_revision: int | None = None
    schedule_id: str | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ScheduleView(CanonicalModel):
    """The durable row projection shared by tools, API and UI."""

    id: str
    name: str
    kind: str
    schedule_type: str | None = None
    expression: str
    run_at: str | None = None
    timezone: str | None = None
    repeat_limit: int | None = None
    claimed_count: int = 0
    completed_count: int = 0
    config_revision: int = 1
    grace_minutes: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    created_by: str | None = None
    origin: str | None = None
    consent_digest: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    next_run_at: str | None = None
    last_run_at: str | None = None
    created_at: str
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "ScheduleView":
        data = dict(row)
        payload = data.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload or "{}")
            except (TypeError, ValueError):
                payload = {}
        config = data.get("config")
        if isinstance(config, str):
            try:
                config = json.loads(config or "{}")
            except (TypeError, ValueError):
                config = {}
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            kind=str(data["kind"]),
            schedule_type=data.get("schedule_type"),
            expression=str(data["expression"]),
            run_at=data.get("run_at"),
            timezone=data.get("timezone"),
            repeat_limit=data.get("repeat_limit"),
            claimed_count=int(data.get("claimed_count") or 0),
            completed_count=int(data.get("completed_count") or 0),
            config_revision=int(data.get("config_revision") or 1),
            grace_minutes=data.get("grace_minutes"),
            config=config if isinstance(config, dict) else {},
            status=str(data.get("status") or "active"),
            created_by=data.get("created_by"),
            origin=data.get("origin"),
            consent_digest=data.get("consent_digest"),
            payload=payload if isinstance(payload, dict) else {},
            enabled=bool(data.get("enabled", 1)),
            next_run_at=data.get("next_run_at"),
            last_run_at=data.get("last_run_at"),
            created_at=str(data["created_at"]),
            updated_at=data.get("updated_at"),
        )


__all__ = [
    "CanonicalModel",
    "DeliveryTarget",
    "ExecutionSpec",
    "ScheduleExpression",
    "ScheduleKind",
    "ScheduleProposal",
    "ScheduleSpec",
    "ScheduleView",
    "canonical_json",
    "sha256_digest",
]
