"""Observe, classify, propose, and (only with human approval) repair failures.

Design constraints, derived from LEO's existing safety model, but implemented
entirely on LEO primitives:

* Evidence comes from LEO's durable ``EventBus`` history (``tool.failed``,
  ``task.failed``, ...). Observation is idempotent by event id.
* Failure signatures are clustered into bounded incidents with explicit retry
  and lifetime attempt ceilings. Transient classes may be retried with
  exponential backoff; schema/permission failures are never auto-retried.
* Every remediation is inert until a HIGH-risk approval, bound by canonical
  argument digest and resolved by a trusted *non-chat, non-self* human, is
  consumed one-shot. A disabled approval gate does not grant self-repair.
* Before any mutation a rollback snapshot is written. The engine never opens or
  edits source files: mutations, independent verification and rollbacks are
  performed by trusted, caller-supplied ``applier`` / ``verifier`` /
  ``rollbacker`` callables. A canned verification mapping is never accepted;
  the verifier must observe a real before/after diff before "applied" can be
  recorded, and no mutation starts without a rollbacker.
* Every state transition writes an append-only ``audit_log`` row and emits a
  ``selfheal.*`` event on the shared bus with a correlation id.

State lives in the existing ``kv`` table; no new tables, no second database and
no parallel approval or audit mechanism are introduced.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import db
from ..config import Settings
from ..contracts import RiskLevel, new_id, utcnow
from ..events import Event, EventBus
from ..logging import get_logger
from ..security.permissions import PermissionManagerImpl, arguments_digest

_LOG = get_logger("selfheal")

# Failure taxonomy. Only transient classes are eligible for bounded automatic
# retry; the rest require a reviewed remediation and an explicit approval.
KIND_SCHEMA = "schema_misuse"
KIND_PERMISSION = "permission_denial"
KIND_TIMEOUT = "timeout"
KIND_PROVIDER = "provider_error"
KIND_UNKNOWN = "unknown"
RETRYABLE_KINDS = frozenset({KIND_TIMEOUT, KIND_PROVIDER})

WATCHED_TOPICS = (
    "tool.failed",
    "task.failed",
    "agent.failed",
    "model.failed",
    "provider.failed",
)

ACTION_APPLY = "selfheal.apply"
ACTION_ROLLBACK = "selfheal.rollback"
ACTOR = "heal:selfheal"

_PREFIX = "selfheal:"
_SEEN_LIMIT = 1000
# Incident rows are durable kv values, not journals: keep the evidence lists
# bounded so repeated failures cannot grow a single row without limit.
_INCIDENT_EVENT_LIMIT = 100
_INCIDENT_SOURCE_LIMIT = 20
_MAX_ERROR_CHARS = 1000
_MAX_SIGNATURE_CHARS = 240
_MAX_STEPS = 8

# A verifier must return an independently observed post-change state so the
# engine can prove a real diff. These aliases are accepted for that field.
_VERIFICATION_STATE_KEYS = ("after", "observed", "read_back", "state", "current")

_SCHEMA_DECISIONS = frozenset(
    {"invalid_arguments", "invalid_output", "unknown_tool", "validation_failed", "schema_violation"}
)
_PERMISSION_DECISIONS = frozenset(
    {"permission_denied", "denied", "out_of_scope", "pending", "permission_error", "approval_denied"}
)
_PROVIDER_TOKENS = (
    "rate limit", "rate_limit", "overloaded", "overload", "capacity", "context length",
    "context window", "context overflow", "billing", "quota exhausted", "model not found",
    "bad gateway", "service unavailable", "gateway timeout", "connection reset",
    "connection refused", "ssl", "tls", "provider error", "upstream", " 429", " 500",
    " 502", " 503", " 504",
)


class HealingDenied(Exception):
    """A healing operation refused to act (missing evidence/approval/verification)."""


class FailureIncident(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    fingerprint: str
    kind: str
    target: str
    decision: str
    signature: str
    sample_error: str = ""
    occurrences: int = 0
    event_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    first_seen: str = Field(default_factory=utcnow)
    last_seen: str = Field(default_factory=utcnow)
    state: str = "observed"
    attempts: int = 0
    next_retry_at: str | None = None
    proposal_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Remediation(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    incident_id: str
    kind: str
    target: str
    title: str
    rationale: str
    steps: list[str] = Field(default_factory=list)
    resource: str
    risk: RiskLevel = RiskLevel.HIGH
    status: str = "proposed"
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_digest: str = ""
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_hashes: dict[str, str] = Field(default_factory=dict)
    approval_id: str | None = None
    snapshot_id: str | None = None
    attempts: int = 0
    evaluation_proposal_id: str | None = None
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    applied_at: str | None = None
    # "applied" is only ever written after an independent verifier proves a
    # real before/after diff. This field makes that distinction machine-readable
    # for health checks and audits; it is never inferred from a canned mapping.
    verification_status: str = "not_verified"
    verification: dict[str, Any] = Field(default_factory=dict)
    verified_at: str | None = None


class RollbackSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    target: str
    proposal_id: str | None = None
    incident_id: str | None = None
    before: Any = None
    digest: str = ""
    status: str = "active"
    actor: str = ACTOR
    approval_id: str | None = None
    created_at: str = Field(default_factory=utcnow)
    rolled_back_at: str | None = None


def _text(value: Any, limit: int = _MAX_ERROR_CHARS) -> str:
    text = str(value or "").replace("\x00", "")
    text = " ".join(text.split())
    return text[:limit]


def _safe_target(value: Any, fallback: str) -> str:
    candidate = _text(value, 120)
    safe = re.sub(r"[^A-Za-z0-9_.:-]", "_", candidate).strip("_.")
    return safe[:120] or fallback


def _bounded_append(values: list[str], value: str, limit: int) -> list[str]:
    """Append once, keeping only the most recent ``limit`` distinct entries."""
    if value in values:
        return values
    values.append(value)
    if len(values) > limit:
        del values[:-limit]
    return values


def classify_failure(topic: str, payload: Mapping[str, Any]) -> str:
    """Classify a failure event using its decision first, then its error text."""
    decision = _text(payload.get("decision"), 80).lower()
    error = _text(payload.get("error"), _MAX_ERROR_CHARS).lower()
    combined = f"{decision} {error}"
    if decision in _SCHEMA_DECISIONS or any(
        token in combined
        for token in ("invalid arguments", "schema violation", "additional properties",
                      "unexpected keyword", "unsupported argument", "unknown tool",
                      "json decode", "jsondecode", "missing required")
    ):
        return KIND_SCHEMA
    if decision in _PERMISSION_DECISIONS or any(
        token in combined
        for token in ("permission denied", "missing capability", "outside the caller",
                      "approval required", "self-approval", "not authorized")
    ):
        return KIND_PERMISSION
    if decision == "timeout" or "timed out" in combined or " timeout" in combined:
        return KIND_TIMEOUT
    if any(token in combined for token in _PROVIDER_TOKENS):
        return KIND_PROVIDER
    return KIND_UNKNOWN


def _signature(error: str) -> str:
    """Normalize volatile ids/numbers so repeated failures share a signature."""
    value = _text(error, _MAX_ERROR_CHARS).lower()
    value = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<id>", value)
    value = re.sub(r"\b[0-9a-f]{16,}\b", "<hex>", value)
    value = re.sub(r"\d+", "#", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:_MAX_SIGNATURE_CHARS]


def _target_for(topic: str, payload: Mapping[str, Any]) -> str:
    for key in ("tool", "tool_name", "target", "task_id", "agent_id", "provider", "model"):
        if payload.get(key):
            return _safe_target(payload[key], topic.split(".")[0])
    return topic.split(".")[0] or "unknown"


class SelfHealingEngine:
    """Bounded observation -> proposal -> approved apply/rollback pipeline."""

    def __init__(
        self,
        events: EventBus | None = None,
        permissions: PermissionManagerImpl | None = None,
        settings: Settings | None = None,
        *,
        actor: str = ACTOR,
        min_occurrences: int = 3,
        max_attempts: int = 3,
        backoff_base_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        evaluation: Any = None,
        sleep: Callable[[float], Any] | None = None,
        now: Callable[[], datetime] | None = None,
        redact_patterns: list[str] | None = None,
        auto_attach: bool = True,
    ) -> None:
        if min_occurrences < 1 or max_attempts < 1:
            raise ValueError("min_occurrences and max_attempts must be positive")
        if backoff_base_seconds <= 0 or max_backoff_seconds <= 0:
            raise ValueError("backoff delays must be positive")
        self.settings = settings or Settings()
        self.events = events if events is not None else EventBus()
        self.permissions = permissions or PermissionManagerImpl(self.settings, self.events)
        self.evaluation = evaluation
        self.actor = str(actor or ACTOR)
        self.min_occurrences = int(min_occurrences)
        self.max_attempts = int(max_attempts)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self._sleep = sleep or asyncio.sleep
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._attached: dict[str, Callable[[Event], Any]] = {}
        patterns = redact_patterns
        if patterns is None:
            patterns = list(self.settings.security.redact_patterns)
        self._patterns: list[re.Pattern[str]] = []
        for pattern in patterns:
            try:
                self._patterns.append(re.compile(pattern))
            except re.error:  # noqa: PERF203 - one bad operator pattern must not disable redaction
                _LOG.warning("invalid selfheal redaction pattern ignored")
        if auto_attach:
            self.attach()

    def attach(self) -> None:
        """Subscribe observation to the shared failure topics (idempotent)."""
        for topic in WATCHED_TOPICS:
            if topic not in self._attached:
                handler = self.observe
                self.events.subscribe(topic, handler)
                self._attached[topic] = handler

    def detach(self) -> None:
        """Stop live observation; durable history can still be scanned later."""
        for topic, handler in self._attached.items():
            self.events.unsubscribe(topic, handler)
        self._attached.clear()

    # ------------------------------------------------------------------
    # Storage helpers (shared kv table, namespaced keys)
    # ------------------------------------------------------------------
    @staticmethod
    def _key(name: str) -> str:
        return f"{_PREFIX}{name}"

    async def _load(self, name: str, model: type[BaseModel] | None = None) -> Any:
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (self._key(name),))
        if not row:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError):
            return None
        return model.model_validate(value) if model is not None else value

    async def _save(self, name: str, value: Any) -> None:
        await db.execute(
            "INSERT OR REPLACE INTO kv(key,value,updated_at) VALUES(?,?,?)",
            (self._key(name), json.dumps(value, ensure_ascii=False, default=str), utcnow()),
        )

    async def _index(self) -> dict[str, str]:
        return await self._load("index") or {}

    def _redact(self, value: Any) -> str:
        text = _text(value)
        for pattern in self._patterns:
            text = pattern.sub("[REDACTED]", text)
        return text

    def _clean(self, value: Any, depth: int = 0) -> Any:
        """Redact any string before it reaches an event or a durable kv row."""
        if isinstance(value, str):
            return self._redact(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= 4:
            return "[truncated]"
        if isinstance(value, Mapping):
            return {str(k): self._clean(v, depth + 1) for k, v in list(value.items())[:50]}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._clean(v, depth + 1) for v in list(value)[:50]]
        return self._redact(str(value))

    async def _emit(self, topic: str, **payload: Any) -> None:
        clean = self._clean(payload)
        clean.setdefault("actor", self.actor)
        clean.setdefault("ts", utcnow())
        correlation = clean.get("correlation_id") or clean.get("incident_id") or clean.get("proposal_id")
        await self.events.publish(
            Event(
                topic=topic,
                source="selfheal",
                payload=clean,
                correlation_id=str(correlation) if correlation is not None else None,
            )
        )

    async def _audit(
        self,
        action: str,
        resource: str,
        decision: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        risk: RiskLevel = RiskLevel.HIGH,
        correlation_id: str | None = None,
    ) -> None:
        await db.execute(
            "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,correlation_id) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                utcnow(), self.actor, action, _text(resource, 240), risk.value,
                _text(decision, 40), self._redact(reason)[:500],
                json.dumps(self._clean(metadata or {}), ensure_ascii=False, default=str)[:2000],
                correlation_id,
            ),
        )

    # ------------------------------------------------------------------
    # Observation and classification
    # ------------------------------------------------------------------
    async def observe(self, event: Event) -> FailureIncident | None:
        """Ingest one failure event; returns the updated incident (or None)."""
        if getattr(event, "source", "") == "selfheal" or event.topic not in WATCHED_TOPICS:
            return None
        event_id = str(getattr(event, "id", "") or "")
        if not event_id:
            return None
        seen: list[str] = await self._load("seen") or []
        if event_id in seen:
            return None

        payload = dict(event.payload or {})
        kind = classify_failure(event.topic, payload)
        target = _target_for(event.topic, payload)
        decision = _text(payload.get("decision"), 80) or "error"
        error = self._redact(payload.get("error") or payload.get("message") or "")
        signature = _signature(error) or f"{event.topic}:{decision}"
        fingerprint = hashlib.sha256(
            f"{kind}|{target}|{signature}".encode("utf-8")
        ).hexdigest()

        index = await self._index()
        incident_id = index.get(fingerprint)
        incident = await self._load(f"incident:{incident_id}", FailureIncident) if incident_id else None
        if incident is None:
            incident_id = new_id("shi")
            incident = FailureIncident(
                id=incident_id,
                fingerprint=fingerprint,
                kind=kind,
                target=target,
                decision=decision,
                signature=signature,
                sample_error=error[:_MAX_ERROR_CHARS],
                first_seen=utcnow(),
            )
            index[fingerprint] = incident_id
        incident.occurrences += 1
        _bounded_append(incident.event_ids, event_id, _INCIDENT_EVENT_LIMIT)
        if event.source:
            _bounded_append(incident.sources, event.source, _INCIDENT_SOURCE_LIMIT)
        if not incident.sample_error and error:
            incident.sample_error = error[:_MAX_ERROR_CHARS]
        incident.decision = decision
        incident.last_seen = utcnow()
        if incident.state in ("recovered", "observed"):
            incident.state = "observed"

        seen.append(event_id)
        await self._save("seen", seen[-_SEEN_LIMIT:])
        await self._save(f"incident:{incident_id}", incident.model_dump())
        await self._save("index", index)

        await self._emit(
            "selfheal.incident",
            incident_id=incident_id,
            fingerprint=fingerprint,
            kind=kind,
            target=target,
            decision=decision,
            occurrences=incident.occurrences,
            state=incident.state,
            source_event=event_id,
        )
        await self._audit(
            "selfheal.observe", f"selfheal/{kind}/{target}", "recorded",
            f"{kind} failure #{incident.occurrences} from {event.topic}",
            {"incident_id": incident_id, "event_id": event_id, "fingerprint": fingerprint},
            risk=RiskLevel.LOW,
            correlation_id=incident_id,
        )
        if self.evaluation is not None and callable(getattr(self.evaluation, "record", None)):
            try:
                await self.evaluation.record(
                    "failure", 1, {"kind": kind, "target": target, "incident_id": incident_id}
                )
            except Exception as exc:  # noqa: BLE001 - metrics must not break observation
                _LOG.warning("selfheal metric write failed", extra={"error": str(exc)})

        if incident.occurrences >= self.min_occurrences and not incident.proposal_id:
            await self.propose(incident_id)
        return incident

    async def scan(self, limit: int = 100) -> int:
        """Observe durable failures from the shared event history (idempotent)."""
        bounded = max(1, min(int(limit), 1000))
        processed = 0
        for topic in WATCHED_TOPICS:
            page = await self.events.history(topic=topic, limit=bounded)
            for event in reversed(page):
                if await self.observe(event) is not None:
                    processed += 1
        return processed

    # ------------------------------------------------------------------
    # Proposal generation (inert until approved)
    # ------------------------------------------------------------------
    def _remediation_text(self, incident: FailureIncident) -> tuple[str, str, list[str]]:
        target = incident.target
        detail = incident.sample_error[:240]
        quoted = re.findall(r"['\"`]([A-Za-z_][A-Za-z0-9_.-]{1,60})['\"`]", incident.sample_error)
        ignored = {"error", "invalid", "arguments", "schema", "required", "additional", "properties"}
        keys = [item for item in quoted if item not in ignored][:3]
        if incident.kind == KIND_SCHEMA:
            steps = [
                f"Update the input_schema and caller contract for tool {target!r} so every "
                "declared argument is accepted and unknown keys are rejected with a "
                "model-visible tool error (not a hard turn exit).",
                "Add a regression test that replays the captured argument shape and asserts "
                "the tool result error names the invalid key and the valid schema.",
            ]
            if keys:
                steps.append("Rejected key(s) seen in evidence: " + ", ".join(repr(k) for k in keys))
            return (
                f"Fix recurring tool-schema mismatch on {target}",
                f"{incident.occurrences} durable tool failures with signature {detail!r}.",
                steps[:_MAX_STEPS],
            )
        if incident.kind == KIND_PERMISSION:
            return (
                f"Review permission gate for {target}",
                f"{incident.occurrences} durable authorization failures with signature {detail!r}.",
                [
                    "Inspect the actor, action, resource and decision in the cited events; do "
                    "not relax a HIGH-risk gate or self-resolve an approval.",
                    "If a named capability or approval is genuinely missing, have a human add "
                    "the narrowest grant and re-run the failing workflow with evidence.",
                    "Add a test asserting denied calls stay denied and are audited.",
                ],
            )
        if incident.kind == KIND_TIMEOUT:
            return (
                f"Bound and repair timeouts on {target}",
                f"{incident.occurrences} durable timeout failures with signature {detail!r}.",
                [
                    "Add a bounded retry with jittered exponential backoff and a per-incident "
                    "attempt ceiling; never retry a permission or schema failure.",
                    "Review the configured timeout for this tool/route and add a test that "
                    "asserts the ceiling and truthful partial-exit behaviour.",
                ],
            )
        if incident.kind == KIND_PROVIDER:
            return (
                f"Classify and recover provider errors on {target}",
                f"{incident.occurrences} durable provider failures with signature {detail!r}.",
                [
                    "Map the provider error to retry / credential rotation / fallback / "
                    "compress / abort; apply at most one bounded in-place repair per attempt.",
                    "Arm a cooldown with jittered backoff and honour Retry-After within a cap.",
                    "Never store credentials in memory, events, or audit metadata.",
                ],
            )
        return (
            f"Investigate recurring failure on {target}",
            f"{incident.occurrences} durable failures with signature {detail!r}.",
            [
                "Collect additional evidence with a human; no automatic code, config or skill "
                "change is proposed for an unclassified failure.",
            ],
        )

    async def _evidence_hashes(self, event_ids: list[str]) -> dict[str, str]:
        """Pin exact durable event payloads; missing/changed evidence fails closed."""
        hashes: dict[str, str] = {}
        for event_id in event_ids[: max(self.min_occurrences, 10)]:
            row = await db.fetchone("SELECT payload FROM events WHERE id=?", (event_id,))
            if row:
                hashes[event_id] = hashlib.sha256(str(row["payload"]).encode("utf-8")).hexdigest()
        return hashes

    async def _evidence_valid(self, incident: FailureIncident, proposal: Remediation) -> str | None:
        if incident.occurrences < self.min_occurrences:
            return "insufficient occurrence evidence"
        ids = set(proposal.evidence_event_ids)
        if len(ids) < self.min_occurrences:
            return "insufficient distinct event evidence"
        for event_id, expected in proposal.evidence_hashes.items():
            row = await db.fetchone("SELECT payload FROM events WHERE id=?", (event_id,))
            if not row or hashlib.sha256(str(row["payload"]).encode("utf-8")).hexdigest() != expected:
                return f"evidence changed or vanished: {event_id}"
        return None

    async def propose(self, incident_id: str) -> dict[str, Any] | None:
        """Create an inert, approval-bound remediation for a recurring incident."""
        incident = await self._load(f"incident:{incident_id}", FailureIncident)
        if incident is None:
            return None
        if incident.proposal_id:
            existing = await self._load(f"proposal:{incident.proposal_id}", Remediation)
            if existing and existing.status in ("proposed", "awaiting_approval", "applying"):
                return existing.model_dump()
        if not incident.signature or not incident.event_ids:
            await self._deny(incident, "no classified failure evidence")
            return None
        hashes = await self._evidence_hashes(sorted(set(incident.event_ids)))
        if len(hashes) < self.min_occurrences:
            await self._deny(incident, "failure evidence is not durable")
            return None

        title, rationale, steps = self._remediation_text(incident)
        proposal_id = new_id("shp")
        evidence_ids = sorted(set(incident.event_ids))[: max(self.min_occurrences, 10)]
        resource = f"selfheal/{incident.kind}/{_safe_target(incident.target, 'unknown')}"
        arguments = {
            "proposal_id": proposal_id,
            "incident_id": incident.id,
            "kind": incident.kind,
            "target": incident.target,
            "event_ids": evidence_ids,
            "steps": steps,
            "rollback_required": True,
        }
        digest = arguments_digest(arguments) or ""
        if not digest:
            await self._deny(incident, "could not bind approval arguments")
            return None
        proposal = Remediation(
            id=proposal_id,
            incident_id=incident.id,
            kind=incident.kind,
            target=incident.target,
            title=title,
            rationale=rationale,
            steps=steps,
            resource=resource,
            arguments=arguments,
            arguments_digest=digest,
            evidence_event_ids=evidence_ids,
            evidence_hashes={key: hashes[key] for key in evidence_ids},
        )
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        incident.proposal_id = proposal_id
        incident.state = "proposed"
        await self._save(f"incident:{incident.id}", incident.model_dump())

        decision = await self.permissions.authorize(
            self.actor, ACTION_APPLY, resource, RiskLevel.HIGH,
            arguments=arguments, arguments_digest=digest,
            proposal_id=proposal_id, incident_id=incident.id,
            evidence_count=incident.occurrences, title=title,
        )
        proposal.approval_id = decision.approval_id
        # A deployment with the HIGH-risk gate disabled is NOT permission to
        # self-modify: without a resolvable approval id the proposal stays inert.
        if decision.allowed and not decision.requires_approval:
            await self._audit(
                ACTION_APPLY, resource, "denied",
                "high-risk approval gate disabled; refusing to self-apply",
                {"proposal_id": proposal_id}, correlation_id=proposal_id,
            )
            await self._emit(
                "selfheal.denied", incident_id=incident.id, proposal_id=proposal_id,
                reason="high-risk approval gate disabled; refusing to self-apply",
            )
        else:
            proposal.status = "awaiting_approval"
            await self._audit(
                ACTION_APPLY, resource, "pending",
                "remediation proposed; awaiting HIGH-risk approval",
                {"proposal_id": proposal_id, "approval_id": decision.approval_id,
                 "arguments_digest": digest},
                correlation_id=proposal_id,
            )
            await self._emit(
                "selfheal.approval",
                incident_id=incident.id, proposal_id=proposal_id,
                approval_id=decision.approval_id, risk=RiskLevel.HIGH.value,
                arguments_digest=digest, title=title, steps=steps,
            )
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        await self._emit(
            "selfheal.proposal",
            incident_id=incident.id, proposal_id=proposal_id, kind=incident.kind,
            target=incident.target, title=title, rationale=rationale, steps=steps,
            status=proposal.status, approval_id=proposal.approval_id,
        )
        if self.evaluation is not None and callable(getattr(self.evaluation, "propose", None)):
            try:
                inert = await self.evaluation.propose(title, rationale, {
                    "incident_id": incident.id, "proposal_id": proposal_id,
                    "evidence_event_ids": evidence_ids, "kind": incident.kind,
                })
                proposal.evaluation_proposal_id = inert.get("id")
                await self._save(f"proposal:{proposal_id}", proposal.model_dump())
            except Exception as exc:  # noqa: BLE001 - inert queue is best-effort
                _LOG.warning("selfheal evaluation proposal failed", extra={"error": str(exc)})
        return proposal.model_dump()

    async def _deny(self, incident: FailureIncident, reason: str) -> None:
        resource = f"selfheal/{incident.kind}/{_safe_target(incident.target, 'unknown')}"
        await self._audit(ACTION_APPLY, resource, "denied", reason,
                          {"incident_id": incident.id}, correlation_id=incident.id)
        await self._emit("selfheal.denied", incident_id=incident.id, reason=reason)

    # ------------------------------------------------------------------
    # Approval gate helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _trusted_resolver(row: Any) -> str | None:
        """Only a resolved, non-chat human distinct from the requester counts."""
        if not row or row["status"] != "approved" or not row["resolved_at"]:
            return None
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        resolver = str(metadata.get("resolved_by") or "").strip() if isinstance(metadata, Mapping) else ""
        if not resolver or resolver.casefold().startswith("chat:") or resolver.casefold() == ACTOR.casefold():
            return None
        if resolver.casefold() == str(row["actor"] or "").casefold():
            return None
        return resolver

    async def _consume_approval(
        self,
        action: str,
        resource: str,
        arguments: Mapping[str, Any],
        digest: str,
        approval_id: str | None,
    ) -> tuple[str, str | None]:
        """Verify and consume one exactly-bound approval.

        The caller must hold the ambient ``db.transaction()``: the conditional
        update, the permission audit row and the proposal/snapshot writes then
        commit atomically. Opening a nested transaction here would deadlock the
        writer connection.
        """
        failure: str | None = None
        resolver: str | None = None
        if not approval_id:
            return "missing HIGH-risk approval id", None
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if not row or row["status"] != "approved":
            return "approval is not approved", None
        resolver = self._trusted_resolver(row)
        if not resolver:
            return "approval resolver is chat, self or unknown", None
        if row["action"] != action or row["resource"] != resource or row["risk"] != RiskLevel.HIGH.value:
            return "approval action/resource/risk binding mismatch", None
        if row["actor"] != self.actor:
            return "approval actor mismatch", None
        try:
            expected = json.loads(row["metadata"] or "{}").get("arguments_digest")
        except (TypeError, ValueError, AttributeError):
            expected = None
        if not digest or not expected or expected != digest:
            return "approval argument digest mismatch", None
        decision = await self.permissions.authorize(
            self.actor, action, resource, RiskLevel.HIGH,
            approval_id=approval_id, arguments=dict(arguments), arguments_digest=digest,
        )
        if not decision.allowed or decision.approval_id != approval_id:
            return "approval binding expired or invalid", None
        cursor = await db.execute(
            "UPDATE approvals SET status='consumed' WHERE id=? AND status='approved'",
            (approval_id,),
        )
        if cursor.rowcount != 1:
            return "approval already consumed", resolver
        return "", resolver

    # ------------------------------------------------------------------
    # Applying a remediation (always through injected trusted callables)
    # ------------------------------------------------------------------
    @staticmethod
    def _verifier_arity(verifier: Callable[..., Any]) -> tuple[int, bool]:
        """Count positional parameters so hosts can use a shape they control."""
        try:
            parameters = inspect.signature(verifier).parameters.values()
        except (TypeError, ValueError):
            return 3, False
        count = 0
        variadic = False
        for parameter in parameters:
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                count += 1
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                variadic = True
        return count, variadic

    def _invoke_verifier(
        self,
        verifier: Callable[..., Any],
        proposal_details: Mapping[str, Any],
        before_state: Any,
        applied: Any,
        context: Mapping[str, Any],
    ) -> Any:
        """Call the host verifier with its supported argument shape.

        Canonical shape is ``verifier(proposal, before, applied)``. Two-argument
        legacy verifiers receive ``(proposal, applied)``; single-argument
        verifiers receive the observation ``context``. A verifier taking four
        (or variadic) arguments also gets that context, including the snapshot
        digests it can use for an independent read-back.
        """
        count, variadic = self._verifier_arity(verifier)
        if variadic or count >= 4:
            args: tuple[Any, ...] = (proposal_details, before_state, applied, context)
        elif count == 3:
            args = (proposal_details, before_state, applied)
        elif count == 2:
            args = (proposal_details, applied)
        elif count == 1:
            args = (context,)
        else:
            args = ()
        return verifier(*args)

    @staticmethod
    def _verification_readback(proof: Any) -> Any:
        """Require a structural proof with an independently observed state."""
        if not isinstance(proof, Mapping):
            raise HealingDenied("verifier must return a mapping with tests_passed, check and after")
        if proof.get("tests_passed") is not True:
            raise HealingDenied("verifier did not report tests_passed=True")
        if not str(proof.get("check") or "").strip():
            raise HealingDenied("verifier did not report a check")
        for key in _VERIFICATION_STATE_KEYS:
            if key in proof:
                return proof[key]
        raise HealingDenied("verifier did not return an independent read-back ('after')")

    async def apply(
        self,
        proposal_id: str,
        *,
        applier: Callable[[Mapping[str, Any]], Any],
        before: Any,
        verifier: Callable[..., Any] | None = None,
        rollbacker: Callable[[Mapping[str, Any]], Any] | None = None,
        approval_id: str | None = None,
        after: Any = None,
        verification: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply a proposed change after verifying evidence and HIGH-risk approval.

        ``applier``, ``verifier`` and ``rollbacker`` are trusted host callbacks:
        this engine never opens, writes or executes source files, config or
        skill code. ``before`` is the full pre-change state used to build the
        rollback snapshot.

        A canned ``verification`` mapping is never accepted: the injected
        ``verifier`` must independently re-read the target state and return a
        mapping with ``tests_passed is True``, a non-empty ``check`` and the
        observed post-change state under ``after``. The engine digests that
        read-back and refuses to claim an apply unless it actually differs from
        the before-state. ``rollbacker`` must be supplied before any mutation;
        if the apply or verification fails, the mutation is reverted and the
        snapshot is marked rolled_back/failed, never silently left active.

        Only status ``applied`` (with ``verification_status == "verified"``)
        counts as a real change in health checks, events and the audit log.
        Proposals stay ``proposed``/``awaiting_approval`` until then.
        """
        result: dict[str, Any] = {
            "status": "denied", "proposal_id": proposal_id, "reason": "",
            "approval_id": approval_id, "snapshot_id": None,
            "snapshot_status": None, "verification": None,
            "verification_status": "not_verified",
        }
        proposal = await self._load(f"proposal:{proposal_id}", Remediation)
        if proposal is None:
            result["reason"] = "unknown proposal"
            return result
        result["approval_id"] = approval_id or proposal.approval_id
        if proposal.status != "awaiting_approval":
            result["reason"] = f"proposal is {proposal.status}, not awaiting approval"
            return result
        if not callable(applier):
            result["reason"] = "no trusted applier supplied"
            return result
        if before is None:
            result["reason"] = "rollback snapshot requires before-state"
            return result
        if verification is not None:
            result["reason"] = "canned verification mapping is not accepted; supply a verifier callable"
            return result
        if not callable(verifier):
            result["reason"] = "independent verifier callable is required before applying"
            return result
        if not callable(rollbacker):
            result["reason"] = "trusted rollbacker callable is required before any mutation"
            return result
        if proposal.attempts >= self.max_attempts:
            result["reason"] = "remediation attempt ceiling reached"
            await self._deny_for_proposal(proposal, result["reason"])
            return result
        incident = await self._load(f"incident:{proposal.incident_id}", FailureIncident)
        if incident is None or (issue := await self._evidence_valid(incident, proposal)):
            result["reason"] = issue or "incident evidence missing"
            await self._deny_for_proposal(proposal, result["reason"])
            return result

        # Bind the human-visible fields to the digest-signed arguments too:
        # changing display steps after approval is approval laundering even if
        # the signed argument blob itself was left untouched.
        expected_arguments = {
            "proposal_id": proposal.id,
            "incident_id": proposal.incident_id,
            "kind": proposal.kind,
            "target": proposal.target,
            "event_ids": proposal.evidence_event_ids,
            "steps": proposal.steps,
            "rollback_required": True,
        }
        digest = arguments_digest(proposal.arguments) or ""
        if proposal.arguments != expected_arguments or digest != proposal.arguments_digest:
            result["reason"] = "proposal arguments changed since approval request"
            await self._deny_for_proposal(proposal, result["reason"])
            return result

        failure = ""
        resolver: str | None = None
        snapshot: RollbackSnapshot | None = None
        supplied = approval_id or proposal.approval_id
        before_clean = self._clean(before)
        async with db.transaction():
            fresh = await self._load(f"proposal:{proposal_id}", Remediation)
            if fresh is None or fresh.status != "awaiting_approval":
                failure = "proposal is no longer awaiting approval"
            elif fresh.arguments_digest != digest:
                failure = "proposal changed during apply"
            else:
                failure, resolver = await self._consume_approval(
                    ACTION_APPLY, fresh.resource, fresh.arguments, digest, supplied
                )
            if not failure:
                snapshot_id = new_id("shs")
                snapshot = RollbackSnapshot(
                    id=snapshot_id,
                    target=fresh.target,
                    proposal_id=fresh.id,
                    incident_id=fresh.incident_id,
                    before=before_clean,
                    digest=arguments_digest(before_clean) or "",
                )
                fresh.status = "applying"
                fresh.attempts += 1
                fresh.approval_id = supplied
                fresh.snapshot_id = snapshot_id
                fresh.updated_at = utcnow()
                await self._save(f"proposal:{fresh.id}", fresh.model_dump())
                await self._save(f"snapshot:{snapshot_id}", snapshot.model_dump())
                proposal = fresh
        if failure:
            result["reason"] = failure
            await self._audit(ACTION_APPLY, proposal.resource, "denied", failure,
                              {"proposal_id": proposal_id, "approval_id": supplied},
                              correlation_id=proposal_id)
            await self._emit("selfheal.denied", incident_id=proposal.incident_id,
                             proposal_id=proposal_id, approval_id=supplied, reason=failure)
            return result

        result["snapshot_id"] = snapshot.id  # type: ignore[union-attr]
        result["snapshot_status"] = snapshot.status  # type: ignore[union-attr]
        context: dict[str, Any] = {
            "phase": "after",
            "proposal": proposal.model_dump(),
            "proposal_id": proposal.id,
            "incident_id": proposal.incident_id,
            "snapshot_id": snapshot.id,
            "before": before_clean,
            "before_digest": snapshot.digest,  # type: ignore[union-attr]
        }
        await self._emit(
            "selfheal.apply", incident_id=proposal.incident_id, proposal_id=proposal_id,
            approval_id=supplied, resolver=resolver, snapshot_id=snapshot.id,  # type: ignore[union-attr]
            status="started", arguments_digest=digest,
            verification_status="not_verified",
        )
        after_digest = ""
        try:
            applied = applier(proposal.model_dump())
            if inspect.isawaitable(applied):
                applied = await applied
            context["applied"] = applied
            proof = self._invoke_verifier(
                verifier, proposal.model_dump(), before_clean, applied, context
            )
            if inspect.isawaitable(proof):
                proof = await proof
            observed = self._verification_readback(proof)
            observed_clean = self._clean(observed)
            after_digest = arguments_digest(observed_clean) or ""
            # A verifier that echoes the before-state, or an applier that did
            # nothing, cannot produce an "applied" claim.
            if not after_digest or after_digest == snapshot.digest:  # type: ignore[union-attr]
                raise HealingDenied(
                    "verification read-back is unchanged from before-state; no real change proven"
                )
            record = dict(self._clean(dict(proof)))
            record.update({
                "after": observed_clean,
                "before_digest": snapshot.digest,  # type: ignore[union-attr]
                "after_digest": after_digest,
                "changed": True,
            })
            proposal.status = "applied"
            proposal.verification_status = "verified"
            proposal.verification = record
            proposal.verified_at = utcnow()
            proposal.applied_at = utcnow()
            proposal.updated_at = utcnow()
            result.update(
                status="applied", verification=record,
                verification_status="verified", reason="applied",
            )
            incident.state = "remediated"
            incident.attempts = proposal.attempts
            incident.next_retry_at = None
            await self._save(f"incident:{incident.id}", incident.model_dump())
        except Exception as exc:  # noqa: BLE001 - every failed apply must be reverted and audited
            reason = self._redact(f"{type(exc).__name__}: {exc}")[:500]
            rolled = snapshot is not None and await self._revert(snapshot, rollbacker, reason)
            result["snapshot_status"] = snapshot.status if snapshot is not None else None
            proposal.status = "rolled_back" if rolled else "failed"
            proposal.verification_status = "not_verified"
            proposal.updated_at = utcnow()
            result.update(status=proposal.status, reason=reason, error=reason)
            incident.state = "proposed" if rolled else "failed"
            await self._save(f"incident:{incident.id}", incident.model_dump())
            await self._emit(
                "selfheal.apply", incident_id=proposal.incident_id, proposal_id=proposal_id,
                approval_id=supplied, status="failed", reason=reason,
                verification_status="not_verified", rolled_back=bool(rolled),
                snapshot_status=result["snapshot_status"],
            )
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        if result["status"] == "applied":
            await self._emit(
                "selfheal.apply", incident_id=proposal.incident_id, proposal_id=proposal_id,
                approval_id=supplied, resolver=resolver, snapshot_id=result["snapshot_id"],
                status="applied", verification=result["verification"],
                verification_status="verified", verified=True,
                before_digest=snapshot.digest,  # type: ignore[union-attr]
                after_digest=after_digest,
            )
        # Audit decision names the evidence level explicitly: "proposed" states
        # are never counted as verified applications by health or reviewers.
        await self._audit(
            ACTION_APPLY, proposal.resource,
            "verified_applied" if result["verification_status"] == "verified" else result["status"],
            result["reason"],
            {"proposal_id": proposal_id, "approval_id": supplied, "resolver": resolver,
             "snapshot_id": result["snapshot_id"], "snapshot_status": result["snapshot_status"],
             "arguments_digest": digest, "verification_status": result["verification_status"],
             "status_before_verification": "proposed",
             "claimed_after": self._clean(after) if after is not None else None},
            correlation_id=proposal_id,
        )
        return result

    async def _deny_for_proposal(self, proposal: Remediation, reason: str) -> None:
        await self._audit(ACTION_APPLY, proposal.resource, "denied", reason,
                          {"proposal_id": proposal.id}, correlation_id=proposal.id)
        await self._emit("selfheal.denied", incident_id=proposal.incident_id,
                         proposal_id=proposal.id, reason=reason)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------
    async def _revert(
        self, snapshot: RollbackSnapshot, rollbacker: Callable[[Mapping[str, Any]], Any] | None,
        reason: str,
    ) -> bool:
        if not callable(rollbacker):
            # A snapshot that cannot be reverted must never remain "active":
            # a caller could otherwise claim an unverified mutation is safe.
            snapshot.status = "unavailable"
            await self._save(f"snapshot:{snapshot.id}", snapshot.model_dump())
            await self._emit("selfheal.rollback", snapshot_id=snapshot.id,
                             proposal_id=snapshot.proposal_id, status="unavailable", reason=reason)
            return False
        try:
            outcome = rollbacker(snapshot.model_dump())
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if outcome is False or (isinstance(outcome, Mapping) and outcome.get("success") is False):
                raise HealingDenied("rollbacker reported failure")
            snapshot.status = "rolled_back"
            snapshot.rolled_back_at = utcnow()
            await self._save(f"snapshot:{snapshot.id}", snapshot.model_dump())
            await self._emit("selfheal.rollback", snapshot_id=snapshot.id,
                             proposal_id=snapshot.proposal_id, status="rolled_back", reason=reason)
            return True
        except Exception as exc:  # noqa: BLE001 - a failed rollback is a loud audit event
            detail = self._redact(f"{type(exc).__name__}: {exc}")[:500]
            snapshot.status = "failed"
            await self._save(f"snapshot:{snapshot.id}", snapshot.model_dump())
            await self._emit("selfheal.rollback", snapshot_id=snapshot.id,
                             proposal_id=snapshot.proposal_id, status="failed", reason=detail)
            return False

    async def request_rollback(self, snapshot_id: str) -> dict[str, Any]:
        """Ask a human for the HIGH-risk approval needed by an explicit rollback."""
        snapshot = await self._load(f"snapshot:{snapshot_id}", RollbackSnapshot)
        if snapshot is None:
            return {"status": "denied", "reason": "unknown snapshot"}
        if snapshot.status != "active":
            return {"status": "denied", "reason": f"snapshot is {snapshot.status}"}
        arguments = {
            "snapshot_id": snapshot.id, "target": snapshot.target,
            "proposal_id": snapshot.proposal_id, "before_digest": snapshot.digest,
        }
        digest = arguments_digest(arguments) or ""
        resource = f"selfheal/rollback/{_safe_target(snapshot.target, 'unknown')}"
        decision = await self.permissions.authorize(
            self.actor, ACTION_ROLLBACK, resource, RiskLevel.HIGH,
            arguments=arguments, arguments_digest=digest,
            snapshot_id=snapshot.id, approval_required=True,
        )
        snapshot.approval_id = decision.approval_id
        await self._save(f"snapshot:{snapshot.id}", snapshot.model_dump())
        await self._emit("selfheal.approval", snapshot_id=snapshot.id,
                         proposal_id=snapshot.proposal_id, approval_id=decision.approval_id,
                         action=ACTION_ROLLBACK, arguments_digest=digest)
        return {"status": "pending", "snapshot_id": snapshot.id,
                "approval_id": decision.approval_id, "arguments_digest": digest,
                "resource": resource}

    async def rollback(
        self, snapshot_id: str, *, rollbacker: Callable[[Mapping[str, Any]], Any],
        approval_id: str | None = None, reason: str = "",
    ) -> dict[str, Any]:
        """Explicitly restore a snapshot; requires its own HIGH-risk approval."""
        snapshot = await self._load(f"snapshot:{snapshot_id}", RollbackSnapshot)
        if snapshot is None or snapshot.status != "active":
            return {"status": "denied", "reason": "snapshot missing or not active"}
        arguments = {
            "snapshot_id": snapshot.id, "target": snapshot.target,
            "proposal_id": snapshot.proposal_id, "before_digest": snapshot.digest,
        }
        digest = arguments_digest(arguments) or ""
        resource = f"selfheal/rollback/{_safe_target(snapshot.target, 'unknown')}"
        supplied = approval_id or snapshot.approval_id
        async with db.transaction():
            failure, resolver = await self._consume_approval(
                ACTION_ROLLBACK, resource, arguments, digest, supplied
            )
            if not failure:
                snapshot.status = "rolling_back"
                await self._save(f"snapshot:{snapshot.id}", snapshot.model_dump())
        if failure:
            await self._audit(ACTION_ROLLBACK, resource, "denied", failure,
                              {"snapshot_id": snapshot.id, "approval_id": supplied},
                              correlation_id=snapshot.id)
            await self._emit("selfheal.denied", snapshot_id=snapshot.id,
                             approval_id=supplied, reason=failure)
            return {"status": "denied", "reason": failure}
        rolled = await self._revert(snapshot, rollbacker, reason or "operator rollback")
        await self._audit(
            ACTION_ROLLBACK, resource, "rolled_back" if rolled else "failed",
            reason or "operator rollback",
            {"snapshot_id": snapshot.id, "approval_id": supplied, "resolver": resolver,
             "arguments_digest": digest}, correlation_id=snapshot.id,
        )
        return {"status": "rolled_back" if rolled else "failed",
                "snapshot_id": snapshot.id, "approval_id": supplied, "resolver": resolver}

    # ------------------------------------------------------------------
    # Bounded retry with exponential backoff
    # ------------------------------------------------------------------
    async def retry(
        self, incident_id: str, operation: Callable[[int], Any], *, reason: str = ""
    ) -> dict[str, Any]:
        """Retry a transient failure with a bounded attempt count and backoff."""
        incident = await self._load(f"incident:{incident_id}", FailureIncident)
        if incident is None:
            return {"status": "denied", "reason": "unknown incident"}
        if incident.kind not in RETRYABLE_KINDS:
            return {"status": "blocked", "reason": f"no automatic retry for {incident.kind}"}
        if incident.state in ("remediated", "exhausted", "recovered"):
            return {"status": "blocked", "reason": f"incident is {incident.state}"}
        if incident.attempts >= self.max_attempts:
            incident.state = "exhausted"
            await self._save(f"incident:{incident.id}", incident.model_dump())
            await self._emit("selfheal.exhausted", incident_id=incident.id,
                             reason="attempt ceiling reached")
            await self.propose(incident.id)
            return {"status": "exhausted", "reason": "attempt ceiling reached"}
        if incident.next_retry_at:
            try:
                scheduled = datetime.fromisoformat(incident.next_retry_at)
                if scheduled.tzinfo is None:
                    scheduled = scheduled.replace(tzinfo=timezone.utc)
                if self._now() < scheduled:
                    return {"status": "waiting", "next_retry_at": incident.next_retry_at}
            except (TypeError, ValueError):
                incident.next_retry_at = None

        attempt = incident.attempts + 1
        error = ""
        try:
            outcome = operation(attempt)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if outcome is False or (isinstance(outcome, Mapping) and outcome.get("success") is False):
                error = "operation reported failure"
        except Exception as exc:  # noqa: BLE001 - captured as bounded retry evidence
            error = self._redact(f"{type(exc).__name__}: {exc}")[:500]
        if not error:
            incident.state = "recovered"
            incident.attempts = attempt
            incident.next_retry_at = None
            await self._save(f"incident:{incident.id}", incident.model_dump())
            await self._emit("selfheal.recovered", incident_id=incident.id,
                             kind=incident.kind, target=incident.target, attempt=attempt)
            await self._audit("selfheal.retry", f"selfheal/{incident.kind}/{incident.target}",
                              "recovered", "bounded retry succeeded",
                              {"incident_id": incident.id, "attempt": attempt},
                              risk=RiskLevel.LOW, correlation_id=incident.id)
            return {"status": "recovered", "attempt": attempt}

        incident.attempts = attempt
        delay = min(self.backoff_base_seconds * (2 ** (attempt - 1)), self.max_backoff_seconds)
        incident.next_retry_at = (self._now() + timedelta(seconds=delay)).isoformat()
        exhausted = attempt >= self.max_attempts
        incident.state = "exhausted" if exhausted else "retrying"
        await self._save(f"incident:{incident.id}", incident.model_dump())
        await self._emit(
            "selfheal.retry", incident_id=incident.id, kind=incident.kind,
            target=incident.target, attempt=attempt, max_attempts=self.max_attempts,
            backoff_seconds=delay, reason=error or reason,
        )
        await self._audit("selfheal.retry", f"selfheal/{incident.kind}/{incident.target}",
                          "failed", error or reason,
                          {"incident_id": incident.id, "attempt": attempt,
                           "backoff_seconds": delay},
                          risk=RiskLevel.LOW, correlation_id=incident.id)
        if exhausted:
            await self._emit("selfheal.exhausted", incident_id=incident.id,
                             reason="attempt ceiling reached", error=error)
            await self.propose(incident.id)
        return {"status": "exhausted" if exhausted else "retrying",
                "attempt": attempt, "backoff_seconds": delay,
                "next_retry_at": incident.next_retry_at, "reason": error}

    async def backoff(self, seconds: float) -> None:
        """Use the configured sleep hook so tests never wait in real time."""
        await self._sleep(max(0.0, float(seconds)))

    # ------------------------------------------------------------------
    # Operator inspection
    # ------------------------------------------------------------------
    async def incidents(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT value FROM kv WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (self._key("incident:%"), max(1, min(int(limit), 1000))),
        )
        return [json.loads(row["value"]) for row in rows]

    async def proposals(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT value FROM kv WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (self._key("proposal:%"), max(1, min(int(limit), 1000))),
        )
        return [json.loads(row["value"]) for row in rows]

    async def snapshots(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT value FROM kv WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (self._key("snapshot:%"), max(1, min(int(limit), 1000))),
        )
        return [json.loads(row["value"]) for row in rows]

    async def audit_trail(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT * FROM audit_log WHERE action LIKE ? ORDER BY id DESC LIMIT ?",
            ("selfheal.%", max(1, min(int(limit), 1000))),
        )
        return [dict(row) for row in rows]

    async def remediation_status(self, limit: int = 100) -> dict[str, Any]:
        """Health counters that never count a proposal as a verified apply."""
        proposals = await self.proposals(limit=limit)
        return {
            "proposed": sum(
                1 for item in proposals
                if item.get("status") in ("proposed", "awaiting_approval")
            ),
            "applying": sum(1 for item in proposals if item.get("status") == "applying"),
            "verified_applied": sum(
                1 for item in proposals
                if item.get("status") == "applied"
                and item.get("verification_status") == "verified"
            ),
            "unverified_applied": sum(
                1 for item in proposals
                if item.get("status") == "applied"
                and item.get("verification_status") != "verified"
            ),
            "rolled_back": sum(1 for item in proposals if item.get("status") == "rolled_back"),
            "failed": sum(1 for item in proposals if item.get("status") == "failed"),
            "denied": sum(1 for item in proposals if item.get("status") == "denied"),
        }

    def governance(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "watched_topics": list(WATCHED_TOPICS),
            "min_occurrences": self.min_occurrences,
            "max_attempts": self.max_attempts,
            "retryable_kinds": sorted(RETRYABLE_KINDS),
            "backoff_base_seconds": self.backoff_base_seconds,
            "max_backoff_seconds": self.max_backoff_seconds,
            "approval_risk": RiskLevel.HIGH.value,
            "independent_verification_required": True,
            "rollbacker_required_before_mutation": True,
            "self_edits_source": False,
        }


__all__ = [
    "ACTION_APPLY", "ACTION_ROLLBACK", "ACTOR", "RETRYABLE_KINDS", "WATCHED_TOPICS",
    "FailureIncident", "HealingDenied", "Remediation", "RollbackSnapshot",
    "SelfHealingEngine", "classify_failure",
]
