"""Derive bounded, evidence-backed memory notes from LEO's own history.

The engine is deliberately *not* a model-driven self-modifier. It reads the
durable transcripts and run records LEO already keeps, applies deterministic
extraction rules, and produces inert proposals. Nothing is written to
``MemoryStore`` until either:

* a HIGH-risk approval was resolved by a trusted non-chat human and consumed
  one-shot with the exact proposal arguments bound by SHA-256, or
* an operator explicitly enabled an :class:`AutoNotePolicy` that names source
  kinds **and** set its separate ``allow_unattended_writes`` flag. Trusting a
  source kind alone never grants an approval-free write.

Safety properties enforced here (not merely documented):

* User text and tool/run metadata are untrusted data. Secret-shaped content,
  prompt-injection shapes, bidi/control characters, raw tool output, tool
  arguments (except the boolean ``dry_run`` marker) and model output are never
  persisted as a note or as provenance.
* Every note carries evidence references (message / tool-call / run ids plus
  content hashes), a category, a bounded confidence and its acquisition
  policy. The raw text behind a hash is never copied into the note.
* Dedupe happens against existing memory and existing proposals both when a
  proposal is created and again immediately before it is written.
* Lifecycle transitions are propose -> accept / reject, with supersede creating
  a new, always-approved proposal. No transition deletes memory.
* A queue of drafts can be coalesced by :meth:`LearningEngine.propose_batch`
  into ONE proposal and ONE argument-digested HIGH-risk approval instead of one
  approval per draft. Only a small number of proposals may be pending at once;
  further drafts are rejected with a truthful reason, and stale pendings expire
  to ``expired`` as a state change only (rows are never deleted).
* Consolidation is a two-phase, approval-bound hook over the existing
  ``Consolidator``; the exact cluster ids are revalidated before it runs.

State lives in the shared ``kv`` table under ``learning:`` keys; audit rows go
to the shared ``audit_log`` and events to the shared ``EventBus``. No new
tables, no second database, no parallel approval mechanism and no changes to
the memory modules are introduced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import db
from ..config import Settings
from ..contracts import MemoryCategory, MemoryKind, MemoryRecord, RiskLevel, new_id, utcnow
from ..events import Event, EventBus
from ..logging import get_logger
from ..memory import Consolidator, MemoryStore, content_similarity, normalize_content
from ..security.permissions import PermissionManagerImpl, arguments_digest

_LOG = get_logger("learning")

ACTION_NOTE = "learning.note"
ACTION_CONSOLIDATE = "learning.consolidate"
ACTOR = "learning:engine"

_PREFIX = "learning:"
_SEEN_LIMIT = 1000
MAX_PROPOSAL_ROWS = 200  # bounded retention of durable ``kv`` proposal rows
MAX_PLAN_ROWS = 50
_MAX_CANDIDATES = 25
_MAX_MESSAGES = 200
_MAX_TOOLS = 200
_MAX_RUNS = 200
MAX_EVIDENCE = 8
MAX_SUPERSEDES = 10
MAX_CLUSTER_MEMBERS = 50
# At most this many writable proposals may be outstanding at once. The cap is
# on proposals (not just approval rows) so an auto/unattended note cannot queue
# an unbounded backlog either. Excess drafts are rejected truthfully, never
# disguised as "awaiting approval" without a matching approval row.
MAX_PENDING_PROPOSALS = 3
# The memory-side dedupe index also records every hash in an accepted batch,
# so it needs more room than the proposal row index.
MAX_MEMORY_INDEX_ROWS = 2000
# A pending proposal is inert after this long. Expiry is a status transition;
# the kv row and its audit trail are retained for inspection.
DEFAULT_PENDING_TTL_SECONDS = 7 * 24 * 3600
MAX_BATCH_DRAFTS = 500  # hard ceiling for one coalesced batch
_BATCH_IDEAS = 8  # how many distinct ideas are rendered into aggregate content
_PENDING_STATUSES = frozenset({"proposed", "awaiting_approval"})
_MAX_TAGS = 10
_MAX_TAG_CHARS = 64
_MAX_DEDUPE_SCAN = 200
_MAX_CLUSTERS = 10
MAX_CONTENT_CHARS = 1200
MIN_CONTENT_CHARS = 8
DEFAULT_SIMILARITY = 0.78

# Only source kinds the engine itself can derive are eligible for an auto
# policy. Chat text (``user``) is data, not an authority, and ``external`` is a
# fallback label for caller-supplied drafts: neither is a source kind here.
_SOURCE_KINDS = frozenset({"user", "tool", "run"})
_CHAT_SOURCE_KINDS = frozenset({"user", "chat", "assistant", "model"})

# Strong token shapes: these are dropped, never merely masked, because a note
# is written to long-lived memory and a fragment is still useful to an attacker.
_SECRET_NAME = (
    r"(?:api[\s_-]*keys?|api[\s_-]*secrets?|access[\s_-]*tokens?|"
    r"refresh[\s_-]*tokens?|auth[\s_-]*tokens?|bearer[\s_-]*tokens?|"
    r"client[\s_-]*secrets?|secret[\s_-]*keys?|private[\s_-]*keys?|"
    r"pass[\s_-]*phrases?|passwords?|passwd|pwd|credentials?|"
    r"secrets?|tokens?|pins?|otps?|authorization)"
)
_SECRET_VALUE = r"[\"'`]?[^\s,;\"'`]{1,}"
_INTRINSIC_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsk(?:-[A-Za-z0-9_-]{4,}){2,}",
        r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b",
        r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
        r"\b(?:AKIA|ASIA|AIDA|AROA|AGPA|AIPA|ANPA|ANVA)[A-Z0-9]{16}\b",
        r"\bAIza[0-9A-Za-z_-]{35}\b",
        r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}\b",
        # Key + value, with punctuation, whitespace or natural phrasing. A
        # short value ("PIN: 12", "the API key: value") is still a disclosure.
        rf"\b{_SECRET_NAME}\b[\s]*[:=]>?[\s]*{_SECRET_VALUE}",
        rf"\b{_SECRET_NAME}\b[\s]+(?:is|are|was|were|equals|set[\s]+to|value[\s]+is)\b[\s]+{_SECRET_VALUE}",
        rf"\b{_SECRET_NAME}\b[\s]+{_SECRET_VALUE}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@",
    )
)
# Injection shapes survive punctuation, articles, case changes and hyphens.
_INJ_SEP = r"[\W_]{0,12}"
_INJECTION_PATTERN = re.compile(
    r"(?i)("
    r"\b(?:ignore|disregard|forget|override|bypass)\b" + _INJ_SEP +
    r"(?:(?:all|any|the|these|those|your|my)\b" + _INJ_SEP + r")?"
    r"(?:previous|prior|above|earlier|preceding|system|developer)\b" + _INJ_SEP +
    r"(?:instructions?|rules?|prompts?|messages?|directions?|commands?)\b"
    r"|\b(?:system|developer)" + _INJ_SEP + r"(?:prompt|mode|instructions?|messages?|role)\b"
    r"|\byou" + _INJ_SEP + r"are" + _INJ_SEP + r"now\b"
    r"|<\|\s*im_start\s*\|>|</?system>"
    r"|\brm[\W_]{0,4}-rf\b|\bsudo\b|\bcurl[\W_]{1,12}|\bwget[\W_]{1,12}|\bchmod[\W_]{0,4}777\b"
    r"|\beval[\W_]{0,4}\(|\bexec[\W_]{0,4}\("
    r")"
)
_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069\ufeff]")
_WHITESPACE = re.compile(r"\s+")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SAFE_TIMESTAMP = re.compile(r"^[0-9T:+\-.Z ]{1,40}$")

# Only explicit, durable user statements are treated as candidate facts. Model
# output is never evidence: a model asserting "remember X" proves nothing.
_USER_STATEMENTS: tuple[tuple[re.Pattern[str], MemoryCategory, float], ...] = (
    (re.compile(r"(?i)^\s*remember\s+(?:that\s+)?(.+)$"), MemoryCategory.SEMANTIC, 0.70),
    (re.compile(r"(?i)^\s*my name is\s+(.+)$"), MemoryCategory.PERSONAL, 0.80),
    (re.compile(r"(?i)^\s*i prefer\s+(.+)$"), MemoryCategory.PERSONAL, 0.70),
    (re.compile(r"(?i)^\s*(?:we|the project|this project)\s+(?:always\s+)?(?:use|uses|prefer|prefers)\s+(.+)$"),
     MemoryCategory.PROJECT, 0.60),
)
_HIGH_RISK_CATEGORIES = frozenset(
    {MemoryCategory.PERSONAL, MemoryCategory.ORGANIZATIONAL, MemoryCategory.PROJECT}
)
_LOW_RISK_CATEGORIES = frozenset(
    {
        MemoryCategory.SEMANTIC,
        MemoryCategory.PROCEDURAL,
        MemoryCategory.EPISODIC,
        MemoryCategory.WORKING,
    }
)


class LearningDenied(Exception):
    """A learning operation was refused (evidence, safety or gate failure)."""


class ApprovalRequired(LearningDenied):
    """An accept/consolidate needs a resolved, exactly-bound approval."""


class AutoNotePolicy(BaseModel):
    """Explicit operator opt-in for low-risk note writes.

    ``trusted_sources`` is a per-kind allowlist of source kinds the engine can
    derive (``user``, ``tool``, ``run``). Wildcards, chat actors, model names
    and the ``external`` fallback are refused. Trusting a source kind only
    makes a draft *auto-eligible*; writes stay approval-bound until the
    operator also sets ``allow_unattended_writes``. Chat text is never promoted
    to ``FACT`` on an unattended write.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    trusted_sources: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = tuple(c.value for c in _LOW_RISK_CATEGORIES)
    max_confidence: float = 0.7
    promote_facts: bool = False
    allow_unattended_writes: bool = False

    def validated(self) -> AutoNotePolicy:
        unknown = [kind for kind in self.trusted_sources if kind not in _SOURCE_KINDS]
        if unknown:
            raise ValueError(
                f"unsupported trusted source kinds: {sorted(unknown)}; "
                f"expected explicit per-kind opt-in from {sorted(_SOURCE_KINDS)}"
            )
        if len(set(self.trusted_sources)) != len(self.trusted_sources):
            raise ValueError("trusted source kinds must be unique")
        if not self.enabled:
            return self
        if not self.trusted_sources:
            raise ValueError("an auto-note policy must name trusted sources")
        unsafe = set(self.allowed_categories) - {c.value for c in _LOW_RISK_CATEGORIES}
        if unsafe:
            raise ValueError(f"auto-note policy cannot trust high-risk categories: {sorted(unsafe)}")
        if not math.isfinite(self.max_confidence) or not 0.0 <= self.max_confidence <= 1.0:
            raise ValueError("max_confidence must be within [0, 1]")
        if (
            self.allow_unattended_writes
            and self.promote_facts
            and _CHAT_SOURCE_KINDS.intersection(self.trusted_sources)
        ):
            raise ValueError("untrusted chat sources cannot be promoted to FACT without approval")
        return self


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    kind: str  # message | tool_call | run | external
    hash: str = ""
    conversation_id: str | None = None
    created_at: str | None = None


class NoteDraft(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str
    category: MemoryCategory = MemoryCategory.SEMANTIC
    kind: MemoryKind = MemoryKind.UNVERIFIED
    confidence: float = 0.6
    source_kind: str = "external"
    conversation_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    reason: str = ""


class LearningProposal(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    draft: NoteDraft
    # ... | expired is a terminal, non-deleting state for stale pendings.
    status: str = "proposed"  # proposed | awaiting_approval | accepted | rejected | superseded | duplicate | denied | failed | expired
    policy: str = "approval"  # approval | auto
    # Bound into the arguments digest: a stale/tampered proposal cannot flip
    # itself to an approval-free write after the policy changed.
    unattended: bool = False
    resource: str
    arguments_digest: str
    approval_id: str | None = None
    memory_id: str | None = None
    risk: str = RiskLevel.HIGH.value
    reason: str = ""
    superseded_by: str | None = None
    created_at: str = Field(default_factory=utcnow)
    expires_at: str | None = None
    resolved_at: str | None = None
    # A batch proposal is still ONE proposal and ONE approval; these fields
    # bind the exact draft set into the arguments digest and let accept()
    # re-verify it against durable state.
    batch_size: int = 1
    batch_fingerprint: str = ""
    batch_hashes: list[str] = Field(default_factory=list)


class ConsolidationPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    category: str | None = None
    clusters: list[list[str]] = Field(default_factory=list)
    resource: str
    arguments_digest: str
    approval_id: str | None = None
    status: str = "awaiting_approval"  # awaiting_approval | applied | denied | empty | failed
    reason: str = ""
    created_at: str = Field(default_factory=utcnow)
    resolved_at: str | None = None


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:32]


def _safe_ref(value: Any, fallback: str) -> str:
    text = str(value or "")
    return text if _SAFE_ID.fullmatch(text) else fallback


def _category_value(value: Any) -> str:
    return value.value if isinstance(value, MemoryCategory) else str(value)


def _kind_value(value: Any) -> str:
    return value.value if isinstance(value, MemoryKind) else str(value)


class LearningEngine:
    """Propose and gate memory notes derived from durable LEO history."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        events: EventBus | None = None,
        permissions: PermissionManagerImpl | None = None,
        settings: Settings | None = None,
        consolidator: Consolidator | None = None,
        *,
        auto_policy: AutoNotePolicy | Mapping[str, Any] | None = None,
        actor: str = ACTOR,
        similarity_threshold: float = DEFAULT_SIMILARITY,
        min_occurrences: int = 3,
        max_pending_proposals: int = MAX_PENDING_PROPOSALS,
        pending_ttl_seconds: float = DEFAULT_PENDING_TTL_SECONDS,
    ) -> None:
        self.settings = settings or Settings()
        self.events = events or EventBus()
        self.store = store or MemoryStore(self.events)
        self.permissions = permissions or PermissionManagerImpl(self.settings, self.events)
        self.consolidator = consolidator or Consolidator(self.store)
        policy = AutoNotePolicy() if auto_policy is None else AutoNotePolicy.model_validate(auto_policy)
        self.auto_policy = policy.validated()
        self.actor = str(actor or ACTOR)
        if not 0.0 <= float(similarity_threshold) <= 1.0:
            raise ValueError("similarity_threshold must be within [0, 1]")
        self.similarity_threshold = float(similarity_threshold)
        if min_occurrences < 2:
            raise ValueError("min_occurrences must be at least 2")
        self.min_occurrences = int(min_occurrences)
        if int(max_pending_proposals) < 1:
            raise ValueError("max_pending_proposals must be at least 1")
        self.max_pending_proposals = int(max_pending_proposals)
        ttl = float(pending_ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("pending_ttl_seconds must be a finite positive number")
        self.pending_ttl_seconds = ttl
        self._patterns: tuple[re.Pattern[str], ...] = self._compile_patterns()

    def _compile_patterns(self) -> tuple[re.Pattern[str], ...]:
        compiled: list[re.Pattern[str]] = []
        for raw in list(self.settings.security.redact_patterns or ()):
            try:
                compiled.append(re.compile(str(raw), re.IGNORECASE))
            except re.error:
                # One bad operator pattern must not silently disable redaction.
                _LOG.warning("invalid learning redaction pattern ignored")
        return tuple(compiled)

    # ------------------------------------------------------------------ state
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

    async def _prune_kv(
        self, prefix: str, keep: int, *, protected_keys: Sequence[str] = ()
    ) -> None:
        """Cap durable rows for one prefix; oldest rows are dropped first.

        Outstanding proposals are protected: dropping one would orphan its
        approval and silently free a cap slot while the approval still exists.
        """
        protect = set(protected_keys)
        rows = await db.fetchall(
            "SELECT key FROM kv WHERE key LIKE ? ORDER BY updated_at DESC, rowid DESC",
            (prefix + "%",),
        )
        for row in rows[max(0, int(keep)):]:
            if row["key"] in protect:
                continue
            await db.execute("DELETE FROM kv WHERE key=?", (row["key"],))

    async def _index(self) -> dict[str, Any]:
        value = await self._load("index") or {}
        return value if isinstance(value, dict) else {}

    async def _save_index(self, index: Mapping[str, Any]) -> None:
        # Bounded: a long-running engine must not grow a single kv row forever.
        seen = [str(item)[:160] for item in list(index.get("seen") or [])[-_SEEN_LIMIT:]]
        proposals = dict(index.get("proposals") or {})
        if len(proposals) > MAX_PROPOSAL_ROWS:
            proposals = dict(list(proposals.items())[-MAX_PROPOSAL_ROWS:])
        memories = dict(index.get("memories") or {})
        if len(memories) > MAX_MEMORY_INDEX_ROWS:
            memories = dict(list(memories.items())[-MAX_MEMORY_INDEX_ROWS:])
        await self._save("index", {"seen": seen, "proposals": proposals, "memories": memories})

    # ------------------------------------------------------ pending / expiry
    @staticmethod
    def _parse_stamp(value: Any) -> datetime | None:
        try:
            stamp = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp

    def _expires_at(self) -> str:
        """Stamp a newly created pending proposal with its bounded TTL."""
        return (datetime.now(timezone.utc) + timedelta(seconds=self.pending_ttl_seconds)).isoformat()

    def _pending_clock(self) -> tuple[str, str]:
        """One clock read for creation and expiry, so the TTL is exact."""
        now = datetime.now(timezone.utc)
        return now.isoformat(), (now + timedelta(seconds=self.pending_ttl_seconds)).isoformat()

    async def _proposal_rows(self) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT value FROM kv WHERE key LIKE ? ORDER BY updated_at DESC",
            (self._key("proposal:") + "%",),
        )
        result: list[dict[str, Any]] = []
        for row in rows[:MAX_PROPOSAL_ROWS]:
            try:
                value = json.loads(row["value"])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    def _is_expired(self, row: Mapping[str, Any]) -> bool:
        if row.get("status") not in _PENDING_STATUSES:
            return False
        now = datetime.now(timezone.utc)
        expires = self._parse_stamp(row.get("expires_at"))
        if expires is None:
            # Older rows predate ``expires_at``: derive it from creation time.
            created = self._parse_stamp(row.get("created_at"))
            if created is None:
                return False  # unparseable durable state must never auto-expire
            expires = created + timedelta(seconds=self.pending_ttl_seconds)
        return expires <= now

    async def _expire_linked_approval(self, approval_id: str | None) -> None:
        """Mark the approval of an expired proposal expired; never delete it."""
        if not approval_id:
            return
        async with db.transaction():
            row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,))
            if row and row["status"] == "pending":
                await db.execute(
                    "UPDATE approvals SET status='expired', resolved_at=? "
                    "WHERE id=? AND status='pending'",
                    (utcnow(), approval_id),
                )

    async def expire_stale_proposals(self) -> list[dict[str, Any]]:
        """Move stale pendings to ``expired`` as a state change only.

        Rows, digests, approval ids and audit history are retained: an operator
        can still see exactly what was proposed and why it was never written.
        A human who already *approved* the row is left alone; ``accept`` will
        still refuse the proposal because its status is no longer writable.
        """
        expired: list[dict[str, Any]] = []
        for row in await self._proposal_rows():
            if not self._is_expired(row):
                continue
            proposal = LearningProposal.model_validate(row)
            proposal.status = "expired"
            proposal.reason = "pending proposal expired before resolution; no memory written"
            proposal.resolved_at = utcnow()
            await self._save(f"proposal:{proposal.id}", proposal.model_dump())
            await self._expire_linked_approval(proposal.approval_id)
            await self._audit(
                ACTION_NOTE, proposal.resource, "expired", proposal.reason,
                {"proposal_id": proposal.id, "approval_id": proposal.approval_id,
                 "batch_size": proposal.batch_size},
                risk=proposal.risk, correlation_id=proposal.id,
            )
            await self._emit(
                "learning.expired", proposal_id=proposal.id, reason=proposal.reason,
                approval_id=proposal.approval_id, batch_size=proposal.batch_size,
            )
            expired.append(proposal.model_dump())
        return expired

    async def _pending_state(self) -> list[dict[str, Any]]:
        """Sweep stale pendings, then return every proposal still outstanding."""
        await self.expire_stale_proposals()
        return [
            row for row in await self._proposal_rows()
            if row.get("status") in _PENDING_STATUSES
        ]

    async def _reject_at_cap(
        self,
        value: NoteDraft,
        *,
        resource: str,
        risk: str,
        digest: str,
        pending: Sequence[Mapping[str, Any]],
        batch: Mapping[str, Any] | None = None,
        batch_hashes: Sequence[str] = (),
        topic: str = "learning.rejected",
    ) -> LearningProposal:
        """Return an audited, truthful rejection instead of an unbacked approval.

        No proposal row is persisted: a capped draft is not part of the
        lifecycle, and writing hundreds of terminal rows would recreate the
        very backlog this cap exists to prevent. The audit row and event keep
        the decision explainable.
        """
        target = str(pending[0].get("id") or "none") if pending else "none"
        reason = (
            f"pending proposal cap ({self.max_pending_proposals}) reached; rejected without "
            f"an approval, coalesce into {target} or retry after resolution"
        )
        proposal = LearningProposal(
            id=new_id("lpro"), draft=value, status="rejected", policy="approval",
            unattended=False, resource=resource, arguments_digest=digest,
            risk=risk, reason=reason, resolved_at=utcnow(),
            batch_size=int(batch.get("size", 1)) if batch else 1,
            batch_fingerprint=str(batch.get("fingerprint", "")) if batch else "",
            batch_hashes=list(batch_hashes),
        )
        await self._audit(
            ACTION_NOTE, resource, "denied", reason,
            {"proposal_id": proposal.id, "pending_cap": self.max_pending_proposals,
             "pending_examples": [str(item.get("id") or "") for item in list(pending)[:3]]},
            risk=risk, correlation_id=proposal.id,
        )
        await self._emit(
            topic, proposal_id=proposal.id, status="rejected", reason=reason,
            batch_size=proposal.batch_size, approval_id=None,
        )
        return proposal

    # --------------------------------------------------------------- sanitize
    def _clean(self, value: Any, depth: int = 0) -> Any:
        """Redact any string before it reaches an event, audit row or kv row."""
        if isinstance(value, str):
            text = value
            for pattern in self._patterns:
                text = pattern.sub("[REDACTED]", text)
            for pattern in _INTRINSIC_SECRET_PATTERNS:
                text = pattern.sub("[REDACTED]", text)
            return text[:MAX_CONTENT_CHARS]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= 4:
            return "[truncated]"
        if isinstance(value, Mapping):
            return {str(k): self._clean(v, depth + 1) for k, v in list(value.items())[:50]}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._clean(v, depth + 1) for v in list(value)[:50]]
        return "[REDACTED]"

    def _sanitize(self, value: Any) -> str:
        """Validate and bound one note content string, failing closed."""
        text = str(value or "")
        if _CONTROL_PATTERN.search(text):
            raise LearningDenied("refusing content with control or bidi characters")
        if any(pattern.search(text) for pattern in _INTRINSIC_SECRET_PATTERNS):
            raise LearningDenied("refusing secret-shaped content")
        for pattern in self._patterns:
            text = pattern.sub("[REDACTED]", text)
        if _INJECTION_PATTERN.search(text):
            raise LearningDenied("refusing instruction-like content")
        text = _WHITESPACE.sub(" ", text).strip()
        if not MIN_CONTENT_CHARS <= len(text) <= MAX_CONTENT_CHARS:
            raise LearningDenied(
                f"note content must be {MIN_CONTENT_CHARS}-{MAX_CONTENT_CHARS} characters"
            )
        return text

    def _sanitize_fragment(self, value: Any, limit: int, *, field: str) -> str:
        """Validate a short, non-content field before it reaches durable state."""
        text = str(value or "")
        if _CONTROL_PATTERN.search(text):
            raise LearningDenied(f"refusing {field} with control or bidi characters")
        if any(pattern.search(text) for pattern in _INTRINSIC_SECRET_PATTERNS):
            raise LearningDenied(f"refusing secret-shaped {field}")
        for pattern in self._patterns:
            text = pattern.sub("[REDACTED]", text)
        if _INJECTION_PATTERN.search(text):
            raise LearningDenied(f"refusing instruction-like {field}")
        return _WHITESPACE.sub(" ", text).strip()[:limit]

    def _sanitize_tag(self, value: Any) -> str:
        """Tags obey the same safety rules but have no minimum length."""
        return self._sanitize_fragment(value, _MAX_TAG_CHARS, field="tag")

    def _normalize_draft(self, draft: NoteDraft | Mapping[str, Any]) -> NoteDraft:
        value = draft if isinstance(draft, NoteDraft) else NoteDraft.model_validate(dict(draft))
        content = self._sanitize(value.content)
        try:
            category = MemoryCategory(_category_value(value.category))
            kind = MemoryKind(_kind_value(value.kind))
        except ValueError as exc:
            raise LearningDenied(f"invalid memory category/kind: {exc}") from exc
        number = float(value.confidence)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise LearningDenied("confidence must be a finite number within [0, 1]")
        source_kind = _safe_ref(value.source_kind, "external")
        conversation_id = (
            _safe_ref(value.conversation_id, "unknown") if value.conversation_id is not None else None
        )
        tags = [self._sanitize_tag(tag) for tag in list(value.tags)[:_MAX_TAGS]]
        if len(value.evidence) > MAX_EVIDENCE:
            raise LearningDenied("too many evidence references")
        evidence: list[EvidenceRef] = []
        for ref in list(value.evidence)[:MAX_EVIDENCE]:
            stamp = (
                self._sanitize_fragment(ref.created_at, 40, field="evidence timestamp")
                if ref.created_at else ""
            )
            created_at = stamp if stamp and _SAFE_TIMESTAMP.fullmatch(stamp) else None
            # Hashes are hex digests; never copy raw text into evidence.
            evidence.append(
                EvidenceRef(
                    id=_safe_ref(ref.id, "unknown"),
                    kind=_safe_ref(ref.kind, "external"),
                    hash=_safe_ref(ref.hash, "")[:64],
                    conversation_id=(
                        _safe_ref(ref.conversation_id, "unknown") if ref.conversation_id else None
                    ),
                    created_at=created_at or None,
                )
            )
        supersedes = [
            _safe_ref(item, "unknown") for item in list(value.supersedes)[:MAX_SUPERSEDES]
        ]
        reason = self._sanitize_fragment(value.reason, 280, field="reason")
        # Rebuild explicitly: unknown/extra fields are untrusted data and must
        # not be persisted in a proposal kv row.
        return NoteDraft(
            content=content, category=category, kind=kind, confidence=number,
            source_kind=source_kind, conversation_id=conversation_id, tags=tags,
            evidence=evidence, supersedes=supersedes, reason=reason,
        )

    # ------------------------------------------------------------------ gates
    def _risk(self, draft: NoteDraft) -> str:
        if draft.category in _HIGH_RISK_CATEGORIES or draft.supersedes:
            return RiskLevel.HIGH.value
        if draft.category not in _LOW_RISK_CATEGORIES:
            return RiskLevel.HIGH.value
        return RiskLevel.LOW.value

    def _auto_eligible(self, draft: NoteDraft) -> bool:
        policy = self.auto_policy
        if not policy.enabled or self._risk(draft) != RiskLevel.LOW.value:
            return False
        if draft.source_kind not in policy.trusted_sources:
            return False
        if _category_value(draft.category) not in policy.allowed_categories:
            return False
        if draft.confidence > policy.max_confidence or draft.supersedes:
            return False
        return True

    @staticmethod
    def _is_chat_source(source_kind: str) -> bool:
        """Chat/assistant text is untrusted data even under an auto policy."""
        return str(source_kind).casefold() in _CHAT_SOURCE_KINDS

    def _arguments(
        self,
        proposal_id: str,
        draft: NoteDraft,
        *,
        unattended: bool = False,
        batch: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "proposal_id": proposal_id,
            "content": draft.content,
            "category": _category_value(draft.category),
            "kind": _kind_value(draft.kind),
            "confidence": round(float(draft.confidence), 4),
            "source_kind": draft.source_kind,
            "conversation_id": draft.conversation_id,
            "unattended": bool(unattended),
            "evidence": [
                {"id": ref.id, "kind": ref.kind, "hash": ref.hash}
                for ref in list(draft.evidence)[:MAX_EVIDENCE]
            ],
            "supersedes": list(draft.supersedes)[:MAX_SUPERSEDES],
        }
        if batch is not None:
            # The canonical digest then binds the exact set of coalesced
            # drafts, so a stale batch cannot be silently repointed at a
            # different queue after the human approved what they saw.
            arguments["batch"] = {
                "size": int(batch.get("size", 1)),
                "fingerprint": str(batch.get("fingerprint", "")),
            }
        return arguments

    @staticmethod
    def _resource(draft: NoteDraft) -> str:
        return f"learning/note/{_category_value(draft.category)}/{_safe_ref(draft.source_kind, 'external')}"

    def _trusted_resolver(self, row: Any) -> str | None:
        """Only a resolved, non-chat human distinct from the requester counts."""
        if not row or row["status"] != "approved" or not row["resolved_at"]:
            return None
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        resolver = ""
        if isinstance(metadata, Mapping):
            resolver = str(metadata.get("resolved_by") or "").strip()
        if not resolver or resolver.casefold().startswith("chat:"):
            return None
        if resolver.casefold() in {self.actor.casefold(), str(row["actor"] or "").casefold()}:
            return None
        return resolver

    async def _consume_approval(
        self,
        *,
        action: str,
        resource: str,
        arguments: Mapping[str, Any],
        digest: str,
        approval_id: str | None,
        correlation_id: str,
    ) -> tuple[str | None, str | None]:
        """Verify and one-shot consume an exactly-bound HIGH-risk approval.

        The conditional update and its audit row share one write transaction,
        so a replay or a concurrent consumer cannot apply the same decision
        twice or leave an approval granted without a trail.
        """
        failure: str | None = "missing HIGH-risk approval id"
        resolver: str | None = None
        if not approval_id:
            return failure, resolver
        async with db.transaction():
            row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
            if not row or row["status"] != "approved":
                failure = "approval is not approved"
                return failure, resolver
            resolver = self._trusted_resolver(row)
            if not resolver:
                failure = "approval resolver is chat, self or unknown"
                return failure, resolver
            if (
                row["action"] != action
                or row["resource"] != resource
                or row["risk"] != RiskLevel.HIGH.value
                or row["actor"] != self.actor
            ):
                return "approval action/resource/risk/actor binding mismatch", resolver
            try:
                expected = json.loads(row["metadata"] or "{}").get("arguments_digest")
            except (TypeError, ValueError, AttributeError):
                expected = None
            if not digest or not expected or expected != digest:
                return "approval argument digest mismatch", resolver
            decision = await self.permissions.authorize(
                self.actor, action, resource, RiskLevel.HIGH,
                approval_id=approval_id, arguments=dict(arguments), arguments_digest=digest,
            )
            if not decision.allowed or decision.approval_id != approval_id:
                return "approval binding expired or invalid", resolver
            cursor = await db.execute(
                "UPDATE approvals SET status='consumed' WHERE id=? AND status='approved'",
                (approval_id,),
            )
            if cursor.rowcount != 1:
                return "approval already consumed", resolver
            await self._audit(
                action, resource, "allowed",
                "HIGH-risk approval consumed by trusted human",
                {"proposal_id": correlation_id, "approval_id": approval_id,
                 "arguments_digest": digest, "resolver": resolver},
                correlation_id=correlation_id,
            )
            failure = None
        return failure, resolver

    # -------------------------------------------------------------- audit/bus
    async def _audit(
        self,
        action: str,
        resource: str,
        decision: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        risk: str = RiskLevel.HIGH.value,
        correlation_id: str | None = None,
    ) -> None:
        await db.execute(
            "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,correlation_id) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                utcnow(), self.actor, action, str(resource)[:240], risk,
                str(decision)[:40], str(reason)[:500],
                json.dumps(self._clean(metadata or {}), ensure_ascii=False, default=str)[:2000],
                correlation_id,
            ),
        )

    async def _emit(self, topic: str, **payload: Any) -> None:
        clean = self._clean(payload)
        clean.setdefault("actor", self.actor)
        clean.setdefault("ts", utcnow())
        correlation = (
            clean.get("proposal_id") or clean.get("plan_id") or clean.get("memory_id") or "learning"
        )
        await self.events.publish(
            Event(
                topic=topic, source="learning", payload=clean,
                correlation_id=str(correlation),
            )
        )

    # ----------------------------------------------------------------- dedupe
    async def _find_duplicate(
        self,
        draft: NoteDraft,
        *,
        exclude_proposal: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Return (proposal_id, memory_id) for a near-duplicate, if any."""
        key = _hash(normalize_content(draft.content))
        index = await self._index()
        proposal_id = str((index.get("proposals") or {}).get(key) or "") or None
        if proposal_id and proposal_id != exclude_proposal:
            other = await self._load(f"proposal:{proposal_id}", LearningProposal)
            if other and other.status in {"proposed", "awaiting_approval", "accepted"}:
                return proposal_id, other.memory_id
        # A coalesced batch records its constituent hashes against the single
        # memory it writes; re-proposing one of them must not create a second
        # note. Only live memories count, so superseded/merged ids are ignored.
        memory_id = str((index.get("memories") or {}).get(key) or "") or None
        if memory_id and await self.store.get(memory_id) is not None:
            return None, memory_id
        for record in await self.store.list(limit=_MAX_DEDUPE_SCAN):
            if record.id in draft.supersedes:
                continue
            if content_similarity(draft.content, record) >= self.similarity_threshold:
                return None, record.id
        return None, None

    # ---------------------------------------------------------------- propose
    async def propose(self, draft: NoteDraft | Mapping[str, Any]) -> dict[str, Any]:
        """Validate a draft, dedupe it and create an inert proposal.

        Auto-eligible drafts stay approval-bound unless the operator set
        ``allow_unattended_writes``. Requesting an approval never grants it,
        and a disabled gate is a denial rather than permission to self-write.
        Content safety is checked before any approval row can be created.
        """
        value = self._normalize_draft(draft)
        pending = await self._pending_state()
        risk = self._risk(value)
        auto = self._auto_eligible(value)
        unattended = auto and bool(self.auto_policy.allow_unattended_writes)
        if auto and self.auto_policy.promote_facts and not (
            unattended and self._is_chat_source(value.source_kind)
        ):
            value.kind = MemoryKind.FACT
        proposal_id = new_id("lpro")
        resource = self._resource(value)
        arguments = self._arguments(proposal_id, value, unattended=unattended)
        digest = arguments_digest(arguments) or ""
        if not digest:
            raise LearningDenied("could not bind approval arguments")

        duplicate_proposal, duplicate_memory = await self._find_duplicate(value)
        reason = ""
        if duplicate_proposal or duplicate_memory:
            status = "duplicate"
            reason = "near-duplicate of existing proposal or memory"
        elif unattended:
            status = "proposed"
            reason = "operator enabled unattended auto-note writes for this source kind"
        else:
            status = "awaiting_approval"
            reason = "auto-eligible note is still approval-bound" if auto else ""

        if status in _PENDING_STATUSES and len(pending) >= self.max_pending_proposals:
            # No approval row is created for a capped draft, and no inert
            # proposal row is left behind to masquerade as pending work.
            rejected = await self._reject_at_cap(
                value, resource=resource, risk=risk, digest=digest, pending=pending,
            )
            return rejected.model_dump()

        created_at, expires_at = self._pending_clock()
        proposal = LearningProposal(
            id=proposal_id, draft=value, status=status,
            policy="auto" if auto else "approval", unattended=unattended,
            resource=resource, arguments_digest=digest, risk=risk, reason=reason,
            created_at=created_at,
            expires_at=expires_at if status in _PENDING_STATUSES else None,
        )
        # Even duplicates are persisted: operators can inspect why a note was
        # refused, and a later accept produces a truthful lifecycle error.
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        if status != "duplicate":
            index = await self._index()
            proposals = dict(index.get("proposals") or {})
            proposals[_hash(normalize_content(value.content))] = proposal_id
            index["proposals"] = proposals
            await self._save_index(index)

        if status == "awaiting_approval":
            decision = await self.permissions.authorize(
                self.actor, ACTION_NOTE, resource, RiskLevel.HIGH,
                arguments=arguments, arguments_digest=digest,
                proposal_id=proposal_id, source_kind=value.source_kind,
                note_risk=risk, unattended=False, title="learn a durable memory note",
                content_preview=value.content[:160],
            )
            proposal.approval_id = decision.approval_id
            if decision.allowed and not decision.requires_approval:
                # A disabled HIGH-risk gate is not consent to self-modify.
                proposal.status = "denied"
                proposal.reason = "HIGH-risk approval gate disabled; refusing to self-write"
            else:
                proposal.reason = "awaiting HIGH-risk approval"
            await self._save(f"proposal:{proposal_id}", proposal.model_dump())
            decision_name = "denied" if proposal.status == "denied" else "pending"
        elif status == "duplicate":
            decision_name = "duplicate"
        else:
            decision_name = "allowed"

        await self._prune_kv(
            self._key("proposal:"), MAX_PROPOSAL_ROWS,
            protected_keys=[self._key(f"proposal:{item['id']}") for item in pending],
        )
        await self._audit(
            ACTION_NOTE, resource, decision_name, proposal.reason or status,
            {"proposal_id": proposal_id, "approval_id": proposal.approval_id,
             "arguments_digest": digest, "source_kind": value.source_kind,
             "duplicate_memory_id": duplicate_memory},
            risk=risk, correlation_id=proposal_id,
        )
        await self._emit(
            "learning.proposed", proposal_id=proposal_id, status=proposal.status,
            policy=proposal.policy, unattended=proposal.unattended,
            risk=risk, category=_category_value(value.category),
            kind=_kind_value(value.kind), approval_id=proposal.approval_id,
            evidence_ids=[ref.id for ref in list(value.evidence)[:MAX_EVIDENCE]],
            duplicate_memory_id=duplicate_memory,
        )
        return proposal.model_dump()

    # ----------------------------------------------------------- batch propose
    def _batch_draft(
        self,
        drafts: Sequence[NoteDraft],
        content_hashes: set[str],
        *,
        reason: str = "",
    ) -> NoteDraft:
        """Build ONE inert draft representing a whole queue of safe drafts."""
        fingerprint = _hash(sorted(content_hashes))
        ideas = [draft.content for draft in drafts]
        prefix = f"Learning batch of {len(drafts)} grounded notes [{fingerprint[:12]}]"
        text = prefix + ": " + " | ".join(ideas)
        if len(text) > MAX_CONTENT_CHARS:
            room = MAX_CONTENT_CHARS - len(prefix) - 2
            kept: list[str] = []
            for idea in ideas:
                candidate = " | ".join([*kept, idea])
                if len(candidate) > room:
                    break
                kept.append(idea)
            text = prefix + ": " + " | ".join(kept)
        content = self._sanitize(text)

        categories = {_category_value(draft.category) for draft in drafts}
        high = sorted(categories & {category.value for category in _HIGH_RISK_CATEGORIES})
        category = MemoryCategory(high[0]) if high else MemoryCategory.SEMANTIC

        evidence: list[EvidenceRef] = []
        seen_refs: set[tuple[str, str]] = set()
        for ref in sorted(
            (ref for draft in drafts for ref in draft.evidence),
            key=lambda ref: (ref.kind, ref.id, ref.hash),
        ):
            key = (ref.kind, ref.id)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            evidence.append(ref)
            if len(evidence) >= MAX_EVIDENCE:
                break

        supersedes: list[str] = []
        for draft in drafts:
            for old in draft.supersedes:
                if old not in supersedes:
                    supersedes.append(old)
            if len(supersedes) >= MAX_SUPERSEDES:
                break
        tags = ["batch"]
        for draft in drafts:
            for tag in draft.tags:
                if tag not in tags:
                    tags.append(tag)
            if len(tags) >= _MAX_TAGS:
                break
        conversations = {draft.conversation_id for draft in drafts}
        note_reason = f"coalesced {len(drafts)} grounded note drafts into one approval-bound proposal"
        if reason:
            note_reason += f"; {reason}"
        return NoteDraft(
            content=content, category=category, kind=MemoryKind.UNVERIFIED,
            confidence=min(float(draft.confidence) for draft in drafts),
            # An aggregate has no single trusted source kind: "external" keeps
            # it out of every unattended auto-write policy.
            source_kind="external",
            conversation_id=next(iter(conversations)) if len(conversations) == 1 else None,
            tags=tags[:_MAX_TAGS], evidence=evidence,
            supersedes=supersedes[:MAX_SUPERSEDES],
            reason=self._sanitize_fragment(note_reason, 280, field="reason"),
        )

    async def propose_batch(
        self,
        drafts: Sequence[NoteDraft | Mapping[str, Any]],
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        """Coalesce a queue of drafts into ONE proposal and ONE HIGH-risk approval.

        Each draft still passes the normal content-safety, evidence and dedupe
        gates. The aggregate content plus a canonical fingerprint of the exact
        draft set are bound into the single arguments digest, so the human who
        resolves the approval sees and authorizes the whole queue. No note is
        written here; accepting the one proposal writes one memory record.
        """
        if isinstance(drafts, (NoteDraft, Mapping)):
            raise LearningDenied("propose_batch expects a sequence of drafts")
        queue = list(drafts)
        if not queue:
            return {
                "status": "empty", "count": 0, "coalesced": 0, "proposal_id": None,
                "approval_id": None, "arguments_digest": None,
                "reason": "no drafts to coalesce",
            }
        if len(queue) > MAX_BATCH_DRAFTS:
            raise LearningDenied(f"batch exceeds {MAX_BATCH_DRAFTS} drafts")
        # Validate the whole queue before writing anything: one hostile draft
        # fails the batch closed rather than leaving a half-approved queue.
        normalized = [self._normalize_draft(item) for item in queue]
        # Sweep first so an expired proposal cannot masquerade as a live
        # duplicate, and so the cap reflects only genuinely outstanding work.
        pending = await self._pending_state()

        unique: list[NoteDraft] = []
        hashes: set[str] = set()
        skipped: list[dict[str, str]] = []
        for draft in normalized:
            key = _hash(normalize_content(draft.content))
            if key in hashes:
                skipped.append({"content_hash": key, "status": "duplicate",
                                "reason": "duplicate within batch"})
                continue
            duplicate_proposal, duplicate_memory = await self._find_duplicate(draft)
            if duplicate_proposal or duplicate_memory:
                skipped.append({
                    "content_hash": key, "status": "duplicate",
                    "reason": "near-duplicate of existing proposal or memory",
                    "proposal_id": duplicate_proposal or "",
                    "memory_id": duplicate_memory or "",
                })
                continue
            hashes.add(key)
            unique.append(draft)

        result: dict[str, Any] = {
            "count": len(queue), "coalesced": len(unique), "skipped": skipped,
            "proposal_id": None, "approval_id": None, "arguments_digest": None,
        }
        if not unique:
            result.update(status="duplicate", reason="every draft is a duplicate")
            return result

        aggregate = self._batch_draft(unique, hashes, reason=reason)
        fingerprint = _hash(sorted(hashes))
        batch_info = {"size": len(unique), "fingerprint": fingerprint}
        proposal_id = new_id("lpro")
        resource = self._resource(aggregate)
        arguments = self._arguments(proposal_id, aggregate, unattended=False, batch=batch_info)
        digest = arguments_digest(arguments) or ""
        if not digest:
            raise LearningDenied("could not bind approval arguments")

        duplicate_proposal, duplicate_memory = await self._find_duplicate(aggregate)
        if duplicate_proposal or duplicate_memory:
            result.update(
                status="duplicate", batch_fingerprint=fingerprint,
                reason="aggregate is a near-duplicate of existing proposal or memory",
                duplicate_proposal_id=duplicate_proposal,
                duplicate_memory_id=duplicate_memory,
            )
            return result

        if len(pending) >= self.max_pending_proposals:
            rejected = await self._reject_at_cap(
                aggregate, resource=resource, risk=RiskLevel.HIGH.value,
                digest=digest, pending=pending, batch=batch_info,
                batch_hashes=sorted(hashes), topic="learning.batch_rejected",
            )
            result.update(
                status="rejected", reason=rejected.reason, proposal_id=rejected.id,
                batch_fingerprint=fingerprint, arguments_digest=digest,
            )
            return result

        created_at, expires_at = self._pending_clock()
        proposal = LearningProposal(
            id=proposal_id, draft=aggregate, status="awaiting_approval",
            policy="approval", unattended=False, resource=resource,
            arguments_digest=digest, risk=RiskLevel.HIGH.value,
            reason=aggregate.reason, created_at=created_at, expires_at=expires_at,
            batch_size=len(unique), batch_fingerprint=fingerprint,
            batch_hashes=sorted(hashes),
        )
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        index = await self._index()
        proposals = dict(index.get("proposals") or {})
        proposals[_hash(normalize_content(aggregate.content))] = proposal_id
        index["proposals"] = proposals
        await self._save_index(index)

        decision = await self.permissions.authorize(
            self.actor, ACTION_NOTE, resource, RiskLevel.HIGH,
            arguments=arguments, arguments_digest=digest,
            proposal_id=proposal_id, source_kind=aggregate.source_kind,
            note_risk=RiskLevel.HIGH.value, unattended=False,
            batch_size=len(unique), batch_fingerprint=fingerprint,
            title=f"learn a coalesced batch of {len(unique)} memory notes",
            content_preview=aggregate.content[:160],
        )
        proposal.approval_id = decision.approval_id
        if decision.allowed and not decision.requires_approval:
            proposal.status = "denied"
            proposal.reason = "HIGH-risk approval gate disabled; refusing to self-write"
        else:
            proposal.reason = f"awaiting HIGH-risk approval for {len(unique)} coalesced drafts"
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        await self._prune_kv(
            self._key("proposal:"), MAX_PROPOSAL_ROWS,
            protected_keys=[self._key(f"proposal:{item['id']}") for item in pending],
        )
        decision_name = "denied" if proposal.status == "denied" else "pending"
        await self._audit(
            ACTION_NOTE, resource, decision_name, proposal.reason,
            {"proposal_id": proposal_id, "approval_id": proposal.approval_id,
             "arguments_digest": digest, "batch_size": len(unique),
             "batch_fingerprint": fingerprint},
            risk=proposal.risk, correlation_id=proposal_id,
        )
        await self._emit(
            "learning.batch_proposed", proposal_id=proposal_id, status=proposal.status,
            batch_size=len(unique), batch_fingerprint=fingerprint,
            risk=proposal.risk, approval_id=proposal.approval_id,
            arguments_digest=digest, category=_category_value(aggregate.category),
        )
        result.update(
            status=proposal.status, reason=proposal.reason, proposal_id=proposal_id,
            approval_id=proposal.approval_id, arguments_digest=digest,
            batch_fingerprint=fingerprint, resource=resource,
            expires_at=proposal.expires_at,
        )
        return result

    # ----------------------------------------------------------------- accept
    async def accept(self, proposal_id: str, approval_id: str | None = None) -> dict[str, Any]:
        """Write an accepted proposal to ``MemoryStore`` through the gate."""
        await self.expire_stale_proposals()
        proposal = await self._load(f"proposal:{proposal_id}", LearningProposal)
        if proposal is None:
            raise KeyError(f"unknown learning proposal: {proposal_id}")
        if proposal.status not in {"proposed", "awaiting_approval"}:
            raise LearningDenied(f"proposal is {proposal.status}, not writable")

        draft = proposal.draft
        try:
            # Re-validate at apply time: kv is durable state, not a trusted
            # channel, and an operator policy may have changed since proposal.
            current = self._normalize_draft(draft)
            bound: dict[str, Any] = {"unattended": proposal.unattended}
            if proposal.batch_size > 1 or proposal.batch_fingerprint:
                bound["batch"] = {
                    "size": proposal.batch_size,
                    "fingerprint": proposal.batch_fingerprint,
                }
            if current.content != draft.content or self._arguments(proposal_id, current, **bound) != self._arguments(proposal_id, draft, **bound):
                raise LearningDenied("proposal arguments changed since approval")
            resource = self._resource(current)
            arguments = self._arguments(proposal_id, current, **bound)
            if resource != proposal.resource or arguments_digest(arguments) != proposal.arguments_digest:
                raise LearningDenied("proposal binding changed since approval")
            # Only a proposal created under an explicit unattended policy can
            # take the approval-free gate. A policy enabled after the proposal,
            # a status flip, or a changed source kind cannot open that gate.
            auto_gate = (
                proposal.unattended
                and proposal.status == "proposed"
                and proposal.policy == "auto"
                and bool(self.auto_policy.allow_unattended_writes)
                and self._auto_eligible(current)
            )
            if auto_gate:
                if self._is_chat_source(current.source_kind) and current.kind == MemoryKind.FACT:
                    raise LearningDenied("chat text cannot be promoted to FACT without approval")
                gate = "auto"
            else:
                # A missing approval is not a rejection: the operator may still
                # supply one. Do not burn the proposal on a missing argument.
                if not approval_id:
                    raise ApprovalRequired("missing HIGH-risk approval id")
                failure, resolver = await self._consume_approval(
                    action=ACTION_NOTE, resource=resource, arguments=arguments,
                    digest=proposal.arguments_digest, approval_id=approval_id,
                    correlation_id=proposal_id,
                )
                if failure:
                    await self._deny(proposal, failure)
                    raise ApprovalRequired(failure)
                gate = f"approval:{resolver}"
            duplicate_proposal, duplicate_memory = await self._find_duplicate(
                current, exclude_proposal=proposal_id
            )
            if duplicate_proposal or duplicate_memory:
                reason = "scope changed: duplicate appeared before acceptance"
                proposal.status = "duplicate"
                proposal.reason = reason
                proposal.resolved_at = utcnow()
                await self._save(f"proposal:{proposal_id}", proposal.model_dump())
                await self._audit(ACTION_NOTE, resource, "duplicate", reason,
                                  {"proposal_id": proposal_id, "duplicate_memory_id": duplicate_memory},
                                  risk=proposal.risk, correlation_id=proposal_id)
                await self._emit("learning.duplicate", proposal_id=proposal_id, reason=reason)
                raise LearningDenied(reason)
        except LearningDenied as exc:
            raise exc

        metadata: dict[str, Any] = {
            "proposal_id": proposal_id,
            "policy": gate,
            "unattended": bool(proposal.unattended),
            "source_kind": current.source_kind,
            "evidence": [ref.model_dump(exclude_none=True) for ref in list(current.evidence)[:MAX_EVIDENCE]],
            "arguments_digest": proposal.arguments_digest,
            "approval_id": approval_id if gate.startswith("approval:") else None,
            "learned_by": self.actor,
        }
        if proposal.batch_size > 1 or proposal.batch_fingerprint:
            metadata["batch_size"] = proposal.batch_size
            metadata["batch_fingerprint"] = proposal.batch_fingerprint
        if current.conversation_id:
            metadata["conversation_id"] = current.conversation_id
        record = MemoryRecord(
            category=current.category,
            kind=current.kind,
            content=current.content,
            metadata=self._clean(metadata),
            source=f"learning:{proposal_id}",
            confidence=current.confidence,
            tags=current.tags,
            supersedes=list(current.supersedes),
        )
        try:
            saved = await self.store.add(record)
        except Exception as exc:  # noqa: BLE001 - a failed write must be recorded truthfully
            proposal.status = "failed"
            proposal.reason = f"memory write failed: {type(exc).__name__}"
            proposal.resolved_at = utcnow()
            await self._save(f"proposal:{proposal_id}", proposal.model_dump())
            await self._audit(ACTION_NOTE, resource, "failed", proposal.reason,
                              {"proposal_id": proposal_id}, risk=proposal.risk,
                              correlation_id=proposal_id)
            await self._emit("learning.failed", proposal_id=proposal_id, reason=proposal.reason)
            raise LearningDenied(proposal.reason) from exc

        for old_id in current.supersedes:
            old = await self.store.get(old_id)
            if old is None:
                continue
            old_metadata = dict(old.metadata)
            old_metadata["status"] = "superseded"
            old_metadata["superseded_by"] = saved.id
            await self.store.update(
                old_id, metadata=old_metadata, action="superseded",
                reason=f"superseded by learning proposal {proposal_id}", actor=self.actor,
            )
            for prior in await self.proposals(limit=MAX_PROPOSAL_ROWS):
                if prior.get("memory_id") != old_id or prior.get("status") == "superseded":
                    continue
                prior["status"] = "superseded"
                prior["superseded_by"] = proposal_id
                prior["resolved_at"] = utcnow()
                await self._save(f"proposal:{prior['id']}", prior)

        proposal.status = "accepted"
        proposal.memory_id = saved.id
        proposal.resolved_at = utcnow()
        proposal.reason = f"written under {gate}"
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        index = await self._index()
        seen = list(index.get("seen") or [])
        for ref in current.evidence:
            marker = f"{ref.kind}:{ref.id}"
            if marker not in seen:
                seen.append(marker)
        index["seen"] = seen[-_SEEN_LIMIT:]
        memories = dict(index.get("memories") or {})
        memories[_hash(normalize_content(current.content))] = saved.id
        # A coalesced batch writes ONE memory. Its constituent hashes point at
        # that same record so re-proposing any of them is a truthful duplicate,
        # not a second note.
        for item in list(proposal.batch_hashes)[:MAX_BATCH_DRAFTS]:
            memories.setdefault(item, saved.id)
        index["memories"] = memories
        await self._save_index(index)
        await self._audit(
            ACTION_NOTE, resource, "allowed", "memory note accepted",
            {"proposal_id": proposal_id, "memory_id": saved.id, "gate": gate,
             "arguments_digest": proposal.arguments_digest, "supersedes": current.supersedes},
            risk=proposal.risk, correlation_id=proposal_id,
        )
        await self._emit(
            "learning.accepted", proposal_id=proposal_id, memory_id=saved.id,
            gate=gate, category=_category_value(saved.category),
            kind=_kind_value(saved.kind), supersedes=current.supersedes,
        )
        return saved.model_dump(mode="json")

    async def _deny(self, proposal: LearningProposal, reason: str) -> None:
        proposal.status = "denied"
        proposal.reason = str(reason)[:280]
        proposal.resolved_at = utcnow()
        await self._save(f"proposal:{proposal.id}", proposal.model_dump())
        await self._audit(
            ACTION_NOTE, proposal.resource, "denied", reason,
            {"proposal_id": proposal.id}, risk=proposal.risk, correlation_id=proposal.id,
        )
        await self._emit("learning.denied", proposal_id=proposal.id, reason=reason)

    # --------------------------------------------------------------- lifecycle
    async def reject(self, proposal_id: str, reason: str = "") -> dict[str, Any]:
        """Refuse a proposal. Never deletes or mutates memory."""
        proposal = await self._load(f"proposal:{proposal_id}", LearningProposal)
        if proposal is None:
            raise KeyError(f"unknown learning proposal: {proposal_id}")
        if proposal.status not in {"proposed", "awaiting_approval"}:
            raise LearningDenied(f"proposal is {proposal.status}, not rejectable")
        proposal.status = "rejected"
        proposal.reason = str(reason or "rejected by operator")[:280]
        proposal.resolved_at = utcnow()
        await self._save(f"proposal:{proposal_id}", proposal.model_dump())
        await self._audit(
            ACTION_NOTE, proposal.resource, "denied", proposal.reason,
            {"proposal_id": proposal_id}, risk=proposal.risk, correlation_id=proposal_id,
        )
        await self._emit("learning.rejected", proposal_id=proposal_id, reason=proposal.reason)
        return proposal.model_dump()

    async def supersede(
        self,
        proposal_id: str,
        content: str,
        *,
        reason: str = "",
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Create a replacement proposal for an accepted note.

        Superseding always needs an approval: it changes the meaning of an
        already-accepted memory even when a trusted auto-note policy is enabled.
        """
        old = await self._load(f"proposal:{proposal_id}", LearningProposal)
        if old is None or old.status != "accepted" or not old.memory_id:
            raise LearningDenied("only an accepted proposal with a memory can be superseded")
        if await self.store.get(old.memory_id) is None:
            raise LearningDenied("the memory being superseded no longer exists")
        draft = NoteDraft(
            content=content,
            category=old.draft.category,
            kind=MemoryKind.UNVERIFIED,
            confidence=(
                float(old.draft.confidence) if confidence is None else float(confidence)
            ),
            source_kind=old.draft.source_kind,
            conversation_id=old.draft.conversation_id,
            tags=list(old.draft.tags),
            supersedes=[old.memory_id],
            reason=reason or f"supersede {proposal_id}",
        )
        proposal = await self.propose(draft)
        if proposal["status"] == "duplicate":
            raise LearningDenied("replacement is a duplicate of an existing note")
        if proposal["policy"] != "approval":
            raise LearningDenied("supersede must always be approval-gated")
        return proposal

    # ----------------------------------------------------------- consolidation
    async def plan_consolidation(self, category: str | None = None) -> dict[str, Any]:
        """Build an inert, digest-bound duplicate-merge plan (no mutation)."""
        found = await self.consolidator.find_duplicates(category)
        if len(found) > _MAX_CLUSTERS:
            # Consolidator.consolidate() merges every cluster it finds. A
            # truncated plan could never match that scope at apply time, so
            # refuse to issue an approval we know cannot be honoured.
            raise LearningDenied(
                f"too many duplicate clusters ({len(found)}); narrow the category"
            )
        oversized = [len(cluster) for cluster in found if len(cluster) > MAX_CLUSTER_MEMBERS]
        if oversized:
            # Same scope rule for cluster size: never issue a digest that hides
            # members the consolidator would merge at apply time.
            raise LearningDenied(
                f"duplicate cluster too large ({max(oversized)} members); narrow the category"
            )
        clusters = [
            sorted(record.id for record in cluster)
            for cluster in found
            if len(cluster) > 1
        ]
        category = str(category) if category else None
        plan_id = new_id("lcp")
        resource = f"learning/consolidate/{_safe_ref(category or 'all', 'all')}"
        digest = arguments_digest({"plan_id": plan_id, "clusters": clusters, "category": category}) or ""
        if not digest:
            raise LearningDenied("could not bind consolidation arguments")
        plan = ConsolidationPlan(
            id=plan_id, category=category, clusters=clusters, resource=resource,
            arguments_digest=digest,
        )
        if not clusters:
            plan.status = "empty"
        await self._save(f"consolidation:{plan_id}", plan.model_dump())
        await self._prune_kv(self._key("consolidation:"), MAX_PLAN_ROWS)
        if clusters:
            decision = await self.permissions.authorize(
                self.actor, ACTION_CONSOLIDATE, resource, RiskLevel.HIGH,
                arguments={"plan_id": plan_id, "clusters": clusters, "category": category},
                arguments_digest=digest, plan_id=plan_id, cluster_count=len(clusters),
            )
            plan.approval_id = decision.approval_id
            if decision.allowed and not decision.requires_approval:
                plan.status = "denied"
                plan.reason = "HIGH-risk approval gate disabled; refusing to consolidate"
            await self._save(f"consolidation:{plan_id}", plan.model_dump())
            await self._prune_kv(self._key("consolidation:"), MAX_PLAN_ROWS)
        decision_name = "pending" if plan.status == "awaiting_approval" else (
            "denied" if plan.status == "denied" else "allowed"
        )
        await self._audit(
            ACTION_CONSOLIDATE, resource, decision_name,
            plan.reason or f"consolidation plan with {len(clusters)} clusters",
            {"plan_id": plan_id, "arguments_digest": digest}, correlation_id=plan_id,
        )
        await self._emit(
            "learning.consolidation_plan", plan_id=plan_id, status=plan.status,
            cluster_count=len(clusters), approval_id=plan.approval_id,
        )
        return plan.model_dump()

    async def _plan_clusters_unchanged(self, plan: ConsolidationPlan) -> bool:
        current = await self.consolidator.find_duplicates(plan.category)
        actual = {tuple(sorted(record.id for record in cluster)) for cluster in current}
        planned = {tuple(cluster) for cluster in plan.clusters}
        return actual == planned

    async def consolidate(self, plan_id: str, approval_id: str | None = None) -> dict[str, Any]:
        """Apply an approved plan through ``Consolidator`` after scope recheck."""
        plan = await self._load(f"consolidation:{plan_id}", ConsolidationPlan)
        if plan is None:
            raise KeyError(f"unknown consolidation plan: {plan_id}")
        if plan.status == "empty":
            return {"plan_id": plan_id, "status": "empty", "removed": 0, "survivors": []}
        if plan.status != "awaiting_approval":
            raise LearningDenied(f"consolidation plan is {plan.status}")
        if not await self._plan_clusters_unchanged(plan):
            reason = "consolidation scope changed since the plan was approved"
            plan.status = "failed"
            plan.reason = reason
            await self._save(f"consolidation:{plan_id}", plan.model_dump())
            await self._audit(ACTION_CONSOLIDATE, plan.resource, "denied", reason,
                              {"plan_id": plan_id}, correlation_id=plan_id)
            raise LearningDenied(reason)
        arguments = {"plan_id": plan_id, "clusters": plan.clusters, "category": plan.category}
        failure, resolver = await self._consume_approval(
            action=ACTION_CONSOLIDATE, resource=plan.resource, arguments=arguments,
            digest=plan.arguments_digest, approval_id=approval_id, correlation_id=plan_id,
        )
        if failure:
            await self._audit(ACTION_CONSOLIDATE, plan.resource, "denied", failure,
                              {"plan_id": plan_id}, correlation_id=plan_id)
            raise ApprovalRequired(failure)
        if not await self._plan_clusters_unchanged(plan):
            reason = "consolidation scope changed during approval consumption"
            plan.status = "failed"
            plan.reason = reason
            await self._save(f"consolidation:{plan_id}", plan.model_dump())
            raise LearningDenied(reason)
        result = await self.consolidator.consolidate(plan.category)
        plan.status = "applied"
        plan.resolved_at = utcnow()
        await self._save(f"consolidation:{plan_id}", plan.model_dump())
        await self._audit(
            ACTION_CONSOLIDATE, plan.resource, "allowed", "consolidation applied",
            {"plan_id": plan_id, "resolver": resolver,
             "removed": list(result.removed_ids),
             "survivors": [record.id for record in result]},
            correlation_id=plan_id,
        )
        await self._emit(
            "learning.consolidated", plan_id=plan_id,
            removed=list(result.removed_ids),
            survivors=[record.id for record in result],
        )
        return {
            "plan_id": plan_id, "status": "applied",
            "removed": list(result.removed_ids),
            "survivors": [record.id for record in result],
        }

    # ------------------------------------------------------------ observation
    async def scan(
        self,
        *,
        conversation_id: str | None = None,
        limit: int = 20,
        include_tools: bool = True,
        include_runs: bool = True,
    ) -> list[dict[str, Any]]:
        """Derive proposals from durable transcripts, tool calls and runs.

        Caller-supplied ``limit`` bounds each source; the whole scan is capped
        at ``_MAX_CANDIDATES`` proposals. Only user text can become a stated
        fact; tool/run observations are conservative procedural notes built
        from counts and ids, never from arguments, errors or outputs.
        """
        limit = max(1, min(int(limit), 50))
        drafts: list[NoteDraft] = []
        drafts.extend(await self._message_drafts(conversation_id, limit))
        if include_tools:
            drafts.extend(await self._tool_drafts(limit))
        if include_runs:
            drafts.extend(await self._run_drafts(limit))
        index = await self._index()
        seen = set(index.get("seen") or ())
        result: list[dict[str, Any]] = []
        for draft in drafts[:_MAX_CANDIDATES]:
            if any(f"{ref.kind}:{ref.id}" in seen for ref in draft.evidence):
                continue
            try:
                result.append(await self.propose(draft))
            except LearningDenied as exc:  # one hostile turn must not abort a scan
                _LOG.warning("learning candidate skipped", extra={"reason": str(exc)})
        return result

    async def _message_drafts(self, conversation_id: str | None, limit: int) -> list[NoteDraft]:
        sql = (
            "SELECT id, conversation_id, role, content, created_at FROM messages "
            "WHERE role='user'"
        )
        params: list[Any] = []
        if conversation_id is not None:
            sql += " AND conversation_id=?"
            params.append(str(conversation_id))
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(min(limit, _MAX_MESSAGES))
        drafts: list[NoteDraft] = []
        for row in await db.fetchall(sql, params):
            raw = str(row["content"] or "")
            try:
                self._sanitize(raw)
            except LearningDenied:
                continue  # never persist, and never try to salvage, hostile text
            for pattern, category, confidence in _USER_STATEMENTS:
                match = pattern.match(raw)
                if not match:
                    continue
                try:
                    content = self._sanitize(match.group(1))
                except LearningDenied:
                    continue
                ref = EvidenceRef(
                    id=f"msg:{row['id']}", kind="message", hash=_hash(raw),
                    conversation_id=_safe_ref(row["conversation_id"], "unknown"),
                    created_at=str(row["created_at"])[:40],
                )
                drafts.append(
                    NoteDraft(
                        content=content, category=category, kind=MemoryKind.UNVERIFIED,
                        confidence=confidence, source_kind="user",
                        conversation_id=_safe_ref(row["conversation_id"], "unknown"),
                        evidence=[ref], reason="explicit durable user statement",
                    )
                )
                break
        return drafts

    async def _tool_drafts(self, limit: int) -> list[NoteDraft]:
        rows = await db.fetchall(
            "SELECT id, tool, args, success, created_at FROM tool_calls "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (min(limit * 10, _MAX_TOOLS),),
        )
        successes: dict[str, list[Any]] = {}
        failures: set[str] = set()
        for row in rows:
            tool = str(row["tool"] or "")
            if not _SAFE_NAME.fullmatch(tool):
                continue
            # Dry runs are previews: never record one as a successful fix.
            dry_run = False
            try:
                args = json.loads(row["args"] or "{}")
                dry_run = isinstance(args, Mapping) and args.get("dry_run") is True
            except (TypeError, ValueError):
                pass
            if int(row["success"] or 0) == 1 and not dry_run:
                successes.setdefault(tool, []).append(row)
            else:
                failures.add(tool)
        drafts: list[NoteDraft] = []
        for tool, calls in successes.items():
            if tool in failures or len(calls) < self.min_occurrences:
                continue
            refs = [
                EvidenceRef(id=f"call:{row['id']}", kind="tool_call",
                            created_at=str(row["created_at"])[:40])
                for row in calls[:MAX_EVIDENCE]
            ]
            drafts.append(
                NoteDraft(
                    content=(
                        f"Tool {tool} completed successfully {len(calls)} times in the recent "
                        "trajectory with no recorded failures"
                    ),
                    category=MemoryCategory.PROCEDURAL, kind=MemoryKind.UNVERIFIED,
                    confidence=0.5, source_kind="tool", evidence=refs,
                    reason="bounded successful tool trajectory",
                )
            )
        return drafts

    async def _run_drafts(self, limit: int) -> list[NoteDraft]:
        rows = await db.fetchall(
            "SELECT id, agent_id, status, started_at FROM agent_runs "
            "WHERE status IN ('completed','success') "
            "ORDER BY started_at DESC, id DESC LIMIT ?",
            (min(limit * 10, _MAX_RUNS),),
        )
        by_agent: dict[str, list[Any]] = {}
        for row in rows:
            agent = str(row["agent_id"] or "")
            if _SAFE_NAME.fullmatch(agent):
                by_agent.setdefault(agent, []).append(row)
        drafts: list[NoteDraft] = []
        for agent, runs in by_agent.items():
            if len(runs) < self.min_occurrences:
                continue
            refs = [
                EvidenceRef(id=f"run:{row['id']}", kind="run",
                            created_at=str(row["started_at"])[:40])
                for row in runs[:MAX_EVIDENCE]
            ]
            drafts.append(
                NoteDraft(
                    content=f"Agent {agent} completed {len(runs)} runs successfully",
                    category=MemoryCategory.PROCEDURAL, kind=MemoryKind.UNVERIFIED,
                    confidence=0.5, source_kind="run", evidence=refs,
                    reason="bounded successful agent run trajectory",
                )
            )
        return drafts

    # ------------------------------------------------------------- inspection
    async def proposals(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT key,value FROM kv WHERE key LIKE ? ORDER BY updated_at DESC",
            (f"{_PREFIX}proposal:%",),
        )
        result: list[dict[str, Any]] = []
        for row in rows[: min(max(0, int(limit)), MAX_PROPOSAL_ROWS)]:
            try:
                result.append(json.loads(row["value"]))
            except (TypeError, ValueError):
                continue
        return result

    async def proposal(self, proposal_id: str) -> dict[str, Any] | None:
        value = await self._load(f"proposal:{proposal_id}")
        return value if isinstance(value, dict) else None

    async def audit_trail(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT * FROM audit_log WHERE actor=? ORDER BY id DESC LIMIT ?",
            (self.actor, min(max(0, int(limit)), MAX_PROPOSAL_ROWS)),
        )
        return [dict(row) for row in rows]

    def governance(self) -> dict[str, Any]:
        """Describe the active write gates for operators and tests."""
        return {
            "actor": self.actor,
            "auto_notes_enabled": bool(self.auto_policy.enabled),
            "trusted_sources": list(self.auto_policy.trusted_sources),
            "auto_categories": list(self.auto_policy.allowed_categories),
            "unattended_auto_writes": bool(self.auto_policy.allow_unattended_writes),
            "auto_notes_approval_bound": not bool(self.auto_policy.allow_unattended_writes),
            "chat_fact_promotion_requires_approval": True,
            "supersede_always_requires_approval": True,
            "consolidation_always_requires_approval": True,
            "approval_action": ACTION_NOTE,
            "consolidation_action": ACTION_CONSOLIDATE,
            "max_content_chars": MAX_CONTENT_CHARS,
            "max_pending_proposals": self.max_pending_proposals,
            "pending_ttl_seconds": self.pending_ttl_seconds,
            "batch_proposals_enabled": True,
        }


__all__ = [
    "ACTION_CONSOLIDATE",
    "ACTION_NOTE",
    "ACTOR",
    "DEFAULT_PENDING_TTL_SECONDS",
    "MAX_BATCH_DRAFTS",
    "MAX_CLUSTER_MEMBERS",
    "MAX_CONTENT_CHARS",
    "MAX_EVIDENCE",
    "MAX_MEMORY_INDEX_ROWS",
    "MAX_PENDING_PROPOSALS",
    "MAX_PLAN_ROWS",
    "MAX_PROPOSAL_ROWS",
    "MAX_SUPERSEDES",
    "ApprovalRequired",
    "AutoNotePolicy",
    "ConsolidationPlan",
    "EvidenceRef",
    "LearningDenied",
    "LearningEngine",
    "LearningProposal",
    "NoteDraft",
]
