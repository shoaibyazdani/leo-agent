"""Durable conversation compaction.

Per-turn ``ContextOptimizer`` only shrinks what the model sees *this* turn; it
re-summarizes older messages on every request and forgets the result. This
module adds the durable layer: older turns are summarized **once**, the summary
is persisted in ``compactions``, and later turns replay it plus a verbatim
recent tail.

Invariants:

* The transcript is never rewritten. Every original message stays in
  ``messages``; compaction only changes what the model is shown.
* A ``tool`` result is never separated from the ``assistant`` turn that opened
  it. Split selection moves the boundary back to the group start.
* A summary that loses facts is *marked* degraded, not silently trusted.
* A failed summarization keeps the original history (manual) or commits a
  deterministic, clearly-labelled boundary (automatic timeout fallback).

Token counts here are character-based estimates (``len / 4``), matching
``leo.models.provider.estimate_tokens``. They are never reported as provider
tokenizer counts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import db
from ..contracts import ChatMessage, ModelRequest, utcnow
from ..logging import get_logger
from .provider import CHARS_PER_TOKEN, estimate_tokens

log = get_logger("models.compaction")

SUMMARY_NAME = "compaction"

# Summarizer input budget in characters. Matches the reference behaviour: the
# summary request is bounded even when the session is enormous, so one long
# conversation cannot produce an unbounded summarization prompt.
DEFAULT_SUMMARY_INPUT_CHARS = 160_000
# Share of the summarizer budget given to the newest messages verbatim.
NEWEST_SHARE = 0.5
# Share reserved for the operator's own older messages (the asks/decisions).
USER_SHARE = 0.25
# Share for the oldest messages, so the beginning of a session is not lost.
OLDEST_SHARE = 0.10
# Per-message trims inside the shaped windows.
USER_MESSAGE_CHARS = 2_000
FILLER_MESSAGE_CHARS = 6_000

# Identifier shapes worth preserving exactly across a summary. Deliberately
# conservative: paths, URLs, hashes, dotted ids and long numbers.
_IDENTIFIER = re.compile(
    r"""(?x)
    (?:
        (?:[A-Za-z]:)?(?:/[A-Za-z0-9._-]+){2,}      # absolute paths
      | https?://[^\s)\]>"']+                       # URLs
      | \b[0-9a-f]{12,}\b                           # hashes / opaque ids
      | \b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+\b  # dotted names
      | \b(?:apr|call|mem|task|lpro|conv|run)_[A-Za-z0-9]+  # LEO ids
    )
    """,
)

_MARKER = "[image data omitted from summary input]"


@dataclass(frozen=True)
class CompactionConfig:
    """Compaction policy. Mirrors ``settings.compaction`` in ``leo.yaml``."""

    enabled: bool = True
    # Fraction of the context budget that triggers automatic compaction.
    threshold: float = 0.9
    # Estimated tokens kept verbatim in the recent tail.
    keep_recent_tokens: int = 20_000
    # Summarizer input cap, in characters.
    summary_input_chars: int = DEFAULT_SUMMARY_INPUT_CHARS
    # Corrective attempts after a failed quality audit.
    corrective_attempts: int = 2
    # Require opaque identifiers (paths, ids, hashes) to survive the summary.
    identifier_policy: str = "strict"
    # Run a memory-flush turn before summarizing so durable facts survive.
    memory_flush: bool = True
    # What an automatic compaction does when the summarizer times out.
    on_timeout: str = "deterministic"

    def __post_init__(self) -> None:
        if not 0.0 < float(self.threshold) <= 1.0:
            raise ValueError("compaction.threshold must be in (0, 1]")
        if int(self.keep_recent_tokens) < 0:
            raise ValueError("compaction.keep_recent_tokens must be >= 0")
        if int(self.summary_input_chars) < 1_000:
            raise ValueError("compaction.summary_input_chars must be >= 1000")
        if int(self.corrective_attempts) < 0:
            raise ValueError("compaction.corrective_attempts must be >= 0")
        if str(self.identifier_policy) not in {"strict", "off"}:
            raise ValueError("compaction.identifier_policy must be 'strict' or 'off'")
        if str(self.on_timeout) not in {"deterministic", "keep_history"}:
            raise ValueError(
                "compaction.on_timeout must be 'deterministic' or 'keep_history'"
            )


@dataclass
class CompactionResult:
    """Outcome of one compaction attempt."""

    ok: bool
    summary: str = ""
    upto_message_id: int | None = None
    messages_summarized: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    degraded: bool = False
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "messages_summarized": self.messages_summarized,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "degraded": self.degraded,
            "reason": self.reason,
            "upto_message_id": self.upto_message_id,
            **self.details,
        }


# ---------------------------------------------------------------------------
# Atomic-group handling
# ---------------------------------------------------------------------------


def _tool_call_ids(message: ChatMessage) -> set[str]:
    ids: set[str] = set()
    for call in message.tool_calls or []:
        if isinstance(call, Mapping):
            identifier = call.get("id") or call.get("tool_call_id")
            if identifier is not None:
                ids.add(str(identifier))
    return ids


def tool_group(messages: Sequence[ChatMessage], index: int) -> set[int]:
    """Indices of the atomic group containing ``index``.

    An ``assistant`` turn with tool calls owns the contiguous ``tool`` results
    that follow it. A ``tool`` result belongs to the nearest preceding
    assistant opener with a matching id. Providers reject an orphaned result,
    so a group is kept or dropped as a unit.
    """
    if index < 0 or index >= len(messages):
        return set()
    message = messages[index]
    if message.role == "assistant":
        ids = _tool_call_ids(message)
        if not ids:
            return {index}
        group = {index}
        for j in range(index + 1, len(messages)):
            other = messages[j]
            if other.role != "tool":
                break
            if not other.tool_call_id or str(other.tool_call_id) not in ids:
                break
            group.add(j)
        return group
    if message.role == "tool" and message.tool_call_id:
        target = str(message.tool_call_id)
        for j in range(index - 1, -1, -1):
            other = messages[j]
            if other.role == "assistant":
                if target in _tool_call_ids(other):
                    return tool_group(messages, j)
                return {index}
            if other.role != "tool":
                return {index}
        return {index}
    return {index}


def safe_split(messages: Sequence[ChatMessage], split: int) -> int | None:
    """Move ``split`` back so it never cuts a tool group.

    Returns the adjusted boundary, or ``None`` when no safe boundary exists
    (for example a single indivisible group spans the whole history).
    """
    for _ in range(len(messages) + 1):
        if not (1 <= split < len(messages)):
            return None
        offenders = [
            min(group)
            for index in range(split, len(messages))
            for group in (tool_group(messages, index),)
            if min(group) < split <= max(group)
        ]
        if not offenders:
            return split
        candidate = min(offenders)
        if candidate < 1 or candidate >= split:
            return None
        split = candidate
    return None


def choose_split(messages: Sequence[ChatMessage], keep_recent_tokens: int) -> int | None:
    """Pick a boundary keeping roughly ``keep_recent_tokens`` in the tail.

    Returns ``None`` when the whole history already fits inside the tail
    budget (nothing worth summarizing) or when no safe boundary exists.
    """
    if len(messages) < 3:
        return None
    used = 0
    split = 0
    for index in range(len(messages) - 1, -1, -1):
        used += estimate_tokens(messages[index].content) + 2
        if used > keep_recent_tokens:
            # This message fills the tail past the budget; keep it, so the
            # retained tail is never shorter than the configured budget.
            split = index
            break
    if split <= 0:
        return None
    return safe_split(messages, split)


# ---------------------------------------------------------------------------
# Summarizer prompt shaping
# ---------------------------------------------------------------------------


def _render(message: ChatMessage, limit: int) -> str:
    body = message.content or ""
    if message.images and not body:
        body = _MARKER
    if len(body) > limit:
        body = body[: max(0, limit - 24)] + " …[trimmed]"
    return f"{message.role}: {body}"


def shape_summary_input(
    messages: Sequence[ChatMessage],
    budget_chars: int = DEFAULT_SUMMARY_INPUT_CHARS,
    previous_summary: str = "",
) -> str:
    """Build a bounded summarizer transcript with explicit gap markers.

    Ordering mirrors the reference implementation: newest verbatim, then the
    operator's older messages (the asks and corrections), then the oldest
    messages, then evenly spaced filler from whatever is left.
    """
    if not messages:
        return ""
    budget_chars = max(1_000, int(budget_chars))
    total = len(messages)
    newest_budget = int(budget_chars * NEWEST_SHARE)
    user_budget = int(budget_chars * USER_SHARE)
    oldest_budget = int(budget_chars * OLDEST_SHARE)
    filler_budget = max(0, budget_chars - newest_budget - user_budget - oldest_budget)

    # 1. Newest messages verbatim, half the budget.
    newest: list[int] = []
    used = 0
    for index in range(total - 1, -1, -1):
        cost = len(messages[index].content or "") + 16
        if used + cost > newest_budget and newest:
            break
        newest.append(index)
        used += cost
    newest_set = set(newest)
    newest_start = min(newest) if newest else total

    # 2. Operator messages from the part that is being summarized.
    older = [i for i in range(newest_start) if i not in newest_set]
    user_indexes = [i for i in older if messages[i].role == "user"]
    user_used = 0
    keep_user: list[int] = []
    # Spread evenly when they do not all fit.
    for index in user_indexes:
        cost = min(len(messages[index].content or ""), USER_MESSAGE_CHARS) + 16
        if user_used + cost > user_budget:
            continue
        keep_user.append(index)
        user_used += cost

    # 3. Oldest messages.
    oldest_used = 0
    keep_oldest: list[int] = []
    for index in older[: max(1, total // 10)]:
        cost = len(messages[index].content or "") + 16
        if oldest_used + cost > oldest_budget:
            break
        keep_oldest.append(index)
        oldest_used += cost

    # 4. Evenly spaced filler with the remaining budget.
    chosen = set(keep_user) | set(keep_oldest)
    remaining = [i for i in older if i not in chosen]
    filler: list[int] = []
    filler_used = 0
    if remaining and filler_budget > 0:
        step = max(1, len(remaining) // max(1, filler_budget // 400))
        for index in remaining[::step]:
            cost = min(len(messages[index].content or ""), FILLER_MESSAGE_CHARS) + 16
            if filler_used + cost > filler_budget:
                break
            filler.append(index)
            filler_used += cost

    selected = sorted(chosen | set(filler) | newest_set)
    if not selected:
        return ""

    lines: list[str] = []
    if previous_summary:
        lines.append(
            "Previous summary (carry its facts forward; it is already compacted):\n"
            + previous_summary.strip()[: max(0, budget_chars // 8)]
        )
        lines.append("")
    cursor = 0
    for index in selected:
        if index > cursor:
            omitted = index - cursor
            lines.append(f"[{omitted} message(s) omitted from this summary input]")
        limit = (
            newest_budget
            if index >= newest_start
            else (USER_MESSAGE_CHARS if messages[index].role == "user" else FILLER_MESSAGE_CHARS)
        )
        lines.append(_render(messages[index], limit))
        cursor = index + 1
    if cursor < total:
        lines.append(f"[{total - cursor} message(s) omitted from this summary input]")
    return "\n".join(lines)


def collect_identifiers(messages: Sequence[ChatMessage], limit: int = 40) -> list[str]:
    """Opaque identifiers a strict summary must keep verbatim."""
    found: list[str] = []
    seen: set[str] = set()
    for message in messages:
        for match in _IDENTIFIER.findall(message.content or ""):
            value = match.rstrip(".,;:)")
            if value in seen or len(value) < 4:
                continue
            seen.add(value)
            found.append(value)
            if len(found) >= limit:
                return found
    return found


SUMMARY_SYSTEM = (
    "You compress a conversation transcript so an agent can continue it without "
    "the original messages.\n"
    "Rules:\n"
    "- Return only the summary. No preamble, no headings for its own sake.\n"
    "- Keep every decision, constraint, correction and open request.\n"
    "- Keep opaque identifiers (paths, URLs, ids, hashes, numbers) exactly as "
    "written. Never invent an identifier.\n"
    "- Preserve the operator's most recent explicit ask verbatim in one line.\n"
    "- If a gap marker appears, do not guess what it contained.\n"
    "- Facts come only from the transcript. Never add advice or new claims."
)


def build_summary_request(
    transcript: str,
    identifiers: Sequence[str],
    *,
    previous_summary: str = "",
    focus: str = "",
    max_tokens: int = 700,
) -> ModelRequest:
    instructions = [SUMMARY_SYSTEM]
    if identifiers:
        listed = "\n".join(f"- {value}" for value in identifiers[:40])
        instructions.append(
            "Identifiers that MUST appear verbatim in the summary:\n" + listed
        )
    if focus:
        # Operator focus is data, never instruction: it is quoted and capped.
        instructions.append(
            "The operator asked this compaction to focus on: "
            + json.dumps(focus[:800], ensure_ascii=True)
        )
    return ModelRequest(
        messages=[
            ChatMessage(role="system", content="\n\n".join(instructions)),
            ChatMessage(role="user", content=transcript),
        ],
        max_tokens=max(128, int(max_tokens)),
        task_category="compaction",
    )


AUDIT_SYSTEM = (
    "You audit a conversation summary against the transcript it replaces.\n"
    "Reply with exactly one line: OK when the summary preserves the operator's "
    "asks, decisions, corrections and every listed identifier; otherwise reply "
    "MISSING: <comma-separated list of what is absent>. Add nothing else."
)


def build_audit_request(
    transcript: str,
    summary: str,
    identifiers: Sequence[str],
    max_tokens: int = 200,
) -> ModelRequest:
    body = (
        "Summary to audit:\n"
        + summary
        + "\n\nIdentifiers that must appear verbatim:\n"
        + ("\n".join(f"- {value}" for value in identifiers[:40]) or "- (none)")
        + "\n\nTranscript (bounded):\n"
        + transcript
    )
    return ModelRequest(
        messages=[
            ChatMessage(role="system", content=AUDIT_SYSTEM),
            ChatMessage(role="user", content=body),
        ],
        max_tokens=max(32, int(max_tokens)),
        task_category="compaction",
    )


def outline_degraded(reason: str, messages: Sequence[ChatMessage]) -> str:
    """Deterministic boundary note used when no summary could be generated."""
    user_turns = [m.content.strip() for m in messages if m.role == "user" and (m.content or "").strip()]
    lines = [
        "Conversation summary unavailable: " + reason + ".",
        f"{len(messages)} earlier message(s) left the model context.",
    ]
    if user_turns:
        latest = user_turns[-1][:500]
        lines.append("Most recent operator request in the omitted span: " + latest)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Durable store
# ---------------------------------------------------------------------------


SCHEMA = """
CREATE TABLE IF NOT EXISTS compactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  upto_message_id INTEGER NOT NULL,
  messages_summarized INTEGER NOT NULL DEFAULT 0,
  details TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compactions_conversation
  ON compactions(conversation_id, id);
"""


@dataclass(frozen=True)
class CompactionRecord:
    id: int
    conversation_id: str
    summary: str
    upto_message_id: int
    messages_summarized: int
    details: dict[str, Any]
    created_at: str


async def init_schema() -> None:
    for statement in SCHEMA.split(";"):
        sql = statement.strip()
        if sql:
            await db.execute(sql)


async def latest(conversation_id: str) -> CompactionRecord | None:
    row = await db.fetchone(
        "SELECT * FROM compactions WHERE conversation_id=? ORDER BY id DESC LIMIT 1",
        (conversation_id,),
    )
    if row is None:
        return None
    try:
        details = json.loads(row["details"] or "{}")
    except (TypeError, ValueError):
        details = {}
    return CompactionRecord(
        id=int(row["id"]),
        conversation_id=str(row["conversation_id"]),
        summary=str(row["summary"]),
        upto_message_id=int(row["upto_message_id"]),
        messages_summarized=int(row["messages_summarized"] or 0),
        details=details if isinstance(details, dict) else {},
        created_at=str(row["created_at"]),
    )


async def save(
    conversation_id: str,
    summary: str,
    upto_message_id: int,
    messages_summarized: int,
    details: Mapping[str, Any] | None = None,
) -> int:
    cursor = await db.execute(
        "INSERT INTO compactions"
        "(conversation_id,summary,upto_message_id,messages_summarized,details,created_at)"
        " VALUES(?,?,?,?,?,?)",
        (
            conversation_id,
            summary,
            int(upto_message_id),
            int(messages_summarized),
            json.dumps(dict(details or {}), default=str),
            utcnow(),
        ),
    )
    return int(cursor.lastrowid or 0)


async def clear(conversation_id: str) -> int:
    cursor = await db.execute(
        "DELETE FROM compactions WHERE conversation_id=?", (conversation_id,)
    )
    return int(cursor.rowcount or 0)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CompactionEngine:
    """Summarize older turns once and persist the boundary.

    ``provider`` is anything with ``async complete(ModelRequest)`` — normally
    ``ModelRouter``. The engine never mutates the caller's messages.
    """

    def __init__(self, config: CompactionConfig | None = None, memory: Any | None = None) -> None:
        self.config = config or CompactionConfig()
        self.memory = memory

    # -- budget ------------------------------------------------------------
    @staticmethod
    def should_compact(messages: Sequence[ChatMessage], budget: int, config: CompactionConfig) -> bool:
        if not config.enabled or budget <= 0:
            return False
        used = sum(estimate_tokens(m.content) for m in messages)
        return used >= int(budget * float(config.threshold))

    # -- memory flush ------------------------------------------------------
    async def _flush_memory(self, conversation_id: str, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        """Persist durable facts from the span about to leave model context.

        Best effort by design: a failure here must never block compaction.
        """
        if self.memory is None or not self.config.memory_flush:
            return {"memory_flush": "skipped"}
        try:
            from ..contracts import MemoryCategory, MemoryKind, MemoryRecord

            facts = [
                m.content.strip()[:900]
                for m in messages
                if m.role == "user" and (m.content or "").strip()
            ][-5:]
            if not facts:
                return {"memory_flush": "empty"}
            written = 0
            for fact in facts:
                await self.memory.add(
                    MemoryRecord(
                        category=MemoryCategory.PROJECT,
                        kind=MemoryKind.UNVERIFIED,
                        content=f"Compacted conversation {conversation_id}: {fact}",
                        confidence=0.4,
                        source="compaction",
                    )
                )
                written += 1
            return {"memory_flush": "ok", "memory_written": written}
        except Exception as exc:  # noqa: BLE001 - maintenance never blocks
            log.warning("compaction memory flush failed", extra={"error": str(exc)[:200]})
            return {"memory_flush": "failed", "memory_error": str(exc)[:200]}

    # -- core --------------------------------------------------------------
    async def compact(
        self,
        conversation_id: str,
        messages: Sequence[ChatMessage],
        *,
        provider: Any,
        budget: int,
        focus: str = "",
        preview: bool = False,
        previous: CompactionRecord | None = None,
        message_ids: Sequence[int] | None = None,
    ) -> CompactionResult:
        """Summarize the older span and persist the boundary.

        ``message_ids`` maps ``messages`` to durable row ids so the boundary is
        exact. When absent the boundary cannot be stored and the result is a
        preview only.
        """
        before = sum(estimate_tokens(m.content) for m in messages)
        if provider is None:
            return CompactionResult(False, reason="no provider for compaction", tokens_before=before)
        if len(messages) < 3:
            return CompactionResult(False, reason="not enough history to compact", tokens_before=before)

        split = choose_split(messages, self.config.keep_recent_tokens)
        if split is None:
            history_tokens = sum(estimate_tokens(m.content) for m in messages)
            if history_tokens <= self.config.keep_recent_tokens:
                return CompactionResult(
                    False,
                    reason=(
                        "history already fits within keep_recent_tokens "
                        f"({history_tokens} <= {self.config.keep_recent_tokens})"
                    ),
                    tokens_before=before,
                )
            return CompactionResult(
                False,
                reason="no safe split point; a tool-call group spans the whole history",
                tokens_before=before,
            )
        older = list(messages[:split])
        recent = list(messages[split:])
        if not older:
            return CompactionResult(False, reason="nothing to summarize", tokens_before=before)

        identifiers = (
            collect_identifiers(older) if self.config.identifier_policy == "strict" else []
        )
        transcript = shape_summary_input(
            older,
            self.config.summary_input_chars,
            previous.summary if previous else "",
        )
        if not transcript.strip():
            return CompactionResult(False, reason="empty summarizer input", tokens_before=before)

        flush = await self._flush_memory(conversation_id, older) if not preview else {}

        summary = ""
        degraded = False
        reason: str | None = None
        attempts = 0
        audit_note: str | None = None
        max_attempts = 1 + max(0, int(self.config.corrective_attempts))
        feedback = ""
        while attempts < max_attempts:
            attempts += 1
            request = build_summary_request(
                transcript + (("\n\n" + feedback) if feedback else ""),
                identifiers,
                previous_summary=previous.summary if previous else "",
                focus=focus,
            )
            try:
                response = await provider.complete(request)
            except Exception as exc:  # noqa: BLE001 - summarized below
                log.warning("compaction summarization failed", extra={"error": str(exc)[:200]})
                reason = f"summarizer error: {type(exc).__name__}"
                summary = ""
                break
            summary = str(getattr(response, "content", "") or "").strip()
            if not summary:
                reason = "summarizer returned no text"
                continue
            audit_note = await self._audit(transcript, summary, identifiers, provider)
            if audit_note is None:
                reason = None
                break
            reason = f"quality audit failed: {audit_note}"
            feedback = (
                "The previous summary failed an audit. Include these verbatim: "
                + audit_note
            )

        if not summary:
            if preview:
                return CompactionResult(
                    False,
                    reason=reason or "summarization failed",
                    tokens_before=before,
                    details={**flush, "attempts": attempts},
                )
            # Automatic path: commit a deterministic boundary so the session
            # stays usable instead of re-timing-out on every turn. The manual
            # path reports the failure and keeps history intact.
            if self.config.on_timeout != "deterministic":
                return CompactionResult(
                    False,
                    reason=reason or "summarization failed",
                    tokens_before=before,
                    details={**flush, "attempts": attempts},
                )
            summary = outline_degraded(reason or "summarization failed", older)
            degraded = True

        if audit_note is not None:
            degraded = True

        after = sum(estimate_tokens(m.content) for m in recent) + estimate_tokens(summary) + 2
        details = {
            **flush,
            "attempts": attempts,
            "audit": audit_note or "ok",
            "identifier_policy": self.config.identifier_policy,
            "identifiers": identifiers[:40],
            "focus": focus[:200] or None,
            "quality_degraded": degraded,
            "split_index": split,
        }

        if preview:
            return CompactionResult(
                True,
                summary=summary,
                messages_summarized=split,
                tokens_before=before,
                tokens_after=after,
                degraded=degraded,
                details=details,
            )

        if message_ids is None:
            return CompactionResult(
                False,
                reason="message ids required to persist a compaction boundary",
                tokens_before=before,
                details=details,
            )
        boundary_ids = [int(i) for i in message_ids[:split]]
        if not boundary_ids:
            return CompactionResult(False, reason="empty boundary", tokens_before=before)
        record_id = await save(
            conversation_id,
            summary,
            max(boundary_ids),
            split,
            details,
        )
        details["compaction_id"] = record_id
        return CompactionResult(
            True,
            summary=summary,
            upto_message_id=max(boundary_ids),
            messages_summarized=split,
            tokens_before=before,
            tokens_after=after,
            degraded=degraded,
            details=details,
        )

    async def _audit(
        self,
        transcript: str,
        summary: str,
        identifiers: Sequence[str],
        provider: Any,
    ) -> str | None:
        """Return ``None`` when the summary passes, else the missing list."""
        if self.config.identifier_policy != "strict" or not identifiers:
            return None
        # Deterministic check first: no model call needed when identifiers are
        # simply present. This is the audit that actually fails closed.
        missing = [value for value in identifiers if value not in summary]
        if not missing:
            return None
        try:
            response = await provider.complete(
                build_audit_request(transcript, summary, missing)
            )
        except Exception:  # noqa: BLE001 - audit is advisory, the local check rules
            return ", ".join(missing[:10])
        verdict = str(getattr(response, "content", "") or "").strip()
        if verdict.upper().startswith("OK"):
            # The model claims coverage but the local check disagrees; the
            # local check wins because it is deterministic.
            return ", ".join(missing[:10])
        if verdict.upper().startswith("MISSING:"):
            listed = verdict.split(":", 1)[1].strip()
            return listed or ", ".join(missing[:10])
        return ", ".join(missing[:10])

    # -- replay ------------------------------------------------------------
    @staticmethod
    def summary_message(record: CompactionRecord) -> ChatMessage:
        header = "Conversation summary (durable compaction; older turns omitted):"
        if record.details.get("quality_degraded"):
            header = (
                "Conversation summary (durable compaction; older turns omitted; "
                "quality degraded — some details may be missing):"
            )
        return ChatMessage(
            role="system",
            content=header + "\n" + record.summary,
            name=SUMMARY_NAME,
        )


__all__ = [
    "CompactionConfig",
    "CompactionEngine",
    "CompactionRecord",
    "CompactionResult",
    "SUMMARY_NAME",
    "build_audit_request",
    "build_summary_request",
    "choose_split",
    "clear",
    "collect_identifiers",
    "init_schema",
    "latest",
    "outline_degraded",
    "safe_split",
    "save",
    "shape_summary_input",
    "tool_group",
]
