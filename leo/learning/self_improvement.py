"""Approval-free self-improvement: learn durable skills and nightly review notes.

This engine is deliberately narrow. It writes only three kinds of durable
state: documentation-only skills in the real ``SkillRegistry``, bounded review
notes in ``kv``, and lifecycle flags on skills it authored itself. Standing
instructions are explicitly out of scope here: ``validate_change`` accepts the
``instruction`` kind as inert, but this PR provides no instruction store, prompt
injection path or writer for it. Durable guidance is expressed as a skill.
It never raises an approval card, never edits code,
and never touches safety rules, the hardline floor, secrets, secrets-shaped
content, untrusted external text, model/provider configuration or permissions.

Every write goes through :func:`validate_change`, which refuses dangerous
instruction shapes, credential shapes, execution primitives and protected
targets *before* anything is persisted. Everything written carries provenance:
the source turns/tasks (as safe ids) and the time it was learned, so "what did
you learn?" has a inspectable answer.

Recurring work is detected deterministically from the tool patterns of
completed turns (``tool.called`` plus ``chat.completed`` events). After the
configured number of distinct turns with the same pattern, a skill is
registered directly: the user corrects it only when it is wrong, in chat via
``/skills correct <name>``. Repeated corrections retire the learned skill
without deleting its history; the nightly pass also flags learned skills left
unused for ``unused_after_days``.

The nightly pass is registered on LEO's durable scheduler. It reads the day's
stalls, dead letters, silent fallbacks, failed turns and expired approvals,
then writes one short note with a proposed change for each observed failure.
The note is durable, carries no approval, and is included in the next
scheduled agent brief; ``review_notes()`` and the health report expose it on
request.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .. import db
from ..contracts import SkillManifest, utcnow
from ..events import EventBus
from ..logging import get_logger
from ..skills.compatibility import scan_security
from ..skills.sanitize import safe_description, scan_text

_LOG = get_logger("self_improvement")

ACTOR = "self-improvement:engine"
KV_PREFIX = "self-improvement:"
NIGHTLY_SCHEDULE_NAME = "nightly-failure-review"
NIGHTLY_WORKER = "self_improvement_review"

MAX_SOURCE_IDS = 12
MAX_TOOLS = 16
MAX_REVIEW_ROWS = 365
MAX_NOTE_CHARS = 1200
MAX_DOC_CHARS = 2000

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:#-]{1,160}$")
_SAFE_TOOL = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_SAFE_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Strong instruction/execution shapes. Advisory sanitizer warnings are treated
# as refusals here: a self-made rule must be LEO's own inert wording, not text
# copied out of a transcript, tool output or web page.
_DANGEROUS = re.compile(
    r"(?i)("
    r"\brm[\W_]{0,4}-rf\b|\bsudo\b|\bchmod[\W_]{0,4}777\b"
    r"|\beval[\W_]{0,4}\(|\bexec[\W_]{0,4}\(|\b__import__[\W_]{0,4}\("
    r"|\bsubprocess\s*\.|\bos\s*\.\s*system\s*\("
    r"|\bcurl\b[^\n|]{0,80}\|\s*(?:ba|z|k)?sh\b|\bwget\b"
    r"|\bignore\b[\W_]{0,12}(?:all\s+|the\s+)?(?:previous|prior|system|developer)\b"
    r"|\byou[\W_]{0,3}are[\W_]{0,3}now\b"
    r"|<\|\s*im_start\s*\|>|</?system>"
    r")"
)
# Key/value credentials are refused even without a well-known token prefix.
_KEY_SECRET = re.compile(
    r"(?i)\b(?:api[_-]?keys?|api[_-]?secrets?|access[_-]?tokens?|"
    r"refresh[_-]?tokens?|auth[_-]?tokens?|client[_-]?secrets?|"
    r"secret[_-]?keys?|private[_-]?keys?|pass(?:word|phrase)s?|credentials?|"
    r"secrets?|tokens?|pins?|otps?)\b[\s]*[:=]>?[\s]*[^\s,;\"'`]{1,}"
)
# Things self-improvement may never author or modify, however the request is
# phrased. This is a deny-list on *targets*, not a claim that any other target
# is safe: manifests are independently restricted to inert documentation.
_PROTECTED_TARGET = re.compile(
    r"(?i)(?:safety|guardrail|hardline|permission|approval|credential|secret|"
    r"token|password|provider|model[_-]?config|risk[_-]?policy|system[_-]?prompt|"
    r"self[_-]?modif|config|security[_-]?rule)"
)
_TRUSTED_SOURCES = frozenset(
    {"recurring_work", "nightly_review", "turn_correction", "operator_request"}
)
_ALLOWED_KINDS = frozenset(
    {"skill", "instruction", "note", "review_note", "correction", "retirement"}
)
_FORBIDDEN_MANIFEST_FIELDS = (
    "permissions", "gates", "dependencies", "entrypoint", "bundle",
    "provenance_verified", "provenance_approval_id",
)
# Failure categories are fixed and their proposed changes are engine-owned
# sentences. No event payload text is ever copied into a note.
_FAILURE_CHANGES: dict[str, str] = {
    "stall": "send a brief progress update before long tool work and split large turns",
    "dead_letter": "reduce repeated retries and name the failing step in the task result",
    "silent_fallback": "make the fallback visible in the next surface instead of switching quietly",
    "failed_turn": "validate tool arguments before the call and report the reason on failure",
    "expired_approval": "narrow the requested action or remind the operator before expiry",
}
# Learned skills unused for a long time are flagged, never silently deleted.
_UNUSED_CHANGE = "flag the learned skill as stale and confirm it still helps before reuse"
_FAILURE_EVENT_TOPICS: dict[str, tuple[str, ...]] = {
    "silent_fallback": (
        "automation.needs_you", "schedule.execution_error",
        "task.delivery_nowhere", "schedule.monitor_failed",
    ),
    "failed_turn": ("run.failed", "skill.run.failed"),
    "expired_approval": ("schedule.expired", "learning.expired"),
}


class SelfImprovementDenied(ValueError):
    """A self-improvement write was refused before any durable mutation."""


def _deny(reason: str) -> None:
    raise SelfImprovementDenied(reason)


def _safe_id(value: Any, *, field: str = "id") -> str:
    """Keep a bounded id, or a hash of it; never copy arbitrary text."""
    text = " ".join(str(value or "").split())[:160]
    _, errors, warnings = scan_text(text)
    if _SAFE_ID.fullmatch(text) and not (errors or warnings or _DANGEROUS.search(text)
                                        or _KEY_SECRET.search(text)):
        return text
    # Unknown text is hashed, never copied: a transcript id can contain a
    # credential or an instruction, and neither belongs in provenance.
    digest = hashlib.sha256(str(value or "").encode("utf-8", "replace")).hexdigest()[:16]
    return f"h:{digest}"


def _safe_tool(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not _SAFE_TOOL.fullmatch(text):
        _deny(f"invalid tool name in recurring pattern: {text[:40]!r}")
    return text


def _text(value: Any, *, field: str) -> str:
    """Validate free text that will become durable self-authored content."""
    clean, errors, warnings = scan_text(str(value or ""), field=field)
    if errors:
        _deny("; ".join(errors))
    if warnings:
        _deny("; ".join(warnings))
    if _DANGEROUS.search(clean) or _KEY_SECRET.search(clean):
        _deny(f"{field} contains a dangerous instruction or credential shape")
    return clean.strip()


def validate_change(change: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one proposed self-improvement write without applying it.

    Only inert skills/standing instructions and notes are accepted. The caller
    must name a trusted internal source; curated or untrusted external text is
    refused even if its content happens to be harmless.
    """
    if not isinstance(change, Mapping):
        _deny("self-improvement change must be an object")
    kind = str(change.get("kind") or "")
    if kind not in _ALLOWED_KINDS:
        _deny(f"unsupported self-improvement kind: {kind[:60]!r}")
    source = str(change.get("source") or "")
    if source not in _TRUSTED_SOURCES:
        _deny(
            f"untrusted source {source[:40]!r}; only LEO's own durable work can "
            "be turned into a durable rule"
        )
    name = str(change.get("name") or "")
    resource = str(change.get("resource") or name)
    for field, value in (("name", name), ("resource", resource)):
        if value and _PROTECTED_TARGET.search(value):
            _deny(f"{field} names a protected target; safety, secrets, approvals and "
                  "model/provider configuration are never self-modified")
    if name and not _SAFE_ID.fullmatch(name):
        _deny("name must be a bounded, plain identifier")
    for field in ("description", "documentation", "content", "title"):
        if field in change and change[field] not in (None, ""):
            _text(change[field], field=field)
    cleaned: dict[str, Any] = {
        "kind": kind, "source": source, "name": name, "resource": resource,
    }
    if kind in {"skill", "instruction"}:
        manifest = change.get("manifest")
        if not isinstance(manifest, Mapping):
            _deny("skill changes require a manifest")
        for field in _FORBIDDEN_MANIFEST_FIELDS:
            value = manifest.get(field)
            if value not in (None, "", False, [], {}):
                _deny(f"self-authored skills cannot set {field}")
        if str(manifest.get("entrypoint") or ""):
            _deny("self-authored skills are documentation-only (no entrypoint)")
        if str(manifest.get("source") or "native") != "native":
            _deny("self-authored skills must be inert native-format manifests")
        if str(manifest.get("created_by") or "agent") != "agent":
            _deny("self-authored skills must be recorded as agent-authored")
        blockers = scan_security(dict(manifest))
        if blockers:
            _deny("; ".join(blockers))
        cleaned["manifest"] = dict(manifest)
    return cleaned


class SelfImprovementEngine:
    """Write approved-by-correctness skills and notes; never ask for a card.

    Standing instructions are out of scope: durable guidance is a skill here.
    """

    def __init__(
        self,
        skills: Any,
        events: EventBus | None = None,
        settings: Any = None,
        *,
        actor: str = ACTOR,
    ) -> None:
        cfg = getattr(getattr(settings, "learning", None), "self_improvement", None)
        self.skills = skills
        self.events = events or EventBus()
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.min_recurrences = max(2, int(getattr(cfg, "min_recurrences", 3) or 3))
        self.max_corrections = max(2, int(getattr(cfg, "max_corrections", 3) or 3))
        self.unused_after_days = max(1, int(getattr(cfg, "unused_after_days", 30) or 30))
        self.actor = str(actor or ACTOR)

    # ------------------------------------------------------------------ state
    @staticmethod
    def _key(name: str) -> str:
        return f"{KV_PREFIX}{name}"

    async def _load(self, name: str) -> Any:
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (self._key(name),))
        if not row:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    async def _save(self, name: str, value: Mapping[str, Any]) -> None:
        await db.execute(
            "INSERT OR REPLACE INTO kv(key,value,updated_at) VALUES(?,?,?)",
            (self._key(name), json.dumps(value, ensure_ascii=False, default=str), utcnow()),
        )

    async def _emit(self, topic: str, **payload: Any) -> None:
        try:
            await self.events.emit(topic, **payload)
        except Exception as exc:  # telemetry must never fail the learning write
            _LOG.warning("self-improvement event failed: %s", exc)

    async def _audit(
        self, action: str, resource: str, decision: str, metadata: Mapping[str, Any]
    ) -> None:
        try:
            await db.execute(
                "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,"
                "correlation_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (utcnow(), self.actor, action, resource[:160], "low", decision, "",
                 json.dumps(metadata, ensure_ascii=False, default=str)[:1000], None),
            )
        except Exception as exc:  # audit is best effort, never the mutation gate
            _LOG.warning("self-improvement audit failed: %s", exc)

    # ---------------------------------------------------------- observations
    async def observe_tool_call(
        self,
        tool: str,
        *,
        conversation_id: str = "",
        call_id: str = "",
        actor: str = "",
    ) -> None:
        """Accumulate one turn's safe tool names for later pattern detection."""
        if not self.enabled or not tool:
            return
        name = _safe_tool(tool)
        conv = _safe_id(conversation_id or "unknown", field="conversation_id")
        key = f"turn:{conv}"
        row = await self._load(key) or {"tools": [], "refs": [], "updated_at": utcnow()}
        tools = [str(item) for item in row.get("tools") or []][:MAX_TOOLS]
        refs = [str(item) for item in row.get("refs") or []][:MAX_SOURCE_IDS]
        if name not in tools:
            tools.append(name)
        ref = _safe_id(call_id or actor or conv)
        if ref not in refs:
            refs.append(ref)
        row.update({"tools": tools[:MAX_TOOLS], "refs": refs[:MAX_SOURCE_IDS],
                    "updated_at": utcnow(), "conversation_id": conv})
        await self._save(key, row)

    async def finish_turn(
        self,
        conversation_id: str,
        *,
        source_id: str = "",
        status: str = "completed",
        task_id: str = "",
    ) -> dict[str, Any] | None:
        """Promote a finished turn's tool pattern; called once per terminal turn."""
        if not self.enabled or str(status or "").lower() not in {"", "completed", "success", "succeeded"}:
            return None
        conv = _safe_id(conversation_id or "unknown", field="conversation_id")
        key = f"turn:{conv}"
        row = await self._load(key)
        if row is None:
            return None
        await db.execute("DELETE FROM kv WHERE key=?", (self._key(key),))
        tools = [str(item) for item in row.get("tools") or []]
        if not tools:
            return None
        refs = [str(item) for item in row.get("refs") or []]
        sources = [item for item in (source_id, task_id, conv, *refs) if item]
        return await self.observe_workflow(
            tools, source_ids=sources, conversation_id=conv,
        )

    async def observe_workflow(
        self,
        tools: Iterable[str],
        *,
        kind: str = "tool_sequence",
        source_ids: Sequence[str] = (),
        conversation_id: str = "",
        task_id: str = "",
    ) -> dict[str, Any]:
        """Count a distinct turn/task with a tool pattern; save a skill at the threshold."""
        if not self.enabled:
            return {"status": "disabled"}
        names = sorted({_safe_tool(item) for item in tools})
        if not names or len(names) > MAX_TOOLS:
            _deny("a recurring pattern needs between 1 and 16 tool names")
        if not _SAFE_ID.fullmatch(kind):
            _deny("invalid workflow kind")
        signature = hashlib.sha256(
            (kind + "|" + ",".join(names)).encode("utf-8")
        ).hexdigest()[:16]
        key = f"workflow:{signature}"
        row = await self._load(key) or {
            "signature": signature, "kind": kind, "tools": names,
            "occurrences": 0, "sources": [], "conversations": [],
            "first_seen": utcnow(), "last_seen": utcnow(),
        }
        source = _safe_id(source_ids[0] if source_ids else (task_id or conversation_id or utcnow()))
        if source in [str(item) for item in row.get("sources") or []]:
            row["status"] = "duplicate"
            return row
        row["occurrences"] = int(row.get("occurrences") or 0) + 1
        row["sources"] = ([str(item) for item in row.get("sources") or []] + [source])[-MAX_SOURCE_IDS:]
        if conversation_id:
            conv = _safe_id(conversation_id)
            conversations = [str(item) for item in row.get("conversations") or []]
            if conv not in conversations:
                row["conversations"] = (conversations + [conv])[-MAX_SOURCE_IDS:]
        if task_id:
            task = _safe_id(task_id)
            tasks = [str(item) for item in row.get("tasks") or []]
            if task not in tasks:
                row["tasks"] = (tasks + [task])[-MAX_SOURCE_IDS:]
        row["last_seen"] = utcnow()
        row["status"] = "observed"
        await self._save(key, row)
        await self._emit(
            "self_improvement.workflow_observed", signature=signature,
            occurrences=row["occurrences"], tools=names, source=source,
        )
        if int(row["occurrences"]) >= self.min_recurrences:
            learned = await self._promote(row)
            row["status"] = "learned"
            row["skill"] = learned["name"]
            row["learned_at"] = learned["learned_at"]
            await self._save(key, row)
        return row

    async def _promote(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Register one durable, inert skill with full provenance; no approval."""
        signature = str(observation["signature"])
        tools = [str(item) for item in observation.get("tools") or []]
        sources = [str(item) for item in observation.get("sources") or []]
        now = utcnow()
        name = f"learned-tool-sequence-{signature[:8]}"
        description = safe_description("Recurring work with " + ", ".join(tools[:3]))
        documentation = "\n".join((
            "Recurring workflow learned from LEO's own completed turns.",
            "Pattern: " + ", ".join(tools),
            "Use when the task matches this tool pattern.",
            "Approach: confirm the task, reuse the recorded tools, then report the result.",
            "Source turns: " + ", ".join(sources[:5]),
            f"Signature: {signature}  Learned at: {now}",
        ))[:MAX_DOC_CHARS]
        origin = {
            "learned_by": "self_improvement",
            "kind": "recurring_workflow",
            "signature": signature,
            "occurrences": int(observation.get("occurrences") or 0),
            "sources": sources[:MAX_SOURCE_IDS],
            "conversations": [str(item) for item in observation.get("conversations") or []][:MAX_SOURCE_IDS],
            "tasks": [str(item) for item in observation.get("tasks") or []][:MAX_SOURCE_IDS],
            "first_seen": str(observation.get("first_seen") or now),
            "last_seen": str(observation.get("last_seen") or now),
            "learned_at": now,
        }
        provenance = {
            "kind": "recurring_workflow", "signature": signature,
            "learned_at": now, "source_kinds": ["tool.called", "chat.completed"],
        }
        manifest: dict[str, Any] = {
            "name": name, "description": description, "version": "0.1.0",
            "author": "leo-agent", "category": "learned-workflows",
            "source": "native", "created_by": "agent", "documentation": documentation,
            "permissions": [], "gates": {}, "dependencies": [], "entrypoint": None,
            "input_schema": {}, "output_schema": {}, "bundle": {}, "tests": [],
            "origin": origin, "provenance": provenance,
            "provenance_verified": False, "provenance_approval_id": None,
        }
        validate_change({
            "kind": "skill", "source": "recurring_work", "name": name,
            "description": description, "documentation": documentation,
            "manifest": manifest,
        })
        previous = await self.skills.get(name)
        if previous is not None:
            previous_origin = dict(getattr(previous, "origin", None) or {})
            if (str(getattr(previous, "created_by", "")) != "agent"
                    or previous_origin.get("learned_by") != "self_improvement"):
                _deny("a skill with this name was not learned by self-improvement")
            parts = [int(item) for item in str(previous.version).split(".")[:3] if item.isdigit()]
            while len(parts) < 3:
                parts.append(0)
            parts[-1] += 1
            manifest["version"] = ".".join(str(item) for item in parts)
        record = await self.skills.register(
            SkillManifest.model_validate(manifest), replace=previous is not None,
        )
        await self._audit("self_improvement.skill", name, "allowed", origin)
        await self._emit(
            "self_improvement.skill_learned", name=name, signature=signature,
            occurrences=origin["occurrences"], sources=sources[:5],
            learned_at=now, approval_id=None,
        )
        return {"name": name, "learned_at": now, "sources": sources, "version": record.version}

    # -------------------------------------------------------- review / notes
    async def write_note(
        self,
        text: str,
        *,
        source: str = "nightly_review",
        sources: Sequence[str] = (),
        title: str = "self-review",
    ) -> dict[str, Any]:
        """Persist one bounded, provenance-bearing note. Never approval-gated."""
        validate_change({
            "kind": "review_note", "source": source, "name": title,
            "title": title, "content": text,
        })
        created_at = utcnow()
        note_id = "note-" + hashlib.sha256(
            (created_at + text).encode("utf-8")
        ).hexdigest()[:12]
        record = {
            "id": note_id, "kind": "note", "title": title, "note": text[:MAX_NOTE_CHARS],
            "source": source, "sources": [_safe_id(item) for item in sources][:MAX_SOURCE_IDS],
            "created_at": created_at, "approval_id": None,
        }
        await self._save(f"note:{note_id}", record)
        await self._audit("self_improvement.note", note_id, "allowed", record)
        await self._emit(
            "self_improvement.note_written", note_id=note_id, source=source,
            sources=record["sources"], approval_id=None,
        )
        return record

    async def review_notes(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = await db.fetchall(
            "SELECT value FROM kv WHERE key LIKE ? ORDER BY key DESC LIMIT ?",
            (self._key("review:") + "%", max(0, min(int(limit), MAX_REVIEW_ROWS))),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                value = json.loads(row["value"])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    async def latest_review_note(self, *, limit_chars: int = 1200) -> str:
        notes = await self.review_notes(limit=1)
        if not notes:
            return ""
        note = str(notes[0].get("note") or "")
        day = str(notes[0].get("day") or "")
        text = f"{day}: {note}" if day else note
        return text[: max(0, int(limit_chars))]

    async def _unused_learned(self, now: datetime) -> list[str]:
        """Learned skills whose last recorded use is older than the threshold."""
        cutoff = now - timedelta(days=self.unused_after_days)
        names: list[str] = []
        for record in await self.skills.list():
            origin = dict(getattr(record, "origin", None) or {})
            if origin.get("learned_by") != "self_improvement":
                continue
            name = str(getattr(record, "name") or "")
            if not _SAFE_ID.fullmatch(name):
                continue
            row = await db.fetchone(
                "SELECT last_viewed_at, last_used_at, last_patched_at, updated_at, state "
                "FROM skill_usage WHERE skill_name=?",
                (name,),
            )
            if row is None or str(row["state"] or "active") == "archived":
                continue
            used = [value for value in (
                _parse_time(row["last_viewed_at"]), _parse_time(row["last_used_at"]),
                _parse_time(row["last_patched_at"]), _parse_time(row["updated_at"]),
            ) if value is not None]
            # A skill with no usage record at all is only aged from its
            # learning time; a used skill ages from its last recorded use.
            last = max(used, default=_parse_time(origin.get("learned_at")))
            if last is not None and last <= cutoff:
                names.append(name)
        return names[:MAX_SOURCE_IDS]

    async def nightly_review(
        self,
        day: str | None = None,
        *,
        now: datetime | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Write one short durable note about the day's failures, with no card."""
        if not self.enabled and not force:
            return {"status": "disabled"}
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        if day is None:
            day = (moment.astimezone(UTC) - timedelta(days=1)).date().isoformat()
        if not _SAFE_DAY.fullmatch(str(day)):
            _deny("day must be YYYY-MM-DD")
        day = str(day)
        key = f"review:{day}"
        existing = await self._load(key)
        if existing is not None and not force:
            existing["status"] = "already-written"
            return existing
        try:
            start = datetime.fromisoformat(day).replace(tzinfo=UTC)
        except ValueError as exc:
            raise SelfImprovementDenied(f"invalid review day: {day}") from exc
        end = start + timedelta(days=1)
        findings, sources = await self._collect(start, end)
        # R4: a learned skill that has not been used recently is flagged, not
        # deleted. The state change is reversible, and one bad skill cannot
        # stop the review note for the day.
        unused = await self._unused_learned(moment)
        for name in unused:
            try:
                if str((await self.skills.usage(name)).get("state") or "active") != "stale":
                    await self.skills.set_state(name, "stale")
            except Exception as exc:  # noqa: BLE001 - one flag must not fail the pass
                _LOG.warning("could not flag unused learned skill %s: %s", name, exc)
        if unused:
            findings.append({
                "kind": "unused_skill", "count": len(unused),
                "change": _UNUSED_CHANGE, "sources": unused[:5],
            })
            for name in unused[:5]:
                if name not in sources:
                    sources.append(name)
            sources = sources[:MAX_SOURCE_IDS]
        total = sum(int(item["count"]) for item in findings)
        lines = [f"Nightly review for {day}: {total} findings."]
        for item in findings:
            if int(item["count"]):
                lines.append(f"- {item['kind']} ({item['count']}): {item['change']}")
        if total == 0:
            lines.append("- No observed failures; keep the current approach.")
        text = "\n".join(lines)[:MAX_NOTE_CHARS]
        note = await self.write_note(
            text, source="nightly_review", sources=sources, title=f"nightly-review-{day}",
        )
        record = {
            "id": note["id"], "kind": "nightly_review", "day": day,
            "note": note["note"], "findings": findings, "sources": sources,
            "created_at": note["created_at"], "approval_id": None,
            "status": "written",
        }
        await self._save(key, record)
        await self._emit(
            "self_improvement.nightly_review", day=day, note_id=note["id"],
            findings={item["kind"]: item["count"] for item in findings},
            sources=len(sources), approval_id=None,
        )
        return record

    async def _collect(
        self, start: datetime, end: datetime
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Count fixed failure categories from durable rows; keep only safe ids."""
        counts = {kind: 0 for kind in _FAILURE_CHANGES}
        sources: list[str] = []

        def add(kind: str, rows: Sequence[Any]) -> None:
            counts[kind] += len(rows)
            for row in rows[:5]:
                value = row if isinstance(row, (str, int)) else (
                    row["id"] if hasattr(row, "keys") and "id" in row.keys() else str(row)
                )
                source = _safe_id(value)
                if source not in sources:
                    sources.append(source)

        begin, finish = start.isoformat(), end.isoformat()
        stall_rows = await db.fetchall(
            "SELECT id FROM messages WHERE created_at>=? AND created_at<? "
            "AND json_extract(metadata,'$.stall_nudge')=1 LIMIT 50",
            (begin, finish),
        )
        add("stall", stall_rows)
        dead_rows = await db.fetchall(
            "SELECT id FROM tasks WHERE json_extract(metadata,'$.dead_letter')=1 "
            "AND COALESCE(completed_at, updated_at)>=? AND COALESCE(completed_at, updated_at)<? "
            "LIMIT 50",
            (begin, finish),
        )
        add("dead_letter", dead_rows)
        failed_rows = await db.fetchall(
            "SELECT id FROM messages WHERE created_at>=? AND created_at<? "
            "AND json_extract(metadata,'$.run_status') IN ('error','failed','incomplete') "
            "LIMIT 50",
            (begin, finish),
        )
        add("failed_turn", failed_rows)
        expired_rows = await db.fetchall(
            "SELECT id FROM approvals WHERE status='expired' AND resolved_at>=? "
            "AND resolved_at<? LIMIT 50",
            (begin, finish),
        )
        add("expired_approval", expired_rows)
        topics = tuple(
            topic for names in _FAILURE_EVENT_TOPICS.values() for topic in names
        )
        placeholders = ",".join("?" for _ in topics)
        event_rows = await db.fetchall(
            f"SELECT id,topic FROM events WHERE created_at>=? AND created_at<? "
            f"AND topic IN ({placeholders}) LIMIT 200",
            (begin, finish, *topics),
        )
        for row in event_rows:
            for kind, names in _FAILURE_EVENT_TOPICS.items():
                if str(row["topic"]) in names:
                    add(kind, [row["id"]])
                    break
        findings = [
            {"kind": kind, "count": int(counts[kind]), "change": change,
             "sources": sources[:5] if counts[kind] else []}
            for kind, change in _FAILURE_CHANGES.items()
        ]
        return findings, sources[:MAX_SOURCE_IDS]

    # ------------------------------------------------------------- lifecycle
    async def record_correction(
        self,
        skill_name: str,
        *,
        source_id: str = "",
    ) -> dict[str, Any]:
        """A correction is a signal, not a command; repeated ones retire a skill."""
        validate_change({
            "kind": "correction", "source": "turn_correction",
            "resource": skill_name, "name": skill_name,
        })
        record = await self.skills.get(skill_name)
        if record is None:
            _deny("cannot correct an unknown skill")
        origin = dict(getattr(record, "origin", None) or {})
        if (str(getattr(record, "created_by", "")) != "agent"
                or origin.get("learned_by") != "self_improvement"):
            _deny("only skills learned by self-improvement can be retired")
        key = f"correction:{_safe_id(skill_name)}"
        row = await self._load(key) or {"skill": skill_name, "sources": [], "count": 0}
        source = _safe_id(source_id or utcnow())
        if source not in [str(item) for item in row.get("sources") or []]:
            row["sources"] = ([str(item) for item in row.get("sources") or []] + [source])[-MAX_SOURCE_IDS:]
            row["count"] = int(row.get("count") or 0) + 1
        retired = bool(row["count"] >= self.max_corrections)
        row["updated_at"] = utcnow()
        row["retired"] = retired
        await self._save(key, row)
        if retired and str(getattr(record, "state", "active")) != "archived":
            await self.skills.set_state(skill_name, "archived")
            await self._audit("self_improvement.retirement", skill_name, "allowed", row)
            await self._emit(
                "self_improvement.skill_retired", name=skill_name,
                corrections=row["count"], sources=row["sources"][:5],
                learned_from=origin, approval_id=None,
            )
        return {"skill": skill_name, "corrections": int(row["count"]), "retired": retired}

    async def learned(self, limit: int = 20) -> list[dict[str, Any]]:
        """Answer "what did you learn?" with sources and timestamps."""
        items: list[dict[str, Any]] = []
        for record in await self.skills.list():
            origin = dict(getattr(record, "origin", None) or {})
            if origin.get("learned_by") != "self_improvement":
                continue
            items.append({
                "kind": "skill", "name": record.name, "version": record.version,
                "state": str(getattr(record, "state", "active")),
                "signature": origin.get("signature"), "sources": origin.get("sources") or [],
                "learned_at": origin.get("learned_at"),
            })
        notes = await self.review_notes(limit=limit)
        items.extend({
            "kind": item.get("kind"), "id": item.get("id"), "day": item.get("day"),
            "sources": item.get("sources") or [], "learned_at": item.get("created_at"),
        } for item in notes)
        return items[: max(0, int(limit))]

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "actor": self.actor,
            "approval_required": False,
            "min_recurrences": self.min_recurrences,
            "max_corrections": self.max_corrections,
            "unused_after_days": self.unused_after_days,
            "nightly_schedule": NIGHTLY_SCHEDULE_NAME,
            # Standing instructions are explicitly out of scope for this PR.
            "mutable_targets": ["skill", "note"],
            "protected_targets": [
                "safety rules", "hardline floor", "secrets", "approvals",
                "permissions", "model/provider configuration",
            ],
        }


def _parse_time(value: Any) -> datetime | None:
    """Parse one durable ISO timestamp, treating naive values as UTC."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


async def recent_review_brief(*, limit_chars: int = 900) -> str:
    """Bounded note text for the next scheduled agent brief (no raw events)."""
    try:
        row = await db.fetchone(
            "SELECT value FROM kv WHERE key LIKE ? ORDER BY key DESC LIMIT 1",
            (KV_PREFIX + "review:%",),
        )
    except Exception as exc:  # a read-only brief must never break a schedule
        _LOG.warning("review brief read failed: %s", exc)
        return ""
    if row is None:
        return ""
    try:
        value = json.loads(row["value"])
    except (TypeError, ValueError):
        return ""
    if not isinstance(value, dict):
        return ""
    day = str(value.get("day") or "")
    note = str(value.get("note") or "")[: max(0, int(limit_chars))]
    return f"{day}: {note}" if day else note


__all__ = [
    "ACTOR", "KV_PREFIX", "NIGHTLY_SCHEDULE_NAME", "NIGHTLY_WORKER",
    "SelfImprovementDenied", "SelfImprovementEngine", "recent_review_brief",
    "validate_change",
]
