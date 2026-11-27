"""Bounded post-turn review: the self-improvement loop.

One completed turn is reviewed against a bounded, durable transcript. The
reviewer is a normal model call with a deliberately small tool scope:
``skill_list``/``skill_search``/``skill_read``/``skill_manage`` and, when
memory capture is enabled, ``memory_search``/``memory_propose``. Every tool
call goes through the real ``ToolExecutorImpl`` (permissions, risk, audit and
scope enforcement), so this engine cannot widen its own authority.

Safety properties enforced here, not merely documented:

* the transcript is read from durable rows and bounded by count/characters;
  unfinished turns are not evidence;
* the reviewer's action is a tool call, never a self-attested claim; a write
  without ``skill_read`` is refused by ``skill_manage`` itself;
* every run writes one bounded artifact (transcript digest, tool-call digests,
  outcome) and one audit row; a no-op review is a successful outcome;
* a failed transcript load, blocked router or malformed output performs zero
  writes and emits ``review.failed`` with a truthful reason;
* reviews are serialized per conversation and globally; shutdown cancels them.

The engine never applies a memory note or skill outside the normal tool path,
never accepts an approval on the operator's behalf, and never deletes a skill.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import db
from ..config import Settings
from ..contracts import ChatMessage, ModelRequest, ToolCall, new_id, utcnow
from ..events import Event, EventBus
from ..logging import get_logger
from ..security.permissions import ApprovalRequired
from ..skills.sanitize import scan_text
from ..task_scope import cap_report
from ..tools.skill_read_marks import reset_skill_read_marks
from .prompts import (
    LEARN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_learn_prompt,
    build_review_prompt,
)
from .transcript import Transcript, load_transcript

_LOG = get_logger("review")

REVIEW_ACTOR = "review:engine"
LEARN_ACTOR = "learn:engine"
REVIEW_SCOPE: frozenset[str] = frozenset({
    "skill_list", "skill_search", "skill_read", "skill_manage",
    "memory_search", "memory_propose",
})
# Explicit /learn may gather sources with read-only tools. There is no
# terminal/shell/write-file tool here: authored content still goes through
# skill_manage, and every path is still confined by the executor.
SOURCE_TOOLS: frozenset[str] = frozenset({
    "read_file", "search_files", "list_files", "dir_tree", "file_hash", "http_get",
})
LEARN_SCOPE: frozenset[str] = frozenset({
    "skill_list", "skill_search", "skill_read", "skill_manage",
}) | SOURCE_TOOLS
_UNSUCCESSFUL_TURN_STATUSES = frozenset({"error", "failed", "cancelled", "canceled"})
_TOOL_MESSAGE_LIMIT = 4_000
_SUMMARY_LIMIT = 600


@dataclass
class ReviewOutcome:
    review_id: str
    conversation_id: str
    status: str = "completed"  # completed | failed | skipped
    mode: str = "review"  # review | learn
    rounds: int = 0
    actions: list[dict[str, Any]] = field(default_factory=list)
    transcript_digest: str = ""
    artifact_path: str | None = None
    # Applied skill creations and writes staged for operator approval. Both are
    # carried on the outcome/artifact and completion event so a channel notifier
    # can name them without re-reading durable state.
    created: list[dict[str, Any]] = field(default_factory=list)
    staged: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    budget_exhausted: bool = False
    # Bounded handoff produced when a hard cap stopped the review: what was
    # done, what was left, and the options for continuing it.
    progress_report: str = ""


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()


def _bounded(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _safe_text(value: Any, limit: int = _SUMMARY_LIMIT) -> str:
    """Bound and sanitize model output before it reaches an artifact or audit."""
    clean, errors, _ = scan_text(_bounded(value, limit), field="review output")
    return "[redacted: credential-shaped content]" if errors else clean


def _tool_call(raw: Any) -> tuple[str, str, dict[str, Any]]:
    """Normalize a provider tool call into ``(id, name, arguments)``."""
    if isinstance(raw, Mapping):
        function = raw.get("function") if isinstance(raw.get("function"), Mapping) else {}
        name = str(raw.get("name") or function.get("name") or "")
        arguments = raw.get("arguments", function.get("arguments", {}))
        call_id = str(raw.get("id") or raw.get("call_id") or new_id("call"))
    else:  # pragma: no cover - defensive
        name = str(getattr(raw, "name", "") or "")
        arguments = getattr(raw, "arguments", {})
        call_id = str(getattr(raw, "id", "") or new_id("call"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    return call_id, name, dict(arguments)


class ReviewEngine:
    """Decide *when* to review and run the bounded loop."""

    def __init__(
        self,
        events: EventBus,
        settings: Settings,
        *,
        router: Any,
        executor: Any,
        skills: Any = None,
        learning: Any = None,
        skip_source: Any = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.events = events
        self.settings = settings
        self.router = router
        self.executor = executor
        self.skills = skills
        self.learning = learning
        self._skip_source = skip_source
        base = artifact_dir or (Path(settings.resolve(settings.paths.data)) / "reviews")
        self.artifact_dir = Path(base)
        self._pending: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._running: set[str] = set()
        self._semaphore = asyncio.Semaphore(1)
        self._stopped = False
        self._stats: dict[str, Any] = {
            "runs": 0, "completed": 0, "failed": 0, "skipped": 0,
            "events_suppressed": 0, "skills_written": 0, "skills_staged": 0,
            "memories_proposed": 0, "learn_runs": 0, "learn_skills": 0,
            "last_run_at": None, "last_outcome": None,
        }

    # -- configuration ------------------------------------------------------

    @property
    def config(self) -> Any:
        return getattr(getattr(self.settings, "learning", None), "review", None)

    @property
    def enabled(self) -> bool:
        cfg = self.config
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    # -- triggering ---------------------------------------------------------

    async def observe(self, event: Event) -> None:
        """Queue one completed turn; spawn a review when its budget is due."""
        if self._stopped or not self.enabled:
            return
        cfg = self.config
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        conversation_id = str(payload.get("conversation_id") or "").strip()
        if not conversation_id:
            return
        status = str(payload.get("status") or "").strip().lower()
        if status in _UNSUCCESSFUL_TURN_STATUSES:
            return
        if self._skip_source is not None:
            try:
                if self._skip_source(
                    channel=payload.get("channel"),
                    conversation_id=conversation_id,
                    source=str(event.source or ""),
                ):
                    self._stats["events_suppressed"] += 1
                    return
            except Exception as exc:  # noqa: BLE001 - a bad predicate must not stop review
                _LOG.warning("review source predicate failed: %s", exc)
        interval = int(getattr(cfg, "interval_turns", 0) or 0)
        if interval <= 0:
            return
        self._pending[conversation_id] = self._pending.get(conversation_id, 0) + 1
        if (
            self._pending[conversation_id] >= interval
            and conversation_id not in self._running
        ):
            self._pending[conversation_id] = 0
            self._spawn(conversation_id)

    def trigger(
        self, conversation_id: str, *, focus: str | None = None
    ) -> bool:
        """Run one review now (operator/command path); returns False when stopped."""
        conversation_id = str(conversation_id or "").strip()
        if not conversation_id or self._stopped:
            return False
        self._spawn(conversation_id, focus=focus, explicit=True)
        return True

    def trigger_learn(
        self, conversation_id: str | None, source: str, *, focus: str | None = None
    ) -> bool:
        """Author a skill from an explicit source; the user asked, so writes are direct."""
        source = str(source or "").strip()
        if not source or self._stopped:
            return False
        cfg = self.config
        if cfg is not None and not getattr(cfg, "learn_enabled", True):
            return False
        task = asyncio.create_task(
            self.run(
                str(conversation_id or ""), focus=focus, explicit=True,
                mode="learn", source=source,
            ),
            name="leo-learn",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    def _spawn(
        self, conversation_id: str, *, focus: str | None = None, explicit: bool = False
    ) -> None:
        if self._stopped:
            return
        task = asyncio.create_task(
            self.run(conversation_id, focus=focus, explicit=explicit),
            name=f"leo-review-{conversation_id[:32]}",
        )
        self._tasks.add(task)
        self._running.add(conversation_id)
        task.add_done_callback(
            lambda done, cid=conversation_id: self._task_done(cid, done)
        )

    def _task_done(self, conversation_id: str, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        self._running.discard(conversation_id)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            _LOG.warning("review task failed: %s", exc)
        interval = int(getattr(self.config, "interval_turns", 0) or 0)
        if (
            not self._stopped
            and interval > 0
            and self._pending.get(conversation_id, 0) >= interval
        ):
            self._pending[conversation_id] = 0
            self._spawn(conversation_id)

    def stop(self) -> None:
        """Cancel in-flight reviews; called during app shutdown."""
        self._stopped = True
        for task in list(self._tasks):
            task.cancel()

    async def drain(self, timeout: float = 5.0) -> None:
        """Await cancelled/running tasks for a bounded shutdown."""
        tasks = list(self._tasks)
        if not tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except (TimeoutError, asyncio.CancelledError):
            pass

    # -- one review ---------------------------------------------------------

    async def run(
        self,
        conversation_id: str,
        *,
        focus: str | None = None,
        explicit: bool = False,
        mode: str = "review",
        source: str | None = None,
    ) -> ReviewOutcome:
        conversation_id = str(conversation_id or "").strip()
        mode = "learn" if str(mode).lower() == "learn" else "review"
        started = time.perf_counter()
        review_id = new_id("learn" if mode == "learn" else "rev")
        cfg = self.config
        if mode == "review" and not conversation_id:
            return ReviewOutcome(review_id, "", status="skipped", error="no conversation id")
        if mode == "learn" and not str(source or "").strip():
            return ReviewOutcome(review_id, conversation_id, status="skipped", error="no source to learn from")
        if not explicit and not self.enabled:
            return ReviewOutcome(
                review_id, conversation_id, status="skipped", mode=mode,
                error="review disabled",
            )
        lock_key = f"{mode}:{conversation_id or 'adhoc'}"
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        async with lock, self._semaphore:
            try:
                outcome = await self._run(
                    review_id, conversation_id, cfg, focus=focus, explicit=explicit,
                    mode=mode, source=source,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one review never breaks the app
                outcome = ReviewOutcome(
                    review_id, conversation_id, status="failed", mode=mode,
                    error=_bounded(f"{type(exc).__name__}: {exc}", 400),
                )
            await self._finish(outcome, cfg)
        outcome.duration_ms = round((time.perf_counter() - started) * 1000, 3)
        return outcome

    async def _run(
        self,
        review_id: str,
        conversation_id: str,
        cfg: Any,
        *,
        focus: str | None,
        explicit: bool,
        mode: str = "review",
        source: str | None = None,
    ) -> ReviewOutcome:
        learn_mode = mode == "learn"
        self._stats["runs"] += 1
        if learn_mode:
            self._stats["learn_runs"] += 1
        # The read-before-write guard must be fresh for THIS pass. A read
        # performed by the foreground turn (whose context a spawned task
        # copies) must not authorize a background write.
        reset_skill_read_marks()
        transcript = Transcript(conversation_id=conversation_id or "")
        if conversation_id:
            try:
                transcript = await load_transcript(
                    conversation_id,
                    max_messages=int(getattr(cfg, "max_messages", 80) or 80),
                    max_chars=int(getattr(cfg, "max_transcript_chars", 60_000) or 60_000),
                    tool_preview_chars=int(getattr(cfg, "tool_preview_chars", 500) or 0),
                )
            except Exception as exc:  # noqa: BLE001 - no transcript, no writes
                if not learn_mode:
                    return ReviewOutcome(
                        review_id, conversation_id, status="failed", mode=mode,
                        error=f"transcript load failed: {_bounded(exc, 300)}",
                    )
                _LOG.warning("learn transcript load failed; continuing with sources: %s", exc)
        if not learn_mode:
            if transcript.empty or transcript.chars < int(getattr(cfg, "min_transcript_chars", 0) or 0):
                return ReviewOutcome(
                    review_id, conversation_id, status="skipped", mode=mode,
                    transcript_digest=transcript.digest,
                    error="transcript is empty or below the minimum review size",
                )
            # An event-time trigger does not always carry a channel, and a
            # direct command carries none at all. Re-check the durable
            # conversation so harness traffic can never be reviewed.
            if self._skip_source is not None:
                try:
                    row = await db.fetchone(
                        "SELECT channel FROM conversations WHERE id=?", (conversation_id,)
                    )
                    if row is not None and self._skip_source(
                        channel=row["channel"], conversation_id=conversation_id, source="review"
                    ):
                        return ReviewOutcome(
                            review_id, conversation_id, status="skipped", mode=mode,
                            transcript_digest=transcript.digest,
                            error="conversation is a harness/synthetic source",
                        )
                except Exception as exc:  # noqa: BLE001 - predicate failure is not a write
                    _LOG.warning("review conversation predicate failed: %s", exc)
        allow_network = bool(getattr(cfg, "learn_allow_network", True))
        tool_specs = self._tool_specs(learn=learn_mode, allow_network=allow_network)
        if not tool_specs:
            return ReviewOutcome(
                review_id, conversation_id, status="failed", mode=mode,
                transcript_digest=transcript.digest,
                error="required tools are not registered",
            )
        capture_memory = (
            not learn_mode
            and bool(getattr(cfg, "capture_memory", True))
            and self.learning is not None
        )
        if not capture_memory:
            tool_specs = [spec for spec in tool_specs if not str(spec.get("name", "")).startswith("memory_")]
        if self.router is None:
            return ReviewOutcome(
                review_id, conversation_id, status="failed", mode=mode,
                transcript_digest=transcript.digest,
                error="model router is unavailable",
            )

        instruction = (
            build_learn_prompt(str(source or ""), focus=focus)
            if learn_mode
            else build_review_prompt(capture_memory=capture_memory, focus=focus)
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=LEARN_SYSTEM_PROMPT if learn_mode else SYSTEM_PROMPT),
            *transcript.messages,
            ChatMessage(role="user", content=instruction),
        ]
        max_rounds = int(getattr(
            cfg, "learn_max_tool_rounds" if learn_mode else "max_tool_rounds",
            24 if learn_mode else 12,
        ) or 12)
        max_input_tokens = int(getattr(
            cfg, "learn_max_input_tokens" if learn_mode else "max_input_tokens",
            96_000 if learn_mode else 48_000,
        ) or 48_000)
        timeout = float(getattr(cfg, "timeout_seconds", 120.0) or 120.0)
        model = str(getattr(cfg, "model", "") or "").strip() or None
        allowed_scope = LEARN_SCOPE if learn_mode else REVIEW_SCOPE
        if learn_mode and not allow_network:
            allowed_scope = allowed_scope - {"http_get"}
        actor = LEARN_ACTOR if learn_mode else REVIEW_ACTOR
        actions: list[dict[str, Any]] = []
        created_skills: list[dict[str, Any]] = []
        staged_writes: list[dict[str, Any]] = []
        summary = ""
        rounds = 0
        budget_exhausted = False
        pending_names: list[str] = []
        cap_reason = ""

        async def loop() -> None:
            nonlocal summary, rounds, budget_exhausted, pending_names, cap_reason
            for index in range(max_rounds):
                pending_names = []
                if self._estimate_tokens(messages, tool_specs) > max_input_tokens:
                    budget_exhausted = True
                    cap_reason = "input token cap reached"
                    return
                request = ModelRequest(
                    messages=list(messages),
                    tools=list(tool_specs),
                    task_category="skill_learn" if learn_mode else "skill_review",
                    task_id=review_id,
                    agent_id="learn" if learn_mode else "review",
                    model=model,
                    max_tokens=4096,
                )
                response = await self.router.complete(request)
                rounds = index + 1
                summary = _bounded(getattr(response, "content", ""), _SUMMARY_LIMIT)
                raw_calls = list(getattr(response, "tool_calls", None) or [])
                if not raw_calls:
                    return
                messages.append(ChatMessage(
                    role="assistant",
                    content=str(getattr(response, "content", "") or ""),
                    tool_calls=[dict(item) if isinstance(item, Mapping) else item for item in raw_calls],
                ))
                for raw in raw_calls:
                    call_id, name, arguments = _tool_call(raw)
                    if name not in allowed_scope or (
                        not capture_memory and name.startswith("memory_")
                    ):
                        messages.append(self._tool_message(
                            call_id,
                            {"success": False, "error": f"tool {name!r} is not available in this review"},
                        ))
                        actions.append({"tool": name, "success": False, "decision": "out_of_scope"})
                        continue
                    if name == "memory_propose":
                        arguments.setdefault("conversation_id", conversation_id)
                        arguments.setdefault("evidence_ids", transcript.evidence_ids(8))
                    elif name == "skill_manage":
                        # Provenance for an authored skill; the manifest writer
                        # bounds and stores these values, it does not trust them.
                        if conversation_id:
                            arguments.setdefault("conversation_id", conversation_id)
                        arguments.setdefault("review_id", review_id)
                    result = await self._execute(
                        name, arguments, conversation_id, call_id,
                        actor=actor, scope=allowed_scope,
                    )
                    messages.append(self._tool_message(call_id, result))
                    action = {
                        "tool": name,
                        "mode": mode,
                        "arguments_digest": _digest(arguments)[:24],
                        "success": bool(result.get("success")),
                        "staged": bool(result.get("staged")),
                        "decision": str(result.get("decision") or ""),
                    }
                    actions.append(action)
                    if action["staged"] and str(result.get("proposal_id") or ""):
                        staged_writes.append({
                            "proposal_id": str(result.get("proposal_id")),
                            "skill": str(result.get("skill") or arguments.get("name") or ""),
                            "action": str(result.get("action") or arguments.get("action") or ""),
                            "category": _bounded(
                                result.get("category") or arguments.get("category"), 64
                            ),
                            "description": _bounded(
                                result.get("description") or arguments.get("description"), 160
                            ),
                        })
                    elif (
                        action["success"]
                        and name == "skill_manage"
                        and str(result.get("action") or arguments.get("action") or "") == "create"
                    ):
                        record = result.get("record")
                        record = record if isinstance(record, Mapping) else {}
                        created_skills.append({
                            "skill": str(result.get("skill") or arguments.get("name") or ""),
                            "category": _bounded(record.get("category") or arguments.get("category"), 64),
                            "description": _bounded(
                                record.get("description") or arguments.get("description"), 160
                            ),
                        })

                if index == max_rounds - 1:
                    # The last round's results were executed but the model never
                    # got a following call to synthesize them into an answer.
                    pending_names = [_tool_call(raw)[1] for raw in raw_calls]

        try:
            await asyncio.wait_for(loop(), timeout=timeout)
        except TimeoutError:
            return ReviewOutcome(
                review_id, conversation_id, status="failed", mode=mode, rounds=rounds,
                actions=actions, transcript_digest=transcript.digest,
                created=list(created_skills), staged=list(staged_writes),
                error=f"{mode} exceeded {timeout:.0f}s",
            )
        except Exception as exc:  # noqa: BLE001 - model/provider failure: no writes remain
            return ReviewOutcome(
                review_id, conversation_id, status="failed", mode=mode, rounds=rounds,
                actions=actions, transcript_digest=transcript.digest,
                created=list(created_skills), staged=list(staged_writes),
                error=_bounded(f"{type(exc).__name__}: {exc}", 400),
            )

        rate_limited = [a for a in actions if a.get("decision") == "rate_limited"]
        reasons = [
            cap_reason,
            "tool-round cap reached" if pending_names else "",
            "per-task tool-call cap reached" if rate_limited else "",
        ]
        reason = "; ".join(part for part in reasons if part)
        report = ""
        if reason:
            budget_exhausted = True
            report = cap_report(
                reason=reason,
                done=[str(a.get("tool") or "") for a in actions],
                unfinished=[
                    "final synthesis of results from " + ", ".join(pending_names[:10])
                ] if pending_names else [],
                unfinished_label="unfinished work",
            )
        self._count_actions(actions)
        return ReviewOutcome(
            review_id=review_id,
            conversation_id=conversation_id,
            status="completed",
            mode=mode,
            rounds=rounds,
            actions=actions,
            created=list(created_skills),
            transcript_digest=transcript.digest,
            staged=list(staged_writes),
            summary=summary,
            budget_exhausted=budget_exhausted,
            progress_report=report,
        )

    async def _execute(
        self,
        name: str,
        arguments: dict[str, Any],
        conversation_id: str,
        call_id: str,
        *,
        actor: str = REVIEW_ACTOR,
        scope: frozenset[str] = REVIEW_SCOPE,
    ) -> dict[str, Any]:
        call = ToolCall(
            id=call_id,
            name=name,
            arguments=arguments,
            actor=actor,
            correlation_id=conversation_id or None,
        )
        try:
            result = await self.executor.execute(call, scope=scope)
        except ApprovalRequired as exc:
            return {
                "success": False, "decision": "pending",
                "approval_id": getattr(exc, "approval_id", None),
                "error": _bounded(exc, 300),
            }
        except Exception as exc:  # noqa: BLE001 - tool errors are data for the next round
            return {"success": False, "error": _bounded(f"{type(exc).__name__}: {exc}", 300)}
        if hasattr(result, "model_dump"):
            payload = dict(result.model_dump(mode="json"))
        else:  # pragma: no cover - defensive
            payload = {"success": bool(getattr(result, "success", False))}
        # The handler's own result is nested under ``output``; the loop reads
        # ``staged`` from the top level. Without this lift a staged skill write
        # was counted as an applied one and the artifact said ``staged: false``
        # for exactly the writes that still needed an operator.
        output = payload.get("output")
        if isinstance(output, Mapping):
            for key in ("staged", "proposal_id", "skill", "action", "category", "description"):
                if key in output and key not in payload:
                    payload[key] = output[key]
        return payload

    @staticmethod
    def _tool_message(call_id: str, payload: Mapping[str, Any]) -> ChatMessage:
        encoded = json.dumps(dict(payload), ensure_ascii=False, default=str)
        return ChatMessage(
            role="tool",
            content=encoded[:_TOOL_MESSAGE_LIMIT],
            tool_call_id=call_id,
        )

    def _tool_specs(
        self, *, learn: bool = False, allow_network: bool = True
    ) -> list[dict[str, Any]]:
        registry = getattr(self.executor, "registry", None)
        if registry is None or not hasattr(registry, "list_specs"):
            return []
        scope = LEARN_SCOPE if learn else REVIEW_SCOPE
        specs = []
        for spec in registry.list_specs():
            name = str(getattr(spec, "name", ""))
            if name not in scope:
                continue
            if name.startswith("memory_") and self.learning is None:
                continue
            if learn and not allow_network and name in {"http_get"}:
                continue
            specs.append(spec.model_dump(mode="json"))
        return specs

    @staticmethod
    def _estimate_tokens(messages: list[ChatMessage], tool_specs: list[dict[str, Any]]) -> int:
        chars = sum(len(message.content or "") for message in messages)
        chars += len(json.dumps(tool_specs, default=str))
        return chars // 4

    def _count_actions(self, actions: list[dict[str, Any]]) -> None:
        for action in actions:
            if not action.get("success"):
                continue
            tool = action.get("tool")
            if tool == "skill_manage":
                if action.get("staged"):
                    self._stats["skills_staged"] += 1
                else:
                    self._stats["skills_written"] += 1
                    if action.get("mode") == "learn":
                        self._stats["learn_skills"] += 1
            elif tool == "memory_propose":
                self._stats["memories_proposed"] += 1

    # -- completion / artifact / audit --------------------------------------

    async def _finish(self, outcome: ReviewOutcome, cfg: Any) -> None:
        self._stats["last_run_at"] = utcnow()
        self._stats["last_outcome"] = outcome.status
        if outcome.status == "completed":
            self._stats["completed"] += 1
        elif outcome.status == "failed":
            self._stats["failed"] += 1
        else:
            self._stats["skipped"] += 1
        outcome.artifact_path = self._write_artifact(outcome)
        await self._audit(outcome)
        prefix = "learn" if outcome.mode == "learn" else "review"
        topic = f"{prefix}.{outcome.status}"
        try:
            await self.events.emit(
                topic,
                review_id=outcome.review_id,
                conversation_id=outcome.conversation_id,
                status=outcome.status,
                mode=outcome.mode,
                actions=len(outcome.actions),
                # Named for a notifier: applied creations and ids an operator
                # still has to decide on, bounded so a long run cannot bloat
                # the event payload.
                created=outcome.created[:10],
                staged=outcome.staged[:10],
                error=_safe_text(outcome.error, 200) if outcome.error else None,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail the run
            _LOG.warning("could not emit review event: %s", exc)

    def _write_artifact(self, outcome: ReviewOutcome) -> str | None:
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            path = self.artifact_dir / f"{outcome.review_id}.json"
            payload = {
                "review_id": outcome.review_id,
                "conversation_id": outcome.conversation_id,
                "mode": outcome.mode,
                "status": outcome.status,
                "rounds": outcome.rounds,
                "actions": outcome.actions,
                "created": outcome.created,
                "staged": outcome.staged,
                "transcript_digest": outcome.transcript_digest,
                "summary": _safe_text(outcome.summary),
                "error": _safe_text(outcome.error, 400) if outcome.error else None,
                "budget_exhausted": outcome.budget_exhausted,
                "progress_report": _safe_text(outcome.progress_report, 1_200),
                "created_at": utcnow(),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            return str(path)
        except Exception as exc:  # noqa: BLE001 - an artifact failure is not a write failure
            _LOG.warning("could not write review artifact: %s", exc)
            return None

    async def _audit(self, outcome: ReviewOutcome) -> None:
        try:
            await db.execute(
                "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,"
                "correlation_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    utcnow(),
                    LEARN_ACTOR if outcome.mode == "learn" else REVIEW_ACTOR,
                    "learn.run" if outcome.mode == "learn" else "review.run",
                    outcome.conversation_id,
                    "low", outcome.status,
                    _safe_text(outcome.error or outcome.status, 200),
                    json.dumps({
                        "review_id": outcome.review_id,
                        "transcript_digest": outcome.transcript_digest,
                        "actions": outcome.actions,
                        "artifact": outcome.artifact_path,
                        "rounds": outcome.rounds,
                        "budget_exhausted": outcome.budget_exhausted,
                        "progress_report": _safe_text(outcome.progress_report, 1_200),
                    }, default=str),
                    outcome.conversation_id,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - audit failure is logged, not hidden
            _LOG.warning("could not audit review run: %s", exc)

    def status(self) -> dict[str, Any]:
        cfg = self.config
        return {
            **self._stats,
            "enabled": self.enabled,
            "interval_turns": int(getattr(cfg, "interval_turns", 0) or 0),
            "capture_memory": bool(getattr(cfg, "capture_memory", True)),
            "pending_conversations": len(self._pending),
            "running_conversations": len(self._running),
        }


__all__ = [
    "LEARN_ACTOR",
    "LEARN_SCOPE",
    "REVIEW_ACTOR",
    "REVIEW_SCOPE",
    "SOURCE_TOOLS",
    "ReviewEngine",
    "ReviewOutcome",
]
