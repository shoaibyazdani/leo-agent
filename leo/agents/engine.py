"""Agent delegation engine: durable tasks, dependency-aware fan-out, verification.

Only one agent is used for a simple objective. Multiple agents are used when a
caller explicitly requests a fan-out, or when a batch of genuinely independent
objectives is supplied. Every run is backed by a task in the injected task
engine, recorded in ``agent_runs``, and (by default) independently verified.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .. import db
from ..config import Settings
from ..contracts import (
    AgentResult,
    AgentSpec,
    ChatMessage,
    ModelRequest,
    ModelResponse,
    TaskRecord,
    TaskStatus,
    ToolCall,
    new_id,
    utcnow,
)
from ..events import EventBus
from ..logging import get_logger
from .isolation import Worktree, WorktreeIsolation
from .registry import CORE_SPECIALIST_COUNT, SPECIALIST_SPECS, AgentRegistry
from .verification import VerificationEngine, VerificationResult, digest as evidence_digest, invoke_callable

_LOG = get_logger("agents")

# These keys are reserved for the engine/permissions layer. A delegating model
# or tool caller must not be able to inject approval flags, service objects or
# the engine's scoped tool facade into a specialist handler context.
RESERVED_DELEGATION_KEYS: frozenset[str] = frozenset({
    "apply", "confirm", "approved_by", "approval_id", "apply_expiry",
    "services", "permissions", "settings", "tools",
})

# Bounded tool loop for model workers. A model may call tools repeatedly, but
# every round must re-send the assistant(tool_calls) message and the matching
# tool messages, and it can never loop forever. The bound must still fit real
# work: at 4 rounds a worker that listed a directory, read one file and searched
# twice ran out mid-investigation and the whole task failed on the ceiling
# rather than on anything being wrong. 12 keeps the loop finite while leaving
# room for read -> act -> verify.
MAX_MODEL_TOOL_ROUNDS = 12


class AgentError(RuntimeError):
    """Base error for delegation failures."""


class AgentLimitError(AgentError):
    """A depth, concurrency or task-count limit was exceeded."""


def _json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


class _DurableTasks:
    """Minimal SQLite task engine used when no task engine is injected.

    The real task engine can be injected by the composition root; this fallback
    keeps the agent engine operational and durable on its own.
    """

    _JSON_FIELDS = {"plan", "history", "dependencies", "checkpoints", "errors", "evidence", "metadata", "resource_limits", "result"}

    @staticmethod
    def _row_to_task(row: Any) -> TaskRecord:
        data: dict[str, Any] = {}
        for key in TaskRecord.model_fields:
            if key not in row.keys():
                continue
            value = row[key]
            if key in _DurableTasks._JSON_FIELDS and value is not None:
                try:
                    value = json.loads(value)
                except (TypeError, ValueError):
                    value = None if key == "result" else value
            if key == "cancellation_requested":
                value = bool(value)
            data[key] = value
        return TaskRecord.model_validate(data)

    async def create(self, objective: str, **kwargs: Any) -> TaskRecord:
        task = TaskRecord(objective=str(objective))
        allowed = set(TaskRecord.model_fields)
        for key, value in kwargs.items():
            if key in allowed:
                setattr(task, key, value)
        await db.execute(
            "INSERT INTO tasks(id,objective,status,plan,current_step,history,dependencies,checkpoints,"
            "result,errors,retry_count,resource_limits,cancellation_requested,owner_agent,created_at,"
            "updated_at,completed_at,evidence,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                task.id, task.objective, task.status.value, _json(task.plan), task.current_step,
                _json(task.history), _json(task.dependencies), _json(task.checkpoints),
                _json(task.result), _json(task.errors), task.retry_count, _json(task.resource_limits),
                int(task.cancellation_requested), task.owner_agent, task.created_at, task.updated_at,
                task.completed_at, _json(task.evidence), _json(task.metadata),
            ),
        )
        return task

    async def save(self, task: TaskRecord) -> None:
        await db.execute(
            "UPDATE tasks SET status=?, result=?, errors=?, history=?, evidence=?, metadata=?, "
            "cancellation_requested=?, completed_at=?, updated_at=? WHERE id=?",
            (
                task.status.value if isinstance(task.status, TaskStatus) else str(task.status),
                _json(task.result), _json(task.errors), _json(task.history), _json(task.evidence),
                _json(task.metadata), int(task.cancellation_requested), task.completed_at, utcnow(), task.id,
            ),
        )

    async def complete(
        self, task_id: str, result: Any, evidence: list[dict[str, Any]], note: str | None = None
    ) -> TaskRecord:
        """Same evidence gate as the durable engine, for the fallback path."""
        if not isinstance(evidence, (list, tuple)) or not evidence:
            raise ValueError("completion requires non-empty evidence")
        if not all(isinstance(item, dict) for item in evidence):
            raise ValueError("evidence entries must be dicts")
        task = await self.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if task.status.terminal or task.cancellation_requested:
            raise ValueError("cannot complete a terminal or cancelled task")
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.evidence = [dict(item) for item in evidence]
        task.completed_at = utcnow()
        if note:
            task.history.append({"event": "completed", "note": note})
        await self.save(task)
        return task

    async def fail(self, task_id: str, error: str) -> TaskRecord:
        task = await self.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if task.status.terminal:
            return task
        task.status = TaskStatus.FAILED
        task.errors.append(str(error))
        task.result = AgentResult(agent_id="unknown", task_id=task_id, success=False, error=str(error))
        await self.save(task)
        return task

    async def update(
        self,
        task_id: str,
        *,
        result: Any = None,
        metadata: dict[str, Any] | None = None,
        status: TaskStatus | None = None,
        note: str | None = None,
    ) -> TaskRecord:
        """Merge metadata/result; mirrors the durable engine's update()."""
        task = await self.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if metadata:
            task.metadata.update(metadata)
        if result is not None:
            task.result = result
        if status is not None and TaskStatus(status) is not task.status and not task.status.terminal:
            task.status = TaskStatus(status)
            task.history.append({
                "event": "transition", "to": TaskStatus(status).value, "note": note or "updated by caller",
            })
        await self.save(task)
        return task

    async def cancel(self, task_id: str) -> bool:
        """Request cancellation; terminal tasks are never flipped.

        The flag is persisted (see :meth:`save`) so a running worker that
        reloads the task observes it; non-running tasks cancel immediately.
        """
        task = await self.get(task_id)
        if task is None or task.status.terminal:
            return False
        if not task.cancellation_requested:
            task.cancellation_requested = True
            task.history.append({"event": "cancellation_requested", "note": "cancellation requested by caller"})
        if task.status not in (TaskStatus.RUNNING, TaskStatus.VERIFYING):
            task.status = TaskStatus.CANCELLED
            task.completed_at = task.completed_at or utcnow()
        await self.save(task)
        return True

    async def run(self, task_id: str, worker: Callable[..., Any]) -> TaskRecord:
        task = await self.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if task.status.terminal or task.cancellation_requested:
            return task  # never replay or run a cancelled task
        task.status = TaskStatus.RUNNING
        task.updated_at = utcnow()
        await self.save(task)
        try:
            result = await worker(task)
            value = result.result if isinstance(result, TaskRecord) else result
            task.result = value
            success = value if isinstance(value, bool) else (
                bool(value.success) if isinstance(value, BaseModel) and hasattr(value, "success")
                else bool(value.get("success", True)) if isinstance(value, Mapping) else True
            )
            collected = list(getattr(value, "evidence", []) or []) if not isinstance(value, Mapping) else list(value.get("evidence") or [])
            if success and task.cancellation_requested:
                task.status = TaskStatus.CANCELLED
            elif success and collected:
                task.status = TaskStatus.COMPLETED
                task.evidence = [dict(item) if isinstance(item, dict) else {"claim": str(item)} for item in collected]
            elif success:
                task.status = TaskStatus.FAILED
                task.errors.append("completion requires evidence")
            else:
                task.status = TaskStatus.FAILED
            task.result = value
        except Exception as exc:  # worker-level failures are durable too
            task.status = TaskStatus.FAILED
            task.errors.append(str(exc))
            task.result = AgentResult(
                agent_id=task.owner_agent or "unknown", task_id=task_id,
                success=False, error=str(exc),
            )
        task.completed_at = utcnow()
        await self.save(task)
        return task

    async def get(self, task_id: str) -> TaskRecord | None:
        row = await db.fetchone("SELECT * FROM tasks WHERE id=?", (task_id,))
        return self._row_to_task(row) if row else None


@dataclass
class AgentRequest:
    """One unit of delegated work. Accepts either ``dependencies`` or ``depends_on``."""

    objective: str
    agent_id: str | None = None
    id: str = field(default_factory=lambda: new_id("req"))
    dependencies: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None
    verify: bool = True
    isolation: bool = False
    expected_output: Any = None
    second_agent: str | None = None
    second_model: str | None = None
    label: str | None = None


def detect_conflicts(results: Sequence[AgentResult]) -> list[dict[str, Any]]:
    """Report results that disagree about the same objective."""
    groups: dict[str, list[AgentResult]] = {}
    for result in results:
        groups.setdefault(_canonical(result.output), []).append(result)
    if len(groups) <= 1:
        return []
    ordered = sorted(groups.items(), key=lambda item: -len(item[1]))
    winner, winners = ordered[0]
    conflicts: list[dict[str, Any]] = []
    for value, group in ordered[1:]:
        conflicts.append({
            "type": "output_conflict",
            "outputs": [{"agent_id": r.agent_id, "task_id": r.task_id, "output": r.output} for r in group],
            "conflicts_with": [{"agent_id": r.agent_id, "task_id": r.task_id} for r in winners],
            "winner_output": winners[0].output if winner else None,
        })
    return conflicts


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


# Automatic routing is limited to the twelve stable core specialists. Extra
# executable SEEDs are still callable by explicit agent_id and remain bound to
# their handlers, but a capability in a new SEED must not silently hijack an
# objective that previously routed to a core role.
_CORE_AGENT_IDS: frozenset[str] = frozenset(
    str(spec["id"]) for spec in SPECIALIST_SPECS[:CORE_SPECIALIST_COUNT]
)

_KEYWORD_AGENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("test", "qa", "regression", "verify tests"), "qa"),
    (("security", "vulnerab", "secret", "threat"), "security"),
    (("deploy", "ci", "pipeline", "devops", "infrastructure"), "devops"),
    (("research", "find sources", "literature", "investigate"), "research"),
    (("browser", "web page", "click", "navigate"), "browser"),
    (("computer", "desktop", "screenshot", "shell command"), "computer"),
    (("database", "sql", "analytics", "chart", "statistics"), "data"),
    (("document", "readme", "docs", "changelog"), "documentation"),
    (("integrate", "api client", "adapter", "webhook"), "integration"),
    (("architect", "design", "module boundary", "trade-off"), "architecture"),
    (("code", "implement", "refactor", "bug", "function", "python"), "coding"),
)


class ScopedTools:
    """A tool view limited to one agent's declared scope.

    An entry in ``AgentSpec.tool_permissions`` matches a tool when it equals
    the tool's name or its registered category; everything else is denied by
    default and audited by the executor. Handlers and the default model
    worker receive this façade instead of the raw executor, so neither can
    widen its own scope.
    """

    def __init__(self, tools: Any, allow: frozenset[str], agent_id: str) -> None:
        self._tools = tools
        self.allow = allow
        self.agent_id = agent_id

    def specs(self) -> list[Any]:
        """In-scope tool specs, for model advertisement."""
        registry = getattr(self._tools, "registry", None)
        list_specs = getattr(registry, "list_specs", None)
        if not callable(list_specs):
            return []
        return [
            spec for spec in list_specs()
            if str(getattr(spec, "name", "")) in self.allow
            or str(getattr(spec, "category", "")) in self.allow
        ]

    async def execute(self, call: Any, approval_id: str | None = None) -> Any:
        """Execute an in-scope call; out-of-scope calls are denied and audited."""
        return await self._tools.execute(call, approval_id=approval_id, scope=self.allow)


class AgentEngine:
    """Durable, verification-aware multi-agent delegation engine."""

    def __init__(
        self,
        registry: AgentRegistry,
        tasks: Any = None,
        router: Any = None,
        tools: Any = None,
        events: EventBus | None = None,
        settings: Any = None,
        *,
        handlers: Mapping[str, Callable[..., Any]] | None = None,
        verifier: VerificationEngine | None = None,
        max_depth: int | None = None,
    ) -> None:
        self.registry = registry
        self.tasks = tasks if tasks is not None else _DurableTasks()
        self.router = router
        self.tools = tools
        self.events = events if events is not None else EventBus()
        self.settings = settings if settings is not None else Settings()
        self._handlers: dict[str, Callable[..., Any]] = dict(handlers or {})
        self.verifier = verifier or VerificationEngine(
            registry=registry, router=router, settings=self.settings, tools=tools
        )
        self.max_depth = max_depth if max_depth is not None else self._depth_from_settings()
        self._active: dict[str, int] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._shared: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._results: dict[str, AgentResult] = {}
        self._message_subscription = False
        if hasattr(self.events, "subscribe") and not self._message_subscription:
            self.events.subscribe("agent.message", self._on_message)
            self._message_subscription = True

    def set_handler(self, agent_id: str, handler: Callable[..., Any]) -> None:
        """Inject a process-local worker for an agent (used for tests and local adapters)."""
        self._handlers[agent_id] = handler

    def _depth_from_settings(self) -> int:
        security = getattr(self.settings, "security", None)
        limit = getattr(security, "max_agent_depth", None)
        if limit is None and isinstance(self.settings, Mapping):
            limit = dict(self.settings.get("security", {})).get("max_agent_depth")
        try:
            return max(1, int(limit or 3))
        except (TypeError, ValueError):
            return 3

    async def _on_message(self, event: Any) -> None:
        payload = dict(getattr(event, "payload", event) or {})
        correlation = str(payload.get("correlation_id") or "*")
        self._messages.setdefault(correlation, []).append(payload)
        shared = self._shared.get(correlation)
        if shared is not None:
            shared.setdefault("messages", []).append(payload)

    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: Any,
        *,
        task_id: str | None = None,
        correlation_id: str | None = None,
        **extra: Any,
    ) -> Any:
        """Agent-to-agent communication on the durable ``agent.message`` topic."""
        payload: dict[str, Any] = {
            "from_agent": from_agent, "to_agent": to_agent, "content": content,
            "task_id": task_id, "correlation_id": correlation_id, "created_at": utcnow(),
            **extra,
        }
        event = await self.events.emit("agent.message", **payload)
        self._messages.setdefault(correlation_id or "*", []).append(payload)
        return event

    # --- request parsing and graph construction -------------------------

    def _known_reference(self, value: Any) -> str | None:
        if isinstance(value, AgentResult):
            return value.task_id if value.task_id in self._results else None
        if isinstance(value, TaskRecord):
            return value.id
        if isinstance(value, str) and (value in self._results or value.startswith("task_")):
            return value
        return None

    def _coerce_request(
        self,
        item: Any,
        *,
        default_agent: str | None,
        context: dict[str, Any],
        verify: bool,
        isolation: bool,
    ) -> AgentRequest:
        if isinstance(item, AgentRequest):
            # Copy rather than mutate: a caller may reuse its request across
            # calls. Honour an explicit ``verify=False`` just like the mapping
            # form does; a delegation-level default must never silently turn
            # verification back on for this particular request.
            request = replace(item)
            request.context = {**context, **item.context}
            if request.agent_id is None:
                request.agent_id = default_agent
            return request
        if isinstance(item, Mapping):
            objective = item.get("objective", item.get("task", item.get("description")))
            if objective is None:
                raise ValueError(f"request has no objective: {item}")
            deps = item.get("dependencies", item.get("depends_on", [])) or []
            if isinstance(deps, str):
                deps = [deps]
            merged_context = {**context, **(item.get("context") or {})}
            return AgentRequest(
                objective=str(objective),
                agent_id=item.get("agent_id") or default_agent,
                id=str(item.get("id") or new_id("req")),
                dependencies=[str(d) for d in deps],
                context=merged_context,
                handler=item.get("handler"),
                verify=bool(item.get("verify", verify)),
                isolation=bool(item.get("isolation", isolation)),
                expected_output=item.get("expected_output"),
                second_agent=item.get("second_agent"),
                second_model=item.get("second_model"),
                label=item.get("label"),
            )
        if isinstance(item, str):
            return AgentRequest(objective=item, agent_id=default_agent, context=dict(context), verify=verify, isolation=isolation)
        raise TypeError(f"unsupported delegation request: {type(item)!r}")

    @staticmethod
    def _topological_waves(requests: Sequence[AgentRequest]) -> list[list[AgentRequest]]:
        by_id = {request.id: request for request in requests}
        pending = {request.id: set(request.dependencies) & set(by_id) for request in requests}
        waves: list[list[AgentRequest]] = []
        emitted: set[str] = set()
        while len(emitted) < len(requests):
            wave = [r for r in requests if r.id not in emitted and not (pending[r.id] - emitted)]
            if not wave:
                raise AgentError("cyclic agent dependency graph")
            waves.append(wave)
            emitted.update(r.id for r in wave)
        return waves

    @staticmethod
    def _capability_score(objective: str, capability: Any) -> int:
        """Score one declared capability against an objective, whole words only."""
        text = str(objective).lower()
        cap = str(capability).strip().lower()
        if not cap:
            return 0
        score = 0
        if len(cap) > 2 and re.search(rf"(?<![a-z0-9]){re.escape(cap)}(?![a-z0-9])", text):
            score += 3  # exact capability phrase is the strongest signal
        # Short fragments such as "root" in "root_cause_candidates" are not
        # a capability signal; they make unrelated objectives misroute.
        for term in (part for part in re.split(r"[_\-\s]+", cap) if len(part) >= 5):
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):
                score += 1
        return score

    async def _select_agent(self, objective: str, context: Mapping[str, Any]) -> str:
        if hasattr(self.router, "select_agent"):
            chosen = await self.router.select_agent(objective, context=dict(context))
            if chosen:
                return str(chosen)
        # Capabilities declared on live, enabled agents drive selection first.
        # Ties are broken by agent id, so a restart or registry reorder can
        # never pick a different specialist for the same objective.
        candidates: list[tuple[int, str]] = []
        seed_ids = getattr(self.registry, "loaded_specialist_ids", set())
        for spec in await self.registry.list(enabled_only=True):
            if spec.id in seed_ids and spec.id not in _CORE_AGENT_IDS:
                continue
            score = sum(self._capability_score(objective, cap) for cap in (spec.capabilities or []))
            if score > 0:
                candidates.append((-score, str(spec.id)))
        if candidates:
            return sorted(candidates)[0][1]
        text = objective.lower()
        for keywords, agent_id in _KEYWORD_AGENTS:
            if any(word in text for word in keywords):
                if await self.registry.get(agent_id):
                    return agent_id
        return "general"

    # --- delegation -----------------------------------------------------

    async def delegate(
        self,
        objective: Any,
        agent_id: str | None = None,
        dependencies: Sequence[Any] | None = None,
        context: Mapping[str, Any] | None = None,
        parallel: bool = False,
        verify: bool = True,
        isolation: bool = False,
    ) -> AgentResult | list[AgentResult]:
        outer_context = dict(context or {})
        try:
            depth = int(outer_context.get("depth", 0) or 0)
        except (TypeError, ValueError):
            depth = 0
        # Nesting is driven from zero upward: a caller-supplied depth must not
        # bypass the limit with a negative (or malformed) value.
        depth = max(0, depth)
        if depth >= self.max_depth:
            raise AgentLimitError(f"maximum agent depth {self.max_depth} exceeded at depth {depth}")
        correlation_id = str(outer_context.get("correlation_id") or new_id("orch"))
        shared = self._shared.setdefault(correlation_id, {
            "correlation_id": correlation_id, "results": {}, "messages": [], "artifacts": {},
        })
        for key, value in (outer_context.get("shared_state") or {}).items():
            shared.setdefault(key, value)
        outer_context.setdefault("correlation_id", correlation_id)

        batch = isinstance(objective, (list, tuple))
        items = list(objective) if batch else [objective]
        roots: list[AgentRequest] = []
        subrequests: list[AgentRequest] = []
        for item in items:
            selected = agent_id or await self._select_agent(str(item), outer_context)
            roots.append(self._coerce_request(item, default_agent=selected, context=outer_context, verify=verify, isolation=isolation))

        for dependency in dependencies or []:
            reference = self._known_reference(dependency)
            if reference:
                for root in roots:
                    if reference not in root.dependencies:
                        root.dependencies.append(reference)
                continue
            dep_request = self._coerce_request(
                dependency, default_agent=None, context=outer_context, verify=verify, isolation=isolation,
            )
            if dep_request.agent_id is None:
                dep_request.agent_id = await self._select_agent(dep_request.objective, outer_context)
            subrequests.append(dep_request)
            for root in roots:
                if dep_request.id not in root.dependencies:
                    root.dependencies.append(dep_request.id)

        request_list = subrequests + roots
        by_id = {request.id: request for request in request_list}
        usage: dict[str, int] = {}
        results_by_request: dict[str, AgentResult] = {}
        outcomes: list[Any] = []

        # Known task references are resolved into results before the waves
        # run: a dependency on prior work must reach the handler (context and
        # evidence), not merely be recorded on the task row.
        for request in request_list:
            for dep in request.dependencies:
                if dep in by_id or dep in results_by_request:
                    continue
                known = await self._task_reference(dep)
                if known is None:
                    raise AgentError(f"unknown dependency: {dep}")
                results_by_request[dep] = self._prior_result(dep, known)

        root_task = await self._create_task(
            _json(objective) if batch else str(objective),
            metadata={
                "correlation_id": correlation_id, "depth": depth, "shared_state": shared,
                "parallel": parallel, "agent_ids": sorted({r.agent_id or "" for r in request_list}),
            },
            owner_agent="orchestrator",
        )
        shared["root_task_id"] = root_task.id

        for wave in self._topological_waves(request_list):
            concurrent = len(wave) > 1 and (parallel or (batch and not any(
                set(r.dependencies) & {other.id for other in wave} for r in wave
            )))
            if concurrent:
                wave_results = await asyncio.gather(
                    *(self._run_request(
                        request, depth=depth, correlation_id=correlation_id, shared=shared,
                        root_task_id=root_task.id, results_by_request=results_by_request, usage=usage,
                    ) for request in wave),
                    return_exceptions=True,
                )
            else:
                wave_results = []
                for request in wave:
                    try:
                        wave_results.append(await self._run_request(
                            request, depth=depth, correlation_id=correlation_id, shared=shared,
                            root_task_id=root_task.id, results_by_request=results_by_request, usage=usage,
                        ))
                    except Exception as exc:
                        wave_results.append(self._failure(request.agent_id or "unknown", "", exc))
            for request, value in zip(wave, wave_results):
                result = self._failure(request.agent_id or "unknown", "", value) if isinstance(value, BaseException) else value
                results_by_request[request.id] = result
                self._results[result.task_id] = result
                shared["results"][request.id] = result.model_dump(mode="json")
                outcomes.append((request, result))

        root_results = [results_by_request[r.id] for r in roots if r.id in results_by_request]
        conflicts = detect_conflicts([r for r in root_results if r.success])
        for request, result in outcomes:
            result.metadata = dict(getattr(result, "metadata", {}) or {})
            result.metadata["conflicts"] = conflicts
        # A conflict is not metadata to be reported alongside a successful
        # answer: contradictory peer outputs must fail closed instead of being
        # silently presented as an agreed result.
        if conflicts and roots:
            reason = f"conflicting peer outputs ({len(conflicts)} conflict group(s)); delegation failed closed"
            for result in root_results:
                result.success = False
                result.error = result.error or reason
        all_ok = bool(root_results) and all(r.success for r in root_results) and not conflicts
        result_payload = [r.model_dump(mode="json") for r in root_results]
        await self._update_task(
            root_task.id,
            result=result_payload,
            metadata={
                "correlation_id": correlation_id, "depth": depth, "shared_state": shared,
                "conflicts": conflicts, "completed": all_ok,
                "result_task_ids": [r.task_id for r in root_results],
            },
            status=TaskStatus.READY,
        )
        # The same evidence gate applies to delegated work: a model reporting
        # success is not sufficient to mark a task COMPLETED.
        if all_ok:
            # The root is evidence-gated too: only genuine agent evidence may
            # complete it. A fabricated "agent returned success" claim would
            # defeat the gate for results that carried no evidence at all.
            evidence: list[dict[str, Any]] = []
            for r in root_results:
                evidence.extend(dict(item) for item in r.evidence if isinstance(item, dict))
            if evidence:
                await self.tasks.complete(root_task.id, result_payload, evidence)
            else:
                await self.tasks.fail(root_task.id, "delegation completed without evidence")
        else:
            await self.tasks.fail(
                root_task.id,
                "delegation failed: " + ", ".join(
                    f"{r.agent_id}: {r.error or 'unsuccessful'}" for r in root_results if not r.success
                ),
            )
        return root_results if batch else root_results[0]

    async def _task_reference(self, reference: str) -> TaskRecord | None:
        return await self.tasks.get(reference)

    def _prior_result(self, reference: str, known: TaskRecord) -> AgentResult:
        """Reconstruct a dependency result from a prior task record.

        Prefers the live in-process result, then a persisted ``AgentResult``
        payload, then a minimal view of the task outcome — the handler must
        see the dependency either way.
        """
        prior = self._results.get(reference)
        if isinstance(prior, AgentResult):
            return prior
        payload = known.result if isinstance(known.result, Mapping) else None
        if isinstance(payload, Mapping) and "agent_id" in payload and "success" in payload:
            try:
                return AgentResult.model_validate(dict(payload))
            except Exception:  # noqa: BLE001 - fall through to the minimal view
                pass
        return AgentResult(
            agent_id=str(known.owner_agent or "prior"),
            task_id=known.id,
            success=known.status is TaskStatus.COMPLETED,
            output=known.result,
            evidence=[dict(item) for item in (known.evidence or []) if isinstance(item, dict)],
            error=(known.errors[-1] if known.errors else None),
        )

    def _failure(self, agent_id: str, task_id: str, error: Any) -> AgentResult:
        return AgentResult(agent_id=agent_id, task_id=task_id, success=False, error=str(error))

    async def _create_task(self, objective: str, *, metadata: dict[str, Any], owner_agent: str | None = None, dependencies: Sequence[str] = ()) -> TaskRecord:
        kwargs = {"metadata": metadata, "owner_agent": owner_agent, "dependencies": list(dependencies)}
        try:
            return await self.tasks.create(objective, **kwargs)
        except TypeError:
            return await self.tasks.create(objective)

    async def _update_task(self, task_id: str, *, result: Any, metadata: dict[str, Any], status: TaskStatus) -> None:
        """Record delegation bookkeeping through the task engine.

        Prefers the engine's public ``update()`` (legal transitions, merged
        metadata, compare-and-swapped saves). Raw SQL is never used: it
        previously replaced metadata wholesale, bypassed the transition
        table, and stamped ``completed_at`` on a live task.
        """
        update = getattr(self.tasks, "update", None)
        if callable(update):
            try:
                outcome = update(
                    task_id, result=result, metadata=metadata, status=status,
                    note="delegation bookkeeping",
                )
                if asyncio.iscoroutine(outcome):
                    await outcome
            except Exception as exc:
                _LOG.debug("task update failed", extra={"task_id": task_id, "error": str(exc)})
            return
        # Minimal engines without update(): merge in memory and persist via
        # whatever save hook exists. Never write around the task engine.
        task = await self.tasks.get(task_id)
        if task is None:
            return
        task.result = result
        task.metadata.update(metadata)
        for method_name in ("save", "persist"):
            method = getattr(self.tasks, method_name, None)
            if not callable(method):
                continue
            try:
                outcome = method(task)
                if asyncio.iscoroutine(outcome):
                    await outcome
            except Exception as exc:
                _LOG.debug("task metadata update failed", extra={"task_id": task_id, "error": str(exc)})
            break

    # --- one agent run --------------------------------------------------

    async def _run_request(
        self,
        request: AgentRequest,
        *,
        depth: int,
        correlation_id: str,
        shared: dict[str, Any],
        root_task_id: str,
        results_by_request: dict[str, AgentResult],
        usage: dict[str, int],
    ) -> AgentResult:
        agent_id = request.agent_id or "general"
        spec = await self.registry.get(agent_id)
        if spec is None or not spec.enabled:
            raise AgentError(f"agent not available: {agent_id}")
        if usage.get(spec.id, 0) >= max(1, spec.task_limit):
            raise AgentLimitError(f"agent {spec.id} reached its task limit of {spec.task_limit}")
        usage[spec.id] = usage.get(spec.id, 0) + 1
        sem = self._semaphores.setdefault(spec.id, asyncio.Semaphore(max(1, spec.task_limit)))

        dependency_results = [
            results_by_request[dep].model_dump(mode="json")
            for dep in request.dependencies if dep in results_by_request
        ]
        task = await self._create_task(
            request.objective,
            metadata={
                "correlation_id": correlation_id, "depth": depth, "parent_task_id": root_task_id,
                "agent_id": spec.id, "dependencies": request.dependencies, "shared_state": shared,
                "label": request.label,
            },
            owner_agent=spec.id,
            dependencies=request.dependencies,
        )
        run_id = await self.registry.start_run(
            spec.id, task_id=task.id,
            metadata={"correlation_id": correlation_id, "depth": depth, "root_task_id": root_task_id},
        )
        shared.setdefault("tasks", {})[request.id] = task.id
        worktree: Worktree | None = None
        isolation_manager: WorktreeIsolation | None = None

        async def worker(task_arg: Any = None, task_engine: Any = None) -> AgentResult:
            nonlocal worktree, isolation_manager
            actual_task_id = getattr(task_arg, "id", None) or task.id
            evidence: list[Any] = []
            scoped = self._scoped_tools(spec)
            handler_context: dict[str, Any] = {
                **{
                    key: value for key, value in dict(request.context).items()
                    if key not in RESERVED_DELEGATION_KEYS
                },
                "correlation_id": correlation_id,
                "depth": depth,
                "shared_state": shared,
                "task_id": actual_task_id,
                "parent_task_id": root_task_id,
                "agent_id": spec.id,
                "agent": spec,
                "dependencies": dependency_results,
                "evidence": evidence,
                "tools": scoped,
                "router": self.router,
                "messages": list(self._messages.get(correlation_id, [])),
            }
            handler_context["send_message"] = _make_sender(self, spec.id, actual_task_id, correlation_id)
            handler_context["delegate"] = _make_delegator(self, handler_context)

            handler = (
                request.handler
                or self._handlers.get(spec.id)
                or self.registry.get_handler(spec.id)
                or spec.metadata.get("handler")
            )
            try:
                # Isolation setup belongs inside the guarded block: a failure
                # here (e.g. a non-git repo) must still close out the run.
                if request.isolation:
                    isolation_manager = request.context.get("isolation_manager") or WorktreeIsolation(
                        repo_path=request.context.get("repo_path") or Path.cwd(),
                        temp_root=request.context.get("worktree_root"),
                    )
                    worktree = await isolation_manager.create(name=spec.id)
                    handler_context["isolation"] = isolation_manager
                    handler_context["worktree"] = worktree
                    handler_context["workspace"] = str(worktree.path)
                raw = await invoke_callable(
                    handler,
                    payload={
                        "objective": request.objective, "context": handler_context,
                        "agent_id": spec.id, "task_id": actual_task_id, "spec": spec,
                        "dependencies": dependency_results, "tools": scoped,
                        "router": self.router, "evidence": evidence,
                        "workspace": handler_context.get("workspace"),
                        "shared_state": shared, "correlation_id": correlation_id,
                        "delegate": handler_context["delegate"],
                        "send_message": handler_context["send_message"],
                    },
                    agent_id=spec.id,
                ) if handler else await self._model_worker(spec, request, handler_context)
                result = self._parse_result(raw, spec.id, actual_task_id, evidence)
            except AgentLimitError as exc:
                # A nested delegation hit a depth/task limit. The run was
                # started already, so close it before propagating; otherwise
                # history keeps a stuck "running" record with no evaluations.
                await self.registry.finish_run(
                    run_id, status="failed", result={"error": str(exc)},
                    evaluations=[{"kind": "execution", "success": False, "reason": "agent limit"}],
                )
                raise
            except Exception as exc:
                result = AgentResult(agent_id=spec.id, task_id=actual_task_id, success=False, error=str(exc))

            result.metadata = dict(getattr(result, "metadata", {}) or {})
            evaluations: list[dict[str, Any]] = []
            if request.verify:
                second_agent = request.second_agent or request.context.get("second_agent")
                second_model = request.second_model or request.context.get("second_model")
                if second_agent == spec.id:
                    verification = VerificationResult(
                        verified=False, score=0.0,
                        reasons=["the verifier must be independent of the producing agent"],
                        checks={"independent_verifier": False},
                    )
                else:
                    verification = await self.verifier.verify(
                        result, objective=request.objective, context=handler_context,
                        expected_output=request.expected_output if request.expected_output is not None else request.context.get("expected_output"),
                        peer_outputs=request.context.get("peer_outputs"),
                        second_agent=second_agent, second_model=second_model,
                        producer_agent_id=spec.id,
                    )
                evaluation = verification.as_evaluation()
                result.metadata["verification"] = evaluation
                evaluations.append(evaluation)
                if not verification.verified:
                    result.success = False
                    result.error = "verification failed: " + "; ".join(verification.reasons)

            if worktree is not None and isolation_manager is not None:
                validation = await isolation_manager.validate(
                    worktree,
                    verification=request.context.get("verification") or result.metadata.get("verification"),
                    allowed_files=request.context.get("allowed_files"),
                )
                result.metadata["worktree"] = worktree.to_dict()
                result.metadata["validation"] = validation.model_dump(mode="json")
                if not validation.verified:
                    result.success = False
                    result.error = "isolated changes rejected: " + "; ".join(validation.reasons)

            result.metadata.update({
                "correlation_id": correlation_id, "depth": depth,
                "root_task_id": root_task_id, "agent_name": spec.name,
            })
            recipient = request.context.get("to_agent") or request.context.get("notify")
            recipients = [recipient] if isinstance(recipient, str) else (list(recipient) if recipient else [])
            for target in recipients:
                await self.send_message(
                    spec.id, target,
                    {"result": result.model_dump(mode="json"), "objective": request.objective},
                    task_id=actual_task_id, correlation_id=correlation_id,
                )

            evaluations.append({
                "kind": "execution", "success": bool(result.success), "duration_ms": result.duration_ms,
                "evidence_count": len(result.evidence), "created_at": utcnow(),
            })
            await self.registry.finish_run(
                run_id,
                status="completed" if result.success else "failed",
                result=result.model_dump(mode="json"),
                evaluations=evaluations,
                metadata={"correlation_id": correlation_id, "depth": depth, "task_id": actual_task_id},
            )
            self._results[actual_task_id] = result
            # The durable task engine requires explicit, evidence-backed completion.
            if task_engine is not None and result.success and result.evidence and hasattr(task_engine, "complete"):
                try:
                    await task_engine.complete(
                        actual_task_id, result.model_dump(mode="json"), result.evidence,
                        note="agent result verified",
                    )
                except Exception as exc:
                    _LOG.debug("task completion failed", extra={"task_id": actual_task_id, "error": str(exc)})
            return result

        try:
            async with sem:
                if self._active.get(spec.id, 0) >= max(1, spec.task_limit):
                    raise AgentLimitError(f"agent {spec.id} reached its task limit of {spec.task_limit}")
                self._active[spec.id] = self._active.get(spec.id, 0) + 1
                try:
                    task_record = await self.tasks.run(task.id, worker)
                finally:
                    self._active[spec.id] -= 1
            result = self._results.get(task.id)
            if result is None:
                raw_result = getattr(task_record, "result", None)
                if isinstance(raw_result, AgentResult):
                    result = raw_result
                else:
                    result = self._parse_result(raw_result, spec.id, task.id, [])
                result.metadata = {
                    **(getattr(result, "metadata", {}) or {}),
                    "correlation_id": correlation_id, "depth": depth,
                }
            # The worker returns its in-memory result even when the durable
            # engine refused to complete the task (e.g. success with no
            # evidence). Reflect that refusal so the result and the task row
            # never disagree about whether the work actually completed.
            if getattr(task_record, "status", None) is TaskStatus.FAILED and result.success:
                result = result.model_copy(deep=True)
                result.success = False
                result.error = result.error or (
                    task_record.errors[-1] if getattr(task_record, "errors", None) else "task failed"
                )
            self._results[task.id] = result
            return result
        except AgentLimitError:
            await self.registry.finish_run(run_id, status="failed", result={"error": "task limit exceeded"},
                                           evaluations=[{"kind": "execution", "success": False, "reason": "task limit"}])
            raise
        except Exception as exc:
            await self.registry.finish_run(run_id, status="failed", result={"error": str(exc)},
                                           evaluations=[{"kind": "execution", "success": False, "reason": str(exc)}])
            return self._failure(spec.id, task.id, exc)

    def _scoped_tools(self, spec: AgentSpec) -> ScopedTools | None:
        """The agent's tool view: its declared permissions, deny-by-default."""
        if self.tools is None:
            return None
        allow = frozenset(
            str(entry).strip() for entry in (spec.tool_permissions or []) if str(entry).strip()
        )
        return ScopedTools(self.tools, allow, spec.id)

    @staticmethod
    def _bounded_context(data: Mapping[str, Any], budget: int | None) -> str:
        """Serialize context within an agent's declared budget.

        ``context_budget`` was seeded for every specialist but never applied;
        the variable-size context in a model prompt was sent unbounded. Long
        payloads are now truncated to the budget so one agent's shared state
        cannot crowd out another's turn.
        """
        text = _json(dict(data))
        try:
            limit = int(budget or 0)
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0 or len(text) <= limit:
            return text
        marker = "...[truncated]"
        return text[: max(0, limit - len(marker))] + marker

    @staticmethod
    def _tool_call_data(raw: Any) -> dict[str, Any]:
        """Normalize a provider tool call into id/name/arguments for execution."""
        if isinstance(raw, Mapping):
            function = raw.get("function") if isinstance(raw.get("function"), Mapping) else {}
            name = str(raw.get("name") or function.get("name") or "")
            arguments = raw.get("arguments", function.get("arguments", {}))
            call_id = str(raw.get("id") or raw.get("call_id") or new_id("call"))
        else:
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
        return {"id": call_id, "name": name, "arguments": dict(arguments)}

    async def _model_worker(self, spec: AgentSpec, request: AgentRequest, context: dict[str, Any]) -> Any:
        """Default worker: bounded, provider-compatible multi-round tool loop."""
        if self.router is None:
            raise AgentError(f"no handler or model router available for agent {spec.id}")
        # Registry preferences are capability tags, not model identifiers.
        # Only names that match configured providers may influence routing;
        # sending a tag as `model` would be rejected by a live provider.
        configured = getattr(self.router, "providers", {}) or {}
        preferred = [name for name in (spec.model_preferences or []) if name in configured]
        scoped = self._scoped_tools(spec)
        # Advertise exactly the agent's declared tool scope; an agent whose
        # permissions reference no registered tool gets no tools at all.
        tool_specs = [s.model_dump(mode="json") for s in scoped.specs()] if scoped else []
        context_data = {
            k: v for k, v in context.items() if k in ("shared_state", "dependencies", "messages")
        }
        context_text = self._bounded_context(context_data, spec.context_budget)
        # Inherit the parent turn's effective reasoning unless this delegation
        # explicitly overrides it. Background work outside a turn falls back to
        # the provider/global default (no request override).
        from ..models.reasoning_state import (
            current_reasoning,
            selection_from_context,
        )

        explicit = request.context.get("reasoning")
        if explicit is None and (
            request.context.get("reasoning_effort") is not None
            or request.context.get("reasoning_show") is not None
        ):
            explicit = {
                "effort": request.context.get("reasoning_effort"),
                "show": request.context.get("reasoning_show"),
                "source": "context",
            }
        reasoning = selection_from_context(explicit) or current_reasoning()
        llm_request = ModelRequest(
            messages=[ChatMessage(role="user", content=(
                f"You are {spec.name}. {spec.purpose}\nObjective: {request.objective}\n"
                f"Context: {context_text}"
            ))],
            task_category=spec.id,
            task_id=context.get("task_id"),
            agent_id=spec.id,
            tools=tool_specs,
            reasoning_effort=reasoning.get("effort") if reasoning else None,
            reasoning_show=bool(reasoning.get("show")) if reasoning else False,
            reasoning_source=str(reasoning.get("source") or "") if reasoning else None,
            metadata={
                # Provider names still take priority, but every capability tag
                # must reach the router: tags such as ``structured``/``tools``
                # rank providers that advertise that capability.
                "preferred_providers": preferred,
                "agent_capabilities": list(spec.model_preferences or []),
                "model_preferences": list(spec.model_preferences or []),
            },
        )
        evidence = list(context.get("evidence") or [])
        content: Any = ""
        for round_index in range(MAX_MODEL_TOOL_ROUNDS):
            response = await self._complete(llm_request)
            if response is None:
                raise AgentError("model router did not return a response")
            content = getattr(response, "content", response)
            raw_calls = list(getattr(response, "tool_calls", None) or [])
            if not raw_calls:
                break
            if round_index == MAX_MODEL_TOOL_ROUNDS - 1:
                raise AgentError(
                    f"model worker exceeded {MAX_MODEL_TOOL_ROUNDS} tool rounds for agent {spec.id}"
                )
            # Providers require the assistant message carrying every tool call
            # followed by one tool message per call, in the same order. Without
            # both halves the second round is rejected by real APIs.
            llm_request.messages.append(ChatMessage(
                role="assistant", content=str(content or ""),
                tool_calls=[dict(raw) if isinstance(raw, Mapping) else raw for raw in raw_calls],
            ))
            for raw in raw_calls:
                data = self._tool_call_data(raw)
                if scoped is None:
                    raise AgentError(
                        f"model requested tool {data['name']!r} but agent {spec.id} has no tools in scope"
                    )
                call = ToolCall(
                    id=data["id"], name=data["name"], arguments=data["arguments"],
                    actor=f"agent:{spec.id}", correlation_id=context.get("correlation_id"),
                )
                try:
                    outcome = await scoped.execute(call)
                    output = getattr(outcome, "output", outcome)
                    success = bool(getattr(outcome, "success", True))
                    if not success:
                        raise AgentError(getattr(outcome, "error", None) or "tool execution failed")
                    rendered = outcome.model_dump(mode="json") if hasattr(outcome, "model_dump") else {"output": output}
                    tool_message = json.dumps(rendered, default=str)
                except Exception as exc:
                    output = {"error": str(exc)}
                    tool_message = json.dumps(output, default=str)
                    evidence.append({
                        "source": "tool", "claim": str(exc), "digest": evidence_digest(output),
                        "tool_call": {
                            "id": data["id"], "tool": data["name"],
                            "arguments": data["arguments"], "output": output, "success": False,
                        },
                        "verified": False,
                    })
                else:
                    evidence.append({
                        "source": f"tool:{data['name']}",
                        "claim": _json(output),
                        "digest": evidence_digest(output),
                        "tool_call": {
                            "id": data["id"], "tool": data["name"],
                            "arguments": data["arguments"], "output": output, "success": True,
                        },
                    })
                llm_request.messages.append(
                    ChatMessage(role="tool", content=tool_message, tool_call_id=data["id"])
                )
            # Next iteration re-sends the full message list, including the
            # assistant(tool_calls)/tool pair appended above.

        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping) and "output" in parsed:
            return {
                "output": parsed["output"],
                "evidence": [*evidence, *(parsed.get("evidence") or [])],
                **{k: v for k, v in parsed.items() if k not in {"output", "evidence"}},
            }
        return {"output": content, "evidence": evidence}

    async def _complete(self, request: ModelRequest) -> Any:
        if hasattr(self.router, "complete"):
            return await self.router.complete(request)
        if hasattr(self.router, "route"):
            provider = await self.router.route(request)
            return await provider.complete(request)
        return None

    def _parse_result(self, raw: Any, agent_id: str, task_id: str, evidence: list[Any]) -> AgentResult:
        collected = list(evidence)
        if hasattr(raw, "evidence"):
            collected.extend(list(getattr(raw, "evidence") or []))
        if isinstance(raw, AgentResult):
            result = raw.model_copy(deep=True)
            result.agent_id = agent_id
            result.task_id = task_id
        elif isinstance(raw, ModelResponse):
            result = AgentResult(agent_id=agent_id, task_id=task_id, success=True, output=raw.content)
        elif isinstance(raw, Mapping):
            success = bool(raw.get("success", True)) and not raw.get("error")
            if any(key in raw for key in ("output", "result", "content")):
                output = raw.get("output", raw.get("result", raw.get("content")))
            else:
                output = {k: v for k, v in raw.items() if k not in {"evidence", "success", "error", "duration_ms", "evaluations"}}
            collected.extend(raw.get("evidence") or [])
            result = AgentResult(
                agent_id=agent_id, task_id=task_id, success=success,
                output=output, error=raw.get("error"), duration_ms=float(raw.get("duration_ms", 0) or 0),
            )
        else:
            result = AgentResult(agent_id=agent_id, task_id=task_id, success=True, output=raw)
        normalized: list[dict[str, Any]] = []
        for item in collected:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            elif item:
                normalized.append({"source": "handler", "claim": str(item)})
        result.evidence = normalized
        return result

    async def integrate_agent_work(
        self,
        result: AgentResult,
        *,
        verification: Any = None,
        isolation_manager: WorktreeIsolation | None = None,
        allowed_files: list[str] | None = None,
        message: str | None = None,
    ) -> Any:
        """Explicitly integrate a verified isolated result (never done automatically)."""
        metadata = getattr(result, "metadata", {}) or {}
        info = metadata.get("worktree")
        if not info:
            raise AgentError("result has no isolated worktree")
        worktree = Worktree(
            path=Path(info["path"]), branch=info["branch"], base=info["base"],
            repo_path=Path(info["repo_path"]),
        )
        manager = isolation_manager or WorktreeIsolation(repo_path=worktree.repo_path)
        integration = await manager.integrate(
            worktree,
            verification=verification or metadata.get("validation"),
            allowed_files=allowed_files,
            message=message,
        )
        if getattr(integration, "merged", False):
            # The branch is merged; the worktree has served its purpose.
            # Best-effort: a cleanup failure must not fail a completed merge.
            try:
                await manager.cleanup(worktree)
            except Exception as exc:  # noqa: BLE001 - merge already succeeded
                _LOG.warning(
                    "merged worktree could not be removed",
                    extra={"path": str(worktree.path), "error": str(exc)},
                )
        return integration


def _make_sender(engine: AgentEngine, agent_id: str, task_id: str, correlation_id: str) -> Callable[..., Any]:
    async def send(to_agent: str, content: Any = "", **kwargs: Any) -> Any:
        return await engine.send_message(
            agent_id, to_agent, content,
            task_id=kwargs.pop("task_id", task_id),
            correlation_id=kwargs.pop("correlation_id", correlation_id),
            **kwargs,
        )
    return send


def _make_delegator(engine: AgentEngine, context: dict[str, Any]) -> Callable[..., Any]:
    async def delegate(objective: Any, **kwargs: Any) -> Any:
        nested = dict(context)
        nested["depth"] = int(context.get("depth", 0)) + 1
        nested["correlation_id"] = context.get("correlation_id")
        return await engine.delegate(objective, context=nested, **kwargs)
    return delegate
