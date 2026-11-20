"""Agent registry: the twelve LEO specialists, dynamic registration and history.

The registry owns the ``agents`` table and the execution/evaluation trail in
``agent_runs``. Handlers are deliberately kept out of the database: they are
process-local callables (or provider adapters), while the database stores only
serializable identity, policy and evaluation data.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .. import db
from ..contracts import AgentSpec, new_id, utcnow
from ..logging import get_logger

_LOG = get_logger("agents")

# The twelve durable core specialists live here; the eight executable
# specialists in ``specs/`` are loaded by ``load_specialists()`` at initialize
# time. The total roster is twenty.
CORE_SPECIALIST_COUNT = 12
SPECIALIST_COUNT = 20


# Plan §7: exactly these twelve specialists, in this order.
SPECIALIST_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "general",
        "name": "General Assistant",
        "purpose": "Handle broad, general-purpose requests and coordinate hand-offs when specialization is needed.",
        "capabilities": ["reasoning", "planning", "summarization", "delegation"],
        "tool_permissions": ["filesystem", "utility"],
        "model_preferences": ["general", "text", "tools"],
        "context_budget": 16000,
        "task_limit": 4,
        "metadata": {"tier": "general"},
    },
    {
        "id": "research",
        "name": "Research Agent",
        "purpose": "Gather, cross-check and cite information from allowed sources.",
        "capabilities": ["web_research", "source_evaluation", "synthesis"],
        "tool_permissions": ["research", "filesystem", "utility"],
        "model_preferences": ["research", "text", "tools"],
        "context_budget": 24000,
        "task_limit": 3,
        "metadata": {"tier": "knowledge"},
    },
    {
        "id": "coding",
        "name": "Coding Agent",
        "purpose": "Implement, refactor and test code changes inside isolated workspaces.",
        "capabilities": ["code_edit", "unit_test", "debugging", "git"],
        "tool_permissions": ["filesystem", "development"],
        "model_preferences": ["coding", "tools", "structured"],
        "context_budget": 32000,
        "task_limit": 3,
        "metadata": {"tier": "engineering", "isolation": True},
    },
    {
        "id": "architecture",
        "name": "Architecture Agent",
        "purpose": "Design module boundaries, interfaces and trade-offs before implementation.",
        "capabilities": ["system_design", "interface_review", "tradeoff_analysis"],
        "tool_permissions": ["filesystem", "utility"],
        "model_preferences": ["reasoning", "text", "structured"],
        "context_budget": 32000,
        "task_limit": 2,
        "metadata": {"tier": "engineering"},
    },
    {
        "id": "browser",
        "name": "Browser Agent",
        "purpose": "Navigate and interact with web pages through the browser adapter.",
        "capabilities": ["browser_navigation", "form_interaction", "dom_inspection"],
        "tool_permissions": ["browser"],
        "model_preferences": ["vision", "tools"],
        "context_budget": 16000,
        "task_limit": 2,
        "metadata": {"tier": "operator"},
    },
    {
        "id": "computer",
        "name": "Computer Operator",
        "purpose": "Operate the local computer: shell, filesystem and desktop tasks under policy.",
        "capabilities": ["shell", "filesystem", "desktop_control"],
        "tool_permissions": ["filesystem", "development", "computer"],
        "model_preferences": ["tools", "vision"],
        "context_budget": 16000,
        "task_limit": 2,
        "metadata": {"tier": "operator", "risk": "medium"},
    },
    {
        "id": "qa",
        "name": "QA Agent",
        "purpose": "Design and run regression, integration and acceptance tests; report defects.",
        "capabilities": ["test_design", "test_execution", "defect_reporting"],
        "tool_permissions": ["filesystem", "development"],
        "model_preferences": ["coding", "tools", "structured"],
        "context_budget": 24000,
        "task_limit": 3,
        "metadata": {"tier": "engineering"},
    },
    {
        "id": "security",
        "name": "Security Agent",
        "purpose": "Review changes and behavior for security weaknesses, secrets and unsafe permissions.",
        "capabilities": ["threat_modeling", "code_audit", "secret_detection"],
        "tool_permissions": ["read_file", "list_files", "search_files", "utility", "audit", "read_only_scan"],
        "model_preferences": ["reasoning", "structured"],
        "context_budget": 24000,
        "task_limit": 2,
        "metadata": {"tier": "engineering", "risk": "high"},
    },
    {
        "id": "devops",
        "name": "DevOps Agent",
        "purpose": "Build, deploy and monitor services through configured infrastructure adapters.",
        "capabilities": ["ci_cd", "deployment", "observability"],
        "tool_permissions": ["development", "research", "deploy"],
        "model_preferences": ["tools", "text"],
        "context_budget": 16000,
        "task_limit": 2,
        "metadata": {"tier": "infrastructure", "risk": "high"},
    },
    {
        "id": "data",
        "name": "Data Analyst",
        "purpose": "Profile, query and interpret structured data; produce evidence-backed conclusions.",
        "capabilities": ["sql", "statistics", "visualization_specs"],
        "tool_permissions": ["utility", "database", "python", "charts"],
        "model_preferences": ["reasoning", "tools", "structured"],
        "context_budget": 24000,
        "task_limit": 3,
        "metadata": {"tier": "knowledge"},
    },
    {
        "id": "documentation",
        "name": "Documentation Agent",
        "purpose": "Write and maintain accurate, concise documentation from verified sources.",
        "capabilities": ["technical_writing", "api_docs", "editing"],
        "tool_permissions": ["filesystem", "utility"],
        "model_preferences": ["text", "structured"],
        "context_budget": 16000,
        "task_limit": 4,
        "metadata": {"tier": "knowledge"},
    },
    {
        "id": "integration",
        "name": "Integration Agent",
        "purpose": "Connect external services and adapters with minimum necessary permissions.",
        "capabilities": ["api_integration", "adapter_design", "credential_references"],
        "tool_permissions": ["filesystem", "research", "integrations"],
        "model_preferences": ["tools", "structured"],
        "context_budget": 20000,
        "task_limit": 3,
        "metadata": {"tier": "infrastructure", "risk": "medium"},
    },
)


def _json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _loads(text: Any, default: Any) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text) if isinstance(text, str) else default
    except (TypeError, ValueError):
        return default


def _spec_from_row(row: Any) -> AgentSpec:
    return AgentSpec(
        id=row["id"],
        name=row["name"],
        purpose=row["purpose"],
        capabilities=_loads(row["capabilities"], []),
        tool_permissions=_loads(row["tool_permissions"], []),
        model_preferences=_loads(row["model_preferences"], []),
        context_budget=int(row["context_budget"]),
        task_limit=int(row["task_limit"]),
        enabled=bool(row["enabled"]),
        metadata=_loads(row["metadata"], {}),
    )


class AgentRegistry:
    """Dynamic, durable registry of agent identities and their execution trail."""

    def __init__(self, *, seed: bool = False) -> None:
        self._specs: dict[str, AgentSpec] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        # Ids loaded from executable SEED modules (not dynamically registered).
        self.loaded_specialist_ids: set[str] = set()
        self._loaded = False
        self._seed = seed

    async def initialize(
        self,
        *,
        services: Mapping[str, Any] | None = None,
        tool_registry: Any = None,
    ) -> "AgentRegistry":
        """Load persisted agents, seed the twelve core roles and load specs.

        ``tool_registry`` is the live registry used to verify every grant a
        specialist declares. When the composition root supplies it, a grant
        that matches neither a tool name nor a live category fails closed
        instead of silently widening an agent's authority.
        """
        await self.refresh()
        await self.seed()
        await self.load_specialists(services=services, tool_registry=tool_registry)
        return self

    async def load_specialists(
        self,
        *,
        services: Mapping[str, Any] | None = None,
        tool_registry: Any = None,
    ) -> list[Any]:
        """Import every ``specs/*.py`` SEED, validate it and bind its handler."""
        package = Path(__file__).resolve().parent / "specs"
        loaded: list[Any] = []
        live_names: set[str] = set()
        live_categories: set[str] = set()
        if tool_registry is not None:
            specs = list(tool_registry.list_specs())
            live_names = {str(spec.name) for spec in specs}
            live_categories = {str(getattr(spec, "category", "")) for spec in specs}
        for info in sorted(pkgutil.iter_modules([str(package)]), key=lambda item: item.name):
            if info.ispkg:
                continue
            module = importlib.import_module(f"{__package__}.specs.{info.name}")
            seed = getattr(module, "SEED", None)
            if not isinstance(seed, Mapping):
                raise RuntimeError(f"agent spec {info.name} has no SEED mapping")
            grants = [str(grant) for grant in (seed.get("tool_permissions") or [])]
            invalid = [
                grant for grant in grants
                if grant not in live_names and grant not in live_categories
            ] if tool_registry is not None else []
            if invalid:
                raise RuntimeError(
                    f"agent spec {info.name} declares grants absent from the live "
                    f"tool registry: {', '.join(sorted(invalid))}"
                )
            agent = AgentSpec.model_validate(dict(seed))
            existing = agent.id in self._specs
            loaded.append(await self.register(agent, replace=existing))
            self.loaded_specialist_ids.add(agent.id)
            factory = getattr(module, "make_handler", None)
            if callable(factory):
                handler = factory(services if services is not None else {})
                if callable(handler):
                    # Never bind a string-valued ``metadata['handler']``: the
                    # engine would treat it as a callable and fail at runtime.
                    self.set_handler(agent.id, handler)
                else:
                    _LOG.warning("specialist handler factory returned a non-callable", extra={"agent_id": agent.id})
            _LOG.debug("specialist loaded", extra={"agent_id": agent.id, "grants": len(grants)})
        return loaded

    async def refresh(self) -> None:
        rows = await db.fetchall("SELECT * FROM agents ORDER BY name")
        self._specs = {row["id"]: _spec_from_row(row) for row in rows}
        self._loaded = True

    async def seed(self) -> list[AgentSpec]:
        if not self._loaded:
            await self.refresh()
        seeded: list[AgentSpec] = []
        for raw in SPECIALIST_SPECS:
            if raw["id"] in self._specs:
                continue
            seeded.append(await self.register(AgentSpec.model_validate(raw)))
        return seeded

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self.refresh()

    async def register(
        self,
        spec: AgentSpec | Mapping[str, Any],
        *,
        handler: Callable[..., Any] | None = None,
        replace: bool = False,
    ) -> AgentSpec:
        """Register (or replace) an agent. Dynamic agents need not be specialists."""
        agent = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(dict(spec))
        if not agent.id or not agent.name:
            raise ValueError("agent requires an id and a name")
        await self._ensure_loaded()
        exists = agent.id in self._specs
        if exists and not replace:
            prior = self._specs[agent.id]
            if prior == agent:
                # Re-registering the exact same seed is a no-op (idempotent
                # loading), but an explicit handler still replaces the binding.
                if handler is not None:
                    self.set_handler(agent.id, handler)
                return prior
            raise ValueError(f"agent already registered: {agent.id}")
        await db.execute(
            "INSERT INTO agents(id,name,purpose,capabilities,tool_permissions,model_preferences,"
            "context_budget,task_limit,enabled,metadata) VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, purpose=excluded.purpose, "
            "capabilities=excluded.capabilities, tool_permissions=excluded.tool_permissions, "
            "model_preferences=excluded.model_preferences, context_budget=excluded.context_budget, "
            "task_limit=excluded.task_limit, enabled=excluded.enabled, metadata=excluded.metadata",
            (
                agent.id, agent.name, agent.purpose, _json(agent.capabilities),
                _json(agent.tool_permissions), _json(agent.model_preferences),
                agent.context_budget, agent.task_limit, int(agent.enabled), _json(agent.metadata),
            ),
        )
        self._specs[agent.id] = agent
        if handler is not None:
            self._handlers[agent.id] = handler
        _LOG.debug("agent registered", extra={"agent_id": agent.id})
        return agent

    async def unregister(self, agent_id: str) -> bool:
        await self._ensure_loaded()
        await db.execute("DELETE FROM agents WHERE id=?", (agent_id,))
        self._specs.pop(agent_id, None)
        self._handlers.pop(agent_id, None)
        return True

    async def enable(self, agent_id: str, enabled: bool = True) -> AgentSpec:
        await self._ensure_loaded()
        spec = self._specs.get(agent_id)
        if spec is None:
            raise KeyError(f"unknown agent: {agent_id}")
        await db.execute("UPDATE agents SET enabled=? WHERE id=?", (int(enabled), agent_id))
        spec.enabled = bool(enabled)
        return spec

    async def get(self, agent_id: str | None) -> AgentSpec | None:
        if not agent_id:
            return None
        await self._ensure_loaded()
        return self._specs.get(agent_id)

    async def list(self, *, enabled_only: bool = False) -> list[AgentSpec]:
        await self._ensure_loaded()
        specs = list(self._specs.values())
        return [s for s in specs if s.enabled] if enabled_only else specs

    @property
    def all(self) -> list[AgentSpec]:
        """Synchronous view of the in-memory cache (call ``initialize`` first)."""
        return list(self._specs.values())

    def get_cached(self, agent_id: str) -> AgentSpec | None:
        return self._specs.get(agent_id)

    def set_handler(self, agent_id: str, handler: Callable[..., Any]) -> None:
        self._handlers[agent_id] = handler

    def get_handler(self, agent_id: str) -> Callable[..., Any] | None:
        return self._handlers.get(agent_id)

    # --- durable execution history and evaluations (plan §7 and §16) ---

    async def start_run(
        self,
        agent_id: str,
        *,
        task_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = run_id or new_id("run")
        await db.execute(
            "INSERT INTO agent_runs(id,agent_id,task_id,status,started_at,finished_at,result,evaluations,metadata) "
            "VALUES(?,?,?,?,?,NULL,NULL,NULL,?)",
            (run_id, agent_id, task_id, "running", utcnow(), _json(metadata or {})),
        )
        return run_id

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        result: Any = None,
        evaluations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if metadata is None:
            await db.execute(
                "UPDATE agent_runs SET status=?, finished_at=?, result=?, evaluations=? WHERE id=?",
                (status, utcnow(), _json(result), _json(evaluations or []), run_id),
            )
        else:
            await db.execute(
                "UPDATE agent_runs SET status=?, finished_at=?, result=?, evaluations=?, metadata=? WHERE id=?",
                (status, utcnow(), _json(result), _json(evaluations or []), _json(metadata), run_id),
            )

    async def record_run(
        self,
        agent_id: str,
        *,
        task_id: str | None = None,
        status: str = "completed",
        started_at: str | None = None,
        result: Any = None,
        evaluations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a complete run in one call (used by external integrations)."""
        run_id = new_id("run")
        await db.execute(
            "INSERT INTO agent_runs(id,agent_id,task_id,status,started_at,finished_at,result,evaluations,metadata) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                run_id, agent_id, task_id, status, started_at or utcnow(), utcnow(),
                _json(result), _json(evaluations or []), _json(metadata or {}),
            ),
        )
        return run_id

    async def history(self, agent_id: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
        if agent_id:
            rows = await db.fetchall(
                "SELECT * FROM agent_runs WHERE agent_id=? ORDER BY started_at DESC LIMIT ?",
                (agent_id, limit),
            )
        else:
            rows = await db.fetchall("SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,))
        return [self._row_to_run(row) for row in rows]

    async def evaluations(self, agent_id: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for run in await self.history(agent_id, limit=limit):
            for evaluation in run.get("evaluations") or []:
                result.append({"run_id": run["id"], "agent_id": run["agent_id"], **evaluation})
        return result

    @staticmethod
    def _row_to_run(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "agent_id": row["agent_id"],
            "task_id": row["task_id"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "result": _loads(row["result"], None),
            "evaluations": _loads(row["evaluations"], []),
            "metadata": _loads(row["metadata"], {}),
        }
