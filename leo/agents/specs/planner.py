"""Planner specialist: bounded, evidence-backed task graphs with explicit scope.

This spec uses a bespoke execution path (:func:`make_handler`) rather than the
default model/delegation worker. The worker decomposes one objective into a
bounded chain of concrete steps and delegates each step through the real
``AgentEngine`` with an explicit ``agent_id`` and a minimal, explicit scope.
It never forwards the parent's tool façade, router, shared state or delegator
to a child, never self-delegates, and records child task ids, evidence and
dependency edges so the resulting graph is auditable. Evidence is concrete:
:func:`node_evidence` writes a content-addressed artifact whose digest the
verifier recomputes from disk; bare model assertions are never emitted.

Tool grants are limited to verified live-registry categories: ``agents``
(agent_list / agent_delegate / agent_status / agent_result), ``tasks``
(task_create / task_list / task_status / task_result / task_cancel) and
``utility`` (calculator / json_parse / current_time).
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

# JSON-serializable seed with exactly the keys used by AgentRegistry rows.
SEED: dict[str, Any] = {
    "id": "planner",
    "name": "Planner Agent",
    "purpose": (
        "Decompose user objectives into bounded, evidence-backed task graphs and "
        "delegate each node only with an explicit agent, scope and dependency set."
    ),
    "capabilities": [
        "decomposition",
        "task_graph",
        "bounded_delegation",
        "evidence_synthesis",
    ],
    "tool_permissions": ["agents", "tasks", "utility"],
    "model_preferences": ["reasoning", "structured", "tools"],
    "context_budget": 24000,
    "task_limit": 3,
    "metadata": {
        "tier": "knowledge",
        "role": "planner",
        "risk": "low",
        "delegation_mode": "explicit_scope",
        "max_nodes": 6,
        "targets": ["research", "qa", "documentation", "coding", "general"],
    },
}

MAX_NODES = int(SEED["metadata"]["max_nodes"])
DEFAULT_TARGETS: tuple[str, ...] = tuple(SEED["metadata"]["targets"])
_FALLBACK_TARGET = "general"
_ENGINE_KEYS = ("agent_engine", "agents_engine")
_REGISTRY_KEYS = ("agent_registry", "agents")
_EVIDENCE_DIR = Path(tempfile.gettempdir()) / "leo-planner-evidence"
_TOOL_SOURCES = frozenset({"tool", "tool_call", "executor", "scoped_tools"})
_ARTIFACT_SOURCES = frozenset({"artifact", "file", "diff", "worktree"})


def node_evidence(claim: str, *, directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Return verifiable evidence for one delegated node's observed claim.

    The claim is written verbatim to a content-addressed local artifact; the
    verifier recomputes its SHA-256 from disk and checks the claim against the
    file content. Self-attested ``claim`` items with no anchor cannot satisfy
    the production evidence gate, so this helper is the supported way for a
    planner node (or its worker) to record evidence.
    """
    text = str(claim)
    if not text.strip():
        raise ValueError("planner node evidence requires a non-empty claim")
    root = Path(directory) if directory is not None else _EVIDENCE_DIR
    root.mkdir(parents=True, exist_ok=True)
    hexdigest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = root / f"{hexdigest}.txt"
    if not path.is_file():
        path.write_text(text, encoding="utf-8")
    return [{"source": "artifact", "path": str(path), "digest": hexdigest, "claim": text}]


def _verifiable(items: Any) -> list[dict[str, Any]]:
    """Keep only evidence with a real audit anchor or a digest-backed artifact."""
    kept: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        entry = dict(item)
        source = str(entry.get("source") or "").strip().lower()
        anchor = entry.get("tool_call")
        if source in _TOOL_SOURCES or source.startswith("tool:"):
            if isinstance(anchor, Mapping) and anchor.get("id") and anchor.get("tool"):
                kept.append(entry)
                continue
        if source in _ARTIFACT_SOURCES and (
            entry.get("path") and (entry.get("digest") or entry.get("sha256"))
        ):
            kept.append(entry)
    return kept


# Keyword rules only choose an explicit target; they never grant authority.
_TARGET_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("research", "find", "gather", "source", "investigate", "lookup"), "research"),
    (("test", "verify", "review", "audit", "check", "validat", "qa"), "qa"),
    (("document", "write", "report", "summari", "explain"), "documentation"),
    (("code", "coding", "implement", "refactor", "fix", "patch", "compile"), "coding"),
)


def _service(mapping: Any, keys: tuple[str, ...], label: str) -> Any:
    if not isinstance(mapping, Mapping):
        raise RuntimeError(f"planner requires an injected services mapping with {label!r}")
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    raise RuntimeError(
        f"planner requires service {label!r} in services "
        f"(accepted keys: {', '.join(keys)})"
    )


def _steps(objective: str) -> list[str]:
    text = str(objective or "").strip()
    if not text:
        raise ValueError("planner requires a non-empty objective")
    chunks: list[str] = []
    for line in text.splitlines():
        clean = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if clean:
            chunks.extend(part.strip() for part in re.split(r";|\bthen\b|\band\b", clean) if part.strip())
    return chunks or [text]


def _limits(context: Mapping[str, Any]) -> int:
    spec = context.get("agent")
    metadata = getattr(spec, "metadata", None) if spec is not None else None
    if not isinstance(metadata, Mapping):
        metadata = SEED["metadata"]
    try:
        return max(1, int(metadata.get("max_nodes", MAX_NODES)))
    except (TypeError, ValueError):
        return MAX_NODES


async def _available_targets(registry: Any) -> list[str]:
    specs = await registry.list(enabled_only=True)
    available = {str(getattr(spec, "id", "")) for spec in specs}
    # A planner must never delegate to itself: that would be an unbounded loop.
    available.discard(SEED["id"])
    ordered = [t for t in DEFAULT_TARGETS if t in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _select_target(step: str, available: list[str]) -> str:
    # Whole-word (or long-prefix) matching: "findings" must not look like a
    # request to "find" sources and steal a QA node from the research target.
    tokens = set(re.findall(r"[a-z]+", step.lower()))

    def matches(word: str) -> bool:
        return word in tokens or (len(word) >= 6 and any(t.startswith(word) for t in tokens))

    for words, target in _TARGET_RULES:
        if target in available and any(matches(word) for word in words):
            return target
    if _FALLBACK_TARGET in available:
        return _FALLBACK_TARGET
    return available[0]


def make_handler(services: Any = None) -> Callable[..., Any]:
    """Build the planner worker bound to the injected agent engine and registry.

    Services are resolved per call from ``ctx['services']`` when supplied by
    the executor and otherwise from the mapping passed here. Missing engine or
    registry is an explicit error: planning without a real delegation path is
    never reported as success.
    """

    async def planner_handler(objective: Any, context: Any, **kwargs: Any) -> dict[str, Any]:
        ctx = dict(context or {})
        injected = ctx.get("services") or services
        engine = _service(injected, _ENGINE_KEYS, "agent_engine")
        registry = _service(injected, _REGISTRY_KEYS, "agent_registry")
        evidence_dir = injected.get("evidence_dir") if isinstance(injected, Mapping) else None

        available = await _available_targets(registry)
        if not available:
            raise RuntimeError("planner has no enabled delegation targets")

        steps = _steps(objective if isinstance(objective, str) else str(objective))
        max_nodes = _limits(ctx)
        planned = steps[:max_nodes]
        omitted = len(steps) - len(planned)
        correlation_id = str(
            kwargs.get("correlation_id") or ctx.get("correlation_id") or "planner"
        )
        parent_depth = ctx.get("depth", 0)
        try:
            depth = int(parent_depth or 0)
        except (TypeError, ValueError):
            depth = 0
        parent_task_id = str(kwargs.get("task_id") or ctx.get("task_id") or "")

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        evidence: list[dict[str, Any]] = []
        previous: Any = None
        for index, step in enumerate(planned):
            target = _select_target(step, available)
            scope = {
                "explicit": True,
                "node": index,
                "total_nodes": len(planned),
                "target": target,
                "parent_objective": str(objective),
                "planner_task_id": parent_task_id,
                "correlation_id": correlation_id,
            }
            depend_on = [getattr(previous, "task_id", None)] if previous is not None else []
            depend_on = [str(task_id) for task_id in depend_on if task_id]
            child = None
            error = None
            try:
                child = await engine.delegate(
                    step,
                    agent_id=target,
                    dependencies=[previous] if previous is not None else [],
                    context={"depth": depth + 1, "correlation_id": correlation_id, "scope": scope},
                    verify=True,
                )
                if isinstance(child, (list, tuple)):
                    child = child[0] if child else None
            except Exception as exc:  # a blocked node is recorded, never hidden
                error = str(exc)

            task_id = str(getattr(child, "task_id", None) or "")
            success = bool(getattr(child, "success", False))
            # Only digest-backed or audit-backed evidence survives the gate;
            # a worker's bare assertion can never make a node complete.
            child_evidence = []
            if success and not error:
                child_evidence = _verifiable(getattr(child, "evidence", None) or [])
            complete = success and not error and bool(child_evidence)
            if not complete and not error:
                error = "delegation returned no verifiable evidence" if success else str(
                    getattr(child, "error", None) or "delegation was unsuccessful"
                )

            node = {
                "id": f"node-{index + 1}",
                "step": step,
                "agent_id": target,
                "scope": scope,
                "depends_on": depend_on,
                "task_id": task_id,
                "status": "completed" if complete else "blocked",
                "output": getattr(child, "output", None),
                "error": error,
            }
            nodes.append(node)
            if depend_on:
                edges.append({"from": depend_on[0], "to": task_id or f"node-{index + 1}"})
            # One content-addressed artifact per node binds the observed step
            # to this plan; the verifier recomputes its digest from disk.
            evidence.extend(node_evidence(step, directory=evidence_dir))
            evidence.extend(child_evidence)
            previous = child if complete else previous

        completed = sum(1 for node in nodes if node["status"] == "completed")
        all_done = bool(nodes) and completed == len(nodes)
        output = {
            "objective": str(objective),
            "strategy": "bounded_dependency_chain",
            "graph": {"nodes": nodes, "edges": edges},
            "completed": completed,
            "blocked": len(nodes) - completed,
            "omitted": omitted,
            "max_nodes": max_nodes,
        }
        result: dict[str, Any] = {"output": output, "evidence": evidence, "success": all_done}
        if not all_done:
            result["error"] = "one or more planned nodes were blocked"
        return result

    return planner_handler


__all__ = ["SEED", "MAX_NODES", "DEFAULT_TARGETS", "node_evidence", "make_handler"]
