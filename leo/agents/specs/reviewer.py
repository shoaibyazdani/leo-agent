"""Independent, read-only code and diff review specialist.

``SEED`` has exactly the keys of a ``leo.agents.registry.SPECIALIST_SPECS`` row
(the loader will expose those rows as ``_SEEDS``), so the integration agent can
register it with ``AgentRegistry.register(AgentSpec.model_validate(SEED))``.

Execution path: this specialist needs bespoke behavior, so it exposes
``make_handler(services)``. The default delegation worker executes tool calls
that arrive from a provider payload and cannot guarantee an independent,
evidence-backed review; this handler instead drives the engine's scoped
``tools`` facade itself and attributes every finding to a concrete source and
line. Wire it with ``registry.set_handler(SEED['id'], make_handler(services))``
or ``engine.set_handler(SEED['id'], make_handler(services))``. When no scoped
facade is injected, ``make_handler`` falls back to a tools/executor object in
``services``; the internal allowlist below still blocks every write tool.

Verifiable evidence: every reviewed source is read through a real, audited
read tool. Each evidence item carries the tool-call id, exact arguments, exact
output and the digest of that output, so verification can match the durable
``tool_calls`` row and recompute the digest. File-backed sources additionally
carry a SHA-256 digest recomputed from the artifact on disk. Self-attested
claims are never emitted; an unverifiable input returns an explicit blocked
status instead of a fake success.

No edit rights: ``tool_permissions`` names only low-risk read tools, and
``_read_tool`` refuses any tool name outside that list. The handler never
writes, delegates or mutates the reviewed work.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...contracts import ToolCall
from ..verification import MAX_ARTIFACT_BYTES, digest as evidence_digest

SEED: dict[str, Any] = {
    "id": "reviewer",
    "name": "Reviewer Agent",
    "purpose": (
        "Independently review code and diffs with read-only tools and return "
        "evidence-backed findings attributed to a file or diff line; never edit "
        "the work under review."
    ),
    "capabilities": [
        "code_review",
        "diff_review",
        "evidence_attribution",
        "read_only_audit",
    ],
    "tool_permissions": [
        "read_file",
        "list_files",
        "search_files",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
    ],
    "model_preferences": ["reasoning", "coding", "structured"],
    "context_budget": 24000,
    "task_limit": 2,
    "metadata": {
        "tier": "engineering",
        "risk": "low",
        "read_only": True,
        "no_edit_rights": True,
        "independent_review": True,
        "handler_factory": "make_handler",
    },
}

_MAX_FINDINGS = 40
_MAX_EXCERPT = 200


@dataclass(frozen=True)
class _Rule:
    category: str
    severity: str
    pattern: re.Pattern[str]
    recommendation: str


_RULES: tuple[_Rule, ...] = (
    _Rule(
        "secret", "high",
        re.compile(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"'][^\"']{6,}[\"']", re.I),
        "Move the credential to a secret store and rotate the exposed value.",
    ),
    _Rule(
        "dynamic_exec", "high",
        re.compile(r"\b(?:eval|exec)\s*\("),
        "Replace dynamic execution with an explicit, validated code path.",
    ),
    _Rule(
        "bare_except", "medium",
        re.compile(r"\bexcept\s*:"),
        "Catch specific exception types so failures are not silently swallowed.",
    ),
    _Rule(
        "debug_print", "low",
        re.compile(r"\bprint\s*\("),
        "Remove the debug print or route it through the application logger.",
    ),
    _Rule(
        "todo", "low",
        re.compile(r"\b(?:TODO|FIXME|XXX)\b"),
        "Resolve the marker or record a tracked follow-up before release.",
    ),
)


@dataclass(frozen=True)
class _Source:
    """One reviewed artifact, together with the audited read that obtained it."""

    label: str
    text: str
    tool: str
    call_id: str
    arguments: dict[str, Any]
    output: Any
    digest: str
    artifact_path: str | None = None
    artifact_digest: str | None = None


def _analyze(text: str, source: str) -> list[dict[str, Any]]:
    """Return deterministic findings with line attribution for one artifact."""
    findings: list[dict[str, Any]] = []
    for number, raw_line in enumerate(str(text).splitlines(), 1):
        if raw_line.startswith("---") or raw_line.startswith("+++"):
            continue
        if raw_line.startswith("-"):  # removed code is not part of the proposal
            continue
        line = raw_line[1:] if raw_line.startswith("+") else raw_line
        for rule in _RULES:
            if rule.pattern.search(line):
                findings.append({
                    "category": rule.category,
                    "severity": rule.severity,
                    "source": source,
                    "line": number,
                    "excerpt": line.strip()[:_MAX_EXCERPT],
                    "recommendation": rule.recommendation,
                })
                if len(findings) >= _MAX_FINDINGS:
                    return findings
    return findings


def _text(output: Any) -> str:
    if isinstance(output, Mapping):
        for key in ("content", "diff"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(output, default=str, ensure_ascii=False)
    return str(output) if output is not None else ""


def _artifact(path: Any) -> tuple[str | None, str | None]:
    """Content-address a regular file, up to the verifier's size limit."""
    try:
        candidate = Path(str(path))
        if not candidate.is_file() or candidate.stat().st_size > MAX_ARTIFACT_BYTES:
            return None, None
        return str(candidate), hashlib.sha256(candidate.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return None, None


async def _read_tool(
    tools: Any,
    name: str,
    arguments: Mapping[str, Any],
    *,
    label: str,
    artifact_path: Any = None,
) -> _Source | None:
    """Invoke one granted read tool and capture an independently checkable record."""
    # Defense in depth: even a mistakenly widened scope cannot turn the
    # reviewer into a writer, because only these names are ever issued.
    if name not in SEED["tool_permissions"] or tools is None:
        return None
    allowed = getattr(tools, "allow", None)
    if allowed is not None and name not in allowed:
        return None
    call = ToolCall(name=name, arguments=dict(arguments), actor="reviewer")
    try:
        outcome = await tools.execute(call)
    except Exception:  # a failed read is a missing source, never a finding
        return None
    if not bool(getattr(outcome, "success", False)):
        return None
    output = getattr(outcome, "output", None)
    path, path_digest = (None, None) if artifact_path is None else _artifact(artifact_path)
    return _Source(
        label=label,
        text=_text(output),
        tool=name,
        call_id=str(getattr(outcome, "call_id", None) or call.id),
        arguments=dict(arguments),
        output=output,
        digest=evidence_digest(output),
        artifact_path=path,
        artifact_digest=path_digest,
    )


def _tool_evidence(source: _Source, claim: str, **extra: Any) -> dict[str, Any]:
    """An evidence item anchored to a durable, digest-checked tool call."""
    item: dict[str, Any] = {
        "source": "tool",
        "tool_call": {
            "id": source.call_id,
            "tool": source.tool,
            "arguments": source.arguments,
            "output": source.output,
            "digest": source.digest,
        },
        "claim": claim,
    }
    item.update(extra)
    return item


def _artifact_evidence(source: _Source) -> dict[str, Any]:
    """An evidence item whose digest the verifier recomputes from disk."""
    return {
        "source": "artifact",
        "path": source.artifact_path,
        "digest": source.artifact_digest,
        "claim": source.artifact_digest,
        "type": "artifact",
    }


def _blocked(reason: str) -> dict[str, Any]:
    """Honest failure: no verifiable source means no review, not a fake pass."""
    return {
        "success": False,
        "error": reason,
        "output": {
            "status": "blocked",
            "read_only": True,
            "summary": reason,
            "findings": [],
            "sources": [],
        },
        "evidence": [],
    }


def make_handler(services: Any = None) -> Callable[..., Any]:
    """Build the reviewer worker used by the delegation engine.

    The returned async callable accepts the engine's standard payload
    (``objective``, ``context``, ``**kwargs``). Review inputs come from, in
    order: explicit ``paths``/``files`` read through ``read_file``, a
    ``diff_path``/``artifact_path`` file containing a patch, or the workspace
    ``git_diff`` when nothing is supplied. Every source is content-addressed
    where a file exists and is always anchored to its audited read call.
    Inline ``diff`` text alone is rejected: without a file or tool call it
    cannot be independently verified.
    """
    async def handler(objective: str = "", context: Any = None, **kwargs: Any) -> dict[str, Any]:
        ctx = dict(context or {})
        tools = ctx.get("tools")
        if tools is None and isinstance(services, Mapping):
            tools = services.get("tools") or services.get("executor")

        sources: list[_Source] = []
        paths = ctx.get("paths") or ctx.get("files") or []
        if isinstance(paths, (str, bytes)):
            paths = [paths]
        for entry in paths:
            path = str(entry)
            source = await _read_tool(
                tools, "read_file", {"path": path},
                label=f"read_file:{path}", artifact_path=path,
            )
            if source is not None:
                sources.append(source)

        diff_path = ctx.get("diff_path") or ctx.get("artifact_path")
        if diff_path:
            path = str(diff_path)
            source = await _read_tool(
                tools, "read_file", {"path": path},
                label=f"diff:{path}", artifact_path=path,
            )
            if source is not None:
                sources.append(source)

        if tools is not None and not sources:
            source = await _read_tool(
                tools, "git_diff", {"context": 3}, label="git_diff",
            )
            if source is not None:
                sources.append(source)

        inline = ctx.get("diff") or ctx.get("patch")
        if inline and not any(source.label.startswith("diff:") for source in sources):
            return _blocked(
                "review blocked: inline diff has no content-addressed artifact; "
                "supply diff_path or paths for an auditable review"
            )
        if not sources:
            return _blocked("review blocked: no verifiable diff, paths or read-only tools available")

        findings: list[dict[str, Any]] = []
        for source in sources:
            findings.extend(_analyze(source.text, source.label))
        by_label = {source.label: source for source in sources}
        summary = f"Reviewed {len(sources)} source(s); {len(findings)} finding(s)."
        claims: list[str] = []

        evidence: list[dict[str, Any]] = []
        for source in sources:
            claim = f"reviewed source {source.label}"
            claims.append(claim)
            evidence.append(_tool_evidence(source, claim, type="review_source"))
            if source.artifact_digest:
                claims.append(source.artifact_digest)
                evidence.append(_artifact_evidence(source))
        for finding in findings:
            claim = f"{finding['category']}: {finding['excerpt']}"
            finding["claim"] = claim
            claims.append(claim)
            source = by_label[finding["source"]]
            evidence.append(_tool_evidence(
                source, claim, type="code_review",
                line=finding["line"], severity=finding["severity"],
            ))

        output = {
            "status": "reviewed",
            "subject": str(objective or ctx.get("subject") or ""),
            "read_only": True,
            "summary": summary,
            "findings": findings,
            "sources": [source.label for source in sources],
            "evidence_claims": claims,
            "artifact_digests": [
                source.artifact_digest for source in sources if source.artifact_digest
            ],
            "tool_scope": list(SEED["tool_permissions"]),
        }
        return {"output": output, "evidence": evidence}

    return handler
