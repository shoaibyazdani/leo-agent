"""Release-auditor specialist: truthful, evidence-backed release readiness.

This specialist uses a bespoke execution path (:func:`make_handler`), not the
default model/delegation worker. It cross-checks claimed capabilities in
documents and status records against local evidence (code files, test files,
the live tool/agent registries and durable status rows), flags capabilities
that are described as live while their verification is mock-only, and emits a
readiness report where every verdict carries its source. Absent evidence is
reported as ``unverified``/``not_ready``; it is never upgraded to success.

The audit is strictly read-only: it never edits files, calls providers, sends
messages, resolves approvals, or writes to the database. Tool grants are
individual read-only tool *names* (not broad categories) verified against the
live ``ToolSpec`` registry; write/send/delegate tools are deliberately excluded.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ... import db
from ...contracts import utcnow

_AGENT_ID = "release-auditor"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Each name is a registered read-only ToolSpec (see tests/test_agents_release_auditor.py).
TOOL_GRANTS: tuple[str, ...] = (
    "read_file",
    "list_files",
    "search_files",
    "event_query",
    "audit_query",
    "audit_tail",
    "agent_list",
    "agent_status",
    "agent_result",
    "task_list",
    "task_status",
    "task_result",
    "integration_status",
    "enrollment_status",
    "skill_list",
    "tool_registry_list",
    "model_providers",
)

SEED: dict[str, Any] = {
    "id": _AGENT_ID,
    "name": "Release Auditor",
    "purpose": (
        "Verify documentation and status claims against code, tests and durable "
        "records; flag mock-only capabilities presented as live; and emit a "
        "readiness report in which every verdict is backed by cited evidence."
    ),
    "capabilities": [
        "claim_verification",
        "status_audit",
        "mock_capability_detection",
        "evidence_attribution",
        "readiness_reporting",
    ],
    "tool_permissions": list(TOOL_GRANTS),
    "model_preferences": ["reasoning", "structured", "tools"],
    "context_budget": 24000,
    "task_limit": 2,
    "metadata": {
        "tier": "assurance",
        "risk": "low",
        "read_only": True,
        "handler": "make_handler(services)",
        "handler_kind": "bespoke:make_handler",
        "approval_authority": "human_only",
        "never_self_approve": True,
    },
}

_LIVE_STATUSES = frozenset({"live", "operational", "connected", "active", "production"})
_MOCK_VERIFICATIONS = frozenset({
    "mock", "mock-tested", "mocked", "stub", "fake", "simulated",
    "not-verified", "unverified", "import-failed",
})
_COUNT_TABLES = frozenset({"tool_calls", "agent_runs", "tasks", "events", "audit_log", "integrations", "skills"})
_CODE_SUFFIXES = frozenset({".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".go", ".rs", ".java"})
_TEST_SUFFIXES = frozenset({".py", ".js", ".ts", ".tsx", ".jsx"})
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})
_SERVICE_KEYS = ("integrations", "integration_records", "gateway", "channel_registry", "channels")
_TOOL_MODULE_REGISTER = ("event_tools", "approval_tools", "agent_tools", "integration_tools",
                         "skill_tools", "task_tools", "model_tools", "file_tools", "memory_tools",
                         "data_tools", "git_tools", "automation_tools", "context_tools")


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    # sqlite3.Row is key-addressable but is not registered as a Mapping.
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        try:
            return dict(zip(value.keys(), value))
        except (TypeError, ValueError):
            pass
    return {}


def _text(value: Any, limit: int = 400) -> str:
    text = value if isinstance(value, str) else str(value) if value is not None else ""
    return text.strip()[:limit]


def _decoded(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_text(item, 300) for item in value if _text(item, 300)]
    return []


def _safe_path(value: str) -> Path | None:
    """Resolve a repo-relative path; never escape the repository root."""
    if not value or "\x00" in value:
        return None
    candidate = Path(value)
    try:
        resolved = (candidate if candidate.is_absolute() else _REPO_ROOT / candidate).resolve()
        resolved.relative_to(_REPO_ROOT)
    except (OSError, ValueError):
        return None
    return resolved


def _read_text(path: str) -> str | None:
    resolved = _safe_path(path)
    if resolved is None or not resolved.is_file():
        return None
    try:
        return resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _path_checks(claim: Mapping[str, Any]) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    checks: dict[str, bool] = {}
    evidence: list[dict[str, Any]] = []
    for key, kind, suffixes in (
        ("code", "code", _CODE_SUFFIXES),
        ("tests", "test", _TEST_SUFFIXES),
        ("doc", "doc", _DOC_SUFFIXES),
    ):
        paths = _paths(claim.get(key) or claim.get(key + "_paths"))
        if not paths:
            continue
        passed = True
        for path in paths:
            resolved = _safe_path(path)
            text = _read_text(path)
            ok = (
                resolved is not None
                and resolved.suffix.lower() in suffixes
                and text is not None
                and bool(text.strip())
            )
            if kind == "test" and ok:
                ok = "def test_" in text or ".test(" in text or "it(" in text
            if kind == "doc" and ok:
                quote = _text(claim.get("quote"), 300)
                needle = quote or _text(claim.get("claim"), 300)
                ok = not needle or needle in text
            passed = passed and ok
            evidence.append({
                "source": f"{kind}:{path}",
                "claim": f"{path} exists as {kind} evidence and supports the claim",
                "value": bool(ok),
            })
        checks[key] = passed
    return checks, evidence


def _normalize_record(record: Any, origin: str) -> dict[str, Any] | None:
    item = _mapping(record)
    name = _text(item.get("name") or item.get("channel") or item.get("id"), 120)
    if not name:
        return None
    metadata = _mapping(item.get("metadata"))
    capabilities = _decoded(item.get("capabilities"), [])
    if not isinstance(capabilities, list):
        capabilities = []
    return {
        "name": name,
        "kind": _text(item.get("kind"), 80) or "unknown",
        "status": _text(item.get("status"), 60).lower(),
        "capabilities": [_text(value, 80) for value in capabilities],
        "verification": _text(item.get("verification"), 80).lower(),
        "live_verified": metadata.get("live_verified"),
        "reason": _text(metadata.get("reason") or item.get("reason"), 300),
        "origin": origin,
    }


def _service_records(services: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in _SERVICE_KEYS:
        source = services.get(key)
        if source is None:
            continue
        records: list[Any] | None = None
        getter = getattr(source, "integration_records", None)
        if callable(getter):
            try:
                records = list(getter() or [])
            except Exception:
                records = None
        if records is None and isinstance(source, Mapping):
            records = list(source.values())
        if records is None:
            snapshot = getattr(source, "snapshot", None)
            if callable(snapshot):
                try:
                    snapshot_value = snapshot()
                    if isinstance(snapshot_value, Mapping):
                        records = list(snapshot_value.values())
                except Exception:
                    records = None
        if records is None:
            continue
        normalized = [item for item in (_normalize_record(r, f"services:{key}") for r in records) if item]
        if normalized:
            return normalized
    return []


async def _status_records(services: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    supplied = context.get("status_records") or context.get("integration_records")
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        records = [item for item in (_normalize_record(r, "context") for r in supplied) if item]
        if records:
            return records
    records = _service_records(services)
    if records:
        return records
    rows = await db.fetchall(
        "SELECT name,kind,status,capabilities,verification,config_ref FROM integrations ORDER BY name"
    )
    return [item for item in (_normalize_record(row, "db:integrations") for row in rows) if item]


def _live_tool_names(services: Mapping[str, Any]) -> set[str]:
    for key in ("tool_registry", "tools_registry"):
        registry = services.get(key)
        specs = getattr(registry, "list_specs", None) if registry is not None else None
        if callable(specs):
            return {str(getattr(s, "name", "")) for s in specs()}
    for key in ("tools", "executor", "tool_executor"):
        service = services.get(key)
        registry = getattr(service, "registry", None)
        specs = getattr(registry, "list_specs", None) if registry is not None else None
        if callable(specs):
            return {str(getattr(s, "name", "")) for s in specs()}
    # Local fallback: enumerate the real ToolSpec declarations without any
    # provider call or credential use. Registration only builds spec objects.
    from importlib import import_module

    from ...tools.builtins import register_builtins
    from ...tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_builtins(registry)
    for module_name in _TOOL_MODULE_REGISTER:
        try:
            import_module(f"...tools.{module_name}", __package__).register(registry, None)
        except Exception:
            continue
    return {str(getattr(s, "name", "")) for s in registry.list_specs()}


async def _live_agent_ids(services: Mapping[str, Any]) -> set[str]:
    for key in ("agent_registry", "agents", "agents_registry"):
        registry = services.get(key)
        listing = getattr(registry, "list", None) if registry is not None else None
        if callable(listing):
            try:
                return {str(getattr(spec, "id", "")) for spec in await listing()}
            except Exception:
                continue
    engine = services.get("agent_engine") or services.get("agents_engine")
    registry = getattr(engine, "registry", None)
    listing = getattr(registry, "list", None) if registry is not None else None
    if callable(listing):
        try:
            return {str(getattr(spec, "id", "")) for spec in await listing()}
        except Exception:
            pass
    rows = await db.fetchall("SELECT id FROM agents")
    return {str(row["id"]) for row in rows}


async def _row_count(table: str) -> int:
    if table not in _COUNT_TABLES:
        return -1
    row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
    return int(row["n"]) if row is not None else -1


def _mock_finding(
    record: Mapping[str, Any], *, claimed_live: bool = False
) -> dict[str, Any] | None:
    """Flag a mock-only capability that is presented as live by a record or claim."""
    status = str(record.get("status") or "")
    verification = str(record.get("verification") or "")
    mock = verification in _MOCK_VERIFICATIONS
    explicit_live = status in _LIVE_STATUSES
    claims_live = record.get("live_verified") is True
    if not mock or not (explicit_live or claims_live or claimed_live):
        return None
    capabilities = ", ".join(record.get("capabilities") or []) or "capabilities"
    return {
        "kind": "mock_only_presented_as_live",
        "severity": "high",
        "integration": record["name"],
        "reason": (
            f"{record['name']} advertises {capabilities} with status '{status}' "
            f"but verification is '{verification}' (no live evidence)"
        ),
        "evidence": [{
            "source": f"{record.get('origin', 'status')}:{record['name']}",
            "claim": f"status={status}; verification={verification}; capabilities={capabilities}",
            "value": False,
        }],
    }


async def _evaluate_claim(
    claim: Mapping[str, Any],
    tools: set[str],
    agents: set[str],
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: dict[str, bool] = {}
    evidence: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    name = _text(claim.get("claim") or claim.get("name") or claim.get("id"), 300) or "unnamed claim"
    source = _text(claim.get("source") or claim.get("doc"), 200) or "context"

    path_checks, path_evidence = _path_checks(claim)
    checks.update(path_checks)
    evidence.extend(path_evidence)

    expected_tools = _paths(claim.get("tools"))
    if expected_tools:
        checks["tools"] = all(value in tools for value in expected_tools)
        evidence.append({
            "source": "tool_registry",
            "claim": f"registered tools: {', '.join(expected_tools)}",
            "value": checks["tools"],
        })

    expected_agents = _paths(claim.get("agents"))
    if expected_agents:
        checks["agents"] = all(value in agents for value in expected_agents)
        evidence.append({
            "source": "agents_table",
            "claim": f"registered agents: {', '.join(expected_agents)}",
            "value": checks["agents"],
        })

    execution = _mapping(claim.get("executed"))
    if execution:
        table = _text(execution.get("table"), 60)
        minimum = execution.get("min_rows", 1)
        try:
            threshold = max(0, int(minimum))
        except (TypeError, ValueError):
            threshold = 1
        count = await _row_count(table)
        checks["executed"] = table in _COUNT_TABLES and count >= threshold
        evidence.append({
            "source": f"db:{table}",
            "claim": f"at least {threshold} durable execution row(s)",
            "value": checks["executed"],
        })

    integration = _text(claim.get("integration"), 120)
    if integration:
        record = records.get(integration)
        claimed_status = _text(claim.get("status"), 60).lower()
        claimed_live = claimed_status in _LIVE_STATUSES or any(
            word in name.lower() for word in ("live", "production", "operational")
        )
        if record is None:
            checks["integration"] = False
            evidence.append({
                "source": f"status:{integration}",
                "claim": "status record exists",
                "value": False,
            })
        else:
            finding = _mock_finding(record, claimed_live=claimed_live)
            if finding is not None and (claimed_live or not claimed_status):
                checks["integration"] = False
                findings.append(finding)
            elif claimed_status and claimed_status != str(record.get("status") or ""):
                checks["integration"] = False
            else:
                checks["integration"] = True
            evidence.append({
                "source": f"{record.get('origin', 'status')}:{integration}",
                "claim": (
                    f"claimed status '{claimed_status or 'any'}' matches recorded "
                    f"status '{record.get('status')}' / verification '{record.get('verification')}'"
                ),
                "value": checks["integration"],
            })

    if checks and all(checks.values()):
        verdict = "verified"
    elif checks:
        verdict = "contradicted"
    else:
        verdict = "unverified"
    if findings:
        verdict = "mock_only"
    report = {
        "claim": name,
        "source": source,
        "verdict": verdict,
        "checks": checks,
        "reasons": [f"failed check: {key}" for key, ok in checks.items() if not ok]
        or (["no evidence supplied; claim cannot be verified"] if not checks else []),
        "findings": findings,
    }
    return report, evidence


def make_handler(services: Any = None) -> Callable[..., Any]:
    """Build the read-only release-audit worker bound to injected services.

    The returned async callable accepts the standard engine keywords
    ``objective`` and ``context`` (plus ignored extras). Claims arrive as
    ``context['claims']``; status records may be supplied as
    ``context['status_records']``, read from injected integration services, or
    read from the durable ``integrations`` table. No service is mandatory: the
    local database and repository files are the default evidence sources. The
    handler always returns a report; a blocked release is an honest verdict,
    not a failed audit.
    """

    async def handle(objective: Any = "", context: Any = None, **kwargs: Any) -> dict[str, Any]:
        ctx = _mapping(context)
        injected: Mapping[str, Any] = _mapping(ctx.get("services") or services)
        records = await _status_records(injected, ctx)
        by_name = {str(record["name"]): record for record in records}
        tools = _live_tool_names(injected)
        agents = await _live_agent_ids(injected)

        supplied = ctx.get("claims") or ctx.get("status_claims") or []
        if isinstance(supplied, Mapping):
            supplied = [supplied]
        if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
            supplied = []
        claims = [item for item in supplied if isinstance(item, Mapping)]
        if not claims:
            objective_text = _text(objective, 300)
            if objective_text:
                claims = [{"claim": objective_text, "source": "objective"}]

        reports: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        for claim in claims:
            report, claim_evidence = await _evaluate_claim(claim, tools, agents, by_name)
            reports.append(report)
            evidence.extend(claim_evidence)
            findings.extend(report["findings"])

        # Status records are audited even when no explicit claim names them:
        # a mock-only row that calls itself live is itself a defect.
        for record in records:
            finding = _mock_finding(record)
            if finding is not None and finding not in findings:
                findings.append(finding)
            evidence.append({
                "source": f"{record.get('origin', 'status')}:{record['name']}",
                "claim": (
                    f"{record['name']} status={record.get('status')} "
                    f"verification={record.get('verification')}"
                ),
                "value": finding is None,
            })

        counts = {verdict: 0 for verdict in ("verified", "contradicted", "mock_only", "unverified")}
        for report in reports:
            counts[report["verdict"]] = counts.get(report["verdict"], 0) + 1
        blocking = [item for item in findings if item["severity"] in ("high", "critical")]
        ready = bool(reports) and not blocking and counts["verified"] == len(reports)
        readiness = "ready" if ready else "blocked" if blocking else "not_ready"
        output = {
            "agent_id": _AGENT_ID,
            "readiness": readiness,
            "ready": ready,
            "objective": _text(objective, 300),
            "counts": {**counts, "claims": len(reports), "findings": len(findings)},
            "claims": reports,
            "findings": findings,
            "status_records": [
                {key: record[key] for key in ("name", "status", "capabilities", "verification", "origin")}
                for record in records
            ],
            "verdict_rule": "verified only when all declared evidence checks pass",
            "generated_at": utcnow(),
        }
        return {"success": True, "agent_id": _AGENT_ID, "output": output, "evidence": evidence}

    return handle


__all__ = ["SEED", "TOOL_GRANTS", "make_handler"]
