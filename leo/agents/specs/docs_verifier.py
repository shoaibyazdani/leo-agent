"""Documentation-verifier specialist: check claims against real local state.

The handler verifies each supplied claim against the filesystem (relative to a
repo root bound at construction) and, for skill claims, the injected
``SkillRegistry``. Every claim is reported as ``verified``, ``stale`` or
``blocked`` with the reason it was decided; a missing service is never reported
as a verified claim and an unsupported claim kind is never silently skipped.
No claim is edited and no approval is resolved.

Services come only from ``make_handler(services)``; the handler context carries
claims and filters, never service objects. This is a bespoke handler
(``handler_kind = bespoke:make_handler``).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SEED: dict[str, Any] = {
    "id": "docs-verifier",
    "name": "Documentation Verifier",
    "purpose": (
        "Verify documentation claims against the live filesystem and skill registry, "
        "reporting each claim as verified, stale or blocked with the evidence used "
        "to decide it."
    ),
    "capabilities": [
        "claim_extraction",
        "filesystem_verification",
        "registry_cross_check",
        "stale_claim_reporting",
    ],
    # Exact live read-only filesystem tool names, not the whole 'filesystem'
    # category (which also grants write_file, file_copy and file_move).
    "tool_permissions": ["read_file", "search_files", "list_files"],
    "model_preferences": ["structured", "text", "tools"],
    "context_budget": 20000,
    "task_limit": 3,
    "metadata": {"handler_kind": "bespoke:make_handler"},
}


async def _resolved(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


def _claims(objective: Any, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = context.get("claims")
    if raw is None and objective:
        text = str(objective).strip()
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"docs-verifier could not parse claim JSON: {exc}") from exc
    if raw is None:
        raise ValueError("docs-verifier requires claims in context['claims'] or a JSON objective")
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("docs-verifier requires a non-empty list of claim objects")
    claims: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            claims.append({"kind": "file", "value": item})
        elif isinstance(item, Mapping):
            claims.append({
                "kind": str(item.get("kind") or "file"),
                "value": str(item.get("value") or item.get("path") or "").strip(),
            })
        else:
            raise ValueError("docs-verifier claim entries must be objects or strings")
    return claims


def make_handler(services: Mapping[str, Any] | None = None) -> Callable[..., Any]:
    """Bind the verifier to an optional repo root and ``skills`` service.

    Both are read here, once, from the injected services mapping. A missing
    ``skills`` service produces ``blocked`` (not ``verified``) skill claims;
    a missing repo root falls back to the process working directory.
    """

    services = dict(services or {})
    skills = services.get("skills")
    configured_root = services.get("repo_root")
    root = Path(str(configured_root)) if configured_root else Path.cwd()

    async def handler(objective: Any = None, context: Any = None, **_: Any) -> dict[str, Any]:
        claims = _claims(objective, dict(context or {}))
        try:
            resolved_root = root.resolve()
        except OSError as exc:
            raise ValueError(f"docs-verifier cannot resolve repo root: {exc}") from exc

        results: list[dict[str, Any]] = []
        for claim in claims:
            kind = claim["kind"].lower()
            value = claim["value"]
            status = "blocked"
            detail = f"unsupported claim kind: {kind!r}"
            checked: dict[str, Any] = {}
            if not value:
                detail = "claim has no value"
            elif kind == "file":
                try:
                    candidate = (resolved_root / value).resolve()
                    inside = candidate == resolved_root or resolved_root in candidate.parents
                except OSError as exc:
                    inside, candidate = False, None
                    detail = f"cannot resolve path: {exc}"
                if candidate is not None and not inside:
                    detail = "claim path escapes the configured repo root"
                elif candidate is not None:
                    status = "verified" if candidate.is_file() else "stale"
                    detail = "file exists" if status == "verified" else "file is missing"
                    checked = {"path": str(candidate)}
            elif kind == "skill":
                if skills is None:
                    detail = "skills service was not injected at bind time"
                else:
                    record = await _resolved(skills.get(value))
                    status = "verified" if record is not None else "stale"
                    detail = "skill is registered" if status == "verified" else "skill is absent"
                    checked = {"name": value}
            results.append({
                "kind": kind,
                "value": value,
                "status": status,
                "detail": detail,
                "checked": checked,
            })

        counts = dict(sorted(Counter(item["status"] for item in results).items()))
        verified = sum(1 for item in results if item["status"] == "verified")
        summary = {
            "count": len(results),
            "verified": verified,
            "stale": counts.get("stale", 0),
            "blocked": counts.get("blocked", 0),
            "all_verified": verified == len(results),
        }
        return {
            "success": True,
            "output": {
                "repo_root": str(resolved_root),
                "summary": summary,
                "claims": results,
                "edited": False,
            },
            "evidence": [
                {
                    "source": "docs-verifier",
                    "claim": (
                        f"Verified {verified}/{len(results)} claims against the local "
                        "filesystem and registry"
                    ),
                    "counts": counts,
                },
                {
                    "source": "docs-verifier",
                    "claim": "Verification is read-only; no documentation was changed",
                    "edited": False,
                },
            ],
        }

    return handler
