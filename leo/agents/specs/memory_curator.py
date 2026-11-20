"""Memory Curator specialist: evidence-first audit and approval-gated consolidation.

This module is intentionally self-contained: the integration agent's loader in
``leo.agents.registry`` imports ``SEED`` and calls :func:`make_handler` with the
runtime service mapping. The handler is bespoke (rather than the default model
delegation path) because memory deletion must never be authorized by chat text.

Real APIs only:

* :class:`~leo.memory.store.MemoryStore` for durable listing, retention data
  and (only after approval) record deletion with ``memory.changed`` events;
* :class:`~leo.memory.search.MemorySearch` for ranked retrieval evidence;
* :class:`~leo.memory.consolidation.Consolidator` for duplicate clustering,
  pair similarity scoring and source-attribution-preserving merges;
* :class:`~leo.security.permissions.PermissionManagerImpl` for HIGH-risk
  approval requests, verification and truthful audit records (injected by the
  runtime, or constructed locally over the same durable tables as a fallback).

**No free-text context flag can authorize a write.** ``objective`` and
``context`` are untrusted (``agent_delegate`` forwards model-supplied values);
they only select the audit query/category. Mutations require a durable approval
whose ``arguments_digest`` binds the exact canonical/temporary ids, category
and operation kind in the current proposal. The approval must have been
resolved by a non-chat resolver that is not the requesting agent, must be
unexpired (verified by ``PermissionManager.authorize``), and is consumed once
through a conditional status transition. Every request, rejection, grant and
consumption is written to ``audit_log`` and to an ``agent.memory_curator``
event, so no deletion is silent.

``tool_permissions`` names only tools/categories verified against the live
registry: the read-only ``memory_search`` / ``memory_list`` tools and the
``audit`` / ``observability`` categories for evidence retrieval. The entire
``memory`` category is deliberately *not* granted because it also contains
``memory_add`` / ``memory_forget``; mutations happen only through this
approval-gated handler.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from ... import db
from ...config import Settings
from ...contracts import MemoryCategory, RiskLevel, utcnow
from ...memory.consolidation import Consolidator
from ...memory.search import MemorySearch
from ...memory.store import MemoryStore, enum_value, is_expired
from ...security.permissions import PermissionManagerImpl, arguments_digest

AGENT_ID = "memory-curator"
APPROVAL_ACTOR = "agent:memory-curator"
ACTION_CONSOLIDATE = "memory_curator.consolidate"
ACTION_EXPIRE = "memory_curator.expire"

#: Same shape as every ``SPECIALIST_SPECS`` / ``_SEEDS`` row in the registry.
SEED: dict[str, Any] = {
    "id": AGENT_ID,
    "name": "Memory Curator",
    "purpose": (
        "Audit durable memories, propose deduplication and expiry with human-visible "
        "evidence, and apply only durable, scope-bound HIGH-risk approvals."
    ),
    "capabilities": [
        "memory_audit",
        "duplicate_detection",
        "retention_review",
        "evidence_reporting",
        "approved_consolidation",
    ],
    "tool_permissions": ["memory_search", "memory_list", "audit", "observability"],
    "model_preferences": ["reasoning", "structured", "tools"],
    "context_budget": 24000,
    "task_limit": 2,
    "metadata": {
        "tier": "knowledge",
        "risk": "high",
        "handler": "make_handler(services)",
        "default_mode": "proposal",
        "requires_approval_for_mutation": True,
        "approval_risk": "high",
    },
}


def _services(services: Any) -> Mapping[str, Any]:
    return services if isinstance(services, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _looks_like_store(value: Any) -> bool:
    return all(
        callable(getattr(value, method, None))
        for method in ("add", "get", "list", "update", "delete", "emit_change")
    )


def _resolve(services: Any) -> tuple[MemoryStore, MemorySearch, Any, Any]:
    """Return (store, search, event_bus, permissions) from trusted services.

    A locally constructed PermissionManagerImpl is the fallback when the
    composition root has not injected one yet. It is a real policy engine over
    the same durable approvals/audit tables; no chat/context value can stand in
    for an approval. Mutation still requires a consumed, scope-bound approval
    even if a deployment disables its own high-risk gate.
    """
    mapping = _services(services)
    store: Any = None
    for key in ("memory_store", "store"):
        if _looks_like_store(mapping.get(key)):
            store = mapping[key]
            break
    if store is None and _looks_like_store(mapping.get("memory")):
        store = mapping["memory"]
    if store is None:
        for key in ("store", "memory_store"):
            nested = _mapping(mapping.get("memory"))
            if _looks_like_store(nested.get(key)):
                store = nested[key]
                break
    if store is None:
        store = MemoryStore()

    search: Any = None
    for key in ("memory_search", "search"):
        if callable(getattr(mapping.get(key), "search", None)):
            search = mapping[key]
            break
    if search is None:
        nested = _mapping(mapping.get("memory"))
        for key in ("search", "memory_search"):
            if callable(getattr(nested.get(key), "search", None)):
                search = nested[key]
                break
    if search is None:
        search = MemorySearch(store)  # type: ignore[arg-type]

    bus = mapping.get("events") or mapping.get("event_bus") or getattr(store, "bus", None)
    permissions = mapping.get("permissions") or mapping.get("permission_manager")
    if not all(
        callable(getattr(permissions, method, None))
        for method in ("authorize", "resolve_approval", "pending_approvals")
    ):
        permissions = PermissionManagerImpl(Settings(), bus)
    return store, search, bus, permissions  # type: ignore[return-value]


def _text(value: Any, limit: int = 200) -> str:
    return " ".join(str(value or "").split())[:limit]


def _dump(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    return {
        "id": getattr(record, "id", None),
        "content": getattr(record, "content", None),
        "category": enum_value(getattr(record, "category", "")),
        "kind": enum_value(getattr(record, "kind", "")),
        "version": getattr(record, "version", None),
        "expires_at": getattr(record, "expires_at", None),
    }


def _category(context: Mapping[str, Any]) -> str | None:
    raw = context.get("category")
    if raw in (None, "", "all"):
        return None
    value = enum_value(raw)
    allowed = {item.value for item in MemoryCategory}
    if value not in allowed:
        raise ValueError(f"invalid memory category: {raw!r}; expected one of {sorted(allowed)}")
    return value


def _evidence(source: str, claim: str, **fields: Any) -> dict[str, Any]:
    return {"source": source, "claim": claim, **fields}


async def _search_evidence(search: Any, query: str, cluster: list[Any]) -> list[dict[str, Any]]:
    """Ranked, real retrieval results that support a proposed duplicate group."""
    if not query.strip():
        return []
    try:
        results = await search.search(query, limit=max(3, min(10, len(cluster) + 2)))
    except Exception as exc:  # a broken injected search must not become a fake proposal
        raise RuntimeError(f"memory search failed during curation: {exc}") from exc
    cluster_ids = {str(getattr(item, "id", "")) for item in cluster}
    seen: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for record in results:
        record_id = str(getattr(record, "id", ""))
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        evidence.append(
            _evidence(
                "MemorySearch.search",
                f"query {query[:80]!r} ranked memory {record_id} at {getattr(record, 'score', 0)}",
                id=record_id,
                score=float(getattr(record, "score", 0) or 0),
                in_duplicate_group=record_id in cluster_ids,
            )
        )
    return evidence


async def _audit(store: Any, search: Any, consolidator: Any, category: str | None, query: str) -> dict[str, Any]:
    """Build a proposal from real store/search/consolidation reads (no writes)."""
    all_records = await store.list(category=category, limit=100_000, include_expired=True)
    live = [record for record in all_records if not is_expired(record)]
    expired = [record for record in all_records if is_expired(record)]
    clusters = await consolidator.find_duplicates(category)

    duplicates: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for cluster in clusters:
        canonical = max(cluster, key=lambda r: (r.updated_at, r.confidence, r.id))
        members = [r for r in cluster if r.id != canonical.id]
        pair_score = await consolidator.similarity(canonical, members[0])
        search_hits = await _search_evidence(search, _text(canonical.content, 120), cluster)
        duplicates.append(
            {
                "canonical": _dump(canonical),
                "duplicates": [_dump(r) for r in members],
                "similarity": round(pair_score, 6),
                "search_evidence": search_hits,
                "proposed_action": "merge duplicates into canonical; retain source attribution",
                "canonical_ids": [r.id for r in cluster],
            }
        )
        evidence.append(
            _evidence(
                "Consolidator.find_duplicates",
                f"propose merging {[r.id for r in members]} into {canonical.id} "
                f"(similarity {pair_score:.3f}): {_text(canonical.content, 160)}",
                canonical_id=canonical.id,
                duplicate_ids=[r.id for r in members],
                similarity=round(pair_score, 6),
            )
        )
        evidence.extend(search_hits)

    expiry_candidates: list[dict[str, Any]] = []
    for record in expired:
        item = {
            "id": record.id,
            "expires_at": record.expires_at,
            "category": enum_value(record.category),
            "kind": enum_value(record.kind),
            "content": _text(record.content, 200),
            "proposed_action": "expire after explicit HIGH-risk approval",
        }
        expiry_candidates.append(item)
        evidence.append(
            _evidence(
                "MemoryStore.list",
                f"expiry candidate {record.id} expired at {record.expires_at}: "
                f"{_text(record.content, 160)}",
                **item,
            )
        )

    if not evidence:
        evidence.append(
            _evidence(
                "MemoryStore.list",
                f"audited {len(live)} live and {len(expired)} expired memories; "
                "no duplicate groups found",
                live_count=len(live),
                expired_count=len(expired),
            )
        )
    return {
        "duplicates": duplicates,
        "expiry_candidates": expiry_candidates,
        "evidence": evidence,
        "summary": {
            "live_memories": len(live),
            "expired_memories": len(expired),
            "duplicate_groups": len(duplicates),
            "duplicate_records": sum(len(group["duplicates"]) for group in duplicates),
            "category": category,
        },
    }


def _operation(kind: str, action: str, category: str | None, canonical_ids: list[str], target_ids: list[str]) -> dict[str, Any]:
    """Bind an approval to the exact ids of this proposal, never a whole category."""
    arguments = {
        "kind": kind,
        "actor": APPROVAL_ACTOR,
        "category": category or "*",
        "canonical_ids": sorted(str(item) for item in canonical_ids),
        "target_ids": sorted(str(item) for item in target_ids),
    }
    digest = arguments_digest(arguments)
    if not digest:
        raise RuntimeError("could not compute approval arguments digest")
    return {
        "kind": kind,
        "action": action,
        "resource": f"memory-curator:{kind}:{digest[:16]}",
        "arguments": arguments,
        "digest": digest,
    }


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("metadata") or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


async def _approval_rows(action: str, resource: str, status: str) -> list[dict[str, Any]]:
    rows = await db.fetchall(
        "SELECT * FROM approvals WHERE action=? AND resource=? AND status=? ORDER BY created_at DESC",
        (action, resource, status),
    )
    return [dict(row) for row in rows]


def _trusted_resolver(row: Mapping[str, Any]) -> str | None:
    """Only a non-chat resolver distinct from the requester may grant authority."""
    if not row.get("resolved_at"):
        return None
    resolver = str(_metadata(row).get("resolved_by") or "").strip()
    if not resolver:
        return None
    lowered = resolver.casefold()
    if lowered.startswith("chat:") or lowered == APPROVAL_ACTOR.casefold():
        return None
    requester = str(row.get("actor") or row.get("requested_by") or "").strip().casefold()
    if requester and lowered == requester:
        return None
    return resolver


async def _audit_rejection(
    action: str, resource: str, reason: str, approval_id: Any = None, resolver: Any = None
) -> None:
    """Write a truthful denial row for an approval we refused to honor."""
    await db.execute(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (
            utcnow(), APPROVAL_ACTOR, action, resource, RiskLevel.HIGH.value, "denied", reason,
            json.dumps({"approval_id": approval_id, "resolver": resolver}, ensure_ascii=False),
        ),
    )


async def _consume(approval_id: str) -> bool:
    """One-shot consumption; the conditional update prevents concurrent replay."""
    async with db.transaction() as conn:
        cursor = await conn.execute(
            "UPDATE approvals SET status='consumed' WHERE id=? AND status='approved'",
            (approval_id,),
        )
        return cursor.rowcount == 1


async def _authorize_operation(permissions: Any, operation: Mapping[str, Any]) -> dict[str, Any]:
    """Consume one valid durable approval, or return the pending request id."""
    action = str(operation["action"])
    resource = str(operation["resource"])
    for row in await _approval_rows(action, resource, "approved"):
        resolver = _trusted_resolver(row)
        if resolver is None:
            await _audit_rejection(
                action, resource, "approval resolver is chat, self or unknown",
                row.get("id"), _metadata(row).get("resolved_by"),
            )
            continue
        decision = await permissions.authorize(
            APPROVAL_ACTOR, action, resource, RiskLevel.HIGH,
            approval_id=row["id"],
            arguments=operation["arguments"],
            arguments_digest=operation["digest"],
            scope_ids=operation["arguments"]["target_ids"],
        )
        if not decision.allowed:
            continue  # PermissionManager already audited the binding/expiry denial
        if not await _consume(row["id"]):
            await _audit_rejection(action, resource, "approval already consumed", row["id"], resolver)
            continue
        return {
            "applied": True,
            "approval_id": row["id"],
            "resolver": resolver,
            "digest": operation["digest"],
            "action": action,
            "resource": resource,
        }

    # No usable approval: reuse the matching pending request or ask for a new one.
    for row in await _approval_rows(action, resource, "pending"):
        if str(_metadata(row).get("arguments_digest") or "") == operation["digest"]:
            return {
                "applied": False,
                "approval_id": row["id"],
                "resolver": None,
                "reason": "awaiting HIGH-risk approval",
                "digest": operation["digest"],
                "action": action,
                "resource": resource,
            }
    decision = await permissions.authorize(
        APPROVAL_ACTOR, action, resource, RiskLevel.HIGH,
        arguments=operation["arguments"],
        arguments_digest=operation["digest"],
        scope_ids=operation["arguments"]["target_ids"],
    )
    return {
        "applied": False,
        "approval_id": decision.approval_id,
        "resolver": None,
        "reason": decision.reason,
        "digest": operation["digest"],
        "action": action,
        "resource": resource,
    }


async def _publish_audit(bus: Any, payload: Mapping[str, Any]) -> None:
    emit = getattr(bus, "emit", None)
    if callable(emit):
        await emit("agent.memory_curator", **dict(payload))


def make_handler(services: Any = None) -> Callable[..., Any]:
    """Build the durable, approval-gated curator worker.

    The returned async callable accepts the standard engine payload keywords
    ``objective`` and ``context`` (plus ignored extras). Those values are
    untrusted and never authorize a mutation. Each call audits the store and
    looks up durable HIGH-risk approvals bound to the exact ids in the current
    proposal; approved and correctly resolved operations are applied once and
    their approvals consumed. Otherwise a pending approval id is returned.
    """
    store, search, bus, permissions = _resolve(services)
    consolidator = Consolidator(store)

    async def handle(objective: str = "", context: Any = None, **kwargs: Any) -> dict[str, Any]:
        ctx: Mapping[str, Any] = _mapping(context)
        category = _category(ctx)
        query = _text(ctx.get("query") or objective or "durable memory", 240)
        proposal = await _audit(store, search, consolidator, category, query)

        applied_evidence: list[dict[str, Any]] = []
        approvals: list[dict[str, Any]] = []
        applied: dict[str, Any] = {
            "consolidated_ids": [],
            "removed_duplicate_ids": [],
            "purged_expiry_ids": [],
            "approvals_consumed": [],
            "category": category,
        }

        if proposal["duplicates"]:
            canonical_ids = [group["canonical"]["id"] for group in proposal["duplicates"]]
            target_ids = [item["id"] for group in proposal["duplicates"] for item in group["duplicates"]]
            # Re-check the exact scope BEFORE authorizing: a category-wide
            # consolidate call must never include records added after the proposal.
            clusters = await consolidator.find_duplicates(category)
            current_scope = sorted(str(record.id) for cluster in clusters for record in cluster)
            if current_scope != sorted(canonical_ids + target_ids):
                await _audit_rejection(
                    ACTION_CONSOLIDATE, "memory-curator:consolidate", "proposal scope changed",
                )
            else:
                operation = _operation(
                    "consolidate", ACTION_CONSOLIDATE, category, canonical_ids, target_ids
                )
                authorization = await _authorize_operation(permissions, operation)
                approvals.append(authorization)
                if authorization["applied"]:
                    consolidation = await consolidator.consolidate(category)
                    applied["consolidated_ids"] = [record.id for record in consolidation.records]
                    applied["removed_duplicate_ids"] = list(consolidation.removed_ids)
                    applied["approvals_consumed"].append({
                        "approval_id": authorization["approval_id"],
                        "resolver": authorization["resolver"],
                        "action": operation["action"],
                        "arguments_digest": operation["digest"],
                    })
                    applied_evidence.append(
                        _evidence(
                            "Consolidator.consolidate",
                            f"HIGH-risk approval {authorization['approval_id']} resolved by "
                            f"{authorization['resolver']}: merged {len(consolidation.removed_ids)} "
                            f"duplicates into {applied['consolidated_ids']}; each merge emits "
                            "memory.changed with source attribution",
                            approval_id=authorization["approval_id"],
                            resolver=authorization["resolver"],
                            arguments_digest=operation["digest"],
                            removed_duplicate_ids=applied["removed_duplicate_ids"],
                        )
                    )

        # Expiry scope is recomputed after consolidation and approved separately.
        current_records = await store.list(
            category=category, limit=100_000, include_expired=True
        )
        expiry_ids = [record.id for record in current_records if is_expired(record)]
        if expiry_ids:
            if sorted(expiry_ids) != sorted(item["id"] for item in proposal["expiry_candidates"]):
                await _audit_rejection(
                    ACTION_EXPIRE, "memory-curator:expire", "expiry scope changed",
                )
            else:
                operation = _operation("expire", ACTION_EXPIRE, category, [], expiry_ids)
                authorization = await _authorize_operation(permissions, operation)
                approvals.append(authorization)
                if authorization["applied"]:
                    for memory_id in expiry_ids:
                        if await store.delete(memory_id):
                            applied["purged_expiry_ids"].append(memory_id)
                    applied["approvals_consumed"].append({
                        "approval_id": authorization["approval_id"],
                        "resolver": authorization["resolver"],
                        "action": operation["action"],
                        "arguments_digest": operation["digest"],
                    })
                    applied_evidence.append(
                        _evidence(
                            "MemoryStore.delete",
                            f"HIGH-risk approval {authorization['approval_id']} resolved by "
                            f"{authorization['resolver']}: purged expired memories {expiry_ids}",
                            approval_id=authorization["approval_id"],
                            resolver=authorization["resolver"],
                            arguments_digest=operation["digest"],
                            purged_expiry_ids=expiry_ids,
                        )
                    )

        was_applied = bool(applied["approvals_consumed"])
        result: dict[str, Any] = {
            "success": True,
            "agent_id": AGENT_ID,
            "status": "applied" if was_applied else "awaiting_approval",
            "requires_approval": not was_applied,
            "summary": proposal["summary"],
            "duplicates": proposal["duplicates"],
            "expiry_candidates": proposal["expiry_candidates"],
            "approvals": approvals,
            "applied": applied if was_applied else None,
            "evidence": proposal["evidence"] + applied_evidence,
        }
        if not was_applied:
            result["reason"] = (
                "read-only audit; a non-chat operator must resolve the HIGH-risk approval(s) "
                "before the next call can apply them"
            )
        await _publish_audit(
            bus,
            {
                "action": "curate",
                "status": result["status"],
                "objective": _text(objective, 240),
                "approval_ids": [item["approval_id"] for item in approvals if item["approval_id"]],
                "approvals_consumed": applied["approvals_consumed"],
                "consolidated_ids": applied["consolidated_ids"],
                "removed_duplicate_ids": applied["removed_duplicate_ids"],
                "purged_expiry_ids": applied["purged_expiry_ids"],
                "evidence_count": len(result["evidence"]),
            },
        )
        return result

    return handle


__all__ = [
    "SEED", "AGENT_ID", "APPROVAL_ACTOR", "ACTION_CONSOLIDATE", "ACTION_EXPIRE",
    "make_handler",
]
