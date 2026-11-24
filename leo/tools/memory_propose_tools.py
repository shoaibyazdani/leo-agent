"""Proposal-only memory writes for the post-turn review.

The review may notice a durable fact, but it must not bypass the learning
engine's sanitizer, dedupe, approval policy or governance. This tool builds a
real ``NoteDraft`` with durable evidence references and hands it to
``LearningEngine.propose`` — the same path every other autonomous note uses.
It never calls ``accept`` and never writes ``memory_items`` directly.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .. import db
from ..contracts import MemoryCategory, MemoryKind, RiskLevel, ToolSpec

_MAX_CONTENT = 1200
_MAX_TAGS = 10
_SOURCE_KINDS = frozenset({"user", "tool", "run", "external"})


def _learning(ctx: dict[str, Any]) -> Any:
    services = ctx.get("services")
    if not isinstance(services, dict) or services.get("learning") is None:
        raise RuntimeError(
            "learning service is not available in the tool context; the "
            "LearningEngine must be injected as ctx['services']['learning']"
        )
    return services["learning"]


def _category(value: Any) -> MemoryCategory:
    raw = str(value or MemoryCategory.SEMANTIC.value).strip().lower()
    try:
        return MemoryCategory(raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in MemoryCategory)
        raise ValueError(f"category must be one of: {allowed}") from exc


def _kind(value: Any) -> MemoryKind:
    raw = str(value or MemoryKind.UNVERIFIED.value).strip().lower()
    try:
        return MemoryKind(raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in MemoryKind)
        raise ValueError(f"kind must be one of: {allowed}") from exc


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError("tags must be a list of strings")
    tags = [str(item).strip() for item in value if str(item).strip()]
    if len(tags) > _MAX_TAGS:
        raise ValueError(f"at most {_MAX_TAGS} tags are accepted")
    return tags[:20]


async def _evidence(
    conversation_id: str | None, evidence_ids: Any
) -> list[Any]:
    """Build durable evidence refs from real message rows."""
    from ..learning import EvidenceRef

    if not conversation_id or not isinstance(evidence_ids, (list, tuple)):
        return []
    ids: list[int] = []
    for value in evidence_ids[:20]:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = await db.fetchall(
        "SELECT id, content FROM messages WHERE conversation_id=? "
        f"AND id IN ({placeholders}) ORDER BY id",
        (conversation_id, *ids),
    )
    refs = []
    for row in rows:
        content = str(row["content"] or "")
        refs.append(EvidenceRef(
            id=str(row["id"]),
            kind="message",
            hash=hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:32],
            conversation_id=conversation_id,
        ))
    return refs


async def memory_propose(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Create an inert learning proposal; the engine decides whether it can write."""
    from ..learning import NoteDraft

    learning = _learning(ctx)
    content = str(args.get("content") or "").strip()
    if not content:
        raise ValueError("content must be a non-empty string")
    if len(content) > _MAX_CONTENT:
        raise ValueError(f"content exceeds {_MAX_CONTENT} characters")
    source_kind = str(args.get("source_kind") or "external").strip().lower()
    if source_kind not in _SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of: {', '.join(sorted(_SOURCE_KINDS))}")
    try:
        confidence = float(args.get("confidence", 0.6))
    except (TypeError, ValueError):
        raise ValueError("confidence must be a number between 0 and 1") from None
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    conversation_id = str(
        args.get("conversation_id")
        or ctx.get("conversation_id")
        or ctx.get("correlation_id")
        or ""
    ).strip() or None
    refs = await _evidence(conversation_id, args.get("evidence_ids"))
    draft = NoteDraft(
        content=content,
        category=_category(args.get("category")),
        kind=_kind(args.get("kind")),
        confidence=confidence,
        source_kind=source_kind,
        conversation_id=conversation_id,
        tags=_tags(args.get("tags")),
        evidence=refs,
        reason="post-turn skill review",
    )
    proposal = await learning.propose(draft)
    return {
        "success": True,
        "proposal": proposal,
        "status": str(proposal.get("status") or ""),
        "approval_id": proposal.get("approval_id"),
        "evidence_count": len(refs),
    }


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="memory_propose",
            description=(
                "Propose one durable memory note through the real LearningEngine. "
                "The engine applies its sanitizer, dedupe and approval policy; this "
                "tool never writes memory directly. One fact per call."
            ),
            category="memory",
            risk=RiskLevel.MEDIUM,
            permissions=["memory:write"],
            input_schema=_schema(
                {
                    "content": {"type": "string", "minLength": 1, "maxLength": _MAX_CONTENT},
                    "category": {"type": "string"},
                    "kind": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source_kind": {
                        "type": "string",
                        "enum": ["user", "tool", "run", "external"],
                    },
                    "conversation_id": {"type": "string", "maxLength": 120},
                    "evidence_ids": {"type": "array", "items": {"type": "integer"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                ["content"],
            ),
            resource_field="",
        ),
        memory_propose,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the memory_propose tool and return its name."""
    names: list[str] = []
    for spec, handler in _SPECS:
        registry.register(spec, handler)
        names.append(spec.name)
    return names


__all__ = ["memory_propose", "register"]
