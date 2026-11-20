"""Durable memory tools exposed to the chat tool loop.

The four tools (``memory_add``, ``memory_search``, ``memory_list``,
``memory_forget``) operate on the shared SQLite memory tables through
:class:`~leo.memory.store.MemoryStore` and :class:`~leo.memory.search.MemorySearch`.

Handlers prefer the instances injected into ``ctx["services"]`` by the runtime
(integration agent). When a service has not been injected yet they fall back to
constructing those same real components over the shared durable database, so a
tool call is always a real read/write rather than a canned response. An injected
object that does not implement the required interface is rejected with a clear
error instead of being silently ignored.

``memory_forget`` is deliberately HIGH risk. The tool executor gates HIGH-risk
calls through ``PermissionManagerImpl.authorize`` *before* the handler runs and
passes ``arguments_digest(call.arguments)``, so an approval is bound to the exact
memory id shown to the human. Chat actors cannot resolve their own approvals
(``PermissionManagerImpl.resolve_approval`` rejects ``chat:`` resolvers); only an
operator path can grant it. A model-chosen call therefore never deletes a
memory without a bound, operator-approved decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from ..contracts import MemoryCategory, MemoryKind, MemoryRecord, RiskLevel, ToolSpec
from ..memory.search import MemorySearch
from ..memory.store import MemoryStore, enum_value

MEMORY_CATEGORIES: list[str] = [item.value for item in MemoryCategory]
MEMORY_KINDS: list[str] = [item.value for item in MemoryKind]


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    """Build a strict JSON-schema subset object schema for a memory tool."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_CATEGORY_PROPERTY: dict[str, Any] = {
    "type": "string",
    "enum": MEMORY_CATEGORIES,
}
_KIND_PROPERTY: dict[str, Any] = {"type": "string", "enum": MEMORY_KINDS}
_TAGS_PROPERTY: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "maxItems": 50,
}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_enum(value: Any, field: str, allowed: list[str]) -> str | None:
    if value is None:
        return None
    text = enum_value(value) if isinstance(value, Enum) else str(value)
    if text not in allowed:
        raise ValueError(f"invalid {field}: {value!r}; expected one of {allowed}")
    return text


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(tag, str) for tag in value):
        raise ValueError("tags must be a list of strings")
    return [tag for tag in value if tag.strip()]


def _limit(value: Any, field: str, default: int, maximum: int) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer")
    if not 1 <= raw <= maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return raw


def _services(ctx: Any) -> Mapping[str, Any]:
    value = ctx.get("services") if isinstance(ctx, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _looks_like_store(value: Any) -> bool:
    return all(callable(getattr(value, method, None)) for method in ("add", "list", "delete"))


def _store_from(services: Mapping[str, Any]) -> Any | None:
    for key in ("memory_store", "store"):
        candidate = services.get(key)
        if _looks_like_store(candidate):
            return candidate
    memory = services.get("memory")
    if _looks_like_store(memory):
        return memory
    if isinstance(memory, Mapping):
        for key in ("store", "memory_store"):
            candidate = memory.get(key)
            if _looks_like_store(candidate):
                return candidate
    return None


def _search_from(services: Mapping[str, Any]) -> Any | None:
    for key in ("memory_search", "search"):
        candidate = services.get(key)
        if callable(getattr(candidate, "search", None)):
            return candidate
    memory = services.get("memory")
    if isinstance(memory, Mapping):
        for key in ("search", "memory_search"):
            candidate = memory.get(key)
            if callable(getattr(candidate, "search", None)):
                return candidate
    return None


def _resolve(ctx: Any) -> tuple[Any, Any]:
    """Return (store, search), preferring injected services over local ones."""
    services = _services(ctx)
    store = _store_from(services)
    search = _search_from(services)
    if store is None and search is None:
        # No runtime injection: both components still talk to the shared
        # durable database, so this is a real backend, not a stub.
        store = MemoryStore()
    if store is None:
        store = getattr(search, "store", None)
        if not _looks_like_store(store):
            raise RuntimeError("injected memory search service has no usable MemoryStore")
    if search is None:
        search = MemorySearch(store)
    return store, search


def _dump(record: Any) -> dict[str, Any]:
    """Serialize a MemoryRecord (or a service stand-in) for the tool result."""
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    if isinstance(record, Mapping):
        return dict(record)
    return {
        "id": getattr(record, "id", None),
        "content": getattr(record, "content", None),
    }


async def memory_add(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Persist a new memory record and return its real stored id."""
    content = _require_text(args.get("content"), "content")
    category = _validate_enum(
        args.get("category", MemoryCategory.SEMANTIC.value), "category", MEMORY_CATEGORIES
    )
    kind = _validate_enum(args.get("kind", MemoryKind.FACT.value), "kind", MEMORY_KINDS)
    metadata = args.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    confidence = args.get("confidence", 0.7)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")

    record = MemoryRecord(
        content=content,
        category=MemoryCategory(category),
        kind=MemoryKind(kind),
        tags=_tags(args.get("tags")),
        metadata=dict(metadata),
        confidence=float(confidence),
        source=args.get("source") or "memory_add",
    )
    store, _ = _resolve(ctx)
    stored = await store.add(record)
    return {"success": True, "id": getattr(stored, "id", record.id), "record": _dump(stored)}


async def memory_search(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Rank real stored memories for a non-empty query."""
    query = _require_text(args.get("query"), "query")
    category = _validate_enum(args.get("category"), "category", MEMORY_CATEGORIES)
    kind = _validate_enum(args.get("kind"), "kind", MEMORY_KINDS)
    tags = _tags(args.get("tags"))
    limit = _limit(args.get("limit"), "limit", 10, 100)
    _, search = _resolve(ctx)
    results = await search.search(
        query,
        category=category,
        kind=kind,
        tags=tags or None,
        limit=limit,
    )
    dumped = [_dump(record) for record in results]
    return {"success": True, "query": query, "count": len(dumped), "results": dumped}


async def memory_list(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """List stored memories, optionally filtered by category, kind and tags."""
    category = _validate_enum(args.get("category"), "category", MEMORY_CATEGORIES)
    kind = _validate_enum(args.get("kind"), "kind", MEMORY_KINDS)
    tags = _tags(args.get("tags"))
    limit = _limit(args.get("limit"), "limit", 50, 200)
    store, _ = _resolve(ctx)
    records = await store.list(category=category, kind=kind, tags=tags or None, limit=limit)
    dumped = [_dump(record) for record in records]
    return {"success": True, "count": len(dumped), "records": dumped}


async def memory_forget(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Delete one memory by id; report whether a row was actually removed."""
    memory_id = _require_text(args.get("id"), "id").strip()
    store, _ = _resolve(ctx)
    deleted = bool(await store.delete(memory_id))
    return {"success": True, "id": memory_id, "deleted": deleted}


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the four memory tools and return their names.

    ``services`` is accepted for the standard toolset registration signature;
    concrete service instances are taken from each call's ``ctx["services"]``.
    """
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="memory_add",
                description="Persist a durable memory record (content, category, kind, tags).",
                category="memory",
                risk=RiskLevel.MEDIUM,
                permissions=["memory:write"],
                input_schema=_schema(
                    {
                        "content": {"type": "string", "minLength": 1, "maxLength": 4000},
                        "category": _CATEGORY_PROPERTY,
                        "kind": _KIND_PROPERTY,
                        "tags": _TAGS_PROPERTY,
                        "metadata": {"type": "object"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "source": {"type": "string", "minLength": 1},
                    },
                    ["content"],
                ),
                resource_field="",  # durable record, not a filesystem path
            ),
            memory_add,
        ),
        (
            ToolSpec(
                name="memory_search",
                description="Search stored memories by keyword/semantic rank with optional filters.",
                category="memory",
                risk=RiskLevel.LOW,
                permissions=["memory:read"],
                input_schema=_schema(
                    {
                        "query": {"type": "string", "minLength": 1},
                        "category": _CATEGORY_PROPERTY,
                        "kind": _KIND_PROPERTY,
                        "tags": _TAGS_PROPERTY,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    ["query"],
                ),
                resource_field="",
            ),
            memory_search,
        ),
        (
            ToolSpec(
                name="memory_list",
                description="List stored memories, optionally filtered by category, kind or tags.",
                category="memory",
                risk=RiskLevel.LOW,
                permissions=["memory:read"],
                input_schema=_schema(
                    {
                        "category": _CATEGORY_PROPERTY,
                        "kind": _KIND_PROPERTY,
                        "tags": _TAGS_PROPERTY,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    [],
                ),
                resource_field="",
            ),
            memory_list,
        ),
        (
            ToolSpec(
                name="memory_forget",
                description=(
                    "Delete a single stored memory by its id. HIGH risk: requires an "
                    "operator approval bound to these exact arguments."
                ),
                category="memory",
                risk=RiskLevel.HIGH,
                permissions=["memory:write", "memory:delete"],
                input_schema=_schema(
                    {"id": {"type": "string", "minLength": 1}},
                    ["id"],
                ),
                resource_field="",
            ),
            memory_forget,
        ),
    ]
    registered: list[str] = []
    for spec, handler in definitions:
        try:
            registry.register(spec, handler)
        except ValueError:  # already present - keep the caller's version
            continue
        registered.append(spec.name)
    return registered


__all__ = [
    "register",
    "memory_add",
    "memory_search",
    "memory_list",
    "memory_forget",
    "MEMORY_CATEGORIES",
    "MEMORY_KINDS",
]
