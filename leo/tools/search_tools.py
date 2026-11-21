"""Unified, source-labelled search over LEO's real durable state.

The three tools (``search_all``, ``search_recent_events``, ``find_related``)
query live components rather than maintaining a separate index:

* ``MemorySearch`` over ``memory_items`` for ranked memory recall;
* the live ``ToolRegistry``'s typed ``ToolSpec`` rows;
* ``SkillRegistry.list()`` manifests;
* ``DurableTaskEngine.list()`` task records;
* ``EventBus.history()`` / the durable ``events`` table.

Services are read from ``ctx["services"]`` first and then from the mapping
passed to :func:`register`. Constructing a component locally is only permitted
when the host has already configured a database, so a tool can never silently
operate on a different store. Every merged result carries an explicit ``source``
label and all result sets are capped by bounded limits.

``find_related`` deliberately uses only real ``memory_links`` rows and real
stored tags; it computes no synthetic similarity and never invents a link.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import db
from ..contracts import RiskLevel, ToolSpec
from ..memory.search import tokenize

TOOL_NAMES: tuple[str, str, str] = (
    "search_all",
    "search_recent_events",
    "find_related",
)
SOURCE_ORDER: tuple[str, str, str, str] = ("memory", "tool", "skill", "task")
MAX_RESULTS = 50
MAX_EVENTS = 100
MAX_RELATED = 50

_SEARCH_KEYS = ("memory_search", "search")
_STORE_KEYS = ("memory_store", "store")
_MEMORY_KEYS = ("memory",)
_TOOL_KEYS = ("tools", "tool_registry", "registry")
_SKILL_KEYS = ("skills", "skill_registry")
_TASK_KEYS = ("tasks", "task_engine", "task_registry")
_EVENT_KEYS = ("events", "event_bus")


def _schema(
    properties: dict[str, dict[str, Any]], required: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_QUERY = {"type": "string", "minLength": 1, "maxLength": 200}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS}
_EVENT_LIMIT = {"type": "integer", "minimum": 1, "maximum": MAX_EVENTS}
_ID = {"type": "string", "minLength": 1, "maxLength": 200}


def _services(ctx: Any) -> Mapping[str, Any]:
    value = ctx.get("services") if isinstance(ctx, Mapping) else None
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError("ctx['services'] must be a mapping of service names to services")
    return value


def _pick(ctx: Any, fallback: Any, keys: tuple[str, ...]) -> Any | None:
    """Prefer the per-call context over services captured at registration."""
    sources = (_services(ctx), fallback if isinstance(fallback, Mapping) else {})
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _db_ready() -> bool:
    return db.database_path_configured()


def _unavailable(label: str, keys: tuple[str, ...]) -> RuntimeError:
    return RuntimeError(
        f"{label} search requires an injected {keys[0]} service "
        f"(accepted keys: {', '.join(keys)}) or an initialized database"
    )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _bounded(value: Any, field: str, default: int, maximum: int) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer")
    if not 1 <= raw <= maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return raw


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump(mode="json"))
    if isinstance(record, Mapping):
        return dict(record)
    return {}


# ------------------------------------------------------------------- services


def _memory_candidates(
    ctx: Any, fallback: Any
) -> tuple[Any | None, Any | None]:
    search = _pick(ctx, fallback, _SEARCH_KEYS)
    store = _pick(ctx, fallback, _STORE_KEYS)
    memory = _pick(ctx, fallback, _MEMORY_KEYS)
    if isinstance(memory, Mapping):
        if search is None:
            search = memory.get("search") or memory.get("memory_search")
        if store is None:
            store = memory.get("store") or memory.get("memory_store")
    elif memory is not None:
        if search is None:
            search = getattr(memory, "search", None)
        if store is None:
            store = getattr(memory, "store", None)
    return search, store


def _memory_search_service(ctx: Any, fallback: Any) -> Any:
    search, store = _memory_candidates(ctx, fallback)
    if search is not None:
        if not callable(getattr(search, "search", None)):
            raise RuntimeError("injected memory search service has no search() method")
        return search
    if store is not None and callable(getattr(store, "list", None)):
        from ..memory.search import MemorySearch

        return MemorySearch(store)
    if not _db_ready():
        raise _unavailable("memory", _SEARCH_KEYS)
    from ..memory.search import MemorySearch
    from ..memory.store import MemoryStore

    return MemorySearch(MemoryStore())


def _memory_store_service(ctx: Any, fallback: Any) -> Any:
    search, store = _memory_candidates(ctx, fallback)
    if store is None and search is not None:
        store = getattr(search, "store", None)
    if store is not None and not (
        callable(getattr(store, "get", None))
        and callable(getattr(store, "get_links", None))
        and callable(getattr(store, "list", None))
    ):
        raise RuntimeError(
            "injected memory store service must provide get(), get_links() and list()"
        )
    if store is not None:
        return store
    if not _db_ready():
        raise _unavailable("memory store", _STORE_KEYS)
    from ..memory.store import MemoryStore

    return MemoryStore()


def _events_service(ctx: Any, fallback: Any) -> Any:
    service = _pick(ctx, fallback, _EVENT_KEYS)
    if service is not None:
        if not callable(getattr(service, "history", None)):
            raise RuntimeError("injected events service has no history() method")
        return service
    if not _db_ready():
        raise _unavailable("event", _EVENT_KEYS)
    from ..events import EventBus

    return EventBus(persist=True)


def _optional_events_service(ctx: Any, fallback: Any) -> Any | None:
    service = _pick(ctx, fallback, _EVENT_KEYS)
    return service if service is not None and callable(getattr(service, "emit", None)) else None


def _skills_service(ctx: Any, fallback: Any) -> Any:
    service = _pick(ctx, fallback, _SKILL_KEYS)
    if service is not None:
        if not callable(getattr(service, "list", None)):
            raise RuntimeError("injected skills service has no list() method")
        return service
    if not _db_ready():
        raise _unavailable("skills", _SKILL_KEYS)
    from ..skills.registry import SkillRegistry

    return SkillRegistry(events=_optional_events_service(ctx, fallback))


def _tasks_service(ctx: Any, fallback: Any) -> Any:
    service = _pick(ctx, fallback, _TASK_KEYS)
    if service is not None:
        if not callable(getattr(service, "list", None)):
            raise RuntimeError("injected tasks service has no list() method")
        return service
    if not _db_ready():
        raise _unavailable("tasks", _TASK_KEYS)
    from ..tasks.engine import DurableTaskEngine

    settings = ctx.get("settings") if isinstance(ctx, Mapping) else None
    events = ctx.get("events") if isinstance(ctx, Mapping) else None
    if settings is None or events is None:
        raise RuntimeError(
            "task service unavailable: settings/events missing from tool context"
        )
    return DurableTaskEngine(settings, events)


def _tools_service(ctx: Any, fallback: Any) -> Any:
    service = _pick(ctx, fallback, _TOOL_KEYS)
    if service is not None:
        if not callable(getattr(service, "list_specs", None)):
            raise RuntimeError("injected tools service has no list_specs() method")
        return service
    if not _db_ready():
        raise _unavailable("tools", _TOOL_KEYS)
    # No live registry was injected: build the real native registry so results
    # are still genuine ToolSpecs, never a fabricated list of names.
    from .builtins import register_builtins
    from .registry import ToolRegistry

    registry = ToolRegistry()
    register_builtins(registry)
    return registry


# --------------------------------------------------------------------- search


def _match_score(needle: str, fields: Any) -> int | None:
    values = [str(value or "").lower() for value in fields]
    if not any(needle in value for value in values):
        return None
    return 3 if needle in values[0] else 1


async def _memory_hits(query: str, limit: int, ctx: Any, fallback: Any) -> list[dict[str, Any]]:
    search = _memory_search_service(ctx, fallback)
    records = await search.search(query, limit=limit)
    wanted = set(tokenize(query))
    hits: list[dict[str, Any]] = []
    for record in records:
        content = str(getattr(record, "content", "") or "")
        tags = list(getattr(record, "tags", None) or [])
        # Keep only lexically grounded hits: the real MemorySearch ranks the
        # candidates, this guard prevents unrelated hash-embedding neighbours
        # from being reported as matches for a keyword query.
        if not wanted & set(tokenize(" ".join([content, *[str(tag) for tag in tags]]))):
            continue
        hits.append(
            {
                "source": "memory",
                "id": str(getattr(record, "id", "")),
                "title": content,
                "snippet": ", ".join(str(tag) for tag in tags),
                "score": round(float(getattr(record, "score", 0.0) or 0.0), 6),
            }
        )
    return hits


async def _tool_hits(query: str, limit: int, ctx: Any, fallback: Any) -> list[dict[str, Any]]:
    specs = _tools_service(ctx, fallback).list_specs()
    needle = query.lower()
    hits: list[dict[str, Any]] = []
    for spec in specs:
        data = _as_dict(spec)
        score = _match_score(
            needle,
            (
                data.get("name"),
                data.get("description"),
                data.get("category"),
                " ".join(str(perm) for perm in (data.get("permissions") or [])),
            ),
        )
        if score is None:
            continue
        hits.append(
            {
                "source": "tool",
                "id": str(data.get("name") or ""),
                "title": str(data.get("name") or ""),
                "snippet": str(data.get("description") or ""),
                "score": score,
            }
        )
    hits.sort(key=lambda row: (-row["score"], row["id"]))
    return hits[:limit]


async def _skill_hits(query: str, limit: int, ctx: Any, fallback: Any) -> list[dict[str, Any]]:
    records = await _skills_service(ctx, fallback).list(format="dicts")
    needle = query.lower()
    hits: list[dict[str, Any]] = []
    for raw in records:
        data = _as_dict(raw)
        score = _match_score(
            needle,
            (
                data.get("name"),
                data.get("description"),
                data.get("category"),
                data.get("documentation"),
            ),
        )
        if score is None:
            continue
        hits.append(
            {
                "source": "skill",
                "id": str(data.get("name") or ""),
                "title": str(data.get("name") or ""),
                "snippet": str(data.get("description") or ""),
                "score": score,
            }
        )
    hits.sort(key=lambda row: (-row["score"], row["id"]))
    return hits[:limit]


async def _task_hits(query: str, limit: int, ctx: Any, fallback: Any) -> list[dict[str, Any]]:
    tasks = await _tasks_service(ctx, fallback).list(limit=max(50, limit * 5))
    needle = query.lower()
    hits: list[dict[str, Any]] = []
    for task in tasks:
        data = _as_dict(task)
        status = getattr(getattr(task, "status", None), "value", data.get("status"))
        score = _match_score(
            needle,
            (
                data.get("id"),
                data.get("objective"),
                " ".join(str(step) for step in (data.get("plan") or [])),
                data.get("owner_agent"),
            ),
        )
        if score is None:
            continue
        hits.append(
            {
                "source": "task",
                "id": str(data.get("id") or ""),
                "title": str(data.get("objective") or ""),
                "snippet": f"status={status}",
                "score": score,
            }
        )
    hits.sort(key=lambda row: (-row["score"], row["id"]))
    return hits[:limit]


_HITS = {
    "memory": _memory_hits,
    "tool": _tool_hits,
    "skill": _skill_hits,
    "task": _task_hits,
}


async def search_all(
    args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None
) -> dict[str, Any]:
    """Merge bounded, explicitly labelled hits from four real backends."""
    query = _text(args.get("query"), "query")
    limit = _bounded(args.get("limit"), "limit", 10, MAX_RESULTS)
    requested = args.get("sources")
    if requested is None:
        requested = list(SOURCE_ORDER)
    if not isinstance(requested, (list, tuple)) or any(
        not isinstance(item, str) or item not in SOURCE_ORDER for item in requested
    ):
        raise ValueError(
            f"sources must be a list of {list(SOURCE_ORDER)}"
        )
    selected = [source for source in SOURCE_ORDER if source in requested]

    results: list[dict[str, Any]] = []
    by_source: dict[str, int] = {}
    for source in selected:
        if len(results) >= limit:
            by_source[source] = 0
            continue
        hits = await _HITS[source](query, limit, ctx, fallback)
        kept = hits[: limit - len(results)]
        by_source[source] = len(kept)
        results.extend(kept)
    return {
        "source": "unified",
        "query": query,
        "sources": selected,
        "limit": limit,
        "count": len(results),
        "by_source": by_source,
        "results": results,
    }


async def search_recent_events(
    args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None
) -> dict[str, Any]:
    """Return a bounded, newest-first page of durable event records."""
    limit = _bounded(args.get("limit"), "limit", 20, MAX_EVENTS)
    topic = _optional_text(args.get("topic"), "topic")
    query = _optional_text(args.get("query"), "query")
    correlation_id = _optional_text(args.get("correlation_id"), "correlation_id")

    bus = _events_service(ctx, fallback)
    # Read a bounded window, then filter: a limit applied before filtering
    # would hide newer matches behind older non-matches.
    events = await bus.history(topic=topic, limit=max(100, limit))
    needle = query.lower() if query else None
    rows: list[dict[str, Any]] = []
    for event in events:
        data = _as_dict(event)
        if correlation_id and data.get("correlation_id") != correlation_id:
            continue
        if needle:
            haystack = f"{data.get('topic') or ''} {data.get('payload') or ''}".lower()
            if needle not in haystack:
                continue
        rows.append(
            {
                "source": "event",
                "id": str(data.get("id") or ""),
                "topic": str(data.get("topic") or ""),
                "payload": data.get("payload") or {},
                "correlation_id": data.get("correlation_id"),
                "created_at": data.get("created_at"),
            }
        )
        if len(rows) >= limit:
            break
    return {
        "source": "events",
        "count": len(rows),
        "limit": limit,
        "filters": {"topic": topic, "query": query, "correlation_id": correlation_id},
        "events": rows,
    }


async def find_related(
    args: dict[str, Any], ctx: dict[str, Any], fallback: Any = None
) -> dict[str, Any]:
    """Find related memories using only real stored links and shared tags."""
    memory_id = _text(args.get("id"), "id")
    limit = _bounded(args.get("limit"), "limit", 10, MAX_RELATED)
    include_tags = args.get("include_tags", True)
    if not isinstance(include_tags, bool):
        raise ValueError("include_tags must be a boolean")

    store = _memory_store_service(ctx, fallback)
    record = await store.get(memory_id)
    if record is None:
        raise KeyError(f"unknown memory: {memory_id}")

    seen = {memory_id}
    linked: list[tuple[float, dict[str, Any]]] = []
    for link in await store.get_links(memory_id):
        source = str(link.get("source") or "")
        target = str(link.get("target") or "")
        peer_id = target if source == memory_id else source
        if not peer_id or peer_id in seen:
            continue
        peer = await store.get(peer_id)
        if peer is None:
            continue
        seen.add(peer_id)
        try:
            weight = float(link.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 0.0
        row = {
            "source": "link",
            "id": str(peer.id),
            "title": str(peer.content),
            "relation": str(link.get("relation") or "related"),
            "direction": "outgoing" if source == memory_id else "incoming",
            "weight": weight,
            "shared_tags": [],
        }
        linked.append((-weight, row))

    tagged: list[tuple[int, dict[str, Any]]] = []
    if include_tags:
        own_tags = {str(tag) for tag in (getattr(record, "tags", None) or [])}
        if own_tags:
            candidates = await store.list(
                tags=sorted(own_tags), limit=max(100, limit * 5)
            )
            for peer in candidates:
                peer_id = str(getattr(peer, "id", ""))
                if not peer_id or peer_id in seen:
                    continue
                shared = sorted(own_tags & {str(tag) for tag in (getattr(peer, "tags", None) or [])})
                if not shared:
                    continue
                seen.add(peer_id)
                tagged.append(
                    (
                        -len(shared),
                        {
                            "source": "tag",
                            "id": peer_id,
                            "title": str(peer.content),
                            "relation": "shared_tag",
                            "direction": "shared_tag",
                            "weight": 0.0,
                            "shared_tags": shared,
                        },
                    )
                )

    linked.sort(key=lambda pair: (pair[0], pair[1]["id"]))
    tagged.sort(key=lambda pair: (pair[0], pair[1]["id"]))
    related = [row for _, row in linked] + [row for _, row in tagged]
    return {
        "source": "memory_links",
        "id": memory_id,
        "limit": limit,
        "count": len(related),
        "returned": min(limit, len(related)),
        "related": related[:limit],
    }


# ---------------------------------------------------------------------- specs

_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="search_all",
            description=(
                "Search memory, tool specs, skill manifests and task records with one query; "
                "results are merged with explicit source labels and bounded limits."
            ),
            category="search",
            risk=RiskLevel.LOW,
            permissions=["search:read"],
            input_schema=_schema(
                {
                    "query": _QUERY,
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SOURCE_ORDER)},
                        "minItems": 1,
                        "maxItems": len(SOURCE_ORDER),
                    },
                    "limit": _LIMIT,
                },
                ("query",),
            ),
            resource_field="",  # logical records, not filesystem paths
        ),
        search_all,
    ),
    (
        ToolSpec(
            name="search_recent_events",
            description=(
                "Read recent durable events with optional topic, query and correlation filters; "
                "results are newest-first and bounded."
            ),
            category="search",
            risk=RiskLevel.LOW,
            permissions=["search:read", "events:read"],
            input_schema=_schema(
                {
                    "topic": {"type": "string", "minLength": 1},
                    "query": {"type": "string", "minLength": 1},
                    "correlation_id": {"type": "string", "minLength": 1},
                    "limit": _EVENT_LIMIT,
                }
            ),
            resource_field="",
        ),
        search_recent_events,
    ),
    (
        ToolSpec(
            name="find_related",
            description=(
                "Find memories connected by real stored memory_links or shared tags; "
                "no synthetic similarity is used."
            ),
            category="search",
            risk=RiskLevel.LOW,
            permissions=["search:read", "memory:read"],
            input_schema=_schema(
                {
                    "id": _ID,
                    "limit": _LIMIT,
                    "include_tags": {"type": "boolean"},
                },
                ("id",),
            ),
            resource_field="",
        ),
        find_related,
    ),
]


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the unified search toolset and return the newly registered names."""
    fallback = dict(services) if isinstance(services, Mapping) else None
    registered: list[str] = []
    for spec, implementation in _SPECS:

        async def handler(
            args: dict[str, Any], ctx: dict[str, Any], _impl=implementation
        ) -> Any:
            return await _impl(args, ctx, fallback)

        try:
            registry.register(spec, handler)
        except ValueError:  # preserve a caller's existing tool of the same name
            continue
        registered.append(spec.name)
    return registered


__all__ = [
    "register",
    "search_all",
    "search_recent_events",
    "find_related",
    "TOOL_NAMES",
    "SOURCE_ORDER",
]
