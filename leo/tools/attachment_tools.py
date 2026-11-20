"""Real attachment management tools backed by :class:`AttachmentStore`.

The four tools (``attachment_list``, ``attachment_info``, ``attachment_add``,
``attachment_remove``) operate on the same durable SQLite ``attachments`` table
and workspace files as the chat upload API. They prefer the ``AttachmentStore``
injected into ``ctx["services"]``; when none is injected they construct a real
store lazily over the already-configured database (never an in-memory stand-in).

Safety: every read of a stored file re-checks path containment against the
store's workspace root *and* opens the original path with ``O_NOFOLLOW`` and an
``fstat`` regular-file/size check. This closes the symlink-swap gap of the
``/attachments/{id}/content`` API endpoint, which only calls ``is_file()`` and
then follows whatever the path points at. Mime, encoding and size are validated
with the store's own rules (constants and ``AttachmentStore.save``); no success
is reported unless a row and a file were actually written/deleted.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .. import db
from ..contracts import RiskLevel, ToolSpec
from ..integrations.attachments import (
    ALLOWED_MIME,
    IMAGE_MIME,
    MAX_BYTES,
    TEXT_MIME,
    AttachmentError,
    AttachmentStore,
)
from .registry import ToolRegistry

_STORE_KEYS = ("attachments", "attachment_store", "attachments_store")
_PREVIEW_CHARS = 16_000


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _limit(value: Any, default: int = 50, maximum: int = 200) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("limit must be an integer")
    if not 1 <= raw <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return raw


def _looks_like_store(value: Any) -> bool:
    return (
        value is not None
        and callable(getattr(value, "save", None))
        and callable(getattr(value, "get", None))
        and callable(getattr(value, "public", None))
        and isinstance(getattr(value, "root", None), (str, Path))
    )


def _store(ctx: Mapping[str, Any], registered_services: Any = None) -> Any:
    """Resolve a real AttachmentStore: injected service first, lazy import next."""
    for source in (ctx.get("services"), registered_services):
        if isinstance(source, Mapping):
            for key in _STORE_KEYS:
                candidate = source.get(key)
                if _looks_like_store(candidate):
                    return candidate
    if not db.database_path_configured():
        raise RuntimeError(
            "attachment service unavailable: no AttachmentStore injected and no database initialized"
        )
    settings = ctx.get("settings")
    if settings is None:
        raise RuntimeError("attachment service unavailable: settings missing from tool context")
    return AttachmentStore(settings)


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


@contextmanager
def _verified_open(path_value: Any, roots: list[Path]) -> Iterator[tuple[Any, Path]]:
    """Yield a handle to a regular file whose resolved path is inside ``roots``.

    The final path component is opened with ``O_NOFOLLOW`` (so a symlink swap of
    the recorded file is refused even if it points back inside the workspace),
    and the regular-file/size checks are made on the very descriptor that is
    returned, avoiding a stat/open time-of-check/time-of-use gap.
    """
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        raise ValueError("attachment record has no file path")
    text = str(path_value)
    if "\x00" in text:
        raise PermissionError("path contains NUL byte")
    original = Path(text)
    try:
        resolved = original.expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise PermissionError("cannot resolve attachment path") from exc
    if not any(_within(Path(root).resolve(), resolved) for root in roots):
        raise PermissionError(f"attachment path outside allowed workspace: {resolved}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(original, flags)
    except OSError as exc:
        raise PermissionError("cannot open attachment file safely") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PermissionError("attachment path is not a regular file")
        if info.st_size > MAX_BYTES:
            raise ValueError("attachment file exceeds the configured size limit")
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "rb") as handle:
        yield handle, resolved


def _validate_encoded(data: Any) -> str:
    """Apply the store's own base64/size rules before delegating to ``save``."""
    if not isinstance(data, str) or not data:
        raise ValueError("attachment data must be base64 encoded")
    if len(data) > int(MAX_BYTES * 4 / 3) + 16:
        raise ValueError("encoded attachment is too large")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 attachment data") from exc
    if not raw:
        raise ValueError("attachment is empty")
    if len(raw) > MAX_BYTES:
        raise ValueError(f"attachment exceeds {MAX_BYTES // (1024 * 1024)} MB limit")
    return data


def _read_local(path_value: Any, ctx: Mapping[str, Any]) -> bytes:
    """Read a caller-supplied local file only from an allowed root."""
    settings = ctx.get("settings")
    if settings is None:
        raise RuntimeError("attachment_add with a path requires settings in the tool context")
    roots = [
        Path(settings.root).resolve(),
        Path(settings.resolve(settings.paths.workspaces)).resolve(),
    ]
    roots.extend(Path(settings.resolve(p)).resolve() for p in settings.security.allowed_roots)
    roots.extend(Path(p).resolve() for p in (ctx.get("allowed_roots") or []))
    with _verified_open(path_value, roots) as (handle, _):
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("attachment file exceeds the configured size limit")
    return raw


async def attachment_list(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    """List stored attachment metadata, verifying each backing file at read time."""
    store = _store(ctx, services)
    limit = _limit(args.get("limit"))
    conversation_id = args.get("conversation_id")
    if conversation_id is not None:
        conversation_id = _require_text(conversation_id, "conversation_id")
        rows = await db.fetchall(
            "SELECT * FROM attachments WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
            (conversation_id, limit),
        )
    else:
        rows = await db.fetchall(
            "SELECT * FROM attachments ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    items: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        mime = str(record.get("mime", "")).lower()
        record["kind"] = "image" if mime in IMAGE_MIME else "file"
        view = store.public(record)
        try:
            with _verified_open(record.get("path"), [store.root]):
                pass
            view["available"] = True
        except (OSError, ValueError, PermissionError, AttachmentError):
            # The row is real but its file is missing or unsafe: report that
            # honestly rather than pretending the attachment is usable.
            view["available"] = False
        items.append(view)
    return {
        "attachments": items,
        "count": len(items),
        "limit": limit,
        "conversation_id": conversation_id,
    }


async def attachment_info(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    """Return verified metadata (and a bounded text preview) for one attachment."""
    store = _store(ctx, services)
    attachment_id = _require_text(args.get("id"), "id")
    record = await store.get(attachment_id)
    if not record or not isinstance(record, Mapping) or not record.get("id"):
        raise KeyError(f"unknown attachment: {attachment_id}")
    record = dict(record)
    with _verified_open(record.get("path"), [store.root]) as (handle, _):
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("attachment file exceeds the configured size limit")
    if hashlib.sha256(raw).hexdigest() != str(record.get("sha256", "")).lower():
        raise AttachmentError("attachment content does not match the stored digest")
    view = store.public(record)
    view["verified"] = True
    if str(record.get("mime", "")).lower() in TEXT_MIME:
        view["text_preview"] = raw.decode("utf-8", errors="replace")[:_PREVIEW_CHARS]
    return view


async def attachment_add(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    """Persist a validated attachment from inline base64 or a local file path."""
    store = _store(ctx, services)
    name = _require_text(args.get("name"), "name")
    mime = str(args.get("mime") or "").split(";")[0].strip().lower()
    if mime not in ALLOWED_MIME:
        raise ValueError(f"unsupported attachment type: {mime}")
    data = args.get("data_base64")
    path = args.get("path")
    if data is not None and path is not None:
        raise ValueError("provide either data_base64 or path, not both")
    if path is not None:
        data = base64.b64encode(_read_local(path, ctx)).decode("ascii")
    data = _validate_encoded(data)  # store's own mime/size/encoding rules
    try:
        record = await store.save(name, mime, data, args.get("conversation_id"))
    except AttachmentError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(record, Mapping) or not record.get("id"):
        raise RuntimeError("attachment store did not persist the file")
    return store.public(dict(record))


async def attachment_remove(args: dict[str, Any], ctx: dict[str, Any], services: Any = None) -> Any:
    """Delete one attachment's file and row; report only what was actually removed."""
    store = _store(ctx, services)
    attachment_id = _require_text(args.get("id"), "id")
    record = await store.get(attachment_id)
    if not record or not isinstance(record, Mapping) or not record.get("id"):
        return {"id": attachment_id, "deleted": False, "found": False}
    # Re-check containment/regular-file status before touching the file.
    with _verified_open(record.get("path"), [store.root]):
        pass
    file_removed = False
    try:
        Path(str(record["path"])).unlink()
        file_removed = True
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AttachmentError("cannot delete attachment file") from exc
    cursor = await db.execute("DELETE FROM attachments WHERE id=?", (attachment_id,))
    if not getattr(cursor, "rowcount", 0):
        raise AttachmentError("attachment row was not deleted")
    return {"id": attachment_id, "deleted": True, "found": True, "file_removed": file_removed}


def register(registry: ToolRegistry, services: Any = None) -> list[str]:
    """Register the four typed attachment tools and return the names added."""

    def wrap(handler: Any) -> Any:
        async def registered(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
            return await handler(args, ctx, services)

        return registered

    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="attachment_list",
                description="List stored attachment metadata with per-file availability.",
                category="attachments",
                risk=RiskLevel.LOW,
                permissions=["attachment:read"],
                input_schema=_schema(
                    {
                        "conversation_id": {"type": "string", "minLength": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    [],
                ),
                resource_field="",  # metadata read, not a caller-supplied path
            ),
            attachment_list,
        ),
        (
            ToolSpec(
                name="attachment_info",
                description="Verify and describe one stored attachment by id.",
                category="attachments",
                risk=RiskLevel.LOW,
                permissions=["attachment:read"],
                input_schema=_schema({"id": {"type": "string", "minLength": 1}}, ["id"]),
                resource_field="",
            ),
            attachment_info,
        ),
        (
            ToolSpec(
                name="attachment_add",
                description="Store a validated attachment from base64 data or a local file path.",
                category="attachments",
                risk=RiskLevel.MEDIUM,
                permissions=["attachment:write"],
                input_schema=_schema(
                    {
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "mime": {"type": "string", "minLength": 1, "maxLength": 100},
                        "data_base64": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                        "conversation_id": {"type": "string", "minLength": 1},
                    },
                    ["name", "mime"],
                ),
                resource_field="path",
            ),
            attachment_add,
        ),
        (
            ToolSpec(
                name="attachment_remove",
                description="Delete a stored attachment's file and metadata row by id.",
                category="attachments",
                risk=RiskLevel.MEDIUM,
                permissions=["attachment:write", "attachment:delete"],
                input_schema=_schema({"id": {"type": "string", "minLength": 1}}, ["id"]),
                resource_field="",
            ),
            attachment_remove,
        ),
    ]
    registered: list[str] = []
    for spec, handler in definitions:
        try:
            registry.register(spec, wrap(handler))
        except ValueError:  # keep a caller's existing tool of the same name
            continue
        registered.append(spec.name)
    return registered


__all__ = [
    "register",
    "attachment_list",
    "attachment_info",
    "attachment_add",
    "attachment_remove",
]
