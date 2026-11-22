"""Two edit tools: write a file, and replace an exact snippet inside one.

A coding agent without these can only mutate the tree through a shell command,
so every edit costs a heredoc, cannot be reviewed as a diff, and dies with the
command when a timeout fires. ``write_file`` and ``patch`` keep edits structured:
the intent is in the arguments, the result is a hash and a diff, and the caller
can see exactly what changed without re-reading the file.

Both tools resolve paths through the same root enforcement as the read tools and
write atomically (temp file in the destination directory, then rename), so a
crash mid-write can never leave a truncated source file.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from ..contracts import RiskLevel, ToolSpec
from .file_tools import (
    MAX_FILE_BYTES,
    _prepare_destination,
    _safe_path,
    _schema,
)
from .registry import ToolRegistry

# A single edit may introduce a file larger than a read cap: a generated module
# is legitimately bigger than the default read window. Still bounded, because an
# unbounded write is an unbounded memory allocation from model output.
MAX_WRITE_BYTES = 2_000_000
MAX_DIFF_LINES = 400


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(destination: Path, data: bytes) -> None:
    """Write through a sibling temp file, then rename over the target.

    ``os.replace`` is atomic on the same filesystem, so a reader either sees the
    old content or the new one - never a half-written file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".leo-tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


async def write_file(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Create or replace a text file inside an allowed root."""
    path = _safe_path(args.get("path"), ctx)
    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    encoding = args.get("encoding") or "utf-8"
    try:
        data = content.encode(encoding)
    except (LookupError, UnicodeEncodeError) as exc:
        raise ValueError(f"content cannot be encoded as {encoding}: {exc}") from None
    if len(data) > MAX_WRITE_BYTES:
        raise ValueError(
            f"content is {len(data)} bytes; the write ceiling is {MAX_WRITE_BYTES}"
        )

    overwrite = bool(args.get("overwrite", True))
    existed = path.is_file()
    # The overwrite guard runs before any mutation, so a refusal cannot leave a
    # half-applied write behind.
    _prepare_destination(path, overwrite)
    previous = path.read_bytes() if existed else None
    if ctx.get("dry_run"):
        return {
            "path": str(path),
            "bytes": len(data),
            "sha256": _digest(data),
            "created": not existed,
            "overwrite": overwrite,
            "dry_run": True,
        }

    _atomic_write(path, data)
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": _digest(data),
        "created": not existed,
        "overwrite": overwrite,
        "previous_bytes": len(previous) if previous is not None else 0,
        "previous_sha256": _digest(previous) if previous is not None else "",
    }


def _replace_once(text: str, old: str, new: str, replace_all: bool, path: Path) -> str:
    matches = text.count(old)
    if matches == 0:
        raise ValueError(
            f"old_string does not occur in {path}; read the file and copy the "
            "text exactly, including indentation"
        )
    if matches > 1 and not replace_all:
        raise ValueError(
            f"old_string occurs {matches} times in {path}; include surrounding "
            "lines to make it unique, or pass replace_all=true"
        )
    return text.replace(old, new, -1 if replace_all else 1)


def _unified(before: str, after: str, path: Path) -> str:
    diff = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            n=2,
        )
    )
    if len(diff) > MAX_DIFF_LINES:
        head = "".join(diff[:MAX_DIFF_LINES])
        return head + f"\n... diff truncated at {MAX_DIFF_LINES} lines\n"
    return "".join(diff)


async def patch(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Replace one exact snippet in a file and report the diff."""
    path = _safe_path(args.get("path"), ctx)
    old_string = args.get("old_string")
    new_string = args.get("new_string")
    if not isinstance(old_string, str) or not old_string:
        raise ValueError("old_string must be a non-empty string")
    if not isinstance(new_string, str):
        raise ValueError("new_string must be a string (use an empty string to delete)")
    if old_string == new_string:
        raise ValueError("old_string and new_string are identical; nothing to change")
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"{path} is {size} bytes; the patch ceiling is {MAX_FILE_BYTES}"
        )
    encoding = args.get("encoding") or "utf-8"
    before = path.read_text(encoding=encoding, errors="strict")
    replace_all = bool(args.get("replace_all", False))
    after = _replace_once(before, old_string, new_string, replace_all, path)
    data = after.encode(encoding)
    if len(data) > MAX_WRITE_BYTES:
        raise ValueError(
            f"patched content is {len(data)} bytes; the write ceiling is {MAX_WRITE_BYTES}"
        )
    replacements = before.count(old_string) if replace_all else 1
    diff = _unified(before, after, path)
    if ctx.get("dry_run"):
        return {
            "path": str(path),
            "replacements": replacements,
            "diff": diff,
            "dry_run": True,
        }

    _atomic_write(path, data)
    return {
        "path": str(path),
        "replacements": replacements,
        "diff": diff,
        "bytes": len(data),
        "sha256": _digest(data),
    }


def register(registry: ToolRegistry, services: Any = None) -> list[str]:
    """Register the edit tools; return the names actually registered.

    Calling twice is safe: an already-registered tool keeps its existing handler.
    """
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="write_file",
                description=(
                    "Create or replace a text file inside an allowed root. Writes are "
                    "atomic and return the sha256 of the new content; existing content "
                    "is replaced, so read the file first when extending it."
                ),
                category="filesystem",
                risk=RiskLevel.HIGH,
                permissions=["file:write"],
                supports_dry_run=True,
                consume_approval=True,
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string"},
                        "encoding": {"type": "string"},
                        "overwrite": {"type": "boolean"},
                    },
                    ["path", "content"],
                ),
                resource_field="path",
            ),
            write_file,
        ),
        (
            ToolSpec(
                name="patch",
                description=(
                    "Replace one exact snippet in an existing file and return a unified "
                    "diff. old_string must match the file text exactly; when it occurs "
                    "more than once, add surrounding lines or pass replace_all=true."
                ),
                category="filesystem",
                risk=RiskLevel.HIGH,
                permissions=["file:read", "file:write"],
                supports_dry_run=True,
                consume_approval=True,
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "old_string": {"type": "string", "minLength": 1},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                        "encoding": {"type": "string"},
                    },
                    ["path", "old_string", "new_string"],
                ),
                resource_field="path",
            ),
            patch,
        ),
    ]
    names: list[str] = []
    for spec, handler in definitions:
        try:
            registry.register(spec, handler)
        except ValueError:  # already present - keep the caller's version
            continue
        names.append(spec.name)
    return names
