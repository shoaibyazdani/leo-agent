"""Workspace-confined file utility tools: copy, move, hash, diff and tree.

Every path operand is resolved and checked against the same allowed roots used
by :mod:`leo.tools.builtins` (settings.root, the configured workspaces directory,
``security.allowed_roots`` and an optional ctx-provided ``allowed_roots``).
Resolution follows symlinks, so symlinks that point outside those roots are
denied. All paths are checked *before* any write, and resolved paths are used
for the actual filesystem operations: a destination can never be redirected
outside the workspace by a symlink or an overwrite.

No injected service is required: the durable state these tools act on is the
filesystem itself. ``services`` is accepted for the common ``register`` contract
and ignored.
"""

from __future__ import annotations

import difflib
import hashlib
import shutil
from pathlib import Path
from typing import Any

from ..contracts import RiskLevel, ToolSpec
from .registry import ToolRegistry

MAX_FILE_BYTES = 1_048_576
HASH_CHUNK_BYTES = 1_048_576
HASH_ALGORITHMS = ("sha256", "sha512", "sha1", "md5", "blake2b")
MAX_TREE_DEPTH = 10
MAX_TREE_ENTRIES = 5_000
MAX_DIFF_LINES = 2_000


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _allowed_roots(ctx: dict[str, Any]) -> list[Path]:
    settings = ctx.get("settings")
    if settings is None:
        raise RuntimeError("file tools require settings in the tool context")
    roots = [
        Path(settings.root).resolve(),
        Path(settings.resolve(settings.paths.workspaces)).resolve(),
    ]
    roots.extend(Path(settings.resolve(p)).resolve() for p in settings.security.allowed_roots)
    extra = ctx.get("allowed_roots")
    if extra:
        roots.extend(Path(p).resolve() for p in extra)
    return roots


def _safe_path(value: Any, ctx: dict[str, Any]) -> Path:
    """Resolve a path operand and confine it to an allowed root."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("path must be a non-empty string")
    text = str(value)
    if "\x00" in text:
        raise PermissionError("path contains NUL byte")
    try:
        candidate = Path(text).expanduser()
        workdir = ctx.get("workdir")
        if workdir and not candidate.is_absolute():
            candidate = Path(str(workdir)).expanduser() / candidate
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise PermissionError(f"cannot resolve path safely: {text}") from exc
    for root in _allowed_roots(ctx):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise PermissionError(f"path outside allowed roots: {resolved}")


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _prepare_destination(destination: Path, overwrite: bool) -> None:
    """Reject unsafe overwrite targets before any filesystem mutation."""
    if not destination.exists():
        return
    if destination.is_dir():
        raise IsADirectoryError(f"destination is a directory: {destination}")
    if not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    # A hard link inside the workspace can name an inode outside it; writing
    # through it would modify a file we never validated.
    if destination.stat().st_nlink > 1:
        raise PermissionError("refusing to overwrite a hard-linked file")


async def file_copy(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    source = _safe_path(args.get("source"), ctx)
    destination = _safe_path(args.get("destination"), ctx)
    if not source.is_file():
        raise FileNotFoundError(f"not a file: {source}")
    if source == destination:
        raise ValueError("source and destination must differ")
    overwrite = bool(args.get("overwrite", False))
    _prepare_destination(destination, overwrite)
    if ctx.get("dry_run"):
        return {
            "source": str(source), "destination": str(destination),
            "copied": False, "dry_run": True,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "overwrite": overwrite,
    }


async def file_move(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    source = _safe_path(args.get("source"), ctx)
    destination = _safe_path(args.get("destination"), ctx)
    if not source.is_file():
        raise FileNotFoundError(f"not a file: {source}")
    if source == destination:
        raise ValueError("source and destination must differ")
    overwrite = bool(args.get("overwrite", False))
    _prepare_destination(destination, overwrite)
    if ctx.get("dry_run"):
        return {
            "source": str(source), "destination": str(destination),
            "moved": False, "dry_run": True,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and overwrite:
        destination.unlink()
    shutil.move(str(source), str(destination))
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "overwrite": overwrite,
    }


def _read_capped(path: Path, cap: int) -> tuple[str, bool]:
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
    return raw[:cap].decode("utf-8", errors="replace"), len(raw) > cap


async def file_hash(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    path = _safe_path(args.get("path"), ctx)
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    algorithm = str(args.get("algorithm", "sha256")).lower()
    if algorithm not in HASH_ALGORITHMS:
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    digest = hashlib.new(algorithm)
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": str(path),
        "algorithm": algorithm,
        "hash": digest.hexdigest(),
        "size": size,
    }


async def file_diff(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    left = _safe_path(args.get("left"), ctx)
    right = _safe_path(args.get("right"), ctx)
    for path in (left, right):
        if not path.is_file():
            raise FileNotFoundError(f"not a file: {path}")
    cap = min(max(int(args.get("max_bytes", MAX_FILE_BYTES)), 1), MAX_FILE_BYTES)
    context = min(max(int(args.get("context", 3)), 0), 20)
    max_lines = min(max(int(args.get("max_lines", MAX_DIFF_LINES)), 1), MAX_DIFF_LINES)
    left_text, left_cut = _read_capped(left, cap)
    right_text, right_cut = _read_capped(right, cap)
    lines = difflib.unified_diff(
        left_text.splitlines(), right_text.splitlines(),
        fromfile=str(left), tofile=str(right), n=context, lineterm="",
    )
    emitted: list[str] = []
    truncated = left_cut or right_cut
    for line in lines:
        if len(emitted) >= max_lines:
            truncated = True
            break
        emitted.append(line)
    return {
        "left": str(left),
        "right": str(right),
        "changed": left_text != right_text,
        "unified": "\n".join(emitted),
        "truncated": truncated,
        "left_truncated": left_cut,
        "right_truncated": right_cut,
    }


async def dir_tree(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    root = _safe_path(args.get("path"), ctx)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    max_depth = min(max(int(args.get("max_depth", 3)), 1), MAX_TREE_DEPTH)
    max_entries = min(max(int(args.get("max_entries", 500)), 1), MAX_TREE_ENTRIES)
    include_files = bool(args.get("include_files", True))
    entries: list[dict[str, Any]] = []
    truncated = False

    def walk(current: Path, depth: int) -> None:
        nonlocal truncated
        if depth > max_depth or truncated:
            return
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            return
        for child in children:
            # Symlinks are never followed: a link could point at a directory
            # outside the allowed roots and leak its contents into the tree.
            if child.is_symlink():
                continue
            try:
                if child.is_dir():
                    kind = "dir"
                elif include_files and child.is_file():
                    kind = "file"
                else:
                    continue
            except OSError:
                continue
            if not _is_within(root, child):
                continue
            if len(entries) >= max_entries:
                truncated = True
                return
            entries.append({
                "path": str(child.relative_to(root)),
                "type": kind,
                "depth": depth,
            })
            if kind == "dir":
                walk(child, depth + 1)

    walk(root, 1)
    text = "\n".join(
        "  " * (entry["depth"] - 1)
        + Path(entry["path"]).name
        + ("/" if entry["type"] == "dir" else "")
        for entry in entries
    )
    return {
        "path": str(root),
        "entries": entries,
        "count": len(entries),
        "truncated": truncated,
        "text": text,
    }


def register(registry: ToolRegistry, services: Any = None) -> list[str]:
    """Register all file utility tools; return the names actually registered.

    Calling twice is safe: an already-registered tool keeps its existing
    handler. ``services`` is accepted for the shared toolset contract but none
    of these tools need an injected service.
    """
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="file_copy",
                description="Copy a regular file to a new path inside an allowed root.",
                category="filesystem",
                risk=RiskLevel.MEDIUM,
                permissions=["file:read", "file:write"],
                input_schema=_schema(
                    {
                        "source": {"type": "string", "minLength": 1},
                        "destination": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                    },
                    ["source", "destination"],
                ),
                resource_field="source",
            ),
            file_copy,
        ),
        (
            ToolSpec(
                name="file_move",
                description="Move a regular file to a new path inside an allowed root.",
                category="filesystem",
                risk=RiskLevel.MEDIUM,
                permissions=["file:read", "file:write"],
                input_schema=_schema(
                    {
                        "source": {"type": "string", "minLength": 1},
                        "destination": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                    },
                    ["source", "destination"],
                ),
                resource_field="source",
            ),
            file_move,
        ),
        (
            ToolSpec(
                name="file_hash",
                description="Compute a cryptographic digest of a file inside an allowed root.",
                category="filesystem",
                risk=RiskLevel.LOW,
                permissions=["file:read"],
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "algorithm": {
                            "type": "string",
                            "enum": list(HASH_ALGORITHMS),
                        },
                    },
                    ["path"],
                ),
                resource_field="path",
            ),
            file_hash,
        ),
        (
            ToolSpec(
                name="file_diff",
                description="Return a unified text diff of two files inside allowed roots.",
                category="filesystem",
                risk=RiskLevel.LOW,
                permissions=["file:read"],
                input_schema=_schema(
                    {
                        "left": {"type": "string", "minLength": 1},
                        "right": {"type": "string", "minLength": 1},
                        "context": {"type": "integer", "minimum": 0, "maximum": 20},
                        "max_bytes": {"type": "integer", "minimum": 1},
                        "max_lines": {"type": "integer", "minimum": 1},
                    },
                    ["left", "right"],
                ),
                resource_field="left",
            ),
            file_diff,
        ),
        (
            ToolSpec(
                name="dir_tree",
                description="List a bounded directory tree without following symlinks.",
                category="filesystem",
                risk=RiskLevel.LOW,
                permissions=["file:read"],
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "max_depth": {"type": "integer", "minimum": 1, "maximum": MAX_TREE_DEPTH},
                        "max_entries": {"type": "integer", "minimum": 1, "maximum": MAX_TREE_ENTRIES},
                        "include_files": {"type": "boolean"},
                    },
                    ["path"],
                ),
                resource_field="path",
            ),
            dir_tree,
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
    "register", "file_copy", "file_move", "file_hash", "file_diff", "dir_tree",
]
