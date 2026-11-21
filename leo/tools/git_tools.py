"""Strictly read-only git inspection tools (status, diff, log, show).

Every command is built from a fixed argv template: callers supply only typed
values that are validated, confined to the workspace root and inserted at a
known position after a ``--`` separator. There is no free-form command string,
no shell, and no path through which a state-changing git subcommand or flag can
reach the subprocess. Git is invoked with optional locks disabled, a scrubbed
environment and a hard timeout so inspection cannot mutate the repository or
hang the tool loop.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ..contracts import RiskLevel, ToolSpec
from .registry import ToolRegistry

GIT_EXECUTABLE = "git"
TIMEOUT_SECONDS = 10.0
MAX_OUTPUT_BYTES = 65_536
MAX_LINES = 2_000
MAX_ENTRIES = 2_000
MAX_PATH_CHARS = 4_096
MAX_REVISION_CHARS = 200

# Names that must never be accepted as a revision token. The argv template
# already makes every mutating subcommand unreachable; this is defence in depth
# for callers that try to smuggle one in as a value.
FORBIDDEN_SUBCOMMANDS = frozenset({
    "add", "am", "apply", "archive", "bisect", "branch", "checkout", "cherry-pick",
    "clean", "clone", "commit", "config", "fetch", "gc", "init", "merge", "mv",
    "prune", "pull", "push", "rebase", "remote", "reset", "restore", "revert",
    "rm", "stash", "status--porcelain", "submodule", "switch", "tag", "worktree",
})

_SCRUB_ENV_KEYS = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
    "GIT_PREFIX", "GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS",
    "GIT_EXTERNAL_DIFF", "GIT_PAGER", "GIT_OPTIONAL_LOCKS", "GIT_ATTR_SOURCE",
    "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS", "GIT_CEILING_DIRECTORIES",
})
_SHELL_META = frozenset(";|&$`\\\n\r\t")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~^@{}\/-]*$")


def _git_env(root: Path) -> dict[str, str]:
    """Build a deterministic environment that cannot redirect the repository."""
    env = dict(os.environ)
    for key in tuple(env):
        if key in _SCRUB_ENV_KEYS or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            env.pop(key, None)
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CEILING_DIRECTORIES": str(root.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "LC_ALL": "C",
    })
    return env


def _run_git(root: Path, args: list[str]) -> dict[str, Any]:
    """Run one allowlisted git command and return bounded, decoded output."""
    argv = [
        GIT_EXECUTABLE, "--no-pager", "--no-optional-locks",
        "-c", "core.fsmonitor=false",
        "-c", "core.hooksPath=" + os.devnull,
        "-c", "color.ui=false",
        "-C", str(root),
        *args,
    ]
    process = subprocess.Popen(
        argv,
        cwd=str(root),
        env=_git_env(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError(f"git {args[0]} timed out after {TIMEOUT_SECONDS}s")
    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()[:500]
        raise RuntimeError(f"git {args[0]} failed (exit {process.returncode}): {detail}")
    byte_truncated = len(stdout) > MAX_OUTPUT_BYTES
    text = stdout[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")
    lines = text.splitlines()
    line_truncated = len(lines) > MAX_LINES
    if line_truncated:
        lines = lines[:MAX_LINES]
    return {
        "text": text,
        "lines": lines,
        "truncated": byte_truncated or line_truncated,
        "exit_code": int(process.returncode),
    }


def _workspace_root(ctx: dict[str, Any]) -> Path:
    settings = ctx.get("settings")
    if settings is None:
        raise RuntimeError("git tools require settings in the tool context")
    return Path(settings.root).resolve()


def _safe_token(value: Any, label: str, *, limit: int, revision: bool = False) -> str:
    """Reject flags, shell syntax, pathspec magic and mutating command names."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if len(value) > limit:
        raise ValueError(f"{label} is longer than {limit} characters")
    if any(char in value for char in _SHELL_META):
        raise PermissionError(f"{label} contains control or shell metacharacters")
    if value.startswith(("-", ":")):
        raise PermissionError(f"{label} must not start with '-' or ':'")
    if revision:
        if value.lower() in FORBIDDEN_SUBCOMMANDS:
            raise PermissionError(f"{label} names a state-changing git command: {value}")
        if ".." in value:
            raise PermissionError(f"{label} must not be a revision range")
        if _REVISION_RE.search(value) is None:
            raise PermissionError(f"{label} is not a plain git revision: {value}")
    return value


def _safe_path(value: Any, root: Path) -> str | None:
    """Resolve a path operand and confine it to the workspace root."""
    if value is None:
        return None
    text = _safe_token(value, "path", limit=MAX_PATH_CHARS)
    if any(char in text for char in "*?["):
        raise PermissionError("glob pathspecs are not allowed")
    candidate = Path(text)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path outside workspace root: {resolved}") from exc
    result = relative.as_posix()
    return None if result == "." else result


def _safe_revision(value: Any) -> str:
    return _safe_token(value, "revision", limit=MAX_REVISION_CHARS, revision=True)


async def git_status(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    root = _workspace_root(ctx)
    path = _safe_path(args.get("path"), root)
    command = ["status", "--porcelain=v1"]
    if args.get("branch"):
        command.append("--branch")
    untracked = args.get("untracked_files")
    if untracked is not None:
        if untracked not in ("normal", "all", "no"):
            raise ValueError(f"unsupported untracked_files mode: {untracked}")
        command.append(f"--untracked-files={untracked}")
    if path:
        command.extend(["--", path])
    output = _run_git(root, command)
    entries = [
        {"status": line[:2], "path": line[3:]}
        for line in output["lines"]
        if len(line) >= 4 and not line.startswith("##")
    ][:MAX_ENTRIES]
    branch = next((line[3:] for line in output["lines"] if line.startswith("##")), None)
    return {
        "tool": "git_status",
        "root": str(root),
        "path": path or ".",
        "branch": branch,
        "entries": entries,
        "count": len(entries),
        "text": output["text"],
        "truncated": output["truncated"],
    }


async def git_diff(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    root = _workspace_root(ctx)
    path = _safe_path(args.get("path"), root)
    context = int(args.get("context", 3))
    if not 0 <= context <= 20:
        raise ValueError("context must be between 0 and 20")
    command = ["diff", "--no-color", "--no-ext-diff", "--no-textconv", f"--unified={context}"]
    if args.get("staged"):
        command.append("--cached")
    if args.get("stat"):
        command.append("--stat")
    if path:
        command.extend(["--", path])
    output = _run_git(root, command)
    return {
        "tool": "git_diff",
        "root": str(root),
        "path": path or ".",
        "staged": bool(args.get("staged", False)),
        "diff": output["text"],
        "changed": bool(output["text"].strip()),
        "truncated": output["truncated"],
    }


async def git_log(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    root = _workspace_root(ctx)
    path = _safe_path(args.get("path"), root)
    count = int(args.get("max_count", 10))
    if not 1 <= count <= 100:
        raise ValueError("max_count must be between 1 and 100")
    command = ["log", "--no-color", f"--max-count={count}"]
    command.append("--oneline" if args.get("oneline", True) else "--format=fuller")
    if path:
        command.extend(["--", path])
    output = _run_git(root, command)
    return {
        "tool": "git_log",
        "root": str(root),
        "path": path or ".",
        "lines": output["lines"],
        "count": len(output["lines"]),
        "text": output["text"],
        "truncated": output["truncated"],
    }


async def git_show(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    root = _workspace_root(ctx)
    revision = _safe_revision(args.get("revision"))
    path = _safe_path(args.get("path"), root)
    command = ["show", "--no-color", "--no-ext-diff", "--no-textconv"]
    if args.get("stat"):
        command.append("--stat")
    command.append(revision)
    if path:
        command.extend(["--", path])
    output = _run_git(root, command)
    return {
        "tool": "git_show",
        "root": str(root),
        "revision": revision,
        "path": path or ".",
        "text": output["text"],
        "truncated": output["truncated"],
    }


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_PATH_PROPERTY = {
    "type": "string",
    "minLength": 1,
    "maxLength": MAX_PATH_CHARS,
    "pattern": "^[^-:]",
}


def register(registry: ToolRegistry, services: Any = None) -> list[str]:
    """Register the four read-only git tools and return the names registered.

    ``services`` is accepted for the shared toolset contract; the durable state
    inspected here is the workspace git repository itself. Re-registering is a
    no-op for names already present, so callers keep their own handler version.
    """
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="git_status",
                description="Read the read-only porcelain status of the workspace git repository.",
                category="development",
                risk=RiskLevel.LOW,
                permissions=["git:read"],
                resource_field="path",
                timeout_seconds=TIMEOUT_SECONDS,
                input_schema=_schema({
                    "path": dict(_PATH_PROPERTY),
                    "branch": {"type": "boolean"},
                    "untracked_files": {"type": "string", "enum": ["normal", "all", "no"]},
                }, []),
            ),
            git_status,
        ),
        (
            ToolSpec(
                name="git_diff",
                description="Read an unstaged or staged diff of the workspace git repository.",
                category="development",
                risk=RiskLevel.LOW,
                permissions=["git:read"],
                resource_field="path",
                timeout_seconds=TIMEOUT_SECONDS,
                input_schema=_schema({
                    "path": dict(_PATH_PROPERTY),
                    "staged": {"type": "boolean"},
                    "stat": {"type": "boolean"},
                    "context": {"type": "integer", "minimum": 0, "maximum": 20},
                }, []),
            ),
            git_diff,
        ),
        (
            ToolSpec(
                name="git_log",
                description="Read a bounded commit log for the workspace git repository.",
                category="development",
                risk=RiskLevel.LOW,
                permissions=["git:read"],
                resource_field="path",
                timeout_seconds=TIMEOUT_SECONDS,
                input_schema=_schema({
                    "path": dict(_PATH_PROPERTY),
                    "max_count": {"type": "integer", "minimum": 1, "maximum": 100},
                    "oneline": {"type": "boolean"},
                }, []),
            ),
            git_log,
        ),
        (
            ToolSpec(
                name="git_show",
                description="Read one commit, tree or blob from the workspace git repository.",
                category="development",
                risk=RiskLevel.LOW,
                permissions=["git:read"],
                resource_field="path",
                timeout_seconds=TIMEOUT_SECONDS,
                input_schema=_schema({
                    "revision": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_REVISION_CHARS,
                        "pattern": _REVISION_RE.pattern,
                    },
                    "path": dict(_PATH_PROPERTY),
                    "stat": {"type": "boolean"},
                }, ["revision"]),
            ),
            git_show,
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
    "register", "git_status", "git_diff", "git_log", "git_show",
    "FORBIDDEN_SUBCOMMANDS", "TIMEOUT_SECONDS", "MAX_OUTPUT_BYTES",
]
