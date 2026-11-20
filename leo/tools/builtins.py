"""Native LEO tools: filesystem, terminal, HTTP and small utilities.

All filesystem paths are resolved and confined to the settings' allowed roots.
Commands never go through a shell; network reads never follow redirects and
never target non-public IPs.

Known limit (F9, public reverse-proxy pivot): the egress check validates the
single address this process connects to. A legitimately public reverse proxy
can still forward the request to a private origin (or to a different host based
on the Host header). Preventing that pivot requires an operator-controlled
allowlist of proxies/origins; it is not visible from client-side DNS or IP
checks, so these tools must never be described as an internal-network boundary.

HTTP bodies are read from ``aiter_raw`` under a cumulative *wire* byte cap and,
when Content-Encoding is gzip/deflate, decompressed incrementally under the
same cap. A highly compressible response therefore cannot be expanded to many
times ``max_bytes`` before the cap applies. Unsupported encodings are rejected.
Cache-busting request headers and response freshness evidence (Date/Age/
Cache-Control/ETag) are surfaced, but a successful GET only proves that a
response was fetched, not that the payload is real-time (F6).
"""

from __future__ import annotations

import ast
import asyncio
import ipaddress
import json
import math
import operator
import re
import shlex
import shutil
import socket
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ..contracts import RiskLevel, ToolSpec
from .registry import ToolRegistry

MAX_FILE_BYTES = 1_048_576
MAX_OUTPUT_BYTES = 65_536
MAX_HTTP_BYTES = 1_048_576
DEFAULT_TERMINAL_TIMEOUT = 10.0
MAX_TERMINAL_TIMEOUT = 30.0
# Pagination is bounded so a guessed `limit=10**9` cannot turn into a memory
# blow-up. Offsets are line counters for read_file and entry counters elsewhere.
MAX_LINE_OFFSET = 1_000_000
MAX_PAGE_LIMIT = 1_000
MAX_SEARCH_LIMIT = 1_000
# rg is allowlisted, but some hosts do not ship it. A missing rg must degrade
# to a bounded, fixed-string grep rather than burning a round on "not found".
RG_FALLBACK_MAX_MATCHES = 50
_RG_FALLBACK_NOTICE = (
    "rg is not installed here; ran a bounded `grep -F` instead. Fixed-string matching "
    "makes the pattern literal (not a regex); matches are capped per file and output is "
    "capped by max_output_bytes. "
    "Prefer search_files/read_file for text and task_list for task data; do not inspect "
    "binary .db files with terminal."
)
# Only rg flags with a safe, equivalent grep spelling are translated. Everything
# else is rejected with a concrete alternative instead of being silently dropped.
_RG_FALLBACK_FLAGS: dict[str, str | None] = {
    "-F": "-F", "--fixed-strings": "-F",
    "-i": "-i", "--ignore-case": "-i",
    "-v": "-v", "--invert-match": "-v",
    "-l": "-l", "--files-with-matches": "-l",
    "-c": "-c", "--count": "-c",
    "-o": "-o", "--only-matching": "-o",
    "-w": "-w", "--word-regexp": "-w",
    "-x": "-x", "--line-regexp": "-x",
    "-n": "-n", "--line-number": "-n",
    "-H": "-H", "--with-filename": "-H",
    "-h": "-h", "--no-filename": "-h",
    "-q": "-q", "--quiet": "-q",
    "-a": "-a", "--text": "-a",
    "-m": "-m", "--max-count": "-m",
    "-A": "-A", "--after-context": "-A",
    "-B": "-B", "--before-context": "-B",
    "-C": "-C", "--context": "-C",
    "-e": "-e", "--regexp": "-e",
    # rg's "no line numbers" flag is honored by omitting the injected -n.
    "-I": None, "--no-line-number": None,
}
_RG_VALUE_FLAGS = frozenset(
    {"-m", "--max-count", "-A", "--after-context", "-B", "--before-context",
     "-C", "--context", "-e", "--regexp"}
)

# Commands that are non-destructive in their common invocation. Anything else is
# denied before a process is started.
ALLOWED_EXECUTABLES = frozenset(
    {
        "cat", "cp", "date", "echo", "find", "grep", "head", "ls", "mkdir",
        "mv", "pwd", "sed", "sort", "tail", "touch",
        "uname", "wc", "whoami", "git", "rg",
    }
)
# Interpreters are never allowlisted by default: `python -c`, `node -e` etc.
# would be arbitrary code execution with the backend process's privileges.
# An operator may explicitly add one via security.allowed_executables.

# Per-executable argument denylists for known code-execution/mutating flags.
_DANGEROUS_ARGS: dict[str, frozenset[str]] = {
    "find": frozenset({"-exec", "-execdir", "-delete", "-ok", "-fprint", "-fprintf", "-fls"}),
    "sed": frozenset({"-i", "--in-place"}),
    "grep": frozenset({"--exec", "--files-with-matches-once"}),
    "rg": frozenset({"--exec", "--passthru-exec"}),
}
# Git subcommands that can execute hooks, move data or change remote state.
_GIT_READ_ONLY = frozenset(
    {"status", "log", "diff", "show", "branch", "ls-files", "rev-parse", "blame", "tag", "help"}
)
# These options can set config (fsmonitor/hooks/aliases/pager/exec-path) or
# redirect git to another repository, turning a "read-only" subcommand into
# arbitrary code execution. The short forms accept an attached value, so they
# are matched by prefix as well as by exact spelling.
_GIT_UNSAFE_OPTIONS = frozenset(
    {
        "--config", "--config-env", "--exec-path", "--git-dir",
        "--work-tree", "--namespace", "--super-prefix",
    }
)

# Patchable seam for tests; no real DNS lookup is performed when the host is
# already a numeric IP address.
resolve_host = socket.getaddrinfo

_SHELL_METACHARACTERS = set(";|&$`<>\n\r")


def terminal_description() -> str:
    """Build the terminal ToolSpec description from the actual policy constants.

    The descriptor is part of the model-visible contract: guessing interpreters
    or shell syntax used to burn whole rounds on denials. Keeping it derived
    from ALLOWED_EXECUTABLES prevents the docs drifting from enforcement.
    """
    allowed = ", ".join(sorted(ALLOWED_EXECUTABLES))
    forbidden = " ".join(
        repr(char) for char in sorted(_SHELL_METACHARACTERS) if char not in ("\n", "\r")
    )
    return (
        "Run ONE allowlisted executable directly, with no shell. "
        f"Allowed executables: {allowed}. "
        "Interpreters (python, python3, node, sh, bash) and every other unlisted "
        "executable are denied; an operator may extend the list via "
        "security.allowed_executables. "
        f"Forbidden shell metacharacters: {forbidden}, newline, carriage return. "
        "No pipes, redirects, command chaining or subshells: pass separate arguments "
        "in `command` instead. For file inspection prefer read_file, list_files or "
        "search_files; use git only with read-only subcommands. If rg is unavailable, "
        "a bounded fixed-string grep fallback is used and reported in `notice`."
    )


def _schema(
    properties: dict[str, dict[str, Any]], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _bounded_int(
    value: Any, name: str, *, minimum: int, maximum: int, default: int | None
) -> int | None:
    """Handlers may be called directly, so re-check bounds enforced by the schema."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _page_start(cursor: Any, entries: list[str]) -> int:
    """Resolve a pagination cursor to an absolute entry index.

    Accepted forms: the integer/``offset:N`` token emitted as ``next_cursor``,
    or a previously returned entry name (resume after that entry). Anything
    else is rejected with a message the model can act on.
    """
    if cursor is None or cursor == "":
        return 0
    if isinstance(cursor, bool):
        raise ValueError("cursor must be a next_cursor token or an entry name")
    if isinstance(cursor, int) and cursor >= 0:
        return cursor
    if not isinstance(cursor, str):
        raise ValueError("cursor must be a next_cursor token or an entry name")
    token = cursor.strip()
    if token.isdigit():
        return int(token)
    if token.startswith("offset:"):
        suffix = token[len("offset:"):]
        if suffix.isdigit():
            return int(suffix)
    if token in entries:
        return entries.index(token) + 1
    raise ValueError(
        f"invalid cursor {token!r}: pass next_cursor from a previous call, "
        "an integer offset, or an entry name from the previous page"
    )


def _allowed_roots(ctx: dict[str, Any]) -> list[Path]:
    settings = ctx["settings"]
    roots = [
        settings.root.resolve(),
        settings.resolve(settings.paths.workspaces).resolve(),
    ]
    roots.extend(settings.resolve(p).resolve() for p in settings.security.allowed_roots)
    extra = ctx.get("allowed_roots")
    if extra:
        roots.extend(Path(p).resolve() for p in extra)
    return roots


def _safe_path(value: Any, ctx: dict[str, Any]) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("path must be a string")
    if "\x00" in str(value):
        raise PermissionError("path contains NUL byte")
    candidate = Path(value).expanduser()
    resolved = candidate.resolve()
    settings = ctx["settings"]
    if getattr(settings.security, "workspace_only", True):
        for root in _allowed_roots(ctx):
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise PermissionError(f"path outside allowed roots: {resolved}")
    return resolved


def _truncate(data: bytes, cap: int) -> tuple[str, bool]:
    if len(data) <= cap:
        return data.decode("utf-8", errors="replace"), False
    return data[:cap].decode("utf-8", errors="replace") + "\n[truncated]", True


def _is_within(root: Path, candidate: Path) -> bool:
    """True when candidate resolves inside an already-resolved root.

    Glob results are not run through ``_safe_path``; resolving here closes the
    escape where ``../*`` patterns or symlinks point outside the allowed root.
    """
    try:
        candidate.resolve().relative_to(root)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


async def read_file(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    path = _safe_path(args.get("path"), ctx)
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    encoding = args.get("encoding", "utf-8")
    size = path.stat().st_size
    max_bytes = max(0, int(args.get("max_bytes", MAX_FILE_BYTES)))
    offset = int(_bounded_int(args.get("offset"), "offset", minimum=0,
                              maximum=MAX_LINE_OFFSET, default=0) or 0)
    limit = _bounded_int(args.get("limit"), "limit", minimum=0,
                         maximum=MAX_PAGE_LIMIT, default=None)

    if offset == 0 and limit is None:
        # Stream only the requested prefix; read_bytes() would load the whole
        # file into memory and defeat max_bytes for very large files. Bounding
        # the read by the file size also avoids overflowing read() with a huge cap.
        with path.open("rb") as handle:
            raw = handle.read(min(max_bytes, size))
        content = raw.decode(encoding, errors="replace")
        return {
            "path": str(path), "content": content, "truncated": size > max_bytes,
            "offset": 0, "limit": None, "lines_returned": content.count("\n"),
            "has_more": size > max_bytes, "next_offset": None,
        }

    # Paginated read: offset/limit are LINE counters, not byte counters. Each
    # readline() is itself capped so one pathological long line cannot allocate
    # more than max_bytes at a time. The byte budget stays authoritative.
    collected: list[bytes] = []
    used = 0
    lines_returned = 0
    has_more = False
    with path.open("rb") as handle:
        for _ in range(offset):
            if not handle.readline():
                break
        while True:
            if limit is not None and lines_returned >= limit:
                has_more = bool(handle.readline(1))
                break
            budget = max_bytes - used
            line = handle.readline(budget + 1 if budget > 0 else 1)
            if not line:
                break
            if len(line) <= budget:
                collected.append(line)
                used += len(line)
                lines_returned += 1
                continue
            if budget > 0:
                collected.append(line[:budget])
                lines_returned += 1
            has_more = True
            break
    content = b"".join(collected).decode(encoding, errors="replace")
    return {
        "path": str(path), "content": content, "truncated": has_more,
        "offset": offset, "limit": limit, "lines_returned": lines_returned,
        "has_more": has_more,
        "next_offset": offset + lines_returned if has_more else None,
    }


async def write_file(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    path = _safe_path(args.get("path"), ctx)
    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    encoding = args.get("encoding", "utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode(encoding)
    path.write_bytes(payload)
    return {"path": str(path), "bytes_written": len(payload)}


async def list_files(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    path = _safe_path(args.get("path"), ctx)
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    pattern = args.get("pattern", "*")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    if Path(pattern).is_absolute():
        raise ValueError("pattern must be relative")
    if ".." in Path(pattern).parts:
        raise ValueError("pattern must not contain '..'")
    recursive = bool(args.get("recursive", False))
    if recursive and pattern == "*":
        pattern = "**/*"
    entries = sorted(
        str(entry.resolve().relative_to(path))
        for entry in path.glob(pattern)
        if _is_within(path, entry)
        and entry.resolve() != path
        and (entry.is_file() or entry.is_dir())
    )
    offset = int(_bounded_int(args.get("offset"), "offset", minimum=0,
                              maximum=MAX_LINE_OFFSET, default=0) or 0)
    limit = _bounded_int(args.get("limit"), "limit", minimum=0,
                         maximum=MAX_PAGE_LIMIT, default=None)
    start = min(_page_start(args.get("cursor"), entries) + offset, len(entries))
    page = entries[start:] if limit is None else entries[start:start + int(limit or 0)]
    has_more = start + len(page) < len(entries)
    return {
        "path": str(path), "files": page, "count": len(page),
        "total": len(entries), "offset": start, "limit": limit,
        "has_more": has_more,
        "next_cursor": str(start + len(page)) if has_more else None,
        "truncated": has_more,
    }


async def search_files(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    requested = _safe_path(args.get("path"), ctx)
    # A file path is a one-entry search rooted at its parent: the caller no
    # longer has to guess that a file must be converted into a directory plus
    # pattern. The original (already confined) path is echoed back unchanged.
    if requested.is_file():
        root = requested.parent
        entries: list[Path] = [requested] if _is_within(root, requested) else []
    elif requested.is_dir():
        root = requested
        entries = sorted(root.rglob("*"))
    else:
        raise FileNotFoundError(f"not a file or directory: {requested}")
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    use_regex = bool(args.get("regex", False))
    try:
        matcher = re.compile(pattern) if use_regex else None
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    limit = int(_bounded_int(args.get("limit"), "limit", minimum=0,
                             maximum=MAX_SEARCH_LIMIT, default=100) or 0)
    offset = int(_bounded_int(args.get("offset"), "offset", minimum=0,
                              maximum=MAX_LINE_OFFSET, default=0) or 0)
    matches: list[dict[str, Any]] = []
    seen = 0
    if limit < 1:
        return {"path": str(requested), "pattern": pattern, "matches": matches,
                "truncated": False, "offset": offset, "total": 0,
                "has_more": False, "next_cursor": None}
    for entry in entries:
        if (
            not _is_within(root, entry)
            or not entry.is_file()
            or entry.stat().st_size > MAX_FILE_BYTES
        ):
            continue
        try:
            lines = entry.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            found = matcher.search(line) if matcher else pattern in line
            if found:
                seen += 1
                if seen <= offset:
                    continue
                matches.append(
                    {
                        "path": str(entry),
                        "line": line_no,
                        "text": line[:1000],
                    }
                )
                if len(matches) >= limit:
                    return {
                        "path": str(requested),
                        "pattern": pattern,
                        "matches": matches,
                        "truncated": True,
                        "offset": offset, "total": seen, "has_more": True,
                        "next_cursor": str(seen),
                    }
    return {"path": str(requested), "pattern": pattern, "matches": matches,
            "truncated": False, "offset": offset, "total": seen,
            "has_more": False, "next_cursor": None}


class _CommandParts(list):
    """Validated argv plus an optional notice about a policy-safe substitution."""

    def __init__(self, values: list[str], *, fallback_notice: str | None = None) -> None:
        super().__init__(values)
        self.fallback_notice = fallback_notice


def _rg_fallback_parts(parts: list[str]) -> list[str]:
    """Translate a safe subset of `rg` argv into bounded `grep -F` argv.

    Only flags with an equivalent grep spelling are accepted; anything else is
    rejected with a concrete alternative rather than silently changing meaning.
    The injected ``-m`` cap bounds work per file, and ``-n``/``-H`` preserve rg's
    ``path:line:match`` shape for text files.
    """
    args = parts[1:]
    out: list[str] = []
    injected: set[str] = set()
    has_pattern = False
    want_numbers = True
    want_filename = True
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            tail = args[index + 1:]
            out.append("--")
            out.extend(tail)
            has_pattern = has_pattern or bool(tail)
            break
        if not arg.startswith("-"):
            out.append(arg)
            has_pattern = True
            index += 1
            continue

        flag, sep, value = arg.partition("=")
        mapped = _RG_FALLBACK_FLAGS.get(flag, ...)
        if mapped is ...:
            raise ValueError(
                f"rg is not installed; flag {flag!r} has no safe grep -F fallback. "
                "Use grep-compatible flags (-i, -n, -l, -c, -m, -A/-B/-C, -e) or "
                "search_files/read_file instead."
            )
        if mapped is None:  # rg's --no-line-number: omit the injected -n
            want_numbers = False
            index += 1
            continue
        if mapped == "-n":
            want_numbers = False
        elif mapped in ("-H", "-h"):
            want_filename = False
        if mapped in ("-F", "-m"):
            injected.add(mapped)
        if flag in _RG_VALUE_FLAGS:
            if not sep:
                index += 1
                if index >= len(args):
                    raise ValueError(f"rg fallback: {flag} requires a value")
                value = args[index]
            if not value:
                raise ValueError(f"rg fallback: {flag} requires a non-empty value")
            if flag in ("-m", "--max-count"):
                try:
                    count = int(value)
                except ValueError as exc:
                    raise ValueError(
                        f"rg fallback: {flag} must be a positive integer <= "
                        f"{RG_FALLBACK_MAX_MATCHES}"
                    ) from exc
                if count < 1:
                    raise ValueError(f"rg fallback: {flag} must be a positive integer")
                out.extend(["-m", str(min(count, RG_FALLBACK_MAX_MATCHES))])
            else:
                out.extend([mapped, value])
                if mapped == "-e":
                    has_pattern = True
            index += 1
            continue
        out.append(mapped)
        index += 1

    if not has_pattern:
        raise ValueError(
            "rg is not installed; the grep -F fallback needs a search pattern "
            "(use -e for patterns beginning with '-')"
        )
    options: list[str] = []
    if "-F" not in injected:
        options.append("-F")
    if "-m" not in injected:
        options.extend(["-m", str(RG_FALLBACK_MAX_MATCHES)])
    if want_numbers:
        options.append("-n")
    if want_filename:
        options.append("-H")
    return ["grep", *options, *out]


def _command_parts(command: Any, ctx: dict[str, Any]) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if any(char in command for char in _SHELL_METACHARACTERS):
        raise PermissionError("shell metacharacters are not allowed")
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"invalid command: {exc}") from exc
    if not parts:
        raise ValueError("command must be a non-empty string")
    executable = parts[0]
    name = Path(executable).name
    allowed = set(ALLOWED_EXECUTABLES)
    extra = getattr(ctx["settings"].security, "allowed_executables", None)
    if extra:
        allowed.update(str(item) for item in extra)
    if name not in allowed:
        raise PermissionError(f"executable not allowed: {name}")
    if "/" in executable:
        _safe_path(executable, ctx)
    fallback_notice: str | None = None
    if name == "rg" and shutil.which("rg") is None:
        if shutil.which("grep") is None:
            raise FileNotFoundError(
                "rg is not installed and grep is unavailable; use search_files/read_file instead"
            )
        parts = _rg_fallback_parts(parts)
        name = "grep"
        fallback_notice = _RG_FALLBACK_NOTICE
    if shutil.which(name) is None:
        raise FileNotFoundError(f"executable not found: {name}")
    for arg in parts[1:]:
        denied = _DANGEROUS_ARGS.get(name, frozenset())
        if any(arg == bad or arg.startswith(bad) for bad in denied):
            raise PermissionError(f"dangerous argument for {name}: {arg}")
        if name == "git" and not arg.startswith("-") and arg not in _GIT_READ_ONLY:
            raise PermissionError(f"git subcommand not allowed: {arg}")
        if name == "git" and arg.startswith("-"):
            head = arg.split("=", 1)[0]
            if (
                head in _GIT_UNSAFE_OPTIONS
                or head.startswith("-c")  # -c / -c<name>=<value>
                or head.startswith("-C")  # -C / -C<path>
            ):
                raise PermissionError(f"unsafe git option: {arg}")
        # Path operands can be hidden behind flags (`--file=/etc/passwd`,
        # `--git-dir=/other/.git`), so validate the value, not just the flag.
        candidate = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
        if candidate.startswith(("/", "~", "..")) or ("/" in candidate and not candidate.startswith("-")):
            _safe_path(candidate, ctx)
    parts[0] = name  # resolve from PATH, never from an arbitrary absolute location
    return _CommandParts(parts, fallback_notice=fallback_notice)


async def terminal(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    parts = _command_parts(args.get("command"), ctx)
    notice = getattr(parts, "fallback_notice", None)
    cwd = _safe_path(args.get("cwd", "."), ctx)
    timeout = min(float(args.get("timeout", DEFAULT_TERMINAL_TIMEOUT)), MAX_TERMINAL_TIMEOUT)
    cap = min(int(args.get("max_output_bytes", MAX_OUTPUT_BYTES)), MAX_OUTPUT_BYTES)
    if ctx.get("dry_run"):
        # All allowlist/path/dangerous-flag checks in _command_parts have
        # already run, so a denied command is reported truthfully here.
        preview = {
            "command": args["command"], "dry_run": True, "exit_code": None,
            "stdout": "", "stderr": "",
        }
        if notice:
            preview.update(notice=notice, command_effective=" ".join(parts))
        return preview
    process = await asyncio.create_subprocess_exec(
        *parts,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    truncated = False
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        out, out_cut = _truncate(stdout, cap)
        err, err_cut = _truncate(stderr, cap)
        truncated = out_cut or err_cut
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"command timed out after {timeout}s") from exc
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
    result = {
        "command": args["command"],
        "exit_code": process.returncode,
        "stdout": out,
        "stderr": err,
        "truncated": truncated,
    }
    if notice:
        result["notice"] = notice
        result["command_effective"] = " ".join(parts)
    return result


def is_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Shared egress policy; also used by the browser operator."""
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or not address.is_global
    )


async def _assert_public_host(host: str) -> list[str]:
    """Resolve once and return the validated public address literals.

    The returned list is the only set of addresses a caller may connect to;
    connecting to a separately-resolved address would re-open the
    DNS-rebinding window this check exists to close. Raises PermissionError
    if any resolved address is non-public, ValueError if unresolvable.
    """
    host = host.strip("[]")
    if host.lower() in {"localhost", "localhost.localdomain"}:
        raise PermissionError(f"network target not allowed: {host}")
    try:
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [
            ipaddress.ip_address(host)
        ]
    except ValueError:
        infos = await asyncio.to_thread(resolve_host, host, None)
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except (ValueError, IndexError, TypeError):
                continue
        if not addresses:
            raise ValueError(f"could not resolve host: {host}")
    for address in addresses:
        if not is_public_ip(address):
            raise PermissionError(
                f"network target not allowed: {host} resolves to non-public address {address}"
            )
    return [str(address) for address in addresses]


# A GET asks for a fresh representation; it does not guarantee one. The
# response headers below let callers see when a cache/CDN answered instead.
_CACHE_BUST_HEADERS = {"Cache-Control": "no-cache"}
# (output key, response header) - output uses underscore keys for JSON stability.
_FRESHNESS_HEADERS = (
    ("date", "date"),
    ("age", "age"),
    ("cache_control", "cache-control"),
    ("etag", "etag"),
)


def _decompressor_for(encoding: str) -> Any | None:
    """Build a bounded incremental decompressor, or None for identity bodies."""
    normalized = encoding.strip().lower()
    if normalized in ("", "identity"):
        return None
    if normalized in ("gzip", "x-gzip"):
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if normalized == "deflate":
        return zlib.decompressobj()
    # brotli/zstd and unknown stackings are not implemented here; rejecting
    # them is safer than letting a third-party decoder allocate unboundedly.
    raise ValueError(f"unsupported Content-Encoding: {encoding!r}")


def _feed_decompressor(decompressor: Any, chunk: bytes, decoded: bytearray, cap: int) -> bool:
    """Append at most ``cap`` decoded bytes; return True if input/output remains."""
    pending = chunk
    while pending:
        room = cap - len(decoded)
        if room <= 0:
            return True
        try:
            out = decompressor.decompress(pending, room)
        except zlib.error as exc:
            raise ValueError(f"invalid compressed HTTP body: {exc}") from exc
        decoded.extend(out)
        tail = decompressor.unconsumed_tail
        if not tail:
            return False
        if len(decoded) >= cap or (not out and len(tail) >= len(pending)):
            return True  # no forward progress: fail closed instead of spinning
        pending = tail  # zlib buffered compressed input in its unconsumed tail
    return False


async def http_get(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    url = args.get("url")
    if not isinstance(url, str):
        raise ValueError("url must be a string")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http and https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("url must contain a hostname")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are not allowed")
    timeout = min(float(args.get("timeout", 10.0)), 30.0)
    cap = min(int(args.get("max_bytes", MAX_HTTP_BYTES)), MAX_HTTP_BYTES)
    validated = await _assert_public_host(parsed.hostname)

    request_url = url
    headers: dict[str, str] = dict(_CACHE_BUST_HEADERS)
    extensions: dict[str, Any] | None = None
    if parsed.hostname not in validated:
        # Pin the connection to a pre-validated address so the policy check
        # and the connection cannot observe two different DNS answers (the
        # DNS-rebinding window). The real host is preserved for the Host
        # header and for TLS SNI/certificate verification via sni_hostname.
        pinned = validated[0]
        pin_host = f"[{pinned}]" if ":" in pinned else pinned
        port = f":{parsed.port}" if parsed.port else ""
        request_url = parsed._replace(netloc=f"{pin_host}{port}").geturl()
        headers["Host"] = parsed.netloc
        extensions = {"sni_hostname": parsed.hostname}

    async with httpx.AsyncClient(
        follow_redirects=False, timeout=timeout, trust_env=False
    ) as client:
        async with client.stream(
            "GET", request_url, headers=headers, extensions=extensions
        ) as response:
            fetched_at = datetime.now(timezone.utc).isoformat()
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")
            freshness = {
                key: str(response.headers[header])[:256]
                if header in response.headers else None
                for key, header in _FRESHNESS_HEADERS
            }
            decompressor = _decompressor_for(
                response.headers.get("content-encoding", "")
            )
            decoded = bytearray()
            raw_total = 0
            truncated = False
            # aiter_bytes() would decompress first: a small gzip body expanding
            # to megabytes would blow past the cap before we could see it. Read
            # the *wire* stream and decompress no more than the cap permits.
            async for raw in response.aiter_raw():
                room = cap - raw_total
                if room <= 0:
                    truncated = True
                    break
                if len(raw) > room:
                    raw = raw[:room]
                    truncated = True
                raw_total += len(raw)
                if decompressor is None:
                    take = raw[: cap - len(decoded)]
                    decoded.extend(take)
                    if len(take) < len(raw):
                        truncated = True
                        break
                elif _feed_decompressor(decompressor, raw, decoded, cap):
                    truncated = True
                    break
            if decompressor is not None and not truncated and len(decoded) < cap:
                try:
                    tail = decompressor.flush(cap - len(decoded))
                except zlib.error as exc:
                    raise ValueError(f"invalid compressed HTTP body: {exc}") from exc
                decoded.extend(tail[: cap - len(decoded)])
    return {
        "url": url,
        "status_code": status_code,
        "content": bytes(decoded).decode("utf-8", errors="replace"),
        "content_type": content_type,
        "truncated": truncated,
        "fetched_at": fetched_at,
        "freshness": freshness,
    }


async def json_parse(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    text = args.get("text", args.get("json"))
    if not isinstance(text, str):
        raise ValueError("text must be a JSON string")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON constant not allowed: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite number not allowed in JSON")
        return parsed

    try:
        return json.loads(text, parse_constant=reject_constant, parse_float=finite_float)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ValueError(f"name not allowed: {node.id}")
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(result, complex) or (
            isinstance(result, float) and not math.isfinite(result)
        ):
            raise ValueError("calculation result is not a finite number")
        return result
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _FUNCTIONS or node.keywords:
            raise ValueError(f"function call not allowed: {node.func.id}")
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ValueError(f"unsupported expression component: {type(node).__name__}")


async def calculator(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    expression = args.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expression) > 1000:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from exc
    try:
        return _evaluate(tree)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"calculation failed: {exc}") from exc


async def current_time(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    zone = args.get("timezone", "utc")
    if zone in ("utc", "UTC", None):
        now = datetime.now(timezone.utc)
    elif zone == "local":
        now = datetime.now().astimezone()
    else:
        raise ValueError("timezone must be 'utc' or 'local'")
    return now.isoformat()


def register_builtins(registry: ToolRegistry) -> list[str]:
    """Register (or refresh) all native tools. Safe to call more than once."""
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="read_file",
                description=(
                    "Read a UTF-8 file inside an allowed root. Use offset (lines already "
                    "read) and limit (lines to return) to page through long files; "
                    "has_more/next_offset report the next page. max_bytes caps bytes."
                ),
                category="filesystem",
                risk=RiskLevel.LOW,
                permissions=["file:read"],
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "encoding": {"type": "string"},
                        "max_bytes": {"type": "integer", "minimum": 1},
                        "offset": {"type": "integer", "minimum": 0,
                                   "maximum": MAX_LINE_OFFSET},
                        "limit": {"type": "integer", "minimum": 0,
                                  "maximum": MAX_PAGE_LIMIT},
                    },
                    ["path"],
                ),
                resource_field="path",
            ),
            read_file,
        ),
        (
            ToolSpec(
                name="write_file",
                description="Create or overwrite a file inside an allowed root.",
                category="filesystem",
                risk=RiskLevel.MEDIUM,
                permissions=["file:write"],
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string"},
                        "encoding": {"type": "string"},
                    },
                    ["path", "content"],
                ),
                resource_field="path",
            ),
            write_file,
        ),
        (
            ToolSpec(
                name="list_files",
                description=(
                    "List files and directories inside an allowed root. Use limit and "
                    "offset for bounded pages; pass the returned next_cursor to resume "
                    "where the previous page stopped."
                ),
                category="filesystem",
                risk=RiskLevel.LOW,
                permissions=["file:read"],
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "pattern": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "offset": {"type": "integer", "minimum": 0,
                                   "maximum": MAX_LINE_OFFSET},
                        "limit": {"type": "integer", "minimum": 0,
                                  "maximum": MAX_PAGE_LIMIT},
                        "cursor": {"type": "string", "minLength": 1},
                    },
                    ["path"],
                ),
                resource_field="path",
            ),
            list_files,
        ),
        (
            ToolSpec(
                name="search_files",
                description=(
                    "Search file contents under an allowed root, or in a SINGLE FILE "
                    "when path points at one. Use limit/offset to page matches."
                ),
                category="filesystem",
                risk=RiskLevel.LOW,
                permissions=["file:read"],
                input_schema=_schema(
                    {
                        "path": {"type": "string", "minLength": 1},
                        "pattern": {"type": "string", "minLength": 1},
                        "regex": {"type": "boolean"},
                        "limit": {"type": "integer", "minimum": 0,
                                  "maximum": MAX_SEARCH_LIMIT},
                        "offset": {"type": "integer", "minimum": 0,
                                   "maximum": MAX_LINE_OFFSET},
                    },
                    ["path", "pattern"],
                ),
                resource_field="path",
            ),
            search_files,
        ),
        (
            ToolSpec(
                name="terminal",
                description=terminal_description(),
                category="development",
                risk=RiskLevel.MEDIUM,
                permissions=["terminal:execute"],
                supports_dry_run=True,
                input_schema=_schema(
                    {
                        "command": {"type": "string", "minLength": 1},
                        "cwd": {"type": "string"},
                        "timeout": {"type": "number", "minimum": 0.1},
                        "max_output_bytes": {"type": "integer", "minimum": 1},
                    },
                    ["command"],
                ),
                resource_field="cwd",
            ),
            terminal,
        ),
        (
            ToolSpec(
                name="http_get",
                description=(
                    "Fetch a public http/https URL without following redirects, under a strict "
                    "raw+decompressed byte cap. Sends Cache-Control: no-cache and returns "
                    "fetched_at plus Date/Age/Cache-Control/ETag evidence; a fetched response "
                    "may still be cached/stale and is not guaranteed real-time."
                ),
                category="research",
                risk=RiskLevel.MEDIUM,
                permissions=["network:read"],
                input_schema=_schema(
                    {
                        "url": {"type": "string", "minLength": 1},
                        "timeout": {"type": "number", "minimum": 0.1},
                        "max_bytes": {"type": "integer", "minimum": 1},
                    },
                    ["url"],
                ),
                resource_field="",  # URL is not a filesystem resource
            ),
            http_get,
        ),
        (
            ToolSpec(
                name="json_parse",
                description="Parse a JSON string into a structured value.",
                category="utility",
                risk=RiskLevel.LOW,
                input_schema=_schema({"text": {"type": "string"}}, ["text"]),
            ),
            json_parse,
        ),
        (
            ToolSpec(
                name="calculator",
                description="Evaluate a numeric expression using a restricted AST.",
                category="utility",
                risk=RiskLevel.LOW,
                input_schema=_schema({"expression": {"type": "string"}}, ["expression"]),
            ),
            calculator,
        ),
        (
            ToolSpec(
                name="current_time",
                description="Return the current UTC or local time as ISO 8601.",
                category="utility",
                risk=RiskLevel.LOW,
                input_schema=_schema(
                    {"timezone": {"type": "string", "enum": ["utc", "local"]}}, []
                ),
            ),
            current_time,
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
