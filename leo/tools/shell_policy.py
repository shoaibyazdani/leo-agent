"""Smart shell policy: hardline floor, deny globs, and smart risk.

The reference policy defaults to ``approvals.mode: smart``: low-risk commands run, genuinely
dangerous commands are denied or escalated, and a hardline blocklist is always
enforced. LEO previously exposed only a shell-free allowlisted terminal. This
module adds the policy layer for the new ``shell`` tool.

The smart classifier here is rule-based rather than an auxiliary-LLM
assessor. It is deliberately conservative: read-only and test/lint commands run,
mutating or unknown commands go through the normal approval gate, and the
hardline floor always denies. It is not a sandbox; shell commands run with the
LEO process's privileges, exactly as a local terminal does.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from pathlib import PurePath

from ..config import Settings
from ..contracts import RiskLevel

SHELL_TOOL = "shell"

# Always-on floor, mirroring the reference hardline patterns. These deny regardless of
# --yolo / mode: off and are not routed through the approval UI.
_HARDLINE: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)\brm\s+(?:-[a-z]*[rf][a-z]*\s+|--\s+)*(?:/|/\*)(?:\s|$)"),
        "filesystem root wipe (rm -rf /)",
    ),
    (
        re.compile(r"(?i)\brm\b[^\n;&|]*\s(?:/|/\*)\s*(?:$|[;&|])"),
        "filesystem root wipe (rm /)",
    ),
    (re.compile(r"--no-preserve-root", re.I), "rm --no-preserve-root"),
    (re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.I), "fork bomb"),
    (re.compile(r"(?i)\bmkfs(?:\.[a-z0-9]+)?\b[^\n]*(?:/dev/|/root)"), "format a block device"),
    (re.compile(r"(?i)\bdd\b[^\n]*\bof=/dev/(?:sd|hd|nvme|disk|mmcblk)"), "write a physical disk"),
    (re.compile(r"(?i)\bformat\s+[a-z]:"), "format a Windows volume"),
    (re.compile(r"(?i)\bClear-Disk\b"), "clear a Windows disk"),
    (re.compile(r"(?i)\bRemove-Item\b[^\n]*-Recurse[^\n]*[a-z]:\\"), "recursive Windows drive delete"),
    (re.compile(r"(?i)\brd\s+/s\s+/q\s+[a-z]:\\"), "recursive Windows drive delete"),
    (re.compile(r"(?i)\b(?:Restart|Stop)-Computer\b"), "power-state change"),
    (
        re.compile(r"(?i)\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|k)?sh\b"),
        "pipe a remote download into a shell",
    ),
)

# Safe read-only / verification commands. Anything not here (or explicitly safe
# below) is escalated in smart mode.
_SAFE_COMMANDS = frozenset(
    {
        "basename", "cat", "cut", "date", "df", "diff", "dirname", "du", "echo",
        "false", "file", "head", "ls", "mypy", "pytest", "pwd", "readlink",
        "realpath", "sort", "stat", "tail", "test", "tr", "tree", "true",
        "uniq", "wc", "which", "whoami",
    }
)

_SAFE_GIT = frozenset(
    {
        "status", "diff", "log", "show", "branch", "ls-files", "rev-parse",
        "blame", "help", "grep", "ls-tree", "cat-file", "reflog", "worktree",
        "remote", "tag", "config", "stash", "describe", "shortlog",
    }
)
_GIT_MUTATING = frozenset(
    {
        "add", "commit", "push", "pull", "fetch", "reset", "clean", "checkout",
        "switch", "merge", "rebase", "cherry-pick", "revert", "init", "clone",
        "mv", "rm", "apply", "am", "bisect", "gc", "prune", "update-ref",
    }
)
_GIT_BRANCH_MUTATING = frozenset(
    {"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy", "--set-upstream-to", "--unset-upstream"}
)
_GIT_UNSAFE_OPTIONS = frozenset(
    {"--config", "--config-env", "--exec-path", "--git-dir", "--work-tree", "--namespace", "--super-prefix"}
)

_SAFE_PY_MODULES = frozenset({"pytest", "unittest", "mypy", "ruff", "compileall"})
_SAFE_NPM_SCRIPTS = frozenset({"test", "lint", "typecheck", "type-check"})
_SAFE_MAKE_TARGETS = frozenset({"test", "tests", "check", "lint", "typecheck"})
_SAFE_CARGO = frozenset({"test", "check", "clippy", "fmt", "metadata", "--version", "-V"})

_SAFE_REINTERPRETED = {
    "git", "find", "sed", "grep", "rg", "python", "python3", "node", "npm",
    "yarn", "pnpm", "bun", "make", "cargo", "go", "tsc", "eslint", "black",
    "prettier", "ruff",
}

_SEGMENT_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_REDIRECT = re.compile(r"(?<![0-9])>{1,2}|<")
_DEVNULL_REDIRECT = re.compile(r"\s*(?:[0-9]?>>?|<)\s*/dev/null")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _hardline(command: str) -> str | None:
    for pattern, reason in _HARDLINE:
        if pattern.search(command):
            return reason
    return None


def preflight(tool_name: str, arguments: dict[str, object], settings: Settings) -> str | None:
    """Return a denial reason before permissions/approval, or None."""
    if tool_name != SHELL_TOOL:
        return None
    shell = settings.security
    if not shell.shell_enabled:
        return "shell disabled by security.shell_enabled"
    command = str(arguments.get("command") or "")
    if "\x00" in command:
        return "command contains a NUL byte"
    reason = _hardline(command)
    if reason:
        return f"hardline blocklist: {reason}"
    for pattern in shell.deny_commands:
        if fnmatch.fnmatch(command.strip(), pattern):
            return f"command denied by security.deny_commands: {pattern}"
    return None


def risk(tool_name: str, arguments: dict[str, object], settings: Settings) -> RiskLevel:
    """Per-command risk used by the executor's approval gate."""
    if tool_name != SHELL_TOOL:
        return RiskLevel.LOW
    shell = settings.security
    if not shell.shell_enabled:
        return RiskLevel.HIGH
    command = str(arguments.get("command") or "")
    if preflight(tool_name, arguments, settings):
        return RiskLevel.HIGH
    if shell.shell_mode == "off":
        return RiskLevel.LOW
    if shell.shell_mode == "manual":
        return RiskLevel.HIGH
    return RiskLevel.LOW if _smart_safe(command) else RiskLevel.HIGH


def _smart_safe(command: str) -> bool:
    text = _DEVNULL_REDIRECT.sub("", command)
    if not text.strip():
        return True
    # Command substitution and redirections can execute or overwrite outside
    # the visible command; escalate them rather than guessing.
    if "$(" in text or "`" in text or _REDIRECT.search(text):
        return False
    segments = [part.strip() for part in _SEGMENT_SPLIT.split(text) if part.strip()]
    if not segments:
        return True
    return all(_segment_safe(segment) for segment in segments)


def _segment_safe(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens:
        return True
    while tokens and _ENV_ASSIGN.match(tokens[0]):
        tokens.pop(0)
    if not tokens:
        return True
    if tokens[0] == "env":
        tokens.pop(0)
        if any(token.startswith("-") for token in tokens):
            return False
        while tokens and _ENV_ASSIGN.match(tokens[0]):
            tokens.pop(0)
        if not tokens:
            return True
    if tokens[0] in {"command", "builtin"}:
        tokens.pop(0)
        if tokens and tokens[0] in {"-v", "-V"}:
            return True
        if tokens and tokens[0].startswith("-"):
            return False
    elif tokens[0] in {"sudo", "doas", "su", "eval", "exec"}:
        return False
    name = PurePath(tokens[0]).name
    args = tokens[1:]
    if name in _SAFE_COMMANDS:
        return True
    if name in _SAFE_REINTERPRETED:
        return _reinterpreted_safe(name, args)
    return False


def _reinterpreted_safe(name: str, args: list[str]) -> bool:
    if name == "git":
        return _git_safe(args)
    if name == "find":
        return not any(arg.startswith(("-delete", "-exec", "-execdir", "-ok", "-fprint", "-fls")) for arg in args)
    if name == "sed":
        return not any(arg in {"-i", "--in-place"} or arg.startswith("--in-place=") for arg in args)
    if name in {"grep", "rg"}:
        return not any(arg.startswith(("--exec", "--passthru-exec")) for arg in args)
    if name in {"python", "python3"}:
        if any(arg in {"--version", "-V"} for arg in args):
            return True
        return len(args) >= 2 and args[0] == "-m" and args[1] in _SAFE_PY_MODULES
    if name == "node":
        return any(arg in {"--version", "-v"} for arg in args)
    if name in {"npm", "yarn", "pnpm", "bun"}:
        if any(arg in {"--version", "-v"} for arg in args):
            return True
        if not args:
            return False
        if args[0] in {"test", "ls", "list"}:
            return True
        return len(args) >= 2 and args[0] == "run" and args[1] in _SAFE_NPM_SCRIPTS
    if name == "make":
        return bool(args) and all(arg in _SAFE_MAKE_TARGETS for arg in args if not arg.startswith("-"))
    if name == "cargo":
        return bool(args) and args[0] in _SAFE_CARGO and (
            args[0] != "fmt" or "--check" in args
        )
    if name == "go":
        return bool(args) and args[0] in {"test", "vet", "build", "version", "env"}
    if name == "tsc":
        return "--noEmit" in args
    if name == "eslint":
        return not any(arg in {"--fix", "--fix-dry-run"} for arg in args)
    if name in {"black", "prettier"}:
        return "--check" in args or "--diff" in args
    if name == "ruff":
        if any(arg in {"--version", "-V"} for arg in args):
            return True
        if any(arg in {"--fix", "--unsafe-fixes"} for arg in args):
            return False
        return bool(args) and args[0] in {"check", "format"} and (
            args[0] != "format" or "--check" in args or "--diff" in args
        )
    return False


def _git_safe(args: list[str]) -> bool:
    if not args:
        return False
    if args[0] in {"--version", "version", "help"}:
        return True
    if args[0].startswith("-"):
        head = args[0].split("=", 1)[0]
        if head in _GIT_UNSAFE_OPTIONS or head.startswith(("-c", "-C")):
            return False
    index = 0
    while index < len(args) and args[index].startswith("-"):
        index += 1
    if index >= len(args):
        return False
    sub = args[index]
    rest = args[index + 1 :]
    if sub in _GIT_MUTATING:
        return False
    if sub not in _SAFE_GIT:
        return False
    if sub == "branch":
        return not any(arg in _GIT_BRANCH_MUTATING for arg in rest)
    if sub == "tag":
        return all(arg in {"-l", "--list", "--contains", "--merged", "--no-merged", "-n", "--sort"} or arg.startswith(("--contains=", "--sort=")) for arg in rest)
    if sub == "remote":
        return not rest or all(arg in {"-v", "--verbose", "show", "get-url"} for arg in rest)
    if sub == "config":
        return bool(rest) and rest[0] in {"--get", "--get-all", "--list", "-l", "--show-origin"}
    if sub == "stash":
        return not rest or rest[0] == "list"
    if sub == "worktree":
        return not rest or rest[0] == "list"
    if sub == "reflog":
        return not rest or rest[0] in {"show", "exists"}
    return True
