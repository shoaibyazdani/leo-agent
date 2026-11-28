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
import os
import re
import shlex
from pathlib import Path, PurePath

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
        "basename", "cat", "cd", "cut", "date", "df", "diff", "dirname", "du",
        "echo", "false", "file", "head", "ls", "mypy", "pytest", "pwd",
        "readlink", "realpath", "sort", "stat", "tail", "test", "tr", "tree",
        "true", "uniq", "wc", "which", "whoami",
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

_REDIRECT = re.compile(r"(?<![0-9])>{1,2}|<")
_DEVNULL_REDIRECT = re.compile(r"\s*(?:[0-9]?>>?|<)\s*/dev/null")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _mask_quoted(text: str, *, mask_double: bool = True) -> str:
    """Blank quoted spans in place so operators inside strings are not syntax.

    A quoted ``|``, ``;`` or ``>`` is data (SQL concatenation, a regex
    alternation, a commit message), not a pipe or redirect. Escaped characters
    are data too. Positions are preserved, so matches here can be cut out of
    the original string. With ``mask_double=False`` only single-quoted spans
    and escapes are blanked, because ``$(...)`` and backticks still run inside
    double quotes.
    """
    chars = list(text)
    quote = ""
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if quote == "'":
            chars[index] = " "
            if char == "'":
                quote = ""
            index += 1
            continue
        if quote == '"':
            if char == "\\" and index + 1 < length:
                chars[index] = chars[index + 1] = " "
                index += 2
                continue
            chars[index] = " "
            if char == '"':
                quote = ""
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            chars[index] = chars[index + 1] = " "
            index += 2
            continue
        if char == "'" or (char == '"' and mask_double):
            quote = char
            chars[index] = " "
        index += 1
    return "".join(chars)


def _split_segments(text: str) -> list[str]:
    """Split on unquoted ``&&``, ``||``, ``;`` and ``|`` only."""
    masked = _mask_quoted(text)
    segments: list[str] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        if masked[index : index + 2] in {"&&", "||"}:
            segments.append(text[start:index].strip())
            index += 2
            start = index
            continue
        if masked[index] in {"|", ";"}:
            segments.append(text[start:index].strip())
            index += 1
            start = index
            continue
        index += 1
    segments.append(text[start:].strip())
    return [segment for segment in segments if segment]


def _strip_devnull(text: str) -> str:
    """Drop unquoted ``>/dev/null`` noise (quoted text is not a redirect)."""
    masked = _mask_quoted(text)
    pieces: list[str] = []
    last = 0
    for match in _DEVNULL_REDIRECT.finditer(masked):
        pieces.append(text[last : match.start()])
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def _has_substitution(text: str) -> bool:
    masked = _mask_quoted(text, mask_double=False)
    return "$(" in masked or "`" in masked


def _has_redirect(text: str) -> bool:
    return bool(_REDIRECT.search(_mask_quoted(text)))


def _redirect_targets(text: str) -> list[str]:
    """File targets of unquoted redirects, with fd duplication skipped."""
    masked = _mask_quoted(text)
    targets: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if masked[index] not in "<>":
            index += 1
            continue
        cursor = index + 1
        while cursor < length and masked[cursor] in "<>":
            cursor += 1
        if cursor < length and masked[cursor] == "&":
            index = cursor + 1
            continue
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break
        if text[cursor] in {"'", '"'}:
            quote = text[cursor]
            end = cursor + 1
            while end < length:
                if quote == '"' and text[end] == "\\":
                    end += 2
                    continue
                if text[end] == quote:
                    break
                end += 1
            targets.append(text[cursor + 1 : end])
            index = end + 1
        else:
            end = cursor
            while (
                end < length
                and not text[end].isspace()
                and masked[end] not in ";&|<>"
            ):
                end += 1
            targets.append(text[cursor:end])
            index = end
    return targets


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
    if os.getenv("LEO_WHATSAPP_SERVE") == "1":
        # This command runs INSIDE leo-whatsapp. Restarting that unit kills the
        # current turn before its answer can be sent, so the operator gets silence
        # and systemd starts a fresh listener a few seconds later. Diagnose freely,
        # but manage this service from Telegram or a real shell.
        for segment in _split_segments(command):
            if (
                re.search(r"\b(?:systemctl|service)\b", segment)
                and re.search(r"\b(?:restart|stop|try-restart|reload-or-restart|kill)\b", segment)
                and re.search(r"\bleo-whatsapp(?:\.service)?\b", segment)
            ):
                return (
                    "refusing to stop leo-whatsapp from inside its own turn; "
                    "restart it from Telegram or the shell"
                )
    return None


def risk(tool_name: str, arguments: dict[str, object], settings: Settings) -> RiskLevel:
    """Per-command risk used by the executor's approval gate (thin wrapper)."""
    if tool_name != SHELL_TOOL:
        return RiskLevel.LOW
    return assess_command(str(arguments.get("command") or ""), settings)[0]


def _smart_safe(command: str) -> bool:
    text = _strip_devnull(command)
    if not text.strip():
        return True
    # Command substitution and redirections can execute or overwrite outside
    # the visible command; escalate them rather than guessing. Quoted operators
    # are data, not syntax.
    if _has_substitution(text) or _has_redirect(text):
        return False
    segments = _split_segments(text)
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


# ---------------------------------------------------------------------------
# Three-tier assessment (LOW runs · MEDIUM runs and is audited · HIGH asks)
#
# The boolean classifier above answered one question: "is this exactly a known
# read-only command?" Everything else escalated. In practice that asked a human
# about `journalctl -u leo-brain`, `sqlite3 db "SELECT …"` and `systemctl status`,
# so approvals said more about the allowlist's coverage than about risk.
# ---------------------------------------------------------------------------

_READ_ONLY_COMMANDS = frozenset(
    {
        "base64", "column", "comm", "env", "free", "hostname",
        "id", "jq", "journalctl", "md5sum", "pgrep", "printenv", "ps", "sha1sum",
        "sha256sum", "ss", "type", "uname", "uptime", "xargs", "zgrep",
    }
)

_MUTATING_COMMANDS = frozenset(
    {
        "chgrp", "chmod", "cp", "install", "ln", "mkdir", "mkfifo", "mv",
        "pip", "pip3", "printf", "tee", "touch", "unzip", "zip", "tar",
    }
)

_DESTRUCTIVE_COMMANDS = frozenset(
    {
        "chown", "dd", "kill", "killall", "mkfs", "mount", "pkill", "reboot",
        "rm", "rmdir", "shred", "shutdown", "swapoff", "swapon", "truncate",
        "umount", "useradd", "userdel", "usermod",
    }
)

_READ_ONLY_SYSTEMCTL = frozenset(
    {
        "cat", "is-active", "is-enabled", "is-failed", "list-jobs", "list-sockets",
        "list-timers", "list-unit-files", "list-units", "show", "status",
    }
)
_MUTATING_SYSTEMCTL = frozenset(
    {"daemon-reload", "disable", "enable", "reload", "reset-failed", "restart", "start", "stop", "try-restart"}
)
_DESTRUCTIVE_SYSTEMCTL = frozenset({"halt", "kexec", "mask", "poweroff", "reboot", "rescue", "suspend", "unmask"})

_SQL_WRITE = re.compile(r"(?i)\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|VACUUM|REINDEX)\b")
_HTTP_WRITE_FLAGS = ("-o", "-O", "-d", "-F", "-T", "-X", "--data", "--upload-file", "--output", "--form")
_AWK_EXEC = re.compile(r"(?i)\bsystem\s*\(|\bENVIRON\b|\|")


def _systemctl_tier(args: list[str]) -> tuple[RiskLevel, str]:
    words = [arg for arg in args if not arg.startswith("-")]
    sub = words[0] if words else ""
    if sub in _DESTRUCTIVE_SYSTEMCTL:
        return RiskLevel.HIGH, f"systemctl {sub} changes host power/unit policy"
    if sub in _MUTATING_SYSTEMCTL:
        return RiskLevel.MEDIUM, f"systemctl {sub} restarts a service"
    if sub in _READ_ONLY_SYSTEMCTL or not sub:
        return RiskLevel.LOW, "systemctl status read-out"
    return RiskLevel.HIGH, f"unclassified systemctl verb ({sub})"


def _sqlite_tier(args: list[str]) -> tuple[RiskLevel, str]:
    statement = " ".join(args[1:]) if len(args) > 1 else ""
    if _SQL_WRITE.search(statement):
        return RiskLevel.MEDIUM, "sqlite write statement"
    if statement.strip() or len(args) >= 2:
        return RiskLevel.LOW, "sqlite read-only query"
    return RiskLevel.HIGH, "interactive sqlite shell"


def _http_tier(name: str, args: list[str]) -> tuple[RiskLevel, str]:
    if any(arg == flag or arg.startswith(flag + "=") for arg in args for flag in _HTTP_WRITE_FLAGS):
        return RiskLevel.MEDIUM, f"{name} downloads or posts"
    return RiskLevel.LOW, f"{name} fetch"


def _awk_tier(args: list[str]) -> tuple[RiskLevel, str]:
    program = " ".join(args)
    if _AWK_EXEC.search(program):
        return RiskLevel.HIGH, "awk can execute or pipe into a shell"
    return RiskLevel.MEDIUM, "awk program is evaluated, not inspected"


_TIER_HANDLERS = {
    "systemctl": _systemctl_tier,
    "service": _systemctl_tier,
    "sqlite3": _sqlite_tier,
    "curl": lambda args: _http_tier("curl", args),
    "wget": lambda args: _http_tier("wget", args),
    "awk": _awk_tier,
    "gawk": _awk_tier,
    "journalctl": lambda _args: (RiskLevel.LOW, "journal read-out"),
}


_INTERPRETERS = frozenset({"python", "python3", "node", "perl", "ruby", "bash", "sh", "zsh", "php"})


def _write_tier(name: str, args: list[str]) -> tuple[RiskLevel, str]:
    if name in _INTERPRETERS:
        return RiskLevel.HIGH, f"{name} runs arbitrary code"
    if name == "git":
        joined = " ".join(args)
        destructive = (
            "--force" in args
            or "-f" in args
            or args[:1] == ["clean"]
            or ("reset" in args and "--hard" in args)
            or ("push" in args and any(arg.startswith("--force") for arg in args))
        )
        if "branch" in args and any(arg in {"-d", "-D", "--delete"} for arg in args):
            return RiskLevel.HIGH, "git branch deletion can lose commits"
        if destructive:
            return RiskLevel.HIGH, "git history or remote rewrite"
        return RiskLevel.MEDIUM, "git write operation"
    if name in {"pip", "pip3"} and args[:1] in (["uninstall"],):
        return RiskLevel.HIGH, "removes an installed package"
    if name == "chmod" and any(arg in {"-R", "--recursive"} for arg in args):
        return RiskLevel.HIGH, "recursive permission change"
    return RiskLevel.MEDIUM, f"{name} writes"


def _segment_tier(segment: str) -> tuple[RiskLevel, str]:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return RiskLevel.HIGH, "unparseable command"
    if not tokens:
        return RiskLevel.LOW, "empty segment"
    while tokens and _ENV_ASSIGN.match(tokens[0]):
        tokens.pop(0)
    if not tokens:
        return RiskLevel.LOW, "environment assignment only"
    if tokens[0] == "env":
        tokens.pop(0)
        while tokens and (_ENV_ASSIGN.match(tokens[0]) or tokens[0] == "-i"):
            tokens.pop(0)
        if not tokens:
            return RiskLevel.LOW, "environment read-out"
    if tokens[0] in {"command", "builtin"}:
        tokens.pop(0)
        if tokens and tokens[0] in {"-v", "-V"}:
            return RiskLevel.LOW, "command lookup"
        if not tokens:
            return RiskLevel.LOW, "command lookup"
    if tokens[0] in {"sudo", "doas", "su", "eval", "exec"}:
        return RiskLevel.HIGH, f"{tokens[0]} escalates privileges"
    name = PurePath(tokens[0]).name
    args = tokens[1:]
    if name in _SAFE_COMMANDS:
        return RiskLevel.LOW, f"{name} is read-only"
    if name in _SAFE_REINTERPRETED:
        if _reinterpreted_safe(name, args):
            return RiskLevel.LOW, f"{name} read-only form"
        return _write_tier(name, args)
    handler = _TIER_HANDLERS.get(name)
    if handler is not None:
        return handler(args)
    if name in _MUTATING_COMMANDS:
        return _write_tier(name, args)
    if name in _READ_ONLY_COMMANDS:
        return RiskLevel.LOW, f"{name} is read-only"
    if name in _DESTRUCTIVE_COMMANDS:
        return RiskLevel.HIGH, f"{name} is destructive"
    return RiskLevel.HIGH, f"unclassified command ({name})"


def _redirect_is_bounded(text: str, settings: Settings) -> bool:
    """True when every file redirect target is inside a trusted root or /tmp.

    A normal `pytest > /tmp/out` is a MEDIUM write; `echo x > /etc/hosts` is
    not. The shell tool does not sandbox redirects, so the target is checked
    explicitly rather than treating every redirect as arbitrary code.
    """
    roots = [settings.root.resolve()]
    roots.extend(settings.resolve(p).resolve() for p in settings.security.allowed_roots)
    roots.append(settings.resolve(settings.paths.workspaces).resolve())
    for raw in _redirect_targets(text):
        target = raw.strip("'\"")
        if not target or target.startswith("&"):
            continue  # fd duplication, e.g. 2>&1
        if "$" in target or "`" in target:
            return False  # the target is computed at run time
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = settings.root / path
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        if resolved == Path("/dev/null"):
            continue
        tmp_root = Path("/tmp").resolve()
        if resolved == tmp_root or tmp_root in resolved.parents:
            continue
        if any(resolved == root or root in resolved.parents for root in roots):
            continue
        return False
    return True


def assess_command(command: str, settings: Settings) -> tuple[RiskLevel, str]:
    """Return ``(risk, reason)`` for one shell command line."""
    shell = settings.security
    if not shell.shell_enabled:
        return RiskLevel.HIGH, "shell disabled by security.shell_enabled"
    if shell.shell_mode == "off":
        return RiskLevel.LOW, "shell policy off"
    if shell.shell_mode == "manual":
        return RiskLevel.HIGH, "shell policy manual: every command asks"
    text = _strip_devnull(str(command or ""))
    if not text.strip():
        return RiskLevel.LOW, "empty command"
    denial = _hardline(text)
    if denial:
        return RiskLevel.HIGH, f"hardline blocklist: {denial}"
    if _has_substitution(text):
        return RiskLevel.HIGH, "command substitution"
    has_redirect = _has_redirect(text)
    if has_redirect and not _redirect_is_bounded(text, settings):
        return RiskLevel.HIGH, "redirection outside allowed roots"
    segments = _split_segments(text)
    decision: tuple[RiskLevel, str] = (RiskLevel.LOW, "read-only command line")
    for segment in segments:
        tier, reason = _segment_tier(segment)
        if tier is RiskLevel.HIGH:
            return RiskLevel.HIGH, reason
        if tier is RiskLevel.MEDIUM:
            decision = (RiskLevel.MEDIUM, reason)
    if has_redirect and decision[0] is RiskLevel.LOW:
        decision = (RiskLevel.MEDIUM, "bounded redirection")
    return decision
