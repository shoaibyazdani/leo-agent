"""Smart shell policy: hardline floor, deny globs, and smart risk.

The reference policy defaults to ``approvals.mode: smart``: low-risk commands run, genuinely
dangerous commands are denied or escalated, and a hardline blocklist is always
enforced. LEO previously exposed only a shell-free allowlisted terminal. This
module adds the policy layer for the new ``shell`` tool.

The smart classifier here is rule-based rather than an auxiliary-LLM
assessor. Read-only commands and bounded everyday dev tools run; a reversible
write or a command LEO does not recognise inside the workspace is MEDIUM (run
and audited rather than prompting); destructive, credential, privilege,
outbound and self-modifying commands stay HIGH. The hardline floor always
denies. It is not a sandbox: shell commands run with the LEO process's
privileges, exactly as a local terminal does.
"""

from __future__ import annotations

import fnmatch
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
    hosts = _url_hosts(" ".join(args))
    if hosts:
        return RiskLevel.HIGH, f"{name} sends a request to an external host ({hosts[0]})"
    remote_targets = [
        arg for arg in args
        if not arg.startswith("-") and "." in arg and "/" not in arg
    ]
    if not any(arg for arg in args if not arg.startswith("-")):
        return RiskLevel.HIGH, f"{name} has no explicit local target"
    if remote_targets and not _host_is_local(remote_targets[0]):
        return RiskLevel.HIGH, f"{name} sends a request to an external host ({remote_targets[0]})"
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
_SHELL_SCRIPTS = (".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd")

# Ordinary development tools. A recognised tool gets a bounded rule (LOW for
# inspection, MEDIUM for a reversible local write) instead of the old blanket
# HIGH; anything that crosses a hard boundary below is still escalated first.
_DEV_TEST_TOOLS = frozenset(
    {"pytest", "py.test", "tox", "jest", "vitest", "mocha", "flake8", "pylint"}
)
_PACKAGE_MANAGERS = frozenset(
    {"pip", "pip3", "uv", "poetry", "pdm", "npm", "yarn", "pnpm", "bun", "npx"}
)
_PACKAGE_READ = frozenset(
    {"list", "ls", "show", "freeze", "check", "why", "info", "version", "--version", "-V"}
)
_PACKAGE_WRITE = frozenset(
    {"install", "add", "ci", "sync", "update", "upgrade", "venv", "init", "build", "run", "exec"}
)
_READ_ONLY_GH = frozenset(
    {"status", "pr", "issue", "run", "repo", "workflow", "release", "gist", "label"}
)
_GH_WRITE_PREFIXES = frozenset({"create", "edit", "close", "reopen", "merge", "delete", "upload", "review"})

# Hard boundaries for the shell. These are checked before any allowlist so a
# familiar tool cannot smuggle a destructive, credential, outbound or
# self-modifying action past the classifier.
_PRIVILEGE_TOOLS = frozenset(
    {"sudo", "doas", "su", "pkexec", "runuser", "sudoedit", "visudo", "setpriv", "passwd", "chsh"}
)
_DESTRUCTIVE_TOOLS = frozenset(_DESTRUCTIVE_COMMANDS) | {
    "shred", "unlink", "chown", "flatpak", "snap",
}
_NETWORK_TOOLS = frozenset(
    {"ssh", "scp", "sftp", "nc", "ncat", "netcat", "telnet", "ftp", "curl", "wget"}
)
_SECRET_FILE_NAMES = frozenset(
    {
        ".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519",
        "id_ecdsa", "id_dsa", "credentials", "credentials.json", "secrets.json",
        "private.key", "secret.key", "shadow", "authorized_keys",
    }
)
_SELF_MODIFY_NAMES = frozenset(
    {"leo.yaml", "config.yaml", "settings.json", "permissions.py", "sudoers", "hosts"}
)
_SENSITIVE_DIRS = frozenset(
    {".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config", ".claude"}
)
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".kdbx")
_CREDENTIAL_FLAG = re.compile(
    r"(?i)^--?(?:token|password|passwd|secret|auth|credential|api[-_]?key|access[-_]?key|private[-_]?key)(?:=|$)"
)
_SECRET_ENV_VAR = re.compile(
    r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)
_URL_HOST = re.compile(r"(?i)\b(?:https?|ftp)://([^/\s'\"<>;|&]+)")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})
_DESTROY_FLAGS = frozenset(
    {
        "--delete", "--destroy", "--purge", "--prune", "--force", "--hard",
        "-delete", "-exec", "-execdir", "-ok", "--exec", "--passthru-exec",
        "-rf", "-rF", "-fr", "-Rf",
    }
)
_REMOTE_WRITE = {
    "docker": frozenset({"push"}),
    "kubectl": frozenset({"apply", "delete", "patch", "scale", "rollout", "create"}),
    "terraform": frozenset({"apply", "destroy"}),
    "aws": frozenset({"s3", "ec2", "lambda"}),
}


def _write_tier(name: str, args: list[str]) -> tuple[RiskLevel, str]:
    if name in _INTERPRETERS:
        return RiskLevel.HIGH, f"{name} runs arbitrary code"
    if name == "git":
        sub = next((arg for arg in args if not arg.startswith("-")), "")
        if sub == "push":
            return RiskLevel.HIGH, "git push publishes a change to a remote"
        if sub.startswith("credential"):
            return RiskLevel.HIGH, "git accesses stored credentials"
        if sub in {"fetch", "pull"}:
            return RiskLevel.MEDIUM, "git synchronises with a remote"
        destructive = (
            "--force" in args
            or "-f" in args
            or sub == "clean"
            or (sub == "reset" and "--hard" in args)
            or (sub == "checkout" and "--" in args)
            or any(arg.startswith("--force") for arg in args)
        )
        if "branch" in args and any(arg in {"-d", "-D", "--delete"} for arg in args):
            return RiskLevel.HIGH, "git branch deletion can lose commits"
        if destructive:
            return RiskLevel.HIGH, "git history or remote rewrite"
        return RiskLevel.MEDIUM, "git write operation"
    if name in {"pip", "pip3", "uv", "poetry", "pdm"} and args[:1] in (["uninstall"], ["remove"]):
        return RiskLevel.HIGH, "removes an installed package"
    if name == "chmod" and any(
        arg in {"-R", "--recursive", "+s", "u+s", "g+s", "4755", "2755"} for arg in args
    ):
        return RiskLevel.HIGH, "recursive permission or setuid change"
    return RiskLevel.MEDIUM, f"{name} writes"


def _allowed_roots(settings: Settings) -> list[Path]:
    roots = [settings.root.resolve()]
    try:
        roots.extend(settings.resolve(p).resolve() for p in settings.security.allowed_roots)
        roots.append(settings.resolve(settings.paths.workspaces).resolve())
    except (OSError, RuntimeError, ValueError):
        pass
    roots.append(Path("/tmp").resolve())
    return roots


def _path_arg(arg: str) -> str | None:
    """A path-valued argument, or None for flags and URLs."""
    text = str(arg or "").strip()
    if not text or text.startswith("-") or "://" in text:
        return None
    return text or None


def _outside_path_reason(args: list[str], settings: Settings) -> str | None:
    roots = _allowed_roots(settings)
    for raw in args:
        candidate = _path_arg(raw)
        if candidate is None:
            continue
        if "$" in candidate or "`" in candidate:
            return f"computed path outside a reviewable location ({candidate})"
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            if ".." not in PurePath(candidate).parts:
                continue
            path = settings.root / path
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            return f"unresolvable path ({candidate})"
        if resolved == Path("/dev/null") or any(
            resolved == root or root in resolved.parents for root in roots
        ):
            continue
        return f"path outside the workspace ({candidate})"
    return None


def _secret_env_reason(segment: str) -> str | None:
    masked = _mask_quoted(segment, mask_double=False)
    markers = frozenset(
        {"TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "CREDENTIAL", "COOKIE", "AUTH"}
    )
    for match in _SECRET_ENV_VAR.finditer(masked):
        name = (match.group(1) or match.group(2) or "").upper()
        if markers.intersection(name.split("_")):
            return f"reads a secret environment variable (${name})"
    return None


def _credential_reason(segment: str, args: list[str]) -> str | None:
    secret = _secret_env_reason(segment)
    if secret:
        return secret
    for arg in args:
        if _CREDENTIAL_FLAG.match(arg):
            return f"passes a credential ({arg.split('=', 1)[0]})"
        candidate = _path_arg(arg)
        if candidate is None:
            continue
        parts = PurePath(candidate).parts
        sensitive_dir = next(
            (part for part in parts if part.lower() in _SENSITIVE_DIRS), None
        )
        if sensitive_dir:
            return f"touches a credential directory ({sensitive_dir})"
        name = PurePath(candidate).name.lower()
        if name in _SECRET_FILE_NAMES or name.startswith(".env.") or name.endswith(
            _SENSITIVE_SUFFIXES
        ):
            return f"touches a credential file ({name})"
    return None


def _host_is_local(host: str) -> bool:
    text = host.strip().lower()
    if text.startswith("["):
        text = text[1:].split("]", 1)[0]
    else:
        text = text.split(":", 1)[0]
    return text in _LOCAL_HOSTS or text.startswith("127.")


def _url_hosts(segment: str) -> list[str]:
    return [host for host in _URL_HOST.findall(segment) if not _host_is_local(host)]


def _outbound_reason(name: str, args: list[str]) -> str | None:
    if name in {"ssh", "scp", "sftp", "nc", "ncat", "netcat", "telnet", "ftp"}:
        return f"{name} opens an outbound connection"
    verbs = _REMOTE_WRITE.get(name)
    if verbs:
        sub = next((arg for arg in args if not arg.startswith("-")), "")
        if sub in verbs:
            return f"{name} {sub} changes remote state"
    known = name in _SAFE_COMMANDS or name in _SAFE_REINTERPRETED or name in _TIER_HANDLERS
    hosts = _url_hosts(" ".join(args))
    if hosts and not known:
        return f"sends data to an external host ({hosts[0]})"
    return None


def _is_write_command(name: str, args: list[str]) -> bool:
    if name in _MUTATING_COMMANDS:
        return True
    if name in _PACKAGE_MANAGERS:
        return True
    if name in {"tee", "sed", "awk", "gawk", "sqlite3", "systemctl", "service"}:
        return True
    return any(arg in {"-i", "--in-place", "-o", "--output", "-d", "--data"} for arg in args)


def _boundary_reason(
    name: str, args: list[str], segment: str, settings: Settings
) -> str | None:
    """A hard boundary outranks every allowlist tier."""
    if name in _PRIVILEGE_TOOLS:
        return f"{name} escalates privileges"
    if any(
        arg in {"--no-sandbox", "+s", "u+s", "g+s", "--privileged", "--cap-add", "--security-opt"}
        or arg.startswith("--userns=")
        for arg in args
    ):
        return "privilege escalation or setuid change"
    if name in _DESTRUCTIVE_TOOLS:
        return f"{name} is destructive"
    destructive_flag = next(
        (arg for arg in args if arg in _DESTROY_FLAGS), None
    )
    if destructive_flag:
        return f"{name} uses destructive option {destructive_flag}"
    credential = _credential_reason(segment, args)
    if credential:
        return credential
    sub = next((arg for arg in args if not arg.startswith("-")), "")
    if (name == "docker" and sub == "login") or (
        name == "git" and sub.startswith("credential")
    ):
        return f"{name} {sub} accesses stored credentials"
    outbound = _outbound_reason(name, args)
    if outbound:
        return outbound
    write = _is_write_command(name, args)
    if write:
        outside = _outside_path_reason(args, settings)
        if outside:
            return outside
        for arg in args:
            candidate = _path_arg(arg)
            if candidate and PurePath(candidate).name.lower() in _SELF_MODIFY_NAMES:
                return f"{name} changes LEO's own configuration ({PurePath(candidate).name})"
    else:
        known = (
            name in _SAFE_COMMANDS
            or name in _SAFE_REINTERPRETED
            or name in _TIER_HANDLERS
            or name in _MUTATING_COMMANDS
            or name in _READ_ONLY_COMMANDS
            or name in _DEV_TEST_TOOLS
            or name in _PACKAGE_MANAGERS
            or name in _NETWORK_TOOLS
            or name == "gh"
        )
        if not known:
            outside = _outside_path_reason([name, *args], settings)
            if outside:
                return outside
    return None


def _script_tier(
    name: str, args: list[str], settings: Settings
) -> tuple[RiskLevel, str]:
    if any(arg in {"-c", "-e", "-p", "--eval", "--command"} for arg in args):
        return RiskLevel.HIGH, f"{name} evaluates arbitrary inline code"
    if args and args[0] == "-m":
        module = args[1] if len(args) > 1 else "?"
        return RiskLevel.HIGH, f"{name} -m {module} runs module code not on the safe list"
    script = next((arg for arg in args if not arg.startswith("-")), None)
    if script is None:
        return RiskLevel.HIGH, f"{name} runs an interpreter with no script"
    if script.endswith(_SHELL_SCRIPTS) or Path(script).suffix in {".py", ".js", ".mjs", ".cjs", ".ts", ".rb", ".pl", ".php"}:
        outside = _outside_path_reason([script], settings)
        if outside:
            return RiskLevel.HIGH, outside
        return RiskLevel.MEDIUM, f"{name} runs workspace script {PurePath(script).name}"
    return RiskLevel.HIGH, f"{name} runs an opaque target ({script})"


def _package_tier(
    name: str, args: list[str], settings: Settings
) -> tuple[RiskLevel, str]:
    subs = [arg for arg in args if not arg.startswith("-")]
    if name == "uv" and subs and subs[0] == "pip":
        subs = subs[1:]
    sub = subs[0] if subs else ""
    if not sub or sub in _PACKAGE_READ:
        return RiskLevel.LOW, f"{name} reads package metadata"
    if sub in {"publish", "release"}:
        return RiskLevel.HIGH, f"{name} {sub} publishes a package to a registry"
    if sub in {"uninstall", "remove"}:
        return RiskLevel.HIGH, f"{name} {sub} removes an installed package"
    if any(arg.startswith("git+") or "://" in arg for arg in args):
        return RiskLevel.HIGH, f"{name} installs from a remote source"
    if any(arg in {"-g", "--global", "--user"} for arg in args):
        return RiskLevel.HIGH, f"{name} {sub} writes outside the workspace"
    outside = _outside_path_reason(args, settings)
    if outside:
        return RiskLevel.HIGH, outside
    if sub in _PACKAGE_WRITE:
        return RiskLevel.MEDIUM, f"{name} {sub} changes the workspace environment"
    return RiskLevel.MEDIUM, f"{name} {sub} is not on the read-only list"


def _gh_tier(name: str, args: list[str]) -> tuple[RiskLevel, str]:
    sub = next((arg for arg in args if not arg.startswith("-")), "")
    if not sub or sub == "help":
        return RiskLevel.LOW, "gh help read-out"
    if sub == "auth":
        return RiskLevel.HIGH, "gh accesses stored credentials"
    if sub == "api":
        return RiskLevel.HIGH, "gh api sends an arbitrary GitHub request"
    if sub not in _READ_ONLY_GH:
        return RiskLevel.MEDIUM, f"gh {sub} is not on the read-only list"
    rest = args[args.index(sub) + 1 :]
    verb = next((arg for arg in rest if not arg.startswith("-")), "")
    if verb in _GH_WRITE_PREFIXES:
        return RiskLevel.HIGH, f"gh {sub} {verb} changes remote state"
    if verb and verb not in {"list", "view", "diff", "checks", "status", "show", "get", "download"}:
        return RiskLevel.MEDIUM, f"gh {sub} {verb} is not on the read-only list"
    return RiskLevel.LOW, f"gh {sub} reads remote state"


def _dev_tool_tier(
    name: str, args: list[str], settings: Settings
) -> tuple[RiskLevel, str] | None:
    if name in _DEV_TEST_TOOLS:
        return RiskLevel.LOW, f"{name} runs the test/verification suite"
    if name in _PACKAGE_MANAGERS:
        return _package_tier(name, args, settings)
    if name == "gh":
        return _gh_tier(name, args)
    if name in {"make", "cmake", "ninja", "gradle", "mvn", "dotnet", "swift", "deno", "go", "cargo"}:
        target = next((arg for arg in args if not arg.startswith("-")), "")
        if target in _SAFE_MAKE_TARGETS or target in _SAFE_CARGO or not target:
            return RiskLevel.LOW, f"{name} runs a verification target"
        return RiskLevel.MEDIUM, f"{name} runs project target {target}"
    return None


def _segment_tier(segment: str, settings: Settings) -> tuple[RiskLevel, str]:
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
            return RiskLevel.HIGH, "env lists the whole process environment (may expose credentials)"
    if tokens[0] in {"command", "builtin"}:
        tokens.pop(0)
        if tokens and tokens[0] in {"-v", "-V"}:
            return RiskLevel.LOW, "command lookup"
        if not tokens:
            return RiskLevel.LOW, "command lookup"
    if tokens[0] in {"sudo", "doas", "su", "eval", "exec"} | _PRIVILEGE_TOOLS:
        return RiskLevel.HIGH, f"{tokens[0]} escalates privileges"
    name = PurePath(tokens[0]).name
    args = tokens[1:]
    boundary = _boundary_reason(name, args, segment, settings)
    if boundary:
        return RiskLevel.HIGH, boundary
    if name == "printenv" and not args:
        return RiskLevel.HIGH, "printenv lists the whole process environment (may expose credentials)"
    if name in _SAFE_COMMANDS:
        return RiskLevel.LOW, f"{name} is read-only"
    if name in {"python", "python3", "node", "bash", "sh", "zsh", "perl", "ruby", "php"}:
        if _reinterpreted_safe(name, args):
            return RiskLevel.LOW, f"{name} read-only form"
        return _script_tier(name, args, settings)
    dev_tier = _dev_tool_tier(name, args, settings)
    if dev_tier is not None and not (
        name in _SAFE_REINTERPRETED and _reinterpreted_safe(name, args)
    ):
        return dev_tier
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
    if name in _DESTRUCTIVE_TOOLS:
        return RiskLevel.HIGH, f"{name} is destructive"
    # An unrecognised executable inside the workspace is reviewable, not
    # automatically a human approval: it runs with the shell tool's cwd and the
    # boundary checks above found no destructive/credential/outbound path.
    return RiskLevel.MEDIUM, f"unrecognised command ({name}) in the workspace; audit it"


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
    if has_redirect:
        for raw in _redirect_targets(text):
            name = PurePath(raw.strip("'\"")).name.lower()
            if name in _SELF_MODIFY_NAMES or name in _SECRET_FILE_NAMES or name.endswith(
                _SENSITIVE_SUFFIXES
            ):
                return RiskLevel.HIGH, f"redirection writes a protected file ({name})"
        if not _redirect_is_bounded(text, settings):
            return RiskLevel.HIGH, "redirection outside allowed roots"
    segments = _split_segments(text)
    decision: tuple[RiskLevel, str] = (RiskLevel.LOW, "read-only command line")
    for segment in segments:
        try:
            words = shlex.split(segment)
        except ValueError:
            words = []
        if words and PurePath(words[0]).name == "cd" and len(words) > 1:
            outside = _outside_path_reason(words[1:2], settings)
            if outside:
                return RiskLevel.HIGH, outside
        tier, reason = _segment_tier(segment, settings)
        if tier is RiskLevel.HIGH:
            return RiskLevel.HIGH, reason
        if tier is RiskLevel.MEDIUM:
            decision = (RiskLevel.MEDIUM, reason)
        elif decision[1] == "read-only command line":
            decision = (RiskLevel.LOW, reason)
    if has_redirect and decision[0] is RiskLevel.LOW:
        decision = (RiskLevel.MEDIUM, "bounded redirection")
    return decision


def card_reason(command: str) -> str:
    """Explain a stored command on an approval card without altering policy.

    Older approval rows carry only the generic "high-risk action" reason. The
    literal command shown on the card is enough to name a specific risk using
    the same classifier, so the operator can see what they are authorising.
    """
    text = str(command or "").strip()
    if not text:
        return "no command text was stored for this approval"
    try:
        return assess_command(text, Settings())[1]
    except Exception:  # noqa: BLE001 - card copy must never break presentation
        return "the command was classified high risk by the shell policy"
