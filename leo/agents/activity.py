"""Read-only ingest of local orchestration-agent transcripts into LEO's DB.

Agent-tool transcripts live outside the LEO tree (Claude Code writes them under
``/private/tmp/claude-501/...``).  This module discovers them, extracts a small,
sanitised projection of each agent's work and stores it durably so the live
dashboard can be rendered from SQLite alone.

Transcript contents are UNTRUSTED DATA.  They are parsed as JSON, never executed,
never treated as instructions, and every string that reaches the UI is sanitised
and redacted here.  No transcript path or raw transcript bytes are exposed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import db
from ..contracts import utcnow

def _default_transcript_root() -> Path:
    """Resolve the conventional transcript directory for the current project."""
    base = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path("/tmp")
    project = "-" + str(Path.cwd().resolve()).replace(os.sep, "-")
    return base / "claude-501" / project


DEFAULT_TRANSCRIPT_ROOT = str(_default_transcript_root())
TRANSCRIPT_GLOB = "a*.output"
MAX_TRANSCRIPT_BYTES = 128 * 1024
MAX_PROMPT_BYTES = 16 * 1024
MAX_FILES = 500
MAX_REPORT_CHARS = 40_000
MAX_PROMPT_CHARS = 600
MAX_ITEM_CHARS = 1_500
MAX_ITEMS = 60
# There is no PID or heartbeat in Claude Code task transcripts. Liveness can
# only be inferred from a very short mtime window, so the state is named
# ``recent`` rather than ``running``: it never claims a live process.
RECENT_SECONDS = 90

STATUSES = ("recent", "completed", "idle")
STATUS_BASIS_FINAL_REPORT = "final-report"
STATUS_BASIS_MTIME_ONLY = "mtime-only (no heartbeat or PID available)"
PROVENANCE_NOTE = (
    "Status is inferred from local transcript files. 'Recent' means the file "
    "changed in the last 90 seconds; it is not a live heartbeat. Transcripts "
    "are owner-verified and parsed read-only."
)

_REDACTORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|"
            r"auth[_-]?token|session[_-]?token|token|secret|password|passwd|pwd|"
            r"client[_-]?secret|private[_-]?key|credential)\b[\"']?\s*[:=]\s*)"
            r"([\"']?)([^\s\"',;]+)"
        ),
        "kv",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}"), "bearer"),
    (
        re.compile(
            r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{12,}|"
            r"xox[baprs]-[A-Za-z0-9-]{8,}|(?:AKIA|ASIA)[0-9A-Z]{16})\b"
        ),
        "prefix",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "jwt",
    ),
    (re.compile(r"(?i)\bhttps?://[^\s/@]+:[^\s/@]+@"), "url"),
)


def _redact(value: str) -> str:
    """Mask credential-shaped substrings without trying to interpret the text."""
    for pattern, kind in _REDACTORS:
        if kind == "kv":
            value = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]{m.group(2)}", value)
        elif kind == "bearer":
            value = pattern.sub("Bearer [redacted]", value)
        else:
            value = pattern.sub("[redacted]", value)
    return value


def _clean(value: Any, *, limit: int) -> str:
    """Coerce untrusted text into bounded, UTF-8 safe, redacted display text."""
    if not isinstance(value, str):
        return ""
    text = value.encode("utf-8", "replace").decode("utf-8", "replace")
    text = "".join(char for char in text if char in "\n\t\r" or ord(char) >= 32)
    text = _redact(text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …"
    return text


_STRUCTURED_FIELD = re.compile(
    r"(?im)^\s*(?:#{1,4}\s*|\*{0,2})"
    r"(?:findings|gaps|issues|fixes|fix[_\s]\d+|test[_\s]?result|blocker|"
    r"files[_\s](?:changed|created|modified|added)|summary)\b\s*[:\[]"
)
_SECTION_HEADER = re.compile(
    r"(?i)^\s*(?:#{1,4}\s*|\*{0,2})"
    r"(findings|gaps|issues|fixes|fix[_\s]\d+|blockers?|tests?)\b"
    r"\s*[:：-]?\s*(.*)$"
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
_SCALAR_FIELD = re.compile(
    r"(?im)^\s*(?:#{1,4}\s*|\*{0,2})(test[_-]?result|test_command|tests|blocker)"
    r"\b\s*[:：]\s*(.*)$"
)
_INLINE_FIELD = re.compile(
    r"(?i)^\s*(?:#{1,4}\s*|\*{0,2})(test[_-]?result|test_command|tests|blocker)\b\s*[:：]\s*(.*)$"
)
_OTHER_FIELD = re.compile(
    r"(?i)^\s*(?:#{1,4}\s*|\*{0,2})"
    r"(files[_\s](?:changed|created|modified|added)|agent[_-]?id|handler|"
    r"verified[_\s]\w+|grants[_\s]\w+|report|scope)\b\s*[:：]"
)
_BARE_FIX = re.compile(r"(?i)^\s*fix[_\s]\d+\s*[:：]\s*(.+)$")


def _is_final_report(text: str) -> bool:
    """Recognise a structured final report without executing or trusting it."""
    if _STRUCTURED_FIELD.search(text):
        return True
    if re.search(r'"(?:findings|gaps|fixes|test_result|blocker)"\s*:', text):
        return True
    if text.count("\n## ") >= 2 and len(text) >= 600:
        return True
    return len(text) >= 1_500


def _add_item(items: list[str], value: str) -> None:
    cleaned = _clean(value, limit=MAX_ITEM_CHARS)
    if cleaned and cleaned.lower() not in {"none", "n/a", "no", "-"} and cleaned not in items:
        if len(items) < MAX_ITEMS:
            items.append(cleaned)


def _json_item(value: Any, keys: tuple[str, ...]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item
    return ""


def _extract_json_report(text: str) -> dict[str, Any]:
    """Recognise JSON-shaped reports (common for audit subagents)."""
    result: dict[str, Any] = {"findings": [], "fixes": [], "test_result": "", "blocker": ""}
    decoder = json.JSONDecoder()
    for match in list(re.finditer(r"\{", text))[:12]:
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except (ValueError, RecursionError):
            continue
        if not isinstance(obj, dict):
            continue
        fields = {str(key).lower(): value for key, value in obj.items()}
        if not any(key in fields for key in ("findings", "gaps", "fixes", "test_result", "blocker")):
            continue
        for key in ("findings", "gaps", "issues"):
            values = fields.get(key)
            if isinstance(values, list):
                for value in values:
                    _add_item(result["findings"], _json_item(
                        value, ("title", "summary", "finding", "description", "evidence", "id")
                    ))
        values = fields.get("fixes")
        if isinstance(values, list):
            for value in values:
                _add_item(result["fixes"], _json_item(
                    value, ("summary", "description", "detail", "title", "fix", "id")
                ))
        for key in ("test_result", "tests"):
            value = fields.get(key)
            if isinstance(value, str) and not result["test_result"]:
                result["test_result"] = _clean(value, limit=400)
        if isinstance(fields.get("blocker"), str) and not result["blocker"]:
            blocker = _clean(fields["blocker"], limit=400)
            if blocker.lower() not in {"none", "n/a"}:
                result["blocker"] = blocker
    return result


def _extract_report(text: str) -> dict[str, Any]:
    """Split a final report into findings, fixes, test result and blocker."""
    findings: list[str] = []
    fixes: list[str] = []
    test_result = ""
    blocker = ""
    section = ""
    pending_continuation = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            pending_continuation = False
            continue

        # Scalars always win over section context: ``blocker:`` is not a fix.
        field = _INLINE_FIELD.match(line)
        if field:
            value = _clean(field.group(2), limit=400)
            name = re.sub(r"[\s_-]+", "", field.group(1).lower())
            if name.startswith("blocker"):
                blocker = blocker or ("" if value.lower() in {"none", "n/a"} else value)
            else:
                test_result = test_result or value
            pending_continuation = False
            continue
        if _OTHER_FIELD.match(line):
            section = ""
            pending_continuation = False
            continue

        header = _SECTION_HEADER.match(line)
        if header:
            name = re.sub(r"[\s_]+", "", header.group(1).lower())
            if name in {"finding", "findings", "gap", "gaps", "issue", "issues"}:
                section = "findings"
            elif name.startswith("fix"):
                section = "fixes"
            else:
                section = ""
            inline = header.group(2).strip().strip("[]() ")
            if inline and section:
                _add_item(findings if section == "findings" else fixes, inline)
            pending_continuation = bool(section)
            continue

        bullet = _BULLET.match(line)
        if bullet:
            if section:
                _add_item(findings if section == "findings" else fixes, bullet.group(1))
            else:
                bare = _BARE_FIX.match(line)
                if bare:
                    _add_item(fixes, bare.group(1))
            pending_continuation = bool(section)
            continue
        if section and pending_continuation and len(line) < MAX_ITEM_CHARS:
            _add_item(findings if section == "findings" else fixes, line)
            continue
        bare = _BARE_FIX.match(line)
        if bare:
            _add_item(fixes, bare.group(1))

    fields = _SCALAR_FIELD.search(text)
    if fields:
        value = _clean(fields.group(2), limit=400)
        if re.sub(r"[\s_-]+", "", fields.group(1).lower()).startswith("blocker"):
            blocker = blocker or ("" if value.lower() in {"none", "n/a"} else value)
        else:
            test_result = test_result or value
    for match in re.finditer(r"(?im)^\s*(?:#{1,4}\s*|\*{0,2})blocker\b\s*[:：]\s*(.*)$", text):
        value = _clean(match.group(1), limit=400)
        if value and value.lower() not in {"none", "n/a"}:
            blocker = blocker or value
            break
    json_report = _extract_json_report(text)
    for item in json_report["findings"]:
        _add_item(findings, item)
    for item in json_report["fixes"]:
        _add_item(fixes, item)
    test_result = test_result or json_report["test_result"]
    blocker = blocker or json_report["blocker"]
    return {
        "findings": findings,
        "fixes": fixes,
        "test_result": test_result,
        "blocker": blocker,
        "report_text": _clean(text, limit=MAX_REPORT_CHARS),
    }


_CATEGORY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bread[\s-]?only\b|\bverification\b"), "Verify"),
    (re.compile(r"(?i)\b(audit(?:ing)?|security review|compliance)\b"), "Audit"),
    (re.compile(r"(?i)\b(fix(?:ing)?|repair(?:ing)?|harden(?:ing)?|patch(?:ing)?|bug|regression)\b"), "Fix"),
    (re.compile(r"(?i)\b(verif(?:y|ying)|test(?:ing)?|diagnos(?:e|ing)|investigat(?:e|ing))\b"), "Verify"),
    (re.compile(r"(?i)\b(research(?:ing)?|source-level research)\b"), "Research"),
    (re.compile(r"(?i)\b(review(?:ing)?)\b"), "Review"),
    (re.compile(r"(?i)\b(implement(?:ing)?|build(?:ing)?|creat(?:e|ing)|writ(?:e|ing))\b"), "Build"),
)

_SUBJECT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)browser|playwright|webdriver"), "browser operator"),
    (re.compile(r"(?i)memory|knowledge|recall"), "memory control"),
    (re.compile(r"(?i)task|scheduler|schedule"), "task engine"),
    (re.compile(r"(?i)approval|permission|gate"), "approvals"),
    (re.compile(r"(?i)attach(?:ment)?|upload"), "attachments"),
    (re.compile(r"(?i)skill|compatibility"), "skills"),
    (re.compile(r"(?i)integration|gateway|channel|enrollment"), "integrations"),
    (re.compile(r"(?i)security|credential|secret|harden"), "security"),
    (re.compile(r"(?i)planner|planning|decompos"), "planner"),
    (re.compile(r"(?i)evaluat|metrics|verification"), "evaluation"),
    (re.compile(r"(?i)release|audit"), "release audit"),
    (re.compile(r"(?i)storage|database|sqlite|db\.py"), "storage"),
    (re.compile(r"(?i)cli|command line"), "cli"),
    (re.compile(r"(?i)\bapi\b|endpoint|route"), "api"),
)


def _subject_from_text(text: str) -> str:
    for pattern, subject in _SUBJECT_RULES:
        if pattern.search(text):
            return subject
    return ""


def _derive_role(prompt: str, agent_id: str) -> str:
    """Derive a stable human-readable label such as ``Tools: memory control``."""
    owned = re.search(r"(?is)owned\s+files?\s*:?\s*(.{0,600})", prompt)
    owned_text = _subject_from_text(owned.group(1)) if owned else ""

    explicit = re.search(r"(?im)^\s*(?:agent[_-]?id|agent)\s*[:：]\s*([A-Za-z0-9][\w-]{1,40})\s*$", prompt)
    agent_label = _subject_from_text(explicit.group(1)) if explicit else ""
    if agent_label and not owned_text:
        return f"Agent: {agent_label}"

    if owned:
        owned_paths = owned.group(1)
        category = "Agent"
        if re.search(r"(?i)(?:^|/|\s)tools?[/\s_]|tool_", owned_paths):
            category = "Tools"
        elif re.search(r"(?i)/tests?/|test_", owned_paths):
            category = "Tests"
        elif re.search(r"(?i)/specs?/", owned_paths):
            category = "Agent"
        subject = owned_text
        if not subject:
            stem = re.search(r"([\w-]+)\.py", owned_paths)
            subject = stem.group(1).replace("_", " ").replace("-", " ") if stem else ""
        return f"{category}: {subject[:48].strip() or agent_id or 'worker'}"

    category = "Agent"
    for pattern, label in _CATEGORY_RULES:
        if pattern.search(prompt):
            category = label
            break

    subject = _subject_from_text(prompt)
    if not subject:
        named = re.search(r"(?i)\byou are (?:the |an? )?([\w-]{3,40})\s+agent\b", prompt)
        if named:
            subject = _subject_from_text(named.group(1)) or named.group(1).replace("-", " ")
    if not subject:
        action = re.search(
            r"(?i)\b(?:audit(?:ing)?|review(?:ing)?|diagnos(?:e|ing)|"
            r"investigat(?:e|ing)|research(?:ing)?|fix(?:ing)?|build(?:ing)?|"
            r"implement(?:ing)?|verif(?:y|ying)|test(?:ing)?)\s+"
            r"(?:(?:the|an?|new)\s+)?([A-Za-z][\w /-]{2,70}?)(?:[.,;:\n]|\s+at\s+|\s+in\s+)",
            prompt,
        )
        if action:
            candidate = action.group(1).strip()
            subject = _subject_from_text(candidate) or candidate
    if not subject and explicit:
        subject = explicit.group(1)
    subject = subject.strip(" \t-:/")[:48].strip() or agent_id or "worker"
    return f"{category}: {subject}"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _event_text(event: Any) -> str:
    if not isinstance(event, dict):
        return ""
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    return _content_text(message.get("content"))


def _iso_timestamp(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 64:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _mtime_iso(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def transcript_root() -> Path:
    configured = os.getenv("LEO_AGENT_TRANSCRIPT_DIR", "").strip()
    return Path(configured or DEFAULT_TRANSCRIPT_ROOT).expanduser().resolve()


def _claude_projects_root() -> Path:
    return (Path.home() / ".claude" / "projects").resolve()


def _current_uid() -> int | None:
    getter = getattr(os, "geteuid", None)
    return int(getter()) if getter is not None else None


def _owner_trusted(path: Path, *, check_mode: bool = True, follow_symlinks: bool = True) -> bool:
    """Fail closed on foreign or group/world-writable transcript material."""
    uid = _current_uid()
    if uid is None:
        return False
    try:
        stat = path.stat() if follow_symlinks else path.lstat()
    except (OSError, ValueError):
        return False
    if stat.st_uid != uid:
        return False
    if check_mode and follow_symlinks and stat.st_mode & 0o022:
        return False
    return True


def _contained_candidate(path: Path, root: Path) -> tuple[Path, str] | None:
    """Accept only owner-verified entries lexically inside root.

    Claude Code stores the ``a*.output`` entries as symlinks to its canonical
    ``~/.claude/projects/<session>/subagents/agent-*.jsonl`` files. Those are
    explicitly allowed when both link and target are owned by the current
    user; every other symlink escape is refused. The resolved target path is
    never stored, logged or returned to the UI.
    """
    lexical = Path(os.path.abspath(os.path.normpath(str(path))))
    if not lexical.is_relative_to(root):
        return None
    if path.is_symlink():
        if not _owner_trusted(path, check_mode=False, follow_symlinks=False):
            return None
        target = path.resolve()
        claude_projects = _claude_projects_root()
        if (
            target.parent.name != "subagents"
            or not re.fullmatch(r"agent-[\w-]+\.jsonl", target.name)
            or not target.is_relative_to(claude_projects)
            or not target.is_file()
            or not _owner_trusted(target)
            or not _owner_trusted(target.parent)
            or not _owner_trusted(claude_projects)
        ):
            return None
        return lexical, "claude-code-symlink"
    if not path.is_file() or not _owner_trusted(path):
        return None
    return lexical, "local-transcript"


def _discover(root: Path) -> list[tuple[Path, str]]:
    """Find owner-verified transcript-shaped files strictly inside root."""
    if not root.is_dir() or not _owner_trusted(root):
        return []
    found: dict[Path, str] = {}
    try:
        for base, dirs, names in os.walk(root, followlinks=False):
            if not _owner_trusted(Path(base)):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
            for name in names:
                if not re.fullmatch(r"a[\w.-]*\.output", name):
                    continue
                candidate = _contained_candidate(Path(base) / name, root)
                if candidate is not None:
                    path, source_kind = candidate
                    found[path] = source_kind
                    if len(found) >= MAX_FILES:
                        return sorted(found.items())
    except (OSError, PermissionError):
        return sorted(found.items())
    return sorted(found.items())


def _parse_transcript(path: Path, fingerprint: str, source_kind: str) -> dict[str, Any] | None:
    """Parse the tail of one transcript. Never executes transcript content."""
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            if stat.st_size > MAX_TRANSCRIPT_BYTES:
                # The prompt is at the start of the file; the report is at the
                # end. Stay inside one 128 KiB read budget by taking both ends.
                head = handle.read(MAX_PROMPT_BYTES)
                handle.seek(stat.st_size - (MAX_TRANSCRIPT_BYTES - MAX_PROMPT_BYTES))
                tail = handle.read(MAX_TRANSCRIPT_BYTES - MAX_PROMPT_BYTES)
            else:
                head = b""
                tail = handle.read(MAX_TRANSCRIPT_BYTES)
    except (OSError, ValueError):
        return None

    head_lines = head.decode("utf-8", "replace").splitlines()
    if head_lines:
        head_lines = head_lines[:-1]  # may end mid-line
    tail_lines = tail.decode("utf-8", "replace").splitlines()
    if stat.st_size > MAX_TRANSCRIPT_BYTES and tail_lines:
        tail_lines = tail_lines[1:]  # may begin mid-line

    events: list[dict[str, Any]] = []
    for line in [*head_lines, *tail_lines]:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError, RecursionError):
            continue
        if isinstance(event, dict):
            events.append(event)
    if not events:
        return None

    prompt = ""
    for event in events:
        if event.get("type") == "user":
            prompt = _event_text(event)
            if prompt:
                break
    agent_id = ""
    for event in events:
        candidate = event.get("agentId")
        if isinstance(candidate, str) and candidate.strip():
            agent_id = candidate.strip()[:120]
            break
    agent_id = agent_id or path.stem[:120]

    final_text = ""
    final_at = ""
    for event in events:
        if event.get("type") != "assistant":
            continue
        text = _event_text(event)
        if text.strip() and _is_final_report(text):
            final_text = text
            final_at = _iso_timestamp(event.get("timestamp"))
    report = _extract_report(final_text) if final_text else {
        "findings": [], "fixes": [], "test_result": "", "blocker": "", "report_text": "",
    }

    started = ""
    last_seen = ""
    for event in events:
        stamp = _iso_timestamp(event.get("timestamp"))
        if stamp:
            if not started:
                started = stamp
            last_seen = stamp
    last_seen = last_seen or _mtime_iso(stat.st_mtime)

    if final_text:
        status = "completed"
        basis = STATUS_BASIS_FINAL_REPORT
    elif time.time() - stat.st_mtime <= RECENT_SECONDS:
        status = "recent"
        basis = STATUS_BASIS_MTIME_ONLY
    else:
        status = "idle"
        basis = STATUS_BASIS_MTIME_ONLY

    return {
        "agent_id": agent_id,
        "role": _derive_role(prompt, agent_id),
        "status": status,
        "status_basis": basis,
        "source_kind": source_kind,
        "trust": "owner-verified",
        "started_at": started,
        "finished_at": final_at or (last_seen if final_text else ""),
        "last_activity_at": last_seen,
        "prompt_excerpt": _clean(prompt, limit=MAX_PROMPT_CHARS),
        "findings": report["findings"],
        "fixes": report["fixes"],
        "test_result": report["test_result"],
        "blocker": report["blocker"],
        "report_text": report["report_text"],
        "fingerprint": fingerprint,
    }


_TABLE = "agent_activity"
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
  agent_id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  status TEXT NOT NULL,
  status_basis TEXT NOT NULL DEFAULT '',
  source_kind TEXT NOT NULL DEFAULT '',
  trust TEXT NOT NULL DEFAULT '',
  started_at TEXT,
  finished_at TEXT,
  last_activity_at TEXT,
  prompt_excerpt TEXT NOT NULL DEFAULT '',
  findings TEXT NOT NULL DEFAULT '[]',
  fixes TEXT NOT NULL DEFAULT '[]',
  test_result TEXT NOT NULL DEFAULT '',
  blocker TEXT NOT NULL DEFAULT '',
  report_text TEXT NOT NULL DEFAULT '',
  fingerprint TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_activity_status ON {_TABLE}(status, last_activity_at);
CREATE INDEX IF NOT EXISTS idx_agent_activity_role ON {_TABLE}(role);
"""
_sync_lock = asyncio.Lock()
_cache: dict[str, dict[str, Any]] = {}


async def _ensure_table() -> None:
    for statement in (s.strip() for s in _SCHEMA.split(";") if s.strip()):
        await db.execute(statement)
    # Additive migration for databases created by the earlier dashboard build.
    columns = {row["name"] for row in await db.fetchall(f"PRAGMA table_info({_TABLE})")}
    for name, declaration in (
        ("status_basis", "TEXT NOT NULL DEFAULT ''"),
        ("source_kind", "TEXT NOT NULL DEFAULT ''"),
        ("trust", "TEXT NOT NULL DEFAULT ''"),
    ):
        if name not in columns:
            await db.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {name} {declaration}")


def _fingerprint(stat: os.stat_result) -> str:
    return f"{stat.st_mtime_ns}:{stat.st_size}"


async def sync_agent_activity() -> dict[str, Any]:
    """Refresh the durable projection from local transcripts (idempotent)."""
    await _ensure_table()
    root = transcript_root()
    if not root.is_dir() or not _owner_trusted(root):
        return {
            "discovered": 0, "upserted": 0, "unchanged": 0,
            "removed": 0, "trusted": False,
        }

    paths = _discover(root)
    async with _sync_lock:
        parsed: list[dict[str, Any]] = []
        unchanged = 0
        fingerprints: dict[str, str] = {}
        for path, source_kind in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            fingerprint = _fingerprint(stat)
            key = str(path)
            cached = _cache.get(key)
            if cached and cached.get("fingerprint") == fingerprint:
                record = dict(cached)
                unchanged += 1
            else:
                record = await asyncio.to_thread(_parse_transcript, path, fingerprint, source_kind)
                if record:
                    _cache[key] = dict(record)
            if record:
                fingerprints[record["agent_id"]] = fingerprint
                parsed.append(record)

        # A transcript can contain several agent ids; keep one row per agent.
        deduped: dict[str, dict[str, Any]] = {}
        for record in parsed:
            deduped[record["agent_id"]] = record

        now = utcnow()
        async with db.transaction():
            existing = {
                row["agent_id"]: row["fingerprint"]
                for row in await db.fetchall(f"SELECT agent_id, fingerprint FROM {_TABLE}")
            }
            upserted = 0
            for agent_id, record in deduped.items():
                if existing.get(agent_id) == fingerprints[agent_id]:
                    continue
                await db.execute(
                    f"""INSERT INTO {_TABLE}
                        (agent_id, role, status, status_basis, source_kind, trust,
                         started_at, finished_at, last_activity_at,
                         prompt_excerpt, findings, fixes, test_result, blocker, report_text,
                         fingerprint, updated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(agent_id) DO UPDATE SET
                          role=excluded.role, status=excluded.status,
                          status_basis=excluded.status_basis,
                          source_kind=excluded.source_kind, trust=excluded.trust,
                          started_at=excluded.started_at, finished_at=excluded.finished_at,
                          last_activity_at=excluded.last_activity_at,
                          prompt_excerpt=excluded.prompt_excerpt, findings=excluded.findings,
                          fixes=excluded.fixes, test_result=excluded.test_result,
                          blocker=excluded.blocker, report_text=excluded.report_text,
                          fingerprint=excluded.fingerprint, updated_at=excluded.updated_at""",
                    (
                        agent_id,
                        record["role"],
                        record["status"],
                        record["status_basis"],
                        record["source_kind"],
                        record["trust"],
                        record["started_at"],
                        record["finished_at"],
                        record["last_activity_at"],
                        record["prompt_excerpt"],
                        json.dumps(record["findings"], ensure_ascii=False),
                        json.dumps(record["fixes"], ensure_ascii=False),
                        record["test_result"],
                        record["blocker"],
                        record["report_text"],
                        record["fingerprint"],
                        now,
                    ),
                )
                upserted += 1
            stale = [key for key in existing if key not in deduped]
            if stale:
                await db.executemany(
                    f"DELETE FROM {_TABLE} WHERE agent_id = ?", [(key,) for key in stale]
                )
            for key in stale:
                _cache.pop(key, None)
        return {
            "discovered": len(paths),
            "upserted": upserted,
            "unchanged": unchanged,
            "removed": len(stale),
            "trusted": True,
        }


def _row_to_record(row: Any) -> dict[str, Any]:
    record = dict(row)
    for field in ("findings", "fixes"):
        try:
            value = json.loads(record.get(field) or "[]")
            record[field] = value if isinstance(value, list) else []
        except (ValueError, TypeError):
            record[field] = []
    return {
        "agent_id": record.get("agent_id", ""),
        "role": record.get("role", ""),
        "status": record.get("status", "idle"),
        "status_basis": record.get("status_basis") or STATUS_BASIS_MTIME_ONLY,
        "source_kind": record.get("source_kind") or "unknown",
        "trust": record.get("trust") or "unverified",
        "started_at": record.get("started_at") or "",
        "finished_at": record.get("finished_at") or "",
        "last_activity_at": record.get("last_activity_at") or "",
        "prompt_excerpt": record.get("prompt_excerpt") or "",
        "findings": record["findings"],
        "fixes": record["fixes"],
        "test_result": record.get("test_result") or "",
        "blocker": record.get("blocker") or "",
        "report_text": record.get("report_text") or "",
    }


def _like(value: str) -> str:
    return "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


async def list_activity(
    *,
    status: str | None = None,
    role_contains: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Filtered, bounded view of the activity table."""
    await _ensure_table()
    clauses: list[str] = []
    params: list[Any] = []
    if status == "all":
        status = None
    if status:
        clauses.append("status = ?")
        params.append(status)
    if role_contains:
        clauses.append("LOWER(role) LIKE ? ESCAPE '\\'")
        params.append(_like(role_contains.lower()))
    if q:
        query = _like(q.lower())
        clauses.append(
            "(LOWER(role) LIKE ? ESCAPE '\\' OR LOWER(prompt_excerpt) LIKE ? ESCAPE '\\' "
            "OR LOWER(findings) LIKE ? ESCAPE '\\' OR LOWER(fixes) LIKE ? ESCAPE '\\' "
            "OR LOWER(test_result) LIKE ? ESCAPE '\\' OR LOWER(blocker) LIKE ? ESCAPE '\\')"
        )
        params.extend([query] * 6)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    total = await db.fetchone(f"SELECT COUNT(*) AS n FROM {_TABLE}{where}", params)
    rows = await db.fetchall(
        f"SELECT * FROM {_TABLE}{where} "
        "ORDER BY (status = 'recent') DESC, last_activity_at DESC, agent_id ASC "
        "LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    all_roles = await db.fetchall(f"SELECT role FROM {_TABLE}")
    role_counts: dict[str, int] = {}
    for row in all_roles:
        role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
    return {
        "agents": [_row_to_record(row) for row in rows],
        "total": int(total["n"]) if total else 0,
        "limit": limit,
        "offset": offset,
        "roles": sorted(role_counts),
        "role_counts": role_counts,
        "provenance_note": PROVENANCE_NOTE,
        "recent_window_seconds": RECENT_SECONDS,
    }


async def activity_summary() -> dict[str, Any]:
    await _ensure_table()
    rows = await db.fetchall(f"SELECT status, findings, fixes FROM {_TABLE}")
    counts = {status: 0 for status in STATUSES}
    findings = fixes = 0
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        for field, key in (("findings", "findings"), ("fixes", "fixes")):
            try:
                value = json.loads(row[field] or "[]")
            except (ValueError, TypeError):
                value = []
            if isinstance(value, list):
                if key == "findings":
                    findings += len(value)
                else:
                    fixes += len(value)
    return {
        "recent": counts.get("recent", 0),
        "completed": counts.get("completed", 0),
        "idle": counts.get("idle", 0),
        "total": len(rows),
        "total_findings": findings,
        "total_fixes": fixes,
        "provenance_note": PROVENANCE_NOTE,
        "recent_window_seconds": RECENT_SECONDS,
        "updated_at": utcnow(),
    }
