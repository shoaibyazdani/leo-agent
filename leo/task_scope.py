"""Per-turn task scoping: tools, skill recommendations and cap reports.

Nothing here calls a model or changes a provider. The classifier is a small,
deterministic keyword pass over the user's own words; the tool sets are
allowlists of already-registered tool names. The chat turn advertises only the
selected subset, which keeps the full catalogue off the wire for ordinary
replies without changing how the model provider builds requests.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

BUDGET_EXHAUSTED_TEXT = "Tool budget exhausted before a final answer"

# Scope names are stable labels, not model-visible instructions.
CHAT = "chat"
UTILITY = "utility"
READ = "read"
SHELL = "shell"
EDIT = "edit"
WEB = "web"
OPS = "ops"
GENERAL = "general"

_UTILITY_TOOLS = frozenset({
    "calculator", "current_time", "json_parse", "token_estimate",
    "text_transform", "base64_codec", "csv_parse",
})

_READ_TOOLS = _UTILITY_TOOLS | frozenset({
    "read_file", "list_files", "search_files", "dir_tree", "file_hash",
    "file_diff", "git_status", "git_diff", "git_log", "git_show",
    "http_get", "http_json", "http_json_paths",
    "memory_search", "memory_list", "task_list", "task_status",
    "task_progress", "task_result", "run_list", "run_status", "run_log",
    "conversation_messages", "conversation_export", "config_get",
    "config_sources", "app_health", "context_stats", "context_preview",
    "skill_list", "skill_search", "skill_read", "model_providers",
    "model_usage", "metrics_snapshot", "audit_query", "event_query",
})

_FILE_TOOLS = frozenset({
    "read_file", "write_file", "list_files", "search_files", "dir_tree",
    "file_copy", "file_move", "file_hash", "file_diff", "patch",
})

_SHELL_TOOLS = frozenset({
    "shell", "terminal", "calculator", "current_time", "token_estimate",
    "git_status", "git_diff", "git_log", "git_show",
}) | _FILE_TOOLS

_WEB_TOOLS = _UTILITY_TOOLS | frozenset({
    "http_get", "http_json", "http_json_paths", "search_all",
    "search_recent_events", "browser_navigate", "browser_extract_text",
    "browser_screenshot", "browser_click", "browser_fill", "vision_analyze",
    "preview_local_file", "preview_close",
})

_OPS_TOOLS = _UTILITY_TOOLS | frozenset({
    "task_create", "task_list", "task_status", "task_progress", "task_cancel",
    "task_result", "run_list", "run_status", "run_log", "run_cancel",
    "schedule_create", "schedule_list", "schedule_get", "schedule_update",
    "schedule_pause", "schedule_resume", "schedule_delete", "schedule_run_now",
    "schedule_cancel", "memory_add", "memory_search", "memory_list",
    "memory_forget", "agent_list", "agent_status", "agent_result",
    "skill_list", "skill_search", "skill_read", "skill_invoke",
    "conversation_messages", "config_get", "app_health", "current_time",
})

# Broad but still curated. Deliberately omits operator-only/state-changing
# surfaces (agents, channel sends, model/attachment/webhook administration and
# skill authoring) so an unclassified task never advertises the whole toolbox.
_GENERAL_TOOLS = (_READ_TOOLS | _SHELL_TOOLS | _WEB_TOOLS | _OPS_TOOLS) - frozenset({
    "agent_delegate", "channel_send", "channel_react", "model_add",
    "skill_manage", "attachment_add", "attachment_remove",
})

SCOPE_TOOLS: dict[str, frozenset[str]] = {
    CHAT: frozenset(),
    UTILITY: _UTILITY_TOOLS,
    READ: _READ_TOOLS,
    SHELL: _SHELL_TOOLS,
    EDIT: _SHELL_TOOLS | frozenset({"preview_local_file", "preview_close"}),
    WEB: _WEB_TOOLS,
    OPS: _OPS_TOOLS,
    GENERAL: _GENERAL_TOOLS,
}

# Order decides priority when a request mixes several kinds of work. Shell and
# edit requests must not silently lose their write/shell tools to a generic
# "read" or "check" keyword.
_SCOPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SHELL, (
        "shell", "terminal", "bash", "zsh", "command line", "run command",
        "execute command", "pytest", "pip install", "grep", "chmod", "mkdir",
    )),
    (EDIT, (
        "write", "edit", "patch", "create a file", "create file", "modify",
        "overwrite", "refactor", "implement", "rename file", "delete file",
        "remove file", "save the file", "fix the bug", "fix bug", "code change",
    )),
    (WEB, (
        "http", "https", "url", "website", "web page", "webpage", "browse",
        "browser", "online search", "web search", "fetch the page",
    )),
    (OPS, (
        "schedule", "reminder", "task ", "task list", "job", "run the task",
        "agent", "remember", "memory", "cancel the run", "run status",
    )),
    (READ, (
        "read", "show", "list", "display", "inspect", "explain", "find",
        "search", "check", "compare", "summarize", "summary", "what does",
        "look at", "review", "git status", "git log", "git diff",
    )),
    (UTILITY, (
        "calculate", "calculating", "compute", "what is", "times", "current time", "date",
        "convert", "json", "encode", "decode",
    )),
)

# Purely conversational turns get no tool specs at all. Keep this list short:
# anything with a task verb is classified by the table above instead.
_CHAT_ONLY = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|good (?:morning|evening|afternoon)|"
    r"how are you|nice|ok|okay|yes|no|sure|bye|goodbye)[!. ]*$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip() + " "


def classify_tool_scope(text: str) -> str:
    """Choose one of :data:`SCOPE_TOOLS` from the user's words only.

    No keyword hit (including a bare greeting) means a no-tool conversation
    turn. An unrecognized but clearly non-trivial request falls back to the
    curated GENERAL scope, never to the full registry.
    """
    value = _normalize(text)
    if not value.strip() or _CHAT_ONLY.fullmatch(value.strip()):
        return CHAT
    for scope, keywords in _SCOPE_KEYWORDS:
        for keyword in keywords:
            if keyword in value:
                return scope
    # A longer request is doing something even without a known keyword; a very
    # short one is a plain conversational reply.
    if len(value.split()) >= 8:
        return GENERAL
    return CHAT


def select_tool_specs(
    specs: Iterable[Any], scope: str
) -> list[dict[str, Any]]:
    """Dump only the specs allowlisted for ``scope``, in registry order."""
    allowed = SCOPE_TOOLS.get(scope, frozenset())
    return [
        spec.model_dump(mode="json") for spec in specs
        if getattr(spec, "name", None) in allowed
    ]


# --- skill recommendation ---------------------------------------------------

_SKILL_STOPWORDS = frozenset({
    "a", "an", "and", "are", "can", "do", "for", "from", "how", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "show", "the", "this",
    "to", "use", "using", "what", "with", "you", "your",
})


def _skill_tokens(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    normalized: set[str] = set()
    for token in tokens:
        if token in _SKILL_STOPWORDS or len(token) < 3:
            continue
        normalized.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return normalized


def match_skill(text: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Match a user request against the *index* entries, not full documents.

    Scoring is local and deterministic: whole name/description phrases and
    distinct token overlaps. A weak single-word coincidence is not a match,
    so a plain reply does not pick up an unrelated skill.
    """
    query = str(text or "").lower()
    query_tokens = _skill_tokens(query)
    if not query_tokens:
        return None
    best: dict[str, Any] | None = None
    best_score = 0.0
    for entry in entries:
        name = str(entry.get("name") or "")
        description = str(entry.get("description") or "")
        category = str(entry.get("category") or "")
        name_tokens = _skill_tokens(name.replace("_", " ").replace("-", " "))
        description_tokens = _skill_tokens(description)
        category_tokens = _skill_tokens(category)
        name_hits = query_tokens & name_tokens
        description_hits = query_tokens & description_tokens
        score = 4.0 * len(name_hits) + 1.0 * len(description_hits)
        if name.replace("_", " ") in query:
            score += 6.0
        if description and description.lower() in query:
            score += 2.0
        if category_tokens & query_tokens:
            score += 0.5
        if score < 3.0 or (not name_hits and len(description_hits) < 3):
            continue
        if score > best_score or (score == best_score and best is not None
                                  and name < str(best.get("name") or "")):
            best, best_score = dict(entry), score
    return best


# --- cap reporting ----------------------------------------------------------

_CAP_NEXT_OPTIONS = (
    "Next options: continue (say 'continue' to resume this work), narrow the "
    "task (one file, one command or a smaller range), or split it into smaller turns."
)


def cap_report(
    *, reason: str, done: Sequence[str], unfinished: Sequence[str],
    unfinished_label: str = "pending tool calls",
) -> str:
    """A bounded progress handoff for a turn stopped by a hard cap."""
    operations = list(dict.fromkeys(str(name) for name in done if str(name)))
    pending = list(dict.fromkeys(str(name) for name in unfinished if str(name)))
    if operations:
        summary = f"{len(operations)} operation(s): " + ", ".join(operations[:20])
        if len(operations) > 20:
            summary += f" (+{len(operations) - 20} more)"
    else:
        summary = "no tool operations completed"
    lines = [
        f"{BUDGET_EXHAUSTED_TEXT} ({reason}).",
        "Progress report:",
        f"- Done so far: {summary}.",
        "- Not finished: " + (
            f"{unfinished_label}: " + ", ".join(pending[:20]) if pending
            else "the requested work was not completed"
        ) + ".",
        _CAP_NEXT_OPTIONS,
    ]
    return "\n".join(lines)


# --- token accounting -------------------------------------------------------

def estimate_tool_spec_tokens(tools: Sequence[Mapping[str, Any]] | None) -> int:
    """Cheap, explicit chars/4 estimate of the advertised tool-spec payload.

    Provider token counts remain authoritative when reported; this number is
    recorded beside them so saved input from scoping can be measured.
    """
    if not tools:
        return 0
    serialized = json.dumps(
        list(tools), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return int(math.ceil(len(serialized) / 4.0))
