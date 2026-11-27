#!/usr/bin/env python3
"""Chat with a running LEO server like a human, and observe it honestly.

This harness sends a deterministic, varied, human-like conversation to
``POST /chat`` with ONE fixed conversation id. By default it sends at least 400
user messages in that single conversation (the hard floor), one at a time, with
a per-request timeout and a bounded number of retries. Every turn is recorded,
including failed ones; a single failure never aborts the run.

Outputs (under ``artifacts/observation/`` by default):

* ``conversation.jsonl`` -- one machine-readable JSON object per turn.
* ``conversation.md`` -- human-readable transcript with a summary header.
* ``summary.json`` -- the structured run summary (also printed to stdout).

Safety properties:

* No environment variable is ever read for credentials or logged. The script
  only uses its CLI arguments; it does not consult ``os.environ`` at all.
* Credentials appearing in prompts or responses are redacted in every output:
  a key-aware scanner recognizes sensitive key names (case/separator tolerant:
  ``authToken``, ``client_secret``, ``API key``, ``PIN``, ``Cookie``,
  ``session_id``, ``authorization``, ...) and consumes the *whole* credential
  span after them — repeated/mixed operator tokens, nested mixed wrappers with
  interior whitespace, brace/backtick/escaped wrappers, escaped or leading
  list separators, nested keyed values and whitespace-only separators — before
  stopping at the next keyed assignment or clause boundary. Scanning resumes
  at the end of each replaced span and repeats to a fixed point, so repeated
  multiword assignments (`API key: x API key: y`) cannot leave a second value
  behind. Escapes are normalized to a fixed point before embedded JSON is
  re-parsed structurally,
  so odd escape runs (including a dangling backslash before a decoded ``=`` or
  an escape in place of ``:``) cannot hide a key. Whole auth schemes in any
  letter case (Basic/Token/Digest as well as Bearer) and their full parameter
  lists are consumed too, as are short/numeric secrets under sensitive keys,
  JWT/provider prefixes and a conservative high-entropy fallback.
  Server-controlled metadata (status, provider, model, tool names, command
  fields, ...) is redacted when a turn is built and again by each writer.
  Nothing in the generated scenario asks for secrets or destructive actions;
  file-writing prompts are confined to a scratch path in the repo.
* ``--allow-short`` is accepted only with ``--dry-run``. A live run always uses
  the 400-message floor, and the summary reports ``successful_turns``,
  ``deficiency_count``, ``floor_satisfied``, ``honest_accounting`` and
  ``run_verdict`` (including ``all_answers_failed`` for error-status runs) so a
  run of 400 failures cannot be reported as 400 observed answers. The server's
  own ``conversation_id`` from the response is logged
  separately from the requested one, and mismatches are counted.
* ``--dry-run`` swaps in a deterministic in-process fake transport. No socket
  is opened and no LEO server is needed.
* Every turn carries a stable ``request_id`` (derived from seed, conversation
  id and turn index). It is sent in the JSON body and as an ``Idempotency-Key``
  header. Before retrying a timed-out turn the observer reads the server's
  durable conversation history (``GET /conversations/{conversation_id}``) and
  adopts the stored reply instead of resending a request the server may still
  be executing. See ``docs/LEO_OBSERVATION_SCENARIOS.md`` for the contract.
* Post-run reporting is failure-isolated: a hostile path claim or a broken
  checker cannot stop ``summary.json`` and ``conversation.md`` from being
  written.

Usage::

    .venv/bin/python scripts/leo_conversation_observer.py --dry-run
    .venv/bin/python scripts/leo_conversation_observer.py            # live, 400 turns
    .venv/bin/python scripts/leo_conversation_observer.py \
        --messages 500 --out-dir artifacts/observation/long-run

Importable API for tests: ``generate_scenario``, ``validate_message_count``,
``redact``, ``error_signature``, ``extract_tool_info``, ``build_turn``,
``is_successful_turn``, ``turn_deficiencies``, ``build_summary``,
``run_conversation``, ``HttpChatTransport``, ``FakeTransport``,
``sanitize_url``, ``check_tool_result_paths``, ``check_output_completeness``,
``request_id_for_turn``, ``main``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8891/chat"
DEFAULT_CONVERSATION_ID = "observer-human-400"
DEFAULT_MESSAGES = 400
MIN_MESSAGES = 400
DEFAULT_SEED = 20261120
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "observation"
REDACTED = "[REDACTED]"
MAX_TOOL_RESULT_CHARS = 2000
MAX_REPORT_CLAIMS = 200
# Stable retry key: body field ``request_id`` plus this header. A server that
# implements the contract stores it with the turn and echoes it back; older
# servers simply ignore the extra field, and the observer falls back to exact
# prompt matching in the durable conversation history.
REQUEST_ID_HEADER = "Idempotency-Key"
REQUEST_ID_FIELD = "request_id"
HISTORY_PENDING_STATUS = "incomplete"

# Status vocabulary used to distinguish a usable answer from an observed
# deficiency. Unknown statuses are recorded verbatim (redacted) but are not
# silently promoted to "success" when no answer content exists.
SUCCESS_STATUSES = frozenset({"completed", "complete", "ok", "success", "succeeded", "done"})
INCOMPLETE_STATUSES = frozenset({"incomplete", "partial", "pending"})
BLOCKED_STATUSES = frozenset({"blocked", "denied", "refused", "cancelled", "canceled"})
ERROR_STATUSES = frozenset({"error", "failed", "failure", "exception", "transport_error"})
# Fixed harness vocabulary must survive every text-redaction pass. In
# particular ``floor_satisfied_with_deficiencies`` is 33 characters and would
# otherwise be swallowed by the conservative 32+ character secret rule.
_KNOWN_SAFE_LITERALS = frozenset(
    SUCCESS_STATUSES
    | INCOMPLETE_STATUSES
    | BLOCKED_STATUSES
    | ERROR_STATUSES
    | {
        "floor_satisfied_clean", "floor_satisfied_with_deficiencies",
        "below_floor_or_incomplete", "all_turns_failed", "all_answers_failed",
        "unknown", "none", "simulated", "live",
    }
)

# The scenario never asks for secrets, credentials, destructive commands or
# anything outside the workspace. Scratch writes are tiny and local. The
# expanded topic/opener pool keeps repeated phrasings rare across 400 turns.
TOPICS = (
    "the API contract",
    "the chat transcript",
    "memory notes",
    "task planning",
    "the skill registry",
    "tool execution",
    "the model router",
    "scratch notes",
    "the observation harness",
    "code review notes",
    "the audit log",
    "permission rules",
    "event history",
    "tool retries",
    "the summary report",
    "the conversation log",
    "the provider fallback",
    "the local database",
    "the scratch area",
    "pending tool calls",
)
OPENERS = (
    "",
    "Quick one: ",
    "By the way, ",
    "Small question: ",
    "Following up: ",
    "One more thing: ",
    "Meanwhile, ",
    "For clarity: ",
    "A quick check: ",
    "While we're here: ",
    "Just so you know: ",
    "Question: ",
)

# Used only when a template/topic/opener combination repeats: a natural
# conversational tail is tried first, so the mechanical "(follow-up N)" suffix
# is a last resort instead of the default way to keep messages distinct.
ANGLES = (
    " We are still in the same conversation.",
    " Keep the answer brief.",
    " Stay factual, please.",
    " No need for a long explanation.",
    " Just a quick check.",
    " Same thread as before.",
    " Please answer in one short paragraph.",
    " Flag anything you are unsure about.",
    " I am still learning how you work.",
    " Keep it practical.",
    " No tools needed unless you must.",
    " This is another question in our chat.",
)

_GREETING_STARTS = ("hey", "thanks", "okay", "ok,", "hi,")


@dataclass(frozen=True)
class PromptTemplate:
    """One reusable human phrasing, tagged with the behaviour it probes."""

    category: str
    text: str


@dataclass(frozen=True)
class ScenarioMessage:
    """A single scheduled user turn."""

    index: int  # 1-based
    category: str
    text: str


# 13 categories x 5 phrasings = 65 templates. Combined with a per-turn topic,
# opener and (when needed) angle, this yields a long, varied but fully
# deterministic script.
PROMPT_TEMPLATES: tuple[PromptTemplate, ...] = (
    # --- identity -------------------------------------------------------
    PromptTemplate("identity", "Hey, before we go further - who are you, and what should I call you?"),
    PromptTemplate("identity", "In one paragraph, describe what kind of assistant you are and how you handle uncertainty."),
    PromptTemplate("identity", "If I described you to a teammate, what three traits would you want me to mention?"),
    PromptTemplate("identity", "Do you have a name or persona you use, or are you just 'the agent'?"),
    PromptTemplate("identity", "What is your current operating mode - live model or something simulated?"),
    # --- capabilities ---------------------------------------------------
    PromptTemplate("capabilities", "What can you actually do for me today, {topic} aside?"),
    PromptTemplate("capabilities", "Give me a concrete example of a task you could finish for me around {topic}."),
    PromptTemplate("capabilities", "Which of your abilities work offline, and which need outside services?"),
    PromptTemplate("capabilities", "What should I NOT ask you to do around {topic}?"),
    PromptTemplate("capabilities", "If I gave you a messy request about {topic}, how would you break it down?"),
    # --- memory_fact ----------------------------------------------------
    PromptTemplate("memory_fact", "Remember this for later: my preferred note format for {topic} is short bullet points."),
    PromptTemplate("memory_fact", "Please keep a fact: I review {topic} notes every Friday morning."),
    PromptTemplate("memory_fact", "Store this preference: when I ask about {topic}, start with a one-line summary."),
    PromptTemplate("memory_fact", "Note that I spell {topic} with a lowercase first letter in my notes."),
    PromptTemplate("memory_fact", "Can you recall the preference I just gave you about {topic} and repeat it back?"),
    # --- tasks ----------------------------------------------------------
    PromptTemplate("tasks", "Create a task for me: draft a short checklist for {topic} tomorrow."),
    PromptTemplate("tasks", "Add a to-do to review {topic} notes, with no reminders or external notifications."),
    PromptTemplate("tasks", "Show me the task list and tell me which item about {topic} is newest."),
    PromptTemplate("tasks", "Make a small task to summarize {topic} in three lines, then list it back to me."),
    PromptTemplate("tasks", "Put a {topic} checklist task on my list for this week, nothing urgent."),
    # --- listings -------------------------------------------------------
    PromptTemplate("listings", "List the skills you have available and point out any that relate to {topic}."),
    PromptTemplate("listings", "Which tools can you call right now? Keep it to names and one-line purposes."),
    PromptTemplate("listings", "Show me the agents you can delegate to, and which fits {topic} best."),
    PromptTemplate("listings", "Of your listed capabilities, which are read-only and which can write files?"),
    PromptTemplate("listings", "If I wanted a second opinion on {topic}, which agent would you pick and why?"),
    # --- architecture ---------------------------------------------------
    PromptTemplate("architecture", "Explain your architecture like I'm a new engineer joining the {topic} project."),
    PromptTemplate("architecture", "Walk me through what happens between my message and your reply, using {topic} as the example."),
    PromptTemplate("architecture", "Where do memories, tools, and model calls fit in your design?"),
    PromptTemplate("architecture", "How do you prevent a tool call from turning into an unbounded loop?"),
    PromptTemplate("architecture", "What's the difference between a skill, a tool, and an agent in your system?"),
    # --- small_files ----------------------------------------------------
    PromptTemplate("small_files", "Please write a tiny scratch note under artifacts/observation/scratch about {topic}; just 2-3 lines."),
    PromptTemplate("small_files", "Create a small markdown file in artifacts/observation/scratch named after {topic} with a short outline."),
    PromptTemplate("small_files", "Write one sentence about {topic} into a scratch file in the workspace, then tell me the path."),
    PromptTemplate("small_files", "Make a minimal note file for {topic} under artifacts/observation/scratch - nothing outside the workspace."),
    PromptTemplate("small_files", "Update the scratch note about {topic} with one extra bullet, then read it back."),
    # --- self_correction ------------------------------------------------
    PromptTemplate("self_correction", "That last answer was too long - say the same thing about {topic} in one sentence."),
    PromptTemplate("self_correction", "I think you overstated that; correct yourself about {topic} and be precise."),
    PromptTemplate("self_correction", "Actually I meant {topic} in the planning sense, not the literal sense. Re-answer."),
    PromptTemplate("self_correction", "Your previous reply mixed up two ideas. Separate them and fix the {topic} part."),
    PromptTemplate("self_correction", "Try again on {topic}: be shorter, and flag anything you are unsure about."),
    # --- tool_failure ---------------------------------------------------
    PromptTemplate("tool_failure", "Imagine a tool call for {topic} failed with 'not found'. How would you recover?"),
    PromptTemplate("tool_failure", "A read-only tool just timed out on {topic}; what is your fallback plan?"),
    PromptTemplate("tool_failure", "Suppose the model provider is unavailable while we discuss {topic} - what do you tell me?"),
    PromptTemplate("tool_failure", "If a tool returned a permission error for {topic}, what would you do next?"),
    PromptTemplate("tool_failure", "Pretend a {topic} tool returned malformed JSON. Walk me through your retry logic."),
    # --- summarization --------------------------------------------------
    PromptTemplate("summarization", "Summarize what we've discussed so far in three bullets, with {topic} first."),
    PromptTemplate("summarization", "What have I asked you about {topic} in this chat? Keep it factual."),
    PromptTemplate("summarization", "Give me a short recap of the last few turns, and say if you are unsure."),
    PromptTemplate("summarization", "Which preferences did I state in this conversation? List them under {topic} if relevant."),
    PromptTemplate("summarization", "Before we continue, what's the running theme of our chat about {topic}?"),
    # --- honesty --------------------------------------------------------
    PromptTemplate("honesty", "Be honest: is your answer coming from a live model or a mock right now?"),
    PromptTemplate("honesty", "If you're running in simulated mode for {topic}, say so plainly instead of role-playing."),
    PromptTemplate("honesty", "What parts of your {topic} answer are verified, and what parts are guesses?"),
    PromptTemplate("honesty", "Do not claim a tool ran if it didn't - did any tool actually run for {topic}?"),
    PromptTemplate("honesty", "When you can't tell whether {topic} data is live, what do you say?"),
    # --- improvements ---------------------------------------------------
    PromptTemplate("improvements", "What is one concrete improvement you'd suggest for how you handle {topic}?"),
    PromptTemplate("improvements", "If you could change one thing about your own workflow for {topic}, what would it be?"),
    PromptTemplate("improvements", "Propose a small, low-risk experiment to make answers about {topic} more useful."),
    PromptTemplate("improvements", "What would make your summaries of {topic} better without new permissions?"),
    PromptTemplate("improvements", "Give me one honest weakness and one realistic next step for {topic}."),
    # --- small_talk -----------------------------------------------------
    PromptTemplate("small_talk", "How's your day going? (Mine has been mostly {topic} notes.)"),
    PromptTemplate("small_talk", "Thanks - that helped with {topic}. Want a quick breather?"),
    PromptTemplate("small_talk", "I'm taking a short break from {topic}; tell me something light."),
    PromptTemplate("small_talk", "Casual check-in: are you keeping up with all these questions?"),
    PromptTemplate("small_talk", "Okay, last one for now about {topic}: anything you'd like to ask me?"),
)


# ---------------------------------------------------------------------------
# Validation and deterministic scenario generation
# ---------------------------------------------------------------------------


def validate_message_count(count: int, *, allow_short: bool = False) -> int:
    """Return *count*, enforcing the 400-message floor unless explicitly waived.

    ``allow_short=True`` exists for fast unit tests and is rejected by the CLI
    unless ``--dry-run`` is also present; a live observation run must always
    use the default floor of :data:`MIN_MESSAGES`.
    """

    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("message count must be an integer")
    if count < 1:
        raise ValueError("message count must be >= 1")
    if count < MIN_MESSAGES and not allow_short:
        raise ValueError(
            f"message count must be >= {MIN_MESSAGES} for a real observation run "
            f"(got {count}); use --allow-short with --dry-run only for tests"
        )
    return count


def _apply_opener(opener: str, text: str) -> str:
    """Attach a conversational opener without producing broken grammar."""

    if not opener:
        return text
    # "By the way, Who are you?" is wrong; "By the way, who are you?" is not.
    # Greeting-style templates already stand alone and do not take an opener.
    if text.lstrip()[:32].lower().startswith(_GREETING_STARTS):
        return text
    if opener.endswith(", ") and text[:1].isupper():
        text = text[0].lower() + text[1:]
    return f"{opener}{text}"


def generate_scenario(
    count: int = DEFAULT_MESSAGES,
    *,
    seed: int = DEFAULT_SEED,
    allow_short: bool = False,
) -> list[ScenarioMessage]:
    """Build a deterministic, varied script of *count* user messages.

    The template cycles in a fixed order; topic, opener and angle are chosen
    with a per-index seeded RNG, so the first N messages are identical
    regardless of the requested run length. Every user message is distinct; a
    repeated combination first gets a natural tail and only then the
    ``(follow-up N)`` suffix. The first message is always the identity opener.
    """

    validate_message_count(count, allow_short=allow_short)
    scenario: list[ScenarioMessage] = []
    used: set[str] = set()
    for index in range(1, count + 1):
        template = PROMPT_TEMPLATES[(index - 1) % len(PROMPT_TEMPLATES)]
        rng = random.Random(f"{seed}:{index}")
        topic = rng.choice(TOPICS)
        opener = "" if index == 1 else rng.choice(OPENERS)
        text = _apply_opener(opener, template.text.format(topic=topic)).strip()
        if text in used:
            for angle in rng.sample(ANGLES, len(ANGLES)):
                candidate = f"{text}{angle}"
                if candidate not in used:
                    text = candidate
                    break
        if text in used:
            text = f"{text} (follow-up {index})"
        used.add(text)
        scenario.append(ScenarioMessage(index=index, category=template.category, text=text))
    return scenario


# ---------------------------------------------------------------------------
# Redaction, normalization and error signatures
# ---------------------------------------------------------------------------

# Key names are matched component-wise so that ``authToken``, ``client_secret``
# and ``X-Session-Id`` are treated the same as ``auth_token`` / ``AUTH TOKEN``.
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "auth", "authorization", "token", "tokens", "secret", "secrets",
        "password", "passwd", "pwd", "passphrase", "credential", "credentials",
        "cookie", "cookies", "session", "bearer", "csrf", "xsrf", "pin",
    }
)
_SENSITIVE_KEY_COMPOUNDS = (
    "apikey", "accesskey", "privatekey", "secretkey", "accesstoken",
    "refreshtoken", "idtoken", "clientsecret", "sessionid", "bearertoken",
    "authtoken", "xauthtoken",
)
# Legitimate token *counts* must survive redaction: they are the observation
# data the harness exists to record, not credentials.
_BENIGN_KEY_NAMES = frozenset(
    {
        "tokens", "token_count", "token_counts", "token_usage", "usage",
        "total_tokens", "prompt_tokens", "completion_tokens", "input_tokens",
        "output_tokens", "cached_tokens", "reasoning_tokens", "new_tokens",
        "used_tokens", "remaining_tokens", "estimated_tokens",
        "total_token_count", "input_token_count", "output_token_count",
        "error_signature", "top_error_signatures", "signature",
    }
)
_TOKEN_COUNT_KEY_RE = re.compile(
    r"(?:prompt|completion|total|input|output|cached|reasoning|new|used|remaining|estimated|tool_spec)"
    r"[_-]tokens(?:[_-](?:count|counts))?"
)
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _is_sensitive_key(key: Any) -> bool:
    """True when a mapping key names a credential (camelCase/underscore safe).

    Word boundaries are implemented by splitting the key into its name
    components, so ``tokenizer`` and ``authenticated`` stay untouched while
    ``authToken`` and ``client_secret`` are recognized. Known token-count keys
    are explicitly exempt so usage statistics remain observable.
    """

    if not isinstance(key, str):
        return False
    lower = key.strip().lower()
    if lower in _BENIGN_KEY_NAMES or _TOKEN_COUNT_KEY_RE.fullmatch(lower):
        return False
    split = _CAMEL_SPLIT_RE.sub("_", key)
    # Keep the components' original order: joining a *set* makes compound
    # detection depend on hash randomization (``apiKey`` became ``keyapi`` in
    # some processes and leaked short credentials).
    components = [part.lower() for part in re.split(r"[^A-Za-z0-9]+", split) if part]
    parts = set(components)
    if parts & _SENSITIVE_KEY_PARTS:
        return True
    compact = "".join(components)
    return any(compound in compact for compound in _SENSITIVE_KEY_COMPOUNDS)


# ---------------------------------------------------------------------------
# Key-aware credential scanner
# ---------------------------------------------------------------------------
# One regex per operator/wrapper spelling always loses to the next spelling:
# whitespace inside `[ x ]`, `{}` wrappers, a second operator token after the
# first, leading list separators, escaped semicolons, nested keyed values, a
# dangling backslash before a decoded `=`, whitespace-only separators and odd
# escapes in place of `:` each need a new alternative. Instead, find a *known
# sensitive key* (case/separator tolerant) and scan the entire credential span
# after it: any number of operator tokens, nested/mixed wrappers with optional
# interior whitespace, escaped separators, quoted values and comma/semicolon
# lists. Stop at a clause boundary or a new keyed assignment so separate
# credentials still get separate placeholders and benign text stays intact.
_AUTH_SCHEME_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?:aws4-hmac-sha256|api[_-]?key|negotiate|bearer|basic|digest|token|ntlm|mac|apikey)"
    r"(?![A-Za-z0-9])"
)
_BENIGN_SCHEME_WORDS = frozenset(
    {
        "count", "counts", "usage", "used", "required", "invalid", "expired",
        "header", "scheme", "value", "auth", "token", "tokens",
    }
)
# Longest first so `:=`, `=>` and `::` win over their one-character prefixes.
# Repeated and mixed operators (`authToken: = qwertyuiop`) are all consumed as
# part of the prefix; none can be mistaken for the credential value itself.
_ASSIGNMENT_OPERATOR_TOKENS = (
    "=>", "->", ":=", "::", "==", "<>", "(:", "(=", "=", ":", "|",
)
_KEY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_KEY_TRAILING_DIGITS_RE = re.compile(r"[0-9]+[A-Za-z0-9]*")
_KEY_JOIN_CHARS = frozenset("-_.")
# Used only *after* a value item: `authToken: a, client_secret: b` must end the
# first span so the second key gets its own placeholder.
_KEYED_ASSIGNMENT_AHEAD_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_\-\.]*[\"'`]?\s*(?:=>|->|:=|::|==|<>|:|=)"
)
# A multiword sensitive key (`API key`, `session id`, `X Session Id`) can lose
# its leading words when an earlier value span ends mid-key. Remember the
# trailing component so a later bare `key: value` / `id: value` is recognized
# as another assignment instead of being skipped. Both are sensitive on their
# own when followed by an operator; the harness's own descriptive phrase
# (`Response conversation id: abc`) is explicitly exempt so recorded
# conversation ids stay observable.
_TRAILING_KEY_COMPONENTS = frozenset({"key", "id", "token", "secret"})
_BARE_SENSITIVE_FRAGMENTS = frozenset({"key", "id"})
_BENIGN_ID_PRECEDERS = frozenset(
    {
        "conversation", "user", "request", "response", "message", "turn",
        "thread", "correlation", "trace", "event", "item",
    }
)
# Whitespace separates the words of these multiword keys; other words after a
# whitespace gap are values, not key components (`abcdefghij session id: x`
# must not turn into a sensitive key spanning the first plain value).
_MULTIWORD_KEY_LEADS = frozenset(
    {
        "api", "x", "auth", "authorization", "proxy", "client", "access",
        "refresh", "session", "bearer", "basic", "cookie", "set",
    }
)
_MULTIWORD_KEY_FOLLOWERS = frozenset({"key", "id", "token", "secret", "session"})
# A scanner pass replaces at least one assignment; bound the fixed-point loop
# anyway so hostile input can never spin forever.
_MAX_REDACT_PASSES = 64
_WRAPPER_OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}
_WRAPPER_CLOSE = frozenset(")]}>")
_REDACTED_VALUE_STRIP = "\"'`[](){}<>\\ "
_REDACTED_LIST_SPLIT_RE = re.compile(r"\s*[,;]\s*")


def _is_redacted_value(value: str | None) -> bool:
    """True when a value consists only of redaction placeholders.

    Handles nested/mixed wrappers (``[[REDACTED]]``, ``{ [REDACTED] }``,
    ``\\[REDACTED\\]``) and lists of placeholders, so repeated writer passes
    are idempotent without suppressing a fresh secret beside an old one.
    """

    if not value:
        return False
    items = [item.strip() for item in _REDACTED_LIST_SPLIT_RE.split(value) if item.strip()]
    return bool(items) and all(
        item.strip(_REDACTED_VALUE_STRIP) == REDACTED.strip("[]") for item in items
    )


def _preceding_word(text: str, pos: int) -> str:
    """Lowercased word (or empty string) directly before the word at *pos*."""

    end = pos
    while end > 0 and text[end - 1].isspace():
        end -= 1
    start = end
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_-."):
        start -= 1
    return text[start:end].lower()


def _sensitive_key_at(
    text: str, pos: int, trailing_components: frozenset[str] = frozenset()
) -> tuple[int, str] | None:
    """Return ``(end, key)`` when a sensitive key starts at *pos*.

    Separator tolerance covers camelCase, underscores, hyphens, dots and
    spaces, so ``authToken``, ``AUTH TOKEN``, ``client-secret``, ``API key``,
    ``X-Session-Id`` and ``PIN`` all resolve. Whitespace only joins known
    multiword-key words: a recognized key is never extended into a plain value
    (``authToken   qwertyuiop``) and an unrelated word never becomes part of a
    key (``abcdefghij session id: x``). A bare trailing component (``key``,
    ``id``, ``token``, ``secret``) followed by an operator is sensitive too,
    which catches concatenated assignments such as
    ``API key=[REDACTED] key: qwertyuiop``. The harness's descriptive
    ``conversation id: <id>`` phrase is the documented benign exception.
    """

    word = _KEY_WORD_RE.match(text, pos)
    if word is None:
        return None
    candidate = word.group(0)
    end = word.end()
    best = (end, candidate) if _is_sensitive_key(candidate) else None
    fragment = candidate.lower()
    preceding = text[pos - 1] if pos else ""
    if (
        best is None
        and (not preceding or not (preceding.isalnum() or preceding in "_-."))
        and fragment in (_BARE_SENSITIVE_FRAGMENTS | trailing_components)
        and not (fragment == "id" and _preceding_word(text, pos) in _BENIGN_ID_PRECEDERS)
        and _operator_at(text, _skip_ws(text, end))
    ):
        best = (end, candidate)
    leading = word.group(0).lower()
    for _ in range(3):
        if end >= len(text):
            break
        separator = text[end]
        following_word: str | None = None
        if separator in " \t":
            jump = end + 1
            while jump < len(text) and text[jump] in " \t":
                jump += 1
            following = _KEY_WORD_RE.match(text, jump) or _KEY_TRAILING_DIGITS_RE.match(text, jump)
            if following is None:
                break
            following_word = following.group(0).lower()
            # Only known key words may cross a whitespace gap. The first word
            # of a not-yet-recognized key must be a plausible lead, and a
            # recognized key may absorb only another key component, never a
            # value word.
            if following_word not in _MULTIWORD_KEY_FOLLOWERS:
                break
            if best is None and leading not in _MULTIWORD_KEY_LEADS:
                break
        elif separator in _KEY_JOIN_CHARS:
            jump = end + 1
            following = _KEY_WORD_RE.match(text, jump) or _KEY_TRAILING_DIGITS_RE.match(text, jump)
            if following is None:
                break
        else:
            break
        candidate = candidate + "_" + following.group(0)
        end = following.end()
        if _is_sensitive_key(candidate):
            best = (end, candidate)
    return best


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _skip_prefix_ws(text: str, pos: int) -> int:
    """Skip the whitespace between a key and its value, including escaped `-n`."""

    return _skip_ws(text, pos)


def _operator_at(text: str, pos: int) -> int:
    """Length of an operator token at *pos*, zero when none starts here.

    A lone `-` only counts when it introduces `->` (possibly with whitespace);
    that keeps hyphenated identifiers and paths out of the scanner's way.
    """

    for token in _ASSIGNMENT_OPERATOR_TOKENS:
        if text.startswith(token, pos):
            return len(token)
    if (
        text[pos:pos + 1] == "-"
        and pos + 1 < len(text)
        and (text[pos + 1] in " \t>" or text[pos + 1] == "-")
    ):
        return 1
    return 0


def _benign_scheme_tail(text: str, pos: int) -> bool:
    """True for descriptive uses such as ``Bearer count: 12``."""

    word = _KEY_WORD_RE.match(text, pos)
    if word is None or word.group(0).lower() not in _BENIGN_SCHEME_WORDS:
        return False
    end = word.end()
    while end < len(text) and text[end] not in "\r\n":
        end += 1
    return not any(char in text[word.end():end] for char in "=,;")


def _consume_prefix(text: str, pos: int, after_scheme: bool = False) -> tuple[int, bool] | None:
    """Consume operators/scheme words between a key and its value.

    Returns ``(value_start, saw_scheme)`` or ``None`` when the words that
    follow only describe a scheme (``count``, ``scheme``, ``header``, ...).
    """

    pos = _skip_prefix_ws(text, pos)
    if pos < len(text) and text[pos] in "\"'`]":  # closing quote after a quoted key
        pos += 1
    pos = _skip_prefix_ws(text, pos)
    saw_scheme = after_scheme
    while pos < len(text):
        operator = _operator_at(text, pos)
        if operator:
            pos = _skip_prefix_ws(text, pos + operator)
            continue
        if text[pos] in ",;":  # leading list separator, escaped or not
            pos = _skip_prefix_ws(text, pos + 1)
            continue
        if text[pos] == "\\" and pos + 1 < len(text) and text[pos + 1] in "=:,;":
            pos = _skip_prefix_ws(text, pos + 2)  # odd escape hiding an operator
            continue
        scheme = _AUTH_SCHEME_RE.match(text, pos)
        if scheme is not None:
            saw_scheme = True
            pos = _skip_prefix_ws(text, scheme.end())
            if not after_scheme and _benign_scheme_tail(text, pos):
                return None
            continue
        if _benign_scheme_tail(text, pos):
            return None
        break
    return pos, saw_scheme


def _has_assignment_separator(text: str, pos: int) -> bool:
    """True when a key is followed by an operator, wrapper or value separator.

    Without this guard a sensitive word embedded in a longer identifier or a
    file path (``test_url_credentials_scrubbed_0/scratch/note.md``) would be
    mistaken for an assignment and the rest of the path would be redacted.
    """

    following = _skip_prefix_ws(text, pos)
    if following < len(text) and text[following] in "\"'`]":
        following = _skip_prefix_ws(text, following + 1)
    if (
        following == pos
        and pos < len(text)
        and text[pos] in "_-./\\"
        and not _operator_at(text, pos)
    ):
        return False
    if following >= len(text):
        return False
    char = text[following]
    return (
        bool(_operator_at(text, following))
        or char in "([{<>\"'`"
        or following > pos and (char.isalnum() or char in "([{<\"'`")
    )


def _has_assignment_ahead(text: str, pos: int) -> bool:
    """True when a new keyed assignment starts at *pos*.

    Besides the single-word form matched by ``_KEYED_ASSIGNMENT_AHEAD_RE``,
    a multiword sensitive key can end directly at an operator (``API key:``,
    ``X Session Id:``). Both forms must end the previous value span so each
    assignment gets its own placeholder instead of one span swallowing the
    next key's leading word.
    """

    if _KEYED_ASSIGNMENT_AHEAD_RE.match(text, pos):
        return True
    key = _sensitive_key_at(text, pos)
    if key is None:
        return False
    end = key[0]
    return text[end : end + 1].isspace() or bool(_operator_at(text, end))


def _consume_quoted(text: str, pos: int) -> int:
    """Consume a quoted string, including escaped quotes and interior spaces."""

    quote = text[pos]
    pos += 1
    while pos < len(text):
        if text[pos] == "\\" and pos + 1 < len(text):
            pos += 2
            continue
        if text[pos] == quote:
            return pos + 1
        if text[pos] in "\r\n":
            return pos
        pos += 1
    return pos


def _scan_value(text: str, pos: int, scheme_mode: bool = False) -> int:
    """Return the end of the whole credential value span starting at *pos*.

    Wrappers nest in any combination and may hold whitespace; backslashes join
    escaped separators; commas and semicolons continue a list. A keyed
    assignment or clause boundary ends the span. In scheme mode parameter
    assignments (``nonce=``, ``response=``, ...) belong to the same header and
    are consumed with it.
    """

    pos = _skip_ws(text, pos)
    while pos < len(text) and text[pos] in ",;":  # leading list separators
        pos = _skip_ws(text, pos + 1)
    value_start = pos
    stack: list[str] = []
    while pos < len(text):
        char = text[pos]
        if char == "\\" and pos + 1 < len(text):
            pos += 2  # escaped separator or wrapper: never a stopping point
            continue
        if char in "\"'`":
            pos = _consume_quoted(text, pos)
            continue
        if char in _WRAPPER_OPEN:
            stack.append(_WRAPPER_OPEN[char])
            pos += 1
            continue
        if char in _WRAPPER_CLOSE:
            if stack and char == stack[-1]:
                stack.pop()
                pos += 1
                continue
            if not stack:
                break  # unmatched closer is a clause boundary
            pos += 1
            continue
        if not stack:
            if char in "\r\n":
                break
            if char in ".!?" and (pos + 1 == len(text) or text[pos + 1].isspace()):
                break
            if char in ",;":
                following = _skip_ws(text, pos + 1)
                if not scheme_mode and pos > value_start and _has_assignment_ahead(text, following):
                    break
                pos = following
                continue
            if char.isspace():
                following = _skip_ws(text, pos)
                if not scheme_mode and following > value_start and _has_assignment_ahead(text, following):
                    break
                pos = following
                continue
        pos += 1
    return pos


def _scan_credentials(text: str) -> str:
    """Replace the complete value of every sensitive key or auth scheme."""

    pieces: list[str] = []
    cursor = 0
    index = 0
    trailing_components: set[str] = set()
    while index < len(text):
        if not text[index].isalpha():
            index += 1
            continue
        found = _sensitive_key_at(text, index, frozenset(trailing_components))
        bare_scheme = found is None and _AUTH_SCHEME_RE.match(text, index) is not None
        if found is None:
            if not bare_scheme:
                word = _KEY_WORD_RE.match(text, index)
                index = word.end() if word else index + 1
                continue
            key_end = _AUTH_SCHEME_RE.match(text, index).end()
        else:
            key_end = found[0]
            if not _has_assignment_separator(text, key_end):
                word = _KEY_WORD_RE.match(text, index)
                index = word.end() if word else key_end
                continue
            # Remember the last word so a bare trailing component appearing in
            # a later assignment in the same string is still recognized.
            last_word = found[1].rsplit("_", 1)[-1].lower()
            if last_word in _TRAILING_KEY_COMPONENTS:
                trailing_components.add(last_word)
        prefix = _consume_prefix(text, key_end, after_scheme=bare_scheme)
        if prefix is None:
            index = key_end
            continue
        value_start, scheme_mode = prefix
        value_end = _scan_value(text, value_start, scheme_mode=scheme_mode)
        if value_end <= value_start or _is_redacted_value(text[value_start:value_end]):
            index = max(index + 1, value_end)
            continue
        pieces.append(text[cursor:index] if bare_scheme else text[cursor:key_end])
        pieces.append(REDACTED if bare_scheme else f"={REDACTED}")
        cursor = value_end
        index = value_end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _scan_credentials_to_fixed_point(text: str) -> str:
    """Repeat the key-aware scanner until no further assignment is consumed.

    A replacement can end immediately before another assignment (or expose a
    bare trailing key component left by an earlier span). Iterating here,
    inside the string branch of :func:`redact`, consumes all of them without
    relying on a caller to call ``redact`` again.
    """

    result = text
    for _ in range(_MAX_REDACT_PASSES):
        updated = _scan_credentials(result)
        if updated == result:
            break
        result = updated
    return result


# Conservative entropy fallback for secrets in innocuously named metadata:
# at least three character classes (one of them a digit or a non-hyphen
# symbol). Hyphenated words, paths, timestamps and ids stay untouched.
_ENTROPY_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9_+/~!@#$%^&*=.~-]{8,}(?![A-Za-z0-9_])")
_NUMERIC_SECRET_RE = re.compile(r"(?<![0-9])[0-9]{12,}(?![0-9])")


def _looks_like_credential(text: str) -> bool:
    """Conservative entropy test: mixed character classes, not a plain word."""

    if len(text) < 8:
        return False
    # Paths and ISO dates are the most common benign mixed-character strings
    # in this harness's output. Leave them to the path/URL-specific handling.
    if "/" in text or "-" in text:
        return False
    lower = any(char.islower() for char in text)
    upper = any(char.isupper() for char in text)
    digit = any(char.isdigit() for char in text)
    # '/' is deliberately not a strong symbol: paths and URLs must survive.
    strong_symbol = any(char in "_+~!@#$%^&*" for char in text)  # not '=', '-', '.' or '/'
    return (digit and (upper or strong_symbol)) or (lower and upper and strong_symbol)


# Shapes that need no key: they are credentials by construction. Everything
# key-dependent is handled by the scanner above, not by another regex variant.
_REDACTION_RULES: tuple[tuple[re.Pattern[str], Any], ...] = (
    # Defense in depth: a URL with embedded credentials is never logged. The
    # CLI also strips userinfo from --url before storing it in the config.
    (re.compile(r"(?i)\b(https?://)[^\s/@:]+:[^\s/@]+@"), lambda m: f"{m.group(1)}{REDACTED}@"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), REDACTED),
    (re.compile(r"\b(?:sk-(?:proj-)?|gh[pousr]_|xox[baprs]-|AKIA|AIza)[A-Za-z0-9_-]{8,}\b"), REDACTED),
    (_ENTROPY_CANDIDATE_RE, lambda m: REDACTED if _looks_like_credential(m.group(0)) else m.group(0)),
    (_NUMERIC_SECRET_RE, REDACTED),
    (re.compile(r"\b[A-Za-z0-9_-]{32,}\b"), REDACTED),
)

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}")
# Every JSON string escape, not just escaped quotes: ``\n``/``\t``/``:``
# can stand where an assignment separator belongs, and ``\`` can hide
# another full level of escaping.
_JSON_ESCAPE_RE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|(.))", re.DOTALL)
_JSON_ESCAPE_VALUES = {
    "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
    '"': '"', "'": "'", "\\": "\\", "/": "/",
}
_MAX_UNESCAPE_PASSES = 64
# An odd run of backslashes leaves one backslash glued to a decoded character
# after the pairs are consumed: ``{"authToken":\\\n"secret"}`` becomes
# ``{"authToken":\<newline>"secret"}``. ``\<newline>``/``\:``/``\[`` are not
# recognized JSON escapes, so the container will not parse structurally.
# Remove that dangling backslash before re-parsing. Quotes/backticks are
# deliberately excluded: ``\"`` is a valid escape inside a JSON string.
_DANGLING_ESCAPE_RE = re.compile(r"\\(?=[ \t\r\n\f:=,;<>|{}\[\]()])")
_JSON_DECODER = json.JSONDecoder()


def _decode_json_escape(match: re.Match[str]) -> str:
    """Decode one escape, leaving unknown escapes intact (no infinite loops)."""

    code = match.group(1)
    if code is not None:
        point = int(code, 16)
        # A lone surrogate would break strict JSONL/UTF-8 writing; keep the
        # original escape instead of emitting an unencodable character.
        if 0xD800 <= point <= 0xDFFF:
            return match.group(0)
        return chr(point)
    char = match.group(2)
    return _JSON_ESCAPE_VALUES.get(char, match.group(0))


def _normalize_escapes(text: str) -> str:
    """Decode JSON string escapes until no recognized escape remains.

    Escaped separators (``authToken\\n: value``, ``key\\u003a value``) and
    deeply double-encoded payloads hide credentials from every text rule.
    Decoding to real whitespace/``:``/``=`` before matching turns them back
    into ordinary assignment boundaries. Each pass also repairs the dangling
    backslash left by an odd escape run, so the result is parseable JSON that
    the structural pass can redact before any text rule runs. The bounded loop
    follows nested escaping to a fixed point instead of assuming a fixed
    (four-, sixteen- or 64-level) encoding depth.
    """

    result = text
    for _ in range(_MAX_UNESCAPE_PASSES):
        updated = _JSON_ESCAPE_RE.sub(_decode_json_escape, result)
        updated = _DANGLING_ESCAPE_RE.sub("", updated)
        if updated == result:
            break
        result = updated
    return result


def _unescape_json_quotes(text: str) -> str:
    """Backward-compatible alias for the broader :func:`_normalize_escapes`."""

    return _normalize_escapes(text)


def _redact_embedded_json(text: str) -> str:
    """Redact embedded JSON objects/arrays structurally when they parse.

    Regex key matching cannot reliably reach a key inside a nested or encoded
    JSON string, and short numeric / 2-4 character values need the structural
    sensitive-key rule. Parse cleanly-formed containers, let :func:`redact`
    walk them, and leave anything unparseable to the text rules.
    """

    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        starts = [pos for pos in (text.find("{", cursor), text.find("[", cursor)) if pos >= 0]
        if not starts:
            chunks.append(text[cursor:])
            break
        start = min(starts)
        chunks.append(text[cursor:start])
        try:
            parsed, end = _JSON_DECODER.raw_decode(text[start:])
        except ValueError:
            chunks.append(text[start])
            cursor = start + 1
            continue
        if isinstance(parsed, (dict, list)):
            try:
                chunks.append(json.dumps(redact(parsed), ensure_ascii=False, separators=(", ", ": ")))
            except Exception:  # noqa: BLE001 - untrusted JSON must not abort redaction
                chunks.append(text[start : start + end])
        else:
            chunks.append(text[start : start + end])
        cursor = start + end
    return "".join(chunks)


def _redact_string(value: str) -> str:
    if value in _KNOWN_SAFE_LITERALS:
        return value
    # Parse the original text first: escaped quotes *inside* a valid JSON
    # string must stay escaped when the container is re-serialized.
    result = _redact_embedded_json(value)
    # Escapes can hide assignment separators, and a JSON container with an
    # escaped newline between key and value only parses *after* decoding.
    # Normalize, then repeat the structural pass before the text rules.
    result = _redact_embedded_json(_normalize_escapes(result))
    result = _scan_credentials_to_fixed_point(result)
    for pattern, replacement in _REDACTION_RULES:
        result = pattern.sub(replacement, result)
    return result


def redact(value: Any) -> Any:
    """Recursively redact credentials in text, dicts and sequences.

    Applied to prompts, responses, logs and the summary when a turn is built
    and again in every writer. Besides token-shaped text patterns, mapping
    keys are inspected: a value stored under ``authToken``, ``client_secret``,
    ``session_id`` or ``Cookie`` is replaced wholesale even when it is short,
    numeric, or nested. Known token-count fields are deliberately exempt.
    """

    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key = redact(key)
            redacted[safe_key] = REDACTED if _is_sensitive_key(key) else redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(redact(item) for item in value)
    return value


def error_signature(kind: str, message: Any) -> str:
    """Normalize an error into a stable, redacted signature for grouping."""

    text = redact(_safe_str(message if message is not None else ""))
    text = _UUID_RE.sub("<id>", text)
    text = _PATH_RE.sub("<path>", text)
    text = re.sub(r"\d+(?:\.\d+)?", "#", text)
    text = re.sub(r"\s+", " ", text).strip().lower()[:140]
    kind = redact(_safe_str(kind))
    return f"{kind}: {text}" if text else kind


def _safe_str(value: Any, fallback: str = "") -> str:
    """Stringify a hostile value without ever raising."""

    try:
        return str(value)
    except Exception:  # noqa: BLE001 - arbitrary response objects are untrusted
        return fallback


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a number, rejecting NaN/Infinity and non-numeric junk."""

    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _sanitize_non_finite(value: Any, depth: int = 0) -> Any:
    """Replace NaN/Infinity individually so sibling numeric fields survive."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth >= 24:
        return "<recursion-limit>"
    if isinstance(value, dict):
        try:
            return {_safe_str(key): _sanitize_non_finite(item, depth + 1) for key, item in value.items()}
        except Exception:  # noqa: BLE001
            return "<unserializable>"
    if isinstance(value, (list, tuple)):
        try:
            return [_sanitize_non_finite(item, depth + 1) for item in value]
        except Exception:  # noqa: BLE001
            return "<unserializable>"
    return value


def _json_dumps(value: Any) -> str:
    """Strict JSON encoding; non-finite numbers and weird objects cannot raise."""

    prepared = _sanitize_non_finite(value)
    try:
        return json.dumps(prepared, default=_safe_str, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return json.dumps({"error": "json_serialization_fallback", "value": _safe_str(value, "<unserializable>")})


def _json_safe(value: Any) -> Any:
    """Coerce arbitrary response values into strictly JSON-serializable data."""

    try:
        return json.loads(_json_dumps(value))
    except Exception:  # noqa: BLE001
        return "<unserializable>"


def _safe_list(value: Any) -> list[Any]:
    """Normalize a response list field (including odd shapes) to a JSON list."""

    if value is None:
        return []
    if isinstance(value, set):
        try:
            return [_json_safe(item) for item in sorted(value, key=_safe_str)]
        except Exception:  # noqa: BLE001
            return [_json_safe(value)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return [_json_safe(value)]


def _normalize_usage(value: Any) -> dict[str, Any]:
    """Keep usage data without letting a strange object abort the run.

    Non-finite floats are replaced field by field (with ``None``) so one NaN
    cannot collapse the entire usage record into an opaque string.
    """

    if value is None:
        return {}
    safe = _json_safe(value)
    return safe if isinstance(safe, dict) else {"raw": safe}


_TOOL_ERROR_KEYS = ("error", "errors", "failure", "reason", "approval_required", "not_executed")
# Structured fields are authoritative. The tolerant forms below still find
# them when a redacted tool result is no longer strict JSON (for example
# `"error": null` next to `"credential_env": [REDACTED]`, or a truncated
# result). A bare substring such as `error": null` is never an error.
_TOLERANT_SUCCESS_RE = re.compile(
    r'(?i)(?<![A-Za-z0-9_])(?:\\?["\'])?success(?:\\?["\'])?\s*:\s*'
    r'(?P<value>true|false|null|"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)'
)
_TOLERANT_ERROR_RE = re.compile(
    r'(?i)(?<![A-Za-z0-9_])(?:\\?["\'])?error(?:\\?["\'])?\s*:\s*'
    r'(?P<value>null|false|true|0|""|\'\'|\{\s*\}|\[\s*\]|'
    r'"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)'
)
_FREE_FORM_ERROR_RE = re.compile(
    r"(?i)\b(?:error|exception|traceback|failed)\b[^\n]{0,160}"
)
_PLACEHOLDER_REPAIR_RE = re.compile(re.escape(REDACTED))


def _empty_tool_value(value: Any) -> bool:
    """True for the null/empty sentinel values tool results use for 'no error'."""

    if value is None or value is False:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "none", "null", "ok", "0")
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return not value
    return False


def _tool_error_text(key: str, value: Any) -> str:
    rendered = value if isinstance(value, str) else _json_dumps(redact(value))
    return f"{key}: {_safe_str(redact(rendered))}"


def _error_from_tool_mapping(parsed: dict[str, Any], *, depth: int = 0) -> str | None:
    """Read ``success``/``error`` (and legacy aliases) from a tool result."""

    for key in _TOOL_ERROR_KEYS:
        if key in parsed and not _empty_tool_value(parsed[key]):
            return _tool_error_text(key, parsed[key])
    if parsed.get("success") is False or parsed.get("ok") is False:
        return "tool reported failure"
    # Outer success/error fields are authoritative. Only inspect nested
    # objects when the outer object did not state an outcome itself.
    stated = any(key in parsed for key in _TOOL_ERROR_KEYS + ("success", "ok"))
    if depth >= 4 or stated:
        return None
    for value in parsed.values():
        if isinstance(value, dict):
            message = _error_from_tool_mapping(value, depth=depth + 1)
            if message:
                return message
        elif isinstance(value, str):
            nested = _parse_tool_json(value)
            if isinstance(nested, dict):
                message = _error_from_tool_mapping(nested, depth=depth + 1)
                if message:
                    return message
    return None


def _parse_tool_json(raw: str) -> Any:
    """Parse a tool result, tolerating redaction placeholders and double encoding."""

    text = raw.strip() if isinstance(raw, str) else ""
    if not text:
        return None
    for _ in range(4):
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            # `[REDACTED]` in a numeric position (e.g. 10.[REDACTED]) makes
            # otherwise valid JSON unparseable. Substituting a zero keeps the
            # surrounding structured fields readable.
            repaired = _PLACEHOLDER_REPAIR_RE.sub("0", text)
            if repaired == text:
                return None
            try:
                parsed = json.loads(repaired)
            except (TypeError, ValueError):
                return None
        if isinstance(parsed, str):
            text = parsed.strip()
            continue
        return parsed
    return None


def _tolerant_tool_outcome(raw: str) -> tuple[str | None, bool]:
    """Return ``(error_message, found_structured_fields)`` from fragile JSON."""

    found = False
    error_match = _TOLERANT_ERROR_RE.search(raw)
    if error_match is not None:
        found = True
        value = error_match.group("value").strip().lower()
        if value not in ("null", "false", "0", '""', "''", "{}", "[]", "none"):
            return _tool_error_text("error", value.strip('"\''))
    success_match = _TOLERANT_SUCCESS_RE.search(raw)
    if success_match is not None:
        found = True
        if success_match.group("value").strip().lower() == "false":
            return "tool reported failure", True
    return None, found


def _tool_error_message(name: str, raw: str) -> str | None:
    """Return an error only for a structured failure, never for ``error: null``.

    Strict JSON is parsed first, then redaction-repaired/double-encoded JSON,
    then tolerant key/value extraction. The free-text heuristic is a last
    resort and is suppressed whenever a structured ``success``/``error`` field
    was found with a non-error value.
    """

    parsed = _parse_tool_json(raw)
    if isinstance(parsed, dict):
        return _error_from_tool_mapping(parsed)
    text = _safe_str(raw)
    message, structured = _tolerant_tool_outcome(text)
    if structured:
        return message
    if _TOLERANT_ERROR_RE.search(text) is not None:
        return None  # a null/empty error field in malformed JSON is not an error
    match = _FREE_FORM_ERROR_RE.search(text)
    return match.group(0).strip() if match else None


def extract_tool_info(tool_results: Any) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Split ``tool_results`` into redacted names, result records and errors."""

    # A single dict/string is a legitimate one-tool response; normalizing it
    # avoids silently reporting "no tools" for a non-list tool_results field.
    names: list[str] = []
    records: list[dict[str, str]] = []
    errors: list[str] = []
    for item in _safe_list(tool_results):
        if not isinstance(item, dict):
            item = {"result": item}
        name = _safe_str(redact(_safe_str(item.get("name") or item.get("tool") or "unknown")))
        raw = item.get("result", item.get("output", ""))
        if not isinstance(raw, str):
            raw = _json_dumps(raw)
        safe = redact(raw)
        if len(safe) > MAX_TOOL_RESULT_CHARS:
            safe = safe[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"
        names.append(name)
        records.append({"name": name, "result": safe})
        message = _tool_error_message(name, raw)
        if message:
            errors.append(error_signature(f"tool:{name}", message))
    return names, records, errors


# ---------------------------------------------------------------------------
# Transports and URL handling
# ---------------------------------------------------------------------------


def sanitize_url(url: str) -> str:
    """Return a loggable form of *url* with userinfo/query/fragment removed.

    The live transport still receives the original URL (a server may require
    credentials). Only the copy stored in the run config is sanitized.
    """

    text = str(url)
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError:
        return REDACTED
    if not parts.scheme or not parts.netloc:
        return redact(text)
    host = parts.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))


class TransportError(Exception):
    """A failed single turn (timeout, HTTP error, malformed response, ...)."""


def request_id_for_turn(conversation_id: str, index: int, seed: int) -> str:
    """Stable, control-free retry key for one logical turn of one run.

    The same id is reused for every attempt, so a server implementing the
    idempotency contract can collapse a timed-out retry into the original
    execution instead of writing a second user/assistant row pair.
    """

    digest = hashlib.sha256(
        f"{seed}|{_safe_str(conversation_id)}|{_safe_int(index)}".encode("utf-8")
    ).hexdigest()[:16]
    # Keep the whole id below the 32-character high-entropy rule so redaction
    # cannot turn a retry key into ``[REDACTED]``.
    return f"observer-{digest}"


def _supports_keyword(method: Any, name: str) -> bool:
    """True when a (possibly third-party) transport method accepts *name*."""

    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


class ChatTransport(Protocol):
    """Minimal interface the observer needs from a chat transport."""

    def send(
        self, *, message: str, conversation_id: str, timeout_seconds: float,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        ...

    def turn_in_history(
        self, *, message: str, conversation_id: str, request_id: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any] | None:
        """Return an already-executed turn's response, or ``None`` if absent."""

        ...


class HttpChatTransport:
    """Standard-library HTTP POST transport for a live LEO server.

    Sends ``request_id`` in the body and as ``Idempotency-Key``. Unknown body
    fields are ignored by older servers. Before a retry, :meth:`turn_in_history`
    reads ``GET /conversations/{conversation_id}`` and matches the stored user
    message by metadata ``request_id`` (falling back to exact prompt text for
    servers that do not store ids yet).
    """

    def __init__(self, url: str = DEFAULT_URL, *, history_url: str | None = None) -> None:
        self.url = url
        self.history_url = history_url

    def _conversation_url(self, conversation_id: str) -> str:
        if self.history_url:
            template = self.history_url
            if "{conversation_id}" in template:
                return template.replace(
                    "{conversation_id}", urllib.parse.quote(conversation_id, safe="")
                )
            return template
        parts = urllib.parse.urlsplit(self.url)
        path = parts.path[:-5] if parts.path.endswith("/chat") else parts.path
        path = path.rstrip("/") + "/conversations/" + urllib.parse.quote(conversation_id, safe="")
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def send(
        self, *, message: str, conversation_id: str, timeout_seconds: float,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "message": message,
            "channel": "api",
            "conversation_id": conversation_id,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if request_id:
            body[REQUEST_ID_FIELD] = request_id
            headers[REQUEST_ID_HEADER] = request_id
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise TransportError(f"invalid JSON response: {exc}") from exc
        if not isinstance(data, dict):
            raise TransportError("invalid response: expected a JSON object")
        return data

    def turn_in_history(
        self, *, message: str, conversation_id: str, request_id: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any] | None:
        """Adopt a server-side turn that completed despite a client timeout."""

        request = urllib.request.Request(
            self._conversation_url(conversation_id), method="GET",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # No conversation row yet means the timed-out request did not
            # commit; a retry is safe in that case.
            if exc.code == 404:
                return None
            raise TransportError(f"history: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise TransportError(f"history: {type(exc).__name__}: {exc}") from exc
        if not isinstance(data, dict):
            raise TransportError("history: expected a JSON object")
        messages = _safe_list(data.get("messages"))
        prompt = _safe_str(message).strip()
        match = -1
        for position, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            if _safe_str(item.get("role")).lower() not in ("user", "human"):
                continue
            metadata = item.get("metadata")
            recorded_id = ""
            if isinstance(metadata, dict):
                recorded_id = _safe_str(metadata.get(REQUEST_ID_FIELD))
            if not recorded_id:
                recorded_id = _safe_str(item.get(REQUEST_ID_FIELD))
            if (request_id and recorded_id == request_id) or (
                prompt and _safe_str(item.get("content")).strip() == prompt
            ):
                match = position
                break
        if match < 0:
            return None
        for item in messages[match + 1:]:
            if isinstance(item, dict) and _safe_str(item.get("role")).lower() == "assistant":
                metadata = item.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                return {
                    "conversation_id": conversation_id,
                    "content": _safe_str(item.get("content") or ""),
                    "status": _safe_str(
                        metadata.get("status") or "completed"
                    ),
                    "provider": _safe_str(metadata.get("provider") or ""),
                    "model": _safe_str(metadata.get("model") or ""),
                    "simulated": metadata.get("simulated"),
                    "usage": metadata.get("usage"),
                    "memories_used": metadata.get("memories_used"),
                    "tool_results": metadata.get("tool_results"),
                    REQUEST_ID_FIELD: request_id,
                    "from_history": True,
                }
        # The user row is durable but the server is still executing: record an
        # honest incomplete turn rather than resending the prompt.
        return {
            "conversation_id": conversation_id,
            "content": "",
            "status": HISTORY_PENDING_STATUS,
            REQUEST_ID_FIELD: request_id,
            "from_history": True,
            "history_pending": True,
        }


class FakeTransport:
    """Deterministic, offline transport used by ``--dry-run`` and unit tests.

    It records every message it receives and can inject transport failures,
    tool errors, varied statuses, a token-shaped content leak, and a token in
    *every* metadata field so the redaction path can be exercised without a
    server or a network. Scratch writes are explicitly marked as simulated.
    """

    def __init__(
        self,
        *,
        failure_indices: tuple[int, ...] | list[int] = (),
        tool_error_indices: tuple[int, ...] | list[int] = (),
        statuses: tuple[str, ...] | list[str] = ("completed",),
        provider: str = "mock",
        model: str = "fake-model",
        simulated: bool | None = True,
        leak_token: bool = False,
        metadata_leak_token: str = "",
        dry_run: bool = True,
        scratch_path: str = "artifacts/observation/scratch/note.md",
        latency_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        history: dict[str, dict[str, Any]] | None = None,
        history_error: Exception | None = None,
    ) -> None:
        self.failure_indices = set(failure_indices)
        self.tool_error_indices = set(tool_error_indices)
        self.statuses = tuple(statuses) or ("completed",)
        self.provider = provider
        self.model = model
        self.simulated = simulated
        self.leak_token = leak_token
        self.metadata_leak_token = metadata_leak_token
        self.dry_run = dry_run
        self.scratch_path = scratch_path
        self.latency_seconds = latency_seconds
        self.sleep = sleep
        self.calls: list[str] = []  # every attempt, including retries
        self.sent_requests: list[dict[str, str]] = []  # ids used on each attempt
        self.history = dict(history or {})
        self.history_error = history_error
        self._turn_by_message: dict[str, int] = {}

    def send(
        self, *, message: str, conversation_id: str, timeout_seconds: float,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(message)
        self.sent_requests.append(
            {"message": message, "conversation_id": conversation_id, "request_id": request_id or ""}
        )
        # Scenario texts are unique, so retries of the same turn map to the
        # same 1-based turn index even though ``calls`` grows on each attempt.
        if message not in self._turn_by_message:
            self._turn_by_message[message] = len(self._turn_by_message) + 1
        index = self._turn_by_message[message]
        if index in self.failure_indices:
            raise TransportError("ConnectionResetError: simulated disconnect during turn")
        if self.latency_seconds > 0:
            self.sleep(self.latency_seconds)
        leak = self.metadata_leak_token
        tool_results: list[dict[str, str]] = []
        if index in self.tool_error_indices:
            tool_results.append(
                {
                    "name": f"read_file{leak}",
                    "result": json.dumps({"error": f"not found: scratch note {leak or 42}"}),
                }
            )
        else:
            result: dict[str, Any] = {
                "ok": True,
                "path": self.scratch_path,
            }
            if self.dry_run:
                result["dry_run"] = True
                result["note"] = "DRY-RUN: simulated scratch write; no file was created"
            tool_results.append({"name": f"write_note{leak}", "result": json.dumps(result)})
        content = f"Observed turn {index} about your question. Here is a short, honest answer."
        if self.leak_token and index == 1:
            content += " Debug header: Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz0123456789"
        # Every server-controlled metadata field can carry the injected token;
        # build_turn() and all writers must redact them before they reach disk.
        return {
            "conversation_id": conversation_id,
            REQUEST_ID_FIELD: request_id or "",
            "content": content,
            "status": f"{self.statuses[(index - 1) % len(self.statuses)]}{leak}",
            "provider": f"{self.provider}{leak}",
            "model": f"{self.model}{leak}",
            "simulated": self.simulated,
            "model_status": f"{'mock' if self.simulated else 'live'}{leak}",
            # Deliberately not JSON-safe: must be normalized, not abort the run.
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "raw": {1, 2, 3}},
            "memories_used": [f"mem-{index}{leak}"],
            "tool_results": tool_results,
            "tool_calls_pending": [f"pending_tool{leak}"] if index == 1 else [],
            "command": f"help{leak}" if index % 7 == 0 else "",
            "command_status": f"ok{leak}" if index % 7 == 0 else "",
            "command_success": True if index % 7 == 0 else None,
            "command_result": {"code": 0, "tag": leak or "ok", "raw": {1, 2}},
        }

    def turn_in_history(
        self, *, message: str, conversation_id: str, request_id: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any] | None:
        """Offline history hook; tests can inject a committed turn by request id."""

        if self.history_error is not None:
            raise self.history_error
        payload = self.history.get(request_id)
        return dict(payload) if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Turn and summary construction
# ---------------------------------------------------------------------------


def is_successful_turn(turn: dict[str, Any]) -> bool:
    """True only for a completed turn with answer content and no pending tools.

    An ``incomplete`` status is never clean even when it carries text, and a
    response that still reports pending tool calls is not a finished answer.
    Unknown statuses are not promoted to success either.
    """

    if turn.get("failure"):
        return False
    status = _safe_str(turn.get("status")).strip().lower()
    if status not in SUCCESS_STATUSES:
        return False
    if _safe_list(turn.get("tool_calls_pending")):
        return False
    return bool(_safe_str(turn.get("assistant_content")).strip())


def turn_deficiencies(turn: dict[str, Any]) -> list[str]:
    """Name the observed deficiencies of one turn (empty list means clean)."""

    reasons: list[str] = []
    status = _safe_str(turn.get("status")).strip().lower()
    content = _safe_str(turn.get("assistant_content")).strip()
    if turn.get("failure") or status == "transport_error":
        reasons.append("transport_failure")
    elif status in ERROR_STATUSES:
        reasons.append("error_status")
    if status in BLOCKED_STATUSES:
        reasons.append("blocked_status")
    # Incomplete-with-content is still an observed deficiency: partial text
    # must not be promoted to a successful answer by the floor accounting.
    if status in INCOMPLETE_STATUSES:
        reasons.append("incomplete_status")
        if not content:
            reasons.append("incomplete_without_content")
    elif (
        status not in ERROR_STATUSES | BLOCKED_STATUSES | INCOMPLETE_STATUSES
        and status != "transport_error"
        and not content
    ):
        reasons.append("empty_answer")
    if (
        status
        not in SUCCESS_STATUSES | ERROR_STATUSES | BLOCKED_STATUSES | INCOMPLETE_STATUSES
        and status != "transport_error"
    ):
        reasons.append("unrecognized_status")
    if _safe_list(turn.get("tool_calls_pending")):
        reasons.append("incomplete_with_pending_tools")
    return reasons


def _fallback_turn(
    spec: ScenarioMessage,
    *,
    attempts: int,
    elapsed_seconds: float,
    timestamp: str,
    error: str,
    request_id: str = "",
) -> dict[str, Any]:
    """A complete, JSON-safe turn used when a response cannot be normalized."""

    turn: dict[str, Any] = {
        "index": spec.index,
        "timestamp": _safe_str(timestamp),
        "category": _safe_str(spec.category),
        "user_message": REDACTED,
        "assistant_content": "",
        "status": "transport_error",
        "provider": "",
        "model": "",
        "simulated": None,
        "model_status": "",
        "response_conversation_id": "",
        "request_id": _safe_str(request_id),
        "from_history": False,
        "history_pending": False,
        "history_check_error": "",
        "usage": {},
        "memories_used": [],
        "tool_names": [],
        "tool_results": [],
        "tool_errors": [],
        "tool_calls_pending": [],
        "command": "",
        "command_status": "",
        "command_success": None,
        "command_result": None,
        "attempts": _safe_int(attempts, 1),
        "attempt_history": [],
        "attempt_errors": [],
        "elapsed_seconds": round(_safe_float(elapsed_seconds), 4),
        "failure": True,
        "error_signature": error or "turn: normalization_failure",
        "progress_error": "",
        "deficiencies": ["turn_normalization_failure"],
    }
    return turn


def build_turn(
    *,
    spec: ScenarioMessage,
    payload: dict[str, Any],
    attempts: int,
    elapsed_seconds: float,
    timestamp: str,
    failure: bool = False,
    error_signature_value: str = "",
    attempt_history: list[dict[str, Any]] | None = None,
    progress_error: str = "",
    history_check_error: str = "",
) -> dict[str, Any]:
    """Convert one transport payload (or a failure) into a redacted turn record.

    No single response field is allowed to abort the run. A hostile value
    (broken ``__str__``, cyclic container, NaN usage, ...) falls back to a
    complete failure turn rather than propagating an exception.
    """

    if failure:
        payload = {}
    try:
        history = _safe_list(attempt_history)
        tool_names, tool_records, tool_errors = extract_tool_info(payload.get("tool_results", []))
        simulated = payload.get("simulated")
        if not isinstance(simulated, bool):
            simulated = None
        status = _safe_str(payload.get("status") or ("transport_error" if failure else "unknown"))
        content = "" if failure else _safe_str(payload.get("content") or "")
        error_value = redact(_safe_str(error_signature_value)) if failure else ""
        if not failure and status.strip().lower() in ERROR_STATUSES:
            error_value = error_signature("status", content)

        command_success = payload.get("command_success")
        if not isinstance(command_success, bool):
            command_success = None
        command_result = payload.get("command_result")
        turn: dict[str, Any] = {
            "index": spec.index,
            "timestamp": _safe_str(timestamp),
            "category": _safe_str(spec.category),
            "user_message": redact(_safe_str(spec.text)),
            "assistant_content": redact(content),
            "status": redact(status),
            "provider": redact(_safe_str(payload.get("provider") or "")),
            "model": redact(_safe_str(payload.get("model") or "")),
            "simulated": simulated,
            "model_status": redact(_safe_str(payload.get("model_status") or "")),
            # The server may assign its own id in /chat. Record it so a
            # mismatch with the requested id is visible in the log.
            "response_conversation_id": redact(_safe_str(payload.get("conversation_id") or "")),
            # Idempotency audit trail: same id on every attempt, and a flag
            # when the answer came from durable history after a timeout.
            "request_id": redact(_safe_str(payload.get(REQUEST_ID_FIELD) or "")),
            "from_history": payload.get("from_history") is True,
            "history_pending": payload.get("history_pending") is True,
            "history_check_error": redact(
                _safe_str(payload.get("history_check_error") or history_check_error)
            ),
            "usage": redact(_normalize_usage(payload.get("usage"))),
            "memories_used": redact(_safe_list(payload.get("memories_used"))),
            "tool_names": redact(tool_names),
            "tool_results": redact(tool_records),
            "tool_errors": redact(tool_errors),
            "tool_calls_pending": redact(_safe_list(payload.get("tool_calls_pending"))),
            "command": redact(_safe_str(payload.get("command") or "")),
            "command_status": redact(_safe_str(payload.get("command_status") or "")),
            "command_success": command_success,
            "command_result": redact(_json_safe(command_result)) if command_result is not None else None,
            "attempts": _safe_int(attempts, 1),
            "attempt_history": redact(history),
            "attempt_errors": redact(
                [
                    _safe_str(entry.get("error_signature") or "")
                    for entry in history
                    if isinstance(entry, dict) and entry.get("error_signature")
                ]
            ),
            "elapsed_seconds": round(_safe_float(elapsed_seconds), 4),
            "failure": failure,
            "error_signature": error_value,
            "progress_error": redact(_safe_str(progress_error)),
        }
        turn["deficiencies"] = turn_deficiencies(turn)
        # Redact once more at construction so the returned turns (not just the
        # written files) are safe to hand to callers.
        return redact(turn)
    except Exception as exc:  # noqa: BLE001 - one hostile field must not kill 400 turns
        signature = error_signature("turn", f"{type(exc).__name__}: {exc}")
        return _fallback_turn(
            spec, attempts=attempts, elapsed_seconds=elapsed_seconds,
            timestamp=timestamp, error=signature,
        )


def build_summary(
    turns: list[dict[str, Any]],
    *,
    conversation_id: str = DEFAULT_CONVERSATION_ID,
    dry_run: bool = False,
    messages_requested: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the structured run summary from the recorded turns alone."""

    total = len(turns)
    failures = sum(1 for turn in turns if turn.get("failure"))
    successful = sum(1 for turn in turns if is_successful_turn(turn))
    deficiency_counts: Counter[str] = Counter()
    deficient_turns = 0
    for turn in turns:
        # Authoritative: recompute from the turn facts rather than trusting a
        # possibly stale (or hostile) `deficiencies` field on the turn itself.
        reasons = turn_deficiencies(turn)
        if reasons:
            deficient_turns += 1
        for reason in reasons:
            deficiency_counts[reason] += 1
    tool_error_count = sum(len(turn.get("tool_errors") or []) for turn in turns)
    total_latency = sum(_safe_float(turn.get("elapsed_seconds")) for turn in turns)
    statuses = Counter(_safe_str(turn.get("status") or "unknown") for turn in turns)
    providers = Counter(_safe_str(turn.get("provider") or "") for turn in turns)
    models = Counter(_safe_str(turn.get("model") or "") for turn in turns)
    command_statuses: Counter[str] = Counter(
        _safe_str(turn.get("command_status") or "none") for turn in turns if turn.get("command")
    )
    pair_counts: Counter[tuple[str, str, str]] = Counter()
    simulated_counts = Counter(
        "simulated" if turn.get("simulated") is True else "live" if turn.get("simulated") is False else "unknown"
        for turn in turns
    )
    for turn in turns:
        pair_counts[
            (
                _safe_str(turn.get("provider") or ""),
                _safe_str(turn.get("model") or ""),
                "simulated" if turn.get("simulated") is True else "live" if turn.get("simulated") is False else "unknown",
            )
        ] += 1
    signatures: Counter[str] = Counter()
    for turn in turns:
        if turn.get("failure") and turn.get("error_signature"):
            signatures[_safe_str(turn["error_signature"])] += 1
        for signature in turn.get("tool_errors") or []:
            signatures[_safe_str(signature)] += 1
        if turn.get("error_signature") and not turn.get("failure"):
            signatures[_safe_str(turn["error_signature"])] += 1
        for signature in turn.get("attempt_errors") or []:
            signatures[_safe_str(signature)] += 1
        if turn.get("progress_error"):
            signatures[_safe_str(turn["progress_error"])] += 1

    # Status-based failures are as real as transport failures for the purpose
    # of "did this run produce any answers?". ``failures`` keeps its original
    # meaning (transport failures); ``failed_turns`` also counts error-status
    # turns so an all-error run cannot masquerade as a completed observation.
    error_status_turns = sum(
        1 for turn in turns if _safe_str(turn.get("status")).strip().lower() in ERROR_STATUSES
    )
    failed_turns = sum(
        1
        for turn in turns
        if turn.get("failure") or _safe_str(turn.get("status")).strip().lower() in ERROR_STATUSES
    )
    all_answers_failed = bool(total) and successful == 0 and deficient_turns == total

    requested = messages_requested if messages_requested is not None else total
    # A floor claim requires 400 *requested and recorded* turns and at least
    # 400 answers with content. 400 transport failures is not 400 observations.
    floor_satisfied = (
        isinstance(requested, int)
        and not isinstance(requested, bool)
        and requested >= MIN_MESSAGES
        and total >= MIN_MESSAGES
        and successful >= MIN_MESSAGES
    )
    if total and failures == total:
        verdict = "all_turns_failed"
    elif all_answers_failed:
        verdict = "all_answers_failed"
    elif floor_satisfied and successful >= MIN_MESSAGES and deficient_turns == 0:
        # Clean means 400 completed answers *and* zero observed deficiencies.
        verdict = "floor_satisfied_clean"
    elif floor_satisfied and successful >= MIN_MESSAGES:
        verdict = "floor_satisfied_with_deficiencies"
    else:
        verdict = "below_floor_or_incomplete"

    expected_id = _safe_str(conversation_id)
    id_mismatches = sum(
        1
        for turn in turns
        if turn.get("response_conversation_id") and _safe_str(turn["response_conversation_id"]) != expected_id
    )

    summary: dict[str, Any] = {
        "total_turns": total,
        "successful_turns": successful,
        "floor_satisfied": floor_satisfied,
        "floor_size": MIN_MESSAGES,
        "failures": failures,
        "failed_turns": failed_turns,
        "error_status_turns": error_status_turns,
        "all_answers_failed": all_answers_failed,
        "deficiency_count": deficient_turns,
        "deficiency_distribution": dict(sorted(deficiency_counts.items())),
        # These fields are the authoritative answer to "did we actually observe
        # 400 answers?" and are the ones printed in the stdout summary.
        "run_verdict": verdict,
        "honest_accounting": {
            "recorded_turns": total,
            "successful_turns": successful,
            "failed_turns": failed_turns,
            "transport_failures": failures,
            "error_status_turns": error_status_turns,
            "deficient_turns": deficient_turns,
            "floor_size": MIN_MESSAGES,
            "floor_satisfied": floor_satisfied,
            "all_answers_failed": all_answers_failed,
            "verdict": verdict,
        },
        "conversation_id_mismatches": id_mismatches,
        "tool_error_count": tool_error_count,
        "tool_invocations": sum(len(turn.get("tool_names") or []) for turn in turns),
        "memories_used_count": sum(len(turn.get("memories_used") or []) for turn in turns),
        "memories_used_turns": sum(1 for turn in turns if turn.get("memories_used")),
        "pending_tool_calls": sum(len(turn.get("tool_calls_pending") or []) for turn in turns),
        "pending_tool_call_turns": sum(1 for turn in turns if turn.get("tool_calls_pending")),
        "slash_command_turns": sum(1 for turn in turns if turn.get("command")),
        "command_success_count": sum(1 for turn in turns if turn.get("command") and turn.get("command_success") is True),
        "command_failure_count": sum(1 for turn in turns if turn.get("command") and turn.get("command_success") is False),
        "command_status_distribution": dict(sorted(command_statuses.items())),
        "retry_count": sum(max(0, _safe_int(turn.get("attempts"), 0) - 1) for turn in turns),
        "history_recovered_turns": sum(1 for turn in turns if turn.get("from_history") is True),
        "history_pending_turns": sum(1 for turn in turns if turn.get("history_pending") is True),
        "history_check_error_count": sum(1 for turn in turns if turn.get("history_check_error")),
        "attempt_error_count": sum(len(turn.get("attempt_errors") or []) for turn in turns),
        "progress_error_count": sum(1 for turn in turns if turn.get("progress_error")),
        "average_latency_seconds": round(total_latency / total, 4) if total else 0.0,
        "total_latency_seconds": round(total_latency, 4),
        "status_distribution": dict(sorted(statuses.items())),
        "providers": dict(sorted(providers.items())),
        "models": dict(sorted(models.items())),
        "simulated_counts": {
            key: simulated_counts.get(key, 0) for key in ("simulated", "live", "unknown")
        },
        "provider_model_pairs": [
            {"provider": provider, "model": model, "mode": mode, "turns": count}
            for (provider, model, mode), count in sorted(pair_counts.items())
        ],
        "provider_model_provenance": [
            {"provider": provider, "model": model, "mode": mode, "turns": count}
            for (provider, model, mode), count in sorted(pair_counts.items())
        ],
        "top_error_signatures": [
            {"signature": signature, "count": count}
            for signature, count in signatures.most_common(10)
        ],
        "category_distribution": dict(
            sorted(Counter(_safe_str(turn.get("category") or "unknown") for turn in turns).items())
        ),
        "first_timestamp": _safe_str(turns[0].get("timestamp")) if turns else "",
        "last_timestamp": _safe_str(turns[-1].get("timestamp")) if turns else "",
        "conversation_id": conversation_id,
        "dry_run": dry_run,
        "messages_requested": requested,
        "config": redact(config or {}),
    }
    return redact(summary)


# ---------------------------------------------------------------------------
# The observation loop
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _jsonl_fallback_line(turn: dict[str, Any], exc: BaseException) -> str:
    """A minimal but valid checkpoint line for a turn that cannot be encoded."""

    record = {
        "index": _safe_int(turn.get("index"), 0),
        "timestamp": _safe_str(turn.get("timestamp")),
        "category": _safe_str(turn.get("category")),
        "user_message": REDACTED,
        "assistant_content": "",
        "status": "transport_error",
        "failure": True,
        "deficiencies": ["jsonl_serialization_failure"],
        "error_signature": error_signature("jsonl", f"{type(exc).__name__}: {exc}"),
    }
    return _json_dumps(redact(record))


def _write_jsonl_line(handle: Any, turn: dict[str, Any]) -> None:
    """Write one checkpoint line; a serialization error never stops the run."""

    try:
        line = _json_dumps(redact(turn))
        parsed = json.loads(line)
        expected = turn.get("index") if isinstance(turn, dict) else None
        if not isinstance(parsed, dict) or parsed.get("index") != expected:
            raise ValueError("jsonl line lost its turn index")
    except Exception as exc:  # noqa: BLE001
        line = _jsonl_fallback_line(turn, exc)
    handle.write(line + "\n")
    handle.flush()


def run_conversation(
    transport: ChatTransport,
    *,
    messages: int = DEFAULT_MESSAGES,
    allow_short: bool = False,
    seed: int = DEFAULT_SEED,
    conversation_id: str = DEFAULT_CONVERSATION_ID,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    jsonl_path: Path | str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Send every scenario message sequentially and return all turn records.

    Failures are caught per attempt; after ``retries`` retries the turn is
    recorded as a failure and the loop continues with the next message. Every
    attempt keeps its own normalized error signature. Retries carry the same
    ``request_id``; before resending, a transport that implements
    ``turn_in_history`` is asked whether the server already committed the turn,
    and its stored reply is adopted instead of duplicating the chat rows. If
    that history check itself fails, the retry is suppressed (a duplicate row
    is worse than one recorded failed turn). If ``jsonl_path`` is given, each
    turn is appended immediately (checkpointing a long run even if it is
    interrupted). A broken progress callback is recorded but never aborts the
    observation.
    """

    validate_message_count(messages, allow_short=allow_short)
    if retries < 0:
        raise ValueError("retries must be >= 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    scenario = generate_scenario(messages, seed=seed, allow_short=allow_short)
    accepts_request_id = _supports_keyword(transport.send, REQUEST_ID_FIELD)
    history_checker = getattr(transport, "turn_in_history", None)
    has_history_check = callable(history_checker) and _supports_keyword(
        history_checker, REQUEST_ID_FIELD
    )

    handle = None
    if jsonl_path is not None:
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="\n")

    turns: list[dict[str, Any]] = []
    try:
        for spec in scenario:
            started = _timestamp()
            began = time.perf_counter()
            payload: dict[str, Any] | None = None
            request_id = request_id_for_turn(conversation_id, spec.index, seed)
            history_error = ""
            # Count actual POSTs, not history reads, so a suppressed retry is
            # never reported as a second attempt.
            attempts = 0
            last_error = ""
            attempt_history: list[dict[str, Any]] = []
            while attempts <= retries:
                attempt_began = time.perf_counter()
                record: dict[str, Any] = {
                    "attempt": attempts + 1,
                    "error_signature": "",
                    "elapsed_seconds": 0.0,
                    "from_history": False,
                    "history_check_error": "",
                }
                attempt_history.append(record)
                if attempts > 0 and has_history_check:
                    # The first attempt may have reached the server before the
                    # client timed out. Never resend without checking durable
                    # history first.
                    try:
                        prior = history_checker(
                            message=spec.text,
                            conversation_id=conversation_id,
                            request_id=request_id,
                            timeout_seconds=timeout_seconds,
                        )
                        if isinstance(prior, dict):
                            payload = dict(prior)
                            payload.setdefault(REQUEST_ID_FIELD, request_id)
                            record["from_history"] = True
                            break
                    except Exception as exc:  # noqa: BLE001 - an opaque history is not permission to resend
                        history_error = error_signature(
                            "history", f"{type(exc).__name__}: {_safe_str(exc, '<unprintable>')}"
                        )
                        record["history_check_error"] = history_error
                        last_error = history_error
                        break
                attempts += 1
                try:
                    if accepts_request_id:
                        candidate = transport.send(
                            message=spec.text,
                            conversation_id=conversation_id,
                            timeout_seconds=timeout_seconds,
                            request_id=request_id,
                        )
                    else:
                        candidate = transport.send(
                            message=spec.text,
                            conversation_id=conversation_id,
                            timeout_seconds=timeout_seconds,
                        )
                    if not isinstance(candidate, dict):
                        raise TransportError("invalid response: expected a JSON object")
                    payload = candidate
                    break
                except Exception as exc:  # one bad turn must not abort the run
                    last_error = error_signature(
                        "transport", f"{type(exc).__name__}: {_safe_str(exc, '<unprintable>')}"
                    )
                    record["error_signature"] = last_error
                    if attempts <= retries and retry_backoff_seconds > 0:
                        sleep(retry_backoff_seconds)
                finally:
                    record["elapsed_seconds"] = round(time.perf_counter() - attempt_began, 4)
            elapsed = time.perf_counter() - began
            progress_error = ""
            try:
                turn = build_turn(
                    spec=spec,
                    payload=payload if payload is not None else {},
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    timestamp=started,
                    failure=payload is None,
                    error_signature_value=last_error,
                    attempt_history=attempt_history,
                    history_check_error=history_error,
                )
            except Exception as exc:  # noqa: BLE001 - belt and braces
                turn = _fallback_turn(
                    spec, attempts=attempts, elapsed_seconds=elapsed, timestamp=started,
                    error=error_signature("turn", f"{type(exc).__name__}: {exc}"),
                    request_id=request_id,
                )
            if progress is not None:
                try:
                    progress(turn)
                except Exception as exc:  # a UI callback must not kill a 400-turn run
                    progress_error = error_signature("progress", f"{type(exc).__name__}: {exc}")
                    turn["progress_error"] = progress_error
            try:
                turns.append(turn)
                if handle is not None:
                    _write_jsonl_line(handle, turn)
            except Exception as exc:  # noqa: BLE001 - checkpointing is best effort
                # The in-memory log is the source of truth; a broken handle is
                # reported by the end-of-run completeness check, not thrown.
                turns[-1:] = [
                    _fallback_turn(
                        spec, attempts=attempts, elapsed_seconds=elapsed, timestamp=started,
                        error=error_signature("jsonl", f"{type(exc).__name__}: {exc}"),
                        request_id=request_id,
                    )
                ]
    finally:
        if handle is not None:
            handle.close()
    return turns


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_jsonl(turns: list[dict[str, Any]], path: Path | str) -> Path:
    """Write the complete machine-readable turn log."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for turn in turns:
            _write_jsonl_line(handle, turn)
    return destination


def _md(value: Any) -> str:
    """Safe, redacted interpolation for Markdown."""

    return _safe_str(redact(value), REDACTED)


def _md_json(value: Any) -> str:
    return _json_dumps(redact(value))


def _markdown_block(value: Any) -> str:
    """Quote arbitrary text as markdown without leaking newlines into layout."""

    text = _md(value).splitlines() or [""]
    return "\n".join("> " + line for line in text)


def write_markdown(
    turns: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path | str,
) -> Path:
    """Write the human-readable transcript with a summary header."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = redact(summary)
    dry_run = bool(summary.get("dry_run"))
    scratch_line = (
        "- Scratch writes: SIMULATED (dry run; no scratch file was created)"
        if dry_run
        else "- Scratch writes: live (see the tool-result path check below)"
    )
    lines = [
        "# LEO conversation observation transcript",
        "",
        f"- Conversation id: `{_md(summary.get('conversation_id', ''))}`",
        f"- Turns: {_md(summary.get('total_turns', 0))}",
        f"- Successful turns: {_md(summary.get('successful_turns', 0))}",
        f"- Observation floor satisfied: {_md(summary.get('floor_satisfied', False))}",
        f"- Run verdict: {_md(summary.get('run_verdict', 'unknown'))}",
        f"- Failures: {_md(summary.get('failures', 0))}",
        f"- Failed turns (transport or error status): {_md(summary.get('failed_turns', 0))}",
        f"- Error-status turns: {_md(summary.get('error_status_turns', 0))}",
        f"- All answers failed: {_md(summary.get('all_answers_failed', False))}",
        f"- Deficiencies: {_md(summary.get('deficiency_count', 0))} {_md(summary.get('deficiency_distribution', {}))}",
        f"- Tool errors: {_md(summary.get('tool_error_count', 0))}",
        f"- Pending tool calls: {_md(summary.get('pending_tool_calls', 0))}",
        f"- Slash-command turns: {_md(summary.get('slash_command_turns', 0))}",
        f"- Average latency (s): {_md(summary.get('average_latency_seconds', 0))}",
        f"- Dry run: {dry_run}",
        scratch_line,
        f"- Status distribution: `{_md_json(summary.get('status_distribution', {}))}`",
        f"- Provider/model provenance: `{_md_json(summary.get('provider_model_pairs', []))}`",
    ]
    path_check = summary.get("tool_result_paths") or {}
    if path_check:
        lines.append(
            f"- Tool-result path check: {'clean' if path_check.get('clean') else 'ISSUES: ' + _md(path_check.get('issues'))}"
        )
    lines.extend(["", "## Top error signatures", ""])
    signatures = summary.get("top_error_signatures") or []
    if signatures:
        for entry in signatures:
            lines.append(f"- {_md(entry.get('count'))}x {_md(entry.get('signature'))}")
    else:
        lines.append("- None recorded.")
    for original in turns:
        turn = redact(original)  # defensive: every writer redacts again
        lines.extend(
            [
                "",
                f"## Turn {_md(turn.get('index'))} - {_md(turn.get('category'))} - {_md(turn.get('status'))}",
                "",
                f"- Timestamp: {_md(turn.get('timestamp'))}",
                f"- Elapsed (s): {_md(turn.get('elapsed_seconds'))}",
                f"- Attempts: {_md(turn.get('attempts'))}",
                f"- Provider/model: {_md(turn.get('provider'))} / {_md(turn.get('model'))}",
                f"- Simulated: {_md(turn.get('simulated'))} ({_md(turn.get('model_status'))})",
                f"- Response conversation id: {_md(turn.get('response_conversation_id'))}",
                f"- Request id: {_md(turn.get('request_id'))}"
                + (" (recovered from durable history)" if turn.get("from_history") else ""),
                f"- Tools: {', '.join(_md(name) for name in (turn.get('tool_names') or [])) or 'none'}",
                f"- Pending tool calls: {_md(turn.get('tool_calls_pending'))}",
                f"- Memories used: {_md(turn.get('memories_used'))}",
                f"- Deficiencies: {_md(turn.get('deficiencies'))}",
            ]
        )
        if turn.get("command"):
            lines.extend(
                [
                    f"- Command: {_md(turn.get('command'))} (status {_md(turn.get('command_status'))}, success {_md(turn.get('command_success'))})",
                    f"- Command result: `{_md_json(turn.get('command_result'))}`",
                ]
            )
        if turn.get("attempt_history"):
            lines.append(
                "- Attempt errors: "
                + _md_json(
                    [
                        entry.get("error_signature")
                        for entry in turn["attempt_history"]
                        if isinstance(entry, dict)
                    ]
                )
            )
        if turn.get("failure"):
            lines.append(f"- Failure signature: {_md(turn.get('error_signature'))}")
        lines.extend(["", "**User:**", "", _markdown_block(turn.get("user_message"))])
        answer = turn.get("assistant_content") or (
            "[no assistant content - transport failure]" if turn.get("failure") else "[empty answer]"
        )
        lines.extend(["", "**Assistant:**", "", _markdown_block(answer)])
        for record in turn.get("tool_results") or []:
            if not isinstance(record, dict):
                continue
            lines.extend(
                [
                    "",
                    f"**Tool result - {_md(record.get('name'))}:**",
                    "",
                    _markdown_block(record.get("result")),
                ]
            )
        for signature in turn.get("tool_errors") or []:
            lines.append(f"- Tool error: {_md(signature)}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _summary_fallback(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep the authoritative fields even if an exotic summary value breaks JSON."""

    return {
        "total_turns": _safe_int(summary.get("total_turns")),
        "successful_turns": _safe_int(summary.get("successful_turns")),
        "floor_satisfied": bool(summary.get("floor_satisfied")),
        "deficiency_count": _safe_int(summary.get("deficiency_count")),
        "run_verdict": _safe_str(summary.get("run_verdict"), "all_turns_failed"),
        "honest_accounting": _json_safe(summary.get("honest_accounting") or {}),
        "error": "summary_serialization_fallback",
    }


def write_summary(summary: dict[str, Any], path: Path | str) -> Path:
    """Write the structured summary as strictly valid, redacted JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(
            _sanitize_non_finite(redact(summary)), ensure_ascii=False, indent=2,
            sort_keys=True, allow_nan=False, default=_safe_str,
        )
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or "total_turns" not in parsed:
            raise ValueError("summary lost its authoritative fields")
    except Exception:  # noqa: BLE001
        text = _json_dumps(redact(_summary_fallback(summary)))
    destination.write_text(text + "\n", encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Post-run checks
# ---------------------------------------------------------------------------

# NUL and other control bytes are never valid in a filesystem path. Binary
# search output (e.g. a `.pyc`) can contain them between path-like fragments;
# letting them into ``Path.resolve()`` raises ValueError and used to kill all
# post-run reporting.
_CONTROL_BYTES_RE = re.compile(r"[\x00-\x1f\x7f]")
_PATH_CLAIM_RE = re.compile(
    r"(?:(?:artifacts/observation/scratch|/Users/[^/\x00-\x1f\x7f\s]+"
    r"|/home/[^/\x00-\x1f\x7f\s]+|/tmp)/[^\x00-\x1f\x7f\s\"'`,)\]]+)"
)


def check_tool_result_paths(
    turns: list[dict[str, Any]],
    *,
    out_dir: Path | str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Audit paths claimed in tool results after the run.

    Claims outside the workspace are flagged. Missing scratch files are flagged
    only for a live run; a dry run explicitly simulates scratch writes and is
    expected to leave no files behind.
    """

    try:
        directory = Path(out_dir).resolve()
        workspace = REPO_ROOT.resolve()
    except (OSError, RuntimeError, ValueError):
        directory = Path(out_dir)
        workspace = REPO_ROOT
    scratch_dir = directory / "scratch"
    claims: list[dict[str, Any]] = []
    issues: list[str] = []
    for turn in _safe_list(turns):
        if not isinstance(turn, dict):
            continue
        for record in _safe_list(turn.get("tool_results")):
            if not isinstance(record, dict):
                continue
            raw = _safe_str(record.get("result") or "")
            candidates: list[str] = []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("path") is not None:
                    candidates.append(_safe_str(parsed["path"]))
            except (TypeError, ValueError):
                pass
            candidates.extend(_PATH_CLAIM_RE.findall(raw))
            # A JSON "path" key and the text regex often find the same claim.
            for candidate in dict.fromkeys(candidates):
                candidate = _safe_str(redact(candidate))
                if not candidate:
                    continue
                if _CONTROL_BYTES_RE.search(candidate):
                    # Never hand NUL/control bytes to Path: resolve()/exists()
                    # raise ValueError. Report an escaped form instead.
                    issues.append(
                        f"turn {turn.get('index')}: tool result path contains control bytes: "
                        f"{ascii(candidate)[:160]}"
                    )
                    continue
                try:
                    candidate_path = Path(candidate)
                    resolved = (
                        candidate_path if candidate_path.is_absolute() else REPO_ROOT / candidate_path
                    ).resolve()
                    in_scratch = resolved == scratch_dir or scratch_dir in resolved.parents
                    in_workspace = resolved == workspace or workspace in resolved.parents
                    location = (
                        "scratch" if in_scratch else "workspace" if in_workspace else "outside_workspace"
                    )
                    exists = resolved.exists()
                except (OSError, RuntimeError, ValueError) as exc:
                    issues.append(
                        f"turn {turn.get('index')}: unreadable tool result path "
                        f"{ascii(candidate)[:120]}: {error_signature('path', exc)}"
                    )
                    continue
                claims.append(
                    {
                        "turn": turn.get("index"),
                        "tool": redact(record.get("name")),
                        "path": candidate,
                        "location": location,
                        "exists": exists,
                        "simulated": dry_run,
                    }
                )
                if location == "outside_workspace":
                    issues.append(f"turn {turn.get('index')}: tool result path outside workspace: {candidate}")
                elif in_scratch and not dry_run and not exists:
                    issues.append(f"turn {turn.get('index')}: claimed scratch file is missing: {candidate}")
    report: dict[str, Any] = {
        "scratch_directory": str(scratch_dir),
        "scratch_directory_created": scratch_dir.is_dir(),
        "simulated": dry_run,
        "claims": claims[:MAX_REPORT_CLAIMS],
        "claim_count": len(claims),
        "issues": issues[:MAX_REPORT_CLAIMS],
        "issue_count": len(issues),
        "clean": not issues,
    }
    return redact(report)


def check_output_completeness(
    paths: dict[str, Path | str],
    turns: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Verify that all three output files exist and contain the whole run."""

    issues: list[str] = []
    files: dict[str, Any] = {}
    for label, raw_path in paths.items():
        path = Path(raw_path)
        entry: dict[str, Any] = {"path": str(path), "exists": False, "bytes": 0}
        try:
            entry["exists"] = path.is_file()
            if entry["exists"]:
                entry["bytes"] = path.stat().st_size
                text = path.read_text(encoding="utf-8")
                if not text.strip():
                    issues.append(f"{label}: file is empty")
                if label == "jsonl":
                    lines = [line for line in text.splitlines() if line.strip()]
                    entry["lines"] = len(lines)
                    entry["expected_lines"] = len(turns)
                    if len(lines) != len(turns):
                        issues.append(f"jsonl: {len(lines)} lines for {len(turns)} turns")
                    for number, line in enumerate(lines[:MAX_REPORT_CLAIMS], start=1):
                        try:
                            record = json.loads(line)
                        except ValueError:
                            issues.append(f"jsonl: line {number} is not valid JSON")
                            break
                        if not isinstance(record, dict) or record.get("index") != number:
                            issues.append(f"jsonl: line {number} has an unexpected record shape")
                            break
                elif label == "markdown":
                    entry["headings"] = sum(
                        1 for turn in turns if f"## Turn {turn.get('index')} - " in text
                    )
                    entry["expected_headings"] = len(turns)
                    if entry["headings"] != len(turns):
                        issues.append(f"markdown: {entry['headings']} turn headings for {len(turns)} turns")
                    if not text.startswith("# LEO conversation observation transcript"):
                        issues.append("markdown: missing transcript header")
                elif label == "summary":
                    try:
                        written = json.loads(text)
                    except ValueError:
                        issues.append("summary: file is not valid JSON")
                    else:
                        entry["total_turns"] = written.get("total_turns")
                        if written.get("total_turns") != len(turns):
                            issues.append("summary: total_turns does not match the turn log")
                        for field in (
                            "successful_turns", "floor_satisfied", "deficiency_count",
                            "run_verdict", "honest_accounting",
                        ):
                            if field not in written:
                                issues.append(f"summary: missing authoritative field {field}")
        except OSError as exc:
            issues.append(f"{label}: unreadable: {error_signature('os', exc)}")
        files[label] = entry
        if not entry["exists"]:
            issues.append(f"{label}: file was not created")
    report: dict[str, Any] = {
        "complete": not issues,
        "files": files,
        "issues": issues[:MAX_REPORT_CLAIMS],
        "issue_count": len(issues),
    }
    return redact(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv_indices(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observe a running LEO server with a 400+ message human-like conversation.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Chat endpoint (default: {DEFAULT_URL}).")
    parser.add_argument(
        "--conversation-id", default=DEFAULT_CONVERSATION_ID,
        help=f"Fixed conversation id used for every turn (default: {DEFAULT_CONVERSATION_ID}).",
    )
    parser.add_argument(
        "--messages", type=int, default=DEFAULT_MESSAGES,
        help=f"Number of user messages; floor is {MIN_MESSAGES} (default: {DEFAULT_MESSAGES}).",
    )
    parser.add_argument(
        "--allow-short", action="store_true",
        help="Waive the 400-message floor. Only accepted together with --dry-run.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic scenario seed.")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries", type=int, default=DEFAULT_RETRIES,
        help="Retries per failed turn before recording a failure (default: 2).",
    )
    parser.add_argument(
        "--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Seconds to wait between attempts (default: 0.5).",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="Output directory for JSONL, Markdown and summary JSON.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use the deterministic in-process fake transport; never open a socket.",
    )
    parser.add_argument(
        "--dry-run-fail-at", default="",
        help="Comma-separated 1-based turn indices at which the fake transport fails.",
    )
    parser.add_argument(
        "--dry-run-tool-errors-at", default="",
        help="Comma-separated 1-based turn indices at which the fake transport returns a tool error.",
    )
    parser.add_argument(
        "--dry-run-leak-token", action="store_true",
        help="Inject a fake token-shaped string in turn 1 to exercise redaction.",
    )
    parser.add_argument(
        "--dry-run-leak-metadata", default="",
        help="Inject this token into every fake response metadata field to exercise redaction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.allow_short and not args.dry_run:
            raise ValueError("--allow-short is only allowed with --dry-run; a live run must send 400 messages")
        validate_message_count(args.messages, allow_short=args.allow_short)
        if args.retries < 0:
            raise ValueError("--retries must be >= 0")
        if args.timeout <= 0:
            raise ValueError("--timeout must be > 0")
        if args.retry_backoff < 0:
            raise ValueError("--retry-backoff must be >= 0")
        fail_at = _csv_indices(args.dry_run_fail_at)
        tool_errors_at = _csv_indices(args.dry_run_tool_errors_at)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        transport: ChatTransport = FakeTransport(
            failure_indices=fail_at,
            tool_error_indices=tool_errors_at,
            statuses=("completed", "incomplete", "completed", "completed"),
            leak_token=args.dry_run_leak_token,
            metadata_leak_token=args.dry_run_leak_metadata,
            dry_run=True,
        )
    else:
        transport = HttpChatTransport(args.url)
    if isinstance(transport, FakeTransport):
        # Injected/offline transports should report claims against this run's
        # scratch directory, not the default repo-relative path.
        transport.scratch_path = str(scratch_dir / "note.md")

    config = {
        # The live transport receives the raw URL; only this logged copy is
        # sanitized (no userinfo, query or fragment).
        "url": sanitize_url(args.url),
        "transport": "fake" if args.dry_run else "http",
        "conversation_id": args.conversation_id,
        "messages": args.messages,
        "seed": args.seed,
        "timeout_seconds": args.timeout,
        "retries": args.retries,
        "allow_short": bool(args.allow_short),
        "dry_run": bool(args.dry_run),
        "scratch_dir": str(scratch_dir),
    }
    started = _timestamp()
    turns = run_conversation(
        transport,
        messages=args.messages,
        allow_short=args.allow_short,
        seed=args.seed,
        conversation_id=args.conversation_id,
        timeout_seconds=args.timeout,
        retries=args.retries,
        retry_backoff_seconds=args.retry_backoff,
        jsonl_path=out_dir / "conversation.jsonl",
    )
    finished = _timestamp()
    jsonl_path = out_dir / "conversation.jsonl"
    markdown_path = out_dir / "conversation.md"
    summary_path = out_dir / "summary.json"
    try:
        summary = build_summary(
            turns,
            conversation_id=args.conversation_id,
            dry_run=bool(args.dry_run),
            messages_requested=args.messages,
            config={**config, "started_at": started, "finished_at": finished},
        )
    except Exception as exc:  # noqa: BLE001 - the recorded turns still deserve a report
        summary = _summary_fallback({"total_turns": len(turns)})
        summary["error"] = error_signature("summary", f"{type(exc).__name__}: {exc}")

    reporting_issues: list[str] = []
    # Post-run reporting is best effort. A checker that raises (or a path claim
    # with control bytes) must not lose the transcript or the authoritative
    # summary after a 400-turn run. The finally block guarantees both files.
    try:
        try:
            summary["tool_result_paths"] = check_tool_result_paths(
                turns, out_dir=out_dir, dry_run=bool(args.dry_run)
            )
        except Exception as exc:  # noqa: BLE001
            signature = error_signature("post_run:tool_result_paths", f"{type(exc).__name__}: {exc}")
            reporting_issues.append(signature)
            summary["tool_result_paths"] = {
                "clean": False, "claims": [], "claim_count": 0,
                "issues": [signature], "issue_count": 1, "error": signature,
            }
        try:
            write_jsonl(turns, jsonl_path)
        except Exception as exc:  # noqa: BLE001
            reporting_issues.append(error_signature("post_run:jsonl", f"{type(exc).__name__}: {exc}"))
        try:
            write_markdown(turns, summary, markdown_path)
        except Exception as exc:  # noqa: BLE001
            reporting_issues.append(error_signature("post_run:markdown", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            if not markdown_path.is_file():
                write_markdown(turns, summary, markdown_path)
        except Exception as exc:  # noqa: BLE001
            reporting_issues.append(
                error_signature("post_run:markdown_fallback", f"{type(exc).__name__}: {exc}")
            )
        try:
            write_summary(summary, summary_path)
        except Exception as exc:  # noqa: BLE001
            reporting_issues.append(error_signature("post_run:summary", f"{type(exc).__name__}: {exc}"))

    try:
        report = check_output_completeness(
            {"jsonl": jsonl_path, "markdown": markdown_path, "summary": summary_path},
            turns,
            summary,
        )
    except Exception as exc:  # noqa: BLE001
        signature = error_signature("post_run:completeness", f"{type(exc).__name__}: {exc}")
        reporting_issues.append(signature)
        report = {"complete": False, "files": {}, "issues": [signature], "issue_count": 1}
    if reporting_issues:
        issues = reporting_issues + [str(issue) for issue in report.get("issues", [])]
        report = {**report, "complete": False, "issues": issues, "issue_count": len(issues)}
    summary["output_check"] = report
    try:
        write_summary(summary, summary_path)
    except Exception as exc:  # noqa: BLE001 - an earlier valid summary still exists
        reporting_issues.append(error_signature("post_run:summary_final", f"{type(exc).__name__}: {exc}"))
    # stdout is another output surface: redact and strictly encode it, and let
    # the authoritative success/floor fields speak for the whole run.
    print(_json_dumps(redact(summary)))
    if not report["complete"]:
        print(
            "observation run finished but output completeness check failed: "
            + "; ".join(str(issue) for issue in report["issues"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
