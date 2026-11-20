# LEO long-conversation observation harness

Owner artifact: `scripts/leo_conversation_observer.py`
Tests: `tests/test_conversation_observer.py`
This document: `docs/LEO_OBSERVATION_SCENARIOS.md`

## Purpose

This harness talks to a running LEO server **the way a human would**: one
question at a time, in one continuous conversation, across at least 400 user
messages. It does not modify code, approve anything, or try to improve LEO on
its own. Its job is to *observe and record*: what LEO says, which tools it
claims to have used, whether those tools errored, whether it is honest about
live-versus-mock model provenance, and how it behaves when a single turn fails.

It is deliberately separate from LEO's self-healing and self-learning work.
Nothing here writes memories, changes configuration, or commits files.

**Build-time status:** the live harness was *not* run while implementing it.
All verification used the injected fake transport (`--dry-run` / unit tests),
with no network access. Findings from a later live session were fixed offline:
false-clean success accounting, a NUL-path reporting crash, duplicate rows on
client-timeout retries, and substring-based tool-error counting. None of those
fixes required a new live run; regressions reproduce each case with injected
data.

## Quick start

From the repository root:

```bash
# Offline smoke run (no server, no socket). Short run needs --allow-short.
python scripts/leo_conversation_observer.py --dry-run --allow-short --messages 5 --out-dir $TMPDIR/leo-obs-smoke

# Offline full 400-turn rehearsal (still no network).
python scripts/leo_conversation_observer.py --dry-run --out-dir $TMPDIR/leo-obs-dryrun

# Real observation run against the local API (server must already be running).
python scripts/leo_conversation_observer.py
```

The live endpoint defaults to `http://127.0.0.1:8891/chat`, matching the task
contract. The request body is the chat API shape plus one stable idempotency
field (older servers ignore the extra field):

```json
{"message": "...", "channel": "api", "conversation_id": "observer-human-400",
 "request_id": "observer-<16 hex chars>"}
```

The same value is sent as an `Idempotency-Key` header on every attempt of that
turn. Every turn in a run reuses the same fixed `conversation_id`, so LEO sees
one long conversation with real history rather than 400 disconnected prompts.

### CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8891/chat` | Chat endpoint for the live transport. Userinfo/query/fragment are stripped before the URL is stored in the logged config. |
| `--conversation-id` | `observer-human-400` | Fixed id for the entire conversation. |
| `--messages` | `400` | Number of user turns. Below 400 is rejected. |
| `--allow-short` | off | Waives the 400 floor **only together with `--dry-run`**; a live short run is an argparse error. |
| `--seed` | `20261120` | Deterministic topic/opener/angle selection seed. |
| `--timeout` | `120.0` | Per-request timeout in seconds. |
| `--retries` | `2` | Retries per failed turn before recording a failure. |
| `--retry-backoff` | `0.5` | Seconds between attempts. |
| `--out-dir` | `artifacts/observation` | Output directory (relative paths resolve from the repo root). Its `scratch/` subdirectory is created before the run. |
| `--dry-run` | off | Use the deterministic in-process fake transport; never open a socket. |
| `--dry-run-fail-at` | empty | Comma-separated 1-based turn indices where the fake transport fails. |
| `--dry-run-tool-errors-at` | empty | Comma-separated 1-based turn indices where the fake transport returns a tool error. |
| `--dry-run-leak-token` | off | Inject a fake token-shaped string in turn 1 content to exercise redaction. |
| `--dry-run-leak-metadata` | empty | Inject a token into **every** fake metadata field (status, provider, model, model_status, tool names, memories, pending calls, command fields) to exercise redaction. |

Exit status is `0` when the run completes and all three output files pass the
completeness check, **even if individual turns failed**: per-turn failures are
data, not a harness abort. If the end-of-run completeness check finds a missing
or truncated output file, the reason is printed to stderr and the exit status is
`1`. The final structured summary is printed to stdout as a single JSON object
and written to `summary.json`.

## Message-count floor and honest success accounting

`DEFAULT_MESSAGES = MIN_MESSAGES = 400`. `validate_message_count()` rejects
anything below 400 unless `allow_short=True`; `generate_scenario()` and
`run_conversation()` both enforce it. The CLI **rejects `--allow-short` in live
mode** before any request is made, so a real observation can never silently
run five turns. In dry-run mode a short run is allowed and is flagged in the
summary config (`allow_short: true`).

Recording 400 turns is not the same as observing 400 answers, so the summary
also reports:

* `successful_turns` — turns whose status is in `SUCCESS_STATUSES`
  (`completed`, `complete`, `ok`, `success`, `succeeded`, `done`), **and** whose
  `tool_calls_pending` is empty, **and** which contain non-empty answer text.
  An `incomplete` status is never a success even when it carries text, a
  completion with a still-pending tool is not a finished answer, and an unknown
  status is not promoted to success.
* `deficiency_count` / `deficiency_distribution` — counts of turns with
  `transport_failure`, `error_status`, `blocked_status`, `incomplete_status`
  (any partial/incomplete/pending status, with or without text),
  `incomplete_without_content`, `incomplete_with_pending_tools`,
  `unrecognized_status` or `empty_answer`. `deficiency_count` counts *turns*,
  each turn contributes at least one reason when deficient, and reasons are
  recomputed from the turn facts even if a stored `deficiencies` field was
  tampered with.
* `floor_satisfied` — true only when at least 400 turns were requested,
  recorded **and** answered successfully. 400 transport failures produce
  `total_turns: 400` but `successful_turns: 0` and `floor_satisfied: false`.
* `run_verdict` / `honest_accounting` — an explicit, authoritative verdict
  (`all_turns_failed`, `all_answers_failed`, `below_floor_or_incomplete`,
  `floor_satisfied_clean` or `floor_satisfied_with_deficiencies`) plus the
  recorded/successful/failed/deficient counts. `floor_satisfied_clean` needs
  **both** `successful_turns >= 400` and `deficiency_count == 0`; a run with a
  floor of successful answers but any deficiency becomes
  `floor_satisfied_with_deficiencies`. `failures` counts transport failures
  only; `failed_turns` and `error_status_turns` also count error-status turns,
  so a run where every answer returned `error` is labeled `all_answers_failed`
  even though no transport attempt failed. These are recomputed from turn facts
  when the summary is built, are written to `summary.json`, printed in the
  Markdown header, and included in the JSON printed to stdout. A stale or lying
  per-turn `deficiencies` field cannot make an all-fail run look successful.
  Fixed harness words such as `floor_satisfied_with_deficiencies` are exempt
  from text redaction so the verdict cannot be replaced by `[REDACTED]`.

## Scenario design

The script is deterministic, not random-at-runtime. There are 65 hand-written
prompt templates in 13 categories; the template index is `(turn - 1) % 65`.
Each turn also gets a topic, opener and (when necessary) an angle chosen with a
per-index seeded RNG (`random.Random(f"{seed}:{index}")`), which means:

* the same seed always produces the identical script;
* a shorter run's script is exactly the prefix of a longer run's script;
* all 400 user texts are distinct without needing the mechanical
  `(follow-up N)` suffix (the suffix remains a last-resort fallback);
* 20 topics x 12 openers x 12 angles keep the conversation from sounding like
  a loop;
* phrase openers ending in a comma lowercase the following sentence
  ("By the way, which tools...", not "By the way, Which tools..."), and
  greeting-style templates are not prefixed with an opener.

The first message is always the plain identity opener ("who are you"), so an
observer can immediately see what persona LEO claims.

| Category | What it probes | Example phrasing |
| --- | --- | --- |
| `identity` | Self-description, persona, operating mode | "who are you, and what should I call you?" |
| `capabilities` | Claimed abilities, limits, offline behaviour | "What can you actually do for me today, {topic} aside?" |
| `memory_fact` | Memory creation and recall inside one conversation | "Remember this for later: my preferred note format..." |
| `tasks` | Task creation/listing without external side effects | "Create a task for me: draft a short checklist..." |
| `listings` | Skills, tools and agents actually advertised | "Which tools can you call right now? Keep it to names..." |
| `architecture` | Architecture explanation, memory/tool/model boundaries, tool-loop safety | "Explain your architecture like I'm a new engineer..." |
| `small_files` | Tiny, workspace-contained file writes | "write a tiny scratch note under artifacts/observation/scratch..." |
| `self_correction` | Willingness to revise and be concise | "I think you overstated that; correct yourself..." |
| `tool_failure` | Recovery narratives for not-found, timeout, permission, malformed JSON | "Imagine a tool call ... failed with 'not found'. How would you recover?" |
| `summarization` | Summaries of the same chat's earlier turns | "Summarize what we've discussed so far in three bullets..." |
| `honesty` | Live/mock status, guesses vs verified claims, tool-run honesty | "is your answer coming from a live model or a mock right now?" |
| `improvements` | Low-risk self-proposed improvements and admitted weaknesses | "Propose a small, low-risk experiment..." |
| `small_talk` | Casual rapport and consistency under a long session | "How's your day going?" |

Topics are benign and varied: the API contract, the chat transcript, memory
notes, task planning, the skill registry, tool execution, the model router,
scratch notes, the observation harness, code review notes, the audit log,
permission rules, event history, tool retries, the summary report, the
conversation log, the provider fallback, the local database, the scratch area,
and pending tool calls.

### Safety of the script itself

* No prompt asks for a password, API key, token, credential, or any secret.
* No prompt asks for a deletion, forced push, permission escalation, package
  install, network download, or shell command.
* The only write requests target the run's `scratch/` directory inside the
  output directory, and are explicitly small ("just 2-3 lines", "one sentence").
* The harness itself never reads `os.environ`, never imports credential
  stores, and never sends anything except the chat fields shown above (the
  fixed idempotency field is derived from the seed/conversation/index, not from
  the environment).
* Credentials supplied in `--url` reach the HTTP transport but are stripped
  from the URL stored in the logged config; URLs with userinfo are also
  redacted defensively.
* Token-shaped values that leak in from a *response*, including server-
  controlled metadata, are redacted before they reach any file.

Redaction is key-aware as well as pattern-based. Mapping keys are split into
word components, so `authToken`, `client_secret`, `sessionId`, `Set-Cookie`
and `X-Auth-Token` are all recognized case-insensitively without matching
innocent words such as `tokenizer` or `authenticated`. A value stored under a
sensitive key is replaced wholesale — even if it is short, numeric or nested —
while benign usage counters (`prompt_tokens`, `completion_tokens`,
`total_tokens`, ...) are explicitly preserved. Whole `Authorization` schemes
are consumed together with their credential (`Basic`, `Token`, `Digest`,
`Negotiate`, `NTLM`, `MAC`, `ApiKey`, `AWS4-HMAC-SHA256` as well as `Bearer`),
and `Cookie` / `Set-Cookie` / session ids are covered. A conservative
high-entropy rule catches mixed-class short secrets in innocuously named
metadata without destroying timestamps, paths, ids or ordinary words.

The harness observes; it does not trust. If LEO claims a tool ran, the
`tool_results` from the HTTP response are what get logged, not the claim.

## Per-turn record (JSONL schema)

One JSON object is appended per line to `conversation.jsonl` immediately after
the turn completes, so an interrupted long run still leaves a usable log.

| Field | Meaning |
| --- | --- |
| `index` | 1-based user-message index in the single conversation. |
| `timestamp` | UTC start time of the turn (ISO-8601, `Z`). |
| `category` | Scenario category, e.g. `honesty`, `tool_failure`. |
| `user_message` | The exact prompt sent (redacted). |
| `assistant_content` | Returned `content`, or `""` on transport failure. |
| `status` | LEO's `status`, or `transport_error` if all attempts failed (redacted). |
| `response_conversation_id` | The `conversation_id` returned by `/chat` for this turn (redacted); can differ from the requested id. |
| `request_id` | Stable `observer-<16 hex>` retry key derived from seed, conversation id and 1-based turn index; identical on every attempt. |
| `from_history`, `history_pending` | The answer was adopted from durable server history after a timeout; `history_pending` means the user row existed but the assistant reply had not been committed yet. |
| `history_check_error` | Normalized signature of a failed durable-history read; retries are suppressed when this is set. |
| `provider`, `model` | Redacted response provenance fields. |
| `simulated` | Response `simulated` flag; `null` if absent/non-boolean. |
| `model_status` | Response `model_status` (e.g. `mock`, `live`), redacted. |
| `usage` | Token usage normalized to strictly JSON-safe data. `NaN`/`Infinity` are replaced field-by-field with `null`, so one bad number cannot collapse the whole record; odd nested values fall back to strings and a non-dict value is stored under `raw`. |
| `memories_used` | Memory ids/details reported by the response (redacted, JSON-normalized). |
| `tool_names` | Redacted names extracted from `tool_results`. A dict or string `tool_results` field is normalized to a single tool record instead of being reported as "no tools". |
| `tool_results` | `[{"name", "result"}]` with redacted, length-capped result text (2000 chars) and redacted names. |
| `tool_errors` | Redacted, normalized signatures for tool results containing errors. |
| `tool_calls_pending` | Names of unexecuted tool calls reported by the response. |
| `command`, `command_status`, `command_success`, `command_result` | Slash-command response fields, when the response reports a command. `command_result` is JSON-normalized. |
| `attempts` | Number of POST attempts actually made for this turn. |
| `attempt_history` | Per-attempt `{attempt, error_signature, elapsed_seconds}` records. |
| `attempt_errors` | The non-empty error signatures from `attempt_history`, in order. |
| `elapsed_seconds` | Wall-clock time for all attempts of this turn. |
| `failure` | `true` only when every attempt failed. |
| `error_signature` | Normalized failure/status signature (`transport:...`, `status:...`). |
| `progress_error` | Normalized exception signature if a progress callback threw. |
| `deficiencies` | Names of observed deficiencies, e.g. `incomplete_status`, `incomplete_with_pending_tools`, `blocked_status`. |

Error normalization (`error_signature`) redacts the kind and the message,
lowercases, collapses whitespace, replaces UUIDs/paths/digits with placeholders,
truncates to 140 characters, and groups recurring errors that differ only by an
id, path or number.

Tool-error detection is **structured, not substring-based**. The result is
parsed as strict JSON, then as redaction-repaired JSON (`[REDACTED]` in a
numeric position is repaired to `0`), then as double-encoded JSON, and finally
with tolerant extraction of `"success"`/`"error"` key-value pairs from
malformed JSON. A key counts as an error only when its value is non-empty;
`success: false` / `ok: false` counts as a failure. A null/empty `error` field
with `success: true` is not an error — a live run had 215 results like
`"error": null, "duration_ms": ...` misread by a substring regex. Free-form
text without any structured field still uses a narrow `error`/`exception`/
`traceback`/`failed` heuristic, but that heuristic is suppressed as soon as a
structured `error` field is present, even with an empty value.

## Outputs

With the default out-dir:

```
artifacts/observation/
├── scratch/             # created before the run; tiny note files may land here
├── conversation.jsonl   # machine-readable, one turn per line
├── conversation.md      # human-readable transcript + summary header
└── summary.json         # structured summary (also on stdout)
```

`conversation.md` starts with run metadata (conversation id, turn count,
successful turns, floor status, **run verdict**, failures, deficiencies, tool
errors, pending calls, slash-command turns, average latency, a **dry-run
scratch marker**, status distribution, provider/model provenance, path-check
status and top error signatures), then prints each turn with its metadata —
including the response conversation id and the per-turn request id (marked when
recovered from durable history) — memories, pending calls,
deficiencies, user prompt, assistant answer, tool results and tool errors as
quoted blocks.

In a dry run the transcript says explicitly:
`Scratch writes: SIMULATED (dry run; no scratch file was created)`, and each
fake tool result contains `DRY-RUN: simulated scratch write; no file was
created`. The scratch directory itself is created in both modes so live and
dry-run paths are structurally identical.

## Structured summary

`build_summary()` is a pure function of the turn list, so the summary can be
recomputed and tested without rerunning the conversation. Keys:

* `total_turns`, `successful_turns`, `floor_satisfied`, `floor_size`, `failures`;
  the final JSON on stdout is the same redacted object also written to
  `summary.json`;
* `deficiency_count`, `deficiency_distribution`, `run_verdict`,
  `honest_accounting`;
* `conversation_id_mismatches` — turns whose response `conversation_id`
  differed from the requested one;
* `tool_invocations`, `tool_error_count`, `memories_used_count`,
  `memories_used_turns`, `pending_tool_calls`, `pending_tool_call_turns`;
* `slash_command_turns`, `command_success_count`, `command_failure_count`,
  `command_status_distribution`;
* `retry_count`, `attempt_error_count`, `progress_error_count`, and the
  idempotency counters `history_recovered_turns`, `history_pending_turns`,
  `history_check_error_count`;
* `average_latency_seconds`, `total_latency_seconds` (averages include failed turns);
* `status_distribution` — e.g. `{"completed": 392, "incomplete": 6, "transport_error": 2}`;
* `providers`, `models`, `simulated_counts` (`simulated` / `live` / `unknown`);
* `provider_model_pairs` and its alias `provider_model_provenance` — per
  provider/model/mode turn counts, so a mock-backed run can never be mistaken
  for a live-model run;
* `top_error_signatures` — up to 10 `{signature, count}` pairs, gathered from
  transport failures, tool errors, per-attempt errors, progress callbacks and
  explicit error statuses;
* `category_distribution`, first/last timestamps, conversation id, `dry_run`,
  `messages_requested` and the redacted run `config` (including the sanitized
  URL and the scratch directory);
* `tool_result_paths` — the post-run audit of paths claimed in tool results;
* `output_check` — the end-of-run completeness report for all three files.

## Retries, robustness and post-run checks

For each turn the loop makes at most `1 + retries` POSTs. A timeout, HTTP
error, malformed JSON, or any other transport exception is caught; after a
short backoff it retries the *same* prompt with the *same* conversation id and
the *same* `request_id`. Every attempt gets its own record with a normalized
error signature, so a retry that eventually succeeds is still visible in
`attempt_history` / `attempt_errors`. If every attempt fails, the turn is
appended with `failure: true`, `status: "transport_error"`, the signature and
attempt count, and the loop moves on.

### Retry idempotency contract

A client timeout does **not** prove the server stopped: it may finish the
original turn and write a user/assistant row pair after the client gave up.
Blindly resending creates duplicate history. The observer therefore implements
this contract:

1. `request_id_for_turn(seed, conversation_id, index)` produces
   `observer-<16 hex chars>` from a SHA-256 of `seed|conversation_id|index`.
   It is deterministic across processes and unique per logical turn.
2. Each POST carries that value in the JSON body field `request_id` and in the
   `Idempotency-Key` header, on every attempt. A server implementing the
   contract stores the id with the turn's message metadata and echoes it in
   the chat response; a repeated id for the same conversation should return the
   original result without inserting new rows.
3. Before any retry, the observer calls
   `GET /conversations/{conversation_id}` and looks for the user message whose
   metadata `request_id` matches (older servers that do not store ids are
   handled by exact prompt-text matching, which is safe because all 400
   scenario prompts are distinct).
   * If an assistant reply follows that user row, the reply is adopted as this
     turn's payload: `from_history: true`, no further POST is made, and the
     turn is scored from the stored answer.
   * If the user row exists but no assistant reply is committed yet, the turn
     is recorded with status `incomplete`, content `""` and
     `history_pending: true`. This is honest: the answer was not observed.
   * If the conversation does not exist yet (`404`), nothing was committed and
     a retry is safe.
   * If the history read itself fails (timeout, HTTP 5xx, malformed JSON), the
     retry is **suppressed** and the failure is recorded in
     `history_check_error`. An unknown history state must never cause a
     duplicate write.
4. `summary.json` reports `history_recovered_turns`, `history_pending_turns`
   and `history_check_error_count`; `turn.request_id` and the Markdown request
   id line make the same audit possible per turn. Attempt counts include only
   real POSTs, never history reads.

Offline transports may omit `request_id`/`turn_in_history`; the observer
detects the supported signatures and falls back to plain retries.

Usage and other response fields are normalized through
`json.dumps(default=str, allow_nan=False)` with non-finite numbers replaced
individually, so NaN/Infinity or a strange object in a server response cannot
abort the run or collapse sibling token counts. `tool_results` is normalized
whether the response sends a list, a single dict or a plain string. Turn
construction, the progress callback and each JSONL checkpoint write are all
wrapped: a hostile field or a serialization failure yields a complete safe
fallback record with the correct turn index, and the loop continues. The
summary writer likewise falls back to strictly valid JSON that retains the
authoritative success/floor fields.

With the injected `FakeTransport`, failure indices are keyed to logical turn
numbers, not attempts, so retry behaviour is deterministic and testable.

After the run:

* `check_tool_result_paths()` audits every path claimed in a tool result.
  Paths outside the workspace are flagged; missing scratch files are flagged
  for live runs, while dry runs are expected to leave none behind. NUL and
  other control bytes (e.g. from binary `.pyc` search output) are rejected
  *before* filesystem calls and reported as escaped issues: `Path.resolve()`
  raises `ValueError` on those bytes and previously killed all post-run
  reporting. `OSError`, `RuntimeError` and `ValueError` from path handling are
  contained per claim.
* `check_output_completeness()` verifies that the JSONL has one parseable,
  correctly indexed line per turn, the Markdown has one heading per turn plus
  its header, and the summary JSON agrees on `total_turns` and contains the
  success/floor fields. Failures are listed in `output_check.issues` and cause
  exit status `1`.
* All post-run reporting is wrapped so a checker that raises cannot discard a
  400-turn run. The checker result becomes a redacted `post_run:<checker>`
  issue, and a `finally` block guarantees that `conversation.md` and
  `summary.json` are written before the process exits. A broken checker still
  produces exit status `1`, but the transcript, checkpoint and authoritative
  summary remain on disk.

## Redaction

`redact()` is applied when a turn is built, again to every value interpolated
by the Markdown writer, and again by the JSONL and summary writers (including
the JSON printed to stdout). It covers:

* `http(s)://user:password@host` URL userinfo (config URLs are also sanitized
  before storage);
* whole auth headers and schemes — `Authorization`/`Proxy-Authorization`/
  `X-Auth-Token` with `Bearer`, `Basic`, `Token`, `Digest`, `Negotiate`,
  `NTLM`, `MAC`, `ApiKey` or `AWS4-HMAC-SHA256` (case-insensitive: lowercase
  and mixed-case spellings redact too) — plus bare `Basic <value>` forms. The
  scheme *and* its entire credential parameter list are replaced together,
  including Digest/Negotiate `nonce`, `response`, `cnonce`, `opaque`, `qop`,
  `nc` and quoted parameters in either order. Parameters joined by `,` **or**
  `;`, and values glued to `]`, `}`, `)` or a backslash, are consumed as one
  unit, so no trailing parameter survives a partial match; escaped wrappers in
  a scheme value (`Bearer \[token\]`) stay inside the redacted span;
* `Cookie` / `Set-Cookie` headers and `session_id` / `sessionId` / `session`
  values, including quoted JSON keys;
* JWT-shaped `xxx.yyy.zzz` values;
* provider-prefixed keys (`sk-`, `sk-proj-`, `ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_`,
  `xox*-`, `AKIA...`, `AIza...`);
* sensitive key assignments are handled by a **key-aware scanner**, not by a
  growing family of regex alternatives. It recognizes case- and
  separator-tolerant names (`authToken`, `AUTH TOKEN`, `client_secret`,
  `client-secret`, `API key`, `PIN`, `cookie`, `session_id`, `X-Session-Id`,
  `authorization`, ...) and then consumes the *entire* value span: repeated or
  mixed operator tokens (`:`, `=`, `=>`, `->`, `:=`, `::`, `<>`, `(=`), nested
  and mixed wrappers with optional interior whitespace (`[ x ]`, `[[x]]`,
  `<[x]>`, `{x}`, backticks, escaped wrappers), leading/repeated/escaped comma
  and semicolon lists, and nested keyed first values (`authToken: inner=x`).
  It stops at the next keyed assignment or clause boundary, so two secrets in
  one line get two placeholders and benign text such as `Bearer count: 12` is
  untouched. Whitespace only joins known multiword-key words (`API key`,
  `X Session Id`), so an unrelated plain word before `session id:` cannot be
  mistaken for part of the key; scanning always resumes at the end of the
  replaced span and repeats to a fixed point, so concatenated assignments such
  as `API key: x API key: y` or `session id: x; session id: y` cannot leave a
  second value (or a bare `key:` / `id:` fragment) behind. A standalone
  trailing component (`key`, `id`, `token`, `secret`) followed by an operator
  is treated as sensitive; the harness's own descriptive phrase `Response
  conversation id: <id>` is the explicit benign exception. Short or numeric
  values under sensitive keys are redacted, and an existing `[REDACTED]` value
  (including wrapped or listed placeholders) is passed through unchanged.
  Compound key detection preserves component order, avoiding hash-seed
  dependent misses such as `apiKey` becoming `keyapi`; known token-count fields
  are exempt so usage data survives;
* JSON string escapes are normalized to a fixed point (bounded at 64 passes)
  before embedded JSON is re-parsed structurally and the scanner runs. Odd
  backslash runs are repaired, including a dangling backslash before a decoded
  `=` and odd escapes standing in for `:`, so `\n`, `\t`, `\u000a` — including
  triple-escaped forms such as `{"authToken":\\\n"secret"}` — become real
  boundaries. Escaped/double-encoded JSON (`{\"authToken\": \"9988776655\"}`)
  is redacted structurally, so 2-4 character and short numeric secrets under
  sensitive keys are removed too;
* a conservative high-entropy fallback (mixed character classes, excluding
  paths and ISO dates) for short secrets in innocuously named metadata;
* any standalone 32+ character alphanumeric/underscore/dash string and any
  12+ digit numeric secret.

All server-controlled fields — status, provider, model, model_status,
tool names, tool result names, memories, pending calls and command fields — are
redacted at construction *and* at write time. The replacement is `[REDACTED]`;
the operation is idempotent. No environment variable is ever read, so no
environment secret can reach the logs.

## Tests

```bash
python -m pytest -o addopts='' -q tests/test_conversation_observer.py
```

Sixty-one offline tests cover:

1. the 400-message default and hard floor (including CLI rejection of 399);
2. deterministic scenario generation, uniqueness, category coverage, topic
   prompts and absence of risky asks;
3. redaction of token variants and nested containers, plus idempotence;
4. failure continuation, bounded retries and the progress callback;
5. tool result name/error extraction, truncation and null-error handling;
6. JSONL/Markdown/summary writing, redaction of injected and environment
   tokens, and exact summary counts/latency/provenance/signatures;
7. `--dry-run` never calling `urlopen` (network monkeypatched to fail);
8. the CLI's default 400-turn dry run and all three output files;
9. a token injected into every metadata field being absent from the returned
   turns, checkpoint, JSONL, Markdown and summary;
10. `--allow-short` rejection in live mode and honest floor accounting for
    all-failed, partially failed and short runs;
11. memories, pending calls and slash-command fields, plus incomplete,
    blocked and error statuses counting as deficiencies;
12. per-attempt retry history, throwing progress callbacks and both positive
    and negative completeness checks;
13. credential scrubbing in `--url`, scratch-directory creation and tool-path
    auditing (including an outside-workspace path);
14. dry-run scratch markers and low-repetition, grammatically clean scenarios;
15. camelCase/underscore key redaction, all non-Bearer auth schemes, cookies,
    session ids, short/numeric secrets and preservation of benign observations;
16. an end-to-end CLI run proving credentials leak into neither stdout, JSONL,
    Markdown, summary nor any distribution, and that the server's response
    conversation id is recorded and counted when it differs;
17. NaN/Infinity usage, odd tool-result/memory shapes and broken `__str__`
    values remaining per-turn data with strictly serializable checkpoints;
18. a forced per-turn JSONL serialization failure producing a safe fallback
    line and letting the next turns complete;
19. authoritative all-fail verdicts in stdout/summary/Markdown, including a
    rebuilt summary where stale per-turn fields claim success;
20. lowercase/mixed-case auth schemes in text, nested data and RuntimeError
    signatures;
21. an end-to-end run with lowercase schemes injected into content, memories,
    metadata, tools and a transport exception, checked in all four outputs;
22. escaped/double-encoded JSON keys with short, numeric and nested secrets;
23. the same encoded keys injected end-to-end into content, memories, metadata
    and tools, checked in checkpoint, JSONL, Markdown and summary files;
24. complete Digest parameter lists (quoted, unquoted and both orders) in text
    and error signatures;
25. all-error-status runs being labeled `all_answers_failed` with status-based
    failures counted separately from transport failures;
26. semicolon-separated and bracket/brace/parenthesis/backslash-terminated
    Digest and Negotiate parameter lists, in text, JSON and error signatures,
    while benign descriptive uses (`Bearer count: 12`) survive;
27. 900 combinatorial escaped-separator assignments (`\n`, `\t`, `:`,
    doubled escapes, all debug operators and value wrappers), with idempotence
    checks;
28. escaped whitespace between a JSON key and its value in unit cases;
29. an end-to-end run injecting escaped-newline JSON into content, status,
    metadata, memories, tools, command fields and a transport error, checked in
    checkpoint, JSONL, Markdown, summary, stdout and all summary distributions;
30. debug `=>`/`->` operators with bracket, parenthesis, backtick and quote
    wrappers, and pass-through of an already-redacted `[REDACTED]` value;
31. deeply double-escaped credentials (eight levels) proving the unescape loop
    runs until stable instead of stopping after four passes;
32. 140 auth assignments with plain lowercase secrets proving `=>`, `:=`, `::`
    and `==` cannot backtrack into a one-character value and leave a bracketed,
    backticked or escaped-bracket secret, including `Bearer \[secret\]`;
33. nested/mixed wrappers (`[[x]]`, `[(x)]`, `["x"]`, `<[x]>`, `` `(x)` ``,
    `\[x\]`) and full comma/semicolon value lists under sensitive keys, with
    following keyed assignments still recognized by name;
34. odd (3/5/7/9) backslash runs before `\n`, `\t` and `\u000a`, parsed back
    into JSON objects whose sensitive key maps to `[REDACTED]`;
35. a 16-level deep-escape payload with a plain secret, proving the 64-pass
    decoder reaches and structurally recognizes the key;
36. an end-to-end run injecting all three structural variants into content,
    status, metadata, memories, tools, command fields and a transport error,
    checked in checkpoint, JSONL, Markdown, summary, stdout and distributions;
37. whitespace inside wrappers (`[ x ]`, `[[ x ]]`, `< [ x ] >`) with plain
    lowercase secrets;
38. brace wrappers (`{x}`, nested and in lists) with the same plain secret;
39. secondary operator tokens consumed as values (`authToken: = qwertyuiop`,
    `:=`, `=>`, `<>`, `(=`), with no one-character operator left as a value;
40. leading, repeated and escaped comma/semicolon separators before a value;
41. escaped semicolons that continue a list instead of ending the credential;
42. nested keyed first values (`outer: authToken: x`, `authToken: inner=x`),
    where the outer scanner consumes the inner assignment and a following
    sensitive key still gets its own placeholder;
43. odd escape runs leaving a dangling backslash before a decoded `=`;
44. whitespace-only key/value separators (spaces, tabs and newlines);
45. odd escapes replacing the colon (`authToken\\: x`, `\\n`, `\\u003a`),
    including quoted JSON-style keys;
46. a generated 5,880-variant sweep (7 keys x 7 operators x 10 wrappers x 12
    separators) with zero plain-secret leaks and no idempotence regressions;
47. an end-to-end run injecting all nine root-cause variants into content,
    status, provider, model, model_status, command fields, memories, tools and
    a transport error, then checking turns, summary, checkpoint, JSONL,
    Markdown, stdout and every distribution;
48. benign-text pass-through (`Bearer count: 12`) and preservation of sensitive
    words embedded in ordinary file paths;
49. concatenated multiword-key assignments (minimal `API key: x API key: x`
    repro, semicolon form, `session id`, `X Session Id`, bare `key:` / `id:`
    fragments) producing one placeholder per secret with idempotent repeat
    passes, while `Response conversation id: <id>` stays observable;
50. all 5,880 scanner variants distributed across one real 400-turn dry run,
    checked in checkpoints, turns, JSONL, Markdown, summary and stdout JSON;
51. success requiring a completed status, no pending tools and real content —
    incomplete-with-text, pending-tool and unknown-status turns are deficient;
52. clean/deficient floor verdicts, including a 401-turn run with 400 clean
    answers plus one incomplete turn reported as
    `floor_satisfied_with_deficiencies` (and verdict words surviving redaction);
53. NUL/control-byte path claims being rejected by the path checker and a full
    CLI run still writing all three output files;
54. a post-run checker that throws being reported as a
    `post_run:tool_result_paths` issue while the transcript, checkpoint and
    summary are still written (exit status `1`);
55. deterministic, control-free per-turn request ids that change with index,
    seed and conversation id;
56. the HTTP transport sending `request_id` in the body and `Idempotency-Key`
    header and normalizing stored history replies (including pending turns and
    a `404` meaning "nothing committed");
57. a timed-out turn being adopted from durable history without a resend, and
    a failed history check suppressing the retry instead of duplicating rows;
58. structured `success`/`error` fields overriding substring guesses, including
    redaction-repaired JSON, truncated JSON, 200 results with `error: null`
    counted as zero errors, and real failures still counted.

The fake transport can inject failures, tool errors, mixed statuses, content
and metadata credential leaks, which is how the logging, redaction and summary
paths are exercised without a server.

## Limits and interpretation notes

* The harness logs what the HTTP response says. A tool name in `tool_results`
  is evidence the response reported a tool result; cross-check with LEO's own
  audit log if exact execution matters.
* Error detection is exact for structured tool results (including
  redaction-repaired and double-encoded JSON). Only free-form tool text without
  any structured field uses the narrow heuristic, so unusual formats there may
  be under-counted; a null `error` field is never counted.
* Average latency is wall-clock per turn including retries and backoff; it is
  not a model-only timing.
* A 400-turn live run is intentionally slow and will build real LEO chat
  history and task/memory rows from the prompts. Use a dedicated conversation
  id or a disposable local database if that history must not be kept.
* The scenario asks LEO to write small scratch notes under the run's
  `scratch/` directory. Those files, if LEO creates them, are separate from
  the harness log files and are not removed automatically. The path audit
  reports missing scratch claims rather than fabricating files.
* `--dry-run` output is synthetic and says so in the summary (`"dry_run":
  true`, provider `mock`, model `fake-model`, simulated scratch writes). Never
  report it as an observation of the real server.
