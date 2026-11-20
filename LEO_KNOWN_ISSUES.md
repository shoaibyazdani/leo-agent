# LEO Known Issues and Blockers

## Fixed in the deep-research hardening round (2026-11-18)

Verified and repaired with regression tests: arbitrary `approval_id` bypass of
HIGH-risk gates; webhook verification failing open when secrets are unset; HMAC
computed over re-serialized JSON instead of raw bytes; `python`/`python3` and
dangerous flags in the terminal allowlist; unauthenticated API on non-loopback
binds; self-service approval resolver; evidence-free completion through
`transition()` and agent delegation; completion after cancellation; double
injection of memories; unreachable context compaction; misleading `1.0.0`
versions for imported skills; OpenClaw bin gates causing self-dependencies;
44 real skills rejected because shell examples in documentation were treated as
executable code; dashboard approvals list reading the wrong JSON key; webhook
replay via unsigned delivery IDs; secret leakage through `extra_header_env`;
agent capability tags sent to providers as literal model IDs. After the fixes,
326 of 327 real upstream SKILL.md files translate and pass LEO registry
validation (the remaining file has no frontmatter and is quarantined). A final
independent review also closed a config-file host bypass, terminal flag-operand
path escapes, the standalone fallback task engine's evidence gap, and a
cross-origin token leak in Electron.

Details and evidence: deep research synthesis.

## Fixed after the deep-research round (2026-11-18)

- **Chat attachment path bypass** (found by exploration, not by the audit
  round): `ask_stream` accepted raw `{path, mime}` dicts as attachment records
  with no upload validation, so a local caller could have any readable file
  (e.g. `~/.ssh/id_rsa`) base64-encoded and sent to the configured provider as
  an image block. Raw dicts are now rejected with an `invalid_attachment`
  error and a failed run, and `AttachmentStore.image_block` refuses any path
  outside the workspace root even for tampered database rows. Regression
  tests in `tests/test_security_regressions.py`.
- **Concurrent task claims (P1)**: `db.transaction()` adds a real
  `BEGIN IMMEDIATE` transaction primitive; `run()` claims a task through one
  atomic conditional update, so a second concurrent *or sequential* `run()`
  on a RUNNING task never starts a second worker (it returns the current
  record instead). Idempotent `create()` performs the kv claim and the task
  insert in one transaction, removing the claimed-but-absent race. `_save()`
  now compare-and-swaps on a new `tasks.version` column (with an idempotent
  `ALTER TABLE` migration) and raises `TaskConflictError` on a lost update —
  a concurrent write (e.g. a cancellation flag) can no longer be silently
  overwritten. Also fixed in the same pass: recovery covers tasks stranded in
  VERIFYING/PLANNING (F-17), and a worker that calls `fail()` itself no
  longer crashes `run()` with an illegal transition (F-18). Regression tests
  in `tests/test_tasks.py`.
- **Webhook signature binding (P1)**: the native v1 scheme
  (`X-LEO-Signature-V1`, produced by `WebhookHandler.sign_v1`) now
  authenticates `timestamp || delivery_id || body` in one HMAC with a
  mandatory `X-Webhook-Request-Timestamp` inside the ±300 s window. A
  captured signed request can no longer be refreshed (timestamp forgery),
  re-labelled (delivery-ID forgery used to suppress a future legitimate
  delivery) or replayed cross-instance; distinct properly-signed deliveries
  of an identical body are now correctly treated as distinct tasks. Legacy
  body-only signatures (GitHub/WhatsApp style) remain an explicit
  compatibility path whose only replay defences are the timestamp window and
  the body-hash dedupe; callers able to use v1 should. The dedupe key update
  is now guarded by the claim token. Regression tests in `tests/test_tasks.py`.
- **Agent tool scoping (P1)**: `AgentSpec.tool_permissions` is now enforced at
  both layers. A new `ScopedTools` façade (a) advertises only the in-scope
  specs to the model — fixing the previous doubly-broken state where model
  workers received either zero tools (dead `.specs` attribute) or every tool
  — and (b) routes execution through an audited `scope` check in the
  executor that denies out-of-scope calls (`decision="out_of_scope"`) before
  validation or permissions. An entry matches a tool by **name or category**;
  an empty scope denies everything. Handlers receive the façade, never the
  raw executor. The twelve specialist seeds were corrected to reference the
  real toolset (e.g. coding = `filesystem`+`development`, security is
  read-only by name: `read_file`/`list_files`/`search_files`/`utility`).
  Regression tests in `tests/test_agents.py`.

  read-only by name: `read_file`/`list_files`/`search_files`/`utility`).
  Regression tests in `tests/test_agents.py`.
- **SSRF / DNS rebinding (P1)**: `http_get` now resolves each host **once**,
  validates every returned address, and pins the connection to a
  pre-validated IP (rewritten URL) while preserving the real host for the
  `Host` header and TLS SNI/certificate verification (`sni_hostname`) — the
  policy check and the connection can no longer observe two different DNS
  answers. The browser operator now resolves DNS names and requires public
  addresses for both `navigate()` and `download()` (previously names were
  accepted with no resolution check at all, so a name with a private A
  record passed unconditionally). Regression tests in `tests/test_tools.py`
  and `tests/test_integrations.py`. **Residual risk (documented, not
  hidden):** Chromium and Playwright's API-request stack perform their own
  resolution at fetch time, so a check-time-public / connect-time-private
  rebind against the *browser* remains theoretically possible; full
  mitigation requires an egress proxy or per-navigation host pinning and is
  a pre-deployment gate for any non-loopback exposure.

- **Agent engine correctness batch (F-13/F-14/F-10/verdict parsing/fallback
  engine)**: worktree integration now commits the agent's work *before* the
  `merge-tree` conflict pre-check (the old check merged the unmodified base
  and was trivially clean), and a failed `git merge` is rolled back with
  `git merge --abort` so the source repository is never left with
  `MERGE_HEAD`; merged worktrees are cleaned up after integration.
  Prior-task dependencies are resolved into results before wave execution
  and actually reach handlers (context and evidence) instead of being
  recorded on the task row and dropped. Peer-tie verification is now
  deterministic: a tie has no majority and fails closed, winners are chosen
  by count then canonical text (set iteration is hash-randomized per
  process). `parse_verdict` no longer reads "cannot approve"/"deny"/"fail"
  as approvals. `_DurableTasks` gained one `cancel` (the second definition
  could flip COMPLETED tasks), persists `cancellation_requested`, and both
  engines expose a public audited `update()` that merges metadata, applies
  legal transitions and never stamps `completed_at` — the orchestrator's
  root bookkeeping now goes through it instead of raw SQL that replaced
  metadata and bypassed the state machine. Regression tests in
  `tests/test_agents.py` and `tests/test_tasks.py`.
- **Scheduler defects (F-15/F-16/F-23)**: cron fields with a scalar base and
  a step (`5/10 * * * *`) now expand to the field's upper bound instead of
  firing once. `Scheduler.tick()` anchors the next run to `scheduled_for`
  (not the late tick moment), catches up bounded missed occurrences (cap 10)
  with a `schedule.catchup_truncated` event beyond that, and preserves each
  occurrence's own idempotency key. The previously dead
  `max_concurrent_tasks` setting is enforced as a durable backlog gate:
  bursts emit `schedule.deferred` and leave `next_run_at` in place for retry
  rather than creating unbounded tasks. Regression tests in
  `tests/test_tasks.py`.
- **Model/context batch (F-19/F-20/F-21, F-11 residual)**: naive ISO
  `expires_at` strings are normalized to UTC instead of raising a
  naive-vs-aware `TypeError` on every chat turn. Budget eviction and
  compaction now move `assistant(tool_calls)` and their matching
  `tool_result` messages as one atomic group, so providers no longer see
  orphaned tool results. Explicit provider preferences scale their offset
  with the list length, so the 11th+ preferred provider still outranks a
  configured default not in the list. `BaseHTTPProvider._redact` now
  re-reads and scrubs every `extra_header_env` value at call time (covering
  rotation) in error messages and retry warnings — closing the provider
  half of F-11. Regression tests in `tests/test_models.py`.
- **Chat/skills defects (F-25/F-26)**: both chat loops now report
  `tool_calls_pending` in the API response / `message.finished` event and in
  the persisted assistant-message metadata when the eight-round chat tool
  budget is exhausted, instead of silently storing an unanswered turn as
  complete.
  `SkillRegistry.replace()` (rollback) now preserves the previous skill's
  `status`, `enabled` flag and local handler, so rolling back a disabled
  skill neither re-enables it nor leaves it non-executable. Regression tests
  in `tests/test_chat_tools.py` and `tests/test_skills.py`.

## Remaining verified defects (not yet fixed)

- **Go-live report follow-ups (2026-11-19)**: the 47-finding robustness
  report (`an external audit report`) items 1-4
  and 6-10 are fixed (path-only tool identity, scoped expiring quota, truthful
  cancellation, explicit incomplete turns, memory prompt isolation, audit
  integrity, input validation, bounded limits, HTTP 400 for malformed
  integration candidates, message index and per-conversation run rejection).
  Item 5 (per-principal scoping) is **deliberately not implemented**: LEO is a
  single-user local-first agent, a `LEO_API_TOKEN` holder is fully trusted,
  and this is now stated explicitly in `LEO_SECURITY.md`. Do not expose a
  token to mutually untrusted users. Remaining partial gaps: `/events` and
  `/metrics` expose totals/caps but no keyset pagination cursor; tool-call
  counters are per-process and non-durable; memory recall has no per-request
  opt-out yet.

- **Design decisions, documented not hidden**: the orchestrator mints its
  own evidence entry at root completion (`delegated agent X returned
  success`) and the root result never passes through
  `VerificationEngine`; evidence validity is structural (a truthy `claim`
  passes); `AgentSpec.context_budget` is seeded and persisted but read
  nowhere; agent isolation is cooperative (worktrees for files, no
  process/network/DB sandbox).
- **Compatibility scope**: importing Hermes/OpenClaw `SKILL.md` produces an
  *inert manifest*, not an executable skill. Bundled scripts, path tokens and
  availability-gate enforcement are not implemented; never claim runtime
  compatibility.

## Blocked (external authorization/credentials required)

1. **Live model provider — environment-conditional, not universally blocked**:
   the `opencode-go` provider points at the OpenCode Go gateway's
   Anthropic-compatible endpoint with the `the configured free-tier model` model ID. The gateway
   requires both a bearer token (`ANTHROPIC_AUTH_TOKEN`) and a stable session ID
   in `x-opencode-session`; `leo.yaml` maps that header to the non-secret
   `LEO_OPENCODE_SESSION` env var via `extra_header_env`. **When the host
   environment supplies both values, the live provider is available**;
   otherwise it degrades to mock (the configured `mock` fallback). An earlier real
   attempt returned HTTP 400 only because the session value was missing then.
   `LEO_PROGRESS.md` records a real-model pass (real calculator/current_time
   tool calls and non-estimated token usage) on a host that supplied them. This
   is a record of that pass, **not a fresh live verification performed for this
   document update** — no live call was made here. The default offline suite
   still uses the mock provider.
2. **Telegram / Discord / WhatsApp**: adapters, webhook verification and
   normalized routing are mock-tested, but no platform credentials/webhook
   secrets are present. No live message has been sent or received.
3. **Remote VPS node**: local node API/client and auth are tested; no remote
   host, token or TLS endpoint was provided. No public deployment attempted.
4. **Upstream skill registries (ClawHub / Hermes Skills Hub)**: not implemented;
   require license/trust review before any network client is written.

## Known limitations

- Token budgeting uses a documented characters/4 estimate, not provider tokenizers.
- Default memory embeddings are deterministic hash/trigram vectors, not a trained
  semantic model; a real embedder can be injected.
- Memory search scores candidates in Python; suitable for modest volumes, not a
  large-scale vector index.
- Tool-call counters are per-process; a restart resets in-memory counters.
- The browser operator validates DNS at the operator layer; Chromium and
  Playwright's API-request stack still perform their own resolution at fetch
  time (see the residual-risk note above).
- Browser operator's real backend requires Playwright/Chromium (installed here);
  the default suite uses the injectable fake backend.
- Discord Ed25519 verification requires an injected verifier or crypto library
  (`cryptography`/`PyNaCl`), not installed by default.
- Integration factory "isolation" is cooperative (restricted view, separate
  thread, timeout), not an OS-level sandbox.
- Scheduler has no distributed lock; run one scheduler per database.
- Recovery never replays workers by design; interrupted tasks become READY or
  BLOCKED for explicit continuation.
- No Docker available, so OS-level sandboxing is not used; path/permission
  checks and audit logs are the enforcement boundary.
- Research findings on upstream ecosystems are timestamped 2026-11-18 and can
  drift; some upstream statistics could not be independently rechecked after
  GitHub API rate limiting.

## Non-goals for this milestone

Public deployment, paid subscriptions, uncontrolled self-modification,
automatic merging of agent-written code, and unrestricted remote terminal
access are deliberately excluded.
