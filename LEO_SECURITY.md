# LEO Security Design

## Trust model (explicit)

LEO is a **single-user, local-first** agent. The default API binds to
`127.0.0.1`. When `LEO_API_TOKEN` is set, every interactive route requires the
bearer token; **a token holder is fully trusted** — conversations, memories,
attachments, events and audit are not partitioned by per-user principals, and
channel/owner fields in requests are data, not authorization boundaries. Do not
expose LEO to multiple mutually untrusted users or a public network. Per-principal
scoping would require a schema/identity migration and is not implemented.

## Boundaries

- **Workspace restriction**: path-bearing tool resources must resolve inside the
  project root or configured `allowed_roots`; otherwise the action is denied.
- **Risk tiers**: low (read/analyze) allowed; medium (modify project files,
  install dependencies, run scripts) allowed by policy flag; high (delete,
  external messages, publish/deploy, destructive infrastructure) requires
  explicit approval via `leo approval approve <id>`.
- **Audit**: every permission decision writes an `audit_log` row with actor,
  action, resource, risk, decision and reason. Tool executions add timing,
  result and error rows.
- **Secrets**: configuration stores only environment-variable *names*. Values
  are read at call time and never logged, echoed into prompts, or returned by
  the API. JSON log handlers apply regex redaction.
- **Untrusted content**: pages, webhooks, tool output and external messages are
  data, not instructions. Prompt templates label such content explicitly.
- **Network**: API binds to `127.0.0.1` by default and refuses non-loopback
  binds unless `LEO_API_TOKEN` is set; that token is required on all interactive
  routes (webhooks authenticate with provider HMAC instead). Provider webhooks
  fail closed when verification material is absent, and signatures are verified
  over the exact raw request bytes (1 MB cap). HTTP tools block private/
  loopback targets at preflight time.
- **Terminal**: the default allowlist excludes interpreters (`python`,
  `python3`, `node -e`), dangerous per-command flags (`find -delete`,
  `sed -i`, `git push`, …) and path arguments outside allowed roots. It is a
  command policy, not an OS sandbox; `security.allowed_executables` is an
  explicit operator opt-in.
- **Learning**: notes are derived only from LEO's own durable transcripts
  (explicit operator statements, bounded tool/run trajectories) and source text
  that fails the secret or instruction-shape sanitizer never becomes a note.
  Every note carries evidence hashes and is deduplicated before the write.
  `external`/chat-injected sources are ineligible; chat text is never promoted
  to `FACT`; superseding an existing memory is always approval-bound. The
  shipped policy writes eligible notes unattended, and
  `learning.auto_notes.enabled: false` restores the HIGH-risk approval gate.
- **Integrations**: new integrations start with minimum permissions; the
  factory refuses unknown licenses/unsafe execution primitives pending review.
- **Agent isolation**: parallel coding agents use Git worktrees and changes are
  never auto-merged without verification.

## Approval workflow

Every tool call is assessed **per call**, not per tool, and the reason is
recorded on the audit row (`metadata.risk_reason`):

| tier | meaning |
|---|---|
| LOW | runs; read-only inspection (a `SELECT`, a `systemctl status`, a reply into the originating conversation) |
| MEDIUM | runs and is audited; bounded reversible writes (an in-workspace edit, `git commit`, `mkdir`, `systemctl restart`, a sqlite write) |
| HIGH | needs an approval; destructive, unreversible, arbitrary-code, credential/config writes, self-modification, outbound to a third party |
| — | the hardline blocklist is a **denial**, not a tier, and is checked before any of this |

An operator who wants the old strict floor sets `security.allow_medium_risk:
false`: MEDIUM then fails closed instead of running. `security.shell_mode` still
overrides the shell tier (`manual` = every command asks, `off` = nothing asks,
though the hardline floor still denies).

1. A tool call classified HIGH returns `ApprovalRequired` with an approval ID.
2. The ID appears in `leo approval list` and the dashboard's Approvals panel.
3. `leo approval approve/deny <id>` records the human decision (durable).
4. The caller retries with the approval ID; permissions verify action,
   resource, actor, risk and expiry before allowing the operation. A valid
   approval cannot waive a path/workspace restriction, and an invalid ID never
   clears the HIGH-risk gate. The HTTP resolver records the authenticated
   principal, not a caller-chosen name.

### Approval hardening (2026-11-19)

The HIGH-risk approval path now binds and consumes decisions tightly:

- **Argument-digest binding**: a pending approval stores an `arguments_digest`
  (SHA-256 of canonical, sorted-key JSON of the tool arguments) in its metadata.
  At retry time the current arguments are re-digested and compared with
  `hmac.compare_digest`; changing target/text/path/program invalidates the
  approval. Authorization still checks action, resource, actor, risk, expiry
  and the path/workspace floor.
- **One-shot consumption**: resolution is serialized in a single
  `db.transaction()` with a conditional `UPDATE ... WHERE id=? AND
  status='pending'`. A second resolve (including a concurrent replay) fails on
  the row-count check, so an approval can authorize at most one decision.
- **Chat-actor self-approval denial**: a resolver whose name starts with
  `chat:` is the requesting principal, never the human gate. It is denied with
  reason `chat actors cannot resolve approvals` and an audited
  `denied_reason=self_approval`, independently of any tool/channel scope.
- **Atomic decision + audit commit**: the status update and its `audit_log`
  row share one transaction connection; if the audit insert fails, the state
  change rolls back with it. The approval cannot be committed without its
  trace, and the audit names the exact approval ID and resolver.
- **Operator-surface intent (2026-11-20)**: the HTTP resolve endpoint rejects
  cross-site browser callers (`Origin` allowlist plus `Sec-Fetch-Site`) and
  requires the approval's own action string echoed back as `confirm_action`.
  A page on another origin can POST to a loopback bind even though CORS hides
  the response, and a stale dashboard tab that auto-POSTed decisions was
  observed resolving pending approvals; blind or drive-by clients now receive
  403/409 and the row stays pending.
- **Bounded cleanup, never deletion**: `leo approval cleanup` reports stale
  pending approvals and, with `--apply`, marks them `expired` with an audit
  row. Rows, digests and approval ids are retained; a chat principal cannot
  run or scope the cleanup.

Regression evidence: `tests/test_tools_approval.py`,
`tests/test_security_regressions.py`, `tests/test_harden_cli.py` and
`tests/test_api.py` (approval binding, replay/expiry, audit traceability,
cross-site rejection and confirmation).

### Proactive schedule hardening (2026-11-26)

- **Unattended side effects fail closed**: a scheduled/unattended tool call of
  MEDIUM or HIGH risk in the outbound, browser, computer, plugin or shell class
  is denied in `ToolExecutorImpl` before `PermissionManager`, quota or dispatch.
  No approval row is created; exactly one durable `automation.needs_you` notice
  names the refused class and provenance. Attended command approvals keep their
  ordinary behavior.
- **Execution-time re-scan**: a stored schedule prompt is scanned again when the
  agent adapter prepares a run. Tampered control/injection/secret text fails the
  task closed with a needs-you notice and never reaches the model or an audit
  row as raw text.
- **Secret sinks**: opt-in tools refuse a detected credential before
  authorization, quota, handler and persistence, storing only a canonical digest
  and refusal category. Intrinsic detection covers vendor-less opaque tokens and
  private-key headers with no matching end marker. All other tools still get
  intrinsic redaction in `tool_calls`, `audit_log`, events, errors and logs.
- **Schedule API authority**: create/patch/delete routes require
  `LEO_API_TOKEN` when configured, validate `Host`, `Origin` and
  `Sec-Fetch-Site`, and record the actor as `api:<authenticated principal>`.
  Body-supplied channel/subject fields are data, never authority; ordinary
  command approval decisions keep their compatibility wrapper.
- **Surface capability manifest**: only TUI, web/Electron streams and Telegram
  with native callbacks can present a card. Unknown surfaces and notify-only
  IPCs are incapable by default. No approval authority is derived from a
  notification.

Evidence: `tests/test_schedule_security.py`, `tests/test_secret_sinks.py`,
`tests/test_unattended_gate.py`, `tests/test_schedule_compat.py` and
[docs/SCHEDULES.md](docs/SCHEDULES.md).

## Residual risks

- **Browser is not uniformly BLOCKED**: the persistent per-actor browser
  operator starts its Playwright/Chromium backend lazily on the first browser
  tool call when the backend is installed, with no platform credential. What
  remains blocked is *live delivery verification* (no real service credentials
  were supplied) and the optional live-browser test, which is opt-in.
  Private/loopback navigation is denied by the SSRF policy and screenshots are
  confined to the workspace.
- **Messaging and desktop**: live messaging adapters remain BLOCKED without
  platform credentials; macOS desktop actions need OS permissions and are only
  mock/dry-run tested. Sandboxing uses path/permission checks rather than OS
  isolation because no container is available.
- **Model gateway is environment-conditional**, not universally blocked:
  available with `ANTHROPIC_AUTH_TOKEN` plus `LEO_OPENCODE_SESSION`, mock
  fallback otherwise. No fresh live call is claimed here.

**Deep-audit gaps that are now closed** (they were listed as open in the
2026-11-18 snapshot and must not be read as current residual risks):
task-claim atomicity, signed webhook delivery IDs/timestamps, per-agent tool
permission enforcement, `extra_header_env` secret scrubbing, and DNS-rebinding
resistance. Evidence: [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md).

Still-open items are listed in [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md):
`/events` and `/metrics` lack a keyset-pagination cursor, tool-call counters
are per-process, memory recall has no per-request opt-out, and per-principal
scoping is deliberately not implemented. Scheduled scripts remain restricted
operator-provisioned executables, not a sandbox, and no live Telegram or gateway
smoke was performed for this release; both limits are recorded in the schedule
release receipt.
