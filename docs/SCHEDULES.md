# Proactive scheduling port: shipped contract

Status: Phases 0–10 are implemented on `feat/proactive-schedule-approvals`.
This document describes the LEO behavior that is actually in the tree. The
read-only source-project corpus remains the behavioral oracle; its tests were
never executed and are never LEO evidence.

## Authority and evidence

- LEO baseline: `18d8e6202e8fd643a96b6e8ea916b176debf99f3`.
- Source-project final head: `7cf9e92cd6277b327b36a51fb897cdc8403902d6`
  (read-only behavioral source, never executed).
- Machine-readable parity fixtures: `tests/fixtures/schedule_parity/`.
- LEO test evidence: `.venv/bin/python -m pytest`, JUnit XML and the release
  receipt under `scripts/release/evidence/`. Upstream test counts
  and CI results are not LEO evidence and are never copied into the receipt.

The frozen fixture set pins the source-project action names, shared schema,
field exposure, grammar examples, delivery tokens, literal card copy and the
four-section acceptance matrix (`Final-Head Parity`, `LEO Adaptations`,
`LEO Hardening`, `Deferred`). Source-project product names are neutralized in
public fixture text; exact source bytes remain bound by fixture hashes.

## Model tool surface

LEO registers nine model-callable tools through the one `ScheduleService`:

`schedule_create`, `schedule_list`, `schedule_get`, `schedule_update`,
`schedule_pause`, `schedule_resume`, `schedule_delete`, `schedule_run_now`,
`schedule_cancel`.

This nine-tool roster is an intentional LEO adaptation: the source project
exposes one compressed management tool with actions. Unknown tool names and
unknown schedule fields are rejected by validation, never ignored or displayed
as if they were honored. `schedule_pause` and `schedule_resume` change only the
enabled state: both take an expected `config_revision`, `schedule_pause` accepts
a bounded `paused_reason` for the audit trail, and neither re-anchors the
stored cadence or first run. `schedule_run_now` returns a durable background
task handle; `schedule_cancel` stops that run without deleting the job.

## Conditional scheduling guidance

`automation.proactive_mode` is `offer` by default; `off` removes the guidance
line only and `create` is an opt-in for a later evaluation campaign. No mode
bypasses a card or a create grant. A short, neutral paragraph is appended to
the system prompt only when the trusted turn context reports a surface that can
present a card, wait for a decision, show full text and rewrite its result.
Blocking CLI/API turns, scheduled or delegated turns, and turns with no trusted
context get no such guidance. The hint tells the model to list existing
schedules first, prefer a one-shot or finite run, call `schedule_create` only
for a clearly user-authorized future moment, describe a job as pending until
consent, and never retry after a decline.

Model-facing fields are `name`, `schedule`, `prompt`, `repeat`, `timezone`,
`deliver`, `failure_deliver`, `skills`, `catch_up`, `wrap_response`, `script`,
`monitor`, `no_agent`, `context_from`, `continuity`, `enabled_toolsets`,
`workdir`, `attach_to_session`, `pinned`, `paused` and `paused_reason`.
`provider`, `model`, `base_url`, `reasoning_effort`, `interpreter` and
`grace_minutes` are operator/programmatic-only and cannot be supplied by a
model. `request_key` is an idempotency control handled by trusted callers.

## Supported grammar

- ISO one-shots, with an offset or interpreted on the configured profile clock.
- `in <duration>` one-shots; a bare duration such as `30m`, `every 2h` or
  `every hour` recurs instead.
- Daily, weekday, weekend and named-weekday times, including comma/`and` day
  lists, and five- or six-field cron with named months/weekdays.
- Duration units: minutes (`m`, `min`, `mins`, `minute`, `minutes`), hours
  (`h`, `hr`, `hrs`, `hour`, `hours`) and days (`d`, `day`, `days`). A missing
  count means one; seconds and weeks are rejected. Counts are capped at about
  ten years, including absurd digit strings that would otherwise overflow.
- Repeats: omitted means forever; `once`, `one`, `1x`, numeric strings and the
  `forever` family normalize; zero, negative, fractional and boolean repeats
  are rejected by LEO hardening. Repeat 1 retires after one claimed run.
- Timezones: IANA names. A nonexistent local time advances to the first valid
  minute after a DST gap; a repeated local time uses the first fold. Existing
  legacy rows with no timezone remain UTC.
- `@every` is legacy-readable only: existing LEO rows still parse through the
  reader, while new schedules reject it with a stable `invalid_schedule` error
  and point at `30m` / `every 30m`. Legacy cadences are capped at the same ten
  years as new durations, so a corrupt row cannot create an unbounded interval.
  It is not a writable LEO extension and no source-parity claim is made for it.

Semantic equality is kind plus cadence/expression or exact instant. Relative
durations measure elapsed time; cron uses IANA wall time. An aware one-shot is
absolute; a naive one-shot uses the configured profile clock.

## Consent and the fail-closed divergence

Create cards show the validated job, exact first run, normalized repeat, full
literal prompt and every delivery/execution field. Choices are `once`,
`session`, `always`, `deny`; edits use `Allow`/`Deny` (`once`, `deny`). A card is
bound to one canonical proposal digest and config revision, is shown before it
can be resolved, and expires at the earlier of the approval timeout and the
first run it displayed. Grants never cover edits, lifecycle actions, future
tools or workspace trust; chat actors cannot resolve their own cards.

LEO does **not** port the source project's no-card auto-save. A model create or
semantic edit without an answerable surface returns a durable
`approval_required` proposal with zero schedule rows and zero approval rows;
an operator can still resolve it later. Trusted direct operator scheduling
(create/patch/delete over HTTP, operator CLI, dashboard) stays unasked but
audited as `api:<authenticated-principal>` or `operator:<surface>`.

The only exception to the no-card rule for model creates is a **live schedule
create grant** covering the current principal/profile or conversation. Such a
grant makes `schedule_create` save immediately with no card, in interactive
turns only; it never covers edits, scheduled, unattended or delegated-agent
turns. Prompts are reworded while a grant is live so the model does not describe
an immediate save as pending approval.

## Delivery grammar

Accepted tokens: omitted (`origin` when capturable, else `local`), `origin`,
`all`, `local`, `local:<id>`, `home`, `bot-chat`, `bot-chat:<profile>`,
`room:<id>`, bare platform home targets, `platform:chat[:thread]`, comma lists
and list/object inputs. `failure_deliver` uses the same grammar on a separate
lane. Unknown tokens are rejected before a schedule is persisted.

`bot-chat` and `bot-chat:<profile>` are durable local conversations, not
byte-identical source-project profiles. `room:<id>` needs a room-capable adapter
plus a configured default room channel, otherwise validation rejects it.
`home`, `local:<id>` and object targets are LEO-native extensions.

Preview and runtime use one resolver. Receipts are keyed by
`platform:chat:thread`; the result and failure lanes stay separate, a `nowhere`
receipt is honest rather than sending to a placeholder, and a legacy
`{channel: target}` caller keeps its old projection. Telegram cards are literal
plain text in lossless chunks; an over-cap body becomes `deny_only` with the
full card retained in the dashboard.

## Native execution contract

Every surviving field has a runtime consumer:

- `script` is an operator-provisioned executable path, off by default
  (`automation.allow_scripts: false`) and confined to
  `automation.scripts_root`. It uses an args array with no shell, minimal
  environment, workdir anchoring into allowed roots, timeout/byte caps and
  durable evidence. It is not a sandbox and is not inline model-authored code.
- `no_agent` is a distinct script-only reminder mode, not inline authoring or a
  per-job state directory.
- `monitor` is an egress-checked public HTTP(S) URL or the restricted script
  runner. First success baselines, unchanged suppresses, change wakes, and a
  failure uses the failure lane and never overwrites the good baseline.
  Headers, cookies, userinfo, private targets, browser execution and JavaScript
  rendering are unsupported and rejected.
- `context_from` injects bounded latest completed outputs with cycle rejection;
  `continuity` injects bounded own previous output; `workdir` anchors tools and
  scripts inside allowed roots (not an OS chroot).
- `enabled_toolsets` intersects the agent scope (`[]` means no tools); `skills`
  injects a frozen creation-time documentation snapshot; `wrap_response`
  controls delivered framing and participates in edit fingerprints.
- `attach_to_session` uses a durable job conversation; `pinned` stores the
  operator-owned provider/model at creation and fails closed with a needs-you
  notice when the target disappears; `grace_minutes` is operator-only late-run
  policy on cron and interval jobs. A legacy one-shot carrying a grace window
  is retired as missed rather than dispatched after the window; an edit that
  turns a recurring job into a one-shot must set the window to zero first.

Removed intermediate scope stays absent: watcher skills/templates, inline
script content, per-job script state directories, gate-context/watchdog and a
connector-call facade.

## Surface capability matrix

Only surfaces with full-text presentation, real inbound identity and terminal
rewrite can receive a card. Unknown surfaces are incapable by default.

| Surface | Card | Wait | Full text | Rewrite | Max card chars |
|---|---|---|---|---|---|
| `tui` | yes | yes | yes | yes | 8000 |
| `web` | yes | yes | yes | yes | 20000 |
| `electron` | yes | yes | yes | yes | 20000 |
| `telegram` | yes | yes | yes | yes | 4096 |
| `cli`, `api` | no | no | yes (`cli`) | no | notify only |
| `a2a`, `whatsapp`, `whatsapp_web`, `agent`, `scheduled`, `chat`, unknown | no | no | no | no | 0 |

Electron adds a notification-only IPC: a notification never carries approval
authority. Notifier-blind, send-only and split gateways that do not advertise
the card protocol produce `approval_required` proposals instead of cards.

## Security hardening

- Scheduled/unattended calls deny MEDIUM/HIGH outbound, browser, computer,
  plugin and shell side effects before `PermissionManager`, before quota and
  before any handler. One durable `automation.needs_you` notice is emitted per
  refusal; no invisible approval row is created.
- Prompts are scanned before persistence and again at execution; a tampered
  stored prompt fails the task closed with a needs-you notice and no agent run.
- Secret preflight precedes authorization, quota, dispatch and persistence.
  Refused tools persist only a digest plus a stable category/rule. All tools
  get intrinsic secret redaction at the sink; vendor-prefixed keys, private-key
  headers and unmatched high-shape tokens are detected without a prefix catalog.
- API schedule mutations require `LEO_API_TOKEN` when configured, validate
  `Host`, `Origin` and `Sec-Fetch-Site`, and record `api:<principal>` as the
  actor. Ordinary command approvals keep their `approved`/`confirm_action`
  compatibility wrapper.
- Occurrences are claimed atomically with task/cursor/counter writes and events
  are published only after commit; existing delivery receipts and one-shot
  retirement semantics remain authoritative.

## Centralized copy, contracts and intentional test updates

LEO has no locale subsystem and no system-prompt cache; the system prompt is
rebuilt every turn. Card copy is centralized English in `KIND_CONFIG`, and
contract stability is checked through DTO, event, API and UI projections instead
of locale catalogs or cache invalidation.

Changed assertions from the port are intentional contract updates, never waived
regressions:

| Area | Intentional update | Evidence |
|---|---|---|
| Tool roster | four legacy tools → seven final-head tools → nine with explicit pause/resume | `tests/test_tools_automation.py`, `tests/test_schedule_compat.py`, `tests/test_secret_sinks.py`, `tests/test_tool_agent_integration.py` |
| Specialist grants | one card through the canonical proposal, exact gates kept for enqueue/delete/trigger | `tests/test_agents_automation_operator.py` |
| Delivery projection | `platform:chat:thread` receipts plus legacy `{channel: target}` view | `tests/test_schedule_delivery_receipts.py`, `tests/test_gateway_delivery.py` |
| Approval UI | shared `LEO_APPROVALS` renderer with literal text and choices | `tests/test_approval_ui_contracts.py`, `tests/test_playwright_approval_cards.py` |
| Desktop IPC | notify-only `notify(title, body)`; no approval IPC | `tests/test_desktop_shell_ipc.py` |
| Gateway split | present/update/ack RPCs, mixed-version fallback to no card | `tests/test_gateway_split.py` |
| API route fixture | unasked schedule mutation tests now use a trusted loopback `Host` | `tests/test_tool_agent_integration.py`, `tests/test_schedule_security.py` |
| Docs truthfulness | status and this contract match the shipped tree | `tests/test_docs_truthfulness.py` |
| Conditional guidance | absent without an answerable trusted turn or with `proactive_mode: off` | `tests/test_proactive_guidance.py` |
| Pause/resume | explicit tools with edit consent, bounded reason and no cadence re-anchor | `tests/test_schedule_pause_resume.py` |

## Migration, rollback and replay

Drain the scheduler and stop workers before migrating. Back up the database
with a WAL checkpoint, then run initialization on a copy: duplicate active
names produce a durable blocker report and block startup with remediation
instructions. Occurrence keys move to the schedule ID with a legacy alias and a
backfill marker so an in-flight occurrence cannot produce two tasks. Restart
replays terminals, live-run cancels never resurrect a task, and a finite
schedule retires with a tombstone while a one-shot keeps its hard-delete path.

Legacy approval rows with a NULL `expires_at` are backfilled with
`created_at + 15 minutes` during initialization. This changes old behaviour:
an approval that was pending for more than 15 minutes at upgrade time becomes
unresolvable immediately after the migration. Resolve or let those approvals
expire **before** upgrading, or re-request them afterwards; the backfill does
not distinguish pending from terminal rows.

Rollback after jobs have executed requires restoring the backup or running the
documented down-migration: older code cannot read `kind=once` rows or the new
approval-card columns. A feature flag alone is not a rollback. Mixed-version
split gateways must not claim new schedule rows or present schedule cards.

## Verification limits

- Playwright DOM tests assert literal rendering, all choices and lossless
  pagination whenever a browser is installed. Chromium is absent in this
  checkout, so the browser tests skip; the static asset contract checks pass.
  No browser verification is claimed.
- Telegram MockTransport and the two-process gateway are covered by offline
  tests. No live Telegram send and no live gateway smoke was performed; those
  checks are recorded as unverified in the release receipt.
- A migration drill on a production copy, backup/restore and a PTY/TUI smoke
  are not part of the local test run and are marked unverified in the receipt.
- External delivery is at-least-once unless the provider supplies idempotency.
  A shown card proves presentation, not human attention.
- Task concurrency, script byte/time caps and monitor egress bound a scheduled
  job, but native scripts still run with the LEO process's OS privileges.

## Reproducing LEO evidence

```bash
cd /Users/macbook/LEO
.venv/bin/python -m pytest -q
.venv/bin/python scripts/release/check_schedule_release.py
```

The Phase 0 baseline command is documented in
`scripts/schedule_parity/run_baseline.py`; it is reproducible only at the
recorded baseline commit and lock hash. The current full-suite and honesty
receipt is LEO-owned and lives in
`scripts/release/evidence/receipt.json`.
