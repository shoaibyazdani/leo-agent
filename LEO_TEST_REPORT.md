# LEO Test Report — 2026-11-18

## Commands and results

```bash
python -m pytest -o addopts='' -q
# 403 passed, 8 skipped in ~51s — HISTORICAL snapshot on 2026-11-18 (full
# working tree then, including the in-flight channel matrix; the committed
# milestone baseline was 171). Not the current roster; re-measure before use.

LEO_LIVE_BROWSER=1 python -m pytest tests/test_browser_live.py -o addopts='' -q
# 1 passed in 1.51s (real Chromium, local HTTP fixture page)
```

The 8 skips are opt-in live/optional-crypto checks (including the live-browser
check); they are skipped in the default offline suite and pass when explicitly
enabled with the required local setup. No test requires network access,
credentials or paid API calls.

## Coverage by module

| Module | Tests | What is verified |
|--------|------:|------------------|
| `tests/test_foundation.py` | 5 | Config, redaction, SQLite roundtrip, event bus, permission tiers/approval/audit |
| `tests/test_app.py` | 3 | Composition, chat with configured provider, memory persistence across restarts |
| `tests/test_api.py` | 1 | Health, chat, tasks, skills, events, metrics, evaluation, audit, dashboard, tool execution |
| `tests/test_models.py` | 16 | Provider adapters (mock HTTP), SSE streaming, retries, tool-call normalization, usage, router, context |
| `tests/test_router_default.py` | 2 | Configured default preference, secret-safe gateway headers |
| `tests/test_tools.py` | 14 | Registry/schema validation, permissions, approvals, dry-run, timeouts, redaction, 9 built-ins |
| `tests/test_memory.py` | 10 | CRUD, filtering, ranking, links, versions, dedup/consolidation, retention, no silent fact promotion |
| `tests/test_agents.py` | 10 | 12 specialists, delegation, parallel/dependencies, limits, verification, worktree isolation |
| `tests/test_skills.py` | 14 | Skill lifecycle, 12 manifests, compatibility layer, integration factory stages and rollback |
| `tests/test_compat_skill_md.py` | 10 | Hermes/OpenClaw SKILL.md translation, gate/provenance semantics, advisory body scan, dangerous-frontmatter rejection |
| `tests/test_security_regressions.py` | 10 | Approval binding, webhook fail-closed/raw-body HMAC, API token auth, terminal allowlist |
| `tests/test_process_e2e.py` | 1 | Fresh-home CLI processes: doctor, mock chat, memory and task persistence across restarts |
| `tests/test_tasks.py` | 21 | Lifecycle, evidence gate, retries/dead-letter, cancellation, recovery, scheduler, webhooks, transition/cancellation guards |
| `tests/test_integrations.py` | 20 | Messaging adapters/webhooks, gateway, browser fake backend, computer operator dry-run |
| `tests/test_node.py` | 3 | Node auth/scope, health monitoring, HTTPS enforcement |
| `tests/test_evaluation.py` | 2 | Metrics/regression detection, gated improvement adoption |
| `tests/test_browser_live.py` | 1 (opt-in) | Real Chromium navigate/extract/screenshot/click on local page |
| `tests/test_desktop_shell.py` | 2 | Electron shell security settings and JS syntax |
| `tests/test_chat_tools.py` | 1 | Autonomous tool call inside a chat turn, with audit |
| `tests/test_channels_registry.py` | 18 | Auto-discovery, `BROKEN_MODULES == {}`, capability classification, fail-closed admission, honest record sync, dynamic webhook/poll routes, signed challenge echo |
| `tests/channels/` | 20 files | Per-channel mock tests for the 19 new adapters plus shared `test_wechat_crypto.py` (signature verification, normalization, send/poll) |

## Manual smoke evidence

- CLI: `doctor`, `ask`, `task create`, `skill list`, `model usage` all returned
  expected JSON/tables using the offline provider config.
- API: local uvicorn on `127.0.0.1:8765`; `/health`, `/chat`, `/tools`,
  `/tools/calculator/execute` (result 42), `/` dashboard HTML verified with curl.
- Live browser: Chromium navigated to a local HTTP fixture, extracted
  "LEO browser OK", produced a PNG screenshot.

## Definition of done (Master plan §20)

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Starts successfully | Tested | `test_app`, CLI `doctor` |
| 2 | CLI interaction | Tested | CLI smoke run |
| 3 | Configured model responds | Tested (offline provider); live is env-conditional, not universally blocked | `test_app`, `test_api` |
| 4 | Invokes tools | Tested | `test_tools`, API smoke |
| 5 | Permissions enforced | Tested | `test_foundation`, `test_tools` |
| 6 | Memory persists across restarts | Tested | `test_memory`, `test_app` |
| 7 | Task created and executed | Tested | `test_tasks` |
| 8 | Task resumes after interruption | Tested (recovery, no replay) | `test_tasks` |
| 9 | Agents delegate and return results | Tested | `test_agents` |
| 10 | Skills registered and tested | Tested | `test_skills`, compat tests |
| 11 | Real messaging integration works | BLOCKED — credentials absent; adapters mock-tested | `test_integrations` |
| 12 | Browser automation works | Tested (local controlled page) | `test_browser_live` |
| 13 | Model routing between configured models | Tested | `test_models`, `test_router_default` |
| 14 | Usage and execution metrics recorded | Tested | `test_models`, `test_evaluation` |
| 15 | Logs and audit trails available | Tested | `test_foundation`, API `/audit`, `.leo/logs/leo.log` |
| 16 | Tests pass | Tested (historical snapshot) | 403 passed, 8 opt-in skips |
| 17 | Installation/operation documented | Tested | `README.md`, docs |

## Deep-research round (2026-11-18)

Ten independent agents tested both upstream projects at pinned commits, ran
real offline chat/tool E2E tests against loopback mock providers (both upstream
projects completed a full turn), and audited LEO source, security, licensing and
327 real upstream `SKILL.md` files. Evidence and fix log:
[deep research synthesis](research/deep-2026-11-18/SYNTHESIS.md).

New regression coverage from that round: approval-ID binding, webhook
fail-closed/raw-byte HMAC, API token auth, terminal command policy, evidence
gate through delegation and `transition()`, cancellation guards, skill-gate
semantics and provenance. The previously green suite exposed 26 documented
defects; the P0 set and several P1s are now fixed and re-tested.

## Honest limitations

See [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md). No live credential-backed
integration or paid model call is claimed **by this report**. The model gateway
is available when the host supplies `ANTHROPIC_AUTH_TOKEN` and
`LEO_OPENCODE_SESSION` (`x-opencode-session`); otherwise it degrades to mock.
`LEO_PROGRESS.md` records a real-model pass, not re-verified here. Upstream
runtime results apply to the pinned commits recorded in the research reports
and may drift.

## Channel matrix round (2026-11-18)

The channel matrix landed after the milestone baseline recorded above. Full
suite on that day's working tree: **403 passed, 8 skipped** (historical
snapshot; previously 145 passed, 1 skipped). This is not a current count — the
tool/agent expansion added many test files since. The 8 skips were opt-in
live/optional-crypto checks, including the live-browser check; the default
offline run needs no network, credentials or paid API calls.

New test files:

- `tests/test_channels_registry.py` — 18 tests covering auto-discovery,
  `BROKEN_MODULES == {}`, capability classification, fail-closed central
  admission, honest record sync, dynamic webhook/poll routes and signed
  challenge echo.
- `tests/channels/` — 20 files covering the 19 new adapters plus shared
  `test_wechat_crypto.py`; mock-tested signature verification, message
  normalization and send/poll transports. No live delivery is claimed.

All 22 channel records remain `status="blocked"` / `verification="mock-tested"` /
`metadata.live_verified=False` until credentials are supplied. See the
[channel matrix](LEO_INTEGRATIONS.md#channel-matrix-2026-11-18) for the env-name
matrix and the honest limitations.
