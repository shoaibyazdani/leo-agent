# OpenClaw end-to-end runtime test (2026-11-18)

**Scope.** Runtime verification of upstream OpenClaw to define what "working" means and what LEO must match. This is an evidence report, not a LEO change request. No LEO source was modified; only this file was added under `<repo>/research/`.

**Test window.** 2026-11-17T21:06Z → 2026-11-17T21:40Z UTC (local date 2026-11-18). Node v24.22.0, npm 11.19.1, git 2.50.1, corepack 0.36.0. `pnpm` was **not** on PATH.

## 0. Artifacts, provenance, and safety posture

| Item | Value |
|---|---|
| Upstream | `https://github.com/openclaw/openclaw`, shallow clone at `$TMPDIR/leo-research-openclaw-e2e/openclaw` |
| Git commit | `75d5bf7e3b4a1825db30b2352803b765e1681aa2` (`2026-11-17 21:06:35 +0000`, `fix(test): isolate alternative SQLite backup scratch (#200785)`) |
| Published npm package tested at runtime | `openclaw@2026.11.4`, CLI build ID `2026.11.4-release-d843d6af39a4-2026-11-15T22-45-53.485Z` |
| Runtime install location | `$TMPDIR/leo-research-openclaw-e2e-harness/node_modules/openclaw` (isolated harness; not the clone) |
| Harness / logs | `$TMPDIR/leo-research-openclaw-e2e-harness/`, logs under `$TMPDIR/leo-research-openclaw-e2e-harness/logs/` |
| Isolated state | `HOME=$TMPDIR/leo-research-openclaw-e2e-harness/home`, state at `…/home/.openclaw/` |

Safety rules observed:

- Downloaded repository content was treated as untrusted data. Commands executed against it were explicitly requested install/test commands; all other execution used the separately installed npm package from the harness.
- All `HOME`/XDG state was redirected into the harness. No host OpenClaw config/credentials were read or written.
- Gateway was started with `--bind loopback`; `lsof` confirmed listeners on `127.0.0.1:19089` and `[::1]:19089` only. No LAN/tailnet/public bind, no tunnel, no public publishing.
- **No external credentials, API keys, or messaging tokens were used or invented.** The only token is the literal local test string `<loopback-test-token>` (a loopback RPC secret generated for this test). The model provider was a locally written mock; no paid/external provider was contacted.
- No destructive commands were run. In particular, the stale build lock was left in place (see §2).

Command wrapper: because macOS lacks GNU `timeout`, CLI probes ran through a harness wrapper that sets the isolated HOME/XDG variables, applies a hard timeout, and appends exact command/exit/stdout/stderr to a log:

```bash
OC=$TMPDIR/leo-research-openclaw-e2e-harness/node_modules/.bin/openclaw
RUN=$TMPDIR/leo-research-openclaw-e2e-harness/run.mjs   # run.mjs <timeoutMs> <logFile> <cmd> [args...]
node "$RUN" 60000 $TMPDIR/leo-research-openclaw-e2e-harness/logs/example.txt "$OC" <subcommand> [flags]
```

The logs record the effective absolute command (`/tmp/.../node_modules/.bin/openclaw …`) plus `EXIT`, `SIGNAL`, `STDOUT`, and `STDERR`. Below, exit codes are the CLI's own status as recorded by the wrapper.

## 1. Install and first-run reality

### 1.1 Published package (worked, but slow)

```bash
npm install --prefix $TMPDIR/leo-research-openclaw-e2e-harness openclaw@2026.11.4 \
  --ignore-scripts --no-audit --no-fund
# EXIT=0; "added 361 packages in 2m"; real 1m49.54s
# npm warn deprecated node-domexception@1.0.0: Use your platform's native DOMException instead
ls -l $TMPDIR/leo-research-openclaw-e2e-harness/node_modules/.bin/openclaw
# -> ../openclaw/openclaw.mjs
```

`--ignore-scripts` was deliberate: the harness install tests whether the shipped runtime works without lifecycle execution. It does. The README's documented global install is different:

```bash
npm install -g openclaw@latest --allow-scripts=openclaw
```

README lines 38–47 say that flag is for npm 12 / npm 11.16+, and that the curl installer (`curl -fsSL https://openclaw.ai/install.sh | bash`) provisions Node itself. No installer script was executed in this test.

### 1.2 Source checkout: npm at root fails exactly as documented

```bash
# cwd=$TMPDIR/leo-research-openclaw-e2e/openclaw
npm install --dry-run --ignore-scripts --no-audit --no-fund
# EXIT=1
# npm error code EUNSUPPORTEDPROTOCOL
# npm error Unsupported URL Type "workspace:": workspace:*
```

README line 96: "The repository is a pnpm workspace. Plain `npm install` at the repository root is not supported." Observed, verified, exit 1.

### 1.3 Source checkout: pinned pnpm install works but is heavy

```bash
corepack pnpm@12.15.0 install --ignore-scripts --frozen-lockfile --reporter=append-only
# EXIT=0; "Done in 3m 33.5s using pnpm v12.15.0"
# node_modules size after install: 2.6 GB
```

`packageManager` pins `pnpm@12.15.0+sha512…`; plain `pnpm` was absent and corepack had to download the pinned binary (a first `corepack pnpm --version` from the harness fetched 12.21.0). Install output included multi-hundred-MB downloads (`@openai/codex` ~149 MB, `onnxruntime-node` ~115 MB, `@github/copilot-sdk-darwin-x64` ~77 MB). This is the real friction: a source build is not a light `npm i` — it needs pnpm/corepack, multi-GB dependency disk, and several minutes even with lifecycle scripts disabled.

### 1.4 Build: two bounded attempts, neither completed

Attempt 1 (180 s budget):

```bash
corepack pnpm@12.15.0 build
# EXIT: null (ETIMEDOUT), SIGNAL=SIGTERM
# stdout tail: [native-protocol] swift/kotlin: generated; plugin Control UI builds;
# [tsdown-build] invocation 1/1 finished in 9.5s / 53.0s; invocation 1/2 finished in 1.0s
# dist/ artifacts were produced, but completion was not observed.
# stderr: many non-fatal rolldown [EVAL] warnings for direct eval in deps and
# extensions/browser/src/browser/page-function-runtime.ts
```

Attempt 2 (480 s budget):

```bash
corepack pnpm@12.15.0 build
# EXIT=1
# Error: Could not acquire …/openclaw/.artifacts/dist-artifacts.lock:
#   retained by PID 37672, started "2026-11-17T21:34:46.791Z", identity "unknown",
#   last seen "2026-11-17T21:34:46.791Z". Inspect owner.json and verify all
#   associated build/check processes, including detached descendants, have stopped;
#   then run `rm -rf -- '…/.artifacts/dist-artifacts.lock'` to release and retry.
#   PID death alone is not sufficient.
# [cause]: Error: file lock stale for …/dist-artifacts.lock/owner.json
# [ELIFECYCLE] Command failed with exit code 1.
```

The timed-out first build poisoned the artifact lock; the retry refused to start and printed a destructive `rm -rf` recovery instruction. Per test rules, the lock was **not** deleted and no third build was run. Verdict: source build is **blocked** in this session (timeout + stale lock), while partial `dist/` output exists. The failure mode itself is a valuable finding (§7).

## 2. Offline-safe CLI matrix

All runs used the isolated harness HOME and absolute binary path. Fresh state means `~/.openclaw/openclaw.json` did not exist.

| Command | Exit | Observed result | Verdict |
|---|---:|---|---|
| `openclaw --version` | 0 | `OpenClaw 2026.11.4 (d843d6a)` | Works |
| `openclaw --help` | 0 | Full command surface (~60 commands: `config`, `doctor`, `gateway`, `cron`, `skills`, `plugins`, `agent`, `health`, `tui`, …) | Works |
| `openclaw doctor --lint --json --non-interactive` (fresh) | 1 | `{"schemaVersion":1,"ok":false,"checksRun":40,"checksSkipped":30,"findings":[5]}`. Warnings: browser profile import, gateway auth missing token, `gateway.mode` unset, heartbeat cadence migration, Node 24.22.0 below recommended patch. | Partially works |
| `openclaw doctor --lint --json --non-interactive` (configured) | 1 | 4 findings; gateway warnings cleared, new `core/doctor/security` warning: plaintext `gateway.auth.token` in config. | Partially works |
| `openclaw doctor --json --non-interactive` (bare advisory) | 0 | Same style of JSON with `"ok":false` and 3 warnings — **exit 0 despite `ok:false`**. | Confusing but by design? See §7 |
| `openclaw skills list --json` | 0 | 55 skills with `source`, `eligible`, `missing.bins/env/config/os`, `userInvocable`, `modelVisible`. | Works |
| `openclaw skills check --json` | 0 | `summary: {total:55, eligible:22, modelVisible:22, commandVisible:21, disabled:0, blocked:0, agentFiltered:0, notInjected:0, missingRequirements:33}` | Works |
| `openclaw skills verify @qa-only-nonexistent/skill --json` | 1 | `{"ok":false,"error":{"type":"cli_error","message":"Skill \"skill\" not found on ClawHub. Run `openclaw skills search skill`…"}}` — network reachable, no credentials; note it echoes only the slug, not `@owner`. | Partially works (clear negative) |
| `openclaw config file` | 0 | `/tmp/.../home/.openclaw/openclaw.json` | Works |
| `openclaw config validate --json` (fresh) | 1 | `{"ok":false,"error":{"type":"cli_error","message":"file not found"},"valid":false,"path":…}` | Confusing error; see §7 |
| `openclaw config schema` | 0 | Full JSON Schema; used to resolve the `auth` mismatch below. | Works |
| `openclaw config patch --file … --dry-run --json` (with `auth:"none"`) | 1 | `checks.schema` failed: `models.providers.mock-local.auth: Invalid input (allowed: "api-key", "aws-sdk", "oauth", "token")` | Docs/runtime mismatch |
| Same patch without `auth` | 0 | `{"ok":true,"operations":13,"checks":{"schema":true,"resolvability":true,…}}`; apply exit 0 ("Applied 13 config update(s). Restart the gateway to apply.") | Works |
| `openclaw config validate --json` (after patch) | 0 | `{"valid":true,"path":…,"warnings":[]}` | Works |
| `openclaw plugins list --json` | 0 | 69 plugins, 43 enabled/`loaded`; `registry.source:"derived"` with diagnostic `persisted-registry-missing` ("Persisted plugin registry is missing or invalid; using derived plugin index."). | Works with diagnostic |
| `openclaw gateway status --json --timeout 5000` (fresh) | 0 | `bindMode:loopback`, `bindHost:127.0.0.1`, port 18789, `probeNote:"Loopback-only gateway…"`, `rpc.kind:"connect"`, `rpc.error:"connect ECONNREFUSED 127.0.0.1:18789"`. Exit 0 even when unreachable. | Partially works |
| `openclaw cron list --json --timeout 5000` (no token) | 1 | `gateway cron.list requires credentials before opening a websocket` + `Fix: configure gateway.auth token/password, pair this device, or pass --token/--password.` | Works (correct gate) |
| `openclaw cron list --json --timeout 5000 --token <loopback-test-token>` (no server yet) | 1 | `Gateway not reachable at ws://127.0.0.1:19089 (ECONNREFUSED)` + `Start it with openclaw gateway run…`, `Bind: loopback`. | Works (good error) |
| `openclaw health --json --token <loopback-test-token>` | 1 | `OpenClaw does not recognize option "--token". Try: openclaw health --help` | CLI flag inconsistency |
| `openclaw health --json` (gateway up, config token) | 0 | `ok:true`, 14 plugins loaded, no errors, `heartbeatSeconds:1800`, `defaultAgentId:"main"`, config hot reload active. | Works |
| `openclaw qa --help` | 1 | `OpenClaw does not know the command "qa". Did you mean this? openclaw qr` | Blocked in published package; `qa` mock lanes are source/dev-only |

One harness-method footgun (not an upstream defect): the first attempt passed each help string as a single argv element (`openclaw "doctor --help"`), which upstream correctly rejected as `OpenClaw does not know the command "doctor --help".` Re-running with proper argument splitting (`${=spec} --help`) made all 12 subcommand-help probes exit 0.

## 3. Config, gateway handshake, and mocked chat turn

### 3.1 Config used (no real credentials)

Applied with `openclaw config patch --file /tmp/.../config.patch.json5`:

- `gateway.mode:"local"`, `bind:"loopback"`, `port:19089`, `auth.mode:"token"`, `auth.token:"<loopback-test-token>"`, `controlUi.enabled:true`
- `models.mode:"merge"`, `models.providers["mock-local"] = { baseUrl:"http://127.0.0.1:19180/v1", api:"openai-completions", models:[{id:"mock-1",name:"Mock 1"}] }` — `apiKey` omitted; loopback auto-no-auth
- `agents.defaults.model.primary:"mock-local/mock-1"`, `update.checkOnStart:false`, `telemetry.enabled:false`
- Later patch: `agents.entries.main = {name:"Main"}`

Two validation surfaces disagreed after the first patch:

```text
openclaw config validate --json  -> EXIT 0 {"valid":true,...,"warnings":[]}
openclaw gateway status --json --timeout 5000 -> EXIT 0 but config.cli.valid=false,
  config.daemon.valid=false,
  issues:[{"path":"agents.entries","message":"agents.entries must contain at least one configured agent"}]
```

Adding `agents.entries.main` made both report valid. The two paths also auto-added `plugins.entries.anthropic/codex.config.sessionCatalog.enabled:false` and migration metadata.

### 3.2 Gateway startup and loopback proof

```bash
cd $TMPDIR/leo-research-openclaw-e2e-harness
HOME=… XDG_…=… OPENCLAW_GATEWAY_TOKEN=<loopback-test-token> \
node $TMPDIR/leo-research-openclaw-e2e-harness/node_modules/openclaw/openclaw.mjs \
  gateway run --port 19089 --bind loopback --auth token --token <loopback-test-token> --verbose
# started as a background process; pid 19109
```

```bash
lsof -nP -iTCP:19089 -sTCP:LISTEN
# node 19109 … TCP 127.0.0.1:19089 (LISTEN)
# node 19109 … TCP [::1]:19089 (LISTEN)     # both loopback only
curl -sS -o …/curl-root.txt -w 'HTTP_CODE=%{http_code}\n' --max-time 5 http://127.0.0.1:19089/
# HTTP_CODE=200  (Control UI shell HTML)
```

Handshake/health:

```text
openclaw gateway status --json --timeout 8000 --token <loopback-test-token>  -> EXIT 0
  port.status:"busy", listeners:[{pid:19109, commandLine:"openclaw-gateway"}]
  rpc.ok:true, rpc.kind:"connect", rpc.capability:"connected_no_operator_scope",
  rpc.auth.role:"operator", rpc.auth.scopes:[],
  rpc.server.version:"2026.11.4", rpc.server.buildId:"2026.11.4-release-d843d6af39a4-…",
  gateway.bindMode:"loopback", gateway.probeUrl:"ws://127.0.0.1:19089"
openclaw cron list --json --timeout 8000 --token <loopback-test-token> -> EXIT 0
  total:1; job "Memory Dreaming Promotion" (managed by memory-core)
openclaw cron list --json --timeout 5000 --token wrong-token -> EXIT 1
  "unauthorized: gateway token mismatch (use this gateway's gateway.auth.token or pair the device)"
```

Startup race worth noting: a `gateway status` run immediately after launch still reported `port.status:"free"` and `ECONNREFUSED`, while `cron list` against the same URL had **already succeeded**. A later `gateway status` reported `busy`, the listener, and `rpc.ok:true`. Probe ordering/readiness is not single-shot deterministic.

### 3.3 Mock provider chat turn

A small harness-only HTTP server (`/tmp/.../mock-provider.mjs`) listened on `127.0.0.1:19180`, served `/v1/chat/completions` (streaming and non-streaming) and `/v1/models`, and logged each request to `/tmp/.../logs/mock-provider-requests.log`. Reply body was the sentinel `MOCK_REPLY_OK`; no auth header was sent by OpenClaw (`credentialSource.kind:"none"`).

```text
openclaw agent --local --message 'Reply with the exact mock sentinel.' \
  --model mock-local/mock-1 --json --timeout 60
# EXIT=1
# "A Gateway is running for this state directory (pid 19109, port 19089).
#  Run without --local to use it, or stop the Gateway first (openclaw gateway stop)."
# (mock server received 0 requests)

openclaw agent --message 'Reply with the exact mock sentinel.' \
  --model mock-local/mock-1 --json --timeout 60
# EXIT=0
# {"runId":"bc411842-…","status":"ok","summary":"completed",
#  "result":{"payloads":[{"text":"MOCK_REPLY_OK","mediaUrl":null}],
#  "meta":{"provider":"mock-local","model":"mock-1",
#          "credentialSource":{"kind":"none"},"assistantTurns":1,
#          "durationMs":8847, …}}}
# mock server log: 1 × POST /v1/chat/completions with model "mock-1"
```

The JSON also exposed `systemPrompt.chars:51649` and `contextBudgetStatus.estimatedPromptTokens:21180` for a one-line prompt — a large fixed prompt overhead. Verdict: full local chat turn **works** with no external credentials; `--local` is intentionally blocked while a gateway owns the state dir.

## 4. Bounded upstream tests

Environment after `pnpm install --ignore-scripts` completed.

```bash
# cwd=$TMPDIR/leo-research-openclaw-e2e/openclaw
node scripts/run-vitest.mjs run --config test/vitest/vitest.unit.config.ts \
  src/poll-params.test.ts src/version.test.ts
# EXIT=1
# "[openclaw] Staging pinned Bun fork openclaw-v1.4.4-… (one-time download)…"
# Startup Error: "No supported SQLite library for Bun on macOS (x64). Apple's
# system SQLite fails OpenClaw's read-only database connections. … Install one
# with brew install sqlite, or set OPENCLAW_SQLITE_LIBRARY to a libsqlite3.dylib
# built for x64."
# [test] FAILED (exit 1)
```

Note: the project's documented wrapper stages a Bun fork and fails on this macOS host for lack of a suitable SQLite dylib. No Homebrew SQLite dylib was present; none was installed.

```bash
node node_modules/vitest/vitest.mjs run --config test/vitest/vitest.unit.config.ts \
  src/poll-params.test.ts src/version.test.ts
# EXIT=0 but stdout: "No test files found, exiting with code 0"
# (the unit config's include/exclude filters discarded the CLI file arguments;
#  stderr listed the excluded test paths). A green exit here is not proof.
```

Direct vitest without the upstream config did execute real tests:

```bash
node node_modules/vitest/vitest.mjs run src/poll-params.test.ts
# EXIT=0; Test Files 1 passed; Tests 3 passed; Duration 3.23s
node node_modules/vitest/vitest.mjs run src/version.test.ts
# EXIT=0; Test Files 1 passed; Tests 16 passed; Duration 2.88s
node node_modules/vitest/vitest.mjs run src/install-sh-version.test.ts
# EXIT=0; Test Files 1 passed; Tests 1 passed; Duration 619ms
```

Each successful run emitted non-fatal Vite warnings about `configLoader: 'native'` (JSON import without import attributes in `scripts/lib/control-ui-i18n-sync-plan.ts`). Total: 20 tests across 3 files passed; the documented runner path is blocked on this host.

## 5. Verdict table

| Area | Verdict | Evidence / blocker |
|---|---|---|
| Clone, provenance | Works | SHA `75d5bf7…`, 65,749 files checked out |
| Published npm install (isolated, no scripts) | Works | 361 packages, 1m50s, exit 0 |
| Root `npm install` from source | Blocked (expected) | `EUNSUPPORTEDPROTOCOL workspace:*`, exit 1 |
| Pinned pnpm source install | Works but heavy | 3m33s, 2.6 GB `node_modules`, exit 0 |
| `pnpm build` from source | Blocked | 180 s timeout (SIGTERM) with partial artifacts; retry exit 1 stale dist artifact lock, recovery requires deleting lock |
| CLI help/version | Works | exit 0, exact outputs captured |
| Doctor (lint / advisory) | Partially works | 40 checks, 30 skipped; warnings; `doctor --lint` exit 1 but bare `doctor --json` exit 0 with `ok:false` |
| Skills list/check/verify | Works | 55 skills / 22 eligible; verify reaches ClawHub and rejects nonexistent ref |
| Config schema/patch/validate | Partially works | good dry-run schema errors; fresh validate says "file not found"; validate vs gateway-status disagree on `agents.entries` |
| Plugins list | Works | 69 discovered, 43 loaded, derived-registry diagnostic |
| Gateway status / handshake / health | Works | loopback only, token auth, `rpc.ok:true`, build ID matched |
| Cron list | Works | credential gate, reachable/unreachable/wrong-token paths all exercised; 1 job |
| Loopback mock chat turn | Works | `MOCK_REPLY_OK` from local mock, `credentialSource.kind:"none"`, 1 provider POST |
| `agent --local` while gateway owns state dir | Blocked (intentional) | explicit actionable error |
| Published package `qa` mock lanes | Blocked | command absent; dev/source-only |
| Documented vitest wrapper | Blocked | Bun fork + missing SQLite dylib on macOS x64 |
| Direct vitest subset | Works | 20 tests / 3 files passed |

## 6. Ranked E2E lessons for LEO

1. **Make a loopback mock-provider chat turn the top E2E gate.** The single most informative upstream test was: isolated config → token-auth loopback gateway → ready RPC → one HTTP mock provider → sentinel reply. It exercised config, auth, handshake, model routing, and reply parsing without credentials. LEO has `tests/test_models.py::test_mock_provider_is_deterministic` and an in-process ASGI chat test in `tests/test_api.py`, but no process-level test that starts the real gateway/CLI and proves a sentinel round-trip.
2. **Two independent validation surfaces must agree.** Upstream `config validate` said valid while `gateway status` rejected the same config for missing `agents.entries`. LEO should add a contract test asserting "config accepted by CLI == config accepted by runtime startup", including the fresh/no-config case.
3. **Test auth negatives before any data RPC.** Upstream cleanly separated `requires credentials before opening a websocket`, `ECONNREFUSED`, and `unauthorized: gateway token mismatch`. LEO should lock these three messages/exit codes into tests so a future refactor cannot silently skip authentication or leak data on a bad token.
4. **Define exit-code and JSON semantics explicitly.** `doctor --lint --json` exits 1 on warnings while bare `doctor --json` exits 0 with `"ok":false`; `gateway status` exits 0 when unreachable; direct vitest exits 0 with "No test files found". LEO should specify, per command class, what exit 0/1/2 mean and test that `ok` agrees with the exit code, and that "0 tests" is never a silent pass.
5. **Shared flags must behave identically on shared commands.** `cron list` accepts `--token`; `health` rejects it. LEO should generate CLI flags from one schema and add a cross-command flag-parity test.
6. **Fresh-state bootstrap is a product surface.** On a new machine: `config validate` = "file not found", doctor says gateway mode/token unset, `cron list` says credentials required, `gateway status` reports default loopback port. LEO should ship a tested "first run" path that produces a runnable config, one command to verify readiness, and errors that name the exact fix.
7. **Test the install and test-runner paths, not just the code.** Upstream's documented `pnpm build` wrapper left a stale lock after a timeout, and its documented vitest wrapper failed on Bun/SQLite while a direct invocation worked. LEO should keep an install/verify smoke test (clean venv, `uv sync`, CLI entrypoint, one offline turn) and assert that its documented test command actually executes a nonzero number of tests.
8. **Keep discovery commands read-only and schema-driven.** Upstream `plugins list`/`skills list`/`config schema` worked without credentials and exposed machine-readable eligibility and missing requirements. LEO already has adapters; add stable JSON output contracts and assert that listing never mutates config/state.
9. **Handle startup races explicitly.** A status probe said "port free / ECONNREFUSED" while an RPC client had already connected. LEO should use bounded readiness retries and test status output only after a ready signal, or document that an early probe is advisory.
10. **Measure prompt and credential hygiene.** Upstream's 51,649-char system prompt for a one-line turn and its `plaintext secret-bearing config fields` doctor warning are both worth matching with telemetry/tests: assert prompt-size budgets and ensure locally generated tokens are flagged as secrets rather than treated as ordinary config.
11. **Pin docs to the shipped version.** Runtime was `2026.11.4` (build `d843d6a`) while cloned HEAD docs show `models.providers.<id>.auth: "none"`; the shipped schema accepts only `api-key|aws-sdk|oauth|token`. Omitting `auth` worked on loopback. LEO should test documentation examples against the exact release, not the moving HEAD.
12. **Treat test-matrix breadth as an explicit capability.** Upstream has 637 npm scripts, 69 discovered plugins, 55 skills, and 22k test files; even a bounded E2E pass touches many surfaces. LEO does not need parity of size, but should have an inventory test that catches regressions in skill/plugin discovery counts, capability gates, and command registration.

## 7. Confusing errors and docs-vs-reality mismatches (candidates to avoid in LEO)

- `config validate --json` on a fresh machine: `file not found` (exit 1), even though the daemon treats an absent config as valid/unreachable-but-fine in `gateway status`. Message does not say "create config with `openclaw configure`".
- Config validity disagreement: `validate` true vs `gateway status` false for `agents.entries`.
- `auth:"none"` from current docs rejected by the published schema with only an allowed-values list; no hint that loopback blank-key auto-no-auth is the release's equivalent.
- `skills verify @owner/slug` error names only the slug (`Skill "skill" not found`), making the documented ref format feel unrecognized.
- `health --token` rejected although sibling commands accept it.
- `gateway status` exit 0 with an unreachable `rpc`, plus a stale "port free" reading immediately after startup.
- `doctor --lint` exit 1 on warnings vs `doctor --json` exit 0 with `ok:false`.
- Build lock error instructs `rm -rf` of a lock path after a timeout and warns "PID death alone is not sufficient" — hostile to bounded/CI runs.
- Vitest wrapper's Bun/SQLite error suggests `brew install sqlite` or `OPENCLAW_SQLITE_LIBRARY`; neither was installed (out of scope). Direct vitest silently reported "No test files found, exiting with code 0" with the unit config.
- Published package lacks the documented-in-source `qa` mock lanes; `openclaw qa` suggests `qr` instead.

## 8. LEO test gaps exposed (no LEO files changed)

Existing LEO tests are predominantly in-process: `tests/test_api.py` uses `httpx.ASGITransport`, `tests/test_models.py` mocks provider HTTP, `tests/test_integrations.py` uses fake subprocess runners/webhook payloads. Add (as future work, not done here):

| Gap | Suggested test shape | Upstream evidence it should match |
|---|---|---|
| Process-level offline E2E turn | Start real gateway + local mock provider on loopback with a local token; assert sentinel reply and provider request count | §3.3 |
| Auth negative matrix | no token / wrong token / correct token across status, cron/health equivalent | §3.2 |
| Fresh-state config bootstrap | empty HOME: doctor/config validate/gateway status give actionable outputs; after baseline config all agree valid | §2, §3.1 |
| Doc-example vs shipped-schema check | Compare documented config snippets against `config schema` of the pinned release | `auth:"none"` mismatch |
| Exit-code/JSON consistency | For each CLI mode assert `ok` ↔ exit code; assert "no tests"/"no work performed" is never exit 0 | doctor, vitest, gateway status |
| Ready-after-start probe | Poll readiness with bounded retries; assert status matches an actual RPC | startup race |
| Secret hygiene | Locally generated token must be flagged by doctor/audit; no plaintext token in logs/JSON output | doctor `core/doctor/security` |
| CLI flag parity | Single flag registry; test that all gateway-backed commands accept the same auth flags | `health --token` |
| Install smoke (clean env) | Fresh venv, `uv sync`, `leo --help`, one offline chat turn; bounded timeouts and no lifecycle scripts | npm/pnpm install findings |
| Test-runner self-check | Documented test command must assert executed test count > 0 | vitest "No test files found, exiting with code 0" |

## 9. Reproduction index

All raw logs are in `$TMPDIR/leo-research-openclaw-e2e-harness/logs/`:

- `pnpm-install.log`, `npm-root-dryrun.log`, `pnpm-build.log`, `pnpm-build2.log`
- `cli-basics.txt`, `subcommand-help.txt`, `baseline-*.txt`, `skills-check.txt`, `skills-verify-help-attempt.txt`
- `config-*.txt`, `configured-*.txt`, `agents-entry-*.txt`, `config-schema.txt`
- `gateway-running-*.txt`, `gateway-status-after.txt`, `gateway-auth-wrong-token.txt`, `health-*.txt`, `curl-root.txt`
- `agent-local-mock.txt`, `agent-gateway-mock.txt`, `mock-provider-requests.log`
- `vitest-subset.log`, `vitest-direct.log`, `vitest-noconfig.log`, `vitest-version.log`, `vitest-install-sh.log`

Harness scripts (all outside the downloaded tree): `run.mjs`, `run-in-repo.mjs`, `summarize.mjs`, `mock-provider.mjs`, `config.patch.json5`, `agents.patch.json5`.
