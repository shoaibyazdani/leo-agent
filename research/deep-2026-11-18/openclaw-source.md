# OpenClaw deep source audit (2026-11-18)

**Target:** canonical repository https://github.com/openclaw/openclaw (MIT, TypeScript)
**Pinned commit / exact SHA:** `75d5bf7e3b4a1825db30b2352803b765e1681aa2`
**Commit subject/date:** `fix(test): isolate alternative SQLite backup scratch (#200785)` / `2026-11-18T05:06:35+08:00`
**Local source copy (read-only, untrusted input):** `$TMPDIR/leo-research-openclaw-src/repo` (1.1 GB working tree, 65,749 checked-out files at clone time)
**Baseline document:** `<repo>/docs/RESEARCH_HERMES_OPENCLAW.md` (doc-level; this report deliberately goes to source)
**Method:** read-only static inspection (`git`, `find`, `rg`, `sed`, file reads) plus official docs in-tree. No build, install, test runner, interpreter, or downloaded executable was run from the source directory; downloaded files were treated as untrusted data, not instructions. No credentials, network services, or publishing. No files under `<repo>` were modified except this report. (Note: other files in the LEO worktree were observed being modified concurrently by another process during this research — e.g. `leo/security/permissions.py` at 02:25:22 local time — so all LEO-side line numbers are a snapshot from when each file was read, not a claim about the current working tree.)

---

## 0. Reproduction commands and observed outputs

```sh
mkdir -p $TMPDIR/leo-research-openclaw-src
cd $TMPDIR/leo-research-openclaw-src
git clone --depth 1 --filter=blob:none https://github.com/openclaw/openclaw.git repo
git -C $TMPDIR/leo-research-openclaw-src/repo rev-parse HEAD
git -C $TMPDIR/leo-research-openclaw-src/repo log -1 --format='%H%n%ad%n%s' --date=iso-strict
du -sh $TMPDIR/leo-research-openclaw-src/repo
```

Observed (verbatim relevant lines):

```
Cloning into 'repo'...
Updating files: 100% (65749/65749), done.
75d5bf7e3b4a1825db30b2352803b765e1681aa2
75d5bf7e3b4a1825db30b2352803b765e1681aa2
2026-11-18T05:06:35+08:00
fix(test): isolate alternative SQLite backup scratch (#200785)
1.1G	$TMPDIR/leo-research-openclaw-src/repo
```

Representative source queries (all run from the downloaded tree as **data**, not executed):

```
rg -n 'connectNonce' src/gateway --glob '!**/*.test.ts'      # connection.ts:325, connect-auth.ts:395
rg -n 'nonce|signedAt|deviceId' src/gateway/server/ws-connection/connect-auth.ts
rg -n 'groupPolicy' extensions/{telegram,whatsapp,discord}/src ...
rg -n 'configSchema|plugin manifest requires' src/plugins/manifest.ts
rg -n 'AutoApproveLocal|groupPolicy' src/config src/gateway
```

Canonical URLs for the pinned tree:
- https://github.com/openclaw/openclaw/tree/75d5bf7e3b4a1825db30b2352803b765e1681aa2
- https://github.com/openclaw/openclaw/commit/75d5bf7e3b4a1825db30b2352803b765e1681aa2
- License: https://github.com/openclaw/openclaw/blob/75d5bf7e3b4a1825db30b2352803b765e1681aa2/LICENSE

All file paths below are **absolute paths inside the pinned checkout** unless explicitly marked LEO paths. Findings marked **VERIFIED** were read directly at this SHA. **PARTIAL** means the code supports part of the claim or an important nuance changes it. **UNVERIFIED** means no source-level or runtime evidence was obtained in this pass. No runtime exploitation, live platform traffic, or performance testing was performed.

---

## 1. Gateway WebSocket control plane

### 1.1 Transport, frame schema, validation

- Default port: `$TMPDIR/leo-research-openclaw-src/repo/src/config/paths.ts:285` — `DEFAULT_GATEWAY_PORT = 18789`; `$TMPDIR/leo-research-openclaw-src/repo/src/gateway/server.ts:74` uses the same fallback. Default bind is `loopback` (`src/config/zod-schema.gateway.ts:87-97`, documented at `src/gateway/connection-details.ts:211`).
- Multiplexed WS + HTTP: `src/config/zod-schema.gateway.ts:76` describes "Single multiplexed port for Gateway WS + HTTP". HTTP upgrade path: `src/gateway/server-http-upgrades.ts` (rate limiting at lines 64, 243-288).
- Frame validation is **generated from TypeBox schemas**, not a hand-written JSON parser and not Ajv at runtime: `packages/gateway-protocol/src/protocol-validator.ts:6-20` defines `ProtocolValidator<T>` with an Ajv-compatible `errors` shape; schemas live under `packages/gateway-protocol/src/schema/` and validators under `packages/gateway-protocol/src/validators/workers.ts:7` (`validateConnectParams`) and `:77` (`validateRequestFrame`).
- WS message handler: `src/gateway/server/ws-connection/message-handler.ts:211-298` parses JSON, applies `validateRequestFrame`, and requires `validateConnectParams` for `connect`. Pre-auth frames are capped: `MAX_PREAUTH_PAYLOAD_BYTES = 64 * 1024` (`packages/gateway-protocol/src/payload-limits.ts:1`) and `MAX_QUEUED_GATEWAY_PREAUTH_FRAMES = 16` (`src/gateway/server-constants.ts:6`, enforced at `message-handler.ts:715` and `message-handler-loader.ts:13`).
- Role separation happens *before* method scope checks: `src/gateway/role-policy.ts:1-23` accepts only `operator`/`node` roles and gates node RPCs separately; a node connect also requires `role === "node"` **and** `client.mode === NODE` (`src/gateway/server/ws-connection/connect-admission.ts:56,251`). Session attachment only proceeds after `bindGatewayConnectOperatorAccess()` and profile-lifecycle checks (`src/gateway/server/ws-connection/connect-session.ts:580-608`), i.e. no unregistered client can attach a session.
- Method-level authorization uses operator scope sets — `CLI_DEFAULT_OPERATOR_SCOPES` at `src/gateway/method-scopes.ts:48-56` includes `operator.admin`, `read`, `write`, approvals, questions, pairing, talk-secrets. Scope names: `src/gateway/operator-scopes.ts:1-27`. Request classification and dynamic session-method scope resolution: `src/gateway/method-scopes.ts:60-345`, `src/gateway/methods/core-method-policy.ts`, `src/shared/session-method-scopes*.ts`. The `x-openclaw-scopes` header can only **narrow** scopes (`src/gateway/server/ws-connection/connect-admission.ts:122-152`), not broaden them — a good design to copy.

### 1.2 Handshake, nonce and device identity

**VERIFIED sequence:**

1. On WS open, the gateway generates a per-connection `connectNonce = randomUUID()` and emits `connect.challenge` with `{nonce, ts, capabilities}` — `src/gateway/server/connection.ts:325-334`.
2. Client replies with `connect` params containing `client`, `role`, `scopes`, `auth` (token/password/deviceToken/bootstrapToken), and `device` (id, publicKey, signedAt, nonce, signature).
3. `authenticateGatewayConnect()` in `src/gateway/server/ws-connection/connect-auth.ts:62-644` is the central gate. Device identity checks at **lines 378-414**:
   - `deriveDeviceIdFromPublicKey(device.publicKey)` must equal `device.id` (line 380-382; implementation `src/infra/device-identity.ts:70-80` — SHA-256 of the normalized raw Ed25519 public key).
   - `signedAt` must be within `DEVICE_SIGNATURE_SKEW_MS = 2 * 60 * 1000` (line 69, 384-389).
   - `device.nonce` must equal the server-issued `connectNonce` (lines 391-396).
   - `resolveDeviceSignaturePayloadVersion()` (`src/gateway/server/ws-connection/handshake-auth-helpers.ts:205-241`) tries v3 then v2; both payloads are pipe-delimited and bind `deviceId | clientId | clientMode | role | scopes | signedAtMs | token | nonce` (v3 adds normalized platform/deviceFamily) — `packages/gateway-client/src/device-auth.ts:22-51`. Signature verification is Ed25519 (`src/infra/device-identity.ts:84-89` → `src/infra/ed25519-signature.ts`).
   - A bad signature closes with 1008 and emits `gateway.auth.failed` security events (connect-auth.ts:350-377).
4. Admission and pairing: `admitGatewayConnect()` (`src/gateway/server/ws-connection/connect-admission.ts`) calls `checkGatewayWsBrowserOrigin` before granting credentials (lines 160-190, 370) and re-checks live authority after awaits — `resolveGatewayConnectPolicyFailure()` (connect-admission.ts:177-205) revalidates identity-tunnel currency, browser origin, role policy and shared-session generation. This "recheck after async" pattern is unusually careful and worth copying.
5. Pairing lanes: `src/gateway/server/ws-connection/connect-device-pairing.ts:186` `requirePairing()`, plan resolution in `connect-pairing-approval-plan.ts:107-168`, node-specific strict floor in `src/gateway/node-pairing-auto-approve.ts:50-67` (fresh, scopeless `role: node`, no browser/control-UI/webchat origin, attributable IP; exclude loopback trusted-proxy).
6. Scope ordering is deliberate: the device signs its requested scopes, then a live identity principal caps them **only after signature verification** (`connect-auth.ts:475-484`, comment at 475-476). A node key never carries operator scopes (lines 468-473). OpenClaw-ID device tokens are additionally bound to the device key that obtained them (`connect-auth.ts:435-448`, `src/gateway/openclaw-id/device-token-binding.ts:83`). Connection finalization re-applies caps via `resolveEffectiveConnectionScopes()` (`connect-admission.ts:153-171`). This "signed request, server-side authority ceiling" ordering is the single best auth pattern for LEO to copy.

**Sharp edge (VERIFIED):** `shouldAllowSilentLocalPairing()` in `handshake-auth-helpers.ts:60-106` deliberately allows *metadata-only* reconnect refreshes to self-approve even when the operator sets `autoApproveLocal: false`, for direct-local native app UI and container-local lanes. The comment explicitly says this avoids "approval churn"; it is intentional, but it means that config flag is not an absolute "always ask" switch. Scope-upgrade silent pairing is restricted to `authMethod` `none`/`token`/`password` (lines 95-104); identity-proxy and bearer-device-token rows keep a durable scope cap.

**Sharp edge (VERIFIED):** local CLI/operator clients get `operator.admin` by default (`src/gateway/method-scopes.ts:48-56`), and shared-secret HTTP bearer auth deliberately restores those full operator defaults (`src/gateway/http-auth-utils.ts:697-718`, comment at 710: "designed-as-is"). Possession of the gateway token is effectively root for the control plane. Also note `authorizeControlUiReadRequestOrReply()` grants full CLI defaults when no auth config exists on the local surface (`http-auth-utils.ts:298-309`), unless this is a remote ingress (which is rejected).

**Sharp edge (VERIFIED):** browser-origin policy has a break-glass config `gateway.controlUi.dangerouslyAllowHostHeaderOriginFallback` (`src/gateway/origin-check.ts:30-40`). Origin checks are only applied when a browser origin is present; non-browser local clients are expected to rely on loopback + device signature.

**Nonce lifetime (VERIFIED with limitation):** the nonce is per-WS-connection and there is no separate global nonce replay cache; replay outside the live connection is blocked by the 2-minute skew and by the signed payload binding to that connection's nonce. I did not find a cross-connection nonce burn list; I also found no evidence that a nonce can be reused to mint a second session on a different connection with the same device key and token (that needs the same challenge, which the attacker does not control). **UNVERIFIED:** formal replay/race proof.

### 1.3 Sessions

- Store path: `src/config/agent-store-source.ts:92` — `openclaw-agent.sqlite` inside the effective agent dir (normally `~/.openclaw/agents/<agentId>/agent/`). Multiple configured store paths and env overrides are handled at lines 84-110.
- Session key builder: `src/routing/session-key.ts:179-237` `buildAgentPeerSessionKey()`. DM `dmScope` defaults to `"main"` at line 193, so different DM peers of the same agent share one main session unless the operator configures `per-peer`, `per-channel-peer`, or `per-account-channel-peer` (enum at `src/config/zod-schema.agents.ts:129`). Group scope defaults to per-group (`session-key.ts:229-235`); `groupScope: "main"` collapses groups into the main session.
- Cross-channel identity links can remap a DM peer ID when scope is not `main` (`session-key.ts:196-205`, `resolveLinkedDirectPeerId()`).
- Subagent sessions are named `agent:<agentId>:subagent:<uuid>` (documented at `$TMPDIR/leo-research-openclaw-src/repo/docs/tools/subagents.md:13`; migration normalizer at `src/infra/state-migrations.session-store.ts:175-177`; spawn modes `isolated`/`fork` at `src/agents/subagents/spawn/subagent-spawn.types.ts:11` and `src/agents/tools/sessions-spawn-tool.ts:96-113`). Tools: `sessions_spawn`, `sessions_yield`, `subagents`; `context:"fork"` is restricted to same-agent spawns (`src/agents/tools/sessions-spawn-visible.ts:223-227`) and runtime `subagent` (`sessions-spawn-tool.ts:403-404`).
- Gateway session lifecycle is very large and highly modular (`src/gateway/session-*.ts`, hundreds of files); notable pieces: `session-access-authority.ts`, `session-sharing.ts`, `sessions-patch-admission.ts`, `server-session-key.ts`, `session-utils-store*.ts`. This is a mature but complex state machine: session publication, lineage, cold storage, worker placement, and subagent visibility are separate concerns. **UNVERIFIED:** crash-consistency under concurrent worker placement; no runtime test was run.

---

## 2. Channel adapters: where authorization actually happens

**VERIFIED entry points:** `extensions/{telegram,whatsapp,discord}/index.ts` each call `defineBundledChannelEntry` and export `telegramPlugin` / `whatsappPlugin` / `discordPlugin`. Implementations: `extensions/telegram/src/channel.ts:263`, `extensions/whatsapp/src/channel.ts:80`, `extensions/discord/src/channel.ts:153`; startup via `gateway.startAccount`.

**Central policy engine:** `src/channels/message-access/runtime.ts:196` `resolveChannelMessageIngressForOwner()` builds allowlists, pairing entries, routes and mention facts. Defaults at lines 92-93 are `dmPolicy ?? "pairing"` and `groupPolicy ?? "disabled"` **at this generic layer**. `src/channels/message-access/decision.ts:264-345` `decideChannelIngress()` evaluates route → optional pre-sender mention → sender → command → event → activation gates, returning `dispatch` only at the end. Sender gates: `src/channels/message-access/sender-gates.ts:44` (`senderGateForDirect`) and `:96` (`senderGateForGroup`). Mention logic: `src/channels/mention-gating.ts:88`. Sharp edge: `applyEventAuthModeToSenderGate()` (`sender-gates.ts:124`) converts a failed sender gate into `effect: "ignore"` for non-`inbound` events when other gates authorize — sender allowlists are not universal across event kinds.

**Per-channel enforcement is adapter-local:**
- Telegram: `extensions/telegram/src/bot-handlers.inbound-authorization.ts:286` `authorizeInboundMessage()` (DM 329-359; group/event 368-397), with memoized resolver results; mention drop at `bot-message-context.body.ts:369`; session binding at `bot-message-context.session.ts:494-506`.
- WhatsApp: `extensions/whatsapp/src/inbound-policy.ts:29` `resolveWhatsAppInboundPolicy()` and `:81` `resolveWhatsAppIngressAccess()` (`dmPolicy ?? "pairing"` at line 39), carried into `extensions/whatsapp/src/inbound/admission.ts:93-115`.
- Discord: `extensions/discord/src/monitor/message-handler.dm-preflight.ts:29-80`, `message-handler.preflight-channel-access.ts:46-60`, sender/mention drops in `message-handler.preflight.ts:266-283, 504-509, 634-670`. `allowBots` defaults true in this file at line 150.

**Untrusted message → agent run:** peer identity/chat IDs come from provider payloads; `src/routing/resolve-route.ts:343,474` → `buildAgentSessionKey()` → `buildAgentPeerSessionKey()`. Turn execution: `src/channels/turn/run-channel-turn.ts:185-219`. **Important boundary observation (VERIFIED):** the generic dispatcher trusts the adapter-supplied `admission` and only drops on `kind !== "dispatch" | "observeOnly"` (lines 188-204); it does **not** re-run channel policy. Correctness therefore depends on every adapter calling the central gate. This is a structural weak point to avoid in LEO: authorize once in a central choke point that *all* adapters must traverse, not in each adapter.

**Group-policy nuance (VERIFIED, partially refutes a naive reading):** `resolveOpenProviderRuntimeGroupPolicy()` (`src/config/runtime-group-policy.ts:54-62`) falls back to `groupPolicy: "open"` when a provider block exists but neither an account policy nor `channels.defaults.groupPolicy` is set (lines 20-27). Telegram and WhatsApp/Discord call this resolver (`extensions/telegram/src/group-access.ts:61`, `extensions/whatsapp/src/inbound-policy.ts:46`, `extensions/discord/src/channel.ts:487`). However, several channel schemas declare a root default of `allowlist` (`src/config/zod-schema.channel-messaging-common.ts:71`) and docs state provider schemas default to allowlist (`docs/gateway/config-channels/shared-policies.md:30`). So "configured providers default open" is **conditional on config materialization**, not universally true. This is exactly the kind of default-resolution ambiguity LEO should avoid with a single, explicit, fail-closed resolver.

**Webhook trust:** Telegram webhook secret is constant-time compared (`extensions/telegram/src/webhook.ts:128-154`, config requires it with `webhookUrl` at `extensions/telegram/src/config-schema.ts:46-68`); Mini App init data is HMAC + age checked (`extensions/telegram/src/miniapp/init-data.ts:40-48`). WhatsApp uses a Baileys socket (no inbound provider webhook signature surface); Discord uses the gateway WS. **UNVERIFIED:** no separate Discord Interactions Ed25519 verifier was found; if such an endpoint exists in another package it was not located.

**Libraries (VERIFIED):** `extensions/telegram/package.json:8-10` uses `grammy` 1.46.0; `extensions/whatsapp/package.json:13` uses `baileys` 7.0.0-rc14; `extensions/discord/package.json:10-14` uses `@discordjs/voice`, `discord-api-types`, `ws` — i.e. a hand-rolled gateway client, **not** the official `discord.js` bot library. The doc's "official Discord gateway bot" is accurate at the protocol level but not at the SDK level (**PARTIAL**).

---

## 3. Plugins, hooks, cron, context, memory, providers

### 3.1 Native plugins — in-process by construction

- Manifest: `PLUGIN_MANIFEST_FILENAME = "openclaw.plugin.json"` (`src/plugins/manifest.ts:46`); 256 KiB read cap and reserved IDs at lines 47-53; `loadPluginManifest()` requires a non-empty `id` (lines 226-231) and an object `configSchema` (lines 233-236) and parses JSON/JSON5. Contract keys are enumerated in `src/plugins/manifest-contract-keys.ts:3-48`; type metadata in `src/plugins/manifest-types.ts:174-241`.
- Loading: Jiti from `src/plugins/plugin-module-loader-cache.ts:13`, native `createRequire(...).require(modulePath)` from `src/plugins/native-module-require.ts:294-322`, module invocation at `src/plugins/loader-module-runtime.ts:217-232`. **VERIFIED: no process/VM sandbox.** `register(api)` is called synchronously (`src/plugins/loader-runtime-candidate.ts:651-659`; `src/plugins/api-lifecycle.ts:62-80` rejects async registration).
- API surface: `buildPluginApi()` in `src/plugins/api-builder.ts:117-152`, with default registrars at lines 14-92 including `registerTool`, `registerHook`, `registerHttpRoute`, `registerGatewayMethod`, `registerProvider`, `registerContextEngine`, `registerMemoryCapability`, `registerAgentHarness`, `registerTrustedToolPolicy`, `registerSessionSchedulerJob`. SDK helper `definePluginEntry` at `src/plugin-sdk/plugin-entry.ts:254`.
- Trust gates: `plugins.allow`/`deny` semantics documented at `src/config/schema.help.agents.ts:16-30`; install-time ClawHub scan confirmation at `src/plugins/clawhub.ts:665-683` and source scanning at `src/plugins/install-package.ts:134-154` (runtime entry `src/plugins/install-security-scan.runtime.ts:660-667`); note that a `warn` policy finding can be **operator-approved with a fingerprint** (`install-security-scan.runtime.ts:530-558`), so the scanner is an advisory gate, not a hard sandbox; non-bundled plugins need `hooks.allowConversationAccess=true` for conversation hooks (`src/plugins/hook-policy-decisions.ts:4-18`); prompt-modifying hooks can be blocked (`src/plugins/registry-registrars.ts:644-652`). Hot reload is generation-based with drain/rollback (`src/gateway/server-plugin-reload.ts:54,180,256,342,573`), and dev reload of bundled code explicitly warns "replaced code stays in memory until the Gateway restarts" (`src/plugins/bundled-plugin-dev-reload.ts:27-39`). A plugin bug is a gateway bug.
- The shipped `openclaw security audit` has a dedicated plugin-trust check set, including a critical warning for unpinned extensions with native skill commands and a specific warning that phantom `plugins.allow` entries could be exploited by registering a new plugin with an allowlisted ID (`src/security/audit-plugins-trust.ts:211-290`). Good idea for LEO.

### 3.2 Hooks and cron

- Workspace/managed hooks are dynamically imported into the gateway process (`src/hooks/loader.ts:27-42,128-133`), and the loader literally logs "Hooks are trusted local code" (lines 121-124). Webhook `transform` modules are also imported in-process (`src/gateway/hooks-mapping.ts:515-534`).
- Modifying hooks run sequentially with merge/terminal-result semantics (`src/plugins/hooks.ts:527-640`); void hooks run concurrently with `Promise.allSettled` (lines 447-500). Priority sort descending (`hooks.ts:137-147`). The fail-closed set is assembled from two places: `createHookRunner()` defaults to `before_agent_run` and `before_channel_ingress` fail-closed (`src/plugins/hooks.ts:171-176`), while `ensureGlobalHookRunner()` adds `before_install` and `before_tool_call` (`src/plugins/hook-runner-global.ts:26-37`); everything else defaults fail-open (`hooks.ts:187`). **VERIFIED** — the earlier doc-level list omitted `before_channel_ingress`.
- No durable hook queue / retry / exactly-once (documented at `docs/automation/hooks.md:139`), which matches the architecture: hooks are in-process event callbacks, not a message bus.
- Cron state persisted in SQLite (`src/cron/store.ts`); scheduler is a worker-hosted actor using unref'd `setTimeout` and `BroadcastChannel` wakeups with `MAX_CRON_TIMER_DELAY_MS` (`src/cron/store/scheduler-actor.worker.ts:56,186-205`); startup recovery runs before dispatch and handover is supported (`src/cron/service/ops-lifecycle.ts:55-105,137-140`). Exactly-once is explicitly not promised (`docs/automation/cron-jobs/how-it-works.md:58`).
- **CLI alias correction (VERIFIED / refutes doc phrasing):** `src/cli/cron-cli/register.ts:30-31` declares `.command("cron").alias("automations")`. `cron` is primary; `automations` is the alias. The existing research doc says the reverse.

### 3.3 Context engine, memory, providers

- `ContextEngine` contract: required `info`, `ingest()`, `assemble()`, `compact()`; optional `bootstrap`, `ingestBatch`, `afterTurn`, `commitTurn`, `maintain`, `dispose` (`src/context-engine/types.ts:347-538`). `info.ownsCompaction` disables built-in auto-compaction (lines 182-188). Slot `plugins.slots.contextEngine` defaults to `legacy` (`src/context-engine/registry-selection.ts:13-46`); failed non-default engines are quarantined in-process with a persisted health mirror and downgraded (`src/context-engine/registry-quarantine.ts:30-60`, `quarantine-health.ts:26-48`). `delegateCompactionToRuntime()` lets plugins reuse stock compaction (`src/context-engine/delegate.ts:45-74`).
- Memory: `extensions/memory-core/openclaw.plugin.json` declares `kind: memory`; `extensions/memory-core/index.ts:316-347` calls `api.registerMemoryCapability`. FTS5 + `vec0` tables in `src/infra/database-engine/sqlite-memory-search-fragments.ts:21,65`; canonical tables `memory_index_chunks` / `memory_index_chunks_fts` (`sqlite-search-schema.ts:13-14`). Bootstrap file order is `AGENTS.md, SOUL.md, IDENTITY.md, USER.md, BOOTSTRAP.md, MEMORY.md` (`src/agents/workspace-bootstrap-policy.ts:30-37`). Pre-compaction flush writes append-only `memory/YYYY-MM-DD.md` and tells the model those bootstrap files are read-only (`extensions/memory-core/src/flush-plan.ts:19-36`). PostgreSQL is a real supported engine (`extensions/memory-core/src/memory/manager-search.worker.ts:137`, `manager-search-postgres.*`). Sharp edges: Unicode FTS token equivalence is hand-rolled (`extensions/memory-core/src/memory/keyword-query.ts:7-27`); vector KNN depends on sqlite-vec with a subprocess fallback (`manager-search-vector.ts:39-43`). **UNVERIFIED:** index corruption/concurrent-writer behavior.
- Providers: `api.registerProvider()` normalizes, rejects duplicate IDs, records `providerIds` and reserves catalog ownership (`src/plugins/registry-registrars-providers.ts:27-57`); contract in `src/plugins/provider-plugin.types.ts:97-120,393-398,618-640`; failover calls plugin hooks in-process but validates returned reason codes (`src/plugins/provider-failover.ts:12-44`). SDK helper `createProviderApiKeyAuthMethod` at `src/plugin-sdk/provider-entry.ts:53`. **UNVERIFIED:** live OAuth rotation/failover outcomes.

---

## 4. Execution, sandbox, browser, nodes

### 4.1 Exec approval ladder — real model

**VERIFIED:** the five documented modes exist as `ExecMode = "deny" | "allowlist" | "ask" | "auto" | "full"` (`src/infra/exec-approvals-core.ts:18`; config enum at `src/config/zod-schema.agent-runtime.ts:223`). They are shorthands over two persisted axes: `security ∈ {allowlist, full, deny}` and `ask ∈ {always, off, on-miss}` (`src/infra/exec-approvals-config.ts:21-27`; mapping table at `docs/tools/exec.md:222-230`). `auto` uses a native reviewer returning allow/deny/ask, with three consecutive denials escalating to a human (`docs/tools/exec.md:234`, `src/agents/bash-tools.exec-host-gateway.ts:729-741`).

Where enforcement actually happens:
- `src/infra/exec-approvals-policy.ts:14-32` `requiresExecApproval()`, `:34-42` `minSecurity()/maxAsk()`; `:88-105` host-policy-change assertion.
- `src/agents/bash-tools.exec-host-node.ts:115-139` hard-throws on deny/allowlist-miss; `:174-219` re-reads live policy on timeout so a concurrent revoke wins.
- `src/agents/bash-tools.exec-host-shared.ts:125-150,251-284` prevents host files from broadening caller contracts and prevents timeout fallbacks from satisfying explicit-human boundaries.
- `src/agents/bash-tools.exec-run.ts:290-334` composes pre-dispatch policy, including `bypassApprovals` for explicit elevation.
- Matching: `src/infra/exec-allowlist-pattern.ts:39-95` glob compilation and normalization; `src/infra/exec-command-resolution.ts:351-428` `matchAllowlist()` — bare `*` is a deliberate bypass of the resolved-path guard, basename patterns do not apply to `./tool` or absolute paths, generated path-only `allow-always` entries are skipped because argv can change. Arg matching uses cwd-bound SHA-256 hashes (`exec-command-resolution.ts:252-261`). All chain groups must satisfy (`src/infra/exec-approvals-allowlist.ts:731-754`).
- Approval durability: 30-minute default timeout (`src/infra/exec-approvals-core.ts:267`); base-hash optimistic concurrency for writes (`src/gateway/server-methods/exec-approvals.ts:43-64`); gateway methods require `operator.admin`; argv/cwd/env and file operands are bound into approvals (`src/infra/system-run-approval-binding.ts:70-92`, including `SYSTEM_RUN_DENIED: approval script operand changed before execution` at line 38).

**Sharp edge (VERIFIED, documented by upstream):** exec does not read or lint script files; a shell wrapper that runs a Python/Node file can hide the real program from approval text, and under `security=full` or `ask=off` (or in a sandbox) the command simply runs (`docs/tools/exec.md:170`). `tools.exec.strictInlineEval` is a separate **opt-in, default false** mitigation for inline interpreter-eval forms (`docs/tools/exec.md:260`). This is a strong argument for LEO to bind approvals to resolved interpreter + file hash, not to the typed command string.

### 4.2 Durable background jobs

- Process registry: `src/agents/bash-process-registry.ts:24-36` — TTL 30 min (clamp 1 min–3 h), `MAX_FINISHED_SESSION_COUNT = 50`, `MAX_FINISHED_SESSION_OUTPUT_CHARS = 2,000,000`.
- Durable store persists only restart facts, never env/output/requester authority (`src/agents/bash-durable-process-store.ts:14-21`). Restart adoption rechecks notification authority (`src/agents/bash-process-durable.ts:242-279`); jobs without a service unit settle as lost (line 26).
- **VERIFIED limitation:** durability requires Linux systemd ≥250 or a macOS logged-in GUI user, and explicitly rejects sandbox, node, worker and one-shot execution (`src/agents/bash-process-durable.ts:305-345`). Docs state limits of 16 services/session and 64/gateway and that the slot is held from admission through running (`docs/gateway/background-process.md:34`; admission at `src/agents/bash-tools.exec-run.ts:280`). `process` tool caps wait at 10 min and poll at 30 s (`src/agents/bash-tools.process.ts:70,112,138-143`).
- Upstream explicitly tells the agent that an ordinary background process "is not a saved, restartable automation" (`src/agents/bash-tools.descriptions.ts:18`). Keeping transient background jobs and scheduler automations conceptually separate is a good design to copy.

### 4.3 Sandbox boundary

- Configuration: `src/agents/sandbox/config.ts:212-264` — default `mode: "off"`, default backend `docker`. Applied to tool execution through `src/agents/core-coding-tools.ts:205-450`; `src/agents/bash-tools.exec-run.ts:337-342` refuses `host=sandbox` with no runtime.
- Docker backend builds `docker|podman exec -i [-t] -w ... --env-file ... <id> /bin/sh -lc <command>` (`src/agents/sandbox/docker-backend.ts:164-220`).
- Validation blocks host binds, `/etc`, Docker socket, credential dirs, `unconfined` seccomp/AppArmor, `network=host`, `container:*` namespace joining — `src/agents/sandbox/validate-sandbox-security.ts:22-52,93-110,313-380`. Break-glass: `dangerouslyAllowContainerNamespaceJoin=true`.
- **VERIFIED caveats:** `capabilities.browser` is Docker-only (`docker-backend.ts:169-172`); `browser.allowHostControl` is a separate grant (`src/agents/sandbox/config.ts:142`, `src/agents/sandbox/context.ts:601`); node/worker/one-shot and elevated host exec remain outside the container by design. The gateway itself stays on the host; only tool execution moves. Sandbox is opt-in and explicitly documented as "not a perfect security boundary" (**PARTIAL** — this phrase was not re-located in source during this pass; the code's break-glass knobs and excluded paths support the underlying caution).

### 4.4 Browser control service

- HTTP control server binds only `127.0.0.1` (`extensions/browser/src/server.ts:83-109`) and fails closed if no auth can be established (lines 62-74). Auth is derived from gateway auth or generated/persisted (`extensions/browser/src/browser/control-auth.ts:16-38,90-145`); constant-time token/password compare (`http-auth.ts:51-65`).
- CSRF middleware: `extensions/browser/src/browser/server-middleware.ts:15-53` rejects mutating requests with `Sec-Fetch-Site: cross-site` or non-loopback Origin/Referer. **Sharp edge (VERIFIED):** requests with *no* Origin/Referer pass this middleware to the auth check (line 35); security rests on the bearer/password credential, not origin alone.
- `browser.request` gateway method requires `operator.admin`; operator-only pages are filtered from tab records; node proxy cannot mutate persistent profiles (`extensions/browser/src/gateway/browser-request.ts:605-607,211-276,439`).
- CDP: relaxed policy only for local managed loopback control plane (`extensions/browser/src/browser/cdp-reachability-policy.ts:68-83`); remote hosts require allowlist/private-network checks (`:44-65`); remote providers (Browserless/Browserbase/Notte) documented at `docs/tools/browser.md:74-78`. Attached-browser tools accept only `http://127.0.0.1:<port>/` with no userinfo/path/query and clear relay ports/tokens (`extensions/browser/src/attached-browser-tool-runtime.ts:31-92`). Navigation SSRF checks pre/post-load (`pw-session-actions.ts:511-525`, `pw-tools-core.snapshot.ts:243-262`).

### 4.5 Nodes / computer use

- Node registration: `src/gateway/node-registry.ts:421` `registerSession()` requires `pairingIdentity`, derives nodeId from `device.id ?? client.id` (line 430), and intersects approved permissions (lines 450-457). Pairing changes invalidate grants (lines 647, 668); invoke-time rechecks at `src/gateway/node-invoke-plugin-policy.ts:267-287` and `node-registry.invoke-stream.ts:201`.
- Capability declarations: `src/node-host/runtime-manifest.ts:47-104` (ephemeral hosts strip `screen.snapshot` / `computer.act` at lines 75/93); macOS `advertisedPermissions` at `apps/macos/Sources/OpenClaw/NodeMode/MacNodeModeCoordinator.swift:1187`; Android snapshot at `apps/android/.../AndroidPermissionSnapshot.kt:29-49`. Invoke requires declared **and** allowlisted commands (`src/gateway/node-command-policy.ts:460-477`).
- Auto-approval is deliberately narrow: `prepareGatewayNodeConnect()` in `src/gateway/server/ws-connection/connect-node-session.ts:27` auto-approves only the *initial* declared caps/commands/permissions for silent/SSH-verified/bootstrap lanes; manifest upgrades still prompt (lines 127-180, comment at 142), then intersects declarations (lines 188-221).
- **UNVERIFIED:** a paired node can *advertise* OS permission booleans; the server intersects them with approval but cannot independently attest the actual OS grant. Spoofability of those booleans was not tested.

---

## 5. Claim-by-claim verification of the existing research doc

| # | Claim from `<repo>/docs/RESEARCH_HERMES_OPENCLAW.md` | Label | Source evidence / correction |
|---|---|---|---|
| 1 | Gateway defaults to `127.0.0.1:18789`, typed WS API, JSON-Schema-validated frames | **VERIFIED (nuance)** | `src/config/paths.ts:285`, `zod-schema.gateway.ts:76,87-97`; validators are generated TypeBox checks (`packages/gateway-protocol/src/protocol-validator.ts:6-20`), not runtime Ajv — the `errors` shape is Ajv-compatible. |
| 2 | Nonce-signed device identity; optional pairing | **VERIFIED** | `src/gateway/server/connection.ts:325-334` challenge; `connect-auth.ts:378-414` id/skew/nonce checks; `handshake-auth-helpers.ts:205-241` v2/v3 payload; Ed25519 at `src/infra/device-identity.ts:70-89`. Nonce is per-connection; skew 2 min. |
| 3 | One long-lived gateway owns channels/sessions/tools/auth/events; foreground under launchd/systemd | **VERIFIED (nuance)** | Single gateway architecture: `src/gateway/server.ts`, `server-start.ts`. Foreground start with supervisor detection: `src/cli/gateway-cli/run.ts:291,384,885` (systemd guard, foreground TTY, consent-gated repair). Service-manager *installation* files were not exhaustively enumerated; launchd/systemd orchestrate restarts and durable jobs (`docs/gateway/background-process.md:22,149,166`), they are not the gateway runtime itself. |
| 4 | Security posture "trusted gateway, untrusted execution, policy enforced by code" | **VERIFIED as wording; PARTIAL as guarantee** | `README.md:20`, `docs/index.md:105`, `docs/start/why-openclaw.md:2`. But in-process plugins/hooks run *inside* the trusted gateway, so the boundary is not uniform. |
| 5 | Exec approval ladder `deny`/`allowlist`/`ask`/`auto`/`full` | **VERIFIED** | `src/infra/exec-approvals-core.ts:18`, `src/config/zod-schema.agent-runtime.ts:223`; these are shorthands over `security` + `ask` axes (`src/infra/exec-approvals-config.ts:21-27`). |
| 6 | Durable jobs: 16/session, 64/gateway; systemd/launchd persistence | **VERIFIED** | `docs/gateway/background-process.md:34`; `src/agents/bash-process-durable.ts:305-345`; admission `src/agents/bash-tools.exec-run.ts:280`. Durability excludes sandbox/node/worker/one-shot. |
| 7 | Sandbox opt-in, Docker default, gateway stays on host, not a perfect boundary | **VERIFIED (nuance)** | `src/agents/sandbox/config.ts:212-264`, `docker-backend.ts:164-220`, `validate-sandbox-security.ts:22-52,93-110,313-380`. Break-glass flags and excluded execution paths confirm the caution. |
| 8 | Browser: separate managed profile, loopback control service, Playwright/CDP, remote CDP providers | **VERIFIED** | `extensions/browser/src/server.ts:62-109`, `control-auth.ts`, `attached-browser-tool-runtime.ts:31-92`, `docs/tools/browser.md:74-78`. |
| 9 | Plugin manifest `openclaw.plugin.json` requires `id` + inline `configSchema` (JSON5); `register(api)`; in-process/unsandboxed | **VERIFIED** | `src/plugins/manifest.ts:46,181-236`; `plugin-module-loader-cache.ts:13`, `native-module-require.ts:294-322`; `api-builder.ts:117-152`; `api-lifecycle.ts:62-80`. |
| 10 | Session SQLite at `~/.openclaw/agents/<agentId>/agent/openclaw-agent.sqlite`; DM scope `main` default | **VERIFIED** | `src/config/agent-store-source.ts:84-110` (line 92 filename); `src/routing/session-key.ts:179-237` (line 193 default `main`). |
| 11 | Memory: Markdown workspace + SQLite/embedding search; optional PostgreSQL | **VERIFIED** | `src/agents/workspace-bootstrap-policy.ts:30-37`; `extensions/memory-core/index.ts:316-347`; `sqlite-memory-search-fragments.ts:21,65`; `manager-search.worker.ts:137`. |
| 12 | Context engines: ingest/assemble/compact/after-turn, `ownsCompaction`, default `legacy`, quarantine fallback | **VERIFIED** | `src/context-engine/types.ts:182-188,347-538`; `registry-selection.ts:13-46`; `registry-quarantine.ts:30-60`. |
| 13 | `openclaw automations` (`openclaw cron` alias) | **REFUTED (alias direction reversed)** | `src/cli/cron-cli/register.ts:30-31` is `.command("cron").alias("automations")`; also `src/cli/program/subcli-command-metadata.ts:54-55`. |
| 14 | Hooks: sequential, trusted, non-sandboxed, no durable queue/exactly-once | **VERIFIED** | `src/hooks/loader.ts:121-133`; `src/plugins/hooks.ts:527-640`; `docs/automation/hooks.md:139`; `docs/automation/cron-jobs/how-it-works.md:58`. |
| 15 | Telegram grammY, WhatsApp Baileys, credentials at `~/.openclaw/credentials/whatsapp/...` | **VERIFIED** | `extensions/telegram/package.json:8-10`; `extensions/whatsapp/package.json:13`; `docs/channels/whatsapp.md:561`. |
| 16 | Discord "official Discord gateway bot" | **PARTIAL** | Uses `@discordjs/voice`, `discord-api-types`, `ws` (`extensions/discord/package.json:10-14`) and a manual gateway monitor; not the `discord.js` client library. Protocol-level statement stands. |
| 17 | Subagents use `agent:<id>:subagent:<uuid>`, `sessions_spawn`/`yield`/`subagents`, `context:"fork"` | **VERIFIED** | `docs/tools/subagents.md:13,30-31` cross-checked with `src/agents/subagents/spawn/subagent-spawn.types.ts:11`, `src/agents/tools/sessions-spawn-tool.ts:96-113`, `src/infra/state-migrations.session-store.ts:175-177`. |
| 18 | Nodes require pairing approval, advertise `computer.act`/screen/camera/location | **VERIFIED (attestation caveat)** | `src/gateway/node-registry.ts:421-457`, `src/node-host/runtime-manifest.ts:47-104`, `src/gateway/node-pairing-auto-approve.ts:50-67`; OS permission truth cannot be independently verified by the gateway (**UNVERIFIED** in practice). |
| 19 | "Non-loopback binds require auth" | **PARTIAL** | `docs/gateway/remote.md:400` states the rule; source shows it as a **critical audit finding** `gateway.bind_no_auth` (`src/security/audit-gateway-config.ts:117-125`), plus audit checks for missing loopback auth, wildcard allowed origins, host-header origin fallback, and trusted-proxy misconfiguration (lines 154-373). I did not locate a hard startup refusal for every non-loopback/no-auth combination, so treat the audit as detection rather than proof of enforcement. |

**Count across 19 claims:** 16 fully or substantially verified (several with documented nuances: TypeBox vs Ajv, conditional group-policy default, gateway-wide trust boundary), 2 partial (Discord SDK identity; non-loopback auth as audit vs hard enforcement), 1 refuted (alias direction of `cron`/`automations`).

---

## 6. Design weaknesses, sharp edges, and things LEO should NOT copy

Ranked roughly by security impact for a system like LEO:

1. **In-process plugin and hook code is the gateway.** `src/plugins/native-module-require.ts:294-322`, `src/hooks/loader.ts:121-133`. A "trusted local code" plugin has full Node process authority: filesystem, env, network, gateway scope, ability to register admin-scoped methods. Do **not** reproduce this in LEO: keep plugins out-of-process (subprocess with a narrow JSON-RPC/MCP-style channel, explicit capability tokens, resource limits) or, if in-process is unavoidable, treat load/enable as equivalent to installing root-level software and require an explicit, recorded operator action. Docs' "trusted gateway, untrusted execution" slogan does **not** cover plugin code.
2. **Shared secret is effectively root.** `src/gateway/method-scopes.ts:48-56`, `src/gateway/http-auth-utils.ts:697-718`. A token/password holder gets `operator.admin` by design. Fine for a single-operator laptop; dangerous if LEO ever exposes the API. LEO's `<repo>/leo/interfaces/api.py:34` currently has **no auth at all** and relies on the `127.0.0.1` bind from `<repo>/leo/config.py:73`. Copy the *scope model*, not the "bearer = admin" shortcut: default to least privilege and require explicit scope grants for admin operations.
3. **Adapter-local authorization with a trusting dispatcher.** `src/channels/turn/run-channel-turn.ts:188-204` trusts `admission` from the adapter. One adapter bug becomes an authorization bypass. LEO's `Gateway.receive()` (`<repo>/leo/integrations/gateway.py:110+`) should enforce sender/policy checks centrally before persisting or emitting `gateway.message`, with adapters supplying only identity facts.
4. **Default-DM "main" session collapses distinct peers.** `src/routing/session-key.ts:193`. If LEO's channel adapters map different senders onto the same `channel + subject` (`gateway.py:_conversation`), invest early in a per-peer scope with explicit identity links; otherwise one untrusted DM can read another person's context.
5. **Group-policy resolution has multiple fallbacks (open/allowlist/disabled) across schema, config, provider and account layers.** `src/config/runtime-group-policy.ts:20-27,54-62` vs `src/config/zod-schema.channel-messaging-common.ts:71`. Ambiguous defaults are a security bug factory; LEO should have exactly one `resolve_policy()` returning an explicit, fail-closed decision with the reason attached.
6. **Exec approval text is not a semantic binding.** `docs/tools/exec.md:170`, `tools.exec.strictInlineEval` default false (`docs/tools/exec.md:260`). A hidden wrapper/script can change what actually executes. LEO's approvals currently match action + resource strings (`<repo>/leo/security/permissions.py:50-70,100-110`) and have no TTL or argument hash — adopt OpenClaw's cwd-bound hashes, env/file operand snapshots and live-policy recheck on timeout, but bind to the *resolved program*, not just the typed string.
7. **"Durable" jobs have a long list of exclusions.** `src/agents/bash-process-durable.ts:305-345` (no sandbox/node/worker/one-shot, host-OS requirements). LEO's scheduler is explicitly single-process (`<repo>/LEO_KNOWN_ISSUES.md`: "no distributed lock"), so don't market job durability beyond what the restarter actually guarantees; use idempotency keys and receipts.
8. **Hooks/cron are at-least-once at best.** No durable hook queue (`docs/automation/hooks.md:139`); cron "not exactly once" (`docs/automation/cron-jobs/how-it-works.md:58`). LEO's webhook dedupe (`<repo>/leo/tasks/webhooks.py:113-160`) is a good start; extend the same delivery-ID idempotency to scheduler-generated agent turns.
9. **Browser CSRF guard allows requests with no Origin/Referer.** `extensions/browser/src/browser/server-middleware.ts:35`. Auth still required, but an originless browser bug is one layer less defense. LEO's HTTP tool DNS rebinding caveat is already documented in `LEO_KNOWN_ISSUES.md`; combine origin checks with resolved-IP pinning after DNS.
10. **Break-glass config knobs are numerous and explicit.** `dangerouslyAllowContainerNamespaceJoin`, `browser.allowHostControl`, `dangerouslyAllowHostHeaderOriginFallback`, `plugins.allow` phantom entries. These are honest, auditable choices, but a large knob surface invites "just set the flag" security erosion. LEO should gate each break-glass action through its existing HIGH-risk approval workflow and audit log (`permissions.py:_audit`) rather than config-only toggles.
11. **Nonce/signature checks are good but not a substitute for authorization.** Device identity proves *which key*, not *which human*. Pairing/scope policy (`connect-pairing-approval-plan.ts`, `node-pairing-auto-approve.ts`) carries the authority. LEO should keep device identity, principal authorization, and per-tool policy as separate layers.
12. **Scale/complexity is itself a risk.** The pinned checkout has 65,749 files; session concerns are split across hundreds of `session-*.ts` modules, with worker placement, quarantine, cold storage and publication layers. LEO should copy the *interfaces and invariants*, not the module count.

**Explicit do-not-copy list:** in-process plugin execution; token-as-root operator scopes; adapter-local-only authorization; default-shared DM sessions; "not a perfect boundary" sandbox language used as a substitute for enforcing one; hook handlers with arbitrary gateway-process authority; marketing "durable" without host/OS preconditions.

---

## 7. Ranked improvements LEO should adopt

Effort: S ≈ half a day, M ≈ 1-3 days, L ≈ 1-2 weeks for one engineer. Verification uses only LEO's existing test suite style (pytest + injected fakes); nothing below requires live credentials.

1. **[P0, M] Central admission gate for all inbound channels.** Move sender/allowlist/pairing/mention decisions into a single `Gateway.authorize_inbound()` choke point that runs *before* `_conversation()` and event emission in `<repo>/leo/integrations/gateway.py`; adapters supply verified identity facts only. Add explicit `dm_scope` (`per-peer` default for external channels) in `<repo>/leo/config.py`. **Verify:** extend `tests/test_api.py` / a new gateway test with two different peer IDs and assert separate conversations plus deny-by-default for unknown senders; add a test that a malicious adapter cannot set `admission=dispatch` without passing the gate.
2. **[P0, M] Authenticated local control plane with scopes, and a real WebSocket contract.** Add a generated schema + `connect.challenge` nonce + Ed25519 device identity (or short-lived loopback tickets for the desktop shell) to the FastAPI/WS surface in `<repo>/leo/interfaces/api.py`; define `operator.read/write/admin`-style scopes with `x-leo-scopes`-equivalent narrowing only. **Verify:** Python tests for stale nonce, wrong key, scope-narrowing-only, origin rejection, and replay on a second connection; mirror OpenClaw's post-await recheck pattern.
3. **[P0, M] Semantic approval binding + TTL + revocation.** Extend `<repo>/leo/security/permissions.py` so HIGH-risk approvals bind to a hash of (resolved executable, argv, cwd, relevant env names, file operands), carry an expiry, and are invalidated by policy changes. Mirror the live-policy recheck on timeout. **Verify:** tests that an approved `python tool.py` cannot be reused as `python other.py`, that a changed cwd/env breaks reuse, that an expired or revoked ID fails, and that audit rows are still written.
4. **[P1, M] Out-of-process plugin boundary.** If/when LEO adds real plugins, run each in a subprocess with a narrow RPC surface and capability tokens; never `import` third-party plugin code into the API process. Provide a manifest analogous to `openclaw.plugin.json` with `id`, `config_schema`, declared tools/hooks, and an explicit `trust` tier. Add a `leo security audit` check for phantom allowlist entries and unpinned plugins (copy the *idea* from `src/security/audit-plugins-trust.ts:211-290`) plus host bind/auth, wildcard origin, and dangerous-flag checks (model after `src/security/audit-gateway-config.ts:117-125,154-198,274-286`). Prefer hard startup refusal over audit-only warnings for non-loopback-without-auth. **Verify:** a plugin that touches `os.environ` or opens a socket must fail; a manifest with undeclared capabilities must be rejected.
5. **[P1, M] Approval-aware exec policy ladder with an explicit mode.** Replace ad-hoc risk tiers with a documented `exec.mode ∈ {deny, allowlist, ask, auto, full}` mapped to LEO's `security`/`ask` axes; make the reviewer (if any) deny-on-unclear and escalate after repeated denials. Default to `ask` for external-channel-origin turns, regardless of local config. **Verify:** table-driven tests over the mode matrix, including inline interpreter forms (`python -c`, `node -e`) and chained shell commands.
6. **[P1, L] Container-backed tool isolation when Docker is available, with honest capability reporting.** Add a sandbox backend adapter behind a `sandbox.mode` config (`off` by default), validate bind mounts/seccomp/network like `src/agents/sandbox/validate-sandbox-security.ts`, and keep the API/gateway outside the container. Update `LEO_KNOWN_ISSUES.md` to state exactly which tool paths remain host-side. **Verify:** tests assert host credential dirs and Docker socket cannot be mounted, `network=host` is rejected, node/worker paths are reported as unsandboxed; skip gracefully when Docker is absent (current environment).
7. **[P1, S] Single fail-closed policy resolver + policy provenance.** Implement one function returning `{decision, reason, source}` for channel/exec/tool policy, eliminating layered fallbacks. Log the resolved policy with each agent turn. **Verify:** property-style tests over combinations of missing provider config, account override, and global default; every case returns an explicit policy, never "implicit open".
8. **[P2, M] Durable scheduler receipts with idempotency keys.** Reuse the webhook delivery-ID dedupe from `<repo>/leo/tasks/webhooks.py` for cron/event-triggered tasks in `<repo>/leo/tasks/scheduler.py:217-263`; add a `receipts` table keyed by `(schedule_id, occurrence_ts)` and document at-least-once semantics. **Verify:** simulate a crash between "task created" and "next_run_at persisted"; restart and assert exactly one task for that occurrence; keep the existing single-scheduler-per-DB warning.
9. **[P2, S] Context engine and memory flush contracts.** Define a small `ContextEngine` protocol (`ingest`, `assemble`, `compact`, `after_turn`, `owns_compaction`) and a single active slot with quarantine/fallback to `legacy`; keep Markdown durable memory plus SQLite FTS search already in `<repo>/leo/memory/store.py`, and add an append-only pre-compaction flush prompt that marks bootstrap files read-only (idea from `extensions/memory-core/src/flush-plan.ts:19-36`). **Verify:** a broken engine is disabled after N failures and falls back; compaction flush produces only `memory/YYYY-MM-DD.md` and never edits `MEMORY.md`.
10. **[P2, S] Browser/node origin and capability hygiene.** For LEO's browser adapter (`<repo>/leo/integrations/browser.py`), require explicit Origin/Referer on mutating requests (unlike OpenClaw's originless pass-through), pin resolved IPs after DNS checks, and model node permissions as declared ∩ approved ∩ runtime-checked (as in `src/gateway/node-registry.ts:421-457`). **Verify:** tests for missing Origin, cross-site Origin, DNS-rebinding double-resolution, and a node requesting an undeclared command.

**Sequencing:** items 1-3 first (they close real privilege-boundary holes in LEO as it stands), then 4-7, then 8-10. No item requires credentials or paid services to verify.

---

## 8. Limits and explicitly unverified

- No runtime execution of OpenClaw; all findings are static at commit `75d5bf7e3b4a1825db30b2352803b765e1681aa2`. No sandbox escape, browser exploit, replay attack, or concurrency stress test was performed.
- Git history/blame was not inspected; "intentional" comments are the authors' own words, not independently established design rationale.
- The checkout is enormous; areas not enumerated in sections 1-7 (e.g. A2A, canvas, meeting-bot, TUI internals, every one of the ~hundreds of model/channel extensions) were not audited. Extension-level claims are limited to the named plugins.
- Discord Interactions HTTP endpoint verification was not located; if it exists outside `extensions/discord`, its auth was not checked.
- OS permission attestation for paired nodes, cron exactly-once behavior under real crashes, and PostgreSQL memory under concurrent writers were not tested.
- LEO-side observations are from static reading of the local tree only; no LEO code was modified and no LEO tests were run as part of this report.

---

## 9. One-paragraph bottom line for LEO

OpenClaw's strongest transferable ideas are **not** its feature breadth but its invariants: a single typed control plane with nonce-bound device identity and post-await authority rechecks; an explicit exec policy ladder with argv/cwd/env-bound approvals and live revocation; a central channel-ingress policy engine; and quarantine/fallback for replaceable context engines. Its clearest negative lessons are equally valuable: in-process plugins and hooks make "trusted gateway" mean "trust every extension with the whole process"; a shared secret doubles as root; per-adapter authorization behind a trusting dispatcher is fragile; and default DM/session and group-policy fallbacks can silently broaden who shares an agent's context. LEO already has better starting instincts in its risk-tier permissions, HMAC webhook dedupe, and narrow node API — the highest-value next work is to make those boundaries *structural* (central admission, scoped control-plane auth, bound approvals) rather than cooperative.
