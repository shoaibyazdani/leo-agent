# Deep research round — synthesis and remediation log (2026-11-18)

**Commissioned goal:** deep, multi-agent research of the Hermes Agent and
OpenClaw markets/repositories, real end-to-end runtime tests, then fixes to LEO.

**Window:** started `2026-11-17T21:06:03Z`; held open for at least 3600 s
(enforced by a background timer writing `END.txt`). Ten independent agents ran
in parallel; all have reported. Upstream repos were cloned into isolated `/tmp`
directories and treated as untrusted data. No upstream source was copied into
LEO, no upstream project was modified, no credentials or paid services were
used, and no public network service was exposed.

## 1. Report index

| Report | What it establishes |
|---|---|
| [hermes-source.md](hermes-source.md) | Source audit at `b076b585…`: loop, prompt tiers, toolsets, gateway, memory/context ABCs, plugins; 30 doc claims checked; known bypasses |
| [hermes-e2e.md](hermes-e2e.md) | Real install failure (Pillow pin conflict), working offline chat + tool round trip against loopback mock, doctor/skills/config inconsistencies, 165 upstream tests passed |
| [openclaw-source.md](openclaw-source.md) | Source audit at `75d5bf7e…`: nonce-signed WebSocket auth, in-process plugin risk, adapter-local admission, exec approval bindings; 19 claims checked |
| [openclaw-e2e.md](openclaw-e2e.md) | npm install, token-authed loopback gateway, full offline `openclaw agent` reply from local mock; build/tooling mismatches |
| [market-landscape.md](market-landscape.md) | 2026 market/standards scan (MCP, AgentSkills, A2A, ACP); verifiable completion, programmable orchestration, interop and enforced security as the frontier |
| [leo-code-audit.md](leo-code-audit.md) | 26 findings with reproductions; 3 P0 security defects and multiple task/state invariants |
| [leo-e2e.md](leo-e2e.md) | Independent LEO CLI/API/browser/Electron verification; fresh-home, persistence, live Chromium; API auth gap |
| [security-licensing.md](security-licensing.md) | MIT license/attribution verification, agent threat model, prioritised hardening backlog |
| [skill-compatibility.md](skill-compatibility.md) | 327 real upstream SKILL.md files tested against LEO's loader; 0 executable; exact root causes |
| [intelligence-memory.md](intelligence-memory.md) | Prompt caching, compaction, routing, memory and orchestration audit with 12 measurable upgrades |

## 2. What the upstream projects actually do

- **Hermes** (MIT, Nous Research): Python `AIAgent`; stable/context/volatile
  prompt tiers; toolsets with approval heuristics; `delegate_task` and versioned
  subagent lifecycle handles; pluggable memory/context engines; FTS5 session
  search; 26 bundled messaging adapters; cron prompts delivered to channels;
  browser CDP supervisor; terminal backends (local/docker/ssh/cloud). Its own
  source documents a scanner bypass, parallel dispatch pipelines and fail-open
  plugin pre-hooks — these are warnings, not a blueprint to copy.
- **OpenClaw** (MIT, OpenClaw Foundation): one host Gateway with a typed
  WebSocket control plane, per-connection nonce and signed device identity;
  channels, sessions, tools and nodes behind it; durable exec jobs with
  cwd-bound argument hashes; managed browser profile; plugin/hook extensions
  that run in-process and are explicitly *not* a security boundary. Its CLI and
  build tooling showed real inconsistencies at the tested version.
- Both upstream projects can complete a full offline chat/tool round trip with
  a local mock provider. That process-level gate is now a LEO roadmap item.

## 3. Verified defects fixed in LEO this round

| Defect | Fix | Regression test |
|---|---|---|
| Any truthy `approval_id` cleared the HIGH-risk gate | `security/permissions.py`: approve only when row matches actor/action/resource/risk, is unexpired and no path violation; executor relies on the permission decision | `tests/test_security_regressions.py` |
| Webhooks accepted unsigned payloads when secrets were unset | `integrations/messaging.py`: all three adapters fail closed | `tests/test_security_regressions.py` |
| HMAC verified over re-serialized JSON | API reads exact raw bytes (1 MB cap), passes `raw_body` through gateway to adapter | same file |
| `python`/`python3`, `find -delete`, `sed -i`, `git push`, arbitrary paths allowed by terminal tool | Allowlist narrowed; per-command dangerous flags and path arguments checked | same file |
| API unauthenticated and non-loopback binds accepted | Optional `LEO_API_TOKEN` bearer auth on interactive routes (webhooks keep HMAC); CLI refuses public bind without token; dashboard/Electron send the token | same file |
| Approval resolver chosen by caller | Server records authenticated principal | same file |
| `transition(..., COMPLETED)` wrote empty evidence | Direct completion rejected; agent delegation now uses evidence-backed `complete()`/`fail()` on both real and fallback task engines | `tests/test_tasks.py` |
| Cancelled tasks could still complete; terminal `run()` raised instead of no-op | Cancellation guards and idempotent terminal run | `tests/test_tasks.py` |
| Verification accepted contradictory evidence | `evidence_supports_output` is mandatory when claims exist | existing suite |
| Memories injected twice; compaction unreachable; `memories_used` reported searched IDs | Single injection through `ContextOptimizer`; provider passed through; selected IDs reported | existing suite |
| Imported skills: documentation shell snippets caused 44 false rejections; Hermes gates became permissions; OpenClaw bins became self-dependencies; numeric names rejected; versions fabricated; nested metadata unscanned | `skills/compatibility.py`, `skills/registry.py`, `contracts.py`: structured `gates`/provenance, advisory body scan, explicit version sentinel, reversible names, metadata scan | `tests/test_compat_skill_md.py` |
| Dashboard approvals list always empty; no resolve action; Electron ignored custom API port | UI contract fixed, approve/deny buttons with escaped output, session-scoped token, desktop port/auth handling | manual verification |

Additional fixes after the first synthesis draft: agent capability tags are no
longer sent to providers as literal model IDs; a process-level fresh-home CLI
E2E gate was added; self-dependencies declared by real Hermes skills are
filtered during import; webhook replays with changed delivery IDs are deduped
on the signed payload with a timestamp window; and `extra_header_env` secrets
are scrubbed from warnings and durable usage metadata. Re-running the corpus harness against the
patched loader: **Hermes 186/186 and OpenClaw 140/140 translated skills pass
LEO's own registry validation** (the single remaining upstream `SKILL.md` has no
frontmatter and is quarantined by design). Translation is still not execution.

**Result:** full suite `145 passed, 1 skipped` (the opt-in live Chromium test),
up from `125 passed, 1 skipped` before the round.

## 4. Remaining work, in priority order

1. Atomic task claims (worker double-execution, idempotency races).
2. Sign `timestamp || delivery_id || body` in `WebhookHandler` (legacy body-only
   signatures remain accepted for compatibility) and add per-route rate limits.
3. Enforce `AgentSpec.tool_permissions` centrally; budget context/tools per agent.
4. Network-layer SSRF controls (DNS rebinding) and secret redaction in any new
   error paths.
5. Add prompt-cache accounting (cache read/write tokens) and a labelled eval
   harness (routing, retrieval, verification, cost) before efficiency claims.
6. Extend the new process-level fresh-home CLI gate
   (`tests/test_process_e2e.py`) with streaming, tool-call and session
   assertions; the basic CLI → mock provider → persistence path now runs in
   separate processes with an isolated HOME and no host credentials.
7. Skill bundles: inventory `scripts/`/`references/`/`assets/`, resolve path
   tokens, evaluate gates; execute third-party scripts only in an OS sandbox
   with explicit approval. Until then, imported skills are inert by design.
8. Standards wedge: MCP client, AgentSkills digest-bound approvals, A2A/ACP
   adapters — only after 1–6 are solid.

## 5. Honest status

Implemented and tested: the fixes in §3 plus the pre-existing local milestone.
Blocked: live model gateway, real messaging channels, remote VPS (no
credentials/infrastructure). Partially implemented: Hermes/OpenClaw skill
*metadata* import, browser/desktop abstractions, context optimisation without
prompt-cache measurement. Planned: §4. No efficiency or compatibility claim in
this report should be read as measured production quality; see the individual
reports for commands, outputs and limitations.

## 6. Final independent verification review

A separate reviewer re-tested the patched tree and found four residual issues,
all now fixed with regression tests:

1. **High** — `cli serve` inspected only the `--host` flag; a config file with
   `server.host: 0.0.0.0` still exposed the unauthenticated API. The guard now
   uses the effective host (`--host` or config) before starting uvicorn.
2. **High** — terminal path confinement could be bypassed with flag operands
   such as `grep --file=/outside/secret` and `git --git-dir=/outside/.git`;
   `sed -i.bak` evaded the exact-match deny list. Flag values are now validated
   as paths and dangerous flags use prefix matching.
3. **Medium** — the standalone `_DurableTasks` fallback completed work without
   evidence and replayed terminal tasks. It now enforces evidence, honours
   cancellation and refuses terminal replay, matching the durable engine.
4. **Low** — Electron injected the bearer token for origins merely starting
   with the API URL string; it now compares parsed URL origins.

The same reviewer confirmed the other fixes (approval binding, webhook
fail-closed/raw bytes/replay/timestamps, terminal interpreters, API token,
dashboard escaping, skill gates/provenance, compaction bridge, secret
scrubbing) with 39 focused checks. Research window: `21:06:03Z → 22:06:13Z`
(60 minutes 10 seconds). Final suite: `145 passed, 1 skipped`; live Chromium
test passes when enabled.
