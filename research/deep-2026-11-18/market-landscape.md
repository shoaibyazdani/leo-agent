# Autonomous Agent Platforms in 2026 — Market, Standards, and LEO Differentiation

**Research date / access date for all links below:** 2026-11-18. **Scope:** official documentation, official specifications, official GitHub repositories and official project blogs only. No credentials, paid services, installations, or runtime tests of competitor products were used. Nothing in `<repo>` was modified except this report.

## 0. Method and evidence labels

- `VERIFIED` — statement read directly from an official spec/docs/repo page listed in §7 on 2026-11-18.
- `PARTIAL` — official page was found but truncated, JS-rendered, redirected to a page that did not answer the question, or a claim is corroborated in only one place.
- `UNVERIFIED` — not established by an official source; includes all performance, adoption, security-effectiveness and "works better" claims.
- `WebSearch` produced no usable results in this environment (it returned a model-generated "no live web access" answer with no sources). All findings therefore come from `WebFetch` against official pages and the project's own prior research (`docs/RESEARCH_HERMES_OPENCLAW.md`, same access date). Search-engine corroboration, independent benchmarks of Hermes/OpenClaw, real-world reliability data and third-party license/ToS reviews are **UNVERIFIED**.
- Repository star counts and issue counts are point-in-time and will drift.

**Bottom line (VERIFIED from docs):** the 2026 frontier is no longer "which agent has more tools". It is (a) *verifiable completion* with evidence and independent checking, (b) *programmable orchestration* that keeps intermediate state out of the model context, (c) *standard interop* (MCP 2026-07-28, AgentSkills, A2A 1.0, ACP 1), and (d) *runtime-enforced security and cost control*. LEO already has durable tasks, memory, delegation, permissions and a desktop shell, but has no MCP client, no workflow-script runtime, no prompt-cache instrumentation, no OTel tracing, and only cooperative (not OS-level) isolation (`<repo>/docs/CAPABILITY_INVENTORY.md`, `<repo>/LEO_KNOWN_ISSUES.md`). That is the gap to attack.

---

## 1. Market map (capabilities, not hype)

### 1.1 Hermes Agent and OpenClaw (baseline)

Sources: [prior official-source research](https://github.com/NousResearch/hermes-agent) / [in-repo research](file://<repo>/docs/RESEARCH_HERMES_OPENCLAW.md); [Hermes README](https://raw.githubusercontent.com/NousResearch/hermes-agent/main/README.md); [OpenClaw README](https://raw.githubusercontent.com/openclaw/openclaw/main/README.md); [OpenClaw architecture](https://github.com/openclaw/openclaw/blob/main/docs/concepts/architecture.md) (all accessed 2026-11-18).

| Dimension | Hermes Agent | OpenClaw |
|---|---|---|
| Shape | Python `AIAgent`, persistent gateway, cron, SQLite/FTS5 | TypeScript single-host Gateway with typed WebSocket control plane; channels/nodes/clients |
| Skills | `SKILL.md`, `metadata.hermes` gates, bundled/optional/user tiers; README claims AgentSkills compatibility | `SKILL.md` + `metadata.openclaw.requires.*`, 7 precedence tiers, ClawHub install/verify |
| Memory | Markdown `MEMORY.md`/`USER.md`, single-active `MemoryProvider`, FTS5 session search | Markdown memory, SQLite/embeddings, PostgreSQL option, replaceable memory plugins |
| Delegation | `delegate_task`, versioned lifecycle handles | `sessions_spawn`/`sessions_yield`, cheaper child model, announce chain |
| Execution | Toolsets, approval modes, 6 terminal backends (local/Docker/SSH/Modal/Daytona/Vercel) | Exec approval ladder, opt-in sandbox, durable background jobs |
| Browser | CDP supervisor with dialog/OOPIF handling | Managed browser profile, Playwright/CDP, remote CDP vendors |
| Automation | Cron agent prompts delivered to channels; HMAC webhooks | Automations scheduler, hooks, HTTP webhooks (no durable exactly-once queue) |
| Security | Default-deny pairing, hardline blocklist, approvals | "Trusted gateway, untrusted execution, policy enforced by code"; in-process plugins unsandboxed |

`VERIFIED` for documented capabilities. **Marketing claim to flag:** Hermes README says "the only agent with a built-in learning loop" and claims agentskills.io compatibility, while the detailed skill-authoring page does not restate the compatibility claim (`PARTIAL`). Neither project has published reliability/latency numbers that this research could verify (`UNVERIFIED`). The Hermes "agent importer" is a one-way migration utility, **not** an OpenClaw compatibility runtime (`VERIFIED` in prior research).

### 1.2 Claude Code (the strongest 2026 reference point for orchestration and UX)

Sources: [overview](https://code.claude.com/docs/en/overview), [dynamic workflows](https://code.claude.com/docs/en/workflows.md), [agent teams](https://code.claude.com/docs/en/agent-teams.md), [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) (accessed 2026-11-18).

`VERIFIED` headline capabilities:
- **Surfaces:** CLI, IDE extensions, desktop app with side-by-side sessions and scheduled tasks, web/cloud, mobile, CI, Slack, channels, Remote Control and `claude --teleport` session handoff.
- **Dynamic workflows:** a JavaScript script written by the model and executed in a background runtime; `agent()`, `pipeline()`, `parallel()`, `phase()`; intermediate results stay in script variables, not the model's context; adversarial verification and multi-angle planning are explicit supported patterns. Bundled `/deep-research` fans out, cross-checks, votes on claims, emits a cited report and labels claims it could not verify as *unverified*. Limits: 16 concurrent agents default (configurable to 256), 4,096 items per `parallel()`/`pipeline()`, 1,000 agents per run, automatic stall restarts up to 5, deterministic replay of completed agents, per-agent token view, prompt-cache prefix staggering (`VERIFIED`).
- **Agent teams:** experimental, disabled by default; lead + independent teammate sessions + shared task list with file-locked claiming + mailbox messaging. Explicitly documented downsides: high token cost, no in-process session resume, no nested teams, task status lag, permission prompts bubble to the lead (`VERIFIED`).
- **Extension surface:** MCP, skills with progressive disclosure, hooks (including quality gates such as `TaskCompleted`), plugins, subagents, Agent SDK, auto memory, `CLAUDE.md`/`AGENTS.md`, managed settings/model allowlists, sandboxing and permission modes.
- **Efficiency:** prompt caching with `tools → system → messages` prefix order, 5m/1h TTL, documented read/write multipliers (0.1× read, 1.25×/2× writes on standard models), `cache_read_input_tokens` metrics, pre-warming via `max_tokens: 0`, cache-invalidation warnings for dynamic tool definitions and JSON key ordering (`VERIFIED`).

### 1.3 OpenAI Codex

Sources: [Codex docs index](https://learn.chatgpt.com/docs), [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) (for the general efficiency comparison), [Terminal-Bench](https://www.tbench.ai/) (accessed 2026-11-18).

`VERIFIED` from the official docs index: CLI, IDE extension, Cloud, desktop/mobile/web apps, Codex SDK, App Server, GitHub Action; local and cloud environments, git worktrees, sandboxing and auto-review, approval profiles; skills, MCP, plugins, hooks, "Site tools (WebMCP)"; scheduled tasks, long-running work, notifications, subagents; browser, computer use, record & replay, visualization, voice, web/image search. The index also lists 2026-dated "what's new" entries (e.g., 2026-11-17 GPT-6.2) (`PARTIAL`: index-level only, release behavior not inspected). Several deep links returned 404 during this research, so details of auto-review, long-running work and WebMCP are **UNVERIFIED**.

### 1.4 Cursor and agent IDEs

Source: [Cursor agent overview](https://cursor.com/docs/agent/overview) (accessed 2026-11-18).
`VERIFIED`: unlimited tool calls, file/shell/browser/image tools, checkpoints that restore files without deleting chat history, queued messages and mid-task steering, side chats, search, `/goal` for long-lived objectives (rolling out; `PARTIAL`), custom modes and `/loop` skill; IDE, CLI, web agents and Agents Window. Cursor is listed as an MCP client in [MCP's intro](https://modelcontextprotocol.io/docs/getting-started/intro) (`VERIFIED`). Background/cloud agent docs URL was 404 → breadth of cloud parallelization **UNVERIFIED**.

### 1.5 Frameworks and "agent OS" contenders

| Project | Verified status/capability (official source) | Implication for LEO |
|---|---|---|
| **LangGraph** | [Overview](https://docs.langchain.com/oss/python/langgraph/overview): "low-level orchestration framework and runtime for long-running, stateful agents"; shared-state graphs, checkpoint/resume, HITL state editing, streaming, short/long-term memory; LangSmith deployment and observability. | The durable-graph pattern is table stakes; LEO's differentiator must be *evidence + policy*, not just checkpoints. |
| **CrewAI** | [Introduction](https://docs.crewai.com/en/introduction): Flows manage state/events/branching; Crews of role-playing agents; claims of production-grade flows and enterprise security. No OSS/enterprise split, memory, MCP or A2A detail on that page (`PARTIAL`). | Role-play crews are commoditized; see "stop doing" §4.1. |
| **Microsoft Agent Framework** | [GitHub](https://github.com/microsoft/agent-framework): MIT, Python/.NET/Go, graph workflows (sequential, concurrent, handoff, group), checkpointing, streaming, HITL, time-travel, A2A hosting, OpenTelemetry, YAML agents, agent skills, DevUI. | Strong reference for workflow + OTel + time-travel; LEO should match the *interfaces*, not the framework. |
| **AutoGen** | [Docs](https://microsoft.github.io/autogen/stable/) and [repo](https://github.com/microsoft/autogen): **archived, no longer maintained**, no fixes/security updates, migration to Microsoft Agent Framework advised (`VERIFIED`). | Evidence that multi-agent-role-play framing aged poorly. |
| **OpenHands** | [Docs](https://docs.openhands.dev/): Software Agent SDK, Agent Server (REST/WebSocket), Agent Canvas UI, community Sandbox Server, benchmark/eval infrastructure; Cloud/Enterprise are commercial; legacy CLI and `openhands serve` GUI deprecated; licenses are per-repository. | Componentized runtime + eval harness is a good model; do not assume a single license. |
| **AutoGPT** | [Repo](https://github.com/Significant-Gravitas/AutoGPT): now a hosted/self-hostable "team of AI experts" with visual block workflows, API + MCP server and marketplace; `autogpt_platform/` is **Polyform Shield** (no competing hosted service), `classic/` and rest MIT. Classic maintenance status unclear (`PARTIAL`). | Licensing trap for reuse; visual workflow builders are crowded. |
| **Google ADK** | Redirected to [adk.dev](https://adk.dev/) before content could be retrieved (`UNVERIFIED` here). A2A's own site names ADK as a supporting framework (`VERIFIED`). | Track but do not base strategy on it. |
| **Goose (Block)** | Named as an MCP Apps client in [MCP docs](https://modelcontextprotocol.io/extensions/apps/overview.md); its docs URL 404'd in this pass (`PARTIAL`). | Minor. |

**2026 signals, low verification:** MCP's client matrix and Terminal-Bench leaderboard surfaced names such as Kiro Crew, PostHog Code, Antigravity CLI, Muse Code and mini-swe-agent (`PARTIAL`). Treat as watchlist only; no capability conclusions drawn.

---

## 2. Standards and interop — what is real as of 2026-11-18

### 2.1 MCP — specification 2026-07-28 (`VERIFIED`)

Sources: [index/llms.txt](https://modelcontextprotocol.io/llms.txt), [intro](https://modelcontextprotocol.io/docs/getting-started/intro), [changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog.md), [security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices.md), [registry](https://modelcontextprotocol.io/registry/) (accessed 2026-11-18).

Major 2026-07-28 changes:
1. **Stateless protocol** — no `initialize` handshake, no protocol-level sessions or `Mcp-Session-Id`; version and client capabilities travel in every request's `_meta`; servers MUST implement `server/discover`.
2. **Patterns** — Multi Round-Trip Requests (`resultType: "input_required"` + `inputRequests`/`inputResponses`), subscriptions via single `subscriptions/listen` POST stream, cancellation, progress, deterministic `tools/list` order.
3. **Caching** — `ttlMs` + `cacheScope` on list/read results; OpenTelemetry trace context in `_meta` (`traceparent`, `tracestate`, `baggage`).
4. **Schema** — JSON Schema 2020-12 keywords allowed, `$ref` resolution required.
5. **Deprecated** — Roots, Sampling, Logging, HTTP+SSE transport, and **Dynamic Client Registration** (migrate to Client ID Metadata Documents). Tasks moved out of core to an extension. 12-month minimum deprecation window.

Security guidance is concrete: confused-deputy/per-client consent, **token passthrough forbidden**, SSRF defenses (HTTPS-only, private/link-local IP blocks, egress proxy, DNS rebinding awareness), state-handle binding to authenticated principals, explicit consent for one-click local server commands, authorization-URL scheme validation, localhost redirect-URI risk, and progressive scope minimization.

Official extensions: **Tasks** (durable `taskId`, `working`/`input_required`/`completed`/`failed`/`cancelled`, `tasks/get`/`tasks/update`/`tasks/cancel`, polling default, notifications via `subscriptions/listen`), **Skills over MCP** (`io.modelcontextprotocol/skills`, SEP-2640 Final; `skills/list`/`skills/get` + `resources/read`, SHA-256 manifests, approvals bound to file digests, untrusted content, host MUST verify), **MCP Apps** (sandboxed iframe, `postMessage` JSON-RPC, `_meta.ui.resourceUri`, host-controlled capabilities), OAuth Client Credentials and Enterprise-Managed Authorization extensions. `VERIFIED` that these specs exist; `PARTIAL` that they are broadly implemented — the [community-maintained client matrix](https://modelcontextprotocol.io/extensions/client-matrix.md) shows *very few* filled cells (ChatGPT partial for enterprise auth; MCP Inspector partial for Apps/Skills). Official [MCP Registry](https://modelcontextprotocol.io/registry/) is **preview**, metadata-only, namespace-verified, delegates code scanning elsewhere, and is meant to be consumed through downstream aggregators, not directly by hosts.

### 2.2 AgentSkills (`VERIFIED`)

Sources: [overview](https://agentskills.io/), [specification](https://agentskills.io/specification.md) (accessed 2026-11-18).
- Directory with required `SKILL.md`; optional `scripts/`, `references/`, `assets/`.
- Required frontmatter: `name` (≤64 chars, lowercase alnum + hyphens, no leading/trailing/consecutive hyphen, **must match directory name**), `description` (1–1024 chars, what + when). Optional: `license`, `compatibility` (≤500), `metadata` (string map), `allowed-tools` (experimental, varying support).
- Progressive disclosure: metadata ~100 tokens at startup, full body <5,000 tokens recommended and <500 lines, resources on demand. `skills-ref validate` is the reference validator.
- Originally developed by Anthropic and released as an open standard; open GitHub development.
- **Origin claim check:** the [client showcase](https://agentskills.io/clients.md) rendered as empty headings via WebFetch, so adopter list is `UNVERIFIED` there. Hermes README claims compatibility but its authoring doc does not repeat it (`PARTIAL`, also noted in prior research). No byte-for-byte compatibility between Hermes/OpenClaw/LEO skills was verified.

### 2.3 A2A 1.0 (`VERIFIED`)

Sources: [A2A home](https://a2a-protocol.org/), [what is A2A](https://a2a-protocol.org/latest/topics/what-is-a2a/), [specification](https://a2a-protocol.org/latest/specification/), [releases](https://github.com/a2aproject/A2A/releases) (accessed 2026-11-18).
- Originated by Google, donated to the Linux Foundation. **v1.0.0 released 2026-03-12**, v1.0.1 on 2026-05-26; v1.0 was breaking (binding refactor, OAuth2 modernized with PKCE/device code, `tasks/list` with filtering/pagination, signed extended Agent Cards, etc.).
- Discovery via `/.well-known/agent-card.json`; capabilities in the card; core operations `message:send` / `message:stream` over JSON-RPC, gRPC, HTTP+JSON/REST or custom bindings; SSE streaming; push notifications via HTTP POST; opaque execution (internal memory/tools hidden).
- Task states include submitted/working/completed/failed/canceled/input_required/rejected/auth_required; optional capabilities are card-gated; auth schemes include API key, HTTP auth, OAuth2/OIDC, mTLS; cards can be signed.
- A2A explicitly complements MCP (agent↔agent vs agent↔tool). This is a **protocol**, not a product; no evidence was found that Hermes or OpenClaw implement it (`UNVERIFIED`).

### 2.4 ACP 1 (`VERIFIED` / `PARTIAL`)

Sources: [homepage](https://agentclientprotocol.com/), [protocol overview](https://agentclientprotocol.com/protocol/overview.md), [README](https://raw.githubusercontent.com/agentclientprotocol/agent-client-protocol/main/README.md) (accessed 2026-11-18).
- Stable wire version `1`; Apache-2.0 contributions; official SDKs in Kotlin, Java, Python, Rust, TypeScript. Scope: editor↔coding-agent communication (LSP analogy), JSON-RPC 2.0.
- Lifecycle: `initialize` → `session/new` or `session/load` → `session/prompt` with streamed `session/update`; `session/request_permission` for approvals; optional `fs/read_text_file`, `fs/write_text_file`, terminal methods, elicitation. Homepage says remote agents are work in progress.
- README links to separate Agents/Clients pages but did not name adopters in the fetched text (`PARTIAL`).

### 2.5 Frontmatter convergence and claims to be skeptical about

- **Real convergence (`VERIFIED`):** all three skill shapes require `name` + `description` in YAML frontmatter and a `SKILL.md` at the root; AgentSkills fixes naming/limits and adds `metadata`; Hermes uses `metadata.hermes` gates, OpenClaw uses `metadata.openclaw.requires`. A shared loader with adapter-specific frontmatter is defensible; a single byte-compatible loader is **not** demonstrated.
- **Real but early (`VERIFIED` + `PARTIAL`):** MCP Skills extension ties approvals to SHA-256 manifests; host and SDK support "still being implemented" per the official page.
- **Marketing/unclear:** "100+ model providers", "self-improving agent", "computer use everywhere", "agentic standard" without a spec link. "WebMCP" is ambiguous: [webmcp.dev](https://webmcp.dev/) describes an **open-source JavaScript library**, not a standards-track effort, and no W3C group page was found (404); Codex docs list "Site tools (WebMCP)" but the deep link 404'd. Do not treat WebMCP as an interoperability commitment (`UNVERIFIED`).
- **Security field is moving:** OWASP announced an **Agent Control Standard (ACS)** on 2026-09-01 aimed at runtime enforcement of what agents can access/decide/do, and a [Securing Agentic Applications V2](https://genai.owasp.org/resource/securing-agentic-applications-v2/) guide dated 2026-11-10; the announcement names "Excessive Agency" as #3 in the 2026 LLM Top 10. Control IDs/checklists were not in the fetched pages (`PARTIAL`).

---

## 3. Fifteen capability differentiators for LEO

Cost scale assumes one experienced engineer working in the existing Python/SQLite/Electron codebase: **S = 1–3 dev-days, M = 4–10, L = 11–25**. "LEO gap" derives from the local architecture/inventory/known-issues files read on 2026-11-18. Every acceptance test is written to be runnable offline with the existing mock provider where possible.

| # | Differentiator | Market evidence | Why it matters / LEO gap | Cost | Measurable acceptance test |
|---|---|---|---|---|---|
| 1 | **Evidence-bound completion with adversarial verification and an explicit verified/unverified claim ledger** | Claude `/deep-research` votes across verifiers and labels unverifiable claims; OpenHands ships eval infra; LEO already requires evidence for completion | Agents fail by *confidently wrong* completion. LEO has evidence fields but no independent cross-check or claim status taxonomy. | M | 30 seeded tasks with 10 false claims: ≥90% false claims flagged, ≤10% false refusals; every completion carries ≥1 checkable artifact and a status label. |
| 2 | **Declarative, resumable workflow scripts with deterministic replay, budgets and no model-context pollution** | Claude dynamic workflows (`agent/pipeline/parallel/phase`, replay, stall restart, 1,000-agent cap); MAF graph workflows with time-travel | Turns orchestration into reviewable, reusable code; separates "what runs next" from the model. LEO has `delegate_task` but no script runtime. | L | 100-item fan-out; `kill -9` at 50%; resume replays completed results, zero duplicate side effects, total tokens within preset cap; script is readable/rerunnable from a file. |
| 3 | **Native MCP 2026-07-28 client: stateless requests, `server/discover`, MRTR, Tasks, caching hints, OTel `_meta`** | MCP changelog and extension specs; extension matrix mostly empty | Being early on the *current* revision (not 2025 sessions) is a real interop lead; most competitors' matrix cells are blank. LEO has no MCP client. | L | Conformance against 3 official example servers: no `initialize`/session header; handles `input_required` and task handles; honors `ttlMs`; trace IDs propagate; survives server restart. |
| 4 | **Skills 2.0: AgentSkills-valid loader + content-addressed manifests, provenance and approval binding; Hermes/OpenClaw adapters** | AgentSkills spec; MCP Skills extension (SHA-256, digest-bound approvals); LEO's translation-only compatibility tests | Converts a fragile folder convention into a supply chain with revocable trust. LEO currently translates frontmatter only. | M–L | Load reference fixtures from all 3 shapes; tamper one byte → reject; changing a file revokes prior approval; provenance shown in UI; 0 executed scripts without explicit grant. |
| 5 | **Crash-safe side effects: idempotency keys persisted with checkpoints, no blind replay** | LangGraph/MAF checkpointing; MCP Tasks durable handles; LEO already refuses replay but has no process-level resume | Reliability is a product feature, not an implementation detail. | M–L | Kill process mid-tool-call across 20 trials; after restart 0 duplicate external actions, all evidence retained, task reaches a defined terminal state. |
| 6 | **Measured token economy: cache-stable prompt assembly, real tokenizers, per-task budgets and cache-hit telemetry** | Anthropic prompt-caching spec (prefix order, invalidation, 0.1× reads); Claude workflow staggering; LEO estimates chars/4 | Directly cuts cost/latency and makes "efficiency" measurable instead of a slogan. | M | ≥70% cache-read ratio on a 20-turn task; ≥30% cost reduction vs uncached baseline; 0 cache hits lost when tools are reordered; per-task cost visible in UI. |
| 7 | **Memory governance: provenance, confidence, TTL, contradiction resolution and quarantine of injected content** | Hermes/OpenClaw curated Markdown memory; MCP untrusted-content rules; LEO has hash embeddings and hybrid search | Memory poisoning is the long-game attack; retrieval quality alone is not enough. | M | Injected webhook/browser text cannot create or edit memory; 100 seeded contradictions produce ≥90% resolved or quarantined items; every memory item has source + timestamp + scope. |
| 8 | **Capability-token policy engine with taint tracking and step-up approval (`OWASP ACS` direction)** | OWASP ACS announcement + Securing Agentic Applications V2; MCP SSRF/token-passthrough/consent rules; Claude treats inter-agent messages as untrusted | Untrusted inbound content must never inherit operator authority. LEO has permission tiers but no taint propagation. | L | 20-case red team (webhook, browser page, skill doc, MCP tool output, teammate message): 0 high-risk action approvals sourced from untrusted text; all denials audited. |
| 9 | **Real execution boundary: process/container sandbox, egress allowlist, fixed DNS-rebinding check** | OpenClaw docs say sandbox is opt-in and "not a perfect boundary"; MCP local-server security; LEO isolation is cooperative and Docker unavailable here | This is the difference between "policy" and "enforcement". | L | Tool cannot read paths outside allowlist; metadata IP and private ranges blocked after DNS resolution at connect time; escaped command cannot reach host credentials. |
| 10 | **Evals as a first-class product surface (golden tasks + red team + cost/latency gates)** | OpenHands eval infra; Terminal-Bench 4.1 reports resolution rate, cost and tokens; METR's time-horizon caveats | Without a regression suite, every model/provider change is a gamble. LEO has test modules, not a product-level eval loop. | M | CI runs 30 offline golden tasks + 20 red-team cases; report pass rate, p95 latency, cost/task; release blocked on >5% regression. |
| 11 | **Timeline observability: OTel traces, token/cost/approval views, replayable run inspector in Electron** | MCP OTel `_meta`; Claude `/workflows` progress view; MAF built-in OTel | Debuggability is what makes autonomy acceptable to users. LEO has audit tables but no visual timeline. | M | Given a run ID, UI reconstructs every prompt/tool/policy decision with secrets redacted; trace export is valid OTel JSON; 0 secrets in logs (automated scan). |
| 12 | **Single control plane with cross-surface handoff + ACP adapter for editor agents** | Claude teleport/remote control/channels; ACP v1 stable; OpenClaw WebSocket control plane | Session continuity is a visible differentiator that competitors are actively shipping. | L | Start task in CLI → approve in Electron → continue in chat with identical task ID; ACP `initialize`/`session/new`/`request_permission` pass an official conformance-style script. |
| 13 | **A2A 1.0 federation with signed Agent Cards, capability negotiation and cross-boundary budgets** | A2A 1.0 spec + 2026-03 release; Linux Foundation governance | Positions LEO as an *equal peer* to other agents rather than a wrapper; few personal-agent products have adopted it. | M–L | Publish LEO card; delegate one task to a reference A2A agent and retrieve artifacts; unsigned card/mismatched version rejected; per-task delegation budget enforced. |
| 14 | **Async human-in-the-loop as a primitive: `input_required`, durable approval queues, fail-closed timeouts** | MCP Tasks/MRTR; Claude HITL + hooks; OpenClaw hooks lack durable guarantees | Long-running autonomy needs pause/resume across hours and devices, not just a modal. | M | Task pauses for input, resumes 12h later with same idempotency key from a different surface; expired approvals fail closed; no orphaned side effects. |
| 15 | **Curated self-improvement: skill proposals from trajectories with tests and human review** | Hermes "learning loop" (marketing claim); MCP Skills untrusted-content rules; LEO non-goal is *uncontrolled* self-modification | Captures the upside of Hermes's headline idea without the security debt. | M | 10 trajectories generate ≥3 draft skills; none activate without approval + passing tests; unsafe drafts quarantined and explainably rejected. |

**Highest leverage combination:** #1 + #2 + #6 + #10 form a single "trustworthy autonomy" engine (verify, orchestrate, measure, regress). #3 + #4 + #13 are the interoperability wedge. #8 + #9 + #11 are the enterprise/security moat. #12 + #14 + #15 are the UX story.

---

## 4. Five things LEO should NOT spend effort on

1. **Generic role-playing "crews" as the primary abstraction.** AutoGen, the archetype, is archived and unmaintained (`VERIFIED`); Claude Code documents agent teams as experimental with high token cost, task-status lag and no nested teams (`VERIFIED`); CrewAI's role-play framing is now commodity. Do focused worker + independent verifier + explicit workflow script instead.
2. **Unreviewed autonomous self-modification or one-click skill-hub installs.** Hermes's "built-in learning loop" is a marketing claim, OpenClaw treats third-party skills as untrusted code, and MCP Skills requires approval bound to file digests (`VERIFIED`). LEO's own non-goals exclude uncontrolled self-modification. Build #15's review gate, not an auto-installer.
3. **A proprietary agent protocol or another gateway to replace MCP/A2A/ACP.** All three now have concrete, versioned specs (`VERIFIED`), and the MCP extension matrix is mostly empty — a gap to fill, not a reason to fork. A custom protocol would burn months and reduce interop.
4. **Breadth races: "20+ channels" and computer use on every OS before depth.** OpenClaw's Windows/Linux computer use is experimental and platform-limited; WhatsApp Via Baileys carries unverified ToS risk; LEO's live channel tests are credential-blocked (`VERIFIED`/`PARTIAL`). Two channels with excellent approval UX beat twenty with mock-only coverage.
5. **Token-maximizing "always fan out" and benchmark chasing.** Claude's own docs warn workflows/ultracode use materially more tokens, cap runs for safety, and recommend small slices first; agent teams show diminishing returns; METR explicitly warns benchmark progress does not imply reliable real-world projects (`VERIFIED`). Optimize for verified outcomes per dollar, not agent count or leaderboard rank. (Related trap: buying a vector DB for all memory — LEO's known limitation is *scoring at modest volume*, so provenance and governance (#7) beat infrastructure first.)

---

## 5. Strategic recommendation — one page

**Thesis.** LEO will not leap ahead by adding more tools or channels. It will leap ahead by becoming the *most auditable autonomous agent OS*: every task has a scripted plan, a budget, an evidence ledger, an adversarial verifier, a capability-token policy, and a replayable trace — and it speaks the 2026 standards (MCP 2026-07-28, AgentSkills, A2A 1.0, ACP 1).

**Sequence (assumes 1–2 engineers; milestones are acceptance tests from §3):**

| Phase | Time | Build | Exit criterion |
|---|---|---|---|
| 0 — Measure before adding | Week 1 | #10 eval harness + #6 prompt-cache/token instrumentation (offline mock provider) | 30-task golden suite in CI; ≥70% cache-hit on a 20-turn task; cost/task baseline published |
| 1 — Trustworthy autonomy | Weeks 2–4 | #1 adversarial verification + claim ledger; #5 idempotent crash recovery; #14 async HITL | 30 seeded false claims caught; 20 kill/restart trials with zero duplicate side effects; 12h approval resume works |
| 2 — Standards wedge | Weeks 4–7 | #3 MCP 2026-07-28 client + Tasks/MRTR; #4 Skills 2.0 manifests + 3 frontmatter adapters | 3 official servers interop; tampered skill rejected; Hermes/OpenClaw/AgentSkills fixtures load with provenance |
| 3 — Programmable scale | Weeks 7–10 | #2 resumable workflow scripts with budgets/deterministic replay; #11 OTel timeline + Electron run inspector | 100-agent fan-out resumes with no duplicate effects; any run reproducible from its trace |
| 4 — Moat and reach | Weeks 10–14 | #8 capability tokens + taint; #9 real sandbox/egress; #12 ACP adapter + cross-surface handoff; #13 A2A federation; #15 reviewed skill proposals | 20-case red team yields 0 escalations; ACP script passes; A2A card/delegation works; no self-authored skill activates unreviewed |

**Stop doing now:** (a) adding channel adapters beyond the two with live tests; (b) building Hermes/OpenClaw runtime compatibility beyond declarative frontmatter translation; (c) any proprietary protocol work; (d) uncontrolled skill autorun or registry auto-install; (e) more "12 specialists" breadth without a verifier and a measured outcome per task. Defer Docker-dependent sandbox claims until Docker exists — label isolation *cooperative* in docs and UI.

**Success metric for the leap:** on the golden suite, LEO achieves ≥90% verified-correct completions at ≥30% lower cost per task than its current baseline, with zero unreviewed high-risk actions and a replayable trace for every run. That is a defensible claim competitors' docs do not currently make.

---

## 6. Explicitly unverified / needs follow-up

- WebSearch: no live results; no independent search-engine corroboration of any finding.
- Competitor runtime behavior, latency, reliability, sandbox escapes, provider quirks, actual extension support in shipped builds — **not tested**.
- Adoption/market share, star counts, "only agent" and "self-improving" marketing claims — **not independently verified**.
- Hermes↔OpenClaw↔AgentSkills byte-level skill compatibility — **not verified**; only shared required frontmatter is verified.
- Codex deep pages (auto-review, long-running work, WebMCP) and Cursor background agents returned 404; A2A "what's new" and ADK docs redirects could not be retrieved. OWASP ACS control details and the Agentic Applications Top 10 entries were not in the fetched announcement.
- A2A specification page was truncated by the fetch tool after ~100k characters; the summary covers the first portion.
- Agent Skills "Client Showcase" was JS-rendered and returned no adopter names.
- Third-party registries' licenses, ClawHub trust envelope, remote-browser vendor terms and WhatsApp ToS — not reviewed (out of scope, no credentials).
- OTel/Tasks/Skills extension *host support* is early per the official community matrix; implementing them is a differentiator precisely because it is not yet ubiquitous.

## 7. Sources (all accessed 2026-11-18)

**Prior in-repo baseline:** `<repo>/docs/RESEARCH_HERMES_OPENCLAW.md`; `<repo>/LEO_ARCHITECTURE.md`; `<repo>/docs/CAPABILITY_INVENTORY.md`; `<repo>/LEO_KNOWN_ISSUES.md`.

**Hermes / OpenClaw:** https://raw.githubusercontent.com/NousResearch/hermes-agent/main/README.md ; https://raw.githubusercontent.com/openclaw/openclaw/main/README.md ; https://github.com/openclaw/openclaw/blob/main/docs/concepts/architecture.md ; https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/creating-skills.md ; https://github.com/openclaw/openclaw/blob/main/docs/tools/skills.md

**Coding agents:** https://code.claude.com/docs/en/overview ; https://code.claude.com/docs/en/workflows.md ; https://code.claude.com/docs/en/agent-teams.md ; https://platform.claude.com/docs/en/build-with-claude/prompt-caching ; https://learn.chatgpt.com/docs ; https://cursor.com/docs/agent/overview ; https://github.com/anthropics/claude-code/pulls (README fetch returned 503)

**Frameworks:** https://docs.langchain.com/oss/python/langgraph/overview ; https://docs.crewai.com/en/introduction ; https://github.com/microsoft/agent-framework ; https://microsoft.github.io/autogen/stable/ ; https://github.com/microsoft/autogen ; https://docs.openhands.dev/ ; https://github.com/Significant-Gravitas/AutoGPT ; https://a2a-protocol.org/ (ADK mention)

**Standards:** https://modelcontextprotocol.io/llms.txt ; https://modelcontextprotocol.io/docs/getting-started/intro ; https://modelcontextprotocol.io/specification/2026-07-28/changelog.md ; https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices.md ; https://modelcontextprotocol.io/extensions/tasks.md ; https://modelcontextprotocol.io/extensions/skills.md ; https://modelcontextprotocol.io/extensions/apps/overview.md ; https://modelcontextprotocol.io/extensions/client-matrix.md ; https://modelcontextprotocol.io/registry/ ; https://modelcontextprotocol.io/docs/2026-07-28/develop/build-with-agent-skills.md ; https://agentskills.io/ ; https://agentskills.io/specification.md ; https://agentskills.io/clients.md ; https://a2a-protocol.org/ ; https://a2a-protocol.org/latest/topics/what-is-a2a/ ; https://a2a-protocol.org/latest/specification/ ; https://github.com/a2aproject/A2A/releases ; https://agentclientprotocol.com/ ; https://agentclientprotocol.com/protocol/overview.md ; https://raw.githubusercontent.com/agentclientprotocol/agent-client-protocol/main/README.md ; https://webmcp.dev/

**Security / evaluation:** https://genai.owasp.org/ ; https://genai.owasp.org/2026/09/01/owasp-genai-security-project-unveils-2026-top-10-for-llm-applications-new-agent-control-standard-and-sponsors-as-community-tops-30000-members/ ; https://genai.owasp.org/resource/securing-agentic-applications-v2/ ; https://genai.owasp.org/llm-top-10/ ; https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/ ; https://www.tbench.ai/ ; https://www.swebench.com/ ; https://docs.letta.com/guides/agents/memory/
