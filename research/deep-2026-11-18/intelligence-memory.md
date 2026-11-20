# Deep research: intelligence, memory, and multi-agent orchestration for LEO

**Date:** 2026-11-18. **Scope:** authoritative public docs/papers plus read-only source audit of `<repo>` (current branch `main`, HEAD `<head>`). **Access date for every URL below:** 2026-11-18. **No LEO source was modified; no credentials, paid calls, or publishing used.** All external source statements are attributed; statements about LEO are grounded in `file:line`.

---

## 0. Executive summary — the ten findings that matter

1. **LEO has no prompt caching at all.** `Usage` has no cache-token fields (`leo/contracts.py:129-136`), no provider body emits `cache_control` (`leo/models/provider.py:472-494`, `615-636`), and no checkpoint/prefix-stability logic exists. Anthropic documents 0.05–0.1× cache-hit pricing, 4 breakpoints, and exact-prefix requirements; Hermes and OpenClaw both implement caching in source. This is the single largest measurable efficiency gap.
2. **Context compaction is dead code in the main chat path.** `LeoApp.ask()` calls `self.context.optimize(...)` without `provider=` (`leo/core/app.py:163`, `:202`), so `_compact()` (`leo/models/context.py:138-167`) never executes; only mechanical truncation runs. Budget is hard-coded to 8,000 estimated tokens, independent of provider window (`leo/config.py:45`).
3. **Memories are injected twice into chat prompts.** `ask()` appends them into the system string (`leo/core/app.py:150-155`) and `ContextOptimizer._memory_messages()` injects a second `name="memory"` system message (`leo/models/context.py:114-125`); dedup only removes byte-identical messages (`leo/models/context.py:62-83`). `memories_used` reports all searched IDs, not the ones actually selected under budget (`leo/core/app.py:215`).
4. **"Intelligent routing" is currently cost sorting plus metadata preference.** `classify_task()` is used only to write a log field (`leo/models/router.py:264`); the category is appended to candidate tuples but omitted from the sort key (`leo/models/router.py:234-235`). `request.model` does not affect provider selection, and agent `model_preferences` are capability tags (e.g. `["general","text","tools"]`, `leo/agents/registry.py:30`) passed as a literal model ID (`leo/agents/engine.py:743-754`), which would send an invalid `model` to a live provider (`leo/models/provider.py:479`).
5. **Structured output is pass-through, not validation.** `response_format` is forwarded raw to Anthropic `output_config.format` / OpenAI `response_format` (`leo/models/provider.py:489-491`, `:628-631`); nothing parses or validates the returned JSON, checks refusals, or checks `finish_reason`/incomplete status. The only validator is "non-empty content" (`leo/models/router.py:101-103`). LEO already has a capable JSON-Schema validator for tool arguments (`leo/tools/registry.py:63-153`) that can be reused.
6. **Memory is a record store, not a memory system.** Writes are user-initiated only (`leo/core/app.py:219-223`); no turn extraction, prefetch, pre-compaction flush, reflection, or scheduled consolidation exists. `Consolidator` is never wired into the app or scheduler (grep: only `leo/memory/consolidation.py`). Default "semantic" retrieval is deterministic hashing/trigram cosine (`leo/memory/search.py:46-68`), i.e. lexical, not trained semantic search. Seven categories (`leo/contracts.py:50-58`) share one undifferentiated table with no category-specific processing.
7. **Agent tool scoping is declarative only.** `AgentSpec.tool_permissions` is stored (`leo/agents/registry.py:221-245`) but never enforced; `_model_worker` passes *all* tool specs to every agent (`leo/agents/engine.py:753`). Hermes enforces `allowed_toolsets`/`blocked_tools`; OpenClaw denies session/message tools to subagents by default.
8. **Solid parts are genuinely solid:** evidence-gated task completion and non-replay recovery (`leo/tasks/engine.py:414-447`, `:505-575`, `:579-612`), worktree isolation with explicit merge and path checks (`leo/agents/isolation.py:173-281`), independent verification with fail-closed reviewers (`leo/agents/verification.py:238-302`), versioned/provenance-preserving memory updates and fact-promotion guard (`leo/memory/store.py:191-322`, `leo/memory/consolidation.py:158-240`), and provider/secret hygiene (`leo/models/provider.py:247-289`).
9. **Evaluation exists but is not an eval harness.** `MemoryEvaluation` computes real precision@k/MRR from caller labels (`leo/memory/evaluation.py:114-208`) but the only fixture has 3 memories and 2 queries (`tests/test_memory.py:348-381`). No labelled routing, compaction, verification, or cost fixtures exist; `Evaluation.report()` aggregates arbitrary recorded numbers (`leo/core/evaluation.py:21-49`).
10. **Several docs overstate efficacy.** "Hybrid search" and "model routing between configured models | Tested" (`docs/CAPABILITY_INVENTORY.md:77-78`, `LEO_TEST_REPORT.md:64`) describe mechanism tests with mocks, not measured quality. "Measured model usage" (`LEO_PROGRESS.md:27`) is measured on mock providers. The master plan itself forbids unmeasured savings claims (`Master_plam.md:344`, `:825`), which the current docs do not satisfy.

---

## 1. Sources and method

Web research used official documentation and primary papers only, fetched via `WebFetch` (and read-only `gh api` for upstream source where the official docs were ambiguous). `WebSearch` was unreliable in this environment (it returned "no live web-search access"); all findings below come from direct fetches. All URLs accessed **2026-11-18**.

| # | Source | URL | Used for |
|---|---|---|---|
| S1 | Anthropic, Prompt caching (redirect from `docs.anthropic.com`) | https://platform.claude.com/docs/en/build-with-claude/prompt-caching | cache breakpoints, thresholds, TTL, token accounting, prefix stability |
| S2 | Anthropic, Effective context engineering for AI agents | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | context budget, compaction content, JIT retrieval, subagent summaries |
| S3 | Anthropic, Built a multi-agent research system | https://www.anthropic.com/engineering/built-multi-agent-research-system | orchestration, parallel fan-out, token economics, eval method |
| S4 | Anthropic, Structured outputs | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | `output_config.format`, strict tools, schema limits, grammar cache |
| S5 | Anthropic, Building effective agents | https://www.anthropic.com/engineering/building-effective-agents | workflow vs agent, routing/evaluator patterns, simplicity |
| S6 | OpenAI, Structured outputs | https://developers.openai.com/api/docs/guides/structured-outputs | strict mode, refusals, incomplete responses, business-rule validation |
| S7 | OpenAI, Evals guide | https://developers.openai.com/api/docs/guides/evals | dataset + graders + run reports; platform deprecated Oct/Nov 2026 |
| S8 | MemGPT (arXiv:2310.08560) | https://arxiv.org/abs/2310.08560 | tiered virtual context, interrupts |
| S9 | Generative Agents (arXiv:2304.03442) | https://arxiv.org/abs/2304.03442 | observation/reflection/planning ablation |
| S10 | LongMemEval (arXiv:2410.10813) | https://arxiv.org/abs/2410.10813 | 500 questions, five memory abilities, 30% drop |
| S11 | Mem0 (arXiv:2504.19413) | https://arxiv.org/abs/2504.19413 | extraction/consolidation, LOCOMO gains, latency/token savings |
| S12 | Zep (arXiv:2501.13956) | https://arxiv.org/abs/2501.13956 | temporal knowledge graph, DMR/LongMemEval gains |
| S13 | Lost in the Middle (arXiv:2307.03172) | https://arxiv.org/abs/2307.03172 | U-shaped position sensitivity |
| S14 | RouteLLM (arXiv:2406.18665) | https://arxiv.org/abs/2406.18665 | preference-trained routers, >2× cost reduction |
| S15 | τ-bench (arXiv:2406.12045) | https://arxiv.org/abs/2406.12045 | final-state grading, pass^k reliability |
| S16 | SWE-bench (arXiv:2310.06770) | https://arxiv.org/abs/2310.06770 | 2,294 issue/PR tasks across 12 repos |
| S17 | Hermes context engine plugin doc | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/developer-guide/context-engine-plugin.md | plugin contract, cache-preserving no-op selection |
| S18 | Hermes memory provider plugin doc | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/developer-guide/memory-provider-plugin.md | `prefetch`, `sync_turn`, `on_pre_compress` v2 checkpoint |
| S19 | Hermes sessions doc | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/user-guide/sessions.md | FTS5 session search, lineage, resume guards |
| S20 | Hermes subagent lifecycle API | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/developer-guide/subagent-lifecycle-api.md | versioned handles, toolset narrowing, non-reconnect |
| S21 | Hermes prompt assembly doc | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/developer-guide/prompt-assembly.md | stable/context/volatile tiers, cache-prefix design |
| S22 | Hermes memory feature doc | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/user-guide/features/memory.md | 2,200/1,375 char limits, frozen snapshot |
| S23 | Hermes context compression & caching doc | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/developer-guide/context-compression-and-caching.md | 50%/85% thresholds, provider usage anchors |
| S24 | Hermes config defaults (source) | https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/config_defaults.py | `threshold: 0.50`, `protect_last_n: 20`, `target_ratio: 0.20` |
| S25 | Hermes configuring-models doc | https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/docs/user-guide/configuring-models.md | model switch resets cache; measured cache-rewrite rates |
| S26 | OpenClaw context engine doc | https://raw.githubusercontent.com/openclaw/openclaw/main/docs/concepts/context-engine.md | hooks, `ownsCompaction`, `commitTurn`, quarantine |
| S27 | OpenClaw memory doc | https://raw.githubusercontent.com/openclaw/openclaw/main/docs/concepts/memory.md | file memory, silent pre-compaction flush, hybrid search |
| S28 | OpenClaw subagents doc | https://raw.githubusercontent.com/openclaw/openclaw/main/docs/tools/subagents.md | isolation defaults, tool denies, announce chain |

Upstream facts in the existing LEO research note (`docs/RESEARCH_HERMES_OPENCLAW.md`) were spot-checked against source: the 50% in-agent threshold, 85% gateway hygiene threshold, and 20 protected messages are confirmed (S23, S24). Two nuances to correct: Hermes' built-in `MEMORY.md`/`USER.md` are flushed at **session boundaries/reset**, not generally "before compression" (S19, S22); `on_pre_compress` is a **provider-plugin** hook with an optional v2 durable-checkpoint contract (S18). Current Hermes char limits are 2,200 and 1,375 (S22), not stated in the LEO note.

---

## 2. State-of-the-art patterns (2025–2026)

### 2.1 Context engineering and compaction

- Treat context as a finite "attention budget" and use the "smallest possible set of high-signal tokens"; every added token degrades attention roughly with *n²* pairwise relationships (S2). Prefer **just-in-time retrieval** (lightweight identifiers, `glob`/`grep`-style discovery) over preloading everything; hybrid = preload stable rules, retrieve the rest (S2).
- Compaction should preserve "architectural decisions, unresolved bugs, and implementation details", drop redundant tool output, and be tuned for **recall first, then precision**. Claude Code keeps the summary plus the five most recently accessed files (S2).
- Hermes operationalizes this with three compression layers: gateway hygiene at **85%** of context (uses API-reported tokens → persisted usage anchor → rough estimate, fires only with ≥4 messages), in-agent `ContextCompressor` at **50%** (floored to 75% below 512K windows), `target_ratio: 0.20`, `protect_last_n: 20` (S23, S24). Usage anchors carry a content fingerprint and survive restarts; image and reasoning token costs are *learned from provider usage* rather than guessed; telemetry records `tokens_reclaimed` with a `token_count_method` label and explicitly says scripted checks are "not vendor tokenizer or billing evidence" (S23).
- Hermes' `ContextEngine` contract has `update_from_response`, `should_compress`, `compress`, optional `select_context` (request-only, runs before cache-control, must be stable to preserve cache reuse), and fail-open defaults (S17). OpenClaw's is richer: `ingest`/`assemble`/`compact`/`afterTurn` plus optional `bootstrap`/`maintain`/`prepareSubagentSpawn`; `assemble` must return `estimatedTokens` and messages fitting `tokenBudget`; `ownsCompaction` disables built-in compaction; `commitTurn` is atomic and idempotent; contract failures quarantine the engine and fall back to `legacy`; tool-result trimming runs regardless of engine (S26).
- **Position matters:** U-shaped performance, best at beginning/end, degraded in the middle (S13). Compaction and memory injection should therefore place high-value items at edges, not bury them.

### 2.2 Prompt caching

Anthropic's documented mechanics (S1): cache hierarchy `tools → system → messages`; at most **4 breakpoints**; minimum cacheable prefix **512–4,096 tokens** depending on model; TTL **5 min (1.25× write)** or **1 hour (2× write)**; cache hits roughly **0.05–0.1×** base input price; `input_tokens` in the response counts **only tokens after the last breakpoint**, with `total_input = cache_read + cache_creation + input`; exact byte-identical prefixes required; 20-block lookback; put `cache_control` on the **last tool** and at the end of the static prefix; avoid volatile timestamps at the breakpoint; changing `output_config.format` invalidates the thread's cache; caches become usable only after the first response starts. Hermes confirms the operational payoff (~75–90% discount) and measures cache rewrites when switching model/transport (14–20% on one native route vs 0 on chat) (S25). Both Hermes and OpenClaw implement `cache_control` in source.

### 2.3 Routing, escalation, and model choice

- Route by task/capability/cost/latency with a trained or calibrated router; RouteLLM reports **>2× cost reduction without quality loss** using preference data and tunable strong/weak thresholds (S14). Anthropic recommends routing as a workflow primitive and warns to add agent complexity only when measured (S5).
- Multi-agent economics: agents use ~4× chat tokens, multi-agent ~15×; token usage alone explained **80% of variance** on BrowseComp; 3–5 parallel subagents cut complex-query time by up to **90%**; Opus lead + Sonnet subagents beat single Opus by **90.2%** on an internal research eval; effort should scale with task complexity (1 agent/3–10 calls for fact-finding, >10 subagents for complex work) (S3).
- Escalation should be bounded in attempts **and** wall-clock/cost, avoid replaying emitted stream output, and record every attempt (LEO does the first two correctly; see §3.1).

### 2.4 Structured output validation

- Anthropic `output_config.format = {type: "json_schema", schema: {...}}`; strict tool use via `strict: true`; `additionalProperties: false` required; numeric/string constraints are *not* API-enforced and must be checked locally; grammar compilation is cached 24h; changing format invalidates prompt cache (S4).
- OpenAI: `strict: true` constrains **completed, non-refusal** output only; applications must handle `refusal`, incomplete responses, API errors, and business rules the schema cannot express (S6).
- Therefore: provider-constrained decoding + local validation + explicit refusal/incomplete states is the pattern; never treat "non-empty string" as structured validity.

### 2.5 Long-term memory

- **Tiers and write path:** MemGPT's "virtual context management" moves data between fast/slow tiers with interrupts (S8); Generative Agents' ablation shows observation, planning, and reflection each contribute critically; memories are stored in natural language and synthesized into higher-level reflections (S9). Mem0 extracts, consolidates, and retrieves salient information; with graph memory it reports **26% relative LLM-judge gain over OpenAI**, graph +~2%, **91% lower p95 latency** and >90% token savings vs full context on LOCOMO (S11). Zep's Graphiti adds temporal knowledge graphs with historical relationships: 94.8% vs 93.4% DMR vs MemGPT, up to **18.5%** LongMemEval accuracy gain and **90%** latency reduction (S12).
- **Retrieval:** hybrid lexical + vector is the norm (OpenClaw `memory_search` combines keyword with optional embeddings; PostgreSQL uses `ts_rank` calibrated `r/(r+0.0027)`) (S27). Recency/confidence/link boosts should be *calibrated*, not arbitrary; recency half-life and IDF are table stakes. LongMemEval (500 questions) tests extraction, multi-session reasoning, **temporal reasoning, knowledge updates, abstention**, and reports a **30% accuracy drop** for commercial assistants across sustained interaction (S10).
- **Flush and durability:** OpenClaw runs a silent pre-compaction memory-flush turn by default (`agents.defaults.compaction.memoryFlush.enabled`) on a private conversation copy, append-only new text, failure does not block compaction; `MEMORY.md` is a curated digest, daily notes are indexed but not bootstrapped, and injected copies are truncated while files stay intact (S27). Hermes freezes the system-prompt memory snapshot at session start specifically to preserve prefix cache, with live state shown through tools (S22), and exposes an optional checkpointed `on_pre_compress` provider hook (S18).
- **Search without LLM:** Hermes `session_search` is SQLite FTS5 over stored messages — "tens of milliseconds" discovery, 1–2 ms scroll — with lineage dedup and ±5 message context (S19). This is a cheap, high-value pattern LEO's message table could support.

### 2.6 Multi-agent orchestration

- Lead agent plans and delegates with **specific objective, output format, tool guidance, and task boundaries**; subagents return condensed 1,000–2,000-token summaries after exploring tens of thousands of tokens (S2, S3). Isolation is both contextual (clean window) and capability-based (narrow toolsets).
- Hermes `delegate_task` gives each child an isolated session/toolset and only returns the final summary; the lifecycle API adds versioned opaque handles, `status/wait/cancel/steer/result/reconnect`, immutable idempotent results capped at 32k chars, toolset narrowing, cooperative cancel, in-process-only reconnect, and metadata retained one hour (S20). OpenClaw spawns `agent:<id>:subagent:<uuid>` sessions, isolated by default, cheaper child model via `agents.defaults.subagents.model`, **no session/message tools by default**, depth/spawn caps, and an announce chain for results (S28).
- Verification: use independent reviewers, rubric-based LLM judges, and evaluate **final state and checkpoints** rather than a single trajectory; start with ~20 real queries; a single judge prompt with a 0–1 score and pass/fail was more consistent than multiple judges (S3, S7). τ-bench shows reliability must be measured with **pass^k** (all k trials succeed): GPT-4o <50% task success and pass^8 <25% in retail (S15).

---

## 3. LEO audit: solid / superficial / missing

### 3.1 Intelligence (models, context, routing, structured output)

**Solid**
- Provider boundary is clean: only `ModelProvider.complete/stream`; credential values from env by name, redacted from payloads, logs, errors and usage metadata (`leo/contracts.py:259-265`, `leo/models/provider.py:247-289`, `:300-312`, `leo/models/router.py:149-158`).
- Escalation is bounded to `escalation_limit + 1` providers and never replays already-emitted stream chunks (`leo/models/router.py:291-329`, `:331-370`); tests assert the bound (`tests/test_models.py:409-443`).
- Per-attempt usage rows include latency, success, estimated flag, category, attempt number and scrubbed errors (`leo/models/router.py:246-289`).
- Context optimizer is non-mutating, deduplicates, trims tool output and enforces a budget (`leo/models/context.py:47-83`, `:127-136`, `:169-190`); tests verify the estimate-based bound (`tests/test_models.py:484-533`).

**Superficial / ineffective in the live path**
- **No caching** (`leo/contracts.py:129-136`; `leo/models/provider.py:472-494`, `:615-636`). No `cache_control`, no stable-prefix assembly, no cache-token accounting. Anthropic API version is hard-coded `2023-06-01` (`leo/models/provider.py:29`, `:431-436`), while structured-output docs use `2026-10-21` (S4) — a real-gateway compatibility risk.
- **Compaction never runs in chat** (`leo/core/app.py:163`, `:202` vs `leo/models/context.py:192-216`). When called directly, `_compact()` truncates each old message to 1,000 chars before summarizing (`leo/models/context.py:145`) and has no preservation checklist, no provenance, no metric.
- **Duplicate memory injection and misleading metric** (`leo/core/app.py:150-155`, `:215`; `leo/models/context.py:114-125`, `:62-83`).
- **Classification is decorative** (`leo/models/router.py:56-98`, `:234-235`, `:264`); `_rank` scores only cost + latency + reliability + explicit metadata preference (`:214-236`). Agent `task_category=spec.id` (`leo/agents/engine.py:750`) is not even a valid `TASK_CATEGORIES` value (`leo/models/router.py:26-39`).
- **Model preferences are misused** (`leo/agents/registry.py:30-152`, `leo/agents/engine.py:743-754`, `leo/models/provider.py:479`, `:619`): capability tags become a literal `model` field. `AgentSpec.context_budget` is never read anywhere; `tool_permissions` never enforced (grep confirms only schema/db references).
- **Structured output is unverified** (`leo/models/provider.py:489-491`, `:628-631`, `:649-657`; `leo/models/router.py:101-103`). Refusals and incomplete OpenAI responses are not surfaced; Anthropic and OpenAI `response_format` shapes are not normalized; tests only assert the wire body (`tests/test_models.py:537-560`).
- **Token accounting is estimate-first** (`leo/models/provider.py:58-97`): `estimate_tokens = ceil(chars/4)`; tool schemas are double-counted in `estimate_request_tokens` (`:65-68` adds descriptions *and* the full JSON dump); the router cannot distinguish cached from uncached input. `memory_budget = budget // 4` is an uncalibrated constant (`leo/models/context.py:96`).
- `_fit()` can orphan tool calls/results by dropping oldest non-system messages without repairing provider turn structure (`leo/models/context.py:169-190`).

**Missing**: provider-window-aware budget; usage anchors with fingerprint; compaction telemetry (`tokens_reclaimed`, method labels); context lifecycle hooks; message IDs/lineage; session FTS5 search; cache-breakpoint planning; cost/deadline-bounded escalation.

### 3.2 Memory

**Solid**
- Durable CRUD with version rows, auditable `memory.changed` events, expiry and user-controlled deletion (`leo/memory/store.py:141-185`, `:191-322`, `:324-351`, `:477-508`).
- Explicit guard against silent promotion of assumption/unverified/temporary to fact (`leo/memory/store.py:19-22`, `:225-238`).
- Links, versions, supersedes and provenance-preserving consolidation with source attribution (`leo/memory/store.py:399-475`; `leo/memory/consolidation.py:158-240`).
- Retrieval combines BM25-ish term scoring, cosine on hashed vectors, confidence and recency, with link boosts and measured `last_stats` (`leo/memory/search.py:143-235`, `:267-302`).
- An honest labelled retrieval evaluator with standard P@k denominator, MRR and recall (`leo/memory/evaluation.py:114-208`).

**Superficial**
- "Semantic" is signed feature hashing of unigrams/stems and character trigrams (`leo/memory/search.py:46-68`, `:253-265`) — no trained embeddings; the docs acknowledge this (`LEO_KNOWN_ISSUES.md:20-24`). Weights (0.45/0.25/0.15/0.15, half-life 30d) are uncalibrated defaults (`leo/memory/search.py:117-122`) with no labelled tuning.
- Seven categories and four kinds are labels on one table; no category-specific write/retention/consolidation policy exists (`leo/contracts.py:50-65`, `leo/memory/store.py:141-181`).
- Only `expires_at` exists — no `valid_from/valid_to`, no contradiction resolution beyond `supersedes`, no access counts, no importance score, no reflection or procedural skill extraction (`leo/contracts.py:160-174`).
- Consolidator and `MemoryEvaluation` are not wired into app, CLI, scheduler, or API (grep: only their own modules and tests).
- Retrieval scans up to 10,000 records and embeds each candidate in Python (`leo/memory/search.py:165-199`) — fine for the documented "modest volumes", not a real index.

**Missing**: automatic turn extraction, prefetch/queued prefetch, pre-compaction flush, JIT identifiers, temporal reasoning/abstention, fact-vs-quote separation for untrusted content, session transcript search, reranking, calibrated weights, held-out retrieval fixtures.

### 3.3 Orchestration and evaluation

**Solid**
- Topological dependency waves, cycle detection, bounded depth, per-agent task limits/semaphores, conflict detection, and shared correlation state (`leo/agents/engine.py:172-189`, `:214-247`, `:345-357`, `:373-487`, `:533-552`).
- Worktree isolation validates verification, path escape, untracked files, deletions, allowed files, and refuses automatic merge; integration checks destination dirtiness and uses `merge-tree` (`:173-281`).
- Verification requires completion, evidence presence/validity/non-contradiction, expected-output or peer consistency; reviewers must be independent; errors fail closed (`leo/agents/verification.py:189-302`, `leo/agents/engine.py:636-658`).
- Durable tasks require explicit evidence-backed completion and never replay workers with possible side effects (`leo/tasks/engine.py:414-447`, `:505-575`, `:579-612`).
- Metrics, regression detection, and gated improvement proposals refuse adoption without tests and a human approver (`leo/core/evaluation.py:21-49`, `:90-150`).

**Superficial**
- No per-agent tool allowlist at execution: all specs are passed to every model worker (`leo/agents/engine.py:753`) despite declared `tool_permissions` (`leo/agents/registry.py:26-160`).
- `shared_state` is an in-process dict (`leo/agents/engine.py:241`, `:388-392`) — agent messages are durable events (`:271-289`) but shared artifacts/results are not durable across restart; no opaque versioned handles, cancel/steer, or announce chain.
- No run-level token/cost/time budget: `AgentRequest` has no timeout or token limit (`leo/agents/engine.py:154-170`); depth limit defaults to 3 (`leo/config.py:26`).
- Verdict parsing is keyword-based and order-sensitive (`leo/agents/verification.py:93-121`): negative words checked before positive, so "not incorrect" is rejected and "APPROVE with caveats" accepted. Evidence-support check is token overlap (`:228-236`), not entailment.
- `detect_conflicts()` simply picks the most frequent canonical output (`leo/agents/engine.py:172-189`) — no judge, rubric, or confidence weighting.
- Evals: 123 unit tests and one 3-record retrieval fixture (`tests/test_memory.py:348-381`); no pass^k, no CI threshold gate, no golden task suite, no measurement of cost savings or cache hit rate.

---

## 4. Ranked upgrade specification (12 items)

Each item states **what to measure**, **how**, and an **acceptance bar**. All baselines must be recorded before the change with the same fixture and seed; report both absolute and relative numbers with date and commit.

### P0 — 1. Prompt caching with stable prefixes and cache-token accounting
**Build:** add `cache_control` support to `ModelRequest`/provider bodies (`tools` → `system` → `messages`, last static block, 1h before 5m), a stable prompt assembler (static rules/tools first, volatile data after breakpoint), and `cache_read_input_tokens`/`cache_creation_input_tokens` in `Usage` + `model_usage.metadata`.
**Measure:** 20-turn simulated conversation fixture (10 runs, fixed seed) against a cache-aware mock/HTTP recorder and, if a live gateway becomes available, one provider. Metrics: cache hit rate = reads/(reads+creations); cached-input share = reads/total input; cost per turn; prefix stability (bytes equal at breakpoint across turns); provider payload token arithmetic.
**Accept:** hit rate ≥0.80 after turn 5; cached-input share ≥0.60 over turns 5–20; ≥30% measured input-cost reduction vs no-cache baseline on the same fixture; `cache_read + cache_creation + input == provider-reported total` in 100% of 50 recorded responses; ≤4 breakpoints and correct model-specific minimum thresholds; no answer diff on 10 golden prompts. Reference S1, S25, LEO gap §3.1.

### P0 — 2. Context engine with provider-aware budget and real compaction in chat
**Build:** `ContextEngine` protocol (`ingest`, `assemble`, `compact`, `afterTurn`), provider-reported usage anchor + fingerprint, ratio trigger (e.g. 0.50, gateway-style 0.85 backstop), protected tail (20 messages / ratio 0.20), preserve decisions/bugs/identifiers, repair tool_call/tool_result pairing, and pass `provider=` from `ask()`.
**Measure:** 30 synthetic long conversations (20–200 turns) with planted continuity facts and tool-result bursts; compare estimate-only vs anchor-based triggers. Metrics: provider-reported `input_tokens ≤ context_window` in 100% of calls; post-compaction continuity-fact recall ≥0.90 on labelled facts; tool-pair invariant 100%; compactions per 100 turns ≤5 (anti-thrash); messages injected per turn ≤1 memory block (fix double injection).
**Accept:** all five thresholds met; `tokens_reclaimed` logged with method and estimate/real label every compaction; zero calls exceed window; no duplicate memory block. Reference S2, S17, S23, S26.

### P0 — 3. Structured output validation and refusal handling
**Build:** normalize caller schema to per-provider shapes; run the existing `_validate_value` validator (`leo/tools/registry.py:63-153`) on parsed output; add `refusal`/`incomplete`/`max_tokens` states to `ModelResponse`; make router validator schema-aware and fail-closed; do not escalate on locally repairable parse errors.
**Measure:** fixture of 100 cases (valid/malformed/missing fields/wrong types/refusal/truncated/embedded prose JSON) across mocked Anthropic and OpenAI transports.
**Accept:** 100% of completed non-refusal outputs are schema-valid or rejected; 0 uncaught parse exceptions; 100% of refusals surfaced as distinct status; business-rule violations (e.g. `minimum`) caught locally in 100% of 20 cases; escalation attempts ≤ bound. Reference S4, S6.

### P0 — 4. Agent capability scoping and bounded runs
**Build:** filter `tools` per `AgentSpec.tool_permissions`; add per-request `max_tokens`, `deadline_ms`, `max_children`, and roll-up token/cost into `AgentResult.metadata`; persist shared artifacts to SQLite/event history keyed by correlation ID.
**Measure:** 12 agents × 10 forbidden-tool attempts = 120 probe cases; 100 delegated runs with injected slow/failing workers; restart test with shared state.
**Accept:** 0 forbidden tools executed; 100% of runs record tokens/cost/latency and stop within deadline ±5%; depth and child limits enforced in 100% of nested probes; shared artifacts recovered after simulated restart; 0 automatic merges. Reference S2, S20, S28.

### P1 — 5. Retrieval quality: hybrid + calibrated ranking
**Build:** pluggable real embedding function with persisted vectors; BM25 (proper length normalization) fused via RRF or learned weights; calibrated confidence/recency/link weights; optional cross-encoder reranker for top-20.
**Measure:** build a held-out fixture of **≥50 labelled queries** over ≥300 memories covering lexical, paraphrase, temporal, and negative cases; report P@3/P@5, recall@5, MRR, nDCG@5 and p95 latency; ablate each score component.
**Accept:** recall@5 ≥0.85, MRR ≥0.75, P@3 ≥0.70, p95 <200 ms at 10k memories, and ≥10% relative improvement over current hash baseline; no regression on the existing 3-record test. Reference S10, S13, S27.

### P1 — 6. Memory write pipeline: extract → typed store → prefetch → flush
**Build:** post-turn extraction into episodic/semantic/procedural with source and confidence; `prefetch(query)` before model calls; pre-compaction flush hook with durable checkpoint; consolidation promoted to a scheduled job; TTL/importance policy per category.
**Measure:** 30 seeded multi-session conversations with 10 planted facts each; replay with/without flush. Metrics: fact retention at session boundary, injected-item precision (labelled), duplicate rate after consolidation, prefetch p95 latency, cost per extracted memory.
**Accept:** ≥95% planted-fact retention; ≥0.80 injected-item precision; duplicate rate ≤5% after consolidation; prefetch p95 <50 ms locally; 0 untrusted-content-as-fact promotions. Reference S8, S9, S11, S18, S27.

### P1 — 7. Temporal memory and contradiction resolution
**Build:** `valid_from`/`valid_to`, explicit supersedes chains, update/abstention policy; retrieval returns current + historical with temporal labels and citations.
**Measure:** 100 LongMemEval-style questions (temporal reasoning, knowledge updates, abstention) plus 20 injected contradiction cases; score with exact-match + judge rubric.
**Accept:** ≥80% accuracy on temporal/update questions; ≥90% correct abstentions; 0 stale facts presented as current; every answer cites memory IDs. Reference S10, S12.

### P1 — 8. Routing v2 driven by labelled tasks and quality parity
**Build:** capability/tier requirements in `_rank`; use `request.model` and `AgentSpec.model_preferences` as *constraints* (not literal model IDs); fixed strong/weak tiers; calibrated router; bounded escalation with cost/deadline cap; preserve cache prefix across routing decisions.
**Measure:** 100 labelled tasks (10 categories) with gold provider tier; 200 held-out prompts with quality judges; 3 parallel runs for variance.
**Accept:** top-1 tier accuracy ≥0.85; quality within 2% of always-strong on held-out judge score; cost reduction ≥25% vs always-strong; escalation attempts ≤`limit+1` and total wall-clock ≤2× single-call p95; cache hit rate not degraded >5% vs single-provider baseline. Reference S3, S14, S5.

### P1 — 9. Accounting accuracy and honest cost reporting
**Build:** parse Anthropic cache fields and OpenAI `prompt_tokens_details`; record `estimated` per field; expose a reconciliation report separating measured from estimated; fix double-count of tool schemas.
**Measure:** 50 mocked payloads with known usage and 10 recorded real payloads when available; reconcile `model_usage` rows against payloads and a tariff table.
**Accept:** 100% exact field reconciliation; cost error ≤1% for measured rows; estimated rows never included in "savings" claims; reconciliation command outputs JSON with counts of measured vs estimated. Reference S1, S23 (Hermes' "not billing evidence" caveat).

### P2 — 10. Independent verification with calibrated verdicts
**Build:** structured verdict schema, rubric-based reviewer prompt, evidence-to-claim entailment (not token overlap), reviewer model distinct from producer, confidence score and calibration report.
**Measure:** 50 true/false claim pairs, 100 verdict phrasings, 20 adversarial ("not incorrect", "APPROVE with caveats") cases; compute false-accept, true-reject, parse accuracy.
**Accept:** false-accept ≤5%, true-reject ≤10%, parse accuracy ≥95%, calibration curve with Brier score reported; 100% of runs log reviewer identity and rubric scores. Reference S3, S7, S15.

### P2 — 11. Agent evaluation harness with reliability metrics
**Build:** versioned fixture packs + `python -m leo.evals run --suite all --seed N` emitting JSON: retrieval P@5/MRR, routing accuracy, compaction continuity, verification false-accept, task success, **pass^k**, token/cost/cache report; CI gate with thresholds; compare against stored baseline.
**Measure:** run 3 seeds before/after each change; report mean ± sd and delta with significance note.
**Accept:** command exists, output schema documented, all six suites populate non-zero results, CI fails when any P0 threshold regresses >5%, and reports are persisted with commit/date. Reference S7, S15, S16, S3.

### P2 — 12. Session search and evidence-backed doc claims
**Build:** SQLite FTS5 over `messages` with lineage dedup ±5 context; rewrite doc claims to link each efficiency/intelligence statement to a metric definition, fixture, and report date.
**Measure:** 100 session-search queries with labelled target message IDs; audit every claim in `LEO_PROGRESS.md`, `LEO_TEST_REPORT.md`, `LEO_INTEGRATIONS.md`, `docs/CAPABILITY_INVENTORY.md`.
**Accept:** recall@5 ≥0.90, p95 <100 ms; 0 claims using mock-provider tests as evidence of real-world quality; 0 unlinked quantitative claims. Reference S19, S7.

---

## 5. Unmeasured or overstated claims in LEO docs

| Doc:line | Claim | Assessment |
|---|---|---|
| `LEO_PROGRESS.md:5` | "All 14 phases delivered to the extent safely achievable locally." | True only at code/test level; live model, integrations, VPS blocked (same file `:31-35`). Should say "implemented and mock-tested". |
| `LEO_PROGRESS.md:17` | "7-category memory with hybrid search" | Categories are labels only (§3.2); "hybrid" = lexical + hash trigram, not trained semantic (LEO_KNOWN_ISSUES.md:20-24). Mechanism tested, quality unmeasured. |
| `LEO_PROGRESS.md:27` | "measured model usage" | Usage rows are recorded, but all tests use `MockProvider` with estimated tokens; no provider-reported token accuracy or cost/savings measurement. |
| `LEO_TEST_REPORT.md:64` | "Model routing between configured models \| Tested" | Tests assert cost/preference selection among mock providers (`tests/test_models.py:370-388`); classification does not affect routing and no quality metric exists. |
| `LEO_TEST_REPORT.md:67` | "125 passed" | Unit tests, not agent-quality evidence; no pass^k or held-out task suite. |
| `LEO_INTEGRATIONS.md:12` | "hybrid search, links, versions, dedup, retention" | Implemented; retrieval precision/recall not measured outside a 3-record test. |
| `docs/CAPABILITY_INVENTORY.md:77-78` | "Hybrid keyword + embedding memory search", "Context optimization / token budgeting, model routing/escalation \| Tested" | Embedding is hashing; token budgeting is chars/4; compaction not wired into `ask()`; routing is cost sort. Overstated as capability parity. |
| `docs/CAPABILITY_INVENTORY.md:80` | "Subagent delegation, isolation, verification \| Tested" | True for mechanics; agent tool permissions unenforced, no handle/cancel/steer, no durability of shared state. |
| `LEO_ARCHITECTURE.md:13` | "working/episodic/semantic/procedural/personal/project/org memory" | Types exist as enum values; no differentiated behavior. |
| `LEO_ARCHITECTURE.md:16` | "12 specialists, delegation, isolation, verification" | Mechanically true; scoping/budget gaps above. |
| `Master_plam.md:6`, `:82` | "significant improvements in intelligence, efficiency…", "MAXIMUM INTELLIGENCE. MINIMUM WASTE." | Aspirational slogans. The same document requires "Never claim token savings without measurement" (`:344`) and "Track performance before and after optimizations" (`:825`) — currently unmet. |
| `docs/RESEARCH_HERMES_OPENCLAW.md:32,38` | Memory "flushed before compression"; "Anthropic cache breakpoints" | Nuance: built-in Hermes memory flush is at session boundaries; `on_pre_compress` is a provider hook with optional v2 checkpoint. Cache behavior verified in source, but the doc omits char limits and cache-stability constraints. |

---

## 6. Recommended order of execution

1. **Measure first, in one day:** add the accounting fixture (#9) and eval harness skeleton (#11) so every later change has a baseline.
2. **Fix the two cheapest high-impact defects:** pass `provider` to `optimize()` and remove duplicate memory injection (#2); map tool permissions and model preferences correctly (#4).
3. **Land caching + structured output** (#1, #3) — both are well-specified by provider docs and directly measurable.
4. **Then memory quality** (#5–#7) and **routing/verification** (#8, #10), each gated by the harness.
5. **Update docs** (#12) only after reports exist; link every claim to measured evidence.

The overall principle from S5 remains the right one: "start with simple prompts, optimize them with comprehensive evaluation", and add complexity "only when it demonstrably improves outcomes". LEO has the scaffolding for demonstration; it now needs the measurements.
