# LEO Progress

## Status note (2026-11-19)

This file contains **current code facts** and **historical run notes**. Treat the
run notes as history, not as current evidence: no stress harness is committed in
this repository, the referenced `agent_reports/` directory does not exist, and
the 1000-scenario/999-scenario narratives cannot be reproduced from this tree.
Do not quote their numbers as today's verification. The current suite count must
be measured with `python -m pytest -o addopts='' -q`; `leo_state.json`
records an older snapshot (689 passed) and is not authoritative.

Current, code-backed counts (measured from the live `LeoApp.initialize()`
composition root on 2026-11-20):

- **95 registered tools** — 95 unique model-callable names in the live
  composition root: shipped toolsets (including 9 browser/computer operator
  tools; the operator-only `json_query` is filtered out) plus 9 builtins. The
  roster contract lives in `tests/test_tool_agent_integration.py`. Earlier
  "18 tools", "90 tools" and "93 tools" notes predate the expansion and are
  stale.
- **35 available slash commands** — the live registry holds 127 registered
  slash commands; 35 are currently available and the other 92 carry explicit
  unavailability reasons. Counts come from `app.commands` in the composition
  root.
- **24 specialist agents** — the live agent registry seeds 24 specialists;
  `tests/test_agents.py` asserts `>= 24`.
- **At least 12 native skills** — `tests/test_skills.py` asserts `>= 12`.

Conversation export is available over HTTP at
`POST /conversations/{id}/export?format=md|json|txt&redact=true`; `GET` on the
same path is a read-only manifest that lists existing files and never writes.
Both routes share the bounded, redacted, path-confined writer used by
`conversation_export` and `/export`; the POST response reports the real path
and byte count (`tests/test_api.py`). The writer serializes target selection,
aggregate budget checks and creation under one lock, and creates through a
verified exports-root directory descriptor so a root swap cannot redirect the
file.

`GET /agent-activity` and `GET /agent-activity/summary` read the durable
projection without side effects for cross-site/unauthenticated browser reads;
the transcript sync (which parses local files and writes SQLite) runs only
with an API token, an explicit `X-Leo-Refresh: 1` header, or a same-origin
browser request with an allowed Origin. This CSRF guard is implemented in
`_allow_activity_refresh` in `leo/interfaces/api.py`.

## Milestone scope (2026-11-18)

Phases 0-14 produced working local implementations for the subsystems listed
under "Verified milestones". This is **not** a blanket claim that every
upstream-equivalent capability is complete or verified: real gaps remain and
are recorded in [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md) and
`docs/LEO_GAP_INVENTORY_PART1.md` / `PART2.md`. The earlier "all 14 phases
delivered" wording was too broad and should not be read as completion.

At that milestone the offline suite reported 689 passed, 8 skipped — a historical
snapshot (superseded by the larger current roster).

**Deep-research hardening round (2026-11-18):** ten parallel agents audited
Hermes/OpenClaw source and runtimes and LEO's code/security/skills. Three P0
security defects, the task evidence/cancellation gaps, and the SKILL.md import
semantics were fixed with regression tests. See
the upstream research notes
Remaining atomicity/replay/tool-scoping gaps were recorded in
[LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md); later rounds fixed several of them.

**Conversation module port (2026-11-18):** the basic chat view was replaced by
a working Hermes-structured conversation module: session switcher sidebar,
streaming markdown thread with tool and approval cards, inline approval
resolution, attachment paste/drop, model/turn status side panel. New API:
`GET /conversations`, enriched `GET /conversations/{id}`, `DELETE /conversations/{id}`
(audit-logged). Files: `leo/interfaces/static/ui/conversation/{chat.css,chat.js,markdown.js}`,
`index.html`, `app.js`, `api.py`. Verified then: full test suite; browser E2E with a
mock-only config (stream, markdown render, session select/delete, tool/approval
events, no console errors). **A running server must be restarted to load the
new API routes**; the UI assets refresh with the page.

**Real-model production pass (2026-11-19, historical):** the live server ran
with the gateway token sourced from the host's local app environment (never
stored in LEO files), with `x-opencode-session` supplied through
`LEO_OPENCODE_SESSION` as configured in `leo.yaml`. Conversation used
`hy4-flash / deepseek-v4.1-flash` with mock as the automatic fallback. Upgrades:
production system prompt with explicit tool-use and untrusted-data rules shared
by `ask`/`ask_stream` (`leo/core/app.py::_conversation_system_prompt`); fixed
`tool.finished` incorrectly reporting `success=false` because
`ToolResult.error: null` was treated as an error — regression test in
`tests/test_chat_tools.py`. Reported then: real calculator and current_time
tool calls streamed through the conversation UI with green success cards and
real (non-estimated) token usage. This is a record of that pass, **not a fresh
live verification performed for this document update**.

## Verified milestones (historical implementation record)

1. **Phase 0 — environment discovery**: workspace was empty except the plan;
   Python 3.12 venv via `uv`, Node 24, git, SQLite available; Docker absent;
   env-sourced provider configuration recorded without secrets.
2. **Phase 1 — foundation**: config, redaction logging, SQLite schema, event
   bus, permissions/approvals/audit, composition root, CLI, API, dashboard.
3. **Phases 2–6 — engines**: provider adapters + router + context optimizer;
   tool registry/executor + built-ins; 7-category memory with hybrid search;
   durable tasks/checkpoints/recovery/scheduler/webhooks; specialist agent
   engine with isolation and verification; native skills + compatibility
   layer + integration factory.
4. **Phases 7–12 — integration & interfaces**: official Hermes/OpenClaw
   research + capability inventory; SKILL.md adapters; messaging gateway and
   Telegram/Discord/WhatsApp adapters (mock-tested); browser operator (fake
   backend + real Chromium local-page check); macOS computer operator
   abstraction; CLI/API/dashboard/Electron shell source; authenticated scoped
   local/VPS node bridge.
5. **Phases 13–14 — optimization & hardening**: measured model usage,
   self-evaluation and gated improvement proposals, security audit trail,
   full test suite, docs, git milestone commit.

## Blocked / conditional (documented, not hidden)

- **Live model gateway — conditional, not universally blocked**: available
  when the host supplies `ANTHROPIC_AUTH_TOKEN` and `LEO_OPENCODE_SESSION`;
  degrades to mock otherwise. Historical real pass above; no fresh live
  verification here. See [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md) and
  [LEO_INTEGRATIONS.md](LEO_INTEGRATIONS.md).
- **Live messaging integrations**: BLOCKED (no platform credentials; adapters
  mock-tested only).
- **Remote VPS deployment / node**: BLOCKED (no host, token or TLS endpoint).

## Historical verification notes (2026-11-19 — not reproducible from this tree)

The following narratives were reported during the 2026-11-19 runs. **No harness
that produces these numbers is committed here, and the `agent_reports/`
directory they reference is absent.** They are retained as history only; they
are not current evidence and were not re-run for this update.

**1000-scenario live stress (historical).** A concurrent harness was reported
against the live instance: read APIs 220, tool executions 300, attachments 80,
security probes 200, real-model SSE 60, conversation CRUD 100, cancel 40. First
run 964/1000 — 36 unknown tool/skill responses returned HTTP 200; fixed to 404
in `api.py`; second run reported **1000/1000 passed** (p95 73 s). Twenty
subagents with disjoint ownership reported fixes across tools, permissions,
DB concurrency, memory, tasks, agents, providers, router, attachments, gateway,
browser/computer operators, skills, evaluation, events, config, CLI and
conversation CSS/JS. All of this is unreproducible from the committed tree.

**Integration claim (historical).** A note claimed "full suite green; 1000/1000
live scenarios green on the definitive code path; live server on port 8891
healthy with `hy4-flash` primary and 18 registered tools" (historical note).
The 18-tool figure is stale; the current registry has **95 tools** in the live
composition root. The stress figure is historical and unverifiable here.

**Browser + computer use as conversation tools (historical).** `leo/tools/operator_tools.py`
registers 5 browser tools (`browser_navigate`, `browser_extract_text`,
`browser_screenshot`, `browser_click`, `browser_fill`) and 4 macOS computer
tools (`computer_notify`, `computer_screenshot`, `computer_clipboard_read`,
`computer_clipboard_write`) through the PermissionManager/audit/approval
pipeline, with a persistent per-actor browser operator; they are registered
from `LeoApp.initialize()`. Tests: `tests/test_operator_tools.py`. A reported
live E2E run had the model open `https://example.com`, extract page text, and
write/read a clipboard value; that run is historical/unreproducible here.
Private/loopback navigation is denied by SSRF policy; screenshots are confined
to the workspace. The browser operator starts its Playwright backend lazily on
the first browser tool call when Playwright/Chromium is available — see
`LEO_SECURITY.md` for the corrected trust statement.

**1000-case memory stress (historical).** A reported live harness covered add
200, search 200, list 100, cross-session 60, edge cases 200, lifecycle 100,
concurrency 140. A planted fact was said to be recalled in 59/60 new
conversations; final run reported 999/1000, the single miss being an empty
response from the model gateway. Two real API defects from that round were
fixed with regression tests in `tests/test_api.py` (invalid category/kind now
HTTP 400; empty memory content now HTTP 400). The stress run itself is
historical/unreproducible here. The note also said "18 tools, 12 skills" — the
tool figure is stale (95 now); the skills assertion is `>= 12`.

## Recovery checklist

1. Read this file, `LEO_BUILD_PLAN.md`, `LEO_KNOWN_ISSUES.md`.
2. Inspect `git status` and run `python -m pytest -o addopts='' -q`
   for a fresh count; do not reuse the numbers in the historical notes.
3. Read `leo_state.json` as a historical snapshot only (it still records the
   old gateway-blocked state and the stale 689-passed count).
4. Resume from the first missing evidence; never assume a previous task succeeded.
