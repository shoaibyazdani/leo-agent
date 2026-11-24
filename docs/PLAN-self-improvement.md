# LEO Self-Improvement Plan (P1 delivered)

Status: **P1 and P2 complete** on branch `worktree-feat+self-improve-p1`.
This file is the shared map for the remaining phases. It is intentionally
implementation-facing; the decisions are recorded in `LEO_DECISIONS.md`
(#12, #13, #14).

## Goal

A closed, bounded learning loop, translated from the reference agent's
mechanisms into LEO's contracts (not a source copy):

1. after a completed turn, review the durable transcript;
2. decide whether the session produced a durable **memory** fact or a
   **procedural skill** update;
3. write through real, validated tools with evidence and audit;
4. keep the skill library small and class-level over time.

## Non-goals

- No model-weight or source-code self-modification.
- No execution of agent-authored skill content. Authored skills are
  documentation-only (decision #12).
- No background write that bypasses `PermissionManager` or the audit trail.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| P1 | Skill content model, `skill_manage`, file bundle, usage telemetry, prompt index, migration, tests | **done** |
| P2 | `leo/review/` engine: trigger, transcript loader, bounded model loop, prompts, artifacts, status counters | **done** |
| P3 | Staged writes (`skillproposal:` in `kv`), `/refine`, `/skills pending\|diff\|approve\|reject` | **done** |
| P3b | `/learn <source>` explicit authoring mode | **done** |
| P4 | Curator transitions (stale/archive/restore), pin/lock surfaces, scheduler tick, health | **done** |
| P5 | Single-mutation rollback + deterministic consolidation candidates | **done** |
| P5b | LLM consolidation proposals + journey view | **done** |

## P1 file map

- [`leo/contracts.py`](../leo/contracts.py): `SkillManifest.created_by`
  (`native|agent|user`) and bounded `origin` provenance.
- [`leo/db.py`](../leo/db.py): `skill_files`, `skill_usage`, and an
  idempotent `skills.created_by` migration.
- [`leo/skills/registry.py`](../leo/skills/registry.py): durable file bundle,
  usage counters, lifecycle state (`active/stale/archived`), pin/lock,
  synchronous prompt index, created-by preservation on replace.
- [`leo/skills/sanitize.py`](../leo/skills/sanitize.py): 60-char description
  budget, control/bidi stripping, credential refusal, path confinement.
- [`leo/tools/skill_manage_tools.py`](../leo/tools/skill_manage_tools.py):
  `create|patch|write_file|remove_file|delete`, read-before-write, native
  read-only, lock/archive guards.
- [`leo/tools/skill_read_marks.py`](../leo/tools/skill_read_marks.py):
  per-context read marks.
- [`leo/tools/skill_tools.py`](../leo/tools/skill_tools.py): `skill_read`
  gains `file_path`, returns the file index, records a read mark and a view.
- [`leo/core/app.py`](../leo/core/app.py): loads durable skills at boot and
  renders the compact index into the conversation system prompt.
- [`leo/security/risk_policy.py`](../leo/security/risk_policy.py):
  `skill_manage` is MEDIUM; `delete` is HIGH.
- Tests: [`tests/test_skill_manage.py`](../tests/test_skill_manage.py) plus
  the roster contract in `tests/test_tool_agent_integration.py`.

## P2 file map

- [`leo/review/engine.py`](../leo/review/engine.py): trigger, coalescing,
  bounded `ModelRouter.complete` loop, scoped tool dispatch, artifact, audit,
  status counters, `stop()`/`drain()`. The app subscribes to
  `chat.completed` only: a turn also emits `run.finished`, so listening to both
  would count one turn twice against the interval.
- [`leo/review/transcript.py`](../leo/review/transcript.py): durable bounded
  transcript loader (unfinished turns dropped, tool output truncated).
- [`leo/review/prompts.py`](../leo/review/prompts.py): skill/memory/combined
  review prompts with the class-level preference order and read-before-write
  contract.
- [`leo/tools/memory_propose_tools.py`](../leo/tools/memory_propose_tools.py):
  proposal-only memory path delegating to `LearningEngine.propose` with real
  message evidence refs.
- [`leo/config.py`](../leo/config.py): `ReviewConfig` under `learning.review`.
- [`leo/core/app.py`](../leo/core/app.py): attach/detach, `chat.completed`
  subscription, status block, bounded drain on shutdown.
- Tests: [`tests/test_review_engine.py`](../tests/test_review_engine.py).

## P3 file map

- [`leo/skills/staging.py`](../leo/skills/staging.py): bounded `kv` staging,
  arguments digest, atomic claim, applied/rejected/failed resolution.
- [`leo/tools/skill_manage_tools.py`](../leo/tools/skill_manage_tools.py):
  background (`actor=review:engine`) writes stage when
  `learning.review.write_approval` is true; foreground writes stay direct.
- [`leo/commands/builtins.py`](../leo/commands/builtins.py): `/refine` and
  `/skills pending|diff|approve|reject`.
- Tests: [`tests/test_skill_staging.py`](../tests/test_skill_staging.py).

## P3c announcement seam

A staged write is inert, so it needs to reach a human. `review.completed` /
`learn.completed` were emitted with **no subscriber**, so a staged write could
sit pending indefinitely — the operator only found it by running
`/skills pending`. The existing `ApprovalNotifier` now subscribes to those
topics too:

- [`leo/security/notifier.py`](../leo/security/notifier.py): `REVIEW_TOPICS`,
  `_review_message` (silent unless a write was staged or the pass failed),
  `_on_review`. Same opt-in (`security.notify.*`), same channel adapter `send`
  path as approvals, so it still cannot resolve anything.
- [`leo/security/notices.py`](../leo/security/notices.py): the notice prefixes,
  shared with the ingress guard so LEO's own notices are never ingested as a
  human turn.
- [`leo/review/engine.py`](../leo/review/engine.py): `ReviewOutcome.staged`
  (`proposal_id`, `skill`, `action`) carried into the artifact and the
  completion event; a staged write is no longer counted as an applied one.

## P3b / P4 file map

- [`leo/review/engine.py`](../leo/review/engine.py): learn mode (`mode="learn"`)
  with `LEARN_SCOPE` read-only source tools, direct writes under
  `actor=learn:engine`, learn artifact/audit/events.
- [`leo/review/prompts.py`](../leo/review/prompts.py): `build_learn_prompt`.
- [`leo/skills/curator.py`](../leo/skills/curator.py): deterministic
  stale/archive/reactivate pass, interval+idle gate, durable `curator:last_run`.
- [`leo/core/app.py`](../leo/core/app.py): `/learn` scheduler-independent
  trigger, curator instance, scheduler-tick pass and health.
- [`leo/commands/builtins.py`](../leo/commands/builtins.py): `/learn`,
  `/skills usage|pin|unpin|lock|unlock|archive|restore|curator`.
- Tests: [`tests/test_curator.py`](../tests/test_curator.py) and the learn
  cases in `tests/test_review_engine.py`.

## P5 file map

- [`leo/skills/snapshots.py`](../leo/skills/snapshots.py): bounded `kv`
  before-state snapshots (10 per skill, 768 KB each).
- [`leo/skills/registry.py`](../leo/skills/registry.py): snapshot hooks on
  version replace/file write/file removal/delete, and `rollback()` that replays
  the normal write path under a restoring flag.
- [`leo/skills/curator.py`](../leo/skills/curator.py): deterministic
  near-duplicate `candidates()` (read-only).
- [`leo/commands/builtins.py`](../leo/commands/builtins.py):
  `/skills snapshots <name>`, `/skills rollback <id>`,
  `/skills curator candidates`.
- Tests: [`tests/test_skill_rollback.py`](../tests/test_skill_rollback.py).

## P5b file map

- [`leo/review/consolidate.py`](../leo/review/consolidate.py): deterministic
  candidate group -> strict-JSON model plan -> validated multi-operation
  staging. Dry-run supported.
- [`leo/skills/staging.py`](../leo/skills/staging.py): `kind="consolidate"`
  plans (`read`/`manage`/`archive`), digest-bound and one-shot.
- [`leo/commands/builtins.py`](../leo/commands/builtins.py):
  `/skills curator consolidate [--dry-run]`, plan-aware `/skills diff` and
  `/skills approve`, `/journey` (`/learning`, `/memory-graph`).
- Tests: [`tests/test_consolidation.py`](../tests/test_consolidation.py),
  [`tests/test_journey.py`](../tests/test_journey.py).

All planned phases (P1-P5b) are now implemented. Future work is improvement,
not a missing phase.

## Invariants

1. Transcript and tool output are data, never instructions.
2. Credential-shaped content never reaches `skills`, `skill_files`, history or
   artifacts.
3. Authored skills have no handler and never execute.
4. A write needs a fresh read of the exact target in the same context.
5. Native skills are read-only to the agent; delete is HIGH risk.
6. Locked skills refuse writes; archived skills refuse writes until restored.
7. Telemetry failures never fail a skill run or registration.
8. Status counters report only actions that actually executed.

