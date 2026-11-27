# LEO Status

Current, code-backed state of the project. History lives in git, not in a
narrative changelog.

## Working today

- Provider-independent agent platform: streaming chat in the dashboard and in
  the terminal (`leo chat`), durable tasks, layered memory, specialist agents,
  native tools and skills, approvals with audit, integrations adapters.
- Measured from the live `LeoApp.initialize()` composition root: 104
  registered tools on Linux (108 on macOS, which adds the four `computer_*`
  operator tools), 42 available slash commands (126 registered slash
  commands in total) and 24 specialist agents.
- Agent-managed skills: `skill_manage` creates documentation-only skills
  (no executable handler) with a fresh-read requirement, a 60-character
  index description, credential/bidi sanitization, audit and usage
  telemetry; native shipped skills are read-only to the agent.
- Post-turn review: after `learning.review.interval_turns` completed turns,
  LEO replays a bounded durable transcript through a small tool scope
  (`skill_manage`, `memory_propose`) and records one artifact plus one audit
  row. Harness and unfinished turns are skipped; a no-op is a successful
  outcome, and a write still needs a fresh `skill_read` of its target.
- Background skill writes are staged for review by default: `/skills pending`
  lists the exact payload and digest, `/skills diff <id>` shows it,
  `/skills approve <id>` replays it once through the same tool (operator
  principal only), and `/skills reject <id>` is terminal. `/refine [focus]`
  starts a review now. Interactive foreground writes remain direct.
- `/learn <path|URL|procedure>` authors a skill explicitly with read-only
  source tools; it writes through the same `skill_manage` validators and is
  not staged because the user asked for it.
- Skill curator: a deterministic, interval+idle-gated pass moves agent-authored
  skills `active -> stale -> archived`; archived skills leave the prompt index
  and remain restorable. Native/pinned skills are exempt, content is never
  edited or deleted, and `/skills pin|lock|archive|restore|curator` expose the
  operator controls. `/skills curator candidates` reports near-duplicate
  groups; it never merges them.
- Skill rollback: authored mutations snapshot their before-state (bounded to
  10 per skill in `kv`); `/skills snapshots <name>` lists them and
  `/skills rollback <id>` restores exactly one through the normal write path.
  Native skills are not snapshotted.
- LLM consolidation (opt-in, `skills.curator.consolidate`): deterministic
  near-duplicate groups are sent to the model for one strict-JSON decision;
  a valid merge becomes a staged multi-operation `read`/`manage`/`archive`
  plan. `/skills approve` replays it; nothing is rewritten automatically and
  native skills never enter a group.
- `/journey` (aliases `/learning`, `/memory-graph`) shows authored skills and
  memory notes on one oldest-first timeline.
- Durable conversation compaction: older turns are summarized once, the summary
  is stored in the `compactions` table, and later turns replay the summary plus
  a verbatim recent tail. `/compact [focus] [--preview|--clear]` runs it
  manually; automatic compaction triggers at a configurable share of the
  context budget. The transcript is never rewritten.
- Vision: image attachments route to providers that declare the `vision`
  capability and reach the model as image content blocks.
- Autonomous memory learning: notes derived from the operator's own durable
  history (explicit statements, bounded tool/run trajectories) are written with
  evidence hashes after the secret/injection sanitizer; `learning.auto_notes`
  in `leo.yaml` controls source kinds and categories, and setting
  `learning.auto_notes.enabled: false` restores a HIGH-risk approval for every
  note. Superseding an existing memory always requires an approval.
- `/reasoning`: an interactive Telegram/CLI menu for effort
  (`none`…`ultra`), global vs session scope, persisted global defaults and
  opt-in concise summary display. Effort is translated to each provider's own
  reasoning parameter, clamped downward when unsupported, and inherited by
  subagents. Raw chain-of-thought is never shown. See
  [docs/REASONING.md](docs/REASONING.md).
- Local-first security model: loopback bind by default, `LEO_API_TOKEN`
  required for any non-loopback bind, HIGH-risk actions gated by approvals.
- Proactive schedules: nine model tools, durable consent cards, native
  script/monitor adapters and unified delivery use the existing SQLite, task
  engine and permission authority. Model-created jobs without an answerable
  surface stay as `approval_required` proposals; direct operator scheduling is
  unasked and audited. Details and capability matrix:
  [docs/SCHEDULES.md](docs/SCHEDULES.md).

## Limits

- Live messaging delivery needs platform credentials; the adapters are
  mock-tested until then.
- Remote node / VPS deployment needs a host, token and TLS endpoint.
- Scheduled scripts are opt-in operator-provisioned executables, not a sandbox;
  URL monitors are public-address HTTP(S) only. Live Telegram/gateway smoke, a
  production-copy migration drill, backup/restore and PTY/TUI verification are
  not claimed by the local suite; see the release receipt in
  `scripts/release/evidence/receipt.json`.
- See [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md) for the itemized list.

## Testing

See [docs/TESTING.md](docs/TESTING.md).
