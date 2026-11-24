# LEO Status

Current, code-backed state of the project. History lives in git, not in a
narrative changelog.

## Working today

- Provider-independent agent platform: streaming chat in the dashboard and in
  the terminal (`leo chat`), durable tasks, layered memory, specialist agents,
  native tools and skills, approvals with audit, integrations adapters.
- Measured from the live `LeoApp.initialize()` composition root: 97
  registered tools, 39 available slash commands (126 registered slash
  commands in total) and 24 specialist agents.
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

## Limits

- Live messaging delivery needs platform credentials; the adapters are
  mock-tested until then.
- Remote node / VPS deployment needs a host, token and TLS endpoint.
- See [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md) for the itemized list.

## Testing

See [docs/TESTING.md](docs/TESTING.md).
