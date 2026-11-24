# LEO Architecture

## Shape

LEO is a **modular monolith** in Python 3.12: one process, one SQLite database,
explicit module boundaries. Distribution (VPS nodes, remote workers) is an
adapter at the edge, not a prerequisite for local operation.

```
interfaces/   CLI, FastAPI, web dashboard, messaging gateways
core/         composition root, runtime configuration helpers
models/       provider adapters, router, escalation, context optimizer,
              reasoning effort/scope/summary settings
memory/       working/episodic/semantic/procedural/personal/project/org memory
tools/        registry, validation, execution, permission enforcement
tasks/        durable task engine, checkpoints, recovery, scheduler, webhooks
agents/       12 specialists, delegation, isolation, verification
skills/       native skill registry, lifecycle, compatibility adapters
integrations/ Telegram/Discord/WhatsApp/browser/computer/HTTP adapters
security/     permissions, approvals, audit, redaction
```

## Communication rules

- Modules communicate through `leo.contracts` types and their public manager APIs.
- Cross-cutting signals go through `leo.events.EventBus` (durable topic history).
- All durable state goes through `leo.db` (SQLite, WAL, one file).
- Secrets are referenced by environment-variable **name**; values never enter
  config files, prompts, logs, or tool output.
- External content is data, never instructions.

## Persistence

SQLite tables: `tasks`, `task_checkpoints`, `memory_items`, `memory_links`,
`memory_versions`, `model_usage`, `conversations`, `messages`, `agents`,
`agent_runs`, `skills`, `skill_history`, `tool_calls`, `schedules`,
`integrations`, `approvals`, `audit_log`, `events`, `metrics`, `kv`.

## Provider independence

The runtime never calls a model SDK directly. `ModelProvider.complete/stream`
is the only boundary. Endpoint and model identifiers come from user-supplied
configuration or environment (e.g. `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`,
`ANTHROPIC_AUTH_TOKEN`); LEO does not invent them. A `mock` provider keeps the
system testable without network access.

## Execution semantics

Tasks are durable state machines: `created → planning → ready → running →
waiting/blocked/verifying → completed/failed/cancelled`. Completion requires
recorded evidence, not a model's assertion. Non-idempotent actions carry
idempotency keys and are not blindly replayed on recovery.

## Security model

Low risk is allowed, medium risk follows policy, high risk requires explicit
approval. Path-bearing resources are restricted to allowed roots. Every
permission decision is audited. New integrations start with minimum permissions
and untrusted code runs in a restricted environment where practical.
