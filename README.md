# LEO — Autonomous Agent OS

A modular, provider-independent personal AI platform: durable tasks, layered
memory, multi-agent delegation, native tools/skills, integrations and a local
CLI/API/dashboard.

## Quick start (local-only)

```bash
# From /Users/macbook/LEO
uv venv --python 3.12 .venv
uv pip install -e .
.venv/bin/python -m leo.interfaces.cli init
.venv/bin/python -m leo.interfaces.cli doctor
.venv/bin/python -m leo.interfaces.cli ask "What can you do?"
.venv/bin/python -m leo.interfaces.cli serve     # http://127.0.0.1:8765
```

Or use the installed entry point: `leo doctor`, `leo ask "..."`, `leo serve`.

## Model configuration

LEO does not hardcode endpoints, model identifiers or credentials. Create
`leo.yaml` (optional) or rely on environment configuration:

```yaml
models:
  default: opencode-go
  fallback: mock
  providers:
    my-provider:
      kind: anthropic-compatible   # or openai-compatible | mock
      base_url: https://provider.example/v1
      api_key_env: MY_API_KEY_ENV  # name of env var holding the secret
      model: exact-model-id-from-provider-docs
      # Declare only what the model really supports. `vision` routes image
      # attachments to this provider as image content blocks; without it an
      # image turn fails with "no enabled provider satisfies the request".
      capabilities: [text, streaming, tools, structured, usage, vision]
```

When `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` / `ANTHROPIC_AUTH_TOKEN` are set,
LEO automatically registers an `opencode-go` provider using those values. The
provider is the **OpenCode Go gateway** (the base URL); the model ID is whatever
the gateway expects (for this environment, `hy4-flash-free`). A provider name
is a local routing key only — never put a model alias in the provider slot or a
different vendor's model ID in `model`, or traffic goes to the wrong backend.
`mock` provider is always available for offline use and tests. Never put a
secret value in config, prompts, logs or tool output.

## Architecture and status

- [LEO_ARCHITECTURE.md](LEO_ARCHITECTURE.md) — module boundaries and state model
- [LEO_BUILD_PLAN.md](LEO_BUILD_PLAN.md) — phase ledger
- [LEO_PROGRESS.md](LEO_PROGRESS.md) — verified milestones
- [LEO_DECISIONS.md](LEO_DECISIONS.md) — engineering decisions
- [LEO_INTEGRATIONS.md](LEO_INTEGRATIONS.md) — real vs blocked integrations; [channel matrix](LEO_INTEGRATIONS.md#channel-matrix-2026-11-18) (22 auto-registered adapters appear in Connected Surfaces; live delivery blocked until credentials are supplied)
- [LEO_SECURITY.md](LEO_SECURITY.md) — permissions, approvals, audit
- [LEO_TEST_REPORT.md](LEO_TEST_REPORT.md) — test evidence
- [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md) — honest limitations
- [docs/CAPABILITY_INVENTORY.md](docs/CAPABILITY_INVENTORY.md) — Hermes/OpenClaw inventory
- [research/deep-2026-11-18/SYNTHESIS.md](research/deep-2026-11-18/SYNTHESIS.md) — deep source/runtime research, fixes and remaining gaps
- [NOTICE](NOTICE) — upstream license attribution
- [docs/GIT_PUSH_PLAN.md](docs/GIT_PUSH_PLAN.md) — pre-push secret audit and procedure

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Safety

The default server binds to `127.0.0.1`. High-risk actions require explicit
approval (`leo approval list` / `approve` / `deny`), and all permission decisions are
audited in the SQLite database at `.leo/data/leo.db`. Stale pending approvals can
be reported and expired without deleting them (`leo approval cleanup --older-than 24`,
dry-run by default; add `--apply` to expire). External content is
treated as untrusted data, never as instructions.

For a non-loopback bind, set `LEO_API_TOKEN` first: the CLI refuses an exposed
bind without it, and all interactive API routes then require
`Authorization: Bearer <token>` (the dashboard has a session-only token field).
Provider webhooks authenticate with their own HMAC signatures and fail closed
when verification secrets are absent.

## Design system

The dashboard uses the **Night Instrument** system documented in
[docs/ui-design.md](docs/ui-design.md): matte graphite panels, warm ivory type,
one signal-teal accent, and locally bundled Space Grotesk / IBM Plex fonts — no
network requests at runtime. Tokens live in `static/ui/theme.css`, reusable
components in `static/ui/components.css`, and behaviour in `static/ui/app.js`.
All eleven views share these files; there are no one-off colors or styles.
Before/after evidence: [docs/gallery/](docs/gallery/) and
[artifacts/screenshots/ui/](artifacts/screenshots/ui/).

The conversation view streams replies over `POST /chat/stream` (SSE via
`fetch`), shows live status and tool-call rows, supports Stop/cancel, and
accepts images and files through `POST /attachments`. See
[docs/CHAT_REALTIME_PLAN.md](docs/CHAT_REALTIME_PLAN.md) and the
[upstream research](research/deep-2026-11-18/realtime-chat-research.md).
