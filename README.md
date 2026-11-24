# LEO — Autonomous Agent OS

A modular, provider-independent personal AI platform: durable tasks, layered
memory, multi-agent delegation, native tools/skills, integrations and a local
CLI/API/dashboard.

## Quick start (local-only)

```bash
git clone https://github.com/shoaibyazdani/leo-agent.git
cd leo-agent
sh install.sh
leo
```

`sh install.sh` creates the Python 3.12 virtualenv with `uv`, installs LEO,
initializes local state and puts a `leo` launcher on your PATH. After that,
bare `leo` opens the terminal chat.

Manual install (already have a checkout):

```bash
uv venv --python 3.12 .venv
uv pip install -e .
uv run python -m leo.interfaces.cli chat
```

`leo chat` (alias `leo tui`) streams replies (and concise reasoning summaries
when enabled), shows tool rows and approval gates live, and accepts `/help`, `/new`, `/model`,
`/reasoning` (interactive effort/scope/display menu), `/tools`, `/agents`,
`/approvals`, `/approve <id>`, `/deny <id>`. Resume a session with
`leo chat --conversation <id>`. `leo serve` starts the dashboard and API on
<http://127.0.0.1:8765>; `leo --help` lists every command.

Add providers without editing YAML by hand:

```bash
leo model presets
leo model add openrouter --preset openrouter --model openrouter/auto \
    --api-key-env OPENROUTER_API_KEY --default
```

Only the environment-variable NAME is stored; the key itself stays in your
environment (for the launcher, put `export OPENROUTER_API_KEY=...` lines in
`~/.config/leo/env`).

## WhatsApp device linking and chat (opt-in)

The official WhatsApp Cloud API path needs Meta business credentials. To use a
**personal** WhatsApp account instead:

```bash
# one-time: install the pinned Node bridge dependencies
(cd leo/integrations/whatsapp_bridge && npm ci)

leo whatsapp status                       # readiness + saved-credential check
leo whatsapp link                         # QR page; scan from Linked devices
leo whatsapp serve --allow +15551234567   # answer that sender (repeatable)
```

`link` serves the QR on a short-lived `127.0.0.1` page; credentials stay in
`.leo/whatsapp/default/` (git-ignored) and are never printed. `serve` runs until
Ctrl+C, gives each allowed sender a durable LEO conversation, and sends the
agent's reply back through the same linked session. It is fail closed: with no
`--allow` and no explicit `--allow-all`, it answers nobody. Messages from the
linked account itself are ignored (no reply loops), and groups need `--groups`.

Modern WhatsApp may address a contact by a privacy **LID** instead of their
phone number. `serve` matches the phone-number alternative when Baileys supplies
it; if the log shows `Ignored message from <digits>`, add those digits as a
second `--allow` (repeatable). Replies need the same model-provider environment
as `leo chat`; `LEO_OPENCODE_SESSION` is a non-secret per-instance ID, not a key.

This uses the unofficial WhatsApp Web protocol (Baileys), not the Cloud API,
and carries WhatsApp ToS/account-ban risk. Inbound text is untrusted content;
high-risk tool calls still pass through LEO's approval gate.

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

When `ANTHROPIC_BASE_URL` names LEO's OpenCode Go inference gateway
(`https://opencode.ai/inference/go/anthropic`) and `ANTHROPIC_AUTH_TOKEN` is
set, LEO automatically registers an `opencode-go` provider using those values,
including the gateway's required `x-opencode-session` header (from
`LEO_OPENCODE_SESSION`) and User-Agent. A token alone injects nothing, and the
IDE's Anthropic endpoint is deliberately ignored: LEO never adopts a shared
shell's endpoint as its own. The model ID is whatever the gateway expects; this
checkout's `leo.yaml` pins `opencode-go` to `deepseek-v4.1-flash` and also
lists the other 30 models returned by
`GET /inference/go/anthropic/v1/models`, each as its own provider named by the
exact model id. Change `models.default` to any of those keys — or pick one in
the dashboard's Model & providers dialog — to switch models. Per-model context
windows (256k–1.05M), vision flags and pricing come from models.dev's
`opencode-go` metadata; `escalation_limit: 0` keeps a failed turn from silently
moving to another model. A provider name is a local routing key only — never put
a model alias in the provider slot or a different vendor's model ID in `model`,
or traffic goes to the wrong backend. `mock` provider is always available for
offline use and tests. Never put a secret value in config, prompts, logs or
tool output.

## Agent persona

The repository ships **empty templates only** in `agent/templates/`. LEO loads
`SOUL.md` (character), `USER.md` (operator profile) and `MEMORY.md` (durable
facts) from `$LEO_HOME` or `~/.config/leo`, in that order and with
8,000 / 1,375 / 2,200 character caps. Copy the templates into the runtime
location and edit the local copies:

```sh
mkdir -p ~/.config/leo
cp agent/templates/SOUL.md agent/templates/USER.md agent/templates/MEMORY.md ~/.config/leo/
```

Never commit a personal persona. `AGENTS.md` and `IDENTITY.md` are optional
reference files and are not injected into the prompt. The loader is read-only;
durable memory writes stay in the SQLite memory module.

## Architecture and status

- [LEO_ARCHITECTURE.md](LEO_ARCHITECTURE.md) — module boundaries and state model
- [STATUS.md](STATUS.md) — current status and limits
- [LEO_DECISIONS.md](LEO_DECISIONS.md) — engineering decisions
- [LEO_INTEGRATIONS.md](LEO_INTEGRATIONS.md) — real vs blocked integrations; [channel matrix](LEO_INTEGRATIONS.md#channel-matrix-2026-11-18) (22 auto-registered adapters appear in Connected Surfaces; live delivery blocked until credentials are supplied)
- [LEO_SECURITY.md](LEO_SECURITY.md) — permissions, approvals, audit
- [docs/REASONING.md](docs/REASONING.md) — `/reasoning` levels, scopes, provider mapping and display privacy
- [docs/TESTING.md](docs/TESTING.md) — how to run the suite
- [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md) — honest limitations
- [NOTICE](NOTICE) — upstream license attribution

## Tests

```bash
uv run pytest -q
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

### Shell policy

The `shell` tool runs through the platform shell for agents with the
`development` scope (Coding, QA, Computer Operator) and for the unscoped chat
loop. It follows a smart approval model:

```yaml
security:
  shell_enabled: true
  shell_mode: smart        # smart | manual | off
  deny_commands:           # always deny, before approvals
    - "git push --force*"
```

- `smart` (default): read-only and test/lint commands run; risky or unknown
  commands request approval; hardline patterns (`rm -rf /`, fork bombs, disk
  wipes, remote downloads piped to a shell) always deny.
- `manual`: every shell command requests approval.
- `off`: risky commands run without prompts (yolo); the hardline floor and
  `deny_commands` still deny.

Each shell approval is consumed once for that exact command. The shell is **not
a sandbox**: commands run with LEO's OS privileges, as a local terminal
does. Use `deny_commands`, approvals, or a container for untrusted work. The
older `terminal` tool remains the no-shell, allowlisted interface.

## Design system

The dashboard uses the **Quiet Depth** language: ambient midnight gradients, a
floating glass capsule for navigation, softly tinted panels instead of hard
cards, a lavender/ice accent, and system-first typography. It keeps the
Night Instrument token contract and layers the new look in
`static/ui/quiet-depth.css` (loaded last), so every view and behaviour hook is
preserved. Base tokens live in `static/ui/theme.css`, reusable components in
`static/ui/components.css`, and behaviour in `static/ui/app.js`. All eleven
views share these files; there are no one-off colors or styles, and no network
requests at runtime. The original system brief remains in
[docs/ui-design.md](docs/ui-design.md) for history. Before/after evidence:
[docs/gallery/](docs/gallery/).

The conversation view streams replies over `POST /chat/stream` (SSE via
`fetch`), shows live status and tool-call rows, supports Stop/cancel, and
accepts images and files through `POST /attachments`. The long-conversation
observer harness is documented in [scripts/README.md](scripts/README.md).
