# LEO Integrations — actual status (2026-11-18)

Statuses are evidence-based. "Mock-tested" means an automated test exercised the
adapter with an injected client; it does **not** mean a live service worked.

## Native capabilities (implemented and tested)

| Area | Implementation | Evidence |
|------|----------------|----------|
| Model providers | Anthropic-compatible + OpenAI-compatible HTTP adapters, mock provider, router, escalation, usage | `tests/test_models.py`, `tests/test_router_default.py` |
| Tools | 9 built-ins with schema validation, permissions, dry-run, redaction, audit | `tests/test_tools.py`, `tests/test_api.py` |
| Memory | 7 categories, 4 kinds, hybrid search, links, versions, dedup, retention | `tests/test_memory.py`, `tests/test_app.py` |
| Tasks | durable state machine, evidence gate, checkpoints, retries, recovery, scheduler, webhooks | `tests/test_tasks.py` |
| Agents | 12 specialists, delegation, parallel waves, isolation, verification | `tests/test_agents.py` |
| Skills | 12 native manifests (7 with real local handlers), registry, lifecycle | `tests/test_skills.py`, `tests/test_compat_skill_md.py` |
| Integration factory | discover → evaluate → generate → test → register → rollback, minimum permissions | `tests/test_skills.py` |
| Gateway | unified conversations, channel/session isolation, central fail-closed admission, dynamic inbound webhooks and poll routes, outbound fan-out | `tests/test_integrations.py`, `tests/test_channels_registry.py` |
| Browser operator | Playwright backend (optional), fake backend, URL/path policy | `tests/test_integrations.py`, `tests/test_browser_live.py` (live local page) |
| Computer operator | macOS AppleScript/screenshot/clipboard via argv subprocess, dry-run default | `tests/test_integrations.py` |
| Hybrid node | authenticated, scoped node API/client (health/tasks only), HTTPS enforcement | `tests/test_node.py` |
| Interfaces | CLI, FastAPI, local dashboard, Electron shell source | `tests/test_api.py`, `tests/test_desktop_shell.py`, CLI smoke run |

## Compatible adapters (implemented, not live-verified)

| Adapter | Origin format | What translation preserves | Evidence | Live status |
|---------|---------------|----------------------------|----------|-------------|
| Generic manifest | JSON/YAML tool manifest | name, description, schemas, permissions, dependencies; refuses execution primitives | `tests/test_skills.py` | N/A (declarative) |
| Hermes `SKILL.md` | `metadata.hermes` frontmatter | gates retained as inert `manifest.gates` (never permissions); body is advisory data; provenance preserved | `tests/test_compat_skill_md.py`; 186 real upstream files validated | METADATA IMPORT TESTED; execution not supported |
| OpenClaw `SKILL.md` | `metadata.openclaw.requires` frontmatter | bins/env/anyBins/config retained as gates; no fabricated self-dependencies or versions | `tests/test_compat_skill_md.py`; 140 real upstream files validated | METADATA IMPORT TESTED; execution not supported |
| Telegram webhook/send | Bot API | secret-header/HMAC verification, normalized messages, send via Bot API | `tests/test_integrations.py` | BLOCKED: `TELEGRAM_BOT_TOKEN` / `TELEGRAM_WEBHOOK_SECRET` not supplied |
| Discord webhook/interactions | Webhook + interaction signature | Ed25519 verifier hook (injectable), normalized messages, send | `tests/test_integrations.py` | BLOCKED: `DISCORD_WEBHOOK_URL` / `DISCORD_BOT_TOKEN` / public key not supplied; Ed25519 needs crypto backend |
| WhatsApp Cloud webhook/send | Cloud API | `X-Hub-Signature-256` verification, normalized messages, send | `tests/test_integrations.py` | BLOCKED: `WHATSAPP_CLOUD_TOKEN`, phone ID, app secret not supplied |

The three messaging rows above are the original core set, not the full matrix.
They are now part of 22 auto-registered channels (3 core + 19 new); see
[Channel matrix](#channel-matrix-2026-11-18) below. Every channel record
currently reports `status="blocked"`, `verification="mock-tested"` and
`metadata.live_verified=False` because no credentials are supplied.

Hermes Agent (MIT, Nous Research) and OpenClaw (MIT, OpenClaw Foundation) were
researched from official sources; no upstream source code was copied. See
[docs/RESEARCH_HERMES_OPENCLAW.md](docs/RESEARCH_HERMES_OPENCLAW.md) and
[docs/CAPABILITY_INVENTORY.md](docs/CAPABILITY_INVENTORY.md). Their plugin
systems are **not** interchangeable; LEO uses declarative adapters only.
Source-level and runtime evidence for the 2026-11-18 round:
deep research synthesis.

## Model provider note

The `opencode-go` provider is built from the real environment variables
(`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_AUTH_TOKEN` by name). One
earlier live attempt returned HTTP 400 because the gateway requires an
`x-opencode-session` header. `leo.yaml` now maps that non-secret header to the
`LEO_OPENCODE_SESSION` env var through `extra_header_env` (alongside a fixed
`User-Agent`), and supports non-secret `extra_headers` generally.

**Honest current status:** the live model gateway is **available when the host
environment provides the token (`ANTHROPIC_AUTH_TOKEN`) and the session value
(`LEO_OPENCODE_SESSION`)**; it **degrades to mock** (the configured `mock`
fallback) when they are absent. It is not a universal BLOCKED capability, and it is not a
claim that live verification was re-run here — `LEO_PROGRESS.md` records a real
production pass on a suitably configured host, while the offline `mock` provider
is what tests use by default. No endpoint, credential or model ID was invented.

## Channel matrix (2026-11-18)

22 channel adapters are auto-registered in `leo.integrations.channels`: 3
pre-existing core adapters (telegram, discord, whatsapp) plus 19 discovered from
`leo/integrations/channels/`. Any module in that package defining a
`MessagingAdapter` subclass with a concrete `channel` is registered
automatically; a module that fails to import becomes an `error` record instead
of a silent gap. Capability split: 17 webhook-capable, 4 poll-capable (matrix,
irc, ntfy, email), 1 send-only (a2a). `BROKEN_MODULES == {}` on a clean tree.

**Status remains honest: all 22 records report `status="blocked"`,
`verification="mock-tested"` and `metadata.live_verified=False`.** No credentials
are present, so live delivery is BLOCKED for every channel. Only environment
variable *names* are listed below — never values.

| Channel | Mode | Credential env vars (names only) | Verification env / inbound auth | Test evidence |
|---|---|---|---|---|
| telegram | webhook | TELEGRAM_BOT_TOKEN | TELEGRAM_WEBHOOK_SECRET — secret header or HMAC-SHA256 | `tests/test_integrations.py` |
| discord | webhook | DISCORD_BOT_TOKEN, DISCORD_WEBHOOK_URL | DISCORD_PUBLIC_KEY — Ed25519 (injectable verifier) | `tests/test_integrations.py` |
| whatsapp (Cloud API) | webhook | WHATSAPP_CLOUD_TOKEN, WHATSAPP_PHONE_NUMBER_ID | WHATSAPP_APP_SECRET — X-Hub-Signature-256 | `tests/test_integrations.py` |
| slack | webhook | SLACK_BOT_TOKEN | SLACK_SIGNING_SECRET — v0 HMAC-SHA256 + 5-min replay window | `tests/channels/test_slack.py` |
| mattermost | webhook | MATTERMOST_TOKEN, MATTERMOST_URL | MATTERMOST_OUTGOING_TOKEN — shared-token check | `tests/channels/test_mattermost.py` |
| line | webhook | LINE_CHANNEL_ACCESS_TOKEN | LINE_CHANNEL_SECRET — base64 HMAC-SHA256 over raw body | `tests/channels/test_line.py` |
| google_chat | webhook | GOOGLE_CHAT_WEBHOOK_URL | GOOGLE_CHAT_VERIFICATION_TOKEN — Bearer token | `tests/channels/test_google_chat.py` |
| msteams | webhook | TEAMS_INCOMING_WEBHOOK_URL | TEAMS_PUBLIC_KEY — injected JWT verifier only; fail-closed without one | `tests/channels/test_teams.py` |
| dingtalk | webhook | DINGTALK_WEBHOOK_URL, DINGTALK_SECRET | DINGTALK_CALLBACK_SECRET — timestamp HMAC | `tests/channels/test_dingtalk.py` |
| feishu | webhook | FEISHU_APP_ID, FEISHU_APP_SECRET | FEISHU_ENCRYPT_KEY — X-Lark-Signature | `tests/channels/test_feishu.py` |
| qqbot | webhook | QQBOT_APP_ID, QQBOT_APP_SECRET | QQBOT_PUBLIC_KEY — Ed25519 | `tests/channels/test_qqbot.py` |
| wecom | webhook | WECOM_CORP_ID, WECOM_SECRET | WECOM_CALLBACK_TOKEN, WECOM_ENCODING_AES_KEY — SHA1 signature + AES (optional `cryptography` backend) | `tests/channels/test_wecom.py` |
| weixin | webhook | WEIXIN_APP_ID, WEIXIN_APP_SECRET | WEIXIN_TOKEN, WEIXIN_ENCODING_AES_KEY — SHA1 signature + AES (same as wecom) | `tests/channels/test_weixin.py` |
| sms | webhook | TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER | TWILIO_AUTH_TOKEN — X-Twilio-Signature (requires TWILIO_WEBHOOK_URL) | `tests/channels/test_sms.py` |
| bluebubbles | webhook | BLUEBUBBLES_URL, BLUEBUBBLES_PASSWORD | BLUEBUBBLES_WEBHOOK_SECRET — LEO convention X-Leo-Signature (community bridge) | `tests/channels/test_bluebubbles.py` |
| signal | webhook | SIGNAL_CLI_URL, SIGNAL_PHONE_NUMBER | SIGNAL_WEBHOOK_SECRET — LEO convention X-Leo-Signature (community bridge) | `tests/channels/test_signal.py` |
| relay | webhook | RELAY_TARGET_URL | RELAY_SECRET — generic signed JSON envelope, HMAC both directions | `tests/channels/test_relay.py` |
| email | poll | SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, IMAP_HOST, IMAP_USER, IMAP_PASSWORD | — (poll only; IMAP BODY.PEEK, per-process UID cursor) | `tests/channels/test_email.py` |
| matrix | poll | MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN | — (client-server /sync poll) | `tests/channels/test_matrix.py` |
| irc | poll | IRC_HOST, IRC_PORT, IRC_NICKNAME | — (minimal RFC-1459 client) | `tests/channels/test_irc.py` |
| ntfy | poll | NTFY_URL | — (`/json?poll=1`) | `tests/channels/test_ntfy.py` |
| a2a | send-only | A2A_AGENT_URL, optional A2A_API_KEY | — (A2A JSON-RPC message/send) | `tests/channels/test_a2a.py` |

The shared `tests/channels/test_wechat_crypto.py` covers the wecom/weixin AES
helper. All 19 per-channel suites use injected clients and/or signed fixtures;
none contacts a live service.

### Gateway, admission and registry evidence

- `tests/test_channels_registry.py` (18 tests) covers auto-discovery,
  `BROKEN_MODULES == {}`, capability classification, fail-closed admission,
  record sync, dynamic webhook/poll routes and signed challenge echo.
- Central fail-closed inbound admission lives in
  `leo/integrations/admission.py`, configured by `ChannelsConfig` in `leo.yaml`
  (`channels.default_mode: allowlist`; optional
  `channels.policies.<name>.mode|allow`). Internal surfaces `cli`/`api`/
  `dashboard`/`local` are always allowed; external channels with an empty
  allowlist deny everyone.
- Webhook route `POST /webhooks/{channel}` is dynamic: 401 on a bad signature,
  403 when a valid sender is not admitted, 404 for an unknown channel. Signed
  subscription challenges are echoed. Poll route: `POST /channels/{channel}/poll`.
- `LeoApp.initialize()` calls `sync_integration_records()`, which upserts every
  record into the `integrations` table — this is why the dashboard's Connected
  Surfaces view now lists all channels with honest status. CLI: `leo integrations`
  (list) and `leo integrations --sync`.

### Honest limitations

- Teams: live JWT verification requires an injected verifier or a fronting
  gateway; without one it fails closed.
- WeChat-family (wecom/weixin): callbacks need query parameters merged into the
  JSON body by a fronting route, and AES decryption needs the optional
  `cryptography` backend.
- BlueBubbles/Signal are community bridges; their webhooks are only accepted
  behind a relay that adds the LEO-convention `X-Leo-Signature`.
- A2A is send-only; an inbound A2A server and task polling are out of scope.
- Slack/Feishu subscription challenge echo is handled by the webhook route.
- The email UID cursor is per-process (no durable cross-restart checkpoint).
- Everything is mock-tested only; no live verification is claimed.

### Runtime enrollment from the Integrations page

Every channel can now be enabled from the dashboard without editing YAML or
exporting secrets into the shell. The Connected Surfaces table has a
**Configure** action per channel that renders a generated form from the
channel's declarative spec (`leo/integrations/channels/specs.py`, 22 specs).
It asks only for that platform's own details — e.g. Telegram bot token +
webhook secret, Discord bot token **or** incoming webhook URL, Twilio SID/token
+ public webhook URL, IMAP/SMTP halves for email — plus an allowlist of sender
IDs and an inbound mode. Fields carry labels and short "where to get this"
help text; optional fields are marked as such.

API: `GET /integrations/catalog` (all specs + honest status), `GET
/integrations/catalog/{channel}`, `POST /integrations/{channel}/enable`,
`POST /integrations/{channel}/disable`, `POST /integrations/{channel}/verify`
(offline check of missing fields — never a live probe). Enrollment writes
through one service (`leo/integrations/enrollment.py`): validate → broker
credentials → save the inbound policy → audit event → re-sync the dashboard
row. Secret values are never returned by any endpoint, event, error or record.

Credentials are referenced by env-var **name** everywhere (adapters call the
broker's `credential(name)`). Submitted values live in process memory, and are
also stored encrypted in SQLite (PBKDF2-HMAC-SHA256 → SHAKE-256 keystream →
HMAC-SHA256 envelope, stdlib only) when `LEO_CREDENTIALS_KEY` is set. Without
that key the UI says "session-only — lost on restart"; wrong/missing keys fail
closed. Admission policies persist in `kv` and are merged over YAML policies
at startup. Evidence: `tests/test_credential_store.py`,
`tests/test_enrollment.py`, `tests/test_integration_enrollment_api.py`,
`tests/channels/test_channel_specs.py`,
`tests/channels/test_credential_enrollment.py`.

## Deferred

- Public deployment, subscriptions, remote VPS provisioning: out of scope without authorization.
- ClawHub/Hermes skill-registry clients: trust/license review required before implementation.
- OS-level sandboxing (containers): Docker unavailable; permission/path enforcement used instead.
- WhatsApp Baileys, SimpleX and Hermes-specific `buzz`/`photon`/`raft` platforms:
  deliberately deferred (ToS risk, unverifiable crypto, no verifiable public API).
