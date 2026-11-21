# LEO Engineering Decisions

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Python 3.12 + asyncio modular monolith; SQLite/WAL | Matches plan recommendation, zero extra infrastructure, easy local recovery and inspection. | 2026-11-18 |
| 2 | Virtualenv at `.venv` managed by `uv` | Reproducible installs, Python 3.12 available locally, no system pollution. | 2026-11-18 |
| 3 | Provider endpoint/model from real environment config; `mock` provider for tests | Plan forbids invented endpoints, credentials or model IDs. | 2026-11-18 |
| 4 | Secrets are env-var *names* in config; never values | Prevents credential leakage into config, prompts, logs and API responses. | 2026-11-18 |
| 5 | Pydantic v2 contracts in `leo/contracts.py` | Stable interfaces for parallel module development without circular imports. | 2026-11-18 |
| 6 | Durable task state in SQLite with evidence requirement | Protects against false completion and unsafe replay. | 2026-11-18 |
| 7 | External ecosystem capabilities are reimplemented natively or via narrow adapters, no source copying | License safety and functional integration rather than superficial imitation. | 2026-11-18 |
| 8 | SKILL.md availability gates are stored separately from permissions and dependencies | Legacy ecosystem gates hide skills when prerequisites are absent; mapping them to grants or registry dependencies is both wrong and insecure. | 2026-11-18 |
| 9 | Webhook verification fails closed; HMAC covers exact raw bytes | Default no-secret configuration previously accepted arbitrary messages. | 2026-11-18 |
| 10 | Non-loopback API binds require `LEO_API_TOKEN`; local-only remains tokenless | Keeps the product usable locally while removing an unauthenticated remote-control surface. | 2026-11-18 |
| 11 | Imported external skills are inert until a sandboxed execution model exists | Parsing SKILL.md is not evidence that third-party scripts are safe or runnable. | 2026-11-18 |

Open assumptions: local-only mode is the default; network integrations remain
DISCONNECTED/BLOCKED until credentials are supplied by the user. The web
dashboard binds to localhost only.
