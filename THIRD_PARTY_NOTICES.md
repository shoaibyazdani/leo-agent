# Third-party notices

LEO itself is licensed under the MIT License (see `LICENSE`). This file lists
third-party software, fonts and reference projects that LEO uses, studies or
ships metadata for, with their licenses and attribution.

## Bundled fonts

IBM Plex Mono and IBM Plex Sans — https://github.com/IBM/plex
  SIL Open Font License 1.1. Full license text: `leo/interfaces/static/fonts/OFL-IBM-Plex.txt`.

Space Grotesk — https://github.com/floriankarsten/space-grotesk
  SIL Open Font License 1.1. Full license text: `leo/interfaces/static/fonts/OFL-Space-Grotesk.txt`.

## Runtime dependencies (Python)

Installed from `uv.lock`. All are permissively licensed; no GPL/AGPL/MPL
runtime dependency is present.

| Package | License family |
|---|---|
| fastapi, starlette, uvicorn | BSD-3-Clause / MIT |
| pydantic, pydantic-settings, pydantic-core | MIT |
| httpx, httpcore, h11 | BSD-3-Clause |
| aiosqlite | MIT |
| typer, click | MIT / BSD-3-Clause |
| rich, pygments, markdown-it-py | MIT |
| pyyaml | MIT |
| opentelemetry-api, opentelemetry-sdk | Apache-2.0 |
| idna | BSD-3-Clause |
| anyio, sniffio, certifi | MIT / Apache-2.0 / MPL-2.0 (certifi: MPL-2.0, data file only) |
| playwright (optional `browser` extra) | Apache-2.0 |
| hatchling (build backend) | MIT |

Dependency inventory generated from `uv.lock` on 2026-11-21. Re-generate before
each release; a new GPL/AGPL/MPL runtime dependency is a release blocker.

## Reference projects (no source copied)

Hermes Agent — https://github.com/NousResearch/hermes-agent
  MIT License, Copyright (c) 2025 Nous Research.
  Studied read-only during integration research; capabilities were
  independently implemented and only declarative `SKILL.md` metadata is
  translated for review.

OpenClaw — https://github.com/openclaw/openclaw
  MIT License, Copyright (c) 2026 OpenClaw Foundation.
  Studied read-only; OpenClaw publishes its own THIRD_PARTY_NOTICES.md covering
  its transitive dependencies — consult it directly if any such material is
  ever imported.

## Desktop shell (optional)

If the Electron shell under `desktop/` is packaged, Electron and Chromium are
distributed under the MIT license and BSD-style licenses respectively; ship
their license texts with the installer.
