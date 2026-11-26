# Testing LEO

```bash
.venv/bin/python -m pytest -q
```

Use the repository virtualenv directly; do not run `uv run pytest` for release
checks, because it can refresh the lockfile. The suite is offline by default: a
mock provider answers, no network calls are made, and every test gets an
isolated SQLite database under a temporary directory.

Optional live checks (opt-in, need a real browser or provider):

```bash
LEO_LIVE_BROWSER=1 .venv/bin/python -m pytest tests/test_browser_live.py
```

Counts move with the tree, so quote the result of the command you actually ran,
not an earlier number. The schedule release receipt records the full-suite
outcome, JUnit hash and unverified surfaces:

```bash
.venv/bin/python -m pytest --junitxml=scripts/release/evidence/junit.xml
.venv/bin/python scripts/release/check_schedule_release.py
```

The receipt is LEO-owned: it cites LEO commands, LEO tests and LEO JUnit XML
only. Source-project test counts or CI results are never used as LEO evidence.

## Playwright and schedule cards

`tests/test_playwright_approval_cards.py` always runs static asset contract
checks. The real DOM checks call `pytest.importorskip("playwright.sync_api")`
and skip when no Chromium binary is installed. A skipped browser test is not
browser verification; the receipt records that limit explicitly.

## Model gateway in tests

Tests run against the mock provider, so no credentials are needed. When a host
supplies `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL`, an
`opencode-go` provider is registered and sends the `x-opencode-session` header
taken from `LEO_OPENCODE_SESSION`. With no live provider configured the platform
degrades to mock only when `models.allow_mock_fallback` is enabled; otherwise
model calls are reported as blocked instead of invented.

## Delivery and surfaces

Telegram approval transport is covered with a MockTransport, including plain
text chunks and callbacks. Live Telegram sends, a live gateway smoke, a production
database migration drill, backup/restore, and a PTY/TUI session are not performed
by the default suite; the release receipt marks those checks unverified rather than
inferring them from unit tests.
