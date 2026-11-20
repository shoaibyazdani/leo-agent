# Testing LEO

```bash
uv run pytest -q
```

The suite is offline by default: a mock provider answers, no network calls are
made, and every test gets an isolated SQLite database under a temporary
directory.

Optional live checks (opt-in, need a real browser or provider):

```bash
LEO_LIVE_BROWSER=1 uv run pytest tests/test_browser_live.py
```

An example release-tree run on 2026-11-21 recorded 1,987 passed and 20 skipped.
Counts move with the tree — re-run the command instead of quoting a number.

## Model gateway in tests

Tests run against the mock provider, so no credentials are needed. When a host
supplies `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL`, an
`opencode-go` provider is registered and sends the `x-opencode-session` header
taken from `LEO_OPENCODE_SESSION`. With no live provider configured the platform
degrades to mock only when `models.allow_mock_fallback` is enabled; otherwise
model calls are reported as blocked instead of invented.
