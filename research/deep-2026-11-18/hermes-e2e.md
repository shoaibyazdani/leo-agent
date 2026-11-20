# Hermes Agent — end-to-end runtime test (2026-11-18)

**QA lead report.** Target: upstream [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), commit `b076b5856cc6b74ab815ea97d866ee5022a5282a` (`2026-11-18T01:56:03+05:30`, subject `fix(codex): close a 400 whose body read fails; drop the raw body once answered`). LEO source was not modified. No API credentials were used; no paid provider was contacted. All model traffic went to a local loopback mock server written by this test.

## 0. Method, isolation, safety

- Checkout: `git clone --depth 1 https://github.com/NousResearch/hermes-agent.git repo` into `$TMPDIR/leo-research-hermes-e2e/repo` (fresh empty `$TMPDIR/leo-research-hermes-e2e/`). All downloaded files were treated as untrusted data; no instructions found in them were followed as operator commands.
- Harness and all scripts/logs/venv: `$TMPDIR/leo-research-hermes-e2e-harness/` (separate directory). Python parsing used `python3 -I` with absolute script paths. CLI entry point was the venv's absolute `$TMPDIR/leo-research-hermes-e2e-harness/venv/bin/hermes`; no interpreter or script was run from inside the checkout as `cwd`.
- Every CLI invocation ran with `HOME=$TMPDIR/leo-research-hermes-e2e-harness/home` and `HERMES_HOME` set to the same path via `$TMPDIR/leo-research-hermes-e2e-harness/run_cmd.py`, which records exact command, stdout, stderr, exit code, and a timeout. For the test suite only, `HERMES_HOME` was separated to `/tmp/.../test-home` to avoid a test guard reacting to `HOME/lib/libc.dylib` probes.
- Mock endpoint: `$TMPDIR/leo-research-hermes-e2e-harness/mock_server.py`, bound to `127.0.0.1:8765` only. Fixture: `/tmp/.../mock-fixture.txt` containing `MOCK_FIXTURE_OK_V1`. Request log: `/tmp/.../logs/mock_requests.jsonl`.
- Cleanup: mock server both mock-server processes killed; `hermes gateway stop` run; final `hermes gateway status` = `✗ Gateway is not running`. Nothing under `<repo>` was changed except this report.

Elapsed wall time: ~02:06 → ~02:32 PKT on 2026-11-18 (26 minutes), with several failed attempts and retries before success.

---

## 1. Install: exact commands, outputs, real failures

### 1.1 Source checkout with the machine's system Python 3.9 fails immediately

```
$ HOME=/tmp/.../home /usr/bin/python3 -I -c "the repository root added to the import path; from hermes_cli.main import main; main()" -- --help
EXIT=1
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "$TMPDIR/leo-research-hermes-e2e/repo/hermes_cli/main.py", line 16, in <module>
    import hermes_bootstrap
  File "$TMPDIR/leo-research-hermes-e2e/repo/hermes_bootstrap.py", line 581, in <module>
    from pm.environments import activate_dependencies, install_state_permission_message
  File "$TMPDIR/leo-research-hermes-e2e/repo/pm/environments.py", line 17, in <module>
    from hermes_constants import get_default_hermes_root, project_venv_dir
  File "$TMPDIR/leo-research-hermes-e2e/repo/hermes_constants.py", line 29, in <module>
    def set_hermes_home_override(path: str | Path | None):
TypeError: unsupported operand type(s) for |: 'type' and 'type'
```

`pyproject.toml` states `requires-python = ">=3.11,<3.15"` and `.python-version` is `3.14`, with an inline comment that only 3.14 is actually supported. The README's one-line install is `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash`; that script was deliberately not executed (untrusted remote code). `setup.py` also blocks wheel/sdist builds outside Nix with an explicit message that only the shell installer, Docker image, Nix, or an editable install are supported.

### 1.2 Isolated venv (outside the checkout)

```
$ uv venv --python <python-toolchain>/python/cpython-3.14-macos-x86_64-none/bin/python3.14 $TMPDIR/leo-research-hermes-e2e-harness/venv
Using CPython 3.14.7 interpreter at: .../python3.14
Creating virtual environment at: $TMPDIR/leo-research-hermes-e2e-harness/venv
EXIT=0
$ /tmp/.../venv/bin/python --version
Python 3.14.7
```

A fresh uv-managed Python 3.14.7 was already present on the machine. No Python 3.14 interpreter was on the default PATH as `python3.14`, so a naive user would have to let the installer download one.

### 1.3 Straight editable install fails with a genuine dependency conflict

```
$ uv pip install --python /tmp/.../venv/bin/python -e $TMPDIR/leo-research-hermes-e2e/repo
EXIT=1
Using Python 3.14.7 environment at: /tmp/.../venv
error: No solution found when resolving dependencies
  cause: Because browser-harness>=0.1.13 depends on pillow==12.3.0 and
         hermes-agent==0.0.0 depends on browser-harness==0.1.13, we can conclude
         that hermes-agent==0.0.0 and pillow==13.0.0 are incompatible.
         And because hermes-agent==0.0.0 depends on pillow==13.0.0, we can
         conclude that hermes-agent==0.0.0 cannot be used.
         And because only hermes-agent==0.0.0 is available and you require
         hermes-agent, we can conclude that your requirements are unsatisfiable.
```

Verified against installed metadata: `browser_harness-0.1.13.dist-info/METADATA` contains `Requires-Dist: pillow==12.3.0`, while `pyproject.toml` pins `pillow==13.0.0`.

### 1.4 The committed lockfile does not rescue the install

```
$ uv export --project $TMPDIR/leo-research-hermes-e2e/repo --no-emit-project --no-dev --no-emit-index-url --format requirements-txt --output-file /tmp/.../requirements.lock.txt
Resolved 355 packages in 6.46s    # exit 0
$ uv pip install --python /tmp/.../venv/bin/python -r /tmp/.../requirements.lock.txt
EXIT=1
error: No solution found when resolving dependencies
  cause: Because browser-harness>=0.1.13 depends on pillow==12.3.0 and you
         require browser-harness==0.1.13, we can conclude that your requirements
         and pillow==13.0.0 are incompatible.
```

The exported lockfile itself has `pillow==13.0.0` (line 585) alongside `browser-harness==0.1.13` (line 19). This is an upstream pin inconsistency at the tested commit, not a local resolver quirk.

### 1.5 Workaround that made the install usable

```
$ python3 -I /tmp/.../filter_reqs.py /tmp/.../requirements.lock.txt /tmp/.../requirements.fixed.txt pillow
REMOVED: ["pillow==13.0.0 ; python_full_version >= '3.14' \\"]
$ printf '\npillow==12.3.0\n' >> /tmp/.../requirements.fixed.txt
$ uv pip install --python /tmp/.../venv/bin/python -r /tmp/.../requirements.fixed.txt
DEPS_EXIT=0        # 355 packages, took >3 min (moved to background, completed normally)
$ uv pip install --python /tmp/.../venv/bin/python --no-deps -e $TMPDIR/leo-research-hermes-e2e/repo
PROJECT_EXIT=0
Installed 1 package in 4ms
 + hermes-agent==0.0.0 (from file://$TMPDIR/leo-research-hermes-e2e/repo)
```

Test tooling installed separately into the same venv: `uv pip install --python .../venv/bin/python pytest==9.1.1 pytest-asyncio==1.4.0`, then `aiohttp` after a test failure (below).

### 1.6 Version/help smoke test

```
$ /tmp/.../venv/bin/hermes --version
EXIT=0
Hermes Agent vgit.b076b58 (2026.9.24) · stable · local b076b5856cc6
Install directory: $TMPDIR/leo-research-hermes-e2e/repo
Install method: git
Python: 3.14.7
OpenAI SDK: 2.54.0
Up to date

$ /tmp/.../venv/bin/hermes --help
EXIT=0
usage: hermes [-h] [--version] [-z PROMPT] [--usage-file PATH] ...
              <command> ...
# ~70 subcommands listed: chat, model, gateway, setup, status, cron, doctor,
# config, skills, tools, sessions, doctor, plugins, security, acp, dashboard, ...
```

Note the version string mixes a git-derived build (`vgit.b076b58`) with a dated release (`2026.9.24`) — confusing but harmless.

---

## 2. CLI paths: exact commands, outputs, exit codes

All commands use the absolute venv entry point and the isolated HOME. Full captures are in `$TMPDIR/leo-research-hermes-e2e-harness/logs/<prefix>.{cmd,out,err,code}`.

### 2.1 Doctor / health

```
$ hermes doctor                      # before any config
EXIT=1
┌─────────────────────────┐
│         🩺 Hermes Doctor │
└─────────────────────────┘
◆ Python Environment
  ✓ Python 3.14.7
  ✓ SQLite 3.53.1
  ✓ Runtime venv staged (...) (active in this process)
◆ Configuration Files
  ✗ ~/./.env file missing
  ⚠ config.yaml not found (using defaults)
◆ Command Installation
  ✓ Hermes entry point exists (/tmp/.../venv/bin/hermes)
  ✗ ~/.local/bin/hermes not found (hermes command may not work outside the venv)
◆ External Tools
  ⚠ ripgrep (rg) not found     ⚠ Docker/Podman not found
  ⚠ Node.js not found (optional; run: hermes pm install node)
  ⚠ Chromium not installed (browser_exec cannot start a local browser)
◆ Tool Availability   (20 toolsets ✓; a2a, browser-cdp, computer_use, discord,
  image_gen, video*, vision, x_search ⚠ with reasons)
────────────────────────────────────────────────────────────
  Found 3 issue(s) to address:
  1. Run 'hermes setup' to create .env
  2. Missing ~/.local/bin/hermes symlink — run 'hermes doctor --fix'
  3. Run 'hermes setup' to configure missing API keys for full tool access
```

After three chat turns and config migration, the same command still exits 1 and now reports 4 issues, including `⚠ Config version outdated (v0 → v53)` before migration and successful WAL/state.db checks (`2 sessions`, `8 messages`, FTS5 tables). `hermes config check` after migration reports `Config version: 53 ✓`.

**Findings:** doctor is useful and detailed, but its exit code is 1 whenever *optional* integrations are absent; a fully functional offline core still reads as "broken". Paths render as `~/./.env` / `~/./config.yaml` under an explicit `HERMES_HOME`, which is confusing. It claims `Node.js not found` even though `node v24.22.0` was on PATH (`<node-version-manager>/.../bin/node`) — it appears to check only its own managed store. `dump` later said the gateway was `running (launchd, the gateway process)` while `gateway status` said `Running manually, not as a system service`.

### 2.2 Status, sessions, cron, tools

```
$ hermes status
EXIT=0
  Model:        (not set)
  Provider:     Auto
  Providers:    none connected
  Gateway:      ✗ stopped
  Platforms:    none configured
  Jobs:         0
# after mock chat: Model: mock-model / Provider: custom /
# Providers: Custom endpoint / Gateway: ✓ running

$ hermes sessions list
EXIT=0   # fresh: "No sessions found."
EXIT=0   # after chat: two rows, columns
         # Preview | Workspace | Last Active | Src | ID
         # src = oneshot, IDs local-<sha256>

$ hermes sessions show <id>
EXIT=2
hermes sessions: 'show' is not a `hermes sessions` command.
Run `hermes sessions --help` to see all commands.
# 'show' is not in the command list; use sessions list/export/browse/stats.

$ hermes cron list
EXIT=0
No scheduled jobs in profile 'default'.
Create one with 'hermes cron create ...' or the /cron command in chat.

$ hermes cron status
EXIT=0
  Scheduler host: the host gateway (the gateway process) serving profiles default
✓ Gateway is running — cron jobs will fire automatically
  the gateway process
  Ticker heartbeat: 6s ago
  No active jobs

$ hermes tools list
EXIT=0
Built-in toolsets (cli):
  ✓ enabled  web, browser, terminal, file, code_execution, vision,
             image_gen, tts, skills, todo, memory, session_search,
             connections, clarify, delegation, cronjob, computer_use
  ✗ disabled  video, video_gen, x_search, stt, kanban, context_engine
Plugin toolsets (cli):
  ✗ disabled  a2a  🔌 A2A
```

### 2.3 Skills list / inspect — a docs-vs-reality mismatch

```
$ hermes skills list
EXIT=0
              Installed Skills
┏━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ Name ┃ Category ┃ Source ┃ Trust ┃ Status ┃
┡━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
└──────┴──────────┴────────┴───────┴────────┘
0 hub-installed, 0 builtin, 0 local — 0 enabled, 0 disabled

$ hermes skills list --source all        # or --source builtin
EXIT=0
# 43 rows (apple-notes, codex, github, pdf, systematic-debugging, ...)
0 hub-installed, 43 builtin, 0 local — 43 enabled, 0 disabled
```

The default `skills list` output is misleading: it says there are zero skills while 43 bundled skills are installed and enabled. `--source all` is required to see them.

```
$ hermes skills inspect /tmp/.../repo/skills/apple/apple-notes/SKILL.md
EXIT=0
Error: Could not find '/tmp/.../SKILL.md' in any source.

$ hermes skills inspect apple-notes
EXIT=0
Resolving 'apple-notes'...
Multiple skills named 'apple-notes' found:
│ skills.sh │ community │ skills-sh/openclaw/openclaw/apple-notes │
│ clawhub   │ community │ apple-notes                             │
Use the full identifier to install a specific one.
```

`inspect` is registry-only: a local bundled `SKILL.md` path cannot be inspected, and a bare name resolves against remote registries (network traffic, no credentials). Also note `hermes dump` reported `skills: 44` while `skills list --source all` reported 43.

### 2.4 Config init / migration

There is no `hermes config init` subcommand. `hermes config --help` lists `{show,edit,get,set,unset,path,env-path,check,migrate}`. `hermes setup` is the wizard (with `--non-interactive`, `--quick`, section flags); its first-run prompt cannot run without a TTY.

```
$ hermes config path
EXIT=0
$TMPDIR/leo-research-hermes-e2e-harness/home/config.yaml

$ hermes config set model.provider custom
EXIT=0   ✓ Set model.provider = custom in .../config.yaml
$ hermes config set model.base_url http://127.0.0.1:8765/v1
EXIT=0   ✓ Set model.base_url = ...
$ hermes config set model.default mock-model
EXIT=0   ✓ Set model.default = ...
$ hermes config get model
EXIT=0
provider: custom
base_url: http://127.0.0.1:8765/v1
default: mock-model
```

`config migrate` on a config written by `config set` is a bad non-interactive experience:

```
$ hermes config migrate
EXIT=TIMEOUT (120s)
🔄 Checking configuration for updates...
  Config version: 0 → 53
  ℹ️  274 optional API key(s) not configured:
     • DEEPSEEK_API_KEY
     ... (≈290 lines of key names) ...
  Configure new keys? [y/N]:
# hangs: waits on stdin with no --yes/--no flag

$ printf 'n\n' | hermes config migrate
EXIT=0
  Configure new keys? [y/N]:   Set later with: hermes config set <key> <value>
# config.yaml now ends with: _config_version: 53; model settings preserved
```

The hand-written minimal config had no `_config_version`, so doctor read it as v0 even though `config check` had previously reported v53 on the *absent* config. Adding `_config_version: 53` via migration fixed that, but required feeding stdin.

### 2.5 Offline-safe extras and error surfaces

```
$ hermes completion zsh                  EXIT=0   # full #compdef script
$ hermes dump                            EXIT=0   # version, OS, python, model,
     provider, api_keys (all not set), toolsets, gateway, skills: 44
$ hermes bogus-command                   EXIT=2
hermes: 'bogus-command' is not a `hermes` command.
Run `hermes --help` to see all commands.
$ hermes skills list                     EXIT=2 (when passed as one argv token by
     a zsh caller that did not word-split) → "'skills list' is not a `hermes`
     command. Did you mean: skills?"  # correct behavior once split properly
$ hermes setup --help                    EXIT=0   # --non-interactive exists
$ hermes gateway stop                    EXIT=0   ✓ Stopped gateway for this profile
$ hermes gateway status                  EXIT=0   ✗ Gateway is not running
```

---

## 3. Full offline chat turn with tool call + streaming

### 3.1 Mock endpoint

`/tmp/.../mock_server.py` (stdlib `http.server`, loopback only) implements:

- `GET /v1/models` → one model `mock-model`, `context_length: 128000`.
- `POST /v1/chat/completions` → if tools are offered and no tool result is present, returns an OpenAI tool call to `read_file` for the fixture; once a `role: tool` message is present, returns the final answer. `stream: true` sends SSE `chat.completion.chunk` events and `[DONE]`. Every request is logged with its JSON body.
- Unknown paths (notably `/api/v1/models`) return 404; Hermes retried discovery and continued.

### 3.2 First attempt: explicit CLI flags alone were not enough

```
$ hermes chat -q 'Read ... using the read_file tool and report its first line.' \
    --oneshot --provider custom --base-url http://127.0.0.1:8765/v1 \
    --api-key mock-local-key -t file --yolo -Q
EXIT=1

It looks like Hermes isn't configured yet -- no API keys or providers found.
  Run:  hermes setup

☤ Hermes Setup — Non-interactive mode
  No interactive TTY detected for the first-run setup prompt.
  Configure Hermes using environment variables or config commands:
    hermes config set model.provider custom
    hermes config set model.base_url http://localhost:8080/v1
    hermes config set model.default your-model-name
```

Despite `--base-url`/`--api-key`, the first-run gate fired. Only `GET /v1/models` was received by the mock. After persisting the three `model.*` config keys (§2.4), the same turn worked.

### 3.3 Successful turn (streaming + real tool execution)

```
$ hermes chat -q 'Read ... using the read_file tool and report its first line.' \
    --oneshot --api-key mock-local-key -t file --yolo -Q
EXIT=0
STDOUT:
MOCK_TURN_OK: the fixture says MOCK_FIXTURE_OK_V1.
STDERR:
session_id: local-066ff729600904109ffda011ca34f58dd9acc2c0109a5c66812b20b7f6e8b63e
```

Mock request log (10 requests total across attempts): after discovery there were exactly two `POST /v1/chat/completions`, both with `stream=True` and four tools (`patch`, `read_file`, `search_files`, `write_file`):

- Request 6: 2 messages (system + user).
- Request 7: 4 messages. Message 2 is the assistant tool call `{"id":"call_mock_read_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/.../mock-fixture.txt\"}"}}`; message 3 is `role: tool`, `tool_call_id: call_mock_read_1`, content `{"content": "1|MOCK_FIXTURE_OK_V1\n2|Second line for tool-result verification.", "total_lines": 2, "file_size": 61, ...}`.

This proves the full loop: SSE request → streamed tool call → real local tool execution → tool result fed back → streamed final answer.

### 3.4 NDJSON streaming contract

```
$ hermes chat -q 'Say MOCK_STREAM_OK and nothing else.' --oneshot \
    --api-key mock-local-key -t file --yolo -Q --format stream-json
EXIT=0
{"type":"system","subtype":"init","model":"","session_id":"","timestamp":1794950467308}
{"type":"tool_use","name":"read_file","tool_call_id":"call_mock_read_1","input":{"path":"/tmp/.../mock-fixture.txt"},"timestamp":...}
{"type":"tool_result","name":"read_file","tool_call_id":"call_mock_read_1","output":"{\"content\": \"1|MOCK_FIXTURE_OK_V1\\n2|...\"}", "duration_ms":511,"is_error":false,...}
{"type":"text","text":"\n\nMOCK_TURN_OK: ","timestamp":...}
{"type":"text","text":"the ",...} ... {"type":"text","text":"MOCK_FIXTURE_OK_V1.",...}
{"type":"result","session_id":"local-0bd8...","exit_code":0,
 "text":"MOCK_TURN_OK: the fixture says MOCK_FIXTURE_OK_V1.",
 "tokens":{"input":0,"output":0,"total":0,"cache_read":0,"cache_write":0},
 "duration_ms":1873,...}
STDERR: Session: local-0bd8... / session_id: local-0bd8...
```

Quirks: the `system/init` event has empty `model` and `session_id`, and `tokens` are all zero even though the mock returned `usage` in every completion. The final answer still reflects the canned fixture, not the literal user prompt, because the mock is scripted — that is expected.

### 3.5 Repeats and safe mode

- Post-migration smoke: `hermes chat -q 'Final offline smoke check.' --oneshot --api-key mock-local-key -t file --yolo -Q` → exit 0, same final answer, new session id.
- `hermes chat -q 'Safe mode offline check.' --oneshot --safe-mode --provider custom --model mock-model --base-url http://127.0.0.1:8765/v1 --api-key mock-local-key -t file --yolo -Q` → exit 0. This matches the documented requirement that safe mode names a model explicitly.
- Each chat turn auto-started/attached the gateway; `hermes sessions list` showed the persisted oneshot sessions, and `hermes status` reported `Gateway: ✓ running` after the turn. We explicitly stopped the gateway at the end.

**No paid provider was contacted.** Only loopback HTTP plus harmless network probes (e.g., doctor's IPv6/OpenRouter reachability check and skills registry resolution) occurred.

---

## 4. Bounded upstream test runs

All runs used `/tmp/.../venv/bin/python -m pytest -q <absolute test paths>` from the harness directory, with a 300 s external timeout.

### 4.1 First subset — environment artifact, then clean

```
pytest -q .../tests/test_hermes_constants.py .../test_hermes_yaml.py \
        .../test_session_retention.py .../test_platforms_marker.py .../test_scratch_dir.py
EXIT=1 (first run)
... 2 failures in test_scratch_dir.py ...
AssertionError: TEST BUG: file I/O against the REAL hermes home:
  /tmp/.../home/lib/libc.dylib
```

Cause: the test suite's `home_io_guard.py` intercepts `os.path.isfile`; on macOS `ctypes.util.find_library('c')` probes `$HOME/lib/libc.dylib`. Because our runner set `HOME == HERMES_HOME`, that probe landed in the guarded tree. Separating `HERMES_HOME` to `/tmp/.../test-home` fixed it:

```
EXIT=0
119 passed, 7 skipped in 10.40s
```

### 4.2 Second subset — missing optional test dependency

```
pytest -q .../test_hermes_bootstrap.py .../test_hermes_logging.py \
        .../test_get_tool_definitions_cache_isolation.py \
        .../test_packaging_metadata.py .../test_project_metadata.py
EXIT=1 (first run)
FAILED tests/test_hermes_logging.py::test_aiohttp_access_line_never_writes_a_query_credential_to_disk
E   ModuleNotFoundError: No module named 'aiohttp'
1 failed, 91 passed, 18 skipped in 14.42s
```

After `uv pip install ... aiohttp`:

```
$ pytest -q .../tests/test_hermes_logging.py
EXIT=0
46 passed, 6 skipped, 1 warning in 15.92s
# warning: aiohttp DeprecationWarning: Bare functions are deprecated, use async ones
```

Net: 165 passed, 13 skipped across the bounded subsets after environment fixes; no genuine product-test regression found. The two initial failures were environment/setup issues, which are themselves findings about onboarding onto this repo.

---

## 5. Operational ergonomics and docs-vs-reality

1. **Install friction is high and real.** Python 3.14-only (system 3.9 fails on syntax), no supported pip/wheel path, curl-piped installer is the documented route, and as of this commit both `uv pip install -e` and the committed `uv.lock` fail because of `pillow 13.0.0` vs `browser-harness==0.1.13` requiring `pillow==12.3.0`. Workaround required manual requirements filtering.
2. **"Not configured" gate ignores CLI overrides.** `--provider custom --base-url ... --api-key ...` still triggered the first-run wizard on a fresh home. The error text itself suggests the config commands that fix it, which is good, but a scripted user must write config before a one-shot call.
3. **Doctor over-reports failure.** Exit 1 for optional integrations; missing `~/.local/bin/hermes` even when the venv entry point works; `Node.js not found` with Node 24 on PATH; `~/./` path rendering; gateway service type inconsistency between `dump` and `gateway status`.
4. **Skill defaults lie.** `skills list` = 0 while `--source all` = 43 builtin; `skills inspect` rejects local paths and hits remote registries for bare names; `dump` says 44.
5. **Migration is noisy and interactive.** ~290 lines listing 274 unset keys, then a `Configure new keys? [y/N]:` prompt that hangs without stdin; no `--yes/--no` flag. Minimal configs written by `config set` are versionless and read as v0.
6. **Chat is a gateway client now.** Docs say "single query mode (non-interactive)", but the turn runs in a gateway process; `--oneshot` is needed to exit, turns/sessions persist, and the gateway stays running afterward. This is documented in `cli.md` but surprising in practice.
7. **Streaming metadata gaps.** `--format stream-json` is genuinely incremental, but init events have empty model/session_id and usage counters are zero despite provider usage being returned.
8. **Discovery noise is tolerated.** Several `GET /api/v1/models` 404s during custom-endpoint model discovery, followed by successful `/v1/models` and chat requests; no user-visible error.
9. **`config check` prints an enormous optional-key list** (200+ lines) even when everything required is satisfied — low signal-to-noise.
10. **Good parts worth copying:** readable Rich tables, actionable hints in errors, clear `--help` for ~70 subcommands, WAL/FTS5 session store that worked immediately, capability gating in `tools list`, and a real tool-call loop that required no special setup beyond a base URL.

---

## 6. Verdict table

| Area tested | Verdict | Evidence |
|---|---|---|
| Clone / commit pin | works | SHA `b076b585…`; clean clone at `$TMPDIR/leo-research-hermes-e2e/repo` |
| Source run on Python 3.9 | blocked (expected) | `TypeError: unsupported operand type(s) for |` |
| Isolated venv + Python 3.14 | works | `Python 3.14.7`, venv outside checkout |
| Dependency install, naive `uv pip install -e` | blocked | pillow 13.0.0 vs 12.3.0 conflict |
| Install from committed lockfile | blocked | same conflict in exported lock |
| Install with workaround | works | filtered requirements + `pillow==12.3.0` + `--no-deps -e` |
| `--version` / `--help` | works | exit 0, full command list |
| `doctor` | partially works | detailed and accurate core checks; exit 1 for optional items, odd paths/service claims |
| `status` | works | exit 0, reflects model/provider/gateway before and after chat |
| Config init via `config set` / `config get` | works | exit 0, values persisted |
| `config migrate` | partially works | needs stdin `n` to avoid hanging; 290 lines of output; fixes `_config_version` |
| `hermes setup` first-run wizard | blocked (no TTY) | explicit "interactive wizard cannot be used here" |
| `skills list` (default) | partially works | exit 0 but reports 0 skills while 43 builtins exist |
| `skills list --source all` | works | 43 builtin skills listed |
| `skills inspect` local path | blocked | "Could not find ... in any source" |
| `skills inspect <name>` | works | resolves registry identifiers with exit 0 |
| `sessions list` | works | exit 0; empty before chat, two persisted oneshot sessions after |
| `sessions show <id>` | blocked (no such command) | exit 2 with command list hint |
| `cron list` / `cron status` | works | exit 0; no jobs, scheduler host reported |
| `tools list` | works | exit 0; enabled/disabled toolsets + reasons |
| Offline chat (mock base URL) | works | exit 0; `MOCK_TURN_OK...` real end-to-end answer |
| Tool call + tool result | works | `read_file` called, real file read, result in request 7 |
| Streaming (`stream=True`, SSE) | works | two streamed completions; mock logged `stream=True` |
| NDJSON `--format stream-json` | partially works | incremental events, but empty init fields and zero token counters |
| Safe-mode chat with CLI flags | works | exit 0, model named explicitly |
| Upstream bounded tests | works (after env fixes) | 165 passed, 13 skipped total |
| Paid-provider chat / OAuth flows | blocked (by design) | no credentials available; never attempted |

---

## 7. Ranked E2E lessons for LEO, and test gaps exposed

1. **Make "fresh install → mock provider → streaming → tool call" a first-class E2E suite.** Hermes only proved itself after a config detour; LEO should have a checked-in test that starts a loopback OpenAI-compatible server, runs the real CLI, and asserts the full request sequence (system/user → assistant tool call → tool result → final streamed answer). LEO has unit tests (`tests/test_chat_tools.py::test_chat_executes_tool_then_answers`, `tests/test_models.py` mock/SSE tests) but no test that crosses CLI → config → provider → tools → session store.
2. **Test install determinism separately from code correctness.** Both `uv pip install -e` and the committed lockfile were unsatisfiable at the upstream commit. LEO should run a clean-venv `uv sync`/lock install in CI with a pinned interpreter and assert the CLI starts; do not assume a lockfile exports cleanly. Add a test that `uv.lock` and `pyproject.toml` dependency pins agree.
3. **Explicit CLI overrides must bypass first-run gates.** Add a LEO test for `leo ask --base-url http://127.0.0.1:<port>/v1 --api-key test ...` against a *fresh* home with no config file; assert it succeeds without a wizard. Hermes' gateway blocked this despite flags and had to be unblocked by writing `model.*` keys.
4. **Assert diagnostic contracts, not just diagnostic presence.** LEO already has `leo doctor` (`leo/interfaces/cli.py:62`) and `LEO_TEST_REPORT.md` mentions it, but add no-TTY/exit-code tests: optional missing integrations should not force a failure exit, output should use real absolute paths (not `~/./...`), and it must not misdetect an installed Node/runtime. Hermes' doctor produced four "issues" while the core was fully working.
5. **Test background-process/session lifecycle.** Hermes chat spawns/attaches a gateway; one-shot turns persist sessions and leave the service running. LEO needs an E2E test that a one-shot CLI query terminates (exit 0), records a session, and leaves no orphan service (or documents that it does). Include `--oneshot`/`--format` equivalents and gateway stop behavior.
6. **Make non-interactive operation a formal mode.** Add `--yes/--no` (or `LEO_NONINTERACTIVE=1`) to any migration/setup command and a test that the command completes with stdin closed. Hermes' `config migrate` printed ~290 lines and then hung at a `y/N` prompt; that is the single most script-breaking behavior found.
7. **Catalog listing defaults must reflect installed content.** Add a snapshot test that a bare `leo skill list` shows bundled skills, not an empty table; Hermes required `--source all` to see 43 builtins and reported 44 in `dump`. Also add `skill inspect <local SKILL.md path>` coverage — Hermes rejected a real local path.
8. **Lock down the NDJSON/SSE event contract.** Hermes streamed correctly but emitted empty `model`/`session_id` in the init event and zero tokens despite provider usage. LEO's router already records usage (`test_router_stream_records_usage`); extend it to the CLI stdout level so telemetry regressions are caught at the boundary users/scripts actually consume.
9. **Add CLI error-surface tests for unknown/partial commands.** Hermes exits 2 with a "Did you mean" suggestion and help hint. LEO should assert exit codes, stderr routing, and suggestions for unknown commands/subcommands; a mis-split argv produced a confusing but informative error here.
10. **Keep approval semantics explicit in the E2E harness.** The successful Hermes turn used `--yolo` to skip dangerous-command prompts; only a read-only `read_file` ran. LEO should test both approved and denied tool paths in the same end-to-end scenario, proving that untrusted tool arguments from a model never inherit operator authority — a principle both projects document and this test deliberately did not try to bypass.

**Coverage that LEO already has** (so the gaps above are additions, not replacements): skill frontmatter translation (`tests/test_compat_skill_md.py`), memory/session primitives, provider retries and SSE (`tests/test_models.py`), gateway inbound/outbound routing and HMAC checks (`tests/test_integrations.py`), and a basic chat-tool loop (`tests/test_chat_tools.py`). The missing layer is the real process-level, fresh-home, loopback-provider E2E run with exact output and exit-code assertions.

---

## 8. Artifacts

- Checkout: `$TMPDIR/leo-research-hermes-e2e/repo` (SHA above).
- Harness: `$TMPDIR/leo-research-hermes-e2e-harness/` — `run_cmd.py`, `mock_server.py`, `filter_reqs.py`, `mock-fixture.txt`, `requirements.lock.txt`, `requirements.fixed.txt`, `venv/`, `home/`, `test-home/`.
- Captures: `$TMPDIR/leo-research-hermes-e2e-harness/logs/` — `00`–`98` and `A1`–`A6` prefixes, each with `.cmd`, `.out`, `.err`, `.code`; plus `mock_requests.jsonl` and mock server stdout.
- Report: `<repo>/research/deep-2026-11-18/hermes-e2e.md` (this file). No other LEO file was touched.
