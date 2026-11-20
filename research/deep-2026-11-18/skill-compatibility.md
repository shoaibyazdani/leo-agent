# LEO ↔ Hermes / OpenClaw SKILL.md compatibility: empirical test (2026-11-18)

## 0. Executive summary

I fetched real `SKILL.md` files from both upstream repositories at pinned commits, fed them into LEO's actual `translate_skill_md()` loader and `SkillRegistry`, and ran a second pass over **every** upstream `SKILL.md` (327 files).

**Result: LEO parses a subset of frontmatter as inert metadata. It does not load skills functionally, and even metadata import is lossy and sometimes wrong.**

- 9 selected real skills: **7/9 translated**, **6/7 registered**, **0/9 executable**.
- Whole corpus: Hermes **144/186** translated (42 rejected); OpenClaw **135/141** translated (6 rejected). Of the 279 translated manifests, **27 are invalid for LEO's own registry** (all OpenClaw `requires.bins` names equal to the skill name → "skill cannot depend on itself").
- Dominant rejection cause: **44 real skills** are rejected because ordinary shell snippets in the Markdown *body* are scanned as dangerous code and turned into blocking errors — even though the adapter itself labels the body "documentation data and is never executed".
- Hermes `requires_toolsets` / `requires_tools` are upstream **visibility gates** (hide the skill when a tool is unavailable), but LEO maps them into **`permissions`** (authorization grants). OpenClaw `requires.bins/env/config/anyBins` are **environment gates**, but LEO maps a subset into **`dependencies`** and drops the rest. Gate semantics are absent from LEO entirely.
- All 135 OpenClaw skills have no `version:` field; LEO silently fabricates `1.0.0`.
- LEO reads only the single `SKILL.md` file. Bundled `scripts/`, `references/`, `assets/`, `${HERMES_SKILL_DIR}` and `{baseDir}` are neither inventoried nor resolved. No upstream script was executed in this test, and none can be executed by LEO's current registry without a locally written Python handler.

Prior design guidance ("similar skill shape suggests a shared LEO skill loader plus two frontmatter adapters", `docs/RESEARCH_HERMES_OPENCLAW.md:68`; `docs/CAPABILITY_INVENTORY.md:9-10`) is **too optimistic**: a shared *parser* is feasible, but "loading" currently means producing an inert manifest, not a runnable skill. Correction and prioritized fixes are in §7–§8.

## 1. Sources, pins and exact commands

All upstream material was treated as untrusted data. I never ran an interpreter, build tool, installer or script from inside a downloaded directory; no upstream instructions were followed. All harness code lives outside the LEO tree.

```
# Fresh, empty target dirs
mkdir -p $TMPDIR/leo-research-skills/hermes $TMPDIR/leo-research-skills/openclaw $TMPDIR/leo-research-skills-harness

# Resolve HEAD (GitHub REST API was rate-limited: HTTP 403, x-ratelimit-remaining: 0)
git ls-remote https://github.com/NousResearch/hermes-agent.git HEAD
git ls-remote https://github.com/openclaw/openclaw.git HEAD

# Pin and fetch (blobless, depth 1), then check out as inert data
git -C $TMPDIR/leo-research-skills/hermes init -q
git -C $TMPDIR/leo-research-skills/hermes remote add origin https://github.com/NousResearch/hermes-agent.git
git -C $TMPDIR/leo-research-skills/hermes fetch -q --depth=1 --filter=blob:none origin b076b5856cc6b74ab815ea97d866ee5022a5282a
git -C $TMPDIR/leo-research-skills/hermes checkout -q FETCH_HEAD
git -C $TMPDIR/leo-research-skills/hermes rev-parse HEAD

git -C $TMPDIR/leo-research-skills/openclaw init -q
git -C $TMPDIR/leo-research-skills/openclaw remote add origin https://github.com/openclaw/openclaw.git
git -C $TMPDIR/leo-research-skills/openclaw fetch -q --depth=1 --filter=blob:none origin b4bf350c7b0d9ec4ecc3758dfd0642cfaaed44ea
git -C $TMPDIR/leo-research-skills/openclaw checkout -q FETCH_HEAD
git -C $TMPDIR/leo-research-skills/openclaw rev-parse HEAD

# Inventory (git path listing; frontmatter read with yaml.safe_load only)
cd $TMPDIR/leo-research-skills-harness
<repo>/python -I inventory_skills.py
<repo>/python -I show_frontmatter.py <eco>:<rel-path> ...

# Actual loader/registry harness (isolated SQLite DB in harness dir)
cd $TMPDIR/leo-research-skills-harness
<repo>/python -I run_compat.py | tee run_compat.log
<repo>/python -I probe_edges.py | tee probe_edges.log
<repo>/python -I probe_register_all.py | tee validation_all.log
```

Pins:

| Repository | Pinned commit | Working copy | `SKILL.md` count |
|---|---|---|---|
| NousResearch/hermes-agent | `b076b5856cc6b74ab815ea97d866ee5022a5282a` | `$TMPDIR/leo-research-skills/hermes` | 186 |
| openclaw/openclaw | `b4bf350c7b0d9ec4ecc3758dfd0642cfaaed44ea` | `$TMPDIR/leo-research-skills/openclaw` | 141 |

Harness files (outside LEO): `$TMPDIR/leo-research-skills-harness/{inventory_skills.py,show_frontmatter.py,run_compat.py,probe_edges.py,probe_register_all.py,*.json,*.log}`. LEO source was **not** modified; no upstream content was copied into the LEO tree. `python -I` was used with the LEO venv interpreter (system `python3` lacks PyYAML); `the repository root added to the import path` was used to import LEO modules. The registry used a fresh DB (`$TMPDIR/leo-research-skills-harness/harness-skills.sqlite3`) and `EventBus(persist=False)` so no LEO state was touched.

## 2. Inputs: selected real skills

| # | Ecosystem | Exact path (relative to pinned repo root) | SHA-256 of file | Why selected |
|---|---|---|---|---|
| 1 | Hermes | `optional-skills/blockchain/evm/SKILL.md` | `11a5ad5e76a00594347810c7fc61a52a5b166be963a51e0599c7f37966856538` | `requires_toolsets: [terminal]`, `scripts/evm_client.py` |
| 2 | Hermes | `skills/productivity/pdf/SKILL.md` | `11126ee5460ff6b0699514198a46ac7c2ff2dd4c6c03ac99b81ac3f5b9d37cf3` | `scripts/`, `references/`, `tests/`; shell examples in body |
| 3 | Hermes | `optional-skills/creative/hyperframes/SKILL.md` | `65f32ac7049ce5cf2fd6f70290597f13a17817e093dbc6d2c3bc020355693728` | gate + `prerequisites.commands` + `scripts/setup.sh` |
| 4 | Hermes | `skills/research/arxiv/SKILL.md` | `9b41877bcb64166dc2355a43aedd3198c3e5214af67f09e1d1b717a989d92f72` | `scripts/search_arxiv.py`; `curl … | python -c` in body |
| 5 | OpenClaw | `skills/tmux/SKILL.md` | `8aeaa6f3dfe06d58e09721778b8c9e654bcbd47590f26f442c4dc87c9349c572` | `requires.bins: [tmux]` (same as skill name) + 2 scripts |
| 6 | OpenClaw | `skills/model-usage/SKILL.md` | `f4e40f006365acc5f519a55f4efacb6c722788026792e498fcf46dc1a0f0c1d1` | `requires.bins`, `{baseDir}`, scripts + references |
| 7 | OpenClaw | `skills/openai-whisper-api/SKILL.md` | `afff70c706a9bae250b046b741e9930340ca87cc1689a82d4415dd4d6b45bfa1` | `requires.bins` + `requires.env` + `primaryEnv` + script |
| 8 | OpenClaw | `skills/meme-maker/SKILL.md` | `682db7cba34a3d3ca791f7cccaee9488810c731d6622d8886a9bcca77c59d2f3` | `requires.bins: [node]`, `{baseDir}/scripts/meme.mjs` |
| 9 | OpenClaw | `skills/skill-creator/SKILL.md` | `7da0631cb21fbd87630b5dbeb48010f66776c28e396c276235f5d245903fa6ed` | no `metadata.openclaw` namespace; scripts/references/assets |

Additional real files used for targeted probes: Hermes `optional-skills/autonomous-ai-agents/grok/SKILL.md`, `optional-skills/research/research-paper-writing/SKILL.md`; OpenClaw `extensions/discord/skills/discord/SKILL.md`, `skills/coding-agent/SKILL.md`, `skills/goplaces/SKILL.md`, `skills/sherpa-onnx-tts/SKILL.md`, `extensions/lobster/SKILL.md`.

Observed upstream frontmatter shapes (data only):

- Hermes: every file has `name`, `description`, `version`, `platforms`; 185/186 `author`, 185/186 `license`, 181/186 `metadata.hermes`; 30 have top-level `dependencies`, 25 `prerequisites`, 3 `required_environment_variables`, 2 `required_commands`. `metadata.hermes` keys seen: `tags` (181), `related_skills` (122), `category` (56), `homepage` (24), `requires_toolsets` (10), `upstream` (12); plus `config`, `blueprint`, `fallback_for_*` in the authoring docs. 51 skills have `scripts/`.
- OpenClaw: no file has `version:`. `metadata.openclaw.requires` shapes: absent 87, `bins` 40, `config` 6, `bins+env` 3, `anyBins` 2, `anyBins+config` 1, `env` 1. Other keys: `emoji`, `install`, `os`, `primaryEnv`, `skillKey`, `always`, `user-invocable` (9), top-level `allowed-tools` (4). 21 skills have `scripts/`.

## 3. Observed outputs (selected skills)

`run_compat.py` output, verbatim core lines:

```
hermes   optional-skills/blockchain/evm/SKILL.md      compatible=True  name=evm version=1.0.0 deps=[] perms=['terminal'] registered=True  run_success=False errors=0
hermes   skills/productivity/pdf/SKILL.md             compatible=False name=None version=None deps=None perms=None registered=None run_success=None errors=1
hermes   optional-skills/creative/hyperframes/SKILL.md compatible=True name=hyperframes version=1.0.0 deps=[] perms=['terminal'] registered=True run_success=False errors=0
hermes   skills/research/arxiv/SKILL.md               compatible=False name=None version=None deps=None perms=None registered=None run_success=None errors=1
openclaw skills/tmux/SKILL.md                         compatible=True  name=tmux version=1.0.0 deps=['tmux'] perms=[] registered=False run_success=None errors=0
openclaw skills/model-usage/SKILL.md                  compatible=True  name=model-usage version=1.0.0 deps=['codexbar'] perms=[] registered=True run_success=False errors=0
openclaw skills/openai-whisper-api/SKILL.md           compatible=True  name=openai-whisper-api version=1.0.0 deps=['curl', 'node'] perms=[] registered=True run_success=False errors=0
openclaw skills/meme-maker/SKILL.md                   compatible=True  name=meme-maker version=1.0.0 deps=['node'] perms=[] registered=True run_success=False errors=0
openclaw skills/skill-creator/SKILL.md                compatible=True  name=skill-creator version=1.0.0 deps=[] perms=[] registered=True run_success=False errors=0
```

Details per skill:

1. **Hermes `evm`** — compatible, `source_format=skill.md:hermes`, `version=1.0.0`, `permissions=['terminal']`, `dependencies=[]`, `category='general'` (upstream category `blockchain` was inside `metadata.hermes` and dropped). `unsupported_fields=['license', 'metadata.hermes (gates retained for review, never executed)', 'platforms']`. Registered with `health=available`, `has_handler=False`. `run()` → `success=False`, error begins `skill has no executable handler; Imported from hermes SKILL.md. # EVM Blockchain ...`. Body contains `scripts/` and shell code; both merely copied into `documentation`.
2. **Hermes `pdf`** — rejected: `errors=["dangerous code or shell string in field 'documentation'"]`; `security_warnings` even says `(body is documentation data and is never executed)`. Matching lines in the body include `python -c "import pypdf, reportlab, pdfplumber"`, `python3 -c 'import os, sys; os.symlink(...)'` and `rm -rf "$stage"`. Directory contains `scripts/`, `references/`, `tests/` — never examined by the loader.
3. **Hermes `hyperframes`** — compatible; `permissions=['terminal']`, `dependencies=[]`; `unsupported_fields` includes `prerequisites` — upstream `prerequisites.commands: [node, ffmpeg, npx]` is entirely lost. `scripts/setup.sh` not imported. Registered, no handler.
4. **Hermes `arxiv`** — rejected for documentation containing `curl -s "https://export.arxiv.org/api/query?..." | python -c "..."`. `scripts/search_arxiv.py` never considered.
5. **OpenClaw `tmux`** — translated but **cannot register**: `validation.valid=False`, error `skill cannot depend on itself` because `requires.bins: [tmux]` became `dependencies=['tmux']` and the skill is also named `tmux`. `metadata.openclaw.requires` reported as "partially translated"; `os`, `install` silently dropped. Two scripts present.
6. **OpenClaw `model-usage`** — compatible, `version=1.0.0` (fabricated), `dependencies=['codexbar']`, registered with `health=degraded` because LEO checks bin names against registered *skill* names. `{baseDir}` remains a literal string in `documentation`.
7. **OpenClaw `openai-whisper-api`** — compatible; `dependencies=['curl', 'node']`; `requires.env: [OPENAI_API_KEY]` and `primaryEnv` **dropped**. Body still contains literal `{baseDir}/scripts/transcribe.sh`. Registered `health=degraded`, no handler.
8. **OpenClaw `meme-maker`** — compatible; `dependencies=['node']`; `scripts/meme.mjs` not imported; registered `health=degraded`, no handler.
9. **OpenClaw `skill-creator`** — compatible with warning `no metadata.hermes or metadata.openclaw namespace found; ecosystem-specific gates ignored`; no version/author/category from upstream; registered, no handler. Body references scripts/references/assets that LEO does not resolve.

**Execution boundary probe** (Hermes `evm`, isolated DB):

```
no handler:   success=False  error="skill has no executable handler; Imported from hermes SKILL.md. …"
dummy handler injected by harness: success=True output={"executed_by":"harness_dummy_handler","args":{"chain":"base","address":"0x0"}}
imported input_schema: {'type': 'object', 'properties': {}}
```

The imported manifest has an empty input schema (no upstream parameters) and runs only when a handler written by the harness is supplied. **No upstream script was executed; no upstream skill was demonstrated to work.**

## 4. Corpus-wide results

Harness pass over all 327 pinned `SKILL.md` files:

| Metric | Hermes | OpenClaw | Total |
|---|---|---|---|
| Files | 186 | 141 | 327 |
| `translate_skill_md` compatible | 144 | 135 | 279 |
| Rejected by translation | 42 | 6 | 48 |
| Translated manifests valid for `validate_manifest` | 144/144 | 108/135 | 252/279 |
| Invalid: "skill cannot depend on itself" | 0 | 27 | 27 |
| Demonstrated executable with upstream behavior | 0 | 0 | 0 |

Rejection categories across the 48 files:

| Reason (exact error) | Count | Examples |
|---|---|---|
| `dangerous code or shell string in field 'documentation'` | 44 | Hermes `pdf`, `arxiv`, `grok`, `excel-author`; OpenClaw `openclaw-debugging`, `node-inspect-debugger`, `python-debugpy` |
| `manifest has no usable 'name'` | 3 | Hermes `optional-skills/finance/3-statement-model/SKILL.md` and `optional-skills/security/1password/SKILL.md`, OpenClaw `skills/1password/SKILL.md` (all have names starting with a digit) |
| `SKILL.md must start with YAML frontmatter delimited by ---` | 1 | OpenClaw `extensions/lobster/SKILL.md` (starts `# Lobster`) |

Version distribution after translation: all 135 OpenClaw skills → `1.0.0`; Hermes versions mostly preserved (`1.0.0` 72, `0.1.0` 17, `1.0.1` 12, `1.1.0` 10, `2.0.0` 7, `1.2.0` 6, …).

Unresolved path tokens surviving in translated documentation: Hermes `scripts/` 53 skills, `references/` 51, `assets/` 8, `${HERMES_SKILL_DIR}` 3; OpenClaw `scripts/` 40, `references/` 36, `assets/` 2, `{baseDir}` 6.

## 5. Compatibility matrix

Legend: **parsed** = `translate_skill_md` returned a manifest; **registered** = `SkillRegistry.register` accepted it; **executable** = `SkillRegistry.run` produced upstream behavior (verified success). All rows below were run against real files.

| Skill | Ecosystem | Gating metadata present | Bundled scripts | Parsed (metadata) | Registered | Runnable | Fields lost / misinterpreted |
|---|---|---|---|---|---|---|---|
| `evm` | Hermes | `requires_toolsets:[terminal]` | `evm_client.py` | Yes | Yes, `available` | No | gate→permission; category/tags/platforms dropped; script path not resolved |
| `pdf` | Hermes | no explicit gate | 8 scripts + references + tests | **No** (false security rejection) | No | No | body shell snippets treated as executable code |
| `hyperframes` | Hermes | `requires_toolsets:[terminal]`, `prerequisites.commands` | `setup.sh` | Yes | Yes, `available` | No | gate→permission; `prerequisites`, platforms dropped |
| `arxiv` | Hermes | no explicit gate | `search_arxiv.py` | **No** (false security rejection) | No | No | `curl … | python -c` in docs rejected |
| `tmux` | OpenClaw | `requires.bins:[tmux]` | 2 scripts | Yes | **No**: self-dependency | No | bins→dependencies; env/os/install gating absent |
| `model-usage` | OpenClaw | `requires.bins:[codexbar]` | 2 scripts + references | Yes | Yes, `degraded` | No | fabricated version; `{baseDir}` literal; bin treated as skill dependency |
| `openai-whisper-api` | OpenClaw | `bins:[curl,node]`, `env:[OPENAI_API_KEY]`, `primaryEnv` | `transcribe.sh` | Yes | Yes, `degraded` | No | `env` + `primaryEnv` dropped (OR bug); `{baseDir}` literal |
| `meme-maker` | OpenClaw | `requires.bins:[node]` | `meme.mjs` | Yes | Yes, `degraded` | No | fabricated version; script not imported |
| `skill-creator` | OpenClaw | none | 4 scripts + references + assets | Yes (classified `unknown`) | Yes, `available` | No | no ecosystem provenance in manifest; resource dirs ignored |

**No row in this matrix is executable.** "Parsed" and "registered" are metadata facts only.

## 6. Root causes in LEO source (exact locations)

Relevant files: `<repo>/leo/skills/compatibility.py`, `<repo>/leo/skills/registry.py`, `<repo>/leo/contracts.py`, `<repo>/tests/test_compat_skill_md.py`.

1. **Body documentation scanned as executable code** — `compatibility.py:396-407` puts the Markdown body into `raw["documentation"]`; `compatibility.py:418` calls `translate_manifest(raw)`; `translate_manifest` calls `scan_security(raw)` at `:261`; `scan_security` applies `_DANGEROUS_CODE_RE` (`:31-46`) to *every* string value at `:123-126`, including documentation. `translate_skill_md:429-434` separately labels the same hits advisory, which contradicts the blocking error. This explains 44 rejections.
2. **OpenClaw gating collapsed into `dependencies` with an OR bug** — `compatibility.py:412-415`: `_string_list(requires.get("bins") or requires.get("env") or namespace.get("requires"))`. If `bins` exists, `env` is dropped; with only `env`/`config`/`anyBins`, `_string_list` receives a dict and returns `[]`. Gated `permissions` are read from `namespace.get("permissions")`, which upstream does not use. `registry.py:206-211` rejects a dependency equal to the skill name → 27 unregisterable skills. `registry.py:830-835` then marks registered skills `degraded` because bin names are checked against registered skill names.
3. **Hermes gates inverted into permissions** — `compatibility.py:409-410` maps `requires_tools`/`requires_toolsets` to `permissions`. Upstream `website/docs/developer-guide/creating-skills.md:115-135` defines these as "hidden when ANY listed toolset/tool is **not** available" (plus `fallback_for_*`, inverse). LEO has no availability evaluation; the gate becomes a grant. A synthetic probe `requires_tools: [shell]` is rejected as `dangerous permission: 'shell'` (`compatibility.py:25-30, 295-302`), although it is only a gate.
4. **Strict name regex rejects legitimate upstream names** — `compatibility.py:160-168` normalizes but cannot keep a leading digit; `registry.py:35,167-169` enforces `^[a-z][a-z0-9_-]{0,63}$`. 3 real skills are dropped: Hermes `3-statement-model`, Hermes `1password`, OpenClaw `1password`.
5. **Single-file loader ignores bundles** — `compatibility.py:369-380` reads exactly one file and never inspects the parent directory. `SkillManifest` (`contracts.py:221-234`) has no field for `scripts/`, `references/`, `assets/`, or a skill directory. No substitution of `${HERMES_SKILL_DIR}` (Hermes authoring doc `:284-302`) or `{baseDir}` (OpenClaw docs `:477`) occurs; tokens remain literal in `documentation`. Directory paths passed to the loader fail with `SKILL.md must start with YAML frontmatter delimited by ---`.
6. **Version fabrication** — `compatibility.py:171-184,402` substitutes `1.0.0` whenever `version` is missing; all 135 OpenClaw skills are affected. No provenance field records this.
7. **Lossy field mapping** — `compatibility.py:396-407` copies unknown top-level fields but `raw.update()` overwrites `dependencies` with `_string_list(metadata.hermes.config or ...)` (config items are dicts → empty list → key removed by `:416`), so real Hermes top-level `dependencies` can be lost (confirmed with a synthetic frontmatter probe: `dependencies: [alpha, beta]` plus `metadata.hermes.config` translated to `dependencies=[]`). OpenClaw `allowed-tools`, `os`, `install`, `primaryEnv`, `user-invocable`, Hermes `platforms`, `prerequisites`, `required_environment_variables` and `required_commands` are only listed in `unsupported_fields` (`compatibility.py:322-323`) and never retained in the manifest beyond `documentation`. `translated` extras (`:341-348`) are a report-only side channel, not part of `manifest` persisted by `registry.py:794-806`.
8. **No graceful handling of frontmatter-less real skill** — `compatibility.py:381-383` hard-rejects `openclaw extensions/lobster/SKILL.md`.
9. **No execution model** — `registry.py:994-1002` returns "skill has no executable handler" for every imported skill; `registry.py:1004-1009` only calls locally supplied Python callables. The adapter docstring (`compatibility.py:1-8`) explicitly promises inert manifests. Functional compatibility therefore cannot be claimed from this code path.

## 7. Prioritized adapter fix list

"Byte-for-byte functional loading" is not achievable by the current adapter and should not be claimed. Two distinct targets must be specified:

- **Target A — lossless, faithful metadata import (achievable now).** Preserve every field, distinguish gates from grants, keep provenance, and label the result `metadata-imported, not executable`.
- **Target B — functional execution (separate project).** Requires a skill-directory bundle type, path containment, sandbox/allowlisted argv, dependency/gate evaluation, and explicit user approval. Merely parsing SKILL.md cannot deliver it, and running third-party scripts in-process is unacceptable.

| Priority | Change | Exact location | Acceptance criterion |
|---|---|---|---|
| P0 | Exclude `documentation` from the blocking security scan; keep advisory warnings for body text. Split `scan_security` into `blocking` (frontmatter execution primitives) and `advisory` (body text). | `compatibility.py:31-46,109-137,261,396-407,418,429-434` | `pdf` and `arxiv` real inputs translate; tests still reject `command: rm -rf /` in frontmatter |
| P0 | Add a structured `requirements`/`gates` representation to `SkillManifest`; map OpenClaw `bins`, `anyBins`, `env`, `config`, `primaryEnv`, `os`, `install` and Hermes `requires_toolsets`, `requires_tools`, `fallback_for_*`, `platforms`, `prerequisites`, `required_environment_variables` into it. Stop mapping gates into `permissions`/`dependencies`. | `contracts.py:221-234`; `compatibility.py:408-416,295-307`; `registry.py:206-211,830-835` | All 135 OpenClaw + 144 Hermes translated manifests pass `validate_manifest`; `tmux` registers; gates survive byte-for-byte in `manifest` |
| P0 | Keep upstream `name` with a deterministic, reversible normalization plus `source_name` provenance; allow leading digits where unambiguous. | `compatibility.py:22,160-168,257`; `registry.py:35,167-169` | `3-statement-model` and `1password` import without collision |
| P1 | Accept a skill **directory** path: locate `SKILL.md`, inventory `scripts/`/`references/`/`assets/` with symlink containment checks, and rewrite `${HERMES_SKILL_DIR}` / `{baseDir}` only to inert, manifest-recorded placeholders. Never import or execute those files. | `compatibility.py:140-157,369-380`; `contracts.py:221-234` | Directory input works; manifest records resource paths; no file outside the bundle is referenced |
| P1 | Preserve `version` faithfully; use an explicit sentinel for "unspecified" instead of `1.0.0`, with `ecosystem`, `source_path`, `upstream_sha` and `extracted_fields` provenance on the manifest. | `compatibility.py:171-184,325-348,396-442`; `contracts.py:221-234`; `registry.py:794-806` | OpenClaw skills no longer claim version `1.0.0`; provenance round-trips through SQLite |
| P1 | Stop overwriting top-level Hermes `dependencies` with `metadata.hermes.config`; retain all unknown frontmatter keys in a `source_metadata` dict instead of only listing them as unsupported. | `compatibility.py:304-307,322-323,341-348,396-416` | A real Hermes skill with both top-level dependencies and a config block keeps both |
| P2 | Handle frontmatter-less SKILL.md files as `quarantined/unknown` with a warning rather than a hard error; never auto-register them. | `compatibility.py:381-389` | `extensions/lobster/SKILL.md` produces a report, not an exception-level rejection |
| P2 | Make gate enforcement explicit: skip/degrade (never silently run) skills whose gates are unsatisfied; use gate evaluation rather than `dep not in self._skills`. | `registry.py:826-843,957-1002` | Health report names the unsatisfied gate; `run()` refuses gated skills consistently |
| P2 | Correct design docs: mark both SKILL.md adapters as "metadata import tested, execution untested/not supported" and remove the implication that a shared loader yields runtime compatibility. | `<repo>/docs/CAPABILITY_INVENTORY.md:9-10,76`; `<repo>/docs/RESEARCH_HERMES_OPENCLAW.md:68,86` | Documentation no longer overstates compatibility |

## 8. Tests to add

All fixtures should be synthetic frontmatter snippets shaped like the real files (do not copy upstream SKILL.md text into the LEO tree). Add to `<repo>/tests/test_compat_skill_md.py` and a new `tests/test_skill_bundle_import.py`:

1. `test_body_shell_examples_are_advisory_not_blocking` — body containing `python -c`, `rm -rf`, `curl … | python -c`; assert `compatible=True`, `security_warnings` non-empty, `errors==[]`; keep existing frontmatter `command:` rejection test.
2. `test_openclaw_bins_equal_to_skill_name_registers` — `name: tmux`, `requires.bins: [tmux]`; assert `validate_manifest(...).valid` and `registry.register` succeeds with self-dependency absent.
3. `test_openclaw_all_require_subkeys_preserved` — `bins`, `env`, `config`, `anyBins`, `primaryEnv`, `os`, `install`; assert each appears in the new structured gate field with exact values.
4. `test_hermes_gates_not_permissions` — `requires_tools: [web_search]`, `fallback_for_toolsets: [browser]`; assert permissions empty and gates preserved; assert `requires_tools: [shell]` no longer errors.
5. `test_numeric_prefixed_skill_name_import` — `name: 3-statement-model`; assert non-empty manifest with reversible `source_name`.
6. `test_directory_bundle_inventory_no_execution` — temp dir with `SKILL.md`, `scripts/x.py`, `references/y.md`, `assets/z.txt`, and a symlink escaping the bundle; assert resources recorded, symlink excluded, `entrypoint is None`, no script run.
7. `test_unspecified_version_provenance` — OpenClaw-shaped frontmatter without version; assert sentinel version and `ecosystem/source_path` recorded, no fabricated `1.0.0`.
8. `test_unknown_frontmatter_fields_retained` — Hermes top-level `dependencies` plus `metadata.hermes.config` list of dicts; assert both survive in `source_metadata`.
9. `test_plain_markdown_skill_quarantined` — no frontmatter; assert report with warning, not hard rejection, and no registration.
10. `test_imported_skill_is_not_executable` — for Hermes and OpenClaw fixtures, register then `run()`; assert `success=False` and error contains "no executable handler"; assert injecting a local handler is the only way to get `success=True`.
11. Corpus regression (CI, offline fixture list of paths+hashes from §2, no network): assert the 9 selected cases behave exactly as in the matrix and that no imported skill reports `has_handler=True`.

## 9. Limitations and non-claims

- Upstream loaders were **not** executed; only repository files and docs were read as data. Compatibility of Hermes ↔ OpenClaw with each other remains untested.
- No credentials, network services, paid services, publishing or destructive commands were used; no upstream scripts, installers or build tools were run.
- "Compatible" in the harness output means `translate_skill_md` returned a manifest. It is not evidence that an upstream skill works, that its tools exist, that its gating is honored, or that its scripts are safe.
- False-positive counts are tied to the pinned commits and the exact dangerous-string regexes in LEO at the time of testing.
- GitHub REST API was rate-limited (403, 60 requests/hour exhausted); pins were obtained via `git ls-remote` instead. No API credentials were used.
- LEO tree changes in this investigation: only this report file.
