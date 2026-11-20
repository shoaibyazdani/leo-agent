# Hermes Agent — deep source-level research (2026-11-18)

**Status:** research only. No LEO code was changed; no Hermes code was executed. Downloaded source was treated as untrusted data (read as text only; all parsing scripts lived outside the clone and were run as `python3 -I <absolute script path> <absolute data path>`).

## 0. Provenance and exact commands

Canonical repo: https://github.com/NousResearch/hermes-agent (MIT, LICENSE present at repo root).

Clone (fresh empty directory, never in LEO):

```
$ mkdir -p $TMPDIR/leo-research-hermes-src && cd /tmp
$ git clone --depth 1 https://github.com/NousResearch/hermes-agent.git $TMPDIR/leo-research-hermes-src/hermes-agent
Cloning into '$TMPDIR/leo-research-hermes-src/hermes-agent'...
Updating files: 100% (26910/26910), done.
$ git -C $TMPDIR/leo-research-hermes-src/hermes-agent rev-parse HEAD
b076b5856cc6b74ab815ea97d866ee5022a5282a
$ git -C $TMPDIR/leo-research-hermes-src/hermes-agent log -1 --format='%H%n%ad%n%s' --date=iso-strict
b076b5856cc6b74ab815ea97d866ee5022a5282a
2026-11-17T13:51:56+05:30
fix(codex): close a 400 whose body read fails; drop the raw body once answered
$ du -sh $TMPDIR/leo-research-hermes-src/hermes-agent
430M	$TMPDIR/leo-research-hermes-src/hermes-agent
$ find $TMPDIR/leo-research-hermes-src/hermes-agent -name '*.py' -not -path '*/.git/*' | wc -l
12636
```

**Examined commit SHA: `b076b5856cc6b74ab815ea97d866ee5022a5282a`** (2026-11-17). All source URLs below use the permalink prefix:

`https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/<path>#L<line>`

Prior doc was `<repo>/docs/RESEARCH_HERMES_OPENCLAW.md` (doc-level only). This report goes to source. Existing doc's repository metadata (stars/issues, docs site claims) was not re-verified; GitHub REST was not queried.

Secondary parsing (safe location, isolated interpreter):

```
$ mkdir -p $TMPDIR/leo-research-tools   # outside the downloaded tree
$ python3 -I $TMPDIR/leo-research-tools/count.py $TMPDIR/leo-research-hermes-src/hermes-agent/toolsets.py
TOOLSETS entries: 57
$ python3 -I $TMPDIR/leo-research-tools/count_tool_defs.py $TMPDIR/leo-research-hermes-src/hermes-agent/tools
literal tool names in tools/: 45
non-literal register call sites in tools/: 5 ['discord_tool.py', 'feishu_drive_tool.py', 'kanban_tools.py', 'mcp_tool_registration.py', 'xai_video_tools.py']
$ python3 -I $TMPDIR/leo-research-tools/count_tools.py $TMPDIR/leo-research-hermes-src/hermes-agent
literal registered tool names: 59 unique: 58
by source top dirs: [('tools', 45), ('tests', 14)]
```

Counts are *literal* `registry.register(name="...")` calls found by AST parsing; loops (MCP, kanban, a few platform tools) add dynamic registrations, so these are lower bounds for the runtime surface, not a full inventory.

---

## 1. Subsystem map — actual source, key symbols

### 1.1 Conversation loop and prompt assembly
- Facade: `run_agent.py` — `class AIAgent` (line 215), `main()` (1530). It owns clients, tools, memory, compression and session state.
- Turn loop: `agent/conversation_loop.py` — `_run_conversation_turn()` (1615), `run_conversation()` (1777). Loop condition at 1720–1722: `while (s.api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call`. Each iteration calls `begin_iteration`, `prepare_iteration`, `assemble_api_request`, `run_preflight_gate`, `perform_api_call` (via `_run_api_retry_loop`, 1546), then `run_tool_round` **or** `finish_text_response` (1756–1758). `_LoopState` (1430) holds all per-iteration state.
- Phase helpers: `agent/turn_context.py` (`build_turn_context`), `agent/turn_request_assembly.py` (`assemble_request`, line 112: "ORDER IS LOAD-BEARING"), `agent/turn_tool_round.py`, `agent/turn_api_call.py` (`perform_api_call`), `agent/turn_finalizer.py`, `agent/tool_executor.py`.
- Prompt tiers: `agent/system_prompt.py` — `build_system_prompt_parts()` (770) returns `{"stable", "context", "volatile"}`; `build_system_prompt()` (836) joins them in that order and caches on `agent._cached_system_prompt`; `invalidate_system_prompt()` (848) rebuilds at compression boundaries; `reconstruct_static_prefix()` (869) restores the byte-stable prefix. `agent/prompt_builder.py` supplies the blocks (`build_memory_guidance` 195, `build_skills_system_prompt` 1240, `build_environment_hints` 1021, `build_context_files_prompt` 1767), but the three-tier assembly itself is in `system_prompt.py`, not `prompt_builder.py`.
- API modes: `hermes_cli/config_defaults.py:755` comments enumerate `chat_completions | anthropic_messages | codex_responses`; adapters in `agent/anthropic_adapter.py`, `agent/codex_responses_adapter.py`, `agent/chat_completion_*`.
- Context-file ingestion: `agent/prompt_builder.py` — `discover_context_files()` (1685) covers `HERMES.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `SOUL.md`; `_scan_context_content()` (84) blocks injection matches from project files, only warns for user-authored `SOUL.md` (84–110).
- Cache strategy: prompt cache breakpoints in `agent/prompt_cache_scope.py` / `agent/prompt_caching.py`; `conversation_loop._sync_failover_system_message()` (1219) and `_ensure_cached_system_prompt_static()` (1243) reconcile prompt bytes across provider failover.

URLs: [conversation_loop.py#L1615](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/conversation_loop.py#L1615), [system_prompt.py#L770](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/system_prompt.py#L770), [prompt_builder.py#L84](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/prompt_builder.py#L84).

### 1.2 Tool registry and toolsets
- `tools/registry.py` — `ToolEntry` dataclass (216) fields: name, toolset, schema, handler, check_fn, requires_env, is_async, dynamic_schema_overrides, execution_timeout, compat_args, self_checked_args, deadline_floor. `ToolRegistry.register()` (840) validates schema shape, refuses cross-toolset shadowing without `override=True`, and requires operator opt-in (`plugins.entries.<id>.allow_tool_override`) for a plugin to override a built-in. `get_definitions()` (1016), `dispatch()` (1066), `check_tool_availability()` (1190).
- Import-time discovery: `discover_builtin_tools()` (121) uses AST to find module-body `registry.register()` calls (`_is_registry_register_call` 84, `_module_registers_tools` 94); result cached on disk (`_discovery_cache_path` 178).
- Availability probes: `_call_check_fn()` (389) runs probes on daemon threads under copied contextvars with timeout; `_CHECK_FN_TTL_SECONDS = 30`, `_CHECK_FN_FAILURE_GRACE_SECONDS = 60`, `_CHECK_FN_GRACE_REUSE_SECONDS = 5` (256–275). Grace logic serves the last-good `True` when a recently healthy probe fails.
- Toolsets: `toolsets.py` — `_ts()` (46), `TOOLSETS` (67) with **57 static entries** (includes composites/bundles), `resolve_toolset()` (355), `get_all_toolsets()` (422), `agent_tool_drops()` (467), `agent_tool_surface()` (474). Dynamic plugin toolsets/aliases merge in via `_registry()` (244).
- Deferred tool schemas: `tools/tool_search.py` — `ToolSearchConfig` (43), `assemble_tool_defs()` (347), bridge tools `tool_search` / `tool_describe`; "auto" is currently an alias of "on", so any deferrable tool activates search (207–217). Listing budget = `min(listing_max_tokens=4000, threshold_pct=5% of context)` (219).
- Parallel dispatch path: `model_tools.py` — `handle_function_call()` (906), `_pre_dispatch_guards()` (792), `_execute_tool()` (849). **Hermes has two tool-dispatch paths**: the executor path (`agent/tool_executor.py::_dispatch_authorized_once`, 676) and `model_tools.handle_function_call` (used by execute_code RPC, ACP, bridges, CLI legacy). Both must independently apply policy.

URLs: [registry.py#L840](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/registry.py#L840), [toolsets.py#L67](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/toolsets.py#L67), [tool_search.py#L43](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/tool_search.py#L43).

### 1.3 Approval and security enforcement — where checks actually happen
- Config/policy: `tools/approval_context.py` — `_VALID_MODES = ("manual", "smart", "off")` (276); defaults resolve to `manual` (285–293). Unattended defaults `approvals.{cron_mode,single_query_mode,unattended_mode} = "deny"` in `hermes_cli/config_defaults.py:1626–1628`.
- Detection: `tools/approval_detection.py` — `HARDLINE_PATTERNS` (176, catastrophic floor), `DANGEROUS_PATTERNS` (498, large regex table), custom matchers (pipe-to-shell, dev-socket redirect, WMIC shadow delete, download-then-run, reverse shell), `detect_hardline_command()` (371), `hardline_scan_incomplete()` (420), `_shell_*` normalizers/deobfuscators (1118–1600). This is a hand-maintained shell language subset with explicit "scan incomplete" handling.
- Gate core: `tools/approval.py` — `_floor_block()` (1310) applies hardline, runtime self-delete, sudo-stdin and operator `approvals.deny` **before** yolo/mode=off; `check_dangerous_command()` (1336), `check_all_command_guards()` (1402), `_run_approval_gate()` (1207), `_human_decision()` (1013), `request_tool_approval()` (1372, plugin escalation), `check_execute_code_guard()` (1480). `_should_skip_container_guards()` (1282) skips prompts for Docker without host mounts, modal/daytona/vercel, and plugin backends declaring `skip_container_guards` (fail-soft False).
- Actual call sites: `tools/terminal_tool.py` — `_check_all_guards()` (167), `_run_approval_guards()` (1190), called at 1670 before environment execution; `tools/code_execution_tool.py:778–786` calls `check_execute_code_guard()` before spawning code; plugin hooks can call `request_tool_approval()` (`hermes_cli/plugins.py:1223`).
- Executor-level enforcement: `agent/tool_executor.py::_dispatch_authorized_once()` (676) applies hosted-scope deny-by-default, scope blocks, plugin `pre_tool_call`, pruned-argument checks, then `agent._tool_guardrails.before_call()` (728). `agent/tool_guardrails.py::ToolCallGuardrailController` (302) implements per-turn repeated-call failure/loop caps. Duplicate guard path in `model_tools._pre_dispatch_guards()` (792).
- Other guards: `tools/terminal_tool_guards.py` (`gateway_lifecycle_block` 561, `self_repo_block` 655, `locked_skill_block` 683), `tools/file_tools_write_guards.py`, `tools/write_approval.py` (`evaluate_gate` 204, staged write review), `tools/path_security.py`, `tools/url_safety.py`, `tools/threat_patterns.py`.

URL: [approval.py#L1336](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/approval.py#L1336).

### 1.4 Skills loader and frontmatter
- Parser: `agent/skill_utils.py` — `parse_frontmatter()` (71): strips BOM, finds `\n---\s*\n`, YAML via `hermes_yaml.safe_load`; **on malformed YAML falls back to naive `key: value` line splitting** (86–90). Gates: `skill_matches_platform()` (109), `skill_matches_environment()` (167, unknown tags fail open, offer-time only), `skill_matches_apps()` (177, unknown app fails closed), `_hermes_metadata()` (728), `_CONDITION_KEYS` (737) = `fallback_for_toolsets`, `requires_toolsets`, `fallback_for_tools`, `requires_tools`, `session_platforms`; `extract_skill_config_vars()` (746) reads `metadata.hermes.config` (`key`/`description`/`default`/`prompt`), resolved to `skills.config.<key>` (795–840).
- Discovery/index: `get_search_roots()`/`get_all_skills_dirs()` (413/427), `resolve_skill_catalog()` (473), `iter_skill_index_files()` (878); project skill quarantine via `get_untrusted_project_skills_root()` (603) and `is_quarantined_project_skill()` (635). Prompt index: `agent/prompt_builder.py` `_parse_skill_file()` (1161), `_build_snapshot_entry()` (1144), `_render_skills_index()` (1303).
- Linter/scan: `tools/skill_linter.py` (name must match directory; description length ≤ 60; version/author/license recommended), `tools/skills_guard.py` — `scan_skill()` (910), `scan_file()` (865), decoded views `_decoded_views()` (751), `should_allow_install()` (988) with trust-level policy and `--force` override except "dangerous" from community/trusted. Install flow: `tools/skills_hub_install.py` (quarantine → scan → move → audit log).
- Real bundled frontmatter sample (`skills/media/youtube-content/SKILL.md`, observed verbatim):

```
name: simplify-code
description: "Parallel 4-agent cleanup of recent code changes."
version: 1.1.0
author: Hermes Agent (inspired by Claude Code /simplify)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, cleanup, refactor, delegation, subagent, parallel, simplify]
    related_skills: [requesting-code-review, test-driven-development]
```

URL: [skill_utils.py#L71](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/skill_utils.py#L71).

### 1.5 Memory provider interface and session store
- `agent/memory_provider.py` — `class MemoryProvider(ABC)` (84). Required: `name`, `is_available`, `initialize`, `get_tool_schemas`; optional: `system_prompt_block`, `prefetch`, `queue_prefetch`, `sync_turn`, `on_turn_start`, `on_pre_compress`, `on_memory_write`, `on_delegation`, `on_session_end`, `on_session_switch`, `backup_paths`, profile lifecycle hooks, config panel hooks. `PRE_COMPRESS_CHECKPOINT_API_VERSION = 2` (37); `ctx_bound()`/`spawn_context_thread()` (21/29) force contextvars (profile/secret scope) onto provider threads. Multiplexing contract documented at 87–102: one provider instance per agent/profile, no module-level profile state.
- `agent/memory_manager.py` — `MemoryManager` (201); `on_pre_compress()` (721) combines provider text; with `require_checkpoint=True` a v2 provider failure aborts compression (keeps uncompressed transcript). Built-in store is `MEMORY.md`/`USER.md` via `tools/memory_tool.py`; writes are gated through `tools/write_approval.py` (`tools/memory_tool.py:91`).
- Session store: `hermes_state.py` — `SessionDB` facade (~655 lines) opening only a profile's `state.db` (198–214). Implementation split across `persistence/{core,sqlite,conversation,gateway,authority,housekeeping,workers}/`. FTS5 schema: `persistence/sqlite/_schema_ddl.py:643,763` (`messages_fts`, `messages_fts_trigram`); query sanitizer `_sanitize_fts5_query()` at `persistence/sqlite/conversation/_store/search.py:341`; `session_search` tool at `tools/session_search_tool.py:952` uses MATCH/snippet plus LIKE fallback — no LLM call. Cross-profile reads confined (`_cross_profile_reads_confined`, `_resolve_profile_db` 553/575).
- Session keys: `gateway/run.py::_parse_session_key()` (2906) documents `agent:{ns}:{platform}:{chat_type}:{chat_id}[:extra...]`; `session_keys.py` extracts profile namespace.

URLs: [memory_provider.py#L84](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/memory_provider.py#L84), [session_search_tool.py#L952](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/session_search_tool.py#L952).

### 1.6 Context engine and compaction
- `agent/context_engine.py` — `ContextEngine(ABC)` (60). Abstract `update_from_response`, `should_compress`, `compress`; hooks `prune_tool_results_only` (122), `select_context` (133, request-only replacement), `on_turn_complete` (155), `should_compress_preflight` (168), `on_session_start/end`, `clone_for_agent` (238), `update_model` (245). Defaults: `threshold_percent = 0.75`, `protect_first_n = 3`, `protect_last_n = 6` at ABC level (80–82); `update_model` recomputes `threshold_tokens`.
- Built-in: `agent/context_compressor.py` — `ContextCompressor.__init__()` (2669) defaults `threshold_percent=0.50, protect_first_n=3, protect_last_n=20`; `_MIN_CTX_TRIGGER_RATIO = 0.85` (2583); deterministic fallback summaries around 4875; privilege/anchor/redaction helpers (`_redact_compaction_text` 1089, `_build_anchor_index` 965, `_build_verbatim_user_section` 876). 5215 lines — compression is a subsystem of its own (`agent/conversation_compression*.py` adds ~10 more modules).
- Gateway safety net: `gateway/run_turn_hygiene.py` (`_hmwa_hygiene_settings`, 94) defaults `threshold_pct=0.85`, 30 s timeout, 5000-message hard limit; comment at 97 explicitly says 0.85 is higher than the agent's 0.50.
- Hook call sites: `agent/conversation_loop.py:1326` (`_apply_context_engine_selection`), 1340 (`select_context`, fail-open), 1376 (`on_turn_complete`, fail-open), `agent/turn_context_compression.py:441` (one compress pass per preflight).

### 1.7 Subagent delegation
- `tools/delegate_tool.py` — `registry.register(name="delegate_task", toolset="delegation", ...)` at 999; `DEFAULT_MAX_ITERATIONS = 250` (108); config `delegation.max_iterations` overrides caller-supplied value (661–666). Child sessions: `_open_child_session_db()` (115) gives each child a dedicated store handle; `_build_child_agent()` (209) derives role from depth and narrows toolsets (252–280).
- `tools/delegate_tool_toolsets.py` — `DELEGATE_BLOCKED_TOOLS` (13) = `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob_manage`; `DEFAULT_TOOLSETS = ["terminal", "file", "web"]`; `_resolve_child_toolsets()` (75) intersects child toolsets with parent's, strips blocked names twice (before and after composite expansion); orchestrators regain `delegation` by role, not by model request.
- Plugin API: `agent/subagent_lifecycle.py` — `SubagentLaunchRequest` (49), versioned `SubagentHandle` (65), `SubagentLifecycleService.launch/status/wait/cancel/result/reconnect/steer` (277–387). Validation (451–470) refuses unknown toolsets and any child toolset that would broaden parent permissions; `blocked_tools` per-tool is explicitly **not** supported (232: "Per-tool blocking is not supported; use allowed_toolsets").
- Backends: `agent/delegation_registry.py` (`configured_backend`, `get_active_provider`), `agent/delegation_provider.py` (`DelegationProvider`), `agent/delegation_backends.py`.

### 1.8 Terminal backends
- `tools/environments/base.py` — `BaseEnvironment` (275) + long-lived process mixins. Built-ins: `local.py::LocalEnvironment` (1105), `docker.py::DockerEnvironment` (640), `ssh.py::SSHEnvironment` (91), `modal.py::ModalEnvironment` (132), `managed_modal.py` (45), `daytona.py` (24), `vercel_sandbox.py` (157). Plugin providers via `agent/terminal_env_registry.py` / `agent/terminal_env_provider.py`.
- Builder/selector: `tools/terminal_tool_backends.py` — `_ENV_BUILDERS` / `_SANDBOX_ROWS` (184+), `_create_environment()` (294), `_check_requirements()` (391); plugin backends running in `plugins.isolation: host` are refused for in-process use (`_refuse_plugin_host_provider`, 235).
- Guard interaction: `_docker_has_host_access()` (`tools/terminal_tool.py:156`) treats `docker_mount_cwd_to_workspace` or any host-looking `docker_volumes` entry as host access; otherwise Docker is treated as isolated and approval prompts are skipped. Config allows `docker_forward_env` and `docker_env` (`terminal_tool_backends.py:63`).

### 1.9 Browser tools and CDP supervisor
- Tools: `tools/browser_use_cli.py` (`browser_exec`, registered at 1046; stable per-task scratch dir at 272), `tools/browser_cdp_tool.py` (`browser_cdp`, registered 503; `_browser_cdp_private_guard` 233), `tools/browser_dialog_tool.py`, `tools/browser_vault_tool.py`.
- Supervisor: `tools/browser_supervisor.py` — `CDPSupervisor` (98), `_SupervisorRegistry` (517); frame tracking in `browser_supervisor_frames.py`, dialogs in `browser_supervisor_dialogs.py`. `DEFAULT_DIALOG_POLICY = must_respond`, `DEFAULT_DIALOG_TIMEOUT_S = 300`, policies `auto_dismiss` / `auto_accept` (46–53). Native dialogs are bridged by injecting a script that XHRs to the magic host `hermes-dialog-bridge.invalid`, intercepted via CDP Fetch before resolution (59–84).
- SSRF/URL policy: `tools/browser_url_policy.py` — `url_is_private()` (69, DNS resolution at check time), `evaluate_url_safety()` (135), `_secret_url_error()` (108) blocks API-key-shaped URLs; cloud metadata is always blocked; private-address floor is relaxed only with `browser.allow_private_urls` or the hybrid local sidecar (117–132). `tools/browser_cdp_tool.py:137–175` applies the same guard to JS expressions, navigate params and current page URL.

### 1.10 Gateway and channel adapters
- Core: `gateway/run.py`, `gateway/session.py`, `gateway/platform_registry.py` (`PlatformRegistry` 197), `gateway/platforms/base.py`, `gateway/authz_mixin.py` (`GatewayAuthorizationMixin` 204; `_is_user_authorized` 855; default-deny for ambiguous policies at 855–870).
- Built-in enum `gateway/config.py::Platform` (235) has 20 core values (local, telegram, discord, whatsapp, whatsapp_cloud, slack, signal, mattermost, matrix, email, sms, dingtalk, api_server, webhook, msgraph_webhook, feishu, wecom, wecom_callback, weixin, bluebubbles, qqbot, relay) plus dynamic pseudo-members for registered plugin adapters. **Observed: 26 bundled platform plugin directories** (`plugins/platforms/*`: a2a, bluebubbles, buzz, dingtalk, discord, email, feishu, google_chat, irc, line, matrix, mattermost, ntfy, photon, qqbot, raft, signal, simplex, slack, sms, teams, telegram, wecom, weixin, whatsapp, whatsapp_cloud).
- Access policy: `gateway/platforms/access_policy_mixin.py` — `_is_dm_allowed()` (51) allows `open` **only** if `GATEWAY_ALLOW_ALL_USERS` or `<PLATFORM>_ALLOW_ALL_USERS` is explicitly truthy (43–54); `pairing`/`allowlist` otherwise. Pairing state in `gateway/pairing.py`.
- Webhook ingress: `gateway/platforms/webhook.py` — `DEFAULT_HOST = None` (57, aiohttp binds both address families, effectively all interfaces), `DEFAULT_PORT = 8644` (61), HMAC schemes (71–76), `_validate_route()` (255) requires a secret and refuses `INSECURE_NO_AUTH` on public binds; rate limit 30/min, body cap 1,048,576 bytes, dedupe TTL 3600 s (236–242). Route-level `script` shells out on payload-derived data (`gateway/platforms/webhook.py` ~733). `/api/ws` uses short-lived loopback tickets (gateway/api_server / control socket code).

### 1.11 Cron scheduler
- `cron/scheduler.py` — `run_job()` around 2525; each run gets a fresh session id `cron_{job_id}_{YYYYmmdd_HHMMSS}` (2538). `_construct_cron_agent()` (2442). `CronPromptInjectionBlocked` (407) blocks assembled prompts (including runtime-loaded skill content) that trip the scanner; `cron/scheduler_prompt.py:398–421` implements the scan. `_resolve_cron_disabled_toolsets()` (419) always removes `messaging`/`clarify`, removes `cronjob` by default (loop prevention, not a security boundary), and layers `agent.disabled_toolsets`. `SILENT_MARKER = "[SILENT]"` (543). Job schema in `cron/job_definition.py:15–18` (`schedule`, `script`, `skills`, `deliver`, `origin`, `enabled_toolsets`, `attach_to_session`). In-flight claims and stale sweeps at 800–1105.

### 1.12 Model provider resolver and fallbacks
- `hermes_cli/auth.py` — `resolve_provider()` (1611), `resolve_api_key_provider_credentials()` (2203), `get_auth_status()` (2113), OAuth flows, external-process credentials, provider-state store. `agent/provider_registry.py::ProviderRegistry` (31) is a generic capability registry; `providers/base.py` and `agent/provider_base.py` provide base classes.
- `hermes_cli/fallback_config.py` — `get_fallback_chain()` (139), `scoped_fallback_chain()` (159), `effective_runtime_provider()` (70). `hermes_cli/config_defaults.py:25` default `fallback_providers: []`. Activation/recovery: `agent/agent_init_fallback.py` (`_init_fallback_chain` 22), `agent/turn_recovery.py` (fallback break around 1856), `agent/fallback_cooldown.py`, `agent/credential_pool.py` (`CredentialPool` 942, cooldown/exhaustion/rotation), `agent/model_metadata.py:281` `MINIMUM_CONTEXT_LENGTH = 64_000` with an Ollama runtime guard at `conversation_loop.py:418–445`.

### 1.13 Plugin system
- Loader: `plugins/plugin_loader.py` — `iter_plugin_dirs()` (109), `load_plugin_module()` (171), `_exec()` (148); `hermes_cli/plugins.py` — `PluginManager` directory scan plus `_scan_entry_points()` (637), precedence "project > user > bundled" (538), default `plugins.enabled = None` means opt-in/nothing enabled (531).
- Manifest: `plugin.yaml` requires `name`, `version`, `description` (`hermes_cli/plugin_validate.py:99–114`); optional `provides_hooks`, `kind`, `requires_hermes`. `hermes_cli/plugins_context.py` exposes 30 `register_*` surfaces including `register_tool`, `register_platform`, `register_memory_provider`, `register_context_engine`, `register_terminal_environment_provider`, `register_hook`, `register_system_prompt_section`, `register_approval_transport`, `register_middleware`.
- Isolation: `hermes_cli/plugin_isolation.py` — default `plugins.isolation: in_process` (32, 101–108); optional `host` moves third-party code to a subprocess; certain surfaces (`register_platform`, `register_platform_handler`, model-provider profiles, dashboard auth) are refused under `host` (38–43, 126–136). Bundled plugins always stay in-process.

---

## 2. Claim verification against the existing research doc

Labels: `[VERIFIED]`, `[PARTIAL]` (true with important qualification), `[REFUTED]`, `[UNVERIFIED]`.

| # | Claim from `docs/RESEARCH_HERMES_OPENCLAW.md` | Label | Source evidence at commit `b076b5` |
|---|---|---|---|
| 1 | `AIAgent` facade in `run_agent.py`; loop in `agent/conversation_loop.py` | **[VERIFIED]** | `run_agent.py:215 class AIAgent`; `conversation_loop.py:1615/1777` |
| 2 | Prompt assembly in `agent/prompt_builder.py` with stable/context/volatile tiers | **[PARTIAL]** | Tiers are assembled in `agent/system_prompt.py:770–845`; `prompt_builder.py` builds the individual blocks. Same architecture, wrong file attribution. |
| 3 | Three API modes: chat_completions, codex_responses, anthropic_messages | **[VERIFIED]** | `config_defaults.py:755`, `run_agent.py:518/603`, adapters present |
| 4 | ~70–85 tools in ~28 toolsets | **[REFUTED/STALE]** | AST count: 45 literal tool registrations in `tools/`, 58 unique across repo incl. tests; 57 static `TOOLSETS` entries. Dynamic MCP/kanban registrations exist, so exact runtime total is config-dependent, but documented figures do not match this commit. |
| 5 | Skills: `SKILL.md`, optional `scripts/`/`references/`, frontmatter name/description/version/author/license/platforms + `metadata.hermes` gates | **[VERIFIED]** | `skill_utils.py:71/737/746`, real bundled SKILL.md sampled; linter requires name==dirname and recommends the rest |
| 6 | Memory: built-in MEMORY.md/USER.md, `MemoryProvider` with `prefetch`/`sync_turn`/`on_pre_compress`/`on_memory_write` + tool schemas | **[VERIFIED]** | `memory_provider.py:84–234`, `memory_tool.py`, `write_approval.py`; pre-compress checkpoint v2 at `memory_manager.py:721` |
| 7 | FTS5-backed `session_search`, no LLM call, `~/.hermes/state.db` | **[VERIFIED]** | `_schema_ddl.py:643/763`, `_store/search.py:341/439`, `session_search_tool.py:952`, `hermes_state.py:212` |
| 8 | `delegate_task` isolated sessions, own terminal/toolset, only final summary returns; default `delegation.max_iterations` **50** | **[PARTIAL → default REFUTED]** | Isolation/blocked tools verified (`delegate_tool_toolsets.py:13`, `delegate_tool.py:115`), but `DEFAULT_MAX_ITERATIONS = 250` (`delegate_tool.py:108`); config override accepted (661). Docs at this commit do not say 50. |
| 9 | Dangerous-command modes smart/manual/off, hardline blocklist, user deny globs | **[VERIFIED]** | `approval_context.py:276`, `approval_detection.py:176/498`, `approval.py:1299–1333` |
| 10 | Container backends skip dangerous-command approval because the container is the boundary | **[VERIFIED with caveats]** | `_should_skip_container_guards()` `approval.py:1282–1296`: Docker only when no host mount; plugin backends can declare the flag; user `deny` rules still apply |
| 11 | `execute_code` can call tools from Python; MCP tools named `mcp__<server>__` | **[VERIFIED]** | `code_execution_tool.py:40` (`SANDBOX_ALLOWED_TOOLS`), `code_execution_rpc.py:28–36` dispatches through `model_tools.handle_function_call`; `mcp_tool_schema.py:157–177` |
| 12 | Browser `browser_exec` state persists; `browser_cdp`; CDP dialog supervisor with must_respond/auto_dismiss/auto_accept | **[VERIFIED]** | `browser_use_cli.py:272/1046`, `browser_cdp_tool.py:503`, `browser_supervisor_dialogs.py:46–53` |
| 13 | Cron jobs are agent prompts; fresh session per run; `[SILENT]` suppression | **[VERIFIED]** | `cron/scheduler.py:2538`, `543`, `cron/scheduler_prompt.py:277–281`; prompt injection scan at 407 |
| 14 | Compaction: preflight 50%, gateway 85%, last 20 messages preserved | **[VERIFIED]** | `context_compressor.py:2669` (0.50/3/20), `run_turn_hygiene.py:97/102` (0.85) |
| 15 | Gateway deterministic session keys `agent:{ns}:{platform}:{chat_type}:{chat_id}`; allowlist/pairing default deny | **[VERIFIED with nuance]** | `gateway/run.py:2906`; `access_policy_mixin.py:43–54` requires explicit env opt-in for `open`; `authz_mixin.py:855` default-denies ambiguous cases |
| 16 | Webhook routes on `0.0.0.0:8644` by default, HMAC, 30/min, 1 h dedupe, 1 MB cap | **[PARTIAL]** | Port 8644, HMAC schemes, 30/min, 3600 s, 1,048,576 B all verified at `webhook.py:57–76/236–242`. But default host is `None` (binds both IPv4/IPv6 all interfaces), not literal `0.0.0.0`; `INSECURE_NO_AUTH` is refused on non-loopback binds (255). |
| 17 | ≥64k context requirement for tool use | **[VERIFIED]** | `agent/model_metadata.py:281` `MINIMUM_CONTEXT_LENGTH = 64_000`; Ollama guard `conversation_loop.py:418–445`; local-model caching guard `model_metadata.py:651–675` |
| 18 | One active pluggable memory/context engine; directory plugins with `plugin.yaml`, MCP, pip entry points | **[VERIFIED]** | `memory_provider.py:1–5`, `context_engine.py:1–8`, `plugin_loader.py`, `plugins.py:531/637`, `plugin_validate.py:99` |
| 19 | Hermes README claims agentskills.io compatibility; authoring page does not | **[UNVERIFIED in source]** | Not testable from the clone; frontmatter shape (`metadata.hermes`) is verified, external standard compatibility was not re-checked |
| 20 | Adapter counts "20–25+" | **[PARTIAL]** | 22 enum values + 26 bundled platform plugin directories (overlapping); "20–25+" is a reasonable snapshot but no single canonical count exists |
| 21 | Catalog installs with commit pinning/security scanning | **[PARTIAL]** | Security scanning/quarantine verified (`skills_hub_install.py:64–236`, `skills_guard.py`); exact commit pinning of every source was not exhaustively verified |

---

## 3. Design weaknesses, bugs and sharp edges visible in source (things LEO should NOT copy)

1. **Two parallel tool dispatch pipelines with duplicated policy.** `agent/tool_executor.py::_dispatch_authorized_once()` (676) and `model_tools.py::handle_function_call()` (906) each implement their own pre-hook/guard sequence. Any new policy added to one path can be missing from the other. LEO should have exactly one dispatch choke point (`leo/tools/executor.py` already is one — keep it that way).
2. **Plugin pre-hooks fail open.** `tool_executor.py:660–673` docstring: "Hook failures never block"; `model_tools.py:811–813` swallows hook exceptions. A security plugin that crashes stops enforcing. ACP edit approval partially compensates only for `write_file`/`patch` (`model_tools.py:823–826`). LEO should fail closed on errors from any security hook.
3. **Approval is explicitly a heuristic, not a boundary.** Upstream `SECURITY.md:137–153`: "Shell is Turing-complete; a denylist over shell strings is structurally incomplete... catches cooperative-mode mistakes, not adversarial output." `approval_detection.py:420 hardline_scan_incomplete()` exists because scans can be truncated/budget-limited; those commands are human-only. LEO must never treat command regexes as a sandbox.
4. **Unattended auto-approve outside recognized contexts.** `approval.py:1432–1439`: "Outside CLI/gateway/ask flows we never block on approvals ... return _approved()". Known `-q`/cron/platform contexts default to `deny` (`config_defaults.py:1626`), but a new unrecognized non-interactive surface silently auto-approves. LEO should default-deny for *all* unknown contexts.
5. **execute_code is a documented escape from terminal approval.** `approval.py:1480–1490`: a script can call `subprocess`/`os.system`/`ctypes` directly; only the whole script is approved once, and a purely local non-interactive session auto-approves. `SECURITY.md:70–88`: terminal backend isolation does **not** confine execute_code, MCP subprocesses, plugin loading, hooks or skill loading. `SANDBOX_ALLOWED_TOOLS` even includes `terminal` and `write_file` (`code_execution_tool.py:40`). LEO should not offer an arbitrary-code tool without OS-level isolation.
6. **Container guard bypass is config-declared.** `_should_skip_container_guards()` trusts a plugin backend's `skip_container_guards` flag (`approval.py:1282–1296`), and Docker's host-access detection only inspects bind mounts (`terminal_tool.py:145–164`); `docker_forward_env`/`docker_env` can inject credentials and the default `docker_network=True` remains (`terminal_tool_backends.py:63`). A misconfigured or malicious backend can disable prompts while retaining reachability.
7. **Malformed skill frontmatter fails open into the prompt.** `prompt_builder._parse_skill_file()` (1161) catches any exception and returns `(True, {}, "")` — the skill stays visible as a compatible entry with no description. Combined with naive YAML fallback (`skill_utils.py:86–90`) and regex-only Skills Guard, install-time scanning is advisory (upstream's own `SECURITY.md:147–153` says skills execute arbitrary Python and operator review is the boundary).
8. **Tool availability can be advertised while the backend is down.** `_CHECK_FN_FAILURE_GRACE_SECONDS = 60` and grace-served last-good `True` in `tools/registry.py:256–275, 389–480`; the model may see a tool whose Docker/Modal/Playwright backend just failed. LEO should mark such tools unavailable immediately or surface degraded status in the tool result.
9. **Plugin tool override is fail-soft.** `registry.register()` (876–906) logs an error and `return`s when a plugin shadows a tool without `override=True`; with operator opt-in it can replace a built-in wholesale. Silent rejection is confusing; silent replacement is worse. LEO should surface both events to the operator.
10. **Webhook ingress combines public bind, payload-derived prompts and route scripts.** Default host `None` binds all interfaces (`webhook.py:57`), each authenticated route can shell out via `script` (~733), and rendered payload text becomes an agent prompt. HMAC/rate limit/dedupe are good, but a leaked route secret yields code-capable prompt injection. LEO should bind loopback by default and separate "render prompt" from "run script" permissions.
11. **SSRF checks are TOCTOU-prone and policy-relaxable.** `browser_url_policy.url_is_private()` (69) resolves DNS when checking, not when the browser connects; `auto_local` and `browser.allow_private_urls` relax the private floor (117–132). CDP tools try to compensate by checking JS expression literals and the current page (`browser_cdp_tool.py:137–175`), but a redirect or rebinding can still bypass. LEO should enforce egress at the network layer.
12. **Context files are an untrusted-input surface.** `prompt_builder.discover_context_files()` (1685) ingests repo-supplied `AGENTS.md`/`CLAUDE.md`/`.cursorrules` into the system prompt; the scanner (84–110) blocks only known patterns and explicitly loads user-authored `SOUL.md` anyway. Prompt content is data, but in practice it steers the loop. LEO should add structural provenance markers and strip instructions from third-party files.
13. **Complexity as a reliability hazard.** 12,636 Python files; `context_compressor.py` alone is 5,215 lines; approval detection contains thousands of lines of regex/normalization; `hermes_state.py` is a 655-line facade over a dozen persistence packages. This is the accumulated cost of years of hotfixes (comments reference hundreds of issue numbers). LEO should copy *interfaces and invariants*, not this implementation style.
14. **Doc/source drift.** The research doc's toolset and delegation-default numbers are already wrong at this commit; any LEO plan that hard-codes those numbers will be wrong. Pin the SHA and re-derive from code.

---

## 4. Ranked improvements for LEO (concrete, with effort and verification)

Current LEO surfaces consulted for tailoring: `<repo>/leo/agents/engine.py`, `leo/tools/executor.py`, `leo/tools/registry.py`, `leo/security/permissions.py`, `leo/models/context.py`, `leo/memory/store.py`, `leo/skills/registry.py`, `leo/skills/compatibility.py`, `leo/tasks/scheduler.py`, `leo/integrations/gateway.py`. LEO already has a single `ToolExecutor` choke point, an `PermissionManagerImpl`, a `ContextOptimizer`, an SQLite memory store and a `SKILL.md` translator — the recommendations below extend those rather than importing Hermes code.

1. **Single dispatch choke point + fail-closed hooks — Effort: S (1–2 days).** Make every tool call (chat, cron, subagent, webhook, MCP, code tool) go through `leo/tools/executor.py::ToolExecutor`; have permission/hook exceptions produce a denial, never a pass. Add a registry-level assertion test that no other module calls tool handlers directly. *Verify:* a unit test with a throwing security hook must deny; grep-based CI check for direct handler calls; compare behavior across all entry points with the same dangerous payload. *Why first:* LEO already has the choke point; Hermes' dual-path drift (#1, #2 above) is the clearest lesson.

2. **Unattended-context policy matrix + non-bypassable floors — Effort: M (2–4 days).** Define explicit modes (interactive / cron / webhook / agent-child / unknown) with default `deny` for unknown, non-bypassable hard floors (operator deny rules, runtime self-delete, cloud metadata), and pattern-keyed session/permanent approvals instead of LEO's current exact `(action, resource)` approval. Fix `<repo>/leo/security/permissions.py:74–81`: a prior approval currently **clears a path violation** (`violation = None`), so an approval can legitimize a path outside allowed roots — path violations must remain unconditional. *Verify:* table-driven tests for every (mode × risk × floor) pair; regression test that an approved action on an out-of-root path is still denied; audit log completeness.

3. **Three-tier prompt assembly with cache-stable ordering — Effort: M (3–5 days).** Split LEO's system prompt into `stable` (identity/guidance), `context` (project files/workspace), `volatile` (skills/memory/time/env) in `leo/models/` or `leo/core/app.py`, with an explicit rebuild boundary after compaction. Mark injected content as data with provenance. *Verify:* hash the three tiers across two turns with unchanged volatile inputs — stable must be byte-identical; cache-hit metric in provider tests; injection test using a fake `AGENTS.md`.

4. **Compaction with protected tail, deterministic fallback and pre-compress memory checkpoint — Effort: M–L (5–8 days).** LEO's `ContextOptimizer._compact()` (`leo/models/context.py:138`) keeps only the last two non-system messages and refuses to synthesize a fallback if the provider fails. Adopt: protected head (system + first N), protected tail (last ~20 or budget-capped), `prune_tool_results_only` before LLM summarization, deterministic fallback summary, and a provider hook whose failure aborts compaction (keeps full transcript). *Verify:* force summarizer 500/timeout and assert a usable summary exists; assert the protected tail is present verbatim; assert no tool result is lost from the persisted transcript (only from the request).

5. **Typed memory-provider and context-engine interfaces with per-profile isolation — Effort: M (4–6 days).** Mirror `MemoryProvider` / `ContextEngine` hook shapes (init, prefetch, sync_turn, pre_compress checkpoint, on_memory_write, clone_for_agent, select_context) but keep ONE implementation active and enforce per-agent instances — no module-level profile state. Threads must copy contextvars. *Verify:* two simulated profiles with different memory providers run interleaved and never cross-read; checkpoint provider failure aborts compaction; plugin engine raising in a hook degrades to the built-in, not to an empty prompt.

6. **Subagent budget/isolation contract — Effort: M (3–5 days).** LEO's `leo/agents/engine.py` already delegates; add the Hermes-style invariants: blocked tools (`delegate_task`, `clarify`, shared-memory writes, messaging/scheduling), child toolsets ⊆ parent toolsets, depth-derived nesting role, dedicated session handle, explicit `max_iterations` cap (choose a deliberate value; do **not** copy Hermes' 250 as a "documented default"), and only a final summary returned. *Verify:* child asking for a broader toolset is rejected; child attempting `memory.write` gets a blocked tool result; nested depth cap holds; child dies with parent but its transcript is flushed first.

7. **Cron/webhook hardening before enabling remote triggers — Effort: M (4–6 days).** Loopback default bind, HMAC with pinned signature family, rate limit, body cap, replay dedupe; scan the *assembled* prompt (including skill content) before a non-interactive agent runs; deny `messaging`/`clarify`/scheduling tools to cron children; use fresh sessions per run. Do not add a route-level "run shell script on payload" feature. *Verify:* unsigned/replayed/oversized requests rejected; a malicious skill with injected cron instructions blocks the run; outbound message tool absent from cron tool surface.

8. **Browser/SSRF boundary at the network layer — Effort: L (1–2 weeks).** If LEO's browser integration grows beyond live tests, put egress filtering in a proxy/firewall (block loopback, link-local, RFC1918, CGNAT, cloud metadata 169.254.169.254) rather than in URL string checks; add a resource-name-based secret-in-URL guard and per-task CDP supervisor for native dialogs (default `must_respond`). *Verify:* DNS-rebinding test (`a.example` resolves public then 127.0.0.1), redirect test, metadata-URL test, dialog-timeout test.

9. **Tool-schema budgeting via deferred tool search — Effort: S–M (2–3 days).** LEO's tool registry is small today; add `tool_search`/`tool_describe` bridge only when schema tokens exceed a configured fraction of context, and keep core safety tools eager. *Verify:* with 100 fake tools, request token count stays under budget and a deferred tool can still be discovered and invoked; safety tool never deferred.

10. **Source-pinned capability tracking — Effort: S (1 day).** Record the upstream SHA in LEO research notes; when citing a capability, cite a symbol and line, not a docs page. Add a quarterly re-verification note. *Verify:* this report's claim table can be regenerated mechanically for a new SHA; no LEO design decision depends on an unverified doc number.

**Explicitly do NOT copy:** in-process plugin loading as the only isolation mode; regex-only approval as a security boundary; `execute_code` with arbitrary subprocess access on the host; public webhook binds with payload-executed scripts; naive YAML fallback in skill parsing; dual tool-dispatch paths; last-good `True` availability caching for security-relevant tools; approval that can override a path-root violation.

---

## 5. Unverified / out of scope in this pass

- No runtime execution of Hermes (no API keys, no paid calls, no live terminal/browser tools); all conclusions are static source analysis.
- Exact runtime tool count with all plugins/MCP enabled is configuration-dependent and was not enumerated.
- Provider behavior, latency, sandbox escapes, browser rendering, plugin marketplace trust and agentskills.io compatibility were not tested.
- OpenClaw was not re-examined here; the prior doc's OpenClaw section remains doc-level only.
- GitHub REST metadata (stars/issues) was not re-queried; the prior snapshot stands.

## 6. Source URL index (all pinned to `b076b5856cc6b74ab815ea97d866ee5022a5282a`)

- [AGENT loop](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/conversation_loop.py#L1615)
- [System prompt tiers](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/system_prompt.py#L770)
- [Tool registry](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/registry.py#L840)
- [Toolsets](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/toolsets.py#L67)
- [Executor policy choke point](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/tool_executor.py#L676)
- [Approval core](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/approval.py#L1402)
- [Approval detection patterns](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/approval_detection.py#L498)
- [Skills frontmatter](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/skill_utils.py#L71)
- [Memory provider ABC](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/memory_provider.py#L84)
- [Context engine ABC](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/context_engine.py#L60)
- [Context compressor defaults](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/agent/context_compressor.py#L2669)
- [Delegation blocked tools](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/delegate_tool_toolsets.py#L13)
- [Terminal backends](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/terminal_tool_backends.py#L294)
- [CDP supervisor](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/browser_supervisor.py#L98)
- [Browser URL policy](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/tools/browser_url_policy.py#L117)
- [Gateway authorization](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/gateway/authz_mixin.py#L855)
- [Webhook ingress](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/gateway/platforms/webhook.py#L57)
- [Cron scheduler](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/cron/scheduler.py#L2525)
- [Fallback chain](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/hermes_cli/fallback_config.py#L139)
- [Plugin context API](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/hermes_cli/plugins_context.py#L587)
- [Upstream trust model / caveats](https://github.com/NousResearch/hermes-agent/blob/b076b5856cc6b74ab815ea97d866ee5022a5282a/SECURITY.md#L137)

---

## 7. Additional verified claims (second pass)

| # | Claim | Label | Evidence |
|---|---|---|---|
| 22 | Session retention default 90 days, auto-pruning off by default | **[VERIFIED]** | `hermes_cli/config_defaults.py:2226 "auto_prune": False`, `:2228 "retention_days": 90`; `session_retention.py:170 run_auto_prune_and_vacuum()` enforces the 90-day default when enabled |
| 23 | `message_uid` tracking for messages | **[VERIFIED]** | `hermes_state.py:431 append_message(..., message_uid: Optional[str] = None)` forwarded to persistence |
| 24 | Authenticated `/api/ws` uses short-lived single-use loopback tickets | **[PARTIAL]** | `gateway/control_socket.py:351/382–409` mints `session-ticket` with `expires_in_seconds: 30`, bound to peer subject and listener instance; single-use semantics implemented in the ticket store (not exhaustively audited here) |
| 25 | `/handoff` moves a CLI session to a platform | **[VERIFIED]** | `gateway/handoff_target.py:1–29 resolve_handoff_home()` validates a deliverable home channel before queueing |
| 26 | Plugins can run in a separate host process (`plugins.isolation: host`) rather than only in-process | **[VERIFIED]** | `hermes_cli/plugin_isolation.py:32/101`, `hermes_cli/plugin_host.py`, `plugin_host_child.py`, `plugin_host_wire.py`; some registration surfaces are explicitly refused in host mode (38–43) |
| 27 | Anthropic cache breakpoints / prompt-cache boundary management | **[VERIFIED]** | `agent/prompt_cache_scope.py`, `agent/prompt_cache_boundary.py`, `agent/prompt_caching.py`, `agent/turn_request_assembly.py:112` |

### Raw command/output appendix (second pass)

```
$ grep -nE '"auto_prune"|"retention_days"|"sessions":' $TMPDIR/leo-research-hermes-src/hermes-agent/hermes_cli/config_defaults.py | head -15
502:        "auto_prune": True,
503:        "retention_days": 7,
2218:    "sessions": {
2226:        "auto_prune": False,
2228:        "retention_days": 90,

$ grep -rn 'ticket' $TMPDIR/leo-research-hermes-src/hermes-agent/gateway/control_socket.py | head -6
2:versioned JSON verbs (``identify``, ``status``, ``resolve-profile``, ``session-ticket``, ...). ...
351:            elif verb == "session-ticket":
382:    def _session_ticket(self, request: dict, peer_subject: Optional[str]) -> dict:
408:        return {"ticket": ticket, "expires_in_seconds": 30,
409:                "instance_id": self.ticket_store.instance_id,

$ grep -rn 'MINIMUM_CONTEXT_LENGTH' $TMPDIR/leo-research-hermes-src/hermes-agent/agent/model_metadata.py | head -2
281:MINIMUM_CONTEXT_LENGTH = 64_000

$ grep -nE '^(class |def |async def )' $TMPDIR/leo-research-hermes-src/hermes-agent/agent/subagent_lifecycle.py | grep -E 'launch|status|wait|cancel|result|reconnect|steer'
277:    def launch(self, request: SubagentLaunchRequest | Mapping[str, Any]) -> SubagentHandle:
323:    def status(self, handle: SubagentHandle | Mapping[str, Any]) -> SubagentStatus:
330:    def wait(self, handle: SubagentHandle | Mapping[str, Any], *, timeout_seconds: Optional[float] = None) -> SubagentTerminalState:
344:    def cancel(self, handle: SubagentHandle | Mapping[str, Any], *, reason: str) -> SubagentCancelResult:
362:    def result(self, handle: SubagentHandle | Mapping[str, Any]) -> SubagentResult:
369:    def reconnect(self, handle: SubagentHandle | Mapping[str, Any]) -> SubagentReconnectResult:
376:    def steer(self, handle: SubagentHandle | Mapping[str, Any], text: str) -> bool:

$ git -C $TMPDIR/leo-research-hermes-src/hermes-agent status --short
(empty — cloned tree unmodified)
```

### 7.1 Concrete evidence of approval-scanner fragility (static only; probe NOT executed)

The clone contains an upstream regression probe at `evals/postmortem/review_probes/scanner_bypass_probe.py`. Its docstring states that a hardline command hidden after a NEWLINE inside a double-quoted `$(grep …)` was approved with zero approval-callback calls in an earlier version (issue #103492 / #103563), and that it reproduces the case using a harmless shadowed command and a marker file. Its presence in the shipping tree is direct evidence that the shell-string denylist is an actively maintained, incomplete heuristic, not a boundary. Supporting scanner code: `tools/approval_detection.py::_command_parser_limit_exceeded()` (1084), `hardline_scan_incomplete()` (420), and the deobfuscation/word-span helpers around 1118–1519. **I did not run this probe** (it imports and executes code from the untrusted clone and shells out); the citation is documentary only.

Implication for LEO: if a shell command is dangerous enough to need approval, LEO should not evaluate it with string matching at the final authority layer — use an OS boundary (container/VM/separate user), and treat any regex gate as UX/friction plus audit signals.

### 7.2 Third-pass claim checks

| # | Claim | Label | Evidence |
|---|---|---|---|
| 28 | SQLite `state.db` in WAL mode | **[VERIFIED]** | `hermes_cli/config_defaults.py:32–35` `database.journal_mode = "wal"` default (documented exception for network filesystems: use `delete`) |
| 29 | Session lineage via compression tip / parent links; `/branch` | **[VERIFIED]** | `persistence/{core,sqlite}/conversation/lineage.py` (`tip()`, `lineage()`, `chain()` at sqlite lines 116/120/125); `hermes_cli/cli_init_mixin.py:479 _new_session_id(branch_of=...)`; `hermes_cli/gateway_chat_commands.py:234` `/branch [title]` |
| 30 | Hook/plugin registry exposes "single-select" memory and context engine slots | **[VERIFIED]** | `agent/memory_provider.py:1–5` ("ONE external provider at a time"), `agent/context_engine.py:1–8` ("one is active"); selection applied in `agent/conversation_loop.py:1326` and `agent/agent_init_memory_provider.py:57` |


### 7.3 LEO-side delta observed while tailoring recommendations (read-only)

- `<repo>/leo/security/permissions.py` (approval block around lines 95–101 at final check; was 74–81 when first read — LEO was being edited concurrently by other work): an `approval_id` matching action/resource sets `allowed=True` and **`violation = None`**, erasing a previously detected path-root violation. Keep path violations unconditional (recommendation 2).
- `<repo>/leo/tools/executor.py:186–205`: authorization occurs once per call at the executor choke point — good; keep this single-path property as Hermes-scale features are added.
- `<repo>/leo/models/context.py:138–160`: `_compact()` keeps only the last 2 non-system messages and returns `None` if the summarizer fails; no deterministic fallback, no protected 20-message tail (recommendation 4).
- `<repo>/leo/tasks/webhooks.py:131–147`: HMAC-SHA256 verification with `hmac.compare_digest` and multiple provider header names is already present; reuse it and add Hermes' replay dedupe/rate limit/1 MB cap rather than copying its script-on-payload feature.
- `<repo>/leo/agents/engine.py:373 delegate()` and `:855 _make_delegator()`: delegation exists; add explicit toolset-subset, blocked-tool, iteration cap and depth cap invariants (recommendation 6).


### 7.4 Cheap pattern worth copying: tool-loop guardrails

`agent/tool_guardrails.py` (`ToolCallGuardrailConfig` 109) has a well-scoped design LEO can adopt without importing Hermes' complexity: warnings for repeated identical failures (2), blocks at 5, same-tool failure halt at 8, idempotent no-progress block at 5, per-turn hard caps on `web_search`/subagent spawns (`LoopCapConfig` 93), and **hard stops default-on only for non-interactive platforms** (`non_interactive_hard_stop_enabled=True`, 129–142). LEO's agent loop (`leo/agents/engine.py`) currently relies on generic iteration limits; adding per-tool repeat detection plus non-interactive hard stops is a small, high-value change. *Verify:* scripted 6 identical failing tool calls must warn then block; interactive sessions only warn; cron sessions halt and produce an auditable terminal result.

### 7.5 Link integrity check

```
$ python3 -I $TMPDIR/leo-research-tools/check_links.py <repo>/research/deep-2026-11-18/hermes-source.md $TMPDIR/leo-research-hermes-src/hermes-agent
pinned URLs: 32 wrong SHA: 0 missing local paths: 1
  MISSING ('<path>', '')
```

The single "missing" path is the literal `<path>` placeholder in the permalink-template line of section 0; all 31 real file URLs point to files present at the examined commit and use the pinned SHA.

### 7.6 Provider failover mechanics (fourth pass)

`agent/turn_recovery.py::_fallback_break()` (~1856) switches provider by calling `conversation_loop._arm_fallback_restart()` (1232), which sets `restart_with_rebuilt_messages=True`, resets retry and compression counters, then breaks to `apply_retry_restarts()` (`conversation_loop.py:1744`). `_LoopState.restart_count` (1469) is a backstop against unbounded redirect/rebuild loops. Failover is therefore a *rebuilt-request restart*, not a silent token-level handoff — a useful invariant for LEO's router (`leo/models/router.py`) if multi-provider fallback is added. Note the sharp edge: rebuilt messages can invalidate a provider-specific prompt-cache prefix; Hermes compensates with `_sync_failover_system_message()` / `_redecorate_prompt_cache_for_provider()` (`conversation_loop.py:1219/1257`).

### 7.7 Progressive skill disclosure (fifth pass)

`tools/skills_tool.py:1–5` states the pattern: `skills_list` returns name/description only; `skill_view` returns full `SKILL.md` or a linked file; usage is bumped on view (`_skill_view_with_bump`, ~771) and byte-identical repeat views collapse to a short stub (dedup cache cleared on compression). The system-prompt index (`prompt_builder._render_skills_index`, 1303) is just a catalog of names + ≤60-char descriptions, so skill bodies are paid for only when used. LEO's `leo/skills/registry.py` stores SKILL.md descriptions in SQLite but has no `skill_view`-style full-text tool; adding one is a cheap context saving. *Verify:* a skill with a 50 KB body adds only its catalog row to the initial prompt; after `skill_view`, the body is in context; a second identical view returns a stub.

### 7.8 Parallel tool calls: more nuanced than the docs (sixth pass)

`agent/tool_dispatch_helpers.py` — `_should_parallelize_tool_batch()` (263), `_is_mcp_tool_parallel_safe()` (126), `_extract_parallel_scope_paths()` (307). The planner admits calls to parallel execution by tool name plus path-scope checks (e.g. two writes to the same file are serialized); `agent/tool_executor.py:1953 execute_tool_calls_segmented()` runs segments as parallel/sequential. `agent/tool_executor.py:133 _MAX_TOOL_WORKERS = 8` and a per-batch deadline (187) bound fan-out. Documentation's simple "concurrent except interactive tools" is a simplification; LEO should implement name/scope-based parallel-safety rather than a flat interactive/non-interactive split.

### 7.9 Tool-result spill/retention pattern (seventh pass)

`tools/tool_result_storage.py` — `maybe_persist_tool_result()` (418) moves large tool output to a spillover file and keeps a bounded preview (`generate_preview`, 272); `enforce_turn_budget()` (517) spills the largest text parts; `cleanup_spillover_cache()` (127) applies retention; `tools/spill_safety.py::ensure_owned_private_dir()` (42) and `open_exclusive()` (61) enforce private, owned directories and exclusive writes. LEO's executor currently redacts/truncates output in memory only; adopting this pattern (private dirs, retention, preview + pointer) prevents both context blowups and cross-session leakage of spilled output.
