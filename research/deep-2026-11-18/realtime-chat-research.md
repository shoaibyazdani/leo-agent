# Real-time chat UI research: Hermes Agent and OpenClaw

**Date:** 2026-11-18. **Purpose:** inform the redesign of LEO's chat surface (live typing/streaming status, tool calls as they happen, attachments). **Scope:** read-only inspection of pinned upstream checkouts and LEO's current code/docs. No LEO source, test, config, or upstream file was modified.

## 0. Sources and method

- Hermes Agent pinned checkout: `$TMPDIR/leo-research-hermes-src/hermes-agent/` (on macOS `/tmp` resolves to `/private/tmp`; both paths work). OpenClaw pinned checkout: `$TMPDIR/leo-research-openclaw-src/repo/`. E2E checkouts under `$TMPDIR/leo-research-*-e2e/` were also present; the source checkouts are the authority for wire schemas. All upstream content was treated as untrusted data: files were only read, never executed.
- Prior research read first to avoid duplication: `<repo>/docs/RESEARCH_HERMES_OPENCLAW.md`, `<repo>/research/deep-2026-11-18/hermes-source.md`, `hermes-e2e.md`, `openclaw-source.md`, `openclaw-e2e.md`, `SYNTHESIS.md`, `security-licensing.md`, and the draft plan `<repo>/docs/CHAT_REALTIME_PLAN.md`.
- Prior reports already established: Hermes CLI `--format stream-json` NDJSON (`system/init`, `tool_use`, `tool_result`, `text`, `result`), provider-level SSE `chat.completion.chunk`, and an authenticated `/api/ws`; OpenClaw's port 18789 WebSocket handshake with `connect.challenge`, Ed25519 device identity and scope caps. This report focuses on the layers those reports did not cover: **UI transport, UI-facing event names, rendering behavior, and attachments**.
- Every claim below has an absolute file path and line number. "NOT FOUND" means no evidence was found in the pinned checkout; it is not proof of absence in every version.

---

## 1. Streaming transport

### 1.1 Hermes Agent — two transports, neither is SSE to the browser

| Layer | Transport | Exact endpoint / event names |
|---|---|---|
| Model provider | SSE over HTTP (`POST /v1/chat/completions`, `stream: true`), `chat.completion.chunk` + `[DONE]` | confirmed at runtime in `hermes-e2e.md:319,355-360` |
| CLI machine output | NDJSON on stdout via `--format stream-json` | `system/init`, `tool_use`, `tool_result`, `text`, `result` (`hermes-e2e.md:362-380`) |
| Web dashboard chat | **WebSocket `/api/pty`** carrying raw PTY bytes; the real Ink TUI runs inside it | `$TMPDIR/leo-research-hermes-src/hermes-agent/hermes_cli/web_routers/chat_ws.py:475-510`; client at `web/src/pages/ChatPage.tsx:1222-1232,1332-1353`; architecture documented at `web/AGENTS.md:7-28` |
| Dashboard tool/status feed | **WebSocket `/api/pub` → `/api/events`**, joined by `?channel=`; `/api/pub` rebroadcasts publisher frames verbatim | `chat_ws.py:710-760`; client `web/src/components/ChatSidebar.tsx:5-25,483-517`; `web/src/lib/eventsFeedClient.ts:68-76` |
| Desktop app / JSON-RPC sidecar | **WebSocket `/api/ws`** (JSON-RPC), plus `/api/console` | `chat_ws.py:650,241`; `apps/desktop/src/app/gateway/hooks/` |
| REST chat-adjacent | `POST /api/chat/image-upload`, `POST /api/chat/file-upload`, `GET /api/sessions/{id}/messages` | `web_routers/files.py:695`, `web_routers/uploads.py:22`, `web/src/lib/api.ts:459` |

The web dashboard deliberately does **not** reimplement chat rendering: it embeds the terminal UI over a PTY and separately consumes an event mirror for the sidebar. No `EventSource`, `text/event-stream`, or streaming HTTP `POST` chat route was found in the dashboard or desktop UI. The `/api/pty` stream is byte-oriented, with one JSON control frame `{"type":"resume","id":...}` and a resize escape `\x1b[RESIZE:<cols>;<rows>]` (`web/AGENTS.md:17-18`).

Gateway event envelope (the desktop app's structured feed):

```
{"jsonrpc":"2.0","method":"event",
 "params":{"type":"message.delta","session_id":"...","payload":{...}}}
```

Definitions: `$TMPDIR/leo-research-hermes-src/hermes-agent/tui_gateway/server.py:575-578`; per-session `seq` added by `tui_gateway/event_replay.py:91-106`; typed envelope in `apps/shared/src/gateway-events.ts:56-83`; ~30 ms coalescing in `tui_gateway/ws.py:125-131,177-200`. The dashboard sidecar mirror (`/api/pub`) sends **newline-framed JSON without the JSON-RPC envelope** (`tui_gateway/event_publisher.py:1-8`; installed at `tui_gateway/entry.py:87-94`).

Exact Hermes event names (`tui_gateway/contracts/events.py`):

- Text/turn: `message.start` (:150-151), `message.delta` (:154-178; fields `text`, optional `rendered`/`verbose`, plus `turn_id`, `segment`, `offset` for dedupe), `message.interim` (:184-195), `message.complete` (:281-305; includes `usage`, `status`, `partial`, `error_surface`, `response_reused`).
- Reasoning: `reasoning.delta`, `reasoning.available`, `thinking.delta` (:179-181).
- Status: `status.update` (:308-315; `kind`, `text`).
- Tools: `tool.generating` (:436-442), `tool.start` (:399-413; `tool_id`, `name`, `context`, `args`, `args_text`, `preview`, `labels`), `tool.progress` (:458-470), `tool.complete` (:416-433; `tool_id`, `name`, `args`, `duration_s`, `result`, `summary`, `result_text`, `inline_diff`, `todos`, `image_path`), `todo.updated` (:473-480), `approval.cancelled` (:559-576).
- **Approval is a server→client JSON-RPC request, not an event**: method `approval`, params `request_id`, redacted `command`, `description`, `choices` ∈ `once|session|always|deny`, `tool_name` (`tui_gateway/contracts/server_requests.py:166-196`; frame at `tui_gateway/server_requests.py:97-98`). Client methods: `approval.pending`, `approval.received`, `approval.respond` (`contracts/prompt_voice.py:305-345`).

NOT FOUND as Hermes event names: `typing`, `agent.thinking`, `chat.completion`, `message.append`, `tool_call`, `tool_result` (the last two exist only in the CLI NDJSON format as `tool_use`/`tool_result`).

Auth for the dashboard: REST uses `X-Hermes-Session-Token` (`web/src/lib/api.ts:105-109,164-177,299-333`; server `hermes_cli/web_server.py:299-310`). WebSocket upgrades cannot set headers, so gated mode mints a **single-use 30 s ticket** from `POST /api/auth/ws-ticket` and appends `?ticket=`; loopback appends `?token=` (`web/src/lib/api.ts:261-296`; route `hermes_cli/dashboard_auth/routes.py:583`; validation `hermes_cli/web_server_chat.py:204-238,263-298`; closes 4401/4403 at `chat_ws.py:102-113`). This is a useful pattern if LEO ever needs SSE via `EventSource`, which cannot set an `Authorization` header.

### 1.2 OpenClaw — one gateway WebSocket, JSON-RPC methods plus named events

The Control UI is a SolidJS app at `$TMPDIR/leo-research-openclaw-src/repo/ui/src/pages/chat/`, with API glue in `ui/src/api/`. It opens a `new WebSocket(transportUrl)` (`ui/src/api/gateway-browser-socket.ts:93`); the only `fetch` in that module is a reachability probe (:61). **SSE for chat: NOT FOUND** (`EventSource`/`text/event-stream` hits are test fixtures or provider/cron code).

Client methods:

- `chat.history` / `chat.startup` — `ui/src/pages/chat/chat-history.ts:90`
- `chat.send` — params schema `ChatSendParamsSchema` at `packages/gateway-protocol/src/schema/logs-chat.ts:404-453`, requires an `idempotencyKey`
- `sessions.messages.subscribe` / `unsubscribe` — `ui/src/api/gateway-chat-events.ts:35`
- `session.typing` — `ui/src/pages/chat/chat-pane-sharing.ts:642`; server publication `src/gateway/server-methods/sessions-typing.ts:221`
- `approval.resolve` / `exec.approval.resolve` / `plugin.approval.resolve` — `ui/src/app/exec-approval.ts:260-275`

Events consumed by the Chat UI: `chat`, `agent`, `session.tool`, `session.operation`, `session.observer` (`ui/src/pages/chat/chat-live-events.ts:34-42`), plus `sessions.changed`, `session.message` (`ui/src/api/gateway-chat-events.ts:60-71`). Broader gateway vocabulary in `src/gateway/events.ts:20-96`: `session.typing`, `session.approval`, `question.requested`/`resolved`, `exec/plugin/openclaw.approval.requested`/`resolved`, `presence`, `progressCard.changed`. NOT FOUND as wire names: `chat.delta`, `message.updated`, `agent.thinking`, `tool.call`, `tool.result`, `chat.subscribe`, `chat.status`. Thinking and tool activity are **stream values inside `agent` events**, not separate events.

`chat` event union (`packages/gateway-protocol/src/schema/logs-chat.ts:477-632`), discriminated by `state`, with common fields `runId`, `sessionKey`, optional `agentId`/`spawnedBy`, and integer `seq`:

- `state:"status"` — `phase` from an 11-value startup vocabulary (`waiting_for_state`, `preparing_worktree`, `preparing_context`, `starting_model`, `waiting_for_model`, …), optional `waitReason: "response"|"slot"`, `retry` (`attempt`, `maxAttempts`, `reason`, `retryAt`) (:511-530).
- `state:"delta"` — `deltaText: string`, optional `replace: bool` (full-content refresh), `message`, `usage` (:532-540).
- Terminals: `final` (`message`, `stopReason`, `yielded`), `aborted` (`localModelMayStillBeFinishing`), `error` (`errorKind` ∈ refusal/timeout/rate_limit/context_length/state_contention/unknown, sanitized `errorDetail`) (:542-632).

`agent` event schema: `{runId, seq, stream, ts, data, spawnedBy?, isHeartbeat?, preamble?}` (`packages/gateway-protocol/src/schema/agent.ts:61-74`; runtime subtype `src/infra/agent-events.types.ts:22-56`). Streams include `tool`, `item`, `thinking`, `lifecycle`, `usage`, `compaction`, `plan`, `approval`.

### 1.3 Transport verdict for LEO

Neither upstream uses SSE for the browser chat: Hermes uses a PTY WebSocket plus an event-mirror WebSocket; OpenClaw uses a typed gateway WebSocket with method calls and events. For a **local, single-user** app, that machinery is not justified. SSE (already implemented in LEO) is sufficient for one-way turn events, provided a separate POST is used for commands (send, resolve approval, cancel). Choose `fetch` + `ReadableStream` over `EventSource` in LEO: it can send the bearer token, support `AbortController` for a Stop button, and report stream errors; the cost is writing a small frame parser instead of using `EventSource`.

---

## 2. Progressive text, thinking/typing, tool calls

### 2.1 Hermes rendering

- TUI: `message.delta` → `recordMessageDelta` (`ui-tui/src/app/createGatewayEventHandler.ts:1817-1818`); the controller **appends to a buffer and replaces the whole rendered block with the accumulated text**, explicitly because an earlier `rendered`-only full replace dropped prior text (`ui-tui/src/app/turnController.ts:812-829`); paint is throttled (:1068-1078). `text: null` deltas are segment boundaries and are dropped (:813-815).
- A tool start flushes the streaming segment before drawing a tool row (`turnController.ts:1008-1023`); tool completion creates result lines and an "analyzing tool output…" status (:930-998); duplicate `tool.complete` rows are skipped (:615).
- Active tools render a `Spinner`, name/context and elapsed seconds (`ui-tui/src/components/thinking.tsx:899-924`); sections can be collapsed/expanded/hidden (`messageLine.tsx:74-76`, `thinking.tsx:759-790`); the live block is assembled at `components/streamingAssistant.tsx:44-63`. `status.update` drives status labels (`createGatewayEventHandler.ts:1298-1326`).
- Desktop React: `message.delta` → `appendAssistantDelta` (`apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/message-stream.ts:264-270`); `message.interim` seals a bubble (:272-286). Deltas are coalesced by contiguous offset (`.../use-message-stream/index.ts:454-479`) and flushed with `appendAssistantTextPart` (:319-348); stamped deltas trim already-applied heads (`segment-identity.ts:47-65`). Tool events map to running/sealed rows (`gateway-event/tools.ts:122-176`), rendered with `aria-expanded` disclosures (`components/assistant-ui/tool/fallback.tsx:1037-1048`). Approval cards show `description` when the command is redacted (`components/assistant-ui/tool/approval.tsx:285-295`).

### 2.2 OpenClaw rendering

- `GatewayChatStreamProjection` keeps one baseline message per `runId`; a `delta` either replaces content (`replace:true`) or appends `deltaText` to the last text block, otherwise pushes a new text block (`packages/gateway-client/src/chat-stream-message.ts:4-45`; projection `chat-stream-projection.ts:19-41`). **If no baseline exists the client flags `missingBaseline` and forces a reconnect** (`ui/src/api/gateway-chat-events.ts:71-77`).
- UI batches `chat` deltas, `agent` and `session.tool` into `animation-frame` publishes; the first token publishes immediately (`ui/src/pages/chat/chat-live-events.runtime.ts:130-151,254-262`); tool-card updates are throttled to 80 ms (`tool-stream.ts:49,118-153`).
- Tool reducer keys by `runId + toolCallId`, rejects at-or-older `seq`, and blocks late non-terminal events after a result (`ui/src/pages/chat/tool-stream.ts:159-226`). Phases: `start` (args), `update` (`partialResult`, `outputOnly`, `diff`, `liveOutput`), `result` (`result`, `isError`, `exitCode`, approval outcome), `input_delta`, `review` (`tool-stream.ts:541-657`). Placeholder cards are created at start and adopt results later (:546-556,592-608); cards are collapsible (`components/chat-message-activity-view.tsx:351-406`, `chat-tool-expanded.solid.tsx:295`).
- Thinking: `stream:"thinking"` updates reasoning only for the active streaming run (`tool-stream.ts:469-483`). Human "typing" is a distinct `session.typing` event/RPC rendered as a three-dot bubble plus draft preview (`components/chat-typing-indicator.tsx:60-99`).
- Approvals/questions: `phase:"review"` tool events carry normalized review records; approvals also arrive as `exec|plugin|openclaw.approval.requested/resolved` with resolve RPCs; questions as `question.requested/resolved` with status filtering and expiry (`ui/src/app/question-prompt.ts:278-306`).
- Late join/reconnect: `sessions.messages.subscribe` with `initialHistory`; the server installs a **provisional observer before** running the history snapshot and then commits it, so events during the snapshot are not lost (`src/gateway/server-methods/sessions-subscriptions.ts:140-149,168-206,227-234`). Catch-up returns `{kind:"delta", messages, deltaCursor, inFlightRun, pendingInputs, …}`; more than 200 events or a byte budget yields `{kind:"reset"}` and the client refetches a tail page (`logs-chat.ts:178-208`). Dedupe uses run/session identity, message id/sequence, `idempotencyKey`, seq fences and terminal recovery (`ui/src/pages/chat/history-merge.ts:78-90,138-160`; `chat-gateway.ts:137-171`; `chat-live-events.runtime.ts:221-240`).

### 2.3 Patterns worth copying

1. **Append, don't replace with the newest chunk** — both UIs accumulate text and re-render the accumulated buffer; OpenClaw additionally supports an explicit `replace` flag for resync.
2. **Monotonic `seq` + run identity** — OpenClaw rejects at-or-older seq and Hermes carries `turn_id`/`segment`/`offset`; this is what makes reconnects and duplicate sockets safe.
3. **Tool rows are first-class, stable objects** — created at `start`, updated in place, sealed at `result`, collapsible, with explicit `isError`/`success` rather than string-sniffing results.
4. **Status is a separate stream from text** — Hermes `status.update`, OpenClaw `state:"status"` with phases and retry metadata; "typing" is its own state, not inferred from silence.
5. **Provisional subscribe before history snapshot** — avoids the classic gap between fetching history and subscribing to live events.
6. **Frame coalescing with immediate first token** — smooth rendering without flooding React/SolidJS state.

### 2.4 Anti-patterns observed upstream

- Hermes global broadcasts use `session_id: ""`; events lacking `seq`/`session_id` bypass dedupe (`apps/desktop/src/app/gateway/gateway-event-dedupe.ts:30-32`).
- Hermes sidebar mirror is explicitly best-effort: 256-item queue, `put_nowait` drops on full, dead WS short-circuits (`tui_gateway/event_publisher.py:28,43-48,68-75`). Two sockets can append every delta twice; a 30 s `(epoch, session_id, seq)` gate is needed (`gateway-event-dedupe.ts:1-34`).
- Hermes has no error flag on `tool.complete`; failure is inferred from result text (`turnController.ts:968-969`). Do not copy this.
- OpenClaw caps live assistant text at 500,000 chars with prefix retirement (`src/gateway/live-chat-projector.ts:23-43`) and tool output at 120,000 chars (`tool-stream.ts:50,97-102`); unbounded UI buffers are a real failure mode even locally.
- Markdown is sanitized before `innerHTML` at `ui/src/components/markdown.ts:224-229`; assistant/tool text must never bypass that. LEO's `app.js:25` has an `esc()` helper but `renderChat()` uses `innerHTML` for the whole thread (`app.js:106-117`); re-rendering that on every delta would be both expensive and dangerous.
- OpenClaw's SAFETY comments show event payloads are cast and then validated in reducers (`chat-live-events.runtime.ts:159-161,317-319`) — keep a validation boundary on untrusted frames.

---

## 3. Attachments in chat

### 3.1 Hermes

- Adapters download media to local cache files; `MessageEvent` carries **paths**, not bytes: `media_urls`/`media_types`/`media_filenames` (`gateway/platforms/event.py:94-98,136-139`). Shared funnel classifies by extension/MIME and writes cache (`gateway/platforms/base.py:1447-1476`); images allow `.jpg/.jpeg/.png/.webp/.gif` (`base.py:1126-1128`); `cache_image_from_bytes` rejects non-image bytes (`base.py:631-638`); files are written 0600 (`base.py:611-628`).
- Telegram: Bot API download, default **20 MB** (2 GB with a local Bot API `base_url`) (`plugins/platforms/telegram/adapter.py:597,5611-5626`); text files ≤100 KB injected into text (:6154-6156). Discord: 20 MB upload, 7 MB reply files (`plugins/platforms/discord/adapter_media.py:24-27`). Slack: rejects >20 MB (`plugins/platforms/slack/adapter.py:4934`). Global inbound cap 128 MiB, configurable, 0 disables (`base.py:498-525`).
- Webchat: browser paste sends a **base64 data URL**, client cap **25 MB** (`web/src/lib/chatImagePaste.ts:15,127-129`); server requires `data:...;base64`, decodes, requires `image/*` and sniffs magic bytes against PNG/JPEG/GIF/BMP/WebP (`hermes_cli/web_routers/files.py:640-692`); stored under `$HERMES_HOME/images` (:695-736). General file picker uses multipart `UploadFile`, cap **16 MiB** (`web_routers/uploads.py:22-54`; `hermes_constants.py:26`).
- To the model: images are attached as native vision pixels or pre-analyzed into text (`gateway/run_inbound.py:1521-1557`); documents become path context notes (:1647-1678); plugin LLM image blocks become OpenAI `data:<mime>;base64,...` (`agent/plugin_llm.py:265-273`).
- Outbound: model replies use `MEDIA:/absolute/path` tags (`gateway/platforms/base.py:1130-1157,4082-4121`), with user-visible failure notices (:3072-3095). The API-server adapter rewrites image `MEDIA:` paths to inline `data:` URLs **max 5 MB specifically to avoid leaking host paths** (`gateway/platforms/api_server.py:707-735`). Tool-result images are OpenAI `image_url` parts and are evicted when stale (`agent/turn_request_assembly.py:160-167`).
- Security: outbound path allow/deny resolution (`base.py:1030-1070`), directory/control-char stripping from untrusted filenames (`base.py:1395-1406`; `event.py:27-43`), WhatsApp bridge symlink rejection (`plugins/platforms/whatsapp/adapter.py:384`; `gateway/session_ingress_media.py:315,439`), SSRF checks before fetch and per redirect (`base.py:646-657`), Pillow full validation with 100 frames/100M pixels (`tools/vision_tools_image_prep.py:245-280`). **No AV/malware scanning of chat attachments was found** (the scanner in `hermes_cli/plugins_install_scan.py` is for plugins/skills, not media).

### 3.2 OpenClaw

- Shared caps: image **6 MB**, audio **16 MB**, video **16 MB**, document **100 MB** (`packages/media-core/src/constants.ts:1-8,36-44`). Saved media uses 0700 directories and a default 5 MB read cap (`src/media/store.ts:36,81-99,435,511,569`).
- WhatsApp: Baileys `downloadMediaMessage(..., "stream")` → `saveMediaStream`, default 50 MB (`extensions/whatsapp/src/inbound/media.ts:14-53`). Telegram: Bot API 20 MB limit with an oversize warning to the user (`extensions/telegram/src/bot-handlers.media.ts:4-16`; `bot-handlers.inbound-pipeline.ts:326-340`); refs are `{path, contentType, fileName, kind}` (:353-360). Discord: CDN attachments via `saveRemoteMedia` behind an SSRF policy (`extensions/discord/src/monitor/message-media.ts:1-9,66-76`). Slack: 8 files max (`extensions/slack/src/monitor/media-types.ts:8`); private URLs refreshed via `files.info` with capped downloads (`monitor/media.ts:255-275,313-334`).
- Control UI: `chat.send` accepts `attachments` with an intentionally permissive envelope `{type, mimeType, fileName, origin:"paste"|"file", content, sizeBytes, durationMs, width, height}` (`logs-chat.ts:341-359,439`); data URLs are strict base64 and bounded by `MAX_IMAGE_BYTES` (`src/gateway/control-ui-public-session-attachments.ts:121-129`); oversized text prompts are staged as `media://inbound/<id>` (`src/agents/harness/current-input-budget.ts:36-52`); attachment text becomes escaped `<file name= mime=>` blocks (`src/media/file-context.ts:5-34`). Composer has paste/upload menus and base64 data-URL handling (`ui/src/pages/chat/components/chat-composer-attachment-menu.tsx:4-9`, `chat-composer-pasted-text.tsx:25-26`).
- Outbound: payloads use `mediaUrl`/`mediaUrls` (`docs/nodes/images.md:59-61`). Managed UI blocks are `{type:"image"|"audio"|"video"|"attachment", url, artifactId, mimeType, sizeBytes, alt|fileName, width/height}` (`src/gateway/managed-image-attachments.ts:434-468`), with `attachment_error` blocks on failure (:180-195). Limits: 12 MB, 4096×4096, 20M pixels; transient images expire after 15 min (:125-132).
- Security: MIME sniffing prefers magic bytes and is bounded to 1 MiB (`packages/media-core/src/mime.ts:5-6,178-197`); declared MIME and extension are arbitrated in `classifyAttachmentBytes` (`attachment-classify.ts:124-136`); inline data URLs are rewritten to sniffed MIME from allowlist `{png,jpeg,webp,gif}` (`inline-image-data-url.ts:8-9,58-73,120-123`); local root guards (`src/media/web-media.ts:653-662`); Slack download errors are redacted (`monitor/media.ts:320-324`). **No AV scanning found.** No explicit image prompt-injection warning found, only generic untrusted-content markers (`current-input-budget.ts:17-22`; `docs/plugins/agent-guidance.md:120`).
- User-facing docs worth copying: `$TMPDIR/leo-research-openclaw-src/repo/docs/nodes/images.md:22-27,42-55,89-104` (path-or-URL input, MIME precedence, per-kind limits, readable errors); Hermes `$TMPDIR/leo-research-hermes-src/hermes-agent/website/docs/user-guide/security.md:815,855,880` (SSRF rules for media).

### 3.3 LEO today (read-only observation, 2026-11-18)

- `POST /chat/stream` (SSE) and `LeoApp.ask_stream()` already exist (`<repo>/leo/interfaces/api.py:132-156`; `<repo>/leo/core/app.py:218-354`). The draft plan at `<repo>/docs/CHAT_REALTIME_PLAN.md:39-41` still describes these as future work; it is stale on that point.
- `AttachmentStore` exists and is instantiated (`<repo>/leo/integrations/attachments.py:37-99`; `app.py:48`): 10 MB cap, MIME allowlist, strict base64, sanitized filename, SHA-256 digest, 16 KB text preview, containment check on the *directory*. But: **no `/attachments` upload or preview route exists in `api.py`**, the dashboard sends no attachments (`app.js:294-305`), and `AttachmentStore.image_block()` (`attachments.py:109-125`) is **never called** — `ask_stream()` only injects a labelled text preview (`app.py:248-254`). Model providers receive no image pixels.
- `ask_stream()` emits `run.started`, `status`, `message.delta`, `tool.started`, `approval.required`, `tool.finished`, `message.finished`, `run.finished`, `error` (`app.py:236-354`). It is not covered by any test (no `ask_stream`/`chat/stream`/attachment references under `<repo>/tests/`).
- Security-relevant gaps: MIME is trusted from the client with no magic-byte sniff; final write path is not resolved/checked after naming (a planted symlink could redirect the write); base64 input size is not bounded before decode; absolute host path is returned in the store record and persisted in message metadata; streamed assistant deltas, tool arguments and tool results are not passed through `_redact()` (only user text and attachment previews are); `tool.finished.success` is inferred by substring-searching the result string (`app.py:316-321`); there is no cancellation, no `seq`/dedupe, no reconnect/history-replay path, and no way to resume a tool after an approval is resolved (see §5.3).
- Auth: `api_auth` guards all routes except `/health` and `/webhooks/*`, accepting `Authorization: Bearer` or `x-leo-token` (`api.py:86-107`). The dashboard stores the token in `sessionStorage` and sets the header via `fetch` (`app.js:21,61-72`). `EventSource` could not do this; `fetch` streaming can.

---

## 4. UI patterns to copy and anti-patterns to avoid

**Copy:** stable run/tool IDs with monotonic sequence numbers; one live assistant node mutated by appending text rather than full-thread `innerHTML` rebuilds; an explicit status state separate from text; collapsible tool cards created at start and sealed at result with an explicit error flag; approvals as first-class cards with a bounded choice set; first-token-immediate/rest-coalesced rendering; provisional subscribe before history fetch; short-lived single-use tickets for browser sockets when token auth is enabled; sniffed MIME and opaque IDs instead of host paths in API responses; per-kind size caps with an actionable user-facing rejection message.

**Avoid:** blocking POST with a blank gap; full-thread re-render on every token; trusting declared MIME/extensions; buffering unbounded text or tool output; inferring tool failure from result text; showing an Approve button that cannot actually resume the tool; emitting absolute file paths to the browser; unbounded base64 bodies; silently dropping failed uploads; re-appending deltas after reconnect without seq/run dedupe; putting untrusted file content in the system prompt without a data label; using `innerHTML` for model/tool output.

---

## 5. Concrete recommendation for LEO

### 5.1 Transport and minimal API changes

1. **Keep SSE `POST /chat/stream`**; do not add a WebSocket control plane now. It is already the right shape for a local single-user app (`api.py:132-156`). Use `fetch()` + `ReadableStream` in the dashboard so the existing bearer-token auth and `AbortController` work unchanged. Do not switch to `EventSource` unless you first add a single-use ticket query parameter like Hermes' `POST /api/auth/ws-ticket`.
2. **Add three small routes**, all behind the existing `api_auth` middleware:
   - `POST /attachments` — multipart file upload preferred (no 33% base64 overhead), plus a base64 JSON variant for pasted images so `AttachmentStore.save()` can be reused as-is. Returns `{id, name, mime, size, sha256, kind}` only — **never `path`**.
   - `GET /attachments/{id}/preview` — auth-gated, `image/*` only, `Content-Type` from the stored record, capped bytes; for chips of non-images return no preview.
   - `POST /runs/{run_id}/cancel` (or propagate `Request.is_disconnected()` from the SSE generator) for the Stop button.
3. **Add `attachment_ids: list[str]` to `ChatStreamRequest`** while keeping the existing inline `attachments` field for compatibility. Resolve IDs server-side via `AttachmentStore.get()`; do not accept client-supplied paths. Build provider multimodal blocks with `AttachmentStore.image_block()` when the provider advertises vision support; otherwise keep the labelled text preview and surface a visible "image stored, model is text-only" notice.
4. **Harden `AttachmentStore` before exposing uploads:** sniff magic bytes and compare with declared MIME; resolve the final path (not just the directory) and require containment; bound the decoded byte length before writing; cap per-provider image payloads at ~5 MB (Hermes API-server precedent); never return or log absolute paths.
5. **Redact streamed content** with a carry-over tail buffer so patterns split across deltas still match; apply `_redact()` to `delta`, tool `arguments`, and tool `result`, in addition to the existing user-text and preview redaction (`app.py:233,253,314,352`).

### 5.2 Event schema (small extension of what exists)

Keep the current event names, add the fields marked `+`:

```jsonc
{"type":"run.started","run_id":"...","conversation_id":"...","seq":0,             /* + */
 "resumable":false /* + */, "attachment_ids":["att_..."] /* + */}
{"type":"status","run_id":"...","seq":1,"status":"thinking|tool|responding|waiting_approval|cancelled"}
{"type":"message.delta","run_id":"...","seq":2,"message_id":"...","delta":"...","index":0}
{"type":"tool.started","run_id":"...","seq":3,"tool_call_id":"...","name":"...","arguments":{},"risk":"high","status":"running"}
{"type":"tool.finished","run_id":"...","seq":n,"tool_call_id":"...","name":"...","success":true,
 "is_error":false,"duration_ms":12,"result":"<bounded, redacted>"}
{"type":"approval.required","run_id":"...","seq":n,"approval_id":"...","tool_call_id":"...",
 "name":"...","reason":"...","choices":["approve","deny"]}
{"type":"message.finished","run_id":"...","seq":n,"content":"...","usage":{},"provider":"...","model":"..."}
{"type":"run.finished","run_id":"...","seq":n,"status":"completed|failed|cancelled"}
{"type":"error","run_id":"...","seq":n,"code":"...","message":"<redacted>"}
```

Rules: `seq` is monotonic within a run and the client drops `seq <= lastSeen`; `tool.finished` must be the only thing that seals a tool row (never `approval.required`); `success` is an explicit boolean from `ToolResult`, not a substring check; all text fields are redacted server-side; no absolute paths, no raw file bytes, no credentials. Emit a `status:"responding"` immediately after the last tool loop so "typing" never goes dark between tool completion and final tokens.

### 5.3 Approval semantics — the one correctness blocker

Today `ToolExecutorImpl` raises `ApprovalRequired` and `ask_stream()` converts it into a JSON string fed back to the model plus an `approval.required` event (`app.py:304-312`). Resolving `POST /approvals/{id}/resolve` changes DB state but **cannot resume that tool call**: `PermissionManagerImpl._approval_is_valid()` requires the approval ID to be supplied *at authorization time* (`<repo>/leo/security/permissions.py:57-73`). Therefore an inline "Approve" button in chat would be a lie unless a run registry is added. Two acceptable options:

- **(a)** Add a paused-run registry keyed by `(run_id, tool_call_id, approval_id)`; on resolve, re-execute that exact `ToolCall` with `approval_id` in context and resume the generator, reusing the existing binding checks (actor/action/resource/risk, timeout). This is the correct UX and matches Hermes/OpenClaw (JSON-RPC `approval` request / `approval.*` events), but is more work.
- **(b)** Ship phase 1 without inline approval: emit `approval.required`, render a read-only blocked tool row, link to the existing Approvals view, and tell the user to re-ask after resolving. Do **not** render an Approve control that implies resume.

### 5.4 Frontend state machine

`idle → queued → thinking → streaming → tool → waiting_approval → streaming → done | error | cancelled`.

- On submit: add the user bubble and attachment chips immediately; create one assistant node and keep its text node reference. On each `message.delta`, append via `textContent` to that node (OpenClaw `chat-stream-message.ts:4-45` model); never rebuild the whole thread mid-stream.
- Status label lives in a separate `aria-live="polite"` element with a subtle dot; `prefers-reduced-motion` replaces the pulse with a static label. First token arrives ≤ one render frame; subsequent deltas are batched into `requestAnimationFrame`.
- Tool rows use the existing `.row`/`.badge` system, keyed by `tool_call_id`: name, redacted argument summary, pulsing dot while running, disclosure (`<details>` / `aria-expanded`) for bounded result text, explicit failed/success badge.
- Attachments: file input + paste + drag/drop; thumbnails via `GET /attachments/{id}/preview` with an auth header fetched as a blob (or pre-upload `URL.createObjectURL`); chips show name/size and a Remove button; unsupported types/sizes fail inline with the server's exact error. Label attached content as untrusted data; never interpret it as instructions (`app.py:254` already does this for previews).
- Stop button replaces Send while running and aborts the fetch; Escape cancels; Enter sends, Shift+Enter inserts newline; focus returns to the composer on terminal events.
- Reconnect: SSE has no built-in resume. On stream drop, refetch `GET /conversations/{id}` (`api.py:159-166`) and **replace** the transcript, then discard late events from the old `run_id` via the `seq`/run-id gate; do not append history to the live transcript. This is the cheap version of OpenClaw's provisional-observer + delta/reset handshake.

### 5.5 Tests

- **Unit:** event schema/serialization and `seq` monotonicity; redaction across chunk boundaries (`api_key: sk-...` split between two deltas); `AttachmentStore` rejects wrong MIME vs magic bytes, >10 MB, empty/invalid base64, traversal names and symlink targets; preview returns no host path.
- **API (httpx `ASGITransport`, as in `<repo>/tests/test_api.py:26-49`):** mock-provider stream emits ordered `run.started → status → message.delta* → tool.started → tool.finished → message.finished → run.finished`; tool-call turn emits tool events within a bounded number of chunks; approval turn emits `approval.required` and (for option (b)) no `tool.finished`; `POST /chat` remains unchanged and still returns the final dict; `POST /attachments` with a tiny PNG round-trips to a preview; bearer token is required on all new routes.
- **Frontend/Playwright:** drive a local mock SSE stream; assert progressive text, status label changes, tool row start→finish, image thumbnail + file chip, inline rejection error, Stop/cancel, reconnect dedupe, reduced-motion fallback, and the 800 px layout. Do not contact paid providers; use the existing loopback mock-provider pattern from `hermes-e2e.md:314-320`.
- **Regression/security:** no credentials in SSE frames, logs, DB metadata, or tool results; tool failure flag is explicit; no `innerHTML` path for model/tool text; full suite remains green (the prior round's baseline is 145 passed, 1 skipped per `SYNTHESIS.md:73`).

### 5.6 Phasing

1. Wire the existing `ask_stream()` into the dashboard via fetch-SSE + state machine (biggest UX win, no new protocol).
2. Add seq/dedupe, status:"responding", explicit tool result flags, streamed redaction, cancel.
3. Add `POST /attachments` + preview + composer, with magic-byte/MIME hardening and image blocks where supported.
4. Decide approval UX (a) resume registry or (b) read-only blocked row; do not ship a fake Approve button.
5. Playwright and API tests, then docs.

---

## 6. Confidence and limits

- All upstream wire names and line numbers were verified directly in the pinned checkouts on 2026-11-18; runtime rendering behavior was inferred from source and from the earlier E2E turn, not re-tested here.
- "NOT FOUND" claims are scoped to these checkouts: Hermes has no SSE chat endpoint in its dashboard/desktop; OpenClaw has no chat SSE or `chat.delta`/`agent.thinking` wire names.
- No AV scanning of chat attachments was found in either upstream project; absence of a hit in grep is not a security guarantee.
- LEO's files were changing concurrently during this research (other work in progress). The LEO state in §3.3/§5 reflects what was read at 2026-11-18; re-check before implementing.
- No upstream source was copied into LEO, no paid provider was contacted, and no destructive command was run.
