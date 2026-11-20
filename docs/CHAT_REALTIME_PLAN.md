# LEO Real-Time Chat & Attachments — Implementation Plan (draft v0.1)

**Status:** research complete (`research/deep-2026-11-18/realtime-chat-research.md`);
backend v1 implemented, frontend and hardening in progress.

**Research verdict:** neither upstream uses browser SSE — Hermes embeds its TUI
over a PTY WebSocket plus an event-mirror WebSocket, OpenClaw uses one typed
gateway WebSocket. For a local single-user app, SSE using `fetch()` +
`ReadableStream` (so bearer auth and `AbortController` work) is sufficient;
do not build a WebSocket control plane now. Copy the useful render patterns:
append to one live node, monotonic `seq` + run IDs for dedupe, stable tool rows
created at start and sealed at result with an explicit error flag, a separate
status stream, and magic-byte MIME sniffing for uploads.
**Goal:** replace the current blocking `POST /chat` UX with a live conversation
surface that shows progressive assistant text, explicit “working/typing” status,
tool calls as they happen, and image/file attachments — without weakening LEO’s
permission, redaction or audit boundaries.

## 1. Problem with the current chat

- `POST /chat` returns only after the full model turn (and up to three tool
  iterations) finish. The UI shows a spinner-free blank gap, then one message.
- Tool calls happen inside `LeoApp.ask()` and are returned as a final array;
  the user cannot see *which* tool ran, with what arguments, or when approval
  is needed.
- Attachments are impossible: `ChatRequest` accepts only a text `message`.
- No streaming, no cancellation, no progress/heartbeat, no reconnect semantics.

## 2. Target behaviour

1. **Progressive text** — assistant tokens render as they arrive.
2. **Explicit status** — `queued → thinking → calling tool → waiting for
   approval → responding → done/failed/cancelled`, with a visible label and a
   subtle animated indicator (reduced-motion fallback: static label).
3. **Tool-call transparency** — each tool call appears as a compact row:
   name, arguments, start time, result or error, and an Approve/Deny control
   when a HIGH-risk approval is required. Statuses update in place.
4. **Attachments** — paste/drop/select images and files; show thumbnails/chips
   before sending; strip or cap unsupported types; never render untrusted
   content as instructions.
5. **Cancellation & recovery** — stop button aborts the in-flight turn;
   reconnecting to a conversation restores history and any pending state.

## 3. Backend design

### 3.1 Streaming endpoint

**Implemented:** `POST /chat/stream` (SSE, `text/event-stream`) in
`leo/interfaces/api.py`, backed by `LeoApp.ask_stream()` that mirrors
`ask()` but yields typed events instead of returning one dict.

Proposed event envelope (JSON per SSE `data:` line):

```
{"type":"run.started","conversation_id":...,"run_id":...}
{"type":"status","status":"thinking"}
{"type":"message.delta","message_id":...,"delta":"..."}
{"type":"tool.started","tool_call_id":...,"name":...,"arguments":{...},"risk":"medium"}
{"type":"approval.required","approval_id":...,"tool_call_id":...}
{"type":"tool.finished","tool_call_id":...,"success":true,"result":{...}}
{"type":"message.finished","content":...,"usage":{...},"provider":...,"model":...}
{"type":"run.finished","status":"completed"}
{"type":"error","code":...,"message":...}
```

Keep `POST /chat` as a compatibility wrapper around the same generator so
tests and API consumers do not break.

### 3.2 Provider streaming

`ModelProvider.stream()` already exists for Anthropic/OpenAI-compatible
providers. Use it in `ask_stream()`; fall back to `complete()` for providers
that do not support streaming, emitting one final `message.delta`.

### 3.3 Attachments

- `POST /attachments` → `{id, name, mime, size, sha256, kind}`; never return a host path.
- Store under `settings.paths.workspaces/<conversation>/attachments/`, with
  path containment, per-file size cap (initial: 10 MB), extension/MIME allow
  list, and rejection of active content (`.html`, `.svg`, scripts).
- Images: generate a downscaled local thumbnail; pass image blocks to the
  provider using its documented image-content format (base64 data URL or URL).
- Non-images: extract a bounded text preview (≤16 KB) and include as labelled
  *data* in the prompt, with source attribution; never execute instructions
  found in uploaded files.
- Add `attachments: list[str]` to `ChatRequest` / stream request.

### 3.4 Security & audit

- Every attachment upload writes an audit row; every tool event is already
  audited by the executor.
- Re-run secret redaction on streamed text, tool arguments/results and
  attachment previews before sending events.
- Keep the approval flow authoritative: a `tool.started` event never implies
  execution; only `tool.finished` after a valid permission decision does.

## 4. Frontend design (Night Instrument system)

- Replace the single `chat-thread` list with a stateful renderer:
  `idle | queued | thinking | tool | approval | streaming | done | error`.
- Assistant message node is created on `run.started` and appended to on each
  `message.delta` (use `textContent`, never `innerHTML`, for model output).
- Tool rows use existing `.row`, `.badge`, `.status` components; active tools
  get a pulsing dot, completed tools a check, failed tools a cross.
- Composer gains an attachment button (paperclip SVG), a hidden file input,
  a drop zone on the thread, removable chips/thumbnails, and a Stop button
  that replaces Send while running.
- Events are consumed with `EventSource` (SSE) or `fetch` + `ReadableStream`;
  automatic reconnect with run-id de-duplication.
- Keyboard: Enter sends, Shift+Enter newline, Escape cancels, focus returns to
  composer after completion. All controls have aria labels and focus rings.

## 5. Testing & acceptance

- Unit: event-schema serialisation; attachment validation (type, size, path,
  redirect/symlink), text extraction, redaction.
- API: SSE endpoint emits ordered events for a mock provider that makes one
  tool call; approval required path emits `approval.required` and does not emit
  `tool.finished` until resolved.
- Browser: Playwright drives a local mock stream; assert progressive text,
  tool row updates, image thumbnail, file chip, cancel, reconnect, reduced
  motion and 800px layout. Capture screenshots into `artifacts/screenshots/ui/`.
- Live: one real streamed turn through the configured `hy4-flash` gateway and
  one image attachment round-trip (if the gateway accepts image blocks).
- Acceptance: no blank gap >500 ms while streaming; tool call visible within
  500 ms of execution; attachments rejected with a clear inline error; full
  suite green; no credential in logs or events.

## 6. Phases

1. Research (in progress) — Hermes/OpenClaw real-time mechanics.
2. Backend event schema + streaming generator + compatibility wrapper.
3. Attachments API + storage + validation.
4. Frontend state machine + components + SSE client.
5. Visual QA rounds + tests + docs.

## 7. Resolved research questions / remaining decisions

- Event names/shape: keep LEO’s names, add `seq`, `message_id`, explicit tool success and `status:"responding"`.
- Transport: SSE via `fetch` + `ReadableStream` is sufficient; WS not justified.
- Attachments: upstream uses opaque media IDs, magic-byte sniffing and data-URL rewrites capped ~5 MB; adopt those.
- Approval UX decision: ship read-only blocked tool rows in phase 1 (no fake Approve control); a resumable paused-run registry is phase 2.
