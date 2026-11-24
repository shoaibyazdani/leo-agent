"""Contracts for the streamed Thinking panel in the conversation UI.

The chat surface is plain browser JavaScript, so these tests read the shipped
assets and assert that provider reasoning travels on its own channel: it is
never appended to the assistant answer buffer, never attached to a tool card,
and only real deltas create real content. The empty state is an explicit,
truthful statement rather than invented reasoning. Runtime tests execute the
shipped event handler in Node so the terminal latch, pending boundary and
authoritative reasoning flags are checked as behaviour, not source strings.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "leo" / "interfaces" / "static"
CHAT_JS_PATH = STATIC / "ui" / "conversation" / "chat.js"
CHAT_CSS_PATH = STATIC / "ui" / "conversation" / "chat.css"
CHAT_JS = CHAT_JS_PATH.read_text(encoding="utf-8")
CHAT_CSS = CHAT_CSS_PATH.read_text(encoding="utf-8")


def block(source: str, start: str, end: str) -> str:
    assert start in source, f"missing marker {start!r}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing closing marker {end!r}"
    return tail.split(end, 1)[0]


def js_block(start: str, end: str) -> str:
    """Extract a complete function declaration (block() returns its body)."""
    return start + block(CHAT_JS, start, end)


def run_node(script: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "harness.js"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


JS_ASSERTIONS = r"""
const trace = [];
const t = (...a) => trace.push(a);
function ok(cond, msg) { if (!cond) throw new Error(msg); }
function saw(name, ...args) {
  return trace.some((row) => row[0] === name && args.every((arg, i) => row[i + 1] === arg));
}
function fake(extra = {}) {
  const node = {
    textContent: '', className: '', hidden: false, isConnected: true, removed: false,
    dataset: {}, classList: { add() {}, remove() {}, contains() { return false; } },
    style: {}, children: [],
    querySelector(sel) { return (this._query && this._query[sel]) || null; },
    querySelectorAll(sel) { return (this._all && this._all[sel]) || []; },
    appendChild(child) { this.children.push(child); return child; },
    append(...children) { this.children.push(...children); },
    remove() { this.removed = true; },
    setAttribute() {}, addEventListener() {},
  };
  return Object.assign(node, extra);
}
function assistantNode() {
  const meta = fake();
  const head = fake({ _query: { '.chat-msg__meta': meta, '.chat-msg__actions': null } });
  return fake({ _query: { '.chat-msg__head': head, '.chat-msg__actions': null } });
}
const state = {
  activeRun: 'pending', lastSeq: -1, stream: null, activeId: 'c1',
  turnReasoningSeen: false, noReasoningNote: false, reasoningClosedRuns: new Set(),
  turnNodes: [], turnReasoningStreams: [], turnTools: new Map(), terminalStatus: null,
};
const NO_REASONING_NOTE = 'No reasoning streamed';
const STATUS_LABELS = { thinking: 'Thinking…', cancelled: 'Cancelled', tool: 'Calling a tool…', responding: 'Responding…' };
const thread = fake();
const threadEl = () => thread;
const domNodes = {};
const $ = (id) => (domNodes[id] || (domNodes[id] = fake()));
const document = { createElement: () => fake() };
const window = { LEOMarkdown: { renderInto: (el, text) => t('renderInto', text) } };
const localStorage = { setItem() {}, getItem() { return null; } };
const toast = (...a) => t('toast', ...a);
const banner = (...a) => t('banner', ...a);
const noticeNode = (text) => fake({ textContent: text });
const maybeScroll = () => {};
const startElapsed = () => t('startElapsed');
const stopElapsed = () => t('stopElapsed');
const setChatStatus = (...a) => t('setChatStatus', ...a);
const setTurnState = (...a) => t('setTurnState', ...a);
const setTurnStatus = (...a) => t('setTurnStatus', ...a);
function applyTerminalStatus(status) { t('applyTerminalStatus', status); state.terminalStatus = status; }
const removeStreamingCarets = () => t('removeStreamingCarets');
const sealStream = (...a) => t('sealStream', ...a);
const appendReasoning = (...a) => t('appendReasoning', ...a);
const toolNode = () => ({ card: fake(), statusEl: fake(), statusText: fake() });
const turnToolRow = () => ({ row: fake(), status: fake() });
const approvalNode = () => fake();
function closeTurnStream(terminal = true, runId = null) {
  t('closeTurnStream', terminal, runId);
  if (terminal && !state.turnReasoningSeen && state.stream && state.stream.node && state.stream.node.isConnected) {
    noteNoReasoning(state.stream);
  }
  state.stream = null;
  state.turnReasoningSeen = false;
}
function ensureAssistantNode() {
  t('ensureAssistantNode');
  if (!state.stream || state.stream.sealed) {
    state.stream = {
      node: assistantNode(), contentEl: fake(), buffer: '', sealed: false,
      runId: state.activeRun === 'pending' ? null : state.activeRun,
    };
  }
  return state.stream;
}
"""

EVENT_FUNCTIONS = "\n\n".join(
    [
        js_block("function noteNoReasoning(stream)", "\n  // The backend's message.finished"),
        js_block("function markReasoningSeen()", "\n  // A later tool round may stream"),
        js_block("function clearNoReasoningNotes(", "\n  // Terminal paths close"),
        js_block("function closeReasoning(", "\n  function reasoningClosed("),
        js_block("function reasoningClosed(", "\n  // Terminal housekeeping"),
        js_block("function applyStreamEvent(event)", "\n  async function streamChat("),
    ]
)


def test_code_still_parses() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--check", str(CHAT_JS_PATH)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_thinking_delta_is_handled_as_its_own_event() -> None:
    branch = block(
        CHAT_JS,
        "} else if (event.type === 'thinking.delta')",
        "} else if (event.type === 'tool.started')",
    )
    assert "appendReasoning(event.delta)" in branch
    assert "state.stream.buffer" not in branch
    assert "queueStreamRender()" not in branch
    assert "reasoningClosed(eventRunId)" in branch
    # The event stays inside the same seq/run_id gate as all other events.
    gate = block(CHAT_JS, "function applyStreamEvent(event)", "if (event.type === 'run.started')")
    assert "event.run_id !== state.activeRun" in gate
    assert "event.seq <= state.lastSeq" in gate


def test_reasoning_is_not_the_answer_and_not_tool_output() -> None:
    append = block(CHAT_JS, "function appendReasoning(delta)", "function queueReasoningRender(")
    assert "typeof delta !== 'string' || !delta" in append
    assert "stream.reasoningBuffer" in append
    assert "state.stream.buffer" not in append
    assert "queueStreamRender()" not in append
    assert "state.turnTools" not in append
    assert "event.content" not in append
    attach = block(CHAT_JS, "function attachReasoningPanel(stream)", "function appendReasoning(")
    assert "stream.node.appendChild(reasoning.panel)" in attach
    assert "never inside .chat-msg__content" in attach
    assert "state.turnReasoningStreams.push(stream)" in attach
    finalize = block(CHAT_JS, "function finalizeReasoning(", "function finalizeTurnReasoning(")
    assert "body.textContent = boundedText(stream.reasoningBuffer, REASONING_RESULT_LIMIT)" in finalize
    # `body` here is the reasoning <pre>, never the markdown answer element.
    assert "contentEl" not in finalize
    assert "queueStreamRender" not in finalize


def test_panel_is_a_labelled_native_disclosure_with_quiet_announcements() -> None:
    node = block(CHAT_JS, "function reasoningNode()", "function attachReasoningPanel(")
    assert "createElement('details')" in node
    assert "createElement('summary')" in node
    assert "setAttribute('aria-label'" in node
    assert "Hide" in node and "Show" in node
    assert "aria-hidden" in node
    assert "label.textContent = 'Thinking'" in node
    assert "setAttribute('aria-live', 'polite')" in node
    assert "setAttribute('aria-atomic', 'true')" in node
    assert "chat-reasoning__status sr-only" in node
    assert "panel.addEventListener('toggle'" in node
    # The streamed text itself is deliberately NOT a live region: no token spam.
    assert "body.setAttribute('aria-live'" not in node
    assert "body.className = 'chat-reasoning__body'" in node


def test_empty_state_is_truthful_and_only_finalized_at_termination() -> None:
    # Old invented-card copy is gone; the only verdict is one honest line.
    assert "No reasoning was streamed by this model." not in CHAT_JS
    assert "No reasoning was streamed for this part of the reply." not in CHAT_JS
    assert "REASONING_EMPTY" not in CHAT_JS
    assert "REASONING_SEGMENT_EMPTY" not in CHAT_JS
    assert "const NO_REASONING_NOTE = 'No reasoning streamed';" in CHAT_JS

    finalize = block(CHAT_JS, "function finalizeReasoning(", "function finalizeTurnReasoning(")
    assert "detachReasoningPanel(stream);" in finalize  # never an empty disclosure
    assert "panel.dataset.reasoningState = 'empty'" not in finalize
    assert "body.textContent = empty" not in finalize
    assert "the model thought" not in CHAT_JS.lower()

    note = block(CHAT_JS, "function noteNoReasoning(stream)", "\n  // The backend's message.finished")
    assert "NO_REASONING_HIDDEN_NOTE" in note
    assert "setTurnStatus(text)" in note
    assert "note.textContent = text" in note
    assert "createElement('span')" in note  # one line, not a card

    # Authoritative flags are consumed without fabricating any panel text.
    finished = block(CHAT_JS, "event.type === 'message.finished'", "event.type === 'error'")
    assert "if (event.reasoning_seen === true) markReasoningSeen();" in finished
    assert "closeReasoning(eventRunId);" in finished
    mark = block(CHAT_JS, "function markReasoningSeen()", "\n  // A later tool round may stream")
    assert "state.turnReasoningSeen = true;" in mark
    assert "appendText" not in mark and "reasoningBuffer" not in mark
    state_event = block(CHAT_JS, "event.type === 'thinking.state'", "event.type === 'message.finished'")
    assert "event.seen === true" in state_event
    assert "markReasoningSeen();" in state_event
    assert "appendReasoning" not in state_event

    # Tool rounds seal early; only a terminal event gives the turn-level verdict.
    seal = block(CHAT_JS, "function sealStream(terminal = false, runId = null)", "function queueStreamRender(")
    assert "if (terminal) finalizeTurnReasoning(true);" in seal
    assert "else finalizeReasoning(state.stream, false);" in seal
    for marker in ("} else if (event.type === 'error')", "} else if (event.type === 'run.finished')"):
        tail = CHAT_JS.split(marker, 1)[1]
        assert "sealStream(true, eventRunId)" in tail.split("}", 1)[0]


def test_reasoning_state_is_bounded_reset_and_flushed() -> None:
    append = block(CHAT_JS, "function appendReasoning(delta)", "function queueReasoningRender(")
    assert "boundedText(stream.reasoningBuffer + delta, REASONING_RESULT_LIMIT)" in append
    assert "panel.open = true" in append
    render = block(CHAT_JS, "function queueReasoningRender()", "function finalizeReasoning(")
    assert "setTimeout(() => {" in render
    assert "RENDER_THROTTLE_MS" in render
    finalize = block(CHAT_JS, "function finalizeReasoning(", "function finalizeTurnReasoning(")
    assert "if (stream.reasoningTimer) { clearTimeout(stream.reasoningTimer);" in finalize
    assert "stream.reasoningRenderQueued = false;" in finalize
    stream_chat = block(CHAT_JS, "async function streamChat(text)", "const headers = {")
    assert "state.turnReasoningStreams = []" in stream_chat
    assert "state.turnReasoningSeen = false" in stream_chat
    assert "state.reasoningClosedRuns = new Set();" in stream_chat


def test_late_reasoning_after_a_tool_round_replaces_provisional_empty_state() -> None:
    # A tool status seals the current assistant block before the model may
    # stream more reasoning. That provisional "no reasoning for this part"
    # text must not survive a later thinking.delta on the same block.
    append = block(CHAT_JS, "function appendReasoning(delta)", "function queueReasoningRender(")
    assert "stream.reasoningFinalized && !stream.reasoningTerminal" in append
    assert "stream.reasoningFinalized = false" in append
    assert "stream.reasoning.body.textContent = ''" in append
    assert "panel.dataset.reasoningState = 'streaming'" in append
    finalize = block(CHAT_JS, "function finalizeReasoning(", "function finalizeTurnReasoning(")
    assert "if (stream.reasoningFinalized && (stream.reasoningTerminal || !terminal)) return;" in finalize


def test_reasoning_panel_css_is_scoped_and_accessible() -> None:
    for needle in (
        ".chat-reasoning {",
        ".chat-reasoning__toggle:focus-visible",
        ".chat-reasoning__toggle::-webkit-details-marker",
        ".chat-reasoning[data-reasoning-state='empty'] .chat-reasoning__body",
        ".chat-reasoning[data-reasoning-state='streaming']",
        "white-space: pre-wrap",
        "max-height:",
        "overflow: auto",
    ):
        assert needle in CHAT_CSS
    reduced = CHAT_CSS.split("@media (prefers-reduced-motion: reduce) {", 1)[1].split("\n}", 1)[0]
    assert ".chat-reasoning__toggle" in reduced
    assert ".chat-reasoning__icon" in reduced


def test_runtime_terminal_delta_cannot_open_a_new_panel() -> None:
    """A delta after message.finished/error/run.finished is inert."""
    driver = r"""
const E = (type, run_id, extra = {}) => Object.assign({ type, run_id }, extra);
function fresh(runId) {
  state.activeRun = runId;
  state.lastSeq = -1;
  state.stream = null;
  state.reasoningClosedRuns = new Set();
  state.turnReasoningSeen = false;
  trace.length = 0;
}

// Real delta streams; then message.finished closes the run.
fresh('pending');
applyStreamEvent(E('run.started', 'r1'));
trace.length = 0;
applyStreamEvent(E('thinking.delta', 'r1', { delta: 'real thought' }));
ok(saw('appendReasoning', 'real thought'), 'real delta was dropped');
fresh('r1');
applyStreamEvent(E('message.finished', 'r1', { content: 'answer', reasoning_seen: false }));
ok(saw('closeTurnStream', true, 'r1'), 'finished run was not closed');
trace.length = 0;
applyStreamEvent(E('thinking.delta', 'r1', { delta: 'late thought' }));
ok(trace.length === 0, 'post-finish delta produced effects: ' + JSON.stringify(trace));
ok(state.stream === null, 'post-finish delta created an assistant node');

// error and run.finished close their runs just as hard.
for (const terminal of ['error', 'run.finished']) {
  fresh('r2');
  applyStreamEvent(E(terminal, 'r2', { status: 'failed', message: 'boom' }));
  trace.length = 0;
  applyStreamEvent(E('thinking.delta', 'r2', { delta: 'late' }));
  ok(trace.length === 0, terminal + ' did not latch the run: ' + JSON.stringify(trace));
  ok(state.stream === null, terminal + ' allowed a new assistant node');
}
"""
    run_node(JS_ASSERTIONS + "\n\n" + EVENT_FUNCTIONS + "\n" + driver)


def test_runtime_pending_boundary_ignores_stale_run_id() -> None:
    """Only run.started may adopt the run id while a POST is pending."""
    driver = r"""
const E = (type, run_id, extra = {}) => Object.assign({ type, run_id }, extra);
state.activeRun = 'pending';
state.lastSeq = -1;
state.stream = null;
trace.length = 0;

applyStreamEvent(E('thinking.delta', 'old-run', { delta: 'stale thought' }));
applyStreamEvent(E('message.delta', 'old-run', { delta: 'stale answer' }));
ok(state.activeRun === 'pending', 'pending boundary adopted a stale run id: ' + state.activeRun);
ok(trace.length === 0, 'stale pre-start event produced effects: ' + JSON.stringify(trace));

trace.length = 0;
applyStreamEvent(E('run.started', 'fresh-run'));
ok(state.activeRun === 'fresh-run', 'run.started did not establish the id');
ok(saw('ensureAssistantNode'), 'run.started did not open the assistant node');
trace.length = 0;
applyStreamEvent(E('thinking.delta', 'old-run', { delta: 'still stale' }));
ok(trace.length === 0, 'old run delta leaked into the new run: ' + JSON.stringify(trace));
applyStreamEvent(E('thinking.delta', 'fresh-run', { delta: 'fresh thought' }));
ok(saw('appendReasoning', 'fresh thought'), 'current run delta was rejected');
"""
    run_node(JS_ASSERTIONS + "\n\n" + EVENT_FUNCTIONS + "\n" + driver)


def test_runtime_authoritative_reasoning_flags_suppress_false_note() -> None:
    """reasoning_seen / final thinking.state beat lost client deltas."""
    driver = r"""
const E = (type, run_id, extra = {}) => Object.assign({ type, run_id }, extra);
function start(runId) {
  state.activeRun = 'pending';
  state.lastSeq = -1;
  state.stream = null;
  state.reasoningClosedRuns = new Set();
  state.turnReasoningSeen = false;
  state.noReasoningNote = false;
  trace.length = 0;
  applyStreamEvent(E('run.started', runId));
  trace.length = 0;
}
const falseNote = () => trace.some((row) => row[0] === 'setTurnStatus' && row[1] === NO_REASONING_NOTE);

// Control: with no reasoning at all the honest one-line note IS emitted.
start('r1');
applyStreamEvent(E('message.finished', 'r1', { content: 'answer', reasoning_seen: false }));
ok(falseNote(), 'harness did not observe the honest empty verdict');

// Lost deltas, but message.finished says reasoning was seen: no false note,
// no invented panel, no new node.
start('r2');
applyStreamEvent(E('message.finished', 'r2', { content: 'answer', reasoning_seen: true }));
// The control above proves the harness prints the note when reasoning_seen
// is false, so absence here means the authoritative flag was honoured.
ok(!falseNote(), 'authoritative reasoning_seen still printed "No reasoning streamed"');
ok(!saw('appendReasoning') && !saw('ensureAssistantNode'), 'authoritative flag fabricated a panel');

// A final thinking.state with seen=true is authoritative too, and is not
// itself a terminal event.
start('r3');
applyStreamEvent(E('thinking.state', 'r3', { final: true, seen: true, empty: false }));
ok(state.turnReasoningSeen === true, 'final thinking.state seen=true was ignored');
ok(!saw('ensureAssistantNode') && !saw('appendReasoning'), 'thinking.state created a panel');
trace.length = 0;
applyStreamEvent(E('thinking.delta', 'r3', { delta: 'now a real late delta' }));
ok(saw('appendReasoning'), 'thinking.state wrongly closed an open run');
"""
    run_node(JS_ASSERTIONS + "\n\n" + EVENT_FUNCTIONS + "\n" + driver)


def test_display_hidden_uses_its_own_terminal_note():
    """An intentionally hidden channel must not read as "the model sent none"."""
    assert "const NO_REASONING_HIDDEN_NOTE = 'Reasoning display is off';" in CHAT_JS
    finished = block(CHAT_JS, "} else if (event.type === 'message.finished') {", "} else if (event.type === 'run.finished') {")
    assert "event.reasoning_display === 'hidden'" in finished
    assert "state.reasoningHidden" in finished
    note = block(CHAT_JS, "function noteNoReasoning(stream)", "\n  // The backend's message.finished")
    assert "NO_REASONING_HIDDEN_NOTE" in note
