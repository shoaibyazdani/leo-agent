"""Contracts for the Thinking-panel lifecycle in the conversation UI.

Two regressions are pinned here:

1. An assistant turn that streams no reasoning must NOT get an empty disclosure
   card. The panel is created lazily by the first real ``thinking.delta`` and
   removed if that never happens; the only substitute is one short honest line.
2. A completed, cancelled or superseded reply must not accumulate a second,
   stale panel or keep an animated "thinking" indicator. The live panel is
   bound to a run_id and to one exact assistant message, and every terminal
   path seals/ releases that state.

The chat surface is plain browser JavaScript. Most contracts read the shipped
assets; the runtime tests execute the shipped functions in Node with a small
DOM/state harness so they verify behaviour, not just source text. They
complement (and, where the old empty-card copy is concerned, supersede)
test_thinking_stream_ui.py.
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


def event_branch(event_type: str, next_marker: str) -> str:
    return block(CHAT_JS, f"event.type === '{event_type}'", next_marker)


def js_block(start: str, end: str) -> str:
    """Extract a complete function declaration (block() returns its body)."""
    return start + block(CHAT_JS, start, end)


def run_node(script: str) -> None:
    """Execute a generated JS harness; a failed assertion fails the test."""
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
  // Emulate the shipped terminal bookkeeping so a false "no reasoning" note
  // is observable in the trace when reasoning_seen was ignored.
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


def test_shipped_chat_javascript_parses() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--check", str(CHAT_JS_PATH)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_panel_is_created_lazily_on_the_first_real_delta() -> None:
    ensure = block(CHAT_JS, "function ensureAssistantNode()", "function thinkingNode()")
    assert "attachReasoningPanel" not in ensure, "assistant nodes must not open a panel up front"
    assert "runId:" in ensure
    assert "reasoning: null" in ensure

    append = block(CHAT_JS, "function appendReasoning(delta)", "function queueReasoningRender()")
    assert "typeof delta !== 'string' || !delta" in append
    assert "if (!stream.reasoning) attachReasoningPanel(stream);" in append
    assert "if (reasoningClosed()) return;" in append  # defensive terminal guard

    delta = event_branch("thinking.delta", "event.type === 'tool.started'")
    assert "appendReasoning(event.delta)" in delta
    # A real delta is the only way into appendReasoning: no terminal/status
    # branch is allowed to create a disclosure panel.
    assert "attachReasoningPanel" not in delta
    assert "state.stream.buffer" not in delta
    assert "queueStreamRender()" not in delta


def test_no_empty_disclosure_card_at_terminal() -> None:
    # The invented empty-card copy is gone entirely.
    assert "No reasoning was streamed by this model." not in CHAT_JS
    assert "No reasoning was streamed for this part of the reply." not in CHAT_JS
    assert "REASONING_EMPTY" not in CHAT_JS
    assert "REASONING_SEGMENT_EMPTY" not in CHAT_JS

    finalize = block(CHAT_JS, "function finalizeReasoning(", "function finalizeTurnReasoning(")
    assert "detachReasoningPanel(stream);" in finalize
    assert "body.textContent = empty" not in finalize
    assert "panel.dataset.reasoningState = 'empty'" not in finalize

    assert "const NO_REASONING_NOTE = 'No reasoning streamed';" in CHAT_JS
    note = block(CHAT_JS, "function noteNoReasoning(stream)", "\n  // The backend's message.finished")
    assert "NO_REASONING_HIDDEN_NOTE" in note
    assert "setTurnStatus(text)" in note
    assert "createElement('span')" in note
    assert "note.textContent = text" in note  # a plain text node
    assert "chat-reasoning-none" in note
    assert "stream.node.querySelectorAll('.chat-reasoning-none')" in note  # one line per message

    # Retraction is scoped to the current run's message, never all turnNodes.
    retract = block(CHAT_JS, "function clearNoReasoningNotes(", "\n  // Terminal paths close")
    assert "state.turnNodes.forEach" not in retract
    assert "stream.node.querySelectorAll('.chat-reasoning-none')" in retract
    assert "runId !== state.activeRun" in retract
    close = block(CHAT_JS, "function closeTurnStream(terminal = true", "\n  function sealStream(")
    assert "clearNoReasoningNotes(stream);" in close
    assert "chat-reasoning-none" in CHAT_CSS


def test_terminal_paths_release_the_stream_without_appending_a_panel() -> None:
    seal = block(CHAT_JS, "function sealStream(terminal = false, runId = null)", "function queueStreamRender()")
    assert "attachReasoningPanel" not in seal
    assert "ensureAssistantNode" not in seal
    assert "state.stream.sealed = true;" in seal
    assert "thinking.remove();" in seal
    assert "if (runId && state.stream.runId && runId !== state.stream.runId) return;" in seal

    finished = event_branch("message.finished", "event.type === 'error'")
    assert "closeTurnStream(true, eventRunId);" in finished
    assert "attachReasoningPanel" not in finished
    assert "ensureAssistantNode" not in finished

    error = event_branch("error", "event.type === 'run.finished'")
    assert "sealStream(true, eventRunId);" in error
    assert "closeTurnStream(true, eventRunId);" in error

    run_finished = block(CHAT_JS, "event.type === 'run.finished'", "\n  }\n\n  async function streamChat")
    assert "sealStream(true, eventRunId);" in run_finished
    assert "closeTurnStream(true, eventRunId);" in run_finished
    assert "attachReasoningPanel" not in run_finished
    assert "ensureAssistantNode" not in run_finished

    # Cancellation and supersession used to skip the seal, leaving an animated
    # indicator and an unsealed assistant node behind.
    cancelled = block(CHAT_JS, "event.status === 'cancelled'", "} else if (event.type === 'message.delta')")
    assert "sealStream(true, eventRunId);" in cancelled
    assert "closeTurnStream(true, eventRunId);" in cancelled
    started = event_branch("run.started", "event.type === 'status'")
    assert "closeReasoning(priorRun);" in started
    assert "sealStream(true, priorRun);" in started
    assert "closeTurnStream(false, priorRun);" in started

    stream_chat = block(CHAT_JS, "async function streamChat(text)", "\n  function setRunning(")
    assert stream_chat.count("closeTurnStream(true)") >= 3  # abort, error, finally
    assert "closeReasoning();" in stream_chat  # abort and finally latch too


def test_panels_are_scoped_to_the_active_run_id() -> None:
    gate = block(CHAT_JS, "function applyStreamEvent(event)", "if (event.type === 'run.started')")
    assert "const eventRunId = event.run_id || null;" in gate
    assert "const priorRun = state.activeRun;" in gate
    assert "event.run_id !== state.activeRun" in gate
    assert "event.seq <= state.lastSeq" in gate
    assert "state.lastSeq = -1;" in gate  # per-run sequence reset
    # pending is NOT a run id: only run.started may adopt one.
    assert "state.activeRun === 'pending' && event.type !== 'run.started'" in gate
    assert "state.activeRun === 'pending' && event.run_id" not in gate

    started = event_branch("run.started", "event.type === 'status'")
    assert "closeTurnStream(false, priorRun);" in started
    assert "state.activeRun = eventRunId;" in started
    assert "state.reasoningClosedRuns.delete(eventRunId);" in started

    delta = event_branch("thinking.delta", "event.type === 'tool.started'")
    assert "eventRunId !== state.stream.runId" in delta
    assert "reasoningClosed(eventRunId)" in delta

    finalize = block(CHAT_JS, "function finalizeTurnReasoning(terminal = false", "function detachReasoningPanel(")
    assert "stream.runId !== runId" in finalize

    attach = block(CHAT_JS, "function attachReasoningPanel(stream)", "function appendReasoning(")
    assert "reasoning.runId = stream.runId" in attach
    assert "reasoning.panel.dataset.reasoningRun" in attach

    close = block(CHAT_JS, "function closeTurnStream(terminal = true", "\n  function sealStream(")
    assert "runId !== state.stream.runId" in close


def test_state_is_cleared_and_old_panels_cannot_be_reattached() -> None:
    close = block(CHAT_JS, "function closeTurnStream(terminal = true", "\n  function sealStream(")
    assert "state.stream = null;" in close
    assert "state.turnReasoningStreams = [];" in close
    assert "state.turnReasoningSeen = false;" in close
    assert "clearNoReasoningNotes(stream);" in close  # retraction when reasoning arrives later

    detach = block(CHAT_JS, "function detachReasoningPanel(stream)", "function noteNoReasoning(")
    assert "panel.parentNode === node" in detach  # detach only from the exact message
    assert "panel.remove();" in detach
    assert "stream.reasoning = null;" in detach

    append = block(CHAT_JS, "function appendReasoning(delta)", "function queueReasoningRender()")
    assert "stream.reasoning.node !== stream.node" in append
    assert "stream.reasoning.panel.parentNode !== stream.node" in append

    attach = block(CHAT_JS, "function attachReasoningPanel(stream)", "function appendReasoning(")
    assert "stream.node.isConnected" in attach
    assert "state.turnReasoningStreams.push(stream)" in attach

    select = block(CHAT_JS, "async function selectConversation(id", "if (!id) { renderEmpty()")
    assert "state.stream = null;" in select
    assert "state.turnReasoningStreams = [];" in select
    assert "state.noReasoningNote = false;" in select
    assert "state.turnNodes = [];" in select

    stream_chat = block(CHAT_JS, "async function streamChat(text)", "const headers = {")
    assert "state.stream = null;" in stream_chat
    assert "state.turnReasoningStreams = [];" in stream_chat
    assert "state.turnReasoningSeen = false;" in stream_chat
    assert "state.noReasoningNote = false;" in stream_chat
    assert "state.turnNodes = [];" in stream_chat


def test_real_reasoning_streaming_is_unchanged() -> None:
    append = block(CHAT_JS, "function appendReasoning(delta)", "function queueReasoningRender()")
    assert "boundedText(stream.reasoningBuffer + delta, REASONING_RESULT_LIMIT)" in append
    assert "panel.dataset.reasoningState = 'streaming'" in append
    assert "panel.open = true" in append
    assert "queueReasoningRender();" in append
    assert "state.turnTools" not in append
    assert "event.content" not in append

    render = block(CHAT_JS, "function queueReasoningRender()", "function finalizeReasoning(")
    assert "stream.reasoning.body.textContent = boundedText(stream.reasoningBuffer, REASONING_RESULT_LIMIT);" in render
    assert "RENDER_THROTTLE_MS" in render

    finalize = block(CHAT_JS, "function finalizeReasoning(", "function finalizeTurnReasoning(")
    assert "body.textContent = boundedText(stream.reasoningBuffer, REASONING_RESULT_LIMIT);" in finalize
    assert "panel.dataset.reasoningState = 'received';" in finalize
    assert "contentEl" not in finalize

    node = block(CHAT_JS, "function reasoningNode()", "function attachReasoningPanel(")
    assert "createElement('details')" in node
    assert "createElement('summary')" in node
    assert "createElement('pre')" in node
    assert "body.className = 'chat-reasoning__body';" in node
    assert "body.textContent = '';" in node
    assert "setAttribute('aria-live', 'polite')" in node
    assert "setAttribute('aria-atomic', 'true')" in node
    assert "body.setAttribute('aria-live'" not in node
    assert "panel.addEventListener('toggle'" in node

    for needle in (
        ".chat-reasoning {",
        ".chat-reasoning__toggle:focus-visible",
        ".chat-reasoning[data-reasoning-state='streaming']",
        "white-space: pre-wrap",
        "max-height:",
        "overflow: auto",
    ):
        assert needle in CHAT_CSS
    reduced = CHAT_CSS.split("@media (prefers-reduced-motion: reduce) {", 1)[1].split("\n}", 1)[0]
    assert ".chat-reasoning__toggle" in reduced
    assert ".chat-reasoning__icon" in reduced


def test_runtime_seal_removes_indicator_and_scopes_note_retraction() -> None:
    """Execute the shipped seal/clear functions, not just their source text."""
    harness = r"""
const trace = [];
const t = (...a) => trace.push(a);
function ok(cond, msg) { if (!cond) throw new Error(msg); }
function saw(name, ...args) {
  return trace.some((row) => row[0] === name && args.every((arg, i) => row[i + 1] === arg));
}
const state = { stream: null, activeRun: 'r1', noReasoningNote: false };
let rendered = '';
const window = { LEOMarkdown: { renderInto(el, text) { t('renderInto', text); rendered = text; } } };
const finalizeTurnReasoning = (...a) => t('finalizeTurnReasoning', ...a);
const finalizeReasoning = (...a) => t('finalizeReasoning', ...a);
const thinking = { removed: false, remove() { this.removed = true; } };
const noteA = { removed: false, remove() { this.removed = true; } };
const node = {
  isConnected: true,
  querySelector(sel) { return sel === '.chat-thinking' ? thinking : null; },
  querySelectorAll(sel) { return sel === '.chat-reasoning-none' ? [noteA] : []; },
};
state.stream = { node, contentEl: {}, buffer: 'final answer', sealed: false, runId: 'r1' };
%s

// A mismatched run must never seal the newer run's node.
sealStream(true, 'foreign-run');
ok(state.stream.sealed === false, 'sealStream sealed another run');
ok(thinking.removed === false, 'sealStream removed another run\'s indicator');

sealStream(true, 'r1');
ok(state.stream.sealed === true, 'terminal seal left the node open');
ok(thinking.removed === true, 'terminal seal left the thinking indicator running');
ok(rendered === 'final answer', 'terminal seal did not render the final buffer');
ok(saw('finalizeTurnReasoning', true), 'terminal seal did not finalize reasoning');

// clearNoReasoningNotes must touch only this run's message.
const noteB = { removed: false, remove() { this.removed = true; } };
const otherNode = { querySelectorAll() { return [noteB]; } };
const streamA = { node, runId: 'r1' };
const streamB = { node: otherNode, runId: 'old-run' };
state.stream = streamA;
state.noReasoningNote = true;
clearNoReasoningNotes(streamA);
ok(noteA.removed === true, 'current message note was not retracted');
ok(noteB.removed === false, 'another run\'s note was retracted');
ok(state.noReasoningNote === false, 'turn-level note flag was not cleared');
state.noReasoningNote = true;
clearNoReasoningNotes(streamB);
ok(noteB.removed === false, 'foreign run note was retracted');
ok(state.noReasoningNote === true, 'foreign run cleared the current note flag');
""" % (
        "\n\n".join(
            [
                js_block("function clearNoReasoningNotes(", "\n  // Terminal paths close"),
                js_block("function sealStream(terminal = false, runId = null)", "function queueStreamRender()"),
            ]
        )
    )
    run_node(harness)


def test_runtime_cancellation_and_supersession_close_the_run() -> None:
    """Terminal and superseding frames must seal and latch their run."""
    driver = r"""
const E = (type, run_id, extra = {}) => Object.assign({ type, run_id }, extra);
function reset(runId) {
  state.activeRun = runId;
  state.lastSeq = -1;
  state.stream = null;
  state.reasoningClosedRuns = new Set();
  state.turnReasoningSeen = false;
  trace.length = 0;
}

// Cancellation: seal + close + latch, then a late delta changes nothing.
reset('pending');
applyStreamEvent(E('run.started', 'r1'));
trace.length = 0;
applyStreamEvent(E('status', 'r1', { status: 'cancelled' }));
ok(saw('sealStream', true, 'r1'), 'cancelled run was not sealed');
ok(saw('closeTurnStream', true, 'r1'), 'cancelled run was not closed');
ok(reasoningClosed('r1'), 'cancelled run was not latched');
trace.length = 0;
applyStreamEvent(E('thinking.delta', 'r1', { delta: 'too late' }));
ok(trace.length === 0, 'late delta after cancel produced effects: ' + JSON.stringify(trace));
ok(state.stream === null, 'late delta after cancel opened an assistant node');

// Supersession by a new run.started: the old node is sealed and latched.
reset('r1');
applyStreamEvent(E('run.started', 'r1'));
trace.length = 0;
applyStreamEvent(E('run.started', 'r2'));
ok(saw('sealStream', true, 'r1'), 'superseded run was not sealed');
ok(saw('closeTurnStream', false, 'r1'), 'superseded run was not released');
ok(state.activeRun === 'r2', 'new run id was not adopted');
ok(reasoningClosed('r1') && !reasoningClosed('r2'), 'supersession latch is wrong');
trace.length = 0;
applyStreamEvent(E('thinking.delta', 'r1', { delta: 'stale' }));
ok(trace.length === 0, 'superseded run delta was accepted: ' + JSON.stringify(trace));
applyStreamEvent(E('thinking.delta', 'r2', { delta: 'fresh' }));
ok(saw('appendReasoning', 'fresh'), 'new run delta was rejected');
"""
    run_node(JS_ASSERTIONS + "\n\n" + EVENT_FUNCTIONS + "\n" + driver)
