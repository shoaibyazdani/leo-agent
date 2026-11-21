"""Static contracts for compact tool cards and the side-panel banner icon.

Both defects were visible in the shipped dashboard: tool cards expanded all
arguments/result JSON by default, and banner icons were inserted into the DOM
as raw SVG markup text (wrapping one character per line in the narrow panel).
The tests read the shipped JS/CSS assets so the regressions stay fixed.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "leo" / "interfaces" / "static"
CHAT_JS = (STATIC / "ui" / "conversation" / "chat.js").read_text(encoding="utf-8")
CHAT_CSS = (STATIC / "ui" / "conversation" / "chat.css").read_text(encoding="utf-8")


def block(source: str, start: str, end: str) -> str:
    assert start in source, f"missing marker {start!r}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing closing marker {end!r}"
    return tail.split(end, 1)[0]


def test_tool_cards_are_collapsed_by_default_with_a_button_summary() -> None:
    node = block(CHAT_JS, "function toolNode(", "function turnToolRow(")
    assert "head.type = 'button'" in node
    assert "head.className = 'chat-tool__head chat-tool__toggle'" in node
    assert "head.setAttribute('aria-controls', body.id)" in node
    assert "head.setAttribute('aria-expanded', 'false')" in node
    assert "head.setAttribute('aria-label', `Show details for ${toolName}`)" in node
    assert "body.hidden = true;" in node
    assert "head.addEventListener('click'" in node
    assert "body.hidden = !expanded;" in node
    assert "aria-expanded" in node and "Hide" in node and "Show" in node
    # Summary row shows icon, name, risk, status and a bounded result excerpt.
    assert "head.append(iconNode('tools', 'icon icon--sm'), nameEl, riskEl, statusEl, excerpt)" in node
    assert "excerpt.className = 'chat-tool__excerpt'" in node
    assert "statusText.textContent" in node
    excerpt_helper = block(CHAT_JS, "function toolExcerpt(", "function setToolResult(")
    assert "replace(/\\s+/g, ' ')" in excerpt_helper
    assert "text.slice(0, limit)}…" in excerpt_helper


def test_live_and_historic_results_update_the_same_compact_summary() -> None:
    helper = block(CHAT_JS, "function setToolResult(tool, value)", "function toolNode(")
    assert "tool.resultPre.textContent = boundedText(text || '')" in helper
    assert "tool.excerpt.textContent = toolExcerpt(text || '')" in helper
    live = block(CHAT_JS, "} else if (event.type === 'tool.finished')", "} else if (event.type === 'approval.required')")
    assert "setToolResult(tool, event.result || '')" in live
    assert "blocked ? 'blocked' : failed ? 'failed' : 'done'" in live
    history = block(CHAT_JS, "function appendHistoryMessage(message, thread)", "async function selectConversation(")
    assert "} else if (message.role === 'tool') {" in history
    assert "setToolResult(tool, result)" in history
    assert "tool.statusText.textContent = 'blocked'" in history
    assert "status: failed ? 'failed' : 'done'" in history


def test_banner_icons_are_real_elements_not_raw_markup_text() -> None:
    helper = block(CHAT_JS, "function iconNode(name, cls = 'icon')", "function timeAgo(")
    assert "host.innerHTML = ico(name, cls)" in helper
    assert "return host.firstElementChild;" in helper
    for needle in (
        "el.append(iconNode(kind === 'danger' ? 'alert' : 'approvals', 'icon icon--sm'), body)",
        "head.append(iconNode('tools', 'icon icon--sm'), nameEl, riskEl, statusEl, excerpt)",
        "row.append(iconNode('tools', 'icon icon--sm'), label, status)",
    ):
        assert needle in CHAT_JS
    assert "el.append(ico(" not in CHAT_JS
    assert "head.append(ico(" not in CHAT_JS
    assert "row.append(ico(" not in CHAT_JS


def test_tool_and_banner_css_keep_the_summary_readable_in_narrow_panels() -> None:
    for needle in (
        ".chat-tool__toggle:focus-visible",
        ".chat-tool__toggle[aria-expanded='true']::after",
        ".chat-tool__excerpt {",
        ".chat-tool__body[hidden] { display: none; }",
        ".chat-banners { display: flex; flex-direction: column; gap: var(--space-3); min-width: 0; }",
        ".chat-banner > .icon { flex: 0 0 auto; }",
        ".chat-banner__title, .chat-banner__text { overflow-wrap: anywhere; white-space: normal; }",
    ):
        assert needle in CHAT_CSS
    banner_rule = CHAT_CSS.split(".chat-banner {", 1)[1].split("}", 1)[0]
    assert "align-items: flex-start" in banner_rule
    reduced = CHAT_CSS.split("@media (prefers-reduced-motion: reduce) {", 1)[1].split("\n}", 1)[0]
    assert ".chat-tool__toggle" in reduced


def test_open_conversation_follows_live_events_without_a_refresh() -> None:
    """Turns produced by other clients must appear in the open thread.

    The sidebar already polls; the thread itself only re-rendered on reload.
    chat.js subscribes to /events/stream with a single-use token, appends rows
    after the last rendered id, and dedupes the locally streamed turn.
    """
    stream = block(CHAT_JS, "function connectEventStream(", "async function selectConversation(")
    assert "api('/events/stream-token'" in stream
    assert "new EventSource(" in stream
    assert "&max_seconds=900" in stream
    assert "source.onerror" in stream and "scheduleEventStreamRetry" in stream
    bus = block(CHAT_JS, "function handleBusEvent(event)", "function scheduleEventStreamRetry(")
    assert "event.topic !== 'chat.completed'" in bus
    assert "payload.channel === 'dashboard'" in bus
    assert "syncLastMessageId()" in bus
    assert "appendNewMessages()" in bus
    append = block(CHAT_JS, "async function appendNewMessages(", "function handleBusEvent(")
    assert "?after_id=${state.lastMessageId || 0}" in append
    assert "appendHistoryMessage(message, thread)" in append
    assert "state.activeId !== id" in append, "a switch mid-fetch must not render stale rows"
    # Both the initial load and the incremental append share one renderer.
    assert "function appendHistoryMessage(message, thread)" in CHAT_JS
    assert CHAT_JS.count("appendHistoryMessage(message") >= 2
    assert "data.latest_id" in CHAT_JS
    assert "connectEventStream();" in CHAT_JS
    # The local turn's tail is recorded so its bus event cannot duplicate rows.
    assert "syncLastMessageId();" in CHAT_JS


def test_repeated_approval_cards_are_grouped_into_one_collapsed_card() -> None:
    """A 400-turn session must not flood the thread with approval cards."""
    helper = block(CHAT_JS, "function appendApprovalCard(node, host)", "// Rehydrate pending approvals")
    assert "chat-approval-group" in helper
    assert "group.dataset.approvalName = name" in helper
    assert "group.dataset.count = String(Number(group.dataset.count || 0) + 1)" in helper
    assert "items.appendChild(node)" in helper
    # Existing approval nodes keep their read-only id so reconciliation still works.
    node = block(CHAT_JS, "function approvalNode(", "function markApprovalResolved(")
    assert "card.dataset.approvalId = id" in node
    assert "card.dataset.approvalName = String(name || 'A tool call')" in node
    assert "open.textContent = 'Decide in Approvals view'" in node
    # Both insertion sites use the grouped helper.
    assert CHAT_JS.count("appendApprovalCard(") >= 3
    assert "threadEl().appendChild(approvalNode(" not in CHAT_JS
    # Collapsed styling exists and remains keyboard accessible.
    css_needles = (
        ".chat-approval-group {",
        ".chat-approval-group__details > summary {",
        ".chat-approval-group__items { display: flex;",
        ".chat-approval-group__details > summary:focus-visible",
    )
    for needle in css_needles:
        assert needle in CHAT_CSS


def test_reasoning_panel_sits_above_the_answer_and_collapses_when_done() -> None:
    """Providers may stream reasoning after the text; it must not land below.

    A stale-looking THINKING card at the end of a finished answer was the
    shipped bug: the panel was appended to the message node. It is inserted
    before the content node, and a terminal run collapses it.
    """
    helper = block(CHAT_JS, "function attachReasoningPanel(stream)", "function appendReasoning(")
    assert ".chat-msg__content" in helper
    assert "insertBefore(reasoning.panel, contentNode)" in helper
    finalize = block(CHAT_JS, "function finalizeReasoning(stream, terminal = false)", "function finalizeTurnReasoning(")
    assert "panel.open = false" in finalize
    assert "terminal && stream.reasoningSeen" in finalize
