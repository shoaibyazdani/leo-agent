"""Card UI contract spec: static checks plus an optional real-browser DOM check.

The browser test is skipped when Playwright has no installed browser. The static
checks always run, so the shipped assets cannot silently drift from the card
protocol even in a source-only checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "leo" / "interfaces" / "static"
JS = (STATIC / "ui" / "approvals.js").read_text(encoding="utf-8")
APP = (STATIC / "ui" / "app.js").read_text(encoding="utf-8")
CHAT = (STATIC / "ui" / "conversation" / "chat.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")


def block(source: str, start: str, end: str) -> str:
    assert start in source, f"missing {start}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing {end}"
    return tail.split(end, 1)[0]


def test_shared_renderer_is_loaded_before_both_consumers() -> None:
    approvals = INDEX.index("/assets/ui/approvals.js")
    assert approvals < INDEX.index("/assets/ui/app.js")
    assert approvals < INDEX.index("/assets/ui/conversation/chat.js")


def test_dashboard_renders_cards_and_acks_before_resolving() -> None:
    render = block(APP, "  async function decideScheduleCard(", "  // ---- runtime channel enrollment")
    assert "LEO_APPROVALS.renderCard" in render
    assert ".dataset.resolve" in render  # legacy command rows remain compatible
    decide = block(APP, "  async function decideScheduleCard(", "  function scheduleCardHost(")
    assert "acknowledgeThenResolve" in decide
    assert "renderTerminal" in decide
    assert "approved:" not in decide
    stream = block(APP, "  function ingestEvent(event)", "  function stopEventStream()")
    assert "'approval.presented'" in stream
    assert "refreshApprovals()" in stream
    assert "window.leo.notify(" in stream
    assert "acknowledge(" not in stream


def test_conversation_renders_literal_cards_and_terminal_rewrites() -> None:
    node = block(CHAT, "  function scheduleCardNode(", "  function rewriteScheduleCard(")
    assert "LEO_APPROVALS.renderCard" in node
    assert "LEO_APPROVALS.acknowledge(" in node
    assert "innerHTML" not in node
    bus = block(CHAT, "  function handleBusEvent(event)", "  function scheduleEventStreamRetry()")
    assert "approval.presented" in bus
    assert "approval.resolved" in bus
    assert "rewriteScheduleCard(" in bus
    assert "SCHEDULE_CARD_KINDS" in bus
    sync = block(CHAT, "  async function syncPendingApprovals()", "  function noticeNode(")
    assert "data.cards" in sync
    assert "scheduleCardNode(" in sync


def test_renderer_never_parses_card_text_as_html() -> None:
    render = block(JS, "  function renderCard(", "  async function requestJson(")
    assert "innerHTML" not in render
    assert "body.textContent = view.body;" in render
    assert "paginateText(view.body)" in render
    assert "document.createTextNode(" in render
    resolve = block(JS, "  async function acknowledgeThenResolve(", "  function renderTerminal(")
    assert "await acknowledge(card);" in resolve
    assert "return resolve(card, choice);" in resolve


def _browser():
    playwright = pytest.importorskip("playwright.sync_api")
    try:
        browser = playwright.sync_api.chromium.launch(headless=True)
    except Exception as exc:  # no browser binary in this checkout
        pytest.skip(f"chromium unavailable: {type(exc).__name__}")
    return browser


def test_browser_renders_literal_prompt_markup_and_every_choice() -> None:
    browser = _browser()
    try:
        page = browser.new_page()
        page.set_content(f"<div id='host'></div><script>{JS}</script>")
        result = page.evaluate("""() => {
          const host = document.getElementById('host');
          const body = '<img src=x onerror="window.PWNED=1"> literal **text**';
          const view = window.LEO_APPROVALS.renderCard(host, {
            approval_id: 'apr_dom', kind: 'schedule_create', title: '⏰ schedule',
            body, deadline: 'answer within 15 minutes',
            choices: [{value:'once',label:'Allow'},{value:'session',label:'Allow for this chat'},
                      {value:'always',label:'Always allow new jobs'},{value:'deny',label:'Deny'}],
            proposal_digest: 'digest-dom', deny_reason: null,
          });
          const terminal = document.createElement('div');
          window.LEO_APPROVALS.renderTerminal(terminal, {
            approval_id: 'apr_dom', kind: 'schedule_create', title: '⏰ schedule',
            body, status: 'denied', outcome: 'denied', proposal_digest: 'digest-dom',
          }, 'denied');
          return {
            pwned: Boolean(window.PWNED),
            images: host.querySelectorAll('img').length,
            text: host.querySelector('.approval-card__body').textContent,
            choices: [...host.querySelectorAll('button[data-choice]')].map(b => b.dataset.choice),
            deadline: host.querySelector('.approval-card__deadline').textContent,
            digest: host.querySelector('.approval-card__meta').textContent,
            terminal: terminal.textContent,
          };
        }""")
        assert result["pwned"] is False
        assert result["images"] == 0
        assert result["text"] == '<img src=x onerror="window.PWNED=1"> literal **text**'
        assert result["choices"] == ["once", "session", "always", "deny"]
        assert "15 minutes" in result["deadline"]
        assert "digest-dom" in result["digest"]
        assert "denied" in result["terminal"]
    finally:
        browser.close()


def test_browser_pagination_is_lossless_and_deny_reason_is_text() -> None:
    browser = _browser()
    try:
        page = browser.new_page()
        page.set_content(f"<div id='host'></div><script>{JS}</script>")
        result = page.evaluate("""() => {
          const host = document.getElementById('host');
          const body = Array.from({length: 100}, (_, i) => `line ${i} **literal**`).join('\\n');
          window.LEO_APPROVALS.renderCard(host, {
            approval_id: 'apr_page', kind: 'schedule_edit', title: 'change',
            body, choices: [{value:'once',label:'Allow'},{value:'deny',label:'Deny'}],
            proposal_digest: 'd', deny_reason: 'too large for a banner',
          });
          const more = host.querySelector('.approval-card__more');
          while (more && more.isConnected) more.click();
          return {
            shown: host.querySelector('.approval-card__body').textContent,
            reason: host.querySelector('.approval-card__deny-reason').textContent,
          };
        }""")
        expected = "\n".join(f"line {i} **literal**" for i in range(100))
        assert result["shown"] == expected
        assert result["reason"] == "Reason: too large for a banner"
    finally:
        browser.close()
