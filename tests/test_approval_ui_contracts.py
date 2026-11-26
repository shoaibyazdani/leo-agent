"""Static contracts for the Phase 4 card projections.

The web/Electron renderer must show the literal full body with textContent,
offer exactly the frozen choices and send only the digest it displayed. The
TUI mirrors the same projection. The Electron bridge is notification-only:
there is no approval IPC and no second event transport.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "leo" / "interfaces" / "static"
APPROVALS_JS = (STATIC / "ui" / "approvals.js").read_text(encoding="utf-8")
TUI = (ROOT / "leo" / "interfaces" / "tui.py").read_text(encoding="utf-8")
MAIN = (ROOT / "desktop" / "main.js").read_text(encoding="utf-8")
PRELOAD = (ROOT / "desktop" / "preload.js").read_text(encoding="utf-8")


def block(source: str, start: str, end: str) -> str:
    assert start in source, f"missing marker {start!r}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing closing marker {end!r}"
    return tail.split(end, 1)[0]


def test_web_renderer_shows_literal_text_and_every_frozen_choice() -> None:
    assert "window.LEO_APPROVALS = Object.freeze(" in APPROVALS_JS
    assert "function scheduleCardView(card)" in APPROVALS_JS
    render = block(APPROVALS_JS, "function renderCard(", "async function requestJson(")
    assert "container.replaceChildren();" in render
    assert "body.textContent = view.body;" in render
    assert "title.textContent = view.title;" in render
    assert "view.choices.forEach(" in render
    assert "button.textContent = choice.label;" in render
    assert "innerHTML" not in render, "literal body must never be parsed as HTML"
    assert "deny_reason" in render
    assert "digest: ${view.proposal_digest}" in render


def test_web_renderer_sends_only_the_displayed_digest_and_choice() -> None:
    resolve = block(APPROVALS_JS, "function resolve(card, choice)", "window.LEO_APPROVALS")
    assert "view.choices.some((item) => item.value === choice)" in resolve
    assert "choice," in resolve
    assert "confirm_proposal_digest: view.proposal_digest" in resolve
    assert "RESOLVE_ROUTE(view.approval_id)" in resolve
    assert "acknowledge(card)" in APPROVALS_JS
    assert "ACK_ROUTE(view.approval_id)" in APPROVALS_JS
    # The renderer never invents a decision or bypasses the routes.
    assert "approved:" not in resolve
    assert "/resolve?" not in APPROVALS_JS


def test_tui_projects_the_same_full_card_and_presents_before_deciding() -> None:
    assert "def _print_approval_card(self, card: Any)" in TUI
    assert "body.splitlines()" in TUI
    assert "choices: " in TUI
    assert "/approve {payload.get('approval_id')} <choice>" in TUI
    resolve = block(TUI, "async def _resolve_schedule_card_tui(", "# ------------------------------------------------------------- commands")
    assert 'surface="tui"' in resolve
    assert "present_schedule_card(" in resolve
    assert "resolve_schedule_card(" in resolve
    assert 'resolver="operator:cli"' in resolve
    assert "confirm_proposal_digest=str(row[\"digest\"] or \"\")" in resolve
    approvals = block(TUI, 'elif name == "approvals":', 'elif name in {"stop", "cancel"}')
    assert "present_schedule_card(" in approvals
    assert "_print_approval_card(" in approvals


def test_electron_bridge_is_notification_only() -> None:
    assert "NOTIFY: 'leo:notify'" in MAIN
    assert "NOTIFY: 'leo:notify'" in PRELOAD
    handler = block(MAIN, "ipcMain.handle(CHANNELS.NOTIFY", "\n});")
    assert "Notification.isSupported()" in handler
    assert "const notification = new Notification({ title: title || 'LEO', body });" in handler
    assert "notification.on('click'" in handler
    assert "showView('approvals')" in handler
    assert "safeText(" in handler
    for forbidden in ("approve", "deny", "resolve", "fetch(", "gateway", "token"):
        assert forbidden not in handler.lower(), forbidden
    assert "notify(title, body)" in PRELOAD
    assert "ipcRenderer.invoke(CHANNELS.NOTIFY" in PRELOAD
    assert "ipcRenderer.send" not in PRELOAD
    # The generic no-argument invoke allowlist excludes the notification channel.
    assert "channel !== CHANNELS.NOTIFY" in PRELOAD
    # No approval IPC channel exists anywhere in the shell.
    assert not re.search(r"leo:(approve|deny|resolve|approval)", PRELOAD)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_approvals_javascript_parses() -> None:
    result = subprocess.run(
        ["node", "--check", str(STATIC / "ui" / "approvals.js")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
