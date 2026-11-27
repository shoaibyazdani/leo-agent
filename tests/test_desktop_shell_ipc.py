"""Phase 9 Electron shell contract: notification-only IPC, no approval authority."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "desktop" / "main.js").read_text(encoding="utf-8")
PRELOAD = (ROOT / "desktop" / "preload.js").read_text(encoding="utf-8")


def block(source: str, start: str, end: str) -> str:
    assert start in source
    tail = source.split(start, 1)[1]
    assert end in tail
    return tail.split(end, 1)[0]


def test_preload_exposes_no_approval_channel() -> None:
    assert "NOTIFY: 'leo:notify'" in PRELOAD
    assert "notify(title, body)" in PRELOAD
    assert "ipcRenderer.invoke(CHANNELS.NOTIFY" in PRELOAD
    assert "ipcRenderer.send" not in PRELOAD
    assert "channel !== CHANNELS.NOTIFY" in PRELOAD
    assert not re.search(r"leo:(approve|deny|resolve|approval)", PRELOAD)
    assert "fetch(" not in PRELOAD
    assert "process.env" not in PRELOAD and "Authorization" not in PRELOAD


def test_notify_handler_has_no_decision_payload_and_focuses_approvals() -> None:
    handler = block(MAIN, "ipcMain.handle(CHANNELS.NOTIFY", "\n});")
    assert "safeText(payload && payload.title ? payload.title : 'LEO', 120)" in handler
    assert "safeText(payload && payload.body ? payload.body : '', 1200)" in handler
    assert "Notification.isSupported()" in handler
    assert "new Notification({ title: title || 'LEO', body })" in handler
    assert "notification.on('click'" in handler
    assert "showView('approvals')" in handler
    assert "mainWindow.show()" in handler
    for forbidden in ("approve", "deny", "resolve", "fetch(", "gateway", "token", "digest"):
        assert forbidden not in handler.lower(), forbidden


def test_main_channel_allowlist_has_no_extra_ipc() -> None:
    channels = block(MAIN, "const CHANNELS = Object.freeze({", "});")
    assert set(re.findall(r"([A-Z_]+):", channels)) == {
        "HEALTH", "OPEN_SETTINGS", "RELOAD", "API_STATUS", "NOTIFY",
    }
    assert "VIEWS" in MAIN and "approvals" in block(MAIN, "const VIEWS = new Set([", "]);")


@pytest.mark.parametrize("script", ["desktop/main.js", "desktop/preload.js"])
def test_shell_javascript_parses(script: str) -> None:
    if shutil.which("node") is None:
        return
    result = subprocess.run(
        ["node", "--check", str(ROOT / script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
