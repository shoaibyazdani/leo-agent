"""Static contracts for the Electron shell's native menu, IPC bridge and hardening.

The GUI cannot be launched in this environment, so these tests read the owned
JavaScript files and assert the security/contract invariants directly. They
complement (and do not weaken) tests/test_desktop_shell.py.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"
MAIN = (DESKTOP / "main.js").read_text(encoding="utf-8")
PRELOAD = (DESKTOP / "preload.js").read_text(encoding="utf-8")
PACKAGE = json.loads((DESKTOP / "package.json").read_text(encoding="utf-8"))

CHANNELS = {
    "HEALTH": "leo:health",
    "OPEN_SETTINGS": "leo:open-settings",
    "RELOAD": "leo:reload",
    "API_STATUS": "leo:api-status",
}


def function_body(source: str, name: str) -> str:
    """Return the brace-balanced body of a top-level `function name(...)`."""
    start = re.search(rf"function\s+{re.escape(name)}\s*\(", source)
    assert start is not None, f"function {name} missing"
    close = source.find(")", start.end())  # skip default-parameter objects in signatures
    assert close != -1
    opening = source.find("{", close)
    assert opening != -1
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"unbalanced function {name}")


def test_desktop_native_menu_with_keyboard_shortcuts() -> None:
    assert re.search(r"const \{[^}]*\bMenu\b[^}]*\} = require\('electron'\)", MAIN) is not None
    assert "Menu.buildFromTemplate(template)" in MAIN
    assert "Menu.setApplicationMenu(" in MAIN
    assert "app.name || 'LEO'" in MAIN  # macOS app menu
    assert "process.platform === 'darwin'" in MAIN
    for accelerator in (
        "CmdOrCtrl+1",  # chat navigation
        "CmdOrCtrl+2",  # tasks
        "CmdOrCtrl+3",  # agents
        "CmdOrCtrl+4",  # integrations
        "CmdOrCtrl+,",  # settings
        "CmdOrCtrl+R",  # reload/refresh backend
        "CmdOrCtrl+Shift+H",  # explicit health check
        "CmdOrCtrl+Q",  # quit
    ):
        assert f"accelerator: '{accelerator}'" in MAIN, accelerator
    for label in ("Open Chat", "Settings…", "Reload Backend", "Quit LEO"):
        assert f"label: '{label}'" in MAIN, label
    assert "menuNavigate('chat')" in MAIN
    assert "menuSettings()" in MAIN
    assert "void refreshHealth()" in MAIN
    assert "void reloadShell()" in MAIN


def test_desktop_typed_invoke_bridge_and_preload_allowlist() -> None:
    for name, channel in CHANNELS.items():
        assert f"{name}: '{channel}'" in MAIN
        assert f"{name}: '{channel}'" in PRELOAD
        assert f"ipcMain.handle(CHANNELS.{name}," in MAIN
    assert "contextBridge.exposeInMainWorld('leo', Object.freeze({" in PRELOAD
    assert "invoke(channel)" in PRELOAD
    assert "return ipcRenderer.invoke(channel);" in PRELOAD  # no extra args
    assert "ALLOWED.includes(channel)" in PRELOAD
    assert "Promise.reject(new Error(`Unknown LEO channel:" in PRELOAD
    assert "onEvent(listener)" in PRELOAD
    assert "ipcRenderer.removeListener(EVENT_CHANNEL, wrapped)" in PRELOAD
    assert "version: '0.1.0'" not in PRELOAD  # no stale hard-coded version
    assert "invoke('health')" in PRELOAD  # live version/status source
    assert PACKAGE["main"] == "main.js"


def test_desktop_health_gated_startup_and_backend_supervision() -> None:
    assert "require('node:http')" in MAIN and "require('node:https')" in MAIN
    assert "new URL('/health', API_ORIGIN)" in MAIN
    assert "async function waitForHealth(" in MAIN
    assert "await waitForHealth()" in MAIN
    assert "loadShellPage(health)" in MAIN
    assert "if (health.mode === 'online') mainWindow.loadURL(API_URL)" in MAIN
    assert "else mainWindow.loadURL(offlinePage(health))" in MAIN
    for state in ("online", "offline", "degraded"):
        assert f"'{state}'" in MAIN, state
    assert "dialog.showMessageBox({" in MAIN  # explicit offline/degraded notice
    assert "data:text/html;charset=utf-8," in MAIN
    # Supervised child: captured stdio, crash/exit reporting, clean shutdown.
    assert "stdio: ['ignore', 'pipe', 'pipe']" in MAIN
    assert "appendBackendLog('stdout', chunk)" in MAIN
    assert "appendBackendLog('stderr', chunk)" in MAIN
    assert "backend.on('error'" in MAIN
    assert "backend.on('exit'" in MAIN
    assert "backend.kill('SIGTERM')" in MAIN
    assert "child.kill('SIGKILL')" in MAIN
    assert "app.on('render-process-gone'" in MAIN
    assert "mainWindow.webContents.on('crashed'" in MAIN
    # Raw TCP reachability is no longer treated as readiness.
    assert "waitForPort" not in MAIN
    assert "node:net" not in MAIN
    assert "stdio: 'ignore'" not in MAIN


def test_desktop_restrictive_content_security_policy() -> None:
    assert "session.defaultSession.webRequest.onHeadersReceived(" in MAIN
    block = function_body(MAIN, "installContentSecurityPolicy")
    assert "responseHeaders['Content-Security-Policy'] = [CSP]" in block
    assert "originOf(details.url) === API_ORIGIN" in block
    assert "content-security-policy-report-only" in block
    directive_block = re.search(r"const CSP = \[(.*?)\]\.join\('; '\)", MAIN, re.S)
    assert directive_block is not None
    directives = dict(
        directive.split(" ", 1) if " " in directive else (directive, "")
        for directive in re.findall(r'"([^"]+)"', directive_block.group(1))
    )
    assert directives["default-src"] == "'self'"
    assert directives["script-src"] == "'self'"
    assert directives["connect-src"] == "'self'"
    assert directives["object-src"] == "'none'"
    assert directives["frame-src"] == "'none'"
    assert directives["base-uri"] == "'none'"
    assert directives["form-action"] == "'self'"
    assert directives["frame-ancestors"] == "'none'"
    assert "unsafe-eval" not in directive_block.group(1)
    assert "'unsafe-inline'" not in directives["script-src"]
    assert re.search(r"\.join\('; '\) \+ ';'", MAIN) is not None


def test_desktop_navigation_hardening() -> None:
    assert "win.webContents.on('will-navigate', guard)" in MAIN
    assert "win.webContents.on('will-redirect', guard)" in MAIN
    assert "win.webContents.setWindowOpenHandler(" in MAIN
    assert "return { action: 'deny' };" in MAIN
    assert "event.preventDefault();" in MAIN
    assert "shell.openExternal(parsed.href)" in MAIN
    assert "if (isSameOrigin(url)) return;" in MAIN
    assert "originOf(value) === API_ORIGIN" in MAIN
    assert "if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;" in MAIN
    assert "devTools: false" in MAIN
    assert "webSecurity: true" in MAIN
    assert "allowRunningInsecureContent: false" in MAIN
    assert "nodeIntegration: false" in MAIN
    assert "contextIsolation: true" in MAIN
    assert "sandbox: true" in MAIN


def test_desktop_safe_renderer_event_channel() -> None:
    assert "const EVENT_CHANNEL = 'leo:shell-event';" in MAIN
    assert "const EVENT_CHANNEL = 'leo:shell-event';" in PRELOAD
    assert "mainWindow.webContents.send(EVENT_CHANNEL, {" in MAIN
    assert "ipcRenderer.on(EVENT_CHANNEL, wrapped)" in PRELOAD
    for field in ("type", "at", "message", "mode", "healthy", "version"):
        assert f"{field}:" in function_body(MAIN, "notifyRenderer"), field
    assert "safeText(payload.message)" in MAIN
    for event in ("startup-state", "backend-error", "backend-exit", "action-failed"):
        assert f"notifyRenderer('{event}'" in MAIN, event
    # Raw backend output is capped in main memory and never forwarded.
    assert "backendLog.push(" in MAIN
    assert "backendLog.shift()" in MAIN
    assert "backendLog" not in function_body(MAIN, "notifyRenderer")


def test_desktop_never_exposes_raw_api_token() -> None:
    for forbidden in (
        "LEO_API_TOKEN", "sessionStorage", "localStorage", "Authorization",
        "Bearer", "token:", "ipcRenderer.send",
    ):
        assert forbidden not in PRELOAD, forbidden
    assert PRELOAD.count("contextBridge.exposeInMainWorld(") == 1
    status = function_body(MAIN, "apiStatus")
    assert "tokenConfigured: Boolean(API_TOKEN)" in status
    assert "tokenSource: API_TOKEN ? 'environment' : 'none'" in status
    assert "tokenValue" not in MAIN
    assert "Bearer" not in status
    for line in MAIN.splitlines():
        if "ipcMain.handle(" in line:
            assert "API_TOKEN" not in line
            assert "Authorization" not in line
    # The env token is used only for same-origin request headers/health probe.
    assert "requestHeaders.Authorization = `Bearer ${API_TOKEN}`" in MAIN
    assert "headers.Authorization = `Bearer ${API_TOKEN}`" in MAIN
    assert MAIN.count("${API_TOKEN}") == 2
    # The renderer-owned session token is untouched by shell code.
    assert "sessionStorage" not in MAIN


def test_desktop_javascript_syntax() -> None:
    for name in ("main.js", "preload.js"):
        result = subprocess.run(
            ["node", "--check", str(DESKTOP / name)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
