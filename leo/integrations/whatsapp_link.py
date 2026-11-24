"""Operator wiring for the opt-in WhatsApp Web device-link bridge.

Importing this module launches nothing. ``leo whatsapp link`` (or any caller
that has made the policy decision) calls :func:`run_device_link`, which starts
the pinned Node bridge and publishes its PNG QR on a short-lived, loopback-only
page so the owner can scan it from WhatsApp → Linked devices.

Truthfulness and scope: this links a *personal* WhatsApp account through the
unofficial WhatsApp Web protocol (Baileys). It is not the official Cloud API
adapter and it carries WhatsApp ToS / account-ban risk. Scanning the QR shown
here is a real device link. Nothing in this module touches message delivery to
LEO; only the link lifecycle is exposed.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .whatsapp_bridge import BridgeConfig, WhatsAppBridgeManager
from .whatsapp_bridge import manager as _manager_module

BRIDGE_DIR = Path(_manager_module.__file__).resolve().parent
_REPO_ROOT = Path(_manager_module.__file__).resolve().parents[3]
DEFAULT_OWNER = "owner"
DEFAULT_TIMEOUT = 600.0
POLL_INTERVAL = 0.5

CONNECTED = "connected"
TERMINAL_STATES = frozenset({"connected", "logged_out", "error", "passkey_required", "cancelled"})
_STATE_TEXT = {
    "idle": "Bridge idle.",
    "starting": "Starting the WhatsApp bridge…",
    "waiting_for_qr": "Waiting for WhatsApp to issue a QR code…",
    "waiting_for_scan": "QR ready — scan it with WhatsApp → Linked devices → Link a device.",
    "connected": "This computer is linked to WhatsApp.",
    "logged_out": "This device was logged out. Run `leo whatsapp link` again.",
    "error": "The bridge reported an error.",
    "passkey_required": "WhatsApp requires a passkey challenge that this bridge does not sign.",
    "cancelled": "The link attempt was cancelled.",
}


class LinkUnavailable(RuntimeError):
    """Raised when Node or the pinned bridge dependencies are missing."""


def state_message(state: str) -> str:
    """Human-readable text for a bridge lifecycle state."""
    return _STATE_TEXT.get(state, f"State: {state}")


def _session_dir(owner: str) -> Path:
    # actor is fixed to the bridge's default actor; owner is an identity claim
    # recorded inside the session directory and enforced by the manager.
    del owner
    return _REPO_ROOT / ".leo" / "whatsapp" / "default"


def bridge_status(owner: str = DEFAULT_OWNER) -> dict[str, Any]:
    """Offline readiness/link report; never launches the bridge."""
    node = shutil.which("node")
    dependencies = (BRIDGE_DIR / "node_modules" / "@whiskeysockets" / "baileys").is_dir()
    session_dir = _session_dir(owner)
    creds = session_dir / "creds.json"
    recorded_owner: str | None = None
    owner_file = session_dir / "owner.json"
    try:
        record = json.loads(owner_file.read_text(encoding="utf-8"))
        if isinstance(record, dict) and isinstance(record.get("owner_id"), str):
            recorded_owner = record["owner_id"]
    except (OSError, ValueError):
        pass
    return {
        "ready": bool(node and dependencies),
        "node": node,
        "dependencies_installed": dependencies,
        "bridge_dir": str(BRIDGE_DIR),
        "session_dir": str(session_dir),
        "linked": creds.is_file(),
        "owner": recorded_owner,
        "requested_owner": owner,
        "hint": None if node and dependencies else (
            "Install Node 20+ and run `npm ci` in "
            f"{BRIDGE_DIR} before linking."
        ),
    }


def ensure_ready() -> None:
    report = bridge_status()
    if report["node"] is None:
        raise LinkUnavailable(
            "Node 20+ was not found on PATH. Install Node, then run "
            f"`npm ci` in {BRIDGE_DIR}."
        )
    if not report["dependencies_installed"]:
        raise LinkUnavailable(
            "The pinned bridge dependencies are not installed. Run "
            f"`npm ci` in {BRIDGE_DIR} (one-time operator step)."
        )


# --- Loopback QR page -------------------------------------------------------


class _PageState:
    """Thread-safe snapshot read by the tiny loopback HTTP server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {
            "state": "idle",
            "linked": False,
            "error": None,
            "user": None,
            "qr": None,
            "expires_in_ms": 0,
            "message": "Starting…",
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._payload)

    def update(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._payload = dict(payload)


class LinkPage:
    """Short-lived 127.0.0.1 page that shows the latest device-link QR.

    Only the loopback interface is bound. The page path carries a random
    token so another local page cannot guess it, no CORS header is sent, and
    every response is ``no-store``. The payload contains the same PNG data URI
    the manager already returns; the owner token / session secret never
    reaches the browser.
    """

    def __init__(self, owner: str = DEFAULT_OWNER) -> None:
        self.owner = owner
        self.state = _PageState()
        self._token = secrets.token_urlsafe(16)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def path(self) -> str:
        return f"/{self._token}/"

    def start(self) -> str:
        page = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "leo-whatsapp-link"
            sys_version = ""

            def log_message(self, *_args: Any) -> None:  # noqa: D401 - silence
                return

            def _host_ok(self) -> bool:
                host = urlsplit(f"//{self.headers.get('Host', '')}").hostname or ""
                return host.lower() in {"127.0.0.1", "localhost", "::1"}

            def _send(self, status: int, body: bytes, content_type: str, **headers: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy",
                                 "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
                                 "script-src 'unsafe-inline'; connect-src 'self'; "
                                 "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                for key, value in headers.items():
                    self.send_header(key.replace("_", "-"), value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if not self._host_ok():
                    self._send(HTTPStatus.MISDIRECTED_REQUEST, b"bad host", "text/plain; charset=utf-8")
                    return
                path = urlsplit(self.path).path
                if path in {page.path, page.path.rstrip("/")}:
                    self._send(HTTPStatus.OK, _LINK_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == page.path + "state":
                    body = json.dumps(page.state.snapshot(), separators=(",", ":")).encode("utf-8")
                    self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
                    return
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, name="leo-whatsapp-link", daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        port = int(server.server_address[1])
        return f"http://127.0.0.1:{port}{self.path}"

    def update(self, status: dict[str, Any]) -> None:
        qr = status.get("qr") if isinstance(status.get("qr"), dict) else None
        state = str(status.get("state") or "idle")
        self.state.update({
            "state": state,
            "linked": bool(status.get("linked")),
            "error": status.get("error"),
            "user": status.get("user"),
            "qr": qr.get("data_uri") if qr else None,
            "expires_in_ms": int(qr.get("expires_in_ms") or 0) if qr else 0,
            "message": _STATE_TEXT.get(state, f"State: {state}"),
        })

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)


_LINK_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Link WhatsApp · LEO</title>
<style>
  :root { color-scheme: dark; --bg:#111315; --panel:#1a1d20; --line:#2b3036;
          --ivory:#ece6d9; --muted:#9aa3ad; --teal:#3ec6b4; --warn:#e0a458; }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:grid; place-items:center;
         background:var(--bg); color:var(--ivory);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  main { width:min(440px,92vw); background:var(--panel); border:1px solid var(--line);
         border-radius:18px; padding:28px; box-shadow:0 24px 60px rgba(0,0,0,.45); }
  h1 { margin:0 0 4px; font-size:20px; letter-spacing:.01em; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:18px; }
  .qr { display:grid; place-items:center; min-height:280px; background:#fff;
        border-radius:14px; padding:14px; }
  .qr img { width:272px; height:272px; image-rendering:pixelated; display:block; }
  .qr .empty { color:#555; font-size:13px; text-align:center; padding:24px; }
  #status { margin-top:16px; font-size:14px; }
  #detail { color:var(--muted); font-size:12px; margin-top:6px; min-height:1.2em; }
  .ok { color:var(--teal); } .warn { color:var(--warn); }
  ol { margin:14px 0 0; padding-left:20px; color:var(--muted); font-size:13px; }
  li { margin:4px 0; }
</style>
</head>
<body>
<main>
  <h1>Link WhatsApp</h1>
  <div class="sub">WhatsApp → Linked devices → Link a device, then scan this code.</div>
  <div class="qr" id="qr"><div class="empty">Waiting for a QR code…</div></div>
  <div id="status">Starting…</div>
  <div id="detail"></div>
  <ol>
    <li>Open WhatsApp on your phone.</li>
    <li>Tap <strong>Linked devices</strong> → <strong>Link a device</strong>.</li>
    <li>Point your phone at the code above.</li>
  </ol>
</main>
<script>
  const qrBox = document.getElementById('qr');
  const status = document.getElementById('status');
  const detail = document.getElementById('detail');
  let lastQr = null;
  async function poll() {
    try {
      const res = await fetch('./state', { cache: 'no-store' });
      const data = await res.json();
      status.textContent = data.message || data.state;
      status.className = data.state === 'connected' ? 'ok'
        : (data.state === 'error' || data.state === 'logged_out' || data.state === 'passkey_required') ? 'warn' : '';
      if (data.user && data.state === 'connected') {
        detail.textContent = 'Linked as ' + (data.user.name || data.user.id || 'this account') + '.';
      } else if (data.qr) {
        detail.textContent = data.expires_in_ms > 0
          ? 'Code refreshes in about ' + Math.max(1, Math.round(data.expires_in_ms / 1000)) + 's.' : '';
      } else {
        detail.textContent = data.error ? 'Error: ' + data.error : '';
      }
      if (data.qr && data.qr !== lastQr) {
        lastQr = data.qr;
        qrBox.innerHTML = '';
        const img = document.createElement('img');
        img.src = data.qr;
        img.alt = 'WhatsApp device-link QR code';
        qrBox.appendChild(img);
      } else if (!data.qr && lastQr && data.state !== 'connected') {
        lastQr = null;
        qrBox.innerHTML = '<div class="empty">Waiting for a QR code…</div>';
      } else if (!data.qr && !lastQr && data.state === 'connected') {
        qrBox.innerHTML = '<div class="empty">Linked. You can close this page.</div>';
      }
    } catch (err) {
      status.textContent = 'Lost contact with the LEO process.';
      status.className = 'warn';
    }
    setTimeout(poll, 1000);
  }
  poll();
</script>
</body>
</html>
"""


# --- Link lifecycle ---------------------------------------------------------


@dataclass
class LinkResult:
    state: str
    linked: bool
    error: str | None = None
    user: dict[str, Any] | None = None
    owner: str = DEFAULT_OWNER
    page_url: str | None = None
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.state == CONNECTED,
            "state": self.state,
            "linked": self.linked,
            "error": self.error,
            "user": self.user,
            "owner": self.owner,
            "page_url": self.page_url,
            "diagnostics": list(self.diagnostics),
        }


async def run_device_link(
    owner: str = DEFAULT_OWNER,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    stay: bool = False,
    open_page: bool = True,
    notify: Callable[[str, Any], None] | None = None,
) -> LinkResult:
    """Start the bridge, publish its QR, and wait for the device link.

    This is the only entry point in LEO that launches the bridge. It returns
    once the link is connected (``stay=False``), reaches a terminal failure, or
    the overall ``timeout`` (bounded 5–3600 s) elapses. With ``stay=True`` it
    keeps the linked session open until the process is interrupted or the
    session ends.
    """
    ensure_ready()
    if not 5.0 <= float(timeout) <= 3600.0:
        raise ValueError("timeout must be between 5 and 3600 seconds")

    emit = notify or (lambda _kind, _data: None)
    page = LinkPage(owner)
    manager = WhatsAppBridgeManager(BridgeConfig(owner_id=owner, actor="default", enabled=True))
    auth = manager.session_auth
    page_url = page.start()
    final: dict[str, Any] = {"state": "idle", "linked": False, "error": None, "user": None}
    result = LinkResult(state="idle", linked=False, owner=owner, page_url=page_url)
    emit("page", page_url)
    if open_page:
        _open_browser(page_url)

    try:
        initial = await manager.start(owner_id=owner, auth=auth)
        page.update(initial)
        final = initial
        last_state = str(initial.get("state") or "")
        emit("state", initial)

        deadline = time.monotonic() + float(timeout)
        while str(final.get("state")) not in TERMINAL_STATES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await manager.cancel(owner_id=owner, auth=auth)
                final = {**final, "state": "cancelled", "error": "timeout", "linked": False}
                page.update(final)
                emit("state", final)
                break
            await asyncio.sleep(min(POLL_INTERVAL, remaining))
            status = await manager.status(owner_id=owner, auth=auth)
            page.update(status)
            final = status
            state = str(status.get("state") or "")
            if state != last_state:
                last_state = state
                emit("state", status)

        if stay and str(final.get("state")) == CONNECTED:
            # Keep the linked socket alive; poll only to surface a disconnect.
            while True:
                await asyncio.sleep(2.0)
                status = await manager.status(owner_id=owner, auth=auth)
                page.update(status)
                final = status
                if str(status.get("state")) != CONNECTED:
                    emit("state", status)
                    break

        if str(final.get("state")) == CONNECTED:
            # Baileys persists linked-device credentials on creds.update. Wait
            # briefly for the file so stopping the child cannot race the save.
            session_dir = getattr(manager, "session_dir", None)
            if session_dir is not None:
                creds = Path(session_dir) / "creds.json"
                for _ in range(10):
                    if creds.is_file():
                        break
                    await asyncio.sleep(0.3)
            # Let the browser paint the success state before the page closes.
            await asyncio.sleep(2.0)

        state = str(final.get("state") or "idle")
        result = LinkResult(
            state=state,
            linked=bool(final.get("linked")),
            error=final.get("error"),
            user=final.get("user"),
            owner=owner,
            page_url=page_url,
            diagnostics=tuple(manager.diagnostics()) if state != CONNECTED else (),
        )
        emit("result", result.as_dict())
        return result
    except Exception as exc:
        result = LinkResult(
            state="error",
            linked=False,
            error=type(exc).__name__,
            owner=owner,
            page_url=page_url,
            diagnostics=tuple(manager.diagnostics()),
        )
        page.update({
            "state": "error",
            "linked": False,
            "error": type(exc).__name__,
            "user": None,
            "qr": None,
            "message": "The bridge could not start. See the terminal for details.",
        })
        emit("error", exc)
        emit("result", result.as_dict())
        return result
    finally:
        try:
            await manager.stop(owner_id=owner, auth=auth)
        except Exception:
            pass
        page.close()


def _open_browser(url: str) -> None:
    try:
        import webbrowser

        webbrowser.open(url, new=1, autoraise=True)
    except Exception:
        # Headless/sandboxed environments simply keep the URL on stdout.
        return


__all__ = [
    "BRIDGE_DIR",
    "CONNECTED",
    "DEFAULT_OWNER",
    "DEFAULT_TIMEOUT",
    "LinkPage",
    "LinkResult",
    "LinkUnavailable",
    "TERMINAL_STATES",
    "bridge_status",
    "ensure_ready",
    "run_device_link",
    "state_message",
]
