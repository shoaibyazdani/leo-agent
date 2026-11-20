"""Short-lived loopback previews for workspace-contained files.

``preview_local_file`` reads a *single* validated regular file and exposes it
from an ephemeral HTTP server bound to 127.0.0.1 on an OS-assigned port. The
returned address is an ordinary http URL, so the existing browser tool can open
it without weakening its scheme guard: the browser's http(s)-only policy stays
exactly as it is, and no interpreter, shell or host browser is launched.

Security properties of this toolset:

* The path is resolved through the shared workspace/allowed-root resolver
  (:func:`leo.tools.file_tools._safe_path`) with symlink resolution, then
  re-checked at request time against the same roots, the same resolved path and
  the same inode. A swap after creation cannot turn a preview into a file read
  outside the roots. Hard links are refused at creation and on every request:
  a link inside a root can name an inode that lives outside it.
* Every HTTP request is written to the durable ``audit_log`` (and emitted as a
  bounded in-process event) with actor, preview token id, resolved path, status
  and byte count. File content is never part of an audit record.
* The listener rejects any Host header other than its own assigned loopback
  authority, so a DNS-rebinding page cannot use the browser as a bridge to it.
* Nothing is derived from the URL except an unguessable token. There is no
  directory listing, no caller-chosen host, no caller-chosen port and no
  caller-chosen path. A token serves exactly one file, for a bounded TTL.
* Content is static bytes with a fixed MIME allowlist. No template engine, CGI,
  interpreter or executable content type is involved. HTML gets a strict CSP
  (including ``sandbox``) and nosniff headers so inline scripts and network
  beacons in a previewed document are not executed.
* Responses and requests are size-bounded, the listening socket is loopback
  only, and every preview (plus the server itself) is torn down on TTL expiry
  or ``preview_close``.
* Loopback navigation is granted only through :func:`registered_preview`:
  ``browser_navigate`` checks that the caller supplied the exact URL issued by
  ``preview_local_file`` for that actor, and only then calls the browser with
  a token-and-port-bound loopback allowance. Ordinary loopback URLs, other
  actors' tokens and the local-file scheme remain blocked by the browser's
  SSRF policy.

``services`` is accepted for the shared ``register(registry, services=...)``
contract and ignored: the durable state here is the process-local server and
the filesystem itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import secrets
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .. import db
from ..contracts import RiskLevel, ToolSpec
from .file_tools import MAX_FILE_BYTES, _allowed_roots, _safe_path

HOST = "127.0.0.1"
DEFAULT_TTL_SECONDS = 120.0
MIN_TTL_SECONDS = 1.0
MAX_TTL_SECONDS = 300.0
MAX_REQUEST_BYTES = 4096
MAX_URL_BYTES = 2048
_URL_PREFIX = "/preview/"
_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
_AUDIT_ACTION = "preview.serve"
_MAX_HOST_HEADER = 128

# The token in a preview URL is a bearer credential. Audit rows, durable
# events and API feeds must only ever carry a salted, truncated digest of it;
# the raw token appears exclusively in the one-time preview tool response.
_TOKEN_AUDIT_SALT = secrets.token_bytes(32)
_TOKEN_DIGEST_CHARS = 16


def token_audit_id(token: str | None) -> str:
    """Stable-within-process correlation id that cannot be replayed as a token."""
    if not token:
        return ""
    digest = hashlib.sha256(
        _TOKEN_AUDIT_SALT + str(token).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest[:_TOKEN_DIGEST_CHARS]}"

# Fixed mapping: never guess a content type from an unknown name, and never
# serve active/executable types (scripts, SVG, PDF with embedded JS, ...).
_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_CSP = (
    "default-src 'none'; script-src 'none'; object-src 'none'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "media-src data: blob:; font-src data:; connect-src 'none'; "
    "form-action 'none'; base-uri 'none'; frame-ancestors 'none'; sandbox"
)


@dataclass(frozen=True)
class _Preview:
    token: str
    path: Path
    roots: tuple[Path, ...]
    actor: str
    call_id: str | None
    port: int
    expires_at: float
    ttl_seconds: float
    device: int
    inode: int
    size: int
    content_type: str


class _PreviewState:
    """Process-local preview table plus its single loopback server."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.previews: dict[str, _Preview] = {}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()
        # Captured from the creating tool call so request-time audits can emit
        # an in-process event on the owner loop. The durable row is synchronous.
        self.events: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None


_STATE = _PreviewState()


def _within_roots(candidate: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _purge_expired_locked(state: _PreviewState) -> int:
    now = time.monotonic()
    expired = [token for token, item in state.previews.items() if item.expires_at <= now]
    for token in expired:
        state.previews.pop(token, None)
    return len(expired)


def _detach_server_locked(state: _PreviewState):
    """Take ownership of the running server so it can be stopped without a lock."""
    if state.server is None:
        return None
    snapshot = (state.server, state.thread)
    state.server = None
    state.thread = None
    state.stopping.set()
    return snapshot


def _stop_server(snapshot: Any) -> None:
    server, thread = snapshot
    try:
        server.shutdown()
    except Exception:
        pass
    try:
        server.server_close()
    except Exception:
        pass
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.0)


def _janitor(state: _PreviewState, stopping: threading.Event) -> None:
    """Expire tokens even when nobody touches the HTTP server again."""
    while not stopping.is_set():
        stopping.wait(0.25)
        if stopping.is_set():
            return
        with state.lock:
            if state.stopping is not stopping:  # a newer generation took over
                return
            _purge_expired_locked(state)
            snapshot = _detach_server_locked(state) if not state.previews else None
        if snapshot is not None:
            _stop_server(snapshot)


def _ensure_server_locked(state: _PreviewState) -> ThreadingHTTPServer:
    if state.server is not None:
        return state.server
    # Port 0 means "let the OS pick": no fixed port, no collision, no caller
    # input. The bind address is hardcoded, never taken from args or settings.
    server = ThreadingHTTPServer((HOST, 0), _PreviewRequestHandler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.2},
        name="leo-preview-http",
        daemon=True,
    )
    state.server = server
    state.thread = thread
    state.stopping = threading.Event()
    threading.Thread(
        target=_janitor,
        args=(state, state.stopping),
        name="leo-preview-ttl",
        daemon=True,
    ).start()
    thread.start()
    return server


def _token_from_path(target: Any) -> str | None:
    """Accept only an exact ``/preview/<token>`` request target."""
    if not isinstance(target, str) or len(target.encode("utf-8", "replace")) > MAX_URL_BYTES:
        return None
    try:
        parts = urlsplit(target)
    except ValueError:
        return None
    if parts.scheme or parts.netloc or parts.query or parts.fragment:
        return None
    path = parts.path
    if not path.startswith(_URL_PREFIX):
        return None
    token = path[len(_URL_PREFIX) :]
    if not token or len(token) > 64 or any(char not in _TOKEN_CHARS for char in token):
        return None
    return token


def _token_from_url(value: str) -> tuple[str, int | None] | None:
    """Parse a preview URL for close; never accept another host or scheme."""
    if len(value) > MAX_URL_BYTES:
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "http" or (parts.hostname or "").lower() != HOST:
        return None
    if parts.username or parts.password or parts.query or parts.fragment:
        return None
    token = _token_from_path(parts.path)
    return (token, port) if token else None


def canonical_preview_url(preview: _Preview) -> str:
    """The one URL issued for this token; never reassemble it from user input."""
    return f"http://{HOST}:{preview.port}{_URL_PREFIX}{preview.token}"


def _audit_served(
    preview: _Preview | None,
    token: str | None,
    status: int,
    byte_count: int,
    method: str,
    failure: str | None = None,
) -> None:
    """Record metadata only. Never include payload bytes in an audit record."""
    record: dict[str, Any] = {
        "actor": preview.actor if preview is not None else "anonymous",
        "token_id": token_audit_id(token),
        "call_id": preview.call_id if preview is not None else None,
        "path": str(preview.path) if preview is not None else None,
        "status": int(status),
        "decision": "allowed" if status == 200 else "denied",
        "bytes": max(0, int(byte_count)),
        "byte_count": max(0, int(byte_count)),
        "method": str(method)[:16],
        "reason": str(failure or "")[:240],
    }
    # The HTTP handler runs on the server thread, outside the executor's event
    # loop and transaction. A short-lived synchronous SQLite write keeps the
    # audit durable even if the loop has moved on; PRAGMA busy_timeout shares
    # the database safely with the aiosqlite writer.
    if db.database_path_configured():
        try:
            connection = sqlite3.connect(str(db.get_db_path()), timeout=5.0)
            try:
                connection.execute("PRAGMA busy_timeout=5000")
                connection.execute(
                    "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,correlation_id) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        datetime.now(timezone.utc).isoformat(), record["actor"],
                        _AUDIT_ACTION, record["path"], "medium",
                        record["decision"], record["reason"],
                        json.dumps(record, default=str, ensure_ascii=False)[:1000],
                        record["call_id"],
                    ),
                )
                connection.commit()
            finally:
                connection.close()
        except Exception:
            pass  # an audit failure must not turn into a file-serving failure
    state = _STATE
    loop, events = state.loop, state.events
    if loop is not None and events is not None and not loop.is_closed():
        try:
            asyncio.run_coroutine_threadsafe(events.emit(_AUDIT_ACTION, **record), loop)
        except Exception:
            pass


def _read_preview(preview: _Preview) -> tuple[bytes | None, str | None]:
    """Read the validated file safely, returning (payload, error_status)."""
    try:
        resolved = preview.path.resolve()
        if resolved != preview.path or not _within_roots(resolved, preview.roots):
            return None, "preview target moved outside allowed roots"
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(preview.path), flags)
    except (OSError, RuntimeError, ValueError):
        return None, "preview file is no longer available"
    try:
        try:
            info = os.fstat(descriptor)
        except OSError:
            return None, "preview file is no longer available"
        if not stat.S_ISREG(info.st_mode):
            return None, "preview target is not a regular file"
        # A hard link inside an allowed root still names the original inode,
        # which may live outside every root. A second link created after the
        # preview was issued changes nlink without changing dev/ino.
        if info.st_nlink > 1:
            return None, "refusing to serve a hard-linked file"
        if (info.st_dev, info.st_ino) != (preview.device, preview.inode):
            return None, "preview file changed since creation"
        if info.st_size > MAX_FILE_BYTES:
            return None, "preview file grew beyond the size limit"
        chunks: list[bytes] = []
        remaining = MAX_FILE_BYTES + 1
        while remaining > 0:
            try:
                block = os.read(descriptor, min(65_536, remaining))
            except OSError:
                return None, "preview file could not be read"
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) > MAX_FILE_BYTES:
        return None, "preview file grew beyond the size limit"
    return payload, None


class _PreviewRequestHandler(BaseHTTPRequestHandler):
    """Static, single-token, loopback-only request handler."""

    protocol_version = "HTTP/1.0"  # no keep-alive: bounded connection lifetime
    server_version = "LEOPreview/1.0"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5.0)
        self._audited = False
        self._preview: _Preview | None = None

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # never log local paths or tokens

    def _host_allowed(self) -> bool:
        """Only our own loopback authority may reach this listener.

        A web page can resolve an attacker name to 127.0.0.1 and issue
        cross-origin requests; browsers still send the attacker's Host header,
        so checking the bound address alone would not stop DNS rebinding.
        """
        value = self.headers.get("Host")
        if not isinstance(value, str) or not value or len(value) > _MAX_HOST_HEADER:
            return False
        host = value.strip().lower()
        try:
            port = int(self.server.server_address[1])
        except (AttributeError, IndexError, TypeError, ValueError):
            return False
        return host in {f"{HOST}:{port}", f"localhost:{port}"}

    def _audit_current(self, status: int, byte_count: int, failure: str | None = None) -> None:
        if getattr(self, "_audited", False):
            return
        self._audited = True
        token = _token_from_path(getattr(self, "path", ""))
        _audit_served(
            getattr(self, "_preview", None), token, status, byte_count,
            getattr(self, "command", "UNKNOWN"), failure,
        )

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # Malformed-line and unsupported-method replies are HTTP events too.
        self._audit_current(code, 0, message or "malformed request")
        super().send_error(code, message, explain)

    def _reply(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self._audit_current(status, len(body), None if status == 200 else body.decode("utf-8", "replace"))
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", _CSP)
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD" and body:
                self.wfile.write(body)
        except (OSError, ValueError):
            pass  # the client went away; the preview is still bounded by TTL

    def _reject(self, status: int, message: str) -> None:
        self._reply(status, message.encode("utf-8"), "text/plain; charset=utf-8")

    def _serve(self) -> None:
        if not self._host_allowed():
            self._reject(400, "Host header must match the assigned loopback listener")
            return
        if self.headers.get("Transfer-Encoding"):
            self._reject(400, "chunked requests are not accepted")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._reject(400, "invalid Content-Length")
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._reject(413, "request body too large")
            return
        token = _token_from_path(self.path)
        if token is None:
            self._reject(404, "not found")
            return

        state = _STATE
        snapshot = None
        with state.lock:
            if _purge_expired_locked(state) and not state.previews:
                snapshot = _detach_server_locked(state)
            preview = state.previews.get(token)
        if snapshot is not None:
            _stop_server(snapshot)
        if preview is None:
            self._reject(410, "preview expired or closed")
            return
        self._preview = preview

        payload, failure = _read_preview(preview)
        if failure is not None:
            self._reject(403, failure)
            return
        with state.lock:
            active = state.previews.get(token) is preview
        if not active:
            self._reject(410, "preview expired or closed")
            return
        self._reply(
            200,
            payload or b"",
            preview.content_type,
            {"Content-Disposition": "inline"},
        )

    def do_GET(self) -> None:
        self._serve()

    def do_HEAD(self) -> None:
        self._serve()

    def do_POST(self) -> None:
        if not self._host_allowed():
            self._reject(400, "Host header must match the assigned loopback listener")
            return
        self._reject(405, "only GET and HEAD are supported")

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST
    do_OPTIONS = do_POST
    do_TRACE = do_POST
    do_CONNECT = do_POST


async def preview_local_file(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Validate one file, then serve it from a bounded loopback HTTP server."""
    ttl = args.get("ttl_seconds", DEFAULT_TTL_SECONDS)
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, (int, float))
        or not math.isfinite(ttl)
        or not MIN_TTL_SECONDS <= ttl <= MAX_TTL_SECONDS
    ):
        raise ValueError(
            f"ttl_seconds must be a number between {MIN_TTL_SECONDS:g} and "
            f"{MAX_TTL_SECONDS:g} seconds"
        )

    raw = args.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path must be a non-empty local file path")
    text = raw.strip()
    if "://" in text or text.lower().startswith("file:"):
        raise ValueError(
            "preview_local_file takes a local workspace path, not a URL; "
            "use browser_navigate for an existing http(s) URL"
        )
    path = _safe_path(text, ctx)  # symlink-resolving allowed-root containment
    if path.is_symlink() or not path.is_file():
        raise ValueError("preview path must be a regular file inside an allowed root")
    content_type = _MIME_TYPES.get(path.suffix.lower())
    if content_type is None:
        raise ValueError(f"unsupported preview type: {path.suffix or '(no extension)'}")
    try:
        info = path.stat()
    except OSError as exc:
        raise ValueError("cannot inspect preview file") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("preview path must be a regular file")
    # As in file_tools._prepare_destination, a link inside the roots may name
    # an inode outside them; containment by path says nothing about its bytes.
    if info.st_nlink > 1:
        raise ValueError("refusing to preview a hard-linked file")
    if info.st_size > MAX_FILE_BYTES:
        raise ValueError(f"preview file is too large ({info.st_size} > {MAX_FILE_BYTES} bytes)")

    token = secrets.token_urlsafe(24)
    state = _STATE
    try:
        with state.lock:
            _purge_expired_locked(state)
            server = _ensure_server_locked(state)
            port = int(server.server_address[1])
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # direct handler call without an event loop
                loop = None
            state.loop = loop
            state.events = ctx.get("events")
            preview = _Preview(
                token=token,
                path=path,
                roots=tuple(_allowed_roots(ctx)),
                actor=str(ctx.get("actor") or "anonymous"),
                call_id=str(ctx["call_id"]) if ctx.get("call_id") else None,
                port=port,
                expires_at=time.monotonic() + float(ttl),
                ttl_seconds=float(ttl),
                device=info.st_dev,
                inode=info.st_ino,
                size=info.st_size,
                content_type=content_type,
            )
            state.previews[token] = preview
    except OSError as exc:
        return {
            "success": False,
            "error": f"could not start loopback preview server: {type(exc).__name__}: {exc}",
        }

    return {
        "success": True,
        "url": f"http://{HOST}:{port}{_URL_PREFIX}{token}",
        "path": str(path),
        "content_type": content_type,
        "size_bytes": info.st_size,
        "ttl_seconds": float(ttl),
        "expires_in_seconds": float(ttl),
        "navigated": False,  # the caller still has to open the URL in the browser
        "simulated": False,  # the loopback server is real, not a dry-run stub
    }


async def preview_close(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Stop one preview by URL, or every preview owned by the calling actor."""
    url = args.get("url")
    if url is not None and (not isinstance(url, str) or not url.strip()):
        raise ValueError("url must be a non-empty preview URL when provided")

    state = _STATE
    actor = str(ctx.get("actor") or "anonymous")
    snapshot = None
    with state.lock:
        _purge_expired_locked(state)
        if url is None:
            tokens = [token for token, item in state.previews.items() if item.actor == actor]
            closed = 0
            for token in tokens:
                state.previews.pop(token, None)
                closed += 1
        else:
            parsed = _token_from_url(url.strip())
            if parsed is None:
                raise ValueError("url must be a loopback preview URL returned by preview_local_file")
            token, port = parsed
            item = state.previews.get(token)
            if item is None or item.actor != actor or (port is not None and port != item.port):
                return {"success": False, "error": "no active preview for that URL"}
            state.previews.pop(token, None)
            closed = 1
        if not state.previews:
            snapshot = _detach_server_locked(state)
    if snapshot is not None:
        await asyncio.to_thread(_stop_server, snapshot)
    return {"success": True, "closed": closed, "actor": actor}


def registered_preview(url: Any, ctx: dict[str, Any]) -> _Preview | None:
    """Return the preview a URL was actually issued for, or ``None``.

    This is the only loopback allowance the browser tools may use. The URL must
    be byte-for-byte the canonical URL from ``preview_local_file``: same host,
    same OS-assigned port, same ``/preview/<token>`` path, no query or fragment,
    an unexpired token and the calling actor's own preview. A file swap or a
    hard link added after issuance invalidates the allowance as well.
    """
    if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL_BYTES:
        return None
    candidate = url.strip()
    parsed = _token_from_url(candidate)
    if parsed is None:
        return None
    token, port = parsed
    state = _STATE
    snapshot = None
    with state.lock:
        if _purge_expired_locked(state) and not state.previews:
            snapshot = _detach_server_locked(state)
        preview = state.previews.get(token)
    if snapshot is not None:
        _stop_server(snapshot)
    if preview is None:
        return None
    if preview.actor != str(ctx.get("actor") or "anonymous"):
        return None
    if port is not None and port != preview.port:
        return None
    if candidate != canonical_preview_url(preview):
        return None
    # Reuse the same validation as an HTTP request; no content is returned.
    _payload, failure = _read_preview(preview)
    return None if failure is not None else preview


def preview_grant_for_url(url: Any, actor: str | None = None) -> dict[str, Any] | None:
    """Exact, revalidated preview capability handed to the browser operator.

    The expiry is the preview's own monotonic deadline; the validator lets the
    browser re-check liveness (and file integrity) before serving any content.
    """
    preview = registered_preview(url, {"actor": actor or "anonymous"})
    if preview is None:
        return None
    canonical = canonical_preview_url(preview)
    return {
        "url": canonical,
        "token": preview.token,
        "port": preview.port,
        "expires_at": preview.expires_at,
        "validator": lambda: registered_preview(
            canonical, {"actor": preview.actor}
        ) is not None,
    }


async def preview_open(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Open an already-issued preview URL in the actor's browser session.

    Not a registered model-callable tool: ``browser_navigate`` calls this only
    after :func:`registered_preview` has matched a live token. That keeps the
    browser's ordinary SSRF policy untouched while making the preview URL
    produced by ``preview_local_file`` genuinely navigable.
    """
    preview = registered_preview(args.get("url"), ctx)
    if preview is None:
        raise ValueError(
            "url must be the active preview URL issued by preview_local_file "
            "for this actor"
        )
    # Imported here to avoid a module-level cycle with the operator tools.
    from .operator_tools import _browser_operator

    operator = _browser_operator(ctx)
    canonical = canonical_preview_url(preview)
    expires_at = getattr(preview, "expires_at", None)

    def validator() -> bool:
        return registered_preview(canonical, ctx) is not None

    # The real operator records this grant before navigating. Custom operators
    # may only implement the original three-argument method; keep that path
    # working without weakening the real one.
    registrar = getattr(operator, "register_preview_grant", None)
    if callable(registrar):
        registrar(
            url=canonical, token=preview.token, port=preview.port,
            expires_at=expires_at, validator=validator if expires_at is not None else None,
        )
    extra: dict[str, Any] = {}
    if expires_at is not None:
        navigate = getattr(operator, "navigate_registered_preview", None)
        try:
            parameters = inspect.signature(navigate).parameters if callable(navigate) else {}
            variadic = any(
                item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
            )
            if "expires_at" in parameters or variadic:
                extra = {"expires_at": expires_at, "preview_validator": validator}
        except (TypeError, ValueError):
            pass
    final_url = await operator.navigate_registered_preview(
        canonical, token=preview.token, port=preview.port, **extra
    )
    return {
        "success": True,
        "url": str(final_url),
        # Correlation digest only. Even though this is the one-time tool
        # response, it is persisted as tool-result message content and echoed
        # by chat/API read paths; a raw token here would be replayable.
        "token_id": token_audit_id(preview.token),
        "path": str(preview.path),
        "content_type": preview.content_type,
        "navigated": True,
        "registered_token": True,
        "simulated": getattr(operator, "_backend_mode", lambda: "live")() != "live",
    }


def register(registry: Any, services: Any = None) -> list[str]:
    """Register preview tools. ``services`` is accepted but not needed."""
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="preview_local_file",
                description=(
                    "Serve one workspace-contained HTML/text/image file from a short-lived "
                    "loopback HTTP server and return the exact http://127.0.0.1 URL that "
                    "browser_navigate may open while its token is active. Static content "
                    "only; no scripts are executed."
                ),
                category="browser",
                risk=RiskLevel.MEDIUM,
                permissions=["browser", "filesystem"],
                resource_field="path",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "ttl_seconds": {
                            "type": "number",
                            "minimum": MIN_TTL_SECONDS,
                            "maximum": MAX_TTL_SECONDS,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            preview_local_file,
        ),
        (
            ToolSpec(
                name="preview_close",
                description=(
                    "Stop a loopback preview started by preview_local_file by passing its "
                    "URL, or stop all of this actor's previews when no URL is given."
                ),
                category="browser",
                risk=RiskLevel.LOW,
                permissions=["browser"],
                resource_field="",  # a preview URL is not a filesystem resource
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string", "minLength": 1, "maxLength": MAX_URL_BYTES}},
                    "additionalProperties": False,
                },
            ),
            preview_close,
        ),
    ]
    registered: list[str] = []
    for spec, handler in definitions:
        try:
            registry.register(spec, handler)
        except ValueError:  # already registered: keep the caller's version
            continue
        registered.append(spec.name)
    return registered


__all__ = [
    "register", "preview_local_file", "preview_close", "preview_open",
    "registered_preview", "canonical_preview_url", "preview_grant_for_url",
    "token_audit_id",
]
