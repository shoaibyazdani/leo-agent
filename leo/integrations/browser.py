"""Browser operator with a persistent, actor-scoped profile and injectable backend.

Playwright is optional. Importing this module never imports Playwright; the
library is loaded lazily when a real backend is first needed, and a clear error
is raised if it is unavailable. ``FakeBrowserPage`` is a *simulated*, stateful
backend for tests: it models navigation/click/fill transitions and records
calls, but it is never reported as a live browser (``record()`` returns the
``degraded`` status and only the capabilities the injected object implements).

Browser sessions (current URL, title and bounded history) and browser profile
directories are scoped per actor and persisted under the operator's profile
root, so two actors cannot share cookies or session state.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import ipaddress
import json
import math
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlparse

from ..contracts import IntegrationRecord, PermissionManager, RiskLevel
from ..tools.builtins import is_public_ip

# Resolution seam for tests; the operator resolves each DNS name exactly
# once and validates every returned address before the browser touches it.
resolve_host = socket.getaddrinfo

# Bounds for data derived from untrusted pages and persisted session state.
_MAX_SNAPSHOT_ELEMENTS = 200
_MAX_TEXT = 240
_MAX_URL = 2048
_MAX_HISTORY = 20
_MAX_EVENTS = 40
_MAX_SESSION_BYTES = 512_000
# Fallback grant lifetime for direct operator calls that do not hand over the
# preview's real monotonic expiry. The preview tool always supplies one.
_DEFAULT_PREVIEW_GRANT_TTL = 300.0

DIALOG_POLICIES = ("dismiss", "accept", "manual")
"""``dismiss`` (safe default), ``accept`` (auto-accept) or ``manual`` (queue)."""

_LIVE_CAPABILITIES = (
    "navigate", "extract_text", "screenshot", "click", "fill", "download",
    "snapshot", "upload_file", "dialogs", "download_events",
)

# Playwright evaluates this function in the page. It assigns stable refs in
# document order and returns only bounded, structured metadata - never page
# scripts or raw HTML.
_SNAPSHOT_JS = """() => {
  const limit = 200, textLimit = 240;
  const clean = (v) => v === undefined || v === null ? "" : String(v).slice(0, textLimit);
  const interactiveTags = new Set(["a", "button", "input", "select", "textarea", "label"]);
  const interactiveRoles = new Set(["button", "link", "textbox", "checkbox", "radio",
      "combobox", "listbox", "option", "switch", "menuitem", "tab", "slider", "searchbox"]);
  const nodes = Array.from(document.querySelectorAll(
      "a,button,input,select,textarea,label,[role],[aria-label],[contenteditable=\\"true\\"]"
  )).slice(0, limit);
  const elements = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const ref = "e" + (i + 1);
    node.setAttribute("data-leo-ref", ref);
    const role = clean(node.getAttribute("role") || (node.tagName || "").toLowerCase());
    const text = clean(node.innerText || node.textContent || "");
    const value = node.tagName === "INPUT" || node.tagName === "TEXTAREA"
      ? clean(node.value || "") : "";
    elements.push({
      ref: ref, tag: clean((node.tagName || "").toLowerCase()), role: role,
      name: clean(node.getAttribute("aria-label") || node.getAttribute("name") ||
                  node.getAttribute("placeholder") || node.getAttribute("title") || text),
      text: text, value: value,
      selector: '[data-leo-ref="' + ref + '"]',
      interactive: interactiveTags.has((node.tagName || "").toLowerCase()) ||
                   interactiveRoles.has(role) || node.isContentEditable === true
    });
  }
  return {
    title: clean(document.title), url: clean(location.href), elements: elements,
    truncated: document.querySelectorAll(
      "a,button,input,select,textarea,label,[role],[aria-label],[contenteditable=\\"true\\"]"
    ).length > limit
  };
}"""


class BrowserError(RuntimeError):
    """Browser operation failed."""


class BrowserPermissionError(BrowserError):
    """Browser operation was denied by policy."""


PLAYWRIGHT_HINT = (
    "Playwright is not installed. Install it with `pip install playwright` and "
    "`playwright install chromium`, or inject a FakeBrowserPage backend."
)


def _clip(value: Any, limit: int = _MAX_TEXT) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _actor_scope_id(actor: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(actor))[:24]
    digest = hashlib.sha256(str(actor).encode("utf-8")).hexdigest()[:12]
    return f"{safe or 'actor'}-{digest}"


def _resolve_within(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


class PlaywrightBrowserBackend:
    """Thin adapter over a Playwright persistent browser context."""

    is_live_backend = True
    started = False

    def __init__(self, profile_dir: Path, *, headless: bool = True, timeout: float = 10.0) -> None:
        self.profile_dir = profile_dir
        self.headless = headless
        self.timeout = timeout
        self._playwright: Any = None
        self._context: Any = None
        self._page: Any = None
        self._dialog_handler: Any = None
        self._download_handler: Any = None
        self._navigation_guard: Any = None
        self._guard_violation: Any = None
        self._guard_route_installed = False
        self._guard_frame_installed = False
        # context.route() never observes requests Chromium follows *after* a
        # 3xx response: the redirected GET reaches the network before any
        # post-navigation check can run. A CDP Fetch handler at request stage
        # is the browser-layer choke point for those requests, including whole
        # redirect chains. Installed with the guard, before any navigation.
        self._cdp_session: Any = None
        self._cdp_interception_installed = False
        self._guard_violations: list[str] = []
        self.started = False

    async def start(self) -> None:
        try:
            from playwright.async_api import async_playwright  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BrowserError(PLAYWRIGHT_HINT) from exc
        try:
            self._playwright = await async_playwright().start()
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                timeout=int(self.timeout * 1000),
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
            self._page.set_default_timeout(int(self.timeout * 1000))
            if self._dialog_handler is not None:
                self.set_dialog_handler(self._dialog_handler)
            if self._download_handler is not None:
                self.set_download_handler(self._download_handler)
            if self._navigation_guard is not None:
                # A backend that was handed a guard before it launched still
                # needs its route registered *after* the context exists.
                await self._install_guard_hooks()
            self.started = True
        except Exception as exc:
            await self.close()
            raise BrowserError(f"could not start Playwright browser: {type(exc).__name__}: {exc}") from exc

    async def navigate(self, url: str, wait_until: str = "domcontentloaded", timeout: float = 10.0) -> str:
        if self._navigation_guard is not None and not self._cdp_interception_installed:
            # Without request-stage interception a 3xx could reach a loopback
            # service that context.route() never observes. Refuse the whole
            # navigation instead of browsing with a guard that has a hole.
            raise BrowserPermissionError(
                "navigation denied: browser-layer redirect blocking unavailable"
            )
        await self._page.goto(url, wait_until=wait_until, timeout=int(timeout * 1000))
        return self._page.url

    async def current_url(self) -> str:
        """Where the live page actually is, not where we last told it to go."""
        return str(self._page.url) if self._page is not None else ""

    async def get_current_url(self) -> str:
        return await self.current_url()

    def set_navigation_guard(self, guard: Any, on_violation: Any = None) -> None:
        """Set the guard policy. Call :meth:`install_navigation_guard` before use.

        This is defense in depth behind the operator's current-URL guard: a
        link, redirect or scripted navigation inside a preview page must not be
        able to reach a loopback service even if an outer check is raced.
        ``guard(url)`` may be sync or async and returns True to allow.

        Policy installation is deliberately separate and asynchronous:
        ``context.route()`` is a coroutine. Storing the policy synchronously
        keeps injected backends simple, while the live backend cannot silently
        drop the route because nobody awaited it.
        """
        self._navigation_guard = guard
        self._guard_violation = on_violation

    async def install_navigation_guard(self) -> None:
        """Await registration of the request guard on the live context."""
        await self._install_guard_hooks()

    async def _install_guard_hooks(self) -> None:
        if self._navigation_guard is None:
            return
        if self._context is not None and not self._guard_route_installed:
            try:
                # Must be awaited: an un-awaited context.route() installs
                # nothing, so the first clicked loopback link would still
                # perform its HTTP request before any content check ran.
                await self._context.route("**/*", self._guard_route)
                self._guard_route_installed = True
            except Exception:
                pass
        # Event registration stays synchronous; the async handler is scheduled
        # by Playwright when a frame actually navigates.
        if self._page is not None and not self._guard_frame_installed:
            try:
                self._page.on("framenavigated", self._guard_frame_navigation)
                self._guard_frame_installed = True
            except Exception:
                pass
        if (
            self._context is not None and self._page is not None
            and not self._cdp_interception_installed
        ):
            # Only Chromium exposes CDP, which is the backend this class
            # launches. Fetch.requestPaused fires before the HTTP request is
            # issued for the initial navigation, for every redirect hop and
            # for renderer-initiated requests alike.
            session: Any = None
            try:
                session = await self._context.new_cdp_session(self._page)
                await session.send(
                    "Fetch.enable",
                    {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]},
                )
                session.on("Fetch.requestPaused", self._on_fetch_request_paused)
                self._cdp_session = session
                self._cdp_interception_installed = True
            except Exception:
                if session is not None:
                    try:
                        await session.detach()
                    except Exception:
                        pass
                # navigate() fails closed while a guard is installed but the
                # browser-layer request interception is unavailable.

    async def _on_fetch_request_paused(self, event: Any) -> None:
        """Decide every live request before Chromium sends it on the wire."""
        request = event.get("request") if isinstance(event, dict) else None
        url = str((request or {}).get("url") or "")
        request_id = str(event.get("requestId") or "") if isinstance(event, dict) else ""
        session = self._cdp_session
        allowed = await self._guard_allows_fetch(url)
        if session is not None and request_id:
            try:
                if allowed:
                    await session.send("Fetch.continueRequest", {"requestId": request_id})
                else:
                    # Record the denial before failing the request: the failed
                    # goto can surface before this handler resumes after the
                    # await, and navigate() must still see why it was blocked.
                    self._record_guard_violation(url)
                    # Abort first: quarantine/audit callbacks must never delay
                    # or prevent the browser-layer rejection of the request.
                    await session.send(
                        "Fetch.failRequest",
                        {"requestId": request_id, "errorReason": "BlockedByClient"},
                    )
            except Exception:
                pass
        if not allowed:
            await self._guard_violated(url)

    async def _guard_allows_fetch(self, url: str) -> bool:
        # Non-network schemes (data:, blob:, about:) have no HTTP request to
        # block and must keep working for ordinary pages. Every http(s)
        # request, including redirect targets, goes through the live policy.
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            return True
        return await self._guard_allows(url)

    def take_guard_violations(self) -> list[str]:
        """Drain browser-layer denials observed during one operation."""
        values = list(self._guard_violations)
        self._guard_violations.clear()
        return values

    async def _guard_allows(self, url: str) -> bool:
        guard = self._navigation_guard
        if guard is None:
            return True
        try:
            value = guard(str(url))
            if inspect.isawaitable(value):
                value = await value
            return bool(value)
        except Exception:
            return False  # an unavailable policy check must fail closed

    async def _guard_route(self, route: Any) -> None:
        request = getattr(route, "request", None)
        url = str(getattr(request, "url", "") or "")
        if await self._guard_allows(url):
            proceed = getattr(route, "continue_", None) or getattr(route, "continue", None)
            if callable(proceed):
                try:
                    await proceed()
                except Exception:
                    pass  # another route handler may already have resolved it
            return
        # Abort first: quarantine/audit callbacks must never delay (or, via a
        # closed context, prevent) the browser-layer rejection of the request.
        abort = getattr(route, "abort", None)
        if callable(abort):
            try:
                await abort()
            except Exception:
                pass
        await self._guard_violated(url)

    async def _guard_frame_navigation(self, frame: Any) -> None:
        url = str(getattr(frame, "url", "") or "")
        if not await self._guard_allows(url):
            await self._guard_violated(url)

    def _record_guard_violation(self, url: str) -> None:
        if not self._guard_violations or self._guard_violations[-1] != url:
            self._guard_violations.append(url)

    async def _guard_violated(self, url: str) -> None:
        self._record_guard_violation(url)
        callback = self._guard_violation
        if callback is None:
            return
        try:
            value = callback(url)
            if inspect.isawaitable(value):
                await value
        except Exception:
            pass

    async def extract_text(self, selector: str = "body", timeout: float = 10.0) -> str:
        return await self._page.locator(selector).first.inner_text(timeout=int(timeout * 1000))

    async def snapshot(self, timeout: float = 10.0) -> dict[str, Any]:
        result = await self._page.evaluate(_SNAPSHOT_JS)
        if not isinstance(result, dict):
            raise BrowserError("browser snapshot returned an unexpected value")
        return result

    async def screenshot(self, path: str | None = None, timeout: float = 10.0) -> bytes:
        options: dict[str, Any] = {"timeout": int(timeout * 1000)}
        if path:
            options["path"] = str(path)
        return await self._page.screenshot(**options)

    async def click(self, selector: str, timeout: float = 10.0) -> None:
        await self._page.locator(selector).first.click(timeout=int(timeout * 1000))

    async def fill(self, selector: str, value: str, timeout: float = 10.0) -> None:
        await self._page.locator(selector).first.fill(value, timeout=int(timeout * 1000))

    async def upload_file(self, selector: str, path: str, timeout: float = 10.0) -> None:
        await self._page.locator(selector).first.set_input_files(path, timeout=int(timeout * 1000))

    async def download(self, url: str, timeout: float = 10.0) -> bytes:
        # Playwright's APIRequestContext follows up to 20 redirects by default.
        # A public URL could otherwise redirect into the private network after
        # the operator's one-shot URL check; refuse redirects outright and let
        # the operator validate any follow-up URL explicitly.
        response = await self._context.request.get(
            url, timeout=int(timeout * 1000), max_redirects=0
        )
        if not response.ok:
            raise BrowserError(f"download failed: HTTP {response.status}")
        return await response.body()

    def set_dialog_handler(self, handler: Any) -> None:
        self._dialog_handler = handler
        if self._page is not None:
            self._page.on("dialog", lambda dialog: asyncio.create_task(handler(dialog)))

    def set_download_handler(self, handler: Any) -> None:
        # Page-triggered downloads (e.g. a link click) are metadata-only events;
        # their bytes are never read into the agent context.
        self._download_handler = handler
        if self._page is not None:
            self._page.on("download", lambda download: asyncio.create_task(handler(download)))

    async def close(self) -> None:
        if self._cdp_session is not None:
            try:
                await self._cdp_session.detach()
            except Exception:
                pass
        for resource in (self._context, self._playwright):
            if resource is None:
                continue
            try:
                await resource.close()
            except Exception:
                pass
        self._cdp_session = None
        self._cdp_interception_installed = False
        self._context = None
        self._playwright = None
        self._page = None
        self._navigation_guard = None
        self._guard_violation = None
        self._guard_route_installed = False
        self._guard_frame_installed = False
        self.started = False


class _FakeDialog:
    """Minimal native-dialog stand-in used by the simulated backend."""

    def __init__(self, kind: str, message: str, default_value: str = "", url: str = "") -> None:
        self.type = kind
        self.message = message
        self.default_value = default_value
        self.url = url
        self.result: dict[str, Any] | None = None

    async def accept(self, prompt_text: str | None = None) -> None:
        self.result = {
            "action": "accept",
            "text": self.default_value if prompt_text is None else str(prompt_text),
        }

    async def dismiss(self) -> None:
        self.result = {"action": "dismiss", "text": None}


class _FakeDownload:
    def __init__(self, url: str, filename: str) -> None:
        self.url = url
        self.suggested_filename = filename


class FakeBrowserPage:
    """Simulated, stateful backend with call recording; no network/playwright.

    Navigation sets the current URL, ``fill`` updates field state and body text,
    and ``click`` either applies an explicit ``click_effects`` transition or
    appends a visible marker to the page text. ``extract_text`` and ``snapshot``
    therefore reflect prior actions instead of returning immutable canned text.
    """

    is_live_backend = False

    def __init__(
        self,
        *,
        pages: Mapping[str, str] | None = None,
        downloads: Mapping[str, bytes] | None = None,
        screenshot_bytes: bytes = b"\x89PNG\r\n\x1a\nfake",
        title: str = "Fake page",
        click_effects: Mapping[str, Mapping[str, Any]] | None = None,
        dialog_handler: Any | None = None,
        download_handler: Any | None = None,
    ) -> None:
        self.pages = dict(pages or {})
        self.downloads = dict(downloads or {})
        self.screenshot_bytes = screenshot_bytes
        self.title = title
        self.click_effects = {str(k): dict(v) for k, v in (click_effects or {}).items()}
        self.current_url: str | None = None
        self.fields: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []
        self.dialogs: list[_FakeDialog] = []
        self._dialog_handler = dialog_handler
        self._download_handler = download_handler

    async def navigate(self, url: str, wait_until: str = "domcontentloaded", timeout: float = 10.0) -> str:
        self.calls.append({"action": "navigate", "url": url, "wait_until": wait_until})
        self.current_url = url
        return url

    async def get_current_url(self) -> str:
        """Same observation seam as the live backend (kept as an attribute for compatibility)."""
        return self.current_url or ""

    def _body_text(self) -> str:
        base = self.pages.get(self.current_url or "", f"Fake page: {self.current_url}")
        changes = [f"[filled {sel}={value}]" for sel, value in self.fields.items()]
        changes.extend(
            str(call.get("marker")) for call in self.calls
            if call.get("action") == "click" and call.get("marker")
        )
        return base + ("\n" + "\n".join(changes) if changes else "")

    async def extract_text(self, selector: str = "body", timeout: float = 10.0) -> str:
        self.calls.append({"action": "extract_text", "selector": selector})
        if selector == "body":
            return self._body_text()
        return self.fields.get(selector, f"[{selector}]")

    async def snapshot(self, timeout: float = 10.0) -> dict[str, Any]:
        self.calls.append({"action": "snapshot"})
        elements: list[dict[str, Any]] = [{
            "ref": "e1", "tag": "body", "role": "document", "name": "",
            "text": _clip(self._body_text()), "value": "", "selector": "body",
            "interactive": False,
        }]
        for index, (selector, value) in enumerate(self.fields.items(), start=2):
            elements.append({
                "ref": f"e{index}", "tag": "input", "role": "textbox",
                "name": selector, "text": "", "value": _clip(value),
                "selector": selector, "interactive": True,
            })
        return {
            "title": self.title, "url": self.current_url or "", "elements": elements,
            "simulated": True, "truncated": False,
        }

    async def screenshot(self, path: str | None = None, timeout: float = 10.0) -> bytes:
        self.calls.append({"action": "screenshot", "path": path})
        return self.screenshot_bytes

    async def click(self, selector: str, timeout: float = 10.0) -> None:
        effect = self.click_effects.get(selector, {})
        marker = str(effect.get("text") or f"clicked:{selector}")
        self.calls.append({"action": "click", "selector": selector, "marker": marker})
        if effect.get("url"):
            self.current_url = str(effect["url"])
        for field, value in (effect.get("fields") or {}).items():
            self.fields[str(field)] = str(value)

    async def fill(self, selector: str, value: str, timeout: float = 10.0) -> None:
        self.calls.append({"action": "fill", "selector": selector, "value": value})
        self.fields[selector] = value

    async def upload_file(self, selector: str, path: str, timeout: float = 10.0) -> None:
        self.calls.append({"action": "upload_file", "selector": selector, "path": path})
        self.fields[selector] = Path(path).name

    async def download(self, url: str, timeout: float = 10.0) -> bytes:
        self.calls.append({"action": "download", "url": url})
        return self.downloads.get(url, b"fake-download")

    def set_dialog_handler(self, handler: Any) -> None:
        self._dialog_handler = handler

    def set_download_handler(self, handler: Any) -> None:
        self._download_handler = handler

    async def emit_dialog(
        self, kind: str, message: str, default_value: str = "", url: str = ""
    ) -> _FakeDialog:
        dialog = _FakeDialog(kind, message, default_value, url or self.current_url or "")
        self.dialogs.append(dialog)
        if self._dialog_handler is not None:
            # Awaiting the handler lets ``manual`` dialogs block until answered.
            await self._dialog_handler(dialog)
        return dialog

    async def emit_download(self, url: str, filename: str) -> None:
        if self._download_handler is not None:
            await self._download_handler(_FakeDownload(url, filename))

    async def close(self) -> None:
        self.calls.append({"action": "close"})


class BrowserOperator:
    """Navigate, inspect, interact with and capture pages under permission policy.

    File-system resources are checked against the permission manager's allowed
    roots. URLs are checked locally (http(s) only; private/loopback hosts are
    blocked unless explicitly allowed) and passed to the permission manager as
    context, since the manager interprets a ``resource`` as a file path.
    """

    def __init__(
        self,
        permissions: PermissionManager,
        *,
        actor: str = "browser",
        profile_dir: str | Path = ".leo/browser-profile",
        timeout: float = 10.0,
        backend: Any | None = None,
        allow_private_hosts: bool = False,
        headless: bool = True,
        base_dir: Path | None = None,
        dialog_policy: str = "dismiss",
    ) -> None:
        if dialog_policy not in DIALOG_POLICIES:
            raise ValueError(f"dialog_policy must be one of {DIALOG_POLICIES}")
        self.permissions = permissions
        self.actor = str(actor)
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.profile_dir = _resolve_within(profile_dir, self.base_dir)
        self.timeout = max(0.1, min(float(timeout), 30.0))
        self.allow_private_hosts = allow_private_hosts
        self.headless = headless
        self._backend = backend
        self._playwright_backend: PlaywrightBrowserBackend | None = None
        self._real_backend_launched = False
        self._live_navigation_succeeded = False
        self._dialog_policy = dialog_policy
        self._dialog_handler_wired = False
        self._download_handler_wired = False
        self._dialogs: dict[str, dict[str, Any]] = {}
        self._dialog_futures: dict[str, asyncio.Future[tuple[str, str | None]]] = {}
        self._dialog_events: list[dict[str, Any]] = []
        self._download_events: list[dict[str, Any]] = []
        self._event_counter = 0
        # Exact preview capability granted by the last registered-preview
        # navigation. Never persisted, never returned with page data.
        self._registered_preview: dict[str, Any] | None = None
        self._pending_preview_grant: dict[str, Any] | None = None

        # Actor isolation: cookies/profile and session metadata live in a scope
        # derived from the actor name; no two actors share them.
        self.scope_id = _actor_scope_id(self.actor)
        self.scope_dir = self.profile_dir / "scopes" / self.scope_id
        self.session_path = self.scope_dir / "session.json"
        self._session = self._load_session()
        self._session["quarantined"] = False
        if backend is not None:
            if getattr(backend, "started", False):
                self._real_backend_launched = True
            # Wire event hooks up front: native dialogs can fire before the
            # first operator call reaches the backend.
            self._wire_backend(backend)

    # ---------------------------------------------------------------- session state

    def _load_session(self) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "actor": self.actor, "url": None, "title": None, "history": [],
            "updated_at": None, "dialog_policy": self._dialog_policy,
        }
        try:
            if not self.session_path.is_file() or self.session_path.stat().st_size > _MAX_SESSION_BYTES:
                return empty
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("actor") != self.actor:
                return empty
            history = [h for h in data.get("history", []) if isinstance(h, dict)][-_MAX_HISTORY:]
            return {
                "actor": self.actor,
                "url": _clip(data["url"], _MAX_URL) if isinstance(data.get("url"), str) else None,
                "title": _clip(data["title"]) if isinstance(data.get("title"), str) else None,
                "history": history,
                "updated_at": data.get("updated_at") if isinstance(data.get("updated_at"), str) else None,
                "dialog_policy": self._dialog_policy,
            }
        except (OSError, ValueError, TypeError):
            return empty

    def _persist_session(self, *, url: str | None = None, title: str | None = None) -> None:
        if url is not None:
            self._session["url"] = _clip(url, _MAX_URL)
            history = self._session.setdefault("history", [])
            history.append({"url": self._session["url"], "at": _now()})
            self._session["history"] = history[-_MAX_HISTORY:]
        if title is not None:
            self._session["title"] = _clip(title)
        self._session["updated_at"] = _now()
        self._session["actor"] = self.actor
        try:
            self.scope_dir.mkdir(parents=True, exist_ok=True)
            temp = self.session_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(self._session, sort_keys=True), encoding="utf-8")
            os.replace(temp, self.session_path)
        except OSError:
            # A read-only workspace must not break browsing; state stays in memory.
            pass

    def session_metadata(self) -> dict[str, Any]:
        """Bounded copy of the actor-scoped, persisted session metadata."""
        return {
            "actor": self.actor,
            "scope_id": self.scope_id,
            "profile_dir": str(self.scope_dir / "profile"),
            "session_path": str(self.session_path),
            "url": self._session.get("url"),
            "title": self._session.get("title"),
            "history": list(self._session.get("history") or []),
            "updated_at": self._session.get("updated_at"),
            "dialog_policy": self._dialog_policy,
            "pending_dialogs": len(self.pending_dialogs()),
            "registered_preview": self.registered_preview(),
            "quarantined": bool(self._session.get("quarantined")),
        }

    # ---------------------------------------------------------------- backend wiring

    async def _get_backend(self) -> Any:
        if self._backend is not None:
            backend = self._backend
            if getattr(backend, "started", False):
                self._real_backend_launched = True
            self._wire_backend(backend)
            await self._install_preview_guard()
            return backend
        if self._playwright_backend is None:
            try:
                self.scope_dir.mkdir(parents=True, exist_ok=True)
                profile = self.scope_dir / "profile"
                profile.mkdir(parents=True, exist_ok=True)
                backend = PlaywrightBrowserBackend(
                    profile, headless=self.headless, timeout=self.timeout
                )
                await backend.start()
            except BrowserError:
                raise
            except Exception as exc:
                # A bad profile path must surface as a browser error, not a raw
                # OSError from mkdir (which callers cannot handle uniformly).
                raise BrowserError(
                    f"could not prepare browser profile: {type(exc).__name__}: {exc}"
                ) from exc
            self._playwright_backend = backend
            self._real_backend_launched = backend.started
        self._wire_backend(self._playwright_backend)
        # The very first (lazy) backend creation happens inside the first
        # navigate call. Install the live request guard here, before any page
        # can navigate, instead of waiting for a preview grant to be adopted.
        await self._install_preview_guard()
        return self._playwright_backend

    def _wire_backend(self, backend: Any) -> None:
        if not self._dialog_handler_wired:
            setter = getattr(backend, "set_dialog_handler", None)
            if callable(setter):
                try:
                    setter(self._handle_dialog)
                    self._dialog_handler_wired = True
                except Exception:
                    pass
        if not self._download_handler_wired:
            setter = getattr(backend, "set_download_handler", None)
            if callable(setter):
                try:
                    setter(self._handle_page_download)
                    self._download_handler_wired = True
                except Exception:
                    pass

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        backend = await self._get_backend()
        try:
            return await asyncio.wait_for(
                getattr(backend, method)(*args, **kwargs), timeout=self.timeout + 0.5
            )
        except asyncio.TimeoutError as exc:
            raise BrowserError(f"browser.{method} timed out after {self.timeout}s") from exc
        except BrowserError:
            raise
        except Exception as exc:
            raise BrowserError(f"browser.{method} failed: {type(exc).__name__}: {exc}") from exc

    async def _authorize(
        self,
        action: str,
        risk: RiskLevel,
        *,
        resource: str | None = None,
        **context: Any,
    ) -> None:
        decision = await self.permissions.authorize(self.actor, action, resource, risk, **context)
        if not decision.allowed or decision.requires_approval:
            raise BrowserPermissionError(f"{action} not permitted: {decision.reason}")

    @staticmethod
    def _parse_url(url: str) -> tuple[Any, str]:
        try:
            parsed = urlparse(str(url))
            # hostname can raise for malformed bracketed hosts even when
            # urlparse itself succeeded, so guard both operations.
            host = (parsed.hostname or "").lower().rstrip(".")
        except ValueError as exc:
            raise BrowserPermissionError(f"invalid URL: {type(exc).__name__}") from exc
        return parsed, host

    def _check_url(self, url: str, allow_private: bool = False, *, strict: bool = False) -> None:
        """Check a URL; ``strict`` ignores the operator-wide private allowance.

        Post-preview confinement and redirect validation always use ``strict``:
        a preview grant (or an old allow_private flag) must never widen the
        policy for a *different* URL.
        """
        parsed, host = self._parse_url(url)
        if parsed.scheme not in {"http", "https"} or not host:
            raise BrowserPermissionError("only absolute http(s) URLs are allowed")
        if parsed.username or parsed.password:
            raise BrowserPermissionError("URLs with embedded credentials are not allowed")
        if allow_private or (self.allow_private_hosts and not strict):
            return
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise BrowserPermissionError(f"private host blocked: {host}")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return  # a DNS name; the caller must also run _check_url_resolved
        # Use the same egress policy as the DNS path and http_get. The previous
        # hand-rolled check let addresses that are non-global but not flagged
        # private/reserved (e.g. 100.64.0.0/10 CGNAT) through as literals.
        if not is_public_ip(address):
            raise BrowserPermissionError(f"private network address blocked: {host}")

    async def _check_url_resolved(
        self, url: str, allow_private: bool = False, *, strict: bool = False
    ) -> None:
        """Lexical URL policy plus one-shot DNS validation for names.

        A DNS name must currently resolve to public addresses before the
        browser may fetch it. Previously names were accepted without any
        resolution check, so a host whose A record was already private
        passed unconditionally. The browser's own resolution still happens
        later, so a check-time-public/connect-time-private rebind remains a
        documented residual risk (see LEO_KNOWN_ISSUES.md).
        """
        self._check_url(url, allow_private, strict=strict)
        if allow_private or (self.allow_private_hosts and not strict):
            return
        _, host = self._parse_url(url)
        try:
            ipaddress.ip_address(host)
            return  # literal; already validated lexically
        except ValueError:
            pass
        try:
            infos = await asyncio.to_thread(resolve_host, host, None)
        except OSError as exc:
            raise BrowserPermissionError(
                f"could not resolve host: {host} ({type(exc).__name__})"
            ) from exc
        if not infos:
            raise BrowserPermissionError(f"could not resolve host: {host}")
        addresses: list[Any] = []
        for info in infos:
            try:
                address = ipaddress.ip_address(info[4][0])
            except (ValueError, IndexError, TypeError):
                continue
            addresses.append(address)
            if not is_public_ip(address):
                raise BrowserPermissionError(
                    f"private network address blocked: {host} resolves to non-public address {address}"
                )
        # An answer with entries but no parseable IP must not be treated as a
        # pass: that would let an unusual/malformed resolver reply bypass the
        # public-address check entirely.
        if not addresses:
            raise BrowserPermissionError(
                f"could not resolve host: {host} (no usable addresses)"
            )

    @staticmethod
    def _check_selector(selector: str) -> str:
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError("selector must be a non-empty string")
        return selector

    def _check_path(self, path: str | Path) -> Path:
        # Resolve relative output paths against the operator's base_dir, the
        # same anchor used for profile_dir; Path.resolve() alone would use the
        # process CWD and write files somewhere unrelated to the workspace.
        return _resolve_within(path, self.base_dir).resolve()

    # ------------------------------------------------------- preview page confinement

    @staticmethod
    def _same_document(first: str | None, second: str | None) -> bool:
        """Fragment-only differences are the same fetched document."""
        if not first or not second:
            return False
        try:
            a, b = urlsplit(str(first)), urlsplit(str(second))
        except ValueError:
            return False
        return (
            a.scheme == b.scheme and a.netloc.lower() == b.netloc.lower()
            and a.path == b.path and a.query == b.query
        )

    @staticmethod
    def _grant_expiry(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return time.monotonic() + _DEFAULT_PREVIEW_GRANT_TTL
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return time.monotonic() + _DEFAULT_PREVIEW_GRANT_TTL
        return number if math.isfinite(number) else time.monotonic() + _DEFAULT_PREVIEW_GRANT_TTL

    @staticmethod
    def _preview_token_id(token: str) -> str:
        """Correlate a grant in metadata without publishing the bearer token."""
        try:
            from ..tools.preview_tools import token_audit_id
            return token_audit_id(token)
        except Exception:  # pragma: no cover - only if preview tools are absent
            return "sha256:" + hashlib.sha256(str(token).encode("utf-8")).hexdigest()[:16]

    def register_preview_grant(
        self, *, url: str, token: str, port: int,
        expires_at: float | None = None, validator: Any = None,
    ) -> dict[str, Any]:
        """Hand the browser the exact preview capability, including its expiry.

        ``preview_open`` calls this before ``navigate_registered_preview`` so
        the operator can confine every later read to the live preview URL. The
        full token stays in process memory and is never written to a session,
        audit record or tool result by the browser layer.
        """
        grant = {
            "url": str(url), "token": str(token), "port": int(port),
            "expires_at": self._grant_expiry(expires_at), "validator": validator,
        }
        self._pending_preview_grant = grant
        return self._preview_grant_metadata(grant)

    @classmethod
    def _preview_grant_metadata(cls, grant: dict[str, Any]) -> dict[str, Any]:
        return {
            "url": grant["url"],
            "token_id": cls._preview_token_id(grant["token"]),
            "port": grant["port"],
            "expires_at": grant["expires_at"],
            "active": cls._grant_is_live(grant),
        }

    @staticmethod
    def _grant_is_live(grant: dict[str, Any]) -> bool:
        if time.monotonic() >= float(grant.get("expires_at", 0.0)):
            return False
        validator = grant.get("validator")
        if callable(validator):
            try:
                return bool(validator())
            except Exception:
                return False
        return True

    def registered_preview(self) -> dict[str, Any] | None:
        """Redacted view of the preview capability currently confining this page."""
        grant = self._registered_preview
        if grant is None:
            return None
        metadata = self._preview_grant_metadata(grant)
        if not metadata["active"]:
            self._clear_preview_grant()
        return metadata

    def _is_live_registered_preview(self, url: str) -> bool:
        grant = self._registered_preview
        if grant is None:
            # A handover from preview_open may be adopted by a redirect (or by
            # the preview navigation itself) once the exact URL is observed.
            pending = self._pending_preview_grant
            if (
                pending is not None
                and self._same_document(str(url), pending["url"])
                and self._grant_is_live(pending)
            ):
                self._registered_preview = pending
                self._pending_preview_grant = None
                # The live guard is installed once per backend and consults
                # _url_permitted on every request, so adopting the grant here
                # takes effect without reinstalling hooks from a sync method.
                grant = self._registered_preview
        if grant is None or not self._same_document(str(url), grant["url"]):
            return False
        if not self._grant_is_live(grant):
            self._clear_preview_grant()
            return False
        return True

    async def _try_register_preview_url(self, url: str) -> bool:
        """Adopt another live, caller-owned preview as the new confinement."""
        try:
            from ..tools.preview_tools import preview_grant_for_url
            grant = preview_grant_for_url(str(url), self.actor)
        except Exception:
            return False
        if grant is None or str(url) != grant["url"]:
            return False
        self._registered_preview = grant
        await self._install_preview_guard()
        return True

    async def _observe_current_url(self) -> str | None:
        """Ask the backend where the page is now (live and simulated backends)."""
        backend = self._playwright_backend or self._backend
        if backend is None:
            return None
        candidate = getattr(backend, "current_url", None)
        if not callable(candidate):
            alternate = getattr(backend, "get_current_url", None)
            candidate = alternate if callable(alternate) else candidate
        if candidate is None:
            return None
        try:
            value = candidate()
            if inspect.isawaitable(value):
                value = await value
        except Exception:
            return None
        return str(value) if value else None

    async def _url_permitted(self, url: str) -> bool:
        """Public under the ordinary SSRF policy, or the exact live preview."""
        if not isinstance(url, str) or not url:
            return False
        try:
            await self._check_url_resolved(url, allow_private=False, strict=True)
            return True
        except BrowserPermissionError:
            return self._is_live_registered_preview(url)

    async def _ensure_current_url_allowed(self, stage: str) -> None:
        current = await self._observe_current_url()
        if current is None:
            # A confined page whose location cannot be observed is not safe to
            # read: accept no content and end the session rather than guess.
            if self._registered_preview is not None:
                await self._quarantine_page(reason=f"{stage}: URL not observable")
                raise BrowserPermissionError(
                    f"{stage} denied: cannot verify the current browser URL"
                )
            return
        if not await self._url_permitted(current):
            await self._quarantine_page(
                reason=f"{stage}: current URL is not permitted", last_url=current
            )
            raise BrowserPermissionError(
                f"{stage} denied: current browser URL is not permitted"
            )

    async def _install_preview_guard(self) -> None:
        """Wire the dynamic URL policy and await hook installation.

        The policy is state-driven (:meth:`_url_permitted`), so one installation
        per backend covers the initial navigation, later grants and grants
        adopted from a redirect. Injection-compatible backends simply get the
        policy; the live backend additionally gets an awaited ``route()``.
        """
        backend = self._playwright_backend or self._backend
        if backend is None:
            return
        setter = getattr(backend, "set_navigation_guard", None)
        if callable(setter):
            try:
                setter(self._url_permitted, self._on_backend_guard_violation)
            except Exception:
                pass
        installer = getattr(backend, "install_navigation_guard", None)
        if callable(installer):
            try:
                value = installer()
                if inspect.isawaitable(value):
                    await value
            except Exception:
                pass

    def _clear_preview_guard(self) -> None:
        """Drop an injected backend's guard; keep the live one installed.

        The live backend's policy reads current operator state on every request,
        so an expired grant already fails closed there. Uninstalling it would
        reopen the browser-layer defence for every later navigation in the same
        session. Backend.close() tears down the live guard when appropriate.
        """
        backend = self._playwright_backend or self._backend
        if backend is None or getattr(backend, "is_live_backend", False):
            return
        setter = getattr(backend, "set_navigation_guard", None)
        if callable(setter):
            try:
                setter(None)
            except Exception:
                pass

    async def _on_backend_guard_violation(self, url: str) -> None:
        if self._registered_preview is None:
            return
        await self._quarantine_page(reason="backend blocked navigation", last_url=str(url))

    def _clear_preview_grant(self) -> None:
        self._registered_preview = None
        self._pending_preview_grant = None
        self._clear_preview_guard()

    async def _quarantine_page(self, *, reason: str, last_url: str | None = None) -> None:
        """Leave the page for good: no further content can be obtained from it."""
        if last_url:
            self._session["url"] = _clip(last_url, _MAX_URL)
            history = self._session.setdefault("history", [])
            history.append({"url": self._session["url"], "at": _now()})
            self._session["history"] = history[-_MAX_HISTORY:]
        self._registered_preview = None
        self._pending_preview_grant = None
        self._session["quarantined"] = True
        self._session["quarantine_reason"] = _clip(reason)
        self._persist_session()
        self._clear_preview_guard()
        backend = self._playwright_backend or self._backend
        if backend is not None:
            try:
                await asyncio.wait_for(backend.close(), timeout=self.timeout + 0.5)
            except Exception:
                pass
            if self._playwright_backend is backend:
                # A fresh backend can be launched later; the quarantined page
                # and its context are never reused.
                self._playwright_backend = None
        self._real_backend_launched = False
        self._live_navigation_succeeded = False

    async def _check_post_action_navigation(self, before: str | None, after: str | None) -> None:
        """A click/fill that changes the URL is a new, policy-checked navigation."""
        if after is None or after == before:
            return
        try:
            # PermissionManagerImpl audits authorize() calls, so this appears in
            # the durable trail as browser.navigate even though the model asked
            # for a click/fill.
            await self._authorize(
                "browser.navigate", RiskLevel.LOW, url=after, via="browser.action"
            )
            if not self._same_document(before, after):
                if self._is_live_registered_preview(after):
                    pass
                elif not await self._try_register_preview_url(after):
                    await self._check_url_resolved(after, allow_private=False, strict=True)
            if self._registered_preview is not None and not self._is_live_registered_preview(after):
                self._clear_preview_grant()
            self._session["quarantined"] = False
            self._persist_session(url=after)
        except BrowserPermissionError as exc:
            await self._quarantine_page(reason="post-action navigation denied", last_url=after)
            raise BrowserPermissionError(
                f"navigation caused by browser action denied: {exc}"
            ) from exc

    # ---------------------------------------------------------------- navigation / DOM

    async def navigate(self, url: str, *, wait_until: str = "domcontentloaded", allow_private: bool = False) -> str:
        await self._check_url_resolved(url, allow_private)
        await self._authorize("browser.navigate", RiskLevel.LOW, url=str(url))
        # A lazy live backend does not exist until _call() below launches it,
        # so stale denials are drained from whichever backend is present now
        # and the current backend is resolved again after navigation.
        pending_backend = self._playwright_backend or self._backend
        stale = getattr(pending_backend, "take_guard_violations", None)
        if callable(stale):
            stale()
        failure: Exception | None = None
        try:
            final_url = str(
                await self._call("navigate", str(url), wait_until=wait_until, timeout=self.timeout)
            )
        except Exception as exc:
            # A blocked redirect makes Chromium fail the navigation; that is
            # still a policy denial, not a generic browser failure.
            failure = exc
            final_url = str(url)
        backend = self._playwright_backend or self._backend
        drain = getattr(backend, "take_guard_violations", None)
        violations = list(drain()) if callable(drain) else []
        if violations:
            blocked = _clip(violations[-1], _MAX_URL)
            await self._quarantine_page(
                reason=f"redirect validation denied: browser blocked {blocked}",
                last_url=blocked,
            )
            raise BrowserPermissionError(
                f"redirect denied: browser blocked a non-public request before fetch: {blocked}"
            ) from failure
        if failure is not None:
            raise failure
        # A per-navigation private allowance applies to the initial target only.
        # Redirects must satisfy the ordinary public policy, except when they
        # land on this actor's own live registered preview URL (or on another
        # live preview handed over by the preview registry). Fragment-only
        # changes are the same document, not a redirect.
        try:
            if final_url != str(url) and not self._same_document(str(url), final_url):
                if not self._is_live_registered_preview(final_url):
                    if not await self._try_register_preview_url(final_url):
                        await self._check_url_resolved(final_url, allow_private=False, strict=True)
            if self._registered_preview is not None and not self._is_live_registered_preview(final_url):
                self._clear_preview_grant()
        except BrowserPermissionError as exc:
            # The browser has already followed the redirect. Refuse every later
            # read, drop the capability and persist where the page really is;
            # the awaited request guard (item 1) prevents the loopback GET from
            # reaching the target in the first place.
            await self._quarantine_page(
                reason="redirect validation denied", last_url=final_url
            )
            raise BrowserPermissionError(f"redirect denied: {exc}") from exc
        self._session["quarantined"] = False
        backend = self._playwright_backend or self._backend
        if self._backend_mode() == "live" and (
            self._real_backend_launched or bool(getattr(backend, "started", False))
        ):
            self._real_backend_launched = True
            self._live_navigation_succeeded = True
        self._persist_session(url=final_url)
        return final_url

    async def navigate_registered_preview(
        self, url: str, *, token: str, port: int,
        expires_at: float | None = None, preview_validator: Any = None,
    ) -> str:
        """Open a loopback preview under an explicit, token-bound allowance.

        This is deliberately not a general loopback bypass: the caller must
        supply the capability token and port that ``preview_local_file`` issued,
        and the URL must be the exact ``/preview/<token>`` form on 127.0.0.1
        (or localhost). Every other URL still goes through :meth:`navigate`,
        which refuses private hosts. No filesystem path is accepted here.
        """
        allowed_chars = frozenset(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        )
        if (
            not isinstance(token, str) or not token or len(token) > 64
            or any(char not in allowed_chars for char in token)
        ):
            raise BrowserPermissionError("preview navigation requires a valid preview token")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise BrowserPermissionError("preview navigation requires a valid preview port")
        if not isinstance(url, str) or not url or len(url) > _MAX_URL:
            raise BrowserPermissionError("invalid preview URL")
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower().rstrip(".")
            url_port = parsed.port
        except ValueError as exc:
            raise BrowserPermissionError(f"invalid preview URL: {type(exc).__name__}") from exc
        if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost"}:
            raise BrowserPermissionError("preview navigation requires the issued loopback http URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise BrowserPermissionError("preview URL must match the issued URL exactly")
        if url_port != port or parsed.path != f"/preview/{token}":
            raise BrowserPermissionError("preview URL token or port does not match the registered preview")
        # The token/port pair was validated by preview_tools.registered_preview
        # before this call; only here does the browser grant a private-host
        # allowance, and it is scoped to this one navigation. Record the exact
        # URL, token, port and expiry so later reads can prove they are still
        # looking at that preview rather than a page it linked to.
        grant = self._pending_preview_grant
        self._pending_preview_grant = None
        if grant is None or (
            grant["url"] != url or grant["token"] != token or grant["port"] != port
        ):
            grant = {
                "url": str(url), "token": str(token), "port": int(port),
                "expires_at": self._grant_expiry(expires_at),
                "validator": preview_validator,
            }
        elif expires_at is not None or preview_validator is not None:
            grant = dict(grant)
            if expires_at is not None:
                grant["expires_at"] = self._grant_expiry(expires_at)
            if preview_validator is not None:
                grant["validator"] = preview_validator
        self._registered_preview = grant
        # Re-assert (and await) the guard when a grant is registered/late-started:
        # an injected or externally started backend may have appeared since the
        # previous navigation.
        await self._install_preview_guard()
        try:
            final_url = await self.navigate(url, allow_private=True)
        except Exception:
            self._clear_preview_grant()
            raise
        if not self._is_live_registered_preview(final_url):
            self._clear_preview_grant()
        return final_url

    async def extract_text(self, selector: str = "body") -> str:
        selector = self._check_selector(selector)
        await self._authorize("browser.extract_text", RiskLevel.LOW, selector=selector)
        await self._ensure_current_url_allowed("browser.extract_text")
        text = str(await self._call("extract_text", selector, timeout=self.timeout))
        # Re-check before handing any page bytes to the caller: a navigation
        # during the backend call must not turn a preview read into an SSRF.
        await self._ensure_current_url_allowed("browser.extract_text")
        return text

    async def snapshot(self, *, max_elements: int | None = None) -> dict[str, Any]:
        """Structured title/URL/element snapshot with stable ``eN`` refs.

        Refs are document-order and map to ``[data-leo-ref="eN"]`` selectors on
        a live page. Data from the page is bounded; simulated backends are
        explicitly labelled so their text is never mistaken for a live flow.
        """
        await self._authorize("browser.snapshot", RiskLevel.LOW)
        await self._ensure_current_url_allowed("browser.snapshot")
        try:
            raw = await self._call("snapshot", timeout=self.timeout)
        except BrowserError:
            # Backends without DOM support still yield a truthful text-only
            # snapshot, explicitly flagged as simulated metadata.
            text = await self.extract_text("body")
            raw = {
                "title": self._session.get("title") or "", "url": self._session.get("url") or "",
                "elements": [{"ref": "e1", "tag": "body", "role": "document", "name": "",
                              "text": text, "value": "", "selector": "body",
                              "interactive": False}],
                "simulated": True, "note": "backend has no DOM snapshot; text only",
            }
        limit = _MAX_SNAPSHOT_ELEMENTS if max_elements is None else max(1, min(int(max_elements), _MAX_SNAPSHOT_ELEMENTS))
        elements: list[dict[str, Any]] = []
        raw_elements = raw.get("elements") if isinstance(raw, Mapping) else None
        for index, item in enumerate(raw_elements if isinstance(raw_elements, list) else [], start=1):
            if not isinstance(item, Mapping):
                continue
            ref = _clip(item.get("ref") or f"e{index}", 32)
            elements.append({
                "ref": ref,
                "tag": _clip(item.get("tag") or "", 32),
                "role": _clip(item.get("role") or "", 64),
                "name": _clip(item.get("name") or ""),
                "text": _clip(item.get("text") or ""),
                "value": _clip(item.get("value") or ""),
                "selector": _clip(item.get("selector") or f'[data-leo-ref="{ref}"]', 256),
                "interactive": bool(item.get("interactive")),
            })
        truncated = bool(isinstance(raw, Mapping) and raw.get("truncated")) or len(elements) > limit
        result = {
            "title": _clip(raw.get("title") if isinstance(raw, Mapping) else ""),
            "url": _clip(raw.get("url") if isinstance(raw, Mapping) else "", _MAX_URL),
            "elements": elements[:limit],
            "element_count": len(elements),
            "truncated": truncated,
            "backend_mode": self._backend_mode(),
            "simulated": bool(isinstance(raw, Mapping) and raw.get("simulated", self._backend_mode() != "live")),
            "note": _clip(raw.get("note") if isinstance(raw, Mapping) else ""),
        }
        await self._ensure_current_url_allowed("browser.snapshot")
        self._persist_session(title=result["title"])
        return result

    async def screenshot(self, path: str | Path | None = None) -> bytes:
        target = self._check_path(path) if path is not None else None
        await self._authorize(
            "browser.screenshot", RiskLevel.MEDIUM, resource=str(target) if target else None
        )
        await self._ensure_current_url_allowed("browser.screenshot")
        if target is not None:
            # The Playwright backend writes directly to this path, so its
            # parent directory must exist before the backend is called.
            target.parent.mkdir(parents=True, exist_ok=True)
        data = await self._call("screenshot", str(target) if target else None, timeout=self.timeout)
        await self._ensure_current_url_allowed("browser.screenshot")
        raw = bytes(data)
        if target is not None:
            target.write_bytes(raw)
        return raw

    async def click(self, selector: str) -> None:
        selector = self._check_selector(selector)
        await self._authorize("browser.click", RiskLevel.MEDIUM, selector=selector)
        await self._ensure_current_url_allowed("browser.click")
        before = await self._observe_current_url()
        await self._call("click", selector, timeout=self.timeout)
        after = await self._observe_current_url()
        await self._check_post_action_navigation(before, after)
        await self._ensure_current_url_allowed("browser.click")

    async def fill(self, selector: str, value: str) -> None:
        selector = self._check_selector(selector)
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        await self._authorize("browser.fill", RiskLevel.MEDIUM, selector=selector)
        await self._ensure_current_url_allowed("browser.fill")
        before = await self._observe_current_url()
        await self._call("fill", selector, value, timeout=self.timeout)
        after = await self._observe_current_url()
        await self._check_post_action_navigation(before, after)
        await self._ensure_current_url_allowed("browser.fill")

    async def upload_file(self, selector: str, path: str | Path) -> dict[str, Any]:
        """Attach a file that already exists inside this operator's workspace.

        Absolute paths are confined too: ``resolve()`` follows symlinks, so a
        symlink pointing outside the workspace cannot smuggle a host file into
        the browser.
        """
        selector = self._check_selector(selector)
        target = self._check_path(path)
        try:
            target.relative_to(self.base_dir)
        except ValueError as exc:
            raise BrowserPermissionError("upload path must stay inside the workspace") from exc
        if not target.is_file():
            raise BrowserPermissionError("upload path must be an existing file")
        await self._authorize(
            "browser.upload", RiskLevel.MEDIUM, resource=str(target), selector=selector
        )
        await self._call("upload_file", selector, str(target), timeout=self.timeout)
        return {
            "success": True, "selector": selector, "name": _clip(target.name, 256),
            "size": target.stat().st_size, "path": str(target),
            "simulated": self._backend_mode() != "live",
        }

    # ---------------------------------------------------------------- native dialogs

    def set_dialog_policy(self, policy: str) -> None:
        if policy not in DIALOG_POLICIES:
            raise ValueError(f"dialog_policy must be one of {DIALOG_POLICIES}")
        self._dialog_policy = policy
        self._session["dialog_policy"] = policy

    async def _handle_dialog(self, dialog: Any) -> None:
        """Capture a native dialog, then apply policy or await an explicit reply."""
        self._event_counter += 1
        dialog_id = f"dlg-{self._event_counter}"
        entry: dict[str, Any] = {
            "id": dialog_id,
            "type": _clip(getattr(dialog, "type", "") or "alert", 32),
            "message": _clip(getattr(dialog, "message", "") or ""),
            "default_value": _clip(getattr(dialog, "default_value", "") or ""),
            "url": _clip(getattr(dialog, "url", "") or self._session.get("url") or "", _MAX_URL),
            "status": "pending", "action": None, "text": None,
            "policy": self._dialog_policy, "timed_out": False,
            "created_at": _now(),
        }
        self._dialogs[dialog_id] = entry
        self._record_event(self._dialog_events, dict(entry))
        if self._dialog_policy != "manual":
            action = "accept" if self._dialog_policy == "accept" else "dismiss"
            await self._finish_dialog(dialog, entry, action, None)
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, str | None]] = loop.create_future()
        self._dialog_futures[dialog_id] = future
        try:
            action, text = await asyncio.wait_for(future, timeout=self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            entry["timed_out"] = True
            action, text = "dismiss", None
        finally:
            self._dialog_futures.pop(dialog_id, None)
        await self._finish_dialog(dialog, entry, action, text)

    async def _finish_dialog(
        self, dialog: Any, entry: dict[str, Any], action: str, text: str | None
    ) -> None:
        try:
            if action == "accept" or action == "prompt":
                await dialog.accept(text)
            else:
                await dialog.dismiss()
        except Exception as exc:
            entry["error"] = _clip(f"{type(exc).__name__}: {exc}")
        entry.update({
            "status": "resolved", "action": "accept" if action == "prompt" else action,
            "text": None if text is None else _clip(text), "resolved_at": _now(),
        })
        self._record_event(self._dialog_events, dict(entry))

    @staticmethod
    def _record_event(bucket: list[dict[str, Any]], event: dict[str, Any]) -> None:
        bucket.append(event)
        del bucket[:-_MAX_EVENTS]

    def pending_dialogs(self) -> list[dict[str, Any]]:
        return [dict(d) for d in self._dialogs.values() if d.get("status") == "pending"]

    def dialog_events(self) -> list[dict[str, Any]]:
        """Bounded dialogue history; messages and prompt text are clipped."""
        return [dict(d) for d in self._dialog_events]

    async def respond_to_dialog(
        self, dialog_id: str, action: str = "accept", text: str | None = None
    ) -> dict[str, Any]:
        """Resolve a queued dialog with accept, dismiss or prompt text.

        ``action="prompt"`` is an alias for accept-with-text. The response is
        permission-gated like any other browser write action.
        """
        if action not in {"accept", "dismiss", "prompt"}:
            raise ValueError("action must be accept, dismiss or prompt")
        if action == "prompt" and not isinstance(text, str):
            raise ValueError("prompt responses require text")
        entry = self._dialogs.get(str(dialog_id))
        if entry is None:
            raise BrowserError(f"unknown dialog: {dialog_id}")
        if entry.get("status") != "pending":
            raise BrowserError(f"dialog already resolved: {dialog_id}")
        await self._authorize(
            "browser.dialog_respond", RiskLevel.MEDIUM, resource=str(dialog_id),
            dialog_type=entry.get("type"), response=action,
        )
        future = self._dialog_futures.get(str(dialog_id))
        if future is None or future.done():
            raise BrowserError(f"dialog is not awaiting a response: {dialog_id}")
        future.set_result((action, text))
        return dict(entry)

    # ---------------------------------------------------------------- downloads

    async def _handle_page_download(self, download: Any) -> None:
        self._record_download_event(
            url=getattr(download, "url", "") or "",
            filename=getattr(download, "suggested_filename", "") or "",
            size=None, source="page", status="started",
        )

    def _record_download_event(
        self, *, url: str, filename: str = "", size: int | None = None,
        source: str = "request", status: str = "completed", path: str | None = None,
    ) -> dict[str, Any]:
        self._event_counter += 1
        event: dict[str, Any] = {
            "id": f"dl-{self._event_counter}",
            "url": _clip(url, _MAX_URL),
            "filename": _clip(filename, 256),
            "path": None if path is None else _clip(path, 512),
            "size": size if isinstance(size, int) and size >= 0 else None,
            "source": source, "status": status, "at": _now(),
        }
        self._record_event(self._download_events, event)
        return event

    def download_events(self) -> list[dict[str, Any]]:
        """Bounded metadata only: no downloaded bytes are retained."""
        return [dict(event) for event in self._download_events]

    async def download(self, url: str, path: str | Path, *, allow_private: bool = False) -> Path:
        await self._check_url_resolved(url, allow_private)
        target = self._check_path(path)
        await self._authorize("browser.download", RiskLevel.MEDIUM, resource=str(target), url=str(url))
        data = bytes(await self._call("download", str(url), timeout=self.timeout))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self._record_download_event(
            url=str(url), filename=target.name, size=len(data), source="request", path=str(target)
        )
        return target

    async def close(self) -> None:
        self._clear_preview_grant()
        if self._backend is not None:
            await self._backend.close()
        if self._playwright_backend is not None:
            await self._playwright_backend.close()
            self._playwright_backend = None
        self._real_backend_launched = False
        for future in self._dialog_futures.values():
            if not future.done():
                future.cancel()
        self._dialog_futures.clear()

    # ---------------------------------------------------------------- honest status

    def _backend_mode(self) -> str:
        backend = self._playwright_backend or self._backend
        if backend is None:
            return "unavailable"
        if getattr(backend, "is_live_backend", False):
            return "live"
        return "fake"

    def record(self) -> IntegrationRecord:
        """Report only capabilities the current backend can actually perform.

        A fake/injected backend is ``degraded`` (never ``configured``), a real
        backend is ``configured`` only after it has launched, and
        ``live_verified`` requires a successful navigation with that real
        backend. Capabilities reflect methods the injected object implements.
        """
        playwright_available = importlib.util.find_spec("playwright") is not None
        mode = self._backend_mode()
        backend = self._playwright_backend or self._backend
        backend_name = type(backend).__name__ if backend is not None else (
            "playwright" if playwright_available else "unavailable"
        )
        launched = mode == "live" and (self._real_backend_launched or bool(getattr(backend, "started", False)))
        live_verified = bool(launched and self._live_navigation_succeeded)
        if mode == "fake":
            capability_methods = {
                "navigate": "navigate", "extract_text": "extract_text",
                "screenshot": "screenshot", "click": "click", "fill": "fill",
                "download": "download", "snapshot": "snapshot",
                "upload_file": "upload_file",
            }
            capabilities = [
                name for name, method in capability_methods.items()
                if callable(getattr(backend, method, None))
            ]
            if callable(getattr(backend, "set_dialog_handler", None)):
                capabilities.append("dialogs")
            if callable(getattr(backend, "download", None)):
                capabilities.append("download_events")
            status = "degraded"
            verification = "mock-tested"
            reason = "injected backend is simulated/mock; no live browser verified"
        else:
            capabilities = list(_LIVE_CAPABILITIES)
            status = "configured" if launched else "blocked"
            verification = "live-verified" if live_verified else "not-verified"
            if launched:
                reason = None
            elif playwright_available:
                reason = "Playwright is available but no real browser session has launched"
            else:
                reason = PLAYWRIGHT_HINT
        return IntegrationRecord(
            name="browser",
            kind="operator",
            status=status,
            capabilities=capabilities,
            verification=verification,
            config_ref=str(self.profile_dir),
            metadata={
                "backend": backend_name,
                "backend_mode": mode,
                "playwright_available": playwright_available,
                "profile_dir": str(self.profile_dir),
                "scope_id": self.scope_id,
                "scope_profile_dir": str(self.scope_dir / "profile"),
                "session_path": str(self.session_path),
                "session_url": self._session.get("url"),
                "allow_private_hosts": self.allow_private_hosts,
                "live_verified": live_verified,
                "simulated": mode == "fake",
                "stateful": mode == "fake" and hasattr(backend, "_body_text"),
                "launched": launched,
                "dialog_policy": self._dialog_policy,
                "pending_dialogs": len(self.pending_dialogs()),
                "download_events": len(self._download_events),
                "registered_preview": self.registered_preview(),
                "quarantined": bool(self._session.get("quarantined")),
                "reason": reason,
            },
        )


__all__ = [
    "BrowserError", "BrowserPermissionError", "BrowserOperator", "DIALOG_POLICIES",
    "FakeBrowserPage", "PLAYWRIGHT_HINT", "PlaywrightBrowserBackend",
]
