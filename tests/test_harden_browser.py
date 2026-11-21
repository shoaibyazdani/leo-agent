"""Regression tests for BrowserOperator hardening (agent 11).

All tests use in-memory page/backends: no Playwright, no real browser and no
network. DNS is patched at the operator's resolution seam.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leo.contracts import Decision, RiskLevel
from leo.integrations import browser as browser_module
from leo.integrations.browser import (
    BrowserError,
    BrowserOperator,
    BrowserPermissionError,
    FakeBrowserPage,
    PlaywrightBrowserBackend,
)

PNG = b"\x89PNG\r\n\x1a\nfake"


class AllowAll:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def authorize(
        self, actor, action, resource=None, risk=RiskLevel.LOW, **context
    ) -> Decision:
        self.calls.append((actor, action, resource, risk, context))
        return Decision(allowed=True, reason="test allow", risk=risk)


class DenyAll:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def authorize(
        self, actor, action, resource=None, risk=RiskLevel.LOW, **context
    ) -> Decision:
        self.calls.append((actor, action, resource, risk, context))
        return Decision(allowed=False, reason="test deny", risk=risk)


class RecordingPage:
    """Fake page backend that records calls and call-time observations."""

    def __init__(
        self,
        *,
        final_url: str | None = None,
        screenshot_bytes: bytes = PNG,
        download_bytes: bytes = b"payload",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.final_url = final_url
        self.screenshot_bytes = screenshot_bytes
        self.download_bytes = download_bytes

    async def navigate(
        self, url: str, wait_until: str = "domcontentloaded", timeout: float = 10.0
    ) -> str:
        self.calls.append({"action": "navigate", "url": url, "wait_until": wait_until})
        return self.final_url if self.final_url is not None else url

    async def extract_text(self, selector: str = "body", timeout: float = 10.0) -> str:
        self.calls.append({"action": "extract_text", "selector": selector})
        return "fake text"

    async def screenshot(self, path: str | None = None, timeout: float = 10.0) -> bytes:
        self.calls.append(
            {
                "action": "screenshot",
                "path": path,
                "parent_exists": Path(path).parent.exists() if path else None,
            }
        )
        return self.screenshot_bytes

    async def click(self, selector: str, timeout: float = 10.0) -> None:
        self.calls.append({"action": "click", "selector": selector})

    async def fill(self, selector: str, value: str, timeout: float = 10.0) -> None:
        self.calls.append({"action": "fill", "selector": selector, "value": value})

    async def download(self, url: str, timeout: float = 10.0) -> bytes:
        self.calls.append({"action": "download", "url": url})
        return self.download_bytes

    async def close(self) -> None:
        self.calls.append({"action": "close"})


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test resolves a real name: every name maps to a public address."""
    monkeypatch.setattr(
        browser_module,
        "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port or 443))],
    )


def make_operator(
    tmp_path: Path,
    backend: Any,
    permissions: Any | None = None,
    **kwargs: Any,
) -> BrowserOperator:
    return BrowserOperator(
        permissions or AllowAll(),
        backend=backend,
        profile_dir=tmp_path / "profile",
        base_dir=tmp_path,
        timeout=1.0,
        **kwargs,
    )


# ------------------------------------------------------------------- path handling


async def test_relative_output_paths_are_anchored_to_base_dir(tmp_path: Path, monkeypatch) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    page = RecordingPage()
    operator = make_operator(tmp_path, page)

    shot = await operator.screenshot("shots/nested/a.png")
    assert shot == PNG
    assert (tmp_path / "shots/nested/a.png").read_bytes() == PNG
    assert not list(elsewhere.rglob("a.png"))

    target = await operator.download("https://example.org/f", "dl/f.bin")
    assert target == tmp_path / "dl/f.bin"
    assert target.read_bytes() == b"payload"


async def test_screenshot_creates_parent_dir_before_backend_writes(tmp_path: Path) -> None:
    page = RecordingPage()
    operator = make_operator(tmp_path, page)

    await operator.screenshot("deeply/nested/shot.png")

    screenshot_call = next(c for c in page.calls if c["action"] == "screenshot")
    assert screenshot_call["parent_exists"] is True
    assert screenshot_call["path"] == str(tmp_path / "deeply/nested/shot.png")


async def test_file_outputs_pass_resolved_path_as_permission_resource(tmp_path: Path) -> None:
    permissions = AllowAll()
    page = RecordingPage()
    operator = make_operator(tmp_path, page, permissions)

    await operator.screenshot("out/shot.png")
    await operator.download("https://example.org/f", "out/f.bin")

    resources = [call[2] for call in permissions.calls]
    assert resources == [str(tmp_path / "out/shot.png"), str(tmp_path / "out/f.bin")]


# ------------------------------------------------------------------- URL / DNS policy


async def test_literal_non_global_address_is_blocked(tmp_path: Path) -> None:
    page = RecordingPage()
    operator = make_operator(tmp_path, page)
    # 100.64.0.0/10 is non-global (CGNAT) but not flagged private/reserved by
    # the old hand-rolled check; it must be denied just like a DNS answer.
    with pytest.raises(BrowserPermissionError, match="private network address"):
        await operator.navigate("http://100.64.0.1/")
    assert page.calls == []
    assert await operator.navigate("https://8.8.8.8/") == "https://8.8.8.8/"


async def test_dns_answer_without_parseable_addresses_is_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        browser_module,
        "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("not-an-ip", port or 80))],
    )
    page = RecordingPage()
    operator = make_operator(tmp_path, page)

    with pytest.raises(BrowserPermissionError, match="no usable addresses"):
        await operator.navigate("https://example.org/")
    with pytest.raises(BrowserPermissionError, match="no usable addresses"):
        await operator.download("https://example.org/f", tmp_path / "f")
    assert page.calls == []


async def test_dns_failure_is_reported_as_permission_error(tmp_path: Path, monkeypatch) -> None:
    def fail(host: str, port: Any) -> Any:
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr(browser_module, "resolve_host", fail)
    page = RecordingPage()
    operator = make_operator(tmp_path, page)

    with pytest.raises(BrowserPermissionError, match="could not resolve host"):
        await operator.navigate("https://example.org/")
    assert page.calls == []


async def test_private_dns_answer_still_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        browser_module,
        "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("169.254.169.254", port or 80))],
    )
    page = RecordingPage()
    operator = make_operator(tmp_path, page)

    with pytest.raises(BrowserPermissionError, match="non-public address"):
        await operator.navigate("https://example.org/")
    assert page.calls == []


# ------------------------------------------------------------------- redirects


async def test_navigate_rejects_redirect_to_private_target(tmp_path: Path) -> None:
    page = RecordingPage(final_url="http://127.0.0.1/admin")
    operator = make_operator(tmp_path, page)

    with pytest.raises(BrowserPermissionError, match="private network address"):
        await operator.navigate("https://example.org/")
    # A denied redirect quarantines the page: no stale browser, session URL or
    # preview grant survives, and the backend is closed.
    assert [c["action"] for c in page.calls] == ["navigate", "close"]
    assert operator.session_metadata()["url"] == "http://127.0.0.1/admin"
    assert operator.session_metadata()["quarantined"] is True
    assert operator.session_metadata()["registered_preview"] is None


async def test_navigate_allows_public_redirect_target(tmp_path: Path) -> None:
    page = RecordingPage(final_url="https://www.example.org/")
    operator = make_operator(tmp_path, page)

    assert await operator.navigate("https://example.org/") == "https://www.example.org/"


async def test_allow_private_bypass_still_requires_http_scheme(tmp_path: Path) -> None:
    page = RecordingPage()
    operator = make_operator(tmp_path, page, allow_private_hosts=True)

    assert await operator.navigate("http://127.0.0.1:8080/") == "http://127.0.0.1:8080/"
    with pytest.raises(BrowserPermissionError, match="http"):
        await operator.navigate("file:///etc/passwd")


# ------------------------------------------------------------------- selectors / inputs


async def test_invalid_selectors_and_values_are_rejected_before_backend(tmp_path: Path) -> None:
    permissions = AllowAll()
    page = RecordingPage()
    operator = make_operator(tmp_path, page, permissions)

    with pytest.raises(ValueError, match="selector"):
        await operator.extract_text("  ")
    with pytest.raises(ValueError, match="selector"):
        await operator.click(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value"):
        await operator.fill("#q", 123)  # type: ignore[arg-type]

    assert page.calls == []
    assert permissions.calls == []


# ------------------------------------------------------------------- permission gating


async def test_permission_denial_prevents_all_backend_calls(tmp_path: Path) -> None:
    page = RecordingPage()
    operator = make_operator(tmp_path, page, DenyAll())

    with pytest.raises(BrowserPermissionError, match="not permitted"):
        await operator.navigate("https://example.org/")
    with pytest.raises(BrowserPermissionError, match="not permitted"):
        await operator.extract_text()
    with pytest.raises(BrowserPermissionError, match="not permitted"):
        await operator.screenshot(tmp_path / "s.png")
    with pytest.raises(BrowserPermissionError, match="not permitted"):
        await operator.click("#a")
    with pytest.raises(BrowserPermissionError, match="not permitted"):
        await operator.fill("#a", "x")
    with pytest.raises(BrowserPermissionError, match="not permitted"):
        await operator.download("https://example.org/f", tmp_path / "f")

    assert page.calls == []
    assert not (tmp_path / "s.png").exists()
    assert not (tmp_path / "f").exists()


async def test_malformed_url_is_a_browser_permission_error(tmp_path: Path) -> None:
    page = RecordingPage()
    operator = make_operator(tmp_path, page)

    for bad in ("http://[::1", "ftp://example.org/", "https://user:pw@example.org/"):
        with pytest.raises(BrowserPermissionError):
            await operator.navigate(bad)
    assert page.calls == []


# ------------------------------------------------------------------- setup / backend


async def test_bad_profile_path_raises_browser_error(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.write_text("not a directory")
    operator = BrowserOperator(
        AllowAll(),
        profile_dir=profile,
        base_dir=tmp_path,
        timeout=1.0,
    )

    with pytest.raises(BrowserError):
        await operator.navigate("https://example.org/")


class _Response:
    def __init__(self, ok: bool, status: int, body: bytes = b"payload") -> None:
        self.ok = ok
        self.status = status
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _APIRequest:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        return self.response


class _RequestContext:
    def __init__(self, response: _Response) -> None:
        self.request = _APIRequest(response)

    @property
    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return self.request.calls


async def test_playwright_download_disables_redirects_and_checks_status(tmp_path: Path) -> None:
    backend = PlaywrightBrowserBackend(tmp_path / "profile")
    context = _RequestContext(_Response(True, 200))
    backend._context = context

    assert await backend.download("https://example.org/f", timeout=1.0) == b"payload"
    url, kwargs = context.calls[0]
    assert url == "https://example.org/f"
    assert kwargs["max_redirects"] == 0
    assert kwargs["timeout"] == 1000

    failing = _RequestContext(_Response(False, 302))
    backend._context = failing
    with pytest.raises(BrowserError, match="HTTP 302"):
        await backend.download("https://example.org/redirect", timeout=1.0)
    assert failing.calls[0][1]["max_redirects"] == 0


# ------------------------------------------------------------------- honest status


async def test_fake_backend_reports_degraded_with_capability_subset(tmp_path: Path) -> None:
    recorder = make_operator(tmp_path, RecordingPage())
    record = recorder.record()
    assert record.status == "degraded"
    assert record.verification == "mock-tested"
    assert record.metadata["backend_mode"] == "fake"
    assert record.metadata["simulated"] is True
    assert record.metadata["live_verified"] is False
    assert "navigate" in record.capabilities
    assert "upload_file" not in record.capabilities  # not implemented by this backend
    assert "snapshot" not in record.capabilities
    assert "dialogs" not in record.capabilities

    fake = make_operator(tmp_path, FakeBrowserPage())
    full = fake.record()
    assert full.status == "degraded" and full.metadata["stateful"] is True
    assert {"snapshot", "upload_file", "dialogs", "download_events"} <= set(full.capabilities)
    assert await fake.navigate("https://example.org/") == "https://example.org/"
    assert fake.record().metadata["live_verified"] is False
    assert fake.record().status == "degraded"

    lazy = BrowserOperator(AllowAll(), profile_dir=tmp_path / "profile", base_dir=tmp_path)
    unlaunched = lazy.record()
    assert unlaunched.status == "blocked"
    assert unlaunched.metadata["live_verified"] is False
    assert unlaunched.metadata["launched"] is False


class _LiveLikeBackend(PlaywrightBrowserBackend):
    """Simulates a real backend's launch/navigate flags without Playwright."""

    async def navigate(self, url, wait_until="domcontentloaded", timeout=10.0):
        self.started = True
        return url


async def test_configured_requires_launch_and_live_verified_requires_navigation(tmp_path: Path) -> None:
    backend = _LiveLikeBackend(tmp_path / "profile")
    operator = make_operator(tmp_path, backend)
    assert operator.record().status == "blocked"
    assert operator.record().verification == "not-verified"
    assert operator.record().metadata["live_verified"] is False

    await operator.navigate("https://example.org/")
    record = operator.record()
    assert record.status == "configured"
    assert record.metadata["live_verified"] is True
    assert record.verification == "live-verified"

    await operator.close()
    assert operator.record().status == "blocked"
    assert operator.record().metadata["live_verified"] is False


# ------------------------------------------------------------------- native dialogs


async def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition was not reached in time")
        await asyncio.sleep(0.005)


async def test_dialog_default_policy_dismisses_and_records_pending_state(tmp_path: Path) -> None:
    page = FakeBrowserPage()
    operator = make_operator(tmp_path, page)
    await operator.navigate("https://example.org/")
    dialog = await page.emit_dialog("confirm", "Really continue?", "yes")

    assert dialog.result == {"action": "dismiss", "text": None}
    assert operator.pending_dialogs() == []
    events = operator.dialog_events()
    assert [e["status"] for e in events] == ["pending", "resolved"]
    assert events[1]["message"] == "Really continue?"
    assert events[1]["action"] == "dismiss"
    assert events[1]["url"] == "https://example.org/"


async def test_manual_dialog_accept_prompt_text_response_path(tmp_path: Path) -> None:
    page = FakeBrowserPage()
    operator = make_operator(tmp_path, page, dialog_policy="manual")
    await operator.navigate("https://example.org/")
    task = asyncio.create_task(page.emit_dialog("prompt", "Name?", "anon"))
    await _wait_until(lambda: bool(operator.pending_dialogs()))

    pending = operator.pending_dialogs()[0]
    assert pending["type"] == "prompt" and pending["default_value"] == "anon"
    assert pending["status"] == "pending"
    with pytest.raises(ValueError, match="action"):
        await operator.respond_to_dialog(pending["id"], "bogus")
    with pytest.raises(ValueError, match="text"):
        await operator.respond_to_dialog(pending["id"], "prompt")
    await operator.respond_to_dialog(pending["id"], "prompt", "Ada")
    dialog = await task

    assert dialog.result == {"action": "accept", "text": "Ada"}
    assert operator.pending_dialogs() == []
    resolved = [e for e in operator.dialog_events() if e["status"] == "resolved"][0]
    assert resolved["action"] == "accept" and resolved["text"] == "Ada"
    with pytest.raises(BrowserError, match="already resolved"):
        await operator.respond_to_dialog(pending["id"], "accept")


async def test_dialog_accept_policy_and_permission_gated_response(tmp_path: Path) -> None:
    page = FakeBrowserPage()
    operator = make_operator(tmp_path, page, dialog_policy="accept")
    dialog = await page.emit_dialog("prompt", "Name", "default")
    assert dialog.result == {"action": "accept", "text": "default"}

    operator.set_dialog_policy("manual")
    with pytest.raises(ValueError, match="dialog_policy"):
        operator.set_dialog_policy("auto")
    task = asyncio.create_task(page.emit_dialog("alert", "Notice"))
    await _wait_until(lambda: bool(operator.pending_dialogs()))
    dialog_id = operator.pending_dialogs()[0]["id"]
    operator.permissions = DenyAll()
    with pytest.raises(BrowserPermissionError, match="dialog_respond not permitted"):
        await operator.respond_to_dialog(dialog_id, "accept")
    assert operator.pending_dialogs()
    operator.permissions = AllowAll()
    await operator.respond_to_dialog(dialog_id, "dismiss")
    assert (await task).result == {"action": "dismiss", "text": None}


# ------------------------------------------------------------------- snapshot / upload / downloads


async def test_structured_snapshot_has_stable_refs_and_simulated_label(tmp_path: Path) -> None:
    page = FakeBrowserPage(pages={"https://example.org/": "base text"})
    operator = make_operator(tmp_path, page)
    await operator.navigate("https://example.org/")
    await operator.fill("#q", "needle")

    snapshot = await operator.snapshot()
    assert snapshot["title"] == "Fake page"
    assert snapshot["url"] == "https://example.org/"
    assert snapshot["simulated"] is True and snapshot["backend_mode"] == "fake"
    refs = [element["ref"] for element in snapshot["elements"]]
    assert refs[0] == "e1" and len(refs) == len(set(refs))
    field = next(element for element in snapshot["elements"] if element["name"] == "#q")
    assert field["value"] == "needle" and field["interactive"] is True
    assert "needle" in snapshot["elements"][0]["text"]
    assert "snapshot" in [call["action"] for call in page.calls]

    clipped = await operator.snapshot(max_elements=1)
    assert len(clipped["elements"]) == 1 and clipped["truncated"] is True
    assert clipped["element_count"] == len(refs)


class _EvalPage:
    async def evaluate(self, expression: str) -> dict[str, Any]:
        assert "data-leo-ref" in expression
        return {
            "title": "Live title", "url": "https://example.org/", "truncated": False,
            "elements": [{"ref": "e1", "tag": "button", "role": "button", "name": "Go",
                          "text": "Go", "value": "", "selector": '[data-leo-ref="e1"]',
                          "interactive": True}],
        }


async def test_playwright_snapshot_is_structured_and_bounded(tmp_path: Path) -> None:
    backend = PlaywrightBrowserBackend(tmp_path / "profile")
    backend._page = _EvalPage()
    snapshot = await backend.snapshot()
    assert snapshot["title"] == "Live title" and snapshot["elements"][0]["ref"] == "e1"


async def test_upload_is_workspace_confined_and_permission_gated(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    page = FakeBrowserPage()
    operator = BrowserOperator(
        AllowAll(), backend=page, profile_dir=workspace / "profile",
        base_dir=workspace, timeout=1.0,
    )
    inside = workspace / "a.txt"
    inside.write_text("data")
    result = await operator.upload_file("#file", "a.txt")
    assert result == {
        "success": True, "selector": "#file", "name": "a.txt", "size": 4,
        "path": str(inside), "simulated": True,
    }
    assert page.calls[-1]["action"] == "upload_file"

    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    with pytest.raises(BrowserPermissionError, match="inside the workspace"):
        await operator.upload_file("#file", outside)
    link = workspace / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(BrowserPermissionError, match="inside the workspace"):
        await operator.upload_file("#file", link)
    with pytest.raises(BrowserPermissionError, match="existing file"):
        await operator.upload_file("#file", workspace / "missing.txt")
    operator.permissions = DenyAll()
    with pytest.raises(BrowserPermissionError, match="upload not permitted"):
        await operator.upload_file("#file", inside)
    operator.permissions = AllowAll()


async def test_download_events_are_bounded_metadata_only(tmp_path: Path) -> None:
    page = FakeBrowserPage(downloads={"https://example.org/f": b"payload"})
    operator = make_operator(tmp_path, page)
    target = await operator.download("https://example.org/f", tmp_path / "f.bin")
    assert target.read_bytes() == b"payload"
    await page.emit_download("https://example.org/click-me", "report.csv")

    events = operator.download_events()
    assert len(events) == 2
    assert events[0]["size"] == 7
    assert events[0]["source"] == "request" and events[0]["status"] == "completed"
    assert events[1]["source"] == "page" and events[1]["filename"] == "report.csv"
    assert events[1]["size"] is None
    assert all("payload" not in str(event) for event in events)

    for index in range(60):
        await page.emit_download(f"https://example.org/{index}", "f")
    assert len(operator.download_events()) <= browser_module._MAX_EVENTS
    await page.emit_download("https://example.org/" + "a" * 3000, "f")
    assert len(operator.download_events()[-1]["url"]) < browser_module._MAX_URL + 32


# ------------------------------------------------------------------- fake state / sessions


async def test_fake_backend_models_observable_state_transitions(tmp_path: Path) -> None:
    page = FakeBrowserPage(
        pages={"https://example.org/": "base text"},
        click_effects={"#go": {"url": "https://example.org/next", "text": "next view"}},
    )
    operator = make_operator(tmp_path, page)
    await operator.navigate("https://example.org/")
    assert await operator.extract_text() == "base text"
    await operator.fill("#q", "needle")
    assert "needle" in await operator.extract_text()
    assert await operator.extract_text("#q") == "needle"
    await operator.click("#go")
    assert page.current_url == "https://example.org/next"
    assert "next view" in await operator.extract_text()
    assert "needle" in await operator.extract_text()


async def test_session_metadata_persists_and_scopes_profiles_per_actor(tmp_path: Path) -> None:
    def actor_operator(actor: str) -> BrowserOperator:
        return BrowserOperator(
            AllowAll(), actor=actor, backend=FakeBrowserPage(),
            profile_dir=tmp_path / "profiles", base_dir=tmp_path, timeout=1.0,
        )

    alice = actor_operator("alice")
    bob = actor_operator("bob")
    await alice.navigate("https://example.org/alice")
    assert alice.session_path.is_file()
    assert alice.session_metadata()["url"] == "https://example.org/alice"
    assert bob.session_metadata()["url"] is None
    assert alice.scope_id != bob.scope_id
    assert alice.session_metadata()["profile_dir"] != bob.session_metadata()["profile_dir"]
    assert "alice" not in str(bob.session_metadata())

    restored = actor_operator("alice")
    assert restored.session_metadata()["url"] == "https://example.org/alice"
    assert restored.record().metadata["session_url"] == "https://example.org/alice"
    assert restored.session_metadata()["history"][-1]["url"] == "https://example.org/alice"


# --------------------------------------------------------------- preview allowance


async def test_registered_preview_navigation_requires_exact_token_and_port(
    tmp_path: Path,
) -> None:
    """Loopback navigation is a token-bound opt-in, not a general bypass."""
    page = RecordingPage()
    operator = make_operator(tmp_path, page)
    url = "http://127.0.0.1:18765/preview/token123"

    assert await operator.navigate_registered_preview(
        url, token="token123", port=18765
    ) == url
    assert [c["url"] for c in page.calls if c["action"] == "navigate"] == [url]

    bad_calls = [
        (url, "othertoken", 18765),          # wrong token
        (url, "token123", 18766),            # wrong port
        ("http://127.0.0.1:18765/admin", "token123", 18765),  # other path
        ("http://example.org:18765/preview/token123", "token123", 18765),
        ("https://127.0.0.1:18765/preview/token123", "token123", 18765),
        ("file:///etc/passwd", "token123", 18765),
        (url + "?host=example.org", "token123", 18765),
    ]
    for candidate, token, port in bad_calls:
        with pytest.raises(BrowserPermissionError):
            await operator.navigate_registered_preview(
                candidate, token=token, port=port
            )
    assert len([c for c in page.calls if c["action"] == "navigate"]) == 1

    # Ordinary calls (including real preview-shaped URLs) keep refusing
    # loopback; only the explicit registered-token method grants the exception.
    with pytest.raises(BrowserPermissionError, match="private network address"):
        await operator.navigate(url)
    with pytest.raises(BrowserPermissionError, match="private network address"):
        await operator.navigate("http://127.0.0.1:9/anything")
    with pytest.raises(BrowserPermissionError, match="http"):
        await operator.navigate("file:///etc/passwd")
    assert len([c for c in page.calls if c["action"] == "navigate"]) == 1


# ------------------------------------------------------ preview confinement regressions


async def test_current_url_guard_covers_all_preview_content_and_interaction(
    tmp_path: Path,
) -> None:
    """A preview page that navigates itself cannot yield any page data."""
    preview = "http://127.0.0.1:18765/preview/token123"
    admin = "http://127.0.0.1:18765/admin"
    page = FakeBrowserPage(pages={preview: "PREVIEW_SECRET", admin: "ADMIN_SECRET"})
    operator = make_operator(tmp_path, page)
    assert await operator.navigate_registered_preview(
        preview, token="token123", port=18765
    ) == preview
    assert await operator.extract_text() == "PREVIEW_SECRET"

    # Simulate a link/script navigation that never went through browser.navigate.
    page.current_url = admin
    actions = [
        lambda: operator.extract_text(),
        lambda: operator.snapshot(),
        lambda: operator.screenshot(),
        lambda: operator.click("#a"),
        lambda: operator.fill("#a", "x"),
    ]
    for action in actions:
        with pytest.raises(BrowserPermissionError, match="not permitted|denied"):
            await action()

    assert "close" in [call["action"] for call in page.calls]
    metadata = operator.session_metadata()
    assert metadata["url"] == admin  # truthful: no stale preview URL
    assert metadata["quarantined"] is True
    assert metadata["registered_preview"] is None
    assert "ADMIN_SECRET" not in str(metadata)


async def test_click_navigation_is_reauthorized_audited_and_persisted(tmp_path: Path) -> None:
    preview = "http://127.0.0.1:18765/preview/token123"
    admin = "http://127.0.0.1:18765/admin"
    page = FakeBrowserPage(
        pages={preview: "PREVIEW_SECRET", admin: "ADMIN_SECRET"},
        click_effects={"#go": {"url": admin, "text": "admin view"}},
    )
    permissions = AllowAll()
    operator = make_operator(tmp_path, page, permissions)
    await operator.navigate_registered_preview(preview, token="token123", port=18765)

    with pytest.raises(BrowserPermissionError):
        await operator.click("#go")

    navigations = [call for call in permissions.calls if call[1] == "browser.navigate"]
    assert any(call[4].get("url") == admin for call in navigations)
    metadata = operator.session_metadata()
    assert metadata["url"] == admin
    assert metadata["registered_preview"] is None and metadata["quarantined"] is True
    assert "close" in [call["action"] for call in page.calls]


async def test_click_to_public_url_is_audited_and_truthfully_persisted(tmp_path: Path) -> None:
    start, end = "https://example.org/", "https://www.example.org/next"
    page = FakeBrowserPage(
        pages={start: "START", end: "END"},
        click_effects={"#go": {"url": end, "text": "next"}},
    )
    permissions = AllowAll()
    operator = make_operator(tmp_path, page, permissions)
    await operator.navigate(start)

    await operator.click("#go")

    navigations = [call for call in permissions.calls if call[1] == "browser.navigate"]
    assert len(navigations) == 2  # the explicit navigate plus the click-induced one
    assert navigations[-1][4].get("url") == end
    assert operator.session_metadata()["url"] == end
    assert operator.session_metadata()["quarantined"] is False
    assert "END" in await operator.extract_text()


async def test_expired_preview_grant_is_not_a_permanent_loopback_pass(tmp_path: Path) -> None:
    preview = "http://127.0.0.1:18765/preview/token123"
    page = FakeBrowserPage(pages={preview: "PREVIEW_SECRET"})
    operator = make_operator(tmp_path, page)
    await operator.navigate_registered_preview(
        preview, token="token123", port=18765, expires_at=time.monotonic() - 1
    )

    with pytest.raises(BrowserPermissionError):
        await operator.extract_text()
    assert operator.session_metadata()["registered_preview"] is None
    assert "PREVIEW_SECRET" not in str(operator.session_metadata())


async def test_registered_preview_url_never_leaks_the_full_token(tmp_path: Path) -> None:
    preview = "http://127.0.0.1:18765/preview/token123"
    operator = make_operator(tmp_path, FakeBrowserPage(pages={preview: "ok"}))
    await operator.navigate_registered_preview(preview, token="token123", port=18765)
    metadata = operator.session_metadata()
    grant_meta = metadata["registered_preview"]
    assert grant_meta["url"] == preview  # exact URL is still needed for confinement
    assert grant_meta["token_id"].startswith("sha256:")
    assert "token123" not in grant_meta["token_id"]
    assert operator.registered_preview() == grant_meta


# ---------------------------------------------------------------- F3 redirect regressions


async def test_allow_private_does_not_leak_into_redirect_validation(tmp_path: Path) -> None:
    page = RecordingPage(final_url="http://127.0.0.1:18765/admin")
    operator = make_operator(tmp_path, page)

    with pytest.raises(BrowserPermissionError, match="private network address"):
        await operator.navigate("https://example.org/", allow_private=True)


async def test_redirect_to_live_registered_preview_is_allowed(tmp_path: Path) -> None:
    preview = "http://127.0.0.1:18765/preview/token123"
    page = RecordingPage(final_url=preview)
    operator = make_operator(tmp_path, page)
    operator.register_preview_grant(
        url=preview, token="token123", port=18765, expires_at=time.monotonic() + 30
    )

    assert await operator.navigate("https://example.org/", allow_private=True) == preview
    metadata = operator.session_metadata()
    assert metadata["registered_preview"]["url"] == preview
    assert metadata["registered_preview"]["active"] is True


# ------------------------------------------------------- Playwright defense in depth


async def test_playwright_route_and_frame_guard_reject_non_public_urls(tmp_path: Path) -> None:
    backend = PlaywrightBrowserBackend(tmp_path / "profile")
    preview = "http://127.0.0.1:18765/preview/token123"
    admin = "http://127.0.0.1:18765/admin"
    denied: list[str] = []

    class _RouteContext:
        def __init__(self) -> None:
            self.routes: list[tuple[str, Any]] = []

        async def route(self, pattern: str, handler: Any) -> None:
            # A real Playwright context requires this coroutine to be awaited;
            # a sync call would silently install no interception at all.
            await asyncio.sleep(0)
            self.routes.append((pattern, handler))

    class _Page:
        def __init__(self) -> None:
            self.url = ""
            self.listeners: list[tuple[str, Any]] = []

        def on(self, event: str, handler: Any) -> None:
            self.listeners.append((event, handler))

    context = _RouteContext()
    page = _Page()
    backend._context = context
    backend._page = page
    backend.set_navigation_guard(
        lambda url: url == preview or url.startswith("https://"),
        on_violation=lambda url: denied.append(url),
    )
    await backend.install_navigation_guard()
    assert backend._guard_route_installed is True
    assert backend._guard_frame_installed is True
    assert [pattern for pattern, _ in context.routes] == ["**/*"]
    assert [event for event, _ in page.listeners] == ["framenavigated"]

    class Route:
        def __init__(self, url: str) -> None:
            self.request = SimpleNamespace(url=url)
            self.actions: list[str] = []

        async def continue_(self) -> None:
            self.actions.append("continue")

        async def abort(self) -> None:
            self.actions.append("abort")

    allowed = Route("https://example.org/")
    await backend._guard_route(allowed)
    assert allowed.actions == ["continue"]

    blocked = Route(admin)
    await backend._guard_route(blocked)
    assert blocked.actions == ["abort"] and denied == [admin]

    await backend._guard_frame_navigation(SimpleNamespace(url=admin))
    assert denied == [admin, admin]

    # The live backend observes where the page is, not where it was sent.
    backend._page = SimpleNamespace(url=admin)
    assert await backend.current_url() == admin


async def test_lazy_backend_installs_guard_before_first_preview_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route must exist before the first navigate() touches the network."""
    url = "http://127.0.0.1:18765/preview/token123"
    events: list[str] = []

    class _Context:
        async def route(self, pattern: str, handler: Any) -> None:
            events.append("route")
            assert pattern == "**/*"

    class _Page:
        def on(self, event: str, handler: Any) -> None:
            events.append("frame")

    class _GuardBackend(_LiveLikeBackend):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._context = _Context()
            self._page = _Page()

        async def start(self) -> None:
            events.append("start")
            self.started = True  # no real Playwright in this structural test

        def set_navigation_guard(self, guard: Any, on_violation: Any = None) -> None:
            events.append("set_guard")
            super().set_navigation_guard(guard, on_violation)

        async def install_navigation_guard(self) -> None:
            events.append("install")
            await super().install_navigation_guard()
            events.append("installed")

        async def navigate(self, url, wait_until="domcontentloaded", timeout=10.0):
            events.append("navigate")
            assert self._guard_route_installed is True
            return await super().navigate(url, wait_until, timeout)

    monkeypatch.setattr(browser_module, "PlaywrightBrowserBackend", _GuardBackend)
    operator = BrowserOperator(
        AllowAll(), profile_dir=tmp_path / "profile", base_dir=tmp_path, timeout=1.0
    )
    assert await operator.navigate_registered_preview(url, token="token123", port=18765) == url

    assert events.count("route") == 1  # installed once per backend
    assert events.index("set_guard") < events.index("route") < events.index("navigate")
    backend = operator._playwright_backend
    assert backend is not None and backend._guard_route_installed is True
    assert backend._guard_frame_installed is True


async def test_global_private_allowance_does_not_widen_preview_confinement(
    tmp_path: Path,
) -> None:
    """A configured allow_private_hosts flag is not a preview-escape pass."""
    preview = "http://127.0.0.1:18765/preview/token123"
    admin = "http://127.0.0.1:18765/admin"
    page = FakeBrowserPage(pages={preview: "PREVIEW_SECRET", admin: "ADMIN_SECRET"})
    operator = make_operator(tmp_path, page, allow_private_hosts=True)
    await operator.navigate_registered_preview(preview, token="token123", port=18765)
    page.current_url = admin
    with pytest.raises(BrowserPermissionError):
        await operator.extract_text()
    assert "ADMIN_SECRET" not in str(operator.session_metadata())

    redirect = RecordingPage(final_url=admin)
    redirect_operator = make_operator(tmp_path, redirect, allow_private_hosts=True)
    with pytest.raises(BrowserPermissionError, match="private network address"):
        await redirect_operator.navigate("https://example.org/", allow_private=True)


# ------------------------------------------------ real Chromium redirect denials


async def test_real_chromium_redirects_never_reach_the_loopback_service(
    tmp_path: Path,
) -> None:
    """301/302/307/308 and chains are blocked before the redirected GET.

    Chromium follows 3xx responses inside the network stack: ``context.route``
    only sees the original request, so a post-navigation URL check is too late
    and the internal GET has already happened. The CDP request-stage guard must
    stop every hop and leave a truthful quarantine behind.
    """
    import sys

    # Other tests simulate a machine without Playwright by clearing these
    # modules. Snapshot before importing so this test leaves no import behind.
    playwright_modules = {
        name: module for name, module in sys.modules.items()
        if name == "playwright" or name.startswith("playwright.")
    }
    pytest.importorskip("playwright.async_api")

    redirects: dict[str, tuple[int, str]] = {
        "/preview/tok301": (301, "/secret"),
        "/preview/tok302": (302, "/secret"),
        "/preview/tok307": (307, "/secret"),
        "/preview/tok308": (308, "/secret"),
        "/preview/tokchain": (302, "/hop1"),
        "/hop1": (307, "/hop2"),
        "/hop2": (308, "/secret"),
    }

    class _RedirectHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            self.server.hits.append(self.path)  # type: ignore[attr-defined]
            target = redirects.get(self.path)
            if target is not None:
                status, location = target
                self.send_response(status)
                self.send_header(
                    "Location",
                    f"http://127.0.0.1:{self.server.server_port}{location}",  # type: ignore[attr-defined]
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = b"INTERNAL_SECRET"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    server.hits = []  # type: ignore[attr-defined]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    port = int(server.server_address[1])
    try:
        for name in ("tok301", "tok302", "tok307", "tok308", "tokchain"):
            preview = f"http://127.0.0.1:{port}/preview/{name}"
            operator = BrowserOperator(
                AllowAll(), actor="test", base_dir=tmp_path,
                profile_dir=tmp_path / f"profile-{name}", timeout=10.0,
            )
            operator.register_preview_grant(
                url=preview, token=name, port=port, expires_at=time.monotonic() + 60
            )
            try:
                try:
                    with pytest.raises(BrowserPermissionError, match="redirect denied"):
                        await operator.navigate_registered_preview(
                            preview, token=name, port=port
                        )
                except BrowserError as exc:
                    if "could not start Playwright browser" not in str(exc):
                        raise
                    pytest.skip(f"real Playwright browser unavailable: {exc}")

                metadata = operator.session_metadata()
                assert metadata["quarantined"] is True
                assert metadata["registered_preview"] is None
                assert operator._session["quarantine_reason"].startswith(
                    "redirect validation denied: browser blocked"
                )
                assert "INTERNAL_SECRET" not in str(operator.record())
            finally:
                await operator.close()

        # Only the original, operator-approved preview GETs may reach the
        # listener. No redirect target - not even an intermediate hop - is
        # fetched, so the internal service never sees a request.
        assert server.hits == [f"/preview/{name}" for name in
                               ("tok301", "tok302", "tok307", "tok308", "tokchain")]
    finally:
        for name in list(sys.modules):
            if name == "playwright" or name.startswith("playwright."):
                sys.modules.pop(name, None)
        sys.modules.update(playwright_modules)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
