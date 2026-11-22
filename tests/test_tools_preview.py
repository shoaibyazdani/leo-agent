"""Loopback preview tools, exercised through the real ToolExecutorImpl.

No shell, no OS browser and no Playwright are used: the HTTP side is a real
loopback server, and the browser acceptance check uses BrowserOperator with an
in-process backend stub.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.interfaces import api as api_module
from leo.integrations import browser as browser_module
from leo.integrations.browser import (
    BrowserError,
    BrowserOperator,
    BrowserPermissionError,
    PlaywrightBrowserBackend,
)
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, operator_tools, preview_tools
from leo.tools.file_tools import MAX_FILE_BYTES

TOOL_NAMES = ["preview_local_file", "preview_close"]
URL_RE = re.compile(r"^http://127\.0\.0\.1:(\d+)/preview/[A-Za-z0-9_-]+$")


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = preview_tools.register(registry)
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    yield SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events,
        permissions=permissions, registry=registry, executor=executor,
        names=names,
    )
    # The server/table are process-local: leave no preview behind for other
    # tests, using only the public close tool.
    for actor in ("test", "other"):
        await executor.execute(ToolCall(name="preview_close", actor=actor))


async def _run(env, name: str, arguments: dict | None = None, **kwargs):
    call = ToolCall(name=name, arguments=arguments or {}, actor=kwargs.pop("actor", "test"), **kwargs)
    return await env.executor.execute(call)


def _fetch(url: str, method: str = "GET"):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, method=method)
    try:
        with opener.open(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()
    except urllib.error.URLError:
        return None, {}, b""


def _fetch_host(url: str, host: str, method: str = "GET"):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Host": host}, method=method)
    try:
        with opener.open(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError:
        return None, b""


class _Backend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def navigate(self, url: str, wait_until: str = "domcontentloaded", timeout: float = 10.0) -> str:
        self.calls.append({"action": "navigate", "url": url})
        return url


def _operator(env, *, allow_private: bool) -> tuple[BrowserOperator, _Backend]:
    backend = _Backend()
    operator = BrowserOperator(
        env.permissions,
        backend=backend,
        actor="test",
        base_dir=env.tmp_path,
        profile_dir=env.tmp_path / "profile",
        timeout=1.0,
        allow_private_hosts=allow_private,
    )
    return operator, backend


async def test_registration_specs_and_validation(env) -> None:
    assert env.names == TOOL_NAMES
    assert preview_tools.register(env.registry, services={}) == []  # idempotent
    specs = {spec.name: spec for spec in env.registry.list_specs()}
    assert set(specs) == set(TOOL_NAMES)

    preview = specs["preview_local_file"]
    assert preview.category == "browser"
    assert preview.risk is RiskLevel.MEDIUM
    assert preview.permissions == ["browser", "filesystem"]
    assert preview.resource_field == "path"
    assert preview.resource_field in preview.input_schema["properties"]
    assert preview.input_schema["required"] == ["path"]
    assert preview.input_schema["additionalProperties"] is False
    assert preview.input_schema["properties"]["ttl_seconds"]["minimum"] >= 1
    assert preview.input_schema["properties"]["ttl_seconds"]["maximum"] <= 600
    assert env.registry.get("preview_local_file").handler is not None

    close = specs["preview_close"]
    assert close.category == "browser"
    assert close.risk is RiskLevel.LOW
    assert close.permissions == ["browser"]
    assert close.resource_field == ""  # URL token, not a filesystem path
    assert close.input_schema["additionalProperties"] is False

    assert env.registry.validate("preview_local_file", {})
    assert any("path" in error for error in env.registry.validate("preview_local_file", {}))
    assert any("ttl_seconds" in error for error in env.registry.validate(
        "preview_local_file", {"path": "x", "ttl_seconds": 9999}
    ))
    assert env.registry.validate("preview_local_file", {"path": "x", "url": "y"})
    assert env.registry.validate("preview_close", {"token": "abc"})


async def test_capability_grants_medium_risk_and_audit_resource(env) -> None:
    target = env.tmp_path / "page.html"
    target.write_text("<h1>granted</h1>")

    env.settings.security.grants = ["browser"]
    denied = await _run(env, "preview_local_file", {"path": str(target)})
    assert not denied.success and denied.decision == "permission_denied"
    assert denied.metadata["missing_capabilities"] == ["filesystem"]

    env.settings.security.grants = ["browser", "filesystem"]
    allowed = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 2})
    assert allowed.success, allowed.error
    assert allowed.risk is RiskLevel.MEDIUM
    assert allowed.decision == "allowed"
    assert allowed.output["path"] == str(target.resolve())

    rows = await db.fetchall(
        "SELECT action,resource,risk,decision FROM audit_log WHERE resource=?", (str(target.resolve()),)
    )
    assert any(
        row["action"] == "tool:preview_local_file"
        and row["risk"] == "medium" and row["decision"] == "allowed"
        for row in rows
    )
    assert any(row["action"] == "tool.execute" for row in rows)
    calls = await db.fetchall("SELECT decision,risk FROM tool_calls WHERE tool='preview_local_file'")
    assert calls[-1]["decision"] == "allowed" and calls[-1]["risk"] == "medium"
    assert await _run(env, "preview_close", {"url": allowed.output["url"]})


async def test_valid_html_gets_navigable_http_url_without_shell_or_file_scheme(env) -> None:
    target = env.tmp_path / "how-i-work.html"
    html = "<!doctype html><html><body><h1>hello preview</h1></body></html>"
    target.write_text(html)

    result = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 30})
    assert result.success, result.error
    url = result.output["url"]
    match = URL_RE.match(url)
    assert match and int(match.group(1)) > 1024  # OS-assigned ephemeral port
    assert result.output["content_type"] == "text/html; charset=utf-8"
    assert result.output["size_bytes"] == len(html.encode())
    assert result.output["navigated"] is False
    assert result.output["simulated"] is False
    assert html not in json.dumps(result.output)  # metadata only, never content
    assert "file:" not in json.dumps(result.output)
    assert preview_tools._STATE.server is not None
    assert preview_tools._STATE.server.server_address[0] == "127.0.0.1"

    status, headers, body = _fetch(url)
    assert status == 200 and body == html.encode()
    assert headers.get("Content-Type", "").startswith("text/html")
    assert "sandbox" in headers.get("Content-Security-Policy", "")
    assert "script-src 'none'" in headers.get("Content-Security-Policy", "")
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("Cache-Control") == "no-store"

    assert _fetch(url, method="HEAD")[0] == 200
    assert _fetch(url, method="HEAD")[2] == b""
    assert _fetch(url, method="POST")[0] == 405
    assert _fetch(url + "/secret")[0] == 404  # no directory listing/traversal
    assert _fetch("http://127.0.0.1:%s/" % match.group(1))[0] == 404
    bogus = re.sub(r"[A-Za-z0-9_-]{10,}$", "A" * 24, url)
    assert _fetch(bogus)[0] == 410

    # The existing browser URL policy accepts the http URL (with the operator's
    # documented loopback opt-in) and still rejects file-scheme navigation.
    operator, backend = _operator(env, allow_private=True)
    assert await operator.navigate(url) == url
    assert backend.calls == [{"action": "navigate", "url": url}]
    with pytest.raises(BrowserPermissionError):
        await operator.navigate("file:///etc/passwd")

    default_operator, _ = _operator(env, allow_private=False)
    with pytest.raises(BrowserPermissionError):
        await default_operator.navigate(url)  # loopback stays private by default

    source = Path(preview_tools.__file__).read_text(encoding="utf-8")
    for forbidden in ("subprocess", "webbrowser", "os.system", "file://", "shell=True"):
        assert forbidden not in source

    await _run(env, "preview_close", {"url": url})


async def test_outside_workspace_symlinks_and_unsafe_types_are_denied(env, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.html"
    secret.write_text("<h1>secret</h1>")
    inside = env.tmp_path / "page.html"
    inside.write_text("<h1>ok</h1>")

    link = env.tmp_path / "linked.html"
    link.symlink_to(secret)
    (env.tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    dotdot = env.tmp_path / ".." / outside.name / "secret.html"

    cases = [
        (str(secret), "outside allowed roots"),
        (str(link), "outside allowed roots"),
        (str(env.tmp_path / "escape" / "secret.html"), "outside allowed roots"),
        (str(dotdot), "outside allowed roots"),
        ("file://" + str(secret), "not a URL"),
        ("http://example.test/page.html", "not a URL"),
        (str(env.tmp_path), "regular file"),
        ("bad\x00.html", "NUL"),
    ]
    for value, expected in cases:
        result = await _run(env, "preview_local_file", {"path": value})
        assert not result.success, (value, result)
        assert expected in (result.error or ""), (value, result.error)

    for name, content, expected in [
        ("script.py", "print('hi')", "unsupported preview type"),
        ("vector.svg", "<svg/>", "unsupported preview type"),
        ("noextension", "text", "unsupported preview type"),
    ]:
        target = env.tmp_path / name
        target.write_text(content)
        result = await _run(env, "preview_local_file", {"path": str(target)})
        assert not result.success and expected in (result.error or "")

    huge = env.tmp_path / "huge.html"
    huge.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    result = await _run(env, "preview_local_file", {"path": str(huge)})
    assert not result.success and "too large" in (result.error or "")

    # Denials never leave an active token behind.
    assert preview_tools._STATE.previews == {}


async def test_symlink_swap_after_creation_is_not_served(env, tmp_path_factory) -> None:
    """Creation-time validation alone is not enough: the token must be rechecked."""
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.html"
    secret.write_text("<h1>secret</h1>")
    visible = env.tmp_path / "page.html"
    visible.write_text("<h1>public</h1>")

    result = await _run(env, "preview_local_file", {"path": str(visible), "ttl_seconds": 30})
    assert result.success, result.error
    url = result.output["url"]
    assert _fetch(url)[0] == 200 and _fetch(url)[2] == b"<h1>public</h1>"

    visible.unlink()
    visible.symlink_to(secret)
    status, _, body = _fetch(url)
    assert status == 403 and b"secret" not in body

    visible.unlink()
    visible.write_text("<h1>again</h1>")  # same name, different inode
    status, _, body = _fetch(url)
    assert status == 403 and b"again" not in body
    await _run(env, "preview_close", {"url": url})


async def test_ttl_expiry_close_and_actor_ownership(env) -> None:
    page = env.tmp_path / "page.txt"
    page.write_text("expires")

    short = await _run(env, "preview_local_file", {"path": str(page), "ttl_seconds": 1})
    assert short.success, short.error
    url = short.output["url"]
    assert _fetch(url)[0] == 200
    await asyncio.sleep(1.3)
    assert _fetch(url)[0] in (404, 410, None)

    live = await _run(env, "preview_local_file", {"path": str(page), "ttl_seconds": 30})
    assert live.success and live.output["url"] != url
    assert _fetch(live.output["url"])[0] == 200
    closed = await _run(env, "preview_close", {"url": live.output["url"]})
    assert closed.success and closed.output["closed"] == 1
    assert _fetch(live.output["url"])[0] in (404, 410, None)
    again = await _run(env, "preview_close", {"url": live.output["url"]})
    assert not again.success and "no active preview" in (again.error or "")

    other = await _run(env, "preview_local_file", {"path": str(page)}, actor="other")
    assert other.success, other.error
    stolen = await _run(env, "preview_close", {"url": other.output["url"]})
    assert not stolen.success
    assert _fetch(other.output["url"])[0] == 200
    assert (await _run(env, "preview_close", {}, actor="test")).output["closed"] == 0
    owned = await _run(env, "preview_close", {}, actor="other")
    assert owned.success and owned.output["closed"] == 1
    assert _fetch(other.output["url"])[0] in (404, 410, None)
    assert preview_tools._STATE.previews == {}
    assert preview_tools._STATE.server is None  # no idle listener left behind

    bad_url = await _run(env, "preview_close", {"url": "file:///tmp/nope"})
    assert not bad_url.success and "loopback preview URL" in (bad_url.error or "")


async def test_hard_linked_files_are_refused_at_creation_and_on_every_request(
    env, tmp_path_factory
) -> None:
    """A link inside a root may name an inode that lives outside every root."""
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.html"
    secret.write_text("<h1>OUTSIDE_SECRET</h1>")
    linked = env.tmp_path / "linked.html"
    try:
        os.link(secret, linked)  # same filesystem in the test temp area
    except OSError as exc:  # pragma: no cover - platform-specific link support
        pytest.skip(f"hard links unavailable: {exc}")

    denied = await _run(env, "preview_local_file", {"path": str(linked)})
    assert not denied.success and "hard-linked" in (denied.error or "")
    assert "OUTSIDE_SECRET" not in json.dumps(denied.output or {})
    linked.unlink()

    visible = env.tmp_path / "page.html"
    visible.write_text("<h1>public</h1>")
    issued = await _run(env, "preview_local_file", {"path": str(visible), "ttl_seconds": 30})
    assert issued.success, issued.error
    url = issued.output["url"]
    assert _fetch(url)[2] == b"<h1>public</h1>"

    # Adding a second link changes nlink but not the resolved path or inode, so
    # only the request-time nlink check can refuse it.
    alias = env.tmp_path / "alias.html"
    os.link(visible, alias)
    status, _, body = _fetch(url)
    assert status == 403 and b"hard-linked" in body and b"public" not in body
    alias.unlink()
    assert _fetch(url)[2] == b"<h1>public</h1>"
    await _run(env, "preview_close", {"url": url})


async def test_served_requests_are_audited_without_file_content(env) -> None:
    target = env.tmp_path / "audit.html"
    html = "<h1>AUDIT_SECRET_MARKER</h1>"
    target.write_text(html)
    issued = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 30})
    assert issued.success, issued.error
    url = issued.output["url"]
    token = url.rsplit("/", 1)[-1]
    audit_id = preview_tools.token_audit_id(token)
    assert audit_id != token and token not in audit_id

    assert _fetch(url)[0] == 200
    assert _fetch(url + "/no-such-path")[0] == 404
    assert _fetch(url, method="HEAD")[0] == 200

    rows = await db.fetchall(
        "SELECT actor,resource,decision,reason,metadata FROM audit_log "
        "WHERE action='preview.serve'"
    )
    served = [
        row for row in rows
        if (json.loads(row["metadata"] or "{}").get("token_id") or "") == audit_id
    ]
    assert len(served) == 2  # the 200 GET and the 200 HEAD for this token
    statuses: set[int] = set()
    methods: set[str] = set()
    for row in served:
        meta = json.loads(row["metadata"])
        statuses.add(meta["status"])
        methods.add(meta["method"])
        assert row["actor"] == "test"
        assert row["resource"] == str(target.resolve())
        assert meta["token_id"] == audit_id
        assert token not in row["metadata"] and token not in row["reason"]
        assert meta["actor"] == "test"
        assert meta["path"] == str(target.resolve())
        assert meta["bytes"] == len(html.encode())
        assert meta["byte_count"] == len(html.encode())
        assert html not in row["reason"] and html not in row["metadata"]
    assert statuses == {200} and methods == {"GET", "HEAD"}

    # Rejected paths are audited too, with no token to attribute them to.
    rejected = [
        json.loads(row["metadata"]) for row in rows
        if json.loads(row["metadata"] or "{}").get("status") == 404
    ]
    assert any(row["method"] == "GET" and row["decision"] == "denied" for row in rejected)
    assert all(html not in json.dumps(row) for row in rejected)

    # The same metadata is emitted on the in-process event bus for live views.
    await asyncio.sleep(0.1)
    events = await env.events.history("preview.serve", limit=500)  # shared bus: window must outlast other modules
    served_events = [event for event in events if event.payload.get("token_id") == audit_id]
    assert len(served_events) == 2
    assert all(token not in json.dumps(event.payload) for event in served_events)
    assert all(
        event.payload["actor"] == "test"
        and event.payload["path"] == str(target.resolve())
        and event.payload["status"] == 200
        and event.payload["bytes"] == len(html.encode())
        and html not in json.dumps(event.payload)
        for event in served_events
    )
    await _run(env, "preview_close", {"url": url})


async def test_host_header_must_match_assigned_loopback_listener(env) -> None:
    target = env.tmp_path / "page.txt"
    target.write_text("for the owner only")
    issued = await _run(env, "preview_local_file", {"path": str(target)})
    assert issued.success, issued.error
    url = issued.output["url"]
    match = URL_RE.match(url)
    assert match
    port = int(match.group(1))

    assert _fetch_host(url, f"127.0.0.1:{port}")[0] == 200
    assert _fetch_host(url, f"localhost:{port}")[0] == 200
    assert _fetch_host(url, "127.0.0.1")[0] == 400
    assert _fetch_host(url, f"evil.example:{port}")[0] == 400
    assert _fetch_host(url, f"127.0.0.1:{port + 1}")[0] == 400
    assert _fetch_host(url, f"LOCALHOST:{port}")[0] == 200
    assert _fetch_host(url, f"127.0.0.1:{port}", method="POST")[0] == 405
    assert _fetch_host(url, f"evil.example:{port}", method="POST")[0] == 400
    await _run(env, "preview_close", {"url": url})


async def test_browser_navigate_only_allows_registered_preview_tokens(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = env.tmp_path / "page.html"
    target.write_text("<h1>preview</h1>")
    issued = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 30})
    assert issued.success, issued.error
    url = issued.output["url"]
    token = url.rsplit("/", 1)[-1]

    operator, backend = _operator(env, allow_private=False)
    monkeypatch.setattr(operator_tools, "_browser_operator", lambda ctx: operator)
    operator_tools.register_operator_tools(
        env.registry, env.settings, env.permissions, include_computer=False
    )
    navigate = env.registry.get("browser_navigate").handler
    ctx = {"settings": env.settings, "permissions": env.permissions, "actor": "test"}

    with pytest.raises(BrowserPermissionError):
        await navigate({"url": "http://127.0.0.1:9/admin"}, ctx)
    with pytest.raises(BrowserPermissionError):
        await navigate({"url": f"http://127.0.0.1:9/preview/{token}"}, ctx)  # wrong port
    with pytest.raises(BrowserPermissionError):
        await navigate({"url": url.replace(token, "unregisteredtoken")}, ctx)
    with pytest.raises(BrowserPermissionError):
        await navigate({"url": "file:///etc/passwd"}, ctx)
    with pytest.raises(BrowserPermissionError):
        await navigate({"url": url}, {**ctx, "actor": "other"})  # not their token
    assert backend.calls == []

    opened = await navigate({"url": url}, ctx)
    assert opened["success"] is True
    assert opened["url"] == url
    assert opened["token_id"] == preview_tools.token_audit_id(token)
    assert token not in opened["token_id"]
    assert opened["registered_token"] is True and opened["path"] == str(target.resolve())
    assert backend.calls == [{"action": "navigate", "url": url}]

    # A closed token cannot be replayed as an allowance.
    await _run(env, "preview_close", {"url": url})
    with pytest.raises(BrowserPermissionError):
        await navigate({"url": url}, ctx)
    assert len(backend.calls) == 1


# --------------------------------------------------------------- F2 / F1 regressions


async def test_agent_message_events_cannot_replay_preview_tokens(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agent.message is visible on the feeds but its payload is sanitized."""
    target = env.tmp_path / "agent-event-leak.html"
    target.write_text("<h1>agent event leak check</h1>")
    issued = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 30})
    assert issued.success, issued.error
    url = issued.output["url"]
    token = url.rsplit("/", 1)[-1]
    port = URL_RE.match(url).group(1)
    digest = preview_tools.token_audit_id(token)
    redacted_url = f"http://127.0.0.1:{port}/preview/[REDACTED]"

    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    async def _pending() -> list[Any]:
        return []

    fake_app = SimpleNamespace(
        health=_health,
        permissions=SimpleNamespace(pending_approvals=_pending),
        events=env.events,
    )
    monkeypatch.setattr(api_module, "_app", fake_app, raising=False)
    monkeypatch.delenv("LEO_API_TOKEN", raising=False)

    payload = {
        "from_agent": "browser-tester",
        "to_agent": "planner",
        "content": {"result": json.dumps({"success": True, "url": url, "token_id": token})},
        "token_id": token,
        "text": f"open {url} then read {url}",
    }
    await env.events.emit("agent.message", **payload)
    # The durable copy is untouched: tools and sub-agents still see the real
    # capability, only the API response is redacted.
    stored = await env.events.history("agent.message", limit=5)
    assert stored and stored[0].payload == payload
    assert token in json.dumps(stored[0].model_dump(mode="json"))

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/status")
        assert status.status_code == 200
        event_feed = await client.get("/events", params={"topic": "agent.message"})
        assert event_feed.status_code == 200
        for body in (status.text, event_feed.text):
            assert token not in body
            assert digest in body
            assert redacted_url in body
        assert status.json()["recent_events"][0]["payload"]["token_id"] == digest
        assert event_feed.json()["events"][0]["payload"]["token_id"] == digest
        assert event_feed.json()["events"][0]["payload"]["content"]["result"].count(
            token
        ) == 0

        # SSE uses the same visible_events() path; the event is emitted after
        # connect because the live feed starts at the current tail.
        stream_token = api_module.issue_stream_token("local")["token"]
        # ASGITransport buffers the generator's body, and its disconnect probe
        # ends the feed immediately. Keep the endpoint connected while a
        # background publisher emits an event after the feed's tail snapshot.
        from starlette.requests import Request as StarletteRequest

        async def _connected(self: Any) -> bool:
            return False

        monkeypatch.setattr(StarletteRequest, "is_disconnected", _connected)

        async def _publish() -> None:
            for _ in range(40):
                await asyncio.sleep(0.05)
                await env.events.emit("agent.message", **payload)

        publisher = asyncio.create_task(_publish())
        try:
            sse = await client.get(
                "/events/stream",
                params={"token": stream_token, "max_events": 1, "max_seconds": 3},
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert sse.status_code == 200
        sse_body = sse.text
        assert '"topic":"agent.message"' in sse_body
        assert token not in sse_body
        assert digest in sse_body and redacted_url in sse_body

        # Nothing readable from any of the three feeds can be replayed as the
        # bearer credential, including the digest that replaces token_id.
        for credential in (digest, "[REDACTED]"):
            assert _fetch(f"http://127.0.0.1:{port}/preview/{credential}")[0] != 200

    # An authenticated API principal still receives the unsanitized payload on
    # the authenticated JSON feed; the durable store is never rewritten.
    monkeypatch.setenv("LEO_API_TOKEN", "agent-event-token")
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        private = await client.get(
            "/status", headers={"Authorization": "Bearer agent-event-token"}
        )
        assert private.status_code == 200
        assert token in private.text

        # A stream token minted by that principal does NOT get that private
        # view: the query token travels in a browser-readable URL, so every SSE
        # frame is redacted regardless of who minted it.
        privileged_stream = api_module.issue_stream_token("api-token")["token"]

        async def _publish() -> None:
            for _ in range(40):
                await asyncio.sleep(0.05)
                await env.events.emit("agent.message", **payload)

        publisher = asyncio.create_task(_publish())
        try:
            private_sse = await client.get(
                "/events/stream",
                params={"token": privileged_stream, "max_events": 1, "max_seconds": 3},
            )
        finally:
            publisher.cancel()
            try:
                await publisher
            except asyncio.CancelledError:
                pass
        assert private_sse.status_code == 200
        private_body = private_sse.text
        assert '"topic":"agent.message"' in private_body
        assert token not in private_body
        assert digest in private_body and redacted_url in private_body
        # Nothing in the privileged SSE frame can be replayed as the bearer
        # credential, while the genuine token still works and stays in the
        # event store (verified above) for in-process consumers.
        for credential in (digest, "[REDACTED]"):
            assert _fetch(f"http://127.0.0.1:{port}/preview/{credential}")[0] != 200
        assert _fetch(url)[0] == 200
        still_stored = await env.events.history("agent.message", limit=1)
        assert token in json.dumps(still_stored[0].model_dump(mode="json"))

    # The DB/model copy is still the genuine preview (verified above); after
    # closing it even the genuine token stops replaying.
    await _run(env, "preview_close", {"url": url})
    assert _fetch(f"http://127.0.0.1:{port}/preview/{token}")[0] != 200


async def test_preview_token_never_reaches_unauthenticated_api_feeds(env) -> None:
    """The bearer token stays in the one-time tool response only."""
    target = env.tmp_path / "leak-check.html"
    target.write_text("<h1>leak check</h1>")
    issued = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 30})
    assert issued.success, issued.error
    url = issued.output["url"]
    token = url.rsplit("/", 1)[-1]
    assert _fetch(url)[0] == 200
    await asyncio.sleep(0.1)

    assert "preview." in api_module.SENSITIVE_EVENT_PREFIXES
    preview_event = SimpleNamespace(topic="preview.serve", payload={"token_id": token})
    assert api_module.is_sensitive_event(preview_event) is True
    assert api_module.visible_events([preview_event], authenticated=False) == []
    assert api_module.visible_events([preview_event], authenticated=True) == [preview_event]

    anonymous = SimpleNamespace(state=SimpleNamespace(principal="local"))
    public = await api_module.audit(anonymous, limit=100)
    assert token not in json.dumps(public)
    public_preview_rows = [r for r in public["audit"] if r.get("action") == "preview.serve"]
    assert public_preview_rows
    assert all("token_id" not in (row.get("metadata") or "") for row in public_preview_rows)

    authenticated = SimpleNamespace(state=SimpleNamespace(principal="api-token"))
    private = await api_module.audit(authenticated, limit=100)
    digest = preview_tools.token_audit_id(token)
    private_preview_rows = [r for r in private["audit"] if r.get("action") == "preview.serve"]
    assert private_preview_rows and digest in private_preview_rows[0]["metadata"]
    assert token not in json.dumps(private)  # digest only, even when authenticated
    # The one-time response carries the URL (and therefore the token); no
    # durable or public surface does.
    assert issued.output["url"] == url and token in issued.output["url"]

    await _run(env, "preview_close", {"url": url})


async def test_browser_click_cannot_use_preview_as_loopback_pivot(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link inside a preview must not expose an arbitrary loopback service."""
    from leo.integrations.browser import FakeBrowserPage
    from leo.tools import operator_tools

    target = env.tmp_path / "linked.html"
    target.write_text("<a href='/admin'>admin</a>")
    issued = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 30})
    assert issued.success, issued.error
    url = issued.output["url"]
    admin = "http://127.0.0.1:18765/admin"

    page = FakeBrowserPage(
        pages={url: "PREVIEW_CONTENT", admin: "ADMIN_SECRET"},
        click_effects={"a": {"url": admin, "text": "admin view"}},
    )
    operator = BrowserOperator(
        env.permissions, backend=page, actor="test", base_dir=env.tmp_path,
        profile_dir=env.tmp_path / "profile", timeout=1.0,
    )
    monkeypatch.setattr(operator_tools, "_browser_operator", lambda ctx: operator)
    operator_tools.register_operator_tools(
        env.registry, env.settings, env.permissions, include_computer=False
    )
    ctx = {"settings": env.settings, "permissions": env.permissions, "actor": "test"}
    navigate = env.registry.get("browser_navigate").handler
    click = env.registry.get("browser_click").handler
    extract = env.registry.get("browser_extract_text").handler

    opened = await navigate({"url": url}, ctx)
    assert opened["registered_token"] is True
    assert operator.session_metadata()["registered_preview"]["url"] == url
    assert await extract({}, ctx) == {"success": True, "result": "PREVIEW_CONTENT"}

    with pytest.raises(BrowserPermissionError):
        await click({"selector": "a"}, ctx)
    assert "close" in [call["action"] for call in page.calls]
    with pytest.raises(BrowserPermissionError):
        await extract({}, ctx)

    metadata = operator.session_metadata()
    assert metadata["registered_preview"] is None and metadata["quarantined"] is True
    assert metadata["url"] == admin  # truthful about where the browser went
    assert "ADMIN_SECRET" not in json.dumps(metadata)
    await _run(env, "preview_close", {"url": url})


# ------------------------------------------------- unauthenticated read-path regressions


async def test_conversation_reads_cannot_replay_preview_tokens(env, monkeypatch) -> None:
    """GET /conversations and the sidebar must not return bearer credentials."""
    target = env.tmp_path / "chat-leak.html"
    target.write_text("<h1>leak check</h1>")
    issued = await _run(env, "preview_local_file", {"path": str(target), "ttl_seconds": 30})
    assert issued.success, issued.error
    url = issued.output["url"]
    token = url.rsplit("/", 1)[-1]
    port = URL_RE.match(url).group(1)
    digest = preview_tools.token_audit_id(token)

    conv_id = "conv-preview-leak"
    await db.execute(
        "INSERT INTO conversations(id,channel,subject,created_at,updated_at) VALUES(?,?,?,?,?)",
        (conv_id, "web", "leak", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    # Legacy rows carried the raw token; the read path must still sanitize them.
    stored = json.dumps({
        "success": True, "url": url, "token_id": token,
        "output": {"url": url, "token_id": token},
    })
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
        "VALUES(?,?,?,?,?)",
        (conv_id, "tool", stored, json.dumps({"name": "preview_local_file"}),
         "2026-01-01T00:00:00+00:00"),
    )

    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get(f"/conversations/{conv_id}")
        assert detail.status_code == 200
        body = detail.text
        assert token not in body
        assert f"http://127.0.0.1:{port}/preview/[REDACTED]" in body
        assert digest in body
        assert json.loads(detail.json()["messages"][0]["content"])["token_id"] == digest

        sidebar = await client.get("/conversations")
        assert sidebar.status_code == 200
        assert token not in sidebar.text
        assert "[REDACTED]" in sidebar.text
        rows = sidebar.json()["conversations"]
        assert any(row["id"] == conv_id and token not in row["preview"] for row in rows)

        # Nothing readable from either endpoint can be replayed as the token.
        redacted = f"http://127.0.0.1:{port}/preview/[REDACTED]"
        assert _fetch(redacted)[0] != 200
        # Every credential-looking string the response exposes is a digest or
        # an id - never the issued token that the preview server accepts.
        assert token not in re.findall(r"[A-Za-z0-9_-]{16,}", body)

    # An authenticated API token still sees the durable value unchanged.
    monkeypatch.setenv("LEO_API_TOKEN", "preview-read-test-token")
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.get(f"/conversations/{conv_id}")
        assert unauthorized.status_code == 401
        private = await client.get(
            f"/conversations/{conv_id}",
            headers={"Authorization": "Bearer preview-read-test-token"},
        )
        assert private.status_code == 200
        assert token in private.text
        assert json.loads(private.json()["messages"][0]["content"])["token_id"] == token

    # The DB/model copy was never mutated: the original capability still works
    # until it is explicitly closed, then replay fails.
    assert _fetch(url)[0] == 200
    await _run(env, "preview_close", {"url": url})
    assert _fetch(url)[0] in (404, 410, None)


# ---------------------------------------------------------- real Playwright first-navigation guard


async def test_real_playwright_first_preview_click_cannot_reach_loopback(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a real browser, the first preview navigation must already be guarded."""
    import sys

    # Other tests simulate a machine without Playwright by clearing these
    # modules. Snapshot before importing so this test leaves no import behind.
    playwright_modules = {
        name: module for name, module in sys.modules.items()
        if name == "playwright" or name.startswith("playwright.")
    }
    pytest.importorskip("playwright.async_api")

    class _CounterHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            self.server.hits.append(self.path)  # type: ignore[attr-defined]
            payload = b"INTERNAL_SECRET"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    counter = ThreadingHTTPServer(("127.0.0.1", 0), _CounterHandler)
    counter.hits = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=counter.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    internal_port = counter.server_address[1]
    url: str | None = None

    try:
        page = env.tmp_path / "guarded.html"
        page.write_text(
            f'<a id="go" href="http://127.0.0.1:{internal_port}/admin">go</a>'
        )
        issued = await _run(env, "preview_local_file", {"path": str(page), "ttl_seconds": 60})
        assert issued.success, issued.error
        url = issued.output["url"]
        token = url.rsplit("/", 1)[-1]
        port = int(URL_RE.match(url).group(1))
        preview = preview_tools.registered_preview(url, {"actor": "test"})
        assert preview is not None
        canonical = preview_tools.canonical_preview_url(preview)

        class _CountingBackend(PlaywrightBrowserBackend):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.requests: list[str] = []

            async def start(self) -> None:
                await super().start()
                self._context.on("request", lambda request: self.requests.append(request.url))

        monkeypatch.setattr(browser_module, "PlaywrightBrowserBackend", _CountingBackend)
        operator = BrowserOperator(
            env.permissions, actor="test", base_dir=env.tmp_path,
            profile_dir=env.tmp_path / "playwright-profile", timeout=15.0,
        )
        try:
            try:
                final = await operator.navigate_registered_preview(
                    url, token=token, port=port,
                    expires_at=preview.expires_at,
                    preview_validator=lambda: preview_tools.registered_preview(
                        canonical, {"actor": "test"}
                    ) is not None,
                )
            except BrowserError as exc:  # pragma: no cover - environment without Chromium
                if "could not start Playwright browser" not in str(exc):
                    raise  # a policy failure is a real test failure, not a skip
                pytest.skip(f"real Playwright browser unavailable: {exc}")
            assert final == url

            backend = operator._playwright_backend
            assert isinstance(backend, _CountingBackend)
            # The guard was installed *during* this first navigation.
            assert backend._guard_route_installed is True
            assert any(request == url for request in backend.requests)
            assert counter.hits == []

            # Clicking the loopback link must not perform the internal GET.
            try:
                await operator.click("#go")
            except BrowserPermissionError:
                pass  # quarantine on the denied navigation is expected
            assert counter.hits == []
            assert "INTERNAL_SECRET" not in str(operator.record())
        finally:
            await operator.close()
    finally:
        for name in list(sys.modules):
            if name == "playwright" or name.startswith("playwright."):
                sys.modules.pop(name, None)
        sys.modules.update(playwright_modules)
        counter.shutdown()
        counter.server_close()
        thread.join(timeout=2.0)
        if url is not None:
            await _run(env, "preview_close", {"url": url})
