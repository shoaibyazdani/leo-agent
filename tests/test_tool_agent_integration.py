"""End-to-end wiring contract: toolsets, specialists, gateway, tasks and API.

These tests use the real composition root, real executor and real databases.
The only substitutions are local/offline adapters and model providers; the
executor's service context is never monkeypatched.
"""

from __future__ import annotations

import sys

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.core import LeoApp
from leo.events import Event
from leo.interfaces import api as api_module
from leo.tools import MODEL_FORBIDDEN_TOOLS, register_all_toolsets
from leo.tools.builtins import register_builtins
from leo.tools.registry import ToolRegistry

SPECIALIST_IDS = (
    "automation-operator", "browser-tester", "evaluator", "memory-curator",
    "planner", "release-auditor", "reviewer", "skill-author",
)


class AllowAll:
    """Offline permission stub for the injected fake browser only."""

    async def authorize(self, actor, action, resource=None, risk=RiskLevel.LOW, **context):
        from leo.contracts import Decision

        return Decision(allowed=True, reason="test policy", risk=risk)


@pytest.fixture
async def app(monkeypatch: pytest.MonkeyPatch) -> LeoApp:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    instance = LeoApp(settings)
    await instance.initialize()
    monkeypatch.setattr(api_module, "_app", instance)
    yield instance
    await instance.shutdown()


async def test_full_tool_roster_is_registered_once(app: LeoApp) -> None:
    names = [spec.name for spec in app.registry.list_specs()]
    assert names and len(names) == len(set(names)), "duplicate tool names"
    # 24 non-operator registrar modules advertise 79 specs (this roster count
    # still includes json_query); the operator registrar adds 9 and builtins add
    # another 10, for 98 advertised names. json_query is deliberately withdrawn
    # from the model-callable catalogue because its caller-supplied data/default
    # arguments let a model feed invented numbers back as evidence, leaving 97
    # unique live names. preview_tools, http_tools and conversation_tools are all
    # part of the shipped roster, not optional extras.
    assert len(app.tool_roster) == 25, (
        "expected the live 25-module tool roster including preview_tools, "
        f"http_tools and conversation_tools; got {sorted(app.tool_roster)}"
    )
    registrars = {key: value for key, value in app.tool_roster.items() if key != "operator_tools"}
    assert len(registrars) == 24, sorted(registrars)
    assert {"preview_tools", "http_tools", "conversation_tools"} <= set(registrars)
    assert sum(len(value) for value in registrars.values()) == 79
    # The operator registrar declares 9 specs, but four of them drive the macOS
    # desktop through osascript/screencapture/pbcopy and are only registered on
    # darwin - on any other platform they would be advertised and then fail at
    # call time. So the roster is platform-correct, not truncated by accident:
    # assert the 5 browser tools everywhere, and the full 9 only where the
    # computer tools can actually run.
    operator_specs = 9 if sys.platform == "darwin" else 5
    assert len(app.tool_roster["operator_tools"]) == operator_specs
    assert len(names) == 88 + operator_specs
    assert "json_query" not in names  # operator/offline only, never model-callable
    assert all(app.tool_roster.values())
    for roster_names in app.tool_roster.values():
        # Registered rosters advertise specs before the forbidden-tool filter
        # runs; json_query and the operator-only actions are intentionally absent
        # from the model-callable registry.
        assert set(roster_names) - MODEL_FORBIDDEN_TOOLS <= set(names)
    builtin = ToolRegistry()
    register_builtins(builtin)
    assert {spec.name for spec in builtin.list_specs()} <= set(names)
    # Operator-only decisions are not exposed as model-callable tools.
    assert not MODEL_FORBIDDEN_TOOLS & set(names)
    assert "approval_resolve" not in names and "webhook_sign" not in names


async def test_all_twenty_specialists_loaded_with_callable_handlers(app: LeoApp) -> None:
    specs = await app.agent_registry.list()
    # The original twelve core roles plus at least the eight executable
    # specialists (the live roster may be extended by additional SEEDs).
    assert len(specs) >= 20
    assert set(SPECIALIST_IDS) <= {spec.id for spec in specs}
    for agent_id in SPECIALIST_IDS:
        handler = app.agent_registry.get_handler(agent_id)
        assert callable(handler), agent_id
        # A string-valued metadata handler is never bound as a callable.
        assert not isinstance(handler, str)
    assert app.services["agent_engine"] is app.agent_engine
    assert app.services["events"] is app.events


async def test_every_toolset_executes_through_real_executor(
    app: LeoApp, tmp_path: Path, monkeypatch, request
) -> None:
    """Each module's tool reaches a handler via executor._context services."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from leo.integrations.browser import FakeBrowserPage
    from leo.integrations.browser import BrowserOperator
    from leo.tools import builtins, operator_tools
    from leo.tools.builtins import is_public_ip as real_is_public

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    sample = workspace / "sample.txt"
    sample.write_text("integration fixture")
    browser = BrowserOperator(
        AllowAll(), backend=FakeBrowserPage(pages={"https://example.org/": "public text"}),
        profile_dir=workspace / "profile", base_dir=workspace, timeout=1.0,
    )
    monkeypatch.setitem(operator_tools._BROWSERS, "integration-test", browser)

    # http_json must fetch a real response. The shared egress policy rejects
    # loopback addresses, so only that policy seam is bridged for this local
    # fixture server; every other address still uses the production check.
    class _JsonHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            body = b'{"value": "integration-http"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    def admit_loopback_only(address):
        return address.is_loopback or real_is_public(address)

    monkeypatch.setattr(builtins, "is_public_ip", admit_loopback_only)
    http_server = ThreadingHTTPServer(("127.0.0.1", 0), _JsonHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    def stop_server() -> None:
        http_server.shutdown()
        http_server.server_close()
        http_thread.join(timeout=5)

    request.addfinalizer(stop_server)

    # conversation_messages reads durable history, so seed a real conversation
    # and turn before the executor matrix calls it.
    conversation_id = "toolset-conversation"
    await db.execute(
        "INSERT INTO conversations(id,channel,subject,created_at,updated_at) VALUES(?,?,?,?,?)",
        (conversation_id, "test", "toolset fixture",
         "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
        (conversation_id, "user", "seed for conversation tool",
         "2026-01-01T00:00:00+00:00"),
    )

    calls: list[tuple[str, dict]] = [
        ("agent_tools", "agent_list", {}),
        ("approval_tools", "approval_list", {}),
        ("attachment_tools", "attachment_list", {}),
        ("automation_tools", "schedule_list", {}),
        ("config_tools", "config_sources", {}),
        ("context_tools", "context_stats", {}),
        ("conversation_tools", "conversation_messages", {"conversation_id": conversation_id}),
        ("data_tools", "base64_codec", {"operation": "encode", "text": "hello"}),
        ("evaluation_tools", "evaluation_proposals", {}),
        ("event_tools", "event_query", {}),
        ("file_tools", "dir_tree", {"path": str(app.settings.root), "max_depth": 1, "max_entries": 3}),
        ("git_tools", "git_status", {}),
        ("health_tools", "app_health", {}),
        ("http_tools", "http_json", {
            "url": f"http://127.0.0.1:{http_server.server_address[1]}/rates",
            "path": "value",
        }),
        ("integration_tools", "integration_status", {}),
        ("memory_tools", "memory_list", {}),
        ("model_tools", "model_providers", {}),
        ("operator_tools", "browser_extract_text", {"selector": "body"}),
        ("preview_tools", "preview_close", {}),
        ("run_tools", "run_list", {}),
        ("search_tools", "search_all", {"query": "integration"}),
        ("skill_tools", "skill_list", {}),
        ("task_tools", "task_list", {}),
        ("webhook_tools", "webhook_routes", {}),
        ("model_config_tools", "model_add", {
            "name": "probe", "preset": "openrouter", "model": "probe/model",
            "config_path": str(app.settings.resolve(app.settings.paths.workspaces)
                              / "toolset-probe" / "probe-leo.yaml"),
        }),
    ]
    assert {module for module, _, _ in calls} == set(app.tool_roster)
    # model_add is HIGH risk: exercise its real write path with the gate
    # relaxed for this coverage run only (writing a throwaway config).
    high_risk_backup = app.settings.security.high_risk_requires_approval
    app.settings.security.high_risk_requires_approval = False
    seen_modules: set[str] = set()
    outputs: dict[str, dict] = {}
    for module, name, arguments in calls:
        assert name in app.tool_roster[module], (module, name)
        result = await app.executor.execute(
            ToolCall(
                name=name, arguments=arguments, actor="integration-test",
                task_id=f"toolset-{module}", correlation_id=f"toolset-{module}",
            )
        )
        assert result.decision == "allowed", f"{module}/{name}: {result.error}"
        assert result.error is None, f"{module}/{name}: {result.error}"
        assert result.success is True, f"{module}/{name}: {result.error}"
        outputs[module] = result.output
        seen_modules.add(module)
    app.settings.security.high_risk_requires_approval = high_risk_backup
    assert len(seen_modules) == 25
    # The added toolsets really ran: a value decoded from the fetched HTTP
    # response, the process-local preview table's close handler, and a real
    # page read from the seeded durable conversation.
    conversation_output = outputs["conversation_tools"]
    assert conversation_output["conversation_id"] == conversation_id
    assert conversation_output["count"] == 1
    assert "seed for conversation tool" in json.dumps(conversation_output["messages"])
    http_output = outputs["http_tools"]
    assert http_output["value"] == "integration-http"  # from the fixture body
    # A successful GET proves a fetch, not CDN freshness: "fetched-http" is the
    # honest provenance, and no canned/default value may stand in for real data.
    assert http_output["source"] == "fetched-http"
    assert http_output["found"] is True
    assert http_output["path"] == "value"
    assert http_output["status_code"] == 200
    canned = {"live-http", "unknown", "placeholder", "stub", "default", "n/a", "mock", "fake"}
    assert not canned & {str(value).lower() for value in http_output.values()}
    assert outputs["preview_tools"]["closed"] == 0


async def test_reserved_delegation_keys_are_stripped_and_schema_rejects_them(app: LeoApp) -> None:
    from leo.contracts import AgentResult

    captured: dict = {}

    async def capture(objective: str = "", context: dict | None = None, **kwargs):
        captured.update(context or {})
        return {"output": "ok", "evidence": [{"source": "test", "claim": "context captured"}]}

    app.agent_engine.set_handler("general", capture)
    result = await app.agent_engine.delegate(
        "inspect context",
        agent_id="general",
        context={
            "apply": True, "confirm": "yes", "approved_by": "model",
            "approval_id": "apr_forged", "apply_expiry": True,
            "services": {"permissions": "forged"}, "permissions": "forged",
            "settings": {}, "tools": "forged", "correlation_id": "safe-corr",
        },
    )
    assert result is not None
    for key in (
        "apply", "confirm", "approved_by", "approval_id", "apply_expiry",
        "services", "permissions", "settings",
    ):
        assert key not in captured, key
    assert captured["tools"] is not None  # engine's own scoped facade remains
    assert captured["correlation_id"] == "safe-corr"

    assert app.registry.validate("agent_delegate", {
        "objective": "x", "context": {"apply": True},
    })
    assert app.registry.validate("agent_delegate", {
        "objective": "x", "context": {"services": {}},
    })
    assert app.registry.validate("agent_delegate", {
        "objective": "x", "context": {"correlation_id": "safe"},
    }) == []


async def test_operator_risk_and_live_action_gating(app: LeoApp) -> None:
    from leo.tools.operator_tools import _live_computer_enabled

    shot = next(spec for spec in app.registry.list_specs() if spec.name == "browser_screenshot")
    assert shot.risk is RiskLevel.MEDIUM
    # A platform or explicit dry_run=False never implies desktop consent.
    assert _live_computer_enabled(app.settings) is False
    app.settings.computer.live_actions = True
    assert _live_computer_enabled(app.settings) is True


async def test_operator_approval_resolution_is_privileged_not_model_callable(app: LeoApp) -> None:
    assert [spec.name for spec in app.operator_registry.list_specs()] == ["approval_resolve"]
    assert [spec.name for spec in app.registry.list_specs()].count("approval_resolve") == 0

    decision = await app.permissions.authorize(
        "requester-agent", "tool:fixture", None, RiskLevel.HIGH
    )
    assert decision.requires_approval and decision.approval_id
    approval_id = decision.approval_id

    chat = await app.executor.execute(ToolCall(
        name="approval_resolve", arguments={"approval_id": approval_id, "approved": True},
        actor="chat:api",
    ))
    assert chat.decision == "unknown_tool" and chat.success is False

    resolved = await app.operator_executor.execute(ToolCall(
        name="approval_resolve", arguments={"approval_id": approval_id, "approved": True},
        actor="operator:local",
    ))
    assert resolved.success is True, resolved.error
    assert resolved.output["resolver"] == "operator:local"
    assert resolved.output["status"] == "approved"

    # The CLI/app facade uses the same privileged path, never the chat registry.
    third = await app.permissions.authorize(
        "requester-agent", "tool:fixture-cli", None, RiskLevel.HIGH
    )
    assert third.approval_id
    cli_result = await app.operator_resolve_approval(third.approval_id, True)
    assert cli_result["success"] is True, cli_result.get("error")

    # API route uses the same privileged executor and rejects other principals.
    second = await app.permissions.authorize(
        "requester-agent", "tool:fixture-2", None, RiskLevel.HIGH
    )
    assert second.approval_id
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.post("/operator/tools/approval_resolve/execute", json={
            "arguments": {"approval_id": second.approval_id, "approved": True},
        })
        assert ok.status_code == 200, ok.text
        assert ok.json()["success"] is True
        tools = await client.get("/tools")
        assert "approval_resolve" not in {item["name"] for item in tools.json()["tools"]}


async def test_gateway_reply_is_idempotent_and_truthful(app: LeoApp, monkeypatch) -> None:
    sends: list[dict] = []

    async def fake_send(channel, target, text, **kwargs):
        sends.append({"channel": channel, "target": target, "text": text, **kwargs})
        return {"success": True, "message_id": "msg-1"}

    async def fake_ask(text, **kwargs):
        return {"content": "reply text", "conversation_id": kwargs.get("conversation_id")}

    monkeypatch.setattr(app.gateway, "send", fake_send)
    monkeypatch.setattr(app, "ask", fake_ask)
    payload = {
        "channel": "cli", "reply_target": "user-1", "text": "ping",
        "outbox_id": "outbox-1", "idempotency_key": "idem-1",
        "conversation_id": "conv-1",
    }
    result = await app._gateway_reply(payload)
    assert result["status"] == "sent" and result["success"] is True
    assert result["delivered"] is True
    assert sends[0]["idempotency_key"] == "idem-1"

    async def failing_send(channel, target, text, **kwargs):
        return {"success": False, "error": "provider down"}

    monkeypatch.setattr(app.gateway, "send", failing_send)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await app._gateway_reply({**payload, "outbox_id": "outbox-2"})


async def test_gateway_outbox_drains_through_installed_reply_handler(app: LeoApp, monkeypatch) -> None:
    replies: list[dict] = []

    async def reply(payload: dict) -> dict:
        replies.append(payload)
        return {"status": "handled"}

    app.gateway.set_reply_handler(reply)
    receipt = await app.gateway.receive({
        "channel": "cli", "sender": "user-1", "sender_id": "user-1",
        "text": "hello from cli", "message_id": "msg-integration-outbox",
    })
    assert receipt["channel"] == "cli"
    stats = await app.gateway.outbox_stats()
    assert stats["pending"] >= 1
    outcome = await app.gateway.drain_outbox(limit=10)
    assert outcome["claimed"] >= 1 and outcome["processed"] >= 1
    assert replies and replies[0]["text"] == "hello from cli"
    assert (await app.gateway.outbox_stats())["pending"] == 0


async def test_scheduler_drain_dispatches_real_task(app: LeoApp) -> None:
    seen: list[str] = []

    async def worker(task: object, engine: object, context: dict) -> None:
        seen.append(context["objective"])
        await engine.complete(task.id, {"ok": True}, [{"source": "integration-worker", "claim": "done"}])

    app.tasks.register_worker("integration-worker", worker)
    task = await app.tasks.create(
        "run one durable scheduled objective", owner_agent="integration-worker"
    )
    outcome = await app.scheduler.drain(limit=10)
    assert any(getattr(item, "id", None) == task.id for item in outcome["dispatch"])
    assert seen == ["run one durable scheduled objective"]
    finished = await app.tasks.get(task.id)
    assert finished.status.value == "completed"
    assert finished.result["ok"] is True


async def test_events_pagination_and_sse_route(app: LeoApp) -> None:
    for index in range(5):
        await app.events.emit("integration.test", index=index)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/events", params={"topic": "integration.test", "limit": 2})
        assert first.status_code == 200
        body = first.json()
        assert body["total"] >= 5 and len(body["events"]) == 2
        assert body["has_more"] is True and body["next_cursor"]
        second = await client.get("/events", params={
            "topic": "integration.test", "limit": 2, "cursor": body["next_cursor"],
        })
        assert second.status_code == 200
        assert second.json()["events"][0]["id"] != body["events"][0]["id"]
        bad = await client.get("/events", params={"cursor": "not-a-cursor"})
        assert bad.status_code == 400

        # EventSource consumes a short-lived single-use token issued by an
        # authenticated POST; the token is required on the stream itself.
        grant = await client.post("/events/stream-token")
        assert grant.status_code == 200 and grant.json()["token"]
        assert (await client.get("/events/stream")).status_code == 401

        async def emit_later() -> None:
            await asyncio.sleep(0.2)
            await app.events.emit("integration.stream", marker="live")

        emitter = asyncio.create_task(emit_later())
        stream = None
        try:
            request = client.build_request("GET", "/events/stream", params={
                "token": grant.json()["token"], "max_events": 1, "max_seconds": 5,
            })
            stream = await client.send(request, stream=True)
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            lines = stream.aiter_lines()
            frame = await asyncio.wait_for(anext_lines(lines), timeout=10.0)
            assert frame.startswith("data: ")
            event = json.loads(frame[6:])
            assert event["topic"] == "integration.stream"
            assert event["payload"]["marker"] == "live"
        finally:
            if stream is not None:
                await stream.aclose()
            emitter.cancel()
            try:
                await emitter
            except asyncio.CancelledError:
                pass


async def anext_lines(iterator) -> str:
    async for line in iterator:
        if line.startswith("data: "):
            return line
    raise AssertionError("stream closed before a data frame")


async def test_sse_token_is_short_lived_single_use_and_bearer_gated(app: LeoApp, monkeypatch) -> None:
    monkeypatch.setenv("LEO_API_TOKEN", "integration-token")
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/events/stream-token")).status_code == 401
        headers = {"Authorization": "Bearer integration-token"}
        grant = await client.post("/events/stream-token", headers=headers)
        assert grant.status_code == 200
        token = grant.json()["token"]
        params = {"token": token, "max_events": 1, "max_seconds": 1}
        first = await client.get("/events/stream", params=params)
        assert first.status_code == 200
        assert first.headers["content-type"].startswith("text/event-stream")
        assert (await client.get("/events/stream", params=params)).status_code == 401
        monkeypatch.setattr(api_module, "STREAM_TOKEN_TTL_SECONDS", -1.0)
        expired = await client.post("/events/stream-token", headers=headers)
        stale = await client.get("/events/stream", params={
            "token": expired.json()["token"], "max_seconds": 1,
        })
        assert stale.status_code == 401
        assert (await client.get("/events")).status_code == 401
        assert (await client.get("/events", headers=headers)).status_code == 200


async def test_sensitive_topics_are_filtered_from_unauthenticated_feeds(app: LeoApp) -> None:
    await app.events.emit("tool.called", output="secret tool output")
    await app.events.emit("gateway.message", text="private message payload")
    await app.events.emit("integration.visible", ok=True)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = (await client.get("/events")).json()
        topics = {event["topic"] for event in page["events"]}
        assert "tool.called" not in topics and "gateway.message" not in topics
        assert "integration.visible" in topics
        status = (await client.get("/status")).json()
        assert all(
            event["topic"] not in ("tool.called", "gateway.message")
            for event in status["recent_events"]
        )
        grant = await client.post("/events/stream-token")
        async def emit_sensitive() -> None:
            await asyncio.sleep(0.2)
            await app.events.emit("tool.called", output="still secret")
            await app.events.emit("integration.visible", ok=True)
        emitter = asyncio.create_task(emit_sensitive())
        stream = None
        try:
            request = client.build_request("GET", "/events/stream", params={
                "token": grant.json()["token"], "max_events": 1, "max_seconds": 5,
            })
            stream = await client.send(request, stream=True)
            # Only the visible topic is emitted; the stream ends after 1 event.
            frame = await asyncio.wait_for(anext_lines(stream.aiter_lines()), timeout=10.0)
            assert json.loads(frame[6:])["topic"] == "integration.visible"
        finally:
            if stream is not None:
                await stream.aclose()
            emitter.cancel()
            try:
                await emitter
            except asyncio.CancelledError:
                pass


def test_cors_rejects_wildcard_with_credentials() -> None:
    assert api_module.cors_origins(["*"]) == []
    assert api_module.cors_origins(["http://localhost:5173", "*"]) == ["http://localhost:5173"]


async def test_health_counts_only_executable_registrations(app: LeoApp) -> None:
    report = await app.health()
    expected_handlers = sum(
        1 for spec in await app.agent_registry.list(enabled_only=True)
        if callable(app.agent_registry.get_handler(spec.id))
    )
    assert report["agents"] == expected_handlers
    assert report["agents"] < len(await app.agent_registry.list())  # handler-less excluded
    assert report["tools"] == len(app.registry.list_specs())
    broken = next(iter(app.registry.list_specs()))
    app.registry.get(broken.name).handler = "not callable"
    try:
        assert (await app.health())["tools"] == report["tools"] - 1
    finally:
        app.registry.get(broken.name).handler = (lambda args, ctx: None)


async def test_cors_allows_configured_dashboard_origin(app: LeoApp) -> None:
    origin = Settings.load().server.cors_origins[0]
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        preflight = await client.options("/events", headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        assert preflight.status_code in (200, 204), preflight.text
        assert preflight.headers.get("access-control-allow-origin") == origin
        assert "authorization" in preflight.headers.get("access-control-allow-headers", "").lower()


async def test_discord_verify_is_honest_and_dashboard_nav_link(app: LeoApp, monkeypatch) -> None:
    for name in ("DISCORD_BOT_TOKEN", "DISCORD_WEBHOOK_URL", "DISCORD_PUBLIC_KEY"):
        monkeypatch.delenv(name, raising=False)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        verify = await client.post("/integrations/discord/verify")
        assert verify.status_code == 200, verify.text
        assert verify.json()["ready"] is False
        assert verify.json()["missing"]
        assert verify.json()["webhook"] is False

        # Any one of the alternative Discord send credentials is sufficient;
        # missing signature material is still surfaced as degraded.
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-bot-token")
        with_token = await client.post("/integrations/discord/verify")
        assert with_token.status_code == 200
        assert with_token.json()["ready"] is True
        assert with_token.json()["degraded"] is True

        policies = await client.get("/integrations/policies")
        assert policies.status_code == 200 and "version" in policies.json()
        refreshed = await client.post("/integrations/policies/refresh", json={})
        assert refreshed.status_code == 200 and "policies" in refreshed.json()
        status = await client.get("/status")
        assert "channel_verification" in status.json()
        page = await client.get("/")
        assert page.status_code == 200
        assert 'href="/agent-activity/live"' in page.text


async def test_chat_uses_durable_history_and_configured_context_window(app: LeoApp, monkeypatch) -> None:
    app.settings.models.providers["mock"].context_window = 24680
    seen: list[list[tuple[str, str]]] = []
    budgets: list[int] = []
    original_complete = app.router.complete
    original_optimize = app.context.optimize

    async def spy_complete(request, *args, **kwargs):
        seen.append([(message.role, message.content) for message in request.messages])
        return await original_complete(request, *args, **kwargs)

    def spy_optimize(messages, budget, *args, **kwargs):
        budgets.append(budget)
        return original_optimize(messages, budget, *args, **kwargs)

    monkeypatch.setattr(app.router, "complete", spy_complete)
    monkeypatch.setattr(app.context, "optimize", spy_optimize)

    await app.ask("first durable question", conversation_id="conv-history")
    await app.ask("second durable question", conversation_id="conv-history")

    assert len(seen) >= 2
    second = seen[-1]
    assert ("user", "first durable question") in second
    assert ("user", "second durable question") in second
    assert [content for role, content in second].index("first durable question") < [
        content for role, content in second
    ].index("second durable question")
    assert budgets and set(budgets) == {24680}


async def test_streamed_tool_turn_persists_atomic_assistant_and_tool_pair(app: LeoApp, monkeypatch) -> None:
    rounds = {"n": 0}

    async def fake_stream(request):
        rounds["n"] += 1
        app.router.last_stream_provider = "mock"
        app.router.last_stream_content = ""
        app.router.last_stream_usage = None
        if rounds["n"] == 1:
            app.router.last_stream_tool_calls = [{
                "id": "call-history", "name": "calculator", "arguments": {"expression": "6*7"},
            }]
            return
            yield  # pragma: no cover - makes this an async generator
        app.router.last_stream_tool_calls = []
        app.router.last_stream_content = "The answer is 42."
        yield "The answer is 42."

    monkeypatch.setattr(app.router, "stream", fake_stream)
    events = [event async for event in app.ask_stream("calculate", conversation_id="conv-tools")]
    assert any(event["type"] == "message.finished" for event in events)
    rows = await db.fetchall(
        "SELECT role, content, metadata FROM messages WHERE conversation_id=? ORDER BY id",
        ("conv-tools",),
    )
    assistant = [row for row in rows if row["role"] == "assistant" and json.loads(row["metadata"] or "{}").get("tool_calls")]
    tool = [row for row in rows if row["role"] == "tool"]
    assert assistant and tool
    assert assistant[0]["metadata"].find("call-history") >= 0
    assert json.loads(tool[0]["metadata"])["tool_call_id"] == "call-history"


async def test_action_routes_are_live_and_capabilities_advertised(app: LeoApp, monkeypatch) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = (await client.get("/health")).json()
        assert health["capabilities"] == {
            "task_run": True, "agent_delegate": True,
            "schedule": True, "channel_send": True,
        }

        task = await app.tasks.create("run through the route", owner_agent="general")
        async def delegate_stub(objective, agent_id=None, **kwargs):
            return SimpleNamespace(success=True, task_id=task.id, model_dump=lambda: {})

        monkeypatch.setattr(app.agent_engine, "delegate", delegate_stub)
        run = await client.post(f"/tasks/{task.id}/run")
        assert run.status_code == 200, run.text
        assert "success" in run.json() and run.json()["task"]["id"] == task.id

        delegated = await client.post("/agents/delegate", json={"objective": "check routes"})
        assert delegated.status_code == 200, delegated.text
        assert delegated.json() == {"success": True, "task_id": task.id, "error": None}

        created = await client.post("/schedules", json={
            "name": "route-schedule", "seconds": 60, "objective": "tick",
        })
        assert created.status_code == 200, created.text
        listed = await client.get("/schedules")
        assert any(item["name"] == "route-schedule" for item in listed.json()["schedules"])
        removed = await client.delete("/schedules/route-schedule")
        assert removed.status_code == 200 and removed.json()["deleted"] == "route-schedule"
        assert (await client.delete("/schedules/route-schedule")).status_code == 404


async def test_channel_send_route_keeps_gateway_gates(app: LeoApp, monkeypatch) -> None:
    from leo.contracts import Decision
    from leo.integrations.gateway import GatewayPermissionError

    class FakeAdapter:
        channel = "route-fake"

        async def send(self, target, text):
            return {"success": True, "message_id": "route-msg-1"}

    monkeypatch.setitem(app.gateway.adapters, "route-fake", FakeAdapter())

    async def allow(actor, action, resource=None, risk=RiskLevel.LOW, **kwargs):
        return Decision(allowed=False, reason="approval required", risk=risk, requires_approval=True)

    monkeypatch.setattr(app.permissions, "authorize", allow)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.post("/integrations/route-fake/send", json={
            "target": "user-1", "text": "hello",
        })
        assert denied.status_code == 409, denied.text

        async def permitted(actor, action, resource=None, risk=RiskLevel.LOW, **kwargs):
            return Decision(allowed=True, reason="allowed", risk=risk)

        monkeypatch.setattr(app.permissions, "authorize", permitted)
        sent = await client.post("/integrations/route-fake/send", json={
            "target": "user-1", "text": "hello",
        })
        assert sent.status_code == 200, sent.text
        assert sent.json()["status"] == "sent" and sent.json()["delivered"] is True
        assert sent.json()["result"]["message_id"] == "route-msg-1"


async def test_event_durability_handlers_and_routes(app: LeoApp) -> None:
    from leo.core.app import DURABLE_SUBSCRIPTIONS

    subscriptions = await app.events.durable_subscriptions()
    attached = {item["name"] for item in subscriptions}
    assert {name for name, _ in DURABLE_SUBSCRIPTIONS} <= attached

    await app.events.emit("task.durable.test", marker="durable")
    stats = await app.events.stats()
    assert "attached_subscriptions" in stats
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        durability = await client.get("/events/durability")
        assert durability.status_code == 200 and isinstance(durability.json(), dict)
        listed = await client.get("/events/subscriptions")
        assert listed.status_code == 200 and attached <= {
            item["name"] for item in listed.json()["subscriptions"]
        }
        drained = await client.post("/events/durability/drain")
        assert drained.status_code == 200
        assert {"outbox", "replay", "deliveries"} <= set(drained.json())
