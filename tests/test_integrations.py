"""Offline verification for messaging adapters, browser/computer operators and gateway.

No live network calls, browser launches or desktop actions occur: HTTP uses
httpx.MockTransport, the browser uses FakeBrowserPage, and desktop commands go
through a fake subprocess runner or a fake tool executor.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from leo import db
from leo.config import ChannelsConfig, Settings
from leo.contracts import Decision, RiskLevel, ToolResult
from leo.events import EventBus
from leo.integrations import browser as browser_module
from leo.integrations.browser import BrowserError, BrowserOperator, BrowserPermissionError, FakeBrowserPage
from leo.integrations.computer import (
    ACTION_CAPABILITIES,
    ComputerError,
    ComputerOperator,
    ComputerPermissionDenied,
    UnsupportedOperationError,
)
from leo.integrations.admission import ChannelAdmission
from leo.integrations.gateway import (
    GATEWAY_MESSAGE_TOPIC,
    GATEWAY_SENT_TOPIC,
    Gateway,
    GatewayPermissionError,
    GatewayReplayError,
)
from leo.integrations.messaging import (
    ADAPTERS,
    DiscordAdapter,
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    TelegramAdapter,
    WhatsAppAdapter,
    WebhookVerificationError,
    integration_records,
)
from leo.security.permissions import PermissionManagerImpl


@pytest_asyncio.fixture
async def integration_db(tmp_path: Path):
    """Explicit async DB fixture as required by the integration stream."""
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    return tmp_path / "test.db"


class AllowAll:
    """Permission-manager stub used to isolate integration behavior from policy."""

    def __init__(self, *, deny_actions: frozenset[str] = frozenset(), deny_channel: str | None = None) -> None:
        self.calls: list[tuple[str, str | None, RiskLevel, dict]] = []
        self.deny_actions = set(deny_actions)
        self.deny_channel = deny_channel

    async def authorize(self, actor, action, resource=None, risk=RiskLevel.LOW, **context):
        self.calls.append((action, resource, risk, context))
        denied = action in self.deny_actions or (
            self.deny_channel is not None and context.get("channel") == self.deny_channel
        )
        return Decision(
            allowed=not denied,
            reason="denied by test policy" if denied else "allowed by test policy",
            risk=risk,
        )

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        return Decision(allowed=approved, reason="resolved by test", risk=RiskLevel.HIGH)


class FakeRunner:
    def __init__(self, stdout: str = "", stderr: str = "", code: int = 0) -> None:
        self.calls: list[tuple[list[str], str | None, float]] = []
        self.stdout = stdout
        self.stderr = stderr
        self.code = code

    def __call__(self, argv: list[str], stdin: str | None, timeout: float):
        self.calls.append((list(argv), stdin, timeout))
        return self.code, self.stdout, self.stderr


class FakeExecutor:
    def __init__(self) -> None:
        self.calls = []

    async def execute(self, call):
        self.calls.append(call)
        return ToolResult(call_id=call.id, tool=call.name, success=True, output={"ok": True})


def _signature(secret: str, payload) -> str:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- messaging


async def test_signature_comparison_is_case_insensitive_and_never_raises(monkeypatch):
    """Malformed signature headers must reject cleanly, never explode."""
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "wa-secret")
    adapter = WhatsAppAdapter()
    payload = {"object": "whatsapp_business_account", "entry": []}
    good = _signature("wa-secret", payload)
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": good}) is True
    # Providers occasionally send upper-case hex; accept it (constant time).
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": good.upper()}) is True
    # A non-ASCII header used to raise TypeError from hmac.compare_digest(str, str).
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": "sha256=ü"}) is False
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": "sha256=\udcff"}) is False


async def test_telegram_secret_token_non_ascii_is_rejected(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "right")
    adapter = TelegramAdapter()
    assert adapter.verify_webhook({}, {"x-telegram-bot-api-secret-token": "rigñt"}) is False


async def test_malformed_payloads_fail_with_messaging_error(monkeypatch):
    adapter = WhatsAppAdapter(require_verification=False)
    with pytest.raises(MessagingError, match="not valid JSON"):
        await adapter.handle_webhook("{not json", {})
    with pytest.raises(MessagingError, match="JSON object"):
        await adapter.handle_webhook("[1, 2, 3]", {})
    with pytest.raises(MessagingError, match="JSON object"):
        await adapter.handle_webhook(42, {})


async def test_telegram_accepts_batched_update_arrays(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    adapter = TelegramAdapter()
    updates = [
        {"message": {"message_id": 1, "from": {"id": 7, "username": "alice"}, "text": "one"}},
        {"message": {"message_id": 2, "from": {"id": 8, "username": "bob"}, "text": "two"}},
    ]
    messages = await adapter.handle_webhook(updates, {"x-hub-signature-256": _signature("tg-secret", updates)})
    assert [m.text for m in messages] == ["one", "two"]
    assert [m.sender for m in messages] == ["alice", "bob"]


async def test_telegram_webhook_secret_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "right")
    adapter = TelegramAdapter()
    assert adapter.verify_webhook({}, {"X-Telegram-Bot-Api-Secret-Token": "right"}) is True
    assert adapter.verify_webhook({}, {"X-Telegram-Bot-Api-Secret-Token": "wrong"}) is False


async def test_telegram_webhook_hmac_and_normalized_message(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    adapter = TelegramAdapter()
    payload = {
        "update_id": 1,
        "message": {
            "message_id": 42,
            "date": 1700000000,
            "from": {"id": 7, "username": "alice"},
            "text": "hello",
            "reply_to_message": {"message_id": 9},
        },
    }
    signature = _signature("tg-secret", payload)
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": signature}) is True
    messages = await adapter.handle_webhook(payload, {"x-hub-signature-256": signature})
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("telegram", "alice", "hello")
    assert message.reply_ref == "9" and message.message_id == "42"
    assert message.timestamp.startswith("2023-11-14")

    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"x-hub-signature-256": "sha256=" + "00" * 32})
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {})


async def test_telegram_send_uses_injected_client_and_hides_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-should-never-leak")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 99}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = TelegramAdapter(client=client)
        result = await adapter.send("12345", "hi")
    assert result["success"] and result["message_id"] == "99"
    assert str(requests[0].url).startswith("https://api.telegram.org/bot")
    assert json.loads(requests[0].content) == {"chat_id": "12345", "text": "hi"}

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"ok": False})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError) as exc:
            await TelegramAdapter(client=client).send("12345", "hi")
    assert "fake-token-should-never-leak" not in str(exc.value)


async def test_whatsapp_hmac_and_webhook_normalization(monkeypatch):
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "wa-secret")
    adapter = WhatsAppAdapter()
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "timestamp": "1700000000",
                            "contacts": [{"wa_id": "15551234"}],
                            "messages": [
                                {
                                    "id": "wamid.1",
                                    "from": "15551234",
                                    "text": {"body": "ping"},
                                    "context": {"id": "wamid.0"},
                                }
                            ],
                        }
                    }
                ]
            }
        ],
    }
    headers = {"x-hub-signature-256": _signature("wa-secret", payload)}
    assert adapter.verify_webhook(payload, headers) is True
    assert adapter.verify_webhook(payload, {"x-hub-signature-256": "sha256=deadbeef"}) is False
    messages = await adapter.handle_webhook(payload, headers)
    assert len(messages) == 1
    assert (messages[0].channel, messages[0].sender, messages[0].text) == ("whatsapp", "15551234", "ping")
    assert messages[0].reply_ref == "wamid.0"
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"x-hub-signature-256": "sha256=deadbeef"})


async def test_whatsapp_send_uses_injected_client(monkeypatch):
    monkeypatch.setenv("WHATSAPP_CLOUD_TOKEN", "fake-wa-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "fake-phone-id")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"messages": [{"id": "wamid.out"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await WhatsAppAdapter(client=client).send("15551234", "hello")
    assert result["message_id"] == "wamid.out"
    assert requests[0].headers["authorization"] == "Bearer fake-wa-token"
    body = json.loads(requests[0].content)
    assert body["messaging_product"] == "whatsapp" and body["text"] == {"body": "hello"}


async def test_discord_interaction_signature_and_normalization(monkeypatch):
    monkeypatch.setenv("DISCORD_PUBLIC_KEY", "ab" * 32)
    seen: list[tuple[str, str, str, bytes]] = []

    def verifier(public_key: str, signature: str, timestamp: str, body: bytes) -> bool:
        seen.append((public_key, signature, timestamp, body))
        return signature == "cd" * 32

    adapter = DiscordAdapter(signature_verifier=verifier)
    payload = {
        "type": 2,
        "id": "int-1",
        "timestamp": "2024-01-01T00:00:00Z",
        "data": {"name": "ask", "options": [{"value": "hello"}]},
        "user": {"id": "1", "username": "bob"},
    }
    headers = {"X-Signature-Ed25519": "cd" * 32, "X-Signature-Timestamp": "1700000000"}
    messages = await adapter.handle_webhook(payload, headers)
    assert len(messages) == 1
    assert (messages[0].channel, messages[0].sender, messages[0].text) == ("discord", "bob", "hello")
    assert seen and seen[0][0] == "ab" * 32

    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(payload, {"x-signature-ed25519": "ef" * 32, "x-signature-timestamp": "1"})
    assert await adapter.handle_webhook({"type": 1}, headers) == []  # PING
    gateway_message = {"t": "MESSAGE_CREATE", "d": {"id": "m2", "content": "from gateway", "author": {"username": "carol"}}}
    normalized = await adapter.handle_webhook(gateway_message, headers)
    assert normalized[0].sender == "carol" and normalized[0].text == "from gateway"


async def test_discord_webhook_send_and_ssrf_guard(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/fake")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = DiscordAdapter(client=client)
        result = await adapter.send("thread-9", "broadcast")
    assert result["success"] and json.loads(requests[0].content)["thread_id"] == "thread-9"
    with pytest.raises(MessagingError, match="not a Discord"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await DiscordAdapter(client=client).send("https://evil.example/hook", "no")


def test_integration_records_blocked_without_credentials(monkeypatch):
    for adapter in ADAPTERS.values():
        for name in adapter.credential_env + adapter.verification_env:
            monkeypatch.delenv(name, raising=False)
    records = integration_records()
    assert len(records) == 3
    assert all(record.status == "blocked" for record in records)
    assert all(record.verification == "mock-tested" for record in records)
    assert all(record.metadata["live_verified"] is False for record in records)
    assert "fake-token" not in json.dumps([r.model_dump() for r in records])

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    record = TelegramAdapter().record()
    assert record.status == "configured" and record.verification == "mock-tested"
    assert record.metadata["credential_env"] == ["TELEGRAM_BOT_TOKEN"]
    assert "fake-token" not in json.dumps(record.model_dump())


# -------------------------------------------------------------------------------- browser


@pytest.fixture
def browser_no_network(monkeypatch):
    """No test may touch DNS: names resolve to a fixed public address."""
    monkeypatch.setattr(
        browser_module, "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port or 443))],
    )


async def test_browser_fake_backend_full_flow(tmp_path: Path, browser_no_network):
    operator = BrowserOperator(
        AllowAll(),
        backend=FakeBrowserPage(
            pages={"https://example.org/": "public text"},
            downloads={"https://example.org/file.txt": b"contents"},
        ),
        profile_dir=tmp_path / "profile",
        base_dir=tmp_path,
        timeout=1.0,
    )
    record = operator.record()
    assert record.status == "degraded"  # injected fake is never "configured"
    assert record.verification == "mock-tested"
    assert record.metadata["backend"] == "FakeBrowserPage"
    assert record.metadata["backend_mode"] == "fake"
    assert record.metadata["live_verified"] is False
    assert await operator.navigate("https://example.org/") == "https://example.org/"
    assert await operator.extract_text() == "public text"
    await operator.click("#go")
    await operator.fill("#q", "query")
    assert await operator.screenshot(tmp_path / "shot.png") == b"\x89PNG\r\n\x1a\nfake"
    assert (tmp_path / "shot.png").read_bytes().startswith(b"\x89PNG")
    target = await operator.download("https://example.org/file.txt", tmp_path / "file.txt")
    assert target.read_bytes() == b"contents"
    actions = [call["action"] for call in operator._backend.calls]
    assert {"navigate", "extract_text", "click", "fill", "screenshot", "download"} <= set(actions)
    await operator.close()


async def test_browser_blocks_private_urls_and_paths(tmp_path: Path):
    operator = BrowserOperator(
        AllowAll(),
        backend=FakeBrowserPage(),
        profile_dir=tmp_path / "profile",
        base_dir=tmp_path,
    )
    for url in ("http://127.0.0.1:8080", "http://localhost", "http://169.254.169.254/latest", "file:///etc/passwd"):
        with pytest.raises(BrowserPermissionError):
            await operator.navigate(url)
    assert await operator.navigate("http://127.0.0.1:8080", allow_private=True) == "http://127.0.0.1:8080"

    restricted = BrowserOperator(
        PermissionManagerImpl(Settings()),
        backend=FakeBrowserPage(),
        profile_dir=tmp_path / "profile2",
        base_dir=tmp_path,
    )
    with pytest.raises(BrowserPermissionError, match="not permitted"):
        await restricted.screenshot(tmp_path / "outside.png")


async def test_browser_without_playwright_fails_gracefully(monkeypatch, tmp_path: Path, browser_no_network):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "playwright" else real_find_spec(name))
    monkeypatch.setitem(sys.modules, "playwright", None)
    operator = BrowserOperator(AllowAll(), profile_dir=tmp_path / "profile", base_dir=tmp_path)
    record = operator.record()
    assert record.status == "blocked" and record.verification == "not-verified"
    assert record.metadata["live_verified"] is False
    with pytest.raises(BrowserError, match="Playwright is not installed"):
        await operator.navigate("https://example.org/")


async def test_browser_blocks_dns_names_resolving_private(tmp_path: Path, monkeypatch):
    """A DNS name whose A record is private is denied before the browser runs."""
    operator = BrowserOperator(
        AllowAll(),
        backend=FakeBrowserPage(),
        profile_dir=tmp_path / "profile",
        base_dir=tmp_path,
    )
    monkeypatch.setattr(
        browser_module, "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("10.0.0.7", port or 80))],
    )
    with pytest.raises(BrowserPermissionError, match="non-public"):
        await operator.navigate("http://internal.example/")
    with pytest.raises(BrowserPermissionError, match="non-public"):
        await operator.download("http://internal.example/x", tmp_path / "x")
    assert operator._backend.calls == []  # nothing reached the page backend

    # A public resolution passes and the URL reaches the backend unchanged.
    monkeypatch.setattr(
        browser_module, "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port or 443))],
    )
    assert await operator.navigate("https://example.org/") == "https://example.org/"


# -------------------------------------------------------------------------------- computer


async def test_computer_dry_run_by_default_and_read_actions():
    runner = FakeRunner(stdout="clipboard text")
    # Explicit capabilities isolate policy from host probes: a fake runner must
    # not be mistaken for evidence that osascript/pbpaste really exist.
    caps = set(ACTION_CAPABILITIES.values())
    operator = ComputerOperator(
        AllowAll(), runner=runner, platform="darwin", dry_run=True, capabilities=caps
    )
    result = await operator.notify("hello")
    assert result["dry_run"] is True and result["executed"] is False and runner.calls == []
    assert (await operator.clipboard_write("secret"))["executed"] is False
    assert runner.calls == []
    # Dry-run suppresses host reads by default: no clipboard content leaves the
    # host just because a caller requested a preview.
    suppressed = await operator.clipboard_read()
    assert suppressed["executed"] is False and suppressed["text"] is None
    assert suppressed["host_read"] == "suppressed" and runner.calls == []
    read = await operator.clipboard_read(allow_host_read=True)
    assert read["executed"] is True and read["text"] == "clipboard text"
    assert runner.calls[0][0] == ["pbpaste"]
    assert operator.record().status == "configured"


async def test_computer_applescript_and_screenshot_use_argv():
    runner = FakeRunner(stdout="OK")
    operator = ComputerOperator(
        AllowAll(), runner=runner, platform="darwin", dry_run=False, timeout=2.0,
        capabilities=set(ACTION_CAPABILITIES.values()),
    )
    await operator.notify('it said "hi"', title="T")
    argv = runner.calls[0][0]
    assert argv[0] == "osascript" and argv[1] == "-e"
    assert '\\"hi\\"' in argv[2] and runner.calls[0][2] == 2.0
    await operator.dialog("Continue?", buttons=("No", "Yes"))
    assert "buttons" in runner.calls[1][0][2]
    shot = await operator.screenshot("/tmp/leo-shot.png")
    assert shot["executed"] is True and runner.calls[2][0][:2] == ["screencapture", "-x"]
    assert runner.calls[2][0][2].endswith("leo-shot.png")


async def test_computer_permission_denial_and_approval():
    settings = Settings()
    permissions = PermissionManagerImpl(settings)
    operator = ComputerOperator(
        permissions, runner=FakeRunner(), platform="darwin", dry_run=False,
        capabilities=set(ACTION_CAPABILITIES.values()), tool_executor=FakeExecutor(),
    )
    with pytest.raises(ComputerPermissionDenied, match="path outside allowed roots"):
        await operator.screenshot("/etc/passwd")
    with pytest.raises(ComputerPermissionDenied) as denied:
        await operator.file_delete(Path.cwd() / "important.txt")
    assert denied.value.decision.approval_id
    approvals = await permissions.pending_approvals()
    assert len(approvals) == 1 and approvals[0]["action"] == "file.delete"


async def test_computer_unsupported_off_macos():
    operator = ComputerOperator(AllowAll(), runner=FakeRunner(), platform="linux", dry_run=False)
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await operator.list_windows()
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await operator.focus_window("Terminal")
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await operator.notify("hi")
    assert operator.record().status == "blocked"


async def test_computer_file_ops_delegated_to_tool_executor(tmp_path: Path):
    executor = FakeExecutor()
    caps = {cap for cap in ACTION_CAPABILITIES.values() if cap.startswith("file.")}
    dry = ComputerOperator(
        AllowAll(), tool_executor=executor, platform="darwin", dry_run=True, capabilities=caps
    )
    assert (await dry.file_write(tmp_path / "a.txt", "hello"))["dry_run"] is True
    assert executor.calls == []
    live = ComputerOperator(
        AllowAll(), tool_executor=executor, platform="darwin", dry_run=False, capabilities=caps
    )
    result = await live.file_write(tmp_path / "a.txt", "hello")
    assert result["executed"] and result["success"] and result["output"] == {"ok": True}
    assert executor.calls[0].name == "file_write" and executor.calls[0].arguments["content"] == "hello"
    assert (await live.file_read(tmp_path / "a.txt"))["success"] is True
    with pytest.raises(ComputerError, match="no tool executor"):
        await ComputerOperator(
            AllowAll(), platform="darwin", dry_run=False, capabilities=caps
        ).file_delete(tmp_path / "a.txt")


# -------------------------------------------------------------------------------- gateway


async def test_gateway_inbound_routing_and_session_isolation(integration_db):
    bus = EventBus(persist=True)
    seen: list = []

    async def collect(event) -> None:
        seen.append(event)

    bus.subscribe(GATEWAY_MESSAGE_TOPIC, collect)
    permissions = AllowAll()
    # Routing test: every sender admitted. The fail-closed default policy is
    # asserted separately in test_channels_registry.py.
    gateway = Gateway(events=bus, permissions=permissions, admission=ChannelAdmission(ChannelsConfig(default_mode="open")))

    first = await gateway.receive(NormalizedMessage(channel="telegram", sender="alice", text="one", reply_ref="7"))
    await gateway.receive(NormalizedMessage(channel="telegram", sender="bob", text="two"))
    await gateway.receive(NormalizedMessage(channel="whatsapp", sender="alice", text="three"))

    sessions = await gateway.sessions()
    assert len(sessions) == 3
    assert len(await gateway.history(channel="telegram")) == 2
    assert len(await gateway.history(subject="alice")) == 2
    unified = await gateway.history()
    assert len(unified) == 3 and {item["role"] for item in unified} == {"user"}
    assert first["conversation_id"] == unified[0]["conversation_id"]
    assert len(seen) == 3 and seen[0].topic == GATEWAY_MESSAGE_TOPIC
    assert seen[0].payload["sender"] == "alice" and seen[0].payload["reply_ref"] == "7"
    assert any(call[0] == "gateway.receive" for call in permissions.calls)


async def test_gateway_history_respects_permissions(integration_db):
    open_admission = ChannelAdmission(ChannelsConfig(default_mode="open"))
    writer = Gateway(events=EventBus(persist=False), permissions=AllowAll(), admission=open_admission)
    await writer.receive(NormalizedMessage(channel="whatsapp", sender="alice", text="private"))
    gateway = Gateway(events=EventBus(persist=False), permissions=AllowAll(deny_channel="whatsapp"), admission=open_admission)
    with pytest.raises(GatewayPermissionError):
        await gateway.history(channel="whatsapp")
    assert len(await gateway.history(channel="telegram")) == 0


async def test_gateway_receive_webhook_verifies_and_routes(integration_db, monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    bus = EventBus(persist=True)
    gateway = Gateway(
        {"telegram": TelegramAdapter()},
        events=bus,
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    payload = {"message": {"message_id": 11, "from": {"username": "zoe"}, "text": "via webhook"}}
    routed = await gateway.receive_webhook("telegram", payload, {"x-telegram-bot-api-secret-token": "tg-secret"})
    assert len(routed) == 1 and routed[0]["text"] == "via webhook"
    assert (await gateway.history())[0]["metadata"]["sender"] == "zoe"
    with pytest.raises(WebhookVerificationError):
        await gateway.receive_webhook("telegram", payload, {"x-telegram-bot-api-secret-token": "wrong"})


async def test_gateway_outbound_fanout_and_high_risk_approval(integration_db, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-tg-token")
    monkeypatch.setenv("WHATSAPP_CLOUD_TOKEN", "fake-wa-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "fake-phone-id")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "telegram" in str(request.url):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        return httpx.Response(200, json={"messages": [{"id": "wamid.1"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bus = EventBus(persist=True)
    sent_events: list = []

    async def collect(event) -> None:
        sent_events.append(event)

    bus.subscribe(GATEWAY_SENT_TOPIC, collect)
    gateway = Gateway(
        {"telegram": TelegramAdapter(client=client), "whatsapp": WhatsAppAdapter(client=client)},
        events=bus,
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    results = await gateway.fanout({"telegram": "123", "whatsapp": "1555"}, "broadcast")
    assert results["telegram"]["success"] and results["whatsapp"]["success"]
    assert len(requests) == 2 and len(sent_events) == 2
    outbound = [item for item in await gateway.history() if item["role"] == "assistant"]
    assert len(outbound) == 2 and all(item["content"] == "broadcast" for item in outbound)
    assert (await gateway.history(channel="telegram"))[0]["role"] == "assistant"
    await client.aclose()

    strict = PermissionManagerImpl(Settings())
    blocked = Gateway({"telegram": TelegramAdapter()}, events=EventBus(persist=False), permissions=strict)
    with pytest.raises(GatewayPermissionError) as exc:
        await blocked.send("telegram", "123", "needs approval")
    assert exc.value.approval_id
    assert len(await strict.pending_approvals()) == 1


async def test_gateway_webhook_replay_is_deduplicated(integration_db, monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    gateway = Gateway(
        {"telegram": TelegramAdapter()},
        events=EventBus(persist=False),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    payload = {"message": {"message_id": 77, "from": {"id": 7}, "text": "once"}}
    headers = {"x-telegram-bot-api-secret-token": "tg-secret"}
    first = await gateway.receive_webhook("telegram", payload, headers)
    assert len(first) == 1
    with pytest.raises(GatewayReplayError, match="duplicate"):
        await gateway.receive_webhook("telegram", payload, headers)
    # A distinct delivery of the same channel is still accepted.
    payload2 = {"message": {"message_id": 78, "from": {"id": 7}, "text": "twice"}}
    second = await gateway.receive_webhook("telegram", payload2, headers)
    assert len(second) == 1
    assert [item["content"] for item in await gateway.history()] == ["once", "twice"]


async def test_failed_verification_does_not_poison_webhook_dedupe(integration_db, monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    gateway = Gateway(
        {"telegram": TelegramAdapter()},
        events=EventBus(persist=False),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    payload = {"message": {"message_id": 91, "from": {"id": 7}, "text": "legit"}}
    with pytest.raises(WebhookVerificationError):
        await gateway.receive_webhook(
            "telegram", payload, {"x-telegram-bot-api-secret-token": "wrong"}
        )
    # The unverified request must not suppress the real delivery.
    routed = await gateway.receive_webhook(
        "telegram", payload, {"x-telegram-bot-api-secret-token": "tg-secret"}
    )
    assert len(routed) == 1 and len(await gateway.history()) == 1


async def test_webhook_replay_window_expires(integration_db, monkeypatch):
    from leo.integrations import gateway as gateway_module

    clock = {"now": time.time()}
    monkeypatch.setattr(gateway_module, "time", type("Clock", (), {"time": staticmethod(lambda: clock["now"])})())
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    gateway = Gateway(
        {"telegram": TelegramAdapter()},
        events=EventBus(persist=False),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
        webhook_replay_window=60.0,
    )
    payload = {"message": {"message_id": 55, "from": {"id": 7}, "text": "again"}}
    headers = {"x-telegram-bot-api-secret-token": "tg-secret"}
    await gateway.receive_webhook("telegram", payload, headers)
    with pytest.raises(GatewayReplayError):
        await gateway.receive_webhook("telegram", payload, headers)
    clock["now"] += 61.0
    # The webhook fingerprint replay window has expired, so the delivery is
    # accepted again; the durable message-id dedupe still prevents a second
    # conversation row for the same provider message.
    delivered = await gateway.receive_webhook("telegram", payload, headers)
    assert len(delivered) == 1 and delivered[0]["duplicate"] is True
    assert len(await gateway.history()) == 1


async def test_concurrent_receives_share_one_deterministic_session(integration_db):
    gateway = Gateway(
        events=EventBus(persist=False),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    routed = await asyncio.gather(
        *(
            gateway.receive(NormalizedMessage(channel="telegram", sender="alice", text=f"m{i}"))
            for i in range(20)
        )
    )
    assert len({item["conversation_id"] for item in routed}) == 1
    sessions = await gateway.sessions(channel="telegram")
    assert len(sessions) == 1 and sessions[0]["message_count"] == 20
    assert len(await gateway.history(channel="telegram", limit=100)) == 20


async def test_gateway_send_denied_creates_no_session(integration_db):
    gateway = Gateway(
        {"telegram": TelegramAdapter()},
        events=EventBus(persist=False),
        permissions=AllowAll(deny_actions={"gateway.send"}),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    with pytest.raises(GatewayPermissionError):
        await gateway.send("telegram", "123", "denied")
    assert await gateway.sessions() == []
    assert await gateway.history() == []


async def test_gateway_send_delivery_failure_is_reported(integration_db):
    class FailingAdapter(MessagingAdapter):
        channel = "failing"

        async def send(self, target: str, text: str) -> dict[str, Any]:
            raise MessagingError("provider refused the message")

    gateway = Gateway(
        {"failing": FailingAdapter()},
        events=EventBus(persist=False),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    result = await gateway.send("failing", "target", "hello")
    assert result["success"] is False and "provider refused" in result["error"]
    assert [item["role"] for item in await gateway.history()] == []
