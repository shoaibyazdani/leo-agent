"""Integration toolset tests through the real ToolExecutorImpl.

Services are injected at the executor boundary with a real Gateway, a real
ChannelAdmission and an offline fake adapter: no credentials, no network and no
canned results. Every assertion checks a real routing call or a real blocked
record (adapter calls, DB rows, emitted events).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import ChannelsConfig, ChannelPolicyConfig, Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.integrations import enrollment
from leo.integrations.admission import ChannelAdmission
from leo.integrations.channels import ALL_ADAPTERS
from leo.integrations.gateway import Gateway, GatewayPermissionError
from leo.integrations.messaging import MessagingAdapter
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl
from leo.tools import integration_tools
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry

CHANNEL = "telegram"
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
FAKE_TOKEN = "fake-token-integration-tests-no-network"
NAMES = ["integration_status", "channel_send", "channel_react", "enrollment_status"]


class ServicesExecutor(ToolExecutorImpl):
    """Real executor; only the service-injection boundary is overridden."""

    def __init__(self, *args: object, services: dict, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.services = services

    def _context(self, call, task_key, dry_run):
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.services
        return ctx


class FakeAdapter(MessagingAdapter):
    """Offline channel adapter: records real send calls, never uses HTTP."""

    channel = CHANNEL
    credential_env = (TOKEN_ENV,)
    supports_webhook = False

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []
        self.reactions: list[tuple[str, str, str]] = []
        self.react_ok = True

    async def send(self, target: str, text: str) -> dict:
        self.calls.append((target, text))
        return {
            "channel": self.channel,
            "success": True,
            "message_id": f"fake-msg-{len(self.calls)}",
        }

    async def react(self, target: str, message_id: str, emoji: str = "\u2764") -> bool:
        self.reactions.append((target, message_id, emoji))
        return self.react_ok


class RecordingChannelRegistry:
    def __init__(self, adapter: FakeAdapter) -> None:
        self.adapter = adapter
        self.calls = 0

    def integration_records(self) -> list:
        self.calls += 1
        return [self.adapter.record()]


class RecordingEnrollment:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_channels(self, *, gateway=None) -> list[dict]:
        self.calls.append(("list_channels", gateway))
        return await enrollment.list_channels(gateway=gateway)

    async def channel_details(self, channel: str, *, gateway=None) -> dict:
        self.calls.append(("channel_details", channel, gateway))
        return await enrollment.channel_details(channel, gateway=gateway)


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.security.high_risk_requires_approval = False
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    assert integration_tools.register(registry) == NAMES
    executor = ServicesExecutor(
        registry, permissions, events, settings, services={}, timeout_seconds=5.0
    )
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        registry=registry, executor=executor,
    )


def build_gateway(env, *, mode: str = "allowlist", allow: list[str] | None = None,
                  with_policy: bool = True):
    adapter = FakeAdapter()
    admission = ChannelAdmission(ChannelsConfig())
    if with_policy:
        admission.set_policy(
            CHANNEL, ChannelPolicyConfig(mode=mode, allow=allow if allow is not None else ["alice"])
        )
    gateway = Gateway(
        {CHANNEL: adapter}, events=env.events, permissions=env.permissions, admission=admission
    )
    return gateway, adapter, admission


def install_services(env, gateway, admission, adapter):
    registry = RecordingChannelRegistry(adapter)
    enrollment_service = RecordingEnrollment()
    env.executor.services = {
        "gateway": gateway,
        "admission": admission,
        "channels": registry,
        "enrollment": enrollment_service,
    }
    return registry, enrollment_service


async def run(env, name: str, arguments: dict | None = None, **kwargs):
    return await env.executor.execute(ToolCall(name=name, arguments=arguments or {}, **kwargs))


async def records_for(tool: str) -> list[dict]:
    return [dict(row) for row in await db.fetchall("SELECT * FROM tool_calls WHERE tool=?", (tool,))]


# --------------------------------------------------------------------------- registration


def test_registration_and_typed_specs():
    registry = ToolRegistry()
    names = integration_tools.register(registry, services=None)
    assert names == NAMES
    specs = {spec.name: spec for spec in registry.list_specs()}
    assert set(specs) == set(NAMES)
    assert specs["integration_status"].category == "integrations"
    assert specs["enrollment_status"].category == "integrations"
    assert specs["channel_send"].category == "integrations"
    assert specs["integration_status"].risk is RiskLevel.LOW
    assert specs["enrollment_status"].risk is RiskLevel.LOW
    assert specs["channel_send"].risk is RiskLevel.HIGH
    assert specs["integration_status"].permissions == ["integration:read"]
    assert set(specs["channel_send"].permissions) == {"integration:send", "network:write"}
    for spec in specs.values():
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""
        assert spec.description and not spec.description.startswith("TODO")
    assert registry.validate("channel_send", {"channel": CHANNEL, "target": "chat:1"})
    assert registry.validate(
        "channel_send", {"channel": CHANNEL, "target": "chat:1", "text": "hi", "extra": 1}
    )
    assert registry.validate("integration_status", {"channel": CHANNEL}) == []


async def test_missing_gateway_service_fails_loudly(env):
    for name in NAMES:
        if name == "channel_send":
            args = {"channel": CHANNEL, "target": "chat:1", "text": "hi"}
        elif name == "channel_react":
            args = {"channel": CHANNEL, "target": "chat:1", "message_id": "42"}
        else:
            args = {"channel": CHANNEL}
        result = await run(env, name, args)
        assert result.success is False
        assert "gateway" in (result.error or "")
        assert "ctx['services']" in (result.error or "")


# --------------------------------------------------------------------------- status tools


async def test_integration_status_reads_real_records_and_policy(env, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env)
    channel_registry, _ = install_services(env, gateway, admission, adapter)
    seen = []
    env.events.subscribe("tool.called", seen.append)

    result = await run(env, "integration_status", {"channel": CHANNEL})
    assert result.success is True, result.error
    assert result.output["status"] == "ok"
    assert result.output["live_verified"] is False
    records = result.output["integrations"]
    assert [record["name"] for record in records] == [CHANNEL]
    assert records[0]["status"] == "configured"
    assert records[0]["verification"] == "mock-tested"
    assert records[0]["metadata"]["credentials_present"] == [TOKEN_ENV]
    assert records[0]["metadata"]["live_verified"] is False
    assert result.output["policies"][CHANNEL]["mode"] == "allowlist"
    assert result.output["summary"] == {"total": 1, "by_status": {"configured": 1}}
    assert channel_registry.calls == 1
    assert adapter.calls == []
    assert [event.payload["tool"] for event in seen] == ["integration_status"]
    assert (await records_for("integration_status"))[0]["decision"] == "allowed"

    blocked = await run(env, "integration_status", {"channel": "not-a-channel"})
    assert blocked.output["integrations"] == []
    assert blocked.output["summary"]["total"] == 0


async def test_enrollment_status_reads_real_enrollment_and_admission(env, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env, allow=["alice"])
    _, enrollment_service = install_services(env, gateway, admission, adapter)

    result = await run(env, "enrollment_status", {"channel": CHANNEL})
    assert result.success is True, result.error
    assert result.output["channel"] == CHANNEL
    detail = result.output["channels"][0]
    assert detail["channel"] == CHANNEL
    assert detail["status"] == "configured"
    assert detail["policy"]["allow"] == ["alice"]
    assert any(field["name"] == TOKEN_ENV and field["configured"] for field in detail["fields"])
    assert FAKE_TOKEN not in json.dumps(result.output)
    assert enrollment_service.calls == [("channel_details", CHANNEL, gateway)]
    assert result.output["admission"][CHANNEL]["mode"] == "allowlist"

    all_result = await run(env, "enrollment_status")
    assert all_result.success is True, all_result.error
    assert len(all_result.output["channels"]) == len(ALL_ADAPTERS)
    assert enrollment_service.calls[-1] == ("list_channels", gateway)
    assert all("channel" in item for item in all_result.output["channels"])


# --------------------------------------------------------------------------- channel_send


async def test_channel_send_routes_through_gateway_with_real_effects(env, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env)
    install_services(env, gateway, admission, adapter)
    sent, tool_events = [], []
    env.events.subscribe("gateway.sent", sent.append)
    env.events.subscribe("tool.called", tool_events.append)

    result = await run(env, "channel_send", {
        "channel": CHANNEL, "target": "chat-42", "text": "hello", "subject": "thread-7",
    })
    assert result.success is True, result.error
    output = result.output
    assert output["status"] == "sent" and output["delivered"] is True
    assert output["success"] is True
    assert output["provider_message_id"] == "fake-msg-1"
    assert output["subject"] == "thread-7"
    assert output["channel"] == CHANNEL and output["target"] == "chat-42"
    # The real Gateway dispatched to the fake adapter exactly once.
    assert adapter.calls == [("chat-42", "hello")]
    assert output["conversation_id"]
    rows = await db.fetchall(
        "SELECT role, content, metadata FROM messages WHERE conversation_id=? ORDER BY id",
        (output["conversation_id"],),
    )
    assert [(row["role"], row["content"]) for row in rows] == [("assistant", "hello")]
    assert json.loads(rows[0]["metadata"])["target"] == "chat-42"
    assert [event.payload["target"] for event in sent] == ["chat-42"]
    assert [event.payload["tool"] for event in tool_events] == ["channel_send"]
    assert (await records_for("channel_send"))[0]["decision"] == "allowed"


async def test_channel_send_blocked_without_credentials(env):
    gateway, adapter, admission = build_gateway(env)
    install_services(env, gateway, admission, adapter)
    blocked_events = []
    env.events.subscribe("integration.blocked", blocked_events.append)

    result = await run(env, "channel_send", {
        "channel": CHANNEL, "target": "chat-42", "text": "hello",
    })
    # The handler ran and returned a truthful blocked report; the executor
    # mirrors handler success=false, so the ToolResult is not a success.
    assert result.decision == "allowed" and result.success is False
    output = result.output
    assert output["status"] == "blocked"
    assert output["success"] is False and output["delivered"] is False
    assert TOKEN_ENV in output["reason"] and "missing credentials" in output["reason"]
    assert output["channel"] == CHANNEL and output["target"] == "chat-42"
    assert adapter.calls == []
    assert [event.payload["reason"] for event in blocked_events] == [output["reason"]]
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM messages"))["n"] == 0


@pytest.mark.parametrize("mode,expected", [
    ("deny", "denied by policy"),
    ("allowlist", "no channel policy configured"),
])
async def test_channel_send_blocked_by_missing_or_denying_policy(env, monkeypatch, mode, expected):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env, mode=mode, with_policy=mode == "deny")
    install_services(env, gateway, admission, adapter)

    result = await run(env, "channel_send", {
        "channel": CHANNEL, "target": "chat-42", "text": "hello",
    })
    assert result.output["status"] == "blocked"
    assert expected in result.output["reason"]
    assert result.output["delivered"] is False
    assert adapter.calls == []


async def test_channel_send_unknown_channel_is_blocked(env, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env)
    install_services(env, gateway, admission, adapter)

    result = await run(env, "channel_send", {
        "channel": "no-such-channel", "target": "chat-42", "text": "hello",
    })
    assert result.output["status"] == "blocked"
    assert "no integration record" in result.output["reason"]
    assert adapter.calls == []


async def test_channel_send_gateway_denial_is_not_reported_as_sent(env, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env)

    class DenyingGateway:
        def __init__(self, inner):
            self.inner = inner
            self.admission = admission
            self.calls = []

        def integration_records(self):
            return self.inner.integration_records()

        async def send(self, channel, target, text, **kwargs):
            self.calls.append((channel, target, text))
            raise GatewayPermissionError("gateway.send not permitted: denied")

    denying = DenyingGateway(gateway)
    env.executor.services = {"gateway": denying, "admission": admission}

    result = await run(env, "channel_send", {
        "channel": CHANNEL, "target": "chat-42", "text": "hello",
    })
    assert result.output["status"] == "blocked"
    assert result.output["success"] is False and result.output["delivered"] is False
    assert "gateway refused send" in result.output["reason"]
    assert "not permitted" in result.output["reason"]
    # The tool really attempted the gateway route; no adapter was reached.
    assert denying.calls == [(CHANNEL, "chat-42", "hello")]
    assert adapter.calls == []


# --------------------------------------------------------------------------- executor gates


async def test_invalid_arguments_never_reach_gateway(env, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env)
    install_services(env, gateway, admission, adapter)

    result = await run(env, "channel_send", {"channel": CHANNEL, "target": "chat-42"})
    assert result.success is False
    assert result.decision == "invalid_arguments"
    assert "text" in (result.error or "")
    assert adapter.calls == []
    assert (await records_for("channel_send"))[0]["decision"] == "invalid_arguments"


async def test_channel_send_high_risk_requires_approval(env, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    gateway, adapter, admission = build_gateway(env)
    install_services(env, gateway, admission, adapter)
    env.settings.security.high_risk_requires_approval = True

    with pytest.raises(ApprovalRequired):
        await run(env, "channel_send", {
            "channel": CHANNEL, "target": "chat-42", "text": "hello",
        })
    assert adapter.calls == []
