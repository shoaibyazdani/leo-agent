"""Channel registry, central admission and Connected Surfaces sync.

Covers the framework the channel adapters build on: auto-discovery of the
adapter matrix, the fail-closed inbound admission resolver, the upsert that
makes channels appear in the dashboard, and the dynamic webhook/poll routes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from leo import db
from leo.config import ChannelPolicyConfig, ChannelsConfig, ModelConfig, ModelProviderConfig, Settings
from leo.contracts import IntegrationRecord
from leo.core import LeoApp
from leo.integrations import channels as channels_module
from leo.integrations.channels import (
    ALL_ADAPTERS,
    BROKEN_MODULES,
    adapter_classes,
    default_adapters,
    integration_records,
    polling_channels,
    sync_integration_records,
    webhook_channels,
)
from leo.integrations.admission import AdmissionDecision, ChannelAdmission
from leo.integrations.gateway import Gateway, GatewayAdmissionError, GatewayError
from leo.integrations.messaging import MessagingAdapter, MessagingError, NormalizedMessage
from leo.interfaces import api as api_module

import httpx


# --------------------------------------------------------------------- discovery


def test_registry_discovers_core_channels_and_reports_broken_modules():
    # The three core adapters from leo.integrations.messaging are merged in.
    assert {"telegram", "discord", "whatsapp"} <= set(ALL_ADAPTERS)
    # A clean tree has no broken package modules; a broken one must surface
    # as an error record, never a silent gap.
    assert BROKEN_MODULES == {}
    records = integration_records()
    names = {record.name for record in records}
    assert {"telegram", "discord", "whatsapp"} <= names


def test_capability_classification_matches_adapter_flags():
    classes = adapter_classes()
    for name, cls in classes.items():
        assert issubclass(cls, MessagingAdapter)
        assert cls.channel == name
        if cls.supports_webhook:
            assert name in webhook_channels()
        else:
            assert name not in webhook_channels()
        if cls.supports_polling:
            assert name in polling_channels()
        else:
            assert name not in polling_channels()
    assert "telegram" in webhook_channels()
    assert "telegram" not in polling_channels()


async def test_base_poll_raises_for_non_polling_adapter():
    adapter = ALL_ADAPTERS["telegram"]()
    with pytest.raises(MessagingError, match="does not support polling"):
        await adapter.poll()


def test_broken_module_becomes_error_record(monkeypatch):
    monkeypatch.setattr(channels_module, "BROKEN_MODULES", {"broken_channel": "SyntaxError: boom"})
    records = integration_records()
    error = [r for r in records if r.name == "channel-module:broken_channel"]
    assert len(error) == 1 and error[0].status == "error"
    assert error[0].verification == "import-failed"
    assert error[0].metadata["reason"] == "SyntaxError: boom"


def test_default_adapters_instantiate_every_channel():
    adapters = default_adapters()
    assert set(adapters) == set(ALL_ADAPTERS)
    assert all(isinstance(adapter, MessagingAdapter) for adapter in adapters.values())


# --------------------------------------------------------------------- admission


def test_internal_surfaces_are_always_admitted():
    admission = ChannelAdmission()
    for channel in ("cli", "api", "dashboard", "local"):
        decision = admission.decide(channel, "local-user")
        assert decision.allowed and decision.source.startswith("internal:")


def test_external_channel_without_allowlist_fails_closed():
    admission = ChannelAdmission()
    decision = admission.decide("telegram", "alice")
    assert isinstance(decision, AdmissionDecision)
    assert decision.allowed is False
    assert "no allowlist" in decision.reason
    assert decision.source == "default:allowlist"


def test_allowlist_admits_members_and_wildcard():
    admission = ChannelAdmission(
        ChannelsConfig(policies={"slack": ChannelPolicyConfig(allow=["alice", "bob"])})
    )
    assert admission.decide("slack", "alice").allowed
    assert not admission.decide("slack", "eve").allowed
    wildcard = ChannelAdmission(ChannelsConfig(policies={"slack": ChannelPolicyConfig(allow=["*"])}))
    assert wildcard.decide("slack", "anyone").allowed
    # Other channels still fail closed with the default policy.
    assert not wildcard.decide("telegram", "alice").allowed


def test_explicit_deny_mode():
    admission = ChannelAdmission(
        ChannelsConfig(policies={"discord": ChannelPolicyConfig(mode="deny", allow=["*"])})
    )
    decision = admission.decide("discord", "owner")
    assert decision.allowed is False and decision.source == "channel-policy:discord"


def test_from_settings_reads_policy():
    settings = Settings(
        channels=ChannelsConfig(policies={"telegram": ChannelPolicyConfig(mode="open")})
    )
    admission = ChannelAdmission.from_settings(settings)
    assert admission.decide("telegram", "anyone").allowed
    assert not admission.decide("whatsapp", "anyone").allowed


def test_invalid_policy_mode_is_rejected_at_config_load():
    with pytest.raises(ValueError):
        ChannelPolicyConfig(mode="sometimes")


async def test_gateway_default_admission_denies_unknown_sender():
    from leo.events import EventBus

    class AllowAll:
        async def authorize(self, actor, action, resource=None, risk=None, **context):
            from leo.contracts import Decision, RiskLevel

            return Decision(allowed=True, reason="test", risk=risk or RiskLevel.LOW)

        async def resolve_approval(self, approval_id, approved, resolver="user"):
            from leo.contracts import Decision, RiskLevel

            return Decision(allowed=approved, reason="test", risk=RiskLevel.HIGH)

    gateway = Gateway({"telegram": ALL_ADAPTERS["telegram"]()}, events=EventBus(persist=False), permissions=AllowAll())
    with pytest.raises(GatewayAdmissionError, match="no allowlist"):
        await gateway.receive(NormalizedMessage(channel="telegram", sender="stranger", text="hi"))
    # Nothing was persisted for a denied sender.
    assert await gateway.history() == []

    admitted = Gateway(
        {"telegram": ALL_ADAPTERS["telegram"]()},
        events=EventBus(persist=False),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(policies={"telegram": ChannelPolicyConfig(allow=["alice"])})),
    )
    routed = await admitted.receive(NormalizedMessage(channel="telegram", sender="alice", text="hi"))
    assert routed["text"] == "hi"
    with pytest.raises(GatewayAdmissionError, match="not in the allowlist"):
        await admitted.receive(NormalizedMessage(channel="telegram", sender="eve", text="no"))


async def test_gateway_poll_rejects_non_polling_channel():
    from leo.events import EventBus

    class AllowAll:
        async def authorize(self, actor, action, resource=None, risk=None, **context):
            from leo.contracts import Decision, RiskLevel

            return Decision(allowed=True, reason="test", risk=risk or RiskLevel.LOW)

        async def resolve_approval(self, approval_id, approved, resolver="user"):
            from leo.contracts import Decision, RiskLevel

            return Decision(allowed=approved, reason="test", risk=RiskLevel.HIGH)

    gateway = Gateway(
        {"telegram": ALL_ADAPTERS["telegram"]()},
        events=EventBus(persist=False),
        permissions=AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    with pytest.raises(GatewayError, match="does not support polling"):
        await gateway.poll("telegram")


# --------------------------------------------------------------------- sync


async def test_sync_upserts_channel_records_with_honest_status(monkeypatch):
    for cls in ALL_ADAPTERS.values():
        for name in cls.credential_env + cls.verification_env:
            monkeypatch.delenv(name, raising=False)
    first = await sync_integration_records()
    assert first["synced"] == len(ALL_ADAPTERS)
    rows = {row["name"]: dict(row) for row in await db.fetchall("SELECT * FROM integrations")}
    assert "telegram" in rows
    assert rows["telegram"]["status"] == "blocked"
    assert "TELEGRAM_BOT_TOKEN" in rows["telegram"]["config_ref"]

    # With the token present the same row flips to configured; upsert, not duplicate.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-for-status-only")
    await sync_integration_records()
    rows = {row["name"]: dict(row) for row in await db.fetchall("SELECT * FROM integrations")}
    assert rows["telegram"]["status"] == "configured"
    assert len(rows) == len(ALL_ADAPTERS)
    assert "fake-token-for-status-only" not in str(rows)


# --------------------------------------------------------------------- API routes


@pytest.fixture
async def offline_app(monkeypatch: pytest.MonkeyPatch) -> LeoApp:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    settings.channels = ChannelsConfig(
        policies={"telegram": ChannelPolicyConfig(mode="open")}
    )
    app = LeoApp(settings)
    await app.initialize()
    monkeypatch.setattr(api_module, "_app", app)
    yield app
    await app.shutdown()


async def test_api_webhook_routes_are_dynamic_and_fail_closed(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unknown = await client.post("/webhooks/nosuchchannel", json={})
        assert unknown.status_code == 404

        payload = {"message": {"message_id": 5, "from": {"username": "zoe"}, "text": "webhook"}}
        accepted = await client.post(
            "/webhooks/telegram",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "tg-secret"},
        )
        assert accepted.status_code == 200 and accepted.json()["messages"] == 1

        bad = await client.post(
            "/webhooks/telegram",
            json=payload,
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )
        assert bad.status_code == 401

        unsigned = await client.post("/webhooks/telegram", json=payload)
        assert unsigned.status_code == 401

    # The admitted webhook was persisted through the gateway.
    history = await offline_app.gateway.history(channel="telegram")
    assert history and history[0]["content"] == "webhook"


async def test_api_webhook_echoes_signed_subscription_challenge(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "slack-secret")
    raw = json.dumps({"type": "url_verification", "challenge": "challenge-token-123"}).encode()
    timestamp = str(int(time.time()))
    signature = "v0=" + hmac.new(
        b"slack-secret", b"v0:" + timestamp.encode() + b":" + raw, hashlib.sha256
    ).hexdigest()
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        accepted = await client.post(
            "/webhooks/slack",
            content=raw,
            headers={"X-Slack-Signature": signature, "X-Slack-Request-Timestamp": timestamp},
        )
        assert accepted.status_code == 200
        assert accepted.json() == {"challenge": "challenge-token-123"}

        rejected = await client.post(
            "/webhooks/slack",
            content=raw,
            headers={"X-Slack-Signature": "v0=" + "00" * 32, "X-Slack-Request-Timestamp": timestamp},
        )
        assert rejected.status_code == 401


async def test_api_poll_route_rejects_non_polling_channel(offline_app: LeoApp) -> None:
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/channels/nosuchchannel/poll")
        assert missing.status_code == 404
        telegram = await client.post("/channels/telegram/poll")
        assert telegram.status_code == 404


async def test_app_start_syncs_channels_into_connected_surfaces(
    offline_app: LeoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    for cls in ALL_ADAPTERS.values():
        for name in cls.credential_env + cls.verification_env:
            monkeypatch.delenv(name, raising=False)
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/integrations")
    rows = {row["name"]: row for row in listed.json()["integrations"]}
    assert {"telegram", "discord", "whatsapp"} <= set(rows)
    assert rows["telegram"]["status"] == "blocked"
    assert rows["telegram"]["kind"] == "messaging"
    assert all(isinstance(record, dict) for record in rows.values())
