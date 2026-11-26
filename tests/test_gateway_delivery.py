"""Gateway delivery regressions: durable outbox, atomic admission, stable identity.

Every test is offline: fake adapters and an injected permission stub, backed by
the isolated SQLite database from conftest.py.
"""

from __future__ import annotations

import json
import time

import pytest

from leo import db
from leo.config import ChannelPolicyConfig, ChannelsConfig, Settings
from leo.contracts import Decision, RiskLevel, utcnow
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.integrations.admission import (
    ADMISSION_DENIED_TOPIC,
    POLICY_KV_KEY,
    POLICY_REFRESHED_TOPIC,
    ChannelAdmission,
    PolicyVersionError,
)
from leo.integrations.channels import (
    ALL_ADAPTERS,
    polling_channels,
    verification_report,
    webhook_channels,
)
from leo.integrations.channels.specs import FieldKind, channel_specs
from leo.integrations.gateway import (
    GATEWAY_MESSAGE_TOPIC,
    Gateway,
    GatewayAdmissionError,
    GatewayError,
)
from leo.integrations.messaging import (
    DiscordAdapter,
    MessagingAdapter,
    NormalizedMessage,
    TelegramAdapter,
    WebhookVerificationError,
    WhatsAppAdapter,
    discord_timestamp_is_fresh,
    hmac_hex,
    implements_webhook,
    is_discord_ping,
)


class AllowAll:
    async def authorize(self, actor, action, resource=None, risk=None, **context):
        return Decision(allowed=True, reason="test", risk=risk or RiskLevel.LOW)

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        return Decision(allowed=approved, reason="test", risk=risk or RiskLevel.HIGH)


class FakeAdapter(MessagingAdapter):
    """Scripted adapter with both real ingress methods overridden."""

    supports_webhook = True
    supports_polling = True

    def __init__(self, channel: str = "fake", messages=None) -> None:
        super().__init__()
        self.channel = channel
        self.messages = list(messages or [])
        self.sent: list[tuple[str, str]] = []

    def verify_webhook(self, payload, headers=None, *, raw_body=None) -> bool:
        return True

    async def handle_webhook(self, payload, headers=None, *, raw_body=None):
        return list(self.messages)

    async def poll(self):
        return list(self.messages)

    async def send(self, target, text):
        self.sent.append((str(target), str(text)))
        return {"channel": self.channel, "target": str(target), "success": True, "message_id": "s1"}


def make_gateway(*, adapters=None, admission=None, bus=None, **kwargs) -> Gateway:
    return Gateway(
        adapters or {},
        events=bus or EventBus(persist=True),
        permissions=AllowAll(),
        admission=admission or ChannelAdmission(ChannelsConfig(default_mode="open")),
        **kwargs,
    )


def msg(**overrides) -> NormalizedMessage:
    values = {
        "channel": "fake", "sender": "alice", "sender_id": "u1",
        "conversation_ref": "chat-1", "text": "hello", "message_id": "m-1",
    }
    values.update(overrides)
    return NormalizedMessage(**values)


# ------------------------------------------------------------- 1. outbox / delivery


async def test_outbox_is_durable_across_restart_and_drain_sends_reply():
    gateway = make_gateway()
    routed = await gateway.receive(msg())
    assert routed["outbox_id"] and routed["reply_target"] == "chat-1"
    pending = await gateway.pending_outbox()
    assert len(pending) == 1
    assert pending[0]["reply_target"] == "chat-1" and pending[0]["text"] == "hello"
    assert (await gateway.outbox_stats())["pending"] == 1

    # Simulate a process restart: a brand new gateway sees the persisted row.
    adapter = FakeAdapter("fake")
    restarted = make_gateway(adapters={"fake": adapter}, bus=EventBus(persist=False))
    seen: list[dict] = []

    async def reply(payload: dict) -> dict:
        seen.append(payload)
        return await restarted.send(
            payload["channel"], payload["reply_target"], "reply:" + payload["text"],
            subject=payload["subject"],
        )

    restarted.set_reply_handler(reply)
    summary = await restarted.drain_outbox()
    assert summary == {"claimed": 1, "processed": 1, "suppressed": 0, "failed": 0, "remaining": 0}
    assert seen[0]["conversation_id"] == routed["conversation_id"]
    assert adapter.sent == [("chat-1", "reply:hello")]
    assert (await restarted.outbox_stats())["processed"] == 1
    assert (await restarted.drain_outbox())["claimed"] == 0
    assert [item["content"] for item in await restarted.history()] == ["hello", "reply:hello"]


async def test_drain_requires_handler_and_retries_turn_failures():
    gateway = make_gateway()
    await gateway.receive(msg(message_id="m-retry", text="retry me"))
    with pytest.raises(GatewayError, match="no outbox reply handler"):
        await gateway.drain_outbox()

    state = {"fail": True}

    async def handler(payload: dict) -> dict:
        if state["fail"]:
            raise RuntimeError("turn exploded")
        return {"ok": True}

    gateway.set_reply_handler(handler)
    failed = await gateway.drain_outbox(max_attempts=2)
    assert failed["failed"] == 1 and failed["remaining"] == 1
    row = (await gateway.pending_outbox())[0]
    assert row["attempts"] == 1 and "turn exploded" in row["last_error"]
    state["fail"] = False
    assert (await gateway.drain_outbox(max_attempts=2))["processed"] == 1
    assert (await gateway.outbox_stats())["processed"] == 1


# ------------------------------------------- 2. atomic admission / idempotency / pinning


async def test_webhook_batch_preflight_commits_nothing_when_one_sender_is_denied():
    bus = EventBus(persist=True)
    denied = msg(sender="eve", sender_id="u2", message_id="m-2")
    adapter = FakeAdapter("fake", messages=[msg(message_id="m-1"), denied])
    admission = ChannelAdmission(
        ChannelsConfig(policies={"fake": ChannelPolicyConfig(allow=["u1"])}), events=bus
    )
    gateway = make_gateway(adapters={"fake": adapter}, admission=admission, bus=bus)
    denials: list = []
    bus.subscribe(ADMISSION_DENIED_TOPIC, denials.append)

    with pytest.raises(GatewayAdmissionError, match="not in the allowlist"):
        await gateway.receive_webhook("fake", {"updates": [1, 2]}, {})
    assert await gateway.history() == []            # not even the admitted sibling
    assert (await gateway.pending_outbox()) == []
    assert (await gateway.outbox_stats())["pending"] == 0
    assert len(denials) == 1
    assert denials[0].payload["policy_source"] == "channel-policy:fake"
    assert denials[0].payload["reason"]
    audit = await db.fetchone(
        "SELECT reason, metadata FROM audit_log WHERE action=?", (ADMISSION_DENIED_TOPIC,)
    )
    assert audit is not None and "channel-policy:fake" in audit["reason"]
    assert json.loads(audit["metadata"])["policy_source"] == "channel-policy:fake"

    # A genuine failure released the replay claim, so the provider can retry.
    adapter.messages = [msg(message_id="m-1")]
    routed = await gateway.receive_webhook("fake", {"updates": [1, 2]}, {})
    assert len(routed) == 1 and len(await gateway.history()) == 1


async def test_webhook_route_channel_is_authoritative_over_adapter_claim():
    bus = EventBus(persist=True)
    adapter = FakeAdapter("fake", messages=[msg(channel="api", sender_id="u1")])
    admission = ChannelAdmission(
        ChannelsConfig(policies={"fake": ChannelPolicyConfig(allow=["u1"])}), events=bus
    )
    gateway = make_gateway(adapters={"fake": adapter}, admission=admission, bus=bus)
    routed = (await gateway.receive_webhook("fake", {"spoof": True}, {}))[0]
    assert routed["channel"] == "fake" and routed["metadata"]["adapter_channel"] == "api"
    assert len(await gateway.sessions(channel="fake")) == 1
    assert await gateway.sessions(channel="api") == []

    # An internal-looking claim cannot escape an external channel's deny policy.
    adapter.messages = [msg(channel="dashboard", sender_id="u1", message_id="m-spoof")]
    strict = ChannelAdmission(
        ChannelsConfig(policies={"fake": ChannelPolicyConfig(mode="deny")}), events=bus
    )
    strict_gateway = make_gateway(adapters={"fake": adapter}, admission=strict, bus=bus)
    with pytest.raises(GatewayAdmissionError, match="denied by policy"):
        await strict_gateway.receive_webhook("fake", {"spoof": True}, {})


async def test_message_id_dedupe_is_durable_across_webhook_and_poll_paths():
    adapter = FakeAdapter("fake", messages=[msg()])
    gateway = make_gateway(adapters={"fake": adapter})
    first = (await gateway.receive_webhook("fake", {"delivery": 1}, {}))[0]
    assert first["duplicate"] is False
    polled = (await gateway.poll("fake"))[0]
    assert polled["duplicate"] is True and polled["message_id"] == first["message_id"]
    directly = await gateway.receive(msg())
    assert directly["duplicate"] is True
    assert len(await gateway.history()) == 1
    assert len(await gateway.pending_outbox()) == 1


# ------------------------------------------------------------- 3. stable identity fields


async def test_core_adapters_populate_stable_ids_and_gateway_routes_by_them(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    payload = {
        "message": {
            "message_id": 42, "date": 1700000000,
            "from": {"id": 7, "username": "renamed-user"},
            "chat": {"id": -10055}, "message_thread_id": 99,
            "text": "hello group",
        }
    }
    telegram = (await TelegramAdapter().handle_webhook(
        payload, {"x-telegram-bot-api-secret-token": "tg-secret"}
    ))[0]
    assert (telegram.sender_id, telegram.sender) == ("7", "renamed-user")
    assert telegram.conversation_ref == "-10055" and telegram.thread == "99"

    # Renaming the display name does not change admission or the session.
    admission = ChannelAdmission(
        ChannelsConfig(policies={"telegram": ChannelPolicyConfig(allow=["7"])})
    )
    gateway = make_gateway(admission=admission)
    routed = await gateway.receive(telegram)
    assert routed["subject"] == "-10055"             # group context, not sender
    assert routed["reply_target"] == "-10055"
    assert routed["metadata"]["sender"] == "renamed-user"
    assert routed["metadata"]["sender_id"] == "7"
    assert (await gateway.sessions(channel="telegram"))[0]["subject"] == "-10055"

    monkeypatch.setenv("WHATSAPP_APP_SECRET", "wa-secret")
    wa_payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "timestamp": "1700000000",
                    "contacts": [{"wa_id": "1555", "name": "Bob"}],
                    "messages": [{"id": "w1", "from": "1555", "text": {"body": "hi"}}],
                }
            }]
        }]
    }
    signature = hmac_hex("wa-secret", json.dumps(wa_payload, separators=(",", ":"), ensure_ascii=False).encode())
    whatsapp = (await WhatsAppAdapter().handle_webhook(
        wa_payload, {"x-hub-signature-256": signature}
    ))[0]
    assert (whatsapp.sender_id, whatsapp.sender, whatsapp.conversation_ref) == ("1555", "Bob", "1555")

    monkeypatch.setenv("DISCORD_PUBLIC_KEY", "ab" * 32)
    discord = (await DiscordAdapter(signature_verifier=lambda *args: True).handle_webhook(
        {
            "type": 2, "id": "int-1", "channel_id": "chan-9",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"name": "ask", "options": [{"value": "hi"}]},
            "user": {"id": "77", "username": "carol"},
        },
        {"x-signature-ed25519": "cd" * 32, "x-signature-timestamp": "1700000000"},
    ))[0]
    assert (discord.sender_id, discord.sender, discord.conversation_ref) == ("77", "carol", "chan-9")


# ------------------------------------- 4. Discord freshness / PING / honest capabilities


async def test_discord_interaction_freshness_and_ping_detection(monkeypatch):
    monkeypatch.setenv("DISCORD_PUBLIC_KEY", "ab" * 32)
    adapter = DiscordAdapter()
    payload = {"type": 2, "id": "int-1", "data": {"name": "ask"}, "user": {"id": "1"}}
    stale = {"x-signature-ed25519": "cd" * 32, "x-signature-timestamp": str(int(time.time()) - 3600)}
    with pytest.raises(WebhookVerificationError, match="freshness window"):
        await adapter.handle_webhook(payload, stale)
    # A fresh timestamp passes the freshness gate (crypto is optional/offline).
    fresh = {"x-signature-ed25519": "cd" * 32, "x-signature-timestamp": str(int(time.time()))}
    with pytest.raises(WebhookVerificationError, match="Ed25519 verification unavailable"):
        await adapter.handle_webhook(payload, fresh)
    assert discord_timestamp_is_fresh(time.time()) is True
    assert discord_timestamp_is_fresh(time.time() - 3600) is False
    assert discord_timestamp_is_fresh("garbage") is False

    assert is_discord_ping({"type": 1}) and is_discord_ping({"type": "1"})
    assert not is_discord_ping({"type": True}) and not is_discord_ping({"type": 2})
    ping_adapter = DiscordAdapter(signature_verifier=lambda *args: True)
    assert await ping_adapter.handle_webhook({"type": "1"}, fresh) == []


async def test_webhook_capability_requires_real_implementation(monkeypatch):
    class DeclarationOnly(MessagingAdapter):
        channel = "telegram"          # overwrites a real channel for this test
        supports_webhook = True
        supports_polling = False

        def verify_webhook(self, payload, headers=None, *, raw_body=None) -> bool:
            return True

    assert implements_webhook(DeclarationOnly) is False
    monkeypatch.setitem(ALL_ADAPTERS, "telegram", DeclarationOnly)
    assert "telegram" not in webhook_channels()
    assert "webhook" not in channel_specs()["telegram"].capabilities
    assert "webhook" not in DeclarationOnly().record().capabilities


async def test_missing_verification_material_reports_degraded(monkeypatch):
    monkeypatch.delenv("DISCORD_PUBLIC_KEY", raising=False)
    health = next(item for item in verification_report() if item["channel"] == "discord")
    assert health["status"] == "degraded"
    assert "DISCORD_PUBLIC_KEY" in health["missing_verification"]

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-bot-token")
    record = DiscordAdapter().record()
    assert record.metadata["verification_status"] == "degraded"
    assert record.metadata["missing_verification"] == ["DISCORD_PUBLIC_KEY"]

    monkeypatch.setenv("DISCORD_PUBLIC_KEY", "ab" * 32)
    ready = next(item for item in verification_report() if item["channel"] == "discord")
    assert ready["status"] == "ready" and ready["missing_verification"] == []


# -------------------------------------- 5. structured denials / versioned refresh


async def test_admission_denial_audit_and_versioned_policy_refresh():
    bus = EventBus(persist=True)
    admission = ChannelAdmission(
        ChannelsConfig(policies={"fake": ChannelPolicyConfig(allow=["u1"])}), events=bus
    )
    gateway = make_gateway(adapters={"fake": FakeAdapter("fake")}, admission=admission, bus=bus)
    with pytest.raises(GatewayAdmissionError, match="not in the allowlist"):
        await gateway.receive(msg(sender="eve", sender_id="u2", message_id="m-denied"))
    denied = await bus.history(ADMISSION_DENIED_TOPIC)
    assert len(denied) == 1
    assert denied[0].payload["policy_source"] == "channel-policy:fake"
    assert "not in the allowlist" in denied[0].payload["reason"]
    audit = await db.fetchone(
        "SELECT reason, metadata FROM audit_log WHERE action=?", (ADMISSION_DENIED_TOPIC,)
    )
    assert audit is not None and json.loads(audit["metadata"])["policy_source"] == "channel-policy:fake"

    version0 = await gateway.policy_version()
    admission.set_policy("fake", ChannelPolicyConfig(allow=["u1", "u3"]))
    version1 = await admission.save_policies()
    assert version1 > version0 and await gateway.policy_version() == version1

    # Simulate another process rewriting the durable policy row.
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (POLICY_KV_KEY, json.dumps({"fake": {"mode": "deny", "allow": []}}), utcnow()),
    )
    refreshed = await gateway.refresh_policies(expected_version=version1)
    assert refreshed["version"] == version1 + 1
    assert refreshed["policies"]["fake"]["mode"] == "deny"
    assert admission.decide("fake", "u1").allowed is False
    events = await bus.history(POLICY_REFRESHED_TOPIC)
    assert events and events[0].payload["version"] == version1 + 1
    with pytest.raises(PolicyVersionError, match="version mismatch"):
        await gateway.refresh_policies(expected_version=version1)


async def test_stale_processing_outbox_rows_are_requeued():
    gateway = make_gateway()
    await gateway.receive(msg(message_id="m-stale", text="crashed worker"))
    await db.execute(
        "UPDATE gateway_outbox SET status='processing', updated_at='2000-01-01T00:00:00+00:00'"
    )
    assert (await gateway.outbox_stats())["processing"] == 1
    assert await gateway.recover_stale_outbox(older_than_seconds=60) == 1
    stats = await gateway.outbox_stats()
    assert stats["processing"] == 0 and stats["pending"] == 1

    async def reply(payload: dict) -> dict:
        return {"channel": payload["channel"], "ok": True}

    gateway.set_reply_handler(reply)
    assert (await gateway.drain_outbox())["processed"] == 1


async def test_send_forwards_an_inline_keyboard_to_a_capable_adapter():
    """The /model picker's buttons ride the same send as its text."""

    class KeyboardAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__(channel="keyboard")
            self.markup: dict | None = None

        async def send(self, target, text, *, reply_markup=None):
            self.markup = reply_markup
            return await super().send(target, text)

    adapter = KeyboardAdapter()
    gateway = make_gateway(adapters={"keyboard": adapter})
    keyboard = {"inline_keyboard": [[{"text": "beam-free", "callback_data": "mdl:beam-free"}]]}

    result = await gateway.send("keyboard", "chat-1", "pick one", reply_markup=keyboard)

    assert result["success"] is True
    assert adapter.markup == keyboard


async def test_a_keyboard_never_breaks_an_adapter_that_cannot_render_one():
    """Older adapters take (target, text) only: the text must still arrive."""
    adapter = FakeAdapter()
    gateway = make_gateway(adapters={"fake": adapter})
    keyboard = {"inline_keyboard": [[{"text": "x", "callback_data": "mdl:x"}]]}

    result = await gateway.send("fake", "chat-1", "pick one", reply_markup=keyboard)

    assert result["success"] is True
    assert adapter.sent == [("chat-1", "pick one")]


def test_channel_specs_expose_tls_and_jwt_settings_with_honest_capabilities():
    specs = channel_specs()
    feishu = specs["feishu"].description
    assert "HMAC-SHA256" in feishu and "AES-256-CBC" in feishu
    assert "SHA1" not in feishu

    teams = {field.name: field for field in specs["msteams"].fields}
    for name in ("TEAMS_APP_ID", "TEAMS_JWT_ISSUER"):
        assert name in teams and teams[name].required is False
        assert teams[name].kind is FieldKind.TEXT
    assert "TEAMS_PUBLIC_KEY" in teams

    irc = {field.name: field for field in specs["irc"].fields}
    for name in ("IRC_TLS", "IRC_ALLOW_INSECURE"):
        assert name in irc and irc[name].required is False
        assert irc[name].kind is FieldKind.CHOICE
    assert irc["IRC_TLS"].default == "1" and irc["IRC_ALLOW_INSECURE"].default == "0"
    assert irc["IRC_PORT"].required is False

    assert specs["msteams"].capabilities == ["send", "webhook"]
    assert specs["irc"].capabilities == ["send", "poll"]
    assert "irc" not in webhook_channels() and "msteams" in webhook_channels()
    assert "msteams" not in polling_channels() and "irc" in polling_channels()

    # Every adapter-declared env name (including optional_env) appears once.
    for channel, cls in ALL_ADAPTERS.items():
        names = [field.name for field in specs[channel].fields]
        for attr in ("credential_env", "verification_env", "poll_env", "optional_env"):
            for env in getattr(cls, attr, ()) or ():
                assert names.count(env) == 1, f"{channel}.{env} declared but missing or duplicated"


async def test_gateway_message_event_still_carries_admitted_payload():
    bus = EventBus(persist=True)
    seen: list = []
    bus.subscribe(GATEWAY_MESSAGE_TOPIC, seen.append)
    gateway = make_gateway(bus=bus)
    routed = await gateway.receive(msg(text="subscriber sees me"))
    assert len(seen) == 1
    assert seen[0].payload["message_id"] == routed["message_id"]
    assert seen[0].payload["outbox_id"] == routed["outbox_id"]


# ------------------------------- 6. approval argument binding / send idempotency


async def test_send_needs_no_approval_even_when_high_risk_requires_it():
    settings = Settings()
    settings.security.high_risk_requires_approval = True
    permissions = PermissionManagerImpl(settings, events=EventBus(persist=False))
    adapter = FakeAdapter("fake")
    gateway = Gateway(
        {"fake": adapter}, events=EventBus(persist=False), permissions=permissions,
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )

    # Outbound sends are MEDIUM: they are audited but never create a tap,
    # even with high_risk_requires_approval enabled.
    sent = await gateway.send("fake", "chat-42", "hello", subject="group-1")
    assert sent["success"] is True
    assert len(adapter.sent) == 1
    assert len(await permissions.pending_approvals()) == 0


async def test_send_replays_idempotent_response_without_second_adapter_call():
    adapter = FakeAdapter("fake")
    gateway = make_gateway(adapters={"fake": adapter})
    first = await gateway.send("fake", "chat-1", "hello", idempotency_key="key-1")
    second = await gateway.send("fake", "chat-1", "hello", idempotency_key="key-1")
    assert first["duplicate"] is False and second["duplicate"] is True
    assert second["message_id"] == first["message_id"]
    assert len(adapter.sent) == 1
    assert len(await gateway.history()) == 1
    assert (await gateway.send("fake", "chat-2", "hello", idempotency_key="key-2"))["duplicate"] is False
    assert len(adapter.sent) == 2


async def test_outbox_suppresses_duplicate_send_when_bookkeeping_fails(monkeypatch):
    adapter = FakeAdapter("fake")
    gateway = make_gateway(adapters={"fake": adapter})
    await gateway.receive(msg(message_id="m-idem", text="reply me"))
    handler_keys: list[str] = []

    async def reply(payload: dict) -> dict:
        handler_keys.append(payload["idempotency_key"])
        return await gateway.send(
            payload["channel"], payload["reply_target"], "answer",
            subject=payload["subject"], idempotency_key=payload["idempotency_key"],
        )

    gateway.set_reply_handler(reply)
    original_finish = gateway._finish_outbox

    async def failing_finish(outbox_id, status, *, error=None, result=None):
        if status == "processed":
            raise RuntimeError("bookkeeping DB failure")
        return await original_finish(outbox_id, status, error=error, result=result)

    # The adapter accepts the send, then the processed-status write fails.
    monkeypatch.setattr(gateway, "_finish_outbox", failing_finish)
    first = await gateway.drain_outbox()
    assert first["failed"] == 1 and first["processed"] == 0
    assert len(handler_keys) == 1 and len(adapter.sent) == 1
    assert (await gateway.outbox_stats())["processing"] == 1

    # Recovery requeues the row; the durable marker must suppress the send.
    monkeypatch.setattr(gateway, "_finish_outbox", original_finish)
    await db.execute("UPDATE gateway_outbox SET status='pending', updated_at=?", (utcnow(),))
    second = await gateway.drain_outbox()
    assert second["suppressed"] == 1 and second["processed"] == 1
    assert len(handler_keys) == 1 and len(adapter.sent) == 1
    assert (await gateway.outbox_stats())["processed"] == 1


async def test_outbox_unconfirmed_handler_result_stays_retryable():
    gateway = make_gateway()
    await gateway.receive(msg(message_id="m-unconfirmed", text="unconfirmed"))
    async def unconfirmed(payload: dict) -> dict:
        return {"success": False, "error": "adapter rejected"}

    gateway.set_reply_handler(unconfirmed)
    result = await gateway.drain_outbox()
    assert result["failed"] == 1 and result["remaining"] == 1
    row = (await gateway.pending_outbox())[0]
    assert row["attempts"] == 1 and "did not confirm delivery" in row["last_error"]


async def test_interrupt_commands_are_handled_at_admission_not_in_the_outbox():
    """A /stop must not wait behind the serial outbox for the turn it targets.

    Inbound messages drain one turn at a time; an interrupt-class command is
    routed out of band at admission, exactly like Hermes's gateway /stop.
    """
    gateway = make_gateway()
    seen: list[tuple[str, str]] = []

    async def handler(conversation_id: str, text: str) -> None:
        seen.append((conversation_id, text))

    gateway.set_interrupt_handler(lambda text: text.strip() == "/stop", handler)
    admitted = await gateway.receive(msg(text="/stop"))

    assert seen == [(admitted["conversation_id"], "/stop")]
    # The admitted row still exists, so the user receives the acknowledgement.
    row = await db.fetchone(
        "SELECT text FROM gateway_outbox WHERE conversation_id=?",
        (admitted["conversation_id"],),
    )
    assert row is not None and row["text"] == "/stop"


# ------------------------------------------------- 7. Phase 8 local/none delivery


async def test_local_note_delivery_needs_no_adapter() -> None:
    """A ``local`` target is a durable assistant note, not an external send."""
    from types import SimpleNamespace

    from leo.core.app import LeoApp
    from leo.schedules.domain import DeliveryTarget

    app = object.__new__(LeoApp)

    async def _terminal(*a, **k):
        return None

    app.task_reporting = SimpleNamespace(on_terminal=_terminal)
    task = SimpleNamespace(
        id="task-local", objective="local brief", status="completed",
        metadata={"schedule_id": "sch_local", "schedule": "local brief"},
    )
    result = await app._channel_delivery(
        task, "local", "local-conv",
        {
            "status": "completed",
            "delivery_lane": "result",
            "schedule_payload": {"name": "local brief", "text": "LOCAL-ONLY-NOTE"},
            "delivery_target": DeliveryTarget(
                kind="local", platform="local", chat_id="local-conv"
            ).model_dump(mode="json"),
        },
    )
    assert result == "local:note:local-conv"
    row = await db.fetchone(
        "SELECT content,metadata FROM messages WHERE conversation_id='local-conv'"
    )
    assert row is not None and "LOCAL-ONLY-NOTE" in row["content"]
    metadata = json.loads(row["metadata"])
    assert metadata["schedule_id"] == "sch_local"
    assert metadata["schedule_note"] is True


async def test_external_delivery_refuses_a_none_address() -> None:
    """A corrupt target must fail before any adapter sees the string None."""
    from types import SimpleNamespace

    from leo.core.app import LeoApp

    app = object.__new__(LeoApp)

    async def _terminal(*a, **k):
        return None

    app.task_reporting = SimpleNamespace(on_terminal=_terminal)
    task = SimpleNamespace(id="task-none", objective="x", status="completed", metadata={})
    with pytest.raises(ValueError, match="no delivery address"):
        await app._channel_delivery(
            task, "telegram", None, {"status": "completed"}
        )
