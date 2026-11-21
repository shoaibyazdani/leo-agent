"""Channel enrollment: broker credentials, admission policy, honest statuses.

All values are fake and every assertion checks that they stay out of returned
details, events and raw credential rows. No network is used.
"""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.config import ChannelPolicyConfig, ChannelsConfig
from leo.events import EventBus
from leo.integrations.admission import ChannelAdmission
from leo.integrations.channels import ALL_ADAPTERS, default_adapters
from leo.integrations.gateway import Gateway
from leo.integrations.secrets import (
    MASTER_KEY_ENV,
    CredentialStore,
    CredentialStoreError,
    default_store,
)

from leo.integrations.enrollment import (
    EnrollmentError,
    channel_details,
    disable_channel,
    enable_channel,
    list_channels,
)

FAKE_TOKEN = "fake-telegram-token-9f4c1a"
FAKE_SECRET = "fake-telegram-webhook-secret-7d3e9b"
FAKE_SLACK = "fake-slack-token-2b8e7d"
TELEGRAM_VALUES = {
    "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
    "TELEGRAM_WEBHOOK_SECRET": FAKE_SECRET,
}


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Gateway:
    """Real gateway with an isolated, fail-closed admission and offline bus."""
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET", MASTER_KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    default_store.clear()
    return Gateway(
        default_adapters(),
        events=EventBus(persist=False),
        admission=ChannelAdmission(ChannelsConfig()),
    )


async def test_enable_telegram_and_sync(monkeypatch: pytest.MonkeyPatch, gateway: Gateway) -> None:
    before = await channel_details("telegram", gateway=gateway)
    assert before["status"] == "blocked"

    seen: list = []

    async def capture(event) -> None:
        seen.append(event)

    gateway.events.subscribe("channel.enabled", capture)

    details = await enable_channel(
        "telegram",
        dict(TELEGRAM_VALUES),
        allowlist=[" alice ", "", "bob"],
        gateway=gateway,
    )

    assert details["channel"] == "telegram"
    assert details["status"] == "configured"
    assert details["policy"] == {"mode": "allowlist", "allow": ["alice", "bob"]}
    assert gateway.admission.snapshot()["telegram"] == details["policy"]
    assert details["mode_note"] is None
    assert details["webhook"] is True
    assert details["capabilities"]

    token_field = next(f for f in details["fields"] if f["name"] == "TELEGRAM_BOT_TOKEN")
    assert token_field["configured"] is True
    assert token_field["required"] is True
    assert token_field["kind"] == "secret"
    # Details (and every other return value) never carry a secret itself.
    assert FAKE_TOKEN not in json.dumps(details)
    assert FAKE_SECRET not in json.dumps(details)

    assert len(seen) == 1
    assert seen[0].topic == "channel.enabled"
    assert seen[0].payload["channel"] == "telegram"

    row = await db.fetchone("SELECT status, config_ref FROM integrations WHERE name=?", ("telegram",))
    assert row is not None
    assert row["status"] == "configured"
    assert "TELEGRAM_BOT_TOKEN" in row["config_ref"]


async def test_enable_validation_errors(gateway: Gateway) -> None:
    with pytest.raises(EnrollmentError, match="TELEGRAM_BOT_TOKEN|Bot Token"):
        await enable_channel("telegram", {}, gateway=gateway)
    with pytest.raises(EnrollmentError, match="unknown channel"):
        await enable_channel("not-a-channel", dict(TELEGRAM_VALUES), gateway=gateway)
    with pytest.raises(EnrollmentError, match="unknown field"):
        await enable_channel(
            "telegram", {**TELEGRAM_VALUES, "MADE_UP_FIELD": "x"}, gateway=gateway
        )
    with pytest.raises(EnrollmentError, match="newlines"):
        await enable_channel(
            "telegram", {**TELEGRAM_VALUES, "TELEGRAM_BOT_TOKEN": FAKE_TOKEN + "\n..."},
            gateway=gateway,
        )

    # None of the rejected calls may leave a credential behind.
    assert default_store.get("TELEGRAM_BOT_TOKEN") is None


async def test_disable_is_scoped_to_one_channel(gateway: Gateway) -> None:
    await default_store.set_async("SLACK_BOT_TOKEN", FAKE_SLACK)
    await enable_channel(
        "telegram", dict(TELEGRAM_VALUES), allowlist=["alice"], gateway=gateway
    )
    assert default_store.get("TELEGRAM_BOT_TOKEN") == FAKE_TOKEN

    seen: list = []

    async def capture(event) -> None:
        seen.append(event)

    gateway.events.subscribe("channel.disabled", capture)

    details = await disable_channel("telegram", gateway=gateway)

    assert default_store.get("TELEGRAM_BOT_TOKEN") is None
    assert default_store.get("SLACK_BOT_TOKEN") == FAKE_SLACK
    # An explicit deny survives in the policy set; removing it would let a
    # permissive YAML base policy revive the disabled channel.
    assert gateway.admission.snapshot()["telegram"] == {"mode": "deny", "allow": []}
    assert details["policy"] == {"mode": "deny", "allow": []}
    assert details["status"] == "blocked"
    assert gateway.admission.decide("telegram", "alice").allowed is False
    assert len(seen) == 1 and seen[0].payload["channel"] == "telegram"

    row = await db.fetchone("SELECT status FROM integrations WHERE name=?", ("telegram",))
    assert row is not None and row["status"] == "blocked"


async def test_encrypted_persistence_and_wrong_key(
    monkeypatch: pytest.MonkeyPatch, gateway: Gateway
) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "right-key")
    await enable_channel("telegram", dict(TELEGRAM_VALUES), gateway=gateway)

    rows = await db.fetchall(
        "SELECT name, salt, nonce, ciphertext, mac FROM credentials ORDER BY name"
    )
    assert {row["name"] for row in rows} == set(TELEGRAM_VALUES)
    blob = b"".join(
        bytes(row[column])
        for row in rows
        for column in ("salt", "nonce", "ciphertext", "mac")
    )
    assert FAKE_TOKEN.encode() not in blob
    assert FAKE_SECRET.encode() not in blob
    assert FAKE_TOKEN not in str([dict(row) for row in rows])
    assert FAKE_SECRET not in str([dict(row) for row in rows])

    fresh = CredentialStore()
    assert await fresh.load() == 2
    assert fresh.get("TELEGRAM_BOT_TOKEN") == FAKE_TOKEN
    assert fresh.get("TELEGRAM_WEBHOOK_SECRET") == FAKE_SECRET

    monkeypatch.setenv(MASTER_KEY_ENV, "wrong-key")
    wrong = CredentialStore()
    assert await wrong.load() == 0
    assert wrong.get("TELEGRAM_BOT_TOKEN") is None


async def test_fail_closed_admission_until_allowlist(gateway: Gateway) -> None:
    await enable_channel(
        "telegram", dict(TELEGRAM_VALUES), allowlist=[], gateway=gateway
    )
    assert gateway.admission.decide("telegram", "anyone").allowed is False
    details = await channel_details("telegram", gateway=gateway)
    assert details["mode_note"] == "inbound denied until allowlist configured"

    await enable_channel(
        "telegram", dict(TELEGRAM_VALUES), allowlist=["alice"], gateway=gateway
    )
    assert gateway.admission.decide("telegram", "alice").allowed is True
    assert gateway.admission.decide("telegram", "bob").allowed is False
    assert (await channel_details("telegram", gateway=gateway))["mode_note"] is None


async def test_discord_either_credential_and_session_only_warning(gateway: Gateway) -> None:
    # The adapter accepts either a bot token or an incoming webhook URL.
    with pytest.raises(EnrollmentError, match="one of"):
        await enable_channel("discord", {}, gateway=gateway)

    details = await enable_channel(
        "discord",
        {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/1/fake"},
        gateway=gateway,
    )
    assert details["status"] == "configured"
    assert details["persistence"] == "session-only"
    assert "lost on restart" in (details["persistence_note"] or "")
    assert "fake" not in json.dumps(details)

    await enable_channel(
        "discord", {"DISCORD_BOT_TOKEN": "fake-bot-token"}, gateway=gateway
    )
    assert (await channel_details("discord", gateway=gateway))["status"] == "configured"


async def test_list_channels_sorted_and_valueless(gateway: Gateway) -> None:
    await enable_channel(
        "telegram", dict(TELEGRAM_VALUES), allowlist=["alice"], gateway=gateway
    )
    channels = await list_channels(gateway=gateway)
    assert [item["channel"] for item in channels] == sorted(ALL_ADAPTERS)
    assert all(FAKE_TOKEN not in json.dumps(item) for item in channels)
    assert all(FAKE_SECRET not in json.dumps(item) for item in channels)


def _permissive_gateway() -> Gateway:
    config = ChannelsConfig(
        policies={"telegram": ChannelPolicyConfig(mode="open", allow=["*"])}
    )
    return Gateway(
        default_adapters(),
        events=EventBus(persist=False),
        admission=ChannelAdmission(config),
    )


async def test_disable_beats_permissive_yaml_and_survives_restart() -> None:
    permissive = _permissive_gateway()
    await enable_channel("telegram", dict(TELEGRAM_VALUES), gateway=permissive)
    assert permissive.admission.decide("telegram", "anyone").allowed is True

    await disable_channel("telegram", gateway=permissive)
    assert permissive.admission.decide("telegram", "anyone").allowed is False

    # A fresh process with the same permissive YAML must load the stored deny.
    restarted = _permissive_gateway().admission
    await restarted.load_policies()
    assert restarted.decide("telegram", "anyone").allowed is False
    assert restarted.snapshot()["telegram"] == {"mode": "deny", "allow": []}


async def test_disable_revokes_environment_sourced_credential(
    monkeypatch: pytest.MonkeyPatch, gateway: Gateway
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-env-token")
    await enable_channel(
        "telegram", {"TELEGRAM_WEBHOOK_SECRET": FAKE_SECRET}, gateway=gateway
    )
    assert default_store.get("TELEGRAM_BOT_TOKEN") == "fake-env-token"

    await disable_channel("telegram", gateway=gateway)
    assert default_store.get("TELEGRAM_BOT_TOKEN") is None
    assert default_store.is_revoked("TELEGRAM_BOT_TOKEN")
    details = await channel_details("telegram", gateway=gateway)
    token = next(field for field in details["fields"] if field["name"] == "TELEGRAM_BOT_TOKEN")
    assert token["configured"] is False


async def test_enable_rolls_back_on_second_credential_failure(
    monkeypatch: pytest.MonkeyPatch, gateway: Gateway
) -> None:
    events: list = []

    async def capture(event) -> None:
        events.append(event)

    gateway.events.subscribe("channel.enabled", capture)
    real_set = default_store.set_async
    calls = 0

    async def flaky_set(name: str, value: str, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CredentialStoreError("encrypted store unavailable")
        return await real_set(name, value, **kwargs)

    monkeypatch.setattr(default_store, "set_async", flaky_set)
    with pytest.raises(EnrollmentError, match="could not store credential"):
        await enable_channel(
            "telegram", dict(TELEGRAM_VALUES), allowlist=["alice"], gateway=gateway
        )

    assert calls == 2
    assert default_store.get("TELEGRAM_BOT_TOKEN") is None
    assert default_store.get("TELEGRAM_WEBHOOK_SECRET") is None
    assert "telegram" not in gateway.admission.snapshot()
    assert events == []
    row = await db.fetchone("SELECT status FROM integrations WHERE name=?", ("telegram",))
    assert row is None or row["status"] == "blocked"


async def test_disable_rolls_back_on_policy_failure(
    monkeypatch: pytest.MonkeyPatch, gateway: Gateway
) -> None:
    await enable_channel(
        "telegram", dict(TELEGRAM_VALUES), allowlist=["alice"], gateway=gateway
    )
    events: list = []

    async def capture(event) -> None:
        events.append(event)

    gateway.events.subscribe("channel.disabled", capture)

    async def boom() -> None:
        raise RuntimeError("policy store unavailable")

    monkeypatch.setattr(gateway.admission, "save_policies", boom)
    with pytest.raises(RuntimeError, match="policy store unavailable"):
        await disable_channel("telegram", gateway=gateway)

    assert default_store.get("TELEGRAM_BOT_TOKEN") == FAKE_TOKEN
    assert default_store.get("TELEGRAM_WEBHOOK_SECRET") == FAKE_SECRET
    assert gateway.admission.snapshot()["telegram"] == {
        "mode": "allowlist", "allow": ["alice"]
    }
    assert events == []
    row = await db.fetchone("SELECT status FROM integrations WHERE name=?", ("telegram",))
    assert row is not None and row["status"] == "configured"


async def test_enable_rolls_back_integration_and_event_on_late_failure(
    monkeypatch: pytest.MonkeyPatch, gateway: Gateway
) -> None:
    import leo.integrations.enrollment as enrollment

    events: list = []

    async def capture(event) -> None:
        events.append(event)

    gateway.events.subscribe("channel.enabled", capture)

    async def boom() -> None:
        raise RuntimeError("integration store unavailable")

    monkeypatch.setattr(enrollment, "sync_integration_records", boom)
    with pytest.raises(RuntimeError, match="integration store unavailable"):
        await enable_channel(
            "telegram", dict(TELEGRAM_VALUES), allowlist=["alice"], gateway=gateway
        )

    assert default_store.get("TELEGRAM_BOT_TOKEN") is None
    assert "telegram" not in gateway.admission.snapshot()
    assert events == []
