"""Runtime channel-enrollment API contract tests (fully offline).

The enrollment functions themselves are exercised through the HTTP surface
only. No delivery, polling or webhook call is ever made, and a submitted
credential VALUE must never reappear in any response body.
"""

from __future__ import annotations

import pytest
import httpx

from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.core import LeoApp
from leo.interfaces import api as api_module

FAKE_TOKEN = "123456:fake-DO-NOT-LEAK-telegram-token-9f8a7b6c5d4e3f2a1b"


@pytest.fixture
async def offline_app(monkeypatch: pytest.MonkeyPatch, tmp_path) -> LeoApp:
    # An absolute, isolated data directory. conftest.py's autouse fixture has
    # already pointed the DB at its own tmp path, and LeoApp.initialize()
    # respects that configured path, so state stays isolated per test.
    settings = Settings()
    settings.paths.data = (tmp_path / "data").resolve()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    app = LeoApp(settings)
    await app.initialize()
    monkeypatch.setattr(api_module, "_app", app)
    yield app
    await app.shutdown()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_module.api), base_url="http://test"
    )


def _assert_no_secret(response: httpx.Response) -> None:
    assert FAKE_TOKEN not in response.text, "credential value leaked into a response body"


def _required_credential_names(entry: dict) -> list[str]:
    """Discover the required credential field names from a channel detail."""
    for key in ("fields", "credentials", "required_fields", "config_fields"):
        value = entry.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            names = [
                field.get("name")
                for field in value
                if field.get("required", False)
                and field.get("name") not in ("allowlist", "mode")
                and field.get("name")
            ]
            if names:
                return names
    # Fallback to telegram's known required credential fields.
    return ["TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET"]


def _channel_row(rows: list[dict], name: str) -> dict:
    return next(row for row in rows if row.get("name") == name)


async def test_catalog_lists_all_channels_without_values(offline_app: LeoApp) -> None:
    async with _client() as client:
        response = await client.get("/integrations/catalog")
        assert response.status_code == 200, response.text
        channels = response.json()["channels"]
        assert len(channels) == 22
        for channel in channels:
            assert isinstance(channel, dict)
            assert channel.get("name") or channel.get("channel")
        # Nothing has been enrolled yet: no credential values in the payload.
        _assert_no_secret(response)


async def test_catalog_unknown_channel_is_404(offline_app: LeoApp) -> None:
    async with _client() as client:
        response = await client.get("/integrations/catalog/no_such_channel")
        assert response.status_code == 404, response.text
        _assert_no_secret(response)


async def test_enable_verify_disable_telegram_flow(offline_app: LeoApp) -> None:
    async with _client() as client:
        # Single-channel detail from the catalog.
        detail = await client.get("/integrations/catalog/telegram")
        assert detail.status_code == 200, detail.text
        _assert_no_secret(detail)

        # Before enrollment the channel is not ready and names its missing
        # credential field(s) without exposing any value.
        before = await client.post("/integrations/telegram/verify")
        assert before.status_code == 200, before.text
        body = before.json()
        assert body["ready"] is False
        assert body["missing"]
        assert {"channel", "ready", "missing", "webhook", "poll", "policy"} <= body.keys()
        assert body["channel"] == "telegram"
        _assert_no_secret(before)

        # Enroll with fake values for every required credential field plus an
        # admission allowlist. Only field *names* and fake values are sent.
        names = _required_credential_names(detail.json())
        enabled = await client.post(
            "/integrations/telegram/enable",
            json={
                "values": {name: FAKE_TOKEN for name in names},
                "allowlist": ["user-123", "chat-456"],
                "mode": "allowlist",
            },
        )
        assert enabled.status_code == 200, enabled.text
        _assert_no_secret(enabled)

        # Dashboard integration row now reports the honest configured status.
        integrations = await client.get("/integrations")
        assert integrations.status_code == 200, integrations.text
        assert _channel_row(integrations.json()["integrations"], "telegram")["status"] == "configured"
        _assert_no_secret(integrations)

        # The catalog carries the channel's admission policy and stays
        # value-free even after enrollment.
        catalog = await client.get("/integrations/catalog")
        assert catalog.status_code == 200, catalog.text
        entry = next(
            c for c in catalog.json()["channels"]
            if c.get("channel") == "telegram" or c.get("name") == "telegram"
        )
        assert "policy" in entry
        assert entry["status"] == "configured"
        _assert_no_secret(catalog)

        # After enrollment the offline check reports ready with no missing
        # fields — still without ever contacting the provider.
        after = await client.post("/integrations/telegram/verify")
        assert after.status_code == 200, after.text
        assert after.json()["ready"] is True
        assert after.json()["missing"] == []
        _assert_no_secret(after)

        # Disabling returns the channel to the blocked status.
        disabled = await client.post("/integrations/telegram/disable")
        assert disabled.status_code == 200, disabled.text
        _assert_no_secret(disabled)
        integrations = await client.get("/integrations")
        assert _channel_row(integrations.json()["integrations"], "telegram")["status"] == "blocked"
        _assert_no_secret(integrations)

        # Sanity: the verify endpoint now reports the channel not ready again.
        after_disable = await client.post("/integrations/telegram/verify")
        assert after_disable.status_code == 200, after_disable.text
        assert after_disable.json()["ready"] is False
        _assert_no_secret(after_disable)


async def test_enable_unknown_channel_is_client_error(offline_app: LeoApp) -> None:
    async with _client() as client:
        response = await client.post(
            "/integrations/no_such_channel/enable", json={"values": {"X": FAKE_TOKEN}}
        )
        assert response.status_code in (400, 404), response.text
        _assert_no_secret(response)
