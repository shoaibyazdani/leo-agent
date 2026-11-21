"""Offline checks for the Telegram poll transport.

No live network: getUpdates goes through httpx.MockTransport.

Policy is deliberately NOT tested here — ChannelAdmission is the single resolver
for who may reach LEO (tests/test_channels_registry.py covers fail-closed and
allowlist admission). This file only asserts the transport's own job: fetch,
normalize, and never re-fetch an update it already saw.
"""

from __future__ import annotations

import httpx
import pytest

from leo.integrations.messaging import MessagingError, TelegramAdapter


def _client(calls: list[dict], updates: list[dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return httpx.Response(200, json={"ok": True, "result": updates})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _update(update_id: int, user_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id + 100,
            "date": 1700000000,
            "text": text,
            "from": {"id": user_id, "username": f"u{user_id}"},
            "chat": {"id": user_id},
        },
    }


async def test_poll_normalizes_updates_into_messages(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    adapter = TelegramAdapter(
        client=_client([], [_update(11, 100000001, "owner"), _update(12, 999, "stranger")])
    )

    messages = await adapter.poll()

    # Every update is normalized; sender filtering is admission's job.
    assert [m.text for m in messages] == ["owner", "stranger"]
    assert [m.sender_id for m in messages] == ["100000001", "999"]
    assert messages[0].channel == "telegram"


async def test_poll_advances_cursor_past_every_update(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    adapter = TelegramAdapter(
        client=_client([], [_update(11, 100000001, "a"), _update(12, 999, "b")])
    )

    await adapter.poll()

    assert adapter._cursor == 13


async def test_poll_sends_offset_on_the_next_call(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    calls: list[dict] = []
    adapter = TelegramAdapter(client=_client(calls, [_update(5, 100000001, "one")]))

    await adapter.poll()
    await adapter.poll()

    assert "offset" not in calls[0]
    assert calls[1]["offset"] == "6"


async def test_poll_without_token_is_blocked_not_silent(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    adapter = TelegramAdapter(client=_client([], []))

    with pytest.raises(MessagingError, match="TELEGRAM_BOT_TOKEN"):
        await adapter.poll()
