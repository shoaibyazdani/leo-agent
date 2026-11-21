"""Offline checks for the Telegram poll transport.

No live network: getUpdates goes through httpx.MockTransport. The claims under
test are the two that matter for an exposed bot:

1. fail closed — no allow-list means no messages leave the adapter;
2. the allow-list is enforced on the sender id, and the cursor advances so the
   same update is never fetched twice.
"""

from __future__ import annotations

import httpx

from leo.integrations.messaging import TelegramAdapter


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


async def test_poll_fails_closed_without_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    calls: list[dict] = []
    adapter = TelegramAdapter(client=_client(calls, [_update(1, 525083981, "hi")]))

    assert await adapter.poll() == []
    # Nothing is even fetched: the adapter refuses before calling Telegram.
    assert calls == []


async def test_poll_keeps_allowlisted_sender_and_drops_strangers(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "525083981")
    calls: list[dict] = []
    adapter = TelegramAdapter(
        client=_client(calls, [_update(11, 525083981, "owner"), _update(12, 999, "stranger")])
    )

    messages = await adapter.poll()

    assert [m.text for m in messages] == ["owner"]
    assert messages[0].sender_id == "525083981"
    # Cursor advanced past BOTH updates, including the dropped one, so the
    # stranger's message is not retried forever.
    assert adapter._cursor == 13


async def test_poll_sends_offset_on_the_next_call(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "525083981")
    calls: list[dict] = []
    adapter = TelegramAdapter(client=_client(calls, [_update(5, 525083981, "one")]))

    await adapter.poll()
    await adapter.poll()

    assert "offset" not in calls[0]
    assert calls[1]["offset"] == "6"
