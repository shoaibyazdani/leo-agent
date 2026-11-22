"""A live progress line must not ring the phone.

Telegram notifies on every new message. The reporter sends one per turn, so a long
autonomous run would buzz the phone just for showing its work. ``silent=True`` maps
to ``disable_notification``: the log stays readable, the phone stays quiet — and the
actual answer, sent without it, still notifies.
"""

from __future__ import annotations

import inspect

import pytest

from leo.integrations.messaging import MessagingAdapter


class _Captured:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def post(self, url, json=None, timeout=None):
        self.payloads.append(dict(json or {}))

        class _R:
            status_code = 200

            def json(self):
                return {"ok": True, "result": {"message_id": 42}}

        return _R()


@pytest.mark.asyncio
async def test_a_silent_send_sets_disable_notification(monkeypatch) -> None:
    from leo.integrations.messaging import TelegramAdapter

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    client = _Captured()
    await TelegramAdapter(client=client).send("123", "⏳ Working — 2 steps", silent=True)

    assert client.payloads[-1].get("disable_notification") is True


@pytest.mark.asyncio
async def test_an_answer_is_not_silenced(monkeypatch) -> None:
    from leo.integrations.messaging import TelegramAdapter

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    client = _Captured()
    await TelegramAdapter(client=client).send("123", "your answer")

    assert "disable_notification" not in client.payloads[-1]


def test_the_base_adapter_accepts_silent() -> None:
    """A channel that cannot silence must still match the caller's signature."""
    assert "silent" in inspect.signature(MessagingAdapter.send).parameters
