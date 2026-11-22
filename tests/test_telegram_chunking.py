"""Long answers must arrive, not vanish.

Telegram rejects any message body over 4096 characters with a 400, and the adapter
sent one message per answer. A long analysis therefore failed delivery silently
from the user's side: the turn completed, the queue said `done`, and nothing
arrived in the chat.

These tests pin the split — natural boundaries first, no part over the limit, and
nothing lost when the answer is many screenfuls long.
"""

from __future__ import annotations

import pytest

from leo.integrations.messaging import (
    MAX_MESSAGE_CHUNKS,
    TELEGRAM_MAX_CHARS,
    TelegramAdapter,
    split_message,
)


def test_a_short_message_is_one_part():
    assert split_message("hello") == ["hello"]


def test_a_long_message_is_split_under_the_limit():
    text = ("paragraph one is here. " * 200 + "\n\n") * 3

    parts = split_message(text)

    assert len(parts) > 1
    assert all(len(part) <= TELEGRAM_MAX_CHARS for part in parts)


def test_paragraphs_are_preferred_over_hard_cuts():
    first = "a" * 2500
    second = "b" * 2500
    parts = split_message(f"{first}\n\n{second}")

    assert parts[0].startswith("a")
    assert parts[1].startswith("b"), "the split should land on the paragraph break"


def test_no_content_is_lost():
    text = "\n\n".join(f"block {index} " + "x" * 900 for index in range(8))

    parts = split_message(text)

    assert "block 0" in parts[0]
    assert any("block 7" in part for part in parts)


def test_an_absurd_message_is_capped_and_says_so():
    text = "y" * (TELEGRAM_MAX_CHARS * (MAX_MESSAGE_CHUNKS + 5))

    parts = split_message(text)

    assert len(parts) == MAX_MESSAGE_CHUNKS + 1
    assert parts[-1].endswith("(truncated)")


class _Resp:
    def __init__(self, ok: bool = True) -> None:
        self.status_code = 200
        self._ok = ok

    def json(self) -> dict:
        return {"ok": self._ok, "result": {"message_id": 1}}


class _Client:
    """Records what would have hit the Telegram API."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def post(self, url, json=None, timeout=None):
        self.payloads.append(dict(json or {}))
        return _Resp()


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")


async def test_the_adapter_sends_every_part():
    client = _Client()
    adapter = TelegramAdapter(client=client)
    text = "z" * (TELEGRAM_MAX_CHARS * 3)

    await adapter.send("100000001", text)

    assert len(client.payloads) == 3
    assert all(len(payload["text"]) <= TELEGRAM_MAX_CHARS for payload in client.payloads)
    assert "".join(payload["text"] for payload in client.payloads).count("z") >= len(text) - 10
    assert client.payloads[0]["chat_id"] == "100000001"


async def test_a_short_answer_is_still_one_send():
    client = _Client()
    adapter = TelegramAdapter(client=client)

    await adapter.send("100000001", "ok")

    assert len(client.payloads) == 1
    assert client.payloads[0]["text"] == "ok"
