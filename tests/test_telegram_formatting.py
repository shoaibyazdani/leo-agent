"""Agent replies are markdown; the channel must render it, and must not lose it.

Reported from the phone as "garbage": every ``**bold**`` and ``` `code` ``` marker
arrived as literal characters, because the adapter sent plain text with no parse
mode. Telegram has a parse mode for exactly this, but it rejects text whose
entities do not balance - so the mode needs a plain-text fallback rather than a
dropped message.
"""

from __future__ import annotations

import pytest

from leo.integrations.messaging import (
    TELEGRAM_PARSE_MODE,
    TelegramAdapter,
    split_message,
)


class _Resp:
    def __init__(self, status_code: int = 200, description: str = "") -> None:
        self.status_code = status_code
        self._description = description

    def json(self) -> dict:
        if self._description:
            return {"ok": False, "description": self._description}
        return {"ok": self.status_code < 400, "result": {"message_id": 1}}


class _Client:
    """Records payloads; can reject the first formatted attempt."""

    def __init__(self, first_status: int = 200, first_description: str = "") -> None:
        self.payloads: list[dict] = []
        self._first_status = first_status
        self._first_description = first_description

    async def post(self, url, json=None, timeout=None):
        self.payloads.append(dict(json or {}))
        if len(self.payloads) == 1 and self._first_status != 200:
            return _Resp(self._first_status, self._first_description)
        return _Resp()


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")


async def test_markdown_is_requested_so_markers_render():
    client = _Client()
    adapter = TelegramAdapter(client=client)

    await adapter.send("100000001", "**bold** and `code`")

    assert client.payloads[0]["parse_mode"] == TELEGRAM_PARSE_MODE == "Markdown"


async def test_unbalanced_entities_fall_back_to_plain_text():
    """A 400 must degrade to readable text, never to a lost message."""
    client = _Client(first_status=400,
                     first_description="Bad Request: can't parse entities: unexpected end")
    adapter = TelegramAdapter(client=client)

    result = await adapter.send("100000001", "**unclosed bold")

    assert len(client.payloads) == 2, "the adapter should retry without parse_mode"
    assert "parse_mode" in client.payloads[0]
    assert "parse_mode" not in client.payloads[1]
    assert client.payloads[1]["text"] == "**unclosed bold"
    assert result["success"] is True


async def test_a_real_failure_is_still_an_error():
    client = _Client(first_status=400, first_description="Bad Request: chat not found")
    adapter = TelegramAdapter(client=client)
    from leo.integrations.messaging import MessagingError

    with pytest.raises(MessagingError):
        await adapter.send("100000001", "hello")


def test_a_code_block_is_never_cut_in_half():
    body = "print(1)\n" * 500           # well past one message
    text = "```python\n" + body

    parts = split_message(text)

    assert len(parts) > 1
    for part in parts:
        assert part.count("```") % 2 == 0, "each part must carry balanced fences"
    assert parts[1].startswith("```")


def test_balanced_content_is_left_alone():
    text = "```\ncode\n```\n\n" + "plain " * 10

    parts = split_message(text)

    assert parts[0].count("```") % 2 == 0
