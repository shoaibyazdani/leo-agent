"""An agent must not answer its own messages.

Two ways LEO's own words came back as a human turn: the platform handing back a
bot message, and the approval notifier writing into the same chat. Both made it
reply to itself, which the operator sees as the agent mixing things up.
"""

from __future__ import annotations

import pytest

from leo.integrations.messaging import TelegramAdapter


def _update(text: str, *, is_bot: bool = False, message_id: int = 7) -> dict:
    return {"update_id": 1, "message": {
        "message_id": message_id, "date": 1,
        "from": {"id": 8354818246, "is_bot": is_bot, "username": "termxxsbot"},
        "chat": {"id": 525083981, "type": "private"},
        "text": text,
    }}


@pytest.fixture()
def adapter(monkeypatch) -> TelegramAdapter:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    return TelegramAdapter()


def test_its_own_bot_message_is_not_a_turn(adapter) -> None:
    assert adapter._messages_from_updates([_update("I am the agent", is_bot=True)]) == []


def test_an_approval_notice_is_not_a_turn(adapter) -> None:
    notice = "🔐 Approval needed\naction: tool:patch\nrisk: high\nid: apr_x"
    assert adapter._messages_from_updates([_update(notice)]) == []


def test_a_human_message_still_arrives(adapter) -> None:
    messages = adapter._messages_from_updates([_update("check the logs")])
    assert len(messages) == 1
    assert messages[0].text == "check the logs"
