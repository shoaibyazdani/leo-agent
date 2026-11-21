"""Telegram photos must reach the vision path, not arrive as empty messages.

The adapter only parses text/caption, so a caption-less photo used to enter the
gateway as "". These tests pin the three steps that fix it: the descriptor is
extracted (largest size), the gateway stores it through the app's sink, and the
turn receives the stored id.
"""

from __future__ import annotations

from leo.config import ChannelsConfig
from leo.events import EventBus
from leo.integrations.admission import ChannelAdmission
from leo.integrations.gateway import Gateway
from leo.integrations.messaging import TelegramAdapter


class _AllowAll:
    async def authorize(self, actor, action, resource=None, risk=None, **context):
        from leo.contracts import Decision, RiskLevel

        return Decision(allowed=True, reason="test", risk=risk or RiskLevel.LOW)

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        from leo.contracts import Decision, RiskLevel

        return Decision(allowed=approved, reason="test", risk=RiskLevel.HIGH)


def _photo_update(*, caption: str | None = None) -> dict:
    message: dict = {
        "message_id": 77,
        "date": 1763700000,
        "from": {"id": 100000001, "username": "operator"},
        "chat": {"id": 100000001, "type": "private"},
        "photo": [
            {"file_id": "small", "file_size": 1000, "width": 90},
            {"file_id": "large", "file_size": 90000, "width": 1024},
        ],
    }
    if caption is not None:
        message["caption"] = caption
    return {"update_id": 1, "message": message}


async def test_photo_descriptor_uses_the_largest_size(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    adapter = TelegramAdapter()

    messages = adapter._messages_from_updates([_photo_update()])

    assert len(messages) == 1
    assert messages[0].text == ""
    assert [a["file_id"] for a in messages[0].attachments] == ["large"]
    assert messages[0].attachments[0]["mime"] == "image/jpeg"


async def test_photo_caption_becomes_text(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    adapter = TelegramAdapter()

    messages = adapter._messages_from_updates([_photo_update(caption="what is this?")])

    assert messages[0].text == "what is this?"
    assert messages[0].attachments, "caption must not drop the photo"


async def test_gateway_stores_attachments_and_passes_the_id():
    stored: list[dict] = []

    async def sink(channel: str, descriptor: dict) -> str:
        stored.append({"channel": channel, "descriptor": descriptor})
        return "att_123"

    gateway = Gateway(
        {},
        events=EventBus(persist=False),
        permissions=_AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    gateway.set_attachment_sink(sink)
    captured: list[dict] = []

    async def handler(payload: dict) -> dict:
        captured.append(payload)
        return {"status": "sent", "success": True, "delivered": True}

    gateway.set_reply_handler(handler)
    await gateway.receive(
        {
            "channel": "telegram",
            "sender": "operator",
            "sender_id": "100000001",
            "conversation_ref": "100000001",
            "message_id": "77",
            "text": "",
            "attachments": [{"kind": "photo", "file_id": "large", "mime": "image/jpeg"}],
        }
    )
    await gateway.drain_outbox(limit=5)

    assert stored and stored[0]["descriptor"]["file_id"] == "large"
    assert captured and captured[0]["attachments"] == ["att_123"]
