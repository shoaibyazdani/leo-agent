"""A blank user row must not poison every later request.

Live 2026-11-27: a captionless photo arrived with text="" and an attachment id.
The streaming path stored text verbatim, so the durable history held an empty
user row. The next turn replayed it and Beam-free rejected the whole request:

    Invalid value for 'messages[11].content': a user message must contain an
    image or text other than whitespace.
"""

from __future__ import annotations

import base64

import pytest

from leo.config import ModelProviderConfig
from leo.contracts import ChatMessage, ModelRequest
from leo.models.provider import build_provider

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode()
IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": PNG},
}


def _first_message(kind: str, message: ChatMessage) -> dict:
    provider = build_provider(
        "probe", ModelProviderConfig(kind=kind, model="probe", base_url="https://example.invalid")
    )
    built = provider._messages(ModelRequest(messages=[message]))
    return built[0] if isinstance(built, list) else built[1][0]


@pytest.mark.parametrize("kind", ["openai-compatible", "anthropic-compatible"])
def test_a_whitespace_user_message_becomes_a_placeholder(kind: str) -> None:
    """Old empty rows exist in history today, so the wire must normalize them."""
    message = _first_message(kind, ChatMessage(role="user", content=" \n\t "))
    assert message["content"] == "[empty message]"


@pytest.mark.parametrize("kind", ["openai-compatible", "anthropic-compatible"])
def test_a_captionless_photo_is_not_emptied_by_the_guard(kind: str) -> None:
    """An image makes a blank caption valid; the guard must leave that path alone."""
    message = _first_message(kind, ChatMessage(role="user", content="", images=[IMAGE]))
    assert isinstance(message["content"], list)
    assert len(message["content"]) == 2  # image + text


@pytest.mark.asyncio
async def test_streaming_stores_a_placeholder_for_a_captionless_photo() -> None:
    from leo import db
    from leo.config import ModelConfig, Settings
    from leo.core import LeoApp

    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    app = LeoApp(settings)
    await app.initialize()
    try:
        record = await app.attachments.save("photo.png", "image/png", PNG)
        events = []
        async for event in app.ask_stream("", channel="test", attachments=[record["id"]]):
            events.append(event)
            if event.get("type") == "run.started":
                break  # the row is durable before the model opens; no need to run it
        row = await db.fetchone(
            "SELECT content FROM messages WHERE role='user' ORDER BY rowid DESC LIMIT 1"
        )
        assert row is not None
        assert row["content"] == "[empty message]", "new captionless rows must not be empty"
    finally:
        await app.shutdown()
