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


async def test_voice_note_is_transcribed_into_the_turn(monkeypatch):
    """A caption-less voice note must reach the brain as text, not as "".

    ``leo serve`` polls through ``Gateway.poll``/``receive``, not through the
    split gateway's own poll loop, so transcription has to live in admission or
    the operator's audio messages arrive empty and are never answered.
    """

    async def fake_transcribe(channel, descriptor, adapters, **kwargs):
        assert channel == "telegram"
        assert descriptor["file_id"] == "voice-abc"
        return "yeh meri awaaz hai"

    monkeypatch.setattr(
        "leo.integrations.transcription.transcribe_channel_voice", fake_transcribe
    )
    gateway = Gateway(
        {},
        events=EventBus(persist=False),
        permissions=_AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
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
            "message_id": "78",
            "text": "",
            "attachments": [{"kind": "voice", "file_id": "voice-abc", "mime": "audio/ogg"}],
        }
    )
    await gateway.drain_outbox(limit=5)

    assert captured and captured[0]["text"] == "yeh meri awaaz hai"


async def test_failed_voice_transcription_still_admits_the_message(monkeypatch):
    """A clip that cannot be transcribed is not a dropped message."""

    async def broken(*args, **kwargs):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(
        "leo.integrations.transcription.transcribe_channel_voice", broken
    )
    gateway = Gateway(
        {},
        events=EventBus(persist=False),
        permissions=_AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
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
            "message_id": "79",
            "text": "",
            "attachments": [{"kind": "voice", "file_id": "voice-bad", "mime": "audio/ogg"}],
        }
    )
    await gateway.drain_outbox(limit=5)

    assert captured, "the turn must survive a failed transcription"
    assert captured[0]["text"], "a silent clip must still produce words to answer"


async def test_one_bad_message_does_not_swallow_the_poll_batch(monkeypatch):
    """The poll cursor moves past the whole batch, so a batch abort loses all of it.

    A voice note that fails admission used to raise out of ``receive_batch``
    after the adapter had already advanced its cursor: the note and every text
    message beside it vanished with no trace.
    """

    class _Adapter:
        channel = "telegram"
        credential_env = ("TELEGRAM_BOT_TOKEN",)
        # implements_polling requires the declared capability, not just a
        # method named poll(); without this the gateway refuses the channel.
        supports_polling = True

        async def poll(self, state=None):
            from leo.integrations.messaging import NormalizedMessage

            return [
                NormalizedMessage(
                    channel="telegram", sender="operator", sender_id="1",
                    conversation_ref="1", message_id="poison", text="bad",
                ),
                NormalizedMessage(
                    channel="telegram", sender="operator", sender_id="1",
                    conversation_ref="1", message_id="good", text="good",
                ),
            ]

    gateway = Gateway(
        {"telegram": _Adapter()},
        events=EventBus(persist=False),
        permissions=_AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    real_receive = gateway.receive_batch
    captured: list[dict] = []

    async def handler(payload: dict) -> dict:
        captured.append(payload)
        return {"status": "sent", "success": True, "delivered": True}

    async def flaky_batch(messages, **kwargs):
        if any(getattr(m, "message_id", "") == "poison" for m in messages):
            raise RuntimeError("one bad row")
        return await real_receive(messages, **kwargs)

    gateway.receive_batch = flaky_batch
    gateway.set_reply_handler(handler)
    await gateway.poll("telegram")
    await gateway.drain_outbox(limit=5)

    assert [p["text"] for p in captured] == ["good"]


async def test_gateway_process_stores_descriptors_before_admitting():
    """The polling process must turn descriptors into stored ids itself.

    In a split deployment the worker never polls, so if the gateway admits a
    bare file_id the worker rejects it and the photo becomes an empty turn.
    """
    from leo.gateway.server import store_inbound_attachments

    seen = []

    async def sink(channel, descriptor):
        seen.append((channel, descriptor))
        return f"att_{descriptor['file_id']}"

    ids = await store_inbound_attachments(
        "telegram",
        [{"file_id": "abc", "mime": "image/jpeg"}],
        sink,
    )
    assert ids == ["att_abc"]
    assert seen == [("telegram", {"file_id": "abc", "mime": "image/jpeg"})]


async def test_gateway_process_drops_bad_attachment_without_raising():
    from leo.gateway.server import store_inbound_attachments

    calls = []

    async def sink(channel, descriptor):
        calls.append(descriptor)
        if descriptor.get("file_id") == "bad":
            raise RuntimeError("download failed")
        return "att_ok"

    ids = await store_inbound_attachments(
        "telegram",
        [{"file_id": "bad"}, {"file_id": "good"}],
        sink,
    )
    assert ids == ["att_ok"], "a failed download must not drop the message"
    assert len(calls) == 2


async def test_gateway_process_no_sink_returns_empty():
    from leo.gateway.server import store_inbound_attachments

    assert await store_inbound_attachments("telegram", [{"file_id": "x"}], None) == []


def _voice_update():
    return {
        "update_id": 2,
        "message": {
            "message_id": 78,
            "date": 1763700000,
            "from": {"id": 100000001, "username": "operator"},
            "chat": {"id": 100000001, "type": "private"},
            "voice": {
                "file_id": "voice-abc",
                "mime_type": "audio/ogg",
                "file_size": 42000,
                "duration": 7,
            },
        },
    }


async def test_voice_descriptor_extracted(monkeypatch):
    """A caption-less voice note must produce a voice descriptor, not an empty turn."""
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    adapter = TelegramAdapter()

    messages = adapter._messages_from_updates([_voice_update()])

    assert len(messages) == 1
    assert messages[0].text == ""
    assert len(messages[0].attachments) == 1
    voice = messages[0].attachments[0]
    assert voice["kind"] == "voice"
    assert voice["file_id"] == "voice-abc"
    assert voice["mime"] == "audio/ogg"
    assert voice["name"] == "voice.ogg"


async def test_transcribe_inbound_voice_builds_text(monkeypatch):
    """Voice descriptors become transcribed text appended to the turn."""
    from leo.gateway.server import transcribe_inbound_voice

    async def fake_transcribe(channel, descriptor, adapters, **kwargs):
        assert channel == "telegram"
        assert descriptor["file_id"] == "voice-abc"
        return "yeh meri awaaz hai"

    monkeypatch.setattr(
        "leo.integrations.transcription.transcribe_channel_voice", fake_transcribe
    )
    text = await transcribe_inbound_voice(
        "telegram", [{"kind": "voice", "file_id": "voice-abc"}], {}
    )
    assert text == "yeh meri awaaz hai"


async def test_transcribe_inbound_voice_ignores_photos(monkeypatch):
    from leo.gateway.server import transcribe_inbound_voice

    async def fake_transcribe(*args, **kwargs):
        raise AssertionError("photos must not be transcribed")

    monkeypatch.setattr(
        "leo.integrations.transcription.transcribe_channel_voice", fake_transcribe
    )
    text = await transcribe_inbound_voice(
        "telegram", [{"kind": "photo", "file_id": "pic"}], {}
    )
    assert text == ""


async def test_transcribe_inbound_voice_failure_is_nonfatal(monkeypatch):
    from leo.gateway.server import transcribe_inbound_voice

    async def fake_transcribe(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "leo.integrations.transcription.transcribe_channel_voice", fake_transcribe
    )
    text = await transcribe_inbound_voice(
        "telegram", [{"kind": "voice", "file_id": "bad"}], {}
    )
    assert text == ""


async def test_local_ollama_transcription_absent_token_is_none(monkeypatch):
    """No credentials means no transcription, not an exception."""
    from leo.integrations.transcription import transcribe_audio

    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert await transcribe_audio(b"not-audio", "audio/ogg", "voice.ogg") is None


async def test_audio_format_mapping():
    from leo.integrations.transcription import _audio_format

    assert _audio_format("audio/ogg", "voice.ogg") == "ogg"
    assert _audio_format("", "clip.mp3") == "mp3"
    assert _audio_format("audio/x-m4a", "a.m4a") == "mp4"
    assert _audio_format("", "weird.bin") == "ogg"
