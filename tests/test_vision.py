"""Images are read by a separate vision model (Hermes' auxiliary vision router).

The defect this pins: LEO's chat model is chosen for reasoning and is text-only on
this box, so an attached photo reached it as pixels and the whole request failed
upstream — the operator's photo was silently lost. The image now goes to a provider
that declares ``vision``, and the turn receives a description it can reason about.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings, VisionConfig
from leo.contracts import ModelResponse, RiskLevel, Usage
from leo.models.vision import VisionDescriber

IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()},
}


def _settings(*, default: str = "deepseek", vision_provider: str | None = None, mode: str = "describe") -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default=default,
        fallback=default,
        providers={
            "deepseek": ModelProviderConfig(
                model="deepseek-v4.1-flash",
                capabilities=["text", "streaming", "tools", "structured", "usage"],
            ),
            "glm-5.3-flash": ModelProviderConfig(
                model="glm-5.3-flash",
                capabilities=["text", "streaming", "tools", "structured", "usage", "vision"],
            ),
        },
        vision=VisionConfig(mode=mode, provider=vision_provider),
    )
    return settings


class _Router:
    def __init__(self, *, text: str = "A screenshot of the LEO dashboard, 3 errors.", fail: bool = False):
        self.text = text
        self.fail = fail
        self.calls: list[tuple[object, str | None]] = []

    async def complete(self, request: Any, *, slot: str | None = None):  # mirrors ModelRouter
        self.calls.append((request, slot))
        if self.fail:
            raise RuntimeError("upstream said no")
        return ModelResponse(content=self.text, model="glm-5.3-flash", provider="opencode-go", usage=Usage())


def test_the_vision_slot_is_its_own_provider() -> None:
    describer = VisionDescriber(_settings(), _Router())
    assert describer.slot() == "glm-5.3-flash", "the text-only chat default must not read images"
    assert describer.handles([IMAGE_BLOCK]) is True


def test_a_configured_vision_provider_wins() -> None:
    describer = VisionDescriber(_settings(vision_provider="glm-5.3-flash"), _Router())
    assert describer.slot() == "glm-5.3-flash"


def test_off_and_native_modes_leave_the_pixels_alone() -> None:
    assert VisionDescriber(_settings(mode="off"), _Router()).handles([IMAGE_BLOCK]) is False
    assert VisionDescriber(_settings(mode="native"), _Router()).handles([IMAGE_BLOCK]) is False


def test_auto_mode_defers_to_a_vision_capable_chat_model() -> None:
    settings = _settings(default="glm-5.3-flash", mode="auto")
    describer = VisionDescriber(settings, _Router())
    # the chat model itself declares vision, so the image attaches natively
    assert describer.handles([IMAGE_BLOCK], main_slot="glm-5.3-flash") is False
    assert describer.handles([IMAGE_BLOCK], main_slot="deepseek") is True


@pytest.mark.asyncio
async def test_describe_sends_the_image_to_the_vision_slot_and_returns_text() -> None:
    router = _Router(text="Chart titled 'Revenue'. Bars: 3, 5, 8.")
    describer = VisionDescriber(_settings(), router)

    text = await describer.describe([IMAGE_BLOCK])

    assert text == "Chart titled 'Revenue'. Bars: 3, 5, 8."
    request, slot = router.calls[0]
    assert slot == "glm-5.3-flash"
    assert request.task_category == "vision"
    assert request.messages[-1].images == [IMAGE_BLOCK]


@pytest.mark.asyncio
async def test_a_failing_reader_returns_none_instead_of_raising() -> None:
    describer = VisionDescriber(_settings(), _Router(fail=True))
    assert await describer.describe([IMAGE_BLOCK]) is None


def test_vision_defaults_to_describe_and_rejects_an_unknown_mode() -> None:
    assert VisionConfig().mode == "describe"
    with pytest.raises(ValueError):
        VisionConfig(mode="sideways")


def test_the_describe_tool_is_low_risk_and_registered() -> None:
    from leo.tools.registry import ToolRegistry
    from leo.tools.vision_tools import register

    registry = ToolRegistry()
    names = register(registry)
    assert names == ["vision_analyze"]
    spec = next(s for s in registry.list_specs() if s.name == "vision_analyze")
    assert spec.risk is RiskLevel.LOW
    assert "path" in spec.input_schema["properties"]


@pytest.mark.asyncio
async def test_the_tool_reads_a_local_png_and_refuses_a_renamed_text_file(tmp_path) -> None:
    from leo.tools.vision_tools import vision_analyze

    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    fake = tmp_path / "notes.png"
    fake.write_text("this is not an image")

    class _Describer:
        def slot(self):
            return "glm-5.3-flash"

        async def describe(self, blocks, *, question=None):
            assert len(blocks) == 1
            return f"read {question}" if question else "read"

    ctx = {"services": {"vision": _Describer()}}
    ok = await vision_analyze({"path": str(png), "question": "what is it?"}, ctx)
    assert ok == {"ok": True, "image": "shot.png", "description": "read what is it?"}

    with pytest.raises(ValueError, match="not image/png"):
        await vision_analyze({"path": str(fake)}, ctx)


def test_config_carries_the_vision_block() -> None:
    settings = Settings()
    assert settings.models.vision.mode == "describe"
    assert settings.models.vision.provider is None
    assert settings.models.vision.max_description_chars >= 1000


@pytest.mark.asyncio
async def test_a_photo_turn_carries_the_description_not_the_pixels() -> None:
    """A text-only chat model must never receive the image block."""
    from leo.core.app import LeoApp

    class _App:
        settings = _settings()
        vision = VisionDescriber(_settings(), _Router(text="A whiteboard with 4 bullets."))
        _images_for_turn = LeoApp._images_for_turn

    prompt, images = await _App()._images_for_turn("what does this say?", [IMAGE_BLOCK])

    assert images == [], "pixels must not reach the chat model"
    assert "A whiteboard with 4 bullets." in prompt
    assert prompt.startswith("what does this say?")


@pytest.mark.asyncio
async def test_an_unreadable_image_is_a_sentence_not_a_dead_turn() -> None:
    from leo.core.app import LeoApp

    class _App:
        settings = _settings()
        vision = VisionDescriber(_settings(), _Router(fail=True))
        _images_for_turn = LeoApp._images_for_turn

    prompt, images = await _App()._images_for_turn("what is this?", [IMAGE_BLOCK])

    assert images == []
    assert "could not read it" in prompt


def _wire_messages(kind: str):
    """Build a real provider for that wire and return its first chat message."""
    from leo.config import ModelProviderConfig
    from leo.contracts import ChatMessage, ModelRequest
    from leo.models.provider import build_provider

    config = ModelProviderConfig(kind=kind, model="glm-5.3-flash", base_url="https://example.invalid")
    provider = build_provider("glm-5.3-flash", config)
    request = ModelRequest(
        messages=[ChatMessage(role="user", content="what is this?", images=[IMAGE_BLOCK])],
    )
    built = provider._messages(request)
    return built[0] if isinstance(built, list) else built[1][0]


def test_the_openai_wire_actually_sends_the_image() -> None:
    """Images were dropped entirely on the OpenAI wire, so the model saw no picture."""
    message = _wire_messages("openai-compatible")

    parts = message["content"]
    assert isinstance(parts, list), "content must become parts when an image is present"
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[-1] == {"type": "text", "text": "what is this?"}


def test_the_anthropic_wire_keeps_its_native_block() -> None:
    message = _wire_messages("anthropic-compatible")

    parts = message["content"]
    assert parts[0] == IMAGE_BLOCK, "the stored block IS the Anthropic shape"


def test_the_openai_wire_translates_the_stored_image_block() -> None:
    """The live bug: an Anthropic-shaped block in an OpenAI content list is ignored,
    so the vision model answers "please upload the image" while the call succeeds."""
    from leo.models.provider import _openai_image_part

    part = _openai_image_part(IMAGE_BLOCK)

    assert part is not None
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    # an already-correct part passes through untouched
    assert _openai_image_part(part) == part


def test_a_url_image_and_a_broken_block_are_handled() -> None:
    from leo.models.provider import _openai_image_part

    url_block = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
    assert _openai_image_part(url_block)["image_url"]["url"] == "https://x/y.png"
    assert _openai_image_part({"type": "image", "source": {}}) is None
    assert _openai_image_part({"type": "image", "source": {"type": "base64"}}) is None
