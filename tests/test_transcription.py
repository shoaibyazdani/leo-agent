"""Voice transcription transport: LEO's gateway is reached on its OpenAI surface.

The gateway exposes Anthropic Messages at ``/anthropic`` and OpenAI Chat
Completions at the sibling root. Audio data URLs are only accepted by the
OpenAI surface, so the configured ``anthropic-compatible`` provider must be
mapped there instead of being called as ``<base>/v1/chat/completions`` (which
404s) or as a hardcoded IDE endpoint.
"""

from __future__ import annotations

import pytest

from leo.config import ModelConfig, ModelProviderConfig, Settings
import leo.integrations.transcription as transcription


def _settings(kind: str, base_url: str) -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default="gw",
        providers={
            "gw": ModelProviderConfig(
                kind=kind,
                base_url=base_url,
                model="gemini-4-argon",
                api_key_env="TEST_STT_TOKEN",
                extra_headers={"User-Agent": "leo-agent-os/0.1.0"},
                extra_header_env={"x-opencode-session": "TEST_STT_SESSION"},
            )
        },
    )
    return settings


def _pin_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setattr(
        Settings, "load", classmethod(lambda cls, path=None: settings)
    )


def test_anthropic_provider_maps_to_the_sibling_openai_root(monkeypatch) -> None:
    monkeypatch.setenv("TEST_STT_TOKEN", "tok")
    monkeypatch.setenv("TEST_STT_SESSION", "sess")
    _pin_settings(
        monkeypatch,
        _settings(
            "anthropic-compatible",
            "https://opencode.ai/inference/go/anthropic",
        ),
    )

    endpoint, token, headers = transcription._transport()

    assert endpoint == "https://opencode.ai/inference/go"
    assert token == "tok"
    assert headers["x-opencode-session"] == "sess"
    assert headers["User-Agent"] == "leo-agent-os/0.1.0"
    assert headers["Authorization"] == "Bearer tok"
    assert (
        transcription._chat_url(endpoint)
        == "https://opencode.ai/inference/go/v1/chat/completions"
    )


def test_openai_provider_root_does_not_double_v1(monkeypatch) -> None:
    monkeypatch.setenv("TEST_STT_TOKEN", "tok")
    monkeypatch.setenv("TEST_STT_SESSION", "sess")
    _pin_settings(
        monkeypatch, _settings("openai-compatible", "https://gw.example/v1")
    )

    endpoint, _, _ = transcription._transport()

    assert endpoint == "https://gw.example/v1"
    assert transcription._chat_url(endpoint) == "https://gw.example/v1/chat/completions"


def test_unknown_anthropic_gateway_is_not_guessed(monkeypatch) -> None:
    monkeypatch.setenv("TEST_STT_TOKEN", "tok")
    monkeypatch.setenv("TEST_STT_SESSION", "sess")
    _pin_settings(
        monkeypatch, _settings("anthropic-compatible", "https://api.example.test")
    )

    endpoint, token, _ = transcription._transport()

    assert endpoint is None
    assert token == "tok"


def test_voice_transcription_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv(transcription.STT_ENABLED_ENV, raising=False)
    assert transcription.voice_transcription_enabled() is False


async def test_disabled_transcription_never_resolves_a_provider(monkeypatch) -> None:
    """A voice note must not spend tokens when the operator left STT off."""
    monkeypatch.delenv(transcription.STT_ENABLED_ENV, raising=False)
    calls: list[bool] = []

    def transport():
        calls.append(True)
        return ("https://gw.example", "tok", {})

    monkeypatch.setattr(transcription, "_transport", transport)

    assert await transcription.transcribe_audio(b"x", "audio/ogg", "v.ogg") is None
    assert calls == []


async def test_no_provider_or_token_skips_transcription(monkeypatch) -> None:
    monkeypatch.setenv(transcription.STT_ENABLED_ENV, "1")
    _pin_settings(monkeypatch, Settings())
    monkeypatch.delenv("TEST_STT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    endpoint, token, _ = transcription._transport()
    assert endpoint is None and token is None
    assert await transcription.transcribe_audio(b"x", "audio/ogg", "v.ogg") is None


class _Response:
    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, response: _Response | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict, dict]] = []

    async def post(self, url: str, json=None, headers=None) -> _Response:
        self.calls.append((url, dict(json or {}), dict(headers or {})))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


async def test_transcribe_audio_uses_chat_completions_and_extracts_text(monkeypatch) -> None:
    monkeypatch.setenv(transcription.STT_ENABLED_ENV, "1")
    monkeypatch.setattr(
        transcription,
        "_transport",
        lambda: ("https://gw.example", "tok", {"Authorization": "Bearer tok"}),
    )
    monkeypatch.delenv(transcription.AUDIO_BLOCK_ENV, raising=False)
    client = _Client(
        _Response(payload={"choices": [{"message": {"content": " hello "}}]})
    )

    text = await transcription.transcribe_audio(
        b"audio-bytes", "audio/wav", "voice.wav", client=client
    )

    assert text == "hello"
    url, payload, headers = client.calls[0]
    assert url == "https://gw.example/v1/chat/completions"
    assert payload["model"] == transcription.DEFAULT_MODEL
    block = payload["messages"][0]["content"][1]
    assert block["type"] == "input_audio"
    assert block["input_audio"]["format"] == "wav"
    assert block["input_audio"]["data"] == "YXVkaW8tYnl0ZXM="
    assert headers["Authorization"] == "Bearer tok"


def test_default_model_is_the_cheap_audio_model() -> None:
    """Never default to a premium multimodal model for a voice note."""
    assert transcription.DEFAULT_MODEL == "mimo-v2.6-flash"


async def test_image_url_audio_block_is_opt_in(monkeypatch) -> None:
    """The legacy data-URL shape is available, but only when asked for."""
    monkeypatch.setenv(transcription.STT_ENABLED_ENV, "1")
    monkeypatch.setenv(transcription.AUDIO_BLOCK_ENV, "image_url")
    monkeypatch.setattr(
        transcription,
        "_transport",
        lambda: ("https://gw.example", "tok", {}),
    )
    client = _Client(
        _Response(payload={"choices": [{"message": {"content": "ok"}}]})
    )

    await transcription.transcribe_audio(
        b"audio-bytes", "audio/ogg", "voice.ogg", client=client
    )

    block = client.calls[0][1]["messages"][0]["content"][1]
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:audio/ogg;base64,")


async def test_transcribe_audio_failures_return_none(monkeypatch) -> None:
    monkeypatch.setenv(transcription.STT_ENABLED_ENV, "1")
    monkeypatch.setattr(
        transcription,
        "_transport",
        lambda: ("https://gw.example", "tok", {}),
    )

    rejected = _Client(_Response(status=400))
    assert await transcription.transcribe_audio(
        b"x", "audio/ogg", "v.ogg", client=rejected
    ) is None

    broken = _Client(RuntimeError("connection dropped"))
    assert await transcription.transcribe_audio(
        b"x", "audio/ogg", "v.ogg", client=broken
    ) is None
