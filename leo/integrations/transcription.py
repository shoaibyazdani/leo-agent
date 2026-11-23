"""Speech-to-text for inbound channel voice notes.

Telegram voice notes arrive as an opaque audio file with no text. This module
turns the downloaded bytes into text through the same model gateway LEO
already uses for chat, so no separate STT service or local model is required.
The audio is sent as the standard OpenAI ``input_audio`` content block, which
the gateway's Chat Completions surface accepts for audio-capable models. A
provider that only accepts the older data-URL workaround (for example a
``gemini`` model) can be selected with ``LEO_STT_AUDIO_BLOCK=image_url``.

LEO's gateway exposes two surfaces on one base: Anthropic Messages at
``/anthropic`` and OpenAI Chat Completions at the gateway root. Audio is only
accepted by the OpenAI surface — the Anthropic path rejects an audio data URL
with "content type image_url requires a native Anthropic Messages target" — so
an ``anthropic-compatible`` provider is mapped to its sibling OpenAI root. No
endpoint is invented: with no configured provider, transcription is skipped.

The default model is ``mimo-v2.6-flash``: it declares ``audio/ogg`` input (the
Telegram voice format) and costs $0.14/$0.28 per million tokens. ``gemini-4-argon``
also works but costs $2/$10 per million, so it is never selected implicitly;
pin it only with ``LEO_STT_MODEL`` and ``LEO_STT_AUDIO_BLOCK=image_url``.

Credentials are read by environment-variable *name* at call time; no token value
is stored or logged. A failed transcription returns ``None`` so the caller can
fall back cleanly rather than dropping the message.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import httpx

_LOG = logging.getLogger(__name__)

# Cheap and audio-capable (ogg/mp3/wav/flac/mp4): $0.14 input / $0.28 output
# per million tokens. A premium multimodal model is opt-in via LEO_STT_MODEL.
DEFAULT_MODEL = "mimo-v2.6-flash"
AUDIO_BLOCK_ENV = "LEO_STT_AUDIO_BLOCK"
DEFAULT_PROMPT = (
    "Transcribe this voice note exactly as spoken. Reply with only the spoken "
    "words - no commentary, no quotes, no translation. Preserve the speaker's "
    "language and writing style (including Roman Urdu if that is what they use)."
)
TIMEOUT_SECONDS = 60.0
_AUDIO_SUFFIXES = {"ogg", "mp3", "wav", "mp4", "m4a", "webm", "flac", "aac"}


def _audio_format(mime: str, name: str) -> str:
    """Map a descriptor mime/name to the data-URL subtype the gateway expects."""
    value = (mime or "").lower()
    if "ogg" in value or "opus" in value:
        return "ogg"
    if "mpeg" in value or "mp3" in value:
        return "mp3"
    if "wav" in value:
        return "wav"
    if "mp4" in value or "m4a" in value:
        return "mp4"
    if "webm" in value:
        return "webm"
    if "flac" in value:
        return "flac"
    suffix = (name or "").rsplit(".", 1)[-1].lower()
    return suffix if suffix in _AUDIO_SUFFIXES else "ogg"


def _openai_root(base_url: str, kind: str) -> str | None:
    """Return the OpenAI-compatible root of a configured provider, or None.

    LEO's inference gateway puts the Anthropic surface under ``/anthropic`` and
    the OpenAI surface at the same root without it. Other providers cannot be
    guessed at safely: returning ``None`` skips transcription instead of
    calling a path that may not exist.
    """
    base = (base_url or "").rstrip("/")
    if not base:
        return None
    value = str(kind or "").lower()
    if value == "openai-compatible":
        return base
    if value == "anthropic-compatible" and base.endswith("/anthropic"):
        return base[: -len("/anthropic")]
    return None


def _chat_url(root: str) -> str:
    """Build the Chat Completions URL without doubling an existing ``/v1``."""
    root = root.rstrip("/")
    return root + ("/chat/completions" if root.endswith("/v1") else "/v1/chat/completions")


def _transport() -> tuple[str | None, str | None, dict[str, str]]:
    """Resolve the OpenAI surface, key and extra headers from the provider.

    Reading the provider config instead of hardcoding env names keeps
    transcription on exactly the same gateway, key and session headers the chat
    router uses; a re-keyed or renamed provider therefore needs no change here.
    """
    endpoint: str | None = None
    token: str | None = None
    headers: dict[str, str] = {}
    try:
        from ..config import Settings

        settings = Settings.load()
        cfg = settings.models.providers.get(settings.models.default)
        if cfg is None and settings.models.providers:
            cfg = next(iter(settings.models.providers.values()))
        if cfg is not None:
            endpoint = _openai_root(str(cfg.base_url or ""), str(cfg.kind or ""))
            if cfg.api_key_env:
                token = os.getenv(cfg.api_key_env)
            headers.update(cfg.extra_headers)
            for header, env_name in cfg.extra_header_env.items():
                value = os.getenv(env_name)
                if value:
                    headers[header] = value
    except Exception as exc:  # config must never break a channel turn
        _LOG.warning("voice transcription config unavailable: %s", type(exc).__name__)
    if token is None:
        token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    if token:
        headers.setdefault("Authorization", f"Bearer {token}")
    return endpoint, token, headers


async def transcribe_audio(
    raw: bytes,
    mime: str,
    name: str = "voice.ogg",
    *,
    model: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Transcribe audio bytes to text. Returns ``None`` on any failure."""
    endpoint, token, headers = _transport()
    if base_url:
        endpoint = _openai_root(base_url, "openai-compatible")
    if not token or not endpoint:
        # No values in the message: the log redactor treats the word "token"
        # as a secret marker and would mangle a format string built around it.
        _LOG.warning("voice transcription skipped: no usable OpenAI-compatible provider")
        return None
    model = model or os.getenv("LEO_STT_MODEL") or DEFAULT_MODEL
    audio_format = _audio_format(mime, name)
    encoded = base64.b64encode(raw).decode("ascii")
    # OpenAI's standard audio block. Some providers only accept the older
    # data-URL workaround; that is an explicit operator choice, never a default.
    block_mode = (os.getenv(AUDIO_BLOCK_ENV) or "input_audio").strip().lower()
    if block_mode == "image_url":
        audio_block: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": f"data:audio/{audio_format};base64,{encoded}"},
        }
    else:
        audio_block = {
            "type": "input_audio",
            "input_audio": {"data": encoded, "format": audio_format},
        }
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or DEFAULT_PROMPT},
                audio_block,
            ],
        }],
    }
    headers = {**headers, "Content-Type": "application/json"}

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    try:
        response = await http.post(
            _chat_url(endpoint), json=payload, headers=headers
        )
        if response.status_code >= 400:
            _LOG.warning("voice transcription failed: HTTP %s", response.status_code)
            return None
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        text = str(content or "").strip()
        return text or None
    except Exception as exc:  # a bad clip must never drop the message
        _LOG.warning("voice transcription error: %s", type(exc).__name__)
        return None
    finally:
        if owns_client:
            await http.aclose()


async def transcribe_channel_voice(
    channel: str,
    descriptor: dict[str, Any],
    adapters: dict[str, Any],
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> str | None:
    """Download one voice descriptor through its adapter, then transcribe it."""
    adapter = adapters.get(channel)
    download = getattr(adapter, "download", None)
    if download is None:
        return None
    try:
        raw, mime, name = await download(descriptor)
    except Exception as exc:
        _LOG.warning("voice download failed: %s", type(exc).__name__)
        return None
    return await transcribe_audio(raw, mime, name, model=model, prompt=prompt)
