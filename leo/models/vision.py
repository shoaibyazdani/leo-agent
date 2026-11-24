"""The vision slot: images are read by their own model, never by the chat model.

Mirrors Hermes' auxiliary vision router. LEO's chat model is chosen for reasoning
and is usually text-only, so attaching pixels to it fails the request upstream and
the operator's photo is silently dropped. Here the image goes to a provider that
declares ``vision``, the answer comes back as text, and the turn continues with a
description it can actually reason about.

Two entry points share the same rule:

- a photo sent in chat (``LeoApp._images_for_turn``), and
- the ``vision_analyze`` tool, for a file the agent names itself.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import Settings, VisionConfig
from ..contracts import ChatMessage, ModelRequest

log = logging.getLogger(__name__)


class VisionDescriber:
    """Resolve the vision slot and read images through it."""

    def __init__(self, settings: Settings, router: Any) -> None:
        self.settings = settings
        self.router = router

    # ---------------------------------------------------------------- routing
    @property
    def config(self) -> VisionConfig:
        return self.settings.models.vision

    def slot(self) -> str | None:
        """The provider slot that reads images: configured first, else first capable."""
        config = self.config
        if not config.enabled or config.mode == "off":
            return None
        providers = self.settings.models.providers
        if config.provider:
            configured = providers.get(config.provider)
            if configured is not None and configured.enabled:
                return config.provider
        for name, provider in providers.items():
            if provider.enabled and "vision" in (provider.capabilities or []):
                return name
        return None

    def vision_declared(self, slot: str | None) -> bool:
        if not slot:
            return False
        provider = self.settings.models.providers.get(slot)
        return bool(provider and "vision" in (provider.capabilities or []))

    def handles(self, images: list[dict[str, Any]] | None, *, main_slot: str | None = None) -> bool:
        """Should these images be described instead of attached?"""
        if not images:
            return False
        mode = self.config.mode
        if not self.config.enabled or mode == "off":
            return False
        if mode == "native":
            return False
        if self.slot() is None:
            return False
        if mode == "auto":
            # Attach natively when the chat model itself declares vision.
            provider = self.settings.models.providers.get(main_slot or "")
            if provider is not None and "vision" in (provider.capabilities or []):
                return False
        return True

    # ------------------------------------------------------------------- call
    async def describe(
        self, images: list[dict[str, Any]], *, question: str | None = None
    ) -> str | None:
        """Read images and return text; ``None`` on any failure, never an exception.

        An unreadable image must cost the operator a sentence, not the whole turn.
        """
        if not images:
            return None
        slot = self.slot()
        if slot is None:
            return None
        request = ModelRequest(
            messages=[
                ChatMessage(role="user", content=question or self.config.prompt, images=list(images))
            ],
            task_category="vision",
            max_tokens=int(self.config.max_tokens),
        )
        try:
            response = await self.router.complete(request, slot=slot)
        except Exception as exc:  # noqa: BLE001 - a broken reader must not kill the turn
            log.warning(
                "vision describe failed: slot=%s error=%s", slot, str(exc)[:200], exc_info=False
            )
            return None
        text = str(getattr(response, "content", "") or "").strip()
        if not text:
            log.warning("vision describe returned no text: slot=%s", slot)
            return None
        limit = max(200, int(self.config.max_description_chars))
        return text[:limit]
