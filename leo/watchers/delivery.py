"""Watcher delivery pacing and failed-delivery reporting.

The watcher sends only through the channel delivery callable supplied by the
app composition root (the same handler scheduled tasks use). One immediate
retry is attempted; a message that still fails stays in durable state and is
attached to the next successful contact instead of disappearing.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from .state import WatcherState

_LOG = logging.getLogger("leo.watchers.delivery")
Send = Callable[[str], Awaitable[Any]]


class DeliveryTracker:
    def __init__(self, send: Send | None, state: WatcherState) -> None:
        self.send = send
        self.state = state

    def _carried(self) -> str:
        carried = "\n".join(
            f"- {str(item.get('text') or '')[:300]}" for item in self.state.pending
        )
        return f"Earlier updates could not be delivered:\n{carried}"

    def _compose(self, text: str) -> str:
        if not self.state.pending:
            return text
        return f"{self._carried()}\n\n{text}"

    async def _attempt_send(self, message: str) -> bool:
        if self.send is None:
            return False
        for attempt in range(2):
            try:
                await self.send(message)
                return True
            except Exception as exc:  # noqa: BLE001 - deferred, never silent
                _LOG.warning(
                    "watcher delivery attempt failed",
                    extra={"attempt": attempt + 1, "error": str(exc)[:200]},
                )
        return False

    async def deliver(self, text: str, *, urgent: bool = False) -> bool:
        """Deliver ``text``, retry once, then carry it forward on failure."""
        if self.send is None:
            self.state.pending.append(
                {"text": text, "at": datetime.now(UTC).isoformat(), "urgent": urgent}
            )
            return False
        if await self._attempt_send(self._compose(text)):
            self.state.pending.clear()
            return True
        self.state.pending.append({"text": text, "at": datetime.now(UTC).isoformat(), "urgent": urgent})
        return False

    async def flush(self) -> bool:
        """Send carried items on this run as their own message, not as a footer.

        A failed flush leaves the items pending; they stay durable and can be
        retried by a later run instead of being dropped or duplicated.
        """
        if not self.state.pending:
            return False
        if await self._attempt_send(self._carried()):
            self.state.pending.clear()
            return True
        return False
