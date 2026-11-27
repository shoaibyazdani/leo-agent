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

    def _compose(self, text: str) -> str:
        if not self.state.pending:
            return text
        carried = "\n".join(f"- {str(item.get('text') or '')[:300]}" for item in self.state.pending)
        return f"Earlier updates could not be delivered:\n{carried}\n\n{text}"

    async def deliver(self, text: str, *, urgent: bool = False) -> bool:
        """Deliver ``text``, retry once, then carry it forward on failure."""
        if self.send is None:
            self.state.pending.append(
                {"text": text, "at": datetime.now(UTC).isoformat(), "urgent": urgent}
            )
            return False
        message = self._compose(text)
        for attempt in range(2):
            try:
                await self.send(message)
                self.state.pending.clear()
                return True
            except Exception as exc:  # noqa: BLE001 - deferred, never silent
                _LOG.warning(
                    "watcher delivery attempt failed",
                    extra={"attempt": attempt + 1, "error": str(exc)[:200]},
                )
        self.state.pending.append({"text": text, "at": datetime.now(UTC).isoformat(), "urgent": urgent})
        return False
