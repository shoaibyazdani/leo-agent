"""The scheduled judgment watcher orchestration.

A run is deliberately boring: if no signal survives preference filtering and
model judgment, nothing is sent. The watcher never holds a tool registry and
never mutates tasks, conversations or approvals; its only side effects are its
own small cursor state and a plain-text message through channel delivery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..contracts import ModelRequest
from .delivery import DeliveryTracker
from .judgment import judge_signals
from .preferences import Preferences, clock_in_range, load_preferences
from .signals import gather_signals
from .state import WatcherState, load_state, save_state

_LOG = logging.getLogger("leo.watchers")
WORKER_WATCHER = "watcher"
WATCHER_SCHEDULE_NAME = "judgment-watcher"


@dataclass
class WatcherRunResult:
    suppressed: bool = False
    reason: str = ""
    signals_seen: int = 0
    worthy: list[str] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    pending: int = 0
    ordinary_sent: int = 0
    pending_flushed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "suppressed": self.suppressed,
            "reason": self.reason,
            "signals_seen": self.signals_seen,
            "worthy": self.worthy,
            "sent": self.sent,
            "pending": self.pending,
            "ordinary_sent": self.ordinary_sent,
            "pending_flushed": self.pending_flushed,
        }


def _zone(config: Any, timezone: str | None = None) -> ZoneInfo:
    name = str(timezone or getattr(config, "timezone", "") or "Asia/Karachi")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def is_quiet(now: datetime, preferences: Preferences, config: Any) -> bool:
    """True when the local wall clock is inside the quiet hours."""
    local = (
        (now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC))
        .astimezone(_zone(config, preferences.timezone))
    )
    start, end = preferences.quiet_hours or ("23:00", "09:00")
    return clock_in_range(local.time().replace(second=0, microsecond=0), start, end)


def _local_day(now: datetime, config: Any, timezone: str | None = None) -> str:
    local = (
        (now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC))
        .astimezone(_zone(config, timezone))
    )
    return local.date().isoformat()


async def register_watcher_schedule(scheduler: Any, config: Any) -> dict[str, Any] | None:
    """Put the watcher on LEO's scheduler without touching engine internals.

    An existing row keeps its cursor: restarts must not shift the cadence or
    replay occurrences. Disabling the config disables the row rather than
    deleting its history.
    """
    try:
        existing = await scheduler.get(WATCHER_SCHEDULE_NAME)
    except KeyError:
        existing = None
    if not bool(getattr(config, "enabled", True)):
        if existing is not None:
            await scheduler.set_enabled(WATCHER_SCHEDULE_NAME, False)
        return None
    seconds = float(getattr(config, "cadence_seconds", 1800) or 1800)
    if existing is not None and str(existing.get("expression")) == str(seconds):
        if not existing.get("enabled"):
            await scheduler.set_enabled(WATCHER_SCHEDULE_NAME, True)
        return existing
    return await scheduler.register_interval(
        WATCHER_SCHEDULE_NAME,
        seconds,
        "Decide whether anything deserves an update",
        {"worker": WORKER_WATCHER},
    )


class JudgmentWatcher:
    def __init__(
        self,
        config: Any,
        *,
        model_client: Callable[[ModelRequest], Awaitable[Any]] | None = None,
        send: Callable[[str], Awaitable[Any]] | None = None,
        preferences_path: str | None = None,
        clock: Callable[[], datetime] | None = None,
        github_fetcher: Callable[..., Awaitable[list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.send = send
        self.preferences_path = preferences_path or getattr(config, "preferences_path", None)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.github_fetcher = github_fetcher

    async def run(self) -> WatcherRunResult:
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        preferences = self._preferences()
        if is_quiet(now, preferences, self.config):
            return WatcherRunResult(suppressed=True, reason="quiet hours")

        state = await load_state()
        signals = await gather_signals(
            state=state,
            now=now,
            config=self.config,
            github_fetcher=self.github_fetcher,
        )
        # A carried failed delivery is news in its own right: flush it on the
        # next run even when nothing new survived judgment.
        tracker = DeliveryTracker(self.send, state)
        flushed = await tracker.flush()
        day = _local_day(now, self.config, preferences.timezone)
        if state.day != day:
            state.day = day
            state.ordinary_sent = 0
        fresh = [item for item in signals if item.id not in state.seen]
        state.mark_seen([item.id for item in signals])
        state.last_run_at = now.astimezone(UTC).isoformat()

        if not fresh:
            await save_state(state)
            return WatcherRunResult(
                signals_seen=len(signals),
                reason="no new signals",
                pending=len(state.pending),
                pending_flushed=flushed,
            )

        decisions = await judge_signals(
            fresh,
            preferences,
            self.model_client,
            model=str(getattr(self.config, "model", "") or "") or None,
        )
        # Urgent classes (failure, reply, moved deadline) go out now and never
        # consume the ordinary-item budget. Ordinary items respect the cap.
        urgent_ids = {item.id for item in fresh if item.urgent}
        urgent = [item for item in decisions if item.signal_id in urgent_ids]
        ordinary = [item for item in decisions if item.signal_id not in urgent_ids]
        allowed_ids = {
            item.signal_id
            for item in ordinary[: max(0, preferences.max_ordinary_per_day - state.ordinary_sent)]
        }
        sent: list[str] = []
        for decision in urgent + ordinary:
            if decision.signal_id not in urgent_ids and decision.signal_id not in allowed_ids:
                continue
            if await tracker.deliver(decision.message, urgent=decision.signal_id in urgent_ids):
                sent.append(decision.signal_id)
                if decision.signal_id in allowed_ids:
                    state.ordinary_sent += 1
        await save_state(state)
        return WatcherRunResult(
            signals_seen=len(signals),
            worthy=[item.signal_id for item in decisions],
            sent=sent,
            pending=len(state.pending),
            ordinary_sent=state.ordinary_sent,
            pending_flushed=flushed,
        )

    def _preferences(self) -> Preferences:
        start = str(getattr(self.config, "waking_start", "09:00") or "09:00")
        end = str(getattr(self.config, "waking_end", "23:00") or "23:00")
        return load_preferences(
            self.preferences_path,
            # Waking hours are the complement of quiet hours.
            default_quiet_hours=(end, start),
            default_max=2,
            default_timezone=str(getattr(self.config, "timezone", "") or None),
        )


__all__ = [
    "WORKER_WATCHER",
    "WATCHER_SCHEDULE_NAME",
    "JudgmentWatcher",
    "WatcherRunResult",
    "is_quiet",
    "register_watcher_schedule",
]
