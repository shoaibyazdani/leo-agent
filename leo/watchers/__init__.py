"""Judgment-driven proactivity.

The watcher is a read-only scheduled job: it gathers signals, asks a model
which ones deserve an interruption, and sends plain text through the existing
channel delivery path. It has no tools, takes no actions and never answers on
behalf of the user.
"""

from .delivery import DeliveryTracker
from .judgment import Decision, judge_signals
from .preferences import Preferences, load_preferences, mentions_topic
from .signals import Signal, gather_signals
from .state import WatcherState, load_state, save_state
from .watcher import (
    WORKER_WATCHER,
    WATCHER_SCHEDULE_NAME,
    JudgmentWatcher,
    WatcherRunResult,
    is_quiet,
    register_watcher_schedule,
)

__all__ = [
    "Decision",
    "DeliveryTracker",
    "JudgmentWatcher",
    "Preferences",
    "Signal",
    "WatcherRunResult",
    "WatcherState",
    "WORKER_WATCHER",
    "WATCHER_SCHEDULE_NAME",
    "gather_signals",
    "is_quiet",
    "judge_signals",
    "load_preferences",
    "load_state",
    "mentions_topic",
    "register_watcher_schedule",
    "save_state",
]
