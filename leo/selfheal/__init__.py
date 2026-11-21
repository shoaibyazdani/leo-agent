"""Approval-gated self-healing for LEO.

The package observes durable failure events, clusters them into bounded
incidents, proposes concrete remediations, and applies a change only after a
HIGH-risk approval was resolved by a trusted non-chat human. It never edits
source files by itself: an application change is performed by a caller-supplied
``applier`` callable, and only after a rollback snapshot exists.

Public entry point: :class:`leo.selfheal.engine.SelfHealingEngine`.
"""

from .engine import (
    ACTION_APPLY,
    ACTION_ROLLBACK,
    ACTOR,
    RETRYABLE_KINDS,
    FailureIncident,
    HealingDenied,
    Remediation,
    RollbackSnapshot,
    SelfHealingEngine,
    WATCHED_TOPICS,
    classify_failure,
)

__all__ = [
    "ACTION_APPLY",
    "ACTION_ROLLBACK",
    "ACTOR",
    "RETRYABLE_KINDS",
    "WATCHED_TOPICS",
    "FailureIncident",
    "HealingDenied",
    "Remediation",
    "RollbackSnapshot",
    "SelfHealingEngine",
    "classify_failure",
]
