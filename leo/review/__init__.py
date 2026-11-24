"""Bounded post-turn review (the self-improvement loop).

The package is deliberately separate from ``leo.learning`` (deterministic
memory notes) and ``leo.agents`` (delegated tasks): a review is a system pass
with its own trigger, transcript bounds, tool scope, artifact and audit row.
"""

from .engine import REVIEW_ACTOR, REVIEW_SCOPE, ReviewEngine, ReviewOutcome
from .transcript import Transcript, load_transcript

__all__ = [
    "REVIEW_ACTOR",
    "REVIEW_SCOPE",
    "ReviewEngine",
    "ReviewOutcome",
    "Transcript",
    "load_transcript",
]
