"""Approval-gated self-learning from LEO's durable history.

The engine turns grounded user statements and bounded tool/run trajectories
into evidence-backed memory proposals. Writes require a trusted human approval
or an explicitly enabled low-risk :class:`AutoNotePolicy`; secrets, raw tool
output, instruction-shaped text and model self-attestation are never learned.

Public entry point: :class:`leo.learning.engine.LearningEngine`.
"""

from .engine import (
    ACTION_CONSOLIDATE,
    ACTION_NOTE,
    ACTOR,
    MAX_CONTENT_CHARS,
    ApprovalRequired,
    AutoNotePolicy,
    ConsolidationPlan,
    EvidenceRef,
    LearningDenied,
    LearningEngine,
    LearningProposal,
    NoteDraft,
)

__all__ = [
    "ACTION_CONSOLIDATE",
    "ACTION_NOTE",
    "ACTOR",
    "MAX_CONTENT_CHARS",
    "ApprovalRequired",
    "AutoNotePolicy",
    "ConsolidationPlan",
    "EvidenceRef",
    "LearningDenied",
    "LearningEngine",
    "LearningProposal",
    "NoteDraft",
]
