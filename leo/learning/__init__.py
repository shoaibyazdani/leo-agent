"""Approval-gated self-learning from LEO's durable history.

The engine turns grounded user statements and bounded tool/run trajectories
into evidence-backed memory proposals. Writes require a trusted human approval
or an explicitly enabled low-risk :class:`AutoNotePolicy`; secrets, raw tool
output, instruction-shaped text and model self-attestation are never learned.

Public entry points: :class:`leo.learning.engine.LearningEngine` for evidence-
backed notes and :class:`leo.learning.self_improvement.SelfImprovementEngine`
for approval-free skills and nightly review notes. Standing instructions are
not authored here; durable guidance is expressed as a documentation-only skill.
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
from .self_improvement import (
    NIGHTLY_SCHEDULE_NAME,
    NIGHTLY_WORKER,
    SelfImprovementDenied,
    SelfImprovementEngine,
    validate_change,
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
    "NIGHTLY_SCHEDULE_NAME",
    "NIGHTLY_WORKER",
    "SelfImprovementDenied",
    "SelfImprovementEngine",
    "validate_change",
]
