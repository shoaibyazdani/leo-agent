"""Agent engine: twelve specialists, durable delegation, isolation, verification."""

from .engine import (
    AgentEngine,
    AgentError,
    AgentLimitError,
    AgentRequest,
    ScopedTools,
    detect_conflicts,
)
from .isolation import (
    Change,
    IntegrationResult,
    VerificationError,
    ValidationResult,
    Worktree,
    WorktreeError,
    WorktreeIsolation,
)
from .registry import SPECIALIST_SPECS, AgentRegistry
from .verification import VerificationEngine, VerificationResult, parse_verdict

__all__ = [
    "AgentRegistry", "AgentEngine", "AgentRequest", "AgentError", "AgentLimitError",
    "ScopedTools", "detect_conflicts", "WorktreeIsolation", "Worktree", "Change",
    "ValidationResult", "IntegrationResult", "WorktreeError", "VerificationError",
    "VerificationEngine", "VerificationResult", "parse_verdict", "SPECIALIST_SPECS",
]
