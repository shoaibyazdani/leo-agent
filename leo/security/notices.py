"""Canonical first lines for notices LEO writes into a chat.

A notice goes to the operator's own channel, and in some topologies the platform
hands that message back as an inbound update. Ingesting it makes the agent answer
its own words, which the operator reads as the agent "mixing things up". The
prefix is therefore shared between the sender (``security.notifier``) and the
ingress guard (``integrations.messaging``) instead of being copied as a literal.

Deliberately import-free: both sides can import this without a cycle.
"""

APPROVAL_PREFIX = "🔐 Approval needed"
REVIEW_CREATED_PREFIX = "✨ Skill created"
REVIEW_STAGED_PREFIX = "📝 Skill write staged"
REVIEW_FAILED_PREFIX = "⚠️ Post-turn review failed"

# Every prefix a channel must refuse to ingest as a human turn.
SELF_NOTICE_PREFIXES = (
    APPROVAL_PREFIX,
    REVIEW_CREATED_PREFIX,
    REVIEW_STAGED_PREFIX,
    REVIEW_FAILED_PREFIX,
)

__all__ = [
    "APPROVAL_PREFIX",
    "REVIEW_CREATED_PREFIX",
    "REVIEW_FAILED_PREFIX",
    "REVIEW_STAGED_PREFIX",
    "SELF_NOTICE_PREFIXES",
]
