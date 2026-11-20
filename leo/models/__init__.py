"""Model engine: provider adapters, routing, escalation and context optimization."""

from .context import ContextOptimizer
from .mock import MockProvider
from .provider import (
    ANTHROPIC_VERSION,
    AnthropicProvider,
    ModelError,
    ModelHTTPError,
    ModelTimeoutError,
    ModelUnsupportedError,
    ModelValidationError,
    OpenAIProvider,
    build_provider,
    estimate_tokens,
    normalize_tool_calls,
    usage_from_counts,
)
from .router import ModelRouter, TASK_CATEGORIES, classify_task

__all__ = [
    "ANTHROPIC_VERSION",
    "AnthropicProvider",
    "ContextOptimizer",
    "ModelError",
    "ModelHTTPError",
    "ModelRouter",
    "ModelTimeoutError",
    "ModelUnsupportedError",
    "ModelValidationError",
    "MockProvider",
    "OpenAIProvider",
    "TASK_CATEGORIES",
    "build_provider",
    "classify_task",
    "estimate_tokens",
    "normalize_tool_calls",
    "usage_from_counts",
]
