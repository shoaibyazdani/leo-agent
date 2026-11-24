"""Canonical reasoning-effort vocabulary and wire clamping for LEO.

Ported from Hermes' ``agent/reasoning_effort.py`` approach: one internal ladder
wider than any single provider wire accepts, and a single clamp that maps a
request onto a wire's supported levels. The clamp never escalates cost: an
unsupported request resolves to the nearest WEAKER supported level, and only
when nothing weaker exists does it fall back to the weakest supported level.

Wire shape stays in the provider adapters; only the vocabulary math lives here.
Unset stays unset (never invent an effort), so a normal turn sends no reasoning
parameter and behaves exactly as before.
"""

from __future__ import annotations

from typing import Iterable

# Low -> high. ``none`` is an explicit disable; ``ultra`` is internal (no wire
# accepts it, every declared set stops at ``max``).
EFFORT_LADDER: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)

# Widest OpenAI-compatible wire vocabulary (OpenRouter / Nous-style gateways).
# A provider may narrow this via ``ModelProviderConfig.reasoning_efforts``.
OPENAI_WIRE_EFFORTS: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
)

# Anthropic-compatible wires take a thinking budget, not a level string; the
# level is clamped onto this set first, then mapped to a token budget.
ANTHROPIC_WIRE_EFFORTS: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
)

# Human aliases accepted by /reasoning. Everything else is rejected rather
# than silently coerced to a default.
_ALIASES: dict[str, str] = {
    "off": "none", "false": "none", "disable": "none", "disabled": "none",
    "min": "minimal",
    "med": "medium",
    "hi": "high",
    "xh": "xhigh",
    "extreme": "max",
}

# Thinking-token budgets by level for Anthropic-compatible wires.
_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1_024,
    "low": 2_048,
    "medium": 4_096,
    "high": 8_192,
    "xhigh": 16_384,
    "max": 32_768,
    "ultra": 32_768,
}


def parse_effort(raw: str | None) -> str | None:
    """Normalize a user-supplied effort token onto :data:`EFFORT_LADDER`.

    Returns ``None`` for empty/unrecognized input so callers can reject it
    explicitly instead of guessing a level.
    """
    token = str(raw or "").strip().lower()
    if not token:
        return None
    token = _ALIASES.get(token, token)
    return token if token in EFFORT_LADDER else None


def clamp_effort(
    effort: str | None,
    supported: Iterable[str] | None = None,
) -> str | None:
    """Clamp ``effort`` onto ``supported`` without ever escalating cost.

    Passes through unchanged when it is already supported, when the supported
    set is unknown/empty, or when it is not a recognized ladder level (custom
    providers may use bespoke names). Otherwise returns the nearest weaker
    supported level, or the weakest supported level when nothing is weaker.
    """
    requested = str(effort or "").strip().lower()
    if not requested:
        return None
    levels = [str(s).strip().lower() for s in (supported or ())]
    levels = [level for level in levels if level in EFFORT_LADDER]
    if not levels or requested not in EFFORT_LADDER or requested in levels:
        return requested
    requested_idx = EFFORT_LADDER.index(requested)
    below = [level for level in levels if EFFORT_LADDER.index(level) < requested_idx]
    if below:
        return max(below, key=EFFORT_LADDER.index)
    return min(levels, key=EFFORT_LADDER.index)


def thinking_budget(effort: str | None, max_tokens: int | None) -> int | None:
    """Return an Anthropic ``thinking.budget_tokens`` for a clamped level.

    ``None`` means no thinking block (effort unset or ``none``). The budget is
    always >= 1024 and strictly below ``max_tokens`` as the API requires.
    """
    level = str(effort or "").strip().lower()
    if not level or level == "none":
        return None
    budget = _THINKING_BUDGETS.get(level, _THINKING_BUDGETS["medium"])
    ceiling = int(max_tokens or 32_000)
    # Leave room for the answer; never exceed the provider's own ceiling.
    budget = min(budget, max(1_024, ceiling - 1_024))
    return max(1_024, budget)


__all__ = [
    "ANTHROPIC_WIRE_EFFORTS", "EFFORT_LADDER", "OPENAI_WIRE_EFFORTS",
    "clamp_effort", "parse_effort", "thinking_budget",
]
