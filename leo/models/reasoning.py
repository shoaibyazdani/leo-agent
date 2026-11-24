"""Canonical reasoning-effort vocabulary and wire clamping for LEO.

One internal ladder is wider than any single provider wire accepts, and one
clamp maps a request onto the levels that wire supports. The clamp never
escalates cost: an unsupported request resolves to the nearest WEAKER supported
level, and only when nothing weaker exists does it fall back to the weakest
supported level.

Wire shape stays in the provider adapters; only the vocabulary math lives here.
Unset stays unset (never invent an effort), so a normal turn sends no reasoning
parameter and behaves exactly as before.
"""

from __future__ import annotations

from typing import Iterable, Mapping

# Low -> high. ``none`` is an explicit disable; ``ultra`` is internal (no wire
# accepts it, every declared set stops at ``max``).
EFFORT_LADDER: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)

# Modern Anthropic models take ``output_config.effort`` (low..max) and
# ``thinking: {type: adaptive}``. They do not accept ``budget_tokens``, and some
# cannot disable thinking at all -- ``none`` is sent as ``thinking.disabled`` and
# the provider-level 400 fallback degrades it transparently.
ANTHROPIC_EFFORT_WIRE: tuple[str, ...] = (
    "low", "medium", "high", "xhigh", "max",
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
    overrides: Mapping[str, str] | None = None,
) -> str | None:
    """Clamp ``effort`` onto ``supported`` without ever escalating cost.

    Passes through unchanged when it is already supported, when the supported
    set is unknown/empty, or when it is not a recognized ladder level (custom
    providers may use bespoke names). ``overrides`` is a declared vendor map
    consulted first (for example ``{"medium": "high"}`` where a model's
    positional middle is high). Otherwise returns the nearest weaker supported
    level, or the weakest enabled supported level when nothing is weaker.
    """
    requested = str(effort or "").strip().lower()
    if not requested:
        return None
    # An explicit disable is not an effort to clamp. Providers that cannot accept
    # a disable token handle it at their own transport boundary.
    if requested == "none":
        return "none"
    levels = [str(s).strip().lower() for s in (supported or ())]
    levels = [level for level in levels if level in EFFORT_LADDER]
    if not levels or requested not in EFFORT_LADDER or requested in levels:
        return requested
    mapped = (overrides or {}).get(requested)
    if mapped in levels:
        return str(mapped)
    requested_idx = EFFORT_LADDER.index(requested)
    # Never resolve an enabled ask down to ``none``: that would silently turn
    # thinking off instead of making it cheaper.
    below = [
        level for level in levels
        if level != "none" and EFFORT_LADDER.index(level) < requested_idx
    ]
    if below:
        return max(below, key=EFFORT_LADDER.index)
    enabled = [level for level in levels if level != "none"]
    return min(enabled, key=EFFORT_LADDER.index) if enabled else requested


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
    if ceiling <= 1_024:
        # The API requires 1024 <= budget < max_tokens. There is no legal
        # thinking block under a ceiling this small.
        return None
    budget = min(budget, ceiling - 1_024)
    return max(1_024, budget)


# Model families whose extended thinking uses adaptive effort instead of a
# token budget. Kept deliberately narrow: an unknown gateway model keeps the
# budget shape, which is the broader-compatibility default for Anthropic-style
# endpoints.
_ADAPTIVE_CLAUDE_PREFIXES: tuple[str, ...] = (
    "claude-fable", "claude-mythos", "claude-opus-5", "claude-sonnet-5",
    "claude-haiku-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6",
)


def uses_adaptive_effort(model: str | None) -> bool:
    """True when ``model`` is a Claude family that takes adaptive effort.

    ``reasoning_style`` on the provider still wins; this only informs ``auto``.
    """
    name = str(model or "").strip().lower()
    return any(name.startswith(prefix) for prefix in _ADAPTIVE_CLAUDE_PREFIXES)


# Response bodies differ, but a 400 that rejects a reasoning field is
# recognizable across providers. The fallback must not swallow unrelated 400s.
_REASONING_REJECTION_HINTS: tuple[str, ...] = (
    "thinking", "reasoning", "budget_tokens", "output_config", "effort",
    "reasoning_effort", "reasoning effort",
)


def is_reasoning_rejection(status_code: int, message: str) -> bool:
    """Whether an HTTP rejection points at reasoning parameters specifically."""
    if int(status_code or 0) not in (400, 422):
        return False
    text = str(message or "").lower()
    return any(hint in text for hint in _REASONING_REJECTION_HINTS)


def applied_note(requested: str | None, applied: str | None) -> str:
    """One-line, truthful explanation when a level was mapped onto a wire."""
    if not requested:
        return ""
    if not applied:
        return f"{requested} was not sent (provider accepted no reasoning parameter)"
    if requested != applied:
        return f"{requested} mapped to {applied} (provider wire stops there)"
    return ""


__all__ = [
    "ANTHROPIC_EFFORT_WIRE", "ANTHROPIC_WIRE_EFFORTS", "EFFORT_LADDER",
    "OPENAI_WIRE_EFFORTS", "applied_note", "clamp_effort",
    "is_reasoning_rejection", "parse_effort", "thinking_budget",
    "uses_adaptive_effort",
]
