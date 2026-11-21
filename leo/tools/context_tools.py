"""Read-only context inspection tools backed by real messages and ContextOptimizer.

``context_stats`` counts real messages (including duplicates removed by the
injected ``ContextOptimizer``), ``token_estimate`` reports an explicitly
labelled characters/4 approximation, and ``context_preview`` returns a bounded,
secret-redacted view of the conversation.

Messages come from the durable ``messages`` table unless the composition root
injects a messages list in ``ctx['services']``. The optimizer is always the real
:class:`~leo.models.context.ContextOptimizer` (preferring the instance injected
under ``ctx['services']['context']``). No tool invents counts or content.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .. import db
from ..contracts import ChatMessage, RiskLevel, ToolSpec
from ..models.context import ContextOptimizer
from ..models.provider import CHARS_PER_TOKEN, estimate_tokens

TOOL_NAMES = ("context_stats", "token_estimate", "context_preview")

_TOKEN_METHOD = "characters/4 estimate (not a provider tokenizer count)"
_CONTEXT_KEYS = ("context", "context_optimizer", "optimizer", "context_service")
_MESSAGE_KEYS = ("messages", "context_messages", "chat_messages")

_MAX_MESSAGES = 200
_MAX_PREVIEW_MESSAGES = 20
_MAX_PREVIEW_CHARS = 4000
_MAX_TEXT_CHARS = 20000

# The current conversation is an execution-context fact, never a tool argument.
# ``correlation_id``/``task_id`` are populated by ToolExecutorImpl from the
# authenticated call, and the chat app sets them to the active conversation id.
_TRUSTED_CONVERSATION_KEYS = ("conversation_id", "current_conversation_id")
_QUOTA_PREFIXES = ("task:", "correlation:", "actor:")

# Settings redaction patterns are honoured too; these cover common secret forms
# that the default settings patterns do not (bearer credentials, JWTs, PINs).
_BUILTIN_REDACTIONS = (
    r"(?i)\bauthorization\s*:\s*bearer\s+\S+",
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}",
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    r"(?i)\b(pin|passcode)\s*[:=]\s*\S+",
)


def _schema(properties: dict[str, dict[str, Any]], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_CONVERSATION_ID = {"type": "string", "minLength": 1, "maxLength": 200}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": _MAX_MESSAGES, "default": 50}
_TEXT = {"type": "string", "minLength": 1, "maxLength": _MAX_TEXT_CHARS}
_BUDGET = {"type": "integer", "minimum": 1, "maximum": 1_000_000}


def _services(ctx: Any) -> Mapping[str, Any]:
    value = ctx.get("services") if isinstance(ctx, Mapping) else None
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError("ctx['services'] must be a mapping of service names to services")
    return value


def _settings(ctx: Any, services: Mapping[str, Any]) -> Any:
    for candidate in (services.get("settings"), ctx.get("settings") if isinstance(ctx, Mapping) else None):
        if candidate is not None:
            return candidate
    return None


def _optimizer(ctx: Any, services: Mapping[str, Any]) -> ContextOptimizer:
    """Use the runtime's real optimizer when injected; otherwise a real local one."""
    for key in _CONTEXT_KEYS:
        candidate = services.get(key)
        if candidate is None:
            continue
        if not (callable(getattr(candidate, "deduplicate", None))
                and callable(getattr(candidate, "optimize", None))):
            raise RuntimeError(
                f"ctx['services']['{key}'] is not a ContextOptimizer-like service "
                "(deduplicate() and optimize() are required)"
            )
        return candidate
    return ContextOptimizer()


def _redact_patterns(ctx: Any, services: Mapping[str, Any]) -> list[re.Pattern[str]]:
    patterns = list(_BUILTIN_REDACTIONS)
    settings = _settings(ctx, services)
    configured = getattr(getattr(settings, "security", None), "redact_patterns", None)
    if isinstance(configured, (list, tuple)):
        patterns.extend(str(pattern) for pattern in configured)
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            continue
    return compiled


def _redact(value: Any, patterns: list[re.Pattern[str]]) -> str:
    text = str(value if value is not None else "")
    for pattern in patterns:
        text = pattern.sub("[REDACTED]", text)
    return text


def _coerce_message(value: Any) -> ChatMessage | None:
    if isinstance(value, ChatMessage):
        return value
    if isinstance(value, Mapping):
        role = value.get("role")
        content = value.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        return ChatMessage(
            role=role,
            content=content,
            name=value.get("name"),
            tool_call_id=value.get("tool_call_id"),
            tool_calls=list(value.get("tool_calls") or []),
        )
    return None


def _current_conversation(ctx: Any) -> str | None:
    """Return the conversation the authenticated call is bound to, if any."""
    if not isinstance(ctx, Mapping):
        return None
    current: str | None = None
    for key in _TRUSTED_CONVERSATION_KEYS:
        value = ctx.get(key)
        if isinstance(value, str) and value:
            current = value
            break
    if current is None:
        # A task-scoped chat call is the strongest binding; the executor derives
        # it from ToolCall.task_id, which the chat app sets to the conversation.
        value = ctx.get("task_id")
        if isinstance(value, str) and value:
            candidate = value[len("task:"):] if value.startswith("task:") else value
            if (candidate and candidate != "global"
                    and not candidate.startswith(_QUOTA_PREFIXES)):
                current = candidate
        if current is None:
            value = ctx.get("correlation_id")
            if isinstance(value, str) and value and not value.startswith(_QUOTA_PREFIXES):
                current = value
    if current is None:
        return None
    # Optional composition-root ownership metadata: if supplied, the actor must
    # own the conversation before any of its messages are readable.
    owners = ctx.get("conversation_actors") or ctx.get("conversation_owners")
    if isinstance(owners, Mapping):
        allowed = owners.get(current)
        actor = ctx.get("actor")
        if allowed is not None and actor is not None and str(allowed) != str(actor):
            return None
    return current


def _bound_conversation(args: dict[str, Any], ctx: Any) -> str | None:
    """Bind an optional model-supplied conversation id to the execution context."""
    supplied = args.get("conversation_id")
    current = _current_conversation(ctx)
    if supplied is not None:
        supplied = str(supplied)
        if current is None or supplied != current:
            raise ValueError(
                "conversation_id does not match the authenticated actor's current conversation"
            )
        return supplied
    return current


async def _messages(args: dict[str, Any], ctx: Any, services: Mapping[str, Any]) -> tuple[list[ChatMessage], str]:
    """Resolve a bounded message list and label where it came from."""
    conversation_id = _bound_conversation(args, ctx)
    limit = int(args.get("limit", 50))

    injected: Any = None
    for key in _MESSAGE_KEYS:
        candidate = services.get(key)
        if candidate is not None:
            injected = candidate
            break
    if injected is None:
        context_service = next(
            (services[key] for key in _CONTEXT_KEYS if services.get(key) is not None), None
        )
        for attribute in ("messages", "chat_messages"):
            candidate = getattr(context_service, attribute, None)
            if isinstance(candidate, (list, tuple)):
                injected = candidate
                break
    if injected is not None:
        if not isinstance(injected, (list, tuple)):
            raise RuntimeError("injected messages service must be a list of messages")
        values = [_coerce_message(item) for item in injected]
        messages = [item for item in values if item is not None]
        return messages[-limit:], "services"

    if conversation_id is None:
        # A context:read caller may never fall back to a cross-conversation scan.
        raise ValueError(
            "context tools require a bound conversation in the tool context "
            "(ctx['conversation_id'] or ctx['correlation_id'])"
        )
    else:
        rows = await db.fetchall(
            "SELECT role, content FROM messages "
            "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
        source = "db:conversation"
    messages = [ChatMessage(role=row["role"], content=row["content"]) for row in rows]
    messages.reverse()  # oldest first, like a provider message list
    return messages, source


def _role_counts(messages: Sequence[ChatMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        counts[message.role] = counts.get(message.role, 0) + 1
    return counts


def _total_chars(messages: Sequence[ChatMessage]) -> int:
    return sum(len(message.content) for message in messages)


def _clip(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    if budget <= 3:
        return text[:budget], True
    return text[: budget - 3] + "...", True


async def context_stats(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Count real messages, duplicates and estimated tokens for a conversation."""
    services = _services(ctx)
    optimizer = _optimizer(ctx, services)
    messages, source = await _messages(args, ctx, services)
    unique = optimizer.deduplicate(messages)
    result: dict[str, Any] = {
        "success": True,
        "source": source,
        "conversation_id": args.get("conversation_id"),
        "message_count": len(messages),
        "unique_count": len(unique),
        "duplicate_count": len(messages) - len(unique),
        "role_counts": _role_counts(messages),
        "total_chars": _total_chars(messages),
        "estimated_tokens": sum(estimate_tokens(message.content) for message in messages),
        "token_method": _TOKEN_METHOD,
        "estimated": True,
    }
    budget = args.get("budget")
    if budget is not None:
        fitted = optimizer.optimize(unique, int(budget))
        result["budget"] = int(budget)
        result["optimized_message_count"] = len(fitted)
        result["optimized_estimated_tokens"] = sum(
            estimate_tokens(message.content) for message in fitted
        )
        result["bytes_dropped_by_budget"] = max(0, _total_chars(unique) - _total_chars(fitted))
    return result


async def token_estimate(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Estimate characters/4 tokens for text or for real stored messages."""
    text = args.get("text")
    if text is not None and args.get("conversation_id") is not None:
        raise ValueError("provide either 'text' or 'conversation_id', not both")
    services = _services(ctx)
    optimizer = _optimizer(ctx, services)
    if text is not None:
        chars = len(text)
        count = estimate_tokens(text)
        source = "text"
        messages: list[ChatMessage] = []
    else:
        messages, source = await _messages(args, ctx, services)
        messages = optimizer.deduplicate(messages)
        chars = _total_chars(messages)
        count = sum(estimate_tokens(message.content) for message in messages)
    result: dict[str, Any] = {
        "success": True,
        "source": source,
        "method": _TOKEN_METHOD,
        "chars_per_token": CHARS_PER_TOKEN,
        "estimated": True,
        "text_chars": chars,
        "estimated_tokens": count,
    }
    if source != "text":
        result["message_count"] = len(messages)
        result["by_role"] = [
            {"role": message.role, "chars": len(message.content),
             "estimated_tokens": estimate_tokens(message.content)}
            for message in messages
        ]
    return result


async def context_preview(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, redacted preview of real messages."""
    services = _services(ctx)
    optimizer = _optimizer(ctx, services)
    patterns = _redact_patterns(ctx, services)
    messages, source = await _messages(args, ctx, services)
    unique = optimizer.deduplicate(messages)
    max_messages = int(args.get("max_messages", 5))
    max_chars = int(args.get("max_chars", 1000))
    items: list[dict[str, Any]] = []
    used = 0
    redacted = False
    for index, message in enumerate(unique[:max_messages]):
        remaining = max_chars - used
        if remaining <= 0:
            break
        content, truncated = _clip(_redact(message.content, patterns), remaining)
        role = _redact(message.role, patterns)
        name = _redact(message.name, patterns) if message.name is not None else None
        if "[REDACTED]" in content or "[REDACTED]" in role or "[REDACTED]" in str(name):
            redacted = True
        used += len(content)
        items.append({
            "index": index,
            "role": role,
            "name": name,
            "content": content,
            "chars": len(content),
            "truncated": truncated,
        })
    return {
        "success": True,
        "source": source,
        "conversation_id": _redact(args.get("conversation_id"), patterns) if args.get("conversation_id") else None,
        "max_messages": max_messages,
        "max_chars": max_chars,
        "count": len(items),
        "total_chars": used,
        "redacted": redacted,
        "messages": items,
        "token_method": _TOKEN_METHOD,
        "estimated_tokens": math.ceil(used / CHARS_PER_TOKEN),
    }


def register(registry: Any, services: Any = None) -> list[str]:
    """Register the three context tools and return their names.

    ``services`` is accepted for the standard toolset signature; the concrete
    optimizer and messages are resolved from each call's ``ctx['services']``,
    with a fallback to a real local optimizer and the durable message table.
    """
    definitions: list[tuple[ToolSpec, Any]] = [
        (
            ToolSpec(
                name="context_stats",
                description="Count real conversation messages, duplicates and estimated tokens.",
                category="context",
                risk=RiskLevel.LOW,
                permissions=["context:read"],
                input_schema=_schema({
                    "conversation_id": _CONVERSATION_ID,
                    "limit": _LIMIT,
                    "budget": _BUDGET,
                }),
                resource_field="",
                supports_dry_run=True,
            ),
            context_stats,
        ),
        (
            ToolSpec(
                name="token_estimate",
                description="Estimate tokens as characters/4 for text or real stored messages.",
                category="context",
                risk=RiskLevel.LOW,
                permissions=["context:read"],
                input_schema=_schema({
                    "text": _TEXT,
                    "conversation_id": _CONVERSATION_ID,
                    "limit": _LIMIT,
                }),
                resource_field="",
                supports_dry_run=True,
            ),
            token_estimate,
        ),
        (
            ToolSpec(
                name="context_preview",
                description="Preview a bounded, secret-redacted slice of real conversation messages.",
                category="context",
                risk=RiskLevel.LOW,
                permissions=["context:read"],
                input_schema=_schema({
                    "conversation_id": _CONVERSATION_ID,
                    "limit": _LIMIT,
                    "max_messages": {
                        "type": "integer", "minimum": 1,
                        "maximum": _MAX_PREVIEW_MESSAGES, "default": 5,
                    },
                    "max_chars": {
                        "type": "integer", "minimum": 1,
                        "maximum": _MAX_PREVIEW_CHARS, "default": 1000,
                    },
                }),
                resource_field="",
                supports_dry_run=True,
            ),
            context_preview,
        ),
    ]
    registered: list[str] = []
    for spec, handler in definitions:
        try:
            registry.register(spec, handler)
        except ValueError:  # keep a caller's existing definition
            continue
        registered.append(spec.name)
    return registered


__all__ = [
    "TOOL_NAMES",
    "register",
    "context_stats",
    "token_estimate",
    "context_preview",
]
