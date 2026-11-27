"""The watcher's model judgment step.

The model gets a strict-JSON decision request and *no tools*. It can only say
whether a candidate signal is worth an interruption and suggest plain text.
Hard policy (never-mention topics, must-tell urgent classes, pacing) is applied
here in code after the response: a model cannot re-include an excluded topic or
invent a new signal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..contracts import ChatMessage, ModelRequest
from .preferences import Preferences, mentions_topic
from .signals import Signal

_LOG = logging.getLogger("leo.watchers.judgment")
_SYSTEM = (
    "You are the judgment step for LEO's background watcher. Decide which "
    "candidate signals deserve a short, plain-text interruption and which are "
    "routine noise. Return STRICT JSON only, no prose and no code fence:\n"
    '{"decisions":[{"id":"<signal id>","worthy":true,"message":"<=200 chars",'
    '"reason":"one line"}]}\n'
    "Rules: only use ids from the supplied candidates; never invent signals. "
    "Never mention topics the operator excluded. Do not promise to carry, "
    "remember or deliver a specific item in the future; a specific promise "
    "belongs to a scheduled job. Do not ask for actions, approvals or replies "
    "to third parties. Failures, arrived replies and moved deadlines are "
    "must-tell classes; you may improve their wording but cannot drop them. "
    "Silence is normal: mark routine signals unworthy when in doubt."
)


@dataclass(frozen=True)
class Decision:
    signal_id: str
    worthy: bool
    message: str
    reason: str


def _parse_json(content: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content or "").strip(), flags=re.I).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def fallback_message(signal: Signal) -> str:
    """Model-free wording, used for urgent items when no model is available."""
    return signal.text.strip()[:240]


async def judge_signals(
    signals: list[Signal],
    preferences: Preferences,
    model_client: Callable[[ModelRequest], Awaitable[Any]] | None,
    *,
    model: str | None = None,
) -> list[Decision]:
    """Apply preferences and model judgment; return only tellable decisions."""
    if not signals:
        return []
    request = ModelRequest(
        messages=[
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(
                role="user",
                content="PREFERENCES:\n"
                + json.dumps(
                    {
                        "tell_topics": list(preferences.tell_topics),
                        "never_topics": list(preferences.never_topics),
                    },
                    ensure_ascii=False,
                )
                + "\nCANDIDATES:\n"
                + json.dumps([item.as_dict() for item in signals], ensure_ascii=False, default=str),
            ),
        ],
        tools=[],
        task_category="watcher_judgment",
        agent_id="watcher",
        model=model,
        max_tokens=1024,
    )
    plan: dict[str, Any] | None = None
    if model_client is not None:
        try:
            response = await model_client(request)
            plan = _parse_json(str(getattr(response, "content", "") or ""))
        except Exception as exc:  # noqa: BLE001 - watcher must stay alive
            _LOG.warning("watcher judgment unavailable", extra={"error": str(exc)[:200]})
    raw: dict[str, Any] = {}
    if isinstance(plan, dict) and isinstance(plan.get("decisions"), list):
        for item in plan["decisions"]:
            if isinstance(item, dict) and str(item.get("id") or "") not in raw:
                raw[str(item.get("id"))] = item

    decisions: list[Decision] = []
    for signal in signals:
        # Hard filters: an excluded topic is never sent, whatever the model said.
        if mentions_topic(signal.topic, preferences.never_topics) or mentions_topic(
            signal.text, preferences.never_topics
        ):
            continue
        if (
            not signal.urgent
            and preferences.tell_topics
            and not mentions_topic(f"{signal.topic} {signal.text}", preferences.tell_topics)
        ):
            continue
        item = raw.get(signal.id, {})
        suggestion = str(item.get("message") or "").strip()
        if mentions_topic(suggestion, preferences.never_topics):
            suggestion = ""
        reason = str(item.get("reason") or "")[:200]
        if signal.urgent:
            decisions.append(
                Decision(
                    signal.id,
                    True,
                    suggestion[:240] or fallback_message(signal),
                    reason or "urgent class",
                )
            )
        elif bool(item.get("worthy")):
            decisions.append(
                Decision(signal.id, True, suggestion[:240] or fallback_message(signal), reason)
            )
    return decisions
