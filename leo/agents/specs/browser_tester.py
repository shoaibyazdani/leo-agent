"""Browser tester specialist: deterministic UI checks via operator tools.

This spec uses a bespoke handler, :func:`make_handler`, instead of the default
model delegation handler. The handler drives the registered browser operator
tools (through the agent's scoped tool view), then reports the final URL, a
text snapshot and a deterministic assertion outcome as evidence.

Check input can be a JSON objective, a mapping objective, or ``context`` keys::

    {"url": "https://example.org/", "selector": "body",
     "assert_contains": "expected text"}

Supported assertions are ``assert_contains``, ``assert_not_contains`` and
``assert_equals``. Every evidence item is bound to the durable, audited tool
call that produced it (call id, exact tool name/arguments/output and output
digest), not to a self-attested claim. Backend provenance is reported from the
operator's ``record()``: a ``FakeBrowserPage``/mock backend is always reported
with ``live_browser=False``, and an uninspectable backend fails closed as
``unverified``. No live browser is launched or claimed by this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from ...tools import operator_tools
from ...contracts import ToolCall
from ..verification import digest

__all__ = ["SEED", "make_handler"]

# Same keys as the _SEEDS/SPECIALIST_SPECS rows in leo.agents.registry.
SEED: dict[str, Any] = {
    "id": "browser-tester",
    "name": "Browser Tester",
    "purpose": (
        "Run deterministic UI checks through the browser operator tools and "
        "report URL, page snapshot and assertion evidence with honest backend provenance."
    ),
    "capabilities": ["browser_checks", "dom_snapshot", "assertion_reporting", "evidence_collection"],
    "tool_permissions": ["browser_navigate", "browser_extract_text"],
    "model_preferences": ["tools", "structured"],
    "context_budget": 16000,
    "task_limit": 2,
    "metadata": {"tier": "operator", "backend_honesty": "explicit", "checks": "deterministic"},
}

_ASSERTIONS = ("assert_contains", "assert_not_contains", "assert_equals")
_CONTEXT_KEYS = ("url", "selector", "expected", "expect", "expected_text") + _ASSERTIONS
_FAKE_BACKEND_TOKENS = ("fake", "mock", "stub", "inmemory", "unverified", "unknown")
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def _parse_check(objective: Any, context: Mapping[str, Any] | None) -> dict[str, Any]:
    check: dict[str, Any] = {}
    if isinstance(objective, Mapping):
        check.update(objective)
    elif isinstance(objective, str):
        text = objective.strip()
        parsed: Any = None
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
        if isinstance(parsed, Mapping):
            check.update(parsed)
        else:
            check["objective"] = text
    if isinstance(context, Mapping):
        for key in ("check", "browser_check", "ui_check"):
            nested = context.get(key)
            if isinstance(nested, Mapping):
                check.update(nested)
        for key in _CONTEXT_KEYS:
            if key in context:
                check[key] = context[key]

    if not check.get("url") and isinstance(check.get("objective"), str):
        match = _URL_RE.search(check["objective"])
        if match:
            check["url"] = match.group(0).rstrip(".,);")
    for key in ("expected", "expect", "expected_text", "assertion"):
        if check.get(key) is not None and not any(check.get(name) is not None for name in _ASSERTIONS):
            check["assert_contains"] = check[key]
            break
    return check


def _result_value(result: Any) -> Any:
    if not getattr(result, "success", True):
        raise RuntimeError(str(getattr(result, "error", None) or "browser tool call failed"))
    output = getattr(result, "output", result)
    if isinstance(output, Mapping):
        for key in ("result", "value", "text"):
            if key in output:
                return output[key]
    return output


def _tool_evidence(
    result: Any,
    call: ToolCall,
    *,
    claim: str,
    backend: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """Bind a claim to the audited tool call and output that produced it.

    The verifier checks the call id/name/arguments against the durable
    ``tool_calls`` row and recomputes the content digest of ``output``; a bare
    claim is never treated as evidence.
    """
    output = getattr(result, "output", result)
    recorded = {
        "id": str(getattr(result, "call_id", None) or call.id),
        "tool": str(getattr(result, "tool", None) or call.name),
        "arguments": dict(call.arguments),
        "output": output,
        "digest": digest(output),
    }
    return {
        "source": "tool",
        "tool_call": recorded,
        "call_id": recorded["id"],
        "tool": recorded["tool"],
        "claim": claim,
        "digest": recorded["digest"],
        "data": output,
        "backend": backend["backend"],
        "live": backend["live"],
        **extra,
    }


def _backend_status(services: Any, context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Report operator provenance; never infer a live browser from a tool result."""
    operator: Any = None
    for source in (context, services):
        if not isinstance(source, Mapping):
            continue
        for key in ("browser_operator", "browser"):
            candidate = source.get(key)
            if candidate is not None and hasattr(candidate, "record"):
                operator = candidate
                break
        if operator is not None:
            break
    if operator is None:
        browsers = getattr(operator_tools, "_BROWSERS", None)
        if isinstance(browsers, Mapping) and len(browsers) == 1:
            operator = next(iter(browsers.values()))
    if operator is None:
        return {
            "backend": "unverified", "live": False,
            "reason": "no browser operator available for provenance inspection",
        }
    try:
        record = operator.record()
        metadata = dict(getattr(record, "metadata", {}) or {})
        backend = str(metadata.get("backend") or "unknown")
        status = str(getattr(record, "status", "") or "").lower()
    except Exception as exc:  # a broken provenance probe must not become a live claim
        return {"backend": "unverified", "live": False, "reason": f"{type(exc).__name__}: {exc}"}
    live = (
        status == "configured"
        and backend != "unknown"
        and not any(token in backend.lower() for token in _FAKE_BACKEND_TOKENS)
    )
    return {
        "backend": backend,
        "live": live,
        "live_verified": bool(metadata.get("live_verified", False)),
        "reason": None if live else f"backend {backend!r} is not a verified live browser",
    }


def make_handler(services: Any = None) -> Callable[..., Any]:
    """Build the browser-tester worker.

    ``services`` may supply ``browser_operator`` (or ``browser``) for backend
    provenance and, for direct use outside the agent engine, ``tools`` with an
    async ``execute(ToolCall)`` method. Inside the engine, the scoped tool view
    passed in ``tools``/``context['tools']`` takes precedence, so the agent can
    never escape its declared tool grants.
    """

    async def handler(objective: Any, context: Any = None, tools: Any = None, **_: Any) -> dict[str, Any]:
        check = _parse_check(objective, context if isinstance(context, Mapping) else None)
        url = check.get("url")
        kind = next((name for name in _ASSERTIONS if check.get(name) is not None), None)
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("browser check requires an absolute http(s) 'url'")
        if kind is None:
            raise ValueError("browser check requires one of: " + ", ".join(_ASSERTIONS))

        tool_view = tools or (context.get("tools") if isinstance(context, Mapping) else None)
        if tool_view is None and isinstance(services, Mapping):
            tool_view = services.get("tools") or services.get("executor")
        if tool_view is None or not hasattr(tool_view, "execute"):
            raise RuntimeError("browser tester requires a tool executor with execute(ToolCall)")

        selector = str(check.get("selector") or "body")
        expected = str(check[kind])
        actor = str(context.get("actor") or "browser-tester") if isinstance(context, Mapping) else "browser-tester"

        nav_call = ToolCall(name="browser_navigate", arguments={"url": url}, actor=actor)
        navigate = await tool_view.execute(nav_call)
        final_url = str(_result_value(navigate))
        snapshot_call = ToolCall(
            name="browser_extract_text", arguments={"selector": selector}, actor=actor
        )
        snapshot_call_result = await tool_view.execute(snapshot_call)
        snapshot = str(_result_value(snapshot_call_result))

        if kind == "assert_contains":
            passed = expected in snapshot
        elif kind == "assert_not_contains":
            passed = expected not in snapshot
        else:
            passed = snapshot.strip() == expected

        backend = _backend_status(services, context if isinstance(context, Mapping) else None)
        evidence = [
            _tool_evidence(navigate, nav_call, backend=backend, claim=final_url,
                           url=final_url, requested_url=url),
            _tool_evidence(snapshot_call_result, snapshot_call, backend=backend, claim=snapshot,
                           snapshot=snapshot, selector=selector),
            # The assertion is derived from the extract_text call above; bind it
            # to that same audited output rather than self-attesting a result.
            _tool_evidence(snapshot_call_result, snapshot_call, backend=backend, claim=expected,
                           assertion={"kind": kind, "expected": expected, "passed": passed},
                           passed=passed),
        ]
        output = {
            "url": final_url,
            "selector": selector,
            "snapshot": snapshot,
            "assertion": {"kind": kind, "expected": expected, "passed": passed},
            "check_passed": passed,
            "backend": backend["backend"],
            "live_browser": backend["live"],
            "backend_reason": backend.get("reason"),
        }
        result: dict[str, Any] = {"output": output, "evidence": evidence, "success": passed}
        if not passed:
            result["error"] = f"assertion failed: {kind} {expected!r}"
        return result

    return handler
