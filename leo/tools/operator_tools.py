"""Conversation-callable browser and computer-use tools.

These adapters expose a curated, permission-gated subset of the existing
``BrowserOperator`` and ``ComputerOperator`` surfaces to the chat tool loop.
They deliberately reuse the same PermissionManager and audit trail as every
other tool, so the normal ``tool.started`` / ``tool.finished`` /
``approval.required`` pipeline applies unchanged.

The browser operator keeps one persistent operator per actor (and therefore a
persistent Playwright page/profile) so a navigate → extract → click sequence
behaves like a single browsing session.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..contracts import RiskLevel, ToolSpec
from ..integrations.browser import BrowserOperator
from ..integrations.computer import ComputerOperator

_BROWSERS: dict[str, BrowserOperator] = {}


async def _navigate(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Navigate, granting loopback only for a live, caller-owned preview URL.

    The preview registry (not the caller) decides whether the URL qualifies;
    arbitrary 127.0.0.1/localhost addresses, other actors' tokens and replayed
    or closed previews never reach the browser's private-host allowance.
    """
    url = str(args["url"])
    from .preview_tools import canonical_preview_url, preview_open, registered_preview

    preview = registered_preview(url, ctx)
    if preview is not None:
        return await preview_open({"url": canonical_preview_url(preview)}, ctx)
    operator = _browser_operator(ctx)
    return await operator.navigate(url)


_BROWSER_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="browser_navigate",
            description=(
                "Open an http(s) URL in the local browser. Private/loopback hosts are "
                "blocked, except the exact active URL returned by preview_local_file, "
                "which is opened with a registered-token loopback allowance."
            ),
            category="browser",
            risk=RiskLevel.MEDIUM,
            permissions=["network", "browser"],
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "minLength": 1}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
        _navigate,
    ),
    (
        ToolSpec(
            name="browser_extract_text",
            description="Extract visible text from the current browser page, optionally limited to a CSS selector.",
            category="browser",
            risk=RiskLevel.LOW,
            permissions=["browser"],
            input_schema={
                "type": "object",
                "properties": {"selector": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
        ),
        lambda args, op: op.extract_text(args.get("selector") or "body"),
    ),
    (
        ToolSpec(
            name="browser_screenshot",
            description="Capture the current browser page to a PNG inside the LEO workspace.",
            category="browser",
            risk=RiskLevel.MEDIUM,
            permissions=["browser", "filesystem"],
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
        ),
        lambda args, op: op.screenshot(args.get("path")),
    ),
    (
        ToolSpec(
            name="browser_click",
            description="Click an element on the current browser page by CSS selector.",
            category="browser",
            risk=RiskLevel.MEDIUM,
            permissions=["browser"],
            input_schema={
                "type": "object",
                "properties": {"selector": {"type": "string", "minLength": 1}},
                "required": ["selector"],
                "additionalProperties": False,
            },
        ),
        lambda args, op: op.click(args["selector"]),
    ),
    (
        ToolSpec(
            name="browser_fill",
            description="Fill a form field on the current browser page by CSS selector.",
            category="browser",
            risk=RiskLevel.MEDIUM,
            permissions=["browser"],
            input_schema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "minLength": 1},
                    "value": {"type": "string"},
                },
                "required": ["selector", "value"],
                "additionalProperties": False,
            },
        ),
        lambda args, op: op.fill(args["selector"], args["value"]),
    ),
]

_COMPUTER_SPECS: list[tuple[ToolSpec, Any]] = [
    (
        ToolSpec(
            name="computer_notify",
            description="Show a macOS desktop notification.",
            category="computer",
            risk=RiskLevel.LOW,
            permissions=["computer"],
            # Real dry-run path: the executor must call the handler so policy is
            # evaluated and a truthful no-execution result is returned.
            supports_dry_run=True,
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "minLength": 1},
                    "title": {"type": "string"},
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        ),
        lambda args, op: op.notify(args["message"], title=args.get("title") or "LEO"),
    ),
    (
        ToolSpec(
            name="computer_screenshot",
            description="Capture the macOS screen to a PNG inside the LEO workspace.",
            category="computer",
            risk=RiskLevel.MEDIUM,
            permissions=["computer", "filesystem"],
            supports_dry_run=True,
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        lambda args, op: op.screenshot(args["path"]),
    ),
    (
        ToolSpec(
            name="computer_clipboard_read",
            description="Read the macOS clipboard as text.",
            category="computer",
            risk=RiskLevel.MEDIUM,
            permissions=["computer"],
            supports_dry_run=True,
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda args, op: op.clipboard_read(),
    ),
    (
        ToolSpec(
            name="computer_clipboard_write",
            description="Write text to the macOS clipboard.",
            category="computer",
            risk=RiskLevel.MEDIUM,
            permissions=["computer"],
            supports_dry_run=True,
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        lambda args, op: op.clipboard_write(args["text"]),
    ),
]


def _browser_operator(ctx: dict[str, Any]) -> BrowserOperator:
    actor = str(ctx.get("actor") or "browser")
    if actor not in _BROWSERS:
        settings = ctx["settings"]
        workspaces = Path(settings.resolve(settings.paths.workspaces))
        _BROWSERS[actor] = BrowserOperator(
            ctx["permissions"],
            actor=actor,
            profile_dir=workspaces / "browser-profile",
            base_dir=settings.root,
            timeout=20.0,
        )
    return _BROWSERS[actor]


# Live desktop actions require an explicit settings opt-in. No platform check,
# environment variable or caller-supplied ``dry_run: False`` implies consent.
# The canonical path is ``settings.computer.live_actions``; flat aliases are
# accepted so a settings object that has not been extended can still opt in.
_LIVE_COMPUTER_FLAG_PATHS: tuple[tuple[str, ...], ...] = (
    ("computer", "live_actions"),
    ("computer", "enable_live_actions"),
    ("operators", "computer", "live_actions"),
    ("operator", "computer", "live_actions"),
    ("computer_live",),
    ("enable_live_computer",),
)


def _settings_value(settings: Any, path: tuple[str, ...]) -> Any:
    current = settings
    for part in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _live_computer_enabled(settings: Any) -> bool:
    for path in _LIVE_COMPUTER_FLAG_PATHS:
        value = _settings_value(settings, path)
        if value is not None:
            return bool(value)
    return False


def _computer_operator(ctx: dict[str, Any]) -> ComputerOperator:
    requested_dry_run = bool(ctx.get("dry_run", True))
    live_allowed = _live_computer_enabled(ctx.get("settings")) and not requested_dry_run
    return ComputerOperator(
        ctx["permissions"],
        actor=str(ctx.get("actor") or "computer"),
        dry_run=not live_allowed,
        timeout=10.0,
    )


def _resolve_output_path(ctx: dict[str, Any], value: str | None, default: str) -> str:
    """Resolve a screenshot target inside the workspace, never process CWD."""
    settings = ctx["settings"]
    root = Path(settings.resolve(settings.paths.workspaces)).resolve()
    candidate = Path(value or default)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("screenshot path must stay inside the workspace")
    return str(resolved)


def register_operator_tools(
    registry: Any,
    settings: Any,
    permissions: Any,
    *,
    include_computer: bool | None = None,
) -> list[str]:
    """Register browser tools and (on macOS) computer tools with ``registry``.

    Returns the list of registered tool names. Handlers are closures over the
    executor-supplied ``ctx`` so every action still passes through the shared
    permission manager and audit log.
    """
    registered: list[str] = []
    specs = list(_BROWSER_SPECS)
    if include_computer if include_computer is not None else sys.platform == "darwin":
        specs.extend(_COMPUTER_SPECS)

    for spec, invoke in specs:
        is_browser = spec.category == "browser"
        is_screenshot = spec.name.endswith("screenshot")

        async def handler(args: dict[str, Any], ctx: dict[str, Any], _invoke=invoke,
                          _browser=is_browser, _shot=is_screenshot) -> Any:
            operator = _browser_operator(ctx) if _browser else _computer_operator(ctx)
            call_args = dict(args)
            if _shot and call_args.get("path"):
                call_args["path"] = _resolve_output_path(ctx, call_args["path"], "browser-shot.png")
            if _invoke is _navigate:
                # Navigation needs the full tool context (actor, settings and
                # events) to check the preview token registry before deciding
                # whether a loopback allowance is warranted.
                result = await _invoke(call_args, ctx)
            else:
                result = await _invoke(call_args, operator)
            if hasattr(result, "model_dump"):
                result = result.model_dump(mode="json")
            if isinstance(result, (str, bytes, int, float)) or result is None:
                return {"success": True, "result": result}
            if isinstance(result, dict) and "success" not in result:
                # Operator results carry ``executed``/``dry_run``. A preview that
                # touched no host is not a successful action; do not relabel it.
                if "executed" in result or "dry_run" in result:
                    return {**result, "success": bool(result.get("executed"))}
                return {"success": True, **result}
            return result

        registry.register(spec, handler)
        registered.append(spec.name)

    # ``settings``/``permissions`` are intentionally not captured: the executor
    # injects both into ctx per call, keeping handlers instance-independent.
    return registered


__all__ = ["register_operator_tools"]
