"""Built-in slash commands implemented against real LEO services.

Nothing here calls a model: every handler either reads durable state through an
existing service or performs a mutation through an existing, audited path. The
dispatcher in :mod:`leo.commands.registry` runs ``PermissionManager.authorize``
before the handler, so the risk declared on each spec is enforced for chat, CLI,
API and gateway callers alike.

Commands LEO cannot honestly perform are registered as *unavailable* specs with
a reason. They stay visible in ``/help`` and ``GET /commands`` but dispatch
returns an explicit ``unsupported`` result instead of pretending to work.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import db
from ..contracts import MemoryCategory, MemoryKind, RiskLevel, ToolCall, new_id, utcnow
from ..security.permissions import ApprovalRequired
from ..tools import conversation_tools
from .registry import (
    ALL_SURFACES,
    ArgumentSpec,
    CHAT_SURFACES,
    CommandContext,
    CommandRegistry,
    CommandResult,
    CommandSpec,
)

# Read-only command output is bounded: a slash result travels into chat panels
# and SSE frames, so one command must never stream an unbounded table.
DEFAULT_LIMIT = 20
MAX_LIMIT = 100
MAX_MESSAGE = 12_000

# Tool/gateway payload topics contain user content and tool output. Without an
# authenticated operator principal they are filtered out of /events and /debug.
SENSITIVE_TOPICS = ("tool.", "gateway.")


# ---------------------------------------------------------------------------
# helpers


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _dump(item) for key, item in value.items()}
    return value


def _truncate(text: str, limit: int = MAX_MESSAGE) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "\n… [truncated]"


def _token_limit(ctx: CommandContext, default: int = DEFAULT_LIMIT, cap: int = MAX_LIMIT) -> int:
    raw = ctx.option("--limit", "-n")
    if raw is None and ctx.argv and ctx.argv[-1].isdigit():
        raw = ctx.argv[-1]
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    return max(1, min(value, cap))


def _kv(argv: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Split free words from ``key=value`` options (no shell interpolation)."""
    words: list[str] = []
    values: dict[str, str] = {}
    for token in argv:
        if "=" in token and not token.startswith("--"):
            key, value = token.split("=", 1)
            if key:
                values[key.strip()] = value
        else:
            words.append(token)
    return words, values


def _redact(ctx: CommandContext, text: str, limit: int = MAX_MESSAGE) -> str:
    redact = getattr(ctx.app, "_redact", None)
    try:
        return str(redact(text))[:limit] if callable(redact) else str(text)[:limit]
    except Exception:  # noqa: BLE001 - redaction must never break a response
        return str(text)[:limit]


async def _audit(
    ctx: CommandContext,
    action: str,
    resource: str,
    reason: str,
    *,
    risk: RiskLevel | str = RiskLevel.LOW,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Record the durable side effect itself, beyond the dispatch decision."""
    await db.execute(
        "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (
            utcnow(), ctx.actor, action, resource,
            RiskLevel(risk).value, "allowed", reason,
            json.dumps(dict(metadata or {}), default=str),
        ),
    )


async def _check_risk(
    ctx: CommandContext, action: str, resource: str, risk: RiskLevel,
    arguments: dict[str, Any],
) -> CommandResult | None:
    """Finer-grained gate for a mutating subcommand inside a mixed-risk spec.

    The dispatcher checks the spec's baseline risk. A single spec can contain
    both read-only and mutating subcommands (``/memory``, ``/agents``), so the
    mutating branch makes its own decision through the same PermissionManager.
    """
    permissions = getattr(ctx.app, "permissions", None)
    if permissions is None or not hasattr(permissions, "authorize"):
        return None
    decision = await permissions.authorize(
        ctx.actor, action, resource, risk,
        approval_id=ctx.approval_id, arguments=arguments,
        command=action, surface=ctx.surface, channel=ctx.channel,
    )
    if getattr(decision, "requires_approval", False):
        return CommandResult.pending(
            ctx.raw.split()[0].lstrip("/") or action,
            f"Approval required for {action} (approval_id={decision.approval_id}). "
            "An operator can run /approve <id> in the CLI, then retry with "
            f"--approval {decision.approval_id}.",
            decision.approval_id,
        )
    if not getattr(decision, "allowed", False):
        return CommandResult.denied(action, f"Permission denied: {decision.reason}")
    return None


async def _check_high_risk(
    ctx: CommandContext, action: str, resource: str, arguments: dict[str, Any]
) -> CommandResult | None:
    return await _check_risk(ctx, action, resource, RiskLevel.HIGH, arguments)


async def _emit(app: Any, topic: str, **payload: Any) -> None:
    emit = getattr(getattr(app, "events", None), "emit", None)
    if callable(emit):
        try:
            await emit(topic, **payload)
        except Exception:  # noqa: BLE001 - observability cannot fail a mutation
            pass


def _require_service(ctx: CommandContext, name: str) -> Any:
    service = getattr(ctx.app, name, None)
    if service is None:
        raise RuntimeError(f"LEO service {name!r} is not available")
    return service


def _operator(ctx: CommandContext) -> bool:
    registry = ctx.registry
    return (
        ctx.principal in registry.operator_principals
        and ctx.surface in registry.operator_surfaces
    )


def _summary(
    ctx: CommandContext, command: str, message: str, data: dict[str, Any] | None = None,
    *, limit: int = MAX_MESSAGE,
) -> CommandResult:
    return CommandResult.ok(command, _redact(ctx, _truncate(message, limit), limit), data or {})


# ---------------------------------------------------------------------------
# help / identity / health


async def help_command(ctx: CommandContext) -> CommandResult:
    filter_text = ""
    page = 1
    for token in ctx.argv:
        if token in ("--page", "-p"):
            continue
        if token.isdigit() and not filter_text:
            page = int(token)
        elif token.startswith("-"):
            continue
        elif not filter_text:
            filter_text = token
    catalog = ctx.registry.help(
        filter_text or None, principal=ctx.principal, surface=ctx.surface, page=page
    )
    return _summary(ctx, "help", catalog["text"], catalog, limit=25_000)


async def commands_command(ctx: CommandContext) -> CommandResult:
    page = ctx.argv[0] if ctx.argv and ctx.argv[0].isdigit() else ctx.option("--page", "-p", default="1")
    catalog = ctx.registry.help(
        principal=ctx.principal, surface=ctx.surface, page=page or 1, per_page=100
    )
    return _summary(ctx, "commands", catalog["text"], catalog, limit=25_000)


async def status_command(ctx: CommandContext) -> CommandResult:
    app = ctx.app
    health = await app.health()
    pending = await app.permissions.pending_approvals()
    reported = getattr(getattr(app, "router", None), "status", None)
    model = reported() if callable(reported) else app.settings.model_status()
    reasoning: dict[str, Any] = {}
    selector = getattr(app, "reasoning_selection", None)
    if callable(selector):
        try:
            selection = await selector(ctx.conversation_id)
            reasoning = {
                key: selection.get(key)
                for key in ("effort", "effort_source", "show", "show_source", "scope")
            }
        except Exception:  # status must survive a broken settings read
            reasoning = {}
    data = {
        "health": health,
        "pending_approvals": len(pending),
        "model": {k: model.get(k) for k in ("status", "simulated", "default", "routable_live_providers") if k in model},
        "reasoning": reasoning,
    }
    message = (
        f"LEO {health.get('status', 'unknown')} (v{health.get('version', '?')}); "
        f"tools={health.get('tools', '?')}, skills={health.get('skills', '?')}, "
        f"agents={health.get('agents', '?')}; pending approvals={len(pending)}; "
        f"model={model.get('status', 'unknown')}"
    )
    if reasoning:
        display = "on" if reasoning.get("show") else "off"
        message += (
            f"; reasoning={reasoning.get('effort') or 'unset'}"
            f" ({reasoning.get('effort_source') or 'unset'}), display={display}"
        )
    return _summary(ctx, "status", message, data)


async def whoami_command(ctx: CommandContext) -> CommandResult:
    allowed = ctx.registry.catalog(principal=ctx.principal, surface=ctx.surface)
    data = {
        "principal": ctx.principal,
        "surface": ctx.surface,
        "channel": ctx.channel,
        "actor": ctx.actor,
        "admin": ctx.principal in ctx.registry.admin_principals,
        "operator": _operator(ctx),
        "allowed_commands": [item["name"] for item in allowed if item["allowed_for_caller"]],
    }
    message = (
        f"principal={ctx.principal}, surface={ctx.surface}, channel={ctx.channel}, "
        f"actor={ctx.actor}; {len(data['allowed_commands'])} commands permitted"
    )
    if ctx.surface == "chat":
        message += "; chat actors cannot resolve approvals"
    return _summary(ctx, "whoami", message, data)


async def version_command(ctx: CommandContext) -> CommandResult:
    health = await ctx.app.health()
    message = f"LEO v{health.get('version', '0.1.0')} (status={health.get('status', 'unknown')})"
    return _summary(ctx, "version", message, {"version": health.get("version")})


_CONFIG_KEYS = {
    "app_name": lambda s: s.app_name,
    "environment": lambda s: s.environment,
    "default_provider": lambda s: s.models.default,
    "fallback_provider": lambda s: s.models.fallback,
    "reasoning_effort": lambda s: s.models.reasoning_effort,
    "reasoning_show": lambda s: s.models.reasoning_show,
    "db_path": lambda s: str(s.db_path),
    "workspace_only": lambda s: s.security.workspace_only,
    "allow_medium_risk": lambda s: s.security.allow_medium_risk,
    "high_risk_requires_approval": lambda s: s.security.high_risk_requires_approval,
    "approval_timeout_seconds": lambda s: s.security.approval_timeout_seconds,
    "max_tool_calls_per_task": lambda s: s.security.max_tool_calls_per_task,
    "max_agent_depth": lambda s: s.security.max_agent_depth,
    "tick_seconds": lambda s: s.automation.tick_seconds,
    "max_concurrent_tasks": lambda s: s.automation.max_concurrent_tasks,
    "server.host": lambda s: s.server.host,
    "server.port": lambda s: s.server.port,
    "live_actions": lambda s: s.computer.live_actions,
}


async def config_command(ctx: CommandContext) -> CommandResult:
    settings = ctx.app.settings
    sub = ctx.argv[0].lower() if ctx.argv else "show"
    if sub in ("set", "unset"):
        return CommandResult.unsupported(
            "config",
            "LEO does not support runtime configuration writes through slash commands; "
            "edit leo.yaml and restart. No value was changed.",
        )
    if sub == "get":
        key = ctx.require(1, "config key")
        if key not in _CONFIG_KEYS:
            raise ValueError(f"unknown config key {key!r}; available: {', '.join(sorted(_CONFIG_KEYS))}")
        value = _CONFIG_KEYS[key](settings)
        return _summary(ctx, "config", f"{key} = {value}", {"key": key, "value": value})
    values = {key: getter(settings) for key, getter in sorted(_CONFIG_KEYS.items())}
    return _summary(ctx, "config", "Read-only LEO configuration snapshot", values)


def _provider_list(providers: Mapping[str, Any], current: str, *, columns: int = 3) -> str:
    """Models in columns, the current one marked — scannable in a chat panel.

    Each provider entry carries exactly one model, so listing the provider name
    AND the model name doubled every row. One name per cell, ``*`` on the
    default, disabled entries named once at the end.
    """
    enabled = [name for name, cfg in sorted(providers.items()) if getattr(cfg, "enabled", True)]
    disabled = [name for name, cfg in sorted(providers.items()) if not getattr(cfg, "enabled", True)]
    width = min(28, max((len(name) for name in enabled), default=8))
    rows: list[str] = []
    step = max(1, columns)
    for index in range(0, len(enabled), step):
        cells = [
            ("* " if name == current else "  ") + name.ljust(width)
            for name in enabled[index:index + step]
        ]
        rows.append(" ".join(cells).rstrip())
    if disabled:
        rows.append("")
        rows.append("hidden: " + ", ".join(disabled))
    return "\n".join(rows) or "(none)"


def _model_picker(
    providers: Mapping[str, Any], current: str, *, page: int = 0,
    page_size: int = 8, columns: int = 2,
) -> dict[str, Any]:
    """Inline keyboard: one button per model, the default marked with a tick.

    A tap sends ``/model <name>`` back through the queue, so the switch still
    runs in the process that owns settings — the gateway only relays the tap.
    Models are paginated (``page_size`` per page) with Back/Next nav buttons
    whose callback_data is ``mdl:pg:<n>``.
    """
    enabled = [name for name, cfg in sorted(providers.items()) if getattr(cfg, "enabled", True)]
    total = len(enabled)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, pages - 1))
    chunk = enabled[page * page_size:(page + 1) * page_size]
    rows: list[list[dict[str, str]]] = []
    step = max(1, columns)
    for index in range(0, len(chunk), step):
        row = []
        for name in chunk[index:index + step]:
            label = ("\u2713 " if name == current else "") + name
            row.append({"text": label[:48], "callback_data": f"mdl:{name}"[:64]})
        rows.append(row)
    nav: list[dict[str, str]] = []
    if page > 0:
        nav.append({"text": "\u2190 Back", "callback_data": f"mdl:pg:{page - 1}"})
    if page < pages - 1:
        nav.append({"text": "Next \u2192", "callback_data": f"mdl:pg:{page + 1}"})
    if nav:
        rows.append(nav)
    return {"inline_keyboard": rows}


def _model_listing_message(
    providers: Mapping[str, Any], current: str, *, page: int = 0,
    page_size: int = 8, columns: int = 3,
) -> str:
    """Bare ``/model`` body: current line, the model grid, one footer line."""
    cfg_current = providers.get(current)
    current_model = str(getattr(cfg_current, "model", "") or current)
    enabled_names = [name for name, cfg in sorted(providers.items())
                     if getattr(cfg, "enabled", True)]
    total = len(enabled_names)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, pages - 1))
    chunk = enabled_names[page * page_size:(page + 1) * page_size]
    # Build a providers view restricted to this page so _provider_list only
    # renders the current page's models.
    page_providers = {name: providers[name] for name in chunk}
    header = f"Current: * {current_model}  ({total} models, page {page + 1}/{pages})"
    body = _provider_list(page_providers, current, columns=columns)
    # Hidden providers belong to the catalogue, not to the page being shown:
    # restricting the list to one page must not make them disappear.
    hidden = [name for name, cfg in sorted(providers.items())
              if not getattr(cfg, "enabled", True)]
    footer = "`/model <name>` switch · `/model list <n>` page · `/model status` details"
    parts = [header, "", body]
    if hidden:
        parts += ["", "hidden: " + ", ".join(hidden)]
    parts += ["", footer]
    return "\n".join(parts)


def _join_body(header: str, body: str) -> str:
    """Header, blank line, body, blank line, one-line footer."""
    return "\n".join([header, "", body, "", "`/model <name>` switch · `/model status` details"])


def _persist_default_provider(provider: str) -> tuple[bool, str | None]:
    """Rewrite only the ``models.default`` scalar, preserving the rest of leo.yaml.

    A full YAML round-trip would expand the catalog's anchors and drop the
    operator's comments, so this edits the single line in place.
    """
    cfg_path = Path(os.getenv("LEO_CONFIG", "leo.yaml"))
    try:
        original = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return False, "leo.yaml is not readable"
    lines = original.splitlines()
    in_models = False
    replaced = False
    for index, line in enumerate(lines):
        if not in_models:
            if line.rstrip() == "models:":
                in_models = True
            continue
        if line and not line[0].isspace():
            break  # next top-level section
        match = re.match(r"^(\s+)default\s*:", line)
        if match:
            lines[index] = f"{match.group(1)}default: {provider}"
            replaced = True
            break
    if not replaced:
        return False, "no models.default entry found in leo.yaml"
    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    try:
        cfg_path.with_name(cfg_path.name + ".bak").write_text(original, encoding="utf-8")
        temporary = cfg_path.with_name(cfg_path.name + ".tmp")
        temporary.write_text(updated, encoding="utf-8")
        os.replace(temporary, cfg_path)
    except OSError:
        return False, "could not write leo.yaml"
    return True, None


async def model_command(ctx: CommandContext) -> CommandResult:
    app = ctx.app
    reported = getattr(getattr(app, "router", None), "status", None)
    status = reported() if callable(reported) else app.settings.model_status()
    providers = app.settings.models.providers
    current = app.settings.models.default
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub == "list":
        page = 0
        if len(ctx.argv) > 1:
            try:
                page = int(ctx.argv[1])
            except (ValueError, TypeError):
                page = 0
        data = {
            name: {"model": cfg.model, "kind": cfg.kind, "enabled": cfg.enabled,
                   "default": name == current}
            for name, cfg in sorted(providers.items())
        }
        return _summary(
            ctx, "model",
            _model_listing_message(providers, current, page=page),
            {**data, "reply_markup": _model_picker(providers, current, page=page)},
        )
    if sub == "status":
        data = dict(status)
        data.setdefault("configured_default", current)
        return _summary(
            ctx, "model",
            f"model status={status.get('status', 'unknown')}, default={current}",
            data,
        )
    if sub == "reset":
        return CommandResult.unsupported(
            "model",
            "There is no implicit reset. Switch explicitly with /model <name>, "
            "or set models.default in leo.yaml and restart; no setting was changed.",
            data={"status": status},
        )
    # /model <name> switches the live default the same way the web UI control
    # does: the router reads settings.models.default per turn, so it applies to
    # the next message, and the choice is persisted so it survives a restart.
    target = providers.get(sub)
    if target is None:
        known = ", ".join(sorted(providers)) or "(none)"
        raise ValueError(f"unknown provider {sub!r}; configured: {known}")
    if not target.enabled:
        return CommandResult.unsupported(
            "model", f"provider {sub!r} is disabled; nothing was changed.",
            data={"status": status},
        )
    if not target.model:
        return CommandResult.unsupported(
            "model", f"provider {sub!r} has no model; nothing was changed.",
            data={"status": status},
        )
    app.settings.models.default = sub
    persisted, warning = _persist_default_provider(sub)
    await _audit(ctx, "slash.model.switch", sub, "default provider switched via slash")
    message = f"Default model: {sub} -> {target.model} (applies to the next turn)"
    if persisted:
        message += "; saved to leo.yaml"
    else:
        message += f"; NOT persisted ({warning or 'unknown reason'})"
    return _summary(
        ctx, "model", message,
        {"default_provider": sub, "model": target.model,
         "persisted": persisted, "warning": warning},
    )


def _reasoning_keyboard(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Inline keyboard for the full /reasoning menu.

    Every button sends a plain action token back through the gateway; the brain
    turns it into the same slash command a typed request uses. No tap writes
    state directly, so CLI, API and Telegram share one mutation path.
    """
    from ..models.reasoning import EFFORT_LADDER

    rows: list[list[dict[str, str]]] = []
    active = selection.get("effort")
    levels = list(EFFORT_LADDER)
    for index in range(0, len(levels), 2):
        row: list[dict[str, str]] = []
        for level in levels[index:index + 2]:
            label = ("✓ " if level == active else "") + level
            row.append({"text": label[:48], "callback_data": f"rsn:level:{level}"[:64]})
        rows.append(row)
    scope = str(selection.get("scope") or "session")
    other = "global" if scope == "session" else "session"
    rows.append([{
        "text": f"Scope: {scope} → {other}"[:48],
        "callback_data": f"rsn:scope:{other}"[:64],
    }])
    if selection.get("show"):
        rows.append([{"text": "Hide summaries", "callback_data": "rsn:hide"}])
    else:
        rows.append([{"text": "Show summaries", "callback_data": "rsn:show"}])
    rows.append([{
        "text": "Detail: detailed" if selection.get("full") else "Detail: concise",
        "callback_data": "rsn:detail",
    }])
    rows.append([{"text": "Reset session overrides", "callback_data": "rsn:reset"}])
    rows.append([{"text": "Reset global default", "callback_data": "rsn:reset-global"}])
    return {"inline_keyboard": rows}


def _reasoning_message(app: Any, selection: Mapping[str, Any]) -> str:
    """Truthful status: effective value, its source, edit scope and display."""
    effort = selection.get("effort") or "unset"
    source = selection.get("effort_source") or "unset"
    show = "summaries on" if selection.get("show") else "summaries off"
    detail = "detailed" if selection.get("full") else "concise"
    lines = [
        "🧠 Reasoning",
        f"Active effort: {effort} ({source})",
        f"Display: {show} ({selection.get('show_source') or 'config'}); detail: {detail}",
        f"Edit scope: {selection.get('scope') or 'session'}",
        "Levels: " + ", ".join(selection.get("levels") or []),
    ]
    provider = _reasoning_provider_line(app, selection)
    if provider:
        lines.append(provider)
    return "\n".join(lines)


def _reasoning_provider_line(app: Any, selection: Mapping[str, Any]) -> str:
    """What the configured default provider will do with the next request."""
    try:
        models = app.settings.models
        name = str(getattr(models, "default", "") or "")
        cfg = (getattr(models, "providers", {}) or {}).get(name)
        provider = (getattr(getattr(app, "router", None), "providers", {}) or {}).get(name)
        parts = [f"Provider: {name or 'unset'}"]
        if cfg is not None:
            model = str(getattr(cfg, "model", "") or "")
            if model:
                parts.append(model)
        describe = getattr(provider, "describe_reasoning", None)
        preview_note = ""
        if callable(describe):
            preview = describe(selection.get("effort")) or {}
            applied = preview.get("applied")
            style = preview.get("style")
            parts.append(
                f"sends: {applied or 'nothing'}" + (f" ({style})" if style else "")
            )
            preview_note = str(preview.get("note") or "")
            if preview_note:
                parts.append(preview_note)
        elif cfg is not None:
            parts.append(f"style={getattr(cfg, 'reasoning_style', 'auto')}")

        reporter = getattr(getattr(app, "router", None), "reasoning_report", None)
        report = reporter() if callable(reporter) else {}
        requested = report.get("requested") if isinstance(report, Mapping) else None
        if requested:
            applied = report.get("applied")
            last = f"last request: {requested}"
            if applied and applied != requested:
                last += f" → {applied}"
            if report.get("style"):
                last += f" ({report['style']})"
            parts.append(last)
            note = str(report.get("note") or "")
            if note and note != preview_note:
                parts.append(note)
        return " · ".join(parts)
    except Exception:  # status rendering must never break the menu
        return ""


async def reasoning_command(ctx: CommandContext) -> CommandResult:
    """Show/set reasoning effort, scope and summary display.

    Session changes apply only to the current conversation; global changes are
    persisted in the database and survive a restart. Effort is translated to
    each provider's own parameter at request time, and an unsupported level is
    clamped downward and reported rather than silently escalated.
    """
    from ..models.reasoning import EFFORT_LADDER, parse_effort

    app = ctx.app
    state = getattr(app, "reasoning_state", None)
    if state is None:
        return CommandResult.unsupported(
            "reasoning", "Reasoning state service is not available; nothing was changed."
        )
    argv = list(ctx.argv)
    global_flag = "--global" in argv
    session_flag = "--session" in argv
    argv = [a for a in argv if a not in ("--global", "--session")]
    conversation_id = ctx.conversation_id or ""

    async def resolve() -> dict[str, Any]:
        # Read the config default at call time: clearing the global value
        # restores the YAML default, and the menu must show that immediately.
        return await state.resolve(
            conversation_id or None,
            default_effort=getattr(app.settings.models, "reasoning_effort", None),
            default_show=bool(getattr(app.settings.models, "reasoning_show", False)),
        )

    def respond(message: str, selection: Mapping[str, Any], **extra: Any) -> CommandResult:
        data = dict(selection)
        data.update(extra)
        data["reply_markup"] = _reasoning_keyboard(selection)
        # Every mutation also re-renders the current state. A one-line
        # confirmation alone would leave the human guessing what the menu now
        # says, especially after a scope switch.
        status = _reasoning_message(app, selection)
        text = f"{message}\n\n{status}" if message.strip() else status
        return _summary(ctx, "reasoning", text, data)

    selection = await resolve()
    target_scope = "global" if global_flag else "session" if session_flag else str(
        selection.get("scope") or "session"
    )

    def session_required() -> CommandResult | None:
        """Refuse a session write with no conversation instead of widening it.

        Silently turning a session-scoped tap into a global write would leak
        one user's choice into every other conversation.
        """
        if target_scope != "session" or conversation_id:
            return None
        return CommandResult.unsupported(
            "reasoning",
            "This surface has no conversation id, so a session change cannot be stored. "
            "Use --global to change the global default.",
            data={"reply_markup": _reasoning_keyboard(selection)},
        )

    arg = argv[0].lower() if argv else ""

    # -- read-only menu / status -----------------------------------------
    if arg in ("", "status", "menu", "list"):
        return respond("", selection)

    # -- edit scope -------------------------------------------------------
    if arg in ("scope", "config", "target"):
        desired = argv[1].lower() if len(argv) > 1 else (
            "global" if selection.get("scope") == "session" else "session"
        )
        if desired not in ("global", "session"):
            return CommandResult.unsupported(
                "reasoning", "scope must be global or session", data={
                    "reply_markup": _reasoning_keyboard(selection)})
        if not conversation_id:
            return CommandResult.unsupported(
                "reasoning",
                "This surface has no conversation id, so the edit scope cannot be stored. "
                "Pass --global to edit the global default.",
                data={"reply_markup": _reasoning_keyboard(selection)},
            )
        await state.set_scope(conversation_id, desired)
        selection = await resolve()
        await _audit(ctx, "slash.reasoning.scope", desired, "reasoning edit scope changed")
        return respond(f"Reasoning edit scope: {desired}", selection)
    if arg in ("global", "session"):
        if not conversation_id and arg == "session":
            return CommandResult.unsupported(
                "reasoning", "This surface has no conversation id; session scope is unavailable.",
                data={"reply_markup": _reasoning_keyboard(selection)},
            )
        await state.set_scope(conversation_id, arg)
        selection = await resolve()
        await _audit(ctx, "slash.reasoning.scope", arg, "reasoning edit scope changed")
        return respond(f"Reasoning edit scope: {arg}", selection)

    # -- display ----------------------------------------------------------
    if arg in ("show", "on", "hide", "off"):
        refused = session_required()
        if refused is not None:
            return refused
        show = arg in ("show", "on")
        if target_scope == "global":
            await state.set_global_display(show)
            await _audit(ctx, "slash.reasoning.display.global", str(show),
                         "global reasoning summary display changed")
        else:
            await state.set_display(conversation_id, show=show)
            await _audit(ctx, "slash.reasoning.display", "show" if show else "hide",
                         "reasoning summary display changed")
        selection = await resolve()
        where = "global" if target_scope == "global" else "this session"
        return respond(f"Reasoning summaries: {'on' if show else 'off'} ({where})", selection)
    if arg in ("full", "all", "detailed", "clamp", "collapse", "short", "detail", "concise"):
        if not conversation_id:
            return CommandResult.unsupported(
                "reasoning", "Detail is a per-session display setting; no conversation id here.",
                data={"reply_markup": _reasoning_keyboard(selection)},
            )
        if arg in ("full", "all", "detailed"):
            full = True
        elif arg in ("clamp", "collapse", "short", "concise"):
            full = False
        else:
            full = not bool(selection.get("full"))
        await state.set_display(conversation_id, full=full)
        selection = await resolve()
        await _audit(ctx, "slash.reasoning.display.detail", str(full),
                     "reasoning summary detail changed")
        return respond(f"Reasoning detail: {'detailed' if full else 'concise'}", selection)

    # -- reset ------------------------------------------------------------
    if arg in ("reset", "clear"):
        if target_scope == "global":
            await state.clear_global()
            sync = getattr(app, "sync_reasoning_defaults", None)
            if callable(sync):
                await sync()
            await _audit(ctx, "slash.reasoning.global.reset", "global",
                         "global reasoning settings cleared")
            selection = await resolve()
            return respond("Global reasoning settings cleared (config default applies).", selection)
        if not conversation_id:
            return CommandResult.unsupported(
                "reasoning", "No active conversation to reset.",
                data={"reply_markup": _reasoning_keyboard(selection)},
            )
        await state.clear(conversation_id)
        await _audit(ctx, "slash.reasoning.reset", conversation_id,
                     "session reasoning overrides cleared")
        selection = await resolve()
        return respond("Session reasoning overrides cleared (global default applies).", selection)

    # -- effort -----------------------------------------------------------
    level = parse_effort(arg)
    if level is None:
        return CommandResult.unsupported(
            "reasoning",
            f"unknown reasoning level {arg!r}; valid: {', '.join(EFFORT_LADDER)} "
            "(aliases: off, min, med, hi, xh, extreme). Nothing was changed.",
            data={"reply_markup": _reasoning_keyboard(selection)},
        )

    if target_scope == "session":
        refused = session_required()
        if refused is not None:
            return refused

    if target_scope == "global":
        await state.set_global(level)
        sync = getattr(app, "sync_reasoning_defaults", None)
        if callable(sync):
            await sync()
        await _audit(ctx, "slash.reasoning.global", level, "global reasoning effort set")
        selection = await resolve()
        return respond(
            f"Global reasoning effort: {level} (persisted; applies to this and new sessions)",
            selection, scope="global",
        )

    await state.set_effort(conversation_id, level)
    await _audit(ctx, "slash.reasoning.effort", level, "session reasoning effort set")
    selection = await resolve()
    return respond(
        f"Reasoning effort: {level} (this session only; applies to the next request)",
        selection, scope="session",
    )


async def usage_command(ctx: CommandContext) -> CommandResult:
    router = _require_service(ctx, "router")
    report = await router.usage_report()
    total_in = sum(int(row.get("input_tokens") or 0) for row in report)
    total_out = sum(int(row.get("output_tokens") or 0) for row in report)
    cost = sum(float(row.get("cost") or 0) for row in report)
    data = {"calls": len(report), "input_tokens": total_in, "output_tokens": total_out, "cost": cost,
            "recent": report[:_token_limit(ctx)]}
    message = f"{len(report)} model calls; {total_in} in / {total_out} out tokens; cost={cost:.4f}"
    return _summary(ctx, "usage", message, data)


async def context_command(ctx: CommandContext) -> CommandResult:
    app = ctx.app
    budget_fn = getattr(app, "_context_budget", None)
    budget = budget_fn() if callable(budget_fn) else 1024
    history_limit = getattr(app, "CONVERSATION_HISTORY_LIMIT", 50)
    conversation = ctx.conversation_id
    rows: list[Any] = []
    if conversation:
        rows = await db.fetchall(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id", (conversation,)
        )
    memories = await app.memory.list(limit=1)
    tools = app.registry.list_specs()
    data = {
        "budget_tokens": budget,
        "history_limit": history_limit,
        "stored_turns": len(rows),
        "memory_records": len(await app.memory.list(limit=MAX_LIMIT)),
        "tool_specs": len(tools),
        "conversation_id": conversation,
        "compaction": "durable: /compact summarizes older turns once and replays the summary",
    }
    message = (
        f"context budget={budget} tokens; history limit={history_limit}; "
        f"stored turns={len(rows)}; memory records={data['memory_records']}; "
        f"tools={len(tools)}. Older turns are compacted durably with /compact; "
        f"each turn is also optimized to fit the budget."
    )
    return _summary(ctx, "context", message, data)


# ---------------------------------------------------------------------------
# sessions and local state


async def new_command(ctx: CommandContext) -> CommandResult:
    explicit = ctx.rest().strip()[:200] or None
    # A channel turn is keyed by (channel, subject) — usually the sender id. A
    # bare /new used to insert subject=NULL, which the gateway can never match,
    # so the command created a conversation nothing would ever use. Reuse the
    # subject of the conversation this command arrived on.
    subject = explicit
    if subject is None and ctx.conversation_id:
        row = await db.fetchone("SELECT subject FROM conversations WHERE id=?", (ctx.conversation_id,))
        subject = str(row["subject"]) if row and row["subject"] else None
    conversation_id = new_id("conv")
    await db.execute(
        "INSERT INTO conversations(id,channel,subject,created_at,updated_at) VALUES(?,?,?,?,?)",
        (conversation_id, ctx.channel, subject, utcnow(), utcnow()),
    )
    await _audit(ctx, "slash.new", conversation_id, "new conversation created")
    await _emit(ctx.app, "slash.session.new", conversation_id=conversation_id, actor=ctx.actor)
    if subject is None:
        return _summary(
            ctx, "new",
            f"New conversation {conversation_id} created. The client adapter must switch to this id to use it.",
            {"conversation_id": conversation_id, "subject": subject},
        )
    return _summary(
        ctx, "new",
        f"New conversation {conversation_id} created; this channel switches to it on your next message.",
        {"conversation_id": conversation_id, "subject": subject},
    )


_UNIT_NAME_RE = re.compile(r"^[A-Za-z0-9@._-]+\.service$")


def _own_units() -> list[str]:
    """Units that actually supervise this process, taken from our own cgroup.

    A hardcoded default unit is a guess. On this host LEO runs as three
    siblings (leo-gateway / leo-ui / leo-brain); leo-agent.service is a
    disabled leftover that cannot bind port 8765, so restarting the guessed
    unit restarted nothing and instead kick-started a crash loop each boot.
    /proc/self/cgroup is authoritative about the slice we actually live in.
    """
    units: list[str] = []
    try:
        text = Path("/proc/self/cgroup").read_text()
    except OSError:
        return []
    for line in text.splitlines():
        _, sep, path = line.partition("::")
        if not sep or not path:
            continue
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if _UNIT_NAME_RE.match(name):
            units.append(name)
    return sorted(set(units))


def _unit_belongs_to_install(unit: str, install_root: str) -> bool:
    """True when this unit actually runs code from our checkout.

    FragmentPath is useless here: it is always /etc/systemd/system/foo.service,
    regardless of which checkout the unit launches. WorkingDirectory and
    ExecStart are what name the install.
    """
    if not install_root:
        return False
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["systemctl", "show", "-p", "WorkingDirectory", "-p", "ExecStart", "--value", unit],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if done.returncode != 0:
        return False
    return install_root in (done.stdout or "")


def _active_leo_units() -> list[str]:
    """Active leo* units whose unit file comes from this checkout.

    Gateways, API and brain are separate units, so restarting only the one we
    happen to live in leaves the others serving the previous config.
    """
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["systemctl", "list-units", "--type=service", "--state=active",
             "--no-legend", "--plain", "--no-pager"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0:
        return []
    root = str(_install_root())
    return sorted({
        parts[0] for parts in (line.split() for line in (done.stdout or "").splitlines())
        if parts and _UNIT_NAME_RE.match(parts[0]) and parts[0].startswith("leo")
        and _unit_belongs_to_install(parts[0], root)
    })


def _restart_targets() -> tuple[list[str], str]:
    """Resolve which units a restart must cover, and how they were chosen.

    Precedence: LEO_SERVICE_UNITS > LEO_SERVICE_UNIT > auto-detection, so an
    explicit operator setting is never overridden.
    """
    multi = (os.getenv("LEO_SERVICE_UNITS") or "").strip()
    if multi:
        return [u.strip() for u in multi.split(",") if u.strip()], "LEO_SERVICE_UNITS"
    single = (os.getenv("LEO_SERVICE_UNIT") or "").strip()
    if single:
        return [single], "LEO_SERVICE_UNIT"
    discovered = _active_leo_units()
    for unit in _own_units():
        if unit not in discovered:
            discovered.append(unit)
    if discovered:
        return sorted(set(discovered)), "auto-detected"
    return ["leo-agent.service"], "fallback default"


def _spawn_restart(units: list[str], delay: int) -> bool:
    """Detached child that restarts the units once this reply has landed."""
    if not units or not shutil.which("systemctl"):
        return False
    try:
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell, no user input
            ["/bin/sh", "-c", f"sleep {delay}; systemctl restart {' '.join(units)}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        return False
    return True


async def _schedule_restart(ctx: CommandContext, command: str = "restart") -> bool:
    """Spawn a detached child that restarts the service once this reply lands."""
    units, source = _restart_targets()
    delay = max(1, int(os.getenv("LEO_RESTART_DELAY_SECONDS") or 3))
    if not _spawn_restart(units, delay):
        return False
    await _audit(
        ctx, f"slash.{command}.restart", ",".join(units),
        f"scheduled restart in {delay}s ({source})",
    )
    return True


def _install_root() -> Path:
    """The checkout LEO runs from: env override, else two levels above the package."""
    override = os.getenv("LEO_INSTALL_ROOT")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path) -> tuple[int, str]:
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=120, check=False
    )
    return done.returncode, (done.stdout or done.stderr or "").strip()


async def update_command(ctx: CommandContext) -> CommandResult:
    """Report available updates; install them only when explicitly asked.

    A bare /update is read-only. Installing needs the word "apply", because an
    update rewrites the checkout and restarts the process - never by surprise.
    Every precondition fails closed: not a checkout, dirty tree, no remote, or a
    diverged branch stops before anything is modified.
    """
    root = _install_root()
    apply_now = any(arg.lower() in ("apply", "install", "yes") for arg in ctx.argv)
    if not (root / ".git").exists():
        return _summary(
            ctx, "update",
            f"Not a git checkout ({root}) — update with your package manager.",
            {"root": str(root), "updated": False},
        )

    code, branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    if code != 0 or not branch or branch == "HEAD":
        return _summary(ctx, "update", f"Cannot determine the checked-out branch in {root}.",
                        {"root": str(root), "updated": False})

    code, dirty = _git("status", "--porcelain", cwd=root)
    if code != 0:
        return _summary(ctx, "update", f"Cannot read the working tree: {dirty[:160]}",
                        {"root": str(root), "branch": branch, "updated": False})
    if dirty:
        return _summary(
            ctx, "update",
            f"{len(dirty.splitlines())} uncommitted change(s) in {root}. Commit or stash them "
            "first — an update must never overwrite local work.",
            {"root": str(root), "branch": branch, "dirty_files": len(dirty.splitlines()),
             "updated": False},
        )

    code, out = _git("fetch", "--quiet", "origin", branch, cwd=root)
    if code != 0:
        return _summary(ctx, "update", f"Cannot reach the remote: {out[:160]}",
                        {"root": str(root), "branch": branch, "updated": False})

    code, behind = _git("rev-list", "--count", f"HEAD..origin/{branch}", cwd=root)
    if code != 0:
        return _summary(ctx, "update", f"Cannot compare with origin/{branch}: {behind[:160]}",
                        {"root": str(root), "branch": branch, "updated": False})
    count = int(behind or 0)
    code, head = _git("rev-parse", "--short", "HEAD", cwd=root)
    if count == 0:
        return _summary(ctx, "update", f"Already up to date on {branch} ({head}).",
                        {"root": str(root), "branch": branch, "behind": 0, "updated": False})

    code, log = _git("log", "--oneline", "-n", "8", f"HEAD..origin/{branch}", cwd=root)
    commits = [line for line in log.splitlines() if line.strip()]
    if not apply_now:
        await _audit(ctx, "slash.update.check", str(root), f"{count} commit(s) available")
        return _summary(
            ctx, "update",
            f"{count} update(s) available on {branch}:\n"
            + "\n".join(f"  • {line}" for line in commits[:8])
            + ("\n  …" if count > len(commits) else "")
            + "\n\nSend /update apply to install and restart.",
            {"root": str(root), "branch": branch, "behind": count,
             "commits": commits, "updated": False},
        )

    code, changed = _git("diff", "--name-only", f"HEAD..origin/{branch}", cwd=root)
    deps_moved = any(
        name in ("pyproject.toml", "uv.lock", "requirements.txt", "install.sh")
        for name in changed.splitlines()
    )
    code, out = _git("merge", "--ff-only", f"origin/{branch}", cwd=root)
    if code != 0:
        first = out.splitlines()[0] if out else "history diverged"
        return _summary(
            ctx, "update",
            f"Update stopped: {branch} cannot be fast-forwarded ({first}). Nothing was changed.",
            {"root": str(root), "branch": branch, "updated": False},
        )

    deps_note = "dependencies unchanged"
    deps_ok = True
    if deps_moved:
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv
                ["uv", "pip", "install", "--python", str(root / ".venv/bin/python"), "-e", "."],
                cwd=str(root), capture_output=True, text=True, timeout=900, check=False,
            )
            deps_ok = done.returncode == 0
            deps_note = "dependencies reinstalled" if deps_ok else "dependency install FAILED"
        except (OSError, subprocess.TimeoutExpired) as exc:
            deps_ok, deps_note = False, f"dependency install failed: {exc}"

    code, head = _git("rev-parse", "--short", "HEAD", cwd=root)
    await _audit(ctx, "slash.update.apply", str(root), f"{count} commit(s) → {head}; {deps_note}")
    restarted = await _schedule_restart(ctx, "update")
    return _summary(
        ctx, "update",
        f"Updated {branch} by {count} commit(s) → {head}. {deps_note}. "
        + ("♻️ Restarting LEO — back in a few seconds. Anything you send meanwhile "
           "arrives once I'm up." if restarted else "Send /restart to load the new code."),
        {"root": str(root), "branch": branch, "commits": count, "head": head,
         "deps_ok": deps_ok, "updated": True, "restarted": restarted},
    )


async def restart_command(ctx: CommandContext) -> CommandResult:
    """Restart the LEO service from chat.

    A process cannot restart itself in place: the reload has to happen after
    this reply is delivered, and it must survive the process dying. So the
    restart is an independent, detached child that sleeps first - the same
    pattern the operator used before there was a command for it.
    """
    units, source = _restart_targets()
    delay = max(1, int(os.getenv("LEO_RESTART_DELAY_SECONDS") or 3))
    if not shutil.which("systemctl"):
        return _summary(
            ctx, "restart",
            "Cannot restart: systemctl is not available on this host. "
            f"Restart the process that runs 'leo serve' manually (targets: {', '.join(units)}).",
            {"units": units, "scheduled": False},
        )
    # Whoever asked gets told when the process is actually back.
    from ..core.restart_notice import mark_restart_pending

    target = None
    if ctx.conversation_id:
        target = await db.fetchone(
            "SELECT channel, subject FROM conversations WHERE id=?", (ctx.conversation_id,)
        )
    if target is not None:
        await mark_restart_pending(
            str(target["channel"] or ""), str(target["subject"] or ""), units=units
        )
    if not _spawn_restart(units, delay):
        return _summary(
            ctx, "restart",
            f"Cannot restart {', '.join(units)}: the detached restart could not be spawned.",
            {"units": units, "scheduled": False, "source": source},
        )
    await _audit(
        ctx, "slash.restart", ",".join(units),
        f"scheduled restart in {delay}s ({source})",
    )
    return _summary(
        ctx, "restart",
        f"♻️ Restarting {', '.join(units)} — back in a few seconds. "
        "Anything you send meanwhile arrives once I'm up.",
        {"units": units, "scheduled": True, "delay_seconds": delay, "source": source},
    )


async def reset_command(ctx: CommandContext) -> CommandResult:
    conversation_id = ctx.conversation_id
    if not conversation_id:
        raise ValueError("/reset needs a conversation_id in the command context")
    row = await db.fetchone("SELECT id FROM conversations WHERE id=?", (conversation_id,))
    if row is None:
        raise ValueError(f"unknown conversation: {conversation_id}")
    async with db.transaction() as conn:
        removed = await conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        await conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (utcnow(), conversation_id))
    await _audit(
        ctx, "slash.reset", conversation_id, "conversation messages cleared in place",
        risk=RiskLevel.HIGH, metadata={"removed": removed.rowcount},
    )
    await _emit(ctx.app, "slash.session.reset", conversation_id=conversation_id, actor=ctx.actor)
    return _summary(
        ctx, "reset",
        f"Cleared {removed.rowcount} messages from conversation {conversation_id} in place; "
        "the conversation id is unchanged.",
        {"conversation_id": conversation_id, "removed": removed.rowcount},
    )


async def sessions_command(ctx: CommandContext) -> CommandResult:
    limit = _token_limit(ctx)
    rows = await db.fetchall(
        "SELECT c.id, c.channel, c.subject, c.created_at, c.updated_at, "
        "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count "
        "FROM conversations c ORDER BY c.updated_at DESC LIMIT ?",
        (limit,),
    )
    items = [dict(row) for row in rows]
    return _summary(ctx, "sessions", f"{len(items)} conversations", {"sessions": items})


async def history_command(ctx: CommandContext) -> CommandResult:
    conversation_id = ctx.conversation_id
    if not conversation_id:
        raise ValueError("/history needs a conversation_id in the command context")
    limit = _token_limit(ctx)
    rows = await db.fetchall(
        "SELECT role, content, created_at FROM messages WHERE conversation_id=? "
        "ORDER BY id DESC LIMIT ?",
        (conversation_id, limit),
    )
    items = [dict(row) for row in reversed(rows)]
    return _summary(
        ctx, "history", f"{len(items)} transcript rows for {conversation_id}",
        {"conversation_id": conversation_id, "messages": items},
    )


async def export_command(ctx: CommandContext) -> CommandResult:
    """Export this conversation in one call, via the shared tool implementation.

    Output is always redacted here (chat surfaces can reach this command), and
    the returned path/size come from the real file that was written.
    """
    conversation_id = ctx.conversation_id
    if not conversation_id:
        raise ValueError("/export needs a conversation_id in the command context")

    words: list[str] = []
    out: str | None = None
    index = 0
    while index < len(ctx.argv):
        token = str(ctx.argv[index])
        if token in ("--out", "-o"):
            if index + 1 >= len(ctx.argv):
                raise ValueError("--out needs a file name; usage: /export [md|json|txt] [--out name]")
            out = str(ctx.argv[index + 1])
            index += 2
            continue
        if token.startswith("--out="):
            out = token.split("=", 1)[1]
            index += 1
            continue
        if token.startswith("-"):
            raise ValueError(f"unknown option {token!r}; usage: /export [md|json|txt] [--out name]")
        words.append(token)
        index += 1
    if len(words) > 1:
        raise ValueError("usage: /export [md|json|txt] [--out name]")
    fmt = words[0] if words else "md"

    result = await conversation_tools.write_export(
        conversation_id=conversation_id,
        fmt=fmt,
        redact=True,
        out=out,
        ctx={"settings": ctx.app.settings},
    )
    await _audit(
        ctx, "slash.export", conversation_id, "conversation transcript exported",
        metadata={"format": result["format"], "bytes": result["bytes"],
                  "count": result["count"], "path": result["path"]},
    )
    await _emit(
        ctx.app, "slash.session.export", conversation_id=conversation_id,
        path=result["path"], bytes=result["bytes"], format=result["format"],
    )
    message = (
        f"Exported {result['count']} message(s) to {result['path']} "
        f"({result['bytes']} bytes, format={result['format']}, redacted=True)"
    )
    return _summary(ctx, "export", message, result)


async def compact_command(ctx: CommandContext) -> CommandResult:
    """Summarize older turns once and persist the boundary.

    The transcript is never rewritten: only what the model is shown changes.
    ``--preview`` shows the summary without committing; ``--clear`` drops the
    stored boundary so the full history is replayed again.
    """
    conversation_id = ctx.conversation_id
    if not conversation_id:
        raise ValueError("/compact needs a conversation_id in the command context")

    preview = False
    clear = False
    focus_words: list[str] = []
    for token in ctx.argv:
        text = str(token)
        if text in ("--preview", "-n"):
            preview = True
        elif text in ("--clear", "--reset"):
            clear = True
        elif text in ("--focus", "-f"):
            continue
        elif text.startswith("-"):
            raise ValueError(
                f"unknown option {text!r}; usage: /compact [focus text] [--preview|--clear]"
            )
        else:
            focus_words.append(text)
    focus = " ".join(focus_words).strip()

    if clear:
        from ..models import compaction as compaction_module

        removed = await compaction_module.clear(conversation_id)
        await _audit(
            ctx, "slash.compact.clear", conversation_id,
            "durable compaction boundary cleared",
            metadata={"removed": removed},
        )
        await _emit(ctx.app, "slash.session.compact_cleared",
                    conversation_id=conversation_id, removed=removed)
        message = (
            f"Cleared {removed} compaction record(s). Full history replays again; "
            "the transcript was never modified."
        )
        return _summary(ctx, "compact", message,
                        {"conversation_id": conversation_id, "cleared": removed})

    data = await ctx.app.compact_conversation(
        conversation_id, focus=focus, preview=preview
    )
    if not data.get("ok"):
        reason = str(data.get("reason") or "compaction did not run")
        await _audit(
            ctx, "slash.compact", conversation_id, "compaction declined",
            metadata={"reason": reason, "preview": preview},
        )
        return CommandResult(
            command="compact",
            status="error",
            success=False,
            message=_redact(ctx, f"Compaction did not run: {reason}", MAX_MESSAGE),
            data=data,
        )

    summarized = int(data.get("messages_summarized") or 0)
    before = int(data.get("tokens_before") or 0)
    after = int(data.get("tokens_after") or 0)
    degraded = bool(data.get("degraded"))
    await _audit(
        ctx, "slash.compact" if not preview else "slash.compact.preview",
        conversation_id,
        "older turns summarized" if not preview else "compaction previewed",
        metadata={
            "messages_summarized": summarized,
            "tokens_before": before,
            "tokens_after": after,
            "degraded": degraded,
            "focus": focus[:200] or None,
        },
    )
    if not preview:
        await _emit(
            ctx.app, "slash.session.compacted", conversation_id=conversation_id,
            messages_summarized=summarized, tokens_before=before,
            tokens_after=after, degraded=degraded,
        )
    note = ""
    if degraded:
        note = (
            " Summary quality is degraded: some older details or identifiers may "
            "be missing. Resend anything essential, or use /new on a larger model."
        )
    verb = "Previewed" if preview else "Compacted"
    message = (
        f"{verb} {summarized} older message(s); estimated context {before} -> "
        f"{after} tokens. The transcript itself is unchanged and every original "
        f"message stays on disk.{note}"
    )
    return _summary(ctx, "compact", message, data)


async def title_command(ctx: CommandContext) -> CommandResult:
    conversation_id = ctx.conversation_id
    if not conversation_id:
        raise ValueError("/title needs a conversation_id in the command context")
    title = ctx.rest().strip()
    if not title:
        row = await db.fetchone("SELECT subject FROM conversations WHERE id=?", (conversation_id,))
        if row is None:
            raise ValueError(f"unknown conversation: {conversation_id}")
        return _summary(ctx, "title", f"title: {row['subject'] or '(untitled)'}",
                        {"conversation_id": conversation_id, "title": row["subject"]})
    title = title[:200]
    result = await db.execute(
        "UPDATE conversations SET subject=?, updated_at=? WHERE id=?",
        (title, utcnow(), conversation_id),
    )
    if result.rowcount < 1:
        raise ValueError(f"unknown conversation: {conversation_id}")
    await _audit(ctx, "slash.title", conversation_id, "conversation title set", metadata={"title": title})
    await _emit(ctx.app, "slash.session.title", conversation_id=conversation_id, title=title)
    return _summary(ctx, "title", f"Title set to {title!r}",
                    {"conversation_id": conversation_id, "title": title})


async def stop_command(ctx: CommandContext) -> CommandResult:
    """Stop work. With no id, stop the turn that is running right now.

    A slash command can arrive while another worker is busy with a long turn, so
    "stop" is a request the live turn observes between steps - not a process kill
    and not a promise that an in-flight tool call is interrupted mid-command.
    """
    run_id = ctx.argv[0] if ctx.argv else ctx.run_id
    if run_id:
        outcome = await ctx.app.cancel_run(run_id)
        return _summary(ctx, "stop", f"Run {run_id}: {outcome}", {"run_id": run_id, "status": outcome})
    conversation_id = ctx.conversation_id
    if not conversation_id:
        raise ValueError("/stop needs a run id here; no conversation is attached to this surface")
    await ctx.app.request_turn_stop(conversation_id, requested_by=f"slash:{ctx.principal or ctx.channel}")
    await _audit(ctx, "slash.stop", conversation_id, "turn stop requested via slash")
    return _summary(
        ctx, "stop",
        "⏹ Stopping — this turn ends at its next step.",
        {"conversation_id": conversation_id, "stopping": True},
    )


async def diff_command(ctx: CommandContext) -> CommandResult:
    staged = False
    stat = False
    paths: list[str] = []
    for token in ctx.argv:
        if token == "--staged":
            staged = True
        elif token == "--stat":
            stat = True
        elif token in ("session", "all"):
            return CommandResult.unsupported(
                "diff",
                f"/diff {token} is unavailable: LEO has no session-scoped or all-changes "
                "snapshot; the git_diff tool reads workspace changes only.",
            )
        elif not token.startswith("-"):
            paths.append(token)
    if len(paths) > 1:
        raise ValueError("/diff accepts at most one path")
    arguments: dict[str, Any] = {"staged": staged, "stat": stat}
    if paths:
        arguments["path"] = paths[0]
    call = ToolCall(name="git_diff", arguments=arguments, actor=ctx.actor,
                    correlation_id=ctx.conversation_id)
    result = await ctx.app.executor.execute(call)
    data = result.model_dump(mode="json")
    if not result.success:
        return CommandResult.error("diff", f"git_diff failed: {result.error or result.decision}",
                                   code="tool_failed", data=data)
    output = _dump(result.output)
    text = output if isinstance(output, str) else json.dumps(output, default=str)
    return _summary(ctx, "diff", text or "(no diff)", data)


# ---------------------------------------------------------------------------
# skills


async def _coerce_skill_arguments(
    skills: Any, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Turn slash text into the JSON types declared by the skill's schema.

    Slash arguments arrive as strings; LEO skill handlers validate declared
    integer/boolean fields strictly. Coercion is limited to the schema and
    never evaluates the value.
    """
    record = await skills.get(name)
    schema = getattr(record, "input_schema", None) or {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    coerced = dict(arguments)
    for key, value in list(coerced.items()):
        spec = properties.get(key)
        if not isinstance(spec, dict) or not isinstance(value, str):
            continue
        kind = spec.get("type")
        if kind == "integer":
            try:
                coerced[key] = int(value)
            except ValueError:
                pass
        elif kind == "number":
            try:
                coerced[key] = float(value)
            except ValueError:
                pass
        elif kind == "boolean" and value.lower() in ("true", "false", "1", "0", "on", "off"):
            coerced[key] = value.lower() in ("true", "1", "on")
    return coerced


async def _run_skill(ctx: CommandContext, name: str, arguments: dict[str, Any]) -> CommandResult:
    skills = _require_service(ctx, "skills")
    arguments = await _coerce_skill_arguments(skills, name, arguments)
    result = await skills.run(
        name, arguments,
        context={"approval_id": ctx.approval_id, "actor": ctx.actor, "surface": ctx.surface},
    )
    data = _dump(result)
    if getattr(result, "decision", "allowed") == "pending":
        return CommandResult.pending(
            "skill", f"Skill {name} requires approval: {getattr(result, 'error', '')}",
            getattr(result, "approval_id", None), data=data,
        )
    if getattr(result, "decision", "allowed") == "denied":
        return CommandResult.denied("skill", f"Skill {name}: {getattr(result, 'error', '')}", data=data)
    if not getattr(result, "success", False):
        return CommandResult.error("skill", getattr(result, "error", None) or "skill failed",
                                   code="skill_failed", data=data)
    output = getattr(result, "output", None)
    text = output if isinstance(output, str) else json.dumps(output, default=str)
    return _summary(ctx, "skill", text or f"Skill {name} completed", data)


async def skills_command(ctx: CommandContext) -> CommandResult:
    skills = _require_service(ctx, "skills")
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub in ("list", "browse"):
        records = await skills.list(format="dicts")
        query = ctx.option("--search", "-q", default="")
        if query:
            needle = str(query).lower()
            records = [r for r in records if needle in json.dumps(r, default=str).lower()]
        items = [_dump(item) for item in records][:_token_limit(ctx)]
        return _summary(ctx, "skills", f"{len(items)} skills", {"skills": items})
    if sub == "search":
        query = ctx.rest(1).strip()
        if not query:
            raise ValueError("usage: /skills search <query>")
        needle = query.lower()
        records = await skills.list(format="dicts")
        items = [
            r for r in records
            if needle in str(r.get("name", "")).lower()
            or needle in str(r.get("description", "")).lower()
            or needle in str(r.get("category", "")).lower()
        ][:_token_limit(ctx)]
        return _summary(ctx, "skills", f"{len(items)} skills match {query!r}", {"skills": items})
    if sub in ("inspect", "show", "get"):
        name = ctx.require(1, "skill name")
        record = await skills.get(name)
        if record is None:
            raise ValueError(f"unknown skill: {name}")
        return _summary(ctx, "skills", f"Skill {name}", {"skill": _dump(record)})
    if sub == "history":
        name = ctx.argv[1] if len(ctx.argv) > 1 else None
        rows = await skills.history(name, limit=_token_limit(ctx))
        return _summary(ctx, "skills", f"{len(rows)} skill history rows", {"history": rows})
    if sub == "run":
        name = ctx.require(1, "skill name")
        _, values = _kv(ctx.argv[2:])
        return await _run_skill(ctx, name, values)
    if sub in ("enable", "disable"):
        name = ctx.require(1, "skill name")
        record = await skills.enable(name) if sub == "enable" else await skills.disable(name)
        await _audit(ctx, f"slash.skills.{sub}", name, f"skill {sub} via slash")
        return _summary(ctx, "skills", f"Skill {name} {sub}d", {"skill": _dump(record)})
    if sub in ("install", "audit", "approve", "reject", "diff", "approval"):
        return CommandResult.unsupported(
            "skills",
            f"/skills {sub} is not wired in LEO: skill installation and provenance review "
            "are CLI/API operations with their own approval flow.",
        )
    raise ValueError(f"unknown /skills subcommand {sub!r}; try list, search, inspect, run, enable, disable, history")


async def skill_command(ctx: CommandContext) -> CommandResult:
    name = ctx.require(0, "skill name")
    _, values = _kv(ctx.argv[1:])
    if values:
        arguments: dict[str, Any] = values
    else:
        instruction = ctx.rest(1).strip()
        if not instruction:
            arguments = {}
        else:
            # A bare instruction is mapped onto the skill's first declared
            # input, which is how /<skill-name> <text> stays useful without
            # pretending to interpret arbitrary free-form skill parameters.
            record = await _require_service(ctx, "skills").get(name)
            schema = getattr(record, "input_schema", None) or {}
            properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = schema.get("required", []) if isinstance(schema, dict) else []
            target = next((key for key in required if key in properties), None)
            if target is None and properties:
                target = next(iter(properties))
            arguments = {target: instruction} if target else {"instruction": instruction}
    result = await _run_skill(ctx, name, arguments)
    result.command = "skill"
    return result


# ---------------------------------------------------------------------------
# memory


async def memory_command(ctx: CommandContext) -> CommandResult:
    memory = _require_service(ctx, "memory")
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub == "list":
        category = ctx.option("--category")
        records = await memory.list(category=category, limit=_token_limit(ctx))
        items = [_dump(record) for record in records]
        return _summary(ctx, "memory", f"{len(items)} memories", {"memories": items})
    if sub == "search":
        query = ctx.rest(1).strip()
        if not query:
            raise ValueError("usage: /memory search <query>")
        results = await _require_service(ctx, "search").search(query, limit=_token_limit(ctx))
        items = [_dump(record) for record in results]
        return _summary(ctx, "memory", f"{len(items)} memories match {query!r}", {"results": items})
    if sub == "get":
        memory_id = ctx.require(1, "memory id")
        record = await memory.get(memory_id)
        if record is None:
            raise ValueError(f"unknown memory: {memory_id}")
        return _summary(ctx, "memory", f"Memory {memory_id}", {"memory": _dump(record)})
    if sub == "add":
        words, options = _kv(ctx.argv[1:])
        # Option values are consumed by ctx.option separately; remove them from
        # the content text so --category semantic only configures the record.
        content_words: list[str] = []
        index = 0
        while index < len(words):
            token = words[index]
            if token in ("--category", "--kind", "--tags", "-t") and index + 1 < len(words):
                index += 2
                continue
            if token.startswith("--") and "=" in token:
                index += 1
                continue
            content_words.append(token)
            index += 1
        content = " ".join(content_words).strip()
        if not content:
            raise ValueError("usage: /memory add <text> [--category c] [--kind k] [--tags a,b]")
        category = MemoryCategory(ctx.option("--category", default="semantic") or "semantic")
        kind = MemoryKind(ctx.option("--kind", default="fact") or "fact")
        tags_raw = ctx.option("--tags", "-t", default=options.get("tags", ""))
        tags = [tag.strip() for tag in str(tags_raw).split(",") if tag.strip()]
        record = await ctx.app.remember(content, category=category, kind=kind, tags=tags)
        await _audit(ctx, "slash.memory.add", record.id, "memory added via slash", metadata={"category": category.value})
        await _emit(ctx.app, "slash.memory.added", memory_id=record.id, actor=ctx.actor)
        return _summary(ctx, "memory", f"Stored memory {record.id}", {"memory": _dump(record)})
    if sub in ("forget", "delete"):
        memory_id = ctx.require(1, "memory id")
        gate = await _check_high_risk(
            ctx, "slash.memory.forget", f"memory:{memory_id}",
            {"argv": list(ctx.argv), "subcommand": sub, "memory_id": memory_id},
        )
        if gate is not None:
            gate.command = "memory"
            return gate
        deleted = await memory.delete(memory_id)
        if not deleted:
            raise ValueError(f"unknown memory: {memory_id}")
        await _audit(ctx, "slash.memory.forget", memory_id, "memory forgotten via slash",
                     risk=RiskLevel.HIGH)
        await _emit(ctx.app, "slash.memory.forgotten", memory_id=memory_id, actor=ctx.actor)
        return _summary(ctx, "memory", f"Forgot memory {memory_id}", {"memory_id": memory_id, "deleted": True})
    raise ValueError(f"unknown /memory subcommand {sub!r}; try list, search, get, add, forget")


async def remember_command(ctx: CommandContext) -> CommandResult:
    content = ctx.rest().strip()
    if not content:
        raise ValueError("usage: /remember <text>")
    record = await ctx.app.remember(content)
    await _audit(ctx, "slash.memory.add", record.id, "memory added via /remember")
    result = _summary(ctx, "remember", f"Stored memory {record.id}", {"memory": _dump(record)})
    result.command = "remember"
    return result


# ---------------------------------------------------------------------------
# tasks and agents


async def tasks_command(ctx: CommandContext) -> CommandResult:
    tasks = _require_service(ctx, "tasks")
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub in ("list", "ls"):
        status = ctx.option("--status")
        records = await tasks.list(status=status, limit=_token_limit(ctx))
        items = [_dump(record) for record in records]
        return _summary(ctx, "tasks", f"{len(items)} tasks", {"tasks": items})
    if sub in ("create", "add", "new"):
        objective = ctx.rest(1).strip()
        if not objective:
            raise ValueError("usage: /tasks create <objective>")
        owner = ctx.option("--owner")
        plan_raw = ctx.option("--plan", default="")
        plan = [step.strip() for step in str(plan_raw).split(",") if step.strip()]
        record = await tasks.create(objective, plan=plan or None, owner_agent=owner)
        await _audit(ctx, "slash.tasks.create", record.id, "task created via slash")
        await _emit(ctx.app, "slash.task.created", task_id=record.id, actor=ctx.actor)
        return _summary(ctx, "tasks", f"Created task {record.id} ({record.status.value})",
                        {"task": _dump(record)})
    if sub in ("status", "get", "show"):
        task_id = ctx.require(1, "task id")
        record = await tasks.get(task_id)
        if record is None:
            raise ValueError(f"unknown task: {task_id}")
        return _summary(ctx, "tasks", f"Task {task_id}: {record.status.value}", {"task": _dump(record)})
    if sub == "cancel":
        task_id = ctx.require(1, "task id")
        await tasks.cancel(task_id)
        await _audit(ctx, "slash.tasks.cancel", task_id, "task cancellation requested via slash")
        return _summary(ctx, "tasks", f"Cancellation requested for task {task_id}",
                        {"task_id": task_id, "cancelled": True})
    if sub == "run":
        task_id = ctx.require(1, "task id")
        worker = getattr(ctx.app, "_default_task_worker", None)
        if not callable(worker):
            return CommandResult.unsupported("tasks", "No default task worker is registered in this app.")
        record = await tasks.get(task_id)
        if record is None:
            raise ValueError(f"unknown task: {task_id}")
        finished = await tasks.run(task_id, worker)
        await _audit(ctx, "slash.tasks.run", task_id, "task run via slash")
        return _summary(ctx, "tasks", f"Task {task_id}: {finished.status.value}",
                        {"task": _dump(finished)})
    raise ValueError(f"unknown /tasks subcommand {sub!r}; try list, create, status, cancel, run")


async def agents_command(ctx: CommandContext) -> CommandResult:
    registry = _require_service(ctx, "agent_registry")
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub in ("list", "ls"):
        enabled_only = ctx.has_flag("--enabled")
        records = await registry.list(enabled_only=enabled_only)
        items = [_dump(record) for record in records][:_token_limit(ctx)]
        return _summary(ctx, "agents", f"{len(items)} agents", {"agents": items})
    if sub in ("status", "get", "show"):
        agent_id = ctx.require(1, "agent id")
        record = await registry.get(agent_id)
        if record is None:
            raise ValueError(f"unknown agent: {agent_id}")
        history = await registry.history(agent_id, limit=10)
        return _summary(ctx, "agents", f"Agent {agent_id}",
                        {"agent": _dump(record), "history": history})
    if sub == "history":
        agent_id = ctx.argv[1] if len(ctx.argv) > 1 else None
        history = await registry.history(agent_id, limit=_token_limit(ctx))
        return _summary(ctx, "agents", f"{len(history)} agent runs", {"history": history})
    if sub in ("delegate", "send"):
        objective = ctx.rest(1).strip()
        if not objective:
            raise ValueError("usage: /agents delegate <objective> [--agent id]")
        agent_id = ctx.option("--agent", "-a")
        gate = await _check_risk(
            ctx, "slash.agents.delegate", f"agent:{agent_id or 'auto'}", RiskLevel.MEDIUM,
            {"argv": list(ctx.argv), "agent_id": agent_id or "auto"},
        )
        if gate is not None:
            gate.command = "agents"
            return gate
        engine = _require_service(ctx, "agent_engine")
        result = await engine.delegate(
            objective,
            agent_id=agent_id,
            context={"source": "slash", "actor": ctx.actor, "conversation_id": ctx.conversation_id},
            verify=not ctx.has_flag("--no-verify"),
        )
        outcomes = result if isinstance(result, (list, tuple)) else [result]
        data = {"results": [_dump(item) for item in outcomes]}
        await _audit(ctx, "slash.agents.delegate", agent_id or "auto", "agent delegation via slash")
        first = outcomes[0] if outcomes else None
        success = bool(getattr(first, "success", False))
        message = f"Delegated to {agent_id or 'auto-selected'} agent: {len(outcomes)} result(s), success={success}"
        return CommandResult(
            command="agents", status="ok" if success else "error", success=success,
            message=_redact(ctx, message), data=data,
        )
    raise ValueError(f"unknown /agents subcommand {sub!r}; try list, status, history, delegate")


async def tools_command(ctx: CommandContext) -> CommandResult:
    registry = _require_service(ctx, "registry")
    sub = ctx.argv[0].lower() if ctx.argv and not ctx.argv[0].startswith("-") else "list"
    if sub in ("enable", "disable"):
        return CommandResult.unsupported(
            "tools",
            "LEO toolsets are registered at composition time; per-session tool enable/disable "
            "is not implemented. /tools list is read-only.",
        )
    query = ctx.rest(1).strip().lower() if sub == "search" else ctx.option("--search", "-q", default="")
    specs = registry.list_specs()
    items = [
        {"name": spec.name, "description": spec.description, "category": spec.category,
         "risk": spec.risk.value if hasattr(spec.risk, "value") else str(spec.risk),
         "permissions": list(getattr(spec, "permissions", []) or [])}
        for spec in specs
    ]
    if query:
        items = [item for item in items if query in json.dumps(item, default=str).lower()]
    items = items[:_token_limit(ctx)]
    return _summary(ctx, "tools", f"{len(items)} tools", {"tools": items})


# ---------------------------------------------------------------------------
# approvals (trusted operator only)


async def approvals_command(ctx: CommandContext) -> CommandResult:
    if not _operator(ctx):
        return CommandResult.denied(
            "approvals",
            "Approval review requires the trusted operator CLI principal and surface.",
        )
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub in ("mode", "set"):
        return CommandResult.unsupported(
            "approvals",
            "Approval modes are configured in leo.yaml; no runtime mode switch is implemented.",
        )
    pending = await ctx.app.permissions.pending_approvals()
    return _summary(ctx, "approvals", f"{len(pending)} pending approvals", {"approvals": pending})


async def _resolve(ctx: CommandContext, approved: bool) -> CommandResult:
    if not _operator(ctx):
        return CommandResult.denied(
            "approve" if approved else "deny",
            "Only the trusted operator CLI principal can resolve an approval; "
            "chat actors can never resolve their own approvals.",
        )
    approval_id = ctx.require(0, "approval id")
    result = await ctx.app.operator_resolve_approval(
        approval_id, approved, actor=f"operator:{ctx.principal}"
    )
    if not result.get("success", False):
        return CommandResult.error(
            "approve" if approved else "deny",
            str(result.get("error") or "approval resolution failed"),
            code="approval_failed", data=result,
        )
    return _summary(
        ctx, "approve" if approved else "deny",
        f"Approval {approval_id} {'approved' if approved else 'denied'} via operator executor",
        {"approval": result},
    )


async def approve_command(ctx: CommandContext) -> CommandResult:
    return await _resolve(ctx, True)


async def deny_command(ctx: CommandContext) -> CommandResult:
    return await _resolve(ctx, False)


# ---------------------------------------------------------------------------
# observability


def _visible(events: list[Any], ctx: CommandContext) -> list[Any]:
    if _operator(ctx) or ctx.surface in ("cli",):
        return events
    return [
        event for event in events
        if not str(getattr(event, "topic", "") or "").startswith(SENSITIVE_TOPICS)
    ]


async def events_command(ctx: CommandContext) -> CommandResult:
    bus = _require_service(ctx, "events")
    sub = ctx.argv[0].lower() if ctx.argv and ctx.argv[0] in ("stats", "list") else "list"
    limit = _token_limit(ctx)
    if sub == "stats":
        data = await bus.stats()
        return _summary(ctx, "events", "Event durability statistics", data)
    topic = ctx.option("--topic")
    if not topic and ctx.argv and ctx.argv[0] not in ("list", "stats"):
        topic = ctx.argv[0]
    rows = _visible(await bus.history(topic=topic, limit=limit), ctx)
    items = [_dump(event) for event in rows]
    return _summary(ctx, "events", f"{len(items)} events", {"events": items})


async def metrics_command(ctx: CommandContext) -> CommandResult:
    data = await ctx.app.metrics()
    return _summary(ctx, "metrics", "LEO metrics snapshot", data)


async def debug_command(ctx: CommandContext) -> CommandResult:
    sub = ctx.argv[0].lower() if ctx.argv else "local"
    if sub in ("upload", "nous"):
        return CommandResult.unsupported(
            "debug",
            "LEO never uploads diagnostic reports to an external service; use /debug local.",
        )
    health = await ctx.app.health()
    durability = await ctx.app.events.stats()
    metrics = await ctx.app.metrics()
    data = {"health": health, "events": durability,
            "counts": {key: len(value) for key, value in metrics.items() if isinstance(value, list)}}
    return _summary(ctx, "debug", "Local diagnostics only; nothing was uploaded", data)


async def audit_command(ctx: CommandContext) -> CommandResult:
    rows = await db.fetchall(
        "SELECT ts, actor, action, resource, risk, decision, reason FROM audit_log "
        "ORDER BY id DESC LIMIT ?",
        (_token_limit(ctx),),
    )
    return _summary(ctx, "audit", f"{len(rows)} audit rows", {"audit": [dict(row) for row in rows]})


async def insights_command(ctx: CommandContext) -> CommandResult:
    usage = await ctx.app.router.usage_report()
    tasks = await ctx.app.tasks.list(limit=MAX_LIMIT)
    events = await ctx.app.events.history(limit=MAX_LIMIT)
    data = {
        "model_calls": len(usage),
        "tasks": len(tasks),
        "events": len(events),
        "completed_tasks": sum(1 for task in tasks if str(getattr(task.status, "value", task.status)) == "completed"),
    }
    return _summary(ctx, "insights", f"{len(usage)} model calls, {len(tasks)} tasks", data)


# ---------------------------------------------------------------------------
# scheduling and channels


def _schedule_name(ctx: CommandContext) -> str:
    name = ctx.require(1, "schedule name")
    if not all(char.isalnum() or char in "-_" for char in name):
        raise ValueError("schedule name must contain only letters, digits, '-' and '_'")
    return name


async def schedule_command(ctx: CommandContext) -> CommandResult:
    scheduler = _require_service(ctx, "scheduler")
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub in ("list", "ls"):
        items = await scheduler.list_schedules()
        return _summary(ctx, "schedule", f"{len(items)} schedules", {"schedules": items})
    if sub in ("add", "create", "interval"):
        name = _schedule_name(ctx)
        seconds = ctx.require(2, "interval seconds")
        objective = ctx.rest(3).strip()
        if not objective:
            raise ValueError("usage: /schedule add <name> <seconds> <objective>")
        created = await scheduler.register_interval(name, float(seconds), objective)
        await _audit(ctx, "slash.schedule.add", name, "interval schedule created via slash")
        return _summary(ctx, "schedule", f"Created interval schedule {name}", {"schedule": created})
    if sub == "cron":
        name = _schedule_name(ctx)
        expression = ctx.require(2, "cron expression")
        objective = ctx.rest(3).strip()
        if not objective:
            raise ValueError("usage: /schedule cron <name> <expression> <objective>")
        created = await scheduler.register_cron(name, expression, objective)
        await _audit(ctx, "slash.schedule.add", name, "cron schedule created via slash")
        return _summary(ctx, "schedule", f"Created cron schedule {name}", {"schedule": created})
    if sub in ("enable", "disable", "pause", "resume"):
        name = _schedule_name(ctx)
        enabled = sub in ("enable", "resume")
        updated = await scheduler.set_enabled(name, enabled)
        await _audit(ctx, "slash.schedule.set_enabled", name,
                     f"schedule {'enabled' if enabled else 'disabled'} via slash")
        return _summary(ctx, "schedule", f"Schedule {name}: enabled={enabled}", {"schedule": updated})
    if sub in ("run", "trigger"):
        name = _schedule_name(ctx)
        created = await scheduler.trigger(name)
        await _audit(ctx, "slash.schedule.trigger", name, "schedule triggered via slash")
        return _summary(ctx, "schedule", f"Triggered {name}: {len(created)} task(s)",
                        {"tasks": [_dump(task) for task in created]})
    if sub in ("remove", "delete", "rm"):
        name = _schedule_name(ctx)
        await scheduler.get(name)  # verify before mutating, same as the API
        async with db.transaction() as conn:
            await conn.execute("DELETE FROM schedules WHERE name=?", (name,))
        await _audit(ctx, "slash.schedule.delete", name, "schedule removed via slash",
                     risk=RiskLevel.MEDIUM, metadata={"name": name})
        await _emit(ctx.app, "schedule.deleted", name=name)
        return _summary(ctx, "schedule", f"Removed schedule {name}", {"deleted": name})
    raise ValueError(f"unknown /schedule subcommand {sub!r}; try list, add, cron, enable, disable, run, remove")


async def loop_command(ctx: CommandContext) -> CommandResult:
    """Map /loop onto the durable scheduler, without claiming in-session loops."""
    scheduler = _require_service(ctx, "scheduler")
    sub = ctx.argv[0].lower() if ctx.argv else "list"
    if sub in ("list", "status"):
        items = await scheduler.list_schedules()
        return _summary(
            ctx, "loop", f"{len(items)} durable recurring schedules (not in-session loops)",
            {"schedules": items},
        )
    if sub in ("stop", "pause", "resume"):
        name = ctx.require(1, "schedule name")
        updated = await scheduler.set_enabled(name, sub == "resume")
        await _audit(ctx, "slash.loop.set_enabled", name, "loop schedule toggled via slash")
        return _summary(ctx, "loop", f"Schedule {name}: enabled={sub == 'resume'}", {"schedule": updated})
    name = ctx.option("--name", default="")
    if not name:
        raise ValueError("usage: /loop <seconds> <prompt> --name <schedule>")
    seconds = ctx.argv[0]
    # Strip the --name option pair from the recurring objective; the scheduler
    # stores exactly the prompt the caller asked to repeat.
    prompt_tokens: list[str] = []
    tokens = ctx.argv[1:]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--name" and index + 1 < len(tokens):
            index += 2
            continue
        if token.startswith("--name="):
            index += 1
            continue
        prompt_tokens.append(token)
        index += 1
    prompt = " ".join(prompt_tokens).strip()
    if not prompt:
        raise ValueError("usage: /loop <seconds> <prompt> --name <schedule>")
    created = await scheduler.register_interval(str(name), float(seconds), prompt)
    await _audit(ctx, "slash.loop.add", str(name), "recurring objective scheduled via slash")
    return _summary(
        ctx, "loop",
        f"Created durable schedule {name}; it runs as a LEO scheduled task, not inside this chat.",
        {"schedule": created},
    )


async def channels_command(ctx: CommandContext) -> CommandResult:
    gateway = _require_service(ctx, "gateway")
    records = [_dump(record) for record in gateway.integration_records()]
    rows = await db.fetchall("SELECT name, kind, status, verification FROM integrations ORDER BY name")
    outbox = await gateway.outbox_stats()
    query = ctx.argv[0].lower() if ctx.argv else ""
    if query and query not in ("list", "status"):
        records = [item for item in records if str(item.get("name", "")).lower() == query]
    data = {
        "adapters": records,
        "integrations": [dict(row) for row in rows],
        "outbox": outbox,
        "note": "credential values are never returned by /channels",
    }
    return _summary(ctx, "channels", f"{len(records)} channel adapters; outbox={outbox}", data)


# ---------------------------------------------------------------------------
# unavailable catalog: visible in help, explicit at dispatch time


def _spec(
    name: str,
    help_text: str,
    handler: Any,
    *,
    aliases: tuple[str, ...] = (),
    args_hint: str = "",
    args: tuple[ArgumentSpec, ...] = (),
    category: str = "general",
    risk: RiskLevel = RiskLevel.LOW,
    min_args: int = 0,
    resource_arg: int | None = None,
    surfaces: frozenset[str] = ALL_SURFACES,
    admin_only: bool = False,
    operator_only: bool = False,
    busy_policy: str = "any",
) -> CommandSpec:
    return CommandSpec(
        name=name, help=help_text, aliases=aliases, args=args, args_hint=args_hint,
        category=category, risk=risk, handler=handler, surfaces=surfaces,
        admin_only=admin_only, operator_only=operator_only, min_args=min_args,
        resource_arg=resource_arg, busy_policy=busy_policy,
    )


# name, aliases, args_hint, help, reason
UNAVAILABLE: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    ("resume", (), "[name]", "Switch to a saved session",
     "Session switching is owned by the chat client; the backend cannot change the active UI conversation. Use /sessions list."),
    ("import", (), "<archive>", "Import a shared profile or session archive",
     "LEO has no profile archive import service."),
    ("retry", (), "", "Resend the last user message",
     "No message-level retry primitive exists; only whole conversations can be created or cleared."),
    ("undo", (), "[N]", "Back up N user turns and re-prompt",
     "No message-level rewind primitive exists in LEO."),
    ("rewind", (), "[N|fork N]", "Rewind to an earlier prompt to edit or fork",
     "No message-level fork or rewind primitive exists in LEO."),
    ("branch", ("fork",), "[--here] [name]", "Branch or fork the current conversation",
     "LEO can create a new conversation (/new) but cannot copy an existing transcript into a branch."),
    ("steer", ("tell", "s"), "<prompt>", "Inject guidance into a running turn",
     "Mid-turn text injection is not supported; /stop ends the live turn at its next step."),
    ("queue", ("q",), "<prompt>|list|rm N|clear", "Queue prompts for later turns",
     "LEO has no shared prompt queue; tasks are durable objectives (/tasks create)."),
    ("bg", ("background",), "<prompt>", "Run a prompt in a background session",
     "LEO has no background chat-session primitive; /tasks create schedules durable work."),
    ("btw", ("side",), "<question>", "Ask a side question without changing context",
     "No side-question/snapshot model path exists in LEO."),
    ("goal", (), "[text|status|pause|resume|clear]", "Durable standing goal with judge loop",
     "LEO has no standing-goal judge; /tasks create stores a durable objective instead."),
    ("subgoal", (), "[text|remove N|clear]", "Add criteria to an active goal",
     "No standing-goal subsystem exists in LEO."),
    ("personality", (), "[name]", "Set a session personality overlay",
     "LEO has no personality overlay service."),
    ("fast", (), "[status|auto|on|off]", "Set provider fast/priority mode",
     "LEO has no runtime priority-processing override."),
    ("verbose", (), "[on|off|full]", "Set tool-output verbosity",
     "LEO has no per-session output verbosity toggle; /v is the short alias of /version."),
    ("yolo", ("elevated", "elev"), "[on|off]", "Bypass approval requirements",
     "LEO deliberately has no approval bypass flag; use the operator approval flow."),
    ("approvals_mode", ("allowlist",), "[manual|smart|off]", "Change approval policy mode",
     "Approval policy is configured in leo.yaml; no runtime mode switch is implemented."),
    ("memory_review", (), "[pending|approve|reject]", "Review staged memory writes",
     "LEO memory writes are authorized and audited at write time; there is no staged-memory review queue."),
    ("learn", (), "<source>", "Distill a reusable skill from a source",
     "No skill-distillation pipeline exists in LEO."),
    ("plan", (), "[task]", "Write an implementation plan file without executing",
     "No plan-file writer exists in LEO; /tasks create stores an objective."),
    ("review", (), "[instructions]", "Spawn an independent reviewer subagent",
     "LEO has a verification engine but no dedicated slash-triggered reviewer subagent."),
    ("refine", (), "", "Run memory/skill self-improvement in a fork",
     "No self-improvement fork service exists in LEO."),
    ("init", (), "[notes]", "Generate AGENTS.md from a repository scan",
     "No repository instruction-file generator exists in LEO."),
    ("insights_upload", (), "[days]", "Upload usage analytics",
     "LEO never uploads analytics; /insights reads local counters only."),
    ("reload", (), "", "Reload .env variables into the running session",
     "Runtime .env reload is not implemented; restart the process."),
    ("reload-mcp", ("reload_mcp",), "", "Reload MCP servers from configuration",
     "LEO has no MCP server registry in this composition root."),
    ("reload-skills", ("reload_skills",), "", "Rescan skills on disk at runtime",
     "LEO skills are registered declaratively at startup; no runtime rescan exists."),
    ("pause", (), "[reason|off]", "Globally pause new work",
     "LEO has no global work-pause switch."),
    ("start", (), "", "Acknowledge platform start pings",
     "This is a messaging-gateway-only control with no LEO handler."),
    ("discard", (), "<admission_id>", "Drop a lost turn so the queue runs again",
     "No gateway admission replay control is exposed to slash commands."),
    ("answer", (), "<id> <text>", "Answer a pending clarify prompt",
     "LEO has no clarify-prompt queue in the slash layer."),
    ("topic", (), "[off|session-id]", "Configure Telegram DM topic sessions",
     "Messaging-platform topic control is not implemented in LEO."),
    ("sethome", ("set-home",), "", "Set the platform home channel",
     "Messaging-platform home-channel control is not implemented in LEO."),
    ("platform", (), "<list|pause|resume> [name]", "Manage a gateway messaging platform",
     "LEO can report channel status (/channels) but cannot pause or resume platforms from slash."),
    ("handoff", (), "<platform>", "Hand the session off to a messaging platform",
     "No cross-platform session handoff service exists in LEO."),
    ("login", (), "[provider|refresh|cancel]", "Provider sign-in flow",
     "LEO uses local configuration and environment references; no interactive login flow exists."),
    ("subscription", ("upgrade",), "", "Manage a hosted plan",
     "LEO is self-hosted; hosted billing surfaces do not apply."),
    ("topup", (), "", "Manage hosted balance",
     "LEO is self-hosted; hosted billing surfaces do not apply."),
    ("voice", (), "[on|off|tts|status]", "Toggle voice mode or spoken replies",
     "LEO has no speech/voice pipeline in this composition root."),
    ("wake", (), "[on|off|status]", "Toggle a wake-word listener",
     "No wake-word listener exists in LEO."),
    ("pet", ("pets",), "[toggle|list|scale n]", "Manage a desktop mascot",
     "LEO has no mascot/pet feature."),
    ("hatch", ("generate-pet",), "[description]", "Generate a mascot",
     "LEO has no mascot/pet generator."),
    ("skin", (), "[name]", "Change display skin",
     "Display theming is client-local; no backend slash handler exists."),
    ("theme", (), "[auto|light|dark]", "Pin the client color theme",
     "Display theming is client-local; no backend slash handler exists."),
    ("plugins", ("plugin",), "[update|auto-update [on|off]]", "List or update plugins",
     "LEO has no plugin marketplace or plugin update handler."),
    ("bundles", (), "", "List skill bundles",
     "LEO skills have declarative bundles but no slash-visible bundle hub."),
    ("curator", (), "[status|run|pause]", "Background skill maintenance",
     "No skill curator service exists in LEO."),
    ("moa", (), "<prompt>", "Run a mixture-of-agents preset",
     "No preset mixture-of-agents runner exists; /agents delegate runs one objective."),
    ("blueprint", ("bp",), "[name] [slot=value]", "Create or browse cron blueprints",
     "LEO has direct schedules (/schedule) but no blueprint catalog."),
    ("kanban", (), "[subcommand]", "Multi-profile collaboration board",
     "LEO has no shared kanban board service."),
    ("journey", ("learning", "memory-graph"), "[list|delete|edit]", "Open a learning timeline",
     "LEO has durable memories but no learning-journey timeline."),
    ("worktree", (), "[new|list|prune]", "Manage isolated git worktrees",
     "No worktree service is registered in the LEO composition root."),
    ("rollback", (), "[number|--all]", "Restore filesystem checkpoints",
     "LEO task checkpoints are state snapshots, not filesystem restore points."),
    ("snapshot", ("snap",), "[create|restore|prune]", "Snapshot LEO config/state",
     "No config/state snapshot service exists in LEO."),
    ("browser", (), "[connect|disconnect|status]", "Attach browser tools to a live browser",
     "Browser/computer tools are model-mediated operator tools; slash must not bypass that gate."),
    ("image", (), "<path>", "Attach a local image to the next prompt",
     "Attachments are uploaded through the API; no local-file slash attachment path exists."),
    ("paste", (), "", "Attach a clipboard image",
     "Clipboard access is client-local; no backend slash handler exists."),
    ("copy", (), "[number]", "Copy an answer to the clipboard",
     "Clipboard access is client-local; no backend slash handler exists."),
    ("prompt", ("compose",), "[initial text]", "Compose the next prompt in an editor",
     "Local editor composition is client-local; no backend slash handler exists."),
    ("palette", (), "", "Open the command palette",
     "The command palette is a client-local UI overlay."),
    ("history_ui", ("history-clear",), "", "Clear the client screen",
     "Screen clearing is client-local; /reset clears durable messages in place."),
    ("quit", ("exit",), "[--delete]", "Exit the CLI",
     "Exiting the client process cannot be done from the backend slash dispatcher."),
    ("autoprune", (), "[on|off|later|status]", "Toggle automatic session pruning",
     "LEO has no automatic session pruning policy."),
    ("codex-runtime", ("codex_runtime",), "[auto|on|off]", "Toggle a Codex runtime",
     "LEO is provider-independent and has no Codex app-server runtime switch."),
    ("egress", (), "[status]", "Show Docker egress proxy status",
     "LEO has no Docker egress proxy subsystem."),
    ("profile", (), "", "Show or switch LEO profile",
     "LEO has a single local profile; no profile switching service exists."),
    ("statusbar", ("sb",), "", "Toggle the client status bar",
     "Status-bar display is client-local."),
    ("timestamps", ("ts",), "[on|off|status]", "Toggle message timestamps",
     "Timestamp display is client-local."),
    ("focus", (), "[on|off|status]", "Toggle reduced transcript view",
     "Focus view is client-local."),
    ("footer", (), "[on|off|status]", "Toggle the gateway footer",
     "Runtime metadata footer is a client/gateway display option, not a backend command."),
    ("battery", (), "[on|off|status]", "Toggle the battery readout",
     "Status-bar display is client-local."),
    ("indicator", (), "[style]", "Set the busy-indicator style",
     "Busy-indicator display is client-local."),
    ("redraw", (), "", "Force a full UI repaint",
     "Terminal repaint is client-local."),
    ("density", (), "[on|off|toggle]", "Toggle compact TUI display",
     "TUI-only display control; no backend handler exists."),
    ("details", ("detail",), "[hidden|collapsed|expanded]", "Control transcript detail visibility",
     "TUI-only display control; no backend handler exists."),
    ("logs", (), "[N]", "Show gateway log buffer",
     "TUI-only log buffer; /events reads durable LEO events."),
    ("mouse", ("scroll",), "[on|off|toggle]", "Set terminal mouse tracking",
     "TUI-only terminal control; no backend handler exists."),
    ("fortune", (), "[daily|random]", "Print a fortune cookie",
     "TUI-only local novelty command; no backend handler exists."),
    ("terminal-setup", (), "[auto|cursor|vscode]", "Configure IDE keybindings",
     "TUI/IDE setup is client-local."),
    ("setup", (), "", "Launch the external setup wizard",
     "Setup is a client-side wizard process; no backend slash handler exists."),
    ("replay", (), "[N|list|load path]", "Replay a subagent spawn tree",
     "TUI-only replay of client transcripts; no backend handler exists."),
    ("replay-diff", (), "<baseline> <candidate>", "Diff two spawn trees",
     "TUI-only replay of client transcripts; no backend handler exists."),
    ("connectors", (), "", "Open connector accounts overlay",
     "TUI-only overlay; /channels reports LEO integration status."),
    ("heapdump", (), "", "Write a V8 heap snapshot",
     "TUI Node.js debug command; not applicable to the Python LEO backend."),
    ("mem", (), "", "Print V8 heap numbers",
     "TUI Node.js debug command; not applicable to the Python LEO backend."),
    ("widgets-reload", (), "", "Rescan TUI widget apps",
     "TUI-only widget registry; no backend handler exists."),
    ("theme-info", (), "", "Print terminal theme probe",
     "TUI-only terminal probe; no backend handler exists."),
)


def register_builtins(registry: CommandRegistry | None = None) -> CommandRegistry:
    """Register every built-in spec on ``registry`` (or a fresh one)."""
    registry = registry or CommandRegistry()

    # -- help and identity ---------------------------------------------------
    registry.register(_spec(
        "help", "List commands; /help skills or /help <filter>, /help 2 for page two",
        help_command, aliases=("h",), args_hint="[filter|page]", category="help",
    ))
    registry.register(_spec(
        "commands", "Show the full command catalog with paging",
        commands_command, aliases=("catalog",), args_hint="[page]", category="help",
    ))
    registry.register(_spec(
        "status", "Session, model, capability and approval status",
        status_command, category="info",
    ))
    registry.register(_spec(
        "whoami", "Show the trusted principal, surface and effective command permissions",
        whoami_command, aliases=("id",), category="info",
    ))
    registry.register(_spec(
        "version", "Show the LEO version",
        version_command, aliases=("v",), category="info",
    ))
    registry.register(_spec(
        "config", "Read configuration; runtime set/unset is explicitly unsupported",
        config_command, args_hint="[show|get <key>]", args=(
            ArgumentSpec("subcommand", "show or get", variadic=True),
        ), category="config",
    ))
    registry.register(_spec(
        "model", "Show model status/providers; runtime switching is not supported",
        model_command, aliases=("models",), args_hint="[status|list|<name>]", category="models",
    ))
    from ..models.reasoning import EFFORT_LADDER
    reasoning_choices = tuple(EFFORT_LADDER) + (
        "status", "menu", "scope", "session", "global", "show", "hide",
        "detail", "full", "clamp", "reset", "--global", "--session",
    )
    registry.register(_spec(
        "reasoning", "Reasoning effort, scope and summary display (interactive menu)",
        reasoning_command, aliases=("think", "thinking", "t"),
        args_hint="[level|scope <global|session>|show|hide|detail|reset] [--global]",
        category="models",
        args=(ArgumentSpec("level", "effort level, scope or display action", variadic=True,
                           choices=reasoning_choices),),
    ))
    registry.register(_spec(
        "usage", "Show recorded model token/cost usage (read-only)",
        usage_command, args_hint="[--limit N]", category="models",
    ))
    registry.register(_spec(
        "context", "Explain context budget, transcript size and memory usage",
        context_command, aliases=("ctx",), category="models",
    ))

    # -- sessions ------------------------------------------------------------
    registry.register(_spec(
        "new", "Create a fresh durable conversation; the client switches to the returned id",
        new_command, aliases=("clear",), args_hint="[title]", category="session",
        risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "restart", "Restart the LEO service (delivery-safe: restarts after this reply)",
        restart_command, aliases=("reboot",), args_hint="", category="session",
        risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "update", "Check for updates; '/update apply' installs them and restarts",
        update_command, args_hint="[apply]", category="session", risk=RiskLevel.HIGH,
    ))
    registry.register(_spec(
        "reset", "Clear this conversation's messages in place (destructive, approval-gated)",
        reset_command, args_hint="", category="session", risk=RiskLevel.HIGH,
    ))
    registry.register(_spec(
        "sessions", "List durable conversations (switching is client-side)",
        sessions_command, aliases=("session", "switch"), args_hint="[--limit N]",
        category="session",
    ))
    registry.register(_spec(
        "history", "Show recent durable transcript rows for this conversation",
        history_command, args_hint="[limit]", category="session",
    ))
    registry.register(_spec(
        "export", "Export this conversation to one redacted md/json/txt file",
        export_command, aliases=("save", "export-session", "save-format"),
        args_hint="[md|json|txt] [--out name]", category="session",
        risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "compact", "Summarize older turns durably; /compact <focus> guides the summary",
        compact_command, aliases=("compress",), args_hint="[focus text] [--preview|--clear]",
        category="session", risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "title", "Show or set this conversation's title",
        title_command, aliases=("name",), args_hint="[title]", category="session",
        risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "stop", "Cancel a live run by id (or the current streaming run)",
        stop_command, aliases=("cancel",), args_hint="[run_id]", category="session",
        risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "diff", "Read workspace git changes through the audited git_diff tool",
        diff_command, args_hint="[--staged] [--stat] [path]", category="development",
    ))

    # -- skills --------------------------------------------------------------
    registry.register(_spec(
        "skills", "List, search, inspect, run, enable/disable or review skill history",
        skills_command, args_hint="[list|search <q>|inspect <name>|run <name> [k=v]|enable|disable|history]",
        category="skills",
    ))
    registry.register(_spec(
        "skill", "Explicitly invoke a named skill (also the fallback for /<skill-name>)",
        skill_command, args_hint="<name> [instruction]", category="skills",
        min_args=1, resource_arg=0,
    ))

    # -- memory --------------------------------------------------------------
    registry.register(_spec(
        "memory", "List, search, get, add or forget durable memories",
        memory_command, args_hint="[list|search <q>|get <id>|add <text>|forget <id>]",
        args=(ArgumentSpec("subcommand", "memory operation", variadic=True),),
        category="memory", risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "remember", "Store a memory directly (/memory add shorthand)",
        remember_command, args_hint="<text>", category="memory", min_args=1,
        risk=RiskLevel.MEDIUM,
    ))

    # -- tasks, agents, tools ------------------------------------------------
    registry.register(_spec(
        "tasks", "List, create, inspect, cancel or run durable tasks",
        tasks_command, aliases=("task",), args_hint="[list|create <objective>|status <id>|cancel <id>|run <id>]",
        category="tasks",
    ))
    registry.register(_spec(
        "agents", "List, inspect, review or delegate to LEO agents",
        agents_command, aliases=("subagents",),
        args_hint="[list|status <id>|history [id]|delegate <objective> [--agent id]]",
        category="agents",
    ))
    registry.register(_spec(
        "tools", "List registered tools; enable/disable is not supported",
        tools_command, args_hint="[list|search <q>]", category="tools",
    ))

    # -- approvals (trusted operator only) ----------------------------------
    registry.register(_spec(
        "approvals", "List pending approvals (operator CLI only)",
        approvals_command, aliases=("pending",), args_hint="[list]", category="security",
        surfaces=frozenset({"cli", "api"}), admin_only=True, operator_only=True,
    ))
    registry.register(_spec(
        "approve", "Approve a pending approval through the operator executor",
        approve_command, args_hint="<approval_id>", category="security",
        min_args=1, resource_arg=0, surfaces=frozenset({"cli", "api"}),
        admin_only=True, operator_only=True, busy_policy="idle",
    ))
    registry.register(_spec(
        "deny", "Deny a pending approval through the operator executor",
        deny_command, args_hint="<approval_id>", category="security",
        min_args=1, resource_arg=0, surfaces=frozenset({"cli", "api"}),
        admin_only=True, operator_only=True, busy_policy="idle",
    ))

    # -- observability -------------------------------------------------------
    registry.register(_spec(
        "events", "Read durable events (sensitive topics hidden from chat)",
        events_command, aliases=("event",), args_hint="[list|stats] [--topic t] [--limit N]",
        category="observability",
    ))
    registry.register(_spec(
        "metrics", "Show LEO metrics, model usage and tool call counters",
        metrics_command, category="observability",
    ))
    registry.register(_spec(
        "debug", "Local diagnostics only; uploads are explicitly unsupported",
        debug_command, aliases=("diagnostics",), args_hint="[local]", category="observability",
        admin_only=True,
    ))
    registry.register(_spec(
        "audit", "Read recent audit-log rows",
        audit_command, args_hint="[limit]", category="observability", admin_only=True,
    ))
    registry.register(_spec(
        "insights", "Summarize local usage counters",
        insights_command, args_hint="[days]", category="observability",
    ))

    # -- scheduling and integrations ----------------------------------------
    registry.register(_spec(
        "schedule", "Manage durable schedules (interval and cron-lite)",
        schedule_command, aliases=("cron", "schedules"),
        args_hint="[list|add <name> <seconds> <objective>|cron <name> <expr> <objective>|enable|disable|run|remove]",
        category="automation", risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "loop", "Create a durable recurring task (not an in-session loop)",
        loop_command, aliases=("proactive",),
        args_hint="[list|<seconds> <prompt> --name <schedule>|stop <name>]",
        category="automation", risk=RiskLevel.MEDIUM,
    ))
    registry.register(_spec(
        "channels", "Show channel adapters, integration records and outbox status",
        channels_command, aliases=("platforms", "gateway", "integrations"),
        args_hint="[list|<channel>]", category="integrations",
        admin_only=True,
    ))

    # -- explicitly unavailable, but discoverable ---------------------------
    categories = {
        "session": {"resume", "import", "retry", "undo", "rewind", "branch",
                    "steer", "queue", "bg", "btw", "quit"},
        "goals": {"goal", "subgoal", "personality", "reasoning", "fast", "verbose", "yolo",
                  "approvals_mode", "memory_review", "learn", "plan", "review", "refine"},
        "maintenance": {"init", "insights_upload", "reload", "reload-mcp", "reload-skills",
                        "update", "restart", "pause", "start", "discard", "answer", "topic",
                        "sethome", "platform", "handoff", "autoprune", "codex-runtime", "egress",
                        "profile", "snapshot"},
        "integrations": {"login", "subscription", "topup", "voice", "wake", "pet", "hatch",
                         "skin", "theme", "plugins", "bundles", "curator", "moa", "blueprint",
                         "kanban", "journey", "browser", "image", "paste", "copy", "prompt",
                         "palette", "connectors"},
        "development": {"worktree", "rollback"},
        "display": {"statusbar", "timestamps", "focus", "footer", "battery", "indicator",
                    "redraw", "density", "details", "logs", "mouse", "fortune", "terminal-setup",
                    "setup", "replay", "replay-diff", "heapdump", "mem", "widgets-reload",
                    "theme-info", "history_ui"},
    }
    category_for = {name: category for category, names in categories.items() for name in names}
    for name, aliases, args_hint, help_text, reason in UNAVAILABLE:
        registry.register(CommandSpec(
            name=name,
            help=help_text,
            aliases=aliases,
            args_hint=args_hint,
            category=category_for.get(name, "unavailable"),
            available=False,
            unavailable_reason=reason,
            handler=None,
        ))
    return registry


__all__ = ["UNAVAILABLE", "register_builtins"]
