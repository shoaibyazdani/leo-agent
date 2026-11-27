"""Handle an inline-keyboard tap, wherever the polling happens.

The tap arrives on the transport that polls Telegram. In a split deployment that
is the *gateway* process, which has adapters and a queue but no agent: wiring the
sink only from ``App`` meant every tap was answered "No handler for this button"
while the buttons sat there looking live.

So the logic lives here and both callers use it — the app (single process) and the
gateway (split), with the app passing its operator-resolve callable so the decision
still goes through the privileged tool path there.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from .. import db
from ..logging import get_logger
from ..security.approval_buttons import (
    ALWAYS_ARG_SCOPED_ACTIONS,
    SESSION_GRANT_SECONDS,
    is_authorized,
    parse_callback_data,
    resolve_locator,
    session_grant_scopable,
)
from ..security.approval_cards import (
    at_or_after,
    card_from_row,
    is_schedule_kind,
    normalize_choice,
    schedule_terminal_text,
)
from ..security.permissions import metadata_trustable

_LOG = get_logger("gateway.callbacks")

ResolveFn = Callable[[str, bool, str], Awaitable[Mapping[str, Any]]]
# A model tap is relayed as a normal slash command, so the switch happens in the
# process that owns settings instead of being written behind its back.
# (text, channel, subject, source_message_id). The source id is optional so a
# tap can be answered by editing the card that was tapped instead of posting a
# duplicate menu below it.
AdmitFn = Callable[..., Awaitable[None]]

MODEL_TAP_PREFIX = "mdl:"
REASONING_TAP_PREFIX = "rsn:"

# Only these tokens may become part of a relayed slash command. A button's
# callback payload is external input; it is parsed as data, never concatenated
# into an arbitrary command.
_REASONING_LEVELS = frozenset({
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
})
_REASONING_ACTIONS = {
    "show": "/reasoning show",
    "hide": "/reasoning hide",
    "detail": "/reasoning detail",
    # The menu labels are explicit: one resets this conversation, the other
    # clears the durable global default. Never let a scope toggle silently
    # change which one a button does.
    "reset": "/reasoning reset --session",
    "reset-global": "/reasoning reset --global",
    "menu": "/reasoning",
    "status": "/reasoning status",
}


def reasoning_tap_command(payload: str) -> str | None:
    """Translate a reasoning-button payload into its slash command.

    Returns None for anything not explicitly supported, so a crafted callback
    cannot smuggle another command through this prefix.
    """
    token = str(payload or "").strip()
    if not token:
        return None
    if token in _REASONING_ACTIONS:
        return _REASONING_ACTIONS[token]
    if token.startswith("level:"):
        level = token[len("level:"):].strip().lower()
        return f"/reasoning {level}" if level in _REASONING_LEVELS else None
    if token.startswith("scope:"):
        scope = token[len("scope:"):].strip().lower()
        return f"/reasoning scope {scope}" if scope in ("global", "session") else None
    return None


def approval_senders(settings: Any, channel: str) -> list[str]:
    """Who may resolve an approval by tapping on this channel.

    The channel allowlist is the authority. The environment variable is only a
    fallback for a deployment that has not written a policy yet — never a widening.
    """
    policies = getattr(getattr(settings, "channels", None), "policies", None) or {}
    policy = policies.get(channel) if isinstance(policies, Mapping) else None
    allow = list(getattr(policy, "allow", []) or [])
    if allow:
        return allow
    import os

    raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "") if channel == "telegram" else ""
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


async def resolve_tap_id(parsed: Mapping[str, str]) -> str | None:
    """Turn a button payload into a real approval id.

    Exact ids pass through. A 43-char locator is a hash, so it is accepted only
    when it reproduces for a pending approval that already exists.
    """
    approval_id = str(parsed.get("approval_id") or "")
    if parsed.get("is_ref") != "1":
        return approval_id or None
    rows = await db.fetchall("SELECT id FROM approvals WHERE status='pending'")
    candidates = [str(row["id"]) for row in rows or []]
    return resolve_locator(str(parsed.get("approval_kind") or ""), approval_id, candidates)


async def tap_blocked_reason(approval_id: str, timeout_seconds: float) -> str | None:
    """Why this button can no longer decide anything, or None when it still can.

    A button outlives the decision it was made for: it sits in chat history long
    after the approval was resolved, consumed or expired. Tapping it used to
    surface a raw exception name ("Could not resolve: ValueError"), which reads
    like a crash in LEO rather than "you already answered this". Status is read
    first so the operator gets the truth in one short line.
    """
    from datetime import datetime, timezone

    row = await db.fetchone(
        "SELECT status, created_at, kind, expires_at, shown_at FROM approvals WHERE id=?",
        (approval_id,),
    )
    if row is None:
        return "That approval no longer exists"
    status = str(row["status"] or "")
    if status == "approved":
        return "Already approved \u2014 no action needed"
    if status == "consumed":
        return "Already approved and used \u2014 no action needed"
    if status == "denied":
        return "Already rejected \u2014 no action needed"
    if status == "expired":
        return "Expired \u2014 the request timed out before it was answered"
    if status != "pending":
        return f"No longer pending ({status})"
    now = datetime.now(timezone.utc)
    if is_schedule_kind(row["kind"]):
        # A schedule card has its own deadline (min of the timeout and the
        # first run it showed). Show the truth before attempting a resolve.
        if at_or_after(row["expires_at"], now):
            return "Expired \u2014 this schedule card's deadline passed"
        if not row["shown_at"]:
            return (
                "This schedule card has not been shown on an answerable surface "
                "yet; open it in the TUI, app or Telegram"
            )
        return None
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
        age = (datetime.now(created.tzinfo) - created).total_seconds()
    except (TypeError, ValueError):
        return None
    if timeout_seconds > 0 and age > timeout_seconds:
        minutes = max(1, int(timeout_seconds) // 60)
        return f"Expired \u2014 this request was only valid for {minutes} min"
    return None


async def handle_tap(
    channel: str,
    descriptor: Mapping[str, Any],
    *,
    settings: Any,
    permissions: Any,
    adapters: Mapping[str, Any],
    resolve: ResolveFn | None = None,
    allowed: Iterable[str] | None = None,
    admit: AdmitFn | None = None,
) -> dict[str, Any]:
    """Authorize, resolve and acknowledge one tap. Never raises."""
    adapter = adapters.get(channel)
    if adapter is None:
        return {"ok": False, "reason": "unknown channel"}
    callback_id = str(descriptor.get("callback_id") or "")
    chat_id = str(descriptor.get("chat_id") or "")
    message_id = str(descriptor.get("message_id") or "")
    sender_id = str(descriptor.get("sender_id") or "")
    thread_id = str(descriptor.get("thread_id") or "")

    async def reply(note: str) -> None:
        if callback_id and hasattr(adapter, "answer_callback_query"):
            await adapter.answer_callback_query(callback_id, note)

    if not is_authorized(sender_id, list(allowed) if allowed is not None
                         else approval_senders(settings, channel)):
        await reply("You are not allowed to resolve approvals")
        _LOG.info("approval tap refused", extra={
            "channel": channel, "sender_id": sender_id, "reason": "not authorized"})
        return {"ok": False, "reason": "not authorized"}

    raw_data = str(descriptor.get("data") or "")
    if raw_data.startswith(REASONING_TAP_PREFIX):
        command = reasoning_tap_command(raw_data[len(REASONING_TAP_PREFIX):])
        if command is None:
            await reply("Unrecognised reasoning setting")
            return {"ok": False, "reason": "unrecognised reasoning payload"}
        if admit is None:
            await reply("This process cannot change reasoning; run /reasoning")
            return {"ok": False, "reason": "no admission path"}
        # The command runs in the process that owns settings and the current
        # conversation route, so a session-scoped tap lands in the right
        # conversation. Passing the source message id lets its reply replace
        # the menu card that was tapped instead of posting a second one.
        await admit(command, channel, chat_id or sender_id, message_id or None)
        await reply("Applying…")
        _LOG.info("reasoning tap relayed", extra={
            "channel": channel, "command": command.split()[1] if len(command.split()) > 1 else "menu",
            "sender_id": sender_id})
        return {"ok": True, "command": command}
    if raw_data.startswith(MODEL_TAP_PREFIX):
        return await _model_tap(
            raw_data[len(MODEL_TAP_PREFIX):], adapter=adapter, reply=reply,
            chat_id=chat_id, message_id=message_id, sender_id=sender_id,
            channel=channel, admit=admit, settings=settings,
        )

    parsed = parse_callback_data(descriptor.get("data"))
    if not parsed:
        await reply("Unrecognised button")
        return {"ok": False, "reason": "unrecognised payload"}

    approval_id = await resolve_tap_id(parsed)
    if not approval_id:
        await reply("That approval is no longer available")
        return {"ok": False, "reason": "unknown approval"}

    blocked = await tap_blocked_reason(
        approval_id, float(getattr(settings.security, "approval_timeout_seconds", 900) or 0)
    )
    if blocked:
        await reply(blocked)
        _LOG.info("approval tap ignored", extra={
            "channel": channel, "approval_id": approval_id, "reason": blocked})
        return {
            "ok": False, "reason": "not actionable", "note": blocked,
            **({"thread_id": thread_id} if thread_id else {}),
        }

    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    if row is None:
        await reply("Unknown approval")
        return {"ok": False, "reason": "unknown approval"}
    actor = f"{channel}-button:{sender_id}"

    if is_schedule_kind(row["kind"]):
        return await _resolve_schedule_tap(
            approval_id, parsed=parsed, row=row, actor=actor,
            channel=channel, sender_id=sender_id, thread_id=thread_id,
            chat_id=chat_id, message_id=message_id, adapter=adapter,
            permissions=permissions, reply=reply,
        )

    decision = str(parsed.get("decision") or "")
    granted = ""
    grant_label = ""
    if decision in {"session", "trust"}:
        row = await db.fetchone(
            "SELECT action, resource, metadata FROM approvals WHERE id=?", (approval_id,)
        )
        if row is None:
            await reply("Unknown approval")
            return {"ok": False, "reason": "unknown approval"}
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        digest = (
            str(metadata.get("arguments_digest") or "")
            if isinstance(metadata, Mapping) else ""
        )
        action_name = str(row["action"])
        if decision == "trust":
            if action_name not in ALWAYS_ARG_SCOPED_ACTIONS:
                await reply("Workspace trust is only available for shell commands")
                return {"ok": False, "reason": "workspace trust unavailable"}
            if not metadata_trustable(action_name, metadata):
                await reply(
                    "Workspace trust does not cover this command — approve it once"
                )
                return {"ok": False, "reason": "workspace trust unavailable"}
            days = max(
                1.0,
                float(getattr(settings.security, "workspace_trust_days", 30) or 30),
            )
            workspace = row["resource"] or str(settings.root)
            granted = await permissions.grant_workspace(
                action_name, workspace, int(days * 86400), actor,
            )
            grant_label = f"workspace for {int(days)}d"
        else:
            # "This command" names the operation: a path for path tools, the
            # canonical arguments digest for shell/terminal. Without that, the
            # grant would cover every future call, not the command the operator
            # is looking at. Refuse and leave the approval pending so
            # Approve-once still works.
            if not session_grant_scopable(action_name, row["resource"], digest):
                await reply("This command grant is not available — approve it once")
                _LOG.info("unscoped always refused", extra={
                    "channel": channel, "approval_id": approval_id,
                    "action": action_name})
                return {"ok": False, "reason": "unscoped approval"}
            if not metadata_trustable(action_name, metadata):
                await reply(
                    "This command grant does not cover a high-risk command — approve it once"
                )
                return {"ok": False, "reason": "hardline approval"}
            granted = await permissions.grant_session(
                action_name, row["resource"], SESSION_GRANT_SECONDS, actor,
                arguments_digest=digest or None,
            )
            grant_label = f"this command for {SESSION_GRANT_SECONDS // 3600}h"
        decision = "approve"

    approved = decision == "approve"
    try:
        if resolve is not None:
            result = await resolve(approval_id, approved, actor)
        else:
            decision_obj = await permissions.resolve_approval(
                approval_id, approved, resolver=actor
            )
            result = {"success": bool(decision_obj.allowed), "error": decision_obj.reason}
    except Exception as exc:  # noqa: BLE001 - a tap must never crash the poller
        await reply(f"Could not resolve: {type(exc).__name__}")
        _LOG.warning("approval tap failed", extra={
            "channel": channel, "approval_id": approval_id, "error": type(exc).__name__})
        return {"ok": False, "reason": type(exc).__name__}

    ok = bool(result.get("success"))
    verdict = "approved" if approved else "rejected"
    note = f"{verdict}: {approval_id}"
    if granted:
        note += f" · {grant_label}"
    await reply(note if ok else f"Failed: {result.get('error') or 'unknown error'}")

    # The message keeps the record; the buttons go away so a resolved approval
    # cannot be tapped twice.
    text = "\n".join(filter(None, [
        "🔐 Approval", f"id: {approval_id}",
        ("✅ " + verdict) if ok else "⚠️ could not resolve",
        note if not ok else "",
    ]))
    if chat_id and message_id and hasattr(adapter, "edit_message_text"):
        await adapter.edit_message_text(chat_id, message_id, text, {"inline_keyboard": []})
    _LOG.info("approval tap resolved", extra={
        "channel": channel, "approval_id": approval_id, "decision": verdict,
        "grant_id": granted or None, "sender_id": sender_id, "thread_id": thread_id})
    return {
        "ok": ok, "decision": verdict, "approval_id": approval_id,
        "grant_id": granted or None,
        **({"thread_id": thread_id} if thread_id else {}),
    }


async def _resolve_schedule_tap(
    approval_id: str,
    *,
    parsed: Mapping[str, str],
    row: Any,
    actor: str,
    channel: str,
    sender_id: str,
    thread_id: str,
    chat_id: str,
    message_id: str,
    adapter: Any,
    permissions: Any,
    reply: Callable[[str], Awaitable[None]],
) -> dict[str, Any]:
    """Resolve one schedule card tap through the manager's atomic path.

    The interactive callback carries only a choice and a card locator. The
    proposal digest comes from the durable row, never from the callback, and
    the manager re-checks shown/expiry/digest/choice under one transaction.
    Sender authorization already happened in :func:`handle_tap`.
    """
    try:
        card = card_from_row(row)
    except ValueError:
        await reply("This card cannot be decided")
        return {"ok": False, "reason": "malformed card"}
    choice = normalize_choice(str(parsed.get("decision") or ""), card.kind)
    if choice is None:
        await reply("That choice is not offered on this card")
        return {"ok": False, "reason": "invalid choice"}
    try:
        resolution = await permissions.resolve_schedule_card(
            approval_id,
            choice=choice,
            resolver=actor,
            confirm_proposal_digest=str(row["digest"] or ""),
            # A button tap carries no free text. Record that explicitly so a
            # denial never looks like a reason the human actually supplied.
            deny_reason=(
                "no reason supplied" if choice == "deny" else None
            ),
        )
    except PermissionError as exc:
        await reply(str(exc))
        _LOG.info("schedule tap refused", extra={
            "channel": channel, "approval_id": approval_id, "reason": str(exc)})
        return {"ok": False, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a tap must never crash the poller
        await reply(f"Could not resolve: {type(exc).__name__}")
        _LOG.warning("schedule tap failed", extra={
            "channel": channel, "approval_id": approval_id,
            "error": type(exc).__name__})
        return {"ok": False, "reason": type(exc).__name__}

    ok = resolution.status == "approved"
    verdict = "approved" if ok else "rejected"
    note = f"{verdict}: {approval_id} · {resolution.outcome or resolution.status}"
    if resolution.grant_id:
        note += " · standing allowance recorded"
    await reply(note)

    updated = await permissions.get_schedule_card(approval_id)
    if updated is not None:
        text = schedule_terminal_text(
            updated, status=resolution.status,
            grant_label=(
                "Standing allowance recorded" if resolution.grant_id else None
            ),
        )
    else:
        text = f"⏰ Schedule approval\nid: {approval_id}\n{verdict}"
    if chat_id and message_id and hasattr(adapter, "edit_message_text"):
        if str(channel) == "telegram":
            from ..integrations.messaging import TELEGRAM_MAX_CHARS

            if len(text) > TELEGRAM_MAX_CHARS:
                # Never silently clip a frozen card body in the terminal rewrite.
                # The full literal text remains in the row/dashboard.
                text = "\n".join([
                    str(card.title or "⏰ Schedule approval"),
                    f"id: {approval_id}",
                    "Full approval text is in the LEO dashboard.",
                    f"{verdict} · {resolution.outcome or resolution.status}",
                ])
        rewrite = adapter.edit_message_text
        try:
            names = inspect.signature(rewrite).parameters
            can_thread = "thread_id" in names or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in names.values()
            )
        except (TypeError, ValueError):
            can_thread = False
        extra = {"thread_id": thread_id or None} if can_thread else {}
        if str(channel) == "telegram" and can_thread:
            extra["plain_text"] = True
        await rewrite(chat_id, message_id, text, {"inline_keyboard": []}, **extra)
        try:
            from .server import mark_terminal_rewrite

            mark_terminal_rewrite(approval_id)
        except Exception:  # noqa: BLE001 - display bookkeeping never fails a tap
            pass
    _LOG.info("schedule tap resolved", extra={
        "channel": channel, "approval_id": approval_id, "decision": verdict,
        "choice": choice, "grant_id": resolution.grant_id,
        "sender_id": sender_id, "thread_id": thread_id})
    return {
        "ok": ok, "decision": verdict, "choice": choice,
        "approval_id": approval_id, "grant_id": resolution.grant_id,
        "schedule_id": resolution.schedule_id,
        **({"thread_id": thread_id} if thread_id else {}),
    }


async def _model_tap(
    name: str,
    *,
    adapter: Any,
    reply: Callable[[str], Awaitable[None]],
    chat_id: str,
    message_id: str,
    sender_id: str,
    channel: str,
    admit: AdmitFn | None,
    settings: Any,
) -> dict[str, Any]:
    """Apply a model chosen from the picker keyboard."""
    name = name.strip()
    if name.startswith("pg:"):
        # Pagination is a pure re-render, not a config change: edit the SAME
        # message in place with the new page's text + buttons. Queuing a fresh
        # `/model list <n>` left the old message behind as stale text and posted
        # a second picker below it. No config write means no admission path is
        # needed, so this works in both the split gateway and the single app.
        try:
            page = int(name[3:])
        except (ValueError, TypeError):
            page = 0
        from ..commands.builtins import _model_listing_message, _model_picker

        providers = settings.models.providers
        current = settings.models.default
        text = _model_listing_message(providers, current, page=page)
        keyboard = _model_picker(providers, current, page=page)
        if chat_id and message_id and hasattr(adapter, "edit_message_text"):
            await adapter.edit_message_text(chat_id, message_id, text, keyboard)
        await reply("")  # stop the tap spinner; the message itself changed
        return {"ok": True, "page": page}
    if not name:
        await reply("No model on that button")
        return {"ok": False, "reason": "empty model"}
    # Only an actual model switch writes config, so only it needs the privileged
    # admission path. Keep this guard below the pagination branch.
    if admit is None:
        await reply("This process cannot switch models; run /model <name>")
        return {"ok": False, "reason": "no admission path"}
    await admit(f"/model {name}", channel, chat_id or sender_id)
    await reply(f"Switching to {name}")
    if chat_id and message_id and hasattr(adapter, "edit_message_text"):
        await adapter.edit_message_text(
            chat_id, message_id,
            f"\U0001f4e6 Model: {name}\nApplying now — the confirmation follows below.",
            {"inline_keyboard": []},
        )
    _LOG.info("model tap relayed", extra={
        "channel": channel, "model": name, "sender_id": sender_id})
    return {"ok": True, "model": name}


def install(settings: Any, permissions: Any, adapters: Mapping[str, Any],
            resolve: ResolveFn | None = None,
            admit: AdmitFn | None = None) -> bool:
    """Attach the tap handler to every adapter that can receive one."""
    async def sink(channel: str, descriptor: Mapping[str, Any]) -> None:
        await handle_tap(channel, descriptor, settings=settings,
                         permissions=permissions, adapters=adapters, resolve=resolve,
                         admit=admit)

    installed = False
    for adapter in adapters.values():
        setter = getattr(adapter, "set_callback_sink", None)
        if callable(setter):
            setter(sink)
            installed = True
    if installed:
        _LOG.info("approval tap handler installed",
                  extra={"adapters": sorted(adapters), "channel_ok": True})
    return installed
