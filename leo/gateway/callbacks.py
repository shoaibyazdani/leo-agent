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

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from .. import db
from ..logging import get_logger
from ..security.approval_buttons import (
    SESSION_GRANT_SECONDS,
    is_authorized,
    parse_callback_data,
    resolve_locator,
)

_LOG = get_logger("gateway.callbacks")

ResolveFn = Callable[[str, bool, str], Awaitable[Mapping[str, Any]]]


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


async def handle_tap(
    channel: str,
    descriptor: Mapping[str, Any],
    *,
    settings: Any,
    permissions: Any,
    adapters: Mapping[str, Any],
    resolve: ResolveFn | None = None,
    allowed: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Authorize, resolve and acknowledge one tap. Never raises."""
    adapter = adapters.get(channel)
    if adapter is None:
        return {"ok": False, "reason": "unknown channel"}
    callback_id = str(descriptor.get("callback_id") or "")
    chat_id = str(descriptor.get("chat_id") or "")
    message_id = str(descriptor.get("message_id") or "")
    sender_id = str(descriptor.get("sender_id") or "")

    answered = False

    async def reply(note: str) -> None:
        """Answer the tap at most once: a second answer is always rejected."""
        nonlocal answered
        if answered or not callback_id:
            return
        if not hasattr(adapter, "answer_callback_query"):
            return
        answered = True
        await adapter.answer_callback_query(callback_id, note)

    async def show(text: str) -> None:
        """Replace the approval message, so the outcome outlives the toast."""
        if chat_id and message_id and hasattr(adapter, "edit_message_text"):
            await adapter.edit_message_text(
                chat_id, message_id, text, {"inline_keyboard": []}
            )

    if not is_authorized(sender_id, list(allowed) if allowed is not None
                         else approval_senders(settings, channel)):
        await reply("You are not allowed to resolve approvals")
        _LOG.info("approval tap refused", extra={
            "channel": channel, "sender_id": sender_id, "reason": "not authorized"})
        return {"ok": False, "reason": "not authorized"}

    parsed = parse_callback_data(descriptor.get("data"))
    if not parsed:
        await reply("Unrecognised button")
        return {"ok": False, "reason": "unrecognised payload"}

    approval_id = await resolve_tap_id(parsed)
    if not approval_id:
        await reply("That approval is no longer available")
        return {"ok": False, "reason": "unknown approval"}

    decision = str(parsed.get("decision") or "")
    actor = f"{channel}-button:{sender_id}"
    # Telegram expires a callback query ~15s after the tap. The resolve below can
    # take minutes (it goes through the privileged tool path), so the spinner is
    # stopped NOW and the outcome is written into the message instead of the
    # toast - otherwise every slow tap ends as "query is too old" on the phone
    # even though the approval really was resolved.
    await reply("")
    granted = ""
    if decision == "session":
        row = await db.fetchone(
            "SELECT action, resource FROM approvals WHERE id=?", (approval_id,)
        )
        if row is None:
            await show("\n".join(["🔐 Approval", f"id: {approval_id}",
                                  "⚠️ could not resolve", "reason: unknown approval"]))
            return {"ok": False, "reason": "unknown approval"}
        granted = await permissions.grant_session(
            str(row["action"]), row["resource"], SESSION_GRANT_SECONDS, actor,
        )
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
        _LOG.warning("approval tap failed", extra={
            "channel": channel, "approval_id": approval_id, "error": type(exc).__name__})
        await show("\n".join(["🔐 Approval", f"id: {approval_id}",
                              "⚠️ could not resolve",
                              f"reason: {type(exc).__name__}"]))
        return {"ok": False, "reason": type(exc).__name__}

    ok = bool(result.get("success"))
    verdict = "approved" if approved else "rejected"
    note = f"{verdict}: {approval_id}"
    if granted:
        note += f" · always for {SESSION_GRANT_SECONDS // 3600}h"
    _LOG.info("approval tap resolved", extra={
        "channel": channel, "approval_id": approval_id, "decision": verdict,
        "grant_id": granted or None, "sender_id": sender_id, "ok": ok})

    # The message keeps the record; the buttons go away so a resolved approval
    # cannot be tapped twice. This is the surface the operator actually reads,
    # because the toast is gone by now.
    text = "\n".join(filter(None, [
        "🔐 Approval", f"id: {approval_id}",
        ("✅ " + verdict) if ok else "⚠️ could not resolve",
        note if not ok else "",
        "" if ok else f"reason: {result.get('error') or 'unknown error'}",
    ]))
    await show(text)
    return {"ok": ok, "decision": verdict, "approval_id": approval_id,
            "grant_id": granted or None}


def install(settings: Any, permissions: Any, adapters: Mapping[str, Any],
            resolve: ResolveFn | None = None) -> bool:
    """Attach the tap handler to every adapter that can receive one."""
    async def sink(channel: str, descriptor: Mapping[str, Any]) -> None:
        await handle_tap(channel, descriptor, settings=settings,
                         permissions=permissions, adapters=adapters, resolve=resolve)

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
