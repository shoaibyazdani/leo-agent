"""Push HIGH-risk approval requests to the operator's own channel.

Why this exists: ``PermissionManagerImpl.authorize`` records a pending approval
and emits ``approval.requested``, but nothing subscribed to that event. An
approval therefore only ever existed in SQLite and on ``GET /approvals`` — the
operator was never told, so a gated action silently stalled until it expired.

Design constraints, all deliberate:

* **Notify only.** The notifier never resolves, approves or denies anything. It
  cannot: it holds no permission manager. A model must never decide an approval,
  and neither may this path.
* **No recursion.** It calls the channel adapter's ``send`` directly rather than
  ``Gateway.send``. Routing through the gateway would re-enter
  ``authorize("gateway.send", HIGH)`` and create an approval *for the approval
  notification* — an infinite loop.
* **Fail soft.** A dead Telegram token must not break the tool call that needed
  the approval. The event bus already isolates handler exceptions; every send
  failure is caught here as well and only logged.
* **No secrets.** Only action, resource, risk, id and expiry are rendered; the
  same fields already exposed by ``pending_approvals()``.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from ..config import Settings
from ..contracts import INCAPABLE_SURFACE, SURFACE_CAPABILITIES, surface_capabilities
from ..events import Event, EventBus
from ..integrations.messaging import (
    MAX_MESSAGE_CHUNKS,
    TELEGRAM_MAX_CHARS,
    AdapterCapabilities,
    MessagingError,
    split_plain_text,
)
from ..logging import get_logger
from datetime import datetime
from .approval_buttons import build_keyboard, build_schedule_keyboard
from .approval_cards import (
    KIND_COMMAND,
    choices_for_kind,
    is_schedule_kind,
    kind_config,
    mark_card_deny_only,
)
from .notices import (
    APPROVAL_PREFIX,
    REVIEW_CREATED_PREFIX,
    REVIEW_FAILED_PREFIX,
    REVIEW_STAGED_PREFIX,
)
from .permissions import reason_is_trustable

_LOG = get_logger("security.notifier")

REQUESTED_TOPIC = "approval.requested"
# A shell/desktop banner is a notification, never a card presentation or proof
# that the human can answer. The event carries bounded copy and no authority.
SCHEDULE_NOTIFICATION_TOPIC = "schedule.notification"
# A background review/learn pass may apply a skill directly or stage it for
# approval. Its completion event carried no subscriber, so either outcome could
# go unnoticed. These topics are the announcement seam; the notifier still
# decides nothing on the operator's behalf.
REVIEW_TOPICS = (
    "review.completed",
    "review.failed",
    "learn.completed",
    "learn.failed",
)
# Cap how many skill writes one notice names: the rest are still listed by
# ``/skills pending`` and a chat message must stay readable on a phone.
MAX_NAMED_STAGED = 3
MAX_NAMED_CREATED = 3


class ApprovalNotifier:
    """Subscribe to approval requests and mirror them to the operator."""

    def __init__(
        self,
        gateway: Any,
        settings: Settings,
        events: EventBus | None = None,
        *,
        notify_only: bool | None = None,
    ) -> None:
        self.gateway = gateway
        self.settings = settings
        self.events = events if events is not None else getattr(gateway, "events", None)
        # ``None`` derives the mode from configuration. A notify-only
        # announcement is distinct from an operator-remote answerable target:
        # it never carries decision buttons and never records presentation.
        self.notify_only = notify_only
        self._subscribed = False
        self._sent = 0
        self._failed = 0
        self._skipped = 0

    # -- lifecycle ---------------------------------------------------------
    @property
    def config(self) -> Any:
        return getattr(self.settings.security, "notify", None)

    def _resolved(self) -> tuple[str, str] | None:
        """Return (channel, target) when notification is actually usable."""
        cfg = self.config
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        channel = str(getattr(cfg, "channel", "") or "").strip()
        target = str(getattr(cfg, "target", "") or "").strip()
        if not channel or not target:
            return None
        foreign = self._foreign_database()
        if foreign is not None:
            _LOG.info(f"approval notifications suppressed: {foreign}")
            return None
        return channel, target

    def _foreign_database(self) -> str | None:
        """Why this process must not announce, or None when it may.

        The notify target is the operator's own address, configured once; the
        database belongs to one process. A CLI run, a test suite or a second
        checkout inherits the operator's config, so it would announce an approval
        that only exists in its own scratch database — the operator taps the
        button and the gateway, reading the real one, answers "That approval no
        longer exists". Announcements are therefore bound to the database the
        operator's gateway actually reads.
        """
        import os
        from pathlib import Path

        from .. import db  # local import: db is initialised after this module loads

        if str(os.environ.get("LEO_NO_NOTIFY") or "").strip().lower() in {"1", "true", "yes", "on"}:
            return "LEO_NO_NOTIFY is set"
        # The owning database is the one the operator's gateway reads: an
        # explicit override when the deployment sets one, otherwise the path
        # the loaded configuration names. Without the config fallback a test or
        # ad-hoc script that inherits the operator's config and points
        # ``db.set_db_path`` at a scratch file claimed no owner at all, so it
        # announced approvals the gateway could never resolve — the live box
        # received empty cards exactly this way while a test suite ran with the
        # service environment.
        expected = str(os.environ.get("LEO_DB_PATH") or self.settings.db_path)
        try:
            current = db.get_db_path()
            claimed = Path(expected).expanduser().resolve()
        except (RuntimeError, OSError, ValueError):
            return None
        if current == claimed:
            return None
        return f"this process writes {current}, the notify target's database is {claimed}"

    def start(self) -> bool:
        """Attach the subscription once. Returns True when notifications are on."""
        if self.events is None:
            _LOG.warning("approval notifier has no event bus")
            return False
        if self._resolved() is None:
            _LOG.info("approval notifications disabled or incomplete; not subscribing")
            return False
        if not self._subscribed:
            self.events.subscribe(REQUESTED_TOPIC, self._on_requested)
            for topic in REVIEW_TOPICS:
                self.events.subscribe(topic, self._on_review)
            self._subscribed = True
            channel, target = self._resolved()  # type: ignore[misc]
            _LOG.info(
                "approval notifications enabled",
                extra={"channel": channel, "target": target},
            )
        return True

    def stop(self) -> None:
        if self._subscribed and self.events is not None:
            self.events.unsubscribe(REQUESTED_TOPIC, self._on_requested)
            for topic in REVIEW_TOPICS:
                self.events.unsubscribe(topic, self._on_review)
        self._subscribed = False

    # -- rendering ---------------------------------------------------------
    def _answerable(self, channel: str | None = None) -> bool:
        """Whether this channel can carry a real decision.

        An explicitly configured operator-remote target is answerable unless
        the operator set notify-only. A send-only / notifier-blind target is
        never answerable, so it gets copy with no buttons. Configuration alone
        cannot make an incapable surface answerable.
        """
        if self.notify_only is not None:
            configured = not self.notify_only
        else:
            cfg = self.config
            if cfg is None:
                return False
            mode = str(getattr(cfg, "mode", "answerable") or "answerable").strip().lower()
            enabled = bool(getattr(cfg, "answerable", True))
            configured = enabled and mode != "notify_only"
        if not configured:
            return False
        if channel is None:
            return True
        caps = surface_capabilities(channel)
        return bool(caps.can_present_card and caps.can_wait_for_decision and caps.full_text)

    def _message(self, payload: dict[str, Any], *, channel: str | None = None) -> str:
        kind = str(payload.get("kind") or KIND_COMMAND)
        if is_schedule_kind(kind):
            return self._schedule_message(payload, answerable=self._answerable(channel))
        return self._command_message(payload)

    def _schedule_message(self, payload: dict[str, Any], *, answerable: bool = True) -> str:
        """Full literal schedule copy. Never truncated, never re-interpreted."""
        kind = str(payload.get("kind") or "schedule_create")
        config = kind_config(kind)
        title = str(payload.get("title") or config.header)
        body = str(payload.get("body") or payload.get("command") or "")
        deadline = str(payload.get("deadline") or "").strip()
        approval_id = str(payload.get("approval_id") or "?")
        lines = [title]
        if body:
            lines.append(body)
        if deadline:
            lines += ["", deadline]
        lines.append(f"id: {approval_id}")
        if answerable:
            lines += ["", "Tap a button below, or review it in the LEO app."]
        else:
            lines += [
                "",
                "Notification only — review and decide in the LEO app or TUI. "
                "This message cannot approve anything.",
            ]
        return "\n".join(lines)

    def _command_message(self, payload: dict[str, Any]) -> str:
        approval_id = str(payload.get("approval_id") or "?")
        action = str(payload.get("action") or "?")
        resource = payload.get("resource")
        risk = str(payload.get("risk") or "?")
        command = str(payload.get("command") or "").strip()
        reason = str(payload.get("risk_reason") or "").strip()
        timeout = int(getattr(self.settings.security, "approval_timeout_seconds", 900))
        minutes = max(1, timeout // 60)
        lines = [
            APPROVAL_PREFIX,
            f"action: {action}",
        ]
        if command:
            lines.append(f"command: `{command[:400]}`")
        if resource:
            label = "workspace" if action in {"tool:shell", "tool:terminal"} else "resource"
            lines.append(f"{label}: {resource}")
        if reason:
            lines.append(f"why: {reason[:200]}")
        lines += [
            f"risk: {risk}",
            f"id: {approval_id}",
            f"expires: {minutes} min after creation",
            "",
            "Tap a button below, or from a terminal:",
            f"approve: leo approval approve {approval_id}",
            f"deny:    leo approval deny {approval_id}",
        ]
        return "\n".join(lines)

    # -- handler -----------------------------------------------------------
    async def _still_actionable(self, approval_id: str) -> bool:
        """True only while the approval is pending and inside its own timeout.

        Announcements can be replayed (a retry, a re-emitted event, an operator
        sweeping old rows). Pushing a decision the human can no longer make trains
        them to ignore the ones they can.
        """
        from .. import db  # local import: the notifier is imported before db is initialised

        try:
            row = await db.fetchone(
                "SELECT status, created_at, kind, expires_at FROM approvals WHERE id=?",
                (approval_id,),
            )
        except Exception:  # noqa: BLE001 - a lookup must not stop a real notification
            return True
        if row is None:
            return False
        if str(row["status"]) != "pending":
            return False
        if is_schedule_kind(row["kind"]):
            # A schedule card has its own deadline (min of the timeout and the
            # first run it showed). Never announce one that already passed.
            expires = str(row["expires_at"] or "").strip()
            if expires:
                try:
                    deadline = datetime.fromisoformat(expires)
                    if deadline.tzinfo is None:
                        return True
                    return datetime.now(deadline.tzinfo) < deadline
                except (TypeError, ValueError):
                    return True
            return True
        timeout = float(getattr(self.settings.security, "approval_timeout_seconds", 900))
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
            age = (datetime.now(created.tzinfo) - created).total_seconds()
        except (TypeError, ValueError):
            return True
        return age <= timeout

    @staticmethod
    def _accepts_plain(send: Any) -> bool:
        try:
            params = inspect.signature(send).parameters
        except (TypeError, ValueError):
            return False
        return "plain_text" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _fits_card(self, channel: str, adapter: Any, text: str) -> bool:
        """Whether this channel can present the *whole* literal card body."""
        caps = SURFACE_CAPABILITIES.get(channel, INCAPABLE_SURFACE)
        if caps.max_card_chars is not None and len(text) > caps.max_card_chars:
            return False
        if not caps.full_text and not AdapterCapabilities.for_adapter(adapter).full_text:
            return False
        try:
            chunks = split_plain_text(str(text), TELEGRAM_MAX_CHARS, MAX_MESSAGE_CHUNKS)
        except MessagingError:
            return False
        return bool(chunks) and "".join(chunks) == text

    async def _send_notice(
        self, adapter: Any, target: str, text: str,
        keyboard: dict[str, Any] | None, *, plain: bool,
    ) -> None:
        """Send once, with literal text on adapters that support it."""
        extra: dict[str, Any] = {}
        if keyboard is not None:
            extra["reply_markup"] = keyboard
        if plain and self._accepts_plain(adapter.send):
            extra["plain_text"] = True
        try:
            await adapter.send(target, text, **extra)
        except TypeError:
            if not extra:
                raise
            await adapter.send(target, text)
            _LOG.info(
                "approval notification sent without a keyboard",
                extra={"target_kind": type(target).__name__},
            )

    async def _on_requested(self, event: Event) -> None:
        resolved = self._resolved()
        if resolved is None:
            self._skipped += 1
            return
        channel, target = resolved
        payload = event.payload if isinstance(event.payload, dict) else {}
        if not await self._still_actionable(str(payload.get("approval_id") or "")):
            self._skipped += 1
            _LOG.info("approval notification skipped: no longer actionable", extra={
                "channel": channel, "approval_id": payload.get("approval_id")})
            return
        text = self._message(payload, channel=channel)
        kind = str(payload.get("kind") or KIND_COMMAND)
        answerable = self._answerable(channel)
        try:
            adapter = self.gateway._adapter(channel)  # noqa: SLF001 - same package
            action = str(payload.get("action") or "")
            keyboard: dict[str, Any] | None
            if is_schedule_kind(kind):
                choices = [
                    str(item) for item in (payload.get("choices") or [])
                    if str(item).strip()
                ] or list(choices_for_kind(kind))
                labels = [str(item) for item in (payload.get("choice_labels") or [])]
                approval_id = str(payload.get("approval_id") or "")
                plain = channel == "telegram"
                fits = not plain or self._fits_card(channel, adapter, text)
                deny_only = bool(payload.get("deny_only")) or not fits
                if not fits and not payload.get("deny_only"):
                    marked = await mark_card_deny_only(
                        approval_id, reason="full card body does not fit this channel"
                    )
                    if marked is not None:
                        payload = {**payload, "deny_only": True}
                if not fits:
                    # Keep the full text in the frozen card/dashboard only. This
                    # message says plainly what the operator can still do.
                    text = "\n".join([
                        str(payload.get("title") or "⏰ Schedule approval"),
                        f"id: {approval_id}",
                        "The full approval text is in the LEO dashboard. "
                        "This channel can only deny it.",
                    ])
                keyboard = (
                    build_schedule_keyboard(
                        approval_id,
                        approval_kind=kind,
                        choices=choices,
                        choice_labels=labels,
                        deny_only=deny_only,
                    )
                    if answerable
                    else None
                )
            else:
                keyboard = build_keyboard(
                    str(payload.get("approval_id") or ""),
                    action,
                    # Path approvals scope Always by resource; shell/terminal
                    # scope it by the exact arguments digest. Either must be present.
                    resource=payload.get("resource"),
                    arguments_digest=payload.get("arguments_digest"),
                    trust_days=float(
                        getattr(self.settings.security, "workspace_trust_days", 30) or 30
                    ),
                    # A destructive/credential/arbitrary-code call is not covered
                    # by a workspace trust, so do not offer a button that would lie.
                    allow_workspace_trust=reason_is_trustable(
                        payload.get("risk_reason")
                    ),
                    # manual mode ignores grants, so offering trust buttons there
                    # would promise something the resolver will not honor.
                    include_session=(
                        getattr(self.settings.security, "approval_mode", "smart")
                        == "smart"
                    ),
                )
            await self._send_notice(
                adapter, target, text, keyboard,
                plain=(channel == "telegram" and is_schedule_kind(kind)),
            )
        except Exception as exc:  # noqa: BLE001 - never break the requesting call
            self._failed += 1
            _LOG.warning(
                "approval notification failed",
                extra={
                    "channel": channel,
                    "approval_id": payload.get("approval_id"),
                    "error": str(exc),
                },
            )
            return
        self._sent += 1
        if is_schedule_kind(kind) and self.events is not None:
            # Shell notification path: bounded copy, no buttons, no shown_at.
            # Delivery of this event proves nothing about human attention or
            # answerability; only a real card presentation does that.
            from .approval_cards import card_from_row, schedule_notification_copy
            from .. import db

            approval_id = str(payload.get("approval_id") or "")
            row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
            try:
                card = card_from_row(row) if row is not None else None
            except ValueError:
                card = None
            if card is not None:
                title, body = schedule_notification_copy(card)
                await self.events.emit(
                    SCHEDULE_NOTIFICATION_TOPIC,
                    approval_id=approval_id,
                    kind=kind,
                    title=title,
                    body=body,
                    notify_only=not answerable,
                    presented=False,
                )
        _LOG.info(
            "approval notification sent",
            extra={"channel": channel, "approval_id": payload.get("approval_id")},
        )

    # -- review announcements ----------------------------------------------
    @staticmethod
    def _review_message(payload: dict[str, Any]) -> str | None:
        """One short notice for a finished background pass, or None to stay quiet.

        Silence is the default: a review that changed no skill is not news, and
        an announcement for every turn would train the operator to ignore the
        ones that matter. Applied creations, staged writes and failures are news.
        """
        status = str(payload.get("status") or "").strip().lower()
        mode = str(payload.get("mode") or "review").strip().lower()
        label = "learn" if mode == "learn" else "review"
        if status == "failed":
            error = " ".join(str(payload.get("error") or "").split())[:200]
            lines = [REVIEW_FAILED_PREFIX, f"pass: {label}"]
            if error:
                lines.append(f"error: {error}")
            lines.append("nothing was written; the artifact names the cause")
            return "\n".join(lines)
        created = [
            item for item in (payload.get("created") or [])
            if isinstance(item, Mapping) and item.get("skill")
        ]
        staged = [
            item for item in (payload.get("staged") or [])
            if isinstance(item, Mapping) and item.get("skill")
        ]
        if not created and not staged:
            return None
        lines = [REVIEW_CREATED_PREFIX if created and not staged else REVIEW_STAGED_PREFIX]
        for item in created[:MAX_NAMED_CREATED]:
            skill = " ".join(str(item.get("skill") or "?").split())[:64]
            category = " ".join(str(item.get("category") or "uncategorized").split())[:64]
            description = " ".join(str(item.get("description") or "").split())[:160]
            lines.append(f"{skill} [{category}]")
            if description:
                lines.append(description)
        remaining_created = len(created) - min(len(created), MAX_NAMED_CREATED)
        if remaining_created > 0:
            lines.append(f"+{remaining_created} more created")
        if staged:
            if created:
                lines.append(REVIEW_STAGED_PREFIX)
            for item in staged[:MAX_NAMED_STAGED]:
                skill = " ".join(str(item.get("skill") or "?").split())[:64]
                action = str(item.get("action") or "write").strip().lower()
                category = " ".join(str(item.get("category") or "").split())[:64]
                description = " ".join(str(item.get("description") or "").split())[:160]
                detail = f"{action}: {skill}"
                if category:
                    detail += f" [{category}]"
                lines.append(f"{detail} ({item.get('proposal_id')})")
                if description:
                    lines.append(description)
            remaining = len(staged) - len(staged[:MAX_NAMED_STAGED])
            if remaining > 0:
                lines.append(f"+{remaining} more")
            lines += [
                "not applied yet. review and apply with:",
                "/skills pending",
                f"/skills approve {staged[0].get('proposal_id')}",
            ]
        return "\n".join(lines)

    async def _on_review(self, event: Event) -> None:
        resolved = self._resolved()
        if resolved is None:
            self._skipped += 1
            return
        channel, target = resolved
        payload = event.payload if isinstance(event.payload, dict) else {}
        text = self._review_message(payload)
        if text is None:
            self._skipped += 1
            return
        try:
            adapter = self.gateway._adapter(channel)  # noqa: SLF001 - same package
            await adapter.send(target, text)
        except Exception as exc:  # noqa: BLE001 - never break the review
            self._failed += 1
            _LOG.warning(
                "review notification failed",
                extra={
                    "channel": channel,
                    "review_id": payload.get("review_id"),
                    "error": str(exc),
                },
            )
            return
        self._sent += 1
        _LOG.info(
            "review notification sent",
            extra={
                "channel": channel,
                "review_id": payload.get("review_id"),
                "status": payload.get("status"),
            },
        )

    def stats(self) -> dict[str, int | bool]:
        return {
            "subscribed": self._subscribed,
            "notify_only": not self._answerable(),
            "sent": self._sent,
            "failed": self._failed,
            "skipped": self._skipped,
        }


__all__ = [
    "ApprovalNotifier",
    "REQUESTED_TOPIC",
    "SCHEDULE_NOTIFICATION_TOPIC",
    "REVIEW_TOPICS",
]
