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

from collections.abc import Mapping
from typing import Any

from ..config import Settings
from ..events import Event, EventBus
from ..logging import get_logger
from datetime import datetime
from .approval_buttons import build_keyboard
from .notices import (
    APPROVAL_PREFIX,
    REVIEW_FAILED_PREFIX,
    REVIEW_STAGED_PREFIX,
)
from .permissions import reason_is_trustable

_LOG = get_logger("security.notifier")

REQUESTED_TOPIC = "approval.requested"
# A background review/learn pass writes nothing an operator sees: its skill
# writes are staged in SQLite and its memory proposals only exist behind
# ``/skills pending``. The completion event carried no subscriber at all, so a
# staged write sat unnoticed until someone happened to ask. These topics are the
# announcement seam; the notifier still decides nothing on the operator's behalf.
REVIEW_TOPICS = (
    "review.completed",
    "review.failed",
    "learn.completed",
    "learn.failed",
)
# Cap how many staged writes one notice names: the rest are still listed by
# ``/skills pending`` and a chat message must stay readable on a phone.
MAX_NAMED_STAGED = 3


class ApprovalNotifier:
    """Subscribe to approval requests and mirror them to the operator."""

    def __init__(
        self,
        gateway: Any,
        settings: Settings,
        events: EventBus | None = None,
    ) -> None:
        self.gateway = gateway
        self.settings = settings
        self.events = events if events is not None else getattr(gateway, "events", None)
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
    def _message(self, payload: dict[str, Any]) -> str:
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
                "SELECT status, created_at FROM approvals WHERE id=?", (approval_id,)
            )
        except Exception:  # noqa: BLE001 - a lookup must not stop a real notification
            return True
        if row is None:
            return False
        if str(row["status"]) != "pending":
            return False
        timeout = float(getattr(self.settings.security, "approval_timeout_seconds", 900))
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
            age = (datetime.now(created.tzinfo) - created).total_seconds()
        except (TypeError, ValueError):
            return True
        return age <= timeout

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
        text = self._message(payload)
        try:
            adapter = self.gateway._adapter(channel)  # noqa: SLF001 - same package
            action = str(payload.get("action") or "")
            keyboard = build_keyboard(
                str(payload.get("approval_id") or ""),
                action,
                # Path approvals scope Always by resource; shell/terminal scope
                # it by the exact arguments digest. Either must be present.
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
            try:
                await adapter.send(target, text, reply_markup=keyboard)
            except TypeError:
                # A channel adapter written before keyboards existed still gets
                # the request: the text carries the approval id, so the operator
                # can resolve it from the CLI. Losing the notification would be
                # worse than losing the buttons.
                await adapter.send(target, text)
                _LOG.info(
                    "approval notification sent without a keyboard",
                    extra={"channel": channel, "approval_id": payload.get("approval_id")},
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
        _LOG.info(
            "approval notification sent",
            extra={"channel": channel, "approval_id": payload.get("approval_id")},
        )

    # -- review announcements ----------------------------------------------
    @staticmethod
    def _review_message(payload: dict[str, Any]) -> str | None:
        """One short line for a finished background pass, or None to stay quiet.

        Silence is the default: a review that wrote nothing is not news, and an
        announcement for every turn would train the operator to ignore the ones
        that matter. Only a staged skill write or a real failure is announced.
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
        staged = [item for item in (payload.get("staged") or []) if isinstance(item, Mapping)]
        if not staged:
            return None
        shown = staged[:MAX_NAMED_STAGED]
        lines = [REVIEW_STAGED_PREFIX]
        for item in shown:
            skill = " ".join(str(item.get("skill") or "?").split())[:64]
            action = str(item.get("action") or "write").strip().lower()
            lines.append(f"{action}: {skill} ({item.get('proposal_id')})")
        remaining = len(staged) - len(shown)
        if remaining > 0:
            lines.append(f"+{remaining} more")
        lines += [
            "not applied yet. review and apply with:",
            f"/skills pending",
            f"/skills approve {shown[0].get('proposal_id')}",
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
            "sent": self._sent,
            "failed": self._failed,
            "skipped": self._skipped,
        }


__all__ = ["ApprovalNotifier", "REQUESTED_TOPIC", "REVIEW_TOPICS"]
