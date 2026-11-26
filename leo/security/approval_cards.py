"""Frozen ApprovalCard v1 protocol, schedule consent copy and projections.

One card protocol serves the database, API, events, TUI, web/Electron,
Telegram callbacks and restart. A card is a *projection* of a durable
``approvals`` row — the row is still the one authority — and it never grants
anything by itself. The resolving path lives in
:class:`leo.security.permissions.PermissionManagerImpl`, which binds the
decision to the proposal digest, the deadline, the shown state and the chosen
choice in one transaction.

The copy is centralized English. The upstream oracle fixes the *contract*
(kinds, choices, labels, deadline semantics, terminal outcomes); LEO's own
wording stays LEO-native.
"""

from __future__ import annotations

import contextvars
import hmac
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from ..contracts import surface_capabilities

from pydantic import BaseModel, ConfigDict, Field

APPROVAL_CARD_PROTOCOL_VERSION = 1
CARD_CAPABILITY_VERSION = 1

KIND_SCHEDULE_CREATE = "schedule_create"
KIND_SCHEDULE_EDIT = "schedule_edit"
KIND_COMMAND = "command"
CardKind = Literal["schedule_create", "schedule_edit", "command"]
CARD_KINDS: tuple[str, ...] = (KIND_SCHEDULE_CREATE, KIND_SCHEDULE_EDIT, KIND_COMMAND)
SCHEDULE_KINDS = frozenset({KIND_SCHEDULE_CREATE, KIND_SCHEDULE_EDIT})

# Canonical choice tokens. These are the values a callback/API sends; labels
# are presentation-only and never parsed as authority.
SCHEDULE_CREATE_CHOICES: tuple[str, ...] = ("once", "session", "always", "deny")
SCHEDULE_EDIT_CHOICES: tuple[str, ...] = ("once", "deny")
COMMAND_CHOICES: tuple[str, ...] = ("once", "deny")

CHOICE_LABELS: dict[str, str] = {
    "once": "Allow",
    "session": "Allow for this chat",
    "always": "Always allow new jobs",
    "deny": "Deny",
}
COMMAND_CHOICE_LABELS: dict[str, str] = {"once": "Approve", "deny": "Deny"}

# Tolerant aliases for callers that use the ordinary approval vocabulary. The
# canonical stored choice is always the value from the kind's choice tuple.
_CHOICE_ALIASES: dict[str, str] = {
    "allow": "once",
    "approve": "once",
    "approved": "once",
    "reject": "deny",
    "rejected": "deny",
    "decline": "deny",
}

DECLINE_MESSAGES: dict[str, str] = {
    "denied": (
        "The user declined this on the schedule card; nothing was saved. Tell them "
        "in one line what you wanted to schedule, and do not retry it."
    ),
    "timeout": (
        "The user did not answer the schedule card in time; nothing was saved. "
        "Do not retry it."
    ),
    "other": "The schedule card was not approved; nothing was saved. Do not retry it.",
    "stale_create": (
        "The schedule card was answered after the first run it showed had passed; "
        "nothing was saved. Offer the job again with a later time if it is still wanted."
    ),
    "stale_edit": (
        "The change card was answered after the new run time it showed had passed; "
        "nothing was saved. Offer the change again with a later time if it is still wanted."
    ),
}


@dataclass(frozen=True)
class CardKindConfig:
    """Frozen behavior for one approval kind (not locale catalogs)."""

    kind: str
    header: str
    choices: tuple[str, ...]
    labels: tuple[str, ...]
    deadline_template: str
    timeout_notice_template: str
    allow_outcome: str
    deny_outcome: str
    timeout_outcome: str
    stale_outcome: str
    deny_only_outcome: str
    terminal_allow: str
    terminal_deny: str


KIND_CONFIG: dict[str, CardKindConfig] = {
    KIND_SCHEDULE_CREATE: CardKindConfig(
        kind=KIND_SCHEDULE_CREATE,
        header="LEO wants to schedule a job",
        choices=SCHEDULE_CREATE_CHOICES,
        labels=tuple(CHOICE_LABELS[item] for item in SCHEDULE_CREATE_CHOICES),
        deadline_template=(
            "If you don't answer within {window} the job will NOT be created."
        ),
        timeout_notice_template=(
            "⌛ Approval timed out after {window} — the job was not created. "
            "Ask me to try again if you still want it, or raise approvals.timeout "
            "in config.yaml."
        ),
        allow_outcome="created",
        deny_outcome="denied — the job was not created",
        timeout_outcome="timed out — the job was not created",
        stale_outcome="stale — the job was not created",
        deny_only_outcome=(
            "denied — the full job could not be shown on this surface"
        ),
        terminal_allow="✅ approved",
        terminal_deny="⛔ denied",
    ),
    KIND_SCHEDULE_EDIT: CardKindConfig(
        kind=KIND_SCHEDULE_EDIT,
        header="LEO wants to change a scheduled job",
        choices=SCHEDULE_EDIT_CHOICES,
        labels=tuple(CHOICE_LABELS[item] for item in SCHEDULE_EDIT_CHOICES),
        deadline_template=(
            "If you don't answer within {window} the job will NOT be changed."
        ),
        timeout_notice_template=(
            "⌛ Approval timed out after {window} — the job was not changed. "
            "Ask me to try again if you still want it, or raise approvals.timeout "
            "in config.yaml."
        ),
        allow_outcome="changed",
        deny_outcome="denied — the job was not changed",
        timeout_outcome="timed out — the job was not changed",
        stale_outcome="stale — the job was not changed",
        deny_only_outcome=(
            "denied — the full change could not be shown on this surface"
        ),
        terminal_allow="✅ approved",
        terminal_deny="⛔ denied",
    ),
    KIND_COMMAND: CardKindConfig(
        kind=KIND_COMMAND,
        header="Approval needed",
        choices=COMMAND_CHOICES,
        labels=tuple(COMMAND_CHOICE_LABELS[item] for item in COMMAND_CHOICES),
        deadline_template="If you don't answer within {window} the action will NOT run.",
        timeout_notice_template=(
            "⌛ Approval timed out after {window} — the action was not run."
        ),
        allow_outcome="approved",
        deny_outcome="denied",
        timeout_outcome="timed out — the action was not run",
        stale_outcome="expired — the action was not run",
        deny_only_outcome="denied",
        terminal_allow="✅ approved",
        terminal_deny="⛔ denied",
    ),
}


# A split worker has no channel adapters: the process that owns them presents
# the card. This task-local delegate is transport plumbing only — the approval
# row and PermissionManager remain the single authority. When it is absent,
# presentation happens in-process exactly as before.
CardPresenter = Callable[[str, str], Awaitable[bool]]
_CURRENT_PRESENTER: contextvars.ContextVar[CardPresenter | None] = contextvars.ContextVar(
    "leo_card_presenter", default=None
)


def current_card_presenter() -> CardPresenter | None:
    return _CURRENT_PRESENTER.get()


@contextmanager
def presenting_cards_via(presenter: CardPresenter | None) -> Iterator[None]:
    """Scope a transport presenter to one task (e.g. one split-worker turn)."""
    token = _CURRENT_PRESENTER.set(presenter)
    try:
        yield
    finally:
        _CURRENT_PRESENTER.reset(token)


def is_schedule_kind(kind: str | None) -> bool:
    return str(kind or "").strip() in SCHEDULE_KINDS


def kind_config(kind: str) -> CardKindConfig:
    try:
        return KIND_CONFIG[str(kind)]
    except KeyError:
        raise ValueError(f"unknown approval card kind: {kind!r}") from None


def normalize_choice(choice: Any, kind: str) -> str | None:
    """Map a caller choice onto the canonical token for this kind.

    Returns ``None`` for a wrong/unknown choice so the resolver can refuse it
    instead of guessing. Aliases exist only for the ordinary allow/deny
    vocabulary; they never widen the kind's frozen choice set.
    """
    config = kind_config(kind)
    text = str(choice or "").strip().lower()
    if not text:
        return None
    text = _CHOICE_ALIASES.get(text, text)
    return text if text in config.choices else None


def choices_for_kind(kind: str, *, deny_only: bool = False) -> list[str]:
    config = kind_config(kind)
    return ["deny"] if deny_only else list(config.choices)


def choice_labels_for(kind: str, choices: Sequence[str]) -> list[str]:
    config = kind_config(kind)
    labels = dict(zip(config.choices, config.labels, strict=False))
    return [labels.get(str(choice), str(choice)) for choice in choices]


def format_window(seconds: float | int | None) -> str:
    """Human window copy ("5 minutes", "2 hours") without locale machinery."""
    try:
        value = int(float(seconds if seconds is not None else 900))
    except (TypeError, ValueError):
        value = 900
    value = max(1, value)
    if value % 3600 == 0:
        count, unit = value // 3600, "hour"
    elif value % 60 == 0:
        count, unit = value // 60, "minute"
    else:
        count, unit = value, "second"
    return f"{count} {unit}{'' if count == 1 else 's'}"


def deadline_text(kind: str, seconds: float | int | None) -> str:
    return kind_config(kind).deadline_template.format(window=format_window(seconds))


def timeout_notice(kind: str, seconds: float | int | None) -> str:
    return kind_config(kind).timeout_notice_template.format(
        window=format_window(seconds)
    )


def parse_instant(value: Any) -> datetime | None:
    """Parse a stored ISO instant into an aware UTC datetime (or None)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def at_or_after(instant: Any, now: datetime) -> bool:
    """True when ``instant`` is at or before ``now`` (a deadline is inclusive)."""
    moment = parse_instant(instant)
    if moment is None:
        return False
    return now >= moment


def format_first_run(value: Any) -> str | None:
    moment = parse_instant(value)
    if moment is None:
        return None
    return moment.strftime("%a %b %d %H:%M UTC")


def delivery_description(described: str | None) -> str:
    """Map a DeliveryPlan description onto honest card copy.

    Phase 2's resolver deliberately reports ``no destination``/``local history
    only``; the card spells the same fact out so a human is never told a result
    will arrive somewhere it will not.
    """
    text = str(described or "").strip()
    lowered = text.lower()
    if not text or lowered in {"no destination", "nowhere", "none"}:
        return "nowhere (saved to the job's history only)"
    if lowered.startswith("local history only"):
        remainder = text[len("local history only"):].strip(" ;")
        base = "local history only (saved to the job's history)"
        return f"{base}; {remainder}" if remainder else base
    return text


@dataclass(frozen=True)
class ScheduleBodySpec:
    """The semantic inputs of a literal schedule card body.

    Phase 5 builds this from a compiled :class:`ScheduleSpec`; Phase 4 only
    renders and stores the resulting literal text. Every field here has a real
    runtime consumer in a later phase, so the card never promises behavior the
    scheduler does not perform.
    """

    name: str
    display: str
    first_run_at: Any = None
    repeat_limit: int | None = None
    paused: bool = False
    prompt: str = ""
    skills: Sequence[str] = ()
    script: str | None = None
    no_agent: bool = False
    monitor: str | None = None
    workdir: str | None = None
    context_from: Sequence[str] = ()
    enabled_toolsets: Sequence[str] | None = None
    attach_to_session: bool = False
    delivery: str = "nowhere (saved to the job's history only)"
    failure_delivery: str | None = None


def render_schedule_body(spec: ScheduleBodySpec) -> str:
    """Render the complete literal create-card body (never truncated here).

    The body is stored verbatim on the card. Surface size limits may later make
    a card deny-only, but the stored text is always the full text.
    """
    parts: list[str] = [str(spec.display or "").strip()]
    if spec.paused:
        parts.append("created paused: it will not run until resumed")
    else:
        first = format_first_run(spec.first_run_at)
        if first:
            parts.append(f"first run {first}")
    if spec.repeat_limit is not None:
        count = int(spec.repeat_limit)
        parts.append(f"{count} run{'' if count == 1 else 's'}")
    head = f'Schedule "{spec.name}" ({", ".join(part for part in parts if part)}): '

    prompt = str(spec.prompt or "").strip()
    script = str(spec.script or "").strip()
    pieces: list[str] = []
    if spec.no_agent:
        if script:
            pieces.append(f"runs script {script}")
        elif prompt:
            pieces.append(prompt)
    elif prompt:
        pieces.append(prompt)
        if spec.skills:
            pieces.append(f"(skills: {', '.join(str(item) for item in spec.skills)})")
        if script:
            pieces.append(f"(after running script {script})")
    elif script:
        pieces.append(f"runs script {script}")

    body = head + " ".join(pieces) if pieces else head.rstrip()
    if str(spec.monitor or "").strip():
        body += f"; checks URL {str(spec.monitor).strip()} first each run"
    if str(spec.workdir or "").strip():
        body += f"; runs in {str(spec.workdir).strip()}"
    if spec.context_from:
        body += f"; reads output of {', '.join(str(item) for item in spec.context_from)}"
    if spec.enabled_toolsets is not None:
        toolsets = [str(item) for item in spec.enabled_toolsets if str(item).strip()]
        body += "; toolsets: " + (", ".join(toolsets) if toolsets else "none")
    if spec.attach_to_session:
        body += "; copied into the chat's history"
    body += f"; results go to {spec.delivery}"
    if spec.failure_delivery and spec.failure_delivery != spec.delivery:
        body += f"; failure notices go to {spec.failure_delivery}"
    return body


def render_edit_body(name: str, changes: Sequence[str]) -> str:
    rendered = "; ".join(str(item).strip().rstrip(";") for item in changes if str(item).strip())
    return f'Change "{name}": {rendered}'


# A display surface may paginate a long literal body, but it may never trim it.
# Twelve pages is a bounded presentation budget; beyond it the card is denied
# rather than shown in a form nobody can review in one sitting.
MAX_CARD_PAGES = 12


def paginate_text(text: str, max_chars: int | None) -> list[str]:
    """Split literal text into lossless, surface-sized pages.

    Newlines and sentence boundaries are preferred, but an unbroken long token
    is still split. ``"".join(paginate_text(t, n)) == t`` for every input.
    """
    body = str(text or "")
    limit = int(max_chars) if isinstance(max_chars, int) and max_chars > 0 else len(body) or 1
    if len(body) <= limit:
        return [body]
    pages: list[str] = []
    rest = body
    while rest:
        if len(rest) <= limit:
            pages.append(rest)
            break
        window = rest[:limit]
        candidates = [
            window.rfind("\n") + 1,
            window.rfind(". ") + 2,
            window.rfind("; ") + 2,
        ]
        cut = max(candidates)
        if cut < limit // 2:
            cut = limit
        pages.append(rest[:cut])
        rest = rest[cut:]
    return pages


def card_pages(card: ApprovalCard | Mapping[str, Any], surface: str) -> list[str]:
    """The literal body split to one surface's declared card size."""
    body = str(card.get("body") or "") if isinstance(card, Mapping) else str(card.body or "")
    return paginate_text(body, surface_capabilities(surface).max_card_chars)


def card_fits_surface(card: ApprovalCard | Mapping[str, Any], surface: str) -> bool:
    """False when a card cannot be reviewed within this surface's page budget."""
    caps = surface_capabilities(surface)
    if not caps.can_present_card or not caps.full_text:
        return False
    return 0 < len(card_pages(card, surface)) <= MAX_CARD_PAGES


def schedule_card_text(card: ApprovalCard) -> str:
    """Full literal card copy for an answerable channel (never truncated)."""
    lines = [str(card.title or "Schedule approval"), str(card.body or "").rstrip()]
    deadline = str(card.deadline or "").strip()
    if deadline:
        lines += ["", deadline]
    lines += ["", f"id: {card.approval_id}", "Tap a button below, or decide in the LEO app."]
    return "\n".join(line for line in lines if line is not None)


def schedule_notification_copy(card: ApprovalCard, *, max_body: int = 1000) -> tuple[str, str]:
    """Short shell-notification copy. A notification is not a card or an answer.

    The full literal body stays on the durable row; this bounded preview is for
    a desktop banner only and never marks a card as shown or answerable.
    """
    title = f"⏰ {str(card.title or 'Schedule approval').strip()}"[:120]
    body = str(card.body or "").strip()[: max(1, int(max_body))]
    suffix = "" if len(str(card.body or "")) <= max_body else " … (full text in LEO)"
    return title, f"{body}{suffix}\nDecision: in the LEO app (this is a notification only)"


class ScheduleChoice(BaseModel):
    """One selectable decision on a projected card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    label: str


class ScheduleCardView(BaseModel):
    """The shared read-only projection rendered by API, TUI, web and Electron."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: int = APPROVAL_CARD_PROTOCOL_VERSION
    capability_version: int = CARD_CAPABILITY_VERSION
    approval_id: str
    proposal_id: str | None = None
    kind: CardKind
    status: str = "pending"
    outcome: str | None = None
    title: str
    body: str
    choices: list[ScheduleChoice] = Field(default_factory=list)
    deny_only: bool = False
    card_revision: int = 1
    proposal_digest: str | None = None
    schedule_id: str | None = None
    config_revision: int | None = None
    proposed_at: str | None = None
    deadline: str | None = None
    expires_at: str | None = None
    shown_first_run: str | None = None
    shown_at: str | None = None
    presented_at: str | None = None
    presented_surface: str | None = None
    answered_at: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None
    origin: str | None = None
    requested_by: str | None = None
    resolver: str | None = None
    decision: str | None = None
    deny_reason: str | None = None
    acknowledged_at: str | None = None
    acknowledged_surface: str | None = None


class ApprovalCard(BaseModel):
    """Frozen ApprovalCard v1.

    Unknown fields are rejected. A row that cannot be projected into this shape
    fails closed rather than being partially displayed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: int = APPROVAL_CARD_PROTOCOL_VERSION
    capability_version: int = CARD_CAPABILITY_VERSION
    approval_id: str
    proposal_id: str | None = None
    kind: CardKind = KIND_COMMAND
    status: str = "pending"
    outcome: str | None = None
    title: str = ""
    body: str = ""
    choices: list[str] = Field(default_factory=list)
    choice_labels: list[str] = Field(default_factory=list)
    deny_only: bool = False
    card_revision: int = 1
    proposal_digest: str | None = None
    schedule_id: str | None = None
    config_revision: int | None = None
    proposed_at: str | None = None
    deadline: str | None = None
    expires_at: str | None = None
    shown_first_run: str | None = None
    shown_at: str | None = None
    presented_at: str | None = None
    presented_surface: str | None = None
    answered_at: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None
    origin: str | None = None
    requested_by: str | None = None
    resolver: str | None = None
    decision: str | None = None
    deny_reason: str | None = None
    acknowledged_at: str | None = None
    acknowledged_surface: str | None = None

    def view(self) -> ScheduleCardView:
        labels = list(self.choice_labels)
        choices: list[ScheduleChoice] = []
        for index, value in enumerate(self.choices):
            label = labels[index] if index < len(labels) else str(value)
            choices.append(ScheduleChoice(value=str(value), label=str(label)))
        return ScheduleCardView(
            protocol_version=self.protocol_version,
            capability_version=self.capability_version,
            approval_id=self.approval_id,
            proposal_id=self.proposal_id,
            kind=self.kind,
            status=self.status,
            outcome=self.outcome,
            title=self.title,
            body=self.body,
            choices=choices,
            deny_only=self.deny_only,
            card_revision=self.card_revision,
            proposal_digest=self.proposal_digest,
            schedule_id=self.schedule_id,
            config_revision=self.config_revision,
            proposed_at=self.proposed_at,
            deadline=self.deadline,
            expires_at=self.expires_at,
            shown_first_run=self.shown_first_run,
            shown_at=self.shown_at,
            presented_at=self.presented_at,
            presented_surface=self.presented_surface,
            answered_at=self.answered_at,
            conversation_id=self.conversation_id,
            run_id=self.run_id,
            tool_call_id=self.tool_call_id,
            origin=self.origin,
            requested_by=self.requested_by,
            resolver=self.resolver,
            decision=self.decision,
            deny_reason=self.deny_reason,
            acknowledged_at=self.acknowledged_at,
            acknowledged_surface=self.acknowledged_surface,
        )

    def event_payload(self) -> dict[str, Any]:
        """Payload copied onto approval events; no secrets, no authority."""
        return {
            "protocol_version": self.protocol_version,
            "capability_version": self.capability_version,
            "approval_id": self.approval_id,
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "status": self.status,
            "outcome": self.outcome,
            "title": self.title,
            "body": self.body,
            "choices": list(self.choices),
            "choice_labels": list(self.choice_labels),
            "deny_only": self.deny_only,
            "card_revision": self.card_revision,
            "proposal_digest": self.proposal_digest,
            "schedule_id": self.schedule_id,
            "config_revision": self.config_revision,
            "proposed_at": self.proposed_at,
            "deadline": self.deadline,
            "expires_at": self.expires_at,
            "shown_first_run": self.shown_first_run,
            "shown_at": self.shown_at,
            "presented_surface": self.presented_surface,
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
            "tool_call_id": self.tool_call_id,
            "origin": self.origin,
            "requested_by": self.requested_by,
        }


def schedule_terminal_text(
    card: ApprovalCard,
    *,
    status: str,
    grant_label: str | None = None,
    note: str | None = None,
) -> str:
    """Kind-preserving terminal rewrite: title + full literal body + outcome."""
    config = kind_config(card.kind)
    if status == "approved":
        verdict = config.terminal_allow
    elif status == "expired":
        verdict = "⌛ expired"
    else:
        verdict = config.terminal_deny
    lines = [str(card.title or config.header).strip(), str(card.body or "").rstrip()]
    if card.deny_reason:
        lines.append(f"reason: {card.deny_reason}")
    lines += ["", f"{verdict} · {card.outcome or status}"]
    if grant_label:
        lines.append(grant_label)
    if note:
        lines.append(note)
    return "\n".join(line for line in lines if line is not None)


def _row_value(row: Any, key: str) -> Any:
    """Read one column from a mapping or an ``sqlite3.Row``."""
    if isinstance(row, Mapping):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def proposal_digest_matches(row: Any, supplied: Any) -> bool:
    """Constant-time compare of the proposal digest shown to the human."""
    expected = str(_row_value(row, "digest") or "").strip()
    candidate = str(supplied or "").strip()
    if not expected or not candidate:
        return False
    return hmac.compare_digest(expected, candidate)


def _load_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _command_card(row: Mapping[str, Any]) -> ApprovalCard:
    """Legacy command approvals projected into the same frozen shape."""
    metadata = _load_json(row.get("metadata"))
    metadata = metadata if isinstance(metadata, dict) else {}
    action = str(row.get("action") or "?")
    resource = row.get("resource")
    risk = str(row.get("risk") or "?")
    lines = [f"action: {action}"]
    command = str(metadata.get("command_preview") or "").strip()
    if command:
        lines.append(f"command: {command}")
    if resource:
        lines.append(f"resource: {resource}")
    reason = str(row.get("reason") or metadata.get("risk_reason") or "").strip()
    if reason:
        lines.append(f"why: {reason}")
    lines.append(f"risk: {risk}")
    config = KIND_CONFIG[KIND_COMMAND]
    return ApprovalCard(
        approval_id=str(row.get("id") or ""),
        kind=KIND_COMMAND,
        status=str(row.get("status") or "pending"),
        title=config.header,
        body="\n".join(lines),
        choices=list(config.choices),
        choice_labels=list(config.labels),
        proposed_at=row.get("created_at"),
        expires_at=row.get("expires_at"),
        shown_at=row.get("shown_at"),
        presented_at=row.get("shown_at"),
        presented_surface=row.get("presented_surface"),
        answered_at=row.get("answered_at") or row.get("resolved_at"),
        origin=row.get("origin"),
        requested_by=row.get("requested_by"),
        card_revision=1,
    )


def card_from_row(row: Any) -> ApprovalCard:
    """Project a durable ``approvals`` row into the frozen card DTO.

    The stored ``card`` JSON is the literal text a human saw; live row columns
    (status, shown_at, answered_at, deny_reason) win because they are updated by
    the resolver after the card was created.
    """
    data = dict(row) if not isinstance(row, Mapping) else dict(row)
    kind = str(data.get("kind") or KIND_COMMAND).strip() or KIND_COMMAND
    if kind == KIND_COMMAND:
        return _command_card(data)
    if kind not in KIND_CONFIG:
        raise ValueError(f"unknown approval kind: {kind!r}")
    stored = _load_json(data.get("card"))
    if not isinstance(stored, dict) or not stored:
        # A schedule row without its literal card is not answerable: refusing
        # is safer than inventing copy the human never saw.
        raise ValueError("schedule approval row has no frozen card body")
    # Live row authority wins over the frozen-at-creation projection.
    updates: dict[str, Any] = {}
    for key in (
        "status", "shown_at", "answered_at", "presented_surface",
        "deny_reason", "expires_at", "digest",
    ):
        if data.get(key) is not None:
            updates[key] = data[key]
    if "digest" in updates:
        updates["proposal_digest"] = updates.pop("digest")
    if data.get("resolved_at") and not updates.get("answered_at"):
        updates["answered_at"] = data["resolved_at"]
    if updates.get("shown_at") and not stored.get("presented_at"):
        updates["presented_at"] = updates["shown_at"]
    merged = {**stored, **updates}
    merged["approval_id"] = str(updates.get("approval_id") or stored.get("approval_id") or data.get("id"))
    merged["kind"] = kind
    merged.setdefault("protocol_version", int(data.get("protocol_version") or APPROVAL_CARD_PROTOCOL_VERSION))
    merged.setdefault("card_revision", int(data.get("card_revision") or 1))
    merged.setdefault("proposal_id", data.get("proposal_id"))
    merged.setdefault("proposal_digest", data.get("digest"))
    merged.setdefault("conversation_id", data.get("conversation_id"))
    merged.setdefault("run_id", data.get("run_id"))
    merged.setdefault("tool_call_id", data.get("tool_call_id"))
    merged.setdefault("origin", data.get("origin"))
    merged.setdefault("requested_by", data.get("requested_by"))
    acknowledged = _load_json(data.get("metadata"))
    if isinstance(acknowledged, dict):
        if acknowledged.get("acknowledged_at"):
            merged["acknowledged_at"] = acknowledged["acknowledged_at"]
        if acknowledged.get("acknowledged_surface"):
            merged["acknowledged_surface"] = acknowledged["acknowledged_surface"]
        if acknowledged.get("resolved_by"):
            merged["resolver"] = acknowledged["resolved_by"]
        if acknowledged.get("choice"):
            merged["decision"] = acknowledged["choice"]
        if acknowledged.get("outcome"):
            merged["outcome"] = acknowledged["outcome"]
        if acknowledged.get("schedule_id"):
            merged["schedule_id"] = acknowledged["schedule_id"]
    return ApprovalCard.model_validate(merged)


async def mark_card_deny_only(approval_id: str, *, reason: str) -> ApprovalCard | None:
    """Restrict one pending schedule card to Deny without losing its full body.

    A transport that cannot present the whole literal text must not offer
    approval choices the human never read. The full body stays in this frozen
    card (and therefore in the dashboard/API); only the offered choices change.
    """
    from .. import db  # local import: db is initialised after this module loads

    async with db.transaction():
        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None or not is_schedule_kind(str(row["kind"] or "")):
            return None
        if str(row["status"] or "") != "pending":
            return card_from_row(row)
        stored = _load_json(row["card"])
        if not isinstance(stored, dict):
            return None
        stored["deny_only"] = True
        stored["choices"] = ["deny"]
        stored["choice_labels"] = ["Deny"]
        stored["deny_reason"] = str(reason or "full card text did not fit the channel")[:300]
        metadata = _load_json(row["metadata"])
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["deny_only"] = True
        await db.execute(
            "UPDATE approvals SET choices=?, card=?, metadata=? WHERE id=? AND status='pending'",
            (
                json.dumps(["deny"], ensure_ascii=False),
                json.dumps(stored, ensure_ascii=False),
                json.dumps(metadata, ensure_ascii=False),
                approval_id,
            ),
        )
    updated = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    return card_from_row(updated) if updated is not None else None


@dataclass
class ScheduleResolution:
    """Outcome of one schedule-card decision."""

    approval_id: str
    kind: str
    status: str
    allowed: bool
    reason: str
    choice: str | None = None
    outcome: str | None = None
    grant_id: str | None = None
    schedule_id: str | None = None
    commit_result: Any = None
    decline_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        # ``commit_result`` stays internal: it belongs to the schedule
        # authority (Phase 5), not the HTTP decision projection.
        return {
            "approval_id": self.approval_id,
            "kind": self.kind,
            "status": self.status,
            "allowed": self.allowed,
            "reason": self.reason,
            "choice": self.choice,
            "outcome": self.outcome,
            "grant_id": self.grant_id,
            "schedule_id": self.schedule_id,
            "decline_message": self.decline_message,
        }


class ApprovalService:
    """Facade over the PermissionManager card APIs.

    It composes the approval row authority with an optional schedule commit
    callback so Phase 5's ``ScheduleService`` can commit a proposal inside the
    same database transaction. It holds no second approval authority of its
    own: every read and write delegates to the PermissionManager.
    """

    def __init__(self, permissions: Any, *, committer: Callable[..., Any] | None = None) -> None:
        self.permissions = permissions
        self._committer = committer

    def set_committer(self, committer: Callable[..., Any] | None) -> None:
        self._committer = committer

    async def create(self, **kwargs: Any) -> ApprovalCard:
        return await self.permissions.create_schedule_card(**kwargs)

    async def present(
        self, approval_id: str, *, surface: str, capabilities: Any = None
    ) -> ApprovalCard:
        return await self.permissions.present_schedule_card(
            approval_id, surface=surface, capabilities=capabilities
        )

    async def acknowledge(self, approval_id: str, *, surface: str) -> ApprovalCard:
        return await self.permissions.acknowledge_schedule_card(
            approval_id, surface=surface
        )

    async def wait(self, approval_id: str, *, timeout: float | None = None) -> ApprovalCard:
        return await self.permissions.wait_for_schedule_card(
            approval_id, timeout=timeout
        )

    async def resolve(
        self,
        approval_id: str,
        *,
        choice: str,
        resolver: str,
        confirm_proposal_digest: str | None,
        deny_reason: str | None = None,
    ) -> ScheduleResolution:
        return await self.permissions.resolve_schedule_card(
            approval_id,
            choice=choice,
            resolver=resolver,
            confirm_proposal_digest=confirm_proposal_digest,
            deny_reason=deny_reason,
            commit=self._committer,
        )

    async def get(self, approval_id: str) -> ApprovalCard | None:
        return await self.permissions.get_schedule_card(approval_id)

    async def list(
        self, *, status: str | None = "pending", include_command: bool = False
    ) -> list[ApprovalCard]:
        return await self.permissions.schedule_cards(
            status=status, include_command=include_command
        )


def project_cards(rows: Sequence[Any]) -> list[ApprovalCard]:
    """Project rows to cards, skipping (never hiding behind) malformed rows."""
    cards: list[ApprovalCard] = []
    for row in rows:
        try:
            cards.append(card_from_row(row))
        except ValueError:
            continue
    return cards


__all__ = [
    "APPROVAL_CARD_PROTOCOL_VERSION",
    "ApprovalCard",
    "ApprovalService",
    "CARD_CAPABILITY_VERSION",
    "CARD_KINDS",
    "CHOICE_LABELS",
    "COMMAND_CHOICES",
    "CardKindConfig",
    "DECLINE_MESSAGES",
    "KIND_COMMAND",
    "KIND_CONFIG",
    "KIND_SCHEDULE_CREATE",
    "KIND_SCHEDULE_EDIT",
    "SCHEDULE_CREATE_CHOICES",
    "SCHEDULE_EDIT_CHOICES",
    "SCHEDULE_KINDS",
    "ScheduleBodySpec",
    "ScheduleCardView",
    "ScheduleChoice",
    "ScheduleResolution",
    "at_or_after",
    "MAX_CARD_PAGES",
    "card_fits_surface",
    "card_from_row",
    "card_pages",
    "current_card_presenter",
    "paginate_text",
    "presenting_cards_via",
    "schedule_card_text",
    "schedule_notification_copy",
    "choice_labels_for",
    "choices_for_kind",
    "deadline_text",
    "delivery_description",
    "format_first_run",
    "format_window",
    "is_schedule_kind",
    "kind_config",
    "mark_card_deny_only",
    "normalize_choice",
    "parse_instant",
    "project_cards",
    "proposal_digest_matches",
    "render_edit_body",
    "render_schedule_body",
    "schedule_terminal_text",
    "timeout_notice",
]
