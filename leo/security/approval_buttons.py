"""Inline-keyboard approvals for chat channels (Telegram first).

Ported from OpenClaw's proven design (``approval-handler.runtime``,
``native-command-callback-data``, ``approval-resolution-ref``). Three
properties are deliberately copied because they are what makes the original
safe, and each one closes a real hole:

1. **Versioned, reserved callback namespace.** ``tga1:<kind>:<decision>:<id>``.
   Anything not matching this exact shape is rejected before it can reach an
   approval. A malformed or foreign callback can never be interpreted as
   "approve".

2. **Checksummed locator instead of a trust boundary.** Telegram caps
   ``callback_data`` at 64 bytes. When the real approval id does not fit,
   OpenClaw swaps it for a SHA-256 base64url *locator* (43 chars). That is a
   lookup key, not a capability: holding it grants nothing on its own. The
   actual authority check is the sender allow-list (see ``is_authorized``),
   applied on every tap.

3. **Sender identity, not chat identity.** Authorization is decided from the
   immutable ``sender_id``, never the mutable username or the chat id, so a
   group member cannot resolve an operator's approval.

The decision is always carried out through the operator (privileged) path,
because ``Permissions.resolve_approval`` intentionally refuses ``chat:`` actors
to prevent a chat from self-approving.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Iterable

_LOG = logging.getLogger(__name__)

# Bump when the payload layout changes; old callbacks then fail closed.
CALLBACK_PREFIX = "tga1:"
TELEGRAM_CALLBACK_DATA_MAX_BYTES = 64
RESOLUTION_REF_LENGTH = 43

APPROVAL_KIND_CODES: dict[str, str] = {"tool": "t", "learning": "l", "gateway": "g"}
DECISION_CODES: dict[str, str] = {
    "approve": "a", "reject": "r", "session": "s", "trust": "w",
}
_KIND_BY_CODE = {code: kind for kind, code in APPROVAL_KIND_CODES.items()}
_DECISION_BY_CODE = {code: decision for decision, code in DECISION_CODES.items()}

KIND_FOR_ACTION: dict[str, str] = {
    "tool:shell": "tool",
    "tool:file_write": "tool",
    "gateway.send": "gateway",
}
KIND_FOR_PREFIX: tuple[tuple[str, str], ...] = (
    ("tool:", "tool"),
    ("learning:", "learning"),
    ("gateway.", "gateway"),
)


def approval_kind_for_action(action: str) -> str:
    """Map a real approval action to a short, stable kind.

    Unknown actions default to ``tool`` rather than raising: the kind is only a
    namespace discriminator, never an authority grant.
    """
    text = str(action or "").strip()
    if text in KIND_FOR_ACTION:
        return KIND_FOR_ACTION[text]
    for prefix, kind in KIND_FOR_PREFIX:
        if text.startswith(prefix):
            return kind
    return "tool"


def fits_callback_data(value: str) -> bool:
    """Telegram rejects the whole send if ``callback_data`` exceeds 64 bytes."""
    return len(str(value).encode("utf-8")) <= TELEGRAM_CALLBACK_DATA_MAX_BYTES


def build_resolution_ref(approval_kind: str, approval_id: str) -> str:
    """Stable 43-char locator for an approval, used when the id cannot fit."""
    digest = hashlib.sha256(f"{approval_kind}\0{approval_id}".encode("utf-8")).digest()
    import base64

    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def is_resolution_ref(value: str) -> bool:
    text = str(value or "")
    if len(text) != RESOLUTION_REF_LENGTH:
        return False
    return all(ch.isalnum() or ch in "-_" for ch in text)


def build_callback_data(approval_kind: str, decision: str, approval_id: str) -> str:
    """Encode one button action into Telegram-private, versioned callback data.

    Falls back to the checksummed locator only when the exact id would overflow
    Telegram's 64-byte ceiling.
    """
    kind_code = APPROVAL_KIND_CODES.get(approval_kind)
    decision_code = DECISION_CODES.get(decision)
    if kind_code is None:
        raise ValueError(f"unknown approval kind: {approval_kind!r}")
    if decision_code is None:
        raise ValueError(f"unknown decision: {decision!r}")
    exact = f"{CALLBACK_PREFIX}{kind_code}:{decision_code}:{approval_id}"
    if fits_callback_data(exact):
        return exact
    ref = build_resolution_ref(approval_kind, approval_id)
    shortened = f"{CALLBACK_PREFIX}{kind_code}:{decision_code}:{ref}"
    if not fits_callback_data(shortened):
        raise ValueError("approval callback data does not fit Telegram's 64-byte limit")
    return shortened


def has_callback_prefix(data: Any) -> bool:
    return isinstance(data, str) and data.startswith(CALLBACK_PREFIX)


def parse_callback_data(data: Any) -> dict[str, str] | None:
    """Decode only payloads emitted by :func:`build_callback_data`.

    Returns ``None`` for anything else — including well-prefixed garbage — so a
    forged tap can never be coerced into a decision.
    """
    if not has_callback_prefix(data) or not fits_callback_data(data):
        return None
    encoded = str(data)[len(CALLBACK_PREFIX):]
    # "<kind>:<decision>:<id>"
    if len(encoded) < 5 or encoded[1] != ":" or encoded[3] != ":":
        return None
    kind = _KIND_BY_CODE.get(encoded[0])
    decision = _DECISION_BY_CODE.get(encoded[2])
    approval_id = encoded[4:]
    if not kind or not decision or not approval_id:
        return None
    return {
        "type": "approval",
        "approval_kind": kind,
        "decision": decision,
        "approval_id": approval_id,
        "is_ref": "1" if is_resolution_ref(approval_id) else "0",
    }


def is_authorized(sender_id: Any, allowed: Iterable[str] | str | None) -> bool:
    """Authorize a tap by immutable sender id against the operator allow-list.

    This is the real gate. The callback payload proves nothing; this does.
    """
    candidate = str(sender_id or "").strip()
    if not candidate:
        return False
    if allowed is None:
        return False
    targets = [allowed] if isinstance(allowed, str) else list(allowed)
    return any(candidate == str(item).strip() for item in targets if str(item or "").strip())


def sign_resolution(secret: str, approval_id: str, decision: str) -> str:
    """Short HMAC tag for audit/replay diagnostics (never the authority check)."""
    key = str(secret or "").encode("utf-8")
    msg = f"{approval_id}\0{decision}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()[:16]


SESSION_GRANT_SECONDS = 8 * 60 * 60

# Actions whose authority lives in their arguments, not in a path resource.
# "Always" for these is scoped to the exact call payload through the approval's
# arguments digest; a cwd or the bare tool name would cover unrelated commands.
ALWAYS_ARG_SCOPED_ACTIONS: frozenset[str] = frozenset({"tool:shell", "tool:terminal"})


def session_grant_scopable(
    action: str, resource: Any, arguments_digest: Any = None
) -> bool:
    """Whether an "Always" grant can name the operation shown.

    Path tools are scoped by their resource. Shell/terminal carry their
    authority in the arguments, so they need the approval's arguments digest;
    without one the grant would cover every call of the tool, so the button is
    omitted and a tap that asks for it is refused.
    """
    if str(action or "").strip() in ALWAYS_ARG_SCOPED_ACTIONS:
        return bool(str(arguments_digest or "").strip())
    return resource is not None and str(resource).strip() != ""


def build_keyboard(
    approval_id: str,
    action: str,
    *,
    resource: Any = None,
    arguments_digest: Any = None,
    trust_days: float | None = None,
    allow_workspace_trust: bool = True,
    include_session: bool = True,
) -> dict[str, Any]:
    """Approve / this-command / trust-workspace / Reject for one approval.

    ``This command`` covers the exact shell command (or the exact resource for
    path tools). ``Trust workspace`` covers ordinary shell/terminal work in the
    checkout for its lifetime; destructive, credential, privilege,
    arbitrary-code and self-modifying commands still ask. Both are scoped,
    expiring grants, never a blanket allow, and are only offered where the
    approval names that scope.
    """
    kind = approval_kind_for_action(action)
    scoped = session_grant_scopable(action, resource, arguments_digest)
    first_row = [
        {"text": "✅ Approve", "callback_data": build_callback_data(kind, "approve", approval_id)},
        {"text": "⛔ Reject", "callback_data": build_callback_data(kind, "reject", approval_id)},
    ]
    rows = [first_row]
    second: list[dict[str, str]] = []
    if include_session and scoped:
        label = (
            f"🔁 This command ({SESSION_GRANT_SECONDS // 3600}h)"
            if str(action or "") in ALWAYS_ARG_SCOPED_ACTIONS
            else f"🔁 Always ({SESSION_GRANT_SECONDS // 3600}h)"
        )
        second.append({
            "text": label,
            "callback_data": build_callback_data(kind, "session", approval_id),
        })
    if (
        include_session
        and allow_workspace_trust
        and str(action or "") in ALWAYS_ARG_SCOPED_ACTIONS
    ):
        days = max(1, int(round(float(trust_days if trust_days is not None else 30))))
        second.append({
            "text": f"🏠 Trust workspace ({days}d)",
            "callback_data": build_callback_data(kind, "trust", approval_id),
        })
    if second:
        rows.append(second)
    return {"inline_keyboard": rows}


def resolve_locator(approval_kind: str, locator: str, candidates: Iterable[str]) -> str | None:
    """Turn a 43-char locator back into the approval id it stands for.

    The locator is a hash, so the only way back is recomputing it for candidate
    ids. Callers pass the pending ids they own; a locator that matches none is
    rejected, never guessed.
    """
    for candidate in candidates or ():
        if build_resolution_ref(approval_kind, str(candidate)) == str(locator):
            return str(candidate)
    return None
