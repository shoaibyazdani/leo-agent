"""Per-call risk: LOW runs, MEDIUM runs and is audited, HIGH needs an approval.

A static per-tool tier is too coarse to be useful. It gave the same answer to
``grep -rn`` and ``rm -rf``, and it charged a HIGH-risk approval for every edit of
a file the agent had just written — 24 approved and 5 expired ``tool:patch``
approvals on the reference box, against a handful of genuinely dangerous
commands.

This module decides per call and returns the reason, so the audit row explains
every tier instead of showing a bare ``high``:

* read-only inspection is LOW                     (nothing to approve)
* a bounded, reversible write is MEDIUM           (runs, recorded, no tap)
* destructive, outbound-to-a-stranger, credential
  or self-modification work stays HIGH            (approval)
* the hardline floor is still a DENIAL, not a tier  (see ``shell_policy.preflight``)

An operator who wants the old strict floor flips ``security.allow_medium_risk``
off; MEDIUM then fails closed instead of running.
"""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any, Mapping

from ..config import Settings
from ..contracts import RiskLevel

# ``leo.tools`` imports the executor, which imports this package: the shell
# policy is imported at call time so neither side needs the other at import.
# Tools that inspect state. They are registered LOW already; named here so a new
# registration that ships a wrong tier does not silently become an approval.
READ_TOOLS: frozenset[str] = frozenset(
    {
        # Only names whose registered tier is above LOW. Everything else that
        # inspects state is already registered LOW, so naming it here would be a
        # second, drifting copy of the roster.
        "metrics_snapshot", "webhook_verify",
    }
)

# Bounded, audited writes: reversible in practice, no human judgement needed.
# They run under MEDIUM (recorded, no approval) unless a path rule below escalates.
BOUNDED_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        # memory_forget is NOT here: a deleted memory has no undelete on the tool
        # surface, so it stays behind the approval its spec and docstring promise.
        "file_copy", "file_move", "memory_add", "task_create", "schedule_create",
        "attachment_add", "attachment_remove",
    }
)

# Files whose content is a credential, a policy or the agent's own runtime
# configuration: changing one is a deliberate, reviewable act.
SENSITIVE_NAMES: frozenset[str] = frozenset(
    {
        ".env", ".netrc", "id_rsa", "id_ed25519", "id_ecdsa", "leo.yaml",
        "openclaw.json", "credentials", "credentials.json", "secrets.json",
        ".compartment_pass", "hosts", "sudoers",
    }
)
SENSITIVE_SUFFIXES: tuple[str, ...] = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".pass")
SENSITIVE_DIRS: frozenset[str] = frozenset({".git", ".ssh", ".gnupg", ".aws", ".config", ".kube", ".docker"})

# Adding a model or changing the agent's own behaviour is self-modification of
# the product, not work on the user's files.
SELF_MODIFY_TOOLS: frozenset[str] = frozenset(
    {"model_add", "apply_improvement"}
)


def assess_call(
    tool: str,
    arguments: Mapping[str, Any],
    default: RiskLevel,
    settings: Settings,
) -> tuple[RiskLevel, str]:
    """Return ``(risk, reason)`` for one tool call."""
    name = str(tool or "")

    if name == "shell":
        from ..tools import shell_policy

        return shell_policy.assess_command(str(arguments.get("command") or ""), settings)

    if name in SELF_MODIFY_TOOLS:
        return RiskLevel.HIGH, "changes the agent's own model or skills"

    if name in ("write_file", "patch", "file_copy", "file_move"):
        return _assess_path(arguments, name)

    if name in ("channel_send", "gateway.send", "send_message"):
        return _assess_outbound(arguments)

    if name == "memory_forget":
        # The one memory operation with no way back through the tool surface.
        return RiskLevel.HIGH, "deletes a stored memory row; there is no undelete"

    if name in READ_TOOLS:
        return RiskLevel.LOW, "read-only inspection"

    if name in BOUNDED_WRITE_TOOLS:
        return RiskLevel.MEDIUM, "bounded, audited write"

    return default, f"registered tier for {name or 'unknown tool'}"


def _assess_path(arguments: Mapping[str, Any], tool: str) -> tuple[RiskLevel, str]:
    raw = arguments.get("path") or arguments.get("destination") or arguments.get("source")
    text = str(raw or "").strip()
    if not text:
        return RiskLevel.MEDIUM, f"{tool} without a path (bounded by the workspace roots)"
    parts = PurePath(text).parts
    lower_parts = {part.lower() for part in parts}
    name = parts[-1].lower() if parts else ""
    if lower_parts & SENSITIVE_DIRS:
        return RiskLevel.HIGH, f"{tool} touches a credential/config directory ({sorted(lower_parts & SENSITIVE_DIRS)[0]})"
    if name in SENSITIVE_NAMES or name.startswith(".env"):
        return RiskLevel.HIGH, f"{tool} writes a credential or runtime config file ({name})"
    if any(name.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
        return RiskLevel.HIGH, f"{tool} writes a key/keystore file ({name})"
    return RiskLevel.MEDIUM, f"{tool} inside the workspace (bounded, audited)"


def _assess_outbound(arguments: Mapping[str, Any]) -> tuple[RiskLevel, str]:
    """A reply to the conversation the turn came from needs no approval."""
    target = str(arguments.get("target") or arguments.get("chat_id") or "").strip()
    subject = str(arguments.get("subject") or arguments.get("conversation_id") or "").strip()
    if target and subject and target == subject:
        return RiskLevel.LOW, "reply into the originating conversation"
    if not target:
        return RiskLevel.MEDIUM, "outbound message without an explicit target"
    return RiskLevel.HIGH, f"outbound message to a different target ({target[:24]})"
