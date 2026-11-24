"""Authoritative slash-command registry, parser and dispatcher for LEO.

Design rules (mirrored from the LEO contracts module):

* One registry is the source of truth for names, aliases, argument hints,
  help text, risk, permissions and handler callables. CLI/API/UI adapters only
  supply context and render the returned :class:`CommandResult`.
* A command is recognised only when the *first* non-whitespace character of a
  message is ``/``. Prose that contains ``/status`` is never executed.
* Command names and aliases are allowlisted. Text is never interpolated into
  shell, SQL or templates; quoted tokens are parsed with :mod:`shlex`, which
  performs lexical quoting only and does no shell expansion.
* Authorization is centralised here: surface/operator/admin policy is checked
  before :meth:`PermissionManager.authorize` runs, and the real permission
  decision (including approval creation) gates the handler. Handlers that call
  an audited LEO service (skills, gateway, operator executor) may still make a
  second, more specific decision; that is intentional and reported truthfully.
* Unavailable commands stay in the registry with an explicit reason so
  ``/help`` never advertises a capability LEO does not have.
"""

from __future__ import annotations

import difflib
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from ..contracts import RiskLevel
from ..security import ApprovalRequired

ALL_SURFACES: frozenset[str] = frozenset({"chat", "cli", "api", "gateway", "desktop"})
CHAT_SURFACES: frozenset[str] = frozenset({"chat", "cli", "desktop"})

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_APPROVAL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class CommandStatus(str, Enum):
    OK = "ok"
    DENIED = "denied"
    PENDING = "pending"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class CommandResult(BaseModel):
    """Transport-neutral outcome of one slash command invocation."""

    command: str
    status: CommandStatus
    success: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None
    code: str = ""
    surface: str = ""
    conversation_id: str | None = None
    principal: str = ""
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def ok(cls, command: str, message: str, data: dict[str, Any] | None = None, **kwargs: Any) -> "CommandResult":
        return cls(command=command, status=CommandStatus.OK, success=True, message=message,
                   data=data or {}, **kwargs)

    @classmethod
    def error(cls, command: str, message: str, *, code: str = "error", **kwargs: Any) -> "CommandResult":
        return cls(command=command, status=CommandStatus.ERROR, success=False, message=message,
                   code=code, **kwargs)

    @classmethod
    def denied(cls, command: str, message: str, **kwargs: Any) -> "CommandResult":
        return cls(command=command, status=CommandStatus.DENIED, success=False, message=message,
                   code="permission_denied", **kwargs)

    @classmethod
    def pending(cls, command: str, message: str, approval_id: str | None, **kwargs: Any) -> "CommandResult":
        return cls(command=command, status=CommandStatus.PENDING, success=False, message=message,
                   approval_id=approval_id, code="approval_required", **kwargs)

    @classmethod
    def unsupported(cls, command: str, reason: str, **kwargs: Any) -> "CommandResult":
        return cls(command=command, status=CommandStatus.UNSUPPORTED, success=False, message=reason,
                   code="unsupported", **kwargs)


@dataclass(frozen=True)
class ArgumentSpec:
    name: str
    description: str = ""
    required: bool = False
    variadic: bool = False
    # Fixed set of accepted tokens, exposed to clients for Tab completion.
    # Empty means free-form (no suggestions offered).
    choices: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "required": self.required, "variadic": self.variadic,
            "choices": list(self.choices),
        }


ADMIN_PRINCIPALS_DEFAULT: frozenset[str] = frozenset(
    {"cli", "local", "operator:cli", "api-token"}
)
OPERATOR_PRINCIPALS_DEFAULT: frozenset[str] = frozenset(
    {"cli", "local", "operator:cli", "api-token"}
)
OPERATOR_SURFACES_DEFAULT: frozenset[str] = frozenset({"cli"})


@dataclass(frozen=True)
class CommandSpec:
    """One command definition. ``handler`` is always async and never the model loop."""

    name: str
    help: str
    args: tuple[ArgumentSpec, ...] = ()
    args_hint: str = ""
    aliases: tuple[str, ...] = ()
    category: str = "general"
    risk: RiskLevel = RiskLevel.LOW
    action: str = ""
    handler: Callable[["CommandContext"], Awaitable[CommandResult]] | None = None
    surfaces: frozenset[str] = ALL_SURFACES
    admin_only: bool = False
    operator_only: bool = False
    # Minimum number of non-option tokens. Checked before authorization so a
    # malformed invocation cannot create a durable approval for a bad call.
    min_args: int = 0
    # Index of the argv token used as the permission resource, when applicable.
    resource_arg: int | None = None
    available: bool = True
    unavailable_reason: str = ""
    # "any" | "idle" | "interrupt": advisory for chat/UI busy-policy wiring.
    busy_policy: str = "any"
    # Ephemeral commands are not written into durable chat history by adapters.
    ephemeral: bool = False

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid command name: {self.name!r}")
        for alias in self.aliases:
            if not _NAME_RE.match(alias):
                raise ValueError(f"invalid alias for /{self.name}: {alias!r}")
        object.__setattr__(self, "action", self.action or f"slash.{self.name}")
        object.__setattr__(self, "risk", RiskLevel(self.risk))
        if not self.available and not self.unavailable_reason:
            raise ValueError(f"/{self.name} is unavailable but has no reason")
        if self.operator_only and not self.available:
            raise ValueError(f"/{self.name}: operator policy requires a real handler")

    def usage(self) -> str:
        return f"/{self.name}" + (f" {self.args_hint}" if self.args_hint else "")

    def validate_argv(self, argv: Sequence[str]) -> None:
        if self.min_args <= 0:
            return
        positional: list[str] = []
        index = 0
        while index < len(argv):
            token = argv[index]
            if token.startswith("-") and token != "-":
                if "=" not in token and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                    index += 2  # --option value
                    continue
                index += 1
                continue
            positional.append(token)
            index += 1
        if len(positional) < self.min_args:
            raise ValueError(f"usage: {self.usage()}")

    def as_dict(
        self,
        *,
        principal: str | None = None,
        surface: str | None = None,
        admin_principals: frozenset[str] = ADMIN_PRINCIPALS_DEFAULT,
        operator_principals: frozenset[str] = OPERATOR_PRINCIPALS_DEFAULT,
        operator_surfaces: frozenset[str] = OPERATOR_SURFACES_DEFAULT,
    ) -> dict[str, Any]:
        allowed: bool | None = None
        if principal is not None and surface is not None:
            allowed = (
                self.available
                and surface in self.surfaces
                and (not self.operator_only or (
                    principal in operator_principals and surface in operator_surfaces
                ))
                and (not self.admin_only or principal in admin_principals)
            )
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "args_hint": self.args_hint,
            "arguments": [item.as_dict() for item in self.args],
            "help": self.help,
            "category": self.category,
            "risk": self.risk.value,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "admin_only": self.admin_only,
            "operator_only": self.operator_only,
            "surfaces": sorted(self.surfaces),
            "busy_policy": self.busy_policy,
            "ephemeral": self.ephemeral,
            "allowed_for_caller": allowed,
        }


@dataclass
class ParsedCommand:
    text: str
    name: str | None = None
    argv: list[str] = field(default_factory=list)
    raw_args: str = ""
    approval_id: str | None = None
    error: str | None = None

    @property
    def is_command(self) -> bool:
        return self.text.lstrip().startswith("/")


@dataclass
class CommandContext:
    """Trusted invocation context. Handlers never read request globals."""

    app: Any
    principal: str
    surface: str
    channel: str
    argv: list[str]
    raw: str
    registry: "CommandRegistry"
    conversation_id: str | None = None
    run_id: str | None = None
    approval_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def actor(self) -> str:
        """Permission actor. Chat actors can never resolve their own approvals."""
        if self.surface == "chat":
            return f"chat:{self.channel or self.principal or 'unknown'}"
        if self.surface == "gateway":
            return f"gateway:{self.channel or self.principal or 'unknown'}"
        return f"{self.surface}:{self.principal or 'unknown'}"

    def require(self, index: int, label: str) -> str:
        try:
            value = self.argv[index]
        except IndexError as exc:
            raise ValueError(f"missing {label}") from exc
        if not str(value).strip():
            raise ValueError(f"missing {label}")
        return str(value)

    def option(self, *names: str, default: str | None = None) -> str | None:
        """Read ``--name value`` or ``--name=value`` from argv."""
        for index, token in enumerate(self.argv):
            for name in names:
                if token == name:
                    if index + 1 < len(self.argv):
                        return str(self.argv[index + 1])
                    raise ValueError(f"{name} needs a value")
                if token.startswith(name + "="):
                    return token.split("=", 1)[1]
        return default

    def has_flag(self, *names: str) -> bool:
        return any(token in names for token in self.argv)

    def rest(self, start: int = 0) -> str:
        return " ".join(str(part) for part in self.argv[start:])


def _extract_approval(argv: list[str]) -> tuple[list[str], str | None, str | None]:
    """Strip the reserved ``--approval`` flag from argv."""
    cleaned: list[str] = []
    approval_id: str | None = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--approval":
            if index + 1 >= len(argv):
                return cleaned, None, "--approval needs an approval id"
            candidate = str(argv[index + 1])
            if not _APPROVAL_ID_RE.match(candidate):
                return cleaned, None, "invalid approval id"
            approval_id = candidate
            index += 2
            continue
        if token.startswith("--approval="):
            candidate = token.split("=", 1)[1]
            if not _APPROVAL_ID_RE.match(candidate):
                return cleaned, None, "invalid approval id"
            approval_id = candidate
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned, approval_id, None


def parse(text: str) -> ParsedCommand | None:
    """Parse leading-slash text. Return ``None`` for ordinary prose.

    Quoting is lexical only (``shlex.split``): ``"/memory add a; rm -rf /"``
    becomes one argument and never reaches a shell.
    """
    if not isinstance(text, str):
        return None
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None
    raw_args = stripped[1:].strip()
    parsed = ParsedCommand(text=text, raw_args=raw_args)
    if not raw_args:
        parsed.error = "empty command; try /help"
        return parsed
    try:
        tokens = shlex.split(raw_args, posix=True, comments=False)
    except ValueError as exc:
        parsed.error = f"invalid command syntax: {exc}"
        return parsed
    if not tokens:
        parsed.error = "empty command; try /help"
        return parsed
    name = tokens[0].lower().lstrip("/")
    if not _NAME_RE.match(name):
        parsed.error = f"invalid command name: {tokens[0]!r}"
        return parsed
    argv, approval_id, error = _extract_approval(tokens[1:])
    parsed.name = name
    parsed.argv = argv
    parsed.approval_id = approval_id
    parsed.error = error
    return parsed


class CommandRegistry:
    """Mutable registry plus the single dispatch path used by every surface."""

    def __init__(
        self,
        *,
        admin_principals: Iterable[str] = ADMIN_PRINCIPALS_DEFAULT,
        operator_principals: Iterable[str] = OPERATOR_PRINCIPALS_DEFAULT,
        operator_surfaces: Iterable[str] = OPERATOR_SURFACES_DEFAULT,
    ) -> None:
        self._specs: dict[str, CommandSpec] = {}
        self._aliases: dict[str, str] = {}
        self.admin_principals = frozenset(str(p) for p in admin_principals)
        self.operator_principals = frozenset(str(p) for p in operator_principals)
        self.operator_surfaces = frozenset(str(s) for s in operator_surfaces)

    # ---- registration and lookup -----------------------------------------
    def register(self, spec: CommandSpec) -> CommandSpec:
        if spec.name in self._specs or spec.name in self._aliases:
            raise ValueError(f"duplicate command name: /{spec.name}")
        for alias in spec.aliases:
            if alias in self._specs or alias in self._aliases:
                raise ValueError(f"duplicate command alias: /{alias}")
            self._aliases[alias] = spec.name
        self._specs[spec.name] = spec
        return spec

    def all(self) -> list[CommandSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def get(self, name: str) -> CommandSpec | None:
        key = str(name).lower().lstrip("/!")
        return self._specs.get(key)

    def resolve(self, name: str) -> tuple[CommandSpec | None, list[str]]:
        """Resolve an exact name/alias, then an unambiguous prefix.

        Prefix behaviour follows the familiar ``/mod`` -> ``/model`` style;
        ambiguity is an error rather than a silent guess. Case is ignored.
        """
        key = str(name).lower().lstrip("/!")
        if key in self._specs:
            return self._specs[key], []
        if key in self._aliases:
            return self._specs[self._aliases[key]], []
        matches = [
            spec for spec in self.all()
            if spec.name.startswith(key) or any(alias.startswith(key) for alias in spec.aliases)
        ]
        if len(matches) == 1:
            return matches[0], []
        return None, [f"/{spec.name}" for spec in matches[:12]]

    # ---- help / catalog ---------------------------------------------------
    def _policy_kwargs(self) -> dict[str, frozenset[str]]:
        return {
            "admin_principals": self.admin_principals,
            "operator_principals": self.operator_principals,
            "operator_surfaces": self.operator_surfaces,
        }

    def catalog(self, *, principal: str | None = None, surface: str | None = None) -> list[dict[str, Any]]:
        return [
            spec.as_dict(principal=principal, surface=surface, **self._policy_kwargs())
            for spec in self.all()
        ]

    def help(
        self,
        filter_text: str | None = None,
        *,
        principal: str | None = None,
        surface: str | None = None,
        page: int = 1,
        per_page: int = 200,
    ) -> dict[str, Any]:
        needle = str(filter_text or "").strip().lower()
        matches = self.all()
        if needle:
            if needle in ("skills", "skill"):
                needle = ""
                matches = [s for s in matches if s.category in ("skills", "skill")]
            else:
                matches = [
                    s for s in matches
                    if needle in s.name or needle in s.help.lower() or needle in s.category
                    or any(needle in alias for alias in s.aliases)
                ]
        try:
            page = max(1, int(page))
        except (TypeError, ValueError):
            page = 1
        per_page = max(1, min(int(per_page), 200))
        total = len(matches)
        shown = matches[(page - 1) * per_page: page * per_page]
        entries = [s.as_dict(principal=principal, surface=surface, **self._policy_kwargs()) for s in shown]
        by_category: dict[str, list[str]] = {}
        for entry in entries:
            line = f"  /{entry['name']}" + (f" {entry['args_hint']}" if entry["args_hint"] else "")
            line += " — " + entry["help"]
            if entry["aliases"]:
                line += "  (aliases: " + ", ".join("/" + a for a in entry["aliases"]) + ")"
            if not entry["available"]:
                line += f"  [UNAVAILABLE: {entry['unavailable_reason']}]"
            elif entry["allowed_for_caller"] is False:
                line += "  [not permitted for this principal/surface]"
            by_category.setdefault(entry["category"], []).append(line)
        lines = [
            f"LEO slash commands — page {page}, {len(shown)} of {total} shown "
            f"({sum(1 for s in self.all() if s.available)} implemented, "
            f"{sum(1 for s in self.all() if not s.available)} explicitly unavailable)"
        ]
        for category in sorted(by_category):
            lines.append(f"\n[{category}]")
            lines.extend(by_category[category])
        if total > page * per_page:
            lines.append(f"\nMore commands: /help {page + 1}")
        lines.append("\nUse /help <filter> to narrow the list; /commands [page] for paging.")
        lines.append(
            "Dynamic skills can also be invoked as /<skill-name> (resolved through the "
            "skill registry); commands marked UNAVAILABLE are reported explicitly and "
            "never forwarded to the model."
        )
        return {
            "commands": entries,
            "total": total,
            "page": page,
            "per_page": per_page,
            "has_more": total > page * per_page,
            "text": "\n".join(lines),
        }

    # ---- dispatch ---------------------------------------------------------
    async def dispatch(
        self,
        text: str,
        *,
        app: Any,
        principal: str = "local",
        surface: str = "chat",
        channel: str = "cli",
        conversation_id: str | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CommandResult:
        parsed = parse(text)
        name = parsed.name if parsed and parsed.name else (parsed.raw_args[:32] if parsed else "")
        meta: dict[str, Any] = dict(metadata or {})
        context = CommandContext(
            app=app, principal=str(principal), surface=str(surface), channel=str(channel),
            argv=list(parsed.argv) if parsed else [], raw=text, registry=self,
            conversation_id=conversation_id, run_id=run_id,
            approval_id=parsed.approval_id if parsed else None, metadata=meta,
        )

        def finish(result: CommandResult) -> CommandResult:
            result.surface = context.surface
            result.conversation_id = conversation_id
            result.principal = context.principal
            return result

        if parsed is None:
            return finish(CommandResult.error("?", "not a slash command", code="not_command"))
        if parsed.error:
            return finish(CommandResult.error(name or "?", parsed.error, code="parse_error"))
        spec, candidates = self.resolve(parsed.name or "")
        # Installed skills are a dynamic slash family. An unknown name that is a
        # real, enabled skill is routed through the /skill spec so the same
        # permission, approval and audit path applies; nothing else falls back.
        if spec is None and not candidates:
            dynamic = await self._dynamic_skill(context, parsed.name or "", parsed.argv)
            if dynamic is not None:
                spec, dynamic_argv = dynamic
                context.argv = dynamic_argv
        if spec is None:
            if candidates:
                message = f"Ambiguous command '/{parsed.name}'. Did you mean: {', '.join(candidates)}?"
                code = "ambiguous_command"
            else:
                close = difflib.get_close_matches(parsed.name or "", [s.name for s in self.all()], n=3, cutoff=0.55)
                message = f"Unknown command '/{parsed.name}'. Type /help for available commands."
                if close:
                    message += " Did you mean: " + ", ".join("/" + item for item in close) + "?"
                code = "unknown_command"
            return finish(CommandResult.error(parsed.name or "?", message, code=code))
        try:
            spec.validate_argv(context.argv)
        except ValueError as exc:
            return finish(CommandResult.error(spec.name, str(exc), code="usage"))

        # Policy denials are audited durably but never turned into an approval
        # request: a forbidden command must not become approvable.
        if surface not in spec.surfaces:
            return finish(await self._policy_denial(
                context, spec, f"/{spec.name} is not available on the {surface} surface"
            ))
        if spec.operator_only and (
            context.principal not in self.operator_principals or surface not in self.operator_surfaces
        ):
            return finish(await self._policy_denial(
                context, spec,
                "/approve, /deny and approval review require the trusted operator CLI principal",
            ))
        if spec.admin_only and context.principal not in self.admin_principals:
            return finish(await self._policy_denial(
                context, spec, f"/{spec.name} requires an administrator principal"
            ))
        if not spec.available:
            return finish(CommandResult.unsupported(spec.name, spec.unavailable_reason))

        initialize = getattr(app, "initialize", None)
        if callable(initialize):
            await initialize()

        resource = f"slash:{spec.name}"
        if spec.resource_arg is not None and len(context.argv) > spec.resource_arg:
            target = str(context.argv[spec.resource_arg]).strip()
            if target and not target.startswith("-"):
                resource = f"slash:{spec.name}:{target}"
        arguments = {
            "argv": list(context.argv), "surface": context.surface,
            "conversation_id": conversation_id, "principal": context.principal,
        }
        permissions = getattr(app, "permissions", None)
        if permissions is not None and hasattr(permissions, "authorize"):
            try:
                decision = await permissions.authorize(
                    context.actor, spec.action, resource, spec.risk,
                    approval_id=context.approval_id, arguments=arguments,
                    command=spec.name, surface=context.surface, channel=context.channel,
                )
            except PermissionError as exc:
                return finish(CommandResult.denied(spec.name, str(exc)))
            if getattr(decision, "requires_approval", False):
                return finish(CommandResult.pending(
                    spec.name,
                    f"Approval required for /{spec.name} (approval_id={decision.approval_id}). "
                    "An operator can run /approve <id> in the CLI, then retry this command with "
                    f"--approval {decision.approval_id}.",
                    decision.approval_id,
                ))
            if not getattr(decision, "allowed", False):
                return finish(CommandResult.denied(
                    spec.name, f"Permission denied for /{spec.name}: {decision.reason}",
                    approval_id=decision.approval_id,
                ))

        if spec.handler is None:
            return finish(CommandResult.unsupported(spec.name, "no handler is wired for this command"))
        try:
            result = await spec.handler(context)
        except ApprovalRequired as exc:  # service-level approval (skills, gateway)
            approval_id = str(getattr(exc, "approval_id", "") or "")
            return finish(CommandResult.pending(
                spec.name, f"Approval required: {exc}", approval_id or None,
            ))
        except (KeyError, LookupError, ValueError, PermissionError) as exc:
            return finish(CommandResult.error(spec.name, self._redact(app, str(exc)), code="invalid_request"))
        except Exception as exc:  # noqa: BLE001 - never leak a traceback into chat
            return finish(CommandResult.error(
                spec.name, f"{type(exc).__name__}: {self._redact(app, str(exc))}",
                code=type(exc).__name__,
            ))
        if not isinstance(result, CommandResult):  # handlers must be explicit
            return finish(CommandResult.error(
                spec.name, "handler returned a non-CommandResult", code="handler_contract",
            ))
        result.command = spec.name
        try:
            emit = getattr(getattr(app, "events", None), "emit", None)
            if callable(emit):
                await emit(
                    "slash.command", command=spec.name, status=result.status.value,
                    surface=context.surface, principal=context.principal,
                    conversation_id=conversation_id,
                )
        except Exception:  # noqa: BLE001 - observability must not break a command
            result.warnings.append("command event could not be persisted")
        return finish(result)

    async def _dynamic_skill(
        self, context: CommandContext, name: str, argv: list[str]
    ) -> tuple[CommandSpec, list[str]] | None:
        """Resolve ``/<installed-skill>`` without widening the static allowlist."""
        if not name or name.startswith("-"):
            return None
        spec = self._specs.get("skill")
        skills = getattr(context.app, "skills", None)
        getter = getattr(skills, "get", None)
        if spec is None or not callable(getter):
            return None
        try:
            record = await getter(name)
        except Exception:  # noqa: BLE001 - a broken registry is not a command
            return None
        if record is None or not getattr(record, "enabled", True):
            return None
        return spec, [name, *argv]

    async def _policy_denial(self, context: CommandContext, spec: CommandSpec, reason: str) -> CommandResult:
        try:
            from .. import db
            from ..contracts import utcnow

            await db.execute(
                "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (utcnow(), context.actor, spec.action, f"slash:{spec.name}",
                 spec.risk.value, "denied", reason, "{}"),
            )
        except Exception:  # noqa: BLE001 - audit must not mask the policy denial
            pass
        return CommandResult.denied(spec.name, reason)

    @staticmethod
    def _redact(app: Any, message: str) -> str:
        redact = getattr(app, "_redact", None)
        try:
            return str(redact(message))[:500] if callable(redact) else message[:500]
        except Exception:  # noqa: BLE001
            return message[:500]


__all__ = [
    "ADMIN_PRINCIPALS_DEFAULT", "ALL_SURFACES", "OPERATOR_PRINCIPALS_DEFAULT",
    "ArgumentSpec", "CommandContext", "CommandRegistry", "CommandResult",
    "CommandSpec", "CommandStatus", "ParsedCommand", "parse",
]
