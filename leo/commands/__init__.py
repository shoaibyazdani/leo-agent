"""Slash commands: one registry, one parser, one dispatch path.

Adapters (chat, CLI, HTTP, gateway) pass a trusted principal, surface and
channel to :meth:`CommandRegistry.dispatch`; they never reimplement command
semantics. Use :func:`create_registry` to build the canonical catalog.
"""

from __future__ import annotations

from .builtins import UNAVAILABLE, register_builtins
from .registry import (
    ADMIN_PRINCIPALS_DEFAULT,
    ALL_SURFACES,
    OPERATOR_PRINCIPALS_DEFAULT,
    OPERATOR_SURFACES_DEFAULT,
    ArgumentSpec,
    CommandContext,
    CommandRegistry,
    CommandResult,
    CommandSpec,
    CommandStatus,
    ParsedCommand,
    parse,
)


def create_registry(**kwargs: object) -> CommandRegistry:
    """Build the authoritative registry with every built-in spec registered."""
    registry = CommandRegistry(**kwargs)  # type: ignore[arg-type]
    return register_builtins(registry)


__all__ = [
    "ADMIN_PRINCIPALS_DEFAULT", "ALL_SURFACES", "OPERATOR_PRINCIPALS_DEFAULT",
    "OPERATOR_SURFACES_DEFAULT", "UNAVAILABLE", "ArgumentSpec", "CommandContext",
    "CommandRegistry", "CommandResult", "CommandSpec", "CommandStatus",
    "ParsedCommand", "create_registry", "parse", "register_builtins",
]
