"""Tool engine: registry, execution, permissions and native tools.

The composition root uses :func:`register_all_toolsets` so every shipped
``*_tools.py`` module is wired through the same registry and executor. A tool
that is not registered here cannot be called by a model or an agent.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Mapping

from .builtins import register_builtins
from .executor import ToolExecutorImpl
from .registry import ToolHandler, ToolRegistration, ToolRegistry

__all__ = [
    "ToolRegistry",
    "ToolRegistration",
    "ToolHandler",
    "ToolExecutorImpl",
    "register_builtins",
    "register_all_toolsets",
    "MODEL_FORBIDDEN_TOOLS",
]

# These are operator-only state transitions, never model-callable. Owners
# remove them from their public register(); removing them here as well keeps
# the chat/agent registry honest even if an older module still exposes them.
#
# ``json_query`` is also excluded: its ``data`` and ``default`` arguments are
# supplied by the caller, so a model could invent a number, feed it back as
# "data" and present the echoed value as evidence. Pure offline callers (tests,
# operator scripts, direct handler use) can still call data_tools.json_query;
# only the composed model-facing catalogue drops it. Live data must come from
# http_json/http_json_paths, whose values are bound to a real fetched response.
MODEL_FORBIDDEN_TOOLS: frozenset[str] = frozenset(
    {"approval_resolve", "webhook_sign", "json_query"}
)


def register_all_toolsets(
    registry: ToolRegistry,
    services: Mapping[str, Any],
    *,
    settings: Any = None,
    permissions: Any = None,
) -> dict[str, list[str]]:
    """Import and register every ``leo/tools/*_tools.py`` module.

    Returns a mapping of module name to registered tool names. Operator tools
    (browser/computer) use their dedicated registrar and still execute through
    the same executor/permission path.
    """
    roster: dict[str, list[str]] = {}
    package = Path(__file__).resolve().parent
    for info in sorted(pkgutil.iter_modules([str(package)]), key=lambda item: item.name):
        if info.ispkg or not info.name.endswith("_tools"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        registrar = getattr(module, "register", None)
        if callable(registrar):
            try:
                names = list(registrar(registry, services=dict(services)))
            except NameError as exc:
                # Compatibility with an in-flight registrar rename: the
                # read-only specs factory is ``_model_specs`` in this build,
                # but ``register()`` still calls ``_specs``. Bind the alias on
                # the module itself; no tool definition is altered.
                if info.name == "approval_tools" and "_specs" in str(exc) and hasattr(module, "_model_specs"):
                    module._specs = module._model_specs
                    names = list(registrar(registry, services=dict(services)))
                else:
                    raise
        elif info.name == "operator_tools" and callable(getattr(module, "register_operator_tools", None)):
            names = list(
                module.register_operator_tools(registry, settings, permissions)
            )
        else:
            raise RuntimeError(f"toolset {info.name} has no register() entry point")
        roster[info.name] = names

    for forbidden in MODEL_FORBIDDEN_TOOLS:
        registry.unregister(forbidden)
    return roster
