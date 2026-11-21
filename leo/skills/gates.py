"""Evaluation of declarative skill availability gates.

Gates are *conditions*, never grants of authority. They are evaluated from
local, bounded observations (environment variable presence, binary discovery,
injected tool inventories and the current platform). No value of an environment
variable, config entry or document is ever disclosed in a result or error.

Translated upstream shapes are supported directly:

* Legacy gateway shape: ``bins`` / ``env`` / ``config`` / ``primaryEnv`` require
  every item; ``anyBins`` and ``os`` require at least one match.
* Legacy agent shape: ``tools`` / ``toolsets`` / ``commands`` require every item;
  ``fallback_for_tools`` / ``fallback_for_toolsets`` require that none of the
  listed capabilities is available.
* A gate may carry its own ``polarity`` (``any`` / ``all`` / ``none``), and the
  whole gate mapping may carry ``mode`` with the same values.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts import SkillManifest

_POLARITIES = frozenset({"any", "all", "none"})

# Group-specific defaults. Unknown groups fail closed: an unverifiable gate must
# never be reported as satisfied.
_ALL_GROUPS = frozenset({
    "bins", "commands", "required_commands", "env", "environment",
    "required_environment_variables", "config", "tools", "toolsets",
    "allowed_tools", "primaryEnv", "prerequisites",
})
_ANY_GROUPS = frozenset({"anyBins", "os", "platforms", "session_platforms"})
_NONE_GROUPS = frozenset({"fallback_for_tools", "fallback_for_toolsets"})
# Descriptive metadata inside a gate mapping, not an availability requirement.
_DESCRIPTIVE_KEYS = frozenset({
    "mode", "polarity", "install", "user_invocable", "always", "description",
    "note", "source", "token", "path_tokens",
})

_MAX_DETAIL_ITEMS = 8
_MAX_DETAIL_CHARS = 240
_PREVIEW_LIMIT = 120


@dataclass(frozen=True)
class GateFailure:
    """One named, unsatisfied gate condition (no values or secret contents)."""

    group: str
    item: str = ""
    reason: str = "not satisfied"

    @property
    def label(self) -> str:
        return f"{self.group}:{self.item}" if self.item else self.group

    def describe(self) -> str:
        return f"{self.label} ({self.reason})"


@dataclass(frozen=True)
class GateEvaluation:
    satisfied: bool
    failures: tuple[GateFailure, ...] = ()
    checked: tuple[str, ...] = ()

    @property
    def detail(self) -> str:
        """Short, bounded detail containing gate *names* only."""
        parts = [failure.describe() for failure in self.failures[:_MAX_DETAIL_ITEMS]]
        if len(self.failures) > _MAX_DETAIL_ITEMS:
            parts.append(f"+{len(self.failures) - _MAX_DETAIL_ITEMS} more gates")
        return ", ".join(parts)[:_MAX_DETAIL_CHARS]


def bounded_preview(value: Any, limit: int = _PREVIEW_LIMIT) -> str:
    """Collapse untrusted text into a bounded single-line preview."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _names(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        keys = {"name", "value", "key", "id"}
        result = [str(value[k]) for k in keys if k in value and value[k] not in (None, "")]
        return result or [str(k) for k in value if k not in _DESCRIPTIVE_KEYS]
    if isinstance(value, (list, tuple, set, frozenset)):
        result: list[str] = []
        for item in value:
            result.extend(_names(item))
        return result
    return [str(value)]


def _as_set(value: Any) -> set[str]:
    return {str(item).strip().lower() for item in _names(value) if str(item).strip()}


def _context_set(context: Mapping[str, Any], keys: tuple[str, ...]) -> set[str]:
    for key in keys:
        if key in context:
            return _as_set(context[key])
    return set()


def _polarity(value: Any, default: str) -> str:
    if isinstance(value, Mapping):
        candidate = value.get("polarity", value.get("mode"))
        if isinstance(candidate, str) and candidate.strip().lower() in _POLARITIES:
            return candidate.strip().lower()
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _POLARITIES:
            return normalized
    return default


def _group_items(value: Any) -> list[str]:
    """Return the concrete values of a gate, honouring dict gate specifications."""
    if isinstance(value, Mapping):
        for key in ("values", "items", "any", "all", "none", "required", "names"):
            if key in value:
                return _names(value[key])
        return _names(value)
    return _names(value)


def _platform_names(value: Any) -> set[str]:
    normalized: set[str] = set()
    for name in _names(value):
        text = str(name).strip().lower()
        if text in {"macos", "osx", "darwin"}:
            normalized.add("darwin")
        elif text in {"win32", "windows", "win"}:
            normalized.add("windows")
        else:
            normalized.add(text)
    return normalized


def _current_platform(context: Mapping[str, Any]) -> str:
    candidate = context.get("platform") or context.get("os")
    if isinstance(candidate, str):
        return _platform_names(candidate).pop()
    return _platform_names(platform.system()).pop()


def _available(
    group: str,
    items: list[str],
    context: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> set[str]:
    """Return the subset of gate items observed as available."""
    if group in {"os", "platforms", "session_platforms"}:
        current = _current_platform(context)
        return {item for item in items if _platform_names(item) == {current}}
    if group in {"bins", "anyBins", "commands", "required_commands"}:
        supplied = _context_set(context, ("available_bins", "bins", "commands"))
        return {
            item for item in items
            if item.lower() in supplied or shutil.which(item) is not None
        }
    if group in {"env", "environment", "required_environment_variables", "primaryEnv"}:
        supplied_env = context.get("env", context.get("environment", environment))
        keys = _as_set(supplied_env)
        return {item for item in items if item.lower() in keys}
    if group in {"config"}:
        supplied = _context_set(context, ("available_config", "config"))
        env_keys = _as_set(context.get("env", context.get("environment", environment)))
        return {item for item in items if item.lower() in supplied or item.lower() in env_keys}
    if group in {"tools", "allowed_tools", "fallback_for_tools"}:
        supplied = _context_set(context, ("available_tools", "tools"))
        return {item for item in items if item.lower() in supplied}
    if group in {"toolsets", "fallback_for_toolsets"}:
        supplied = _context_set(context, ("available_toolsets", "toolsets"))
        return {item for item in items if item.lower() in supplied}
    if group == "prerequisites":
        supplied = _context_set(context, ("available_prerequisites", "prerequisites"))
        return {item for item in items if item.lower() in supplied}
    # Unknown gate: never assume an unverified capability exists.
    return set()


def _check_group(
    group: str,
    raw: Any,
    context: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> tuple[bool, list[GateFailure]]:
    items = _group_items(raw)
    if not items:
        return True, []
    polarity = _polarity(raw, "none" if group.startswith("fallback_for_") else (
        "any" if group in _ANY_GROUPS else "all"
    ))
    # Per-item negation: "!tool" means the tool must be absent.
    positive = [item for item in items if not item.startswith("!")]
    negative = [item.lstrip("!") for item in items if item.startswith("!")]

    failures: list[GateFailure] = []
    positive_available = _available(group, positive, context, environment) if positive else set()
    if polarity in {"all", "any"}:
        if polarity == "all":
            for item in positive:
                if item not in positive_available:
                    failures.append(GateFailure(group, item, "missing or unavailable"))
        elif not positive_available:
            failures.append(GateFailure(group, "|".join(positive), "none available"))
    elif polarity == "none" and positive_available:
        for item in positive:
            if item in positive_available:
                failures.append(GateFailure(group, item, "must be absent"))

    if negative:
        negative_available = _available(group, negative, context, environment)
        for item in negative:
            if negative_available and item in negative_available:
                failures.append(GateFailure(group, item, "must be absent"))
        if polarity == "any" and not positive_available and not negative_available:
            # `any` with only negated items still needs at least one to hold.
            pass
    return not failures, failures


def evaluate_gates(
    gates: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
    *,
    environment: Mapping[str, Any] | None = None,
    env: Mapping[str, Any] | None = None,
) -> GateEvaluation:
    """Evaluate a translated gate mapping; fail closed on unknown conditions."""
    if not gates:
        return GateEvaluation(satisfied=True)
    if not isinstance(gates, Mapping):
        return GateEvaluation(False, (GateFailure("gates", reason="must be an object"),))
    context = context if isinstance(context, Mapping) else {}
    environment = environment if environment is not None else (env if env is not None else os.environ)
    mode = gates.get("mode", gates.get("polarity", "all"))
    mode = mode.strip().lower() if isinstance(mode, str) else "all"
    if mode not in _POLARITIES:
        return GateEvaluation(False, (GateFailure("mode", str(mode), "unsupported polarity"),))

    group_results: list[tuple[str, bool, list[GateFailure]]] = []
    for group, raw in gates.items():
        if group in _DESCRIPTIVE_KEYS or raw in (None, "", [], {}, False):
            continue
        if group == "prerequisites" and isinstance(raw, Mapping):
            # Upstream prerequisites are a nested mapping of gate groups.
            for sub_group, sub_value in raw.items():
                if sub_value in (None, "", [], {}, False):
                    continue
                target = {
                    "command": "commands", "commands": "commands",
                    "env": "env", "environment": "env", "tools": "tools",
                    "bins": "bins",
                }.get(str(sub_group), str(sub_group))
                group_results.append((f"prerequisites.{sub_group}", *_check_group(target, sub_value, context, environment)))
            continue
        group_results.append((str(group), *_check_group(str(group), raw, context, environment)))

    if not group_results:
        return GateEvaluation(satisfied=True)
    failures = [failure for _, _, group_failures in group_results for failure in group_failures]
    checked = tuple(name for name, _, _ in group_results)
    if mode == "all":
        satisfied = all(ok for _, ok, _ in group_results)
    elif mode == "any":
        satisfied = any(ok for _, ok, _ in group_results)
        if not satisfied:
            failures = [GateFailure("any", "|".join(checked), "no gate group is satisfied"), *failures]
    else:  # none
        satisfied = not any(ok for _, ok, _ in group_results)
        if not satisfied:
            passed = [name for name, ok, _ in group_results if ok]
            failures = [GateFailure("none", "|".join(passed), "gate group must be absent"), *failures]
    return GateEvaluation(satisfied=satisfied, failures=tuple(failures), checked=checked)


def execution_block(manifest: SkillManifest) -> str | None:
    """Return a fixed reason why an external manifest must not be executed."""
    if manifest.source == "native":
        return None
    if manifest.entrypoint:
        return "external manifests cannot declare an entrypoint"
    if not manifest.provenance_verified or not manifest.provenance_approval_id:
        return "external provenance is not verified or approved; no executable handler is granted"
    return None
