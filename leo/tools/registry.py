"""Tool registration and JSON-schema (subset) argument validation.

Handlers have the shape ``async def handler(args: dict, ctx: dict) -> Any``.
``ctx`` is built by the executor and always contains at least ``settings``,
``db_path`` and ``actor``.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..contracts import ToolSpec

ToolHandler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]]

_FIELD_RE = re.compile(r"field '([^']+)'")
# Fields the model most often copies between tools. When one is rejected on the
# wrong tool, point at the canonical home first (e.g. limit -> search_files).
_CANONICAL_FIELD_HOME = {
    "limit": "search_files",
    "offset": "read_file",
    "cursor": "list_files",
}

_TYPE_NAMES: dict[str, str | tuple[str, ...]] = {
    "object": "dict",
    "array": "list",
    "string": "str",
    "number": "number",
    "integer": "int",
    "boolean": "bool",
    "null": "None",
}


@dataclass
class ToolRegistration:
    """A spec together with its callable. Spec fields are proxied for convenience."""

    spec: ToolSpec
    handler: ToolHandler

    def __getattr__(self, item: str) -> Any:
        # Only called for attributes missing from this dataclass.
        return getattr(self.spec, item)


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True  # unknown type keywords are not enforced by this subset


def _type_label(expected: str) -> str:
    return _TYPE_NAMES.get(expected, expected)


def _validate_value(
    value: Any, schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    if not isinstance(schema, dict):
        return

    if "enum" in schema and isinstance(schema["enum"], list):
        if not any(value == option and type(value) is type(option) for option in schema["enum"]):
            errors.append(f"field '{path}': must be one of {schema['enum']!r}")

    if "const" in schema and value != schema["const"]:
        errors.append(f"field '{path}': must equal {schema['const']!r}")

    expected = schema.get("type")
    allowed_types: list[str] = []
    if isinstance(expected, str):
        allowed_types = [expected]
    elif isinstance(expected, list):
        allowed_types = [t for t in expected if isinstance(t, str)]

    if allowed_types:
        matched = [t for t in allowed_types if _type_matches(value, t)]
        if not matched:
            labels = " or ".join(_type_label(t) for t in allowed_types)
            errors.append(
                f"field '{path}': expected {labels}, got {type(value).__name__}"
            )
            return
        active = matched[0]
    else:
        active = ""

    if active == "object" or (not active and isinstance(value, dict)):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    field_path = f"{path}.{key}" if path else key
                    errors.append(f"field '{field_path}': missing required field")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        if schema.get("additionalProperties") is False:
            unknown = [key for key in value if key not in properties]
            for key in unknown:
                field_path = f"{path}.{key}" if path else key
                errors.append(
                    f"field '{field_path}': unknown field (additionalProperties=false)"
                )
        for key, sub_schema in properties.items():
            if key in value and isinstance(sub_schema, dict):
                _validate_value(value[key], sub_schema, f"{path}.{key}" if path else key, errors)
        return

    if active == "array" or (not active and isinstance(value, list)):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_value(item, items, f"{path}[{index}]", errors)
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"field '{path}': expected at least {min_items} item(s)")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"field '{path}': expected at most {max_items} item(s)")
        return

    if active == "string" or (not active and isinstance(value, str)):
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        pattern = schema.get("pattern")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append(f"field '{path}': must be at least {min_len} character(s)")
        if isinstance(max_len, int) and len(value) > max_len:
            errors.append(f"field '{path}': must be at most {max_len} character(s)")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    errors.append(f"field '{path}': does not match pattern {pattern!r}")
            except re.error:
                pass
        return

    if active in ("number", "integer") or (
        not active and isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"field '{path}': must be a finite number")
            return
        if isinstance(value, (int, float)) and abs(value) > 10**15:
            errors.append(f"field '{path}': magnitude exceeds supported range")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"field '{path}': must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"field '{path}': must be <= {maximum}")


class ToolRegistry:
    """In-memory registry of named tools and their async handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}

    def register(
        self, spec: ToolSpec, handler: ToolHandler, *, replace: bool = False
    ) -> ToolRegistration:
        if not isinstance(spec, ToolSpec):
            raise TypeError("spec must be a ToolSpec")
        if not callable(handler):
            raise TypeError("handler must be callable")
        name = spec.name.strip()
        if not name:
            raise ValueError("tool name must not be empty")
        if name in self._tools and not replace:
            raise ValueError(f"tool already registered: {name}")
        spec.name = name
        registration = ToolRegistration(spec=spec, handler=handler)
        self._tools[name] = registration
        return registration

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolRegistration | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return [registration.spec for registration in self._tools.values()]

    @staticmethod
    def _properties(spec: Any) -> dict[str, Any]:
        schema = getattr(spec, "input_schema", None) or {}
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        return properties if isinstance(properties, dict) else {}

    def _suggest(self, tool_name: str, field: str, valid_fields: list[str]) -> str | None:
        """Suggest a valid field here, or a qualified field on another tool."""
        suggestions: list[str] = []
        if field not in valid_fields:
            elsewhere = [
                f"{other_name}.{field}"
                for other_name, other in self._tools.items()
                if other_name != tool_name and field in self._properties(other.spec)
            ]
            preferred = _CANONICAL_FIELD_HOME.get(field)
            elsewhere.sort(
                key=lambda item: (
                    0 if item.split(".", 1)[0] == preferred else 1,
                    item,
                )
            )
            suggestions.extend(f"'{item}'" for item in elsewhere[:2])
            close = difflib.get_close_matches(field, valid_fields, n=1, cutoff=0.6)
            if close:
                suggestions.append(f"'{close[0]}'")
        else:
            close = difflib.get_close_matches(field, valid_fields, n=2, cutoff=0.85)
            suggestions.extend(f"'{item}'" for item in close if item != field)
        return " or ".join(suggestions) if suggestions else None

    def _actionable_errors(self, name: str, errors: list[str]) -> list[str]:
        """Add tool/field/valid-field/suggestion context, keeping raw detail."""
        registration = self.get(name)
        valid_fields = list(self._properties(registration.spec)) if registration else []
        valid_text = ", ".join(valid_fields) if valid_fields else "(none declared)"
        actionable: list[str] = []
        for error in errors:
            match = _FIELD_RE.search(error)
            if not match:
                actionable.append(f"tool '{name}': {error}")
                continue
            path = match.group(1)
            field = path.split(".")[0].split("[")[0] or path
            if "unknown field" in error:
                headline = f"unknown field '{field}'"
            elif "missing required" in error:
                headline = f"missing required field '{field}'"
            else:
                headline = f"invalid value for field '{path}'"
            parts = [f"tool '{name}'", headline, f"valid fields: {valid_text}"]
            suggestion = None if "missing required" in error else self._suggest(
                name, field, valid_fields
            )
            if suggestion:
                parts.append(f"did you mean {suggestion}?")
            parts.append(f"validator detail: {error}")
            actionable.append("; ".join(parts))
        return actionable

    def validate(self, name: str, arguments: dict[str, Any]) -> list[str]:
        """Return actionable, human-readable errors; empty means valid.

        The raw validator message is retained as trailing detail so existing
        callers that look for 'expected str', 'unknown field', etc. keep working.
        """
        registration = self.get(name)
        if registration is None:
            return [f"unknown tool: {name!r}"]
        if not isinstance(arguments, dict):
            return [
                f"tool '{name}': arguments must be an object, "
                f"got {type(arguments).__name__}"
            ]
        errors: list[str] = []
        schema = registration.spec.input_schema or {"type": "object", "properties": {}}
        _validate_value(arguments, schema, "", errors)
        return self._actionable_errors(name, errors) if errors else errors
