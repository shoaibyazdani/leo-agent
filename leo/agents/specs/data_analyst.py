"""Data-analyst specialist: real, reproducible profiling of supplied data.

The handler computes statistics locally from data passed as an *input* to the
objective/context. Services are resolved only by ``make_handler(services)`` at
bind time -- this module never reads a service object out of the handler
context, never calls a model, and never reports a number it did not compute.
When no usable data is supplied it fails closed with a clear error.

This is a bespoke handler (``handler_kind = bespoke:make_handler``), not the
default delegation worker. Its tool grants are exact tool names from the live
data toolset, not broad categories.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

SEED: dict[str, Any] = {
    "id": "data-analyst",
    "name": "Data Analyst",
    "purpose": (
        "Profile and analyse supplied structured data (records, CSV or JSON) and "
        "report numerically verified summaries with the rows they were derived from."
    ),
    "capabilities": [
        "data_profiling",
        "descriptive_statistics",
        "csv_json_ingestion",
        "evidence_reporting",
    ],
    # Exact live tool names. No category-wide grant: granting 'data' would also
    # admit unrelated transforms. `json_query` is intentionally absent: it let a
    # model supply fabricated JSON as `data`, which broke provenance.
    "tool_permissions": ["csv_parse", "text_transform"],
    "model_preferences": ["reasoning", "tools", "structured"],
    "context_budget": 24000,
    "task_limit": 3,
    "metadata": {"handler_kind": "bespoke:make_handler"},
}


def _parse_data(value: Any, fmt: str | None = None) -> list[dict[str, Any]]:
    """Coerce context-supplied *data* (never services) into records."""
    if value is None:
        raise ValueError("data-analyst requires input data in context['data']")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("data-analyst received empty input data")
        kind = (fmt or "").lower()
        if kind not in ("json", "csv"):
            kind = "json" if text.startswith(("[", "{")) else "csv"
        if kind == "json":
            try:
                value = json.loads(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"data-analyst could not parse JSON data: {exc}") from exc
        else:
            value = list(csv.DictReader(io.StringIO(text)))
    if isinstance(value, Mapping):
        value = value.get("rows") if isinstance(value.get("rows"), list) else [value]
    if not isinstance(value, list) or not value:
        raise ValueError("data-analyst requires a non-empty list of record objects")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("data-analyst data rows must be objects")
        rows.append({str(key): cell for key, cell in item.items()})
    return rows


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _number(cell: Any) -> float | int | None:
    """Recognise numeric values, including the string cells CSV produces."""
    if isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        return cell
    if isinstance(cell, str) and _NUMBER_RE.match(cell.strip()):
        text = cell.strip()
        try:
            return float(text) if any(mark in text for mark in ".eE") else int(text)
        except ValueError:
            return None
    return None


def _profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all reported figures directly from the supplied rows."""
    keys = sorted({key for row in rows for key in row})
    fields: dict[str, Any] = {}
    for key in keys:
        present = [row[key] for row in rows if key in row]
        values = [cell for cell in present if cell is not None and cell != ""]
        numeric: list[float | int] = []
        for cell in values:
            number = _number(cell)
            if number is not None:
                numeric.append(number)
        stats: dict[str, Any] = {
            "present": len(present),
            "non_empty": len(values),
            "missing": len(rows) - len(present),
            "types": dict(sorted(Counter(type(cell).__name__ for cell in values).items())),
        }
        if numeric:
            stats.update({
                "numeric_count": len(numeric),
                "min": min(numeric),
                "max": max(numeric),
                "mean": sum(numeric) / len(numeric),
            })
        fields[key] = stats
    return {
        "method": "local-descriptive-statistics",
        "row_count": len(rows),
        "field_count": len(keys),
        "fields": fields,
    }


def make_handler(services: Mapping[str, Any] | None = None) -> Callable[..., Any]:
    """Bind the data-analysis handler.

    No service is required: the computation is local and deterministic. If a
    ``skills`` service is injected it is available to the engine's scoped tools,
    but the handler itself never constructs services from the call context.
    """

    async def handler(objective: Any = None, context: Any = None, **_: Any) -> dict[str, Any]:
        ctx = dict(context or {})
        rows = _parse_data(ctx.get("data"), ctx.get("format"))
        report = _profile(rows)
        numeric_fields = [
            name for name, stats in report["fields"].items() if "numeric_count" in stats
        ]
        evidence = [
            {
                "source": "data-analyst",
                "claim": (
                    f"Profiled {report['row_count']} rows across {report['field_count']} fields "
                    "from supplied data"
                ),
                "row_count": report["row_count"],
            },
            {
                "source": "data-analyst",
                "claim": f"Numeric fields computed directly from rows: {', '.join(numeric_fields) or 'none'}",
                "fields": numeric_fields,
            },
        ]
        return {
            "success": True,
            "output": {
                "report": report,
                "method": report["method"],
                "row_count": report["row_count"],
                "field_count": report["field_count"],
            },
            "evidence": evidence,
        }

    return handler
