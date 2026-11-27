#!/usr/bin/env python3
"""Extract a static, brand-neutral scheduling oracle corpus.

This script parses reference files as data. It never imports or executes code from
that corpus. Run with ``python -I`` when the source checkout is untrusted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ORACLE_COMMIT = "7cf9e92cd6277b327b36a51fb897cdc8403902d6"
SCHEMA_VERSION = 1
ACTIONS = ("create", "list", "update", "pause", "resume", "remove", "run", "cancel")
SOURCE_FILES = (
    "tools/cronjob_tools.py",
    "tools/approval.py",
    "tools/approval_prompt.py",
    "cron/jobs.py",
    "cron/job_args.py",
    "cron/jobs_time.py",
    "cron/jobs_tristate.py",
    "cron/jobs_invariants.py",
    "cron/jobs_model_pin.py",
    "cron/scheduler.py",
    "cron/scheduler_prompt.py",
    "cron/scheduler_delivery.py",
    "cron/scheduler_delivery_room.py",
    "cron/monitor.py",
    "gateway/platforms/base_exec_approval.py",
    "locales/en.yaml",
    "tests/tools/test_cronjob_create_approval.py",
    "tests/tools/test_cronjob_tools.py",
    "tests/tools/test_approval_prompt_schedule.py",
    "tests/cron/test_jobs.py",
    "tests/cron/test_duration_dst.py",
    "tests/cron/test_per_job_timezone.py",
    "tests/cron/test_job_args.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def neutralize(text: str) -> str:
    """Remove source-project branding without embedding brand names in LEO files."""
    text = re.sub(r"\b[A-Z][A-Za-z]+ wants\b", "Upstream wants", text)
    text = re.sub(r"\b[A-Z][A-Z0-9_]*_HOME\b", "SOURCE_PROJECT_HOME", text)
    text = re.sub(r"`[a-z]+ model` / `/[a-z]+`", "`/model`", text)
    text = re.sub(r"`[a-z]+ model`", "`/model`", text)
    text = re.sub(r"\b[a-z]+ config set\b", "config set", text)
    return text


def read_ast(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


def assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise ValueError(f"missing assignment: {name}")


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"missing function: {name}")


def safe_eval(node: ast.AST, *, functions: dict[str, ast.FunctionDef] | None = None,
              names: dict[str, Any] | None = None) -> Any:
    """Evaluate only literal AST shapes and the local script-description template."""
    names = names or {}
    functions = functions or {}
    if isinstance(node, ast.Starred):
        return safe_eval(node.value, functions=functions, names=names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {safe_eval(key, functions=functions, names=names): safe_eval(value, functions=functions, names=names)
                for key, value in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = []
        for item in node.elts:
            value = safe_eval(item, functions=functions, names=names)
            values.extend(value if isinstance(item, ast.Starred) else [value])
        return tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = safe_eval(node.operand, functions=functions, names=names)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return safe_eval(node.left, functions=functions, names=names) + safe_eval(node.right, functions=functions, names=names)
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id == "None":
            return None
        if node.id in functions:
            return f"<{node.id}>"
        raise ValueError(f"unsupported name: {node.id}")
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(safe_eval(value.value, functions=functions, names=names)))
            else:
                raise ValueError("unsupported f-string component")
        return "".join(parts)
    if isinstance(node, ast.Lambda):
        return ast.unparse(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_script_description":
        local = dict(names)
        for argument in node.args:
            local["home"] = safe_eval(argument, functions=functions, names=local)
        returns = [item for item in functions["_script_description"].body if isinstance(item, ast.Return)]
        if len(returns) != 1:
            raise ValueError("script description must have one return")
        return safe_eval(returns[0].value, functions=functions, names=local)
    raise ValueError(f"unsupported AST node: {type(node).__name__}")


def yaml_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if text.startswith('"'):
        return json.loads(text)
    if text.startswith("'"):
        return ast.literal_eval(text)
    if text in {"null", "~"}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        return text


def yaml_section(path: Path, wanted: tuple[str, ...]) -> dict[str, Any]:
    """Read a small nested mapping of scalar YAML values without a YAML dependency."""
    parents: list[tuple[int, str]] = []
    result: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while parents and indent <= parents[-1][0]:
            parents.pop()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            continue
        if tuple(name for _, name in parents) == wanted and value.strip():
            result[key] = yaml_scalar(value)
        elif not value.strip():
            parents.append((indent, key))
    return result


def function_record(source: str, tree: ast.Module, name: str) -> dict[str, Any]:
    node = function(tree, name)
    text = ast.get_source_segment(source, node) or ""
    return {
        "name": name,
        "line_start": node.lineno,
        "line_end": node.end_lineno,
        "sha256": sha256_bytes(text.encode("utf-8")),
    }


def extract_tool_schema(source_root: Path) -> dict[str, Any]:
    path = source_root / "tools/cronjob_tools.py"
    source, tree = read_ast(path)
    functions = {"_script_description": function(tree, "_script_description")}
    raw_schema = safe_eval(assignment(tree, "CRONJOB_SCHEMA"), functions=functions)
    schema = {
        "name": raw_schema["name"],
        "description": neutralize(raw_schema["description"]),
        "parameters": raw_schema["parameters"],
    }
    schema["parameters"]["properties"]["script"]["description"] = neutralize(
        schema["parameters"]["properties"]["script"]["description"]
    )
    schema["parameters"]["properties"]["pinned"]["description"] = neutralize(
        schema["parameters"]["properties"]["pinned"]["description"]
    )
    forwarded = list(safe_eval(assignment(tree, "_HANDLER_FORWARDED_ARGS")))
    register = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "register")
    keywords = {keyword.arg: keyword.value for keyword in register.keywords}
    compat = list(safe_eval(keywords["compat_args"], names={"_HANDLER_FORWARDED_ARGS": forwarded}))
    jobless = safe_eval(assignment(tree, "_JOBLESS_ACTIONS"), functions={
        name: function(tree, name) for name in ("_action_create", "_action_list")
    })
    job_actions_node = assignment(tree, "_JOB_ACTIONS")
    job_actions = {
        safe_eval(key): ast.unparse(value)
        for key, value in zip(job_actions_node.keys, job_actions_node.values)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "tool_name": schema["name"],
        "toolset": safe_eval(keywords["toolset"]),
        "description": schema["description"],
        "source_description_sha256": sha256_bytes(raw_schema["description"].encode("utf-8")),
        "description_policy": "source-project branding neutralized; source digest binds exact text",
        "parameters": schema["parameters"],
        "handler_forwarded_args": forwarded,
        "compat_args": compat,
        "jobless_actions": jobless,
        "job_actions": job_actions,
        "declared_actions": list(ACTIONS),
        "action_aliases": {"run": ["run_now", "trigger"]},
        "get_action_present": False,
        "shared_schema_notes": [
            "The source has one shared schema, not a per-action JSON Schema.",
            "Only action is required by the declared schema.",
            "The declared schema has no enum and no additionalProperties=false.",
            "include_disabled, skill, reason and grace_minutes are accepted by the handler but absent from the declared schema.",
            "monitor_script and monitor_url are legacy compatibility arguments, not declared model fields.",
        ],
        "source": function_record(source, tree, "_cronjob_handler"),
    }


def extract_actions(source_root: Path, tool_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    properties = tool_schema["parameters"]["properties"]
    common_create = [
        "schedule", "prompt", "name", "repeat", "deliver", "failure_deliver", "skill", "skills",
        "script", "monitor", "context_from", "continuity", "enabled_toolsets", "workdir", "no_agent",
        "attach_to_session", "paused", "paused_reason", "pinned", "timezone", "grace_minutes",
        "catch_up", "wrap_response",
    ]
    effective = {
        "create": common_create,
        "list": ["include_disabled"],
        "update": ["job_id", *[field for field in common_create if field not in {"paused", "paused_reason"}]],
        "pause": ["job_id", "reason"],
        "resume": ["job_id"],
        "remove": ["job_id"],
        "run": ["job_id", "prompt"],
        "cancel": ["job_id", "reason"],
    }
    required = {
        "create": ["schedule"], "list": [], "update": ["job_id"], "pause": ["job_id"],
        "resume": ["job_id"], "remove": ["job_id"], "run": ["job_id"], "cancel": ["job_id"],
    }
    action_descriptions = {
        "create": "action='create' schedules a job from a prompt and/or skills",
        "list": "'list' inspects jobs",
        "update": "'update'/'pause'/'resume'/'remove' manage one by job_id (always list first — never guess job IDs)",
        "pause": "'update'/'pause'/'resume'/'remove' manage one by job_id (always list first — never guess job IDs)",
        "resume": "'update'/'pause'/'resume'/'remove' manage one by job_id (always list first — never guess job IDs)",
        "remove": "'update'/'pause'/'resume'/'remove' manage one by job_id (always list first — never guess job IDs)",
        "run": "'run' fires a job immediately in the BACKGROUND (returns a handle at once, outcome re-enters the conversation when done — do not wait or poll; optional 'prompt' adds transient context for that fire only)",
        "cancel": "'cancel' stops a job's LIVE run without touching its schedule (a no-op when nothing is in flight)",
    }
    result_keys = {
        "create": ["success", "job_id", "name", "skill", "skills", "schedule", "repeat", "deliver", "next_run_at", "job", "message"],
        "list": ["success", "count", "jobs"],
        "update": ["success", "job"],
        "pause": ["success", "job"],
        "resume": ["success", "job"],
        "remove": ["success", "message", "removed_job"],
        "run": ["success", "job", "executed", "execution_mode", "delegation_id", "note", "execution_success", "execution_skipped", "execution_error"],
        "cancel": ["job", "success", "cancelled", "execution_id", "status", "pid", "detail"],
    }
    consent = {
        "create": "schedule when a card is answerable; LEO diverges fail-closed when it is not",
        "update": "schedule_edit only for card-listed semantic changes",
        "list": None, "pause": None, "resume": None, "remove": None, "run": None, "cancel": None,
    }
    fixtures: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        action_properties = {"action": properties["action"]}
        for field in effective[action]:
            if field in properties:
                action_properties[field] = properties[field]
        fixtures[action] = {
            "schema_version": SCHEMA_VERSION,
            "oracle_commit": ORACLE_COMMIT,
            "tool_name": tool_schema["tool_name"],
            "action": action,
            "aliases": tool_schema["action_aliases"].get(action, []),
            "description": tool_schema["parameters"]["properties"]["action"]["description"],
            "action_description": action_descriptions[action],
            "input_schema": {
                "type": "object",
                "properties": action_properties,
                "required": required[action],
            },
            "effective_handler_fields": effective[action],
            "fields_not_in_declared_schema": [field for field in effective[action] if field not in properties],
            "result_keys": result_keys[action],
            "upstream_consent": consent[action],
            "schema_derivation": "derived from the shared schema plus the final-head handler; not an upstream per-action schema",
        }
    return fixtures


def extract_fields(source_root: Path, tool_schema: dict[str, Any]) -> dict[str, Any]:
    properties = tool_schema["parameters"]["properties"]
    forwarded = set(tool_schema["handler_forwarded_args"])
    semantics = {
        "action": "Dispatches create/list/update/pause/resume/remove/run/cancel; run_now and trigger are run aliases.",
        "schedule": "Parsed to once, interval or cron before persistence; bare durations recur and 'in <duration>' is one-shot.",
        "prompt": "Full self-contained job instruction; run uses it as transient one-fire context.",
        "name": "Optional saved label; blank falls back to prompt/skill/script-derived text.",
        "repeat": "Omitted is forever for recurring and one for one-shot; numeric strings normalize; non-positive becomes forever in the oracle.",
        "deliver": "Comma-separated routing tokens resolved by the same scheduler resolver used at run time.",
        "failure_deliver": "Optional separate failure lane; empty string clears on update.",
        "skill": "Legacy singular skill alias; canonicalized into ordered unique skills.",
        "skills": "Ordered unique skill names loaded before the job prompt; [] clears on update.",
        "script": "Relative file under the source project's scripts directory; stdout feeds the agent unless no_agent is true.",
        "monitor": "http(s) URL or script path; first successful sample baselines, unchanged suppresses, change wakes the agent.",
        "no_agent": "Requires script; stdout is the result, empty stdout sends nothing, nonzero/timeout is an error.",
        "context_from": "Existing job IDs whose latest completed output is injected; self denotes this job.",
        "continuity": "True adds self to context_from; false removes it.",
        "enabled_toolsets": "Absent inherits defaults; [] is an explicit zero-tool allowlist.",
        "workdir": "Absolute existing directory anchoring tools/scripts; relative and missing paths reject.",
        "attach_to_session": "Makes an eligible single delivery conversation continuable; broadcast and local are not attached.",
        "paused": "Create-only disabled state; explicit run remains available.",
        "paused_reason": "Create-only auditable reason requiring paused=true.",
        "pinned": "True snapshots the current operator provider/model; false clears the pin; no separate pinned field is stored.",
        "timezone": "IANA zone for recurring wall-clock schedules; empty clears on update.",
        "grace_minutes": "Cron-only catch-up override in whole minutes from 0 through 1440; 0 is preserved.",
        "catch_up": "Tri-state missed-run policy; only real booleans or null are accepted by storage.",
        "wrap_response": "on/off/inherit maps to true/false/null for delivered framing.",
        "include_disabled": "Handler-only list switch; model dispatch defaults it to true.",
        "reason": "Handler-only pause/cancel reason; absent from the declared model schema.",
        "provider": "Operator/programmatic inference provider; never forwarded by the model handler.",
        "model": "Operator/programmatic inference model; never forwarded by the model handler.",
        "base_url": "Operator/programmatic endpoint with credential-origin validation; never forwarded by the model handler.",
        "reasoning_effort": "Operator/programmatic inference effort override; absent from the model surface.",
        "interpreter": "Operator/programmatic Python interpreter for scripts; absent from the model surface.",
        "monitor_script": "Legacy stored script monitor alias; model-facing monitor is canonical.",
        "monitor_url": "Legacy stored URL monitor alias; model-facing monitor is canonical.",
        "session_id": "Internal background-run completion routing key, not a model field.",
        "approve": "Internal model-boundary flag; direct Python callers omit it.",
    }
    storage = {
        "script": "script", "monitor": "monitor_script or monitor_url", "no_agent": "no_agent",
        "context_from": "context_from", "continuity": "context_from contains self",
        "enabled_toolsets": "enabled_toolsets", "skills": "skills and legacy skill",
        "wrap_response": "wrap_response", "attach_to_session": "attach_to_session",
        "pinned": "provider/model", "grace_minutes": "grace_minutes", "repeat": "repeat.times",
    }
    exposures: dict[str, list[str]] = {}
    for field in properties:
        exposures.setdefault(field, []).append("declared_model_schema")
    for field in forwarded:
        exposures.setdefault(field, []).append("model_handler")
    for field in tool_schema["compat_args"]:
        exposures.setdefault(field, []).append("legacy_compat")
    for field in ("include_disabled", "monitor", "paused", "session_id", "approve"):
        exposures.setdefault(field, []).append("internal_or_handler_direct")
    for field in ("provider", "model", "base_url", "reasoning_effort", "interpreter"):
        exposures.setdefault(field, []).append("operator_programmatic_only")
    fields: dict[str, Any] = {}
    for field in sorted(exposures):
        fields[field] = {
            "exposure": exposures[field],
            "declared_schema": properties.get(field),
            "semantics": semantics.get(field, "See the frozen shared schema and handler source."),
            "storage": storage.get(field, "handler/local value"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "declared_model_fields": sorted(properties),
        "handler_forwarded_fields": tool_schema["handler_forwarded_args"],
        "legacy_compat_fields": tool_schema["compat_args"],
        "operator_only_fields": ["provider", "model", "base_url", "reasoning_effort", "interpreter"],
        "internal_fields": ["approve", "session_id"],
        "fields": fields,
        "required_runtime_semantics": sorted({
            "script", "no_agent", "monitor", "context_from", "continuity", "enabled_toolsets",
            "skills", "wrap_response", "workdir", "attach_to_session", "pinned", "grace_minutes",
        }),
        "leo_field_adaptations": {
            "script": "operator-allowlisted executable path, opt-in, args supplied by operator provisioning rather than a new model field",
            "no_agent": "distinct native script-only execution with durable evidence",
            "monitor": "existing egress-checked HTTP client or restricted script runner; no headers/cookies/private targets",
            "context_from": "latest completed native task outputs, cycle rejection and bounded injection",
            "continuity": "bounded own previous completed output with a first-run note",
            "enabled_toolsets": "intersect the agent allowlist; [] means no tools",
            "skills": "freeze a creation-time documentation snapshot and inject it at run time",
            "wrap_response": "control delivered text and include it in the edit fingerprint",
            "workdir": "anchor tools/scripts inside LEO allowed roots; not an OS chroot",
            "attach_to_session": "create/use a durable native job conversation and expose a compact next-reply brief",
            "pinned": "store operator-owned provider/model; missing target fails closed with needs-you",
            "grace_minutes": "operator/programmatic-only late-run policy; model requests are rejected",
        },
        "removed_scope_absent": [
            "inline script authoring", "per-job script state directories", "gate-context/watchdog machinery",
            "connector-call facade", "watcher skills/templates",
        ],
    }


def parser_examples(source_root: Path) -> list[tuple[int, str]]:
    _, tree = read_ast(source_root / "tests/cron/test_jobs.py")
    examples: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "parse_schedule":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                examples.append((node.lineno, node.args[0].value))
    return examples


def extract_grammar(source_root: Path) -> dict[str, Any]:
    expected: dict[str, dict[str, Any]] = {
        "every hour": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "every day": {"kind": "interval", "minutes": 1440, "display": "every 1440m"},
        "30m": {"kind": "interval", "minutes": 30, "display": "every 30m"},
        "in 30m": {"kind": "once", "run_at_rule": "configured now + 30 elapsed minutes", "display": "once in 30m"},
        "every 2h": {"kind": "interval", "minutes": 120, "display": "every 120m"},
        "every monday 9am": {"kind": "cron", "expr": "0 9 * * 1", "display": "every monday 9am"},
        "0 9 * * *": {"kind": "cron", "expr": "0 9 * * *", "display": "0 9 * * *"},
        "2030-01-15T14:00:00": {"kind": "once", "run_at_rule": "2030-01-15T14:00:00 in configured profile zone", "display": "once at 2030-01-15 14:00"},
        "2026-06-22T20:07:00": {"kind": "once", "run_at_rule": "naive wall time in configured profile zone, not per-job zone", "display": "once at 2026-06-22 20:07"},
        "monday banana at 9am": {"error_contains": "Invalid schedule"},
        "funday at 9am": {"error_contains": "Invalid schedule"},
        "every monday": {"error_contains": "Invalid duration"},
        "every monday 25am": {"error_contains": "Invalid duration"},
        "every monday 9pm pizza": {"error_contains": "Invalid duration"},
        "0 9 * * FUNDAY": {"error_contains": "Invalid cron expression"},
        "every sunday 9am": {"kind": "cron", "expr": "0 9 * * 0", "display": "every sunday 9am"},
        "every mon 9am": {"kind": "cron", "expr": "0 9 * * 1", "display": "every mon 9am"},
        "every fri 5pm": {"kind": "cron", "expr": "0 17 * * 5", "display": "every fri 5pm"},
        "every day at 9am": {"kind": "cron", "expr": "0 9 * * *", "display": "every day at 9am"},
        "every day 7am": {"kind": "cron", "expr": "0 7 * * *", "display": "every day 7am"},
        "every weekday at 9am": {"kind": "cron", "expr": "0 9 * * 1-5", "display": "every weekday at 9am"},
        "every weekend at 10am": {"kind": "cron", "expr": "0 10 * * 0,6", "display": "every weekend at 10am"},
        "weekdays at 9am": {"kind": "cron", "expr": "0 9 * * 1-5", "display": "weekdays at 9am"},
        "monday at 9:30": {"kind": "cron", "expr": "30 9 * * 1", "display": "monday at 9:30"},
        "daily at 7am": {"kind": "cron", "expr": "0 7 * * *", "display": "daily at 7am"},
        "Monday, Wednesday at 9am": {"kind": "cron", "expr": "0 9 * * 1,3", "display": "Monday, Wednesday at 9am"},
        "monday and friday at 5pm": {"kind": "cron", "expr": "0 17 * * 1,5", "display": "monday and friday at 5pm"},
        "every tue, thu 8am": {"kind": "cron", "expr": "0 8 * * 2,4", "display": "every tue, thu 8am"},
        "every monday 14:30": {"kind": "cron", "expr": "30 14 * * 1", "display": "every monday 14:30"},
        "every monday 9:05am": {"kind": "cron", "expr": "5 9 * * 1", "display": "every monday 9:05am"},
        "every monday noon": {"kind": "cron", "expr": "0 12 * * 1", "display": "every monday noon"},
        "every monday midnight": {"kind": "cron", "expr": "0 0 * * 1", "display": "every monday midnight"},
        "every monday at 7": {"kind": "cron", "expr": "0 7 * * 1", "display": "every monday at 7"},
        "every monday 12am": {"kind": "cron", "expr": "0 0 * * 1", "display": "every monday 12am"},
        "every monday 12pm": {"kind": "cron", "expr": "0 12 * * 1", "display": "every monday 12pm"},
        "Every Monday 9AM": {"kind": "cron", "expr": "0 9 * * 1", "display": "Every Monday 9AM"},
        "every 30m": {"kind": "interval", "minutes": 30, "display": "every 30m"},
        "every 1d": {"kind": "interval", "minutes": 1440, "display": "every 1440m"},
    }
    cases: list[dict[str, Any]] = []
    for line, value in parser_examples(source_root):
        if value not in expected:
            raise ValueError(f"parser example lacks a frozen expectation: {value!r}")
        cases.append({
            "id": f"test-jobs-{line}", "input": value, "origin": "tests/cron/test_jobs.py",
            "source_line": line, **expected[value],
        })
    extra = [
        ("six-field-cron", "0 0 9 * * *", {"kind": "cron", "expr": "0 0 9 * * *", "display": "0 0 9 * * *"}, "cron/jobs.py:813-819"),
        ("named-month-cron", "0 9 1 JAN *", {"kind": "cron", "expr": "0 9 1 JAN *", "display": "0 9 1 JAN *"}, "tests/cron/test_jobs.py:218-230"),
        ("named-weekday-cron", "0 9 * * MON", {"kind": "cron", "expr": "0 9 * * MON", "display": "0 9 * * MON"}, "tests/cron/test_jobs.py:218-230"),
        ("named-weekday-list", "0 9 * * MON,WED,FRI", {"kind": "cron", "expr": "0 9 * * MON,WED,FRI", "display": "0 9 * * MON,WED,FRI"}, "tests/cron/test_jobs.py:218-230"),
        ("named-weekday-range", "*/15 9-17 * * MON-FRI", {"kind": "cron", "expr": "*/15 9-17 * * MON-FRI", "display": "*/15 9-17 * * MON-FRI"}, "tests/cron/test_jobs.py:218-230"),
        ("date-only", "2030-01-15", {"kind": "once", "run_at_rule": "midnight in configured profile zone", "display": "once at 2030-01-15 00:00"}, "cron/jobs.py:820-837"),
        ("at-every-rejected", "@every 1h", {"error_contains": "Invalid schedule"}, "cron/jobs.py:813-864"),
        ("seconds-duration-rejected", "30s", {"error_contains": "Invalid schedule"}, "cron/jobs.py:678-690,854-864"),
        ("negative-duration-rejected", "-5m", {"error_contains": "Invalid schedule"}, "cron/jobs.py:678-690,854-864"),
        ("seven-field-cron-rejected", "0 0 9 * * * *", {"error_contains": "Invalid cron expression"}, "cron/jobs.py:813-819,854-864"),
        ("zero-interval-oracle-gap", "0m", {"kind": "interval", "minutes": 0, "display": "every 0m", "oracle_gap": "no positive-duration validation in parse_schedule"}, "cron/jobs.py:678-690,787-789"),
    ]
    for case_id, value, expectation, source in extra:
        cases.append({"id": case_id, "input": value, "origin": source.split(":")[0], "source_line": source, **expectation})
    _, jobs_tree = read_ast(source_root / "cron/jobs.py")
    weekdays = safe_eval(assignment(jobs_tree, "_WEEKDAY_TO_CRON_DOW"))
    dayspecs = safe_eval(assignment(jobs_tree, "_DAYSPEC_TO_CRON_DOW"))
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "cases": cases,
        "duration_units": {
            "m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1,
            "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
            "d": 1440, "day": 1440, "days": 1440,
        },
        "duration_count_optional": True,
        "duration_negative_cases": [
            {"input": value, "error_contains": "Invalid duration"}
            for value in ("abc", "hourx", "30x", "", "m30")
        ],
        "weekday_aliases": weekdays,
        "day_spec_aliases": dayspecs,
        "clock_forms": ["9am", "9:30am", "14:00", "7", "noon", "midnight", "12am", "12pm"],
        "timezone_policy": {
            "naive_one_shot": "configured profile clock at parse time; the per-job timezone does not reinterpret it",
            "aware_one_shot": "absolute instant; timezone pin does not move it",
            "relative_and_interval": "elapsed UTC duration, then rendered in the active zone",
            "cron": "IANA wall clock; both folds are tried and the earliest instant strictly after the base wins",
            "dst_gap": "no explicit advance-to-first-valid-minute policy in final head; LEO must define and test its safety policy",
            "dst_fold": "first occurrence for a new schedule; second occurrence only when strictly after a base in the second",
        },
        "semantic_equality": ["kind", "interval minutes or cron expression", "exact one-shot timestamp"],
        "source": {
            "parse_schedule": function_record(read_ast(source_root / "cron/jobs.py")[0], jobs_tree, "parse_schedule"),
            "normalize_repeat": function_record(read_ast(source_root / "cron/jobs.py")[0], jobs_tree, "normalize_repeat_value"),
        },
    }


def extract_delivery() -> dict[str, Any]:
    tokens = [
        {"token": "omitted", "oracle": True, "meaning": "origin when a capturable origin exists, otherwise local"},
        {"token": "origin", "oracle": True, "meaning": "captured platform/chat/thread, else first configured home channel, else nowhere"},
        {"token": "all", "oracle": True, "meaning": "every connected platform with a configured home channel at fire time; excludes bot-chat"},
        {"token": "local", "oracle": True, "meaning": "history only; no external target"},
        {"token": "bot-chat", "oracle": True, "meaning": "own source-project profile conversation"},
        {"token": "bot-chat:<profile>", "oracle": True, "meaning": "named machine-local profile conversation; missing profile rejects create"},
        {"token": "room:<id>", "oracle": True, "meaning": "machine-local hosted room; malformed or unresolved room produces no target"},
        {"token": "room:<id>@<profile>", "oracle": True, "meaning": "room hosted by a named local profile"},
        {"token": "<platform>:<chat>", "oracle": True, "meaning": "explicit chat; Telegram/Slack may adopt a durable origin thread"},
        {"token": "<platform>:<chat>:<thread>", "oracle": True, "meaning": "explicit chat and thread/topic"},
        {"token": "<platform>", "oracle": True, "meaning": "that platform's configured home channel, or same-platform origin fallback"},
        {"token": "comma list", "oracle": True, "meaning": "tokens resolve independently, order preserved, targets deduplicated by platform/chat/thread"},
        {"token": "list/tuple input", "oracle": True, "meaning": "normalized to a comma-separated string"},
        {"token": "failure_deliver", "oracle": True, "meaning": "separate failure lane using the same grammar; local suppresses external failure notices"},
        {"token": "home", "oracle": False, "meaning": "not a final-head token; treated as a bare platform name and normally resolves nowhere"},
        {"token": "local:<id>", "oracle": False, "meaning": "not a final-head token; planned LEO-native durable conversation mapping"},
        {"token": "object target", "oracle": False, "meaning": "not parsed by final head; planned LEO-native validated object form"},
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "tokens": tokens,
        "normalization": {
            "none_or_empty_create": "origin if origin exists, else local",
            "empty_runtime": "local",
            "case_sensitive": ["origin", "local"],
            "case_insensitive": ["all", "bot-chat", "room"],
            "unknown_token": "resolves nowhere; final head does not reject every typo at create",
        },
        "target_identity": ["platform.lower()", "chat_id", "thread_id"],
        "result_and_failure_lanes_separate": True,
        "leo_adaptation": {
            "home": "optional native alias for configured home; never claim source-project parity",
            "local:<id>": "durable local conversation by ID",
            "object target": "validated native target objects; reject unknown keys",
            "unknown_tokens": "reject before persistence instead of displaying a guessed recipient",
            "bot-chat": "durable local conversation, not byte-identical source-project profile behavior",
            "room": "requires a room-capable adapter and configured default room channel, otherwise reject",
        },
    }


def extract_repeat() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "normalization": [
            {"input": None, "stored_times": None, "meaning": "forever for recurring; one-shot creation overrides to 1"},
            {"input": "once", "stored_times": 1}, {"input": "one", "stored_times": 1}, {"input": "1x", "stored_times": 1},
            {"input": "forever", "stored_times": None}, {"input": "infinite", "stored_times": None},
            {"input": "inf", "stored_times": None}, {"input": "none", "stored_times": None}, {"input": "", "stored_times": None},
            {"input": "3", "stored_times": 3}, {"input": 3, "stored_times": 3},
            {"input": 0, "stored_times": None, "oracle_gap": "non-positive normalizes to forever"},
            {"input": -2, "stored_times": None, "oracle_gap": "non-positive normalizes to forever"},
            {"input": True, "stored_times": 1, "oracle_gap": "bool is accepted as an integer"},
            {"input": False, "stored_times": None, "oracle_gap": "bool is accepted as an integer"},
            {"input": 3.9, "stored_times": 3, "oracle_gap": "fractional numbers are truncated"},
            {"input": "banana", "error": "Invalid repeat value"},
        ],
        "declared_schema_type": "integer",
        "runtime_accepts_numeric_strings": True,
        "one_shot_default": 1,
        "completed_counter_preserved_on_update": True,
        "manual_runs_do_not_consume_repeat_or_cadence": True,
        "leo_hardening": "reject zero/negative/unknown values instead of converting them to forever",
    }


def extract_consent(source_root: Path) -> dict[str, Any]:
    source, tree = read_ast(source_root / "tools/cronjob_tools.py")
    en = source_root / "locales/en.yaml"
    gateway = yaml_section(en, ("gateway", "exec_approval"))
    cli = yaml_section(en, ("approval",))
    tui = yaml_section(en, ("cli", "tui"))
    cli_approval = yaml_section(en, ("cli", "approval"))
    gateway = {
        key: neutralize(value)
        for key, value in gateway.items()
        if key.startswith("schedule_") or key in {
            "action_deny", "text_choice_deny", "text_choice_joiner", "text_choice_last",
        }
    }
    cli = {key: neutralize(value) for key, value in cli.items()}
    nowhere = "nowhere (saved to the job's history only)"
    cases = [
        {
            "case_id": "create-recurring-three-runs", "kind": "schedule",
            "job": {"name": "Water", "schedule_display": "every day at 8am", "next_run_at": "2026-11-25T08:00:00+00:00", "repeat": {"times": 3}, "prompt": "Remind the user to drink a glass of water.", "deliver": "local"},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": "2026-11-25T08:00:00+00:00",
            "body": f'Schedule "Water" (every day at 8am, first run Wed Nov 25 08:00 UTC, 3 runs): Remind the user to drink a glass of water.; results go to {nowhere}',
        },
        {
            "case_id": "create-script-only", "kind": "schedule",
            "job": {"name": "Check", "schedule_display": "every 2h", "next_run_at": "2026-11-25T09:00:00+00:00", "repeat": {"times": None}, "prompt": "", "script": "check.py", "no_agent": True, "deliver": "local"},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": "2026-11-25T09:00:00+00:00",
            "body": f'Schedule "Check" (every 2h, first run Wed Nov 25 09:00 UTC): runs script check.py; results go to {nowhere}',
        },
        {
            "case_id": "create-paused", "kind": "schedule",
            "job": {"name": "Water", "schedule_display": "every day at 8am", "next_run_at": None, "repeat": {"times": 3}, "prompt": "Remind the user to drink a glass of water.", "deliver": "local", "enabled": False},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": None,
            "body": f'Schedule "Water" (every day at 8am, created paused: it will not run until resumed, 3 runs): Remind the user to drink a glass of water.; results go to {nowhere}',
        },
        {
            "case_id": "create-one-shot", "kind": "schedule",
            "job": {"name": "Dentist", "schedule_display": "once at 2026-11-27 09:00", "schedule": {"kind": "once", "run_at": "2026-11-27T09:00:00+00:00"}, "next_run_at": "2026-11-27T09:00:00+00:00", "repeat": {"times": 1}, "prompt": "Call the dentist.", "deliver": "telegram:12345"},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": "2026-11-27T09:00:00+00:00",
            "body": 'Schedule "Dentist" (once, Fri Nov 27 09:00 UTC): Call the dentist.; results go to telegram:12345',
        },
        {
            "case_id": "create-execution-extras", "kind": "schedule",
            "job": {"name": "Brief", "schedule_display": "every day at 8am", "next_run_at": "2026-11-25T08:00:00+00:00", "repeat": {"times": None}, "prompt": "Summarize updates.", "skills": ["maps"], "script": "collect.py", "monitor_url": "https://example.com/price", "workdir": "/workspace/brief", "context_from": ["source-job"], "enabled_toolsets": [], "attach_to_session": True, "deliver": "local", "failure_deliver": "telegram"},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": "2026-11-25T08:00:00+00:00",
            "body": f'Schedule "Brief" (every day at 8am, first run Wed Nov 25 08:00 UTC): Summarize updates. (skills: maps) (after running script collect.py); checks URL https://example.com/price first each run; runs in /workspace/brief; reads output of source-job; toolsets: none; copied into the chat\'s history; results go to {nowhere}; failure notices go to nowhere (telegram has no destination configured; saved to the job\'s history only)',
        },
        {
            "case_id": "edit-prompt", "kind": "schedule_edit",
            "job": {"name": "Water", "prompt": "Remind the user to drink a glass of water."},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": None,
            "body": 'Change "Water": prompt: Remind the user to drink a glass of water. -> Drink tea instead.',
        },
        {
            "case_id": "edit-execution-mode", "kind": "schedule_edit",
            "job": {"name": "Water", "no_agent": False, "script": None},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": None,
            "body": 'Change "Water": what runs: agent task -> script only; script: (none) -> check.py',
        },
        {
            "case_id": "edit-delivery", "kind": "schedule_edit",
            "job": {"name": "Water", "deliver": "local"},
            "shown_at": "2026-11-25T07:00:00+00:00", "shown_first_run": None,
            "body": f'Change "Water": delivers to: local ({nowhere}) -> telegram (nowhere (telegram has no destination configured; saved to the job\'s history only))',
        },
    ]
    choice_ids = {
        "schedule": ["once", "session", "always", "deny"],
        "schedule_edit": ["once", "deny"],
    }
    for case in cases:
        kind = case["kind"]
        case.update({
            "header": gateway["schedule_header" if kind == "schedule" else "schedule_edit_header"],
            "choices": choice_ids[kind],
            "choice_labels": [gateway["schedule_action_once"], tui["approval_deny"]] if kind == "schedule_edit" else [
                gateway["schedule_action_once"], gateway["schedule_action_session"], gateway["schedule_action_always"], tui["approval_deny"],
            ],
            "deadline": gateway["schedule_deadline"].format(window="5 minutes") if kind == "schedule" else gateway["schedule_edit_deadline"].format(window="5 minutes"),
            "timeout_notice": gateway["schedule_timed_out_notice"].format(window="5 minutes") if kind == "schedule" else gateway["schedule_edit_timed_out_notice"].format(window="5 minutes"),
            "cli_header": cli["schedule_header" if kind == "schedule" else "schedule_edit_header"].format(description=case["body"]),
            "cli_menu": cli["choose_long" if kind == "schedule" else "choose_short"],
            "source": "tools/cronjob_tools.py:_approval_task/_approval_when/_approve_new_job/_approve_job_edit; locales/en.yaml",
        })
    card_job_fields: set[str] = set()
    for name in ("_approval_when", "_approval_task", "_approve_new_job", "_approve_job_edit", "_schedule_shown"):
        for node in ast.walk(function(tree, name)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    card_job_fields.add(node.args[0].value)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "job":
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    card_job_fields.add(node.slice.value)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "key" and isinstance(node.iter, ast.Tuple):
                for pair in node.iter.elts:
                    if isinstance(pair, ast.Tuple) and pair.elts and isinstance(pair.elts[0], ast.Constant):
                        card_job_fields.add(str(pair.elts[0].value))
    for field_record in safe_eval(assignment(tree, "_EDIT_CARD_FIELDS")):
        card_job_fields.add(field_record[0])
    # Legacy stored monitor aliases are rendered by the card even though the
    # model-facing field is the canonical ``monitor`` argument.
    card_job_fields.update({"monitor_script", "monitor_url"})
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "card_allowed_keys": ["case_id", "kind", "job", "shown_at", "shown_first_run", "body", "header", "choices", "choice_labels", "deadline", "timeout_notice", "cli_header", "cli_menu", "source"],
        "card_job_allowed_fields": sorted(card_job_fields),
        "card_repeat_allowed_keys": ["times", "completed"],
        "cases": cases,
        "copy": {
            "gateway": gateway,
            "cli": {key: cli[key] for key in ("schedule_header", "schedule_edit_header", "choose_long", "choose_short", "prompt_long", "prompt_short", "allowed_once", "allowed_session", "allowed_always", "denied", "cancelled", "timeout")},
            "tui": {key: tui[key] for key in ("approval_title_schedule", "approval_title_schedule_edit", "approval_schedule_once", "approval_schedule_session", "approval_schedule_always", "approval_deny", "approval_schedule_truncated")},
            "classic_outcomes": {key: cli_approval[key] for key in ("schedule_label", "schedule_timeout", "schedule_edit_timeout", "schedule_outcome_deny", "schedule_edit_outcome_deny")},
        },
        "decline_messages": {
            "denied": "The user declined this on the schedule card; nothing was saved. Tell them in one line what you wanted to schedule, and do not retry it.",
            "timeout": "The user did not answer the schedule card in time; nothing was saved. Do not retry it.",
            "other": "The schedule card was not approved; nothing was saved. Do not retry it.",
            "stale_create": "The schedule card was answered after the first run it showed had passed; nothing was saved. Offer the job again with a later time if it is still wanted.",
            "stale_edit": "The change card was answered after the new run time it showed had passed; nothing was saved. Offer the change again with a later time if it is still wanted.",
        },
        "source_functions": {
            name: function_record(source, tree, name)
            for name in ("_approval_when", "_approval_task", "_audience", "_ask_schedule_card", "_approve_new_job", "_edit_fingerprint", "_approve_job_edit")
        },
        "leo_divergence": "No answerable surface means durable approval_required with no schedule write and no approval row; operator resolution revalidates and commits.",
    }


def parse_scenarios(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"^- (\{.*?^\})", text, flags=re.MULTILINE | re.DOTALL)
    return [json.loads(block) for block in blocks]


def extract_acceptance(analysis_dir: Path) -> dict[str, Any]:
    scenarios = parse_scenarios(analysis_dir / "acceptance-matrix.md")
    by_id = {scenario["id"]: scenario for scenario in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError("duplicate acceptance scenario id")
    deferred_ids = {
        "PARITY-ELIG-02", "PARITY-ELIG-04", "PARITY-ELIG-05", "PARITY-ELIG-06",
        "PARITY-PROMPT-03", "PARITY-QUALITY-01", "HARD-QUAL-04", "HARD-QUAL-05", "HARD-QUAL-06",
    }
    adaptation_ids = {
        "PARITY-ELIG-03", "PARITY-EDIT-04", "PARITY-DELIV-05", "PARITY-RENDER-10",
        "PARITY-SURF-01", "PARITY-SURF-02", "PARITY-SURF-03", "HARD-QUAL-10",
    }
    hardening_ids = {"PARITY-QUALITY-02", "PARITY-VERIFY-01", "PARITY-VERIFY-02"}
    sections: dict[str, list[dict[str, Any]]] = {
        "Final-Head Parity": [], "LEO Adaptations": [], "LEO Hardening": [], "Deferred": [],
    }

    def add(section: str, scenario: dict[str, Any], **updates: Any) -> None:
        item = dict(scenario)
        item.update(updates)
        item["core_gate"] = section != "Deferred"
        sections[section].append(item)

    for scenario in scenarios:
        scenario_id = scenario["id"]
        item = dict(scenario)
        if scenario_id == "PARITY-CREATE-09":
            item["then"] = "Concurrent valid creates have distinct proposal/card identities and transactional name reservation; one answer cannot resolve both."
        if scenario_id == "HARD-SEC-04":
            item["then"] = "The real shown boolean controls staleness; no-card model edits fail closed to approval_required, while trusted operator edits re-read and CAS under the write transaction."
        if scenario_id in deferred_ids:
            add("Deferred", item, disposition="removed from the core gate or explicitly deferred by the revised plan")
        elif scenario_id in adaptation_ids:
            if scenario_id == "PARITY-SURF-01":
                item["then"] = "Only TUI, web/Electron stream and Telegram with native callbacks may receive cards; every other adapter remains incapable and produces approval_required."
                item["surfaces"] = ["TUI", "web", "Electron", "Telegram", "all other LEO adapters"]
            elif scenario_id == "PARITY-SURF-03":
                item["then"] = "Typed decisions work only on a planned answerable surface; send-only, notifier-blind and unverified adapters never receive a card."
            elif scenario_id == "PARITY-RENDER-10":
                item["then"] = "Centralized English copy plus DTO, event, API and UI contract tests replace source-project locale assumptions; LEO has no locale subsystem."
            elif scenario_id == "HARD-QUAL-10":
                item["then"] = "Native URL/script monitors preserve a good baseline on failure, suppress unchanged output and wake on exact-byte change; export comparison remains deferred."
            add("LEO Adaptations", item)
        elif scenario_id.startswith("HARD-") or scenario_id in hardening_ids:
            add("LEO Hardening", item)
        else:
            add("Final-Head Parity", item)

    native = [
        ("LEO-ADAPT-01", "Native durable authority", "Schedules, proposals, tombstones, approvals and grants use LEO's existing SQLite store; no parallel schedule or approval authority exists."),
        ("LEO-ADAPT-02", "Native tool roster", "LEO exposes schedule_create/list/get/update/delete/run_now/cancel; upstream has no get action, so schedule_get is a documented native addition."),
        ("LEO-ADAPT-03", "Operator-provisioned scripts", "Scheduled scripts are opt-in executable paths under an allowlisted root, with args arrays, no shell, minimal environment, timeout/byte caps and real run evidence."),
        ("LEO-ADAPT-04", "Native monitors", "URL monitors use the existing public-address egress client and script monitors use the restricted runner; authenticated headers, private targets and browser rendering are rejected."),
        ("LEO-ADAPT-05", "Native local destinations", "bot-chat/profile map to durable local conversations; local:<id> and validated objects are native extensions; room requires a capable adapter and configured default channel."),
        ("LEO-ADAPT-06", "Honest unsupported fields", "Every surviving field has a runtime consumer or an explicit rejection; unknown fields and removed intermediate scope never persist."),
        ("LEO-ADAPT-07", "Surface capability default", "Unknown, send-only, notifier-blind and ambiguous surfaces default incapable; only verified full-text, inbound-identity, terminal-rewrite surfaces receive cards."),
        ("LEO-ADAPT-08", "LEO grammar compatibility", "Existing @every schedules remain readable; any new @every compatibility is labeled LEO-native because final head rejects it."),
    ]
    for scenario_id, area, then in native:
        add("LEO Adaptations", {"id": scenario_id, "area": area, "priority": "P0 - LEO adaptation", "given": "The native port is exercised.", "when": "The corresponding LEO contract tests run.", "then": then, "surfaces": ["LEO native implementation"]})
    add("LEO Hardening", {"id": "LEO-HARD-REPEAT-CAP", "area": "Hardening / bounded recurrence", "priority": "P1 - LEO hardening", "given": "automation.max_repeat_runs is configured.", "when": "A model creates a recurring schedule.", "then": "The cap applies only to model-created schedules, is disclosed on the card, and omitted repeat remains forever when no cap is configured.", "surfaces": ["compiler", "card", "scheduler"]})
    add("LEO Hardening", {"id": "LEO-HARD-PROPOSAL-MODE", "area": "Hardening / proposal mode", "priority": "P1 - LEO hardening", "given": "automation.proactive_mode is off or offer.", "when": "A turn evaluates scheduling guidance.", "then": "Off disables guidance only; offer uses the normal card path; neither mode bypasses consent or create grants.", "surfaces": ["prompt", "tool", "policy"]})

    plan_text = (analysis_dir / "revised-plan.md").read_text(encoding="utf-8")
    deferred_section = plan_text.split("## deferred\n", 1)[1].split("## phases\n", 1)[0]
    for index, line in enumerate((line for line in deferred_section.splitlines() if line.startswith("- ")), start=1):
        add("Deferred", {"id": f"DEFERRED-PLAN-{index:02d}", "area": "Deferred / revised plan", "priority": "Deferred", "given": "A deferred capability is requested.", "when": "Release scope is evaluated.", "then": line[2:], "surfaces": ["not shipped"]})
    add("Deferred", {"id": "DEFERRED-REGISTRATION", "area": "Deferred / dead registration result", "priority": "Deferred", "given": "A durable schedule row is the scheduler input.", "when": "Registration result fields are requested.", "then": "No scheduler_registered result field or external registration hook is shipped; LEO's durable row is already the scheduler input.", "surfaces": ["tool result"]})
    add("Deferred", {"id": "DEFERRED-EXPORT-COMPARISON", "area": "Deferred / monitor export comparison", "priority": "Deferred", "given": "A future monitor compares exports.", "when": "Export comparison is requested.", "then": "Read-only export comparison is follow-up scope; native monitor baseline/change/failure semantics remain in the core gate.", "surfaces": ["not shipped"]})

    all_ids = [item["id"] for group in sections.values() for item in group]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("acceptance sections contain duplicate ids")
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "section_order": list(sections),
        "sections": sections,
        "core_gate_sections": ["Final-Head Parity", "LEO Adaptations", "LEO Hardening"],
        "deferred_is_not_core_gate": True,
        "removed_from_core_gate": sorted(deferred_ids),
        "source_scenario_count": len(scenarios),
    }


def build_fixtures(source_root: Path, analysis_dir: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise ValueError(f"oracle source directory is missing: {source_root}")
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != ORACLE_COMMIT:
        raise ValueError(f"oracle commit mismatch: {commit}")
    status = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError("oracle source checkout must be clean")
    tool_schema = extract_tool_schema(source_root)
    actions = extract_actions(source_root, tool_schema)
    fixtures: dict[str, Any] = {
        "tool_schema.json": tool_schema,
        "fields.json": extract_fields(source_root, tool_schema),
        "grammar.json": extract_grammar(source_root),
        "delivery_tokens.json": extract_delivery(),
        "repeat.json": extract_repeat(),
        "consent_cards.json": extract_consent(source_root),
        "acceptance_matrix.json": extract_acceptance(analysis_dir),
    }
    for action, fixture in actions.items():
        fixtures[f"actions/{action}.json"] = fixture
    return fixtures


def write_fixtures(output_dir: Path, fixtures: dict[str, Any], source_root: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative, payload in sorted(fixtures.items()):
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sources = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        sources[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "oracle_commit": ORACLE_COMMIT,
        "oracle_role": "read-only behavioral source; never executed",
        "source_checkout_clean": True,
        "source_files": sources,
        "fixture_files": {relative: sha256_file(output_dir / relative) for relative in sorted(fixtures)},
        "fixture_count": len(fixtures),
        "brand_neutralization": "source-project names are neutralized in public text; source digests bind exact upstream bytes",
    }
    (output_dir / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "tests/fixtures/schedule_parity")
    parser.add_argument("--check", action="store_true", help="compare generated bytes without writing")
    args = parser.parse_args()
    fixtures = build_fixtures(args.oracle_source.resolve(), args.analysis_dir.resolve())
    if args.check:
        mismatches = []
        for relative, payload in sorted(fixtures.items()):
            target = args.output_dir / relative
            expected = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            if not target.exists() or target.read_text(encoding="utf-8") != expected:
                mismatches.append(relative)
        if mismatches:
            print("fixture mismatch: " + ", ".join(mismatches))
            return 1
        print(f"{len(fixtures)} fixtures match the frozen oracle")
        return 0
    manifest = write_fixtures(args.output_dir.resolve(), fixtures, args.oracle_source.resolve())
    print(f"wrote {manifest['fixture_count']} fixtures at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
