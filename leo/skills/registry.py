"""Native skill registry, lifecycle and safe execution.

Skills are declarative ``SkillManifest`` records plus an optional, locally
implemented handler. External manifests are translated by
``leo.skills.compatibility``; this module never executes code supplied by a
candidate, shell command string or dynamic import. All durable state lives in
the shared ``skills`` / ``skill_history`` tables and all lifecycle changes are
published as ``skill.*`` events.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

from pydantic import BaseModel, Field

from .. import db
from ..contracts import RiskLevel, SkillManifest, utcnow
from ..events import EventBus
from ..logging import get_logger
from ..security.permissions import ApprovalRequired
from .gates import bounded_preview, evaluate_gates, execution_block

_LOG = get_logger("skills")

Handler = Callable[[dict[str, Any], dict[str, Any] | None], Any]

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_VERSION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?$")
_DEP_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_PERM_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_DANGEROUS_PERMISSION_RE = re.compile(
    r"(?:^|[.:_-])(?:eval|exec|shell|subprocess|dynamic[_-]?import|code[_-]?exec|"
    r"arbitrary[_-]?code|os[_-]?system)(?:$|[.:_-])",
    re.IGNORECASE,
)
_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null", "any"}
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "but", "not", "into", "has", "have", "had", "its", "our", "you", "your",
    "they", "their", "been", "will", "would", "can", "could", "about", "than",
}


def _await_self(value: Any) -> Awaitable[Any]:
    async def _identity() -> Any:
        return value

    return _identity()


class ValidationResult(BaseModel):
    """Result of :meth:`SkillRegistry.validate` (also usable with ``await``)."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.valid

    def __await__(self):
        return _await_self(self).__await__()


class SkillRecord(SkillManifest):
    """A manifest with its registry lifecycle metadata."""

    status: str = "registered"
    health: str = "available"
    enabled: bool = True
    registered_at: str = Field(default_factory=utcnow)

    @property
    def has_handler(self) -> bool:
        # Filled in by the registry when a record is returned.
        return bool(self.model_extra and self.model_extra.get("has_handler", False))

    def __await__(self):
        return _await_self(self).__await__()


class HealthReport(BaseModel):
    name: str
    healthy: bool
    status: str
    detail: str
    checked_at: str = Field(default_factory=utcnow)

    def __await__(self):
        return _await_self(self).__await__()


class SkillRunResult(BaseModel):
    skill: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    validation_errors: list[str] = Field(default_factory=list)
    decision: str = "allowed"
    approval_id: str | None = None

    def __await__(self):
        return _await_self(self).__await__()


class SkillList(list):
    """A plain list that can also be awaited for API symmetry."""

    def __await__(self):
        return _await_self(self).__await__()


# ---------------------------------------------------------------------------
# Schema validation (small JSON Schema subset, no external dependency)
# ---------------------------------------------------------------------------

def _validate_schema_shape(schema: Any, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        errors.append(f"{path} must be an object")
        return
    typ = schema.get("type")
    if typ is not None:
        types = typ if isinstance(typ, list) else [typ]
        if not isinstance(types, list) or not all(t in _JSON_TYPES for t in types):
            errors.append(f"{path}.type contains an unsupported JSON Schema type")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict) or not all(isinstance(k, str) for k in properties):
            errors.append(f"{path}.properties must be an object with string keys")
        else:
            for key, value in properties.items():
                _validate_schema_shape(value, f"{path}.properties.{key}", errors)
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            errors.append(f"{path}.required must be a list of strings")
        elif isinstance(properties, dict):
            for name in required:
                if name not in properties:
                    errors.append(f"{path}.required references unknown property '{name}'")
    if "items" in schema:
        _validate_schema_shape(schema["items"], f"{path}.items", errors)

    # Constraint keywords are data too. Malformed constraints used to pass
    # manifest validation and then raise re.error/TypeError from _validate_value
    # during run(), turning a bad manifest into an unhandled exception.
    enum = schema.get("enum")
    if enum is not None and not isinstance(enum, list):
        errors.append(f"{path}.enum must be a list")
    for keyword in ("minimum", "maximum"):
        if keyword in schema:
            value = schema[keyword]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{path}.{keyword} must be a number")
    minimum, maximum = schema.get("minimum"), schema.get("maximum")
    if (
        isinstance(minimum, (int, float)) and not isinstance(minimum, bool)
        and isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
        and minimum > maximum
    ):
        errors.append(f"{path}.minimum must not exceed maximum")
    for keyword in ("minLength", "maxLength"):
        if keyword in schema:
            value = schema[keyword]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{path}.{keyword} must be a non-negative integer")
    min_len, max_len = schema.get("minLength"), schema.get("maxLength")
    if (
        isinstance(min_len, int) and not isinstance(min_len, bool)
        and isinstance(max_len, int) and not isinstance(max_len, bool)
        and min_len > max_len
    ):
        errors.append(f"{path}.minLength must not exceed maxLength")
    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            errors.append(f"{path}.pattern must be a string")
        else:
            try:
                re.compile(pattern)
            except re.error:
                errors.append(f"{path}.pattern is not a valid regular expression")
    if "additionalProperties" in schema:
        extra = schema["additionalProperties"]
        if not isinstance(extra, bool):
            _validate_schema_shape(extra, f"{path}.additionalProperties", errors)


def validate_manifest(manifest: SkillManifest | dict[str, Any]) -> ValidationResult:
    """Validate a manifest's name, version, schemas, permissions and dependencies."""
    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(manifest, dict):
        try:
            manifest = SkillManifest.model_validate(manifest)
        except Exception as exc:  # pydantic validation error
            return ValidationResult(valid=False, errors=[f"invalid manifest structure: {exc}"])
    if not isinstance(manifest, SkillManifest):
        return ValidationResult(valid=False, errors=["manifest must be a SkillManifest or mapping"])

    if not isinstance(manifest.name, str) or not _NAME_RE.match(manifest.name):
        errors.append("name must be 1-64 lowercase letters, digits, '_' or '-'")
    if not isinstance(manifest.version, str) or not _VERSION_RE.match(manifest.version):
        errors.append("version must look like 1.2.3 (optional pre-release/build suffix)")
    if not manifest.description or not isinstance(manifest.description, str):
        errors.append("description must be a non-empty string")
    if not isinstance(manifest.category, str) or not manifest.category.strip():
        errors.append("category must be a non-empty string")

    for field_name in ("input_schema", "output_schema"):
        schema = getattr(manifest, field_name)
        if not isinstance(schema, dict):
            errors.append(f"{field_name} must be an object")
            continue
        if field_name == "input_schema" and schema and schema.get("type", "object") != "object":
            errors.append("input_schema.type must be 'object' when specified")
        _validate_schema_shape(schema, field_name, errors)

    if not isinstance(manifest.permissions, list):
        errors.append("permissions must be a list")
    else:
        seen: set[str] = set()
        for perm in manifest.permissions:
            if not isinstance(perm, str) or not _PERM_RE.match(perm):
                errors.append(f"invalid permission token: {perm!r}")
            elif perm in seen:
                errors.append(f"duplicate permission: {perm}")
            elif _DANGEROUS_PERMISSION_RE.search(perm):
                errors.append(f"unsafe permission is not allowed: {perm}")
            else:
                seen.add(perm)

    if not isinstance(manifest.dependencies, list):
        errors.append("dependencies must be a list")
    else:
        seen_deps: set[str] = set()
        for dep in manifest.dependencies:
            if not isinstance(dep, str) or not _DEP_RE.match(dep):
                errors.append(f"invalid dependency: {dep!r}")
            elif dep == manifest.name:
                errors.append("skill cannot depend on itself")
            elif dep in seen_deps:
                errors.append(f"duplicate dependency: {dep}")
            else:
                seen_deps.add(dep)

    if not manifest.documentation:
        warnings.append("manifest has no documentation")
    if not manifest.tests:
        warnings.append("manifest lists no tests")
    if manifest.source != "native":
        warnings.append("non-native source: execution capability must be verified independently")
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _validate_value(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    typ = schema.get("type")
    if typ is not None and typ != "any":
        types = typ if isinstance(typ, list) else [typ]
        def matches(name: str) -> bool:
            if name == "object":
                return isinstance(value, dict)
            if name == "array":
                return isinstance(value, list)
            if name == "string":
                return isinstance(value, str)
            if name == "boolean":
                return isinstance(value, bool)
            if name == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if name == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if name == "null":
                return value is None
            return True
        if not any(matches(t) for t in types):
            errors.append(f"{path}: expected {'|'.join(types)}, got {type(value).__name__}")
            return
    # The same keyword types are checked by _validate_schema_shape; these
    # guards keep a legacy/persisted malformed schema from raising here.
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value not in enum")
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    minimum = schema.get("minimum")
    if numeric and isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        if value < minimum:
            errors.append(f"{path}: below minimum {minimum}")
    maximum = schema.get("maximum")
    if numeric and isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        if value > maximum:
            errors.append(f"{path}: above maximum {maximum}")
    if isinstance(value, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and not isinstance(min_len, bool) and len(value) < min_len:
            errors.append(f"{path}: shorter than minLength {min_len}")
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and not isinstance(max_len, bool) and len(value) > max_len:
            errors.append(f"{path}: longer than maxLength {max_len}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if not re.search(pattern, value):
                    errors.append(f"{path}: does not match pattern")
            except re.error:
                errors.append(f"{path}: invalid pattern in schema")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{path}[{index}]", errors)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: required")
        for key, item in value.items():
            if key in properties:
                _validate_value(item, properties[key], f"{path}.{key}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property not allowed")


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema:
        _validate_value(arguments, schema, "arguments", errors)
    return errors


def _risk_for(manifest: SkillManifest) -> RiskLevel:
    # External provenance is itself risk: a translated manifest is a claim made
    # by untrusted data, regardless of how harmless its permission list looks.
    if manifest.source != "native":
        return RiskLevel.HIGH
    text = " ".join(manifest.permissions).lower()
    if any(token in text for token in ("write", "delete", "network", "net:", "fs:", "exec")):
        return RiskLevel.MEDIUM
    gate_text = json.dumps(manifest.gates, default=str).lower()
    if any(token in gate_text for token in ("env", "config", "network", "credential")):
        return RiskLevel.HIGH
    if manifest.gates:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _accepts_context(handler: Callable[..., Any]) -> bool:
    try:
        params = list(inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        return False
    return any(p.kind is p.VAR_POSITIONAL or p.kind is p.VAR_KEYWORD for p in params) or len(
        [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    ) >= 2


# ---------------------------------------------------------------------------
# Native skill handlers -- pure Python or fixed-argument local tools only
# ---------------------------------------------------------------------------

async def _summarize_text(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    text = arguments["text"]
    limit = int(arguments.get("max_sentences", 3))
    if limit < 1:
        raise ValueError("max_sentences must be >= 1")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return {"summary": "", "sentence_count": 0, "source_length": len(text)}
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    counts = Counter(w for w in words if len(w) > 3 and w not in _STOPWORDS)
    ranked = []
    for index, sentence in enumerate(sentences):
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", sentence.lower())
        score = sum(counts.get(t, 0) for t in tokens) / (len(tokens) ** 0.5 or 1.0)
        ranked.append((score, -index, sentence))
    ranked.sort(reverse=True)
    chosen = [item[2] for item in ranked[:limit]]
    return {
        "summary": " ".join(chosen),
        "sentence_count": len(sentences),
        "source_length": len(text),
    }


async def _analyze_code(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    import ast

    code = arguments["code"]
    tree = ast.parse(code)
    functions = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    import_names = sorted(imports)
    max_depth = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            nested = depth + 1 if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)) else depth
            visit(child, nested)

    visit(tree, 0)
    return {
        "lines": len(code.splitlines()),
        "functions": functions,
        "classes": classes,
        "imports": import_names,
        "max_block_depth": max_depth,
        "syntax": "valid",
    }


def _resolve_local_root(arguments: dict[str, Any], context: dict[str, Any] | None) -> Path:
    roots = (context or {}).get("allowed_roots") or [Path.cwd()]
    allowed = [Path(r).expanduser().resolve() for r in roots]
    candidate_value = arguments.get("path", ".")
    candidate = Path(candidate_value).expanduser()
    if not candidate.is_absolute():
        candidate = allowed[0] / candidate
    resolved = candidate.resolve()
    if not any(_is_within(resolved, root) for root in allowed):
        raise ValueError("path is outside the allowed workspace roots")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


async def _file_inventory(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    root = _resolve_local_root(arguments, context)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    limit = min(int(arguments.get("max_entries", 100)), 1000)
    if limit < 1:
        raise ValueError("max_entries must be >= 1")
    entries: list[dict[str, Any]] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if len(entries) >= limit:
                truncated = True
                break
            path = Path(dirpath) / filename
            try:
                if not _is_within(path.resolve(), root):
                    continue
                entries.append({
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "type": "symlink" if path.is_symlink() else "file",
                })
            except OSError:
                continue
        if truncated:
            break
    return {"root": str(root), "entries": entries, "count": len(entries), "truncated": truncated}


async def _git_status(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    root = _resolve_local_root(arguments, context)
    # Fixed argv: no shell, no user-controlled flags, bounded runtime.
    result = subprocess.run(
        ["git", "status", "--porcelain", "-b"],
        cwd=root, capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"git status failed: {result.stderr.strip() or 'not a git repository'}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    branch = lines[0][3:] if lines and lines[0].startswith("## ") else None
    changes = [{"status": line[:2].strip(), "path": line[3:]} for line in lines if not line.startswith("## ")]
    return {"root": str(root), "branch": branch, "changes": changes, "count": len(changes)}


def _get_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


async def _json_transform(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    data = arguments["data"]
    operation = arguments.get("operation", "keys")
    if operation == "keys":
        result: Any = sorted(data.keys()) if isinstance(data, dict) else []
    elif operation == "get":
        result = _get_path(data, arguments["path"])
    elif operation == "select":
        keys = arguments.get("keys", [])
        if not isinstance(data, dict):
            raise ValueError("select requires an object")
        result = {key: data[key] for key in keys}
    elif operation == "length":
        result = len(data)
    elif operation == "type":
        result = type(data).__name__
    else:
        raise ValueError(f"unsupported operation: {operation}")
    return {"operation": operation, "result": result}


async def _report_generator(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    title = arguments.get("title", "Report")
    sections = arguments.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("sections must be a list")
    fmt = arguments.get("format", "markdown")
    if fmt not in ("markdown", "text"):
        raise ValueError("format must be 'markdown' or 'text'")
    lines = [f"# {title}" if fmt == "markdown" else title, ""]
    for section in sections:
        heading = section.get("heading", "Section") if isinstance(section, dict) else str(section)
        body = section.get("body", "") if isinstance(section, dict) else ""
        findings = section.get("findings", []) if isinstance(section, dict) else []
        lines.append(f"## {heading}" if fmt == "markdown" else heading)
        if body:
            lines.append(str(body))
        for finding in findings:
            lines.append(f"- {finding}" if fmt == "markdown" else f"* {finding}")
        lines.append("")
    return {"report": "\n".join(lines).rstrip() + "\n", "format": fmt, "section_count": len(sections)}


async def _system_health(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "ok",
        "platform": platform.system(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "note": "local process information only; no network probes or external claims",
    }


def _manifest(
    name: str,
    description: str,
    *,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
    documentation: str,
    permissions: list[str] | None = None,
    category: str = "general",
    tests: list[str] | None = None,
    version: str = "1.0.0",
) -> SkillManifest:
    return SkillManifest(
        name=name,
        description=description,
        version=version,
        author="leo-native",
        category=category,
        permissions=permissions or [],
        input_schema=input_schema,
        output_schema=output_schema,
        source="native",
        documentation=documentation,
        tests=tests or ["tests/test_skills.py"],
    )


def _build_native_skills() -> dict[str, tuple[SkillManifest, Handler | None]]:
    """Definitions for the built-in capability library.

    A ``None`` handler is deliberate: the manifest is registered and documented,
    but LEO makes no claim that an external service is connected or verified.
    """
    return {
        "summarize_text": (
            _manifest(
                "summarize_text",
                "Extractive summary of a text using local word-frequency scoring.",
                category="research",
                input_schema={"type": "object", "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "max_sentences": {"type": "integer", "minimum": 1, "maximum": 20},
                }, "required": ["text"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {
                    "summary": {"type": "string"}, "sentence_count": {"type": "integer"},
                }, "required": ["summary"]},
                documentation="Pure Python. Does not call a model or network service.",
            ),
            _summarize_text,
        ),
        "web_research_request": (
            _manifest(
                "web_research_request",
                "Describe a web research request for a configured research provider.",
                category="research",
                permissions=["net:read"],
                input_schema={"type": "object", "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "max_sources": {"type": "integer", "minimum": 1, "maximum": 20},
                }, "required": ["query"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"request": {"type": "object"}}},
                documentation=(
                    "Declarative request only. No browser, search provider or network adapter is "
                    "bundled or verified; running it without a handler is not supported."
                ),
            ),
            None,
        ),
        "git_status": (
            _manifest(
                "git_status",
                "Read git working-tree status from a local repository.",
                category="development",
                permissions=["git:read"],
                input_schema={"type": "object", "properties": {
                    "path": {"type": "string"},
                }, "additionalProperties": False},
                output_schema={"type": "object", "properties": {
                    "branch": {"type": "string"}, "changes": {"type": "array"},
                }},
                documentation=(
                    "Runs the fixed command `git status --porcelain -b` with no shell. "
                    "Paths must resolve inside an allowed workspace root."
                ),
            ),
            _git_status,
        ),
        "analyze_code": (
            _manifest(
                "analyze_code",
                "Static structural analysis of Python source (functions, classes, imports, depth).",
                category="development",
                input_schema={"type": "object", "properties": {
                    "code": {"type": "string", "minLength": 1},
                }, "required": ["code"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {
                    "functions": {"type": "integer"}, "classes": {"type": "integer"},
                    "imports": {"type": "array"}, "syntax": {"type": "string"},
                }, "required": ["functions", "classes"]},
                documentation="Uses Python's ast module. Never imports or executes the supplied code.",
            ),
            _analyze_code,
        ),
        "file_inventory": (
            _manifest(
                "file_inventory",
                "List files and sizes under a workspace-local directory.",
                category="computer",
                permissions=["fs:read"],
                input_schema={"type": "object", "properties": {
                    "path": {"type": "string"},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
                }, "additionalProperties": False},
                output_schema={"type": "object", "properties": {
                    "entries": {"type": "array"}, "count": {"type": "integer"},
                }},
                documentation="Read-only directory walk restricted to allowed roots; skips paths escaping the root.",
            ),
            _file_inventory,
        ),
        "json_transform": (
            _manifest(
                "json_transform",
                "Pure-Python JSON operations: keys, get, select, length, type.",
                category="development",
                input_schema={"type": "object", "properties": {
                    "data": {},
                    "operation": {"type": "string", "enum": ["keys", "get", "select", "length", "type"]},
                    "path": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                }, "required": ["data"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"result": {}}, "required": ["result"]},
                documentation="Deterministic in-memory transformation; no file or network access.",
            ),
            _json_transform,
        ),
        "schedule_reminder": (
            _manifest(
                "schedule_reminder",
                "Describe a reminder to be scheduled by a configured scheduler.",
                category="productivity",
                permissions=["schedule:write"],
                input_schema={"type": "object", "properties": {
                    "message": {"type": "string", "minLength": 1},
                    "due_at": {"type": "string"},
                }, "required": ["message"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"request": {"type": "object"}}},
                documentation=(
                    "No scheduler service is attached or verified. This manifest documents the "
                    "interface only and has no executable handler."
                ),
            ),
            None,
        ),
        "memory_lookup": (
            _manifest(
                "memory_lookup",
                "Describe a lookup in LEO's memory subsystem.",
                category="research",
                permissions=["memory:read"],
                input_schema={"type": "object", "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "category": {"type": "string"},
                }, "required": ["query"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"request": {"type": "object"}}},
                documentation=(
                    "Interface placeholder. A memory manager must be attached and tested before "
                    "any lookup is claimed to work; no handler is provided here."
                ),
            ),
            None,
        ),
        "task_status": (
            _manifest(
                "task_status",
                "Describe a durable task status query.",
                category="productivity",
                permissions=["task:read"],
                input_schema={"type": "object", "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                }, "required": ["task_id"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"request": {"type": "object"}}},
                documentation="Interface placeholder; requires an attached task engine and verification.",
            ),
            None,
        ),
        "security_scan_request": (
            _manifest(
                "security_scan_request",
                "Describe a security scan request for a configured scanner.",
                category="infrastructure",
                permissions=["security:read"],
                input_schema={"type": "object", "properties": {
                    "target": {"type": "string", "minLength": 1},
                    "scope": {"type": "string"},
                }, "required": ["target"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {"request": {"type": "object"}}},
                documentation=(
                    "No scanner is bundled or verified. This skill does not perform scanning and "
                    "must not be represented as an active security control."
                ),
            ),
            None,
        ),
        "report_generator": (
            _manifest(
                "report_generator",
                "Render a structured markdown or plain-text report from sections and findings.",
                category="research",
                input_schema={"type": "object", "properties": {
                    "title": {"type": "string"},
                    "format": {"type": "string", "enum": ["markdown", "text"]},
                    "sections": {"type": "array", "items": {"type": "object"}},
                }, "required": ["title", "sections"], "additionalProperties": False},
                output_schema={"type": "object", "properties": {
                    "report": {"type": "string"}, "section_count": {"type": "integer"},
                }, "required": ["report"]},
                documentation="Pure string formatting; content is data, never instructions.",
            ),
            _report_generator,
        ),
        "system_health": (
            _manifest(
                "system_health",
                "Report local runtime health information for the current LEO process.",
                category="infrastructure",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                output_schema={"type": "object", "properties": {
                    "status": {"type": "string"}, "platform": {"type": "string"},
                }, "required": ["status"]},
                documentation="Local-only: Python version, platform and PID. No network probes or remote hosts.",
            ),
            _system_health,
        ),
    }


_NATIVE_DEFINITIONS = _build_native_skills()
NATIVE_SKILLS: dict[str, SkillManifest] = {name: manifest for name, (manifest, _) in _NATIVE_DEFINITIONS.items()}
NATIVE_HANDLERS: dict[str, Handler] = {name: handler for name, (_, handler) in _NATIVE_DEFINITIONS.items() if handler}
NATIVE_SKILL_NAMES = frozenset(NATIVE_SKILLS)


def native_skills() -> Iterator[tuple[SkillManifest, Handler | None]]:
    """Yield independent copies of every native manifest and its handler."""
    for name, (manifest, handler) in _NATIVE_DEFINITIONS.items():
        yield manifest.model_copy(deep=True), NATIVE_HANDLERS.get(name)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Register, validate, execute and audit skills.

    The constructor is synchronous and seeds native manifests in memory; their
    durable rows are written lazily on the first asynchronous call.
    """

    def __init__(
        self,
        events: EventBus | None = None,
        permissions: Any = None,
        actor: str = "leo",
        auto_seed: bool = True,
        *,
        available_tools: Any = None,
        available_toolsets: Any = None,
        gate_context: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.events = events or EventBus(persist=True)
        self.permissions = permissions
        self.actor = actor
        self.available_tools = available_tools
        self.available_toolsets = available_toolsets
        self.gate_context = gate_context
        self._skills: dict[str, SkillRecord] = {}
        self._handlers: dict[str, Handler] = {}
        self._seeded = False
        if auto_seed:
            for manifest, handler in native_skills():
                self._skills[manifest.name] = SkillRecord(
                    **manifest.model_dump(), status="registered", health="available"
                )
                if handler is not None:
                    self._handlers[manifest.name] = handler

    async def _emit(self, topic: str, **payload: Any) -> None:
        await self.events.emit(topic, **payload)

    async def _ensure_seeded(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        # The shared skills table is the durable source of truth for lifecycle
        # state. Load it before seeding natives: otherwise a skill disabled (or
        # registered) by another registry instance looks "unknown" or enabled
        # again here. Handlers are deliberately NOT restored from the database
        # -- only locally supplied/native handlers are ever executable.
        try:
            rows = await db.fetchall("SELECT * FROM skills")
        except Exception as exc:
            _LOG.warning("could not load persisted skills: %s", exc)
            rows = []
        loaded: set[str] = set()
        for row in rows:
            name = row["name"]
            try:
                record = self._record_from_row(row)
            except Exception as exc:
                _LOG.warning("skipping malformed persisted skill %r: %s", name, exc)
                continue
            loaded.add(name)
            self._skills[name] = record
        for manifest, _ in native_skills():
            if manifest.name in loaded:
                continue
            record = SkillRecord(**manifest.model_dump(), status="registered", health="available")
            await self._write_row(record)
            await self._history(
                manifest.name, manifest.version, "native.seeded",
                {"source": manifest.source, "health": record.health},
            )

    async def _write_row(self, record: SkillRecord) -> None:
        now = utcnow()
        await db.execute(
            "INSERT OR REPLACE INTO skills(name,description,version,author,category,permissions,"
            "input_schema,output_schema,entrypoint,status,health,source,manifest,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.name, record.description, record.version, record.author, record.category,
                json.dumps(record.permissions), json.dumps(record.input_schema),
                json.dumps(record.output_schema), record.entrypoint, record.status, record.health,
                record.source, record.model_dump_json(), now, now,
            ),
        )

    async def _history(self, name: str, version: str, action: str, detail: Any = None) -> None:
        await db.execute(
            "INSERT INTO skill_history(skill_name,version,action,detail,changed_at) VALUES(?,?,?,?,?)",
            (name, version, action, json.dumps(detail or {}, default=str, sort_keys=True), utcnow()),
        )

    @staticmethod
    def _record_from_row(row: Any) -> SkillRecord:
        manifest = json.loads(row["manifest"]) if row["manifest"] else {}
        if not isinstance(manifest, dict):
            manifest = {}
        # Persisted manifests are SkillRecord dumps, so they already contain
        # lifecycle keys. Drop them: the dedicated columns are authoritative.
        lifecycle = {"status", "health", "enabled", "registered_at", "has_handler"}
        data = {key: value for key, value in manifest.items() if key not in lifecycle}
        base = SkillManifest.model_validate(data)
        return SkillRecord(
            **base.model_dump(), status=row["status"], health=row["health"] or "available",
            enabled=row["status"] != "disabled",
        )

    # -- validation / health -------------------------------------------------

    def validate(self, manifest: SkillManifest | dict[str, Any]) -> ValidationResult:
        return validate_manifest(manifest)

    @staticmethod
    def _gate_tokens(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [str(key) for key in value]
        if isinstance(value, (list, tuple, set, frozenset)):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        return []

    def _gate_context(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a bounded gate context from injected inventories and live state."""
        merged: dict[str, Any] = {}
        if self.gate_context is not None:
            try:
                extra = self.gate_context()
                if isinstance(extra, dict):
                    merged.update(extra)
            except Exception as exc:
                _LOG.warning("gate context provider failed: %s", exc)
        if context:
            merged.update(context)
        tools = set(NATIVE_HANDLERS)
        tools.update(name for name, handler in self._handlers.items() if handler is not None)
        for key in ("available_tools", "tools"):
            tools.update(self._gate_tokens(merged.get(key)))
        toolsets = set()
        for key in ("available_toolsets", "toolsets"):
            toolsets.update(self._gate_tokens(merged.get(key)))
        if self.available_toolsets:
            toolsets.update(self._gate_tokens(self.available_toolsets))
        if self.available_tools:
            tools.update(self._gate_tokens(self.available_tools))
        merged["available_tools"] = sorted(tools)
        merged["available_toolsets"] = sorted(toolsets)
        return merged

    def _gate_evaluation(self, record: SkillRecord, context: dict[str, Any] | None = None):
        return evaluate_gates(record.gates, self._gate_context(context))

    def _unsatisfied_gates(self, record: SkillRecord, context: dict[str, Any] | None = None) -> list[str]:
        """Named failing gate labels (convenience wrapper for callers/tests)."""
        return [failure.label for failure in self._gate_evaluation(record, context).failures]

    def _health_status(self, record: SkillRecord) -> str:
        """Derive health from live registry state, not a stored snapshot."""
        for dep in record.dependencies:
            dependency = self._skills.get(dep)
            if dependency is None or not dependency.enabled or dependency.status == "disabled":
                return "degraded"
        if not record.enabled or record.status == "disabled":
            return "degraded"
        if not self._gate_evaluation(record).satisfied:
            return "degraded"
        if execution_block(record):
            return "unavailable"
        return "available"

    def health_check(self, name: str) -> HealthReport:
        record = self._skills.get(name)
        if record is None:
            return HealthReport(name=name, healthy=False, status="unavailable", detail="skill is not registered")
        missing = [dep for dep in record.dependencies if dep not in self._skills]
        disabled = [
            dep for dep in record.dependencies
            if dep in self._skills
            and (not self._skills[dep].enabled or self._skills[dep].status == "disabled")
        ]
        if missing or disabled or not record.enabled or record.status == "disabled":
            if missing:
                detail = f"missing dependencies: {', '.join(missing)}"
            elif disabled:
                detail = f"disabled dependencies: {', '.join(disabled)}"
            else:
                detail = "skill is disabled"
            return HealthReport(name=name, healthy=False, status="degraded", detail=detail)
        evaluation = self._gate_evaluation(record)
        if not evaluation.satisfied:
            return HealthReport(
                name=name, healthy=False, status="degraded",
                detail="unsatisfied availability gates: " + evaluation.detail,
            )
        blocked = execution_block(record)
        if blocked:
            return HealthReport(name=name, healthy=False, status="unavailable", detail=blocked)
        if name not in self._handlers:
            return HealthReport(
                name=name, healthy=True, status="available",
                detail="no executable handler; documentation-only skill: " + bounded_preview(record.documentation, 80),
            )
        return HealthReport(name=name, healthy=True, status="available", detail="handler loaded locally")

    # -- CRUD ----------------------------------------------------------------

    async def register(
        self,
        manifest: SkillManifest | dict[str, Any],
        handler: Handler | None = None,
        *,
        replace: bool = False,
    ) -> SkillRecord:
        if isinstance(manifest, dict):
            manifest = SkillManifest.model_validate(manifest)
        result = self.validate(manifest)
        if not result.valid:
            raise ValueError("invalid skill manifest: " + "; ".join(result.errors))
        await self._ensure_seeded()
        if manifest.name in self._skills and not replace:
            raise ValueError(f"skill already registered: {manifest.name}")
        if replace and manifest.name not in self._skills:
            raise KeyError(f"unknown skill: {manifest.name}")

        if manifest.source != "native" and manifest.entrypoint:
            raise ValueError("external manifests cannot declare an entrypoint")
        health = "unavailable" if execution_block(manifest) else "available"
        missing = [dep for dep in manifest.dependencies if dep not in self._skills]
        if missing:
            health = "degraded"
        elif not evaluate_gates(manifest.gates, self._gate_context()).satisfied:
            health = "degraded"
        # A version replace/rollback must not silently re-enable a disabled
        # skill or drop its local handler (F-26). Preserve lifecycle state
        # from the previous record unless an explicit override is supplied.
        previous = self._skills.get(manifest.name)
        status = previous.status if replace and previous is not None else "registered"
        enabled = previous.enabled if replace and previous is not None else True
        record = SkillRecord(
            **manifest.model_dump(), status=status, health=health, enabled=enabled
        )
        await self._write_row(record)
        if handler is not None:
            self._handlers[manifest.name] = handler
        # With replace=True and no new handler, the existing local handler
        # (if any) is deliberately kept so rollback stays executable.
        self._skills[manifest.name] = record
        action = "version.changed" if replace else "registered"
        await self._history(manifest.name, manifest.version, action, {
            "source": manifest.source, "health": health, "missing_dependencies": missing,
            "has_handler": handler is not None,
            "provenance": {
                "source_path": bounded_preview(manifest.provenance.get("source_path"), 160),
                "content_hash": manifest.provenance.get("content_hash"),
                "version_source": manifest.version_source,
            },
        })
        await self._emit(
            "skill.updated" if replace else "skill.registered",
            name=manifest.name, version=manifest.version, source=manifest.source, health=health,
        )
        return self._with_handler_flag(record)

    async def replace(
        self, manifest: SkillManifest | dict[str, Any], handler: Handler | None = None,
        *, reason: str = "updated",
    ) -> SkillRecord:
        """Replace a registered skill's manifest/version (used for rollback)."""
        if isinstance(manifest, dict):
            manifest = SkillManifest.model_validate(manifest)
        previous = await self.get(manifest.name)
        if previous is None:
            raise KeyError(f"unknown skill: {manifest.name}")
        record = await self.register(manifest, handler, replace=True)
        await self._history(manifest.name, manifest.version, "version.restored", {
            "from": previous.version, "to": manifest.version, "reason": reason,
        })
        return record

    async def get(self, name: str) -> SkillRecord | None:
        await self._ensure_seeded()
        record = self._skills.get(name)
        return self._with_handler_flag(record) if record else None

    async def list(self, format: str = "objects") -> SkillList:
        if format not in ("objects", "dicts"):
            raise ValueError("format must be 'objects' or 'dicts'")
        await self._ensure_seeded()
        records = [self._with_handler_flag(self._skills[n]) for n in sorted(self._skills)]
        if format == "dicts":
            return SkillList(r.model_dump(mode="json") for r in records)
        return SkillList(records)

    async def enable(self, name: str) -> SkillRecord:
        return await self._set_status(name, "enabled", True, "skill.enabled")

    async def disable(self, name: str) -> SkillRecord:
        return await self._set_status(name, "disabled", False, "skill.disabled")

    async def _set_status(self, name: str, status: str, enabled: bool, topic: str) -> SkillRecord:
        await self._ensure_seeded()
        record = self._skills.get(name)
        if record is None:
            raise KeyError(f"unknown skill: {name}")
        if record.status == status:
            return self._with_handler_flag(record)
        record.status, record.enabled = status, enabled
        await self._write_row(record)
        await self._history(name, record.version, "enabled" if enabled else "disabled", {"status": status})
        await self._emit(topic, name=name, version=record.version)
        return self._with_handler_flag(record)

    async def approve_external(
        self,
        name: str,
        approval_id: str | None = None,
        *,
        reason: str = "verified provenance",
        verified_by: str | None = None,
    ) -> SkillRecord:
        """Grant executable status to an external manifest after audited approval.

        The flag is deliberately separate from ``register(handler=...)``: a
        locally supplied handler never launders untrusted provenance. Risk is
        HIGH, so the permission manager must resolve an approval before the
        manifest can run. Both the decision and this change are audited.
        """
        await self._ensure_seeded()
        record = self._skills.get(name)
        if record is None:
            raise KeyError(f"unknown skill: {name}")
        if record.source == "native":
            raise ValueError("native skills require no provenance approval")
        if self.permissions is None:
            raise PermissionError("provenance approval requires a permission manager for audit")
        decision = await self.permissions.authorize(
            self.actor, "skill.provenance.approve", name, risk=RiskLevel.HIGH,
            approval_id=approval_id, source=record.source,
            content_hash=record.provenance.get("content_hash"),
            reason=bounded_preview(reason, 80),
        )
        if decision.requires_approval:
            raise ApprovalRequired(
                decision.approval_id or "",
                f"external provenance approval required for skill: {name}",
            )
        if not decision.allowed:
            raise PermissionError(f"provenance approval denied: {decision.reason}")
        granted = decision.approval_id or approval_id
        if not granted:
            raise PermissionError("a resolved approval id is required for provenance approval")
        record.provenance_verified = True
        record.provenance_approval_id = granted
        record.provenance["verified_by"] = bounded_preview(verified_by or self.actor, 80)
        record.provenance["verification"] = "verified"
        await self._write_row(record)
        await self._history(name, record.version, "skill.provenance.approved", {
            "approval_id": granted,
            "content_hash": record.provenance.get("content_hash"),
            "source_path": bounded_preview(record.provenance.get("source_path"), 160),
            "verified_by": record.provenance["verified_by"],
        })
        await self._emit(
            "skill.provenance.approved", name=name, approval_id=granted,
            content_hash=record.provenance.get("content_hash"),
        )
        return self._with_handler_flag(record)

    async def unregister(self, name: str) -> None:
        await self._ensure_seeded()
        record = self._skills.pop(name, None)
        if record is None:
            raise KeyError(f"unknown skill: {name}")
        self._handlers.pop(name, None)
        await db.execute("DELETE FROM skills WHERE name=?", (name,))
        await self._history(name, record.version, "unregistered", {})
        await self._emit("skill.unregistered", name=name, version=record.version)

    def _with_handler_flag(self, record: SkillRecord) -> SkillRecord:
        copy = record.model_copy(deep=True)
        if copy.model_extra is None:
            copy.model_extra = {}
        copy.model_extra["has_handler"] = record.name in self._handlers
        # Stored health goes stale when a skill is disabled or a dependency is
        # unregistered; derive it from live state instead (same as health_check).
        copy.health = self._health_status(record)
        return copy

    # -- execution -----------------------------------------------------------

    async def run(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> SkillRunResult:
        started = time.perf_counter()
        arguments = arguments or {}
        await self._ensure_seeded()
        record = self._skills.get(name)
        if record is None:
            return SkillRunResult(skill=name, success=False, error="unknown skill")
        if not record.enabled or record.status == "disabled":
            return SkillRunResult(skill=name, success=False, error="skill is disabled")

        evaluation = self._gate_evaluation(record, context)
        if not evaluation.satisfied:
            error = "availability gates not satisfied: " + evaluation.detail
            result = SkillRunResult(skill=name, success=False, error=error)
            await self._history(name, record.version, "run.gate.blocked", {"gates": evaluation.detail})
            await self._emit("skill.run.gate_blocked", name=name, gates=evaluation.detail)
            return result

        blocked = execution_block(record)
        if blocked:
            result = SkillRunResult(skill=name, success=False, error="no executable handler: " + blocked)
            await self._history(name, record.version, "run.failed", {"error": blocked})
            await self._emit("skill.run.failed", name=name, error=blocked)
            return result

        validation_errors = _validate_arguments(arguments, record.input_schema)
        if validation_errors:
            result = SkillRunResult(
                skill=name, success=False, error="input validation failed",
                validation_errors=validation_errors,
            )
            await self._history(name, record.version, "run.failed", {"error": result.error})
            await self._emit("skill.run.failed", name=name, error=result.error, errors=validation_errors)
            return result

        if self.permissions is not None:
            supplied_approval = (context or {}).get("approval_id")
            decision = await self.permissions.authorize(
                self.actor, f"skill.run.{name}", name, risk=_risk_for(record),
                approval_id=supplied_approval,
            )
            if decision.requires_approval:
                return SkillRunResult(
                    skill=name, success=False, error="approval required",
                    decision="pending", approval_id=decision.approval_id,
                )
            if not decision.allowed:
                return SkillRunResult(skill=name, success=False, error=f"permission denied: {decision.reason}", decision="denied")

        handler = self._handlers.get(name)
        if handler is None:
            # Never copy an imported Markdown body into an error or event.
            preview = bounded_preview(record.documentation, 80)
            error = "skill has no executable handler; no local implementation is available"
            if preview:
                error += f" (documentation preview: {preview})"
            result = SkillRunResult(skill=name, success=False, error=error, validation_errors=validation_errors)
            await self._history(name, record.version, "run.failed", {"error": "no executable handler"})
            await self._emit("skill.run.failed", name=name, error="no executable handler")
            return result

        try:
            raw = handler(arguments, context) if _accepts_context(handler) else handler(arguments)
            if inspect.isawaitable(raw):
                output = await raw
            else:
                output = raw
        except Exception as exc:
            error = f"handler failed: {type(exc).__name__}: {bounded_preview(exc, 120)}"
            result = SkillRunResult(skill=name, success=False, error=error, validation_errors=validation_errors)
            await self._history(name, record.version, "run.failed", {"error": error})
            await self._emit("skill.run.failed", name=name, error=error)
            return result

        output_errors: list[str] = []
        if record.output_schema:
            _validate_value(output, record.output_schema, "output", output_errors)
        if output_errors:
            error = "output validation failed: " + "; ".join(output_errors)
            result = SkillRunResult(
                skill=name, success=False, output=output, error=error, validation_errors=output_errors,
            )
            await self._history(name, record.version, "run.failed", {"error": error})
            await self._emit("skill.run.failed", name=name, error=error)
            return result

        result = SkillRunResult(
            skill=name, success=True, output=output, validation_errors=validation_errors,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        await self._history(name, record.version, "run", {"success": True})
        await self._emit("skill.run", name=name, version=record.version, success=True)
        return result

    async def history(self, name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        await self._ensure_seeded()
        if name:
            rows = await db.fetchall(
                "SELECT * FROM skill_history WHERE skill_name=? ORDER BY id DESC LIMIT ?", (name, limit)
            )
        else:
            rows = await db.fetchall("SELECT * FROM skill_history ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]
