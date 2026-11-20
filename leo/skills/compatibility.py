"""Translation of external tool manifests into LEO's skill contract.

Only declarative fields are carried across. An external manifest never supplies
executable code, a shell command or an import path: unknown execution primitives
are reported and dangerous ones are rejected. Translation therefore produces an
inert, documented skill manifest -- it is not evidence that an external service
or integration works.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Awaitable

import yaml
from pydantic import BaseModel, Field

from ..contracts import SkillManifest

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_VERSION_RE = re.compile(r"^\d+(?:\.\d+){1,2}(?:[-+][0-9A-Za-z.-]+)?$")
_SIMPLE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_DANGEROUS_TOKEN_RE = re.compile(
    r"(?:^|[.:_\-/])(?:eval|exec|execute_code|code_exec|shell|subprocess|popen|system|"
    r"dynamic[_-]?import|import_module|__import__|os\.system|node|python|bash|sh|cmd|"
    r"arbitrary[_-]?code|run_command|command|script)(?:$|[.:_\-/])",
    re.IGNORECASE,
)
_DANGEROUS_CODE_RE = [
    re.compile(r"\beval\s*\(", re.I),
    re.compile(r"\bexec\s*\(", re.I),
    re.compile(r"__import__\s*\(", re.I),
    re.compile(r"importlib(?:\.|\s)", re.I),
    re.compile(r"\bos\s*\.\s*system\s*\(", re.I),
    re.compile(r"\bsubprocess\s*\.", re.I),
    re.compile(r"\b(?:popen|run_shell|shell_execute|execute_shell)\s*\(", re.I),
    re.compile(r"(?:^|\s)(?:ba|z|k)?sh\s+-c\b", re.I),
    re.compile(r"\bpython(?:3)?\s+-c\b", re.I),
    re.compile(r"\bnode\s+-e\b", re.I),
    re.compile(r"curl[^\n|]*\|\s*(?:ba)?sh\b", re.I),
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\brequire\s*\(", re.I),
    re.compile(r"\bnew\s+Function\s*\(", re.I),
]

_RECOGNIZED_FIELDS = {
    "name", "skill_name", "id", "title", "description", "summary", "help", "doc",
    "version", "skill_version", "author", "maintainer", "provider", "category", "tags",
    "permissions", "required_permissions", "scopes", "dependencies", "requires", "depends_on",
    "input_schema", "inputSchema", "parameters", "schema", "args", "arguments",
    "output_schema", "outputSchema", "returns", "response_schema", "documentation", "gates",
}
_SENSITIVE_FIELDS = {
    "license", "endpoint", "url", "base_url", "api_key", "api_key_env", "token",
    "credentials", "command", "execution", "runtime", "entrypoint", "code", "script",
}
# Path tokens are replaced with inert placeholders. They are never resolved to
# host paths and bundled files are never opened or executed.
_PATH_TOKEN_RE = re.compile(
    r"(?:\$\{|\$|%)?(?:baseDir|BASE_DIR|SKILL_DIR|HERMES_SKILL_DIR|OPENCLAW_SKILL_DIR)(?:\}|%)?",
    re.IGNORECASE,
)
_BUNDLE_DIRS = ("scripts", "references", "assets")
_PATH_PLACEHOLDER = "<bundle-root>"
_BODY_LIMIT = 4000
_PROVENANCE_VALUE_LIMIT = 200


def _bounded_value(value: Any, limit: int = _PROVENANCE_VALUE_LIMIT) -> str:
    text = " ".join(str(value).split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _path_label(source: Any) -> str:
    if isinstance(source, (str, Path)) and len(str(source)) < 4096:
        path = Path(str(source))
        if path.exists():
            return str(path.resolve())
        return str(source)
    return "<inline-content>"


def _file_provenance(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError:
        return {"source_path": str(path.resolve()), "content_hash": None, "read_error": "unreadable"}
    return {
        "source_path": str(path.resolve()),
        "content_hash": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _inventory_bundle(root: Path) -> tuple[dict[str, Any], list[str]]:
    """Inventory bundled files without reading or executing their contents."""
    warnings: list[str] = []
    resolved_root = root.resolve()
    files: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for directory in _BUNDLE_DIRS:
        candidate = root / directory
        if not candidate.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(candidate):
            dirnames.sort()
            for filename in sorted(filenames):
                if len(files) + len(rejected) >= 500:
                    warnings.append("bundle inventory truncated at 500 entries")
                    return {
                        "skill_md": "SKILL.md", "files": files, "rejected": rejected,
                        "directories": list(_BUNDLE_DIRS), "executed": [],
                    }, warnings
                path = Path(dirpath) / filename
                relative = path.relative_to(root).as_posix()
                try:
                    resolved = path.resolve()
                except (OSError, RuntimeError):
                    rejected.append({"path": relative, "reason": "could not resolve"})
                    warnings.append(f"bundle entry {relative!r} could not be resolved and was skipped")
                    continue
                if not _is_within(resolved, resolved_root):
                    rejected.append({"path": relative, "reason": "symlink escapes bundle root"})
                    warnings.append(f"bundle entry {relative!r} escapes the bundle root and was skipped")
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    size = -1
                files.append({
                    "path": relative,
                    "kind": directory,
                    "size_bytes": size,
                    "symlink": path.is_symlink(),
                })
    return {
        "skill_md": "SKILL.md", "files": files, "rejected": rejected,
        "directories": list(_BUNDLE_DIRS), "executed": [],
    }, warnings


def _replace_path_tokens(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in tokens:
            tokens.append(token)
        return _PATH_PLACEHOLDER

    return _PATH_TOKEN_RE.sub(substitute, text), tokens


def _version_is_declared(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.strip()
    if candidate.startswith("v"):
        candidate = candidate[1:]
    return bool(_VERSION_RE.match(candidate))


def _await_self(value: Any) -> Awaitable[Any]:
    async def _identity() -> Any:
        return value

    return _identity()


class CompatibilityReport(BaseModel):
    """Translation outcome, including unsupported fields and security warnings."""

    name: str | None = None
    compatible: bool
    source_format: str = "generic-json"
    manifest: SkillManifest | None = None
    translated: dict[str, Any] = Field(default_factory=dict)
    supported_fields: list[str] = Field(default_factory=list)
    unsupported_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    security_warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    classification: str = "compatible-adapter"

    @property
    def ok(self) -> bool:
        return self.compatible

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def __await__(self):
        return _await_self(self).__await__()


# Public aliases for callers with different naming conventions.
TranslationReport = CompatibilityReport


def _walk(value: Any) -> Any:
    """Yield every key/value pair in a nested mapping/list structure."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def scan_security(data: dict[str, Any]) -> list[str]:
    """Return blocking reasons for dangerous execution primitives in a manifest."""
    reasons: list[str] = []
    if not isinstance(data, dict):
        return ["manifest must be an object"]
    for key, value in _walk(data):
        normalized = key.strip().lower().replace(" ", "_")
        if normalized in {
            "exec", "eval", "execute", "execution", "shell", "command", "cmd",
            "subprocess", "script", "code", "dynamic_import", "import_path",
            "import_module", "run_shell", "shell_command", "loader",
        }:
            if value not in (None, "", False, [], {}):
                reasons.append(f"dangerous execution primitive field: {key!r}")
        if isinstance(value, str):
            for pattern in _DANGEROUS_CODE_RE:
                if pattern.search(value):
                    reasons.append(f"dangerous code or shell string in field {key!r}")
                    break
    for field in ("permissions", "required_permissions", "scopes"):
        value = data.get(field)
        if isinstance(value, list):
            for permission in value:
                if isinstance(permission, str) and _DANGEROUS_TOKEN_RE.search(permission):
                    reasons.append(f"dangerous permission: {permission!r}")
    entrypoint = data.get("entrypoint")
    if isinstance(entrypoint, str) and _DANGEROUS_TOKEN_RE.search(entrypoint):
        reasons.append(f"dangerous entrypoint: {entrypoint!r}")
    return sorted(set(reasons))


def _load_manifest(data: dict[str, Any] | str | Path) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(data, (str, Path)):
        text = str(data)
        path = Path(text)
        if len(text) < 4096 and path.is_file():
            text = path.read_text(encoding="utf-8")
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return None, f"invalid JSON/YAML manifest: {exc}"
        if not isinstance(loaded, dict):
            return None, "manifest root must be an object"
        return loaded, None
    if isinstance(data, BaseModel):
        data = data.model_dump()
    if not isinstance(data, dict):
        return None, "manifest must be a mapping, JSON string, YAML string or path"
    return data, None


def _normalize_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = re.sub(r"[^a-z0-9_-]+", "_", value.strip().lower()).strip("_")
    if not name:
        return None
    if not _NAME_RE.match(name):
        name = name[:64].rstrip("_-")
    return name if _NAME_RE.match(name) else None


def _normalize_version(value: Any, default: str = "1.0.0") -> str:
    if not isinstance(value, str):
        return default
    version = value.strip()
    if version.startswith("v"):
        version = version[1:]
    version = re.sub(r"[^0-9A-Za-z.+-]", ".", version)
    if not re.match(r"^\d+(?:\.\d+)?", version):
        return default
    parts = version.split(".", 2)
    while len(parts) < 3:
        parts.append("0")
    normalized = ".".join(parts)
    return normalized if _VERSION_RE.match(normalized) else default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return []


def _build_input_schema(data: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    for key in ("input_schema", "inputSchema"):
        schema = data.get(key)
        if isinstance(schema, dict) and (schema.get("type") or "properties" in schema):
            return schema
        if isinstance(schema, dict) and schema:
            warnings.append(f"{key} lacks JSON Schema markers; treating its keys as parameters")
            return _parameters_to_schema(schema)
    for key in ("parameters", "schema", "args", "arguments"):
        schema = data.get(key)
        if isinstance(schema, dict):
            if schema.get("type") == "object" or "properties" in schema:
                return schema
            return _parameters_to_schema(schema)
        if isinstance(schema, list):
            props: dict[str, Any] = {}
            required: list[str] = []
            for item in schema:
                if not isinstance(item, dict):
                    continue
                name = _normalize_name(item.get("name"))
                if not name:
                    continue
                spec = {k: v for k, v in item.items() if k not in {"name", "required", "default"}}
                spec.setdefault("type", "string")
                if "default" in item:
                    spec["default"] = item["default"]
                props[name] = spec
                if item.get("required"):
                    required.append(name)
            return {"type": "object", "properties": props, "required": required, "additionalProperties": False}
    return {"type": "object", "properties": {}}


def _parameters_to_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    for key, value in parameters.items():
        if str(key).startswith("$"):
            continue
        spec = dict(value) if isinstance(value, dict) else {"type": "string", "description": str(value)}
        if spec.pop("required", False):
            required.append(str(key))
        props[str(key)] = spec
    return {
        "type": "object", "properties": props, "required": required,
        "additionalProperties": False,
    }


def translate_manifest(
    data: dict[str, Any] | str | Path,
    *,
    default_version: str = "1.0.0",
    ecosystem: str | None = None,
    source_name: str | None = None,
    gates: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
) -> CompatibilityReport:
    """Translate a generic external JSON/YAML tool manifest into a SkillManifest.

    Returns a report rather than raising for ordinary incompatibilities. Security
    rejections are reported in ``errors`` with ``compatible=False``.

    ``default_version`` is only used when no version was declared; callers that
    import external skills must pass an explicit sentinel so an absent version
    is never presented as a real upstream version.
    """
    raw, load_error = _load_manifest(data)
    if gates is None and isinstance(raw, dict) and isinstance(raw.get("gates"), dict):
        gates = raw["gates"]
    if provenance is None and isinstance(data, (str, Path)) and len(str(data)) < 4096:
        path = Path(str(data))
        if path.is_file():
            provenance = _file_provenance(path)
    if load_error or raw is None:
        return CompatibilityReport(compatible=False, errors=[load_error or "empty manifest"])

    name = _normalize_name(raw.get("name") or raw.get("skill_name") or raw.get("id") or raw.get("title"))
    if not name:
        return CompatibilityReport(compatible=False, errors=["manifest has no usable 'name'"])

    errors = scan_security(raw)
    if errors:
        return CompatibilityReport(
            name=name, compatible=False, errors=errors,
            security_warnings=["external execution primitives are never imported or executed by LEO"],
        )

    supported: list[str] = []
    unsupported: list[str] = []
    warnings: list[str] = []
    security_warnings: list[str] = []
    for key in raw:
        (supported if key in _RECOGNIZED_FIELDS else unsupported).append(str(key))
        if key in _SENSITIVE_FIELDS:
            security_warnings.append(
                f"field {key!r} is not executed or verified by the compatibility layer"
            )

    description = next(
        (raw[k] for k in ("description", "summary", "help", "doc") if isinstance(raw.get(k), str) and raw[k].strip()),
        "",
    )
    if not description:
        warnings.append("no description provided; substituted a neutral description")
        description = f"Compatible adapter for external skill '{name}' (unverified)."

    input_schema = _build_input_schema(raw, warnings)
    output_schema: dict[str, Any] = {}
    for key in ("output_schema", "outputSchema", "returns", "response_schema"):
        value = raw.get(key)
        if isinstance(value, dict) and value:
            output_schema = value
            break

    permissions: list[str] = []
    for key in ("permissions", "required_permissions", "scopes"):
        for token in _string_list(raw.get(key)):
            if _SIMPLE_TOKEN_RE.match(token) and not _DANGEROUS_TOKEN_RE.search(token):
                if token not in permissions:
                    permissions.append(token)
            else:
                security_warnings.append(f"dropped unsupported permission token: {token!r}")

    dependencies: list[str] = []
    for dep in _string_list(raw.get("dependencies") or raw.get("requires") or raw.get("depends_on")):
        if not _SIMPLE_TOKEN_RE.match(dep) or dep == name:
            continue
        if dep in dependencies:
            warnings.append(f"dropped duplicate dependency: {dep}")
            continue
        dependencies.append(dep)
    if any(dep == name for dep in _string_list(raw.get("dependencies"))):
        warnings.append("dropped self-dependency declared by the external skill")
    author = next(
        (str(raw[k]) for k in ("author", "maintainer", "provider") if raw.get(k)),
        "external",
    )
    tags = _string_list(raw.get("tags"))
    category = str(raw.get("category") or (tags[0] if tags else "general"))
    declared_version = raw.get("version") or raw.get("skill_version")
    version = _normalize_version(declared_version, default=default_version)
    version_source = "declared" if declared_version else "unspecified"
    if declared_version and not _version_is_declared(declared_version):
        version_source = "invalid-declared"
        warnings.append(
            f"declared version {_bounded_value(declared_version)!r} is invalid; "
            f"normalized to {version} and marked invalid-declared"
        )

    docs = [raw["documentation"]] if isinstance(raw.get("documentation"), str) and raw["documentation"] else []
    docs.append(
        "Declarative compatibility adapter only: no external code is imported, no command is "
        "executed and no service is contacted. Registration and verification are required "
        "before any integration can be claimed as working."
    )
    if unsupported:
        docs.append("Unsupported source fields (retained for review only): " + ", ".join(sorted(unsupported)))

    manifest = SkillManifest(
        name=name,
        description=description,
        version=version,
        author=author,
        category=category,
        permissions=permissions,
        dependencies=dependencies,
        gates=gates or {},
        ecosystem=ecosystem,
        source_name=source_name,
        version_source=version_source,
        input_schema=input_schema,
        output_schema=output_schema,
        entrypoint=None,
        source="compatible-adapter",
        documentation="\n".join(docs),
        tests=["tests/test_skills.py::test_compatibility_translation"],
        provenance=dict(provenance or {}),
        bundle=dict(bundle or {}),
    )
    translated = manifest.model_dump(mode="json")
    if tags:
        translated["tags"] = tags
    if "license" in raw:
        translated["license"] = str(raw["license"])
    for key in ("endpoint", "url", "base_url", "api_key_env", "official", "verification", "reimplementation"):
        if key in raw:
            translated[key] = raw[key]

    return CompatibilityReport(
        name=name,
        compatible=True,
        source_format="yaml" if isinstance(data, str) and not data.strip().startswith("{") else "generic-json",
        manifest=manifest,
        translated=translated,
        supported_fields=sorted(set(supported)),
        unsupported_fields=sorted(set(unsupported)),
        warnings=warnings,
        security_warnings=security_warnings,
    )


translate = translate_manifest
translate_external_manifest = translate_manifest


_SKILL_MD_RE = re.compile(r"\A---\s*\n(?P<front>.*?)\n---\s*(?:\n(?P<body>.*))?\Z", re.DOTALL)


_KNOWN_FRONT_FIELDS = _RECOGNIZED_FIELDS | {
    "platforms", "prerequisites", "required_environment_variables", "required_commands",
    "allowed-tools", "metadata",
}
_KNOWN_HERMES_FIELDS = {
    "requires_tools", "requires_toolsets", "fallback_for_tools", "fallback_for_toolsets",
    "config", "tags", "related_skills", "category", "homepage", "upstream", "blueprint",
}
_KNOWN_OPENCLAW_FIELDS = {
    "requires", "bins", "anyBins", "env", "config", "primaryEnv", "os", "install",
    "permissions", "skillKey", "always", "user-invocable", "emoji",
}


def _quarantined_report(
    name: str, label: str, reason: str, provenance: dict[str, Any], bundle: dict[str, Any],
    source_format: str = "skill.md:unknown",
) -> CompatibilityReport:
    """Return an inert, path-bearing report for an untranslatable source file."""
    warning = f"quarantined {label}: {reason}; content was not translated or executed"
    clean_name = _normalize_name(name) or "quarantined_skill"
    return CompatibilityReport(
        name=clean_name,
        compatible=False, classification="quarantined", source_format=source_format,
        warnings=[warning], errors=[warning],
        manifest=SkillManifest(
            name=clean_name,
            description="Quarantined external file; not translated and never executable.",
            version="0.0.0", source="quarantined", entrypoint=None,
            documentation="Quarantined: no usable YAML frontmatter; no content was executed.",
            tests=["tests/test_compat_skill_md.py"],
            provenance={**provenance, "quarantined": True}, bundle=bundle,
        ),
    )


def translate_skill_md(source: str | Path) -> CompatibilityReport:
    """Translate a Hermes/OpenClaw ``SKILL.md`` file or skill *directory*.

    Both ecosystems use SKILL.md but with different metadata namespaces
    (``metadata.hermes`` vs ``metadata.openclaw``). Only declarative frontmatter
    is carried across; bundled files are inventoried, never opened or executed.
    Path tokens in the Markdown body become inert placeholders. Provenance is
    retained on the manifest so it survives persistence.
    """
    text = str(source)
    source_path: Path | None = None
    root: Path | None = None
    if len(text) < 4096:
        candidate = Path(text)
        if candidate.is_dir():
            root = candidate
            matches = [p for p in sorted(root.iterdir()) if p.is_file() and p.name.lower() == "skill.md"]
            if len(matches) != 1:
                found = "none" if not matches else ", ".join(p.name for p in matches)
                return CompatibilityReport(
                    compatible=False,
                    errors=[f"skill bundle {_path_label(root)} must contain exactly one SKILL.md (found {found})"],
                )
            source_path = matches[0]
        elif candidate.is_file():
            source_path = candidate

    label = _path_label(root if root is not None else (source_path or text))
    content = source_path.read_bytes() if source_path is not None else text.encode("utf-8")
    provenance: dict[str, Any] = {
        "source_path": label,
        "skill_md_path": source_path.name if source_path else None,
        "content_hash": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "source_kind": "bundle" if root else ("file" if source_path else "inline"),
    }
    bundle: dict[str, Any] = {
        "skill_md": source_path.name if source_path else None,
        "files": [], "rejected": [], "path_tokens": [], "executed": [],
    }
    bundle_warnings: list[str] = []
    if root is not None:
        bundle, bundle_warnings = _inventory_bundle(root)
        provenance["bundle_root"] = _path_label(root)

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _quarantined_report(
            source_path.stem if source_path else "quarantined_skill", label,
            f"file is not valid UTF-8 ({exc.__class__.__name__})", provenance, bundle,
        )
    match = _SKILL_MD_RE.match(text)
    if not match:
        return _quarantined_report(
            source_path.parent.name if root and source_path else (source_path.stem if source_path else "quarantined_skill"),
            label, "no YAML frontmatter delimited by ---", provenance, bundle,
        )
    try:
        front = yaml.safe_load(match.group("front")) or {}
    except yaml.YAMLError as exc:
        return _quarantined_report(
            source_path.parent.name if root and source_path else (source_path.stem if source_path else "quarantined_skill"),
            label, f"invalid frontmatter: {exc.__class__.__name__}", provenance, bundle,
        )
    if not isinstance(front, dict):
        return _quarantined_report(
            source_path.parent.name if root and source_path else (source_path.stem if source_path else "quarantined_skill"),
            label, "frontmatter must be a mapping", provenance, bundle,
        )
    body = (match.group("body") or "").strip()
    body, path_tokens = _replace_path_tokens(body)
    bundle["path_tokens"] = path_tokens
    if len(body) > _BODY_LIMIT:
        body = body[:_BODY_LIMIT] + "\n[body preview truncated]"
        bundle_warnings.append("Markdown body preview truncated to 4000 characters")
    metadata = front.get("metadata") if isinstance(front.get("metadata"), dict) else {}
    ecosystem = "hermes" if "hermes" in metadata else ("openclaw" if "openclaw" in metadata else "unknown")

    # Keep all frontmatter fields so the security scanner can reject dangerous
    # execution primitives instead of silently dropping them during mapping.
    # The Markdown body is data, not code: it is carried as documentation only
    # and is deliberately excluded from the blocking execution-primitive scan.
    raw: dict[str, Any] = {
        key: value for key, value in front.items() if key not in {"metadata", "documentation"}
    }
    declared_version = front.get("version") or front.get("skill_version")
    raw.update({
        "name": front.get("name"),
        "description": front.get("description"),
        "version": declared_version,
        "author": front.get("author", "external"),
        "license": front.get("license"),
        "category": front.get("category", "general"),
    })
    namespace = metadata.get(ecosystem) if isinstance(metadata.get(ecosystem), dict) else {}
    meta_risks = scan_security({"metadata": metadata})
    if meta_risks:
        return CompatibilityReport(
            name=_normalize_name(front.get("name")), compatible=False,
            source_format=f"skill.md:{ecosystem}", errors=meta_risks,
            security_warnings=["external execution primitives in skill metadata are never imported"],
        )

    # Availability gates are conditions, not grants: keep them separate from
    # authorization (permissions) and registry dependencies.
    gates: dict[str, Any] = {}
    if ecosystem == "hermes":
        gates = {
            "tools": _string_list(namespace.get("requires_tools")),
            "toolsets": _string_list(namespace.get("requires_toolsets")),
            "fallback_for_tools": _string_list(namespace.get("fallback_for_tools")),
            "fallback_for_toolsets": _string_list(namespace.get("fallback_for_toolsets")),
            "platforms": _string_list(front.get("platforms")),
            "prerequisites": front.get("prerequisites") or {},
            "environment": _string_list(front.get("required_environment_variables")),
            "commands": _string_list(front.get("required_commands")),
        }
    elif ecosystem == "openclaw":
        requires = namespace.get("requires") if isinstance(namespace.get("requires"), dict) else {}
        gates = {
            "bins": _string_list(requires.get("bins")),
            "anyBins": _string_list(requires.get("anyBins")),
            "env": _string_list(requires.get("env")),
            "config": _string_list(requires.get("config")),
            "primaryEnv": namespace.get("primaryEnv"),
            "os": _string_list(namespace.get("os")),
            "install": namespace.get("install"),
            "allowed_tools": _string_list(front.get("allowed-tools")),
            "user_invocable": namespace.get("user-invocable"),
        }
    gates = {key: value for key, value in gates.items() if value not in (None, "", [], {})}
    explicit_permissions = front.get("permissions") or front.get("required_permissions") or front.get("scopes")
    if ecosystem == "openclaw" and namespace.get("permissions"):
        explicit_permissions = namespace.get("permissions")
    raw["permissions"] = _string_list(explicit_permissions)
    raw = {key: value for key, value in raw.items() if value not in (None, "", [], {})}

    unknown_frontmatter: dict[str, str] = {}
    for key, value in front.items():
        if key not in _KNOWN_FRONT_FIELDS:
            unknown_frontmatter[str(key)] = _bounded_value(value)
    known_namespace = _KNOWN_HERMES_FIELDS if ecosystem == "hermes" else (
        _KNOWN_OPENCLAW_FIELDS if ecosystem == "openclaw" else set()
    )
    for key, value in namespace.items():
        if key not in known_namespace:
            unknown_frontmatter[f"metadata.{ecosystem}.{key}"] = _bounded_value(value)
    provenance.update({
        "ecosystem": ecosystem,
        "source_name": str(front.get("name") or "") or None,
        "unsupported_frontmatter": unknown_frontmatter,
        "frontmatter_keys": sorted(str(key) for key in front),
    })

    report = translate_manifest(
        raw,
        default_version="0.0.0",
        ecosystem=ecosystem,
        source_name=str(front.get("name") or "") or None,
        gates=gates,
        provenance=provenance,
        bundle=bundle,
    )
    if report.manifest and not declared_version:
        report.warnings.append(
            "upstream skill declared no version; 0.0.0 is an explicit sentinel, not a real version"
        )
    report.source_format = f"skill.md:{ecosystem}"
    report.classification = "compatible-adapter"
    report.warnings.extend(bundle_warnings)
    report.security_warnings.extend(
        warning for warning in bundle_warnings if "escapes" in warning or "skipped" in warning
    )
    report.security_warnings.extend(
        f"bundled path token {token!r} is never resolved to a host path" for token in path_tokens[:10]
    )
    unsupported = list(report.unsupported_fields) + list(unknown_frontmatter)
    if ecosystem == "unknown":
        report.warnings.append("no metadata.hermes or metadata.openclaw namespace found; ecosystem-specific gates ignored")
    if ecosystem == "openclaw" and "requires" in namespace:
        unsupported.append("metadata.openclaw.requires (availability gates retained in manifest.gates)")
    if ecosystem == "hermes" and "hermes" in metadata:
        unsupported.append("metadata.hermes (availability gates retained in manifest.gates, never executed)")
    report.unsupported_fields = sorted(set(unsupported))
    if body:
        body_risks = scan_security({"documentation": body})
        if body_risks:
            report.security_warnings.extend(
                risk + " (body is documentation data and is never executed)" for risk in body_risks
            )
    if report.manifest:
        report.manifest.documentation = (
            f"Imported from {ecosystem or 'unknown'} SKILL.md. "
            + (body + "\n\n" if body else "")
            + "Declarative adapter only: no source scripts are imported or executed."
        )
        report.manifest.tests.append("tests/test_compat_skill_md.py")
        report.translated["gates"] = gates
        report.translated["provenance"] = provenance
        report.translated["bundle"] = bundle
    return report
