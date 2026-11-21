"""Autonomous integration factory with review, approval and rollback.

The factory only inspects caller-supplied local candidates and performs no
network access. Generated adapters are declarative Python modules with a
validation-only handler; their tests execute the generated source inside a
scrubbed ``python -I`` child with CPU/memory limits and a hard timeout. A
passing report bound to the exact source hash and review version is required
before registration. High-risk registrations are authorised through
``PermissionManagerImpl`` and require an explicitly resolved approval.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml
from pydantic import BaseModel, Field

from .. import db
from ..config import Settings
from ..contracts import RiskLevel, SkillManifest, utcnow
from ..events import EventBus
from ..logging import get_logger
from ..security.permissions import ApprovalRequired, PermissionManagerImpl
from ..skills.compatibility import CompatibilityReport, translate_manifest
from ..skills.registry import NATIVE_SKILL_NAMES, SkillRecord, SkillRegistry

_LOG = get_logger("integrations")

_SPDX_ALLOWLIST = {
    "mit", "apache-2.0", "apache-2", "bsd", "bsd-2-clause", "bsd-3-clause",
    "bsd-4-clause", "isc", "cc0", "cc0-1.0",
}
_RESTRICTED_LICENSES = {
    "gpl-1.0", "gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-2.1", "lgpl-3.0",
    "sspl-1.0", "proprietary", "cc-by-nc", "cc-by-nc-sa",
}
_SAFE_MINIMUM_PERMISSIONS = frozenset({
    "skill:read", "skill:run", "memory:read", "task:read", "fs:read",
    "git:read", "net:read", "security:read", "schedule:read",
})
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")

# OS-level limits for the untrusted adapter child process. The wall-clock limit
# is enforced by the parent; these bound CPU and address space inside the child.
_ISOLATED_MEMORY_BYTES = 256 * 1024 * 1024
_ISOLATED_OUTPUT_LIMIT = 4096

# This harness is the only code executed by `python -I -c`: the adapter source,
# the test callable source and all metadata arrive as JSON on stdin. No project
# module, environment variable or inherited file descriptor is available to it.
_ISOLATED_HARNESS = r'''
import asyncio
import hashlib
import io
import json
import os
import sys
import threading
import time
import types

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None


def _apply_limits(cpu_seconds, memory_bytes):
    """Set kernel limits where available; some platforms reject AS/RSS limits."""
    applied = []
    if resource is not None:
        for name, soft, hard in (
            ("RLIMIT_CPU", cpu_seconds, cpu_seconds + 1),
            ("RLIMIT_AS", memory_bytes, memory_bytes),
            ("RLIMIT_DATA", memory_bytes, memory_bytes),
            ("RLIMIT_RSS", memory_bytes, memory_bytes),
            ("RLIMIT_CORE", 0, 0),
        ):
            limit = getattr(resource, name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (soft, hard))
                applied.append(name)
            except (ValueError, OSError):
                pass

    # Enforce memory even when the platform refuses RLIMIT_AS/RSS lowering:
    # a daemon watchdog kills this child if peak RSS exceeds the budget.
    if resource is not None:
        def _watch_memory():
            while True:
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss = peak if sys.platform == "darwin" else peak * 1024
                if rss > memory_bytes:
                    os._exit(97)
                time.sleep(0.01)

        threading.Thread(target=_watch_memory, name="adapter-memory-watchdog", daemon=True).start()
    return applied




class _AdapterView:
    """Inert view of the generated module; no host services are exposed."""

    def __init__(self, name, module, manifest, review, source):
        self.name = name
        self.source = source
        self.manifest = manifest
        self.review = review
        self.verified = False
        self._module = module

    def validate(self, arguments):
        return list(self._module.validate(arguments))

    async def run(self, arguments, context=None):
        result = self._module.handle(arguments, context)
        if hasattr(result, "__await__"):
            result = await result
        return result


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _main():
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    cpu_seconds = int(payload["cpu_seconds"])
    memory_bytes = int(payload["memory_bytes"])
    applied_limits = _apply_limits(cpu_seconds, memory_bytes)
    result = {
        "passed": False, "details": None, "error": None, "failure": None,
        "source_sha256": "", "review_version": payload.get("review_version", ""),
        "stdout": "", "limits_applied": applied_limits,
    }
    source = payload.get("source", "")
    try:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        result["source_sha256"] = digest
        if digest != payload.get("source_sha256"):
            raise ValueError("adapter source hash mismatch in isolated process")
        module = types.ModuleType("leo_isolated_adapter")
        exec(compile(source, "<generated-adapter>", "exec"), module.__dict__)
        adapter = _AdapterView(
            payload.get("name", "adapter"), module,
            payload.get("manifest") or {}, payload.get("review") or {}, source,
        )
        test_source = payload.get("test_source", "")
        test_name = payload.get("test_name", "")
        if not test_source or not test_name:
            raise ValueError("test callable source is unavailable")
        namespace = {"__name__": "leo_isolated_test", "asyncio": asyncio,
                     "json": json, "hashlib": hashlib,
                     "LEO_CPU_LIMIT_SECONDS": cpu_seconds,
                     "LEO_MEMORY_LIMIT_BYTES": memory_bytes}
        exec(compile(test_source, "<adapter-test>", "exec"), namespace)
        test_callable = namespace.get(test_name)
        if not callable(test_callable):
            raise ValueError("test callable could not be reconstructed")
        captured = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = captured
        try:
            if payload.get("test_is_async"):
                raw = asyncio.run(test_callable(adapter))
            else:
                raw = test_callable(adapter)
                if hasattr(raw, "__await__"):
                    async def _await_result():
                        return await raw
                    raw = asyncio.run(_await_result())
        finally:
            sys.stdout = real_stdout
            result["stdout"] = captured.getvalue()[:4096]
        result["details"] = _json_safe(raw)
        result["passed"] = raw is True or (
            isinstance(raw, dict) and raw.get("passed") is True
        )
        if not result["passed"]:
            result["error"] = "adapter test returned a non-passing result"
            result["failure"] = {
                "code": "non_passing_test_result",
                "expected": "True or {'passed': True}",
                "raw": _json_safe(raw),
            }
    except BaseException as exc:
        result["passed"] = False
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        result["failure"] = {
            "code": "isolated_test_error",
            "message": result["error"],
        }
    sys.stdout.write(json.dumps(result, default=str))


_main()
'''


def _json_safe(value: Any) -> Any:
    """Return a value that can be persisted as JSON without surprises."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _test_passed(value: Any) -> tuple[bool, dict[str, Any] | None]:
    """Only explicit True and {'passed': True} are passes; keep raw detail."""
    if value is True or (isinstance(value, dict) and value.get("passed") is True):
        return True, None
    return False, {
        "code": "non_passing_test_result",
        "expected": "True or {'passed': True}",
        "raw": _json_safe(value),
    }


def _await_self(value: Any) -> Awaitable[Any]:
    async def _identity() -> Any:
        return value

    return _identity()


class IntegrationReview(BaseModel):
    """Review report for one integration candidate."""

    name: str
    classification: str
    license: str
    license_status: str
    license_compatible: bool
    risk_score: int
    risk_level: RiskLevel
    security_warnings: list[str] = Field(default_factory=list)
    dependency_check: dict[str, Any] = Field(default_factory=dict)
    compatibility: dict[str, Any] = Field(default_factory=dict)
    verdict: str
    requires_approval: bool = False
    reasons: list[str] = Field(default_factory=list)
    version: str = "1.0.0"

    @property
    def approved(self) -> bool:
        return self.verdict == "approve"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def __await__(self):
        return _await_self(self).__await__()


class AdapterTestError(ValueError):
    """Registration blocked by a missing, stale or failed adapter test report."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = dict(details or {})
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


class AdapterTestReport(BaseModel):
    name: str
    passed: bool
    error: str | None = None
    duration_ms: float = 0.0
    timeout_seconds: float = 2.0
    details: Any = None
    source_sha256: str = ""
    review_version: str = ""
    mode: str = "isolated"
    failure: dict[str, Any] | None = None

    def __await__(self):
        return _await_self(self).__await__()


class RestrictedAdapter:
    """Minimal, inert view of a generated adapter handed to isolation tests."""

    def __init__(self, name: str, source: str, manifest: dict[str, Any], review: dict[str, Any]) -> None:
        self.name = name
        self.source = source
        self.manifest = manifest
        self.review = review
        self.verified = False

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        schema = self.manifest.get("input_schema") or {}
        properties = schema.get("properties", {})
        errors: list[str] = []
        for required in schema.get("required", []):
            if required not in arguments:
                errors.append(f"{required}: required")
        for key, value in arguments.items():
            spec = properties.get(key)
            if not spec:
                continue
            expected = spec.get("type")
            if expected == "string" and not isinstance(value, str):
                errors.append(f"{key}: expected string")
            if expected == "object" and not isinstance(value, dict):
                errors.append(f"{key}: expected object")
            if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                errors.append(f"{key}: expected integer")
        return errors

    async def run(self, arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a structured envelope; perform no external operation."""
        errors = self.validate(arguments)
        if errors:
            return {"status": "invalid", "errors": errors, "adapter": self.name}
        return {
            "status": "unverified",
            "adapter": self.name,
            "arguments": dict(arguments),
            "note": "declarative adapter: no external call is performed",
        }


def _check_license(raw: Any) -> tuple[str, bool, str]:
    if raw is None or not str(raw).strip():
        return "unknown", False, "license is missing; requires review"
    text = str(raw).strip()
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9.+-]+", text)}
    if tokens & _RESTRICTED_LICENSES:
        return "restricted", False, f"license '{text}' is not on the permissive SPDX allowlist"
    allowed = tokens & _SPDX_ALLOWLIST
    if allowed:
        unknown = tokens - _SPDX_ALLOWLIST - {"or", "and", "with", "license", "spdx"}
        note = f"license '{text}' is allowlisted" + (
            f" (unrecognised tokens: {', '.join(sorted(unknown))})" if unknown else ""
        )
        return "compatible", True, note
    return "unknown", False, f"unknown SPDX identifier '{text}'; requires review"


def _dependency_check(candidate: dict[str, Any], native_names: frozenset[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"native": [], "python_packages": [], "missing": [], "invalid": []}
    deps = candidate.get("dependencies") or candidate.get("requires") or []
    if isinstance(deps, str):
        deps = [deps]
    if not isinstance(deps, list):
        return {**report, "error": "dependencies must be a list"}
    for dep in deps:
        if not isinstance(dep, str) or not _MODULE_RE.match(dep):
            report["invalid"].append(str(dep))
            continue
        if dep in native_names:
            report["native"].append(dep)
            continue
        try:
            found = importlib.util.find_spec(dep.split(".")[0]) is not None
        except (ImportError, ValueError, AttributeError):
            found = False
        (report["python_packages"] if found else report["missing"]).append(dep)
    return report


def _clean_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.loads(json.dumps(value, default=str))
        for key, value in candidate.items()
    }


class IntegrationFactory:
    """Discover -> evaluate -> generate -> test -> register -> rollback."""

    def __init__(
        self,
        events: EventBus | None = None,
        permissions: PermissionManagerImpl | Any | None = None,
        registry: SkillRegistry | None = None,
        actor: str = "integration-factory",
        settings: Settings | None = None,
        timeout_seconds: float = 2.0,
        *,
        trusted_internal: bool = False,
    ) -> None:
        self.events = events or EventBus(persist=True)
        self.settings = settings or Settings()
        self.actor = actor
        self.permissions = permissions if permissions is not None else PermissionManagerImpl(
            self.settings, events=self.events
        )
        # The default registry shares the factory's permission manager so an
        # external manifest can receive an audited provenance approval.
        self.registry = registry or SkillRegistry(
            events=self.events, permissions=self.permissions, actor=self.actor
        )
        self.timeout_seconds = timeout_seconds
        # Opt-in for trusted, in-process tests only. Untrusted adapter code is
        # always executed in a child process by default.
        self.trusted_internal = bool(trusted_internal)
        self._candidates: dict[str, dict[str, Any]] = {}
        self._reviews: dict[str, IntegrationReview] = {}
        self._adapters: dict[str, str] = {}
        self._adapters_by_candidate: dict[int, str] = {}

    async def _emit(self, topic: str, **payload: Any) -> None:
        await self.events.emit(topic, **payload)

    def _kv_key(self, name: str, suffix: str) -> str:
        return f"integration:{name}:{suffix}"

    async def _kv_get(self, key: str) -> Any:
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(row["value"]) if row else None

    async def _kv_set(self, key: str, value: Any) -> None:
        await db.execute(
            "INSERT OR REPLACE INTO kv(key,value,updated_at) VALUES(?,?,?)",
            (key, json.dumps(value, default=str), utcnow()),
        )

    async def _skill_history(self, name: str, version: str, action: str, detail: Any = None) -> None:
        await db.execute(
            "INSERT INTO skill_history(skill_name,version,action,detail,changed_at) VALUES(?,?,?,?,?)",
            (name, version, action, json.dumps(detail or {}, default=str, sort_keys=True), utcnow()),
        )

    @staticmethod
    def _resolve_name(candidate: dict[str, Any] | str) -> str:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
            return candidate["name"]
        raise ValueError("candidate must be a mapping with a 'name' or a registered name")

    # -- 1. discovery --------------------------------------------------------

    async def discover(self, source: Any) -> list[dict[str, Any]]:
        """Discover candidates from a caller-supplied local JSON list/file.

        No remote source is contacted. Invalid entries are skipped.
        """
        loaded: Any = source
        if isinstance(source, (str, Path)):
            text = str(source)
            path = Path(text)
            if len(text) < 4096 and path.is_file():
                text = path.read_text(encoding="utf-8")
            try:
                loaded = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ValueError(f"invalid candidate source: {exc}") from exc
        if isinstance(loaded, dict):
            loaded = loaded.get("candidates", loaded.get("integrations", [loaded]))
        if not isinstance(loaded, (list, tuple)):
            raise ValueError("source must be a list of candidates, a JSON list/file, or a mapping with 'candidates'")
        candidates: list[dict[str, Any]] = []
        for item in loaded:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                continue
            candidate = _clean_candidate(item)
            candidate.setdefault("source", "external")
            self._candidates[candidate["name"]] = candidate
            candidates.append(candidate)
        await self._emit("integration.discovered", count=len(candidates), names=[c["name"] for c in candidates])
        return candidates

    # -- 2-6. evaluation -----------------------------------------------------

    async def evaluate(self, candidate: dict[str, Any] | str) -> IntegrationReview:
        name = self._resolve_name(candidate)
        if isinstance(candidate, str):
            if name not in self._candidates:
                raise KeyError(f"unknown candidate: {name}")
            candidate = self._candidates[name]
        candidate = _clean_candidate(candidate)
        self._candidates[name] = candidate
        compatibility: CompatibilityReport = translate_manifest(candidate)
        reasons: list[str] = []
        security_warnings = list(compatibility.security_warnings)
        warnings = list(compatibility.warnings)

        version = str(candidate.get("version") or (compatibility.manifest.version if compatibility.manifest else "1.0.0"))
        license_text = candidate.get("license", "")
        license_status, license_ok, license_note = _check_license(license_text)
        reasons.append(license_note)

        deps = _dependency_check(candidate, NATIVE_SKILL_NAMES)
        security_flags = list(compatibility.errors)
        if security_flags:
            reasons.extend(security_flags)

        score = 5
        if security_flags:
            score = 100
        declared = candidate.get("permissions") or []
        if isinstance(declared, list):
            for permission in declared:
                if isinstance(permission, str) and any(
                    token in permission.lower() for token in ("write", "delete", "admin", "net:", "network")
                ):
                    score += 10
                    security_warnings.append(f"elevated permission requested: {permission}")
        if any(candidate.get(key) for key in ("endpoint", "url", "base_url")):
            score += 15
            security_warnings.append("external network endpoint declared but not verified")
        if license_status == "unknown":
            score += 15
        elif license_status == "restricted":
            score += 25
        if deps["missing"] or deps["invalid"]:
            score += 10
            reasons.append("some dependencies are missing or invalid")
        if candidate.get("official") and candidate.get("verification") != "verified":
            score += 15
            warnings.append("official status is claimed by the candidate but not verified")
        stated_risk = str(candidate.get("risk", "")).lower()
        if stated_risk == "high":
            score += 40
        elif stated_risk == "medium":
            score += 15
        # Availability gates are conditions, not grants, but gates that depend
        # on environment/config/network state are an untrusted claim and must
        # raise the review bar (mirrors SkillRegistry._risk_for).
        gates = (compatibility.manifest.gates if compatibility.manifest else None) or candidate.get("gates") or {}
        if isinstance(gates, dict) and gates:
            gate_text = json.dumps(gates, default=str).lower()
            if any(token in gate_text for token in ("env", "config", "network", "credential")):
                score += 60
                security_warnings.append("manifest declares environment/config/network gates; requires review")
                warnings.append("manifest declares environment/config/network gates; requires review")
            else:
                score += 20
                warnings.append("manifest declares availability gates; requires review")
        if not candidate.get("tests"):
            score += 5
            warnings.append("candidate supplies no verification tests")
        score = max(0, min(100, score))
        risk = RiskLevel.HIGH if score >= 60 else (RiskLevel.MEDIUM if score >= 25 else RiskLevel.LOW)

        if security_flags or not compatibility.compatible:
            classification = "deferred"
            verdict = "reject"
        else:
            if candidate.get("source") == "native":
                classification = "native"
            elif candidate.get("reimplementation"):
                classification = "independent-reimplementation"
            elif candidate.get("official") and candidate.get("verification") == "verified":
                classification = "official-integration"
                warnings.append("official claim comes from the candidate and is not independently verified")
            else:
                classification = "compatible-adapter"
            if risk is RiskLevel.HIGH or license_status == "restricted" or candidate.get("requires_approval"):
                verdict = "review"
            elif not license_ok or deps["missing"] or deps["invalid"]:
                verdict = "review"
            else:
                verdict = "approve"

        requires_approval = bool(
            risk is RiskLevel.HIGH or license_status == "restricted" or candidate.get("requires_approval")
        )
        review = IntegrationReview(
            name=name,
            classification=classification,
            license=str(license_text or "unknown"),
            license_status=license_status,
            license_compatible=license_ok,
            risk_score=score,
            risk_level=risk,
            security_warnings=security_warnings,
            dependency_check=deps,
            compatibility=compatibility.to_dict(),
            verdict=verdict,
            requires_approval=requires_approval,
            reasons=reasons + warnings,
            version=version,
        )
        self._reviews[name] = review
        await self._kv_set(self._kv_key(name, "review"), review.to_dict())
        await self._emit(
            "integration.evaluated", name=name, verdict=verdict,
            classification=classification, risk=risk.value, risk_score=score,
        )
        return review

    # -- 7. adapter generation ----------------------------------------------

    async def generate_adapter(self, candidate: dict[str, Any] | str) -> str:
        """Generate a declarative, restricted Python adapter source string."""
        name = self._resolve_name(candidate)
        review = self._reviews.get(name) or await self.evaluate(candidate)
        if review.verdict == "reject":
            raise ValueError(f"refusing to generate adapter for unsafe candidate: {name}; " + "; ".join(review.reasons[:5]))
        translation = translate_manifest(review.compatibility.get("translated") or self._candidates.get(name, {}))
        manifest = translation.manifest or SkillManifest(
            name=name, description="Unverified external skill", source="factory"
        )
        # Use Python literals (not JSON): the generated module is really
        # executed inside the isolated child, where JSON `null` is a NameError.
        manifest_json = repr(manifest.model_dump(mode="json"))
        review_json = repr(review.model_dump(mode="json"))
        source = f'''"""Generated LEO adapter for {json.dumps(name)}.

Restricted sandbox contract: declarative data plus a validation-only handler.
No dynamic imports, no shell invocation, no network calls and no candidate code
are executed. Generated code is UNVERIFIED until an isolated test passes and a
registration is explicitly approved.
"""

from __future__ import annotations

from typing import Any

ADAPTER_KIND = "declarative"
ADAPTER_NAME = {json.dumps(name)}
REQUIRES_APPROVAL = {bool(review.requires_approval)!r}
MANIFEST: dict[str, Any] = {manifest_json}
REVIEW: dict[str, Any] = {review_json}

# Execution primitives are never accepted from an external manifest.
FORBIDDEN_PRIMITIVES = ("eval", "exec", "shell", "subprocess", "dynamic_import", "command")


def validate(arguments: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors; perform no side effects."""
    schema = MANIFEST.get("input_schema", {{}})
    properties = schema.get("properties", {{}})
    errors: list[str] = []
    for field in schema.get("required", []):
        if field not in arguments:
            errors.append(f"{{field}}: required")
    for key, value in arguments.items():
        spec = properties.get(key, {{}})
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            errors.append(f"{{key}}: expected string")
        if expected == "object" and not isinstance(value, dict):
            errors.append(f"{{key}}: expected object")
        if expected == "integer" and not isinstance(value, int):
            errors.append(f"{{key}}: expected integer")
    return errors


async def handle(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a call and return an inert envelope (no external operation)."""
    errors = validate(arguments)
    if errors:
        return {{"status": "invalid", "adapter": ADAPTER_NAME, "errors": errors}}
    return {{
        "status": "unverified",
        "adapter": ADAPTER_NAME,
        "arguments": dict(arguments),
        "note": "declarative adapter; no external service is contacted",
    }}
'''
        # The generated module must parse and must contain no execution primitive calls.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "compile", "__import__"}:
                    raise ValueError("generated adapter violates the restricted sandbox contract")
        self._adapters[name] = source
        self._adapters_by_candidate[id(candidate)] = name if isinstance(candidate, dict) else name
        await self._kv_set(self._kv_key(name, "adapter"), {"version": review.version, "source": source})
        await self._skill_history(name, review.version, "adapter.generated", {"classification": review.classification})
        await self._emit("integration.adapter.generated", name=name, classification=review.classification)
        return source

    def _adapter_for(self, name: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        source = self._adapters.get(name)
        if source is None:
            raise KeyError(f"no generated adapter for {name}; call generate_adapter first")
        review_row = self._reviews.get(name)
        review = review_row.model_dump(mode="json") if review_row else {}
        translated = (review.get("compatibility") or {}).get("translated") or {}
        return source, translated, review

    # -- 8-9. isolated test --------------------------------------------------

    @staticmethod
    def _test_source(test_callable: Callable[..., Any]) -> tuple[str, str, bool]:
        """Serialize the test callable so it can run inside the child process."""
        if not inspect.isfunction(test_callable):
            raise AdapterTestError(
                "adapter_test_callable_invalid",
                "test callable must be a Python function with available source",
            )
        try:
            source = textwrap.dedent(inspect.getsource(test_callable))
        except (OSError, TypeError) as exc:
            raise AdapterTestError(
                "adapter_test_callable_source_unavailable",
                f"cannot serialize test callable: {exc}",
            ) from exc
        return source, test_callable.__name__, inspect.iscoroutinefunction(test_callable)

    @staticmethod
    def _run_isolated(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Execute adapter + test source in a scrubbed, resource-bounded child."""
        env = {"PATH": "/usr/bin:/bin", "LANG": "C", "PYTHONIOENCODING": "utf-8"}
        if os.name == "nt":  # pragma: no cover - Windows fallback
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE, "env": env, "close_fds": True,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _ISOLATED_HARNESS], **kwargs
        )

        def _kill() -> None:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:  # pragma: no cover - Windows fallback
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()

        data = json.dumps(payload, default=str).encode("utf-8")
        try:
            stdout, stderr = proc.communicate(input=data, timeout=max(timeout, 0.001))
        except subprocess.TimeoutExpired:
            _kill()
            stdout, stderr = proc.communicate()
            return {
                "passed": False, "details": None,
                "error": f"adapter test timed out after {timeout}s in isolated process",
                "failure": {
                    "code": "isolation_timeout", "timeout_seconds": timeout,
                    "stderr": stderr.decode("utf-8", "replace")[-_ISOLATED_OUTPUT_LIMIT:],
                },
                "source_sha256": "", "stdout": "",
            }
        err_text = stderr.decode("utf-8", "replace")[-_ISOLATED_OUTPUT_LIMIT:]
        if proc.returncode != 0:
            return {
                "passed": False, "details": None,
                "error": f"isolated adapter process exited with {proc.returncode}",
                "failure": {"code": "isolation_process_failed", "exit_code": proc.returncode,
                            "stderr": err_text},
                "source_sha256": "", "stdout": "",
            }
        lines = stdout.decode("utf-8", "replace").splitlines()
        try:
            result = json.loads(lines[-1])
            if not isinstance(result, dict):
                raise ValueError("result is not an object")
            return result
        except (ValueError, IndexError) as exc:
            return {
                "passed": False, "details": None,
                "error": f"invalid isolated adapter result: {exc}",
                "failure": {"code": "isolation_result_invalid", "message": str(exc),
                            "stdout": stdout.decode("utf-8", "replace")[-_ISOLATED_OUTPUT_LIMIT:],
                            "stderr": err_text},
                "source_sha256": "", "stdout": "",
            }

    async def _finish_test(
        self,
        name: str,
        *,
        passed: bool,
        error: str | None,
        details: Any,
        source_sha256: str,
        review_version: str,
        mode: str,
        timeout: float,
        started: float,
        failure: dict[str, Any] | None,
    ) -> AdapterTestReport:
        report = AdapterTestReport(
            name=name, passed=bool(passed), error=error, details=_json_safe(details),
            source_sha256=source_sha256, review_version=review_version, mode=mode,
            failure=failure, timeout_seconds=timeout,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        await self._kv_set(self._kv_key(name, "test"), report.model_dump(mode="json"))
        await self._skill_history(name, review_version, "adapter.tested", {
            "passed": report.passed, "error": error, "source_sha256": source_sha256,
            "review_version": review_version, "mode": mode,
        })
        await self._emit(
            "integration.adapter.tested", name=name, passed=report.passed,
            source_sha256=source_sha256, review_version=review_version, mode=mode,
        )
        return report

    async def test_adapter(
        self,
        name: str | dict[str, Any],
        test_callable: Callable[..., Any],
        timeout: float | None = None,
        *,
        trusted_internal: bool = False,
    ) -> AdapterTestReport:
        """Run the generated adapter in a scrubbed OS child process.

        The adapter source and the test callable source are compiled inside a
        fresh ``python -I`` process with no inherited environment, bounded
        CPU/memory and a hard wall-clock timeout. In-process execution is only
        available with an explicit trusted-internal opt-in on both the factory
        and this call.
        """
        resolved = self._resolve_name(name)
        timeout = timeout if timeout is not None else self.timeout_seconds
        started = time.perf_counter()
        mode = "trusted-in-process" if trusted_internal else "isolated"
        try:
            source, manifest, review = self._adapter_for(resolved)
        except KeyError as exc:
            return await self._finish_test(
                resolved, passed=False, error=str(exc), details=None,
                source_sha256="", review_version="", mode=mode, timeout=timeout,
                started=started, failure={"code": "adapter_source_missing", "message": str(exc)},
            )
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        review_version = str(review.get("version") or "1.0.0")

        if trusted_internal and not self.trusted_internal:
            return await self._finish_test(
                resolved, passed=False, error="in-process tests require the trusted internal flag",
                details=None, source_sha256=source_sha256, review_version=review_version,
                mode=mode, timeout=timeout, started=started,
                failure={"code": "trusted_internal_flag_required", "mode": mode},
            )
        test_source = ""
        test_name = ""
        test_is_async = inspect.iscoroutinefunction(test_callable)
        if not trusted_internal:
            try:
                test_source, test_name, test_is_async = self._test_source(test_callable)
            except AdapterTestError as exc:
                return await self._finish_test(
                    resolved, passed=False, error=str(exc), details=None,
                    source_sha256=source_sha256, review_version=review_version,
                    mode=mode, timeout=timeout, started=started, failure=exc.to_dict(),
                )

        if trusted_internal:
            raw: Any = None
            failure: dict[str, Any] | None = None
            error: str | None = None
            adapter = RestrictedAdapter(resolved, source, manifest, review)
            try:
                params = [
                    p for p in inspect.signature(test_callable).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
                ]
            except (TypeError, ValueError):
                params = []
            call = (lambda: test_callable()) if not params else (lambda: test_callable(adapter))

            async def _invoke() -> Any:
                if test_is_async:
                    result = call()
                else:
                    result = await asyncio.to_thread(call)
                if inspect.isawaitable(result):
                    return await result
                return result

            try:
                raw = await asyncio.wait_for(_invoke(), timeout=timeout)
            except asyncio.TimeoutError:
                error = f"adapter test timed out after {timeout}s"
                failure = {"code": "trusted_test_timeout", "timeout_seconds": timeout}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failure = {"code": "trusted_test_error", "message": error}
            if error is None:
                passed, failure = _test_passed(raw)
                error = None if passed else "test callable reported a non-passing result"
            else:
                passed = False
        else:
            payload = {
                "name": resolved, "source": source, "source_sha256": source_sha256,
                "review_version": review_version, "manifest": manifest, "review": review,
                "test_source": test_source, "test_name": test_name,
                "test_is_async": test_is_async,
                "cpu_seconds": max(1, int(math.ceil(timeout)) + 1),
                "memory_bytes": _ISOLATED_MEMORY_BYTES,
            }
            try:
                child = await asyncio.wait_for(
                    asyncio.to_thread(self._run_isolated, payload, timeout), timeout + 0.5
                )
            except asyncio.TimeoutError:
                child = {
                    "passed": False, "details": None,
                    "failure": {"code": "isolation_timeout", "timeout_seconds": timeout},
                }
            raw = child.get("details")
            passed, default_failure = _test_passed(raw)
            failure = child.get("failure") if isinstance(child.get("failure"), dict) else default_failure
            if passed and child.get("source_sha256") != source_sha256:
                passed = False
                failure = {"code": "adapter_source_hash_mismatch",
                           "expected": source_sha256, "actual": child.get("source_sha256")}
            if passed and str(child.get("review_version")) != review_version:
                passed = False
                failure = {"code": "review_version_mismatch",
                           "expected": review_version, "actual": child.get("review_version")}
            error = None if passed else str(child.get("error") or (failure or {}).get("code") or "adapter test failed")

        return await self._finish_test(
            resolved, passed=passed, error=error, details=raw,
            source_sha256=source_sha256, review_version=review_version,
            mode=mode, timeout=timeout, started=started, failure=failure,
        )

    # -- 10-12. approval, registration, rollback -----------------------------

    async def _authorize_registration(
        self, review: IntegrationReview, approval_id: str | None
    ) -> None:
        decision = await self.permissions.authorize(
            self.actor,
            "integration.register",
            review.name,
            risk=RiskLevel.HIGH if review.requires_approval else RiskLevel.LOW,
            approval_id=approval_id,
            integration=review.name,
            version=review.version,
        )
        if decision.requires_approval:
            raise ApprovalRequired(
                decision.approval_id or "",
                f"high-risk integration registration requires approval: {review.name}",
            )
        if not decision.allowed:
            raise PermissionError(decision.reason)

    @staticmethod
    def _minimum_permissions(requested: list[str], declared: list[str]) -> list[str]:
        """Never grant write/admin/execution capabilities to a new integration."""
        allowed = set(requested or []) & set(declared or []) & _SAFE_MINIMUM_PERMISSIONS
        return sorted(allowed)

    async def versions(self, name: str) -> list[dict[str, Any]]:
        values = await self._kv_get(self._kv_key(name, "versions"))
        return values if isinstance(values, list) else []

    async def _require_test_report(
        self,
        name: str,
        source: str,
        review_version: str,
        supplied: dict[str, Any] | AdapterTestReport | None = None,
    ) -> dict[str, Any]:
        """Registration is fail-closed: the test must match this exact source."""
        expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        report = supplied
        if isinstance(report, AdapterTestReport):
            report = report.model_dump(mode="json")
        if report is None:
            report = await self._kv_get(self._kv_key(name, "test"))
        if not isinstance(report, dict):
            raise AdapterTestError(
                "adapter_test_missing",
                f"no adapter test report for {name}; call test_adapter first",
                {"name": name, "expected_source_sha256": expected_hash,
                 "expected_review_version": review_version},
            )
        actual_hash = str(report.get("source_sha256") or "")
        actual_version = str(report.get("review_version") or "")
        binding = {
            "name": name, "expected_source_sha256": expected_hash,
            "actual_source_sha256": actual_hash,
            "expected_review_version": review_version,
            "actual_review_version": actual_version,
            "mode": report.get("mode"),
            "error": report.get("error"),
            "failure": report.get("failure"),
            "details": _json_safe(report.get("details")),
        }
        if actual_hash != expected_hash:
            raise AdapterTestError(
                "adapter_test_stale_source",
                f"adapter test report is bound to a different source for {name}", binding,
            )
        if actual_version != review_version:
            raise AdapterTestError(
                "adapter_test_stale_review",
                f"adapter test report is bound to a different review version for {name}", binding,
            )
        if report.get("passed") is not True:
            raise AdapterTestError(
                "adapter_test_failed",
                f"adapter test did not pass for {name}", binding,
            )
        return report

    async def register(
        self,
        name: str,
        permissions: list[str] | None = None,
        *,
        approval_id: str | None = None,
        test_report: dict[str, Any] | AdapterTestReport | None = None,
        provenance_approval_id: str | None = None,
        **context: Any,
    ) -> SkillRecord:
        """Register a reviewed, tested integration with minimum permissions.

        High-risk registrations and the external manifest's provenance each
        require their own resolved approval. Pass both approval IDs on the
        retry; the provenance grant is issued only by
        ``SkillRegistry.approve_external`` and is audited there.
        """
        review = self._reviews.get(name) or await self.evaluate(name)
        if review.verdict == "reject":
            raise ValueError(f"refusing to register unsafe candidate: {name}; " + "; ".join(review.reasons[:5]))

        candidate = self._candidates.get(name, {})
        if name not in self._adapters:
            await self.generate_adapter(candidate or name)
        report = await self._require_test_report(
            name, self._adapters[name], str(review.version), test_report
        )
        if review.requires_approval or review.risk_level is RiskLevel.HIGH:
            await self._authorize_registration(review, approval_id)
        translation = translate_manifest(candidate or {"name": name, "description": review.name})
        base = translation.manifest or SkillManifest(name=name, description="Unverified external skill")
        declared = list(base.permissions)
        minimal = self._minimum_permissions(list(permissions or []), declared)
        manifest = base.model_copy(deep=True)
        manifest.permissions = minimal
        manifest.version = review.version
        manifest.source = f"factory:{review.classification}"
        manifest.documentation = (
            (manifest.documentation or "") + "\nRegistered by IntegrationFactory with minimum permissions; "
            "verification: not-verified. No external integration is claimed to work."
        )

        existing = await self.registry.get(name)
        record = await (
            self.registry.replace(manifest, handler=None, reason="integration registration")
            if existing else self.registry.register(manifest, handler=None)
        )
        # An imported manifest is inert until an audited, HIGH-risk provenance
        # approval is resolved. Never launder the integration.register approval
        # into this separate grant.
        approved = await self.registry.approve_external(
            name,
            provenance_approval_id,
            reason=f"adapter test {str(report.get('source_sha256') or '')[:16]}",
            verified_by=self.actor,
        )
        approved_manifest = manifest.model_copy(update={
            "provenance_verified": True,
            "provenance_approval_id": approved.provenance_approval_id,
            "provenance": dict(approved.provenance),
        })
        entry = {
            "version": manifest.version,
            "action": "integration.registered",
            "manifest": approved_manifest.model_dump(mode="json"),
            "review": {
                "classification": review.classification,
                "verdict": review.verdict,
                "risk": review.risk_level.value,
                "risk_score": review.risk_score,
            },
            "permissions": minimal,
            "approval_id": approval_id,
            "provenance_approval_id": approved.provenance_approval_id,
            "test": {
                "source_sha256": report.get("source_sha256"),
                "review_version": report.get("review_version"),
                "mode": report.get("mode"),
            },
            "changed_at": utcnow(),
        }
        history = await self.versions(name)
        history.append(entry)
        await self._kv_set(self._kv_key(name, "versions"), history)
        await self._skill_history(name, manifest.version, "integration.registered", {
            "permissions": minimal, "classification": review.classification, "approval_id": approval_id,
            "provenance_approval_id": approved.provenance_approval_id,
        })
        now = utcnow()
        await db.execute(
            "INSERT OR REPLACE INTO integrations(name,kind,status,config_ref,capabilities,verification,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (name, review.classification, "registered", None,
             json.dumps(sorted(minimal)), "not-verified", now, now),
        )
        await self._emit(
            "integration.registered", name=name, version=manifest.version,
            classification=review.classification, permissions=minimal,
            provenance_approval_id=approved.provenance_approval_id,
        )
        return approved

    async def rollback(self, name: str, version: str) -> SkillRecord:
        """Restore a previously registered manifest version from durable history."""
        history = await self.versions(name)
        entry = next((item for item in history if str(item.get("version")) == str(version)), None)
        if entry is None:
            raise KeyError(f"unknown version {version} for integration {name}")
        current = await self.registry.get(name)
        if current is None:
            raise KeyError(f"integration is not registered: {name}")
        if current.version == str(version):
            raise ValueError(f"{name} is already at version {version}")
        manifest = SkillManifest.model_validate(entry["manifest"])
        record = await self.registry.replace(manifest, handler=None, reason=f"rollback to {version}")
        rollback_entry = {
            "version": current.version,
            "action": "integration.rolled_back",
            "rolled_back_to": str(version),
            "changed_at": utcnow(),
        }
        history.append(rollback_entry)
        await self._kv_set(self._kv_key(name, "versions"), history)
        await self._skill_history(name, str(version), "integration.rolled_back", {
            "from": current.version, "to": str(version),
        })
        await db.execute(
            "UPDATE integrations SET status='rolled_back', updated_at=? WHERE name=?", (utcnow(), name)
        )
        await self._emit("integration.rolled_back", name=name, version=str(version), from_version=current.version)
        return record
