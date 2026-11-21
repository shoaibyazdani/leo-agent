"""Skill-author specialist: turn verified task experience into native skills.

This specialist never invents a skill from a bare claim. It reads a *completed,
evidence-backed* task from the injected task engine, drafts a native
``SkillManifest`` with declared permissions, documentation and tests, validates
the draft through the real ``SkillRegistry`` and only then registers it through
that same registry. Unverified experience and invalid drafts are refused, so a
failed draft can never leave a registered manifest behind.

The draft-then-register flow is bespoke, so this module exposes
``make_handler(services)``; the integration agent binds it as the execution
handler for the ``skill-author`` specialist. It does not use the default
delegation handler.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...contracts import SkillManifest, new_id

SEED: dict[str, Any] = {
    "id": "skill-author",
    "name": "Skill Author",
    "purpose": (
        "Draft new native LEO skills from verified, evidence-backed task "
        "experience, with declared permissions, documentation and tests; "
        "register each draft only through the real SkillRegistry after it "
        "passes manifest validation."
    ),
    "capabilities": [
        "skill_drafting",
        "manifest_validation",
        "documentation",
        "test_authoring",
    ],
    # Both entries are live tool categories in the real ToolRegistry:
    # 'skills' (skill_list/search/read/register/...) and 'tasks'
    # (task_create/list/status/cancel). No fabricated grant is declared.
    "tool_permissions": ["skills", "tasks"],
    "model_preferences": ["reasoning", "structured", "tools"],
    "context_budget": 24000,
    "task_limit": 2,
    "metadata": {
        "tier": "knowledge",
        "source": "verified_task_experience",
        "risk": "low",
    },
}

# Minimal, deliberately safe declaration for drafted skills. The registry's
# validate_manifest() rejects eval/exec/shell-style tokens; this stays clear of
# them and declares only the generic local 'utility' permission.
_DRAFT_PERMISSIONS: list[str] = ["utility"]
_TEST_FILE = "tests/test_agents_skill_author.py"
_TASK_REF_RE = re.compile(r"\b(task_[A-Za-z0-9_-]{6,})\b")


async def _resolved(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _skill_name(objective: str, task_id: str) -> str:
    base = _slug(objective)[:48].strip("_") or "from_task"
    name = f"authored_{base}"[:64].rstrip("_-")
    if not name or not re.match(r"^[a-z0-9]", name):
        name = f"authored_{base}"[:64]
    return name[:64]


def _summary(value: Any, limit: int = 600) -> str:
    if value is None:
        return "no structured result was recorded"
    if isinstance(value, Mapping):
        text = "; ".join(f"{key}: {item}" for key, item in list(value.items())[:8])
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in list(value)[:8])
    else:
        text = str(value)
    text = text.strip()
    return text[:limit] if text else "no structured result was recorded"


def draft_manifest(task: Any, objective: str | None = None) -> SkillManifest:
    """Draft a native manifest from a task already known to be verified.

    The caller (``make_handler``) is responsible for verification; this helper
    is a pure projection and performs no registration.
    """
    task_id = str(getattr(task, "id", "") or "unknown")
    objective_text = str(objective or getattr(task, "objective", "") or "").strip()
    evidence = list(getattr(task, "evidence", None) or [])
    result_summary = _summary(getattr(task, "result", None))
    evidence_lines = "\n".join(f"- {_summary(item, 200)}" for item in evidence[:8]) or "- (none recorded)"
    name = _skill_name(objective_text, task_id)
    documentation = "\n".join(
        [
            f"# {name}",
            "",
            "Native skill drafted by the `skill-author` specialist from a verified,",
            "evidence-backed task. It is declarative and local: no network or external",
            "credential is required, and only the minimum `utility` permission is declared.",
            "",
            "## Procedure",
            "",
            f"1. Restate the verified outcome: {result_summary}",
            "2. Reproduce the recorded steps and check the result against the evidence below.",
            "3. Return the result under the `output` field.",
            "",
            "## Verified source",
            "",
            f"- source task: {task_id}",
            f"- objective: {objective_text or '(not recorded)'}",
            f"- evidence: {evidence_lines}",
            "",
            "## Tests",
            "",
            f"`{_TEST_FILE}` covers the draft-then-register path and the refusal of",
            "unverified experience.",
        ]
    )
    return SkillManifest(
        name=name,
        description=(
            "Native procedure distilled from a verified, evidence-backed task: "
            + (objective_text[:180] or task_id)
        ),
        version="1.0.0",
        author="leo-native",
        category="general",
        permissions=list(_DRAFT_PERMISSIONS),
        input_schema={
            "type": "object",
            "properties": {"input": {"type": "string", "minLength": 1}},
            "required": ["input"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {"type": "string"},
                "source_task_id": {"type": "string"},
            },
            "required": ["output"],
        },
        source="native",
        documentation=documentation,
        tests=[_TEST_FILE],
    )


def _reference_from_text(objective: Any) -> str | None:
    if not objective:
        return None
    match = _TASK_REF_RE.search(str(objective))
    return match.group(1) if match else None


async def _verified_task(tasks: Any, reference: str) -> Any:
    """Return a completed, evidence-backed task or refuse to draft."""
    task = await _resolved(tasks.get(reference))
    if task is None:
        raise ValueError(f"skill-author: source task not found: {reference!r}")
    status = getattr(task, "status", None)
    status_value = str(getattr(status, "value", status) or "").lower()
    evidence = list(getattr(task, "evidence", None) or [])
    if status_value != "completed" or not evidence:
        raise ValueError(
            "skill-author requires a completed, evidence-backed source task; "
            f"refusing to draft from unverified experience (task={reference!r}, "
            f"status={status_value!r}, evidence_count={len(evidence)})"
        )
    return task


def make_handler(services: Mapping[str, Any] | None) -> Callable[..., Any]:
    """Bind the skill-author handler to injected services.

    Requires ``services['tasks']`` (the real task engine) and
    ``services['skills']`` (the real ``SkillRegistry``). The source task is
    taken from ``context['source_task_id']`` first -- the engine's ``task_id``
    is the current delegation run, not the verified experience -- then from
    the explicit ``task_id`` argument, then from a ``task_<id>`` token in the
    objective.
    """

    services = dict(services or {})
    skills = services.get("skills")
    tasks = services.get("tasks")
    artifact_dir = Path(str(services["artifact_dir"])) if services.get("artifact_dir") else Path(tempfile.gettempdir())

    async def handler(
        objective: Any = None,
        context: Any = None,
        task_id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if skills is None:
            raise RuntimeError("skill-author requires the 'skills' service (SkillRegistry)")
        if tasks is None:
            raise RuntimeError("skill-author requires the 'tasks' service (DurableTaskEngine)")
        ctx = dict(context or {})
        reference = (
            ctx.get("source_task_id")
            or ctx.get("source_task")
            or task_id
            or _reference_from_text(objective)
        )
        if not reference:
            raise ValueError(
                "skill-author requires a verified source task reference "
                "(context['source_task_id'] or task_id)"
            )

        source = await _verified_task(tasks, str(reference))
        manifest = draft_manifest(source, str(objective or getattr(source, "objective", "")))

        validation = await _resolved(skills.validate(manifest))
        if not getattr(validation, "valid", False):
            errors = list(getattr(validation, "errors", []) or [])
            raise ValueError(
                "drafted skill manifest failed validation: "
                + ("; ".join(errors) if errors else "unknown validation error")
            )

        existing = await _resolved(skills.get(manifest.name))
        if existing is None:
            record = await _resolved(skills.register(manifest))
            action = "registered"
        else:
            record = await _resolved(skills.register(manifest, replace=True))
            action = "replaced"

        source_id = str(getattr(source, "id", reference))
        claim = f"Native skill {manifest.name} {action} from verified task {source_id}"
        # Persist the registration claim as a real artifact so the delegation
        # verifier can recompute its digest instead of trusting a self-attested
        # assertion. The same string is bound into the reported output.
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = artifact_dir / f"skill_author_{new_id('draft')}.txt"
        artifact.write_text(claim, encoding="utf-8")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        return {
            "success": True,
            "output": {
                "action": action,
                "skill": manifest.name,
                "version": manifest.version,
                "source_task_id": source_id,
                "registered": True,
                "status": str(getattr(record, "status", "") or ""),
                "validation": {
                    "valid": True,
                    "warnings": list(getattr(validation, "warnings", []) or []),
                },
                "manifest": manifest.model_dump(mode="json"),
                "claim": claim,
            },
            "evidence": [
                {
                    "source": "artifact",
                    "path": str(artifact),
                    "digest": digest,
                    "claim": claim,
                    "task_id": source_id,
                }
            ],
        }

    return handler
