"""LLM consolidation planning — a proposal, never an automatic rewrite.

Near-duplicate groups come from the deterministic curator. For each group the
model is asked for ONE strict-JSON decision: keep the group as-is, or merge it
into a class-level umbrella with a bounded set of operations. The planner then:

* validates every field (name, 60-char description, content size/path/secret
  hygiene, absorbed subset),
* converts the decision into explicit operations (read-before-write, manage,
  archive), and
* stages ONE ``consolidate`` proposal through :mod:`leo.skills.staging`.

Nothing is applied here. ``/skills approve`` replays the operations through the
real tool executor and archives siblings as a reversible state transition.
Native/hub skills never enter a group, and deletion is not an operation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .. import db
from ..contracts import ChatMessage, ModelRequest, utcnow
from ..events import EventBus
from ..logging import get_logger
from ..skills import staging
from ..skills.sanitize import (
    MAX_DOCUMENTATION_CHARS,
    scan_text,
    validate_description,
    validate_relative_path,
)

_LOG = get_logger("review.consolidate")
_ACTOR = "curator:engine"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MAX_DOC = 6_000
_MAX_REFS = 10
_SYSTEM = (
    "You are LEO's skill consolidation planner. You are given a group of "
    "agent-authored skills that a deterministic similarity pass found related.\n\n"
    "Decide whether ONE class-level umbrella skill would serve the user better "
    "than the siblings. Merge only when a human maintainer would write these as "
    "one skill with labeled subsections. If the group is genuinely distinct, "
    "return {\"action\":\"keep\"}.\n\n"
    "Return STRICT JSON only, no prose and no code fence:\n"
    "{\"action\":\"merge\",\"umbrella\":{\"name\":\"lowercase-hyphenated\","
    "\"description\":\"<=60 chars, trigger first\",\"content\":\"full SKILL.md "
    "body\"},\"absorbed\":[\"skill-name\",...],\"references\":[{\"path\":"
    "\"references/topic.md\",\"content\":\"...\"}],\"reason\":\"one line\"}\n\n"
    "Rules: the umbrella name is class-level, never a PR/issue number or a "
    "fix-X/debug-Y artifact. The content must preserve each absorbed skill's "
    "unique rules and pitfalls (distilled, not pasted), and must not invent "
    "commands. Absorbed names must come from the group. Never include a native "
    "skill. No deletion operation exists; siblings are archived only after the "
    "umbrella lands."
)


def _parse_json(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class ConsolidationEngine:
    """Turn deterministic candidate groups into staged, reviewable plans."""

    def __init__(
        self,
        settings: Any,
        router: Any,
        skills: Any,
        curator: Any,
        events: EventBus | None = None,
    ) -> None:
        self.settings = settings
        self.router = router
        self.skills = skills
        self.curator = curator
        self.events = events

    @property
    def config(self) -> Any:
        return getattr(getattr(self.settings, "skills", None), "curator", None)

    @property
    def enabled(self) -> bool:
        cfg = self.config
        return bool(cfg is not None and getattr(cfg, "consolidate", False))

    def status(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "enabled": self.enabled,
            "max_groups": int(getattr(cfg, "max_groups", 3) or 3),
            "model": str(getattr(cfg, "model", "") or ""),
        }

    async def _records(self, names: list[str]) -> list[dict[str, Any]]:
        records = []
        for name in names:
            record = await self.skills.get(name)
            if record is None or record.created_by != "agent" or not record.enabled:
                continue
            try:
                files = [item["path"] for item in await self.skills.files(name)]
            except Exception:  # noqa: BLE001 - a missing bundle is not fatal
                files = []
            records.append({
                "name": record.name,
                "description": record.description,
                "documentation": str(record.documentation or "")[:_MAX_DOC],
                "files": files,
            })
        return records

    async def _plan(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if self.router is None:
            raise RuntimeError("model router is unavailable")
        cfg = self.config
        model = str(getattr(cfg, "model", "") or "").strip() or None
        request = ModelRequest(
            messages=[
                ChatMessage(role="system", content=_SYSTEM),
                ChatMessage(
                    role="user",
                    content="GROUP:\n" + json.dumps(records, ensure_ascii=False, default=str),
                ),
            ],
            task_category="skill_consolidate",
            agent_id="curator",
            model=model,
            max_tokens=4096,
        )
        response = await self.router.complete(request)
        plan = _parse_json(str(getattr(response, "content", "") or ""))
        if plan is None:
            raise ValueError("model returned no parseable JSON plan")
        return await self._operations(records, plan)

    async def _operations(
        self, records: list[dict[str, Any]], plan: dict[str, Any]
    ) -> dict[str, Any]:
        action = str(plan.get("action") or "").strip().lower()
        if action == "keep":
            return {"action": "keep", "reason": str(plan.get("reason") or "")[:200]}
        if action != "merge":
            raise ValueError(f"unsupported consolidation action: {action!r}")
        umbrella = plan.get("umbrella")
        if not isinstance(umbrella, dict):
            raise TypeError("merge plan has no umbrella object")
        name = str(umbrella.get("name") or "").strip()
        if not _NAME_RE.match(name):
            raise ValueError("umbrella name must be lowercase-hyphenated, 1-64 chars")
        description, errors = validate_description(str(umbrella.get("description") or ""))
        if errors:
            raise ValueError("; ".join(errors))
        content, scan_errors, _ = scan_text(
            str(umbrella.get("content") or ""), field="umbrella content"
        )
        if scan_errors:
            raise ValueError("; ".join(scan_errors))
        if not content.strip():
            raise ValueError("umbrella content is empty")
        if len(content) > MAX_DOCUMENTATION_CHARS:
            raise ValueError(
                f"umbrella content exceeds {MAX_DOCUMENTATION_CHARS} characters"
            )
        group = {record["name"] for record in records}
        absorbed = plan.get("absorbed")
        if not isinstance(absorbed, list) or not absorbed:
            raise ValueError("merge plan must name at least one absorbed skill")
        absorbed_names = [str(value).strip() for value in absorbed]
        unknown = [value for value in absorbed_names if value not in group]
        if unknown:
            raise ValueError(f"absorbed skills are not in this group: {unknown}")
        references = plan.get("references") or []
        if not isinstance(references, list) or len(references) > _MAX_REFS:
            raise ValueError(f"references must be a list of at most {_MAX_REFS} files")
        operations: list[dict[str, Any]] = []
        existing = name in group
        if not existing and await self.skills.get(name) is not None:
            raise ValueError(
                "umbrella name belongs to an existing skill outside this group"
            )
        if existing:
            operations.append({"op": "read", "name": name})
            operations.append({
                "op": "manage",
                "arguments": {
                    "action": "patch", "name": name, "content": content,
                },
            })
        else:
            operations.append({
                "op": "manage",
                "arguments": {
                    "action": "create", "name": name, "description": description,
                    "category": "general", "content": content,
                },
            })
        existing_paths = {
            str(path) for record in records for path in record.get("files", [])
        }
        for item in references:
            if not isinstance(item, dict):
                raise TypeError("each reference must be an object")
            path, path_error = validate_relative_path(str(item.get("path") or ""))
            if path_error:
                raise ValueError(path_error)
            file_content, ref_errors, _ = scan_text(
                str(item.get("content") or ""), field=path or "reference"
            )
            if ref_errors:
                raise ValueError("; ".join(ref_errors))
            if path in existing_paths:
                # Overwriting an existing supporting file needs a fresh read
                # in the same replayed context.
                operations.append({"op": "read", "name": name, "file_path": path})
            operations.append({
                "op": "manage",
                "arguments": {
                    "action": "write_file", "name": name,
                    "file_path": path, "file_content": file_content,
                },
            })
        for sibling in absorbed_names:
            if sibling != name:
                operations.append({"op": "archive", "name": sibling})
        return {
            "action": "merge",
            "umbrella": name,
            "absorbed": [value for value in absorbed_names if value != name],
            "reason": str(plan.get("reason") or "")[:300],
            "operations": staging.validate_plan(operations),
        }

    async def propose(
        self, *, dry_run: bool = False, max_groups: int | None = None
    ) -> dict[str, Any]:
        cfg = self.config
        if cfg is None or not getattr(cfg, "consolidate", False):
            return {
                "status": "disabled", "proposed": 0, "kept": 0, "failed": 0,
                "plans": [], "proposal_ids": [],
                "reason": "skills.curator.consolidate is false",
            }
        limit = max(1, int(max_groups or getattr(cfg, "max_groups", 3) or 3))
        groups = await self.curator.candidates(limit=limit * 3)
        counts = {
            "status": "completed", "proposed": 0, "kept": 0, "failed": 0,
            "dry_run": bool(dry_run), "plans": [], "proposal_ids": [],
        }
        for group in groups[:limit]:
            names = [str(value) for value in group.get("skills", [])]
            records = await self._records(names)
            if len(records) < 2:
                counts["kept"] += 1
                continue
            try:
                plan = await self._plan(records)
            except Exception as exc:  # noqa: BLE001 - one group cannot stop the pass
                counts["failed"] += 1
                _LOG.warning("consolidation plan failed for %s: %s", names, exc)
                continue
            if plan.get("action") == "keep":
                counts["kept"] += 1
                continue
            if dry_run:
                counts["proposed"] += 1
                counts["plans"].append({"group": names, **plan})
                continue
            try:
                proposal = await staging.stage_plan(
                    title=f"Consolidate {', '.join(names)}",
                    operations=list(plan["operations"]),
                    actor=_ACTOR,
                    reason=str(plan.get("reason") or "near-duplicate group"),
                )
            except Exception as exc:  # noqa: BLE001 - report, do not half-stage
                counts["failed"] += 1
                _LOG.warning("consolidation staging failed for %s: %s", names, exc)
                continue
            counts["proposed"] += 1
            counts["proposal_ids"].append(proposal["id"])
            counts["plans"].append({"group": names, **plan})
        await self._record(counts)
        return counts

    async def _record(self, counts: dict[str, Any]) -> None:
        try:
            await db.execute(
                "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    utcnow(), _ACTOR, "curator.consolidate", "skills", "low",
                    "proposed" if counts["proposed"] else counts["status"],
                    f"{counts['proposed']} plans, {counts['kept']} kept, {counts['failed']} failed",
                    json.dumps(counts, default=str),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - audit failure is logged
            _LOG.warning("could not audit consolidation pass: %s", exc)
        if self.events is not None:
            try:
                await self.events.emit(
                    "skill.consolidation.proposed",
                    proposed=counts["proposed"], kept=counts["kept"],
                    failed=counts["failed"], dry_run=counts["dry_run"],
                )
            except Exception as exc:  # noqa: BLE001 - telemetry must not fail
                _LOG.warning("could not emit consolidation event: %s", exc)


__all__ = ["ConsolidationEngine"]
