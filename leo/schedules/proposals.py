"""Durable proposal reservation and commit metadata for ScheduleService.

A model request is first recorded as a :class:`ScheduleProposal`, never as a
schedule. The proposal reserves the request key, the active name and the
prospective schedule id in one transaction. Commit-only metadata (the computed
next run, display text and config revision) travels inside the proposal's
``spec.config`` under a private key so a later human decision commits exactly
what the card showed, without a second scheduling store.

The SQL authority remains :class:`leo.schedules.store.ScheduleStore`;
this module only builds domain objects and derives human-readable edit copy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from .domain import (
    DeliveryTarget,
    ExecutionSpec,
    ScheduleExpression,
    ScheduleProposal,
    ScheduleSpec,
    ScheduleView,
    canonical_json,
    sha256_digest,
)

COMMIT_META_KEY = "__leo_commit"
_PRIVATE_PREFIX = "__leo_"


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def commit_meta(proposal: ScheduleProposal) -> dict[str, Any]:
    raw = proposal.spec.config.get(COMMIT_META_KEY, {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def with_commit_meta(
    spec: ScheduleSpec, result: Any, *, actor: str | None = None
) -> ScheduleSpec:
    """Return a copy of *spec* carrying the compiler's commit-time facts."""
    meta = {
        "next_run_at": _iso(getattr(result, "next_run_at", None)),
        "first_run_at": _iso(getattr(result, "first_run_at", None)),
        "config_revision": int(getattr(result, "config_revision", 1) or 1),
        "display": str(getattr(result, "display", "") or ""),
        "config_digest": str(getattr(result, "config_digest", "") or ""),
        "reanchors": bool(getattr(result, "reanchors", True)),
    }
    if actor:
        meta["actor"] = str(actor)
    # The compiler exposes the full semantic payload as ``result.config``;
    # ``spec.config`` itself carries only a prior row's base config. Persisting
    # ``result.config`` is what lets a later edit reconstruct the schedule and
    # lets Phase 6 adapters read every execution field.
    config = strip_commit_meta(getattr(result, "config", None) or spec.config)
    config[COMMIT_META_KEY] = meta
    return spec.model_copy(update={"config": config})


def strip_commit_meta(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop private commit keys before a spec is persisted or displayed."""
    return {
        str(key): value
        for key, value in dict(config or {}).items()
        if not str(key).startswith(_PRIVATE_PREFIX)
    }


def display_expression(spec: ScheduleSpec) -> str:
    expression = spec.expression
    if expression.kind == "once":
        return f"once at {expression.run_at or 'an unset time'}"
    return str(expression.expression or spec.config.get("schedule", {}).get("display") or "")


def objective_for(spec: ScheduleSpec) -> str:
    """The durable task objective for one occurrence.

    Phase 6 adapters consume the richer execution fields; the task engine still
    needs a non-empty objective, so a script/skill-only schedule gets a stable
    human schedule description instead of an empty string.
    """
    prompt = str(spec.prompt or "").strip()
    if prompt:
        return prompt
    execution = spec.execution
    if execution.skills:
        return "Use skill: " + ", ".join(str(item) for item in execution.skills)
    if execution.script:
        return f"Run script {execution.script}"
    return f"Scheduled job {spec.name}"


def schedule_payload(spec: ScheduleSpec) -> dict[str, Any]:
    """The legacy-compatible schedule payload plus every execution field.

    ``Scheduler`` pops ``objective`` and forwards the rest to the task metadata,
    so later execution adapters read exactly the validated spec that was shown.
    """
    return {
        "objective": objective_for(spec),
        "prompt": spec.prompt,
        "name": spec.name,
        "schedule": {
            "kind": spec.expression.kind,
            "expression": spec.expression.expression,
            "run_at": spec.expression.run_at,
            "timezone": spec.expression.timezone,
        },
        "repeat_limit": spec.repeat_limit,
        "delivery": [target.model_dump(mode="json") for target in spec.delivery],
        "failure_delivery": [
            target.model_dump(mode="json") for target in spec.failure_delivery
        ],
        "execution": spec.execution.model_dump(mode="json"),
        # Frozen at creation/update time so a later skill edit cannot change
        # what an already-created schedule injects.
        "skill_snapshot": dict(spec.config.get("skill_snapshot") or {}),
        "paused": spec.paused,
        "paused_reason": spec.paused_reason,
        "catch_up": spec.catch_up,
        "spec_digest": spec.digest(),
    }


def spec_from_view(view: ScheduleView) -> ScheduleSpec:
    """Reconstruct the latest validated spec from a durable row.

    New rows carry the compiler's semantic config. Legacy rows lack it; they
    are reconstructed from the same legacy columns ``Scheduler`` already reads,
    so an edit never invents fields that were not stored.
    """
    config = strip_commit_meta(view.config)
    schedule = config.get("schedule")
    schedule = dict(schedule) if isinstance(schedule, Mapping) else {}
    kind = str(schedule.get("kind") or view.schedule_type or view.kind or "cron")
    if kind == "once":
        run_at = schedule.get("run_at") or view.run_at or view.expression
        expression = ScheduleExpression(
            kind="once", run_at=str(run_at) if run_at else None, timezone=view.timezone
        )
    else:
        body = schedule.get("expression")
        if body is None:
            body = view.expression
        expression = ScheduleExpression(
            kind=kind if kind in ("cron", "interval") else "cron",
            expression=str(body) if body not in (None, "") else None,
            timezone=schedule.get("timezone") or view.timezone,
        )

    delivery = _targets(config.get("delivery"))
    failure_delivery = _targets(config.get("failure_delivery"))
    execution_raw = config.get("execution")
    execution = (
        ExecutionSpec.model_validate(execution_raw)
        if isinstance(execution_raw, Mapping)
        else ExecutionSpec()
    )
    payload = view.payload if isinstance(view.payload, Mapping) else {}
    prompt = str(config.get("prompt") or payload.get("objective") or "")
    repeat = config.get("repeat_limit", view.repeat_limit)
    paused = bool(config.get("paused", view.status == "paused" or not view.enabled))
    return ScheduleSpec(
        name=view.name,
        expression=expression,
        prompt=prompt,
        repeat_limit=int(repeat) if isinstance(repeat, int) else None,
        delivery=delivery,
        failure_delivery=failure_delivery,
        execution=execution,
        paused=paused,
        paused_reason=config.get("paused_reason"),
        catch_up=config.get("catch_up"),
        config=config,
    )


def _targets(value: Any) -> list[DeliveryTarget]:
    if not isinstance(value, (list, tuple)):
        return []
    targets: list[DeliveryTarget] = []
    for item in value:
        if isinstance(item, DeliveryTarget):
            targets.append(item)
        elif isinstance(item, Mapping):
            try:
                targets.append(DeliveryTarget.model_validate(dict(item)))
            except ValueError:
                continue
    return targets


def build_create_proposal(
    *,
    spec: ScheduleSpec,
    principal: str,
    origin: str | None,
    conversation_id: str | None,
    run_id: str | None,
    tool_call_id: str | None,
    request_key: str | None,
    digest: str,
    schedule_id: str,
) -> ScheduleProposal:
    return ScheduleProposal(
        active_name=spec.name,
        principal=principal,
        operation="create",
        request_key=request_key,
        status="pending",
        digest=digest,
        spec=spec,
        origin=origin,
        conversation_id=conversation_id,
        run_id=run_id,
        tool_call_id=tool_call_id,
        config_revision=1,
        schedule_id=schedule_id,
    )


def build_update_proposal(
    *,
    spec: ScheduleSpec,
    principal: str,
    origin: str | None,
    conversation_id: str | None,
    run_id: str | None,
    tool_call_id: str | None,
    request_key: str | None,
    digest: str,
    schedule_id: str,
    base_revision: int,
) -> ScheduleProposal:
    # An edit never reserves the active name: the schedule already owns it and
    # a pending edit must not block unrelated creates. The CAS revision is the
    # binding that makes a stale edit fail closed.
    return ScheduleProposal(
        active_name=None,
        principal=principal,
        operation="update",
        request_key=request_key,
        status="pending",
        digest=digest,
        spec=spec,
        origin=origin,
        conversation_id=conversation_id,
        run_id=run_id,
        tool_call_id=tool_call_id,
        config_revision=max(1, int(base_revision)),
        schedule_id=schedule_id,
    )


def build_remove_proposal(
    *,
    spec: ScheduleSpec,
    principal: str,
    origin: str | None,
    conversation_id: str | None,
    run_id: str | None,
    tool_call_id: str | None,
    request_key: str | None,
    digest: str,
    schedule_id: str,
    base_revision: int,
) -> ScheduleProposal:
    return ScheduleProposal(
        active_name=None,
        principal=principal,
        operation="remove",
        request_key=request_key,
        status="pending",
        digest=digest,
        spec=spec,
        origin=origin,
        conversation_id=conversation_id,
        run_id=run_id,
        tool_call_id=tool_call_id,
        config_revision=max(1, int(base_revision)),
        schedule_id=schedule_id,
    )


def remove_digest(spec: ScheduleSpec, schedule_id: str, base_revision: int) -> str:
    return sha256_digest(
        {
            "operation": "remove",
            "schedule_id": schedule_id,
            "config_revision": int(base_revision),
            "spec": spec.canonical_payload(),
        }
    )


def changes_between(base: ScheduleSpec, new: ScheduleSpec) -> list[str]:
    """Human-readable edit copy for an allow/deny card."""
    changes: list[str] = []
    if base.name != new.name:
        changes.append(f"name: {base.name} -> {new.name}")
    old_schedule = canonical_json(base.expression.canonical_payload())
    new_schedule = canonical_json(new.expression.canonical_payload())
    if old_schedule != new_schedule:
        changes.append(
            f"schedule: {display_expression(base)} -> {display_expression(new)}"
        )
    if base.prompt != new.prompt:
        changes.append("prompt changed")
    if base.repeat_limit != new.repeat_limit:
        changes.append(
            f"runs: {_repeat_text(base.repeat_limit)} -> {_repeat_text(new.repeat_limit)}"
        )
    if _targets_semantic(base.delivery) != _targets_semantic(new.delivery):
        changes.append("delivery target changed")
    if _targets_semantic(base.failure_delivery) != _targets_semantic(new.failure_delivery):
        changes.append("failure delivery target changed")
    old_execution = base.execution.model_dump(mode="json")
    new_execution = new.execution.model_dump(mode="json")
    for key in sorted(set(old_execution) | set(new_execution)):
        if old_execution.get(key) != new_execution.get(key):
            changes.append(f"{key} changed")
    if bool(base.paused) != bool(new.paused):
        changes.append("paused state changed")
    if base.paused_reason != new.paused_reason:
        changes.append("paused reason changed")
    if base.catch_up != new.catch_up:
        changes.append("catch-up changed")
    return changes or ["configuration changed"]


def _targets_semantic(targets: Sequence[DeliveryTarget]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in targets:
        data = target.model_dump(mode="json")
        data.pop("label", None)
        result.append(data)
    return result


def _repeat_text(value: int | None) -> str:
    return "forever" if value is None else str(value)


def related_schedule_ids(
    specs: Iterable[tuple[str, ScheduleSpec]],
) -> dict[str, Sequence[str]]:
    """Reference map used by the compiler's cycle detector."""
    return {
        schedule_id: tuple(str(item) for item in spec.execution.context_from)
        for schedule_id, spec in specs
    }


__all__ = [
    "COMMIT_META_KEY",
    "build_create_proposal",
    "build_remove_proposal",
    "build_update_proposal",
    "changes_between",
    "commit_meta",
    "display_expression",
    "objective_for",
    "related_schedule_ids",
    "remove_digest",
    "schedule_payload",
    "spec_from_view",
    "strip_commit_meta",
    "with_commit_meta",
]
