"""Model-facing schedule tools backed by the one ScheduleService authority.

Every mutation goes through :class:`leo.schedules.service.ScheduleService`; no
tool writes a schedule row directly. A model create/edit/remove either presents
a durable consent card on an answerable surface or fails closed to an
``approval_required`` proposal with no schedule write. Read-only calls and
manual run/cancel never invent execution evidence.

Schemas are neutral and closed (``additionalProperties=false``). Provider,
model, base_url and grace_minutes are operator/programmatic-only fields and are
not part of this surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import RiskLevel, ToolSpec, TurnContext
from ..schedules.compiler import MAX_PAUSE_REASON_CHARS
from ..schedules.outcome import ScheduleOutcome, record_schedule_outcome
from ..schedules.service import ScheduleCompileError, ScheduleServiceError

_SERVICE_KEYS = ("schedule_service", "schedules")
_PAUSE_REASON_DEFAULT = "Paused through schedule_pause."


def _schema(
    properties: dict[str, dict[str, Any]], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": False,
    }


# Every surviving final-head model field. ``provider``/``model``/``base_url``/
# ``grace_minutes`` are deliberately absent: only an operator/programmatic
# caller may set them, and the compiler rejects them from this surface.
_FIELD_PROPERTIES: dict[str, dict[str, Any]] = {
    "schedule": {
        "type": "string",
        "description": (
            "When the job runs: '30m' / 'every 2h' (recurring interval), "
            "'in 30m' (one-shot by duration), 'every monday 9am' / "
            "'weekdays at 9am', a 5- or 6-field cron expression, or an ISO "
            "timestamp."
        ),
    },
    "prompt": {
        "type": "string",
        "description": (
            "Self-contained task instruction run on each occurrence. For a "
            "reminder this is ONLY the exact message to deliver, in the "
            "user's own words — never instructions about style, language, "
            "length, channel, or delivery, and never a restatement of the "
            "delivery target; delivery belongs in 'deliver'."
        ),
    },
    "name": {"type": "string", "description": "Optional human-friendly name."},
    "repeat": {
        "type": ["integer", "string"],
        "description": (
            "Optional number of runs. Omit for forever on recurring jobs and "
            "one run on one-shots. 'once' is 1; 'forever' means unlimited."
        ),
    },
    "deliver": {
        "type": ["string", "array"],
        "description": (
            "Where output is posted: origin/local/all/home, "
            "bot-chat[:profile], room:<id>, or platform:chat[:thread]. "
            "Comma lists and arrays are accepted."
        ),
    },
    "failure_deliver": {
        "type": ["string", "array"],
        "description": "Optional separate target for failure notices only.",
    },
    "skills": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Ordered skill names loaded before the prompt.",
    },
    "catch_up": {
        "type": "boolean",
        "description": "Whether a missed recurring run is caught up once.",
    },
    "wrap_response": {
        "type": "string",
        "enum": ["on", "off", "inherit"],
        "description": "Delivered-text header/footer override.",
    },
    "script": {
        "type": "string",
        "description": "Operator-provisioned script path run before the prompt.",
    },
    "monitor": {
        "type": "string",
        "description": "Change-detector URL or script; unchanged output skips the run.",
    },
    "no_agent": {
        "type": "boolean",
        "default": False,
        "description": "Run the script only and deliver its stdout; requires script.",
    },
    "context_from": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Schedule ids whose latest completed output is injected.",
    },
    "continuity": {
        "type": "boolean",
        "description": "Inject this job's own previous completed output.",
    },
    "enabled_toolsets": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Restrict the job agent to these toolsets; [] means none.",
    },
    "workdir": {
        "type": "string",
        "description": "Absolute allowed directory the job runs from.",
    },
    "timezone": {
        "type": "string",
        "description": "IANA timezone the schedule is read in.",
    },
    "attach_to_session": {
        "type": "boolean",
        "description": "Mirror delivered output into the job's own conversation.",
    },
    "pinned": {
        "type": "boolean",
        "description": "Lock the current operator-owned provider/model to this job.",
    },
}
_CREATE_ONLY: dict[str, dict[str, Any]] = {
    "paused": {"type": "boolean", "description": "Create paused; resume to schedule."},
    "paused_reason": {
        "type": "string",
        "description": "Auditable reason; requires paused=true.",
    },
}
_IDEMPOTENCY: dict[str, dict[str, Any]] = {
    "request_key": {
        "type": "string",
        "description": "Optional idempotency key for safe retries of this request.",
    }
}
_TARGET: dict[str, dict[str, Any]] = {
    "schedule_id": {"type": "string", "description": "Stable schedule id."},
    "name": {"type": "string", "description": "Schedule name when id is unknown."},
}


def _service(ctx: dict[str, Any], registered: Any) -> Any:
    for source in (ctx.get("services"), registered):
        if not source:
            continue
        if isinstance(source, Mapping):
            for key in _SERVICE_KEYS:
                value = source.get(key)
                if value is not None:
                    return value
        else:
            for key in _SERVICE_KEYS:
                value = getattr(source, key, None)
                if value is not None:
                    return value
    raise RuntimeError(
        "schedule service unavailable: inject 'schedule_service' via "
        "ctx['services'] or register(services=...)"
    )


def _turn(ctx: dict[str, Any]) -> TurnContext | None:
    data = ctx.get("turn_context")
    if isinstance(data, TurnContext):
        return data
    if isinstance(data, Mapping):
        try:
            return TurnContext.model_validate(dict(data))
        except ValueError:
            return None
    return None


def _control(arguments: Mapping[str, Any]) -> tuple[dict[str, Any], str | None, int | None]:
    payload = dict(arguments or {})
    request_key = payload.pop("request_key", None)
    revision = payload.pop("config_revision", None)
    return (
        payload,
        str(request_key) if isinstance(request_key, str) and request_key else None,
        int(revision) if isinstance(revision, int) and not isinstance(revision, bool) else None,
    )


def _target(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    return (
        str(payload.pop("schedule_id")) if payload.get("schedule_id") else None,
        str(payload.pop("name")) if payload.get("name") else None,
    )


def _fail(exc: BaseException, operation: str, *, name: str | None = None) -> None:
    code = str(getattr(exc, "code", "") or type(exc).__name__)
    message = str(getattr(exc, "message", "") or exc)
    record_schedule_outcome(
        ScheduleOutcome(
            operation=operation,
            status="failed",
            name=name,
            message=f"{code}: {message}",
        )
    )
    raise ValueError(f"{code}: {message}") from exc


async def _create(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, request_key, _ = _control(args)
    try:
        return await _service(ctx, registered).create(
            payload,
            actor=ctx.get("actor"),
            turn=_turn(ctx),
            request_key=request_key or ctx.get("call_id"),
            tool_call_id=ctx.get("call_id"),
            run_id=ctx.get("correlation_id"),
        )
    except (ScheduleServiceError, ScheduleCompileError, ValueError, KeyError) as exc:
        _fail(exc, "create", name=str(payload.get("name") or "") or None)


async def _list(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    try:
        return await _service(ctx, registered).list(
            enabled_only=bool(args.get("enabled_only", False)),
            include_paused=bool(args.get("include_paused", True)),
        )
    except (ScheduleServiceError, ValueError, KeyError) as exc:
        _fail(exc, "list")


async def _get(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, _, _ = _control(args)
    schedule_id, name = _target(payload)
    try:
        return await _service(ctx, registered).get(
            schedule_id=schedule_id, name=name
        )
    except (ScheduleServiceError, ValueError, KeyError) as exc:
        _fail(exc, "get", name=name)


async def _update(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, request_key, revision = _control(args)
    schedule_id, name = _target(payload)
    try:
        return await _service(ctx, registered).update(
            schedule_id=schedule_id,
            name=name,
            arguments=payload,
            actor=ctx.get("actor"),
            turn=_turn(ctx),
            request_key=request_key or ctx.get("call_id"),
            tool_call_id=ctx.get("call_id"),
            run_id=ctx.get("correlation_id"),
            expected_config_revision=revision,
        )
    except (ScheduleServiceError, ScheduleCompileError, ValueError, KeyError) as exc:
        _fail(exc, "update", name=name)


async def _pause(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, request_key, revision = _control(args)
    schedule_id, name = _target(payload)
    reason = payload.pop("paused_reason", None)
    try:
        if reason is not None:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("paused_reason must be non-empty text.")
            if len(reason) > MAX_PAUSE_REASON_CHARS:
                raise ValueError(
                    f"paused_reason must be at most {MAX_PAUSE_REASON_CHARS} characters."
                )
            reason = reason.strip()
        # ScheduleService owns consent, CAS, audit and outcome recording. This
        # wrapper only shapes the one field that changes; cadence is untouched.
        return await _service(ctx, registered).update(
            schedule_id=schedule_id,
            name=name,
            arguments={
                "paused": True,
                "paused_reason": reason or _PAUSE_REASON_DEFAULT,
            },
            actor=ctx.get("actor"),
            turn=_turn(ctx),
            request_key=request_key or ctx.get("call_id"),
            tool_call_id=ctx.get("call_id"),
            run_id=ctx.get("correlation_id"),
            expected_config_revision=revision,
        )
    except (ScheduleServiceError, ScheduleCompileError, ValueError, KeyError) as exc:
        _fail(exc, "pause", name=name)


async def _resume(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, request_key, revision = _control(args)
    schedule_id, name = _target(payload)
    try:
        return await _service(ctx, registered).update(
            schedule_id=schedule_id,
            name=name,
            arguments={"paused": False},
            actor=ctx.get("actor"),
            turn=_turn(ctx),
            request_key=request_key or ctx.get("call_id"),
            tool_call_id=ctx.get("call_id"),
            run_id=ctx.get("correlation_id"),
            expected_config_revision=revision,
        )
    except (ScheduleServiceError, ScheduleCompileError, ValueError, KeyError) as exc:
        _fail(exc, "resume", name=name)


async def _delete(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, request_key, revision = _control(args)
    schedule_id, name = _target(payload)
    try:
        return await _service(ctx, registered).remove(
            schedule_id=schedule_id,
            name=name,
            actor=ctx.get("actor"),
            turn=_turn(ctx),
            request_key=request_key,
            tool_call_id=ctx.get("call_id"),
            run_id=ctx.get("correlation_id"),
            expected_config_revision=revision,
            authorized_by=(
                str(ctx.get("authorization"))
                if ctx.get("authorization")
                else None
            ),
        )
    except (ScheduleServiceError, ValueError, KeyError) as exc:
        _fail(exc, "remove", name=name)


async def _run_now(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, request_key, _ = _control(args)
    schedule_id, name = _target(payload)
    try:
        return await _service(ctx, registered).run_now(
            schedule_id=schedule_id,
            name=name,
            prompt=payload.get("prompt"),
            run_at=payload.get("run_at"),
            request_key=request_key,
            actor=ctx.get("actor"),
            turn=_turn(ctx),
            authorized_by=(
                str(ctx.get("authorization"))
                if ctx.get("authorization")
                else None
            ),
        )
    except (ScheduleServiceError, ValueError, KeyError) as exc:
        _fail(exc, "run_now", name=name)


async def _cancel(args: dict[str, Any], ctx: dict[str, Any], registered: Any) -> Any:
    payload, _, _ = _control(args)
    task_id = payload.pop("task_id", None)
    schedule_id, name = _target(payload)
    try:
        return await _service(ctx, registered).cancel(
            schedule_id=schedule_id,
            name=name,
            task_id=str(task_id) if task_id else None,
            actor=ctx.get("actor"),
            turn=_turn(ctx),
            authorized_by=(
                str(ctx.get("authorization"))
                if ctx.get("authorization")
                else None
            ),
        )
    except (ScheduleServiceError, ValueError, KeyError) as exc:
        _fail(exc, "cancel", name=name)


def _tool_defs() -> list[tuple[ToolSpec, Any]]:
    return [
        (
            ToolSpec(
                name="schedule_create",
                description=(
                    "Create a durable scheduled job. A new job is shown to the "
                    "human for approval before it is saved; with no answerable "
                    "surface it returns a durable approval_required proposal and "
                    "creates nothing."
                ),
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:write"],
                input_schema=_schema(
                    {**_FIELD_PROPERTIES, **_CREATE_ONLY, **_IDEMPOTENCY},
                    ["schedule"],
                ),
                resource_field="",
                secret_scan=True,
            ),
            _create,
        ),
        (
            ToolSpec(
                name="schedule_list",
                description="List durable schedules with their real next/last run state.",
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:read"],
                input_schema=_schema(
                    {
                        "enabled_only": {"type": "boolean"},
                        "include_paused": {"type": "boolean"},
                    }
                ),
                resource_field="",
                secret_scan=True,
            ),
            _list,
        ),
        (
            ToolSpec(
                name="schedule_get",
                description="Read one durable schedule by id or name.",
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:read"],
                input_schema=_schema({**_TARGET}),
                resource_field="",
                secret_scan=True,
            ),
            _get,
        ),
        (
            ToolSpec(
                name="schedule_update",
                description=(
                    "Edit an existing schedule by id (or name). A semantic no-op "
                    "returns no_change without re-anchoring; a real edit is shown "
                    "to the human for approval before it is saved."
                ),
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:write"],
                input_schema=_schema(
                    {
                        **_FIELD_PROPERTIES,
                        **_TARGET,
                        **_IDEMPOTENCY,
                        "config_revision": {
                            "type": "integer",
                            "description": "Expected revision; a stale value fails closed.",
                        },
                    }
                ),
                resource_field="",
                secret_scan=True,
            ),
            _update,
        ),
        (
            ToolSpec(
                name="schedule_pause",
                description=(
                    "Pause an existing schedule without deleting it or changing "
                    "its cadence. A model-initiated pause needs an approved edit "
                    "card; an operator pause is applied directly."
                ),
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:write"],
                input_schema=_schema(
                    {
                        **_TARGET,
                        **_IDEMPOTENCY,
                        "config_revision": {
                            "type": "integer",
                            "description": "Expected revision; a stale value fails closed.",
                        },
                        "paused_reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_PAUSE_REASON_CHARS,
                            "description": "Short auditable reason for the pause.",
                        },
                    }
                ),
                resource_field="",
                secret_scan=True,
            ),
            _pause,
        ),
        (
            ToolSpec(
                name="schedule_resume",
                description=(
                    "Resume a paused schedule without re-anchoring its time. A "
                    "model-initiated resume needs an approved edit card."
                ),
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:write"],
                input_schema=_schema(
                    {
                        **_TARGET,
                        **_IDEMPOTENCY,
                        "config_revision": {
                            "type": "integer",
                            "description": "Expected revision; a stale value fails closed.",
                        },
                    }
                ),
                resource_field="",
                secret_scan=True,
            ),
            _resume,
        ),
        (
            ToolSpec(
                name="schedule_delete",
                description=(
                    "Remove a schedule by id or name. A human-approved edit card "
                    "is required for a model-initiated removal."
                ),
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:write"],
                input_schema=_schema(
                    {
                        **_TARGET,
                        **_IDEMPOTENCY,
                        "config_revision": {"type": "integer"},
                    }
                ),
                resource_field="",
                secret_scan=True,
            ),
            _delete,
        ),
        (
            ToolSpec(
                name="schedule_run_now",
                description=(
                    "Run the named schedule once now as a normal background task "
                    "and return its handle. This does not consume a repeat count "
                    "or change the saved cadence. A chat-initiated run requires "
                    "an operator turn or a fresh approved card; the schedule "
                    "service enforces this gate."
                ),
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:write", "tasks:write"],
                input_schema=_schema(
                    {
                        **_TARGET,
                        "prompt": {
                            "type": "string",
                            "description": "Optional transient context for this run only.",
                        },
                        "run_at": {"type": "string", "description": "Optional ISO instant."},
                        **_IDEMPOTENCY,
                    }
                ),
                resource_field="",
                secret_scan=True,
            ),
            _run_now,
        ),
        (
            ToolSpec(
                name="schedule_cancel",
                description=(
                    "Cancel the named schedule's in-flight background task. The "
                    "schedule itself stays saved; returns a no-op when no task "
                    "is running. A chat-initiated cancel requires an operator "
                    "turn or a fresh approved card; the schedule service "
                    "enforces this gate."
                ),
                category="automation",
                risk=RiskLevel.LOW,
                permissions=["schedules:write", "tasks:write"],
                input_schema=_schema(
                    {
                        **_TARGET,
                        "task_id": {
                            "type": "string",
                            "description": "Optional exact task id to cancel.",
                        },
                    }
                ),
                resource_field="",
                secret_scan=True,
            ),
            _cancel,
        ),
    ]


_TOOL_DEFS: list[tuple[ToolSpec, Any]] = _tool_defs()


def register(registry: Any, services: Any = None) -> list[str]:
    """Register all automation tools and return their names in order."""
    registered: list[str] = []
    for spec, implementation in _TOOL_DEFS:

        async def handler(
            args: dict[str, Any], ctx: dict[str, Any], _impl=implementation
        ) -> Any:
            return await _impl(args, ctx, services)

        registry.register(spec, handler)
        registered.append(spec.name)
    return registered


__all__ = ["register"]
