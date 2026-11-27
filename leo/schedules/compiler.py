"""Side-effect-free schedule compiler.

:class:`ScheduleCompiler` is the only place that turns model/operator
arguments into a validated canonical :class:`~leo.schedules.domain.ScheduleSpec`.
It parses the final-head grammar, rejects unknown/unsupported fields, scans
unsafe text, merges edits over the latest durable spec, validates every
surviving execution field, resolves delivery and computes the first run and
configuration digests.

Nothing here writes to SQLite, the event bus or the scheduler. It only reads
the clock, the filesystem (script/workdir validation) and injected rosters.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .clock import Clock, SystemClock
from .delivery import (
    DeliveryContext,
    DeliveryPlan,
    DeliveryResolutionError,
    DeliveryResolver,
)
from .domain import (
    DeliveryTarget,
    ExecutionSpec,
    ScheduleExpression,
    ScheduleSpec,
    canonical_json,
    sha256_digest,
)
from .grammar import (
    ParsedSchedule,
    ScheduleGrammarError,
    default_timezone_name,
    next_occurrence,
    normalize_repeat,
    parse_cron_expression,
    parse_schedule,
    resolve_timezone,
)
from .scan import ScheduleSafetyError, require_safe

UTC = timezone.utc

# Final-head declared model fields plus handler-forwarded compatibility fields.
MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "schedule",
        "prompt",
        "repeat",
        "deliver",
        "failure_deliver",
        "skills",
        "skill",
        "catch_up",
        "wrap_response",
        "script",
        "monitor",
        "no_agent",
        "context_from",
        "continuity",
        "enabled_toolsets",
        "workdir",
        "timezone",
        "attach_to_session",
        "pinned",
        "paused",
        "paused_reason",
        # Legacy handler-compatible aliases, canonicalized into ``monitor``.
        "monitor_script",
        "monitor_url",
    }
)

OPERATOR_FIELDS: frozenset[str] = frozenset(
    {"provider", "model", "base_url", "grace_minutes", "script_args"}
)

# Known but non-schedule fields. Rejected with a precise code rather than
# treated as typos, because callers routinely pass a whole tool payload.
_NOT_SCHEDULE_FIELDS: frozenset[str] = frozenset(
    {
        "action",
        "job_id",
        "include_disabled",
        "reason",
        "session_id",
        "approve",
        "config",
        "config_revision",
        "schedule_id",
        "reasoning_effort",
        "interpreter",
    }
)

_MISSING = object()
_MAX_CONTEXT_REFS = 32
_MAX_TOOL_CHARS = 512
# A pause reason travels into the audit trail and edit cards; cap it so a
# lifecycle action cannot turn into an unbounded durable blob.
MAX_PAUSE_REASON_CHARS = 500


class ScheduleCompileError(ValueError):
    """A stable, preview-safe rejection from the compiler."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field


def _fail(code: str, message: str, *, field: str | None = None) -> ScheduleCompileError:
    return ScheduleCompileError(code, message, field=field)


def _stored_moment(value: Any) -> datetime:
    """Parse a durable timestamp without leaking a raw ValueError to callers."""
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise _fail(
            "invalid_schedule", "Stored schedule has an invalid timestamp.",
            field="schedule",
        ) from None


@dataclass(frozen=True)
class CompileResult:
    """Validated output, still not persisted."""

    spec: ScheduleSpec
    parsed_schedule: ParsedSchedule
    display: str
    first_run_at: datetime | None
    next_run_at: datetime | None
    config_revision: int
    config: dict[str, Any]
    config_digest: str
    proposal_digest: str
    edit_fingerprint: str | None
    delivery_plan: DeliveryPlan
    failure_delivery_plan: DeliveryPlan
    changed: bool
    reanchors: bool
    warnings: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        return self.proposal_digest

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.model_dump(mode="json"),
            "display": self.display,
            "first_run_at": self.first_run_at.isoformat() if self.first_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "config_revision": self.config_revision,
            "config": self.config,
            "config_digest": self.config_digest,
            "proposal_digest": self.proposal_digest,
            "edit_fingerprint": self.edit_fingerprint,
            "changed": self.changed,
            "reanchors": self.reanchors,
            "audience": self.delivery_plan.describe(),
            "failure_audience": self.failure_delivery_plan.describe(),
            "warnings": list(self.warnings),
        }


class ScheduleCompiler:
    """Deterministic compiler under an injected clock and injected rosters."""

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        settings: Any | None = None,
        default_timezone: str | None = None,
        minimum_lead_seconds: int = 120,
        past_grace_seconds: int = 120,
        scripts_root: str | Path | None = None,
        allowed_workdir_roots: Sequence[str | Path] | None = None,
        available_toolsets: Sequence[str] | None = None,
        known_schedule_ids: Sequence[str] | None = None,
        reference_schedules: Mapping[str, Sequence[str]] | None = None,
        pinned_provider: str | None = None,
        pinned_model: str | None = None,
        skill_docs: Mapping[str, str] | None = None,
        known_skills: Sequence[str] | None = None,
        delivery_context: DeliveryContext | None = None,
        max_prompt_chars: int = 20_000,
        max_name_chars: int = 200,
    ) -> None:
        self.clock = clock or SystemClock()
        self.default_timezone = resolve_timezone(
            default_timezone, default=default_timezone_name(settings)
        )
        self.minimum_lead_seconds = max(0, int(minimum_lead_seconds))
        self.past_grace_seconds = max(0, int(past_grace_seconds))
        self.scripts_root = (
            Path(scripts_root).expanduser().resolve() if scripts_root is not None else None
        )
        self.allowed_workdir_roots = tuple(
            Path(root).expanduser().resolve() for root in (allowed_workdir_roots or ())
        )
        self.available_toolsets = (
            frozenset(str(item) for item in available_toolsets)
            if available_toolsets is not None
            else None
        )
        self.known_schedule_ids = (
            frozenset(str(item) for item in known_schedule_ids)
            if known_schedule_ids is not None
            else None
        )
        self.reference_schedules = {
            str(key): tuple(str(item) for item in value)
            for key, value in (reference_schedules or {}).items()
        }
        self.pinned_provider = (pinned_provider or "").strip() or None
        self.pinned_model = (pinned_model or "").strip() or None
        # Creation-time skill documentation snapshot. ``None`` disables the
        # snapshot (legacy/compiler-only callers); an explicit mapping freezes
        # the docs into the durable config so later skill edits cannot change
        # what an already-created schedule runs.
        self.skill_docs = (
            {str(key): str(value) for key, value in skill_docs.items()}
            if skill_docs is not None
            else None
        )
        # A live registry gives us authority to reject invented skill names;
        # compiler-only callers with no registry keep the legacy permissive path.
        self.known_skills = (
            frozenset(str(name) for name in known_skills)
            if known_skills is not None
            else None
        )
        self.delivery_context = delivery_context or DeliveryContext()
        self.max_prompt_chars = int(max_prompt_chars)
        self.max_name_chars = int(max_name_chars)

    # ------------------------------------------------------------------
    # Public compile entry points
    # ------------------------------------------------------------------
    def compile_create(
        self,
        arguments: Mapping[str, Any],
        *,
        operator: bool = False,
        now: datetime | None = None,
        delivery_context: DeliveryContext | None = None,
    ) -> CompileResult:
        return self._compile(
            arguments,
            base=None,
            operation="create",
            operator=operator,
            now=now,
            base_revision=1,
            schedule_id=None,
            current_next_run=None,
            delivery_context=delivery_context,
        )

    def compile_update(
        self,
        base: ScheduleSpec | Mapping[str, Any] | Any,
        arguments: Mapping[str, Any],
        *,
        operator: bool = False,
        now: datetime | None = None,
        base_revision: int = 1,
        schedule_id: str | None = None,
        current_next_run: datetime | str | None = None,
        delivery_context: DeliveryContext | None = None,
        allow_pause_edit: bool = False,
    ) -> CompileResult:
        """Compile an edit; ``allow_pause_edit`` admits only the pause fields.

        Explicit pause/resume tools pass this flag through ScheduleService.
        Ordinary edits still reject ``paused`` as create-only, so the pause
        state has one intentional, auditable mutation path.
        """
        base_spec = self._coerce_base(base)
        return self._compile(
            arguments,
            base=base_spec,
            operation="update",
            operator=operator,
            now=now,
            base_revision=max(1, int(base_revision)),
            schedule_id=schedule_id,
            current_next_run=current_next_run,
            delivery_context=delivery_context,
            allow_pause_edit=allow_pause_edit,
        )

    def compile(
        self,
        arguments: Mapping[str, Any],
        *,
        operation: str = "create",
        base: ScheduleSpec | Mapping[str, Any] | Any | None = None,
        **kwargs: Any,
    ) -> CompileResult:
        if operation == "create":
            return self.compile_create(arguments, **kwargs)
        if operation == "update":
            if base is None:
                raise _fail("schedule_required", "An update requires the latest schedule spec.", field="base")
            return self.compile_update(base, arguments, **kwargs)
        raise _fail("invalid_operation", f"Unknown compile operation {operation!r}.")

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _compile(
        self,
        arguments: Mapping[str, Any],
        *,
        base: ScheduleSpec | None,
        operation: str,
        operator: bool,
        now: datetime | None,
        base_revision: int,
        schedule_id: str | None,
        current_next_run: datetime | str | None,
        delivery_context: DeliveryContext | None,
        allow_pause_edit: bool = False,
    ) -> CompileResult:
        if not isinstance(arguments, Mapping):
            raise _fail("invalid_arguments", "Schedule arguments must be a mapping.")
        args = {str(key): value for key, value in arguments.items()}
        if operation == "update":
            # A missing JSON field and an explicit ``null`` both mean "leave the
            # stored value unchanged" on an edit; final-head uses ''/[] to
            # clear. Dropping nulls also keeps a full-schema tool call from
            # wiping untouched fields.
            args = {key: value for key, value in args.items() if value is not None}
        self._reject_bad_fields(
            args, operation=operation, operator=operator,
            allow_pause_edit=allow_pause_edit,
        )

        # Parse and scan text before constructing anything durable-looking.
        prompt = self._merged_string(
            args, "prompt", base.prompt if base else "", field="prompt"
        )
        if not prompt and not isinstance(prompt, str):
            raise _fail("invalid_prompt", "prompt must be text.", field="prompt")
        if len(prompt) > self.max_prompt_chars:
            raise _fail("prompt_too_long", "prompt is too long.", field="prompt")
        self._scan_field("prompt", prompt)
        for skill in self._merged_skills(args, base):
            self._scan_field("skills", skill, scope="display")

        now_utc = self._now(now)
        parsed, display, reanchors, stored_timezone = self._resolve_schedule(
            args, base, operation=operation, now=now_utc
        )

        warnings: list[str] = []
        repeat_limit = self._merged_repeat(args, base, parsed, operation=operation)
        execution = self._merged_execution(args, base, parsed, operator=operator)

        # The runtime applies grace_minutes to every schedule kind: cron,
        # interval and one-shot alike (a stale one-shot is retired as missed).
        # Validation must not reject a window the runtime will honour, so no
        # kind-specific grace check belongs here.

        self._validate_supplied_skills(args, base, operation, execution.skills)

        if execution.no_agent and not execution.script:
            raise _fail(
                "no_agent_requires_script",
                "no_agent=True requires a script.",
                field="no_agent",
            )
        if execution.monitor and execution.no_agent:
            raise _fail(
                "monitor_incompatible_with_no_agent",
                "monitor cannot be combined with no_agent.",
                field="monitor",
            )
        if not execution.no_agent and not prompt and not execution.skills:
            raise _fail(
                "missing_payload",
                "A schedule requires a prompt or at least one skill.",
                field="prompt",
            )

        paused, paused_reason = self._merged_pause(
            args, base, operation=operation, allow_pause_edit=allow_pause_edit
        )
        catch_up = self._merged_catch_up(args, base)

        resolver = DeliveryResolver(delivery_context or self.delivery_context)
        delivery_plan, failure_plan, failure_explicit = self._resolve_delivery_plans(
            args, base, operation=operation, resolver=resolver
        )

        expression = ScheduleExpression(
            kind=parsed.kind,
            expression=parsed.expression or None,
            run_at=parsed.run_at.isoformat() if parsed.kind == "once" and parsed.run_at else None,
            timezone=stored_timezone,
        )
        name = self._merged_name(args, base, prompt, execution)
        spec = ScheduleSpec(
            name=name,
            expression=expression,
            prompt=prompt,
            repeat_limit=repeat_limit,
            delivery=list(delivery_plan.targets),
            # Empty means "failures follow the result delivery"; a non-empty
            # list is an explicit separate lane. Preserve that distinction so
            # a later result-delivery edit does not strand failure notices.
            failure_delivery=list(failure_plan.targets) if failure_explicit else [],
            execution=execution,
            paused=paused,
            paused_reason=paused_reason,
            catch_up=catch_up,
            config=dict(base.config) if base else {},
        )
        # Re-scan the effective prompt/name so an unsafe stored record cannot be
        # edited around the preflight.
        self._scan_field("prompt", spec.prompt)
        self._scan_field("name", spec.name)

        first_run, next_run = self._first_runs(
            parsed,
            now_utc,
            paused=paused,
            operation=operation,
            reanchors=reanchors,
            current_next_run=current_next_run,
            operator=operator,
        )

        skill_snapshot = self._skill_snapshot(
            execution,
            base,
            refresh=("skills" in args or "skill" in args or base is None),
        )
        config = self._config_payload(spec, parsed, display, skill_snapshot=skill_snapshot)
        new_semantic = self._semantic_payload(spec)
        # Display spelling ("every 24h" vs "every 1d") is presentation, not
        # semantic config, so it must not make a no-op edit look changed.
        config_digest = sha256_digest(new_semantic)
        revision = 1 if operation == "create" else base_revision + 1

        base_semantic = self._semantic_payload(base) if base is not None else None
        changed = base_semantic is None or canonical_json(base_semantic) != canonical_json(
            new_semantic
        )

        proposal_digest = spec.digest()
        edit_fingerprint = None
        if operation == "update":
            edit_fingerprint = sha256_digest(
                {
                    "operation": "update",
                    "schedule_id": schedule_id or "",
                    "config_revision": revision,
                    "spec_digest": proposal_digest,
                }
            )
            proposal_digest = edit_fingerprint

        return CompileResult(
            spec=spec,
            parsed_schedule=parsed,
            display=display,
            first_run_at=first_run,
            next_run_at=next_run,
            config_revision=revision,
            config=config,
            config_digest=config_digest,
            proposal_digest=proposal_digest,
            edit_fingerprint=edit_fingerprint,
            delivery_plan=delivery_plan,
            failure_delivery_plan=failure_plan,
            changed=changed,
            reanchors=reanchors,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Field admission
    # ------------------------------------------------------------------
    def _reject_bad_fields(
        self,
        args: Mapping[str, Any],
        *,
        operation: str,
        operator: bool,
        allow_pause_edit: bool = False,
    ) -> None:
        for key in args:
            if key in _NOT_SCHEDULE_FIELDS:
                raise _fail(
                    "not_a_schedule_field",
                    f"{key!r} is not part of a schedule spec.",
                    field=key,
                )
            if key in OPERATOR_FIELDS:
                if not operator:
                    raise _fail(
                        "operator_only_field",
                        f"{key!r} is operator/programmatic-only.",
                        field=key,
                    )
                continue
            if key not in MODEL_FIELDS:
                raise _fail(
                    "unknown_field",
                    f"Unknown schedule field {key!r}.",
                    field=key,
                )
            if (
                operation == "update"
                and not allow_pause_edit
                and key in ("paused", "paused_reason")
            ):
                raise _fail(
                    "not_editable",
                    f"{key!r} is create-only; use pause/resume.",
                    field=key,
                )

    # ------------------------------------------------------------------
    # Schedule expression
    # ------------------------------------------------------------------
    def _resolve_schedule(
        self,
        args: Mapping[str, Any],
        base: ScheduleSpec | None,
        *,
        operation: str,
        now: datetime,
    ) -> tuple[ParsedSchedule, str, bool, str | None]:
        has_schedule = "schedule" in args
        has_timezone = "timezone" in args
        if operation == "create" and not has_schedule:
            raise _fail("schedule_required", "schedule is required for create.", field="schedule")
        if has_schedule:
            raw_schedule = args["schedule"]
            if raw_schedule is None or (isinstance(raw_schedule, str) and not raw_schedule.strip()):
                raise _fail("schedule_required", "schedule must not be empty.", field="schedule")
            timezone_value = args.get("timezone") if has_timezone else None
            try:
                parsed = parse_schedule(
                    raw_schedule,
                    now=now,
                    default_timezone=self.default_timezone,
                    timezone=timezone_value,
                )
            except ScheduleGrammarError as exc:
                raise _fail(
                    exc.code,
                    exc.message,
                    field="timezone" if exc.code == "invalid_timezone" else "schedule",
                ) from None
            reanchors = base is None or canonical_json(
                self._schedule_semantic(parsed)
            ) != canonical_json(self._schedule_semantic(base.expression))
            return parsed, parsed.display, reanchors, parsed.timezone

        # Update without an explicit schedule: reuse the latest expression.
        assert base is not None
        expression = base.expression
        if has_timezone:
            timezone_value = args.get("timezone")
            stored_timezone = resolve_timezone(
                timezone_value, default=self.default_timezone
            )
            zone = stored_timezone
        elif expression.timezone is None:
            # A legacy null zone stays UTC; an unrelated edit must not silently
            # reinterpret it under the new system-local default.
            zone = "UTC"
            stored_timezone = None
        else:
            stored_timezone = expression.timezone
            try:
                zone = resolve_timezone(expression.timezone, default=self.default_timezone)
            except ScheduleGrammarError as exc:
                raise _fail(exc.code, exc.message, field="timezone") from None
        if expression.kind == "once":
            run_at = expression.run_at
            if not run_at:
                raise _fail("invalid_schedule", "Stored one-shot has no run_at.", field="schedule")
            moment = _stored_moment(run_at)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
            parsed = ParsedSchedule(
                kind="once",
                expression="",
                timezone=zone,
                display=f"once at {moment.strftime('%Y-%m-%d %H:%M')}",
                run_at=moment.astimezone(UTC),
            )
        elif expression.expression is None:
            raise _fail("invalid_schedule", "Stored schedule has no expression.", field="schedule")
        elif expression.expression.strip().lower().startswith("@every"):
            try:
                parsed = parse_schedule(
                    expression.expression,
                    now=now,
                    default_timezone=self.default_timezone,
                    timezone=zone,
                    allow_legacy_every=True,
                )
            except ScheduleGrammarError as exc:
                raise _fail(exc.code, exc.message, field="schedule") from None
        elif expression.kind == "interval":
            try:
                seconds = float(expression.expression)
            except ValueError:
                raise _fail(
                    "invalid_schedule",
                    "Stored interval expression is not numeric.",
                    field="schedule",
                ) from None
            if seconds <= 0:
                raise _fail(
                    "invalid_schedule",
                    "Stored interval expression must be positive.",
                    field="schedule",
                )
            minutes = int(seconds // 60) if seconds % 60 == 0 else max(1, round(seconds / 60))
            stored_expression = (
                str(int(seconds)) if float(seconds).is_integer() else str(seconds)
            )
            parsed = ParsedSchedule(
                kind="interval",
                expression=stored_expression,
                timezone=zone,
                display=f"every {minutes}m",
                minutes=minutes,
            )
        else:
            try:
                parsed_cron = parse_cron_expression(expression.expression)
            except ScheduleGrammarError as exc:
                raise _fail(exc.code, exc.message, field="schedule") from None
            parsed = ParsedSchedule(
                kind="cron",
                expression=parsed_cron.expression,
                timezone=zone,
                display=expression.expression,
                cron=parsed_cron,
            )
        reanchors = has_timezone and stored_timezone != base.expression.timezone
        stored_config = base.config if isinstance(base.config, dict) else {}
        stored_schedule = stored_config.get("schedule")
        stored_display = (
            stored_schedule.get("display")
            if isinstance(stored_schedule, dict)
            else None
        )
        return (
            parsed,
            str(stored_display or parsed.display),
            reanchors,
            stored_timezone,
        )

    def _first_runs(
        self,
        parsed: ParsedSchedule,
        now: datetime,
        *,
        paused: bool,
        operation: str,
        reanchors: bool,
        current_next_run: datetime | str | None,
        operator: bool,
    ) -> tuple[datetime | None, datetime | None]:
        if operation == "update" and not reanchors and current_next_run is not None:
            existing = (
                _stored_moment(current_next_run)
                if isinstance(current_next_run, str)
                else current_next_run
            )
            if existing.tzinfo is None:
                existing = existing.replace(tzinfo=UTC)
            existing = existing.astimezone(UTC)
            if parsed.kind == "once" and not paused:
                # Keeping the old instant must not silently re-arm a stale
                # one-shot. Operators keep the same compile-time grace as a
                # re-anchored edit; model edits may never preserve a past time.
                # A job that stays paused has no live instant to re-arm; the
                # check applies when it is resumed instead.
                delta = (existing - now.astimezone(UTC)).total_seconds()
                if delta < (-self.past_grace_seconds if operator else 0):
                    raise _fail(
                        "one_shot_in_past",
                        "The preserved one-shot time is in the past.",
                        field="schedule",
                    )
            return existing, None if paused else existing

        try:
            candidate = next_occurrence(parsed, now).astimezone(UTC)
        except ScheduleGrammarError as exc:
            raise _fail(exc.code, exc.message, field="schedule") from None

        if parsed.kind == "once":
            if operation == "create" or reanchors:
                delta = (candidate - now.astimezone(UTC)).total_seconds()
                if operator:
                    # Pausing suppresses the live next_run, but it does not
                    # license a new past instant: only *preserving* an existing
                    # anchor is allowed while paused (resume re-checks it).
                    if delta < -self.past_grace_seconds:
                        raise _fail(
                            "one_shot_in_past",
                            "The requested one-shot time is in the past.",
                            field="schedule",
                        )
                elif delta < self.minimum_lead_seconds:
                    raise _fail(
                        "one_shot_lead_too_short",
                        "A model-created one-shot must be at least "
                        f"{self.minimum_lead_seconds} seconds in the future.",
                        field="schedule",
                    )
        return candidate, None if paused else candidate

    # ------------------------------------------------------------------
    # Repeat / pause / catch-up
    # ------------------------------------------------------------------
    def _merged_repeat(
        self,
        args: Mapping[str, Any],
        base: ScheduleSpec | None,
        parsed: ParsedSchedule,
        *,
        operation: str,
    ) -> int | None:
        if "repeat" in args:
            try:
                return normalize_repeat(args["repeat"], kind=parsed.kind)
            except ScheduleGrammarError as exc:
                raise _fail(exc.code, exc.message, field="repeat") from None
        if operation == "update" and base is not None:
            # A repeat edit is a run-limit patch: an absent repeat never resets
            # the stored limit, even when the schedule kind changes.
            if parsed.kind == "once":
                if base.repeat_limit in (None, 1):
                    return 1
                raise _fail(
                    "invalid_repeat",
                    "A one-shot schedule can only run once.",
                    field="repeat",
                )
            return base.repeat_limit
        try:
            return normalize_repeat(None, kind=parsed.kind)
        except ScheduleGrammarError as exc:  # pragma: no cover - defensive
            raise _fail(exc.code, exc.message, field="repeat") from None

    def _merged_pause(
        self,
        args: Mapping[str, Any],
        base: ScheduleSpec | None,
        *,
        operation: str,
        allow_pause_edit: bool = False,
    ) -> tuple[bool, str | None]:
        if operation == "update" and not allow_pause_edit:
            return bool(base.paused) if base else False, base.paused_reason if base else None
        paused = self._as_bool(args.get("paused", False), "paused", default=False)
        reason: str | None = None
        if "paused_reason" in args:
            raw = args["paused_reason"]
            if raw is not None and not isinstance(raw, str):
                raise _fail("invalid_paused_reason", "paused_reason must be text.", field="paused_reason")
            reason = (raw or "").strip() or None
        if operation == "update" and base is not None:
            # Pause-only requests carry one field; preserve the other from the
            # latest row so a resume cannot silently rewrite the schedule.
            if "paused" not in args:
                paused = bool(base.paused)
            if "paused_reason" not in args:
                reason = base.paused_reason if paused else None
        if reason is not None and not paused:
            raise _fail(
                "paused_reason_requires_paused",
                "paused_reason requires paused=True.",
                field="paused_reason",
            )
        if reason is not None and len(reason) > MAX_PAUSE_REASON_CHARS:
            raise _fail(
                "paused_reason_too_long",
                f"paused_reason must be at most {MAX_PAUSE_REASON_CHARS} characters.",
                field="paused_reason",
            )
        if paused and reason is None:
            reason = (
                "Created paused; awaiting operator approval."
                if operation == "create"
                else "Paused by an authorized schedule update."
            )
        return paused, reason

    def _merged_catch_up(self, args: Mapping[str, Any], base: ScheduleSpec | None) -> bool | None:
        if "catch_up" in args:
            value = args["catch_up"]
            if value is None:
                return None
            return self._as_bool(value, "catch_up")
        return base.catch_up if base else None

    # ------------------------------------------------------------------
    # Execution fields
    # ------------------------------------------------------------------
    def _merged_execution(
        self,
        args: Mapping[str, Any],
        base: ScheduleSpec | None,
        parsed: ParsedSchedule,
        *,
        operator: bool,
    ) -> ExecutionSpec:
        current = base.execution if base else ExecutionSpec()
        data = current.model_dump()

        if "script" in args:
            data["script"] = self._normalize_script(args["script"], "script", operator=operator)
        if "script_args" in args:
            data["script_args"] = self._normalize_string_list(
                args["script_args"], "script_args", max_items=128
            )
        if data.get("script_args") and not data.get("script"):
            raise _fail(
                "script_args_requires_script",
                "script_args requires a script.",
                field="script_args",
            )
        if "no_agent" in args:
            data["no_agent"] = self._as_bool(args["no_agent"], "no_agent", default=False)

        if "monitor" in args:
            data["monitor"] = self._normalize_monitor(args["monitor"], operator=operator)
        legacy_script = "monitor_script" in args
        legacy_url = "monitor_url" in args
        if legacy_script and legacy_url:
            raise _fail(
                "monitor_conflict",
                "monitor_script and monitor_url are mutually exclusive.",
                field="monitor",
            )
        if legacy_script or legacy_url:
            if "monitor" in args:
                raise _fail(
                    "monitor_conflict",
                    "Use either monitor or the legacy monitor_script/monitor_url form.",
                    field="monitor",
                )
            legacy_value = args["monitor_script"] if legacy_script else args["monitor_url"]
            data["monitor"] = self._normalize_monitor(legacy_value, operator=operator)

        context_from = list(current.context_from)
        continuity = bool(current.continuity)
        if "context_from" in args:
            context_from = self._normalize_context_from(args["context_from"])
        if "continuity" in args:
            continuity = self._as_bool(args["continuity"], "continuity", default=False)
            has_self = any(ref.lower() == "self" for ref in context_from)
            if continuity and not has_self:
                context_from.append("self")
            elif not continuity and has_self:
                context_from = [ref for ref in context_from if ref.lower() != "self"]
        data["context_from"] = self._validate_context_refs(context_from)
        # ``continuity`` is the persisted boolean mirror of context_from's
        # "self" marker; keep both consistent for views/cards.
        data["continuity"] = continuity or any(
            ref.lower() == "self" for ref in data["context_from"]
        )

        if "enabled_toolsets" in args:
            value = args["enabled_toolsets"]
            if value is None:
                data["enabled_toolsets"] = None
            else:
                toolsets = self._normalize_string_list(
                    value, "enabled_toolsets", max_items=128
                )
                if self.available_toolsets is not None:
                    unknown = [item for item in toolsets if item not in self.available_toolsets]
                    if unknown:
                        raise _fail(
                            "unknown_toolset",
                            f"Unknown toolset(s): {', '.join(sorted(unknown))}.",
                            field="enabled_toolsets",
                        )
                data["enabled_toolsets"] = toolsets

        if "skills" in args or "skill" in args:
            data["skills"] = self._merged_skills(args, base)

        if "wrap_response" in args:
            value = args["wrap_response"]
            data["wrap_response"] = self._normalize_wrap_response(value, operator=operator)

        if "workdir" in args:
            data["workdir"] = self._normalize_workdir(args["workdir"])

        if "attach_to_session" in args:
            data["attach_to_session"] = self._as_bool(
                args["attach_to_session"], "attach_to_session", default=False
            )

        # Operator/programmatic fields.
        if "provider" in args:
            data["provider"] = self._optional_text(args["provider"], "provider")
        if "model" in args:
            data["model"] = self._optional_text(args["model"], "model")
        if "base_url" in args:
            data["base_url"] = self._normalize_base_url(args["base_url"])
        if data.get("base_url") and not data.get("provider"):
            raise _fail(
                "base_url_requires_provider",
                "base_url requires an explicit provider.",
                field="base_url",
            )
        if "grace_minutes" in args:
            data["grace_minutes"] = self._normalize_grace_minutes(args["grace_minutes"])

        if "pinned" in args:
            pinned = self._as_bool(args["pinned"], "pinned", default=False)
            if pinned:
                if not data.get("provider") or not data.get("model"):
                    if self.pinned_provider and self.pinned_model:
                        data["provider"] = self.pinned_provider
                        data["model"] = self.pinned_model
                    else:
                        raise _fail(
                            "pinned_target_missing",
                            "pinned=True requires a resolvable current provider/model.",
                            field="pinned",
                        )
            elif operator:
                # pinned=false releases the lock; an operator may still set an
                # explicit provider/model in the same call.
                if "provider" not in args:
                    data["provider"] = None
                if "model" not in args:
                    data["model"] = None
            data["pinned"] = pinned
        else:
            data["pinned"] = bool(current.pinned)

        if data.get("script_args") and not data.get("script"):
            raise _fail(
                "script_args_requires_script",
                "script_args requires a script.",
                field="script_args",
            )
        if data.get("no_agent") and not data.get("script"):
            raise _fail(
                "no_agent_requires_script",
                "no_agent=True requires a script.",
                field="no_agent",
            )
        if data.get("monitor") and data.get("no_agent"):
            raise _fail(
                "monitor_incompatible_with_no_agent",
                "monitor cannot be combined with no_agent.",
                field="monitor",
            )
        if data.get("monitor") and data.get("script"):
            # One adapter must own the occurrence. A monitor is itself a URL or
            # a restricted script; a second pre-processing script would make
            # field precedence ambiguous, so reject rather than drop a field.
            raise _fail(
                "monitor_incompatible_with_script",
                "monitor cannot be combined with script.",
                field="monitor",
            )
        try:
            return ExecutionSpec.model_validate(data)
        except ValueError as exc:
            raise _fail("invalid_execution", "Invalid execution field combination.", field="execution") from exc

    def _merged_skills(self, args: Mapping[str, Any], base: ScheduleSpec | None) -> list[str]:
        if "skills" in args:
            if args["skills"] is None:
                return list(base.execution.skills) if base else []
            return self._normalize_string_list(args["skills"], "skills", max_items=64)
        if "skill" in args:
            value = args["skill"]
            if value is None:
                return list(base.execution.skills) if base else []
            if not isinstance(value, str):
                raise _fail("invalid_skills", "skill must be text.", field="skill")
            return [value.strip()] if value.strip() else []
        return list(base.execution.skills) if base else []

    def _validate_supplied_skills(
        self,
        args: Mapping[str, Any],
        base: ScheduleSpec | None,
        operation: str,
        skills: Sequence[str],
    ) -> None:
        """Reject only skill names newly supplied by this request.

        Base skills were checked against the registry when the job was created.
        The registry is live and may have dropped a skill since then; an
        unrelated edit must not be blocked by that drift.
        """
        if self.known_skills is None:
            return
        if operation == "create":
            supplied = skills
        else:
            if "skills" not in args and "skill" not in args:
                return
            preserved = frozenset(base.execution.skills) if base else frozenset()
            supplied = [name for name in skills if name not in preserved]
        for skill in supplied:
            if skill not in self.known_skills:
                raise _fail(
                    "unknown_skill",
                    f"Unknown skill {skill!r}; it is not in the live skill registry.",
                    field="skills",
                )

    def _skill_snapshot(
        self,
        execution: ExecutionSpec,
        base: ScheduleSpec | None,
        *,
        refresh: bool,
    ) -> dict[str, str]:
        """Freeze each referenced skill's documentation into the config.

        A schedule runs later, when the operator may have edited the skill. The
        persisted snapshot is what the job actually executes, so the card and
        the runtime cannot silently diverge.
        """
        if not execution.skills:
            return {}
        if not refresh and base is not None:
            stored = base.config.get("skill_snapshot")
            if isinstance(stored, Mapping):
                return {
                    str(key): str(value)
                    for key, value in stored.items()
                    if str(key) in execution.skills
                }
        if self.skill_docs is None:
            return {}
        snapshot: dict[str, str] = {}
        total = 0
        for name in execution.skills:
            doc = str(self.skill_docs.get(name) or "").strip()
            if not doc:
                continue
            remaining = max(0, 16_000 - total)
            if remaining <= 0:
                break
            bounded = doc[: min(4_000, remaining)]
            snapshot[name] = bounded
            total += len(bounded)
        return snapshot

    def _normalize_context_from(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise _fail("invalid_context_from", "context_from must be a list.", field="context_from")
        return self._normalize_string_list(value, "context_from", max_items=_MAX_CONTEXT_REFS)

    def _validate_context_refs(self, refs: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            if not ref:
                continue
            if len(ref) > _MAX_TOOL_CHARS:
                raise _fail("invalid_context_from", "context_from reference is too long.", field="context_from")
            if ref in seen:
                continue
            seen.add(ref)
            cleaned.append(ref)
        if self.known_schedule_ids is not None:
            unknown = [
                ref
                for ref in cleaned
                if ref.lower() != "self" and ref not in self.known_schedule_ids
            ]
            if unknown:
                raise _fail(
                    "unknown_context_schedule",
                    f"Unknown context_from schedule(s): {', '.join(sorted(unknown))}.",
                    field="context_from",
                )
        self._reject_context_cycles(cleaned)
        return cleaned

    def _reject_context_cycles(self, refs: list[str]) -> None:
        if not self.reference_schedules:
            return
        visiting: set[str] = set()
        visited: set[str] = set()

        def walk(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for child in self.reference_schedules.get(node, ()):
                if walk(child):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        if any(ref.lower() != "self" and walk(ref) for ref in refs):
            raise _fail(
                "context_cycle",
                "context_from introduces a schedule cycle.",
                field="context_from",
            )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------
    @staticmethod
    def _as_bool(value: Any, field: str, *, default: bool | object = _MISSING) -> bool:
        if value is None and default is not _MISSING:
            return bool(default)
        if isinstance(value, bool):
            return value
        raise _fail("invalid_field", f"{field} must be true or false.", field=field)

    @staticmethod
    def _optional_text(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise _fail("invalid_field", f"{field} must be text.", field=field)
        text = value.strip()
        if "\x00" in text:
            raise _fail("invalid_field", f"{field} contains a NUL byte.", field=field)
        return text or None

    def _normalize_string_list(
        self, value: Any, field: str, *, max_items: int
    ) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise _fail("invalid_field", f"{field} must be a list.", field=field)
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            if not isinstance(item, str):
                raise _fail("invalid_field", f"{field} entries must be text.", field=field)
            text = item.strip()
            if not text:
                continue
            if "\x00" in text or len(text) > _MAX_TOOL_CHARS:
                raise _fail("invalid_field", f"{field} contains an invalid entry.", field=field)
            if text not in result:
                result.append(text)
        if len(result) > max_items:
            raise _fail("invalid_field", f"{field} has too many entries.", field=field)
        return result

    def _normalize_script(self, value: Any, field: str, *, operator: bool) -> str | None:
        raw = self._optional_text(value, field)
        if raw is None:
            return None
        if any(char.isspace() for char in raw):
            raise _fail(
                "script_command_line",
                "script must name a file, not a shell command line.",
                field=field,
            )
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            if self.scripts_root is None:
                raise _fail(
                    "script_not_allowed",
                    "Relative scripts require an operator-configured scripts root.",
                    field=field,
                )
            candidate = self.scripts_root / candidate
        resolved = candidate.resolve(strict=False)
        if self.scripts_root is not None:
            root = self.scripts_root.resolve()
            if resolved != root and root not in resolved.parents:
                raise _fail(
                    "script_outside_allowed_root",
                    "script resolves outside the allowlisted scripts root.",
                    field=field,
                )
        if not resolved.is_file():
            raise _fail("script_not_found", "script file does not exist.", field=field)
        # Preserve a relative spelling so the allowlisted root can move.
        return raw if not Path(raw).expanduser().is_absolute() else str(resolved)

    def _normalize_monitor(self, value: Any, *, operator: bool) -> str | None:
        raw = self._optional_text(value, "monitor")
        if raw is None:
            return None
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in raw):
            # Newlines/control characters can smuggle headers or split a URL.
            raise _fail(
                "invalid_monitor",
                "monitor contains a control character.",
                field="monitor",
            )
        lowered = raw.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            parts = urlsplit(raw)
            if (
                parts.scheme not in ("http", "https")
                or not parts.hostname
                or parts.username
                or parts.password
                or len(raw) > 4096
            ):
                raise _fail(
                    "invalid_monitor",
                    "monitor URL must be a plain http(s) URL without credentials.",
                    field="monitor",
                )
            if self._is_private_monitor_host(parts.hostname):
                raise _fail(
                    "monitor_private_target",
                    "monitor targets must be public network addresses.",
                    field="monitor",
                )
            return raw
        return self._normalize_script(raw, "monitor", operator=operator)

    @staticmethod
    def _is_private_monitor_host(host: str) -> bool:
        """Reject a literal private/loopback monitor host before persistence.

        DNS-resolved private targets are rejected at execution by the existing
        egress client; this closes the literal-address case so an obviously
        internal target never reaches a card. Headers, cookies and userinfo are
        not part of the schedule schema and are rejected with the URL checks.
        """
        text = str(host or "").strip("[]").lower()
        if text in {"localhost", "localhost.localdomain", "ip6-localhost"}:
            return True
        try:
            address = ipaddress.ip_address(text)
        except ValueError:
            return False
        return not address.is_global or address.is_private or address.is_loopback

    def _normalize_wrap_response(self, value: Any, *, operator: bool) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, bool):
            if not operator:
                raise _fail(
                    "invalid_wrap_response",
                    "wrap_response must be 'on', 'off', or 'inherit'.",
                    field="wrap_response",
                )
            return "on" if value else "off"
        if not isinstance(value, str):
            raise _fail(
                "invalid_wrap_response",
                "wrap_response must be 'on', 'off', or 'inherit'.",
                field="wrap_response",
            )
        normalized = value.strip().lower()
        if normalized == "inherit":
            return None
        if normalized in ("on", "off"):
            return normalized
        raise _fail(
            "invalid_wrap_response",
            "wrap_response must be 'on', 'off', or 'inherit'.",
            field="wrap_response",
        )

    def _normalize_workdir(self, value: Any) -> str | None:
        raw = self._optional_text(value, "workdir")
        if raw is None:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise _fail(
                "invalid_workdir",
                "workdir must be an absolute path.",
                field="workdir",
            )
        resolved = path.resolve(strict=False)
        if not resolved.exists():
            raise _fail("invalid_workdir", "workdir does not exist.", field="workdir")
        if not resolved.is_dir():
            raise _fail("invalid_workdir", "workdir is not a directory.", field="workdir")
        if self.allowed_workdir_roots:
            if not any(
                resolved == root or root in resolved.parents
                for root in self.allowed_workdir_roots
            ):
                raise _fail(
                    "workdir_outside_allowed_roots",
                    "workdir is outside LEO's allowed roots.",
                    field="workdir",
                )
        return str(resolved)

    def _normalize_base_url(self, value: Any) -> str | None:
        raw = self._optional_text(value, "base_url")
        if raw is None:
            return None
        raw = raw.rstrip("/")
        parts = urlsplit(raw)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username
            or parts.password
            or len(raw) > 2048
        ):
            raise _fail(
                "invalid_base_url",
                "base_url must be a plain http(s) endpoint.",
                field="base_url",
            )
        return raw

    def _normalize_grace_minutes(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise _fail(
                "invalid_grace_minutes",
                "grace_minutes must be a whole number of minutes.",
                field="grace_minutes",
            )
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if not re.fullmatch(r"[+-]?\d+", text):
                raise _fail(
                    "invalid_grace_minutes",
                    "grace_minutes must be a whole number of minutes.",
                    field="grace_minutes",
                )
            value = int(text)
        if isinstance(value, float):
            if not value.is_integer():
                raise _fail(
                    "invalid_grace_minutes",
                    "grace_minutes must be a whole number of minutes.",
                    field="grace_minutes",
                )
            value = int(value)
        if not isinstance(value, int):
            raise _fail(
                "invalid_grace_minutes",
                "grace_minutes must be a whole number of minutes.",
                field="grace_minutes",
            )
        if not 0 <= value <= 1440:
            raise _fail(
                "invalid_grace_minutes",
                "grace_minutes must be between 0 and 1440.",
                field="grace_minutes",
            )
        return value

    # ------------------------------------------------------------------
    # Name and delivery
    # ------------------------------------------------------------------
    def _merged_name(
        self,
        args: Mapping[str, Any],
        base: ScheduleSpec | None,
        prompt: str,
        execution: ExecutionSpec,
    ) -> str:
        if "name" in args:
            raw = args["name"]
            if raw is not None and not isinstance(raw, str):
                raise _fail("invalid_name", "name must be text.", field="name")
            candidate = (raw or "").strip()
            if candidate:
                name = candidate
            elif base is not None:
                name = base.name
            else:
                name = self._derive_name(prompt, execution)
        else:
            name = base.name if base is not None else self._derive_name(prompt, execution)
        if len(name) > self.max_name_chars:
            raise _fail("name_too_long", "name is too long.", field="name")
        return name

    @staticmethod
    def _derive_name(prompt: str, execution: ExecutionSpec) -> str:
        if prompt.strip():
            return prompt.strip()[:50].strip() or "scheduled job"
        if execution.skills:
            return execution.skills[0][:50].strip() or "scheduled job"
        if execution.script and execution.no_agent:
            return Path(execution.script).name[:50].strip() or "scheduled job"
        return "scheduled job"

    def _merged_string(
        self, args: Mapping[str, Any], key: str, fallback: str, *, field: str
    ) -> str:
        if key not in args:
            return fallback
        value = args[key]
        if value is None:
            return fallback
        if not isinstance(value, str):
            raise _fail("invalid_field", f"{field} must be text.", field=field)
        return value

    def _resolve_delivery_plans(
        self,
        args: Mapping[str, Any],
        base: ScheduleSpec | None,
        *,
        operation: str,
        resolver: DeliveryResolver,
    ) -> tuple[DeliveryPlan, DeliveryPlan, bool]:
        try:
            if operation == "create":
                result_plan = resolver.resolve_result(
                    args.get("deliver"), mode="create", strict=True
                )
            elif "deliver" in args:
                value = args["deliver"]
                if value is None or (isinstance(value, str) and not value.strip()):
                    result_plan = resolver.resolve_result(None, mode="runtime", strict=True)
                else:
                    result_plan = resolver.resolve_result(value, mode="create", strict=True)
            else:
                result_plan = DeliveryPlan.configured(
                    base.delivery if base else [], lane="result"
                )

            failure_explicit = False
            if operation == "create":
                raw_failure = args.get("failure_deliver", _MISSING)
                if raw_failure is _MISSING or raw_failure is None:
                    failure_plan = result_plan
                elif isinstance(raw_failure, str) and not raw_failure.strip():
                    failure_plan = result_plan
                else:
                    failure_plan = resolver.resolve_failure(
                        raw_failure, mode="create", strict=True
                    )
                    failure_explicit = True
            elif "failure_deliver" in args:
                value = args["failure_deliver"]
                if value is None or (isinstance(value, str) and not value.strip()):
                    failure_plan = result_plan
                else:
                    failure_plan = resolver.resolve_failure(
                        value, mode="create", strict=True
                    )
                    failure_explicit = True
            elif base is not None and base.failure_delivery:
                failure_plan = DeliveryPlan.configured(
                    base.failure_delivery, lane="failure"
                )
                failure_explicit = True
            else:
                # No explicit failure lane was ever stored: failures follow the
                # (possibly just-edited) result delivery.
                failure_plan = result_plan
        except DeliveryResolutionError as exc:
            raise _fail(exc.code, exc.message, field="deliver") from None
        return result_plan, failure_plan, failure_explicit

    # ------------------------------------------------------------------
    # Digests and helpers
    # ------------------------------------------------------------------
    def _scan_field(self, field: str, value: str, *, scope: str = "prompt") -> None:
        if not value:
            return
        try:
            require_safe(value, field=field, scope=scope)
        except ScheduleSafetyError as exc:
            raise _fail(exc.code, exc.message, field=field) from None

    def _now(self, now: datetime | None) -> datetime:
        moment = now if now is not None else self.clock.now()
        if moment.tzinfo is None:
            return moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)

    @staticmethod
    def _coerce_base(base: ScheduleSpec | Mapping[str, Any] | Any) -> ScheduleSpec:
        if isinstance(base, ScheduleSpec):
            return base
        if isinstance(base, Mapping):
            try:
                return ScheduleSpec.model_validate(base)
            except ValueError as exc:
                raise _fail("invalid_base", str(exc), field="base") from None
        inner = getattr(base, "spec", None)
        if isinstance(inner, ScheduleSpec):
            return inner
        raise _fail("invalid_base", "The update base must be a ScheduleSpec.", field="base")

    @staticmethod
    def _target_payload(target: DeliveryTarget) -> dict[str, Any]:
        data = target.model_dump(mode="json", exclude_none=False)
        # Labels are presentation copy, not schedule semantics.
        data.pop("label", None)
        return data

    @staticmethod
    def _schedule_semantic(expression: ScheduleExpression | ParsedSchedule) -> dict[str, Any]:
        if isinstance(expression, ParsedSchedule):
            kind = expression.kind
            body = expression.expression or None
            timezone_name = expression.timezone
            run_at = (
                expression.run_at.astimezone(UTC).isoformat()
                if expression.run_at is not None
                else None
            )
        else:
            kind = expression.kind
            body = expression.expression
            timezone_name = expression.timezone
            run_at = expression.run_at
            if run_at:
                moment = _stored_moment(run_at)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=UTC)
                run_at = moment.astimezone(UTC).isoformat()
        return {
            "kind": kind,
            "expression": body,
            "run_at": run_at,
            "timezone": timezone_name,
        }

    def _semantic_payload(self, spec: ScheduleSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "schedule": self._schedule_semantic(spec.expression),
            "prompt": spec.prompt,
            "repeat_limit": spec.repeat_limit,
            "delivery": [self._target_payload(item) for item in spec.delivery],
            "failure_delivery": [self._target_payload(item) for item in spec.failure_delivery],
            "execution": spec.execution.model_dump(mode="json", exclude_none=False),
            "paused": spec.paused,
            "paused_reason": spec.paused_reason,
            "catch_up": spec.catch_up,
        }

    def _config_payload(
        self,
        spec: ScheduleSpec,
        parsed: ParsedSchedule,
        display: str,
        *,
        skill_snapshot: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = self._semantic_payload(spec)
        payload["schedule"] = {**payload["schedule"], "display": display}
        # The snapshot is execution evidence, not schedule semantics; it must
        # not participate in the semantic digest used for no-op detection.
        if skill_snapshot:
            payload["skill_snapshot"] = dict(skill_snapshot)
        return payload


__all__ = [
    "CompileResult",
    "MODEL_FIELDS",
    "OPERATOR_FIELDS",
    "ScheduleCompileError",
    "ScheduleCompiler",
]
