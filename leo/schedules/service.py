"""The one create/update/remove/run/cancel/propose authority for schedules.

Execution is still owned by :class:`leo.tasks.scheduler.Scheduler` and
:class:`leo.tasks.engine.DurableTaskEngine`; this service only decides whether
a request may become a durable row, and records what really happened.

Three authority paths exist, in priority order:

1. Trusted operator direct (``operator:cli``, authenticated ``operator:api`` and
   same-origin ``operator:dashboard``): commit immediately, unasked, audited.
2. A live schedule-create grant: commit a *create* without a card. Grants never
   cover edits, removal, run, cancel or future tools.
3. Model-initiated create/edit/remove: reserve a durable proposal and either
   present a full card on an answerable surface, or fail closed to
   ``approval_required`` with no schedule write and no invisible approval row.

The service never opens a second database authority and never writes a parallel
schedule store.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import db
from ..contracts import Decision, RiskLevel, TurnContext, current_turn_context
from ..security.approval_cards import (
    KIND_SCHEDULE_CREATE,
    KIND_SCHEDULE_EDIT,
    ApprovalService,
    ScheduleBodySpec,
    card_from_row,
    delivery_description,
    render_edit_body,
    render_schedule_body,
)
from .clock import Clock, SystemClock
from .compiler import CompileResult, ScheduleCompileError, ScheduleCompiler
from .delivery import DeliveryContext, Origin, delivery_context_from_settings
from .domain import (
    DeliveryTarget,
    ScheduleProposal,
    ScheduleSpec,
    ScheduleView,
    new_schedule_id,
)
from .outcome import ScheduleOutcome, record_schedule_outcome
from .proposals import (
    build_create_proposal,
    build_remove_proposal,
    build_update_proposal,
    changes_between,
    commit_meta,
    remove_digest,
    schedule_payload,
    spec_from_view,
    strip_commit_meta,
    with_commit_meta,
)
from .store import ScheduleStore

_OPERATOR_PREFIX = "operator"
_CREATE = "create"
_UPDATE = "update"
_REMOVE = "remove"
_RUN = "run_now"
_CANCEL = "cancel"


class ScheduleServiceError(ValueError):
    """A stable, user-safe schedule rejection."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field


def _fail(code: str, message: str, *, field: str | None = None) -> ScheduleServiceError:
    return ScheduleServiceError(code, message, field=field)


class ScheduleService:
    """One schedule mutation authority; no parallel store or executor."""

    def __init__(
        self,
        *,
        store: ScheduleStore,
        scheduler: Any,
        engine: Any,
        permissions: Any,
        events: Any = None,
        settings: Any = None,
        clock: Clock | None = None,
        scripts_root: str | Path | None = None,
        allowed_workdir_roots: Sequence[str | Path] | None = None,
        available_toolsets: Sequence[str] | None = None,
        default_timezone: str | None = None,
        minimum_lead_seconds: int = 120,
        pinned_provider: str | None = None,
        pinned_model: str | None = None,
        skills: Any = None,
        adapters: Mapping[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.scheduler = scheduler
        self.engine = engine
        self.permissions = permissions
        self.events = events
        self.settings = settings
        self.clock = clock or getattr(store, "clock", None) or SystemClock()
        self.scripts_root = scripts_root
        self.allowed_workdir_roots = tuple(allowed_workdir_roots or ())
        self.available_toolsets = (
            tuple(str(item) for item in available_toolsets)
            if available_toolsets is not None
            else None
        )
        self.default_timezone = default_timezone
        self.minimum_lead_seconds = int(minimum_lead_seconds)
        self.pinned_provider = pinned_provider
        self.pinned_model = pinned_model
        # Optional live skill registry; documentation is snapshotted into the
        # durable config at create/update time, never read at run time.
        self.skills = skills
        self.adapters = adapters
        self.approval = ApprovalService(permissions, committer=self.commit)
        # The Phase 4 resolver calls this exactly once, inside its transaction.
        if hasattr(permissions, "set_schedule_committer"):
            permissions.set_schedule_committer(self.commit)

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------
    async def get(
        self, *, schedule_id: str | None = None, name: str | None = None
    ) -> dict[str, Any]:
        view = await self._resolve_view(schedule_id=schedule_id, name=name)
        return {"schedule": view.model_dump(mode="json"), "found": True}

    async def list(
        self, *, enabled_only: bool = False, include_paused: bool = True
    ) -> dict[str, Any]:
        views = await self.store.list(
            enabled_only=enabled_only, include_paused=include_paused
        )
        schedules = [view.model_dump(mode="json") for view in views]
        return {
            "schedules": schedules,
            "count": len(schedules),
            "enabled_only": bool(enabled_only),
        }

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    async def create(
        self,
        arguments: Mapping[str, Any],
        *,
        actor: str | None = None,
        turn: TurnContext | None = None,
        request_key: str | None = None,
        tool_call_id: str | None = None,
        run_id: str | None = None,
        operator: bool = False,
    ) -> dict[str, Any]:
        turn = turn or current_turn_context()
        actor = self._actor(actor, turn)
        principal = self._principal(turn, actor)
        is_operator = operator or self._is_operator(turn, actor)

        replay = await self._replay_request(principal, _CREATE, request_key, turn)
        if replay is not None:
            return replay

        compiler = await self._compiler()
        result = compiler.compile_create(
            arguments,
            operator=is_operator,
            delivery_context=self._delivery_context(turn),
        )
        if await self.store.name_is_active(result.spec.name):
            raise _fail(
                "duplicate_name",
                f"An active schedule named {result.spec.name!r} already exists.",
                field="name",
            )

        proposal = build_create_proposal(
            spec=with_commit_meta(result.spec, result, actor=actor),
            principal=principal,
            origin=self._origin(turn, actor),
            conversation_id=self._conversation(turn),
            run_id=run_id,
            tool_call_id=tool_call_id,
            request_key=request_key,
            digest=result.proposal_digest,
            schedule_id=new_schedule_id("sch"),
        )

        card = None
        outcome: ScheduleOutcome
        extra: dict[str, Any] = {}
        async with db.transaction():
            stored = await self._create_proposal(proposal)
            if is_operator or await self._has_create_grant(principal, turn, actor=actor):
                await self._commit_create(stored, actor=actor)
                await self.store.update_proposal(
                    stored.id, status="committed", schedule_id=stored.schedule_id
                )
                await self._audit(
                    actor if is_operator else f"grant:{actor}",
                    "schedule.created",
                    stored.schedule_id,
                    "operator direct create" if is_operator else "create grant",
                    {"schedule_id": stored.schedule_id, "name": result.spec.name},
                )
                outcome = self._outcome(stored, "created", result.spec.name, result)
            elif self._can_answer(turn):
                card = await self._present_create_card(stored, result, turn, actor)
                outcome = self._outcome(
                    stored, "approval_required", result.spec.name, result
                )
                extra["card"] = card.model_dump(mode="json")
            else:
                outcome = self._outcome(
                    stored, "approval_required", result.spec.name, result
                )
        # Recording happens only after the transaction committed, so a failed
        # write can never be remembered as a successful schedule outcome.
        return self._finish(outcome, **extra)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    async def update(
        self,
        *,
        schedule_id: str | None = None,
        name: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        actor: str | None = None,
        turn: TurnContext | None = None,
        request_key: str | None = None,
        tool_call_id: str | None = None,
        run_id: str | None = None,
        expected_config_revision: int | None = None,
    ) -> dict[str, Any]:
        turn = turn or current_turn_context()
        actor = self._actor(actor, turn)
        principal = self._principal(turn, actor)
        is_operator = self._is_operator(turn, actor)
        view = await self._resolve_view(schedule_id=schedule_id, name=name)
        if (
            expected_config_revision is not None
            and int(expected_config_revision) != int(view.config_revision)
        ):
            raise _fail(
                "stale_update",
                f"Schedule config revision is {view.config_revision}, not "
                f"{expected_config_revision}.",
                field="config_revision",
            )

        replay = await self._replay_request(principal, _UPDATE, request_key, turn)
        if replay is not None:
            return replay

        base_spec = spec_from_view(view)
        compiler = await self._compiler()
        result = compiler.compile_update(
            base_spec,
            dict(arguments or {}),
            operator=is_operator,
            base_revision=view.config_revision,
            schedule_id=view.id,
            current_next_run=view.next_run_at,
            delivery_context=self._delivery_context(turn),
            # Explicit pause/resume tools are the only callers that send these
            # fields on an update; their consent and CAS path is unchanged.
            allow_pause_edit=(
                "paused" in (arguments or {})
                or "paused_reason" in (arguments or {})
            ),
        )
        if not result.changed:
            outcome = ScheduleOutcome(
                operation=_UPDATE,
                status="no_change",
                name=view.name,
                schedule_id=view.id,
                message="the requested change is semantically identical",
            )
            return self._finish(outcome)

        proposal = build_update_proposal(
            spec=with_commit_meta(result.spec, result, actor=actor),
            principal=principal,
            origin=self._origin(turn, actor),
            conversation_id=self._conversation(turn),
            run_id=run_id,
            tool_call_id=tool_call_id,
            request_key=request_key,
            digest=result.proposal_digest,
            schedule_id=view.id,
            base_revision=view.config_revision,
        )
        changes = changes_between(base_spec, result.spec)
        outcome: ScheduleOutcome
        extra: dict[str, Any] = {"changes": changes}

        async with db.transaction():
            stored = await self._create_proposal(proposal)
            if is_operator:
                await self._commit_update(stored)
                await self.store.update_proposal(stored.id, status="committed")
                await self._audit(
                    actor, "schedule.updated", stored.schedule_id,
                    "operator direct update",
                    {
                        "schedule_id": stored.schedule_id,
                        "name": result.spec.name,
                        "changes": changes,
                    },
                )
                outcome = self._outcome(stored, "updated", result.spec.name, result)
            elif self._can_answer(turn):
                card = await self._present_edit_card(
                    stored, result, turn, actor, changes, schedule_id=view.id
                )
                outcome = self._outcome(
                    stored, "approval_required", result.spec.name, result
                )
                extra["card"] = card.model_dump(mode="json")
            else:
                outcome = self._outcome(
                    stored, "approval_required", result.spec.name, result
                )
        return self._finish(outcome, **extra)

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------
    async def remove(
        self,
        *,
        schedule_id: str | None = None,
        name: str | None = None,
        actor: str | None = None,
        turn: TurnContext | None = None,
        request_key: str | None = None,
        tool_call_id: str | None = None,
        run_id: str | None = None,
        expected_config_revision: int | None = None,
        authorized_by: str | None = None,
    ) -> dict[str, Any]:
        turn = turn or current_turn_context()
        actor = self._actor(actor, turn)
        principal = self._principal(turn, actor)
        # ``authorized_by`` is set only by the automation specialist after its
        # own durable ``automation:mutate`` approval was re-verified. It is the
        # exact existing gate for delete, not the operator path and not a card.
        is_operator = self._is_operator(turn, actor) or bool(authorized_by)
        if authorized_by:
            approval = await db.fetchone(
                "SELECT status FROM approvals WHERE id=?", (str(authorized_by),)
            )
            if approval is None or str(approval["status"]) != "approved":
                raise _fail(
                    "invalid_authorization",
                    "The supplied mutation approval is not approved.",
                    field="authorization",
                )
        view = await self._resolve_view(schedule_id=schedule_id, name=name)
        if (
            expected_config_revision is not None
            and int(expected_config_revision) != int(view.config_revision)
        ):
            raise _fail(
                "stale_update",
                f"Schedule config revision is {view.config_revision}, not "
                f"{expected_config_revision}.",
                field="config_revision",
            )

        replay = await self._replay_request(principal, _REMOVE, request_key, turn)
        if replay is not None:
            return replay

        spec = spec_from_view(view)
        proposal = build_remove_proposal(
            spec=spec,
            principal=principal,
            origin=self._origin(turn, actor),
            conversation_id=self._conversation(turn),
            run_id=run_id,
            tool_call_id=tool_call_id,
            request_key=request_key,
            digest=remove_digest(spec, view.id, view.config_revision),
            schedule_id=view.id,
            base_revision=view.config_revision,
        )
        changes = ["remove this schedule"]
        outcome: ScheduleOutcome
        extra: dict[str, Any] = {"changes": changes}

        async with db.transaction():
            stored = await self._create_proposal(proposal)
            if is_operator:
                await self._commit_remove(stored)
                await self.store.update_proposal(stored.id, status="committed")
                await self._audit(
                    actor, "schedule.removed", stored.schedule_id,
                    (
                        "specialist-approved removal"
                        if authorized_by
                        else "operator direct removal"
                    ),
                    {
                        "schedule_id": stored.schedule_id,
                        **({"authorized_by": authorized_by} if authorized_by else {}),
                    },
                )
                outcome = self._outcome(stored, "removed", view.name, None)
            elif self._can_answer(turn):
                card = await self._present_edit_card(
                    stored, None, turn, actor, changes,
                    schedule_id=view.id, config_revision=view.config_revision,
                )
                outcome = self._outcome(
                    stored, "approval_required", view.name, None
                )
                extra["card"] = card.model_dump(mode="json")
            else:
                outcome = self._outcome(
                    stored, "approval_required", view.name, None
                )
        return self._finish(outcome, **extra)

    # ------------------------------------------------------------------
    # Run / cancel (never definition changes)
    # ------------------------------------------------------------------
    async def run_now(
        self,
        *,
        schedule_id: str | None = None,
        name: str | None = None,
        prompt: str | None = None,
        run_at: str | datetime | None = None,
        request_key: str | None = None,
        actor: str | None = None,
        turn: TurnContext | None = None,
    ) -> dict[str, Any]:
        turn = turn or current_turn_context()
        actor = self._actor(actor, turn)
        view = await self._resolve_view(schedule_id=schedule_id, name=name)
        if not view.enabled or view.status != "active":
            raise _fail(
                "schedule_disabled",
                f"Schedule {view.name!r} is not active.",
                field="name",
            )
        spec = spec_from_view(view)
        objective = schedule_payload(spec)["objective"]
        transient = str(prompt or "").strip()
        if transient:
            objective = f"{objective}\n\nOne-off context: {transient}"

        moment = self._parse_run_at(run_at)
        key = request_key or moment.isoformat()
        metadata = {
            "schedule": view.name,
            "schedule_id": view.id,
            "scheduled_for": moment.isoformat(),
            "manual_run": True,
            "schedule_payload": schedule_payload(spec),
        }
        task = await self.engine.create(
            objective,
            metadata=metadata,
            idempotency_key=f"schedule:{view.id}:manual:{key}",
        )
        status = getattr(task.status, "value", task.status)
        await self._audit(
            actor, "schedule.run", view.id, "manual background run",
            {"schedule_id": view.id, "task_id": task.id, "manual_run": True},
        )
        if self.events is not None:
            await self.events.emit(
                "schedule.triggered",
                name=view.name,
                schedule_id=view.id,
                task_id=task.id,
                task_count=1,
                manual_run=True,
                scheduled_for=moment.isoformat(),
            )
        outcome = ScheduleOutcome(
            operation=_RUN,
            status="running",
            name=view.name,
            schedule_id=view.id,
            message="background task created; no repeat or cadence was consumed",
            detail={"task_id": task.id, "task_status": str(status), "manual_run": True},
        )
        return self._finish(
            outcome,
            task_id=task.id,
            task_status=str(status),
            triggered=True,
            task_count=1,
            manual_run=True,
            executed=False,
            run_at=moment.isoformat(),
        )

    async def cancel(
        self,
        *,
        schedule_id: str | None = None,
        name: str | None = None,
        task_id: str | None = None,
        actor: str | None = None,
        turn: TurnContext | None = None,
    ) -> dict[str, Any]:
        turn = turn or current_turn_context()
        actor = self._actor(actor, turn)
        view = await self._resolve_view(schedule_id=schedule_id, name=name)
        if task_id:
            rows = await db.fetchall(
                "SELECT id,status,metadata FROM tasks WHERE id=?", (str(task_id),)
            )
        else:
            rows = await db.fetchall(
                "SELECT id,status,metadata FROM tasks "
                "WHERE status NOT IN ('completed','failed','cancelled') "
                "AND (json_extract(metadata,'$.schedule_id')=? "
                "OR json_extract(metadata,'$.schedule')=?)",
                (view.id, view.name),
            )
        cancelled: list[dict[str, Any]] = []
        for row in rows or []:
            metadata = self._load_json(row["metadata"])
            if task_id:
                if str(metadata.get("schedule_id") or "") != view.id and str(
                    metadata.get("schedule") or ""
                ) != view.name:
                    raise _fail(
                        "task_not_for_schedule",
                        "The task does not belong to this schedule.",
                        field="task_id",
                    )
            if str(row["status"]) in ("completed", "failed", "cancelled"):
                continue
            try:
                await self.engine.cancel(str(row["id"]))
            except ValueError:
                continue
            cancelled.append({"task_id": str(row["id"]), "status": "cancelled"})
        await self._audit(
            actor, "schedule.cancel", view.id, "stop live run only",
            {
                "schedule_id": view.id,
                "tasks": [item["task_id"] for item in cancelled],
            },
        )
        if self.events is not None:
            await self.events.emit(
                "schedule.cancelled",
                name=view.name,
                schedule_id=view.id,
                tasks=[item["task_id"] for item in cancelled],
            )
        status = "cancelled" if cancelled else "no_live_run"
        outcome = ScheduleOutcome(
            operation=_CANCEL,
            status=status,
            name=view.name,
            schedule_id=view.id,
            message="live tasks stopped; the schedule itself was not deleted",
            detail={"tasks": cancelled},
        )
        return self._finish(
            outcome,
            cancelled=bool(cancelled),
            tasks=cancelled,
            schedule_deleted=False,
        )

    async def withdraw_turn(
        self,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
        reason: str = "turn stopped",
    ) -> dict[str, Any]:
        """Withdraw pending cards/proposals for a stopped turn without committing.

        Stopping a turn must never leave an invisible pending schedule: the
        exact rows created by that run are marked ``withdrawn`` (releasing the
        active-name reservation) and no schedule is written. A later decision
        cannot resolve them because every resolver requires ``pending``.
        """
        run_id = str(run_id).strip() if run_id else None
        conversation_id = (
            str(conversation_id).strip() if conversation_id else None
        )
        if not run_id and not conversation_id:
            return {
                "withdrawn": 0,
                "proposals": 0,
                "approvals": 0,
                "schedule_written": False,
            }
        now = self._now_iso()
        async with db.transaction():
            proposal_clauses = ["status='pending'"]
            if run_id:
                proposal_clauses.append("run_id=?")
            if conversation_id:
                proposal_clauses.append("conversation_id=?")
            proposal_params: list[Any] = [reason, now]
            proposal_params.extend(
                value for value in (run_id, conversation_id) if value
            )
            cursor = await db.execute(
                "UPDATE schedule_proposals SET status='withdrawn', error=?, "
                "updated_at=? WHERE " + " AND ".join(proposal_clauses),
                proposal_params,
            )
            proposals = int(getattr(cursor, "rowcount", 0) or 0)

            approval_clauses = ["status='pending'", "kind IN (?,?)"]
            approval_params: list[Any] = [
                now,
                reason,
                KIND_SCHEDULE_CREATE,
                KIND_SCHEDULE_EDIT,
            ]
            if run_id:
                approval_clauses.append("run_id=?")
                approval_params.append(run_id)
            if conversation_id:
                approval_clauses.append("conversation_id=?")
                approval_params.append(conversation_id)
            cursor = await db.execute(
                "UPDATE approvals SET status='withdrawn', answered_at=?, "
                "deny_reason=? WHERE " + " AND ".join(approval_clauses),
                approval_params,
            )
            approvals = int(getattr(cursor, "rowcount", 0) or 0)
        if self.events is not None and (proposals or approvals):
            await self.events.emit(
                "schedule.withdrawn",
                run_id=run_id,
                conversation_id=conversation_id,
                proposals=proposals,
                approvals=approvals,
                reason=reason,
            )
        return {
            "withdrawn": proposals + approvals,
            "proposals": proposals,
            "approvals": approvals,
            "schedule_written": False,
        }

    # ------------------------------------------------------------------
    # Approval commit callback (runs inside the decision transaction)
    # ------------------------------------------------------------------
    async def commit(self, *, card: Any, choice: str, resolver: str) -> dict[str, Any]:
        proposal_id = str(getattr(card, "proposal_id", "") or "")
        if not proposal_id:
            # Legacy/Phase 4 card proxies carry no proposal. The card decision
            # is still durable and audited by the PermissionManager; there is
            # simply no schedule to commit here.
            return {"schedule_id": getattr(card, "schedule_id", None), "legacy": True}
        proposal = await self.store.get_proposal(proposal_id)
        if proposal is None:
            # A legacy card created before proposals existed (or a test proxy)
            # remains resolvable, but it writes no schedule.
            return {"schedule_id": getattr(card, "schedule_id", None), "legacy": True}
        if proposal.status == "committed":
            return {
                "schedule_id": proposal.schedule_id,
                "operation": proposal.operation,
                "idempotent": True,
            }
        if proposal.status != "pending":
            raise _fail(
                "proposal_not_pending",
                f"Proposal {proposal_id!r} is {proposal.status}.",
            )
        card_digest = str(getattr(card, "proposal_digest", "") or "")
        if card_digest and card_digest != proposal.digest:
            raise _fail(
                "digest_mismatch",
                "The card digest does not match the reserved proposal.",
            )
        if proposal.operation == _CREATE:
            schedule_id = await self._commit_create(proposal, actor=resolver)
            await self.store.update_proposal(
                proposal.id, status="committed", schedule_id=schedule_id
            )
        elif proposal.operation == _UPDATE:
            schedule_id = await self._commit_update(proposal)
            await self.store.update_proposal(proposal.id, status="committed")
        elif proposal.operation == _REMOVE:
            schedule_id = await self._commit_remove(proposal)
            await self.store.update_proposal(proposal.id, status="committed")
        else:
            raise _fail(
                "invalid_operation",
                f"Proposal operation {proposal.operation!r} cannot commit.",
            )
        return {
            "schedule_id": schedule_id,
            "operation": proposal.operation,
            "name": proposal.spec.name,
        }

    # ------------------------------------------------------------------
    # Durable writes (caller owns the transaction)
    # ------------------------------------------------------------------
    async def _commit_create(
        self, proposal: ScheduleProposal, *, actor: str
    ) -> str:
        spec = proposal.spec
        schedule_id = proposal.schedule_id or new_schedule_id("sch")
        if await self.store.name_is_active(spec.name):
            raise _fail(
                "duplicate_name",
                f"An active schedule named {spec.name!r} already exists.",
                field="name",
            )
        meta = commit_meta(proposal)
        revision = int(meta.get("config_revision", 1) or 1)
        next_run = None if spec.paused else meta.get("next_run_at")
        now = self._now_iso()
        config = strip_commit_meta(spec.config)
        try:
            await db.execute(
                "INSERT INTO schedules("
                "id,name,kind,expression,payload,enabled,next_run_at,last_run_at,"
                "created_at,schedule_type,run_at,timezone,repeat_limit,claimed_count,"
                "completed_count,config_revision,grace_minutes,config,status,created_by,"
                "origin,consent_digest,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    schedule_id,
                    spec.name,
                    spec.expression.kind,
                    self._row_expression(spec),
                    json.dumps(schedule_payload(spec), default=str, ensure_ascii=False),
                    0 if spec.paused else 1,
                    next_run,
                    None,
                    now,
                    spec.expression.kind,
                    spec.expression.run_at,
                    spec.expression.timezone,
                    spec.repeat_limit,
                    0,
                    0,
                    revision,
                    spec.execution.grace_minutes,
                    json.dumps(config, default=str, ensure_ascii=False),
                    "paused" if spec.paused else "active",
                    str(meta.get("actor") or proposal.principal),
                    proposal.origin,
                    proposal.digest,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise _fail(
                "duplicate_name",
                f"An active schedule named {spec.name!r} already exists.",
                field="name",
            ) from exc
        if self.events is not None:
            await self.events.emit(
                "schedule.created",
                schedule_id=schedule_id,
                name=spec.name,
                config_revision=revision,
                origin=proposal.origin,
            )
        return schedule_id

    async def _commit_update(self, proposal: ScheduleProposal) -> str:
        schedule_id = str(proposal.schedule_id or "")
        current = await self.store.get(schedule_id)
        if current is None:
            raise _fail("unknown_schedule", f"Unknown schedule {schedule_id!r}.")
        expected = int(proposal.config_revision or current.config_revision)
        if int(current.config_revision) != expected:
            raise _fail(
                "stale_update",
                f"Schedule was changed at revision {current.config_revision}; "
                f"this proposal targeted {expected}.",
                field="config_revision",
            )
        spec = proposal.spec
        meta = commit_meta(proposal)
        revision = int(meta.get("config_revision", expected + 1) or expected + 1)
        config = strip_commit_meta(spec.config)
        now = self._now_iso()
        # A paused row is not claimable because ``enabled=0``. Keeping its
        # stored instant lets an approved resume continue the same cadence
        # instead of silently re-anchoring to the resume time.
        next_run = meta.get("next_run_at") or current.next_run_at
        try:
            cursor = await db.execute(
                "UPDATE schedules SET name=?,kind=?,expression=?,payload=?,enabled=?,"
                "next_run_at=?,schedule_type=?,run_at=?,timezone=?,repeat_limit=?,"
                "config_revision=?,grace_minutes=?,config=?,status=?,origin=?,"
                "consent_digest=?,updated_at=? WHERE id=? AND config_revision=?",
                (
                    spec.name,
                    spec.expression.kind,
                    self._row_expression(spec),
                    json.dumps(schedule_payload(spec), default=str, ensure_ascii=False),
                    0 if spec.paused else 1,
                    next_run,
                    spec.expression.kind,
                    spec.expression.run_at,
                    spec.expression.timezone,
                    spec.repeat_limit,
                    revision,
                    spec.execution.grace_minutes,
                    json.dumps(config, default=str, ensure_ascii=False),
                    "paused" if spec.paused else "active",
                    proposal.origin,
                    proposal.digest,
                    now,
                    schedule_id,
                    expected,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise _fail(
                "duplicate_name",
                f"An active schedule named {spec.name!r} already exists.",
                field="name",
            ) from exc
        if int(getattr(cursor, "rowcount", 0) or 0) < 1:
            raise _fail(
                "stale_update",
                "The schedule changed after this edit was proposed.",
                field="config_revision",
            )
        if self.events is not None:
            await self.events.emit(
                "schedule.updated",
                schedule_id=schedule_id,
                name=spec.name,
                config_revision=revision,
            )
        return schedule_id

    async def _commit_remove(self, proposal: ScheduleProposal) -> str:
        schedule_id = str(proposal.schedule_id or "")
        current = await self.store.get(schedule_id)
        if current is None:
            raise _fail("unknown_schedule", f"Unknown schedule {schedule_id!r}.")
        expected = int(proposal.config_revision or current.config_revision)
        if int(current.config_revision) != expected:
            raise _fail(
                "stale_update",
                f"Schedule was changed at revision {current.config_revision}; "
                f"this proposal targeted {expected}.",
                field="config_revision",
            )
        await self.store.add_tombstone(
            current.id,
            current.name,
            "removed",
            payload=current.config,
            origin=proposal.origin,
            created_by=proposal.principal,
            created_at=current.created_at,
            metadata={"config_revision": expected, "consent_digest": proposal.digest},
        )
        cursor = await db.execute(
            "DELETE FROM schedules WHERE id=? AND config_revision=?",
            (schedule_id, expected),
        )
        if int(getattr(cursor, "rowcount", 0) or 0) < 1:
            raise _fail(
                "stale_update",
                "The schedule changed after this removal was proposed.",
                field="config_revision",
            )
        if self.events is not None:
            await self.events.emit(
                "schedule.removed", schedule_id=schedule_id, name=current.name
            )
        return schedule_id

    # ------------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------------
    async def _present_create_card(
        self,
        proposal: ScheduleProposal,
        result: CompileResult,
        turn: TurnContext,
        actor: str,
    ) -> Any:
        spec = result.spec
        body = render_schedule_body(
            ScheduleBodySpec(
                name=spec.name,
                display=result.display,
                first_run_at=result.first_run_at,
                repeat_limit=spec.repeat_limit,
                paused=spec.paused,
                prompt=spec.prompt,
                skills=spec.execution.skills,
                script=spec.execution.script,
                no_agent=spec.execution.no_agent,
                monitor=spec.execution.monitor,
                workdir=spec.execution.workdir,
                context_from=spec.execution.context_from,
                enabled_toolsets=spec.execution.enabled_toolsets,
                attach_to_session=spec.execution.attach_to_session,
                delivery=delivery_description(result.delivery_plan.describe()),
                failure_delivery=delivery_description(
                    result.failure_delivery_plan.describe()
                ),
            )
        )
        card = await self.permissions.create_schedule_card(
            kind=KIND_SCHEDULE_CREATE,
            body=body,
            proposal_digest=proposal.digest,
            proposal_id=proposal.id,
            schedule_id=proposal.schedule_id,
            config_revision=commit_meta(proposal).get("config_revision", 1),
            requested_by=actor,
            conversation_id=proposal.conversation_id,
            run_id=proposal.run_id,
            tool_call_id=proposal.tool_call_id,
            origin=proposal.origin,
            shown_first_run=(
                result.first_run_at.isoformat() if result.first_run_at else None
            ),
        )
        return await self.permissions.present_schedule_card(
            card.approval_id,
            surface=str(turn.surface or turn.channel or "chat"),
            capabilities=turn.capabilities,
        )

    async def _present_edit_card(
        self,
        proposal: ScheduleProposal,
        result: CompileResult | None,
        turn: TurnContext,
        actor: str,
        changes: list[str],
        *,
        schedule_id: str,
        config_revision: int | None = None,
    ) -> Any:
        spec = proposal.spec
        first_run = getattr(result, "first_run_at", None) if result is not None else None
        new_revision = (
            commit_meta(proposal).get("config_revision")
            if result is not None
            else config_revision
        )
        card = await self.permissions.create_schedule_card(
            kind=KIND_SCHEDULE_EDIT,
            body=render_edit_body(spec.name, changes),
            proposal_digest=proposal.digest,
            proposal_id=proposal.id,
            schedule_id=schedule_id,
            config_revision=new_revision,
            requested_by=actor,
            conversation_id=proposal.conversation_id,
            run_id=proposal.run_id,
            tool_call_id=proposal.tool_call_id,
            origin=proposal.origin,
            shown_first_run=(
                first_run.isoformat()
                if first_run is not None and spec.expression.kind == "once"
                else None
            ),
        )
        return await self.permissions.present_schedule_card(
            card.approval_id,
            surface=str(turn.surface or turn.channel or "chat"),
            capabilities=turn.capabilities,
        )

    # ------------------------------------------------------------------
    # Replay / lookup
    # ------------------------------------------------------------------
    async def _replay_request(
        self,
        principal: str,
        operation: str,
        request_key: str | None,
        turn: TurnContext | None,
    ) -> dict[str, Any] | None:
        if not request_key:
            return None
        existing = await self.store.find_proposal_by_request(
            principal, operation, str(request_key)
        )
        if existing is None:
            return None
        name = existing.spec.name
        if existing.status == "pending":
            card_id = await self._pending_card_id(existing.id)
            if card_id and self._can_answer(turn):
                try:
                    await self.permissions.present_schedule_card(
                        card_id,
                        surface=str(turn.surface or turn.channel or "chat"),
                        capabilities=turn.capabilities,
                    )
                except (KeyError, ValueError, PermissionError):
                    pass
            outcome = ScheduleOutcome(
                operation=operation,
                status="approval_required",
                name=name,
                schedule_id=existing.schedule_id,
                approval_id=card_id,
                proposal_id=existing.id,
                message="replayed the already-reserved proposal",
            )
            return self._finish(outcome, replayed=True)
        if existing.status == "committed":
            status = {
                _CREATE: "created",
                _UPDATE: "updated",
                _REMOVE: "removed",
            }.get(operation, "created")
            outcome = ScheduleOutcome(
                operation=operation,
                status=status,
                name=name,
                schedule_id=existing.schedule_id,
                proposal_id=existing.id,
                message="replayed an already-committed request",
            )
            return self._finish(outcome, replayed=True)
        status = {
            "denied": "denied",
            "expired": "timed_out",
            "failed": "failed",
        }.get(existing.status, "failed")
        outcome = ScheduleOutcome(
            operation=operation,
            status=status,
            name=name,
            schedule_id=existing.schedule_id,
            proposal_id=existing.id,
            message=existing.error or f"proposal is {existing.status}",
        )
        return self._finish(outcome, replayed=True)

    async def _pending_card_id(self, proposal_id: str) -> str | None:
        row = await db.fetchone(
            "SELECT id FROM approvals WHERE proposal_id=? AND status='pending' "
            "ORDER BY created_at DESC LIMIT 1",
            (proposal_id,),
        )
        return str(row["id"]) if row is not None else None

    async def _resolve_view(
        self, *, schedule_id: str | None, name: str | None
    ) -> ScheduleView:
        view = None
        if schedule_id:
            view = await self.store.get(str(schedule_id))
        if view is None and name:
            view = await self.store.get_by_name(str(name))
        if view is None:
            label = schedule_id or name or "schedule"
            raise _fail("unknown_schedule", f"Unknown schedule {label!r}.")
        return view

    async def _create_proposal(self, proposal: ScheduleProposal) -> ScheduleProposal:
        try:
            return await self.store.create_proposal(proposal)
        except sqlite3.IntegrityError as exc:
            text = str(exc).lower()
            if "request" in text or "unique" in text:
                existing = None
                if proposal.request_key:
                    existing = await self.store.find_proposal_by_request(
                        proposal.principal, proposal.operation, proposal.request_key
                    )
                if existing is not None:
                    return existing
            if "active_name" in text or "name" in text:
                raise _fail(
                    "duplicate_name",
                    f"An active schedule named {proposal.spec.name!r} already exists.",
                    field="name",
                ) from exc
            raise _fail("proposal_conflict", "The schedule proposal conflicted with an existing row.") from exc

    # ------------------------------------------------------------------
    # Authority helpers
    # ------------------------------------------------------------------
    async def _compiler(self) -> ScheduleCompiler:
        views = await self.store.list()
        known = [view.id for view in views]
        references: dict[str, Sequence[str]] = {}
        for view in views:
            spec = spec_from_view(view)
            references[view.id] = tuple(spec.execution.context_from)
        return ScheduleCompiler(
            clock=self.clock,
            settings=self.settings,
            default_timezone=self.default_timezone,
            minimum_lead_seconds=self.minimum_lead_seconds,
            scripts_root=self.scripts_root,
            allowed_workdir_roots=self.allowed_workdir_roots,
            available_toolsets=self.available_toolsets,
            known_schedule_ids=known,
            reference_schedules=references,
            pinned_provider=self.pinned_provider,
            pinned_model=self.pinned_model,
            skill_docs=await self._skill_docs(),
            delivery_context=self._delivery_context(None),
        )

    async def _skill_docs(self) -> dict[str, str]:
        """Current skill documentation for the creation-time snapshot."""
        if self.skills is None:
            return {}
        lister = getattr(self.skills, "list", None)
        if not callable(lister):
            return {}
        try:
            records = await lister()
        except Exception:  # noqa: BLE001 - a snapshot must not block scheduling
            return {}
        docs: dict[str, str] = {}
        for record in records or []:
            name = str(getattr(record, "name", "") or "")
            if not name:
                continue
            documentation = str(getattr(record, "documentation", "") or "")
            description = str(getattr(record, "description", "") or "")
            text = documentation.strip() or description.strip()
            if text:
                docs[name] = text
        return docs

    def _delivery_context(self, turn: TurnContext | None) -> DeliveryContext:
        origin: Origin | None = None
        if turn is not None:
            origin = Origin(
                platform=str(turn.channel or turn.surface or "chat"),
                chat_id=str(turn.conversation_id or turn.target or turn.principal or "chat"),
                thread_id=turn.thread_id,
            )
        return delivery_context_from_settings(
            self.settings, origin=origin, adapters=self.adapters
        )

    @staticmethod
    def _actor(actor: str | None, turn: TurnContext | None) -> str:
        if actor:
            return str(actor)
        if turn is not None and turn.principal:
            return str(turn.principal)
        return "chat"

    @staticmethod
    def _principal(turn: TurnContext | None, actor: str) -> str:
        if turn is not None and turn.principal:
            return str(turn.principal)
        return str(actor or "chat")

    @staticmethod
    def _origin(turn: TurnContext | None, actor: str) -> str | None:
        if turn is not None:
            return turn.origin or turn.channel or None
        return None

    @staticmethod
    def _conversation(turn: TurnContext | None) -> str | None:
        return str(turn.conversation_id) if turn is not None and turn.conversation_id else None

    @staticmethod
    def _is_operator(turn: TurnContext | None, actor: str) -> bool:
        if turn is not None:
            if str(turn.source or "") == "operator":
                return True
            principal = str(turn.principal or "")
            if principal == _OPERATOR_PREFIX or principal.startswith(
                f"{_OPERATOR_PREFIX}:"
            ):
                return True
        text = str(actor or "")
        return text == _OPERATOR_PREFIX or text.startswith(f"{_OPERATOR_PREFIX}:")

    @staticmethod
    def _can_answer(turn: TurnContext | None) -> bool:
        if turn is None:
            return False
        capabilities = turn.capabilities
        return bool(
            capabilities.can_present_card
            and capabilities.can_wait_for_decision
            and capabilities.full_text
        )

    async def _has_create_grant(
        self, principal: str, turn: TurnContext | None, *, actor: str | None = None
    ) -> bool:
        checker = getattr(self.permissions, "has_live_schedule_create_grant", None)
        if not callable(checker):
            return False
        conversation_id = turn.conversation_id if turn is not None else None
        # A specialist card is *requested by* its fixed agent actor, so a
        # session/always grant created from that card is keyed by the actor.
        # Check the human principal first, then the fixed actor; both are
        # explicit durable grant subjects, never inferred from the request body.
        candidates = [str(principal or "")]
        if actor and str(actor) != str(principal):
            candidates.append(str(actor))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                if await checker(
                    principal=candidate,
                    conversation_id=conversation_id,
                    profile=candidate,
                ):
                    return True
            except Exception:  # noqa: BLE001 - a grant lookup must fail closed
                return False
        return False

    # ------------------------------------------------------------------
    # Result / audit / small helpers
    # ------------------------------------------------------------------
    def _outcome(
        self,
        proposal: ScheduleProposal,
        status: str,
        name: str,
        result: CompileResult | None,
    ) -> ScheduleOutcome:
        message = ""
        if status == "approval_required":
            message = "a card or durable proposal is waiting for an operator decision"
        elif result is not None:
            message = result.display
        return ScheduleOutcome(
            operation=proposal.operation,
            status=status,
            name=name,
            schedule_id=proposal.schedule_id,
            proposal_id=proposal.id,
            message=message,
        )

    def _finish(self, outcome: ScheduleOutcome, **extra: Any) -> dict[str, Any]:
        card = extra.get("card")
        if isinstance(card, Mapping):
            outcome.approval_id = outcome.approval_id or card.get("approval_id")
            outcome.proposal_id = outcome.proposal_id or card.get("proposal_id")
        record_schedule_outcome(outcome)
        data = outcome.as_dict()
        data.setdefault("scheduled", outcome.successful)
        data.setdefault("applied", outcome.successful)
        data.setdefault("card", extra.get("card"))
        data.update(extra)
        return data

    async def _audit(
        self,
        actor: str,
        action: str,
        resource: str | None,
        reason: str,
        metadata: Mapping[str, Any],
    ) -> None:
        decision = Decision(allowed=True, reason=reason, risk=RiskLevel.MEDIUM)
        recorder = getattr(self.permissions, "record_audit", None)
        if callable(recorder):
            await recorder(
                actor, action, resource, RiskLevel.MEDIUM, decision,
                metadata=dict(metadata),
            )
            return
        await db.execute(
            "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                self._now_iso(), actor, action, resource, RiskLevel.MEDIUM.value,
                "allowed", reason,
                json.dumps(dict(metadata), default=str, ensure_ascii=False),
            ),
        )

    def _parse_run_at(self, value: str | datetime | None) -> datetime:
        if value is None:
            return self._now()
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if moment.tzinfo is None:
            from datetime import timezone

            moment = moment.replace(tzinfo=timezone.utc)
        from datetime import timezone

        return moment.astimezone(timezone.utc)

    def _now(self) -> datetime:
        return self.clock.now()

    def _now_iso(self) -> str:
        return self._now().isoformat()

    @staticmethod
    def _row_expression(spec: ScheduleSpec) -> str:
        if spec.expression.kind == "once":
            return str(spec.expression.run_at or "")
        return str(spec.expression.expression or "")

    @staticmethod
    def _load_json(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        try:
            loaded = json.loads(value or "{}")
        except (TypeError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}


__all__ = ["ScheduleService", "ScheduleServiceError", "ScheduleCompileError"]
