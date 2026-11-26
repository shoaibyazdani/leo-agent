"""Compiler field admission, merged-edit validation and lead-time policy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompileError, ScheduleCompiler
from leo.schedules.delivery import DeliveryContext, Origin
from leo.schedules.domain import ScheduleView
from leo.schedules.proposals import spec_from_view

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)
BASE = {"schedule": "every day at 9am", "prompt": "Brief", "deliver": "local"}


@pytest.fixture
def roots(tmp_path):
    (tmp_path / "job.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    workdir = tmp_path / "work"
    workdir.mkdir()
    return tmp_path, workdir


def _compiler(roots, **overrides) -> ScheduleCompiler:
    scripts_root, workdir = roots
    values = {
        "scripts_root": scripts_root,
        "allowed_workdir_roots": [scripts_root],
        "known_schedule_ids": {"sch_a", "sch_b"},
        "available_toolsets": {"web", "terminal"},
        "pinned_provider": "provider-x",
        "pinned_model": "model-y",
        "delivery_context": DeliveryContext(origin=Origin("telegram", "1")),
    }
    values.update(overrides)
    return ScheduleCompiler(FrozenClock(NOW), default_timezone="UTC", **values)


def _create(roots, **overrides):
    args = dict(BASE)
    args.update(overrides)
    return _compiler(roots).compile_create(args)


def _error(roots, code: str, **overrides):
    with pytest.raises(ScheduleCompileError) as caught:
        _create(roots, **overrides)
    assert caught.value.code == code
    return caught.value


def test_full_surviving_field_set_compiles(roots) -> None:
    scripts_root, workdir = roots
    result = _create(
        roots,
        script="job.sh",
        context_from=["sch_a"],
        continuity=True,
        enabled_toolsets=[],
        skills=["maps", "brief"],
        wrap_response="off",
        workdir=str(workdir),
        attach_to_session=True,
        pinned=True,
        catch_up=False,
    )
    execution = result.spec.execution
    assert execution.script == "job.sh"
    assert execution.context_from == ["sch_a", "self"]
    assert execution.continuity is True
    assert execution.enabled_toolsets == []
    assert execution.skills == ["maps", "brief"]
    assert execution.wrap_response == "off"
    assert execution.workdir == str(workdir)
    assert execution.attach_to_session is True
    assert execution.pinned is True
    assert execution.provider == "provider-x"
    assert execution.model == "model-y"
    assert result.spec.catch_up is False


def test_script_only_mode_requires_and_accepts_a_script(roots) -> None:
    result = _create(roots, prompt="", no_agent=True, script="job.sh")
    assert result.spec.execution.no_agent is True
    assert result.spec.prompt == ""

    _error(roots, "no_agent_requires_script", prompt="x", no_agent=True)
    _error(roots, "script_command_line", prompt="x", script="echo hi")
    _error(roots, "script_not_found", prompt="x", script="missing.sh")
    _error(roots, "script_outside_allowed_root", prompt="x", script="../job.sh")


def test_script_requires_an_allowlisted_root_for_relative_paths(roots) -> None:
    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots, scripts_root=None).compile_create(
            {**BASE, "prompt": "x", "script": "job.sh"}
        )
    assert caught.value.code == "script_not_allowed"


def test_monitor_url_script_alias_and_conflicts(roots) -> None:
    url = _create(roots, monitor="https://example.test/feed")
    assert url.spec.execution.monitor == "https://example.test/feed"

    script = _create(roots, monitor="job.sh")
    assert script.spec.execution.monitor == "job.sh"

    legacy = _create(roots, monitor_script="job.sh")
    assert legacy.spec.execution.monitor == "job.sh"

    _error(
        roots,
        "monitor_conflict",
        monitor_script="job.sh",
        monitor_url="https://example.test",
    )
    _error(
        roots,
        "monitor_incompatible_with_no_agent",
        script="job.sh",
        no_agent=True,
        monitor="https://example.test/feed",
    )
    _error(roots, "invalid_monitor", monitor="https://user:pass@example.test")


def test_context_from_validation_and_cycle_rejection(roots) -> None:
    unknown = _error(roots, "unknown_context_schedule", context_from=["sch_missing"])
    assert unknown.field == "context_from"

    compiler = _compiler(
        roots, reference_schedules={"sch_a": ("sch_b",), "sch_b": ("sch_a",)}
    )
    with pytest.raises(ScheduleCompileError) as caught:
        compiler.compile_create({**BASE, "context_from": ["sch_a"]})
    assert caught.value.code == "context_cycle"


def test_toolsets_workdir_and_wrap_validation(roots) -> None:
    _error(roots, "unknown_toolset", enabled_toolsets=["shell"])
    _error(roots, "invalid_workdir", workdir="relative/path")
    _error(roots, "invalid_workdir", workdir="/does/not/exist")
    _error(roots, "workdir_outside_allowed_roots", workdir="/tmp")
    _error(roots, "invalid_wrap_response", wrap_response="sometimes")


def test_unknown_and_operator_only_fields_are_rejected(roots) -> None:
    _error(roots, "unknown_field", surprise=True)
    for key in ("action", "job_id", "include_disabled", "reason", "session_id"):
        _error(roots, "not_a_schedule_field", **{key: "x"})
    for key, value in (
        ("provider", "openai"),
        ("model", "gpt"),
        ("base_url", "https://example.test"),
        ("grace_minutes", 5),
        ("script_args", ["--x"]),
    ):
        _error(roots, "operator_only_field", **{key: value})


def test_operator_fields_validate_and_grace_is_recurring_only(roots) -> None:
    result = _compiler(roots).compile_create(
        {
            **BASE,
            "provider": "custom",
            "model": "m",
            "base_url": "https://example.test/v1/",
            "grace_minutes": 0,
        },
        operator=True,
    )
    assert result.spec.execution.base_url == "https://example.test/v1"
    assert result.spec.execution.grace_minutes == 0

    # The scheduler honours grace on intervals as well as cron, so the
    # compiler must not reject a field the runtime will actually apply.
    interval = _compiler(roots).compile_create(
        {**BASE, "schedule": "every 2h", "grace_minutes": 5}, operator=True
    )
    assert interval.spec.execution.grace_minutes == 5

    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots).compile_create(
            {**BASE, "schedule": "in 2h", "grace_minutes": 5}, operator=True
        )
    assert caught.value.code == "grace_requires_recurring"

    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots).compile_create(
            {**BASE, "base_url": "https://example.test"}, operator=True
        )
    assert caught.value.code == "base_url_requires_provider"


def test_paused_and_paused_reason(roots) -> None:
    paused = _create(roots, paused=True)
    assert paused.spec.paused is True
    assert paused.next_run_at is None
    assert paused.first_run_at is not None
    assert paused.spec.paused_reason

    _error(roots, "paused_reason_requires_paused", paused_reason="because")


def test_model_one_shot_lead_time_and_operator_grace(roots) -> None:
    _error(
        roots,
        "one_shot_lead_too_short",
        schedule=(NOW + timedelta(seconds=30)).isoformat(),
    )
    operator_now = _compiler(roots).compile_create(
        {**BASE, "schedule": (NOW + timedelta(seconds=1)).isoformat()},
        operator=True,
    )
    assert operator_now.first_run_at is not None

    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots).compile_create(
            {**BASE, "schedule": (NOW - timedelta(seconds=300)).isoformat()},
            operator=True,
        )
    assert caught.value.code == "one_shot_in_past"


def test_merged_edit_preserves_untouched_fields_and_clears_with_empty_values(roots) -> None:
    created = _create(
        roots,
        skills=["maps"],
        context_from=["sch_a"],
        continuity=True,
        enabled_toolsets=[],
        wrap_response="on",
    )
    updated = _compiler(roots).compile_update(
        created.spec,
        {
            "script": "",
            "skills": [],
            "context_from": [],
            "continuity": False,
            "enabled_toolsets": None,
            "wrap_response": "inherit",
            "timezone": "",
            "name": "",
        },
        base_revision=1,
        schedule_id="sch_1",
        current_next_run=created.first_run_at,
    )
    assert updated.spec.execution.skills == []
    assert updated.spec.execution.context_from == []
    assert updated.spec.execution.continuity is False
    assert updated.spec.execution.enabled_toolsets == []
    assert updated.spec.execution.wrap_response is None
    assert updated.spec.execution.script is None
    assert updated.spec.name == created.spec.name
    assert updated.changed is True


def test_merged_edit_cannot_bypass_no_agent_or_paused_rules(roots) -> None:
    created = _create(roots, script="job.sh", no_agent=True)
    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots).compile_update(
            created.spec,
            {"script": ""},
            base_revision=1,
            schedule_id="sch_1",
        )
    assert caught.value.code == "no_agent_requires_script"

    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots).compile_update(
            created.spec,
            {"paused": True},
            base_revision=1,
            schedule_id="sch_1",
        )
    assert caught.value.code == "not_editable"


def test_invalid_timezone_and_repeat_are_stable_rejections(roots) -> None:
    _error(roots, "invalid_timezone", timezone="Mars/Olympus_Mons")
    _error(roots, "invalid_repeat", repeat=0)
    _error(roots, "invalid_repeat", repeat="banana")
    once = _create(roots, schedule="in 2h")
    assert once.spec.repeat_limit == 1


def test_failure_delivery_lane_is_independent(roots) -> None:
    result = _create(roots, deliver="origin", failure_deliver="local")
    assert result.delivery_plan.identities == (("telegram", "1", None),)
    assert result.failure_delivery_plan.local_only is True
    assert result.failure_delivery_plan.lane == "failure"


def test_spec_from_view_preserves_grace_when_config_lacks_it() -> None:
    """A legacy edit must not silently clear the operator's late-run window."""
    view = ScheduleView(
        id="sch-1", name="job", kind="cron", expression="0 9 * * *",
        grace_minutes=7, config={}, created_at=NOW.isoformat(),
    )
    assert spec_from_view(view).execution.grace_minutes == 7

    # An explicit compiled value still wins over the legacy column.
    with_config = view.model_copy(
        update={"config": {"execution": {"grace_minutes": 2}}}
    )
    assert spec_from_view(with_config).execution.grace_minutes == 2


def test_preserved_one_shot_instant_applies_the_past_check(roots) -> None:
    """An edit that keeps the old run_at must not re-arm a stale one-shot."""
    created = _compiler(roots).compile_create(
        {**BASE, "schedule": (NOW + timedelta(hours=1)).isoformat()},
        operator=True,
    )
    stale = (NOW - timedelta(hours=3)).isoformat()
    for operator in (False, True):
        with pytest.raises(ScheduleCompileError) as caught:
            _compiler(roots).compile_update(
                created.spec, {"prompt": "revised"}, operator=operator,
                current_next_run=stale,
            )
        assert caught.value.code == "one_shot_in_past"

    fresh = _compiler(roots).compile_update(
        created.spec, {"prompt": "revised"},
        current_next_run=(NOW + timedelta(hours=1)).isoformat(),
    )
    assert fresh.first_run_at == created.first_run_at


def test_corrupt_stored_timestamps_have_stable_invalid_schedule_code(roots) -> None:
    created = _create(roots)
    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots).compile_update(
            created.spec, {"prompt": "x"}, current_next_run="not-a-date"
        )
    assert caught.value.code == "invalid_schedule"

    one_shot = _create(roots, schedule=(NOW + timedelta(hours=1)).isoformat()).spec
    corrupt = one_shot.model_copy(
        update={
            "expression": one_shot.expression.model_copy(
                update={"run_at": "not-a-date"}
            )
        }
    )
    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(roots).compile_update(corrupt, {"prompt": "x"})
    assert caught.value.code == "invalid_schedule"


def test_unknown_skill_names_are_rejected_when_the_registry_is_available(roots) -> None:
    compiler = _compiler(
        roots, known_skills=["maps"], skill_docs={"maps": "Map documentation."}
    )
    known = compiler.compile_create({**BASE, "skills": ["maps"]})
    assert known.spec.execution.skills == ["maps"]

    with pytest.raises(ScheduleCompileError) as caught:
        compiler.compile_create({**BASE, "skills": ["ghost"]})
    assert caught.value.code == "unknown_skill"
    assert caught.value.field == "skills"

    with pytest.raises(ScheduleCompileError) as caught:
        compiler.compile_update(
            known.spec, {"skills": ["maps", "ghost"]}, base_revision=1
        )
    assert caught.value.code == "unknown_skill"


def test_unknown_skill_is_permissive_without_a_registry(roots) -> None:
    result = _create(roots, skills=["ghost"])
    assert result.spec.execution.skills == ["ghost"]
