"""Canonical schedule digest binding and update fingerprints."""

from __future__ import annotations

from datetime import datetime, timezone

from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompiler
from leo.schedules.delivery import DeliveryContext, HomeChannel, Origin

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def _compiler(tmp_path) -> ScheduleCompiler:
    context = DeliveryContext(
        origin=Origin("telegram", "123", "5"),
        home_channels=(HomeChannel("telegram", "999"),),
        connected_platforms=frozenset({"telegram"}),
    )
    return ScheduleCompiler(
        FrozenClock(NOW),
        default_timezone="UTC",
        scripts_root=tmp_path,
        delivery_context=context,
    )


def _base(tmp_path, **overrides):
    (tmp_path / "noop.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    args = {
        "schedule": "every day at 9am",
        "prompt": "Send the morning brief.",
        "deliver": "origin",
        "repeat": 3,
    }
    args.update(overrides)
    return _compiler(tmp_path).compile_create(args)


def test_argument_insertion_order_does_not_change_the_digest(tmp_path) -> None:
    (tmp_path / "noop.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    compiler = _compiler(tmp_path)
    left = compiler.compile_create(
        {
            "schedule": "every monday 9am",
            "prompt": "Brief",
            "name": "Monday",
            "deliver": "origin",
            "repeat": 2,
        }
    )
    right = compiler.compile_create(
        {
            "repeat": 2,
            "deliver": "origin",
            "name": "Monday",
            "prompt": "Brief",
            "schedule": "every monday 9am",
        }
    )
    assert left.spec.canonical_json() == right.spec.canonical_json()
    assert left.spec.digest() == right.spec.digest()
    assert left.config_digest == right.config_digest
    assert left.proposal_digest == right.proposal_digest


def test_prompt_text_changes_the_digest(tmp_path) -> None:
    base = _base(tmp_path, prompt="Brief A")
    changed = _base(tmp_path, prompt="Brief B")
    assert base.spec.digest() != changed.spec.digest()
    assert base.config_digest != changed.config_digest


def test_delivery_and_execution_change_the_digest(tmp_path) -> None:
    base = _base(tmp_path)
    local = _base(tmp_path, deliver="local")
    toolsets = _base(tmp_path, enabled_toolsets=["web"])
    skills = _base(tmp_path, skills=["maps"])
    script = _base(
        tmp_path,
        no_agent=True,
        script="noop.sh",
        prompt="",
    )
    digests = {
        base.spec.digest(),
        local.spec.digest(),
        toolsets.spec.digest(),
        skills.spec.digest(),
        script.spec.digest(),
    }
    assert len(digests) == 5


def test_schedule_timezone_and_repeat_change_the_digest(tmp_path) -> None:
    base = _base(tmp_path)
    other_time = _base(tmp_path, schedule="every day at 10am")
    other_zone = _base(tmp_path, timezone="Asia/Kolkata")
    other_repeat = _base(tmp_path, repeat=4)
    assert len(
        {
            base.spec.digest(),
            other_time.spec.digest(),
            other_zone.spec.digest(),
            other_repeat.spec.digest(),
        }
    ) == 4


def test_semantically_equal_interval_spellings_share_a_digest(tmp_path) -> None:
    daily = _base(tmp_path, schedule="every 24h")
    day = _base(tmp_path, schedule="every 1d")
    assert daily.spec.digest() == day.spec.digest()
    assert daily.config_digest == day.config_digest
    # The canonical display is minute-based for both spellings.
    assert daily.display == day.display == "every 1440m"


def test_noop_update_keeps_spec_digest_and_reports_unchanged(tmp_path) -> None:
    created = _base(tmp_path)
    compiler = _compiler(tmp_path)
    updated = compiler.compile_update(
        created.spec,
        {},
        base_revision=1,
        schedule_id="sch_1",
        current_next_run=created.first_run_at,
    )
    assert updated.spec.digest() == created.spec.digest()
    assert updated.changed is False
    assert updated.reanchors is False
    assert updated.config_revision == 2


def test_edit_fingerprint_changes_with_revision_and_schedule_id(tmp_path) -> None:
    created = _base(tmp_path)
    compiler = _compiler(tmp_path)
    first = compiler.compile_update(
        created.spec,
        {"prompt": "Changed"},
        base_revision=1,
        schedule_id="sch_1",
        current_next_run=created.first_run_at,
    )
    second = compiler.compile_update(
        created.spec,
        {"prompt": "Changed"},
        base_revision=2,
        schedule_id="sch_1",
        current_next_run=created.first_run_at,
    )
    other_id = compiler.compile_update(
        created.spec,
        {"prompt": "Changed"},
        base_revision=1,
        schedule_id="sch_2",
        current_next_run=created.first_run_at,
    )
    assert first.edit_fingerprint != second.edit_fingerprint
    assert first.edit_fingerprint != other_id.edit_fingerprint
    assert first.proposal_digest == first.edit_fingerprint


def test_update_schedule_change_changes_semantics(tmp_path) -> None:
    created = _base(tmp_path)
    updated = _compiler(tmp_path).compile_update(
        created.spec,
        {"schedule": "every day at 10am"},
        base_revision=1,
        schedule_id="sch_1",
    )
    assert updated.changed is True
    assert updated.reanchors is True
    assert updated.spec.digest() != created.spec.digest()
