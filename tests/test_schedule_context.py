"""Phase 6: context_from, continuity, missing sources and bounded injection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.schedules.clock import FrozenClock
from leo.schedules.compiler import ScheduleCompileError, ScheduleCompiler
from leo.schedules.delivery import DeliveryContext, Origin
from leo.schedules.execution import ScheduleExecution
from leo.schedules.store import ScheduleStore
from leo.tasks import DurableTaskEngine, Scheduler
from leo.events import EventBus

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    clock = FrozenClock(NOW)
    store = ScheduleStore(clock=clock)
    engine = DurableTaskEngine(settings, events)
    scheduler = Scheduler(engine, settings, events, clock=clock)
    execution = ScheduleExecution(
        settings=settings,
        store=store,
        clock=clock,
        allowed_workdir_roots=[tmp_path],
    )
    tmp_path.mkdir(exist_ok=True)
    return SimpleNamespace(
        settings=settings,
        events=events,
        clock=clock,
        store=store,
        engine=engine,
        scheduler=scheduler,
        execution=execution,
        tmp_path=tmp_path,
    )


async def _schedule(env, schedule_id: str, name: str) -> None:
    await db.execute(
        "INSERT INTO schedules(id,name,kind,expression,created_at) VALUES(?,?,?,?,?)",
        (schedule_id, name, "interval", "60", NOW.isoformat()),
    )


async def _completed(env, schedule_id: str, objective: str, output: str) -> str:
    task = await env.engine.create(
        objective, metadata={"schedule_id": schedule_id, "schedule": "source"}
    )
    await db.execute(
        "UPDATE tasks SET status='completed', result=?, completed_at=? WHERE id=?",
        (json.dumps({"output": output}), NOW.isoformat(), task.id),
    )
    return task.id


async def _prepare(env, payload, *, task_id: str = "t_current") -> object:
    task = SimpleNamespace(
        id=task_id,
        objective=str(payload.get("objective") or "job"),
        metadata={"schedule_id": payload.get("__schedule_id_hint", "sch_job")},
    )
    return await env.execution.prepare_agent(
        task=task,
        context={"schedule_payload": payload, "objective": payload.get("objective") or "job"},
    )


async def test_context_from_injects_latest_completed_only(env) -> None:
    await _schedule(env, "sch_a", "news")
    await _completed(env, "sch_a", "first run", "OLD-OUTPUT")
    await _completed(env, "sch_a", "second run", "LATEST-OUTPUT")
    prepared = await _prepare(
        env,
        {
            "name": "job",
            "execution": {"context_from": ["sch_a"]},
        },
    )
    assert "context_from: news" in prepared.objective
    assert "LATEST-OUTPUT" in prepared.objective
    assert "OLD-OUTPUT" not in prepared.objective


async def test_missing_source_is_reported_honestly(env) -> None:
    prepared = await _prepare(
        env,
        {"name": "job", "execution": {"context_from": ["sch_missing"]}},
    )
    assert "missing or was removed" in prepared.objective


async def test_continuity_first_run_then_previous_output(env) -> None:
    await _schedule(env, "sch_job", "job")
    first = await _prepare(
        env,
        {
            "__schedule_id_hint": "sch_job",
            "name": "job",
            "execution": {"continuity": True, "context_from": ["self"]},
        },
    )
    assert "first run" in first.objective

    await _completed(env, "sch_job", "previous run", "PREVIOUS-OUTPUT")
    later = await _prepare(
        env,
        {
            "__schedule_id_hint": "sch_job",
            "name": "job",
            "execution": {"continuity": True, "context_from": ["self"]},
        },
    )
    assert "PREVIOUS-OUTPUT" in later.objective
    assert "first run" not in later.objective


async def test_injection_is_bounded(env) -> None:
    await _schedule(env, "sch_a", "huge")
    await _completed(env, "sch_a", "huge", "Z" * 50_000)
    prepared = await _prepare(
        env,
        {"name": "job", "execution": {"context_from": ["sch_a"]}},
    )
    assert "[truncated]" in prepared.objective
    assert len(prepared.objective) < 20_000


def test_context_cycles_are_rejected_at_create_and_update(tmp_path) -> None:
    (tmp_path / "job.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    compiler = ScheduleCompiler(
        FrozenClock(NOW),
        default_timezone="UTC",
        scripts_root=tmp_path,
        allowed_workdir_roots=[tmp_path],
        known_schedule_ids={"sch_a", "sch_b"},
        reference_schedules={"sch_a": ("sch_b",), "sch_b": ("sch_a",)},
        delivery_context=DeliveryContext(origin=Origin("telegram", "1")),
    )
    base = {"schedule": "every day at 9am", "prompt": "Brief", "deliver": "local"}
    with pytest.raises(ScheduleCompileError) as caught:
        compiler.compile_create({**base, "context_from": ["sch_a"]})
    assert caught.value.code == "context_cycle"

    clean = compiler.compile_create({**base, "context_from": []})
    with pytest.raises(ScheduleCompileError) as caught:
        compiler.compile_update(
            clean.spec,
            {"context_from": ["sch_a"]},
            base_revision=1,
            schedule_id="sch_a",
        )
    assert caught.value.code == "context_cycle"
