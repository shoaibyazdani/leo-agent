"""Phase 6: attach_to_session durable local conversation notes."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.core.app import LeoApp
from leo.schedules.clock import FrozenClock
from leo.schedules.execution import (
    ScheduleExecution,
    latest_session_note,
    record_session_note,
)
from leo.schedules.store import ScheduleStore

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


async def test_record_session_note_creates_durable_conversation() -> None:
    await db.initialize()
    conversation_id = await record_session_note(
        schedule_id="sch_1",
        name="Daily brief",
        task_id="task_1",
        status="completed",
        text="A concise outcome.",
        clock=FrozenClock(NOW),
    )
    assert conversation_id == "schedule-sch_1"
    conversation = await db.fetchone(
        "SELECT * FROM conversations WHERE id=?", (conversation_id,)
    )
    assert conversation is not None
    assert conversation["channel"] == "schedule"

    row = await db.fetchone(
        "SELECT role,content,metadata FROM messages WHERE conversation_id=? "
        "ORDER BY id DESC LIMIT 1",
        (conversation_id,),
    )
    assert row["role"] == "assistant"
    assert "A concise outcome." in row["content"]
    import json

    metadata = json.loads(row["metadata"])
    assert metadata["schedule_note"] is True
    assert metadata["schedule_id"] == "sch_1"
    assert metadata["task_id"] == "task_1"
    assert metadata["tool_calls"] == []

    brief = await latest_session_note("sch_1", limit_chars=200)
    assert "A concise outcome." in brief


async def test_next_reply_in_the_job_conversation_sees_the_note() -> None:
    await db.initialize()
    from leo.models import compaction

    await compaction.init_schema()
    await record_session_note(
        schedule_id="sch_2",
        name="Weekly digest",
        task_id="task_2",
        status="completed",
        text="DIGEST-OUTPUT",
        clock=FrozenClock(NOW),
    )
    app = object.__new__(LeoApp)
    app.round_budget = 10
    history = await LeoApp._conversation_history(app, "schedule-sch_2")
    assert any(
        message.role == "assistant" and "DIGEST-OUTPUT" in message.content
        for message in history
    )


async def test_attach_to_session_flag_controls_the_note() -> None:
    await db.initialize()
    clock = FrozenClock(NOW)
    execution = ScheduleExecution(
        settings=Settings(),
        store=ScheduleStore(clock=clock),
        clock=clock,
    )
    task = SimpleNamespace(
        id="task_3",
        objective="job",
        metadata={"schedule_id": "sch_3", "schedule": "job"},
    )
    await execution.record_outcome(
        task=task,
        context={
            "schedule_payload": {
                "name": "job",
                "execution": {"attach_to_session": False},
            }
        },
        status="completed",
        text="should not appear",
    )
    assert (
        await db.fetchone(
            "SELECT id FROM conversations WHERE id='schedule-sch_3'"
        )
        is None
    )

    await execution.record_outcome(
        task=task,
        context={
            "schedule_payload": {
                "name": "job",
                "execution": {"attach_to_session": True},
            }
        },
        status="failed",
        text="FAILED-OUTPUT",
    )
    brief = await latest_session_note("sch_3")
    assert "FAILED-OUTPUT" in brief
    assert "FAILED" in brief
