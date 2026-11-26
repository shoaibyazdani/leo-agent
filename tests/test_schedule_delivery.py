"""Phase 8: one resolver for preview and runtime, with honest local notes."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.core.app import LeoApp
from leo.schedules.compiler import ScheduleCompileError, ScheduleCompiler
from leo.schedules.clock import FrozenClock
from leo.schedules.delivery import (
    DeliveryContext,
    DeliveryResolutionError,
    Origin,
    delivery_context_from_settings,
    resolve_delivery,
    task_delivery_plan,
)
from leo.schedules.proposals import schedule_payload

NOW = FrozenClock(datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))


def _settings(**overrides) -> Settings:
    settings = Settings()
    settings.automation.home_channels = ["telegram:900", "slack:C9"]
    settings.automation.bot_profiles = ["work"]
    settings.automation.local_conversations = ["conv1"]
    settings.automation.default_room_channel = "r1"
    for key, value in overrides.items():
        setattr(settings.automation, key, value)
    return settings


def _compiler(settings: Settings, context: DeliveryContext) -> ScheduleCompiler:
    return ScheduleCompiler(
        NOW,
        default_timezone="UTC",
        scripts_root=None,
        known_schedule_ids=set(),
        delivery_context=context,
    )


def _resolve(settings, turn=None, adapters=None):
    origin = (
        Origin("telegram", str(turn.conversation_id), turn.thread_id)
        if turn is not None else None
    )
    return delivery_context_from_settings(settings, origin=origin, adapters=adapters)


def test_preview_and_runtime_resolve_identical_targets() -> None:
    settings = _settings()
    context = _resolve(settings)
    result = _compiler(settings, context).compile_create({
        "schedule": "every day at 9am",
        "prompt": "Brief me",
        "deliver": "telegram:555:7,local:conv1",
        "failure_deliver": "bot-chat:work",
    })
    assert result.delivery_plan.identities == (("telegram", "555", "7"), ("local", "conv1", None))
    assert result.failure_delivery_plan.identities == (("bot-chat", "work", None),)
    payload = schedule_payload(result.spec)
    runtime, _ = task_delivery_plan(payload, lane="result")
    failure, explicit = task_delivery_plan(payload, lane="failure")
    assert runtime.identities == result.delivery_plan.identities
    assert failure.identities == result.failure_delivery_plan.identities
    assert explicit is True
    assert runtime.keyed_targets["telegram:555:7"].chat_id == "555"


def test_home_all_and_named_local_forms_use_configured_context() -> None:
    settings = _settings()
    context = _resolve(settings)
    home = _compiler(settings, context).compile_create({
        "schedule": "every day at 9am", "prompt": "x", "deliver": "home",
    })
    all_targets = _compiler(settings, context).compile_create({
        "schedule": "every day at 9am", "prompt": "x", "deliver": "all",
    })
    assert home.delivery_plan.identities == (("telegram", "900", None),)
    assert all_targets.delivery_plan.identities == (
        ("telegram", "900", None), ("slack", "C9", None)
    )
    local = _compiler(settings, context).compile_create({
        "schedule": "every day at 9am", "prompt": "x", "deliver": ["local:conv1"],
    })
    assert local.delivery_plan.targets[0].kind == "local"


def test_origin_uses_trusted_turn_identity_not_a_body_field() -> None:
    settings = _settings()
    turn = SimpleNamespace(conversation_id="c-1", thread_id="44", channel="telegram")
    context = _resolve(settings, turn=turn)
    result = _compiler(settings, context).compile_create({
        "schedule": "every day at 9am", "prompt": "x", "deliver": "origin",
    })
    assert result.delivery_plan.identities == (("telegram", "c-1", "44"),)


def test_room_needs_a_room_capable_adapter_and_default_channel() -> None:
    settings = _settings()
    no_adapter = _resolve(settings, adapters={"telegram": object()})
    with pytest.raises(ScheduleCompileError) as caught:
        _compiler(settings, no_adapter).compile_create({
            "schedule": "every day at 9am", "prompt": "x", "deliver": "room:r1",
        })
    assert caught.value.code == "room_unavailable"

    capable = _resolve(settings, adapters={"matrix": SimpleNamespace(supports_rooms=True)})
    result = _compiler(settings, capable).compile_create({
        "schedule": "every day at 9am", "prompt": "x", "deliver": "room:r1",
    })
    assert result.delivery_plan.identities == (("room", "r1", None),)


def test_nowhere_is_honest_and_never_a_literal_none_target() -> None:
    # The token resolver refuses a placeholder address rather than sending to it.
    with pytest.raises(DeliveryResolutionError) as caught:
        plan = resolve_delivery(
            "telegram:None",
            context=DeliveryContext(connected_platforms=frozenset({"telegram"})),
        )
    assert caught.value.code == "invalid_target"
    nowhere = task_delivery_plan({"delivery": []}, lane="result")[0]
    assert nowhere.nowhere is True
    assert nowhere.describe() == "no destination"


async def test_local_note_delivery_is_visible_to_the_next_turn() -> None:
    await db.initialize()
    from leo.models import compaction

    await compaction.init_schema()
    app = object.__new__(LeoApp)

    async def _terminal(*a, **k):
        return None

    app.task_reporting = SimpleNamespace(on_terminal=_terminal)
    task = SimpleNamespace(
        id="task-delivery-1",
        objective="Morning brief",
        metadata={"schedule_id": "sch_delivery", "schedule": "Morning brief"},
        status="completed",
    )
    from leo.schedules.domain import DeliveryTarget

    outcome = await app._channel_delivery(
        task, "local", "conv1",
        {
            "status": "completed",
            "delivery_lane": "result",
            "delivery_target_key": "local:conv1:",
            "delivery_target": DeliveryTarget(kind="local", platform="local", chat_id="conv1").model_dump(mode="json"),
            "schedule_payload": {"name": "Morning brief", "text": "LOCAL-NOTE-BODY"},
        },
    )
    assert outcome == "local:note:conv1"
    row = await db.fetchone(
        "SELECT role,content,metadata FROM messages WHERE conversation_id='conv1' "
        "ORDER BY id DESC LIMIT 1"
    )
    assert row["role"] == "assistant"
    assert "LOCAL-NOTE-BODY" in row["content"]
    import json

    metadata = json.loads(row["metadata"])
    assert metadata["schedule_id"] == "sch_delivery"
    assert metadata["task_id"] == "task-delivery-1"
    assert metadata["delivery_lane"] == "result"
    app.round_budget = 10
    history = await LeoApp._conversation_history(app, "conv1")
    assert any("LOCAL-NOTE-BODY" in item.content for item in history)


def test_object_form_all_expands_every_configured_home() -> None:
    settings = _settings()
    context = _resolve(settings)
    plan = resolve_delivery({"token": "all", "label": "all operators"}, context=context)
    assert plan.identities == (("telegram", "900", None), ("slack", "C9", None))
    assert all(route.audience == "all operators" for route in plan.routes)
