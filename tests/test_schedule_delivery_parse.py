"""Delivery token parsing/resolution shared by preview and runtime."""

from __future__ import annotations

import pytest

from leo.schedules.delivery import (
    DeliveryContext,
    DeliveryPlan,
    DeliveryResolutionError,
    DeliveryResolver,
    HomeChannel,
    Origin,
    RoomInfo,
)


def _context(**overrides) -> DeliveryContext:
    values = {
        "origin": Origin("telegram", "123", "5"),
        "home_channels": (
            HomeChannel("telegram", "999"),
            HomeChannel("slack", "C1", "T1"),
        ),
        "connected_platforms": frozenset({"telegram", "slack"}),
        "bot_profiles": frozenset({"work", "personal"}),
        "rooms": {"r1": RoomInfo("r1", "work")},
        "room_capable": True,
        "default_room_channel": "r1",
        "local_conversations": frozenset({"conv1"}),
    }
    values.update(overrides)
    return DeliveryContext(**values)


def _resolve(value, *, context=None, lane="result"):
    return DeliveryResolver(context or _context()).resolve(value, lane=lane)


def test_origin_resolves_to_this_chat_with_full_identity() -> None:
    plan = _resolve("origin")
    assert plan.identities == (("telegram", "123", "5"),)
    assert plan.describe() == "this chat"
    assert plan.is_origin_only() is True
    assert plan.nowhere is False


def test_origin_falls_back_to_first_home_then_honest_nowhere() -> None:
    with_home = _resolve("origin", context=_context(origin=None))
    assert with_home.identities == (("telegram", "999", None),)
    assert with_home.describe() == "home channel for telegram"
    assert with_home.routes[0].resolved_from == "origin_fallback"

    no_home = _resolve("origin", context=DeliveryContext(connected_platforms=frozenset()))
    assert no_home.nowhere is True
    assert no_home.describe() == "no destination"
    assert no_home.unresolved[0].code == "nowhere"


def test_omitted_delivery_defaults_to_origin_on_create_and_local_at_runtime() -> None:
    create = _resolve(None)
    assert create.identities == (("telegram", "123", "5"),)
    runtime = DeliveryResolver(_context()).resolve(None, mode="runtime")
    assert runtime.identities == (("local", None, None),)
    assert runtime.local_only is True


def test_all_expands_configured_home_channels_and_excludes_bot_chat() -> None:
    plan = _resolve("all")
    assert plan.identities == (("telegram", "999", None), ("slack", "C1", "T1"))
    assert all(route.resolved_from == "home" for route in plan.routes)
    assert plan.describe() == "home channel for telegram, home channel for slack"


def test_local_and_local_conversation_are_history_only() -> None:
    local = _resolve("local")
    assert local.local_only is True
    assert local.identities == (("local", None, None),)
    assert local.describe() == "local history only"

    conversation = _resolve("local:conv1")
    assert conversation.local_only is True
    assert conversation.identities == (("local", "conv1", None),)
    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("local:missing")
    assert caught.value.code == "unknown_local_conversation"


def test_bot_chat_own_and_named_profiles() -> None:
    own = _resolve("bot-chat")
    assert own.routes[0].target.kind == "bot_chat"
    assert own.identities == (("bot-chat", "", None),)

    named = _resolve("bot-chat:work")
    assert named.identities == (("bot-chat", "work", None),)
    assert named.describe() == "local agent chat (work)"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("bot-chat:missing")
    assert caught.value.code == "unknown_bot_profile"


def test_room_requires_capability_and_a_resolved_room() -> None:
    room = _resolve("room:r1")
    assert room.routes[0].target.kind == "room"
    assert room.identities == (("room", "r1", None),)
    assert room.describe() == "room r1"

    named = _resolve("room:r1@work")
    assert named.routes[0].target.profile == "work"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("room:r1", context=_context(room_capable=False))
    assert caught.value.code == "room_unavailable"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("room:missing")
    assert caught.value.code == "unresolved_room"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("room:r1@missing")
    assert caught.value.code == "unknown_bot_profile"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("room:bad room")
    assert caught.value.code == "unresolved_room"


def test_explicit_platform_chat_thread_and_origin_thread_adoption() -> None:
    explicit = _resolve("telegram:555:77")
    assert explicit.identities == (("telegram", "555", "77"),)
    assert explicit.describe() == "telegram chat 555 thread 77"

    adopted = _resolve("telegram:123")
    assert adopted.identities == (("telegram", "123", "5"),)

    no_adoption = _resolve("slack:123")
    assert no_adoption.identities == (("slack", "123", None),)


def test_matrix_room_ids_keep_their_colons_in_explicit_and_token_targets() -> None:
    context = _context(
        connected_platforms=frozenset({"telegram", "slack", "matrix"}),
        home_channels=(
            HomeChannel("matrix", "!ops:example.org"),
            HomeChannel("telegram", "999"),
        ),
    )
    structured = _resolve(
        {"platform": "matrix", "chat_id": "!ops:example.org"}, context=context
    )
    assert structured.identities == (("matrix", "!ops:example.org", None),)
    assert structured.routes[0].resolved_from == "explicit"

    token = _resolve("matrix:!ops:example.org", context=context)
    assert token.identities == (("matrix", "!ops:example.org", None),)


def test_bare_platform_home_and_origin_fallback() -> None:
    telegram = _resolve("telegram")
    assert telegram.identities == (("telegram", "999", None),)
    assert telegram.describe() == "home channel for telegram"

    origin_fallback = _resolve(
        "telegram",
        context=_context(home_channels=(HomeChannel("slack", "C1"),)),
    )
    assert origin_fallback.identities == (("telegram", "123", "5"),)
    assert origin_fallback.describe() == "this chat"

    nowhere = _resolve(
        "slack",
        context=_context(
            origin=None,
            home_channels=(HomeChannel("telegram", "999"),),
        ),
    )
    assert nowhere.nowhere is True


def test_home_alias_is_native_and_deterministic() -> None:
    plan = _resolve("home")
    assert plan.identities == (("telegram", "999", None),)
    assert plan.routes[0].resolved_from == "home"


def test_comma_list_list_input_dedupe_and_order() -> None:
    combined = _resolve("telegram:123,local,telegram:123:5,local")
    assert combined.identities == (("telegram", "123", "5"), ("local", None, None))
    assert combined.local_only is True

    listed = _resolve(["telegram:123", "slack"])
    assert listed.identities == (
        ("telegram", "123", "5"),
        ("slack", "C1", "T1"),
    )

    deduped = _resolve("telegram:123:5,telegram:123:5")
    assert deduped.identities == (("telegram", "123", "5"),)


def test_object_target_forms_and_unknown_keys() -> None:
    explicit = _resolve({"platform": "telegram", "chat": "555", "thread": "77"})
    assert explicit.identities == (("telegram", "555", "77"),)

    local = _resolve({"kind": "local", "conversation_id": "conv1"})
    assert local.identities == (("local", "conv1", None),)

    bot = _resolve({"kind": "bot_chat", "profile": "work"})
    assert bot.identities == (("bot-chat", "work", None),)

    room = _resolve({"kind": "room", "room_id": "r1"})
    assert room.identities == (("room", "r1", None),)

    labelled = _resolve({"token": "telegram:555", "label": "the team channel"})
    assert labelled.routes[0].audience == "the team channel"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve({"platform": "telegram", "chat": "1", "surprise": True})
    assert caught.value.code == "invalid_target"


def test_unknown_tokens_and_unverified_adapters_are_rejected() -> None:
    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("not-a-real-target")
    assert caught.value.code == "unknown_token"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("matrix:room")
    assert caught.value.code == "adapter_unavailable"

    with pytest.raises(DeliveryResolutionError) as caught:
        _resolve("telegram:")
    assert caught.value.code == "invalid_target"


def test_failure_lane_is_separate_and_local_suppresses_external_notices() -> None:
    resolver = DeliveryResolver(_context())
    result = resolver.resolve_result("telegram:123")
    failure = resolver.resolve_failure("local")
    assert result.lane == "result"
    assert failure.lane == "failure"
    assert failure.local_only is True
    assert failure.identities == (("local", None, None),)
    assert result.identities != failure.identities


def test_configured_plan_preserves_persisted_targets_without_guessing() -> None:
    from leo.schedules.domain import DeliveryTarget

    plan = DeliveryPlan.configured(
        [
            DeliveryTarget(kind="external", platform="telegram", chat_id="42", thread_id="7"),
            DeliveryTarget(kind="local", platform="local"),
        ]
    )
    assert plan.identities == (("telegram", "42", "7"), ("local", None, None))
    assert plan.local_only is True
    assert plan.describe().startswith("local history only")
