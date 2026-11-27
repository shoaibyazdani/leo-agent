"""Final-head schedule grammar, repeat normalization and DST policy."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from leo.schedules.grammar import (
    ParsedSchedule,
    ScheduleGrammarError,
    default_timezone_name,
    next_occurrence,
    normalize_repeat,
    normalize_timezone,
    parse_clock_time,
    parse_cron_expression,
    parse_duration,
    parse_schedule,
    resolve_wall_time,
    wall_time_instants,
)

FIXTURES = Path(__file__).parent / "fixtures" / "schedule_parity"
NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _parse(text: str, *, zone: str = "UTC", now: datetime = NOW) -> ParsedSchedule:
    return parse_schedule(text, now=now, default_timezone=zone)


def test_every_positive_grammar_fixture_case() -> None:
    for case in _fixture("grammar.json")["cases"]:
        if "kind" not in case:
            continue
        if case.get("oracle_gap"):
            # LEO hardening deliberately rejects a zero cadence.
            with pytest.raises(ScheduleGrammarError) as caught:
                _parse(case["input"])
            assert caught.value.code == "non_positive_interval"
            continue
        parsed = _parse(case["input"])
        assert parsed.kind == case["kind"], case["id"]
        assert parsed.display == case["display"], case["id"]
        if case["kind"] == "interval":
            assert parsed.minutes == case["minutes"], case["id"]
            # LEO persists interval cadence in seconds for its scheduler.
            assert parsed.expression == str(case["minutes"] * 60), case["id"]
        if case["kind"] == "cron":
            assert parsed.expression == case["expr"], case["id"]
        if case["kind"] == "once":
            assert parsed.run_at is not None, case["id"]


def test_every_negative_grammar_fixture_case() -> None:
    for case in _fixture("grammar.json")["cases"]:
        if "kind" in case or "error_contains" not in case:
            continue
        with pytest.raises(ScheduleGrammarError) as caught:
            _parse(case["input"])
        assert case["error_contains"] in str(caught.value), case["id"]


def test_duration_negative_fixture_cases() -> None:
    for case in _fixture("grammar.json")["duration_negative_cases"]:
        with pytest.raises(ScheduleGrammarError) as caught:
            parse_duration(case["input"])
        assert case["error_contains"] in str(caught.value)


def test_duration_units_and_optional_count() -> None:
    grammar = _fixture("grammar.json")
    for unit, multiplier in grammar["duration_units"].items():
        assert parse_duration(f"2{unit}") == 2 * multiplier
    assert grammar["duration_count_optional"] is True
    assert parse_duration("hour") == 60
    assert parse_duration("day") == 1440
    with pytest.raises(ScheduleGrammarError):
        parse_duration("1w")
    with pytest.raises(ScheduleGrammarError):
        parse_duration("30s")


@pytest.mark.parametrize(
    "text",
    ["1000000000d", "in 1000000000d", "99999999999999999999m"],
)
def test_absurd_durations_are_stable_invalid_duration(text: str) -> None:
    """A huge model-supplied count must never leak OverflowError."""
    with pytest.raises(ScheduleGrammarError) as caught:
        _parse(text)
    assert caught.value.code == "invalid_duration"

    with pytest.raises(ScheduleGrammarError) as caught:
        parse_duration(text.removeprefix("in ").strip())
    assert caught.value.code == "invalid_duration"


def test_oversized_digit_strings_are_stable_invalid_duration() -> None:
    """A >4300-digit count must not leak CPython's int() ValueError."""
    huge = "9" * 4301
    for text in (f"{huge}m", f"{huge}d", f"{'0' * 4301}{huge}m"):
        with pytest.raises(ScheduleGrammarError) as caught:
            parse_duration(text)
        assert caught.value.code == "invalid_duration"
        assert "too large" in str(caught.value)

    for text in (f"in {huge}m", f"every {huge}m"):
        with pytest.raises(ScheduleGrammarError) as caught:
            _parse(text)
        assert caught.value.code == "invalid_duration"
        assert "too large" in str(caught.value)

    # Bounded leading zeros are still a normal duration.
    assert parse_duration(f"{'0' * 100}7m") == 7


def test_in_duration_keeps_too_large_wording() -> None:
    """An oversized 'in …' count is too large, not an out-of-range instant."""
    huge = "9" * 4301
    for text in ("in 1000000000d", "in 99999999999999999999m", f"in {huge}m"):
        with pytest.raises(ScheduleGrammarError) as caught:
            _parse(text)
        assert isinstance(caught.value, ScheduleGrammarError)
        assert caught.value.code == "invalid_duration"
        assert str(caught.value) == (
            "Invalid duration: the interval is too large. Use at most about 10 years."
        )

    # Ordinary bad durations still get the 'in …'-specific guidance.
    with pytest.raises(ScheduleGrammarError) as caught:
        _parse("in 30s")
    assert caught.value.code == "invalid_duration"
    assert "after 'in '" in str(caught.value)


def test_legacy_every_is_capped_for_readers() -> None:
    """Legacy rows stay readable, but never yield an unbounded cadence."""
    for text in (
        f"@every {'9' * 4301}s",
        "@every 999999999999999999999999w",
        "@every 999999999999999999999999d",
    ):
        with pytest.raises(ScheduleGrammarError) as caught:
            parse_schedule(
                text, now=NOW, default_timezone="UTC", allow_legacy_every=True
            )
        assert caught.value.code == "invalid_schedule"
        assert "too large" in str(caught.value)

    # Writers still reject the legacy form outright.
    with pytest.raises(ScheduleGrammarError) as caught:
        _parse("@every 999999999999999999999999w")
    assert caught.value.code == "invalid_schedule"


def test_interval_next_occurrence_wraps_overflow() -> None:
    """A corrupt/stored huge interval still rejects with a stable code."""
    parsed = ParsedSchedule(
        kind="interval",
        expression="99999999999999999999",
        timezone="UTC",
        display="every absurd",
    )
    with pytest.raises(ScheduleGrammarError) as caught:
        next_occurrence(parsed, NOW)
    assert caught.value.code == "invalid_duration"


def test_clock_forms_match_final_head() -> None:
    assert parse_clock_time("9am") == (9, 0)
    assert parse_clock_time("9:30am") == (9, 30)
    assert parse_clock_time("14:00") == (14, 0)
    assert parse_clock_time("7") == (7, 0)
    assert parse_clock_time("noon") == (12, 0)
    assert parse_clock_time("midnight") == (0, 0)
    assert parse_clock_time("12am") == (0, 0)
    assert parse_clock_time("12pm") == (12, 0)
    assert parse_clock_time("25am") is None
    assert parse_clock_time("9:99") is None


def test_bare_duration_recurs_and_in_duration_is_one_shot() -> None:
    recurring = _parse("30m")
    once = _parse("in 30m")
    assert recurring.kind == "interval"
    assert once.kind == "once"
    assert once.run_at == NOW + timedelta(minutes=30)
    assert once.display == "once in 30m"


def test_iso_one_shots_naive_and_aware() -> None:
    naive = _parse("2030-01-15T14:00:00", zone="America/New_York")
    assert naive.run_at == datetime(2030, 1, 15, 19, 0, tzinfo=timezone.utc)
    aware = _parse("2030-01-15T14:00:00+02:00", zone="America/New_York")
    assert aware.run_at == datetime(2030, 1, 15, 12, 0, tzinfo=timezone.utc)
    date_only = _parse("2030-01-15")
    assert date_only.run_at == datetime(2030, 1, 15, 0, 0, tzinfo=timezone.utc)


def test_six_field_cron_and_named_fields() -> None:
    parsed = _parse("0 0 9 * * *")
    assert parsed.kind == "cron"
    assert parsed.cron is not None and parsed.cron.has_seconds is True
    assert next_occurrence(parsed, NOW) == datetime(2026, 11, 27, 9, 0, tzinfo=timezone.utc)

    named = _parse("0 9 * * MON,WED,FRI")
    assert next_occurrence(named, NOW).weekday() in (0, 2, 4)

    with pytest.raises(ScheduleGrammarError) as caught:
        _parse("0 0 9 * * * *")
    assert caught.value.code == "invalid_cron_expression"


def test_cron_field_validation_is_stable() -> None:
    with pytest.raises(ScheduleGrammarError) as caught:
        parse_cron_expression("0 9 * * FUNDAY")
    assert caught.value.code == "invalid_cron_expression"
    with pytest.raises(ScheduleGrammarError):
        parse_cron_expression("60 * * * *")
    with pytest.raises(ScheduleGrammarError):
        parse_cron_expression("*/0 * * * *")
    assert parse_cron_expression("0 9 * * 7").matches(
        datetime(2026, 11, 29, 9, 0)  # Sunday
    )


def test_dst_gap_advances_to_first_valid_minute() -> None:
    new_york = ZoneInfo("America/New_York")
    resolved = resolve_wall_time(datetime(2024, 3, 10, 2, 30), new_york)
    assert resolved.isoformat() == "2024-03-10T03:00:00-04:00"

    parsed = _parse("30 2 * * *", zone="America/New_York")
    first = next_occurrence(
        parsed, datetime(2024, 3, 10, 0, 0, tzinfo=timezone.utc)
    )
    assert first.isoformat() == "2024-03-10T03:00:00-04:00"


def test_dst_fold_uses_first_occurrence_for_new_schedules_but_next_run_can_use_second() -> None:
    new_york = ZoneInfo("America/New_York")
    instants = wall_time_instants(datetime(2024, 11, 3, 1, 30), new_york)
    assert [item.astimezone(timezone.utc) for item in instants] == [
        datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc),
        datetime(2024, 11, 3, 6, 30, tzinfo=timezone.utc),
    ]
    assert resolve_wall_time(datetime(2024, 11, 3, 1, 30), new_york).astimezone(
        timezone.utc
    ) == datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc)

    parsed = _parse("30 1 * * *", zone="America/New_York")
    first = next_occurrence(parsed, datetime(2024, 11, 3, 5, 0, tzinfo=timezone.utc))
    assert first.astimezone(timezone.utc) == datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc)
    second = next_occurrence(parsed, datetime(2024, 11, 3, 5, 31, tzinfo=timezone.utc))
    assert second.astimezone(timezone.utc) == datetime(2024, 11, 3, 6, 30, tzinfo=timezone.utc)


def test_timezone_normalization_and_defaults() -> None:
    assert normalize_timezone("America/New_York") == "America/New_York"
    assert normalize_timezone(None) is None
    assert normalize_timezone("") is None
    with pytest.raises(ScheduleGrammarError) as caught:
        normalize_timezone("Mars/Olympus_Mons")
    assert caught.value.code == "invalid_timezone"
    assert default_timezone_name()

    explicit = _parse("every day at 9am", zone="UTC")
    assert explicit.timezone == "UTC"
    pinned = parse_schedule(
        "every day at 9am", now=NOW, default_timezone="UTC", timezone="Asia/Kolkata"
    )
    assert pinned.timezone == "Asia/Kolkata"


def test_repeat_normalization_and_leo_hardening() -> None:
    cases = _fixture("repeat.json")["normalization"]
    expected_errors = {case["input"] for case in cases if "error" in case}
    for case in cases:
        value = case["input"]
        if value in expected_errors or value in (0, -2, True, False, 3.9):
            with pytest.raises(ScheduleGrammarError) as caught:
                normalize_repeat(value, kind="cron")
            assert caught.value.code == "invalid_repeat"
            continue
        assert normalize_repeat(value, kind="cron") == case["stored_times"]

    assert normalize_repeat(None, kind="once") == 1
    assert normalize_repeat("forever", kind="once") == 1
    with pytest.raises(ScheduleGrammarError):
        normalize_repeat(2, kind="once")


def test_legacy_every_is_readable_only() -> None:
    with pytest.raises(ScheduleGrammarError) as caught:
        _parse("@every 1h")
    assert caught.value.code == "invalid_schedule"

    legacy = parse_schedule(
        "@every 90s", now=NOW, default_timezone="UTC", allow_legacy_every=True
    )
    assert legacy.kind == "interval"
    assert legacy.legacy_every is True
    assert next_occurrence(legacy, NOW) == NOW + timedelta(seconds=90)


def test_stable_rejection_codes() -> None:
    with pytest.raises(ScheduleGrammarError) as caught:
        _parse("funday at 9am")
    assert caught.value.code == "invalid_schedule"
    with pytest.raises(ScheduleGrammarError) as caught:
        _parse("every monday")
    assert caught.value.code == "invalid_duration"
    with pytest.raises(ScheduleGrammarError) as caught:
        _parse("2030-13-99T99:99:99")
    assert caught.value.code in ("invalid_timestamp", "invalid_schedule")
