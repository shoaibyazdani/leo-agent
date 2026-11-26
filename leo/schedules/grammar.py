"""Final-head schedule grammar, repeat normalization and timezone/DST helpers.

This module is pure: it parses text into a canonical
:class:`~leo.schedules.domain.ScheduleExpression`, normalizes repeat values and
computes wall-clock occurrences in an IANA zone. It never imports the database
or the scheduler and never writes anything.

Canonical interval expressions are stored in *seconds* because that is the unit
LEO's existing durable ``schedules.expression`` column and
:class:`leo.tasks.scheduler.Scheduler` consume. The final-head oracle reasons in
minutes; :attr:`ParsedSchedule.minutes` exposes that value and the display text
is minute-based.

DST policy (LEO-native, tested):

* a nonexistent local wall time (spring-forward gap) advances to the first
  valid minute after the gap;
* an ambiguous local wall time (fall-back fold) resolves to the first
  occurrence for a new schedule; cron *next-run* computation tries both folds
  and keeps the earliest instant strictly after the base, so the second
  occurrence is used only when the base is already inside it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc

ErrorCode = Literal[
    "invalid_schedule",
    "invalid_duration",
    "invalid_cron_expression",
    "invalid_timestamp",
    "invalid_timezone",
    "invalid_repeat",
    "invalid_clock_time",
    "non_positive_interval",
    "no_schedule_match",
]

_DURATION_UNITS: dict[str, int] = {
    "m": 1,
    "min": 1,
    "mins": 1,
    "minute": 1,
    "minutes": 1,
    "h": 60,
    "hr": 60,
    "hrs": 60,
    "hour": 60,
    "hours": 60,
    "d": 1440,
    "day": 1440,
    "days": 1440,
}
_DURATION_RE = re.compile(
    r"^(?P<count>\d*)\s*(?P<unit>"
    r"m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|"
    r"d|day|days)$",
    re.IGNORECASE,
)

_WEEKDAY_ALIASES: dict[str, str] = {
    "sunday": "0",
    "sun": "0",
    "monday": "1",
    "mon": "1",
    "tuesday": "2",
    "tue": "2",
    "tues": "2",
    "wednesday": "3",
    "wed": "3",
    "weds": "3",
    "thursday": "4",
    "thu": "4",
    "thur": "4",
    "thurs": "4",
    "friday": "5",
    "fri": "5",
    "saturday": "6",
    "sat": "6",
}
_DAY_SPEC_ALIASES: dict[str, str] = {
    "day": "*",
    "daily": "*",
    "everyday": "*",
    "weekday": "1-5",
    "weekdays": "1-5",
    "weekend": "0,6",
    "weekends": "0,6",
}

_MONTH_NAMES: dict[str, int] = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_CRON_WEEKDAY_NAMES: dict[str, int] = {
    name: index
    for index, name in enumerate(("sun", "mon", "tue", "wed", "thu", "fri", "sat"))
}
_CRON_TOKEN_RE = re.compile(r"^[A-Za-z0-9*,/-]+$")
# Fields are ordered second, minute, hour, day-of-month, month, weekday.
_CRON_LIMITS: tuple[tuple[int, int], ...] = (
    (0, 59), (0, 59), (0, 23), (1, 31), (1, 12), (0, 6)
)
_CRON_FULL: tuple[frozenset[int], ...] = (
    frozenset(range(0, 60)),
    frozenset(range(0, 60)),
    frozenset(range(0, 24)),
    frozenset(range(1, 32)),
    frozenset(range(1, 13)),
    frozenset(range(0, 7)),
)

_CLOCK_RE = re.compile(r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<meridiem>am|pm)?$")


class ScheduleGrammarError(ValueError):
    """A stable, user-safe schedule grammar rejection.

    ``code`` is the machine-readable contract; ``str(error)`` is safe to show
    in a tool result. Rejected input text is never echoed beyond the short
    ``quoted`` fragment the caller already supplied as a schedule string.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _error(code: ErrorCode, message: str) -> ScheduleGrammarError:
    return ScheduleGrammarError(code, message)


# ---------------------------------------------------------------------------
# Duration and clock parsing
# ---------------------------------------------------------------------------


def parse_duration(text: Any) -> int:
    """Parse a final-head duration into whole minutes.

    Accepted units are minutes (``m``/``min``/``minutes``), hours
    (``h``/``hr``/``hours``) and days (``d``/``day``/``days``). A missing count
    means one. Seconds and weeks are intentionally rejected.
    """

    if not isinstance(text, str):
        raise _error("invalid_duration", "Invalid duration: expected text such as '30m'.")
    cleaned = text.strip().lower()
    match = _DURATION_RE.match(cleaned)
    if not match:
        raise _error(
            "invalid_duration",
            f"Invalid duration: {text.strip()!r}. "
            "Use '30m', '2h', '1d', or a bare unit such as 'hour'.",
        )
    count = int(match.group("count")) if match.group("count") else 1
    return count * _DURATION_UNITS[match.group("unit")]


def parse_clock_time(text: str) -> tuple[int, int] | None:
    """Parse ``9am``/``9:30am``/``14:00``/``7``/``noon``/``midnight``.

    A bare number is a 24-hour hour, matching final head. Returns ``None`` for
    unrecognized text so the natural-language fallback can report an invalid
    duration, as the frozen fixtures require.
    """

    cleaned = (text or "").strip().lower().replace(" ", "")
    if not cleaned:
        return None
    if cleaned in ("noon", "midday"):
        return (12, 0)
    if cleaned == "midnight":
        return (0, 0)
    match = _CLOCK_RE.match(cleaned)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    meridiem = match.group("meridiem")
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    return (hour, minute)


def _natural_schedule_to_cron(rest: str) -> str | None:
    """Convert a natural day/time phrase to a 5-field cron expression."""

    tokens = rest.lower().replace(",", " ").split()
    if not tokens:
        return None
    dow = _DAY_SPEC_ALIASES.get(tokens[0])
    index = 1
    if dow is None:
        days: list[str] = []
        index = len(tokens)
        for position, token in enumerate(tokens):
            if token == "and":
                continue
            mapped = _WEEKDAY_ALIASES.get(token)
            if mapped is None:
                index = position
                break
            if mapped not in days:
                days.append(mapped)
        if not days:
            return None
        dow = ",".join(days)
    time_tokens = tokens[index:]
    if time_tokens and time_tokens[0] == "at":
        time_tokens = time_tokens[1:]
    if not time_tokens:
        return None
    parsed = parse_clock_time(" ".join(time_tokens))
    if parsed is None:
        return None
    hour, minute = parsed
    return f"{minute} {hour} * * {dow}"


# ---------------------------------------------------------------------------
# Cron parsing and matching (5- and 6-field, named months/weekdays)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedCron:
    """A validated cron expression with second/minute/hour/... sets."""

    expression: str
    has_seconds: bool
    fields: tuple[frozenset[int], ...]

    def matches(self, moment: datetime) -> bool:
        """Whether the date/time fields of ``moment`` match (seconds included)."""

        second, minute, hour, day, month, weekday = self.fields
        if moment.second not in second or moment.minute not in minute or moment.hour not in hour:
            return False
        if moment.month not in month:
            return False
        cron_dow = (moment.weekday() + 1) % 7
        day_restricted = day != _CRON_FULL[3]
        dow_restricted = weekday != _CRON_FULL[5]
        if day_restricted and dow_restricted:
            return moment.day in day or cron_dow in weekday
        if day_restricted:
            return moment.day in day
        if dow_restricted:
            return cron_dow in weekday
        return True


def _cron_field_int(token: str, index: int, expression: str) -> int:
    text = token.strip().lower()
    if not text:
        raise _error("invalid_cron_expression", f"Invalid cron expression {expression!r}: empty field.")
    if index == 4 and text[:3] in _MONTH_NAMES:
        return _MONTH_NAMES[text[:3]]
    if index == 5 and text[:3] in _CRON_WEEKDAY_NAMES:
        return _CRON_WEEKDAY_NAMES[text[:3]]
    try:
        return int(text)
    except ValueError:
        raise _error(
            "invalid_cron_expression",
            f"Invalid cron expression {expression!r}: bad field value {token!r}.",
        ) from None


def _parse_cron_field(field: str, index: int, expression: str) -> frozenset[int]:
    low, high = _CRON_LIMITS[index]
    values: set[int] = set()
    for part in field.split(","):
        base, _, step_text = part.partition("/")
        try:
            step = int(step_text) if step_text else 1
        except ValueError:
            raise _error(
                "invalid_cron_expression",
                f"Invalid cron expression {expression!r}: bad step {step_text!r}.",
            ) from None
        if step <= 0:
            raise _error("invalid_cron_expression", f"Invalid cron expression {expression!r}: non-positive step.")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            first, last = base.split("-", 1)
            start = _cron_field_int(first, index, expression)
            end = _cron_field_int(last, index, expression)
        else:
            start = _cron_field_int(base, index, expression)
            # "5/10" means 5, 15, 25, ... rather than a single value.
            end = high if step_text else start
        if start > end:
            raise _error(
                "invalid_cron_expression",
                f"Invalid cron expression {expression!r}: range start exceeds end.",
            )
        values.update(range(start, end + 1, step))
    if index == 5:
        # Cron accepts 7 as an alias for Sunday. Convert before the range check.
        values = {0 if value == 7 else value for value in values}
    invalid = sorted(value for value in values if not low <= value <= high)
    if invalid:
        raise _error(
            "invalid_cron_expression",
            f"Invalid cron expression {expression!r}: field out of range {invalid}.",
        )
    if not values:
        raise _error("invalid_cron_expression", f"Invalid cron expression {expression!r}: empty field.")
    return frozenset(values)


def parse_cron_expression(expression: str) -> ParsedCron:
    """Validate a 5- or 6-field cron expression.

    Five fields are ``minute hour day month weekday``; six fields prepend a
    seconds field. Named months (``JAN``) and weekdays (``MON``) are accepted.
    """

    text = (expression or "").strip()
    parts = text.split()
    if len(parts) not in (5, 6):
        raise _error(
            "invalid_cron_expression",
            f"Invalid cron expression {text!r}: expected 5 or 6 fields.",
        )
    if not all(_CRON_TOKEN_RE.match(part) for part in parts):
        raise _error(
            "invalid_cron_expression",
            f"Invalid cron expression {text!r}: unsupported characters.",
        )
    has_seconds = len(parts) == 6
    if not has_seconds:
        parts = ["0", *parts]
    fields = tuple(
        _parse_cron_field(part, index, text) for index, part in enumerate(parts)
    )
    return ParsedCron(expression=text, has_seconds=has_seconds, fields=fields)


def _cron_wall_times(
    parsed: ParsedCron, start: datetime, *, max_days: int = 366 * 8
) -> Iterator[datetime]:
    """Yield matching naive local wall-clock datetimes at/after ``start``."""

    seconds, minutes, hours, days, months, weekdays = parsed.fields
    ordered_hours = sorted(hours)
    ordered_minutes = sorted(minutes)
    ordered_seconds = sorted(seconds)
    current = start.date()
    last = start.date() + timedelta(days=max_days)
    while current <= last:
        if current.month in months:
            cron_dow = (current.weekday() + 1) % 7
            day_restricted = days != _CRON_FULL[3]
            dow_restricted = weekdays != _CRON_FULL[5]
            if day_restricted and dow_restricted:
                matches_day = current.day in days or cron_dow in weekdays
            elif day_restricted:
                matches_day = current.day in days
            elif dow_restricted:
                matches_day = cron_dow in weekdays
            else:
                matches_day = True
            if matches_day:
                for hour in ordered_hours:
                    for minute in ordered_minutes:
                        for second in ordered_seconds:
                            candidate = datetime(
                                current.year, current.month, current.day,
                                hour, minute, second,
                            )
                            if candidate >= start:
                                yield candidate
        current += timedelta(days=1)


# ---------------------------------------------------------------------------
# IANA timezone and DST resolution
# ---------------------------------------------------------------------------


def normalize_timezone(value: Any) -> str | None:
    """Validate and canonicalize an IANA timezone name.

    ``None``/``""`` mean "inherit"; anything else must resolve through
    ``zoneinfo`` or it is rejected before a proposal can exist.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise _error(
            "invalid_timezone",
            "timezone must be an IANA name such as 'America/New_York'.",
        )
    name = value.strip()
    if not name:
        return None
    try:
        return ZoneInfo(name).key
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise _error(
            "invalid_timezone",
            f"Unknown timezone {name!r}; use an IANA key from zoneinfo.available_timezones().",
        ) from exc


def system_timezone_name() -> str:
    """Best-effort IANA name for the system local zone, defaulting to UTC."""

    candidates: list[str] = []
    env = os.environ.get("TZ")
    if env:
        candidates.append(env)
    try:
        target = os.path.realpath("/etc/localtime")
        marker = "/zoneinfo/"
        if marker in target:
            candidates.append(target.split(marker, 1)[1])
    except OSError:
        pass
    for path in ("/etc/timezone", Path("/var/db/timezone/zoneinfo").as_posix()):
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if text and "/" in text:
            candidates.append(text.splitlines()[0])
    for candidate in candidates:
        try:
            return ZoneInfo(candidate.strip()).key
        except (ZoneInfoNotFoundError, ValueError):
            continue
    return "UTC"


def default_timezone_name(settings: Any | None = None) -> str:
    """New-schedule default zone: ``automation.default_timezone`` then system local.

    ``settings`` is optional so the compiler stays side-effect free and usable
    without the full application composition root. LEO's current
    ``AutomationConfig`` has no timezone field; a future config addition is
    picked up through ``getattr`` without a second source of truth.
    """

    if settings is not None:
        automation = getattr(settings, "automation", None)
        configured = getattr(automation, "default_timezone", None)
        if configured:
            normalized = normalize_timezone(str(configured))
            if normalized:
                return normalized
    return system_timezone_name()


def resolve_timezone(value: Any, *, default: str | None = None) -> str:
    """Resolve an optional pin against the configured default."""

    explicit = normalize_timezone(value)
    if explicit:
        return explicit
    base = normalize_timezone(default) if default else None
    return base or system_timezone_name()


def _valid_wall_time(naive: datetime, zone: ZoneInfo) -> bool:
    for fold in (0, 1):
        instant = naive.replace(tzinfo=zone, fold=fold)
        # Round-trip through UTC: ``astimezone(zone)`` on an already-zoned
        # datetime preserves the wall clock and cannot detect a gap.
        if instant.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive:
            return True
    return False


def _first_valid_after_gap(naive: datetime, zone: ZoneInfo, *, limit_minutes: int = 24 * 60) -> datetime:
    candidate = naive.replace(second=0, microsecond=0)
    for _ in range(limit_minutes + 1):
        if _valid_wall_time(candidate, zone):
            return candidate
        candidate += timedelta(minutes=1)
    # A date with no valid wall time at all cannot exist in a real zone; fail
    # closed rather than inventing an instant.
    raise _error("invalid_timezone", "The requested local time does not exist in this timezone.")


def wall_time_instants(naive: datetime, zone: ZoneInfo) -> list[datetime]:
    """All absolute instants for a naive wall time, or the gap-advanced instant.

    Ambiguous (folded) times produce two instants, earliest first. A
    nonexistent time advances to the first valid minute after the gap, matching
    LEO's documented DST safety policy.
    """

    valid: list[datetime] = []
    for fold in (0, 1):
        instant = naive.replace(tzinfo=zone, fold=fold)
        if instant.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive:
            valid.append(instant)
    if valid:
        # Datetime equality ignores ``fold`` for the same tzinfo, so dedupe on
        # the absolute UTC instant instead of on the datetime object.
        by_instant = {instant.astimezone(UTC): instant for instant in valid}
        return [by_instant[moment] for moment in sorted(by_instant)]
    first = _first_valid_after_gap(naive, zone)
    return [first.replace(tzinfo=zone, fold=0)]


def resolve_wall_time(naive: datetime, zone: ZoneInfo) -> datetime:
    """Resolve one wall time: first fold when ambiguous, first valid minute in a gap."""

    return wall_time_instants(naive, zone)[0]


def next_cron_instant(parsed: ParsedCron, zone: ZoneInfo, after: datetime) -> datetime:
    """First instant strictly after ``after`` matching the cron expression in ``zone``."""

    after_utc = after.astimezone(UTC) if after.tzinfo else after.replace(tzinfo=UTC)
    local = after_utc.astimezone(zone)
    # A fall-back fold means the same wall clock occurs twice, so the wall
    # candidate for the *second* occurrence can be earlier than the base's own
    # wall clock. Start a little earlier and let the absolute comparison below
    # enforce "strictly after".
    start = local.replace(tzinfo=None) - timedelta(hours=2)
    best: datetime | None = None
    for wall in _cron_wall_times(parsed, start):
        for instant in wall_time_instants(wall, zone):
            instant_utc = instant.astimezone(UTC)
            if instant_utc > after_utc:
                if best is None or instant_utc < best.astimezone(UTC):
                    best = instant
        if best is not None:
            # Wall times are generated in ascending order, but a gap can move a
            # later wall candidate ahead of an earlier one. One extra day of
            # candidates is enough to prove the minimum; stop once the next
            # wall day cannot beat the best instant.
            next_day = wall.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            try:
                next_first = next(_cron_wall_times(parsed, next_day))
            except StopIteration:
                next_first = None
            if next_first is None:
                break
            next_instant = wall_time_instants(next_first, zone)[0].astimezone(UTC)
            if next_instant > best.astimezone(UTC):
                break
    if best is None:
        raise _error(
            "no_schedule_match",
            f"No occurrence found for cron expression {parsed.expression!r}.",
        )
    return best


# ---------------------------------------------------------------------------
# Top-level schedule parser
# ---------------------------------------------------------------------------

ScheduleKind = Literal["cron", "interval", "once"]


@dataclass(frozen=True)
class ParsedSchedule:
    """Canonical output of the final-head schedule grammar."""

    kind: ScheduleKind
    expression: str
    timezone: str
    display: str
    minutes: int | None = None
    run_at: datetime | None = None
    cron: ParsedCron | None = None
    legacy_every: bool = False


def _interval_schedule(minutes: int, *, display: str, timezone_name: str) -> ParsedSchedule:
    if minutes <= 0:
        raise _error(
            "non_positive_interval",
            "Invalid duration: a recurring interval must be positive.",
        )
    return ParsedSchedule(
        kind="interval",
        # LEO's scheduler consumes seconds; final-head semantic minutes are
        # exposed separately and used for display/equality.
        expression=str(minutes * 60),
        timezone=timezone_name,
        display=display,
        minutes=minutes,
    )


def parse_schedule(
    schedule: Any,
    *,
    now: datetime,
    default_timezone: str | None = None,
    timezone: Any = None,
    allow_legacy_every: bool = False,
) -> ParsedSchedule:
    """Parse a final-head schedule string into a canonical expression.

    ``now`` is required and must be aware (or is treated as UTC). New schedules
    pass ``allow_legacy_every=False``; readers of existing LEO rows may pass
    ``True`` to keep ``@every`` rows readable without making it a writable
    form.
    """

    if not isinstance(schedule, str) or not schedule.strip():
        raise _error("invalid_schedule", "Invalid schedule: schedule is required.")
    original = schedule.strip()
    text = original
    lowered = text.lower()
    default_zone = resolve_timezone(default_timezone)
    expression_zone = resolve_timezone(timezone, default=default_zone)
    now_utc = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)

    is_every = lowered.startswith("every ")
    rest = text[6:].strip() if is_every else lowered

    natural_cron = _natural_schedule_to_cron(rest)
    if natural_cron is not None:
        parsed_cron = parse_cron_expression(natural_cron)
        return ParsedSchedule(
            kind="cron",
            expression=natural_cron,
            timezone=expression_zone,
            display=original,
            cron=parsed_cron,
        )

    if is_every and not _starts_with_legacy_every(rest):
        minutes = parse_duration(rest)
        return _interval_schedule(
            minutes,
            display=f"every {minutes}m",
            timezone_name=expression_zone,
        )

    if _starts_with_legacy_every(text):
        if not allow_legacy_every:
            raise _error(
                "invalid_schedule",
                "Invalid schedule: '@every' is legacy-readable only; use '30m' or 'every 30m'.",
            )
        seconds = _parse_legacy_every(text)
        expression_text = str(int(seconds)) if float(seconds).is_integer() else str(seconds)
        return ParsedSchedule(
            kind="interval",
            expression=expression_text,
            timezone=expression_zone,
            display=original,
            minutes=max(1, int(seconds // 60)),
            legacy_every=True,
        )

    parts = text.split()
    if len(parts) >= 5:
        parsed_cron = parse_cron_expression(text)
        return ParsedSchedule(
            kind="cron",
            expression=parsed_cron.expression,
            timezone=expression_zone,
            display=original,
            cron=parsed_cron,
        )

    if "T" in text or re.match(r"^\d{4}-\d{2}-\d{2}", text):
        try:
            parsed_dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise _error("invalid_timestamp", f"Invalid timestamp {text!r}: {exc}") from None
        if parsed_dt.tzinfo is None:
            # A naive one-shot is interpreted on the configured profile clock;
            # a per-job timezone pin never reinterprets it.
            local_zone = ZoneInfo(default_zone)
            local_dt = parsed_dt.replace(tzinfo=local_zone)
            local_dt = resolve_wall_time(parsed_dt, local_zone)
            display_dt = local_dt.astimezone(local_zone)
        else:
            local_dt = parsed_dt
            display_dt = parsed_dt
        run_at = local_dt.astimezone(UTC)
        return ParsedSchedule(
            kind="once",
            expression="",
            timezone=expression_zone,
            display=f"once at {display_dt.strftime('%Y-%m-%d %H:%M')}",
            run_at=run_at,
        )

    if lowered.startswith("in "):
        duration_text = text[3:].strip()
        try:
            minutes = parse_duration(duration_text)
        except ScheduleGrammarError:
            raise _error(
                "invalid_duration",
                f"Invalid duration {duration_text!r} after 'in '. Use 'in 30m' or 'in 2h'.",
            ) from None
        run_at = now_utc + timedelta(minutes=minutes)
        return ParsedSchedule(
            kind="once",
            expression="",
            timezone=expression_zone,
            display=f"once in {duration_text}",
            run_at=run_at,
        )

    try:
        minutes = parse_duration(text)
    except ScheduleGrammarError:
        raise _error(
            "invalid_schedule",
            f"Invalid schedule {original!r}. Use an interval ('30m', 'every 2h'), "
            "a one-shot ('in 30m', ISO timestamp), a day/time phrase "
            "('weekdays at 9am'), or a 5/6-field cron expression.",
        ) from None
    return _interval_schedule(
        minutes,
        display=f"every {minutes}m",
        timezone_name=expression_zone,
    )


_LEGACY_EVERY_RE = re.compile(
    r"^@every\s+(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>s|sec|secs|second|seconds|m|min|mins|minute|minutes"
    r"|h|hour|hours|d|day|days|w|week|weeks)?$",
    re.IGNORECASE,
)


def _starts_with_legacy_every(text: str) -> bool:
    return text.strip().lower().startswith("@every")


def _parse_legacy_every(text: str) -> float:
    match = _LEGACY_EVERY_RE.match(text.strip())
    if not match:
        raise _error("invalid_schedule", f"Invalid schedule {text.strip()!r}.")
    amount = float(match.group("amount"))
    unit = (match.group("unit") or "s").lower()
    if unit.startswith("w"):
        seconds = amount * 604800.0
    elif unit.startswith("d"):
        seconds = amount * 86400.0
    elif unit.startswith("h"):
        seconds = amount * 3600.0
    elif unit.startswith("m"):
        seconds = amount * 60.0
    else:
        seconds = amount
    if seconds <= 0:
        raise _error("invalid_schedule", "@every interval must be positive.")
    # Legacy rows keep their exact second cadence, including sub-second ones.
    return seconds


def normalize_repeat(value: Any, *, kind: ScheduleKind) -> int | None:
    """Normalize a repeat value into ``None`` (forever) or a positive limit.

    LEO hardening (documented against the oracle gaps):

    * booleans and fractional numbers are rejected instead of coerced;
    * zero and negative values are rejected instead of becoming forever;
    * ``once`` defaults to one run and rejects an explicit limit above one.
    """

    if value is None:
        return 1 if kind == "once" else None
    if isinstance(value, bool):
        raise _error("invalid_repeat", "Invalid repeat value: booleans are not run limits.")
    if isinstance(value, float):
        raise _error("invalid_repeat", "Invalid repeat value: use a whole number of runs.")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("forever", "infinite", "inf", "none", ""):
            return 1 if kind == "once" else None
        if text in ("once", "one", "1x"):
            normalized = 1
        else:
            if not re.fullmatch(r"[+-]?\d+", text):
                raise _error(
                    "invalid_repeat",
                    f"Invalid repeat value {value!r}: use an integer, 'forever', or 'once'.",
                )
            normalized = int(text)
    elif isinstance(value, int):
        normalized = value
    else:
        raise _error(
            "invalid_repeat",
            f"Invalid repeat value {value!r}: use an integer, 'forever', or 'once'.",
        )
    if normalized <= 0:
        raise _error("invalid_repeat", "Invalid repeat value: run limits must be positive.")
    if kind == "once":
        if normalized != 1:
            raise _error("invalid_repeat", "A one-shot schedule can only run once.")
        return 1
    return normalized


def next_occurrence(parsed: ParsedSchedule, after: datetime) -> datetime:
    """First run instant strictly after ``after`` for a parsed schedule."""

    after_utc = after.astimezone(UTC) if after.tzinfo else after.replace(tzinfo=UTC)
    if parsed.kind == "once":
        if parsed.run_at is None:
            raise _error("invalid_schedule", "A one-shot schedule requires a run time.")
        return parsed.run_at
    if parsed.kind == "interval":
        # Interval expressions are canonical seconds (new and legacy rows).
        return after_utc + timedelta(seconds=float(parsed.expression))
    if parsed.cron is None:
        raise _error("invalid_schedule", "A cron schedule requires a cron expression.")
    return next_cron_instant(parsed.cron, ZoneInfo(parsed.timezone), after_utc)


__all__ = [
    "ParsedCron",
    "ParsedSchedule",
    "ScheduleGrammarError",
    "default_timezone_name",
    "next_cron_instant",
    "next_occurrence",
    "normalize_repeat",
    "normalize_timezone",
    "parse_clock_time",
    "parse_cron_expression",
    "parse_duration",
    "parse_schedule",
    "resolve_timezone",
    "resolve_wall_time",
    "system_timezone_name",
    "wall_time_instants",
]
