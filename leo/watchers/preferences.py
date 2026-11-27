"""Plain-text operator preferences for the judgment watcher.

The file is deliberately boring: no YAML nesting and no schema to remember.

    # topics to tell
    - deadlines
    - github
    # never mention
    - health
    - politics
    # quiet hours
    23:00-08:00
    # max items per day
    2

Every watcher run re-reads the file, so edits take effect on the next run.
Missing or malformed lines fall back to safe defaults rather than failing the
run; a missing file means "tell me about up to two ordinary things per day,
outside the configured waking hours".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from pathlib import Path

_TELL = {"topics to tell", "topics to tell about", "tell", "tell about", "topics"}
_NEVER = {"topics never to mention", "never mention", "never", "exclude", "excluded topics"}
_QUIET = {"quiet hours", "quiet", "silent hours"}
_MAX = {"max items per day", "max ordinary items per day", "maximum items per day", "max"}
_ZONE = {"timezone", "time zone"}
_RANGE = re.compile(r"^(\d{1,2}):(\d{2})\s*(?:-|–|—|to)\s*(\d{1,2}):(\d{2})$", re.I)


@dataclass(frozen=True)
class Preferences:
    tell_topics: tuple[str, ...] = ()
    never_topics: tuple[str, ...] = ()
    quiet_hours: tuple[str, str] | None = None
    max_ordinary_per_day: int = 2
    timezone: str | None = None


def _clean(value: str) -> str:
    return value.strip().strip("\"'").strip()


def _topics(value: str) -> tuple[str, ...]:
    parts = [item.strip() for item in value.split(",")]
    return tuple(_clean(item) for item in parts if _clean(item))


def _clock(value: str) -> str | None:
    match = _RANGE.match(_clean(value))
    if match is None:
        return None
    hour, minute, end_hour, end_minute = (int(item) for item in match.groups())
    if max(hour, end_hour) > 23 or max(minute, end_minute) > 59:
        return None
    return f"{hour:02d}:{minute:02d}-{end_hour:02d}:{end_minute:02d}"


def load_preferences(
    path: str | Path | None,
    *,
    default_quiet_hours: tuple[str, str] = ("23:00", "09:00"),
    default_max: int = 2,
    default_timezone: str | None = None,
) -> Preferences:
    """Read the preferences file; defaults apply when a line is missing."""
    tell: list[str] = []
    never: list[str] = []
    quiet: str | None = None
    maximum = max(0, int(default_max))
    timezone = default_timezone
    section: str | None = None

    text = ""
    if path:
        try:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            text = ""
    for raw in text.splitlines():
        # Comment-style headings are common in a hand-written file:
        #   # never mention
        #   - health
        stripped = raw.strip()
        if stripped.startswith("#"):
            heading = _clean(stripped.lstrip("#").rstrip(":")).lower()
            if heading in (_TELL | _NEVER | _QUIET | _MAX | _ZONE):
                section = heading
            continue
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # A bare time range under the quiet-hours heading contains colons, so
        # parse it before the generic key: value split.
        if section in _QUIET and _clock(line) is not None:
            quiet = _clock(line)
            continue
        key, sep, value = line.partition(":")
        lowered = _clean(key).lower()
        if sep and lowered in (_TELL | _NEVER | _QUIET | _MAX | _ZONE):
            section = lowered
            value = value.strip()
        elif line.startswith(("-", "*", "•")):
            value = line.lstrip("-*• ").strip()
        else:
            value = ""
            if not sep and section in _QUIET and _clock(line) is not None:
                # A bare time range under the quiet-hours heading is valid.
                value = line
            elif not sep and section in _MAX and re.search(r"\d", line):
                value = line
        if not value:
            continue
        if section in _TELL:
            tell.extend(_topics(value))
        elif section in _NEVER:
            never.extend(_topics(value))
        elif section in _QUIET:
            parsed = _clock(value)
            if parsed:
                quiet = parsed
        elif section in _MAX:
            digits = re.search(r"\d+", value)
            if digits:
                maximum = max(0, int(digits.group()))
        elif section in _ZONE:
            candidate = _clean(value)
            if candidate:
                timezone = candidate

    if quiet is None:
        quiet = f"{default_quiet_hours[0]}-{default_quiet_hours[1]}"
    start, _, end = quiet.partition("-")
    return Preferences(
        tell_topics=tuple(dict.fromkeys(tell)),
        never_topics=tuple(dict.fromkeys(never)),
        quiet_hours=(start, end),
        max_ordinary_per_day=maximum,
        timezone=timezone,
    )


def mentions_topic(text: str, topics: tuple[str, ...] | list[str]) -> bool:
    """True when any configured topic appears as a whole term in the text.

    Matching is case-insensitive and respects term boundaries, so "health"
    excludes a health update but not an unrelated "healthy" status. Topics that
    start or end with punctuation (e.g. "C++") do not require a boundary there.
    """
    haystack = str(text or "").casefold()
    for topic in topics:
        term = str(topic).strip().casefold()
        if not term:
            continue
        before = r"(?<!\w)" if term[0].isalnum() or term[0] == "_" else ""
        after = r"(?!\w)" if term[-1].isalnum() or term[-1] == "_" else ""
        if re.search(before + re.escape(term) + after, haystack):
            return True
    return False


def clock_in_range(value: time, start: str, end: str) -> bool:
    """Inclusive start, exclusive end; ranges that cross midnight are supported."""
    begin = time.fromisoformat(start)
    finish = time.fromisoformat(end)
    if begin <= finish:
        return begin <= value < finish
    return value >= begin or value < finish
