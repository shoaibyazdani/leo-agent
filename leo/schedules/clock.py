"""Injectable clocks for schedule anchoring, deadlines and deterministic tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


def utcnow() -> datetime:
    """Aware UTC now. Schedule times are always stored/evaluated in UTC."""
    return datetime.now(timezone.utc)


@runtime_checkable
class Clock(Protocol):
    """The one time source injected into schedulers and approval checks."""

    def now(self) -> datetime: ...


class SystemClock:
    """Real wall clock in UTC."""

    def now(self) -> datetime:
        return utcnow()


class FrozenClock:
    """A clock that only moves when a test moves it explicitly."""

    def __init__(self, moment: datetime | str | None = None) -> None:
        self._moment = self._normalize(moment if moment is not None else utcnow())

    @staticmethod
    def _normalize(value: datetime | str) -> datetime:
        moment = datetime.fromisoformat(value) if isinstance(value, str) else value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._moment

    def set(self, moment: datetime | str) -> datetime:
        self._moment = self._normalize(moment)
        return self._moment

    def advance(self, seconds: float = 0.0, **kwargs: float) -> datetime:
        delta = timedelta(seconds=float(seconds), **kwargs)
        if delta < timedelta(0):
            raise ValueError("a frozen clock cannot move backwards")
        self._moment = self._moment + delta
        return self._moment


__all__ = ["Clock", "FrozenClock", "SystemClock", "utcnow"]
