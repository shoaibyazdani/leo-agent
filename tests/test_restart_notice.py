"""/restart must confirm that the process actually came back.

The command replies before the process dies, so from the chat a clean restart and
a crash loop look identical. The marker is what closes that gap: written before
the restart, answered by the next boot, and silent when nobody asked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from leo import db
from leo.core.restart_notice import (
    RESTART_ANNOUNCEMENT,
    RESTART_MARKER_KEY,
    announce_restart_if_pending,
    mark_restart_pending,
    record_boot,
)


class _Adapter:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send(self, target: str, text: str, **kwargs: Any) -> dict[str, Any]:
        self.sent.append((str(target), str(text)))
        return {"ok": True}


def _app(adapter: _Adapter | None = None, channel: str = "telegram") -> SimpleNamespace:
    adapters = {channel: adapter} if adapter is not None else {}
    return SimpleNamespace(gateway=SimpleNamespace(adapters=adapters))


@pytest.fixture
async def notice_db(tmp_path):
    db.set_db_path(tmp_path / "restart.db")
    await db.initialize()
    yield db


@pytest.mark.asyncio
async def test_a_requested_restart_is_announced_once(notice_db) -> None:
    adapter = _Adapter()
    await mark_restart_pending("telegram", "525083981", units=["leo-agent"])

    text = await announce_restart_if_pending(_app(adapter))

    assert text == RESTART_ANNOUNCEMENT
    assert adapter.sent == [("525083981", RESTART_ANNOUNCEMENT)]
    # the marker is consumed: a later boot stays quiet
    assert await announce_restart_if_pending(_app(adapter)) is None
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_a_cold_start_says_nothing(notice_db) -> None:
    adapter = _Adapter()
    assert await announce_restart_if_pending(_app(adapter)) is None
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_a_stale_marker_is_dropped_instead_of_announced(notice_db) -> None:
    """A marker from a crashed boot must not announce hours later."""
    adapter = _Adapter()
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    await db.execute(
        "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?)",
        (RESTART_MARKER_KEY, f'{{"channel": "telegram", "subject": "525083981", "at": "{stale}"}}',
         stale),
    )

    assert await announce_restart_if_pending(_app(adapter)) is None
    assert adapter.sent == []
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (RESTART_MARKER_KEY,))
    assert row is None, "a stale marker must not survive to fire on the next boot"


@pytest.mark.asyncio
async def test_a_missing_adapter_does_not_leave_the_marker_behind(notice_db) -> None:
    await mark_restart_pending("telegram", "525083981", units=["leo-agent"])
    assert await announce_restart_if_pending(_app(None)) is None
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (RESTART_MARKER_KEY,))
    assert row is None


@pytest.mark.asyncio
async def test_the_marker_records_who_asked(notice_db) -> None:
    await mark_restart_pending("telegram", "525083981", units=["leo-agent", "leo-gateway"])
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (RESTART_MARKER_KEY,))
    assert row is not None
    import json

    payload = json.loads(str(row["value"]))
    assert payload["channel"] == "telegram"
    assert payload["subject"] == "525083981"
    assert payload["units"] == ["leo-agent", "leo-gateway"]


@pytest.mark.asyncio
async def test_the_notice_waits_for_an_adapter_that_registers_late(notice_db) -> None:
    """Boot registers adapters after initialize(); the notice must not give up."""
    adapter = _Adapter()
    late = SimpleNamespace(gateway=SimpleNamespace(adapters={}))
    await mark_restart_pending("telegram", "525083981", units=["leo-agent"])

    async def _deliver_later() -> None:
        import asyncio

        await asyncio.sleep(0.05)
        late.gateway.adapters["telegram"] = adapter

    import asyncio

    task = asyncio.create_task(_deliver_later())
    text = await announce_restart_if_pending(late, wait_seconds=2.0, poll_seconds=0.05)
    await task

    assert text == RESTART_ANNOUNCEMENT
    assert adapter.sent == [("525083981", RESTART_ANNOUNCEMENT)]


@pytest.mark.asyncio
async def test_a_restart_without_a_request_still_reports(notice_db) -> None:
    """systemd bouncing the unit is a restart too — the operator wants to know."""
    adapter = _Adapter()
    await record_boot()  # previous boot stamp
    await db.execute(
        "INSERT INTO conversations(id, channel, subject, created_at, updated_at) "
        "VALUES(?,?,?,?,?)",
        ("conv_t1", "telegram", "525083981",
         datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
    )

    text = await announce_restart_if_pending(_app(adapter), restarted=True)

    assert text == RESTART_ANNOUNCEMENT
    assert adapter.sent == [("525083981", RESTART_ANNOUNCEMENT)]


@pytest.mark.asyncio
async def test_a_cold_boot_without_a_marker_or_history_is_silent(notice_db) -> None:
    adapter = _Adapter()
    assert await announce_restart_if_pending(_app(adapter), restarted=False) is None
    assert adapter.sent == []
