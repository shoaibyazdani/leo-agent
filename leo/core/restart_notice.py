"""Tell the operator when a requested restart has actually come back.

`/restart` replies before it dies ("back in a few seconds"), which is honest but
incomplete: from the chat there is no way to tell a clean restart from a crash
loop that never returns. The command therefore leaves a short-lived marker and
the next boot answers it:

    ♻ Gateway restarted successfully. Your session continues.

A cold start with no marker says nothing — booting is not news.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import db

_LOG = logging.getLogger(__name__)

RESTART_MARKER_KEY = "restart:pending"
LAST_BOOT_KEY = "runtime:last_boot"
RESTART_ANNOUNCEMENT = "♻ Gateway restarted successfully. Your session continues."
# Long enough for a slow stop + start, short enough that a marker left behind by
# a crashed boot cannot announce itself hours later.
RESTART_MARKER_TTL = timedelta(minutes=15)
# A boot this soon after the previous one is a restart, not a cold start: the
# operator asked, systemd corrected a crash, or a deploy bounced the unit.
RESTART_WINDOW = timedelta(minutes=15)
# A boot this soon after the previous one is a crash loop, not a restart the
# operator asked for: one notice is enough.
CRASH_LOOP_FLOOR = timedelta(seconds=60)


async def mark_restart_pending(channel: str, subject: str, *, units: list[str] | None = None) -> None:
    """Remember who asked for the restart, so the next boot can answer them."""
    if not channel or not subject:
        return
    payload = {
        "channel": str(channel),
        "subject": str(subject),
        "units": list(units or []),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    await db.execute(
        "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (RESTART_MARKER_KEY, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
    )


async def _read_marker() -> dict[str, Any] | None:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (RESTART_MARKER_KEY,))
    if row is None:
        return None
    try:
        payload = json.loads(str(row["value"]))
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


async def _take_marker() -> dict[str, Any] | None:
    """Read and clear the marker in one step: one boot, one announcement."""
    payload = await _read_marker()
    await db.execute("DELETE FROM kv WHERE key=?", (RESTART_MARKER_KEY,))
    return payload


def _is_fresh(payload: dict[str, Any]) -> bool:
    try:
        asked_at = datetime.fromisoformat(str(payload.get("at")))
    except (TypeError, ValueError):
        return True  # no usable timestamp: the marker itself is the intent
    if asked_at.tzinfo is None:
        asked_at = asked_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - asked_at <= RESTART_MARKER_TTL


async def _seconds_since_previous_boot() -> float | None:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (LAST_BOOT_KEY,))
    if row is None:
        return None
    try:
        previous = datetime.fromisoformat(str(row["value"]))
    except (TypeError, ValueError):
        return None
    if previous.tzinfo is None:
        previous = previous.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - previous).total_seconds()


async def _previous_boot_was_recent() -> bool:
    age = await _seconds_since_previous_boot()
    return age is not None and age <= RESTART_WINDOW.total_seconds()


async def record_boot() -> None:
    """Stamp this boot, so the next one can tell a restart from a cold start."""
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (LAST_BOOT_KEY, now, now),
    )


async def _last_chat(app: Any) -> tuple[str, str] | None:
    """Fallback target: the conversation most recently active on a live channel."""
    adapters = getattr(getattr(app, "gateway", None), "adapters", {}) or {}
    if not adapters:
        return None
    placeholders = ",".join("?" for _ in adapters)
    row = await db.fetchone(
        f"SELECT channel, subject FROM conversations WHERE channel IN ({placeholders}) "
        "AND subject <> '' ORDER BY updated_at DESC LIMIT 1",
        tuple(adapters),
    )
    if row is None:
        return None
    return str(row["channel"] or ""), str(row["subject"] or "")


async def announce_restart_if_pending(
    app: Any, *, wait_seconds: float = 0.0, poll_seconds: float = 1.0,
    restarted: bool | None = None,
) -> str | None:
    """Say the restart landed, once, to the chat that asked for it.

    Booting happens before channel adapters are registered, so a boot-time call
    can arrive a moment too early. ``wait_seconds`` lets it wait for the adapter
    instead of dropping a message the operator is expecting.
    """
    payload = await _read_marker()
    if restarted is None:
        restarted = await _previous_boot_was_recent()
    # "Tell me when it is up" applies to every start, not only to a /restart:
    # the operator restarts LEO from the shell too. The only boot that stays
    # silent is a crash loop's second start within a minute.
    age = await _seconds_since_previous_boot()
    if age is not None and age <= CRASH_LOOP_FLOOR.total_seconds():
        _LOG.info("boot too close to the previous one; not announcing")
        await _take_marker()
        return None
    if payload is None:
        target = await _last_chat(app)
        if target is None:
            return None
        payload = {"channel": target[0], "subject": target[1], "at": datetime.now(timezone.utc).isoformat()}
    if not _is_fresh(payload):
        await _take_marker()
        _LOG.info("stale restart marker dropped")
        return None
    channel = str(payload.get("channel") or "")
    subject = str(payload.get("subject") or "")

    def _adapter() -> Any:
        return getattr(getattr(app, "gateway", None), "adapters", {}).get(channel)

    adapter = _adapter()
    waited = 0.0
    while adapter is None and waited < max(0.0, wait_seconds):
        await asyncio.sleep(max(0.05, poll_seconds))
        waited += poll_seconds
        adapter = _adapter()
    await _take_marker()  # one boot, one announcement
    if adapter is None or not subject:
        _LOG.info("restart announcement skipped: no adapter", extra={"channel": channel})
        return None
    try:
        await adapter.send(subject, RESTART_ANNOUNCEMENT)
    except Exception as exc:  # noqa: BLE001 - a boot must never fail on courtesy
        _LOG.warning("restart announcement failed", extra={"channel": channel, "error": str(exc)})
        return None
    _LOG.info("restart announced", extra={"channel": channel, "subject": subject})
    return RESTART_ANNOUNCEMENT
