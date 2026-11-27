"""Versioned, idempotent schedule/approval migration and fail-closed ambiguity."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest

from leo import db
from leo.config import Settings
from leo.events import EventBus
from leo.security import PermissionManagerImpl
from leo.tasks import DurableTaskEngine, Scheduler

LEGACY_SCHEMA = """
CREATE TABLE schedules (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, expression TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}', enabled INTEGER NOT NULL DEFAULT 1,
  next_run_at TEXT, last_run_at TEXT, created_at TEXT NOT NULL
);
CREATE TABLE approvals (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, resolved_at TEXT, status TEXT NOT NULL,
  actor TEXT, action TEXT NOT NULL, resource TEXT, risk TEXT NOT NULL,
  reason TEXT, requested_by TEXT, metadata TEXT
);
CREATE TABLE approval_grants (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
  action TEXT NOT NULL, resource TEXT, granted_by TEXT,
  scope TEXT NOT NULL DEFAULT 'session', metadata TEXT
);
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


async def _legacy_db(tmp_path: Path) -> Path:
    path = tmp_path / "legacy.db"
    conn = await aiosqlite.connect(path)
    try:
        await conn.executescript(LEGACY_SCHEMA)
        await conn.commit()
    finally:
        await conn.close()
    return path


async def _seed_legacy(path: Path) -> None:
    conn = await aiosqlite.connect(path)
    try:
        rows = [
            (
                "sch_interval", "interval-job", "interval", "60",
                json.dumps({"objective": "heartbeat"}), 1,
                "2026-11-26T10:00:00+00:00", None, "2026-11-20T00:00:00+00:00",
            ),
            (
                "sch_cron", "cron-job", "cron", "*/5 * * * *",
                json.dumps({"objective": "scan"}), 1,
                "2026-11-26T10:05:00+00:00", None, "2026-11-20T00:00:00+00:00",
            ),
            (
                "sch_every", "every-job", "interval", "@every 90s",
                json.dumps({"objective": "poll"}), 1,
                "2026-11-26T10:01:30+00:00", None, "2026-11-20T00:00:00+00:00",
            ),
            (
                "sch_paused", "paused-job", "interval", "300",
                json.dumps({"objective": "hold"}), 0,
                "2026-11-26T10:05:00+00:00", None, "2026-11-20T00:00:00+00:00",
            ),
            (
                "sch_once", "legacy-once", "interval", "60",
                json.dumps({"objective": "remind", "one_shot": True}), 1,
                "2026-11-26T10:00:30+00:00", None, "2026-11-20T00:00:00+00:00",
            ),
        ]
        await conn.executemany(
            "INSERT INTO schedules(id,name,kind,expression,payload,enabled,"
            "next_run_at,last_run_at,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.execute(
            "INSERT INTO approvals(id,created_at,resolved_at,status,actor,action,"
            "resource,risk,reason,requested_by,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                "apr_legacy", "2026-11-25T00:00:00+00:00", None, "pending",
                "agent-a", "tool:danger", None, "high", "needs approval", "agent-a", "{}",
            ),
        )
        await conn.execute(
            "INSERT INTO approval_grants(id,created_at,expires_at,action,resource,"
            "granted_by,scope,metadata) VALUES(?,?,?,?,?,?,?,?)",
            (
                "grn_legacy", "2026-11-20T00:00:00+00:00",
                "2035-01-01T00:00:00+00:00", "tool:read_file", "project/file.txt",
                "operator", "session", "{}",
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_legacy_rows_are_migrated_idempotently(tmp_path: Path) -> None:
    path = await _legacy_db(tmp_path)
    await _seed_legacy(path)
    db.set_db_path(path)

    await db.initialize()

    rows = {row["id"]: row for row in await db.fetchall("SELECT * FROM schedules")}
    # Interval, cron and @every definitions and cursors are untouched.
    assert rows["sch_interval"]["kind"] == "interval"
    assert rows["sch_interval"]["expression"] == "60"
    assert rows["sch_interval"]["next_run_at"] == "2026-11-26T10:00:00+00:00"
    assert rows["sch_interval"]["schedule_type"] == "interval"
    assert rows["sch_interval"]["status"] == "active"
    assert rows["sch_cron"]["expression"] == "*/5 * * * *"
    assert rows["sch_cron"]["next_run_at"] == "2026-11-26T10:05:00+00:00"
    assert rows["sch_every"]["expression"] == "@every 90s"
    assert rows["sch_every"]["schedule_type"] == "interval"
    # Legacy enabled=0 becomes a paused schedule, not an active name.
    assert rows["sch_paused"]["enabled"] == 0
    assert rows["sch_paused"]["status"] == "paused"
    # Legacy one-shot becomes the canonical once form without moving the cursor.
    once = rows["sch_once"]
    assert once["kind"] == "once"
    assert once["schedule_type"] == "once"
    assert once["run_at"] == "2026-11-26T10:00:30+00:00"
    assert once["next_run_at"] == "2026-11-26T10:00:30+00:00"
    assert once["repeat_limit"] == 1
    assert once["created_by"] == "legacy"
    assert once["updated_at"] == "2026-11-20T00:00:00+00:00"

    # Old approvals become command approvals with the legacy timeout expiry.
    approval = await db.fetchone("SELECT * FROM approvals WHERE id='apr_legacy'")
    assert approval["kind"] == "command"
    assert approval["protocol_version"] == 1
    assert approval["expires_at"] == "2026-11-25T00:15:00+00:00"

    # Old grants retain their observable authorization behavior and new columns
    # stay unset (they never become schedule create grants).
    grant = await db.fetchone("SELECT * FROM approval_grants WHERE id='grn_legacy'")
    assert grant["action"] == "tool:read_file"
    assert grant["resource"] == "project/file.txt"
    assert grant["principal"] is None and grant["revoked_at"] is None
    manager = PermissionManagerImpl(Settings(), EventBus())
    assert await manager._has_live_grant("tool:read_file", "project/file.txt") is True

    # Version marker and indexes are present.
    version = await db.fetchone(
        "SELECT value FROM kv WHERE key='schedule_schema_version'"
    )
    assert version["value"] == str(db.SCHEDULE_SCHEMA_VERSION)
    indexes = {row["name"] for row in await db.fetchall("PRAGMA index_list(schedules)")}
    assert "uq_schedules_active_name" in indexes
    assert await db.schedule_migration_blockers() == []

    # Re-running initialization is a no-op against the same durable values.
    for _ in range(3):
        await db.initialize()
    count_row = await db.fetchone("SELECT COUNT(*) AS n FROM schedules")
    assert count_row["n"] == 5
    again = await db.fetchone("SELECT * FROM schedules WHERE id='sch_once'")
    assert again["kind"] == "once"
    assert again["run_at"] == "2026-11-26T10:00:30+00:00"
    assert again["repeat_limit"] == 1
    assert again["next_run_at"] == "2026-11-26T10:00:30+00:00"
    # The schema-version marker is stable and unique.
    versions = await db.fetchall(
        "SELECT value FROM kv WHERE key='schedule_schema_version'"
    )
    assert len(versions) == 1

    # Migrated kind=once still retires through the real Scheduler.
    settings = Settings()
    events = EventBus()
    scheduler = Scheduler(DurableTaskEngine(settings, events), settings, events)
    await scheduler.trigger("legacy-once", now="2035-01-01T00:00:00+00:00")
    assert await db.fetchone("SELECT 1 FROM schedules WHERE name='legacy-once'") is None


@pytest.mark.asyncio
async def test_occurrence_key_backfill_is_idempotent_and_never_overwrites_id_keys(
    tmp_path: Path,
) -> None:
    path = await _legacy_db(tmp_path)
    conn = await aiosqlite.connect(path)
    try:
        await conn.execute(
            "INSERT INTO schedules(id,name,kind,expression,payload,enabled,"
            "next_run_at,last_run_at,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "sch_legacy", "legacy-job", "interval", "60",
                json.dumps({"objective": "work"}), 1,
                "2026-11-26T10:00:00+00:00", None, "2026-11-20T00:00:00+00:00",
            ),
        )
        await conn.executemany(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)",
            [
                (
                    "task:idempotency:schedule:legacy-job:2026-11-26T10:00:00+00:00",
                    "task_legacy", "2026-11-26T10:00:00+00:00",
                ),
                (
                    "task:idempotency:schedule:legacy-job:2026-11-26T10:05:00+00:00",
                    "task_legacy_2", "2026-11-26T10:05:00+00:00",
                ),
                # A partially migrated database already has one ID key; the
                # backfill must preserve it rather than trust the name alias.
                (
                    "task:idempotency:schedule:sch_legacy:2026-11-26T10:00:00+00:00",
                    "task_already", "2026-11-26T10:00:00+00:00",
                ),
            ],
        )
        await conn.commit()
    finally:
        await conn.close()
    db.set_db_path(path)

    await db.initialize()
    first = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        ("task:idempotency:schedule:sch_legacy:2026-11-26T10:00:00+00:00",),
    )
    second = await db.fetchone(
        "SELECT value FROM kv WHERE key=?",
        ("task:idempotency:schedule:sch_legacy:2026-11-26T10:05:00+00:00",),
    )
    assert first["value"] == "task_already"  # never overwritten
    assert second["value"] == "task_legacy_2"
    marker = await db.fetchone(
        "SELECT value FROM kv WHERE key='schedule_occurrence_key_version'"
    )
    assert marker["value"] == str(db._OCCURRENCE_KEY_VERSION)

    # Re-running initialization is a no-op: one alias per legacy key, the
    # marker stays, and the schema version is not rewritten into a competing
    # row.
    for _ in range(3):
        await db.initialize()
    aliases = await db.fetchall(
        "SELECT key,value FROM kv WHERE key LIKE "
        "'task:idempotency:schedule:sch_legacy:%' ORDER BY key"
    )
    assert len(aliases) == 2
    versions = await db.fetchall(
        "SELECT value FROM kv WHERE key='schedule_schema_version'"
    )
    assert len(versions) == 1
    assert versions[0]["value"] == str(db.SCHEDULE_SCHEMA_VERSION)
    markers = await db.fetchall(
        "SELECT value FROM kv WHERE key='schedule_occurrence_key_version'"
    )
    assert len(markers) == 1


@pytest.mark.asyncio
async def test_duplicate_active_names_block_and_report_instead_of_merging(
    tmp_path: Path,
) -> None:
    path = await _legacy_db(tmp_path)
    conn = await aiosqlite.connect(path)
    try:
        await conn.executemany(
            "INSERT INTO schedules(id,name,kind,expression,payload,enabled,"
            "next_run_at,last_run_at,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            [
                ("sch_a", "duplicate", "interval", "60", "{}", 1,
                 "2026-11-26T10:00:00+00:00", None, "2026-11-20T00:00:00+00:00"),
                ("sch_b", "duplicate", "interval", "120", "{}", 1,
                 "2026-11-26T10:00:00+00:00", None, "2026-11-20T00:00:00+00:00"),
            ],
        )
        await conn.commit()
    finally:
        await conn.close()
    db.set_db_path(path)

    with pytest.raises(db.ScheduleMigrationBlocked, match="duplicate active schedule"):
        await db.initialize()

    blockers = await db.schedule_migration_blockers()
    assert len(blockers) == 1
    assert blockers[0]["kind"] == "duplicate_active_name"
    assert blockers[0]["name"] == "duplicate"
    assert "sch_a" in blockers[0]["detail"] and "sch_b" in blockers[0]["detail"]
    # The unique index is not created while the ambiguity exists.
    indexes = {row["name"] for row in await db.fetchall("PRAGMA index_list(schedules)")}
    assert "uq_schedules_active_name" not in indexes

    # Operator remediation: remove one duplicate, then startup succeeds and the
    # durable blocker report is cleared.
    conn = await aiosqlite.connect(path)
    try:
        await conn.execute("DELETE FROM schedules WHERE id='sch_b'")
        await conn.commit()
    finally:
        await conn.close()
    await db.initialize()
    assert await db.schedule_migration_blockers() == []
    indexes = {row["name"] for row in await db.fetchall("PRAGMA index_list(schedules)")}
    assert "uq_schedules_active_name" in indexes
    assert (await db.fetchone("SELECT value FROM kv WHERE key='schedule_schema_version'"))[
        "value"
    ] == str(db.SCHEDULE_SCHEMA_VERSION)


@pytest.mark.asyncio
async def test_a_paused_duplicate_does_not_occupy_the_active_name(tmp_path: Path) -> None:
    path = await _legacy_db(tmp_path)
    conn = await aiosqlite.connect(path)
    try:
        await conn.executemany(
            "INSERT INTO schedules(id,name,kind,expression,payload,enabled,"
            "next_run_at,last_run_at,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            [
                ("sch_live", "same", "interval", "60", "{}", 1,
                 "2026-11-26T10:00:00+00:00", None, "2026-11-20T00:00:00+00:00"),
                ("sch_off", "same", "interval", "120", "{}", 0,
                 "2026-11-26T10:00:00+00:00", None, "2026-11-20T00:00:00+00:00"),
            ],
        )
        await conn.commit()
    finally:
        await conn.close()
    db.set_db_path(path)

    await db.initialize()
    rows = {
        row["id"]: row
        for row in await db.fetchall("SELECT id,status FROM schedules WHERE name='same'")
    }
    assert rows["sch_live"]["status"] == "active"
    assert rows["sch_off"]["status"] == "paused"
    assert await db.schedule_migration_blockers() == []
