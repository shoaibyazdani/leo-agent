"""Foundation tests: config, redaction, DB, events, permissions."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel
from leo.events import Event, EventBus
from leo.logging import JsonFormatter, RedactingFilter
from leo.security import PermissionManagerImpl


def test_config_defaults_and_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEO_ENVIRONMENT", "test")
    monkeypatch.setenv("LEO_CONFIG", str(tmp_path / "missing.yaml"))
    # A live provider comes from the operator's environment; mock is never
    # injected unless the operator explicitly opts in. The env shim accepts
    # only LEO's OpenCode Go inference gateway, not the IDE's endpoint.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://opencode.ai/inference/go/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("LEO_ALLOW_MOCK_FALLBACK", raising=False)
    monkeypatch.delenv("LEO_MODELS_ALLOW_MOCK_FALLBACK", raising=False)
    settings = Settings.load()
    settings.paths.data = tmp_path / "data"
    settings.ensure_dirs()
    assert settings.db_path.parent.exists()
    assert settings.environment == "test"
    assert "opencode-go" in settings.models.providers
    assert "mock" not in settings.models.providers


def test_redaction_and_json_logging() -> None:
    flt = RedactingFilter([r"api_key\s*[:=]\s*\S+"])
    record = logging.LogRecord("leo.test", logging.INFO, __file__, 1, "api_key: super-secret", (), None)
    assert flt.filter(record)
    formatted = JsonFormatter().format(record)
    assert "super-secret" not in formatted
    assert "[REDACTED]" in formatted


@pytest.mark.asyncio
async def test_db_roundtrip() -> None:
    await db.execute("INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)", ("a", "1", "now"))
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", ("a",))
    assert row and row["value"] == "1"


@pytest.mark.asyncio
async def test_event_bus_history_and_wait() -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.subscribe("test.topic", handler)
    await bus.emit("test.topic", value=42)
    assert len(seen) == 1 and seen[0].payload["value"] == 42
    assert (await bus.history("test.topic"))[0].payload["value"] == 42
    waiter = asyncio.create_task(bus.wait_for("test.topic", timeout=1))
    await asyncio.sleep(0)
    await bus.emit("test.topic", value=43)
    assert (await waiter).payload["value"] == 43


@pytest.mark.asyncio
async def test_permissions_allow_deny_approve() -> None:
    settings = Settings()
    bus = EventBus()
    manager = PermissionManagerImpl(settings, bus)
    low = await manager.authorize("test", "tool:read_file", "project/file.txt", RiskLevel.LOW)
    assert low.allowed

    escaped = await manager.authorize("test", "tool:read_file", "/etc/passwd", RiskLevel.LOW)
    assert not escaped.allowed and "outside allowed roots" in escaped.reason

    pending = await manager.authorize("test", "tool:send_message", None, RiskLevel.HIGH)
    assert pending.requires_approval and not pending.allowed and pending.approval_id
    granted = await manager.resolve_approval(pending.approval_id, True)
    assert granted.allowed
    retry = await manager.authorize(
        "test", "tool:send_message", None, RiskLevel.HIGH, approval_id=pending.approval_id
    )
    assert retry.allowed

    rows = await db.fetchall("SELECT action, decision FROM audit_log")
    assert len(rows) >= 4


# ---------------------------------------------------------------------------
# SQLite concurrency / migration regressions (agent 03 hardening)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_initialize_fresh_db(tmp_path: Path) -> None:
    """Parallel initializers of one fresh DB must all succeed and leave WAL on.

    Regression: ``executescript(SCHEMA)`` and ``PRAGMA journal_mode=WAL`` used
    to raise "database is locked" when several coroutines initialized the same
    brand-new database at once (busy_timeout does not apply to journal-mode
    switches / whole-script DDL).
    """
    path = tmp_path / "concurrent-fresh.db"
    db.set_db_path(path)
    for _ in range(5):
        results = await asyncio.gather(
            *[db.initialize() for _ in range(12)], return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        assert not failures, failures
    assert (await db.fetchone("PRAGMA journal_mode"))[0] == "wal"
    # Every table from SCHEMA exists.
    names = {r["name"] for r in await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"events", "tasks", "kv", "messages", "memory_items"} <= names


@pytest.mark.asyncio
async def test_concurrent_initialize_migrates_old_schema_once(tmp_path: Path) -> None:
    """Concurrent migration of a pre-``version`` DB adds the column exactly once.

    Regression: the PRAGMA-table_info check followed by ALTER TABLE was a
    check-then-act race between parallel initializers, producing
    "duplicate column name: version".
    """
    for trial in range(8):
        path = tmp_path / f"old-{trial}.db"
        db.set_db_path(path)
        conn = await aiosqlite.connect(path)
        await conn.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, objective TEXT NOT NULL, "
            "status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        await conn.commit()
        await conn.close()

        results = await asyncio.gather(
            *[db.initialize() for _ in range(12)], return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        assert not failures, failures

        cols = [r["name"] for r in await db.fetchall("PRAGMA table_info(tasks)")]
        assert cols.count("version") == 1


class _SetupFailingConn:
    """Minimal connection stub whose PRAGMA setup raises."""

    def __init__(self) -> None:
        self.row_factory = None
        self.closed = False

    async def execute(self, sql: str, params=None):  # noqa: ANN001, ANN201
        if "foreign_keys" in sql:
            raise sqlite3.OperationalError("setup failure")
        return None

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connections_closed_when_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection setup failures must not leak the connection/thread.

    Regression: the PRAGMA setup in ``connection()``/``transaction()`` ran
    before the ``try/finally`` that closes the connection, so a failure there
    leaked it.
    """
    created: list[_SetupFailingConn] = []

    async def fake_connect(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        conn = _SetupFailingConn()
        created.append(conn)
        return conn

    monkeypatch.setattr(db.aiosqlite, "connect", fake_connect)

    with pytest.raises(sqlite3.OperationalError):
        async with db.connection():
            pass
    assert created and created[-1].closed

    with pytest.raises(sqlite3.OperationalError):
        async with db.transaction():
            pass
    assert created and created[-1].closed


@pytest.mark.asyncio
async def test_transaction_atomic_rollback_and_foreign_keys(tmp_path: Path) -> None:
    """Writes in a transaction are atomic; FK cascades are enforced per connection."""
    db.set_db_path(tmp_path / "txn.db")
    await db.initialize()

    with pytest.raises(RuntimeError):
        async with db.transaction() as conn:
            await conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES('a','1','now')"
            )
            raise RuntimeError("abort")
    assert await db.fetchone("SELECT 1 FROM kv WHERE key='a'") is None

    async with db.transaction() as conn:
        await conn.execute(
            "INSERT INTO conversations(id,channel,subject,created_at,updated_at) "
            "VALUES('c1','cli',NULL,'now','now')"
        )
        await conn.execute(
            "INSERT INTO messages(conversation_id,role,content,created_at) "
            "VALUES('c1','user','hi','now')"
        )
    await db.execute("DELETE FROM conversations WHERE id='c1'")
    assert await db.fetchone("SELECT 1 FROM messages WHERE conversation_id='c1'") is None
    with pytest.raises(sqlite3.IntegrityError):
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,created_at) "
            "VALUES('missing','user','x','now')"
        )
