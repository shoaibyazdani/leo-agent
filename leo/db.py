"""SQLite persistence layer. One database, WAL mode, explicit transactions.

Modules own their tables but share this connection helper so that durable state,
audit trails and event history live in one inspectable file.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import aiosqlite

_db_path: Path | None = None

# A task-local handle to the open write transaction, if any. Helpers below use
# it so a read-modify-write sequence can include ordinary ``db.execute`` calls
# (audits, events) in the SAME transaction instead of opening a second, blocked
# connection. ContextVar keeps concurrent asyncio tasks isolated.
_active_transaction: ContextVar[aiosqlite.Connection | None] = ContextVar(
    "leo_active_transaction", default=None
)

SCHEMA = """
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, topic TEXT NOT NULL, source TEXT NOT NULL,
  payload TEXT NOT NULL, correlation_id TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic, created_at);

CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL,
  action TEXT NOT NULL, resource TEXT, risk TEXT, decision TEXT, reason TEXT,
  metadata TEXT, correlation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);

CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, resolved_at TEXT, status TEXT NOT NULL,
  actor TEXT, action TEXT NOT NULL, resource TEXT, risk TEXT NOT NULL,
  reason TEXT, requested_by TEXT, metadata TEXT
);

CREATE TABLE IF NOT EXISTS turn_cancels (
  conversation_id TEXT PRIMARY KEY, requested_at TEXT NOT NULL, requested_by TEXT
);

CREATE TABLE IF NOT EXISTS approval_grants (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
  action TEXT NOT NULL, resource TEXT, granted_by TEXT, scope TEXT NOT NULL DEFAULT 'session',
  metadata TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, objective TEXT NOT NULL, status TEXT NOT NULL,
  plan TEXT NOT NULL DEFAULT '[]', current_step INTEGER NOT NULL DEFAULT 0,
  history TEXT NOT NULL DEFAULT '[]', dependencies TEXT NOT NULL DEFAULT '[]',
  checkpoints TEXT NOT NULL DEFAULT '[]', result TEXT, errors TEXT NOT NULL DEFAULT '[]',
  retry_count INTEGER NOT NULL DEFAULT 0, resource_limits TEXT NOT NULL DEFAULT '{}',
  cancellation_requested INTEGER NOT NULL DEFAULT 0, owner_agent TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT,
  evidence TEXT NOT NULL DEFAULT '[]', metadata TEXT NOT NULL DEFAULT '{}',
  version INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, created_at);

CREATE TABLE IF NOT EXISTS attachments (
  id TEXT PRIMARY KEY, conversation_id TEXT, name TEXT NOT NULL, mime TEXT NOT NULL,
  size INTEGER NOT NULL, sha256 TEXT NOT NULL, path TEXT NOT NULL,
  text_preview TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attachments_conversation ON attachments(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS task_checkpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, seq INTEGER NOT NULL,
  state TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_items (
  id TEXT PRIMARY KEY, category TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}', source TEXT, confidence REAL NOT NULL DEFAULT 0.5,
  tags TEXT NOT NULL DEFAULT '[]', version INTEGER NOT NULL DEFAULT 1,
  embedding TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  expires_at TEXT, supersedes TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_category ON memory_items(category);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_items(kind);

CREATE TABLE IF NOT EXISTS memory_links (
  source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0, PRIMARY KEY(source, target, relation)
);

CREATE TABLE IF NOT EXISTS memory_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL, version INTEGER NOT NULL,
  content TEXT NOT NULL, metadata TEXT NOT NULL, changed_at TEXT NOT NULL, reason TEXT
);

CREATE TABLE IF NOT EXISTS model_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, provider TEXT NOT NULL,
  model TEXT NOT NULL, task_id TEXT, agent_id TEXT, input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0, latency_ms REAL NOT NULL DEFAULT 0,
  cost REAL NOT NULL DEFAULT 0, success INTEGER NOT NULL DEFAULT 1, metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON model_usage(ts);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, channel TEXT NOT NULL, subject TEXT, created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
-- SQLite does not auto-index foreign keys. The sidebar's per-conversation
-- message counts and previews would otherwise scan the whole messages table.
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL,
  capabilities TEXT NOT NULL DEFAULT '[]', tool_permissions TEXT NOT NULL DEFAULT '[]',
  model_preferences TEXT NOT NULL DEFAULT '[]', context_budget INTEGER NOT NULL DEFAULT 16000,
  task_limit INTEGER NOT NULL DEFAULT 4, enabled INTEGER NOT NULL DEFAULT 1,
  metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS agent_runs (
  id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, task_id TEXT, status TEXT NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT, result TEXT, evaluations TEXT,
  metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS skills (
  name TEXT PRIMARY KEY, description TEXT NOT NULL, version TEXT NOT NULL,
  author TEXT, category TEXT, permissions TEXT NOT NULL DEFAULT '[]',
  input_schema TEXT NOT NULL DEFAULT '{}', output_schema TEXT NOT NULL DEFAULT '{}',
  entrypoint TEXT, status TEXT NOT NULL DEFAULT 'registered', health TEXT NOT NULL DEFAULT 'unknown',
  source TEXT NOT NULL DEFAULT 'native', manifest TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, skill_name TEXT NOT NULL, version TEXT NOT NULL,
  action TEXT NOT NULL, detail TEXT, changed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id TEXT PRIMARY KEY, tool TEXT NOT NULL, actor TEXT, args TEXT,
  risk TEXT, decision TEXT, success INTEGER, duration_ms REAL,
  error TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_created ON tool_calls(created_at);

CREATE TABLE IF NOT EXISTS transport_cursors (
  channel TEXT NOT NULL, account TEXT NOT NULL DEFAULT '',
  value TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(channel, account)
);

CREATE TABLE IF NOT EXISTS transport_heartbeats (
  channel TEXT NOT NULL, account TEXT NOT NULL DEFAULT '',
  pid INTEGER, note TEXT NOT NULL DEFAULT '', beat_at TEXT NOT NULL,
  PRIMARY KEY(channel, account)
);

CREATE TABLE IF NOT EXISTS gateway_admissions (
  admission_id TEXT PRIMARY KEY, request_id TEXT, channel TEXT NOT NULL,
  subject TEXT NOT NULL, principal TEXT NOT NULL DEFAULT '',
  payload TEXT NOT NULL, payload_digest TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(channel, payload_digest)
);

CREATE TABLE IF NOT EXISTS gateway_queue (
  admission_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'queued',
  attempts INTEGER NOT NULL DEFAULT 0, claimed_by TEXT, claimed_at TEXT,
  claim_expires_at TEXT, last_error TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turn_leases (
  conversation_id TEXT PRIMARY KEY, holder TEXT NOT NULL,
  acquired_at TEXT NOT NULL, expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gateway_routes (
  channel TEXT NOT NULL, subject TEXT NOT NULL, conversation_id TEXT NOT NULL,
  updated_at TEXT NOT NULL, PRIMARY KEY(channel, subject)
);

CREATE TABLE IF NOT EXISTS gateway_attach_tickets (
  ticket TEXT PRIMARY KEY, created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL, used_at TEXT
);

CREATE TABLE IF NOT EXISTS gateway_sessions (
  session_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
  last_seen_at TEXT, revoked_at TEXT
);

-- Durable, actor-scoped tool-call quotas, one row per bucket/window. The
-- executor increments it inside BEGIN IMMEDIATE so the check and charge are
-- atomic across processes and survive restarts.
CREATE TABLE IF NOT EXISTS tool_quotas (
  bucket TEXT NOT NULL, window_start REAL NOT NULL, count INTEGER NOT NULL,
  PRIMARY KEY(bucket, window_start)
);

CREATE TABLE IF NOT EXISTS schedules (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, expression TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}', enabled INTEGER NOT NULL DEFAULT 1,
  next_run_at TEXT, last_run_at TEXT, created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS integrations (
  name TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
  config_ref TEXT, capabilities TEXT NOT NULL DEFAULT '[]',
  verification TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

-- Runtime channel credentials: encrypted envelopes, never plaintext.
CREATE TABLE IF NOT EXISTS credentials (
  name TEXT PRIMARY KEY, salt BLOB NOT NULL, nonce BLOB NOT NULL,
  ciphertext BLOB NOT NULL, mac BLOB NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, name TEXT NOT NULL,
  value REAL NOT NULL, labels TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, ts);

CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
);
"""

# Execute schema statements one at a time instead of via ``executescript``.
# A script is run as one C-level call whose schema changes can raise
# "database is locked" under concurrent initializers without the busy handler
# getting a chance to wait between statements. Individual, idempotent
# CREATE ... IF NOT EXISTS statements serialize cleanly via busy_timeout.
_SCHEMA_STATEMENTS = tuple(s.strip() for s in SCHEMA.split(";") if s.strip())


def set_db_path(path: str | Path) -> None:
    global _db_path
    _db_path = Path(path).expanduser().resolve()
    _db_path.parent.mkdir(parents=True, exist_ok=True)


def get_db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("database not initialized: call db.set_db_path() first")
    return _db_path


def database_path_configured() -> bool:
    """Whether a host (tests, embedding apps) already chose the DB path."""
    return _db_path is not None


@asynccontextmanager
async def connection() -> AsyncIterator[aiosqlite.Connection]:
    conn = await aiosqlite.connect(get_db_path(), timeout=10.0)
    try:
        conn.row_factory = sqlite3.Row
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    finally:
        # Always close, even when PRAGMA setup raises before the inner try is
        # entered; otherwise a locked/erroring DB leaks connections (and their
        # aiosqlite worker threads) under stress.
        await conn.close()


@asynccontextmanager
async def transaction() -> AsyncIterator[aiosqlite.Connection]:
    """A single serialized write transaction: ``BEGIN IMMEDIATE`` .. ``COMMIT``.

    All statements in the block share one connection and commit atomically;
    competing writers wait up to ``busy_timeout``. Use this for any
    read-modify-write sequence whose intermediate state must never be
    observable (claims, dedupe-and-insert pairs, approval + its audit row).
    The :func:`execute`/:func:`fetchone`/:func:`fetchall` helpers automatically
    reuse this connection inside the block, so writes made through them commit
    or roll back atomically with the transaction body. Event bus persistence
    is best-effort and stays outside the block by design.
    """
    conn = await aiosqlite.connect(
        # isolation_level=None must be set at connect time: it is applied on
        # aiosqlite's worker thread, so explicit BEGIN/COMMIT control is safe.
        get_db_path(), timeout=10.0, isolation_level=None
    )
    token = _active_transaction.set(conn)
    try:
        conn.row_factory = sqlite3.Row
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        try:
            await conn.execute("BEGIN IMMEDIATE")
            yield conn
            await conn.execute("COMMIT")
        except BaseException:
            try:
                await conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001 - rollback is best-effort cleanup
                pass
            raise
    finally:
        _active_transaction.reset(token)
        # See connection(): never leak the connection if PRAGMA/BEGIN fails.
        await conn.close()


async def _enable_wal(conn: aiosqlite.Connection) -> None:
    """Best-effort switch to WAL mode.

    Changing journal mode takes a brief exclusive lock and, unlike ordinary
    writes, ignores ``busy_timeout``. When several workers initialize the same
    fresh database at once, one wins and the others get "database is locked";
    that is harmless because the winner leaves the file in WAL mode. Keep this
    out of ``SCHEMA``/``executescript`` so a lock race cannot abort schema
    creation.
    """
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:  # pragma: no cover - timing dependent
        message = str(exc).lower()
        if "locked" not in message and "busy" not in message:
            raise


async def _migrate(conn: aiosqlite.Connection) -> None:
    """Idempotent in-place migrations for databases created before a change."""
    columns = {
        row["name"]
        for row in await (await conn.execute("PRAGMA table_info(tasks)")).fetchall()
    }
    if "version" not in columns:
        # Compare-and-swap guard for task rows (durable engine lost-update defence).
        try:
            await conn.execute("ALTER TABLE tasks ADD COLUMN version INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError as exc:  # pragma: no cover - timing dependent
            # A concurrent initializer added the column between our PRAGMA
            # check and the ALTER. The schema now has it exactly once.
            if "duplicate column" not in str(exc).lower():
                raise


async def initialize(path: str | Path | None = None) -> Path:
    if path is not None:
        set_db_path(path)
    target = get_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    async with connection() as conn:
        await _enable_wal(conn)
        for statement in _SCHEMA_STATEMENTS:
            await conn.execute(statement)
        await _migrate(conn)
    return target


def _transaction_connection() -> aiosqlite.Connection | None:
    return _active_transaction.get()


async def execute(sql: str, params: Sequence[Any] = ()) -> aiosqlite.Cursor:
    conn = _transaction_connection()
    if conn is not None:
        return await conn.execute(sql, tuple(params))
    async with connection() as own_conn:
        return await own_conn.execute(sql, tuple(params))


async def executemany(sql: str, params: Sequence[Sequence[Any]]) -> None:
    conn = _transaction_connection()
    if conn is not None:
        await conn.executemany(sql, [tuple(p) for p in params])
        return
    async with connection() as own_conn:
        await own_conn.executemany(sql, [tuple(p) for p in params])


async def fetchone(sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    conn = _transaction_connection()
    if conn is not None:
        cursor = await conn.execute(sql, tuple(params))
        return await cursor.fetchone()
    async with connection() as own_conn:
        cursor = await own_conn.execute(sql, tuple(params))
        return await cursor.fetchone()


async def fetchall(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    conn = _transaction_connection()
    if conn is not None:
        cursor = await conn.execute(sql, tuple(params))
        return list(await cursor.fetchall())
    async with connection() as own_conn:
        cursor = await own_conn.execute(sql, tuple(params))
        return list(await cursor.fetchall())
