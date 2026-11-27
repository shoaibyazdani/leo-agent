"""SQLite persistence layer. One database, WAL mode, explicit transactions.

Modules own their tables but share this connection helper so that durable state,
audit trails and event history live in one inspectable file.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

import aiosqlite

_db_path: Path | None = None

_log = logging.getLogger("leo.db")

# The version recorded in ``kv`` once the schedule schema/migration has fully
# committed. It is deliberately only written after the partial unique indexes
# exist, so a blocked migration never claims success.
# Version 2 adds the schedule-ID occurrence-key backfill so a task claimed
# under the legacy name-based key is still found after the cutover.
SCHEDULE_SCHEMA_VERSION = 2
_SCHEDULE_SCHEMA_VERSION_KEY = "schedule_schema_version"
# Durable marker written only once the occurrence-key aliases are in place.
# A v1 database without this marker may still hold name-keyed claims.
_OCCURRENCE_KEY_VERSION_KEY = "schedule_occurrence_key_version"
_OCCURRENCE_KEY_VERSION = 2
_OCCURRENCE_KEY_PREFIX = "task:idempotency:schedule:"
# Matches ``SecurityConfig.approval_timeout_seconds``. Legacy approval rows had
# no expiry column; the migration gives them the timeout they were created
# under instead of inventing a new policy.
_LEGACY_APPROVAL_TIMEOUT_SECONDS = 900


@dataclass
class _TransactionState:
    """The write transaction owned by exactly one asyncio task.

    ``ContextVar`` values are inherited (copied) by child tasks. Storing the
    owning task here lets :func:`transaction` and the SQL helpers distinguish a
    genuine same-task nested call (savepoint) from a child task that merely
    inherited the context copy and must open its own connection.
    """

    conn: aiosqlite.Connection
    owner: Any
    depth: int = 0
    savepoint_seq: int = 0
    callbacks: list[Callable[[], Any]] = field(default_factory=list)


_active_transaction: ContextVar[_TransactionState | None] = ContextVar(
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
  created_by TEXT NOT NULL DEFAULT 'native',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, skill_name TEXT NOT NULL, version TEXT NOT NULL,
  action TEXT NOT NULL, detail TEXT, changed_at TEXT NOT NULL
);

-- Supporting documents for an authored skill (references/templates/scripts).
-- Kept out of the manifest JSON so a 100 KB reference cannot bloat every
-- registry reload. The manifest only carries an index of paths.
CREATE TABLE IF NOT EXISTS skill_files (
  skill_name TEXT NOT NULL, path TEXT NOT NULL, content TEXT NOT NULL,
  sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY (skill_name, path)
);

-- Usage/state telemetry for lifecycle maintenance. Native code skills are
-- seeded as ``created_by='native'`` and are never curated.
CREATE TABLE IF NOT EXISTS skill_usage (
  skill_name TEXT PRIMARY KEY,
  created_by TEXT NOT NULL DEFAULT 'native',
  views INTEGER NOT NULL DEFAULT 0,
  invocations INTEGER NOT NULL DEFAULT 0,
  failures INTEGER NOT NULL DEFAULT 0,
  patches INTEGER NOT NULL DEFAULT 0,
  last_viewed_at TEXT,
  last_used_at TEXT,
  last_patched_at TEXT,
  state TEXT NOT NULL DEFAULT 'active',
  pinned INTEGER NOT NULL DEFAULT 0,
  locked INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skill_usage_state ON skill_usage(state, updated_at);

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
  thread_id TEXT, reply_target TEXT,
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

-- The schedule table stays the one schedule authority. The legacy-compatible
-- definition is (id, name, kind, expression, payload, enabled, next_run_at,
-- last_run_at, created_at) while the Phase 1 columns are additive durable state
-- for later phases, backfilled and idempotent for legacy databases.
CREATE TABLE IF NOT EXISTS schedules (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, expression TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}', enabled INTEGER NOT NULL DEFAULT 1,
  next_run_at TEXT, last_run_at TEXT, created_at TEXT NOT NULL,
  schedule_type TEXT, run_at TEXT, timezone TEXT,
  repeat_limit INTEGER, claimed_count INTEGER NOT NULL DEFAULT 0,
  completed_count INTEGER NOT NULL DEFAULT 0,
  config_revision INTEGER NOT NULL DEFAULT 1, grace_minutes INTEGER,
  config TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'active',
  created_by TEXT, origin TEXT, consent_digest TEXT, updated_at TEXT
);

-- A model-initiated create/edit reservation. It is not a second schedule: a
-- proposal has no execution cursor and can only commit through ScheduleService.
CREATE TABLE IF NOT EXISTS schedule_proposals (
  id TEXT PRIMARY KEY, active_name TEXT, principal TEXT NOT NULL,
  operation TEXT NOT NULL, request_key TEXT, status TEXT NOT NULL DEFAULT 'pending',
  digest TEXT NOT NULL, spec TEXT NOT NULL DEFAULT '{}',
  origin TEXT, conversation_id TEXT, run_id TEXT, tool_call_id TEXT,
  config_revision INTEGER, schedule_id TEXT, error TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

-- Durable retirement/removal record so an active name may be reused by policy
-- while the removed schedule remains inspectable by ID.
CREATE TABLE IF NOT EXISTS schedule_tombstones (
  id TEXT PRIMARY KEY, schedule_id TEXT NOT NULL, name TEXT NOT NULL,
  reason TEXT NOT NULL, payload TEXT NOT NULL DEFAULT '{}',
  origin TEXT, created_by TEXT, created_at TEXT NOT NULL,
  removed_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
);

-- Fail-closed migration report. Rows here are durable and are cleared only by a
-- successful re-initialization after the operator resolves the ambiguity.
CREATE TABLE IF NOT EXISTS schedule_migration_blockers (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, name TEXT,
  detail TEXT NOT NULL, created_at TEXT NOT NULL
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
    """A serialized write transaction, reentrant only for its owning task.

    The outermost call runs ``BEGIN IMMEDIATE`` .. ``COMMIT`` on one connection
    and all statements in the block commit atomically; competing writers wait
    up to ``busy_timeout``. A nested call from the *same* asyncio task opens a
    ``SAVEPOINT`` instead of a second connection, so a failed inner block rolls
    back only its own writes. A child task never reuses the parent connection
    (which would corrupt ownership and can deadlock); it opens its own and
    waits for the writer lock.

    The :func:`execute`/:func:`fetchone`/:func:`fetchall` helpers automatically
    reuse an owned transaction connection, so writes made through them commit
    or roll back atomically with the block. :func:`after_commit` callbacks are
    awaited only after the outer ``COMMIT`` and are discarded on rollback.
    """
    own_state = _transaction_state()
    if own_state is not None:
        # Same-task nested transaction -> savepoint. Callbacks registered in a
        # nested scope are dropped if that savepoint rolls back.
        own_state.depth += 1
        savepoint = f"leo_sp_{own_state.depth}_{own_state.savepoint_seq}"
        own_state.savepoint_seq += 1
        callback_mark = len(own_state.callbacks)
        await own_state.conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield own_state.conn
        except BaseException:
            del own_state.callbacks[callback_mark:]
            try:
                await own_state.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            finally:
                await own_state.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                own_state.depth -= 1
            raise
        else:
            await own_state.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            own_state.depth -= 1
        return

    conn = await aiosqlite.connect(
        # isolation_level=None must be set at connect time: it is applied on
        # aiosqlite's worker thread, so explicit BEGIN/COMMIT control is safe.
        get_db_path(), timeout=10.0, isolation_level=None
    )
    state = _TransactionState(conn=conn, owner=asyncio.current_task())
    token = _active_transaction.set(state)
    callbacks: list[Callable[[], Any]] = []
    try:
        conn.row_factory = sqlite3.Row
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        try:
            await conn.execute("BEGIN IMMEDIATE")
            yield conn
        except BaseException:
            state.callbacks.clear()
            try:
                await conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001 - rollback is best-effort cleanup
                pass
            raise
        else:
            await conn.execute("COMMIT")
            # Capture and clear before the connection closes; callbacks run
            # after the write lock is released, never before commit.
            callbacks, state.callbacks = list(state.callbacks), []
    finally:
        _active_transaction.reset(token)
        # See connection(): never leak the connection if PRAGMA/BEGIN fails.
        await conn.close()
    await _run_callbacks(callbacks)


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


async def _table_columns(conn: aiosqlite.Connection, table: str) -> set[str]:
    rows = await (await conn.execute(f"PRAGMA table_info({table})")).fetchall()
    return {row["name"] for row in rows}


async def _add_column(
    conn: aiosqlite.Connection, table: str, column: str, declaration: str
) -> None:
    """Add one column idempotently, tolerating a concurrent initializer."""
    if column in await _table_columns(conn, table):
        return
    try:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    except sqlite3.OperationalError as exc:  # pragma: no cover - timing dependent
        # A concurrent initializer added the column between our PRAGMA check
        # and the ALTER. The schema now has it exactly once.
        if "duplicate column" not in str(exc).lower():
            raise


class ScheduleMigrationBlocked(RuntimeError):
    """A schedule migration cannot proceed without an operator decision.

    The report is persisted in ``schedule_migration_blockers`` *before* this is
    raised, so a restart still knows why startup was blocked.
    """

    def __init__(self, blockers: list[dict[str, str]]) -> None:
        self.blockers = blockers
        names = ", ".join(sorted({item.get("name", "") for item in blockers if item.get("name")}))
        super().__init__(
            "schedule migration blocked: duplicate active schedule name(s): "
            f"{names}. Rename the duplicate rows, disable all but one, or delete "
            "the extra rows, then restart. Details are recorded in the "
            "schedule_migration_blockers table."
        )


# New schedule columns. Kept explicit so a legacy database is upgraded in one
# deterministic pass and a fresh database is a no-op.
_SCHEDULE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("schedule_type", "TEXT"),
    ("run_at", "TEXT"),
    ("timezone", "TEXT"),
    ("repeat_limit", "INTEGER"),
    ("claimed_count", "INTEGER NOT NULL DEFAULT 0"),
    ("completed_count", "INTEGER NOT NULL DEFAULT 0"),
    ("config_revision", "INTEGER NOT NULL DEFAULT 1"),
    ("grace_minutes", "INTEGER"),
    ("config", "TEXT NOT NULL DEFAULT '{}'"),
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("created_by", "TEXT"),
    ("origin", "TEXT"),
    ("consent_digest", "TEXT"),
    ("updated_at", "TEXT"),
)

_APPROVAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("kind", "TEXT NOT NULL DEFAULT 'command'"),
    ("protocol_version", "INTEGER NOT NULL DEFAULT 1"),
    ("proposal_id", "TEXT"),
    ("digest", "TEXT"),
    ("choices", "TEXT NOT NULL DEFAULT '[]'"),
    ("expires_at", "TEXT"),
    ("shown_at", "TEXT"),
    ("answered_at", "TEXT"),
    ("origin", "TEXT"),
    ("conversation_id", "TEXT"),
    ("run_id", "TEXT"),
    ("tool_call_id", "TEXT"),
    ("card_revision", "INTEGER"),
    ("presented_surface", "TEXT"),
    ("deny_reason", "TEXT"),
    ("card", "TEXT"),
)

_APPROVAL_GRANT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("principal", "TEXT"),
    ("subject_type", "TEXT"),
    ("subject_id", "TEXT"),
    ("conversation_id", "TEXT"),
    ("revoked_at", "TEXT"),
)

# Trusted gateway facts that must survive admission -> claim -> worker -> reply.
# Thread and reply target are provider facts; the principal is the immutable
# sender identity. Conversation ownership is the durable route keyed by
# (channel, subject), never a request-body target.
_GATEWAY_ADMISSION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("thread_id", "TEXT"),
    ("reply_target", "TEXT"),
)

_BLOCKERS_DDL = """
CREATE TABLE IF NOT EXISTS schedule_migration_blockers (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, name TEXT,
  detail TEXT NOT NULL, created_at TEXT NOT NULL
)
""".strip()

# Partial unique indexes. They are created in ``_migrate`` (not in SCHEMA)
# because a legacy database lacks the columns they reference until the ALTERs
# above have run. "Active" is ``status='active'``; legacy ``enabled=0`` rows are
# migrated to ``paused`` so they never occupy an active name.
_SCHEDULE_INDEXES: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_schedules_active_name "
    "ON schedules(name) WHERE status='active'",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_schedule_proposals_active_name "
    "ON schedule_proposals(active_name) "
    "WHERE active_name IS NOT NULL AND status IN ('pending','approved')",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_schedule_proposals_request_key "
    "ON schedule_proposals(principal, operation, request_key) "
    "WHERE request_key IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_approval_grants_subject "
    "ON approval_grants(principal, subject_type, subject_id, action) "
    "WHERE principal IS NOT NULL AND revoked_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_schedule_tombstones_schedule "
    "ON schedule_tombstones(schedule_id)",
)


def _parse_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _payload_one_shot(payload: Any) -> bool:
    """Match the scheduler's legacy ``payload.one_shot`` interpretation."""
    if not isinstance(payload, Mapping):
        return False
    flag = payload.get("one_shot")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() in ("1", "true", "yes", "on")
    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _create_index(conn: aiosqlite.Connection, statement: str) -> None:
    try:
        await conn.execute(statement)
    except sqlite3.OperationalError as exc:  # pragma: no cover - race dependent
        if "already exists" not in str(exc).lower():
            raise


async def _schema_version(conn: aiosqlite.Connection) -> int:
    try:
        cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (_SCHEDULE_SCHEMA_VERSION_KEY,))
        row = await cursor.fetchone()
    except sqlite3.OperationalError:
        return 0
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


async def _set_schema_version(conn: aiosqlite.Connection, version: int) -> None:
    await conn.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (_SCHEDULE_SCHEMA_VERSION_KEY, str(version), _now_iso()),
    )


async def _legacy_backfill_schedules(conn: aiosqlite.Connection) -> None:
    """One-time backfill of the additive schedule columns.

    ``payload.one_shot`` becomes the canonical ``kind='once'`` form with
    ``run_at`` equal to the existing aware ``next_run_at`` and ``repeat_limit=1``.
    The cursor (``next_run_at``) is never rewritten, so a legacy one-shot still
    fires exactly when it did before.
    """
    cursor = await conn.execute(
        "SELECT id,name,kind,expression,payload,enabled,next_run_at,created_at,"
        "schedule_type,run_at,repeat_limit,status,created_by,updated_at FROM schedules"
    )
    rows = await cursor.fetchall()
    for row in rows:
        one_shot = _payload_one_shot(_parse_json(row["payload"]))
        status = "active" if bool(row["enabled"]) else "paused"
        schedule_type = "once" if one_shot else (row["schedule_type"] or row["kind"])
        created_by = row["created_by"] or "legacy"
        updated_at = row["updated_at"] or row["created_at"]
        if one_shot:
            run_at = row["run_at"] or row["next_run_at"]
            repeat_limit = (
                row["repeat_limit"] if row["repeat_limit"] is not None else 1
            )
            await conn.execute(
                "UPDATE schedules SET kind='once', schedule_type=?, status=?, "
                "created_by=?, updated_at=?, run_at=?, repeat_limit=? WHERE id=?",
                (schedule_type, status, created_by, updated_at, run_at, repeat_limit, row["id"]),
            )
        else:
            await conn.execute(
                "UPDATE schedules SET schedule_type=?, status=?, created_by=?, "
                "updated_at=? WHERE id=?",
                (schedule_type, status, created_by, updated_at, row["id"]),
            )


async def _backfill_occurrence_keys(conn: aiosqlite.Connection) -> None:
    """Alias legacy name-keyed occurrence claims to their schedule IDs.

    Older schedulers deduplicated an occurrence on
    ``task:idempotency:schedule:<name>:<instant>``. Phase 7 moves the canonical
    key to ``schedule:<id>:<instant>``. Without an alias an occurrence that was
    claimed under the old key but whose cursor was not advanced (a crash window)
    would create a second task after the upgrade. This pass inserts the ID-keyed
    alias for every still-known schedule name and records a durable marker so
    the rewrite runs once and is idempotent on every later startup.
    """
    cursor = await conn.execute(
        "SELECT id,name,status,created_at FROM schedules "
        "ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, created_at"
    )
    rows = await cursor.fetchall()
    name_to_id: dict[str, str] = {}
    for row in rows:
        # The pinning order above means an active row wins over a paused
        # duplicate; a genuinely ambiguous duplicate still blocks startup.
        name_to_id.setdefault(str(row["name"]), str(row["id"]))

    if name_to_id:
        ordered_names = sorted(name_to_id, key=len, reverse=True)
        claimed = await (
            await conn.execute(
                "SELECT key,value FROM kv WHERE key LIKE ?",
                (f"{_OCCURRENCE_KEY_PREFIX}%",),
            )
        ).fetchall()
        for item in claimed:
            key = str(item["key"])
            rest = key[len(_OCCURRENCE_KEY_PREFIX):]
            for name in ordered_names:
                marker = f"{name}:"
                if not rest.startswith(marker):
                    continue
                instant = rest[len(marker):]
                if not instant:
                    break
                alias = f"{_OCCURRENCE_KEY_PREFIX}{name_to_id[name]}:{instant}"
                if alias != key:
                    await conn.execute(
                        "INSERT OR IGNORE INTO kv(key,value,updated_at) VALUES(?,?,?)",
                        (alias, str(item["value"]), _now_iso()),
                    )
                break

    await conn.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (_OCCURRENCE_KEY_VERSION_KEY, str(_OCCURRENCE_KEY_VERSION), _now_iso()),
    )


async def _legacy_backfill_approvals(conn: aiosqlite.Connection) -> None:
    cursor = await conn.execute(
        "SELECT id,created_at,expires_at FROM approvals WHERE expires_at IS NULL"
    )
    rows = await cursor.fetchall()
    for row in rows:
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            expires = (created + timedelta(seconds=_LEGACY_APPROVAL_TIMEOUT_SECONDS)).isoformat()
        except (TypeError, ValueError):
            expires = row["created_at"]
        await conn.execute(
            "UPDATE approvals SET kind=COALESCE(NULLIF(kind,''),'command'), "
            "protocol_version=COALESCE(protocol_version,1), expires_at=? WHERE id=?",
            (expires, row["id"]),
        )
    # Columns added with defaults already populate legacy rows; this is a
    # defensive normalization for databases that had a partial migration.
    await conn.execute(
        "UPDATE approvals SET kind='command' WHERE kind IS NULL OR kind=''"
    )
    await conn.execute(
        "UPDATE approvals SET protocol_version=1 WHERE protocol_version IS NULL"
    )


async def _duplicate_active_names(conn: aiosqlite.Connection) -> list[dict[str, str]]:
    cursor = await conn.execute(
        "SELECT name, COUNT(*) AS n, GROUP_CONCAT(id) AS ids FROM schedules "
        "WHERE status='active' GROUP BY name HAVING COUNT(*) > 1 ORDER BY name"
    )
    rows = await cursor.fetchall()
    return [
        {
            "kind": "duplicate_active_name",
            "name": str(row["name"]),
            "detail": (
                f"{int(row['n'])} active schedules share this name "
                f"(ids: {row['ids']}); a unique active name is required"
            ),
        }
        for row in rows
    ]


async def _record_blockers(blockers: list[dict[str, str]]) -> None:
    """Persist the blocker report on a fresh connection (survives rollback)."""
    async with connection() as conn:
        await conn.execute(_BLOCKERS_DDL)
        await conn.execute("DELETE FROM schedule_migration_blockers")
        for item in blockers:
            await conn.execute(
                "INSERT INTO schedule_migration_blockers(kind,name,detail,created_at) "
                "VALUES(?,?,?,?)",
                (item["kind"], item.get("name"), item["detail"], _now_iso()),
            )


async def _migrate(conn: aiosqlite.Connection) -> None:
    """Idempotent in-place migrations for databases created before a change."""
    # Compare-and-swap guard for task rows (durable engine lost-update defence).
    await _add_column(conn, "tasks", "version", "INTEGER NOT NULL DEFAULT 0")
    # Provenance of the manifest itself; older rows predate authored skills
    # and default to ``native`` so no existing row changes meaning.
    await _add_column(conn, "skills", "created_by", "TEXT NOT NULL DEFAULT 'native'")

    # Additive schedule/approval/consent domain columns.
    for column, declaration in _SCHEDULE_COLUMNS:
        await _add_column(conn, "schedules", column, declaration)
    for column, declaration in _APPROVAL_COLUMNS:
        await _add_column(conn, "approvals", column, declaration)
    for column, declaration in _APPROVAL_GRANT_COLUMNS:
        await _add_column(conn, "approval_grants", column, declaration)
    for column, declaration in _GATEWAY_ADMISSION_COLUMNS:
        await _add_column(conn, "gateway_admissions", column, declaration)

    version = await _schema_version(conn)
    if version < SCHEDULE_SCHEMA_VERSION:
        await _legacy_backfill_schedules(conn)
        await _legacy_backfill_approvals(conn)
        await _backfill_occurrence_keys(conn)

    # Fail closed before creating the unique index: a partial unique index
    # cannot represent the ambiguity of two active rows with the same name.
    blockers = await _duplicate_active_names(conn)
    if blockers:
        raise ScheduleMigrationBlocked(blockers)

    try:
        for statement in _SCHEDULE_INDEXES:
            await _create_index(conn, statement)
    except sqlite3.IntegrityError:
        # A writer inserted a conflicting active name between the check above
        # and index creation. Re-read and fail closed with the same report.
        blockers = await _duplicate_active_names(conn)
        if blockers:
            raise ScheduleMigrationBlocked(blockers) from None
        raise

    # A successful pass means the operator resolved any previous blocker.
    try:
        await conn.execute("DELETE FROM schedule_migration_blockers")
    except sqlite3.OperationalError:
        pass
    if version < SCHEDULE_SCHEMA_VERSION:
        await _set_schema_version(conn, SCHEDULE_SCHEMA_VERSION)


async def initialize(path: str | Path | None = None) -> Path:
    if path is not None:
        set_db_path(path)
    target = get_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with connection() as conn:
            await _enable_wal(conn)
            for statement in _SCHEMA_STATEMENTS:
                await conn.execute(statement)
            await _migrate(conn)
    except ScheduleMigrationBlocked as exc:
        await _record_blockers(exc.blockers)
        raise
    return target


async def schedule_migration_blockers() -> list[dict[str, Any]]:
    """Return the durable migration blocker report (empty when healthy)."""
    try:
        rows = await fetchall(
            "SELECT kind,name,detail,created_at FROM schedule_migration_blockers ORDER BY id"
        )
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


def _transaction_connection() -> aiosqlite.Connection | None:
    state = _transaction_state()
    return state.conn if state is not None else None


def _transaction_state() -> _TransactionState | None:
    """The active transaction only when the *current task* owns it."""
    state = _active_transaction.get()
    if state is None:
        return None
    if state.owner is not asyncio.current_task():
        return None
    return state


def in_transaction() -> bool:
    """Whether the current task is inside an owned write transaction."""
    return _transaction_state() is not None


async def after_commit(callback: Callable[[], Any]) -> None:
    """Run ``callback`` after the current task's transaction commits.

    Inside a transaction the callback is deferred; a rollback (including a
    rolled-back nested savepoint) drops it. Outside a transaction it runs
    immediately. Callbacks are only awaited after ``COMMIT`` returns, so no
    subscriber can observe uncommitted state or fire after a rollback.
    """
    state = _transaction_state()
    if state is None:
        await _invoke_callback(callback)
        return
    state.callbacks.append(callback)


async def _invoke_callback(callback: Callable[[], Any]) -> None:
    try:
        result = callback()
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - one listener must not undo a commit
        _log.exception("after-commit callback failed")


async def _run_callbacks(callbacks: list[Callable[[], Any]]) -> None:
    for callback in callbacks:
        await _invoke_callback(callback)


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
