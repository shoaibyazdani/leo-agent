"""Runtime credential broker for channel enrollment.

LEO's standing rule is that credentials are referenced by environment-variable
*name*, never stored as plaintext in config. This module keeps that rule while
allowing the dashboard to enroll a channel at runtime:

- ``credential(name)`` is what adapters call. It checks the process-local
  broker first, then the environment. No adapter ever holds a secret value.
- Values submitted through the UI are held in memory only after validation and
  successful persistence. If ``LEO_CREDENTIALS_KEY`` is set, they are
  additionally written to SQLite as an authenticated, encrypted envelope
  (PBKDF2-HMAC-SHA256 → SHAKE-256 keystream → HMAC-SHA256 encrypt-then-MAC,
  stdlib only). Without that key, enrollment is session-only and says so —
  nothing is silently written in the clear.
- Losing ``LEO_CREDENTIALS_KEY`` makes stored envelopes unrecoverable: they
  fail closed instead of decrypting to garbage, and no stale plaintext from a
  previous successful load survives in the broker.
- Disabling a channel installs a durable deny marker: an environment variable
  with the same name cannot resurrect a revoked credential, even after restart.

Plaintext values are never logged, returned by an API, or included in errors.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
from typing import Any, Iterable

from .. import db
from ..contracts import utcnow
from ..logging import get_logger

_LOG = get_logger("integrations.secrets")

MASTER_KEY_ENV = "LEO_CREDENTIALS_KEY"
REVOKE_KV_KEY = "credential_revocations"
# \Z (and fullmatch) reject a trailing newline, which `$` would otherwise allow.
_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{1,63}\Z")
_MISSING = object()


class CredentialStoreError(RuntimeError):
    """Credential storage, decryption or validation failed."""


def _encrypt(name: str, value: str, master: str) -> dict[str, bytes]:
    salt = os.urandom(16)
    nonce = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", master.encode("utf-8"), salt, 210_000, dklen=32)
    data = value.encode("utf-8")
    stream = hashlib.shake_256(key + nonce).digest(len(data))
    ciphertext = bytes(a ^ b for a, b in zip(data, stream))
    mac = hmac.new(key, name.encode("utf-8") + nonce + ciphertext, hashlib.sha256).digest()
    return {"salt": salt, "nonce": nonce, "ciphertext": ciphertext, "mac": mac}


def _decrypt(name: str, envelope: dict[str, bytes], master: str) -> str:
    salt = bytes(envelope["salt"])
    nonce = bytes(envelope["nonce"])
    ciphertext = bytes(envelope["ciphertext"])
    mac = bytes(envelope["mac"])
    key = hashlib.pbkdf2_hmac("sha256", master.encode("utf-8"), salt, 210_000, dklen=32)
    expected = hmac.new(key, name.encode("utf-8") + nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise CredentialStoreError("stored credential failed authentication")
    stream = hashlib.shake_256(key + nonce).digest(len(ciphertext))
    try:
        return bytes(a ^ b for a, b in zip(ciphertext, stream)).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CredentialStoreError("stored credential is invalid") from exc


def _in_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run_sync(coro: Any) -> Any:
    """Run persistence from a synchronous caller, never from inside a loop."""
    return asyncio.run(coro)


class CredentialStore:
    """Process-local broker with optional encrypted-at-rest persistence."""

    def __init__(self) -> None:
        self._memory: dict[str, str] = {}
        self._revoked: set[str] = set()
        # Names only: never values, ciphertext or master keys.
        self.last_load_failures: list[str] = []

    @property
    def persistent(self) -> bool:
        return bool(os.getenv(MASTER_KEY_ENV))

    @staticmethod
    def validate_name(name: str) -> str:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise CredentialStoreError("invalid credential name")
        return name

    def is_revoked(self, name: str) -> bool:
        return name in self._revoked

    def _insert_sql(self) -> str:
        return (
            "INSERT INTO credentials(name,salt,nonce,ciphertext,mac,updated_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "salt=excluded.salt, nonce=excluded.nonce, ciphertext=excluded.ciphertext, "
            "mac=excluded.mac, updated_at=excluded.updated_at"
        )

    def _insert_params(self, name: str, envelope: dict[str, bytes]) -> tuple[Any, ...]:
        return (
            name, envelope["salt"], envelope["nonce"],
            envelope["ciphertext"], envelope["mac"], utcnow(),
        )

    async def _set_revocation(self, name: str, revoked: bool) -> None:
        """Persist an explicit allow/deny marker for an env-sourced name."""
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (REVOKE_KV_KEY,))
        markers: dict[str, str] = {}
        if row is not None:
            try:
                stored = json.loads(str(row["value"]))
                if isinstance(stored, dict):
                    markers = {str(key): str(value) for key, value in stored.items()}
            except (TypeError, ValueError):
                markers = {}
        if revoked:
            markers[name] = utcnow()
        else:
            markers.pop(name, None)
        payload = json.dumps(markers, separators=(",", ":"), sort_keys=True)
        await db.execute(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (REVOKE_KV_KEY, payload, utcnow()),
        )
        # Memory follows successful persistence, never precedes it.
        if revoked:
            self._revoked.add(name)
        else:
            self._revoked.discard(name)

    def set(self, name: str, value: str, *, persist: bool = True) -> str:
        """Install a value; returns ``encrypted`` or ``memory`` (mode used)."""
        name = self.validate_name(name)
        if not isinstance(value, str) or not value:
            raise CredentialStoreError("credential value must be a non-empty string")
        in_loop = _in_running_loop()
        master = os.getenv(MASTER_KEY_ENV)
        if persist and master:
            if in_loop:
                raise CredentialStoreError("use the async set_async() inside the event loop")
            envelope = _encrypt(name, value, master)
            _run_sync(db.execute(self._insert_sql(), self._insert_params(name, envelope)))
            mode = "encrypted"
        else:
            if not in_loop:
                _run_sync(self._forget_persisted_async(name))
            mode = "memory"
        if in_loop:
            self._revoked.discard(name)  # memory-only: no storage to update
        else:
            _run_sync(self._set_revocation(name, False))
        self._memory[name] = value
        return mode

    async def set_async(self, name: str, value: str, *, persist: bool = True) -> str:
        name = self.validate_name(name)
        if not isinstance(value, str) or not value:
            raise CredentialStoreError("credential value must be a non-empty string")
        master = os.getenv(MASTER_KEY_ENV)
        if persist and master:
            envelope = _encrypt(name, value, master)
            await db.execute(self._insert_sql(), self._insert_params(name, envelope))
            mode = "encrypted"
        else:
            await self._forget_persisted_async(name)
            mode = "memory"
        await self._set_revocation(name, False)
        self._memory[name] = value
        return mode

    def get(self, name: str) -> str | None:
        """Broker value first, environment second; deny markers override both."""
        if name in self._revoked:
            return None
        value = self._memory.get(name)
        return value if value is not None else os.getenv(name)

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    async def delete(self, name: str) -> bool:
        name = self.validate_name(name)
        existed = name in self._memory or bool(os.getenv(name))
        await self._forget_persisted_async(name)
        self._memory.pop(name, None)
        return existed

    async def revoke(self, name: str) -> bool:
        """Deny an env-sourced credential and remove any broker/encrypted copy."""
        name = self.validate_name(name)
        existed = name in self._memory or bool(os.getenv(name)) or name in self._revoked
        await self._forget_persisted_async(name)
        await self._set_revocation(name, True)
        self._memory.pop(name, None)
        return existed

    async def load(self, *, strict: bool = False) -> int:
        """Decrypt stored envelopes into memory at startup. Returns count.

        Decryption happens into a local dict: if any envelope fails (wrong key,
        rotation, tampering), the broker keeps *no* plaintext from this or a
        previous load and reports zero loaded credentials. ``last_load_failures``
        exposes the affected names; ``strict=True`` raises instead of returning.
        """
        self.last_load_failures = []
        revoke_row = await db.fetchone("SELECT value FROM kv WHERE key=?", (REVOKE_KV_KEY,))
        revoked: set[str] = set()
        if revoke_row is not None:
            try:
                stored = json.loads(str(revoke_row["value"]))
                if isinstance(stored, dict):
                    revoked = {str(name) for name in stored}
            except (TypeError, ValueError):
                _LOG.warning("stored credential deny markers invalid")
        self._revoked = revoked
        if not self.persistent:
            self._memory = {}
            return 0
        rows = await db.fetchall("SELECT * FROM credentials")
        loaded: dict[str, str] = {}
        failures: list[str] = []
        for row in rows:
            name = str(row["name"])
            try:
                loaded[name] = _decrypt(
                    name,
                    {
                        "salt": row["salt"], "nonce": row["nonce"],
                        "ciphertext": row["ciphertext"], "mac": row["mac"],
                    },
                    os.environ[MASTER_KEY_ENV],
                )
            except Exception as exc:  # noqa: BLE001 - never log ciphertext or values
                failures.append(name)
                _LOG.warning("stored credential could not be loaded", extra={"error": type(exc).__name__})
        if failures:
            self._memory = {}
            self.last_load_failures = sorted(failures)
            if strict:
                raise CredentialStoreError(
                    f"{len(failures)} stored credential(s) could not be decrypted"
                )
            return 0
        self._memory = loaded
        return len(loaded)

    async def stored_names(self) -> list[str]:
        rows = await db.fetchall("SELECT name FROM credentials ORDER BY name")
        return sorted(set(self._memory) | {str(row["name"]) for row in rows})

    def clear(self) -> None:
        """Drop in-memory state (tests and shutdown)."""
        self._memory.clear()
        self._revoked.clear()
        self.last_load_failures = []

    async def snapshot(self, names: Iterable[str]) -> dict[str, Any]:
        """Capture broker state so an enrollment step can be rolled back."""
        names = sorted({self.validate_name(name) for name in names})
        rows: dict[str, dict[str, Any] | None] = {}
        for name in names:
            row = await db.fetchone("SELECT * FROM credentials WHERE name=?", (name,))
            rows[name] = dict(row) if row is not None else None
        revoke_row = await db.fetchone("SELECT value,updated_at FROM kv WHERE key=?", (REVOKE_KV_KEY,))
        return {
            "names": names,
            "memory": {name: self._memory.get(name, _MISSING) for name in names},
            "rows": rows,
            "revoked": set(self._revoked),
            "revoke_row": dict(revoke_row) if revoke_row is not None else None,
        }

    async def restore(self, snapshot: dict[str, Any]) -> None:
        """Best-effort undo of a credential write (never echoes values)."""
        for name, value in snapshot["memory"].items():
            if value is _MISSING:
                self._memory.pop(name, None)
            else:
                self._memory[name] = value
        self._revoked = set(snapshot["revoked"])
        for name, row in snapshot["rows"].items():
            if row is None:
                await db.execute("DELETE FROM credentials WHERE name=?", (name,))
            else:
                await db.execute(self._insert_sql(), (
                    name, row["salt"], row["nonce"], row["ciphertext"], row["mac"],
                    row["updated_at"],
                ))
        revoke_row = snapshot["revoke_row"]
        if revoke_row is None:
            await db.execute("DELETE FROM kv WHERE key=?", (REVOKE_KV_KEY,))
        else:
            await db.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (REVOKE_KV_KEY, revoke_row["value"], revoke_row["updated_at"]),
            )

    async def _forget_persisted_async(self, name: str) -> None:
        await db.execute("DELETE FROM credentials WHERE name=?", (name,))


default_store = CredentialStore()


def credential(name: str, default: str | None = None) -> str | None:
    """Read a credential by env-var name through the process broker."""
    value = default_store.get(name)
    return value if value is not None else default


__all__ = [
    "MASTER_KEY_ENV", "CredentialStore", "CredentialStoreError",
    "credential", "default_store",
]
