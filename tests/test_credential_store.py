"""Credential broker: memory fallback, encrypted persistence, fail-closed decrypt."""

from __future__ import annotations

import pytest

from leo import db
from leo.integrations.secrets import (
    MASTER_KEY_ENV,
    CredentialStore,
    CredentialStoreError,
    credential,
    default_store,
)


async def test_memory_mode_writes_no_plaintext_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MASTER_KEY_ENV, raising=False)
    store = CredentialStore()
    assert store.set("LEO_TEST_TOKEN", "plain-fake-value") == "memory"
    assert store.get("LEO_TEST_TOKEN") == "plain-fake-value"
    rows = await db.fetchall("SELECT * FROM credentials")
    assert rows == []


async def test_encrypted_persistence_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "correct-horse-battery-staple")
    writer = CredentialStore()
    assert await writer.set_async("TELEGRAM_BOT_TOKEN", "super-secret-token-123") == "encrypted"

    rows = await db.fetchall("SELECT * FROM credentials")
    assert len(rows) == 1
    assert "super-secret-token-123" not in str(dict(rows[0]))

    reader = CredentialStore()
    assert await reader.load() == 1
    assert reader.get("TELEGRAM_BOT_TOKEN") == "super-secret-token-123"


async def test_wrong_key_and_tampering_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "right-key")
    await CredentialStore().set_async("SLACK_BOT_TOKEN", "top-secret-value")
    monkeypatch.setenv(MASTER_KEY_ENV, "wrong-key")
    reader = CredentialStore()
    assert await reader.load() == 0
    assert reader.get("SLACK_BOT_TOKEN") is None

    monkeypatch.setenv(MASTER_KEY_ENV, "right-key")
    await db.execute("UPDATE credentials SET mac=? WHERE name=?", (b"\x00" * 32, "SLACK_BOT_TOKEN"))
    tampered = CredentialStore()
    assert await tampered.load() == 0
    assert tampered.get("SLACK_BOT_TOKEN") is None


async def test_delete_and_environment_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEO_FALLBACK_TOKEN", "env-value")
    store = CredentialStore()
    assert store.get("LEO_FALLBACK_TOKEN") == "env-value"
    store._memory["LEO_FALLBACK_TOKEN"] = "broker-value"
    assert store.get("LEO_FALLBACK_TOKEN") == "broker-value"
    await store.delete("LEO_FALLBACK_TOKEN")
    assert store.get("LEO_FALLBACK_TOKEN") == "env-value"  # delete broker only; env remains


def test_name_validation_and_default_credential_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(CredentialStoreError, match="invalid credential name"):
        CredentialStore().set("lower-case", "x")
    with pytest.raises(CredentialStoreError, match="non-empty"):
        CredentialStore().set("GOOD_NAME", "")
    monkeypatch.delenv("LEO_MISSING_TOKEN", raising=False)
    assert credential("LEO_MISSING_TOKEN") is None
    assert credential("LEO_MISSING_TOKEN", "fallback") == "fallback"
    default_store.clear()


@pytest.mark.parametrize("bad", ["LEO_OK\n", "LEO_TOKEN\r", "LEO\x00TOKEN", "LEO_TOKEN\nX"])
def test_credential_names_reject_control_suffixes(bad: str) -> None:
    with pytest.raises(CredentialStoreError, match="invalid credential name"):
        CredentialStore.validate_name(bad)
    store = CredentialStore()
    with pytest.raises(CredentialStoreError, match="invalid credential name"):
        store.set(bad, "fake-value")


async def test_failed_rotation_clears_memory_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "first-key")
    store = CredentialStore()
    await store.set_async("LEO_ROTATE_TOKEN", "fake-rotation-value")
    assert store.get("LEO_ROTATE_TOKEN") == "fake-rotation-value"

    monkeypatch.setenv(MASTER_KEY_ENV, "rotated-key")
    assert await store.load() == 0
    assert store.get("LEO_ROTATE_TOKEN") is None
    assert store.last_load_failures == ["LEO_ROTATE_TOKEN"]
    with pytest.raises(CredentialStoreError, match="could not be decrypted"):
        await store.load(strict=True)


async def test_set_async_leaves_memory_untouched_on_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "write-key")
    store = CredentialStore()
    real_execute = db.execute

    async def failing_write(sql: str, params: tuple = ()) -> object:
        if sql.startswith("INSERT INTO credentials"):
            raise RuntimeError("encrypted store unavailable")
        return await real_execute(sql, params)

    monkeypatch.setattr(db, "execute", failing_write)
    with pytest.raises(RuntimeError, match="encrypted store unavailable"):
        await store.set_async("LEO_WRITE_TOKEN", "fake-write-value")
    assert "LEO_WRITE_TOKEN" not in store._memory
    assert store.get("LEO_WRITE_TOKEN") is None


def test_sync_set_leaves_memory_untouched_on_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "write-key")

    async def failing_write(sql: str, params: tuple = ()) -> object:
        if sql.startswith("INSERT INTO credentials"):
            raise RuntimeError("encrypted store unavailable")
        raise AssertionError("unexpected db call")

    monkeypatch.setattr(db, "execute", failing_write)
    store = CredentialStore()
    with pytest.raises(RuntimeError, match="encrypted store unavailable"):
        store.set("LEO_SYNC_TOKEN", "fake-sync-value")
    assert "LEO_SYNC_TOKEN" not in store._memory


async def test_revoke_denies_env_value_across_restart_until_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEO_ENV_TOKEN", "fake-env-value")
    monkeypatch.setenv(MASTER_KEY_ENV, "persist-key")
    store = CredentialStore()
    assert store.get("LEO_ENV_TOKEN") == "fake-env-value"
    assert await store.revoke("LEO_ENV_TOKEN") is True
    assert store.get("LEO_ENV_TOKEN") is None

    restarted = CredentialStore()
    await restarted.load()
    assert restarted.is_revoked("LEO_ENV_TOKEN")
    assert restarted.get("LEO_ENV_TOKEN") is None

    await restarted.set_async("LEO_ENV_TOKEN", "fake-new-value")
    assert restarted.get("LEO_ENV_TOKEN") == "fake-new-value"
    third = CredentialStore()
    await third.load()
    assert not third.is_revoked("LEO_ENV_TOKEN")
