"""Shared test fixtures: every test gets an isolated SQLite database."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from leo import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch):
    db.set_db_path(tmp_path / "test.db")
    # A test process inherits the operator's environment, including the real
    # Telegram token and notify target when the suite is run beside a live
    # service. Pin the database and turn the notifier off for the whole suite:
    # otherwise a test that builds the real app announces approvals that only
    # exist in its scratch database, the operator taps an empty card, and the
    # gateway answers "cannot resolve" because the row was never in its database.
    # The notifier's own tests opt back in explicitly.
    monkeypatch.setenv("LEO_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LEO_NO_NOTIFY", "1")
    # Channel credentials are the other half of the inherited environment:
    # a live Telegram token in the operator's shell makes "offline" tests see
    # telegram as configured instead of blocked, and makes the base-adapter
    # typing test reach a token path it never should. Strip every adapter's
    # credential/verification variable so tests start from a clean slate;
    # tests that need a configured channel set their own values explicitly.
    from leo.integrations.channels import ALL_ADAPTERS

    for adapter_cls in ALL_ADAPTERS.values():
        for name in adapter_cls.credential_env + adapter_cls.verification_env:
            monkeypatch.delenv(name, raising=False)
    asyncio.run(db.initialize())
    yield
    db._db_path = None
    from leo.integrations.secrets import default_store

    default_store.clear()
