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
    asyncio.run(db.initialize())
    yield
    db._db_path = None
    from leo.integrations.secrets import default_store

    default_store.clear()
