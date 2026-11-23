"""Shared test fixtures: every test gets an isolated SQLite database."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from leo import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch):
    db.set_db_path(tmp_path / "test.db")
    # A test process inherits the operator's environment, and the approval
    # notifier reads the operator's own target from it. Pin the target's database
    # to this test's file so a run can never announce an approval that only exists
    # in a scratch database — the operator then taps a button for a row the real
    # gateway cannot find. Explicitly clears the kill switch so a suppression test
    # has to set it itself.
    monkeypatch.setenv("LEO_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("LEO_NO_NOTIFY", raising=False)
    asyncio.run(db.initialize())
    yield
    db._db_path = None
    from leo.integrations.secrets import default_store

    default_store.clear()
