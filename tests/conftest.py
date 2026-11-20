"""Shared test fixtures: every test gets an isolated SQLite database."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from leo import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path):
    db.set_db_path(tmp_path / "test.db")
    asyncio.run(db.initialize())
    yield
    db._db_path = None
    from leo.integrations.secrets import default_store

    default_store.clear()
