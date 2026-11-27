"""Reentrant transaction ownership, after-commit callbacks and event deferral."""

from __future__ import annotations

import asyncio

import pytest

from leo import db
from leo.events import EventBus


async def _kv(key: str) -> str | None:
    row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
    return str(row["value"]) if row is not None else None


async def _insert(key: str, value: str = "1") -> None:
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?)", (key, value, "now")
    )


@pytest.mark.asyncio
async def test_same_task_nested_transactions_use_savepoints() -> None:
    async with db.transaction():
        await _insert("outer")
        assert db.in_transaction() is True
        async with db.transaction():
            await _insert("inner")
            # A nested block shares the owning connection and sees its writes.
            assert await _kv("inner") == "1"
        assert await _kv("outer") == "1"
        assert await _kv("inner") == "1"
    assert await _kv("outer") == "1"
    assert await _kv("inner") == "1"


@pytest.mark.asyncio
async def test_nested_rollback_reverts_only_the_savepoint() -> None:
    with pytest.raises(RuntimeError, match="inner abort"):
        async with db.transaction():
            await _insert("outer")
            async with db.transaction():
                await _insert("inner")
                raise RuntimeError("inner abort")
    # The outer block never committed because the exception escaped it, so
    # neither row survives. The point is that the inner rollback did not abort
    # the outer connection or leave a partial state before propagating.
    assert await _kv("outer") is None
    assert await _kv("inner") is None

    async with db.transaction():
        await _insert("kept")
        with pytest.raises(RuntimeError, match="savepoint"):
            async with db.transaction():
                await _insert("dropped")
                raise RuntimeError("savepoint")
        assert await _kv("kept") == "1"
        assert await _kv("dropped") is None
    assert await _kv("kept") == "1"
    assert await _kv("dropped") is None


@pytest.mark.asyncio
async def test_after_commit_callbacks_wait_for_commit_never_for_rollback() -> None:
    calls: list[str] = []

    async def committed() -> None:
        calls.append("committed")

    async def rolled_back() -> None:
        calls.append("rolled-back")

    async with db.transaction():
        await _insert("a")
        await db.after_commit(committed)
        assert calls == []
    assert calls == ["committed"]

    with pytest.raises(RuntimeError, match="abort"):
        async with db.transaction():
            await db.after_commit(rolled_back)
            raise RuntimeError("abort")
    assert calls == ["committed"]

    # Outside a transaction the callback runs immediately.
    await db.after_commit(committed)
    assert calls == ["committed", "committed"]


@pytest.mark.asyncio
async def test_savepoint_rollback_drops_only_callbacks_registered_inside_it() -> None:
    calls: list[str] = []

    async def outer() -> None:
        calls.append("outer")

    async def inner() -> None:
        calls.append("inner")

    async with db.transaction():
        await db.after_commit(outer)
        with pytest.raises(RuntimeError, match="drop"):
            async with db.transaction():
                await db.after_commit(inner)
                raise RuntimeError("drop")
    assert calls == ["outer"]


@pytest.mark.asyncio
async def test_child_task_never_reuses_the_parent_connection() -> None:
    observed: dict[str, object] = {}
    child_ready = asyncio.Event()

    async def child() -> None:
        # A ContextVar copy would expose the parent state; ownership checking
        # must make the child see "no transaction" and open its own connection.
        observed["in_transaction"] = db.in_transaction()
        observed["row"] = await _kv("uncommitted")
        child_ready.set()

    async with db.transaction():
        await _insert("uncommitted")
        task = asyncio.create_task(child())
        await asyncio.wait_for(child_ready.wait(), timeout=2)
        # The child's independent connection cannot see the parent's write.
        assert observed["in_transaction"] is False
        assert observed["row"] is None
        await task
    assert await _kv("uncommitted") == "1"


@pytest.mark.asyncio
async def test_event_bus_publishes_after_commit_and_never_after_rollback() -> None:
    bus = EventBus()
    seen: list[int] = []

    async def handler(event) -> None:  # noqa: ANN001
        seen.append(event.payload["value"])

    bus.subscribe("tx.topic", handler)

    async with db.transaction():
        await bus.emit("tx.topic", value=1)
        assert seen == []
        # Nothing is persisted while the write transaction is open.
        assert await db.fetchone("SELECT 1 FROM events WHERE topic='tx.topic'") is None
    assert seen == [1]
    assert len(await bus.history("tx.topic")) == 1

    with pytest.raises(RuntimeError, match="rollback"):
        async with db.transaction():
            await bus.emit("tx.topic", value=2)
            raise RuntimeError("rollback")
    assert seen == [1]
    assert len(await bus.history("tx.topic")) == 1
