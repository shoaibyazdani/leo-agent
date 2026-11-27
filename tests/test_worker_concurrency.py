"""Worker concurrency and a ceiling on lease renewal.

Two failures observed on the live bot:

1. A long turn blocked every later message. One worker, one claim loop, so a
   multi-step analysis left a user's message sitting `queued` while every service
   reported healthy — from the outside, indistinguishable from the bot being down.
2. Lease renewal had no ceiling. Renewal exists so a slow turn is not mistaken for
   a dead one; without a maximum it makes a *stuck* turn immortal: it renews
   forever, never expires, never retries, and looks exactly like a busy worker.

These tests pin both, plus the property that matters when several loops share a
brain: no admission is answered twice.
"""

from __future__ import annotations

import asyncio
import weakref

import httpx
import pytest

from leo.gateway import GatewayStore
from leo.gateway.server import create_api

MASTER = "master-t0ken"


def payload(mid: str, text: str = "hello") -> dict:
    return {"text": text, "message_id": mid, "sender_id": "100000001",
            "conversation_ref": "100000001"}


class _Brain:
    """Records what it was asked and how many turns it served at once."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.asked: list[str] = []
        self.peak = 0

    async def initialize(self) -> None:
        return None

    async def ask(self, text, channel=None, subject=None, conversation_id=None):
        self.asked.append(text)
        self.peak = max(self.peak, len(self.asked) - self.peak if False else self._in_flight())
        await asyncio.sleep(self.delay)
        return {"content": f"answer:{text}", "conversation_id": "conv_x"}

    def _in_flight(self) -> int:
        return len(self.asked)


def _wire(monkeypatch, store: GatewayStore, brain, *, concurrency: int, once=True,
          lease_seconds: int = 30, turn_max_seconds: float = 600.0):
    """Run a worker whose client talks to an in-process gateway."""
    from leo.gateway import worker as worker_module

    api = create_api(store, token=MASTER, adapters={})
    transport = httpx.ASGITransport(app=api)
    original = worker_module.GatewayClient

    class _Client(original):
        def __init__(self, base_url, token, client=None):
            super().__init__(
                base_url, token,
                client=httpx.AsyncClient(transport=transport, base_url="http://gateway",
                                         headers={"Authorization": f"Bearer {token}"}),
            )

    worker_module.GatewayClient = _Client
    return worker_module, original, brain


async def _run(worker_module, brain, **kwargs) -> int:
    return await worker_module.run_worker("http://gateway", token=MASTER, brain=brain,
                                          **kwargs)


# ------------------------------------------------------------------- concurrency
async def test_one_long_turn_no_longer_blocks_the_queue(monkeypatch):
    """Two messages, two loops: both are answered without waiting for each other."""
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload("1", "first"))
    await store.admit(channel="telegram", subject="1", payload=payload("2", "second"))

    brain = _Brain(delay=0.4)
    worker_module, original, _ = _wire(monkeypatch, store, brain, concurrency=2)
    try:
        answered = await _run(worker_module, brain, wait_seconds=0.0, once=True,
                              concurrency=2)
    finally:
        worker_module.GatewayClient = original

    assert answered == 2
    assert sorted(brain.asked) == ["first", "second"]
    assert (await store.stats())["queue"] == {"done": 2}


async def test_concurrency_never_answers_one_admission_twice(monkeypatch):
    """The queue's claim is atomic, so N loops are still exactly-once per turn."""
    store = GatewayStore()
    for index in range(6):
        await store.admit(channel="telegram", subject="1",
                          payload=payload(str(index), f"m{index}"))

    brain = _Brain(delay=0.05)
    worker_module, original, _ = _wire(monkeypatch, store, brain, concurrency=4)
    try:
        # once=True: each of the 4 loops answers exactly one turn, then exits
        answered = await _run(worker_module, brain, wait_seconds=0.0, once=True,
                              concurrency=4)
    finally:
        worker_module.GatewayClient = original

    assert answered == 4, "four loops should each answer exactly one turn"
    assert len(set(brain.asked)) == len(brain.asked), "a turn was answered twice"
    assert (await store.stats())["queue"]["done"] == 4
    assert (await store.stats())["queue"].get("queued", 0) == 2


async def test_a_single_loop_still_works(monkeypatch):
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload("9", "solo"))
    brain = _Brain()
    worker_module, original, _ = _wire(monkeypatch, store, brain, concurrency=1)
    try:
        answered = await _run(worker_module, brain, wait_seconds=0.0, once=True,
                              concurrency=1)
    finally:
        worker_module.GatewayClient = original
    assert answered == 1


async def test_run_worker_clamps_concurrency(monkeypatch):
    """A direct caller cannot spawn more loops than the CLI or drain can."""
    from leo.config import MAX_TURN_CONCURRENCY
    from leo.gateway import worker as worker_module

    store = GatewayStore()
    worker_module, original, _ = _wire(monkeypatch, store, _Brain(), concurrency=1)
    started: list[str] = []

    async def fake_loop(client, app, me, **kwargs) -> int:
        started.append(me)
        return 0

    monkeypatch.setattr(worker_module, "_claim_loop", fake_loop)
    try:
        answered = await _run(worker_module, _Brain(), wait_seconds=0.0, once=True,
                              concurrency=999)
    finally:
        worker_module.GatewayClient = original

    assert answered == 0
    assert len(started) == MAX_TURN_CONCURRENCY, started
    assert len(set(started)) == MAX_TURN_CONCURRENCY, "loop identities must be unique"


# ------------------------------------------------------------------ lease ceiling
async def test_cancelling_the_lock_wait_cancels_the_keepalive(monkeypatch):
    """An admission nobody answers must not keep renewing its lease."""
    from leo.gateway import worker as worker_module

    admission = {"admission_id": "adm_wait", "channel": "telegram", "subject": "1",
                 "conversation_id": None, "payload": {"text": "hello"}}
    started = asyncio.Event()
    renewals: list[asyncio.Task] = []

    async def fake_keep_lease(*args, **kwargs) -> None:
        renewals.append(asyncio.current_task())
        started.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(worker_module, "_keep_lease", fake_keep_lease)

    class _Client:
        async def claim(self, worker, *, lease_seconds=30, wait_seconds=25):
            return dict(admission)

    held = asyncio.Lock()
    await held.acquire()  # a longer turn of this same conversation owns the lock
    locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
    locks["adm_wait"] = held  # no conversation_id → outbox id fallback
    loop = asyncio.create_task(
        worker_module._claim_loop(
            _Client(), _Brain(), "w1", wait_seconds=0.0, lease_seconds=30,
            once=True, turn_max_seconds=60.0, conversation_locks=locks,
        ),
        name="claim-under-test",
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.sleep(0)  # let the loop reach the blocked lock.acquire()
        assert not loop.done()
        loop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        held.release()

    assert len(renewals) == 1
    assert renewals[0].cancelled(), "keepalive outlived the cancelled lock wait"
    leaked = [task for task in asyncio.all_tasks()
              if task.get_name() == "leo-lease" and not task.done()]
    assert not leaked, leaked


async def test_renewal_stops_after_the_maximum_lifetime(monkeypatch):
    """A stuck turn must stop renewing, so its lease can expire and be retried."""
    from leo.gateway import worker as worker_module

    renewals: list[float] = []

    class _Client:
        async def renew(self, admission_id, worker, lease_seconds=30):
            renewals.append(lease_seconds)
            return True

    task = asyncio.create_task(
        worker_module._keep_lease(_Client(), "adm_1", "w1", 3, max_seconds=3)
    )
    await asyncio.wait_for(task, timeout=10)

    # interval is lease/3 = 1s, ceiling 3s → a couple of renewals, then stop
    assert 1 <= len(renewals) <= 3, renewals


async def test_a_turn_past_its_ceiling_is_stopped_not_redone(monkeypatch):
    """The ceiling must END the turn, not hand the same job to a second worker.

    Live symptom: a tool-heavy turn hit the 10-minute ceiling, renewals stopped,
    the lease lapsed, the queue re-claimed the admission and every attempt
    restarted the work from scratch — attempts 3, no progress, one stuck row.
    """
    store = GatewayStore()
    admission_id, _ = await store.admit(
        channel="telegram", subject="1", payload=payload("9", "long job")
    )

    class SlowBrain(_Brain):
        def __init__(self) -> None:
            super().__init__(delay=30.0)
            self.cancelled = False

        async def ask(self, text, channel=None, subject=None, conversation_id=None):
            try:
                return await super().ask(
                    text, channel=channel, subject=subject, conversation_id=conversation_id
                )
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    brain = SlowBrain()
    worker_module, original, _ = _wire(monkeypatch, store, brain, concurrency=1)
    try:
        answered = await _run(worker_module, brain, wait_seconds=0.0, once=True,
                              concurrency=1, turn_max_seconds=1.5, lease_seconds=3)
    finally:
        worker_module.GatewayClient = original

    from leo import db

    row = await store.admission(admission_id)
    queue = await db.fetchone(
        "SELECT status, attempts FROM gateway_queue WHERE admission_id=?", (admission_id,)
    )
    assert row["status"] == "done"      # terminal: neither left claimed nor requeued
    assert queue["status"] == "done"
    assert queue["attempts"] == 1       # no second attempt redoing the same work
    assert brain.cancelled is True      # the turn was actually stopped, not abandoned
    assert answered == 0
    assert (await store.stats())["queue"] == {"done": 1}


async def test_a_renewed_turn_expires_once_renewal_stops():
    """The queue behaviour behind the ceiling: no renewal → reclaimable."""
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload("7"))
    await store.claim("worker-a", lease_seconds=-1)  # as if renewals had stopped

    reclaimed = await store.reclaim_expired()
    claimed_again = await store.claim("worker-b", lease_seconds=30)

    # The ids are the contract: the caller stops one typing task per id.
    assert reclaimed == [admission_id]
    assert claimed_again is not None and claimed_again["admission_id"] == admission_id


@pytest.mark.parametrize("bad", [0, -5])
async def test_a_stuck_turn_is_visible_in_attempts(bad):
    """Attempts are counted when a lease expires, so a poison turn cannot loop."""
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload("5"))
    await store.claim("w", lease_seconds=bad)
    await store.reclaim_expired()
    claimed = await store.claim("w2", lease_seconds=30)
    row = await store.stats()
    assert claimed is not None
    assert row["queue"]["claimed"] == 1
    assert admission_id


async def test_worker_loops_keep_one_turn_per_conversation(monkeypatch):
    """More claim loops mean more chats in parallel, never two turns in one chat."""
    from leo import db

    store = GatewayStore()
    await store.save_route("telegram", "1", "conv-same")
    ids: list[str] = []
    for index, text in enumerate(("first", "second")):
        admission_id, _ = await store.admit(
            channel="telegram", subject="1",
            payload=payload(str(index), text),
        )
        ids.append(admission_id)
    for index, admission_id in enumerate(ids):
        await db.execute(
            "UPDATE gateway_queue SET created_at=? WHERE admission_id=?",
            (f"2026-01-01T00:00:0{index}+00:00", admission_id),
        )

    class OrderedBrain(_Brain):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[str] = []
            self.active = 0
            self.peak_overlap = 0

        async def ask(self, text, channel=None, subject=None, conversation_id=None):
            self.seen.append(str(text))
            self.active += 1
            self.peak_overlap = max(self.peak_overlap, self.active)
            await asyncio.sleep(0.05)
            self.active -= 1
            return {"content": f"answer:{text}", "conversation_id": conversation_id}

    brain = OrderedBrain()
    worker_module, original, _ = _wire(monkeypatch, store, brain, concurrency=2)
    try:
        answered = await _run(worker_module, brain, wait_seconds=0.0, once=True,
                              concurrency=2)
    finally:
        worker_module.GatewayClient = original

    assert answered == 2
    assert brain.seen == ["first", "second"]
    assert brain.peak_overlap == 1, "two worker loops ran turns of one chat together"


async def test_rows_without_a_route_do_not_share_one_lock(monkeypatch):
    """With no conversation_id the fallback key is the admission id, per row."""
    store = GatewayStore()
    for index, text in enumerate(("first", "second")):
        await store.admit(channel="telegram", subject=str(index),
                          payload=payload(str(index), text))

    class NoRouteBrain(_Brain):
        def __init__(self) -> None:
            super().__init__()
            self.both_started = asyncio.Event()

        async def ask(self, text, channel=None, subject=None, conversation_id=None):
            assert conversation_id is None
            self.asked.append(str(text))
            self._in_flight_count = getattr(self, "_in_flight_count", 0) + 1
            if self._in_flight_count == 2:
                self.both_started.set()
            try:
                # Deterministic overlap check: both turns must be live at once.
                await asyncio.wait_for(self.both_started.wait(), timeout=1.0)
            finally:
                self._in_flight_count -= 1
            return {"content": f"answer:{text}", "conversation_id": conversation_id}

    brain = NoRouteBrain()
    worker_module, original, _ = _wire(monkeypatch, store, brain, concurrency=2)
    try:
        answered = await _run(worker_module, brain, wait_seconds=0.0, once=True,
                              concurrency=2)
    finally:
        worker_module.GatewayClient = original

    assert answered == 2
    assert sorted(brain.asked) == ["first", "second"]
    assert brain.both_started.is_set(), "conversation-less rows serialized behind one key"
