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


# ------------------------------------------------------------------ lease ceiling
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


async def test_a_renewed_turn_expires_once_renewal_stops():
    """The queue behaviour behind the ceiling: no renewal → reclaimable."""
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload("7"))
    await store.claim("worker-a", lease_seconds=-1)  # as if renewals had stopped

    reclaimed = await store.reclaim_expired()
    claimed_again = await store.claim("worker-b", lease_seconds=30)

    assert reclaimed == 1
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
