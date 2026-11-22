"""The gateway split: durable queue between transport and brain.

Transport and brain are separate processes, so the queue between them is the only
thing holding a user's message. These tests pin the guarantees that make the split
worth having: an admission is recorded once, a claim is a lease that comes back
when a worker dies, a poison payload fails loudly instead of looping, and a
completed turn is terminal.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from leo.gateway import GatewayStore
from leo.gateway.server import create_api


def payload(text: str = "hello", mid: str = "1") -> dict:
    return {"text": text, "message_id": mid, "sender_id": "100000001",
            "conversation_ref": "100000001"}


async def test_admission_is_recorded_once():
    store = GatewayStore()

    first, created = await store.admit(channel="telegram", subject="1", payload=payload())
    again, created_again = await store.admit(channel="telegram", subject="1", payload=payload())

    assert created is True
    assert created_again is False
    assert first == again            # same admission, not a second one
    assert (await store.stats())["queue"] == {"queued": 1}


async def test_claim_is_a_lease_and_a_second_claim_gets_nothing():
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload())

    claimed = await store.claim("worker-a", lease_seconds=60)

    assert claimed is not None
    assert claimed["claimed_by"] == "worker-a"
    assert await store.claim("worker-b", lease_seconds=60) is None


async def test_an_expired_lease_comes_back_to_the_queue():
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("dead-worker", lease_seconds=-1)  # already expired

    reclaimed = await store.reclaim_expired()
    claimed_again = await store.claim("worker-b", lease_seconds=60)

    # The ids matter, not the count: the caller stops one typing task per id.
    assert len(reclaimed) == 1 and reclaimed[0].startswith("adm_")
    assert claimed_again is not None
    assert claimed_again["claimed_by"] == "worker-b"


async def test_failure_requeues_then_buries_after_max_attempts():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())

    await store.claim("worker", lease_seconds=60)
    queued_again = await store.fail(admission_id, "boom", retry=True, max_attempts=2)

    await store.claim("worker", lease_seconds=60)
    buried = await store.fail(admission_id, "boom", retry=True, max_attempts=2)

    assert queued_again == "queued"     # first failure: worth another try
    assert buried == "failed"           # attempts exhausted: stop looping
    assert await store.claim("worker", lease_seconds=60) is None


async def test_complete_is_terminal():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("worker", lease_seconds=60)

    await store.complete(admission_id)

    record = await store.admission(admission_id)
    assert record is not None and record["status"] == "done"
    assert await store.claim("worker", lease_seconds=60) is None


async def test_routing_pins_a_subject_to_one_conversation():
    store = GatewayStore()

    assert await store.route("telegram", "1") is None
    await store.save_route("telegram", "1", "conv_abc")
    assert await store.route("telegram", "1") == "conv_abc"
    await store.save_route("telegram", "1", "conv_def")   # /new switches it
    assert await store.route("telegram", "1") == "conv_def"


async def test_turn_lease_excludes_another_holder():
    store = GatewayStore()

    assert await store.acquire_turn("conv_1", "worker-a", ttl_seconds=60) is True
    assert await store.acquire_turn("conv_1", "worker-b", ttl_seconds=60) is False
    await store.release_turn("conv_1", "worker-a")
    assert await store.acquire_turn("conv_1", "worker-b", ttl_seconds=60) is True


class _FakeAdapter:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send(self, target: str, text: str) -> dict:
        self.sent.append((target, text))
        return {"ok": True}


def _api(store: GatewayStore, token: str = "t0ken", adapters: dict | None = None):
    return create_api(store, token=token, adapters=adapters if adapters is not None else {})


def _client(api, token: str = "t0ken") -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_claim_and_complete_delivers_through_the_adapter():
    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="100000001",
                                       payload=payload())

    async with _client(api) as client:
        claimed = (await client.post("/v1/claim", json={"worker": "w1"})).json()["admission"]
        assert claimed["admission_id"] == admission_id
        done = (await client.post("/v1/complete", json={
            "admission_id": admission_id, "answer": "hi back",
            "conversation_id": "conv_xyz",
        })).json()

    assert done["delivered"] is True
    assert adapter.sent == [("100000001", "hi back")]
    assert await store.route("telegram", "100000001") == "conv_xyz"


async def test_api_rejects_a_bad_token():
    api = _api(GatewayStore())
    async with _client(api, token="wrong") as client:
        response = await client.post("/v1/claim", json={"worker": "w1"})
    assert response.status_code == 401


async def test_health_reports_queue_and_cursor():
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload())
    async with _client(_api(store)) as client:
        body = (await client.get("/health")).json()
    assert body["ok"] is True
    assert body["queue"] == {"queued": 1}


async def test_worker_answers_a_claimed_turn_once():
    """One worker, one admission, one answer — with no adapters in the brain."""
    from leo.gateway import worker as worker_module

    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    await store.admit(channel="telegram", subject="100000001", payload=payload("ping"))

    class _Brain:
        def __init__(self) -> None:
            self.asked: list[tuple] = []

        async def initialize(self) -> None:
            return None

        async def ask(self, text, channel=None, subject=None, conversation_id=None):
            self.asked.append((text, channel, subject, conversation_id))
            return {"content": "pong", "conversation_id": "conv_worker"}

    real_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": "Bearer t0ken"},
    )
    original = worker_module.GatewayClient

    class _Client(original):
        def __init__(self, base_url, token, client=None):
            super().__init__(base_url, token, client=real_client)

    worker_module.GatewayClient = _Client
    brain = _Brain()
    try:
        answered = await worker_module.run_worker(
            "http://gateway", token="t0ken", wait_seconds=0.0, once=True, brain=brain
        )
    finally:
        worker_module.GatewayClient = original
        await real_client.aclose()

    assert answered == 1
    assert brain.asked and brain.asked[0][0] == "ping"
    assert adapter.sent == [("100000001", "pong")]
    assert (await store.admission_by_status("done")) is not None


class _TypingAdapter(_FakeAdapter):
    """Counts typing pulses so a leaked one is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.pulses = 0

    async def typing(self, target: str) -> dict:
        self.pulses += 1
        return {"ok": True}


async def test_reclaim_reports_a_stranded_claim_so_its_typing_pulse_can_stop():
    """The leak that made a finished agent look like it was still working.

    An admission can end in a non-retryable state (`cancelled`) while still
    holding a claim row. `reclaim_expired` only requeued `claimed` rows, so such
    a row kept its typing task alive forever: nothing else ever posts
    /v1/complete or /v1/fail for it. The ids must be reported even though the
    row is not requeued, because the pulse belongs to this process.
    """
    from leo import db as _db

    store = GatewayStore()
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001", payload=payload()
    )
    await store.claim("dead-worker", lease_seconds=1)
    # A stop path ends the admission but leaves the claim columns behind.
    await _db.execute(
        "UPDATE gateway_queue SET status='cancelled', claim_expires_at=? "
        "WHERE admission_id=?",
        ("2020-01-01T00:00:00+00:00", admission_id),
    )

    reclaimed = await store.reclaim_expired()

    assert admission_id in reclaimed, "stranded claim was not reported, so its typing task leaks"
    # It must not be requeued: cancelled work is terminal.
    row = await _db.fetchone(
        "SELECT status FROM gateway_queue WHERE admission_id=?", (admission_id,)
    )
    assert row["status"] == "cancelled"
