"""Lease renewal and the no-inbound API: two processes, one transport owner.

Renewal separates "slow" from "dead": without it the only thing a lease says is
that time passed, so a healthy turn that runs long looks exactly like a crashed
worker and gets answered twice. These tests pin that, plus the second half of the
same problem - a brain that serves the dashboard must never poll a token the
gateway already owns.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from leo.gateway import GatewayStore
from leo.gateway.server import create_api


def payload(mid: str = "1", text: str = "hello") -> dict:
    return {"text": text, "message_id": mid, "sender_id": "100000001",
            "conversation_ref": "100000001"}


def _client(api, token: str = "t0ken") -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": f"Bearer {token}"},
    )


def _api(store: GatewayStore):
    return create_api(store, token="t0ken", adapters={})


# --------------------------------------------------------------------- renewal
async def test_renew_extends_a_held_lease():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("worker-a", lease_seconds=60)

    assert await store.renew(admission_id, "worker-a", lease_seconds=120) is True
    row = await store.stats()
    assert row["queue"] == {"claimed": 1}


async def test_renew_refuses_a_worker_that_does_not_hold_it():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("worker-a", lease_seconds=60)

    assert await store.renew(admission_id, "worker-b", lease_seconds=120) is False


async def test_renew_refuses_after_the_lease_expired():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("worker-a", lease_seconds=-1)  # already gone

    assert await store.renew(admission_id, "worker-a", lease_seconds=60) is False


async def test_a_renewed_turn_is_not_reclaimed_by_another_worker():
    """The whole point: a live worker keeps its turn, a silent one loses it."""
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("worker-a", lease_seconds=1)

    await store.renew(admission_id, "worker-a", lease_seconds=60)
    await store.reclaim_expired()

    assert await store.claim("worker-b", lease_seconds=60) is None


async def test_renew_over_the_api_reports_ownership():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    api = _api(store)

    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1"})
        mine = await client.post("/v1/renew", json={"admission_id": admission_id, "worker": "w1"})
        theirs = await client.post("/v1/renew", json={"admission_id": admission_id, "worker": "w2"})

    assert mine.json()["renewed"] is True
    assert theirs.json()["renewed"] is False


async def test_the_worker_renews_while_a_turn_is_running():
    """A turn slower than the lease must still finish, once."""
    from leo.gateway import worker as worker_module

    store = GatewayStore()
    api = _api(store)
    await store.admit(channel="telegram", subject="100000001", payload=payload(text="slow"))

    renewals: list[str] = []

    class _Brain:
        async def initialize(self) -> None:
            return None

        async def ask(self, text, channel=None, subject=None, conversation_id=None):
            await asyncio.sleep(1.2)          # longer than the 1s lease below
            return {"content": "done", "conversation_id": "conv_slow"}

    real_transport = httpx.ASGITransport(app=api)
    original_client = worker_module.GatewayClient

    class _Client(original_client):
        def __init__(self, base_url, token, client=None):
            super().__init__(
                base_url, token,
                client=httpx.AsyncClient(transport=real_transport, base_url="http://gateway",
                                         headers={"Authorization": "Bearer t0ken"}),
            )

        async def renew(self, admission_id, worker, lease_seconds=30):
            renewals.append(admission_id)
            return await super().renew(admission_id, worker, lease_seconds)

    worker_module.GatewayClient = _Client
    try:
        answered = await worker_module.run_worker(
            "http://gateway", token="t0ken", wait_seconds=0.0, once=True,
            brain=_Brain(), lease_seconds=1,
        )
    finally:
        worker_module.GatewayClient = original_client

    assert answered == 1
    assert renewals, "the worker never renewed its lease during the turn"
    assert (await store.admission_by_status("done")) is not None


# ------------------------------------------------------------------- no inbound
async def test_start_background_can_leave_polling_to_a_gateway():
    from leo.core.app import LeoApp

    app = LeoApp()
    try:
        await app.start_background(inbound=False)
        assert app._inbound_task is None, "brain must not poll when a gateway owns the token"
        assert app._scheduler_task is not None  # everything else still runs
    finally:
        for task in (app._scheduler_task, app._gateway_task, app._inbound_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *[t for t in (app._scheduler_task, app._gateway_task, app._inbound_task)
              if t is not None],
            return_exceptions=True,
        )


def test_no_inbound_switch_is_read_from_the_environment(monkeypatch):
    from leo.interfaces import api as api_module

    monkeypatch.delenv("LEO_DISABLE_INBOUND", raising=False)
    assert api_module._disable_inbound() is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("LEO_DISABLE_INBOUND", value)
        assert api_module._disable_inbound() is True
    monkeypatch.setenv("LEO_DISABLE_INBOUND", "0")
    assert api_module._disable_inbound() is False
