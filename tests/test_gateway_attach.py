"""Attach tickets and worker sessions: the brain never holds the master token.

A worker needs to call claim/complete. Handing it the operator's master token would
mean every worker, every log and every crash report carries the credential that
also mints new tickets. Instead the gateway mints a single-use ticket, the worker
exchanges it for a short-lived session token, and only that token rides on the turn
routes.
"""

from __future__ import annotations

import httpx
import pytest

from leo.gateway import GatewayStore
from leo.gateway.sessions import SessionStore
from leo.gateway.server import create_api

MASTER = "master-t0ken"


def payload(mid: str = "1", text: str = "hello") -> dict:
    return {"text": text, "message_id": mid, "sender_id": "100000001",
            "conversation_ref": "100000001"}


def _client(api, token: str = MASTER) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": f"Bearer {token}"},
    )


def _api(store: GatewayStore):
    return create_api(store, token=MASTER, adapters={})


# ------------------------------------------------------------------- the store
async def test_a_ticket_is_single_use():
    sessions = SessionStore()
    ticket, _ = await sessions.issue_ticket(ttl_seconds=60)

    first = await sessions.redeem_ticket(ticket)
    second = await sessions.redeem_ticket(ticket)

    assert first and isinstance(first, str)
    assert second is None, "a ticket must not work twice"


async def test_an_expired_ticket_is_refused():
    sessions = SessionStore()
    ticket, _ = await sessions.issue_ticket(ttl_seconds=-1)

    assert await sessions.redeem_ticket(ticket) is None


async def test_a_session_token_is_valid_until_revoked():
    sessions = SessionStore()
    ticket, _ = await sessions.issue_ticket()
    token = await sessions.redeem_ticket(ticket)
    assert token is not None

    assert await sessions.session_valid(token) is True
    assert await sessions.revoke(token) is True
    assert await sessions.session_valid(token) is False


async def test_session_stats_count_live_only():
    sessions = SessionStore()
    ticket_a, _ = await sessions.issue_ticket()
    ticket_b, _ = await sessions.issue_ticket()
    live = await sessions.redeem_ticket(ticket_a)
    dead = await sessions.redeem_ticket(ticket_b)
    await sessions.revoke(dead or "")

    stats = await sessions.stats()

    assert stats["live_sessions"] == 1
    assert stats["unused_tickets"] == 0
    assert await sessions.session_valid(live or "") is True


# --------------------------------------------------------------------- the API
async def test_attach_needs_the_master_token():
    store = GatewayStore()
    api = _api(store)
    sessions = SessionStore()
    ticket, _ = await sessions.issue_ticket()
    worker_token = await sessions.redeem_ticket(ticket)

    async with _client(api, token=worker_token or "") as client:
        refused = await client.post("/v1/attach")
    async with _client(api) as client:
        allowed = await client.post("/v1/attach")

    assert refused.status_code == 401, "a worker must not mint more tickets"
    assert allowed.status_code == 200
    assert allowed.json()["ticket"]


async def test_a_session_token_can_work_the_queue():
    store = GatewayStore()
    api = _api(store)
    await store.admit(channel="telegram", subject="100000001", payload=payload())

    async with _client(api) as master:
        minted = await master.post("/v1/attach")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api),
                                 base_url="http://gateway") as anonymous:
        exchanged = await anonymous.post("/v1/session", json={"ticket": minted.json()["ticket"]})
        assert exchanged.status_code == 200
        worker_token = exchanged.json()["session_token"]

        claimed = await anonymous.post(
            "/v1/claim",
            json={"worker": "w1", "lease_seconds": 30},
            headers={"Authorization": f"Bearer {worker_token}"},
        )
        assert claimed.status_code == 200
        admission = claimed.json()["admission"]
        assert admission is not None

        done = await anonymous.post(
            "/v1/complete",
            json={"admission_id": admission["admission_id"], "answer": "hi"},
            headers={"Authorization": f"Bearer {worker_token}"},
        )
        assert done.status_code == 200


async def test_a_spent_ticket_cannot_open_a_second_session():
    api = _api(GatewayStore())
    async with _client(api) as master:
        minted = (await master.post("/v1/attach")).json()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api),
                                 base_url="http://gateway") as anonymous:
        first = await anonymous.post("/v1/session", json={"ticket": minted["ticket"]})
        second = await anonymous.post("/v1/session", json={"ticket": minted["ticket"]})

    assert first.status_code == 200
    assert second.status_code == 401


async def test_health_reports_session_counts():
    async with _client(_api(GatewayStore())) as client:
        body = (await client.get("/health")).json()
    assert "sessions" in body
    assert body["sessions"]["live_sessions"] == 0


# ------------------------------------------------------------------ the worker
async def test_a_worker_can_start_from_a_ticket_alone(monkeypatch):
    """The payoff: LEO_GATEWAY_TICKET is enough; no master token in the worker."""
    from leo.gateway import worker as worker_module

    store = GatewayStore()
    api = _api(store)
    await store.admit(channel="telegram", subject="100000001", payload=payload("ping"))

    async with _client(api) as master:
        ticket = (await master.post("/v1/attach")).json()["ticket"]

    transport = httpx.ASGITransport(app=api)
    original = worker_module.GatewayClient

    class _Client(original):
        def __init__(self, base_url, token, client=None):
            super().__init__(
                base_url, token,
                client=httpx.AsyncClient(transport=transport, base_url="http://gateway",
                                         headers={"Authorization": f"Bearer {token}"}),
            )

    class _Brain:
        async def initialize(self) -> None:
            return None

        async def ask(self, text, channel=None, subject=None, conversation_id=None):
            return {"content": "pong", "conversation_id": "conv_ticket"}

    worker_module.GatewayClient = _Client
    monkeypatch.setenv("LEO_GATEWAY_TICKET", ticket)
    monkeypatch.delenv("LEO_GATEWAY_TOKEN", raising=False)
    try:
        answered = await worker_module.run_worker(
            "http://gateway", wait_seconds=0.0, once=True, brain=_Brain()
        )
    finally:
        worker_module.GatewayClient = original

    assert answered == 1
    assert (await store.admission_by_status("done")) is not None
