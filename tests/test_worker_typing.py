"""The typing indicator must survive the split.

Before the split the reply path lived beside the adapters, and the gateway pulsed
``typing`` on the channel while a reply was being produced. Afterwards the brain
answers over an API and the reply path never touches an adapter, so the indicator
silently disappeared — transport hints belong to the transport, not the brain.

These tests pin: a claim starts the pulse, completion stops it, and a lease that
is re-claimed by someone else does not leave two pulses running.
"""

from __future__ import annotations

import asyncio

import httpx

from leo.gateway import GatewayStore
from leo.gateway.server import create_api

MASTER = "master-t0ken"


class _TypingAdapter:
    """Records typing pulses instead of reaching a chat platform."""

    def __init__(self) -> None:
        self.typed: list[str] = []

    async def send_typing(self, target: str) -> None:
        self.typed.append(target)

    async def send(self, target: str, text: str) -> dict:
        return {"ok": True}


def _payload(mid: str = "1", text: str = "hello") -> dict:
    return {"text": text, "message_id": mid, "sender_id": "100000001",
            "conversation_ref": "100000001"}


def _client(api) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": f"Bearer {MASTER}"},
    )


async def test_a_claim_starts_the_typing_pulse():
    store = GatewayStore()
    adapter = _TypingAdapter()
    api = create_api(store, token=MASTER, adapters={"telegram": adapter})
    await store.admit(channel="telegram", subject="100000001", payload=_payload())

    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1", "lease_seconds": 30})
        await asyncio.sleep(0.2)          # the first pulse is immediate
        first = len(adapter.typed)
        await client.post("/v1/complete", json={"admission_id": (await _admission(store))[0],
                                                "answer": "hi"})
        await asyncio.sleep(0.2)
        after_complete = len(adapter.typed)

    assert first >= 1, "claiming a turn must show the channel that work started"
    assert adapter.typed[0] == "100000001"
    assert after_complete == first, "completion must stop the pulse"


async def _admission(store: GatewayStore) -> tuple[str, str]:
    row = await store.admission_by_status("claimed")
    if row is None:
        row = await store.admission_by_status("done")
    assert row is not None
    return row["admission_id"], row["subject"]


async def test_failure_also_stops_the_pulse():
    store = GatewayStore()
    adapter = _TypingAdapter()
    api = create_api(store, token=MASTER, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="100000001",
                                        payload=_payload())

    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1", "lease_seconds": 30})
        await asyncio.sleep(0.1)
        before = len(adapter.typed)
        await client.post("/v1/fail", json={"admission_id": admission_id, "error": "boom"})
        await asyncio.sleep(0.2)
        after = len(adapter.typed)

    assert before >= 1
    assert after == before


async def test_reclaiming_does_not_leave_two_pulses_running():
    store = GatewayStore()
    adapter = _TypingAdapter()
    api = create_api(store, token=MASTER, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="100000001",
                                        payload=_payload())

    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1", "lease_seconds": -1})
        await asyncio.sleep(0.1)
        # a second claim for the same admission (the first worker died)
        await client.post("/v1/claim", json={"worker": "w2", "lease_seconds": 30})
        await asyncio.sleep(0.1)
        sample = len(adapter.typed)
        await asyncio.sleep(0.3)
        later = len(adapter.typed)

    assert admission_id
    # one pulse per 4s interval: a doubled pulse would roughly double the rate
    assert later - sample <= 1, f"pulses stacked: {later - sample} in 0.3s"


async def test_a_channel_without_a_typing_hook_is_silently_fine():
    store = GatewayStore()

    class _Plain:
        async def send(self, target: str, text: str) -> dict:
            return {"ok": True}

    api = create_api(store, token=MASTER, adapters={"telegram": _Plain()})
    await store.admit(channel="telegram", subject="1", payload=_payload())

    async with _client(api) as client:
        claimed = await client.post("/v1/claim", json={"worker": "w1", "lease_seconds": 30})
        await asyncio.sleep(0.1)

    assert claimed.status_code == 200


async def test_an_unknown_channel_does_not_break_the_claim():
    store = GatewayStore()
    api = create_api(store, token=MASTER, adapters={})
    await store.admit(channel="irc", subject="1", payload=_payload())

    async with _client(api) as client:
        claimed = await client.post("/v1/claim", json={"worker": "w1", "lease_seconds": 30})

    assert claimed.status_code == 200
    assert claimed.json()["admission"] is not None
