"""Durable transport position: a restart must resume, not replay or drop.

The Telegram poll offset used to live in a process variable, so after a restart
LEO re-asked ``getUpdates`` with no offset: it either replayed a day of messages
(deduped, but wasteful) or, if the platform had already aged them out, silently
lost what arrived while it was down.

These tests pin the new behaviour: the offset is read from SQLite, written back
after every poll, a jump in ids is reported as a gap instead of being smoothed
over, and each poll records a heartbeat.
"""

from __future__ import annotations

import pytest

from leo import db
from leo.integrations.gateway import Gateway
from leo.integrations.messaging import MessagingAdapter, TelegramAdapter
from leo.integrations.transport_state import TransportState


class _Resp:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, batches: list[list[dict]]) -> None:
        self.batches = batches
        self.calls: list[dict] = []

    async def get(self, url: str, params: dict | None = None, timeout: float | None = None):
        self.calls.append(dict(params or {}))
        batch = self.batches.pop(0) if self.batches else []
        return _Resp({"ok": True, "result": batch})


def _update(update_id: int, user: int = 100000001, text: str = "hi") -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1763700000,
            "text": text,
            "from": {"id": user, "username": "operator"},
            "chat": {"id": user, "type": "private"},
        },
    }


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")


async def test_cursor_survives_a_restart():
    first = TransportState("telegram")
    await first.set_cursor(91)

    # a fresh instance stands in for a new process: same SQLite, no memory
    assert await TransportState("telegram").cursor() == 91


async def test_poll_sends_the_stored_offset_and_advances_it():
    await TransportState("telegram").set_cursor(50)
    client = _Client([[_update(50), _update(51)]])
    adapter = TelegramAdapter(client=client)

    await adapter.poll(TransportState("telegram"))

    assert client.calls[0]["offset"] == "50"
    assert await TransportState("telegram").cursor() == 52


async def test_poll_without_stored_state_starts_clean_then_persists():
    client = _Client([[_update(7)]])
    adapter = TelegramAdapter(client=client)

    await adapter.poll(TransportState("telegram"))

    assert "offset" not in client.calls[0]  # nothing to resume from yet
    assert await TransportState("telegram").cursor() == 8


async def test_a_jump_in_update_ids_is_reported_as_a_gap(caplog):
    await TransportState("telegram").set_cursor(10)
    client = _Client([[_update(40)]])
    adapter = TelegramAdapter(client=client)

    with caplog.at_level("WARNING"):
        await adapter.poll(TransportState("telegram"))

    assert any("poll gap" in record.getMessage() for record in caplog.records)


async def test_a_poll_cycle_records_a_heartbeat():
    """Liveness belongs to the gateway's poll cycle, not to one adapter call."""
    gateway = Gateway({"telegram": TelegramAdapter(client=_Client([[]]))})

    await gateway.poll("telegram")

    beat = await TransportState("telegram").last_beat()
    assert beat is not None
    assert beat["beat_at"]
    assert beat["note"] == "0 inbound"


async def test_gateway_poll_uses_durable_state(monkeypatch):
    """The gateway owns the store, so adapters stay transport-only."""

    class _Fake(TelegramAdapter):
        async def poll(self, state=None):
            assert state is not None, "gateway must pass a TransportState"
            await state.set_cursor(123)
            await state.beat(note="fake")
            return []

    gateway = Gateway({"telegram": _Fake(client=_Client([[]]))})
    await gateway.poll("telegram")

    assert await TransportState("telegram").cursor() == 123
    assert (await TransportState("telegram").last_beat()) is not None
