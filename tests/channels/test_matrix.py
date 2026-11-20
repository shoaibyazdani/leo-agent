"""Offline verification for the Matrix channel adapter.

No live network calls: every HTTP exchange goes through ``httpx.MockTransport``
and credentials are fake values that must never surface in errors or records.
"""

from __future__ import annotations

import json

import httpx
import pytest

from leo.integrations.channels.matrix import MatrixAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage

ENV_NAMES = ("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN")
FAKE_TOKEN = "fake-matrix-token-never-leak"
HOMESERVER = "https://homeserver.example.org"

HISTORICAL_EVENT = {
    "type": "m.room.message",
    "sender": "@old:homeserver.example.org",
    "event_id": "$old-1:homeserver.example.org",
    "origin_server_ts": 1690000000000,
    "content": {"msgtype": "m.text", "body": "posted before the first poll"},
}

NEW_EVENTS = [
    {
        "type": "m.room.member",  # state event: never a message
        "sender": "@alice:homeserver.example.org",
        "event_id": "$state-1:homeserver.example.org",
        "origin_server_ts": 1700000000000,
        "content": {"membership": "join"},
    },
    {
        "type": "m.room.message",
        "sender": "@alice:homeserver.example.org",
        "event_id": "$new-1:homeserver.example.org",
        "origin_server_ts": 1700000000000,
        "content": {
            "msgtype": "m.text",
            "body": "hello matrix",
            "m.relates_to": {"m.in_reply_to": {"event_id": "$orig-1:homeserver.example.org"}},
        },
    },
    {  # m.room.message without a body: skipped defensively
        "type": "m.room.message",
        "sender": "@bob:homeserver.example.org",
        "origin_server_ts": 1700000001000,
        "content": {"msgtype": "m.text"},
    },
    {  # no event_id: the replay guard needs one, so it is skipped
        "type": "m.room.message",
        "sender": "@bob:homeserver.example.org",
        "origin_server_ts": 1700000002000,
        "content": {"msgtype": "m.text", "body": "no id"},
    },
    "not-even-a-dict",
]


def _sync_body(events: list, next_batch: str) -> dict:
    return {
        "next_batch": next_batch,
        "rooms": {"join": {"!room:homeserver.example.org": {"timeline": {"events": events}}}},
    }


@pytest.fixture(autouse=True)
def clean_matrix_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _configured(monkeypatch: pytest.MonkeyPatch, *, homeserver: str = HOMESERVER) -> None:
    monkeypatch.setenv("MATRIX_HOMESERVER", homeserver)
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", FAKE_TOKEN)


# ------------------------------------------------------------------------ send


async def test_matrix_send_uses_injected_client_and_hides_token(monkeypatch):
    _configured(monkeypatch)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"event_id": "$evt-1:homeserver.example.org"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MatrixAdapter(client=client).send("!room:homeserver.example.org", "hello")
    assert result == {
        "channel": "matrix",
        "target": "!room:homeserver.example.org",
        "success": True,
        "message_id": "$evt-1:homeserver.example.org",
    }
    request = requests[0]
    assert request.method == "PUT"
    assert request.headers["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert json.loads(request.content) == {"msgtype": "m.text", "body": "hello"}
    path = request.url.path
    assert path.startswith(
        "/_matrix/client/v3/rooms/!room:homeserver.example.org/send/m.room.message/"
    )
    txn = path.rsplit("/", 1)[-1]
    assert len(txn) == 32 and all(ch in "0123456789abcdef" for ch in txn)


async def test_matrix_send_accepts_alias_and_encodes_fragment_safely(monkeypatch):
    _configured(monkeypatch)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"event_id": "$evt-2:homeserver.example.org"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await MatrixAdapter(client=client).send("#alias:homeserver.example.org", "hi")
    # A raw '#' would be read as a URL fragment: it must arrive percent-encoded.
    assert str(requests[0].url).startswith(
        f"{HOMESERVER}/_matrix/client/v3/rooms/%23alias:homeserver.example.org/"
    )


async def test_matrix_send_blocked_without_envs(monkeypatch):
    adapter = MatrixAdapter()
    with pytest.raises(MessagingError, match="matrix send blocked: MATRIX_HOMESERVER is not set"):
        await adapter.send("!room:homeserver.example.org", "hi")
    monkeypatch.setenv("MATRIX_HOMESERVER", HOMESERVER)
    with pytest.raises(MessagingError, match="matrix send blocked: MATRIX_ACCESS_TOKEN is not set"):
        await adapter.send("!room:homeserver.example.org", "hi")


async def test_matrix_send_rejects_invalid_room_and_homeserver(monkeypatch):
    _configured(monkeypatch)
    adapter = MatrixAdapter()
    with pytest.raises(MessagingError, match="invalid Matrix room id"):
        await adapter.send("room-without-a-sigil", "hi")
    monkeypatch.setenv("MATRIX_HOMESERVER", "not a url")
    with pytest.raises(MessagingError, match="MATRIX_HOMESERVER is not a valid"):
        await adapter.send("!room:homeserver.example.org", "hi")


async def test_matrix_send_failures_report_no_secrets(monkeypatch):
    _configured(monkeypatch)

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"errcode": "M_UNKNOWN"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError, match="matrix send failed: HTTP 400") as exc:
            await MatrixAdapter(client=client).send("!room:homeserver.example.org", "hi")
    assert FAKE_TOKEN not in str(exc.value)

    def exploding(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"dial {HOMESERVER} with {FAKE_TOKEN}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(exploding)) as client:
        with pytest.raises(MessagingError, match="matrix send failed: ConnectError") as exc:
            await MatrixAdapter(client=client).send("!room:homeserver.example.org", "hi")
    assert FAKE_TOKEN not in str(exc.value)
    assert "dial" not in str(exc.value)


# ------------------------------------------------------------------------ poll


async def test_matrix_poll_first_call_ignores_history_and_adopts_cursor(monkeypatch):
    _configured(monkeypatch)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_sync_body([HISTORICAL_EVENT], "cursor-1"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = MatrixAdapter(client=client)
        # First /sync returns initial state + recent timeline as history:
        # nothing is normalized, next_batch becomes the cursor.
        assert await adapter.poll() == []
    params = dict(requests[0].url.params)
    assert params == {"timeout": "0"}  # no `since` on the first call
    assert requests[0].headers["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert requests[0].url.path == "/_matrix/client/v3/sync"


async def test_matrix_poll_normalizes_then_stays_idempotent(monkeypatch):
    _configured(monkeypatch)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        since = dict(request.url.params).get("since")
        if not since:
            return httpx.Response(200, json=_sync_body([HISTORICAL_EVENT], "cursor-1"))
        return httpx.Response(200, json=_sync_body(NEW_EVENTS, "cursor-2"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = MatrixAdapter(client=client)
        assert await adapter.poll() == []
        messages = await adapter.poll()
        assert dict(requests[1].url.params)["since"] == "cursor-1"
        assert len(messages) == 1
        message = messages[0]
        assert isinstance(message, NormalizedMessage)
        assert (message.channel, message.sender, message.text) == (
            "matrix",
            "@alice:homeserver.example.org",
            "hello matrix",
        )
        assert message.message_id == "$new-1:homeserver.example.org"
        assert message.reply_ref == "$orig-1:homeserver.example.org"
        assert message.timestamp.startswith("2023-11-14")
        # Third poll returns the SAME mocked body: the cursor advanced and the
        # replay guard drops the already-normalized event id.
        assert await adapter.poll() == []
        assert dict(requests[2].url.params)["since"] == "cursor-2"


async def test_matrix_poll_defensive_failures(monkeypatch):
    _configured(monkeypatch)
    adapter = MatrixAdapter(client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"not json")
    )))
    try:
        assert await adapter.poll() == []
    finally:
        await adapter.aclose()

    adapter = MatrixAdapter(client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"errcode": "M_UNKNOWN_TOKEN"})
    )))
    try:
        with pytest.raises(MessagingError, match="matrix poll failed: HTTP 401"):
            await adapter.poll()
    finally:
        await adapter.aclose()


async def test_matrix_poll_blocked_without_envs(monkeypatch):
    monkeypatch.setenv("MATRIX_HOMESERVER", HOMESERVER)
    with pytest.raises(MessagingError, match="matrix poll blocked: MATRIX_ACCESS_TOKEN is not set"):
        await MatrixAdapter().poll()


# ---------------------------------------------------------------------- record


async def test_matrix_record_blocked_configured_and_capabilities(monkeypatch):
    adapter = MatrixAdapter()
    record = adapter.record()
    assert record.status == "blocked"
    assert record.capabilities == ["send", "poll"]  # no webhook transport
    assert record.verification == "mock-tested"
    assert record.metadata["credential_env"] == ["MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN"]
    assert record.metadata["reason"] == "missing credentials: MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN"

    _configured(monkeypatch)
    record = MatrixAdapter().record()
    assert record.status == "configured"
    assert record.config_ref == "MATRIX_HOMESERVER,MATRIX_ACCESS_TOKEN"
    assert record.metadata["live_verified"] is False
    assert FAKE_TOKEN not in json.dumps(record.model_dump())
