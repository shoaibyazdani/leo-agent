"""Offline verification for the ntfy channel adapter.

No live network calls: every HTTP exchange goes through ``httpx.MockTransport``
and credentials are fake values that must never surface in errors or records.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from leo.integrations.channels.ntfy import NtfyAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage

ENV_NAMES = ("NTFY_URL", "NTFY_USER", "NTFY_PASSWORD", "NTFY_TOPIC")
FAKE_PASSWORD = "fake-ntfy-password-never-leak"
NTFY_URL = "https://ntfy.example.org"

STREAM = "\n".join(
    [
        json.dumps({"id": "open-1", "time": 1700000000, "event": "open", "topic": "my-topic"}),
        json.dumps(
            {
                "id": "m-1",
                "time": 1700000001,
                "event": "message",
                "topic": "my-topic",
                "title": "alert title",
                "message": "first",
            }
        ),
        json.dumps({"id": "keep-1", "time": 1700000002, "event": "keepalive", "topic": "my-topic"}),
        json.dumps({"id": "m-2", "time": 1700000005, "event": "message", "topic": "my-topic", "message": "second"}),
        "this line is not json",
        json.dumps({"id": "m-3", "time": 1700000006, "event": "message", "topic": "my-topic", "message": None}),
    ]
)


@pytest.fixture(autouse=True)
def clean_ntfy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _configured(monkeypatch: pytest.MonkeyPatch, *, url: str = NTFY_URL) -> None:
    monkeypatch.setenv("NTFY_URL", url)


# ------------------------------------------------------------------------ send


async def test_ntfy_send_posts_raw_text(monkeypatch):
    _configured(monkeypatch)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"X-Id": "msg-out-7"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await NtfyAdapter(client=client).send("my-topic", "hello ntfy")
    assert result == {"channel": "ntfy", "target": "my-topic", "success": True, "message_id": "msg-out-7"}
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{NTFY_URL}/my-topic"
    assert request.headers["content-type"] == "text/plain"
    assert request.content == b"hello ntfy"
    assert "authorization" not in request.headers  # anonymous publish by default


async def test_ntfy_send_optional_basic_auth_from_env(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("NTFY_USER", "ntfy-user")
    monkeypatch.setenv("NTFY_PASSWORD", FAKE_PASSWORD)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)  # no X-Id header: message_id stays None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await NtfyAdapter(client=client).send("my-topic", "hi")
    assert result == {"channel": "ntfy", "target": "my-topic", "success": True, "message_id": None}
    expected = "Basic " + base64.b64encode(f"ntfy-user:{FAKE_PASSWORD}".encode()).decode("ascii")
    assert requests[0].headers["authorization"] == expected


async def test_ntfy_send_blocked_without_url(monkeypatch):
    with pytest.raises(MessagingError, match="ntfy send blocked: NTFY_URL is not set"):
        await NtfyAdapter().send("my-topic", "hi")


async def test_ntfy_send_rejects_invalid_url_and_topic(monkeypatch):
    _configured(monkeypatch)
    adapter = NtfyAdapter()
    for topic in ("has/slash", "has space", ""):
        with pytest.raises(MessagingError, match="invalid ntfy topic"):
            await adapter.send(topic, "hi")
    monkeypatch.setenv("NTFY_URL", "ftp://not-http")
    with pytest.raises(MessagingError, match="NTFY_URL is not a valid"):
        await adapter.send("my-topic", "hi")


async def test_ntfy_send_failures_report_no_secrets(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("NTFY_USER", "ntfy-user")
    monkeypatch.setenv("NTFY_PASSWORD", FAKE_PASSWORD)

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": 40001, "http": 400})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(MessagingError, match="ntfy send failed: HTTP 400") as exc:
            await NtfyAdapter(client=client).send("my-topic", "hi")
    assert FAKE_PASSWORD not in str(exc.value)

    def exploding(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError(f"boom reading {NTFY_URL} as ntfy-user:{FAKE_PASSWORD}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(exploding)) as client:
        with pytest.raises(MessagingError, match="ntfy send failed: ReadError") as exc:
            await NtfyAdapter(client=client).send("my-topic", "hi")
    assert FAKE_PASSWORD not in str(exc.value)
    assert "boom" not in str(exc.value)


# ------------------------------------------------------------------------ poll


async def test_ntfy_poll_filters_stream_events_and_advances_cursor(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("NTFY_TOPIC", "my-topic")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        since = dict(request.url.params).get("since")
        if since is None:
            return httpx.Response(200, content=STREAM.encode())
        return httpx.Response(
            200,
            content=json.dumps(
                {"id": "m-4", "time": 1700000010, "event": "message", "topic": "my-topic", "message": "third"}
            ).encode()
            + b"\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = NtfyAdapter(client=client)
        messages = await adapter.poll()
        assert dict(requests[0].url.params) == {"poll": "1"}  # no cursor yet
        assert [(m.sender, m.text, m.message_id) for m in messages] == [
            ("alert title", "first", "m-1"),  # title acts as the sender
            ("my-topic", "second", "m-2"),  # no title: the topic names the source
        ]
        assert all(isinstance(m, NormalizedMessage) for m in messages)
        assert all(m.channel == "ntfy" for m in messages)
        assert messages[0].timestamp.startswith("2023-11-14")
        assert messages[1].timestamp.startswith("2023-11-14")

        followups = await adapter.poll()
        # The cursor is the newest message time seen so far.
        assert dict(requests[1].url.params) == {"poll": "1", "since": "1700000005"}
        assert [(m.text, m.message_id) for m in followups] == [("third", "m-4")]


async def test_ntfy_poll_requires_url_and_topic(monkeypatch):
    with pytest.raises(MessagingError, match="ntfy poll blocked: NTFY_URL is not set"):
        await NtfyAdapter().poll()
    _configured(monkeypatch)
    with pytest.raises(MessagingError, match="ntfy poll blocked: NTFY_TOPIC is not set"):
        await NtfyAdapter().poll()


async def test_ntfy_poll_http_failure(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setenv("NTFY_TOPIC", "my-topic")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(404, json={"code": 40401}))
    )
    try:
        with pytest.raises(MessagingError, match="ntfy poll failed: HTTP 404"):
            await NtfyAdapter(client=client).poll()
    finally:
        await client.aclose()


# ---------------------------------------------------------------------- record


async def test_ntfy_record_blocked_configured_and_capabilities(monkeypatch):
    adapter = NtfyAdapter()
    record = adapter.record()
    assert record.status == "blocked"
    assert record.capabilities == ["send", "poll"]
    assert record.metadata["credential_env"] == ["NTFY_URL"]

    _configured(monkeypatch)
    record = NtfyAdapter().record()
    assert record.status == "configured"
    assert record.config_ref == "NTFY_URL"
    assert FAKE_PASSWORD not in json.dumps(record.model_dump())
