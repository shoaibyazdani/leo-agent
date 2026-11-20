"""Offline verification for the A2A (Agent2Agent) channel adapter.

No live network calls: HTTP goes through httpx.MockTransport, and the
credentials are fake values that must never leak into records or errors.
"""

from __future__ import annotations

import json

import httpx
import pytest

from leo.integrations.channels.a2a import A2AAdapter
from leo.integrations.messaging import MessagingError


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_send_uses_published_jsonrpc_shape_with_bearer_key(monkeypatch):
    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example/a2a")
    monkeypatch.setenv("A2A_API_KEY", "fake-a2a-key-should-never-leak")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": "task-9"})

    async with _client(handler) as client:
        result = await A2AAdapter(client=client).send("peer-agent", "hello agent")

    assert result == {
        "channel": "a2a",
        "target": "peer-agent",
        "success": True,
        "message_id": "task-9",
    }
    request = requests[0]
    assert str(request.url) == "https://peer.example/a2a"
    assert request.headers["authorization"] == "Bearer fake-a2a-key-should-never-leak"
    body = json.loads(request.content)
    assert body["jsonrpc"] == "2.0"
    assert body["method"] == "message/send"
    assert isinstance(body["id"], str) and body["id"]
    message = body["params"]["message"]
    assert message["role"] == "user" and message["kind"] == "message"
    assert message["parts"] == [{"kind": "text", "text": "hello agent"}]
    assert isinstance(message["messageId"], str) and message["messageId"]
    # Fresh client-generated ids per send: the request id and message id differ.
    assert body["id"] != message["messageId"]


async def test_send_without_api_key_omits_authorization_header(monkeypatch):
    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example/a2a")
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {"id": "obj-task-7"}})

    async with _client(handler) as client:
        result = await A2AAdapter(client=client).send("peer-agent", "hi")

    assert "authorization" not in requests[0].headers
    assert result["message_id"] == "obj-task-7"


async def test_send_posts_to_url_without_path_exactly_as_given(monkeypatch):
    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example")
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": "t-1"})

    async with _client(handler) as client:
        await A2AAdapter(client=client).send("peer-agent", "hi")

    # No path invention: the POST goes to the configured endpoint as given.
    assert str(requests[0].url) == "https://peer.example"


async def test_send_extracts_task_id_defensively(monkeypatch):
    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example/a2a")
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    cases = [
        ("task-9", "task-9"),
        ({"id": "obj-7"}, "obj-7"),
        ({"taskId": "task-42"}, "task-42"),
        ({"id": None, "taskId": 99}, "99"),
        ({"anything": "else"}, None),
        ([1, 2, 3], None),
        (None, None),
        ("", None),
    ]
    for result_value, expected in cases:

        def handler(request: httpx.Request, result=result_value) -> httpx.Response:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": result})

        async with _client(handler) as client:
            sent = await A2AAdapter(client=client).send("peer-agent", "hi")
        assert sent["message_id"] == expected, result_value


async def test_send_peer_error_truncates_untrusted_text(monkeypatch):
    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example/a2a")
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    flood = "P" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": "1", "error": {"code": -32000, "message": flood}},
        )

    async with _client(handler) as client:
        with pytest.raises(MessagingError) as exc:
            await A2AAdapter(client=client).send("peer-agent", "hi")

    assert "peer reported error" in str(exc.value)
    assert flood not in str(exc.value)  # peer text is never echoed verbatim
    assert "P" * 200 in str(exc.value)  # only a truncated echo appears


async def test_send_blocked_when_agent_url_missing(monkeypatch):
    monkeypatch.delenv("A2A_AGENT_URL", raising=False)
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    with pytest.raises(MessagingError, match="A2A_AGENT_URL is not set"):
        await A2AAdapter().send("peer-agent", "hi")


async def test_send_rejects_non_http_urls(monkeypatch):
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    for bad in ("ftp://peer.example/rpc", "file:///etc/passwd", "not a url", "http://user:pw@peer/x"):
        monkeypatch.setenv("A2A_AGENT_URL", bad)
        with pytest.raises(MessagingError, match="not a valid http"):
            await A2AAdapter().send("peer-agent", "hi")


async def test_send_maps_http_error_status(monkeypatch):
    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example/a2a")
    monkeypatch.delenv("A2A_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": -32600, "message": "invalid request"}})

    async with _client(handler) as client:
        with pytest.raises(MessagingError, match="HTTP 400"):
            await A2AAdapter(client=client).send("peer-agent", "hi")


async def test_send_wraps_transport_errors_without_echoing_them(monkeypatch):
    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example/a2a")
    monkeypatch.delenv("A2A_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused by peer")

    async with _client(handler) as client:
        with pytest.raises(MessagingError, match="ConnectError") as exc:
            await A2AAdapter(client=client).send("peer-agent", "hi")

    assert "connection refused by peer" not in str(exc.value)


def test_record_reports_send_only_capability_and_env_names(monkeypatch):
    monkeypatch.delenv("A2A_AGENT_URL", raising=False)
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    blocked = A2AAdapter().record()
    assert blocked.status == "blocked"
    assert blocked.capabilities == ["send"]  # no webhook, no polling
    assert blocked.config_ref == "A2A_AGENT_URL"
    assert blocked.metadata["reason"] == "missing credentials: A2A_AGENT_URL"

    monkeypatch.setenv("A2A_AGENT_URL", "https://peer.example/a2a")
    configured = A2AAdapter().record()
    assert configured.status == "configured"
    assert configured.capabilities == ["send"]
    assert configured.config_ref == "A2A_AGENT_URL"
    assert configured.metadata["credential_env"] == ["A2A_AGENT_URL"]
    assert configured.verification == "mock-tested"
