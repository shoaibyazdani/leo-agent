"""Secure node bridge tests (no real network)."""

from __future__ import annotations

from typing import Any

import pytest
import httpx

from leo.integrations.node import (
    NODE_PROTOCOL_VERSION,
    NodeCapabilityError,
    NodeClient,
    NodeConfig,
    create_node_app,
)


@pytest.fixture
def node_token(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("LEO_NODE_TOKEN", "test-token-value")
    return "test-token-value"


class FakeHttp:
    """Records requests and returns a canned health payload; no HTTP occurs."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.payload = payload or {
            "node": "vps", "role": "vps", "status": "ok", "ts": "2026-01-01T00:00:00+00:00",
            "protocol_version": NODE_PROTOCOL_VERSION,
            "capabilities": {"health": "1.0", "tasks:read": "1.0", "tasks:create": "1.0"},
        }

    def _response(self, method: str, url: str) -> httpx.Response:
        self.calls.append((method, url))
        return httpx.Response(200, json=self.payload, request=httpx.Request(method, url))

    async def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        return self._response("GET", url)

    async def post(self, url: str, headers: dict[str, str] | None = None,
                   json: dict[str, Any] | None = None) -> httpx.Response:
        return self._response("POST", url)

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_node_app_auth_and_scope(node_token: str) -> None:
    app = create_node_app(NodeConfig(name="vps", role="vps"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 401
        headers = {"Authorization": f"Bearer {node_token}"}
        ok = await client.get("/health", headers=headers)
        assert ok.status_code == 200 and ok.json()["role"] == "vps"
        assert (await client.get("/tasks", headers=headers)).status_code == 403


@pytest.mark.asyncio
async def test_health_advertises_versioned_capabilities(node_token: str) -> None:
    app = create_node_app(NodeConfig(name="vps", role="vps", allowed_operations=["health", "tasks:read"]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = (await client.get(
            "/health", headers={"Authorization": f"Bearer {node_token}"}
        )).json()
    assert payload["protocol_version"] == NODE_PROTOCOL_VERSION
    # No task engine is attached, so task operations are not advertised even
    # though the config lists them.
    assert payload["capabilities"] == {"health": "1.0"}


@pytest.mark.asyncio
async def test_node_client_and_monitor(node_token: str) -> None:
    app = create_node_app(NodeConfig(name="vps", role="vps", endpoint="http://127.0.0.1:9"))
    transport = httpx.ASGITransport(app=app)
    inner = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:9")
    client = NodeClient(
        NodeConfig(name="vps", endpoint="http://127.0.0.1:9", allowed_operations=["health"]),
        client=inner,
    )
    result = await client.monitor()
    assert result["status"] == "ok"
    assert result["observed_capabilities"] == ["health"]
    from leo import db

    row = await db.fetchone(
        "SELECT status, capabilities, config_ref, verification FROM integrations WHERE name='vps'"
    )
    assert row["verification"] == "health-checked"
    assert row["capabilities"] == "health"
    assert row["config_ref"] == f"node-protocol:{NODE_PROTOCOL_VERSION}"
    await client.close()


@pytest.mark.asyncio
async def test_unadvertised_operation_is_rejected_before_http(node_token: str) -> None:
    http = FakeHttp({
        "node": "vps", "role": "vps", "status": "ok", "ts": "t",
        "protocol_version": NODE_PROTOCOL_VERSION,
        "capabilities": {"health": "1.0"},
    })
    client = NodeClient(NodeConfig(name="vps", endpoint="http://127.0.0.1:9"), client=http)
    with pytest.raises(NodeCapabilityError, match="does not advertise"):
        await client.list_tasks()
    assert [url for _, url in http.calls] == ["http://127.0.0.1:9/health"]

    # Even a locally configured but unadvertised create must be refused.
    with pytest.raises(NodeCapabilityError, match="does not advertise"):
        await client.create_task("unreachable")
    assert all(not url.endswith("/tasks") for _, url in http.calls)


@pytest.mark.asyncio
async def test_incompatible_protocol_blocks_operations(node_token: str) -> None:
    http = FakeHttp({
        "node": "vps", "role": "vps", "status": "ok", "ts": "t",
        "protocol_version": "9.9",
        "capabilities": {"health": "1.0", "tasks:read": "1.0"},
    })
    client = NodeClient(NodeConfig(name="vps", endpoint="http://127.0.0.1:9"), client=http)
    with pytest.raises(NodeCapabilityError, match="protocol"):
        await client.list_tasks()
    assert len(http.calls) == 1  # only the negotiation request


@pytest.mark.asyncio
async def test_monitor_persists_observed_capabilities_with_mismatch(node_token: str) -> None:
    from leo import db

    http = FakeHttp({
        "node": "vps", "role": "vps", "status": "ok", "ts": "2026-01-01T00:00:00+00:00",
        "protocol_version": "9.9", "capabilities": {"health": "1.0"},
    })
    client = NodeClient(NodeConfig(name="vps", endpoint="http://127.0.0.1:9"), client=http)
    result = await client.monitor()
    assert result["monitor_status"] == "mismatch"
    assert result["protocol_version"] == "9.9"
    row = await db.fetchone(
        "SELECT status, capabilities, config_ref, verification FROM integrations WHERE name='vps'"
    )
    assert row["status"] == "mismatch"
    assert row["verification"] == "protocol-mismatch"
    assert row["capabilities"] == "health"  # not the static config list
    assert row["config_ref"] == "node-protocol:9.9"


def test_rejects_plain_http_remote() -> None:
    with pytest.raises(ValueError, match="https"):
        NodeClient(NodeConfig(name="remote", endpoint="http://example.com"))
