"""Secure local/VPS node bridge.

Scope is deliberately narrow: health and task status/creation only. No remote
terminal, no unrestricted file access, no credential forwarding. Transport must
be HTTPS unless the peer is loopback, and authentication uses a bearer token
whose *name* is configured (the value is read from the environment at call time).
"""

from __future__ import annotations

import hmac
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..logging import get_logger

_LOG = get_logger("node")

# Capability negotiation is explicit: both ends must agree on the wire format.
# Each advertised operation is versioned independently so a node can add an
# operation without silently redefining an existing one.
NODE_PROTOCOL_VERSION = "1.0"
CAPABILITY_VERSIONS: dict[str, str] = {
    "health": "1.0",
    "tasks:read": "1.0",
    "tasks:create": "1.0",
}


class NodeCapabilityError(RuntimeError):
    """Requested operation was not advertised by the peer (or protocol differs)."""


class NodeConfig(BaseModel):
    name: str = "local"
    role: str = "local"  # local | vps
    endpoint: str | None = None
    token_env: str = "LEO_NODE_TOKEN"
    allowed_operations: list[str] = Field(default_factory=lambda: ["health", "tasks:read", "tasks:create"])


class NodeTaskRequest(BaseModel):
    objective: str
    plan: list[str] | None = None


def _check_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("endpoint must be an http(s) URL")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("non-loopback endpoints require https")


def _advertised_capabilities(config: NodeConfig, tasks: Any = None) -> dict[str, str]:
    """Versioned operations this node actually offers, not merely config claims."""
    available = {"health"}
    if tasks is not None:
        available.update({"tasks:read", "tasks:create"})
    return {
        operation: version
        for operation, version in CAPABILITY_VERSIONS.items()
        if operation in available and (operation == "health" or operation in config.allowed_operations)
    }


def create_node_app(config: NodeConfig, tasks: Any = None) -> FastAPI:
    """Create a narrowly scoped node API. `tasks` is an optional task engine."""
    app = FastAPI(title=f"LEO node: {config.name}", version="0.1.0")

    async def authenticate(authorization: str | None = Header(default=None)) -> None:
        token = os.getenv(config.token_env)
        if not token:
            raise HTTPException(503, "node token not configured")
        expected = f"Bearer {token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(401, "unauthorized")

    @app.get("/health")
    async def health(_: None = Depends(authenticate)) -> dict[str, Any]:
        return {
            "node": config.name,
            "role": config.role,
            "status": "ok",
            "ts": datetime.now(timezone.utc).isoformat(),
            "protocol_version": NODE_PROTOCOL_VERSION,
            "capabilities": _advertised_capabilities(config, tasks),
        }

    @app.get("/tasks")
    async def list_tasks(_: None = Depends(authenticate)) -> dict[str, Any]:
        if "tasks:read" not in config.allowed_operations or tasks is None:
            raise HTTPException(403, "operation not allowed")
        return {"tasks": [t.model_dump(mode="json") for t in await tasks.list()]}

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str, _: None = Depends(authenticate)) -> dict[str, Any]:
        if "tasks:read" not in config.allowed_operations or tasks is None:
            raise HTTPException(403, "operation not allowed")
        task = await tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "task not found")
        return task.model_dump(mode="json")

    @app.post("/tasks")
    async def create_task(req: NodeTaskRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        if "tasks:create" not in config.allowed_operations or tasks is None:
            raise HTTPException(403, "operation not allowed")
        task = await tasks.create(req.objective, plan=req.plan)
        return task.model_dump(mode="json")

    return app


class NodeClient:
    def __init__(self, config: NodeConfig, client: httpx.AsyncClient | None = None) -> None:
        if not config.endpoint:
            raise ValueError("remote node requires an endpoint")
        _check_endpoint(config.endpoint)
        self.config = config
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self.observed_capabilities: dict[str, str] = {}
        self.protocol_version: str | None = None
        self._negotiated = False

    def _headers(self) -> dict[str, str]:
        token = os.getenv(self.config.token_env)
        if not token:
            raise RuntimeError(f"node token missing in env var {self.config.token_env}")
        return {"Authorization": f"Bearer {token}"}

    def _observe(self, payload: dict[str, Any]) -> None:
        advertised = payload.get("capabilities", {})
        if isinstance(advertised, dict):
            self.observed_capabilities = {str(k): str(v) for k, v in advertised.items()}
        elif isinstance(advertised, (list, tuple, set)):
            self.observed_capabilities = {str(op): str(payload.get("protocol_version", "unknown")) for op in advertised}
        else:
            self.observed_capabilities = {}
        self.protocol_version = payload.get("protocol_version") or payload.get("protocol")
        self._negotiated = True

    async def health(self) -> dict[str, Any]:
        response = await self._client.get(f"{self.config.endpoint}/health", headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        self._observe(payload)
        return payload

    async def _require(self, operation: str) -> None:
        """Refuse unadvertised operations before issuing the operation request."""
        if operation not in self.config.allowed_operations:
            raise NodeCapabilityError(f"operation {operation!r} is not enabled in node config")
        if not self._negotiated:
            await self.health()
        if self.protocol_version != NODE_PROTOCOL_VERSION:
            raise NodeCapabilityError(
                f"node protocol {self.protocol_version!r} is incompatible with {NODE_PROTOCOL_VERSION!r}"
            )
        if operation not in self.observed_capabilities:
            raise NodeCapabilityError(
                f"node {self.config.name} does not advertise {operation!r}; "
                f"advertised: {sorted(self.observed_capabilities)}"
            )

    async def list_tasks(self) -> list[dict[str, Any]]:
        await self._require("tasks:read")
        response = await self._client.get(f"{self.config.endpoint}/tasks", headers=self._headers())
        response.raise_for_status()
        return response.json()["tasks"]

    async def create_task(self, objective: str, plan: list[str] | None = None) -> dict[str, Any]:
        await self._require("tasks:create")
        response = await self._client.post(
            f"{self.config.endpoint}/tasks", headers=self._headers(), json={"objective": objective, "plan": plan}
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()

    async def monitor(self) -> dict[str, Any]:
        """Persist the *observed* health snapshot; never claim static config is verified."""
        result = await self.health()
        observed = sorted(self.observed_capabilities)
        expected = set(self.config.allowed_operations) | {"health"}
        protocol_mismatch = self.protocol_version != NODE_PROTOCOL_VERSION
        missing = sorted(expected - set(observed))
        unhealthy = result.get("status") != "ok"
        mismatch = protocol_mismatch or bool(missing) or unhealthy
        status = "mismatch" if mismatch else str(result.get("status", "unknown"))
        if protocol_mismatch:
            verification = "protocol-mismatch"
        elif missing or unhealthy:
            verification = "capability-mismatch"
        else:
            verification = "health-checked"
        ts = str(result.get("ts") or datetime.now(timezone.utc).isoformat())
        await db.execute(
            "INSERT INTO integrations(name,kind,status,capabilities,config_ref,verification,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET status=excluded.status, "
            "capabilities=excluded.capabilities, config_ref=excluded.config_ref, "
            "verification=excluded.verification, updated_at=excluded.updated_at",
            (
                self.config.name, "node", status, ",".join(observed),
                f"node-protocol:{self.protocol_version}", verification, ts, ts,
            ),
        )
        _LOG.info(
            "node health recorded",
            extra={"event": "node.monitor", "node": self.config.name, "status": status},
        )
        return {
            **result,
            "observed_capabilities": observed,
            "protocol_version": self.protocol_version,
            "monitor_status": status,
            "verification": verification,
        }
