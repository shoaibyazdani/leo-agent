"""The brain process: ``leo serve --worker`` claims turns from the gateway.

The worker holds no adapters and no transport state. It claims an admission,
runs the turn with the same ``LeoApp`` the single-process install uses, and
reports the answer back so the gateway can deliver it. That is the whole point
of the split: the brain can be restarted mid-turn without the transport noticing.

A worker that dies mid-turn leaves a lease behind; the gateway's claim path
reclaims it after the expiry, so the turn runs again rather than vanishing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Any

import httpx

from .server import token_path

_LOG = logging.getLogger(__name__)


class GatewayClientError(RuntimeError):
    """The gateway refused or was unreachable."""


class GatewayClient:
    """Thin HTTP client for the gateway's local API."""

    def __init__(self, base_url: str, token: str, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client = client

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(60.0, connect=5.0),
            )
        # Set per call, not at construction: a worker's token changes from the
        # placeholder it boots with to the session token it exchanges for.
        self._client.headers["Authorization"] = f"Bearer {self.token}"
        return self._client

    async def health(self) -> dict[str, Any]:
        response = await self._http().get("/health")
        if response.status_code != 200:
            raise GatewayClientError(f"health failed: HTTP {response.status_code}")
        return response.json()

    async def claim(self, worker: str, *, lease_seconds: int = 60,
                    wait_seconds: float = 25.0) -> dict[str, Any] | None:
        response = await self._http().post(
            "/v1/claim",
            json={"worker": worker, "lease_seconds": lease_seconds,
                  "wait_seconds": wait_seconds},
        )
        if response.status_code != 200:
            raise GatewayClientError(f"claim failed: HTTP {response.status_code}")
        return response.json().get("admission")

    async def complete(self, admission_id: str, answer: str,
                       conversation_id: str = "") -> dict[str, Any]:
        response = await self._http().post(
            "/v1/complete",
            json={"admission_id": admission_id, "answer": answer,
                  "conversation_id": conversation_id},
        )
        if response.status_code != 200:
            raise GatewayClientError(f"complete failed: HTTP {response.status_code}")
        return response.json()

    async def redeem_ticket(self, ticket: str) -> str:
        """Exchange a single-use attach ticket for a session token."""
        response = await self._http().post("/v1/session", json={"ticket": ticket})
        if response.status_code != 200:
            raise GatewayClientError(
                f"attach ticket rejected: HTTP {response.status_code}"
            )
        token = response.json().get("session_token")
        if not token:
            raise GatewayClientError("attach ticket produced no session token")
        return str(token)

    async def renew(self, admission_id: str, worker: str,
                    lease_seconds: int = 30) -> bool:
        response = await self._http().post(
            "/v1/renew",
            json={"admission_id": admission_id, "worker": worker,
                  "lease_seconds": lease_seconds},
        )
        if response.status_code != 200:
            return False
        return bool(response.json().get("renewed"))

    async def fail(self, admission_id: str, error: str, *, retry: bool = True) -> dict[str, Any]:
        response = await self._http().post(
            "/v1/fail", json={"admission_id": admission_id, "error": error, "retry": retry}
        )
        if response.status_code != 200:
            raise GatewayClientError(f"fail failed: HTTP {response.status_code}")
        return response.json()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def resolve_token(token: str | None = None) -> str:
    """Env wins, then the file the gateway wrote on first run."""
    if token:
        return token
    from_env = os.getenv("LEO_GATEWAY_TOKEN")
    if from_env:
        return from_env
    try:
        with open(token_path(), "r", encoding="utf-8") as handle:
            stored = handle.read().strip()
    except FileNotFoundError:
        raise GatewayClientError(
            "no gateway token: start the gateway once, or set LEO_GATEWAY_TOKEN"
        ) from None
    if not stored:
        raise GatewayClientError("gateway token file is empty")
    return stored


DEFAULT_TURN_MAX_SECONDS = 600.0


async def _keep_lease(
    client: GatewayClient,
    admission_id: str,
    worker: str,
    lease_seconds: int,
    max_seconds: float = DEFAULT_TURN_MAX_SECONDS,
) -> None:
    """Renew a lease while a turn runs, but never past ``max_seconds``.

    Renewal exists so a slow turn is not mistaken for a dead one. Without a
    ceiling it does the opposite: a turn that is genuinely stuck keeps renewing
    forever, so it can never expire, never gets retried, and from the outside
    looks exactly like a busy worker. After the ceiling the renewals stop, the
    lease runs out and the queue hands the work to someone else with the attempt
    counted.
    """
    interval = max(1.0, lease_seconds / 3)
    deadline = asyncio.get_running_loop().time() + max(0.0, float(max_seconds))
    while True:
        await asyncio.sleep(interval)
        if asyncio.get_running_loop().time() >= deadline:
            _LOG.warning(
                "turn exceeded its maximum lifetime; letting the lease expire",
                extra={"admission_id": admission_id, "max_seconds": max_seconds},
            )
            return
        try:
            await client.renew(admission_id, worker, lease_seconds)
        except (httpx.HTTPError, GatewayClientError):
            continue  # a transient gateway blip must not abort the turn


async def _claim_loop(
    client: GatewayClient,
    app: Any,
    me: str,
    *,
    wait_seconds: float,
    lease_seconds: int,
    once: bool,
    turn_max_seconds: float,
) -> int:
    """One claim/answer loop. Several of these share a client and a brain."""
    answered = 0
    while True:
        try:
            admission = await client.claim(me, lease_seconds=lease_seconds,
                                           wait_seconds=wait_seconds)
        except (httpx.HTTPError, GatewayClientError) as exc:
            _LOG.warning("gateway unreachable, retrying", extra={"error": str(exc)})
            await asyncio.sleep(2.0)
            continue
        if admission is None:
            if once:
                return answered
            continue

        payload = admission.get("payload") or {}
        text = str(payload.get("text") or "")
        admission_id = str(admission.get("admission_id") or "")
        keepalive = asyncio.create_task(
            _keep_lease(client, admission_id, me, lease_seconds, turn_max_seconds),
            name="leo-lease",
        )
        try:
            result = await app.ask(
                text,
                channel=str(admission.get("channel") or "gateway"),
                subject=str(admission.get("subject") or ""),
                conversation_id=admission.get("conversation_id") or None,
            )
            answer = str(result.get("content") or "")
            await client.complete(admission_id, answer,
                                  str(result.get("conversation_id") or ""))
            answered += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a broken turn must not stop the worker
            _LOG.warning("turn failed", extra={"admission_id": admission_id,
                                               "error": str(exc)})
            try:
                await client.fail(admission_id, f"{type(exc).__name__}: {exc}")
            except (httpx.HTTPError, GatewayClientError):
                pass
        finally:
            keepalive.cancel()
            await asyncio.gather(keepalive, return_exceptions=True)
        if once:
            return answered


async def run_worker(
    base_url: str,
    *,
    token: str | None = None,
    wait_seconds: float = 25.0,
    lease_seconds: int = 30,
    once: bool = False,
    brain: Any = None,
    concurrency: int = 1,
    turn_max_seconds: float = DEFAULT_TURN_MAX_SECONDS,
) -> int:
    """Claim and answer turns until cancelled. Returns the number answered.

    ``concurrency`` runs that many claim loops against one brain. A single loop
    is head-of-line blocking: one long turn leaves every later message queued,
    which looks to a user exactly like the bot being down.
    """
    from .. import db
    from ..core.app import LeoApp

    # A worker must use the database the operator pointed at, not whatever path
    # the settings file happens to resolve to: a split install that quietly fans
    # out into two databases loses turns between them.
    if not db.database_path_configured():
        explicit = os.getenv("LEO_DB_PATH")
        if explicit:
            db.set_db_path(explicit)

    # A worker should never hold the master token: if a ticket is provided it is
    # exchanged for a session token, and that is all this process ever carries.
    ticket = os.getenv("LEO_GATEWAY_TICKET")
    client = GatewayClient(base_url, token or "unauthenticated")
    if ticket:
        client.token = await client.redeem_ticket(ticket)
    elif not token:
        client.token = resolve_token(None)
    app = brain or LeoApp()
    await app.initialize()
    me = worker_id()

    loops = [
        asyncio.create_task(
            _claim_loop(client, app, f"{me}:{index + 1}",
                        wait_seconds=wait_seconds, lease_seconds=lease_seconds,
                        once=once, turn_max_seconds=turn_max_seconds),
            name=f"leo-claim-{index + 1}",
        )
        for index in range(max(1, int(concurrency)))
    ]
    try:
        results = await asyncio.gather(*loops, return_exceptions=True)
    finally:
        for task in loops:
            task.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
        await client.aclose()

    return sum(result for result in results if isinstance(result, int))
