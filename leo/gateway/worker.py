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
from collections.abc import Mapping
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
                       conversation_id: str = "",
                       reply_markup: dict[str, Any] | None = None,
                       edit_message_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"admission_id": admission_id, "answer": answer,
                                "conversation_id": conversation_id}
        if reply_markup:
            body["reply_markup"] = reply_markup
        if edit_message_id:
            body["edit_message_id"] = edit_message_id
        response = await self._http().post("/v1/complete", json=body)
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


DEFAULT_TURN_MAX_SECONDS = 1800.0


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


def _answer_markup(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The inline keyboard a slash command attached to its answer, if any."""
    command_result = result.get("command_result")
    if not isinstance(command_result, Mapping):
        return None
    data = command_result.get("data")
    if not isinstance(data, Mapping):
        return None
    markup = data.get("reply_markup")
    return dict(markup) if isinstance(markup, Mapping) else None


async def _abandon_turn(
    client: GatewayClient,
    admission_id: str,
    turn: "asyncio.Task[Any]",
    limit: float,
) -> None:
    """End a turn that outlived its ceiling, and say so in the chat.

    Letting the lease simply lapse is what made a long turn loop: the worker kept
    working while the queue handed the same admission to another worker, so every
    attempt restarted the job from scratch. Stopping the turn here keeps exactly
    one worker on the work, and the operator is told why it ended instead of
    watching a turn that never answers.
    """
    minutes = max(1, int(limit // 60))
    _LOG.warning(
        f"turn stopped at its {minutes}-minute limit: {admission_id}",
        extra={"admission_id": admission_id, "limit_seconds": limit},
    )
    turn.cancel()
    try:
        await asyncio.wait_for(turn, timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    except Exception:  # noqa: BLE001 - the turn is already abandoned; its error is noise
        pass
    note = (
        f"⏹ Stopped: this turn ran longer than the {minutes}-minute limit. "
        "Nothing was lost — ask again with a narrower scope, or raise "
        "`--turn-max-seconds` on the brain unit."
    )
    try:
        await client.complete(admission_id, note, "")
        return
    except (httpx.HTTPError, GatewayClientError):
        pass
    # The lease may already have lapsed, in which case the admission cannot be
    # completed. Bury it without a retry so it is not restarted from zero.
    try:
        await client.fail(admission_id, f"turn exceeded its {minutes}-minute limit", retry=False)
    except (httpx.HTTPError, GatewayClientError):
        pass


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
        turn = asyncio.create_task(
            app.ask(
                text,
                channel=str(admission.get("channel") or "gateway"),
                subject=str(admission.get("subject") or ""),
                conversation_id=admission.get("conversation_id") or None,
            ),
            name="leo-turn",
        )
        try:
            # Whichever finishes first decides the fate of the turn. Without this
            # race the ceiling only stopped RENEWING: the worker kept working,
            # another worker re-claimed the lapsed admission, and a long turn was
            # redone from scratch on every attempt — three attempts, no progress.
            finished, _pending = await asyncio.wait(
                {turn, keepalive}, return_when=asyncio.FIRST_COMPLETED
            )
            if turn not in finished:
                await _abandon_turn(client, admission_id, turn, turn_max_seconds)
                continue
            result = turn.result()
            answer = str(result.get("content") or "")
            if str(result.get("status") or "") == "cancelled":
                # A stopped turn is terminal. Bury the admission BEFORE delivery so
                # that a failed send cannot fall through to the retry path and
                # re-run the work the operator just cancelled — which is what made
                # a stopped turn start typing again.
                try:
                    await client.fail(admission_id, "cancelled at operator request",
                                      retry=False)
                except (httpx.HTTPError, GatewayClientError):
                    pass
            await client.complete(
                admission_id, answer,
                str(result.get("conversation_id") or ""),
                reply_markup=_answer_markup(result),
                edit_message_id=payload.get("source_message_id") or None,
            )
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
            for helper in (keepalive, turn):
                helper.cancel()
            await asyncio.gather(keepalive, turn, return_exceptions=True)
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
