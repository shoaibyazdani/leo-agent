"""The gateway split: durable queue between transport and brain.

Transport and brain are separate processes, so the queue between them is the only
thing holding a user's message. These tests pin the guarantees that make the split
worth having: an admission is recorded once, a claim is a lease that comes back
when a worker dies, a poison payload fails loudly instead of looping, and a
completed turn is terminal.
"""

from __future__ import annotations

import asyncio
import sqlite3

import httpx
import pytest

from leo.gateway import GatewayStore
from leo.gateway.server import create_api


def payload(text: str = "hello", mid: str = "1") -> dict:
    return {"text": text, "message_id": mid, "sender_id": "100000001",
            "conversation_ref": "100000001"}


async def test_admission_is_recorded_once():
    store = GatewayStore()

    first, created = await store.admit(channel="telegram", subject="1", payload=payload())
    again, created_again = await store.admit(channel="telegram", subject="1", payload=payload())

    assert created is True
    assert created_again is False
    assert first == again            # same admission, not a second one
    assert (await store.stats())["queue"] == {"queued": 1}


async def test_claim_is_a_lease_and_a_second_claim_gets_nothing():
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload())

    claimed = await store.claim("worker-a", lease_seconds=60)

    assert claimed is not None
    assert claimed["claimed_by"] == "worker-a"
    assert await store.claim("worker-b", lease_seconds=60) is None


async def test_an_expired_lease_comes_back_to_the_queue():
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("dead-worker", lease_seconds=-1)  # already expired

    reclaimed = await store.reclaim_expired()
    claimed_again = await store.claim("worker-b", lease_seconds=60)

    # The ids matter, not the count: the caller stops one typing task per id.
    assert len(reclaimed) == 1 and reclaimed[0].startswith("adm_")
    assert claimed_again is not None
    assert claimed_again["claimed_by"] == "worker-b"


async def test_failure_requeues_then_buries_after_max_attempts():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())

    await store.claim("worker", lease_seconds=60)
    queued_again = await store.fail(admission_id, "boom", retry=True, max_attempts=2)

    await store.claim("worker", lease_seconds=60)
    buried = await store.fail(admission_id, "boom", retry=True, max_attempts=2)

    assert queued_again == "queued"     # first failure: worth another try
    assert buried == "failed"           # attempts exhausted: stop looping
    assert await store.claim("worker", lease_seconds=60) is None


async def test_complete_is_terminal():
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("worker", lease_seconds=60)

    await store.complete(admission_id)

    record = await store.admission(admission_id)
    assert record is not None and record["status"] == "done"
    assert await store.claim("worker", lease_seconds=60) is None


async def test_routing_pins_a_subject_to_one_conversation():
    store = GatewayStore()

    assert await store.route("telegram", "1") is None
    await store.save_route("telegram", "1", "conv_abc")
    assert await store.route("telegram", "1") == "conv_abc"
    await store.save_route("telegram", "1", "conv_def")   # /new switches it
    assert await store.route("telegram", "1") == "conv_def"


async def test_turn_lease_excludes_another_holder():
    store = GatewayStore()

    assert await store.acquire_turn("conv_1", "worker-a", ttl_seconds=60) is True
    assert await store.acquire_turn("conv_1", "worker-b", ttl_seconds=60) is False
    await store.release_turn("conv_1", "worker-a")
    assert await store.acquire_turn("conv_1", "worker-b", ttl_seconds=60) is True


class _FakeAdapter:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.edits: list[tuple[str, str, str, dict]] = []
        self.markup: dict | None = None

    async def send(self, target: str, text: str, **kwargs) -> dict:
        self.sent.append((target, text))
        self.markup = kwargs.get("reply_markup")
        return {"ok": True}

    async def edit_message_text(self, target, message_id, text, reply_markup=None) -> bool:
        self.edits.append((str(target), str(message_id), str(text), dict(reply_markup or {})))
        return True


class _CardAdapter(_FakeAdapter):
    """Transport with the native callback + terminal rewrite proof."""

    supports_approval_cards = True
    supports_full_text = True

    async def send(self, target: str, text: str, **kwargs) -> dict:
        await super().send(target, text, **kwargs)
        return {"ok": True, "message_id": "m-1"}


def _api(store: GatewayStore, token: str = "t0ken", adapters: dict | None = None):
    return create_api(store, token=token, adapters=adapters if adapters is not None else {})


def _client(api, token: str = "t0ken") -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_claim_and_complete_delivers_through_the_adapter():
    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="100000001",
                                       payload=payload())

    async with _client(api) as client:
        claimed = (await client.post("/v1/claim", json={"worker": "w1"})).json()["admission"]
        assert claimed["admission_id"] == admission_id
        done = (await client.post("/v1/complete", json={
            "admission_id": admission_id, "answer": "hi back",
            "conversation_id": "conv_xyz",
        })).json()

    assert done["delivered"] is True
    assert adapter.sent == [("100000001", "hi back")]
    assert await store.route("telegram", "100000001") == "conv_xyz"


async def test_api_rejects_a_bad_token():
    api = _api(GatewayStore())
    async with _client(api, token="wrong") as client:
        response = await client.post("/v1/claim", json={"worker": "w1"})
    assert response.status_code == 401


async def test_health_reports_queue_and_cursor():
    store = GatewayStore()
    await store.admit(channel="telegram", subject="1", payload=payload())
    async with _client(_api(store)) as client:
        body = (await client.get("/health")).json()
    assert body["ok"] is True
    assert body["queue"] == {"queued": 1}


async def test_worker_answers_a_claimed_turn_once():
    """One worker, one admission, one answer — with no adapters in the brain."""
    from leo.gateway import worker as worker_module

    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    await store.admit(channel="telegram", subject="100000001", payload=payload("ping"))

    class _Brain:
        def __init__(self) -> None:
            self.asked: list[tuple] = []

        async def initialize(self) -> None:
            return None

        async def ask(self, text, channel=None, subject=None, conversation_id=None):
            self.asked.append((text, channel, subject, conversation_id))
            return {"content": "pong", "conversation_id": "conv_worker"}

    real_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": "Bearer t0ken"},
    )
    original = worker_module.GatewayClient

    class _Client(original):
        def __init__(self, base_url, token, client=None):
            super().__init__(base_url, token, client=real_client)

    worker_module.GatewayClient = _Client
    brain = _Brain()
    try:
        answered = await worker_module.run_worker(
            "http://gateway", token="t0ken", wait_seconds=0.0, once=True, brain=brain
        )
    finally:
        worker_module.GatewayClient = original
        await real_client.aclose()

    assert answered == 1
    assert brain.asked and brain.asked[0][0] == "ping"
    assert adapter.sent == [("100000001", "pong")]
    assert (await store.admission_by_status("done")) is not None


class _TypingAdapter(_FakeAdapter):
    """Counts typing pulses so a leaked one is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.pulses = 0

    async def send_typing(self, target: str) -> dict:
        self.pulses += 1
        return {"ok": True}


async def test_reclaim_reports_a_stranded_claim_so_its_typing_pulse_can_stop():
    """The leak that made a finished agent look like it was still working.

    An admission can end in a non-retryable state (`cancelled`) while still
    holding a claim row. `reclaim_expired` only requeued `claimed` rows, so such
    a row kept its typing task alive forever: nothing else ever posts
    /v1/complete or /v1/fail for it. The ids must be reported even though the
    row is not requeued, because the pulse belongs to this process.
    """
    from leo import db as _db

    store = GatewayStore()
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001", payload=payload()
    )
    await store.claim("dead-worker", lease_seconds=1)
    # A stop path ends the admission but leaves the claim columns behind.
    await _db.execute(
        "UPDATE gateway_queue SET status='cancelled', claim_expires_at=? "
        "WHERE admission_id=?",
        ("2020-01-01T00:00:00+00:00", admission_id),
    )

    reclaimed = await store.reclaim_expired()

    assert admission_id in reclaimed, "stranded claim was not reported, so its typing task leaks"
    # It must not be requeued: cancelled work is terminal.
    row = await _db.fetchone(
        "SELECT status FROM gateway_queue WHERE admission_id=?", (admission_id,)
    )
    assert row["status"] == "cancelled"


async def test_a_settled_claim_is_reported_once_and_lets_go_of_its_claim() -> None:
    """Clearing the pulse must also settle the row.

    A terminal row that kept `claim_expires_at` matched the next poll again, so the
    same dead id called `_stop_typing` about once a second for as long as the row
    lived. Reporting once is the whole point: the task is stopped, then forgotten.
    """
    from leo import db as _db

    store = GatewayStore()
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000002", payload=payload()
    )
    await store.claim("dead-worker", lease_seconds=1)
    await _db.execute(
        "UPDATE gateway_queue SET status='cancelled', claim_expires_at=? "
        "WHERE admission_id=?",
        ("2020-01-01T00:00:00+00:00", admission_id),
    )

    first = await store.reclaim_expired()
    second = await store.reclaim_expired()

    assert admission_id in first
    assert admission_id not in second, "a settled claim must not be reported again"
    row = await _db.fetchone(
        "SELECT status, claim_expires_at, claimed_by FROM gateway_queue WHERE admission_id=?",
        (admission_id,),
    )
    assert row["status"] == "cancelled", "terminal state must not be rewritten"
    assert row["claim_expires_at"] is None
    assert row["claimed_by"] is None


async def test_an_answer_can_carry_an_inline_keyboard():
    """/model sends its picker with the answer, so the buttons ride delivery."""
    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="100000001",
                                        payload=payload())
    keyboard = {"inline_keyboard": [
        [{"text": "beam-free", "callback_data": "mdl:beam-free"}],
    ]}

    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1"})
        done = (await client.post("/v1/complete", json={
            "admission_id": admission_id, "answer": "pick one",
            "conversation_id": "conv_pick", "reply_markup": keyboard,
        })).json()

    assert done["delivered"] is True
    assert adapter.sent == [("100000001", "pick one")]
    assert adapter.markup == keyboard


async def test_an_answer_without_a_keyboard_keeps_a_text_only_adapter_working():
    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="100000001",
                                        payload=payload())
    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1"})
        done = (await client.post("/v1/complete", json={
            "admission_id": admission_id, "answer": "plain",
        })).json()

    assert done["delivered"] is True
    assert adapter.markup is None


async def test_an_answer_can_replace_the_tapped_menu_card():
    """A reasoning tap edits its own card; it must not post a second full menu."""
    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001",
        payload=payload(text="/reasoning high", mid="cb-1"),
    )
    keyboard = {"inline_keyboard": [[{"text": "medium", "callback_data": "rsn:level:medium"}]]}

    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1"})
        done = (await client.post("/v1/complete", json={
            "admission_id": admission_id,
            "answer": "Reasoning effort: high",
            "conversation_id": "conv_reasoning",
            "reply_markup": keyboard,
            "edit_message_id": "77",
        })).json()

    assert done["delivered"] is True
    assert adapter.sent == []
    # The supplied picker is not reattached: a consumed tap becomes text-only.
    assert adapter.edits == [
        ("100000001", "77", "Reasoning effort: high", {"inline_keyboard": []})
    ]


async def test_thread_principal_and_reply_target_survive_admission_and_the_worker():
    """The gateway hands the brain the admitted identity, not a channel guess."""
    from leo.gateway import worker as worker_module

    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram",
        subject="routing-key",
        principal="sender-9",
        thread_id="topic-7",
        reply_target="chat-target",
        payload=payload("ping", "m-thread"),
    )

    class _Brain:
        def __init__(self) -> None:
            self.turn = None

        async def initialize(self) -> None:
            return None

        async def ask(self, text, channel=None, subject=None, conversation_id=None, *, turn=None):
            self.turn = turn
            return {"content": "pong", "conversation_id": "conv_thread"}

    real_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://gateway",
        headers={"Authorization": "Bearer t0ken"},
    )
    original = worker_module.GatewayClient

    class _Client(original):
        def __init__(self, base_url, token, client=None):
            super().__init__(base_url, token, client=real_client)

    worker_module.GatewayClient = _Client
    brain = _Brain()
    try:
        async with _client(api) as client:
            claimed = (
                await client.post("/v1/claim", json={"worker": "w1"})
            ).json()["admission"]
            assert claimed["principal"] == "sender-9"
            assert claimed["thread_id"] == "topic-7"
            assert claimed["reply_target"] == "chat-target"
            assert claimed["conversation_id"] is None
            # Put the admission back so the worker can claim it after the probe.
            await client.post("/v1/fail", json={
                "admission_id": admission_id, "error": "probe", "retry": True,
            })
        answered = await worker_module.run_worker(
            "http://gateway", token="t0ken", wait_seconds=0.0, once=True, brain=brain
        )
    finally:
        worker_module.GatewayClient = original
        await real_client.aclose()

    assert answered == 1
    assert brain.turn is not None
    assert brain.turn.principal == "sender-9"
    assert brain.turn.thread_id == "topic-7"
    assert brain.turn.target == "chat-target"
    assert brain.turn.source == "gateway"
    assert brain.turn.channel == "telegram"
    # Delivery goes to the admission's reply target, never the routing subject.
    assert adapter.sent == [("chat-target", "pong")]
    assert await store.route("telegram", "routing-key") == "conv_thread"


async def test_complete_rejects_forged_sender_identity_and_ignores_body_target():
    store = GatewayStore()
    adapter = _FakeAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram",
        subject="routing-key",
        principal="sender-9",
        thread_id="topic-7",
        reply_target="chat-target",
        payload=payload("ping", "m-identity"),
    )
    async with _client(api) as client:
        await client.post("/v1/claim", json={"worker": "w1"})
        wrong_principal = await client.post("/v1/complete", json={
            "admission_id": admission_id,
            "answer": "spoofed",
            "principal": "someone-else",
        })
        wrong_thread = await client.post("/v1/complete", json={
            "admission_id": admission_id,
            "answer": "spoofed-thread",
            "thread_id": "other-topic",
        })
        done = (await client.post("/v1/complete", json={
            "admission_id": admission_id,
            "answer": "real answer",
            "principal": "sender-9",
            "thread_id": "topic-7",
            # A body cannot redirect the reply target.
            "reply_target": "evil-target",
        })).json()

    assert wrong_principal.status_code == 409
    assert wrong_thread.status_code == 409
    assert done["delivered"] is True
    assert adapter.sent == [("chat-target", "real answer")]


# ---------------------------------------------------------------------------
# Phase 9: schedule cards across the split gateway


def _schedule_permissions(settings=None):
    from leo.config import Settings
    from leo.security.permissions import PermissionManagerImpl

    settings = settings or Settings()
    settings.security.approval_poll_seconds = 0.01
    return PermissionManagerImpl(settings), settings


async def _schedule_card(permissions, body='Schedule "Water" (daily): Drink water.'):
    return await permissions.create_schedule_card(
        kind="schedule_create", body=body, proposal_digest="gw-card",
        requested_by="chat:telegram", conversation_id="conv-gw",
        origin="gateway:telegram",
    )


def _schedule_service(permissions, settings, clock):
    """A service wired to the same durable store the gateway reads."""
    from leo.events import EventBus
    from leo.schedules.clock import FrozenClock
    from leo.schedules.service import ScheduleService
    from leo.schedules.store import ScheduleStore
    from leo.tasks import DurableTaskEngine, Scheduler

    events = EventBus()
    engine = DurableTaskEngine(settings, events)
    scheduler = Scheduler(engine, settings, events, clock=clock)
    store = ScheduleStore(clock=clock)
    return ScheduleService(
        store=store, scheduler=scheduler, engine=engine, permissions=permissions,
        events=events, settings=settings, clock=clock,
    )


def _gateway_turn() -> object:
    from leo.contracts import TurnContext

    return TurnContext(
        principal="chat:telegram", channel="telegram", surface="telegram",
        conversation_id="conv-gw", source="chat",
    )


async def _independent_approval_row(approval_id: str):
    """Read with a fresh connection: inside a write txn this sees no row."""
    import aiosqlite

    from leo import db

    conn = await aiosqlite.connect(db.get_db_path(), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        return await (
            await conn.execute(
                "SELECT id, status, shown_at, digest FROM approvals WHERE id=?",
                (approval_id,),
            )
        ).fetchone()
    finally:
        await conn.close()


async def test_create_commits_proposal_and_card_before_split_gateway_presentation() -> None:
    """The gateway's second SQLite connection must see the card during create."""
    from datetime import datetime, timezone

    from leo import db
    from leo.gateway.server import GatewayStore
    from leo.schedules.clock import FrozenClock
    from leo.security.approval_cards import presenting_cards_via

    permissions, settings = _schedule_permissions()
    clock = FrozenClock(datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    service = _schedule_service(permissions, settings, clock)
    store = GatewayStore()
    adapter = _CardAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001", payload=payload("schedule it"),
        principal="100000001", thread_id="77",
    )

    seen: list[str] = []
    async with _client(api) as client:
        await store.claim("worker", lease_seconds=60)

        async def present(approval_id: str, surface: str) -> bool:
            row = await _independent_approval_row(approval_id)
            assert row is not None, "approval not visible to a second connection"
            assert row["status"] == "pending"
            seen.append(str(row["digest"]))
            response = await client.post("/v1/approval/present", json={
                "admission_id": admission_id, "approval_id": approval_id,
                "confirm_proposal_digest": row["digest"],
            })
            assert response.status_code == 200, response.text
            return True

        with presenting_cards_via(present):
            out = await service.create(
                {"name": "water", "schedule": "every day at 8am", "prompt": "Drink water."},
                actor="chat:telegram", turn=_gateway_turn(),
            )

    assert out["status"] == "approval_required"
    card = out["card"]
    assert seen == [card["proposal_digest"]]
    assert len(adapter.sent) == 1 and card["approval_id"] in adapter.sent[0][1]
    row = await db.fetchone(
        "SELECT status, shown_at FROM approvals WHERE id=?", (card["approval_id"],)
    )
    assert row["status"] == "pending" and row["shown_at"]
    proposal = await db.fetchone(
        "SELECT status FROM schedule_proposals WHERE id=?", (card["proposal_id"],)
    )
    assert proposal is not None and proposal["status"] == "pending"
    # The card is resolvable from another process once the human taps it.
    resolution = await permissions.resolve_schedule_card(
        card["approval_id"], choice="once", resolver="telegram-button:100000001",
        confirm_proposal_digest=card["proposal_digest"], commit=service.commit,
    )
    assert resolution.status == "approved"


async def test_create_preserves_the_proposal_when_the_presenter_fails() -> None:
    """A transport failure must not roll back the human's committed proposal."""
    from datetime import datetime, timezone

    from leo import db
    from leo.schedules.clock import FrozenClock
    from leo.security.approval_cards import presenting_cards_via

    permissions, settings = _schedule_permissions()
    clock = FrozenClock(datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    service = _schedule_service(permissions, settings, clock)
    observed: list[str] = []

    async def broken_presenter(approval_id: str, surface: str) -> bool:
        row = await _independent_approval_row(approval_id)
        assert row is not None, "approval not visible to a second connection"
        observed.append(str(row["status"]))
        raise RuntimeError("gateway is down")

    with presenting_cards_via(broken_presenter):
        with pytest.raises(PermissionError, match="card presentation failed"):
            await service.create(
                {"name": "water", "schedule": "every day at 8am", "prompt": "Drink water."},
                actor="chat:telegram", turn=_gateway_turn(),
            )

    assert observed == ["pending"]
    approval = await db.fetchone("SELECT status, shown_at FROM approvals")
    proposal = await db.fetchone("SELECT status, active_name FROM schedule_proposals")
    assert approval is not None and approval["status"] == "pending"
    assert approval["shown_at"] is None
    assert proposal is not None and proposal["active_name"] == "water"


async def test_health_advertises_the_card_protocol_only_for_capable_channels() -> None:
    from leo.gateway.server import APPROVAL_RPC_PROTOCOL
    from leo.integrations.messaging import TelegramAdapter

    api = _api(GatewayStore(), adapters={"telegram": TelegramAdapter(require_verification=False)})
    async with _client(api) as client:
        body = (await client.get("/health")).json()
    assert body["approval_protocol"] == APPROVAL_RPC_PROTOCOL
    assert body["approval_cards"]["telegram"]["can_present_card"] is True
    assert body["approval_cards"]["telegram"]["terminal_edit"] is True


async def test_present_ack_and_update_rewrite_the_full_card() -> None:
    from leo import db
    from leo.security.approval_cards import schedule_terminal_text

    permissions, _ = _schedule_permissions()
    card = await _schedule_card(permissions)
    store = GatewayStore()
    adapter = _CardAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001", payload=payload(),
        principal="100000001", thread_id="77",
    )
    await store.claim("worker", lease_seconds=60)

    async with _client(api) as client:
        bad = await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "wrong",
        })
        assert bad.status_code == 409
        presented = await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
            "principal": "100000001", "thread_id": "77",
        })
        assert presented.status_code == 200
        assert presented.json()["presented"] is True
        acked = await client.post("/v1/approval/ack", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
        })
        assert acked.json()["acked"] is True

    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target == "100000001"
    assert card.body in text and card.deadline in text
    buttons = [b for row in (adapter.markup or {})["inline_keyboard"] for b in row]
    assert len(buttons) == 4
    # The gateway only sends; presentation is still recorded by the worker.
    row = await db.fetchone("SELECT shown_at FROM approvals WHERE id=?", (card.approval_id,))
    assert row["shown_at"] is None

    await permissions.present_schedule_card(card.approval_id, surface="telegram")
    resolution = await permissions.resolve_schedule_card(
        card.approval_id, choice="once", resolver="telegram-button:100000001",
        confirm_proposal_digest="gw-card",
    )
    assert resolution.status == "approved"
    updated = await permissions.get_schedule_card(card.approval_id)
    async with _client(api) as client:
        rewritten = await client.post("/v1/approval/update", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
            "text": schedule_terminal_text(updated, status=resolution.status),
        })
    assert rewritten.status_code == 200
    assert rewritten.json()["updated"] is True
    assert len(adapter.edits) == 1
    assert card.body in adapter.edits[0][2]
    assert adapter.edits[0][3] == {"inline_keyboard": []}


async def test_incapable_gateway_channel_and_mixed_version_get_no_card() -> None:
    from leo import db

    permissions, _ = _schedule_permissions()
    card = await _schedule_card(permissions)
    store = GatewayStore()
    admission_id, _ = await store.admit(channel="whatsapp", subject="1", payload=payload())
    await store.claim("worker", lease_seconds=60)

    api = _api(store, adapters={"whatsapp": _FakeAdapter()})
    async with _client(api) as client:
        refused = await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
        })
    assert refused.status_code == 409

    old_api = create_api(GatewayStore(), token="t0ken", adapters={}, approval_protocol=None)
    async with _client(old_api) as client:
        health = (await client.get("/health")).json()
        missing = await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
        })
    assert "approval_protocol" not in health and "approval_cards" not in health
    assert missing.status_code == 404
    row = await db.fetchone("SELECT shown_at FROM approvals WHERE id=?", (card.approval_id,))
    assert row["shown_at"] is None


async def test_worker_presents_through_gateway_waits_and_rewrites() -> None:
    """One brain, one gateway, one shared approval row: the split card path."""
    from leo import db
    from leo.gateway import worker as worker_module

    permissions, _ = _schedule_permissions()
    store = GatewayStore()
    adapter = _CardAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="100000001",
                                       payload=payload("schedule it"))

    class Brain:
        def __init__(self):
            self.permissions = permissions
            self.approval_id = ""
            self.capabilities = None

        async def initialize(self):
            return None

        async def ask(self, text, channel=None, subject=None, conversation_id=None, turn=None):
            self.capabilities = turn.capabilities
            card = await _schedule_card(self.permissions)
            self.approval_id = card.approval_id
            await self.permissions.present_schedule_card(
                card.approval_id, surface=turn.surface, capabilities=turn.capabilities,
            )
            return {"content": "waiting for the card", "conversation_id": "conv-gw"}

    brain = Brain()

    async def decide():
        for _ in range(500):
            row = await db.fetchone(
                "SELECT shown_at FROM approvals WHERE id=?", (brain.approval_id,)
            ) if brain.approval_id else None
            if row is not None and row["shown_at"]:
                # Stay pending long enough for the worker's lease renewal to run.
                await asyncio.sleep(1.2)
                await permissions.resolve_schedule_card(
                    brain.approval_id, choice="once",
                    resolver="telegram-button:100000001",
                    confirm_proposal_digest="gw-card",
                )
                return
            await asyncio.sleep(0.01)

    real_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://gateway",
        headers={"Authorization": "Bearer t0ken"},
    )
    original = worker_module.GatewayClient

    class _Client(original):
        renewals = 0

        def __init__(self, base_url, token, client=None):
            super().__init__(base_url, token, client=real_client)

        async def renew(self, *args, **kwargs):
            type(self).renewals += 1
            return await super().renew(*args, **kwargs)

    worker_module.GatewayClient = _Client
    decider = asyncio.create_task(decide())
    try:
        answered = await worker_module.run_worker(
            "http://gateway", token="t0ken", wait_seconds=0.0, once=True,
            lease_seconds=1, brain=brain
        )
    finally:
        worker_module.GatewayClient = original
        await asyncio.gather(decider, return_exceptions=True)
        await real_client.aclose()

    assert answered == 1
    assert brain.capabilities.can_present_card is True
    assert _Client.renewals >= 1, "the lease must renew while the card is pending"
    assert len(adapter.sent) == 2  # card, then the turn's final answer
    assert brain.approval_id in adapter.sent[0][1]
    row = await db.fetchone("SELECT status, shown_at FROM approvals WHERE id=?", (brain.approval_id,))
    assert row["status"] == "approved" and row["shown_at"]
    assert len(adapter.edits) == 1
    assert "approved" in adapter.edits[0][2]


def test_mixed_version_gateway_turn_is_capability_incapable() -> None:
    from leo.gateway.worker import admission_turn

    admission = {"channel": "telegram", "principal": "1", "subject": "1"}
    assert admission_turn(admission).capabilities.can_present_card is False


async def test_old_gateway_worker_never_presents_a_schedule_card() -> None:
    """A mixed-version gateway has no approval RPCs: the worker stays incapable."""
    from leo.gateway import worker as worker_module

    store = GatewayStore()
    adapter = _CardAdapter()
    api = create_api(store, token="t0ken", adapters={"telegram": adapter},
                     approval_protocol=None)
    await store.admit(channel="telegram", subject="100000001", payload=payload("hello"))

    class Brain:
        def __init__(self):
            self.capabilities = None

        async def initialize(self):
            return None

        async def ask(self, text, channel=None, subject=None, conversation_id=None, turn=None):
            self.capabilities = turn.capabilities
            return {"content": "ok", "conversation_id": "conv-old"}

    brain = Brain()
    real_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://gateway",
        headers={"Authorization": "Bearer t0ken"},
    )
    original = worker_module.GatewayClient

    class _Client(original):
        def __init__(self, base_url, token, client=None):
            super().__init__(base_url, token, client=real_client)

    worker_module.GatewayClient = _Client
    try:
        answered = await worker_module.run_worker(
            "http://gateway", token="t0ken", wait_seconds=0.0, once=True, brain=brain
        )
    finally:
        worker_module.GatewayClient = original
        await real_client.aclose()

    assert answered == 1
    assert brain.capabilities.can_present_card is False
    assert adapter.sent == [("100000001", "ok")]


async def test_gateway_refuses_a_card_that_exceeds_the_channel_limit() -> None:
    permissions, _ = _schedule_permissions()
    card = await _schedule_card(permissions, body="X" * 5000)
    store = GatewayStore()
    adapter = _CardAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(channel="telegram", subject="1", payload=payload())
    await store.claim("worker", lease_seconds=60)
    async with _client(api) as client:
        response = await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
        })
    assert response.status_code == 409
    assert adapter.sent == []


async def test_oversized_terminal_rewrite_degrades_to_a_dashboard_pointer() -> None:
    """A 502 is not an acceptable answer to a frozen card that cannot fit."""
    from leo.integrations.messaging import TELEGRAM_MAX_CHARS

    permissions, _ = _schedule_permissions()
    card = await _schedule_card(permissions)
    store = GatewayStore()
    adapter = _CardAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001", payload=payload(),
        principal="100000001", thread_id="77",
    )
    await store.claim("worker", lease_seconds=60)
    async with _client(api) as client:
        presented = await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
        })
        assert presented.status_code == 200
        await client.post("/v1/approval/ack", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
        })
        rewritten = await client.post("/v1/approval/update", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
            "text": "⏰ Schedule approval\n" + ("X" * (TELEGRAM_MAX_CHARS + 100))
                    + "\n\napproved · once",
        })

    assert rewritten.status_code == 200
    assert rewritten.json()["updated"] is True
    assert len(adapter.edits) == 1
    text = adapter.edits[0][2]
    assert len(text) <= TELEGRAM_MAX_CHARS
    assert "Full approval text is in the LEO dashboard." in text
    # The worker supplied this verdict; no durable decision exists yet.
    assert "approved · once" in text
    assert "X" * 100 not in text


async def test_update_verdict_reports_the_real_edit_result() -> None:
    """A failed rewrite must not be reported as a successful terminal edit."""
    permissions, _ = _schedule_permissions()
    card = await _schedule_card(permissions)
    store = GatewayStore()

    class _FailedEdit(_CardAdapter):
        async def edit_message_text(self, target, message_id, text, reply_markup=None) -> bool:
            await super().edit_message_text(target, message_id, text, reply_markup)
            return False

    adapter = _FailedEdit()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001", payload=payload(),
        principal="100000001",
    )
    await store.claim("worker", lease_seconds=60)
    async with _client(api) as client:
        assert (await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
        })).status_code == 200
        await client.post("/v1/approval/ack", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
        })
        rewritten = await client.post("/v1/approval/update", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card", "text": "terminal card",
        })

    assert rewritten.status_code == 200
    assert rewritten.json()["updated"] is False
    assert len(adapter.edits) == 1


async def test_oversized_rewrite_verdict_comes_from_the_recorded_decision() -> None:
    """A trailing note must not be reported as the card's verdict."""
    from leo.integrations.messaging import TELEGRAM_MAX_CHARS
    from leo.security.approval_cards import schedule_terminal_text

    permissions, _ = _schedule_permissions()
    card = await _schedule_card(permissions)
    store = GatewayStore()
    adapter = _CardAdapter()
    api = _api(store, adapters={"telegram": adapter})
    admission_id, _ = await store.admit(
        channel="telegram", subject="100000001", payload=payload(),
        principal="100000001", thread_id="77",
    )
    await store.claim("worker", lease_seconds=60)
    async with _client(api) as client:
        assert (await client.post("/v1/approval/present", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
        })).status_code == 200
        await client.post("/v1/approval/ack", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
        })
    await permissions.present_schedule_card(card.approval_id, surface="telegram")
    resolution = await permissions.resolve_schedule_card(
        card.approval_id, choice="deny", resolver="telegram-button:100000001",
        confirm_proposal_digest="gw-card", deny_reason="not now",
    )
    assert resolution.status == "denied"
    updated = await permissions.get_schedule_card(card.approval_id)
    terminal = schedule_terminal_text(
        updated, status=resolution.status, note="a trailing operator note",
    )
    assert terminal.splitlines()[-1] == "a trailing operator note"

    async with _client(api) as client:
        rewritten = await client.post("/v1/approval/update", json={
            "admission_id": admission_id, "approval_id": card.approval_id,
            "confirm_proposal_digest": "gw-card",
            "text": "⏰ Schedule approval\n" + ("X" * (TELEGRAM_MAX_CHARS + 100))
                    + "\n\n✅ approved\n" + terminal.splitlines()[-1],
        })

    assert rewritten.status_code == 200
    text = adapter.edits[0][2]
    assert "⛔ denied" in text
    assert "approved" not in text
    assert "a trailing operator note" not in text
