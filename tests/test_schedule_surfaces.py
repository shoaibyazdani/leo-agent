"""Phase 9 surface matrix: cards only where the whole contract can be kept.

A capable surface must show the literal full body, take a real decision and
rewrite the result. Everything else - including unknown adapters and shell
notifications - is incapable by default.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import db
from leo.contracts import INCAPABLE_SURFACE, SurfaceCapabilities, surface_capabilities
from leo.gateway.server import (
    APPROVAL_RPC_PROTOCOL,
    approval_channel_capable,
    gateway_approval_channels,
)
from leo.integrations.messaging import TelegramAdapter
from leo.security.approval_cards import (
    ApprovalCard,
    card_fits_surface,
    card_pages,
    paginate_text,
    schedule_notification_copy,
)
from leo.security.notifier import SCHEDULE_NOTIFICATION_TOPIC, ApprovalNotifier
from leo.config import Settings
from leo.events import Event, EventBus


def test_capability_matrix_admits_only_answerable_surfaces() -> None:
    for surface in ("tui", "web", "electron", "telegram"):
        caps = surface_capabilities(surface)
        assert caps.can_present_card and caps.can_wait_for_decision and caps.full_text
        assert caps.can_rewrite
        assert caps.max_card_chars and caps.max_card_chars > 0
    for surface in ("cli", "api", "a2a", "whatsapp", "whatsapp_web", "agent",
                    "scheduled", "chat", "matrix", "signal", "discord"):
        caps = surface_capabilities(surface)
        assert not caps.can_present_card
        assert not caps.can_wait_for_decision
    unknown = surface_capabilities("brand-new-adapter")
    assert unknown == INCAPABLE_SURFACE
    assert SurfaceCapabilities().can_present_card is False


def test_card_size_limits_are_lossless_and_bounded() -> None:
    body = "alpha beta gamma\n" * 5000
    for limit in (7, 100, 4096, 20000):
        pages = paginate_text(body, limit)
        assert "".join(pages) == body
        assert all(len(page) <= limit for page in pages)
    card = ApprovalCard(approval_id="apr_pages", body=body, title="t")
    assert "".join(card_pages(card, "web")) == body
    assert card_fits_surface(card, "web")
    assert not card_fits_surface(card, "telegram")  # 4096 chars x 12 pages
    assert not card_fits_surface(card, "a2a")


def test_pagination_never_silently_drops_a_long_token() -> None:
    body = "A" * 400 + "\n" + "B" * 400
    pages = paginate_text(body, 100)
    assert len(pages) > 1
    assert "".join(pages) == body


def test_notification_copy_is_short_and_carries_no_authority() -> None:
    card = ApprovalCard(
        approval_id="apr_n", kind="schedule_create", title="LEO wants to schedule a job",
        body='Schedule "Water": ' + "x" * 5000, status="pending",
    )
    title, body = schedule_notification_copy(card)
    assert len(title) <= 120
    assert len(body) <= 1100
    assert "notification only" in body
    assert card.body not in body
    assert "approve" not in body.lower()


@pytest.mark.asyncio
async def test_schedule_notification_event_is_not_presentation(monkeypatch) -> None:
    from leo import db

    monkeypatch.delenv("LEO_NO_NOTIFY", raising=False)
    monkeypatch.setenv("LEO_DB_PATH", str(db.get_db_path()))
    settings = Settings()
    settings.security.notify.enabled = True
    settings.security.notify.channel = "telegram"
    settings.security.notify.target = "42"
    bus = EventBus(persist=False)

    permissions = _permissions(settings)
    card = await permissions.create_schedule_card(
        kind="schedule_create", body='Schedule "Water": drink water.',
        proposal_digest="surface-digest", requested_by="chat:web",
        conversation_id="conv-surface",
    )

    class Adapter:
        async def send(self, target, text, **kwargs):
            return {"message_id": "1"}

    seen: list[dict] = []
    bus.subscribe(SCHEDULE_NOTIFICATION_TOPIC, lambda event: _record(seen, event))
    notifier = ApprovalNotifier(
        SimpleNamespace(_adapter=lambda channel: Adapter()), settings, bus,
    )
    await notifier._on_requested(
        Event(topic="approval.requested", payload={**card.event_payload()})
    )
    assert len(seen) == 1
    assert seen[0]["presented"] is False
    assert seen[0]["approval_id"] == card.approval_id
    row = await db.fetchone(
        "SELECT shown_at, presented_surface FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["shown_at"] is None and row["presented_surface"] is None


async def _record(target: list[dict], event) -> None:
    target.append(dict(event.payload))


def _permissions(settings: Settings):
    from leo.security.permissions import PermissionManagerImpl

    return PermissionManagerImpl(settings)


def test_gateway_protocol_health_filters_mixed_versions_and_channels() -> None:
    adapter = TelegramAdapter(require_verification=False)
    assert approval_channel_capable("telegram", adapter) is True
    assert approval_channel_capable("whatsapp", adapter) is False
    healthy = {
        "approval_protocol": APPROVAL_RPC_PROTOCOL,
        "approval_cards": {
            "telegram": {
                "can_present_card": True, "can_wait_for_decision": True,
                "full_text": True, "can_rewrite": True, "terminal_edit": True,
            },
            "whatsapp": {"can_present_card": False},
        },
    }
    assert gateway_approval_channels(healthy) == frozenset({"telegram"})
    assert gateway_approval_channels({}) == frozenset()  # old gateway
    assert gateway_approval_channels({
        "approval_protocol": APPROVAL_RPC_PROTOCOL + 1,
        "approval_cards": healthy["approval_cards"],
    }) == frozenset()


@pytest.mark.asyncio
async def test_failed_transport_presentation_never_marks_a_card_shown() -> None:
    from leo.security.approval_cards import presenting_cards_via

    permissions = _permissions(Settings())
    card = await permissions.create_schedule_card(
        kind="schedule_create", body='Schedule "Water": drink water.',
        proposal_digest="no-transport", requested_by="chat:web",
    )

    async def unavailable(_approval_id: str, _surface: str) -> bool:
        return False

    with presenting_cards_via(unavailable):
        with pytest.raises(PermissionError):
            await permissions.present_schedule_card(card.approval_id, surface="telegram")
    row = await db.fetchone(
        "SELECT shown_at, presented_surface FROM approvals WHERE id=?", (card.approval_id,)
    )
    assert row["shown_at"] is None and row["presented_surface"] is None
