"""Approval notifications must reach the operator and never grant authority."""

from __future__ import annotations

import pytest

from leo.config import Settings
from leo.contracts import RiskLevel
from leo.events import Event, EventBus
from leo.security.notices import REVIEW_FAILED_PREFIX, REVIEW_STAGED_PREFIX
from leo.security.notifier import MAX_NAMED_STAGED, REVIEW_TOPICS, ApprovalNotifier


@pytest.fixture(autouse=True)
def _notifications_opt_in(monkeypatch):
    """The suite-wide kill switch is off here: this module tests the notifier."""
    monkeypatch.delenv("LEO_NO_NOTIFY", raising=False)


class FakeAdapter:
    channel = "telegram"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.raise_on_send: Exception | None = None

    async def send(self, target: str, text: str) -> dict:
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append((target, text))
        return {"message_id": "1"}


class FakeGateway:
    def __init__(self, adapter: FakeAdapter | None = None, events: EventBus | None = None) -> None:
        self._adapters = {"telegram": adapter or FakeAdapter()}
        self.events = events

    def _adapter(self, channel: str):
        return self._adapters[channel]


def _settings(enabled: bool = True, channel: str = "telegram", target: str = "525083981") -> Settings:
    settings = Settings()
    settings.security.notify.enabled = enabled
    settings.security.notify.channel = channel
    settings.security.notify.target = target
    return settings


def _request(approval_id: str = "apr_test", action: str = "gateway.send") -> Event:
    return Event(
        topic="approval.requested",
        payload={
            "approval_id": approval_id,
            "actor": "gateway",
            "action": action,
            "resource": "telegram:525083981",
            "risk": "high",
        },
    )


async def _pending_approval(approval_id: str, action: str = "gateway.send") -> None:
    """A notification must correspond to a real, still-pending approval."""
    from leo import db
    from leo.contracts import utcnow

    await db.execute(
        "INSERT OR REPLACE INTO approvals(id,created_at,status,actor,action,resource,risk,"
        "reason,requested_by,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (approval_id, utcnow(), "pending", "operator", action, None, "high",
         "test", "operator", "{}"),
    )


@pytest.mark.asyncio
async def test_request_is_pushed_to_operator_channel() -> None:
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    assert notifier.start() is True

    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="gateway.send",
                   resource="telegram:525083981", risk="high")

    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target == "525083981"
    assert "apr_1" in text
    assert "gateway.send" in text
    assert notifier.stats()["sent"] == 1


@pytest.mark.asyncio
async def test_disabled_does_not_subscribe() -> None:
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(enabled=False), events=bus)

    assert notifier.start() is False
    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="x", risk="high")
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_partial_config_is_not_enough() -> None:
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    for settings in (_settings(channel=""), _settings(target="")):
        notifier = ApprovalNotifier(FakeGateway(adapter, bus), settings, events=bus)
        assert notifier.start() is False
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_a_foreign_database_never_announces(monkeypatch, tmp_path) -> None:
    """A scratch database must not push buttons the real gateway cannot resolve.

    Real symptom: an approval from a test/CLI process arrives on Telegram, the
    operator taps Approve, and the gateway answers "That approval no longer
    exists" because the row lives in a database it does not read.
    """
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    monkeypatch.setenv("LEO_DB_PATH", str(tmp_path / "elsewhere.db"))

    assert notifier.start() is False
    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="x", risk="high")
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_a_scratch_database_without_an_explicit_owner_never_announces(monkeypatch) -> None:
    """No LEO_DB_PATH still means the configured database owns the buttons."""
    from leo import db

    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    monkeypatch.delenv("LEO_DB_PATH", raising=False)
    assert db.get_db_path() != _settings().db_path

    assert notifier.start() is False
    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="x", risk="high")
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_the_owning_database_announces(monkeypatch) -> None:
    from leo import db

    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    monkeypatch.setenv("LEO_DB_PATH", str(db.get_db_path()))

    assert notifier.start() is True
    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="x", risk="high")
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_kill_switch_disables_notifications(monkeypatch) -> None:
    monkeypatch.setenv("LEO_NO_NOTIFY", "1")
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)

    assert notifier.start() is False
    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="x", risk="high")
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_send_failure_never_breaks_the_requesting_call() -> None:
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    adapter.raise_on_send = RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    notifier.start()

    # publish() gathers handlers with return_exceptions=True, so this must not raise.
    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="x", risk="high")

    assert adapter.sent == []
    assert notifier.stats()["failed"] == 1


@pytest.mark.asyncio
async def test_notifier_cannot_resolve_an_approval() -> None:
    """Notification is informational: it holds no permission manager."""
    notifier = ApprovalNotifier(FakeGateway(), _settings())
    assert not hasattr(notifier, "permissions")
    assert not hasattr(notifier, "resolve_approval")


@pytest.mark.asyncio
async def test_notifier_uses_adapter_directly_not_gateway_send() -> None:
    """Gateway.send would re-enter authorize(HIGH) and loop on itself."""
    calls: list[str] = []

    class RecordingGateway(FakeGateway):
        async def send(self, *args, **kwargs):  # pragma: no cover - must not run
            calls.append("gateway.send")
            raise AssertionError("notifier must not route through Gateway.send")

    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(RecordingGateway(adapter, bus), _settings(), events=bus)
    notifier.start()
    await _pending_approval("apr_1")
    await bus.emit("approval.requested", approval_id="apr_1", action="x", risk="high")

    assert calls == []
    assert len(adapter.sent) == 1


def test_message_contains_no_secret_material() -> None:
    notifier = ApprovalNotifier(FakeGateway(), _settings())
    text = notifier._message(
        {"approval_id": "apr_1", "action": "gateway.send",
         "resource": "telegram:525083981", "risk": "high"}
    )
    assert "token" not in text.lower()
    assert "525083981" in text


def test_message_shows_the_command_and_reason() -> None:
    notifier = ApprovalNotifier(FakeGateway(), _settings())
    text = notifier._message({
        "approval_id": "apr_2", "action": "tool:shell",
        "resource": "/root/workspace", "risk": "high",
        "command": "rsvg-convert -z 2 -o girl.png girl.svg",
        "risk_reason": "unclassified command (rsvg-convert)",
    })
    assert "command:" in text and "rsvg-convert" in text
    assert "why:" in text and "unclassified" in text
    assert "workspace:" in text


def test_start_is_idempotent() -> None:
    bus = EventBus(persist=False)
    notifier = ApprovalNotifier(FakeGateway(events=bus), _settings(), events=bus)
    assert notifier.start() is True
    assert notifier.start() is True
    handlers = bus._handlers["approval.requested"]
    assert len(handlers) == 1


def test_stop_unsubscribes() -> None:
    bus = EventBus(persist=False)
    notifier = ApprovalNotifier(FakeGateway(events=bus), _settings(), events=bus)
    notifier.start()
    notifier.stop()
    assert bus._handlers["approval.requested"] == []
    for topic in REVIEW_TOPICS:
        assert bus._handlers.get(topic, []) == []


# ---------------------------------------------------------------------------
# background review notices
#
# A staged skill write used to be announced nowhere: ``review.completed`` was
# emitted with no subscriber, so a write could sit pending for days. These tests
# pin the announcement, not the review itself.


@pytest.mark.asyncio
async def test_staged_skill_write_is_announced() -> None:
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    assert notifier.start() is True

    await bus.emit(
        "review.completed",
        review_id="rev_1",
        conversation_id="conv_1",
        status="completed",
        mode="review",
        actions=3,
        staged=[{"proposal_id": "sprop_abc", "skill": "repo_change_review", "action": "create"}],
    )

    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target == "525083981"
    assert text.startswith(REVIEW_STAGED_PREFIX)
    assert "repo_change_review" in text
    assert "/skills approve sprop_abc" in text


@pytest.mark.asyncio
async def test_a_review_that_staged_nothing_stays_quiet() -> None:
    """Silence is the default: no news is not a notification."""
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    notifier.start()

    await bus.emit(
        "review.completed", review_id="rev_2", status="completed",
        mode="review", actions=2, staged=[],
    )

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_a_failed_learn_pass_is_announced() -> None:
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(FakeGateway(adapter, bus), _settings(), events=bus)
    notifier.start()

    await bus.emit(
        "learn.failed", review_id="learn_1", status="failed", mode="learn",
        actions=0, error="TimeoutError: learn exceeded 120s",
    )

    assert len(adapter.sent) == 1
    text = adapter.sent[0][1]
    assert text.startswith(REVIEW_FAILED_PREFIX)
    assert "learn" in text and "120s" in text


def test_many_staged_writes_are_summarized_not_dumped() -> None:
    staged = [
        {"proposal_id": f"sprop_{index}", "skill": f"skill-{index}", "action": "create"}
        for index in range(7)
    ]
    text = ApprovalNotifier._review_message({
        "status": "completed", "mode": "review", "staged": staged,
    })
    assert text is not None
    assert text.count("sprop_") == MAX_NAMED_STAGED + 1  # +1 from the approve hint
    assert "+4 more" in text


@pytest.mark.asyncio
async def test_review_notices_are_off_without_operator_config() -> None:
    """The same opt-in gates review notices: no channel named, no message."""
    bus = EventBus(persist=False)
    adapter = FakeAdapter()
    notifier = ApprovalNotifier(
        FakeGateway(adapter, bus), _settings(enabled=False), events=bus
    )

    assert notifier.start() is False
    await bus.emit("review.completed", review_id="rev_3", status="completed",
                   mode="review", staged=[{"proposal_id": "sprop_x", "skill": "x"}])
    assert adapter.sent == []
