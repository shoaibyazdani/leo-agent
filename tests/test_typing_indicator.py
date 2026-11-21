"""Typing indicator: it must fire, and it must never break delivery.

Telegram clears "typing…" after ~5 seconds, so the gateway pings on an interval
while a turn runs. The indicator is cosmetic — a failing ping cannot be allowed
to fail the reply.
"""

from __future__ import annotations

import httpx

from leo.config import ChannelsConfig
from leo.events import EventBus
from leo.integrations.gateway import Gateway
from leo.integrations.admission import ChannelAdmission
from leo.integrations.messaging import MessagingAdapter, NormalizedMessage, TelegramAdapter


class _AllowAll:
    async def authorize(self, actor, action, resource=None, risk=None, **context):
        from leo.contracts import Decision, RiskLevel

        return Decision(allowed=True, reason="test", risk=risk or RiskLevel.LOW)

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        from leo.contracts import Decision, RiskLevel

        return Decision(allowed=approved, reason="test", risk=RiskLevel.HIGH)


def _telegram(requests: list[httpx.Request]) -> TelegramAdapter:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": True})

    return TelegramAdapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_telegram_typing_posts_send_chat_action(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    requests: list[httpx.Request] = []
    adapter = _telegram(requests)

    await adapter.typing("525083981")

    assert len(requests) == 1
    assert requests[0].url.path.endswith("/sendChatAction")
    body = requests[0].content.decode()
    assert "525083981" in body and "typing" in body


async def test_base_adapter_typing_is_a_no_op():
    class _Silent(MessagingAdapter):
        channel = "silent"

    assert await _Silent().typing("anyone") is None


async def test_typing_failure_does_not_block_the_reply():
    class _BrokenTyping(MessagingAdapter):
        channel = "broken"

        async def typing(self, target: str) -> None:
            raise RuntimeError("indicator down")

        async def send(self, target: str, text: str) -> dict:
            return {"channel": self.channel, "target": target, "success": True, "message_id": "1"}

    gateway = Gateway(
        {"broken": _BrokenTyping()},
        events=EventBus(persist=False),
        permissions=_AllowAll(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
    )
    gateway.set_reply_handler(
        lambda payload: _confirm(gateway, payload)
    )
    await gateway.receive(
        NormalizedMessage(channel="broken", sender="eve", text="hi", sender_id="1", conversation_ref="1")
    )

    summary = await gateway.drain_outbox(limit=5)

    assert summary["processed"] == 1 and summary["failed"] == 0


async def _confirm(gateway: Gateway, payload: dict) -> dict:
    return await gateway.send(
        payload["channel"],
        payload["reply_target"],
        "pong",
        idempotency_key=payload["idempotency_key"],
    )
