"""A keyboard tap must switch the model in the single-process app too.

LEO ran as three units (gateway / brain / ui) when the model picker was built, so
the tap was relayed as an admission to the process that owns settings. With the
services collapsed into one unit, `handle_channel_callback` installed no
admission path at all: every picker button answered "this process cannot switch
models; run /model <name>" — visible buttons, inert behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from leo.core.app import LeoApp


class _RecordingAdapter:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send(self, target: str, text: str, **kwargs: Any) -> dict[str, Any]:
        self.sent.append((str(target), str(text)))
        return {"ok": True}


class _Stub:
    """Just enough of LeoApp for the two hook methods under test."""

    # The methods under test are LeoApp's own; the stub only stands in for the
    # collaborators they touch.
    _admit_slash_from_tap = LeoApp._admit_slash_from_tap

    def __init__(self, adapter: _RecordingAdapter | None = None) -> None:
        self.gateway = SimpleNamespace(adapters={"telegram": adapter} if adapter else {})
        self.calls: list[dict[str, Any]] = []
        self.settings: Any = SimpleNamespace()
        self.permissions: Any = SimpleNamespace()
        self.commands: Any = None

    def _approval_senders(self, channel: str) -> set[str]:
        return {"100000001"}

    async def operator_resolve_approval(self, approval_id: str, approved: bool, *, actor: str):
        return {"ok": True}

    async def commands_dispatch(self, text: str, **kwargs: Any):
        self.calls.append({"text": text, **kwargs})
        return SimpleNamespace(message=f"done: {text}")


@pytest.mark.asyncio
async def test_the_tap_handler_is_given_an_admission_path(monkeypatch) -> None:
    """No admit callable is the bug: the picker answers "cannot switch models"."""
    captured: dict[str, Any] = {}

    async def fake_handle_tap(channel: str, descriptor: dict[str, Any], **kwargs: Any):
        captured.update(kwargs)
        captured["channel"] = channel
        return {"ok": True}

    monkeypatch.setattr("leo.gateway.callbacks.handle_tap", fake_handle_tap)

    stub = _Stub()

    await LeoApp.handle_channel_callback(
        stub, "telegram", {"callback_id": "c1", "chat_id": "100000001",
                           "message_id": "9", "sender_id": "100000001",
                           "data": "mdl:beam-free"},
    )

    assert callable(captured.get("admit")), "a tap with no admission path is an inert button"
    assert captured["channel"] == "telegram"


@pytest.mark.asyncio
async def test_an_admitted_tap_runs_the_command_and_answers_the_chat() -> None:
    adapter = _RecordingAdapter()
    stub = _Stub(adapter)
    stub.commands = SimpleNamespace(dispatch=lambda text, **kw: stub.commands_dispatch(text, **kw))

    await LeoApp._admit_slash_from_tap(stub, "/model beam-free", "telegram", "100000001")

    assert [c["text"] for c in stub.calls] == ["/model beam-free"]
    assert stub.calls[0]["channel"] == "telegram"
    assert stub.calls[0]["principal"] == "channel:100000001"
    assert adapter.sent == [("100000001", "done: /model beam-free")]


@pytest.mark.asyncio
async def test_a_tap_without_an_adapter_still_dispatches() -> None:
    """The switch must persist even if the reply cannot be delivered."""
    stub = _Stub(adapter=None)
    stub.commands = SimpleNamespace(dispatch=lambda text, **kw: stub.commands_dispatch(text, **kw))

    await LeoApp._admit_slash_from_tap(stub, "/model beam-free", "telegram", "100000001")

    assert [c["text"] for c in stub.calls] == ["/model beam-free"]
