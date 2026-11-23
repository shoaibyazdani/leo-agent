"""A tap must resolve the approval in the process that actually polls Telegram.

The split deployment put polling in the gateway and the agent in the worker. The
tap handler was installed only by the app, so every real tap was answered "No
handler for this button" while the buttons looked alive. These tests pin the
gateway-side path, which is the one that sees the tap.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import ChannelsConfig, Settings
from leo.contracts import RiskLevel
from leo.gateway.callbacks import handle_tap, install
from leo.security.approval_buttons import build_callback_data
from leo.security.permissions import PermissionManagerImpl

OWNER = "100000001"
STRANGER = "999999999"


class FakeAdapter:
    channel = "telegram"

    def __init__(self) -> None:
        self.answers: list[str] = []
        self.edits: list[tuple[str, str, str, dict]] = []

    def set_callback_sink(self, sink) -> None:
        self.sink = sink

    async def answer_callback_query(self, callback_id: str, text: str = "") -> bool:
        self.answers.append(str(text))
        return True

    async def edit_message_text(self, target, message_id, text, reply_markup=None) -> bool:
        self.edits.append((str(target), str(message_id), str(text), dict(reply_markup or {})))
        return True


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "taps.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.security.high_risk_requires_approval = True
    settings.channels = ChannelsConfig(
        policies={"telegram": {"mode": "allowlist", "allow": [OWNER]}}
    )
    permissions = PermissionManagerImpl(settings)
    adapter = FakeAdapter()
    adapters = {"telegram": adapter}
    decision = await permissions.authorize(
        "operator", "tool:shell", "shell:ls", RiskLevel.HIGH
    )
    return SimpleNamespace(
        settings=settings, permissions=permissions, adapter=adapter, adapters=adapters,
        approval_id=decision.approval_id, tmp_path=tmp_path,
    )


def _tap(approval_id: str, decision: str = "approve", sender: str = OWNER) -> dict:
    return {
        "callback_id": "cb-1",
        "data": build_callback_data("tool", decision, approval_id),
        "sender_id": sender,
        "chat_id": OWNER,
        "message_id": "71",
    }


async def test_install_attaches_a_sink_to_every_adapter(env) -> None:
    assert install(env.settings, env.permissions, env.adapters) is True
    assert callable(env.adapter.sink)


async def test_a_tap_approves_the_approval_without_an_agent(env) -> None:
    result = await handle_tap(
        "telegram", _tap(env.approval_id), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters,
    )

    assert result["ok"] is True and result["decision"] == "approved"
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (env.approval_id,))
    assert row["status"] == "approved"
    assert env.adapter.answers and "approved" in env.adapter.answers[-1]
    assert env.adapter.edits, "the message must be updated so it cannot be tapped twice"


async def test_a_stranger_cannot_resolve(env) -> None:
    result = await handle_tap(
        "telegram", _tap(env.approval_id, sender=STRANGER), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters,
    )

    assert result["ok"] is False
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (env.approval_id,))
    assert row["status"] == "pending", "unauthorized tap must change nothing"
    assert "not allowed" in env.adapter.answers[-1]


async def test_always_writes_a_scoped_grant_and_approves(env) -> None:
    # "Always" is offered for path-bearing writes, where the resource names the
    # operation. Shell approvals are covered by the refusal test below.
    path = str(env.tmp_path / "app.py")
    decision = await env.permissions.authorize(
        "operator", "tool:file_write", path, RiskLevel.HIGH,
    )
    result = await handle_tap(
        "telegram", _tap(decision.approval_id, decision="session"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters,
    )

    assert result["ok"] is True and result["grant_id"]
    grants = await db.fetchall("SELECT action, resource FROM approval_grants")
    assert [(g["action"], g["resource"]) for g in grants] == [("tool:file_write", path)]
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (decision.approval_id,))
    assert row["status"] == "approved"


async def test_always_is_refused_for_a_shell_approval(env) -> None:
    """A shell command's authority is in its arguments, not in a cwd, so
    "Always" must not become "every high-risk shell command for 8 hours". The
    approval stays pending so the operator can still approve it once.

    Both shapes the executor can produce are covered: no cwd at all (the common
    case) and a cwd, which still does not identify the command.
    """
    for resource in (None, str(env.tmp_path)):
        decision = await env.permissions.authorize(
            "operator", "tool:shell", resource, RiskLevel.HIGH,
        )
        assert decision.approval_id

        result = await handle_tap(
            "telegram", _tap(decision.approval_id, decision="session"),
            settings=env.settings, permissions=env.permissions, adapters=env.adapters,
        )

        assert result["ok"] is False and result["reason"] == "unscoped approval"
        assert "not available" in env.adapter.answers[-1]
        row = await db.fetchone(
            "SELECT status FROM approvals WHERE id=?", (decision.approval_id,)
        )
        assert row["status"] == "pending", "the operator must still approve once"
        # And a different HIGH shell call is still gated.
        later = await env.permissions.authorize(
            "operator", "tool:shell", resource, RiskLevel.HIGH,
        )
        assert later.allowed is False and later.requires_approval is True

    assert await db.fetchall("SELECT id FROM approval_grants") == []


async def test_a_foreign_payload_is_refused(env) -> None:
    descriptor = _tap(env.approval_id)
    descriptor["data"] = "totally-made-up"

    result = await handle_tap(
        "telegram", descriptor, settings=env.settings,
        permissions=env.permissions, adapters=env.adapters,
    )

    assert result["ok"] is False
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (env.approval_id,))
    assert row["status"] == "pending"


async def test_the_app_supplied_resolver_is_used_when_present(env) -> None:
    seen: list[tuple[str, bool, str]] = []

    async def resolve(approval_id: str, approved: bool, actor: str):
        seen.append((approval_id, approved, actor))
        return {"success": True}

    result = await handle_tap(
        "telegram", _tap(env.approval_id, decision="reject"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters, resolve=resolve,
    )

    assert result["ok"] is True and result["decision"] == "rejected"
    assert seen == [(env.approval_id, False, f"telegram-button:{OWNER}")]


# "message is not modified" is success, not failure — pinned at the transport layer,
# because the progress reporter's whole stability depends on it.
def test_edit_message_text_accepts_not_modified(monkeypatch) -> None:
    from leo.integrations.messaging import TelegramAdapter

    class _Resp:
        status_code = 400

        def json(self):
            return {"ok": False, "description": "Bad Request: message is not modified"}

    class _Client:
        async def post(self, url, json=None, timeout=None):
            return _Resp()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    adapter = TelegramAdapter(client=_Client())

    import asyncio

    assert asyncio.get_event_loop_policy() is not None
    assert asyncio.run(adapter.edit_message_text("1", "2", "same text")) is True


@pytest.mark.asyncio
async def test_a_model_tap_is_relayed_as_a_slash_command(tmp_path, monkeypatch) -> None:
    """A picker button must switch the model in the process that owns settings.

    The gateway polls and sees the tap but must not edit config behind the
    brain's back: it queues ``/model <name>`` as ordinary work.
    """
    db.set_db_path(tmp_path / "taps.db")
    from leo import db as db_module
    await db_module.initialize()

    adapter = FakeAdapter()
    admitted: list[tuple[str, str, str]] = []

    async def admit(text: str, channel: str, subject: str) -> None:
        admitted.append((text, channel, subject))

    settings = Settings()
    settings.channels = ChannelsConfig(policies={"telegram": {"allow": [OWNER], "mode": "allowlist"}})
    permissions = PermissionManagerImpl(settings)

    result = await handle_tap(
        "telegram",
        {"callback_id": "cb1", "chat_id": OWNER, "message_id": "42",
         "sender_id": OWNER, "data": "mdl:beam-free"},
        settings=settings, permissions=permissions,
        adapters={"telegram": adapter}, admit=admit,
    )

    assert result["ok"] is True and result["model"] == "beam-free"
    assert admitted == [("/model beam-free", "telegram", OWNER)]
    assert adapter.edits and adapter.edits[-1][2].count("beam-free") == 1
    # the picker is removed so the same button cannot be tapped twice
    assert adapter.edits[-1][3]["inline_keyboard"] == []


@pytest.mark.asyncio
async def test_a_stranger_cannot_tap_the_model_picker(tmp_path) -> None:
    db.set_db_path(tmp_path / "taps2.db")
    from leo import db as db_module
    await db_module.initialize()

    adapter = FakeAdapter()
    admitted: list[tuple[str, str, str]] = []

    async def admit(text: str, channel: str, subject: str) -> None:
        admitted.append((text, channel, subject))

    settings = Settings()
    settings.channels = ChannelsConfig(policies={"telegram": {"allow": [OWNER], "mode": "allowlist"}})
    result = await handle_tap(
        "telegram",
        {"callback_id": "cb2", "chat_id": STRANGER, "message_id": "43",
         "sender_id": STRANGER, "data": "mdl:beam-free"},
        settings=settings, permissions=PermissionManagerImpl(settings),
        adapters={"telegram": adapter}, admit=admit,
    )

    assert result["ok"] is False
    assert admitted == []


@pytest.mark.asyncio
async def test_a_page_tap_asks_the_brain_for_that_page(tmp_path) -> None:
    """Next/Back must render a page the same way a model tap renders a switch."""
    db.set_db_path(tmp_path / "taps3.db")
    from leo import db as db_module
    await db_module.initialize()

    adapter = FakeAdapter()
    admitted: list[tuple[str, str, str]] = []

    async def admit(text: str, channel: str, subject: str) -> None:
        admitted.append((text, channel, subject))

    from leo.config import ModelProviderConfig

    settings = Settings()
    settings.channels = ChannelsConfig(policies={"telegram": {"allow": [OWNER], "mode": "allowlist"}})
    # Enough providers that page 2 (index 2 -> "page 3") actually exists.
    settings.models.providers = {
        f"m{i:02d}": ModelProviderConfig(model=f"model-{i}") for i in range(20)
    }
    result = await handle_tap(
        "telegram",
        {"callback_id": "cb3", "chat_id": OWNER, "message_id": "44",
         "sender_id": OWNER, "data": "mdl:pg:2"},
        settings=settings, permissions=PermissionManagerImpl(settings),
        adapters={"telegram": adapter}, admit=admit,
    )

    assert result == {"ok": True, "page": 2}
    # Pagination is a pure re-render: it edits the SAME message in place and
    # never queues a command, so no stale text is left behind and no second
    # picker is posted below it.
    assert admitted == []
    assert "page 3" in adapter.edits[-1][2]
    # The new page keeps its navigation buttons (not stripped away).
    assert adapter.edits[-1][3]["inline_keyboard"] != []


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ["mdl:beam-free"])
async def test_no_admission_path_is_reported_not_crashed(tmp_path, data: str) -> None:
    """Both branches need an admission path; calling None must never be reached.

    The page branch was added above the guard once, so a page tap in a process
    without an admission path raised TypeError instead of answering.
    """
    db.set_db_path(tmp_path / "taps4.db")
    from leo import db as db_module
    await db_module.initialize()

    adapter = FakeAdapter()
    settings = Settings()
    settings.channels = ChannelsConfig(policies={"telegram": {"allow": [OWNER], "mode": "allowlist"}})
    result = await handle_tap(
        "telegram",
        {"callback_id": "cb4", "chat_id": OWNER, "message_id": "45",
         "sender_id": OWNER, "data": data},
        settings=settings, permissions=PermissionManagerImpl(settings),
        adapters={"telegram": adapter}, admit=None,
    )

    assert result == {"ok": False, "reason": "no admission path"}
    assert adapter.edits == []
