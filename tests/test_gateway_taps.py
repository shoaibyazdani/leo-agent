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
        self.edits: list[tuple[str, str, str]] = []

    def set_callback_sink(self, sink) -> None:
        self.sink = sink

    async def answer_callback_query(self, callback_id: str, text: str = "") -> bool:
        self.answers.append(str(text))
        return True

    async def edit_message_text(self, target, message_id, text, reply_markup=None) -> bool:
        self.edits.append((str(target), str(message_id), str(text)))
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
    result = await handle_tap(
        "telegram", _tap(env.approval_id, decision="session"), settings=env.settings,
        permissions=env.permissions, adapters=env.adapters,
    )

    assert result["ok"] is True and result["grant_id"]
    grants = await db.fetchall("SELECT action, resource FROM approval_grants")
    assert [(g["action"], g["resource"]) for g in grants] == [("tool:shell", "shell:ls")]
    row = await db.fetchone("SELECT status FROM approvals WHERE id=?", (env.approval_id,))
    assert row["status"] == "approved"


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
