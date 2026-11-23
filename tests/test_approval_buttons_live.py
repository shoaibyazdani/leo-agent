"""Approval buttons: a tap on Telegram resolves the approval, and only for the owner.

The gap this closes: an approval request arrived as plain text, so the operator had
to leave the chat and run a CLI command. Three properties matter and each is pinned
here:

1. the notification carries an inline keyboard,
2. a tap is dispatched as a callback (never normalised into a chat turn),
3. the decision is authorized by immutable sender id and resolved through the
   privileged path, with "Always" producing a scoped, expiring grant.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from leo import db
from leo.config import ChannelsConfig, Settings
from leo.contracts import RiskLevel
from leo.security.approval_buttons import (
    build_callback_data,
    build_keyboard,
    parse_callback_data,
)
from leo.security.permissions import PermissionManagerImpl

OWNER = "100000001"
STRANGER = "999999999"


class _Resp:
    def __init__(self, payload: dict | None = None, status: int = 200) -> None:
        self.status_code = status
        self._payload = payload or {"ok": True, "result": {"message_id": 5}}

    def json(self) -> dict:
        return self._payload


class _Client:
    """Records every Telegram call so the payload can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url, json=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1], dict(json or {})))
        return _Resp()

    async def get(self, url, params=None, timeout=None):
        self.calls.append(("getUpdates", {"params": params, "result": []}))
        return _Resp({"ok": True, "result": []})


@pytest.fixture
def adapter(monkeypatch):
    from leo.integrations.messaging import TelegramAdapter

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "button-token")
    client = _Client()
    return TelegramAdapter(client=client), client


# ------------------------------------------------------------------ the keyboard


def test_keyboard_offers_approve_always_and_reject() -> None:
    keyboard = build_keyboard("apr_123", "tool:file_write", resource="/project/app.py")

    rows = keyboard["inline_keyboard"]
    labels = [button["text"] for row in rows for button in row]
    assert any("Approve" in text for text in labels)
    assert any("Always" in text for text in labels)
    assert any("Reject" in text for text in labels)
    for row in rows:
        for button in row:
            parsed = parse_callback_data(button["callback_data"])
            assert parsed is not None, button["callback_data"]
            assert parsed["approval_id"] == "apr_123"


def test_keyboard_omits_always_when_the_grant_cannot_be_scoped() -> None:
    """Without a resource (path tools) or an arguments digest (shell), the
    grant would cover every future call, so the button must not be offered."""
    for keyboard in (
        build_keyboard("apr_123", "tool:shell"),
        build_keyboard("apr_123", "tool:shell", resource=None),
        build_keyboard("apr_123", "tool:shell", resource="   "),
        # A cwd does not identify which shell command was approved.
        build_keyboard("apr_123", "tool:shell", resource="/project"),
        build_keyboard("apr_123", "tool:terminal", resource="/project"),
        build_keyboard("apr_123", "tool:memory_forget"),
    ):
        labels = [button["text"] for row in keyboard["inline_keyboard"] for button in row]
        assert not any("Always" in text for text in labels), labels
        assert any("Approve" in text for text in labels)
        assert any("Reject" in text for text in labels)


def test_keyboard_offers_scoped_grants_for_a_shell_approval() -> None:
    """Shell gets the exact-command grant and the workspace trust."""
    for action in ("tool:shell", "tool:terminal"):
        keyboard = build_keyboard(
            "apr_123", action, resource="/project", arguments_digest="a" * 64
        )
        labels = [button["text"] for row in keyboard["inline_keyboard"] for button in row]
        assert any("This command" in text for text in labels), labels
        assert any("Trust workspace" in text for text in labels), labels


def test_keyboard_omits_trust_for_a_dangerous_reason() -> None:
    keyboard = build_keyboard(
        "apr_123", "tool:shell", arguments_digest="a" * 64,
        allow_workspace_trust=False,
    )
    labels = [button["text"] for row in keyboard["inline_keyboard"] for button in row]
    assert any("This command" in text for text in labels), labels
    assert not any("Trust workspace" in text for text in labels), labels


def test_session_decision_survives_the_round_trip() -> None:
    data = build_callback_data("tool", "session", "apr_abc")

    parsed = parse_callback_data(data)

    assert parsed == {
        "type": "approval", "approval_kind": "tool", "decision": "session",
        "approval_id": "apr_abc", "is_ref": "0",
    }


def test_foreign_or_malformed_payloads_are_refused() -> None:
    for data in ("", "approve", "tg:tool:approve:apr_1", "tga1:x:a:apr_1", None, 42):
        assert parse_callback_data(data) is None, data


# ------------------------------------------------------------------ transport


async def test_send_attaches_the_keyboard_to_the_message(adapter) -> None:
    telegram, client = adapter
    keyboard = build_keyboard("apr_777", "gateway.send")

    await telegram.send(OWNER, "approval needed", reply_markup=keyboard)

    method, payload = client.calls[-1]
    assert method == "sendMessage"
    assert payload["reply_markup"] == keyboard


async def test_a_tap_is_dispatched_and_never_becomes_a_chat_turn(adapter) -> None:
    telegram, client = adapter
    seen: list[tuple[str, dict]] = []

    async def sink(channel, descriptor):
        seen.append((channel, descriptor))

    telegram.set_callback_sink(sink)
    update = {
        "update_id": 10,
        "callback_query": {
            "id": "cb-1",
            "data": build_callback_data("tool", "approve", "apr_9"),
            "from": {"id": int(OWNER)},
            "message": {"message_id": 71, "chat": {"id": int(OWNER)}},
        },
    }

    await telegram._dispatch_callbacks([update])

    assert len(seen) == 1
    channel, descriptor = seen[0]
    assert channel == "telegram"
    assert descriptor["data"].endswith("apr_9")
    assert descriptor["sender_id"] == OWNER
    assert descriptor["message_id"] == "71"


async def test_a_tap_without_a_sink_is_acknowledged(adapter) -> None:
    telegram, client = adapter

    await telegram._dispatch_callbacks([{
        "callback_query": {"id": "cb-2", "data": "tga1:t:a:apr_1", "from": {"id": 1}},
    }])

    method, payload = client.calls[-1]
    assert method == "answerCallbackQuery"
    assert payload["callback_query_id"] == "cb-2"


# ------------------------------------------------------------------ authorization


def _settings(tmp_path: Path, requires_approval: bool = True) -> Settings:
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.security.workspace_only = True
    settings.security.high_risk_requires_approval = requires_approval
    settings.channels = ChannelsConfig(
        policies={"telegram": {"mode": "allowlist", "allow": [OWNER]}}
    )
    return settings


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "buttons.db")
    await db.initialize()
    settings = _settings(tmp_path)
    permissions = PermissionManagerImpl(settings)
    return SimpleNamespace(settings=settings, permissions=permissions, tmp_path=tmp_path)


async def test_a_granted_session_skips_the_next_approval(env) -> None:
    path = str(env.tmp_path / "app.py")
    first = await env.permissions.authorize(
        "operator", "tool:file_write", path, RiskLevel.HIGH,
    )
    assert first.allowed is False and first.requires_approval is True

    await env.permissions.grant_session("tool:file_write", path, 600, f"telegram-button:{OWNER}")

    second = await env.permissions.authorize(
        "operator", "tool:file_write", path, RiskLevel.HIGH,
    )
    assert second.allowed is True
    assert second.reason == "approval grant"


async def test_a_grant_does_not_leak_to_another_resource(env) -> None:
    path = str(env.tmp_path / "app.py")
    other_path = str(env.tmp_path / "secrets.py")
    await env.permissions.grant_session("tool:file_write", path, 600, OWNER)

    other = await env.permissions.authorize(
        "operator", "tool:file_write", other_path, RiskLevel.HIGH,
    )
    assert other.allowed is False, "a grant is scoped to one resource"


async def test_an_expired_grant_is_ignored(env) -> None:
    path = str(env.tmp_path / "app.py")
    await env.permissions.grant_session("tool:file_write", path, 60, OWNER)
    await db.execute("UPDATE approval_grants SET expires_at=?", ("2000-01-01T00:00:00+00:00",))

    decision = await env.permissions.authorize(
        "operator", "tool:file_write", path, RiskLevel.HIGH,
    )

    assert decision.allowed is False and decision.requires_approval is True


async def test_an_unscoped_grant_cannot_be_created(env) -> None:
    """Shell needs a digest; path tools need a resource. Neither may be NULL."""
    with pytest.raises(ValueError, match="digest"):
        await env.permissions.grant_session("tool:shell", None, 600, OWNER)
    with pytest.raises(ValueError, match="digest"):
        await env.permissions.grant_session("tool:shell", "  ", 600, OWNER)
    with pytest.raises(ValueError, match="needs a resource"):
        await env.permissions.grant_session("tool:file_write", None, 600, OWNER)
    rows = await db.fetchall("SELECT action FROM approval_grants")
    assert rows == []


async def test_a_shell_grant_is_scoped_to_the_exact_arguments_digest(env) -> None:
    """Always on a shell command covers that command, not every shell call."""
    first = await env.permissions.authorize(
        "operator", "tool:shell", None, RiskLevel.HIGH,
        arguments_digest="digest-one",
    )
    assert first.requires_approval is True

    await env.permissions.grant_session(
        "tool:shell", None, 600, OWNER, arguments_digest="digest-one",
    )

    same = await env.permissions.authorize(
        "operator", "tool:shell", None, RiskLevel.HIGH,
        arguments_digest="digest-one",
    )
    assert same.allowed is True and same.reason == "approval grant"

    other = await env.permissions.authorize(
        "operator", "tool:shell", None, RiskLevel.HIGH,
        arguments_digest="digest-two",
    )
    assert other.allowed is False and other.requires_approval is True


async def test_a_workspace_trust_covers_normal_shell_only(env) -> None:
    """Trust workspace skips ordinary work; destructive commands still ask."""
    workspace = str(env.tmp_path.resolve())
    first = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason="unclassified command (rsvg-convert)",
    )
    assert first.requires_approval is True

    await env.permissions.grant_workspace("tool:shell", workspace, 600, OWNER)

    normal = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason="unclassified command (rsvg-convert)",
    )
    assert normal.allowed is True and normal.reason == "approval grant"

    destructive = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason="rm is destructive",
    )
    assert destructive.allowed is False and destructive.requires_approval is True


async def test_approval_mode_off_skips_the_high_risk_gate(env) -> None:
    env.settings.security.approval_mode = "off"
    decision = await env.permissions.authorize(
        "operator", "tool:shell", str(env.tmp_path.resolve()), RiskLevel.HIGH,
        risk_reason="rm is destructive",
    )
    assert decision.allowed is True
    assert decision.reason == "approval mode off"


async def test_workspace_grants_can_be_listed_and_revoked(env) -> None:
    workspace = str(env.tmp_path.resolve())
    grant_id = await env.permissions.grant_workspace(
        "tool:shell", workspace, 600, OWNER
    )

    grants = await env.permissions.list_grants()
    assert any(
        item["id"] == grant_id and item["scope"] == "workspace"
        for item in grants
    )
    assert await env.permissions.revoke_grant(grant_id) is True
    assert await env.permissions.list_grants() == []

    decision = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason="unclassified command",
    )
    assert decision.allowed is False and decision.requires_approval is True


async def test_manual_mode_ignores_grants(env) -> None:
    workspace = str(env.tmp_path.resolve())
    await env.permissions.grant_workspace("tool:shell", workspace, 600, OWNER)
    env.settings.security.approval_mode = "manual"

    decision = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason="unclassified command",
    )
    assert decision.allowed is False and decision.requires_approval is True


async def test_a_legacy_unscoped_grant_never_authorizes(env) -> None:
    """A NULL-resource row written before the guard existed stays inert."""
    await db.execute(
        "INSERT INTO approval_grants(id,created_at,expires_at,action,resource,granted_by,scope) "
        "VALUES(?,?,?,?,?,?,?)",
        ("grn_legacy", "2026-01-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00",
         "tool:shell", None, OWNER, "session"),
    )

    decision = await env.permissions.authorize(
        "operator", "tool:shell", None, RiskLevel.HIGH,
    )

    assert decision.allowed is False and decision.requires_approval is True
