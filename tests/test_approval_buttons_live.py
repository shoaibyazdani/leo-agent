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
from leo.contracts import RiskLevel, RiskReason
from leo.security.approval_buttons import (
    build_callback_data,
    build_keyboard,
    parse_callback_data,
)
from leo.security.permissions import PermissionManagerImpl, reason_is_trustable
from leo.tools.shell_policy import assess_command

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
        risk_reason=RiskReason("ordinary command", trustable=True),
    )
    assert first.requires_approval is True

    await env.permissions.grant_session(
        "tool:shell", None, 600, OWNER, arguments_digest="digest-one",
    )

    same = await env.permissions.authorize(
        "operator", "tool:shell", None, RiskLevel.HIGH,
        arguments_digest="digest-one",
        risk_reason=RiskReason("ordinary command", trustable=True),
    )
    assert same.allowed is True and same.reason == "approval grant"

    other = await env.permissions.authorize(
        "operator", "tool:shell", None, RiskLevel.HIGH,
        arguments_digest="digest-two",
        risk_reason=RiskReason("ordinary command", trustable=True),
    )
    assert other.allowed is False and other.requires_approval is True


async def test_a_workspace_trust_covers_normal_shell_only(env) -> None:
    """Trust workspace skips ordinary work; destructive commands still ask."""
    workspace = str(env.tmp_path.resolve())
    first = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (rsvg-convert)", trustable=True),
    )
    assert first.requires_approval is True

    await env.permissions.grant_workspace("tool:shell", workspace, 600, OWNER)

    normal = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (rsvg-convert)", trustable=True),
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
        risk_reason=RiskReason("unclassified command", trustable=True),
    )
    assert decision.allowed is False and decision.requires_approval is True


async def test_manual_mode_ignores_grants(env) -> None:
    workspace = str(env.tmp_path.resolve())
    await env.permissions.grant_workspace("tool:shell", workspace, 600, OWNER)
    env.settings.security.approval_mode = "manual"

    decision = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command", trustable=True),
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


# ------------------------------------------------------------------ trust, patterns, memory


async def test_only_a_delegable_reason_gets_the_trust_button(env) -> None:
    from leo.security.permissions import reason_is_trustable

    ordinary = RiskReason("ordinary command", trustable=True)
    assert reason_is_trustable(ordinary) is True
    # A HIGH classifier reason, including an interpreter one, is never delegable.
    assert reason_is_trustable(RiskReason("python runs arbitrary code", trustable=False)) is False
    assert reason_is_trustable(RiskReason("git history or remote rewrite", trustable=False)) is False
    assert reason_is_trustable("unclassified command") is False  # no structured flag
    keyboard = build_keyboard(
        "apr_123", "tool:shell", resource=str(env.tmp_path),
        arguments_digest="a" * 64,
        allow_workspace_trust=reason_is_trustable(ordinary),
    )
    labels = [button["text"] for row in keyboard["inline_keyboard"] for button in row]
    assert any("Trust workspace" in text for text in labels)


async def test_workspace_trust_survives_a_restart_and_covers_terminal(env) -> None:
    workspace = str(env.tmp_path.resolve())
    await env.permissions.grant_workspace(
        "tool:shell", workspace, int(30 * 86400), OWNER
    )
    restarted = PermissionManagerImpl(env.settings)

    for action in ("tool:shell", "tool:terminal"):
        decision = await restarted.authorize(
            "operator", action, workspace, RiskLevel.HIGH,
            risk_reason=RiskReason("ordinary command", trustable=True),
        )
        assert decision.allowed is True and decision.reason == "approval grant"


async def test_command_pattern_covers_prefix_variants_but_not_hardlines(env) -> None:
    workspace = str(env.tmp_path.resolve())
    await env.permissions.grant_command_pattern(
        "pytest *", granted_by=OWNER, workspace=workspace
    )

    def authorize(command: str, reason: str = RiskReason("unclassified command (pytest)", trustable=True)):
        return env.permissions.authorize(
            "operator", "tool:shell", workspace, RiskLevel.HIGH,
            risk_reason=reason, review_command=command, arguments_digest=command,
        )

    assert (await authorize("pytest tests/test_x.py")).allowed is True
    assert (await authorize("pytest -q")).allowed is True
    assert (await authorize("pytest-other -q")).requires_approval is True
    assert (await authorize("git status")).requires_approval is True
    assert (await authorize("pytest -q", "rm is destructive")).requires_approval is True


async def test_directory_grant_covers_only_its_tree(env) -> None:
    src = env.tmp_path / "src"
    src.mkdir()
    inside, outside = src / "app.py", env.tmp_path / "other" / "app.py"
    await env.permissions.grant_directory("tool:patch", str(src), granted_by=OWNER)

    assert (await env.permissions.authorize(
        "operator", "tool:patch", str(inside), RiskLevel.HIGH,
    )).allowed is True
    assert (await env.permissions.authorize(
        "operator", "tool:patch", str(outside), RiskLevel.HIGH,
    )).requires_approval is True


async def test_standing_grants_are_listed_and_revoked(env) -> None:
    workspace = str(env.tmp_path.resolve())
    grant_id = await env.permissions.grant_standing(
        "pattern", "git *", granted_by=OWNER, workspace=workspace
    )
    listed = await env.permissions.list_grants()
    assert any(item["id"] == grant_id and item["scope"] == "standing" for item in listed)

    allowed = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (git)", trustable=True), review_command="git diff",
    )
    assert allowed.allowed is True

    assert await env.permissions.revoke_grant(grant_id) is True
    assert (await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (git)", trustable=True), review_command="git diff",
    )).requires_approval is True


async def test_an_approved_kind_is_remembered_for_that_task_only(env) -> None:
    workspace = str(env.tmp_path.resolve())
    first = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (pytest)", trustable=True), task_id="conv-1",
        command_preview="pytest tests/test_a.py", arguments_digest="digest-a",
    )
    assert first.requires_approval is True
    await env.permissions.resolve_approval(first.approval_id, True, resolver=OWNER)

    same_task = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (pytest)", trustable=True), task_id="conv-1",
        review_command="pytest -q", arguments_digest="digest-b",
    )
    assert same_task.allowed is True and same_task.reason == "approval grant"

    other_task = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason=RiskReason("unclassified command (pytest)", trustable=True), task_id="conv-2",
        review_command="pytest -q", arguments_digest="digest-b",
    )
    assert other_task.requires_approval is True


async def test_a_hardline_approval_is_not_remembered_for_the_task(env) -> None:
    workspace = str(env.tmp_path.resolve())
    first = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason="rm is destructive", task_id="conv-1",
        command_preview="rm -rf build", arguments_digest="digest-a",
    )
    await env.permissions.resolve_approval(first.approval_id, True, resolver=OWNER)

    second = await env.permissions.authorize(
        "operator", "tool:shell", workspace, RiskLevel.HIGH,
        risk_reason="rm is destructive", task_id="conv-1",
        review_command="rm -rf dist", arguments_digest="digest-b",
    )
    assert second.requires_approval is True


# Every HIGH branch of the shell classifier, keyed by a command that reaches it.
# The corpus is deliberately broad: a new HIGH reason may not open a grant hole
# just because nobody thought to add it to a substring deny-list.
_HIGH_REASON_COMMANDS = (
    "rm -rf /", "echo $(whoami)", "echo x > .env", "echo x > /etc/x",
    "systemctl reboot", "systemctl frobnicate", "sqlite3",
    "curl https://example.invalid", "curl -s", "curl example.invalid",
    "wget https://example.invalid", "awk 'BEGIN { system(\"id\") }'",
    "git push origin main", "git push --force origin main", "git credential fill",
    "git branch -D old", "git checkout -- README.md", "pip uninstall foo",
    "chmod -R 777 .", "cp a $OUTSIDE/x", 'cp a "/tmp/x\x00y"',
    "cp a /etc/x", "echo $API_TOKEN", "curl --token abc localhost",
    "ls ~/.ssh", "cat .env", "cat server.pem", "ssh user@host",
    "kubectl apply -f x.yaml", "some-unknown-tool https://example.invalid",
    "sudo ls", "passwd", "docker run --privileged x", "rm -rf build", "find . -delete",
    "docker login", "cp a leo.yaml", "python -c 'print(1)'",
    "python -m http.server", "python", "python mystery.bin", "npm publish",
    "npm uninstall foo", "pip install git+https://example.invalid/pkg",
    "pip install -g foo", "gh auth status", "gh api /user", "gh pr merge 7",
    "env", "printenv", "cd /etc && ls", "echo 'unterminated",
)


async def test_every_high_classifier_reason_defeats_trust_grants_and_reviewer(env) -> None:
    """R2 floor: no HIGH reason is delegable, whatever words it uses."""
    reasons: dict[str, object] = {}
    for command in _HIGH_REASON_COMMANDS:
        risk, reason = assess_command(command, env.settings)
        assert risk is RiskLevel.HIGH, (command, risk, reason)
        assert reason_is_trustable(reason) is False, reason
        reasons[str(reason)] = reason
    # Policy-wide HIGH outcomes also travel through the same structured flag.
    from leo.config import Settings as _Settings

    disabled = _Settings()
    disabled.security.shell_enabled = False
    manual = _Settings()
    manual.security.shell_mode = "manual"
    for settings in (disabled, manual):
        risk, reason = assess_command("ls", settings)
        assert risk is RiskLevel.HIGH and reason_is_trustable(reason) is False
        reasons[str(reason)] = reason
    assert len(reasons) >= 40, "the HIGH-reason corpus lost coverage"

    workspace = str(env.tmp_path.resolve())
    await env.permissions.grant_workspace("tool:shell", workspace, 600, OWNER)
    await env.permissions.grant_session(
        "tool:shell", None, 600, OWNER, arguments_digest="grant-digest"
    )
    await env.permissions.grant_command_pattern(
        "git *", granted_by=OWNER, workspace=workspace
    )
    env.settings.security.approval_mode = "smart"
    env.settings.security.reviewer_enabled = True
    reviewed: list[str] = []

    async def reviewer(**kwargs):
        reviewed.append(str(kwargs.get("command") or ""))
        return "allow", "trusted"

    env.permissions.set_reviewer(reviewer)

    for text, reason in reasons.items():
        for supplied in (reason, text):  # structured and legacy string callers
            decision = await env.permissions.authorize(
                "operator", "tool:shell", workspace, RiskLevel.HIGH,
                risk_reason=supplied, review_command="git push origin main",
                command_preview="git push origin main",
                arguments_digest="grant-digest", task_id="conv-1",
            )
            assert decision.allowed is False, (text, supplied, decision.reason)
            assert decision.requires_approval is True, (text, supplied, decision.reason)
    assert reviewed == [], "a HIGH classifier reason must never reach the reviewer"
