"""Source-aware unattended gate: no invisible approvals, one durable notice."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall, ToolSpec
from leo.events import EventBus
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry


def _schema(**properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    executed: list[str] = []

    def register(name, category, risk, schema=None):
        async def handler(args: dict, ctx: dict):  # type: ignore[no-untyped-def]
            executed.append(name)
            return {"executed": name}

        registry.register(
            ToolSpec(
                name=name,
                description=f"{name} fixture",
                category=category,
                risk=risk,
                resource_field="",
                input_schema=schema or _schema(),
            ),
            handler,
        )

    register(
        "outbound_probe", "integrations", RiskLevel.HIGH,
        _schema(target={"type": "string"}, text={"type": "string"}),
    )
    register(
        "browser_probe", "browser", RiskLevel.MEDIUM,
        _schema(url={"type": "string"}),
    )
    register(
        "computer_probe", "computer", RiskLevel.MEDIUM,
        _schema(path={"type": "string"}),
    )
    register(
        "plugin_probe", "plugin", RiskLevel.MEDIUM,
        _schema(action={"type": "string"}),
    )
    register(
        "shell", "development", RiskLevel.HIGH,
        _schema(command={"type": "string"}),
    )
    register("read_probe", "utility", RiskLevel.LOW)
    # A reply into the originating conversation is LOW under the risk policy;
    # it must still be refused when nobody is watching the unattended turn.
    register(
        "channel_send", "integrations", RiskLevel.HIGH,
        _schema(target={"type": "string"}, subject={"type": "string"},
                text={"type": "string"}),
    )
    # R5: outbound send names are owned by the category-first gate, not by the
    # read-only allowlist. Declaring them LOW and uncategorized must not help.
    for name in ("gateway.send", "send_message"):
        register(name, "utility", RiskLevel.LOW,
                 _schema(target={"type": "string"}, text={"type": "string"}))
    # MEDIUM tools outside the five original categories: an unattended turn
    # must not use them to write files, memory, tasks or schedules.
    for name in ("write_file", "patch", "memory_add", "task_create", "schedule_create"):
        register(name, "utility", RiskLevel.MEDIUM, _schema(path={"type": "string"}))
    # Read-only allowlisted name deliberately registered above LOW to prove
    # the allowlist is consulted, not just the risk tier.
    register("read_file", "utility", RiskLevel.MEDIUM, _schema(path={"type": "string"}))
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return {
        "executed": executed,
        "executor": executor,
        "registry": registry,
        "permissions": permissions,
    }


def _call(
    name: str, arguments: dict, *, source: str, correlation_id: str | None = None
) -> ToolCall:
    return ToolCall(
        name=name,
        arguments=arguments,
        actor="scheduled-worker",
        source=source,
        origin="scheduled",
        principal="scheduled",
        conversation_id="conv_scheduled",
        correlation_id=correlation_id,
    )


CASES = [
    ("outbound_probe", {"target": "stranger", "text": "hello"}, "outbound"),
    ("browser_probe", {"url": "https://example.com"}, "browser"),
    ("computer_probe", {"path": "/tmp/should-not-run"}, "computer"),
    ("plugin_probe", {"action": "run"}, "plugin"),
    ("shell", {"command": "echo $(whoami)"}, "shell"),
]


@pytest.mark.parametrize("source", ["scheduled", "unattended"])
async def test_unattended_side_effects_are_denied_without_an_approval(env, source):
    for name, arguments, category in CASES:
        # One call per turn: each distinct turn earns its own notice.
        result = await env["executor"].execute(
            _call(name, arguments, source=source, correlation_id=f"turn-{name}")
        )
        assert result.success is False, name
        assert result.decision == "denied", name
        assert "unattended" in (result.error or ""), name
        assert result.metadata["unattended_gate"]["category"] == category, name
        assert result.metadata["unattended_gate"]["source"] == source
        # The handler never ran.
        assert name not in env["executed"]

    # Not one invisible approval row; one durable needs-you notice per refused
    # call (never a card and never notification-as-authority).
    approvals = await db.fetchone("SELECT COUNT(*) AS n FROM approvals")
    assert approvals["n"] == 0
    notices = await db.fetchone(
        "SELECT COUNT(*) AS n FROM events WHERE topic='automation.needs_you'"
    )
    assert notices["n"] == len(CASES)
    rows = await db.fetchall(
        "SELECT payload FROM events WHERE topic='automation.needs_you'"
    )
    import json

    envelopes = [json.loads(row["payload"]) for row in rows]
    payloads = [item.get("payload", item) for item in envelopes]
    assert {item["category"] for item in payloads} == {
        "outbound", "browser", "computer", "plugin", "shell",
    }
    assert all(item["principal"] == "scheduled" for item in payloads)
    assert all(item["conversation_id"] == "conv_scheduled" for item in payloads)


@pytest.mark.parametrize(
    "name", ["write_file", "patch", "memory_add", "task_create", "schedule_create"]
)
async def test_uncategorized_medium_tools_are_denied_in_unattended_turns(env, name):
    result = await env["executor"].execute(_call(name, {"path": "/tmp/x"}, source="scheduled"))
    assert result.success is False, name
    assert result.decision == "denied", name
    assert "unattended" in (result.error or ""), name
    assert result.metadata["unattended_gate"]["category"] == "side_effect", name
    assert name not in env["executed"]
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM approvals"))["n"] == 0


async def test_allowlisted_medium_read_still_runs_unattended(env):
    result = await env["executor"].execute(
        _call("read_file", {"path": "/tmp/x"}, source="scheduled")
    )
    assert result.success is True
    assert "read_file" in env["executed"]


@pytest.mark.parametrize("name", ["channel_send", "gateway.send", "send_message"])
async def test_outbound_names_are_not_allowlisted_and_stay_denied(env, name):
    """R5: send names are denied by name before the allowlist is consulted."""
    from leo.tools.executor import _UNATTENDED_ALLOWED_TOOLS

    assert name not in _UNATTENDED_ALLOWED_TOOLS
    result = await env["executor"].execute(
        _call(name, {"target": "chat:origin", "text": "scheduled reply"},
              source="scheduled", correlation_id=f"turn-{name}")
    )
    assert result.success is False and result.decision == "denied"
    assert result.metadata["unattended_gate"] == {
        "source": "scheduled", "category": "outbound", "risk": "medium",
    }
    assert name not in env["executed"]


async def test_low_risk_or_attended_calls_still_run(env):
    # A read-only tool under a scheduled source is not a gated side effect.
    read = await env["executor"].execute(_call("read_probe", {}, source="scheduled"))
    assert read.success is True and "read_probe" in env["executed"]

    # An attended chat call keeps ordinary MEDIUM behavior.
    browser = await env["executor"].execute(
        ToolCall(
            name="browser_probe",
            arguments={"url": "https://example.com"},
            actor="chat:web",
            source="chat",
            principal="operator",
        )
    )
    assert browser.success is True
    assert "browser_probe" in env["executed"]


async def test_attended_high_risk_still_uses_the_ordinary_approval(env):
    """The gate must not remove existing command approvals for attended turns."""
    with pytest.raises(ApprovalRequired):
        await env["executor"].execute(
            ToolCall(
                name="outbound_probe",
                arguments={"target": "stranger", "text": "hello"},
                actor="chat:web",
                source="chat",
                principal="operator",
            )
        )
    pending = await db.fetchone(
        "SELECT COUNT(*) AS n FROM approvals WHERE status='pending'"
    )
    assert pending["n"] == 1
    # An attended turn does not produce the unattended notice.
    notices = await db.fetchone(
        "SELECT COUNT(*) AS n FROM events WHERE topic='automation.needs_you'"
    )
    assert notices["n"] == 0


async def test_unattended_reply_to_origin_is_denied_but_attended_reply_still_runs(env):
    """LOW tier for a reply is an attended convenience, not unattended consent."""
    args = {
        "target": "chat:origin", "subject": "chat:origin", "text": "a scheduled reply",
    }
    attended = await env["executor"].execute(
        ToolCall(
            name="channel_send", arguments=args, actor="chat:web",
            source="chat", origin="tui", principal="operator",
        )
    )
    assert attended.success is True and "channel_send" in env["executed"]
    assert attended.risk.value == "low"

    denied = await env["executor"].execute(
        _call("channel_send", args, source="scheduled", correlation_id="turn-reply")
    )
    assert denied.success is False and denied.decision == "denied"
    assert denied.metadata["unattended_gate"]["category"] == "outbound"
    assert denied.metadata["unattended_gate"]["risk"] == "low"
    assert env["executed"].count("channel_send") == 1  # only the attended call
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM approvals"))["n"] == 0
    notices = await db.fetchall(
        "SELECT payload FROM events WHERE topic='automation.needs_you'"
    )
    assert len(notices) == 1
    envelope = json.loads(notices[0]["payload"])
    assert (envelope.get("payload") or envelope)["category"] == "outbound"


async def test_repeated_denials_in_one_turn_emit_one_notice_and_keep_every_audit(env):
    args = {"target": "stranger", "text": "hello"}
    for _ in range(3):
        result = await env["executor"].execute(
            _call("outbound_probe", args, source="scheduled",
                  correlation_id="turn-three-denials")
        )
        assert result.success is False and result.decision == "denied"

    notices = await db.fetchall(
        "SELECT payload FROM events WHERE topic='automation.needs_you'"
    )
    assert len(notices) == 1
    calls = await db.fetchall("SELECT id FROM tool_calls WHERE tool='outbound_probe'")
    audits = await db.fetchall(
        "SELECT id FROM audit_log WHERE action='tool.execute' AND decision='denied'"
    )
    assert len(calls) == len(audits) == 3


async def test_new_turn_on_the_same_conversation_gets_a_new_notice(env):
    args = {"target": "stranger", "text": "hello"}
    for turn in ("turn-a", "turn-b"):
        await env["executor"].execute(
            _call("outbound_probe", args, source="scheduled", correlation_id=turn)
        )
    notices = await db.fetchall(
        "SELECT payload FROM events WHERE topic='automation.needs_you'"
    )
    assert len(notices) == 2


async def test_declared_schedule_results_still_leave_by_the_delivery_lane():
    """The gate closes tool sends, not the task engine's terminal delivery."""
    from leo.core import LeoApp

    sends: list[dict] = []

    class Gateway:
        async def send(self, channel, target, text, **kwargs):
            sends.append({"channel": channel, "target": target, "text": text, **kwargs})
            return {"success": True, "message_id": "msg-1"}

    class Reporting:
        async def on_terminal(self, task, status):
            return None

    app = LeoApp.__new__(LeoApp)
    app.gateway = Gateway()
    app.task_reporting = Reporting()
    task = SimpleNamespace(
        id="task-1", status="completed", metadata={"schedule_id": "s1"}
    )
    context = {
        "status": "completed",
        "delivery_text_override": "the scheduled result",
        "delivery_target": {"kind": "channel"},
    }
    receipt = await app._channel_delivery(task, "telegram", "chat:origin", context)
    assert receipt == "telegram:msg-1"
    assert sends == [{
        "channel": "telegram", "target": "chat:origin",
        "text": "the scheduled result", "subject": "task:task-1", "thread_id": None,
    }]
