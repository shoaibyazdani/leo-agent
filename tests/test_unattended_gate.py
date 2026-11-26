"""Source-aware unattended gate: no invisible approvals, one durable notice."""

from __future__ import annotations

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
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return {
        "executed": executed,
        "executor": executor,
        "registry": registry,
        "permissions": permissions,
    }


def _call(name: str, arguments: dict, *, source: str) -> ToolCall:
    return ToolCall(
        name=name,
        arguments=arguments,
        actor="scheduled-worker",
        source=source,
        origin="scheduled",
        principal="scheduled",
        conversation_id="conv_scheduled",
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
        result = await env["executor"].execute(_call(name, arguments, source=source))
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
