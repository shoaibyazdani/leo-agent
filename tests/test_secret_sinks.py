"""Secret preflight happens before every durable/authorization sink.

A refused secret must never be written to tool_calls, audit_log, events, logs or
quota state, and must never create an approval. Only a canonical digest and a
stable refusal category/rule are persisted.
"""

from __future__ import annotations

import json
import logging

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall, ToolSpec
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl, arguments_digest
from leo.tools import automation_tools
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry

SECRET = "sk-ant-api03-" + "Qu9x" * 10


@pytest.fixture
async def env(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    executed: list[dict] = []

    async def handler(args: dict, ctx: dict):  # type: ignore[no-untyped-def]
        executed.append(dict(args))
        return {"echo": dict(args)}

    registry.register(
        ToolSpec(
            name="secret_probe",
            description="an opt-in secret-scanning tool",
            category="test",
            risk=RiskLevel.MEDIUM,
            resource_field="",
            secret_scan=True,
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "nested": {"type": "object"},
                },
                "additionalProperties": False,
            },
        ),
        handler,
    )
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return {
        "executed": executed,
        "executor": executor,
        "events": events,
        "registry": registry,
        "settings": settings,
    }


async def test_refused_secret_reaches_no_durable_sink(env, caplog):
    caplog.set_level(logging.DEBUG)
    arguments = {"text": "please remember", "nested": {"api_key": SECRET}}
    expected_digest = arguments_digest(arguments)

    authorized: list[tuple] = []
    real_authorize = env["executor"].permissions.authorize

    async def spy_authorize(*args, **kwargs):  # type: ignore[no-untyped-def]
        authorized.append((args, kwargs))
        return await real_authorize(*args, **kwargs)

    env["executor"].permissions.authorize = spy_authorize  # type: ignore[method-assign]

    result = await env["executor"].execute(
        ToolCall(
            name="secret_probe",
            arguments=arguments,
            actor="chat:web",
            source="chat",
            principal="operator",
        )
    )

    assert result.success is False
    assert result.decision == "denied"
    assert "secret preflight" in (result.error or "")
    assert SECRET not in (result.error or "")
    assert env["executed"] == []
    # Refusal category + digest are recorded on the result, never the raw value.
    refusal = result.metadata["secret_refusal"]
    assert refusal["digest"] == expected_digest
    assert refusal["category"] == "secret_detected"
    assert SECRET not in json.dumps(refusal, default=str)

    # Authorization was never reached, so no approval could be created.
    assert authorized == []
    approvals = await db.fetchone("SELECT COUNT(*) AS n FROM approvals")
    assert approvals["n"] == 0
    # No quota was charged.
    quotas = await db.fetchone("SELECT COUNT(*) AS n FROM tool_quotas")
    assert quotas["n"] == 0

    sinks: list[str] = []

    call_row = await db.fetchone(
        "SELECT args,error FROM tool_calls WHERE tool='secret_probe'"
    )
    assert call_row is not None
    sinks.append(str(call_row["args"]))
    sinks.append(str(call_row["error"]))

    audit_rows = await db.fetchall(
        "SELECT reason,metadata FROM audit_log WHERE action='tool.execute'"
    )
    assert audit_rows
    for row in audit_rows:
        sinks.append(str(row["reason"]))
        sinks.append(str(row["metadata"]))

    event_rows = await db.fetchall("SELECT topic,payload FROM events")
    assert event_rows
    for row in event_rows:
        sinks.append(str(row["payload"]))

    sinks.append(caplog.text)

    for sink in sinks:
        assert SECRET not in sink, "raw refused secret reached a durable/log sink"

    # The persisted tool_calls args are the sanitized projection.
    stored = json.loads(call_row["args"])
    assert stored == {
        "refused": "secret_detected",
        "rule": refusal["rule"],
        "digest": expected_digest,
    }
    # Audited metadata names the refusal but never the text.
    audit_meta = json.loads(audit_rows[0]["metadata"])
    assert audit_meta["secret_refusal"]["digest"] == expected_digest


async def test_schedule_tools_are_opted_in_and_refuse_before_the_handler(env):
    """The first opt-in tools are schedule tools; refusal happens before any
    Scheduler dependency is resolved or a schedule row is written."""
    schedule_registry = ToolRegistry()
    assert automation_tools.register(schedule_registry) == [
        "schedule_create", "schedule_list", "schedule_get", "schedule_update",
        "schedule_pause", "schedule_resume", "schedule_delete",
        "schedule_run_now", "schedule_cancel",
    ]
    specs = {spec.name: spec for spec in schedule_registry.list_specs()}
    assert all(spec.secret_scan is True for spec in specs.values())

    executor = ToolExecutorImpl(
        schedule_registry,
        PermissionManagerImpl(env["settings"], env["events"]),
        env["events"],
        env["settings"],
    )
    result = await executor.execute(
        ToolCall(
            name="schedule_create",
            arguments={
                "name": "leaky",
                "schedule": "*/5 * * * *",
                "prompt": f"send this credential onward {SECRET}",
            },
            actor="chat:web",
            source="chat",
            principal="operator",
        )
    )

    assert result.success is False
    assert result.decision == "denied"
    assert "secret preflight" in (result.error or "")
    assert await db.fetchone("SELECT id FROM schedules WHERE name='leaky'") is None
    # No approval and no quota charge for the refused call.
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM approvals"))["n"] == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM tool_quotas"))["n"] == 0


async def test_safe_arguments_still_dispatch(env):
    result = await env["executor"].execute(
        ToolCall(
            name="secret_probe",
            arguments={"text": "a perfectly ordinary reminder", "nested": {"n": 1}},
            actor="chat:web",
            source="chat",
            principal="operator",
        )
    )
    assert result.success is True
    assert env["executed"][0]["text"] == "a perfectly ordinary reminder"


# Vendor-prefix-free credentials must be caught by intrinsic shape detection,
# not by a catalog of known token prefixes. Neither string appears in a rule.
OPAQUE_TOKEN = "aB3dEf7gHj9kLm2nPq4rSt6vWx8yZ0aB3dEf7g"
PARTIAL_KEY = "-----BEGIN RSA PRIVATE KEY"


@pytest.mark.parametrize("secret", [OPAQUE_TOKEN, PARTIAL_KEY])
async def test_intrinsic_credentials_are_refused_before_every_sink(env, caplog, secret):
    caplog.set_level(logging.DEBUG)
    arguments = {"text": "remember this", "nested": {"value": secret}}

    result = await env["executor"].execute(
        ToolCall(
            name="secret_probe", arguments=arguments, actor="chat:web",
            source="chat", principal="operator",
        )
    )

    assert result.success is False
    assert result.decision == "denied"
    assert "secret preflight" in (result.error or "")
    assert secret not in (result.error or "")
    assert env["executed"] == []
    assert result.metadata["secret_refusal"]["digest"] == arguments_digest(arguments)
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM approvals"))["n"] == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM tool_quotas"))["n"] == 0

    sinks: list[str] = [caplog.text]
    row = await db.fetchone("SELECT args,error FROM tool_calls WHERE tool='secret_probe'")
    sinks.extend([str(row["args"]), str(row["error"])])
    for audit in await db.fetchall("SELECT reason,metadata FROM audit_log"):
        sinks.extend([str(audit["reason"]), str(audit["metadata"])])
    for event in await db.fetchall("SELECT payload FROM events"):
        sinks.append(str(event["payload"]))
    for sink in sinks:
        assert secret not in sink
    stored = json.loads(row["args"])
    assert set(stored) == {"refused", "rule", "digest"}


async def test_ordinary_tool_sinks_redact_intrinsic_secrets_without_a_refusal(env, caplog):
    """Opt-in tools refuse; every other tool still cannot persist a credential."""
    caplog.set_level(logging.DEBUG)
    registry = ToolRegistry()

    async def handler(args: dict, ctx: dict):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"upstream replied with {OPAQUE_TOKEN}")

    registry.register(
        ToolSpec(
            name="ordinary_probe", description="ordinary tool", category="test",
            risk=RiskLevel.LOW, resource_field="", input_schema=_probe_schema(),
        ),
        handler,
    )
    executor = ToolExecutorImpl(
        registry, PermissionManagerImpl(env["settings"], env["events"]),
        env["events"], env["settings"],
    )
    result = await executor.execute(
        ToolCall(
            name="ordinary_probe", arguments={"text": OPAQUE_TOKEN},
            actor="chat:web", source="chat", principal="operator",
        )
    )

    assert result.decision == "error"  # the handler ran; the tool is not opt-in
    for sink in (result.error or "", caplog.text):
        assert OPAQUE_TOKEN not in sink
    row = await db.fetchone("SELECT args,error FROM tool_calls WHERE tool='ordinary_probe'")
    for value in (row["args"], row["error"]):
        assert OPAQUE_TOKEN not in value
    for audit in await db.fetchall("SELECT reason,metadata FROM audit_log WHERE action='tool.execute'"):
        assert OPAQUE_TOKEN not in str(audit["reason"])
        assert OPAQUE_TOKEN not in str(audit["metadata"])
    for event in await db.fetchall("SELECT payload FROM events"):
        assert OPAQUE_TOKEN not in str(event["payload"])


async def test_private_key_body_is_redacted_at_every_sink(env, caplog):
    """A complete PEM block must not leave its base64 body in a durable sink."""
    caplog.set_level(logging.DEBUG)
    body = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="
    pem = f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"
    registry = ToolRegistry()

    async def handler(args: dict, ctx: dict):  # type: ignore[no-untyped-def]
        return {"echo": dict(args)}

    registry.register(
        ToolSpec(
            name="pem_probe", description="ordinary tool", category="test",
            risk=RiskLevel.LOW, resource_field="", input_schema=_probe_schema(),
        ),
        handler,
    )
    executor = ToolExecutorImpl(
        registry, PermissionManagerImpl(env["settings"], env["events"]),
        env["events"], env["settings"],
    )
    result = await executor.execute(
        ToolCall(
            name="pem_probe", arguments={"text": pem}, actor="chat:web",
            source="chat", principal="operator",
        )
    )
    assert result.success is True
    sinks = [caplog.text, json.dumps(result.model_dump(mode="json"), default=str)]
    row = await db.fetchone("SELECT args FROM tool_calls WHERE tool='pem_probe'")
    sinks.append(row["args"])
    for event in await db.fetchall("SELECT payload FROM events"):
        sinks.append(event["payload"])
    for sink in sinks:
        assert body not in sink
        assert "PRIVATE KEY" not in sink


def _probe_schema() -> dict:
    return {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "additionalProperties": False,
    }
