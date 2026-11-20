"""Webhook tools execute through ToolExecutorImpl against durable state.

Signatures are checked against the real WebhookHandler v1 binding, tampering is
rejected by the real verifier, and delivery history is read from real event rows
written by WebhookHandler.handle (never from a fake repository).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ToolCall
from leo.events import EventBus
from leo.integrations.channels import webhook_channels
from leo.security.permissions import PermissionManagerImpl
from leo.tasks import (
    DurableTaskEngine,
    DuplicateDelivery,
    InvalidSignature,
    WebhookHandler,
)
from leo.tools import ToolExecutorImpl, ToolRegistry, webhook_tools

TOOL_NAMES = list(webhook_tools.TOOL_NAMES)
SECRET = "webhook-tools-test-secret-7f3a"


def sender_headers(body, delivery_id, timestamp=None):
    """Produce headers as the trusted webhook sender would; never via a tool."""
    return WebhookHandler.sign_v1(
        body, SECRET, delivery_id=delivery_id, timestamp=timestamp
    )


class ServicesExecutor(ToolExecutorImpl):
    """Inject the integration-owned services mapping into each tool context."""

    def __init__(self, *args: object, services: dict | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.injected_services = services or {}

    def _context(self, call, task_key, dry_run):  # type: ignore[no-untyped-def]
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.injected_services
        return ctx


@pytest.fixture
async def env(tmp_path, monkeypatch):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    monkeypatch.setenv(webhook_tools.DEFAULT_SECRET_ENV, SECRET)
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    engine = DurableTaskEngine(settings, events)
    handler = WebhookHandler(engine, events)
    services = {"tasks": engine, "webhook": handler, "events": events}
    registry = ToolRegistry()
    assert webhook_tools.register(registry, services=services) == TOOL_NAMES
    executor = ServicesExecutor(
        registry, permissions, events, settings, services=services
    )
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions, engine=engine,
        handler=handler, services=services, registry=registry, executor=executor,
    )


async def run(env, name: str, arguments: dict | None = None):
    return await env.executor.execute(ToolCall(name=name, arguments=arguments or {}))


# --------------------------------------------------------------------------- registration


async def test_registration_specs_and_schema_validation(env):
    specs = env.registry.list_specs()
    assert [spec.name for spec in specs] == TOOL_NAMES
    by_name = {spec.name: spec for spec in specs}
    for spec in specs:
        assert spec.category == "webhook"
        assert spec.permissions and spec.description
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""  # identifiers are not file paths
    assert by_name["webhook_routes"].risk.value == "low"
    assert by_name["webhook_deliveries"].risk.value == "low"
    assert by_name["webhook_verify"].risk.value == "medium"
    assert "webhook_sign" not in by_name
    assert "webhook_sign" not in webhook_tools.TOOL_NAMES

    oracle = await run(env, "webhook_sign", {"body": {"a": 1}, "delivery_id": "d-oracle"})
    assert oracle.success is False and "unknown tool" in (oracle.error or "")
    assert oracle.output is None

    extra = await run(env, "webhook_routes", {"unknown": True})
    assert extra.success is False and extra.decision == "invalid_arguments"
    assert "unknown field" in extra.error

    over_limit = await run(env, "webhook_deliveries", {"limit": 1000})
    assert over_limit.success is False and over_limit.decision == "invalid_arguments"
    assert "maximum" in over_limit.error or "100" in over_limit.error

    wrong_type = await run(env, "webhook_verify", {"body": "x", "headers": "nope"})
    assert wrong_type.success is False and wrong_type.decision == "invalid_arguments"


# --------------------------------------------------------------------------- sign/verify


async def test_sender_signature_binding_and_no_tool_signing_oracle(env):
    body = {"objective": "deploy", "priority": 2}
    now = int(time.time())
    headers = sender_headers(body, "d-sign-1", now)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    mac = hmac.new(
        SECRET.encode("utf-8"),
        b"t=" + str(now).encode() + b",d=d-sign-1," + canonical,
        hashlib.sha256,
    ).hexdigest()
    assert headers["X-LEO-Signature-V1"] == f"v1={mac}"
    assert headers["X-Delivery-ID"] == "d-sign-1"

    oracle = await run(env, "webhook_sign", {
        "body": body, "delivery_id": "d-sign-1", "timestamp": now,
    })
    assert oracle.success is False and oracle.output is None
    assert "X-LEO-Signature-V1" not in str(oracle.output)
    assert SECRET not in json.dumps(oracle.model_dump(mode="json"))


async def test_sender_signs_raw_string_body_bytes(env):
    raw = '{"objective":"raw body"}'
    headers = sender_headers(raw, "d-raw", 1_700_000_000)
    mac = hmac.new(
        SECRET.encode("utf-8"),
        b"t=1700000000,d=d-raw," + raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert headers["X-LEO-Signature-V1"] == f"v1={mac}"


async def test_verify_accepts_real_signature_and_rejects_tampering(env):
    body = {"objective": "verified delivery", "count": 1}
    now = int(time.time())
    headers = sender_headers(body, "d-verify-1", now)

    valid = await run(env, "webhook_verify", {"body": body, "headers": headers})
    assert valid.success, valid.error
    assert valid.output["verified"] is True
    assert valid.output["delivery_id"] == "d-verify-1"
    assert valid.output["timestamp"] == now

    tampered_body = await run(env, "webhook_verify", {
        "body": {**body, "count": 2}, "headers": headers,
    })
    assert tampered_body.success is False and "signature" in tampered_body.error.lower()

    forged_id = dict(headers, **{"X-Delivery-ID": "d-other"})
    tampered_id = await run(env, "webhook_verify", {"body": body, "headers": forged_id})
    assert tampered_id.success is False and "signature" in tampered_id.error.lower()

    forged_ts = dict(headers, **{"X-Webhook-Request-Timestamp": str(now + 60)})
    tampered_ts = await run(env, "webhook_verify", {"body": body, "headers": forged_ts})
    assert tampered_ts.success is False and "signature" in tampered_ts.error.lower()

    stale = WebhookHandler.sign_v1(
        body, SECRET, delivery_id="d-stale", timestamp=now - 400
    )
    stale_result = await run(env, "webhook_verify", {"body": body, "headers": stale})
    assert stale_result.success is False and "timestamp" in stale_result.error.lower()

    no_v1 = await run(env, "webhook_verify", {
        "body": body, "headers": {"X-Delivery-ID": "d-verify-1"},
    })
    assert no_v1.success is False and "v1" in no_v1.error.lower()

    mismatch = await run(env, "webhook_verify", {
        "body": body, "headers": headers, "delivery_id": "d-other",
    })
    assert mismatch.success is False and "delivery" in mismatch.error.lower()

    # Verification is read-only: no delivery was ingested by any of the calls.
    assert await env.engine.list() == []


async def test_verify_fails_closed_without_secret(env, monkeypatch):
    now = int(time.time())
    body = {"objective": "missing secret"}
    headers = sender_headers(body, "d-nosecret", now)
    monkeypatch.delenv(webhook_tools.DEFAULT_SECRET_ENV, raising=False)

    # There is no tool that can mint a MAC, with or without the secret configured.
    assert "webhook_sign" not in {spec.name for spec in env.registry.list_specs()}
    oracle = await run(env, "webhook_sign", {
        "body": body, "delivery_id": "d-nosecret", "timestamp": now,
    })
    assert oracle.success is False and oracle.output is None

    verified = await run(env, "webhook_verify", {"body": body, "headers": headers})
    assert verified.success is False and "not set" in verified.error
    assert verified.output is None


# --------------------------------------------------------------------------- routes


async def test_routes_are_derived_from_real_channel_registry(env):
    result = await run(env, "webhook_routes")
    assert result.success, result.error
    expected = {f"/webhooks/{name}" for name in webhook_channels()}
    assert {route["path"] for route in result.output["routes"]} == expected
    assert result.output["count"] == len(expected)
    for route in result.output["routes"]:
        assert route["method"] == "POST"
        assert route["transport"] == "signed-webhook"
        assert "secret" not in route
    channel = sorted(webhook_channels())[0]
    filtered = await run(env, "webhook_routes", {"channel": channel})
    assert filtered.success, filtered.error
    assert filtered.output["count"] == 1
    assert filtered.output["routes"][0]["channel"] == channel
    assert filtered.output["routes"][0]["path"] == f"/webhooks/{channel}"
    assert result.output["signature"]["secret_configured"] is True
    assert result.output["signature"]["secret_env"] == webhook_tools.DEFAULT_SECRET_ENV
    assert result.output["task_intake"]["http_route"] is None
    assert SECRET not in json.dumps(result.output)


# --------------------------------------------------------------------------- deliveries


async def test_deliveries_are_bounded_read_only_and_real(env):
    body = {"objective": "recorded delivery"}
    first_signed = WebhookHandler.sign_v1(body, SECRET, delivery_id="d-history-0")
    for index in range(3):
        signed = (
            first_signed
            if index == 0
            else WebhookHandler.sign_v1(body, SECRET, delivery_id=f"d-history-{index}")
        )
        accepted = await env.handler.handle({"body": body, "headers": signed})
        assert accepted["status"] == "accepted"
        assert await env.engine.get(accepted["task_id"]) is not None

    # The real handler records a rejected delivery (and emits webhook.rejected)
    # for replayed signed deliveries.
    with pytest.raises(DuplicateDelivery):
        await env.handler.handle({"body": body, "headers": first_signed})

    invalid = WebhookHandler.sign_v1(body, SECRET, delivery_id="d-bad-signature")
    invalid["X-LEO-Signature-V1"] = "v1=" + "0" * 64
    with pytest.raises(InvalidSignature):
        await env.handler.handle({"body": body, "headers": invalid})

    webhook_events_before = await db.fetchall(
        "SELECT id FROM events WHERE topic IN ('webhook.accepted','webhook.rejected')"
    )
    tasks_before = await db.fetchall("SELECT id FROM tasks")
    first = await run(env, "webhook_deliveries", {"limit": 2})
    assert first.success, first.error
    assert len(first.output["deliveries"]) == 2
    assert first.output["total"] == 4
    assert first.output["limit"] == 2
    assert first.output["offset"] == 0
    assert first.output["next_offset"] == 2

    second = await run(env, "webhook_deliveries", {"limit": 2, "offset": 2})
    assert second.success, second.error
    assert len(second.output["deliveries"]) == 2
    assert second.output["next_offset"] is None
    all_ids = {
        item["delivery_id"]
        for item in first.output["deliveries"] + second.output["deliveries"]
    }
    assert all_ids == {"d-history-0", "d-history-1", "d-history-2"}

    replayed = await run(env, "webhook_deliveries", {"delivery_id": "d-history-0"})
    assert replayed.success and replayed.output["total"] == 2
    assert {item["status"] for item in replayed.output["deliveries"]} == {
        "accepted", "rejected",
    }
    rejected_entry = next(
        item for item in replayed.output["deliveries"] if item["status"] == "rejected"
    )
    assert "duplicate" in rejected_entry["reason"].lower()

    accepted = await run(env, "webhook_deliveries", {"delivery_id": "d-history-1"})
    assert accepted.success and accepted.output["total"] == 1
    entry = accepted.output["deliveries"][0]
    assert entry["status"] == "accepted"
    assert entry["delivery_id"] == "d-history-1"
    assert entry["task_id"] is not None and entry["task_status"] is not None

    unknown = await run(env, "webhook_deliveries", {"delivery_id": "d-not-found"})
    assert unknown.success and unknown.output["total"] == 0
    assert unknown.output["deliveries"] == []

    assert SECRET not in json.dumps(first.output)
    # Executor audit/tool events are expected; delivery history and tasks are not touched.
    assert await db.fetchall(
        "SELECT id FROM events WHERE topic IN ('webhook.accepted','webhook.rejected')"
    ) == webhook_events_before
    assert await db.fetchall("SELECT id FROM tasks") == tasks_before
