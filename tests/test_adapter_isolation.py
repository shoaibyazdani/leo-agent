"""Regressions: adapter tests must be OS-isolated and registration fail-closed."""

from __future__ import annotations

import hashlib

import pytest

from leo import db
from leo.events import EventBus
from leo.integrations import factory as factory_module
from leo.integrations.factory import AdapterTestError, IntegrationFactory
from leo.security.permissions import ApprovalRequired

CANDIDATE = {
    "name": "isolated_notes",
    "description": "Search local notes through a declarative interface.",
    "version": "1.0.0",
    "license": "MIT",
    "parameters": {"query": {"type": "string", "description": "query", "required": True}},
    "permissions": ["skill:read", "skill:run"],
}

PARENT_CALLS: list[str] = []
TRUSTED_CALLS: list[str] = []


async def _prepared(candidate: dict | None = None, **kwargs) -> IntegrationFactory:
    factory = IntegrationFactory(events=EventBus(), **kwargs)
    entry = candidate or CANDIDATE
    await factory.discover([entry])
    await factory.evaluate(entry["name"])
    await factory.generate_adapter(entry["name"])
    return factory


def _expect_hash(factory: IntegrationFactory, name: str) -> str:
    return hashlib.sha256(factory._adapters[name].encode("utf-8")).hexdigest()


def _pass(adapter):
    import os

    return {
        "passed": (
            adapter.name == "isolated_notes"
            and adapter.validate({"query": "hello"}) == []
            and os.environ.get("LEO_ADAPTER_CANARY") is None
            and 0 < LEO_CPU_LIMIT_SECONDS <= 10
            and 0 < LEO_MEMORY_LIMIT_BYTES <= 512 * 1024 * 1024
        ),
        "cpu_limit": LEO_CPU_LIMIT_SECONDS,
        "memory_limit": LEO_MEMORY_LIMIT_BYTES,
        "canary": os.environ.get("LEO_ADAPTER_CANARY"),
    }


def _returns_none(adapter):
    return None


def _returns_failure(adapter):
    return {"passed": False, "error": "boom", "stage": "child"}


def _returns_truthy_lookalike(adapter):
    return {"ok": True, "success": True}


def _parent_side_effect(adapter):
    PARENT_CALLS.append("executed-in-parent")
    return True


def _trusted_probe(adapter):
    TRUSTED_CALLS.append("executed-with-opt-in")
    return True


def _sleep_forever(adapter):
    import time

    time.sleep(30)
    return True


def _allocates_too_much(adapter):
    import time

    blob = bytearray(96 * 1024 * 1024)
    for offset in range(0, len(blob), 4096):
        blob[offset] = 1
    time.sleep(1)
    return True


async def test_child_process_is_isolated_scrubbed_and_bounded(monkeypatch):
    monkeypatch.setenv("LEO_ADAPTER_CANARY", "must-not-leak")
    factory = await _prepared()
    report = await factory.test_adapter(CANDIDATE["name"], _pass)

    assert report.passed is True and report.error is None and report.failure is None
    assert report.mode == "isolated"
    assert report.source_sha256 == _expect_hash(factory, CANDIDATE["name"])
    assert report.review_version == "1.0.0"
    assert report.details["canary"] is None
    stored = await db.fetchone(
        "SELECT value FROM kv WHERE key=?", (factory._kv_key(CANDIDATE["name"], "test"),)
    )
    assert stored
    import json

    assert json.loads(stored["value"])["source_sha256"] == report.source_sha256


async def test_untrusted_test_callable_does_not_execute_in_parent(monkeypatch):
    PARENT_CALLS.clear()
    monkeypatch.setenv("LEO_ADAPTER_CANARY", "must-not-leak")
    factory = await _prepared()
    report = await factory.test_adapter(CANDIDATE["name"], _parent_side_effect)

    assert report.passed is False
    assert report.failure["code"] == "isolated_test_error"
    assert PARENT_CALLS == []


@pytest.mark.parametrize("callable_,raw", [
    (_returns_none, None),
    (_returns_failure, {"passed": False, "error": "boom", "stage": "child"}),
    (_returns_truthy_lookalike, {"ok": True, "success": True}),
])
async def test_only_explicit_pass_counts_and_raw_detail_is_retained(callable_, raw):
    factory = await _prepared()
    report = await factory.test_adapter(CANDIDATE["name"], callable_)

    assert report.passed is False
    assert report.failure["code"] == "non_passing_test_result"
    assert report.failure["raw"] == raw
    assert report.details == raw


async def test_timeout_kills_child_with_structured_failure():
    factory = await _prepared()
    report = await factory.test_adapter(CANDIDATE["name"], _sleep_forever, timeout=0.05)

    assert report.passed is False
    assert report.failure["code"] == "isolation_timeout"
    assert "timed out" in report.error
    assert report.source_sha256 == _expect_hash(factory, CANDIDATE["name"])


async def test_memory_budget_kills_child(monkeypatch):
    monkeypatch.setattr(factory_module, "_ISOLATED_MEMORY_BYTES", 48 * 1024 * 1024)
    factory = await _prepared()
    report = await factory.test_adapter(CANDIDATE["name"], _allocates_too_much, timeout=3)

    assert report.passed is False
    assert report.failure["code"] == "isolation_process_failed"
    assert report.failure["exit_code"] != 0


async def test_in_process_path_requires_explicit_trusted_internal_flag():
    PARENT_CALLS.clear()
    TRUSTED_CALLS.clear()
    factory = await _prepared()
    blocked = await factory.test_adapter(
        CANDIDATE["name"], _parent_side_effect, trusted_internal=True
    )
    assert blocked.passed is False
    assert blocked.failure["code"] == "trusted_internal_flag_required"
    assert PARENT_CALLS == []

    trusted = await _prepared(trusted_internal=True)
    report = await trusted.test_adapter(
        CANDIDATE["name"], _trusted_probe, trusted_internal=True
    )
    assert report.passed is True and report.mode == "trusted-in-process"
    assert TRUSTED_CALLS == ["executed-with-opt-in"]


async def test_register_requires_a_passing_report():
    factory = await _prepared()
    name = CANDIDATE["name"]

    with pytest.raises(AdapterTestError) as missing:
        await factory.register(name, ["skill:read", "skill:run"])
    assert missing.value.code == "adapter_test_missing"
    assert missing.value.details["expected_source_sha256"] == _expect_hash(factory, name)

    failed = await factory.test_adapter(name, _returns_failure)
    assert failed.passed is False
    with pytest.raises(AdapterTestError) as rejected:
        await factory.register(name, ["skill:read"])
    assert rejected.value.code == "adapter_test_failed"
    assert rejected.value.details["details"] == {"passed": False, "error": "boom", "stage": "child"}

    passed = await factory.test_adapter(name, _pass)
    assert passed.passed is True
    with pytest.raises(ApprovalRequired) as pending:
        await factory.register(name, ["skill:read", "skill:run", "shell:exec"])
    approval_id = pending.value.approval_id
    await factory.permissions.resolve_approval(approval_id, True, resolver="operator")
    record = await factory.register(
        name, ["skill:read", "skill:run", "shell:exec"], provenance_approval_id=approval_id
    )
    assert record.permissions == ["skill:read", "skill:run"]
    assert record.provenance_verified and record.provenance_approval_id == approval_id
    assert (await factory.versions(name))[0]["test"]["source_sha256"] == passed.source_sha256


async def test_register_rejects_stale_source_and_review_bindings():
    factory = await _prepared()
    name = CANDIDATE["name"]
    report = await factory.test_adapter(name, _pass)
    assert report.passed is True
    payload = report.model_dump(mode="json")
    key = factory._kv_key(name, "test")

    await factory._kv_set(key, {**payload, "source_sha256": "0" * 64})
    with pytest.raises(AdapterTestError) as stale_source:
        await factory.register(name)
    assert stale_source.value.code == "adapter_test_stale_source"
    assert stale_source.value.details["expected_source_sha256"] == _expect_hash(factory, name)

    await factory._kv_set(key, {**payload, "review_version": "9.9.9"})
    with pytest.raises(AdapterTestError) as stale_review:
        await factory.register(name)
    assert stale_review.value.code == "adapter_test_stale_review"
    assert stale_review.value.details["expected_review_version"] == "1.0.0"

    upgraded = {**CANDIDATE, "version": "2.0.0"}
    await factory.discover([upgraded])
    await factory.evaluate(name)
    await factory.generate_adapter(name)
    with pytest.raises(AdapterTestError) as regenerated:
        await factory.register(name)
    assert regenerated.value.code == "adapter_test_stale_source"
