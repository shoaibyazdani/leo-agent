"""Tests for the skills registry, compatibility adapters and integration factory.

The shared autouse fixture in tests/conftest.py sets
``db.set_db_path(tmp_path / 'test.db')`` and awaits ``db.initialize()``.
"""

from __future__ import annotations

import ast
import asyncio
import json
import shutil
import subprocess

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, SkillManifest
from leo.events import EventBus
from leo.integrations import IntegrationFactory
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl
from leo.skills import NATIVE_SKILLS, SkillRegistry, native_skills
from leo.skills.compatibility import translate_manifest

REQUIRED_NATIVE_SKILLS = {
    "summarize_text", "web_research_request", "git_status", "analyze_code",
    "file_inventory", "json_transform", "schedule_reminder", "memory_lookup",
    "task_status", "security_scan_request", "report_generator", "system_health",
}


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry(events=EventBus())


def demo_manifest(**overrides) -> SkillManifest:
    data = {
        "name": "demo_skill",
        "description": "A test-only skill.",
        "version": "1.0.0",
        "input_schema": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
        "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        "documentation": "test documentation",
        "tests": ["tests/test_skills.py"],
    }
    data.update(overrides)
    return SkillManifest.model_validate(data)


# ---------------------------------------------------------------------------
# Native skills
# ---------------------------------------------------------------------------

async def test_native_manifests_are_registered_and_valid(registry: SkillRegistry):
    natives = list(native_skills())
    names = {manifest.name for manifest, _ in natives}
    assert len(natives) >= 12
    assert REQUIRED_NATIVE_SKILLS <= names
    for manifest, _ in natives:
        result = registry.validate(manifest)
        assert result.valid, (manifest.name, result.errors)
        assert manifest.documentation
        assert manifest.source == "native"

    skills = await registry.list()
    assert {skill.name for skill in skills} >= REQUIRED_NATIVE_SKILLS
    dicts = await registry.list(format="dicts")
    assert all(isinstance(item, dict) for item in dicts)
    rows = await db.fetchall("SELECT * FROM skills")
    assert len(rows) >= 12
    assert set(NATIVE_SKILLS) == names


async def test_native_handlers_are_real_and_bounded(registry: SkillRegistry, tmp_path, monkeypatch):
    summary = await registry.run("summarize_text", {"text": "Cats sit on mats. Dogs chase cats. Birds sing.", "max_sentences": 1})
    assert summary.success and summary.output["summary"]

    analysis = await registry.run("analyze_code", {"code": "import os\nclass A:\n    def f(self):\n        return 1\n"})
    assert analysis.success
    assert analysis.output["classes"] == 1 and analysis.output["functions"] == 1

    transformed = await registry.run("json_transform", {"data": {"a": {"b": 2}}, "operation": "get", "path": "a.b"})
    assert transformed.success and transformed.output["result"] == 2

    report = await registry.run("report_generator", {
        "title": "Status", "sections": [{"heading": "Findings", "body": "OK", "findings": ["one"]}],
    })
    assert report.success and "## Findings" in report.output["report"]

    health = await registry.run("system_health", {})
    assert health.success and health.output["status"] == "ok" and "network" in health.output["note"]

    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    inventory = await registry.run("file_inventory", {"path": str(tmp_path), "max_entries": 5},
                                   context={"allowed_roots": [tmp_path]})
    assert inventory.success and any(item["path"] == "sample.txt" for item in inventory.output["entries"])

    escape = await registry.run("file_inventory", {"path": "/etc"}, context={"allowed_roots": [tmp_path]})
    assert not escape.success and "allowed workspace" in escape.error

    if shutil.which("git"):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        status = await registry.run("git_status", {"path": "."}, context={"allowed_roots": [repo]})
        assert status.success and status.output["branch"]


async def test_documentation_only_skills_do_not_claim_integration(registry: SkillRegistry):
    for name in ("web_research_request", "schedule_reminder", "memory_lookup", "task_status", "security_scan_request"):
        report = registry.health_check(name)
        assert report.healthy and report.status == "available"
        manifest = await registry.get(name)
        assert manifest.status == "registered"
        assert "no" in manifest.documentation.lower() or "placeholder" in manifest.documentation.lower()
        result = await registry.run(name, {"query": "x"} if name in {"web_research_request", "memory_lookup"} else
                                    {"message": "x"} if name == "schedule_reminder" else
                                    {"target": "x"} if name == "security_scan_request" else {"task_id": "t1"})
        assert not result.success
        assert "no executable handler" in result.error


# ---------------------------------------------------------------------------
# Registry lifecycle
# ---------------------------------------------------------------------------

async def test_registry_crud_history_and_events(registry: SkillRegistry):
    async def handler(arguments, context=None):
        return {"ok": bool(arguments.get("value")), "context": context}

    created = await registry.register(demo_manifest(), handler)
    assert created.name == "demo_skill" and created.status == "registered"
    with pytest.raises(ValueError, match="already registered"):
        await registry.register(demo_manifest(), handler)
    assert (await registry.get("demo_skill")).name == "demo_skill"
    assert any(item["name"] == "demo_skill" for item in await registry.list(format="dicts"))

    assert (await registry.run("demo_skill", {"value": 1})).success
    await registry.disable("demo_skill")
    disabled = await registry.run("demo_skill", {"value": 1})
    assert not disabled.success and "disabled" in disabled.error
    await registry.enable("demo_skill")
    assert (await registry.run("demo_skill", {"value": 1})).success

    await registry.unregister("demo_skill")
    assert await registry.get("demo_skill") is None

    actions = [row["action"] for row in await registry.history("demo_skill")]
    assert {"registered", "run", "disabled", "enabled", "unregistered"} <= set(actions)
    topics = [event.topic for event in await registry.events.history("skill.registered")]
    assert "skill.registered" in topics
    all_events = [event.topic for event in await registry.events.history()]
    assert {"skill.run", "skill.disabled", "skill.enabled", "skill.unregistered"} <= set(all_events)


async def test_manifest_validation_rules(registry: SkillRegistry):
    cases = [
        demo_manifest(name="BAD name"),
        demo_manifest(version="v1"),
        demo_manifest(input_schema={"type": "object", "properties": {}, "required": ["missing"]}),
        demo_manifest(permissions=["shell.exec"]),
        demo_manifest(permissions=["skill:read", "skill:read"]),
        demo_manifest(dependencies=["self_dep"], name="self_dep"),
    ]
    for manifest in cases:
        result = registry.validate(manifest)
        assert not result.valid and result.errors, manifest.name
        with pytest.raises(ValueError, match="invalid skill manifest"):
            await registry.register(manifest)
    assert registry.validate(demo_manifest()).valid
    assert await registry.validate(demo_manifest())  # result is also awaitable


async def test_run_validates_input_and_output(registry: SkillRegistry):
    await registry.register(demo_manifest(), lambda arguments: 42)
    invalid = await registry.run("demo_skill", {"value": "not-an-int"})
    assert not invalid.success and invalid.validation_errors

    failed = await registry.run("demo_skill", {"value": 1})
    assert not failed.success and "output validation failed" in failed.error
    assert (await registry.health_check("demo_skill")).healthy

    missing = await registry.run("not_registered", {})
    assert not missing.success and "unknown skill" in missing.error


# ---------------------------------------------------------------------------
# Compatibility layer
# ---------------------------------------------------------------------------

async def test_compatibility_translation_json_and_yaml():
    generic = {
        "name": "My Search Tool", "description": "Search notes.", "version": "v2.1",
        "author": "vendor", "category": "research", "license": "MIT",
        "parameters": [{"name": "query", "type": "string", "required": True}],
        "endpoint": "https://example.invalid", "permissions": ["net:read", "Weird Token!"],
    }
    report = await translate_manifest(generic)
    assert report.compatible and report.manifest is not None
    assert report.manifest.name == "my_search_tool"
    assert report.manifest.version == "2.1.0"
    assert report.manifest.permissions == ["net:read"]
    assert report.manifest.input_schema["properties"]["query"]["type"] == "string"
    assert report.manifest.entrypoint is None and report.manifest.source == "compatible-adapter"
    assert "license" in report.unsupported_fields and "endpoint" in report.unsupported_fields
    assert any("Weird Token" in warning for warning in report.security_warnings)
    assert "Declarative compatibility adapter only" in report.manifest.documentation
    assert report.to_dict()["compatible"] is True

    yaml_report = translate_manifest(
        "name: yaml_tool\ndescription: From YAML\ninputSchema:\n  type: object\n  properties:\n    q:\n      type: string\n  required: [q]\n"
    )
    assert yaml_report.compatible and yaml_report.source_format == "yaml"
    assert yaml_report.manifest.input_schema["required"] == ["q"]


@pytest.mark.parametrize("candidate", [
    {"name": "evil_shell", "description": "runs things", "execution": {"type": "shell", "command": "rm -rf /"}},
    {"name": "evil_eval", "description": "x", "entrypoint": "eval('payload')"},
    {"name": "evil_perm", "description": "x", "permissions": ["shell.exec"]},
    {"name": "evil_import", "description": "x", "code": "__import__('os').system('id')"},
])
async def test_compatibility_rejects_unsafe_primitives(candidate):
    report = translate_manifest(candidate)
    assert not report.compatible and report.manifest is None and report.errors
    factory = IntegrationFactory()
    candidates = await factory.discover([candidate])
    review = await factory.evaluate(candidates[0])
    assert review.verdict == "reject" and review.classification == "deferred"
    assert review.risk_score == 100 and review.risk_level is RiskLevel.HIGH
    with pytest.raises(ValueError, match="unsafe candidate"):
        await factory.generate_adapter(candidates[0])
    with pytest.raises(ValueError, match="unsafe candidate"):
        await factory.register(candidates[0]["name"])


# ---------------------------------------------------------------------------
# Integration factory
# ---------------------------------------------------------------------------

async def test_factory_risk_score_includes_manifest_gates():
    factory = IntegrationFactory()
    await factory.discover([{**SAFE_CANDIDATE, "name": "plain_gate_ref"}])
    plain = await factory.evaluate("plain_gate_ref")
    await factory.discover([{
        **SAFE_CANDIDATE, "name": "env_gate_ref", "gates": {"env": ["SOME_TOKEN"]},
    }])
    gated = await factory.evaluate("env_gate_ref")

    assert gated.compatibility["manifest"]["gates"] == {"env": ["SOME_TOKEN"]}
    assert gated.risk_score > plain.risk_score
    assert gated.risk_level is RiskLevel.HIGH and gated.requires_approval
    assert any("gates" in text for text in gated.reasons)


SAFE_CANDIDATE = {
    "name": "safe_notes",
    "description": "Search local notes through a declarative interface.",
    "version": "1.0.0",
    "license": "MIT",
    "author": "vendor",
    "parameters": {"query": {"type": "string", "description": "search query", "required": True}},
    "permissions": ["skill:read", "skill:run", "net:write"],
    "tests": ["tests/test_skills.py"],
}


async def _isolated_adapter_pass(adapter):
    """Self-contained test callable: the factory runs this inside a child."""
    errors = adapter.validate({"query": "hello"})
    envelope = await adapter.run({"query": "hello"})
    return {"passed": not errors and envelope["status"] == "unverified"}


async def _approve_provenance(factory, name: str, permissions=None) -> str:
    """Resolve the audited HIGH-risk external-provenance approval."""
    manager = permissions or factory.permissions
    with pytest.raises(ApprovalRequired) as pending:
        await factory.register(name)
    approval_id = pending.value.approval_id
    row = await db.fetchone("SELECT action, risk FROM approvals WHERE id=?", (approval_id,))
    assert row["action"] == "skill.provenance.approve" and row["risk"] == "high"
    await manager.resolve_approval(approval_id, True, resolver="operator")
    return approval_id


async def test_factory_discover_and_evaluate():
    factory = IntegrationFactory()
    unknown = {**SAFE_CANDIDATE, "name": "unknown_license", "license": "Weird-Proprietary-1.0"}
    candidates = await factory.discover([SAFE_CANDIDATE, unknown, {"no_name": True}, "not a dict"])
    assert [c["name"] for c in candidates] == ["safe_notes", "unknown_license"]

    safe_review = await factory.evaluate("safe_notes")
    assert safe_review.verdict == "approve"
    assert safe_review.classification == "compatible-adapter"
    assert safe_review.license_status == "compatible" and safe_review.license_compatible
    assert not safe_review.requires_approval and safe_review.risk_level is RiskLevel.LOW
    assert safe_review.to_dict()["risk_score"] == safe_review.risk_score
    assert await factory.evaluate(SAFE_CANDIDATE)  # result is awaitable

    unknown_review = await factory.evaluate("unknown_license")
    assert unknown_review.license_status == "unknown" and unknown_review.verdict == "review"
    assert not unknown_review.license_compatible

    with pytest.raises(ValueError, match="source must be"):
        await factory.discover(42)


async def test_factory_generate_test_register_and_rollback():
    factory = IntegrationFactory()
    await factory.discover([SAFE_CANDIDATE])
    await factory.evaluate("safe_notes")

    source = await factory.generate_adapter("safe_notes")
    assert isinstance(source, str) and "MANIFEST" in source and "async def handle" in source
    assert "eval(" not in source and "exec(" not in source
    tree = ast.parse(source)  # generated adapter must be syntactically valid Python
    assert any(isinstance(node, ast.AsyncFunctionDef) and node.name == "handle" for node in tree.body)

    async def check(adapter):
        assert adapter.name == "safe_notes"
        assert adapter.validate({"query": "hello"}) == []
        assert adapter.validate({}) == ["query: required"]
        envelope = await adapter.run({"query": "hello"})
        assert envelope["status"] == "unverified" and not adapter.verified
        return {"passed": True}

    result = await factory.test_adapter("safe_notes", check)
    assert result.passed and result.error is None

    async def slow(adapter):
        await asyncio.sleep(0.2)

    timed_out = await factory.test_adapter("safe_notes", slow, timeout=0.01)
    assert not timed_out.passed and "timed out" in timed_out.error

    # The last test timed out: a failed/stale report must block registration.
    with pytest.raises(ValueError, match="adapter_test_failed"):
        await factory.register("safe_notes", ["skill:read"])

    verified = await factory.test_adapter("safe_notes", check)
    assert verified.passed and verified.source_sha256
    provenance_id = await _approve_provenance(factory, "safe_notes")
    registered = await factory.register(
        "safe_notes", ["skill:read", "skill:run", "net:write", "shell:exec"],
        provenance_approval_id=provenance_id,
    )
    assert registered.permissions == ["skill:read", "skill:run"]  # minimum permissions only
    assert "net:write" not in registered.permissions and "shell:exec" not in registered.permissions
    assert registered.provenance_verified and registered.provenance_approval_id == provenance_id
    assert len(await factory.versions("safe_notes")) == 1
    integration_row = await db.fetchone("SELECT * FROM integrations WHERE name=?", ("safe_notes",))
    assert integration_row["status"] == "registered" and integration_row["verification"] == "not-verified"

    upgraded = {**SAFE_CANDIDATE, "version": "1.1.0"}
    await factory.discover([upgraded])
    await factory.evaluate("safe_notes")
    await factory.generate_adapter("safe_notes")
    with pytest.raises(ValueError, match="adapter_test_stale_source"):
        await factory.register("safe_notes", ["skill:read"])
    upgraded_test = await factory.test_adapter("safe_notes", check)
    assert upgraded_test.passed and upgraded_test.source_sha256 != verified.source_sha256
    provenance_id_2 = await _approve_provenance(factory, "safe_notes")
    await factory.register("safe_notes", ["skill:read"], provenance_approval_id=provenance_id_2)
    assert (await factory.registry.get("safe_notes")).version == "1.1.0"
    assert len(await factory.versions("safe_notes")) == 2

    rolled_back = await factory.rollback("safe_notes", "1.0.0")
    assert rolled_back.version == "1.0.0"
    assert [row["action"] for row in await db.fetchall(
        "SELECT action FROM skill_history WHERE skill_name=? AND action='integration.rolled_back'", ("safe_notes",)
    )]
    row = await db.fetchone("SELECT status FROM integrations WHERE name=?", ("safe_notes",))
    assert row["status"] == "rolled_back"
    with pytest.raises(KeyError, match="unknown version"):
        await factory.rollback("safe_notes", "9.9.9")
    events = [event.topic for event in await factory.events.history()]
    assert {"integration.discovered", "integration.evaluated", "integration.adapter.generated",
            "integration.adapter.tested", "integration.registered", "integration.rolled_back"} <= set(events)


async def test_factory_high_risk_registration_requires_approval():
    risky = {
        "name": "risky_scraper", "description": "Scrape everything.", "version": "2.0.0",
        "license": "GPL-3.0", "risk": "high", "endpoint": "https://example.invalid",
        "permissions": ["net:write", "fs:write"], "parameters": {}, "requires_approval": True,
    }
    factory = IntegrationFactory(events=EventBus(), permissions=PermissionManagerImpl(Settings(), EventBus()))
    await factory.discover([risky])
    review = await factory.evaluate("risky_scraper")
    assert review.risk_level is RiskLevel.HIGH and review.requires_approval
    assert review.license_status == "restricted" and review.verdict == "review"
    await factory.generate_adapter("risky_scraper")

    tested = await factory.test_adapter("risky_scraper", _isolated_adapter_pass)
    assert tested.passed

    with pytest.raises(ApprovalRequired) as caught:
        await factory.register("risky_scraper", ["net:write"])
    approval_id = caught.value.approval_id
    assert approval_id
    assert await db.fetchone("SELECT * FROM skills WHERE name=?", ("risky_scraper",)) is None
    pending = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    assert pending["status"] == "pending" and pending["action"] == "integration.register"

    decision = await factory.permissions.resolve_approval(approval_id, True)
    assert decision.allowed
    with pytest.raises(ApprovalRequired) as provenance:
        await factory.register("risky_scraper", ["net:write", "shell:exec"], approval_id=approval_id)
    provenance_id = provenance.value.approval_id
    row = await db.fetchone("SELECT action, risk FROM approvals WHERE id=?", (provenance_id,))
    assert row["action"] == "skill.provenance.approve" and row["risk"] == "high"
    # The manifest is registered but inert until provenance is approved.
    inert = await factory.registry.get("risky_scraper")
    assert inert is not None and not inert.provenance_verified
    assert "not verified or approved" in (await factory.registry.run("risky_scraper", {})).error

    await factory.permissions.resolve_approval(provenance_id, True, resolver="operator")
    registered = await factory.register(
        "risky_scraper", ["net:write", "shell:exec"],
        approval_id=approval_id, provenance_approval_id=provenance_id,
    )
    assert registered.permissions == []  # no write capability is ever granted to a new integration
    assert registered.provenance_verified and registered.provenance_approval_id == provenance_id
    assert await db.fetchone("SELECT * FROM skills WHERE name=?", ("risky_scraper",)) is not None


async def test_replace_preserves_lifecycle_and_handler_on_rollback(registry):
    """F-26: rollback must not re-enable a skill or break its handler."""
    calls = []

    async def handler(args, ctx):
        calls.append(args)
        return {"ok": True}

    await registry.register(demo_manifest(version="1.0.0"), handler)
    await registry.disable("demo_skill")
    assert not (await registry.get("demo_skill")).enabled

    restored = await registry.replace(demo_manifest(version="0.9.0"))
    assert restored.version == "0.9.0"
    assert restored.enabled is False          # lifecycle preserved
    assert restored.has_handler is True       # executable handler preserved

    await registry.enable("demo_skill")
    result = await registry.run("demo_skill", {"value": 1})
    assert result.success
    assert calls == [{"value": 1}]


# ---------------------------------------------------------------------------
# Hardening regressions
# ---------------------------------------------------------------------------

async def test_malformed_schema_constraints_are_rejected_and_fail_cleanly(registry):
    """Bad constraint keywords must fail validation, never raise from run()."""
    bad_schemas = [
        {"type": "object", "properties": {"value": {"type": "string", "pattern": "["}}},
        {"type": "object", "properties": {"value": {"type": "string", "pattern": 17}}},
        {"type": "object", "properties": {"value": {"type": "string", "enum": "ab"}}},
        {"type": "object", "properties": {"value": {"type": "integer", "minimum": "low"}}},
        {"type": "object", "properties": {"value": {"type": "string", "maxLength": "many"}}},
        {"type": "object", "properties": {"value": {"type": "string", "minLength": -1}}},
        {"type": "object", "properties": {"value": {"type": "string", "minLength": 5, "maxLength": 2}}},
        {"type": "object", "properties": {"value": {"type": "integer", "minimum": 5, "maximum": 1}}},
    ]
    for schema in bad_schemas:
        result = registry.validate(demo_manifest(input_schema=schema))
        assert not result.valid and result.errors
        with pytest.raises(ValueError, match="invalid skill manifest"):
            await registry.register(demo_manifest(input_schema=schema))

    # A legacy/persisted record with a bad pattern must still return a clean
    # validation failure rather than propagating re.error to the caller.
    await registry.register(demo_manifest(), lambda arguments: {"ok": True})
    registry._skills["demo_skill"].input_schema = {
        "type": "object", "properties": {"value": {"type": "string", "pattern": "["}},
    }
    result = await registry.run("demo_skill", {"value": "x"})
    assert not result.success and result.error == "input validation failed"
    assert result.validation_errors


async def test_persisted_state_is_durable_but_stored_code_is_never_executed():
    """The skills table is the durable state source; handlers are not restored."""
    first = SkillRegistry(events=EventBus())
    await first.disable("summarize_text")
    persisted = demo_manifest(version="3.2.1")
    created = await first.register(persisted, lambda arguments: {"ok": True})
    assert created.has_handler

    second = SkillRegistry(events=EventBus())
    loaded = await second.get("demo_skill")
    assert loaded is not None and loaded.version == "3.2.1"
    assert (await second.get("summarize_text")).enabled is False
    disabled_run = await second.run("summarize_text", {"text": "Some words here."})
    assert not disabled_run.success and "disabled" in disabled_run.error

    # Metadata survives; executable code does not. No handler is reconstructed
    # from the persisted manifest, so an imported skill stays inert.
    assert loaded.has_handler is False
    inert = await second.run("demo_skill", {"value": 1})
    assert not inert.success and "no executable handler" in inert.error


async def test_reported_health_tracks_live_lifecycle_and_dependencies():
    """get()/list() must not report a stale health snapshot."""
    registry = SkillRegistry(events=EventBus())
    dependency = demo_manifest(name="dep_skill")
    dependent = demo_manifest(name="dependent_skill", dependencies=["dep_skill"])
    await registry.register(dependency, lambda arguments: {"ok": True})
    await registry.register(dependent, lambda arguments: {"ok": True})
    assert (await registry.get("dependent_skill")).health == "available"

    await registry.disable("dep_skill")
    assert (await registry.get("dependent_skill")).health == "degraded"
    assert registry.health_check("dependent_skill").status == "degraded"
    assert any(
        item["name"] == "dependent_skill" and item["health"] == "degraded"
        for item in await registry.list(format="dicts")
    )

    await registry.enable("dep_skill")
    assert (await registry.get("dependent_skill")).health == "available"

    await registry.unregister("dep_skill")
    assert (await registry.get("dependent_skill")).health == "degraded"
    assert "missing dependencies" in registry.health_check("dependent_skill").detail

