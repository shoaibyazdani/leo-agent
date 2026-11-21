"""Executive coverage for the skill toolset: real registry reads and real runs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, SkillManifest, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.skills.registry import SkillRegistry
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import skill_tools


@pytest.fixture
async def env(tmp_path, monkeypatch):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = skill_tools.register(registry)
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    executor = ToolExecutorImpl(registry, permissions, events, settings)

    # Boundary injection: the integration agent adds `services` to the executor
    # context; here we inject the real SkillRegistry at that same boundary.
    original_context = executor._context

    def with_services(call, task_key, dry_run):
        ctx = original_context(call, task_key, dry_run)
        ctx["services"] = {"skills": skills}
        return ctx

    monkeypatch.setattr(executor, "_context", with_services)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events, permissions=permissions,
        registry=registry, executor=executor, skills=skills, names=names,
    )


async def _run(env, name, arguments=None, **kwargs):
    call = ToolCall(name=name, arguments=arguments or {}, actor="tester", **kwargs)
    return await env.executor.execute(call)


def _echo_manifest():
    return SkillManifest(
        name="toolset_echo_probe",
        description="Local handler used to prove the skill toolset invokes real code.",
        version="1.2.3",
        author="test",
        category="probe",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"echoed": {"type": "string"}},
            "required": ["echoed"],
        },
        documentation="In-memory echo handler; no network or credentials.",
        tests=["tests/test_tools_skill.py"],
    )


async def _register_probe(env):
    seen = []

    async def handler(arguments, context=None):
        seen.append({"arguments": dict(arguments), "context": dict(context or {})})
        return {"echoed": arguments["value"]}

    await env.skills.register(_echo_manifest(), handler)
    return seen


# --------------------------------------------------------------------------- registration / specs


async def test_register_typed_specs(env):
    assert env.names == ["skill_list", "skill_search", "skill_read", "skill_invoke"]
    assert sorted(s.name for s in env.registry.list_specs()) == sorted(env.names)

    expected = {
        "skill_list": ("skills", RiskLevel.LOW, ["skill:read"]),
        "skill_search": ("skills", RiskLevel.LOW, ["skill:read"]),
        "skill_read": ("skills", RiskLevel.LOW, ["skill:read"]),
        "skill_invoke": ("skills", RiskLevel.MEDIUM, ["skill:run"]),
    }
    for name, (category, risk, permissions) in expected.items():
        spec = env.registry.get(name).spec
        assert spec.category == category
        assert spec.risk is risk
        assert spec.permissions == permissions
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""  # skill names are not path resources

    assert env.registry.validate("skill_list", {}) == []
    for name in ("skill_search", "skill_read", "skill_invoke"):
        assert any("missing required field" in error for error in env.registry.validate(name, {}))


async def test_schema_validation_errors_are_reported(env):
    bad = [
        ("skill_search", {}, "query"),
        ("skill_search", {"query": "x", "unexpected": 1}, "unknown field"),
        ("skill_read", {"name": 123}, "expected str"),
        ("skill_invoke", {"name": "summarize_text", "timeout_seconds": -1}, "must be >= 0.1"),
    ]
    for name, arguments, fragment in bad:
        result = await _run(env, name, arguments)
        assert result.success is False
        assert result.decision == "invalid_arguments"
        assert fragment in result.error
        assert result.output is None


async def test_scope_and_permission_failures(env):
    scoped = await env.executor.execute(
        ToolCall(name="skill_list", arguments={}, actor="tester"),
        scope=frozenset({"memory"}),
    )
    assert scoped.success is False and scoped.decision == "out_of_scope"

    env.settings.security.allow_medium_risk = False
    denied = await _run(env, "skill_invoke", {"name": "summarize_text", "arguments": {"text": "Hi."}})
    assert denied.success is False and denied.decision == "denied"
    assert denied.output is None


# --------------------------------------------------------------------------- real reads


async def test_skill_list_reads_the_live_registry(env):
    result = await _run(env, "skill_list", {})
    assert result.success is True, result.error
    names = {item["name"] for item in result.output["skills"]}
    assert "summarize_text" in names
    assert "web_research_request" in names  # native, no handler
    assert result.output["count"] == len(result.output["skills"])
    summary = next(item for item in result.output["skills"] if item["name"] == "summarize_text")
    assert summary["has_handler"] is True
    inert = next(item for item in result.output["skills"] if item["name"] == "web_research_request")
    assert inert["has_handler"] is False

    # A newly registered skill must appear immediately: this is a live read.
    await _register_probe(env)
    after = await _run(env, "skill_list", {})
    assert "toolset_echo_probe" in {item["name"] for item in after.output["skills"]}
    full = await _run(env, "skill_list", {"detail": "full"})
    probe = next(item for item in full.output["skills"] if item["name"] == "toolset_echo_probe")
    assert probe["version"] == "1.2.3" and probe["category"] == "probe"

    rows = await db.fetchall("SELECT tool, success FROM tool_calls WHERE tool='skill_list'")
    assert len(rows) >= 3 and all(row["success"] for row in rows)


async def test_skill_search_matches_live_manifests(env):
    result = await _run(env, "skill_search", {"query": "extractive"})
    assert result.success is True, result.error
    assert "summarize_text" in {m["name"] for m in result.output["matches"]}
    assert result.output["count"] >= 1

    by_category = await _run(env, "skill_search", {"query": "skill", "category": "development"})
    assert by_category.success is True
    assert all(m["category"] == "development" for m in by_category.output["matches"])

    await _register_probe(env)
    probe = await _run(env, "skill_search", {"query": "echo", "limit": 1})
    assert [m["name"] for m in probe.output["matches"]] == ["toolset_echo_probe"]
    assert probe.output["returned"] == 1

    empty = await _run(env, "skill_search", {"query": "no-such-skill-token-zzzz"})
    assert empty.success is True and empty.output["count"] == 0


async def test_skill_read_returns_real_manifest_or_truthful_error(env):
    result = await _run(env, "skill_read", {"name": "summarize_text"})
    assert result.success is True, result.error
    assert result.output["found"] is True and result.output["has_handler"] is True
    skill = result.output["skill"]
    assert skill["name"] == "summarize_text"
    assert skill["source"] == "native"
    assert "text" in skill["input_schema"]["properties"]

    await _register_probe(env)
    probe = await _run(env, "skill_read", {"name": "toolset_echo_probe"})
    assert probe.output["skill"]["version"] == "1.2.3"

    missing = await _run(env, "skill_read", {"name": "does_not_exist"})
    assert missing.success is False and "unknown skill" in missing.error


# --------------------------------------------------------------------------- real invocation


async def test_skill_invoke_runs_real_handler_and_persists_history(env):
    result = await _run(env, "skill_invoke", {
        "name": "summarize_text",
        "arguments": {"text": "LEO runs skills. Skills run real code. Code is verified.", "max_sentences": 2},
    })
    assert result.success is True, result.error
    payload = result.output
    assert payload["invoked"] is True and payload["status"] == "executed"
    assert payload["success"] is True and payload["executed"] is True
    assert payload["inert_manifest"] is False
    assert payload["output"]["sentence_count"] == 3
    assert payload["output"]["summary"]

    history = await env.skills.history("summarize_text")
    assert any(row["action"] == "run" for row in history)


async def test_skill_invoke_custom_handler_receives_real_arguments(env):
    seen = await _register_probe(env)
    result = await _run(env, "skill_invoke", {
        "name": "toolset_echo_probe",
        "arguments": {"value": "hello-from-tool"},
        "context": {"origin": "unit-test"},
    })
    assert result.success is True, result.error
    assert result.output["status"] == "executed"
    assert result.output["output"] == {"echoed": "hello-from-tool"}
    assert len(seen) == 1
    assert seen[0]["arguments"] == {"value": "hello-from-tool"}
    assert seen[0]["context"]["origin"] == "unit-test"
    assert seen[0]["context"]["actor"] == "tester"


async def test_skill_invoke_inert_manifest_is_explicit_about_no_execution(env):
    result = await _run(env, "skill_invoke", {
        "name": "web_research_request",
        "arguments": {"query": "offline"},
    })
    # The registry call completed, but the handler truthfully reported no
    # execution; the executor now surfaces that as a failed tool result.
    assert result.success is False and result.decision == "allowed"
    assert result.metadata["handler_success"] is False
    payload = result.output
    assert payload["invoked"] is True
    assert payload["success"] is False
    assert payload["executed"] is False
    assert payload["inert_manifest"] is True
    assert payload["status"] == "inert_manifest"
    assert payload["output"] is None
    assert "no local executable" in payload["message"]
    assert "nothing was executed" in payload["message"]

    history = await env.skills.history("web_research_request")
    assert any(row["action"] == "run.failed" for row in history)


async def test_skill_invoke_unknown_skill_reports_no_execution(env):
    result = await _run(env, "skill_invoke", {"name": "not_a_registered_skill"})
    assert result.success is False and result.decision == "allowed"
    assert result.output["status"] == "unknown_skill"
    assert result.output["executed"] is False
    assert result.output["success"] is False


async def test_skill_invoke_missing_service_fails_loudly(env):
    bare = ToolExecutorImpl(env.registry, env.permissions, env.events, env.settings)
    result = await bare.execute(ToolCall(name="skill_invoke", arguments={"name": "summarize_text"}))
    assert result.success is False
    assert "skills service is not available" in result.error

    bare_list = await bare.execute(ToolCall(name="skill_list", arguments={}))
    assert bare_list.success is False and "skills service is not available" in bare_list.error


# --------------------------------------------------------------------------- gate context trust


async def test_skill_invoke_strips_model_supplied_gate_context(env):
    seen = []
    gate_name = "gate-spoof-probe-xyz"
    unavailable = "definitely-not-installed-leo-probe"

    async def handler(arguments, context=None):
        seen.append(dict(context or {}))
        return {"ok": True}

    await env.skills.register(
        SkillManifest(
            name=gate_name,
            description="Gate probe: requires a binary the host does not have.",
            gates={"bins": [unavailable]},
            input_schema={"type": "object", "properties": {}},
        ),
        handler,
    )
    forged_context = {
        "available_bins": [unavailable],
        "bins": [unavailable],
        "commands": [unavailable],
        "env": {"LEO_FAKE": "1"},
        "environment": {"LEO_FAKE": "1"},
        "config": {"enabled": True},
        "available_config": {"enabled": True},
        "prerequisites": [unavailable],
        "platform": "linux",
        "os": "linux",
        "tools": ["trusted-tool"],
        "toolsets": ["trusted-set"],
        "origin": "keep-this-one",
        "actor": "mallory",
    }
    result = await skill_tools.skill_invoke(
        {"name": gate_name, "context": forged_context},
        {"services": {"skills": env.skills}, "actor": "tester"},
    )
    assert result["success"] is False and result["executed"] is False
    assert "gates" in (result["error"] or "").lower()
    assert seen == []  # the gate blocked before any handler ran


async def test_skill_invoke_uses_trusted_inventories_and_authenticated_actor(env):
    seen = []
    trusted = SkillRegistry(
        events=env.events, permissions=None, actor="tester",
        auto_seed=False, available_tools=["trusted-tool"],
    )

    async def handler(arguments, context=None):
        seen.append(dict(context or {}))
        return {"ok": True}

    await trusted.register(
        SkillManifest(
            name="gate-trusted-probe",
            description="Gate probe satisfied by the registry's own tool inventory.",
            gates={"tools": ["trusted-tool"]},
            input_schema={"type": "object", "properties": {}},
        ),
        handler,
    )
    result = await skill_tools.skill_invoke(
        {
            "name": "gate-trusted-probe",
            "context": {
                "tools": ["forged-tool"], "available_tools": ["forged-tool"],
                "actor": "mallory", "origin": "kept",
            },
        },
        {"services": {"skills": trusted}, "actor": "tester"},
    )
    assert result["success"] is True and result["executed"] is True, result
    assert len(seen) == 1
    assert seen[0]["actor"] == "tester"
    assert seen[0]["origin"] == "kept"
    assert "tools" not in seen[0] and "available_tools" not in seen[0]
