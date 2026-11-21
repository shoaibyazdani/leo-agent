"""Runtime-status tools: exact live-state payloads through the real executor.

Services are injected at the registration/executor boundary. Every assertion
checks a concrete synthesized status derived from fake app engines, integration
records or durable DB rows; no network, credentials or canned success.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo import __version__, db
from leo.config import Settings
from leo.contracts import IntegrationRecord, RiskLevel, ToolCall, ToolSpec
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import health_tools
from leo.tools.executor import ToolExecutorImpl
from leo.tools.health_tools import TOOL_NAMES, register
from leo.tools.registry import ToolRegistry


class InjectedExecutor(ToolExecutorImpl):
    """Executor whose service injection happens in the real context builder."""

    def __init__(self, *args, services, **kwargs):
        super().__init__(*args, **kwargs)
        self.services = services

    def _context(self, call, task_key, dry_run):
        ctx = super()._context(call, task_key, dry_run)
        ctx["services"] = self.services
        return ctx


@pytest.fixture
async def base(tmp_path):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.models.providers = {
        "local": SimpleNamespace(model="local-1"),
        "mock": SimpleNamespace(model="mock-1"),
    }
    settings.models.default = "mock"
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    return SimpleNamespace(
        settings=settings, events=events, permissions=permissions,
        db_path=str(db.get_db_path()),
    )


def make_env(base, services, *, via_context=False):
    registry = ToolRegistry()
    assert register(registry, services={} if via_context else services) == list(TOOL_NAMES)
    if via_context:
        executor = InjectedExecutor(
            registry, base.permissions, base.events, base.settings, services=services
        )
    else:
        executor = ToolExecutorImpl(registry, base.permissions, base.events, base.settings)
    return SimpleNamespace(registry=registry, executor=executor, services=services)


async def run(env, name, **arguments):
    return await env.executor.execute(ToolCall(name=name, arguments=arguments))


def tool_entry(name, *, tier, status, verification, reason, source):
    return {
        "name": name, "tier": tier, "status": status,
        "verification": verification, "reason": reason, "source": source,
    }


# --------------------------------------------------------------------------- registration


def test_typed_specs_and_validation(base):
    registry = ToolRegistry()
    assert register(registry) == list(TOOL_NAMES)
    specs = {spec.name: spec for spec in registry.list_specs()}
    assert set(specs) == set(TOOL_NAMES)
    for spec in specs.values():
        assert isinstance(spec, ToolSpec)
        assert spec.category == "health"
        assert spec.risk is RiskLevel.LOW
        assert spec.permissions and all(spec.permissions)
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.resource_field == ""
        assert spec.description and "TODO" not in spec.description
    assert specs["app_health"].permissions == ["health:read"]
    assert specs["integration_matrix"].permissions == ["health:read", "integration:read"]
    assert specs["capability_status"].permissions == ["health:read", "capability:read"]
    assert specs["integration_matrix"].input_schema["properties"]["status"]["enum"] == [
        "configured", "degraded", "blocked"
    ]
    assert specs["capability_status"].input_schema["properties"]["tier"]["enum"] == [
        "all", "native", "mock-tested", "credential-blocked"
    ]
    assert registry.validate("integration_matrix", {"status": "live"}) != []
    assert registry.validate("capability_status", {"tier": "live"}) != []
    assert registry.validate("app_health", {"unexpected": 1}) != []
    assert registry.validate("capability_status", {"tier": "credential-blocked"}) == []
    assert register(registry) == []


async def test_permission_scope_denial_is_audited(base):
    env = make_env(base, {"app": SimpleNamespace(initialized=False)})
    result = await env.executor.execute(
        ToolCall(name="app_health"), scope=frozenset({"integration_matrix"})
    )
    assert result.success is False
    assert result.decision == "out_of_scope"
    rows = [dict(r) for r in await db.fetchall("SELECT tool,decision,success FROM tool_calls")]
    assert any(r["tool"] == "app_health" and r["decision"] == "out_of_scope" for r in rows)


# --------------------------------------------------------------------------- app_health


class FakeApp:
    def __init__(self, payload=None, *, initialized=True):
        self.payload = payload
        self.initialized = initialized
        self.calls = 0

    async def health(self):
        self.calls += 1
        return dict(self.payload)


class FakeRegistry:
    def __init__(self, names):
        self.names = names

    def list_specs(self):
        return [ToolSpec(name=name, description=f"live tool {name}") for name in self.names]

    def get(self, name):
        return SimpleNamespace(handler=lambda *args: None)


class FakeSkills:
    def __init__(self, names):
        self.names = names

    async def list(self):
        return list(self.names)


class FakeAgents:
    async def list(self):
        return ["agent-a", "agent-b"]


class FakePermissions:
    async def pending_approvals(self):
        return [{"id": "approval-1"}]


async def test_app_health_returns_exact_api_payload(base):
    api_payload = {
        "status": "ok", "version": "0.1.0", "db": "/tmp/other-leo.db",
        "providers": {"mock": "mock-model"}, "default_provider": "mock",
        "tools": 18, "skills": 12, "agents": 22, "pending_approvals": 0,
    }
    app = FakeApp(api_payload)
    env = make_env(base, {"app": app})
    result = await run(env, "app_health")
    assert result.success is True, result.error
    assert result.output == {**api_payload, "source": "app.health()"}
    assert app.calls == 1


async def test_app_health_composes_same_fields_from_live_engines(base):
    services = {
        "app": SimpleNamespace(initialized=True),
        "registry": FakeRegistry(["alpha", "beta"]),
        "skills": FakeSkills(["s1", "s2", "s3"]),
        "agent_registry": FakeAgents(),
        "permissions": FakePermissions(),
    }
    env = make_env(base, services, via_context=True)
    result = await run(env, "app_health")
    assert result.success is True, result.error
    assert result.output == {
        "status": "ok", "version": __version__, "db": base.db_path,
        "providers": {"local": "local-1", "mock": "mock-1"},
        "default_provider": "mock", "source": "composed",
        "tools": 2, "skills": 3, "agents": 2, "pending_approvals": 1,
    }


async def test_app_health_reports_not_initialized_without_count_fabrication(base):
    env = make_env(base, {
        "app": SimpleNamespace(initialized=False),
        "settings": base.settings,
        "registry": FakeRegistry(["alpha"]),
    })
    result = await run(env, "app_health")
    assert result.success is True, result.error
    assert result.output == {
        "status": "not-initialized", "version": __version__, "db": base.db_path,
        "providers": {"local": "local-1", "mock": "mock-1"},
        "default_provider": "mock", "source": "composed",
    }
    assert "tools" not in result.output


async def test_app_health_reports_not_initialized_from_ctx_without_services(base):
    env = make_env(base, {})
    result = await run(env, "app_health")
    assert result.success is True, result.error
    assert result.output == {
        "status": "not-initialized", "version": __version__, "db": base.db_path,
        "providers": {"local": "local-1", "mock": "mock-1"},
        "default_provider": "mock", "source": "composed",
    }


async def test_app_health_raises_when_initialized_engines_are_missing(base):
    env = make_env(base, {"app": SimpleNamespace(initialized=True), "settings": base.settings})
    result = await run(env, "app_health")
    assert result.success is False
    assert "initialized app_health requires" in (result.error or "")


# --------------------------------------------------------------------------- integration_matrix


class FakeGateway:
    def __init__(self, records):
        self.records = records
        self.calls = 0

    def integration_records(self):
        self.calls += 1
        return list(self.records)


class FakeAdmission:
    def __init__(self, policies):
        self.policies = policies

    def snapshot(self):
        return dict(self.policies)


def telegram_blocked():
    return IntegrationRecord(
        name="telegram", kind="messaging", status="blocked", capabilities=["send"],
        verification="mock-tested", config_ref="TELEGRAM_BOT_TOKEN",
        metadata={
            "reason": "missing credentials: TELEGRAM_BOT_TOKEN",
            "credential_env": ["TELEGRAM_BOT_TOKEN"], "credentials_present": [],
            "live_verified": False,
        },
    )


def discord_configured():
    return IntegrationRecord(
        name="discord", kind="messaging", status="configured", capabilities=["webhook", "send"],
        verification="mock-tested", config_ref="DISCORD_TOKEN",
        metadata={
            "reason": None, "credential_env": ["DISCORD_TOKEN"],
            "credentials_present": ["DISCORD_TOKEN"], "live_verified": False,
        },
    )


def browser_blocked():
    return IntegrationRecord(
        name="browser", kind="operator", status="blocked", capabilities=["navigate"],
        verification="mock-tested", config_ref="/tmp/profile",
        metadata={"reason": "browser backend unavailable", "live_verified": False},
    )


def integration_services(records, policies=None):
    if policies is None:
        policies = {r.name: {"mode": "allow"} for r in records if r.kind == "messaging"}
    return {"gateway": FakeGateway(records), "admission": FakeAdmission(policies)}


def matrix_entry(name, kind, status, record_status, capabilities, reason, policy, source="gateway"):
    return {
        "name": name, "kind": kind, "status": status, "record_status": record_status,
        "verification": "mock-tested", "live_verified": False,
        "capabilities": capabilities, "reason": reason, "policy": policy, "source": source,
    }


async def test_integration_matrix_reports_exact_honest_statuses(base):
    gateway = FakeGateway([telegram_blocked(), discord_configured(), browser_blocked()])
    env = make_env(base, {"gateway": gateway, "admission": FakeAdmission({
        "telegram": {"mode": "allow"}, "discord": {"mode": "allow"}
    })})
    result = await run(env, "integration_matrix")
    assert result.success is True, result.error
    assert result.output == {
        "status": "ok", "queried": True, "live_verified": False,
        "channel": None, "status_filter": None,
        "integrations": [
            matrix_entry("browser", "operator", "blocked", "blocked", ["navigate"],
                         "browser backend unavailable", None),
            matrix_entry("discord", "messaging", "configured", "configured", ["send", "webhook"],
                         None, "allow"),
            matrix_entry("telegram", "messaging", "blocked", "blocked", ["send"],
                         "missing credentials: TELEGRAM_BOT_TOKEN", "allow"),
        ],
        "summary": {"total": 3, "configured": 1, "degraded": 0, "blocked": 2},
    }
    assert gateway.calls == 1


async def test_integration_matrix_degraded_for_policy_deny_and_partial_credentials(base):
    partial = IntegrationRecord(
        name="email", kind="messaging", status="blocked", capabilities=["send"],
        verification="mock-tested", config_ref="EMAIL_TOKEN",
        metadata={
            "reason": "webhook secret missing",
            "credential_env": ["EMAIL_TOKEN", "EMAIL_SECRET"],
            "credentials_present": ["EMAIL_TOKEN"], "live_verified": False,
        },
    )
    env = make_env(base, {
        "gateway": FakeGateway([telegram_blocked(), discord_configured(), partial]),
        "admission": FakeAdmission({"telegram": {"mode": "allow"}, "discord": {"mode": "deny"}}),
    })
    result = await run(env, "integration_matrix", status="degraded")
    assert result.success is True, result.error
    assert result.output == {
        "status": "ok", "queried": True, "live_verified": False,
        "channel": None, "status_filter": "degraded",
        "integrations": [
            matrix_entry("discord", "messaging", "degraded", "configured", ["send", "webhook"],
                         "channel 'discord' is denied by admission policy", "deny"),
            matrix_entry("email", "messaging", "degraded", "blocked", ["send"],
                         "webhook secret missing", None),
        ],
        "summary": {"total": 2, "configured": 0, "degraded": 2, "blocked": 0},
    }
    no_policy_env = make_env(base, {
        "gateway": FakeGateway([discord_configured()]), "admission": FakeAdmission({}),
    })
    missing_policy = await run(no_policy_env, "integration_matrix", channel="discord")
    assert missing_policy.output["integrations"][0]["status"] == "degraded"
    assert missing_policy.output["integrations"][0]["reason"] == (
        "no admission policy for channel 'discord' (fail-closed)"
    )


async def test_integration_matrix_reads_durable_records_without_services(base):
    await db.executemany(
        "INSERT INTO integrations(name,kind,status,config_ref,capabilities,verification,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        [
            ("telegram", "messaging", "blocked", "TELEGRAM_BOT_TOKEN", '["send"]', "mock-tested", "t", "t"),
            ("browser", "operator", "blocked", "/tmp/profile", '["navigate"]', "mock-tested", "t", "t"),
        ],
    )
    env = make_env(base, {"admission": FakeAdmission({})})
    result = await run(env, "integration_matrix", channel="telegram")
    assert result.success is True, result.error
    assert result.output == {
        "status": "ok", "queried": True, "live_verified": False,
        "channel": "telegram", "status_filter": None,
        "integrations": [
            matrix_entry("telegram", "messaging", "blocked", "blocked", ["send"],
                         "missing credentials: TELEGRAM_BOT_TOKEN", None, source="database"),
        ],
        "summary": {"total": 1, "configured": 0, "degraded": 0, "blocked": 1},
    }
    browser = await run(env, "integration_matrix", channel="browser")
    assert browser.output["integrations"][0]["reason"] == "integration is blocked; no reason recorded"


async def test_integration_matrix_fails_without_real_records(base):
    env = make_env(base, {})
    result = await run(env, "integration_matrix")
    assert result.success is False
    assert "requires real integration records" in (result.error or "")


# --------------------------------------------------------------------------- capability_status


async def test_capability_status_distinguishes_all_tiers(base):
    tools = ToolRegistry()
    tools.register(ToolSpec(name="echo", description="echo"), lambda args, ctx: {"ok": True})
    tools.register(ToolSpec(name="ping", description="ping"), lambda args, ctx: {"ok": True})
    services = integration_services([telegram_blocked(), discord_configured(), browser_blocked()])
    services.update({
        "tools": tools,
        "skills": FakeSkills([
            SimpleNamespace(name="summarize", source="native", has_handler=True),
            SimpleNamespace(name="legacy_md", source="legacy-skill-md", has_handler=False),
        ]),
    })
    env = make_env(base, services)
    result = await run(env, "capability_status")
    assert result.success is True, result.error
    assert result.output == {
        "status": "ok", "queried": True, "live_verified": False, "tier": "all",
        "capabilities": [
            tool_entry("integration:browser", tier="environment-blocked", status="blocked",
                       verification="mock-tested", reason="browser backend unavailable",
                       source="integration-record"),
            tool_entry("integration:discord", tier="mock-tested", status="configured",
                       verification="mock-tested", reason=None, source="integration-record"),
            tool_entry("integration:telegram", tier="credential-blocked", status="blocked",
                       verification="mock-tested", reason="missing credentials: TELEGRAM_BOT_TOKEN",
                       source="integration-record"),
            tool_entry("skill:legacy_md", tier="unverified", status="unverified",
                       verification="legacy-skill-md", reason="no executable handler",
                       source="skill-registry"),
            tool_entry("skill:summarize", tier="native", status="available",
                       verification="native-handler", reason=None, source="skill-registry"),
            tool_entry("tool:echo", tier="native", status="available",
                       verification="native", reason=None, source="tool-registry"),
            tool_entry("tool:ping", tier="native", status="available",
                       verification="native", reason=None, source="tool-registry"),
        ],
        "summary": {
            "total": 7, "native": 3, "mock_tested": 1, "credential_blocked": 1,
            "environment_blocked": 1, "unverified": 1,
        },
    }
    blocked = await run(env, "capability_status", tier="credential-blocked")
    assert blocked.output["capabilities"] == [
        tool_entry("integration:telegram", tier="credential-blocked", status="blocked",
                   verification="mock-tested", reason="missing credentials: TELEGRAM_BOT_TOKEN",
                   source="integration-record"),
    ]
    assert blocked.output["summary"]["total"] == 1
    native = await run(env, "capability_status", tier="native")
    assert {item["name"] for item in native.output["capabilities"]} == {
        "skill:summarize", "tool:echo", "tool:ping"
    }


async def test_capability_status_reports_unexecutable_tool_as_unverified(base):
    class RegistryWithoutHandler:
        def list_specs(self):
            return [ToolSpec(name="inert", description="declared only")]

        def get(self, name):
            return None

    env = make_env(base, {"tools": RegistryWithoutHandler()})
    result = await run(env, "capability_status")
    assert result.success is True, result.error
    assert result.output["capabilities"] == [
        tool_entry("tool:inert", tier="unverified", status="unavailable",
                   verification="not-verified", reason="tool has no executable handler",
                   source="tool-registry"),
    ]
    assert result.output["summary"]["native"] == 0


async def test_capability_status_requires_real_state(base):
    env = make_env(base, {})
    result = await run(env, "capability_status")
    assert result.success is False
    assert "requires services['tools']" in (result.error or "")


async def test_all_tools_write_audit_rows(base):
    app = FakeApp({
        "status": "ok", "version": "0.1.0", "db": "x", "providers": {},
        "default_provider": "mock",
    })
    services = integration_services([discord_configured()])
    services["app"] = app
    env = make_env(base, services)
    for name, args in (
        ("app_health", {}),
        ("integration_matrix", {}),
        ("capability_status", {}),
    ):
        result = await run(env, name, **args)
        assert result.success is True, (name, result.error)
    rows = {
        row["tool"]: dict(row) for row in await db.fetchall(
            "SELECT tool,decision,success,risk FROM tool_calls WHERE tool IN "
            "('app_health','integration_matrix','capability_status')"
        )
    }
    assert set(rows) == set(TOOL_NAMES)
    assert all(row["decision"] == "allowed" and row["success"] == 1 and row["risk"] == "low"
               for row in rows.values())


def test_module_exports_expected_names():
    assert health_tools.__all__ == ["TOOL_NAMES", "register"]
