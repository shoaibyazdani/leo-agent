"""Slash-command registry, parser, authorization and execution tests.

Every execution test uses the real composition root with the isolated SQLite
database from tests/conftest.py, so a successful command result must be backed
by a durable row and a failing one must not invent success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leo import db
from leo.commands import (
    UNAVAILABLE,
    CommandSpec,
    create_registry,
    parse,
)
from leo.commands.registry import CommandStatus
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.core import LeoApp


@pytest.fixture
def registry():
    return create_registry()


@pytest.fixture
async def app(monkeypatch: pytest.MonkeyPatch) -> LeoApp:
    # Slash-command tests exercise the command layer, not agent-spec grant
    # validation. Fall back to unvalidated specialist loading only when a
    # concurrently changing tool registry makes an unrelated spec stale.
    from leo.agents.registry import AgentRegistry

    original_load = AgentRegistry.load_specialists

    async def tolerant_load(self, *, services=None, tool_registry=None):
        try:
            return await original_load(self, services=services, tool_registry=tool_registry)
        except RuntimeError as exc:
            if "declares grants absent" not in str(exc):
                raise
            return await original_load(self, services=services, tool_registry=None)

    monkeypatch.setattr(AgentRegistry, "load_specialists", tolerant_load)

    settings = Settings()
    settings.models = ModelConfig(
        default="mock", providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    )
    instance = LeoApp(settings)
    await instance.initialize()
    yield instance
    await instance.shutdown()


async def call(registry, app, text: str, **overrides) -> object:
    options = dict(
        app=app, principal="local", surface="cli", channel="cli",
        conversation_id="conv-commands",
    )
    options.update(overrides)
    return await registry.dispatch(text, **options)


# ---------------------------------------------------------------------------
# parser


def test_parse_accepts_leading_slash_quotes_and_reserved_flags():
    parsed = parse('  /memory add "remember this" --tags alpha,beta --approval apr_123')
    assert parsed is not None
    assert parsed.is_command
    assert parsed.name == "memory"
    assert parsed.argv == ["add", "remember this", "--tags", "alpha,beta"]
    assert parsed.approval_id == "apr_123"
    assert parsed.raw_args.startswith("memory add")


def test_parse_rejects_prose_empty_command_bad_name_and_unbalanced_quotes():
    assert parse("please check /status") is None
    assert parse("no slash here") is None
    for text in ("/", "/   ", '/memory add "unterminated', "/bad!name value", "/memory add x --approval"):
        parsed = parse(text)
        assert parsed is not None and parsed.error, text


def test_parser_never_expands_shell_metacharacters():
    parsed = parse('/memory add "a; rm -rf / && echo $HOME"')
    assert parsed.argv == ["add", "a; rm -rf / && echo $HOME"]


# ---------------------------------------------------------------------------
# registry, aliases and help


def test_resolve_exact_alias_prefix_case_and_ambiguity(registry):
    assert registry.resolve("COMPRESS")[0].name == "compact"
    assert registry.resolve("/compress")[0].name == "compact"
    assert registry.resolve("id")[0].name == "whoami"
    assert registry.resolve("models")[0].name == "model"
    assert registry.resolve("cr")[0].name == "schedule"  # unique alias prefix
    assert registry.resolve("sk")[0] is None and registry.resolve("sk")[1]
    assert registry.resolve("definitely_not_a_command")[0] is None


def test_duplicate_names_and_aliases_are_rejected(registry):
    with pytest.raises(ValueError, match="duplicate command name"):
        registry.register(CommandSpec(name="help", help="duplicate"))
    with pytest.raises(ValueError, match="duplicate command alias"):
        registry.register(CommandSpec(name="other", help="duplicate alias", aliases=("h",)))
    with pytest.raises(ValueError, match="unavailable but has no reason"):
        CommandSpec(name="broken", help="broken", available=False)


async def test_help_lists_implemented_and_explicitly_unavailable_commands(registry):
    result = await call(registry, object(), "/help")
    assert result.status is CommandStatus.OK
    text = result.message
    assert "LEO slash commands" in text
    for name in ("/help", "/new", "/memory", "/tasks", "/agents", "/approve", "/channels"):
        assert name in text
    assert "[UNAVAILABLE:" in text
    assert "/compact" in text and "/compress" in text  # alias discoverability
    # /compact is now implemented; it must not be advertised as unavailable.
    assert "UNAVAILABLE: LEO only optimizes context per turn" not in text
    assert result.data["total"] == len(registry.all())
    for entry in result.data["commands"]:
        assert entry["help"] and entry["category"]
        assert entry["risk"] in {"low", "medium", "high"}
        if not entry["available"]:
            assert entry["unavailable_reason"]


async def test_help_filter_and_paging(registry):
    skills = await call(registry, object(), "/help skills")
    assert skills.status is CommandStatus.OK
    assert "/skill" in skills.message
    assert "/skills" in skills.message
    assert skills.data["total"] < len(registry.all())
    page_two = await call(registry, object(), "/commands 2")
    assert page_two.status is CommandStatus.OK and page_two.data["page"] == 2
    assert "page 2" in page_two.message


async def test_unknown_command_and_parse_errors_never_reach_a_model(registry, app):
    unknown = await call(registry, app, "/not_a_command")
    assert unknown.status is CommandStatus.ERROR and unknown.code == "unknown_command"
    assert "Unknown command" in unknown.message
    ambiguous = await call(registry, app, "/sk")
    assert ambiguous.code == "ambiguous_command"
    assert "Did you mean" in ambiguous.message
    broken = await call(registry, app, '/memory add "unterminated')
    assert broken.code == "parse_error"
    prose = await call(registry, app, "check /status please")
    assert prose.code == "not_command"
    # No command event was emitted for any of the rejected invocations.
    rows = await db.fetchall("SELECT payload FROM events WHERE topic='slash.command'")
    assert rows == []


async def test_missing_arguments_are_rejected_before_any_authorization(registry, app):
    result = await call(registry, app, "/skill")
    assert result.code == "usage"
    assert "usage" in result.message.lower()
    rows = await db.fetchall("SELECT action FROM audit_log WHERE action='slash.skill'")
    assert rows == []


# ---------------------------------------------------------------------------
# authorization and safety policy


async def test_operator_commands_denied_on_chat_and_audited(registry, app):
    for text in ("/approvals", "/approve apr_test", "/deny apr_test"):
        result = await call(registry, app, text, surface="chat", principal="web-user",
                            channel="telegram")
        assert result.status is CommandStatus.DENIED, text
        assert result.success is False
    rows = await db.fetchall(
        "SELECT action, decision FROM audit_log WHERE decision='denied' AND resource LIKE 'slash:%'"
    )
    actions = {row["action"] for row in rows}
    assert {"slash.approvals", "slash.approve", "slash.deny"} <= actions


async def test_admin_only_commands_denied_for_non_admin(registry, app):
    result = await call(registry, app, "/audit", surface="chat", principal="web-user")
    assert result.status is CommandStatus.DENIED
    assert "administrator" in result.message
    assert (await call(registry, app, "/audit", surface="cli", principal="local")).status is CommandStatus.OK


async def test_medium_risk_policy_denial_is_truthful(registry, app):
    app.settings.security.allow_medium_risk = False
    try:
        result = await call(registry, app, "/new")
        assert result.status is CommandStatus.DENIED
        assert "medium-risk actions disabled" in result.message
        assert (await db.fetchone("SELECT id FROM conversations")) is None
    finally:
        app.settings.security.allow_medium_risk = True


async def test_unavailable_commands_report_a_reason_instead_of_executing(registry, app):
    for text in ("/resume", "/voice", "/retry", "/plugins", "/steer hi"):
        result = await call(registry, app, text)
        assert result.status is CommandStatus.UNSUPPORTED, text
        assert result.code == "unsupported"
        assert result.unavailable_reason if hasattr(result, "unavailable_reason") else result.message
        assert result.message
    rows = await db.fetchall("SELECT action FROM audit_log WHERE action LIKE 'slash.voice%'")
    assert rows == []


# ---------------------------------------------------------------------------
# read-only commands against real services


async def test_status_version_whoami_and_help_execute(registry, app):
    status = await call(registry, app, "/status")
    assert status.status is CommandStatus.OK
    assert status.data["health"]["status"] == "ok"
    assert status.data["pending_approvals"] == 0
    version = await call(registry, app, "/version")
    assert version.data["version"] == "0.1.0"
    identity = await call(registry, app, "/whoami", surface="chat", principal="web-user",
                          channel="telegram")
    assert identity.data["actor"] == "chat:telegram"
    assert identity.data["operator"] is False
    assert "cannot resolve approvals" in identity.message


async def test_model_usage_context_and_config_are_read_only(
    registry, app, monkeypatch, tmp_path
):
    # /model <name> now persists models.default; point it at a scratch file so
    # the test can never rewrite the operator's real leo.yaml.
    scratch = tmp_path / "leo.yaml"
    scratch.write_text("models:\n  default: mock\n  providers: {}\n", encoding="utf-8")
    monkeypatch.setenv("LEO_CONFIG", str(scratch))

    model = await call(registry, app, "/model")
    assert model.status is CommandStatus.OK and model.data["status"] == "simulated"
    listed = await call(registry, app, "/model list")
    assert "mock" in listed.message
    assert "Current:" in listed.message
    switched = await call(registry, app, "/model mock")
    assert switched.status is CommandStatus.OK
    assert switched.data["default_provider"] == "mock"
    assert switched.data["persisted"] is True
    assert "saved to leo.yaml" in switched.message
    assert scratch.read_text(encoding="utf-8").count("default: mock") == 1
    unknown_provider = await call(registry, app, "/model other")
    assert unknown_provider.status is CommandStatus.ERROR
    reset = await call(registry, app, "/model reset")
    assert reset.status is CommandStatus.UNSUPPORTED
    assert "no runtime model switch" not in reset.message
    usage = await call(registry, app, "/usage")
    assert usage.status is CommandStatus.OK and usage.data["calls"] == 0
    context = await call(registry, app, "/context")
    assert context.status is CommandStatus.OK
    assert context.data["compaction"].startswith("durable:")
    config = await call(registry, app, "/config get default_provider")
    assert config.data["value"] == "mock"
    unknown = await call(registry, app, "/config get api_key")
    assert unknown.status is CommandStatus.ERROR
    refused = await call(registry, app, "/config set default_provider=other")
    assert refused.status is CommandStatus.UNSUPPORTED
    assert app.settings.models.default == "mock"


async def test_events_metrics_debug_insights_and_channels(registry, app):
    await call(registry, app, "/status")
    events = await call(registry, app, "/events")
    assert events.status is CommandStatus.OK
    assert any(event["topic"] == "slash.command" for event in events.data["events"])
    stats = await call(registry, app, "/events stats")
    assert stats.status is CommandStatus.OK
    metrics = await call(registry, app, "/metrics")
    assert metrics.status is CommandStatus.OK and "model_usage" in metrics.data
    debug = await call(registry, app, "/debug upload")
    assert debug.status is CommandStatus.UNSUPPORTED
    local = await call(registry, app, "/debug local")
    assert local.status is CommandStatus.OK
    insights = await call(registry, app, "/insights")
    assert insights.status is CommandStatus.OK
    channels = await call(registry, app, "/channels")
    assert channels.status is CommandStatus.OK
    assert "credential values are never returned" in channels.data["note"]
    assert channels.data["outbox"]["pending"] == 0


async def test_sensitive_events_hidden_from_chat(registry, app):
    await app.events.emit("tool.finished", tool="secret_tool", result="private output")
    chat = await call(registry, app, "/events tool.finished", surface="chat",
                      principal="web-user")
    assert chat.status is CommandStatus.OK
    assert all(not event["topic"].startswith("tool.") for event in chat.data["events"])
    operator = await call(registry, app, "/events tool.finished", surface="cli",
                          principal="local")
    assert any(event["topic"].startswith("tool.") for event in operator.data["events"])


async def test_tools_list_and_unsupported_toggle(registry, app):
    listed = await call(registry, app, "/tools list")
    assert listed.status is CommandStatus.OK and listed.data["tools"]
    assert all("risk" in tool for tool in listed.data["tools"])
    filtered = await call(registry, app, "/tools search git_diff")
    assert any(tool["name"] == "git_diff" for tool in filtered.data["tools"])
    disabled = await call(registry, app, "/tools disable git_diff")
    assert disabled.status is CommandStatus.UNSUPPORTED
    assert "not implemented" in disabled.message


async def test_diff_uses_the_audited_git_tool(registry, app):
    unsupported = await call(registry, app, "/diff session")
    assert unsupported.status is CommandStatus.UNSUPPORTED
    result = await call(registry, app, "/diff --stat")
    assert result.status is CommandStatus.OK
    assert result.data["tool"] == "git_diff"
    calls = await db.fetchall("SELECT tool FROM tool_calls WHERE tool='git_diff'")
    assert calls


# ---------------------------------------------------------------------------
# skills


async def test_skills_list_search_inspect_and_run(registry, app):
    listed = await call(registry, app, "/skills list")
    assert listed.status is CommandStatus.OK and listed.data["skills"]
    searched = await call(registry, app, "/skills search summarize")
    assert any(item["name"] == "summarize_text" for item in searched.data["skills"])
    inspected = await call(registry, app, "/skills inspect summarize_text")
    assert inspected.status is CommandStatus.OK
    assert inspected.data["skill"]["name"] == "summarize_text"
    run = await call(registry, app, '/skills run summarize_text text="Alpha beta. Gamma delta." max_sentences=1')
    assert run.status is CommandStatus.OK
    assert run.data["success"] is True
    assert "summary" in run.data["output"]
    unknown = await call(registry, app, "/skills run no_such_skill text=x")
    assert unknown.status is CommandStatus.ERROR and unknown.code == "skill_failed"
    assert "unknown skill" in unknown.message
    install = await call(registry, app, "/skills install anything")
    assert install.status is CommandStatus.UNSUPPORTED


async def test_skill_alias_and_dynamic_skill_family(registry, app):
    direct = await call(registry, app, "/skill summarize_text one two three")
    assert direct.status is CommandStatus.OK and direct.command == "skill"
    dynamic = await call(registry, app, '/summarize_text text="one. two."')
    assert dynamic.status is CommandStatus.OK and dynamic.command == "skill"
    assert dynamic.data["success"] is True
    unknown = await call(registry, app, "/not_an_installed_skill text=x")
    assert unknown.code == "unknown_command"


# ---------------------------------------------------------------------------
# memory and approvals


async def test_memory_add_search_list_get_and_forget_with_approval(registry, app):
    added = await call(registry, app, '/memory add "remember this fact" --tags alpha,beta')
    assert added.status is CommandStatus.OK
    memory_id = added.data["memory"]["id"]
    stored = await db.fetchone("SELECT content, tags FROM memory_items WHERE id=?", (memory_id,))
    assert stored is not None and "remember this fact" in stored["content"]
    assert "alpha" in stored["tags"]
    found = await call(registry, app, "/memory search remember")
    assert any(item["id"] == memory_id for item in found.data["results"])
    listed = await call(registry, app, "/memory list")
    assert any(item["id"] == memory_id for item in listed.data["memories"])
    fetched = await call(registry, app, f"/memory get {memory_id}")
    assert fetched.data["memory"]["id"] == memory_id

    pending = await call(registry, app, f"/memory forget {memory_id}")
    assert pending.status is CommandStatus.PENDING and pending.approval_id
    assert (await db.fetchone("SELECT content FROM memory_items WHERE id=?", (memory_id,))) is not None
    row = await db.fetchone("SELECT status, action, risk FROM approvals WHERE id=?", (pending.approval_id,))
    assert row["status"] == "pending" and row["risk"] == "high"
    assert row["action"] == "slash.memory.forget"

    # A supplied approval id is bound to the exact argv; changing it must fail.
    wrong = await call(registry, app, f'/memory forget {memory_id} other --approval {pending.approval_id}')
    assert wrong.status is CommandStatus.DENIED
    await app.permissions.resolve_approval(pending.approval_id, True, resolver="operator:test")
    done = await call(registry, app, f"/memory forget {memory_id} --approval {pending.approval_id}")
    assert done.status is CommandStatus.OK and done.data["deleted"] is True
    assert (await db.fetchone("SELECT id FROM memory_items WHERE id=?", (memory_id,))) is None
    audit = await db.fetchall("SELECT action FROM audit_log WHERE action='slash.memory.forget'")
    assert len(audit) >= 1


async def test_approve_and_deny_resolve_only_through_operator(registry, app):
    added = await call(registry, app, "/remember operator approval test")
    memory_id = added.data["memory"]["id"]
    pending = await call(registry, app, f"/memory forget {memory_id}")
    approval_id = pending.approval_id

    chat = await call(registry, app, f"/approve {approval_id}", surface="chat",
                      principal="web-user", channel="telegram")
    assert chat.status is CommandStatus.DENIED
    assert "not available on the chat surface" in chat.message
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,)))["status"] == "pending"

    resolved = await call(registry, app, f"/approve {approval_id}", surface="cli", principal="local")
    assert resolved.status is CommandStatus.OK
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?", (approval_id,)))["status"] == "approved"
    approved_retry = await call(registry, app, f"/memory forget {memory_id} --approval {approval_id}")
    assert approved_retry.status is CommandStatus.OK

    second = await call(registry, app, "/remember second approval test")
    second_pending = await call(registry, app, f"/memory forget {second.data['memory']['id']}")
    denied = await call(registry, app, f"/deny {second_pending.approval_id}")
    assert denied.status is CommandStatus.OK
    assert (await db.fetchone("SELECT status FROM approvals WHERE id=?",
                              (second_pending.approval_id,)))["status"] == "denied"
    approvals_list = await call(registry, app, "/approvals")
    assert approvals_list.status is CommandStatus.OK


# ---------------------------------------------------------------------------
# tasks, agents, sessions and scheduling


async def test_tasks_lifecycle_is_durable(registry, app):
    created = await call(registry, app, "/tasks create write the release notes")
    assert created.status is CommandStatus.OK
    task_id = created.data["task"]["id"]
    row = await db.fetchone("SELECT objective, status FROM tasks WHERE id=?", (task_id,))
    assert row["objective"] == "write the release notes"
    assert row["status"] == "created"
    status = await call(registry, app, f"/tasks status {task_id}")
    assert status.data["task"]["id"] == task_id
    listed = await call(registry, app, "/tasks list")
    assert any(task["id"] == task_id for task in listed.data["tasks"])
    cancelled = await call(registry, app, f"/tasks cancel {task_id}")
    assert cancelled.status is CommandStatus.OK
    assert (await call(registry, app, f"/tasks status {task_id}")).data["task"]["status"] == "cancelled"
    missing = await call(registry, app, "/tasks status task_missing")
    assert missing.status is CommandStatus.ERROR
    bad_status = await call(registry, app, "/tasks list --status not-a-status")
    assert bad_status.status is CommandStatus.ERROR


async def test_agents_list_status_and_unknown_delegate(registry, app):
    listed = await call(registry, app, "/agents list")
    assert listed.status is CommandStatus.OK and listed.data["agents"]
    first = listed.data["agents"][0]["id"]
    status = await call(registry, app, f"/agents status {first}")
    assert status.status is CommandStatus.OK
    unknown = await call(registry, app, "/agents status missing-agent")
    assert unknown.status is CommandStatus.ERROR
    delegate = await call(registry, app, "/agents delegate do something --agent missing-agent")
    assert delegate.status is CommandStatus.ERROR and delegate.success is False


async def test_new_title_history_sessions_and_reset(registry, app):
    created = await call(registry, app, "/new Sprint planning")
    assert created.status is CommandStatus.OK
    conversation_id = created.data["conversation_id"]
    row = await db.fetchone("SELECT subject FROM conversations WHERE id=?", (conversation_id,))
    assert row["subject"] == "Sprint planning"
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
        (conversation_id, "user", "draft note", "2026-01-01T00:00:00+00:00"),
    )
    titled = await call(registry, app, "/title Sprint review", conversation_id=conversation_id)
    assert titled.status is CommandStatus.OK
    assert (await db.fetchone("SELECT subject FROM conversations WHERE id=?", (conversation_id,)))["subject"] == "Sprint review"
    history = await call(registry, app, "/history", conversation_id=conversation_id)
    assert history.status is CommandStatus.OK
    assert any(message["content"] == "draft note" for message in history.data["messages"])
    sessions = await call(registry, app, "/sessions")
    assert any(item["id"] == conversation_id for item in sessions.data["sessions"])

    pending = await call(registry, app, "/reset", conversation_id=conversation_id)
    assert pending.status is CommandStatus.PENDING and pending.approval_id
    assert (await db.fetchall("SELECT id FROM messages WHERE conversation_id=?", (conversation_id,)))
    await app.permissions.resolve_approval(pending.approval_id, True, resolver="operator:test")
    done = await call(registry, app, f"/reset --approval {pending.approval_id}",
                      conversation_id=conversation_id)
    assert done.status is CommandStatus.OK and done.data["removed"] >= 1
    assert (await db.fetchall("SELECT id FROM messages WHERE conversation_id=?", (conversation_id,))) == []


def test_export_is_registered_as_a_real_command(registry):
    spec = registry.resolve("export")[0]
    assert spec is not None and spec.available and callable(spec.handler)
    assert spec.aliases == ("save", "export-session", "save-format")
    assert "export" not in {name for name, *_ in UNAVAILABLE}
    assert "/export" in registry.help("export")["text"]


async def _seed_export_conversation(conversation_id: str, content: str) -> None:
    await db.execute(
        "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at) "
        "VALUES(?,?,?,?,?)",
        (conversation_id, "cli", "export test", "2026-01-01T00:00:00+00:00",
         "2026-01-01T00:00:00+00:00"),
    )
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
        (conversation_id, "user", content, "2026-01-01T00:00:00+00:00"),
    )


async def test_export_writes_redacted_markdown_with_truthful_path(registry, app, tmp_path):
    app.settings.paths.exports = tmp_path / "exports"
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    await _seed_export_conversation("conv-export", f"remember token={secret} please")

    result = await call(registry, app, "/export", conversation_id="conv-export")
    assert result.status is CommandStatus.OK, result.message
    assert result.command == "export"
    path = Path(result.data["path"])
    assert path.is_file()
    assert path.parent == (tmp_path / "exports").resolve()
    assert path.suffix == ".md"
    body = path.read_text(encoding="utf-8")
    assert "# Conversation export" in body
    assert "remember" in body
    assert secret not in body
    assert "[REDACTED]" in body
    assert result.data["bytes"] == path.stat().st_size
    assert result.data["count"] == 1
    assert result.data["redacted"] is True
    assert str(path) in result.message
    assert (await db.fetchall("SELECT id FROM audit_log WHERE action='slash.export'"))


async def test_export_parses_formats_and_out_names_without_path_escape(registry, app, tmp_path):
    app.settings.paths.exports = tmp_path / "exports"
    await _seed_export_conversation("conv-export", "plain note")

    for fmt, body_marker in (("json", '"messages"'), ("txt", "conversation_id:")):
        result = await call(registry, app, f"/export {fmt} --out report",
                            conversation_id="conv-export")
        assert result.status is CommandStatus.OK, result.message
        path = Path(result.data["path"])
        assert path.name == f"report.{fmt}"
        assert body_marker in path.read_text(encoding="utf-8")
        assert result.data["format"] == fmt
    aliased = await call(registry, app, "/save md --out=aliased",
                         conversation_id="conv-export")
    assert aliased.status is CommandStatus.OK and Path(aliased.data["path"]).name == "aliased.md"

    invalid = await call(registry, app, "/export html", conversation_id="conv-export")
    assert invalid.status is CommandStatus.ERROR and invalid.code == "invalid_request"
    assert "expected one of" in invalid.message
    escaped = await call(registry, app, "/export md --out ../../outside",
                         conversation_id="conv-export")
    assert escaped.status is CommandStatus.ERROR and escaped.code == "invalid_request"
    assert "path separators" in escaped.message
    assert not (tmp_path / "outside.md").exists() and not (tmp_path.parent / "outside.md").exists()
    unknown_option = await call(registry, app, "/export md --bogus",
                                conversation_id="conv-export")
    assert unknown_option.status is CommandStatus.ERROR and "unknown option" in unknown_option.message


async def test_export_bounds_each_message_and_reports_failure_truthfully(registry, app, tmp_path):
    app.settings.paths.exports = tmp_path / "exports"
    await _seed_export_conversation("conv-export", "x" * 50_000)

    result = await call(registry, app, "/export", conversation_id="conv-export")
    assert result.status is CommandStatus.OK
    path = Path(result.data["path"])
    body = path.read_text(encoding="utf-8")
    assert len(body) < 10_000  # no 50 KB transcript row is echoed wholesale
    assert "[truncated]" in body
    assert result.data["bytes"] == path.stat().st_size

    missing_context = await call(registry, app, "/export", conversation_id=None)
    assert missing_context.status is CommandStatus.ERROR
    assert "conversation_id" in missing_context.message
    unknown = await call(registry, app, "/export", conversation_id="conv-does-not-exist")
    assert unknown.status is CommandStatus.ERROR
    assert "unknown conversation" in unknown.message
    assert list((tmp_path / "exports").glob("*.md")) == [path]


async def test_export_through_app_slash_path_is_truthful(app, tmp_path):
    app.settings.paths.exports = tmp_path / "exports"
    answer = await app.ask("/export md --out asked", channel="cli",
                           conversation_id="conv-ask")
    assert answer["command"] == "export"
    assert answer["status"] == "completed"
    assert "Exported" in answer["content"] and "asked.md" in answer["content"]
    written = tmp_path / "exports" / "asked.md"
    assert written.is_file()
    assert written.stat().st_size > 0


async def test_schedule_cron_loop_toggle_trigger_and_remove(registry, app):
    added = await call(registry, app, "/schedule add heartbeat 60 check the queue")
    assert added.status is CommandStatus.OK
    assert (await db.fetchone("SELECT name FROM schedules WHERE name='heartbeat'")) is not None
    cron = await call(registry, app, "/schedule cron nightly '0 2 * * *' archive logs")
    assert cron.status is CommandStatus.OK
    listed = await call(registry, app, "/schedule list")
    assert {item["name"] for item in listed.data["schedules"]} >= {"heartbeat", "nightly"}
    disabled = await call(registry, app, "/schedule disable heartbeat")
    assert disabled.status is CommandStatus.OK and disabled.data["schedule"]["enabled"] is False
    enabled = await call(registry, app, "/schedule enable heartbeat")
    assert enabled.data["schedule"]["enabled"] is True
    triggered = await call(registry, app, "/schedule run heartbeat")
    assert triggered.status is CommandStatus.OK
    loop = await call(registry, app, "/loop 120 check battery --name battery_check")
    assert loop.status is CommandStatus.OK
    assert "not inside this chat" in loop.message
    stored = await db.fetchone("SELECT payload FROM schedules WHERE name='battery_check'")
    assert "check battery" in stored["payload"]
    removed = await call(registry, app, "/schedule remove nightly")
    assert removed.status is CommandStatus.OK
    assert (await db.fetchone("SELECT name FROM schedules WHERE name='nightly'")) is None
    audit = await db.fetchall("SELECT action FROM audit_log WHERE action LIKE 'slash.schedule%'")
    assert audit


async def test_stop_cancels_a_live_run_truthfully(registry, app):
    app._start_run("run-test", "conv-commands")
    result = await call(registry, app, "/stop run-test")
    assert result.status is CommandStatus.OK and result.data["status"] == "cancelled"
    assert app._runs["run-test"]["status"] == "cancelled"
    missing = await call(registry, app, "/stop unknown-run")
    assert missing.status is CommandStatus.ERROR


# ---------------------------------------------------------------------------
# catalog completeness


def test_discovery_vocabulary_is_covered(registry):
    """Names from the compatibility inventory must be discoverable."""
    expected = {
        # implemented families
        "help", "commands", "status", "whoami", "version", "config", "model", "usage",
        "context", "new", "reset", "sessions", "history", "export", "title", "stop", "diff",
        "skills", "skill", "memory", "tasks", "agents", "tools", "approvals", "approve",
        "deny", "events", "metrics", "debug", "audit", "insights", "schedule", "loop",
        "channels", "remember",
        # explicitly unavailable, but present with a reason
        "compact", "resume", "import", "retry", "undo", "rewind", "branch",
        "steer", "queue", "bg", "btw", "goal", "subgoal", "personality", "reasoning",
        "fast", "verbose", "yolo", "learn", "plan", "review", "init", "reload",
        "reload-mcp", "reload-skills", "update", "restart", "pause", "start", "discard",
        "answer", "topic", "sethome", "platform", "handoff", "login", "subscription",
        "topup", "voice", "wake", "pet", "hatch", "skin", "theme", "plugins", "bundles",
        "curator", "moa", "blueprint", "kanban", "journey", "worktree", "rollback",
        "snapshot", "browser", "image", "paste", "copy", "prompt", "palette", "quit",
        "autoprune", "codex-runtime", "egress", "profile", "statusbar", "timestamps",
        "focus", "footer", "battery", "indicator", "redraw", "density", "details",
        "logs", "mouse", "fortune", "terminal-setup", "setup", "replay", "replay-diff",
        "connectors", "heapdump", "mem", "widgets-reload", "theme-info",
    }
    names = {spec.name for spec in registry.all()}
    aliases = {alias for spec in registry.all() for alias in spec.aliases}
    missing = expected - names - aliases
    assert not missing, f"missing slash commands: {sorted(missing)}"
    assert all(spec.handler is None or callable(spec.handler) for spec in registry.all())


def test_model_listing_is_a_scannable_grid() -> None:
    """/model must be a list of models, not a block per provider.

    Live complaint: 31 providers rendered ~90 lines, each with the provider name,
    a fake ``--provider <name>`` token and the model name again on the next line.
    A chat panel has room for a grid, not an audit.
    """
    from leo.commands.builtins import _model_listing_message
    from leo.config import ModelProviderConfig

    providers = {
        f"model-{index:02d}": ModelProviderConfig(
            kind="openai-compatible", base_url="https://x.test", model=f"model-{index:02d}"
        )
        for index in range(30)
    }
    providers["off-01"] = ModelProviderConfig(
        kind="openai-compatible", base_url="https://x.test", model="off-01", enabled=False
    )

    text = _model_listing_message(providers, "model-07")

    assert text.splitlines()[0] == "Current: * model-07  (30 models)"
    assert len(text.splitlines()) <= 16, "30 models must not become 90 lines"
    assert "--provider" not in text
    assert "hidden: off-01" in text
    assert "* model-07" in text
    for name in ("model-00", "model-12", "model-29"):
        assert text.count(name) == 1
    assert text.count("model-07") == 2  # header + grid cell
