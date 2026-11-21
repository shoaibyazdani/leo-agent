"""Regression tests for CLI argument validation, error handling and JSON output.

All commands are exercised in-process through Typer's CliRunner against the
isolated test database installed by tests/conftest.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

from leo import db
from leo.config import ModelProviderConfig, Settings
from leo.contracts import RiskLevel
from leo.interfaces import cli as cli_mod

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fresh_cli_state() -> Any:
    # The CLI caches Settings/LeoApp in a module-level dict. Clear it so each
    # test starts with the isolated DB and its own settings.
    cli_mod._state.clear()
    yield
    cli_mod._state.clear()


def _invoke(*args: str) -> Any:
    return runner.invoke(cli_mod.cli, list(args))


def _json(result: Any) -> Any:
    assert result.exit_code != 0 or result.stdout.strip(), result.output
    return json.loads(result.stdout)


def _no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEO_API_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# serve: non-loopback bind protection
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []

    def fake_run(app: str, **kwargs: Any) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


def test_serve_blocks_non_loopback_without_token(monkeypatch, fake_uvicorn) -> None:
    _no_token(monkeypatch)
    result = _invoke("serve", "--host", "0.0.0.0")
    assert result.exit_code != 0
    assert "non-loopback bind" in result.stderr
    assert fake_uvicorn == []


def test_serve_blocks_configured_non_loopback_without_token(monkeypatch, fake_uvicorn) -> None:
    _no_token(monkeypatch)
    settings = Settings()
    settings.server.host = "0.0.0.0"
    settings.server.port = 8899
    monkeypatch.setitem(cli_mod._state, "settings", settings)
    result = _invoke("serve")
    assert result.exit_code != 0
    assert "non-loopback bind" in result.stderr
    assert fake_uvicorn == []


def test_serve_rejects_blank_token(monkeypatch, fake_uvicorn) -> None:
    monkeypatch.setenv("LEO_API_TOKEN", "   ")
    result = _invoke("serve", "--host", "0.0.0.0")
    assert result.exit_code != 0
    assert fake_uvicorn == []


def test_serve_allows_non_loopback_with_token(monkeypatch, fake_uvicorn) -> None:
    monkeypatch.setenv("LEO_API_TOKEN", "unit-test-token")
    result = _invoke("serve", "--host", "0.0.0.0", "--port", "9123")
    assert result.exit_code == 0, result.output
    assert len(fake_uvicorn) == 1
    app, kwargs = fake_uvicorn[0]
    assert app == "leo.interfaces.api:api"
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9123


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1", "[::1]", " 127.0.0.1 "],
)
def test_serve_allows_loopback_without_token(monkeypatch, fake_uvicorn, host: str) -> None:
    _no_token(monkeypatch)
    result = _invoke("serve", "--host", host)
    assert result.exit_code == 0, result.output
    assert len(fake_uvicorn) == 1
    assert fake_uvicorn[0][1]["host"] == host.strip()


def test_serve_rejects_out_of_range_port(monkeypatch, fake_uvicorn) -> None:
    _no_token(monkeypatch)
    result = _invoke("serve", "--host", "127.0.0.1", "--port", "70000")
    assert result.exit_code != 0
    body = _json(result)
    assert body["ok"] is False
    assert "invalid port" in body["error"]
    assert fake_uvicorn == []


# ---------------------------------------------------------------------------
# Missing ids / invalid commands produce JSON errors, never tracebacks
# ---------------------------------------------------------------------------


def test_task_show_missing_is_json_error() -> None:
    result = _invoke("task", "show", "task-does-not-exist")
    assert result.exit_code == 1
    body = _json(result)
    assert body["ok"] is False
    assert "task not found" in body["error"]
    assert "task-does-not-exist" in body["error"]


def test_task_cancel_missing_is_json_error() -> None:
    result = _invoke("task", "cancel", "task-does-not-exist")
    assert result.exit_code == 1
    body = _json(result)
    assert body["ok"] is False
    assert "not found" in body["error"]


def test_approval_approve_missing_is_json_error() -> None:
    result = _invoke("approval", "approve", "apr-does-not-exist")
    assert result.exit_code == 1
    body = _json(result)
    assert body["ok"] is False
    assert "not found" in body["error"]


def test_approval_deny_missing_is_json_error() -> None:
    result = _invoke("approval", "deny", "apr-does-not-exist")
    assert result.exit_code == 1
    body = _json(result)
    assert body["ok"] is False
    assert "not found" in body["error"]


# ---------------------------------------------------------------------------
# approval cleanup: dry-run default, scoped apply, never deletes
# ---------------------------------------------------------------------------


def _create_pending_approval(actor: str, action: str) -> str:
    app = cli_mod.ready()
    decision = asyncio.run(
        app.permissions.authorize(actor, action, None, RiskLevel.HIGH)
    )
    assert decision.approval_id, decision
    return decision.approval_id


def _approval_rows() -> list[dict[str, Any]]:
    rows = asyncio.run(db.fetchall("SELECT * FROM approvals ORDER BY created_at"))
    return [dict(r) for r in rows]


def test_approval_cleanup_defaults_to_dry_run_and_keeps_rows_pending() -> None:
    approval_id = _create_pending_approval("learning:engine", "learning.note")
    result = _invoke("approval", "cleanup", "--older-than", "0")
    assert result.exit_code == 0, result.output
    body = _json(result)
    assert body["apply"] is False
    assert body["candidates"] == 1
    assert body["sample"][0]["id"] == approval_id
    rows = _approval_rows()
    assert len(rows) == 1 and rows[0]["status"] == "pending"


def test_approval_cleanup_apply_expires_rows_without_deleting_and_audits() -> None:
    approval_id = _create_pending_approval("learning:engine", "learning.note")
    result = _invoke("approval", "cleanup", "--older-than", "0", "--apply")
    assert result.exit_code == 0, result.output
    body = _json(result)
    assert body["apply"] is True
    assert body["expired"] == 1
    rows = _approval_rows()
    assert len(rows) == 1, "cleanup must never delete approval rows"
    assert rows[0]["status"] == "expired"
    assert rows[0]["resolved_at"]
    audits = asyncio.run(db.fetchall(
        "SELECT * FROM audit_log WHERE reason LIKE 'pending approval expired by cleanup%'"))
    assert len(audits) == 1
    assert audits[0]["actor"] == "operator"
    # An expired approval can no longer be decided either way.
    denied = _invoke("approval", "deny", approval_id)
    assert denied.exit_code == 1
    assert "approval already expired" in _json(denied)["error"]


def test_approval_cleanup_scopes_to_source_and_is_idempotent() -> None:
    _create_pending_approval("learning:engine", "learning.note")
    _create_pending_approval("learning:engine", "learning.note")
    heal_id = _create_pending_approval("heal:selfheal", "selfheal.apply")
    result = _invoke(
        "approval", "cleanup", "--older-than", "0", "--apply",
        "--source", "learning:engine",
    )
    assert result.exit_code == 0, result.output
    body = _json(result)
    assert body["expired"] == 2
    statuses = {row["id"]: row["status"] for row in _approval_rows()}
    assert sum(1 for status in statuses.values() if status == "expired") == 2
    assert statuses[heal_id] == "pending"
    # Second scoped run: nothing left to expire.
    again = _json(_invoke(
        "approval", "cleanup", "--older-than", "0", "--apply",
        "--source", "learning:engine",
    ))
    assert again["expired"] == 0
    # Unscoped dry run still sees the leftover pending approval.
    dry = _json(_invoke("approval", "cleanup", "--older-than", "0"))
    assert dry["candidates"] == 1


def test_approval_cleanup_rejects_chat_principals_and_negative_windows() -> None:
    _create_pending_approval("learning:engine", "learning.note")
    resolver = _invoke(
        "approval", "cleanup", "--older-than", "0", "--apply",
        "--resolver", "chat:leo",
    )
    assert resolver.exit_code == 1
    assert "cannot resolve approvals" in _json(resolver)["error"]
    scoped = _invoke(
        "approval", "cleanup", "--older-than", "0", "--apply",
        "--source", "chat:leo",
    )
    assert scoped.exit_code == 1
    assert "chat actor" in _json(scoped)["error"]
    window = _invoke("approval", "cleanup", "--older-than", "-1", "--apply")
    assert window.exit_code == 1
    assert "older_than_hours" in _json(window)["error"]
    # The rejected runs touched nothing.
    assert _approval_rows()[0]["status"] == "pending"


def test_task_cancel_terminal_is_json_error() -> None:
    created = _json(_invoke("task", "create", "harden cli cancel"))
    task_id = created["id"]
    first = _invoke("task", "cancel", task_id)
    assert first.exit_code == 0
    second = _invoke("task", "cancel", task_id)
    assert second.exit_code == 1
    body = _json(second)
    assert body["ok"] is False
    assert "cannot cancel terminal" in body["error"]


# ---------------------------------------------------------------------------
# Malformed / invalid arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["bogus", "SEMANTIC", ""])
def test_memory_add_rejects_invalid_category(category: str) -> None:
    result = _invoke("memory", "add", "some-fact", "--category", category)
    assert result.exit_code == 2
    body = _json(result)
    assert body["ok"] is False
    assert "invalid category" in body["error"]
    assert "semantic" in body["error"]


def test_memory_add_rejects_invalid_kind() -> None:
    result = _invoke("memory", "add", "some-fact", "--kind", "bogus")
    assert result.exit_code == 2
    body = _json(result)
    assert "invalid kind" in body["error"]


def test_task_list_rejects_invalid_status() -> None:
    result = _invoke("task", "list", "--status", "bogus")
    assert result.exit_code == 2
    body = _json(result)
    assert "invalid status" in body["error"]


def test_skill_run_rejects_malformed_json() -> None:
    result = _invoke("skill", "run", "summarize_text", "--arguments", "{not json")
    assert result.exit_code == 2
    body = _json(result)
    assert "invalid JSON arguments" in body["error"]


def test_skill_run_rejects_non_object_json() -> None:
    for raw in ("[1, 2]", "42", "null"):
        result = _invoke("skill", "run", "summarize_text", "--arguments", raw)
        assert result.exit_code == 2, raw
        body = _json(result)
        assert "JSON object" in body["error"]


def test_ask_rejects_empty_message() -> None:
    result = _invoke("ask", "   ")
    assert result.exit_code == 2
    body = _json(result)
    assert "must not be empty" in body["error"]


def test_task_create_rejects_empty_objective() -> None:
    result = _invoke("task", "create", "  ")
    assert result.exit_code == 2
    body = _json(result)
    assert "must not be empty" in body["error"]


# ---------------------------------------------------------------------------
# Successful commands still emit structured JSON
# ---------------------------------------------------------------------------


def test_model_list_emits_objects_not_repr_strings() -> None:
    result = _invoke("model", "list")
    assert result.exit_code == 0, result.output
    providers = _json(result)
    assert providers
    for provider in providers.values():
        assert isinstance(provider, dict)
        assert "kind" in provider
        assert "model" in provider


def test_model_status_reports_explicit_simulated_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicit mock provider is a valid offline opt-in; with no live
    # provider, the status is simulated rather than blocked or live.
    settings = Settings()
    settings.models.providers = {"mock": ModelProviderConfig(kind="mock", model="mock-1")}
    settings.models.default = "mock"
    settings.models.fallback = None
    monkeypatch.setitem(cli_mod._state, "settings", settings)
    result = _invoke("model", "status")
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["simulated"] is True
    assert payload["status"] == "simulated"
    assert "mock" in payload["mock_providers"]


def test_skill_run_emits_structured_result() -> None:
    result = _invoke(
        "skill", "run", "summarize_text",
        "--arguments", '{"text": "LEO is a local agent. It runs offline. It is testable."}',
    )
    assert result.exit_code == 0, result.output
    payload = _json(result)["result"]
    assert isinstance(payload, dict)
    assert payload["success"] is True
    assert isinstance(payload["output"], dict)
    assert "summary" in payload["output"]


def test_skill_run_unknown_skill_is_json_error_with_nonzero_exit() -> None:
    result = _invoke("skill", "run", "no-such-skill", "--arguments", "{}")
    assert result.exit_code == 1
    payload = _json(result)["result"]
    assert payload["success"] is False
    assert payload["error"] == "unknown skill"
