"""Tests for the tool registry, executor and native tools."""

from __future__ import annotations

import asyncio
import gzip
import json
import zlib
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall, ToolSpec
from leo.events import EventBus
from leo.security.permissions import ApprovalRequired, PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry, builtins, register_builtins


async def _echo(args: dict, ctx: dict):
    return {"echo": args, "actor": ctx["actor"], "db_path": ctx["db_path"]}


def _register(env, name: str, risk: RiskLevel, handler=_echo, schema=None, **extra):
    spec = ToolSpec(
        name=name,
        description=f"test tool {name}",
        risk=risk,
        input_schema=schema or {"type": "object", "properties": {}, "additionalProperties": True},
        **extra,
    )
    env.registry.register(spec, handler)
    return spec


async def _run(env, name: str, arguments: dict | None = None, approval_id=None, **kwargs):
    call = ToolCall(name=name, arguments=arguments or {}, **kwargs)
    return await env.executor.execute(call, approval_id=approval_id)


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events,
        permissions=permissions, registry=registry, executor=executor,
    )


# --------------------------------------------------------------------------- registry


async def test_registration_and_validation(env):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "count": {"type": "integer", "enum": [1, 2]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "count"],
        "additionalProperties": False,
    }
    spec = _register(env, "sample", RiskLevel.LOW, schema=schema)

    assert env.registry.get("sample").spec is spec
    assert env.registry.get("sample").name == "sample"
    assert [s.name for s in env.registry.list_specs()] == ["sample"]
    assert env.registry.validate("sample", {"name": "a", "count": 2, "tags": ["x"]}) == []

    errors = env.registry.validate("sample", {"count": 3, "extra": 1, "tags": [4]})
    joined = " ".join(errors)
    assert "name" in joined and "missing required" in joined
    assert "count" in joined and "one of" in joined
    assert "extra" in joined and "unknown field" in joined
    assert "tags[0]" in joined and "expected str" in joined

    wrong_type = env.registry.validate("sample", {"name": 1, "count": "1"})
    assert any("expected str" in e for e in wrong_type)
    assert any("expected int" in e or "one of" in e for e in wrong_type)
    assert env.registry.validate("missing_tool", {}) == ["unknown tool: 'missing_tool'"]

    with pytest.raises(ValueError, match="already registered"):
        env.registry.register(ToolSpec(name="sample", description="dup"), _echo)
    assert env.registry.unregister("sample") is True
    assert env.registry.unregister("sample") is False
    assert env.registry.get("sample") is None


async def test_actionable_validation_errors(builtin_env):
    """A rejected field must name the tool, field, valid fields and a suggestion."""
    env = builtin_env
    # `limit` is valid on search_files/list_files, not on terminal.
    bad = await _run(env, "terminal", {"command": "pwd", "limit": 10})
    assert not bad.success and bad.decision == "invalid_arguments"
    text = bad.error or ""
    assert "tool 'terminal'" in text
    assert "unknown field 'limit'" in text
    assert "valid fields:" in text and "command" in text
    assert "did you mean 'search_files.limit'" in text
    assert "unknown field (additionalProperties=false)" in text  # raw detail kept

    # Type errors are equally actionable and keep the raw validator message.
    wrong = await _run(env, "read_file", {"path": 42})
    assert not wrong.success
    assert "tool 'read_file'" in (wrong.error or "")
    assert "invalid value for field 'path'" in (wrong.error or "")
    assert "expected str" in (wrong.error or "")
    assert "valid fields:" in (wrong.error or "")


async def test_read_file_offset_and_limit_are_bounded(builtin_env):
    env = builtin_env
    path = env.tmp_path / "lines.txt"
    path.write_text("".join(f"line-{n}\n" for n in range(1, 11)), encoding="utf-8")

    first = await _run(env, "read_file", {"path": str(path), "limit": 2})
    assert first.success, first.error
    assert first.output["content"] == "line-1\nline-2\n"
    assert first.output["has_more"] is True and first.output["truncated"] is True
    assert first.output["next_offset"] == 2

    second = await _run(env, "read_file", {
        "path": str(path), "offset": first.output["next_offset"], "limit": 2,
    })
    assert second.success and second.output["content"] == "line-3\nline-4\n"
    assert second.output["offset"] == 2 and second.output["next_offset"] == 4

    tail = await _run(env, "read_file", {"path": str(path), "offset": 9})
    assert tail.success and tail.output["content"] == "line-10\n"
    assert tail.output["has_more"] is False

    beyond = await _run(env, "read_file", {"path": str(path), "limit": 10 ** 6})
    assert not beyond.success and beyond.decision == "invalid_arguments"
    assert "must be <=" in (beyond.error or "")


async def test_list_files_limit_and_cursor(builtin_env):
    env = builtin_env
    for n in range(5):
        (env.tmp_path / f"f{n}.txt").write_text(str(n), encoding="utf-8")

    first = await _run(env, "list_files", {"path": str(env.tmp_path), "limit": 2})
    assert first.success, first.error
    assert len(first.output["files"]) == 2 and first.output["count"] == 2
    assert first.output["total"] >= 5 and first.output["has_more"] is True
    cursor = first.output["next_cursor"]
    assert cursor

    second = await _run(env, "list_files", {
        "path": str(env.tmp_path), "limit": 2, "cursor": cursor,
    })
    assert second.success, second.error
    assert second.output["files"] != first.output["files"]
    assert second.output["offset"] == 2
    assert len(set(first.output["files"] + second.output["files"])) == 4

    # Cursor may also name the last entry of the previous page.
    resumed = await _run(env, "list_files", {
        "path": str(env.tmp_path), "cursor": second.output["files"][-1],
    })
    assert resumed.success and resumed.output["offset"] == 4

    invalid = await _run(env, "list_files", {
        "path": str(env.tmp_path), "cursor": "not-a-real-entry",
    })
    assert not invalid.success and "invalid cursor" in (invalid.error or "")


async def test_search_files_accepts_file_path(builtin_env):
    env = builtin_env
    target = env.tmp_path / "one.txt"
    target.write_text("alpha\nneedle here\n", encoding="utf-8")
    other = env.tmp_path / "two.txt"
    other.write_text("needle elsewhere\n", encoding="utf-8")

    result = await _run(env, "search_files", {
        "path": str(target), "pattern": "needle",
    })
    assert result.success, result.error
    assert result.output["path"] == str(target)
    assert [m["line"] for m in result.output["matches"]] == [2]
    assert all(m["path"] == str(target) for m in result.output["matches"])

    outside = env.tmp_path.parent / "outside-search.txt"
    outside.write_text("needle secret\n", encoding="utf-8")
    denied = await _run(env, "search_files", {"path": str(outside), "pattern": "needle"})
    assert not denied.success and denied.decision == "denied"
    assert "outside allowed roots" in (denied.error or "")


def test_terminal_spec_documents_actual_policy(builtin_env):
    spec = builtin_env.registry.get("terminal").spec
    description = spec.description
    allowed = description.split("Allowed executables: ", 1)[1].split(".", 1)[0]
    for executable in builtins.ALLOWED_EXECUTABLES:
        assert executable in allowed
    for interpreter in ("python", "python3", "node", "sh", "bash"):
        assert interpreter not in allowed
    for symbol in (";", "|", "&", "$", "`", "<", ">"):
        assert symbol in description
    assert "security.allowed_executables" in description


# --------------------------------------------------------------------------- permissions


async def test_low_medium_high_permission_flow(env):
    _register(env, "low_tool", RiskLevel.LOW)
    _register(env, "medium_tool", RiskLevel.MEDIUM)
    _register(env, "high_tool", RiskLevel.HIGH, schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    })

    low = await _run(env, "low_tool")
    assert low.success and low.decision == "allowed" and low.risk is RiskLevel.LOW
    medium = await _run(env, "medium_tool")
    assert medium.success and medium.decision == "allowed" and medium.risk is RiskLevel.MEDIUM

    arguments = {"path": str(env.tmp_path / "high.txt")}
    with pytest.raises(ApprovalRequired) as raised:
        await _run(env, "high_tool", arguments, task_id="t-high")
    approval_id = raised.value.approval_id
    assert approval_id
    row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
    assert row["status"] == "pending" and row["action"] == "tool:high_tool"
    assert row["resource"] == arguments["path"]

    await env.permissions.resolve_approval(approval_id, True)
    approved = await _run(env, "high_tool", arguments, approval_id=approval_id)
    assert approved.success and approved.decision == "allowed"
    assert "prior approval" in approved.metadata["reason"]

    with pytest.raises(ApprovalRequired) as second:
        await _run(env, "high_tool", arguments)
    denied_id = second.value.approval_id
    await env.permissions.resolve_approval(denied_id, False)
    denied = await _run(env, "high_tool", arguments, approval_id=denied_id)
    assert not denied.success and denied.decision == "denied"
    assert "denied" in denied.error

    audit = await db.fetchall("SELECT * FROM audit_log WHERE action='tool:low_tool'")
    assert audit and audit[0]["decision"] == "allowed"


async def test_medium_risk_disabled_by_policy(env):
    env.settings.security.allow_medium_risk = False
    _register(env, "medium_tool", RiskLevel.MEDIUM)
    result = await _run(env, "medium_tool")
    assert not result.success
    assert result.decision == "denied"
    assert "medium-risk" in result.error


async def test_path_escape_denied(env):
    register_builtins(env.registry)
    outside = str(env.tmp_path.parent / "outside-secret.txt")
    result = await _run(env, "read_file", {"path": outside})
    assert not result.success
    assert "path outside allowed roots" in result.error
    assert result.decision == "denied"

    with pytest.raises(PermissionError):
        await builtins.read_file({"path": outside}, {"settings": env.settings})


# --------------------------------------------------------------------------- executor controls


async def test_dry_run_does_not_execute(env):
    calls: list[dict] = []

    async def handler(args, ctx):
        calls.append(args)
        return "executed"

    _register(env, "dry_tool", RiskLevel.LOW, handler)
    result = await _run(env, "dry_tool", {"value": 1}, dry_run=True)
    assert result.success and result.output is None
    assert result.metadata["dry_run"] is True
    assert calls == []
    row = await db.fetchone("SELECT * FROM tool_calls WHERE id=?", (result.call_id,))
    assert row["success"] == 1 and row["decision"] == "allowed"


async def test_timeout_and_handler_error(env):
    async def slow(args, ctx):
        await asyncio.sleep(0.3)
        return "too late"

    async def broken(args, ctx):
        raise RuntimeError("boom")

    _register(env, "slow_tool", RiskLevel.LOW, slow, timeout_seconds=0.01)
    _register(env, "broken_tool", RiskLevel.LOW, broken)
    timed_out = await _run(env, "slow_tool")
    assert not timed_out.success and timed_out.decision == "timeout"
    assert "timed out" in timed_out.error
    assert "structured tool" in (timed_out.error or "")
    assert timed_out.duration_ms < 250

    failed = await _run(env, "broken_tool")
    assert not failed.success and failed.decision == "error"
    assert "RuntimeError: boom" in failed.error
    assert await db.fetchone("SELECT * FROM tool_calls WHERE id=?", (failed.call_id,))
    assert await env.events.history("tool.failed")


async def test_timeout_error_points_away_from_binary_targets(env):
    async def slow(args, ctx):
        await asyncio.sleep(0.3)
        return "too late"

    _register(env, "binary_tool", RiskLevel.LOW, slow, timeout_seconds=0.01)
    result = await _run(env, "binary_tool", {"path": "data.sqlite3"})
    assert not result.success and result.decision == "timeout"
    assert "timed out after 0.01s" in (result.error or "")
    for suggestion in ("structured", "task_list", "search_files"):
        assert suggestion in (result.error or "")
    # The message changed; the deadline and its metadata did not.
    assert result.metadata["timeout_seconds"] == 0.01
    assert result.duration_ms < 250

    # A binary-looking argument on any tool gets the same guidance.
    nested = await _run(env, "binary_tool", {"args": ["strings", "blob.bin"]})
    assert not nested.success and "structured" in (nested.error or "")

    plain = await _run(env, "binary_tool", {"query": "text"})
    assert not plain.success and "Reduce the scope" in (plain.error or "")
    assert "task_list" not in (plain.error or "")


async def test_per_task_call_limit(env):
    env.settings.security.max_tool_calls_per_task = 1
    _register(env, "limited", RiskLevel.LOW)
    first = await _run(env, "limited", task_id="task-1")
    second = await _run(env, "limited", task_id="task-1")
    other = await _run(env, "limited", task_id="task-2")
    assert first.success
    assert not second.success and second.decision == "rate_limited"
    assert "limit exceeded" in second.error
    assert other.success


async def test_secret_redaction_in_output_events_and_audit(env):
    secret = "sk-abcdefghijklmnop123456"

    async def handler(args, ctx):
        return {"status": "ok", "note": f"api_key: {secret}"}

    _register(env, "leaky", RiskLevel.LOW, handler)
    result = await _run(env, "leaky", {"api_token": secret})
    assert result.success
    assert secret not in str(result.output)
    assert "[REDACTED]" in str(result.output)

    row = await db.fetchone("SELECT * FROM tool_calls WHERE id=?", (result.call_id,))
    assert secret not in row["args"]
    events = await env.events.history("tool.called")
    assert events and secret not in events[0].model_dump_json()


# --------------------------------------------------------------------------- builtins


@pytest.fixture
def builtin_env(env):
    register_builtins(env.registry)
    return env


async def test_filesystem_builtins(builtin_env):
    env = builtin_env
    path = env.tmp_path / "sample.txt"
    path.write_text("alpha\nneedle here\n", encoding="utf-8")

    read = await _run(env, "read_file", {"path": str(path)})
    assert read.success and read.output["content"] == "alpha\nneedle here\n"

    listing = await _run(env, "list_files", {"path": str(env.tmp_path)})
    assert listing.success and "sample.txt" in listing.output["files"]

    search = await _run(env, "search_files", {"path": str(env.tmp_path), "pattern": "needle"})
    assert search.success and search.output["matches"][0]["line"] == 2

    written = await _run(
        env, "write_file",
        {"path": str(env.tmp_path / "sub" / "new.txt"), "content": "hello"},
    )
    assert written.success and written.output["bytes_written"] == 5
    assert (env.tmp_path / "sub" / "new.txt").read_text() == "hello"

    missing = await _run(env, "read_file", {"path": str(env.tmp_path / "nope.txt")})
    assert not missing.success and "not a file" in missing.error


async def test_utility_builtins(builtin_env):
    env = builtin_env
    parsed = await _run(env, "json_parse", {"text": '{"a": [1, 2]}'})
    assert parsed.success and parsed.output == {"a": [1, 2]}
    invalid = await _run(env, "json_parse", {"text": "{oops"})
    assert not invalid.success and "invalid JSON" in invalid.error

    calculated = await _run(env, "calculator", {"expression": "2 * (3 + 4)"})
    assert calculated.success and calculated.output == 14
    unsafe = await _run(env, "calculator", {"expression": "__import__('os').system('echo pwned')"})
    assert not unsafe.success
    assert await _run(env, "calculator", {"expression": "2 ** 10000"}) is not None

    now = await _run(env, "current_time")
    assert now.success and "T" in now.output


async def test_terminal_builtin(builtin_env):
    env = builtin_env
    result = await _run(env, "terminal", {"command": "echo hello"})
    assert result.success, result.error
    assert result.output["exit_code"] == 0
    assert "hello" in result.output["stdout"]

    for command in ("rm -rf /", "echo a; echo b", "curl http://example.com"):
        denied = await _run(env, "terminal", {"command": command})
        assert not denied.success
        assert "not allowed" in denied.error

    escape = await _run(env, "terminal", {"command": "pwd", "cwd": str(env.tmp_path.parent)})
    assert not escape.success and escape.decision == "denied"


async def test_rg_missing_falls_back_to_bounded_grep(builtin_env, monkeypatch):
    env = builtin_env
    target = env.tmp_path / "search.txt"
    target.write_text("needle alpha\nneedle beta\nother line\n", encoding="utf-8")
    real_which = builtins.shutil.which
    monkeypatch.setattr(
        builtins.shutil, "which",
        lambda name: None if name == "rg" else real_which(name),
    )
    ctx = {"settings": env.settings}

    parts = builtins._command_parts(f"rg needle {target}", ctx)
    assert parts[0] == "grep"
    assert parts[1:5] == ["-F", "-m", str(builtins.RG_FALLBACK_MAX_MATCHES), "-n"]
    assert "-H" in parts and parts[-2:] == ["needle", str(target)]
    assert "rg is not installed" in parts.fallback_notice

    result = await _run(env, "terminal", {"command": f"rg needle {target}"})
    assert result.success, result.error
    assert result.output["exit_code"] == 0
    assert result.output["stdout"].count("needle") == 2  # one path:line:match per hit
    assert "alpha" in result.output["stdout"] and "beta" in result.output["stdout"]
    assert "rg is not installed" in result.output["notice"]
    assert "Fixed-string" in result.output["notice"]
    assert result.output["command_effective"].startswith("grep -F -m 50")

    dry = await _run(env, "terminal", {"command": f"rg needle {target}"}, dry_run=True)
    assert dry.success and dry.output["notice"] == result.output["notice"]

    # A user-supplied max is clamped, and unsupported rg flags fail explicitly.
    clamped = builtins._command_parts(f"rg -m 500 needle {target}", ctx)
    assert clamped[clamped.index("-m") + 1] == str(builtins.RG_FALLBACK_MAX_MATCHES)
    unsupported = await _run(env, "terminal", {"command": f"rg --json needle {target}"})
    assert not unsupported.success
    assert "no safe grep -F fallback" in (unsupported.error or "")
    assert "search_files" in (unsupported.error or "")

    monkeypatch.setattr(builtins.shutil, "which", lambda name: None)
    no_grep = await _run(env, "terminal", {"command": f"rg needle {target}"})
    assert not no_grep.success and "search_files" in (no_grep.error or "")


async def test_read_file_streams_only_max_bytes(builtin_env, monkeypatch):
    """max_bytes must bound what is read, not just what is returned."""
    env = builtin_env
    path = env.tmp_path / "big.txt"
    path.write_text("0123456789" * 1000, encoding="utf-8")

    def no_full_read(self, *a, **k):
        raise AssertionError("read_file must not load the whole file")

    monkeypatch.setattr(builtins.Path, "read_bytes", no_full_read)
    out = await builtins.read_file(
        {"path": str(path), "max_bytes": 10}, {"settings": env.settings}
    )
    assert out["content"] == "0123456789"
    assert out["truncated"] is True

    # An oversized cap must not overflow read(); it just returns the full file.
    full = await builtins.read_file(
        {"path": str(path), "max_bytes": 10 ** 30}, {"settings": env.settings}
    )
    assert full["content"] == "0123456789" * 1000 and full["truncated"] is False


async def test_list_files_glob_cannot_escape_root(builtin_env, monkeypatch):
    env = builtin_env
    ctx = {"settings": env.settings}
    # '../*' used to enumerate siblings and yield the root itself as '.' names.
    with pytest.raises(ValueError, match="'\\.\\.'"):
        await builtins.list_files({"path": str(env.tmp_path), "pattern": "../*"}, ctx)
    for bad in (None, 123, "", "/etc/*", "**/../*"):
        with pytest.raises(ValueError):
            await builtins.list_files({"path": str(env.tmp_path), "pattern": bad}, ctx)

    # Defense in depth: even if a glob returns out-of-root entries, only real
    # in-root children survive. Regression for the old ['.', '.'] output.
    (env.tmp_path / "ok.txt").write_text("ok")
    def escaping_glob(self, pattern):
        yield self                 # resolves to root -> old code emitted '.'
        yield self.parent          # outside the allowed root
        yield self / "ok.txt"      # the one legitimate child
    monkeypatch.setattr(builtins.Path, "glob", escaping_glob)
    out = await builtins.list_files({"path": str(env.tmp_path), "pattern": "*"}, ctx)
    assert out["files"] == ["ok.txt"]


async def test_search_files_skips_outside_symlinks_and_bad_regex(builtin_env):
    env = builtin_env
    ctx = {"settings": env.settings}
    outside = env.tmp_path.parent / "outside-secret-search.txt"
    outside.write_text("TOPSECRETVALUE\n", encoding="utf-8")
    (env.tmp_path / "leak.txt").symlink_to(outside)
    (env.tmp_path / "plain.txt").write_text("nothing here\n", encoding="utf-8")

    out = await builtins.search_files(
        {"path": str(env.tmp_path), "pattern": "TOPSECRETVALUE"}, ctx
    )
    assert out["matches"] == []

    with pytest.raises(ValueError, match="invalid regex"):
        await builtins.search_files(
            {"path": str(env.tmp_path), "pattern": "[", "regex": True}, ctx
        )

    zero = await builtins.search_files(
        {"path": str(env.tmp_path), "pattern": "nothing", "limit": 0}, ctx
    )
    assert zero["matches"] == []


async def test_terminal_rejects_git_config_injection(builtin_env):
    env = builtin_env
    ctx = {"settings": env.settings}
    for cmd in (
        "git --config=core.fsmonitor=evil.sh status",
        "git -ccore.fsmonitor=evil.sh status",
        "git -c core.fsmonitor=evil.sh status",
        "git --config-env=x=PATH status",
        "git --exec-path=/tmp/x status",
        "git -C /etc status",
    ):
        with pytest.raises(PermissionError):
            builtins._command_parts(cmd, ctx)
    assert builtins._command_parts("git status", ctx) == ["git", "status"]


class _FakeResponse:
    def __init__(
        self,
        body: bytes = b"hello world",
        status: int = 200,
        headers: dict[str, str] | None = None,
        raw_chunks: list[bytes] | None = None,
    ):
        self.body = body
        self.status_code = status
        self.headers = headers or {"content-type": "text/plain", "location": "/next"}
        self.raw_chunks = raw_chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_raw(self):
        # http_get must cap the WIRE stream; aiter_bytes() decompresses first.
        if self.raw_chunks is not None:
            for chunk in self.raw_chunks:
                yield chunk
        else:
            yield self.body[:6]
            yield self.body[6:]

    async def aiter_bytes(self):
        raise AssertionError("http_get must read aiter_raw under the byte cap")


class _FakeClient:
    instances: list[dict] = []
    requests: list[dict] = []  # every stream() call, for pinning assertions

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, headers=None, extensions=None):
        assert method == "GET"
        _FakeClient.requests.append(
            {"method": method, "url": url, "headers": headers, "extensions": extensions}
        )
        if url.endswith("/redirect"):
            return _FakeResponse(b"", status=302)
        if url.endswith("/big"):
            return _FakeResponse(b"x" * 5000)
        if url.endswith("/gzip"):
            # 2 MiB of zeros in ~2 KiB on the wire: aiter_bytes() would expand
            # the whole body before http_get could apply max_bytes.
            compressed = gzip.compress(b"x" * (2 * 1024 * 1024))
            return _FakeResponse(
                compressed,
                headers={"content-type": "application/json", "content-encoding": "gzip"},
                raw_chunks=[compressed],  # one oversized wire chunk, like httpx
            )
        if url.endswith("/br"):
            return _FakeResponse(
                b"", headers={"content-type": "application/json", "content-encoding": "br"}
            )
        return _FakeResponse()


@pytest.fixture
def no_network(monkeypatch):
    _FakeClient.instances = []
    _FakeClient.requests = []
    monkeypatch.setattr(builtins.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(
        builtins, "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port or 443))],
    )


async def test_http_get_builtin(builtin_env, no_network):
    env = builtin_env
    result = await _run(env, "http_get", {"url": "https://example.com/page"})
    assert result.success, result.error
    assert result.output["status_code"] == 200
    assert result.output["content"] == "hello world"
    assert _FakeClient.instances[0]["follow_redirects"] is False
    assert _FakeClient.instances[0]["trust_env"] is False

    redirect = await _run(env, "http_get", {"url": "https://example.com/redirect"})
    assert redirect.success and redirect.output["status_code"] == 302

    capped = await _run(env, "http_get", {"url": "https://example.com/big", "max_bytes": 100})
    assert capped.success and capped.output["truncated"] is True
    assert len(capped.output["content"]) <= 101


async def test_http_get_security_denials(builtin_env, no_network, monkeypatch):
    env = builtin_env
    for url in ("file:///etc/passwd", "http://127.0.0.1:8080/", "http://[::1]/", "http://localhost/"):
        result = await _run(env, "http_get", {"url": url})
        assert not result.success, url
        assert "allowed" in result.error or "only http" in result.error

    before = len(_FakeClient.instances)
    monkeypatch.setattr(
        builtins, "resolve_host",
        lambda host, port: [(2, 1, 6, "", ("10.0.0.5", port or 80))],
    )
    private = await _run(env, "http_get", {"url": "http://internal.example/"})
    assert not private.success and "non-public" in private.error
    assert len(_FakeClient.instances) == before


async def test_http_get_pins_connection_to_validated_ip(builtin_env, no_network):
    """DNS rebinding is impossible: the check and the connection share one answer."""
    env = builtin_env
    result = await _run(env, "http_get", {"url": "https://example.com/page?x=1"})
    assert result.success, result.error
    assert result.output["url"] == "https://example.com/page?x=1"  # original URL reported
    request = _FakeClient.requests[-1]
    # The connection target is the pre-validated address, never a second
    # DNS answer; the real host is preserved for Host and TLS SNI.
    assert request["url"] == "https://93.184.216.34/page?x=1"
    assert request["headers"] == {
        "Host": "example.com", "Cache-Control": "no-cache",
    }
    assert request["extensions"] == {"sni_hostname": "example.com"}

    # An IP-literal URL is already validated: no rewriting happens.
    literal = await _run(env, "http_get", {"url": "http://93.184.216.34/direct"})
    assert literal.success, literal.error
    direct = _FakeClient.requests[-1]
    assert direct["url"] == "http://93.184.216.34/direct"
    assert direct["headers"] == {"Cache-Control": "no-cache"}
    assert direct["extensions"] is None


REAL_DECOMPRESSOBJ = zlib.decompressobj


class _DecompressorSpy:
    """Records the output bound of every incremental decompression call."""

    outputs: list[int] = []
    limits: list[int] = []

    def __init__(self, *args, **kwargs):
        self._inner = REAL_DECOMPRESSOBJ(*args, **kwargs)

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        _DecompressorSpy.limits.append(max_length)
        out = self._inner.decompress(data, max_length)
        _DecompressorSpy.outputs.append(len(out))
        return out

    @property
    def unconsumed_tail(self) -> bytes:
        return self._inner.unconsumed_tail

    def flush(self, length: int = -1) -> bytes:
        return self._inner.flush(length)


async def test_http_get_gzip_expansion_is_bounded_by_max_bytes(builtin_env, no_network, monkeypatch):
    """A 2 KiB gzip body expanding to 2 MiB must not allocate 2 MiB (F2)."""
    _DecompressorSpy.outputs = []
    _DecompressorSpy.limits = []
    monkeypatch.setattr(builtins.zlib, "decompressobj", _DecompressorSpy)

    result = await _run(
        builtin_env, "http_get", {"url": "https://example.com/gzip", "max_bytes": 1024}
    )
    assert result.success, result.error
    assert result.output["truncated"] is True
    assert len(result.output["content"]) <= 1024
    assert _DecompressorSpy.outputs, "decompression path was not used"
    assert max(_DecompressorSpy.outputs) <= 1024
    assert _DecompressorSpy.limits and max(_DecompressorSpy.limits) <= 1024

    unsupported = await _run(builtin_env, "http_get", {"url": "https://example.com/br"})
    assert unsupported.success is False
    assert "unsupported Content-Encoding" in unsupported.error


async def test_http_get_surfaces_fetch_time_and_cache_evidence(builtin_env, no_network):
    result = await _run(builtin_env, "http_get", {"url": "https://example.com/page"})
    assert result.success, result.error
    assert result.output["fetched_at"]  # provenance is a timestamp, not a claim of freshness
    assert result.output["freshness"] == {
        "date": None, "age": None, "cache_control": None, "etag": None,
    }
    assert _FakeClient.requests[-1]["headers"]["Cache-Control"] == "no-cache"


# --------------------------------------------------------------------------- durable records


async def test_tool_call_rows_and_events(builtin_env):
    env = builtin_env
    ok = await _run(env, "calculator", {"expression": "6*7"}, task_id="task-a")
    bad = await _run(env, "calculator", {"expression": "1/0"}, task_id="task-a")

    assert ok.success and ok.output == 42
    assert not bad.success
    row = await db.fetchone("SELECT * FROM tool_calls WHERE id=?", (ok.call_id,))
    assert row["tool"] == "calculator"
    assert row["risk"] == "low" and row["decision"] == "allowed"
    assert row["success"] == 1 and row["duration_ms"] >= 0
    assert json.loads(row["args"]) == {"expression": "6*7"}

    failed_row = await db.fetchone("SELECT * FROM tool_calls WHERE id=?", (bad.call_id,))
    assert failed_row["success"] == 0 and failed_row["error"]

    called = await env.events.history("tool.called")
    failed = await env.events.history("tool.failed")
    assert any(e.payload.get("call_id") == ok.call_id for e in called)
    assert any(e.payload.get("call_id") == bad.call_id for e in failed)

    audit = await db.fetchall("SELECT * FROM audit_log WHERE action='tool:calculator'")
    assert len(audit) >= 2


# --- go-live robustness: quota charging, expiry, reset (report item 1) ---
async def test_quota_charges_only_dispatched_calls_and_expires(env):
    env.settings.security.max_tool_calls_per_task = 1
    _register(env, "quota", RiskLevel.LOW, schema={
        "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"],
    })
    invalid = await _run(env, "quota", {}, task_id="q")
    assert invalid.decision == "invalid_arguments"  # validation is free
    dry = await _run(env, "quota", {"value": 1}, task_id="q", dry_run=True)
    assert dry.success  # dry-run is free too
    real = await _run(env, "quota", {"value": 1}, task_id="q")
    assert real.success
    limited = await _run(env, "quota", {"value": 1}, task_id="q")
    assert limited.decision == "rate_limited"

    status = env.executor.quota_status("task:q", 1)
    assert status["used"] == 1 and status["remaining"] == 0
    assert status["reset_at"] > 0
    assert env.executor.reset_quota("task:q") == 1
    assert (await _run(env, "quota", {"value": 1}, task_id="q")).success

    # A short window expires the charge automatically (no permanent lockout).
    env.executor.QUOTA_WINDOW_SECONDS = 0.05
    await asyncio.sleep(0.1)
    assert (await _run(env, "quota", {"value": 1}, task_id="q")).success


# --- durable, actor-scoped quotas (second remediation wave) ---
async def test_durable_quota_survives_executor_restart(env):
    env.settings.security.max_tool_calls_per_task = 1
    _register(env, "durable", RiskLevel.LOW)
    call = ToolCall(name="durable", task_id="restart", actor="alice")
    assert (await env.executor.execute(call)).success

    restarted = ToolExecutorImpl(env.registry, env.permissions, env.events, env.settings)
    limited = await restarted.execute(call)
    assert limited.decision == "rate_limited"
    assert limited.metadata["quota"]["used"] == 1
    assert limited.metadata["quota"]["durable"] is True
    rows = await db.fetchall("SELECT bucket, count FROM tool_quotas")
    assert rows and all(row["count"] == 1 for row in rows)
    assert rows[0]["bucket"] == "actor:alice|task:restart"


async def test_durable_quota_is_actor_scoped(env):
    env.settings.security.max_tool_calls_per_task = 1
    _register(env, "actor_scoped", RiskLevel.LOW)
    alice = ToolCall(name="actor_scoped", task_id="shared", actor="alice")
    bob = ToolCall(name="actor_scoped", task_id="shared", actor="bob")
    assert (await env.executor.execute(alice)).success
    assert (await env.executor.execute(bob)).success  # other actor has its own bucket
    assert (await env.executor.execute(alice)).decision == "rate_limited"


async def test_client_quota_hint_is_clamped_to_server_ceiling(env):
    env.settings.security.max_tool_calls_per_task = 1000
    env.settings.security.max_tool_calls_ceiling = 2
    _register(env, "capped", RiskLevel.LOW)
    hinted = ToolCall(name="capped", task_id="cap", actor="alice", max_tool_calls=1000)
    assert env.executor._limit_for(hinted) == 2
    assert (await env.executor.execute(hinted)).success
    assert (await env.executor.execute(hinted)).success
    limited = await env.executor.execute(hinted)
    assert limited.decision == "rate_limited"
    assert limited.metadata["quota"]["limit"] == 2


async def test_concurrent_workers_cannot_overshoot_durable_quota(env):
    env.settings.security.max_tool_calls_per_task = 1
    _register(env, "parallel", RiskLevel.LOW)
    workers = [
        ToolExecutorImpl(env.registry, env.permissions, env.events, env.settings)
        for _ in range(2)
    ]
    results = await asyncio.gather(*(
        worker.execute(ToolCall(name="parallel", task_id="parallel", actor="alice"))
        for worker in workers
    ))
    assert sorted(result.decision for result in results) == ["allowed", "rate_limited"]


# --- go-live robustness: strict numbers, dry-run parity, NUL paths ---
async def test_schema_rejects_nonfinite_numbers(env):
    _register(env, "number", RiskLevel.LOW, schema={
        "type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"],
    })
    for value in (float("nan"), float("inf"), 10**20):
        result = await _run(env, "number", {"value": value})
        assert result.decision == "invalid_arguments"


async def test_json_parse_rejects_nonstandard_literals(env):
    register_builtins(env.registry)
    for text in ('{"x": NaN}', '{"x": Infinity}', '{"x": 1e400}'):
        result = await _run(env, "json_parse", {"text": text})
        assert not result.success, text


async def test_dry_run_reports_real_denial_and_nul_path_is_clean(env):
    register_builtins(env.registry)
    dry = await _run(env, "terminal", {"command": "rm -rf /"}, dry_run=True)
    assert not dry.success
    nul = await _run(env, "read_file", {"path": "bad\u0000path"})
    assert not nul.success and "NUL" in str(nul.error)


# --- declared ToolSpec contracts: output_schema and permissions ---
async def test_output_schema_is_enforced_before_success_and_audit(env):
    async def bad_handler(args, ctx):
        return {"text": 123}

    _register(env, "bad_output", RiskLevel.LOW, bad_handler, output_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    })
    result = await _run(env, "bad_output")
    assert not result.success and result.decision == "invalid_output"
    assert "schema" in result.error and result.output is None
    assert result.metadata["output_schema_errors"]
    calls = await db.fetchall("SELECT decision, success FROM tool_calls WHERE tool='bad_output'")
    assert calls[0]["decision"] == "invalid_output" and calls[0]["success"] == 0
    audit = await db.fetchall("SELECT decision FROM audit_log WHERE action='tool.execute'")
    assert any(row["decision"] == "invalid_output" for row in audit)
    emitted = [e for e in await env.events.history("tool.called") if e.payload.get("tool") == "bad_output"]
    assert emitted == []

    async def good_handler(args, ctx):
        return {"text": "ok"}

    _register(env, "good_output", RiskLevel.LOW, good_handler, output_schema={
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
    })
    assert (await _run(env, "good_output")).success is True


async def test_declared_permissions_require_full_caller_grants(env):
    ran: list[int] = []

    async def handler(args, ctx):
        ran.append(1)
        return {"ok": True}

    _register(env, "gated", RiskLevel.LOW, handler, permissions=["file:read", "file:write"])
    # An explicit (even empty) grant policy activates capability enforcement;
    # an unset policy preserves the legacy risk-only behaviour.
    env.settings.security.grants = []
    denied = await _run(env, "gated")
    assert not denied.success and denied.decision == "permission_denied"
    assert "missing capability" in denied.error
    assert denied.metadata["missing_capabilities"] == ["file:read", "file:write"]
    assert ran == []

    env.settings.security.grants = ["file:read"]
    partial = await _run(env, "gated")
    assert not partial.success and partial.metadata["missing_capabilities"] == ["file:write"]
    assert ran == []

    env.settings.security.grants = ["file:read", "file:write"]
    assert (await _run(env, "gated")).success is True
    assert len(ran) == 1

    env.settings.security.grants = []
    explicit = await env.executor.execute(
        ToolCall(name="gated"), grants=["file:read", "file:write"]
    )
    assert explicit.success is True and len(ran) == 2
    tool_rows = await db.fetchall("SELECT decision FROM tool_calls WHERE tool='gated'")
    assert [row["decision"] for row in tool_rows] == ["permission_denied"] * 2 + ["allowed"] * 2


async def test_handler_reported_failure_is_not_published_as_success(env):
    async def preview(args, ctx):
        return {"executed": False, "success": False, "error": "dry-run only"}

    _register(env, "preview", RiskLevel.LOW, preview)
    result = await _run(env, "preview")
    assert result.success is False
    assert result.metadata["handler_success"] is False
    assert "dry-run only" in result.error
    assert result.output["executed"] is False  # payload stays inspectable

    failed = [e for e in await env.events.history("tool.failed") if e.payload.get("tool") == "preview"]
    assert failed and "dry-run only" in failed[0].payload["error"]
    succeeded = [e for e in await env.events.history("tool.called") if e.payload.get("tool") == "preview"]
    assert succeeded == []
