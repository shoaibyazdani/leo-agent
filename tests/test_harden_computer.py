"""Hardening regression tests for ComputerOperator.

No real desktop actions, subprocesses or file mutations occur: every command goes
through an injected fake runner and every delegated file op through a fake tool
executor. Permission decisions come from in-memory stubs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from leo.contracts import Decision, RiskLevel, ToolResult
from leo.integrations.computer import (
    ComputerError,
    ComputerOperator,
    ComputerPermissionDenied,
    UnsupportedOperationError,
    _applescript,
)


# --------------------------------------------------------------------------- fakes


class AllowAll:
    async def authorize(self, actor, action, resource=None, risk=RiskLevel.LOW, **context):
        return Decision(allowed=True, reason="allowed by test policy", risk=risk)

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        return Decision(allowed=approved, reason="resolved", risk=RiskLevel.HIGH)


class RecordingPermissions:
    """Records every authorization and can deny by action or by resource path."""

    def __init__(self, denied_actions=(), denied_resources=(), approval_actions=()) -> None:
        self.denied_actions = set(denied_actions)
        self.denied_resources = set(denied_resources)
        self.approval_actions = set(approval_actions)
        self.calls: list[tuple[str, str, str | None, RiskLevel]] = []

    async def authorize(self, actor, action, resource=None, risk=RiskLevel.LOW, **context):
        self.calls.append((actor, action, resource, risk))
        if action in self.approval_actions:
            return Decision(
                allowed=False, reason="approval required", risk=risk,
                requires_approval=True, approval_id="apr_test",
            )
        if action in self.denied_actions or resource in self.denied_resources:
            return Decision(allowed=False, reason=f"path outside allowed roots: {resource}", risk=risk)
        return Decision(allowed=True, reason="allowed", risk=risk)

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        return Decision(allowed=approved, reason="resolved", risk=RiskLevel.HIGH)


class FakeRunner:
    def __init__(self, stdout: str = "", stderr: str = "", code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.code = code
        self.calls: list[tuple[list[str], str | None, float]] = []

    def __call__(self, argv: list[str], stdin: str | None, timeout: float):
        self.calls.append((list(argv), stdin, timeout))
        return self.code, self.stdout, self.stderr


class FakeExecutor:
    def __init__(self, *, success: bool = True, explode: Exception | None = None) -> None:
        self.calls = []
        self.success = success
        self.explode = explode

    async def execute(self, call):
        self.calls.append(call)
        if self.explode is not None:
            raise self.explode
        return ToolResult(call_id=call.id, tool=call.name, success=self.success, output={"ok": True})


# --------------------------------------------------------------------------- escaping


def test_applescript_escapes_backslash_quote_and_control_chars():
    literal = _applescript('a\\b"c\td\ne\rf')
    assert literal.startswith('"') and literal.endswith('"')
    inner = literal[1:-1]
    assert "\\\\" in inner          # backslash
    assert '\\"' in inner           # double quote
    assert "\\t" in inner           # tab
    assert "\\n" in inner           # newline
    assert "\\r" in inner           # carriage return
    # No raw control characters remain in the generated AppleScript source.
    assert "\t" not in inner and "\n" not in inner and "\r" not in inner


async def test_notify_message_cannot_break_out_of_applescript_string():
    runner = FakeRunner(stdout="ok")
    op = ComputerOperator(
        AllowAll(), runner=runner, probe_runner=FakeRunner(), platform="darwin", dry_run=False
    )
    payload = 'hello" & do shell script "touch /tmp/pwned" & "'
    await op.notify(payload, title='ti"tle')
    script = runner.calls[0][0][2]
    assert script.startswith("display notification ")
    # Extract the message literal and prove every quote inside it is escaped,
    # so the injected text stays data instead of becoming AppleScript code.
    message_literal = script[len("display notification ") :].split(" with title ", 1)[0]
    assert message_literal.startswith('"') and message_literal.endswith('"')
    assert '"' not in message_literal[1:-1].replace('\\"', "")


# --------------------------------------------------------------------------- file.move destination gating


async def test_file_move_authorizes_both_source_and_destination(tmp_path: Path):
    source = str((tmp_path / "inside" / "a.txt").resolve())
    destination = str((tmp_path / "inside" / "b.txt").resolve())
    permissions = RecordingPermissions()
    executor = FakeExecutor()
    op = ComputerOperator(permissions, tool_executor=executor, probe_runner=FakeRunner(), platform="darwin", dry_run=False)
    result = await op.file_move(source, destination)

    resources = [call[2] for call in permissions.calls]
    assert source in resources and destination in resources
    assert result["destination_resource"] == destination
    assert result["destination_decision"] == "allowed"
    assert len(executor.calls) == 1
    assert executor.calls[0].name == "file_move"
    assert executor.calls[0].arguments["destination"] == destination


async def test_file_move_denied_destination_blocks_execution(tmp_path: Path):
    source = str((tmp_path / "inside" / "a.txt").resolve())
    destination = "/etc/evil.txt"
    # _path() resolves symlinks (on macOS /etc -> /private/etc), so the policy
    # sees the resolved form; deny exactly that resource.
    destination_resolved = str(Path(destination).resolve())
    permissions = RecordingPermissions(denied_resources={destination_resolved})
    executor = FakeExecutor()
    op = ComputerOperator(permissions, tool_executor=executor, probe_runner=FakeRunner(), platform="darwin", dry_run=False)
    with pytest.raises(ComputerPermissionDenied, match="path outside allowed roots"):
        await op.file_move(source, destination)
    assert executor.calls == []
    resources = [call[2] for call in permissions.calls]
    assert source in resources and destination_resolved in resources


async def test_file_move_dry_run_authorizes_but_does_not_execute(tmp_path: Path):
    source = str((tmp_path / "in" / "a.txt").resolve())
    destination = str((tmp_path / "in" / "b.txt").resolve())
    permissions = RecordingPermissions()
    executor = FakeExecutor()
    op = ComputerOperator(
        permissions, tool_executor=executor, probe_runner=FakeRunner(), platform="darwin", dry_run=True
    )
    result = await op.file_move(source, destination)
    assert result["dry_run"] is True and result["executed"] is False
    assert executor.calls == []
    assert len(permissions.calls) == 2  # source + destination still gated in dry-run


# --------------------------------------------------------------------------- subprocess results / argv


async def test_subprocess_nonzero_exit_is_surfaced_as_failure():
    runner = FakeRunner(code=1, stderr="boom")
    op = ComputerOperator(
        AllowAll(), runner=runner, probe_runner=FakeRunner(), platform="darwin", dry_run=False
    )
    results = [
        await op.notify("x"),
        await op.dialog("y"),
        await op.screenshot("/tmp/harden-shot.png"),
        await op.clipboard_read(),
        await op.clipboard_write("z"),
    ]
    for result in results:
        assert result["executed"] is True
        assert result["success"] is False
        assert result["exit_code"] == 1
    assert results[0]["stderr"] == "boom"


async def test_subprocess_zero_exit_is_success():
    runner = FakeRunner(stdout="ok")
    op = ComputerOperator(AllowAll(), runner=runner, platform="darwin", dry_run=False)
    for result in (await op.notify("x"), await op.clipboard_read()):
        assert result["success"] is True and result["exit_code"] == 0


async def test_all_subprocess_calls_use_argv_and_clamped_timeout():
    runner = FakeRunner(stdout="ok")
    op = ComputerOperator(
        AllowAll(), runner=runner, probe_runner=FakeRunner(), platform="darwin",
        dry_run=False, timeout=7.0,
    )
    await op.notify("m")
    await op.dialog("d")
    await op.screenshot("/tmp/harden-shot.png")
    await op.clipboard_read()
    await op.clipboard_write("t")

    assert len(runner.calls) == 5
    for argv, stdin, timeout in runner.calls:
        assert isinstance(argv, list) and argv
        assert timeout == 7.0
    assert runner.calls[0][0][0] == "osascript" and runner.calls[0][0][1] == "-e"
    assert runner.calls[1][0][0] == "osascript"
    assert runner.calls[2][0][:2] == ["screencapture", "-x"]
    assert runner.calls[3][0] == ["pbpaste"]
    assert runner.calls[4][0] == ["pbcopy"] and runner.calls[4][1] == "t"


def test_timeout_is_clamped_to_safe_range():
    probe = FakeRunner()
    assert ComputerOperator(AllowAll(), probe_runner=probe, timeout=0).timeout == 0.1
    assert ComputerOperator(AllowAll(), probe_runner=probe, timeout=9999).timeout == 30.0
    assert ComputerOperator(AllowAll(), probe_runner=probe, timeout=5).timeout == 5.0


async def test_runner_exceptions_are_wrapped_in_computer_error():
    def timed_out(argv, stdin, timeout):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    def missing(argv, stdin, timeout):
        raise FileNotFoundError(argv[0])

    op = ComputerOperator(
        AllowAll(), runner=timed_out, probe_runner=FakeRunner(), platform="darwin", dry_run=False
    )
    with pytest.raises(ComputerError, match="timed out"):
        await op.notify("x")

    op = ComputerOperator(
        AllowAll(), runner=missing, probe_runner=FakeRunner(), platform="darwin", dry_run=False
    )
    with pytest.raises(ComputerError, match="could not be started"):
        await op.notify("x")


# --------------------------------------------------------------------------- dry-run / permission / platform / delegation


async def test_mutations_are_dry_run_by_default():
    runner = FakeRunner(stdout="clip")
    op = ComputerOperator(
        AllowAll(), runner=runner, probe_runner=FakeRunner(), platform="darwin", dry_run=True
    )
    for result in (await op.notify("x"), await op.dialog("y"),
                   await op.screenshot("/tmp/harden-shot.png"), await op.clipboard_write("z")):
        assert result["dry_run"] is True and result["executed"] is False
    assert runner.calls == []


async def test_dry_run_suppresses_clipboard_read_unless_opted_in():
    runner = FakeRunner(stdout="clip")
    op = ComputerOperator(
        AllowAll(), runner=runner, probe_runner=FakeRunner(), platform="darwin", dry_run=True
    )
    result = await op.clipboard_read()
    assert result["dry_run"] is True and result["executed"] is False
    assert result["text"] is None and result["host_read"] == "suppressed"
    assert runner.calls == []

    opted_in = await op.clipboard_read(allow_host_read=True)
    assert opted_in["executed"] is True and opted_in["text"] == "clip"
    assert runner.calls[0][0] == ["pbpaste"]


async def test_dry_run_suppresses_file_read_unless_opted_in():
    executor = FakeExecutor()
    op = ComputerOperator(
        AllowAll(), tool_executor=executor, probe_runner=FakeRunner(), platform="darwin", dry_run=True
    )
    result = await op.file_read("/tmp/harden-read.txt")
    assert result["dry_run"] is True and result["executed"] is False
    assert result["host_read"] == "suppressed" and executor.calls == []

    opted_in = await op.file_read("/tmp/harden-read.txt", allow_host_read=True)
    assert opted_in["executed"] is True and opted_in["host_read"] == "allowed"
    assert executor.calls[0].name == "file_read"


async def test_permission_denial_blocks_execution():
    permissions = RecordingPermissions(denied_actions={"computer.notify"})
    runner = FakeRunner()
    op = ComputerOperator(
        permissions, runner=runner, probe_runner=FakeRunner(), platform="darwin", dry_run=False
    )
    with pytest.raises(ComputerPermissionDenied):
        await op.notify("x")
    assert runner.calls == []


async def test_pending_approval_denial_carries_decision():
    permissions = RecordingPermissions(approval_actions={"computer.notify"})
    runner = FakeRunner()
    op = ComputerOperator(
        permissions, runner=runner, probe_runner=FakeRunner(), platform="darwin", dry_run=False
    )
    with pytest.raises(ComputerPermissionDenied) as denied:
        await op.notify("x")
    assert "approval_id=apr_test" in str(denied.value)
    assert denied.value.decision.approval_id == "apr_test"
    assert runner.calls == []


async def test_desktop_actions_require_macos_when_live():
    op = ComputerOperator(AllowAll(), runner=FakeRunner(), platform="linux", dry_run=False)
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await op.notify("x")
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await op.dialog("x")
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await op.screenshot("/tmp/harden-shot.png")
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await op.clipboard_read()
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await op.clipboard_write("x")
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await op.list_windows()
    with pytest.raises(UnsupportedOperationError, match="macOS"):
        await op.focus_window("Terminal")


async def test_executor_exception_is_wrapped(tmp_path: Path):
    executor = FakeExecutor(explode=ValueError("bad"))
    op = ComputerOperator(
        AllowAll(), tool_executor=executor, probe_runner=FakeRunner(), platform="darwin", dry_run=False
    )
    with pytest.raises(ComputerError, match="file.write failed: ValueError: bad"):
        await op.file_write(tmp_path / "a.txt", "x")


def test_record_status_reflects_platform():
    probe = FakeRunner()
    assert ComputerOperator(AllowAll(), probe_runner=probe, platform="darwin").record().status == "configured"
    blocked = ComputerOperator(AllowAll(), probe_runner=probe, platform="linux").record()
    assert blocked.status == "blocked"
    assert blocked.metadata["window_management"] == "unsupported"


# --------------------------------------------------------------------------- capability probing


class SelectiveProbe:
    def __init__(self, binaries: set[str]) -> None:
        self.binaries = binaries
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], stdin: str | None, timeout: float):
        self.calls.append(list(argv))
        binary = argv[1] if len(argv) >= 2 and argv[0] == "which" else None
        return (0 if binary in self.binaries else 1, "", "")


async def test_probe_withholds_unavailable_capabilities_before_authorization():
    permissions = RecordingPermissions()
    runner = FakeRunner(stdout="clip")
    probe = SelectiveProbe({"pbpaste"})
    op = ComputerOperator(
        permissions, runner=runner, probe_runner=probe, platform="darwin", dry_run=False
    )
    assert op.capabilities == frozenset({"clipboard.read"})
    with pytest.raises(UnsupportedOperationError, match="unavailable"):
        await op.notify("x")
    assert permissions.calls == []
    assert runner.calls == []
    assert (await op.clipboard_read())["executed"] is True
    assert runner.calls[0][0] == ["pbpaste"]


async def test_every_action_is_gated_before_authorization():
    permissions = RecordingPermissions()
    op = ComputerOperator(
        permissions, capabilities=(), tool_executor=FakeExecutor(),
        runner=FakeRunner(), platform="darwin", dry_run=False,
    )
    actions = [
        op.notify("x"), op.dialog("x"), op.screenshot("/tmp/x.png"),
        op.clipboard_read(), op.clipboard_write("x"), op.file_read("/tmp/x"),
        op.file_write("/tmp/x", "y"), op.file_move("/tmp/a", "/tmp/b"),
        op.file_delete("/tmp/x"), op.list_windows(), op.focus_window("Terminal"),
    ]
    for action in actions:
        with pytest.raises(UnsupportedOperationError):
            await action
    assert permissions.calls == []


async def test_dry_run_reads_can_be_opted_in_at_construction():
    runner = FakeRunner(stdout="clip")
    op = ComputerOperator(
        AllowAll(), runner=runner, probe_runner=FakeRunner(), platform="darwin",
        dry_run=True, allow_host_reads=True,
    )
    assert (await op.clipboard_read())["executed"] is True
    assert (await op.clipboard_read(allow_host_read=False))["executed"] is False


def test_record_exposes_effective_dry_run_and_probed_capabilities():
    probe = SelectiveProbe({"osascript", "screencapture", "pbpaste", "pbcopy"})
    op = ComputerOperator(AllowAll(), probe_runner=probe, platform="darwin", dry_run=True)
    record = op.record()
    assert record.metadata["dry_run"] is True
    assert record.metadata["allow_host_reads"] is False
    assert record.capabilities == sorted({"notify", "dialog", "screenshot", "clipboard.read", "clipboard.write"})
    assert "file.read" in record.metadata["unavailable_capabilities"]
    assert ComputerOperator(
        AllowAll(), probe_runner=probe, platform="darwin", dry_run=False
    ).record().metadata["dry_run"] is False
    assert ComputerOperator(
        AllowAll(), probe_runner=SelectiveProbe(set()), platform="darwin"
    ).record().capabilities == []
