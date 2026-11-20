"""macOS computer operator.

Desktop actions are deliberately narrow: AppleScript through ``osascript`` for
notifications/dialogs, ``screencapture`` for screenshots, ``pbcopy``/``pbpaste``
for the clipboard, and tool-executor delegation for file operations. Every
subprocess uses an argument vector (never ``shell=True``) and a timeout.

Capabilities are probed through the injected runner (``which`` queries only:
no clipboard/file content is read) and unavailable actions are rejected before
authorization. All actions are authorized through the permission manager first.
Dry-run evaluates policy and records the decision without touching the host;
clipboard/file *reads* are suppressed too unless explicitly opted in.
Window management is explicitly unsupported off macOS. No desktop action is
performed by the module's tests; a fake runner/executor is injected instead.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from ..contracts import (
    IntegrationRecord,
    PermissionManager,
    RiskLevel,
    ToolCall,
    ToolExecutor,
    new_id,
    utcnow,
)


class ComputerError(RuntimeError):
    """Computer operation failed."""


class UnsupportedOperationError(ComputerError):
    """Capability is unavailable on the current platform."""


class ComputerPermissionDenied(ComputerError):
    """Permission manager denied the action (or an approval is pending)."""

    def __init__(self, reason: str, decision: Any = None) -> None:
        self.decision = decision
        super().__init__(reason)


SubprocessRunner = Callable[[list[str], str | None, float], tuple[int, str, str]]

# Every action maps to one capability. A capability is available only after a
# probe succeeds (or when the caller injects an explicit capability set), so an
# advertised capability never means "the binary might exist".
ACTION_CAPABILITIES: dict[str, str] = {
    "computer.notify": "notify",
    "computer.dialog": "dialog",
    "computer.screenshot": "screenshot",
    "computer.clipboard.read": "clipboard.read",
    "computer.clipboard.write": "clipboard.write",
    "computer.windows.list": "windows.list",
    "computer.windows.focus": "windows.focus",
    "file.read": "file.read",
    "file.write": "file.write",
    "file.move": "file.move",
    "file.delete": "file.delete",
}

# Probes query for the binary only; they never read clipboard content or files.
CAPABILITY_PROBES: dict[str, str] = {
    "notify": "osascript",
    "dialog": "osascript",
    "screenshot": "screencapture",
    "clipboard.read": "pbpaste",
    "clipboard.write": "pbcopy",
}


def _which_runner(argv: list[str], stdin: str | None, timeout: float) -> tuple[int, str, str]:
    """Local, side-effect-free probe fallback when no runner was injected."""
    if len(argv) >= 2 and argv[0] == "which":
        return (0 if shutil.which(argv[1]) else 1, "", "")
    return (1, "", "")


def _default_runner(argv: list[str], stdin: str | None, timeout: float) -> tuple[int, str, str]:
    completed = subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _applescript(value: str) -> str:
    # Escape backslashes first, then every character AppleScript treats specially
    # inside a double-quoted string. A literal tab or newline in an ``osascript
    # -e`` argument can otherwise terminate/corrupt the generated source.
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


class ComputerOperator:
    """Local desktop abstraction with risk-classified, audited actions."""

    ACTION_RISKS: dict[str, RiskLevel] = {
        "computer.notify": RiskLevel.LOW,
        "computer.dialog": RiskLevel.MEDIUM,
        "computer.screenshot": RiskLevel.MEDIUM,
        "computer.clipboard.read": RiskLevel.LOW,
        "computer.clipboard.write": RiskLevel.MEDIUM,
        "computer.windows.list": RiskLevel.LOW,
        "computer.windows.focus": RiskLevel.MEDIUM,
        "file.read": RiskLevel.LOW,
        "file.write": RiskLevel.MEDIUM,
        "file.move": RiskLevel.MEDIUM,
        "file.delete": RiskLevel.HIGH,
    }

    def __init__(
        self,
        permissions: PermissionManager,
        *,
        actor: str = "computer",
        dry_run: bool = True,
        tool_executor: ToolExecutor | None = None,
        platform: str | None = None,
        runner: SubprocessRunner | None = None,
        timeout: float = 5.0,
        probe_runner: SubprocessRunner | None = None,
        capabilities: Iterable[str] | None = None,
        allow_host_reads: bool = False,
    ) -> None:
        self.permissions = permissions
        self.actor = actor
        self.dry_run = dry_run
        self.tool_executor = tool_executor
        self.platform = platform or sys.platform
        self.runner: SubprocessRunner = runner or _default_runner
        self.timeout = max(0.1, min(float(timeout), 30.0))
        self.probe_runner: SubprocessRunner = probe_runner or runner or _which_runner
        self.allow_host_reads = bool(allow_host_reads)
        self._probed = capabilities is None
        self.capabilities: frozenset[str] = (
            frozenset(str(cap) for cap in capabilities)
            if capabilities is not None
            else self._probe_capabilities()
        )

    @property
    def is_macos(self) -> bool:
        return self.platform == "darwin"

    def _probe_capabilities(self) -> frozenset[str]:
        """Probe host binaries; file ops require a real tool executor."""
        if not self.is_macos:
            return frozenset()
        available: set[str] = set()
        for capability, binary in CAPABILITY_PROBES.items():
            try:
                code, _, _ = self.probe_runner(["which", binary], None, self.timeout)
            except (OSError, subprocess.SubprocessError):
                continue
            if code == 0:
                available.add(capability)
        if self.tool_executor is not None:
            available.update(cap for cap in ACTION_CAPABILITIES.values() if cap.startswith("file."))
        return frozenset(available)

    def _require_capability(self, action: str) -> str:
        capability = ACTION_CAPABILITIES.get(action)
        if capability is None:
            raise ComputerError(f"unknown computer action: {action}")
        if capability in self.capabilities:
            return capability
        if not self.is_macos:
            reason = "macOS-only desktop tools are unavailable on this platform"
        elif capability.startswith("windows."):
            reason = "window management requires macOS accessibility automation and is not enabled"
        else:
            reason = "capability probe failed; the required host tool is unavailable"
        raise UnsupportedOperationError(f"{action} is unavailable: {reason}") from None

    def _host_read_allowed(self, override: bool | None) -> bool:
        """Dry-run suppresses clipboard/file reads unless explicitly allowed."""
        opt_in = self.allow_host_reads if override is None else bool(override)
        return (not self.dry_run) or opt_in

    async def _authorize(self, action: str, resource: str | None = None) -> Any:
        risk = self.ACTION_RISKS[action]
        decision = await self.permissions.authorize(self.actor, action, resource, risk)
        if not decision.allowed or decision.requires_approval:
            detail = f"approval_id={decision.approval_id}" if decision.approval_id else ""
            raise ComputerPermissionDenied(
                f"{action} not permitted: {decision.reason} {detail}".strip(), decision
            )
        return decision

    def _check_macos(self, capability: str) -> None:
        if not self.is_macos:
            raise UnsupportedOperationError(
                f"{capability} requires macOS (current platform: {self.platform}); "
                "use an execution node with macOS for desktop automation"
            )

    def _execute(self, argv: list[str], stdin: str | None = None) -> tuple[int, str, str]:
        try:
            return self.runner(list(argv), stdin, self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise ComputerError(f"command timed out after {self.timeout}s: {argv[0]}") from exc
        except OSError as exc:
            raise ComputerError(f"command could not be started: {argv[0]} ({type(exc).__name__})") from exc

    def _result(
        self,
        action: str,
        *,
        resource: str | None,
        dry_run: bool,
        decision: Any,
        **details: Any,
    ) -> dict[str, Any]:
        return {
            "action": action,
            "resource": resource,
            "risk": self.ACTION_RISKS[action].value,
            "dry_run": dry_run,
            "executed": False,
            "decision": decision.reason,
            "created_at": utcnow(),
            **details,
        }

    async def notify(self, message: str, *, title: str = "LEO") -> dict[str, Any]:
        action = "computer.notify"
        self._require_capability(action)
        decision = await self._authorize(action)
        result = self._result(action, resource=None, dry_run=self.dry_run, decision=decision, message=str(message))
        if self.dry_run:
            return result
        self._check_macos("desktop notifications")
        script = f"display notification {_applescript(str(message))} with title {_applescript(str(title))}"
        code, out, err = self._execute(["osascript", "-e", script])
        result.update(executed=True, success=code == 0, stdout=out, stderr=err, exit_code=code)
        return result

    async def dialog(
        self, message: str, *, title: str = "LEO", buttons: tuple[str, ...] = ("OK",)
    ) -> dict[str, Any]:
        action = "computer.dialog"
        self._require_capability(action)
        decision = await self._authorize(action)
        result = self._result(action, resource=None, dry_run=self.dry_run, decision=decision, message=str(message))
        if self.dry_run:
            return result
        self._check_macos("dialogs")
        labels = ", ".join(_applescript(str(b)) for b in buttons) or _applescript("OK")
        default = _applescript(str(buttons[-1])) if buttons else _applescript("OK")
        script = (
            f"display dialog {_applescript(str(message))} with title {_applescript(str(title))} "
            f"buttons {{{labels}}} default button {default}"
        )
        code, out, err = self._execute(["osascript", "-e", script])
        result.update(
            executed=True, success=code == 0, stdout=out.strip(), stderr=err,
            exit_code=code, response=out.strip(),
        )
        return result

    async def screenshot(self, path: str | Path) -> dict[str, Any]:
        action = "computer.screenshot"
        self._require_capability(action)
        target = Path(path).expanduser().resolve()
        decision = await self._authorize(action, str(target))
        result = self._result(action, resource=str(target), dry_run=self.dry_run, decision=decision, path=str(target))
        if self.dry_run:
            return result
        self._check_macos("screenshots")
        target.parent.mkdir(parents=True, exist_ok=True)
        code, out, err = self._execute(["screencapture", "-x", str(target)])
        result.update(executed=True, success=code == 0, stdout=out, stderr=err, exit_code=code)
        return result

    async def clipboard_read(
        self, *, allow_host_read: bool | None = None, allow_host_reads: bool | None = None
    ) -> dict[str, Any]:
        action = "computer.clipboard.read"
        self._require_capability(action)
        decision = await self._authorize(action)
        result = self._result(action, resource=None, dry_run=self.dry_run, decision=decision)
        opt_in = allow_host_read if allow_host_read is not None else allow_host_reads
        if not self._host_read_allowed(opt_in):
            result.update(text=None, host_read="suppressed")
            return result
        self._check_macos("clipboard access")
        code, out, err = self._execute(["pbpaste"])
        result.update(
            executed=True, success=code == 0, text=out, stdout=out, stderr=err,
            exit_code=code, host_read="allowed",
        )
        return result

    async def clipboard_write(self, text: str) -> dict[str, Any]:
        action = "computer.clipboard.write"
        self._require_capability(action)
        decision = await self._authorize(action)
        result = self._result(action, resource=None, dry_run=self.dry_run, decision=decision)
        if self.dry_run:
            return result
        self._check_macos("clipboard access")
        code, out, err = self._execute(["pbcopy"], str(text))
        result.update(executed=True, success=code == 0, stdout=out, stderr=err, exit_code=code)
        return result

    async def file_read(
        self, path: str | Path, *, allow_host_read: bool | None = None, allow_host_reads: bool | None = None
    ) -> dict[str, Any]:
        return await self._file_action(
            "file.read", "file_read", {"path": self._path(path)}, mutation=False,
            allow_host_read=allow_host_read if allow_host_read is not None else allow_host_reads,
        )

    async def file_write(self, path: str | Path, content: str) -> dict[str, Any]:
        return await self._file_action(
            "file.write", "file_write", {"path": self._path(path), "content": str(content)}
        )

    async def file_move(self, source: str | Path, destination: str | Path) -> dict[str, Any]:
        return await self._file_action(
            "file.move", "file_move", {"source": self._path(source), "destination": self._path(destination)}
        )

    async def file_delete(self, path: str | Path) -> dict[str, Any]:
        return await self._file_action("file.delete", "file_delete", {"path": self._path(path)})

    @staticmethod
    def _path(value: str | Path) -> str:
        return str(Path(value).expanduser().resolve())

    async def _file_action(
        self, action: str, tool_name: str, arguments: dict[str, Any], *, mutation: bool = True,
        allow_host_read: bool | None = None,
    ) -> dict[str, Any]:
        self._require_capability(action)
        resource = arguments.get("path") or arguments.get("source")
        decision = await self._authorize(action, resource)
        # A move touches two paths. Gating only the source would let a file inside
        # the workspace be relocated anywhere the tool executor permits, so the
        # destination must pass the same policy check before any work is done.
        destination = arguments.get("destination") if action == "file.move" else None
        destination_decision = await self._authorize(action, destination) if destination else None
        perform = (not self.dry_run) if mutation else self._host_read_allowed(allow_host_read)
        result = self._result(
            action,
            resource=resource,
            dry_run=self.dry_run,
            decision=decision,
            arguments=arguments,
            destination_resource=destination,
            destination_decision=destination_decision.reason if destination_decision else None,
        )
        if not perform:
            if not mutation:
                result["host_read"] = "suppressed"
            return result
        if self.tool_executor is None:
            raise ComputerError(f"no tool executor configured for {action}")
        call = ToolCall(
            id=new_id("call"), name=tool_name, arguments=arguments, actor=self.actor, dry_run=False
        )
        try:
            tool_result = await self.tool_executor.execute(call)
        except Exception as exc:
            raise ComputerError(f"{action} failed: {type(exc).__name__}: {exc}") from exc
        result.update(
            executed=True,
            success=tool_result.success,
            output=tool_result.output,
            error=tool_result.error,
            tool_call_id=tool_result.call_id,
        )
        if not mutation:
            result["host_read"] = "allowed"
        return result

    async def list_windows(self) -> dict[str, Any]:
        """Window enumeration requires macOS accessibility permission; disabled by default."""
        action = "computer.windows.list"
        self._require_capability(action)
        decision = await self._authorize(action)
        self._check_macos("window management")
        raise UnsupportedOperationError(
            "window enumeration requires macOS accessibility automation and is not enabled"
        ) from None

    async def focus_window(self, title: str) -> dict[str, Any]:
        """Window focus requires macOS accessibility permission; disabled by default."""
        action = "computer.windows.focus"
        self._require_capability(action)
        decision = await self._authorize(action)
        self._check_macos("window management")
        raise UnsupportedOperationError(
            "window focus requires macOS accessibility automation and is not enabled"
        ) from None

    def record(self) -> IntegrationRecord:
        available = sorted(self.capabilities)
        unverified = sorted(set(ACTION_CAPABILITIES.values()) - self.capabilities)
        return IntegrationRecord(
            name="computer",
            kind="operator",
            status="configured" if self.is_macos else "blocked",
            capabilities=available,
            verification="capability-probed" if self._probed else "declared",
            config_ref="osascript,screencapture,pbcopy,pbpaste",
            metadata={
                "platform": self.platform,
                # The effective mode of this instance, not a static declaration.
                "dry_run": self.dry_run,
                "dry_run_by_default": self.dry_run,
                "allow_host_reads": self.allow_host_reads,
                "available_capabilities": available,
                "unavailable_capabilities": unverified,
                "tool_executor": type(self.tool_executor).__name__ if self.tool_executor else None,
                "window_management": "unsupported",
                "live_verified": False,
                "reason": None if self.is_macos else "macOS-only desktop tools are unavailable",
            },
        )
