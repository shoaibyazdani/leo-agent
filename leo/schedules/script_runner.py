"""Restricted scheduled-script execution.

A scheduled script is an *operator-provisioned executable*, not model-authored
code and not a sandbox. It is opt-in (``automation.allow_scripts``), must
resolve inside ``automation.scripts_root``, is invoked with an argv array and
never through a shell, and runs with a minimal environment, a timeout and a
bounded stdout/stderr capture.

This module owns only process execution. Delivery, evidence and agent hand-off
live in :mod:`leo.schedules.execution`.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# A deliberately small environment: no inherited API keys, tokens or operator
# shell state. A script that needs a credential must be provisioned with one
# explicitly by the operator, never by inheritance from the LEO process.
_MINIMAL_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
}
_READ_CHUNK = 65_536


class ScriptExecutionError(RuntimeError):
    """A stable, safe script refusal or failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ScriptResult:
    """The bounded outcome of one script execution."""

    argv: list[str]
    cwd: str | None
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool
    duration_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
            "duration_ms": self.duration_ms,
        }


def _bounded_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


async def _read_capped(
    stream: asyncio.StreamReader | None, cap: int
) -> tuple[bytes, bool]:
    """Read at most ``cap`` bytes, draining the rest so the child never blocks."""
    if stream is None:
        return b"", False
    kept = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            break
        room = cap - len(kept)
        if room > 0:
            kept.extend(chunk[:room])
            if len(chunk) > room:
                truncated = True
        else:
            truncated = True
    return bytes(kept), truncated


class ScriptRunner:
    """Allowlisted, shell-free script execution under one root."""

    def __init__(
        self,
        *,
        scripts_root: str | Path | None = None,
        allow_scripts: bool = False,
        timeout_seconds: int = 60,
        max_output_bytes: int = 200_000,
        allowed_workdir_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self.scripts_root = (
            Path(scripts_root).expanduser().resolve()
            if scripts_root is not None
            else None
        )
        self.allow_scripts = bool(allow_scripts)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_output_bytes = max(1, int(max_output_bytes))
        self.allowed_workdir_roots = tuple(
            Path(root).expanduser().resolve()
            for root in (allowed_workdir_roots or ())
        )

    # ------------------------------------------------------------------
    # Resolution / validation
    # ------------------------------------------------------------------
    def resolve(self, script: Any) -> Path:
        """Validate the path and return the resolved executable.

        Rejections are stable codes so a card or failure receipt can say *why*
        without leaking the path beyond what the operator already configured.
        """
        if not self.allow_scripts:
            raise ScriptExecutionError(
                "scripts_disabled",
                "Scheduled script execution is disabled; the operator must set "
                "automation.allow_scripts=true.",
            )
        if not isinstance(script, (str, Path)) or not str(script).strip():
            raise ScriptExecutionError("invalid_script", "script must name a file.")
        raw = str(script)
        if "\x00" in raw:
            raise ScriptExecutionError("invalid_script", "script contains a NUL byte.")
        if any(char.isspace() for char in raw):
            raise ScriptExecutionError(
                "script_command_line",
                "script must name a file, not a shell command line.",
            )
        candidate = Path(raw).expanduser()
        if self.scripts_root is None:
            raise ScriptExecutionError(
                "script_not_allowed",
                "No automation.scripts_root is configured.",
            )
        if not candidate.is_absolute():
            candidate = self.scripts_root / candidate
        # Check the link *before* resolving: ``resolve`` follows it, so checking
        # afterwards could never see the link itself.
        if candidate.is_symlink():
            raise ScriptExecutionError(
                "script_symlink", "symbolic-link scripts are not allowed."
            )
        candidate = candidate.resolve(strict=False)
        root = self.scripts_root
        if candidate != root and root not in candidate.parents:
            raise ScriptExecutionError(
                "script_outside_allowed_root",
                "script resolves outside the allowlisted scripts root.",
            )
        if not candidate.is_file():
            raise ScriptExecutionError("script_not_found", "script file does not exist.")
        if not os.access(candidate, os.X_OK):
            raise ScriptExecutionError(
                "script_not_executable", "script is not executable."
            )
        return candidate

    def validate_args(self, args: Any) -> list[str]:
        if args is None:
            return []
        if not isinstance(args, (list, tuple)):
            raise ScriptExecutionError("invalid_script_args", "script_args must be a list.")
        cleaned: list[str] = []
        for item in args:
            if not isinstance(item, str):
                raise ScriptExecutionError(
                    "invalid_script_args", "script_args entries must be text."
                )
            if "\x00" in item:
                raise ScriptExecutionError(
                    "invalid_script_args", "script_args contains a NUL byte."
                )
            cleaned.append(item)
        if len(cleaned) > 128:
            raise ScriptExecutionError(
                "invalid_script_args", "script_args has too many entries."
            )
        return cleaned

    def validate_workdir(self, workdir: Any) -> Path | None:
        if workdir in (None, ""):
            return None
        if not isinstance(workdir, (str, Path)):
            raise ScriptExecutionError("invalid_workdir", "workdir must be a path.")
        raw = str(workdir)
        if "\x00" in raw:
            raise ScriptExecutionError("invalid_workdir", "workdir contains a NUL byte.")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise ScriptExecutionError("invalid_workdir", "workdir must be absolute.")
        resolved = candidate.resolve(strict=False)
        if not resolved.is_dir():
            raise ScriptExecutionError(
                "invalid_workdir", "workdir does not exist or is not a directory."
            )
        if self.allowed_workdir_roots:
            if not any(
                resolved == root or root in resolved.parents
                for root in self.allowed_workdir_roots
            ):
                raise ScriptExecutionError(
                    "workdir_outside_allowed_roots",
                    "workdir is outside LEO's allowed roots.",
                )
        return resolved

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def run(
        self,
        script: Any,
        args: Sequence[str] | None = None,
        *,
        workdir: str | Path | None = None,
    ) -> ScriptResult:
        executable = self.resolve(script)
        argv = [str(executable), *self.validate_args(args)]
        cwd = self.validate_workdir(workdir)
        if cwd is None and self.scripts_root is not None:
            cwd = self.scripts_root
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd) if cwd is not None else None,
                env=dict(_MINIMAL_ENV),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise ScriptExecutionError(
                "script_start_failed", f"could not start script: {exc}"
            ) from exc

        try:
            stdout, stderr, _ = await asyncio.wait_for(
                asyncio.gather(
                    _read_capped(process.stdout, self.max_output_bytes),
                    _read_capped(process.stderr, self.max_output_bytes),
                    process.wait(),
                ),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            raise ScriptExecutionError(
                "script_timeout",
                f"script exceeded the {self.timeout_seconds}s timeout.",
            ) from exc
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

        out_raw, out_cut = stdout
        err_raw, err_cut = stderr
        return ScriptResult(
            argv=argv,
            cwd=str(cwd) if cwd is not None else None,
            exit_code=int(process.returncode or 0),
            stdout=_bounded_text(out_raw),
            stderr=_bounded_text(err_raw),
            truncated=bool(out_cut or err_cut),
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )


__all__ = [
    "ScriptExecutionError",
    "ScriptResult",
    "ScriptRunner",
]
