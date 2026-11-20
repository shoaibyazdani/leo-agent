"""Process-level, fresh-home E2E gate: CLI → mock provider → persistence.

Upstream Hermes and OpenClaw were both verified end-to-end at process level
against loopback mock providers in the 2026-11-18 research round. This test
closes the equivalent gap for LEO: every step runs in a *new process* with an
isolated config, database and HOME, and no external services.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(tmp_path: Path) -> dict[str, str]:
    # A deliberately minimal environment: no host credentials or model gateway.
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "LEO_CONFIG": str(tmp_path / "leo.yaml"),
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
    }


def _cli(tmp_path: Path, *args: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "leo.interfaces.cli", *args],
        cwd=ROOT,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, f"CLI {args} failed: {completed.stderr[-2000:]}"
    return completed.stdout


def test_fresh_home_cli_chat_memory_and_tasks_survive_restart(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "workspaces").mkdir()
    (tmp_path / "exports").mkdir()
    (tmp_path / "leo.yaml").write_text(
        "\n".join([
            "models:",
            "  default: mock",
            "  fallback: mock",
            "  providers:",
            "    mock:",
            "      kind: mock",
            "      model: mock-1",
            "paths:",
            f"  data: {tmp_path / 'data'}",
            f"  logs: {tmp_path / 'logs'}",
            f"  workspaces: {tmp_path / 'workspaces'}",
            f"  exports: {tmp_path / 'exports'}",
        ]),
        encoding="utf-8",
    )

    health = _cli(tmp_path, "doctor")
    assert '"status": "ok"' in health
    assert str(tmp_path / "data" / "leo.db") in health

    answer = _cli(tmp_path, "ask", "Say hello using only offline tools.")
    assert '"provider": "mock"' in answer
    assert '"content"' in answer

    remembered = _cli(tmp_path, "memory", "add", "fresh-home-fact-42")
    assert "fresh-home-fact-42" in remembered

    task = _cli(tmp_path, "task", "create", "write release notes")
    assert '"status": "created"' in task

    # New process, same durable state: no in-memory cache can make this pass.
    found = _cli(tmp_path, "memory", "search", "fresh-home-fact-42")
    assert "fresh-home-fact-42" in found
    tasks = _cli(tmp_path, "task", "list")
    assert "write release notes" in tasks
