"""Phase 6: restricted scheduled-script execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.core.app import LeoApp
from leo.schedules.clock import FrozenClock
from leo.schedules.execution import (
    WORKER_SCRIPT,
    ExecutionError,
    ScheduleExecution,
    select_adapter,
)
from leo.schedules.script_runner import ScriptExecutionError, ScriptRunner
from leo.schedules.store import ScheduleStore


def _script(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def scripts_root(tmp_path):
    root = tmp_path / "scripts"
    root.mkdir()
    return root


def _runner(scripts_root: Path, **overrides) -> ScriptRunner:
    values = {
        "scripts_root": scripts_root,
        "allow_scripts": True,
        "timeout_seconds": 5,
        "max_output_bytes": 10_000,
    }
    values.update(overrides)
    return ScriptRunner(**values)


async def _execution(tmp_path, scripts_root, **overrides) -> ScheduleExecution:
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.automation.allow_scripts = True
    values = {
        "settings": settings,
        "store": ScheduleStore(clock=FrozenClock("2026-11-26T10:00:00+00:00")),
        "clock": FrozenClock("2026-11-26T10:00:00+00:00"),
        "scripts_root": scripts_root,
        "allowed_workdir_roots": [tmp_path],
    }
    values.update(overrides)
    return ScheduleExecution(**values)


def test_allowlist_root_escape_symlink_and_nul(scripts_root, tmp_path) -> None:
    runner = _runner(scripts_root)
    outside = tmp_path / "escape.sh"
    outside.write_text("#!/bin/sh\necho escaped\n", encoding="utf-8")
    outside.chmod(0o755)

    with pytest.raises(ScriptExecutionError) as caught:
        runner.resolve("../escape.sh")
    assert caught.value.code == "script_outside_allowed_root"

    with pytest.raises(ScriptExecutionError) as caught:
        runner.resolve(str(outside))
    assert caught.value.code == "script_outside_allowed_root"

    real = _script(scripts_root, "real.sh", "#!/bin/sh\necho ok\n")
    link = scripts_root / "link.sh"
    link.symlink_to(real)
    with pytest.raises(ScriptExecutionError) as caught:
        runner.resolve("link.sh")
    assert caught.value.code == "script_symlink"

    with pytest.raises(ScriptExecutionError) as caught:
        runner.resolve("bad\x00.sh")
    assert caught.value.code == "invalid_script"

    with pytest.raises(ScriptExecutionError) as caught:
        runner.validate_args(["ok", "bad\x00arg"])
    assert caught.value.code == "invalid_script_args"

    with pytest.raises(ScriptExecutionError) as caught:
        runner.validate_args("not-a-list")
    assert caught.value.code == "invalid_script_args"


def test_scripts_are_opt_in(scripts_root) -> None:
    runner = _runner(scripts_root, allow_scripts=False)
    _script(scripts_root, "job.sh", "#!/bin/sh\necho hi\n")
    with pytest.raises(ScriptExecutionError) as caught:
        runner.resolve("job.sh")
    assert caught.value.code == "scripts_disabled"


async def test_args_are_an_argv_array_without_a_shell(scripts_root) -> None:
    _script(scripts_root, "args.sh", '#!/bin/sh\nprintf \'%s\' "$1"\n')
    result = await _runner(scripts_root).run("args.sh", ["; echo hacked"])
    assert result.stdout == "; echo hacked"
    assert result.exit_code == 0
    assert result.argv[0].endswith("args.sh")


async def test_minimal_environment_is_not_inherited(scripts_root, monkeypatch) -> None:
    monkeypatch.setenv("LEO_SCRIPT_SECRET", "must-not-leak")
    _script(
        scripts_root,
        "env.sh",
        '#!/bin/sh\nprintf \'%s\' "${LEO_SCRIPT_SECRET:-absent}"\n',
    )
    result = await _runner(scripts_root).run("env.sh")
    assert result.stdout == "absent"


async def test_nonzero_exit_timeout_and_byte_cap(scripts_root, tmp_path) -> None:
    _script(scripts_root, "fail.sh", "#!/bin/sh\necho boom >&2\nexit 3\n")
    failed = await _runner(scripts_root).run("fail.sh")
    assert failed.exit_code == 3 and "boom" in failed.stderr

    _script(scripts_root, "slow.sh", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(ScriptExecutionError) as caught:
        await _runner(scripts_root, timeout_seconds=1).run("slow.sh")
    assert caught.value.code == "script_timeout"

    _script(scripts_root, "big.sh", "#!/bin/sh\nprintf '%05000d' 0\n")
    capped = await _runner(scripts_root, max_output_bytes=100).run("big.sh")
    assert capped.truncated is True
    assert len(capped.stdout) <= 100

    execution = await _execution(tmp_path, scripts_root)
    task = SimpleNamespace(id="t1", objective="job", metadata={})
    with pytest.raises(ExecutionError) as caught:
        await execution.run_script(
            task=task,
            context={
                "schedule_payload": {
                    "name": "job",
                    "execution": {"script": "fail.sh", "no_agent": True},
                }
            },
        )
    assert caught.value.code == "script_failed"


async def test_no_agent_and_agent_output_modes(scripts_root, tmp_path) -> None:
    payload_base = {
        "name": "job",
        "objective": "job",
        "execution": {"script": "args.sh", "script_args": ["hello"]},
    }
    _script(scripts_root, "args.sh", '#!/bin/sh\nprintf \'%s\' "$1"\n')
    assert select_adapter(payload_base) == "script"
    assert select_adapter({**payload_base, "execution": {"script": "x.sh", "no_agent": True}}) == "script"
    assert select_adapter({"name": "job", "text": "drink water"}) == "reminder"
    assert select_adapter({"name": "job", "execution": {}}) == "agent"

    execution = await _execution(tmp_path, scripts_root)
    task = SimpleNamespace(id="t1", objective="job", metadata={"schedule_id": "sch_1"})
    context = {
        "schedule_payload": payload_base,
        "objective": "job",
        "schedule_id": "sch_1",
    }
    outcome = await execution.run_script(task=task, context=context)
    assert outcome["script"]["stdout"] == "hello"
    assert outcome["execution"].get("no_agent", False) is False

    # no_agent=True: the stdout is the whole result.
    class FakeEngine:
        def __init__(self) -> None:
            self.completed: list[tuple[str, object]] = []

        async def complete(self, task_id, result, evidence, note=None):
            self.completed.append((task_id, result))
            return None

        async def fail(self, task_id, error):
            raise AssertionError(error)

    app = object.__new__(LeoApp)
    app.schedule_execution = execution
    app.events = SimpleNamespace(emit=lambda *a, **k: asyncio.sleep(0))

    async def _heartbeat(*_args, **_kwargs):
        return None

    app._start_heartbeat = _heartbeat

    async def _failed(*_args, **_kwargs):
        return None

    app._schedule_worker_failed = _failed

    no_agent_payload = {
        **payload_base,
        "execution": {"script": "args.sh", "script_args": ["hi"], "no_agent": True},
    }
    engine = FakeEngine()
    result = await LeoApp._schedule_script_worker(
        app,
        SimpleNamespace(id="t2", objective="job", metadata={"schedule_id": "sch_1"}),
        engine,
        {"schedule_payload": no_agent_payload, "objective": "job", "schedule_id": "sch_1"},
    )
    assert result["output"] == "hi"
    assert engine.completed[0][1]["stdout"] == "hi"

    # no_agent=False: stdout is fed to the agent objective.
    captured: dict[str, object] = {}

    async def _agent(task, engine, context, *, objective=None, extra_objective=None):
        captured["extra"] = extra_objective
        return {"output": "agent answer"}

    app._run_schedule_agent = _agent
    engine = FakeEngine()
    result = await LeoApp._schedule_script_worker(
        app,
        SimpleNamespace(id="t3", objective="job", metadata={"schedule_id": "sch_1"}),
        engine,
        {"schedule_payload": payload_base, "objective": "job", "schedule_id": "sch_1"},
    )
    assert result["output"] == "agent answer"
    assert "hello" in str(captured["extra"])
    assert engine.completed == []
    assert WORKER_SCRIPT == "schedule_script"
