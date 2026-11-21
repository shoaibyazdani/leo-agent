"""Tests for the live agent-activity dashboard and its transcript ingest."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from leo import db
from leo.agents import activity
from leo.interfaces import api as api_module

ROOT = Path(__file__).resolve().parents[1]


def _stamp(offset_seconds: float = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()


def _write_transcript(
    directory: Path,
    name: str,
    *,
    agent_id: str,
    prompt: str,
    report: str | None = None,
    narrative: str = "I am still working on this.",
    age_seconds: float = 0,
) -> Path:
    """Create a synthetic Claude Code task transcript (data only, no execution)."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "type": "user",
            "timestamp": _stamp(age_seconds + 60),
            "agentId": agent_id,
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "api-request",
            "timestamp": _stamp(age_seconds + 50),
            "agentId": agent_id,
            "params": {"ignored": "token=do-not-read-me"},
        },
        {
            "type": "assistant",
            "timestamp": _stamp(age_seconds + 30),
            "agentId": agent_id,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": narrative}],
            },
        },
    ]
    if report is not None:
        lines.append(
            {
                "type": "assistant",
                "timestamp": _stamp(age_seconds),
                "agentId": agent_id,
                "message": {"role": "assistant", "content": [{"type": "text", "text": report}]},
            }
        )
    path = directory / name
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(path, (old, old))
    return path


@pytest.fixture
def transcript_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "transcripts"
    root.mkdir()
    monkeypatch.setenv("LEO_AGENT_TRANSCRIPT_DIR", str(root))
    activity._cache.clear()
    return root


async def test_sync_extracts_roles_status_findings_and_redaction(transcript_dir: Path) -> None:
    _write_transcript(
        transcript_dir / "session" / "tasks",
        "aaaaudit.output",
        agent_id="audit-browser",
        prompt="You are auditing the browser operator integration for credential leaks.",
        report=(
            "findings:\n"
            "- Browser adapter logs token sk-abcdefghijklmnop1234 during login\n"
            "- Second verified finding: stale cookies persisted\n"
            "fixes:\n"
            "1. Redact the browser log fields\n"
            "2. Expire stale cookies on shutdown\n"
            "test_result: 12 passed\n"
            "blocker: none"
        ),
    )
    _write_transcript(
        transcript_dir / "session2" / "tasks",
        "aaamemory.output",
        agent_id="memory-worker",
        prompt="OWNED FILES: leo/tools/memory_tools.py, tests/test_memory_tools.py\nBuild the tools.",
    )
    _write_transcript(
        transcript_dir / "session3" / "tasks",
        "aaafix.output",
        agent_id="fix-tasks",
        prompt="Fix the task engine scheduler cancellation bug.",
    )
    _write_transcript(
        transcript_dir / "session4" / "tasks",
        "aaaplanner.output",
        agent_id="planner",
        prompt="You are the planner agent.",
        report="fix_1: Added bounded dependency chains\ntest_result: 8 passed\nblocker: none",
    )

    first = await activity.sync_agent_activity()
    assert first["upserted"] == 4
    second = await activity.sync_agent_activity()
    assert second["upserted"] == 0  # idempotent by mtime/size fingerprint

    rows = {row["agent_id"]: row for row in (await activity.list_activity(limit=50))["agents"]}
    assert set(rows) == {"audit-browser", "memory-worker", "fix-tasks", "planner"}
    assert rows["audit-browser"]["role"] == "Audit: browser operator"
    assert rows["memory-worker"]["role"] == "Tools: memory control"
    assert rows["fix-tasks"]["role"] == "Fix: task engine"
    assert rows["planner"]["role"] == "Agent: planner"

    assert rows["audit-browser"]["status"] == "completed"
    assert rows["memory-worker"]["status"] == "recent"
    assert rows["fix-tasks"]["status"] == "recent"
    assert rows["memory-worker"]["status_basis"].startswith("mtime-only")
    assert rows["audit-browser"]["status_basis"] == "final-report"
    assert rows["audit-browser"]["trust"] == "owner-verified"
    assert rows["audit-browser"]["source_kind"] == "local-transcript"
    assert rows["planner"]["status"] == "completed"

    audit = rows["audit-browser"]
    assert len(audit["findings"]) == 2
    assert len(audit["fixes"]) == 2
    assert audit["test_result"] == "12 passed"
    assert audit["blocker"] == ""
    assert "token [redacted]" in audit["findings"][0]
    assert "sk-abcdefghijklmnop1234" not in json.dumps(audit, ensure_ascii=False)
    assert audit["report_text"].startswith("findings:")
    assert rows["planner"]["fixes"] == ["Added bounded dependency chains"]

    schema = {row["name"] for row in await db.fetchall("PRAGMA table_info(agent_activity)")}
    assert {"agent_id", "role", "status", "started_at", "finished_at", "prompt_excerpt",
            "findings", "fixes", "test_result", "blocker", "fingerprint"}.issubset(schema)
    assert all("token=" not in row["prompt_excerpt"] for row in rows.values())


async def test_stale_transcript_without_report_is_idle_and_refresh_updates_row(transcript_dir: Path) -> None:
    path = _write_transcript(
        transcript_dir / "old",
        "aaaold.output",
        agent_id="old-agent",
        prompt="You are diagnosing the CLI startup path.",
        age_seconds=7200,
    )
    await activity.sync_agent_activity()
    row = (await activity.list_activity())["agents"][0]
    assert row["status"] == "idle"
    assert row["role"].startswith("Verify:")

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    await activity.sync_agent_activity()
    rows = await db.fetchall("SELECT fingerprint FROM agent_activity WHERE agent_id = 'old-agent'")
    assert len(rows) == 1
    assert rows[0]["fingerprint"].startswith(str(path.stat().st_mtime_ns))


async def test_findings_fixes_scalars_and_evidence_are_preserved(transcript_dir: Path) -> None:
    report = (
        "## Gaps\n"
        "- Gap one: no audit trail for tool calls\n"
        "- Gap two: stale approvals remain pending\n"
        "## Fixes\n"
        "- Added audit rows for every tool call\n"
        "fixed the stale approval purge\n"
        "test_result: pytest 5 passed\n"
        "blocker: needs a live gateway check"
    )
    _write_transcript(
        transcript_dir,
        "aaareport.output",
        agent_id="reporter",
        prompt="Audit the approval gates in this repository.",
        report=report,
    )
    await activity.sync_agent_activity()
    row = (await activity.list_activity())["agents"][0]
    assert row["findings"] == [
        "Gap one: no audit trail for tool calls",
        "Gap two: stale approvals remain pending",
    ]
    assert row["fixes"] == ["Added audit rows for every tool call", "fixed the stale approval purge"]
    assert row["test_result"] == "pytest 5 passed"
    assert row["blocker"] == "needs a live gateway check"
    assert row["report_text"] == report


async def test_pagination_filters_and_summary_counts(transcript_dir: Path) -> None:
    for index in range(12):
        report = "findings:\n- one finding\nfixes:\n- one fix\ntest_result: 1 passed\nblocker: none" if index < 5 else None
        _write_transcript(
            transcript_dir,
            f"aaa{index:02d}.output",
            agent_id=f"agent-{index:02d}",
            prompt="Audit the browser operator." if index % 2 == 0 else "Fix the task engine.",
            report=report,
        )
    await activity.sync_agent_activity()

    page = await activity.list_activity(limit=5, offset=0)
    assert page["total"] == 12
    assert len(page["agents"]) == 5
    assert page["limit"] == 5 and page["offset"] == 0
    assert len((await activity.list_activity(limit=5, offset=10))["agents"]) == 2
    assert len((await activity.list_activity(limit=5, offset=50))["agents"]) == 0
    assert len((await activity.list_activity(status="completed"))["agents"]) == 5
    assert len((await activity.list_activity(status="recent"))["agents"]) == 7
    assert len((await activity.list_activity(status="all"))["agents"]) == 12
    assert len((await activity.list_activity(role_contains="BROWSER"))["agents"]) == 6
    assert len((await activity.list_activity(q="task engine"))["agents"]) == 6
    assert len((await activity.list_activity(q="%"))["agents"]) == 0  # LIKE wildcards escaped
    assert page["role_counts"] and "roles" in page

    summary = await activity.activity_summary()
    assert summary["total"] == 12
    assert summary["completed"] == 5
    assert summary["recent"] == 7
    assert summary["recent_window_seconds"] == 90
    assert "not a live heartbeat" in summary["provenance_note"]
    assert summary["total_findings"] == 5
    assert summary["total_fixes"] == 5


async def test_transcript_discovery_is_strictly_contained(transcript_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "aaaoutside.output"
    _write_transcript(outside.parent, outside.name, agent_id="outside-agent", prompt="Fix the CLI.")
    _write_transcript(
        transcript_dir / "nested" / "tasks",
        "aaainner.output",
        agent_id="inside-agent",
        prompt="Fix the CLI.",
    )
    link = transcript_dir / "aaalink.output"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pass
    await activity.sync_agent_activity()
    ids = [row["agent_id"] for row in (await activity.list_activity())["agents"]]
    assert ids == ["inside-agent"]
    assert str(outside) not in json.dumps(await activity.list_activity(), ensure_ascii=False)


async def test_canonical_claude_code_symlinks_are_ingested(
    transcript_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude-projects"
    target_dir = projects / "session" / "subagents"
    monkeypatch.setattr(activity, "_claude_projects_root", lambda: projects)
    _write_transcript(
        target_dir,
        "agent-linked.jsonl",
        agent_id="linked-agent",
        prompt="Audit the browser operator.",
    )
    link = transcript_dir / "aaalink.output"
    try:
        link.symlink_to(target_dir / "agent-linked.jsonl")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    await activity.sync_agent_activity()
    ids = [row["agent_id"] for row in (await activity.list_activity())["agents"]]
    assert ids == ["linked-agent"]
    assert str(target_dir) not in json.dumps(await activity.list_activity(), ensure_ascii=False)

    try:
        target_dir.chmod(0o777)  # a planted target directory is not trusted
        await activity.sync_agent_activity()
        assert (await activity.list_activity())["agents"] == []
    finally:
        target_dir.chmod(0o755)


async def test_api_routes_and_page(transcript_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEO_API_TOKEN", raising=False)
    _write_transcript(
        transcript_dir / "s" / "tasks",
        "aaaapi.output",
        agent_id="api-agent",
        prompt="Audit the browser operator.",
        report="findings:\n- api finding\nfixes:\n- api fix\ntest_result: 2 passed\nblocker: none",
    )
    transport = httpx.ASGITransport(app=api_module.api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/agent-activity?status=completed&role_contains=browser&limit=1")
        assert listed.status_code == 200
        payload = listed.json()
        assert payload["total"] == 1
        assert payload["agents"][0]["agent_id"] == "api-agent"
        assert payload["limit"] == 1
        assert payload["recent_window_seconds"] == 90
        assert payload["agents"][0]["trust"] == "owner-verified"
        assert payload["agents"][0]["source_kind"] == "local-transcript"
        bounded = await client.get("/agent-activity?limit=1000")
        assert bounded.status_code == 200 and bounded.json()["limit"] == 200
        all_status = await client.get("/agent-activity?status=all&limit=1000")
        assert all_status.status_code == 200 and all_status.json()["total"] == 1
        assert (await client.get("/agent-activity?status=bogus")).status_code == 400
        assert (await client.get("/agent-activity?offset=-1")).status_code == 400
        summary = await client.get("/agent-activity/summary")
        assert summary.status_code == 200
        assert summary.json()["completed"] == 1
        assert summary.json()["total_findings"] == 1
        assert summary.json()["total_fixes"] == 1
        page = await client.get("/agent-activity/live")
        assert page.status_code == 200
        assert "Agent Activity" in page.text
        assert str(transcript_dir) not in page.text


async def test_recent_state_uses_short_mtime_window_and_is_not_a_liveness_claim(
    transcript_dir: Path,
) -> None:
    _write_transcript(
        transcript_dir, "aaafresh.output", agent_id="fresh",
        prompt="Fix the CLI.", age_seconds=60,
    )
    _write_transcript(
        transcript_dir, "aaaaging.output", agent_id="aging",
        prompt="Fix the CLI.", age_seconds=300,
    )
    await activity.sync_agent_activity()
    rows = {row["agent_id"]: row for row in (await activity.list_activity())["agents"]}
    assert rows["fresh"]["status"] == "recent"
    assert rows["aging"]["status"] == "idle"
    assert rows["fresh"]["status_basis"].startswith("mtime-only")
    assert rows["aging"]["status_basis"].startswith("mtime-only")
    assert activity.RECENT_SECONDS == 90
    assert "not a live heartbeat" in activity.PROVENANCE_NOTE
    assert "no heartbeat or PID" in activity.STATUS_BASIS_MTIME_ONLY


async def test_untrusted_transcript_roots_and_files_are_refused(
    transcript_dir: Path, tmp_path: Path
) -> None:
    trusted = _write_transcript(
        transcript_dir / "trusted", "aaatrusted.output", agent_id="trusted", prompt="Fix the CLI."
    )
    nested = transcript_dir / "plantable"
    nested.mkdir()
    _write_transcript(nested, "aaaplanted.output", agent_id="planted", prompt="Fix the CLI.")
    assert activity._owner_trusted(trusted)
    trusted.chmod(0o666)
    assert not activity._owner_trusted(trusted)
    trusted.chmod(0o644)
    try:
        nested.chmod(0o777)
        result = await activity.sync_agent_activity()
        assert result["trusted"] is True
        ids = [row["agent_id"] for row in (await activity.list_activity())["agents"]]
        assert ids == ["trusted"]  # world-writable directory skipped
    finally:
        nested.chmod(0o755)
    try:
        transcript_dir.chmod(0o777)
        result = await activity.sync_agent_activity()
        assert result["trusted"] is False
        assert result["discovered"] == 0
    finally:
        transcript_dir.chmod(0o755)


async def test_existing_activity_table_gains_provenance_columns() -> None:
    await db.execute(
        """CREATE TABLE agent_activity (
             agent_id TEXT PRIMARY KEY, role TEXT NOT NULL, status TEXT NOT NULL,
             started_at TEXT, finished_at TEXT, last_activity_at TEXT,
             prompt_excerpt TEXT NOT NULL DEFAULT '', findings TEXT NOT NULL DEFAULT '[]',
             fixes TEXT NOT NULL DEFAULT '[]', test_result TEXT NOT NULL DEFAULT '',
             blocker TEXT NOT NULL DEFAULT '', report_text TEXT NOT NULL DEFAULT '',
             fingerprint TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)"""
    )
    await activity.list_activity()  # triggers additive migration
    columns = {row["name"] for row in await db.fetchall("PRAGMA table_info(agent_activity)")}
    assert {"status_basis", "source_kind", "trust"}.issubset(columns)


def test_static_assets_disclose_status_provenance() -> None:
    static = ROOT / "leo" / "interfaces" / "static"
    html = (static / "agents-live.html").read_text(encoding="utf-8")
    css = (static / "ui" / "agents-live.css").read_text(encoding="utf-8")
    js = (static / "ui" / "agents-live.js").read_text(encoding="utf-8")
    assert 'id="stat-recent"' in html
    assert 'value="recent"' in html
    assert 'id="detail-provenance"' in html
    assert "not a live heartbeat" in html
    assert "owner-verified" in html
    assert ".badge--recent" in css
    assert 'recent: "Recent"' in js
    assert "status_basis" in js and "source_kind" in js and "trust" in js
    assert "not a live heartbeat" in js


def test_static_assets_have_accessibility_and_reduced_motion_hooks() -> None:
    static = ROOT / "leo" / "interfaces" / "static"
    html = (static / "agents-live.html").read_text(encoding="utf-8")
    css = (static / "ui" / "agents-live.css").read_text(encoding="utf-8")
    js = (static / "ui" / "agents-live.js").read_text(encoding="utf-8")

    assert "<html lang=\"en\">" in html
    assert 'aria-label="Role filters"' in html
    assert 'aria-label="Search agents by role, finding or fix"' in html
    assert 'aria-label="Pause automatic refresh"' in html
    assert 'aria-label="Close agent details"' in html
    assert 'role="alert"' in html
    assert "id=\"empty-state\"" in html and "id=\"last-updated\"" in html
    assert "skip-link" in html and "<caption class=\"sr-only\">" in html

    for token in ("--label:", "--secondaryLabel:", "--systemBackground:",
                  "--secondarySystemBackground:", "--separator:", "--accent:"):
        assert token in css
    assert "-apple-system, BlinkMacSystemFont, \"SF Pro Text\"" in css
    assert "@media (prefers-color-scheme: dark)" in css
    assert "@media (prefers-contrast: more)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ":focus-visible" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "min-height: var(--control-height)" in css
    assert "font-size: 11px" in css

    assert "new AbortController()" in js
    assert "document.hidden" in js
    assert "REFRESH_MS = 2000" in js
    assert "textContent" in js
    assert "visibilitychange" in js
    assert "innerHTML" not in js
