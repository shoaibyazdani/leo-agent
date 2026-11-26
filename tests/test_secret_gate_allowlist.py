"""Focused contract for the reviewed secret-gate exceptions.

The gate itself is ``python3 scripts/check-secrets.py --all``; these checks
keep the new exceptions explicit and pattern-restricted so they cannot become
blanket approvals for real credentials.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _scanner():
    path = ROOT / "scripts" / "check-secrets.py"
    spec = importlib.util.spec_from_file_location("leo_check_secrets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewed_entries_are_explicit_and_pattern_restricted() -> None:
    scanner = _scanner()
    assert scanner.is_reviewed("tests/test_model_presets.py", 64, "openai project key")
    assert not scanner.is_reviewed("tests/test_model_presets.py", 64, "private key")
    assert scanner.is_reviewed("object:commit:1d0cba3cf52c:msg", 21, "long hex")
    assert not scanner.is_reviewed("object:commit:1d0cba3cf52c:msg", 21, "openai key")
    for entry, reasons in scanner.REVIEWED_ALLOWLIST.items():
        for line_spec, reason in reasons.items():
            assert reason.strip(), (entry, line_spec)


def test_dead_line_exceptions_are_removed_and_current_ones_are_explicit() -> None:
    """Exceptions must track real fixtures, not accumulate line-number drift."""
    scanner = _scanner()
    dead = [
        ("tests/channels/test_credential_enrollment.py", "153", "assigned secret"),
        ("tests/test_enrollment.py", "203", "discord webhook"),
        ("tests/test_harden_config.py", "185", "anthropic key"),
        ("tests/test_harden_config.py", "196", "openai key"),
        ("tests/test_harden_config.py", "313", "assigned secret"),
        ("tests/test_harden_events.py", "236", "assigned secret"),
        ("tests/test_integrations.py", "209", "assigned secret"),
        ("tests/test_tool_agent_integration.py", "387", "assigned secret"),
        ("tests/test_tools_context.py", "88", "bearer token"),
        ("tests/test_tools_run.py", "213", "bearer token"),
    ]
    for path, line, label in dead:
        assert line not in scanner.REVIEWED_ALLOWLIST.get(path, {}), (path, line)
        assert not scanner.is_reviewed(path, int(line), label)

    current = [
        ("tests/test_agents_automation_operator.py", 41, "assigned secret"),
        ("tests/test_tools_http.py", 669, "assigned secret"),
        ("tests/test_tools_http.py", 686, "openai key"),
        ("tests/test_tools_http.py", 686, "assigned secret"),
        ("tests/test_tools_http.py", 1373, "openai key"),
    ]
    for path, line, label in current:
        assert scanner.is_reviewed(path, line, label)
    # A synthetic fixture line is not a blanket exception for the file.
    assert not scanner.is_reviewed("tests/test_tools_http.py", 99999, "openai key")
