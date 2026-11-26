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
