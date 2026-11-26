#!/usr/bin/env python3
"""Validate the LEO-owned proactive-scheduling release receipt.

The receipt may only cite LEO commands, LEO tests and LEO-produced JUnit XML.
It never treats source-project tests or CI results as evidence, and checks that
could not be performed locally must stay explicitly partial/unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "scripts/release/evidence/receipt.json"
DEFAULT_JUNIT = ROOT / "scripts/release/evidence/junit.xml"
RECEIPT_COMMAND = (
    ".venv/bin/python -m pytest "
    "--junitxml=scripts/release/evidence/junit.xml"
)
REQUIRED_CHECKS = (
    "migration_drill",
    "backup_restore",
    "tui_pty",
    "playwright_dom",
    "telegram_mock_transport",
    "telegram_live_smoke",
    "split_gateway_two_process",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def junit_summary(path: Path) -> dict[str, int]:
    """Return counts from pytest's own XML, never from a prose claim."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    return {
        "total": len(cases),
        "passed": sum(
            1 for case in cases
            if not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
        ),
        "failed": sum(1 for case in cases if case.find("failure") is not None),
        "errors": sum(1 for case in cases if case.find("error") is not None),
        "skipped": sum(1 for case in cases if case.find("skipped") is not None),
    }


def _check(status: str, reason: str, evidence: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "reason": reason, "evidence": evidence or []}


def build_receipt(junit_path: Path) -> dict[str, Any]:
    """Build an honest receipt from a local full-suite JUnit run."""
    summary = junit_summary(junit_path)
    relative = junit_path.resolve().relative_to(ROOT).as_posix()
    return {
        "schema_version": 1,
        "owner": "leo",
        "evidence_source": "leo_local_test_run",
        "upstream_test_evidence": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": RECEIPT_COMMAND,
        "junit_xml": relative,
        "junit_sha256": sha256_file(junit_path),
        "summary": summary,
        "checks": {
            "migration_drill": _check(
                "unverified",
                "Migration unit tests ran, but no production-copy WAL drill was performed locally.",
                ["tests/test_schedule_migration.py"],
            ),
            "backup_restore": _check(
                "unverified",
                "No backup/restore drill was performed; rollback remains a documented procedure.",
                [],
            ),
            "tui_pty": _check(
                "partial",
                "TUI card projection tests ran; no interactive PTY session was driven.",
                ["tests/test_schedule_tui.py"],
            ),
            "playwright_dom": _check(
                "partial",
                "Static UI contract checks passed; Chromium is absent, so real DOM checks skipped.",
                ["tests/test_playwright_approval_cards.py"],
            ),
            "telegram_mock_transport": _check(
                "verified",
                "Offline MockTransport tests cover literal plain-text cards, chunks and callbacks.",
                ["tests/test_messaging_telegram.py"],
            ),
            "telegram_live_smoke": _check(
                "unverified",
                "No live Telegram credential or send was used; live delivery is unverified.",
                [],
            ),
            "split_gateway_two_process": _check(
                "partial",
                "Offline split-gateway tests ran; no separately deployed two-process smoke was performed.",
                ["tests/test_gateway_split.py"],
            ),
        },
        "notes": [
            "Only LEO's own pytest run produced the counts in this receipt.",
            "Source-project tests were not executed and are not evidence of LEO behavior.",
        ],
    }


def validate_receipt(path: Path = DEFAULT_RECEIPT) -> list[str]:
    """Return blockers; an empty list means the receipt is release-honest."""
    blockers: list[str] = []
    if not path.is_file():
        return [f"missing release receipt: {path}"]
    raw = path.read_text(encoding="utf-8")
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"invalid receipt JSON: {exc}"]

    if receipt.get("schema_version") != 1:
        blockers.append("unsupported receipt schema_version")
    if receipt.get("owner") != "leo" or receipt.get("evidence_source") != "leo_local_test_run":
        blockers.append("receipt is not LEO-owned")
    if receipt.get("upstream_test_evidence") is not False:
        blockers.append("receipt must not claim source-project test evidence")
    lower_raw = raw.lower()
    for phrase in ("upstream tests passed", "oracle tests passed", "upstream ci passed"):
        if phrase in lower_raw:
            blockers.append(f"forbidden evidence claim: {phrase}")
    if ".venv/bin/python -m pytest" not in str(receipt.get("command", "")):
        blockers.append("receipt does not name the LEO pytest command")

    junit = ROOT / str(receipt.get("junit_xml") or "")
    if not junit.is_file():
        blockers.append("missing JUnit XML")
    else:
        if sha256_file(junit) != receipt.get("junit_sha256"):
            blockers.append("JUnit hash mismatch")
        summary = junit_summary(junit)
        if summary != receipt.get("summary"):
            blockers.append(f"summary/JUnit mismatch: {summary} vs {receipt.get('summary')}")
        if summary["failed"] or summary["errors"]:
            blockers.append("JUnit run contains failures or errors")

    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        blockers.append("missing checks object")
    else:
        for name in REQUIRED_CHECKS:
            entry = checks.get(name)
            if not isinstance(entry, dict):
                blockers.append(f"missing check: {name}")
                continue
            status = entry.get("status")
            if status not in {"verified", "partial", "unverified"}:
                blockers.append(f"{name}: invalid status {status!r}")
            if not str(entry.get("reason") or "").strip():
                blockers.append(f"{name}: missing honesty reason")
            evidence = entry.get("evidence")
            if not isinstance(evidence, list):
                blockers.append(f"{name}: evidence must be a list")
                continue
            for item in evidence:
                if not (ROOT / str(item)).is_file():
                    blockers.append(f"{name}: missing evidence path {item}")
            if status == "verified" and not evidence:
                blockers.append(f"{name}: verified claim has no evidence")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--record", type=Path, metavar="JUNIT_XML",
                        help="write a fresh receipt from a local JUnit run")
    args = parser.parse_args()
    if args.record is not None:
        receipt = build_receipt(args.record.resolve())
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Recorded {args.receipt}")
    blockers = validate_receipt(args.receipt.resolve())
    for blocker in blockers:
        print(f"BLOCKER: {blocker}", file=sys.stderr)
    if blockers:
        return 1
    print("Release receipt is LEO-owned, reproducible and honest about unverified surfaces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
