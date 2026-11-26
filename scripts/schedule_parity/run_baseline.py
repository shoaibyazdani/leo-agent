#!/usr/bin/env python3
"""Record the LEO pre-change scheduling baseline without running oracle code."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "18d8e6202e8fd643a96b6e8ea916b176debf99f3"
EXPECTED_LOCK_SHA256 = "fc2422bbd18dbf9dcb69c399b5fdb4670390df98ac8901c21514879283692045"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "tests/fixtures/schedule_parity/baseline"
DEFAULT_ALLOWLIST = REPO_ROOT / "tests/fixtures/schedule_parity/baseline_failures.json"
BASELINE_TESTS = [
    "tests/test_tasks.py",
    "tests/test_tools_automation.py",
    "tests/test_tools_approval.py",
    "tests/test_agents_automation_operator.py",
    "tests/test_scheduler_delivery.py",
    "tests/test_reminder_delivery.py",
    "tests/test_api.py",
    "tests/test_tui.py",
    "tests/test_gateway_taps.py",
    "tests/test_gateway_split.py",
    "tests/test_ui_contracts.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def load_allowlist(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("allowed_failures"), list):
        raise ValueError("allowlist must contain an allowed_failures list")
    allowed: dict[str, str] = {}
    for entry in payload["allowed_failures"]:
        if not isinstance(entry, dict):
            raise ValueError("each allowed failure must be an object")
        test_id = entry.get("test_id")
        reason = entry.get("reason")
        if not isinstance(test_id, str) or not test_id or not isinstance(reason, str) or not reason:
            raise ValueError("each allowed failure needs a non-empty test_id and reason")
        if test_id in allowed:
            raise ValueError(f"duplicate allowlisted test id: {test_id}")
        allowed[test_id] = reason
    return allowed


def parse_junit(path: Path) -> list[dict[str, Any]]:
    """Return stable test identities and outcomes from pytest's JUnit XML."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    records: dict[str, dict[str, Any]] = {}
    for suite in suites:
        for case in suite.findall("testcase"):
            classname = str(case.get("classname") or "")
            name = str(case.get("name") or "")
            test_id = f"{classname}::{name}" if classname else name
            if not test_id or test_id in records:
                raise ValueError(f"missing or duplicate JUnit test id: {test_id!r}")
            outcome = "passed"
            detail = None
            for tag, value in (
                ("failure", "failed"),
                ("error", "error"),
                ("skipped", "skipped"),
            ):
                child = case.find(tag)
                if child is not None:
                    outcome = value
                    detail = child.get("message") or (child.text or "").strip() or None
                    break
            else:
                xfailure = case.find("xfailure")
                if xfailure is not None:
                    outcome = "xfailed"
                    detail = xfailure.get("message") or (xfailure.text or "").strip() or None
            records[test_id] = {
                "test_id": test_id,
                "file": case.get("file"),
                "line": int(case.get("line")) if case.get("line") else None,
                "outcome": outcome,
                "detail": detail,
            }
    declared = sum(int(suite.get("tests") or 0) for suite in suites)
    if declared != len(records):
        raise ValueError(f"JUnit suite count {declared} does not match {len(records)} test cases")
    return [records[key] for key in sorted(records)]


def evaluate_baseline(
    tests: list[dict[str, Any]], allowed: dict[str, str]
) -> tuple[dict[str, Any], list[str]]:
    """Compare outcomes with the named-failure allowlist; return summary and blockers."""
    counts = Counter(record["outcome"] for record in tests)
    failed_ids = {
        record["test_id"] for record in tests if record["outcome"] in {"failed", "error"}
    }
    unexpected = sorted(failed_ids - set(allowed))
    allowlist_misses = sorted(set(allowed) - failed_ids)
    summary = {
        "total": len(tests),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["error"],
        "skipped": counts["skipped"],
        "xfailed": counts["xfailed"],
        "allowed_failures": sorted(failed_ids & set(allowed)),
        "unexpected_failures": unexpected,
        "allowlist_entries_not_observed": allowlist_misses,
    }
    blockers = list(unexpected)
    if allowlist_misses:
        blockers.append("allowlist names failures that were not observed: " + ", ".join(allowlist_misses))
    return summary, blockers


def stable_receipt_digest(receipt: dict[str, Any]) -> str:
    stable = {
        key: receipt[key]
        for key in (
            "schema_version",
            "phase",
            "commit",
            "lock_sha256",
            "python",
            "platform",
            "command",
            "tests",
        )
    }
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()


def build_receipt(
    *,
    junit_path: Path,
    output_dir: Path,
    allowed: dict[str, str],
    returncode: int,
) -> dict[str, Any]:
    tests = parse_junit(junit_path)
    summary, comparison_blockers = evaluate_baseline(tests, allowed)
    command = [
        ".venv/bin/python",
        "-m",
        "pytest",
        *BASELINE_TESTS,
        f"--junitxml={junit_path.relative_to(REPO_ROOT)}",
    ]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "phase": 0,
        "commit": _git("rev-parse", "HEAD"),
        "lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "command": command,
        "junit_xml": str(junit_path.relative_to(REPO_ROOT)),
        "junit_sha256": sha256_file(junit_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "tracked_changes": _git("diff", "--name-only", "HEAD").splitlines(),
        "tests": tests,
        "summary": summary,
        "pytest_returncode": returncode,
        "comparison_blockers": comparison_blockers,
        "status": "passed" if returncode == 0 and not comparison_blockers else "failed",
        "output_dir": str(output_dir),
    }
    receipt["receipt_sha256"] = stable_receipt_digest(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    junit_path = output_dir / "junit.xml"
    receipt_path = output_dir / "receipt.json"

    commit = _git("rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        print(f"baseline requires commit {EXPECTED_COMMIT}, found {commit}", file=sys.stderr)
        return 2
    lock_hash = sha256_file(REPO_ROOT / "uv.lock")
    if lock_hash != EXPECTED_LOCK_SHA256:
        print(f"baseline lock hash changed: {lock_hash}", file=sys.stderr)
        return 2
    allowed = load_allowlist(args.allowlist.resolve())
    command = [
        sys.executable,
        "-m",
        "pytest",
        *BASELINE_TESTS,
        f"--junitxml={junit_path}",
    ]
    print("Running LEO baseline: " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if not junit_path.exists():
        print("pytest did not produce JUnit XML", file=sys.stderr)
        return completed.returncode or 2
    receipt = build_receipt(
        junit_path=junit_path,
        output_dir=output_dir,
        allowed=allowed,
        returncode=completed.returncode,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt["summary"], indent=2))
    print(f"Receipt: {receipt_path}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
