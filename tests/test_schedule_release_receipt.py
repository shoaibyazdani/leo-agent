"""The release receipt is LEO-owned, reproducible and honest about limits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release import check_schedule_release as release  # noqa: E402


def test_recorded_receipt_validates_against_its_junit() -> None:
    assert release.validate_receipt() == []
    receipt = json.loads(release.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["owner"] == "leo"
    assert receipt["evidence_source"] == "leo_local_test_run"
    assert receipt["upstream_test_evidence"] is False
    assert ".venv/bin/python -m pytest" in receipt["command"]
    assert receipt["summary"] == release.junit_summary(ROOT / receipt["junit_xml"])
    assert receipt["junit_sha256"] == release.sha256_file(ROOT / receipt["junit_xml"])


def test_unverified_surfaces_are_not_promoted_to_verified() -> None:
    receipt = json.loads(release.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    checks = receipt["checks"]
    assert set(checks) == set(release.REQUIRED_CHECKS)
    for name in ("migration_drill", "backup_restore", "telegram_live_smoke"):
        assert checks[name]["status"] == "unverified"
        assert checks[name]["reason"]
    assert checks["playwright_dom"]["status"] == "partial"
    assert "chromium is absent" in checks["playwright_dom"]["reason"].lower()
    assert checks["telegram_mock_transport"]["status"] == "verified"
    for name, entry in checks.items():
        assert entry["status"] in {"verified", "partial", "unverified"}, name
        assert entry["reason"], name
        for evidence in entry["evidence"]:
            assert (ROOT / evidence).is_file(), evidence


def test_validator_rejects_forged_or_upstream_claimed_receipts(tmp_path) -> None:
    original = json.loads(release.DEFAULT_RECEIPT.read_text(encoding="utf-8"))

    forged = dict(original, upstream_test_evidence=True)
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    blockers = release.validate_receipt(path)
    assert any("source-project test evidence" in item for item in blockers)

    wrong_hash = json.loads(json.dumps(original))
    wrong_hash["junit_sha256"] = "0" * 64
    path.write_text(json.dumps(wrong_hash), encoding="utf-8")
    assert any("JUnit hash mismatch" in item for item in release.validate_receipt(path))

    incomplete = json.loads(json.dumps(original))
    incomplete["checks"].pop("split_gateway_two_process")
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    assert any("split_gateway_two_process" in item for item in release.validate_receipt(path))
