"""Static checks for the open-source release plan and its hygiene tooling.

These tests read owned planning/script files only. They do not touch Git state,
stage files, scan real credentials, or authorize any publication action.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The release tree removes the planning documents these checks read (Gate PF
# REMOVE). The published repository has no plan to verify, so the module is
# skipped there instead of failing on absent files.
pytestmark = pytest.mark.skipif(
    not (ROOT / "docs" / "GIT_OPEN_SOURCE_PLAN.md").exists(),
    reason="release planning documents are not part of this tree",
)


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def lower(relative: str) -> str:
    return read(relative).lower()


ACTIVE_PUBLICATION = re.compile(
    r"^\s*git\s+(?:push|remote\s+add|rebase|filter-repo)\b|^\s*git\s+add\s+-A\b",
    re.IGNORECASE,
)


def test_push_plan_is_obsolete_and_has_no_active_commands() -> None:
    push_plan = read("docs/GIT_PUSH_PLAN.md")
    assert push_plan.splitlines()[0].startswith("# OBSOLETE"), "line 1 must mark the plan obsolete"
    assert "GIT_OPEN_SOURCE_PLAN.md" in push_plan
    assert "No push has been performed" in push_plan
    for command in ("git push", "git remote add", "git rebase", "git filter-repo", "git add -A"):
        assert command not in push_plan.lower(), f"obsolete plan still contains {command}"
    assert "not safe to publish" in push_plan.lower()


def test_plan_has_all_gates() -> None:
    plan = lower("docs/GIT_OPEN_SOURCE_PLAN.md")
    for gate in ("gate s", "gate ts", "gate l", "gate t", "gate p", "gate d", "gate h", "gate c", "gate a"):
        assert gate in plan, f"missing {gate}"
    assert "no push happens without explicit user confirmation" in plan
    assert "prepared except for blocked owner decisions" in plan
    assert not any(ACTIVE_PUBLICATION.match(line) for line in plan.splitlines()), "plan contains an active publication command"


def test_plan_records_transcript_evidence_and_rotation_prerequisite() -> None:
    plan = read("docs/GIT_OPEN_SOURCE_PLAN.md")
    assert "9 times across 4 transcript files" in plan
    assert "oc_sk_" in plan
    assert "~/.claude/projects/**/*.jsonl" in plan
    assert "subagents" in plan
    assert "~/.claude/settings.json" in plan
    assert "LEO_OPENCODE_SESSION" in plan
    assert "not a credential" in plan.lower()
    assert "*.jsonl" in plan
    assert "transcript bundle" in plan.lower()
    assert "hard prerequisite" in plan.lower()
    assert "shoaibyazdani" in plan


def test_gate_t_is_honest_and_checks_every_js_file() -> None:
    plan = read("docs/GIT_OPEN_SOURCE_PLAN.md")
    assert "MODE=worktree" in plan
    assert "working tree only" in plan
    assert "must never be cited as such" in plan
    assert "commit hash tested" in plan
    assert "node --check" in plan
    for js in (
        "desktop/main.js",
        "desktop/preload.js",
        "leo/interfaces/static/ui/agents-live.js",
        "leo/interfaces/static/ui/app.js",
        "leo/interfaces/static/ui/conversation/chat.js",
        "leo/interfaces/static/ui/conversation/markdown.js",
        "leo/interfaces/static/ui/onboarding.js",
    ):
        assert js in plan, f"Gate T omits {js}"


def test_gate_p_uses_explicit_path_sets_and_full_scrub_list() -> None:
    plan = read("docs/GIT_OPEN_SOURCE_PLAN.md")
    assert "git ls-files --cached --others --exclude-standard" in plan
    assert "log --all --format" in plan
    assert "Ignored-data review" in plan
    assert "757" in plan and "26 files" in plan
    for doc in (
        "docs/GIT_OPEN_SOURCE_PLAN.md",
        "docs/LEO_GAP_INVENTORY_PART1.md",
        "docs/LEO_GAP_INVENTORY_PART2.md",
        "docs/LEO_TOOL_AGENT_EXPANSION_CONTRACT.md",
        "handoff_2026-11-18.md",
        "Master_plam.md",
        "DECISIONS.md",
        "LEO_DECISIONS.md",
        "LEO_ARCHITECTURE.md",
        "LEO_BUILD_PLAN.md",
        ".leo/logs/leo.log",
    ):
        assert doc in plan, f"scrub list omits {doc}"


def test_gate_d_checks_published_docs_and_cross_references() -> None:
    plan = read("docs/GIT_OPEN_SOURCE_PLAN.md")
    assert "OBSOLETE at line 1" in plan
    assert "active publication commands" in plan
    assert "git[[:space:]]+(push|remote[[:space:]]+add|rebase|filter-repo)" in plan
    assert "screenshot OCR" in plan
    assert "research token-inventory row" in plan
    assert "research notes" in plan
    assert "skill-compatibility" in plan
    assert "133 PNGs" in plan


def test_scanner_covers_all_secret_families_and_scopes() -> None:
    scanner = lower("scripts/check-secrets.py")
    for needle in (
        "sk-proj-",
        "sk_live_",
        "aiza",
        "github_pat_",
        "jwt",
        "hooks\\.slack\\.com/services",
        "discord webhook",
        "webhook url with token",
        "long hex",
        "reviewed_allowlist",
        "cat-file --batch-all-objects",
        "fsck",
        "--unreachable",
        "commit",
        "tag",
        "--transcripts",
        'rglob("*.jsonl")',
        "leo_secret_transcript_dirs",
        "subagents",
    ):
        assert needle in scanner, f"scanner missing {needle}"
    assert "allow_words" not in scanner
    assert "is_fixture" not in scanner
    assert 'startswith("tests/")' not in scanner


def test_hook_preserves_existing_hook() -> None:
    hook = read("scripts/install-git-hooks.sh")
    assert "cp -p" in hook
    assert "backup" in hook.lower()
    assert "date -u" in hook
    assert 'read -r backup' in hook
    assert 'python3 "$(git rev-parse --show-toplevel)/scripts/check-secrets.py"' in hook
    assert "mv \"$tmp\" \"$hook\"" in hook


def test_license_inventory_and_build_gate() -> None:
    plan = read("docs/GIT_OPEN_SOURCE_PLAN.md")
    assert "idna 3.20" in plan
    assert "BSD-3-Clause" in plan
    assert "hatchling>=1.25" in plan or "hatchling >=1.25" in plan
    assert 'license = "MIT"' in plan
    assert 'license-files = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]' in plan
    assert "python3 -m build" in plan
    assert "wheel and one sdist" in plan
    assert "THIRD_PARTY_NOTICES.md" in plan
    assert "upstream LICENSE" in plan


def test_backup_clone_and_release_log_requirements() -> None:
    plan = read("docs/GIT_OPEN_SOURCE_PLAN.md")
    assert "non-`/tmp`" in plan or "non-`/tmp" in plan
    assert "SHA256SUMS" in plan
    assert "No untracked file is deleted" in plan
    assert "git config user.name" in plan
    assert "pre_rewrite_head" in plan and "post_rewrite_head" in plan
    assert "never rewrite published history" in plan.lower()
    assert "git remote -v` is empty" in plan
    assert "credential-free" in plan
    assert "Transcript scan (before:" in plan
    assert "Credential account from gh auth status" in plan
    assert "All-object scan" in plan
    assert "Staging review" in plan
    assert "Explicit user confirmation of publication" in plan
