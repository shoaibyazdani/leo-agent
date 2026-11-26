"""Truthfulness guards for the LEO status documents.

No network, provider calls or running server. Roster counts are read from the
local composition root so a doc cannot pin numbers that the code no longer has.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

INVENTORY = ROOT / "docs" / "CAPABILITY_INVENTORY.md"
KNOWN_ISSUES = ROOT / "LEO_KNOWN_ISSUES.md"
INTEGRATIONS = ROOT / "LEO_INTEGRATIONS.md"
SECURITY = ROOT / "LEO_SECURITY.md"
TEST_REPORT = ROOT / "docs" / "TESTING.md"
PROGRESS = ROOT / "STATUS.md"
GIT_PLAN = ROOT / "docs" / "GIT_PUSH_PLAN.md"
HANDOFF = ROOT / "handoff_2026-11-18.md"

HISTORICAL_MARKERS = (
    "historical", "snapshot", "earlier", "superseded", "not current",
    "stale", "that day's", "was 403", "records", "not asserted",
)

# Floors observed in the live composition root on 2026-11-20. If the shipped
# roster legitimately grows, docs must be updated; if it silently shrinks, these
# guards force a review instead of letting a stale claim pass. The old 93-tool,
# 34-command and 20-agent figures must never satisfy these floors.
MIN_LIVE_TOOLS = 94
MIN_LIVE_COMMANDS = 35  # currently available slash commands, not all 127 specs
MIN_LIVE_AGENTS = 24

TEST_PATH_RE = re.compile(r"(?:tests/)[A-Za-z0-9_./-]+\.py")


def _platform_tool_counts(live_tools: int) -> set[int]:
    """Tool totals a truthful doc may quote: this platform and the other one.

    The four ``computer_*`` operator tools register only on macOS, so a doc
    that states both figures is correct on either host and this test must not
    depend on the platform it runs on.
    """
    from leo.tools.operator_tools import _COMPUTER_SPECS

    other = live_tools - len(_COMPUTER_SPECS) if sys.platform == "darwin" else \
        live_tools + len(_COMPUTER_SPECS)
    return {live_tools, other}



def _live_roster_counts() -> tuple[int, int, int, int]:
    """Return (unique tools, available slash commands, all commands, agents)."""
    import asyncio

    from leo.config import ModelConfig, ModelProviderConfig, Settings
    from leo.core import LeoApp

    async def measure() -> tuple[int, int, int, int]:
        settings = Settings()
        settings.models = ModelConfig(
            default="mock",
            providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
        )
        app = LeoApp(settings)
        await app.initialize()
        try:
            names = [spec.name for spec in app.registry.list_specs()]
            agents = await app.agent_registry.list()
            catalog = app.commands.catalog(principal="user", surface="chat")
            available = sum(1 for item in catalog if item.get("available") is True)
            return len(set(names)), available, len(catalog), len(agents)
        finally:
            await app.shutdown()

    return asyncio.run(measure())


def _table_rows(text: str) -> list[list[str]]:
    """Split markdown table lines into cells, skipping headers/separators."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        if cells[0] in {"Capability", "Capability area", "Area", "Adapter"}:
            continue
        rows.append(cells)
    return rows


def test_legacy_inventory_status_rule_is_gone() -> None:
    if not INVENTORY.exists():
        pytest.skip("legacy inventory document is not part of this tree")
    text = INVENTORY.read_text(encoding="utf-8").lower()
    assert "no leo integration test exists" not in text
    assert "planned, untested" not in text
    assert "source-verified upstream" in text  # upstream evidence kept distinct


def test_implemented_inventory_rows_cite_existing_test_paths() -> None:
    if not INVENTORY.exists():
        pytest.skip("legacy inventory document is not part of this tree")
    checked = 0
    for cells in _table_rows(INVENTORY.read_text(encoding="utf-8")):
        if len(cells) < 11:
            continue  # summary/overlay tables, not capability rows
        classification, status = cells[-2], cells[-1]
        implemented = classification == "native" or bool(
            re.search(r"unit-tested|mock-tested|implemented and|tested", status, re.I)
        )
        if not implemented:
            continue
        paths = TEST_PATH_RE.findall(status)
        assert paths, f"implemented row lacks a test path: {cells[0]}"
        for path in paths:
            assert (ROOT / path).is_file(), f"cited test path does not exist: {path}"
        checked += 1
    assert checked >= 20, f"expected at least 20 implemented rows, found {checked}"


def test_model_gateway_status_is_environment_conditional() -> None:
    for path in (KNOWN_ISSUES, INTEGRATIONS, TEST_REPORT):
        lower = path.read_text(encoding="utf-8").lower()
        assert "x-opencode-session" in lower, f"{path.name}: session header missing"
        assert "leo_opencode_session" in lower, f"{path.name}: session env var missing"
        assert "degrades to mock" in lower, f"{path.name}: mock fallback missing"
        assert "live verification is therefore blocked" not in lower, path.name
        assert "live end-to-end model verification remains" not in lower, path.name
        assert "remains **blocked**" not in lower, path.name


def test_security_documents_approval_hardening() -> None:
    lower = SECURITY.read_text(encoding="utf-8").lower()
    for token in ("arguments_digest", "one-shot", "self-approval", "atomic decision + audit"):
        assert token in lower, f"LEO_SECURITY.md missing {token!r}"


def test_tool_command_and_agent_counts_match_test_backed_roster() -> None:
    """Current claims must match the live roster; old counts need a marker."""
    live_tools, live_commands, all_commands, live_agents = _live_roster_counts()
    # Numeric guards: a shrunken registry or a stale doc must fail explicitly.
    # These floors reject the old 93-tool / 34-command / 20-agent claims.
    assert live_tools >= MIN_LIVE_TOOLS, f"live tool roster shrank to {live_tools}"
    assert live_commands >= MIN_LIVE_COMMANDS, f"live slash roster shrank to {live_commands}"
    assert all_commands >= live_commands
    assert live_agents >= MIN_LIVE_AGENTS, f"live agent roster shrank to {live_agents}"
    # The platform-agnostic tool allowance must stay a tight pair, not a range:
    # any other total is still a stale claim.
    from leo.tools.operator_tools import _COMPUTER_SPECS

    allowed_tools = _platform_tool_counts(live_tools)
    delta = -len(_COMPUTER_SPECS) if sys.platform == "darwin" else len(_COMPUTER_SPECS)
    assert allowed_tools == {live_tools, live_tools + delta}, allowed_tools

    progress = PROGRESS.read_text(encoding="utf-8")
    current = progress.split("## Milestone scope", 1)[0]
    claim_patterns = (
        (r"(\d+)\s+registered\s+tools\b", _platform_tool_counts(live_tools)),
        (r"(\d+)\s+available\s+slash\s+commands\b", {live_commands}),
        (r"(\d+)\s+registered\s+slash\s+commands\b", {all_commands}),
        (r"(\d+)\s+specialist\s+agents\b", {live_agents}),
    )
    for pattern, allowed in claim_patterns:
        claims = {int(n) for n in re.findall(pattern, current)}
        assert claims, f"current-state claim missing for {pattern!r}"
        assert claims <= allowed, (
            f"current-state claims {claims} for {pattern!r}, live composition root has {sorted(allowed)}"
        )
    for stale in ("90 registered tools", "93 registered tools", "34 available slash commands",
                  "20 specialist agents"):
        assert stale not in current, f"stale current-state claim: {stale}"

    markers = ("stale", "historical", "earlier")
    for line in progress.splitlines():
        for pattern, allowed in claim_patterns:
            for match in re.finditer(pattern, line):
                if int(match.group(1)) not in allowed:
                    assert any(marker in line.lower() for marker in markers), line
        if "18 tools, 12 skills" in line:
            assert any(marker in line.lower() for marker in ("stale", "historical")), line
    assert "All 14 phases delivered" not in progress

    if not INVENTORY.exists():
        return  # the release tree omits the legacy design-rationale document
    inventory = INVENTORY.read_text(encoding="utf-8")
    normalized_inventory = " ".join(inventory.split())  # markdown line wrapping
    for phrase in (
        f"{live_tools} unique model-callable tools",
        f"{live_commands} available slash commands",
        f"{all_commands} registered slash commands",
        f"{live_agents} specialist agents",
    ):
        assert phrase in normalized_inventory, (
            f"CAPABILITY_INVENTORY.md is missing current claim: {phrase}"
        )
    inventory_patterns = (
        (r"(\d+)\s+unique\s+model-callable\s+tools\b", live_tools),
        (r"(\d+)\s+available\s+slash\s+commands\b", live_commands),
        (r"(\d+)\s+registered\s+slash\s+commands\b", all_commands),
        (r"(\d+)\s+specialist\s+agents\b", live_agents),
        (r"(\d+)-tool\b", live_tools),
        (r"(\d+)-command\b", live_commands),
        (r"(\d+)-agent\b", live_agents),
    )
    for pattern, live in inventory_patterns:
        for match in re.finditer(pattern, normalized_inventory):
            if int(match.group(1)) != live:
                window = normalized_inventory[max(0, match.start() - 160): match.end() + 160]
                assert any(marker in window.lower() for marker in (*markers, "not current")), (
                    f"unqualified stale count in CAPABILITY_INVENTORY.md: {match.group(0)}"
                )


def test_suite_counts_are_marked_historical() -> None:
    """No stale pass count may be presented as the current roster."""
    candidates = [path for path in (PROGRESS, GIT_PLAN, INVENTORY) if path.exists()]
    for path in candidates:
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"\b(689|403) passed\b", line):
                assert any(marker in line.lower() for marker in HISTORICAL_MARKERS), (
                    f"{path.name}: unqualified stale test count: {line}"
                )
    report = TEST_REPORT.read_text(encoding="utf-8")
    assert "# Testing LEO" in report
    assert "pytest" in report


def test_git_plan_head_is_not_stale() -> None:
    """An obsolete plan must say so; only an active plan may carry a HEAD."""
    if not GIT_PLAN.exists():
        # The release tree removes the planning documents (Gate PF REMOVE);
        # there is no plan in the publish tree to check for staleness.
        pytest.skip("git planning documents are not part of this tree")
    text = GIT_PLAN.read_text(encoding="utf-8")
    assert "742f867" not in text
    lines = text.splitlines()
    obsolete = bool(lines) and bool(re.match(r"^#\s+OBSOLETE\b", lines[0])) and bool(
        re.search(r"^\*\*Status:\*\*\s*OBSOLETE\b", text, re.M)
    )
    if obsolete:
        # An obsolete document is not an execution plan. It must point at its
        # successor and must not carry a HEAD claim that silently goes stale.
        assert "GIT_OPEN_SOURCE_PLAN.md" in text
        assert not re.search(r"\bHEAD\b", text), "obsolete plan still tracks HEAD"
        assert "No push has been performed" in text
        return

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head in text, f"GIT_PUSH_PLAN.md does not cite current HEAD {head}"
    for sha in set(re.findall(r"\b[0-9a-f]{7,40}\b", text.lower())):
        assert head.startswith(sha), f"active plan cites stale commit {sha}, current is {head}"
    assert "No push has been performed yet." in text


def test_stress_narratives_are_not_published() -> None:
    """Unreproducible stress numbers are no longer part of the tree at all."""
    text = PROGRESS.read_text(encoding="utf-8").lower()
    assert "1000/1000" not in text and "999/1000" not in text
    assert "agent_reports/" not in text


def test_security_does_not_contradict_browser_startup_or_closed_gaps() -> None:
    text = SECURITY.read_text(encoding="utf-8")
    assert "remain BLOCKED without explicit user" not in text
    assert "starts its Playwright/Chromium backend lazily" in text
    assert "Verified remaining gaps from the 2026-11-18 deep audit" not in text
    assert "Deep-audit gaps that are now closed" in text


def test_schedule_docs_match_the_shipped_capability_matrix() -> None:
    """Phase 10 docs describe the code, not the old Phase 0 design snapshot."""
    from leo.contracts import SURFACE_CAPABILITIES

    text = (ROOT / "docs" / "SCHEDULES.md").read_text(encoding="utf-8")
    assert "Phase 0 only" not in text
    for heading in (
        "## Supported grammar", "## Consent and the fail-closed divergence",
        "## Delivery grammar", "## Native execution contract",
        "## Surface capability matrix", "## Migration, rollback and replay",
        "## Verification limits",
    ):
        assert heading in text, heading

    def yn(value: bool) -> str:
        return "yes" if value else "no"

    for surface, capabilities in SURFACE_CAPABILITIES.items():
        if not capabilities.can_present_card:
            continue
        row = (
            f"| `{surface}` | {yn(capabilities.can_present_card)} | "
            f"{yn(capabilities.can_wait_for_decision)} | {yn(capabilities.full_text)} | "
            f"{yn(capabilities.can_rewrite)} | {capabilities.max_card_chars} |"
        )
        assert row in text, f"capability matrix row missing for {surface}"
    for incapable in ("a2a", "whatsapp", "whatsapp_web", "agent", "scheduled", "chat"):
        assert f"`{incapable}`" in text

    lowered = text.lower()
    for token in (
        "automation.allow_scripts", "scripts_root", "room:<id>", "bot-chat:<profile>",
        "no locale subsystem", "no system-prompt cache", "approval_required",
        "api:<authenticated-principal>", "sec-fetch-site", "chrom",
    ):
        assert token in lowered, token
    assert "upstream tests" not in lowered or "never executed" in lowered


def test_schedule_docs_do_not_claim_browser_or_live_surface_verification() -> None:
    schedules = (ROOT / "docs" / "SCHEDULES.md").read_text(encoding="utf-8").lower()
    testing = " ".join(TEST_REPORT.read_text(encoding="utf-8").lower().split())
    assert "chromium is absent" in schedules
    assert "no browser verification is claimed" in schedules
    assert "no live telegram send" in schedules
    assert "no live gateway smoke" in schedules
    assert "skipped browser test is not browser verification" in testing
    assert "not performed by the default suite" in testing


def test_handoff_is_marked_historical() -> None:
    if not HANDOFF.exists():
        # The handoff note is personal and deliberately untracked; a clean
        # checkout (release worktree) has no such file to check.
        pytest.skip("personal handoff note not present in this tree")
    text = HANDOFF.read_text(encoding="utf-8")
    assert "## Current-state note (2026-11-19)" in text
    for line in text.splitlines():
        if "0 files, 0 directories" in line:
            assert "stale" in line.lower(), line
