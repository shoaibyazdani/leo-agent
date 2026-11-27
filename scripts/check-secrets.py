#!/usr/bin/env python3
"""Fail on credential-like strings in staged files, the working tree, Git objects, or transcripts.

Usage (from any directory):

    python3 scripts/check-secrets.py                       # staged files only (pre-commit hook)
    python3 scripts/check-secrets.py --all                 # release path set + all Git objects
    python3 scripts/check-secrets.py --transcripts         # ~/.claude/projects/**/*.jsonl
    python3 scripts/check-secrets.py --all --transcripts   # full gate, including transcript re-scan
    python3 scripts/check-secrets.py --allow path/to/file.py:42   # add a reviewed finding to the session allowlist

Design rules:
  * Never print a candidate secret; matches are replaced with [REDACTED] and truncated.
  * No skip-by-keyword and no skip-by-directory-named-tests. Every exception is an
    explicit, reviewed allowlist entry keyed by file and line (or a documented
    line range for hash-only manifests) in REVIEWED_ALLOWLIST below.
  * "--all" uses the explicit release path set from git ls-files --cached
    --others --exclude-standard. Ignored local data (.leo/, .venv/) is NOT
    scanned here; Gate P of docs/GIT_OPEN_SOURCE_PLAN.md reviews it separately
    and it must never be staged.
  * "--all" also scans every Git object via git cat-file --batch-all-objects,
    every unreachable object reported by git fsck --unreachable, and the
    message body of every commit and tag object (not just blob content).
  * "--transcripts" scans main session transcripts plus subagents/ and
    workflow transcripts under ~/.claude/projects/**/*.jsonl. Extra export
    bundles can be supplied with --transcript-dir or LEO_SECRET_TRANSCRIPT_DIRS.
  * Image OCR is NOT performed here. The 133 tracked PNGs are checked by the
    documented OCR procedure in Gate S of docs/GIT_OPEN_SOURCE_PLAN.md.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Pattern fragments are concatenated so this scanner does not itself contain a
# matchable credential literal. Prefixes alone are not findings; a minimum
# entropy/length suffix is required.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private key", re.compile(r"-----BEGIN " + r"[A-Z ]*" + r"PRIVATE KEY-----")),
    ("anthropic key", re.compile(r"sk-" + r"ant-" + r"[A-Za-z0-9_-]{12,}")),
    ("openai project key", re.compile(r"sk-proj-[A-Za-z0-9_-]{12,}")),
    ("openai key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("stripe live key", re.compile(r"sk_live_[A-Za-z0-9]{12,}")),
    ("opencode key", re.compile(r"oc_sk_" + r"[A-Za-z0-9_-]{12,}")),
    ("google api key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("github fine-grained pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("github pat", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{6,}\.eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}")),
    ("slack webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{10,}")),
    ("discord webhook", re.compile(
        r"https://(?:canary\.|ptb\.)?discord(?:app)?\.com/api/webhooks/"
        r"[0-9]{10,}/[A-Za-z0-9_-]{20,}"
    )),
    ("webhook url with token", re.compile(
        r"https://[A-Za-z0-9.-]+/[A-Za-z0-9_./-]*(?:webhook|hooks?)[A-Za-z0-9_./-]*"
        r"[?&](?:token|key|secret)=[A-Za-z0-9._-]{12,}",
        re.IGNORECASE,
    )),
    ("bearer token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}")),
    ("assigned secret", re.compile(
        r"(?i)(?:api[_-]?key|auth[_-]?token|access[_-]?token|password|secret|webhook)"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9._-]{16,}[\"']?"
    )),
    ("long hex", re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{40,}(?![0-9A-Fa-f])")),
]

# Explicit reviewed allowlist. Keys are "path:line", "path:start-end", or
# an open-ended range written as "start-". Each entry MUST carry a reason.
# A reason prefixed "only=<label>[|<label>]:" restricts the exception to those
# pattern families (a synthetic fixture can match more than one family);
# whole-file ranges are used ONLY for hash-only manifests and research notes
# documenting upstream SHAs, never for provider-key prefixes.
REVIEWED_ALLOWLIST: dict[str, dict[str, str]] = {
    "uv.lock": {"1-": "only=long hex: lockfile integrity hashes, not credentials"},
    "tests/test_harden_config.py": {
        "280-281": "synthetic invalid-token parameter",
        "285": "only=anthropic key: synthetic fixture at its current line",
        "430-458": "synthetic provider fixtures and hex fixture",
    },
    "tests/channels/test_bluebubbles.py": {"23": "test fixture", "25": "test fixture"},
    "tests/channels/test_credential_enrollment.py": {"70": "test fixture"},
    "tests/channels/test_dingtalk.py": {
        "26": "test fixture", "28": "test fixture", "29": "test fixture",
    },
    "tests/channels/test_irc.py": {"21": "test fixture"},
    "tests/channels/test_ntfy.py": {"19": "test fixture"},
    "tests/channels/test_qqbot.py": {"26": "test fixture", "27": "test fixture"},
    "tests/channels/test_relay.py": {"22": "test fixture"},
    "tests/channels/test_signal.py": {"30": "test fixture", "32": "test fixture"},
    "tests/channels/test_slack.py": {"21": "test fixture", "83": "test fixture"},
    "tests/channels/test_teams.py": {"32": "test fixture"},
    "tests/test_agents_automation_operator.py": {
        "40": "test fixture",
        "42": "only=assigned secret: synthetic fixture in an earlier revision",
        "43": "only=assigned secret: synthetic fixture at its current line",
        "41": "only=assigned secret: synthetic fixture in an earlier revision",
        "33": "only=assigned secret: synthetic fixture in an earlier revision",
    },
    "tests/test_agents_reviewer.py": {"143": "test fixture"},
    "tests/test_commands.py": {
        "514": "test fixture",
        # Two credential-shaped pattern families match these synthetic fixtures.
        "498": "synthetic secret fixture in an earlier revision",
        "499": "synthetic secret fixture in an earlier revision",
    },
    # Every entry is restricted to the exact pattern families its fixture
    # matches, at both current and historical line numbers; a future real
    # credential on one of these lines is still reported by the gate.
    "tests/test_tools_http.py": {
        "108": "only=openai key: synthetic fixture",
        "123": "only=openai key: synthetic fixture",
        "643": "only=assigned secret: synthetic fixture",
        "660": "only=openai key|assigned secret: synthetic fixture",
        "710": "only=openai key: redaction assertion",
        "788": "only=openai key|assigned secret: synthetic fixture",
        "805": "only=openai key: redaction assertion",
        "866": "only=openai key|assigned secret: synthetic fixture",
        "937": "only=openai key|assigned secret: synthetic fixture",
        "980": "only=openai key|assigned secret: synthetic fixture",
        "990": "only=openai key|assigned secret: synthetic fixture",
        "1021": "only=openai key|assigned secret: synthetic fixture",
        "1260": "only=openai key|assigned secret: synthetic fixture",
        "1347": "only=openai key|assigned secret: synthetic fixture",
        # The same fixtures after later test insertions; the older line
        # numbers stay for historical blobs.
        "669": "only=assigned secret: synthetic fixture at its current line",
        "686": "only=openai key|assigned secret: synthetic fixture at its current line",
        "736": "only=openai key: redaction assertion at its current line",
        "814": "only=openai key|assigned secret: synthetic fixture at its current line",
        "831": "only=openai key: redaction assertion at its current line",
        "892": "only=openai key|assigned secret: synthetic fixture at its current line",
        "963": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1006": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1016": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1047": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1286": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1373": "only=openai key|assigned secret: synthetic fixture at its current line",
        # Insertions from follow-up issues shifted these known fixtures again.
        "113": "only=openai key: synthetic fixture at its current line",
        "128": "only=openai key: synthetic fixture at its current line",
        "706": "only=assigned secret: synthetic fixture at its current line",
        "723": "only=openai key|assigned secret: synthetic fixture at its current line",
        "773": "only=openai key: redaction assertion at its current line",
        "851": "only=openai key|assigned secret: synthetic fixture at its current line",
        "868": "only=openai key: redaction assertion at its current line",
        "929": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1000": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1043": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1053": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1084": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1323": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1410": "only=openai key|assigned secret: synthetic fixture at its current line",
        "709": "only=openai key|assigned secret: synthetic fixture at its current line",
        "726": "only=openai key|assigned secret: synthetic fixture at its current line",
        "776": "only=openai key|assigned secret: synthetic fixture at its current line",
        "854": "only=openai key|assigned secret: synthetic fixture at its current line",
        "871": "only=openai key|assigned secret: synthetic fixture at its current line",
        "932": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1003": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1046": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1056": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1087": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1326": "only=openai key|assigned secret: synthetic fixture at its current line",
        "1413": "only=openai key|assigned secret: synthetic fixture at its current line",
    },
    "tests/test_tools_webhook.py": {"33": "test fixture"},
    "tests/channels/test_a2a.py": {"42": "test fixture"},
    "tests/channels/test_wechat_crypto.py": {"61": "only=long hex: synthetic hex fixture"},
    "tests/test_enrollment.py": {"35": "test fixture"},
    "tests/test_agent_activity_dashboard.py": {"95": "test fixture", "151": "test fixture"},
    "tests/test_tools_run.py": {"156": "test fixture", "202": "test fixture"},
    "leo/integrations/whatsapp_bridge/manager.py": {
        "84": "only=long hex: SHA-256 integrity pin of the shipped bridge.js",
    },
    "scripts/leo_conversation_observer.py": {
        "1607": "synthetic leak-token fixture used to verify redaction on disk",
        "728": "synthetic bearer fixture in an earlier revision (two pattern families)",
    },
    "tests/test_self_improvement.py": {
        "146": "only=assigned secret: synthetic secret-refusal fixture",
        "169": "only=assigned secret: synthetic secret-refusal fixture",
    },
    "junit.xml": {
        "1": "only=anthropic key|stripe live key: generated pytest evidence containing synthetic fixture names",
    },
    "leo/tools/shell_policy.py": {
        "671": "only=assigned secret: local variable naming a classifier helper result",
    },
    "tests/test_api.py": {
        "241": "test fixture", "345": "synthetic failure-message fixture",
        "346": "only=assigned secret: synthetic failure-message fixture at its current line",
        "452": "test fixture", "489": "test fixture",
        "556": "synthetic secret fixture", "593": "synthetic secret fixture",
        "557": "only=assigned secret: synthetic secret fixture at its current line",
        "594": "only=assigned secret: synthetic secret fixture at its current line",
        "257": "only=assigned secret: synthetic failure-message fixture in an earlier revision",
        "468": "only=assigned secret: synthetic secret fixture in an earlier revision",
        "505": "only=assigned secret: synthetic secret fixture in an earlier revision",
    },
    # Pinned commits and SHA-256 evidence in the scheduling parity corpus are
    # integrity metadata, never credentials.
    "docs/SCHEDULES.md": {"1-": "only=long hex: pinned commit identifiers"},
    "scripts/release/evidence/junit.xml": {
        "1": "synthetic parametrized test-case names from existing redaction fixtures, not live credentials",
    },
    "scripts/release/evidence/receipt.json": {
        "1-": "only=long hex: JUnit SHA-256 evidence hash",
    },
    "scripts/schedule_parity/extract_fixtures.py": {
        "19": "only=long hex: pinned oracle commit",
    },
    "scripts/schedule_parity/run_baseline.py": {
        "18-19": "only=long hex: pinned baseline commit and lock hash",
    },
    "tests/fixtures/schedule_parity/acceptance_matrix.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/cancel.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/create.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/list.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/pause.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/remove.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/resume.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/run.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/actions/update.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/baseline/receipt.json": {"1-": "only=long hex: pinned baseline and evidence hashes"},
    "tests/fixtures/schedule_parity/baseline_failures.json": {"1-": "only=long hex: pinned baseline commit"},
    "tests/fixtures/schedule_parity/consent_cards.json": {"1-": "only=long hex: pinned commit and source hashes"},
    "tests/fixtures/schedule_parity/delivery_tokens.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/fields.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/grammar.json": {"1-": "only=long hex: pinned commit and source hashes"},
    "tests/fixtures/schedule_parity/oracle_manifest.json": {"1-": "only=long hex: pinned oracle and fixture hashes"},
    "tests/fixtures/schedule_parity/repeat.json": {"1-": "only=long hex: pinned oracle commit"},
    "tests/fixtures/schedule_parity/tool_schema.json": {"1-": "only=long hex: pinned commit and schema hashes"},
    "tests/test_schedule_safety.py": {
        "71-139": "synthetic secret and control-character fixtures exercising the scanner and redactor",
    },
    "tests/test_schedule_security.py": {
        "28": "synthetic unsafe-prompt fixture",
        "110": "assertion that the synthetic token is absent in an earlier revision",
        "204": "only=openai key: assertion that the synthetic token is absent",
    },
    "tests/test_secret_sinks.py": {
        "290": "synthetic PEM fixture used to verify redaction",
    },
    "tests/test_chat_tools.py": {
        "1298": "test fixture",
        "1147": "only=assigned secret: synthetic key fixture in an earlier revision",
        "1315": "only=assigned secret: synthetic key fixture in an earlier revision",
    },
    "tests/test_conversation_observer.py": {
        "92-103": "synthetic tokens exercising the redactor",
        "154": "test fixture",
        "179": "test fixture",
        "222": "test fixture",
        "91": "synthetic bearer fixture in an earlier revision (two pattern families)",
        "153": "only=assigned secret: synthetic leak fixture in an earlier revision",
        "172": "synthetic secret fixture in an earlier revision (two pattern families)",
        "215": "only=openai key: redaction assertion in an earlier revision",
    },
    "tests/test_learning_engine.py": {
        "138": "synthetic token in a redaction test",
        "162": "assertion that the secret is absent",
        "179": "synthetic token in a memory-capture test",
        "186": "synthetic token in a note-draft test",
        "464": "synthetic token in an error-redaction test",
        "474": "assertion that the secret is absent",
        # Same fixtures at their current and earlier line numbers. Restricted to
        # one pattern family so a future credential in this file still fails.
        "126": "synthetic redaction fixture (two pattern families)",
        "140": "synthetic redaction fixture (two pattern families)",
        "150": "only=openai key: redaction assertion",
        "164": "only=openai key: redaction assertion",
        "167": "synthetic capture fixture (two pattern families)",
        "174": "synthetic draft fixture (two pattern families)",
        "181": "synthetic capture fixture (two pattern families)",
        "188": "synthetic draft fixture (two pattern families)",
        "431": "synthetic error-redaction fixture (two pattern families)",
        "441": "only=openai key: redaction assertion",
        "495": "synthetic error-redaction fixture (two pattern families)",
        "505": "only=openai key: redaction assertion",
    },
    "tests/test_model_presets.py": {
        "64": "only=openai project key: synthetic provider preset fixture",
    },
    "tests/test_onboarding.py": {"37": "synthetic fixture"},
    "tests/test_self_evolution_integration.py": {
        "146": "synthetic token", "257": "synthetic token", "305": "synthetic token",
    },
    "tests/test_selfheal.py": {
        "633": "test fixture",
        "535": "only=assigned secret: synthetic provider-error fixture in an earlier revision",
    },
    "tests/test_tools.py": {
        "216": "test placeholder", "358": "test fixture",
        "334": "synthetic secret fixture in an earlier revision (two pattern families)",
    },
    "tests/test_tools_conversation.py": {
        "36": "synthetic token", "855": "synthetic stripe-shaped fixture", "956": "test fixture",
        "29": "synthetic token fixture in an earlier revision (two pattern families)",
        "903": "only=stripe live key: synthetic token fixture",
        "1004": "only=stripe live key: synthetic audit-id fixture",
    },
    "tests/test_tools_preview.py": {
        "815": "test fixture",
        "778": "only=bearer token: synthetic header fixture in an earlier revision",
    },
    "tests/test_whatsapp_bridge.py": {
        "1419": "test fixture",
        "524": "only=bearer token: synthetic header fixture in an earlier revision",
        "1389": "only=bearer token: synthetic header fixture in an earlier revision",
    },
    # Historical research notes pin upstream commit SHAs; long-hex matches
    # there are integrity metadata, never provider credentials. A trailing
    # slash is a reviewed path-prefix key, so the scanner need not repeat
    # historical product names in a public allowlist.
    "research/deep-2026-11-18/": {
        "1-": "only=long hex: pinned upstream commit identifiers in historical notes",
    },
    "research/deep-2026-11-18/leo-e2e.md": {
        "197": "only=assigned secret: documented synthetic leak fixture",
        "205": "only=assigned secret: documented synthetic leak fixture",
        "447": "only=assigned secret: documented synthetic leak fixture",
    },
    # A cherry-pick's source SHA in a commit message is structural metadata.
    "object:commit:1d0cba3cf52c:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:27c6073db5ec:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:42cfe0c03ae7:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:69dc4e16432c:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:85e13e4f748f:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:898982bbbf67:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:959be04d514b:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:a2f3a03cadcd:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:d076edd6b417:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:e43c509136ad:msg": {"1-": "only=long hex: cherry-picked commit reference"},
    "object:commit:f99b540085d9:msg": {"1-": "only=long hex: cherry-picked commit reference"},
}


def git(*args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def staged_files() -> list[str]:
    result = git("diff", "--cached", "--name-only", "--diff-filter=ACM", "-z")
    return [p for p in result.stdout.decode(errors="replace").split("\0") if p]


def release_path_set() -> list[str]:
    result = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return [p for p in result.stdout.decode(errors="replace").split("\0") if p]


def all_object_ids() -> tuple[list[str], list[str]]:
    """Return (all object ids, unreachable object ids) from the local object store."""
    listing = git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)")
    oids: list[str] = []
    for line in listing.stdout.decode(errors="replace").splitlines():
        fields = line.split()
        if fields:
            oids.append(fields[0])
    fsck = git("fsck", "--unreachable", "--no-reflogs")
    unreachable: list[str] = []
    for line in fsck.stdout.decode(errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "unreachable":
            unreachable.append(fields[-1])
    if listing.returncode != 0:
        print("warning: git cat-file --batch-all-objects failed", file=sys.stderr)
    if fsck.returncode != 0:
        print("warning: git fsck --unreachable reported an error or dangling objects", file=sys.stderr)
    combined = list(dict.fromkeys([*oids, *unreachable]))
    return combined, unreachable


def object_batches(oids: list[str]):
    """Yield (oid, type, content bytes) for every requested object."""
    if not oids:
        return
    result = git("cat-file", "--batch", input_bytes=("\n".join(oids) + "\n").encode())
    buf = result.stdout
    pos = 0
    while pos < len(buf):
        newline = buf.find(b"\n", pos)
        if newline < 0:
            break
        header = buf[pos:newline].decode(errors="replace").split()
        pos = newline + 1
        if len(header) < 3:
            continue
        oid, kind, size = header[0], header[1], int(header[2])
        content = buf[pos:pos + size]
        pos += size + 1
        yield oid, kind, content


def transcript_files(extra_dirs: list[str]) -> list[Path]:
    roots: list[Path] = [Path.home() / ".claude" / "projects"]
    for raw in extra_dirs:
        roots.extend(Path(p).expanduser() for p in raw.split(os.pathsep) if p)
    found: dict[Path, None] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if path.is_file():
                found[path.resolve()] = None
    return list(found)


def display(path: Path) -> str:
    """Never echo a full home directory path in scanner output."""
    try:
        return "~/" + path.resolve().relative_to(Path.home()).as_posix()
    except ValueError:
        return path.name


def is_reviewed(path_key: str, line: int, label: str) -> bool:
    keys = [path_key]
    keys.extend(
        key for key in REVIEWED_ALLOWLIST if key.endswith("/") and path_key.startswith(key)
    )
    for key in keys:
        for spec, reason in REVIEWED_ALLOWLIST.get(key, {}).items():
            only = (
                set(reason.split(":", 1)[0][len("only="):].split("|"))
                if reason.startswith("only=") else None
            )
            if only is not None and label not in only:
                continue
            if spec == "*":
                return True
            if "-" in spec:
                start, _, end = spec.partition("-")
                if int(start) <= line and (not end or line <= int(end)):
                    return True
            elif spec.isdigit() and int(spec) == line:
                return True
    return False


def scan_lines(text: str, source: str, allow_key: str, allow_hex: bool = True) -> int:
    findings = 0
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS:
            if label == "long hex" and not allow_hex:
                continue
            if is_reviewed(allow_key, number, label):
                continue
            if pattern.search(line):
                redacted = pattern.sub("[REDACTED]", line).strip()[:160]
                print(f"{source}:{number}: {label}: {redacted}", file=sys.stderr)
                findings += 1
                break
    return findings


def scan_path(path: Path, allow_key: str) -> int:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        print(f"warning: cannot read {allow_key}: {exc}", file=sys.stderr)
        return 0
    text = raw.decode("utf-8", errors="ignore")
    # Binary assets are scanned for ASCII credential signatures only; hosted
    # image OCR is a separate, documented Gate S procedure.
    allow_hex = path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".db", ".sqlite", ".pyc"}
    return scan_lines(text, allow_key, allow_key, allow_hex=allow_hex)


def blob_path_index() -> dict[str, list[str]]:
    """Map blob id -> repository path(s) so the file+line allowlist applies.

    Without this, a reviewed fixture was reported once per historical blob with
    an ``object:`` key the allowlist could never match, and every commit's
    tree/parent line was reported as a "secret" — a gate that can never pass.
    """
    result = git("rev-list", "--objects", "--all")
    index: dict[str, list[str]] = {}
    for line in result.stdout.decode(errors="replace").splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1]:
            index.setdefault(parts[0], []).append(parts[1])
    return index


def scan_objects(oids: list[str], unreachable: list[str]) -> int:
    findings = 0
    unreachable_set = set(unreachable)
    blob_paths = blob_path_index()
    for oid, kind, content in object_batches(oids):
        tag = "unreachable " if oid in unreachable_set else ""
        source = f"object:{tag}{kind}:{oid[:12]}"
        text = content.decode("utf-8", errors="ignore")
        if kind == "blob":
            paths = blob_paths.get(oid, [])
            if paths:
                findings += _scan_blob_by_path(text, paths)
            else:
                findings += scan_lines(text, source, f"object:{oid}")
        elif kind == "tree":
            # Tree entries are "mode type <sha> path": the hashes are
            # structural metadata, never credentials.
            findings += scan_lines(text, source, f"object:{oid}", allow_hex=False)
        elif kind in {"commit", "tag"}:
            # Headers (tree/parent/object ids, author/committer/tagger) are
            # metadata: their SHAs are structural, never credentials. The
            # message body is scanned separately so no commit text is exempt.
            header, _, message = text.partition("\n\n")
            findings += scan_lines(header, source, f"object:{oid}", allow_hex=False)
            # Keep the allowlist key identical to the displayed source so a
            # reviewed cherry-pick SHA can be admitted by its short object id.
            findings += scan_lines(message, f"{source}:msg", f"{source}:msg")
    return findings


_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".db", ".sqlite", ".pyc"}


def _scan_blob_by_path(text: str, paths: list[str]) -> int:
    """Scan a blob under its repository path keys so the allowlist applies.

    A line is skipped only when *every* path that references this blob
    allowlists that line and label — a shared blob cannot use one reviewed
    fixture to smuggle content into a path that was never reviewed.
    """
    findings = 0
    allow_hex = not all(Path(p).suffix.lower() in _BINARY_SUFFIXES for p in paths)
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS:
            if label == "long hex" and not allow_hex:
                continue
            if all(is_reviewed(path, number, label) for path in paths):
                continue
            if pattern.search(line):
                redacted = pattern.sub("[REDACTED]", line).strip()[:160]
                print(
                    f"object:blob({','.join(paths)[:80]}):{number}: {label}: {redacted}",
                    file=sys.stderr,
                )
                findings += 1
                break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="scan the explicit release path set")
    parser.add_argument("--objects", action="store_true", help="scan every Git object and commit/tag message")
    parser.add_argument("--transcripts", action="store_true", help="scan Claude session/subagent/workflow transcripts")
    parser.add_argument("--transcript-dir", action="append", default=[], help="extra transcript/export bundle directory")
    parser.add_argument("--allow", action="append", default=[], metavar="PATH:LINE", help="session-only reviewed exception")
    args = parser.parse_args()

    for entry in args.allow:
        key, sep, line = entry.rpartition(":")
        if not sep or not key or not line.isdigit():
            parser.error(f"--allow expects PATH:LINE, got {entry!r}")
        REVIEWED_ALLOWLIST.setdefault(key, {})[line] = "session reviewed exception (record reason in release log)"

    if args.all:
        paths = release_path_set()
        scope = "release path set"
        print("note: ignored local data is not scanned; review it in Gate P and never stage it", file=sys.stderr)
    else:
        paths = staged_files()
        scope = "staged files"

    findings = 0
    for rel in paths:
        findings += scan_path(ROOT / rel, rel)

    object_ids: list[str] = []
    unreachable: list[str] = []
    if args.all or args.objects:
        object_ids, unreachable = all_object_ids()
        findings += scan_objects(object_ids, unreachable)

    transcripts: list[Path] = []
    if args.transcripts:
        extra = [*args.transcript_dir, os.environ.get("LEO_SECRET_TRANSCRIPT_DIRS", "")]
        transcripts = transcript_files(extra)
        for path in transcripts:
            key = display(path)
            findings += scan_path(path, key)

    print(
        f"Scanned {len(paths)} {scope}, {len(object_ids)} Git objects "
        f"({len(unreachable)} unreachable), {len(transcripts)} transcript files."
    )
    if findings:
        print(
            f"\n{findings} potential secret(s) found. Fix the value (rotate at the provider), "
            "or add an explicit reviewed file+line allowlist entry with a reason; "
            "never stage transcripts or credential stores.",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
