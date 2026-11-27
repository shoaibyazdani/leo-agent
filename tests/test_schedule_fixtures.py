"""Deterministic Phase 0 contract tests for the frozen scheduling oracle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.schedule_parity import extract_fixtures as extractor
from scripts.schedule_parity import run_baseline

FIXTURES = Path(__file__).parent / "fixtures" / "schedule_parity"
ORACLE_SOURCE = os.environ.get("LEO_SCHEDULE_ORACLE_SOURCE")
ANALYSIS_DIR = (
    Path(ORACLE_SOURCE).resolve().parent
    if ORACLE_SOURCE
    else Path("/Users/macbook/.claude/projects/-Users-macbook-LEO/6f293eb9-a422-47af-a78a-4844224d6020/pr-158289-analysis")
)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def source_available() -> bool:
    return bool(ORACLE_SOURCE and Path(ORACLE_SOURCE).is_dir() and ANALYSIS_DIR.is_dir())


def validate_golden_card(card: dict, contract: dict) -> None:
    """Reject card additions unless the frozen oracle explicitly allows them."""
    if set(card) != set(contract["card_allowed_keys"]):
        raise ValueError("golden card has fields without an oracle schema entry")
    if set(card["job"]) - set(contract["card_job_allowed_fields"]):
        raise ValueError("golden card job has fields without an oracle entry")
    repeat = card["job"].get("repeat")
    if repeat is not None and set(repeat) - set(contract["card_repeat_allowed_keys"]):
        raise ValueError("golden card repeat has fields without an oracle entry")


def test_manifest_pins_commit_sources_and_every_fixture_hash():
    manifest = load("oracle_manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["oracle_commit"] == extractor.ORACLE_COMMIT
    assert manifest["oracle_role"] == "read-only behavioral source; never executed"
    assert manifest["source_checkout_clean"] is True
    assert manifest["fixture_count"] == 15
    expected_files = set(manifest["fixture_files"]) | {"oracle_manifest.json"}
    actual_files = {
        str(path.relative_to(FIXTURES))
        for path in FIXTURES.rglob("*.json")
        if "baseline" not in path.parts and path.name != "baseline_failures.json"
    }
    assert actual_files == expected_files
    for relative, expected_hash in manifest["fixture_files"].items():
        assert hashlib.sha256((FIXTURES / relative).read_bytes()).hexdigest() == expected_hash


def test_frozen_source_hashes_and_regeneration_match_when_corpus_is_available():
    manifest = load("oracle_manifest.json")
    if not source_available():
        pytest.skip("set LEO_SCHEDULE_ORACLE_SOURCE to compare against the read-only corpus")
    source_root = Path(ORACLE_SOURCE).resolve()
    for relative, record in manifest["source_files"].items():
        path = source_root / relative
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    generated = extractor.build_fixtures(source_root, ANALYSIS_DIR)
    assert set(generated) == set(manifest["fixture_files"])
    for relative, payload in generated.items():
        assert load(relative) == payload


def test_tool_schema_field_names_and_exposure_are_exact():
    schema = load("tool_schema.json")
    fields = load("fields.json")
    expected_model_fields = {
        "action", "attach_to_session", "catch_up", "context_from", "continuity",
        "deliver", "enabled_toolsets", "failure_deliver", "job_id", "monitor", "name",
        "no_agent", "paused", "paused_reason", "pinned", "prompt", "repeat", "schedule",
        "script", "skills", "timezone", "workdir", "wrap_response",
    }
    assert set(schema["parameters"]["properties"]) == expected_model_fields
    assert fields["declared_model_fields"] == sorted(expected_model_fields)
    assert schema["parameters"]["required"] == ["action"]
    assert "enum" not in schema["parameters"]["properties"]["action"]
    assert "additionalProperties" not in schema["parameters"]
    assert schema["parameters"]["properties"]["repeat"]["type"] == "integer"
    assert set(fields["operator_only_fields"]) == {"provider", "model", "base_url", "reasoning_effort", "interpreter"}
    assert set(fields["operator_only_fields"]).isdisjoint(expected_model_fields)
    assert {"skill", "reason", "grace_minutes"}.isdisjoint(expected_model_fields)
    assert {"skill", "reason", "grace_minutes"} <= set(fields["handler_forwarded_fields"])
    assert {"monitor_script", "monitor_url"} <= set(fields["legacy_compat_fields"])
    assert fields["internal_fields"] == ["approve", "session_id"]
    assert set(fields["required_runtime_semantics"]) == {
        "script", "no_agent", "monitor", "context_from", "continuity", "enabled_toolsets",
        "skills", "wrap_response", "workdir", "attach_to_session", "pinned", "grace_minutes",
    }
    assert set(fields["leo_field_adaptations"]) == set(fields["required_runtime_semantics"])
    assert fields["removed_scope_absent"] == [
        "inline script authoring", "per-job script state directories", "gate-context/watchdog machinery",
        "connector-call facade", "watcher skills/templates",
    ]
    if source_available():
        assert schema == extractor.extract_tool_schema(Path(ORACLE_SOURCE).resolve())


def test_every_final_head_action_has_one_fixture_and_get_is_not_invented():
    actions = ("create", "list", "update", "pause", "resume", "remove", "run", "cancel")
    action_files = sorted(path.stem for path in (FIXTURES / "actions").glob("*.json"))
    assert action_files == sorted(actions)
    schema = load("tool_schema.json")
    assert schema["declared_actions"] == list(actions)
    assert schema["action_aliases"] == {"run": ["run_now", "trigger"]}
    assert schema["get_action_present"] is False
    assert set(schema["jobless_actions"]) == {"create", "list"}
    assert set(schema["job_actions"]) == {*actions[2:], "run_now", "trigger"}
    for action in actions:
        fixture = load(f"actions/{action}.json")
        assert fixture["action"] == action
        assert fixture["tool_name"] == "cronjob_manage"
        assert fixture["aliases"] == (["run_now", "trigger"] if action == "run" else [])
        assert fixture["action_description"]
        assert "additionalProperties" not in fixture["input_schema"]
        for field, definition in fixture["input_schema"]["properties"].items():
            assert definition == schema["parameters"]["properties"][field]
        assert set(fixture["effective_handler_fields"]) >= set(fixture["input_schema"]["required"])
        assert fixture["result_keys"]
        if action in {"create", "update"}:
            assert fixture["upstream_consent"]
        else:
            assert fixture["upstream_consent"] is None
    assert load("actions/pause.json")["effective_handler_fields"] == ["job_id", "reason"]
    assert load("actions/resume.json")["effective_handler_fields"] == ["job_id"]
    assert load("actions/cancel.json")["result_keys"] == [
        "job", "success", "cancelled", "execution_id", "status", "pid", "detail",
    ]


def test_grammar_fixtures_cover_source_examples_aliases_and_negative_cases():
    grammar = load("grammar.json")
    cases = grammar["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert ("kind" in case) != ("error_contains" in case)
        if "kind" in case:
            assert case["kind"] in {"once", "interval", "cron"}
        if case.get("kind") == "cron":
            assert len(case["expr"].split()) in {5, 6}
    by_input = {case["input"]: case for case in cases}
    assert by_input["30m"]["kind"] == "interval"
    assert by_input["in 30m"]["kind"] == "once"
    assert by_input["every 1d"]["minutes"] == 1440
    assert by_input["0 0 9 * * *"]["kind"] == "cron"
    assert "error_contains" in by_input["@every 1h"]
    assert "error_contains" in by_input["30s"]
    assert by_input["0m"]["oracle_gap"]
    assert grammar["duration_units"] == {
        "m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1,
        "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
        "d": 1440, "day": 1440, "days": 1440,
    }
    assert grammar["duration_count_optional"] is True
    assert grammar["duration_negative_cases"] == [
        {"input": value, "error_contains": "Invalid duration"}
        for value in ("abc", "hourx", "30x", "", "m30")
    ]
    assert grammar["weekday_aliases"]["sunday"] == "0"
    assert grammar["weekday_aliases"]["sat"] == "6"
    assert grammar["day_spec_aliases"]["weekdays"] == "1-5"
    assert grammar["day_spec_aliases"]["weekends"] == "0,6"
    assert grammar["clock_forms"] == ["9am", "9:30am", "14:00", "7", "noon", "midnight", "12am", "12pm"]
    assert grammar["semantic_equality"] == ["kind", "interval minutes or cron expression", "exact one-shot timestamp"]
    assert "no explicit" in grammar["timezone_policy"]["dst_gap"]
    if source_available():
        source_examples = extractor.parser_examples(Path(ORACLE_SOURCE).resolve())
        frozen_examples = [
            (case["source_line"], case["input"])
            for case in cases if case["id"].startswith("test-jobs-")
        ]
        assert frozen_examples == source_examples


def test_repeat_normalization_and_accounting_are_frozen():
    repeat = load("repeat.json")
    by_input = {json.dumps(item["input"], sort_keys=True): item for item in repeat["normalization"]}
    assert by_input[json.dumps(None)]["stored_times"] is None
    assert by_input[json.dumps("once")]["stored_times"] == 1
    assert by_input[json.dumps("3")]["stored_times"] == 3
    assert by_input[json.dumps("forever")]["stored_times"] is None
    assert by_input[json.dumps(0)]["oracle_gap"]
    assert by_input[json.dumps(True)]["stored_times"] == 1
    assert by_input[json.dumps(3.9)]["stored_times"] == 3
    assert "error" in by_input[json.dumps("banana")]
    assert repeat["declared_schema_type"] == "integer"
    assert repeat["runtime_accepts_numeric_strings"] is True
    assert repeat["one_shot_default"] == 1
    assert repeat["completed_counter_preserved_on_update"] is True
    assert repeat["manual_runs_do_not_consume_repeat_or_cadence"] is True
    assert "reject" in repeat["leo_hardening"]


def test_delivery_tokens_distinguish_oracle_forms_from_native_extensions():
    delivery = load("delivery_tokens.json")
    tokens = {item["token"]: item for item in delivery["tokens"]}
    oracle_tokens = {
        "omitted", "origin", "all", "local", "bot-chat", "bot-chat:<profile>",
        "room:<id>", "room:<id>@<profile>", "<platform>:<chat>",
        "<platform>:<chat>:<thread>", "<platform>", "comma list", "list/tuple input",
        "failure_deliver",
    }
    native_tokens = {"home", "local:<id>", "object target"}
    assert {name for name, item in tokens.items() if item["oracle"]} == oracle_tokens
    assert {name for name, item in tokens.items() if not item["oracle"]} == native_tokens
    assert set(delivery["leo_adaptation"]) == native_tokens | {"unknown_tokens", "bot-chat", "room"}
    assert delivery["normalization"]["unknown_token"].startswith("resolves nowhere")
    assert delivery["target_identity"] == ["platform.lower()", "chat_id", "thread_id"]
    assert delivery["result_and_failure_lanes_separate"] is True


def test_golden_cards_are_literal_complete_and_schema_closed():
    contract = load("consent_cards.json")
    cases = {case["case_id"]: case for case in contract["cases"]}
    assert set(cases) == {
        "create-recurring-three-runs", "create-script-only", "create-paused", "create-one-shot",
        "create-execution-extras", "edit-prompt", "edit-execution-mode", "edit-delivery",
    }
    for case in cases.values():
        validate_golden_card(case, contract)
        assert case["body"]
        assert case["cli_header"].endswith(case["body"])
        assert len(case["choices"]) == len(case["choice_labels"])
        if case["kind"] == "schedule":
            assert case["choices"] == ["once", "session", "always", "deny"]
            assert case["choice_labels"] == ["Allow", "Allow for this chat", "Always allow new jobs", "Deny"]
            assert "results go to " in case["body"]
            assert case["body"].startswith(f'Schedule "{case["job"]["name"]}"')
        else:
            assert case["choices"] == ["once", "deny"]
            assert case["choice_labels"] == ["Allow", "Deny"]
            assert case["body"].startswith(f'Change "{case["job"]["name"]}": ')
            assert " -> " in case["body"]
        if case["job"].get("prompt"):
            assert case["job"]["prompt"] in case["body"]
        if case["job"].get("enabled") is False:
            assert case["shown_first_run"] is None
            assert "created paused" in case["body"] and "first run" not in case["body"]
        elif case["kind"] == "schedule":
            assert case["shown_first_run"] == case["job"]["next_run_at"]
            assert "first run" in case["body"] or case["job"]["schedule"]["kind"] == "once"
    assert cases["create-recurring-three-runs"]["body"] == (
        'Schedule "Water" (every day at 8am, first run Wed Nov 25 08:00 UTC, 3 runs): '
        'Remind the user to drink a glass of water.; results go to nowhere (saved to the job\'s history only)'
    )
    assert cases["create-script-only"]["body"] == (
        'Schedule "Check" (every 2h, first run Wed Nov 25 09:00 UTC): runs script check.py; '
        'results go to nowhere (saved to the job\'s history only)'
    )
    assert cases["create-paused"]["body"] == (
        'Schedule "Water" (every day at 8am, created paused: it will not run until resumed, 3 runs): '
        'Remind the user to drink a glass of water.; results go to nowhere (saved to the job\'s history only)'
    )
    assert cases["create-one-shot"]["body"] == (
        'Schedule "Dentist" (once, Fri Nov 27 09:00 UTC): Call the dentist.; results go to telegram:12345'
    )
    assert cases["edit-prompt"]["body"] == (
        'Change "Water": prompt: Remind the user to drink a glass of water. -> Drink tea instead.'
    )
    extras = cases["create-execution-extras"]["body"]
    for fragment in (
        "skills: maps", "after running script collect.py", "checks URL https://example.com/price first each run",
        "runs in /workspace/brief", "reads output of source-job", "toolsets: none",
        "copied into the chat's history", "failure notices go to nowhere",
    ):
        assert fragment in extras
    assert "nothing was saved." in contract["decline_messages"]["stale_create"]
    assert "no schedule write and no approval row" in contract["leo_divergence"]


def test_golden_card_schema_rejects_unregistered_card_and_job_fields():
    contract = load("consent_cards.json")
    card = copy.deepcopy(contract["cases"][0])
    card["unregistered"] = True
    with pytest.raises(ValueError, match="fields without an oracle schema entry"):
        validate_golden_card(card, contract)
    card = copy.deepcopy(contract["cases"][0])
    card["job"]["unregistered"] = True
    with pytest.raises(ValueError, match="fields without an oracle entry"):
        validate_golden_card(card, contract)


def test_consent_function_digests_match_source_when_available():
    if not source_available():
        pytest.skip("set LEO_SCHEDULE_ORACLE_SOURCE to compare card function digests")
    source, tree = extractor.read_ast(Path(ORACLE_SOURCE).resolve() / "tools/cronjob_tools.py")
    contract = load("consent_cards.json")
    for name, record in contract["source_functions"].items():
        assert extractor.function_record(source, tree, name) == record


def test_acceptance_matrix_has_four_sections_and_excludes_removed_core_scenarios():
    matrix = load("acceptance_matrix.json")
    assert matrix["section_order"] == ["Final-Head Parity", "LEO Adaptations", "LEO Hardening", "Deferred"]
    assert set(matrix["sections"]) == set(matrix["section_order"])
    assert matrix["core_gate_sections"] == matrix["section_order"][:3]
    assert matrix["deferred_is_not_core_gate"] is True
    assert matrix["source_scenario_count"] == 93
    all_items = [item for items in matrix["sections"].values() for item in items]
    ids = [item["id"] for item in all_items]
    assert len(ids) == len(set(ids))
    by_id = {item["id"]: item for item in all_items}
    for scenario_id in matrix["removed_from_core_gate"]:
        assert by_id[scenario_id]["core_gate"] is False
    core = [item for item in all_items if item["core_gate"]]
    prohibited = (
        "save as before", "locale-catalog", "prompt-cache", "quiet hours", "snooze",
        "interruption budget", "scheduler_registered", "registration fails",
    )
    for item in core:
        text = item["then"].lower()
        assert not any(phrase in text for phrase in prohibited), item["id"]
    assert all(item["core_gate"] is False for item in matrix["sections"]["Deferred"])
    assert any(item["id"] == "LEO-ADAPT-02" for item in matrix["sections"]["LEO Adaptations"])
    assert any(item["id"] == "LEO-HARD-REPEAT-CAP" for item in matrix["sections"]["LEO Hardening"])


def test_baseline_receipt_matches_junit_and_records_every_test_outcome():
    receipt = load("baseline/receipt.json")
    assert receipt["schema_version"] == 1 and receipt["phase"] == 0
    assert receipt["commit"] == run_baseline.EXPECTED_COMMIT
    assert receipt["lock_sha256"] == run_baseline.EXPECTED_LOCK_SHA256
    assert receipt["status"] == "passed"
    assert receipt["tracked_changes"] == []
    assert receipt["command"] == [
        ".venv/bin/python", "-m", "pytest", *run_baseline.BASELINE_TESTS,
        "--junitxml=tests/fixtures/schedule_parity/baseline/junit.xml",
    ]
    assert receipt["junit_sha256"] == hashlib.sha256((FIXTURES / "baseline/junit.xml").read_bytes()).hexdigest()
    parsed = run_baseline.parse_junit(FIXTURES / "baseline/junit.xml")
    assert parsed == receipt["tests"]
    assert len(parsed) == 262
    assert len({record["test_id"] for record in parsed}) == 262
    assert {record["outcome"] for record in parsed} == {"passed"}
    assert receipt["summary"] == {
        "total": 262, "passed": 262, "failed": 0, "errors": 0, "skipped": 0,
        "xfailed": 0, "allowed_failures": [], "unexpected_failures": [],
        "allowlist_entries_not_observed": [],
    }
    assert run_baseline.stable_receipt_digest(receipt) == receipt["receipt_sha256"]
    allowlist = load("baseline_failures.json")
    assert allowlist["commit"] == receipt["commit"]
    assert allowlist["allowed_failures"] == []


def _write_junit(path: Path, outcomes: dict[str, str]) -> None:
    suite = ET.Element("testsuite", tests=str(len(outcomes)))
    for name, outcome in outcomes.items():
        case = ET.SubElement(suite, "testcase", classname="tests.test_sample", name=name)
        if outcome in {"failed", "error", "skipped"}:
            tag = "failure" if outcome == "failed" else outcome
            ET.SubElement(case, tag, message=outcome)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def test_baseline_allowlist_rejects_unnamed_failures_and_stale_entries(tmp_path):
    junit = tmp_path / "junit.xml"
    _write_junit(junit, {"passes": "passed", "known": "failed"})
    tests = run_baseline.parse_junit(junit)
    summary, blockers = run_baseline.evaluate_baseline(tests, {"tests.test_sample::known": "pre-existing"})
    assert summary["allowed_failures"] == ["tests.test_sample::known"]
    assert blockers == []
    summary, blockers = run_baseline.evaluate_baseline(tests, {})
    assert summary["unexpected_failures"] == ["tests.test_sample::known"]
    assert blockers == ["tests.test_sample::known"]
    summary, blockers = run_baseline.evaluate_baseline(tests, {"tests.test_sample::missing": "pre-existing"})
    assert summary["allowlist_entries_not_observed"] == ["tests.test_sample::missing"]
    assert blockers


def test_baseline_parser_rejects_duplicate_test_ids(tmp_path):
    junit = tmp_path / "junit.xml"
    suite = ET.Element("testsuite", tests="2")
    for _ in range(2):
        ET.SubElement(suite, "testcase", classname="tests.test_sample", name="duplicate")
    ET.ElementTree(suite).write(junit, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JUnit test id"):
        run_baseline.parse_junit(junit)
