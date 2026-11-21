"""Offline tests for the human-like LEO conversation observation harness.

Every test uses the injected fake transport. No socket is ever opened, no LEO
server is required, and no environment credential is read or logged.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import leo_conversation_observer as obs  # noqa: E402


def _run(tmp_path: Path, **kwargs) -> list[dict]:
    defaults = {
        "messages": 4,
        "allow_short": True,
        "retries": 1,
        "retry_backoff_seconds": 0.0,
    }
    defaults.update(kwargs)
    return obs.run_conversation(obs.FakeTransport(), **defaults)


def test_message_count_floor(tmp_path: Path) -> None:
    """400 is the default and the hard floor; only tests may waive it."""

    assert obs.DEFAULT_MESSAGES == 400
    assert obs.MIN_MESSAGES == 400
    assert len(obs.generate_scenario()) == 400
    with pytest.raises(ValueError, match="400"):
        obs.generate_scenario(399)
    for bad in (0, -1):
        with pytest.raises(ValueError, match=">= 1"):
            obs.generate_scenario(bad)
    with pytest.raises(ValueError):
        obs.generate_scenario("400")  # type: ignore[arg-type]
    assert len(obs.generate_scenario(3, allow_short=True)) == 3

    with pytest.raises(SystemExit) as exc:
        obs.main(["--messages", "399", "--dry-run", "--out-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert not (tmp_path / "conversation.jsonl").exists()


def test_deterministic_scenario() -> None:
    """Same seed -> same script; different seed -> different script; all unique."""

    first = obs.generate_scenario(400)
    second = obs.generate_scenario(400)
    assert first == second
    assert [message.text for message in first] == [message.text for message in second]

    texts = [message.text for message in first]
    assert len(set(texts)) == 400
    assert [message.index for message in first] == list(range(1, 401))
    assert first[0].category == "identity"
    assert "who are you" in first[0].text

    expected_categories = {template.category for template in obs.PROMPT_TEMPLATES}
    assert {message.category for message in first} == expected_categories
    for theme in (
        "remember", "task", "skills", "tools", "agents", "architecture",
        "scratch file", "correct yourself", "failed", "summarize", "simulated",
        "improvement", "How's your day",
    ):
        assert any(theme.lower() in text.lower() for text in texts), theme

    # Changing the seed changes phrasing/topic choices, but the prefix of a
    # shorter run with the same seed is stable.
    assert [message.text for message in obs.generate_scenario(400, seed=7)] != texts
    assert obs.generate_scenario(200, allow_short=True) == first[:200]

    forbidden = ("password", "api key", "secret", "token=", "rm -rf", "sudo", "delete all")
    assert not [text for text in texts if any(word in text.lower() for word in forbidden)]


def test_redaction() -> None:
    """Token-shaped strings are removed from text and nested containers."""

    leak = (
        "Header Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz0123456789 "
        "api_key=hunter2hunter2 secret: 'abcd1234abcd1234' "
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"
    )
    cleaned = obs.redact(leak)
    for fragment in ("sk-abcdefghijklmnopqrstuvwxyz0123456789", "hunter2hunter2", "eyJhbGciOi"):
        assert fragment not in cleaned
    assert obs.REDACTED in cleaned
    assert obs.redact("Nothing sensitive here.") == "Nothing sensitive here."
    assert obs.redact(42) == 42
    nested = obs.redact({"n": [{"token": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}]})
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" not in json.dumps(nested)
    # Redaction is idempotent: writing twice never restores a token.
    assert obs.redact(cleaned) == cleaned


def test_failure_continuation_and_bounded_retries() -> None:
    """A broken turn is retried a bounded number of times, then skipped."""

    transport = obs.FakeTransport(failure_indices=(2, 3))
    turns = obs.run_conversation(
        transport,
        messages=5,
        allow_short=True,
        retries=1,
        retry_backoff_seconds=0.0,
    )
    assert len(turns) == 5
    assert [turn["index"] for turn in turns] == [1, 2, 3, 4, 5]
    failed = [turn for turn in turns if turn["failure"]]
    assert [turn["index"] for turn in failed] == [2, 3]
    assert all(turn["status"] == "transport_error" for turn in failed)
    assert all(turn["assistant_content"] == "" for turn in failed)
    assert all(turn["attempts"] == 2 for turn in failed)
    assert all("connectionreseterror" in turn["error_signature"] for turn in failed)
    assert all(turn["attempts"] == 1 for turn in turns if not turn["failure"])
    # 3 successful turns + 2 failing turns x 2 attempts = 7 POSTs.
    assert len(transport.calls) == 7
    assert transport.calls[0] == obs.generate_scenario(5, allow_short=True)[0].text

    progress: list[int] = []
    obs.run_conversation(
        obs.FakeTransport(failure_indices=(1,)),
        messages=2,
        allow_short=True,
        retries=0,
        progress=lambda turn: progress.append(turn["index"]),
    )
    assert progress == [1, 2]

    with pytest.raises(ValueError, match="retries"):
        obs.run_conversation(obs.FakeTransport(), messages=1, allow_short=True, retries=-1)


def test_extract_tool_info() -> None:
    """Tool names, redacted results and error signatures are separated."""

    names, records, errors = obs.extract_tool_info(
        [
            {"name": "write_note", "result": json.dumps({"ok": True, "path": "artifacts/observation/scratch/a.md"})},
            {"name": "read_file", "result": json.dumps({"error": "not found: note 42"})},
            {"name": "bad", "result": json.dumps({"error": None, "ok": True})},
            {"name": "leaky", "result": "api_key=abcdefghijklmnopqrstuvwxyz0123456"},
        ]
    )
    assert names == ["write_note", "read_file", "bad", "leaky"]
    assert len(records) == 4 and records[1]["name"] == "read_file"
    assert len(errors) == 1
    assert errors[0].startswith("tool:read_file:")
    assert "not found" in errors[0]
    assert "abcdefghijklmnopqrstuvwxyz0123456" not in json.dumps(records)
    long_result = "the quick brown fox " * 250
    _, truncated, _ = obs.extract_tool_info([{"name": "big", "result": long_result}])
    assert len(truncated[0]["result"]) < 4000
    assert truncated[0]["result"].endswith("...[truncated]")
    # Odd-but-plausible shapes are normalized instead of silently reported as
    # "no tools": a dict/string represents one tool result.
    assert obs.extract_tool_info("not-a-list") == (
        ["unknown"], [{"name": "unknown", "result": "not-a-list"}], []
    )
    single = obs.extract_tool_info({"name": "solo", "result": "ok"})
    assert single[0] == ["solo"] and single[1][0]["result"] == "ok"


def test_log_writing_and_summary_correctness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """JSONL, Markdown and summary agree, and no secret reaches any file."""

    env_secret = "sk-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    monkeypatch.setenv("LEO_OBSERVER_UNRELATED_SECRET", env_secret)
    transport = obs.FakeTransport(
        failure_indices=(2,),
        tool_error_indices=(1,),
        statuses=("completed", "incomplete"),
        leak_token=True,
    )
    jsonl_path = tmp_path / "logs" / "conversation.jsonl"
    turns = obs.run_conversation(
        transport,
        messages=4,
        allow_short=True,
        retries=1,
        retry_backoff_seconds=0.0,
        jsonl_path=jsonl_path,
        conversation_id="observer-test-conversation",
    )
    jsonl_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 4
    assert [json.loads(line)["index"] for line in jsonl_lines] == [1, 2, 3, 4]

    summary = obs.build_summary(
        turns,
        conversation_id="observer-test-conversation",
        dry_run=True,
        messages_requested=4,
    )
    jsonl_path_2 = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")

    assert [json.loads(line) for line in jsonl_path_2.read_text(encoding="utf-8").splitlines()] == turns
    assert md_path.read_text(encoding="utf-8").startswith("# LEO conversation observation transcript")
    markdown = md_path.read_text(encoding="utf-8")
    assert "## Turn 1" in markdown and "**User:**" in markdown and "**Assistant:**" in markdown
    assert "tool:read_file" in markdown and "transport_error" in markdown
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary == summary

    for path in (jsonl_path, jsonl_path_2, md_path, summary_path):
        content = path.read_text(encoding="utf-8")
        assert env_secret not in content
        assert "sk-abcdefghijklmnopqrstuvwxyz0123456789" not in content
        assert "hunter2hunter2" not in content

    assert summary["total_turns"] == 4
    assert summary["messages_requested"] == 4
    assert summary["failures"] == 1
    assert summary["tool_error_count"] == 1
    assert summary["tool_invocations"] == 3  # one per successful turn
    assert summary["status_distribution"] == {"completed": 2, "incomplete": 1, "transport_error": 1}
    assert summary["providers"] == {"": 1, "mock": 3}
    assert summary["models"] == {"": 1, "fake-model": 3}
    assert summary["simulated_counts"] == {"simulated": 3, "live": 0, "unknown": 1}
    assert any(pair["provider"] == "mock" and pair["mode"] == "simulated" for pair in summary["provider_model_pairs"])
    expected_latency = round(sum(turn["elapsed_seconds"] for turn in turns) / 4, 4)
    assert summary["average_latency_seconds"] == expected_latency
    signatures = {entry["signature"]: entry["count"] for entry in summary["top_error_signatures"]}
    assert signatures["tool:read_file: error: not found: scratch note #"] == 1
    assert any(signature.startswith("transport:") for signature in signatures)
    assert summary["dry_run"] is True
    assert summary["conversation_id"] == "observer-test-conversation"


def test_dry_run_never_opens_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--dry-run`` must be usable with no server and no network."""

    def forbidden(*args, **kwargs):  # pragma: no cover - failure is the assertion
        raise AssertionError("network access attempted during dry run")

    monkeypatch.setattr(obs.urllib.request, "urlopen", forbidden)
    assert obs.main(["--dry-run", "--allow-short", "--messages", "2", "--out-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["total_turns"] == 2


def test_cli_dry_run_default_400(tmp_path: Path) -> None:
    """Without --messages the CLI sends exactly the default 400 turns."""

    assert obs.main(["--dry-run", "--out-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_turns"] == 400
    assert summary["messages_requested"] == 400
    assert len((tmp_path / "conversation.jsonl").read_text(encoding="utf-8").splitlines()) == 400
    assert summary["failures"] == 0
    assert "## Turn 400" in (tmp_path / "conversation.md").read_text(encoding="utf-8")
    # The fake server returns 100 incomplete (but non-empty) answers and one
    # completed turn with a pending tool. Neither is a clean success now.
    assert summary["successful_turns"] == 299
    assert summary["deficiency_count"] == 101
    assert summary["deficiency_distribution"]["incomplete_status"] == 100
    assert summary["deficiency_distribution"]["incomplete_with_pending_tools"] == 1
    assert summary["floor_satisfied"] is False
    assert summary["run_verdict"] == "below_floor_or_incomplete"


# ---------------------------------------------------------------------------
# Regressions for the adversarial review findings
# ---------------------------------------------------------------------------


def test_metadata_token_is_redacted_in_all_outputs_and_turns(tmp_path: Path) -> None:
    """A token injected into EVERY server-controlled metadata field must vanish.

    The Markdown writer previously interpolated status/provider/model/model_status
    and tool names directly. This regression covers all three output files, the
    returned turns, the summary distributions and tool error signatures.
    """

    token = "LEAKTOKENabcdefghijklmnopqrstuvwxyz01"  # matches the 32+ char rule
    transport = obs.FakeTransport(
        metadata_leak_token=token,
        leak_token=True,
        tool_error_indices=(1,),
        statuses=("incomplete",),
        dry_run=True,
    )
    checkpoint = tmp_path / "checkpoints" / "conversation.jsonl"
    turns = obs.run_conversation(
        transport, messages=3, allow_short=True, retries=0, jsonl_path=checkpoint
    )
    summary = obs.build_summary(turns, dry_run=True, messages_requested=3)
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    jsonl_path = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")

    for path in (checkpoint, md_path, jsonl_path, summary_path):
        assert token not in path.read_text(encoding="utf-8"), path
    assert token not in json.dumps(turns)
    assert token not in json.dumps(summary)
    for turn in turns:
        for key in ("status", "provider", "model", "model_status", "command", "command_status"):
            assert token not in str(turn[key]), key
        assert all(token not in name for name in turn["tool_names"])
        assert all(token not in record["name"] for record in turn["tool_results"])
        assert all(token not in str(item) for item in turn["tool_errors"])
        assert all(token not in str(item) for item in turn["memories_used"])
        assert all(token not in str(item) for item in turn["tool_calls_pending"])
        assert token not in json.dumps(turn["command_result"])
    assert token not in json.dumps(summary["status_distribution"])
    assert token not in json.dumps(summary["providers"])
    assert token not in json.dumps(summary["models"])
    assert token not in json.dumps(summary["top_error_signatures"])
    # Redaction must not hide legitimate observations: the transcript still
    # reports the incomplete status and the tool error.
    markdown = md_path.read_text(encoding="utf-8")
    assert "[REDACTED]" in markdown
    assert "## Top error signatures" in markdown
    assert summary["tool_error_count"] == 1


def test_allow_short_is_dry_run_only_and_floor_requires_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live runs cannot waive the floor; 400 failures are not 400 observations."""

    def forbidden(*args, **kwargs):  # pragma: no cover - failure is the assertion
        raise AssertionError("a live request was attempted")

    monkeypatch.setattr(obs.urllib.request, "urlopen", forbidden)
    with pytest.raises(SystemExit) as exc:
        obs.main(["--messages", "3", "--allow-short", "--out-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert not (tmp_path / "conversation.jsonl").exists()

    all_failed = obs.run_conversation(
        obs.FakeTransport(failure_indices=tuple(range(1, 401))),
        messages=400,
        retries=0,
        retry_backoff_seconds=0.0,
    )
    summary = obs.build_summary(all_failed, messages_requested=400)
    assert summary["total_turns"] == 400
    assert summary["successful_turns"] == 0
    assert summary["failures"] == 400
    assert summary["floor_satisfied"] is False
    assert summary["deficiency_count"] == 400
    assert summary["deficiency_distribution"]["transport_failure"] == 400

    partial = obs.run_conversation(
        obs.FakeTransport(failure_indices=(1, 2)),
        messages=400,
        retries=0,
        retry_backoff_seconds=0.0,
    )
    partial_summary = obs.build_summary(partial, messages_requested=400)
    assert partial_summary["successful_turns"] == 398
    assert partial_summary["floor_satisfied"] is False
    assert partial_summary["deficiency_count"] == 2

    short_summary = obs.build_summary(
        obs.run_conversation(obs.FakeTransport(), messages=2, allow_short=True, retries=0),
        messages_requested=2,
        config={"allow_short": True},
    )
    assert short_summary["total_turns"] == 2
    assert short_summary["floor_satisfied"] is False
    assert short_summary["config"]["allow_short"] is True


class ScriptedResponseTransport:
    """Tiny transport with exact control over statuses and content."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def send(self, *, message: str, conversation_id: str, timeout_seconds: float) -> dict:
        self.calls.append(message)
        index = len(self.calls)
        statuses = ("incomplete", "blocked", "completed", "error")
        contents = ("", "blocked by policy", "a real answer", "")
        return {
            "conversation_id": conversation_id,
            "content": contents[index - 1],
            "status": statuses[index - 1],
            "provider": "scripted",
            "model": "scripted-model",
            "simulated": False,
            "model_status": "live",
            # Deliberately awkward usage object: must not abort the run.
            "usage": {"prompt_tokens": 1, "details": {1, 2, 3}, "ratio": float("nan")},
            "memories_used": [f"mem-{index}", {"id": f"m{index}", "raw": {1, 2}}],
            "tool_results": [],
            "tool_calls_pending": [f"pending-{index}"] if index == 1 else [],
            "command": "/status" if index == 3 else "",
            "command_status": "ok" if index == 3 else "",
            "command_success": True if index == 3 else None,
            "command_result": {"code": 0, "raw": {4, 5}} if index == 3 else None,
        }


def test_response_fields_and_deficiencies_are_recorded(tmp_path: Path) -> None:
    """memories/pending/slash fields survive; empty incomplete turns count as deficiencies."""

    turns = obs.run_conversation(
        ScriptedResponseTransport(), messages=4, allow_short=True, retries=0
    )
    summary = obs.build_summary(turns, messages_requested=4)

    assert turns[0]["memories_used"][0] == "mem-1"
    assert turns[0]["memories_used"][1] == {"id": "m1", "raw": "{1, 2}"}  # nested set normalized
    assert turns[0]["tool_calls_pending"] == ["pending-1"]
    assert turns[2]["command"] == "/status"
    assert turns[2]["command_status"] == "ok"
    assert turns[2]["command_success"] is True
    assert turns[2]["command_result"]["code"] == 0
    # Every response field, including awkward usage values, is JSON-safe.
    for turn in turns:
        json.dumps(turn, allow_nan=False)
        assert isinstance(turn["usage"], dict)

    assert turns[0]["deficiencies"] == [
        "incomplete_status", "incomplete_without_content", "incomplete_with_pending_tools",
    ]
    assert turns[1]["deficiencies"] == ["blocked_status"]
    assert turns[2]["deficiencies"] == []
    assert turns[3]["deficiencies"] == ["error_status"]
    assert obs.is_successful_turn(turns[0]) is False
    assert obs.is_successful_turn(turns[1]) is False
    assert obs.is_successful_turn(turns[2]) is True

    assert summary["successful_turns"] == 1
    assert summary["deficiency_count"] == 3
    assert summary["deficiency_distribution"] == {
        "blocked_status": 1,
        "error_status": 1,
        "incomplete_status": 1,
        "incomplete_without_content": 1,
        "incomplete_with_pending_tools": 1,
    }
    assert summary["memories_used_count"] == 8
    assert summary["memories_used_turns"] == 4
    assert summary["pending_tool_calls"] == 1
    assert summary["pending_tool_call_turns"] == 1
    assert summary["slash_command_turns"] == 1
    assert summary["command_success_count"] == 1
    assert summary["command_failure_count"] == 0
    assert summary["command_status_distribution"] == {"ok": 1}
    assert summary["floor_satisfied"] is False

    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    markdown = md_path.read_text(encoding="utf-8")
    assert "Memories used:" in markdown
    assert "Pending tool calls:" in markdown
    assert "- Command: /status" in markdown
    assert "Deficiencies: ['incomplete_status', 'incomplete_without_content', 'incomplete_with_pending_tools']" in markdown


def test_retry_history_progress_isolation_and_completeness_checks(tmp_path: Path) -> None:
    """Per-attempt signatures, throwing callbacks and missing files are handled."""

    transport = obs.FakeTransport(failure_indices=(1,))
    turns = obs.run_conversation(
        transport,
        messages=2,
        allow_short=True,
        retries=2,
        retry_backoff_seconds=0.0,
    )
    failed = turns[0]
    assert failed["attempts"] == 3
    assert [entry["attempt"] for entry in failed["attempt_history"]] == [1, 2, 3]
    assert len(failed["attempt_errors"]) == 3
    assert all(entry.startswith("transport:") for entry in failed["attempt_errors"])
    assert failed["error_signature"] == failed["attempt_errors"][-1]
    assert all("elapsed_seconds" in entry for entry in failed["attempt_history"])
    summary = obs.build_summary(turns, messages_requested=2)
    assert summary["retry_count"] == 2
    assert summary["attempt_error_count"] == 3

    def broken_progress(turn: dict) -> None:
        raise RuntimeError("progress callback exploded")

    progress_turns = obs.run_conversation(
        obs.FakeTransport(),
        messages=2,
        allow_short=True,
        retries=0,
        progress=broken_progress,
        jsonl_path=tmp_path / "conversation.jsonl",
    )
    assert len(progress_turns) == 2
    assert all(turn["progress_error"].startswith("progress:") for turn in progress_turns)
    progress_summary = obs.build_summary(progress_turns, messages_requested=2)
    assert progress_summary["progress_error_count"] == 2
    assert all(
        json.loads(line)["progress_error"].startswith("progress:")
        for line in (tmp_path / "conversation.jsonl").read_text(encoding="utf-8").splitlines()
    )

    jsonl_path = obs.write_jsonl(progress_turns, tmp_path / "complete.jsonl")
    md_path = obs.write_markdown(progress_turns, progress_summary, tmp_path / "complete.md")
    summary_path = obs.write_summary(progress_summary, tmp_path / "complete.json")
    report = obs.check_output_completeness(
        {"jsonl": jsonl_path, "markdown": md_path, "summary": summary_path},
        progress_turns,
        progress_summary,
    )
    assert report["complete"] is True and report["issue_count"] == 0

    # A truncated log and a deleted transcript must be detected after the run.
    (tmp_path / "broken.jsonl").write_text(
        (tmp_path / "complete.jsonl").read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    broken_report = obs.check_output_completeness(
        {"jsonl": tmp_path / "broken.jsonl", "markdown": tmp_path / "missing.md"},
        progress_turns,
        progress_summary,
    )
    assert broken_report["complete"] is False
    assert any("jsonl" in issue for issue in broken_report["issues"])
    assert any("markdown" in issue for issue in broken_report["issues"])


def test_url_credentials_scrubbed_from_config_and_scratch_paths_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Userinfo in --url never reaches logs; the scratch dir and paths are audited."""

    raw_url = "http://observer:supersecretpw@127.0.0.1:8891/chat?token=abcdefghijklmnop"
    assert obs.sanitize_url(raw_url) == "http://127.0.0.1:8891/chat"
    assert obs.sanitize_url("not a url") == "not a url"

    seen: dict[str, str] = {}

    class SpyTransport(obs.FakeTransport):
        def __init__(self, url: str, **kwargs) -> None:
            seen["url"] = url
            super().__init__(dry_run=False, **kwargs)

        def send(self, **kwargs) -> dict:
            # Emulate a real tool that actually creates the claimed file.
            scratch = Path(getattr(self, "scratch_path", ""))
            if scratch.name:
                scratch.parent.mkdir(parents=True, exist_ok=True)
                scratch.write_text("scratch note\n", encoding="utf-8")
            return super().send(**kwargs)

    monkeypatch.setattr(obs, "HttpChatTransport", SpyTransport)
    assert obs.main(["--url", raw_url, "--out-dir", str(tmp_path)]) == 0
    assert seen["url"] == raw_url  # the wire transport still gets the real URL

    summary_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "conversation.md"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "supersecretpw" not in summary_path.read_text(encoding="utf-8")
    assert "observer:" not in summary_path.read_text(encoding="utf-8")
    assert "supersecretpw" not in capsys.readouterr().out
    assert summary["config"]["url"] == "http://127.0.0.1:8891/chat"
    assert summary["config"]["transport"] == "http"
    # Turn 1 still has a pending tool call, so 399 answers are not 400 clean
    # answers even though no transport attempt failed.
    assert summary["successful_turns"] == 399
    assert summary["deficiency_distribution"] == {"incomplete_with_pending_tools": 1}
    assert summary["floor_satisfied"] is False
    assert summary["run_verdict"] == "below_floor_or_incomplete"
    assert summary["output_check"]["complete"] is True

    scratch_dir = tmp_path / "scratch"
    assert scratch_dir.is_dir()
    path_report = summary["tool_result_paths"]
    assert path_report["scratch_directory_created"] is True
    assert path_report["simulated"] is False
    assert path_report["claim_count"] >= 400
    assert path_report["clean"] is True
    assert all(claim["location"] == "scratch" for claim in path_report["claims"])
    assert "DRY-RUN" not in markdown_path.read_text(encoding="utf-8")

    outside = [
        {
            "index": 1,
            "tool_results": [{"name": "write_note", "result": json.dumps({"path": "/etc/passwd"})}],
        }
    ]
    outside_report = obs.check_tool_result_paths(outside, out_dir=tmp_path, dry_run=False)
    assert outside_report["clean"] is False
    assert any("outside workspace" in issue for issue in outside_report["issues"])


def test_dry_run_scratch_marker_and_scenario_variety(tmp_path: Path) -> None:
    """Dry runs label simulated writes; follow-up suffixes are now a last resort."""

    texts = [message.text for message in obs.generate_scenario(400)]
    suffixes = [text for text in texts if "(follow-up " in text]
    assert len(suffixes) <= 5, len(suffixes)
    comma_openers = [opener for opener in obs.OPENERS if opener.endswith(", ")]
    assert len(obs.TOPICS) >= 20 and len(obs.OPENERS) >= 12
    for opener in comma_openers:
        for text in texts:
            if text.startswith(opener):
                remainder = text[len(opener):]
                assert not remainder[:1].isupper(), text
    assert not any("By the way, Hey" in text for text in texts)

    transport = obs.FakeTransport(dry_run=True)
    turns = obs.run_conversation(transport, messages=1, allow_short=True, retries=0)
    result = json.loads(turns[0]["tool_results"][0]["result"])
    assert result["dry_run"] is True
    assert "DRY-RUN" in result["note"]
    summary = obs.build_summary(turns, dry_run=True, messages_requested=1)
    markdown = obs.write_markdown(turns, summary, tmp_path / "conversation.md").read_text(encoding="utf-8")
    assert "Scratch writes: SIMULATED (dry run; no scratch file was created)" in markdown
    assert "DRY-RUN: simulated scratch write" in markdown


# ---------------------------------------------------------------------------
# Regressions for the second adversarial review (F1-F5)
# ---------------------------------------------------------------------------


def test_camel_case_keys_auth_schemes_cookies_and_short_secrets() -> None:
    """Credentials are redacted by key name even when short, numeric or camelCase."""

    cases = (
        ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA=="),
        ('"Authorization": "Token s3cr3t!x9"', "s3cr3t!x9"),
        ("Authorization: Digest 9988776655", "9988776655"),
        ("authToken=s3cr3t!x9", "s3cr3t!x9"),
        ("client_secret = ab12CD34", "ab12CD34"),
        ("session_id=9988776655", "9988776655"),
        ("Cookie: session_id=9988776655", "9988776655"),
        ('{"sessionId": "9988776655", "apiKey": "ab12CD34"}', "9988776655"),
        ('{"authToken": "s3cr3t!x9"}', "s3cr3t!x9"),
    )
    for text, secret in cases:
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text

    nested = obs.redact(
        {
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            "authToken": "s3cr3t!x9",
            "client_secret": "ab12CD34",
            "session_id": 9988776655,
            "Cookie": "session_id=9988776655",
            "nested": {"meta": {"tag": "ab12CD34"}},
            "tokenizer": "not a credential",
            "authenticated": True,
        }
    )
    assert nested["usage"] == {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    assert nested["tokenizer"] == "not a credential"
    assert nested["authenticated"] is True
    assert nested["authToken"] == nested["client_secret"] == obs.REDACTED
    assert nested["session_id"] == nested["Cookie"] == obs.REDACTED
    assert nested["nested"]["meta"]["tag"] == obs.REDACTED

    # Benign observation data with hyphens, slashes and digits must survive.
    for benign in (
        "observer-human-400", "fake-model", "artifacts/observation/scratch/note.md",
        "ConnectionResetError: simulated disconnect", "http://127.0.0.1:8891/chat",
        "DRY-RUN: simulated scratch write",
    ):
        assert obs.redact(benign) == benign


class CredentialLeakTransport(obs.FakeTransport):
    """Inject credentials through every server-controlled surface the CLI logs."""

    def send(self, **kwargs) -> dict:
        data = super().send(**kwargs)
        data["conversation_id"] = "server-assigned-conversation"
        data["status"] = "completed Authorization: Basic dXNlcjpwYXNzd29yZA=="
        data["provider"] = 'authToken="s3cr3t!x9"'
        data["model"] = "client_secret=ab12CD34"
        data["model_status"] = "session_id=9988776655"
        data["command_status"] = "Cookie: session_id=9988776655"
        data["memories_used"] = [
            {"authToken": "s3cr3t!x9", "client_secret": "ab12CD34", "session_id": 9988776655}
        ]
        data["tool_results"] = [
            {
                "name": "write_note",
                "result": json.dumps(
                    {
                        "authorization": "Basic dXNlcjpwYXNzd29yZA==",
                        "cookie": "session_id=9988776655",
                    }
                ),
            }
        ]
        return data


def test_credentials_never_leak_in_stdout_or_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: no credential reaches stdout, JSONL, Markdown, summary or distributions."""

    monkeypatch.setattr(obs, "FakeTransport", CredentialLeakTransport)
    assert obs.main(
        ["--dry-run", "--allow-short", "--messages", "3", "--out-dir", str(tmp_path)]
    ) == 0
    stdout = capsys.readouterr().out
    files = {
        name: (tmp_path / name).read_text(encoding="utf-8")
        for name in ("conversation.jsonl", "conversation.md", "summary.json")
    }
    secrets = ("dXNlcjpwYXNzd29yZA==", "s3cr3t!x9", "ab12CD34", "9988776655")
    for label, text in {"stdout": stdout, **files}.items():
        for secret in secrets:
            assert secret not in text, (label, secret)
    summary = json.loads(files["summary.json"])
    assert json.loads(stdout) == summary
    for distribution in (
        summary["status_distribution"], summary["providers"], summary["models"],
        summary["simulated_counts"], summary["provider_model_pairs"],
        summary["top_error_signatures"], summary["category_distribution"],
    ):
        for secret in secrets:
            assert secret not in json.dumps(distribution), secret
    assert obs.REDACTED in files["conversation.md"]

    # The server-assigned id is recorded, not silently replaced by our config.
    turns = [json.loads(line) for line in files["conversation.jsonl"].splitlines()]
    assert [turn["response_conversation_id"] for turn in turns] == [
        "server-assigned-conversation"
    ] * 3
    assert summary["conversation_id_mismatches"] == 3
    assert "Response conversation id: server-assigned-conversation" in files["conversation.md"]


class HostileUsageTransport:
    """Alternate awkward-but-plausible / hostile response shapes."""

    def __init__(self) -> None:
        self.calls = 0

    def send(self, *, message: str, conversation_id: str, timeout_seconds: float) -> dict:
        self.calls += 1
        if self.calls == 1:
            usage: dict = {
                "prompt_tokens": 12,
                "completion_tokens": float("nan"),
                "total_tokens": float("inf"),
                "details": {"ok": True},
            }
            tool_results = {"name": "solo", "result": "ok"}
            memories_used = "one-memory"
        else:
            usage = {"prompt_tokens": float("-inf"), "raw": {1, 2}}
            tool_results = "plain-string-result"
            memories_used = {"id": "m1"}
        return {
            "conversation_id": conversation_id,
            "content": "an answer",
            "status": "completed",
            "usage": usage,
            "tool_results": tool_results,
            "memories_used": memories_used,
        }


def test_hostile_usage_and_response_shapes_do_not_abort(tmp_path: Path) -> None:
    """NaN/Infinity, odd tool/memory shapes and bad __str__ stay per-turn data."""

    checkpoint = tmp_path / "checkpoint.jsonl"
    turns = obs.run_conversation(
        HostileUsageTransport(), messages=3, allow_short=True, retries=0,
        jsonl_path=checkpoint,
    )
    assert len(turns) == 3
    assert all(not turn["failure"] for turn in turns)
    assert turns[0]["usage"]["prompt_tokens"] == 12
    assert turns[0]["usage"]["completion_tokens"] is None
    assert turns[0]["usage"]["total_tokens"] is None
    assert turns[0]["usage"]["details"] == {"ok": True}
    assert turns[1]["usage"]["prompt_tokens"] is None
    assert turns[0]["tool_names"] == ["solo"]
    assert turns[1]["tool_names"] == ["unknown"]
    assert turns[0]["memories_used"] == ["one-memory"]
    assert turns[1]["memories_used"] == [{"id": "m1"}]
    for turn in turns:
        json.dumps(turn, allow_nan=False)  # strictly serializable
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["index"] for line in lines] == [1, 2, 3]
    summary = obs.build_summary(turns, messages_requested=3)
    assert summary["successful_turns"] == 3
    assert summary["tool_invocations"] == 3


class Unstringifiable:
    def __str__(self) -> str:  # pragma: no cover - called only to be caught
        raise RuntimeError("cannot stringify")

    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("cannot repr")


def test_jsonl_serialization_fallback_keeps_the_run_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-turn write failure produces a safe fallback line, not an abort."""

    real_dumps = obs._json_dumps

    def flaky(value):
        if isinstance(value, dict) and "hostile" in value:
            raise TypeError("simulated serialization failure")
        return real_dumps(value)

    monkeypatch.setattr(obs, "_json_dumps", flaky)

    def inject_hostile(turn: dict) -> None:
        if turn["index"] == 1:
            turn["hostile"] = Unstringifiable()

    checkpoint = tmp_path / "checkpoint.jsonl"
    turns = obs.run_conversation(
        obs.FakeTransport(), messages=3, allow_short=True, retries=0,
        progress=inject_hostile, jsonl_path=checkpoint,
    )
    assert len(turns) == 3
    lines = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    assert [record["index"] for record in lines] == [1, 2, 3]
    assert lines[0]["deficiencies"] == ["jsonl_serialization_failure"]
    assert lines[0]["error_signature"].startswith("jsonl: typeerror")
    assert [lines[1]["index"], lines[2]["index"]] == [2, 3]
    assert lines[1]["deficiencies"] == [] and lines[2]["deficiencies"] == []
    obs.write_jsonl(turns, tmp_path / "final.jsonl")
    assert len((tmp_path / "final.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_stdout_and_summary_report_all_failures_authoritatively(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run of nothing but failures must never look success-like anywhere."""

    assert obs.main(
        [
            "--dry-run", "--allow-short", "--messages", "6",
            "--dry-run-fail-at", "1,2,3,4,5,6", "--out-dir", str(tmp_path),
        ]
    ) == 0
    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    assert summary["total_turns"] == 6
    assert summary["successful_turns"] == 0
    assert summary["failures"] == 6
    assert summary["deficiency_count"] == 6
    assert summary["deficiency_distribution"] == {"transport_failure": 6}
    assert summary["floor_satisfied"] is False
    assert summary["run_verdict"] == "all_turns_failed"
    assert summary["honest_accounting"] == {
        "recorded_turns": 6, "successful_turns": 0, "failed_turns": 6,
        "transport_failures": 6, "error_status_turns": 6,
        "deficient_turns": 6, "floor_size": 400, "floor_satisfied": False,
        "all_answers_failed": True, "verdict": "all_turns_failed",
    }
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["run_verdict"] == summary["run_verdict"]
    assert "- Run verdict: all_turns_failed" in (tmp_path / "conversation.md").read_text(encoding="utf-8")

    # Stale/lying per-turn fields cannot inflate success in a summary rebuild.
    turns = obs.run_conversation(obs.FakeTransport(), messages=2, allow_short=True, retries=0)
    for turn in turns:
        turn["deficiencies"] = []
        turn["failure"] = True
        turn["assistant_content"] = "claim of success"
        turn["status"] = "completed"
    rebuilt = obs.build_summary(turns, messages_requested=2)
    assert rebuilt["successful_turns"] == 0
    assert rebuilt["deficiency_count"] == 2
    assert rebuilt["deficiency_distribution"] == {
        "transport_failure": 2, "incomplete_with_pending_tools": 1,
    }
    assert rebuilt["run_verdict"] == "all_turns_failed"


# ---------------------------------------------------------------------------
# Regressions for the third adversarial review (case, encoding, params, verdict)
# ---------------------------------------------------------------------------


def test_case_insensitive_auth_schemes_in_text_and_error_signatures() -> None:
    """Lowercase and mixed-case schemes redact like their canonical spellings."""

    cases = (
        ("digest 9988776655", "9988776655"),
        ("bearer abcd1234", "abcd1234"),
        ("basic abcdef!ghijkl", "abcdef!ghijkl"),
        ('Authorization: digest 9988776655, nonce="a1b2c3d4"', "9988776655"),
        ('"Authorization": "BeArEr abcd1234"', "abcd1234"),
        ("session: bAsIc abcdef!ghijkl", "abcdef!ghijkl"),
    )
    for text, secret in cases:
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text

    nested = obs.redact(
        {
            "content": "debug: digest 9988776655",
            "memories_used": ["memory: bearer abcd1234"],
            "metadata": {"authorization": "basic abcdef!ghijkl"},
        }
    )
    assert "9988776655" not in nested["content"] and obs.REDACTED in nested["content"]
    assert all(
        "abcd1234" not in item and obs.REDACTED in item for item in nested["memories_used"]
    )
    assert nested["metadata"] == {"authorization": obs.REDACTED}

    for message in (
        "RuntimeError: bearer abcd1234",
        "RuntimeError: digest 9988776655",
        "RuntimeError: basic abcdef!ghijkl",
    ):
        signature = obs.error_signature("RuntimeError", message)
        for secret in ("abcd1234", "9988776655", "abcdef!ghijkl"):
            assert secret not in signature, signature
        assert signature.lower().startswith("runtimeerror:")


class LowercaseSchemeLeakTransport(obs.FakeTransport):
    """Inject lowercase/mixed-case schemes into every server-controlled surface."""

    def send(self, **kwargs) -> dict:
        index = len(self.calls) + 1
        if index == 2:
            raise RuntimeError(
                "RuntimeError: bearer abcd1234, digest 9988776655, basic abcdef!ghijkl"
            )
        data = super().send(**kwargs)
        data["content"] += (
            ' debug: digest 9988776655, nonce="a1b2c3d4", response="deadbeef"; '
            "bearer abcd1234; BASIC abcdef!ghijkl"
        )
        data["status"] = "completed basic abcdef!ghijkl"
        data["provider"] = "digest 9988776655"
        data["model"] = "BeArEr abcd1234"
        data["model_status"] = "DiGeSt 9988776655"
        data["command"] = "bearer abcd1234"
        data["command_status"] = "basic abcdef!ghijkl"
        data["memories_used"] = ["memory: digest 9988776655", {"authorization": "bearer abcd1234"}]
        data["tool_results"] = [
            {
                "name": "read_file",
                "result": json.dumps(
                    {
                        "authorization": "basic abcdef!ghijkl",
                        "note": "digest 9988776655",
                    }
                ),
            }
        ]
        return data


def test_case_insensitive_schemes_never_reach_any_output(tmp_path: Path) -> None:
    """End-to-end: lowercase schemes in content, memory, metadata and errors."""

    secrets = ("9988776655", "abcd1234", "abcdef!ghijkl", "a1b2c3d4", "deadbeef")
    checkpoint = tmp_path / "checkpoint.jsonl"
    turns = obs.run_conversation(
        LowercaseSchemeLeakTransport(),
        messages=3,
        allow_short=True,
        retries=0,
        jsonl_path=checkpoint,
    )
    summary = obs.build_summary(turns, messages_requested=3, dry_run=True)
    jsonl_path = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")

    for path in (checkpoint, jsonl_path, md_path, summary_path):
        text = path.read_text(encoding="utf-8")
        for secret in secrets:
            assert secret not in text, (path.name, secret)
    for label, value in (("turns", turns), ("summary", summary)):
        for secret in secrets:
            assert secret not in json.dumps(value), (label, secret)

    assert turns[1]["failure"] is True
    assert all(secret not in turns[1]["error_signature"] for secret in secrets)
    assert all(
        secret not in json.dumps(turn["memories_used"])
        for turn in turns
        for secret in secrets
    )
    for field in ("providers", "models", "status_distribution"):
        for secret in secrets:
            assert secret not in json.dumps(summary[field]), (field, secret)
    assert "[REDACTED]" in md_path.read_text(encoding="utf-8")


def test_escaped_json_keys_and_short_secrets_are_redacted() -> None:
    """Double-encoded keys cannot hide short, numeric or nested credentials."""



    cases = (
        (r'{\"authToken\": \"9988776655\"}', "9988776655"),
        (r'{\"client_secret\": \"ab12\"}', "ab12"),
        (r'{\"sessionId\": 1234}', "1234"),
        (r'{\"outer\": {\"apiKey\": \"q7r\"}}', "q7r"),
        (r'prefix {\"authorization\": \"xy9\"} suffix', "xy9"),
    )
    for text, secret in cases:
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text

    nested = obs.redact({"meta": r'{\"client_secret\": \"ab12\"}'})
    assert "ab12" not in json.dumps(nested)


class EncodedJsonLeakTransport(obs.FakeTransport):
    """Inject escaped/double-encoded JSON keys through every response surface."""

    def send(self, **kwargs) -> dict:
        data = super().send(**kwargs)
        data["content"] += r' debug header {\"authToken\": \"9988776655\"}'
        data["memories_used"] = [r'memory {\"client_secret\": \"ab12\"}']
        data["provider"] = r'provider {\"sessionId\": 9988776655}'
        data["model_status"] = r'model {\"apiKey\": \"q7r\"}'
        data["status"] = "completed"
        data["tool_results"] = [
            {
                "name": "read_file",
                "result": json.dumps({"error": r'{\"authorization\": \"xy9\"}'}),
            }
        ]
        return data


def test_escaped_json_keys_never_reach_any_output(tmp_path: Path) -> None:
    """End-to-end: encoded sensitive keys in content, memory, metadata, tools."""

    secrets = ("9988776655", "ab12", "q7r", "xy9", "1234")
    checkpoint = tmp_path / "checkpoint.jsonl"
    turns = obs.run_conversation(
        EncodedJsonLeakTransport(),
        messages=3,
        allow_short=True,
        retries=0,
        jsonl_path=checkpoint,
    )
    summary = obs.build_summary(turns, messages_requested=3, dry_run=True)
    jsonl_path = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")

    for path in (checkpoint, jsonl_path, md_path, summary_path):
        text = path.read_text(encoding="utf-8")
        for secret in secrets:
            assert secret not in text, (path.name, secret)
    for label, value in (("turns", turns), ("summary", summary)):
        for secret in secrets:
            assert secret not in json.dumps(value), (label, secret)


def test_digest_parameter_lists_are_consumed_in_full() -> None:
    """Every auth parameter, quoted or not and in either order, disappears."""

    cases = (
        (
            'Authorization: Digest 9988776655, nonce="a1b2c3d4", response="deadbeef", '
            'qop=auth, nc=00000001, cnonce="xyz789", opaque="op1234"',
            ("9988776655", "a1b2c3d4", "deadbeef", "xyz789", "op1234", "00000001"),
        ),
        (
            'Authorization: Digest username="alice", realm="api", nonce="n1a2b3", '
            'uri="/chat", response="r1e2s3", opaque="o1p2q3"',
            ("n1a2b3", "r1e2s3", "o1p2q3"),
        ),
        (
            'Authorization: digest nonce="n1a2b3", response="r1e2s3", username="alice", '
            'cnonce="c1n2o3"',
            ("n1a2b3", "r1e2s3", "c1n2o3"),
        ),
        (
            "authorization: Digest 9988776655, cnonce=c1n2o3, response=r1e2s3, opaque=o1p2q3",
            ("9988776655", "c1n2o3", "r1e2s3", "o1p2q3"),
        ),
    )
    for text, secrets in cases:
        cleaned = obs.redact(text)
        for secret in secrets:
            assert secret not in cleaned, (text, secret)
        assert obs.REDACTED in cleaned
        assert obs.redact(cleaned) == cleaned

    signature = obs.error_signature(
        "RuntimeError",
        'RuntimeError: digest 9988776655, nonce="a1b2c3d4", response="deadbeef"',
    )
    for secret in ("9988776655", "a1b2c3d4", "deadbeef"):
        assert secret not in signature


class StatusOnlyTransport:
    """A transport that returns a fixed status sequence without any failures."""

    def __init__(self, statuses: tuple[str, ...]) -> None:
        self.statuses = statuses
        self.calls = 0

    def send(self, *, message: str, conversation_id: str, timeout_seconds: float) -> dict:
        status = self.statuses[self.calls]
        self.calls += 1
        return {
            "conversation_id": conversation_id,
            "content": "a real answer" if status == "completed" else "",
            "status": status,
            "provider": "mock",
            "model": "fake-model",
            "tool_results": [],
        }


def test_all_error_status_runs_are_labeled_as_all_fail(tmp_path: Path) -> None:
    """Status errors are failures even when no transport exception occurred."""

    turns = obs.run_conversation(
        StatusOnlyTransport(("error", "failed", "exception", "error")),
        messages=4,
        allow_short=True,
        retries=0,
    )
    summary = obs.build_summary(turns, messages_requested=4)

    assert summary["successful_turns"] == 0
    assert summary["failures"] == 0  # transport failures remain a separate count
    assert summary["failed_turns"] == 4
    assert summary["error_status_turns"] == 4
    assert summary["deficiency_distribution"] == {"error_status": 4}
    assert summary["all_answers_failed"] is True
    assert summary["run_verdict"] == "all_answers_failed"
    assert summary["floor_satisfied"] is False
    assert summary["honest_accounting"] == {
        "recorded_turns": 4,
        "successful_turns": 0,
        "failed_turns": 4,
        "transport_failures": 0,
        "error_status_turns": 4,
        "deficient_turns": 4,
        "floor_size": 400,
        "floor_satisfied": False,
        "all_answers_failed": True,
        "verdict": "all_answers_failed",
    }
    markdown = obs.write_markdown(turns, summary, tmp_path / "conversation.md").read_text(
        encoding="utf-8"
    )
    assert "- Run verdict: all_answers_failed" in markdown
    assert "- All answers failed: True" in markdown

    mixed = obs.build_summary(
        obs.run_conversation(
            StatusOnlyTransport(("completed", "error")),
            messages=2,
            allow_short=True,
            retries=0,
        ),
        messages_requested=2,
    )
    assert mixed["successful_turns"] == 1
    assert mixed["failed_turns"] == 1
    assert mixed["all_answers_failed"] is False
    assert mixed["run_verdict"] == "below_floor_or_incomplete"


# ---------------------------------------------------------------------------
# Regressions for the residual leak classes (F1-F5)
# ---------------------------------------------------------------------------


def test_semicolon_and_terminated_auth_parameter_lists_are_redacted() -> None:
    """F1: `;`-joined params and glued terminators cannot leave credentials."""

    cases = (
        (
            'Authorization: Digest 9988776655; nonce="a1b2c3d4"; response="deadbeef"; qop=auth',
            ("9988776655", "a1b2c3d4", "deadbeef"),
        ),
        (
            "Authorization: Negotiate TlRMTVNTUAAB; qop=auth; nc=00000001; opaque=op1234",
            ("TlRMTVNTUAAB", "00000001", "op1234"),
        ),
        (
            'Proxy-Authorization: digest nonce=n1a2b3; response=r1e2s3; cnonce=c1n2o3',
            ("n1a2b3", "r1e2s3", "c1n2o3"),
        ),
        (
            r'Authorization: Digest username=\"alice\"; nonce=\"n1a2b3\"; response=\"r1e2s3\"',
            ("n1a2b3", "r1e2s3"),
        ),
        ("Authorization: Digest 9988776655] trailing", ("9988776655",)),
        ("Authorization: Digest 9988776655} trailing", ("9988776655",)),
        ("Authorization: Digest 9988776655) trailing", ("9988776655",)),
        (r"Authorization: Digest 9988776655\ nonce=\"a1b2c3d4\"", ("9988776655", "a1b2c3d4")),
        (
            r'{"Authorization": "Digest 9988776655; nonce=\"a1b2c3d4\"]"}',
            ("9988776655", "a1b2c3d4"),
        ),
        (
            'Authorization: Digest 9988776655; nonce=a1b2c3d4; response=r1e2s3]} ',
            ("9988776655", "a1b2c3d4", "r1e2s3"),
        ),
    )
    for text, secrets in cases:
        cleaned = obs.redact(text)
        for secret in secrets:
            assert secret not in cleaned, (text, secret)
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text

    signature = obs.error_signature(
        "RuntimeError",
        'RuntimeError: digest 9988776655; nonce="a1b2c3d4"; response="deadbeef"',
    )
    for secret in ("9988776655", "a1b2c3d4", "deadbeef"):
        assert secret not in signature

    # Widened tails must not swallow benign descriptive uses of scheme words.
    for benign in ("Bearer count: 12", "Token scheme value", "digest header"):
        assert obs.redact(benign) == benign


def test_escaped_assignment_separators_are_redacted_combinatorially() -> None:
    """F2: decoded `\\n`/`\\t`/`\\u003a` separators are assignment boundaries."""

    secret = "s3cr3t!x9"
    keys = ("authToken", "client_secret", "session_id", "apiKey", "password")
    escaped_whitespace = (r"\n", r"\t", r"\r", r"\u000a", r"\u0009", r"\\n", r"\\u000a", r"\n\t")
    operators = (":", "=", "=>", "->")
    values = (secret, f"[{secret}]", f"({secret})", f'"{secret}"', f"`{secret}`")

    checked = 0
    for key, whitespace, operator, value in itertools.product(
        keys, escaped_whitespace, operators, values
    ):
        text = f"{key}{whitespace} {operator} {value}"
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.redact(cleaned) == cleaned, text
        checked += 1

    # Escaped colon/equals stand on their own: no additional operator appears.
    for key, separator, value in itertools.product(
        keys, (r":", r"\\u003a", r"=", r"\\u003d"), values
    ):
        text = f"{key}{separator} {value}"
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.redact(cleaned) == cleaned, text
        checked += 1

    assert checked == 900
    nested = obs.redact({"meta": [r'{"apiKey":\n"s3cr3t!x9"}', r"authToken: s3cr3t!x9"]})
    assert "s3cr3t!x9" not in json.dumps(nested)
    assert obs.redact(nested) == nested


def test_escaped_newline_between_json_key_and_value_is_redacted() -> None:
    """F3: escaped whitespace after a JSON colon cannot hide the value."""

    cases = (
        (r'{"authToken":\n"9988776655"}', "9988776655"),
        (r'{"client_secret":\t"ab12"}', "ab12"),
        (r'{"outer": {"apiKey":\n"q7r"}}', "q7r"),
        (r'[{"authorization":\r"xy9"}]', "xy9"),
        (r'prefix {"authToken":\u000a"s3cr3t!x9"} suffix', "s3cr3t!x9"),
        (r'{"authToken":\\n"s3cr3t!x9"}', "s3cr3t!x9"),
    )
    for text, secret in cases:
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text


class EscapedNewlineJsonLeakTransport(obs.FakeTransport):
    """Inject escaped-newline JSON keys into every logged output surface."""

    def send(self, **kwargs) -> dict:
        index = len(self.calls) + 1
        if index == 2:
            raise RuntimeError(r'RuntimeError: {"authToken":\n"s3cr3t!x9"}')
        data = super().send(**kwargs)
        data["content"] += r' debug {"authToken":\n"9988776655"}'
        data["status"] = r'completed {"client_secret":\t"ab12"}'
        data["provider"] = r'provider {"apiKey":\n"q7r"}'
        data["model"] = r'model {"authorization":\r"xy9"}'
        data["model_status"] = r'status {"sessionId":\u000a"9988776655"}'
        data["command"] = r'cmd {"password":\n"s3cr3t!x9"}'
        data["command_status"] = r'cmd {"Cookie":\t"ab12"}'
        data["memories_used"] = [
            r'memory {"authToken":\n"9988776655"}',
            {"client_secret": "ab12"},
        ]
        data["tool_results"] = [
            {
                "name": "read_file",
                "result": json.dumps({"error": r'{"authorization":\n"xy9"}'}),
            }
        ]
        return data


def test_escaped_newline_json_never_reaches_any_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """F3 end-to-end: checkpoint, JSONL, Markdown, summary and stdout are safe."""

    secrets = ("9988776655", "ab12", "q7r", "xy9", "s3cr3t!x9")
    checkpoint = tmp_path / "checkpoint.jsonl"
    turns = obs.run_conversation(
        EscapedNewlineJsonLeakTransport(),
        messages=3,
        allow_short=True,
        retries=0,
        jsonl_path=checkpoint,
    )
    summary = obs.build_summary(turns, messages_requested=3, dry_run=True)
    jsonl_path = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")
    assert obs.main(
        ["--dry-run", "--allow-short", "--messages", "2", "--out-dir", str(tmp_path / "cli")]
    ) == 0
    cli_stdout = capsys.readouterr().out

    surfaces = {
        "checkpoint": checkpoint.read_text(encoding="utf-8"),
        "jsonl": jsonl_path.read_text(encoding="utf-8"),
        "markdown": md_path.read_text(encoding="utf-8"),
        "summary": summary_path.read_text(encoding="utf-8"),
        "stdout": cli_stdout,
    }
    for label, text in surfaces.items():
        for secret in secrets:
            assert secret not in text, (label, secret)
    for label, value in (("turns", turns), ("summary", summary)):
        for secret in secrets:
            assert secret not in json.dumps(value), (label, secret)
    for field in ("providers", "models", "status_distribution", "top_error_signatures"):
        for secret in secrets:
            assert secret not in json.dumps(summary[field]), (field, secret)
    assert "s3cr3t!x9" not in turns[1]["error_signature"]
    assert obs.REDACTED in md_path.read_text(encoding="utf-8")


def test_debug_assignment_operators_and_wrappers_are_redacted() -> None:
    """F4: `=>`, `->`, brackets, parentheses and backticks cannot wrap secrets."""

    secret = "s3cr3t!x9"
    cases = (
        f"authToken => {secret}",
        f"authToken -> {secret}",
        f"authToken: [{secret}]",
        f"authToken: ({secret})",
        f"authToken: `{secret}`",
        f'client_secret => "{secret}"',
        f"session_id -> [{secret}]",
        f"password => ({secret})",
        f"apiKey -> `{secret}`",
        f"Authorization => Digest {secret}",
        f"Authorization -> Bearer {secret}",
        r"client_secret\n=> [" + secret + "]",
    )
    for text in cases:
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text

    # Idempotence comes from inspecting the captured value, not from banning
    # `[`: already-redacted assignments must survive another writer pass.
    for already_redacted in (
        "authToken=[REDACTED]",
        "authToken: [REDACTED]",
        "authToken -> `REDACTED`",
        "client_secret => (REDACTED)",
    ):
        assert obs.redact(already_redacted) == already_redacted


def test_deeply_double_escaped_credentials_are_redacted() -> None:
    """F5: unescaping loops until stable (bounded at 16), not four times."""

    secret = "s3cr3t!x9"
    payload = r'{\"authToken\": \"s3cr3t!x9\"}'
    encoded = payload
    for _ in range(8):  # 256 backslashes; a four-pass loop cannot reach the key
        encoded = encoded.replace("\\", "\\\\")
    assert encoded.count("\\") > 4 * payload.count("\\")

    cleaned = obs.redact(encoded)
    assert secret not in cleaned
    assert obs.REDACTED in cleaned
    assert obs.redact(cleaned) == cleaned

    nested = obs.redact({"content": encoded})
    assert secret not in json.dumps(nested)


# ---------------------------------------------------------------------------
# Regressions for the structural leak review (A-D)
#
# Plain lowercase secrets are deliberate: with no digits or symbols, the
# entropy fallback cannot mask a structural redaction failure. Every case
# below failed on at least one of the four leak classes before the fix.
# ---------------------------------------------------------------------------

_PLAIN_SECRET = "qwertyuiop"
_PLAIN_SECRET_2 = "asdfghjkl"


def _odd_escaped_json(
    key: str, secret: str, backslashes: int = 3, code: str = "n"
) -> str:
    r"""Build ``{"key":\\\n"secret"}`` with an odd backslash run before *code*."""

    return '{"' + key + '":' + "\\" * backslashes + code + '"' + secret + '"}'


def test_auth_operator_backtracking_cannot_leave_wrapped_secret() -> None:
    """A: `=>`, `:=`, `::` and `==` must not backtrack into a one-char value."""

    secret = _PLAIN_SECRET
    keys = ("authToken", "Authorization", "client_secret", "password", "Cookie")
    operators = ("=>", "->", ":=", "::", "==", ":", "=")
    values = (f"[{secret}]", f"`{secret}`", f"\\[{secret}\\]", f"({secret})")
    checked = 0
    for key, operator, value in itertools.product(keys, operators, values):
        text = f"{key} {operator} {value}"
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text
        checked += 1
    assert checked == 140

    # Escaped brackets and backticks must also stay inside a scheme value.
    for text in (
        f"Authorization => Bearer \\[{secret}\\]",
        f"authorization -> bearer [{secret}]",
        f"Authorization := Token `{secret}`",
        f"Authorization == Basic \\[{secret}\\]",
    ):
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text


def test_nested_wrappers_and_sensitive_value_lists_are_consumed() -> None:
    """B: mixed wrappers and comma/semicolon lists cannot hide plain secrets."""

    secret, second = _PLAIN_SECRET, _PLAIN_SECRET_2
    cases = (
        (f"authToken: [[{secret}]]", (secret,)),
        (f"authToken: [({secret})]", (secret,)),
        (f'authToken: ["{secret}"]', (secret,)),
        (f"authToken: <[{secret}]>", (secret,)),
        (f"authToken: `({secret})`", (secret,)),
        (f"authToken: \\[{secret}\\]", (secret,)),
        (f"authToken: \\`({secret})\\`", (secret,)),
        (f"authToken: {secret}, {second}", (secret, second)),
        (f"authToken: [{secret}, {second}]", (secret, second)),
        (f"client_secret: {secret}; session_id: {second}", (secret, second)),
        (f"authToken: {secret}, client_secret: {second}", (secret, second)),
        (f"authToken: [[{secret}]], password: ({second})", (secret, second)),
    )
    for text, secrets in cases:
        cleaned = obs.redact(text)
        for value in secrets:
            assert value not in cleaned, text
        assert obs.REDACTED in cleaned, text
        assert obs.redact(cleaned) == cleaned, text

    # A following sensitive key must still be recognized by name instead of
    # being swallowed as an anonymous item of the first key's value list.
    cleaned = obs.redact(f"authToken: {secret}, client_secret: {second}")
    assert "client_secret=" in cleaned, cleaned
    assert cleaned.count(obs.REDACTED) == 2, cleaned


def test_odd_escaped_json_key_is_structurally_redacted() -> None:
    """C: odd escape runs normalize to parseable JSON before text rules run."""

    secret = _PLAIN_SECRET
    for backslashes in (3, 5, 7, 9):
        for code in ("n", "t", "u000a"):
            text = _odd_escaped_json("authToken", secret, backslashes, code)
            cleaned = obs.redact(text)
            assert secret not in cleaned, text
            assert obs.REDACTED in cleaned, text
            assert obs.redact(cleaned) == cleaned, text
            # The key was recognized structurally, not by a text rule that
            # happened to consume a fragment of the value.
            assert json.loads(cleaned) == {"authToken": obs.REDACTED}, (text, cleaned)

    for key in ("client_secret", "apiKey", "password", "sessionId"):
        text = _odd_escaped_json(key, secret)
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert f'"{key}": "{obs.REDACTED}"' in cleaned, (text, cleaned)
        assert obs.redact(cleaned) == cleaned, text

    nested = '{"outer": ' + _odd_escaped_json("apiKey", secret, 5) + "}"
    cleaned = obs.redact(nested)
    assert secret not in cleaned
    assert json.loads(cleaned) == {"outer": {"apiKey": obs.REDACTED}}


def test_deep_escape_regression_recognizes_key_structurally() -> None:
    """D: more than 16 escape layers still reach the structural sensitive key."""

    secret = _PLAIN_SECRET
    payload = '{\\"authToken\\": \\"' + secret + '\\"}'
    encoded = payload
    for _ in range(16):
        encoded = encoded.replace("\\", "\\\\")
    assert encoded.count("\\") > 16 * payload.count("\\")

    normalized = obs._normalize_escapes(encoded)
    assert '"authToken"' in normalized and secret in normalized
    cleaned = obs.redact(encoded)
    assert secret not in cleaned
    assert obs.REDACTED in cleaned
    # Structural recognition: the parsed object keeps the key and maps it to
    # the placeholder, rather than merely losing the secret to a text rule.
    assert json.loads(cleaned) == {"authToken": obs.REDACTED}
    assert obs.redact(cleaned) == cleaned
    nested = obs.redact({"content": encoded})
    assert secret not in json.dumps(nested)


class StructuralLeakTransport(obs.FakeTransport):
    """Inject A/B/C variants into every server-controlled output surface."""

    def send(self, **kwargs) -> dict:
        index = len(self.calls) + 1
        if index == 2:
            raise RuntimeError(f"RuntimeError: authToken := [{_PLAIN_SECRET}]")
        data = super().send(**kwargs)
        data["content"] += f" debug {_odd_escaped_json('authToken', _PLAIN_SECRET)}"
        data["status"] = f"completed authToken: [[{_PLAIN_SECRET}]]"
        data["provider"] = f"provider client_secret: \\[{_PLAIN_SECRET}\\]"
        data["model"] = f"model Authorization => Bearer [{_PLAIN_SECRET}]"
        data["model_status"] = (
            f"status password: {_PLAIN_SECRET}, {_PLAIN_SECRET_2}"
        )
        data["command"] = (
            f"cmd authToken: [{_PLAIN_SECRET}], session_id: {_PLAIN_SECRET_2}"
        )
        data["command_status"] = _odd_escaped_json(
            "client_secret", _PLAIN_SECRET, 5, "u000a"
        )
        data["memories_used"] = [
            f"memory authToken: <[{_PLAIN_SECRET}]>",
            {"client_secret": _PLAIN_SECRET},
        ]
        data["tool_results"] = [
            {
                "name": "read_file",
                "result": json.dumps({"error": f"authToken => `{_PLAIN_SECRET}`"}),
            }
        ]
        return data


def test_structural_leaks_never_reach_any_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A/B/C end to end: checkpoint, JSONL, Markdown, summary and stdout."""

    secrets = (_PLAIN_SECRET, _PLAIN_SECRET_2)
    checkpoint = tmp_path / "checkpoint.jsonl"
    turns = obs.run_conversation(
        StructuralLeakTransport(),
        messages=3,
        allow_short=True,
        retries=0,
        jsonl_path=checkpoint,
    )
    summary = obs.build_summary(turns, messages_requested=3, dry_run=True)
    jsonl_path = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")
    assert obs.main(
        ["--dry-run", "--allow-short", "--messages", "2", "--out-dir", str(tmp_path / "cli")]
    ) == 0
    cli_stdout = capsys.readouterr().out

    surfaces = {
        "checkpoint": checkpoint.read_text(encoding="utf-8"),
        "jsonl": jsonl_path.read_text(encoding="utf-8"),
        "markdown": md_path.read_text(encoding="utf-8"),
        "summary": summary_path.read_text(encoding="utf-8"),
        "stdout": cli_stdout,
    }
    for label, text in surfaces.items():
        for secret in secrets:
            assert secret not in text, (label, secret)
    for label, value in (("turns", turns), ("summary", summary)):
        for secret in secrets:
            assert secret not in json.dumps(value), (label, secret)
    for field in ("providers", "models", "status_distribution", "top_error_signatures"):
        for secret in secrets:
            assert secret not in json.dumps(summary[field]), (field, secret)
    assert secrets[0] not in turns[1]["error_signature"]
    assert obs.REDACTED in md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Regressions for the key-aware scanner (nine residual root causes)
#
# The plain lowercase secret is deliberate: with no digit or symbol the
# entropy fallback cannot hide a structural miss. Every case below leaked
# under the previous per-operator regex layer.
# ---------------------------------------------------------------------------

_SCANNER_SECRET = "qwertyuiop"  # same plain secret as _PLAIN_SECRET
_SCANNER_CASES: dict[str, tuple[str, ...]] = {
    "whitespace_inside_wrappers": (
        f"authToken: [ {_SCANNER_SECRET} ]",
        f"authToken: [[ {_SCANNER_SECRET} ]]",
        f"client_secret => {{ {_SCANNER_SECRET} }}",
        f"password: < [ {_SCANNER_SECRET} ] >",
        f"cookie := [ [ {_SCANNER_SECRET} ] ]",
        f"session_id -> {{ [ {_SCANNER_SECRET} ] }}",
    ),
    "brace_wrappers": (
        f"authToken: {{{_SCANNER_SECRET}}}",
        f"client_secret => {{ {_SCANNER_SECRET} }}",
        f"password: {{{_SCANNER_SECRET}}}, session_id: {{{_SCANNER_SECRET}}}",
        f"Authorization: Bearer {{{_SCANNER_SECRET}}}",
        f"authToken: [[{{{_SCANNER_SECRET}}}]]",
    ),
    "secondary_operator_tokens": (
        f"authToken: = {_SCANNER_SECRET}",
        f"authToken: := {_SCANNER_SECRET}",
        f"authToken: => {_SCANNER_SECRET}",
        f"authToken::= {_SCANNER_SECRET}",
        f"authToken : : {_SCANNER_SECRET}",
        f"authToken = == {_SCANNER_SECRET}",
        f"authorization - -> {_SCANNER_SECRET}",
        f"Cookie <> {_SCANNER_SECRET}",
        f"password (= {_SCANNER_SECRET})",
    ),
    "leading_list_separators": (
        f"authToken: , {_SCANNER_SECRET}",
        f"authToken: ; {_SCANNER_SECRET}",
        f"authToken: , , {_SCANNER_SECRET}",
        f"authToken: [, {_SCANNER_SECRET}]",
        f"authToken: ; ; {_SCANNER_SECRET}",
        f"client_secret => ,{_SCANNER_SECRET}",
        r"password: \, " + _SCANNER_SECRET,
    ),
    "escaped_semicolons": (
        f"authToken: a1b2\\; {_SCANNER_SECRET}",
        f"Authorization: Digest a1b2\\; {_SCANNER_SECRET}",
        f"authToken: a1b2\\;c3d4\\; {_SCANNER_SECRET}",
        f"authToken: [a1b2\\; {_SCANNER_SECRET}]",
        f"client_secret: a1b2\\; {_SCANNER_SECRET}\\;",
        f"authToken: {_SCANNER_SECRET}\\, {_SCANNER_SECRET}",
    ),
    "nested_keyed_first_value": (
        f"outer: authToken: {_SCANNER_SECRET}",
        f"authToken: inner={_SCANNER_SECRET}",
        f"config: authToken: {_SCANNER_SECRET}, client_secret: {_PLAIN_SECRET_2}",
        f"meta: [authToken: {_SCANNER_SECRET}]",
        f"authToken: data {{ session_id: {_SCANNER_SECRET} }}",
    ),
    "dangling_escape_before_decoded_equals": (
        r'{"authToken":\\\="' + f'"{_SCANNER_SECRET}"' + "}",
        r'{"authToken":\="' + f'"{_SCANNER_SECRET}"' + "}",
        r"authToken\\=" + _SCANNER_SECRET,
        r'{"authToken":\\= "' + _SCANNER_SECRET + '"}',
        r"password\=" + _SCANNER_SECRET,
    ),
    "whitespace_only_separators": (
        f"authToken   {_SCANNER_SECRET}",
        f"client_secret\t{_SCANNER_SECRET}",
        f"password    {_SCANNER_SECRET}, session_id  {_PLAIN_SECRET_2}",
        f"authToken\n{_SCANNER_SECRET}",
        f"PIN   {_SCANNER_SECRET}",
    ),
    "odd_escapes_replacing_colon": (
        r"authToken\\n: " + _SCANNER_SECRET,
        r'{"authToken"\\n: "' + _SCANNER_SECRET + '"}',
        r'{"authToken"\\u003a "' + _SCANNER_SECRET + '"}',
        r"authToken\\: " + _SCANNER_SECRET,
        r'{"authToken"\\\: "' + _SCANNER_SECRET + '"}',
        r'{"API key"\\: "' + _SCANNER_SECRET + '"}',
    ),
}


def _assert_scanner_cases(cause: str) -> None:
    for text in _SCANNER_CASES[cause]:
        cleaned = obs.redact(text)
        assert _SCANNER_SECRET not in cleaned, (cause, text, cleaned)
        assert obs.REDACTED in cleaned, (cause, text, cleaned)
        assert obs.redact(cleaned) == cleaned, (cause, text, cleaned)


def test_scanner_whitespace_inside_wrappers() -> None:
    """Root cause 1: `[ x ]`, `[[ x ]]` and whitespace inside mixed wrappers."""

    _assert_scanner_cases("whitespace_inside_wrappers")


def test_scanner_brace_wrappers() -> None:
    """Root cause 2: `{x}` wrappers, including nested and list forms."""

    _assert_scanner_cases("brace_wrappers")


def test_scanner_secondary_operator_tokens() -> None:
    """Root cause 3: a second operator token can never become the value."""

    _assert_scanner_cases("secondary_operator_tokens")


def test_scanner_leading_list_separators() -> None:
    """Root cause 4: leading, repeated and escaped `,`/`;` are part of the span."""

    _assert_scanner_cases("leading_list_separators")


def test_scanner_escaped_semicolons() -> None:
    """Root cause 5: escaped semicolons join list items instead of ending the span."""

    _assert_scanner_cases("escaped_semicolons")


def test_scanner_nested_keyed_first_value() -> None:
    """Root cause 6: an inner keyed value is consumed with its outer key."""

    _assert_scanner_cases("nested_keyed_first_value")


def test_scanner_dangling_escape_before_decoded_equals() -> None:
    """Root cause 7: odd escape runs never leave `\\=` in front of the secret."""

    _assert_scanner_cases("dangling_escape_before_decoded_equals")


def test_scanner_whitespace_only_separators() -> None:
    """Root cause 8: spaces, tabs and newlines without any operator still bind."""

    _assert_scanner_cases("whitespace_only_separators")


def test_scanner_odd_escapes_replacing_colon() -> None:
    """Root cause 9: odd escapes in place of `:` cannot hide a keyed value."""

    _assert_scanner_cases("odd_escapes_replacing_colon")


def _scanner_variants() -> list[str]:
    """7 keys x 7 operators x 10 wrappers x 12 separators = 5,880 strings."""

    secret = _SCANNER_SECRET
    keys = ("authToken", "client_secret", "API key", "password", "PIN", "cookie", "session_id")
    operators = (":", "=", "=>", "->", ":=", "::", "<>")
    wrappers = (
        ("{", "}"), ("[ ", " ]"), ("[[", "]]"), ("<[", "]>"), ("(", ")"),
        ("`", "`"), ("{ ", " }"), ("\\{", "\\}"), ("[", ", " + secret + "]"),
        ("", ""),
    )
    separators = (
        "", " ", "= ", ": ", ":= ", ", ", "; ", r"\, ", r"\n", r"\t ", r"\=", r"\: ",
    )
    variants = [
        f"{key}{operator}{separator}{before}{secret}{after}"
        for key, operator, separator, (before, after) in itertools.product(
            keys, operators, separators, wrappers
        )
    ]
    assert len(variants) == 5880
    return variants


def test_key_scanner_5880_variant_sweep_has_zero_leaks() -> None:
    """7 keys x 7 operators x 10 wrappers x 12 separators = 5,880 variants."""

    secret = _SCANNER_SECRET
    for text in _scanner_variants():
        cleaned = obs.redact(text)
        assert secret not in cleaned, text
        assert obs.redact(cleaned) == cleaned, text


def test_concatenated_multiword_key_assignments_do_not_leak() -> None:
    """A redacted span must end before the next assignment, with no leftover key."""

    expected_placeholders = (
        f"API key: {_PLAIN_SECRET} API key: {_PLAIN_SECRET}",
        f"API key: {_PLAIN_SECRET}; API key: {_PLAIN_SECRET}",
        f"session id: {_PLAIN_SECRET} session id: {_PLAIN_SECRET}",
        f"session id: {_PLAIN_SECRET}; session id: {_PLAIN_SECRET}",
        f"X Session Id: {_PLAIN_SECRET} X Session Id: {_PLAIN_SECRET}",
        f"X Session Id: {_PLAIN_SECRET}; X Session Id: {_PLAIN_SECRET}",
        f"apiKey: {_PLAIN_SECRET} key: {_PLAIN_SECRET_2}",
        f"session id: {_PLAIN_SECRET} id: {_PLAIN_SECRET_2}",
        f"session: {_PLAIN_SECRET} id: {_PLAIN_SECRET_2}",
    )
    for text in expected_placeholders:
        cleaned = obs.redact(text)
        assert _PLAIN_SECRET not in cleaned, (text, cleaned)
        assert _PLAIN_SECRET_2 not in cleaned, (text, cleaned)
        assert cleaned.count(obs.REDACTED) == 2, (text, cleaned)
        # Repeated writer passes must neither restore nor duplicate a secret.
        assert obs.redact(cleaned) == cleaned, (text, cleaned)

    # A bare trailing component is sensitive with an operator, but ordinary
    # descriptive prose about a conversation id stays observable.
    assert obs.redact(f"key: {_PLAIN_SECRET}") == f"key={obs.REDACTED}"
    assert obs.redact(f"id: {_PLAIN_SECRET}") == f"id={obs.REDACTED}"
    benign = "Response conversation id: server-assigned-conversation"
    assert obs.redact(benign) == benign


class ScannerVariantTransport(obs.FakeTransport):
    """Spread all 5,880 scanner variants evenly across 400 dry-run turns."""

    def __init__(self, variants: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.batches: dict[int, list[str]] = {index: [] for index in range(1, 401)}
        for offset, variant in enumerate(variants):
            self.batches[offset % 400 + 1].append(variant)
        assert all(self.batches.values())

    def send(self, *, message: str, conversation_id: str, timeout_seconds: float) -> dict:
        data = super().send(
            message=message, conversation_id=conversation_id, timeout_seconds=timeout_seconds
        )
        index = self._turn_by_message[message]
        payload = " | ".join(self.batches[index])
        data["content"] = f"turn {index}: {payload}"
        data["provider"] = f"provider {payload}"
        data["memories_used"] = [f"memory {payload}"]
        return data


def test_5880_scanner_variants_distributed_over_400_turn_dry_run(tmp_path: Path) -> None:
    """Every variant crosses a real 400-turn write path without leaking."""

    variants = _scanner_variants()
    checkpoint = tmp_path / "checkpoint.jsonl"
    transport = ScannerVariantTransport(variants, dry_run=True)
    turns = obs.run_conversation(transport, messages=400, retries=0, jsonl_path=checkpoint)
    assert len(turns) == 400
    assert [turn["index"] for turn in turns] == list(range(1, 401))
    assert sum(len(batch) for batch in transport.batches.values()) == 5880

    summary = obs.build_summary(turns, dry_run=True, messages_requested=400)
    jsonl_path = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")
    surfaces = {
        "checkpoint": checkpoint.read_text(encoding="utf-8"),
        "jsonl": jsonl_path.read_text(encoding="utf-8"),
        "markdown": md_path.read_text(encoding="utf-8"),
        "summary": summary_path.read_text(encoding="utf-8"),
        "stdout": obs._json_dumps(obs.redact(summary)),
    }
    for label, text in surfaces.items():
        assert _SCANNER_SECRET not in text, label
    # Every variant stays redacted under another writer pass, including when
    # it crosses an embedded JSON container in a metadata field.
    for text in variants:
        cleaned = obs.redact(text)
        assert obs.redact(cleaned) == cleaned, text
    assert summary["total_turns"] == 400
    assert obs.REDACTED in md_path.read_text(encoding="utf-8")


_SCANNER_PAYLOAD = " | ".join(
    next(iter(cases)) for cases in _SCANNER_CASES.values()
)


class KeyScannerLeakTransport(obs.FakeTransport):
    """Inject all nine root-cause variants into every server-controlled field."""

    def send(self, **kwargs) -> dict:
        index = len(self.calls) + 1
        if index == 2:
            raise RuntimeError(f"RuntimeError: {_SCANNER_PAYLOAD}")
        data = super().send(**kwargs)
        data["content"] += f" debug {_SCANNER_PAYLOAD}"
        data["status"] = f"completed {_SCANNER_PAYLOAD}"
        data["provider"] = f"provider {_SCANNER_PAYLOAD}"
        data["model"] = f"model {_SCANNER_PAYLOAD}"
        data["model_status"] = f"status {_SCANNER_PAYLOAD}"
        data["command"] = f"cmd {_SCANNER_PAYLOAD}"
        data["command_status"] = f"cmd {_SCANNER_PAYLOAD}"
        data["memories_used"] = [f"memory {_SCANNER_PAYLOAD}"]
        data["tool_results"] = [
            {
                "name": f"read_file {_SCANNER_PAYLOAD}",
                "result": json.dumps({"error": _SCANNER_PAYLOAD}),
            }
        ]
        data["tool_calls_pending"] = [f"pending {_SCANNER_PAYLOAD}"]
        return data


def test_key_scanner_leaks_never_reach_any_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """All nine causes stay out of turns, files, distributions and stdout."""

    checkpoint = tmp_path / "checkpoint.jsonl"
    turns = obs.run_conversation(
        KeyScannerLeakTransport(),
        messages=3,
        allow_short=True,
        retries=0,
        jsonl_path=checkpoint,
    )
    summary = obs.build_summary(turns, messages_requested=3, dry_run=True)
    jsonl_path = obs.write_jsonl(turns, tmp_path / "conversation.jsonl")
    md_path = obs.write_markdown(turns, summary, tmp_path / "conversation.md")
    summary_path = obs.write_summary(summary, tmp_path / "summary.json")
    assert obs.main(
        [
            "--dry-run", "--allow-short", "--messages", "2",
            "--out-dir", str(tmp_path / "cli"),
            "--dry-run-leak-metadata", _SCANNER_PAYLOAD,
        ]
    ) == 0
    cli_stdout = capsys.readouterr().out

    surfaces = {
        "checkpoint": checkpoint.read_text(encoding="utf-8"),
        "jsonl": jsonl_path.read_text(encoding="utf-8"),
        "markdown": md_path.read_text(encoding="utf-8"),
        "summary": summary_path.read_text(encoding="utf-8"),
        "stdout": cli_stdout,
    }
    for label, text in surfaces.items():
        assert _SCANNER_SECRET not in text, label
    for label, value in (("turns", turns), ("summary", summary)):
        assert _SCANNER_SECRET not in json.dumps(value), label
    for field in ("providers", "models", "status_distribution", "top_error_signatures"):
        assert _SCANNER_SECRET not in json.dumps(summary[field]), field
    assert _SCANNER_SECRET not in turns[1]["error_signature"]
    assert obs.REDACTED in md_path.read_text(encoding="utf-8")


def test_scanner_preserves_benign_text_and_sensitive_words_in_paths() -> None:
    """A key-looking word inside an identifier or path is not an assignment."""

    for text in (
        "Bearer count: 12",
        "Token scheme value",
        "digest header",
        "Nothing sensitive here.",
        "/tmp/pytest/test_url_credentials_scrubbed_0/scratch/note.md",
        "artifacts/observation/scratch/note.md",
    ):
        assert obs.redact(text) == text


# ---------------------------------------------------------------------------
# Live-session findings: honest accounting, NUL paths, retry ids, error fields
# ---------------------------------------------------------------------------


def _clean_turn(index: int) -> dict:
    """A minimal turn record that is genuinely clean under the new rules."""

    return {
        "index": index,
        "timestamp": "2026-11-20T00:00:00Z",
        "category": "identity",
        "assistant_content": "a real answer",
        "status": "completed",
        "failure": False,
        "tool_calls_pending": [],
        "tool_names": [],
        "tool_results": [],
        "tool_errors": [],
        "memories_used": [],
        "elapsed_seconds": 0.01,
    }


def test_success_requires_completed_status_and_no_pending_tools() -> None:
    """Incomplete-with-content and pending-tool turns are not clean answers."""

    cases = (
        (
            {"status": "incomplete", "assistant_content": "partial answer"},
            False,
            {"incomplete_status"},
        ),
        (
            {"status": "completed", "assistant_content": "answer", "tool_calls_pending": ["t"]},
            False,
            {"incomplete_with_pending_tools"},
        ),
        (
            {"status": "incomplete", "assistant_content": "", "tool_calls_pending": ["t"]},
            False,
            {"incomplete_status", "incomplete_without_content", "incomplete_with_pending_tools"},
        ),
        (
            {"status": "unknown", "assistant_content": "answer"},
            False,
            {"unrecognized_status"},
        ),
        (
            {"status": "completed", "assistant_content": "   "},
            False,
            {"empty_answer"},
        ),
        (
            {"status": "error", "assistant_content": "answer"},
            False,
            {"error_status"},
        ),
    )
    for changes, expected_success, expected_reasons in cases:
        turn = {**_clean_turn(1), **changes}
        assert obs.is_successful_turn(turn) is False, changes
        assert set(obs.turn_deficiencies(turn)) == expected_reasons, changes

    for status in obs.SUCCESS_STATUSES:
        turn = {**_clean_turn(1), "status": status}
        assert obs.is_successful_turn(turn) is True, status
        assert obs.turn_deficiencies(turn) == [], status


def test_floor_verdicts_distinguish_clean_from_deficient_runs() -> None:
    """Only 400 completed answers with zero deficiencies earn the clean verdict."""

    turns = [_clean_turn(index) for index in range(1, 402)]
    clean = obs.build_summary(turns, messages_requested=401)
    assert clean["successful_turns"] == 401
    assert clean["deficiency_count"] == 0
    assert clean["floor_satisfied"] is True
    assert clean["run_verdict"] == "floor_satisfied_clean"

    with_incomplete = [dict(turn) for turn in turns]
    with_incomplete[0]["status"] = "incomplete"
    summary = obs.build_summary(with_incomplete, messages_requested=401)
    assert summary["successful_turns"] == 400
    assert summary["deficiency_count"] == 1
    assert summary["deficiency_distribution"] == {"incomplete_status": 1}
    assert summary["floor_satisfied"] is True
    assert summary["run_verdict"] == "floor_satisfied_with_deficiencies"

    with_pending = [dict(turn) for turn in turns]
    with_pending[0]["tool_calls_pending"] = ["still-running"]
    pending_summary = obs.build_summary(with_pending, messages_requested=401)
    assert pending_summary["successful_turns"] == 400
    assert pending_summary["deficiency_distribution"] == {"incomplete_with_pending_tools": 1}
    assert pending_summary["run_verdict"] == "floor_satisfied_with_deficiencies"

    partial = obs.build_summary([_clean_turn(i) for i in range(1, 401)], messages_requested=400)
    assert partial["floor_satisfied"] is True
    assert partial["run_verdict"] == "floor_satisfied_clean"

    all_incomplete = obs.build_summary(
        [{**_clean_turn(i), "status": "incomplete"} for i in range(1, 401)],
        messages_requested=400,
    )
    assert all_incomplete["successful_turns"] == 0
    assert all_incomplete["deficiency_count"] == 400
    assert all_incomplete["run_verdict"] == "all_answers_failed"


def test_control_bytes_in_tool_result_paths_do_not_raise(tmp_path: Path) -> None:
    """NUL bytes from binary search output must not kill post-run reporting."""

    turns = [
        {
            "index": 1,
            "tool_results": [
                {"name": "search_files", "result": json.dumps({"path": "/tmp/pyt\x00hon/note.md"})},
            ],
        },
        {
            "index": 2,
            "tool_results": [
                {
                    "name": "read_file",
                    "result": "binary .pyc output: \x00/tmp/evil\x00module.pyc \x7f trailing",
                },
            ],
        },
    ]
    report = obs.check_tool_result_paths(turns, out_dir=tmp_path, dry_run=True)
    assert report["issue_count"] >= 2
    assert report["clean"] is False
    assert all("\x00" not in issue for issue in report["issues"])
    assert not any(claim["path"] for claim in report["claims"] if "\x00" in claim["path"])


class NulPathTransport(obs.FakeTransport):
    """A tool result that claims a path containing a NUL byte."""

    def send(self, **kwargs) -> dict:
        data = super().send(**kwargs)
        data["tool_results"] = [
            {"name": "search_files", "result": json.dumps({"path": "/tmp/pyt\x00hon/note.md"})},
        ]
        return data


def test_nul_path_claim_still_writes_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: a NUL path claim cannot prevent the three output files."""

    monkeypatch.setattr(obs, "FakeTransport", NulPathTransport)
    assert obs.main(
        ["--dry-run", "--allow-short", "--messages", "2", "--out-dir", str(tmp_path)]
    ) == 0
    capsys.readouterr()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_turns"] == 2
    assert summary["tool_result_paths"]["clean"] is False
    assert summary["tool_result_paths"]["issue_count"] >= 2
    assert "## Turn 2" in (tmp_path / "conversation.md").read_text(encoding="utf-8")
    assert len((tmp_path / "conversation.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_raising_post_run_checker_still_writes_transcript_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A checker that throws is reported, not allowed to discard the run."""

    def explode(**kwargs) -> dict:
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(obs, "check_tool_result_paths", explode)
    code = obs.main(
        ["--dry-run", "--allow-short", "--messages", "3", "--out-dir", str(tmp_path)]
    )
    stdout = capsys.readouterr().out
    assert code == 1
    summary = json.loads(stdout)
    assert summary["total_turns"] == 3
    assert summary["tool_result_paths"]["clean"] is False
    assert any("post_run:tool_result_paths" in issue for issue in summary["output_check"]["issues"])
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["run_verdict"] == summary["run_verdict"]
    markdown = (tmp_path / "conversation.md").read_text(encoding="utf-8")
    assert "## Turn 3" in markdown
    assert len((tmp_path / "conversation.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_request_id_is_stable_per_turn() -> None:
    """One logical turn keeps one control-free retry key across attempts."""

    first = obs.request_id_for_turn(obs.DEFAULT_CONVERSATION_ID, 7, obs.DEFAULT_SEED)
    assert first == obs.request_id_for_turn(obs.DEFAULT_CONVERSATION_ID, 7, obs.DEFAULT_SEED)
    assert first != obs.request_id_for_turn(obs.DEFAULT_CONVERSATION_ID, 8, obs.DEFAULT_SEED)
    assert first != obs.request_id_for_turn(obs.DEFAULT_CONVERSATION_ID, 7, 8)
    assert first != obs.request_id_for_turn("other-conversation", 7, obs.DEFAULT_SEED)
    assert first.startswith("observer-")
    assert not obs._CONTROL_BYTES_RE.search(first)


class _FakeHTTPResponse:
    """Minimal context-manager response for a monkeypatched urlopen."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_http_transport_sends_request_id_and_reads_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id travels in body and header; history replies are normalized."""

    seen: list = []

    def fake_urlopen(request, timeout=None):  # noqa: ANN001 - urlopen shim
        seen.append(request)
        if request.get_method() == "POST":
            return _FakeHTTPResponse({"conversation_id": "c1", "content": "live", "status": "completed"})
        if "pending" in request.full_url:
            return _FakeHTTPResponse(
                {"messages": [{"role": "user", "content": "hi", "metadata": {"request_id": "rid-1"}}]}
            )
        return _FakeHTTPResponse(
            {
                "messages": [
                    {"role": "user", "content": "hi", "metadata": {"request_id": "rid-1"}},
                    {"role": "assistant", "content": "committed answer", "metadata": {}},
                ]
            }
        )

    monkeypatch.setattr(obs.urllib.request, "urlopen", fake_urlopen)
    transport = obs.HttpChatTransport("http://127.0.0.1:8891/chat")
    transport.send(message="hi", conversation_id="c1", timeout_seconds=5, request_id="rid-1")
    post = seen[-1]
    assert json.loads(post.data)["request_id"] == "rid-1"
    assert any(
        key.lower() == obs.REQUEST_ID_HEADER.lower() and value == "rid-1"
        for key, value in post.header_items()
    )

    prior = transport.turn_in_history(
        message="hi", conversation_id="c1", request_id="rid-1", timeout_seconds=5
    )
    assert prior is not None and prior["content"] == "committed answer"
    assert prior["from_history"] is True and prior[obs.REQUEST_ID_FIELD] == "rid-1"
    assert seen[-1].get_method() == "GET"
    assert seen[-1].full_url.endswith("/conversations/c1")

    pending_transport = obs.HttpChatTransport(
        "http://127.0.0.1:8891/chat", history_url="http://127.0.0.1:8891/conversations/pending"
    )
    pending = pending_transport.turn_in_history(
        message="hi", conversation_id="c1", request_id="rid-1", timeout_seconds=5
    )
    assert pending is not None and pending["status"] == obs.HISTORY_PENDING_STATUS
    assert pending["content"] == "" and pending["history_pending"] is True


def test_http_history_404_means_no_committed_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing conversation row is not an error: there is nothing to adopt."""

    def missing(request, timeout=None):  # noqa: ANN001 - urlopen shim
        raise obs.urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)

    monkeypatch.setattr(obs.urllib.request, "urlopen", missing)
    transport = obs.HttpChatTransport("http://127.0.0.1:8891/chat")
    assert transport.turn_in_history(
        message="hi", conversation_id="missing", request_id="rid", timeout_seconds=5
    ) is None


def test_retry_adopts_committed_history_instead_of_resending() -> None:
    """A timeout followed by a durable history read must not duplicate rows."""

    rid = obs.request_id_for_turn("c1", 1, obs.DEFAULT_SEED)
    committed = {
        "conversation_id": "c1",
        "content": "committed answer",
        "status": "completed",
        "from_history": True,
    }
    transport = obs.FakeTransport(failure_indices=(1,), history={rid: committed})
    turns = obs.run_conversation(
        transport, messages=2, allow_short=True, retries=2, retry_backoff_seconds=0.0,
        conversation_id="c1",
    )
    scenario = obs.generate_scenario(2, allow_short=True)
    assert transport.calls == [scenario[0].text, scenario[1].text]
    assert turns[0]["from_history"] is True
    assert turns[0]["assistant_content"] == "committed answer"
    assert obs.is_successful_turn(turns[0])
    assert turns[0]["request_id"] == rid
    assert transport.sent_requests[0]["request_id"] == rid
    assert turns[0]["attempt_history"][-1]["from_history"] is True
    summary = obs.build_summary(turns, messages_requested=2)
    assert summary["history_recovered_turns"] == 1
    assert summary["history_pending_turns"] == 0
    assert summary["retry_count"] == 0


def test_failed_history_check_suppresses_resend() -> None:
    """When durable history cannot be read, do not risk duplicate chat rows."""

    transport = obs.FakeTransport(
        failure_indices=(1,), history_error=RuntimeError("history unavailable")
    )
    turns = obs.run_conversation(
        transport, messages=2, allow_short=True, retries=2, retry_backoff_seconds=0.0
    )
    scenario = obs.generate_scenario(2, allow_short=True)
    assert transport.calls == [scenario[0].text, scenario[1].text]
    assert turns[0]["failure"] is True
    assert turns[0]["attempts"] == 1
    assert turns[0]["history_check_error"].startswith("history:")
    assert turns[0]["attempt_history"][-1]["history_check_error"] == turns[0]["history_check_error"]
    summary = obs.build_summary(turns, messages_requested=2)
    assert summary["history_check_error_count"] == 1
    assert summary["successful_turns"] == 1


def test_structured_tool_success_and_error_fields_are_authoritative() -> None:
    """`error: null` under `success: true` is not 215 tool errors."""

    fragile_json = (
        '{"call_id": "abc", "success": true, '
        '"output": {"credential_env_configured": "[REDACTED]"}, '
        '"error": null, "duration_ms": 10.[REDACTED], "risk": "low"}'
    )
    results = [
        {"name": "a", "result": json.dumps({"success": True, "error": None, "output": {"n": 1}})},
        {"name": "b", "result": fragile_json},
        {"name": "c", "result": '{"success": true, "error": null'},
        {"name": "d", "result": json.dumps({"success": False, "error": None})},
        {"name": "e", "result": json.dumps({"success": True, "error": "not found"})},
        {"name": "f", "result": json.dumps({"ok": False, "error": None})},
        {"name": "g", "result": "plain text: error\": null is not a failure"},
    ]
    names, records, errors = obs.extract_tool_info(results)
    assert names == [entry["name"] for entry in results]
    assert len(records) == len(results)
    assert len(errors) == 3, errors
    assert any(error.startswith("tool:d:") for error in errors)
    assert any(error.startswith("tool:e:") for error in errors)
    assert any(error.startswith("tool:f:") for error in errors)
    assert not any('error": null' in error for error in errors)
    assert all("tool:a:" not in error and "tool:b:" not in error for error in errors)

    # A pure `error: null` result in a 200-turn run produces zero errors.
    repeated = obs.run_conversation(
        _NullErrorTransport(), messages=200, allow_short=True, retries=0
    )
    summary = obs.build_summary(repeated, messages_requested=200)
    assert summary["tool_invocations"] == 200
    assert summary["tool_error_count"] == 0
    assert summary["successful_turns"] == 200


class _NullErrorTransport(obs.FakeTransport):
    """Every tool result reports success with a null error field."""

    def send(self, **kwargs) -> dict:
        data = super().send(**kwargs)
        data["tool_calls_pending"] = []
        data["tool_results"] = [
            {
                "name": "model_providers",
                "result": json.dumps(
                    {
                        "success": True,
                        "output": {"count": 1},
                        "error": None,
                        "duration_ms": 10.5,
                        "risk": "low",
                        "decision": "allowed",
                    }
                ),
            }
        ]
        return data
