"""Composition-root integration tests for proposal-only self-evolution.

These tests cover the wiring added in ``LeoApp``: failure/chat/run event hooks,
the shared HIGH-risk approval gate, truthful health reporting and the promise
that no secret and no autonomous code change happens on the way.
"""

from __future__ import annotations

import json

import pytest

from leo import db
from leo.config import ModelConfig, ModelProviderConfig, Settings
from leo.core import LeoApp
from leo.core.app import _EVOLUTION_PENDING_LIMIT, _EVOLUTION_TRANSLATED_ID_SUFFIX
from leo.learning import LearningEngine
from leo.selfheal import SelfHealingEngine


@pytest.fixture
def mock_settings() -> Settings:
    settings = Settings()
    settings.models = ModelConfig(
        default="mock",
        fallback="mock",
        providers={"mock": ModelProviderConfig(kind="mock", model="mock-1")},
    )
    return settings


@pytest.fixture
async def app(mock_settings: Settings):
    application = LeoApp(mock_settings)
    await application.initialize()
    yield application
    await application.shutdown()


def _dump(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


async def _emit_tool_failure(app: LeoApp, error: str) -> None:
    await app.events.emit(
        "tool.failed",
        call_id="call-1",
        tool="list_files",
        actor="chat:test",
        decision="invalid_arguments",
        success=False,
        dry_run=False,
        error=error,
    )


async def _approval(action: str) -> dict:
    row = await db.fetchone(
        "SELECT * FROM approvals WHERE action=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (action,),
    )
    assert row is not None, f"no approval requested for {action}"
    return dict(row)


async def test_initialize_wires_real_engines_and_truthful_capabilities(app: LeoApp) -> None:
    assert isinstance(app.healing, SelfHealingEngine)
    assert isinstance(app.learning, LearningEngine)
    for key in ("healing", "self_healing", "healing_engine"):
        assert app.services[key] is app.healing
    for key in ("learning", "self_learning", "learning_engine"):
        assert app.services[key] is app.learning

    # The event hooks really exist on the shared bus, not just in a docstring.
    for topic in ("tool.failed", "run.failed", "chat.completed", "run.finished"):
        assert topic in app.events._handlers

    health = await app.health()
    # The pre-existing capability contract must not be widened by this feature.
    assert set(health["capabilities"]) == {
        "task_run", "agent_delegate", "schedule", "channel_send",
    }
    evolution = health["self_evolution"]
    assert evolution["healing"]["attached"] is True
    assert evolution["learning"]["attached"] is True
    assert evolution["remediation_mode"] == "proposal-only"
    assert evolution["autonomous_code_changes"] is False
    assert evolution["applied_by_app"] is False
    assert evolution["hooks_attached"] is True
    assert "tool.failed" in evolution["healing"]["governance"]["watched_topics"]
    assert evolution["healing"]["governance"]["self_edits_source"] is False
    assert evolution["learning"]["governance"]["auto_notes_enabled"] is False
    assert evolution["healing"]["applied_remediations"] == 0
    assert evolution["healing"]["active_fixes"] == 0
    assert evolution["learning"]["accepted_notes"] == 0


async def test_tool_failed_hook_produces_approval_bound_proposal_only(app: LeoApp) -> None:
    for _ in range(3):
        await _emit_tool_failure(app, "invalid arguments: unsupported 'limit'")

    assert app._evolution_hooks["tool_failed"] == 3
    incidents = await app.healing.incidents()
    assert len(incidents) == 1
    assert incidents[0]["kind"] == "schema_misuse"
    assert incidents[0]["occurrences"] == 3

    proposals = await app.healing.proposals()
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["status"] == "awaiting_approval"
    assert proposal["risk"] == "high"
    assert proposal["approval_id"]

    approval = await _approval("selfheal.apply")
    assert approval["status"] == "pending"
    assert approval["risk"] == "high"
    assert approval["actor"] == "heal:selfheal"
    metadata = json.loads(approval["metadata"])
    assert metadata["arguments_digest"] == proposal["arguments_digest"]

    # Chat actors can never resolve their own agent's approval.
    with pytest.raises(PermissionError, match="chat actors"):
        await app.permissions.resolve_approval(approval["id"], True, "chat:attacker")

    # Even a trusted human's approval does not make the app apply anything:
    # an explicit trusted applier/verifier is required and is never wired here.
    await app.operator_resolve_approval(approval["id"], True)
    assert (await db.fetchone(
        "SELECT status FROM approvals WHERE id=?", (approval["id"],)
    ))["status"] == "approved"
    assert (await app.healing.proposals())[0]["status"] == "awaiting_approval"
    assert await app.healing.snapshots() == []

    health = await app.health()
    evolution = health["self_evolution"]
    assert evolution["healing"]["incidents"] == 1
    assert evolution["healing"]["proposals"] == 1
    assert evolution["healing"]["applied_remediations"] == 0
    assert evolution["healing"]["active_fixes"] == 0
    assert evolution["healing"]["snapshots"] == 0


async def test_run_failed_hook_translates_and_redacts_uncatalogued_failures(app: LeoApp) -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    captured: list[dict] = []

    async def capture(event) -> None:
        captured.append(event.model_dump(mode="json"))

    app.events.subscribe("selfheal.*", capture)
    for _ in range(3):
        await app.events.emit(
            "run.failed",
            run_id="run-42",
            status="failed",
            decision="timeout",
            error=f"tool timed out after 10s api_key={secret}",
        )

    assert app._evolution_hooks["run_failed"] == 3
    assert "run.failed" not in app.healing.governance()["watched_topics"]
    incidents = await app.healing.incidents()
    assert len(incidents) == 1
    assert incidents[0]["kind"] == "timeout"
    assert incidents[0]["target"] == "run-42"
    assert incidents[0]["occurrences"] == 3
    assert secret not in _dump(incidents)
    assert secret not in _dump(await app.healing.proposals())
    assert secret not in _dump(captured)
    assert secret not in _dump(await app.healing.audit_trail())
    assert secret not in _dump(await app.health())


async def test_chat_completed_hook_proposes_notes_without_writing_memory(app: LeoApp) -> None:
    before = await app.memory.list(limit=100)
    result = await app.ask(
        "Remember that the release checklist lives in docs/release.md", channel="chat"
    )
    assert result["conversation_id"]
    assert app._evolution_hooks["chat_completed"] == 1
    # One turn only queues a hint: the scan is throttled, so no approval is
    # requested until the batch is flushed.
    assert len(app._evolution_learning_pending) == 1
    assert await app.learning.proposals() == []

    await app._scan_learning_batch()
    proposals = await app.learning.proposals()
    matching = [p for p in proposals if "release checklist" in p["draft"]["content"]]
    assert len(matching) == 1
    proposal = matching[0]
    assert "digest" in proposal["draft"]["tags"]
    assert proposal["status"] == "awaiting_approval"
    # The note is low-risk by category, but the write itself still needs a
    # HIGH-risk approval; that is the gate that keeps it proposal-only.
    assert proposal["risk"] == "low"
    assert proposal["policy"] == "approval"
    assert proposal["approval_id"]

    approval = await _approval("learning.note")
    assert approval["risk"] == "high"
    assert approval["actor"] == "learning:engine"
    assert approval["status"] == "pending"

    # Observation only: no memory note is created until an operator approves.
    assert len(await app.memory.list(limit=100)) == len(before)
    health = await app.health()
    assert health["self_evolution"]["learning"]["accepted_notes"] == 0
    assert health["self_evolution"]["remediation_mode"] == "proposal-only"

    with pytest.raises(PermissionError, match="chat actors"):
        await app.permissions.resolve_approval(approval["id"], True, "chat:attacker")

    await app.operator_resolve_approval(approval["id"], True)
    assert len(await app.memory.list(limit=100)) == len(before)

    # The explicit, human-approved accept is the only path that writes memory.
    from leo.learning import ApprovalRequired

    with pytest.raises(ApprovalRequired):
        await app.learning.accept(proposal["id"])
    saved = await app.learning.accept(proposal["id"], approval["id"])
    assert saved["content"] == proposal["draft"]["content"]
    assert len(await app.memory.list(limit=100)) == len(before) + 1
    assert (await app.health())["self_evolution"]["learning"]["accepted_notes"] == 1


async def test_run_finished_hook_only_learns_from_successful_runs(app: LeoApp) -> None:
    from leo.contracts import utcnow

    for index in range(3):
        await db.execute(
            "INSERT INTO tool_calls(id,tool,actor,args,risk,decision,success,duration_ms,error,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (f"call-{index}", "list_files", "chat:test", "{}", "low", "allowed",
             1, 5.0, None, utcnow()),
        )
    await app.events.emit("run.finished", run_id="run-1", status="completed")
    assert app._evolution_hooks["run_finished"] == 1
    await app._scan_learning_batch()

    proposals = await app.learning.proposals()
    assert any("Tool list_files completed successfully 3 times" in p["draft"]["content"]
               for p in proposals)

    count = len(proposals)
    await app.events.emit("run.finished", run_id="run-2", status="failed")
    assert app._evolution_hooks["run_finished"] == 2
    await app._scan_learning_batch()
    assert len(await app.learning.proposals()) == count  # failures are not a success signal


async def test_untrusted_secret_text_is_never_captured_or_learned(app: LeoApp) -> None:
    from leo.contracts import new_id, utcnow

    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    captured: list[dict] = []

    async def capture(event) -> None:
        captured.append(event.model_dump(mode="json"))

    for pattern in ("selfheal.*", "learning.*"):
        app.events.subscribe(pattern, capture)

    conversation_id = new_id("conv")
    await db.execute(
        "INSERT INTO conversations(id,channel,subject,created_at,updated_at) VALUES(?,?,?,?,?)",
        (conversation_id, "test", None, utcnow(), utcnow()),
    )
    await db.execute(
        "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
        (conversation_id, "user",
         f"Remember that my api key is {secret} and password=swordfish", utcnow()),
    )
    await app.events.emit(
        "chat.completed", conversation_id=conversation_id, channel="test", provider="mock"
    )
    for _ in range(3):
        await _emit_tool_failure(app, f"invalid arguments near api_key={secret}")
        await app.events.emit(
            "run.failed", run_id="run-secret", status="failed", decision="timeout",
            error=f"tool timed out api_key={secret}",
        )

    assert await app.learning.proposals() == []  # secret-bearing note failed closed
    assert await app.healing.proposals()  # healing still records a redacted proposal
    for value in (secret, "swordfish"):
        assert value not in _dump(await app.healing.incidents())
        assert value not in _dump(await app.healing.proposals())
        assert value not in _dump(await app.learning.proposals())
        assert value not in _dump(captured)
        assert value not in _dump(await app.health())

    audit = await db.fetchall("SELECT reason,metadata FROM audit_log")
    kv = await db.fetchall("SELECT value FROM kv")
    assert all(value not in _dump(audit) and value not in _dump(kv)
               for value in (secret, "swordfish"))


async def test_run_failed_translation_is_durable_and_replayable_after_restart(
    mock_settings: Settings,
) -> None:
    """F7: the redacted task.failed copy must survive INSERT OR IGNORE and replay."""
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    first = LeoApp(mock_settings)
    await first.initialize()
    try:
        emitted = [
            await first.events.emit(
                "run.failed",
                run_id="run-replay",
                status="failed",
                decision="timeout",
                error=f"tool timed out api_key={secret}",
                correlation_id="corr-replay",
            )
            for _ in range(3)
        ]
        assert first._evolution_hooks["run_failed"] == 3

        # The raw source rows keep their own ids; the redacted translation is a
        # separate durable row, not a dropped INSERT OR IGNORE for the same id.
        source_ids = {
            row["id"] for row in await db.fetchall("SELECT id FROM events WHERE topic='run.failed'")
        }
        assert source_ids == {event.id for event in emitted}
        rows = await db.fetchall(
            "SELECT id, topic, payload, correlation_id FROM events "
            "WHERE topic='task.failed' ORDER BY created_at, rowid"
        )
        assert len(rows) == 3
        for index, row in enumerate(rows):
            assert row["id"] == f"{emitted[index].id}{_EVOLUTION_TRANSLATED_ID_SUFFIX}"
            assert row["id"] not in source_ids
            assert row["correlation_id"] == "corr-replay"
            assert secret not in row["payload"]
            assert "[REDACTED]" in row["payload"]

        incidents = await first.healing.incidents()
        assert len(incidents) == 1
        assert incidents[0]["target"] == "run-replay"
        assert incidents[0]["occurrences"] == 3
    finally:
        await first.shutdown()

    # Simulate a restart where derived kv state is gone but the event log is not.
    await db.execute("DELETE FROM kv WHERE key LIKE 'selfheal:%'")
    second = LeoApp(mock_settings)
    await second.initialize()
    try:
        replayed = await second.healing.incidents()
        assert len(replayed) == 1
        assert replayed[0]["target"] == "run-replay"
        assert replayed[0]["occurrences"] == 3
        assert secret not in _dump(replayed)
        assert secret not in _dump(await second.healing.proposals())
        assert secret not in _dump(await second.health())

        # Replaying the same bounded history again is idempotent: no duplicate
        # incident or occurrence inflation from the deterministic ids.
        assert await second.healing.scan(limit=10) == 0
        assert len(await second.healing.incidents()) == 1
        assert (await second.healing.incidents())[0]["occurrences"] == 3
    finally:
        await second.shutdown()


async def test_health_counts_only_verified_applications_and_separates_failures(
    app: LeoApp,
) -> None:
    from leo.contracts import utcnow

    async def set_proposal(proposal_id: str, **changes: object) -> None:
        key = f"selfheal:proposal:{proposal_id}"
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (key,))
        assert row is not None
        value = json.loads(row["value"])
        value.update(changes)
        await db.execute(
            "UPDATE kv SET value=?, updated_at=? WHERE key=?",
            (json.dumps(value, ensure_ascii=False), utcnow(), key),
        )

    for error in ("invalid arguments: unsupported 'limit'",
                  "invalid arguments: unsupported 'cursor'"):
        for _ in range(3):
            await _emit_tool_failure(app, error)
    proposals = await app.healing.proposals()
    assert len(proposals) == 2
    pending, failed = sorted(proposals, key=lambda item: item["id"])

    async def health_counts() -> dict:
        return (await app.health())["self_evolution"]["healing"]

    health = await health_counts()
    assert health["applied_remediations"] == 0
    assert health["pending_remediations"] == 2
    assert health["verification_failures"] == 0
    assert health["active_fixes"] == 0

    # An "applied" status without an independent verifier is not an apply.
    await set_proposal(
        failed["id"], status="applied", verification_status="not_verified",
        applied_at=utcnow(),
    )
    health = await health_counts()
    assert health["applied_remediations"] == 0
    assert health["pending_remediations"] == 1
    assert health["verification_failures"] == 1
    assert health["remediation_status"]["unverified_applied"] == 1

    # Only a verified application is counted as applied.
    await set_proposal(
        failed["id"], status="applied", verification_status="verified",
        applied_at=utcnow(), verified_at=utcnow(), verification={"check": "observed"},
    )
    health = await health_counts()
    assert health["applied_remediations"] == 1
    assert health["pending_remediations"] == 1
    assert health["verification_failures"] == 0

    # A verification failure stays distinct from both pending and applied.
    await set_proposal(failed["id"], status="failed", verification_status="not_verified")
    health = await health_counts()
    assert health["applied_remediations"] == 0
    assert health["pending_remediations"] == 1
    assert health["verification_failures"] == 1
    assert health["verified_failed"] == 1
    assert pending["status"] == "awaiting_approval"
    assert not any(hasattr(app, name) for name in ("apply", "accept", "rollback"))


async def test_health_reports_missing_engines_truthfully(app: LeoApp, monkeypatch) -> None:
    app.healing = None
    app.learning = None
    app._detach_self_evolution()

    health = await app.health()
    evolution = health["self_evolution"]
    assert evolution["hooks_attached"] is False
    assert evolution["remediation_mode"] == "proposal-only"
    assert evolution["autonomous_code_changes"] is False
    assert "healing" not in evolution and "learning" not in evolution


async def _emit_completions(
    app: LeoApp, count: int, *, channel: str, conversation_id: str,
) -> None:
    """Simulate a long chat session without paying for N real model turns."""
    from leo.contracts import utcnow

    for index in range(count):
        stamp = utcnow()
        await db.execute(
            "INSERT OR IGNORE INTO conversations(id,channel,subject,created_at,updated_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, channel, None, stamp, stamp),
        )
        await db.execute(
            "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
            (conversation_id, "user",
             f"Remember that item {index} lives in docs/item-{index}.md", stamp),
        )
        await app.events.emit(
            "chat.completed", conversation_id=conversation_id, channel=channel,
            provider="mock", status="completed",
        )


async def _learning_approval_count() -> int:
    row = await db.fetchone(
        "SELECT COUNT(*) AS n FROM approvals WHERE action='learning.note'"
    )
    return int(row["n"] if row else 0)


async def test_400_harness_completions_produce_no_proposals_or_approvals(
    app: LeoApp,
) -> None:
    """A synthetic 400-turn run must not flood the operator with approvals."""
    app._evolution_learning_scans = 0  # discard the empty startup scan
    await _emit_completions(
        app, 400, channel="api", conversation_id="observer-human-400",
    )
    assert app._evolution_hooks["chat_completed"] == 400
    assert len(app._evolution_learning_pending) == 0
    assert app._evolution_learning_scans == 0
    assert await app.learning.proposals() == []
    assert await _learning_approval_count() == 0

    evolution = (await app.health())["self_evolution"]
    assert evolution["proposals_suppressed"] >= 400
    assert evolution["proposals_batched"] == 0
    learning = evolution["learning"]
    assert learning["harness_events_suppressed"] == 400
    assert learning["harness_proposals_enabled"] is False
    assert learning["proposals"] == 0
    assert learning["pending_hints"] == 0

    # Startup / periodic catch-up cannot mine harness history either.
    await app.evolution_catchup()
    assert await app.learning.proposals() == []
    assert await _learning_approval_count() == 0


async def test_400_real_completions_are_throttled_into_few_digest_approvals(
    app: LeoApp,
) -> None:
    """Real traffic yields a bounded number of batched, approval-gated notes."""
    app._evolution_learning_scans = 0
    app._evolution_hooks["learning_scans"] = 0
    app._evolution_proposals_batched = 0
    app._evolution_proposals_suppressed = 0
    await _emit_completions(
        app, 400, channel="chat", conversation_id="real-human-400",
    )

    proposals = await app.learning.proposals()
    assert 1 <= len(proposals) <= 4  # one digest per 100 completions, no more
    assert app._evolution_learning_scans == 4
    assert app._evolution_learning_digest_proposals == len(proposals)
    assert await _learning_approval_count() == len(proposals)
    assert all("digest" in item["draft"]["tags"] for item in proposals)
    assert all(item["risk"] == "low" and item["status"] == "awaiting_approval"
               for item in proposals)

    evolution = (await app.health())["self_evolution"]
    assert evolution["proposals_batched"] >= 1
    assert evolution["proposals_suppressed"] >= 396
    learning = evolution["learning"]
    assert learning["digest_proposals"] == len(proposals)
    assert learning["batched"] == evolution["proposals_batched"]
    assert learning["scans"] == 4
    assert learning["harness_events_suppressed"] == 0
    assert learning["proposals"] == len(proposals)
    assert learning["accepted_notes"] == 0
    assert evolution["applied_by_app"] is False

    # No digest writes memory until a human consumes its HIGH-risk approval.
    before = len(await app.memory.list(limit=100))
    approval = await _approval("learning.note")
    await app.operator_resolve_approval(approval["id"], True)
    assert len(await app.memory.list(limit=100)) == before


async def test_pending_learning_hints_queue_is_bounded(app: LeoApp) -> None:
    """Coalescing drops old hints instead of growing memory with turn count."""
    await _emit_completions(
        app, 40, channel="chat", conversation_id="real-human-queue",
    )
    assert len(app._evolution_learning_pending) == _EVOLUTION_PENDING_LIMIT
    assert app._evolution_hooks["learning_dropped"] == 40 - _EVOLUTION_PENDING_LIMIT
    assert (await app._scan_learning_batch())["batched"] >= 1
    assert len(app._evolution_learning_pending) == 0


async def test_explicit_opt_in_allows_harness_learning(app: LeoApp, monkeypatch) -> None:
    """The harness exclusion is the default, not a permanent prohibition."""
    monkeypatch.setenv("LEO_LEARNING_ALLOW_HARNESS", "1")
    await _emit_completions(
        app, 100, channel="observer", conversation_id="observer-opt-in",
    )
    proposals = await app.learning.proposals()
    assert len(proposals) == 1
    assert "digest" in proposals[0]["draft"]["tags"]
    evolution = (await app.health())["self_evolution"]
    assert evolution["learning"]["harness_proposals_enabled"] is True
    assert evolution["learning"]["harness_events_suppressed"] == 0
