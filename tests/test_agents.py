"""Agent registry, engine, verification and worktree isolation tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.agents import (
    AgentEngine,
    AgentError,
    AgentLimitError,
    AgentRegistry,
    AgentRequest,
    VerificationError,
    WorktreeError,
    WorktreeIsolation,
)
from leo.config import Settings
from leo.events import EventBus
from leo.agents.engine import MAX_MODEL_TOOL_ROUNDS
from leo.contracts import AgentResult, AgentSpec, ModelResponse, Usage

EXPECTED_IDS = [
    "general", "research", "coding", "architecture", "browser", "computer",
    "qa", "security", "devops", "data", "documentation", "integration",
]


_EVIDENCE_DIR = Path(tempfile.mkdtemp(prefix="leo-agent-evidence-"))


def evidence(claim: object) -> list[dict[str, str]]:
    """Write a real artefact so verification can recompute its digest.

    This replaces the old ``{"source": "test-fixture", "claim": ...}``
    self-attestation. The assertions in the tests are unchanged; only the
    fixture now provides the concrete evidence the production gate requires.
    """
    payload = claim if isinstance(claim, str) else json.dumps(claim, sort_keys=True, default=str)
    payload = str(payload)
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path = _EVIDENCE_DIR / f"{content_hash}.txt"
    path.write_text(payload, encoding="utf-8")
    return [{"source": "artifact", "path": str(path), "digest": content_hash, "claim": payload}]


@pytest.fixture
async def registry():
    return await AgentRegistry().initialize()


@pytest.fixture
async def engine(registry):
    return AgentEngine(registry)


async def test_registry_seeds_twenty_and_persists(registry):
    specs = await registry.list()
    # Twelve durable core roles plus the shipped executable specialist SEEDs.
    assert [spec.id for spec in specs][:12] == EXPECTED_IDS
    roster_size = len(specs)
    assert roster_size >= 24  # the program's 24-agent goal must hold
    assert all(spec.name and spec.purpose and spec.capabilities for spec in specs)
    assert all(spec.tool_permissions and spec.model_preferences for spec in specs)
    assert all(spec.context_budget > 0 and spec.task_limit > 0 for spec in specs)
    assert len(await db.fetchall("SELECT id FROM agents")) == roster_size

    second = await AgentRegistry().initialize()
    assert len(await second.list()) == roster_size  # seeding is idempotent

    dynamic = await second.register({
        "id": "temp-critic", "name": "Temp Critic", "purpose": "Temporary reviewer",
        "capabilities": ["review"], "tool_permissions": [], "model_preferences": [],
    })
    assert dynamic.id in {s.id for s in await second.list()}
    await second.enable("temp-critic", False)
    assert not (await second.get("temp-critic")).enabled
    assert set(EXPECTED_IDS) <= {s.id for s in await second.list(enabled_only=True)}
    assert await second.unregister("temp-critic")
    assert await second.get("temp-critic") is None
    assert len(await db.fetchall("SELECT id FROM agents")) == roster_size


async def test_single_delegation_with_fake_handler_and_evidence(engine, registry):
    calls: list[str] = []

    def handle(objective, context, **kwargs):
        calls.append(objective)
        return {"output": {"answer": "hello"}, "evidence": evidence("hello")}

    engine.set_handler("general", handle)
    result = await engine.delegate("Say hello", agent_id="general")

    assert isinstance(result.success, bool) and result.success
    assert result.output == {"answer": "hello"}
    assert len(result.evidence) == 1
    assert result.metadata["verification"]["verified"] is True
    assert calls == ["Say hello"]

    tasks = await db.fetchall("SELECT * FROM tasks")
    agent_tasks = [t for t in tasks if t["owner_agent"] == "general"]
    assert len(agent_tasks) == 1
    assert "correlation_id" in agent_tasks[0]["metadata"]

    runs = await registry.history("general")
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert [e["kind"] for e in runs[0]["evaluations"]] == ["verification", "execution"]
    assert (await registry.evaluations("general"))[0]["verified"] is True


async def test_parallel_dependencies_shared_state_and_conflict_detection(engine, registry):
    barrier = asyncio.Barrier(2)
    started: list[str] = []

    async def researcher(objective, context):
        started.append("research")
        await barrier.wait()
        context["shared_state"]["research_done"] = True
        return {"output": "source found", "evidence": evidence("source found")}

    async def auditor(objective, context):
        started.append("qa")
        await barrier.wait()
        return {"output": "audit passed", "evidence": evidence("audit passed")}

    def coordinator(objective, context, dependencies):
        assert {d["output"] for d in dependencies} == {"source found", "audit passed"}
        assert context["shared_state"]["research_done"] is True
        return {"output": "final report", "evidence": evidence("final report")}

    engine._handlers.update({"research": researcher, "qa": auditor, "general": coordinator})
    result = await engine.delegate(
        "Combine findings",
        agent_id="general",
        dependencies=[
            {"id": "dep-research", "objective": "Find sources", "agent_id": "research"},
            {"id": "dep-audit", "objective": "Audit findings", "agent_id": "qa"},
        ],
        parallel=True,
    )

    assert result.success and result.output == "final report"
    assert sorted(started) == ["qa", "research"]  # both ran concurrently
    assert len(await registry.history()) == 3
    rows = await db.fetchall("SELECT metadata FROM tasks")
    assert any("dep-research" in r["metadata"] for r in rows)
    assert result.metadata["conflicts"] == []

    # Conflicting peer outputs fail closed: they are reported AND the
    # delegation is refused instead of being presented as an agreed success.
    engine._handlers["research"] = lambda objective, context: {"output": "A", "evidence": evidence("A")}
    engine._handlers["qa"] = lambda objective, context: {"output": "B", "evidence": evidence("B")}
    conflicted = await engine.delegate([
        {"objective": "opposite A", "agent_id": "research"},
        {"objective": "opposite B", "agent_id": "qa"},
    ])
    assert len(conflicted) == 2
    assert all(not r.success for r in conflicted)
    assert all("conflict" in (r.error or "").lower() for r in conflicted)
    assert all(r.metadata["conflicts"] for r in conflicted)
    assert conflicted[0].metadata["conflicts"][0]["type"] == "output_conflict"
    roots = await db.fetchall("SELECT status,metadata FROM tasks WHERE owner_agent='orchestrator'")
    assert any(
        t["status"] == "failed" and "output_conflict" in (t["metadata"] or "")
        for t in roots
    )


async def test_explicit_batch_fanout_and_single_agent_for_simple_tasks(engine, registry):
    async def handler(objective, context):
        await asyncio.sleep(0.01)
        return {"output": objective, "evidence": evidence(objective)}

    engine._handlers["research"] = handler
    engine._handlers["documentation"] = handler

    results = await engine.delegate([
        {"objective": "Research topic", "agent_id": "research"},
        {"objective": "Document topic", "agent_id": "documentation"},
    ])
    assert isinstance(results, list) and len(results) == 2
    assert {r.output for r in results} == {"Research topic", "Document topic"}
    assert len(await registry.history()) == 2

    single = await engine.delegate("Just one task", agent_id="research", parallel=True)
    assert single.success
    assert len(await registry.history()) == 3  # no spawning for a simple objective


async def test_depth_and_task_limits(engine):
    settings = Settings()
    settings.security.max_agent_depth = 1
    engine.settings = settings
    engine.max_depth = 1

    async def nested(objective, context):
        await context["delegate"]("nested work", agent_id="general")
        return {"output": "unreachable", "evidence": evidence("unreachable")}

    engine._handlers["general"] = nested
    result = await engine.delegate("root")
    assert not result.success
    assert "depth" in result.error.lower()

    with pytest.raises(AgentLimitError):
        await engine.delegate("too deep", context={"depth": 3})

    # task_limit from the agent spec bounds simultaneous work for that agent.
    limits = Settings()
    engine.settings = limits
    engine.max_depth = limits.security.max_agent_depth
    spec = await engine.registry.get("research")
    spec.task_limit = 1
    engine._handlers["research"] = lambda objective, context: {"output": objective, "evidence": evidence(objective)}
    batch = await engine.delegate([
        {"objective": "first", "agent_id": "research"},
        {"objective": "second", "agent_id": "research"},
    ])
    assert all(not r.success for r in batch) or sum(r.success for r in batch) == 1


async def test_verification_rejects_missing_evidence_and_second_reviewers(engine):
    engine._handlers["general"] = lambda objective, context: {"output": "unsupported claim"}
    rejected = await engine.delegate("Claim something", agent_id="general")
    assert not rejected.success
    assert "verification failed" in rejected.error
    assert rejected.metadata["verification"]["checks"]["evidence_present"] is False

    engine._handlers["general"] = lambda objective, context: {
        "output": "verified result", "evidence": evidence("verified result"),
    }
    engine.verifier.handlers["qa"] = lambda **kwargs: {"verified": False, "verdict": "REJECT"}
    second_rejected = await engine.delegate(
        "Review this", agent_id="general", context={"second_agent": "qa"},
    )
    assert not second_rejected.success
    assert second_rejected.metadata["verification"]["evaluator_agent_id"] == "qa"

    engine.verifier.handlers["qa"] = lambda **kwargs: "APPROVE"

    class Router:
        async def complete(self, request):
            assert request.model == "mock-critic"
            return ModelResponse(content="APPROVE", model="mock-critic", provider="mock", usage=Usage())

    engine.verifier.router = Router()
    approved = await engine.delegate(
        "Review this again", agent_id="general",
        context={"second_agent": "qa", "second_model": "mock-critic"},
    )
    assert approved.success
    assert approved.metadata["verification"]["evaluator_model"] == "mock-critic"


async def test_agent_message_events(engine):
    events = EventBus()
    engine.events = events
    engine.events.subscribe("agent.message", engine._on_message)

    def handler(objective, context):
        asyncio.create_task(context["send_message"]("qa", "please review"))
        return {"output": "done", "evidence": evidence("done")}

    engine._handlers["general"] = handler
    result = await engine.delegate("Send a note", agent_id="general", context={"to_agent": "documentation"})
    await asyncio.sleep(0)  # allow the queued message task to publish

    history = await events.history("agent.message")
    assert {e.payload["to_agent"] for e in history} == {"qa", "documentation"}
    assert all(e.payload["correlation_id"] == result.metadata["correlation_id"] for e in history)
    assert all(e.payload["from_agent"] == "general" for e in history)


# --- worktree isolation -------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "leo@example.test")
    _git(repo, "config", "user.name", "LEO Test")
    (repo / "app.txt").write_text("original\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


async def test_worktree_validate_rejects_unverified_and_integrate_is_explicit(git_repo, tmp_path):
    isolation = WorktreeIsolation(git_repo, temp_root=tmp_path / "worktrees")
    assert await isolation.is_git_repo()
    worktree = await isolation.create(name="coding")

    # Creating or inspecting a worktree never changes the source repository.
    assert (git_repo / "app.txt").read_text() == "original\n"
    (worktree.path / "app.txt").write_text("updated\n", encoding="utf-8")

    unverified = await isolation.validate(worktree)
    assert not unverified.verified
    assert any("not been verified" in reason for reason in unverified.reasons)
    with pytest.raises(VerificationError):
        await isolation.integrate(worktree, verification=None)

    (worktree.path / "untrusted.txt").write_text("untracked\n", encoding="utf-8")
    with_untracked = await isolation.validate(worktree, verification={"verified": True})
    assert not with_untracked.verified
    assert any("untrusted" in reason for reason in with_untracked.reasons)
    (worktree.path / "untrusted.txt").unlink()

    validated = await isolation.validate(worktree, verification={"verified": True})
    assert validated.verified
    assert any(change["path"] == "app.txt" for change in validated.changes)
    assert (git_repo / "app.txt").read_text() == "original\n"

    merged = await isolation.integrate(worktree, verification={"verified": True}, message="apply verified work")
    assert merged.merged
    assert (git_repo / "app.txt").read_text() == "updated\n"
    await isolation.cleanup(worktree)
    assert not worktree.path.exists()


async def test_worktree_reports_non_git_repo(tmp_path):
    isolation = WorktreeIsolation(tmp_path / "plain", temp_root=tmp_path / "wt")
    (tmp_path / "plain").mkdir()
    assert not await isolation.is_git_repo()
    with pytest.raises(WorktreeError, match="not a git repository"):
        await isolation.create()


async def test_engine_isolation_never_auto_merges(engine, git_repo, tmp_path):
    def coding_handler(objective, context):
        workspace = Path(context["workspace"])
        (workspace / "app.txt").write_text("written by isolated agent\n", encoding="utf-8")
        return {"output": "changed app.txt", "evidence": evidence("changed app.txt")}

    engine._handlers["coding"] = coding_handler
    result = await engine.delegate(
        "Edit app.txt", agent_id="coding", isolation=True,
        context={"repo_path": str(git_repo), "worktree_root": str(tmp_path / "agent-wt")},
    )
    assert result.success
    assert "worktree" in result.metadata and "validation" in result.metadata
    assert (git_repo / "app.txt").read_text() == "original\n"  # never auto-merged

    integration = await engine.integrate_agent_work(result, message="explicit integration")
    assert integration.merged
    assert (git_repo / "app.txt").read_text() == "written by isolated agent\n"
    # A merged worktree has served its purpose and is cleaned up.
    info = result.metadata["worktree"]
    assert not Path(info["path"]).exists()


# ----------------------------------------------------------------------
# Per-agent tool scoping (P1 fix: declared permissions are now enforced)
# ----------------------------------------------------------------------
@pytest.fixture
async def scoped_engine():
    from leo.security.permissions import PermissionManagerImpl
    from leo.tools import ToolExecutorImpl, ToolRegistry, register_builtins

    settings = Settings()
    settings.security.allowed_roots = [Path.cwd()]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    tool_registry = ToolRegistry()
    register_builtins(tool_registry)
    executor = ToolExecutorImpl(tool_registry, permissions, events, settings)
    registry = await AgentRegistry().initialize()
    engine = AgentEngine(registry, tools=executor, events=events, settings=settings)
    return SimpleNamespace(engine=engine, registry=registry, executor=executor)


async def test_tool_scope_is_enforced_and_audited(scoped_engine):
    from leo.contracts import ToolCall

    env = scoped_engine
    coding = await env.registry.get("coding")
    scoped = env.engine._scoped_tools(coding)
    assert {spec.name for spec in scoped.specs()} == {
        "read_file", "write_file", "list_files", "search_files", "terminal",
    }
    # In-scope tool runs (terminal is a real allowlisted executable).
    allowed = await scoped.execute(
        ToolCall(name="terminal", arguments={"command": "ls"}, actor="tester")
    )
    assert allowed.success
    # Out-of-scope tool is denied before permissions are consulted.
    denied = await scoped.execute(
        ToolCall(name="http_get", arguments={"url": "https://example.com"}, actor="tester")
    )
    assert not denied.success
    assert denied.decision == "out_of_scope"
    assert "scope" in str(denied.error)
    rows = await db.fetchall("SELECT tool, decision FROM tool_calls WHERE decision='out_of_scope'")
    assert any(row["tool"] == "http_get" for row in rows)


async def test_empty_tool_scope_is_denied_by_default(scoped_engine):
    from leo.contracts import AgentSpec, ToolCall

    empty = scoped_engine.engine._scoped_tools(
        AgentSpec(id="minimal", name="Minimal", purpose="no tools", tool_permissions=[])
    )
    assert empty.specs() == []
    denied = await empty.execute(
        ToolCall(name="calculator", arguments={"expression": "1+1"}, actor="tester")
    )
    assert not denied.success
    assert denied.decision == "out_of_scope"


async def test_model_worker_advertises_only_scoped_tools(scoped_engine):
    import json as _json

    from leo.contracts import ModelResponse

    captured: list = []

    class FakeRouter:
        providers: dict = {}

        async def complete(self, request):
            captured.append(request)
            return ModelResponse(content=_json.dumps({"output": "ok"}), model="m", provider="p")

    env = scoped_engine
    env.engine.router = FakeRouter()
    security = await env.registry.get("security")
    await env.engine._model_worker(
        security, AgentRequest(objective="audit the tree"), {}
    )
    advertised = {tool["name"] for tool in captured[0].tools}
    assert "read_file" in advertised and "calculator" in advertised
    assert "write_file" not in advertised  # security is read-only by name
    assert "terminal" not in advertised and "http_get" not in advertised


async def test_handler_receives_scoped_tools_not_the_raw_executor(scoped_engine):
    from leo.agents import ScopedTools
    from leo.contracts import ToolCall

    observed: dict = {}

    def handler(objective, context, **kwargs):
        observed["context_tools"] = context["tools"]
        observed["payload_tools"] = kwargs["tools"]
        return {"output": "ok", "evidence": evidence("ok")}

    env = scoped_engine
    env.engine.set_handler("general", handler)
    result = await env.engine.delegate("Say hello", agent_id="general")
    assert result.success
    assert isinstance(observed["context_tools"], ScopedTools)
    assert observed["payload_tools"] is observed["context_tools"]
    # The general agent (filesystem+utility) cannot run the terminal tool.
    denied = await observed["context_tools"].execute(
        ToolCall(name="terminal", arguments={"command": "ls"}, actor="handler")
    )
    assert denied.decision == "out_of_scope"


# ----------------------------------------------------------------------
# Batch A fixes: dependency delivery, legal bookkeeping, merge conflicts
# ----------------------------------------------------------------------
async def test_prior_task_dependency_reaches_handler(engine):
    observed: dict = {}

    def first_handler(objective, context, **kwargs):
        return {"output": "research findings", "evidence": evidence("research findings")}

    def second_handler(objective, context, **kwargs):
        observed["dependencies"] = context["dependencies"]
        return {"output": "report written", "evidence": evidence("report written")}

    engine.set_handler("research", first_handler)
    engine.set_handler("documentation", second_handler)
    first = await engine.delegate("Research the topic", agent_id="research")
    assert first.success

    second = await engine.delegate(
        "Write the report", agent_id="documentation", dependencies=[first]
    )
    assert second.success
    deps = observed["dependencies"]
    assert isinstance(deps, list) and len(deps) == 1
    assert deps[0]["task_id"] == first.task_id
    assert deps[0]["output"] == "research findings"


async def test_root_task_bookkeeping_merges_metadata_and_stays_legal(engine):
    def handler(objective, context, **kwargs):
        return {"output": "ok", "evidence": evidence("ok")}

    engine.set_handler("general", handler)
    result = await engine.delegate("Do the thing", agent_id="general")
    assert result.success
    rows = await db.fetchall("SELECT * FROM tasks WHERE owner_agent='orchestrator'")
    root = rows[0]
    meta = json.loads(root["metadata"])
    assert meta.get("completed") is True       # bookkeeping merged in
    assert "agent_ids" in meta                 # creation-time metadata preserved
    assert root["status"] == "completed"       # via the legal completion path
    assert root["completed_at"]                # stamped by complete(), not bookkeeping


async def test_worktree_conflict_precheck_rolls_back_cleanly(engine, git_repo, tmp_path):
    """F-13: the conflict check must see the agent's commit, not the base."""
    def coding_handler(objective, context):
        workspace = Path(context["workspace"])
        (workspace / "app.txt").write_text("agent change\n", encoding="utf-8")
        return {"output": "changed app.txt", "evidence": evidence("changed app.txt")}

    engine._handlers["coding"] = coding_handler
    result = await engine.delegate(
        "Edit app.txt", agent_id="coding", isolation=True,
        context={"repo_path": str(git_repo), "worktree_root": str(tmp_path / "wt-conflict")},
    )
    assert result.success

    # The target moves to a conflicting state before integration.
    (git_repo / "app.txt").write_text("conflicting upstream change\n", encoding="utf-8")
    _git(git_repo, "add", "app.txt")
    _git(git_repo, "commit", "-m", "upstream change")

    with pytest.raises(VerificationError, match="conflict"):
        await engine.integrate_agent_work(result, message="should not merge")

    # The source repository is clean: no MERGE_HEAD, target content intact.
    assert not (git_repo / ".git" / "MERGE_HEAD").exists()
    assert (git_repo / "app.txt").read_text() == "conflicting upstream change\n"


def test_parse_verdict_negatives_win_over_positive_substrings():
    from leo.agents import parse_verdict

    assert parse_verdict("cannot approve this result") is False
    assert parse_verdict("I must deny this change") is False
    assert parse_verdict("this patch will fail review") is False
    assert parse_verdict("approved and verified") is True
    assert parse_verdict("no clear signal here") is None


async def test_peer_majority_is_required_and_deterministic():
    from leo.agents import VerificationEngine
    from leo.contracts import AgentResult

    verifier = VerificationEngine()
    result = AgentResult(
        agent_id="a", task_id="t", success=True, output="A",
        evidence=evidence("A"),
    )

    # A two-way tie has no majority and fails closed, whatever the hash seed.
    tie = await verifier.verify(result, objective="o", peer_outputs=["A", "B"])
    assert tie.verified is False

    majority = await verifier.verify(result, objective="o", peer_outputs=["A", "A", "B"])
    assert majority.verified is True

    minority = await verifier.verify(result, objective="o", peer_outputs=["B", "B", "A"])
    assert minority.verified is False


# ----------------------------------------------------------------------
# Hardening regressions: evidence gate, run records, budgets, limits
# ----------------------------------------------------------------------
async def test_nested_depth_limit_finishes_run_record(engine, registry):
    """A nested delegation blocked by the depth cap must not leak a run."""
    settings = Settings()
    settings.security.max_agent_depth = 1
    engine.settings = settings
    engine.max_depth = 1

    async def nested(objective, context):
        await context["delegate"]("nested", agent_id="general")
        return {"output": "unreachable", "evidence": evidence("unreachable")}

    engine.set_handler("general", nested)
    result = await engine.delegate("root", agent_id="general")
    assert not result.success
    assert "depth" in result.error.lower()

    runs = await registry.history("general")
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"  # never left stuck at "running"
    assert any(e.get("kind") == "execution" for e in runs[0]["evaluations"])


async def test_isolation_setup_failure_finishes_run_record(engine, registry, tmp_path):
    """A worktree setup failure is a normal failed result, not a stuck run."""
    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()
    engine.set_handler("coding", lambda objective, context: {"output": "x", "evidence": evidence("x")})
    result = await engine.delegate(
        "edit", agent_id="coding", isolation=True,
        context={"repo_path": str(non_repo), "worktree_root": str(tmp_path / "wt")},
    )
    assert not result.success
    assert "git" in (result.error or "").lower()

    runs = await registry.history("coding")
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["evaluations"]


async def test_no_evidence_result_matches_durable_task_status(engine, registry):
    """Success without evidence is refused, consistently, at every layer."""
    engine.set_handler("general", lambda objective, context: {"output": "unsupported claim"})
    result = await engine.delegate("Claim", agent_id="general", verify=False)
    assert not result.success
    assert "evidence" in (result.error or "").lower()

    tasks = await db.fetchall("SELECT * FROM tasks")
    agent_task = next(t for t in tasks if t["owner_agent"] == "general")
    root_task = next(t for t in tasks if t["owner_agent"] == "orchestrator")
    assert agent_task["status"] == "failed"
    assert root_task["status"] == "failed"
    # The root was not completed with a fabricated "returned success" claim.
    assert "delegated agent" not in (root_task["evidence"] or "")


async def test_agent_request_verify_flag_respected_and_not_mutated(engine):
    """Per-request verify=False wins and caller-owned requests stay untouched."""
    engine.set_handler("general", lambda objective, context: {
        "output": "ok", "evidence": evidence("ok"),
    })
    request = AgentRequest(objective="Do it", agent_id="general", verify=False)
    result = await engine.delegate(request, verify=True)
    assert result.success
    assert "verification" not in result.metadata  # explicit verify=False honoured
    assert request.verify is False                 # caller-owned request not mutated
    assert request.context == {}                   # nor was its context overwritten


async def test_context_budget_bounds_model_prompt(engine):
    """The seeded context_budget actually bounds the model prompt context."""
    captured: list = []

    class Router:
        providers: dict = {}

        async def complete(self, request):
            captured.append(request)
            return ModelResponse(content='{"output": "ok"}', model="m", provider="p")

    engine.router = Router()
    spec = await engine.registry.get("general")
    spec.context_budget = 120
    await engine._model_worker(
        spec, AgentRequest(objective="do it"),
        {"shared_state": {"blob": "x" * 5000}, "dependencies": [], "messages": []},
    )
    oversized = captured[0].messages[0].content.split("Context: ", 1)[1]
    assert len(oversized) <= 120
    assert oversized.endswith("[truncated]")

    await engine._model_worker(
        spec, AgentRequest(objective="do it"),
        {"shared_state": {}, "dependencies": [], "messages": []},
    )
    small = captured[1].messages[0].content.split("Context: ", 1)[1]
    assert len(small) < 120
    assert "[truncated]" not in small


async def test_negative_context_depth_cannot_bypass_limit(engine):
    """A caller-supplied negative depth must not buy extra nesting levels."""
    engine.max_depth = 1
    seen: list[int] = []

    async def nested(objective, context):
        seen.append(context["depth"])
        if context["depth"] < 1:
            await context["delegate"]("child", agent_id="general")
        return {"output": "ok", "evidence": evidence("ok")}

    engine.set_handler("general", nested)
    await engine.delegate("root", agent_id="general", context={"depth": -2})
    assert seen == [0]  # root clamped to 0; its child at depth 1 is blocked


# ----------------------------------------------------------------------
# Final verification fixes: tool rounds, concrete evidence, capability routing
# ----------------------------------------------------------------------
async def test_model_worker_runs_multi_round_tool_loop_with_provider_messages(scoped_engine):
    """Tool outputs must be fed back to the model with the matching pair."""
    env = scoped_engine

    class Router:
        providers: dict = {}

        def __init__(self) -> None:
            self.requests: list = []

        async def complete(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    content="thinking", model="m", provider="p",
                    tool_calls=[{
                        "id": "call-1", "type": "function",
                        "function": {"name": "calculator", "arguments": json.dumps({"expression": "1+1"})},
                    }],
                )
            assert [message.role for message in request.messages] == ["user", "assistant", "tool"]
            assert request.messages[1].tool_calls[0]["id"] == "call-1"
            assert request.messages[2].tool_call_id == "call-1"
            assert "2" in request.messages[2].content
            return ModelResponse(content='{"output": "the answer is 2"}', model="m", provider="p")

    router = Router()
    env.engine.router = router
    spec = AgentSpec(
        id="calc-agent", name="Calculator", purpose="calculate",
        tool_permissions=["utility"],
    )
    raw = await env.engine._model_worker(
        spec, AgentRequest(objective="what is 1+1"), {"correlation_id": "rounds"}
    )
    assert len(router.requests) == 2  # the tool result reached a second model call
    assert raw["output"] == "the answer is 2"
    assert len(raw["evidence"]) == 1
    anchor = raw["evidence"][0]
    assert anchor["tool_call"]["id"] == "call-1"
    assert anchor["tool_call"]["tool"] == "calculator"
    assert anchor["tool_call"]["arguments"] == {"expression": "1+1"}

    # The recorded call is now checkable against the durable audit table, by
    # digest, and by replay of the same call.
    anchor["replay"] = True
    result = AgentResult(
        agent_id="calc-agent", task_id="t", success=True, output="2",
        evidence=raw["evidence"],
    )
    verdict = await env.engine.verifier.verify(result, objective="what is 1+1")
    assert verdict.verified, verdict.reasons
    assert verdict.checks["tool_calls_confirmed"] is True
    assert verdict.checks["evidence_digest_valid"] is True
    assert verdict.checks["replay_checked"] is True


async def test_model_worker_tool_rounds_are_bounded(scoped_engine):
    env = scoped_engine

    class AlwaysTools:
        providers: dict = {}

        def __init__(self) -> None:
            self.count = 0

        async def complete(self, request):
            self.count += 1
            return ModelResponse(
                content="again", model="m", provider="p",
                tool_calls=[{
                    "id": f"call-{self.count}", "name": "calculator",
                    "arguments": {"expression": "1+1"},
                }],
            )

    router = AlwaysTools()
    env.engine.router = router
    spec = AgentSpec(id="loop", name="Loop", purpose="loop", tool_permissions=["utility"])
    with pytest.raises(AgentError, match="tool rounds"):
        await env.engine._model_worker(spec, AgentRequest(objective="loop"), {})
    assert router.count == MAX_MODEL_TOOL_ROUNDS


async def test_verification_fails_closed_on_self_attestation_and_bad_digests():
    from leo.agents import VerificationEngine

    verifier = VerificationEngine()
    claimed = AgentResult(
        agent_id="a", task_id="t", success=True, output="done",
        evidence=[{"source": "handler", "claim": "done", "supports": True, "verified": True}],
    )
    verdict = await verifier.verify(claimed, objective="o")
    assert verdict.verified is False
    assert verdict.checks["evidence_valid"] is False
    assert verdict.checks["evidence_verifiable"] is False
    assert any("self-attested" in reason for reason in verdict.reasons)

    forged = AgentResult(
        agent_id="a", task_id="t", success=True, output="done",
        evidence=[{
            "source": "tool", "claim": "done", "digest": "0" * 64,
            "tool_call": {
                "id": "call-never-audited", "tool": "calculator",
                "arguments": {"expression": "1+1"}, "output": 2, "success": True,
            },
        }],
    )
    forged_verdict = await verifier.verify(forged, objective="o")
    assert forged_verdict.verified is False
    assert forged_verdict.checks["evidence_digest_valid"] is False
    assert forged_verdict.checks["tool_calls_confirmed"] is False


async def test_agent_selection_uses_declared_capabilities_with_tie_break(engine, registry):
    for agent_id in ("zz-cap", "aa-cap"):
        await registry.register({
            "id": agent_id, "name": agent_id, "purpose": "capability probe",
            "capabilities": ["zephyrquartz"], "tool_permissions": [],
            "model_preferences": [],
        })
    # Equal capability score resolves to the lexicographically first id.
    assert await engine._select_agent("please do zephyrquartz", {}) == "aa-cap"

    await registry.register({
        "id": "solo-cap", "name": "Solo", "purpose": "capability probe",
        "capabilities": ["solarflare"], "tool_permissions": [], "model_preferences": [],
    })
    assert await engine._select_agent("solarflare analysis", {}) == "solo-cap"
    assert await engine._select_agent("an entirely unrelated objective", {}) == "general"
