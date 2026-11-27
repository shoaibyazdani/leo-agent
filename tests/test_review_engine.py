"""Post-turn review: bounded loop, real tool dispatch, evidence and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ModelResponse, SkillManifest
from leo.events import Event, EventBus
from leo.learning import LearningEngine
from leo.memory.store import MemoryStore
from leo.review import ReviewEngine
from leo.security.permissions import PermissionManagerImpl
from leo.skills.registry import SkillRegistry
from leo.tools import (
    ToolExecutorImpl,
    ToolRegistry,
    memory_propose_tools,
    memory_tools,
    skill_manage_tools,
    skill_tools,
)


class ScriptedRouter:
    """Return a fixed tool-call script, then a final answer."""

    def __init__(self, steps: list[dict]):
        self.steps = list(steps)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        step = self.steps.pop(0) if self.steps else {
            "content": "Nothing to save.", "tool_calls": [],
        }
        return ModelResponse(
            content=str(step.get("content") or ""),
            model="scripted",
            provider="script",
            tool_calls=list(step.get("tool_calls") or []),
        )


def _call(call_id: str, tool: str, **arguments):
    return {"id": call_id, "name": tool, "arguments": arguments}


@pytest.fixture
async def env(tmp_path, monkeypatch):
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    settings.learning.review.min_transcript_chars = 10
    # P2 tests exercise direct application; the staged path is covered by
    # tests/test_skill_staging.py with approval explicitly enabled.
    settings.learning.review.write_approval = False
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = (
        skill_tools.register(registry)
        + skill_manage_tools.register(registry)
        + memory_tools.register(registry)
        + memory_propose_tools.register(registry)
    )
    skills = SkillRegistry(events=events, permissions=None, actor="tester")
    await skills.ready()
    memory = MemoryStore(events)
    learning = LearningEngine(
        store=memory, events=events, permissions=permissions, settings=settings
    )
    executor = ToolExecutorImpl(registry, permissions, events, settings)
    services = {
        "skills": skills, "learning": learning, "memory": memory,
        "permissions": permissions, "events": events, "settings": settings,
        "tools": executor,
    }

    original_context = executor._context

    def with_services(call, task_key, dry_run):
        ctx = original_context(call, task_key, dry_run)
        ctx["services"] = services
        return ctx

    monkeypatch.setattr(executor, "_context", with_services)

    async def seed(
        messages: list[tuple[str, str, dict | None]] | None = None,
        *, channel: str = "cli",
    ) -> str:
        conversation_id = "conv-review"
        await db.execute(
            "INSERT OR REPLACE INTO conversations(id,channel,subject,created_at,updated_at) "
            "VALUES(?,?,?,?,?)",
            (conversation_id, channel, "review fixture", "2026-01-01T00:00:00+00:00",
             "2026-01-01T00:00:00+00:00"),
        )
        rows = messages or [
            ("user", "Please set up the staging deploy workflow for this repository.", {}),
            ("assistant", "I used the terminal tool, then verified the service is healthy.", {"run_status": "completed"}),
            ("user", "Always stage first, then promote only after a clean health check.", {}),
            ("assistant", "Understood; I will remember that ordering.", {"run_status": "completed"}),
        ]
        for role, content, meta in rows:
            await db.execute(
                "INSERT INTO messages(conversation_id,role,content,metadata,created_at) "
                "VALUES(?,?,?,?,?)",
                (conversation_id, role, content, json.dumps(meta or {}), "2026-01-01T00:00:00+00:00"),
            )
        return conversation_id

    def engine(router, **kwargs):
        return ReviewEngine(
            events, settings,
            router=router,
            executor=executor,
            skills=skills,
            learning=learning,
            artifact_dir=tmp_path / "reviews",
            **kwargs,
        )

    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events, permissions=permissions,
        registry=registry, executor=executor, skills=skills, memory=memory,
        learning=learning, services=services, names=names, seed=seed, engine=engine,
    )


async def test_review_creates_a_documentation_only_skill(env):
    assert env.settings.learning.review.write_approval is False
    conversation_id = await env.seed()
    router = ScriptedRouter([
        {"tool_calls": [_call(
            "c1", "skill_manage", action="create", name="staging-deploys",
            description="Use when deploying the staging server.",
            category="devops",
            content=(
                "# Staging Deploys\n\nDeploy staging without touching production.\n\n"
                "## When to Use\n\n- The user asks for a staging deploy.\n\n"
                "## Procedure\n\n1. Deploy staging.\n2. Verify health.\n3. Promote.\n"
            ),
        )]},
        {"content": "Created staging-deploys from the verified workflow."},
    ])
    review = env.engine(router)
    outcome = await review.run(conversation_id, explicit=True)

    assert outcome.status == "completed", outcome.error
    assert outcome.rounds == 2
    assert outcome.actions[0]["tool"] == "skill_manage"
    assert outcome.actions[0]["success"] is True
    assert outcome.artifact_path is not None
    from pathlib import Path

    artifact = json.loads(Path(outcome.artifact_path).read_text(encoding="utf-8"))
    assert artifact["status"] == "completed"
    assert artifact["transcript_digest"] == outcome.transcript_digest
    assert artifact["actions"][0]["tool"] == "skill_manage"

    record = await env.skills.get("staging-deploys")
    assert record is not None
    assert record.created_by == "agent"
    assert getattr(record, "has_handler", False) is False
    assert record.origin.get("conversation_id") == conversation_id
    assert record.origin.get("review_id") == outcome.review_id

    rows = await db.fetchall(
        "SELECT decision, metadata FROM audit_log WHERE action='review.run'"
    )
    assert len(rows) == 1
    assert rows[0]["decision"] == "completed"
    assert json.loads(rows[0]["metadata"])["review_id"] == outcome.review_id
    assert review.status()["skills_written"] == 1


async def test_patch_without_a_fresh_read_is_refused(env):
    conversation_id = await env.seed()
    await env.skills.register(SkillManifest(
        name="release-notes", description="Use when writing release notes.",
        category="docs", documentation="# Release Notes\n\nOld step.\n",
        created_by="agent",
    ))
    # Simulate a foreground turn that read the skill: the review must not
    # inherit that mark as authorization for its own write.
    from leo.tools.skill_read_marks import mark_skill_read

    mark_skill_read("release-notes")
    router = ScriptedRouter([
        {"tool_calls": [_call(
            "c1", "skill_manage", action="patch", name="release-notes",
            old_string="Old step.", new_string="New step.",
        )]},
        {"content": "The patch was refused."},
    ])
    outcome = await env.engine(router).run(conversation_id, explicit=True)
    assert outcome.status == "completed"
    assert outcome.actions[0]["success"] is False
    record = await env.skills.get("release-notes")
    assert "Old step." in record.documentation


async def test_read_then_patch_succeeds(env):
    conversation_id = await env.seed()
    await env.skills.register(SkillManifest(
        name="release-notes", description="Use when writing release notes.",
        category="docs", documentation="# Release Notes\n\nOld step.\n",
        created_by="agent",
    ))
    router = ScriptedRouter([
        {"tool_calls": [_call("c1", "skill_read", name="release-notes")]},
        {"tool_calls": [_call(
            "c2", "skill_manage", action="patch", name="release-notes",
            old_string="Old step.", new_string="New step.",
        )]},
        {"content": "Patched release-notes."},
    ])
    outcome = await env.engine(router).run(conversation_id, explicit=True)
    assert outcome.status == "completed", outcome.error
    assert [action["success"] for action in outcome.actions] == [True, True]
    record = await env.skills.get("release-notes")
    assert "New step." in record.documentation


async def test_memory_propose_goes_through_the_learning_engine(env):
    conversation_id = await env.seed()
    router = ScriptedRouter([
        {"tool_calls": [_call(
            "c1", "memory_propose",
            content="User prefers staging before production deploys.",
            category="personal", confidence=0.7,
        )]},
        {"content": "Proposed one memory note."},
    ])
    outcome = await env.engine(router).run(conversation_id, explicit=True)
    assert outcome.status == "completed", outcome.error
    assert outcome.actions[0]["tool"] == "memory_propose"
    assert outcome.actions[0]["success"] is True

    proposals = await env.learning.proposals(limit=10)
    assert len(proposals) == 1
    assert proposals[0]["draft"]["content"] == "User prefers staging before production deploys."
    assert proposals[0]["draft"]["evidence"]
    # Proposal-only: no memory row was written by propose().
    assert await env.memory.list(limit=10) == []


async def test_short_transcript_skips_without_a_model_call(env):
    conversation_id = await env.seed([
        ("user", "hi", {}),
        ("assistant", "hello", {"run_status": "completed"}),
    ])
    env.settings.learning.review.min_transcript_chars = 10_000
    router = ScriptedRouter([])
    outcome = await env.engine(router).run(conversation_id, explicit=True)
    assert outcome.status == "skipped"
    assert router.requests == []
    assert outcome.artifact_path is not None


async def test_out_of_scope_tools_are_never_executed(env):
    conversation_id = await env.seed()
    router = ScriptedRouter([
        {"tool_calls": [_call("c1", "terminal", command="echo should-not-run")]},
        {"content": "Nothing to save."},
    ])
    outcome = await env.engine(router).run(conversation_id, explicit=True)
    assert outcome.status == "completed"
    assert outcome.actions[0]["decision"] == "out_of_scope"
    assert outcome.actions[0]["success"] is False


async def test_tool_rounds_are_bounded(env):
    conversation_id = await env.seed()
    env.settings.learning.review.max_tool_rounds = 2
    router = ScriptedRouter([
        {"tool_calls": [_call("c1", "skill_list")]},
        {"tool_calls": [_call("c2", "skill_list")]},
        {"tool_calls": [_call("c3", "skill_list")]},
        {"content": "should not be reached"},
    ])
    outcome = await env.engine(router).run(conversation_id, explicit=True)
    assert outcome.status == "completed"
    assert outcome.rounds == 2
    assert len(router.requests) == 2


async def test_tool_round_cap_produces_a_progress_report(env):
    conversation_id = await env.seed()
    env.settings.learning.review.max_tool_rounds = 2
    router = ScriptedRouter([
        {"tool_calls": [_call("c1", "skill_list")]},
        {"content": "working", "tool_calls": [_call("c2", "skill_list")]},
    ])
    outcome = await env.engine(router).run(conversation_id, explicit=True)
    assert outcome.status == "completed"
    assert outcome.rounds == 2
    assert outcome.budget_exhausted is True
    for phrase in ("Progress report:", "Done so far:", "Not finished:", "Next options:"):
        assert phrase in outcome.progress_report
    assert "skill_list" in outcome.progress_report
    artifact = json.loads(Path(outcome.artifact_path).read_text(encoding="utf-8"))
    assert artifact["progress_report"] == " ".join(outcome.progress_report.split())


async def test_observe_spawns_after_the_interval(env):
    conversation_id = await env.seed()
    env.settings.learning.review.interval_turns = 2
    router = ScriptedRouter([
        {"tool_calls": [_call(
            "c1", "skill_manage", action="create", name="observed-skill",
            description="Use when proving the trigger.",
            content="# Observed\n\nA trigger probe.\n",
        )]},
        {"content": "Created observed-skill."},
    ])
    review = env.engine(router, skip_source=lambda **kwargs: False)
    event = Event(
        topic="chat.completed",
        payload={"conversation_id": conversation_id, "status": "completed", "channel": "cli"},
    )
    await review.observe(event)
    assert await env.skills.get("observed-skill") is None
    await review.observe(event)
    await review.drain()
    assert await env.skills.get("observed-skill") is not None
    assert review.status()["completed"] >= 1


async def test_direct_run_skips_a_harness_conversation(env):
    conversation_id = await env.seed(channel="harness")
    router = ScriptedRouter([])
    review = env.engine(
        router, skip_source=lambda **kwargs: kwargs.get("channel") == "harness"
    )
    outcome = await review.run(conversation_id, explicit=True)
    assert outcome.status == "skipped"
    assert "harness" in (outcome.error or "")
    assert router.requests == []


async def test_harness_sources_are_suppressed(env):
    conversation_id = await env.seed()
    env.settings.learning.review.interval_turns = 1
    router = ScriptedRouter([])
    review = env.engine(router, skip_source=lambda **kwargs: True)
    await review.observe(Event(
        topic="chat.completed",
        payload={"conversation_id": conversation_id, "status": "completed", "channel": "harness"},
    ))
    await review.drain()
    assert router.requests == []
    assert review.status()["events_suppressed"] == 1
    assert review.status()["runs"] == 0


async def test_review_engine_is_exposed_to_the_skill_scope():
    from leo.review.engine import REVIEW_SCOPE

    assert {"skill_manage", "skill_read", "memory_propose"} <= REVIEW_SCOPE
    assert "skill_invoke" not in REVIEW_SCOPE
    assert "terminal" not in REVIEW_SCOPE


async def test_refine_command_starts_a_review_with_focus(env):
    from leo.commands.builtins import refine_command
    from leo.commands.registry import CommandContext

    class _FakeReview:
        def __init__(self):
            self.calls = []

        def trigger(self, conversation_id, *, focus=None):
            self.calls.append((conversation_id, focus))
            return True

    fake = _FakeReview()
    ctx = CommandContext(
        app=SimpleNamespace(review=fake),
        principal="user", surface="chat", channel="cli",
        argv=["focus", "on", "deploys"], raw="/refine focus on deploys",
        registry=None, conversation_id="conv-review",
    )
    result = await refine_command(ctx)
    assert result.success is True
    assert fake.calls == [("conv-review", "focus on deploys")]


async def test_refine_command_is_truthful_without_a_review_service():
    from leo.commands.builtins import refine_command
    from leo.commands.registry import CommandContext

    ctx = CommandContext(
        app=SimpleNamespace(review=None),
        principal="user", surface="chat", channel="cli",
        argv=[], raw="/refine", registry=None, conversation_id="conv-review",
    )
    result = await refine_command(ctx)
    assert result.success is False
    assert result.status.value == "unsupported"


async def test_learn_authors_directly_from_a_read_only_source(env):
    from leo.contracts import RiskLevel, ToolSpec
    from leo.skills import staging

    async def read_file(args, ctx):
        return {"path": args["path"], "content": "POST /v1/items creates an item."}

    env.registry.register(
        ToolSpec(
            name="read_file", description="read a file", category="files",
            risk=RiskLevel.LOW, permissions=["fs:read"],
            input_schema={
                "type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"], "additionalProperties": False,
            },
        ),
        read_file,
    )
    env.settings.learning.review.write_approval = True
    router = ScriptedRouter([
        {"tool_calls": [_call("c1", "read_file", path="docs/api.md")]},
        {"tool_calls": [_call(
            "c2", "skill_manage", action="create", name="items-api",
            description="Use when calling the items API.",
            content=(
                "# Items API\n\nCreate items through the documented endpoint.\n\n"
                "## When to Use\n\n- The user asks to create an item.\n\n"
                "## Procedure\n\n1. `POST /v1/items`.\n"
            ),
        )]},
        {"content": "Created items-api from docs/api.md."},
    ])
    review = env.engine(router)
    outcome = await review.run("", explicit=True, mode="learn", source="docs/api.md")

    assert outcome.status == "completed", outcome.error
    assert outcome.mode == "learn"
    assert [action["tool"] for action in outcome.actions] == ["read_file", "skill_manage"]
    record = await env.skills.get("items-api")
    assert record is not None
    assert record.created_by == "agent"
    assert record.origin.get("review_id") == outcome.review_id
    # Explicit /learn writes directly even when background review writes stage.
    assert await staging.pending() == []
    from pathlib import Path

    artifact = json.loads(Path(outcome.artifact_path).read_text(encoding="utf-8"))
    assert artifact["mode"] == "learn"
    rows = await db.fetchall(
        "SELECT action FROM audit_log WHERE action IN ('learn.run','review.run')"
    )
    assert [row["action"] for row in rows] == ["learn.run"]
    assert review.status()["learn_skills"] == 1


async def test_learn_command_starts_authoring(env):
    from leo.commands.builtins import learn_command
    from leo.commands.registry import CommandContext

    class _FakeReview:
        def __init__(self):
            self.calls = []

        def trigger_learn(self, conversation_id, source, *, focus=None):
            self.calls.append((conversation_id, source))
            return True

    fake = _FakeReview()
    ctx = CommandContext(
        app=SimpleNamespace(review=fake),
        principal="operator", surface="cli", channel="cli",
        argv=["docs/api.md"], raw="/learn docs/api.md", registry=None,
        conversation_id="conv-learn",
    )
    result = await learn_command(ctx)
    assert result.success is True
    assert fake.calls == [("conv-learn", "docs/api.md")]


async def test_learn_scope_is_read_only_and_excludes_memory(env):
    from leo.review.engine import LEARN_SCOPE

    assert {"read_file", "search_files", "list_files", "dir_tree", "http_get"} <= LEARN_SCOPE
    assert "terminal" not in LEARN_SCOPE
    assert "memory_propose" not in LEARN_SCOPE
    assert "skill_manage" in LEARN_SCOPE


async def test_a_staged_skill_write_is_reported_as_staged(env):
    """The handler's flags must reach the artifact, the outcome and the event.

    Real symptom: the review artifact said ``staged: false`` and the counters
    said ``skills_written`` for exactly the write that was waiting for an
    operator, because the envelope carries the handler's result under
    ``output`` and the loop read ``staged`` from the top level.
    """
    env.settings.learning.review.write_approval = True
    conversation_id = await env.seed()
    router = ScriptedRouter([
        {"tool_calls": [_call(
            "c1", "skill_manage", action="create", name="staged-lesson",
            description="Use when the review stages a write.",
            category="development",
            content=(
                "# Staged lesson\n\n## When to Use\n\n- A background review stages a write.\n\n"
                "## Procedure\n\n1. Stage it.\n2. Wait for the operator.\n"
            ),
        )]},
        {"content": "Staged one skill write."},
    ])
    seen: list[dict] = []

    async def _capture(event):
        seen.append(event.payload)

    env.events.subscribe("review.completed", _capture)
    review = env.engine(router)
    outcome = await review.run(conversation_id, explicit=True)

    assert outcome.status == "completed", outcome.error
    assert outcome.actions[0]["staged"] is True
    assert await env.skills.get("staged-lesson") is None
    assert [item["skill"] for item in outcome.staged] == ["staged-lesson"]
    assert outcome.staged[0]["proposal_id"].startswith("sprop_")
    assert outcome.staged[0]["category"] == "development"
    assert outcome.staged[0]["description"] == "Use when the review stages a write."
    assert outcome.created == []
    assert review.status()["skills_staged"] == 1
    assert review.status()["skills_written"] == 0

    from pathlib import Path

    artifact = json.loads(Path(outcome.artifact_path).read_text(encoding="utf-8"))
    assert artifact["staged"][0]["proposal_id"] == outcome.staged[0]["proposal_id"]
    # The completion event is what a channel notifier announces, so the ids
    # have to be on the event itself rather than only in the artifact.
    assert seen and seen[0]["staged"] == outcome.staged
    assert seen[0]["created"] == []


async def test_applied_skill_creation_reaches_outcome_artifact_and_event(env):
    conversation_id = await env.seed()
    router = ScriptedRouter([
        {"tool_calls": [_call(
            "c1", "skill_manage", action="create", name="applied-lesson",
            description="Use when a review applies a skill directly.",
            category="development",
            content=(
                "# Applied lesson\n\n## When to Use\n\n- A review applies a skill.\n\n"
                "## Procedure\n\n1. Apply it.\n"
            ),
        )]},
        {"content": "Created applied-lesson."},
    ])
    seen: list[dict] = []

    async def _capture(event):
        seen.append(event.payload)

    env.events.subscribe("review.completed", _capture)
    outcome = await env.engine(router).run(conversation_id, explicit=True)

    assert outcome.status == "completed", outcome.error
    assert outcome.staged == []
    assert outcome.created == [{
        "skill": "applied-lesson",
        "category": "development",
        "description": "Use when a review applies a skill directly.",
    }]
    assert await env.skills.get("applied-lesson") is not None
    from pathlib import Path

    artifact = json.loads(Path(outcome.artifact_path).read_text(encoding="utf-8"))
    assert artifact["created"] == outcome.created
    assert seen and seen[0]["created"] == outcome.created
