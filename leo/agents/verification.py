"""Independent verification of agent results.

A model's assertion is not evidence. Verification now requires a *concrete*
anchor: either a recorded tool call that can be checked against the durable
tool-call audit table (and replayed when the caller opts into replay for a
safe call), or an artifact whose digest is recomputed from the file on disk.
Self-attested fields (``claim``, ``supports=True``, ``verified=True``) are
never sufficient on their own. Reviewers are never the producing agent.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .. import db
from ..contracts import AgentResult, ChatMessage, ModelRequest, ToolCall, new_id, utcnow
from ..logging import get_logger

_LOG = get_logger("agents.verify")

_STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "true", "false", "none", "null", "result", "output", "answer", "into", "via", "are", "was"}

# Only these source labels may describe tool or artifact evidence. A producer
# cannot bless its own claim by choosing an arbitrary ``source`` string.
_TOOL_SOURCES = frozenset({"tool", "tool_call", "executor", "scoped_tools"})
_ARTIFACT_SOURCES = frozenset({"artifact", "file", "diff", "worktree"})
_ALLOWED_DECISIONS = frozenset({"allowed", "validated"})
MAX_ARTIFACT_BYTES = 2_000_000


class VerificationResult(BaseModel):
    verified: bool
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    evidence_count: int = 0
    evaluator_agent_id: str | None = None
    evaluator_model: str | None = None
    created_at: str = Field(default_factory=utcnow)

    def as_evaluation(self) -> dict[str, Any]:
        return {
            "kind": "verification",
            "verified": self.verified,
            "score": self.score,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
            "evaluator_agent_id": self.evaluator_agent_id,
            "evaluator_model": self.evaluator_model,
            "created_at": self.created_at,
        }


def canonical(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).strip().lower()
    except (TypeError, ValueError):
        return str(value).strip().lower()


def digest(value: Any) -> str:
    """Content digest used to bind evidence to an observed output."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _tokens(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS}


async def invoke_callable(fn: Callable[..., Any], *, payload: Mapping[str, Any], **extra: Any) -> Any:
    """Call a sync or async worker/reviewer with only the arguments it accepts."""
    values: dict[str, Any] = {**dict(payload), **extra}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        result = fn(**values)
    else:
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            result = fn(**values)
        elif any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
            result = fn(**values)
        else:
            kwargs = {name: values[name] for name in params if name in values}
            if not kwargs and params:
                # Convention for simple handlers: one dict argument.
                first = next(iter(params.values()))
                if first.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    result = fn(dict(values))
                else:
                    result = fn()
            else:
                result = fn(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def parse_verdict(raw: Any) -> bool | None:
    """Parse an explicit approval/rejection from text, JSON or a mapping."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, Mapping):
        for key in ("verified", "approved", "success", "accepted"):
            if key in raw:
                value = raw[key]
                if isinstance(value, bool):
                    return value
                return parse_verdict(value)
        verdict = str(raw.get("verdict", raw.get("decision", ""))).lower()
        if verdict:
            return parse_verdict(verdict)
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    # JSON embedded in prose is common for model reviewers.
    match = re.search(r"\{\s*\"(?:verified|approved)\"\s*:\s*(true|false)\s*", text)
    if match:
        return match.group(1) == "true"
    negative = (
        "reject", "deny", "denied", "cannot", "not approve", "not verified",
        "unverified", "inconsistent", "incorrect", "fail", "failed", "false",
    )
    positive = ("approve", "verified", "consistent", "correct", "passed", "accepted", "true")
    if any(word in text for word in negative):
        return False
    if any(word in text for word in positive):
        return True
    return None


def _matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return canonical(expected) == canonical(actual) or canonical(expected) in canonical(actual)
    return canonical(expected) == canonical(actual)


def _claim_in_output(claim: Any, output: Any) -> bool:
    """Exact structural binding; deliberately not fuzzy token overlap."""
    if not isinstance(claim, str) or not claim.strip():
        return False
    if canonical(claim) in canonical(output):
        return True
    if isinstance(output, Mapping):
        return any(_claim_in_output(claim, value) for value in output.values())
    if isinstance(output, (list, tuple, set)):
        return any(_claim_in_output(claim, value) for value in output)
    return False


def _tool_anchor(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract a concrete tool-call record, or None for a self-attested item."""
    recorded = item.get("tool_call")
    if not isinstance(recorded, Mapping):
        recorded = item
    call_id = str(recorded.get("id") or recorded.get("call_id") or item.get("call_id") or "")
    tool = str(recorded.get("tool") or recorded.get("name") or item.get("tool") or "")
    if not call_id or not tool:
        return None
    source = str(item.get("source") or "").strip()
    if source and source not in _TOOL_SOURCES and not source.startswith("tool:"):
        return None
    arguments = recorded.get("arguments", item.get("arguments", {}))
    if not isinstance(arguments, Mapping):
        arguments = {}
    output = recorded.get("output", recorded.get("data", item.get("data", item.get("output"))))
    declared = str(recorded.get("digest") or item.get("digest") or item.get("sha256") or "")
    return {
        "call_id": call_id,
        "tool": tool,
        "arguments": dict(arguments),
        "output": output,
        "digest": declared,
        "claim": item.get("claim") or recorded.get("claim") or "",
        "replay": item.get("replay") is True or recorded.get("replay") is True,
    }


class VerificationEngine:
    """Rejects unsupported or inconsistent results; optionally uses a reviewer."""

    def __init__(
        self,
        registry: Any = None,
        router: Any = None,
        settings: Any = None,
        *,
        handlers: Mapping[str, Callable[..., Any]] | None = None,
        require_evidence: bool = True,
        min_score: float = 0.7,
        tools: Any = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.settings = settings
        self.handlers: dict[str, Callable[..., Any]] = dict(handlers or {})
        self.require_evidence = require_evidence
        self.min_score = min_score
        # Optional executor used only for explicit, opt-in replay of recorded
        # tool calls. A verifier without one still verifies the durable audit
        # row and the content digest, but never claims a replay happened.
        self.tools = tools

    async def _complete(self, request: ModelRequest) -> Any:
        if self.router is None:
            return None
        if hasattr(self.router, "complete"):
            return await self.router.complete(request)
        if hasattr(self.router, "route"):
            provider = await self.router.route(request)
            return await provider.complete(request)
        return None

    def _evidence_items(self, result: AgentResult) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in result.evidence or []:
            if isinstance(item, Mapping):
                items.append(dict(item))
            elif item:
                items.append({"source": "handler", "claim": str(item)})
        return items

    async def _check_tool_anchor(
        self, anchor: Mapping[str, Any]
    ) -> tuple[bool, bool, bool, list[str]]:
        """Return (audit_ok, digest_ok, replay_ok, reasons)."""
        reasons: list[str] = []
        call_id = str(anchor["call_id"])
        tool = str(anchor["tool"])
        arguments = dict(anchor["arguments"])
        declared = str(anchor["digest"])
        output = anchor["output"]

        digest_ok = bool(declared) and digest(output) == declared
        if not digest_ok:
            reasons.append(f"tool {tool!r} evidence digest does not match its recorded output")

        audit_ok = False
        try:
            row = await db.fetchone(
                "SELECT tool,args,success,decision FROM tool_calls WHERE id=?", (call_id,)
            )
        except Exception as exc:  # database failures fail closed
            row = None
            reasons.append(f"tool-call audit lookup failed: {type(exc).__name__}")
        if row is not None:
            try:
                row_args = json.loads(row["args"]) if row["args"] else {}
            except (TypeError, ValueError):
                row_args = None
            audit_ok = (
                row["tool"] == tool
                and isinstance(row_args, Mapping)
                and canonical(row_args) == canonical(arguments)
                and bool(row["success"])
                and str(row["decision"]) in _ALLOWED_DECISIONS
            )
        if not audit_ok:
            reasons.append(f"tool call {call_id!r} has no matching successful audit record")

        replay_ok = True
        if anchor["replay"]:
            if self.tools is None or not hasattr(self.tools, "execute"):
                replay_ok = False
                reasons.append(f"tool call {call_id!r} requested replay but no executor is available")
            else:
                try:
                    replayed = await self.tools.execute(
                        ToolCall(
                            id=new_id("replay"), name=tool, arguments=dict(arguments),
                            actor="verification", correlation_id=call_id,
                        )
                    )
                    replay_output = getattr(replayed, "output", replayed)
                    replay_ok = bool(getattr(replayed, "success", True)) and digest(replay_output) == declared
                except Exception as exc:
                    replay_ok = False
                    reasons.append(f"replay of tool call {call_id!r} failed: {type(exc).__name__}: {exc}")
                if not replay_ok and not any("replay" in reason for reason in reasons):
                    reasons.append(f"replay of tool call {call_id!r} produced a different digest")
        return audit_ok, digest_ok, replay_ok, reasons

    async def _check_artifact(
        self, item: Mapping[str, Any]
    ) -> tuple[bool, Any, list[str]]:
        """Recompute an artifact digest from disk; return (ok, content, reasons)."""
        reasons: list[str] = []
        raw_path = item.get("path")
        declared = str(item.get("digest") or item.get("sha256") or "")
        if not isinstance(raw_path, (str, Path)) or not declared:
            return False, None, ["artifact evidence requires a path and digest"]
        try:
            path = Path(raw_path)
            if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
                return False, None, [f"artifact {path} is missing, not a regular file or too large"]
            content = path.read_bytes()
        except (OSError, ValueError) as exc:
            return False, None, [f"artifact could not be read: {type(exc).__name__}"]
        actual = hashlib.sha256(content).hexdigest()
        ok = actual == declared
        if not ok:
            reasons.append(f"artifact {path} digest mismatch")
        claim = item.get("claim")
        if ok and isinstance(claim, str) and claim.strip():
            text = content.decode("utf-8", errors="replace")
            if canonical(claim) != canonical(text) and canonical(claim) != declared:
                ok = False
                reasons.append("artifact claim does not match the artefact content or digest")
        return ok, content.decode("utf-8", errors="replace"), reasons

    async def verify(
        self,
        result: AgentResult,
        *,
        objective: str,
        context: Mapping[str, Any] | None = None,
        expected_output: Any = None,
        peer_outputs: list[Any] | None = None,
        second_agent: str | None = None,
        second_model: str | None = None,
        require_evidence: bool | None = None,
        producer_agent_id: str | None = None,
    ) -> VerificationResult:
        context = dict(context or {})
        need_evidence = self.require_evidence if require_evidence is None else require_evidence
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        evaluator_agent: str | None = None
        evaluator_model: str | None = None

        checks["completed"] = bool(result.success) and not result.error
        if not checks["completed"]:
            reasons.append(f"agent reported failure: {result.error or 'unknown error'}")

        evidence = self._evidence_items(result)
        checks["evidence_present"] = bool(evidence)
        if not evidence:
            reasons.append("no evidence recorded for the result")

        contradictions = [
            e for e in evidence
            if e.get("supports") is False or e.get("verified") is False or e.get("valid") is False
            or str(e.get("status", "")).lower() in {"rejected", "invalid", "failed"}
        ]
        checks["evidence_not_contradictory"] = not contradictions
        if contradictions:
            reasons.append("evidence explicitly contradicts or fails to support the output")

        if expected_output is not None:
            consistent = _matches(expected_output, result.output)
        elif peer_outputs:
            # Majority consistency across independent peers. A tie has no
            # majority and fails closed; the winner is chosen by count and
            # then by canonical text so a restart can never flip a verdict
            # (set iteration is hash-randomized per process).
            canonical_outputs = [canonical(o) for o in peer_outputs]
            if canonical_outputs:
                winner, votes = sorted(
                    Counter(canonical_outputs).items(), key=lambda kv: (-kv[1], kv[0])
                )[0]
                consistent = canonical(result.output) == winner and votes * 2 > len(canonical_outputs)
            else:
                consistent = True
        else:
            consistent = not result.error
        checks["output_consistent"] = consistent
        if not consistent:
            reasons.append("output is inconsistent with the expected or peer results")

        # --- concrete evidence: tool calls, digests, artifacts, replay -----
        tool_anchors: list[dict[str, Any]] = []
        artifact_items: list[dict[str, Any]] = []
        self_attested = 0
        for item in evidence:
            anchor = _tool_anchor(item)
            if anchor is not None:
                tool_anchors.append(anchor)
                continue
            source = str(item.get("source") or "").strip().lower()
            if source in _ARTIFACT_SOURCES and (item.get("path") or item.get("digest") or item.get("sha256")):
                artifact_items.append(item)
                continue
            # A bare claim / supports=True / verified=True is a model assertion,
            # not evidence. It can never satisfy the evidence gate.
            if item.get("claim") or item.get("supports") is True or item.get("verified") is True:
                self_attested += 1

        checks["evidence_valid"] = not self_attested if need_evidence else True
        if self_attested:
            reasons.append(
                f"{self_attested} evidence item(s) are self-attested (claim/supports/verified "
                "without a tool-call audit record or a recomputed artifact digest)"
            )
        checks["evidence_verifiable"] = bool(tool_anchors or artifact_items) if need_evidence else True
        if need_evidence and not checks["evidence_verifiable"]:
            reasons.append("no verifiable tool call or content-addressed artifact in evidence")

        audit_ok = True
        digest_ok = True
        replay_ok = True
        replay_attempted = False
        for anchor in tool_anchors:
            call_audit_ok, call_digest_ok, call_replay_ok, call_reasons = await self._check_tool_anchor(anchor)
            audit_ok = audit_ok and call_audit_ok
            digest_ok = digest_ok and call_digest_ok
            replay_ok = replay_ok and call_replay_ok
            replay_attempted = replay_attempted or anchor["replay"]
            reasons.extend(call_reasons)
        if tool_anchors:
            checks["tool_calls_confirmed"] = audit_ok
            checks["evidence_digest_valid"] = digest_ok
            if replay_attempted:
                checks["replay_checked"] = replay_ok

        artifact_ok = True
        anchors: list[tuple[Any, Any, str]] = [
            (a["claim"], a["output"], a["digest"]) for a in tool_anchors
        ]
        for item in artifact_items:
            ok, content, item_reasons = await self._check_artifact(item)
            artifact_ok = artifact_ok and ok
            reasons.extend(item_reasons)
            anchors.append((item.get("claim"), content, str(item.get("digest") or item.get("sha256") or "")))
        if artifact_items:
            checks["artifact_digests_valid"] = artifact_ok

        # Bind evidence to this specific output. Exact/structural matching and
        # content digests replace the old token-overlap heuristic.
        if anchors:
            bound = any(
                (claim and (_matches(claim, result.output) or _claim_in_output(claim, result.output)))
                or (declared and declared == digest(result.output))
                for claim, _, declared in anchors
            )
        else:
            bound = False
        checks["evidence_binds_output"] = bound if need_evidence else True
        if need_evidence and not bound:
            reasons.append("verified evidence is not bound to the reported output")

        if second_agent:
            evaluator_agent = second_agent
            handler = self.handlers.get(second_agent)
            if handler is None and self.registry is not None and hasattr(self.registry, "get_handler"):
                handler = self.registry.get_handler(second_agent)
            if handler is None:
                checks["second_agent"] = False
                reasons.append(f"requested second agent {second_agent!r} is not available")
            else:
                try:
                    raw = await invoke_callable(
                        handler,
                        payload={"objective": f"Independently verify this result. Return APPROVE or REJECT.\nResult: {result.output}\nEvidence: {evidence}"},
                        context=context,
                        agent_id=second_agent,
                        role="verifier",
                    )
                    verdict = parse_verdict(raw)
                except Exception as exc:  # reviewer errors fail closed
                    verdict = False
                    reasons.append(f"second agent review failed: {exc}")
                checks["second_agent"] = verdict is True
                if verdict is not True:
                    reasons.append(f"second agent {second_agent!r} did not approve the result")

        if second_model:
            evaluator_model = second_model
            request = ModelRequest(
                messages=[ChatMessage(role="user", content=(
                    "Independently verify the following agent result using its evidence. "
                    "Reply with APPROVE or REJECT only.\n"
                    f"Objective: {objective}\nOutput: {result.output}\nEvidence: {evidence}"
                ))],
                model=second_model,
                task_category="verification",
                agent_id=producer_agent_id or result.agent_id,
            )
            try:
                response = await self._complete(request)
                verdict = parse_verdict(getattr(response, "content", response))
            except Exception as exc:
                verdict = False
                reasons.append(f"second model review failed: {exc}")
            checks["second_model"] = verdict is True
            if verdict is not True:
                reasons.append(f"second model {second_model!r} did not approve the result")

        score = sum(1.0 for ok in checks.values() if ok) / max(len(checks), 1)
        mandatory = (
            "completed", "evidence_present", "output_consistent",
            "evidence_not_contradictory", "evidence_valid", "evidence_verifiable",
            "evidence_binds_output",
        )
        if tool_anchors:
            mandatory += ("tool_calls_confirmed", "evidence_digest_valid")
            if replay_attempted:
                mandatory += ("replay_checked",)
        if artifact_items:
            mandatory += ("artifact_digests_valid",)
        if second_agent:
            mandatory += ("second_agent",)
        if second_model:
            mandatory += ("second_model",)
        verified = all(checks.get(name, False) for name in mandatory) and score >= self.min_score
        return VerificationResult(
            verified=verified,
            score=round(score, 3),
            reasons=reasons,
            checks=checks,
            evidence_count=len(evidence),
            evaluator_agent_id=evaluator_agent,
            evaluator_model=evaluator_model,
        )
