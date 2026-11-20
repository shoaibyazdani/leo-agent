"""Git worktree isolation for parallel coding agents.

Worktrees are created in a temporary directory. Nothing is merged by creation,
inspection or validation: ``integrate()`` is the only operation that can merge,
and it refuses unverified, untrusted or conflicting changes. No code path
deletes files in the source repository.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..contracts import new_id, utcnow
from ..logging import get_logger

_LOG = get_logger("agents.isolation")


class WorktreeError(RuntimeError):
    """The worktree operation could not be performed."""


class VerificationError(WorktreeError):
    """Changes failed validation and must not be integrated."""


@dataclass
class Worktree:
    path: Path
    branch: str
    base: str
    repo_path: Path
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "branch": self.branch, "base": self.base,
            "repo_path": str(self.repo_path), "created_at": self.created_at,
        }


@dataclass
class Change:
    path: str
    status: str  # A (added), M (modified), D (deleted), ? (untracked), R (renamed), ...

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "status": self.status}


class ValidationResult(BaseModel):
    verified: bool
    reasons: list[str] = Field(default_factory=list)
    changes: list[dict[str, str]] = Field(default_factory=list)
    diff_digest: str = ""
    checked_at: str = Field(default_factory=utcnow)


class IntegrationResult(BaseModel):
    merged: bool
    commit: str | None = None
    message: str = ""
    changes: list[dict[str, str]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def _verified(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("verified", value.get("approved", False)))
    return bool(getattr(value, "verified", False))


def _allowed(path_text: str, allowed_files: list[str] | None) -> bool:
    if allowed_files is None:
        return True
    candidate = Path(path_text)
    for allowed in allowed_files:
        root = Path(allowed)
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            if candidate.match(allowed):
                return True
    return False


class WorktreeIsolation:
    """Create, validate and (only on request) integrate isolated agent work."""

    def __init__(
        self,
        repo_path: str | Path | None = None,
        *,
        temp_root: str | Path | None = None,
        verifier: Any = None,
    ) -> None:
        self.repo_path = Path(repo_path or Path.cwd()).expanduser().resolve()
        self.temp_root = Path(temp_root).expanduser().resolve() if temp_root else Path(tempfile.gettempdir()) / "leo-worktrees"
        self.verifier = verifier
        self._worktrees: dict[str, Worktree] = {}

    async def _git(self, *args: str, cwd: Path | None = None) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(cwd or self.repo_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        # Preserve leading spaces: porcelain status uses column 1 for the index state.
        return process.returncode or 0, out.decode("utf-8", "replace").rstrip("\n"), err.decode("utf-8", "replace").strip()

    async def is_git_repo(self) -> bool:
        code, out, _ = await self._git("rev-parse", "--is-inside-work-tree")
        return code == 0 and out == "true"

    async def _require_repo(self) -> None:
        if not await self.is_git_repo():
            raise WorktreeError(f"not a git repository: {self.repo_path}; worktree isolation unavailable")

    async def create(self, *, name: str | None = None, base: str = "HEAD") -> Worktree:
        """Create a detached-feature worktree for one agent. Never merges."""
        await self._require_repo()
        code, head, err = await self._git("rev-parse", "--verify", base)
        if code != 0:
            raise WorktreeError(f"invalid base {base!r}: {err}")
        self.temp_root.mkdir(parents=True, exist_ok=True)
        token = new_id("wt")[:10]
        branch = f"leo/agent-{name or token}-{token}"
        path = self.temp_root / f"leo-{name or ''}-{token}".replace(" ", "_")
        code, _, err = await self._git("worktree", "add", "-b", branch, str(path), head)
        if code != 0:
            raise WorktreeError(f"git worktree add failed: {err}")
        worktree = Worktree(path=path, branch=branch, base=head, repo_path=self.repo_path)
        self._worktrees[str(path)] = worktree
        _LOG.debug("worktree created", extra={"agent_id": name, "task_id": branch})
        return worktree

    async def changes(self, worktree: Worktree) -> list[Change]:
        if not worktree.path.exists():
            raise WorktreeError(f"worktree no longer exists: {worktree.path}")
        code, out, err = await self._git("status", "--porcelain=v1", "--untracked-files=all", cwd=worktree.path)
        if code != 0:
            raise WorktreeError(f"git status failed: {err}")
        changes: list[Change] = []
        for line in out.splitlines():
            if len(line) < 4:
                continue
            status, path_text = line[:2], line[3:]
            if " -> " in path_text:
                path_text = path_text.split(" -> ", 1)[1]
            changes.append(Change(path=path_text.strip().strip('"'), status=status.strip() or "?"))
        return changes

    async def diff(self, worktree: Worktree) -> str:
        code, out, err = await self._git("diff", "--binary", cwd=worktree.path)
        if code != 0:
            raise WorktreeError(f"git diff failed: {err}")
        return out

    async def validate(
        self,
        worktree: Worktree,
        *,
        verification: Any = None,
        allowed_files: list[str] | None = None,
        require_verification: bool = True,
    ) -> ValidationResult:
        """Reject untrusted or unverified changes. Never mutates either tree."""
        reasons: list[str] = []
        if not worktree.path.exists():
            raise WorktreeError(f"worktree no longer exists: {worktree.path}")
        if require_verification and not _verified(verification):
            reasons.append("changes have not been verified")
        if self.verifier is not None:
            try:
                kwargs = {"worktree": worktree, "changes": await self.changes(worktree), "diff": await self.diff(worktree)}
                try:
                    outcome = self.verifier(**kwargs)
                except TypeError:
                    outcome = self.verifier(worktree, kwargs["changes"], kwargs["diff"])
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                if not _verified(outcome):
                    reasons.append("external verifier rejected the changes")
            except Exception as exc:
                reasons.append(f"external verifier failed: {exc}")

        changes = await self.changes(worktree)
        root = worktree.path.resolve()
        for change in changes:
            rel = change.path
            if rel.startswith(".git/") or rel == ".git":
                reasons.append(f"git metadata must not be modified: {rel}")
            if rel.startswith("/") or ".." in Path(rel).parts:
                reasons.append(f"path escapes the worktree: {rel}")
                continue
            try:
                resolved = (root / rel).resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                reasons.append(f"path escapes the worktree: {rel}")
                continue
            if change.status.startswith("?"):
                reasons.append(f"untracked (untrusted) file: {rel}")
            if "D" in change.status or change.status.startswith("D"):
                reasons.append(f"deletion is not allowed without explicit review: {rel}")
            if not _allowed(rel, allowed_files):
                reasons.append(f"file outside the allowed set: {rel}")

        diff = await self.diff(worktree)
        digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        verified = not reasons
        return ValidationResult(
            verified=verified, reasons=reasons,
            changes=[c.to_dict() for c in changes], diff_digest=digest,
        )

    async def integrate(
        self,
        worktree: Worktree,
        *,
        verification: Any,
        allowed_files: list[str] | None = None,
        message: str | None = None,
        target_ref: str | None = None,
    ) -> IntegrationResult:
        """Explicitly merge verified work into the source repository."""
        await self._require_repo()
        validation = await self.validate(worktree, verification=verification, allowed_files=allowed_files)
        if not validation.verified:
            raise VerificationError("refusing to integrate unverified changes: " + "; ".join(validation.reasons))
        changes = await self.changes(worktree)
        if not changes:
            return IntegrationResult(merged=False, message="no changes to integrate", changes=[c.to_dict() for c in changes])

        paths = [c.path for c in changes]
        # Protect unrelated, uncommitted work in the destination tree.
        code, status_out, err = await self._git("status", "--porcelain=v1", "--untracked-files=all", cwd=self.repo_path)
        if code != 0:
            raise WorktreeError(f"git status failed in source repo: {err}")
        dirty = {line[3:].strip() for line in status_out.splitlines() if len(line) >= 4}
        overlap = sorted(dirty.intersection(set(paths)))
        if overlap:
            raise VerificationError("destination has uncommitted changes to the same files: " + ", ".join(overlap))

        if target_ref is None:
            code, target_ref, err = await self._git("branch", "--show-current")
            if code != 0 or not target_ref:
                target_ref = "HEAD"

        code, _, err = await self._git("add", "--", *paths, cwd=worktree.path)
        if code != 0:
            raise WorktreeError(f"git add failed: {err}")
        commit_message = message or f"leo: integrate verified work from {worktree.branch}"
        code, _, err = await self._git("commit", "-m", commit_message, cwd=worktree.path)
        if code != 0:
            raise WorktreeError(f"git commit failed: {err}")
        # Conflict pre-check against the commit that will actually be merged.
        # Checking before the commit was vacuous: the branch still pointed at
        # the unmodified base, and merging an ancestor is trivially clean.
        code, _, err = await self._git("merge-tree", "--write-tree", target_ref, worktree.branch)
        if code != 0:
            raise VerificationError(f"integration would conflict with {target_ref}: {err}")
        code, _, err = await self._git("merge", "--no-ff", "--no-edit", "-m", commit_message, worktree.branch)
        if code != 0:
            # Never leave MERGE_HEAD or unmerged paths in the source repo.
            await self._git("merge", "--abort")
            raise WorktreeError(f"git merge failed and was rolled back: {err}")
        code, commit, _ = await self._git("rev-parse", "HEAD")
        return IntegrationResult(
            merged=True, commit=commit if code == 0 else None, message=commit_message,
            changes=[c.to_dict() for c in changes],
        )

    async def cleanup(self, worktree: Worktree, *, force: bool = False) -> None:
        """Remove only the worktree this helper created."""
        code, _, err = await self._git(
            "worktree", "remove", *(["--force"] if force else []), str(worktree.path)
        )
        if code != 0:
            raise WorktreeError(f"git worktree remove failed: {err}")
        if worktree.path.exists():
            shutil.rmtree(worktree.path)
        self._worktrees.pop(str(worktree.path), None)
