"""Read-only signal gathering for the judgment watcher.

Everything here is a SELECT or a configured external read. Nothing here can
change a task, approve a grant, answer a conversation or call a mutating tool.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable

from .. import db
from .state import WatcherState

_LOG = logging.getLogger("leo.watchers.signals")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_API = "https://api.github.com/repos/{repo}/issues"
_MAX_TEXT = 600


@dataclass(frozen=True)
class Signal:
    id: str
    source: str
    kind: str
    topic: str
    text: str
    urgent: bool = False
    at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "topic": self.topic,
            "text": self.text,
            "urgent": self.urgent,
            "at": self.at,
        }


def _loads(value: Any, default: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return default if default is not None else {}


def _clip(value: Any, limit: int = _MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _result_text(value: Any) -> str:
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        value = _loads(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "summary", "result", "answer", "output", "text"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


async def default_github_fetcher(repo: str, *, token_env: str = "") -> list[dict[str, Any]]:
    """Read one configured repository's issues and PRs from the public API.

    The host is pinned and redirects are refused; a token is read from an
    environment variable *name* and is never included in a signal or log.
    """
    if not _REPO.fullmatch(repo):
        raise ValueError("invalid GitHub repository name")
    import httpx

    from ..integrations.secrets import credential

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "leo-watcher",
    }
    token = credential(token_env) if token_env else None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=False, trust_env=False) as client:
        response = await client.get(
            _API.format(repo=repo),
            params={"state": "all", "sort": "updated", "per_page": 20},
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("unexpected GitHub response")
    return [item for item in payload if isinstance(item, dict)]


async def gather_signals(
    *,
    state: WatcherState,
    now: datetime,
    config: Any,
    github_fetcher: Callable[..., Awaitable[list[dict[str, Any]]]] | None = None,
) -> list[Signal]:
    """Collect bounded signals from the durable store and configured sources."""
    since = _iso(now - timedelta(hours=24))
    if state.last_run_at:
        since = max(since, str(state.last_run_at))
    signals: list[Signal] = []
    failed_tasks: set[str] = set()

    rows = await db.fetchall(
        "SELECT id,topic,payload,created_at FROM events "
        "WHERE topic IN ('task.dead_letter','task.delivery_failed') "
        "AND created_at>=? ORDER BY created_at DESC LIMIT 100",
        (since,),
    )
    for row in rows:
        payload = _loads(row["payload"])
        task_id = str(payload.get("task_id") or "")
        if row["topic"] == "task.dead_letter":
            failed_tasks.add(task_id)
            reason = _clip(payload.get("error") or "task failed", 240)
            text = f"Scheduled task failed: {reason}"
        else:
            channel = _clip(payload.get("channel") or "channel", 60)
            reason = _clip(payload.get("error") or "delivery failed", 240)
            text = f"Delivery to {channel} failed: {reason}"
        signals.append(
            Signal(
                id=f"delivery:{row['id']}",
                source="delivery",
                kind="dead_letter" if row["topic"] == "task.dead_letter" else "delivery_failure",
                topic="failure",
                text=text,
                urgent=True,
                at=str(row["created_at"]),
            )
        )

    rows = await db.fetchall(
        "SELECT id,objective,status,result,metadata,updated_at FROM tasks "
        "WHERE status IN ('completed','failed') AND updated_at>=? "
        "ORDER BY updated_at DESC LIMIT 100",
        (since,),
    )
    for row in rows:
        task_id = str(row["id"])
        metadata = _loads(row["metadata"])
        payload = metadata.get("schedule_payload") if isinstance(metadata, dict) else None
        if isinstance(payload, dict) and payload.get("worker") == "watcher":
            continue  # the watcher never reports on itself
        failed = str(row["status"]) == "failed"
        if failed and task_id in failed_tasks:
            continue
        result = row["result"]
        if not failed and not (str(result or "").strip() or _loads(result, {})):
            continue  # a routine completion with no result is not news
        text = _clip(
            _result_text(row["result"])
            or (payload.get("objective") if isinstance(payload, dict) else "")
            or row["objective"]
        )
        signals.append(
            Signal(
                id=f"task:{task_id}:{row['status']}",
                source="tasks",
                kind="failed_task" if failed else "task_outcome",
                topic="failure" if failed else "task",
                text=f"Scheduled task {'failed' if failed else 'finished'}: {text}",
                urgent=failed,
                at=str(row["updated_at"]),
            )
        )

    # Latest message from role "user" with no LEO answer yet: the user is
    # waiting. An explicit waiting flag in conversation metadata also counts.
    rows = await db.fetchall(
        "SELECT c.id,c.channel,c.subject,c.metadata,m.created_at,m.role,m.content "
        "FROM conversations c JOIN messages m ON m.id=("
        "  SELECT id FROM messages WHERE conversation_id=c.id ORDER BY id DESC LIMIT 1"
        ") WHERE m.role='user' AND m.created_at<? "
        "ORDER BY m.created_at DESC LIMIT 50",
        (_iso(now - timedelta(minutes=10)),),
    )
    for row in rows:
        metadata = _loads(row["metadata"])
        if not metadata.get("waiting_on_leo", True):
            continue
        subject = _clip(row["subject"] or row["channel"] or "conversation", 80)
        signals.append(
            Signal(
                id=f"conversation:{row['id']}:{row['created_at']}",
                source="conversation",
                kind="waiting_conversation",
                topic="conversation",
                text=f"Waiting on LEO in {subject}: {_clip(row['content'], 240)}",
                urgent=False,
                at=str(row["created_at"]),
            )
        )

    for signal in await _github_signals(state=state, now=now, config=config, fetcher=github_fetcher):
        signals.append(signal)
    return signals[: max(0, int(getattr(config, "max_signals", 50) or 50))]


async def _github_signals(
    *,
    state: WatcherState,
    now: datetime,
    config: Any,
    fetcher: Callable[..., Awaitable[list[dict[str, Any]]]] | None,
) -> list[Signal]:
    repos = [str(item).strip() for item in (getattr(config, "github_repos", None) or []) if str(item).strip()]
    if not repos:
        return []
    token_env = str(getattr(config, "github_token_env", "") or "")
    fetch = fetcher or default_github_fetcher
    signals: list[Signal] = []
    snapshot = dict(state.github)
    for repo in repos:
        if not _REPO.fullmatch(repo):
            continue
        try:
            if "token_env" in inspect.signature(fetch).parameters:
                items = await fetch(repo, token_env=token_env)
            else:
                items = await fetch(repo)
        except Exception as exc:  # noqa: BLE001 - one broken source must not stop the run
            _LOG.warning("github signal source failed", extra={"repo": repo, "error": str(exc)[:200]})
            continue
        for item in items[:20]:
            number = str(item.get("number") or "")
            key = f"{repo}#{number}"
            if not number:
                continue
            comments = int(item.get("comments") or 0)
            due = str((item.get("milestone") or {}).get("due_on") or "")
            state_value = {
                "comments": comments,
                "state": str(item.get("state") or ""),
                "due": due,
            }
            previous = snapshot.get(key)
            title = _clip(item.get("title") or "untitled", 140)
            base = f"{repo} {'PR' if item.get('pull_request') else 'issue'} #{number}: {title}"
            if previous is None:
                signals.append(
                    Signal(
                        id=f"github:{key}:opened",
                        source="github",
                        kind="github_open",
                        topic="github",
                        text="New " + base,
                        at=str(item.get("updated_at") or _iso(now)),
                    )
                )
            else:
                if comments > int(previous.get("comments") or 0):
                    signals.append(
                        Signal(
                            id=f"github:{key}:comments:{comments}",
                            source="github",
                            kind="github_reply",
                            topic="reply",
                            text=f"Reply arrived on {base}",
                            urgent=True,
                            at=str(item.get("updated_at") or _iso(now)),
                        )
                    )
                if due and due != str(previous.get("due") or ""):
                    signals.append(
                        Signal(
                            id=f"github:{key}:due:{due}",
                            source="github",
                            kind="github_deadline",
                            topic="deadline",
                            text=f"Deadline moved on {base} to {due}",
                            urgent=True,
                            at=str(item.get("updated_at") or _iso(now)),
                        )
                    )
                if str(previous.get("state") or "") != state_value["state"]:
                    signals.append(
                        Signal(
                            id=f"github:{key}:state:{state_value['state']}",
                            source="github",
                            kind="github_state",
                            topic="github",
                            text=f"Status changed on {base}: {state_value['state']}",
                            at=str(item.get("updated_at") or _iso(now)),
                        )
                    )
            snapshot[key] = state_value
    state.github = dict(list(snapshot.items())[-200:])
    return signals
