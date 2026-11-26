"""Bounded change-detection monitors for scheduled jobs.

A monitor runs through the existing egress-checked HTTP client
(:func:`leo.tools.builtins.http_get`) or the restricted script runner. Its job
is to decide whether an occurrence should wake the job at all:

* the first successful observation establishes a baseline in the schedule
  runtime kv and suppresses delivery;
* an unchanged observation suppresses the occurrence (no task, no delivery);
* a changed observation wakes the job with bounded old/new content;
* a fetch/script failure uses the failure lane and never overwrites the good
  baseline.

Model-supplied headers, cookies, URL userinfo and private-network targets are
not part of the schedule schema and are rejected here as defence in depth.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .clock import Clock, SystemClock
from .script_runner import ScriptExecutionError, ScriptResult, ScriptRunner
from .store import ScheduleStore

BASELINE_KEY = "monitor_baseline"
_PRIVATE_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
}


class MonitorError(RuntimeError):
    """A stable monitor rejection/failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class MonitorResult:
    """The outcome of one monitor observation."""

    changed: bool
    suppressed: bool
    first_run: bool
    source: str
    old_content: str = ""
    new_content: str = ""
    status_code: int | None = None
    truncated: bool = False
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    # The baseline entry this observation would persist. With ``persist=False``
    # the caller commits it atomically with the occurrence it belongs to, so a
    # crash between the fetch and task creation cannot silently consume the
    # change (the next tick would otherwise compare against the new baseline).
    baseline_value: dict[str, Any] | None = None


def _is_private_literal(host: str) -> bool:
    """Reject a literal private/loopback address before any connection."""
    text = str(host or "").strip("[]").lower()
    if text in _PRIVATE_HOSTS:
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    return not address.is_global or address.is_private or address.is_loopback


def validate_monitor_url(url: Any) -> str:
    if not isinstance(url, str) or not url.strip():
        raise MonitorError("invalid_monitor", "monitor URL must be text.")
    raw = url.strip()
    if (
        "\x00" in raw
        or len(raw) > 4096
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in raw)
    ):
        raise MonitorError("invalid_monitor", "monitor URL is invalid.")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise MonitorError(
            "invalid_monitor", "monitor URL must be a plain http(s) URL."
        )
    if parts.username or parts.password:
        raise MonitorError(
            "invalid_monitor", "monitor URLs may not embed credentials."
        )
    if _is_private_literal(parts.hostname):
        raise MonitorError(
            "monitor_private_target",
            "monitor targets must be public network addresses.",
        )
    return raw


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


class Monitor:
    """One monitor observation with durable baseline semantics."""

    def __init__(
        self,
        *,
        store: ScheduleStore,
        clock: Clock | None = None,
        script_runner: ScriptRunner | None = None,
        http_fetcher: Callable[..., Any] | None = None,
        timeout_seconds: float = 20.0,
        max_bytes: int = 200_000,
        display_chars: int = 4_000,
    ) -> None:
        self.store = store
        self.clock = clock or getattr(store, "clock", None) or SystemClock()
        self.script_runner = script_runner
        self._http_fetcher = http_fetcher
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_bytes = max(1, int(max_bytes))
        self.display_chars = max(1, int(display_chars))

    async def _fetch(self, url: str) -> tuple[str, int | None, bool]:
        fetcher = self._http_fetcher
        if fetcher is None:
            from ..tools.builtins import http_get

            fetcher = http_get
        payload = await fetcher(
            {"url": url, "max_bytes": self.max_bytes, "timeout": self.timeout_seconds},
            {},
        )
        if not isinstance(payload, Mapping):
            raise MonitorError("monitor_failed", "monitor fetch returned no data.")
        status = payload.get("status_code")
        try:
            status_code = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_code = None
        if status_code is not None and not 200 <= status_code < 300:
            raise MonitorError(
                "monitor_http_status",
                f"monitor returned HTTP {status_code}.",
            )
        return (
            str(payload.get("content") or ""),
            status_code,
            bool(payload.get("truncated")),
        )

    async def _observe(
        self, target: str, *, workdir: str | None = None
    ) -> tuple[str, int | None, bool, str]:
        lowered = target.strip().lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            url = validate_monitor_url(target)
            content, status, truncated = await self._fetch(url)
            return content, status, truncated, "url"
        if self.script_runner is None:
            raise MonitorError(
                "monitor_script_unavailable", "script monitors are not configured."
            )
        try:
            result: ScriptResult = await self.script_runner.run(
                target, [], workdir=workdir
            )
        except ScriptExecutionError as exc:
            raise MonitorError(exc.code, exc.message) from None
        if result.exit_code != 0:
            raise MonitorError(
                "monitor_script_failed",
                f"monitor script exited {result.exit_code}: "
                + _bounded(result.stderr.strip() or "no stderr", 400),
            )
        return result.stdout, None, result.truncated, "script"

    async def check(
        self,
        *,
        schedule_id: str,
        target: str,
        workdir: str | None = None,
        baseline_override: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> MonitorResult:
        """Observe the target and apply baseline suppression semantics.

        ``persist=False`` returns the baseline the caller should commit (in
        ``baseline_value``) instead of writing it. The scheduler uses this so
        the baseline and the occurrence's task are one transaction; a retried
        preflight on the same tick sees ``baseline_override`` and suppresses
        the second changed observation instead of waking twice.
        """
        if baseline_override is not None:
            baseline = dict(baseline_override)
        else:
            old_raw = await self.store.runtime_get(schedule_id, BASELINE_KEY)
            baseline = None
            if old_raw:
                try:
                    loaded = json.loads(old_raw)
                    baseline = loaded if isinstance(loaded, dict) else None
                except (TypeError, ValueError):
                    baseline = None
        try:
            content, status, truncated, source = await self._observe(
                target, workdir=workdir
            )
        except Exception as exc:  # noqa: BLE001 - the baseline must survive failure
            code = str(getattr(exc, "code", "") or "monitor_failed")
            message = str(getattr(exc, "message", "") or exc)
            return MonitorResult(
                changed=False,
                suppressed=False,
                first_run=baseline is None,
                source="error",
                error=message,
                evidence={"reason": code, "failed": True},
            )

        bounded_content = _bounded(content, self.max_bytes)
        digest = _digest(bounded_content)
        now = self.clock.now().isoformat()
        entry = {
            "digest": digest,
            "content": _bounded(bounded_content, self.display_chars),
            "fetched_at": now,
            "source": source,
        }

        async def _store_baseline() -> None:
            await self.store.runtime_set(
                schedule_id, BASELINE_KEY, json.dumps(entry, ensure_ascii=False)
            )

        if baseline is None:
            # First successful observation only establishes truth. Waking on it
            # would deliver an unasked "change" the operator never chose.
            if persist:
                await _store_baseline()
            return MonitorResult(
                changed=False,
                suppressed=True,
                first_run=True,
                source=source,
                new_content=_bounded(bounded_content, self.display_chars),
                status_code=status,
                truncated=truncated,
                evidence={"reason": "baseline", "digest": digest},
                baseline_value=entry,
            )

        old_digest = str(baseline.get("digest") or "")
        old_content = str(baseline.get("content") or "")
        if old_digest == digest:
            return MonitorResult(
                changed=False,
                suppressed=True,
                first_run=False,
                source=source,
                old_content=old_content,
                new_content=_bounded(bounded_content, self.display_chars),
                status_code=status,
                truncated=truncated,
                evidence={"reason": "unchanged", "digest": digest},
            )

        # A real change updates the baseline: the next observation must compare
        # against what the operator was just told about. When deferred, the
        # caller commits this entry with the occurrence task.
        if persist:
            await _store_baseline()
        return MonitorResult(
            changed=True,
            suppressed=False,
            first_run=False,
            source=source,
            old_content=_bounded(old_content, self.display_chars),
            new_content=_bounded(bounded_content, self.display_chars),
            status_code=status,
            truncated=truncated,
            evidence={"reason": "changed", "digest": digest, "previous": old_digest},
            baseline_value=entry,
        )


__all__ = [
    "BASELINE_KEY",
    "Monitor",
    "MonitorError",
    "MonitorResult",
    "validate_monitor_url",
]
