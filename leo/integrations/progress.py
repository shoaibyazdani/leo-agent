"""Live progress for a working turn, on the channel that asked for it.

A long turn is silent: the operator sends a request, the agent runs twenty tool
calls, and the only thing on the screen is a typing indicator. This mirrors the
turn into ONE message that is edited in place, so the work is visible without a
flood of notifications:

    ⏳ Working — 6 min — iteration 36/250
    shell     cd /root/hermes-workspace && pytest -q
    ✍️ Writing leo/integrations/messaging.py
    patch     leo/gateway/server.py

Deliberate properties:

* One message, edited. Telegram rate-limits edits, so flushes are throttled and a
  burst of tool calls collapses into the latest state instead of many sends.
* Redacted, truncated arguments. A command can contain a token; the preview is
  squeezed through the same redaction patterns the executor uses and capped.
* Cosmetic only. A failed send or edit is counted and never raised, because a
  progress line must never break the turn it is describing.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from typing import Any

from ..logging import get_logger

_LOG = get_logger("integrations.progress")

DEFAULT_INTERVAL_SECONDS = 4.0
DEFAULT_IDLE_SECONDS = 45.0
DEFAULT_MAX_STEPS = 3
ARGUMENT_PREVIEW_CHARS = 52

# Tools whose effect is a file write read better with a marker than a bare name.
_WRITE_TOOLS = frozenset({"write_file", "patch", "file_move", "file_copy"})
_REDACTIONS = (
    re.compile(r"(?i)(token|secret|password|api[_-]?key|bearer)\s*[=:]\s*\S+"),
    re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b"),          # telegram bot tokens
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
)


def _preview(value: Any) -> str:
    text = " ".join(str(value).split())
    for pattern in _REDACTIONS:
        text = pattern.sub("[redacted]", text)
    if len(text) > ARGUMENT_PREVIEW_CHARS:
        text = text[: ARGUMENT_PREVIEW_CHARS - 1] + "…"
    return text


def _short_path(value: Any) -> str:
    """Drop the leading workspace so a path reads as `leo/app.py`, not `/root/...`."""
    text = str(value or "")
    for marker in ("/leo-agent-import/", "/hermes-workspace/"):
        if marker in text:
            return text.split(marker, 1)[1]
    return text


def _step_line(tool: str, arguments: dict[str, Any] | None, ok: bool | None) -> str:
    args = arguments or {}
    marker = "" if ok is not False else " ⚠️"
    if tool in _WRITE_TOOLS:
        target = _short_path(args.get("path") or args.get("destination") or args.get("source") or "")
        return f"✍️ {_preview(target)}{marker}"
    detail = args.get("command") or args.get("path") or args.get("query") or ""
    label = "shell" if tool in {"shell", "terminal"} else tool
    return f"{label} {_preview(_short_path(detail) if tool not in {'shell', 'terminal'} else detail)}{marker}"


class ProgressReporter:
    """Edits one chat message per working turn."""

    def __init__(self, gateway: Any, settings: Any) -> None:
        self.gateway = gateway
        self.settings = settings
        cfg = getattr(settings, "progress", None)
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.interval = float(
            getattr(cfg, "min_interval_seconds", DEFAULT_INTERVAL_SECONDS) or DEFAULT_INTERVAL_SECONDS
        )
        self.max_steps = int(getattr(cfg, "max_steps", DEFAULT_MAX_STEPS) or DEFAULT_MAX_STEPS)
        self.idle_seconds = float(
            getattr(cfg, "idle_seconds", DEFAULT_IDLE_SECONDS) or DEFAULT_IDLE_SECONDS
        )
        self._steps: deque[str] = deque(maxlen=self.max_steps)
        self._channel = ""
        self._target = ""
        self._budget = 0
        self._iteration = 0
        self._started = 0.0
        self._message_id = ""
        self._last_flush = 0.0
        self._last_step = 0.0
        self._last_text = ""
        self._active = False
        self._ticker: asyncio.Task[Any] | None = None
        self.sent = 0
        self.failed = 0

    # -- lifecycle ---------------------------------------------------------
    def begin(self, channel: str, target: str, budget: int = 0) -> bool:
        """Start reporting for one turn. Returns False when nothing will be sent."""
        if not self.enabled or not channel or not target:
            return False
        # Built-in surfaces have no outbound adapter: a CLI turn has no chat to
        # report into, and inventing one would be noise.
        try:
            adapter = self.gateway._adapter(channel)  # noqa: SLF001 - same package
        except Exception:  # noqa: BLE001 - unknown channel is simply not reportable
            return False
        if not hasattr(adapter, "send"):
            return False
        self._channel, self._target = str(channel), str(target)
        self._budget = int(budget or 0)
        self._iteration = 0
        self._started = time.monotonic()
        self._message_id = ""
        self._last_text = ""
        self._steps.clear()
        self._active = True
        self._last_flush = 0.0
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no running loop: report on the next explicit flush
            self._ticker = None
        else:
            self._ticker = loop.create_task(self._tick())
        return True

    async def end(self, verdict: str = "done") -> None:
        """Stop reporting and leave a final line in the same message."""
        if not self._active:
            return
        self._active = False
        if self._ticker is not None:
            # end() can be reached from inside the ticker (idle finish); cancelling
            # the running task there would raise before the final line is written.
            if asyncio.current_task() is not self._ticker:
                self._ticker.cancel()
            self._ticker = None
        elapsed = self._duration()
        icon = "✅" if verdict == "done" else "⚠️"
        self._steps.append(f"{icon} {verdict} in {elapsed} — {self._iteration} step(s)")
        await self._flush(force=True)

    # -- content -----------------------------------------------------------
    def note(self, tool: str, arguments: dict[str, Any] | None = None,
             ok: bool | None = None) -> None:
        """Record one executed tool call."""
        if not self._active:
            return
        self._iteration += 1
        self._last_step = time.monotonic()
        self._steps.append(_step_line(tool, arguments, ok))
        if not self._message_id and self._last_step - self._started >= 0:
            # First step: let the message appear immediately instead of waiting a
            # whole interval, so the operator sees work start.
            self._last_flush = 0.0

    # -- rendering --------------------------------------------------------
    def _duration(self) -> str:
        seconds = max(0, int(time.monotonic() - self._started))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60} min"

    def _coalesced(self) -> list[str]:
        """Collapse a run of the same tool into one line: `shell ×12 · last …`.

        Twenty shell calls in a row are one fact, not twenty lines, and a reader
        wants the shape of the work plus the current step.
        """
        lines: list[str] = []
        for line in self._steps:
            label = line.split(" ", 1)[0]
            for index in range(len(lines) - 1, -1, -1):
                prior = lines[index]
                if prior.split(" ", 1)[0] == label and label.startswith(("shell", "terminal", "read_file", "search_files")):
                    count = 2
                    if "×" in prior:
                        try:
                            count = int(prior.split("×", 1)[1].split(" ", 1)[0]) + 1
                        except (IndexError, ValueError):
                            count = 2
                    lines[index] = f"{label} ×{count} · {line.split(' ', 1)[1] if ' ' in line else ''}".rstrip()
                    break
            else:
                lines.append(line)
        return lines[-self.max_steps:]

    def render(self) -> str:
        head = f"⏳ {self._duration()} · {self._iteration} steps" if self._iteration else f"⏳ {self._duration()}"
        lines = self._coalesced()
        if lines:
            head += f" · {lines[-1].split(' ', 1)[-1][:24]}"
        return "\n".join([head, *lines])

    async def _flush(self, *, force: bool = False) -> None:
        if not (self._active or force) or not self._target:
            return
        now = time.monotonic()
        if not force and now - self._last_flush < self.interval:
            return
        self._last_flush = now
        text = self.render()
        if text == self._last_text and not force:
            # Nothing new to say. Re-editing identical text is what Telegram
            # answers with "message is not modified", and the retry storm that
            # followed opened a fresh message every few seconds.
            return
        try:
            adapter = self.gateway._adapter(self._channel)  # noqa: SLF001 - same package
            if not self._message_id:
                result = await adapter.send(self._target, text)
                if isinstance(result, dict):
                    self._message_id = str(result.get("message_id") or "")
                self._last_text = text
                self.sent += 1
                _LOG.info("progress message opened", extra={
                    "channel": self._channel, "message_id": self._message_id})
            else:
                edited = await adapter.edit_message_text(  # type: ignore[attr-defined]
                    self._target, self._message_id, text, {"inline_keyboard": []}
                )
                self._last_text = text
                if not edited:
                    # A message can be edited only for 48h and only if it still
                    # exists; when that fails, start a fresh one for the rest.
                    result = await adapter.send(self._target, text)
                    if isinstance(result, dict):
                        self._message_id = str(result.get("message_id") or "")
                    self.sent += 1
        except Exception as exc:  # noqa: BLE001 - cosmetic only
            self.failed += 1
            _LOG.warning("progress update failed", extra={
                "channel": self._channel, "error": type(exc).__name__})

    async def _tick(self) -> None:
        """Heartbeat the elapsed time, and close the message when work stops.

        The turn's own end is not observable from here - the caller may be a chat
        loop, a task, or a worker finishing over HTTP - so the report ends itself
        once no new step has arrived for ``idle_seconds``. That keeps a crashed or
        abandoned turn from leaving "Working..." on screen forever.
        """
        try:
            while self._active:
                await asyncio.sleep(self.interval)
                if not self._active:
                    return
                idle = time.monotonic() - max(self._last_step, self._started)
                if self._iteration and idle >= self.idle_seconds:
                    await self.end("done")
                    return
                self._last_flush = 0.0  # the heartbeat is always worth a flush
                await self._flush(force=True)
        except asyncio.CancelledError:  # normal shutdown of the ticker
            raise

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "active": self._active,
            "steps": self._iteration, "sent": self.sent, "failed": self.failed,
        }
