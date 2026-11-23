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
import fnmatch
import re
import time
from collections import deque
from typing import Any

from ..logging import get_logger

_LOG = get_logger("integrations.progress")

DEFAULT_INTERVAL_SECONDS = 4.0
DEFAULT_IDLE_SECONDS = 20.0
DEFAULT_MAX_STEPS = 3
DEFAULT_MIN_STEPS = 6
DEFAULT_SIGNIFICANT_TOOLS = (
    "shell", "terminal", "patch", "write_file", "file_move", "file_copy",
    "git_*", "browser_navigate",
)
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


# One glyph per tool family. A row reads at a glance: the shape of the work,
# then the one detail that identifies it. Text-only glyphs (no emoji) keep the
# column alignment stable on every client.
_TOOL_GLYPH = {
    "shell": "❯", "terminal": "❯",
    "write_file": "✎", "patch": "✎", "file_move": "✎", "file_copy": "✎",
    "read_file": "◦", "list_files": "◦", "search_files": "◦", "grep": "◦",
    "browser_navigate": "◇", "browser_extract_text": "◇", "browser_click": "◇",
    "http_get": "◇", "http_json": "◇", "web_search": "◇",
}
_DEFAULT_GLYPH = "·"
# Tool labels are padded to this width so the detail column lines up. A longer
# label simply pushes its own row out; alignment is a nicety, not a contract.
_LABEL_WIDTH = 9


def _step_line(tool: str, arguments: dict[str, Any] | None, ok: bool | None) -> str:
    """One aligned row: `❯ shell  `pytest -q``.

    The tool name is monospace so the column reads as a set, not as prose. The
    detail is monospace too: a command and a path are literal strings, and a
    proportional font makes them harder to scan or copy.
    """
    args = arguments or {}
    glyph = _TOOL_GLYPH.get(tool, _DEFAULT_GLYPH)
    label = tool if tool not in {"terminal"} else "shell"
    if tool in _WRITE_TOOLS:
        detail = _short_path(
            args.get("path") or args.get("destination") or args.get("source") or ""
        )
    elif tool in {"shell", "terminal"}:
        # A command is the interesting part, but the interpreter preamble around
        # it (`cd x &&`, `set -a; . env`) crowds a one-line report. Show the verb.
        detail = _command_summary(args.get("command") or "")
    else:
        detail = _short_path(args.get("command") or args.get("path") or args.get("query") or "")
    warn = " ⚠" if ok is False else ""
    if not detail:
        return f"{glyph} `{label}`{warn}"
    return f"{glyph} `{{label}}` {_mono(_preview(detail))}{warn}".replace("{label}", label)


def _mono(text: str) -> str:
    """Wrap in a code span, without letting a backtick in the text break it."""
    return "`" + text.replace("`", "'") + "`" if text else ""


def _command_summary(value: Any) -> str:
    """Reduce a shell command to the one clause a reader recognises.

    `cd /root/x && pytest -q tests/` reads as `pytest -q tests/`; a piped chain
    keeps its first meaningful clause. The preview cap still applies afterwards.
    """
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    for separator in ("&&", ";", "||"):
        head = text.split(separator, 1)[0].strip()
        if head and not head.startswith(("cd ", "set ", "source ", "export ")):
            return head
    # Only a cd/env preamble: fall through to the last clause, which is the work.
    for separator in ("&&", ";"):
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if parts:
            return parts[-1]
    return text


_COALESCED_TOOLS = frozenset({"shell", "terminal", "read_file", "search_files", "list_files"})


def _row_label(row: str) -> str:
    """The tool label of a rendered row (`❯ `shell` `x`` -> `shell`), or ''."""
    match = re.search(r"`([a-z_]+)`", row)
    return match.group(1) if match else ""


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
        self.min_steps = int(getattr(cfg, "min_steps", DEFAULT_MIN_STEPS) or DEFAULT_MIN_STEPS)
        raw_significant = getattr(cfg, "significant_tools", None)
        self.significant_tools = tuple(
            str(pattern) for pattern in (raw_significant or DEFAULT_SIGNIFICANT_TOOLS)
        )
        self._steps: deque[str] = deque(maxlen=max(self.max_steps, 12))
        self._significant_seen = False
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
        self._flushing = False
        self._verdict = "done"
        self._ticker: asyncio.Task[Any] | None = None
        self.sent = 0
        self.failed = 0

    # -- lifecycle ---------------------------------------------------------
    async def begin(self, channel: str, target: str, budget: int = 0) -> bool:
        """Start reporting for one turn. Returns False when nothing will be sent.

        A previous report is closed first: two open "⏳ Working" messages in one
        chat is how a log looks broken — the older one freezes at its last step
        while the new turn starts another.
        """
        if self._active:
            await self.end("done")
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
        self._significant_seen = False
        self._active = True
        self._last_flush = 0.0
        self._loop_iteration = None
        self._loop_max = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no running loop: report on the next explicit flush
            self._ticker = None
        else:
            self._ticker = loop.create_task(self._tick())
        return True

    async def end(self, verdict: str = "done") -> None:
        """Close the report. The finished message is the final state, not a new one.

        A completion line was appended here *and* the header states the same
        completion, so every finished turn showed the outcome twice. The header
        now owns that fact; nothing is appended.
        """
        if not self._active:
            return
        self._active = False
        self._verdict = verdict
        if self._ticker is not None:
            # end() can be reached from inside the ticker (idle finish); cancelling
            # the running task there would raise before the final line is written.
            if asyncio.current_task() is not self._ticker:
                self._ticker.cancel()
            self._ticker = None
        await self._flush(force=True)

    def _is_significant(self, tool: str) -> bool:
        """True when this tool call is real work rather than a lookup.

        `memory_search` or `read_file` on its own is an answer, not a task: the
        operator does not want a "done" footer under a one-line reply. A shell
        command, a patch or a git call is work, and earns a live report.
        """
        name = str(tool or "").strip().lower()
        if not name:
            return False
        for pattern in self.significant_tools:
            if fnmatch.fnmatchcase(name, str(pattern).lower()):
                return True
        return False

    def _worth_opening(self) -> bool:
        """Whether this turn has done enough to deserve a message at all."""
        return self._significant_seen or self._iteration >= self.min_steps

    # -- content -----------------------------------------------------------
    def note(self, tool: str, arguments: dict[str, Any] | None = None,
             ok: bool | None = None) -> None:
        """Record one executed tool call."""
        if not self._active:
            return
        self._iteration += 1
        self._last_step = time.monotonic()
        self._steps.append(_step_line(tool, arguments, ok))
        if self._is_significant(tool):
            self._significant_seen = True
        if not self._message_id and self._worth_opening():
            # The FIRST report opens immediately once the turn is worth
            # reporting, so the operator sees work start instead of waiting a
            # whole interval. Later steps must respect the throttle: forcing a
            # flush per step is what produced a progress message per tool call.
            self._last_flush = 0.0

    def set_iteration(self, iteration: int, *, max_iterations: int | None = None) -> None:
        """Declare an iterative execution context, e.g. `iter 3/10`.

        Iteration numbers are facts the caller already has; a percentage would
        be invented, because most objectives have no measurable completion
        metric. ``iteration`` without a maximum still renders (`iter 3`), which
        stays truthful.
        """
        self._loop_iteration = max(0, int(iteration))
        if max_iterations is not None:
            self._loop_max = max(0, int(max_iterations))

    def _iteration_line(self) -> str:
        """`iter 3/10 · ` appended to the status line, or ''."""
        current = getattr(self, "_loop_iteration", None)
        if current is None:
            return ""
        limit = getattr(self, "_loop_max", None)
        if limit:
            return f"iter {current}/{limit} · "
        return f"iter {current} · "

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
            # The row is `❯ `shell` `cmd``: the coalescing key is the tool label
            # inside its code span, not the first word (which is now a glyph).
            label = _row_label(line)
            for index in range(len(lines) - 1, -1, -1):
                prior = lines[index]
                if _row_label(prior) == label and label in _COALESCED_TOOLS:
                    count = 2
                    if "×" in prior:
                        try:
                            # The row is `❯ `shell` ×2 · `git status``, so the
                            # count sits between `×` and the middot.
                            count = int(prior.split("×", 1)[1].split("·", 1)[0].strip()) + 1
                        except (IndexError, ValueError):
                            count = 2
                    # Rebuild the row in the same shape as a single step: the
                    # glyph and label of the newest call, then the count and the
                    # detail. Reusing the raw line tail would print the label
                    # twice (`shell ×2 · `shell` `git status``).
                    glyph = line.split(" ", 1)[0]
                    detail = re.sub(r"^\S+\s+`[a-z_]+`\s*", "", line)
                    lines[index] = f"{glyph} `{label}` ×{count} · {detail}".rstrip()
                    break
            else:
                lines.append(line)
        return lines[-self.max_steps:]

    def render(self) -> str:
        """One compact block: a status line, then the work itself.

        The header must never repeat a step. It used to append the latest step's
        text so a single-line report still said something; the effect was
        "done in 12s — 0 step(s)" appearing twice in the same message, which
        reads like a bug because it is one.
        """
        steps = self._coalesced()
        head = self._status_line(steps)
        # A one-step report already lives entirely in the header when that step
        # is the completion line; printing it again would duplicate it.
        body = [line for line in steps if line not in head]
        return "\n".join([head, *body]) if body else head

    def _status_line(self, steps: list[str]) -> str:
        """One bold status line: `**⏳ 12s** · 3 steps`.

        Bold on the elapsed time, not on the whole line: the eye lands on how
        long the work has been running, and the step count reads as secondary.
        """
        elapsed = self._duration()
        count = self._iteration
        done = not self._active
        failed = str(getattr(self, "_verdict", "done")) not in {"done", ""}
        counted = f"{count} step{'s' if count != 1 else ''}" if count else ""
        loop = self._iteration_line()
        if done:
            icon = "⚠️" if failed else "✅"
            verb = "stopped" if failed else "done"
            # Iteration stays visible on completion: "iter 7/10" in a done
            # line tells the operator exactly how far the loop got.
            tail = (f" · {loop}{counted}" if counted else f" · {loop.rstrip(' ·')}") if loop else (
                f" · {counted}" if counted else ""
            )
            return f"{icon} **{verb} in {elapsed}**{tail}"
        tail = (f" · {loop}{counted}" if counted else f" · {loop.rstrip(' ·')}") if loop else (
            f" · {counted}" if counted else ""
        )
        return f"⏳ **{elapsed}**{tail}"

    async def _flush(self, *, force: bool = False) -> None:
        if not (self._active or force) or not self._target:
            return
        # One flush at a time. Two concurrent flushes both saw an empty
        # `_message_id` and each opened its own message, which is the duplicate
        # a reader sees as the same turn posted twice.
        if self._flushing:
            return
        now = time.monotonic()
        if not force and now - self._last_flush < self.interval:
            return
        self._flushing = True
        try:
            await self._flush_once(force=force)
        finally:
            self._flushing = False

    async def _flush_once(self, *, force: bool) -> None:
        now = time.monotonic()
        # A turn that never did enough to deserve a report must stay silent even
        # at the final forced flush: closing a message that was never opened is
        # what put a "done" footer under every one-line answer.
        if not self._message_id and not self._worth_opening():
            return
        self._last_flush = now
        text = self.render()
        if text == self._last_text and self._message_id:
            # Nothing new to say, and the message already shows this text.
            # Re-editing identical text is what Telegram answers with
            # "message is not modified", and the retry storm that followed
            # opened a fresh message every few seconds.
            return
        try:
            adapter = self.gateway._adapter(self._channel)  # noqa: SLF001 - same package
            if not self._message_id:
                result = await adapter.send(self._target, text, silent=True)
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
                if edited:
                    self._last_text = text
                else:
                    # The edit did not land. Only then is a fresh message the
                    # right answer; setting `_last_text` first made a failed edit
                    # look delivered and a retry open a duplicate.
                    result = await adapter.send(self._target, text, silent=True)
                    if isinstance(result, dict):
                        new_id = str(result.get("message_id") or "")
                        if new_id:
                            self._message_id = new_id
                    self._last_text = text
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
