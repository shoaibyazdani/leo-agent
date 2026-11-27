"""Interactive terminal chat for LEO — ``leo chat`` (alias ``leo tui``).

A streaming REPL over the same in-process engine the dashboard uses: the
``ask_stream`` event pipeline (thinking deltas, message deltas, tool rows,
approval gates) rendered with ``rich``. No new dependencies are required —
``rich`` is already core and ``readline`` gives history/editing when present.

Operator commands (``/approve``, ``/deny``) run as the trusted
``operator:cli`` principal; a chat actor can never resolve its own gate.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import sys
import threading
import time
from typing import Any, Callable

from rich.console import Console
from rich.text import Text

from ..contracts import TurnContext, surface_capabilities
from ..security.approval_cards import (
    card_from_row,
    is_schedule_kind,
    paginate_text,
    schedule_terminal_text,
)

HELP = """\
[bold]LEO terminal chat[/bold]

  Type a message and press Enter. Streaming replies, thinking, tool rows and
  approval gates are shown live. Ctrl+C interrupts the current turn.

[bold]Commands[/bold]
  /help              this help
  /new               start a new conversation
  /id                show the current conversation id
  /model             model status and usage
  /tools             registered tools
  /agents            registered specialist agents
  /approvals         pending high-risk approvals
  /approve <id> [once|session|always]   decide a pending gate (operator)
  /deny <id> [reason]                  deny a pending gate
  Both work while a turn is still waiting for a schedule card.
  /stop              stop the turn that is running now
  /quit or /exit     leave the chat
"""


def parse_command(line: str) -> tuple[str, list[str]] | None:
    """Return ``(command, args)`` for a slash command, else ``None``."""
    text = str(line).strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


def tool_line(name: str, arguments: Any, limit: int = 120) -> str:
    """One compact line for a tool call, safe for any argument shape."""
    if arguments is None:
        return f"{name}()"
    rendered = arguments if isinstance(arguments, str) else json.dumps(arguments, default=str)
    rendered = " ".join(str(rendered).split())
    if len(rendered) > limit:
        rendered = rendered[:limit] + "…"
    return f"{name}({rendered})" if rendered else f"{name}()"


class ChatTUI:
    """A streaming terminal session. The console is injectable for tests."""

    def __init__(
        self,
        app: Any,
        console: Console | None = None,
        conversation_id: str | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.app = app
        self.console = console or Console(highlight=False)
        self.conversation_id = conversation_id
        self._input = input_fn or input
        self._streamed_text = False
        self._thinking_open = False
        self._turning = False
        self._eof = False
        self._terminal_printed: set[str] = set()

    # ------------------------------------------------------------------ run
    async def run(self) -> int:
        """Read lines until EOF or /quit. Returns a process exit code.

        Input is read by a persistent worker while a turn streams, so ``/stop``
        (and Ctrl+C) reach the turn that is running. Any other line typed
        mid-turn is queued for the next turn, except for the commands that
        must act on the live turn (``/stop`` and card decisions).
        """
        await self.app.initialize()
        interactive = sys.stdin.isatty() and self._input is input
        prompt = "\x1b[36myou ›\x1b[0m " if interactive else ""
        if interactive:
            try:
                import readline  # noqa: F401 - history and line editing
            except Exception:  # pragma: no cover - platform dependent
                pass
            self.console.print(Text.from_markup(
                f"[bold cyan]LEO[/bold cyan] terminal chat · conversation "
                f"[dim]{self.conversation_id or '(new)'}[/dim]\n"
                "[dim]Type /help for commands. /stop or Ctrl+C interrupts a turn.[/dim]\n"
            ))
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._queued: list[str] = []
        self._start_reader(prompt)
        while True:
            try:
                line = await self._next_line()
            except KeyboardInterrupt:
                self.console.print()
                return 0
            if line is None:
                self.console.print()
                return 0
            text = line.strip()
            if not text:
                continue
            command = parse_command(text)
            if command is not None:
                name, args = command
                if name in {"quit", "exit", "q"}:
                    return 0
                if await self.run_command(name, args):
                    continue
                self.console.print(f"[yellow]unknown command /{name}[/yellow] — /help")
                continue
            await self._turn_with_interrupts(text)

    def _start_reader(self, prompt: str) -> None:
        """Read stdin on a daemon thread so a turn keeps streaming while waiting.

        A daemon thread (not ``asyncio.to_thread``) matters: the built-in
        executor is joined at loop shutdown, and a worker still blocked in
        ``input()`` would hang the process on exit.
        """
        loop = asyncio.get_running_loop()

        def _deliver(value: str | None) -> None:
            try:
                loop.call_soon_threadsafe(self._lines.put_nowait, value)
            except RuntimeError:  # loop already closed at process exit
                pass

        def _read() -> None:
            while True:
                try:
                    line = self._input(prompt)
                except (EOFError, KeyboardInterrupt):
                    _deliver(None)
                    return
                _deliver(line)

        threading.Thread(target=_read, name="leo-tui-input", daemon=True).start()

    async def _next_line(self) -> str | None:
        if self._queued:
            return self._queued.pop(0)
        if self._eof:
            return None
        return await self._lines.get()

    async def _turn_with_interrupts(self, text: str) -> None:
        """Stream one turn while watching input for an interrupt-class command."""
        self._turning = True
        task = asyncio.create_task(self.turn(text))
        try:
            while not task.done():
                getter = asyncio.create_task(self._lines.get())
                try:
                    done, _ = await asyncio.wait(
                        {task, getter}, return_when=asyncio.FIRST_COMPLETED
                    )
                except KeyboardInterrupt:
                    getter.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await getter
                    await self._request_stop()
                    continue
                if getter in done:
                    # Process the line even when the turn finished at the same
                    # moment: discarding it would lose user input (or the EOF
                    # sentinel) and park the next prompt forever.
                    line = getter.result()
                    if line is None:
                        self._eof = True
                    else:
                        parsed = parse_command(line)
                        if parsed is not None and parsed[0] in {"stop", "cancel"}:
                            await self._request_stop()
                        elif (
                            parsed is not None
                            and parsed[0] in {"approve", "deny"}
                            and parsed[1]
                        ):
                            # A schedule card can be decided while the turn is
                            # still running; ordinary input stays queued.
                            await self.run_command(parsed[0], parsed[1])
                        else:
                            self._queued.append(line)
                            self.console.print(Text("  (queued for the next turn)", style="dim"))
                    continue
                getter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter
                break
            await task
        except KeyboardInterrupt:
            await self._request_stop()
            try:
                await task
            except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                self.console.print(f"[red]error:[/red] {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a TUI must not crash the session
            self.console.print(f"[red]error:[/red] {type(exc).__name__}: {exc}")
        finally:
            self._turning = False

    async def _request_stop(self) -> None:
        """Ask the live turn to stop; the server owns the truthful final row."""
        if not self._turning:
            self.console.print(Text("no turn is running", style="dim"))
            return
        if not self.conversation_id:
            self.console.print(Text("nothing is running yet", style="dim"))
            return
        await self.app.request_turn_stop(self.conversation_id, requested_by="tui:operator")
        self.console.print(Text("⏹ stop requested — this turn ends at its next step", style="yellow"))

    # ---------------------------------------------------------------- turns
    async def turn(self, text: str) -> dict[str, Any]:
        """Stream one turn; returns the final event payload (or an error dict)."""
        self._streamed_text = False
        self._thinking_open = False
        started = time.monotonic()
        final: dict[str, Any] = {}
        error: dict[str, Any] | None = None
        # The terminal is a card-capable surface even though the transport
        # channel is named "cli". Never let the channel name downgrade it.
        turn = TurnContext(
            principal="chat", channel="cli", conversation_id=self.conversation_id,
            surface="tui", capabilities=surface_capabilities("tui"),
            origin="tui", source="chat",
        )
        ask = self.app.ask_stream
        try:
            params = inspect.signature(ask).parameters.values()
            accepts_turn = any(
                param.name == "turn"
                or param.kind is inspect.Parameter.VAR_KEYWORD
                for param in params
            )
        except (TypeError, ValueError):
            accepts_turn = False
        kwargs = {"turn": turn} if accepts_turn else {}
        async for event in self._card_event_stream(
            ask(text, "cli", None, self.conversation_id, **kwargs)
        ):
            kind = event.get("type")
            if kind == "run.started":
                self.conversation_id = event.get("conversation_id") or self.conversation_id
            elif kind == "thinking.delta":
                delta = str(event.get("delta") or "")
                if delta:
                    if not self._thinking_open:
                        self.console.print(Text("◇ thinking", style="dim italic"))
                        self._thinking_open = True
                    self.console.print(Text(delta, style="dim"), end="", markup=False)
            elif kind == "message.delta":
                delta = str(event.get("delta") or "")
                if delta:
                    if self._thinking_open:
                        self.console.print()
                        self._thinking_open = False
                    self._streamed_text = True
                    self.console.print(Text(delta), end="", markup=False)
            elif kind == "tool.started":
                self.console.print(Text(
                    f"  ⚙ {tool_line(str(event.get('name') or 'tool'), event.get('arguments'))}",
                    style="dim",
                ))
            elif kind == "tool.finished":
                ok = not event.get("is_error") and event.get("success", True) is not False
                mark, style = ("✓", "green") if ok else ("✗", "red")
                detail = "" if ok else first_line(event.get("result"))
                self.console.print(Text(
                    f"  {mark} {event.get('name') or 'tool'}{(' — ' + detail) if detail else ''}",
                    style=style,
                ))
            elif kind == "approval.required":
                self.console.print(Text(
                    f"  ⚠ approval required for {event.get('name')}: {event.get('reason') or ''}",
                    style="yellow",
                ))
                self.console.print(Text(
                    f"    /approve {event.get('approval_id')}   or   /deny {event.get('approval_id')}",
                    style="yellow dim",
                ))
            elif kind in {"approval.presented", "schedule.approval_required"}:
                self._print_approval_card(event)
            elif kind == "message.finished":
                final = event
                if self._thinking_open:
                    self.console.print()
                    self._thinking_open = False
                if not self._streamed_text and str(event.get("content") or "").strip():
                    self.console.print(event.get("content"))
            elif kind == "error":
                error = event
                self.console.print(Text(
                    f"error: {event.get('message') or event.get('code') or 'unknown'}",
                    style="red",
                ))
        if self._thinking_open or self._streamed_text:
            self.console.print()
        status = str(final.get("status") or ("error" if error else "completed"))
        elapsed = time.monotonic() - started
        provider = "/".join(part for part in (final.get("provider"), final.get("model")) if part)
        meta = " · ".join(part for part in (status, provider, f"{elapsed:.1f}s") if part)
        self.console.print(Text(f"— {meta}", style="dim"))
        if final:
            final.setdefault("status", status)
            return final
        return error or {"status": status}

    # ------------------------------------------------------------- approvals
    async def _card_event_stream(self, stream: Any) -> Any:
        """Surface schedule cards emitted on the event bus into this turn.

        The approval row is still the authority. This is only a live view: it
        prints the literal card when the server records presentation, and the
        kind-preserving terminal text when the card is decided elsewhere.
        """
        events = getattr(self.app, "events", None)
        if events is None:
            async for event in stream:
                yield event
            return

        async def on_presented(event: Any) -> None:
            payload = event.payload if hasattr(event, "payload") else {}
            if str(payload.get("presented_surface") or "") != "tui":
                return
            if self.conversation_id and payload.get("conversation_id") not in (
                None, self.conversation_id
            ):
                return
            self._print_approval_card(payload)
            approval_id = str(payload.get("approval_id") or "")
            if approval_id:
                # Acknowledgement is receipt, not a decision: shown_at is
                # already recorded by present_schedule_card.
                with contextlib.suppress(Exception):
                    await self.app.permissions.acknowledge_schedule_card(
                        approval_id, surface="tui"
                    )

        async def on_resolved(event: Any) -> None:
            payload = event.payload if hasattr(event, "payload") else {}
            if not is_schedule_kind(str(payload.get("kind") or "")):
                return
            if str(payload.get("presented_surface") or "") != "tui":
                return
            approval_id = str(payload.get("approval_id") or "")
            if not approval_id or approval_id in self._terminal_printed:
                return
            card = await self.app.permissions.get_schedule_card(approval_id)
            if card is None:
                return
            self._terminal_printed.add(approval_id)
            self._print_terminal(card, str(payload.get("status") or ""))

        for topic, handler in (
            ("approval.presented", on_presented),
            ("approval.resolved", on_resolved),
        ):
            events.subscribe(topic, handler)
        try:
            async for event in stream:
                yield event
        finally:
            for topic, handler in (
                ("approval.presented", on_presented),
                ("approval.resolved", on_resolved),
            ):
                events.unsubscribe(topic, handler)

    def _print_terminal(self, card: Any, status: str) -> None:
        text = schedule_terminal_text(card, status=status or str(card.status or ""))
        for line in text.splitlines():
            self.console.print(Text(line), markup=False)

    def _print_approval_card(self, card: Any) -> None:
        """Render the full literal card plus every offered choice.

        This is a projection only: it neither records presentation nor decides
        anything. /approvals calls present_schedule_card first when the TUI is
        the surface that actually showed the card.
        """
        payload = card.model_dump(mode="json") if hasattr(card, "model_dump") else dict(card or {})
        title = str(payload.get("title") or "Approval needed")
        self.console.print(Text(f"  ⏰ {title}", style="yellow"))
        body = str(payload.get("body") or "")
        limit = surface_capabilities("tui").max_card_chars
        pages = paginate_text(body, limit)
        if len(pages) == 1:
            for line in body.splitlines() or [""]:
                self.console.print(Text("    " + line), markup=False)
        else:
            for index, page in enumerate(pages):
                self.console.print(Text(
                    f"    [page {index + 1}/{len(pages)}]", style="dim"
                ))
                for line in page.splitlines() or [""]:
                    self.console.print(Text("    " + line), markup=False)
        choices = payload.get("choices") or []
        labels = payload.get("choice_labels") or []
        rendered = []
        for index, item in enumerate(choices):
            if isinstance(item, dict):
                value = str(item.get("value") or "")
                label = str(item.get("label") or value)
            else:
                value = str(item)
                label = str(labels[index]) if index < len(labels) else value
            rendered.append(f"{label} [{value}]")
        if rendered:
            self.console.print(Text("    choices: " + "  |  ".join(rendered), style="yellow dim"))
        deadline = str(payload.get("deadline") or "").strip()
        if deadline:
            self.console.print(Text("    " + deadline, style="dim"))
        self.console.print(Text(
            f"    /approve {payload.get('approval_id')} <choice>   or   "
            f"/deny {payload.get('approval_id')}",
            style="yellow dim",
        ))

    async def _resolve_schedule_card_tui(
        self, approval_id: str, *, approved: bool, choice: str,
        deny_reason: str | None = None,
    ) -> dict[str, Any]:
        """Present-on-display then decide a schedule card as the operator CLI."""
        from .. import db
        from ..security.approval_cards import normalize_choice

        row = await db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None:
            return {"success": False, "error": "approval not found"}
        card = card_from_row(row)
        canonical = normalize_choice("deny" if not approved else choice, card.kind)
        if canonical is None:
            return {
                "success": False,
                "error": f"choice must be one of: {', '.join(card.choices)}",
            }
        if not row["shown_at"]:
            try:
                await self.app.permissions.present_schedule_card(
                    approval_id, surface="tui"
                )
            except Exception as exc:  # noqa: BLE001 - report, never crash the REPL
                return {"success": False, "error": str(exc)}
        with contextlib.suppress(Exception):
            await self.app.permissions.acknowledge_schedule_card(
                approval_id, surface="tui"
            )
        try:
            resolution = await self.app.permissions.resolve_schedule_card(
                approval_id,
                choice=canonical,
                resolver="operator:cli",
                confirm_proposal_digest=str(row["digest"] or ""),
                deny_reason=deny_reason,
            )
        except Exception as exc:  # noqa: BLE001 - report, never crash the REPL
            return {"success": False, "error": str(exc)}
        self._terminal_printed.add(approval_id)
        updated = await self.app.permissions.get_schedule_card(approval_id)
        if updated is not None:
            self._print_terminal(updated, resolution.status)
        return {
            "success": resolution.status == "approved",
            "status": resolution.status,
            "outcome": resolution.outcome,
            "grant_id": resolution.grant_id,
            "error": (
                None if resolution.status == "approved"
                else resolution.outcome or resolution.reason
            ),
        }

    # ------------------------------------------------------------- commands
    async def run_command(self, name: str, args: list[str]) -> bool:
        """Execute a slash command. False means unknown (caller prints help)."""
        if name == "help":
            self.console.print(Text.from_markup(HELP))
        elif name == "new":
            self.conversation_id = None
            self.console.print(Text("new conversation", style="dim"))
        elif name == "id":
            self.console.print(Text(self.conversation_id or "(none yet)", style="dim"))
        elif name == "model":
            status = self.app.settings.model_status()
            routable = ", ".join(status.get("routable_providers") or []) or "none"
            self.console.print(Text(
                f"default={status.get('default')}  routable={routable}  "
                f"simulated={status.get('simulated')}",
                style="dim",
            ))
        elif name == "tools":
            specs = self.app.registry.list_specs() if hasattr(self.app.registry, "list_specs") else []
            self.console.print(Text(f"{len(specs)} tools: " + ", ".join(s.name for s in specs), style="dim"))
        elif name == "agents":
            agents = await self.app.agent_registry.list() if hasattr(self.app, "agent_registry") else []
            self.console.print(Text(f"{len(agents)} agents: " + ", ".join(a.name for a in agents), style="dim"))
        elif name == "approvals":
            pending = await self.app.permissions.pending_approvals()
            if not pending:
                self.console.print(Text("no pending approvals", style="dim"))
            for row in pending:
                if is_schedule_kind(row.get("kind")):
                    # Listing the full card on this capable surface is the
                    # presentation event; the decision still needs a choice.
                    try:
                        await self.app.permissions.present_schedule_card(
                            str(row.get("id")), surface="tui"
                        )
                    except Exception:  # noqa: BLE001 - listing must still show the row
                        pass
                    try:
                        self._print_approval_card(card_from_row(row))
                    except ValueError:
                        self.console.print(Text(
                            f"{row.get('id')}  {row.get('action')}  "
                            f"risk={row.get('risk')}  (card body unavailable)",
                            style="red",
                        ))
                    continue
                self.console.print(Text(
                    f"{row.get('id')}  {row.get('action')}  risk={row.get('risk')}  "
                    f"requested_by={row.get('requested_by')}",
                    style="yellow",
                ))
        elif name in {"stop", "cancel"}:
            if args:
                self.console.print(Text(f"usage: /{name}", style="yellow"))
                return True
            await self._request_stop()
        elif name in {"approve", "deny"}:
            if not args:
                self.console.print(Text(f"usage: /{name} <approval_id> [choice]", style="yellow"))
                return True
            from .. import db

            row = await db.fetchone("SELECT kind FROM approvals WHERE id=?", (args[0],))
            if row is not None and is_schedule_kind(row["kind"]):
                # Optional choice: once|session|always on create, once on edit.
                # A denial may carry a short reason; it is never a choice.
                if name == "deny":
                    choice = "deny"
                    reason = " ".join(args[1:]).strip() or None
                else:
                    choice = args[1].strip().lower() if len(args) > 1 else "once"
                    reason = None
                result = await self._resolve_schedule_card_tui(
                    args[0], approved=(name == "approve"), choice=choice,
                    deny_reason=reason,
                )
            else:
                result = await self.app.operator_resolve_approval(
                    args[0], name == "approve", actor="operator:cli"
                )
            if result.get("success", True) is False:
                self.console.print(Text(f"failed: {result.get('error')}", style="red"))
            else:
                self.console.print(Text(f"{name}d {args[0]}", style="green"))
        else:
            return False
        return True


def first_line(value: Any) -> str:
    text = str(value or "").strip().splitlines()
    return text[0][:140] if text else ""


def run_tui(app: Any, conversation_id: str | None = None) -> int:
    """Entry point used by the CLI commands."""
    return asyncio.run(ChatTUI(app, conversation_id=conversation_id).run())
