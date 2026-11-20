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
import json
import sys
import time
from typing import Any, Callable

from rich.console import Console
from rich.text import Text

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
  /approve <id>      approve a pending gate (operator principal)
  /deny <id>         deny a pending gate
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

    # ------------------------------------------------------------------ run
    async def run(self) -> int:
        """Read lines until EOF or /quit. Returns a process exit code."""
        await self.app.initialize()
        interactive = sys.stdin.isatty() and self._input is input
        if interactive:
            try:
                import readline  # noqa: F401 - history and line editing
            except Exception:  # pragma: no cover - platform dependent
                pass
            self.console.print(Text.from_markup(
                f"[bold cyan]LEO[/bold cyan] terminal chat · conversation "
                f"[dim]{self.conversation_id or '(new)'}[/dim]\n"
                "[dim]Type /help for commands. Ctrl+C interrupts a turn.[/dim]\n"
            ))
        while True:
            try:
                line = self._input("\x1b[36myou ›\x1b[0m " if interactive else "")
            except (EOFError, KeyboardInterrupt):
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
            try:
                await self.turn(text)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]interrupted[/yellow] — the turn was cancelled client-side.")
            except Exception as exc:  # noqa: BLE001 - a TUI must not crash the session
                self.console.print(f"[red]error:[/red] {type(exc).__name__}: {exc}")
        return 0  # pragma: no cover - loop returns above

    # ---------------------------------------------------------------- turns
    async def turn(self, text: str) -> dict[str, Any]:
        """Stream one turn; returns the final event payload (or an error dict)."""
        self._streamed_text = False
        self._thinking_open = False
        started = time.monotonic()
        final: dict[str, Any] = {}
        error: dict[str, Any] | None = None
        async for event in self.app.ask_stream(
            text, "cli", None, self.conversation_id
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
                self.console.print(Text(
                    f"{row.get('id')}  {row.get('action')}  risk={row.get('risk')}  "
                    f"requested_by={row.get('requested_by')}",
                    style="yellow",
                ))
        elif name in {"approve", "deny"}:
            if not args:
                self.console.print(Text(f"usage: /{name} <approval_id>", style="yellow"))
                return True
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
