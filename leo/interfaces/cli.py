"""Functional command-line interface for LEO."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import typer
from rich.console import Console
from rich.table import Table

from ..config import Settings
from ..contracts import MemoryCategory, MemoryKind, TaskStatus
from ..core import LeoApp

cli = typer.Typer(name="leo", help="LEO autonomous agent platform", no_args_is_help=False)
task_app = typer.Typer(help="Durable task management")
skill_app = typer.Typer(help="Native skills")
memory_app = typer.Typer(help="Persistent memory")
approval_app = typer.Typer(help="High-risk action approvals")
model_app = typer.Typer(help="Model providers and usage")
agent_app = typer.Typer(help="Specialized agents")
cli.add_typer(task_app, name="task")
cli.add_typer(skill_app, name="skill")
cli.add_typer(memory_app, name="memory")
cli.add_typer(approval_app, name="approval")
cli.add_typer(model_app, name="model")
cli.add_typer(agent_app, name="agent")

console = Console()
_state: dict[str, Any] = {}


def _settings() -> Settings:
    if "settings" not in _state:
        _state["settings"] = Settings.load()
    return _state["settings"]


def _app() -> LeoApp:
    if "app" not in _state:
        _state["app"] = LeoApp(_settings())
    return _state["app"]


def ready() -> LeoApp:
    """Return an initialized app (each CLI invocation is a fresh process)."""
    app = _app()
    asyncio.run(app.initialize())
    return app


def run(coro: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return asyncio.run(coro(*args, **kwargs))


def show(data: Any) -> None:
    console.print_json(json.dumps(data, default=str, ensure_ascii=False))


def fail(message: str, code: int = 1, **details: Any) -> None:
    """Emit a machine-readable error object and exit with a non-zero status."""
    payload: dict[str, Any] = {"ok": False, "error": message}
    payload.update(details)
    show(payload)
    raise typer.Exit(code)


@contextmanager
def _handle_errors(action: str) -> Iterator[None]:
    """Translate engine lookup/validation errors into JSON CLI errors."""
    try:
        yield
    except KeyError as exc:
        detail = str(exc.args[0]) if exc.args else str(exc)
        fail(f"{action}: not found ({detail})")
    except ValueError as exc:
        fail(f"{action}: {exc}")
    except PermissionError as exc:
        # e.g. a chat principal attempting to resolve its own approval gate.
        fail(f"{action}: {exc}")


def _enum(value: str, enum_type: Any, label: str) -> Any:
    """Parse an enum argument without leaking a pydantic/Enum traceback."""
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(str(getattr(item, "value", item)) for item in enum_type)
        fail(f"invalid {label}: {value!r} (expected one of: {allowed})", code=2)


def _is_loopback_host(host: str) -> bool:
    """Recognize loopback addresses/names that uvicorn may bind to."""
    value = host.strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if value in {"localhost", "ip6-localhost", "ip6-loopback", "::1", "0:0:0:0:0:0:0:1"}:
        return True
    if value.startswith("127."):
        parts = value.split(".")
        return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)
    return False


@cli.command()
def doctor() -> None:
    """Initialize and show health of the local LEO instance."""
    show(run(ready().health))


@cli.command()
def ask(message: str, channel: str = "cli", conversation: str | None = None) -> None:
    """Send a message to LEO."""
    if not message.strip():
        fail("message must not be empty", code=2)
    show(run(_app().ask, message, channel=channel, conversation_id=conversation))


def _start_chat(conversation: str | None) -> None:
    from .tui import run_tui

    raise typer.Exit(run_tui(ready(), conversation))


@cli.command("chat")
def chat(conversation: str | None = typer.Option(
    None, "--conversation", "-c", help="resume this conversation id (default: new)")
) -> None:
    """Interactive terminal chat — streaming replies, tools and approvals."""
    _start_chat(conversation)


@cli.command("tui")
def tui(conversation: str | None = typer.Option(
    None, "--conversation", "-c", help="resume this conversation id (default: new)")
) -> None:
    """Alias for `leo chat`."""
    _start_chat(conversation)


@cli.command()
def serve(
    host: str | None = None,
    port: int | None = None,
    worker: bool = typer.Option(False, "--worker", help="Answer turns claimed from a gateway"),
    no_inbound: bool = typer.Option(
        False, "--no-inbound",
        help="Serve the API/dashboard but leave polling to a gateway process",
    ),
    gateway_url: str | None = typer.Option(None, "--gateway-url", help="Gateway base URL"),
    once: bool = typer.Option(False, "--once", help="Answer one turn and exit"),
    concurrency: int = typer.Option(
        1, "--concurrency", min=1, max=32,
        help="Turns answered in parallel; 1 means a long turn blocks the queue",
    ),
    turn_max_seconds: float = typer.Option(
        600.0, "--turn-max-seconds",
        help="Stop renewing a lease after this long, so a stuck turn can be retried",
    ),
) -> None:
    """Start the local HTTP API and web dashboard, or run as a gateway worker."""
    import uvicorn

    if worker:
        _serve_worker(gateway_url or os.getenv("LEO_GATEWAY_URL"), once=once,
                      concurrency=concurrency, turn_max_seconds=turn_max_seconds)
        return

    settings = _settings()
    effective_host = (host or settings.server.host).strip()
    if not _is_loopback_host(effective_host):
        token = os.getenv("LEO_API_TOKEN")
        if not token or not token.strip():
            raise typer.BadParameter(
                "refusing a non-loopback bind without LEO_API_TOKEN; set it before exposing the API"
            )
    effective_port = port or settings.server.port
    if not 1 <= effective_port <= 65535:
        fail(f"invalid port: {effective_port} (expected 1-65535)", code=2)
    if no_inbound:
        # uvicorn imports the module-level app, so the switch travels by env.
        os.environ["LEO_DISABLE_INBOUND"] = "1"
    uvicorn.run(
        "leo.interfaces.api:api",
        host=effective_host,
        port=effective_port,
        log_level="info",
    )


def _serve_worker(url: str | None, *, once: bool = False, concurrency: int = 1,
                  turn_max_seconds: float = 600.0) -> None:
    """Run the brain against a gateway instead of owning the channels."""
    import asyncio

    if not url:
        raise typer.BadParameter(
            "worker mode needs a gateway: pass --gateway-url or set LEO_GATEWAY_URL"
        )
    from ..gateway.worker import GatewayClientError, run_worker

    try:
        answered = asyncio.run(run_worker(url, once=once, concurrency=concurrency,
                                          turn_max_seconds=turn_max_seconds))
    except GatewayClientError as exc:
        fail(str(exc), code=2)
        return
    typer.echo(f"worker stopped after answering {answered} turn(s)")


gateway_app = typer.Typer(help="Run LEO as a transport gateway")
cli.add_typer(gateway_app, name="gateway")


@gateway_app.command("run")
def gateway_run(
    host: str = typer.Option("127.0.0.1", help="bind address (loopback by default)"),
    port: int = typer.Option(8766, help="bind port"),
) -> None:
    """Run the gateway: channel adapters, durable queue, local API."""
    _require_loopback(host)
    from ..gateway.server import run_gateway

    run_gateway(host=host, port=port)


@gateway_app.command("attach")
def gateway_attach(
    url: str = typer.Option("http://127.0.0.1:8766", help="gateway base URL"),
    ttl: int = typer.Option(300, help="ticket lifetime in seconds"),
) -> None:
    """Mint a single-use attach ticket for a worker (needs the master token)."""
    import asyncio

    import httpx

    async def _mint() -> dict:
        from ..gateway.worker import GatewayClientError, resolve_token

        try:
            bearer = resolve_token()
        except GatewayClientError as exc:
            fail(str(exc), code=2)
            raise SystemExit(2)
        async with httpx.AsyncClient(base_url=url.rstrip("/"),
                                     headers={"Authorization": f"Bearer {bearer}"},
                                     timeout=httpx.Timeout(20.0, connect=5.0)) as http:
            response = await http.post("/v1/attach", params={"ttl_seconds": ttl})
        if response.status_code != 200:
            raise RuntimeError(f"attach failed: HTTP {response.status_code}")
        return response.json()

    try:
        minted = asyncio.run(_mint())
    except RuntimeError as exc:
        fail(str(exc), code=1)
        return
    typer.echo(f"ticket:  {minted['ticket']}")
    typer.echo(f"expires: {minted['expires_at']}")
    typer.echo("")
    typer.echo("Give it to one worker, then it is spent:")
    typer.echo(f"  LEO_GATEWAY_TICKET={minted['ticket']} leo serve --worker --gateway-url {url}")


@gateway_app.command("install")
def gateway_install(
    dry_run: bool = typer.Option(False, "--dry-run", help="print the unit, write nothing"),
) -> None:
    """Write the gateway's systemd unit from the shipped template."""
    import shutil
    from pathlib import Path

    template = Path(__file__).resolve().parents[2] / "deploy" / "leo-gateway@.service"
    if not template.exists():
        fail(f"unit template missing: {template}", code=2)
        return
    unit_text = template.read_text()
    target = Path("/etc/systemd/system/leo-gateway@.service")
    if dry_run:
        typer.echo(unit_text)
        return
    if not os.access("/etc/systemd/system", os.W_OK):
        fail("writing the unit needs root: re-run with sudo", code=2)
        return
    shutil.copyfile(template, target)
    typer.echo(f"wrote {target}")
    typer.echo("then: systemctl daemon-reload && systemctl enable --now "
               f"leo-gateway@{os.getenv('USER') or 'root'}.service")


@gateway_app.command("list")
def gateway_list() -> None:
    """Show this home's gateway, whether it is up, and any live sessions."""
    import asyncio
    import json as _json

    import httpx

    from .. import db as _db
    from ..gateway.store import GatewayStore

    home = os.getenv("LEO_HOME") or os.path.expanduser("~/.config/leo")
    url = os.getenv("LEO_GATEWAY_URL") or "http://127.0.0.1:8766"
    typer.echo(f"home:    {home}")
    typer.echo(f"gateway: {url}")

    async def _health() -> dict | None:
        from ..gateway.worker import GatewayClientError, resolve_token

        try:
            bearer = resolve_token()
        except GatewayClientError:
            return None
        async with httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {bearer}"},
                                     timeout=httpx.Timeout(10.0, connect=3.0)) as http:
            response = await http.get("/health")
        return response.json() if response.status_code == 200 else None

    try:
        health = asyncio.run(_health())
    except Exception:
        health = None
    if health is None:
        typer.echo("state:   unreachable (or no token)")
        return
    typer.echo(f"state:   up · pid {health.get('pid')}")
    typer.echo(f"queue:   {health.get('queue')}")
    typer.echo(f"cursor:  {health.get('cursors')}")
    typer.echo(f"sessions:{_json.dumps(health.get('sessions') or {}, sort_keys=True)}")


@gateway_app.command("status")
def gateway_status(as_json: bool = typer.Option(False, "--json")) -> None:
    """Show queue depth, channel cursors and heartbeats without a server."""
    import asyncio
    import json as _json

    from .. import db as _db
    from ..gateway.store import GatewayStore

    settings = _settings()

    async def _stats() -> dict:
        if _db.database_path_configured():
            await _db.initialize()
        else:
            await _db.initialize(settings.db_path)
        return await GatewayStore().stats()

    stats = asyncio.run(_stats())
    if as_json:
        typer.echo(_json.dumps(stats, indent=2, sort_keys=True))
        return
    queue = stats.get("queue") or {}
    typer.echo("queue: " + (", ".join(f"{k}={v}" for k, v in sorted(queue.items())) or "empty"))
    for channel, value in sorted((stats.get("cursors") or {}).items()):
        typer.echo(f"cursor {channel}: {value}")
    for channel, beat in sorted((stats.get("heartbeats") or {}).items()):
        typer.echo(f"heartbeat {channel}: pid={beat.get('pid')} at {beat.get('beat_at')}")


def _require_loopback(host: str) -> None:
    if not _is_loopback_host(host):
        fail(f"refusing to bind the gateway to {host}; it must stay on loopback", code=2)


@cli.command("init")
def init_project() -> None:
    """Initialize data directories and database."""
    app = ready()
    show({"initialized": True, "db": str(app.settings.db_path)})


@task_app.command("create")
def task_create(objective: str, agent: str | None = None) -> None:
    if not objective.strip():
        fail("objective must not be empty", code=2)
    with _handle_errors("task create"):
        task = run(ready().tasks.create, objective, owner_agent=agent)
    show(task.model_dump(mode="json"))


@task_app.command("list")
def task_list(status: str | None = None) -> None:
    status_value = _enum(status, TaskStatus, "status") if status is not None else None
    show([t.model_dump(mode="json") for t in run(lambda: ready().tasks.list(status=status_value))])


@task_app.command("show")
def task_show(task_id: str) -> None:
    with _handle_errors("task show"):
        task = run(ready().tasks.get, task_id)
    if not task:
        fail(f"task not found: {task_id}")
    show(task.model_dump(mode="json"))


@task_app.command("cancel")
def task_cancel(task_id: str) -> None:
    with _handle_errors("task cancel"):
        cancelled = run(ready().tasks.cancel, task_id)
    show({"cancelled": cancelled})


@skill_app.command("list")
def skill_list() -> None:
    skills = run(lambda: ready().skills.list())
    table = Table(title="LEO Skills")
    for column in ("name", "version", "category", "status", "health"):
        table.add_column(column)
    for skill in skills:
        table.add_row(str(skill.name), skill.version, skill.category, skill.status, skill.health)
    console.print(table)


@skill_app.command("run")
def skill_run(name: str, arguments: str = "{}") -> None:
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError) as exc:
        fail(f"invalid JSON arguments: {exc}", code=2)
    if not isinstance(parsed, dict):
        fail("skill arguments must be a JSON object", code=2)
    with _handle_errors("skill run"):
        result = run(ready().skills.run, name, parsed)
    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    show({"result": payload})
    if isinstance(payload, dict) and payload.get("success") is False:
        raise typer.Exit(1)


@memory_app.command("add")
def memory_add(content: str, category: str = "semantic", kind: str = "fact") -> None:
    memory_category = _enum(category, MemoryCategory, "category")
    memory_kind = _enum(kind, MemoryKind, "kind")
    with _handle_errors("memory add"):
        record = run(ready().remember, content, category=memory_category, kind=memory_kind)
    show(record.model_dump(mode="json"))


@memory_app.command("search")
def memory_search(query: str, limit: int = 10) -> None:
    results = run(ready().search.search, query, limit=limit)
    show([m.model_dump(mode="json") for m in results])


@approval_app.command("list")
def approval_list() -> None:
    show(run(lambda: ready().permissions.pending_approvals()))


@approval_app.command("approve")
def approval_approve(approval_id: str, resolver: str = "user") -> None:
    with _handle_errors("approval approve"):
        decision = run(ready().permissions.resolve_approval, approval_id, True, resolver)
    show(decision.model_dump(mode="json"))


@approval_app.command("deny")
def approval_deny(approval_id: str, resolver: str = "user") -> None:
    with _handle_errors("approval deny"):
        decision = run(ready().permissions.resolve_approval, approval_id, False, resolver)
    show(decision.model_dump(mode="json"))


@approval_app.command("cleanup")
def approval_cleanup(
    source: str = typer.Option(None, "--source", help="only approvals requested by this actor (e.g. learning:engine)"),
    older_than: float = typer.Option(24.0, "--older-than", help="hours a pending approval must have been idle"),
    resolver: str = typer.Option("operator", "--resolver", help="who is expiring them (never a chat: actor)"),
    apply: bool = typer.Option(False, "--apply", help="actually expire them; default is a dry run"),
    limit: int = typer.Option(500, "--limit", help="maximum approvals to consider"),
) -> None:
    """Report stale pending approvals; --apply expires them (never deletes)."""
    with _handle_errors("approval cleanup"):
        report = run(
            ready().permissions.cleanup_approvals,
            source=source, older_than_hours=older_than,
            resolver=resolver, apply=apply, limit=limit,
        )
    show(report)


@model_app.command("presets")
def model_presets() -> None:
    """List built-in provider presets usable with `leo model add`."""
    from ..models.presets import PRESETS

    show({
        name: {
            "kind": preset["kind"],
            "base_url": preset["base_url"],
            "api_key_env": preset["api_key_env"],
            "capabilities": preset["capabilities"],
        }
        for name, preset in PRESETS.items()
    })


@model_app.command("add")
def model_add(
    name: str,
    preset: str = typer.Option("openai-compatible", "--preset", help="provider preset, see `leo model presets`"),
    model: str = typer.Option(..., "--model", help="exact model id the provider serves"),
    api_key_env: str = typer.Option(None, "--api-key-env", help="env-var NAME holding the key (never the key)"),
    base_url: str = typer.Option(None, "--base-url", help="override the preset endpoint"),
    capabilities: str = typer.Option(None, "--capabilities", help="comma-separated; e.g. text,tools,vision"),
    make_default: bool = typer.Option(False, "--default", help="also set models.default"),
    config_path: str = typer.Option("leo.yaml", "--config", help="config file to update"),
) -> None:
    """Add a model provider to the config (validated, with a backup)."""
    from ..models.presets import add_provider, build_provider_entry

    caps = [item.strip() for item in capabilities.split(",") if item.strip()] if capabilities else None
    with _handle_errors("model add"):
        entry = build_provider_entry(
            preset, model, api_key_env=api_key_env, base_url=base_url, capabilities=caps
        )
        result = add_provider(
            name, entry, config_path=config_path, make_default=make_default
        )
    show(result)


@model_app.command("list")
def model_list() -> None:
    show({name: provider.model_dump(mode="json") for name, provider in _settings().models.providers.items()})


@model_app.command("status")
def model_status() -> None:
    """Report live-vs-simulated model state explicitly."""
    show(_settings().model_status())


@model_app.command("usage")
def model_usage() -> None:
    show(run(lambda: ready().router.usage_report()))


@agent_app.command("list")
def agent_list() -> None:
    show([a.model_dump(mode="json") for a in run(lambda: ready().agent_registry.list())])


@cli.command("events")
def events(topic: str | None = None, limit: int = 20) -> None:
    show([e.model_dump(mode="json") for e in run(lambda: ready().events.history(topic, limit))])


@cli.command("metrics")
def metrics() -> None:
    show(run(ready().metrics))


@cli.command("integrations")
def integrations(sync: bool = typer.Option(False, "--sync", help="upsert channel records into the local database")) -> None:
    """List every channel adapter with its honest status, or sync them to the DB."""
    if sync:
        ready()  # initializes the database the records are synced into
        from ..integrations.channels import sync_integration_records

        show(run(sync_integration_records))
        return
    from ..integrations.channels import integration_records

    show([record.model_dump(mode="json") for record in integration_records()])


def _should_open_tui() -> bool:
    """Bare ``leo`` opens the chat only for an interactive terminal."""
    import sys

    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


@cli.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Bare invocation: open the terminal chat, or print help when piped."""
    if ctx.invoked_subcommand is not None:
        return
    if _should_open_tui():
        _start_chat(None)
    typer.echo(ctx.get_help())
    raise typer.Exit(0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
