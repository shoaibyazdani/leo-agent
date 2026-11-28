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
grants_app = typer.Typer(help="Session and workspace approval grants")
approval_app.add_typer(grants_app, name="grants")
model_app = typer.Typer(help="Model providers and usage")
agent_app = typer.Typer(help="Specialized agents")
whatsapp_app = typer.Typer(help="WhatsApp device linking", no_args_is_help=True)
cli.add_typer(task_app, name="task")
cli.add_typer(skill_app, name="skill")
cli.add_typer(memory_app, name="memory")
cli.add_typer(approval_app, name="approval")
cli.add_typer(model_app, name="model")
cli.add_typer(agent_app, name="agent")
cli.add_typer(whatsapp_app, name="whatsapp")

console = Console()
_state: dict[str, Any] = {}


def _model_presets() -> dict[str, Any]:
    from ..models.presets import PRESETS

    return dict(PRESETS)


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
def doctor(
    as_json: bool = typer.Option(
        False, "--json", help="print the unchanged health JSON snapshot"
    ),
    fix: bool = typer.Option(False, "--fix", help="apply safe repairs and recheck"),
    config: str | None = typer.Option(None, "--config", help="config file to diagnose"),
) -> None:
    """Run pass/fail diagnostics; exits non-zero if any check fails."""
    if as_json:
        show(run(ready().health))
        return
    from ..doctor import run_diagnostics

    settings = Settings.load(config) if config else Settings.load()
    results = run_diagnostics(settings, config_path=config, fix=fix)
    for item in results:
        status = "PASS" if item.ok else "FAIL"
        marker = " (fixed)" if item.fixed else ""
        typer.echo(f"{status} {item.name}: {item.detail}{marker}")
    passed = sum(1 for item in results if item.ok)
    typer.echo(f"{passed}/{len(results)} checks passed")
    if passed != len(results):
        raise typer.Exit(1)


@cli.command()
def onboard(
    mode: str = typer.Option("quick", "--mode", help="quick or advanced setup"),
    quick: bool = typer.Option(False, "--quick", help="alias for --mode quick"),
    advanced: bool = typer.Option(False, "--advanced", help="alias for --mode advanced"),
    provider: str | None = typer.Option(None, "--provider", help="provider preset"),
    key: str | None = typer.Option(
        None, "--key", help="API key (hidden prompt is safer; only stored in the env file)"
    ),
    key_env: str | None = typer.Option(None, "--key-env", help="env-var NAME for the key"),
    base_url: str | None = typer.Option(None, "--base-url", help="override the preset endpoint"),
    model: str | None = typer.Option(None, "--model", help="exact default model id"),
    fallback: list[str] = typer.Option([], "--fallback", help="fallback model id (repeatable)"),
    vision_model: str | None = typer.Option(None, "--vision-model", help="vision model id"),
    channel: list[str] = typer.Option(
        [], "--channel", help="telegram or whatsapp (repeatable, or 'both')"
    ),
    telegram_owner: list[str] = typer.Option(
        [], "--telegram-owner", help="allowed Telegram sender id (repeatable)"
    ),
    whatsapp_owner: list[str] = typer.Option(
        [], "--whatsapp-owner", help="allowed WhatsApp sender (repeatable)"
    ),
    no_service: bool = typer.Option(False, "--no-service", help="skip service installation"),
    skip_probe: bool = typer.Option(
        False, "--skip-probe", help="offline/lab use only: skip the live key check"
    ),
    config: str = typer.Option("leo.yaml", "--config", help="config file to update"),
    env_file: str | None = typer.Option(None, "--env-file", help="private env file for keys"),
    yes: bool = typer.Option(False, "--yes", help="accept QuickStart defaults without prompts"),
    skip_doctor: bool = typer.Option(
        False, "--skip-doctor", help="finish before diagnostics (automation only)"
    ),
) -> None:
    """Set up a provider, models, channels and a boot service, then run doctor."""
    from ..models.catalog import fetch_catalog
    from ..onboarding_wizard import (
        WizardAnswers,
        apply_env_file,
        run_wizard as run_onboard_wizard,
    )

    env_path = env_file
    env_values = apply_env_file(env_path)

    def choose(label: str, options: tuple[str, ...], default: str) -> str:
        rendered = "/".join(options)
        while True:
            answer = typer.prompt(f"{label} [{rendered}]", default=default).strip().lower()
            if answer in options:
                return answer
            typer.echo(f"choose one of: {rendered}")

    requested_mode = "advanced" if advanced else ("quick" if quick else mode)
    if advanced and quick:
        fail("--quick and --advanced are mutually exclusive", code=2)
        return
    chosen_mode = requested_mode if yes else choose("Setup mode", ("quick", "advanced"), requested_mode)
    presets_ = _model_presets()
    chosen_provider = provider or (
        "openrouter" if yes else typer.prompt(
            f"Provider preset ({', '.join(presets_)})", default="openrouter"
        )
    )
    chosen_provider = str(chosen_provider).strip().lower()
    template = presets_.get(chosen_provider)
    if template is None:
        fail(f"unknown provider preset: {chosen_provider}", code=2)
        return
    env_name = key_env or template.get("api_key_env")
    secret = key or (
        os.getenv(env_name) if env_name else None
    ) or (env_values.get(env_name) if env_name else None)
    if not yes and not secret and template.get("api_key_env"):
        secret = typer.prompt(
            f"API key for {env_name} (hidden; stored only in the env file)",
            hide_input=True,
        )
    chosen_model = model
    chosen_fallbacks = list(fallback)
    chosen_vision = vision_model
    if not yes and chosen_mode == "advanced" and not chosen_model:
        listed = fetch_catalog(
            str(template.get("base_url") or ""), api_key=secret,
            kind=str(template.get("kind") or "openai-compatible"),
        )
        if listed.ok and listed.models:
            for index, item in enumerate(listed.models[:20], start=1):
                typer.echo(f"  {index:2}. {item}")
            chosen_model = typer.prompt("Default model id")
            more = typer.prompt("Fallback model ids (comma-separated; blank for none)", default="")
            chosen_fallbacks = [x.strip() for x in more.split(",") if x.strip()]
            chosen_vision = typer.prompt(
                "Vision model id (blank to use the default)", default=""
            ) or None

    selected_channels = list(channel)
    if not yes and not selected_channels:
        raw = typer.prompt(
            "Channels [none/telegram/whatsapp/both]", default="none"
        ).strip().lower()
        selected_channels = [] if raw == "none" else [raw]
    owners = {
        "telegram": list(telegram_owner),
        "whatsapp": list(whatsapp_owner),
    }
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN") or env_values.get("TELEGRAM_BOT_TOKEN")
    whatsapp_token = os.getenv("WHATSAPP_CLOUD_TOKEN") or env_values.get("WHATSAPP_CLOUD_TOKEN")
    phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID") or env_values.get("WHATSAPP_PHONE_NUMBER_ID")
    if not yes:
        if "telegram" in selected_channels or "both" in selected_channels:
            telegram_token = telegram_token or typer.prompt(
                "Telegram bot token (hidden)", hide_input=True
            )
            if not owners["telegram"]:
                raw = typer.prompt("Allowed Telegram sender ids (comma-separated)")
                owners["telegram"] = [x.strip() for x in raw.split(",") if x.strip()]
        if "whatsapp" in selected_channels or "both" in selected_channels:
            whatsapp_token = whatsapp_token or typer.prompt(
                "WhatsApp cloud token (hidden)", hide_input=True
            )
            phone_id = phone_id or typer.prompt("WhatsApp phone number id")
            if not owners["whatsapp"]:
                raw = typer.prompt("Allowed WhatsApp senders (comma-separated)")
                owners["whatsapp"] = [x.strip() for x in raw.split(",") if x.strip()]

    answers = WizardAnswers(
        mode=chosen_mode,
        preset=chosen_provider,
        api_key=secret,
        api_key_env=key_env,
        base_url=base_url,
        model=chosen_model,
        fallback_models=chosen_fallbacks,
        vision_model=chosen_vision,
        channels=selected_channels,
        channel_owners=owners,
        telegram_token=telegram_token,
        whatsapp_token=whatsapp_token,
        whatsapp_phone_id=phone_id,
        install_service=not no_service,
        skip_probe=skip_probe,
    )
    result = run_onboard_wizard(
        answers,
        config_path=config,
        env_file=env_path,
        catalog=fetch_catalog,
    )
    if not result.ok:
        fail(f"onboarding failed: {result.error}")
        return
    show(result.as_dict())
    if not skip_doctor:
        doctor(config=config)


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
        1800.0, "--turn-max-seconds",
        help="Stop the turn after this long (a tool-heavy turn legitimately runs long)",
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
                  turn_max_seconds: float = 1800.0) -> None:
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


@grants_app.command("list")
def grants_list(
    include_expired: bool = typer.Option(False, "--all", help="include expired grants"),
) -> None:
    """List live session and workspace grants."""
    show(run(lambda: ready().permissions.list_grants(include_expired=include_expired)))


@grants_app.command("revoke")
def grants_revoke(grant_id: str) -> None:
    """Revoke one grant immediately."""
    with _handle_errors("grant revoke"):
        revoked = run(ready().permissions.revoke_grant, grant_id)
    show({"grant_id": grant_id, "revoked": revoked})


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
def model_list(
    all_models: bool = typer.Option(
        False, "--all", help="fetch each provider's live model catalog"
    ),
    provider: str | None = typer.Option(None, "--provider", help="only this configured slot"),
    refresh: bool = typer.Option(
        False, "--refresh", help="always fetch live catalogs (the default for --all)"
    ),
) -> None:
    """List configured slots, or --all for the providers' live catalogs."""
    slots = _settings().models.providers
    if provider and provider not in slots:
        fail(f"unknown provider slot: {provider}")
        return
    if not all_models:
        show({
            name: item.model_dump(mode="json")
            for name, item in slots.items() if provider in (None, name)
        })
        return

    from ..models.catalog import fetch_catalog

    details: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    every_model: list[str] = []
    for name, item in slots.items():
        if provider and name != provider:
            continue
        if item.kind == "mock" or not item.base_url:
            details[name] = {
                "source": "configured",
                "models": [item.model] if item.model else [],
                "detail": "no live catalog endpoint",
            }
            continue
        key = os.getenv(item.api_key_env) if item.api_key_env else None
        catalog = fetch_catalog(
            item.base_url, api_key=key, kind=item.kind,
            timeout=float(item.timeout_seconds or 8.0),
            extra_headers=dict(item.extra_headers or {}),
        )
        if catalog.ok and catalog.models:
            models = list(catalog.models)
            source = "live"
        else:
            models = []
            source = "unavailable"
            errors.append({"provider": name, "detail": catalog.detail})
        details[name] = {"source": source, "models": models, "detail": catalog.detail}
        for value in models:
            if value not in every_model:
                every_model.append(value)
    show({
        "all": True,
        "refreshed": True,  # --all always fetches; --refresh is a compatibility alias
        "providers": details,
        "models": every_model,
        "errors": errors,
    })


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


@whatsapp_app.command("link")
def whatsapp_link(
    owner: str = typer.Option("owner", "--owner", help="local owner id recorded for this session"),
    timeout: float = typer.Option(600.0, "--timeout", help="seconds to wait for the QR scan"),
    stay: bool = typer.Option(False, "--stay", help="keep the linked session open until Ctrl+C"),
    no_browser: bool = typer.Option(False, "--no-browser", help="print the local page URL instead of opening it"),
) -> None:
    """Link this computer as a WhatsApp device by scanning a QR code.

    Uses the unofficial WhatsApp Web protocol (Baileys), not the Meta Cloud
    API. This links your personal account and carries WhatsApp ToS and
    account-ban risk; continue only if you accept that for this account.
    Credentials stay in .leo/whatsapp/ and are never printed.
    """
    from ..integrations.whatsapp_link import LinkUnavailable, run_device_link, state_message

    def notify(kind: str, data: Any) -> None:
        if kind == "page":
            console.print(f"Link page: [cyan]{data}[/cyan]")
            console.print("Scan the code shown there with WhatsApp → Linked devices → Link a device.")
        elif kind == "state" and isinstance(data, dict):
            state = str(data.get("state") or "")
            if state == "connected":
                user = data.get("user") or {}
                who = user.get("name") or user.get("id") or "this account"
                console.print(f"[green]Connected[/green] as {who}. Credentials saved.")
            elif state in {"logged_out", "error", "cancelled"}:
                detail = data.get("error")
                console.print(f"[yellow]{state_message(state)}[/yellow]" + (f" ({detail})" if detail else ""))
            else:
                console.print(state_message(state))
        elif kind == "error":
            console.print(f"[red]Bridge failed:[/red] {type(data).__name__}: {data}")

    try:
        result = run(
            run_device_link,
            owner,
            timeout=timeout,
            stay=stay,
            open_page=not no_browser,
            notify=notify,
        )
    except LinkUnavailable as exc:
        fail(str(exc), code=2)
        return
    except ValueError as exc:
        fail(str(exc), code=2)
        return
    show(result.as_dict())
    if result.as_dict()["state"] != "connected":
        raise typer.Exit(1)


@whatsapp_app.command("status")
def whatsapp_status(
    owner: str = typer.Option("owner", "--owner", help="local owner id recorded for this session"),
) -> None:
    """Show bridge readiness and whether saved device credentials exist."""
    from ..integrations.whatsapp_link import bridge_status

    show(bridge_status(owner))


@whatsapp_app.command("logout")
def whatsapp_logout(
    owner: str = typer.Option("owner", "--owner", help="local owner id recorded for this session"),
    yes: bool = typer.Option(False, "--yes", help="do not prompt for confirmation"),
) -> None:
    """Unlink this computer and delete the saved WhatsApp credentials."""
    from ..integrations.whatsapp_link import LinkUnavailable, ensure_ready
    from ..integrations.whatsapp_bridge import BridgeConfig, BridgeError, WhatsAppBridgeManager

    if not yes:
        typer.confirm(
            "Unlink this computer from WhatsApp and delete saved credentials?",
            abort=True,
        )

    async def forget() -> dict[str, Any]:
        ensure_ready()
        manager = WhatsAppBridgeManager(BridgeConfig(owner_id=owner, actor="default", enabled=True))
        auth = manager.session_auth
        async with manager:
            await manager.start(owner_id=owner, auth=auth)
            return await manager.repair(owner_id=owner, forget=True, auth=auth)

    try:
        result = run(forget)
    except LinkUnavailable as exc:
        fail(str(exc), code=2)
        return
    except (ValueError, PermissionError) as exc:
        fail(f"whatsapp logout: {exc}", code=2)
        return
    except BridgeError as exc:
        fail(f"whatsapp logout: {exc}")
        return
    show({"unlinked": True, "state": result.get("state"), "linked": bool(result.get("linked"))})


@whatsapp_app.command("serve")
def whatsapp_serve(
    owner: str = typer.Option("owner", "--owner", help="local owner id recorded for this session"),
    allow: list[str] = typer.Option(None, "--allow", help="phone number allowed to chat (repeatable, with country code)"),
    allow_all: bool = typer.Option(False, "--allow-all", help="answer ANY sender — explicit, risky opt-in"),
    groups: bool = typer.Option(False, "--groups", help="also answer group chats (sender must still be allowed)"),
    debug: bool = typer.Option(False, "--debug", help="print sanitized bridge diagnostics (never message text)"),
) -> None:
    """Answer WhatsApp messages with LEO until Ctrl+C.

    Requires a linked session (`leo whatsapp link`). Inbound text is treated as
    untrusted content; high-risk tool calls still require approval. Without
    --allow or --allow-all this refuses to answer anyone.
    """
    from ..integrations.whatsapp_msg import run_message_agent
    from ..integrations.whatsapp_bridge import BridgeError
    from ..integrations.whatsapp_link import LinkUnavailable

    def notify(kind: str, data: Any) -> None:
        if kind == "warning" and isinstance(data, dict):
            names = ", ".join(str(name) for name in (data.get("missing") or []))
            console.print(
                f"[yellow]Model environment incomplete:[/yellow] {names} is unset. "
                "Replies will fail until the same provider variables as `leo chat` are exported."
            )
        elif kind == "status" and isinstance(data, dict):
            user = data.get("user") or {}
            who = user.get("name") or user.get("id") or "this account"
            console.print(f"[green]Connected[/green] as {who}. Listening for messages…")
            if allow_all:
                console.print("[yellow]--allow-all: ANY WhatsApp sender can talk to LEO.[/yellow]")
            else:
                console.print("Allowed numbers: " + ", ".join(allow or []))
        elif kind == "message" and isinstance(data, dict):
            console.print(f"Message from {data.get('sender')} ({data.get('chars')} chars) — answering…")
        elif kind == "replied" and isinstance(data, dict):
            console.print(f"[green]Replied[/green] to {data.get('sender')} ({data.get('chars')} chars).")
        elif kind == "ignored" and isinstance(data, dict):
            console.print(f"Ignored message from {data.get('sender')} (not allowed).")
        elif kind == "failed" and isinstance(data, dict):
            console.print(f"[yellow]Turn failed[/yellow] for {data.get('sender')}: {data.get('error')}")
        elif kind == "diagnostic":
            console.print(f"[dim][diag] {data}[/dim]")

    try:
        result = run(
            run_message_agent,
            owner,
            allow=list(allow or []),
            allow_all=allow_all,
            include_groups=groups,
            debug=debug,
            notify=notify,
        )
    except LinkUnavailable as exc:
        fail(str(exc), code=2)
        return
    except BridgeError as exc:
        fail(f"whatsapp serve: {exc}", code=2)
        return
    except KeyboardInterrupt:
        console.print("\nStopped.")
        return
    show(result.as_dict())
    if result.as_dict()["state"] != "connected":
        raise typer.Exit(1)


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
