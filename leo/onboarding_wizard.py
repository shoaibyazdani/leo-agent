"""Shared building blocks for ``leo onboard``.

The wizard writes non-secret configuration to YAML and credentials only to a
private environment file. Every step is reachable from the CLI; the flow itself
is a library function so tests can drive it without a terminal, a network or a
real service manager.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import Settings
from .models.catalog import ProbeResult, probe_provider
from .models.presets import PRESETS, build_provider_entry

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_CHANNELS = ("telegram", "whatsapp")
_CHOICES = ("none", "telegram", "whatsapp", "both")


def default_env_file() -> Path:
    """Private env file shared by the launcher, services and the wizard."""
    return Path(os.getenv("LEO_ENV_FILE", str(Path.home() / ".config" / "leo" / "env"))).expanduser()


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Parse ``KEY=value`` / ``export KEY=value`` lines without executing them."""
    target = Path(path) if path else default_env_file()
    values: dict[str, str] = {}
    if not target.is_file():
        return values
    for line in target.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry.startswith("export "):
            entry = entry[len("export "):].lstrip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.match(name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[name] = value
    return values


def apply_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Load an env file into this process; never override an explicit export."""
    values = load_env_file(path)
    for name, value in values.items():
        os.environ.setdefault(name, value)
    return values


def write_env_file(path: str | Path, updates: dict[str, str]) -> Path:
    """Merge credential values into a 0600 env file (values never go to YAML)."""
    target = Path(path).expanduser()
    for name, value in updates.items():
        if not _ENV_NAME.match(name):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if not isinstance(value, str):
            raise ValueError(f"value for {name} must be text")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if target.is_file():
        lines = target.read_text(encoding="utf-8").splitlines()
    replaced: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        entry = line.strip()
        candidate = entry[len("export "):].lstrip() if entry.startswith("export ") else entry
        name = candidate.split("=", 1)[0].strip() if "=" in candidate else ""
        if name in updates and name not in replaced:
            rendered.append(f"{name}={shlex.quote(updates[name])}")
            replaced.add(name)
        else:
            rendered.append(line)
    for name, value in updates.items():
        if name not in replaced:
            rendered.append(f"{name}={shlex.quote(value)}")
    content = "\n".join(rendered).rstrip("\n") + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.chmod(tmp, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    os.chmod(target, 0o600)
    return target


@dataclass
class WizardAnswers:
    """Answers gathered interactively or from non-interactive CLI options."""

    mode: str = "quick"
    preset: str = "openrouter"
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    model: str | None = None
    fallback_models: list[str] = field(default_factory=list)
    vision_model: str | None = None
    channels: list[str] = field(default_factory=list)
    channel_owners: dict[str, list[str]] = field(default_factory=dict)
    telegram_token: str | None = None
    whatsapp_token: str | None = None
    whatsapp_phone_id: str | None = None
    install_service: bool = True
    skip_probe: bool = False


@dataclass
class WizardResult:
    ok: bool
    steps: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    config_path: str = ""
    env_file: str = ""
    models: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "steps": self.steps,
            "error": self.error,
            "config_path": self.config_path,
            "env_file": self.env_file,
            "models": list(self.models),
        }


def _step(name: str, detail: str) -> dict[str, str]:
    return {"step": name, "detail": detail}


def _resolve_preset(preset: str) -> tuple[str, dict[str, Any]]:
    name = (preset or "").strip().lower()
    if name not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose one of: {', '.join(sorted(PRESETS))}")
    return name, PRESETS[name]


def _channel_list(channels: list[str]) -> list[str]:
    selected: list[str] = []
    for item in channels:
        name = (item or "").strip().lower()
        if name in ("both", "all"):
            selected.extend(_CHANNELS)
        elif name == "none" or not name:
            continue
        elif name in _CHANNELS:
            selected.append(name)
        else:
            raise ValueError(f"unknown channel {item!r}; choose from: {', '.join(_CHOICES)}")
    return list(dict.fromkeys(selected))


def _owners(values: list[str]) -> list[str]:
    return [item.strip() for item in values if str(item).strip()]


def _config_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_config(path: Path, data: dict[str, Any]) -> Path:
    """Validate first, then replace the config atomically with a backup."""
    Settings.model_validate(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        backup = path.with_name(f"{path.name}.bak-onboard")
        shutil.copy2(path, backup)
    rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def run_wizard(
    answers: WizardAnswers,
    *,
    config_path: str | Path,
    env_file: str | Path | None = None,
    repo_dir: str | Path | None = None,
    probe: Callable[..., ProbeResult] | None = None,
    catalog: Callable[..., Any] | None = None,
    service_installer: Callable[..., dict[str, Any]] | None = None,
) -> WizardResult:
    """Validate the key first, then update config, env file and service.

    Re-running updates the same slot names, so it never duplicates providers or
    drops unrelated configuration.
    """
    result = WizardResult(ok=False, config_path=str(config_path), env_file="")
    try:
        mode = (answers.mode or "quick").strip().lower()
        if mode not in ("quick", "advanced"):
            raise ValueError("mode must be quick or advanced")
        name, template = _resolve_preset(answers.preset)
        result.steps.append(_step("mode", mode))
        result.steps.append(_step("provider", f"{name} ({template['kind']})"))

        env_target = Path(env_file) if env_file else default_env_file()
        result.env_file = str(env_target)
        env_name = answers.api_key_env or template.get("api_key_env")
        if answers.api_key is not None and not str(answers.api_key).strip():
            raise ValueError("API key must not be empty")
        if (
            answers.api_key is None and env_name
            and not (os.getenv(env_name) or load_env_file(env_target).get(env_name))
        ):
            raise ValueError(f"missing credential: set {env_name} or paste a key")

        # Validate before any file is touched. The probe is a single cheap GET
        # (catalog read) and no response body or key is retained in the result.
        check = probe or probe_provider
        key = answers.api_key if answers.api_key is not None else (
            os.getenv(env_name) if env_name else None
        )
        if key is None and env_name:
            key = load_env_file(env_target).get(env_name)
        verdict = check(
            answers.base_url or template.get("base_url") or "",
            api_key=key,
            kind=template["kind"],
            skip=answers.skip_probe,
        )
        if not verdict.ok:
            raise ValueError(f"key validation failed: {verdict.detail}")
        result.steps.append(_step("key", f"validated via {env_name or 'local provider'}; value hidden"))

        # Model choice uses the live catalog when available; a second call is
        # avoided when the probe already returned the list.
        models = tuple(verdict.models or ())
        if catalog and not models:
            listed = catalog(
                answers.base_url or template.get("base_url") or "",
                api_key=key, kind=template["kind"],
            )
            models = tuple(getattr(listed, "models", listed) or ())
        if not models and not answers.model:
            raise ValueError(
                "no model id available: pass --model or choose a provider with a live catalog"
            )
        model = (answers.model or (models[0] if models else "")).strip()
        if not model:
            raise ValueError("model id must not be empty")
        result.models = models
        result.steps.append(_step("model", f"default={model}; fallback(s)={len(answers.fallback_models)}"))

        fallbacks = [item.strip() for item in answers.fallback_models if str(item).strip()]
        if mode == "quick" and not fallbacks and len(models) > 1:
            fallbacks = [models[1]]
        vision_model = (answers.vision_model or "").strip()
        if mode == "quick" and not vision_model and "vision" in template["capabilities"]:
            # QuickStart uses the selected model's declared vision capability;
            # Advanced lets the operator name a different vision slot.
            vision_model = model
        if vision_model:
            result.steps.append(_step("vision", f"model={vision_model}"))

        channels = _channel_list(answers.channels)
        result.steps.append(_step("channel", ", ".join(channels) if channels else "none"))

        updates: dict[str, str] = {}
        # Persist the key even when it came from an exported shell variable or
        # a previous env file: services and the launcher read the env file, not
        # the shell that happened to run the wizard.
        if env_name and key:
            updates[env_name] = key
        telegram_owners = _owners(answers.channel_owners.get("telegram") or [])
        whatsapp_owners = _owners(answers.channel_owners.get("whatsapp") or [])
        if "telegram" in channels:
            if not answers.telegram_token:
                raise ValueError("telegram requires a bot token")
            updates["TELEGRAM_BOT_TOKEN"] = answers.telegram_token
            if telegram_owners:
                updates["TELEGRAM_ALLOWED_USER_IDS"] = ",".join(telegram_owners)
        if "whatsapp" in channels:
            if not answers.whatsapp_token or not answers.whatsapp_phone_id:
                raise ValueError("whatsapp requires a cloud token and phone number id")
            updates["WHATSAPP_CLOUD_TOKEN"] = answers.whatsapp_token
            updates["WHATSAPP_PHONE_NUMBER_ID"] = answers.whatsapp_phone_id

        config = Path(config_path).expanduser().resolve()
        data = _config_data(config)
        models_cfg = data.setdefault("models", {})
        if not isinstance(models_cfg, dict):
            raise ValueError("models config must be a mapping")
        providers = models_cfg.setdefault("providers", {})
        if not isinstance(providers, dict):
            raise ValueError("models.providers must be a mapping")

        capabilities = list(template["capabilities"])
        entry = build_provider_entry(
            name, model, api_key_env=env_name,
            base_url=answers.base_url, capabilities=capabilities,
        )
        providers[name] = entry
        fallback_names: list[str] = []
        for index, fallback in enumerate(fallbacks, start=1):
            slot = f"{name}-fallback-{index}"
            providers[slot] = build_provider_entry(
                name, fallback, api_key_env=env_name,
                base_url=answers.base_url, capabilities=capabilities,
            )
            fallback_names.append(slot)
        if vision_model:
            vision_slot = f"{name}-vision"
            vision_caps = list(capabilities)
            if "vision" not in vision_caps:
                vision_caps.append("vision")
            providers[vision_slot] = build_provider_entry(
                name, vision_model, api_key_env=env_name,
                base_url=answers.base_url, capabilities=vision_caps,
            )
            vision_cfg = models_cfg.setdefault("vision", {})
            if not isinstance(vision_cfg, dict):
                raise ValueError("models.vision must be a mapping")
            vision_cfg["enabled"] = True
            vision_cfg["provider"] = vision_slot

        models_cfg["default"] = name
        # The router exposes one fallback slot; additional fallbacks stay as
        # configured slots for later use rather than silently changing routing.
        models_cfg["fallback"] = fallback_names[0] if fallback_names else None

        channels_cfg = data.setdefault("channels", {})
        if not isinstance(channels_cfg, dict):
            raise ValueError("channels config must be a mapping")
        policies = channels_cfg.setdefault("policies", {})
        if not isinstance(policies, dict):
            raise ValueError("channels.policies must be a mapping")
        if "telegram" in channels:
            policies["telegram"] = {"mode": "allowlist", "allow": telegram_owners}
        if "whatsapp" in channels:
            policies["whatsapp"] = {"mode": "allowlist", "allow": whatsapp_owners}

        secrets = [value for value in (
            answers.api_key, answers.telegram_token, answers.whatsapp_token,
            answers.whatsapp_phone_id,
        ) if value]
        Settings.model_validate(data)

        # Confirm the no-secrets invariant on the exact bytes we are about to
        # persist, before either file is written.
        rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        for secret in secrets:
            if secret in rendered:
                raise ValueError("refusing to write a credential value into the YAML config")

        _write_config(config, data)
        if updates:
            write_env_file(env_target, updates)
        result.steps.append(_step("config", f"updated {config}; keys only in {env_target}"))

        if answers.install_service:
            installer = service_installer or install_service
            service = installer(
                repo_dir=Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[1],
                env_file=env_target, config_path=config,
            )
            if not service.get("ok"):
                raise ValueError(f"service install failed: {service.get('detail', 'unknown error')}")
            result.steps.append(_step("service", service.get("detail", "installed")))
        else:
            result.steps.append(_step("service", "skipped"))
        result.ok = True
        return result
    except (ValueError, OSError, yaml.YAMLError) as exc:
        result.error = str(exc)
        return result


_UNIT = """[Unit]
Description=LEO local service
After=network-online.target

[Service]
Type=simple
WorkingDirectory={repo}
EnvironmentFile=-{env_file}
Environment=LEO_CONFIG={config}
ExecStart={python} -m leo.interfaces.cli serve --host 127.0.0.1 --port 8765
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def _python_executable(repo_dir: Path) -> str:
    local = repo_dir / ".venv" / "bin" / "python"
    if local.is_file():
        return str(local)
    if sys.platform == "win32":
        local = repo_dir / ".venv" / "Scripts" / "python.exe"
        if local.is_file():
            return str(local)
    return sys.executable


def install_service(
    *,
    repo_dir: Path,
    env_file: Path,
    config_path: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Install an auto-start service without sudo (user systemd / LaunchAgent)."""
    runner = run or subprocess.run
    repo = Path(repo_dir).resolve()
    env = Path(env_file).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    python = _python_executable(repo)

    def execute(*args: str, optional: bool = False) -> str:
        completed = runner(list(args), capture_output=True, text=True, check=False)
        if completed.returncode and not optional:
            raise RuntimeError(
                f"{' '.join(args[:2])} failed (exit {completed.returncode})"
            )
        return str(getattr(completed, "stderr", "") or "").strip()

    try:
        if sys.platform.startswith("linux"):
            unit_dir = Path.home() / ".config" / "systemd" / "user"
            unit_dir.mkdir(parents=True, exist_ok=True)
            unit = unit_dir / "leo.service"
            unit.write_text(
                _UNIT.format(repo=repo, env_file=env, config=config, python=python),
                encoding="utf-8",
            )
            execute("systemctl", "--user", "daemon-reload", optional=True)
            execute("systemctl", "--user", "enable", "--now", "leo.service")
            detail = "systemd user service enabled (leo.service)"
        elif sys.platform == "darwin":
            home = Path.home()
            wrapper = home / ".config" / "leo" / "serve.sh"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text(
                "#!/bin/sh\n"
                f'. "{env}"\n'
                f'cd "{repo}" || exit 1\n'
                f'exec "{python}" -m leo.interfaces.cli serve '
                '--host 127.0.0.1 --port 8765\n',
                encoding="utf-8",
            )
            wrapper.chmod(0o700)
            logs = home / ".config" / "leo" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            plist = home / "Library" / "LaunchAgents" / "com.leo.agent.plist"
            plist.parent.mkdir(parents=True, exist_ok=True)
            plist.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                '<plist version="1.0"><dict>'
                '<key>Label</key><string>com.leo.agent</string>'
                f'<key>ProgramArguments</key><array><string>/bin/sh</string>'
                f'<string>{wrapper}</string></array>'
                '<key>RunAtLoad</key><true/>'
                '<key>KeepAlive</key><true/>'
                f'<key>StandardOutPath</key><string>{logs / "serve.log"}</string>'
                f'<key>StandardErrorPath</key><string>{logs / "serve.err.log"}</string>'
                '</dict></plist>\n',
                encoding="utf-8",
            )
            uid = os.getuid()
            execute("launchctl", "bootout", f"gui/{uid}", str(plist), optional=True)
            execute("launchctl", "bootstrap", f"gui/{uid}", str(plist))
            detail = "LaunchAgent installed (com.leo.agent)"
        else:
            # Windows uses the task scheduler / manual run; the wizard must not
            # fail just because there is no systemd or launchd here.
            return {
                "ok": True,
                "detail": f"no systemd/LaunchAgent on {sys.platform}; start with `leo serve`",
            }
        return {"ok": True, "detail": detail}
    except (OSError, RuntimeError) as exc:
        return {"ok": False, "detail": str(exc)}


def service_status(
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Report whether the installed service is currently active."""
    runner = run or subprocess.run
    if sys.platform.startswith("linux"):
        unit = Path.home() / ".config" / "systemd" / "user" / "leo.service"
        if not unit.is_file():
            return {"ok": False, "active": False, "detail": "no service unit installed"}
        completed = runner(
            ["systemctl", "--user", "is-active", "leo.service"],
            capture_output=True, text=True, check=False,
        )
        active = completed.returncode == 0 and "active" in str(completed.stdout)
        return {"ok": active, "active": active, "detail": "leo.service"}
    if sys.platform == "darwin":
        plist = Path.home() / "Library" / "LaunchAgents" / "com.leo.agent.plist"
        if not plist.is_file():
            return {"ok": False, "active": False, "detail": "no LaunchAgent installed"}
        completed = runner(
            ["launchctl", "list", "com.leo.agent"],
            capture_output=True, text=True, check=False,
        )
        active = completed.returncode == 0 and "com.leo.agent" in str(completed.stdout)
        return {"ok": active, "active": active, "detail": "com.leo.agent"}
    return {
        "ok": True, "active": True,
        "detail": f"no local service manager on {sys.platform}; not applicable",
    }


def restart_service(
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Safe repair for a stopped service: restart, do not reinstall."""
    runner = run or subprocess.run
    if sys.platform.startswith("linux"):
        completed = runner(
            ["systemctl", "--user", "restart", "leo.service"],
            capture_output=True, text=True, check=False,
        )
    elif sys.platform == "darwin":
        completed = runner(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.leo.agent"],
            capture_output=True, text=True, check=False,
        )
    else:
        return {"ok": True, "detail": f"no local service manager on {sys.platform}; not applicable"}
    return {
        "ok": completed.returncode == 0,
        "detail": f"service restart exit {completed.returncode}",
    }
