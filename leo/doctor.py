"""Real diagnostics for ``leo doctor``.

Each check returns one pass/fail row with a redacted detail line. Only safe,
reversible repairs are attempted under ``--fix``: recreating missing local
directories/database and restarting a stopped service. Bad credentials, wrong
model ids and missing secrets are reported, never guessed at or rewritten.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import yaml

from .config import Settings
from .models.catalog import probe_provider
from .onboarding_wizard import (
    default_env_file,
    load_env_file,
    restart_service,
    service_status,
)

_CHANNEL_CREDENTIALS = {
    "telegram": ("TELEGRAM_BOT_TOKEN",),
    "whatsapp": ("WHATSAPP_CLOUD_TOKEN", "WHATSAPP_PHONE_NUMBER_ID"),
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fixable: bool = False
    fixed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "fixable": self.fixable,
            "fixed": self.fixed,
        }


@dataclass
class _Check:
    name: str
    run: Callable[[], tuple[bool, str]]
    repair: Callable[[], tuple[bool, str]] | None = None


def _config_check(settings_source: Callable[[], Settings], path: Path) -> _Check:
    def run() -> tuple[bool, str]:
        if not path.is_file():
            return False, f"config file not found: {path}"
        try:
            settings_source()
        except (ValueError, OSError) as exc:
            return False, f"invalid config at {path}: {type(exc).__name__}"
        return True, f"config parsed: {path}"

    return _Check("config", run)


def _keys_check(settings: Settings, env_file: Path) -> _Check:
    def run() -> tuple[bool, str]:
        needed = sorted({
            provider.api_key_env
            for provider in settings.models.providers.values()
            if provider.enabled and provider.api_key_env
        })
        if not needed:
            return True, "no key-backed providers configured (names only)"
        if not env_file.is_file():
            return False, f"missing env file {env_file}; expected key names: {', '.join(needed)}"
        missing = [name for name in needed if not (os.getenv(name) or "").strip()]
        if missing:
            return False, f"missing key env var(s): {', '.join(missing)}"
        return True, f"key names set: {', '.join(needed)}"

    return _Check("keys", run)


def _provider_check(settings: Settings, probe: Callable[..., Any]) -> _Check:
    def run() -> tuple[bool, str]:
        name = settings.models.default
        provider = settings.models.providers.get(name)
        if provider is None:
            return False, f"default provider {name!r} is not configured"
        if not provider.enabled:
            return False, f"default provider {name!r} is disabled"
        if provider.kind == "mock" or not provider.base_url:
            return True, f"{name}: offline provider, no live probe required"
        key = os.getenv(provider.api_key_env) if provider.api_key_env else None
        verdict = probe(
            provider.base_url, api_key=key, kind=provider.kind,
            timeout=float(provider.timeout_seconds or 8.0),
        )
        if not verdict.ok:
            return False, f"{name}: {verdict.detail}"
        return True, f"{name}: key accepted ({verdict.detail})"

    return _Check("provider", run)


def _models_check(settings: Settings, path: Path) -> _Check:
    def run() -> tuple[bool, str]:
        providers = settings.models.providers
        # Settings.load repairs dead default/fallback names at runtime, which
        # would hide the very misconfiguration a doctor run must report.
        raw: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            except (OSError, yaml.YAMLError):
                pass
        declared = raw.get("models") if isinstance(raw.get("models"), dict) else {}
        default = declared.get("default", settings.models.default)
        fallback = declared.get("fallback", settings.models.fallback)
        problems: list[str] = []
        for role, name in (("default", default), ("fallback", fallback)):
            if not name:
                continue
            provider = providers.get(name)
            if provider is None:
                problems.append(f"{role}={name} (not configured)")
            elif not provider.enabled:
                problems.append(f"{role}={name} (disabled)")
            elif not provider.model.strip():
                problems.append(f"{role}={name} (no model id)")
        if problems:
            return False, "unresolvable model slot(s): " + ", ".join(problems)
        parts = [f"default={default}"]
        if fallback:
            parts.append(f"fallback={fallback}")
        vision = settings.models.vision.provider
        if vision:
            parts.append(f"vision={vision}")
        return True, ", ".join(parts)

    return _Check("models", run)


def _db_check(settings: Settings) -> _Check:
    def run() -> tuple[bool, str]:
        path = settings.db_path
        if not path.is_file():
            return False, f"database not found: {path}"
        try:
            uri = f"file:{quote(str(path.resolve()))}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
                conn.execute("SELECT 1").fetchone()
                result = conn.execute("PRAGMA quick_check").fetchone()
            if result and str(result[0]).lower() != "ok":
                return False, f"database integrity check failed: {result[0]}"
        except sqlite3.Error as exc:
            return False, f"database unreadable: {type(exc).__name__}"
        return True, f"database readable: {path}"

    def repair() -> tuple[bool, str]:
        from . import db

        try:
            settings.ensure_dirs()
            asyncio.run(db.initialize(settings.db_path))
        except (OSError, sqlite3.Error) as exc:
            return False, f"database repair failed: {type(exc).__name__}"
        return True, f"recreated database directories and schema: {settings.db_path}"

    return _Check("database", run, repair)


def _service_check(status: Callable[..., dict[str, Any]]) -> _Check:
    def run() -> tuple[bool, str]:
        current = status()
        return bool(current.get("ok")), str(current.get("detail", "unknown service state"))

    def repair() -> tuple[bool, str]:
        restarted = restart_service()
        return bool(restarted.get("ok")), str(restarted.get("detail", "restart failed"))

    return _Check("service", run, repair)


def _channels_check(settings: Settings) -> _Check:
    def run() -> tuple[bool, str]:
        policies = settings.channels.policies
        active = {
            name: policy for name, policy in policies.items()
            if policy.mode != "deny"
        }
        if not active:
            return True, "no external channels configured"
        problems: list[str] = []
        ok: list[str] = []
        for name, policy in sorted(active.items()):
            for env_name in _CHANNEL_CREDENTIALS.get(name, ()):
                if not (os.getenv(env_name) or "").strip():
                    problems.append(f"{name}: missing {env_name}")
                    break
            else:
                if policy.mode == "allowlist" and not policy.allow:
                    problems.append(f"{name}: empty allowlist")
                else:
                    ok.append(name)
        if problems:
            return False, "; ".join(problems)
        return True, f"channels verified (names only): {', '.join(ok)}"

    return _Check("channels", run)


def run_diagnostics(
    settings: Settings,
    *,
    config_path: str | Path | None = None,
    env_file: str | Path | None = None,
    fix: bool = False,
    probe: Callable[..., Any] | None = None,
    service: Callable[..., dict[str, Any]] | None = None,
    config_source: Callable[[], Settings] | None = None,
) -> list[CheckResult]:
    """Run every check once; with ``fix``, retry only safely repaired rows."""
    path = Path(config_path or os.getenv("LEO_CONFIG", "leo.yaml")).expanduser().resolve()
    env = Path(env_file) if env_file else default_env_file()
    env = env.expanduser().resolve()
    # The launcher sources this file; direct `python -m leo... doctor` must see
    # the same credentials for the check to be truthful.
    for name, value in load_env_file(env).items():
        os.environ.setdefault(name, value)

    checker = probe or probe_provider
    status = service or service_status
    checks = [
        _config_check(config_source or (lambda: Settings.load(path)), path),
        _keys_check(settings, env),
        _provider_check(settings, checker),
        _models_check(settings, path),
        _db_check(settings),
        _service_check(status),
        _channels_check(settings),
    ]
    results: list[CheckResult] = []
    for check in checks:
        ok, detail = check.run()
        fixed = False
        if not ok and fix and check.repair is not None:
            repaired, repair_detail = check.repair()
            if repaired:
                ok, detail = check.run()
                fixed = ok
            if not ok:
                detail = f"{detail} ({repair_detail})" if repair_detail else detail
        results.append(
            CheckResult(check.name, ok, detail, fixable=check.repair is not None, fixed=fixed)
        )
    return results
