"""Runtime channel enrollment (dashboard/CLI facing, transport agnostic).

This module is the single write path that turns a channel spec into broker
credentials plus an inbound admission policy. It deliberately has no HTTP
framework dependency: callers supply an optional :class:`Gateway` and get plain
dictionaries back.

Security invariants:

- Field names are environment-variable *names*, never secret values.
- Secret values are written only through the credential broker, never returned
  from these helpers and never embedded in errors, events or logs.
- Credentials for one channel are deleted by its spec's field names only, so
  disabling a channel cannot touch an unrelated channel's credentials.
- Enrollment is atomic: credentials, admission policy, integration records and
  events are staged with a rollback journal; a late failure undoes earlier
  writes instead of leaving a half-enabled or half-disabled channel.
- Disabling installs a durable ``deny`` policy and credential deny markers, so
  neither permissive YAML nor an exported environment variable can revive the
  channel after a restart.
- No network calls: status comes from honest integration records, not from a
  live probe of the messaging platform.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from .. import db
from ..config import ChannelPolicyConfig
from ..logging import get_logger
from .channels import ALL_ADAPTERS, integration_records, sync_integration_records
from .channels.specs import channel_specs, spec_for
from .onboarding import channel_guide, device_link_status, field_guidance, validate_values
from .secrets import CredentialStoreError, default_store

if TYPE_CHECKING:  # pragma: no cover - annotation only; avoids a heavy import cycle
    from .gateway import Gateway


_LOG = get_logger("integrations.enrollment")

POLICY_FIELDS = frozenset({"allowlist", "mode"})
MODE_NOTE = "inbound denied until allowlist configured"
# Channels whose adapters accept either credential for sending. Requiring both
# in the UI would be wrong (DiscordAdapter.record() also accepts either one).
ALTERNATIVE_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "discord": (("DISCORD_BOT_TOKEN", "DISCORD_WEBHOOK_URL"),),
}
_POLICY_KV_KEY = "channel_policies"
_INTEGRATION_COLUMNS = (
    "name,kind,status,config_ref,capabilities,verification,created_at,updated_at"
)
_Undo = Callable[[], Awaitable[None]]


class EnrollmentError(ValueError):
    """Enrollment input was invalid or a credential could not be stored.

    ``field_errors`` carries field-level, value-free messages so a guided UI can
    show "what to fix" next to the right input. The exception text itself stays
    a single safe sentence and never contains a submitted credential.
    """

    def __init__(self, message: str, *, field_errors: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field_errors = field_errors or []


@asynccontextmanager
async def _atomic() -> AsyncIterator[list[_Undo]]:
    """Run staged writes; undo them in reverse order if any stage fails."""
    undo: list[_Undo] = []
    try:
        yield undo
    except BaseException:
        for rollback in reversed(undo):
            try:
                await rollback()
            except Exception as exc:  # noqa: BLE001 - preserve the original error
                _LOG.warning("enrollment rollback failed", extra={"error": type(exc).__name__})
        raise


async def _snapshot_policy(gateway: "Gateway") -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    row = await db.fetchone("SELECT value,updated_at FROM kv WHERE key=?", (_POLICY_KV_KEY,))
    return dict(gateway.admission.snapshot()), dict(row) if row is not None else None


async def _restore_policy(gateway: "Gateway", snapshot: tuple[dict[str, Any], dict[str, Any] | None]) -> None:
    policies, row = snapshot
    gateway.admission.config.policies = {
        str(name): ChannelPolicyConfig.model_validate(value) for name, value in policies.items()
    }
    if row is None:
        await db.execute("DELETE FROM kv WHERE key=?", (_POLICY_KV_KEY,))
    else:
        await _upsert_kv(_POLICY_KV_KEY, row["value"], row["updated_at"])


async def _upsert_kv(key: str, value: Any, updated_at: Any) -> None:
    await db.execute(
        "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, updated_at),
    )


async def _snapshot_integrations() -> list[dict[str, Any]]:
    return [dict(row) for row in await db.fetchall(f"SELECT {_INTEGRATION_COLUMNS} FROM integrations")]


async def _restore_integrations(rows: list[dict[str, Any]]) -> None:
    current = {str(row["name"]) for row in await db.fetchall("SELECT name FROM integrations")}
    keep = {str(row["name"]) for row in rows}
    for name in current - keep:
        await db.execute("DELETE FROM integrations WHERE name=?", (name,))
    for row in rows:
        await db.execute(
            "INSERT INTO integrations(name,kind,status,config_ref,capabilities,verification,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "kind=excluded.kind, status=excluded.status, config_ref=excluded.config_ref, "
            "capabilities=excluded.capabilities, verification=excluded.verification, "
            "updated_at=excluded.updated_at",
            tuple(row[column] for column in _INTEGRATION_COLUMNS.split(",")),
        )


def _field_name(field: Any) -> str:
    return str(getattr(field, "name", "") or "")


def _field_label(field: Any) -> str:
    return str(getattr(field, "label", "") or _field_name(field))


def _field_kind(field: Any) -> str:
    kind = getattr(field, "kind", "text")
    return str(getattr(kind, "value", kind))


def _spec(channel: str) -> Any:
    spec = channel_specs().get(channel)
    if spec is None:
        spec = spec_for(channel)
    if spec is None:
        raise EnrollmentError(f"unknown channel: {channel}")
    return spec


def _fields(spec: Any) -> list[Any]:
    return [field for field in (getattr(spec, "fields", None) or []) if _field_name(field)]


def _credential_fields(spec: Any) -> list[Any]:
    """Spec fields that map to broker credentials (policy fields excluded)."""
    return [field for field in _fields(spec) if _field_name(field) not in POLICY_FIELDS]


def _normalize(field: Any, raw: Any) -> str:
    """Strip a submitted value; multi-line tokens are always a mistake."""
    value = "" if raw is None else str(raw).strip()
    if value and _field_kind(field) != "textarea" and ("\n" in value or "\r" in value):
        # Name the field/label only. The submitted token is never echoed.
        raise EnrollmentError(
            f"{_field_label(field)} ({_field_name(field)}) must not contain newlines"
        )
    return value


async def enable_channel(
    channel: str,
    values: dict[str, str],
    *,
    allowlist: list[str] | None = None,
    mode: str | None = None,
    gateway: "Gateway" | None = None,
) -> dict[str, Any]:
    """Validate, broker and audit a channel's credentials and inbound policy."""
    if channel not in ALL_ADAPTERS:
        raise EnrollmentError(f"unknown channel: {channel}")
    spec = _spec(channel)
    fields = _fields(spec)
    known = {_field_name(field) for field in fields}

    # Reject input before any state changes: no partial enrollment on a typo.
    unknown = sorted(set(values) - known)
    if unknown:
        raise EnrollmentError(
            f"unknown field(s) for channel '{channel}': {', '.join(unknown)}"
        )

    # Guided validation happens before any write. Errors are attached per field
    # (label + what to do) and never echo the submitted value.
    validation_input: dict[str, Any] = dict(values)
    if allowlist is not None:
        validation_input["allowlist"] = ", ".join(str(item) for item in allowlist)
    if mode is not None:
        validation_input["mode"] = mode
    errors = validate_values(channel, validation_input)
    if errors:
        raise EnrollmentError(errors[0]["message"], field_errors=errors)

    credential_fields = _credential_fields(spec)
    resolved: dict[str, str] = {}
    for field in credential_fields:
        name = _field_name(field)
        value = _normalize(field, values.get(name)) if name in values else ""
        if not value:
            # An omitted field keeps an already-configured (broker or env) value.
            existing = default_store.get(name)
            value = str(existing).strip() if existing else ""
        if getattr(field, "required", False) and not value:
            # Name the label and env var; never include a value (there is none).
            raise EnrollmentError(f"{_field_label(field)} ({name}) is required")
        resolved[name] = value

    for group in ALTERNATIVE_GROUPS.get(channel, ()):
        if not any(resolved.get(name) for name in group):
            raise EnrollmentError(
                f"one of {', '.join(group)} is required to enable {channel}"
            )

    names = [_field_name(field) for field in credential_fields]
    async with _atomic() as undo:
        cred_snapshot = await default_store.snapshot(names)
        undo.append(lambda snap=cred_snapshot: default_store.restore(snap))
        for field in credential_fields:
            name = _field_name(field)
            value = resolved[name]
            if value and (name in values or value != (default_store.get(name) or "")):
                try:
                    await default_store.set_async(name, value)
                except CredentialStoreError as exc:
                    raise EnrollmentError(f"could not store credential for {name}: {exc}") from exc

        if gateway is not None and (allowlist is not None or mode is not None):
            policy_snapshot = await _snapshot_policy(gateway)
            undo.append(lambda snap=policy_snapshot: _restore_policy(gateway, snap))
            policy = ChannelPolicyConfig(
                mode=mode or "allowlist",
                allow=[str(item).strip() for item in (allowlist or []) if str(item).strip()],
            )
            gateway.admission.set_policy(channel, policy)
            await gateway.admission.save_policies()

        integration_snapshot = await _snapshot_integrations()
        undo.append(lambda snap=integration_snapshot: _restore_integrations(snap))
        await sync_integration_records()

        if gateway is not None:
            state: dict[str, Any] = {}

            async def drop_event() -> None:
                event = state.get("event")
                if event is not None:
                    await db.execute("DELETE FROM events WHERE id=?", (event.id,))

            undo.append(drop_event)
            state["event"] = await gateway.events.emit("channel.enabled", channel=channel)

    return await channel_details(channel, gateway=gateway)


async def disable_channel(
    channel: str, *, gateway: "Gateway" | None = None
) -> dict[str, Any]:
    """Remove a channel's own credentials/policy and re-sync its record."""
    if channel not in ALL_ADAPTERS:
        raise EnrollmentError(f"unknown channel: {channel}")
    spec = _spec(channel)
    credential_fields = _credential_fields(spec)
    names = [_field_name(field) for field in credential_fields]

    async with _atomic() as undo:
        cred_snapshot = await default_store.snapshot(names)
        undo.append(lambda snap=cred_snapshot: default_store.restore(snap))
        # Delete by this spec's field names only. No wildcard, no other channel.
        # A deny marker keeps an exported environment variable from restoring
        # the credential on this process or after a restart.
        for name in names:
            try:
                await default_store.revoke(name)
            except CredentialStoreError as exc:
                raise EnrollmentError(f"could not remove credential for {name}: {exc}") from exc

        if gateway is not None:
            policy_snapshot = await _snapshot_policy(gateway)
            undo.append(lambda snap=policy_snapshot: _restore_policy(gateway, snap))
            # Never remove the policy: removing it would fall back to the YAML
            # base, which may be `mode: open`. Persist an explicit deny instead.
            gateway.admission.set_policy(channel, ChannelPolicyConfig(mode="deny", allow=[]))
            await gateway.admission.save_policies()

        integration_snapshot = await _snapshot_integrations()
        undo.append(lambda snap=integration_snapshot: _restore_integrations(snap))
        await sync_integration_records()

        if gateway is not None:
            state: dict[str, Any] = {}

            async def drop_event() -> None:
                event = state.get("event")
                if event is not None:
                    await db.execute("DELETE FROM events WHERE id=?", (event.id,))

            undo.append(drop_event)
            state["event"] = await gateway.events.emit("channel.disabled", channel=channel)

    return await channel_details(channel, gateway=gateway)


async def channel_details(
    channel: str, *, gateway: "Gateway" | None = None
) -> dict[str, Any]:
    """Describe a channel for the dashboard. Never includes secret values."""
    if channel not in ALL_ADAPTERS:
        raise EnrollmentError(f"unknown channel: {channel}")
    spec = _spec(channel)
    adapter_cls = ALL_ADAPTERS[channel]

    records = (
        gateway.integration_records() if gateway is not None else integration_records()
    )
    record = next((item for item in records if item.name == channel), None)
    policy: dict[str, Any] | None = None
    if gateway is not None:
        policy = gateway.admission.snapshot().get(channel)

    field_details: list[dict[str, Any]] = []
    for field in _fields(spec):
        name = _field_name(field)
        if name in POLICY_FIELDS:
            configured = bool(
                policy
                and (
                    policy.get("allow") if name == "allowlist" else policy.get("mode")
                )
            )
        else:
            configured = bool(default_store.get(name))
        options = getattr(field, "options", None) or []
        if not isinstance(options, (list, tuple, set)):
            options = []
        guidance = field_guidance(
            name, kind=_field_kind(field), label=_field_label(field)
        )
        help_text = str(getattr(field, "help", "") or "")
        field_details.append(
            {
                "name": name,
                "label": _field_label(field),
                "kind": _field_kind(field),
                "required": bool(getattr(field, "required", False)),
                "help": help_text,
                "options": list(options),
                "configured": configured,
                # Guided-form metadata. All of it is static/derived from the
                # spec: no submitted or stored credential value is included.
                "default": str(getattr(field, "default", "") or ""),
                "example": guidance["example"],
                "placeholder": guidance["placeholder"],
                "hint": guidance["hint"] or help_text,
                "validation": guidance["validation"],
                "sensitive": guidance["sensitive"],
            }
        )

    description = str(getattr(spec, "description", "") or "").strip()
    if not description:
        description = (adapter_cls.__doc__ or "").strip().splitlines()[0].strip()

    # A missing policy, or one that is not open and has no allowlist, is closed.
    mode_note = None
    if not policy or (str(policy.get("mode")) != "open" and not policy.get("allow")):
        mode_note = MODE_NOTE

    docs_url = str(getattr(spec, "docs_hint", "") or "")
    return {
        "channel": channel,
        "label": str(getattr(spec, "label", "") or channel),
        "description": description,
        "capabilities": list(record.capabilities) if record is not None else [],
        "status": record.status if record is not None else "blocked",
        "fields": field_details,
        # Guided onboarding metadata for the dashboard wizard.
        "docs_url": docs_url,
        "guide": channel_guide(channel, fields=field_details, docs_url=docs_url),
        "device_link": device_link_status(channel),
        "policy": policy,
        "webhook": bool(getattr(adapter_cls, "supports_webhook", False)),
        "poll": bool(getattr(adapter_cls, "supports_polling", False)),
        "mode_note": mode_note,
        # Without LEO_CREDENTIALS_KEY, values live only in this process: the UI
        # must tell the user they will need to re-enter them after a restart.
        "persistence": "encrypted" if default_store.persistent else "session-only",
        "persistence_note": (
            None
            if default_store.persistent
            else "No LEO_CREDENTIALS_KEY set: credentials stay in memory only and are lost on restart."
        ),
    }


async def list_channels(*, gateway: "Gateway" | None = None) -> list[dict[str, Any]]:
    """Details for every registered channel, sorted by channel name."""
    return [await channel_details(name, gateway=gateway) for name in sorted(ALL_ADAPTERS)]


__all__ = [
    "EnrollmentError",
    "enable_channel",
    "disable_channel",
    "channel_details",
    "list_channels",
]
