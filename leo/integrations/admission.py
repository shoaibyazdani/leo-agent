"""Central inbound admission for every channel.

The OpenClaw audit's structural lesson: per-adapter authorization behind a
trusting dispatcher means one adapter bug becomes an authorization bypass.
LEO therefore decides inbound admission in exactly one place — this module —
and adapters only supply identity facts (channel, sender).

The resolver is fail closed:

- ``internal`` surfaces (CLI/API/dashboard) are the local user's own
  interfaces and are always allowed.
- An external channel in ``allowlist`` mode with an **empty** allowlist
  denies everyone. A channel nobody configured must never fall open.
- The decision always carries a machine-readable ``source`` so every reject
  can be traced to the exact policy that produced it. There are no layered
  fallbacks and no implicit "open".
"""

from __future__ import annotations

import json
from typing import Any

from .. import db
from ..config import ChannelPolicyConfig, ChannelsConfig
from ..contracts import BaseRecord, utcnow
from ..events import Event
from ..logging import get_logger

_LOG = get_logger("integrations.admission")


def _dump(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


POLICY_KV_KEY = "channel_policies"
POLICY_VERSION_KV_KEY = "channel_policies_version"
ADMISSION_DENIED_TOPIC = "gateway.admission.denied"
POLICY_REFRESHED_TOPIC = "channel.policy.refreshed"


class PolicyVersionError(RuntimeError):
    """Optimistic version check failed during a policy refresh."""


class AdmissionDecision(BaseRecord):
    allowed: bool
    reason: str
    source: str


class ChannelAdmission:
    """Single policy resolver for inbound channel messages."""

    def __init__(self, config: ChannelsConfig | None = None, *, events: Any = None) -> None:
        self.config = config or ChannelsConfig()
        # YAML/config-file policies are the immutable base; runtime enrollment
        # overlays DB-backed policies on top of them.
        self._base_policies: dict[str, dict[str, Any]] = {
            name: policy.model_dump() for name, policy in self.config.policies.items()
        }
        self.events = events
        self._version = 0

    @classmethod
    def from_settings(cls, settings: Any, *, events: Any = None) -> "ChannelAdmission":
        return cls(getattr(settings, "channels", None), events=events)

    async def load_policies(self) -> None:
        """Merge DB-backed policies over the config-file base at startup."""
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (POLICY_KV_KEY,))
        merged: dict[str, dict[str, Any]] = dict(self._base_policies)
        if row is not None:
            try:
                stored = json.loads(str(row["value"]))
                if isinstance(stored, dict):
                    for name, value in stored.items():
                        if isinstance(value, dict):
                            merged[str(name)] = ChannelPolicyConfig.model_validate(value).model_dump()
            except (TypeError, ValueError) as exc:
                _LOG.warning("stored channel policies invalid; using config base", extra={"error": type(exc).__name__})
        self.config.policies = {
            name: ChannelPolicyConfig.model_validate(value) for name, value in merged.items()
        }
        self._version = await self.policy_version()

    async def policy_version(self) -> int:
        """Durable monotonic policy version, shared by all gateway processes."""
        row = await db.fetchone("SELECT value FROM kv WHERE key=?", (POLICY_VERSION_KV_KEY,))
        if row is None:
            return 0
        try:
            return max(0, int(str(row["value"])))
        except (TypeError, ValueError):
            return 0

    async def save_policies(self) -> int:
        """Persist the current policy set and bump the durable version."""
        payload = json.dumps(
            {name: policy.model_dump() for name, policy in self.config.policies.items()},
            separators=(",", ":"),
        )
        now = utcnow()
        async with db.transaction() as conn:
            version = await self._version_in(conn) + 1
            await conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (POLICY_KV_KEY, payload, now),
            )
            await conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (POLICY_VERSION_KV_KEY, str(version), now),
            )
        self._version = version
        return version

    async def refresh_policies(self, *, expected_version: int | None = None) -> dict[str, Any]:
        """Reload policies from durable storage under an optimistic version gate.

        Returns the new version and snapshot; emits a structured
        ``channel.policy.refreshed`` event and an audit record.
        """
        current = await self.policy_version()
        if expected_version is not None and int(expected_version) != current:
            raise PolicyVersionError(
                f"policy version mismatch: expected {expected_version}, found {current}"
            )
        await self.load_policies()
        version = current + 1
        async with db.transaction() as conn:
            await conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (POLICY_VERSION_KV_KEY, str(version), utcnow()),
            )
        self._version = version
        result = {
            "version": version,
            "previous_version": current,
            "source": POLICY_KV_KEY,
            "policies": self.snapshot(),
        }
        await self._publish(POLICY_REFRESHED_TOPIC, result)
        await self._audit(
            action="channel.policy.refresh",
            resource=POLICY_KV_KEY,
            reason=f"policies refreshed from {POLICY_KV_KEY}",
            metadata=result,
        )
        return result

    @staticmethod
    async def _version_in(conn: Any) -> int:
        cursor = await conn.execute("SELECT value FROM kv WHERE key=?", (POLICY_VERSION_KV_KEY,))
        row = await cursor.fetchone()
        if row is None:
            return 0
        try:
            return max(0, int(str(row["value"])))
        except (TypeError, ValueError):
            return 0

    async def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        if self.events is None:
            return
        event = Event(topic=topic, source="admission", payload=payload)
        publish = getattr(self.events, "publish", None)
        if publish is not None:
            await publish(event)
            return
        emit = getattr(self.events, "emit", None)
        if emit is not None:
            await emit(topic, **payload)

    async def _audit(
        self,
        *,
        action: str,
        resource: str | None,
        reason: str,
        metadata: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        await db.execute(
            "INSERT INTO audit_log(ts,actor,action,resource,risk,decision,reason,metadata,correlation_id) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                utcnow(), "gateway", action, resource, "low",
                "denied" if "denied" in action else "allowed",
                reason, _dump(metadata), correlation_id,
            ),
        )

    async def record_denial(
        self,
        channel: str,
        sender: str,
        decision: AdmissionDecision,
        *,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Emit a structured denial event plus an audit record with policy provenance."""
        payload = {
            "channel": channel,
            "sender": sender,
            "reason": decision.reason,
            "policy_source": decision.source,
            "policy_version": self._version,
            "decision": "denied",
        }
        if self.events is not None:
            event = Event(
                topic=ADMISSION_DENIED_TOPIC,
                source="admission",
                payload=payload,
                correlation_id=correlation_id,
            )
            publish = getattr(self.events, "publish", None)
            if publish is not None:
                await publish(event)
            else:
                await self.events.emit(ADMISSION_DENIED_TOPIC, **payload)
        await self._audit(
            action=ADMISSION_DENIED_TOPIC,
            resource=channel,
            reason=f"{decision.reason} [policy={decision.source}]",
            metadata=payload,
            correlation_id=correlation_id,
        )
        return payload

    def set_policy(self, channel: str, policy: ChannelPolicyConfig) -> None:
        self.config.policies[channel] = policy

    def remove_policy(self, channel: str) -> None:
        self.config.policies.pop(channel, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: policy.model_dump() for name, policy in self.config.policies.items()}

    def decide(self, channel: str, sender: str) -> AdmissionDecision:
        config = self.config
        if channel in config.internal:
            return AdmissionDecision(
                allowed=True, reason="internal surface", source=f"internal:{channel}"
            )
        policy = config.policies.get(channel)
        mode = policy.mode if policy is not None else config.default_mode
        source = f"channel-policy:{channel}" if policy is not None else f"default:{mode}"
        if mode == "open":
            return AdmissionDecision(allowed=True, reason="channel policy is open", source=source)
        if mode == "deny":
            return AdmissionDecision(
                allowed=False, reason="channel denied by policy", source=source
            )
        allow = list(policy.allow) if policy is not None else []
        if not allow:
            return AdmissionDecision(
                allowed=False,
                reason=f"channel '{channel}' has no allowlist; inbound denied (fail closed)",
                source=source,
            )
        if "*" in allow or sender in allow:
            return AdmissionDecision(
                allowed=True, reason="sender admitted by allowlist", source=source
            )
        return AdmissionDecision(
            allowed=False,
            reason=f"sender '{sender}' is not in the allowlist for channel '{channel}'",
            source=source,
        )


__all__ = [
    "ADMISSION_DENIED_TOPIC",
    "POLICY_KV_KEY",
    "POLICY_REFRESHED_TOPIC",
    "POLICY_VERSION_KV_KEY",
    "AdmissionDecision",
    "ChannelAdmission",
    "PolicyVersionError",
]
