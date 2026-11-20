"""Tests for the declarative channel catalog (:mod:`...channels.specs`).

These tests keep the dashboard's enable-channel form honest: every registered
channel must have a spec, every declared environment variable must appear
exactly once, field kinds must match the secret/non-secret split, the shared
policy fields must be present with the right options, and no secret *value*
may ever leak into a dumped spec (only names and hints).
"""

from __future__ import annotations

import json

import pytest

from leo.integrations.channels import ALL_ADAPTERS
from leo.integrations.channels.specs import (
    FieldKind,
    channel_specs,
    is_special_field,
    spec_for,
)

_SPECIAL = {"allowlist", "mode"}
_SECRET_HINTS = ("TOKEN", "SECRET", "PASSWORD", "KEY")
_TEXT_HINTS = ("URL", "HOST", "HOMESERVER", "PORT", "ID", "NUMBER", "NICKNAME", "FROM")


def _declared_envs(cls) -> list[str]:
    names: list[str] = []
    for attr in ("credential_env", "verification_env", "poll_env"):
        names.extend(getattr(cls, attr, ()) or ())
    return names


# (a) every registered channel has a spec with human label and description
def test_every_channel_has_spec_with_label_and_description():
    specs = channel_specs()
    assert set(specs) == set(ALL_ADAPTERS), "a registered channel is missing from the catalog"
    for channel, spec in specs.items():
        assert spec.channel == channel
        assert spec.label.strip(), f"{channel}: empty label"
        assert spec.description.strip(), f"{channel}: empty description"
        assert spec.fields, f"{channel}: no fields"
        assert spec.capabilities and "send" in spec.capabilities


# (b) every credential/verification/poll env name appears exactly once
@pytest.mark.parametrize("channel", sorted(ALL_ADAPTERS))
def test_declared_env_appears_exactly_once(channel):
    spec = spec_for(channel)
    field_names = [field.name for field in spec.fields]
    for env in set(_declared_envs(ALL_ADAPTERS[channel])):
        assert field_names.count(env) == 1, (
            f"{channel}: {env} appears {field_names.count(env)} times, expected exactly once"
        )


def test_duplicate_envs_are_deduplicated():
    # relay declares RELAY_SECRET in both credential_env and verification_env;
    # sms declares TWILIO_AUTH_TOKEN in both — neither may yield two fields.
    for channel in ("relay", "sms"):
        field_names = [f.name for f in spec_for(channel).fields]
        assert all(field_names.count(name) == 1 for name in field_names), (
            f"{channel}: duplicate fields found"
        )


# (c) secret-looking fields are SECRET; URLs/hosts/ports/ids are TEXT or NUMBER
def test_field_kinds_match_names():
    for channel, spec in channel_specs().items():
        for field in spec.fields:
            if is_special_field(field.name):
                continue
            upper = field.name.upper()
            if any(hint in upper for hint in _SECRET_HINTS):
                assert field.kind == FieldKind.SECRET, (
                    f"{channel}.{field.name}: secret-looking field is {field.kind}"
                )
            elif any(hint in upper for hint in _TEXT_HINTS):
                assert field.kind in (FieldKind.TEXT, FieldKind.NUMBER), (
                    f"{channel}.{field.name}: non-secret field is {field.kind}"
                )


# (d) allowlist + mode fields exist with correct shape
def test_policy_fields_on_every_spec():
    for channel, spec in channel_specs().items():
        by_name = {field.name: field for field in spec.fields}

        allowlist = by_name.get("allowlist")
        assert allowlist is not None, f"{channel}: missing allowlist field"
        assert allowlist.kind == FieldKind.TEXTAREA
        assert allowlist.required is False
        assert "Comma-separated" in allowlist.help

        mode = by_name.get("mode")
        assert mode is not None, f"{channel}: missing mode field"
        assert mode.kind == FieldKind.CHOICE
        assert mode.options == ["allowlist", "open", "deny"]
        assert mode.default == "allowlist"
        assert mode.required is False


def test_is_special_field():
    assert is_special_field("allowlist") is True
    assert is_special_field("mode") is True
    assert is_special_field("TELEGRAM_BOT_TOKEN") is False
    for channel, spec in channel_specs().items():
        env_fields = [f.name for f in spec.fields if not is_special_field(f.name)]
        assert set(env_fields).isdisjoint(_SPECIAL)


# (e) spec_for("unknown") raises
def test_spec_for_unknown_channel_raises():
    with pytest.raises((KeyError, ValueError)) as exc_info:
        spec_for("unknown")
    message = str(exc_info.value)
    assert "unknown" in message.lower()
    assert "telegram" in message, "error should list known channels"


def test_spec_for_known_channel():
    assert spec_for("telegram").label == "Telegram"
    assert spec_for("email").capabilities == ["send", "poll"]


# (f) no secret VALUES anywhere in the dumped specs (only names)
def test_no_secret_values_in_dumped_specs(monkeypatch):
    canaries = {
        "TELEGRAM_BOT_TOKEN": "CANARY_TELEGRAM_TOKEN_9f3a",
        "SLACK_BOT_TOKEN": "CANARY_SLACK_TOKEN_7b2d",
        "A2A_API_KEY": "CANARY_A2A_KEY_4c8e",
        "SMTP_PASSWORD": "CANARY_SMTP_PASSWORD_1c6b",
        "RELAY_SECRET": "CANARY_RELAY_SECRET_8d2f",
    }
    for env, value in canaries.items():
        monkeypatch.setenv(env, value)

    dumped = json.dumps([spec.model_dump(mode="json") for spec in channel_specs().values()])
    for env, value in canaries.items():
        assert env in dumped, f"field name {env} should be present"
        assert value not in dumped, f"secret value for {env} leaked into the catalog"
