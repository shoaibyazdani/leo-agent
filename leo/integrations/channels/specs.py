"""Declarative channel catalog used by the Integrations dashboard form.

This module is pure metadata: it defines no ``MessagingAdapter`` subclass, so
the auto-discovery in :mod:`leo.integrations.channels` ignores it. It is the
single source of truth for what the "enable a channel" form asks for: one field
per credential/verification/poll environment variable (by *name*, never value),
plus the two shared admission-policy fields (``allowlist`` and ``mode``) that
feed :class:`leo.config.ChannelPolicyConfig`.

Everything here is honest about provenance: official platform APIs are labelled
as such, community bridges are labelled as bridges, and no claim of live
delivery is made. Secret *values* never appear — only variable names and hints
about where to obtain them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "ChannelField",
    "ChannelSpec",
    "FieldKind",
    "channel_specs",
    "is_special_field",
    "spec_for",
]


class FieldKind(StrEnum):
    SECRET = "secret"
    TEXT = "text"
    NUMBER = "number"
    CHOICE = "choice"
    TEXTAREA = "textarea"


class ChannelField(BaseModel):
    name: str
    label: str
    kind: FieldKind
    required: bool = True
    help: str = ""
    options: list[str] = Field(default_factory=list)
    default: str = ""


class ChannelSpec(BaseModel):
    channel: str
    label: str
    description: str
    fields: list[ChannelField]
    capabilities: list[str]
    docs_hint: str = ""


# Shared admission-policy fields. They map onto ChannelPolicyConfig(mode, allow)
# and are deliberately last in every spec's field list.
_ALLOWLIST_HELP = "Comma-separated sender/user/chat IDs allowed to send messages; empty denies everyone"
_MODE_HELP = "How inbound senders are admitted: allowlist (only listed IDs), open (anyone), or deny (everyone blocked)"
_SPECIAL_FIELDS = frozenset({"allowlist", "mode"})


def is_special_field(name: str) -> bool:
    """True for the shared policy fields (not environment-variable names)."""
    return name in _SPECIAL_FIELDS


def _meta(
    label: str,
    kind: FieldKind,
    help: str,
    *,
    required: bool = True,
    options: tuple[str, ...] = (),
    default: str = "",
) -> tuple[str, FieldKind, str, bool, list[str], str]:
    return (label, kind, help, required, list(options), default)


# Curated metadata for EVERY environment variable used by the 22 registered
# adapters (credential_env, verification_env, poll_env and the optional extras
# below). Tokens/passwords/secrets/keys are SECRET; URLs/hosts/ports/ids/names
# are TEXT or NUMBER. Help text says what to obtain, not what to type.
_FIELD_META: dict[str, tuple[str, FieldKind, str, bool, list[str], str]] = {
    # telegram
    "TELEGRAM_BOT_TOKEN": _meta("Bot token", FieldKind.SECRET, "Bot token from @BotFather in Telegram."),
    "TELEGRAM_WEBHOOK_SECRET": _meta(
        "Webhook secret", FieldKind.SECRET,
        "Secret token you set with Telegram setWebhook (X-Telegram-Bot-Api-Secret-Token).",
    ),
    # discord — either the bot token or an incoming webhook is enough to send;
    # the public key is only needed to verify signed interaction ingress.
    "DISCORD_BOT_TOKEN": _meta("Bot token", FieldKind.SECRET, "Bot token from the Discord Developer Portal (Bot → Token). Either this or an incoming webhook URL is enough.", required=False),
    "DISCORD_WEBHOOK_URL": _meta("Webhook URL", FieldKind.TEXT, "HTTPS incoming webhook URL from a Discord channel's Integrations. Either this or a bot token is enough.", required=False),
    "DISCORD_PUBLIC_KEY": _meta("Application public key", FieldKind.SECRET, "Public key from the Discord Developer Portal used to verify Ed25519 interactions (signed inbound only).", required=False),
    # whatsapp (Cloud API)
    "WHATSAPP_CLOUD_TOKEN": _meta("Cloud API token", FieldKind.SECRET, "Permanent access token from the Meta WhatsApp Cloud API app."),
    "WHATSAPP_PHONE_NUMBER_ID": _meta("Phone number ID", FieldKind.TEXT, "Phone number ID from Meta Business Manager (WhatsApp → Phone numbers)."),
    "WHATSAPP_APP_SECRET": _meta("App secret", FieldKind.SECRET, "Meta app secret used to verify X-Hub-Signature-256 webhook signatures."),
    "WHATSAPP_API_VERSION": _meta("Graph API version", FieldKind.TEXT, "Graph API version to call; leave empty to use v20.0.", required=False, default="v20.0"),
    # slack
    "SLACK_BOT_TOKEN": _meta("Bot token", FieldKind.SECRET, "Slack bot token (xoxb-...) from your app's OAuth & Permissions page."),
    "SLACK_SIGNING_SECRET": _meta("Signing secret", FieldKind.SECRET, "Signing secret from your Slack app's Basic Information page."),
    # mattermost
    "MATTERMOST_TOKEN": _meta("Access token", FieldKind.SECRET, "Personal access token or bot token from your Mattermost account."),
    "MATTERMOST_URL": _meta("Server URL", FieldKind.TEXT, "Base URL of your Mattermost server (e.g. https://chat.example.com)."),
    "MATTERMOST_OUTGOING_TOKEN": _meta("Outgoing token", FieldKind.SECRET, "Shared token configured on the Mattermost outgoing webhook."),
    # line
    "LINE_CHANNEL_ACCESS_TOKEN": _meta("Channel access token", FieldKind.SECRET, "Channel access token from the LINE Developers Console (Messaging API)."),
    "LINE_CHANNEL_SECRET": _meta("Channel secret", FieldKind.SECRET, "Channel secret from the LINE Developers Console (Messaging API)."),
    # google_chat
    "GOOGLE_CHAT_WEBHOOK_URL": _meta("Webhook URL", FieldKind.TEXT, "Google Chat space incoming webhook URL from Integrations settings."),
    "GOOGLE_CHAT_VERIFICATION_TOKEN": _meta("Verification token", FieldKind.SECRET, "Bearer token expected on inbound Google Chat webhook requests."),
    # msteams
    "TEAMS_INCOMING_WEBHOOK_URL": _meta("Incoming webhook URL", FieldKind.TEXT, "Microsoft Teams channel incoming webhook (Workflows) URL."),
    "TEAMS_PUBLIC_KEY": _meta("Public key", FieldKind.SECRET, "Public key used to verify inbound Teams JWTs; verification fails closed without an injected verifier."),
    "TEAMS_APP_ID": _meta("Application ID", FieldKind.TEXT, "Microsoft Teams app ID expected as the inbound JWT audience (required for signed webhooks).", required=False),
    "TEAMS_JWT_ISSUER": _meta("JWT issuer", FieldKind.TEXT, "Expected issuer of inbound Teams JWTs; leave empty to use Microsoft's default issuer.", required=False),
    # dingtalk
    "DINGTALK_WEBHOOK_URL": _meta("Webhook URL", FieldKind.TEXT, "Custom robot webhook URL from your DingTalk group."),
    "DINGTALK_SECRET": _meta("Robot secret", FieldKind.SECRET, "Custom robot signing secret (SEC...) from the DingTalk group settings."),
    "DINGTALK_CALLBACK_SECRET": _meta("Callback secret", FieldKind.SECRET, "Shared secret for verifying DingTalk callback signatures."),
    # feishu
    "FEISHU_APP_ID": _meta("App ID", FieldKind.TEXT, "App ID from the Feishu/Lark open platform console."),
    "FEISHU_APP_SECRET": _meta("App secret", FieldKind.SECRET, "App secret from the Feishu/Lark open platform console."),
    "FEISHU_ENCRYPT_KEY": _meta("Encrypt key", FieldKind.SECRET, "Encrypt key configured for the Feishu event callback."),
    # qqbot
    "QQBOT_APP_ID": _meta("App ID", FieldKind.TEXT, "Bot AppID from the QQ bot developer platform (bot.q.qq.com)."),
    "QQBOT_APP_SECRET": _meta("App secret", FieldKind.SECRET, "Bot AppSecret from the QQ bot developer platform."),
    "QQBOT_PUBLIC_KEY": _meta("Public key", FieldKind.SECRET, "Bot public key used to verify Ed25519-signed QQ callbacks."),
    # wecom
    "WECOM_CORP_ID": _meta("Corp ID", FieldKind.TEXT, "WeCom CorpID from the admin console (My Enterprise → CorpID)."),
    "WECOM_SECRET": _meta("App secret", FieldKind.SECRET, "Secret of the WeCom app used to send messages."),
    "WECOM_CALLBACK_TOKEN": _meta("Callback token", FieldKind.SECRET, "Token configured on the WeCom callback URL."),
    "WECOM_ENCODING_AES_KEY": _meta("Encoding AES key", FieldKind.SECRET, "EncodingAESKey configured on the WeCom callback URL."),
    # weixin
    "WEIXIN_APP_ID": _meta("App ID", FieldKind.TEXT, "AppID of the WeChat Official Account."),
    "WEIXIN_APP_SECRET": _meta("App secret", FieldKind.SECRET, "AppSecret of the WeChat Official Account."),
    "WEIXIN_TOKEN": _meta("Callback token", FieldKind.SECRET, "Token configured on the WeChat Official Account callback URL."),
    "WEIXIN_ENCODING_AES_KEY": _meta("Encoding AES key", FieldKind.SECRET, "EncodingAESKey configured on the WeChat callback URL."),
    # sms (Twilio)
    "TWILIO_ACCOUNT_SID": _meta("Account SID", FieldKind.TEXT, "Twilio Account SID from the console (starts with AC)."),
    "TWILIO_AUTH_TOKEN": _meta("Auth token", FieldKind.SECRET, "Twilio Auth Token from the console; also signs inbound webhooks."),
    "TWILIO_FROM_NUMBER": _meta("From number", FieldKind.TEXT, "Twilio phone number in E.164 format used as the sender."),
    "TWILIO_WEBHOOK_URL": _meta("Webhook URL", FieldKind.TEXT, "Public HTTPS URL Twilio calls for message webhooks.", required=False),
    "TWILIO_BASE_URL": _meta("API base URL", FieldKind.TEXT, "Twilio API base URL override; leave empty for https://api.twilio.com.", required=False),
    # email (SMTP send + IMAP poll)
    "SMTP_HOST": _meta("SMTP host", FieldKind.TEXT, "Hostname of your SMTP server (e.g. smtp.example.com)."),
    "SMTP_USER": _meta("SMTP user", FieldKind.TEXT, "Username for SMTP authentication."),
    "SMTP_PASSWORD": _meta("SMTP password", FieldKind.SECRET, "Password or app password for SMTP authentication."),
    "EMAIL_FROM": _meta("From address", FieldKind.TEXT, "From address used for outbound mail."),
    "IMAP_HOST": _meta("IMAP host", FieldKind.TEXT, "Hostname of your IMAP server (e.g. imap.example.com)."),
    "IMAP_USER": _meta("IMAP user", FieldKind.TEXT, "Username for IMAP authentication."),
    "IMAP_PASSWORD": _meta("IMAP password", FieldKind.SECRET, "Password or app password for IMAP authentication."),
    "SMTP_PORT": _meta("SMTP port", FieldKind.NUMBER, "SMTP port override; leave empty for 587.", required=False, default="587"),
    "IMAP_PORT": _meta("IMAP port", FieldKind.NUMBER, "IMAP port override; leave empty for 993.", required=False, default="993"),
    # matrix
    "MATRIX_HOMESERVER": _meta("Homeserver URL", FieldKind.TEXT, "Base URL of your Matrix homeserver (e.g. https://matrix.example.com)."),
    "MATRIX_ACCESS_TOKEN": _meta("Access token", FieldKind.SECRET, "Matrix user access token obtained from the homeserver's /login endpoint."),
    # irc
    "IRC_HOST": _meta("Server host", FieldKind.TEXT, "Hostname of the IRC server (e.g. irc.libera.chat)."),
    "IRC_PORT": _meta("Server port", FieldKind.NUMBER, "IRC server port; leave empty for the verified-TLS default 6697.", required=False, default="6697"),
    "IRC_NICKNAME": _meta("Nickname", FieldKind.TEXT, "Nickname to register on the IRC network."),
    "IRC_CHANNELS": _meta("Channels", FieldKind.TEXT, "Comma-separated channels to join after registration (optional).", required=False),
    "IRC_PASSWORD": _meta("Server password", FieldKind.SECRET, "Server/NickServ password, if the network requires one (optional).", required=False),
    "IRC_TLS": _meta("TLS transport", FieldKind.CHOICE, "Use verified TLS with hostname/certificate checking (default: on). Cleartext needs IRC_TLS=0 and IRC_ALLOW_INSECURE=1.", required=False, options=("1", "0"), default="1"),
    "IRC_ALLOW_INSECURE": _meta("Allow insecure cleartext", FieldKind.CHOICE, "Explicit opt-in for unencrypted IRC connections; leave at 0 unless TLS is unavailable.", required=False, options=("0", "1"), default="0"),
    # ntfy
    "NTFY_URL": _meta("Server URL", FieldKind.TEXT, "Base URL of your ntfy server (e.g. https://ntfy.sh)."),
    "NTFY_TOPIC": _meta("Topic", FieldKind.TEXT, "Topic that poll() subscribes to (optional).", required=False),
    "NTFY_USER": _meta("Basic-auth user", FieldKind.TEXT, "Basic-auth username for ntfy, if required (optional).", required=False),
    "NTFY_PASSWORD": _meta("Basic-auth password", FieldKind.SECRET, "Basic-auth password for ntfy, if required (optional).", required=False),
    # bluebubbles (community bridge)
    "BLUEBUBBLES_URL": _meta("Server URL", FieldKind.TEXT, "Base URL of your BlueBubbles server (e.g. http://localhost:1234)."),
    "BLUEBUBBLES_PASSWORD": _meta("Server password", FieldKind.SECRET, "Password of your BlueBubbles server."),
    "BLUEBUBBLES_WEBHOOK_SECRET": _meta("Webhook secret", FieldKind.SECRET, "Shared secret for LEO's X-Leo-Signature webhook header."),
    # signal (community bridge)
    "SIGNAL_CLI_URL": _meta("signal-cli URL", FieldKind.TEXT, "Base URL of the signal-cli REST API (usually http://localhost:8080)."),
    "SIGNAL_PHONE_NUMBER": _meta("Phone number", FieldKind.TEXT, "Signal account phone number in E.164 format (e.g. +15551234567)."),
    "SIGNAL_WEBHOOK_SECRET": _meta("Webhook secret", FieldKind.SECRET, "Shared secret for LEO's X-Leo-Signature webhook header."),
    "SIGNAL_CLI_USER": _meta("Basic-auth user", FieldKind.TEXT, "Basic-auth username for signal-cli REST, if required (optional).", required=False),
    "SIGNAL_CLI_PASSWORD": _meta("Basic-auth password", FieldKind.SECRET, "Basic-auth password for signal-cli REST, if required (optional).", required=False),
    # a2a
    "A2A_AGENT_URL": _meta("Agent endpoint URL", FieldKind.TEXT, "Full JSON-RPC endpoint URL of the peer agent's A2A agent card."),
    "A2A_API_KEY": _meta("API key", FieldKind.SECRET, "Optional bearer token sent to the peer A2A agent.", required=False),
    # relay
    "RELAY_TARGET_URL": _meta("Target URL", FieldKind.TEXT, "HTTP endpoint that receives signed JSON envelopes."),
    "RELAY_SECRET": _meta("Shared secret", FieldKind.SECRET, "Shared HMAC secret used to sign and verify relay envelopes."),
}

# (label, description, docs_hint) per channel — honest about official vs bridge.
_CHANNEL_META: dict[str, tuple[str, str, str]] = {
    "telegram": (
        "Telegram",
        "Official Telegram Bot API sender and webhook update gateway; the bot token comes from @BotFather.",
        "https://core.telegram.org/bots/api",
    ),
    "discord": (
        "Discord",
        "Official Discord bot and interaction gateway using Ed25519-signed interactions and incoming webhooks.",
        "https://discord.com/developers/docs/interactions/receiving-and-responding",
    ),
    "whatsapp": (
        "WhatsApp",
        "Official Meta WhatsApp Cloud API sender and webhook gateway (not the community WhatsApp bridge).",
        "https://developers.facebook.com/docs/whatsapp/cloud-api",
    ),
    "slack": (
        "Slack",
        "Official Slack Web API sender with signed v0 webhook verification and a five-minute replay window.",
        "https://api.slack.com/authentication/verifying-requests-from-slack",
    ),
    "mattermost": (
        "Mattermost",
        "Official Mattermost REST API sender with outgoing-webhook ingress and a shared-token check.",
        "https://docs.mattermost.com/developer/webhooks-outgoing.html",
    ),
    "line": (
        "LINE",
        "Official LINE Messaging API sender with base64 HMAC-signed webhook verification.",
        "https://developers.line.biz/en/docs/messaging-api/",
    ),
    "google_chat": (
        "Google Chat",
        "Official Google Chat incoming-webhook sender with bearer-token verification; no inbound message API.",
        "https://developers.google.com/chat/how-tos/webhooks",
    ),
    "msteams": (
        "Microsoft Teams",
        "Official Teams incoming-webhook sender with signed-JWT ingress (app ID, public key and optional issuer); verification fails closed.",
        "https://learn.microsoft.com/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook",
    ),
    "dingtalk": (
        "DingTalk",
        "Official DingTalk custom-robot webhook sender with timestamped callback signature verification.",
        "https://open.dingtalk.com/document/robots/custom-robot-access",
    ),
    "feishu": (
        "Feishu / Lark",
        "Official Feishu open-platform bot with HMAC-SHA256-signed, AES-256-CBC encrypted event callbacks.",
        "https://open.feishu.cn/document/home/index",
    ),
    "qqbot": (
        "QQ Bot",
        "Community bridge over Tencent's QQ bot API with Ed25519-signed callbacks.",
        "https://bot.q.qq.com/wiki/",
    ),
    "wecom": (
        "WeCom (WeChat Work)",
        "Official WeCom bot with SHA1-signed, AES-encrypted JSON-mode callbacks.",
        "https://developer.work.weixin.qq.com/document/path/91770",
    ),
    "weixin": (
        "WeChat Official Account",
        "Community bridge for WeChat Official Accounts with SHA1-signed, AES-encrypted callbacks.",
        "https://developers.weixin.qq.com/doc/offiaccount/Getting_Started/Overview.html",
    ),
    "sms": (
        "SMS (Twilio)",
        "Official Twilio REST SMS sender with X-Twilio-Signature webhook verification.",
        "https://www.twilio.com/docs/messaging",
    ),
    "bluebubbles": (
        "BlueBubbles",
        "Community bridge to Apple Messages; webhook auth uses a LEO-convention signature header, not an Apple API.",
        "https://bluebubbles.app/",
    ),
    "signal": (
        "Signal",
        "Community signal-cli bridge; webhook auth uses a LEO-convention signature header, not an official Signal API.",
        "https://signal-cli.github.io/",
    ),
    "relay": (
        "Relay",
        "Generic signed-JSON channel for LEO-to-LEO or custom transports, with HMAC in both directions.",
        "",
    ),
    "email": (
        "Email",
        "SMTP sending plus IMAP polling for plain email; there is no signed webhook, so sender identity comes from the allowlist.",
        "RFC 5321 (SMTP) / RFC 3501 (IMAP)",
    ),
    "matrix": (
        "Matrix",
        "Official Matrix client-server /sync polling channel using a user access token from your homeserver.",
        "https://spec.matrix.org/latest/client-server-api/",
    ),
    "irc": (
        "IRC",
        "Minimal RFC 1459 IRC client with polling and verified TLS by default; cleartext needs an explicit dual opt-in.",
        "https://datatracker.ietf.org/doc/html/rfc1459",
    ),
    "ntfy": (
        "ntfy",
        "ntfy-compatible pub/sub channel using the JSON poll endpoint, with optional basic auth and a subscribed topic.",
        "https://ntfy.sh/docs/",
    ),
    "a2a": (
        "Agent2Agent (A2A)",
        "Send-only Agent2Agent JSON-RPC client; it declares neither webhook nor polling capability.",
        "https://google.github.io/A2A/",
    ),
}

# Extra optional environment variables seen in adapters but not declared in a
# class attribute. They are included with required=False so the form can expose
# them without blocking enablement.
_OPTIONAL_ENV: dict[str, tuple[str, ...]] = {
    "whatsapp": ("WHATSAPP_API_VERSION",),
    "email": ("SMTP_PORT", "IMAP_PORT"),
    "irc": ("IRC_CHANNELS", "IRC_PASSWORD", "IRC_TLS", "IRC_ALLOW_INSECURE"),
    "msteams": ("TEAMS_APP_ID", "TEAMS_JWT_ISSUER"),
    "ntfy": ("NTFY_TOPIC", "NTFY_USER", "NTFY_PASSWORD"),
    "signal": ("SIGNAL_CLI_USER", "SIGNAL_CLI_PASSWORD"),
    "sms": ("TWILIO_WEBHOOK_URL", "TWILIO_BASE_URL"),
    "a2a": ("A2A_API_KEY",),
}


def _dedupe(names) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _field_from_env(name: str) -> ChannelField:
    try:
        label, kind, help_text, required, options, default = _FIELD_META[name]
    except KeyError:
        raise KeyError(f"No field metadata for environment variable {name!r}; add it to _FIELD_META") from None
    return ChannelField(
        name=name,
        label=label,
        kind=kind,
        required=required,
        help=help_text,
        options=list(options),
        default=default,
    )


def _policy_fields() -> list[ChannelField]:
    return [
        ChannelField(
            name="allowlist",
            label="Allowed senders",
            kind=FieldKind.TEXTAREA,
            required=False,
            help=_ALLOWLIST_HELP,
        ),
        ChannelField(
            name="mode",
            label="Admission mode",
            kind=FieldKind.CHOICE,
            required=False,
            help=_MODE_HELP,
            options=["allowlist", "open", "deny"],
            default="allowlist",
        ),
    ]


def _build_spec(channel: str, cls) -> ChannelSpec:
    try:
        label, description, docs_hint = _CHANNEL_META[channel]
    except KeyError:
        raise KeyError(f"No channel metadata for {channel!r}; add it to _CHANNEL_META") from None

    declared = _dedupe(
        list(cls.credential_env)
        + list(cls.verification_env)
        + list(getattr(cls, "poll_env", ()) or ())
        + list(getattr(cls, "optional_env", ()) or ())
    )
    declared_set = set(declared)
    optional = [n for n in _dedupe(_OPTIONAL_ENV.get(channel, ())) if n not in declared_set]

    # Import lazily: specs.py is auto-discovered as a channel package module.
    from ..messaging import implements_polling, implements_webhook

    capabilities = ["send"]
    if implements_webhook(cls):
        capabilities.append("webhook")
    if implements_polling(cls):
        capabilities.append("poll")

    return ChannelSpec(
        channel=channel,
        label=label,
        description=description,
        fields=[_field_from_env(name) for name in declared + optional] + _policy_fields(),
        capabilities=capabilities,
        docs_hint=docs_hint,
    )


def channel_specs() -> dict[str, ChannelSpec]:
    """A spec for every registered channel, keyed by channel name.

    Imports the registry lazily so importing this module during package
    auto-discovery cannot create a circular import.
    """
    from . import ALL_ADAPTERS

    return {channel: _build_spec(channel, cls) for channel, cls in ALL_ADAPTERS.items()}


def spec_for(channel: str) -> ChannelSpec:
    """Return one channel's spec, raising a clear error for unknown channels."""
    specs = channel_specs()
    try:
        return specs[channel]
    except KeyError:
        known = ", ".join(sorted(specs))
        raise KeyError(f"Unknown channel {channel!r}; known channels: {known}") from None
