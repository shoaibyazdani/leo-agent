"""Outbound media: whitelist, size guard, multipart method choice, digest binding.

Everything here is offline. Telegram goes through ``httpx.MockTransport`` so no
bytes reach the API, and the gateway tests use an in-memory adapter plus a
recording permission stub over the isolated test database from conftest.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from leo.config import ChannelsConfig
from leo.contracts import Decision, RiskLevel
from leo.events import EventBus
from leo.integrations.admission import ChannelAdmission
from leo.integrations.gateway import Gateway
from leo.integrations.messaging import (
    TELEGRAM_MAX_MEDIA_BYTES,
    MessagingAdapter,
    MessagingError,
    TelegramAdapter,
    resolve_outbound_media,
)

TOKEN = "media-token-must-never-leak"


@pytest.fixture
def outbox(tmp_path: Path) -> Path:
    """A private whitelist root, never the real .leo/media/outbound."""
    root = tmp_path / "outbound"
    root.mkdir()
    return root


def make_file(root: Path, name: str, data: bytes = b"\x89PNG\r\n\x1a\n-fake") -> Path:
    path = root / name
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------- whitelist


def test_path_outside_the_whitelist_is_refused(tmp_path: Path, outbox: Path):
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")

    with pytest.raises(MessagingError) as exc:
        resolve_outbound_media(secret, outbox)

    assert "not under" in str(exc.value)
    # The refused path must not be echoed into a log line or a chat message.
    assert str(secret) not in str(exc.value)


def test_relative_escape_by_parent_traversal_is_refused(tmp_path: Path, outbox: Path):
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")

    with pytest.raises(MessagingError):
        resolve_outbound_media(outbox / ".." / "secret.txt", outbox)


def test_symlink_inside_the_whitelist_cannot_escape_it(tmp_path: Path, outbox: Path):
    secret = tmp_path / "outside.txt"
    secret.write_text("outside the whitelist")
    link = outbox / "innocent.png"
    link.symlink_to(secret)

    with pytest.raises(MessagingError) as exc:
        resolve_outbound_media(link, outbox)

    assert "not under" in str(exc.value)


def test_missing_empty_and_directory_targets_are_refused(outbox: Path):
    with pytest.raises(MessagingError, match="does not exist"):
        resolve_outbound_media(outbox / "nope.png", outbox)
    with pytest.raises(MessagingError, match="empty"):
        resolve_outbound_media(make_file(outbox, "zero.png", b""), outbox)
    with pytest.raises(MessagingError, match="not a regular file"):
        resolve_outbound_media(outbox, outbox)


def test_a_file_inside_the_whitelist_resolves_to_its_real_path(outbox: Path):
    path = make_file(outbox, "chart.png")

    assert resolve_outbound_media(path, outbox) == path.resolve()


# ------------------------------------------------------------------- sendPhoto


def telegram_with(handler):
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
    return TelegramAdapter(client=client), requests


async def test_image_is_sent_as_sendphoto_multipart(monkeypatch, outbox: Path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    body_bytes = b"\x89PNG\r\n\x1a\npayload"
    path = make_file(outbox, "chart.png", body_bytes)
    adapter, requests = telegram_with(
        lambda request: httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})
    )

    async with adapter:
        result = await adapter.send_media("525083981", path, caption="**hi**", root=outbox)

    assert result["success"] is True
    assert result["message_id"] == "7"
    assert result["media_method"] == "sendPhoto"
    assert result["media"] == "chart.png"
    assert result["media_bytes"] == len(body_bytes)

    request = requests[0]
    assert str(request.url).endswith("/sendPhoto")
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content
    assert b'name="photo"; filename="chart.png"' in body
    assert body_bytes in body
    assert b'name="chat_id"' in body and b"525083981" in body
    assert b'name="caption"' in body and b"**hi**" in body


async def test_non_image_goes_as_document_to_preserve_the_bytes(monkeypatch, outbox: Path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    path = make_file(outbox, "report.csv", b"a,b\n1,2\n")
    adapter, requests = telegram_with(
        lambda request: httpx.Response(200, json={"ok": True, "result": {"message_id": 8}})
    )

    async with adapter:
        result = await adapter.send_media("42", path, root=outbox)

    assert result["media_method"] == "sendDocument"
    assert str(requests[0].url).endswith("/sendDocument")
    assert b'name="document"; filename="report.csv"' in requests[0].content
    # No caption was requested, so no caption field is sent.
    assert b'name="caption"' not in requests[0].content


async def test_oversized_file_is_refused_before_any_upload(monkeypatch, outbox: Path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    path = outbox / "huge.bin"
    with path.open("wb") as handle:
        handle.truncate(TELEGRAM_MAX_MEDIA_BYTES + 1)  # sparse, not 50 MB of RAM

    adapter, requests = telegram_with(
        lambda request: httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})
    )

    async with adapter:
        with pytest.raises(MessagingError, match="exceeds"):
            await adapter.send_media("42", path, root=outbox)

    assert requests == []


async def test_api_failure_raises_and_never_leaks_the_token(monkeypatch, outbox: Path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    path = make_file(outbox, "chart.png")
    adapter, _ = telegram_with(lambda request: httpx.Response(413, json={"ok": False}))

    async with adapter:
        with pytest.raises(MessagingError) as exc:
            await adapter.send_media("42", path, root=outbox)

    assert "HTTP 413" in str(exc.value)
    assert TOKEN not in str(exc.value)
    assert "api.telegram.org" not in str(exc.value)


async def test_missing_credentials_block_media_without_http(monkeypatch, outbox: Path):
    # The real environment on this box has a token; clear it so the test proves
    # the guard rather than accidentally borrowing live credentials.
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    adapter, requests = telegram_with(
        lambda request: httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
    )

    async with adapter:
        with pytest.raises(MessagingError, match="TELEGRAM_BOT_TOKEN"):
            await adapter.send_media("42", make_file(outbox, "chart.png"), root=outbox)

    assert requests == []


async def test_channels_without_media_support_refuse_instead_of_dropping_the_file():
    class TextOnly(MessagingAdapter):
        channel = "textonly"
        credential_env = ()

    with pytest.raises(MessagingError, match="does not support outbound media"):
        await TextOnly().send_media("42", "whatever.png")


# ------------------------------------------------------------------- gateway


class MediaAdapter(MessagingAdapter):
    """Offline adapter that records text and media sends separately."""

    channel = "telegram"
    credential_env = ()
    supports_webhook = False

    def __init__(self) -> None:
        super().__init__()
        self.text_calls: list[tuple[str, str]] = []
        self.media_calls: list[tuple[str, str, str]] = []

    async def send(self, target: str, text: str) -> dict:
        self.text_calls.append((target, text))
        return {"channel": self.channel, "success": True, "message_id": "text-1"}

    async def send_media(self, target, media, *, caption="", root=None) -> dict:
        self.media_calls.append((target, str(media), caption))
        return {
            "channel": self.channel,
            "success": True,
            "message_id": "media-1",
            "media_bytes": Path(str(media)).stat().st_size,
        }


class RecordingPermission:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def authorize(self, actor, action, resource=None, risk=RiskLevel.LOW, **context):
        self.calls.append({"action": action, "resource": resource, "risk": risk, **context})
        return Decision(allowed=True, reason="test", risk=risk)

    async def resolve_approval(self, approval_id, approved, resolver="user"):
        return Decision(allowed=approved, reason="test", risk=RiskLevel.HIGH)


def build_gateway(
    adapter: MediaAdapter,
    permission: RecordingPermission,
    root: Path | None = None,
) -> Gateway:
    return Gateway(
        {"telegram": adapter},
        events=EventBus(persist=False),
        permissions=permission,
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
        outbound_media_root=root,
    )


async def test_gateway_digest_binds_the_resolved_media_path(outbox: Path):
    adapter, permission = MediaAdapter(), RecordingPermission()
    gateway = build_gateway(adapter, permission, outbox)
    path = make_file(outbox, "chart.png", b"abcdef")

    result = await gateway.send(
        "telegram", "chat-1", "caption text", media=str(path)
    )

    assert result["success"] is True
    assert result["message_id"] == "media-1"
    assert adapter.text_calls == []          # the media branch really ran
    assert adapter.media_calls == [("chat-1", str(path.resolve()), "caption text")]

    authorized = permission.calls[-1]
    assert authorized["risk"] is RiskLevel.MEDIUM
    # The digest must cover the file, not just the text that was typed.
    assert authorized["arguments"]["media"] == str(path.resolve())
    assert authorized["arguments"]["media_bytes"] == 6
    assert authorized["arguments"]["caption"] == "caption text"
    assert authorized["arguments_digest"]


async def test_a_different_file_produces_a_different_digest(outbox: Path):
    adapter, permission = MediaAdapter(), RecordingPermission()
    gateway = build_gateway(adapter, permission, outbox)
    first = make_file(outbox, "one.png", b"first")
    second = make_file(outbox, "two.png", b"second")

    await gateway.send("telegram", "chat-1", "hi", media=str(first))
    await gateway.send("telegram", "chat-1", "hi", media=str(second))

    digests = [call["arguments_digest"] for call in permission.calls]
    assert digests[0] != digests[1], "swapping the attachment must change the approval arguments"


async def test_text_only_send_keeps_its_original_arguments():
    adapter, permission = MediaAdapter(), RecordingPermission()
    gateway = build_gateway(adapter, permission, outbox)

    await gateway.send("telegram", "chat-1", "plain hello")

    assert adapter.text_calls == [("chat-1", "plain hello")]
    assert adapter.media_calls == []
    assert set(permission.calls[-1]["arguments"]) == {"channel", "target", "text", "subject"}


async def test_gateway_refuses_media_outside_the_whitelist(tmp_path: Path, outbox: Path):
    adapter, permission = MediaAdapter(), RecordingPermission()
    gateway = build_gateway(adapter, permission, outbox)
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")

    with pytest.raises(MessagingError, match="not under"):
        await gateway.send(
            "telegram", "chat-1", "exfil", media=str(secret)
        )

    # Refused before authorization and before any adapter call.
    assert adapter.media_calls == []
    assert adapter.text_calls == []
    assert permission.calls == []


async def test_media_send_without_a_supported_adapter_is_refused(outbox: Path):
    class NoMedia(MessagingAdapter):
        channel = "telegram"
        credential_env = ()

        async def send(self, target: str, text: str) -> dict:
            return {"channel": self.channel, "success": True, "message_id": "x"}

    adapter = NoMedia()
    adapter.send_media = None  # attribute present but not callable
    gateway = Gateway(
        {"telegram": adapter},
        events=EventBus(persist=False),
        permissions=RecordingPermission(),
        admission=ChannelAdmission(ChannelsConfig(default_mode="open")),
        outbound_media_root=outbox,
    )

    with pytest.raises(MessagingError, match="does not support outbound media"):
        await gateway.send(
            "telegram", "chat-1", "hi",
            media=str(make_file(outbox, "chart.png")),
        )

# ------------------------------------------------- the whitelist root is not caller-chosen


async def test_send_does_not_accept_a_root_override():
    """A tool call must not be able to widen the attachment whitelist."""
    import inspect

    params = inspect.signature(Gateway.send).parameters
    assert "media_root" not in params, "media_root is config-owned, not a per-send argument"
    assert "outbound_media_root" in inspect.signature(Gateway.__init__).parameters


async def test_a_file_outside_the_gateway_root_is_refused(outbox: Path, tmp_path: Path):
    """The gateway root decides, not the caller: a sibling directory is out."""
    outside = tmp_path / "outside"
    outside.mkdir()
    stray = outside / "chart.png"
    stray.write_bytes(b"nope")
    adapter, permission = MediaAdapter(), RecordingPermission()
    gateway = build_gateway(adapter, permission, outbox)

    with pytest.raises(MessagingError) as exc:
        await gateway.send("telegram", "chat-1", "exfil", media=str(stray))

    assert "not under" in str(exc.value)
    assert adapter.media_calls == []
    assert permission.calls == []


def test_the_tool_schema_does_not_expose_a_root_override():
    from leo.tools import integration_tools
    from leo.tools.registry import ToolRegistry

    registry = ToolRegistry()
    integration_tools.register(registry)
    spec = registry.get("channel_send").spec
    props = spec.input_schema["properties"]
    assert "media" in props and "caption" in props
    assert "media_root" not in props
    assert spec.input_schema["additionalProperties"] is False


def test_settings_default_the_outbound_media_root_inside_the_workspace():
    from leo.config import Settings

    settings = Settings()
    resolved = settings.resolve(settings.paths.outbound_media)
    assert resolved == settings.root / ".leo/media/outbound"


# ------------------------------------------------- bare names and cwd independence


def test_a_bare_name_resolves_inside_the_whitelist(outbox: Path, monkeypatch: pytest.MonkeyPatch):
    """An agent that just wrote a file should not have to know the worker's cwd."""
    make_file(outbox, "chart.png")
    elsewhere = outbox.parent / "worker-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = resolve_outbound_media("chart.png", outbox)

    assert resolved == (outbox / "chart.png").resolve()


def test_a_bare_name_never_escapes_the_whitelist(outbox: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The convenience of a bare name must not become a way around containment."""
    (tmp_path / "secret.png").write_bytes(b"secret")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(MessagingError) as exc:
        resolve_outbound_media("secret.png", outbox)

    assert "not under" in str(exc.value)


def test_the_relative_path_from_the_whitelist_parent_still_works(outbox: Path, monkeypatch: pytest.MonkeyPatch):
    make_file(outbox, "chart.png")
    monkeypatch.chdir(outbox.parent)

    resolved = resolve_outbound_media("outbound/chart.png", outbox)

    assert resolved == (outbox / "chart.png").resolve()
