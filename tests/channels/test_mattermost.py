"""Offline Mattermost adapter tests: shared-token verification, outgoing-webhook
normalization, incoming-webhook / REST send, record.

No live network access: HTTP goes through ``httpx.MockTransport`` and every
token value is a fake string whose absence from error messages and record
dumps is explicitly asserted.
"""

from __future__ import annotations

import json
import urllib.parse

import httpx
import pytest

from leo.integrations.channels.mattermost import MattermostAdapter
from leo.integrations.messaging import MessagingError, NormalizedMessage, WebhookVerificationError

OUTGOING_TOKEN = "mattermost-outgoing-token"
FAKE_TOKEN = "fake-mattermost-token-should-never-leak"
WEBHOOK_URL = "https://mm.example.com/hooks/abc123"

FORM = {
    "token": OUTGOING_TOKEN,
    "user_name": "alice",
    "text": "deploy the site",
    "timestamp": "1700000000",
    "post_id": "post-1",
    "channel_name": "town-square",
}


def _raw(form: dict[str, str]) -> bytes:
    return urllib.parse.urlencode(form).encode("utf-8")


# ---------------------------------------------------------------------- verify


def test_mattermost_verify_compares_shared_form_token(monkeypatch):
    monkeypatch.setenv("MATTERMOST_OUTGOING_TOKEN", OUTGOING_TOKEN)
    adapter = MattermostAdapter()
    assert adapter.verify_webhook(FORM, raw_body=_raw(FORM)) is True
    # Already-parsed dict payload path when no raw body is available.
    assert adapter.verify_webhook(FORM) is True
    assert adapter.verify_webhook({"token": OUTGOING_TOKEN}) is True
    wrong = {**FORM, "token": "someone-elses-token"}
    assert adapter.verify_webhook(wrong, raw_body=_raw(wrong)) is False
    stripped = {key: value for key, value in FORM.items() if key != "token"}
    assert adapter.verify_webhook(stripped, raw_body=_raw(stripped)) is False
    assert adapter.verify_webhook({"token": "nope"}) is False


def test_mattermost_verify_fails_closed_without_token_env(monkeypatch):
    monkeypatch.delenv("MATTERMOST_OUTGOING_TOKEN", raising=False)
    adapter = MattermostAdapter()  # require_verification=True by default
    assert adapter.verify_webhook(FORM, raw_body=_raw(FORM)) is False
    assert adapter.verify_webhook(FORM) is False


# --------------------------------------------------------------- handle webhook


async def test_mattermost_handle_webhook_normalizes_form_fields(monkeypatch):
    monkeypatch.setenv("MATTERMOST_OUTGOING_TOKEN", OUTGOING_TOKEN)
    adapter = MattermostAdapter()
    messages = await adapter.handle_webhook(FORM, None, raw_body=_raw(FORM))
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, NormalizedMessage)
    assert (message.channel, message.sender, message.text) == ("mattermost", "alice", "deploy the site")
    assert message.timestamp.startswith("2023-11-14")
    assert message.reply_ref is None
    assert message.message_id == "post-1"
    # Already-parsed dict payload (route parsed the form) normalizes identically.
    again = await adapter.handle_webhook(FORM)
    assert len(again) == 1 and again[0].message_id == "post-1"


async def test_mattermost_handle_webhook_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("MATTERMOST_OUTGOING_TOKEN", OUTGOING_TOKEN)
    adapter = MattermostAdapter()
    wrong = {**FORM, "token": "imposter-token"}
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(wrong, None, raw_body=_raw(wrong))
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(FORM, None, raw_body=b"token=&user_name=x&text=hi")
    with pytest.raises(WebhookVerificationError):
        await adapter.handle_webhook(FORM, None, raw_body=b"")


async def test_mattermost_handle_webhook_skips_empty_text_and_tolerates_garbage(monkeypatch):
    monkeypatch.setenv("MATTERMOST_OUTGOING_TOKEN", OUTGOING_TOKEN)
    adapter = MattermostAdapter()
    for payload in (
        {key: value for key, value in FORM.items() if key != "text"},
        {**FORM, "text": ""},
    ):
        assert await adapter.handle_webhook(payload, None, raw_body=_raw(payload)) == []
    raw_no_text = urllib.parse.urlencode(
        {key: value for key, value in FORM.items() if key != "text"}
    ).encode("utf-8")
    assert await adapter.handle_webhook(raw_no_text, None, raw_body=raw_no_text) == []
    # Verified but hostile field values never crash the parser.
    hostile = {**FORM, "user_name": None, "timestamp": "not-a-number", "post_id": {"weird": True}}
    messages = await adapter.handle_webhook(hostile)  # dict payload path
    assert len(messages) == 1
    assert messages[0].sender == "unknown"  # str(None or "unknown")
    assert messages[0].timestamp  # from_epoch fell back to "now" for garbage
    assert messages[0].message_id == str({"weird": True})
    weird_raw = f"token={OUTGOING_TOKEN}&text=hi&timestamp=garbage&user_name=".encode("utf-8")
    messages = await adapter.handle_webhook(weird_raw, None, raw_body=weird_raw)
    assert len(messages) == 1
    assert messages[0].sender == "unknown" and messages[0].message_id is None


# ------------------------------------------------------------------------ send


async def test_mattermost_send_via_incoming_webhook_url():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MattermostAdapter(client=client).send(WEBHOOK_URL, "hello")
    assert result == {
        "channel": "mattermost",
        "target": WEBHOOK_URL,
        "success": True,
        "message_id": None,  # incoming webhooks answer plain-text "ok", no post id
    }
    assert str(requests[0].url) == WEBHOOK_URL
    assert json.loads(requests[0].content) == {"text": "hello"}


async def test_mattermost_send_via_api_v4_posts(monkeypatch):
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    monkeypatch.setenv("MATTERMOST_TOKEN", FAKE_TOKEN)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"id": "post-9", "channel_id": "chan-1", "message": "hello"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = MattermostAdapter(client=client)
        result = await adapter.send("chan-1", "hello")
        assert result == {
            "channel": "mattermost",
            "target": "chan-1",
            "success": True,
            "message_id": "post-9",
        }
        # A trailing slash in the configured base URL must not change the endpoint.
        monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com/")
        result = await adapter.send("chan-1", "again")
        assert result["success"] is True and result["message_id"] == "post-9"
    assert str(requests[0].url) == "https://mm.example.com/api/v4/posts"
    assert str(requests[1].url) == "https://mm.example.com/api/v4/posts"
    # Exact header assertion: any token leakage would be visible here on failure.
    assert requests[0].headers["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert requests[1].headers["authorization"] == f"Bearer {FAKE_TOKEN}"
    assert json.loads(requests[0].content) == {"channel_id": "chan-1", "message": "hello"}


async def test_mattermost_send_rejects_non_webhook_urls():
    adapter = MattermostAdapter()  # validation happens before any network access
    for target in (
        "https://evil.example.com/hook",  # not a /hooks/ path
        "https://mm.example.com/api/v4/posts",  # the REST API is not a webhook URL
        "https://mm.example.com",  # no path at all
        "https://[::invalid",  # unparsable
    ):
        with pytest.raises(MessagingError, match="not a Mattermost webhook URL"):
            await adapter.send(target, "hi")


async def test_mattermost_send_blocked_without_credentials(monkeypatch):
    monkeypatch.delenv("MATTERMOST_TOKEN", raising=False)
    monkeypatch.delenv("MATTERMOST_URL", raising=False)
    with pytest.raises(MessagingError, match="MATTERMOST_TOKEN is not set") as exc:
        await MattermostAdapter().send("chan-1", "hi")
    assert FAKE_TOKEN not in str(exc.value)
    monkeypatch.setenv("MATTERMOST_TOKEN", FAKE_TOKEN)
    with pytest.raises(MessagingError, match="MATTERMOST_URL is not set") as exc:
        await MattermostAdapter().send("chan-1", "hi")
    assert FAKE_TOKEN not in str(exc.value)


async def test_mattermost_send_failures_hide_token(monkeypatch):
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    monkeypatch.setenv("MATTERMOST_TOKEN", FAKE_TOKEN)

    def http_400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"invalid_or_missing_token {FAKE_TOKEN}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(http_400)) as client:
        with pytest.raises(MessagingError, match="mattermost send failed: HTTP 400") as exc:
            await MattermostAdapter(client=client).send("chan-1", "hi")
    assert FAKE_TOKEN not in str(exc.value)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach mm.example.com with {FAKE_TOKEN}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
        with pytest.raises(MessagingError, match="mattermost send failed: ConnectError") as exc:
            await MattermostAdapter(client=client).send("chan-1", "hi")
    assert FAKE_TOKEN not in str(exc.value)


# ---------------------------------------------------------------------- record


def test_mattermost_record_blocked_partial_and_configured(monkeypatch):
    for name in ("MATTERMOST_TOKEN", "MATTERMOST_URL", "MATTERMOST_OUTGOING_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    record = MattermostAdapter().record()
    assert (record.name, record.kind) == ("mattermost", "messaging")
    assert record.status == "blocked"
    assert record.capabilities == ["send", "webhook"]
    assert record.verification == "mock-tested"
    assert record.config_ref == "MATTERMOST_TOKEN,MATTERMOST_URL,MATTERMOST_OUTGOING_TOKEN"
    assert record.metadata["credential_env"] == ["MATTERMOST_TOKEN", "MATTERMOST_URL"]
    assert record.metadata["verification_env"] == ["MATTERMOST_OUTGOING_TOKEN"]
    assert record.metadata["live_verified"] is False
    assert FAKE_TOKEN not in json.dumps(record.model_dump())

    # A lone token enables nothing: both credentials are required to post.
    monkeypatch.setenv("MATTERMOST_TOKEN", FAKE_TOKEN)
    partial = MattermostAdapter().record()
    assert partial.status == "blocked"
    assert partial.metadata["credentials_present"] == ["MATTERMOST_TOKEN"]
    assert FAKE_TOKEN not in json.dumps(partial.model_dump())

    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    configured = MattermostAdapter().record()
    assert configured.status == "configured"
    assert "MATTERMOST_URL" in configured.config_ref
    assert FAKE_TOKEN not in json.dumps(configured.model_dump())
