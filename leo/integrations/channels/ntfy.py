"""ntfy adapter: official self-hostable ntfy.sh HTTP publish/subscribe API.

Targets the **official ntfy HTTP API** as documented at docs.ntfy.sh — the
same API the public ntfy.sh service speaks, on a server the operator chooses
(``NTFY_URL`` points at ntfy.sh or any self-hosted instance). Publishing is
an unauthenticated-or-basic-auth POST with the message as the raw body;
subscribing uses the buffered ``poll=1`` form of the JSON stream so the
request terminates instead of holding an SSE connection open.

``NTFY_URL`` is read from the environment by *name* at call time, as are the
optional ``NTFY_USER``/``NTFY_PASSWORD`` (basic auth, only used when both
are present) and ``NTFY_TOPIC`` (the topic ``poll()`` subscribes to). No
credential value is ever stored on the adapter or echoed in errors. All
HTTP goes through the injectable ``httpx.AsyncClient`` so tests run offline.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from ..messaging import (
    MessagingAdapter,
    MessagingError,
    NormalizedMessage,
    from_epoch,
    ref_or_none,
)
from ..secrets import credential


def _valid_url(url: str) -> bool:
    """http(s) URL with a host, no embedded credentials, no traversal."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and ".." not in url
    )


def _valid_topic(topic: str) -> bool:
    """Topics are single path segments: no slashes, no whitespace."""
    return bool(topic) and "/" not in topic and not any(ch.isspace() for ch in topic)


def _basic_auth(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")


class NtfyAdapter(MessagingAdapter):
    """ntfy publisher plus JSON-line poll inbound.

    Honest status: official ntfy API, self-hostable server. Two limits:

    * ``poll()`` subscribes to a single ``NTFY_TOPIC``; multi-topic polling
      means one adapter instance per topic (each instance keeps its own
      ``since`` cursor).
    * attachment/media fields in poll events are ignored — only the textual
      ``message`` body is normalized.
    """

    channel = "ntfy"
    credential_env = ("NTFY_URL",)
    verification_env = ()
    supports_webhook = False
    supports_polling = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cursor: float | None = None  # last seen message time (unix seconds)

    # ------------------------------------------------------------------ helpers

    def _base_url(self, action: str) -> str:
        base = credential("NTFY_URL")
        if not base:
            raise MessagingError(f"ntfy {action} blocked: NTFY_URL is not set")
        if not _valid_url(base):
            raise MessagingError(f"ntfy {action} blocked: NTFY_URL is not a valid http(s) URL")
        return base.rstrip("/")

    def _auth_headers(self) -> dict[str, str]:
        user = credential("NTFY_USER")
        password = credential("NTFY_PASSWORD")
        if user and password:
            return {"Authorization": _basic_auth(user, password)}
        return {}

    # --------------------------------------------------------------------- send

    async def send(self, target: str, text: str) -> dict[str, Any]:
        base = self._base_url("send")
        topic = str(target)
        if not _valid_topic(topic):
            raise MessagingError("ntfy send blocked: invalid ntfy topic")
        url = f"{base}/{quote(topic, safe='')}"
        headers = {"Content-Type": "text/plain", **self._auth_headers()}
        try:
            response = await self._http().post(
                url, content=str(text).encode("utf-8"), headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            # Never echo httpx's message: it can contain the URL or password.
            raise MessagingError(f"ntfy send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"ntfy send failed: HTTP {response.status_code}")
        # ntfy answers with the event id in the X-Id header; absent on some
        # instances, so the id stays None there.
        message_id: str | None = None
        for key, value in response.headers.items():
            if key.lower() == "x-id":
                message_id = ref_or_none(value)
                break
        return {
            "channel": self.channel,
            "target": topic,
            "success": True,
            "message_id": message_id,
        }

    # --------------------------------------------------------------------- poll

    async def poll(self) -> list[NormalizedMessage]:
        base = self._base_url("poll")
        topic = credential("NTFY_TOPIC")
        if not topic:
            raise MessagingError("ntfy poll blocked: NTFY_TOPIC is not set")
        if not _valid_topic(topic):
            raise MessagingError("ntfy poll blocked: invalid ntfy topic")
        url = f"{base}/{quote(topic, safe='')}/json"
        params: dict[str, str] = {"poll": "1"}
        if self._cursor is not None:
            # ntfy accepts a unix timestamp: everything strictly newer than
            # the last message we normalized. Keeps polls cursor-stable.
            params["since"] = str(int(self._cursor))
        try:
            response = await self._http().get(
                url, params=params, headers=self._auth_headers(), timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"ntfy poll failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"ntfy poll failed: HTTP {response.status_code}")
        return self._parse_stream(response.text)

    def _parse_stream(self, body: str) -> list[NormalizedMessage]:
        """ntfy streams newline-delimited JSON events; only ``message`` events
        carry user content (``open``/``keepalive`` are transport chatter)."""
        messages: list[NormalizedMessage] = []
        max_time: float | None = self._cursor
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue  # untrusted stream: skip anything unparseable
            if not isinstance(item, dict) or item.get("event") != "message":
                continue
            text = item.get("message")
            if text is None:
                continue
            time_value = item.get("time")
            if isinstance(time_value, (int, float)) and not isinstance(time_value, bool):
                if max_time is None or time_value > max_time:
                    max_time = float(time_value)
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    # ntfy has no sender identity: the title is the closest
                    # thing to one, else the topic the message arrived on.
                    sender=str(item.get("title") or item.get("topic") or "unknown"),
                    text=str(text),
                    timestamp=from_epoch(time_value),
                    reply_ref=None,
                    message_id=ref_or_none(item.get("id")),
                )
            )
        self._cursor = max_time
        return messages


__all__ = ["NtfyAdapter"]
