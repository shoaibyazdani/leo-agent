"""Matrix adapter: official client-server API over an injectable client.

Targets the **official Matrix client-server API** (versioned ``/_matrix/client/v3``
endpoints as published in the Matrix specification). The homeserver named by
``MATRIX_HOMESERVER`` is self-hosted or any provider — the adapter speaks the
spec, not a vendor dialect. Live delivery is never implied by this module: all
HTTP goes through the injectable ``httpx.AsyncClient`` so tests run offline.

``MATRIX_HOMESERVER`` and ``MATRIX_ACCESS_TOKEN`` are read from the
environment by *name* at call time; the access token is never stored on the
adapter or echoed in errors.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
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

# Bounded replay guard: a /sync answer that re-delivers already-normalized
# event ids (server replay, limited-timeline backfill after a rejoin) is
# filtered instead of double-routing messages through the gateway.
_MAX_SEEN_EVENT_IDS = 1000


def _ms_to_iso(value: Any) -> str:
    """Matrix ``origin_server_ts`` is milliseconds since the Unix epoch."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return from_epoch(value / 1000.0)
    return from_epoch(value)


class MatrixAdapter(MessagingAdapter):
    """Matrix sender (room events) plus long-poll inbound via ``/sync``.

    Honest status: this is the official Matrix client-server API — no bridge,
    no vendor extension. Two limitations kept deliberately:

    * ``send`` accepts room ids (``!...``) and room aliases (``#...``). The
      spec requires a room id on the ``/send`` path; aliases are accepted
      syntactically and the homeserver's response is the source of truth.
    * the first ``poll()`` returns no messages. Without a ``since`` cursor a
      ``/sync`` call returns the initial state (the recent timeline as
      history), not "new" messages, so the first call only records
      ``next_batch`` and subsequent polls diff from that cursor.
    """

    channel = "matrix"
    credential_env = ("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN")
    verification_env = ()
    supports_webhook = False
    supports_polling = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cursor: str | None = None
        self._synced_once = False
        self._seen_event_ids: OrderedDict[str, None] = OrderedDict()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _valid_homeserver(url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )

    @staticmethod
    def _valid_room_id(room_id: str) -> bool:
        # Room ids start with '!' and aliases with '#'. A bare '#' would be
        # read as a URL fragment, so targets are percent-encoded on the way
        # into the path below.
        return room_id.startswith(("!", "#"))

    def _credentials(self, action: str) -> tuple[str, str]:
        homeserver = credential("MATRIX_HOMESERVER")
        if not homeserver:
            raise MessagingError(f"matrix {action} blocked: MATRIX_HOMESERVER is not set")
        token = credential("MATRIX_ACCESS_TOKEN")
        if not token:
            raise MessagingError(f"matrix {action} blocked: MATRIX_ACCESS_TOKEN is not set")
        if not self._valid_homeserver(homeserver):
            raise MessagingError(
                f"matrix {action} blocked: MATRIX_HOMESERVER is not a valid http(s) URL"
            )
        return homeserver, token

    # --------------------------------------------------------------------- send

    async def send(self, target: str, text: str) -> dict[str, Any]:
        homeserver, token = self._credentials("send")
        room_id = str(target)
        if not self._valid_room_id(room_id):
            raise MessagingError("matrix send blocked: invalid Matrix room id")
        # ':' and '!' are legal in a path segment; everything else is escaped
        # so an alias/room id can never smuggle path segments.
        room_path = quote(room_id, safe="!:")
        url = (
            f"{homeserver.rstrip('/')}/_matrix/client/v3/rooms/{room_path}"
            f"/send/m.room.message/{uuid.uuid4().hex}"
        )
        try:
            response = await self._http().put(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"msgtype": "m.text", "body": str(text)},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # Never echo httpx's message: it can contain the URL or token.
            raise MessagingError(f"matrix send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"matrix send failed: HTTP {response.status_code}")
        message_id: str | None = None
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            message_id = ref_or_none(data.get("event_id"))
        return {
            "channel": self.channel,
            "target": room_id,
            "success": True,
            "message_id": message_id,
        }

    # --------------------------------------------------------------------- poll

    async def poll(self) -> list[NormalizedMessage]:
        homeserver, token = self._credentials("poll")
        url = f"{homeserver.rstrip('/')}/_matrix/client/v3/sync"
        params: dict[str, str] = {"timeout": "0"}
        if self._cursor:
            params["since"] = self._cursor
        try:
            response = await self._http().get(
                url, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise MessagingError(f"matrix poll failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"matrix poll failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            return []
        next_batch = data.get("next_batch")
        if isinstance(next_batch, str) and next_batch:
            self._cursor = next_batch
        if not self._synced_once:
            # First /sync returns the initial state + recent timeline as
            # *history*, not new messages; adopting next_batch as the cursor
            # means the next poll only sees genuinely new events.
            self._synced_once = True
            return []
        return self._timeline_messages(data)

    def _timeline_messages(self, data: dict[str, Any]) -> list[NormalizedMessage]:
        rooms = data.get("rooms")
        if not isinstance(rooms, dict):
            return []
        join = rooms.get("join")
        if not isinstance(join, dict):
            return []
        messages: list[NormalizedMessage] = []
        for room in join.values():
            if not isinstance(room, dict):
                continue
            timeline = room.get("timeline")
            if not isinstance(timeline, dict):
                continue
            events = timeline.get("events")
            if not isinstance(events, list):
                continue
            for event in events:
                message = self._normalize_event(event)
                if message is not None:
                    messages.append(message)
        return messages

    def _normalize_event(self, event: Any) -> NormalizedMessage | None:
        if not isinstance(event, dict) or event.get("type") != "m.room.message":
            return None
        content = event.get("content")
        if not isinstance(content, dict):
            return None
        body = content.get("body")
        if not isinstance(body, str):
            return None
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            return None  # without an id the replay guard could not work
        if event_id in self._seen_event_ids:
            return None
        self._seen_event_ids[event_id] = None
        while len(self._seen_event_ids) > _MAX_SEEN_EVENT_IDS:
            self._seen_event_ids.popitem(last=False)
        reply_ref = None
        relates_to = content.get("m.relates_to")
        if isinstance(relates_to, dict):
            in_reply_to = relates_to.get("m.in_reply_to")
            if isinstance(in_reply_to, dict):
                reply_ref = ref_or_none(in_reply_to.get("event_id"))
        return NormalizedMessage(
            channel=self.channel,
            sender=str(event.get("sender") or "unknown"),
            text=body,
            timestamp=_ms_to_iso(event.get("origin_server_ts")),
            reply_ref=reply_ref,
            message_id=event_id,
        )


__all__ = ["MatrixAdapter"]
