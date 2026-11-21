"""Agent2Agent (A2A) protocol client — LEO's standards wedge.

Outbound channel speaking the published Agent2Agent protocol transport shape:
a JSON-RPC 2.0 request POSTed to the peer agent's HTTP endpoint with method
``message/send`` and a user message carrying a single text ``Part``.

``A2A_AGENT_URL`` must be the peer's *complete* JSON-RPC endpoint URL (the
``url`` field of its agent card) — the adapter never appends or invents a
path, so when the URL has no path component the POST goes to the URL exactly
as given; operators supply the full endpoint. An optional ``A2A_API_KEY`` is
read at call time and sent as a bearer token when set.

This adapter is a request/response *client*: it only sends. Inbound A2A server
support (receiving pushes from other agents) is out of scope, and A2A task
polling (``message/poll``-style long-running task checks) is future work —
the adapter declares neither webhook nor polling capability, and ``record()``
is inherited unchanged from the base class.

Peer responses are untrusted data: they are parsed defensively, never crash
the adapter, and any peer-provided string echoed into an error message is
hard-truncated to 200 characters so a hostile peer cannot flood logs or use
error text as an injection channel. No credential value (the API key) is
ever stored on the adapter, logged, or echoed in an error message.
"""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from ..messaging import MessagingAdapter, MessagingError, ref_or_none
from ..secrets import credential

# Peer-provided strings echoed into errors are hard-truncated to this length:
# peer data is data — never instructions, and never log-fodder.
_PEER_ECHO_LIMIT = 200


def _valid_http_url(url: str) -> bool:
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


def _clip(value: Any, limit: int = _PEER_ECHO_LIMIT) -> str:
    """Render untrusted peer data for an error message, hard-truncated."""
    text = value if isinstance(value, str) else str(value)
    if len(text) > limit:
        return text[:limit] + "…[truncated]"
    return text


def _error_detail(error: Any) -> str:
    """Truncated rendering of a JSON-RPC error object for an error message."""
    if isinstance(error, dict):
        fragments: list[str] = []
        code = error.get("code")
        if isinstance(code, int) and not isinstance(code, bool):
            fragments.append(f"code {code}")
        message = error.get("message")
        if message is not None:
            fragments.append(_clip(message))
        if not fragments:
            fragments.append(_clip(error))
        return f" ({'; '.join(fragments)})"
    return f" ({_clip(error)})"


def _task_id(result: Any) -> str | None:
    """Extract the task/message id from a JSON-RPC result, defensively."""
    if isinstance(result, str):
        return result or None
    if isinstance(result, dict):
        for key in ("id", "taskId"):
            value = result.get(key)
            if value is not None:
                return ref_or_none(value)
    return None


class A2AAdapter(MessagingAdapter):
    """Send text to another A2A-compatible agent over JSON-RPC 2.0.

    Request/response client only: no inbound webhook, no task polling (see
    the module docstring for scope and the documented endpoint semantics).
    """

    channel = "a2a"
    credential_env = ("A2A_AGENT_URL",)
    # No shared secret: authentication is the bearer key above, and this
    # adapter accepts no inbound traffic that would need verifying.
    verification_env = ()
    supports_webhook = False
    supports_polling = False

    async def send(self, target: str, text: str) -> dict[str, Any]:
        url = credential("A2A_AGENT_URL")
        if not url:
            raise MessagingError("a2a send blocked: A2A_AGENT_URL is not set")
        if not _valid_http_url(url):
            raise MessagingError("a2a send blocked: A2A_AGENT_URL is not a valid http(s) URL")
        headers: dict[str, str] = {}
        api_key = credential("A2A_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        rpc_request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": str(text)}],
                    "messageId": uuid.uuid4().hex,
                    "kind": "message",
                }
            },
        }
        try:
            response = await self._http().post(
                url, headers=headers, json=rpc_request, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            # Never echo the URL or httpx's exception text: both could carry
            # the bearer key or peer-controlled strings.
            raise MessagingError(f"a2a send failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise MessagingError(f"a2a send failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            data = None  # unparseable peer body: nothing to extract, never fatal
        message_id: str | None = None
        if isinstance(data, dict):
            error = data.get("error")
            if error is not None:
                raise MessagingError(f"a2a send failed: peer reported error{_error_detail(error)}")
            message_id = _task_id(data.get("result"))
        return {
            "channel": self.channel,
            "target": str(target),
            "success": True,
            "message_id": message_id,
        }
