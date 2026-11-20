"""Executor-level coverage for the live HTTP JSON tools.

The HTTP side is a *real* loopback server, never a fake client. The shared
egress policy normally rejects loopback addresses, so the tests patch only the
two policy seams in ``leo.tools.builtins`` (``resolve_host`` and
``is_public_ip``) to admit exactly 127.0.0.1 for the mock server. Every other
address still goes through the production policy, which is what the SSRF tests
below assert. The actual GET is still performed by ``builtins.http_get``, and a
thin client spy records the pinned URL/Host/SNI without changing behavior.
"""

from __future__ import annotations

import gzip
import ipaddress
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest

from leo import db
from leo.config import Settings
from leo.contracts import ChatMessage, RiskLevel, ToolCall
from leo.events import EventBus
from leo.models.context import ContextOptimizer
from leo.security.permissions import PermissionManagerImpl
from leo.tools import (
    MODEL_FORBIDDEN_TOOLS,
    ToolExecutorImpl,
    ToolRegistry,
    builtins,
    http_tools,
    register_all_toolsets,
)
from leo.tools.builtins import is_public_ip as REAL_IS_PUBLIC

NAMES = ["http_json", "http_json_paths"]
SOURCE = "fetched-http"
REAL_ASYNC_CLIENT = httpx.AsyncClient

# Invisible Unicode from every relevant general category: Cf (format/bidi),
# Mn (combining marks), Cc (controls). A fixed five-codepoint list missed most
# of these; the module must use the category rule instead.
INVISIBLE_CODEPOINTS = [
    0x00AD,  # SOFT HYPHEN (Cf)
    0x200B,  # ZERO WIDTH SPACE (Cf)
    0x200C,  # ZERO WIDTH NON-JOINER (Cf)
    0x200D,  # ZERO WIDTH JOINER (Cf)
    0x200E,  # LEFT-TO-RIGHT MARK (Cf)
    0x202A,  # LEFT-TO-RIGHT EMBEDDING (Cf)
    0x202D,  # LEFT-TO-RIGHT OVERRIDE (Cf)
    0x202E,  # RIGHT-TO-LEFT OVERRIDE (Cf)
    0x2066,  # LEFT-TO-RIGHT ISOLATE (Cf)
    0x2068,  # FIRST STRONG ISOLATE (Cf)
    0x2060,  # WORD JOINER (Cf)
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE (Cf)
    0x034F,  # COMBINING GRAPHEME JOINER (Mn)
    0xFE0F,  # VARIATION SELECTOR-16 (Mn)
    0x0301,  # COMBINING ACUTE ACCENT (Mn)
    0x001F,  # UNIT SEPARATOR (Cc)
]

RATES_DOCUMENT = {
    "provider": "mock-exchange",
    "source": "cached",  # deliberately misleading: output.source is our own marker
    "rates": {"PKR": 276.741572, "USD": 1.0, "EUR": 0.92},
    "history": [{"value": 276.741572}, {"value": 277.125}],
    "meta": {"unit": "PKR per USD"},
}


class _ServerState:
    def __init__(self) -> None:
        self.requests: list[dict[str, str]] = []
        self.encodings: dict[str, str] = {}
        self.routes: dict[str, tuple[int, str, bytes]] = {
            "/rates": (200, "application/json", json.dumps(RATES_DOCUMENT).encode()),
            "/big": (
                200,
                "application/json",
                json.dumps({"filler": "x" * 10000, "rates": {"PKR": 276.741572}}).encode(),
            ),
            "/bigvalue": (
                200,
                "application/json",
                json.dumps({"blob": "x" * 3000, "rates": {"PKR": 276.741572}}).encode(),
            ),
            "/bad": (200, "application/json", b'{"rates": '),
            "/nan": (200, "application/json", b'{"rates": {"PKR": NaN}}'),
            "/error": (500, "application/json", b'{"rates": {"PKR": 999.0}}'),
            "/redirect": (
                302,
                "application/json",
                b'{"rates": {"PKR": 999.0}}',
            ),
            "/html": (200, "text/html", b'{"rates": {"PKR": 999.0}}'),
            "/scalar": (200, "application/json", b"42"),
            "/cjk": (200, "application/json", json.dumps({"text": "值" * 40}).encode()),
            "/cjkbig": (200, "application/json", json.dumps({"text": "值" * 1400}).encode()),
            "/secret": (
                200,
                "application/json",
                json.dumps({"note": "sk-abcdefghijklmnop123456"}).encode(),
            ),
            "/keysecret": (
                200,
                "application/json",
                json.dumps({"note": "key=supersecretvalue123"}).encode(),
            ),
            "/paths": (
                200,
                "application/json",
                json.dumps({"ok": 1, "a.b": 2, "": 3, "L" * 250: 4, "z": 5}).encode(),
            ),
            "/secretpaths": (
                200,
                "application/json",
                json.dumps({"safe": 1, "sk-abcdefghijklmnop123456": 2}).encode(),
            ),
        }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        state: _ServerState = self.server.state  # type: ignore[attr-defined]
        path = urlsplit(self.path).path
        state.requests.append({
            "path": path,
            "host": self.headers.get("Host", ""),
            "cache_control": self.headers.get("Cache-Control", ""),
        })
        status, content_type, body = state.routes.get(
            path, (404, "application/json", b'{"error":"not found"}')
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        encoding = state.encodings.get(path)
        if encoding:
            self.send_header("Content-Encoding", encoding)
        # Deliberately cache-ish response: fetched != fresh, and the tool must
        # surface the evidence rather than claiming real-time data.
        self.send_header("Cache-Control", "no-cache, max-age=60")
        self.send_header("Date", "Wed, 20 Nov 2026 00:00:00 GMT")
        self.send_header("Age", "3")
        self.send_header("ETag", '"v1"')
        if status in (301, 302, 307, 308):
            self.send_header("Location", "/rates")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass  # no server chatter in test output


@pytest.fixture(scope="module")
def mock_server():
    state = _ServerState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _ClientSpy:
    """Delegating client: same behavior, plus a record of the stream() call."""

    calls: list[dict[str, object]] = []
    instances: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self._client = REAL_ASYNC_CLIENT(**kwargs)  # type: ignore[arg-type]
        _ClientSpy.instances.append(dict(kwargs))

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._client.__aexit__(*exc)

    def stream(self, method, url, headers=None, extensions=None):
        _ClientSpy.calls.append(
            {"method": method, "url": url, "headers": headers, "extensions": extensions}
        )
        return self._client.stream(method, url, headers=headers, extensions=extensions)


def _loopback_only_policy(address):
    """Admit the mock server's address; every other address uses real policy."""
    if address == ipaddress.ip_address("127.0.0.1"):
        return True
    return REAL_IS_PUBLIC(address)


@pytest.fixture
async def env(tmp_path, monkeypatch, mock_server):
    state, port = mock_server
    state.requests.clear()
    _ClientSpy.calls.clear()
    _ClientSpy.instances.clear()

    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = http_tools.register(registry, services={})
    executor = ToolExecutorImpl(registry, permissions, events, settings)

    # Test-only policy bridge: the host name resolves to the loopback mock and
    # 127.0.0.1 is admitted. No other address is exempted.
    monkeypatch.setattr(
        builtins,
        "resolve_host",
        lambda host, p: [(2, 1, 6, "", ("127.0.0.1", p or port))],
    )
    monkeypatch.setattr(builtins, "is_public_ip", _loopback_only_policy)
    monkeypatch.setattr(builtins.httpx, "AsyncClient", _ClientSpy)  # type: ignore[misc]

    return SimpleNamespace(
        tmp_path=tmp_path,
        state=state,
        port=port,
        settings=settings,
        events=events,
        permissions=permissions,
        registry=registry,
        executor=executor,
        names=names,
    )


def _url(env, path: str = "/rates") -> str:
    return f"http://rates.test:{env.port}{path}"


async def _run(env, name: str, arguments: dict | None = None, **kwargs):
    call = ToolCall(
        name=name, arguments=arguments or {}, actor="http-tester", **kwargs
    )
    return await env.executor.execute(call)


def _assert_error(result, fragment: str) -> None:
    assert result.success is False
    assert result.output is None
    assert result.error and fragment in result.error, result.error
    assert result.decision in ("error", "denied", "invalid_arguments")


# --------------------------------------------------------------------------- specs


async def test_register_typed_specs_and_validation(env):
    assert env.names == NAMES
    assert http_tools.register(env.registry, services={}) == []  # idempotent
    assert [spec.name for spec in env.registry.list_specs()] == NAMES

    for name in NAMES:
        spec = env.registry.get(name).spec
        assert spec.category == "research"
        assert spec.risk is RiskLevel.MEDIUM
        assert spec.permissions == ["network:read"]
        assert spec.resource_field == ""
        assert spec.output_schema  # executor rejects malformed handler output
        schema = spec.input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        # No model-authorable data channel: these are the whole fabrication path.
        assert "data" not in schema["properties"]
        assert "default" not in schema["properties"]

    assert env.registry.validate("http_json", {"url": "http://x/", "path": "a.b"}) == []
    for arguments in (
        {"url": "http://x/", "path": "a", "data": "{}"},
        {"url": "http://x/", "path": "a", "default": 279.5},
        {"url": "http://x/"},  # path is mandatory
        {"path": "a"},  # url is mandatory
        {"url": "http://x/", "path": "", "extra": 1},
        {"url": "http://x/", "path": "a", "max_bytes": 10**9},  # hard cap
        {"url": "http://x/", "path": "a", "timeout": 10**6},  # hard cap
    ):
        assert env.registry.validate("http_json", arguments)

    assert env.registry.validate("http_json_paths", {"url": "http://x/"}) == []
    assert env.registry.validate(
        "http_json_paths", {"url": "http://x/", "prefix": "rates", "limit": 2}
    ) == []
    assert env.registry.validate("http_json_paths", {"url": "http://x/", "data": "{}"})
    assert env.registry.validate(
        "http_json_paths", {"url": "http://x/", "limit": 10_000}
    )


# --------------------------------------------------------------------------- live extraction


async def test_http_json_extracts_exact_value_from_live_response(env):
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    assert result.success, result.error
    assert result.decision == "allowed"
    out = result.output
    assert out["value"] == 276.741572
    assert out["path"] == "rates.PKR" and out["found"] is True
    assert out["url"] == _url(env) and out["status_code"] == 200
    assert out["source"] == SOURCE and out["fetched_at"]
    assert out["freshness"]["age"] == "3"
    assert out["freshness"]["cache_control"] == "no-cache, max-age=60"
    assert out["freshness"]["etag"] == '"v1"'
    # http.server adds its own Date header; httpx joins repeated headers.
    assert "Wed, 20 Nov 2026 00:00:00 GMT" in out["freshness"]["date"]
    # Chat-loop safety: provenance and value are the FIRST fields, and the
    # complete message (envelope included) fits the 900-char budget.
    assert list(out)[:2] == ["source", "value"]
    assert http_tools._final_message_size(out) <= http_tools.MAX_TOOL_MESSAGE_CHARS
    # source comes from the tool, never from the untrusted body's "cached" key.
    assert out["source"] != RATES_DOCUMENT["source"]
    assert len(env.state.requests) == 1
    assert len(_ClientSpy.calls) == 1
    assert "PKR" in env.state.requests[0]["path"] or env.state.requests[0]["path"] == "/rates"

    nested = await _run(
        env, "http_json", {"url": _url(env), "path": "history.1.value"}
    )
    assert nested.success and nested.output["value"] == 277.125
    assert len(env.state.requests) == 2  # one GET per tool call, no second hop


async def test_http_json_uses_shared_pinning_policy(env):
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    assert result.success, result.error

    request = _ClientSpy.calls[-1]
    assert request["method"] == "GET"
    # Connection is pinned to the single pre-validated address literal.
    assert request["url"] == f"http://127.0.0.1:{env.port}/rates"
    # The real host is preserved for Host and TLS SNI/certificate verification,
    # and a fresh representation is requested explicitly.
    assert request["headers"] == {
        "Host": f"rates.test:{env.port}", "Cache-Control": "no-cache",
    }
    assert request["extensions"] == {"sni_hostname": "rates.test"}
    assert env.state.requests[-1]["host"] == f"rates.test:{env.port}"
    assert env.state.requests[-1]["cache_control"] == "no-cache"

    options = _ClientSpy.instances[-1]
    assert options["follow_redirects"] is False
    assert options["trust_env"] is False
    assert options["timeout"] == 10.0


async def test_http_json_paths_discovers_live_paths(env):
    under_rates = await _run(
        env, "http_json_paths", {"url": _url(env), "prefix": "rates"}
    )
    assert under_rates.success, under_rates.error
    out = under_rates.output
    assert out["paths"] == ["rates.EUR", "rates.PKR", "rates.USD"]
    assert out["has_more"] is False and out["omitted_paths"] == 0
    assert out["prefix"] == "rates" and out["source"] == SOURCE
    assert out["url"] == _url(env) and out["status_code"] == 200
    assert out["fetched_at"] and out["freshness"]["etag"] == '"v1"'
    assert list(out)[:2] == ["source", "paths"]

    root = await _run(env, "http_json_paths", {"url": _url(env)})
    assert root.success and "rates" in root.output["paths"]
    assert root.output["prefix"] == "" and root.output["source"] == SOURCE

    indexed = await _run(
        env, "http_json_paths", {"url": _url(env), "prefix": "history", "limit": 1}
    )
    assert indexed.success and indexed.output["paths"] == ["history.0"]
    assert indexed.output["has_more"] is True

    # Every discovery result is fetched fresh; none reuses the previous body.
    assert len(env.state.requests) == 3


# --------------------------------------------------------------------------- fail closed


@pytest.mark.parametrize(
    ("path", "arguments", "fragment"),
    [
        ("/rates", {"path": "rates.DOES_NOT_EXIST"}, "absent"),
        ("/rates", {"path": ""}, "path"),  # caught by schema minLength
        ("/rates", {"path": "rates.PKR.deep"}, "cannot traverse"),
        ("/error", {"path": "rates.PKR"}, "2xx"),
        ("/redirect", {"path": "rates.PKR"}, "2xx"),
        ("/big", {"path": "filler", "max_bytes": 200}, "truncated"),
        ("/bad", {"path": "rates.PKR"}, "invalid JSON"),
        ("/nan", {"path": "rates.PKR"}, "non-standard JSON constant"),
        ("/html", {"path": "rates.PKR"}, "not JSON"),
        ("/bigvalue", {"path": "blob"}, "too large"),
    ],
)
async def test_http_json_fail_closed(env, path, arguments, fragment):
    result = await _run(
        env, "http_json", {"url": _url(env, path), **arguments}
    )
    _assert_error(result, fragment)


async def test_redirect_body_is_never_parsed_or_followed(env):
    result = await _run(
        env, "http_json", {"url": _url(env, "/redirect"), "path": "rates.PKR"}
    )
    _assert_error(result, "2xx")
    # No second hop to /rates, and no value read from the redirect body.
    assert [item["path"] for item in env.state.requests] == ["/redirect"]


async def test_paths_discovery_fail_closed(env):
    _assert_error(
        await _run(
            env, "http_json_paths", {"url": _url(env), "prefix": "no.such.prefix"}
        ),
        "absent",
    )
    _assert_error(
        await _run(env, "http_json_paths", {"url": _url(env), "prefix": "provider"}),
        "object or array",
    )
    _assert_error(
        await _run(env, "http_json_paths", {"url": _url(env, "/bad")}),
        "invalid JSON",
    )
    _assert_error(
        await _run(env, "http_json_paths", {"url": _url(env, "/error")}),
        "2xx",
    )


# --------------------------------------------------------------------------- SSRF


async def test_private_and_loopback_hosts_are_denied(env, monkeypatch):
    before = len(_ClientSpy.calls)

    # The literal address is checked by the real policy in _assert_public_host.
    monkeypatch.setattr(builtins, "is_public_ip", REAL_IS_PUBLIC)
    literal = await _run(
        env, "http_json", {"url": f"http://127.0.0.1:{env.port}/rates", "path": "rates.PKR"}
    )
    _assert_error(literal, "not allowed")
    localhost = await _run(
        env, "http_json", {"url": "http://localhost/rates", "path": "rates.PKR"}
    )
    _assert_error(localhost, "not allowed")

    # A hostname that resolves to a private address is denied even with the
    # mock-server bridge in place (the bridge exempts only 127.0.0.1).
    monkeypatch.setattr(
        builtins,
        "resolve_host",
        lambda host, p: [(2, 1, 6, "", ("10.0.0.5", p or 80))],
    )
    private = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    _assert_error(private, "non-public")
    discovery = await _run(env, "http_json_paths", {"url": _url(env)})
    _assert_error(discovery, "non-public")

    assert len(_ClientSpy.calls) == before  # no request left the process
    assert env.state.requests == []


async def test_non_http_and_credentialed_urls_are_denied(env):
    before = len(_ClientSpy.calls)
    for url in ("file:///etc/passwd", "http://user:pass@rates.test/rates"):
        result = await _run(env, "http_json", {"url": url, "path": "rates.PKR"})
        assert result.success is False, url
        assert "only http" in result.error or "credentials" in result.error
    assert len(_ClientSpy.calls) == before
    assert env.state.requests == []


# --------------------------------------------------------------------------- provenance / audit


async def test_one_get_one_audit_row_and_no_fabricated_data(env):
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    assert result.success and result.output["value"] == 276.741572

    calls = await db.fetchall("SELECT * FROM tool_calls")
    assert len(calls) == 1
    assert calls[0]["tool"] == "http_json"
    assert json.loads(calls[0]["args"]) == {
        "url": _url(env),
        "path": "rates.PKR",
    }
    assert "json_query" not in [spec.name for spec in env.registry.list_specs()]

    # The permission manager also writes its own decision row
    # (action='tool:http_json'); the executor writes exactly one final row.
    audit = await db.fetchall("SELECT * FROM audit_log WHERE action='tool.execute'")
    assert len(audit) == 1
    assert audit[0]["decision"] == "allowed"
    permission_audit = await db.fetchall(
        "SELECT * FROM audit_log WHERE action='tool:http_json'"
    )
    assert len(permission_audit) == 1

    called = await env.events.history("tool.called")
    failed = await env.events.history("tool.failed")
    assert len(called) == 1 and failed == []
    assert called[0].payload["output"]["value"] == 276.741572
    assert called[0].payload["output"]["source"] == SOURCE
    assert len(_ClientSpy.calls) == 1


async def test_no_cached_or_static_data_is_ever_reported_live(env):
    first = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    assert first.success and first.output["value"] == 276.741572

    changed = dict(RATES_DOCUMENT)
    changed["rates"] = dict(RATES_DOCUMENT["rates"], PKR=1.5)
    env.state.routes["/rates"] = (200, "application/json", json.dumps(changed).encode())
    try:
        second = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
        assert second.success, second.error
        assert second.output["value"] == 1.5
        assert second.output["source"] == SOURCE
        assert len(env.state.requests) == 2
        assert len(_ClientSpy.calls) == 2
    finally:
        env.state.routes["/rates"] = (
            200,
            "application/json",
            json.dumps(RATES_DOCUMENT).encode(),
        )


async def test_output_schema_rejects_malformed_handler_output(env, monkeypatch):
    """The optional output_schema is enforced by the real executor."""

    async def broken(args, ctx):
        return {"url": args["url"], "status_code": "not-an-int", "path": "x",
                "found": True, "value": 1, "source": SOURCE}

    monkeypatch.setattr(env.registry.get("http_json"), "handler", broken)
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    assert result.success is False
    assert result.decision == "invalid_output"
    assert "output schema violation" in result.error
    assert env.state.requests == []  # schema guard runs without any network I/O


# --------------------------------------------------------------------------- F1: final-serialization bound


async def test_cjk_value_survives_the_real_chat_loop_trim(env):
    """The bound is measured with ensure_ascii=True on the final tool message."""
    result = await _run(env, "http_json", {"url": _url(env, "/cjk"), "path": "text"})
    assert result.success, result.error
    value = "值" * 40  # 240 escaped chars, not 40; the old 1500-char bound was wrong
    assert result.output["value"] == value
    assert list(result.output)[:2] == ["source", "value"]
    assert http_tools._final_message_size(result.output) <= 900

    # Exactly the chat loop's serialization (json.dumps default=ensure_ascii=True)
    # and ContextOptimizer's 2000-char per-tool trim.
    content = json.dumps(result.model_dump(mode="json"), default=str)
    assert len(content) <= 900
    message = ChatMessage(role="tool", content=content, tool_call_id="call-cjk")
    trimmed = ContextOptimizer(tool_output_chars=2000)._trim_tool_output([message])
    assert trimmed[0].content == content  # nothing was cut
    parsed = json.loads(trimmed[0].content)
    assert parsed["output"]["value"] == value
    assert parsed["output"]["source"] == SOURCE

    # A value that cannot fit the final serialization is a clear failure, never
    # a half-escaped string shipped with a provenance marker.
    oversized = await _run(env, "http_json", {"url": _url(env, "/cjkbig"), "path": "text"})
    _assert_error(oversized, "too large")


async def test_long_echo_fields_are_bounded_and_hashed(env):
    long_path = "/" + "p" * 300
    env.state.routes[long_path] = (200, "application/json", b'{"v": 1}')
    url = f"http://rates.test:{env.port}{long_path}"
    result = await _run(env, "http_json", {"url": url, "path": "v"})
    assert result.success, result.error
    echoed = result.output["url"]
    assert "..." in echoed and "#" in echoed
    assert len(echoed) <= http_tools.MAX_ECHO_CHARS + 24
    assert result.output["url_hash"] == http_tools._digest(url)
    assert long_path not in json.dumps(result.output, ensure_ascii=True)


# --------------------------------------------------------------------------- F3: no model-callable fabrication path


def test_json_query_is_absent_from_the_composed_catalogue():
    """Check the real register_all_toolsets path, not an isolated fixture."""
    assert "json_query" in MODEL_FORBIDDEN_TOOLS
    registry = ToolRegistry()
    register_all_toolsets(registry, services={})
    names = {spec.name for spec in registry.list_specs()}
    assert "json_query" not in names
    assert {"http_json", "http_json_paths"} <= names
    # Pure offline callers still have the handler; only the model catalogue drops it.
    from leo.tools.data_tools import json_query

    assert callable(json_query)


# --------------------------------------------------------------------------- F4: redaction stays truthful


@pytest.mark.parametrize(
    ("path", "fragment"),
    [("/secret", "redaction pattern"), ("/keysecret", "redaction pattern")],
)
async def test_sensitive_values_are_refused_not_redacted(env, path, fragment):
    result = await _run(env, "http_json", {"url": _url(env, path), "path": "note"})
    _assert_error(result, fragment)
    assert "[REDACTED]" not in (result.error or "")
    called = await env.events.history("tool.called")
    assert called == []  # no tool.called event carrying a redacted 'value'


# --------------------------------------------------------------------------- F5: URL secret hygiene


async def test_query_secrets_are_never_echoed(env):
    secret = "supersecretvalue123"
    token = "tok_abcdefghijklmnop"
    url = _url(env, "/rates") + f"?key={secret}&page=2&api_token={token}"
    result = await _run(env, "http_json", {"url": url, "path": "rates.PKR"})
    assert result.success, result.error
    assert result.output["url"] == f"http://rates.test:{env.port}/rates?page=***"
    assert result.output["url_hash"] == http_tools._digest(url)
    blob = json.dumps(result.model_dump(mode="json"), ensure_ascii=True)
    assert secret not in blob and token not in blob
    called = await env.events.history("tool.called")
    assert called and secret not in called[0].model_dump_json()
    assert token not in called[0].model_dump_json()
    # The executor still records the model-supplied URL in tool_calls.args;
    # broadening that config-level redactor is a separate follow-up.


async def test_secret_in_url_path_is_masked_too(env):
    secret = "sk-abcdefghijklmnop123456"
    path = f"/v1/{secret}"
    env.state.routes[path] = (200, "application/json", b'{"v": 1}')
    url = f"http://rates.test:{env.port}{path}"
    result = await _run(env, "http_json", {"url": url, "path": "v"})
    assert result.success, result.error
    assert secret not in json.dumps(result.output, ensure_ascii=True)
    assert "[masked]" in result.output["url"]


# --------------------------------------------------------------------------- F6: fetched, not necessarily fresh


async def test_freshness_evidence_and_fetched_provenance(env):
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    assert result.success and result.output["source"] == SOURCE
    assert result.output["fetched_at"]
    freshness = result.output["freshness"]
    assert freshness["age"] == "3"
    assert freshness["cache_control"] == "no-cache, max-age=60"
    assert freshness["etag"] == '"v1"'
    assert "Wed, 20 Nov 2026 00:00:00 GMT" in freshness["date"]
    assert env.state.requests[-1]["cache_control"] == "no-cache"
    for name in NAMES:
        description = env.registry.get(name).spec.description
        assert SOURCE in description
        assert "cached" in description and "real-time" in description


# --------------------------------------------------------------------------- F7: honest enumeration


async def test_paths_reports_omitted_and_incomplete_enumerations(env):
    result = await _run(
        env, "http_json_paths", {"url": _url(env, "/paths"), "limit": 1}
    )
    assert result.success, result.error
    # "a.b" (dotted), "" (empty), and the over-long key cannot be expressed;
    # the second valid key is not reached because limit=1.
    assert result.output["paths"] == ["ok"]
    assert result.output["omitted_paths"] == 4
    assert result.output["has_more"] is True

    secret_paths = await _run(
        env, "http_json_paths", {"url": _url(env, "/secretpaths")}
    )
    assert secret_paths.success, secret_paths.error
    assert secret_paths.output["paths"] == ["safe"]
    assert secret_paths.output["omitted_paths"] == 1
    assert secret_paths.output["has_more"] is True
    assert "sk-abcdefghijklmnop123456" not in json.dumps(secret_paths.output)


# --------------------------------------------------------------------------- re-review: redaction-aware budget (cross-cutting)


def _rebuild_executor(env, patterns: list[str]) -> None:
    """The executor compiles redaction patterns at construction time."""
    env.settings.security.redact_patterns = list(patterns)
    env.executor = ToolExecutorImpl(
        env.registry, env.permissions, env.events, env.settings
    )


async def test_expanding_redaction_is_measured_before_the_message_budget(env):
    """A short match becomes '[REDACTED]': the output grows after redaction."""
    blob = "Z" * 60
    env.state.routes["/expand"] = (
        200, "application/json", json.dumps({"blob": blob}).encode()
    )
    args = {"url": _url(env, "/expand"), "path": "blob"}
    baseline = await _run(env, "http_json", args)
    assert baseline.success, baseline.error
    out = baseline.output
    patterns = [re.compile("Z")]

    plain_size = http_tools._final_message_size(out)
    expanded_size = http_tools._final_message_size(out, patterns)
    assert plain_size <= http_tools.MAX_TOOL_MESSAGE_CHARS
    assert expanded_size > plain_size + 400
    assert expanded_size > http_tools.MAX_TOOL_MESSAGE_CHARS
    # The size guard itself must use the post-redaction serialization.
    with pytest.raises(ValueError, match="too large"):
        http_tools._assert_message_budget(out, "extracted value", patterns)
    with pytest.raises(ValueError, match="redaction"):
        http_tools._assert_redaction_stability(out, patterns, "extracted value")

    # End to end: the executor redacts the same way; no redacted result is
    # published as a successful HTTP extraction.
    _rebuild_executor(env, ["Z"])
    result = await _run(env, "http_json", args)
    _assert_error(result, "redaction")
    assert "[REDACTED]" not in (result.error or "")
    called = await env.events.history("tool.called")
    assert all("[REDACTED]" not in event.model_dump_json() for event in called)
    content = json.dumps(result.model_dump(mode="json"), default=str)
    assert len(content) <= http_tools.MAX_TOOL_MESSAGE_CHARS


@pytest.mark.parametrize("pattern", ["2026", "no-cache", '"v1"'])
async def test_redacted_provenance_fields_are_refused_not_silently_published(env, pattern):
    """F4: fetched_at/freshness text must not become '[REDACTED]' on success."""
    _rebuild_executor(env, [pattern])
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    _assert_error(result, "redaction")
    assert "[REDACTED]" not in (result.error or "")
    called = await env.events.history("tool.called")
    assert called == []

    discovery = await _run(env, "http_json_paths", {"url": _url(env)})
    _assert_error(discovery, "redaction")
    assert "[REDACTED]" not in (discovery.error or "")


async def test_url_hash_matching_a_pattern_is_refused(env):
    digest = http_tools._digest(_url(env))
    _rebuild_executor(env, [re.escape(digest[:8])])
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    _assert_error(result, "url_hash")
    assert digest not in (result.error or "")


# --------------------------------------------------------------------------- re-review: path/key sanitization (F5)


async def test_bare_secret_tokens_are_masked_without_configured_patterns(env):
    """Intrinsic token shapes hold even when the operator clears the redactor."""
    _rebuild_executor(env, [])
    secret = "sk-abcdefghijklmnop123456"
    path = f"/v1/{secret}"
    env.state.routes[path] = (200, "application/json", b'{"v": 1}')
    result = await _run(
        env, "http_json", {"url": f"http://rates.test:{env.port}{path}", "path": "v"}
    )
    assert result.success, result.error
    blob = json.dumps(result.output, ensure_ascii=True)
    assert secret not in blob
    assert "[masked]" in result.output["url"]

    prefix_result = await _run(
        env,
        "http_json",
        {"url": _url(env, "/secretpaths"), "path": "safe"},
    )
    assert prefix_result.success, prefix_result.error
    assert "sk-abcdefghijklmnop123456" not in json.dumps(prefix_result.output)


async def test_secret_looking_response_keys_are_omitted_and_value_keys_refused(env):
    """Every emitted key is sanitized; secret keys never reach the chat loop."""
    _rebuild_executor(env, [])
    secret = "ghp_" + "a" * 20
    env.state.routes["/barekeys"] = (
        200, "application/json",
        json.dumps({"safe": 1, secret: 2, "bad\nkey": 3}).encode(),
    )
    result = await _run(env, "http_json_paths", {"url": _url(env, "/barekeys")})
    assert result.success, result.error
    assert result.output["paths"] == ["safe"]
    assert result.output["omitted_paths"] == 2
    assert result.output["has_more"] is True
    blob = json.dumps(result.output, ensure_ascii=True)
    assert secret not in blob and "bad\\nkey" not in blob

    env.state.routes["/objkeys"] = (
        200, "application/json", json.dumps({"obj": {secret: 1}}).encode()
    )
    extracted = await _run(
        env, "http_json", {"url": _url(env, "/objkeys"), "path": "obj"}
    )
    _assert_error(extracted, "redaction pattern")
    assert secret not in (extracted.error or "")
    called = await env.events.history("tool.called")
    assert all(secret not in event.model_dump_json() for event in called)


async def test_unknown_shape_path_tokens_are_bounded_and_hashed(env):
    # resolve_path caps a segment at 200 chars; 180 is still long enough to
    # exercise the echo-side bound (64 chars per segment plus a hash).
    long_key = "k" * 180
    env.state.routes["/weird"] = (
        200, "application/json", json.dumps({long_key: {"v": 2}, "v": 1}).encode()
    )
    result = await _run(env, "http_json", {"url": _url(env, "/weird"), "path": long_key})
    assert result.success, result.error
    echoed = result.output["path"]
    blob = json.dumps(result.output, ensure_ascii=True)
    assert long_key not in blob
    assert "..." in echoed and "#" in echoed
    assert len(echoed) <= http_tools.MAX_ECHO_CHARS + 24

    discovery = await _run(
        env, "http_json_paths", {"url": _url(env, "/weird"), "prefix": long_key}
    )
    assert discovery.success, discovery.error
    assert discovery.output["prefix"] == echoed
    assert long_key not in json.dumps(discovery.output, ensure_ascii=True)

    control_token = "a\x00b"
    assert http_tools._sanitize_token(control_token, []) == (
        f"[masked]#{http_tools._digest(control_token)}"
    )
    assert "\x00" not in http_tools._sanitize_token(control_token, [])


async def test_absent_path_errors_are_bounded_and_secret_free(env):
    secret = "sk-abcdefghijklmnop123456"
    secret_result = await _run(
        env, "http_json", {"url": _url(env), "path": f"rates.{secret}"}
    )
    _assert_error(secret_result, "absent")
    assert secret not in (secret_result.error or "")

    long_path = ".".join(f"seg{i}_" + "x" * 30 for i in range(10))
    long_result = await _run(
        env, "http_json", {"url": _url(env), "path": long_path}
    )
    _assert_error(long_result, "absent")
    assert long_path not in (long_result.error or "")
    assert len(long_result.error or "") < 400


# --------------------------------------------------------------------------- re-review: compressed-stream honesty (F2)


def test_compressed_stream_limit_is_documented(env):
    """The shared GET decodes one member; tool descriptions must say so."""
    doc = (http_tools.__doc__ or "").lower()
    assert "concatenated gzip" in doc
    assert "unused_data" in doc
    for name in NAMES:
        description = env.registry.get(name).spec.description.lower()
        assert "concatenated/truncated compressed members are refused" in description


async def test_single_gzip_member_still_extracts_the_exact_value(env):
    body = json.dumps(RATES_DOCUMENT).encode()
    env.state.routes["/gzip"] = (200, "application/json", gzip.compress(body))
    env.state.encodings["/gzip"] = "gzip"
    result = await _run(env, "http_json", {"url": _url(env, "/gzip"), "path": "rates.PKR"})
    assert result.success, result.error
    assert result.output["value"] == 276.741572
    assert len(env.state.requests) == 1


async def test_concatenated_or_incomplete_gzip_is_refused_not_partially_published(env):
    body = json.dumps({"rates": {"PKR": 276.741572}}).encode()
    injected = json.dumps({"rates": {"PKR": 999.0}}).encode()
    env.state.routes["/gzipcat"] = (
        200, "application/json", gzip.compress(body) + gzip.compress(injected)
    )
    env.state.encodings["/gzipcat"] = "gzip"
    result = await _run(
        env, "http_json", {"url": _url(env, "/gzipcat"), "path": "rates.PKR"}
    )
    _assert_error(result, "partially decoded")
    assert "999.0" not in (result.error or "")

    # A member whose gzip trailer is missing decodes to a complete-looking JSON
    # document but is not a complete compressed representation either.
    env.state.routes["/gzipcut"] = (
        200, "application/json", gzip.compress(body)[:-8]
    )
    env.state.encodings["/gzipcut"] = "gzip"
    incomplete = await _run(
        env, "http_json", {"url": _url(env, "/gzipcut"), "path": "rates.PKR"}
    )
    _assert_error(incomplete, "partially decoded")
    assert len(env.state.requests) == 2  # one GET per attempt, no retry


# --------------------------------------------------------------------------- adversarial re-review: resolve_path failures (F1)


@pytest.mark.parametrize("redactor", ["empty", "default", "expanding", "generic"])
async def test_resolve_path_exception_is_sanitized_and_budgeted(env, redactor):
    """resolve_path quotes the rejected segment; the exception must not publish it."""
    secret = "sk-abcdefghijklmnop123456"
    if redactor == "empty":
        _rebuild_executor(env, [])
        segment = secret
    elif redactor == "default":
        _rebuild_executor(env, list(Settings().security.redact_patterns))
        segment = secret
    elif redactor == "generic":
        # A pattern matching the fixed error wording must not turn the whole
        # error into '[REDACTED]' on its way through the executor.
        _rebuild_executor(env, ["path"])
        segment = secret
    else:
        _rebuild_executor(env, ["Z"])
        segment = "Z" * 200
        # Premise: without the scrub, 200 short matches expand to 2000
        # '[REDACTED]' markers in the model-visible error.
        raw = f"path segment 1 ({segment!r}) is not a valid list index"
        assert http_tools._final_message_size({"error": raw}, [re.compile("Z")]) > 2000

    result = await _run(env, "http_json", {"url": _url(env), "path": f"history.{segment}"})
    _assert_error(result, "could not be resolved")
    error = result.error or ""
    assert segment not in error and "[REDACTED]" not in error
    assert "[masked]#" in error  # correlation without publishing the raw segment
    content = json.dumps(result.model_dump(mode="json"), default=str)
    assert len(content) <= http_tools.MAX_TOOL_MESSAGE_CHARS
    failed = await env.events.history("tool.failed")
    failure_blob = json.dumps([item.model_dump(mode="json") for item in failed], default=str)
    assert segment not in failure_blob and "[REDACTED]" not in failure_blob

    # The prefix lookup uses the same guarded seam.
    prefix_result = await _run(
        env, "http_json_paths", {"url": _url(env), "prefix": f"history.{segment}"}
    )
    _assert_error(prefix_result, "could not be resolved")
    assert segment not in (prefix_result.error or "")
    assert "[REDACTED]" not in (prefix_result.error or "")


async def test_absent_secret_path_with_no_configured_patterns_is_masked(env):
    """The non-exception branch holds too when the operator clears the redactor."""
    _rebuild_executor(env, [])
    secret = "sk-abcdefghijklmnop123456"
    result = await _run(env, "http_json", {"url": _url(env), "path": f"rates.{secret}"})
    _assert_error(result, "absent")
    assert secret not in (result.error or "")
    assert "[REDACTED]" not in (result.error or "")


async def test_resolve_path_exception_respects_the_message_budget(env, monkeypatch):
    _rebuild_executor(env, [])
    monkeypatch.setattr(http_tools, "MAX_TOOL_MESSAGE_CHARS", 120)
    secret = "sk-abcdefghijklmnop123456"
    result = await _run(env, "http_json", {"url": _url(env), "path": f"history.{secret}"})
    _assert_error(result, "too large")
    assert secret not in (result.error or "")
    assert "[REDACTED]" not in (result.error or "")


# --------------------------------------------------------------------------- adversarial re-review: prefix usability (F2)


async def test_every_advertised_path_resolves_with_http_json(env):
    """Discovery must never advertise a path that cannot be looked up."""
    for route, prefix in (("/rates", None), ("/rates", "rates"), ("/paths", None)):
        arguments = {"url": _url(env, route)}
        if prefix:
            arguments["prefix"] = prefix
        discovery = await _run(env, "http_json_paths", arguments)
        assert discovery.success, discovery.error
        # Withheld/omitted child keys are now explained; a complete enumeration
        # still has no reason to report.
        assert (discovery.output["reason"] is None) == (
            discovery.output["omitted_paths"] == 0
        )
        assert discovery.output["paths"]
        for path in discovery.output["paths"]:
            fetched = await _run(env, "http_json", {"url": _url(env, route), "path": path})
            assert fetched.success and fetched.output["found"] is True, (path, fetched.error)


@pytest.mark.parametrize("mode", ["intrinsic", "hashed", "configured"])
async def test_masked_prefix_withholds_unusable_paths(env, mode):
    secret = "sk-abcdefghijklmnop123456"
    if mode == "intrinsic":
        _rebuild_executor(env, [])
        key = secret
    elif mode == "configured":
        _rebuild_executor(env, ["mysecretkey"])
        key = "mysecretkey"
    else:
        _rebuild_executor(env, [])
        key = "k" * 180  # bounded and hashed rather than masked
    env.state.routes["/maskedprefix"] = (
        200, "application/json",
        json.dumps({key: {"a": 1}, "safe": {"b": 2}}).encode(),
    )
    result = await _run(
        env, "http_json_paths", {"url": _url(env, "/maskedprefix"), "prefix": key}
    )
    assert result.success, result.error
    out = result.output
    assert out["paths"] == []
    assert out["omitted_paths"] >= 1
    assert out["has_more"] is True
    assert "prefix was masked or hashed" in out["reason"]
    assert out["prefix"] != key
    blob = json.dumps(out, ensure_ascii=True)
    assert key not in blob
    assert http_tools._final_message_size(out) <= http_tools.MAX_TOOL_MESSAGE_CHARS


# --------------------------------------------------------------------------- adversarial re-review: invisible separators (F3)


@pytest.mark.parametrize("codepoint", [0x200B, 0x200C, 0x2060])
async def test_zero_width_separators_cannot_hide_secrets(env, codepoint):
    sep = chr(codepoint)

    # Intrinsic detection with every configured pattern cleared.
    _rebuild_executor(env, [])
    secret = f"sk-{sep}abcdefghijklmnop123456"
    env.state.routes["/zwsecret"] = (
        200, "application/json", json.dumps({"note": secret}).encode()
    )
    result = await _run(env, "http_json", {"url": _url(env, "/zwsecret"), "path": "note"})
    _assert_error(result, "redaction pattern")
    assert "sk-" not in (result.error or "") and sep not in (result.error or "")

    # Configured patterns (including patterns written with a separator) match
    # the normalized text as well.
    _rebuild_executor(env, [f"supersecret{sep}value"])
    env.state.routes["/zwconfig"] = (
        200, "application/json", json.dumps({"note": "supersecretvalue"}).encode()
    )
    configured = await _run(env, "http_json", {"url": _url(env, "/zwconfig"), "path": "note"})
    _assert_error(configured, "redaction pattern")
    assert "[REDACTED]" not in (configured.error or "")

    # Echoed path/URL tokens are masked as a whole, separator included.
    assert http_tools._sanitize_token(secret, []) == f"[masked]#{http_tools._digest(secret)}"
    assert sep not in http_tools._sanitize_token(f"pre{sep}post", [])
    absent = await _run(
        env, "http_json", {"url": _url(env), "path": f"rates.{secret}"}
    )
    _assert_error(absent, "absent")
    assert secret not in (absent.error or "") and sep not in (absent.error or "")
    prefix = await _run(
        env, "http_json_paths", {"url": _url(env), "prefix": f"rat{sep}es"}
    )
    _assert_error(prefix, "absent")
    assert sep not in (prefix.error or "")
    called = await env.events.history("tool.called")
    assert all(secret not in item.model_dump_json() for item in called)


# --------------------------------------------------------------------------- adversarial re-review: envelope and final counters (F4/F5)


def test_envelope_measurement_reserves_room_for_executor_metadata():
    empty = dict(http_tools._ENVELOPE)
    empty["metadata"] = {}
    empty_overhead = len(json.dumps(empty, ensure_ascii=True, default=str)) - len("null")
    assert http_tools._ENVELOPE_OVERHEAD >= empty_overhead + 20
    assert http_tools._ENVELOPE["metadata"]  # not the optimistic {} of before


async def test_lower_optimizer_trim_is_not_covered_by_the_900_char_budget(env):
    result = await _run(env, "http_json", {"url": _url(env), "path": "rates.PKR"})
    assert result.success, result.error
    content = json.dumps(result.model_dump(mode="json"), default=str)

    # Modeling the *real* executor metadata (not just the allowance) keeps the
    # representative message inside the documented bound.
    envelope = dict(http_tools._ENVELOPE)
    envelope["output"] = result.output
    envelope["metadata"] = result.metadata
    assert len(json.dumps(envelope, ensure_ascii=True, default=str)) <= 900

    message = ChatMessage(role="tool", content=content, tool_call_id="call-trim")
    assert len(content) < 900
    kept = ContextOptimizer(tool_output_chars=len(content))._trim_tool_output([message])
    assert kept[0].content == content
    # A caller lowering the trim below the modeled bound does cut the message:
    # the 900-char budget is not a promise about other optimizer settings.
    cut = ContextOptimizer(tool_output_chars=len(content) - 10)._trim_tool_output([message])
    assert cut[0].content != content


async def test_paths_budget_is_recomputed_with_final_counters(env, monkeypatch):
    """Intermediate trials must not be mistaken for the final message size."""

    def trial_size(output, patterns=None):
        # Pretend the extra digits in the FINAL omitted_paths push the message
        # over budget; the in-progress count (1) still fits.
        return 1000 if output["omitted_paths"] == 2 else 100

    monkeypatch.setattr(http_tools, "_final_message_size", trial_size)
    result = await _run(env, "http_json_paths", {"url": _url(env), "limit": 3})
    assert result.success, result.error
    # Five root keys, limit 3: the final omitted_paths is 2 (not the 1 used by
    # the in-progress trials), so one tail path is dropped on recomputation.
    assert len(result.output["paths"]) == 2
    assert result.output["omitted_paths"] == 3
    assert result.output["has_more"] is True
    # The final drop withholds a key, so reason must explain it now.
    assert result.output["reason"] and "withheld" in result.output["reason"]


# --------------------------------------------------------------------------- re-review: category-based invisible separators


@pytest.mark.parametrize("codepoint", INVISIBLE_CODEPOINTS)
async def test_invisible_unicode_cannot_split_secrets_on_any_surface(env, codepoint):
    """Cf/Cc/Cs/Mn separators must not defeat detection on any surface."""
    sep = chr(codepoint)
    secret = f"sk-{sep}abcdefghijklmnop123456"
    visible = secret.replace(sep, "")
    _rebuild_executor(env, [])  # intrinsic detection must hold with no config

    # (1) Fetched values and object keys: refuse, never publish a redacted value.
    env.state.routes["/invisible-value"] = (
        200, "application/json",
        json.dumps({
            "note": secret,
            "assigned": f"api{sep}key: supersecretvalue123",
        }).encode(),
    )
    value = await _run(
        env, "http_json", {"url": _url(env, "/invisible-value"), "path": "note"}
    )
    assigned = await _run(
        env, "http_json", {"url": _url(env, "/invisible-value"), "path": "assigned"}
    )
    for result in (value, assigned):
        _assert_error(result, "redaction pattern")
        assert all(
            marker not in (result.error or "")
            for marker in (secret, visible, sep, "[REDACTED]")
        )

    env.state.routes["/invisible-key-value"] = (
        200, "application/json", json.dumps({"obj": {secret: 1}}).encode()
    )
    keyed = await _run(
        env, "http_json", {"url": _url(env, "/invisible-key-value"), "path": "obj"}
    )
    _assert_error(keyed, "redaction pattern")
    assert visible not in (keyed.error or "") and sep not in (keyed.error or "")

    # (2) Response keys in enumeration: withheld, counted, explained, never echoed.
    env.state.routes["/invisible-keys"] = (
        200, "application/json", json.dumps({"safe": 1, secret: 2}).encode()
    )
    discovery = await _run(
        env, "http_json_paths", {"url": _url(env, "/invisible-keys")}
    )
    assert discovery.success, discovery.error
    out = discovery.output
    assert out["paths"] == ["safe"]
    assert out["omitted_paths"] == 1 and out["has_more"] is True
    assert out["reason"]
    blob = json.dumps(out, ensure_ascii=True)
    assert all(marker not in blob for marker in (secret, visible, sep))

    # (3) Echoed path/prefix/URL tokens: masked as a whole, separator gone.
    assert http_tools._sanitize_token(secret, []) == f"[masked]#{http_tools._digest(secret)}"
    assert sep not in http_tools._sanitize_token(f"pre{sep}post", [])
    url_echo = http_tools._sanitize_url(f"http://host.test/v1/pre{sep}post", [])
    assert sep not in url_echo and "[masked]" in url_echo
    absent = await _run(
        env, "http_json", {"url": _url(env), "path": f"rates.{secret}"}
    )
    prefix = await _run(
        env, "http_json_paths", {"url": _url(env), "prefix": f"rat{sep}es"}
    )
    broken = await _run(
        env, "http_json", {"url": _url(env), "path": f"history.{secret}"}
    )
    _assert_error(absent, "absent")
    _assert_error(prefix, "absent")
    _assert_error(broken, "could not be resolved")
    assert "[masked]#" in (broken.error or "")
    for result in (absent, prefix, broken):
        for marker in (secret, visible, sep, "[REDACTED]"):
            assert marker not in (result.error or ""), (marker, result.error)

    # (4) Every error message stays inside the final chat budget and no
    # tool.called event carries the raw token.
    for result in (value, assigned, keyed, absent, prefix, broken):
        content = json.dumps(result.model_dump(mode="json"), default=str)
        assert len(content) <= http_tools.MAX_TOOL_MESSAGE_CHARS
        assert "[REDACTED]" not in content
    called = await env.events.history("tool.called")
    assert all(secret not in item.model_dump_json() for item in called)


# ------------------------------------------------------------------- re-review: untrusted content-type header


@pytest.mark.parametrize("redactor", ["empty", "default"])
async def test_non_json_content_type_header_is_bounded_and_scrubbed(env, redactor):
    if redactor == "empty":
        _rebuild_executor(env, [])
    else:
        _rebuild_executor(env, list(Settings().security.redact_patterns))

    # 1200 server-controlled characters cannot inflate a chat message.
    long_type = "text/html; " + "h" * 1200
    env.state.routes["/longtype"] = (200, long_type, b'{"v": 1}')
    long_result = await _run(
        env, "http_json", {"url": _url(env, "/longtype"), "path": "v"}
    )
    _assert_error(long_result, "not JSON")
    assert long_type not in (long_result.error or "")
    assert "h" * 1200 not in (long_result.error or "")
    assert len(long_result.error or "") < 400
    long_message = json.dumps(long_result.model_dump(mode="json"), default=str)
    assert len(long_message) <= http_tools.MAX_TOOL_MESSAGE_CHARS

    # A secret in the header is masked rather than published raw (empty
    # redactor) or rewritten to '[REDACTED]' (default redactor).
    secret = "sk-abcdefghijklmnop123456"
    env.state.routes["/secret-type"] = (200, f"text/html; note={secret}", b'{"v": 1}')
    secret_result = await _run(
        env, "http_json", {"url": _url(env, "/secret-type"), "path": "v"}
    )
    _assert_error(secret_result, "not JSON")
    assert secret not in (secret_result.error or "")
    assert "[REDACTED]" not in (secret_result.error or "")
    assert "[masked]" in (secret_result.error or "")
    secret_message = json.dumps(secret_result.model_dump(mode="json"), default=str)
    assert secret not in secret_message and "[REDACTED]" not in secret_message
    assert len(secret_message) <= http_tools.MAX_TOOL_MESSAGE_CHARS
    called = await env.events.history("tool.called")
    assert called == []
    failed = await env.events.history("tool.failed")
    assert all(secret not in item.model_dump_json() for item in failed)


# ------------------------------------------- re-review: invisible content-type markers


@pytest.mark.parametrize("codepoint", INVISIBLE_CODEPOINTS)
async def test_invisible_separator_cannot_split_non_json_content_type(
    env, monkeypatch, codepoint
):
    """A marker split by any invisible codepoint gets the plain text/html refusal.

    A real HTTP header cannot carry non-ASCII or Cc codepoints, so patch the
    single shared GET seam and hand the untrusted header value directly to the
    tools. The marker check, error scrubbing and executor redaction stay real.
    """
    sep = chr(codepoint)

    async def fake_get(args, ctx):
        return {
            "content": '{"rates": {"PKR": 999.0}}',
            "status_code": 200,
            "content_type": fake_get.content_type,
            "fetched_at": "Wed, 20 Nov 2026 00:00:00 GMT",
            "freshness": {},
            "truncated": False,
        }

    async def run(content_type):
        fake_get.content_type = content_type
        monkeypatch.setattr(builtins, "http_get", fake_get)
        return (
            await _run(
                env, "http_json", {"url": _url(env), "path": "rates.PKR"}
            ),
            await _run(env, "http_json_paths", {"url": _url(env)}),
        )

    def assert_same_refusal(baseline, result, header, label):
        _assert_error(baseline, "not JSON")
        _assert_error(result, "not JSON")
        assert baseline.error and result.error
        # Same refusal wording, independent of the scrubbed header detail.
        assert result.error.split(" (content-type:", 1)[0] == (
            baseline.error.split(" (content-type:", 1)[0]
        ), (label, repr(header))
        # The invisible separator and the body value are never echoed.
        assert sep not in result.error, (label, repr(header))
        assert "999" not in result.error, (label, repr(header))

    plain = await run("text/html")
    marker = "text/html"
    for pos in range(len(marker) + 1):
        split = await run(marker[:pos] + sep + marker[pos:])
        for index, (baseline, result) in enumerate(zip(plain, split)):
            assert_same_refusal(baseline, result, marker, NAMES[index])

    # The same category rule protects every other definitively non-JSON type.
    for marker in http_tools._NON_JSON_TYPES:
        if marker == "text/html":
            continue
        mid = len(marker) // 2
        baseline = await run(marker)
        split = await run(marker[:mid] + sep + marker[mid:])
        for index, (plain_result, result) in enumerate(zip(baseline, split)):
            assert_same_refusal(plain_result, result, marker, NAMES[index])


# ------------------------------------------------------------------- re-review: truthful withholding reason


async def test_withheld_child_keys_are_explained_in_reason(env):
    secret = "sk-abcdefghijklmnop123456"
    env.state.routes["/withheld"] = (
        200, "application/json",
        json.dumps({"safe": 1, secret: 2, "L" * 250: 3, "a.b": 4}).encode(),
    )
    result = await _run(env, "http_json_paths", {"url": _url(env, "/withheld")})
    assert result.success, result.error
    out = result.output
    assert out["paths"] == ["safe"]
    assert out["omitted_paths"] == 3
    assert out["has_more"] is True
    assert out["reason"] and "withheld" in out["reason"]
    assert "omitted_paths" in out["reason"]
    assert secret not in json.dumps(out, ensure_ascii=True)


def test_lone_surrogates_are_invisible_for_matching():
    """Cs is part of the category rule, not just the printable separators."""
    for codepoint in (0xD800, 0xDC00, 0xDFFF):
        sep = chr(codepoint)
        secret = f"sk-{sep}abcdefghijklmnop123456"
        assert http_tools._normalize_for_match(secret) == secret.replace(sep, "")
        assert http_tools._matches_secret(secret, [])
        assert http_tools._sanitize_token(secret, []) == (
            f"[masked]#{http_tools._digest(secret)}"
        )
