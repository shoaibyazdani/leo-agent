"""WhatsApp Web bridge: offline lifecycle, authentication and secret-hygiene tests.

Everything is synthetic: the Node child is a fake object and the HTTP control
plane is an in-memory transport. No WhatsApp, browser, npm install, socket or
external network is used by the default fixtures. Two dedicated regressions use
only loopback sockets and the local Python interpreter to prove that the owner
token reaches the FD-inheriting child and nobody else. The fixture monkeypatches
credential environment variables to prove they never reach the child.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo.integrations.whatsapp_bridge import (
    BridgeAuthError,
    BridgeConfig,
    BridgeError,
    BridgeNotEnabled,
    BridgeTimeout,
    BridgeUnavailable,
    SessionAuthError,
    SessionOwnershipError,
    WhatsAppBridgeManager,
)
from leo.integrations.whatsapp_bridge import manager as manager_module

BRIDGE_DIR = Path(manager_module.__file__).resolve().parent
LEAK_TOKEN = "fake-external-token-that-must-not-leak-1234567890"


def _data_uri(marker: bytes = b"qr") -> str:
    return "data:image/png;base64," + base64.b64encode(marker * 24).decode("ascii")


class FakeStream:
    """Queue-based stand-in that stays open until the manager cancels it."""

    def __init__(self) -> None:
        self._lines: asyncio.Queue[bytes] = asyncio.Queue()
        self.closed = False

    def push(self, text: str) -> None:
        self._lines.put_nowait(text.encode("utf-8"))

    def close(self) -> None:
        self.closed = True
        self._lines.put_nowait(b"")

    async def readline(self) -> bytes:
        if self.closed and self._lines.empty():
            return b""
        return await self._lines.get()


class FakeProcess:
    """Minimal asyncio-subprocess stand-in with observable cleanup calls."""

    def __init__(self, *, returncode: int | None = None, hang_first_wait: bool = False) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.stdout = FakeStream()
        self.stderr = FakeStream()
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.hang_first_wait = hang_first_wait

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.hang_first_wait and self.wait_calls == 1:
            await asyncio.sleep(10)
        return self.returncode if self.returncode is not None else 0


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, *, chunks: list[bytes] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self._chunks = chunks
        self.read_chunks = 0

    @property
    def content(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def json(self) -> dict:
        return self._payload

    async def aiter_bytes(self, chunk_size: int = 8192):
        del chunk_size
        for chunk in self._chunks if self._chunks is not None else [self.content]:
            self.read_chunks += 1
            yield chunk


class FakeBridgeServer:
    """State machine mirroring the bridge's documented control protocol."""

    def __init__(self) -> None:
        self.expected_auth: str | None = None
        self.phase = "idle"
        self.linked = False
        self.error: str | None = None
        self.sequence = 1
        self.rotation = 0
        self.qr: dict | None = None
        self.passkey_required: dict | None = None
        self.calls: list[tuple[str, str, dict | None]] = []
        self.wait_callbacks: list = []
        self.wait_calls = 0
        self.start_calls = 0
        self.cancel_calls = 0
        self.repair_calls = 0
        self.repair_payloads: list[dict | None] = []
        self.typing_calls: list[dict] = []
        self.repair_error: str | None = None
        self.force_unauthorized = False
        self.status_overrides: dict = {}
        self.next_payload: dict | None = None
        self.chunks: list[bytes] | None = None
        self.health_chunks: list[bytes] | None = None
        self.release: asyncio.Event | None = None
        self.qr_requested: asyncio.Event | None = None
        self.received_auth: list[str] = []

    def set_phase(self, phase: str, error: str | None = None) -> None:
        self.phase = phase
        self.error = error
        self.sequence += 1

    def issue_qr(self, marker: bytes = b"qr") -> None:
        self.rotation += 1
        self.qr = {
            "data_uri": _data_uri(marker),
            "expires_at": time.time() + 30,
            "rotation": self.rotation,
        }
        self.set_phase("waiting_for_scan")

    def expire_qr(self) -> None:
        self.qr = None
        self.sequence += 1

    def snapshot(self) -> dict:
        result = {
            "protocol": "whatsapp-bridge/1",
            "phase": self.phase,
            "state": self.phase,
            "linked": self.linked,
            "error": self.error,
            "passkey_required": self.passkey_required,
            "user": {
                "id": "15551234567@s.whatsapp.net",
                "name": "Owner\nName",
                "token": LEAK_TOKEN,
            },
            "qr": self.qr,
            "rotation": self.rotation,
            "sequence": self.sequence,
            "reconnect_attempt": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            # Malicious/extra fields that the manager must never forward.
            "owner_token": LEAK_TOKEN,
            "qr_raw": "PAIRING-STRING-SECRET",
        }
        result.update(self.status_overrides)
        if self.next_payload is not None:
            payload = self.next_payload
            self.next_payload = None
            result = payload
        return result

    def handle(self, method: str, path: str, body: dict | None, auth: str | None) -> FakeResponse:
        self.calls.append((method, path, body))
        if auth:
            self.received_auth.append(auth)
        if self.force_unauthorized or auth != self.expected_auth or not self.expected_auth:
            return FakeResponse(401, {"error": "unauthorized"})
        if method == "GET" and path == "/health":
            return FakeResponse(
                200, {"status": "ok", "protocol": "whatsapp-bridge/1"},
                chunks=self.health_chunks,
            )
        if method == "GET" and path == "/status":
            return FakeResponse(200, self.snapshot(), chunks=self.chunks)
        if method == "GET" and path == "/qr":
            # Deliberately do not pre-validate expires_at here: malformed values
            # must be rejected by the manager's own finite/range checks.
            if not self.qr:
                return FakeResponse(404, {"error": "no_qr"})
            return FakeResponse(200, dict(self.qr))
        if method == "POST" and path == "/start":
            self.start_calls += 1
            self.set_phase("waiting_for_qr")
            return FakeResponse(202, self.snapshot())
        if method == "POST" and path == "/cancel":
            self.cancel_calls += 1
            # Mirror the fixed bridge: a cancelled attempt is not linked.
            self.linked = False
            self.qr = None
            self.set_phase("cancelled")
            return FakeResponse(200, self.snapshot())
        if method == "POST" and path == "/re-pair":
            self.repair_calls += 1
            self.repair_payloads.append(body)
            if self.repair_error:
                return FakeResponse(409, {"error": self.repair_error})
            self.linked = False
            self.issue_qr(b"repaired")
            return FakeResponse(200, self.snapshot())
        if method == "POST" and path == "/wait":
            self.wait_calls += 1
            after = (body or {}).get("after", -1)
            if self.wait_callbacks:
                callback = self.wait_callbacks.pop(0)
                callback(self)
            if after != self.sequence:
                result = self.snapshot()
                result["timed_out"] = False
                return FakeResponse(200, result)
            result = self.snapshot()
            result["timed_out"] = True
            return FakeResponse(200, result)
        if method == "POST" and path == "/typing":
            payload = body or {}
            if payload.get("state") not in {"composing", "paused"}:
                return FakeResponse(400, {"error": "invalid_request"})
            self.typing_calls.append(payload)
            return FakeResponse(200, {"ok": True})
        return FakeResponse(404, {"error": "not_found"})


class FakeClient:
    def __init__(self, base_url: str, headers: dict, server: FakeBridgeServer) -> None:
        self.base_url = base_url
        self.headers = dict(headers)
        self.server = server
        self.closed = False
        self.stream_calls: list[tuple[str, str]] = []

    async def request(self, method: str, path: str, json: dict | None = None, timeout: float | None = None):
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        assert token and token not in path, "token must never appear in a request URL"
        assert "token=" not in path and "?" not in path
        if method == "GET" and path == "/qr" and self.server.qr_requested is not None:
            self.server.qr_requested.set()
        if method == "GET" and path == "/qr" and self.server.release is not None:
            # Hold the response so a concurrent cancel can supersede it.
            try:
                await asyncio.wait_for(self.server.release.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        return self.server.handle(method, path, json, self.headers.get("Authorization"))

    @contextlib.asynccontextmanager
    async def stream(self, method: str, path: str, json: dict | None = None, timeout: float | None = None):
        # Exercise the real streaming path of the manager without a network.
        self.stream_calls.append((method, path))
        yield await self.request(method, path, json=json, timeout=timeout)

    async def aclose(self) -> None:
        self.closed = True


class FakeClientFactory:
    def __init__(self, server: FakeBridgeServer) -> None:
        self.server = server
        self.clients: list[FakeClient] = []

    def __call__(self, *, base_url: str, headers: dict) -> FakeClient:
        client = FakeClient(base_url, headers, self.server)
        self.server.expected_auth = headers.get("Authorization")
        self.clients.append(client)
        return client


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("AWS_SECRET_ACCESS_KEY", "WHATSAPP_CLOUD_TOKEN", "LEO_API_TOKEN", "NODE_OPTIONS"):
        monkeypatch.setenv(name, LEAK_TOKEN)
    monkeypatch.setenv("LEO_WHATSAPP_BRIDGE_TOKEN", LEAK_TOKEN)
    # The session root is pinned to the application repo; tests repoint that
    # pin at their private temporary repo instead of trusting a caller value.
    monkeypatch.setattr(manager_module, "_APP_REPO_ROOT", tmp_path)

    server = FakeBridgeServer()
    factory = FakeClientFactory(server)
    spawns: list[dict] = []
    processes: list[FakeProcess] = []

    def process_factory(*, argv: list[str], cwd: str, env: dict[str, str], pass_fds=()) -> FakeProcess:
        process = FakeProcess()
        spawns.append({"argv": list(argv), "cwd": cwd, "env": dict(env), "pass_fds": tuple(pass_fds)})
        processes.append(process)
        return process

    config = BridgeConfig(
        owner_id="alice",
        actor="alice",
        enabled=True,
        repo_root=tmp_path,
        bridge_dir=BRIDGE_DIR,
        startup_timeout=0.5,
        poll_interval=0.001,
        stop_timeout=0.05,
        max_wait_seconds=5.0,
    )
    manager = WhatsAppBridgeManager(config, process_factory=process_factory, http_client=factory)
    yield SimpleNamespace(
        manager=manager, server=server, factory=factory, spawns=spawns,
        processes=processes, process_factory=process_factory, config=config,
        session_dir=tmp_path / ".leo" / "whatsapp" / "alice",
        auth=manager.session_auth,
    )


async def _started(h) -> dict:
    result = await h.manager.start(owner_id="alice", auth=h.auth)
    assert h.server.start_calls == 1
    return result


# --- Enablement, validation, ownership --------------------------------------


async def test_disabled_manager_never_launches(h):
    disabled = WhatsAppBridgeManager(
        BridgeConfig(owner_id="alice", repo_root=h.config.repo_root, enabled=False),
        process_factory=h.process_factory,
        http_client=h.factory,
    )
    with pytest.raises(BridgeNotEnabled) as exc:
        await disabled.start(owner_id="alice", auth=disabled.session_auth)
    assert exc.value.code == "bridge_disabled"
    assert not h.spawns


@pytest.mark.parametrize("owner,actor", [("../escape", "alice"), ("alice", "../escape"), ("", "alice"), ("alice", "a/b")])
def test_config_rejects_unsafe_identifiers(owner, actor):
    with pytest.raises(ValueError):
        BridgeConfig(owner_id=owner, actor=actor)


async def test_owner_only_access_and_session_ownership(h):
    await _started(h)
    with pytest.raises(SessionOwnershipError):
        await h.manager.status(owner_id="bob", auth=h.auth)
    with pytest.raises(SessionOwnershipError):
        await h.manager.qr(owner_id="bob", auth=h.auth)
    owner_file = h.session_dir / "owner.json"
    record = json.loads(owner_file.read_text(encoding="utf-8"))
    record["owner_id"] = "bob"
    owner_file.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SessionOwnershipError):
        await h.manager.status(owner_id="alice", auth=h.auth)


async def test_session_directory_is_scoped_and_restrictive(h):
    await _started(h)
    assert h.session_dir.parent == h.config.session_root
    assert h.config.session_root.name == "whatsapp"
    assert h.session_dir.stat().st_mode & 0o777 == 0o700
    assert (h.session_dir / "owner.json").stat().st_mode & 0o777 == 0o600
    assert h.manager.session_dir == h.session_dir


# --- H3: per-session auth secret (owner id is not authority) -----------------


@pytest.mark.parametrize("method", ["start", "status", "qr", "cancel", "repair", "wait", "wait_for_connection", "stop"])
@pytest.mark.parametrize("auth", [None, "not-the-session-secret", "alice", "nön-ascii-secret"])
async def test_owner_id_alone_never_authorizes_a_call(h, method, auth):
    call = getattr(h.manager, method)
    kwargs = {"timeout": 0.1} if method in ("wait", "wait_for_connection") else {}
    with pytest.raises(SessionAuthError):
        await call(owner_id="alice", auth=auth, **kwargs)
    assert not h.spawns


async def test_auth_is_per_session_and_stops_are_authenticated(h):
    await _started(h)
    with pytest.raises(SessionAuthError):
        await h.manager.stop(owner_id="alice")  # no auth
    assert h.manager.is_running
    other = WhatsAppBridgeManager(h.config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(SessionAuthError):
        # Another session's own secret (and any guessed string) is not authority
        # for this manager.
        await h.manager.stop(owner_id="alice", auth=other.session_auth)
    assert h.manager.is_running
    assert other.session_auth != h.auth
    await h.manager.stop(owner_id="alice", auth=h.auth)
    assert not h.manager.is_running


async def test_qr_is_bound_to_authenticated_session_and_not_logged(h):
    await _started(h)
    h.server.issue_qr()
    qr = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert qr is not None
    with pytest.raises(SessionAuthError):
        await h.manager.qr(owner_id="alice", auth="guess")
    assert qr["data_uri"] not in " ".join(h.manager.diagnostics())
    assert qr["data_uri"] not in repr(h.manager)
    assert "no-store" in h.factory.clients[0].headers["Cache-Control"]
    source = (BRIDGE_DIR / "bridge.js").read_text(encoding="utf-8")
    assert "no-store" in source and "Vary" in source
    assert "console.log" not in source


# --- H1/H2: launch hardening -------------------------------------------------


async def test_start_uses_scrubbed_env_pinned_child_and_inherited_fd(h):
    await _started(h)
    spawn = h.spawns[0]
    env = spawn["env"]
    token = env["LEO_WHATSAPP_BRIDGE_TOKEN"]

    assert re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token)
    assert token != LEAK_TOKEN and token != h.auth
    assert env.get("LEO_WHATSAPP_ACTOR") == "alice"
    assert env.get("LEO_WHATSAPP_SESSION_DIR") == str(h.session_dir)
    assert env.get("LEO_WHATSAPP_PASSKEY_MODE") == "report"
    assert set(env) <= {
        "PATH", "HOME", "TMPDIR", "TMP", "TEMP", "SYSTEMROOT", "LANG", "LC_ALL",
        "LEO_WHATSAPP_BRIDGE_TOKEN", "LEO_WHATSAPP_BRIDGE_FD", "LEO_WHATSAPP_ACTOR",
        "LEO_WHATSAPP_SESSION_ROOT", "LEO_WHATSAPP_SESSION_DIR",
        "LEO_WHATSAPP_BRIDGE_PORT", "LEO_WHATSAPP_PASSKEY_MODE",
    }
    for secret in ("AWS_SECRET_ACCESS_KEY", "WHATSAPP_CLOUD_TOKEN", "LEO_API_TOKEN", "NODE_OPTIONS"):
        assert secret not in env
        assert LEAK_TOKEN not in env.values()
    assert token not in " ".join(spawn["argv"])
    assert token not in h.factory.clients[0].base_url
    assert h.factory.clients[0].headers["Authorization"] == f"Bearer {token}"
    assert spawn["cwd"] == str(BRIDGE_DIR)
    assert spawn["argv"][1].endswith("bridge.js")
    # The listener is kept by the parent and its FD (not the port) is handed
    # to the child. Port squatter races are impossible by construction.
    listener = h.manager._listener
    assert listener is not None and listener.fileno() >= 0
    assert spawn["pass_fds"] == (listener.fileno(),)
    assert env["LEO_WHATSAPP_BRIDGE_FD"] == str(listener.fileno())
    assert listener.getsockname()[0] == "127.0.0.1"
    assert env["LEO_WHATSAPP_BRIDGE_PORT"] == str(listener.getsockname()[1])
    assert not hasattr(manager_module, "_free_port")

    first = token
    await h.manager.stop(owner_id="alice", auth=h.auth)
    assert listener.fileno() == -1
    second_manager = WhatsAppBridgeManager(h.config, process_factory=h.process_factory, http_client=h.factory)
    await second_manager.start(owner_id="alice", auth=second_manager.session_auth)
    assert h.spawns[1]["env"]["LEO_WHATSAPP_BRIDGE_TOKEN"] != first
    await second_manager.stop(owner_id="alice", auth=second_manager.session_auth)


async def test_squatter_cannot_claim_the_parent_listener(h):
    await _started(h)
    port = h.manager._listener.getsockname()[1]
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            squatter.bind(("127.0.0.1", port))
    finally:
        squatter.close()
    # The fake child never saw a request before FD ownership was verified.
    assert h.server.calls == [("GET", "/health", None), ("POST", "/start", {}), ("GET", "/status", None)]


async def test_no_token_is_sent_if_child_fd_verification_fails(h, monkeypatch):
    def bad_verify(*_args, **_kwargs):
        raise BridgeUnavailable("The bridge process does not own the listener.")

    monkeypatch.setattr(h.manager, "_verify_child_listener", bad_verify)
    with pytest.raises(BridgeUnavailable):
        await h.manager.start(owner_id="alice", auth=h.auth)
    assert h.server.calls == []
    assert h.manager._client is None
    assert h.manager._listener is None
    # The child was spawned before verification: it must not be orphaned.
    assert h.processes[0].terminate_calls == 1
    assert h.processes[0].wait_calls >= 1
    assert h.manager._token == ""


@pytest.mark.parametrize("failure_point", ["verify", "client", "task"])
async def test_any_post_spawn_start_failure_shuts_everything_down(h, monkeypatch, failure_point):
    def boom(*_args, **_kwargs):
        raise RuntimeError("injected-start-failure")

    if failure_point == "verify":
        monkeypatch.setattr(h.manager, "_verify_child_listener", boom)
    elif failure_point == "client":
        monkeypatch.setattr(h.manager, "_make_client", boom)
    else:
        original_create_task = asyncio.create_task

        def failing_create_task(coro, *_args, **_kwargs):
            if str(getattr(coro, "__qualname__", "")).endswith("_drain_stream"):
                coro.close()
                raise RuntimeError("injected-start-failure")
            return original_create_task(coro, *_args, **_kwargs)

        monkeypatch.setattr(manager_module.asyncio, "create_task", failing_create_task)

    with pytest.raises(RuntimeError, match="injected-start-failure"):
        await h.manager.start(owner_id="alice", auth=h.auth)
    assert len(h.processes) == 1
    process = h.processes[0]
    assert process.terminate_calls == 1
    assert process.wait_calls >= 1
    assert h.manager._proc is None
    assert h.manager._client is None
    assert h.manager._listener is None
    assert h.manager._token == ""
    assert h.server.calls == []


@pytest.mark.parametrize("binary", ["/bin/sh", "/etc/passwd", "/usr/bin/env", "../node"])
async def test_caller_supplied_node_binary_is_rejected(h, binary):
    config = BridgeConfig(
        owner_id="alice", actor="alice", enabled=True, repo_root=h.config.repo_root,
        node_binary=binary,
    )
    manager = WhatsAppBridgeManager(config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(BridgeUnavailable):
        await manager.start(owner_id="alice", auth=manager.session_auth)
    assert not h.spawns


@pytest.mark.parametrize("binary", ["/bin/sh", "/etc/passwd"])
async def test_pinned_non_node_paths_are_still_rejected(h, binary):
    config = BridgeConfig(
        owner_id="alice", actor="alice", enabled=True, repo_root=h.config.repo_root,
        node_binary=binary, node_path_allowlist=(binary,),
    )
    manager = WhatsAppBridgeManager(config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(BridgeUnavailable):
        await manager.start(owner_id="alice", auth=manager.session_auth)
    assert not h.spawns


async def test_non_executable_fake_node_is_rejected(h, tmp_path: Path):
    fake = tmp_path / "node"
    fake.write_bytes(b"\x7fELF" + b"Node.js" + b"NODE_MODULE_VERSION" + b"\x00" * 64)
    os.chmod(fake, 0o644)
    config = BridgeConfig(
        owner_id="alice", actor="alice", enabled=True, repo_root=h.config.repo_root,
        node_binary=str(fake), node_path_allowlist=(str(fake),),
    )
    manager = WhatsAppBridgeManager(config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(BridgeUnavailable):
        await manager.start(owner_id="alice", auth=manager.session_auth)
    assert not h.spawns


async def test_which_spoofing_to_non_node_binary_is_rejected(h, monkeypatch):
    monkeypatch.setattr(manager_module.shutil, "which", lambda _name: "/bin/sh")
    with pytest.raises(BridgeUnavailable):
        await h.manager.start(owner_id="alice", auth=h.auth)
    assert not h.spawns


async def test_executable_non_node_binary_is_rejected_by_version_probe(h):
    """Embedded 'Node.js' strings must never be enough to pin a binary.

    sys.executable is a real executable native binary, so the old marker
    scan accepted it; executing it with --version proves it is not Node.
    """
    fake = Path(sys.executable).resolve()
    config = BridgeConfig(
        owner_id="alice", actor="alice", enabled=True, repo_root=h.config.repo_root,
        node_binary=str(fake), node_path_allowlist=(str(fake),),
    )
    manager = WhatsAppBridgeManager(config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(BridgeUnavailable):
        await manager.start(owner_id="alice", auth=manager.session_auth)
    assert not h.spawns
    source = (BRIDGE_DIR / "manager.py").read_text(encoding="utf-8")
    assert "_NODE_IDENTITY_MARKERS" not in source
    assert not hasattr(manager_module, "_NODE_IDENTITY_MARKERS")
    assert '"--version"' in source and "_NODE_VERSION_RE" in source


async def test_pinned_node_path_with_real_identity_is_allowed(h):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    resolved = str(Path(node).resolve())
    config = BridgeConfig(
        owner_id="alice", actor="alice", enabled=True, repo_root=h.config.repo_root,
        node_binary=resolved, node_path_allowlist=(resolved,),
    )
    manager = WhatsAppBridgeManager(config, process_factory=h.process_factory, http_client=h.factory)
    await manager.start(owner_id="alice", auth=manager.session_auth)
    assert h.spawns[0]["argv"][0] == resolved
    await manager.stop(owner_id="alice", auth=manager.session_auth)


async def test_arbitrary_bridge_dir_and_script_digest_are_rejected(h, monkeypatch):
    config = BridgeConfig(
        owner_id="alice", actor="alice", enabled=True, repo_root=h.config.repo_root,
        bridge_dir=Path("/etc"),
    )
    manager = WhatsAppBridgeManager(config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(BridgeError) as exc:
        await manager.start(owner_id="alice", auth=manager.session_auth)
    assert exc.value.code == "invalid_bridge_path"
    assert not h.spawns

    monkeypatch.setattr(manager_module, "_BRIDGE_JS_SHA256", "0" * 64)
    with pytest.raises(BridgeUnavailable):
        await h.manager.start(owner_id="alice", auth=h.auth)
    assert not h.spawns


async def test_session_root_symlinks_and_credential_modes(h):
    await _started(h)
    cred = h.session_dir / "creds.json"
    cred.write_text("{}", encoding="utf-8")
    os.chmod(cred, 0o644)
    await h.manager.stop(owner_id="alice", auth=h.auth)

    manager = WhatsAppBridgeManager(h.config, process_factory=h.process_factory, http_client=h.factory)
    await manager.start(owner_id="alice", auth=manager.session_auth)
    assert cred.stat().st_mode & 0o777 == 0o600
    await manager.stop(owner_id="alice", auth=manager.session_auth)

    link = h.session_dir / "creds-link.json"
    link.symlink_to(cred)
    manager = WhatsAppBridgeManager(h.config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(BridgeError) as exc:
        await manager.start(owner_id="alice", auth=manager.session_auth)
    assert exc.value.code == "invalid_session_path"
    link.unlink()

    root_link = h.config.repo_root / ".leo" / "whatsapp-link"
    root_link.symlink_to(h.config.session_root, target_is_directory=True)
    monkeypatched = BridgeConfig(owner_id="alice", actor="alice", enabled=True, repo_root=h.config.repo_root)
    manager = WhatsAppBridgeManager(monkeypatched, process_factory=h.process_factory, http_client=h.factory)
    # A symlinked *actor* directory is rejected even though the root is pinned.
    (h.config.session_root / "bob").symlink_to(h.session_dir, target_is_directory=True)
    bob_config = BridgeConfig(owner_id="bob", actor="bob", enabled=True, repo_root=h.config.repo_root)
    bob_manager = WhatsAppBridgeManager(bob_config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(BridgeError) as exc:
        await bob_manager.start(owner_id="bob", auth=bob_manager.session_auth)
    assert exc.value.code == "invalid_session_path"


# --- Secret hygiene and child launch ----------------------------------------


async def test_start_failure_sanitizes_child_output_and_cleans_up(h):
    h.processes.clear()
    original_factory = h.process_factory

    def failing_factory(*, argv, cwd, env, pass_fds=()):
        process = original_factory(argv=argv, cwd=cwd, env=env, pass_fds=pass_fds)
        process.returncode = 1
        process.stderr.push(f"token={env['LEO_WHATSAPP_BRIDGE_TOKEN']}\n")
        return process

    manager = WhatsAppBridgeManager(h.config, process_factory=failing_factory, http_client=h.factory)
    with pytest.raises(BridgeUnavailable):
        await manager.start(owner_id="alice", auth=manager.session_auth)
    diagnostics = " ".join(manager.diagnostics())
    assert "[REDACTED]" in diagnostics
    assert LEAK_TOKEN not in diagnostics
    assert manager._token == ""
    assert manager._client is None
    assert manager._listener is None
    assert not manager.is_running


async def test_status_and_events_never_forward_secrets(h):
    await _started(h)
    token = h.spawns[0]["env"]["LEO_WHATSAPP_BRIDGE_TOKEN"]
    h.server.issue_qr()
    line = json.dumps({
        "ts": 1, "event": "qr", "phase": "waiting_for_scan", "sequence": 7,
        "token": token, "qr": _data_uri(), "message": LEAK_TOKEN,
    })
    h.processes[0].stdout.push(line)
    await asyncio.sleep(0.01)

    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["state"] == "waiting_for_scan"
    assert status["qr_available"] is True
    assert status["user"]["id"] == "15551234567@s.whatsapp.net"
    assert status["user"]["name"] == "OwnerName"
    assert "owner_token" not in status and "qr_raw" not in status
    assert LEAK_TOKEN not in repr(status)
    assert "owner_token" not in repr(h.manager)
    assert token not in repr(h.manager)

    events = h.manager.events
    assert events and events[0]["event"] == "qr"
    assert set(events[0]) <= {"event", "phase", "state", "error", "code", "sequence", "rotation", "has_signer"}
    assert LEAK_TOKEN not in repr(events)
    assert token not in repr(events)
    assert h.auth not in repr(events)
    assert _data_uri() not in repr(events)
    assert LEAK_TOKEN not in " ".join(h.manager.diagnostics())


async def test_long_child_lines_are_redacted_before_truncation(h):
    await _started(h)
    token = h.spawns[0]["env"]["LEO_WHATSAPP_BRIDGE_TOKEN"]
    qr = _data_uri()
    line = "x" * 600 + " token:" + token + " qr:" + qr
    h.processes[0].stderr.push(line)
    await asyncio.sleep(0.01)
    diagnostics = " ".join(h.manager.diagnostics())
    assert token not in diagnostics
    assert qr not in diagnostics
    assert "[REDACTED]" in diagnostics and "[QR_REDACTED]" in diagnostics


# --- QR, wait, cancel, re-pair ----------------------------------------------


async def test_qr_lifecycle_rotation_and_expiry(h):
    await _started(h)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    h.server.issue_qr(b"first")
    first = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert first["rotation"] == 1
    assert first["data_uri"].startswith("data:image/png;base64,")
    assert first["expires_in_ms"] > 0
    assert (await h.manager.status(owner_id="alice", auth=h.auth))["qr_available"] is True

    h.server.expire_qr()
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    assert (await h.manager.status(owner_id="alice", auth=h.auth))["qr_available"] is False

    h.server.issue_qr(b"second")
    second = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert second["rotation"] == 2
    assert second["data_uri"] != first["data_uri"]


async def test_status_never_publishes_a_superseded_qr(h):
    await _started(h)
    h.server.issue_qr(b"first")  # rotation 1, distinct image
    first = await h.manager.qr(owner_id="alice", auth=h.auth)
    h.server.issue_qr(b"second")  # rotation 2, distinct image
    second = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert first["rotation"] == 1 and second["rotation"] == 2

    # A delayed /status response carrying the older QR must be suppressed just
    # like the equivalent /qr response, and qr_available must follow suit.
    h.server.status_overrides["qr"] = {
        "data_uri": first["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": first["rotation"],
    }
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"] is None
    assert status["qr_available"] is False

    h.server.status_overrides["qr"] = {
        "data_uri": second["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": second["rotation"],
    }
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"]["rotation"] == 2
    assert status["qr_available"] is True


async def test_invalid_and_oversized_qr_payloads_are_rejected(h):
    await _started(h)
    h.server.qr = {"data_uri": "javascript:alert(1)", "expires_at": time.time() + 10, "rotation": 1}
    with pytest.raises(BridgeError) as exc:
        await h.manager.qr(owner_id="alice", auth=h.auth)
    assert exc.value.code == "invalid_qr"
    h.server.qr = {"data_uri": _data_uri() + "A" * 70_000, "expires_at": time.time() + 10, "rotation": 2}
    with pytest.raises(BridgeError) as exc:
        await h.manager.qr(owner_id="alice", auth=h.auth)
    assert exc.value.code == "invalid_qr"


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), -float("inf"), 1e300, True, "1234", None])
async def test_non_finite_or_invalid_qr_expiry_is_rejected(h, expiry):
    await _started(h)
    h.server.qr = {"data_uri": _data_uri(), "expires_at": expiry, "rotation": 1}
    with pytest.raises(BridgeError) as exc:
        await h.manager.qr(owner_id="alice", auth=h.auth)
    assert exc.value.code == "invalid_qr"


async def test_qr_expiry_accepts_seconds_and_milliseconds(h):
    await _started(h)
    h.server.qr = {"data_uri": _data_uri(b"seconds"), "expires_at": time.time() + 30, "rotation": 1}
    seconds = await h.manager.qr(owner_id="alice", auth=h.auth)
    h.server.qr = {"data_uri": _data_uri(b"millis"), "expires_at": (time.time() + 30) * 1000, "rotation": 2}
    millis = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert 25_000 < seconds["expires_in_ms"] <= 30_000
    assert 25_000 < millis["expires_in_ms"] <= 30_000


async def test_dead_child_is_reported_as_error(h):
    await _started(h)
    h.processes[0].returncode = 1
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["state"] == "error"
    assert status["error"] == "bridge_exited"
    assert status["running"] is False
    assert status["linked"] is False


async def test_wait_reaches_terminal_state_and_times_out(h):
    await _started(h)

    def connect(server: FakeBridgeServer) -> None:
        server.linked = True
        server.set_phase("connected")

    h.server.wait_callbacks.append(connect)
    result = await h.manager.wait(owner_id="alice", timeout=2, auth=h.auth)
    assert result["state"] == "connected"
    assert result["linked"] is True

    with pytest.raises(BridgeTimeout) as exc:
        await h.manager.wait(owner_id="alice", states={"logged_out"}, timeout=0.2, auth=h.auth)
    assert exc.value.code == "bridge_timeout"

    with pytest.raises(BridgeError):
        await h.manager.wait(owner_id="alice", timeout=0, auth=h.auth)


async def test_cancel_keeps_credentials_and_clears_qr(h):
    await _started(h)
    h.server.issue_qr()
    h.server.linked = True
    h.server.set_phase("connected")
    result = await h.manager.cancel(owner_id="alice", auth=h.auth)
    assert result["state"] == "cancelled"
    assert result["qr"] is None
    assert h.server.cancel_calls == 1
    assert (h.session_dir / "owner.json").is_file()
    assert not list(h.session_dir.glob("creds.json"))


async def test_repair_starts_fresh_qr_and_forget_is_explicit(h):
    await _started(h)
    h.server.issue_qr()
    h.server.linked = True
    h.server.set_phase("connected")
    result = await h.manager.repair(owner_id="alice", auth=h.auth, forget=True)
    assert result["rotation"] == 2
    assert result["qr_available"] is True
    assert result["linked"] is False
    assert h.server.repair_payloads == [{"forget": True}]
    with pytest.raises(BridgeError):
        await h.manager.repair(owner_id="alice", auth=h.auth, forget="yes")  # type: ignore[arg-type]


# --- Cross-generation QR identity -------------------------------------------


async def test_cancel_rejects_cross_generation_qr(h):
    """A QR published before cancel stays stale even with an inflated rotation."""
    await _started(h)
    h.server.issue_qr(b"old")
    old = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert old is not None

    await h.manager.cancel(owner_id="alice", auth=h.auth)
    # A delayed render from the cancelled generation arrives afterwards. The
    # child may report a larger rotation; generation identity still wins.
    stale = {
        "data_uri": old["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": 99,
        "generation": 1,
    }
    h.server.qr = dict(stale)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    h.server.status_overrides["qr"] = dict(stale)
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"] is None
    assert status["qr_available"] is False


async def test_typing_publishes_presence_for_one_chat(h):
    await _started(h)
    await h.manager.typing(owner_id="alice", to="15551234567@s.whatsapp.net", auth=h.auth)
    assert h.server.typing_calls == [
        {"to": "15551234567@s.whatsapp.net", "state": "composing"},
    ]
    await h.manager.typing(
        owner_id="alice", to="15551234567@s.whatsapp.net", state="paused", auth=h.auth,
    )
    assert h.server.typing_calls[-1]["state"] == "paused"


async def test_typing_rejects_an_unknown_state_and_a_bad_target(h):
    await _started(h)
    with pytest.raises(BridgeError) as exc:
        await h.manager.typing(owner_id="alice", to="15551234567@s.whatsapp.net", state="idle", auth=h.auth)
    assert exc.value.code == "invalid_request"
    with pytest.raises(BridgeError) as exc:
        await h.manager.typing(owner_id="alice", to="not-a-jid", auth=h.auth)
    assert exc.value.code == "invalid_target"
    assert h.server.typing_calls == []


async def test_repair_surfaces_explicit_credential_drain_failure(h):
    await _started(h)
    h.server.repair_error = "creds_drain_failed"
    with pytest.raises(BridgeError) as exc:
        await h.manager.repair(owner_id="alice", auth=h.auth, forget=True)
    assert exc.value.code == "creds_drain_failed"
    assert exc.value.status_code == 409
    assert h.server.repair_payloads == [{"forget": True}]


async def test_repair_rejects_cross_generation_qr(h):
    """Re-pair starts a new generation; old renders cannot be resurrected."""
    await _started(h)
    h.server.issue_qr(b"old")
    old = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert old is not None

    result = await h.manager.repair(owner_id="alice", auth=h.auth)
    assert result["qr_available"] is True
    stale = {
        "data_uri": old["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": 100,
    }
    h.server.qr = dict(stale)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    h.server.status_overrides["qr"] = dict(stale)
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"] is None and status["qr_available"] is False

    # A genuinely fresh render in the new generation is still published.
    h.server.issue_qr(b"fresh")
    fresh = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert fresh is not None and fresh["data_uri"] != old["data_uri"]


async def test_published_image_is_single_use_with_inflated_rotation(h):
    """F1: the same image must never be republished with a higher rotation."""
    await _started(h)
    h.server.issue_qr(b"published")
    published = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert published is not None and published["rotation"] == 1

    replay = {
        "data_uri": published["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": 3,
        # The child may claim any generation/rotation it likes; neither may
        # move the manager's authoritative generation watermark.
        "generation": 99,
    }
    h.server.qr = dict(replay)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    h.server.status_overrides["qr"] = dict(replay)
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"] is None and status["qr_available"] is False
    assert h.manager._generation == 1

    # A genuinely new image in the same generation is still published.
    h.server.issue_qr(b"next")
    fresh = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert fresh is not None and fresh["rotation"] == 2


async def test_cross_generation_replay_after_identity_cache_eviction(h):
    """F2: consumed image digests never age out of the membership set."""
    await _started(h)
    # Publish far more than the old 64-entry identity cache could hold.
    for index in range(1, 66):
        h.server.qr = {
            "data_uri": _data_uri(f"image-{index}".encode("ascii")),
            "expires_at": time.time() + 30,
            "rotation": index,
        }
        published = await h.manager.qr(owner_id="alice", auth=h.auth)
        assert published is not None and published["rotation"] == index
    assert len(h.manager._qr_consumed) == 65

    await h.manager.cancel(owner_id="alice", auth=h.auth)
    assert h.manager._generation == 2
    # image-1 fell out of the old fixed-size cache. Even with a forged
    # generation and a huge rotation it must stay rejected.
    replay = {
        "data_uri": _data_uri(b"image-1"),
        "expires_at": time.time() + 30,
        "rotation": 1000,
        "generation": 1,
    }
    h.server.qr = dict(replay)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    h.server.status_overrides["qr"] = dict(replay)
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"] is None and status["qr_available"] is False

    # The new generation can still publish a never-seen image.
    h.server.qr = {
        "data_uri": _data_uri(b"after-cancel"),
        "expires_at": time.time() + 30,
        "rotation": 1,
    }
    fresh = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert fresh is not None and fresh["data_uri"] == _data_uri(b"after-cancel")


async def test_forged_maximal_rotation_cannot_poison_publication_watermark(h):
    """F3: child rotation is bounded metadata, never the manager's watermark."""
    await _started(h)
    forged_rotation = manager_module.MAX_FORWARDED_INT
    assert forged_rotation == 2147483647
    h.server.qr = {
        "data_uri": _data_uri(b"forged-max"),
        "expires_at": time.time() + 30,
        "rotation": forged_rotation,
        "generation": 10**9,  # child claim: never authoritative
    }
    forged = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert forged is not None and forged["rotation"] == forged_rotation
    assert h.manager._generation == 1
    assert h.manager._qr_publication == 1
    assert h.manager._qr_identity == (1, 1)

    # Ordinary, small rotations must still publish: the manager's own
    # publication sequence, not the child's rotation, is the ordering key.
    for index, marker in enumerate((b"legit-one", b"legit-two", b"legit-three"), start=2):
        h.server.qr = {
            "data_uri": _data_uri(marker),
            "expires_at": time.time() + 30,
            "rotation": index,
            "generation": 1,
        }
        published = await h.manager.qr(owner_id="alice", auth=h.auth)
        assert published is not None, index
        assert published["rotation"] == index
    assert h.manager._qr_publication == 4
    assert h.manager._qr_identity == (1, 4)

    # The forged image itself remains single-use even with a fresh generation
    # claim and the same maximal rotation, on both read paths.
    replay = {
        "data_uri": forged["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": forged_rotation,
        "generation": 999,
    }
    h.server.qr = dict(replay)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    h.server.status_overrides["qr"] = dict(replay)
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"] is None and status["qr_available"] is False

    # Fresh images still publish through the status path after the forgery.
    h.server.status_overrides["qr"] = {
        "data_uri": _data_uri(b"after-forgery"),
        "expires_at": time.time() + 30,
        "rotation": 1,
    }
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr_available"] is True
    assert status["qr"]["rotation"] == 1


async def test_consumed_ledger_resets_per_generation_and_rejects_replays(h):
    """F4: exhausting a generation's ledger must not permanently block QR."""
    await _started(h)
    cap = manager_module.MAX_CONSUMED_QR_HASHES
    first_uri = _data_uri(b"bulk-0")
    for index in range(cap):
        h.server.qr = {
            "data_uri": _data_uri(f"bulk-{index}".encode("ascii")),
            "expires_at": time.time() + 30,
            "rotation": index + 1,
        }
        assert await h.manager.qr(owner_id="alice", auth=h.auth) is not None, index
    assert len(h.manager._qr_consumed) == cap

    # At the cap the manager fails closed for novel images in this generation.
    for index in range(2):
        h.server.qr = {
            "data_uri": _data_uri(f"overflow-{index}".encode("ascii")),
            "expires_at": time.time() + 30,
            "rotation": cap + 1 + index,
        }
        assert await h.manager.qr(owner_id="alice", auth=h.auth) is None

    # A legitimate lifecycle advance archives the ledger and frees the budget.
    await h.manager.cancel(owner_id="alice", auth=h.auth)
    assert h.manager._generation == 2
    assert not h.manager._qr_consumed
    assert len(h.manager._qr_seen) == cap

    h.server.qr = {
        "data_uri": _data_uri(b"next-generation"),
        "expires_at": time.time() + 30,
        "rotation": 1,
    }
    fresh = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert fresh is not None and fresh["data_uri"] == _data_uri(b"next-generation")

    # Same-generation replay of the live image with a maximal rotation.
    same_gen_replay = {
        "data_uri": fresh["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": manager_module.MAX_FORWARDED_INT,
    }
    h.server.qr = dict(same_gen_replay)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None

    # Cross-generation replay of an image from the exhausted generation.
    cross_gen_replay = {
        "data_uri": first_uri,
        "expires_at": time.time() + 30,
        "rotation": manager_module.MAX_FORWARDED_INT,
        "generation": 99,
    }
    h.server.qr = dict(cross_gen_replay)
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None
    h.server.status_overrides["qr"] = dict(cross_gen_replay)
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr"] is None and status["qr_available"] is False


async def test_image_published_via_status_is_single_use_in_qr(h):
    """Both read paths share one consumed-image ledger."""
    await _started(h)
    h.server.issue_qr(b"status-only")
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["qr_available"] is True
    h.server.qr = {
        "data_uri": status["qr"]["data_uri"],
        "expires_at": time.time() + 30,
        "rotation": 42,
    }
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None


# --- F1: the shipped Node bridge must drain credential saves on forget -------

_BRIDGE_HARNESS_PRELUDE = r"""
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

// Minimal in-memory stand-ins: the harness exercises the REAL cancelAttempt,
// closeSocket, /re-pair and clearAuthFiles code from bridge.js, with no express,
// qrcode, Baileys install, network or listener involved.
const routes = new Map();
const sockets = [];
const stubApp = {
  disable() {},
  use(..._args) {},
  get(route, handler) { routes.set('GET ' + route, handler); },
  post(route, ...handlers) { routes.set('POST ' + route, handlers[handlers.length - 1]); },
  listen(_opts, cb) { if (cb) cb(); return { on() {}, close(cb) { if (cb) cb(); } }; },
};
function express() { return stubApp; }
express.json = () => (_req, _res, next) => next();
const QRCode = { toDataURL: async () => 'data:image/png;base64,' + 'A'.repeat(128) };

function makeEv() {
  const handlers = new Map();
  const captured = new Map();
  return {
    on(name, fn) {
      if (!handlers.has(name)) handlers.set(name, new Set());
      handlers.get(name).add(fn);
      if (!captured.has(name)) captured.set(name, new Set());
      captured.get(name).add(fn);
    },
    removeAllListeners(name) { handlers.get(name)?.clear(); },
    emit(name, ...args) { for (const fn of handlers.get(name) || []) fn(...args); },
    // A handler captured before teardown can still fire late.
    fireCaptured(name, ...args) { for (const fn of captured.get(name) || []) fn(...args); },
  };
}
const baileysStub = {
  fetchLatestBaileysVersion: async () => ({ version: [1, 2, 3] }),
  useMultiFileAuthState: async () => ({
    state: {},
    saveCreds: async () => {
      await fsp.writeFile(path.join(sessionDir, 'creds.json'), '{"saved":true}');
    },
  }),
  makeWASocket: () => {
    const sock = { ev: makeEv(), user: {}, end() {} };
    sockets.push(sock);
    return sock;
  },
};
"""

_BRIDGE_HARNESS_EPILOGUE = r"""
const credsPath = path.join(sessionDir, 'creds.json');
async function harnessFileExists(target) {
  try { await fsp.access(target); return true; } catch { return false; }
}
const fakeSock = { ev: makeEv(), end() {}, user: {} };
socket = fakeSock;
await fsp.writeFile(credsPath, '{"old":true}');
// A queued save finishes only AFTER /re-pair has begun: without a drained
// cancelAttempt it recreates creds.json after clearAuthFiles().
credsChain = new Promise((resolve) => setTimeout(resolve, 40))
  .then(() => fsp.writeFile(credsPath, '{"delayed":true}'));
const res = { status(c) { this.code = c; return this; }, json(p) { this.payload = p; return this; } };
const next = (err) => { throw new Error('unexpected error middleware: ' + (err && err.message)); };
await routes.get('POST /re-pair')({ body: { forget: true } }, res, next);
if (await harnessFileExists(credsPath)) throw new Error('delayed save resurrected creds.json');
// A handler captured before teardown firing late must not write either.
fakeSock.ev.fireCaptured('creds.update');
await new Promise((resolve) => setTimeout(resolve, 60));
if (await harnessFileExists(credsPath)) throw new Error('late event resurrected creds.json');
process.stdout.write('NO_CREDS_AFTER_FORGET\n');
// The new socket's own saves must still work (writes are not stuck off).
const newSock = sockets[sockets.length - 1];
newSock.ev.emit('creds.update');
await credsChain;
if (!(await harnessFileExists(credsPath))) throw new Error('new socket cannot save credentials');
process.stdout.write('RESULT_OK\n');
"""


def test_repair_forget_drains_delayed_credential_saves(tmp_path: Path):
    """Delayed saveCreds() must not resurrect credentials after re-pair forget.

    Runs the real bridge.js control flow under Node with stubbed npm modules.
    The old synchronous cancelAttempt() fails this test: the delayed write
    lands after clearAuthFiles() and creds.json survives the forget.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    session_dir = tmp_path / "alice"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "owner.json").write_text(json.dumps({"owner_id": "alice"}), encoding="utf-8")

    source = (BRIDGE_DIR / "bridge.js").read_text(encoding="utf-8")
    if source.startswith("#!"):
        source = source.split("\n", 1)[1]
    source = re.sub(r"^import .*;\s*$", "", source, flags=re.MULTILINE)
    source = source.replace("await import('@whiskeysockets/baileys')", "baileysStub")
    assert "await import(" not in source
    harness = tmp_path / "bridge_harness.mjs"
    harness.write_text(_BRIDGE_HARNESS_PRELUDE + source + _BRIDGE_HARNESS_EPILOGUE, encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "LEO_WHATSAPP_BRIDGE_TOKEN": "a" * 40,
        "LEO_WHATSAPP_ACTOR": "alice",
        "LEO_WHATSAPP_SESSION_ROOT": str(tmp_path),
        "LEO_WHATSAPP_SESSION_DIR": str(session_dir),
        "LEO_WHATSAPP_BRIDGE_FD": "3",
        "LEO_WHATSAPP_BRIDGE_PORT": "12345",
    }
    result = subprocess.run(
        [node, str(harness)], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "NO_CREDS_AFTER_FORGET" in result.stdout, result.stdout + result.stderr
    assert "RESULT_OK" in result.stdout, result.stdout + result.stderr


_BRIDGE_HARNESS_REQUEUE_EPILOGUE = r"""
const credsPath = path.join(sessionDir, 'creds.json');
let capturedAuth = null;
const originalAuth = baileysStub.useMultiFileAuthState;
baileysStub.useMultiFileAuthState = async (dir) => {
  capturedAuth = await originalAuth(dir);
  return capturedAuth;
};
await openSocket(false);
if (!capturedAuth) throw new Error('auth state was not captured');
await fsp.writeFile(credsPath, '{"old":true}');
let requeues = 0;
// A broken saveCreds keeps rescheduling itself. Every write goes through the
// guarded authState.saveCreds, so fencing must silence even this runaway.
const runaway = () => {
  if (requeues >= 200) return;  // bounded only so the harness itself can exit
  credsChain = credsChain.then(async () => {
    requeues += 1;
    await capturedAuth.saveCreds();  // guarded: a no-op once writes are fenced
    runaway();
  });
};
runaway();
const res = { status(c) { this.code = c; return this; }, json(p) { this.payload = p; return this; } };
let nextErr = null;
const next = (err) => { nextErr = err; };
await routes.get('POST /re-pair')({ body: { forget: true } }, res, next);
if (!nextErr) throw new Error('runaway credential chain reported success');
if (nextErr.bridgeErrorCode !== 'creds_drain_failed') throw new Error('wrong error: ' + nextErr);
if (res.code !== undefined || res.payload !== undefined) throw new Error('unexpected success response');
if (credsWriteHandler !== null) throw new Error('runaway handler was not replaced');
if (requeues < 64) throw new Error('drain limit was not exercised');
const before = await fsp.readFile(credsPath, 'utf8');
await new Promise((resolve) => setTimeout(resolve, 120));
const after = await fsp.readFile(credsPath, 'utf8');
if (before !== after) throw new Error('credential file changed after the drain failed');
if (credsWriteHandler !== null) throw new Error('handler was restored after the fence');
process.stdout.write('REQUEUE_FAIL_CLOSED\n');
"""


def test_repair_forget_fails_closed_on_self_requeuing_credential_saves(tmp_path: Path):
    """A chain that never settles must not report a successful forget."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    session_dir = tmp_path / "alice"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "owner.json").write_text(json.dumps({"owner_id": "alice"}), encoding="utf-8")

    source = (BRIDGE_DIR / "bridge.js").read_text(encoding="utf-8")
    if source.startswith("#!"):
        source = source.split("\n", 1)[1]
    source = re.sub(r"^import .*;\s*$", "", source, flags=re.MULTILINE)
    source = source.replace("await import('@whiskeysockets/baileys')", "baileysStub")
    assert "await import(" not in source
    harness = tmp_path / "bridge_requeue_harness.mjs"
    harness.write_text(_BRIDGE_HARNESS_PRELUDE + source + _BRIDGE_HARNESS_REQUEUE_EPILOGUE, encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "LEO_WHATSAPP_BRIDGE_TOKEN": "a" * 40,
        "LEO_WHATSAPP_ACTOR": "alice",
        "LEO_WHATSAPP_SESSION_ROOT": str(tmp_path),
        "LEO_WHATSAPP_SESSION_DIR": str(session_dir),
        "LEO_WHATSAPP_BRIDGE_FD": "3",
        "LEO_WHATSAPP_BRIDGE_PORT": "12345",
    }
    result = subprocess.run(
        [node, str(harness)], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "REQUEUE_FAIL_CLOSED" in result.stdout, result.stdout + result.stderr


async def test_malicious_linked_claim_outside_connected_phase_is_dropped(h):
    await _started(h)
    for phase in ("cancelled", "waiting_for_scan", "error", "idle"):
        h.server.phase = phase
        h.server.linked = True
        status = await h.manager.status(owner_id="alice", auth=h.auth)
        assert status["linked"] is False, phase
    h.server.phase = "connected"
    assert (await h.manager.status(owner_id="alice", auth=h.auth))["linked"] is True


async def test_qr_result_from_superseded_generation_is_dropped(h):
    await _started(h)
    h.server.issue_qr()
    h.server.release = asyncio.Event()
    h.server.qr_requested = asyncio.Event()
    task = asyncio.create_task(h.manager.qr(owner_id="alice", auth=h.auth))
    await asyncio.wait_for(h.server.qr_requested.wait(), timeout=1.0)
    # Cancel while the QR read is in flight: generation changes, so the
    # in-flight result is stale and must never reach the caller.
    await h.manager.cancel(owner_id="alice", auth=h.auth)
    h.server.release.set()
    assert await asyncio.wait_for(task, timeout=1.0) is None


async def test_terminal_states_are_reported(h):
    await _started(h)
    h.server.passkey_required = {"has_signer": True}
    h.server.set_phase("passkey_required")
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["state"] == "passkey_required"
    assert status["passkey_required"] == {"has_signer": True}

    h.server.passkey_required = None
    h.server.linked = False
    h.server.set_phase("logged_out", error="logged_out")
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["state"] == "logged_out"
    assert status["error"] == "logged_out"
    assert status["linked"] is False


async def test_auth_failure_is_generic(h):
    await _started(h)
    wrong = "Bearer wrong-token-value-1234567890"
    h.factory.clients[0].headers["Authorization"] = wrong
    with pytest.raises(BridgeAuthError) as exc:
        await h.manager.status(owner_id="alice", auth=h.auth)
    assert wrong not in str(exc.value)
    assert LEAK_TOKEN not in str(exc.value)
    h.server.force_unauthorized = True
    h.factory.clients[0].headers["Authorization"] = f"Bearer {h.spawns[0]['env']['LEO_WHATSAPP_BRIDGE_TOKEN']}"
    with pytest.raises(BridgeAuthError):
        await h.manager.qr(owner_id="alice", auth=h.auth)


# --- M3: bounded streaming responses and forwarded fields -------------------


async def test_oversized_streamed_response_is_rejected(h):
    await _started(h)
    huge = "A" * (manager_module.MAX_RESPONSE_BYTES + 1)
    h.server.next_payload = {"phase": "connected", "started_at": huge}
    h.server.chunks = [b"x" * (manager_module.MAX_RESPONSE_BYTES // 2 + 1)] * 8
    with pytest.raises(BridgeError) as exc:
        await h.manager.status(owner_id="alice", auth=h.auth)
    assert exc.value.code == "response_too_large"
    # The incremental cap stopped the stream before reading all chunks.
    assert h.server.calls[-1][1] == "/status"


async def test_readiness_response_uses_the_streamed_byte_cap(h):
    h.server.health_chunks = [b"x" * (manager_module.MAX_RESPONSE_BYTES // 2 + 1)] * 8
    with pytest.raises(BridgeError) as exc:
        await h.manager.start(owner_id="alice", auth=h.auth)
    assert exc.value.code == "response_too_large"
    assert h.server.calls[0] == ("GET", "/health", None)
    assert ("GET", "/health") in h.factory.clients[0].stream_calls
    # A rejected readiness body is a failed start, not a live session.
    assert h.manager._client is None and h.manager._token == ""
    assert h.processes[0].terminate_calls == 1


async def test_forwarded_fields_are_length_and_range_bounded(h):
    await _started(h)
    h.server.status_overrides = {
        "sequence": 10**20,
        "rotation": -99,
        "reconnect_attempt": 2**40,
        "started_at": "A" * 200,
        "error": "raw error with spaces and\nnewlines",
        "user": {"id": "x" * 500, "name": "y" * 1000},
        "passkey_required": {"has_signer": "yes"},
    }
    status = await h.manager.status(owner_id="alice", auth=h.auth)
    assert status["sequence"] == 0 and status["rotation"] == 0 and status["reconnect_attempt"] == 0
    assert status["started_at"] is None
    assert status["error"] in (None, "unspecified_error")
    assert status["user"]["id"] is None and len(status["user"]["name"]) <= 128
    assert status["passkey_required"] == {"has_signer": False}
    assert not math.isnan(float(status["sequence"]))

    h.server.status_overrides = {}
    h.server.qr = {"data_uri": _data_uri(), "expires_at": time.time() + 10, "rotation": 10**30}
    qr = await h.manager.qr(owner_id="alice", auth=h.auth)
    assert qr["rotation"] == 0
    assert 0 <= qr["expires_in_ms"] <= 300_000


# --- Process cleanup ---------------------------------------------------------


async def test_stop_terminates_awaits_and_clears_token(h):
    await _started(h)
    process = h.processes[0]
    client = h.factory.clients[0]
    token = h.spawns[0]["env"]["LEO_WHATSAPP_BRIDGE_TOKEN"]
    await h.manager.stop(owner_id="alice", auth=h.auth)
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert client.closed is True
    assert h.manager._token == ""
    assert token not in repr(h.manager)
    assert h.manager.is_running is False
    await h.manager.stop(owner_id="alice", auth=h.auth)
    assert process.terminate_calls == 1  # idempotent
    assert (await h.manager.status(owner_id="alice", auth=h.auth))["state"] == "idle"
    assert await h.manager.qr(owner_id="alice", auth=h.auth) is None


@pytest.mark.parametrize("error", [RuntimeError("transport exploded"), KeyboardInterrupt("interrupt")])
async def test_stop_erases_token_when_aclose_raises_any_exception(h, error):
    await _started(h)
    client = h.factory.clients[0]

    async def broken_aclose() -> None:
        raise error

    client.aclose = broken_aclose  # type: ignore[method-assign]
    # A non-HTTP (even BaseException) failure during teardown must not strand
    # the bearer token or leave a live transport behind.
    await h.manager.stop(owner_id="alice", auth=h.auth)
    assert h.manager._token == ""
    assert h.manager._client is None
    assert client.closed is False  # it failed, but erasure still happened


async def test_stop_erases_token_when_teardown_raises_before_aclose(h):
    await _started(h)

    def broken_close() -> None:
        raise ValueError("listener close failed")

    h.manager._listener = SimpleNamespace(close=broken_close)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="listener close failed"):
        await h.manager.stop(owner_id="alice", auth=h.auth)
    assert h.manager._token == ""
    assert h.manager._client is None


async def test_stop_kills_child_that_ignores_terminate(h):
    h.processes.clear()
    original = h.process_factory

    def stubborn_factory(*, argv, cwd, env, pass_fds=()):
        process = original(argv=argv, cwd=cwd, env=env, pass_fds=pass_fds)
        process.hang_first_wait = True
        return process

    manager = WhatsAppBridgeManager(h.config, process_factory=stubborn_factory, http_client=h.factory)
    await manager.start(owner_id="alice", auth=manager.session_auth)
    await manager.stop(owner_id="alice", auth=manager.session_auth)
    assert h.processes[0].terminate_calls == 1
    assert h.processes[0].kill_calls == 1
    assert manager.is_running is False


async def test_context_manager_always_stops(h):
    manager = WhatsAppBridgeManager(h.config, process_factory=h.process_factory, http_client=h.factory)
    with pytest.raises(RuntimeError, match="boom"):
        async with manager:
            await manager.start(owner_id="alice", auth=manager.session_auth)
            raise RuntimeError("boom")
    assert h.processes[0].terminate_calls == 1
    assert manager._token == ""


# --- Real loopback child: only the FD-inheriting process gets the token ------


_CHILD_SCRIPT = r'''
import json, os, socket
fd = int(os.environ["LEO_WHATSAPP_BRIDGE_FD"])
token = os.environ["LEO_WHATSAPP_BRIDGE_TOKEN"].encode()
listener = socket.socket(fileno=fd)
allowed = False
while True:
    try:
        conn, _ = listener.accept()
    except OSError:
        break
    with conn:
        request = b""
        while b"\r\n\r\n" not in request and len(request) < 16384:
            chunk = conn.recv(4096)
            if not chunk:
                break
            request += chunk
        if b"Authorization: Bearer " + token in request:
            allowed = True
            path = request.split(b" ")[1] if b" " in request else b"/"
            if path == b"/qr":
                body = b'{"error":"no_qr"}'
                status = b"404 Not Found"
            else:
                body = json.dumps({
                    "phase": "waiting_for_qr", "linked": False, "sequence": 2,
                    "rotation": 0, "reconnect_attempt": 0,
                    "started_at": "2026-01-01T00:00:00Z",
                }).encode()
                status = b"200 OK"
        else:
            body = b'{"error":"unauthorized"}'
            status = b"401 Unauthorized"
        conn.sendall(b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n" +
                     b"Cache-Control: no-store\r\nContent-Length: " + str(len(body)).encode() +
                     b"\r\nConnection: close\r\n\r\n" + body)
'''


async def test_real_child_receives_token_only_via_inherited_fd(h, tmp_path: Path):
    monkeypatch_root = tmp_path
    original_root = manager_module._APP_REPO_ROOT
    manager_module._APP_REPO_ROOT = monkeypatch_root
    try:
        seen_ports: list[int] = []
        squatter_errors: list[str] = []

        async def real_factory(*, argv, cwd, env, pass_fds=()):
            # A local squatter tries to grab the exact port before the child binds.
            port = int(env["LEO_WHATSAPP_BRIDGE_PORT"])
            seen_ports.append(port)
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
                squatter_errors.append("squatter bound the port")
                probe.listen(1)
            except OSError as exc:
                squatter_errors.append(type(exc).__name__)
            finally:
                probe.close()
            return await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-c", _CHILD_SCRIPT,
                cwd=cwd, env=env, pass_fds=tuple(pass_fds),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        config = BridgeConfig(
            owner_id="alice", actor="alice", enabled=True, repo_root=tmp_path,
            startup_timeout=3.0, poll_interval=0.01, stop_timeout=1.0,
        )
        manager = WhatsAppBridgeManager(config, process_factory=real_factory)
        try:
            status = await manager.start(owner_id="alice", auth=manager.session_auth)
            assert status["state"] == "waiting_for_qr"
            assert squatter_errors and all(name == "OSError" for name in squatter_errors)
            assert seen_ports == [manager._listener.getsockname()[1]]
            # The child accepted our token-bearing request; a squatter on the
            # same port could neither bind nor observe the Authorization header.
            assert h.server.received_auth == []  # this test never uses the fake server
        finally:
            await manager.stop(owner_id="alice", auth=manager.session_auth)
    finally:
        manager_module._APP_REPO_ROOT = original_root


# --- Static contract of the shipped Node bridge -----------------------------


def test_package_and_bridge_contract():
    package = json.loads((BRIDGE_DIR / "package.json").read_text(encoding="utf-8"))
    assert package["type"] == "module"
    assert package["scripts"]["start"] == "node bridge.js"
    assert package["dependencies"]["@whiskeysockets/baileys"] == "7.0.0-rc13"
    assert "npm install" in (BRIDGE_DIR / "README.md").read_text(encoding="utf-8")
    readme = (BRIDGE_DIR / "README.md").read_text(encoding="utf-8")
    assert ".leo/whatsapp/" in readme and "node_modules/" in readme
    # Node identity is a compatibility probe, not a cryptographic guarantee.
    assert "not cryptographic identity" in readme.lower() or "NOT a cryptographic identity" in readme
    assert "hash" in readme.lower() and "realpath" in readme.lower()

    source = (BRIDGE_DIR / "bridge.js").read_text(encoding="utf-8")
    for needle in (
        "timingSafeEqual", "127.0.0.1", "connection.update", "useMultiFileAuthState",
        "printQRInTerminal: false", "getMessage", "passkey_required", "78",
        "express.json", "qrcode", "SIGTERM", "timingSafeEqual",
        # H1/M1/M2/L2 contract markers.
        "LEO_WHATSAPP_BRIDGE_FD", "app.listen({ fd: bridgeFd }", "qrRenderGeneration",
        "state.linked = false", "hardenSessionFiles", "no-store",
        # F1: forget drains the socket and queued credential saves first.
        "async function cancelAttempt", "await closeSocket()", "drainCredsChain",
        "credsWritesBlocked", "await clearAuthFiles()", "assertNoCredsFile",
        # A self-requeuing chain fails closed and returns an explicit error.
        "class CredsDrainError", "fenceCredsWrites", "throw new CredsDrainError()",
        "credsWriteHandler", "creds_drain_failed",
    ):
        assert needle in source, needle
    assert source.count("await assertNoCredsFile()") >= 2  # before and after openSocket
    assert re.search(r"await openSocket\(false\);\s*\n\s*if \(body\.forget === true\) \{", source)
    assert "console.log" not in source
    assert "token=" not in source  # token never travels in a URL/query string

    import hashlib
    digest = hashlib.sha256((BRIDGE_DIR / "bridge.js").read_bytes()).hexdigest()
    assert digest == manager_module._BRIDGE_JS_SHA256

    lockfile = BRIDGE_DIR / "package-lock.json"
    if lockfile.exists():
        lock = json.loads(lockfile.read_text(encoding="utf-8"))
        assert lock["name"] == package["name"]
        assert lock["packages"][""]["dependencies"] == package["dependencies"]

    node = shutil.which("node")
    if node:
        check = subprocess.run([node, "--check", str(BRIDGE_DIR / "bridge.js")], capture_output=True, timeout=30)
        assert check.returncode == 0, check.stderr.decode("utf-8", "replace")
