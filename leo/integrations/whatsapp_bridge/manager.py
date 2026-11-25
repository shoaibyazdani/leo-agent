"""Authenticated manager for the opt-in WhatsApp Web device-link bridge.

This module never talks to WhatsApp itself. It launches the local Node bridge,
hands it a freshly generated owner token through the child environment (never
argv, URLs, files or logs), and offers a small async API for the pairing
lifecycle: ``start`` / ``status`` / ``qr`` / ``wait`` / ``cancel`` / ``repair`` /
``stop``.

Security invariants enforced here:

* Nothing starts unless ``BridgeConfig.enabled`` is explicitly true.
* Every public lifecycle call requires BOTH the owner id and the per-manager
  session token (``session_auth``). An owner id is an identity claim, not a
  credential; it is never accepted as the sole authority. ``status`` and
  ``stop`` are authenticated exactly like the mutating calls.
* Session paths are pinned to the application repository, validated against
  the ``.leo/whatsapp`` root and checked for symlinks; credential files are
  0600. Actor ids can never escape the root.
* The parent BINDS and KEEPS the loopback listening socket and passes its FD
  to the child. Nobody can squat the port between port selection and child
  bind, and the token is not sent until the child/FD ownership is verified.
* The child is the ``node`` found on PATH or an operator-pinned absolute path,
  running the pinned ``bridge.js`` digest. Caller supplied binary/script paths
  are rejected. Executing a binary and checking ``node --version`` is NOT a
  cryptographic identity check: a malicious executable can print that string.
  For production, pin an operator-verified realpath and enforce its hash out
  of band; the version probe only rejects obviously wrong binaries.
* The child gets a scrubbed environment (allowlist only), a fixed cwd, no shell,
  and a per-process random token. The token is never logged, returned, or put in
  a URL. All responses are streamed under an incremental byte cap and every
  forwarded field is length/range bounded.
* Child output is redacted BEFORE truncation, and QR/Q lifecycle results are
  generation-checked so a cancelled or superseded QR is never published. QR
  is a pairing bearer credential: it is only returned to an authenticated
  session and never logged. Each image is SINGLE-USE within a generation: the
  SHA-256 of its PNG bytes is consumed on first publication in either ``qr()``
  or ``status()``. The manager owns BOTH ordering keys: the lifecycle
  generation and a monotonic publication sequence that advances only when a
  genuinely new, validated image is published. Child-supplied generation and
  rotation are bounded informational metadata, never an ordering key. A
  generation advance archives the consumed ledger (for cross-generation replay
  defense) and starts a fresh one, so a hostile child cannot permanently
  exhaust the per-generation publication budget.

The Node bridge and this manager are deliberately not wired into the app: a
maintainer must enable them explicitly and accept the WhatsApp ToS/account-ban
tradeoff first. See README.md.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import math
import os
import re
import secrets
import shutil
import socket
import stat
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from ...logging import get_logger

_LOG = get_logger("integrations.whatsapp_bridge")

PROTOCOL = "whatsapp-bridge/1"
# Pinned from the module location. Callers cannot redirect the session root or
# the bridge script directory; tests may monkeypatch these two module constants.
_APP_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = Path(__file__).resolve().parent
# SHA-256 of the shipped bridge.js. Recompute when bridge.js is changed.
_BRIDGE_JS_SHA256 = "b4322176717f4429be16fc8dc316e543f38e5582ab014a579bc98f8bfdb078e1"
_BRIDGE_SCRIPT_MAX_BYTES = 1_000_000
_NODE_COMMAND_ALLOWLIST = frozenset({"node"})
# Only the first bytes of `node --version` are read; a hostile binary cannot
# flood the parent with probe output. This probe is a compatibility check, not
# proof of identity: a crafted executable can forge the version string. The
# resolved realpath is pinned for execution, and production operators must
# verify that realpath/hash out of band.
_NODE_VERSION_MAX_BYTES = 256
_NODE_VERSION_TIMEOUT = 5.0
_NODE_VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+")
_NODE_MAGIC = (
    b"\x7fELF",            # ELF (Linux)
    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit LE (macOS)
    b"\xce\xfa\xed\xfe",  # Mach-O 32-bit LE
    b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit BE
    b"\xfe\xed\xfa\xce",  # Mach-O 32-bit BE
    b"MZ",                # PE (Windows)
)
ACTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_JID_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
# WhatsApp JIDs are opaque provider ids. Modern LID/device forms can contain
# characters outside the legacy alphabet, so accept any non-whitespace,
# non-control string that carries an "@domain".
_JID_OPAQUE_RE = re.compile(r"^[^\s\x00-\x1f]{3,128}$")
_CODE_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def _is_jid(value: Any) -> bool:
    return isinstance(value, str) and "@" in value and bool(_JID_OPAQUE_RE.fullmatch(value))
_SAFE_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_BRIDGE_JS = "bridge.js"
_OWNER_FILE = "owner.json"
_TOKEN_ENV = "LEO_WHATSAPP_BRIDGE_TOKEN"
_FD_ENV = "LEO_WHATSAPP_BRIDGE_FD"
_QR_PREFIX = "data:image/png;base64,"
_QR_B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
MAX_QR_DATA_URI_CHARS = 65_536
# A generation's consumed-image budget. Within a generation there is NO
# eviction: a digest published once stays rejected for that whole generation.
# The ledger is archived (and a fresh one installed) only when the MANAGER
# legitimately advances the lifecycle generation, so a child cannot spend this
# budget forever. Reaching the cap fails closed until the next generation.
MAX_CONSUMED_QR_HASHES = 10_000
# Number of past generations whose consumed ledgers stay searchable so an old
# image cannot be replayed in a later generation. These are bounded archives,
# not an unbounded never-evict ledger; a QR's TTL is far shorter than this
# window. Within the current generation there are never false negatives.
MAX_QR_GENERATION_LEDGERS = 8
MAX_RESPONSE_BYTES = 262_144
MAX_LOG_LINE_CHARS = 512
MAX_RAW_LOG_LINE_BYTES = 16_384
MAX_QUEUED_EVENTS = 64
MAX_QUEUED_LOG_LINES = 32
MAX_FORWARDED_INT = 2**31 - 1
MAX_TIMESTAMP_CHARS = 64
MAX_EXPIRES_MS = 300_000
MAX_EXPIRES_EPOCH = 1e12
# Message transport: matches bridge.js. Text is capped so a batch response
# stays under MAX_RESPONSE_BYTES and one /send body stays under the bridge cap.
MAX_MESSAGE_TEXT_CHARS = 2_000
MAX_MESSAGE_BATCH = 10
MAX_MESSAGE_TIMEOUT = 30.0
MAX_MESSAGE_ID_CHARS = 128
_SECRET_ENV_KEY_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|PASSKEY)", re.IGNORECASE)
_SECRET_FIELD_RE = re.compile(
    r'("(?:authorization|token|qr|qr_raw|data_uri|password|secret|api_key|session_auth)"\s*:\s*")[^"]{6,}(")',
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"Bearer [A-Za-z0-9_.\-]{8,}")

# Environment allowlist for the child. No inherited API keys, tokens, cloud
# credentials, proxy auth, NODE_OPTIONS or LEO_* settings cross this boundary.
_CHILD_ENV_ALLOWLIST = ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "SYSTEMROOT", "LANG", "LC_ALL")

TERMINAL_STATES = frozenset({"connected", "logged_out", "error", "passkey_required", "cancelled"})


class BridgeError(RuntimeError):
    """A bridge request failed. ``message`` is always safe to display."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "bridge_error",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class BridgeNotEnabled(BridgeError):
    def __init__(self) -> None:
        super().__init__(
            "WhatsApp device linking is disabled. Enable the bridge explicitly to use it.",
            code="bridge_disabled",
            status_code=403,
        )


class BridgeUnavailable(BridgeError):
    def __init__(self, message: str = "The WhatsApp bridge is not available.") -> None:
        super().__init__(message, code="bridge_unavailable", status_code=503)


class BridgeAuthError(BridgeError):
    def __init__(self) -> None:
        super().__init__("The WhatsApp bridge rejected this owner's token.", code="bridge_unauthorized", status_code=403)


class SessionAuthError(BridgeAuthError):
    """The caller did not present this manager session's auth secret."""

    def __init__(self) -> None:
        BridgeError.__init__(
            self,
            "This WhatsApp bridge session requires its session token.",
            code="session_auth_required",
            status_code=403,
        )


class BridgeTimeout(BridgeError):
    def __init__(self, message: str = "The WhatsApp bridge did not respond in time.") -> None:
        super().__init__(message, code="bridge_timeout", status_code=504)


class SessionOwnershipError(BridgeError):
    def __init__(self) -> None:
        super().__init__("This pairing session belongs to a different owner.", code="session_ownership", status_code=403)


def _safe_code(raw: Any, fallback: str = "unspecified_error") -> str:
    return raw if isinstance(raw, str) and _CODE_RE.fullmatch(raw) else fallback


def _bounded_int(value: Any, default: int = 0, maximum: int = MAX_FORWARDED_INT) -> int:
    """Only forward small, non-negative ints from an untrusted child payload."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        return default
    return value


def _bounded_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > MAX_TIMESTAMP_CHARS:
        return None
    return value if _SAFE_TIMESTAMP_RE.fullmatch(value) else None


def _maybe_await(value: Any) -> Any:
    """Accept sync fakes (terminate/kill/wait) and real asyncio subprocesses."""
    if hasattr(value, "__await__"):
        return value
    async def _done() -> Any:
        return value
    return _done()


@dataclass(frozen=True)
class BridgeConfig:
    owner_id: str
    actor: str = "default"
    enabled: bool = False
    # Accepted only when it resolves to the pinned application repository.
    repo_root: Path | None = None
    # Accepted only when it resolves to the pinned package directory.
    bridge_dir: Path | None = None
    # Only "node" from PATH, or an absolute path in node_path_allowlist.
    node_binary: str = "node"
    node_path_allowlist: tuple[str, ...] = ()
    port: int | None = None
    startup_timeout: float = 10.0
    poll_interval: float = 0.1
    stop_timeout: float = 5.0
    max_wait_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.owner_id, str) or not ACTOR_RE.fullmatch(self.owner_id):
            raise ValueError("owner_id must be 1-64 safe characters")
        if not isinstance(self.actor, str) or not ACTOR_RE.fullmatch(self.actor):
            raise ValueError("actor must be 1-64 safe characters")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if self.port is not None and (not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535):
            raise ValueError("port must be between 1 and 65535")
        for name in ("startup_timeout", "poll_interval", "stop_timeout", "max_wait_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 300:
                raise ValueError(f"{name} must be between 0 and 300 seconds")
        if not isinstance(self.node_binary, str) or not re.fullmatch(r"[A-Za-z0-9_./\\-]{1,256}", self.node_binary):
            raise ValueError("node_binary must be a plain command name or pinned absolute path")
        if not isinstance(self.node_path_allowlist, tuple):
            raise ValueError("node_path_allowlist must be a tuple of absolute paths")
        for entry in self.node_path_allowlist:
            if not isinstance(entry, str) or not entry.startswith("/") or len(entry) > 256 or ".." in entry:
                raise ValueError("node_path_allowlist entries must be pinned absolute paths")
        if self.repo_root is not None and not isinstance(self.repo_root, (str, Path)):
            raise ValueError("repo_root must be a Path")
        if self.bridge_dir is not None and not isinstance(self.bridge_dir, (str, Path)):
            raise ValueError("bridge_dir must be a Path")

    @property
    def session_root(self) -> Path:
        base = Path(_APP_REPO_ROOT).resolve()
        if self.repo_root is not None and Path(self.repo_root).resolve() != base:
            raise BridgeError(
                "Session root must be the pinned application repository.",
                code="invalid_session_path",
            )
        root = base / ".leo" / "whatsapp"
        if any(part.is_symlink() for part in (base, base / ".leo", root)):
            raise BridgeError("Session root contains a symlink.", code="invalid_session_path")
        return root

    @property
    def bridge_path(self) -> Path:
        if self.bridge_dir is not None and Path(self.bridge_dir).resolve() != _PACKAGE_DIR.resolve():
            raise BridgeError("Bridge script directory is not pinned.", code="invalid_bridge_path")
        script = _PACKAGE_DIR.resolve() / _BRIDGE_JS
        if script.is_symlink():
            raise BridgeError("Bridge script may not be a symlink.", code="invalid_bridge_path")
        return script


class WhatsAppBridgeManager:
    """Launch and control one actor's device-link bridge."""

    def __init__(
        self,
        config: BridgeConfig,
        *,
        process_factory: Callable[..., Any] | None = None,
        http_client: Any = None,
    ) -> None:
        self.config = config
        self._process_factory = process_factory or self._spawn_process
        self._http_client = http_client
        self._proc: Any = None
        self._client: Any = None
        self._listener: socket.socket | None = None
        self._token = ""
        self._port: int | None = None
        self._session_dir: Path | None = None
        self._stdout_task: asyncio.Task[Any] | None = None
        self._stderr_task: asyncio.Task[Any] | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_QUEUED_EVENTS)
        self._log_tail: deque[str] = deque(maxlen=MAX_QUEUED_LOG_LINES)
        # A real per-session credential. The owner id is only a claim and is
        # never sufficient authority for a manager call.
        self._session_auth = secrets.token_urlsafe(32)
        # Values of credential-like env vars are remembered for output redaction
        # only; a broken child must not be able to echo them back into logs.
        self._secret_values = tuple(sorted(
            {value for key, value in os.environ.items() if _SECRET_ENV_KEY_RE.search(key) and len(value) >= 8} | {self._session_auth},
            key=len,
            reverse=True,
        ))
        self._generation = 0
        # QR identity is (MANAGER generation, MANAGER publication sequence).
        # The manager alone advances both halves: generation on lifecycle
        # changes, publication count only after a genuinely new, validated
        # image is published. Child-supplied generation/rotation fields are
        # informational metadata and can never move this watermark.
        self._qr_identity: tuple[int, int] = (0, 0)
        self._qr_publication = 0
        # Digests consumed in the CURRENT generation (SHA-256 of decoded PNG
        # bytes), capped at MAX_CONSUMED_QR_HASHES with no within-generation
        # eviction. _advance_generation() archives this set and installs a
        # fresh one, so a child that floods a generation cannot block QR
        # publication forever.
        self._qr_consumed: set[str] = set()
        # Union of the current ledger and the bounded archived generations,
        # for O(1) same-generation and cross-generation replay checks. Digests
        # of a generation that falls out of the archive are removed here too.
        self._qr_seen: set[str] = set()
        self._qr_consumed_history: deque[tuple[int, frozenset[str]]] = deque()
        # The one image that is currently published. It is cached so ordinary
        # re-reads of the same live QR do not count as a new publication; any
        # other appearance of a consumed digest is a replay and is rejected.
        self._active_qr: dict[str, Any] | None = None
        self._active_qr_digest: str | None = None
        self._active_qr_identity: tuple[int, int] = (0, 0)  # (generation, publication)
        self._lifecycle_lock = asyncio.Lock()
        self._qr_lock = asyncio.Lock()

    def __repr__(self) -> str:
        # Never let a default repr/the tokens leak into a traceback or log.
        return f"WhatsAppBridgeManager(actor={self.config.actor!r}, enabled={self.config.enabled!r}, running={self.is_running})"

    @property
    def session_auth(self) -> str:
        """The secret the caller must pass to every public lifecycle method."""
        return self._session_auth

    @property
    def is_running(self) -> bool:
        return self._proc is not None and getattr(self._proc, "returncode", None) is None

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Sanitized lifecycle events (no QR data, token or raw error text)."""
        return tuple(self._events)

    def diagnostics(self) -> tuple[str, ...]:
        """Bounded, redacted child output for troubleshooting."""
        return tuple(self._log_tail)

    # --- Authentication and session ownership --------------------------------

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise BridgeNotEnabled()

    def _require_auth(self, owner_id: str | None, auth: str | None) -> None:
        # Constant-time comparison of a high-entropy secret. An owner id alone
        # (even a correct one) can never authorize a manager call.
        if not isinstance(auth, str) or not auth.isascii():
            raise SessionAuthError()
        try:
            valid = secrets.compare_digest(auth, self._session_auth)
        except TypeError:
            valid = False
        if not valid:
            raise SessionAuthError()
        self._require_owner(owner_id)

    def _require_owner(self, owner_id: str | None) -> None:
        if owner_id is not None and (not isinstance(owner_id, str) or owner_id != self.config.owner_id):
            raise SessionOwnershipError()
        if self._session_dir is not None:
            owner_file = self._session_dir / _OWNER_FILE
            if owner_file.is_symlink():
                raise SessionOwnershipError()
            try:
                record = json.loads(owner_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise SessionOwnershipError() from None
            if not isinstance(record, dict) or record.get("owner_id") != self.config.owner_id:
                raise SessionOwnershipError()

    def _prepare_session(self) -> Path:
        root = self.config.session_root
        directory = root / self.config.actor
        if directory.is_symlink():
            raise BridgeError("Invalid pairing session path.", code="invalid_session_path")
        resolved = directory.resolve()
        if resolved.parent != root or not str(root).endswith(os.path.join(".leo", "whatsapp")):
            raise BridgeError("Invalid pairing session path.", code="invalid_session_path")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink() or directory.resolve().parent != root:
            raise BridgeError("Invalid pairing session path.", code="invalid_session_path")
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        # Linked-device credentials are bearer tokens. Refuse symlinks and
        # tighten any file a previous/partial run left in the session dir.
        for entry in directory.iterdir():
            if entry.is_symlink():
                raise BridgeError("Invalid pairing session file.", code="invalid_session_path")
            if entry.name == _OWNER_FILE:
                continue
            try:
                os.chmod(entry, 0o700 if entry.is_dir() else 0o600)
            except OSError:
                pass
        owner_file = directory / _OWNER_FILE
        record = {
            "actor": self.config.actor,
            "owner_id": self.config.owner_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            fd = os.open(owner_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if owner_file.is_symlink() or self._read_owner_file(owner_file) != self.config.owner_id:
                raise SessionOwnershipError() from None
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True)
            try:
                os.chmod(owner_file, 0o600)
            except OSError:
                pass
        self._session_dir = directory
        return directory

    @staticmethod
    def _read_owner_file(owner_file: Path) -> str:
        try:
            data = json.loads(owner_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if not isinstance(data, dict):
            return ""
        owner_id = data.get("owner_id")
        return owner_id if isinstance(owner_id, str) and ACTOR_RE.fullmatch(owner_id) else ""

    # --- Binary and script pinning -------------------------------------------

    @staticmethod
    async def _verify_node_binary(path: Path) -> Path:
        """Reject non-executables and binaries that do not print a Node version.

        This is a compatibility probe, not proof of identity: any local program
        can forge ``node --version``. Production deployments must pin an
        operator-verified realpath and check its hash independently. Only the
        resolved realpath returned here is executed, which prevents PATH/symlink
        substitution after the probe.
        """
        try:
            resolved = path.resolve(strict=True)
            mode = resolved.stat().st_mode
            if not stat.S_ISREG(mode) or not (mode & 0o111) or not os.access(resolved, os.X_OK):
                raise BridgeUnavailable("Configured Node binary is not executable.")
            with resolved.open("rb") as handle:
                if handle.read(4) not in _NODE_MAGIC:
                    raise BridgeUnavailable("Configured Node binary is not a Node executable.")
        except BridgeUnavailable:
            raise
        except OSError:
            raise BridgeUnavailable("Configured Node binary cannot be inspected.") from None

        # Marker scanning is trivially spoofable. Execute the very same resolved
        # realpath and require real Node's `--version` signature; stderr is
        # discarded and stdout is capped so a hostile binary cannot flood logs
        # or memory. The returned realpath is what the child later executes.
        probe: Any = None
        try:
            probe = await asyncio.create_subprocess_exec(
                str(resolved), "--version",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env={key: os.environ[key] for key in _CHILD_ENV_ALLOWLIST if key in os.environ},
            )
            output = await asyncio.wait_for(
                probe.stdout.read(_NODE_VERSION_MAX_BYTES), timeout=_NODE_VERSION_TIMEOUT,
            )
            returncode = await asyncio.wait_for(probe.wait(), timeout=_NODE_VERSION_TIMEOUT)
        except (OSError, ValueError):
            raise BridgeUnavailable("Configured Node binary cannot be executed.") from None
        except asyncio.TimeoutError:
            raise BridgeUnavailable("Configured Node binary did not answer a version probe.") from None
        finally:
            if probe is not None and probe.returncode is None:
                try:
                    probe.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await probe.wait()
                except (OSError, asyncio.TimeoutError):
                    pass
        if returncode != 0 or not _NODE_VERSION_RE.match(output.decode("utf-8", "replace").strip()):
            raise BridgeUnavailable("Configured Node binary has no Node identity.")
        return resolved

    async def _resolve_node(self) -> str:
        binary = self.config.node_binary
        if "/" in binary or "\\" in binary or binary not in _NODE_COMMAND_ALLOWLIST:
            pinned = {Path(entry).resolve() for entry in self.config.node_path_allowlist}
            candidate = Path(binary).expanduser()
            if not candidate.is_absolute() or candidate.resolve() not in pinned:
                raise BridgeUnavailable("Configured Node binary is not operator-pinned.")
            return str(await self._verify_node_binary(candidate))
        found = shutil.which("node")
        if not found:
            raise BridgeUnavailable("Node.js is not installed; install Node 20+ to use the bridge.")
        return str(await self._verify_node_binary(Path(found)))

    def _verify_bridge_script(self) -> Path:
        script = self.config.bridge_path
        if not script.is_file() or script.stat().st_size > _BRIDGE_SCRIPT_MAX_BYTES:
            raise BridgeUnavailable("The bridge script is missing or invalid.")
        try:
            with script.open("rb") as handle:
                digest = hashlib.sha256(handle.read(_BRIDGE_SCRIPT_MAX_BYTES + 1)).hexdigest()
        except OSError:
            raise BridgeUnavailable("The bridge script cannot be read.") from None
        if not secrets.compare_digest(digest, _BRIDGE_JS_SHA256):
            raise BridgeUnavailable("The bridge script does not match its pinned checksum.")
        return script

    # --- Process and transport plumbing -------------------------------------

    def _spawn_process(
        self,
        *,
        argv: list[str],
        cwd: str,
        env: dict[str, str],
        pass_fds: tuple[int, ...] = (),
    ) -> Any:
        return asyncio.create_subprocess_exec(
            argv[0], *argv[1:],
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            pass_fds=pass_fds,
            start_new_session=True,
        )

    def _launch(self, argv: list[str], cwd: str, env: dict[str, str], pass_fds: tuple[int, ...]) -> Any:
        """Pass the inherited listener FD when the process factory supports it."""
        factory = self._process_factory
        try:
            params = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_fds = "pass_fds" in params or any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        kwargs: dict[str, Any] = {"argv": argv, "cwd": cwd, "env": env}
        if accepts_fds:
            kwargs["pass_fds"] = pass_fds
        return _maybe_await(factory(**kwargs))

    def _build_env(self, token: str, fd: int, session_dir: Path, port: int) -> dict[str, str]:
        env = {key: os.environ[key] for key in _CHILD_ENV_ALLOWLIST if key in os.environ}
        env.update({
            _TOKEN_ENV: token,
            _FD_ENV: str(fd),
            "LEO_WHATSAPP_ACTOR": self.config.actor,
            "LEO_WHATSAPP_SESSION_ROOT": str(self.config.session_root),
            "LEO_WHATSAPP_SESSION_DIR": str(session_dir),
            "LEO_WHATSAPP_BRIDGE_PORT": str(port),
            "LEO_WHATSAPP_PASSKEY_MODE": "report",
        })
        return env

    @staticmethod
    def _bind_listener() -> socket.socket:
        """Bind and KEEP a loopback listener; never close it before child spawn."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # No SO_REUSEADDR: another local process must not be able to bind
            # this exact address while the parent holds it.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            listener.bind(("127.0.0.1", 0))
            listener.listen(8)
            return listener
        except OSError:
            listener.close()
            raise BridgeUnavailable("Could not reserve a loopback listener.") from None

    def _verify_child_listener(self, proc: Any, listener: socket.socket) -> None:
        """Verify the child owns the inherited listener before any token is sent."""
        fd = listener.fileno()
        if fd < 0 or not stat.S_ISSOCK(os.fstat(fd).st_mode):
            raise BridgeUnavailable("The bridge listener is not a socket.")
        pid = getattr(proc, "pid", None)
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise BridgeUnavailable("The bridge process did not start correctly.")
        # A child that dies instantly is handled (and its redacted output kept)
        # by _wait_ready; it has not received a token either way.
        # Linux exposes the inherited FD set. On macOS/BSD there is no /proc;
        # the guarantee comes from the parent-held socket + pass_fds inheritance.
        proc_fd = Path("/proc") / str(pid) / "fd" / str(fd)
        if proc_fd.exists():
            try:
                link = os.readlink(proc_fd)
            except OSError:
                link = ""
            inode = os.fstat(fd).st_ino
            if link != f"socket:[{inode}]":
                raise BridgeUnavailable("The bridge process does not own the listener.")
        # The caller claims the fresh listener before verification so failures
        # can be cleaned up; only a *different* pre-existing listener is fatal.
        if self._listener is not None and self._listener is not listener:
            raise BridgeUnavailable("A bridge listener already exists.")

    def _make_client(self, port: int, token: str) -> Any:
        headers = {
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-store, no-cache",
            "Pragma": "no-cache",
        }
        if self._http_client is not None:
            if hasattr(self._http_client, "request"):
                return self._http_client
            return self._http_client(
                base_url=f"http://127.0.0.1:{port}",
                headers=headers,
            )
        return httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            headers=headers,
            timeout=5.0,
            trust_env=False,
            limits=httpx.Limits(max_connections=4),
        )

    def _redact(self, text: str) -> str:
        for secret in (*self._secret_values, self._token):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = _SECRET_FIELD_RE.sub(r"\1[REDACTED]\2", text)
        text = _BEARER_RE.sub("Bearer [REDACTED]", text)
        return re.sub(r"data:image/[A-Za-z0-9;,+/=_-]{16,}", "[QR_REDACTED]", text)

    async def _drain_stream(self, name: str, stream: Any) -> None:
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except (OSError, ValueError):
                break
            if not line:
                break
            if isinstance(line, bytes):
                text = line.decode("utf-8", "replace")
            else:
                text = str(line)
            # Redact the WHOLE line first. Truncating first can split a secret
            # or leave a recognizable prefix of it in the diagnostic tail.
            redacted = self._redact(text)
            if len(text) > MAX_RAW_LOG_LINE_BYTES:
                redacted = "[LONG_LINE_REDACTED] " + redacted
            display = redacted[:MAX_LOG_LINE_CHARS].strip()
            markers = [
                marker for marker in ("[REDACTED]", "[QR_REDACTED]", "[LONG_LINE_REDACTED]")
                if marker in redacted
            ]
            if len(redacted) > MAX_LOG_LINE_CHARS and markers:
                # Keep evidence of every redaction even when the secret sat
                # past the display limit; never show the raw tail itself.
                display = f"{display} {' '.join(markers)}"
            if display:
                self._log_tail.append(f"{name}: {display}")
            if name == "stdout":
                self._record_event(redacted)

    def _record_event(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        event = {
            key: _safe_code(payload.get(key), "") or None
            for key in ("event", "phase", "state", "error", "code")
        }
        event["sequence"] = _bounded_int(payload.get("sequence")) or None
        event["rotation"] = _bounded_int(payload.get("rotation")) or None
        event["has_signer"] = payload.get("has_signer") if isinstance(payload.get("has_signer"), bool) else None
        self._events.append(event)

    async def _limited_body(self, response: Any) -> bytes:
        """Read a response incrementally, aborting past MAX_RESPONSE_BYTES."""
        aiter_bytes = getattr(response, "aiter_bytes", None)
        if callable(aiter_bytes):
            body = bytearray()
            async for chunk in aiter_bytes(chunk_size=8192):
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise BridgeError(
                        "The WhatsApp bridge returned an oversized response.",
                        code="response_too_large",
                        status_code=413,
                    )
                body.extend(chunk)
            return bytes(body)
        content = getattr(response, "content", None)
        if isinstance(content, (bytes, bytearray)):
            body = bytes(content)
        elif isinstance(content, str):
            body = content.encode("utf-8")
        else:
            text = getattr(response, "text", "")
            body = text.encode("utf-8") if isinstance(text, str) else b""
        if len(body) > MAX_RESPONSE_BYTES:
            raise BridgeError(
                "The WhatsApp bridge returned an oversized response.",
                code="response_too_large",
                status_code=413,
            )
        return body

    @staticmethod
    def _decode_payload(body: bytes) -> dict[str, Any]:
        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise BridgeError("The WhatsApp bridge returned an invalid response.", code="invalid_response") from None
        if not isinstance(data, dict):
            raise BridgeError("The WhatsApp bridge returned an invalid response.", code="invalid_response")
        return data

    async def _finish_request(self, response: Any, allow_404: bool) -> dict[str, Any] | None:
        status = getattr(response, "status_code", 0)
        if status in (401, 403):
            # The stream is owned by the caller's context manager (or already
            # buffered for fake clients), so no explicit close is needed here.
            raise BridgeAuthError()
        if status == 404 and allow_404:
            return None
        if status >= 500:
            raise BridgeUnavailable("The WhatsApp bridge reported a server error.")
        if status >= 400:
            code = "request_rejected"
            try:
                payload = self._decode_payload(await self._limited_body(response))
                # Child-supplied codes are allowlisted to safe lowercase tokens.
                code = _safe_code(payload.get("error"), code)
            except BridgeError:
                pass
            raise BridgeError("The WhatsApp bridge rejected the request.", code=code, status_code=status)
        return self._decode_payload(await self._limited_body(response))

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        if self._client is None:
            raise BridgeUnavailable("The WhatsApp bridge is not running.")
        stream = getattr(self._client, "stream", None)
        if callable(stream):
            try:
                async with stream(method, path, json=payload, timeout=timeout) as response:
                    return await self._finish_request(response, allow_404)
            except (httpx.HTTPError, OSError):
                raise BridgeUnavailable("The WhatsApp bridge is unreachable.") from None
        try:
            response = await self._client.request(method, path, json=payload, timeout=timeout)
        except (httpx.HTTPError, OSError):
            raise BridgeUnavailable("The WhatsApp bridge is unreachable.") from None
        return await self._finish_request(response, allow_404)

    # --- Public lifecycle ----------------------------------------------------

    def _archive_generation_qr(self) -> None:
        """Archive this generation's consumed digests, then reset the ledger.

        Called only from _advance_generation(), which is reachable only from
        authenticated lifecycle transitions (start/cancel/repair/stop). The
        archive is bounded; the current generation's set never evicts a digest
        while it is live, so replay detection has no false negatives within a
        generation. Archived generations still reject cross-generation
        replays until they age out of the bounded window.
        """
        if self._qr_consumed:
            self._qr_consumed_history.append((self._generation, frozenset(self._qr_consumed)))
        while len(self._qr_consumed_history) > MAX_QR_GENERATION_LEDGERS:
            _old_generation, old_digests = self._qr_consumed_history.popleft()
            self._qr_seen.difference_update(old_digests)
        self._qr_consumed.clear()

    def _advance_generation(self) -> int:
        """Advance the manager-owned generation and retire its live QR.

        This is the ONLY place the generation changes. Child payloads cannot
        bump it. A generation change legitimately resets the per-generation
        consumed-image budget; the old ledger is archived for replay defense.
        """
        self._archive_generation_qr()
        self._generation += 1
        self._active_qr = None
        self._active_qr_digest = None
        self._active_qr_identity = (0, 0)
        return self._generation

    async def start(self, owner_id: str | None = None, *, auth: str | None = None) -> dict[str, Any]:
        """Explicitly launch the bridge and begin a device-link attempt."""
        self._require_enabled()
        self._require_auth(owner_id, auth)
        async with self._lifecycle_lock:
            if self.is_running:
                return await self._status_unlocked(owner_id, auth)
            script = self._verify_bridge_script()
            node = await self._resolve_node()
            session_dir = self._prepare_session()
            token = secrets.token_urlsafe(32)
            listener = self._bind_listener()
            try:
                env = self._build_env(token, listener.fileno(), session_dir, int(listener.getsockname()[1]))
            except BaseException:
                listener.close()
                raise
            argv = [node, str(script)]
            cwd = str(script.parent)

            self._events.clear()
            self._log_tail.clear()
            self._advance_generation()
            try:
                proc = await self._launch(argv, cwd, env, (listener.fileno(),))
            except BaseException:
                listener.close()
                raise

            # Claim ownership immediately after spawn so ANY later failure can
            # be routed through _shutdown(): no orphan child, open listener or
            # live token. Storing the token does not send it; no request is made
            # until _verify_child_listener has confirmed FD ownership.
            self._listener = listener
            self._proc = proc
            self._token = token
            try:
                self._verify_child_listener(proc, listener)
                self._port = int(listener.getsockname()[1])
                self._client = self._make_client(self._port, token)
                self._stdout_task = asyncio.create_task(
                    self._drain_stream("stdout", getattr(proc, "stdout", None))
                )
                self._stderr_task = asyncio.create_task(
                    self._drain_stream("stderr", getattr(proc, "stderr", None))
                )
                # Let the drain tasks consume anything the child wrote before
                # readiness was checked, including on a fast startup failure.
                await asyncio.sleep(0)
                await self._wait_ready()
                await self._request("POST", "/start", {})
                return await self._status_unlocked(owner_id, auth)
            except BaseException:
                await self._shutdown()
                raise

    async def _wait_ready(self) -> None:
        # Give the output-drain tasks one turn: a child that fails instantly
        # may already have written a redacted diagnostic line.
        await asyncio.sleep(0)
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            if getattr(self._proc, "returncode", None) is not None:
                raise BridgeUnavailable("The bridge process exited during startup.")
            try:
                # Same streamed, incrementally capped path as every other
                # request: a compromised child cannot OOM the parent through
                # an unbounded /health body during startup.
                await self._request("GET", "/health", timeout=1.0)
                return
            except BridgeUnavailable:
                # Not listening yet (or a transient 5xx): keep polling.
                await asyncio.sleep(self.config.poll_interval)
        raise BridgeTimeout("The bridge did not become ready in time.")

    def _require_running(self) -> None:
        if self._proc is None or self._client is None:
            raise BridgeUnavailable("The WhatsApp bridge is not running.")

    def _local_status(self) -> dict[str, Any]:
        died = self._proc is not None and not self.is_running
        return {
            "protocol": PROTOCOL,
            "actor": self.config.actor,
            "owner_id": self.config.owner_id,
            "state": "error" if died else "idle",
            "linked": False,
            "error": "bridge_exited" if died else None,
            "qr": None,
            "qr_available": False,
            "passkey_required": None,
            "user": None,
            "rotation": 0,
            "sequence": 0,
            "running": self.is_running,
            "pid": getattr(self._proc, "pid", None),
            "timed_out": False,
        }

    def _public_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        phase = payload.get("phase", payload.get("state"))
        user = payload.get("user")
        safe_user: dict[str, str | None] | None = None
        if isinstance(user, dict):
            identifier = user.get("id")
            name = user.get("name")
            safe_user = {
                "id": identifier if isinstance(identifier, str) and _JID_RE.fullmatch(identifier) else None,
                "name": re.sub(r"[^\x20-\x7e]", "", name)[:128] if isinstance(name, str) else None,
            }
        qr = self._public_qr(payload.get("qr")) if isinstance(payload.get("qr"), dict) else None
        # A child claiming `linked` in any non-connected phase (cancelled,
        # re-pairing, error) is not trusted. The process must also still run.
        linked = payload.get("linked") is True and phase == "connected" and self.is_running
        return {
            "protocol": PROTOCOL,
            "actor": self.config.actor,
            "owner_id": self.config.owner_id,
            "state": _safe_code(phase, "unknown"),
            "linked": linked,
            "error": _safe_code(payload.get("error")) if payload.get("error") is not None else None,
            "qr": qr,
            "qr_available": qr is not None,
            "passkey_required": {
                "has_signer": payload["passkey_required"].get("has_signer") is True,
            } if isinstance(payload.get("passkey_required"), dict) else None,
            "user": safe_user,
            "rotation": _bounded_int(payload.get("rotation")),
            "sequence": _bounded_int(payload.get("sequence")),
            "reconnect_attempt": _bounded_int(payload.get("reconnect_attempt")),
            "started_at": _bounded_timestamp(payload.get("started_at")),
            "running": self.is_running,
            "pid": getattr(self._proc, "pid", None),
            "timed_out": payload.get("timed_out") is True,
        }

    def _public_qr(self, qr: dict[str, Any]) -> dict[str, Any] | None:
        data_uri = qr.get("data_uri")
        if not isinstance(data_uri, str) or not data_uri.startswith(_QR_PREFIX):
            raise BridgeError("The bridge returned an invalid QR image.", code="invalid_qr")
        encoded = data_uri[len(_QR_PREFIX):]
        if len(data_uri) > MAX_QR_DATA_URI_CHARS or not encoded or not _QR_B64_RE.fullmatch(encoded):
            raise BridgeError("The bridge returned an oversized or invalid QR image.", code="invalid_qr")
        expires_at = qr.get("expires_at")
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
            raise BridgeError("The bridge returned an invalid QR expiry.", code="invalid_qr")
        expires_at = float(expires_at)
        # NaN/inf and absurd epochs are never forwarded to a caller.
        if not math.isfinite(expires_at) or expires_at < 0:
            raise BridgeError("The bridge returned an invalid QR expiry.", code="invalid_qr")
        # Accept both epoch seconds and the bridge's epoch milliseconds.
        if expires_at > MAX_EXPIRES_EPOCH:
            expires_at /= 1000.0
        if not math.isfinite(expires_at) or not 0 <= expires_at <= MAX_EXPIRES_EPOCH:
            raise BridgeError("The bridge returned an invalid QR expiry.", code="invalid_qr")
        return {
            "data_uri": data_uri,
            "expires_at": expires_at,
            "expires_in_ms": max(0, min(MAX_EXPIRES_MS, int(expires_at * 1000 - time.time() * 1000))),
            "rotation": _bounded_int(qr.get("rotation")),
        }

    @staticmethod
    def _qr_digest(data_uri: Any) -> str | None:
        """Hash the QR's PNG bytes, not its base64 spelling.

        Hashing decoded bytes also catches the same image re-sent with a
        different (but equivalent) data URI encoding. Malformed images fail
        closed: they can never enter the consumed ledger.
        """
        if not isinstance(data_uri, str) or not data_uri.startswith(_QR_PREFIX):
            return None
        encoded = data_uri[len(_QR_PREFIX):]
        if not encoded or len(encoded) > MAX_QR_DATA_URI_CHARS or not _QR_B64_RE.fullmatch(encoded):
            return None
        try:
            image = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return None
        if not image:
            return None
        return hashlib.sha256(image).hexdigest()

    def _accept_qr(
        self,
        qr: dict[str, Any] | None,
        generation: int,
        raw: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Publish an image at most once per generation; never resurrect one.

        ``generation`` is the manager's authoritative lifecycle generation and
        the publication sequence below is the manager's own monotonic counter.
        Child-supplied ``generation``/``rotation`` fields in ``raw`` are
        deliberately ignored for ordering: a child cannot poison the watermark
        by claiming a huge rotation (e.g. 2147483647) or a newer generation.
        Its rotation is only bounded informational metadata in the response,
        used to recognize a plain re-read of the one live image.

        Re-reads of the live image are served from the manager cache so polling
        still works, but they are NOT publications: an image reappearing with a
        different rotation, in an older generation, or after another image was
        published is rejected as a replay. Within a generation the consumed
        ledger never evicts, so a digest cannot become fresh again. Generation
        advances reset that ledger but archive it for cross-generation checks.
        """
        del raw  # child-supplied generation/rotation claims are never trusted
        if qr is None:
            return None
        digest = self._qr_digest(qr.get("data_uri"))
        if digest is None:
            return None
        if generation < self._active_qr_identity[0]:
            # Defensive: a read captured before a lifecycle change must not use
            # the new generation's cache.
            return None
        if generation != self._active_qr_identity[0]:
            self._active_qr = None
            self._active_qr_digest = None
            self._active_qr_identity = (0, 0)

        rotation = qr["rotation"]  # bounded child metadata, NOT an ordering key
        if digest in self._qr_seen:
            # SINGLE USE. Only the exact live image with its original rotation
            # can be returned; every other reappearance (including an inflated
            # rotation in the same generation or a cross-generation replay) is
            # dropped by both qr() and status().
            if (self._active_qr is not None and digest == self._active_qr_digest
                    and rotation == self._active_qr.get("rotation")
                    and generation == self._active_qr_identity[0]):
                cached = dict(self._active_qr)
                expires_at = cached.get("expires_at")
                if isinstance(expires_at, (int, float)) and not isinstance(expires_at, bool):
                    cached["expires_in_ms"] = max(0, min(MAX_EXPIRES_MS, int(expires_at * 1000 - time.time() * 1000)))
                return cached
            return None

        if len(self._qr_consumed) >= MAX_CONSUMED_QR_HASHES:
            # This generation's budget is spent. Fail closed until the manager
            # legitimately advances the generation; never evict a live digest.
            return None

        # The watermark is assigned HERE, from manager-owned values only, and
        # advances only for a genuinely new, validated, not-yet-seen image.
        publication = self._qr_publication + 1
        identity = (generation, publication)
        if identity <= self._qr_identity:
            # Defensive invariant: this can only happen if the manager itself
            # mis-advances its generation/publication counters.
            return None
        self._qr_consumed.add(digest)
        self._qr_seen.add(digest)
        self._qr_publication = publication
        self._qr_identity = identity
        self._active_qr = dict(qr)
        self._active_qr_digest = digest
        self._active_qr_identity = identity
        return qr

    def _filter_status_qr(
        self,
        public: dict[str, Any],
        raw: dict[str, Any] | None,
        generation: int,
    ) -> dict[str, Any]:
        """Apply the same generation guard to status/wait/lifecycle payloads."""
        if public["qr"] is None:
            return public
        raw_qr = raw.get("qr") if isinstance(raw, dict) and isinstance(raw.get("qr"), dict) else None
        if self._accept_qr(public["qr"], generation, raw_qr) is None:
            public["qr"] = None
            public["qr_available"] = False
        return public

    async def _status_unlocked(self, owner_id: str | None, auth: str | None) -> dict[str, Any]:
        del auth  # authenticated by the caller
        if not self.is_running:
            return self._local_status()
        generation = self._generation
        async with self._qr_lock:
            payload = await self._request("GET", "/status")
            if generation != self._generation or not self.is_running:
                return self._local_status()
            public = self._public_status(payload or {})
            # Same monotonicity guard as qr(): a status snapshot must never
            # resurrect a QR that a later read or a cancel has superseded.
            return self._filter_status_qr(public, payload, generation)

    async def status(self, owner_id: str | None = None, *, auth: str | None = None) -> dict[str, Any]:
        """Authenticated state snapshot. Deliberately contains no raw token."""
        self._require_auth(owner_id, auth)
        return await self._status_unlocked(owner_id, auth)

    async def qr(self, owner_id: str | None = None, *, auth: str | None = None) -> dict[str, Any] | None:
        """Live QR data URI for the authenticated session, or None when absent."""
        self._require_auth(owner_id, auth)
        if not self.is_running:
            return None
        generation = self._generation
        # Serialize QR reads so a stale render cannot overtake a fresh one.
        async with self._qr_lock:
            payload = await self._request("GET", "/qr", allow_404=True)
            if generation != self._generation or not self.is_running or payload is None:
                return None
            public = self._public_qr(payload)
            # A cancelled/re-paired generation must never republish an earlier
            # render, even when the child reports a larger rotation for it.
            return self._accept_qr(public, generation, payload)

    async def wait(
        self,
        owner_id: str | None = None,
        states: frozenset[str] | set[str] | tuple[str, ...] = TERMINAL_STATES,
        timeout: float = 30.0,
        *,
        auth: str | None = None,
    ) -> dict[str, Any]:
        """Long-poll until one of ``states`` is observed or ``timeout`` elapses."""
        self._require_auth(owner_id, auth)
        self._require_running()
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= self.config.max_wait_seconds:
            raise BridgeError("Wait timeout must be between 0 and the configured maximum.", code="invalid_timeout")
        targets = frozenset(str(state) for state in states)
        deadline = time.monotonic() + float(timeout)
        after = _bounded_int((await self._request("GET", "/status") or {}).get("sequence"))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeTimeout("Timed out waiting for the bridge state.")
            generation = self._generation
            payload = await self._request("POST", "/wait", {
                "after": after,
                "timeout_ms": max(100, min(30_000, int(remaining * 1000))),
            })
            result = self._public_status(payload or {})
            async with self._qr_lock:
                if generation != self._generation:
                    result["qr"] = None
                    result["qr_available"] = False
                else:
                    self._filter_status_qr(result, payload, generation)
            if result["state"] in targets:
                return result
            if result["sequence"] > after:
                after = int(result["sequence"])

    @staticmethod
    def _public_message(raw: Any) -> dict[str, Any] | None:
        """Normalize one bridge message; drop anything malformed."""
        if not isinstance(raw, dict):
            return None
        message_id = raw.get("id")
        chat = raw.get("chat")
        sender = raw.get("sender")
        text = raw.get("text")
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= MAX_MESSAGE_ID_CHARS:
            return None
        if not _is_jid(chat) or not _is_jid(sender):
            return None
        if not isinstance(text, str) or not text or len(text) > MAX_MESSAGE_TEXT_CHARS:
            return None
        timestamp = raw.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            timestamp = 0
        chat_alt = raw.get("chat_alt")
        sender_alt = raw.get("sender_alt")
        return {
            "id": message_id,
            "chat": chat,
            "sender": sender,
            "chat_alt": chat_alt if _is_jid(chat_alt) else None,
            "sender_alt": sender_alt if _is_jid(sender_alt) else None,
            "text": text,
            "timestamp": _bounded_int(timestamp),
            "is_group": raw.get("is_group") is True,
        }

    async def messages(
        self,
        owner_id: str | None = None,
        *,
        after: int = 0,
        timeout: float = 10.0,
        limit: int = MAX_MESSAGE_BATCH,
        auth: str | None = None,
    ) -> dict[str, Any]:
        """Authenticated long-poll for inbound messages after ``after``.

        Content is returned only to the authenticated manager; the bridge never
        logs it and the cursor (``sequence``) is monotonic for this process.
        """
        self._require_auth(owner_id, auth)
        self._require_running()
        if isinstance(after, bool) or not isinstance(after, int) or not 0 <= after <= MAX_FORWARDED_INT:
            raise BridgeError("after must be a non-negative integer.", code="invalid_request")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_MESSAGE_TIMEOUT:
            raise BridgeError("Message timeout must be between 0 and 30 seconds.", code="invalid_timeout")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_MESSAGE_BATCH:
            raise BridgeError(f"limit must be between 1 and {MAX_MESSAGE_BATCH}.", code="invalid_request")
        payload = await self._request("POST", "/messages", {
            "after": after,
            "timeout_ms": max(100, min(int(MAX_MESSAGE_TIMEOUT * 1000), int(float(timeout) * 1000))),
            "limit": limit,
        }, timeout=float(timeout) + 5.0) or {}
        raw_messages = payload.get("messages")
        messages: list[dict[str, Any]] = []
        if isinstance(raw_messages, list):
            for raw in raw_messages[:limit]:
                public = self._public_message(raw)
                if public is not None:
                    messages.append(public)
        return {
            "messages": messages,
            "sequence": _bounded_int(payload.get("sequence")),
        }

    async def send(
        self,
        owner_id: str | None = None,
        *,
        to: str,
        text: str,
        auth: str | None = None,
    ) -> dict[str, Any]:
        """Send one text message through the linked session."""
        self._require_auth(owner_id, auth)
        self._require_running()
        if not _is_jid(to):
            raise BridgeError("Invalid WhatsApp target.", code="invalid_target")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_TEXT_CHARS:
            raise BridgeError(
                f"WhatsApp text must be 1-{MAX_MESSAGE_TEXT_CHARS} characters.",
                code="invalid_text",
            )
        payload = await self._request("POST", "/send", {"to": to, "text": text}, timeout=20.0) or {}
        message_id = payload.get("id")
        return {
            "to": to,
            "id": message_id if isinstance(message_id, str) and len(message_id) <= MAX_MESSAGE_ID_CHARS else None,
        }

    async def typing(
        self,
        owner_id: str | None = None,
        *,
        to: str,
        state: str = "composing",
        auth: str | None = None,
    ) -> None:
        """Publish a chat-state presence ("composing" / "paused") for one chat.

        WhatsApp expires a composing state after a few seconds, so callers that
        do long work must repeat this. Failures are non-fatal for the caller.
        """
        self._require_auth(owner_id, auth)
        self._require_running()
        if not _is_jid(to):
            raise BridgeError("Invalid WhatsApp target.", code="invalid_target")
        if state not in {"composing", "paused"}:
            raise BridgeError("Invalid presence state.", code="invalid_request")
        await self._request("POST", "/typing", {"to": to, "state": state}, timeout=10.0)

    async def wait_for_connection(
        self,
        owner_id: str | None = None,
        timeout: float = 30.0,
        *,
        auth: str | None = None,
    ) -> dict[str, Any]:
        return await self.wait(
            owner_id,
            states={"connected", "logged_out", "error", "passkey_required", "cancelled"},
            timeout=timeout,
            auth=auth,
        )

    async def cancel(self, owner_id: str | None = None, *, auth: str | None = None) -> dict[str, Any]:
        """Abort the current attempt without deleting saved credentials."""
        self._require_auth(owner_id, auth)
        async with self._lifecycle_lock:
            if not self.is_running:
                return self._local_status()
            generation = self._advance_generation()
            payload = await self._request("POST", "/cancel", {})
            if generation != self._generation or not self.is_running:
                return self._local_status()
            public = self._public_status(payload or {})
            async with self._qr_lock:
                return self._filter_status_qr(public, payload, generation)

    async def repair(
        self,
        owner_id: str | None = None,
        *,
        forget: bool = False,
        auth: str | None = None,
    ) -> dict[str, Any]:
        """Start a fresh pairing attempt. ``forget`` destroys saved credentials."""
        self._require_auth(owner_id, auth)
        async with self._lifecycle_lock:
            self._require_running()
            if not isinstance(forget, bool):
                raise BridgeError("forget must be a bool.", code="invalid_request")
            generation = self._advance_generation()
            payload = await self._request("POST", "/re-pair", {"forget": forget})
            if generation != self._generation or not self.is_running:
                return self._local_status()
            public = self._public_status(payload or {})
            async with self._qr_lock:
                return self._filter_status_qr(public, payload, generation)

    async def _shutdown(self) -> None:
        """Internal teardown; intentionally bypasses the public auth gate."""
        self._advance_generation()
        proc = self._proc
        client = self._client
        listener = self._listener
        self._proc = None
        self._client = None
        self._listener = None
        self._port = None
        try:
            if listener is not None:
                try:
                    listener.close()
                except OSError:
                    pass

            if proc is not None:
                if getattr(proc, "returncode", None) is None:
                    try:
                        await _maybe_await(proc.terminate())
                    except (ProcessLookupError, OSError):
                        pass
                try:
                    await asyncio.wait_for(_maybe_await(proc.wait()), timeout=self.config.stop_timeout)
                except (asyncio.TimeoutError, OSError):
                    try:
                        await _maybe_await(proc.kill())
                    except (ProcessLookupError, OSError):
                        pass
                    try:
                        await asyncio.wait_for(_maybe_await(proc.wait()), timeout=1.0)
                    except (asyncio.TimeoutError, OSError):
                        pass

            for task in (self._stdout_task, self._stderr_task):
                if task is not None:
                    task.cancel()
            tasks = [task for task in (self._stdout_task, self._stderr_task) if task is not None]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._stdout_task = None
            self._stderr_task = None

            if client is not None and hasattr(client, "aclose"):
                try:
                    await _maybe_await(client.aclose())
                except BaseException:
                    # A broken transport can raise anything (including a
                    # non-HTTP error). It must never strand the bearer token.
                    pass
                finally:
                    self._client = None
                    self._token = ""
            # Only after the streams are drained: the redactor needs the token
            # while child output may still be arriving.
        finally:
            # Every failure path erases the transport and the token, even if
            # teardown itself raised before reaching the normal cleanup point.
            self._client = None
            self._token = ""
            self._stdout_task = None
            self._stderr_task = None

    async def stop(self, owner_id: str | None = None, *, auth: str | None = None) -> None:
        """Authenticated teardown: child, listener, transport and token."""
        self._require_auth(owner_id, auth)
        async with self._lifecycle_lock:
            await self._shutdown()

    aclose = stop

    async def __aenter__(self) -> WhatsAppBridgeManager:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self._shutdown()


__all__ = [
    "BridgeAuthError",
    "BridgeConfig",
    "BridgeError",
    "BridgeNotEnabled",
    "BridgeTimeout",
    "BridgeUnavailable",
    "PROTOCOL",
    "SessionAuthError",
    "SessionOwnershipError",
    "TERMINAL_STATES",
    "WhatsAppBridgeManager",
]
