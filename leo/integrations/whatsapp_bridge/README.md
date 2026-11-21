# WhatsApp Web device-link bridge (opt-in, not wired in)

This directory contains a **standalone, explicitly enabled** bridge that can
link a personal WhatsApp account via the unofficial WhatsApp Web protocol
([Baileys](https://github.com/WhiskeySockets/Baileys)). It is a reimplementation
from observed protocol behavior, not copied source.

**Status: offline-capable code only.** Nothing in LEO imports or starts this
bridge automatically. Linking a personal account carries WhatsApp ToS and
account-ban risk; linked-device credentials are a bearer credential for the
whole account. A maintainer must make that policy decision before any real
linking. The existing WhatsApp Cloud API adapter remains the official path.

## Files

| File | Role |
| --- | --- |
| `bridge.js` | Node ESM process. Inherits the parent-bound loopback listener FD, Baileys socket, generation-checked QR lifecycle, token-gated HTTP API with no-store responses, sanitized NDJSON events on stdout. Its SHA-256 is pinned in `manager.py`. |
| `package.json` | Pins `@whiskeysockets/baileys` 7.0.0-rc13, `express` 4.21.2, `qrcode` 1.5.4; `npm start` runs the bridge. |
| `package-lock.json` | Locked transitive dependency graph. Use `npm ci`, not a floating `npm install`, when reproducing a deployment. |
| `manager.py` | Async Python manager: binds and keeps the listener, verifies the child's FD, pins the Node binary and script digest, owns the random token and per-session auth secret, owns the QR generation/publication watermark and generation-scoped replay ledger, normalizes/stream-limits payloads. |
| `__init__.py` | Public library boundary; importing it does not launch anything. |

## Operator setup (explicit steps)

1. Install **Node 20+** (`node --version`).
2. Install dependencies in this directory — this is a deliberate operator step,
   never performed by the Python manager and never on app startup:

   ```sh
   cd leo/integrations/whatsapp_bridge   # from the repository root
   npm ci               # uses the committed package-lock.json
   npm run check        # node --check bridge.js
   ```

   Node is resolved only through `shutil.which("node")` (or an explicit
   operator-pinned absolute path in `BridgeConfig.node_path_allowlist`). The
   manager resolves the realpath, checks it is an executable and runs
   `node --version` before launch; arbitrary binaries and script directories
   are rejected, and `bridge.js` must match the SHA-256 pinned in `manager.py`.

   **This is not cryptographic identity.** A malicious executable on the local
   machine can simply print `v20.x.y` for the version probe. For production,
   configure an operator-pinned absolute realpath and verify its hash out of
   band (package signature, system package manager, or your own pinned hash).
   The probe only rejects obviously wrong binaries; it does not prove the
   binary is Node.

3. Add the following to the repository `.gitignore` **before** linking a device
   (the repo already ignores `.leo/` and `node_modules/`, these rules make the
   bridge-specific intent explicit and still cover a moved session root):

   ```gitignore
   # WhatsApp Web bridge (linked-session credentials are account bearer tokens)
   .leo/whatsapp/
   leo/integrations/whatsapp_bridge/node_modules/
   leo/integrations/whatsapp_bridge/.env
   auth_info_baileys/
   whatsapp-session*
   ```

   Verify with `git check-ignore -v .leo/whatsapp/default/creds.json`. A linked
   session must never be committed, attached to an issue, or logged.

4. Enable in code by constructing `WhatsAppBridgeManager(BridgeConfig(owner_id=...,
   enabled=True))` from a caller that has already checked the policy decision.
   The manager generates a fresh `secrets.token_urlsafe(32)` per process and
   passes it only in the child environment (`LEO_WHATSAPP_BRIDGE_TOKEN`).
   Do not set that variable yourself; do not put it in a URL, argv, `.env`, or
   a log line.

   Every public manager call (`start`, `status`, `qr`, `wait`, `cancel`,
   `repair`, `stop`) requires `auth=manager.session_auth` in addition to the
   owner id. The owner id is an identity claim, not a credential: it is never
   accepted as sole authority. Keep `session_auth` in the authenticated caller's
   server-side session; never put it in a URL or log. The session root is pinned
   to the application repository (no caller redirect), symlinks are rejected,
   and credential files are 0600.

Example (library use; no network until `start()` is called):

```python
from leo.integrations.whatsapp_bridge import BridgeConfig, WhatsAppBridgeManager

async with WhatsAppBridgeManager(BridgeConfig(owner_id="alice", actor="alice", enabled=True)) as bridge:
    auth = bridge.session_auth                 # keep in the authenticated session
    await bridge.start(owner_id="alice", auth=auth)   # launch child + begin pairing
    while True:
        status = await bridge.status(owner_id="alice", auth=auth)
        if status["qr_available"]:
            qr_png_data_uri = status["qr"]["data_uri"]   # send to the owner's UI only
        result = await bridge.wait(owner_id="alice", timeout=10, auth=auth)
        if result["state"] in {"connected", "logged_out", "error", "passkey_required", "cancelled"}:
            break
    # bridge.cancel(...), bridge.repair(..., auth=auth), bridge.repair(..., forget=True), bridge.qr(...)
```

## Wire protocol (all routes require `Authorization: Bearer <owner token>`)

The manager **binds and keeps** a loopback listener, then passes its FD to the
child via `pass_fds`; the child serves on that inherited FD. There is no window
for a local port squatter, and no token is sent until the child/FD ownership is
verified. The manager sends the token as a header over `http://127.0.0.1:<port>`;
the port is allocated at launch. The bridge rejects unknown Host headers
(DNS-rebinding defense). Bodies are JSON, capped at 8 KB; all `/qr` and `/status`
responses are `Cache-Control: no-store` and `Vary: Authorization`.

| Route | Purpose |
| --- | --- |
| `GET /health` | Liveness and protocol version. |
| `GET /status` | State machine snapshot: `starting`, `waiting_for_qr`, `waiting_for_scan`, `connected`, `logged_out`, `passkey_required`, `error`, `cancelled`. Includes QR metadata but never the raw Baileys QR string. |
| `GET /qr` | Current QR as a bounded PNG data URI with `expires_at`/`rotation`; 404 when expired or not yet issued. |
| `POST /wait` | `{after: <sequence>, timeout_ms: 100..30000}` long-poll for a state change. |
| `POST /start` | Begin a link attempt (idempotent while connected). |
| `POST /cancel` | Abort the attempt, drop the QR, keep saved credentials. |
| `POST /re-pair` | `{forget: true}` also removes saved credentials before restarting. Destructive; requires explicit opt-in. Answers `409 creds_drain_failed` instead of success when queued credential saves cannot be proven drained; writes are fenced and no forget is attempted. |

Baileys emits QR strings on `connection.update`; renders are serialized and
generation-checked, so a QR that finishes after cancel/re-pair or after a newer
QR is dropped. Each published rotation yields a fresh data URI, a new rotation
number, and a TTL (default 30 s, bounded 5–120 s). `linked` is only reported
while the phase is `connected` and the process is running; cancel and re-pair
clear it. The manager additionally tags each published QR with a **manager-owned
identity**: the monotonic lifecycle generation plus a publication sequence that
the manager increments only when it publishes a genuinely new, validated image.
A render from an older generation is dropped by both `/status` and `/qr`.
Generation advances only on start/cancel/re-pair/shutdown. Child-supplied
`generation` and `rotation` are bounded informational metadata (rotation is
still returned to the caller) and are **never an ordering key**: a forged
`rotation: 2147483647` on a novel image cannot raise the watermark and block
later legitimate QRs.

**QR images are single-use per generation.** The manager hashes the decoded PNG
bytes of every image it publishes (SHA-256) and records that hash in the
consumed ledger for the current generation. A digest published once cannot be
published again through `/status` or `/qr` during that generation, even after
65+ other images. Ordinary re-reads of the one live QR are served from the
manager's own cache; any reappearance of a consumed image with a different
rotation, an older generation, or after another image was published is
rejected as a replay. The ledger never evicts a digest *within* its generation:
reaching the 10,000-image cap fails closed for that generation instead of
forgetting a consumed hash. A legitimate generation advance archives the ledger
and installs a fresh one, so a hostile child cannot spend that budget forever.
The last 8 generations' ledgers remain searchable so cross-generation replays
are still rejected; archives past that bounded window are discarded, and a QR
is expired long before then.

Credential saves are drained and frozen before a forget. If saves requeue
themselves past the drain limit, the handler is replaced, the forget is
aborted, and the request fails closed rather than reporting success with
`creds.json` still writable.
On expiry the bridge drops the QR; if none arrives within 120 s it reports
`error: qr_timeout`. `connection === 'close'` maps `lastDisconnect.error`
status 401 to terminal `logged_out`, 515 to a 1 s restart, other reasons to a
3 s restart (bounded at 10 attempts). Passkey challenges are surfaced as
`passkey_required` with a sanitized NDJSON event; the bridge does not implement
WebAuthn signing, and `LEO_WHATSAPP_PASSKEY_MODE=exit` exits 78 for supervisors
that want to handle it elsewhere (upstream convention).

NDJSON stdout events contain only `ts`, `event`, `actor`, `sequence`, `phase`
and a few allowlisted booleans/codes. The token, QR string and QR image are
never printed. The manager additionally redacts token text and data URIs from
bounded child-output tails.

## Security notes

* Session state lives under `.leo/whatsapp/<actor>/` (mode 0700); `owner.json`
  records the single owner id (mode 0600) and is re-checked on every manager
  call. Actor ids are validated and path traversal is rejected.
* The child env is an allowlist (`PATH`, `HOME`, temp vars, locale, plus the
  `LEO_WHATSAPP_*` settings above). Inherited API keys, tokens and `NODE_OPTIONS`
  are dropped; no shell is used; cwd is this directory. Only the resolved Node
  realpath and pinned `bridge.js` digest may run. Node identity is a local
  compatibility probe, not a cryptographic guarantee: pin and hash the realpath
  yourself in production.
* A per-process random token is required on **every** route, compared in
  constant time by the bridge. Missing/wrong tokens produce generic 401s.
  Manager calls additionally require the per-session `session_auth` secret;
  knowing the owner id is not enough. Child output is redacted before display
  truncation; response bodies are streamed under an incremental byte cap and
  all forwarded fields are bounded.
* Baileys logs are silenced (its records can contain QR strings). Fatal stderr
  messages are generic.
* The manager normalizes all responses through allowlists and rejects malformed
  or oversized QR data URIs. QR replay defense is hash-based and
  generation-scoped: digests are never evicted within a generation, and a
  bounded archive of past generations' ledgers rejects cross-generation
  replays. Both ordering keys (generation and publication sequence) are
  manager-owned; child-supplied rotation is informational metadata only, so it
  cannot poison the watermark or permanently disable QR publication. Timeouts
  bound startup, waits and shutdown; `stop()` terminates, then kills if needed,
  and always clears the in-memory token.
* Tests (`tests/test_whatsapp_bridge.py`) use an in-memory fake child process
  and fake HTTP transport. No WhatsApp account, browser, npm install, or
  external network is involved.
