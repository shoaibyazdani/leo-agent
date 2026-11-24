#!/usr/bin/env node
/**
 * LEO WhatsApp Web device-link bridge (reimplementation from observed behavior).
 *
 * This is a standalone, opt-in local process. It is NOT started by LEO unless a
 * caller explicitly enables it (see manager.py). It speaks a small, token-gated
 * JSON control protocol over loopback HTTP and emits sanitized NDJSON lifecycle
 * events on stdout. It never prints the owner token, the raw Baileys QR string,
 * or the QR image to stdout/stderr.
 *
 * Runtime contract (all values are supplied by the parent as environment):
 *   LEO_WHATSAPP_BRIDGE_TOKEN   required, 32..256 chars, [A-Za-z0-9_-]
 *   LEO_WHATSAPP_ACTOR          required, [A-Za-z0-9][A-Za-z0-9_-]{0,63}
 *   LEO_WHATSAPP_SESSION_ROOT   absolute .leo/whatsapp root
 *   LEO_WHATSAPP_SESSION_DIR    absolute session directory inside the root
 *   LEO_WHATSAPP_BRIDGE_FD      inherited listening TCP socket (bound by the parent)
 *   LEO_WHATSAPP_BRIDGE_PORT    loopback TCP port (informational; FD is authoritative)
 * Optional: LEO_WHATSAPP_QR_TTL_MS, LEO_WHATSAPP_QR_MAX_CHARS,
 *           LEO_WHATSAPP_PASSKEY_MODE=report|exit (exit uses upstream code 78).
 *
 * npm install is an explicit operator step; see README.md. No network access
 * happens on startup beyond Baileys' own version lookup and socket connection.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

import express from 'express';
import QRCode from 'qrcode';

const PROTOCOL = 'whatsapp-bridge/1';
const TOKEN_RE = /^[A-Za-z0-9_-]{32,256}$/;
const ACTOR_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const HOST_ALLOWLIST = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);
const MAX_BODY_BYTES = '8kb';
const MAX_JSON_NUMBER = 2 ** 31 - 1;
const DEFAULT_QR_TTL_MS = 30_000;
const MIN_QR_TTL_MS = 5_000;
const MAX_QR_TTL_MS = 120_000;
const DEFAULT_QR_MAX_CHARS = 2048;
const QR_URI_LIMIT = 65_536;
const QR_WAIT_TIMEOUT_MS = 120_000;
const RECONNECT_FAST_MS = 1_000;   // Baileys reason 515 (restart after pairing)
const RECONNECT_SLOW_MS = 3_000;
const MAX_RECONNECTS = 10;
const PASSKEY_REQUIRED_EXIT = 78;
const CREDS_DRAIN_LIMIT = 64;
// Message transport bounds. Text is capped so one JSON body (and one response
// batch) stays well under MAX_BODY_BYTES / the manager's response cap.
const MAX_MESSAGE_TEXT_CHARS = 2_000;
const MAX_QUEUED_MESSAGES = 200;
const MAX_MESSAGE_BATCH = 10;
const MAX_MESSAGE_DEDUPE = 500;
// WhatsApp JIDs are opaque provider ids: an account/device/group identifier
// followed by "@domain". Accept any non-whitespace, non-control string; the
// old strict alphabet rejected legitimate modern ids (e.g. LID/device forms).
const JID_RE = /^[^\s\x00-\x1f]{3,128}$/;
const MESSAGE_WAIT_MAX_MS = 30_000;

function isJid(value) {
  return typeof value === 'string' && value.includes('@') && JID_RE.test(value);
}

function jidShape(value) {
  if (typeof value !== 'string') return { type: typeof value };
  const at = value.lastIndexOf('@');
  const bad = [];
  for (const char of value) {
    if (/[\s\x00-\x1f]/.test(char) && bad.length < 5) bad.push(char.codePointAt(0));
  }
  return {
    len: value.length,
    domain: safeCode(at >= 0 ? value.slice(at + 1) : '', 'none'),
    has_at: at >= 0,
    bad,
  };
}

const ERR = {
  config: 2,
  bind: 3,
  generic: 1,
};

function fatal(message, code) {
  // Deliberately generic: no environment values, error messages or stack traces.
  process.stderr.write(`bridge fatal: ${message}\n`);
  process.exit(code);
}

function intEnv(name, fallback, min, max) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  if (!/^\d+$/.test(raw)) fatal(`invalid ${name}`, ERR.config);
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < min || value > max) fatal(`invalid ${name}`, ERR.config);
  return value;
}

const token = process.env.LEO_WHATSAPP_BRIDGE_TOKEN || '';
const actor = process.env.LEO_WHATSAPP_ACTOR || '';
const sessionRootRaw = process.env.LEO_WHATSAPP_SESSION_ROOT || '';
const sessionDirRaw = process.env.LEO_WHATSAPP_SESSION_DIR || '';
const port = intEnv('LEO_WHATSAPP_BRIDGE_PORT', 0, 1, 65535);
const bridgeFd = intEnv('LEO_WHATSAPP_BRIDGE_FD', 0, 3, 65535);
const qrTtlMs = intEnv('LEO_WHATSAPP_QR_TTL_MS', DEFAULT_QR_TTL_MS, MIN_QR_TTL_MS, MAX_QR_TTL_MS);
const qrMaxChars = intEnv('LEO_WHATSAPP_QR_MAX_CHARS', DEFAULT_QR_MAX_CHARS, 256, 4096);
const passkeyMode = process.env.LEO_WHATSAPP_PASSKEY_MODE === 'exit' ? 'exit' : 'report';

if (!TOKEN_RE.test(token)) fatal('missing or invalid owner token', ERR.config);
if (!ACTOR_RE.test(actor)) fatal('invalid actor', ERR.config);
if (!sessionRootRaw || !path.isAbsolute(sessionRootRaw) || !sessionDirRaw || !path.isAbsolute(sessionDirRaw)) {
  fatal('invalid session paths', ERR.config);
}
const sessionRoot = path.resolve(sessionRootRaw);
const sessionDir = path.resolve(sessionDirRaw);
if (path.relative(sessionRoot, sessionDir).startsWith('..') || path.relative(sessionRoot, sessionDir) === '') {
  fatal('session directory escapes its root', ERR.config);
}
if (sessionDir !== path.join(sessionRoot, actor)) fatal('session directory does not match actor', ERR.config);
if (!bridgeFd) fatal('missing inherited listener', ERR.config);

// --- Sanitized introspection ------------------------------------------------

const state = {
  phase: 'idle',          // idle|starting|waiting_for_qr|waiting_for_scan|connected|logged_out|passkey_required|error|cancelled
  linked: false,
  errorCode: null,        // generic code only, never a raw error message
  user: null,             // {id, name} from Baileys; display metadata only
  passkeyRequired: null,  // {has_signer}
  qr: null,               // {data_uri, expires_at, rotation, sequence}
  rotation: 0,
  sequence: 0,
  reconnectAttempt: 0,
  startedAt: new Date().toISOString(),
  stopping: false,
};

const waiters = new Set();
// Inbound direct/group messages, bounded in count and text length. Content is
// never written to NDJSON logs; only an authenticated /messages caller reads it.
const messageQueue = [];
const messageWaiters = new Set();
const messageIdOrder = [];
const messageIds = new Set();
let messageSequence = 0;
let tokenBytes = Buffer.from(token, 'utf8');
let socket = null;
let attempt = 0;
let qrTimer = null;
let qrWatchdog = null;
let reconnectTimer = null;
let credsChain = Promise.resolve();
// Credential writes are frozen synchronously whenever a socket is torn down;
// only a freshly installed socket handler re-enables them. This prevents a
// delayed `creds.update` save from recreating creds.json after a forget.
let credsWritesBlocked = false;
let credsEpoch = 0;
// The active write handler is replaceable: fencing a runaway chain drops it so
// no captured listener can enqueue another save after a failed drain.
let credsWriteHandler = null;

class CredsDrainError extends Error {
  constructor() {
    super('creds_drain_limit');
    this.name = 'CredsDrainError';
    this.bridgeErrorCode = 'creds_drain_failed';
  }
}
let shutdownCode = 0;
// QR renders are serialized and generation-checked. A render that finishes
// after cancel/re-pair or after a newer QR was issued is dropped, never published.
let qrRenderGeneration = 0;
let qrRenderChain = Promise.resolve();

function safeCode(raw, fallback = 'unspecified_error') {
  if (typeof raw !== 'string' || !/^[a-z0-9_]{1,40}$/.test(raw)) return fallback;
  return raw;
}

function emit(event, fields = {}) {
  // Allowlist: unknown fields from Baileys or callers can never reach stdout.
  const line = JSON.stringify({
    ts: Date.now(),
    event: safeCode(event, 'event'),
    actor,
    sequence: state.sequence,
    phase: state.phase,
    ...fields,
  });
  process.stdout.write(`${line}\n`);
}

function snapshot(extra = {}) {
  return {
    protocol: PROTOCOL,
    actor,
    phase: state.phase,
    state: state.phase,
    linked: state.linked,
    error: state.errorCode,
    passkey_required: state.passkeyRequired,
    user: state.user,
    qr: state.qr,
    rotation: state.rotation,
    sequence: state.sequence,
    reconnect_attempt: state.reconnectAttempt,
    started_at: state.startedAt,
    uptime_ms: Date.now() - Date.parse(state.startedAt),
    ...extra,
  };
}

function setPhase(phase, errorCode = null) {
  state.phase = phase;
  state.errorCode = safeCode(errorCode);
  state.sequence += 1;
  emit('state', { phase, error: state.errorCode });
  for (const waiter of [...waiters]) {
    if (waiter.after !== state.sequence) {
      clearTimeout(waiter.timer);
      waiters.delete(waiter);
      waiter.resolve(snapshot());
    }
  }
}

// --- Inbound message queue --------------------------------------------------

function rememberMessageId(id) {
  // Bounded FIFO dedupe: Baileys may re-deliver after a reconnect.
  if (messageIds.has(id)) return false;
  messageIds.add(id);
  messageIdOrder.push(id);
  while (messageIdOrder.length > MAX_MESSAGE_DEDUPE) {
    messageIds.delete(messageIdOrder.shift());
  }
  return true;
}

function unwrapMessage(message) {
  let current = message;
  for (let depth = 0; depth < 3 && current; depth += 1) {
    const wrapped = current.ephemeralMessage?.message
      || current.viewOnceMessage?.message
      || current.viewOnceMessageV2?.message
      || current.documentWithCaptionMessage?.message;
    if (!wrapped) break;
    current = wrapped;
  }
  return current;
}

function extractMessageText(message) {
  const content = unwrapMessage(message);
  if (!content || typeof content !== 'object') return '';
  const text = content.conversation
    ?? content.extendedTextMessage?.text
    ?? content.imageMessage?.caption
    ?? content.videoMessage?.caption
    ?? content.documentMessage?.caption
    ?? content.buttonsResponseMessage?.selectedDisplayText
    ?? content.listResponseMessage?.title
    ?? '';
  return typeof text === 'string' ? text.slice(0, MAX_MESSAGE_TEXT_CHARS) : '';
}

function drainMessages(after, limit) {
  const pending = [];
  for (const message of messageQueue) {
    if (message.sequence > after) pending.push(message);
    if (pending.length >= limit) break;
  }
  return pending;
}

function resolveMessageWaiters() {
  for (const waiter of [...messageWaiters]) {
    const messages = drainMessages(waiter.after, waiter.limit);
    if (!messages.length) continue;
    clearTimeout(waiter.timer);
    messageWaiters.delete(waiter);
    waiter.resolve({
      messages,
      sequence: messageSequence,
      timed_out: false,
    });
  }
}

function pushMessage(message) {
  if (!rememberMessageId(message.id)) return;
  messageSequence += 1;
  message.sequence = messageSequence;
  messageQueue.push(message);
  while (messageQueue.length > MAX_QUEUED_MESSAGES) messageQueue.shift();
  // Never log the content, sender or chat id: only that a message arrived.
  emit('message_received', { sequence: messageSequence });
  resolveMessageWaiters();
}

function handleMessagesUpsert(payload) {
  if (!payload || !Array.isArray(payload.messages)) return;
  // 'notify' is a live message; 'append' is history/offline. Both are safe to
  // surface, but restricted delivery receipts and protocol stubs are skipped.
  const kind = typeof payload.type === 'string' ? payload.type : '';
  if (kind && kind !== 'notify' && kind !== 'append') return;
  if (!payload.messages.length) return;
  // Sanitized counters only: never a JID, sender, or message text.
  const stats = {
    total: payload.messages.length,
    accepted: 0, from_me: 0, bad_jid: 0, stub: 0, protocol: 0, no_text: 0, bad_id: 0, error: 0,
  };
  for (const raw of payload.messages) {
    try {
      const key = raw?.key;
      if (!key) { stats.bad_jid += 1; stats.chat_shape = jidShape(undefined); continue; }
      if (key.fromMe === true) { stats.from_me += 1; continue; }
      const chat = isJid(key.remoteJid) ? key.remoteJid : '';
      if (!chat) {
        stats.bad_jid += 1;
        stats.chat_shape = jidShape(key.remoteJid);
        continue;
      }
      if (chat === 'status@broadcast' || chat.endsWith('@broadcast')) { stats.bad_jid += 1; continue; }
      // In group messages participant is the sender; in direct chats it may be
      // absent or a device-qualified id, so the chat JID is the safe fallback.
      const sender = isJid(key.participant) ? key.participant : chat;
      if (raw?.messageStubType !== undefined && raw?.messageStubType !== null) { stats.stub += 1; continue; }
      if (raw?.message?.protocolMessage) { stats.protocol += 1; continue; }
      const id = typeof key.id === 'string' && key.id.length <= 128 ? key.id : '';
      if (!id) { stats.bad_id += 1; continue; }
      const text = extractMessageText(raw.message);
      if (!text.trim()) { stats.no_text += 1; continue; }
      const seconds = Number(raw.messageTimestamp);
      const timestamp = Number.isFinite(seconds) && seconds >= 0 && seconds <= MAX_JSON_NUMBER
        ? Math.floor(seconds) : Math.floor(Date.now() / 1000);
      // WhatsApp increasingly addresses direct chats by LID. Baileys supplies
      // the phone-number alternative in *_Alt/senderPn; surface it so the
      // caller can match its allowlist and reply to the PN JID when present.
      const chatAlt = isJid(key.remoteJidAlt) ? key.remoteJidAlt : '';
      const senderAlt = isJid(key.participantAlt) ? key.participantAlt
        : (isJid(key.senderPn) ? key.senderPn : '');
      pushMessage({
        id,
        chat,
        sender,
        chat_alt: chatAlt,
        sender_alt: senderAlt,
        text,
        timestamp,
        is_group: chat.endsWith('@g.us'),
      });
      stats.accepted += 1;
    } catch {
      // One malformed message must never break the socket or the stream.
      stats.error += 1;
    }
  }
  emit('upsert_summary', { kind: safeCode(kind, 'unknown'), ...stats });
}

function clearMessageQueue() {
  messageQueue.length = 0;
  for (const waiter of [...messageWaiters]) {
    clearTimeout(waiter.timer);
    messageWaiters.delete(waiter);
    waiter.resolve({ messages: [], sequence: messageSequence, timed_out: true });
  }
}

function clearTimers() {
  for (const timer of [qrTimer, qrWatchdog, reconnectTimer]) {
    if (timer) clearTimeout(timer);
  }
  qrTimer = null;
  qrWatchdog = null;
  reconnectTimer = null;
}

function clearQr(reason = 'qr_expired') {
  const had = state.qr !== null;
  state.qr = null;
  if (had) emit(reason, { rotation: state.rotation });
}

function scheduleQrExpiry(rotation) {
  if (qrTimer) clearTimeout(qrTimer);
  qrTimer = setTimeout(() => {
    qrTimer = null;
    if (state.qr && state.qr.rotation === rotation) {
      clearQr('qr_expired');
      state.sequence += 1;
    }
  }, qrTtlMs);
  qrTimer.unref?.();
}

function armQrWatchdog(thisAttempt) {
  if (qrWatchdog) clearTimeout(qrWatchdog);
  qrWatchdog = setTimeout(() => {
    qrWatchdog = null;
    if (thisAttempt === attempt && state.phase === 'waiting_for_qr' && !state.qr) {
      setPhase('error', 'qr_timeout');
    }
  }, QR_WAIT_TIMEOUT_MS);
  qrWatchdog.unref?.();
}

// --- Baileys lifecycle ------------------------------------------------------

// Silent logger: Baileys log records can include QR strings and device names.
// All diagnostics from this process go through emit()/fatal() above.
const silentLogger = {
  level: 'silent',
  trace() {}, debug() {}, info() {}, warn() {}, error() {}, child() { return silentLogger; },
};

function disconnectStatus(update) {
  const error = update?.lastDisconnect?.error;
  const direct = error?.output?.statusCode ?? error?.statusCode;
  return Number.isInteger(direct) ? direct : 0;
}

function fenceCredsWrites() {
  // Replace (not merely disable) the handler and detach from the runaway
  // chain. Queued saves still check the blocked flag and their captured epoch,
  // so none of them can write after this point.
  credsWritesBlocked = true;
  credsEpoch += 1;
  credsWriteHandler = null;
  credsChain = Promise.resolve();
  const current = socket;
  try {
    current?.ev?.removeAllListeners?.('creds.update');
  } catch {
    // A half-open Baileys socket can throw; never leak the error text.
  }
}

async function drainCredsChain() {
  // A queued save can schedule another one; drain until the chain is stable.
  for (let i = 0; i < CREDS_DRAIN_LIMIT; i++) {
    const pending = credsChain;
    await pending.catch(() => {});
    if (credsChain === pending) return;
  }
  // Fail closed: a self-requeuing chain can never be proven drained, so the
  // caller must NOT delete credentials and report a successful forget.
  fenceCredsWrites();
  throw new CredsDrainError();
}

async function closeSocket() {
  // Freeze writes BEFORE the first await. A creds.update callback that fires
  // mid-teardown can no longer enqueue a save that outlives a forget.
  credsWritesBlocked = true;
  credsWriteHandler = null;
  const current = socket;
  socket = null;
  if (current) {
    try {
      current.ev?.removeAllListeners?.('connection.update');
      current.ev?.removeAllListeners?.('creds.update');
    } catch {
      // A half-open Baileys socket can throw; never leak the error text.
    }
  }
  // Wait for every already queued save to settle before the caller is allowed
  // to delete the credential files. A chain that never settles is fenced and
  // reported, never treated as a successful drain.
  let drainFailure = null;
  try {
    await drainCredsChain();
  } catch (err) {
    drainFailure = err;
  }
  if (!current) {
    if (drainFailure) throw drainFailure;
    return;
  }
  try {
    current.end?.(undefined);
  } catch {
    // Closing a half-open Baileys socket can throw; never leak the error text.
  }
  if (drainFailure) throw drainFailure;
}

function handleConnection(update, thisAttempt) {
  if (thisAttempt !== attempt || state.stopping) return;

  if (typeof update.qr === 'string' && update.qr) {
    const rawQr = update.qr;
    if (rawQr.length > qrMaxChars) {
      setPhase('error', 'qr_too_large');
      void cancelAttempt().catch(() => setPhase('error', 'creds_drain_failed'));
      return;
    }
    const thisAttempt = attempt;
    const renderGen = ++qrRenderGeneration;
    // Serialize renders: the last requested QR wins, and only if its
    // generation is still current when the render finishes.
    qrRenderChain = qrRenderChain
      .then(() => publishQr(rawQr, thisAttempt, renderGen))
      .catch(() => {
        if (thisAttempt === attempt && renderGen === qrRenderGeneration &&
            !state.stopping && !state.linked && state.phase !== 'connected') {
          setPhase('error', 'qr_render_failed');
        }
      });
    return;
  }

  if (update.passkeyRequired && state.phase !== 'connected') {
    const hasSigner = Boolean(update.passkeyRequired.hasSigner ?? update.passkeyRequired);
    state.passkeyRequired = { has_signer: hasSigner };
    setPhase('passkey_required');
    emit('passkey_required', { has_signer: hasSigner });
    if (passkeyMode === 'exit') {
      state.stopping = true;
      shutdownCode = PASSKEY_REQUIRED_EXIT;
      void shutdown();
    }
    return;
  }
  if (update.passkeyError) {
    state.passkeyRequired = null;
    setPhase('error', 'passkey_error');
    return;
  }

  if (update.connection === 'open') {
    clearQr('qr_consumed');
    if (qrWatchdog) clearTimeout(qrWatchdog);
    qrWatchdog = null;
    state.linked = true;
    state.reconnectAttempt = 0;
    state.passkeyRequired = null;
    const user = socket?.user || {};
    const id = typeof user.id === 'string' ? user.id.slice(0, 128) : null;
    const name = typeof user.name === 'string' ? user.name.slice(0, 128) : null;
    state.user = id ? { id, name: name || null } : null;
    setPhase('connected');
    emit('connected', { linked: true });
    return;
  }

  if (update.connection === 'close') {
    const code = disconnectStatus(update);
    state.linked = false;
    state.user = null;
    clearQr('qr_consumed');
    emit('disconnected', { code });
    if (code === 401) {
      // Logged out is terminal: reconnecting with dead creds would spin forever.
      clearTimers();
      setPhase('logged_out');
      return;
    }
    const delay = code === 515 ? RECONNECT_FAST_MS : RECONNECT_SLOW_MS;
    if (state.reconnectAttempt >= MAX_RECONNECTS) {
      setPhase('error', 'reconnect_exhausted');
      return;
    }
    setPhase('waiting_for_qr', 'reconnecting');
    state.reconnectAttempt += 1;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (thisAttempt === attempt && !state.stopping) {
        void openSocket(true).catch(() => setPhase('error', 'creds_drain_failed'));
      }
    }, delay);
    reconnectTimer.unref?.();
  }
}

async function publishQr(rawQr, thisAttempt, renderGen) {
  if (thisAttempt !== attempt || renderGen !== qrRenderGeneration || state.stopping || state.linked) return;
  const dataUri = await withTimeout(QRCode.toDataURL(rawQr, {
    type: 'image/png',
    errorCorrectionLevel: 'M',
    margin: 2,
    width: 360,
  }), 5000);
  // Check AGAIN after the await: cancel/re-pair, a newer QR or a completed
  // connection invalidates this render. Publishing it would resurrect a
  // stale pairing credential.
  if (thisAttempt !== attempt || renderGen !== qrRenderGeneration || state.stopping ||
      state.linked || state.phase === 'connected') return;
  if (typeof dataUri !== 'string' || !dataUri.startsWith('data:image/png;base64,') || dataUri.length > QR_URI_LIMIT) {
    throw new Error('qr_render_failed');
  }
  if (qrWatchdog) clearTimeout(qrWatchdog);
  qrWatchdog = null;
  state.rotation += 1;
  const rotation = state.rotation;
  state.qr = {
    data_uri: dataUri,
    expires_at: Date.now() + qrTtlMs,
    rotation,
    sequence: state.sequence,
  };
  setPhase('waiting_for_scan');
  // QR content is intentionally absent from the event and from every log.
  emit('qr', { rotation, expires_at: state.qr.expires_at });
  scheduleQrExpiry(rotation);
}

async function withTimeout(promise, ms) {
  let handle;
  const timeout = new Promise((_, reject) => {
    handle = setTimeout(() => reject(new Error('timeout')), ms);
    handle.unref?.();
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (handle) clearTimeout(handle);
  }
}

async function openSocket(reconnect = false) {
  await closeSocket();
  clearTimers();
  clearQr('qr_consumed');
  const thisAttempt = ++attempt;
  qrRenderGeneration += 1;
  // A new socket is never a linked session until 'open' is observed again.
  state.linked = false;
  state.user = null;
  state.passkeyRequired = null;
  state.reconnectAttempt = reconnect ? state.reconnectAttempt : 0;
  setPhase('starting');

  try {
    // Dynamic import: a missing npm install yields a clean, reportable error
    // instead of preventing the control API from starting.
    const baileys = await import('@whiskeysockets/baileys');
    if (thisAttempt !== attempt || state.stopping) return;

    await fsp.mkdir(sessionDir, { recursive: true, mode: 0o700 });
    await hardenSessionFiles();
    const authState = await baileys.useMultiFileAuthState(sessionDir);
    let version;
    try {
      const latest = await withTimeout(baileys.fetchLatestBaileysVersion(), 3000);
      if (Array.isArray(latest?.version)) version = latest.version;
    } catch {
      // Offline operation: Baileys can use its bundled version.
    }
    if (thisAttempt !== attempt || state.stopping) return;

    const sock = baileys.makeWASocket({
      ...(version ? { version } : {}),
      auth: authState.state,
      logger: silentLogger,
      printQRInTerminal: false,
      browser: ['LEO Bridge', 'Chrome', '120.0'],
      syncFullHistory: false,
      markOnlineOnConnect: false,
      // Baileys 7 requires this for E2EE retry handshakes during pairing.
      getMessage: async () => ({ conversation: '' }),
    });
    socket = sock;
    credsEpoch += 1;
    credsChain = Promise.resolve();
    // Only this socket's handler may write now. Writes from a previous attempt
    // are permanently fenced by their captured epoch and the blocked flag.
    credsWritesBlocked = false;
    const writeEpoch = credsEpoch;
    const rawSaveCreds = typeof authState.saveCreds === 'function'
      ? authState.saveCreds.bind(authState)
      : async () => {};
    // Even a captured saveCreds callback cannot write after this attempt has
    // been fenced: every write goes through a guard bound to this epoch.
    authState.saveCreds = async () => {
      if (credsWritesBlocked || writeEpoch !== credsEpoch || credsWriteHandler === null) return;
      await rawSaveCreds();
      await hardenSessionFiles();
    };
    credsWriteHandler = () => {
      if (credsWritesBlocked || thisAttempt !== attempt || credsWriteHandler === null) return;
      credsChain = credsChain
        .then(() => {
          if (credsWritesBlocked || writeEpoch !== credsEpoch || credsWriteHandler === null) return;
          return authState.saveCreds();
        })
        .catch(() => {});
    };
    sock.ev.on('creds.update', credsWriteHandler);
    sock.ev.on('connection.update', (update) => {
      try {
        handleConnection(update, thisAttempt);
      } catch {
        setPhase('error', 'connection_handler_failed');
      }
    });
    sock.ev.on('messages.upsert', (payload) => {
      try {
        handleMessagesUpsert(payload);
      } catch {
        emit('message_handler_failed', {});
      }
    });
    setPhase('waiting_for_qr');
    armQrWatchdog(thisAttempt);
  } catch (err) {
    // A fenced credential chain is a hard failure: a re-pair must never
    // report success with credentials still writable.
    if (err instanceof CredsDrainError) throw err;
    if (thisAttempt === attempt && !state.stopping) setPhase('error', 'socket_failed');
  }
}

async function cancelAttempt() {
  attempt += 1;
  qrRenderGeneration += 1;
  clearMessageQueue();
  // Cancelling ends the linked session claim, not just the QR. Callers may
  // not keep reporting `linked` after cancel or re-pair.
  state.linked = false;
  state.user = null;
  state.passkeyRequired = null;
  clearTimers();
  clearQr('qr_consumed');
  // Awaited by every caller: no credential deletion or new socket may start
  // until the old socket's queued saves have fully drained.
  await closeSocket();
}

// Linked-device credentials are account bearer tokens: never leave them
// group/world readable, and never follow a symlink planted in the session dir.
async function hardenSessionFiles() {
  let entries;
  try {
    entries = await fsp.readdir(sessionDir, { withFileTypes: true });
  } catch {
    return;
  }
  await Promise.all(entries.map(async (entry) => {
    if (entry.name === 'owner.json') return;
    const target = path.join(sessionDir, entry.name);
    if (entry.isDirectory()) {
      try { await fsp.chmod(target, 0o700); } catch {}
    } else if (entry.isFile()) {
      try { await fsp.chmod(target, 0o600); } catch {}
    }
    // Symlinks and other entry types are left untouched and never followed.
  }));
}

async function clearAuthFiles() {
  await fsp.mkdir(sessionDir, { recursive: true, mode: 0o700 });
  const entries = await fsp.readdir(sessionDir);
  await Promise.all(entries
    .filter((name) => name !== 'owner.json')
    .map((name) => fsp.rm(path.join(sessionDir, name), { recursive: true, force: true })));
}

// Fail closed if a late save recreated the credential file during a forget.
async function assertNoCredsFile() {
  try {
    await fsp.access(path.join(sessionDir, 'creds.json'), fs.constants.F_OK);
  } catch {
    return;
  }
  throw new Error('creds_survived_forget');
}

// --- HTTP control plane -----------------------------------------------------

const app = express();
app.disable('x-powered-by');
app.disable('etag');

app.use((req, res, next) => {
  // The QR image is a pairing bearer credential. Every response on this
  // control plane (especially /qr and /status, which embeds it) is private
  // to the authenticated process and must never be cached by a proxy.
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, private');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  res.setHeader('Vary', 'Authorization');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  const rawHost = String(req.headers.host || '');
  const host = rawHost.startsWith('[') ? rawHost.slice(0, rawHost.indexOf(']') + 1) : rawHost.split(':')[0];
  if (!HOST_ALLOWLIST.has(host.toLowerCase())) {
    res.status(400).json({ error: 'invalid_host' });
    return;
  }
  const header = String(req.headers.authorization || '');
  const match = /^Bearer ([A-Za-z0-9_-]{32,256})$/.exec(header);
  const supplied = match ? Buffer.from(match[1], 'utf8') : Buffer.alloc(0);
  // Length check avoids a timing side channel in timingSafeEqual.
  if (supplied.length !== tokenBytes.length || !crypto.timingSafeEqual(supplied, tokenBytes)) {
    res.status(401).json({ error: 'unauthorized' });
    return;
  }
  next();
});

const jsonBody = express.json({ limit: MAX_BODY_BYTES, strict: true });

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    protocol: PROTOCOL,
    actor,
    phase: state.phase,
    linked: state.linked,
    uptime_ms: Date.now() - Date.parse(state.startedAt),
  });
});

app.get('/status', (_req, res) => res.json(snapshot()));

app.get('/qr', (_req, res) => {
  if (!state.qr) {
    res.status(404).json({ error: 'no_qr' });
    return;
  }
  const expiresInMs = Math.max(0, state.qr.expires_at - Date.now());
  res.json({
    data_uri: state.qr.data_uri,
    expires_at: state.qr.expires_at,
    expires_in_ms: expiresInMs,
    rotation: state.qr.rotation,
  });
});

app.post('/wait', jsonBody, (req, res) => {
  const body = req.body && typeof req.body === 'object' ? req.body : {};
  const after = body.after;
  const timeoutMs = body.timeout_ms ?? 1000;
  if (!Number.isInteger(after) || after < 0 || after > MAX_JSON_NUMBER ||
      !Number.isInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > 30_000) {
    res.status(400).json({ error: 'invalid_request' });
    return;
  }
  if (state.sequence !== after) {
    res.json({ ...snapshot(), timed_out: false });
    return;
  }
  const waiter = { after, resolve: null, timer: null };
  const done = (result) => {
    clearTimeout(waiter.timer);
    waiters.delete(waiter);
    res.json(result);
  };
  waiter.resolve = done;
  waiters.add(waiter);
  waiter.timer = setTimeout(() => done({ ...snapshot(), timed_out: true }), timeoutMs);
  waiter.timer.unref?.();
});

app.post('/messages', jsonBody, (req, res) => {
  const body = req.body && typeof req.body === 'object' ? req.body : {};
  const after = body.after;
  const timeoutMs = body.timeout_ms ?? 1000;
  const limit = body.limit ?? MAX_MESSAGE_BATCH;
  if (!Number.isInteger(after) || after < 0 || after > MAX_JSON_NUMBER ||
      !Number.isInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > MESSAGE_WAIT_MAX_MS ||
      !Number.isInteger(limit) || limit < 1 || limit > MAX_MESSAGE_BATCH) {
    res.status(400).json({ error: 'invalid_request' });
    return;
  }
  const pending = drainMessages(after, limit);
  if (pending.length || state.stopping) {
    res.json({ messages: pending, sequence: messageSequence, timed_out: false });
    return;
  }
  const waiter = { after, limit, resolve: null, timer: null };
  const done = (result) => {
    clearTimeout(waiter.timer);
    messageWaiters.delete(waiter);
    res.json(result);
  };
  waiter.resolve = done;
  messageWaiters.add(waiter);
  waiter.timer = setTimeout(
    () => done({ messages: [], sequence: messageSequence, timed_out: true }),
    timeoutMs,
  );
  waiter.timer.unref?.();
});

app.post('/send', jsonBody, async (req, res, next) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    const to = body.to;
    const text = body.text;
    if (!isJid(to) ||
        typeof text !== 'string' || !text.trim() || text.length > MAX_MESSAGE_TEXT_CHARS) {
      res.status(400).json({ error: 'invalid_request' });
      return;
    }
    if (state.phase !== 'connected' || !state.linked || !socket) {
      res.status(409).json({ error: 'not_connected' });
      return;
    }
    const sent = await withTimeout(socket.sendMessage(to, { text }), 15_000);
    const messageId = sent?.key?.id;
    res.json({
      ok: true,
      id: typeof messageId === 'string' ? messageId.slice(0, 128) : null,
      to,
    });
  } catch (err) {
    // The generic error handler returns a code only; never echo text or JIDs.
    next(err);
  }
});

app.post('/start', jsonBody, (_req, res) => {
  if (state.stopping) {
    res.status(409).json({ error: 'stopping' });
    return;
  }
  if (state.phase === 'connected') {
    res.json(snapshot());
    return;
  }
  void openSocket(false).catch(() => setPhase('error', 'creds_drain_failed'));
  res.status(202).json(snapshot());
});

app.post('/cancel', jsonBody, async (_req, res, next) => {
  try {
    await cancelAttempt();
    setPhase('cancelled');
    res.json(snapshot());
  } catch (err) {
    next(err);
  }
});

app.post('/re-pair', jsonBody, async (req, res, next) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    if (body.forget !== undefined && typeof body.forget !== 'boolean') {
      res.status(400).json({ error: 'invalid_request' });
      return;
    }
    if (state.stopping) {
      res.status(409).json({ error: 'stopping' });
      return;
    }
    // The whole shutdown (socket + queued credential saves) must finish
    // before deletion; otherwise a delayed save resurrects the bearer creds.
    await cancelAttempt();
    if (body.forget === true) {
      await clearAuthFiles();
      await assertNoCredsFile();
    }
    await openSocket(false);
    if (body.forget === true) {
      // A late save could have recreated credentials while the fresh socket
      // was opening. Verify again so a forget can never report false success.
      try {
        await assertNoCredsFile();
      } catch (err) {
        await cancelAttempt().catch(() => {});
        throw err;
      }
    }
    res.json(snapshot());
  } catch (err) {
    next(err);
  }
});

app.use((err, _req, res, _next) => {
  // Never echo the rejected body or error message: it may contain a QR. A
  // fenced credential chain gets an explicit, non-success result instead.
  const busy = err?.bridgeErrorCode === 'creds_drain_failed';
  const status = err?.type === 'entity.too.large' ? 413 : busy ? 409 : 400;
  res.status(status).json({
    error: busy ? 'creds_drain_failed' : status === 413 ? 'payload_too_large' : 'invalid_request',
  });
});

app.use((_req, res) => res.status(404).json({ error: 'not_found' }));

// --- Process lifecycle ------------------------------------------------------

let server = null;

async function shutdown() {
  if (state.stopping && server === null) return;
  state.stopping = true;
  try {
    await cancelAttempt();
  } catch {
    // A fenced credential chain is already safe; still finish the shutdown.
  }
  for (const waiter of [...waiters]) {
    clearTimeout(waiter.timer);
    waiters.delete(waiter);
    waiter.resolve(snapshot({ timed_out: true }));
  }
  await new Promise((resolve) => {
    if (!server) return resolve();
    server.close(() => resolve());
    setTimeout(resolve, 2000).unref?.();
  });
  await credsChain.catch(() => {});
  process.exit(shutdownCode);
}

process.on('SIGTERM', () => void shutdown());
process.on('SIGINT', () => void shutdown());
process.on('uncaughtException', () => fatal('uncaught_error', ERR.generic));
process.on('unhandledRejection', () => fatal('unhandled_error', ERR.generic));

try {
  // The parent binds and KEEPS the listening socket, then passes its FD here.
  // No other local process can claim the port or intercept the owner token.
  server = app.listen({ fd: bridgeFd }, () => {
    emit('started', { port, session: actor });
  });
  server.on('error', () => fatal('bind_failed', ERR.bind));
} catch {
  fatal('bind_failed', ERR.bind);
}

// Keep the process alive for the HTTP API; Baileys starts only on POST /start.
