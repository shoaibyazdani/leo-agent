// LEO Electron shell. The backend is always localhost-only; the renderer gets
// no Node integration and no direct filesystem/credential access. The main
// process owns the API token (if any) and never sends it to the renderer.
'use strict';

const { app, BrowserWindow, dialog, Menu, ipcMain, Notification, shell, session } = require('electron');
const { spawn } = require('node:child_process');
const http = require('node:http');
const https = require('node:https');
const path = require('node:path');

const API_URL = process.env.LEO_API_URL || 'http://127.0.0.1:8765';
const API_ORIGIN = originOf(API_URL) || 'http://127.0.0.1:8765';
const ROOT = path.resolve(__dirname, '..');
const PYTHON = process.env.LEO_PYTHON || path.join(ROOT, '.venv', 'bin', 'python');
const API_TOKEN = process.env.LEO_API_TOKEN || ''; // main process only; never bridged

const EVENT_CHANNEL = 'leo:shell-event';
const CHANNELS = Object.freeze({
  HEALTH: 'leo:health',
  OPEN_SETTINGS: 'leo:open-settings',
  RELOAD: 'leo:reload',
  API_STATUS: 'leo:api-status',
  // Notification-only: shows schedule copy, never approves or denies. This is
  // not a second event transport; it is a one-way display request.
  NOTIFY: 'leo:notify',
});

const VIEWS = new Set([
  'chat', 'tasks', 'agents', 'skills', 'memory', 'models', 'tools',
  'approvals', 'integrations', 'events', 'settings',
]);

// The dashboard uses inline style attributes but no inline/eval scripts.
// Scripts, frames, objects, plugins and cross-origin connections are denied.
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self'",
  "media-src 'self' blob:",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "frame-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join('; ') + ';';

/** @typedef {'starting'|'online'|'offline'|'degraded'} ShellMode */
/**
 * @typedef {Object} HealthResult
 * @property {boolean} ok
 * @property {ShellMode} mode
 * @property {boolean} healthy
 * @property {number|null} httpStatus
 * @property {string|null} version
 * @property {string|null} status
 * @property {string|null} reason
 * @property {string|null} checkedAt
 */

let backend = null;
let backendFailure = null;
let stoppingBackend = false;
let mainWindow = null;
/** @type {HealthResult} */
let startupState = healthResult('offline', null, 'startup not completed');
const backendLog = [];

function originOf(value) {
  try {
    const parsed = new URL(String(value));
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
    return parsed.origin;
  } catch {
    return null;
  }
}

function isSameOrigin(value) {
  return originOf(value) === API_ORIGIN;
}

function safeText(value, limit = 300) {
  const text = String(value == null ? '' : value)
    .replace(/[\u0000-\u001f\u007f]/g, ' ')
    .trim();
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

function apiPort() {
  try { return Number(new URL(API_URL).port) || 80; } catch { return 8765; }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** @returns {HealthResult} */
function healthResult(mode, httpStatus = null, reason = null, extra = {}) {
  return {
    ok: mode === 'online',
    mode,
    healthy: mode === 'online',
    httpStatus,
    version: typeof extra.version === 'string' ? extra.version : null,
    status: typeof extra.status === 'string' ? extra.status : null,
    reason: reason ? safeText(reason) : null,
    checkedAt: new Date().toISOString(),
  };
}

function classifyHealth(statusCode, body) {
  if (statusCode !== 200) {
    return healthResult('degraded', statusCode, `health endpoint returned HTTP ${statusCode}`);
  }
  let payload;
  try { payload = JSON.parse(body); } catch {
    return healthResult('degraded', statusCode, 'health endpoint returned invalid JSON');
  }
  if (!payload || typeof payload !== 'object') {
    return healthResult('degraded', statusCode, 'health endpoint returned an empty response');
  }
  const status = typeof payload.status === 'string' ? payload.status : '';
  const version = typeof payload.version === 'string' ? payload.version : null;
  if (status === 'ok') return healthResult('online', statusCode, null, { status, version });
  return healthResult('degraded', statusCode, `backend reported status ${status || 'unknown'}`, { status, version });
}

// Raw response bodies and error messages stay here: only whitelisted fields
// are returned to callers and forwarded to the renderer.
function probeHealth(timeoutMs = 1500) {
  return new Promise((resolve) => {
    let target;
    try { target = new URL('/health', API_ORIGIN); } catch {
      resolve(healthResult('offline', null, 'invalid API URL'));
      return;
    }
    const transport = target.protocol === 'https:' ? https : http;
    const headers = {};
    if (API_TOKEN) headers.Authorization = `Bearer ${API_TOKEN}`;
    const request = transport.get(target, { headers }, (response) => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => { if (body.length < 4096) body += chunk; });
      response.on('error', () => resolve(healthResult('offline', null, 'health response failed')));
      response.on('end', () => resolve(classifyHealth(response.statusCode, body)));
    });
    request.setTimeout(timeoutMs, () => request.destroy(new Error('health check timed out')));
    request.on('error', (err) => resolve(healthResult('offline', null, err && err.message ? err.message : err)));
  });
}

async function waitForHealth(attempts = 60, intervalMs = 250) {
  let last = healthResult('offline', null, 'health check did not run');
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (backendFailure) return healthResult('offline', null, backendFailure);
    last = await probeHealth();
    if (last.mode !== 'offline') return last; // online or degraded: an explicit answer
    await delay(intervalMs);
  }
  return last;
}

async function refreshHealth() {
  const health = await probeHealth();
  setStartupState(health);
  return health;
}

function appendBackendLog(stream, chunk) {
  // Untrusted backend output is memory-only and capped; never sent as-is.
  backendLog.push(`${stream} ${new Date().toISOString()} ${safeText(chunk, 1000)}`);
  if (backendLog.length > 100) backendLog.shift();
}

function startBackend() {
  if (process.env.LEO_EXTERNAL_API === '1') return;
  backendFailure = null;
  stoppingBackend = false;
  backend = spawn(PYTHON, ['-m', 'leo.interfaces.cli', 'serve', '--host', '127.0.0.1', '--port', String(apiPort())], {
    cwd: ROOT,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (backend.stdout) backend.stdout.on('data', (chunk) => appendBackendLog('stdout', chunk));
  if (backend.stderr) backend.stderr.on('data', (chunk) => appendBackendLog('stderr', chunk));
  backend.on('error', (err) => {
    backendFailure = safeText(err && err.message ? err.message : err);
    backend = null;
    notifyRenderer('backend-error', { message: 'Backend failed to start' });
  });
  backend.on('exit', (code, signal) => {
    backend = null;
    if (stoppingBackend) return;
    const reason = `Backend exited (code ${code === null ? 'none' : code}, signal ${signal || 'none'})`;
    backendFailure = reason;
    notifyRenderer('backend-exit', { message: reason });
    setStartupState(healthResult('offline', null, reason));
    dialog.showErrorBox('LEO backend stopped', reason);
  });
}

function stopBackend() {
  if (!backend) return Promise.resolve();
  stoppingBackend = true;
  const child = backend;
  backend = null;
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      if (!child.killed) child.kill('SIGKILL');
      resolve();
    }, 2000);
    child.once('exit', () => { clearTimeout(timer); resolve(); });
    child.kill('SIGTERM');
  });
}

function notifyRenderer(type, payload = {}) {
  if (!mainWindow || mainWindow.isDestroyed() || mainWindow.webContents.isDestroyed()) return;
  // Allowlist of fields only: no raw backend output, headers or credentials.
  mainWindow.webContents.send(EVENT_CHANNEL, {
    type: safeText(type, 60),
    at: new Date().toISOString(),
    message: payload.message ? safeText(payload.message) : null,
    mode: payload.mode ? safeText(payload.mode, 20) : null,
    healthy: typeof payload.healthy === 'boolean' ? payload.healthy : null,
    version: payload.version ? safeText(payload.version, 60) : null,
  });
}

function setStartupState(next) {
  startupState = next;
  notifyRenderer('startup-state', {
    mode: next.mode, healthy: next.healthy, version: next.version, reason: next.reason,
  });
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.setTitle(`LEO — ${next.mode}`);
}

function offlinePage(state) {
  const degraded = state.mode === 'degraded';
  const title = degraded ? 'LEO is degraded' : 'LEO is offline';
  const detail = state.reason || 'The local API did not pass its health check.';
  const html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>' +
    title + '</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;' +
    'background:#101317;color:#e7e9ea;font:15px -apple-system,sans-serif}' +
    'main{max-width:520px;padding:32px}h1{font-size:22px}p{line-height:1.6;color:#a8b0b8}' +
    '</style></head><body><main><h1>' + title + '</h1><p>' + escapeHtml(safeText(detail, 400)) +
    '</p><p>Use the LEO menu (Reload Backend) to retry. No remote service is contacted.</p>' +
    '</main></body></html>';
  return `data:text/html;charset=utf-8,${encodeURIComponent(html)}`;
}

function loadShellPage(health) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  if (health.mode === 'online') mainWindow.loadURL(API_URL);
  else mainWindow.loadURL(offlinePage(health));
}

async function reloadShell() {
  await stopBackend();
  startBackend();
  const health = await waitForHealth();
  startupState = health;
  loadShellPage(health);
  setStartupState(health);
  return {
    ok: health.mode === 'online',
    mode: health.mode,
    healthy: health.healthy,
    reason: health.reason,
    checkedAt: health.checkedAt,
  };
}

// The view allowlist is validated before interpolation into the renderer script.
async function showView(view) {
  if (!VIEWS.has(view)) return { ok: false, reason: 'unknown-view' };
  if (!mainWindow || mainWindow.isDestroyed() || startupState.mode !== 'online') {
    return { ok: false, reason: 'dashboard-unavailable' };
  }
  const script = `(() => {
    const button = document.querySelector('[data-view="${JSON.stringify(view)}"]');
    if (!button) return false;
    button.click();
    return true;
  })()`;
  try {
    const clicked = await mainWindow.webContents.executeJavaScript(script);
    return { ok: Boolean(clicked), view };
  } catch {
    return { ok: false, reason: 'view-activation-failed' };
  }
}

function openSettings() {
  return showView('settings');
}

// Only presence/source is reported; the token value never leaves this process.
function apiStatus() {
  return {
    ok: true,
    apiPort: apiPort(),
    apiOrigin: API_ORIGIN,
    external: process.env.LEO_EXTERNAL_API === '1',
    tokenConfigured: Boolean(API_TOKEN),
    tokenSource: API_TOKEN ? 'environment' : 'none',
    shellVersion: app.getVersion(),
  };
}

function openExternal(value) {
  const parsed = originOf(value) ? new URL(value) : null;
  if (!parsed) return; // http(s) only: no file:, javascript: or data: URLs
  try { void shell.openExternal(parsed.href); } catch { /* ignored */ }
}

function installNavigationGuards(win) {
  const guard = (event, url) => {
    if (isSameOrigin(url)) return;
    event.preventDefault();
    openExternal(url);
  };
  win.webContents.on('will-navigate', guard);
  win.webContents.on('will-redirect', guard);
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (!isSameOrigin(url)) openExternal(url);
    return { action: 'deny' }; // no privileged popup windows
  });
}

function installContentSecurityPolicy() {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    const responseHeaders = { ...(details.responseHeaders || {}) };
    if (originOf(details.url) === API_ORIGIN) {
      for (const key of Object.keys(responseHeaders)) {
        const lower = key.toLowerCase();
        if (lower === 'content-security-policy' || lower === 'content-security-policy-report-only') {
          delete responseHeaders[key];
        }
      }
      responseHeaders['Content-Security-Policy'] = [CSP];
    }
    callback({ responseHeaders });
  });
}

function installAuthHeaders() {
  if (!API_TOKEN) return;
  session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
    const requestHeaders = { ...details.requestHeaders };
    if (originOf(details.url) === API_ORIGIN) {
      requestHeaders.Authorization = `Bearer ${API_TOKEN}`;
    }
    callback({ requestHeaders });
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      devTools: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  installNavigationGuards(mainWindow);
  mainWindow.webContents.on('crashed', () => {
    // The renderer is gone; the dialog is the fallback for this event.
    dialog.showErrorBox('LEO renderer crashed', 'Reload LEO from the menu to continue.');
  });
  mainWindow.on('closed', () => { mainWindow = null; });
}

function menuNavigate(view) {
  void showView(view).then((result) => {
    if (!result.ok) notifyRenderer('action-failed', { message: `Could not open ${view}` });
  });
}

function menuSettings() {
  void openSettings().then((result) => {
    if (!result.ok) notifyRenderer('action-failed', { message: 'Could not open settings' });
  });
}

function buildApplicationMenu() {
  const template = [];
  if (process.platform === 'darwin') {
    template.push({
      label: app.name || 'LEO',
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { label: 'Quit LEO', accelerator: 'CmdOrCtrl+Q', click: () => app.quit() },
      ],
    });
  }
  template.push({
    label: 'LEO',
    submenu: [
      { label: 'Open Chat', accelerator: 'CmdOrCtrl+1', click: () => menuNavigate('chat') },
      { label: 'Open Tasks', accelerator: 'CmdOrCtrl+2', click: () => menuNavigate('tasks') },
      { label: 'Open Agents', accelerator: 'CmdOrCtrl+3', click: () => menuNavigate('agents') },
      { label: 'Open Integrations', accelerator: 'CmdOrCtrl+4', click: () => menuNavigate('integrations') },
      { type: 'separator' },
      { label: 'Settings…', accelerator: 'CmdOrCtrl+,', click: () => menuSettings() },
      { label: 'Check Health', accelerator: 'CmdOrCtrl+Shift+H', click: () => { void refreshHealth(); } },
      { label: 'Reload Backend', accelerator: 'CmdOrCtrl+R', click: () => { void reloadShell(); } },
      { type: 'separator' },
      { role: 'quit', accelerator: 'CmdOrCtrl+Q' },
    ],
  });
  template.push({ label: 'Window', submenu: [{ role: 'minimize' }, { role: 'close' }] });
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// Typed invoke bridge. Handlers take no renderer-controlled arguments and
// return only whitelisted, serialisable fields.
ipcMain.handle(CHANNELS.HEALTH, async () => refreshHealth());
ipcMain.handle(CHANNELS.OPEN_SETTINGS, async () => openSettings());
ipcMain.handle(CHANNELS.RELOAD, async () => reloadShell());
ipcMain.handle(CHANNELS.API_STATUS, async () => apiStatus());
// Notification-only bridge: bounded text, no buttons, no approval channel.
ipcMain.handle(CHANNELS.NOTIFY, async (_event, payload = {}) => {
  const title = safeText(payload && payload.title ? payload.title : 'LEO', 120);
  const body = safeText(payload && payload.body ? payload.body : '', 1200);
  if (!Notification.isSupported()) return { shown: false, reason: 'unsupported' };
  const notification = new Notification({ title: title || 'LEO', body });
  // Clicking only focuses the in-app approvals view. There is no button, no
  // decision payload and no approval authority in this IPC channel.
  notification.on('click', () => { void showView('approvals'); if (mainWindow && !mainWindow.isDestroyed()) mainWindow.show(); });
  notification.show();
  return { shown: true };
});

app.whenReady().then(async () => {
  installContentSecurityPolicy();
  installAuthHeaders();
  buildApplicationMenu();
  startBackend();
  const health = await waitForHealth();
  startupState = health;
  createWindow();
  loadShellPage(health);
  setStartupState(health);
  if (health.mode !== 'online') {
    dialog.showMessageBox({
      type: 'warning',
      title: health.mode === 'degraded' ? 'LEO is degraded' : 'LEO is offline',
      message: health.reason || 'The local backend did not pass its health check.',
      buttons: ['OK'],
    }).catch(() => {});
  }
}).catch((err) => {
  dialog.showErrorBox('LEO failed to start', safeText(err && err.message ? err.message : err));
  app.quit();
});

app.on('render-process-gone', () => {
  dialog.showErrorBox('LEO renderer stopped', 'Reload LEO from the menu to continue.');
});

app.on('window-all-closed', () => {
  void stopBackend().then(() => app.quit());
});

app.on('before-quit', () => {
  stoppingBackend = true; // suppress crash dialog for an intentional shutdown
  if (backend && !backend.killed) backend.kill('SIGTERM');
});
