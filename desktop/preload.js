// Narrow, typed preload bridge: four invoke channels and one outbound event
// channel. No credentials, storage, filesystem or arbitrary IPC access is
// exposed to the renderer. The API token stays in renderer session storage
// (or in the main process for request headers); its value is never bridged.
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

const CHANNELS = Object.freeze({
  HEALTH: 'leo:health',
  OPEN_SETTINGS: 'leo:open-settings',
  RELOAD: 'leo:reload',
  API_STATUS: 'leo:api-status',
  // Notification-only. Deliberately excluded from the generic no-argument
  // invoke allowlist; notify() passes bounded text and nothing else.
  NOTIFY: 'leo:notify',
});
const EVENT_CHANNEL = 'leo:shell-event';
const ALLOWED = Object.freeze(
  Object.values(CHANNELS).filter((channel) => channel !== CHANNELS.NOTIFY),
);

contextBridge.exposeInMainWorld('leo', Object.freeze({
  platform: process.platform,
  // Live version/status come from invoke('health') and invoke('api-status');
  // no hard-coded version is published to the renderer.
  invoke(channel) {
    if (!ALLOWED.includes(channel)) {
      return Promise.reject(new Error(`Unknown LEO channel: ${String(channel)}`));
    }
    return ipcRenderer.invoke(channel); // no arguments: no renderer-supplied data
  },
  onEvent(listener) {
    if (typeof listener !== 'function') return () => {};
    const wrapped = (_event, payload) => { listener(payload); };
    ipcRenderer.on(EVENT_CHANNEL, wrapped);
    return () => { ipcRenderer.removeListener(EVENT_CHANNEL, wrapped); };
  },
  // Notification-only: shows schedule copy. There is no approve/deny here and
  // no second event transport; a notification grants nothing.
  notify(title, body) {
    return ipcRenderer.invoke(CHANNELS.NOTIFY, {
      title: String(title == null ? 'LEO' : title).slice(0, 120),
      body: String(body == null ? '' : body).slice(0, 1200),
    });
  },
}));
