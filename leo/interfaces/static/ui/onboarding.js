/* LEO guided channel setup — runtime-injected wizard.
   Loaded by app.js so the core dashboard keeps its raw modal if this module is
   unavailable. All markup is injected into the existing Integrations view;
   index.html is not modified. The local QR is a one-time setup code rendered
   from a payload supplied by this host. It is NEVER described as a WhatsApp
   device link: LEO has no WhatsApp Web/Baileys bridge. */

(() => {
  'use strict';
  if (window.LEO_ONBOARDING) return;

  const FEATURED = ['whatsapp', 'telegram', 'discord', 'slack'];
  const STEPS = ['Method', 'Details', 'Access', 'Check'];
  const POLICY_FIELDS = new Set(['allowlist', 'mode']);
  const POLL_MS = 3000;
  const POLL_LIMIT = 100;

  const state = {
    root: null, dialog: null, channel: '', opener: null, step: 0,
    method: 'cloud', details: null, validation: null,
    form: { values: {}, allowlist: [], mode: '' }, errors: {},
    qr: null, qrError: '', pollTimer: null, clockTimer: null, busy: false,
  };

  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const ico = (name, cls = 'icon') => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const safeList = (v) => Array.isArray(v) ? v.map((x) => String(x)) : [];

  // Known mode ids always get human labels, even when a catalog supplies bare
  // strings or an object shape this build has not seen before.
  const MODE_LABELS = {
    allowlist: 'Only people on the list',
    open: 'Anyone',
    deny: 'Nobody',
  };
  const CHOICE_KINDS = new Set(['choice', 'select', 'enum', 'dropdown']);
  const TEXTAREA_KINDS = new Set(['textarea', 'multiline', 'list']);

  // Built-in, value-free guidance used whenever the backend guide is missing
  // or empty. Every entry has real words: the wizard never renders an empty
  // list item, whatever the catalog returns.
  const BUILTIN_GUIDES = {
    telegram: [
      { title: '1. Create a bot', detail: 'Message @BotFather in Telegram, create a bot and copy the bot token it gives you.' },
      { title: '2. Fill in the details', detail: 'Paste the credentials on the next screen, then choose who may message LEO. Start with your own ID.' },
      { title: '3. Check, then enable', detail: 'Check my settings validates fields offline. It never sends a test message.' },
    ],
    whatsapp: [
      { title: '1. Create a Meta app', detail: 'Add the WhatsApp product in Meta for Developers and open the Cloud API setup page.' },
      { title: '2. Copy three values', detail: 'Get the access token, Phone number ID and App secret. The Phone number ID is not your phone number.' },
      { title: '3. Check, then enable', detail: 'Point the webhook at LEO, run the offline check and enable. No test message is sent.' },
    ],
    default: [
      { title: '1. Get the credentials', detail: 'Find the values in your provider dashboard.' },
      { title: '2. Choose who may message LEO', detail: 'Start with your own ID. An empty list blocks everyone.' },
      { title: '3. Check, then enable', detail: 'The check is offline only; it never sends a test message.' },
    ],
  };
  const CARD_GUIDES = {
    whatsapp: [
      { title: '', detail: 'A Meta app with the WhatsApp product added.' },
      { title: '', detail: 'A Cloud API access token and app secret from Meta.' },
      { title: '', detail: 'A Phone number ID — a Meta ID, NOT your WhatsApp phone number.' },
      { title: '', detail: 'A public HTTPS webhook pointing at LEO, and your own number in the allowed senders list.' },
    ],
    default: [
      { title: '', detail: 'Get the credentials from your provider dashboard.' },
      { title: '', detail: 'Use your own ID first; an empty allowed-senders list blocks everyone.' },
      { title: '', detail: 'Check the fields, then enable. No live message is sent by the check.' },
    ],
  };

  // Only primitives are text. Coercing an object with String() is what rendered
  // "[object Object]" in the old select and blank labels in the old form.
  function firstText(source, keys) {
    if (!source || typeof source !== 'object') return '';
    for (const key of keys) {
      const value = source[key];
      if (value === null || value === undefined) continue;
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        const text = String(value).trim();
        if (text && text !== '[object Object]') return text;
      }
    }
    return '';
  }
  function humanize(value) {
    const text = String(value ?? '').trim();
    if (!text || text === '[object Object]') return '';
    return text.replace(/[_-]+/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/\s+/g, ' ').replace(/^./, (ch) => ch.toUpperCase());
  }
  function asObjects(value) {
    if (!Array.isArray(value)) return [];
    return value.map((item) => typeof item === 'string' ? { name: item }
      : (item && typeof item === 'object' ? item : null)).filter(Boolean);
  }

  // Accept guide.steps, guide.guide (object or array), a bare step array and
  // the catalog aliases. Strings, {title, detail}, {text}, {label} and
  // {step, text} all become renderable step text.
  function guideArray(source, depth = 0) {
    if (Array.isArray(source)) return source;
    if (source && typeof source === 'object' && depth < 3) {
      for (const key of ['steps', 'guide', 'items', 'list']) {
        const value = source[key];
        if (Array.isArray(value)) return value;
        if (value && typeof value === 'object') {
          const nested = guideArray(value, depth + 1);
          if (nested.length) return nested;
        }
      }
    }
    return [];
  }
  function normalizeGuideSteps(source, channel = state.channel) {
    const steps = [];
    guideArray(source).forEach((item) => {
      let title = '';
      let detail = '';
      if (typeof item === 'string') detail = item.trim();
      else if (item && typeof item === 'object') {
        title = firstText(item, ['title', 'label', 'name', 'heading']);
        detail = firstText(item, ['text', 'detail', 'description', 'message', 'body', 'content']);
        if (!title && !detail) {
          const step = firstText(item, ['step', 'number']);
          if (step) title = /^\d+$/.test(step) ? `Step ${step}` : step;
        }
      }
      title = String(title).trim();
      detail = String(detail).trim();
      if (!title && !detail) return; // never emit an empty <li>
      if (title && detail === title) detail = '';
      steps.push({ title, detail, docsUrl: firstText(item, ['docs_url', 'url', 'href']) });
    });
    if (steps.length) return steps;
    const key = String(channel || '').toLowerCase();
    return BUILTIN_GUIDES[key] || BUILTIN_GUIDES.default;
  }

  // Steps are written with textContent, never interpolated HTML, so guide
  // text cannot introduce markup or collapse into empty list items.
  function renderGuideItems(host, steps) {
    if (!host) return;
    host.textContent = '';
    steps.forEach((item) => {
      const title = String(item?.title || '').trim();
      const detail = String(item?.detail || (typeof item === 'string' ? item : '')).trim();
      if (!title && !detail) return;
      const li = document.createElement('li');
      if (title) {
        const strong = document.createElement('strong');
        strong.textContent = title;
        li.appendChild(strong);
      }
      if (detail) {
        const span = document.createElement('span');
        span.textContent = `${title ? ' ' : ''}${detail}`;
        li.appendChild(span);
      }
      const url = firstText(item, ['docsUrl', 'docs_url', 'url', 'href']);
      if (/^https:\/\//i.test(url)) {
        const link = document.createElement('a');
        link.className = 'ob-link';
        link.href = url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = 'Open the guide';
        li.appendChild(link);
      }
      if (li.textContent.trim()) host.appendChild(li);
    });
  }
  function guideItems(source, channel) {
    const key = String(source || '');
    if (key === 'card') return channel === 'whatsapp' ? CARD_GUIDES.whatsapp : CARD_GUIDES.default;
    if (key === 'notes') return asObjects(state.details?.guide?.notes)
      .map((note) => ({ title: '', detail: firstText(note, ['name', 'text', 'detail', 'title', 'label', 'message']) }))
      .filter((note) => note.detail);
    // Pass the whole details payload: guideArray() finds nested guide.steps,
    // guide.guide or a top-level catalog step array.
    if (key === 'tail') return normalizeGuideSteps(state.details, channel).slice(1);
    if (key === 'details') return normalizeGuideSteps(state.details, channel);
    return normalizeGuideSteps(null, channel);
  }
  function hydrateGuideLists(root) {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll('[data-ob-guide-steps]').forEach((list) => {
      const items = guideItems(list.dataset.obGuideSteps, list.dataset.obGuideChannel || state.channel);
      if (!items.length) { list.closest('.ob-guide')?.remove(); return; }
      renderGuideItems(list, items);
    });
  }

  function toast(message, kind = 'info') {
    if (window.LEO && typeof window.LEO.toast === 'function') window.LEO.toast(message, kind);
  }
  function refresh() {
    if (window.LEO && typeof window.LEO.refresh === 'function') window.LEO.refresh();
  }

  // ---- transport ---------------------------------------------------------
  function bearer() {
    try { return sessionStorage.getItem('leo_api_token') || ''; } catch { return ''; }
  }

  async function request(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const token = bearer();
    if (token && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
    if (options.body && !headers.has('content-type')) headers.set('content-type', 'application/json');
    const response = await fetch(path, { ...options, headers });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body && body.detail;
      const message = typeof detail === 'string' ? detail
        : detail && (detail.message || (Array.isArray(detail.errors) && detail.errors[0]?.message))
        || body.message || `${response.status} ${response.statusText}`;
      const error = new Error(message);
      error.status = response.status;
      error.body = body;
      throw error;
    }
    return body;
  }

  // Onboarding writes carry a custom header (CSRF) and only ever use a URL on
  // this host. Never place a token or setup code in a log or an event payload.
  const WRITE_HEADERS = { 'X-Leo-Onboarding': '1' };

  // ---- data --------------------------------------------------------------
  function integrations() {
    const data = window.LEO && window.LEO.data;
    return Array.isArray(data?.integrations?.integrations) ? data.integrations.integrations : [];
  }
  function rowFor(channel) {
    return integrations().find((row) => String(row.name) === channel) || null;
  }
  function labelFor(channel) {
    return state.details?.label || channel;
  }
  // Cards must not reuse a wizard's label: state.details belongs to whichever
  // channel was opened last, not to every row on the page.
  function cardLabel(channel) {
    return rowFor(channel)?.label || channel;
  }
  function statusFor(channel, row) {
    if (row?.status === 'configured') {
      return { tone: 'ok', icon: 'check', label: 'Credentials saved', detail: 'Offline check only — live delivery is not tested.' };
    }
    if (row?.status === 'error') {
      return { tone: 'blocked', icon: 'alert', label: 'Needs attention', detail: 'Open the setup steps to review what is missing.' };
    }
    if (channel === 'whatsapp') {
      return { tone: 'blocked', icon: 'alert', label: 'Needs Cloud API credentials', detail: 'Personal QR device linking is not available in LEO.' };
    }
    return { tone: 'idle', icon: 'clock', label: 'Not connected', detail: 'A few details are needed before LEO can use this channel.' };
  }

  function fallbackDetails(channel) {
    return {
      channel, label: channel, status: 'blocked', fields: [], docs_url: '',
      guide: {
        steps: [
          { title: '1. Get the credentials', detail: 'Find the values in your provider dashboard.' },
          { title: '2. Choose who may message LEO', detail: 'Start with your own ID. An empty list blocks everyone.' },
          { title: '3. Check, then enable', detail: 'The check is offline only; it never sends a test message.' },
        ],
        notes: [], docs_url: '',
      },
      device_link: { status: 'blocked', reason: 'Device linking is not available in LEO.' },
      qr_setup: { available: false, note: 'No guided QR setup is available in this runtime.' },
      persistence_note: null, mode_note: null,
    };
  }

  async function loadDetails(channel) {
    try { return await request(`/onboarding/${encodeURIComponent(channel)}/guide`); }
    catch {
      try {
        const catalog = await request(`/integrations/catalog/${encodeURIComponent(channel)}`);
        return {
          ...catalog,
          guide: catalog.guide || fallbackDetails(channel).guide,
          qr_setup: catalog.qr_setup || { available: false, note: 'No guided QR setup is available in this runtime.' },
        };
      } catch { return fallbackDetails(channel); }
    }
  }

  // ---- card surface ------------------------------------------------------
  function guideHTML(source, channel = state.channel) {
    return `<details class="ob-guide"><summary>${ico('chevron')} What do I need?</summary><ol class="ob-guide__list" data-ob-guide-steps="${esc(source)}" data-ob-guide-channel="${esc(channel)}"></ol></details>`;
  }

  function cardHTML(channel) {
    const row = rowFor(channel);
    const status = statusFor(channel, row);
    const whatsapp = channel === 'whatsapp';
    const title = whatsapp ? 'WhatsApp' : cardLabel(channel);
    const subtitle = whatsapp ? 'Meta Cloud API · QR device linking unavailable' : 'Messaging channel';
    return `<article class="ob-card${whatsapp ? ' ob-card--featured' : ''}">
      <div class="ob-card__head"><span class="ob-card__icon">${ico(whatsapp ? 'link' : 'integrations')}</span>
        <div><h3 class="ob-card__title">${esc(title)}</h3><p class="ob-card__subtitle">${esc(subtitle)}</p></div>
      </div>
      <div class="ob-card__status ob-status--${status.tone}">${ico(status.icon)}<span>${esc(status.label)}<small>${esc(status.detail)}</small></span></div>
      ${guideHTML('card', channel)}
      <div class="ob-card__foot"><button class="ob-btn ob-btn--primary" type="button" data-ob-configure="${esc(channel)}">${whatsapp ? 'Set up WhatsApp' : 'Set up channel'}</button></div>
    </article>`;
  }

  function renderCards() {
    const root = state.root;
    if (!root || !root.isConnected) return;
    const rows = integrations().filter((row) => row.kind === 'messaging' && !String(row.name).startsWith('channel-module:'));
    const names = new Set([...FEATURED, ...rows.map((row) => String(row.name))]);
    const ordered = FEATURED.filter((name) => names.has(name))
      .concat([...names].filter((name) => !FEATURED.includes(name)).sort());
    const cards = ordered.slice(0, 6).map(cardHTML).join('');
    const rest = ordered.slice(6).map((name) => `<button class="ob-btn" type="button" data-ob-configure="${esc(name)}">${esc(name)}</button>`).join('');
    root.innerHTML = `<div class="ob-heading"><div><h2>Connect a channel</h2>
        <p>Choose a channel and follow the steps. LEO saves credentials locally and does not send a test message.</p>
      </div></div>
      <div class="ob-cards">${cards}</div>
      ${rest ? `<details class="ob-more"><summary>More channels (${ordered.length - 6})</summary><div class="ob-more__list">${rest}</div></details>` : ''}`;
    hydrateGuideLists(root);
  }

  function injectStyles() {
    if (document.querySelector('link[data-ob-styles]')) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/assets/ui/onboarding.css';
    link.dataset.obStyles = '1';
    document.head.appendChild(link);
  }

  function mount() {
    injectStyles();
    const host = document.getElementById('view-integrations');
    if (!host) return;
    let root = document.getElementById('ob-root');
    if (!root) {
      root = document.createElement('section');
      root.id = 'ob-root';
      root.className = 'ob-root';
      root.setAttribute('aria-label', 'Guided channel setup');
      host.insertBefore(root, host.querySelector('.panel'));
    }
    state.root = root;
    if (!host.dataset.obBound) {
      host.dataset.obBound = '1';
      host.addEventListener('click', onHostClick);
      window.addEventListener('leo:data', renderCards);
    }
    renderCards();
  }

  function onHostClick(event) {
    const button = event.target.closest('[data-ob-configure], [data-configure]');
    if (!button || !hostContains(button)) return;
    const channel = button.dataset.obConfigure || button.dataset.configure;
    if (!channel) return;
    // The core app also delegates [data-configure] at document level. This
    // capture at the host replaces its raw form with the guided wizard.
    event.preventDefault();
    event.stopPropagation();
    openWizard(channel, button);
  }
  function hostContains(element) {
    const host = document.getElementById('view-integrations');
    return !!host && host.contains(element);
  }

  // ---- wizard shell ------------------------------------------------------
  function openWizard(channel, opener) {
    closeDialog();
    state.channel = channel;
    state.opener = opener || document.activeElement;
    state.step = 0;
    state.method = 'cloud';
    state.details = fallbackDetails(channel);
    state.validation = null;
    state.form = { values: {}, allowlist: [], mode: '' };
    state.errors = {};
    state.qr = null;
    state.qrError = '';

    const backdrop = document.createElement('div');
    backdrop.className = 'ob-modal-backdrop';
    backdrop.id = 'ob-modal-backdrop-onboarding';
    backdrop.innerHTML = `<div class="ob-modal" role="dialog" aria-modal="true" aria-labelledby="ob-dialog-title">
      <div class="ob-modal__head"><div>
        <h3 class="ob-modal__title" id="ob-dialog-title">${esc(channel === 'whatsapp' ? 'Set up WhatsApp' : `Set up ${channel}`)}</h3>
        <p class="ob-modal__subtitle" id="ob-dialog-subtitle">Short steps, plain language, no credentials shown back to you.</p>
      </div><button class="ob-btn ob-btn--quiet" type="button" data-ob-action="close" aria-label="Close setup">${ico('x')}</button></div>
      <div class="ob-modal__body" id="ob-dialog-body"></div>
      <div class="ob-modal__foot"><span class="ob-note" id="ob-foot-note">Offline check only · nothing is sent</span>
        <span class="ob-modal__foot-actions"><button class="ob-btn" type="button" data-ob-action="back" id="ob-back">Back</button>
        <button class="ob-btn ob-btn--primary" type="button" data-ob-action="next" id="ob-next">Continue</button></span>
      </div>
    </div>`;
    document.body.appendChild(backdrop);
    state.dialog = backdrop;
    backdrop.addEventListener('click', onDialogClick);
    backdrop.addEventListener('keydown', trapFocus);
    backdrop.querySelector('[data-ob-action="close"]').focus();
    renderDialogBody();

    loadDetails(channel).then((details) => {
      if (!state.dialog || state.channel !== channel) return;
      captureFields();
      state.details = details;
      document.getElementById('ob-dialog-title').textContent = `Set up ${details.label || channel}`;
      renderDialogBody();
    });
  }

  function closeDialog() {
    stopPolling();
    if (state.clockTimer) { clearInterval(state.clockTimer); state.clockTimer = null; }
    if (state.dialog) state.dialog.remove();
    state.dialog = null;
    if (state.opener && state.opener.isConnected) state.opener.focus();
    state.opener = null;
  }

  function trapFocus(event) {
    if (event.key === 'Escape') { event.preventDefault(); closeDialog(); return; }
    if (event.key !== 'Tab' || !state.dialog) return;
    const focusable = [...state.dialog.querySelectorAll('button:not([disabled]), input:not([disabled]), a[href], summary')]
      .filter((el) => el.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }

  function captureFields() {
    if (!state.dialog) return;
    state.dialog.querySelectorAll('[data-ob-field]').forEach((el) => {
      const name = el.dataset.obField;
      const raw = String(el.value || '').trim();
      if (name === 'allowlist') state.form.allowlist = raw.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
      else if (name === 'mode') { if (raw) state.form.mode = raw; }
      else if (raw) state.form.values[name] = raw;
    });
  }

  function restoreFields() {
    if (!state.dialog) return;
    state.dialog.querySelectorAll('[data-ob-field]').forEach((el) => {
      const name = el.dataset.obField;
      if (name === 'allowlist') el.value = state.form.allowlist.join(', ');
      else if (name === 'mode') el.value = state.form.mode || el.value;
      else if (state.form.values[name]) el.value = state.form.values[name];
    });
    showFieldErrors();
  }

  function showFieldErrors() {
    if (!state.dialog) return;
    state.dialog.querySelectorAll('[data-ob-field]').forEach((el) => {
      const message = state.errors[el.dataset.obField];
      const holder = el.closest('.ob-field')?.querySelector('.ob-field__error');
      if (!holder) return;
      holder.hidden = !message;
      holder.querySelector('span').textContent = message || '';
      el.setAttribute('aria-invalid', message ? 'true' : 'false');
    });
  }

  // ---- step content ------------------------------------------------------
  function renderDialogBody() {
    const body = state.dialog?.querySelector('#ob-dialog-body');
    if (!body) return;
    body.innerHTML = `<ol class="ob-progress" aria-label="Setup progress">${STEPS.map((label, i) =>
      `<li class="${i === state.step ? 'is-current' : i < state.step ? 'is-done' : ''}" data-step="${i + 1}">${esc(label)}</li>`).join('')}</ol>`
      + stepContent();
    hydrateGuideLists(body);
    restoreFields();
    const back = state.dialog.querySelector('#ob-back');
    const next = state.dialog.querySelector('#ob-next');
    back.disabled = state.step === 0;
    next.textContent = state.step === STEPS.length - 1 ? `Enable ${labelFor(state.channel)}` : 'Continue';
    const note = state.dialog.querySelector('#ob-foot-note');
    if (note) note.textContent = state.details?.persistence_note
      ? 'Credentials stay in memory only until LEO_CREDENTIALS_KEY is set.'
      : 'Offline check only · nothing is sent';
  }

  // Every step is present in the DOM but inactive ones are hidden with both
  // `hidden` and `inert`. A transparent-but-present step was showing through
  // behind the admission select in older builds.
  function stepContent() {
    return [stepMethod(), stepFields(), stepAccess(), stepCheck()].map((html, index) =>
      `<div class="ob-step" data-ob-step="${index}"${index === state.step ? '' : ' hidden inert'}>${html}</div>`).join('');
  }

  function stepMethod() {
    if (state.channel !== 'whatsapp') {
      const guide = state.details?.guide || {};
      const docs = /^https:\/\//i.test(String(state.details?.docs_url || guide.docs_url || ''))
        ? (state.details.docs_url || guide.docs_url) : '';
      return `<section class="ob-section"><h4 class="ob-section__title">How to connect ${esc(labelFor(state.channel))}</h4>
        <ol class="ob-list" aria-label="Setup steps" data-ob-guide-steps="details" data-ob-guide-channel="${esc(state.channel)}"></ol>
        ${docs ? `<a class="ob-link" href="${esc(docs)}" target="_blank" rel="noopener noreferrer">Open the official setup guide</a>` : ''}
        ${guideHTML('notes', state.channel)}
        ${state.details?.device_link?.status === 'blocked' ? `<div class="ob-callout ob-callout--warn">${ico('alert')}<span>Device linking is not available in this channel.</span></div>` : ''}
      </section>`;
    }
    return `<section class="ob-section"><h4 class="ob-section__title">Choose a connection method</h4>
      <p class="ob-section__text">Pick how you want to connect. LEO is honest about what each option can do.</p>
      <div class="ob-methods">
        <button class="ob-method" type="button" data-ob-action="method" data-method="cloud" aria-pressed="${state.method === 'cloud'}">
          <span class="ob-method__title">${ico('check')} Meta Cloud API <span class="ob-saved">Supported</span></span>
          <span class="ob-method__text">Official Meta setup. Needs a Meta app, a business phone number and a public HTTPS webhook. It does not link a personal device.</span>
        </button>
        <div class="ob-method" role="note" aria-disabled="true">
          <span class="ob-method__title">${ico('link')} Web / QR link <span class="ob-status--blocked">Not available</span></span>
          <span class="ob-method__text">LEO has no WhatsApp Web bridge, so it cannot link a personal device. You can view a local setup QR below — it is a one-time code for this computer, not a device link.</span>
          <button class="ob-btn" type="button" data-ob-action="method" data-method="webqr">View local setup QR</button>
        </div>
      </div>
      ${state.method === 'cloud' ? `<div class="ob-callout ob-callout--accent">${ico('bolt')}<span><strong>Cloud API uses credentials.</strong> The Phone number ID is a Meta ID, not your WhatsApp phone number. Continue to guided fields.</span></div>
        ${guideHTML('tail', state.channel)}` : ''}
      ${state.method === 'webqr' ? `<div class="ob-callout ob-callout--warn">${ico('alert')}<span><strong>Scan with WhatsApp is not supported.</strong> Open WhatsApp &gt; Linked Devices &gt; Link a Device only works with a live device-link QR from a bridge. This QR cannot be scanned there.</span></div>
        ${renderQrPanel()}` : ''}
    </section>`;
  }

  function stepFields() {
    const fields = asObjects(state.details?.fields)
      .map((field, index) => normalizeField(field, index))
      .filter((field) => !POLICY_FIELDS.has(field.name));
    if (!fields.length) return `<section class="ob-section"><h4 class="ob-section__title">Enter your credentials</h4>
      <div class="ob-callout ob-callout--warn">${ico('alert')}<span>The guided field list is unavailable in this runtime. You can still open the basic form; the backend validates everything before saving.</span></div>
      <button class="ob-btn" type="button" data-ob-action="fallback">Open basic form</button></section>`;
    return `<section class="ob-section"><h4 class="ob-section__title">Where to find these values</h4>
      <p class="ob-section__text">Leave a saved secret blank to keep it. Nothing you type is shown back after saving.</p>
      ${state.channel === 'whatsapp' ? `<div class="ob-callout ob-callout--accent">${ico('alert')}<span><strong>Phone number ID is not your phone number.</strong> Copy the Meta Phone number ID from Business Manager; use your own phone number in Allowed senders on the next step.</span></div>` : ''}
      <div class="ob-form">${fields.map(fieldHTML).join('')}</div>
    </section>`;
  }

  function stepAccess() {
    const fields = asObjects(state.details?.fields)
      .map((field, index) => normalizeField(field, index))
      .filter((field) => POLICY_FIELDS.has(field.name));
    const synthesized = fields.length ? fields : [
      { name: 'allowlist', label: 'Allowed senders', kind: 'textarea', required: false, configured: false,
        hint: 'One per line or separated by commas. Use international format, e.g. +15551234567.' },
      { name: 'mode', label: 'Admission mode', kind: 'choice', required: false, configured: false,
        options: [
          { value: 'allowlist', label: 'Only people on the list' },
          { value: 'open', label: 'Anyone' },
          { value: 'deny', label: 'Nobody' },
        ], default: 'allowlist' },
    ];
    return `<section class="ob-section"><h4 class="ob-section__title">Who can message LEO?</h4>
      <div class="ob-callout">${ico('alert')}<span><strong>Start with your own number</strong> in international format (e.g. +15551234567). An empty allowed-senders list blocks everyone.</span></div>
      <div class="ob-form">${synthesized.map((field, index) => fieldHTML(field, index)).join('')}</div>
      ${state.details?.mode_note ? `<p class="ob-note ob-note--warn">${esc(state.details.mode_note)}</p>` : ''}
    </section>`;
  }

  function stepCheck() {
    const callback = /^https:\/\//i.test(String(state.details?.callback_url || '')) ? state.details.callback_url : '';
    return `<section class="ob-section"><h4 class="ob-section__title">Check, then enable</h4>
      <p class="ob-section__text">Check my settings validates fields without sending a message, linking a device, or contacting your provider.</p>
      <div class="ob-callout ob-callout--accent">${ico('check')}<span>After enabling, credentials are saved. Sending and inbound replies are not tested by LEO, even when the offline check passes.</span></div>
      ${state.channel === 'whatsapp' && state.method === 'webqr'
        ? `<p class="ob-note ob-note--warn">Your local setup code does not link WhatsApp. Cloud API credentials are still required to send.</p>` : ''}
      ${state.details?.persistence_note ? `<p class="ob-note ob-note--warn">${esc(state.details.persistence_note)} Set LEO_CREDENTIALS_KEY to keep credentials after a restart.</p>` : ''}
      ${callback ? `<div class="ob-copy"><label class="ob-sr-only" for="ob-callback-url">Webhook callback URL</label>
        <input class="ob-input ob-input--mono" id="ob-callback-url" value="${esc(callback)}" readonly />
        <button class="ob-btn" type="button" data-ob-action="copy" data-ob-kind="callback">${ico('copy')} Copy</button></div>
        <p class="ob-note">Set this URL, and your webhook verify token, in the Meta dashboard.</p>` : ''}
      <button class="ob-btn" type="button" data-ob-action="check">${ico('check')} Check my settings</button>
      <div class="ob-check-result" id="ob-check-result" hidden aria-live="polite"></div>
      <div class="ob-result" id="ob-result" hidden aria-live="polite"></div>
    </section>`;
  }

  // ---- field controls ----------------------------------------------------
  // The backend uses {name,label,hint,placeholder,...}; catalog payloads use
  // {key,title,description,example,type}. Normalise both before rendering so
  // no control is ever created without a visible label, hint or placeholder.
  function normalizeField(spec) {
    const field = (spec && typeof spec === 'object') ? spec : (typeof spec === 'string' ? { name: spec } : {});
    const name = firstText(field, ['name', 'key', 'id', 'field', 'env', 'variable']) || 'field';
    const labelText = firstText(field, ['label', 'title', 'name', 'key', 'id']) || humanize(name) || 'Setting';
    const hint = field.hint || field.help || firstText(field, ['description', 'tooltip', 'info', 'message'])
      || `Enter ${labelText.toLowerCase()} exactly as provided by your provider.`;
    const kind = firstText(field, ['kind', 'type', 'input_type', 'control', 'format']).toLowerCase() || 'text';
    const configured = field.configured === true || field.saved === true;
    const sensitive = field.sensitive === true || field.secret === true || field.is_secret === true
      || ['secret', 'password', 'token', 'credential'].includes(kind)
      || /token|secret|password|api[_-]?key|credential/i.test(name);
    const example = field.placeholder || field.example || field.default || firstText(field, ['sample']);
    const placeholder = sensitive && configured
      ? 'Leave blank to keep the saved value'
      : (example || (hint.length <= 48 ? hint : `Enter ${labelText.toLowerCase()}`));
    return {
      __obNormalized: true,
      name, label: labelText, hint, kind, configured, sensitive, placeholder,
      required: field.required === true && !configured,
      default: firstText(field, ['default', 'value']),
      // Keep the raw shape so normalizeOptions can read options, choices and
      // values arrays without an empty one masking the others.
      options: field,
    };
  }

  // Flatten strings, {value,label}, {value,name}, {id,title}, and nested
  // options/choices/values arrays into objects with primitive value + label.
  function collectOptionValues(source, depth = 0) {
    if (Array.isArray(source)) {
      const out = [];
      source.forEach((item) => out.push(...collectOptionValues(item, depth + 1)));
      return out;
    }
    if (source && typeof source === 'object' && depth < 4) {
      const out = [];
      ['options', 'choices', 'values', 'items'].forEach((key) => {
        if (source[key] !== undefined && source[key] !== null) out.push(...collectOptionValues(source[key], depth + 1));
      });
      if (!out.length && ['value', 'id', 'label', 'name', 'title'].some((key) => source[key] !== undefined)) out.push(source);
      return out;
    }
    return source === null || source === undefined ? [] : [source];
  }
  function normalizeOptions(source, name = '') {
    const options = [];
    collectOptionValues(source).forEach((entry) => {
      let value = '';
      let label = '';
      if (typeof entry === 'string' || typeof entry === 'number' || typeof entry === 'boolean') {
        value = String(entry).trim();
        label = MODE_LABELS[value] || humanize(value);
      } else if (entry && typeof entry === 'object') {
        const explicitValue = firstText(entry, ['value', 'id', 'key']);
        const explicitLabel = firstText(entry, ['label', 'title', 'text', 'name']);
        const candidate = firstText(entry, ['name', 'label', 'title', 'id', 'key']) || explicitLabel;
        value = explicitValue || (MODE_LABELS[candidate] ? candidate : (candidate || explicitLabel));
        label = explicitLabel || MODE_LABELS[value] || humanize(value);
      }
      value = String(value || '').trim();
      label = String(label || '').trim();
      // Never let an unnormalised object reach an <option> as "[object Object]".
      if (!value || !label || value === '[object Object]' || label === '[object Object]') return;
      if (!options.some((option) => option.value === value)) options.push({ value, label });
    });
    if (!options.length && String(name) === 'mode') {
      return Object.entries(MODE_LABELS).map(([value, label]) => ({ value, label }));
    }
    return options;
  }

  function fieldHTML(field) {
    const spec = field && field.__obNormalized === true ? field : normalizeField(field);
    const { name, kind, configured, sensitive, label, hint, placeholder, required } = spec;
    const id = `ob-f-${name.replace(/[^A-Za-z0-9_-]/g, '_')}`;
    const described = `${id}-hint ${id}-error`;
    const requiredAttr = required ? ' aria-required="true"' : '';
    let control;
    if (TEXTAREA_KINDS.has(kind)) {
      control = `<textarea class="ob-input ob-textarea" id="${esc(id)}" data-ob-field="${esc(name)}" rows="3" placeholder="${esc(placeholder)}" aria-describedby="${esc(described)}"${requiredAttr}></textarea>`;
    } else if (CHOICE_KINDS.has(kind)) {
      const chosen = name === 'mode' && state.form.mode ? state.form.mode : (spec.default || '');
      const options = normalizeOptions(spec.options, name);
      control = `<select class="ob-input ob-select" id="${esc(id)}" data-ob-field="${esc(name)}" aria-describedby="${esc(described)}"${requiredAttr}>${options.map((option) => {
        const labelText = name === 'mode' && MODE_LABELS[option.value] ? MODE_LABELS[option.value] : option.label;
        const selected = option.value === String(chosen) ? ' selected' : '';
        return `<option value="${esc(option.value)}"${selected}>${esc(labelText)}</option>`;
      }).join('')}</select>`;
    } else {
      const inputType = sensitive || ['secret', 'password'].includes(kind) ? 'password'
        : ['number', 'email', 'url', 'tel'].includes(kind) ? kind : 'text';
      control = `<input class="ob-input" id="${esc(id)}" type="${inputType}"
        data-ob-field="${esc(name)}" autocomplete="off" placeholder="${esc(placeholder)}" aria-describedby="${esc(described)}"${requiredAttr} />`;
    }
    return `<div class="ob-field${TEXTAREA_KINDS.has(kind) ? ' ob-field--wide' : ''}">
      <label class="ob-field__label" for="${esc(id)}">${esc(label)}${required ? ' (required)' : ''}${configured ? ' <span class="ob-saved">Saved</span>' : ''}</label>
      ${control}
      <p class="ob-field__hint" id="${esc(id)}-hint">${esc(hint)}</p>
      <p class="ob-field__error" id="${esc(id)}-error" role="alert" hidden>${ico('alert')}<span></span></p>
    </div>`;
  }

  // ---- local setup QR ----------------------------------------------------
  function codeFromUrl(raw) {
    const match = /[#?&]code=([A-Za-z0-9_-]{20,128})/.exec(String(raw || ''));
    return match ? match[1] : '';
  }
  function sameOriginUrl(raw) {
    if (!raw) return '';
    try {
      const url = new URL(String(raw), location.href);
      return url.origin === location.origin ? url.href : '';
    } catch { return ''; }
  }
  function svgSource(body) {
    const qr = body.qr && typeof body.qr === 'object' ? body.qr : {};
    const candidate = body.qr_svg || qr.data_uri || qr.svg || '';
    const raw = String(candidate || '');
    if (!raw) return '';
    if (raw.startsWith('data:image/svg+xml')) return raw;
    if (raw.startsWith('<svg')) return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(raw)}`;
    return '';
  }
  function normalizeStart(body) {
    const setupUrl = String(body.setup_url || body.qr?.payload || '');
    const expiresIn = Number(body.expires_in || 0);
    const expiresAt = String(body.expires_at || '');
    let deadline = Date.now() + (expiresIn > 0 ? expiresIn : 60) * 1000;
    const parsed = Date.parse(expiresAt);
    if (Number.isFinite(parsed) && parsed > 0) deadline = parsed;
    const status = statusInfo(body);
    return {
      svgSrc: svgSource(body),
      token: String(body.token || ''),
      sessionId: String(body.session_id || ''),
      code: codeFromUrl(setupUrl) || String(body.code || ''),
      setupUrl: sameOriginUrl(setupUrl),
      statusUrl: String(body.status_url || (body.token ? `/onboarding/whatsapp/status?token=${encodeURIComponent(body.token)}` : '')),
      expiresAt, deadline, status,
    };
  }
  function statusInfo(body) {
    const raw = String(body?.status || body?.state || '').toLowerCase();
    let tone = 'warn', icon = 'clock', label = 'Waiting for confirmation';
    let message = String(body?.message || 'Open the setup link on this computer, or scan it with your phone camera, then confirm the code.');
    if (['paired', 'confirmed'].includes(raw)) {
      tone = 'ok'; icon = 'check'; label = 'Local setup confirmed';
      message = String(body?.message || 'No WhatsApp device is linked. Finish with the official Cloud API credentials.');
    } else if (['failed', 'expired'].includes(raw)) {
      tone = 'blocked'; icon = 'alert'; label = 'Setup expired or failed';
    } else if (raw === 'blocked') {
      tone = 'blocked'; icon = 'alert'; label = 'Not available';
    }
    if (body?.linked === true) { tone = 'ok'; icon = 'check'; label = 'Live link reported'; message = String(body.message || 'The backend reported a device link.'); }
    return {
      raw, tone, icon, label, message,
      nextStep: String(body?.next_step || ''),
      configured: body?.configured === true,
      missing: safeList(body?.missing),
    };
  }
  function maskedCode(code) {
    return code ? `${'•'.repeat(Math.max(4, Math.min(12, code.length - 4)))}${code.slice(-4)}` : '';
  }
  function countdownText() {
    if (!state.qr) return '';
    const seconds = Math.max(0, Math.ceil((state.qr.deadline - Date.now()) / 1000));
    return seconds ? `Code expires in ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}` : 'Code expired — regenerate to continue';
  }

  function renderQrPanel() {
    // Keep the .ob-qr wrapper in every state: startQr() swaps this panel by
    // replacing that node, so an unwrapped placeholder would never update.
    if (state.qrError) return `<div class="ob-qr" id="ob-qr-panel">
      <div class="ob-qr__placeholder">Setup QR unavailable</div>
      <div class="ob-qr__details"><div class="ob-callout ob-callout--warn">${ico('alert')}<span>${esc(state.qrError)} Start a new setup or use the official Cloud API steps.</span></div>
        <div class="ob-qr__actions"><button class="ob-btn" type="button" data-ob-action="start-qr">Try again</button>
        <button class="ob-btn ob-btn--quiet" type="button" data-ob-action="method" data-method="cloud">Use Cloud API instead</button></div>
      </div>
    </div>`;
    const qr = state.qr;
    if (!qr) return '<div class="ob-qr" id="ob-qr-panel"><div class="ob-qr__placeholder" role="status">Preparing a local setup code…</div></div>';
    const status = qr.status;
    const expired = Date.now() >= qr.deadline;
    const visual = qr.svgSrc
      ? `<img class="ob-qr__image" id="ob-qr-image" src="${esc(qr.svgSrc)}" alt="Local setup QR code. This is not a WhatsApp device-link QR." />`
      : '<div class="ob-qr__placeholder" id="ob-qr-image">No QR image was returned by this runtime.</div>';
    return `<div class="ob-qr" id="ob-qr-panel">
      <div class="ob-qr__visual">${visual}<p class="ob-qr__caption">Local setup code · not a WhatsApp device link</p></div>
      <div class="ob-qr__details">
        <div class="ob-qr__status ob-status--${expired ? 'blocked' : status.tone}" id="ob-qr-status" role="status" aria-live="polite">${ico(expired ? 'alert' : status.icon)}<span id="ob-qr-status-text">${esc(expired ? 'Code expired' : status.label)}</span></div>
        <p class="ob-qr__message" id="ob-qr-message">${esc(status.message)}</p>
        <p class="ob-qr__countdown" id="ob-qr-countdown">${esc(countdownText())}</p>
        <ol class="ob-list">
          <li><strong>Scan with WhatsApp:</strong> not available here. This QR is a local setup code, not a device link.</li>
          <li>Open the setup link on this computer, or scan it with your phone camera.</li>
          <li>Confirm the one-time code. It expires quickly and works once.</li>
        </ol>
        ${qr.setupUrl ? `<div class="ob-copy"><label class="ob-sr-only" for="ob-qr-link">Setup link</label>
          <input class="ob-input ob-input--mono" id="ob-qr-link" value="${esc(qr.setupUrl)}" readonly />
          <button class="ob-btn" type="button" data-ob-action="copy" data-ob-kind="link">${ico('copy')} Copy link</button></div>` : ''}
        ${qr.code ? `<div class="ob-copy"><span class="ob-sr-only">One-time setup code</span>
          <input class="ob-input ob-input--mono" value="${esc(maskedCode(qr.code))}" readonly aria-label="One-time setup code, hidden" />
          <button class="ob-btn" type="button" data-ob-action="copy" data-ob-kind="code">${ico('copy')} Copy code</button></div>` : ''}
        <div class="ob-qr__actions">
          ${qr.setupUrl ? `<a class="ob-btn" href="${esc(qr.setupUrl)}">Open setup page</a>` : ''}
          <button class="ob-btn ob-btn--primary" type="button" data-ob-action="confirm" ${expired ? 'disabled' : ''}>I confirmed on this computer</button>
          <button class="ob-btn" type="button" data-ob-action="start-qr">Regenerate</button>
          <button class="ob-btn ob-btn--quiet" type="button" data-ob-action="method" data-method="cloud">Use Cloud API instead</button>
        </div>
        <p class="ob-note" id="ob-qr-extra">${status.missing.length ? `Still needed: ${esc(status.missing.join(', '))}` : ''}</p>
        <p class="ob-note ob-note--danger" id="ob-qr-error" role="alert" hidden></p>
        ${status.nextStep ? `<p class="ob-note">${esc(status.nextStep)}</p>` : ''}
      </div>
    </div>`;
  }

  function startQr() {
    stopPolling();
    state.qr = null;
    state.qrError = '';
    const host = state.dialog?.querySelector('#ob-dialog-body');
    if (host) { const panel = host.querySelector('.ob-qr'); if (panel) panel.outerHTML = renderQrPanel(); }
    request('/onboarding/whatsapp/start', { method: 'POST', headers: WRITE_HEADERS, body: '{}' })
      .then((body) => {
        if (!state.dialog || state.channel !== 'whatsapp' || state.method !== 'webqr') return;
        state.qr = normalizeStart(body);
        const panel = state.dialog.querySelector('.ob-qr');
        if (panel) panel.outerHTML = renderQrPanel();
        startClock();
        startPolling();
      })
      .catch((error) => {
        if (!state.dialog || state.channel !== 'whatsapp' || state.method !== 'webqr') return;
        state.qrError = error.status === 404 || error.status === 405
          ? 'Guided QR setup is not available in this LEO runtime.'
          : `The setup code could not be created: ${error.message}`;
        const panel = state.dialog.querySelector('.ob-qr');
        if (panel) panel.outerHTML = renderQrPanel();
      });
  }

  function startClock() {
    if (state.clockTimer) clearInterval(state.clockTimer);
    state.clockTimer = setInterval(() => {
      const el = state.dialog?.querySelector('#ob-qr-countdown');
      if (!el) { stopClock(); return; }
      el.textContent = countdownText();
      if (Date.now() >= (state.qr?.deadline || 0)) {
        const button = state.dialog.querySelector('[data-ob-action="confirm"]');
        if (button) button.disabled = true;
        stopPolling();
      }
    }, 1000);
  }
  function stopClock() {
    if (state.clockTimer) { clearInterval(state.clockTimer); state.clockTimer = null; }
  }
  function stopPolling() {
    if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  }
  function startPolling() {
    stopPolling();
    if (!state.qr?.statusUrl) return;
    let ticks = 0;
    state.pollTimer = setInterval(async () => {
      if (!state.dialog || document.hidden || !state.qr) return;
      if (++ticks > POLL_LIMIT) { stopPolling(); return; }
      try {
        const body = await request(state.qr.statusUrl);
        applyStatus(body);
        if (['paired', 'confirmed', 'failed', 'expired', 'blocked'].includes(statusInfo(body).raw)) stopPolling();
      } catch (error) {
        if (error.status === 404 || error.status === 405) {
          stopPolling();
          const extra = state.dialog?.querySelector('#ob-qr-extra');
          if (extra) extra.textContent = 'Live status is not available in this runtime. Confirm on the setup page, then check again.';
        }
      }
    }, POLL_MS);
  }
  function applyStatus(body) {
    if (!state.qr || !state.dialog) return;
    state.qr.status = statusInfo(body);
    const status = state.qr.status;
    const holder = state.dialog.querySelector('#ob-qr-status');
    if (holder) holder.innerHTML = `${ico(status.icon)}<span id="ob-qr-status-text">${esc(status.label)}</span>`;
    const message = state.dialog.querySelector('#ob-qr-message');
    if (message) message.textContent = status.message;
    const extra = state.dialog.querySelector('#ob-qr-extra');
    if (extra) {
      const parts = [];
      if (status.configured) parts.push('Settings saved.');
      if (status.missing.length) parts.push(`Still needed: ${status.missing.join(', ')}`);
      extra.textContent = parts.join(' ');
    }
  }
  async function confirmQr() {
    if (!state.qr || state.busy) return;
    state.busy = true;
    const errorEl = state.dialog?.querySelector('#ob-qr-error');
    try {
      const attempts = [];
      if (state.qr.token || state.qr.sessionId) attempts.push({
        path: '/onboarding/whatsapp/confirm',
        body: { token: state.qr.token, session_id: state.qr.sessionId },
      });
      if (state.qr.code) attempts.push({ path: '/onboarding/redeem', body: { code: state.qr.code } });
      if (!attempts.length) throw new Error('No confirmation route was returned by this runtime.');
      let last;
      for (const attempt of attempts) {
        try {
          const body = await request(attempt.path, { method: 'POST', headers: WRITE_HEADERS, body: JSON.stringify(attempt.body) });
          applyStatus(body);
          toast('Setup code confirmed locally. No WhatsApp device was linked.', 'success');
          stopPolling();
          return;
        } catch (error) {
          last = error;
          if (error.status !== 404 && error.status !== 405) throw error;
        }
      }
      throw last;
    } catch (error) {
      if (errorEl) { errorEl.textContent = error.message; errorEl.hidden = false; }
    } finally { state.busy = false; }
  }

  // ---- validation and enrollment ----------------------------------------
  function requestBody() {
    captureFields();
    return {
      values: { ...state.form.values },
      allowlist: [...state.form.allowlist],
      mode: state.form.mode || 'allowlist',
    };
  }

  // The backend returns field errors as a list of {field, message} records;
  // older payloads may use a name -> message map. Normalise both shapes.
  function errorMap(value) {
    const result = {};
    if (Array.isArray(value)) value.forEach((entry) => {
      if (entry && typeof entry === 'object') {
        const name = String(entry.field || entry.name || '');
        if (name) result[name] = String(entry.message || 'Check this value and try again.');
      }
    });
    else if (value && typeof value === 'object') Object.entries(value).forEach(([name, message]) => {
      result[name] = typeof message === 'string' ? message : String(message?.message || 'Check this value and try again.');
    });
    return result;
  }

  function renderCheckResult(validation, verify) {
    const host = state.dialog?.querySelector('#ob-check-result');
    if (!host) return;
    const items = [];
    if (validation) {
      const firstError = Object.values(errorMap(validation.errors))[0];
      items.push({ icon: validation.valid ? 'check' : 'alert', text: validation.valid
        ? 'Your entries pass the offline field check.'
        : `Needs attention: ${firstError || 'check the highlighted fields'}.` });
      safeList(validation.missing).forEach((item) => items.push({ icon: 'alert', text: `Still needed: ${item}` }));
    }
    if (verify) {
      items.push({ icon: verify.ready ? 'check' : 'alert', text: verify.ready
        ? 'Saved settings are complete. Live delivery is still untested.'
        : `Saved settings are incomplete: ${safeList(verify.missing).join(', ') || 'details missing'}.` });
      const verification = verify.verification;
      if (verification && typeof verification === 'object') {
        const status = String(verification.status || 'unknown');
        items.push({ icon: status === 'degraded' || status === 'blocked' ? 'alert' : 'check',
          text: `Verification status: ${status}. ${verification.message || ''}`.trim() });
      }
    }
    host.innerHTML = `<ul class="ob-checklist">${items.map((item) =>
      `<li>${ico(item.icon)}<span>${esc(item.text)}</span></li>`).join('')}</ul>
      <p class="ob-note">Offline check only · no test message was sent and no device was linked.</p>`;
    host.hidden = false;
  }

  async function checkSettings() {
    captureFields();
    const body = requestBody();
    const host = state.dialog?.querySelector('#ob-check-result');
    if (host) { host.textContent = 'Checking your settings…'; host.hidden = false; }
    try {
      const validation = await request(`/integrations/${encodeURIComponent(state.channel)}/validate`, { method: 'POST', body: JSON.stringify(body) });
      state.validation = validation;
      state.errors = errorMap(validation.errors);
      showFieldErrors();
      let verify = null;
      try { verify = await request(`/integrations/${encodeURIComponent(state.channel)}/verify`, { method: 'POST' }); }
      catch { /* verify is optional; the offline field check still reports */ }
      renderCheckResult(validation, verify);
      return validation.valid === true;
    } catch (error) {
      if (error.status === 404 || error.status === 405) {
        renderCheckResult({ valid: true, missing: [], errors: {} }, null);
        const note = state.dialog?.querySelector('#ob-check-result .ob-note');
        if (note) note.textContent = 'The field-check route is unavailable here. The backend still validates values when you enable.';
        return true;
      }
      if (host) { host.textContent = error.message; }
      return false;
    }
  }

  async function enableChannel() {
    if (state.busy) return;
    state.busy = true;
    const button = state.dialog?.querySelector('#ob-next');
    if (button) button.setAttribute('aria-busy', 'true');
    try {
      const valid = await checkSettings();
      if (!valid) {
        const policyError = Object.keys(state.errors).some((name) => ['allowlist', 'mode'].includes(name));
        state.step = policyError ? 2 : 1;
        renderDialogBody();
        toast('Fix the highlighted fields, then continue.', 'warn');
        return;
      }
      const result = await request(`/integrations/${encodeURIComponent(state.channel)}/enable`, {
        method: 'POST', body: JSON.stringify(requestBody()),
      });
      const resultHost = state.dialog?.querySelector('#ob-result');
      if (resultHost) {
        resultHost.textContent = `${labelFor(state.channel)} settings saved. Sending and inbound replies are not tested by LEO.`;
        resultHost.hidden = false;
      }
      toast(`${labelFor(state.channel)} settings saved — live delivery not tested`, 'success');
      closeDialog();
      refresh();
      window.dispatchEvent(new CustomEvent('leo:onboarding-enabled', { detail: { channel: state.channel, result } }));
    } catch (error) {
      const resultHost = state.dialog?.querySelector('#ob-result');
      if (resultHost) {
        const errors = errorMap(error.body?.detail?.errors || error.body?.errors);
        state.errors = errors;
        showFieldErrors();
        const messages = Object.values(errors);
        resultHost.textContent = messages.length ? messages.join(' ') : error.message;
        resultHost.hidden = false;
      }
    } finally {
      state.busy = false;
      if (button) button.removeAttribute('aria-busy');
    }
  }

  async function copyValue(kind) {
    let value = '';
    if (kind === 'code') value = state.qr?.code || '';
    else if (kind === 'link') value = state.qr?.setupUrl || '';
    else if (kind === 'callback') value = state.details?.callback_url || '';
    if (!value) { toast('Nothing to copy yet', 'warn'); return; }
    try { await navigator.clipboard.writeText(value); toast(kind === 'code' ? 'Setup code copied' : 'Link copied', 'success'); }
    catch { toast('Clipboard unavailable — select the value and copy it', 'warn'); }
  }

  function onDialogClick(event) {
    const action = event.target.closest('[data-ob-action]');
    if (!action) return;
    const name = action.dataset.obAction;
    if (name === 'close') return closeDialog();
    if (name === 'back') { captureFields(); state.step = Math.max(0, state.step - 1); renderDialogBody(); return; }
    if (name === 'next') {
      if (state.step === STEPS.length - 1) return enableChannel();
      if (state.step === 0 && state.channel === 'whatsapp' && state.method === 'webqr') {
        // QR is informational; Cloud API credentials are still required.
      }
      captureFields();
      state.step = Math.min(STEPS.length - 1, state.step + 1);
      renderDialogBody();
      return;
    }
    if (name === 'method') {
      state.method = action.dataset.method === 'webqr' ? 'webqr' : 'cloud';
      renderDialogBody();
      if (state.method === 'webqr') startQr(); else { stopPolling(); stopClock(); }
      return;
    }
    if (name === 'start-qr') return startQr();
    if (name === 'confirm') return confirmQr();
    if (name === 'copy') return copyValue(action.dataset.obKind);
    if (name === 'check') return checkSettings();
    if (name === 'fallback') {
      const channel = state.channel;
      closeDialog();
      window.dispatchEvent(new CustomEvent('leo:open-raw-channel-form', { detail: { channel } }));
    }
  }

  window.LEO_ONBOARDING = { mount, open: openWizard, render: renderCards, get state() { return state; } };
})();
