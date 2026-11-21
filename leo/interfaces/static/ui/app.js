/* LEO Night Instrument — dashboard behaviour.
   The conversation view is owned by ui/conversation/chat.js (the Hermes-derived
   conversation module); this file drives every other view plus shared chrome. */

(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const VIEWS = {
    chat: ['Conversation', 'UNIFIED SESSION'], tasks: ['Tasks', 'DURABLE EXECUTION'],
    agents: ['Agents', 'SPECIALIST FLEET'], skills: ['Skills', 'CAPABILITY LIBRARY'],
    memory: ['Memory', 'LAYERED RECALL'], models: ['Models', 'INTELLIGENCE ENGINE'],
    tools: ['Tools', 'TOOL ENGINE'], approvals: ['Approvals', 'HUMAN CONTROL'],
    integrations: ['Integrations', 'CONNECTED SURFACES'], events: ['Events', 'EXECUTION TRACE'],
    settings: ['Settings', 'LOCAL CONFIGURATION'],
  };
  const ENDPOINTS = {
    health: '/health', tasks: '/tasks', agents: '/agents', skills: '/skills',
    memory: '/memory', usage: '/models/usage', tools: '/tools', approvals: '/approvals',
    integrations: '/integrations', events: '/events?limit=40', audit: '/audit?limit=10',
    schedules: '/schedules',
  };
  // A Run action is exposed only when the API advertises task_run in /health
  // (and POST /tasks/{task_id}/run exists), so no dead button is shown.
  const TASK_RUN_ROUTE = (id) => `/tasks/${encodeURIComponent(id)}/run`;
  const TASK_STATUSES = ['created', 'planning', 'ready', 'running', 'waiting',
    'blocked', 'verifying', 'completed', 'failed', 'cancelled'];
  // Live event stream. If the route is missing or drops, /events polling
  // remains the source of truth (see loadAll/schedulePoll).
  const EVENTS_STREAM_ROUTE = '/events/stream';
  const EVENTS_STREAM_TOKEN_ROUTE = '/events/stream-token';
  const EVENT_STREAM_LIMIT = 50;
  const EVENT_RETRY_MIN_MS = 1000;
  const EVENT_RETRY_MAX_MS = 30000;
  const state = {
    view: 'chat', data: {},
    pollMs: 15000, timer: null, token: sessionStorage.getItem('leo_api_token') || '',
    compact: sessionStorage.getItem('leo_compact') === '1', modal: null, modalConfirm: false,
    catalog: {}, modalForm: false, modalSilent: false,
    eventItems: [], eventSource: null, eventRetryTimer: null,
    eventBackoffMs: EVENT_RETRY_MIN_MS, eventStreamLive: false, eventStreamUnsupported: false,
    shellHealth: null, shellStatus: null,
  };

  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const ico = (name, cls = 'icon') => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const fmt = (v) => v === undefined || v === null ? '—' : typeof v === 'number' ? v.toLocaleString() : esc(v);
  const short = (v, n = 90) => { const s = String(v ?? ''); return s.length > n ? s.slice(0, n) + '…' : esc(s); };

  // /health advertises capability flags as either an object or a list.
  function hasCapability(name) {
    const health = state.data.health || {};
    const flags = health.capabilities || health.features || {};
    if (Array.isArray(flags)) return flags.includes(name);
    if (flags && typeof flags === 'object' && flags[name] === true) return true;
    return health[name] === true;
  }
  function taskRunEnabled() {
    return hasCapability('task_run') || state.data.tasks?.run_enabled === true;
  }

  // Skills wrap their result in {result: SkillRunResult}; tools return the
  // ToolResult directly. Show whichever payload the backend actually returned.
  function unwrapResult(result) {
    if (result && typeof result === 'object' && result.result && typeof result.result === 'object'
        && ('success' in result.result || 'error' in result.result)) return result.result;
    return result;
  }
  function formatResult(result) {
    if (result === undefined || result === null) return '';
    if (typeof result === 'string') return result;
    try { return JSON.stringify(result, null, 2); } catch { return String(result); }
  }
  function showModalOutput(payload) {
    const text = formatResult(payload);
    if (!text) return;
    $('modal-output-text').textContent = text.length > 8000 ? `${text.slice(0, 8000)}\n… truncated` : text;
    $('modal-output').hidden = false;
  }

  function badge(value) {
    const v = String(value ?? 'unknown').toLowerCase();
    const good = ['ok', 'completed', 'approved', 'available', 'healthy', 'configured', 'enabled', 'success', 'true'];
    const warn = ['pending', 'running', 'waiting', 'degraded', 'verifying', 'blocked', 'created', 'ready'];
    const bad = ['failed', 'denied', 'error', 'unavailable', 'dangerous', 'high', 'cancelled'];
    const kind = good.includes(v) ? 'success' : warn.includes(v) ? 'warn' : bad.includes(v) ? 'danger' : '';
    return `<span class="badge ${kind ? 'badge--' + kind : ''}">${esc(v)}</span>`;
  }

  function setAvailability(id, enabled, label) {
    const el = $(id);
    if (!el) return;
    el.textContent = label;
    el.className = `badge ${enabled ? 'badge--success' : ''}`;
  }

  function loading(id, rows = 3) {
    const el = $(id); if (!el) return;
    el.innerHTML = Array.from({ length: rows }, () => '<div class="skeleton skeleton--wide"></div><div class="skeleton skeleton--short"></div>').join('');
  }
  function errorState(id, message) {
    const el = $(id); if (!el) return;
    el.innerHTML = `<div class="error-state">${ico('alert')}<span class="error-state__title">Could not load this view</span><span class="error-state__text">${esc(message)}</span><button class="btn btn--secondary" data-retry="1">Retry</button></div>`;
  }
  function empty(id, title, text, icon = 'memory') {
    const el = $(id); if (!el) return;
    el.innerHTML = `<div class="empty">${ico(icon, 'icon icon--lg')}<span class="empty__title">${esc(title)}</span><span class="empty__text">${esc(text)}</span></div>`;
  }
  function toast(message, kind = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast--${kind}`;
    el.innerHTML = `${ico(kind === 'danger' ? 'alert' : kind === 'success' ? 'check' : 'bolt', 'icon icon--sm')}<span class="toast__text">${esc(message)}</span><button class="toast__close" aria-label="Dismiss message">${ico('x', 'icon icon--sm')}</button>`;
    const dismiss = () => el.remove();
    el.querySelector('.toast__close').addEventListener('click', dismiss);
    $('toasts').appendChild(el);
    setTimeout(dismiss, 5000);
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    // An explicit token (token validation) must not be replaced by the saved
    // session token.
    if (state.token && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${state.token}`);
    if (options.body && !headers.has('content-type')) headers.set('content-type', 'application/json');
    const response = await fetch(path, { ...options, headers });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401) toast('API token required — add it in Settings.', 'warn');
      throw new Error(body.detail || `${response.status} ${response.statusText}`);
    }
    return body;
  }

  // Shared surface for the conversation module (and future views).
  window.LEO = { api, toast, get data() { return state.data; } };
  // Guided onboarding refreshes this view after enabling a channel.
  window.LEO.refresh = () => loadAll();

  async function loadAll() {
    const entries = await Promise.all(Object.entries(ENDPOINTS).map(async ([key, url]) => {
      try { return [key, await api(url)]; } catch (error) { return [key, { __error: error.message }]; }
    }));
    entries.forEach(([key, value]) => { state.data[key] = value; });
    renderHealth(); renderAll();
    window.dispatchEvent(new CustomEvent('leo:data', { detail: state.data }));
  }
  function schedulePoll() {
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(loadAll, state.pollMs);
  }

  function renderHealth() {
    const h = state.data.health || {};
    const ok = h.status === 'ok';
    const pill = $('health-pill'), rail = $('rail-status');
    pill.className = `status ${ok ? 'status--live' : 'status--danger'}`;
    pill.querySelector('.status__dot').style.background = ok ? 'var(--accent)' : 'var(--danger)';
    $('health-text').textContent = h.__error ? 'OFFLINE' : `${h.status === 'ok' ? 'HEALTHY' : String(h.status || 'unknown').toUpperCase()} · ${h.default_provider || '—'}`;
    rail.className = `status ${ok ? 'status--live' : 'status--danger'}`;
    $('rail-status-text').textContent = ok ? 'Runtime healthy' : 'Runtime unavailable';
    $('rail-meta').textContent = `LOCAL · ${h.version ? 'v' + h.version : 'v0.1.0'}`;
    const snapshot = state.shellHealth ? { ...h, shell: state.shellHealth } : h;
    $('settings-snapshot').textContent = JSON.stringify(snapshot, null, 2);
    $('auth-badge').textContent = state.token ? 'token set' : 'no token';
    $('auth-badge').className = `badge ${state.token ? 'badge--success' : ''}`;
  }

  function renderTasks() {
    const tasks = state.data.tasks?.tasks || [];
    if (state.data.tasks?.__error) return errorState('task-list', state.data.tasks.__error);
    const counts = TASK_STATUSES;
    $('task-stats').innerHTML = counts.map((s) => `<div class="stat"><div class="stat__label">${s}</div><div class="stat__value ${s === 'running' ? 'is-accent' : ''}">${tasks.filter((t) => t.status === s).length}</div></div>`).join('');
    $('task-count').textContent = `${tasks.length} total`;
    const hint = $('task-hint');
    if (hint) hint.textContent = taskRunEnabled()
      ? 'Objectives are queued for durable workers; use Run to start one immediately.'
      : 'Objectives are queued for durable workers. This backend has no manual run route.';
    if (!tasks.length) return empty('task-list', 'No tasks yet', 'Create an objective to start durable, evidence-backed execution.', 'tasks');
    $('task-list').innerHTML = `<div class="list">${tasks.slice(0, 30).map((t) => `
      <div class="row"><span class="row__icon">${ico('tasks')}</span>
        <span><span class="row__title">${esc(t.objective)}</span><br/><span class="row__meta">${esc(t.id)} · step ${fmt(t.current_step)}</span></span>
        <span class="row__actions">${taskRunEnabled() ? `<button class="btn btn--secondary btn--sm" data-run-task="${esc(t.id)}">Run</button>` : ''}${badge(t.status)}${['completed', 'failed', 'cancelled'].includes(t.status) ? '' : `<button class="icon-btn" data-cancel="${esc(t.id)}" data-tip="Cancel task" aria-label="Cancel task ${esc(t.id)}">${ico('x', 'icon icon--sm')}</button>`}</span>
      </div>`).join('')}</div>
      <details class="disclosure"><summary>View task data</summary><div class="disclosure__body"><pre class="code">${esc(JSON.stringify(tasks.slice(0, 10), null, 2))}</pre></div></details>`;
  }

  function renderAgents() {
    const agents = state.data.agents?.agents || [];
    if (state.data.agents?.__error) return errorState('agent-list', state.data.agents.__error);
    const lede = $('agent-lede');
    if (lede) lede.textContent = `${agents.length} registered specialists from the live agent registry. Delegation is evidence-gated and isolated; no agent merges code automatically.`;
    $('agent-stats').innerHTML = [['Registered', agents.length], ['Enabled', agents.filter((a) => a.enabled !== false).length], ['Specialties', new Set(agents.flatMap((a) => a.capabilities || [])).size]].map(([k, v]) => `<div class="stat"><div class="stat__label">${k}</div><div class="stat__value is-accent">${v}</div></div>`).join('');
    $('agent-count').textContent = `${agents.length} specialists`;
    if (!agents.length) return empty('agent-list', 'No agents registered', 'The live agent registry returned no specialists.', 'agents');
    $('agent-list').innerHTML = `<div class="agent-grid">${agents.map((a) => `<article class="agent-card">
      <div class="agent-card__head"><span class="row__icon">${ico('agents')}</span><span><strong class="row__title">${esc(a.name)}</strong><br/><span class="agent-card__id">${esc(a.id)}</span></span></div>
      <p class="agent-card__purpose">${esc(short(a.purpose, 110))}</p>
      <div class="tag-list">${(a.capabilities || []).slice(0, 3).map((c) => `<span class="badge">${esc(c)}</span>`).join('')}</div>
      <div class="agent-card__foot"><span class="row__meta">limit ${fmt(a.task_limit)} · budget ${fmt(a.context_budget)}</span>${badge(a.enabled === false ? 'disabled' : 'enabled')}</div>
    </article>`).join('')}</div>`;
  }

  function renderAgentDelegation() {
    const ready = hasCapability('agent_delegate');
    setAvailability('delegate-availability', ready, ready ? 'available' : 'unavailable');
    const select = $('delegate-agent');
    if (!select) return;
    const previous = select.value;
    const agents = (state.data.agents?.agents || []).filter((a) => a.enabled !== false);
    select.replaceChildren();
    const auto = document.createElement('option');
    auto.value = ''; auto.textContent = 'Auto-select specialist';
    select.appendChild(auto);
    agents.forEach((a) => {
      const option = document.createElement('option');
      option.value = String(a.id); option.textContent = `${a.name || a.id} (${a.id})`;
      select.appendChild(option);
    });
    if ([...select.options].some((o) => o.value === previous)) select.value = previous;
    select.disabled = !ready;
    $('delegate-submit').disabled = !ready;
    $('delegate-hint').textContent = ready
      ? 'Delegation runs through the real agent engine; the returned task id is evidence-gated.'
      : 'agent_delegate is not advertised by /health; this control is disabled.';
  }

  function renderSchedules() {
    const ready = hasCapability('schedule');
    setAvailability('schedule-availability', ready, ready ? 'available' : 'unavailable');
    $('schedule-submit').disabled = !ready;
    const host = $('schedule-list');
    if (!host) return;
    if (!ready) { host.innerHTML = '<p class="hint">Scheduling is not advertised by /health.</p>'; return; }
    if (state.data.schedules?.__error) { host.innerHTML = `<p class="error-text">${esc(state.data.schedules.__error)}</p>`; return; }
    const schedules = state.data.schedules?.schedules || [];
    if (!schedules.length) { host.innerHTML = '<p class="hint">No schedules registered.</p>'; return; }
    host.innerHTML = `<div class="list">${schedules.map((s) => `<div class="row"><span class="row__icon">${ico('clock')}</span><span><span class="row__title">${esc(s.name)}</span><br/><span class="row__meta">${esc(s.kind)} · ${esc(s.expression)} · next ${esc(s.next_run_at || '—')}</span></span><span class="row__actions">${badge(s.enabled ? 'enabled' : 'disabled')}<button class="btn btn--danger btn--sm" data-delete-schedule="${esc(s.name || s.id)}">Delete</button></span></div>`).join('')}</div>`;
  }

  function renderChannelSend() {
    const ready = hasCapability('channel_send');
    setAvailability('send-availability', ready, ready ? 'available' : 'unavailable');
    const select = $('send-channel');
    if (!select) return;
    const previous = select.value;
    const channels = (state.data.integrations?.integrations || []).filter((i) =>
      i.kind === 'messaging' && i.status === 'configured' && !String(i.name).startsWith('channel-module:'));
    select.replaceChildren();
    const placeholder = document.createElement('option');
    placeholder.value = ''; placeholder.textContent = channels.length ? 'Select a channel' : 'No configured channels';
    select.appendChild(placeholder);
    channels.forEach((i) => {
      const option = document.createElement('option');
      option.value = String(i.name); option.textContent = String(i.name);
      select.appendChild(option);
    });
    if ([...select.options].some((o) => o.value === previous)) select.value = previous;
    select.disabled = !ready || !channels.length;
    $('send-submit').disabled = !ready || !channels.length;
    $('send-hint').textContent = ready
      ? 'Delivery status comes from the backend; provider acceptance is not reported as delivery.'
      : 'channel_send is not advertised by /health; this control is disabled.';
  }

  function showResultIn(id, payload) {
    const el = $(id);
    if (!el) return;
    const text = formatResult(payload);
    if (!text) { el.hidden = true; return; }
    el.textContent = text.length > 8000 ? `${text.slice(0, 8000)}\n… truncated` : text;
    el.hidden = false;
  }

  function renderSkills() {
    const skills = state.data.skills?.skills || [];
    if (state.data.skills?.__error) return errorState('skill-list', state.data.skills.__error);
    const q = $('skill-search').value.trim().toLowerCase();
    const shown = skills.filter((s) => !q || `${s.name} ${s.category} ${s.description}`.toLowerCase().includes(q));
    $('skill-count').textContent = `${shown.length} of ${skills.length}`;
    if (!shown.length) return empty('skill-list', 'No matching skills', 'Try another search term or register a native skill.', 'skills');
    $('skill-list').innerHTML = `<div class="list">${shown.map((s) => {
      const metadataOnly = s.has_handler !== true || s.enabled === false || /metadata/i.test(String(s.health || ''));
      const action = metadataOnly
        ? `<button class="btn btn--secondary btn--sm" data-skill-disabled="${esc(s.name)}" disabled title="Metadata-only skill: no local handler">${ico('play', 'icon icon--sm')} Run unavailable</button>`
        : `<button class="btn btn--secondary btn--sm" data-skill="${esc(s.name)}">${ico('play', 'icon icon--sm')} Run</button>`;
      return `<div class="row"><span class="row__icon">${ico('skills')}</span><span><span class="row__title">${esc(s.name)}</span><br/><span class="row__meta">${esc(s.category)} · ${esc(s.source)} · v${esc(s.version)}${metadataOnly ? ' · metadata only' : ''}</span></span><span class="row__actions">${badge(s.health || s.status)}${action}</span></div>`;
    }).join('')}</div>`;
  }

  function renderMemory() {
    const source = state.memoryQuery ? (state.data.memorySearch?.results || []) : (state.data.memory?.memories || []);
    if (state.data.memory?.__error) return errorState('memory-list', state.data.memory.__error);
    $('memory-count').textContent = state.memoryQuery ? `${source.length} matches` : `${source.length} recent`;
    if (!source.length) return empty('memory-list', state.memoryQuery ? 'Nothing recalled' : 'Memory is empty', state.memoryQuery ? 'No memory matched those terms. Try broader keywords.' : 'Add a durable fact to give LEO long-term context.', 'memory');
    $('memory-list').innerHTML = `<div class="list">${source.map((m) => `<div class="row"><span class="row__icon">${ico('memory')}</span><span><span class="row__title">${esc(m.content)}</span><br/><span class="row__meta">${esc(m.category)} · ${esc(m.kind)} · confidence ${fmt(m.confidence)} · v${fmt(m.version)}</span></span><span>${badge(m.category)}</span></div>`).join('')}</div>`;
  }

  function renderProviders() {
    // Rendered independently of usage: a fresh install has providers but no
    // measured calls, and the Providers tab must still be populated.
    const providers = state.data.health?.providers || {};
    const entries = Object.entries(providers);
    $('provider-list').innerHTML = entries.length
      ? `<div class="list">${entries.map(([name, model]) => `<div class="row"><span class="row__icon">${ico('models')}</span><span><span class="row__title">${esc(name)}</span><br/><span class="row__meta">${esc(typeof model === 'string' ? model : model?.model || 'no model')}</span></span><span>${badge('configured')}</span></div>`).join('')}</div>`
      : '<div class="empty"><span class="empty__title">No providers configured</span><span class="empty__text">Add a provider in leo.yaml to enable model calls.</span></div>';
  }

  function renderModels() {
    renderProviders();
    const rows = state.data.usage?.usage || [];
    if (state.data.usage?.__error) return errorState('usage-list', state.data.usage.__error);
    const totalTokens = rows.reduce((n, r) => n + (r.input_tokens || 0) + (r.output_tokens || 0), 0);
    const success = rows.filter((r) => r.success).length;
    const estimated = rows.filter((r) => r.metadata?.estimated).length;
    $('usage-stats').innerHTML = [
      ['Calls', rows.length], ['Tokens', totalTokens], ['Success rate', rows.length ? Math.round(success / rows.length * 100) + '%' : '—'], ['Estimated', estimated],
    ].map(([k, v], i) => `<div class="stat"><div class="stat__label">${k}</div><div class="stat__value ${i === 2 ? 'is-accent' : ''}">${fmt(v)}</div></div>`).join('');
    if (!rows.length) return empty('usage-list', 'No model calls yet', 'Usage appears after the first chat or delegated task.', 'models');
    const byProvider = rows.reduce((acc, r) => { acc[r.provider] = (acc[r.provider] || 0) + 1; return acc; }, {});
    $('usage-list').innerHTML = `<div class="progress" role="progressbar" aria-label="Model call success rate" aria-valuenow="${rows.length ? Math.round(success / rows.length * 100) : 0}" aria-valuemin="0" aria-valuemax="100" style="--value:${rows.length ? Math.round(success / rows.length * 100) : 0}%"><div class="progress__fill"></div></div>
      <div class="tag-list mt-4">${Object.entries(byProvider).map(([name, count]) => `<span class="badge badge--accent">${esc(name)} · ${count} calls</span>`).join('')}</div>
      <div class="list mt-4">${rows.slice(0, 25).map((r) => `<div class="row"><span class="row__icon">${ico('models')}</span><span><span class="row__title">${esc(r.provider)} / ${esc(r.model)}</span><br/><span class="row__meta">${fmt(r.input_tokens)} in · ${fmt(r.output_tokens)} out · ${fmt(Math.round(r.latency_ms))}ms · ${esc(r.metadata?.task_category || 'task')}</span></span><span>${badge(r.success ? 'success' : 'failed')}</span></div>`).join('')}</div>`;
  }

  function renderTools() {
    const tools = state.data.tools?.tools || [];
    if (state.data.tools?.__error) return errorState('tool-list', state.data.tools.__error);
    const q = $('tool-search').value.trim().toLowerCase();
    const shown = tools.filter((t) => !q || `${t.name} ${t.category}`.toLowerCase().includes(q));
    $('tool-count').textContent = `${shown.length} of ${tools.length}`;
    if (!shown.length) return empty('tool-list', 'No matching tools', 'Adjust the filter to see the native tool registry.', 'tools');
    $('tool-list').innerHTML = `<table class="table"><thead><tr><th>Tool</th><th>Category</th><th>Risk</th><th>Permissions</th><th></th></tr></thead><tbody>${shown.map((t) => `<tr><td><strong>${esc(t.name)}</strong></td><td>${esc(t.category)}</td><td>${badge(t.risk)}</td><td class="row__meta">${esc((t.permissions || []).join(', ') || '—')}</td><td><button class="btn btn--secondary btn--sm" data-tool="${esc(t.name)}">${ico('play', 'icon icon--sm')} Execute</button></td></tr>`).join('')}</tbody></table>`;
  }

  function renderApprovals() {
    const pending = state.data.approvals?.approvals || [];
    const audit = state.data.audit?.audit || [];
    if (state.data.approvals?.__error) return errorState('approval-list', state.data.approvals.__error);
    $('approval-count').textContent = `${pending.length} pending`;
    $('approval-list').innerHTML = pending.length ? pending.map((a) => `<div class="approval"><span><strong>${esc(a.action)}</strong><br/><span class="row__meta">${esc(a.id)} · ${esc(a.risk)} · ${esc(a.actor)}</span></span><span class="approval__actions"><button class="btn btn--danger btn--sm" data-resolve="${esc(a.id)}" data-action="${esc(a.action)}" data-decision="false">Deny</button><button class="btn btn--primary btn--sm" data-resolve="${esc(a.id)}" data-action="${esc(a.action)}" data-decision="true">Approve</button></span></div>`).join('') : '';
    if (!pending.length) empty('approval-list', 'All clear', 'No high-risk actions are waiting for a human decision.', 'approvals');
    $('approval-audit').innerHTML = audit.length ? `<div class="list">${audit.slice(0, 8).map((a) => `<div class="row"><span class="row__icon">${ico('check')}</span><span><span class="row__title">${esc(a.action)}</span><br/><span class="row__meta">${esc(a.decision)} · ${esc(a.reason || '')}</span></span><span>${badge(a.decision)}</span></div>`).join('')}</div>` : '<p class="hint">No decisions recorded yet.</p>';
  }

  // ---- runtime channel enrollment --------------------------------------
  // Builds a single form field from the catalog's field descriptor. Secret
  // inputs are always created blank — catalog never supplies (or echoes) a
  // value, and leaving a configured secret blank means "keep current value".
  function channelFieldHTML(f) {
    const name = String(f.name ?? '');
    const kind = String(f.kind ?? 'text');
    // Guided hints/defaults travel with the catalog; values never do.
    const helpText = f.hint || f.help;
    const help = helpText ? `<p class="hint">${esc(helpText)}</p>` : '';
    const required = f.required && !f.configured ? ' required' : '';
    const id = `cfg-${name.replace(/[^A-Za-z0-9_-]/g, '_')}`;
    const placeholder = f.placeholder || f.example || f.default || '';
    const saved = f.configured ? ' <small class="hint">(saved)</small>' : '';
    let control;
    if (kind === 'textarea') {
      control = `<textarea class="textarea" id="${esc(id)}" rows="3" data-field-name="${esc(name)}" data-field-kind="${esc(kind)}" placeholder="comma-separated values"${required}></textarea>`;
    } else if (kind === 'choice') {
      const options = Array.isArray(f.options) ? f.options : [];
      const opts = options.map((o) => {
        const val = (o && typeof o === 'object') ? (o.value ?? o.name ?? o.label) : o;
        const lbl = (o && typeof o === 'object') ? (o.label ?? o.name ?? o.value) : o;
        const sel = String(val) === String(f.default) ? ' selected' : '';
        return `<option value="${esc(val)}"${sel}>${esc(lbl)}</option>`;
      }).join('');
      control = `<select class="select" id="${esc(id)}" data-field-name="${esc(name)}" data-field-kind="${esc(kind)}"${required}>${opts}</select>`;
    } else if (kind === 'secret') {
      const ph = f.configured ? ' placeholder="leave blank to keep current value"' : '';
      control = `<input class="input" id="${esc(id)}" type="password" autocomplete="off" data-field-name="${esc(name)}" data-field-kind="${esc(kind)}"${ph}${required} />`;
    } else if (kind === 'number') {
      control = `<input class="input" id="${esc(id)}" type="number" autocomplete="off" data-field-name="${esc(name)}" data-field-kind="${esc(kind)}"${placeholder ? ` placeholder="${esc(placeholder)}"` : ''}${required} />`;
    } else {
      control = `<input class="input" id="${esc(id)}" type="text" autocomplete="off" data-field-name="${esc(name)}" data-field-kind="${esc(kind)}"${placeholder ? ` placeholder="${esc(placeholder)}"` : ''}${required} />`;
    }
    return `<div class="field"><label class="label" for="${esc(id)}">${esc(f.label || name)}${saved}</label>${control}${help}</div>`;
  }

  // Warning/policy notes plus every field, shown inside the existing modal.
  function channelFieldsHTML(info) {
    const parts = [];
    if (/community|unofficial/i.test(String(info.description || ''))) {
      parts.push('<p class="error-text">Unofficial bridge — use at your own risk.</p>');
    }
    if (info.persistence_note) parts.push(`<p class="error-text">${esc(info.persistence_note)}</p>`);
    const fields = Array.isArray(info.fields) ? info.fields : [];
    fields.forEach((f) => parts.push(channelFieldHTML(f)));
    const policy = info.policy && typeof info.policy === 'object' ? JSON.stringify(info.policy) : String(info.policy || '');
    if (info.mode_note) parts.push(`<p class="error-text">${esc(info.mode_note)}</p>`);
    if (fields.some((f) => f.name === 'allowlist') || /allowlist/i.test(policy)) {
      parts.push('<p class="hint">Empty allowlist denies all inbound messages.</p>');
    }
    // The guided wizard is preferred; this remains its honest fallback.
    if (info.docs_url && /^https:\/\//i.test(String(info.docs_url))) {
      parts.push(`<p class="hint"><a href="${esc(info.docs_url)}" target="_blank" rel="noopener noreferrer">Open the official setup guide</a></p>`);
    }
    return parts.join('');
  }

  // Collect the enable() request body: non-empty values keyed by env name,
  // allowlist split on commas/newlines, and mode from the select.
  function collectChannelForm() {
    const body = { values: {} };
    $('modal-fields').querySelectorAll('[data-field-name]').forEach((input) => {
      const name = input.dataset.fieldName;
      const raw = String(input.value || '').trim();
      if (name === 'allowlist') {
        body.allowlist = raw.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
      } else if (name === 'mode') {
        if (raw) body.mode = raw;
      } else if (raw) {
        body.values[name] = raw; // blank secrets are omitted, never echoed
      }
    });
    return body;
  }

  async function openChannelConfigure(channel) {
    try {
      if (!state.catalog[channel]) state.catalog[channel] = await api(`/integrations/catalog/${encodeURIComponent(channel)}`);
      const info = state.catalog[channel] || {};
      const label = info.label || channel;
      openModal({
        title: `Configure ${label}`,
        text: info.description || '',
        runLabel: 'Enable',
        silent: true,
        fields: channelFieldsHTML(info),
        run: async (body) => {
          const result = await api(`/integrations/${encodeURIComponent(channel)}/enable`, { method: 'POST', body: JSON.stringify(body) });
          delete state.catalog[channel];
          toast(`${label} enabled`, 'success');
          return result;
        },
      });
    } catch (err) { toast(err.message, 'danger'); }
  }

  function openChannelDisable(channel, label) {
    openModal({
      title: `Disable ${label}?`,
      text: `Inbound messages from ${label} will no longer be admitted. Runtime credentials for this channel are cleared.`,
      confirm: true,
      silent: true,
      run: async () => {
        const result = await api(`/integrations/${encodeURIComponent(channel)}/disable`, { method: 'POST' });
        delete state.catalog[channel];
        toast(`${label} disabled`, 'warn');
        return result;
      },
    });
  }

  function renderIntegrations() {
    const items = state.data.integrations?.integrations || [];
    if (state.data.integrations?.__error) return errorState('integration-list', state.data.integrations.__error);
    const count = (status) => items.filter((i) => i.status === status).length;
    $('integration-count').textContent = items.length
      ? `${count('configured')} configured · ${count('blocked')} blocked · ${items.length} total`
      : '—';
    if (!items.length) return empty('integration-list', 'No integrations registered', 'Channels appear here once configured.', 'integrations');
    // Configured first, then errors (visible), then blocked channels; names ascending.
    const rank = (i) => i.status === 'configured' ? 0 : i.status === 'error' ? 1 : 2;
    const ordered = items.slice().sort((a, b) => rank(a) - rank(b) || String(a.name).localeCompare(String(b.name)));
    $('integration-list').innerHTML = `<table class="table"><thead><tr><th>Integration</th><th>Kind</th><th>Status</th><th>Capabilities</th><th>Configuration</th><th>Actions</th></tr></thead><tbody>${ordered.map((i) => {
      let caps = i.capabilities;
      if (typeof caps === 'string') { try { caps = JSON.parse(caps); } catch { caps = []; } }
      caps = Array.isArray(caps) ? caps : [];
      const channel = String(i.name ?? '');
      const configured = i.status === 'configured';
      // Only registered channel rows have an enrollment catalog entry.
      const isChannel = i.kind === 'messaging' && !channel.startsWith('channel-module:');
      const configure = isChannel ? `<button class="btn btn--secondary btn--sm" data-configure="${esc(channel)}">${ico('settings', 'icon icon--sm')} Configure</button>` : '';
      const disable = configured && isChannel ? `<button class="btn btn--danger btn--sm" data-disable="${esc(channel)}">${ico('x', 'icon icon--sm')} Disable</button>` : '';
      return `<tr><td><strong>${esc(i.name)}</strong></td><td>${esc(i.kind)}</td><td>${badge(i.status)}</td><td class="row__meta">${esc(caps.join(', ') || '—')}</td><td class="row__meta">${esc(i.config_ref || '—')}</td><td class="row__actions">${configure}${disable}</td></tr>`;
    }).join('')}</tbody></table>`;
  }

  // Merge the SSE backlog with the last /events poll, dedupe by event id and
  // keep the visible list bounded so a long-lived stream cannot grow the DOM.
  function visibleEvents() {
    const snapshot = state.data.events?.events || [];
    const byId = new Map();
    [...state.eventItems, ...snapshot].forEach((event) => {
      const key = event?.id || `${event?.topic}:${event?.created_at}:${event?.source || ''}`;
      if (!byId.has(key)) byId.set(key, event);
    });
    return [...byId.values()]
      .sort((a, b) => String(b?.created_at || '').localeCompare(String(a?.created_at || '')))
      .slice(0, EVENT_STREAM_LIMIT);
  }

  function renderEvents() {
    const events = visibleEvents();
    if (state.data.events?.__error && !events.length) return errorState('event-list', state.data.events.__error);
    const mode = state.eventStreamLive ? 'live' : (state.eventStreamUnsupported ? 'polling · stream unavailable' : 'polling');
    $('event-count').textContent = `${events.length} recent · ${mode}`;
    if (!events.length) return empty('event-list', 'No events yet', 'Task transitions, approvals and agent activity appear here as they are emitted.', 'events');
    $('event-list').innerHTML = `<div class="list">${events.map((e) => `<div class="row"><span class="row__icon">${ico('events')}</span><span><span class="row__title">${esc(e.topic)}</span><br/><span class="row__meta">${esc(e.created_at || '')}</span></span><span class="row__meta">${esc(short(JSON.stringify(e.payload || {}), 70))}</span></div>`).join('')}</div>`;
  }

  function renderAll() {
    renderTasks(); renderAgents(); renderAgentDelegation(); renderSkills(); renderMemory();
    renderModels(); renderTools(); renderApprovals(); renderIntegrations();
    renderSchedules(); renderChannelSend(); renderEvents();
  }

  function setView(name) {
    if (!VIEWS[name]) return;
    state.view = name;
    document.querySelectorAll('.nav__item').forEach((b) => {
      const active = b.dataset.view === name;
      b.classList.toggle('is-active', active);
      if (active) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
    });
    document.querySelectorAll('.view').forEach((v) => { v.hidden = v.id !== `view-${name}`; });
    $('view-title').textContent = VIEWS[name][0];
    $('view-eyebrow').textContent = VIEWS[name][1];
    document.title = `${VIEWS[name][0]} — LEO`;
    // The capsule folds secondary sections into "More". Keep it lit while one
    // of those sections is active, and close the popover after a selection.
    const moreBtn = $('nav-more-btn');
    const moreMenu = $('nav-more-menu');
    if (moreBtn) moreBtn.classList.toggle('is-active', !!document.querySelector('.nav__more-menu .nav__item.is-active'));
    if (moreMenu && !moreMenu.hidden) { moreMenu.hidden = true; if (moreBtn) moreBtn.setAttribute('aria-expanded', 'false'); }
  }

  function openModal({ title, text, value = '{}', run, confirm = false, fields = null, runLabel = null, silent = false, showResult = false }) {
    state.modal = run; state.modalConfirm = confirm;
    state.modalForm = typeof fields === 'string'; state.modalSilent = silent;
    state.modalShowResult = showResult;
    $('modal-title').textContent = title;
    $('modal-text').textContent = text;
    $('modal-args').value = value;
    $('modal-error').hidden = true;
    $('modal-error').textContent = '';
    $('modal-output').hidden = true;
    $('modal-output-text').textContent = '';
    const fieldsEl = $('modal-fields');
    if (state.modalForm) {
      fieldsEl.innerHTML = fields;
      fieldsEl.style.display = 'grid';
      fieldsEl.style.gap = 'var(--space-4)';
    } else {
      fieldsEl.innerHTML = '';
      fieldsEl.style.display = 'none';
    }
    $('modal-args-label').hidden = state.modalForm;
    $('modal-args').hidden = state.modalForm;
    $('modal-args-field').hidden = confirm;
    $('modal-run-label').textContent = runLabel || (confirm ? 'Confirm' : 'Run');
    $('modal-backdrop').hidden = false;
    const first = state.modalForm ? fieldsEl.querySelector('input, select, textarea') : (confirm ? $('modal-cancel') : $('modal-args'));
    if (first) first.focus();
  }
  function closeModal() { $('modal-backdrop').hidden = true; state.modal = null; }

  // ---- events -----------------------------------------------------------
  $('nav').addEventListener('click', (e) => { const b = e.target.closest('[data-view]'); if (b) setView(b.dataset.view); });

  // Capsule "More" popover. Items live inside #nav, so the delegate above
  // performs the view switch; this only owns open/close and focus affordances.
  (function initNavMore() {
    const btn = $('nav-more-btn');
    const menu = $('nav-more-menu');
    if (!btn || !menu) return;
    const setOpen = (open) => {
      menu.hidden = !open;
      btn.setAttribute('aria-expanded', String(open));
      if (open) menu.querySelector('button, a')?.focus();
    };
    const close = () => setOpen(false);
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      setOpen(menu.hidden);
    });
    menu.addEventListener('click', (e) => { if (e.target.closest('[data-view], a')) close(); });
    // Tabbing past the last item returns focus to the page; dismiss the popover.
    menu.addEventListener('focusout', (e) => {
      if (!menu.contains(e.relatedTarget) && e.relatedTarget !== btn) close();
    });
    document.addEventListener('click', (e) => {
      if (!menu.hidden && !e.target.closest('.nav__more')) close();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !menu.hidden) { close(); btn.focus(); }
    });
  })();
  $('menu-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const open = $('menu').hidden;
    $('menu').hidden = !open;
    $('menu-btn').setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('click', (e) => {
    if (!$('menu').hidden && !e.target.closest('.menu-wrap')) { $('menu').hidden = true; $('menu-btn').setAttribute('aria-expanded', 'false'); }
    const retry = e.target.closest('[data-retry]'); if (retry) loadAll();
    const cancel = e.target.closest('[data-cancel]');
    if (cancel) openModal({
      title: 'Cancel this task?',
      text: `Task ${cancel.dataset.cancel} is still active. Cancellation is recorded and workers observe it before their next step; it cannot be undone.`,
      confirm: true,
      run: () => api(`/tasks/${encodeURIComponent(cancel.dataset.cancel)}/cancel`, { method: 'POST' })
        .then((result) => { toast('Cancellation requested', 'warn'); loadAll(); return result; }),
    });
    const runTask = e.target.closest('[data-run-task]');
    if (runTask && taskRunEnabled()) openModal({
      title: 'Run this objective?',
      text: `Task ${runTask.dataset.runTask} will be started by the backend runner.`,
      confirm: true,
      showResult: true,
      run: () => api(TASK_RUN_ROUTE(runTask.dataset.runTask), { method: 'POST' }),
    });
    const delSchedule = e.target.closest('[data-delete-schedule]');
    if (delSchedule && hasCapability('schedule')) {
      const key = delSchedule.dataset.deleteSchedule;
      openModal({
        title: 'Delete this schedule?',
        text: `Schedule ${key} will stop producing new objectives. Past tasks are not removed.`,
        confirm: true,
        showResult: true,
        run: async () => {
          const result = await api(`/schedules/${encodeURIComponent(key)}`, { method: 'DELETE' });
          return { ...result, success: result?.deleted === key || result?.success === true };
        },
      });
    }
    const resolve = e.target.closest('[data-resolve]');
    if (resolve) api(`/approvals/${encodeURIComponent(resolve.dataset.resolve)}/resolve`, { method: 'POST', body: JSON.stringify({ approved: resolve.dataset.decision === 'true', resolver: 'dashboard', confirm_action: resolve.dataset.action || '' }) }).then(() => { toast('Decision recorded', 'success'); loadAll(); }).catch((err) => toast(err.message, 'danger'));
    const skill = e.target.closest('[data-skill]');
    if (skill) openModal({ title: `Run ${skill.dataset.skill}`, text: 'Arguments are passed to the locally registered handler. No external script is executed.', run: (args) => api(`/skills/${encodeURIComponent(skill.dataset.skill)}/run`, { method: 'POST', body: JSON.stringify({ arguments: args }) }) });
    const tool = e.target.closest('[data-tool]');
    if (tool) openModal({ title: `Execute ${tool.dataset.tool}`, text: 'Schema, permissions and audit rules still apply. A high-risk call will request approval instead of running.', run: (args) => api(`/tools/${encodeURIComponent(tool.dataset.tool)}/execute`, { method: 'POST', body: JSON.stringify({ arguments: args }) }) });
    const cfg = e.target.closest('[data-configure]');
    if (cfg) return openChannelConfigure(cfg.dataset.configure);
    const dis = e.target.closest('[data-disable]');
    if (dis) {
      const label = dis.closest('tr')?.querySelector('strong')?.textContent || dis.dataset.disable;
      return openChannelDisable(dis.dataset.disable, label);
    }
  });
  $('menu').addEventListener('click', async (e) => {
    const item = e.target.closest('[data-action], [data-view]');
    if (!item) return;
    $('menu').hidden = true; $('menu-btn').setAttribute('aria-expanded', 'false');
    if (item.dataset.view) return setView(item.dataset.view);
    if (item.dataset.action === 'refresh') { loadAll(); refreshShellHealth(); return; }
    if (item.dataset.action === 'compact') { state.compact = !state.compact; sessionStorage.setItem('leo_compact', state.compact ? '1' : '0'); $('app').classList.toggle('app--compact', state.compact); $('compact-toggle').checked = state.compact; }
    if (item.dataset.action === 'copy-health') copyHealth();
  });
  $('refresh-btn').addEventListener('click', () => { loadAll(); refreshShellHealth(); });
  $('copy-health').addEventListener('click', copyHealth);
  async function copyHealth() {
    try { await navigator.clipboard.writeText(JSON.stringify(state.data.health || {}, null, 2)); toast('Runtime status copied', 'success'); }
    catch { toast('Clipboard unavailable in this context', 'warn'); }
  }

  $('task-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try { await api('/tasks', { method: 'POST', body: JSON.stringify({ objective: $('objective').value.trim() }) }); $('objective').value = ''; toast('Objective queued — no manual run route in this backend', 'info'); loadAll(); }
    catch (err) { toast(err.message, 'danger'); }
  });
  $('delegate-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!hasCapability('agent_delegate')) return;
    const objective = $('delegate-objective').value.trim();
    if (!objective) return;
    const body = { objective };
    const agent = $('delegate-agent').value;
    if (agent) body.agent_id = agent;
    const button = $('delegate-submit'); button.disabled = true; button.setAttribute('aria-busy', 'true');
    try {
      const result = await api('/agents/delegate', { method: 'POST', body: JSON.stringify(body) });
      showResultIn('delegate-output', result);
      if (result?.success === true) {
        toast(`Delegated — task ${result.task_id || 'id not returned'}`, 'success');
        $('delegate-objective').value = '';
      } else toast(result?.error || 'Delegation did not report success', 'danger');
      loadAll();
    } catch (err) {
      showResultIn('delegate-output', { error: err.message });
      toast(err.message, 'danger');
    } finally {
      button.disabled = false; button.removeAttribute('aria-busy');
      renderAgentDelegation();
    }
  });
  $('schedule-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!hasCapability('schedule')) return;
    const name = $('schedule-name').value.trim();
    const seconds = Number($('schedule-seconds').value);
    const objective = $('schedule-objective').value.trim();
    if (!name || !objective || !Number.isFinite(seconds) || seconds <= 0) { toast('Provide a name, objective and positive interval.', 'warn'); return; }
    const button = $('schedule-submit'); button.disabled = true; button.setAttribute('aria-busy', 'true');
    try {
      const result = await api('/schedules', { method: 'POST', body: JSON.stringify({ name, seconds, objective }) });
      showResultIn('schedule-output', result);
      if (result?.name === name || result?.id) { toast(`Schedule registered: ${name}`, 'success'); $('schedule-objective').value = ''; }
      else toast('Backend did not return a schedule', 'danger');
      loadAll();
    } catch (err) {
      showResultIn('schedule-output', { error: err.message });
      toast(err.message, 'danger');
    } finally {
      button.disabled = false; button.removeAttribute('aria-busy');
    }
  });
  $('send-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!hasCapability('channel_send')) return;
    const channel = $('send-channel').value;
    const target = $('send-target').value.trim();
    const text = $('send-text').value.trim();
    if (!channel || !target || !text) return;
    const button = $('send-submit'); button.disabled = true; button.setAttribute('aria-busy', 'true');
    try {
      const result = await api(`/integrations/${encodeURIComponent(channel)}/send`, {
        method: 'POST', body: JSON.stringify({ target, text }),
      });
      showResultIn('send-output', result);
      // Provider acceptance is not delivery: require both flags from backend.
      const delivered = result?.delivered === true && /^(delivered|success|sent|ok)$/i.test(String(result?.status || ''));
      if (delivered) toast(`Delivered via ${channel}`, 'success');
      else toast(`Not delivered via ${channel} (${result?.status || 'unknown'})${result?.error ? ': ' + result.error : ''}`, 'danger');
    } catch (err) {
      showResultIn('send-output', { error: err.message });
      toast(err.message, 'danger');
    } finally {
      button.disabled = false; button.removeAttribute('aria-busy');
    }
  });
  $('memory-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try { await api('/memory', { method: 'POST', body: JSON.stringify({ content: $('mem-content').value.trim(), category: $('mem-category').value }) }); $('mem-content').value = ''; state.memoryQuery = ''; $('mem-query').value = ''; toast('Memory stored', 'success'); loadAll(); }
    catch (err) { toast(err.message, 'danger'); }
  });
  $('memory-search-form').addEventListener('submit', async (e) => {
    e.preventDefault(); const q = $('mem-query').value.trim(); state.memoryQuery = q;
    if (!q) return loadAll();
    try { state.data.memorySearch = await api(`/memory/search?q=${encodeURIComponent(q)}`); renderMemory(); }
    catch (err) { toast(err.message, 'danger'); }
  });
  $('skill-search-btn').addEventListener('click', renderSkills);
  $('skill-search').addEventListener('input', renderSkills);
  $('tool-search').addEventListener('input', renderTools);
  document.querySelectorAll('[data-model-tab]').forEach((tab) => tab.addEventListener('click', () => {
    document.querySelectorAll('[data-model-tab]').forEach((t) => { t.classList.toggle('is-active', t === tab); t.setAttribute('aria-selected', String(t === tab)); });
    const usage = tab.dataset.modelTab === 'usage';
    $('models-usage').hidden = !usage; $('models-providers').hidden = usage;
  }));
  document.querySelector('.tabs').addEventListener('keydown', (e) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
    const tabs = [...document.querySelectorAll('[data-model-tab]')];
    const next = (tabs.indexOf(document.activeElement) + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
    tabs[next].focus(); tabs[next].click();
  });

  $('save-token').addEventListener('click', async () => {
    const value = $('api-token').value.trim();
    if (!value) {
      state.token = ''; sessionStorage.removeItem('leo_api_token');
      $('api-token').value = ''; toast('Token cleared', 'success'); loadAll(); return;
    }
    try {
      // Validate the candidate against a real authenticated endpoint before
      // storing it: a typo must not silently lock every poll into a 401.
      await api('/status', { headers: { Authorization: `Bearer ${value}` } });
      state.token = value; sessionStorage.setItem('leo_api_token', value);
      $('api-token').value = ''; toast('Token validated and saved for this session', 'success'); loadAll();
    } catch (err) { toast(`Token rejected: ${err.message}`, 'danger'); }
  });
  $('compact-toggle').addEventListener('change', (e) => {
    state.compact = e.target.checked; sessionStorage.setItem('leo_compact', state.compact ? '1' : '0');
    $('app').classList.toggle('app--compact', state.compact);
  });
  $('poll-range').addEventListener('input', (e) => { state.pollMs = Number(e.target.value) * 1000; $('poll-value').textContent = e.target.value; schedulePoll(); });

  $('modal-close').addEventListener('click', closeModal);
  $('modal-cancel').addEventListener('click', closeModal);
  $('modal-backdrop').addEventListener('click', (e) => { if (e.target === $('modal-backdrop')) closeModal(); });
  $('modal-backdrop').addEventListener('keydown', (e) => {
    if (e.key === 'Escape') return closeModal();
    if (e.key !== 'Tab') return;
    const focusables = [...$('modal-backdrop').querySelectorAll('button, input, select, textarea')].filter((el) => !el.hidden);
    const first = focusables[0], last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
  $('modal-run').addEventListener('click', async () => {
    if (!state.modal) return;
    let args;
    if (state.modalForm) {
      args = collectChannelForm();
    } else {
      try { args = JSON.parse($('modal-args').value || '{}'); } catch { $('modal-error').textContent = 'Arguments must be valid JSON.'; $('modal-error').hidden = false; return; }
    }
    const button = $('modal-run'); button.disabled = true; button.setAttribute('aria-busy', 'true');
    try {
      const result = await state.modal(args);
      const payload = unwrapResult(result);
      if (state.modalShowResult) {
        // Confirmations that start real work keep the dialog open so their
        // backend payload (success and errors) can be inspected.
        showModalOutput(payload);
        if (payload && payload.success === true) toast('Run started', 'success');
        else toast(payload?.error || 'Backend did not report success', 'danger');
      } else if (state.modalConfirm || state.modalSilent) {
        closeModal();
        if (result?.tool_results) result.tool_results.forEach((t) => toast(t.name || 'tool', 'info'));
      } else {
        // Run/Execute: render the backend's real result and only claim success
        // when the backend itself reported success:true. Keep the dialog open
        // so the output can be inspected (and a failure retried).
        showModalOutput(payload);
        if (payload && payload.success === true) toast('Completed successfully', 'success');
        else toast(payload?.error || 'Backend did not report success', 'danger');
      }
      loadAll();
    } catch (err) { $('modal-error').textContent = err.message; $('modal-error').hidden = false; }
    button.disabled = false; button.removeAttribute('aria-busy');
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('menu').hidden) { $('menu').hidden = true; $('menu-btn').setAttribute('aria-expanded', 'false'); }
  });

  // ---- live event stream -------------------------------------------------
  // Route contract (backend): POST /events/stream-token issues a short-lived
  // single-use token; GET /events/stream?token=... is text/event-stream with
  // `data: <JSON Event>` frames. EventSource cannot set a Bearer header, so the
  // token is fetched via the authenticated api() helper for every (re)connect.
  async function streamUrl() {
    const grant = await api(EVENTS_STREAM_TOKEN_ROUTE, { method: 'POST' });
    return `${EVENTS_STREAM_ROUTE}?token=${encodeURIComponent(grant.token)}`;
  }
  function ingestEvent(event) {
    if (!event || typeof event !== 'object' || !event.topic) return;
    state.eventItems.unshift(event);
    if (state.eventItems.length > EVENT_STREAM_LIMIT) state.eventItems.length = EVENT_STREAM_LIMIT;
    renderEvents();
  }

  function stopEventStream() {
    if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
    if (state.eventRetryTimer) { clearTimeout(state.eventRetryTimer); state.eventRetryTimer = null; }
    state.eventStreamLive = false;
  }

  function scheduleEventRetry() {
    if (state.eventRetryTimer || document.hidden || state.eventStreamUnsupported) return;
    const delay = state.eventBackoffMs;
    state.eventBackoffMs = Math.min(EVENT_RETRY_MAX_MS, Math.max(EVENT_RETRY_MIN_MS, delay * 2));
    state.eventRetryTimer = setTimeout(() => {
      state.eventRetryTimer = null;
      connectEventStream();
    }, delay);
  }

  async function connectEventStream() {
    if (state.eventSource || document.hidden || state.eventStreamUnsupported) return;
    let url;
    try { url = await streamUrl(); }
    catch {
      // Without a valid stream token the feed is unavailable; polling
      // remains the authoritative data path.
      state.eventStreamUnsupported = true; renderEvents(); return;
    }
    if (state.eventSource || document.hidden || state.eventStreamUnsupported) return;
    let source;
    try { source = new EventSource(url); }
    catch { state.eventStreamUnsupported = true; renderEvents(); return; }
    state.eventSource = source;
    const onEvent = (message) => {
      try { ingestEvent(JSON.parse(message.data)); } catch { /* ignore malformed frames */ }
    };
    source.addEventListener('message', onEvent);
    source.addEventListener('event', onEvent);
    source.addEventListener('open', () => {
      state.eventStreamLive = true;
      state.eventBackoffMs = EVENT_RETRY_MIN_MS;
      renderEvents();
    });
    source.addEventListener('error', () => {
      source.close();
      if (state.eventSource === source) state.eventSource = null;
      state.eventStreamLive = false;
      // A missing/blocked route leaves polling as the working data path;
      // transient drops reconnect with exponential backoff.
      scheduleEventRetry();
      renderEvents();
    });
  }

  async function probeEventStream() {
    if (state.eventStreamUnsupported) return;
    // Connect through the token route; /events polling remains the fallback.
    await connectEventStream();
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopEventStream();
      state.eventBackoffMs = EVENT_RETRY_MIN_MS;
    } else {
      probeEventStream();  // reconnect after idle; no catch-up flood
      loadAll();
      refreshShellHealth();
    }
  });

  // ---- Electron shell bridge --------------------------------------------
  // The preload exposes only invoke(channel) and onEvent(listener). In a
  // plain browser both are absent: HTTP /health polling stays authoritative.
  function applyShellEvent(payload) {
    if (!payload || typeof payload !== 'object') return;
    const status = $('shell-status');
    if (!status) return;
    const type = String(payload.type || 'shell');
    const bad = payload.healthy === false || ['backend-error', 'backend-exit', 'action-failed'].includes(type);
    status.hidden = false;
    status.className = `status ${bad ? 'status--danger' : payload.healthy === true ? 'status--live' : 'status--warn'}`;
    const detail = payload.message
      || [payload.mode, payload.healthy === true ? 'healthy' : payload.healthy === false ? 'degraded' : '', payload.version ? `v${payload.version}` : '']
        .filter(Boolean).join(' · ');
    $('shell-status-text').textContent = `${type}${detail ? ` · ${detail}` : ''}`;
    if (bad) toast(payload.message || `${type} reported by the shell`, type === 'action-failed' ? 'warn' : 'danger');
    if (type === 'startup-state' || type === 'health') {
      state.shellHealth = payload;
      renderHealth();
    }
  }

  async function refreshShellHealth() {
    const bridge = window.leo;
    if (!bridge || typeof bridge.invoke !== 'function') return;
    try {
      const health = await bridge.invoke('leo:health');
      if (health && typeof health === 'object') { state.shellHealth = health; renderHealth(); }
    } catch { /* shell health is supplementary; HTTP health still works */ }
    try { state.shellStatus = await bridge.invoke('leo:api-status'); }
    catch { /* supplementary */ }
  }

  function initShellBridge() {
    const bridge = window.leo;
    if (!bridge || typeof bridge.invoke !== 'function') {
      const status = $('shell-status');
      if (status) status.hidden = true;
      return;
    }
    if (typeof bridge.onEvent === 'function') {
      const unsubscribe = bridge.onEvent(applyShellEvent);
      window.addEventListener('beforeunload', () => {
        if (typeof unsubscribe === 'function') unsubscribe();
      }, { once: true });
    }
    refreshShellHealth();
  }

  // ---- guided channel setup ---------------------------------------------
  // The wizard is a separate module. If it cannot load, the original field
  // form above remains fully usable, so setup never depends on this asset.
  document.addEventListener('leo:open-raw-channel-form', (e) => {
    if (e.detail?.channel) openChannelConfigure(e.detail.channel);
  });
  // Empty-state call to action in the Send panel: open the guided setup, or
  // fall back to the legacy form when the module is unavailable.
  $('send-connect')?.addEventListener('click', () => {
    if (window.LEO_ONBOARDING) window.LEO_ONBOARDING.open('whatsapp', $('send-connect'));
    else document.dispatchEvent(new CustomEvent('leo:open-raw-channel-form', { detail: { channel: 'whatsapp' } }));
  });
  function loadOnboarding() {
    if (window.LEO_ONBOARDING) return window.LEO_ONBOARDING.mount();
    if (document.querySelector('script[data-leo-onboarding]')) return;
    const script = document.createElement('script');
    script.src = '/assets/ui/onboarding.js';
    script.dataset.leoOnboarding = '1';
    script.addEventListener('load', () => { if (window.LEO_ONBOARDING) window.LEO_ONBOARDING.mount(); });
    document.head.appendChild(script);
  }

  // ---- boot -------------------------------------------------------------
  if (state.compact) $('app').classList.add('app--compact');
  $('compact-toggle').checked = state.compact;
  $('auth-badge').textContent = state.token ? 'token set' : 'no token';
  Object.keys(ENDPOINTS).forEach((key) => loading(key === 'tasks' ? 'task-list' : key === 'agents' ? 'agent-list' : key === 'skills' ? 'skill-list' : key === 'memory' ? 'memory-list' : key === 'usage' ? 'usage-list' : key === 'tools' ? 'tool-list' : key === 'approvals' ? 'approval-list' : key === 'integrations' ? 'integration-list' : key === 'events' ? 'event-list' : null, 2));
  loadOnboarding();
  loadAll().then(() => { schedulePoll(); probeEventStream(); });
  initShellBridge();
})();
