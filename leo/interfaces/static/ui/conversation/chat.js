/* LEO conversation module — the LEO chat surface.
 *
 * Layout:
 *   - ChatSessionList  -> sessions sidebar (new chat, search, select, delete)
 *   - the chat pane    -> thread with markdown streaming, tool cards and
 *                         inline approvals, composer with image paste/drop
 *   - ChatSidebar     -> model card + live badge + banners + turn status
 *
 * The runtime is web-native (SSE /chat/stream); stream events drive the
 * thread, tool cards and approvals.
 */

(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const SESSION_LIMIT = 50;
  const SESSION_REFRESH_MS = 20000;
  const RENDER_THROTTLE_MS = 90;
  const HISTORY_RESULT_LIMIT = 4000;
  const REASONING_RESULT_LIMIT = 8000;
  const SLASH_PALETTE_LIMIT = 40;
  // Reasoning is provider-supplied data, never a substitute for an answer.
  // A disclosure panel is created only by a real thinking.delta. When no delta
  // arrives, the truthful verdict is one short line, never invented reasoning.
  const NO_REASONING_NOTE = 'No reasoning streamed';
  const NO_REASONING_HIDDEN_NOTE = 'Reasoning display is off';

  const state = {
    activeId: localStorage.getItem('leo_chat_conversation') || null,
    conversations: [],
    search: '',
    attachments: [],
    attachmentErrors: [],
    activeRun: null,
    abortController: null,
    lastSeq: -1,
    running: false,
    stream: null,          // live assistant node { contentEl, buffer, sealed }
    // Provider-supplied reasoning lives in its own panel per assistant block.
    // No panel exists until a real delta arrives; these lists only track
    // panels that actually exist, so a later turn cannot reattach an old one.
    turnReasoningStreams: [],
    turnReasoningSeen: false,
    noReasoningNote: false,
    // The backend reports whether summaries were intentionally hidden. When
    // they were, the terminal one-liner says "display is off", not "none".
    reasoningHidden: false,
    // Per-run latch: once a run reaches a terminal event (or is superseded),
    // its reasoning channel is closed forever. A late delta from that run must
    // never reopen a panel or create a fresh assistant message.
    reasoningClosedRuns: new Set(),
    turnNodes: [],  // assistant messages of this turn, for one-line bookkeeping
    turnTools: new Map(),  // tool_call_id -> { name, headStatus, resultNode }
    elapsedTimer: null,
    turnStart: 0,
    // approval_id -> { card, toolCallId, title, body }. Inline cards are
    // read-only blockers; only the Approvals view can decide them.
    pendingApprovals: new Map(),
    terminalStatus: null,   // last terminal run status; `finally` never clears it
    streamDropped: false,   // dropped/aborted stream => reload durable history
    // Live updates: the open thread follows the durable event feed so turns
    // produced by other clients (API, channels, another tab) appear without a
    // refresh. Incremental fetches resume after the last rendered row id.
    lastMessageId: 0,
    eventSource: null,
    eventRetryTimer: null,
    scrollPinned: true,
    panelHidden: sessionStorage.getItem('leo_chat_panel_hidden') === '1',
    // Slash-command palette: the backend's chat-surface catalog, fetched once
    // and filtered locally as the user types.
    commands: null,
    commandsLoading: false,
    paletteMatches: [],
    paletteIndex: 0,
  };

  /* ---- shared helpers from the dashboard core --------------------------- */
  const leo = () => window.LEO || {};
  const api = (path, options = {}) => (leo().api ? leo().api(path, options) : bareApi(path, options));
  const toast = (message, kind = 'info') => (leo().toast ? leo().toast(message, kind) : null);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

  async function bareApi(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const token = sessionStorage.getItem('leo_api_token') || '';
    if (token) headers.set('Authorization', `Bearer ${token}`);
    if (options.body && !headers.has('content-type')) headers.set('content-type', 'application/json');
    const response = await fetch(path, { ...options, headers });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
    return body;
  }

  const ico = (name, cls = 'icon') => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  // ico() is an HTML *string*: DOM append() would insert it as literal text.
  // Parse it into a real SVG element before adding it to any visible surface.
  function iconNode(name, cls = 'icon') {
    const host = document.createElement('span');
    host.innerHTML = ico(name, cls);
    return host.firstElementChild;
  }

  function timeAgo(value) {
    if (!value) return '—';
    const then = Date.parse(String(value).replace(' ', 'T'));
    if (Number.isNaN(then)) return String(value);
    const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
    if (seconds < 60) return 'just now';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.round(hours / 24);
    return days < 7 ? `${days}d ago` : new Date(then).toLocaleDateString();
  }

  function fmtDuration(ms) {
    const total = Math.max(0, Math.round(ms / 1000));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return m ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
  }

  function boundedText(value, limit = HISTORY_RESULT_LIMIT) {
    const text = String(value ?? '');
    return text.length > limit ? `${text.slice(0, limit)}\n… truncated (${text.length} chars)` : text;
  }

  /* ---- status / side panel ---------------------------------------------- */

  const STATUS_LABELS = {
    live: 'Ready', thinking: 'Thinking…', tool: 'Calling a tool…',
    responding: 'Responding…', waiting_approval: 'Approval required',
    cancelled: 'Cancelled', incomplete: 'Incomplete', failed: 'Failed',
  };
  const TERMINAL_STATUSES = ['completed', 'incomplete', 'cancelled', 'failed'];

  function setChatStatus(status, text) {
    const pill = $('chat-status');
    pill.className = `status status--${status || 'live'}`;
    pill.querySelector('.status__dot').style.background = 'currentColor';
    $('chat-status-text').textContent = text;
  }

  function setTurnState(kind, label) {
    const badge = $('chat-turn-state');
    badge.textContent = label;
    badge.className = `badge badge--${kind}`;
  }

  function setTurnStatus(text) { $('chat-turn-status').textContent = text; }

  // Only the backend's terminal run status decides how a turn is presented.
  // Never infer success from the stream closing or from an assistant delta.
  function applyTerminalStatus(status) {
    const terminal = TERMINAL_STATUSES.includes(status) ? status : 'failed';
    state.terminalStatus = terminal;
    if (terminal === 'completed') {
      setChatStatus('live', 'Ready');
      setTurnState('success', 'done');
      setTurnStatus('turn complete');
    } else if (terminal === 'incomplete') {
      setChatStatus('incomplete', STATUS_LABELS.incomplete);
      setTurnState('warn', 'incomplete');
      setTurnStatus('tool calls pending — turn not fully answered');
    } else if (terminal === 'cancelled') {
      setChatStatus('cancelled', 'Cancelled');
      setTurnState('', 'cancelled');
      setTurnStatus('cancelled');
    } else {
      setChatStatus('danger', 'Failed');
      setTurnState('danger', 'failed');
      setTurnStatus('failed');
    }
    // A turn that streamed no reasoning says so honestly. Cancellation and
    // failure already explain themselves; only successful/incomplete answers
    // get the extra one-line verdict.
    if (state.noReasoningNote && (terminal === 'completed' || terminal === 'incomplete')) {
      setTurnStatus(state.reasoningHidden ? NO_REASONING_HIDDEN_NOTE : NO_REASONING_NOTE);
    }
  }

  function startElapsed() {
    state.turnStart = Date.now();
    stopElapsed();
    $('chat-turn-elapsed').textContent = '0s';
    state.elapsedTimer = setInterval(() => {
      $('chat-turn-elapsed').textContent = fmtDuration(Date.now() - state.turnStart);
    }, 500);
  }
  function stopElapsed() {
    if (state.elapsedTimer) { clearInterval(state.elapsedTimer); state.elapsedTimer = null; }
  }

  function banner({ kind = 'warn', title, text, actions = [] }) {
    const host = $('chat-banners');
    const el = document.createElement('div');
    el.className = `chat-banner chat-banner--${kind}`;
    const body = document.createElement('div');
    body.className = 'chat-banner__body';
    const t = document.createElement('span');
    t.className = 'chat-banner__title';
    t.textContent = title;
    const p = document.createElement('span');
    p.className = 'chat-banner__text';
    p.textContent = text;
    body.append(t, p);
    actions.forEach(({ label, run }) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn--secondary btn--sm';
      btn.textContent = label;
      btn.addEventListener('click', () => { el.remove(); run(); });
      body.appendChild(btn);
    });
    el.append(iconNode(kind === 'danger' ? 'alert' : 'approvals', 'icon icon--sm'), body);
    host.appendChild(el);
    return el;
  }
  function clearBanners() { $('chat-banners').replaceChildren(); }

  function updateModelCard(health) {
    // `default_provider` is a catalog slot key; on a single-gateway install it
    // equals the model id, so it must not be shown as the provider name.
    const slot = health.default_provider || '—';
    const upstream = health.provider || '';
    const fromMap = health.providers?.[slot];
    const model = health.model
      || (typeof fromMap === 'string' ? fromMap : (fromMap?.model || ''))
      || slot;
    $('chat-model-name').textContent = upstream ? `${upstream} / ${model}` : model;
    const ok = health.status === 'ok';
    const badge = $('chat-live-badge');
    badge.className = `status ${ok ? 'status--live' : 'status--danger'}`;
    $('chat-live-text').textContent = ok ? 'live' : 'offline';
    $('chat-session-provider').textContent = upstream || '—';
    $('chat-session-tools').textContent = health.tools ?? '—';
    $('chat-session-skills').textContent = health.skills ?? '—';
    $('chat-session-agents').textContent = health.agents ?? '—';
  }

  /* ---- sessions sidebar (ChatSessionList port) ---------------------------- */

  function sessionTitle() {
    if (!state.activeId) { $('chat-title').textContent = 'New conversation'; return; }
    const row = state.conversations.find((c) => c.id === state.activeId);
    $('chat-title').textContent = row ? row.title : state.activeId;
  }

  async function refreshSessions() {
    const statusText = $('chat-sessions-status-text');
    try {
      const data = await api('/conversations');
      state.conversations = data.conversations || [];
      $('chat-sessions-status').className = 'status status--live';
      statusText.textContent = `${state.conversations.length} conversation${state.conversations.length === 1 ? '' : 's'}`;
      renderSessions();
      sessionTitle();
      if (state.activeId && !state.conversations.some((c) => c.id === state.activeId)) {
        // deleted elsewhere (or pre-module id): fall back gracefully
        if (!state.running) selectConversation(state.conversations[0]?.id || null, { silent: true });
      }
    } catch (error) {
      $('chat-sessions-status').className = 'status status--danger';
      statusText.textContent = 'load failed';
      renderSessionsError(error.message);
    }
  }

  function renderSessionsError(message) {
    const list = $('chat-session-list');
    list.innerHTML = `<div class="sess-empty">${esc(message)}<br/><button class="btn btn--secondary btn--sm" id="sess-retry">Retry</button></div>`;
    list.querySelector('#sess-retry').addEventListener('click', refreshSessions);
  }

  function renderSessions() {
    const list = $('chat-session-list');
    const query = state.search.trim().toLowerCase();
    const shown = state.conversations.filter((row) =>
      !query || `${row.title} ${row.preview} ${row.channel}`.toLowerCase().includes(query));
    list.replaceChildren();
    if (!shown.length) {
      const el = document.createElement('div');
      el.className = 'sess-empty';
      el.textContent = state.conversations.length
        ? 'No conversation matches that search.'
        : 'No conversations yet — your chats will appear here.';
      list.appendChild(el);
      return;
    }
    for (const row of shown) {
      const item = document.createElement('div');
      item.className = 'sess-row' + (row.id === state.activeId ? ' is-active' : '');
      item.setAttribute('role', 'listitem');

      const body = document.createElement('button');
      body.type = 'button';
      body.className = 'sess-row__body';
      body.style.cssText = 'border:0;background:transparent;padding:0;text-align:left;min-width:0';
      const title = document.createElement('span');
      title.className = 'sess-row__title';
      title.textContent = row.title;
      const meta = document.createElement('span');
      meta.className = 'sess-row__meta';
      meta.textContent = `${timeAgo(row.updated_at)} · ${row.message_count} msg · ${row.channel}`;
      body.append(title, meta);
      body.addEventListener('click', () => selectConversation(row.id));

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'sess-row__del';
      del.setAttribute('aria-label', `Delete conversation ${row.title}`);
      del.innerHTML = ico('trash', 'icon icon--sm');
      del.addEventListener('click', (e) => { e.stopPropagation(); confirmDelete(row, item); });

      item.append(body, del);
      list.appendChild(item);
    }
  }

  function confirmDelete(row, anchor) {
    document.querySelectorAll('.chat-confirm').forEach((n) => n.remove());
    const pop = document.createElement('div');
    pop.className = 'chat-confirm';
    pop.innerHTML = `
      <span class="chat-confirm__text">Delete this conversation? Its messages are removed permanently and the action is audit-logged.</span>
      <div class="chat-confirm__actions">
        <button type="button" class="btn btn--secondary btn--sm" data-x="keep">Keep</button>
        <button type="button" class="btn btn--danger btn--sm" data-x="del">Delete</button>
      </div>`;
    anchor.appendChild(pop);
    const close = () => pop.remove();
    pop.querySelector('[data-x="keep"]').addEventListener('click', close);
    pop.querySelector('[data-x="del"]').addEventListener('click', async () => {
      close();
      try {
        await api(`/conversations/${encodeURIComponent(row.id)}`, { method: 'DELETE' });
        if (row.id === state.activeId) {
          if (state.running) { toast('Conversation deleted. The running turn still streams here.', 'warn'); }
          else selectConversation(null, { silent: true });
        }
        toast('Conversation deleted', 'success');
        refreshSessions();
      } catch (error) {
        toast(error.message, 'danger');
      }
    });
    setTimeout(() => document.addEventListener('click', function onDoc(e) {
      if (!pop.contains(e.target)) { pop.remove(); document.removeEventListener('click', onDoc); }
    }), 0);
  }

  /* ---- thread rendering ---------------------------------------------------- */

  const threadEl = () => $('chat-thread');
  const scrollEl = () => $('chat-scroll');

  function maybeScroll(force = false) {
    const scroller = scrollEl();
    if (!scroller) return;
    if (force || state.scrollPinned) scroller.scrollTop = scroller.scrollHeight;
  }

  function attachScrollPinning() {
    const scroller = scrollEl();
    scroller.addEventListener('scroll', () => {
      const nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
      state.scrollPinned = nearBottom;
      const jump = $('chat-jump');
      if (jump) jump.hidden = nearBottom;
    });
  }

  function renderEmpty() {
    const thread = threadEl();
    thread.innerHTML = `
      <div class="chat-hero">
        <span class="chat-hero__mark">${ico('chat', 'icon icon--lg')}</span>
        <span class="chat-hero__title">A quiet session</span>
        <span class="chat-hero__text">One continuous, auditable history per conversation. LEO reaches for tools only when the task needs them, and high-risk actions always wait for your approval.</span>
        <div class="examples">
          <button class="badge badge--accent" data-example="Summarize the current system health in a short table">Summarize system health</button>
          <button class="badge badge--accent" data-example="Search memory for our reliability preferences and list what you find">Search memory</button>
          <button class="badge badge--accent" data-example="Create a task to review the deploy checklist">Create a review task</button>
        </div>
      </div>`;
    thread.querySelectorAll('[data-example]').forEach((chip) => {
      chip.addEventListener('click', () => {
        $('message').value = chip.dataset.example;
        autosize();
        $('message').focus();
      });
    });
  }

  function roleHead(role, meta) {
    const head = document.createElement('div');
    head.className = 'chat-msg__head';
    const tag = document.createElement('span');
    tag.className = 'chat-msg__role';
    tag.textContent = role === 'user' ? 'YOU' : 'LEO';
    const metaEl = document.createElement('span');
    metaEl.className = 'chat-msg__meta';
    metaEl.textContent = meta || '';
    head.append(tag, metaEl);
    return head;
  }

  function messageNode({ role, content, meta, time }) {
    const article = document.createElement('article');
    article.className = `chat-msg chat-msg--${role === 'user' ? 'user' : 'assistant'}`;
    article.append(roleHead(role, [time, meta].filter(Boolean).join(' · ')));
    const body = document.createElement('div');
    body.className = 'chat-msg__content';
    if (role === 'user') body.textContent = content;
    else window.LEOMarkdown.renderInto(body, content);
    article.appendChild(body);
    if (role === 'assistant') {
      const actions = document.createElement('div');
      actions.className = 'chat-msg__actions';
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.className = 'chat-msg__copy';
      copy.textContent = 'copy';
      copy.addEventListener('click', () => {
        navigator.clipboard.writeText(content).then(() => {
          copy.textContent = 'copied';
          setTimeout(() => { copy.textContent = 'copy'; }, 1600);
        }).catch(() => toast('Clipboard unavailable in this context', 'warn'));
      });
      actions.appendChild(copy);
      article.appendChild(actions);
    }
    return article;
  }

  function attachmentChips(metadata) {
    const host = document.createElement('div');
    host.className = 'chat-msg__attachments';
    const items = Array.isArray(metadata?.attachments) ? metadata.attachments : [];
    items.forEach((att) => {
      const chip = document.createElement('span');
      chip.className = 'attach-chip';
      if (String(att.mime || '').startsWith('image/')) {
        const img = document.createElement('img');
        img.alt = '';
        img.src = `/attachments/${encodeURIComponent(att.id)}/content`;
        chip.appendChild(img);
      } else {
        chip.insertAdjacentHTML('afterbegin', ico('file', 'icon icon--sm'));
      }
      const label = document.createElement('span');
      label.textContent = att.name || 'attachment';
      chip.appendChild(label);
      host.appendChild(chip);
    });
    return host;
  }

  // ONE summary row per tool by default: icon + name + risk + status + a short
  // result excerpt. Full arguments/result are behind an explicit disclosure.
  // Live-streamed and historic cards both go through toolNode, so they match.
  let toolCardSeq = 0;
  function toolExcerpt(value, limit = 120) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim();
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
  }
  function setToolResult(tool, value) {
    const text = typeof value === 'string' ? value : JSON.stringify(value ?? '', null, 2);
    tool.resultPre.textContent = boundedText(text || '');
    tool.excerpt.textContent = toolExcerpt(text || '');
  }

  function toolNode({ name, risk, arguments: args, status = 'running', toolCallId = '' }) {
    const toolName = name || 'tool';
    const card = document.createElement('div');
    card.className = `chat-tool chat-tool--${status === 'running' ? 'running' : 'idle'}`;
    if (toolCallId) card.dataset.toolCallId = toolCallId;
    // A real <button> gives Enter/Space activation and keyboard focus for free;
    // aria-expanded/aria-label keep the collapsed state legible to screen readers.
    const head = document.createElement('button');
    head.type = 'button';
    head.className = 'chat-tool__head chat-tool__toggle';
    const nameEl = document.createElement('span');
    nameEl.className = 'chat-tool__name';
    nameEl.textContent = toolName;
    const riskEl = document.createElement('span');
    riskEl.className = 'chat-tool__risk';
    riskEl.textContent = risk || '';
    const statusEl = document.createElement('span');
    statusEl.className = `status status--${status === 'running' ? 'tool' : status === 'failed' ? 'danger' : 'success'}`;
    statusEl.innerHTML = '<span class="status__dot"></span>';
    const statusText = document.createElement('span');
    statusText.textContent = status === 'running' ? 'running' : status === 'failed' ? 'failed' : 'done';
    statusEl.appendChild(statusText);
    const excerpt = document.createElement('span');
    excerpt.className = 'chat-tool__excerpt';
    excerpt.textContent = '';
    const body = document.createElement('div');
    body.className = 'chat-tool__body';
    body.id = `chat-tool-body-${++toolCardSeq}`;
    body.hidden = true;
    head.id = `chat-tool-toggle-${toolCardSeq}`;
    head.setAttribute('aria-controls', body.id);
    head.setAttribute('aria-expanded', 'false');
    head.setAttribute('aria-label', `Show details for ${toolName}`);
    head.append(iconNode('tools', 'icon icon--sm'), nameEl, riskEl, statusEl, excerpt);
    const argsBlock = document.createElement('div');
    const argsLabel = document.createElement('span');
    argsLabel.className = 'chat-tool__label';
    argsLabel.textContent = 'arguments';
    const argsPre = document.createElement('pre');
    const argsText = typeof args === 'string' ? args : JSON.stringify(args ?? {}, null, 2);
    argsPre.textContent = boundedText(argsText || '{}');
    argsBlock.append(argsLabel, argsPre);
    const resultBlock = document.createElement('div');
    const resultLabel = document.createElement('span');
    resultLabel.className = 'chat-tool__label';
    resultLabel.textContent = 'result';
    const resultPre = document.createElement('pre');
    resultPre.textContent = '';
    resultBlock.append(resultLabel, resultPre);
    body.append(argsBlock, resultBlock);
    head.addEventListener('click', () => {
      const expanded = head.getAttribute('aria-expanded') !== 'true';
      head.setAttribute('aria-expanded', String(expanded));
      head.setAttribute('aria-label', `${expanded ? 'Hide' : 'Show'} details for ${toolName}`);
      body.hidden = !expanded;
    });
    card.append(head, body);
    return { card, statusEl, statusText, resultPre, head, toggle: head, body, excerpt };
  }

  function turnToolRow(id, name) {
    const host = $('chat-turn-tools');
    host.hidden = false;
    const row = document.createElement('div');
    row.className = 'chat-turn-tool';
    row.dataset.toolId = id;
    const label = document.createElement('span');
    label.className = 'chat-turn-tool__name';
    label.textContent = name;
    const status = document.createElement('span');
    status.textContent = 'running';
    row.append(iconNode('tools', 'icon icon--sm'), label, status);
    host.appendChild(row);
    return { row, status };
  }

  // A blocked tool call is recorded, never resumed from this card. There is
  // no paused-run registry in the backend: the only decision path is the
  // Approvals view, which POSTs /approvals/{id}/resolve.
  function approvalNode({ approval_id, name, reason, tool_call_id = '' }) {
    const id = String(approval_id || '');
    const card = document.createElement('div');
    card.className = 'chat-approval chat-approval--blocked';
    card.dataset.approvalId = id;
    card.dataset.approvalName = String(name || 'A tool call');
    if (tool_call_id) card.dataset.toolCallId = tool_call_id;
    const title = document.createElement('div');
    title.className = 'chat-approval__title';
    title.innerHTML = `${ico('alert', 'icon icon--sm')}<span>Tool blocked — approval required</span>`;
    const body = document.createElement('div');
    body.className = 'chat-approval__reason';
    body.textContent = `${name || 'A tool call'} did not run. ${reason || 'A high-risk action needs a human decision.'} This card is read-only; it cannot resume the tool.`;
    const meta = document.createElement('div');
    meta.className = 'chat-approval__meta';
    meta.textContent = [tool_call_id ? `tool_call_id ${tool_call_id}` : '', `approval ${id || 'unknown'}`]
      .filter(Boolean).join(' · ');
    const actions = document.createElement('div');
    actions.className = 'chat-approval__actions';
    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'btn btn--secondary btn--sm';
    open.textContent = 'Decide in Approvals view';
    open.addEventListener('click', () => document.querySelector('[data-view="approvals"]')?.click());
    actions.appendChild(open);
    card.append(title, body, meta, actions);
    state.pendingApprovals.set(id, { card, toolCallId: tool_call_id, title, body, meta });
    return card;
  }

  function markApprovalResolved(id) {
    const record = state.pendingApprovals.get(id);
    if (!record) return;
    record.card.classList.remove('chat-approval--blocked');
    record.card.classList.add('chat-approval--resolved');
    record.title.innerHTML = `${ico('check', 'icon icon--sm')}<span>Decision recorded in Approvals view</span>`;
    record.body.textContent = 'The decision was recorded elsewhere. This blocked tool call was not resumed by this conversation.';
    const actions = record.card.querySelector('.chat-approval__actions');
    if (actions) actions.remove();
  }

  // Repeated approvals for the same tool are grouped into one collapsed card.
  // Every approval id remains present as a row, so reconciliation and the
  // Approvals view still see all pending decisions.
  function appendApprovalCard(node, host) {
    const thread = host || threadEl();
    if (!thread || !node) return;
    const name = node.dataset.approvalName || 'A tool call';
    let group = thread.lastElementChild;
    if (!group || !group.classList.contains('chat-approval-group') || group.dataset.approvalName !== name) {
      group = document.createElement('div');
      group.className = 'chat-approval-group';
      group.dataset.approvalName = name;
      group.dataset.count = '0';
      const details = document.createElement('details');
      details.className = 'chat-approval-group__details';
      const summary = document.createElement('summary');
      summary.className = 'chat-approval-group__summary';
      const count = document.createElement('span');
      count.className = 'chat-approval-group__count';
      summary.append(iconNode('alert', 'icon icon--sm'), count);
      const items = document.createElement('div');
      items.className = 'chat-approval-group__items';
      details.append(summary, items);
      group.appendChild(details);
      thread.appendChild(group);
    }
    const items = group.querySelector('.chat-approval-group__items');
    const count = group.querySelector('.chat-approval-group__count');
    items.appendChild(node);
    group.dataset.count = String(Number(group.dataset.count || 0) + 1);
    if (count) count.textContent = `${group.dataset.count} blocked approval${group.dataset.count === '1' ? '' : 's'} — ${name}`;
  }

  // Rehydrate pending approvals after a reload and reconcile decisions made
  // in the Approvals view (app.js emits leo:data after every poll/action).
  async function syncPendingApprovals() {
    let data;
    try { data = await api('/approvals'); } catch { return; }
    const pending = Array.isArray(data?.approvals) ? data.approvals : [];
    const ids = new Set(pending.map((a) => String(a?.id || '')));
    for (const id of [...state.pendingApprovals.keys()]) {
      if (!ids.has(id)) { markApprovalResolved(id); state.pendingApprovals.delete(id); }
    }
    const thread = threadEl();
    if (!thread) return;
    const shown = new Set([...thread.querySelectorAll('[data-approval-id]')].map((n) => n.dataset.approvalId));
    for (const record of pending) {
      const id = String(record?.id || '');
      if (!id || shown.has(id) || state.pendingApprovals.has(id)) continue;
      let meta = {};
      try { meta = JSON.parse(record.metadata || '{}') || {}; } catch { meta = {}; }
      if (state.activeId && meta.task_id && String(meta.task_id) !== String(state.activeId)) continue;
      const card = approvalNode({
        approval_id: id,
        name: record.action || meta.name || 'a tool call',
        reason: record.reason,
        tool_call_id: meta.tool_call_id || record.tool_call_id || '',
      });
      appendApprovalCard(card, thread);
      shown.add(id);
      maybeScroll();
    }
  }

  function noticeNode(text, danger = false) {
    const el = document.createElement('div');
    el.className = `chat-notice${danger ? ' chat-notice--danger' : ''}`;
    el.textContent = text;
    return el;
  }

  function updateMessageCount() {
    const count = threadEl().querySelectorAll('.chat-msg').length;
    $('chat-msg-count').textContent = `${count} message${count === 1 ? '' : 's'}`;
  }

  /* ---- conversation history ---------------------------------------------- */

  // One history row -> one thread node. Shared by the initial load and the
  // incremental live append so both render identically. Returns whether the
  // row represents an incomplete assistant turn.
  function appendHistoryMessage(message, thread) {
    const host = thread || threadEl();
    let historyIncomplete = false;
    if (message.role === 'user') {
      const node = messageNode({ role: 'user', content: message.content, time: timeAgo(message.created_at) });
      const chips = attachmentChips(message.metadata);
      if (chips.childElementCount) node.appendChild(chips);
      host.appendChild(node);
    } else if (message.role === 'assistant') {
      const meta = [message.metadata?.provider, message.metadata?.model].filter(Boolean).join('/');
      const pending = Array.isArray(message.metadata?.tool_calls_pending)
        ? message.metadata.tool_calls_pending : [];
      historyIncomplete = message.metadata?.run_status === 'incomplete' || pending.length > 0;
      host.appendChild(messageNode({ role: 'assistant', content: message.content, meta, time: timeAgo(message.created_at) }));
      if (historyIncomplete) {
        host.appendChild(noticeNode(`incomplete — ${pending.length} tool call(s) pending; this turn was not fully answered.`, true));
      }
    } else if (message.role === 'tool') {
      let parsed = {};
      try { parsed = JSON.parse(message.content || '{}'); } catch { parsed = {}; }
      const blocked = Boolean(parsed.approval_required);
      const failed = Boolean(parsed.error) || blocked;
      const args = message.metadata?.arguments ?? parsed.arguments ?? {};
      const tool = toolNode({
        name: message.metadata?.name || parsed.name || 'tool',
        arguments: args,
        status: failed ? 'failed' : 'done',
        toolCallId: message.metadata?.tool_call_id || '',
      });
      if (blocked) {
        tool.card.dataset.approvalId = parsed.approval_required;
        tool.statusEl.className = 'status status--warn';
        tool.statusText.textContent = 'blocked';
      }
      const result = parsed.result !== undefined
        ? parsed.result
        : (parsed.output !== undefined ? parsed.output : message.content);
      setToolResult(tool, result);
      host.appendChild(tool.card);
    }
    if (Number.isFinite(Number(message.id))) {
      state.lastMessageId = Math.max(state.lastMessageId || 0, Number(message.id));
    }
    return historyIncomplete;
  }

  /* ---- live updates from other clients ------------------------------------ */

  // Resume position after a locally streamed turn. The live SSE stream renders
  // the user's own turn node-by-node, so once it ends the durable rows must not
  // be appended a second time when the bus reports the same completion.
  async function syncLastMessageId() {
    const id = state.activeId;
    if (!id) return;
    try {
      const data = await api(
        `/conversations/${encodeURIComponent(id)}?after_id=${Number.MAX_SAFE_INTEGER}`,
      );
      if (state.activeId === id) {
        state.lastMessageId = Math.max(state.lastMessageId || 0, Number(data.latest_id) || 0);
      }
    } catch { /* best effort; the next bus event retries the sync */ }
  }

  async function appendNewMessages() {
    const id = state.activeId;
    if (!id || state.running) return;
    const data = await api(
      `/conversations/${encodeURIComponent(id)}?after_id=${state.lastMessageId || 0}`,
    );
    if (state.activeId !== id) return;  // user switched while the fetch was in flight
    const messages = data.messages || [];
    let incomplete = false;
    const thread = threadEl();
    for (const message of messages) {
      if (appendHistoryMessage(message, thread)) incomplete = true;
    }
    state.lastMessageId = Math.max(state.lastMessageId || 0, Number(data.latest_id) || 0);
    if (messages.length) {
      maybeScroll();
      updateMessageCount();
      refreshSessions();
      if (incomplete && !state.running) applyTerminalStatus('incomplete');
      else if (!state.running && state.terminalStatus === 'incomplete') applyTerminalStatus('incomplete');
    }
  }

  function handleBusEvent(event) {
    if (!event || event.topic !== 'chat.completed') return;
    const payload = event.payload || {};
    // The sidebar counters follow every conversation, including channels.
    refreshSessions();
    const conversationId = payload.conversation_id || '';
    if (!conversationId || conversationId !== state.activeId) return;
    if (state.running) return;  // the local turn renders live; never double-render
    // Our own dashboard channel: the stream already rendered it. Foreign
    // channels (api, whatsapp, telegram...) are the reason this feed exists.
    if (payload.channel === 'dashboard') { syncLastMessageId(); return; }
    appendNewMessages().catch(() => {});
  }

  function scheduleEventStreamRetry() {
    if (state.eventRetryTimer) return;
    state.eventRetryTimer = setTimeout(() => {
      state.eventRetryTimer = null;
      connectEventStream();
    }, 3000);
  }

  function connectEventStream() {
    if (state.eventSource || typeof EventSource === 'undefined') return;
    api('/events/stream-token', { method: 'POST' }).then(({ token }) => {
      if (state.eventSource) return;
      // The token is single-use, so EventSource's built-in reconnect replays a
      // dead token; every close goes through here and mints a fresh one.
      const source = new EventSource(
        `/events/stream?token=${encodeURIComponent(token)}&max_seconds=900`,
      );
      state.eventSource = source;
      source.onmessage = (e) => {
        let event = null;
        try { event = JSON.parse(e.data); } catch { return; }
        try { handleBusEvent(event); } catch { /* a feed bug must not kill the feed */ }
      };
      source.onerror = () => {
        source.close();
        state.eventSource = null;
        scheduleEventStreamRetry();
      };
    }).catch(scheduleEventStreamRetry);
  }

  async function selectConversation(id, { silent = false } = {}) {
    if (state.running) { toast('Wait for the current turn to finish first.', 'warn'); return; }
    state.activeId = id || null;
    if (id) localStorage.setItem('leo_chat_conversation', id);
    else localStorage.removeItem('leo_chat_conversation');
    $('chat-shell').classList.remove('chat-history-open');
    renderSessions();
    sessionTitle();
    const thread = threadEl();
    thread.replaceChildren();
    state.stream = null;
    state.turnReasoningStreams = [];
    state.turnReasoningSeen = false;
    state.noReasoningNote = false;
    state.reasoningHidden = false;
    state.reasoningClosedRuns = new Set();
    state.turnNodes = [];
    state.pendingApprovals.clear();
    updateMessageCount();
    if (!id) { renderEmpty(); maybeScroll(true); return; }
    if (!silent) setChatStatus('live', 'Loading…');
    try {
      const data = await api(`/conversations/${encodeURIComponent(id)}`);
      $('chat-channel-badge').textContent = data.channel || 'dashboard';
      let historyIncomplete = false;
      state.lastMessageId = 0;
      for (const message of data.messages || []) {
        if (appendHistoryMessage(message, thread)) historyIncomplete = true;
      }
      state.lastMessageId = Math.max(
        state.lastMessageId || 0, Number(data.latest_id) || 0,
      );
      if (!thread.childElementCount) renderEmpty();
      state.scrollPinned = true;
      maybeScroll(true);
      if (!silent) {
        if (historyIncomplete) applyTerminalStatus('incomplete');
        else setChatStatus('live', 'Ready');
      }
      await syncPendingApprovals();
    } catch (error) {
      if (!silent) setChatStatus('live', 'Ready');
      thread.appendChild(noticeNode(`could not load history: ${error.message}`, true));
      if (!thread.childElementCount) renderEmpty();
    }
    updateMessageCount();
  }

  /* ---- live streaming turn ------------------------------------------------- */

  function ensureAssistantNode() {
    if (state.stream && !state.stream.sealed) return state.stream;
    const node = messageNode({ role: 'assistant', content: '' });
    node.querySelector('.chat-msg__head').appendChild(thinkingNode());
    threadEl().appendChild(node);
    state.stream = {
      node,
      contentEl: node.querySelector('.chat-msg__content'),
      buffer: '',
      sealed: false,
      renderQueued: false,
      renderTimer: null,
      // Every panel is bound to the run and message that created it. No panel
      // is attached here: that happens only on the first real thinking.delta.
      runId: state.activeRun && state.activeRun !== 'pending' ? state.activeRun : null,
      reasoning: null,
    };
    state.turnNodes.push(node);
    maybeScroll();
    return state.stream;
  }

  function thinkingNode() {
    const el = document.createElement('span');
    el.className = 'chat-thinking';
    el.innerHTML = '<span class="chat-thinking__dots"><i></i><i></i><i></i></span> thinking';
    return el;
  }

  /* Streamed reasoning is a separate channel. A native <details>/<summary>
     gives keyboard toggling and focus for free; only the one-line status is
     aria-live, so tokens never produce screen-reader announcements. */
  function reasoningNode() {
    const panel = document.createElement('details');
    panel.className = 'chat-reasoning';
    panel.dataset.reasoningState = 'empty';
    const toggle = document.createElement('summary');
    toggle.className = 'chat-reasoning__toggle';
    toggle.setAttribute('aria-label', 'Show thinking');
    const icon = document.createElement('span');
    icon.className = 'chat-reasoning__icon';
    icon.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    label.className = 'chat-reasoning__label';
    label.textContent = 'Thinking';
    toggle.append(icon, label);
    const status = document.createElement('span');
    status.className = 'chat-reasoning__status sr-only';
    status.setAttribute('aria-live', 'polite');
    status.setAttribute('aria-atomic', 'true');
    status.textContent = '';
    const body = document.createElement('pre');
    body.className = 'chat-reasoning__body';
    body.textContent = '';
    panel.append(toggle, status, body);
    panel.addEventListener('toggle', () => {
      toggle.setAttribute('aria-label', `${panel.open ? 'Hide' : 'Show'} thinking`);
    });
    return { panel, toggle, body, status };
  }

  function attachReasoningPanel(stream) {
    if (!stream || stream.reasoning || !stream.node || !stream.node.isConnected) return stream && stream.reasoning;
    const reasoning = reasoningNode();
    reasoning.node = stream.node;  // the exact assistant message this panel belongs to
    reasoning.runId = stream.runId || null;
    reasoning.panel.dataset.reasoningRun = reasoning.runId || '';
    // The panel belongs ABOVE the answer: providers may stream reasoning after
    // the visible text, and appending would leave a stale-looking THINKING
    // card below the response. Insert before the content node when it exists.
    const contentNode = stream.node.querySelector('.chat-msg__content');
    if (contentNode && contentNode.parentNode === stream.node) {
      stream.node.insertBefore(reasoning.panel, contentNode);
    } else {
      stream.node.appendChild(reasoning.panel);  // never inside .chat-msg__content
    }
    stream.reasoning = reasoning;
    stream.reasoningBuffer = '';
    stream.reasoningSeen = false;
    stream.reasoningFinalized = false;
    stream.reasoningTerminal = false;
    stream.reasoningAnnounced = false;
    stream.reasoningRenderQueued = false;
    stream.reasoningTimer = null;
    state.turnReasoningStreams.push(stream);
    return stream.reasoning;
  }

  function appendReasoning(delta) {
    if (typeof delta !== 'string' || !delta) return;
    if (reasoningClosed()) return;  // terminal run: never open a panel or node
    const stream = ensureAssistantNode();
    // A panel detached with its run (or moved to another message) is stale:
    // never reattach it to this turn. Only the first real delta makes a panel.
    if (stream.reasoning && (stream.reasoning.node !== stream.node || stream.reasoning.panel.parentNode !== stream.node)) {
      stream.reasoning = null;
    }
    if (!stream.reasoning) attachReasoningPanel(stream);
    // A tool round can seal this block early. Real reasoning arrived
    // afterwards: clear the provisional text and re-arm the throttled render.
    if (stream.reasoningFinalized && !stream.reasoningTerminal) {
      stream.reasoningFinalized = false;
      stream.reasoning.body.textContent = '';
    }
    state.turnReasoningSeen = true;
    stream.reasoningSeen = true;
    stream.reasoningBuffer = boundedText(stream.reasoningBuffer + delta, REASONING_RESULT_LIMIT);
    stream.reasoning.panel.dataset.reasoningState = 'streaming';
    stream.reasoning.panel.open = true;  // real reasoning is visible live, still collapsible
    if (!stream.reasoningAnnounced) {
      stream.reasoningAnnounced = true;
      stream.reasoning.status.textContent = 'Reasoning was streamed by the model in the Thinking panel.';
    }
    queueReasoningRender();
  }

  function queueReasoningRender() {
    const stream = state.stream;
    if (!stream || !stream.reasoning || stream.reasoningFinalized || stream.reasoningRenderQueued) return;
    stream.reasoningRenderQueued = true;
    stream.reasoningTimer = setTimeout(() => {
      if (!stream.reasoning || stream.reasoningFinalized) return;
      stream.reasoningRenderQueued = false;
      stream.reasoningTimer = null;
      stream.reasoning.body.textContent = boundedText(stream.reasoningBuffer, REASONING_RESULT_LIMIT);
      maybeScroll();
    }, RENDER_THROTTLE_MS);
  }

  function finalizeReasoning(stream, terminal = false) {
    if (!stream || !stream.reasoning) return;
    // A tool round can seal a panel early; a terminal event still gets to
    // replace its provisional "this part" message with the truthful verdict.
    if (stream.reasoningFinalized && (stream.reasoningTerminal || !terminal)) return;
    if (terminal) stream.reasoningTerminal = true;
    if (stream.reasoningTimer) { clearTimeout(stream.reasoningTimer); stream.reasoningTimer = null; }
    stream.reasoningRenderQueued = false;
    const { panel, body, status } = stream.reasoning;
    if (stream.reasoningSeen) {
      body.textContent = boundedText(stream.reasoningBuffer, REASONING_RESULT_LIMIT);
      panel.dataset.reasoningState = 'received';
      if (!stream.reasoningAnnounced) {
        stream.reasoningAnnounced = true;
        status.textContent = 'Reasoning was streamed by the model in the Thinking panel.';
      }
    } else {
      // No real delta ever reached this panel. An empty disclosure is not a
      // thinking transcript: detach it. A short, honest line is recorded by
      // closeTurnStream instead; no reasoning text is ever invented here.
      detachReasoningPanel(stream);
      stream.reasoningFinalized = true;
      return;
    }
    if (terminal && stream.reasoningSeen) {
      // Finished turns keep the transcript but collapse it: a completed answer
      // must not end with an open, stale THINKING block.
      panel.open = false;
    }
    stream.reasoningFinalized = true;
  }

  function finalizeTurnReasoning(terminal = false, runId = (state.stream && state.stream.runId) || null) {
    state.turnReasoningStreams.slice().forEach((stream) => {
      if (runId && stream.runId && stream.runId !== runId) return;  // foreign run
      finalizeReasoning(stream, terminal);
    });
  }

  // Detach only the panel owned by this exact assistant message. Checking the
  // parent keeps a panel from a different turn/message from being removed.
  function detachReasoningPanel(stream) {
    if (!stream || !stream.reasoning) return;
    const { panel, node } = stream.reasoning;
    if (node && panel.parentNode === node) panel.remove();
    stream.reasoning = null;
    if (stream.reasoningTimer) { clearTimeout(stream.reasoningTimer); stream.reasoningTimer = null; }
    stream.reasoningRenderQueued = false;
  }

  function noteNoReasoning(stream) {
    const text = state.reasoningHidden ? NO_REASONING_HIDDEN_NOTE : NO_REASONING_NOTE;
    state.noReasoningNote = true;
    setTurnStatus(text);
    if (!stream || !stream.node) return;
    // One line on this message, never a large card and never invented text.
    stream.node.querySelectorAll('.chat-reasoning-none').forEach((note) => note.remove());
    const note = document.createElement('span');
    note.className = 'chat-reasoning-none';
    note.textContent = text;
    stream.node.appendChild(note);
  }

  // The backend's message.finished.reasoning_seen and final thinking.state are
  // authoritative: a lost delta must not become a false "no reasoning" verdict.
  // This only flips bookkeeping — no panel and no reasoning text is invented.
  function markReasoningSeen() {
    state.turnReasoningSeen = true;
    clearNoReasoningNotes(state.stream);
  }

  // A later tool round may stream real reasoning after an earlier segment was
  // reported empty. Retract that one-line verdict for the CURRENT run and its
  // one assistant message only; other turns' notes are history, not ours to
  // delete.
  function clearNoReasoningNotes(stream) {
    if (!stream || !stream.node) return;
    const runId = stream.runId;
    if (runId && state.activeRun && state.activeRun !== 'pending' && runId !== state.activeRun) return;
    stream.node.querySelectorAll('.chat-reasoning-none').forEach((note) => note.remove());
    if (!runId || !state.activeRun || state.activeRun === runId) state.noReasoningNote = false;
  }

  // Terminal paths close the reasoning channel for exactly one run. The latch
  // keeps working after state.stream is null, which is what stops a trailing
  // thinking.delta from opening a brand-new assistant message.
  function closeReasoning(runId = (state.activeRun && state.activeRun !== 'pending' ? state.activeRun : null)) {
    if (runId && runId !== 'pending') state.reasoningClosedRuns.add(runId);
  }

  function reasoningClosed(runId = state.activeRun) {
    return Boolean(runId && runId !== 'pending' && state.reasoningClosedRuns.has(runId));
  }

  // Terminal housekeeping: never creates a panel. It removes only this run's
  // stale/disconnected panel and clears the live stream so later events (or a
  // history reload) cannot reattach an old panel to a newer answer.
  function closeTurnStream(terminal = true, runId = null) {
    if (!state.stream) return;
    if (runId && state.stream.runId && runId !== state.stream.runId) return;
    const stream = state.stream;
    finalizeTurnReasoning(terminal, runId || stream.runId);
    if (terminal) {
      if (!state.turnReasoningSeen && stream.node && stream.node.isConnected) noteNoReasoning(stream);
      else clearNoReasoningNotes(stream);
    }
    state.stream = null;
    state.turnReasoningStreams = [];
    state.turnReasoningSeen = false;
  }

  function sealStream(terminal = false, runId = null) {
    if (!state.stream) return;
    // Supersession/cancellation must not seal a newer run's assistant block.
    if (runId && state.stream.runId && runId !== state.stream.runId) return;
    state.stream.sealed = true;
    // A pending throttled render would re-add the streaming caret after this
    // non-streaming render, so cancel it before painting the final text.
    if (state.stream.renderTimer) {
      clearTimeout(state.stream.renderTimer);
      state.stream.renderTimer = null;
    }
    const thinking = state.stream.node.querySelector('.chat-thinking');
    if (thinking) thinking.remove();
    window.LEOMarkdown.renderInto(state.stream.contentEl, state.stream.buffer);
    if (terminal) finalizeTurnReasoning(true);
    else finalizeReasoning(state.stream, false);
  }

  // The caret is markdown-renderer state ({ streaming: true } adds it). Remove
  // it explicitly on every terminal path, not just on a successful answer.
  function removeStreamingCarets() {
    threadEl().querySelectorAll('.chat-msg--assistant .md-caret').forEach((caret) => caret.remove());
  }

  function queueStreamRender() {
    const stream = state.stream;
    if (!stream || stream.renderQueued || stream.sealed) return;
    stream.renderQueued = true;
    stream.renderTimer = setTimeout(() => {
      stream.renderQueued = false;
      stream.renderTimer = null;
      if (stream.sealed || !stream.node.isConnected) return;  // never re-add a caret after a seal
      window.LEOMarkdown.renderInto(stream.contentEl, stream.buffer, { streaming: true });
      const head = stream.node.querySelector('.chat-msg__head');
      if (head && !head.querySelector('.chat-msg__meta').textContent) {
        head.querySelector('.chat-msg__meta').textContent = 'streaming…';
      }
      maybeScroll();
    }, RENDER_THROTTLE_MS);
  }

  function applyStreamEvent(event) {
    const eventRunId = event.run_id || null;
    const priorRun = state.activeRun;
    // A fresh run.started supersedes any run we were tracking — including one
    // adopted from a stale out-of-order frame while activeRun was 'pending'.
    const newRun = event.type === 'run.started' && eventRunId && priorRun
      && priorRun !== 'pending' && priorRun !== eventRunId;
    // A reconnected/replaced stream must not paint another run's events into
    // this thread; once a run is known, ignore every other run_id.
    if (!newRun && state.activeRun && state.activeRun !== 'pending' && event.run_id && event.run_id !== state.activeRun) return;
    // 'pending' means the POST has not acknowledged a run yet. Only
    // run.started may establish its id: adopting an arbitrary frame here lets
    // a previous run's trailing delta be counted as part of the new turn.
    if (state.activeRun === 'pending' && event.type !== 'run.started') return;
    if (state.stream && !state.stream.runId && state.activeRun && state.activeRun !== 'pending') {
      state.stream.runId = state.activeRun;
    }
    if (newRun) state.lastSeq = -1;  // every run has its own seq sequence
    if (typeof event.seq === 'number') {
      if (event.seq <= state.lastSeq) return;
      state.lastSeq = event.seq;
    }

    if (event.type === 'run.started') {
      if (newRun) {
        // The superseded run is terminal for this UI. Its animated thinking
        // indicator and open assistant node must be sealed here, not left
        // spinning next to the new answer.
        closeReasoning(priorRun);
        sealStream(true, priorRun);
        closeTurnStream(false, priorRun);  // never reattach the old run's panel
      }
      state.activeRun = eventRunId;
      state.reasoningClosedRuns.delete(eventRunId);  // a fresh run reopens the latch
      if (!state.activeId && event.conversation_id) {
        state.activeId = event.conversation_id;
        localStorage.setItem('leo_chat_conversation', state.activeId);
      }
      $('chat-channel-badge').textContent = 'dashboard';
      ensureAssistantNode();
      setTurnState('warn', 'running');
      setTurnStatus('thinking');
      startElapsed();
    } else if (event.type === 'status') {
      setChatStatus(event.status, STATUS_LABELS[event.status] || event.status);
      if (event.status === 'thinking') { ensureAssistantNode(); setTurnState('warn', 'thinking'); setTurnStatus('thinking'); }
      if (event.status === 'tool') {
        sealStream();
        setTurnState('warn', 'tools');
        setTurnStatus(`calling ${event.count || 1} tool(s)`);
      }
      if (event.status === 'responding') { setTurnState('accent', 'responding'); setTurnStatus('composing the reply'); }
      if (event.status === 'cancelled') {
        // Full cleanup: seal the node and remove the animated thinking
        // indicator before releasing the stream. closeTurnStream alone leaves
        // both behind on a cancelled run.
        closeReasoning(eventRunId);
        sealStream(true, eventRunId);
        removeStreamingCarets();
        closeTurnStream(true, eventRunId);  // no stale panel on a cancelled run
        setTurnState('', 'cancelled'); setTurnStatus('cancelled by user');
      }
    } else if (event.type === 'message.delta') {
      ensureAssistantNode();  // a delta after tool cards opens a new assistant block
      state.stream.buffer += event.delta || '';
      queueStreamRender();
    } else if (event.type === 'thinking.delta') {
      // Real provider reasoning only. It goes to its own labelled panel: never
      // into the answer buffer, the copy button, or a tool card. A delta from
      // another run must not open a panel on this run's assistant message.
      // A delta from a CLOSED run is ignored outright — even if state.stream
      // is already null — so it cannot create a new empty assistant node.
      if (reasoningClosed(eventRunId)) return;
      if (eventRunId && state.stream && state.stream.runId && eventRunId !== state.stream.runId) return;
      appendReasoning(event.delta);
    } else if (event.type === 'tool.started') {
      const tool = toolNode({ name: event.name, risk: event.risk, arguments: event.arguments, status: 'running', toolCallId: event.tool_call_id });
      state.turnTools.set(event.tool_call_id, { name: event.name, ...tool });
      threadEl().appendChild(tool.card);
      const mini = turnToolRow(event.tool_call_id, event.name);
      state.turnTools.get(event.tool_call_id).mini = mini;
      maybeScroll();
    } else if (event.type === 'tool.finished') {
      const tool = state.turnTools.get(event.tool_call_id);
      if (tool) {
        tool.card.classList.remove('chat-tool--running');
        const blocked = Boolean(tool.approvalPending);
        const failed = event.is_error || !event.success;
        tool.statusEl.className = `status status--${blocked ? 'warn' : failed ? 'danger' : 'success'}`;
        tool.statusText.textContent = blocked ? 'blocked' : failed ? 'failed' : 'done';
        setToolResult(tool, event.result || '');
        if (tool.mini) tool.mini.status.textContent = blocked ? 'blocked' : failed ? 'failed' : 'done';
      }
    } else if (event.type === 'approval.required') {
      setChatStatus('waiting_approval', STATUS_LABELS.waiting_approval);
      setTurnState('warn', 'approval');
      setTurnStatus('waiting for your decision');
      const blocked = state.turnTools.get(event.tool_call_id);
      if (blocked) {
        blocked.approvalPending = true;
        blocked.card.dataset.approvalId = event.approval_id;
        blocked.statusEl.className = 'status status--warn';
        blocked.statusText.textContent = 'blocked';
        if (blocked.mini) blocked.mini.status.textContent = 'blocked';
      }
      appendApprovalCard(approvalNode({
        approval_id: event.approval_id, name: event.name, reason: event.reason,
        tool_call_id: event.tool_call_id,
      }), threadEl());
      banner({
        kind: 'warn', title: 'Approval required',
        text: `${event.name || 'A tool'} needs your decision before it can run.`,
        actions: [{ label: 'Open Approvals view', run: () => document.querySelector('[data-view="approvals"]')?.click() }],
      });
      maybeScroll();
    } else if (event.type === 'thinking.state') {
      // Explicit per-segment verdict from the backend. A final state with
      // seen=true means reasoning existed even if its deltas were lost:
      // suppress the false note, but never synthesize transcript text. This
      // is NOT a terminal event — a later tool round can stream more.
      if (!reasoningClosed(eventRunId) && event.final && event.seen === true) markReasoningSeen();
    } else if (event.type === 'message.finished') {
      // The backend's flag is authoritative, independent of client deltas.
      if (event.reasoning_seen === true) markReasoningSeen();
      state.reasoningHidden = event.reasoning_display === 'hidden';
      closeReasoning(eventRunId);
      sealStream();
      removeStreamingCarets();
      const target = state.stream;
      if (target) {
        if (event.content) { target.buffer = event.content; window.LEOMarkdown.renderInto(target.contentEl, target.buffer); }
        const head = target.node.querySelector('.chat-msg__head');
        const meta = head.querySelector('.chat-msg__meta');
        meta.textContent = [
          [event.provider, event.model].filter(Boolean).join('/'),
          event.usage ? `${event.usage.input_tokens || 0} in · ${event.usage.output_tokens || 0} out` : '',
        ].filter(Boolean).join(' · ');
        const actions = document.createElement('div');
        actions.className = 'chat-msg__actions';
        const copy = document.createElement('button');
        copy.type = 'button';
        copy.className = 'chat-msg__copy';
        copy.textContent = 'copy';
        copy.addEventListener('click', () => {
          navigator.clipboard.writeText(target.buffer).then(() => {
            copy.textContent = 'copied';
            setTimeout(() => { copy.textContent = 'copy'; }, 1600);
          }).catch(() => toast('Clipboard unavailable in this context', 'warn'));
        });
        actions.appendChild(copy);
        if (!target.node.querySelector('.chat-msg__actions')) target.node.appendChild(actions);
      }
      if (event.usage) {
        $('chat-turn-tokens').textContent = `${event.usage.input_tokens || 0} in · ${event.usage.output_tokens || 0} out`;
      }
      if (Array.isArray(event.memories_used)) {
        $('chat-turn-memories').textContent = `${event.memories_used.length} in context`;
      }
      const pending = Array.isArray(event.tool_calls_pending) ? event.tool_calls_pending : [];
      if (event.status === 'completed' && !pending.length) {
        setTurnState('success', 'done');
        setTurnStatus('turn complete');
      } else {
        // An assistant answer with unresolved tool calls is not a success.
        applyTerminalStatus('incomplete');
        const detail = pending.length
          ? `${pending.length} tool call(s) still pending: ${pending.join(', ')}`
          : 'the backend did not mark this turn completed';
        threadEl().appendChild(noticeNode(`incomplete: ${detail}`, true));
        banner({ kind: 'warn', title: 'Turn incomplete', text: `${detail}. Ask again to continue.` });
      }
      // The message is over: release its live stream now. run.finished must not
      // append a fresh panel to this (or any other) completed answer.
      closeTurnStream(true, eventRunId);
    } else if (event.type === 'error') {
      closeReasoning(eventRunId);
      sealStream(true, eventRunId);
      closeTurnStream(true, eventRunId);
      threadEl().appendChild(noticeNode(`error: ${event.message || 'stream failed'}`, true));
      banner({ kind: 'danger', title: 'The turn failed', text: event.message || 'stream failed' });
      applyTerminalStatus('failed');
      toast(event.message || 'Stream failed', 'danger');
    } else if (event.type === 'run.finished') {
      closeReasoning(eventRunId);
      sealStream(true, eventRunId);
      closeTurnStream(true, eventRunId);
      applyTerminalStatus(event.status);
      stopElapsed();
    }
  }

  // A command whose busy policy is "interrupt" (/stop) must run while a turn is
  // live — Hermes's interrupt_then_dispatch. It goes to the dedicated endpoint
  // so the running stream keeps its state instead of being replaced or queued.
  function commandName(text) {
    const match = /^\s*\/([a-z][a-z0-9_-]*)/i.exec(text || '');
    return match ? match[1].toLowerCase() : null;
  }

  function isInterruptCommand(text) {
    const name = commandName(text);
    if (!name) return false;
    // The backend's catalog is authoritative when it has loaded; before that,
    // the built-in stop aliases are the known interrupt commands.
    const catalog = Array.isArray(state.commands) ? state.commands : [];
    const row = catalog.find(
      (cmd) => cmd.name === name || (Array.isArray(cmd.aliases) && cmd.aliases.includes(name)),
    );
    if (row) return row.busy_policy === 'interrupt';
    return name === 'stop' || name === 'cancel';
  }

  async function requestStop(text) {
    const thread = threadEl();
    thread.appendChild(messageNode({ role: 'user', content: text }));
    if (!state.activeId) {
      thread.appendChild(noticeNode('nothing is running in this conversation'));
      return;
    }
    try {
      await api(`/conversations/${encodeURIComponent(state.activeId)}/stop`, { method: 'POST' });
      thread.appendChild(noticeNode('stop requested — the turn ends at its next step'));
      maybeScroll(true);
    } catch (error) {
      toast(error.message || 'stop failed', 'danger');
    }
  }

  async function streamChat(text) {
    const ids = state.attachments.map((record) => record.id);
    const userNode = messageNode({ role: 'user', content: text, meta: ids.length ? `${ids.length} attachment(s)` : '' });
    const chips = attachmentChips({ attachments: state.attachments.map(({ id, name, mime }) => ({ id, name, mime })) });
    if (chips.childElementCount) userNode.appendChild(chips);
    // The uploaded records belong to this turn only — clear the composer so a
    // follow-up message never silently re-attaches them.
    state.attachments = [];
    renderAttachments();
    const thread = threadEl();
    const hero = thread.querySelector('.chat-hero');
    if (hero) hero.remove();
    thread.appendChild(userNode);

    const controller = new AbortController();
    state.abortController = controller;
    state.activeRun = 'pending';
    state.lastSeq = -1;
    state.running = true;
    state.stream = null;
    state.turnReasoningStreams = [];
    state.turnReasoningSeen = false;
    state.noReasoningNote = false;
    state.reasoningHidden = false;
    state.reasoningClosedRuns = new Set();
    state.turnNodes = [];
    state.terminalStatus = null;
    state.streamDropped = false;
    let sawRunFinished = false;
    state.turnTools = new Map();
    $('chat-turn-tools').hidden = true;
    $('chat-turn-tools').replaceChildren();
    $('chat-turn-tokens').textContent = '—';
    $('chat-turn-memories').textContent = '—';
    clearBanners();
    setRunning(true);
    setChatStatus('thinking', STATUS_LABELS.thinking);
    setTurnState('warn', 'running');
    setTurnStatus('sending your message');
    startElapsed();
    maybeScroll(true);

    const headers = { 'content-type': 'application/json' };
    const token = sessionStorage.getItem('leo_api_token') || '';
    if (token) headers.Authorization = `Bearer ${token}`;

    const body = {
      message: text,
      channel: 'dashboard',
      conversation_id: state.activeId || undefined,
      attachment_ids: ids,
    };

    let buffer = '';
    try {
      const response = await fetch('/chat/stream', {
        method: 'POST', headers, signal: controller.signal, body: JSON.stringify(body),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `HTTP ${response.status}`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let index;
        while ((index = buffer.indexOf('\n\n')) >= 0) {
          const frame = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          const line = frame.split('\n').find((part) => part.startsWith('data:'));
          if (!line) continue;
          let event;
          try { event = JSON.parse(line.slice(5).trim()); } catch { continue; }
          applyStreamEvent(event);
          if (state.terminalStatus) sawRunFinished = true;
        }
      }
    } catch (error) {
      state.streamDropped = true;
      // A locally aborted/dropped stream closes the reasoning channel too:
      // a trailing delta must not resurrect an assistant node after abort.
      closeReasoning();
      if (error.name === 'AbortError') {
        sealStream(true);
        closeTurnStream(true);
        applyTerminalStatus('cancelled');
        banner({ kind: 'warn', title: 'Stopped locally', text: 'The server run may still finish; the durable history is checked below.' });
      } else {
        sealStream(true);
        closeTurnStream(true);
        thread.appendChild(noticeNode(`error: ${error.message}`, true));
        applyTerminalStatus('failed');
        banner({ kind: 'danger', title: 'The turn failed', text: error.message });
        toast(error.message, 'danger');
      }
    } finally {
      state.running = false;
      closeReasoning();  // latch the run before activeRun is cleared
      state.activeRun = null;
      state.abortController = null;
      stopElapsed();
      setRunning(false);
      if (!sawRunFinished && !state.streamDropped) state.streamDropped = true;
      // A dropped stream is terminal too: remove any stale panel and record the
      // same truthful verdict instead of leaving a provisional empty card.
      sealStream(Boolean(state.terminalStatus) || state.streamDropped);
      closeTurnStream(true);
      if (state.terminalStatus) applyTerminalStatus(state.terminalStatus);
      else if (state.streamDropped) applyTerminalStatus('incomplete');
      updateMessageCount();
      refreshSessions();
      // Locally streamed rows carry no ids here; record the durable tail so a
      // late chat.completed bus event cannot append the same turn again.
      syncLastMessageId();
      if (state.streamDropped) {
        // The stream can close before run.finished (proxy drop, network loss).
        // The backend persists messages before streaming, so refetch the
        // durable transcript instead of trusting partial client-side state.
        const droppedId = state.activeId;
        if (droppedId) {
          thread.appendChild(noticeNode('stream dropped — reloading durable history', true));
          selectConversation(droppedId, { silent: true });
        }
      }
    }
  }

  function setRunning(running) {
    $('send').hidden = running;
    $('stop-btn').hidden = !running;
    $('message').disabled = false;
  }

  /* ---- attachments ----------------------------------------------------------- */

  // Rejections stay visible until dismissed: they are never silently erased
  // by a subsequent render of the accepted attachment chips.
  function renderAttachmentErrors() {
    const list = $('attachment-list');
    list.querySelectorAll('.attach-error').forEach((n) => n.remove());
    state.attachmentErrors.forEach((message, index) => {
      const error = document.createElement('div');
      error.className = 'attach-error';
      error.textContent = message;
      const dismiss = document.createElement('button');
      dismiss.type = 'button';
      dismiss.setAttribute('aria-label', `Dismiss upload error: ${message}`);
      dismiss.textContent = '×';
      dismiss.addEventListener('click', () => {
        state.attachmentErrors.splice(index, 1);
        renderAttachmentErrors();
      });
      error.appendChild(dismiss);
      list.appendChild(error);
    });
  }

  function pushAttachmentError(message) {
    state.attachmentErrors.push(message);
    renderAttachmentErrors();
  }

  function renderAttachments() {
    const list = $('attachment-list');
    list.replaceChildren();
    state.attachments.forEach((record, index) => {
      const chip = document.createElement('span');
      chip.className = 'attach-chip';
      if (record.mime && record.mime.startsWith('image/') && record.objectUrl) {
        const img = document.createElement('img');
        img.src = record.objectUrl; img.alt = '';
        chip.appendChild(img);
      } else {
        chip.insertAdjacentHTML('afterbegin', ico('file', 'icon icon--sm'));
      }
      const label = document.createElement('span');
      label.textContent = `${record.name} (${Math.round((record.size || 0) / 1024)} KB)`;
      chip.appendChild(label);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${record.name}`);
      remove.innerHTML = ico('x', 'icon icon--sm');
      remove.addEventListener('click', () => {
        if (record.objectUrl) URL.revokeObjectURL(record.objectUrl);
        state.attachments.splice(index, 1);
        renderAttachments();
      });
      chip.appendChild(remove);
      list.appendChild(chip);
    });
    renderAttachmentErrors();
  }

  async function handleFiles(files) {
    for (const file of Array.from(files || [])) {
      if (file.size > 10 * 1024 * 1024) {
        pushAttachmentError(`${file.name} exceeds the 10 MB limit. (rejected)`);
        continue;
      }
      let dataUrl;
      try {
        dataUrl = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result || ''));
          reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
          reader.readAsDataURL(file);
        });
      } catch (error) {
        pushAttachmentError(`${error.message} (rejected)`);
        continue;
      }
      try {
        const record = await api('/attachments', {
          method: 'POST',
          body: JSON.stringify({
            name: file.name, mime: file.type || 'application/octet-stream',
            data_base64: dataUrl.split(',')[1] || '',
            conversation_id: state.activeId || undefined,
          }),
        });
        record.objectUrl = file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
        state.attachments.push(record);
      } catch (error) {
        pushAttachmentError(`${file.name}: ${error.message} (rejected)`);
      }
    }
    renderAttachments();
  }

  /* ---- composer ---------------------------------------------------------------- */

  function autosize() {
    const input = $('message');
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  }

  /* ---- slash-command palette ---------------------------------------------- */

  // The palette opens only while the command token itself is being typed.
  // Once a space (or another line) appears the user is writing arguments, so
  // the list gets out of the way instead of hijacking Enter.
  function slashQuery(value) {
    const match = /^\s*\/([A-Za-z0-9_-]*)$/.exec(String(value ?? ''));
    return match ? match[1].toLowerCase() : null;
  }

  async function loadCommandCatalog() {
    if (Array.isArray(state.commands) || state.commandsLoading) return;
    state.commandsLoading = true;
    try {
      const payload = await api('/commands');
      state.commands = Array.isArray(payload?.commands) ? payload.commands : [];
    } catch {
      // An unavailable catalog must not break the composer; keep it empty so a
      // failed fetch is not repeated on every keystroke.
      state.commands = [];
    } finally {
      state.commandsLoading = false;
    }
    if (slashQuery($('message').value) !== null) renderCommandPalette();
  }

  function commandMatches(query) {
    const catalog = Array.isArray(state.commands) ? state.commands : [];
    return catalog
      // The backend labels entries the chat surface cannot run. Offering them
      // would promise a command that dispatch refuses.
      .filter((cmd) => cmd && cmd.allowed_for_caller !== false)
      .filter((cmd) => {
        if (!query) return true;
        const names = [cmd.name, ...(Array.isArray(cmd.aliases) ? cmd.aliases : [])];
        return names.some((name) => String(name).toLowerCase().startsWith(query));
      })
      .slice(0, SLASH_PALETTE_LIMIT);
  }

  // Once a command name is typed, offer the fixed choices its first argument
  // declares (e.g. /think -> none, minimal, low, medium, high, xhigh, max).
  // Only the token under the cursor is completed; free-form commands return
  // null and keep the normal command-name palette behaviour.
  function argumentQuery(value) {
    const raw = String(value ?? '');
    const match = /^\s*\/([A-Za-z0-9_-]+)\s+(.*)$/.exec(raw);
    if (!match) return null;
    const tokenMatch = /(?:^|\s)([A-Za-z0-9_-]*)$/.exec(match[2]);
    if (!tokenMatch) return null;
    const catalog = Array.isArray(state.commands) ? state.commands : [];
    const cmd = catalog.find((c) => {
      if (!c || c.allowed_for_caller === false) return false;
      const names = [c.name, ...(Array.isArray(c.aliases) ? c.aliases : [])];
      return names.some((n) => String(n).toLowerCase() === match[1].toLowerCase());
    });
    if (!cmd) return null;
    const args = Array.isArray(cmd.arguments) ? cmd.arguments : [];
    const spec = args.find((a) => a && Array.isArray(a.choices) && a.choices.length);
    if (!spec) return null;
    return { token: tokenMatch[1].toLowerCase(), choices: spec.choices, hint: cmd.args_hint || '' };
  }

  function argumentMatches(aq) {
    return aq.choices
      .filter((choice) => String(choice).toLowerCase().startsWith(aq.token))
      .slice(0, SLASH_PALETTE_LIMIT)
      .map((choice) => ({
        _option: true, name: String(choice), description: aq.hint,
      }));
  }

  function hideCommandPalette() {
    const palette = $('command-palette');
    if (!palette) return;
    palette.hidden = true;
    palette.replaceChildren();
    state.paletteMatches = [];
    const input = $('message');
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
  }

  function setPaletteActive(index) {
    const palette = $('command-palette');
    if (!palette) return;
    const items = Array.from(palette.querySelectorAll('.chat-palette__item'));
    if (!items.length) return;
    state.paletteIndex = Math.max(0, Math.min(index, items.length - 1));
    items.forEach((item, i) => {
      const active = i === state.paletteIndex;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    const active = items[state.paletteIndex];
    $('message').setAttribute('aria-activedescendant', active.id);
    active.scrollIntoView({ block: 'nearest' });
  }

  function commandPaletteItem(cmd, index) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'chat-palette__item';
    item.id = `command-palette-${index}`;
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', 'false');
    const head = document.createElement('span');
    head.className = 'chat-palette__head';
    const name = document.createElement('span');
    name.className = 'chat-palette__name';
    name.textContent = cmd._option ? cmd.name : `/${cmd.name}`;
    head.appendChild(name);
    if (!cmd._option && cmd.args_hint) {
      const args = document.createElement('span');
      args.className = 'chat-palette__args';
      args.textContent = cmd.args_hint;
      head.appendChild(args);
    }
    const help = document.createElement('span');
    help.className = 'chat-palette__help';
    help.textContent = cmd._option ? (cmd.description || '') : (cmd.help || '');
    item.append(head, help);
    // mousedown would blur the textarea and close the list before click lands.
    item.addEventListener('mousedown', (event) => event.preventDefault());
    item.addEventListener('click', () => acceptCommandPalette(index));
    return item;
  }

  function renderCommandPalette() {
    const input = $('message');
    const palette = $('command-palette');
    if (!palette || !input) return;
    const query = slashQuery(input.value);
    const aq = query === null ? argumentQuery(input.value) : null;
    if (query === null && aq === null) { hideCommandPalette(); return; }
    if (!Array.isArray(state.commands)) {
      hideCommandPalette();
      loadCommandCatalog();
      return;
    }
    const matches = aq ? argumentMatches(aq) : commandMatches(query);
    if (!matches.length) { hideCommandPalette(); return; }
    state.paletteMatches = matches;
    if (state.paletteIndex >= matches.length) state.paletteIndex = 0;
    palette.replaceChildren(...matches.map((cmd, index) => commandPaletteItem(cmd, index)));
    palette.hidden = false;
    input.setAttribute('aria-expanded', 'true');
    setPaletteActive(state.paletteIndex);
  }

  function movePalette(delta) {
    const palette = $('command-palette');
    if (!palette || palette.hidden || !state.paletteMatches.length) return;
    const count = state.paletteMatches.length;
    setPaletteActive((state.paletteIndex + delta + count) % count);
  }

  function acceptCommandPalette(index = state.paletteIndex) {
    const palette = $('command-palette');
    if (!palette || palette.hidden) return false;
    const cmd = state.paletteMatches[index];
    if (!cmd) return false;
    const input = $('message');
    if (cmd._option) {
      // Replace only the token under the cursor, preserving the command name
      // and any earlier arguments already typed.
      const raw = String(input.value ?? '');
      const parts = /^(.*?)([A-Za-z0-9_-]*)$/.exec(raw);
      input.value = `${parts[1]}${cmd.name} `;
    } else {
      input.value = `/${cmd.name} `;
    }
    hideCommandPalette();
    autosize();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    return true;
  }

  /* ---- model dialog (ModelInfoCard port) ------------------------------------------ */

  function openModelDialog(health) {
    document.querySelectorAll('.chat-model-backdrop').forEach((n) => n.remove());
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop chat-model-backdrop';
    const dialog = document.createElement('div');
    dialog.className = 'modal chat-model-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.innerHTML = `
      <div class="modal__head">
        <div><h3 class="modal__title">Model &amp; providers</h3>
        <p class="modal__text">Choose the default model. The change applies to the next turn and is written to leo.yaml.</p></div>
        <button type="button" class="icon-btn" aria-label="Close dialog">${ico('x')}</button>
      </div>
      <div class="chat-provider-list"></div>
      <div class="modal__actions"><button type="button" class="btn btn--secondary" data-close>Close</button></div>`;
    const list = dialog.querySelector('.chat-provider-list');
    const providers = health.providers || {};
    // A catalog can be long. Make the scrollport keyboard-reachable and let a
    // wheel anywhere over the dialog drive it, not just over the rows.
    list.tabIndex = 0;
    list.setAttribute('aria-label', 'Configured providers');
    dialog.addEventListener('wheel', (e) => {
      if (e.target.closest('.chat-provider-list')) return;
      if (list.scrollHeight <= list.clientHeight) return;
      list.scrollTop += e.deltaY;
      e.preventDefault();
    }, { passive: false });
    const count = Object.keys(providers).length;
    const textEl = dialog.querySelector('.modal__text');
    if (textEl) {
      const upstream = health.provider ? ` on ${health.provider}` : '';
      textEl.textContent = `${count} model${count === 1 ? '' : 's'}${upstream} configured · click one to switch. The choice is saved to leo.yaml.`;
    }

    let selecting = false;
    const rows = [];

    async function chooseProvider(name) {
      if (selecting) return;
      selecting = true;
      list.setAttribute('aria-busy', 'true');
      rows.forEach((button) => { button.disabled = true; });
      try {
        const result = await api('/models/default', {
          method: 'POST',
          body: JSON.stringify({ provider: name }),
        });
        if (state.health) state.health.default_provider = name;
        updateModelCard(state.health || {});
        rows.forEach((button) => {
          const active = button.dataset.provider === name;
          button.classList.toggle('is-active', active);
          button.setAttribute('aria-pressed', String(active));
          const badge = button.querySelector('.badge--accent');
          if (active && !badge) {
            button.insertAdjacentHTML('beforeend', '<span class="badge badge--accent">default</span>');
          } else if (!active && badge) {
            badge.remove();
          }
        });
        toast(result.persisted ? `Default model: ${name}` : `Using ${name} for this session`, result.persisted ? 'success' : 'warn');
        if (!result.persisted && result.warning) toast(result.warning, 'warn');
        // Keep the dashboard header status in sync without waiting for the poll.
        leo().refresh?.();
      } catch (err) {
        toast(`Could not switch model: ${err.message}`, 'danger');
      } finally {
        selecting = false;
        list.removeAttribute('aria-busy');
        rows.forEach((button) => { button.disabled = false; });
      }
    }

    Object.entries(providers).forEach(([name, info]) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'chat-provider chat-provider--select';
      row.dataset.provider = name;
      row.title = `Use ${name}`;
      const active = name === health.default_provider;
      row.classList.toggle('is-active', active);
      row.setAttribute('aria-pressed', String(active));
      const label = document.createElement('span');
      label.className = 'chat-provider__name';
      label.textContent = name;
      const model = document.createElement('span');
      model.className = 'chat-provider__model';
      model.textContent = typeof info === 'string' ? info : (info?.model || '—');
      row.append(label, model);
      if (active) {
        row.insertAdjacentHTML('beforeend', '<span class="badge badge--accent">default</span>');
      }
      row.addEventListener('click', () => chooseProvider(name));
      rows.push(row);
      list.appendChild(row);
    });
    if (!Object.keys(providers).length) {
      list.innerHTML = '<p class="hint">No providers configured yet.</p>';
    }
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);
    const close = () => { backdrop.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    dialog.querySelector('.icon-btn').addEventListener('click', close);
    dialog.querySelector('[data-close]').addEventListener('click', close);
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
  }

  /* ---- wiring ---------------------------------------------------------------------- */

  function wire() {
    // sessions
    $('chat-new').addEventListener('click', () => selectConversation(null));
    $('chat-sessions-refresh').addEventListener('click', refreshSessions);
    $('chat-search').addEventListener('input', (e) => { state.search = e.target.value; renderSessions(); });

    // composer
    $('chat-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const input = $('message');
      const text = input.value.trim();
      if (state.running) {
        // Interrupt-class input is handled inline; anything else typed mid-turn
        // is left in the composer (there is no prompt queue on this surface).
        if (isInterruptCommand(text)) {
          input.value = '';
          hideCommandPalette();
          autosize();
          await requestStop(text);
        }
        return;
      }
      if (!text && !state.attachments.length) return;
      input.value = '';
      hideCommandPalette();
      autosize();
      await streamChat(text || 'Please review the attached file.');
      input.focus();
    });
    $('message').addEventListener('input', () => { autosize(); renderCommandPalette(); });
    $('message').addEventListener('keydown', (e) => {
      const palette = $('command-palette');
      if (palette && !palette.hidden) {
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
          movePalette(e.key === 'ArrowDown' ? 1 : -1);
          return;
        }
        if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey && !e.isComposing)) {
          if (acceptCommandPalette()) {
            e.preventDefault();
            e.stopPropagation();
            return;
          }
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();  // the global Escape handler stops a running turn
          hideCommandPalette();
          return;
        }
      }
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        $('chat-form').requestSubmit();
      }
    });
    $('attach-btn').addEventListener('click', () => $('file-input').click());
    $('file-input').addEventListener('change', (e) => { handleFiles(e.target.files); e.target.value = ''; });
    $('stop-btn').addEventListener('click', async () => {
      const run = state.activeRun;
      if (run && run !== 'pending') {
        try { await api(`/runs/${encodeURIComponent(run)}/cancel`, { method: 'POST' }); } catch { /* abort below still applies */ }
      }
      if (state.abortController) state.abortController.abort();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && state.running && $('modal-backdrop').hidden && !document.querySelector('.chat-confirm')) {
        $('stop-btn').click();
      }
    });

    // drag & drop + paste anywhere on the chat pane
    const main = document.querySelector('.chat-main');
    main.addEventListener('dragover', (e) => { e.preventDefault(); main.classList.add('is-drop-target'); });
    main.addEventListener('dragleave', () => main.classList.remove('is-drop-target'));
    main.addEventListener('drop', (e) => {
      e.preventDefault(); main.classList.remove('is-drop-target');
      if (e.dataTransfer && e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    });
    main.addEventListener('paste', (e) => {
      const files = Array.from(e.clipboardData?.files || []);
      if (files.length) { e.preventDefault(); handleFiles(files); }
    });

    // side panel toggles
    const shell = $('chat-shell');
    if (state.panelHidden) shell.classList.add('chat-side-hidden');
    $('chat-panel-toggle').addEventListener('click', () => {
      state.panelHidden = !state.panelHidden;
      shell.classList.toggle('chat-side-hidden', state.panelHidden);
      sessionStorage.setItem('leo_chat_panel_hidden', state.panelHidden ? '1' : '0');
    });
    $('chat-history-toggle').addEventListener('click', () => {
      shell.classList.toggle('chat-history-open');
    });
    $('chat-model-btn').addEventListener('click', () => openModelDialog(state.health || {}));

    // dashboard data refresh hook (dispatched by app.js after each poll)
    window.addEventListener('leo:data', (e) => {
      const data = e.detail || {};
      state.health = data.health || {};
      updateModelCard(state.health);
      // Decisions taken in the Approvals view arrive here after its poll.
      syncPendingApprovals();
    });
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) syncPendingApprovals();
    });

    attachScrollPinning();

    // jump-to-latest affordance
    const jump = document.createElement('button');
    jump.type = 'button';
    jump.id = 'chat-jump';
    jump.className = 'btn btn--secondary btn--sm';
    jump.hidden = true;
    jump.style.cssText = 'position:sticky;bottom:12px;align-self:center;z-index:5;margin:0 auto';
    jump.textContent = 'Jump to latest ↓';
    jump.addEventListener('click', () => { state.scrollPinned = true; maybeScroll(true); jump.hidden = true; });
    scrollEl().appendChild(jump);
  }

  /* ---- boot -------------------------------------------------------------------------- */

  function boot() {
    if (!window.LEOMarkdown) { setTimeout(boot, 60); return; }
    wire();
    setChatStatus('live', 'Ready');
    updateModelCard({});
    refreshSessions().then(async () => {
      if (state.activeId) await selectConversation(state.activeId, { silent: true });
      else { renderEmpty(); await syncPendingApprovals(); }
    });
    setInterval(refreshSessions, SESSION_REFRESH_MS);
    connectEventStream();
    autosize();
    $('message').focus();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
