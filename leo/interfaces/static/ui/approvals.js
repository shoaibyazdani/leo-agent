// Frozen ApprovalCard v1 renderer shared by the web dashboard and the
// Electron shell. It renders the literal full body with textContent (never
// innerHTML), offers every choice the card froze, and POSTs the proposal
// digest the human actually saw. It holds no approval authority: the server
// re-checks shown state, deadline, digest, choice and identity.
//
// Phase 9 wires this module into the dashboard. Phase 4 ships the projection
// and its contract so the renderer cannot drift from the server DTO.
(function () {
  'use strict';

  const ACK_ROUTE = (id) => `/approvals/${encodeURIComponent(id)}/ack`;
  const RESOLVE_ROUTE = (id) => `/approvals/${encodeURIComponent(id)}/resolve`;
  // Browser view budget. Splitting is lossless: joining every page reproduces
  // the frozen body exactly, and every page is inserted with textContent.
  const BODY_PAGE_CHARS = 1200;

  function paginateText(text, limit = BODY_PAGE_CHARS) {
    const source = String(text == null ? '' : text);
    const size = Math.max(1, Number(limit) || BODY_PAGE_CHARS);
    if (source.length <= size) return [source];
    const pages = [];
    let rest = source;
    while (rest) {
      if (rest.length <= size) { pages.push(rest); break; }
      const window = rest.slice(0, size);
      const cut = Math.max(
        window.lastIndexOf('\n') + 1,
        window.lastIndexOf('. ') + 2,
        window.lastIndexOf('; ') + 2,
      );
      const take = cut >= size / 2 ? cut : size;
      pages.push(rest.slice(0, take));
      rest = rest.slice(take);
    }
    return pages;
  }

  function esc(value) {
    return String(value == null ? '' : value);
  }

  // Plain projection: no DOM, no network, no authority.
  function scheduleCardView(card) {
    const source = card && typeof card === 'object' ? card : {};
    const values = Array.isArray(source.choices) ? source.choices : [];
    const labels = Array.isArray(source.choice_labels) ? source.choice_labels : [];
    // Accept both the raw ApprovalCard shape (string choices + choice_labels)
    // and the shared ScheduleCardView shape (choices as {value,label}).
    const choices = values.map((item, index) => {
      if (item && typeof item === 'object') {
        return { value: esc(item.value), label: esc(item.label || item.value) };
      }
      return {
        value: esc(item),
        label: esc(index < labels.length ? labels[index] : item),
      };
    });
    return {
      protocol_version: Number(source.protocol_version || 1),
      capability_version: Number(source.capability_version || 1),
      approval_id: esc(source.approval_id),
      proposal_id: source.proposal_id == null ? null : esc(source.proposal_id),
      kind: esc(source.kind || 'command'),
      status: esc(source.status || 'pending'),
      title: esc(source.title || 'Approval needed'),
      body: esc(source.body || ''),
      choices,
      deny_only: source.deny_only === true,
      deadline: esc(source.deadline || ''),
      proposal_digest: esc(source.proposal_digest || ''),
      card_revision: Number(source.card_revision || 1),
      expires_at: source.expires_at == null ? null : esc(source.expires_at),
      shown_first_run: source.shown_first_run == null ? null : esc(source.shown_first_run),
      outcome: source.outcome == null ? null : esc(source.outcome),
      deny_reason: source.deny_reason == null ? null : esc(source.deny_reason),
      conversation_id: source.conversation_id == null ? null : esc(source.conversation_id),
      origin: source.origin == null ? null : esc(source.origin),
    };
  }

  // DOM rendering uses textContent for every literal value. A prompt that
  // contains markup stays text; it is never interpreted as HTML.
  function renderCard(container, card, options = {}) {
    const view = scheduleCardView(card);
    const onChoice = typeof options.onChoice === 'function' ? options.onChoice : null;
    container.replaceChildren();

    const title = document.createElement('h3');
    title.className = 'approval-card__title';
    title.textContent = view.title;
    container.appendChild(title);

    const body = document.createElement('pre');
    body.className = 'approval-card__body';
    const pages = paginateText(view.body);
    if (pages.length === 1) {
      body.textContent = view.body;
    } else {
      body.textContent = pages[0];
      const more = document.createElement('button');
      more.className = 'approval-card__more btn btn--sm';
      more.type = 'button';
      let shown = 1;
      more.textContent = `Show next part (${shown}/${pages.length})`;
      more.addEventListener('click', () => {
        // Append, never replace: every character stays visible as text.
        body.appendChild(document.createTextNode(pages[shown++]));
        if (shown >= pages.length) more.remove();
        else more.textContent = `Show next part (${shown}/${pages.length})`;
      });
      container.appendChild(body);
      container.appendChild(more);
    }
    if (pages.length === 1) container.appendChild(body);

    if (view.deadline) {
      const deadline = document.createElement('p');
      deadline.className = 'approval-card__deadline';
      deadline.textContent = view.deadline;
      container.appendChild(deadline);
    }

    const actions = document.createElement('div');
    actions.className = 'approval-card__actions';
    view.choices.forEach((choice) => {
      const button = document.createElement('button');
      button.className = 'btn btn--sm';
      button.type = 'button';
      button.dataset.choice = choice.value;
      button.dataset.approvalId = view.approval_id;
      button.textContent = choice.label;
      if (onChoice) button.addEventListener('click', () => onChoice(view, choice.value));
      actions.appendChild(button);
    });
    container.appendChild(actions);

    if (view.deny_reason) {
      const reason = document.createElement('p');
      reason.className = 'approval-card__deny-reason';
      reason.textContent = `Reason: ${view.deny_reason}`;
      container.appendChild(reason);
    }

    const meta = document.createElement('p');
    meta.className = 'approval-card__meta';
    meta.textContent = `id: ${view.approval_id} · kind: ${view.kind} · digest: ${view.proposal_digest}`;
    container.appendChild(meta);
    return view;
  }

  async function requestJson(url, body) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : response.statusText;
      const error = new Error(typeof detail === 'string' ? detail : 'Approval decision failed');
      error.status = response.status;
      error.body = payload;
      throw error;
    }
    return payload;
  }

  function acknowledge(card) {
    const view = scheduleCardView(card);
    if (!view.approval_id) return Promise.reject(new Error('card has no approval id'));
    return requestJson(ACK_ROUTE(view.approval_id), {});
  }

  function resolve(card, choice) {
    const view = scheduleCardView(card);
    if (!view.approval_id) return Promise.reject(new Error('card has no approval id'));
    const offered = view.choices.some((item) => item.value === choice);
    if (!offered) return Promise.reject(new Error('that choice is not offered on this card'));
    // A web button carries no free text. Ask for a bounded reason when the
    // human denies, and record the gap explicitly if the prompt is dismissed.
    let denyReason = null;
    if (choice === 'deny' && typeof window !== 'undefined' && typeof window.prompt === 'function') {
      try {
        const supplied = window.prompt('Reason for denying this schedule (optional, max 500 characters):');
        const trimmed = supplied == null ? '' : String(supplied).trim().slice(0, 500);
        denyReason = trimmed || 'no reason supplied';
      } catch (_) {
        denyReason = null;  // a shell without prompts still records the gap
      }
    }
    // Only the digest displayed by this renderer is sent back. The server
    // compares it to the durable row and refuses a stale card.
    return requestJson(RESOLVE_ROUTE(view.approval_id), {
      choice,
      confirm_proposal_digest: view.proposal_digest,
      ...(denyReason ? { deny_reason: denyReason } : {}),
    });
  }

  // Presentation receipt first, decision second. The server still owns every
  // check: this only orders the two calls a human-facing client must make.
  async function acknowledgeThenResolve(card, choice) {
    await acknowledge(card);
    return resolve(card, choice);
  }

  // A decided card keeps its kind and full literal text. This replaces the
  // live card in place; nothing here decides anything.
  function renderTerminal(container, card, status = null) {
    const view = scheduleCardView(card);
    container.replaceChildren();
    const title = document.createElement('h3');
    title.className = 'approval-card__title';
    title.textContent = view.title;
    container.appendChild(title);
    const body = document.createElement('pre');
    body.className = 'approval-card__body';
    body.textContent = view.body;
    container.appendChild(body);
    if (view.deny_reason) {
      const reason = document.createElement('p');
      reason.className = 'approval-card__deny-reason';
      reason.textContent = `reason: ${view.deny_reason}`;
      container.appendChild(reason);
    }
    const verdict = document.createElement('p');
    verdict.className = 'approval-card__terminal';
    const state = String(status || view.status || '');
    const label = state === 'approved' ? '✅ approved'
      : state === 'expired' ? '⌛ expired'
        : state === 'resolved' ? 'decision recorded' : '⛔ denied';
    verdict.textContent = `${label} · ${view.outcome || state}`;
    container.appendChild(verdict);
    return view;
  }

  window.LEO_APPROVALS = Object.freeze({
    scheduleCardView,
    renderCard,
    renderTerminal,
    acknowledge,
    acknowledgeThenResolve,
    resolve,
    paginateText,
    ACK_ROUTE,
    RESOLVE_ROUTE,
    BODY_PAGE_CHARS,
  });
})();
