/* Live agent activity view. Untrusted transcript text is only ever inserted
   with textContent; no data string is evaluated as HTML or code. */
(() => {
  "use strict";

  const REFRESH_MS = 2000;
  const PAGE_SIZE = 50;
  const STATUS_LABELS = { recent: "Recent", completed: "Completed", idle: "Idle" };
  const SVG_NS = "http://www.w3.org/2000/svg";

  const el = (id) => document.getElementById(id);
  const state = {
    status: "all",
    role: "",
    q: "",
    limit: PAGE_SIZE,
    offset: 0,
    total: 0,
    rows: [],
    selectedId: "",
    paused: false,
    controller: null,
    searchTimer: null,
  };

  function setConnection(text, kind) {
    const node = el("connection-state");
    node.textContent = text;
    node.setAttribute("data-state", kind || "ok");
  }

  function showError(message) {
    el("error-message").textContent = message;
    el("error-state").hidden = false;
    el("empty-state").hidden = true;
    setConnection("Update failed", "error");
  }

  function clearNotice() {
    el("error-state").hidden = true;
  }

  function formatTime(value) {
    if (!value || typeof value !== "string") return "—";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value.slice(0, 19).replace("T", " ");
    return parsed.toLocaleString([], { hour12: false, year: "2-digit", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  }

  function safeStatus(value) {
    return Object.prototype.hasOwnProperty.call(STATUS_LABELS, value) ? value : "idle";
  }

  function buildBadge(value) {
    const status = safeStatus(value);
    const badge = document.createElement("span");
    badge.className = "badge badge--" + status;
    badge.setAttribute(
      "aria-label",
      status === "recent"
        ? "Status: Recent — file changed within 90 seconds; not a live heartbeat"
        : "Status: " + STATUS_LABELS[status]
    );
    badge.title = status === "recent" ? "mtime-based, not a live process check" : STATUS_LABELS[status];
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", "#status-" + status);
    svg.appendChild(use);
    badge.appendChild(svg);
    badge.appendChild(document.createTextNode(STATUS_LABELS[status]));
    return badge;
  }

  function buildCell(text, className) {
    const cell = document.createElement("td");
    if (className) cell.className = className;
    cell.textContent = text == null || text === "" ? "—" : String(text);
    return cell;
  }

  function renderRoles(data) {
    const nav = el("role-filters");
    nav.replaceChildren();
    const counts = data.role_counts && typeof data.role_counts === "object" ? data.role_counts : {};
    const roles = Array.isArray(data.roles) ? data.roles : [];
    if (roles.length === 0) {
      const empty = document.createElement("p");
      empty.className = "sidebar__loading";
      empty.textContent = "No roles reported yet.";
      nav.appendChild(empty);
      return;
    }
    for (const role of roles) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "role-filter";
      button.setAttribute("aria-label", "Filter by role " + role);
      button.setAttribute("aria-pressed", state.role === role ? "true" : "false");
      button.dataset.role = role;
      const label = document.createElement("span");
      label.textContent = role;
      const count = document.createElement("span");
      count.className = "role-filter__count";
      count.textContent = String(counts[role] == null ? 0 : counts[role]);
      button.append(label, count);
      button.addEventListener("click", () => {
        state.role = state.role === role ? "" : role;
        state.offset = 0;
        renderRoles(data);
        refresh();
      });
      nav.appendChild(button);
    }
  }

  function renderRows(data) {
    const rows = Array.isArray(data.agents) ? data.agents : [];
    state.rows = rows;
    state.total = Number(data.total) || 0;
    const body = el("activity-rows");
    body.replaceChildren();

    for (const row of rows) {
      const tr = document.createElement("tr");
      const agentCell = document.createElement("td");
      const open = document.createElement("button");
      open.type = "button";
      open.className = "agent-button";
      open.textContent = row.agent_id || "agent";
      open.setAttribute("aria-label", "View details for agent " + (row.agent_id || "unknown"));
      open.addEventListener("click", () => openDetail(row.agent_id));
      agentCell.appendChild(open);
      tr.appendChild(agentCell);
      tr.appendChild(buildCell(row.role, "role-cell"));
      const statusCell = document.createElement("td");
      statusCell.appendChild(buildBadge(row.status));
      tr.appendChild(statusCell);
      tr.appendChild(buildCell(Array.isArray(row.findings) ? row.findings.length : 0, "numeric"));
      tr.appendChild(buildCell(Array.isArray(row.fixes) ? row.fixes.length : 0, "numeric"));
      tr.appendChild(buildCell(formatTime(row.started_at), "cell-time"));
      tr.appendChild(buildCell(formatTime(row.finished_at), "cell-time"));
      tr.appendChild(buildCell(formatTime(row.last_activity_at), "cell-time"));
      body.appendChild(tr);
    }

    const from = state.total === 0 ? 0 : state.offset + 1;
    const to = Math.min(state.offset + rows.length, state.total);
    el("page-info").textContent = from + "–" + to + " of " + state.total + " agents";
    el("prev").disabled = state.offset === 0;
    el("next").disabled = state.offset + state.limit >= state.total;

    const empty = el("empty-state");
    if (state.total === 0) {
      empty.hidden = false;
      el("empty-message").textContent = state.q || state.role || state.status !== "all"
        ? "No agents match these filters."
        : "No agent activity yet. Local transcripts will appear here when agents run.";
    } else {
      empty.hidden = true;
    }
  }

  function renderSummary(data) {
    const values = [
      ["stat-recent", data.recent],
      ["stat-completed", data.completed],
      ["stat-findings", data.total_findings],
      ["stat-fixes", data.total_fixes],
    ];
    for (const [id, value] of values) {
      el(id).textContent = typeof value === "number" ? String(value) : "—";
    }
  }

  function renderList(data) {
    if (data.provenance_note) el("provenance-note").textContent = data.provenance_note;
    renderRoles(data);
    renderRows(data);
  }

  function addEvidenceList(container, items, emptyText) {
    container.replaceChildren();
    if (!Array.isArray(items) || items.length === 0) {
      const empty = document.createElement("p");
      empty.className = "detail-empty";
      empty.textContent = emptyText;
      container.appendChild(empty);
      return;
    }
    for (const item of items) {
      const pre = document.createElement("pre");
      pre.className = "detail-item";
      pre.textContent = typeof item === "string" ? item : JSON.stringify(item);
      container.appendChild(pre);
    }
  }

  function fillDetail(row) {
    el("detail-title").textContent = row.agent_id || "agent";
    el("detail-role").textContent = (row.role || "role unknown") + " · " + (STATUS_LABELS[safeStatus(row.status)] || "Idle");
    el("detail-prompt").textContent = row.prompt_excerpt || "No prompt recorded.";
    addEvidenceList(el("detail-findings"), row.findings, "No findings reported yet.");
    addEvidenceList(el("detail-fixes"), row.fixes, "No fixes recorded yet.");
    const tests = [];
    if (row.test_result) tests.push("Test result: " + row.test_result);
    if (row.blocker) tests.push("Blocker: " + row.blocker);
    el("detail-tests").textContent = tests.length ? tests.join("\n\n") : "No test result or blocker reported.";
    el("detail-provenance").textContent = [
      "Status: " + (STATUS_LABELS[safeStatus(row.status)] || "Idle"),
      "Basis: " + (row.status_basis || "mtime-only (no heartbeat or PID available)"),
      "Source: " + (row.source_kind || "unknown"),
      "Trust: " + (row.trust || "unverified"),
    ].join("\n");
    el("detail-report").textContent = row.report_text || "No final report yet — this agent is still working or stopped before reporting.";
  }

  function openDetail(agentId) {
    const row = state.rows.find((item) => item.agent_id === agentId);
    if (!row) return;
    state.selectedId = agentId;
    state.returnFocus = document.activeElement;
    fillDetail(row);
    const drawer = el("detail");
    drawer.hidden = false;
    drawer.setAttribute("aria-hidden", "false");
    el("close-detail").focus();
  }

  function closeDetail() {
    const drawer = el("detail");
    drawer.hidden = true;
    drawer.setAttribute("aria-hidden", "true");
    state.selectedId = "";
    if (state.returnFocus && typeof state.returnFocus.focus === "function") state.returnFocus.focus();
  }

  async function refresh() {
    if (document.hidden || state.paused) return;
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    setConnection("Updating…", "ok");
    const params = new URLSearchParams({
      limit: String(state.limit),
      offset: String(state.offset),
    });
    if (state.status !== "all") params.set("status", state.status);
    if (state.role) params.set("role_contains", state.role);
    if (state.q) params.set("q", state.q);

    try {
      const [listResponse, summaryResponse] = await Promise.all([
        fetch("/agent-activity?" + params.toString(), { signal: controller.signal, headers: { Accept: "application/json" } }),
        fetch("/agent-activity/summary", { signal: controller.signal, headers: { Accept: "application/json" } }),
      ]);
      if (!listResponse.ok || !summaryResponse.ok) throw new Error("HTTP " + listResponse.status);
      const [data, summary] = await Promise.all([listResponse.json(), summaryResponse.json()]);
      if (controller.signal.aborted) return;
      clearNotice();
      renderRows(data);
      renderSummary(summary);
      renderRoles(data);
      if (state.selectedId) {
        const selected = data.agents.find((item) => item.agent_id === state.selectedId);
        if (selected) fillDetail(selected);
      }
      el("last-updated").textContent = "Last updated: " + new Date().toLocaleTimeString([], { hour12: false });
      setConnection("Live", "ok");
    } catch (error) {
      if (error && error.name === "AbortError") return;
      showError("Could not load agent activity. " + (error && error.message ? String(error.message) : ""));
    }
  }

  el("pause").addEventListener("click", () => {
    state.paused = !state.paused;
    el("pause").setAttribute("aria-pressed", state.paused ? "true" : "false");
    el("pause").setAttribute("aria-label", state.paused ? "Resume automatic refresh" : "Pause automatic refresh");
    el("pause-icon-use").setAttribute("href", state.paused ? "#icon-play" : "#icon-pause");
    if (state.paused) {
      if (state.controller) state.controller.abort();
      setConnection("Paused", "ok");
    } else {
      refresh();
    }
  });
  el("refresh").addEventListener("click", () => refresh());
  el("retry").addEventListener("click", () => refresh());
  el("close-detail").addEventListener("click", closeDetail);
  el("prev").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.limit);
    refresh();
  });
  el("next").addEventListener("click", () => {
    state.offset += state.limit;
    refresh();
  });
  el("search").addEventListener("input", (event) => {
    state.q = event.target.value.trim();
    state.offset = 0;
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(refresh, 250);
  });
  for (const radio of document.querySelectorAll('input[name="status"]')) {
    radio.addEventListener("change", (event) => {
      if (event.target.checked) {
        state.status = event.target.value;
        state.offset = 0;
        refresh();
      }
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !el("detail").hidden) closeDetail();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (state.controller) state.controller.abort();
      if (!state.paused) setConnection("Paused — tab hidden", "ok");
    } else if (!state.paused) {
      refresh();
    }
  });
  window.addEventListener("pagehide", () => {
    if (state.controller) state.controller.abort();
  });

  renderSummary({});
  refresh();
  window.setInterval(refresh, REFRESH_MS);
})();
