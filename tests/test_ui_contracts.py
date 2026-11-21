"""Static contracts for the conversation UI fixes.

The dashboard is plain browser JavaScript, so these tests read the shipped
assets and assert the behavioural contracts that cannot be verified by the
backend test suite: read-only approval blockers, honest turn status, real
modal output, truthful task labels and live config data.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "leo" / "interfaces" / "static"

INDEX = (STATIC / "index.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "ui" / "app.js").read_text(encoding="utf-8")
CHAT_JS = (STATIC / "ui" / "conversation" / "chat.js").read_text(encoding="utf-8")
COMPONENTS_CSS = (STATIC / "ui" / "components.css").read_text(encoding="utf-8")
CHAT_CSS = (STATIC / "ui" / "conversation" / "chat.css").read_text(encoding="utf-8")
ONBOARDING_JS = (STATIC / "ui" / "onboarding.js").read_text(encoding="utf-8")
ONBOARDING_CSS = (STATIC / "ui" / "onboarding.css").read_text(encoding="utf-8")


def block(source: str, start: str, end: str) -> str:
    """Return the source slice between two markers, failing if either is absent."""
    assert start in source, f"missing marker {start!r}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing closing marker {end!r}"
    return tail.split(end, 1)[0]


@pytest.mark.parametrize(
    "needle",
    [
        "chat-approval--blocked",
        "card.dataset.approvalId = id;",
        "card.dataset.toolCallId = tool_call_id;",
        "pendingApprovals: new Map()",
        "Approvals view, which POSTs /approvals/{id}/resolve.",
        "open.textContent = 'Decide in Approvals view';",
    ],
)
def test_inline_approval_is_a_read_only_blocker(needle: str) -> None:
    assert needle in CHAT_JS


def test_inline_approval_never_resolves_or_resumes_a_tool() -> None:
    node = block(CHAT_JS, "function approvalNode(", "function markApprovalResolved(")
    assert "resolve(" not in node
    assert "deny" not in node.lower()
    assert "Approve'" not in node
    assert "/approvals/" not in node
    assert "did not run" in node
    assert "cannot resume the tool" in node
    assert "chat-approval--blocked" in CHAT_CSS
    live = block(CHAT_JS, "} else if (event.type === 'tool.finished')", "} else if (event.type === 'approval.required')")
    assert "blocked.approvalPending = true;" in CHAT_JS
    assert "blocked ? 'blocked' : failed ? 'failed' : 'done'" in live


def test_pending_approvals_rehydrate_and_reconcile() -> None:
    sync = block(CHAT_JS, "async function syncPendingApprovals()", "/* ---- conversation history")
    for needle in ("api('/approvals')", "data?.approvals", "markApprovalResolved(id)",
                   "meta.tool_call_id", "[data-approval-id]", "state.pendingApprovals.has(id)"):
        assert needle in sync
    data_hook = block(CHAT_JS, "window.addEventListener('leo:data'", "document.addEventListener('visibilitychange'")
    assert "syncPendingApprovals();" in data_hook
    boot = block(CHAT_JS, "function boot()", "if (document.readyState === 'loading')")
    assert "syncPendingApprovals()" in boot
    assert "state.pendingApprovals.clear();" in CHAT_JS  # history reload rehydrates cleanly


@pytest.mark.parametrize(
    "needle",
    [
        "const HISTORY_RESULT_LIMIT = 4000",
        "message.metadata?.arguments ?? parsed.arguments ?? {}",
        "toolCallId: message.metadata?.tool_call_id || ''",
        "boundedText(argsText || '{}')",
        "tool.resultPre.textContent = boundedText(",
        "tool.statusText.textContent = 'blocked'",
    ],
)
def test_historic_tool_cards_show_bounded_real_data(needle: str) -> None:
    assert needle in CHAT_JS


def test_incomplete_and_failed_turns_are_never_success() -> None:
    finished = block(CHAT_JS, "} else if (event.type === 'message.finished')", "} else if (event.type === 'error')")
    assert "event.tool_calls_pending" in finished
    assert "event.status === 'completed'" in finished
    assert "applyTerminalStatus('incomplete')" in finished
    assert "incomplete:" in finished
    assert "setTurnState('success', 'done')" in finished
    run_finished = block(CHAT_JS, "} else if (event.type === 'run.finished')", "async function streamChat(")
    assert "applyTerminalStatus(event.status)" in run_finished
    assert "setChatStatus('live', 'Ready')" not in run_finished
    error = block(CHAT_JS, "} else if (event.type === 'error')", "} else if (event.type === 'run.finished')")
    assert "applyTerminalStatus('failed')" in error
    history = block(CHAT_JS, "let historyIncomplete = false;", "if (!thread.childElementCount) renderEmpty();")
    assert "message.metadata?.run_status === 'incomplete'" in history
    assert "not fully answered" in history


def test_finally_preserves_terminal_status_and_refetches_dropped_streams() -> None:
    finally_block = block(CHAT_JS, "} finally {", "  function setRunning(")
    assert "setChatStatus('live', 'Ready')" not in finally_block
    assert "applyTerminalStatus(state.terminalStatus)" in finally_block
    assert "state.streamDropped" in finally_block
    assert "selectConversation(droppedId, { silent: true })" in finally_block
    gate = block(CHAT_JS, "function applyStreamEvent(event)", "if (typeof event.seq === 'number')")
    assert "event.run_id !== state.activeRun" in gate
    assert "return;" in gate
    catch = block(CHAT_JS, "} catch (error) {", "} finally {")
    assert "state.streamDropped = true;" in catch


def test_upload_rejection_errors_survive_rerender() -> None:
    for needle in ("attachmentErrors: []", "function renderAttachmentErrors()",
                   "function pushAttachmentError(", "list.querySelectorAll('.attach-error')",
                   "renderAttachmentErrors();", "(rejected)"):
        assert needle in CHAT_JS
    assert "el.textContent = `${file.name}: ${error.message}`" not in CHAT_JS


def test_modal_shows_backend_output_and_only_toasts_real_success() -> None:
    assert 'id="modal-output"' in INDEX
    assert 'id="modal-output-text"' in INDEX
    assert ".modal__output" in COMPONENTS_CSS
    for needle in ("function showModalOutput(payload)", "function unwrapResult(result)",
                   "$('modal-output-text').textContent", "showModalOutput(payload)",
                   "payload && payload.success === true", "Backend did not report success"):
        assert needle in APP_JS
    modal_run = block(APP_JS, "$('modal-run').addEventListener('click'", "document.addEventListener('keydown'")
    assert "toast('Completed successfully', 'success')" not in modal_run.replace(
        "if (payload && payload.success === true) toast('Completed successfully', 'success');", ""
    )


def test_metadata_only_skills_get_a_disabled_action() -> None:
    skills = block(APP_JS, "function renderSkills()", "function renderMemory()")
    assert "s.has_handler !== true" in skills
    assert "data-skill-disabled=" in skills
    assert "disabled title=\"Metadata-only skill" in skills
    assert "Run unavailable" in skills


def test_task_creation_is_a_queue_action_with_gated_run_route() -> None:
    assert "> Queue objective</button>" in INDEX
    assert "Create task" not in INDEX
    assert "TASK_RUN_ROUTE = (id) => `/tasks/${encodeURIComponent(id)}/run`" in APP_JS
    assert "function taskRunEnabled()" in APP_JS
    assert "data-run-task=" in APP_JS
    assert "data-run-task=\"${esc(t.id)}\"" in APP_JS
    submit = block(APP_JS, "$('task-form').addEventListener('submit'", "$('memory-form').addEventListener")
    assert "Objective queued" in submit
    assert "'Task created'" not in submit


def test_config_views_use_live_data() -> None:
    assert 'id="agent-lede"' in INDEX
    agents = block(APP_JS, "function renderAgents()", "function renderSkills()")
    assert "$('agent-lede')" in agents
    assert "agents.length" in agents
    assert "twelve" not in agents
    models = block(APP_JS, "function renderProviders()", "function renderTools()")
    assert "renderProviders();" in models
    assert "state.data.health?.providers" in models
    providers_call = models.split("function renderModels()", 1)[1]
    assert providers_call.split("\n", 1)[1].lstrip().startswith("renderProviders();")
    task_statuses = block(APP_JS, "const TASK_STATUSES = [", "const EVENTS_STREAM_ROUTE")
    for status in ("created", "planning", "ready", "running", "waiting", "blocked",
                   "verifying", "completed", "failed", "cancelled"):
        assert f"'{status}'" in task_statuses
    tasks = block(APP_JS, "function renderTasks()", "function renderAgents()")
    assert "const counts = TASK_STATUSES;" in tasks
    events = block(APP_JS, "function renderEvents()", "function renderAll(")
    assert "e.created_at" in events
    assert "e.ts" not in events
    integrations = block(APP_JS, "function renderIntegrations()", "function renderEvents()")
    assert "i.kind === 'messaging'" in integrations
    assert "channel-module:" in integrations
    configure = integrations.split("const configure =", 1)[1].split("const disable =", 1)[0]
    assert "isChannel ?" in configure


def test_token_save_validates_before_storing() -> None:
    handler = block(APP_JS, "$('save-token').addEventListener('click', async () => {", "$('compact-toggle').addEventListener")
    assert "api('/status', { headers: { Authorization: `Bearer ${value}` } })" in handler
    validation = handler.index("api('/status'")
    storage = handler.index("sessionStorage.setItem('leo_api_token', value)")
    assert validation < storage
    assert "Token rejected" in handler
    assert "api() header" not in handler  # explicit header, not a token-only check
    assert "!headers.has('Authorization')" in APP_JS


def test_events_are_live_sse_with_backoff_pause_and_bounded_list() -> None:
    for needle in ("EVENTS_STREAM_ROUTE = '/events/stream'", "EVENTS_STREAM_TOKEN_ROUTE = '/events/stream-token'", "new EventSource(url)", "await streamUrl()",
                   "EVENT_RETRY_MIN_MS", "EVENT_RETRY_MAX_MS", "scheduleEventRetry()",
                   "document.hidden", "stopEventStream()", "probeEventStream()",
                   "source.addEventListener('message', onEvent)", "source.addEventListener('event', onEvent)",
                   "state.eventItems.length > EVENT_STREAM_LIMIT", "state.eventItems.length = EVENT_STREAM_LIMIT",
                   "text/event-stream", "visibleEvents()", "slice(0, EVENT_STREAM_LIMIT)"):
        assert needle in APP_JS
    assert "will stream here" not in APP_JS
    events = block(APP_JS, "function renderEvents()", "function renderAll(")
    assert "live" in events and "polling" in events


def test_event_stream_degrades_to_polling_when_route_is_unavailable() -> None:
    probe = block(APP_JS, "async function probeEventStream()", "document.addEventListener('visibilitychange'")
    assert "await connectEventStream()" in probe
    connect = block(APP_JS, "async function connectEventStream()", "async function probeEventStream()")
    assert "await streamUrl()" in connect
    assert "new EventSource(url)" in connect
    assert "state.eventStreamUnsupported = true" in connect
    assert "renderEvents();" in connect
    retry = block(APP_JS, "function scheduleEventRetry()", "async function connectEventStream()")
    assert "state.eventStreamUnsupported" in retry
    assert "setTimeout(" in retry
    assert "loadAll().then(() => { schedulePoll(); probeEventStream(); });" in APP_JS


def test_task_run_is_capability_gated_and_shows_backend_result() -> None:
    assert "function hasCapability(name)" in APP_JS
    assert "hasCapability('task_run')" in APP_JS
    run = block(APP_JS, "const runTask = e.target.closest('[data-run-task]')", "const delSchedule =")
    assert "taskRunEnabled()" in run
    assert "showResult: true" in run
    assert "TASK_RUN_ROUTE(runTask.dataset.runTask)" in run
    tasks = block(APP_JS, "function renderTasks()", "function renderAgents()")
    assert "${taskRunEnabled() ? `<button class=\"btn btn--secondary btn--sm\" data-run-task=" in tasks
    modal = block(APP_JS, "if (state.modalShowResult) {", "} else if (state.modalConfirm || state.modalSilent)")
    assert "showModalOutput(payload)" in modal
    assert "payload.success === true" in modal
    assert "payload?.error || 'Backend did not report success'" in modal


def test_agent_delegation_control_is_real_and_truthful() -> None:
    for marker in ('id="delegate-form"', 'id="delegate-agent"', 'id="delegate-objective"',
                   'id="delegate-submit"', 'id="delegate-output"', 'id="delegate-availability"'):
        assert marker in INDEX
    render = block(APP_JS, "function renderAgentDelegation()", "function renderSchedules()")
    assert "hasCapability('agent_delegate')" in render
    assert "select.disabled = !ready" in render
    assert "$('delegate-submit').disabled = !ready" in render
    handler = block(APP_JS, "$('delegate-form').addEventListener('submit'", "$('schedule-form').addEventListener")
    assert "api('/agents/delegate'" in handler
    assert "body.agent_id = agent" in handler
    assert "result?.success === true" in handler
    assert "result.task_id" in handler
    assert "Delegation did not report success" in handler
    assert "showResultIn('delegate-output', result)" in handler


def test_schedule_management_is_real_and_truthful() -> None:
    for marker in ('id="schedule-form"', 'id="schedule-name"', 'id="schedule-seconds"',
                   'id="schedule-objective"', 'id="schedule-list"', 'id="schedule-output"',
                   'id="schedule-availability"'):
        assert marker in INDEX
    render = block(APP_JS, "function renderSchedules()", "function renderChannelSend()")
    assert "hasCapability('schedule')" in render
    assert "data-delete-schedule=" in render
    assert "s.next_run_at" in render
    handler = block(APP_JS, "$('schedule-form').addEventListener('submit'", "$('send-form').addEventListener")
    assert "api('/schedules'" in handler
    assert "JSON.stringify({ name, seconds, objective })" in handler
    assert "result?.name === name || result?.id" in handler
    assert "Backend did not return a schedule" in handler
    deletion = block(APP_JS, "const delSchedule = e.target.closest('[data-delete-schedule]')", "const resolve =")
    assert "method: 'DELETE'" in deletion
    assert "encodeURIComponent(key)" in deletion
    assert "result?.deleted === key || result?.success === true" in deletion


def test_channel_send_reports_real_delivery_not_acceptance() -> None:
    for marker in ('id="send-form"', 'id="send-channel"', 'id="send-target"',
                   'id="send-text"', 'id="send-output"', 'id="send-availability"'):
        assert marker in INDEX
    render = block(APP_JS, "function renderChannelSend()", "function showResultIn(")
    assert "hasCapability('channel_send')" in render
    assert "i.status === 'configured'" in render
    assert "$('send-submit').disabled = !ready || !channels.length" in render
    handler = block(APP_JS, "$('send-form').addEventListener('submit'", "$('memory-form').addEventListener")
    assert "/integrations/${encodeURIComponent(channel)}/send" in handler
    assert "JSON.stringify({ target, text })" in handler
    assert "result?.delivered === true" in handler
    assert "Not delivered via ${channel}" in handler
    assert "showResultIn('send-output', result)" in handler


def test_electron_shell_bridge_is_consumed_with_browser_fallback() -> None:
    assert 'id="shell-status"' in INDEX
    bridge = block(APP_JS, "// ---- Electron shell bridge", "// ---- boot")
    for needle in ("window.leo", "typeof bridge.invoke !== 'function'", "status.hidden = true",
                   "bridge.onEvent(applyShellEvent)", "bridge.invoke('leo:health')",
                   "bridge.invoke('leo:api-status')", "beforeunload", "unsubscribe()",
                   "backend-error", "backend-exit", "action-failed", "startup-state"):
        assert needle in bridge
    assert "initShellBridge();" in APP_JS
    assert "state.shellHealth ? { ...h, shell: state.shellHealth } : h" in APP_JS
    assert "refreshShellHealth();" in APP_JS


def test_accessibility_and_reduced_motion_preserved() -> None:
    assert "sr-only" in INDEX
    assert 'aria-live="polite"' in INDEX
    assert "aria-label=" in INDEX
    assert "@media (prefers-reduced-motion: reduce)" in CHAT_CSS
    assert "animation: none" in CHAT_CSS


# ---- guided channel onboarding -------------------------------------------


def test_onboarding_module_is_loaded_at_runtime_without_html_changes() -> None:
    # index.html is not owned by this feature: the module injects its own
    # markup and stylesheet into the existing Integrations surface.
    loader = block(APP_JS, "  function loadOnboarding() {", "  // ---- boot")
    assert "script.src = '/assets/ui/onboarding.js'" in loader
    assert "script.dataset.leoOnboarding = '1'" in loader
    assert "window.LEO_ONBOARDING.mount()" in loader
    assert "loadOnboarding();" in APP_JS
    assert "window.LEO.refresh = () => loadAll();" in APP_JS
    assert "leo:open-raw-channel-form" in APP_JS
    mount = block(ONBOARDING_JS, "  function injectStyles() {", "  function onHostClick(")
    assert "link.href = '/assets/ui/onboarding.css'" in mount
    assert "data-ob-styles" in mount
    assert "document.getElementById('view-integrations')" in mount
    assert "host.insertBefore(root, host.querySelector('.panel'))" in mount
    assert "window.addEventListener('leo:data', renderCards)" in mount


def test_configure_button_uses_wizard_and_keeps_raw_form_fallback() -> None:
    host_click = block(ONBOARDING_JS, "  function onHostClick(", "  function hostContains(")
    assert "[data-ob-configure], [data-configure]" in host_click
    assert "event.stopPropagation()" in host_click
    assert "openWizard(channel, button)" in host_click
    fallback = block(ONBOARDING_JS, "    if (name === 'fallback') {", "\n  }\n\n  window.LEO_ONBOARDING")
    assert "leo:open-raw-channel-form" in fallback
    assert "data-ob-action=\"fallback\"" in ONBOARDING_JS
    assert "The guided field list is unavailable" in ONBOARDING_JS
    # The original modal is still wired for calls that bypass the wizard.
    click = block(APP_JS, "    const cfg = e.target.closest('[data-configure]')", "    const dis =")
    assert "openChannelConfigure(cfg.dataset.configure)" in click


def test_onboarding_uses_both_start_response_shapes_without_inventing_a_qr() -> None:
    for needle in (
        "request('/onboarding/whatsapp/start', { method: 'POST', headers: WRITE_HEADERS, body: '{}' })",
        "const WRITE_HEADERS = { 'X-Leo-Onboarding': '1' };",
        "body.qr_svg || qr.data_uri || qr.svg",
        "if (raw.startsWith('data:image/svg+xml')) return raw;",
        "if (raw.startsWith('<svg')) return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(raw)}`",
        "body.setup_url || body.qr?.payload",
        "String(body.token || '')",
        "String(body.session_id || '')",
        "codeFromUrl(setupUrl)",
        "body.status_url || (body.token ? `/onboarding/whatsapp/status?token=${encodeURIComponent(body.token)}` : '')",
        "if (!raw) return '';",  # never render an empty QR placeholder as scannable
    ):
        assert needle in ONBOARDING_JS
    qr = block(ONBOARDING_JS, "  function renderQrPanel() {", "  function startQr() {")
    assert "qr.svgSrc" in qr
    assert 'id="ob-qr-image"' in qr
    assert "alt=\"Local setup QR code. This is not a WhatsApp device-link QR.\"" in qr
    assert "No QR image was returned" in qr
    assert "qrError" in qr
    assert 'id="ob-qr-status" role="status" aria-live="polite"' in qr


def test_status_polling_and_confirmation_cover_documented_and_current_routes() -> None:
    poll = block(ONBOARDING_JS, "  function startPolling() {", "  function applyStatus(")
    assert "POLL_MS = 3000" in ONBOARDING_JS
    assert "POLL_LIMIT = 100" in ONBOARDING_JS
    assert "document.hidden" in poll
    assert "stopPolling();" in poll
    assert "error.status === 404 || error.status === 405" in poll
    assert "Live status is not available in this runtime" in poll
    confirm = block(ONBOARDING_JS, "  async function confirmQr() {", "  // ---- validation")
    assert "'/onboarding/whatsapp/confirm'" in confirm
    assert "'/onboarding/redeem'" in confirm
    assert "WRITE_HEADERS" in confirm
    assert "token: state.qr.token, session_id: state.qr.sessionId" in confirm
    assert "code: state.qr.code" in confirm
    assert "error.status !== 404 && error.status !== 405" in confirm
    assert "applyStatus(body)" in confirm
    status = block(ONBOARDING_JS, "  function statusInfo(body) {", "  function maskedCode(")
    assert "body?.next_step" in status
    assert "safeList(body?.missing)" in status
    assert "body?.configured === true" in status


def test_qr_copy_status_and_recovery_are_honest() -> None:
    assert "Scan with WhatsApp is not supported" in ONBOARDING_JS
    assert "This QR is a local setup code, not a device link" in ONBOARDING_JS
    assert "not a WhatsApp device-link QR" in ONBOARDING_JS
    assert "Open WhatsApp &gt; Linked Devices &gt; Link a Device" in ONBOARDING_JS
    assert "Cloud API credentials are still required to send" in ONBOARDING_JS
    assert "Guided QR setup is not available in this LEO runtime" in ONBOARDING_JS
    assert "maskedCode(qr.code)" in ONBOARDING_JS
    assert "'•'" in ONBOARDING_JS
    assert "function copyValue(kind)" in ONBOARDING_JS
    assert "navigator.clipboard.writeText(value)" in ONBOARDING_JS
    assert "data-ob-kind=\"code\"" in ONBOARDING_JS
    assert "data-ob-kind=\"link\"" in ONBOARDING_JS
    assert "Code expires in " in ONBOARDING_JS
    assert "Code expired — regenerate to continue" in ONBOARDING_JS
    assert 'data-ob-action="start-qr"' in ONBOARDING_JS
    assert 'data-ob-action="confirm"' in ONBOARDING_JS
    assert "console.log" not in ONBOARDING_JS
    assert "localStorage" not in ONBOARDING_JS


def test_method_choice_is_visible_without_a_fake_device_link() -> None:
    methods = block(ONBOARDING_JS, "  function stepMethod() {", "  function stepFields() {")
    assert "Meta Cloud API" in methods
    assert "Supported" in methods
    assert "Web / QR link" in methods
    assert "Not available" in methods
    assert 'data-method="cloud"' in methods
    assert 'data-method="webqr"' in methods
    assert "no WhatsApp Web bridge" in methods
    assert 'data-ob-action="method"' in methods
    assert "aria-pressed=" in methods
    assert "What do I need?" in ONBOARDING_JS
    assert '<details class="ob-guide">' in ONBOARDING_JS


def test_guided_pattern_applies_to_all_channel_forms() -> None:
    steps = block(ONBOARDING_JS, "  const STEPS =", "  const POLICY_FIELDS")
    assert "'Method', 'Details', 'Access', 'Check'" in steps
    assert "function renderDialogBody()" in ONBOARDING_JS
    assert "class=\"ob-progress\"" in ONBOARDING_JS
    assert "function fieldHTML(field)" in ONBOARDING_JS
    assert "field.hint || field.help" in ONBOARDING_JS
    assert "field.placeholder || field.example || field.default" in ONBOARDING_JS
    assert "Saved" in ONBOARDING_JS
    assert "data-ob-field=" in ONBOARDING_JS
    assert 'id="${esc(id)}-error"' in ONBOARDING_JS
    assert "showFieldErrors()" in ONBOARDING_JS
    assert "aria-invalid" in ONBOARDING_JS
    assert "An empty allowed-senders list blocks everyone" in ONBOARDING_JS
    assert "+15551234567" in ONBOARDING_JS
    assert "Phone number ID is not your phone number" in ONBOARDING_JS


def test_check_and_enable_are_offline_honest_and_refresh_the_view() -> None:
    check = block(ONBOARDING_JS, "  async function checkSettings() {", "  async function enableChannel() {")
    assert "/validate" in check
    assert "/verify" in check
    assert "validation.valid === true" in check
    assert "error.status === 404 || error.status === 405" in check
    assert "offline field check" in check
    assert "no test message was sent and no device was linked" in ONBOARDING_JS
    enable = block(ONBOARDING_JS, "  async function enableChannel() {", "  async function copyValue(")
    assert "/enable" in enable
    assert "const valid = await checkSettings()" in enable
    assert "live delivery not tested" in enable
    assert "closeDialog();" in enable
    assert "refresh();" in enable
    assert "leo:onboarding-enabled" in enable
    assert "error.body?.detail?.errors" in enable
    assert "state.details.callback_url" in ONBOARDING_JS
    assert "data-ob-kind=\"callback\"" in ONBOARDING_JS
    assert "Set this URL, and your webhook verify token" in ONBOARDING_JS


def test_onboarding_dialog_is_accessible_and_respects_apple_conventions() -> None:
    assert 'role="dialog" aria-modal="true"' in ONBOARDING_JS
    assert 'aria-labelledby="ob-dialog-title"' in ONBOARDING_JS
    assert "aria-label=\"Close setup\"" in ONBOARDING_JS
    assert "function trapFocus(event)" in ONBOARDING_JS
    assert "event.key === 'Escape'" in ONBOARDING_JS
    assert "state.opener.focus()" in ONBOARDING_JS
    assert 'class="ob-sr-only"' in ONBOARDING_JS
    assert "aria-live=\"polite\"" in ONBOARDING_JS
    for needle in ("-apple-system, BlinkMacSystemFont, \"SF Pro Text\"",
                   "@media (prefers-color-scheme: light)",
                   "@media (prefers-reduced-motion: reduce)",
                   "min-height: 28px",
                   "--space-4, 16px",
                   "font-variant-numeric: tabular-nums",
                   "@media (pointer: coarse)"):
        assert needle in ONBOARDING_CSS
    assert "animation: none !important" in ONBOARDING_CSS
    assert "box-shadow: var(--ob-focus)" in ONBOARDING_CSS
    assert "--ob-accent:" in ONBOARDING_CSS


def test_qr_panel_swap_works_from_every_placeholder_state() -> None:
    """startQr() swaps .ob-qr; loading/error branches must expose that node."""
    panel = block(ONBOARDING_JS, "  function renderQrPanel() {", "  function startQr() {")
    assert panel.count('class="ob-qr" id="ob-qr-panel"') >= 3
    assert 'if (!qr) return \'<div class="ob-qr" id="ob-qr-panel">' in panel
    assert 'if (state.qrError) return `<div class="ob-qr" id="ob-qr-panel">' in panel
    start = block(ONBOARDING_JS, "  function startQr() {", "  function startClock() {")
    assert "panel.outerHTML = renderQrPanel();" in start


def test_backend_field_errors_are_normalised_before_display() -> None:
    """The /validate route returns a list; field errors must attach to inputs."""
    normalizer = block(ONBOARDING_JS, "  function errorMap(value) {", "  function renderCheckResult(")
    assert "Array.isArray(value)" in normalizer
    assert "entry.field || entry.name" in normalizer
    check = block(ONBOARDING_JS, "  async function checkSettings() {", "  async function enableChannel() {")
    assert "state.errors = errorMap(validation.errors);" in check
    assert "showFieldErrors();" in check


def test_send_empty_state_links_back_to_guided_setup() -> None:
    assert 'id="send-connect"' in INDEX
    wiring = block(APP_JS, "  $('send-connect')?.addEventListener('click'", "  function loadOnboarding()")
    assert "window.LEO_ONBOARDING.open('whatsapp'" in wiring
    assert "leo:open-raw-channel-form" in wiring  # graceful fallback


# ---- onboarding screen regressions (empty steps / blank fields / [object Object])


def test_method_screen_normalizes_guide_steps_and_never_emits_empty_items() -> None:
    builtin = block(ONBOARDING_JS, "  const BUILTIN_GUIDES =", "  const CARD_GUIDES =")
    assert "telegram:" in builtin and "whatsapp:" in builtin and "default:" in builtin
    for channel, following in (("telegram", "whatsapp:"), ("whatsapp", "default:")):
        section = builtin.split(f"{channel}:", 1)[1].split(following, 1)[0]
        assert section.count("{ title:") == 3  # safe built-in 3-step guide

    normalizer = block(ONBOARDING_JS, "  function normalizeGuideSteps(", "  function renderGuideItems(")
    for needle in (
        "['title', 'label', 'name', 'heading']",
        "['text', 'detail', 'description', 'message', 'body', 'content']",
        "['step', 'number']",
        "BUILTIN_GUIDES[key] || BUILTIN_GUIDES.default",
        "if (!title && !detail) return;",  # no empty <li> from an empty step object
    ):
        assert needle in normalizer
    arrays = block(ONBOARDING_JS, "  function guideArray(source, depth = 0)", "  function normalizeGuideSteps(")
    assert "['steps', 'guide', 'items', 'list']" in arrays  # guide.steps / guide.guide / catalog arrays

    renderer = block(ONBOARDING_JS, "  function renderGuideItems(", "  function guideItems(")
    assert "document.createElement('li')" in renderer
    assert "strong.textContent = title" in renderer
    assert "span.textContent = `${title ? ' ' : ''}${detail}`" in renderer
    assert "if (li.textContent.trim()) host.appendChild(li);" in renderer
    assert "<li>" not in renderer

    method = block(ONBOARDING_JS, "  function stepMethod() {", "  function stepFields() {")
    assert 'data-ob-guide-steps="details"' in method
    assert 'data-ob-guide-channel="${esc(state.channel)}"' in method
    assert "<li>" not in method
    assert "hydrateGuideLists(body);" in ONBOARDING_JS
    assert "<li></li>" not in ONBOARDING_JS
    # No list item consists solely of one (possibly empty) interpolation.
    assert not re.search(r"<li>\s*\$\{[^}]*\}\s*</li>", ONBOARDING_JS)


def test_details_screen_renders_a_label_hint_placeholder_and_error_slot_per_field() -> None:
    normalizer = block(ONBOARDING_JS, "  function normalizeField(spec) {", "  // Flatten strings,")
    for needle in (
        "['name', 'key', 'id', 'field', 'env', 'variable']",
        "['label', 'title', 'name', 'key', 'id']",
        "field.hint || field.help",
        "field.placeholder || field.example || field.default",
        "['kind', 'type', 'input_type', 'control', 'format']",
        "['description', 'tooltip', 'info', 'message']",
        "__obNormalized: true",
        "Leave blank to keep the saved value",
    ):
        assert needle in normalizer

    renderer = block(ONBOARDING_JS, "  function fieldHTML(field) {", "  // ---- local setup QR")
    for needle in (
        '<label class="ob-field__label" for="${esc(id)}"',
        'aria-describedby="${esc(described)}"',
        'const described = `${id}-hint ${id}-error`',
        'id="${esc(id)}-hint"',
        'id="${esc(id)}-error"',
        'role="alert" hidden',
        'type="${inputType}"',
        "sensitive || ['secret', 'password'].includes(kind) ? 'password'",
        'class="ob-saved">Saved',
        "placeholder=\"${esc(placeholder)}\"",
    ):
        assert needle in renderer
    # One shared visible label wraps the three alternative control branches.
    for tag in ('<input class="ob-input"', '<textarea class="ob-input', '<select class="ob-input'):
        assert renderer.count(tag) == 1
    assert renderer.count('<label class="ob-field__label"') == 1
    assert '<option value="${esc(option.value)}"${selected}>${esc(labelText)}</option>' in renderer

    details = block(ONBOARDING_JS, "  function stepFields() {", "  function stepAccess() {")
    assert "asObjects(state.details?.fields)" in details
    assert "normalizeField(field, index)" in details


def test_access_screen_normalizes_choice_shapes_and_known_modes() -> None:
    options = block(ONBOARDING_JS, "  function collectOptionValues(", "  function fieldHTML(field) {")
    for needle in (
        "'options', 'choices', 'values', 'items'",
        "['value', 'id', 'key']",
        "['label', 'title', 'text', 'name']",
        "MODE_LABELS[candidate] ? candidate",
        "value === '[object Object]' || label === '[object Object]'",
        "if (!options.length && String(name) === 'mode')",
    ):
        assert needle in options
    for needle in (
        "allowlist: 'Only people on the list'",
        "open: 'Anyone'",
        "deny: 'Nobody'",
    ):
        assert needle in ONBOARDING_JS
    choice = block(ONBOARDING_JS, "  function fieldHTML(field) {", "  // ---- local setup QR")
    assert "name === 'mode' && MODE_LABELS[option.value] ? MODE_LABELS[option.value] : option.label" in choice
    access = block(ONBOARDING_JS, "  function stepAccess() {", "  function stepCheck() {")
    assert "asObjects(state.details?.fields)" in access
    assert "normalizeField(field, index)" in access
    assert "Admission mode" in access or "Admission mode" in ONBOARDING_JS


def test_hidden_steps_and_modal_stacking_prevent_ghost_text() -> None:
    content = block(ONBOARDING_JS, "  function stepContent() {", "  function stepMethod() {")
    assert 'class="ob-step" data-ob-step=' in content
    assert "hidden inert" in content  # inactive steps are display:none AND inert
    assert ".ob-step[hidden], [hidden] { display: none !important; }" in ONBOARDING_CSS
    assert ".ob-step[inert]" in ONBOARDING_CSS
    assert "background-color: var(--ob-bg)" in ONBOARDING_CSS
    assert "isolation: isolate" in ONBOARDING_CSS
    assert "select.ob-input, .ob-select { position: relative; z-index: 2;" in ONBOARDING_CSS
    assert ".ob-select option" in ONBOARDING_CSS
    assert "@media (prefers-reduced-motion: reduce)" in ONBOARDING_CSS


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_onboarding_javascript_passes_node_check() -> None:
    result = subprocess.run(
        ["node", "--check", str(STATIC / "ui" / "onboarding.js")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", ["app.js", "onboarding.js"])
def test_shipped_javascript_parses(script: str) -> None:
    path = STATIC / "ui" / script
    result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_slash_palette_is_fed_by_the_chat_surface_catalog() -> None:
    """Typing "/" must open a real command list, not a hard-coded copy."""
    assert 'id="command-palette"' in INDEX
    assert 'role="listbox"' in INDEX
    assert 'aria-controls="command-palette"' in INDEX
    palette = block(CHAT_JS, "/* ---- slash-command palette", "/* ---- model dialog")
    for needle in ("api('/commands')", "allowed_for_caller !== false",
                   "aria-activedescendant", "hideCommandPalette", "acceptCommandPalette",
                   "cmd.args_hint", "cmd.help"):
        assert needle in palette
    assert "$('message').addEventListener('input', () => { autosize(); renderCommandPalette(); });" in CHAT_JS
    keys = block(CHAT_JS, "$('message').addEventListener('keydown'", "$('attach-btn')")
    for needle in ("ArrowDown", "movePalette", "acceptCommandPalette()", "hideCommandPalette()"):
        assert needle in keys
    for needle in (".chat-palette {", ".chat-palette[hidden]", ".chat-palette__item.is-active"):
        assert needle in CHAT_CSS
