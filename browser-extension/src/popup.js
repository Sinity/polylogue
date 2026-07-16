const DEFAULT_RECEIVER = "http://127.0.0.1:8765";
const AUTO_REFRESH_MS = 8000;
const CAPTURE_MESSAGE_TIMEOUT_MS = 15000;
// `operator_status.js` is loaded immediately before this file as a classic
// script.  Its top-level function declarations therefore occupy the shared
// global lexical environment; binding the same names here makes the browser
// reject this entire script before the popup can start.  Keep the namespace as
// the sole cross-script binding and call through it instead.
const operatorStatusApi = globalThis.PolylogueOperatorStatus;

let currentCaptureQueue = { entries: [], dropped_count: 0 };
let currentBackfillJobs = [];
let currentLaunchJobs = [];
let currentLaunchContext = { launchEnabled: false, ownerInstanceId: null, launchSource: "live", cachedAt: null };
let currentReceiverOnline = true;

function hostMatches(hostname, domain) {
  return hostname === domain || hostname.endsWith(`.${domain}`);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function hostLabel(url) {
  try {
    const parsed = new URL(url);
    if (hostMatches(parsed.hostname, "chatgpt.com")) return "ChatGPT";
    if (hostMatches(parsed.hostname, "claude.ai")) return "Claude.ai";
    if (hostMatches(parsed.hostname, "grok.com")) return "Grok";
    if (hostMatches(parsed.hostname, "x.com") || hostMatches(parsed.hostname, "twitter.com")) return "Grok / X";
    return parsed.hostname;
  } catch {
    return "Unknown";
  }
}

function providerFromUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (hostMatches(parsed.hostname, "chatgpt.com")) return "chatgpt";
    if (hostMatches(parsed.hostname, "claude.ai")) return "claude-ai";
    if (hostMatches(parsed.hostname, "grok.com") || hostMatches(parsed.hostname, "x.com") || hostMatches(parsed.hostname, "twitter.com")) {
      return "grok";
    }
  } catch {
    return "unknown";
  }
  return "unknown";
}

function providerLogo(provider) {
  const labels = {
    chatgpt: "GPT",
    "claude-ai": "C",
    grok: "G",
    unknown: "?",
  };
  const safeProvider = provider || "unknown";
  return `<span class="provider-logo ${escapeHtml(safeProvider)}">${escapeHtml(labels[safeProvider] || "?")}</span>`;
}

function contentScriptFiles(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) {
      return [
        "src/common.js",
        "src/operator_status.js",
        "src/content/message_layer.js",
        "src/content/ambient_surface.js",
        "src/content/chatgpt.js",
      ];
    }
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) {
      return [
        "src/common.js",
        "src/operator_status.js",
        "src/content/message_layer.js",
        "src/content/ambient_surface.js",
        "src/content/claude.js",
      ];
    }
    if (
      parsed.hostname === "grok.com" ||
      parsed.hostname.endsWith(".grok.com") ||
      parsed.hostname === "x.com" ||
      parsed.hostname.endsWith(".x.com") ||
      parsed.hostname === "twitter.com" ||
      parsed.hostname.endsWith(".twitter.com")
    ) {
      return ["src/common.js", "src/content/grok.js"];
    }
  } catch {
    return [];
  }
  return [];
}

async function ensureCaptureScripts(tab) {
  const files = contentScriptFiles(tab?.url || "");
  if (!tab?.id || !files.length) return false;
  for (const file of files) {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: [file] });
  }
  return true;
}

function timeoutError(label, timeoutMs) {
  const error = new Error(`${label}_timeout_after_${timeoutMs}ms`);
  error.name = "PolylogueTimeoutError";
  return error;
}

function withTimeout(promise, timeoutMs, label) {
  let timer = 0;
  const timeout = new Promise((_resolve, reject) => {
    timer = window.setTimeout(() => reject(timeoutError(label, timeoutMs)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) window.clearTimeout(timer);
  });
}

async function capturePageFromTab(tab) {
  return withTimeout(
    chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage" }),
    CAPTURE_MESSAGE_TIMEOUT_MS,
    "capture_message",
  );
}

function setBadge(kind, text) {
  const badge = document.getElementById("badge");
  badge.className = `badge ${kind}`;
  badge.textContent = text;
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function relativeAge(iso) {
  const then = Date.parse(iso || "");
  if (!Number.isFinite(then)) return "--";
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.round(minutes / 60)}h`;
}

function stateExplanation(state) {
  return operatorStatusApi.operatorPresentationForState(state);
}

function conversationKey(provider, providerSessionId) {
  return provider && providerSessionId ? `${provider}:${providerSessionId}` : null;
}

function renderTimeline(items) {
  const node = document.getElementById("timeline");
  if (!node) return;
  const events = Array.isArray(items) ? items.slice(0, 8) : [];
  if (!events.length) {
    node.innerHTML = '<div class="log-meta">No decisions recorded for this conversation yet.</div>';
    return;
  }
  node.innerHTML = events.map((entry) => {
    const presentation = operatorStatusApi.eventPresentation(entry);
    return `<div class="log-item"><div class="log-time">${escapeHtml(relativeAge(entry.at))}</div><div><div class="log-title">${escapeHtml(presentation.label)}</div><div class="log-meta">${escapeHtml(presentation.detail || "")}</div></div></div>`;
  }).join("");
}

function tabState(tab, ledger) {
  const provider = providerFromUrl(tab?.url || tab?.pendingUrl || "");
  const sessionId = (() => {
    try {
      const url = new URL(tab?.url || tab?.pendingUrl || "");
      const parts = url.pathname.split("/").filter(Boolean);
      if (provider === "chatgpt") return parts[parts.indexOf("c") + 1] || null;
      if (provider === "claude-ai") return parts[0] === "chat" ? parts[1] || null : null;
      if (provider === "grok") {
        const pathId = parts.find((part, index) => parts[index - 1] === "chat" || parts[index - 1] === "grok");
        if (pathId) return pathId;
        const queryId = url.searchParams.get("conversation") || url.searchParams.get("conversationId");
        if (queryId) return queryId;
        if (!(parts[0] === "i" && parts[1] === "grok")) return null;
        let hash = 0x811c9dc5;
        for (const char of `${url.origin}${url.pathname}${url.search}`) {
          hash ^= char.charCodeAt(0);
          hash = Math.imul(hash, 0x01000193);
        }
        return `dom:${(hash >>> 0).toString(16).padStart(8, "0")}`;
      }
    } catch {
      return null;
    }
    return null;
  })();
  return { provider, sessionId, ledger: ledger[conversationKey(provider, sessionId)] || {} };
}

function activeConversationState(tab, globalState, ledger) {
  const context = tabState(tab, ledger);
  if (!globalState?.provider || !globalState?.provider_session_id) return globalState || {};
  if (!context.provider || !context.sessionId) return globalState || {};
  if (
    globalState?.provider === context.provider
    && globalState?.provider_session_id === context.sessionId
  ) return globalState;

  const item = context.ledger;
  return {
    online: globalState?.online ?? true,
    provider: context.provider,
    provider_session_id: context.sessionId,
    active_page_state: "conversation",
    archive_state: item.archive_state || null,
    capture_mode: item.capture_mode || null,
    asset_acquisition: item.asset_acquisition || null,
    turn_count: item.turn_count ?? null,
    attachment_count: item.attachment_count ?? null,
    last_receiver_request_id: item.receiver_request_id || null,
    updated_at: item.updated_at || null,
  };
}

function renderOpenTabs(tabs, ledger, activeTabId, receiverOnline = true) {
  const node = document.getElementById("open-tabs");
  if (!node) return;
  const supported = (Array.isArray(tabs) ? tabs : []).map((tab) => ({ tab, ...tabState(tab, ledger) }))
    .filter(({ provider, sessionId }) => provider !== "unknown" && Boolean(sessionId));
  document.getElementById("open-tab-count").textContent = String(supported.length);
  if (!supported.length) {
    node.innerHTML = '<div class="log-meta">No supported conversation tabs are open.</div>';
    return;
  }
  node.innerHTML = supported.map(({ tab, provider, sessionId, ledger: item }) => {
    const status = operatorStatusApi.operatorStatusForState({
      online: receiverOnline,
      captured: item.archive_state?.state === "archived" || Boolean(item.receiver_request_id),
      archive_state: item.archive_state,
      capture_mode: item.capture_mode,
    });
    const active = tab.id === activeTabId ? " active" : "";
    const title = tab.title || sessionId || "Untitled conversation";
    return `<div class="tab-item${active}"><div>${providerLogo(provider)} <strong>${escapeHtml(title)}</strong></div><span class="state-chip ${escapeHtml(status.tone)}">${escapeHtml(status.label)}</span></div>`;
  }).join("");
}

function renderLog(items) {
  const log = document.getElementById("log");
  const safeItems = Array.isArray(items) ? items.slice(0, 8) : [];
  document.getElementById("log-count").textContent = String(Array.isArray(items) ? items.length : 0);
  if (!safeItems.length) {
    log.innerHTML = '<div class="log-meta">No capture attempts recorded yet.</div>';
    return;
  }
  log.innerHTML = safeItems
    .map((entry) => {
      const provider = entry.provider || "unknown";
      const title = entry.ok
        ? `${provider} ${entry.provider_session_id || ""}`.trim()
        : entry.error || "capture failed";
      const meta = [entry.archive_state, entry.capture_mode, entry.receiver_request_id].filter(Boolean).join(" · ");
      return `<div class="log-item"><div class="log-time">${escapeHtml(relativeAge(entry.at))}</div><div><div class="log-title">${providerLogo(provider)} ${escapeHtml(title)}</div><div class="log-meta">${escapeHtml(meta || entry.reason || "")}</div></div></div>`;
    })
    .join("");
}

function fidelityLabel(captureMode) {
  if (captureMode === "native_full") return "Native";
  if (captureMode === "dom_degraded") return "DOM fallback";
  return captureMode || "--";
}

function costTokensLabel(state) {
  const totalTokens = state?.usage?.total_tokens ?? state?.total_tokens ?? null;
  const costUsd = state?.cost_usd ?? state?.cost?.usd ?? null;
  const parts = [];
  if (Number.isFinite(costUsd)) parts.push(`$${Number(costUsd).toFixed(3)}`);
  if (Number.isFinite(totalTokens)) parts.push(`${totalTokens} tokens`);
  return parts.length ? parts.join(" · ") : "Unavailable";
}

function truncateForDisplay(value, limit = 200) {
  const text = String(value ?? "");
  return text.length > limit ? `${text.slice(0, limit - 3)}...` : text;
}

function renderAssetAcquisition(assetAcquisition) {
  const summaryNode = document.getElementById("assets");
  const detailNode = document.getElementById("asset-failures");
  if (!assetAcquisition || typeof assetAcquisition !== "object") {
    summaryNode.textContent = "--";
    detailNode.textContent = "";
    return;
  }
  const attempted = Number(assetAcquisition.attempted) || 0;
  const acquired = Number(assetAcquisition.acquired) || 0;
  const failed = Array.isArray(assetAcquisition.failed) ? assetAcquisition.failed : [];
  const skipped = Number(assetAcquisition.skipped_over_budget) || 0;
  if (!attempted) {
    summaryNode.textContent = "none";
    detailNode.textContent = "";
    return;
  }
  const parts = [`${acquired} acquired`, `${failed.length} failed`];
  if (skipped) parts.push(`${skipped} skipped`);
  summaryNode.textContent = parts.join(" · ");
  const shown = failed.slice(0, 5);
  const overflow = failed.length - shown.length;
  detailNode.textContent = shown.length
    ? shown
        .map((item) => `${item.provider_attachment_id || "asset"}: ${truncateForDisplay(item.error || "unknown")}`)
        .join("; ") + (overflow > 0 ? `; +${overflow} more` : "")
    : "";
}

function workKindLabel(kind) {
  return {
    capture_retry: "Capture retry",
    backfill: "Backfill",
    sol_pro: "Sol Pro",
  }[kind] || "Work";
}

function renderWorkQueue() {
  const countNode = document.getElementById("work-count");
  const listNode = document.getElementById("work-queue");
  if (!countNode || !listNode) return;
  const items = operatorStatusApi.normalizeWorkItems({
    captureQueue: currentCaptureQueue,
    backfillJobs: currentBackfillJobs,
    launchJobs: currentLaunchJobs,
    ownerInstanceId: currentLaunchContext.ownerInstanceId,
    receiverOnline: currentReceiverOnline,
  });
  countNode.textContent = String(items.length);
  if (!items.length) {
    listNode.innerHTML = '<div class="empty">No queued, running, or recently completed external work.</div>';
    return;
  }
  listNode.innerHTML = items.slice(0, 12).map((item) => {
    const meta = [item.phase, item.cadence, `Owner: ${item.owner}`].filter(Boolean).join(" · ");
    const extra = [
      item.cooldown ? `Cooldown/backoff: ${item.cooldown}` : null,
      item.kind === "sol_pro" ? `Handoff: ${item.handoff}` : null,
      item.kind === "sol_pro" && currentLaunchContext.launchSource === "cached"
        ? `Last known ${relativeAge(currentLaunchContext.cachedAt)} ago`
        : null,
    ]
      .filter(Boolean)
      .join(" · ");
    return `<div class="work-item"><div class="work-head"><div class="work-title"><span class="work-kind">${escapeHtml(workKindLabel(item.kind))}</span><br>${escapeHtml(item.title)}</div><span class="work-pill ${escapeHtml(item.status.tone)}">${escapeHtml(item.status.label)}</span></div><div class="work-meta">${escapeHtml(meta)}</div>${extra ? `<div class="work-meta">${escapeHtml(extra)}</div>` : ""}</div>`;
  }).join("");
}

function renderQueue(queue) {
  currentCaptureQueue = queue && typeof queue === "object" ? queue : { entries: [], dropped_count: 0 };
  const countNode = document.getElementById("queue-count");
  const listNode = document.getElementById("queue-log");
  const entries = Array.isArray(currentCaptureQueue.entries) ? currentCaptureQueue.entries : [];
  const droppedCount = Number(currentCaptureQueue.dropped_count) || 0;
  countNode.textContent = droppedCount ? `${entries.length} (+${droppedCount} dropped)` : String(entries.length);
  if (!entries.length) {
    listNode.innerHTML = '<div class="log-meta">No captures queued for retry.</div>';
    renderWorkQueue();
    return;
  }
  listNode.innerHTML = entries
    .map((entry) => {
      const session = entry.envelope?.session || {};
      const provider = session.provider || "unknown";
      const providerSessionId = session.provider_session_id || "";
      const title = session.title || `${provider} ${providerSessionId}`.trim();
      const meta = [
        `attempt ${entry.attempts || 0}`,
        entry.next_attempt_at ? operatorStatusApi.normalizeWorkItems({ captureQueue: { entries: [entry] } })[0]?.cadence : null,
        entry.last_error,
      ]
        .filter(Boolean)
        .join(" · ");
      return `<div class="log-item"><div class="log-time">${escapeHtml(relativeAge(entry.enqueued_at))}</div><div><div class="log-title">${providerLogo(provider)} ${escapeHtml(title)}</div><div class="log-meta">${escapeHtml(meta)}</div></div></div>`;
    })
    .join("");
  renderWorkQueue();
}

function renderReceiverPairing(pairing, health = null, configuredUrl = "") {
  const statusNode = document.getElementById("receiver-pairing-status");
  const detailNode = document.getElementById("receiver-pairing-detail");
  if (!statusNode || !detailNode) return;
  const presentation = operatorStatusApi.receiverPairingPresentation({ pairing, health, configuredUrl });
  statusNode.textContent = presentation.headline;
  detailNode.textContent = presentation.detail;
  statusNode.title = pairing?.receiver_id || "";
}

function renderReceiverHealth(health, pairing = null, configuredUrl = "") {
  const node = document.getElementById("receiver-health");
  if (!node) return;
  if (!health) {
    node.textContent = "--";
    node.title = "";
    node.className = "state-chip neutral";
    renderReceiverPairing(pairing, health, configuredUrl);
    return;
  }
  const labels = {
    ok: "Online",
    recovered: "Recovered",
    unauthorized: "Token required",
    pairing_mismatch: "Identity mismatch",
    unreachable: "Receiver offline",
    error: "Receiver error",
  };
  const tone = ["ok", "recovered"].includes(health.status)
    ? "ok"
    : ["unauthorized", "pairing_mismatch", "error"].includes(health.status)
      ? "bad"
      : "warn";
  node.textContent = labels[health.status] || health.status || "--";
  node.title = health.detail || "";
  node.className = `state-chip ${tone}`;
  currentReceiverOnline = ["ok", "recovered"].includes(health.status);
  renderReceiverPairing(pairing || health.pairing || null, health, configuredUrl);
  renderWorkQueue();
}

let selectedBackfillJobId = null;

function renderBackfill(jobs) {
  const statusNode = document.getElementById("backfill-status");
  if (!statusNode) return;
  const list = Array.isArray(jobs) ? [...jobs].sort((left, right) => {
    const leftActive = ["running", "paused"].includes(left.status) ? 1 : 0;
    const rightActive = ["running", "paused"].includes(right.status) ? 1 : 0;
    return rightActive - leftActive || String(right.updated_at || right.created_at || "").localeCompare(String(left.updated_at || left.created_at || ""));
  }) : [];
  currentBackfillJobs = list;
  renderWorkQueue();
  const job = list.find((candidate) => candidate.id === selectedBackfillJobId) || list[0] || null;
  const selector = document.getElementById("backfill-job");
  if (selector) {
    selector.innerHTML = list.length
      ? list.map((candidate) => `<option value="${escapeHtml(candidate.id)}">${escapeHtml(`${candidate.job_title || `${candidate.provider} history backfill`} · ${candidate.status}`)}</option>`).join("")
      : '<option value="">No jobs yet</option>';
  }
  if (!job) {
    statusNode.textContent = "idle";
    return;
  }
  selectedBackfillJobId = job.id;
  if (selector) selector.value = job.id;
  const contractBlocked = job.cooldown_reason === "receiver_contract_incompatible";
  const recoveryBlocked = job.cooldown_reason === "browser_profile_recovery_required";
  const bridgeBlocked = job.cooldown_reason === "backfill_bridge_response_too_large";
  statusNode.textContent = contractBlocked
    ? `${job.provider} · receiver upgrade required`
    : recoveryBlocked
      ? `${job.provider} · profile recovery required`
      : bridgeBlocked
        ? `${job.provider} · conversation bridge limit reached`
      : `${job.provider} · ${job.status}`;
  document.getElementById("backfill-cursor").textContent = job.inventory_complete ? `${job.inventory_cursor} · complete` : job.inventory_cursor || "--";
  const progress = job.progress || {};
  document.getElementById("backfill-progress").textContent = `${progress.complete || 0}/${progress.total || 0} · ${progress.retry || 0} retry · ${progress.no_turns || 0} empty · ${(progress.error || 0) + (progress.operator_action || 0)} attention`;
  const cooldown = job.cooldown_until_ms ? new Date(job.cooldown_until_ms).toLocaleTimeString() : "none";
  document.getElementById("backfill-rate").textContent = `${Math.round((job.learned_cadence_ms || 0) / 1000)}s · ${cooldown}`;
  document.getElementById("backfill-last").textContent = contractBlocked
    ? "Receiver ACK contract is stale. Upgrade/restart receiver, then Resume."
    : recoveryBlocked
      ? "Browser profile was replaced. Inspect exported ledger before starting a new job."
      : bridgeBlocked
        ? "A native conversation exceeded the bounded bridge. It is held; Resume explicitly retries it with compact capture."
      : job.recovery_checkpoint_error
        ? `Profile recovery checkpoint failed: ${job.recovery_checkpoint_error}`
      : job.last_ack?.receiver_request_id || job.last_error || "--";
  const resumeButton = document.getElementById("backfill-resume");
  if (resumeButton) resumeButton.disabled = recoveryBlocked;
}

async function refreshBackfills() {
  const result = await chrome.runtime.sendMessage({ type: "polylogue.backfill.status" });
  renderBackfill(result?.jobs || []);
  return result;
}

let selectedLaunchJobId = null;
let selectedLaunchJob = null;

function launchEventLabel(job) {
  const events = Array.isArray(job?.events) ? job.events : [];
  const latest = events[events.length - 1];
  if (!latest) return job?.last_error || "--";
  return [latest.kind, latest.detail].filter(Boolean).join(" · ");
}

function renderLaunch(jobs, { launchEnabled = false, ownerInstanceId = null, launchSource = "live", cachedAt = null } = {}) {
  const statusNode = document.getElementById("launch-status");
  if (!statusNode) return;
  const list = Array.isArray(jobs) ? [...jobs].sort((left, right) => {
    const active = new Set(["leased", "uploading", "submitting", "submission_unknown", "submitted", "cooldown", "paused"]);
    return Number(active.has(right.status)) - Number(active.has(left.status))
      || String(right.updated_at || right.created_at || "").localeCompare(String(left.updated_at || left.created_at || ""));
  }) : [];
  currentLaunchJobs = list;
  currentLaunchContext = { launchEnabled: Boolean(launchEnabled), ownerInstanceId, launchSource, cachedAt };
  renderWorkQueue();
  const job = list.find((candidate) => candidate.job_id === selectedLaunchJobId) || list[0] || null;
  selectedLaunchJob = job;
  selectedLaunchJobId = job?.job_id || null;
  const enabled = document.getElementById("launch-enabled");
  if (enabled) enabled.checked = Boolean(launchEnabled);
  const selector = document.getElementById("launch-job");
  if (selector) {
    selector.innerHTML = list.length
      ? list.map((candidate) => `<option value="${escapeHtml(candidate.job_id)}">${escapeHtml(`#${candidate.queue_position || "?"} ${candidate.job_title || candidate.job_id} · ${candidate.status}/${candidate.phase} · ${candidate.cadence_minutes}m · ${candidate.attachments?.length || 0} files`)}</option>`).join("")
      : '<option value="">No jobs yet</option>';
    if (job) selector.value = job.job_id;
  }
  if (!job) {
    statusNode.textContent = launchEnabled ? "idle" : "disabled";
    for (const id of ["launch-title", "launch-phase", "launch-cadence", "launch-owner", "launch-handoff", "launch-last"]) {
      document.getElementById(id).textContent = "--";
    }
    for (const id of ["launch-poll", "launch-now", "launch-pause", "launch-resume", "launch-retry", "launch-confirm-existing", "launch-confirm-absent", "launch-cancel", "launch-inspect"]) {
      document.getElementById(id).disabled = true;
    }
    return;
  }
  statusNode.textContent = launchSource === "cached"
    ? `receiver offline · last known ${job.status}`
    : launchEnabled ? job.status : `${job.status} · disabled`;
  document.getElementById("launch-title").textContent = job.job_title || "GPT-5.6 Sol Pro work";
  document.getElementById("launch-phase").textContent = job.phase || job.status;
  const due = job.next_attempt_at ? new Date(job.next_attempt_at).toLocaleTimeString() : "now";
  const remainingMinutes = job.next_attempt_at
    ? Math.max(0, Math.ceil((Date.parse(job.next_attempt_at) - Date.now()) / 60_000))
    : 0;
  const cooldown = job.cooldown_reason
    ? `${job.cooldown_reason} until ${due}${remainingMinutes ? ` (~${remainingMinutes}m)` : ""}`
    : `due ${due}`;
  document.getElementById("launch-cadence").textContent = `${job.cadence_minutes}m cadence · ${cooldown}`;
  document.getElementById("launch-owner").textContent = job.lease_owner
    ? (job.lease_owner === ownerInstanceId ? "this extension" : "another extension")
    : "unclaimed";
  const normalizedJob = operatorStatusApi.normalizeWorkItems({
    launchJobs: [job],
    ownerInstanceId,
    receiverOnline: currentReceiverOnline,
  })[0];
  document.getElementById("launch-handoff").textContent = normalizedJob?.handoff || "Awaiting completion handoff";
  const lastEvent = launchEventLabel(job);
  document.getElementById("launch-last").textContent = launchSource === "cached"
    ? `Last known ${relativeAge(cachedAt)} ago · ${lastEvent}`
    : lastEvent;
  const active = ["leased", "uploading", "submitting", "submitted"].includes(job.status);
  const liveControls = launchSource === "live" && currentReceiverOnline;
  const protectedCircuit = ["rate_limited", "safety_locked"].includes(job.cooldown_reason)
    && (!job.next_attempt_at || Date.parse(job.next_attempt_at) > Date.now());
  document.getElementById("launch-pause").disabled = !liveControls || active || ["completed", "cancelled"].includes(job.status);
  document.getElementById("launch-now").disabled = !liveControls || active
    || ["completed", "cancelled"].includes(job.status)
    || protectedCircuit;
  document.getElementById("launch-resume").disabled = !liveControls || protectedCircuit
    || !["paused", "cooldown", "failed"].includes(job.status);
  document.getElementById("launch-retry").disabled = !liveControls || protectedCircuit
    || !["paused", "cooldown", "failed"].includes(job.status);
  document.getElementById("launch-confirm-existing").disabled = !liveControls
    || job.status !== "submission_unknown"
    || (!job.tab_id && !job.conversation_url);
  document.getElementById("launch-confirm-absent").disabled = !liveControls || job.status !== "submission_unknown";
  document.getElementById("launch-cancel").disabled = !liveControls || ["completed", "cancelled"].includes(job.status);
  document.getElementById("launch-inspect").disabled = !job.tab_id && !job.conversation_url;
}

async function refreshLaunches() {
  try {
    const result = await chrome.runtime.sendMessage({ type: "polylogue.launch.status" });
    if (!result?.ok) throw new Error(result?.error || "launch_status_unavailable");
    renderLaunch(result.jobs || [], result);
    return result;
  } catch (error) {
    renderLaunch([], { launchEnabled: false });
    document.getElementById("launch-status").textContent = "unavailable";
    throw error;
  }
}

function ambientFallback(raw, tabUrl) {
  const settings = raw && typeof raw === "object" ? raw : {};
  const disabledSites = settings.disabled_sites && typeof settings.disabled_sites === "object"
    ? settings.disabled_sites
    : {};
  let hostname = "";
  try { hostname = new URL(tabUrl || "").hostname; } catch { hostname = ""; }
  return {
    enabled: settings.enabled !== false,
    disabled_sites: disabledSites,
    site: hostname || null,
    site_enabled: hostname ? disabledSites[hostname] !== true : true,
  };
}

function renderAmbient(ambient, tabUrl = "", assertions = null) {
  const enabledNode = document.getElementById("ambient-enabled");
  const siteNode = document.getElementById("ambient-site-enabled");
  const siteLabel = document.getElementById("ambient-site");
  const assertionNode = document.getElementById("assertion-status");
  if (enabledNode) enabledNode.checked = ambient?.enabled !== false;
  if (siteNode) {
    siteNode.checked = ambient?.site_enabled !== false;
    siteNode.disabled = !providerFromUrl(tabUrl) || providerFromUrl(tabUrl) === "unknown";
  }
  if (siteLabel) siteLabel.textContent = ambient?.site || hostLabel(tabUrl || "") || "Current site";
  if (assertionNode) {
    assertionNode.textContent = assertions?.persistence_supported
      ? "The receiver advertises assertion persistence, but this extension build has no authenticated write handler. Save remains disabled."
      : "Text selections form a local assertion candidate only. Save stays disabled because the authenticated receiver advertises no assertion route.";
  }
}

function renderReverse(reverse) {
  const statusNode = document.getElementById("reverse-status");
  const detailNode = document.getElementById("reverse-detail");
  if (!statusNode || !detailNode) return;
  const enabled = Boolean(reverse?.enabled);
  statusNode.textContent = enabled ? "Configured on" : "Off — safe default";
  statusNode.className = `state-chip ${enabled ? "warn" : "neutral"}`;
  detailNode.textContent = enabled
    ? "Posting is configured elsewhere, but this ambient surface never posts or submits. The receiver's independent posting gate is still required."
    : "No ambient control can post or submit. Extension and receiver posting gates remain independently required.";
}

async function loadMissionSnapshot() {
  try {
    const result = await chrome.runtime.sendMessage({ type: "polylogue.missionControl.status", refresh: false });
    if (!result?.ok || (!result.state && !result.work && !result.receiver)) return null;
    return result;
  } catch {
    return null;
  }
}

function renderDebugLog(items) {
  const debug = document.getElementById("debug-log");
  const safeItems = Array.isArray(items) ? items.slice(0, 24) : [];
  document.getElementById("debug-count").textContent = String(Array.isArray(items) ? items.length : 0);
  if (!safeItems.length) {
    debug.innerHTML = '<div class="log-meta">No debug events yet.</div>';
    return;
  }
  debug.innerHTML = safeItems
    .map((entry) => {
      const title = [entry.stage, entry.method, entry.path].filter(Boolean).join(" ");
      const meta = [
        entry.ok === false ? "error" : entry.ok === true ? "ok" : null,
        entry.status,
        entry.archive_state,
        entry.capture_mode,
        entry.provider,
        entry.provider_session_id,
        entry.receiver_request_id || entry.request_id,
      ]
        .filter(Boolean)
        .join(" · ");
      return `<div class="log-item"><div class="log-time">${escapeHtml(relativeAge(entry.at))}</div><div><div class="log-title">${escapeHtml(title || "event")}</div><div class="log-meta">${escapeHtml(meta || entry.error || "")}</div></div></div>`;
    })
    .join("");
}

function setActionState(button, state, text = "") {
  if (!button) return;
  button.dataset.state = state;
  button.disabled = state === "busy";
  const status = button.querySelector(".button-status");
  if (status) status.textContent = text;
}

async function withAction(buttonId, fn, labels = {}) {
  const button = document.getElementById(buttonId);
  const busyText = labels.busy || "Working";
  const okText = labels.ok || "Done";
  const badText = labels.bad || "Failed";
  setActionState(button, "busy", busyText);
  try {
    const result = await fn();
    setActionState(button, "ok", okText);
    window.setTimeout(() => setActionState(button, "idle", ""), 1600);
    return result;
  } catch (error) {
    setActionState(button, "bad", badText);
    window.setTimeout(() => setActionState(button, "idle", ""), 3000);
    throw error;
  }
}

async function render() {
  const stored = await chrome.storage.local.get({
    polylogueCaptureLog: [],
    polylogueDebugLog: [],
    polylogueCaptureQueue: { entries: [], dropped_count: 0 },
    polylogueState: null,
    polylogueSessionLedger: {},
    polylogueConversationTimeline: {},
    polylogueReceiverPairing: null,
    polylogueAmbientSettings: { enabled: true, disabled_sites: {} },
    postingEnabled: false,
    receiverAuthToken: "",
    receiverBaseUrl: DEFAULT_RECEIVER,
  });
  renderLog(stored.polylogueCaptureLog);
  renderDebugLog(stored.polylogueDebugLog);
  document.getElementById("receiver-url").value = stored.receiverBaseUrl;
  document.getElementById("receiver-token").value = stored.receiverAuthToken || "";
  document.getElementById("receiver").textContent = stored.receiverBaseUrl;

  const [tab, openTabs, mission] = await Promise.all([
    activeTab(),
    chrome.tabs.query({}),
    loadMissionSnapshot(),
  ]);
  const extensionBuild = document.getElementById("extension-build");
  if (extensionBuild) {
    const runtime = mission?.extension;
    extensionBuild.textContent = runtime?.contract_epoch
      ? `${runtime.contract_epoch} · ${runtime.manifest_version || "unversioned"}`
      : "Legacy / stale runtime";
    extensionBuild.title = [runtime?.extension_id, runtime?.instance_id].filter(Boolean).join(" · ");
  }
  const currentProvider = providerFromUrl(tab?.url || "");
  document.getElementById("page").innerHTML = `${providerLogo(currentProvider)} <span>${escapeHtml(hostLabel(tab?.url || ""))}</span>`;

  const state = mission?.state || activeConversationState(tab, stored.polylogueState, stored.polylogueSessionLedger || {});
  const pairing = mission?.receiver?.pairing || state?.receiver_pairing || stored.polylogueReceiverPairing || null;
  const health = mission?.receiver?.health || state?.receiver_health || (
    state?.error === "unauthorized"
      ? { status: "unauthorized", detail: "unauthorized", pairing }
      : state?.error === "receiver_pairing_mismatch"
        ? { status: "pairing_mismatch", detail: "receiver_pairing_mismatch", pairing }
        : state?.online === false
          ? { status: "unreachable", detail: state?.error || "receiver_unavailable", pairing }
          : null
  );
  currentReceiverOnline = health
    ? ["ok", "recovered"].includes(health.status)
    : state?.online !== false;
  if (health) renderReceiverHealth(health, pairing, mission?.receiver?.configured_url || stored.receiverBaseUrl);
  else renderReceiverPairing(pairing, null, stored.receiverBaseUrl);

  const status = operatorStatusApi.operatorStatusForState(state || {});
  const requestNode = document.getElementById("receiver-request");
  const modeNode = document.getElementById("mode");
  const turnsNode = document.getElementById("turns");
  const updatedNode = document.getElementById("updated");
  requestNode.textContent = state?.last_receiver_request_id || "--";
  modeNode.textContent = state?.capture_mode || state?.archive_state?.state || "--";
  document.getElementById("fidelity").textContent = fidelityLabel(state?.capture_mode);
  document.getElementById("cost-tokens").textContent = costTokensLabel(state);
  renderAssetAcquisition(state?.asset_acquisition);
  const lastSession = state?.last_capture || {};
  const capturedCount = state?.turn_count ?? lastSession.turn_count ?? null;
  const visibleCount = state?.archive_state?.indexed_message_count ?? null;
  turnsNode.textContent = capturedCount === null && visibleCount === null
    ? "--"
    : `${capturedCount ?? "--"} captured / ${visibleCount ?? "--"} visible`;
  updatedNode.textContent = state?.updated_at ? `${relativeAge(state.updated_at)} ago` : "--";

  const explanation = stateExplanation(state);
  const [badgeKind, badgeText] = explanation.badge;
  document.getElementById("archive").textContent = explanation.archive;
  document.getElementById("state").textContent = explanation.headline;
  document.getElementById("state-detail").textContent = explanation.detail;
  setBadge(badgeKind, badgeText);
  const operatorState = document.getElementById("operator-state");
  if (operatorState) {
    operatorState.textContent = status.label;
    operatorState.className = `state-chip ${status.tone}`;
  }
  const fidelityFlag = document.getElementById("fidelity-flag");
  if (fidelityFlag) fidelityFlag.hidden = !status.partialFidelity;

  const activeProvider = state?.provider || providerFromUrl(tab?.url || "");
  const activeSessionId = state?.provider_session_id || tabState(tab, stored.polylogueSessionLedger || {}).sessionId;
  renderTimeline(mission?.timeline || stored.polylogueConversationTimeline?.[conversationKey(activeProvider, activeSessionId)] || []);
  renderOpenTabs(openTabs, stored.polylogueSessionLedger || {}, tab?.id, currentReceiverOnline);

  if (mission?.work) {
    renderQueue(mission.work.capture_queue || stored.polylogueCaptureQueue);
    renderBackfill(mission.work.backfill_jobs || []);
    renderLaunch(mission.work.launch_jobs || [], {
      launchEnabled: mission.work.launch_enabled,
      ownerInstanceId: mission.work.launch_owner_instance_id,
      launchSource: mission.work.launch_source || "live",
      cachedAt: mission.work.launch_cached_at || null,
    });
  } else {
    renderQueue(stored.polylogueCaptureQueue);
    await refreshBackfills().catch(() => renderBackfill([]));
    await refreshLaunches().catch(() => undefined);
  }

  const ambient = mission?.ambient || ambientFallback(stored.polylogueAmbientSettings, tab?.url || "");
  renderAmbient(ambient, tab?.url || "", mission?.assertions || null);
  renderReverse(mission?.reverse || { enabled: Boolean(stored.postingEnabled) });
}

async function refreshStatus(reason = "popup_manual") {
  await chrome.runtime.sendMessage({ type: "polylogue.status", reason });
  await render();
}

async function currentActiveConversationState() {
  const [stored, tab] = await Promise.all([
    chrome.storage.local.get({ polylogueState: null, polylogueSessionLedger: {} }),
    activeTab(),
  ]);
  return activeConversationState(tab, stored.polylogueState, stored.polylogueSessionLedger || {});
}

document.getElementById("check").addEventListener("click", async () => {
  await withAction("check", () => refreshStatus("popup_manual"), { busy: "Checking" });
});

document.getElementById("sync-open-tabs").addEventListener("click", async () => {
  await withAction("sync-open-tabs", async () => {
    await chrome.runtime.sendMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });
    await render();
  }, { busy: "Syncing" });
});

document.getElementById("copy-ref").addEventListener("click", async () => {
  await withAction("copy-ref", async () => {
    const state = await currentActiveConversationState();
    const provider = state.provider || state.last_capture?.provider;
    const providerSessionId = state.provider_session_id || state.last_capture?.provider_session_id;
    const ref = provider && providerSessionId ? `${provider}:${providerSessionId}` : "";
    if (ref && window.navigator.clipboard?.writeText) await window.navigator.clipboard.writeText(ref);
  }, { busy: "Copying", ok: "Copied" });
});

document.getElementById("open-polylogue").addEventListener("click", async () => {
  await withAction("open-polylogue", async () => {
    const [stored, state] = await Promise.all([
      chrome.storage.local.get({ receiverBaseUrl: DEFAULT_RECEIVER }),
      currentActiveConversationState(),
    ]);
    const providerSessionId = state.provider_session_id || state.last_capture?.provider_session_id;
    const url = `${String(stored.receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, "")}/?q=${encodeURIComponent(providerSessionId || "")}`;
    await chrome.tabs.create({ url });
  }, { busy: "Opening" });
});

document.getElementById("check-receiver").addEventListener("click", async () => {
  try {
    await withAction(
      "check-receiver",
      async () => {
        const health = await chrome.runtime.sendMessage({ type: "polylogue.checkReceiverHealth" });
        renderReceiverHealth(health, health?.pairing || null, health?.endpoint || "");
        if (health?.status === "unreachable") throw new Error(health?.detail || "unreachable");
      },
      { busy: "Checking", ok: "Checked", bad: "Unreachable" },
    );
  } catch {
    // withAction already reflects the failure via button state + renderReceiverHealth;
    // swallow here so a genuinely unreachable receiver doesn't surface as an
    // unhandled rejection from this click listener.
  }
});

document.getElementById("reset-pairing")?.addEventListener("click", async () => {
  const confirmed = typeof globalThis.confirm !== "function"
    || globalThis.confirm("Reset the trusted receiver identity? Capture, backfill, and Sol Pro queues will be preserved.");
  if (!confirmed) return;
  await withAction("reset-pairing", async () => {
    const result = await chrome.runtime.sendMessage({ type: "polylogue.receiverPairing.reset" });
    renderReceiverHealth(result?.health, result?.pairing, result?.health?.endpoint || "");
    await render();
  }, { busy: "Resetting", ok: "Pairing reset" });
});

document.getElementById("retry-captures")?.addEventListener("click", async () => {
  await withAction("retry-captures", async () => {
    await chrome.runtime.sendMessage({ type: "polylogue.retryCaptureQueue" });
    await render();
  }, { busy: "Retrying", ok: "Queue checked" });
});

document.getElementById("ambient-enabled")?.addEventListener("change", async (event) => {
  await chrome.runtime.sendMessage({
    type: "polylogue.ambient.configure",
    enabled: Boolean(event.target.checked),
  });
  await render();
});

document.getElementById("ambient-site-enabled")?.addEventListener("change", async (event) => {
  const tab = await activeTab();
  await chrome.runtime.sendMessage({
    type: "polylogue.ambient.configure",
    url: tab?.url || "",
    site_enabled: Boolean(event.target.checked),
  });
  await render();
});

document.getElementById("save").addEventListener("click", async () => {
  await withAction("save", async () => {
    const receiverBaseUrl = document.getElementById("receiver-url").value;
    const receiverAuthToken = document.getElementById("receiver-token").value;
    await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver", receiverBaseUrl, receiverAuthToken });
    await refreshStatus("popup_configure_receiver");
  }, { busy: "Saving", ok: "Saved" });
});

document.getElementById("capture").addEventListener("click", async () => {
  await withAction("capture", async () => {
    const tab = await activeTab();
    if (!tab?.id) return;
    let result = await capturePageFromTab(tab).catch((error) => ({
      ok: false,
      error: String(error.message || error)
    }));
    if (!result?.ok && (await ensureCaptureScripts(tab))) {
      result = await capturePageFromTab(tab).catch((error) => ({
        ok: false,
        error: String(error.message || error)
      }));
    }
    if (!result?.ok) {
      await chrome.runtime.sendMessage({
        type: "polylogue.capturePageFailed",
        error: result?.error || "capture_page_failed",
        tab_id: tab.id,
        tab_url: tab.url || tab.pendingUrl || null,
      });
    }
    await render();
  }, { busy: "Capturing" });
});

document.getElementById("debug-toggle").addEventListener("click", () => {
  document.getElementById("debug-panel").toggleAttribute("hidden");
});

document.getElementById("debug-export").addEventListener("click", async () => {
  await withAction("debug-export", async () => {
    const stored = await chrome.storage.local.get({ polylogueDebugLog: [], polylogueCaptureLog: [], polylogueState: null });
    const payload = {
      exported_at: new Date().toISOString(),
      state: stored.polylogueState || null,
      debug_log: Array.isArray(stored.polylogueDebugLog) ? stored.polylogueDebugLog : [],
      capture_log: Array.isArray(stored.polylogueCaptureLog) ? stored.polylogueCaptureLog : [],
    };
    const blob = new globalThis.Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
    const url = globalThis.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `polylogue-browser-capture-debug-${Date.now()}.json`;
    link.click();
    globalThis.URL.revokeObjectURL(url);
  }, { busy: "Exporting", ok: "Exported" });
});

document.getElementById("backfill-start")?.addEventListener("click", async () => {
  await withAction("backfill-start", async () => {
    const provider = document.getElementById("backfill-provider").value;
    const cutoffValue = document.getElementById("backfill-cutoff").value;
    if (!cutoffValue) throw new Error("backfill_cutoff_required");
    const response = await chrome.runtime.sendMessage({
      type: "polylogue.backfill.start",
      provider,
      cutoff: new Date(`${cutoffValue}T00:00:00Z`).toISOString(),
    });
    selectedBackfillJobId = response.job.id;
    await refreshBackfills();
  }, { busy: "Starting", ok: "Started" });
});

document.getElementById("backfill-job")?.addEventListener("change", async (event) => {
  selectedBackfillJobId = event.target.value || null;
  await refreshBackfills();
});

for (const action of ["pause", "resume", "cancel"]) {
  document.getElementById(`backfill-${action}`)?.addEventListener("click", async () => {
    if (!selectedBackfillJobId) return;
    await withAction(`backfill-${action}`, async () => {
      await chrome.runtime.sendMessage({ type: "polylogue.backfill.control", job_id: selectedBackfillJobId, action });
      await refreshBackfills();
    }, { busy: `${action}…`, ok: action });
  });
}

document.getElementById("backfill-export")?.addEventListener("click", async () => {
  if (!selectedBackfillJobId) return;
  await withAction("backfill-export", async () => {
    const response = await chrome.runtime.sendMessage({ type: "polylogue.backfill.export", job_id: selectedBackfillJobId });
    const blob = new globalThis.Blob([`${JSON.stringify(response.ledger, null, 2)}\n`], { type: "application/json" });
    const url = globalThis.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `polylogue-backfill-${selectedBackfillJobId}.json`;
    link.click();
    globalThis.URL.revokeObjectURL(url);
  }, { busy: "Exporting", ok: "Exported" });
});

document.getElementById("launch-enabled")?.addEventListener("change", async (event) => {
  await chrome.runtime.sendMessage({
    type: "polylogue.launch.configure",
    launchEnabled: Boolean(event.target.checked),
  });
  await refreshLaunches();
});

document.getElementById("launch-job")?.addEventListener("change", async (event) => {
  selectedLaunchJobId = event.target.value || null;
  await refreshLaunches();
});

document.getElementById("launch-poll")?.addEventListener("click", async () => {
  await withAction("launch-poll", async () => {
    await chrome.runtime.sendMessage({ type: "polylogue.launch.poll" });
    await refreshLaunches();
  }, { busy: "Checking", ok: "Checked" });
});

for (const action of ["pause", "resume", "retry", "cancel"]) {
  document.getElementById(`launch-${action}`)?.addEventListener("click", async () => {
    if (!selectedLaunchJobId) return;
    await withAction(`launch-${action}`, async () => {
      await chrome.runtime.sendMessage({
        type: "polylogue.launch.control",
        job_id: selectedLaunchJobId,
        action,
      });
      await refreshLaunches();
    }, { busy: `${action}…`, ok: action });
  });
}

document.getElementById("launch-now")?.addEventListener("click", async () => {
  if (!selectedLaunchJobId) return;
  await withAction("launch-now", async () => {
    await chrome.runtime.sendMessage({
      type: "polylogue.launch.control",
      job_id: selectedLaunchJobId,
      action: "launch_now",
    });
    await refreshLaunches();
  }, { busy: "Prioritizing", ok: "Queued now" });
});

document.getElementById("launch-inspect")?.addEventListener("click", async () => {
  if (selectedLaunchJob?.tab_id) {
    await chrome.tabs.update(selectedLaunchJob.tab_id, { active: true });
  } else if (selectedLaunchJob?.conversation_url) {
    await chrome.tabs.create({ url: selectedLaunchJob.conversation_url, active: true });
  }
});

function exactChatGptConversation(url) {
  try {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (parsed.protocol !== "https:" || parsed.hostname !== "chatgpt.com" || parts.length !== 2 || parts[0] !== "c") {
      return null;
    }
    return { conversation_id: parts[1], conversation_url: parsed.href };
  } catch {
    return null;
  }
}

document.getElementById("launch-confirm-existing")?.addEventListener("click", async () => {
  if (!selectedLaunchJobId || selectedLaunchJob?.status !== "submission_unknown") return;
  const tab = selectedLaunchJob.tab_id ? await chrome.tabs.get(selectedLaunchJob.tab_id) : null;
  const conversation = exactChatGptConversation(tab?.url || selectedLaunchJob.conversation_url || "");
  if (!conversation) {
    await withAction("launch-confirm-existing", async () => {
      throw new Error("The inspected tab is not an exact ChatGPT conversation URL");
    });
    return;
  }
  if (!globalThis.confirm(`Confirm that this job submitted conversation ${conversation.conversation_id}.`)) return;
  await withAction("launch-confirm-existing", async () => {
    await chrome.runtime.sendMessage({
      type: "polylogue.launch.control",
      job_id: selectedLaunchJobId,
      action: "confirm_existing_conversation",
      inspection_receipt: "operator inspected the retained launch tab and confirmed the matching ChatGPT conversation exists",
      ...conversation,
    });
    await refreshLaunches();
  }, { busy: "Recording…", ok: "Submitted" });
});

document.getElementById("launch-confirm-absent")?.addEventListener("click", async () => {
  if (!selectedLaunchJobId || selectedLaunchJob?.status !== "submission_unknown") return;
  if (!globalThis.confirm("Confirm that you inspected ChatGPT and no matching conversation exists. This permits one new submission.")) return;
  await withAction("launch-confirm-absent", async () => {
    await chrome.runtime.sendMessage({
      type: "polylogue.launch.control",
      job_id: selectedLaunchJobId,
      action: "confirm_no_conversation",
      inspection_receipt: "operator inspected ChatGPT and confirmed that no matching conversation exists",
    });
    await refreshLaunches();
  }, { busy: "Recording…", ok: "Requeued" });
});

void (async () => {
  await render();
  void refreshStatus("popup_open").catch(() => render());
  const refreshTimer = window.setInterval(() => {
    void refreshStatus("popup_auto").catch(() => render());
  }, AUTO_REFRESH_MS);
  refreshTimer.unref?.();
})();
