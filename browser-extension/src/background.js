const DEFAULT_RECEIVER = "http://127.0.0.1:8765";
const BACKGROUND_CAPTURE_MIN_INTERVAL_MS = 30000;
const ACTIVE_TAB_STATE_MIN_INTERVAL_MS = 4000;
const CAPTURE_LOG_LIMIT = 80;
const DEBUG_LOG_LIMIT = 160;
const POST_POLL_INTERVAL_MS = 5000;
const recentBackgroundCaptures = new Map();
const recentActiveTabStateChecks = new Map();
// command_id -> true once dispatched to a content script this SW lifetime, so a
// fast poll cannot deliver the same command twice before its ack lands.
const inFlightPostCommands = new Set();
const pendingPostCommandAcks = new Map();
let postPollTimer = 0;

function injectionPlanForUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) {
      return [
        { files: ["src/content/chatgpt_bridge.js"], world: "MAIN" },
        { files: ["src/common.js", "src/content/chatgpt.js"] },
      ];
    }
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) {
      return [
        { files: ["src/content/claude_bridge.js"], world: "MAIN" },
        { files: ["src/common.js", "src/content/claude.js"] },
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
      return [{ files: ["src/common.js", "src/content/grok.js"] }];
    }
  } catch {
    return [];
  }
  return [];
}

async function receiverSettings() {
  const stored = await chrome.storage.local.get({
    receiverAuthToken: "",
    receiverBaseUrl: DEFAULT_RECEIVER,
  });
  return {
    authToken: String(stored.receiverAuthToken || ""),
    baseUrl: String(stored.receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, ""),
  };
}

async function saveReceiverSettings(receiverBaseUrl, receiverAuthToken = "") {
  await chrome.storage.local.set({
    receiverAuthToken: String(receiverAuthToken || ""),
    receiverBaseUrl: String(receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, "") || DEFAULT_RECEIVER,
  });
  return receiverSettings();
}

function sessionKey(provider, providerSessionId) {
  return `${provider || "unknown"}:${providerSessionId || "unknown"}`;
}

async function appendCaptureLog(entry) {
  const stored = await chrome.storage.local.get({ polylogueCaptureLog: [] });
  const prior = Array.isArray(stored.polylogueCaptureLog) ? stored.polylogueCaptureLog : [];
  const next = [
    {
      at: new Date().toISOString(),
      ...entry,
    },
    ...prior,
  ].slice(0, CAPTURE_LOG_LIMIT);
  await chrome.storage.local.set({ polylogueCaptureLog: next });
  return next;
}

function sanitizeDebugDetails(value, depth = 0) {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") return value.length > 160 ? `${value.slice(0, 157)}...` : value;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) return { count: value.length };
  if (typeof value !== "object" || depth > 2) return String(value);
  const redactedKeys = new Set(["body", "envelope", "raw_provider_payload", "text", "turns", "messages", "content"]);
  const out = {};
  for (const [key, item] of Object.entries(value)) {
    if (redactedKeys.has(key)) {
      out[key] = "[redacted]";
      continue;
    }
    out[key] = sanitizeDebugDetails(item, depth + 1);
  }
  return out;
}

async function appendDebugLog(entry) {
  const stored = await chrome.storage.local.get({ polylogueDebugLog: [] });
  const prior = Array.isArray(stored.polylogueDebugLog) ? stored.polylogueDebugLog : [];
  const next = [
    {
      at: new Date().toISOString(),
      ...sanitizeDebugDetails(entry),
    },
    ...prior,
  ].slice(0, DEBUG_LOG_LIMIT);
  await chrome.storage.local.set({ polylogueDebugLog: next });
  return next;
}

async function updateSessionLedger({ provider, providerSessionId, patch }) {
  if (!provider || !providerSessionId) return null;
  const stored = await chrome.storage.local.get({ polylogueSessionLedger: {} });
  const ledger =
    stored.polylogueSessionLedger && typeof stored.polylogueSessionLedger === "object"
      ? stored.polylogueSessionLedger
      : {};
  const key = sessionKey(provider, providerSessionId);
  const next = {
    ...(ledger[key] || {}),
    provider,
    provider_session_id: providerSessionId,
    updated_at: new Date().toISOString(),
    ...patch,
  };
  await chrome.storage.local.set({ polylogueSessionLedger: { ...ledger, [key]: next } });
  return next;
}

function badgeForState(state) {
  if (!state.online) return { text: "off", color: "#9b2c2c" };
  const archiveState = state.archive_state?.state;
  if (archiveState === "archived" || state.captured) return { text: "ok", color: "#14764e" };
  if (archiveState === "failed" || state.error) return { text: "err", color: "#ad2f2f" };
  if (archiveState === "spooled_only" || archiveState === "ingest_pending") return { text: "…", color: "#9a5b00" };
  if (state.capture_mode === "dom_degraded") return { text: "dom", color: "#8a5a00" };
  return { text: "on", color: "#325d8f" };
}

async function setState(state) {
  const nextState = { ...state, updated_at: new Date().toISOString() };
  await chrome.storage.local.set({ polylogueState: nextState });
  const badge = badgeForState(nextState);
  await chrome.action.setBadgeText({ text: badge.text });
  await chrome.action.setBadgeBackgroundColor({ color: badge.color });
}

function buildReceiverRequestId() {
  const random = Math.random().toString(36).slice(2, 10);
  return `polylogue-ext-${Date.now().toString(36)}-${random}`;
}

async function requestHeaders({ hasBody = false, requestId = "" } = {}) {
  const settings = await receiverSettings();
  const headers = {};
  if (hasBody) headers["Content-Type"] = "application/json";
  if (settings.authToken) headers.Authorization = `Bearer ${settings.authToken}`;
  if (requestId) headers["X-Request-ID"] = requestId;
  return headers;
}

async function postJson(path, payload) {
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  await appendDebugLog({ stage: "receiver_request", method: "POST", path, request_id: requestId, has_body: true });
  try {
    const response = await fetch(`${settings.baseUrl}${path}`, {
      method: "POST",
      headers: await requestHeaders({ hasBody: true, requestId }),
      body: JSON.stringify(payload)
    });
    const receiverRequestId = response.headers.get("X-Request-ID") || requestId;
    const body = await response.json().catch(() => ({}));
    await appendDebugLog({
      stage: "receiver_response",
      method: "POST",
      path,
      request_id: requestId,
      receiver_request_id: receiverRequestId,
      ok: response.ok,
      status: response.status,
      provider: body.provider || payload?.session?.provider || null,
      provider_session_id: body.provider_session_id || payload?.session?.provider_session_id || null,
      archive_state: body.state || null,
      artifact_ref: body.artifact_ref || null,
    });
    if (!response.ok) {
      const error = new Error(body.error || `HTTP ${response.status}`);
      error.receiverRequestId = receiverRequestId;
      throw error;
    }
    return { ...body, receiver_request_id: receiverRequestId };
  } catch (error) {
    await appendDebugLog({
      stage: "receiver_error",
      method: "POST",
      path,
      request_id: requestId,
      receiver_request_id: error.receiverRequestId || null,
      error: String(error.message || error),
    });
    throw error;
  }
}

async function getJson(path) {
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  await appendDebugLog({ stage: "receiver_request", method: "GET", path, request_id: requestId });
  try {
    const response = await fetch(`${settings.baseUrl}${path}`, {
      headers: await requestHeaders({ requestId }),
    });
    const receiverRequestId = response.headers.get("X-Request-ID") || requestId;
    const body = await response.json().catch(() => ({}));
    await appendDebugLog({
      stage: "receiver_response",
      method: "GET",
      path,
      request_id: requestId,
      receiver_request_id: receiverRequestId,
      ok: response.ok,
      status: response.status,
      provider: body.provider || null,
      provider_session_id: body.provider_session_id || null,
      archive_state: body.state || null,
    });
    if (!response.ok) {
      const error = new Error(body.error || `HTTP ${response.status}`);
      error.receiverRequestId = receiverRequestId;
      throw error;
    }
    return { ...body, receiver_request_id: receiverRequestId };
  } catch (error) {
    await appendDebugLog({
      stage: "receiver_error",
      method: "GET",
      path,
      request_id: requestId,
      receiver_request_id: error.receiverRequestId || null,
      error: String(error.message || error),
    });
    throw error;
  }
}

async function refreshReceiverState() {
  try {
    const status = await getJson("/v1/status");
    await setState({
      online: true,
      captured: false,
      status,
      last_receiver_request_id: status.receiver_request_id || null,
    });
  } catch (error) {
    await setState({
      online: false,
      captured: false,
      error: String(error.message || error),
    });
  }
}

async function ensureCaptureScripts(tab) {
  const plan = injectionPlanForUrl(tab?.url || tab?.pendingUrl || "");
  if (!tab?.id || !plan.length || !chrome.scripting?.executeScript) return false;
  for (const step of plan) {
    const details = { target: { tabId: tab.id }, files: step.files };
    if (step.world) details.world = step.world;
    await chrome.scripting.executeScript(details);
  }
  return true;
}

async function captureTab(tab, reason = "background") {
  if (!tab?.id || !injectionPlanForUrl(tab.url || tab.pendingUrl || "").length) return null;
  const now = Date.now();
  const lastCaptureAt = recentBackgroundCaptures.get(tab.id) || 0;
  if (reason !== "extension_installed_or_updated" && now - lastCaptureAt < BACKGROUND_CAPTURE_MIN_INTERVAL_MS) {
    return { ok: false, skipped: true, reason: "background_capture_throttled" };
  }
  recentBackgroundCaptures.set(tab.id, now);
  await ensureCaptureScripts(tab);
  try {
    const result = await chrome.tabs.sendMessage(tab.id, {
      type: "polylogue.capturePage",
      reason
    });
    if (result?.ok) {
      const envelopeSession = result.envelope?.session || {};
      const provider = result.captureResult?.provider || envelopeSession.provider;
      const providerSessionId = result.captureResult?.provider_session_id || envelopeSession.provider_session_id;
      await updateSessionLedger({
        provider,
        providerSessionId,
        patch: {
          reason,
          tab_id: tab.id,
          tab_url: tab.url || tab.pendingUrl || null,
          capture_mode: envelopeSession.provider_meta?.capture_fidelity || null,
          turn_count: Array.isArray(envelopeSession.turns) ? envelopeSession.turns.length : null,
          attachment_count: Array.isArray(envelopeSession.turns)
            ? envelopeSession.turns.reduce((count, turn) => count + (Array.isArray(turn.attachments) ? turn.attachments.length : 0), 0)
            : null,
          archive_state: result.archiveState || null,
          receiver_request_id: result.captureResult?.receiver_request_id || result.archiveState?.receiver_request_id || null,
          last_error: null,
        },
      });
      await appendCaptureLog({
        ok: true,
        reason,
        provider,
        provider_session_id: providerSessionId,
        tab_id: tab.id,
        archive_state: result.archiveState?.state || null,
        receiver_request_id: result.captureResult?.receiver_request_id || result.archiveState?.receiver_request_id || null,
      });
      await setState({
        online: true,
        captured: true,
        last_capture: result.captureResult || result,
        archive_state: result.archiveState || null,
        provider,
        provider_session_id: providerSessionId,
        capture_mode: envelopeSession.provider_meta?.capture_fidelity || null,
        last_receiver_request_id:
          result.captureResult?.receiver_request_id || result.archiveState?.receiver_request_id || null
      });
      await appendDebugLog({
        stage: "capture_result",
        ok: true,
        reason,
        provider,
        provider_session_id: providerSessionId,
        capture_mode: envelopeSession.provider_meta?.capture_fidelity || null,
        archive_state: result.archiveState?.state || null,
        receiver_request_id: result.captureResult?.receiver_request_id || result.archiveState?.receiver_request_id || null,
      });
    }
    return result;
  } catch (error) {
    await appendCaptureLog({
      ok: false,
      reason,
      tab_id: tab.id,
      tab_url: tab.url || tab.pendingUrl || null,
      error: String(error.message || error),
    });
    await appendDebugLog({
      stage: "capture_result",
      ok: false,
      reason,
      tab_id: tab.id,
      error: String(error.message || error),
    });
    return { ok: false, error: String(error.message || error) };
  }
}

async function captureSupportedTabs(reason) {
  if (!chrome.tabs?.query) return;
  const tabs = await chrome.tabs.query({});
  await Promise.allSettled(tabs.map((tab) => captureTab(tab, reason)));
}

// ---- Outbound posting (reverse channel) ---------------------------------
//
// Disabled by default. The local receiver only serves post commands when its
// own POLYLOGUE_BROWSER_POST_ENABLED=1 guard is set; the extension adds a second
// independent guard (`postingEnabled`, default false) so a misconfigured
// receiver still cannot drive the page without an explicit opt-in here.

async function postingSettings() {
  const stored = await chrome.storage.local.get({ postingEnabled: false });
  return { postingEnabled: Boolean(stored.postingEnabled) };
}

async function savePostingSettings(postingEnabled) {
  await chrome.storage.local.set({ postingEnabled: Boolean(postingEnabled) });
  return postingSettings();
}

function providerTokenForUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) return "chatgpt";
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) return "claude";
  } catch {
    return null;
  }
  return null;
}

function archiveProviderForUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) return "chatgpt";
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) return "claude-ai";
    if (
      parsed.hostname === "grok.com" ||
      parsed.hostname.endsWith(".grok.com") ||
      parsed.hostname === "x.com" ||
      parsed.hostname.endsWith(".x.com") ||
      parsed.hostname === "twitter.com" ||
      parsed.hostname.endsWith(".twitter.com")
    ) {
      return "grok";
    }
  } catch {
    return null;
  }
  return null;
}

function conversationIdForUrl(url) {
  try {
    const parsed = new URL(url || "");
    const parts = parsed.pathname.split("/").filter(Boolean);
    const provider = archiveProviderForUrl(url);
    if (provider === "chatgpt") {
      const marker = parts.indexOf("c");
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
      if (parsed.searchParams.get("temporary-chat") === "true") return null;
      return null;
    }
    if (provider === "claude-ai") {
      return parts[0] === "chat" && parts[1] ? parts[1] : null;
    }
    if (provider === "grok") {
      return parts.find((part, index) => parts[index - 1] === "chat" || parts[index - 1] === "grok") || null;
    }
  } catch {
    return null;
  }
  return null;
}

async function refreshActiveTabArchiveState(tab, reason = "tab_state") {
  const url = tab?.url || tab?.pendingUrl || "";
  const provider = archiveProviderForUrl(url);
  const providerSessionId = conversationIdForUrl(url);
  const throttleKey = `${tab?.id || "active"}:${provider || "unsupported"}:${providerSessionId || "none"}`;
  const now = Date.now();
  const lastCheckedAt = recentActiveTabStateChecks.get(throttleKey) || 0;
  if (now - lastCheckedAt < ACTIVE_TAB_STATE_MIN_INTERVAL_MS) return;
  recentActiveTabStateChecks.set(throttleKey, now);

  try {
    if (provider && providerSessionId) {
      const query = new URLSearchParams({ provider, provider_session_id: providerSessionId });
      const state = await getJson(`/v1/archive-state?${query.toString()}`);
      await setState({
        online: true,
        captured: Boolean(state.captured),
        archive_state: state,
        provider,
        provider_session_id: providerSessionId,
        active_page_state: "conversation",
        active_tab_id: tab?.id || null,
        passive_reason: reason,
        last_receiver_request_id: state.receiver_request_id || null,
      });
      return;
    }

    const status = await getJson("/v1/status");
    await setState({
      online: true,
      captured: false,
      status,
      provider,
      provider_session_id: null,
      active_page_state: provider ? "supported_no_session" : "unsupported",
      active_tab_id: tab?.id || null,
      passive_reason: reason,
      last_receiver_request_id: status.receiver_request_id || null,
    });
  } catch (error) {
    await setState({
      online: false,
      captured: false,
      provider,
      provider_session_id: providerSessionId,
      active_page_state: provider ? "receiver_error" : "unsupported",
      active_tab_id: tab?.id || null,
      passive_reason: reason,
      error: String(error.message || error),
      last_receiver_request_id: error.receiverRequestId || null,
    });
  }
}

async function refreshCurrentActiveTab(reason = "active_tab") {
  if (!chrome.tabs?.query) {
    await refreshReceiverState();
    return;
  }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    await refreshReceiverState();
    return;
  }
  await refreshActiveTabArchiveState(tab, reason);
}

async function ackPostCommand(commandId, result) {
  try {
    await postJson(`/v1/post-commands/${encodeURIComponent(commandId)}/ack`, result);
    pendingPostCommandAcks.delete(commandId);
    inFlightPostCommands.delete(commandId);
    return true;
  } catch (error) {
    await appendCaptureLog({ ok: false, reason: "post_ack", command_id: commandId, error: String(error.message || error) });
    pendingPostCommandAcks.set(commandId, result);
    return false;
  }
}

async function retryPendingPostCommandAcks() {
  for (const [commandId, result] of [...pendingPostCommandAcks.entries()]) {
    await ackPostCommand(commandId, result);
  }
}

async function findTabForCommand(command) {
  if (!chrome.tabs?.query) return null;
  const tabs = await chrome.tabs.query({});
  const provider = command.provider;
  const target = command.target || {};
  const wantNew = !target.conversation_id || target.conversation_id === "new";
  let fallback = null;
  for (const tab of tabs) {
    const url = tab.url || tab.pendingUrl || "";
    if (providerTokenForUrl(url) !== provider) continue;
    if (wantNew) {
      if (conversationIdForUrl(url)) continue;
      if (tab.active) return tab;
      fallback = fallback || tab;
      continue;
    }
    if (conversationIdForUrl(url) === target.conversation_id) return tab;
  }
  return wantNew ? fallback : null;
}

async function dispatchPostCommand(command) {
  if (!command || !command.command_id || inFlightPostCommands.has(command.command_id)) return;
  inFlightPostCommands.add(command.command_id);
  let terminalAckRecorded = false;
  try {
    const tab = await findTabForCommand(command);
    if (!tab?.id) {
      terminalAckRecorded = await ackPostCommand(command.command_id, { status: "failed", detail: "no_matching_tab" });
      return;
    }
    await ensureCaptureScripts(tab);
    let result;
    try {
      result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.postReply", command });
    } catch (error) {
      terminalAckRecorded = await ackPostCommand(command.command_id, {
        status: "failed",
        detail: String(error.message || error),
        observed_url: tab.url || null,
      });
      return;
    }
    terminalAckRecorded = await ackPostCommand(command.command_id, {
      status: result?.status === "submitted" ? "submitted" : "failed",
      detail: result?.detail || null,
      observed_url: result?.observed_url || tab.url || null,
    });
  } finally {
    if (terminalAckRecorded) inFlightPostCommands.delete(command.command_id);
  }
}

async function pollPostCommands() {
  const { postingEnabled } = await postingSettings();
  if (!postingEnabled) return;
  await retryPendingPostCommandAcks();
  for (const provider of ["chatgpt", "claude"]) {
    let body;
    try {
      body = await getJson(`/v1/post-commands?provider=${provider}`);
    } catch {
      continue;
    }
    if (!body?.post_enabled || !Array.isArray(body.commands)) continue;
    for (const command of body.commands) {
      await dispatchPostCommand(command);
    }
  }
}

async function startPostPolling() {
  const { postingEnabled } = await postingSettings();
  if (!postingEnabled) {
    stopPostPolling();
    return;
  }
  if (postPollTimer) return;
  postPollTimer = globalThis.setInterval(() => {
    void pollPostCommands();
  }, POST_POLL_INTERVAL_MS);
  void pollPostCommands();
}

function stopPostPolling() {
  if (postPollTimer) {
    globalThis.clearInterval(postPollTimer);
    postPollTimer = 0;
  }
}

void startPostPolling();

chrome.runtime.onInstalled?.addListener(() => {
  void refreshCurrentActiveTab("extension_installed");
});

chrome.runtime.onStartup?.addListener(() => {
  void refreshCurrentActiveTab("browser_startup");
});

chrome.tabs?.onActivated?.addListener((activeInfo) => {
  void (async () => {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    await refreshActiveTabArchiveState(tab, "tab_activated");
  })();
});

chrome.tabs?.onUpdated?.addListener((tabId, changeInfo, tab) => {
  if (changeInfo?.status !== "complete" && !changeInfo?.url) return;
  void (async () => {
    await refreshActiveTabArchiveState(tab?.id ? tab : await chrome.tabs.get(tabId), "tab_updated");
  })();
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message.type === "polylogue.configureReceiver") {
      const settings = await saveReceiverSettings(message.receiverBaseUrl || DEFAULT_RECEIVER, message.receiverAuthToken || "");
      sendResponse({ ok: true, receiverBaseUrl: settings.baseUrl, authConfigured: Boolean(settings.authToken) });
      return;
    }
    if (message.type === "polylogue.capture") {
      const result = await postJson("/v1/browser-captures", message.envelope);
      const session = message.envelope?.session || {};
      await updateSessionLedger({
        provider: session.provider || result.provider,
        providerSessionId: session.provider_session_id || result.provider_session_id,
        patch: {
          capture_mode: session.provider_meta?.capture_fidelity || null,
          turn_count: Array.isArray(session.turns) ? session.turns.length : null,
          attachment_count: Array.isArray(session.turns)
            ? session.turns.reduce((count, turn) => count + (Array.isArray(turn.attachments) ? turn.attachments.length : 0), 0)
            : null,
          receiver_request_id: result.receiver_request_id || null,
          artifact_ref: result.artifact_ref || null,
          last_error: null,
        },
      });
      await appendCaptureLog({
        ok: true,
        reason: message.reason || "content_script_capture",
        provider: session.provider || result.provider,
        provider_session_id: session.provider_session_id || result.provider_session_id,
        capture_mode: session.provider_meta?.capture_fidelity || null,
        receiver_request_id: result.receiver_request_id || null,
        artifact_ref: result.artifact_ref || null,
      });
      await setState({
        online: true,
        captured: true,
        last_capture: result,
        provider: session.provider || result.provider,
        provider_session_id: session.provider_session_id || result.provider_session_id,
        capture_mode: session.provider_meta?.capture_fidelity || null,
        last_receiver_request_id: result.receiver_request_id || null
      });
      sendResponse(result);
      return;
    }
    if (message.type === "polylogue.archiveState") {
      const query = new URLSearchParams({
        provider: message.provider,
        provider_session_id: message.provider_session_id
      });
      const state = await getJson(`/v1/archive-state?${query.toString()}`);
      await setState({
        online: true,
        captured: Boolean(state.captured),
        archive_state: state,
        provider: message.provider,
        provider_session_id: message.provider_session_id,
        last_receiver_request_id: state.receiver_request_id || null
      });
      sendResponse(state);
      return;
    }
    if (message.type === "polylogue.status") {
      const status = await getJson("/v1/status");
      await setState({
        online: true,
        captured: false,
        status,
        last_receiver_request_id: status.receiver_request_id || null
      });
      sendResponse(status);
      return;
    }
    if (message.type === "polylogue.captureSupportedTabs") {
      await captureSupportedTabs(message.reason || "popup_sync_open_tabs");
      sendResponse({ ok: true });
      return;
    }
    if (message.type === "polylogue.configurePosting") {
      const settings = await savePostingSettings(message.postingEnabled);
      await startPostPolling();
      sendResponse({ ok: true, postingEnabled: settings.postingEnabled });
      return;
    }
    if (message.type === "polylogue.pollPostCommands") {
      await pollPostCommands();
      sendResponse({ ok: true });
      return;
    }
  })().catch(async (error) => {
    await appendCaptureLog({
      ok: false,
      reason: message.type || "runtime_message",
      error: String(error.message || error),
      receiver_request_id: error.receiverRequestId || null,
    });
    await appendDebugLog({
      stage: "runtime_message_error",
      message_type: message.type || "runtime_message",
      receiver_request_id: error.receiverRequestId || null,
      error: String(error.message || error),
    });
    await setState({
      online: false,
      captured: false,
      error: String(error.message || error),
      last_receiver_request_id: error.receiverRequestId || null
    });
    sendResponse({
      ok: false,
      error: String(error.message || error),
      receiver_request_id: error.receiverRequestId || null
    });
  });
  return true;
});
