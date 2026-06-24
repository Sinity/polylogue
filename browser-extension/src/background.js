const DEFAULT_RECEIVER = "http://127.0.0.1:8765";
const BACKGROUND_CAPTURE_MIN_INTERVAL_MS = 30000;
const recentBackgroundCaptures = new Map();

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

async function setState(state) {
  await chrome.storage.local.set({ polylogueState: { ...state, updated_at: new Date().toISOString() } });
  const text = state.captured ? "ok" : state.online ? "on" : "off";
  await chrome.action.setBadgeText({ text });
  await chrome.action.setBadgeBackgroundColor({ color: state.captured ? "#1f7a4d" : state.online ? "#805ad5" : "#9b2c2c" });
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
  const response = await fetch(`${settings.baseUrl}${path}`, {
    method: "POST",
    headers: await requestHeaders({ hasBody: true, requestId }),
    body: JSON.stringify(payload)
  });
  const receiverRequestId = response.headers.get("X-Request-ID") || requestId;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `HTTP ${response.status}`);
    error.receiverRequestId = receiverRequestId;
    throw error;
  }
  return { ...body, receiver_request_id: receiverRequestId };
}

async function getJson(path) {
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  const response = await fetch(`${settings.baseUrl}${path}`, {
    headers: await requestHeaders({ requestId }),
  });
  const receiverRequestId = response.headers.get("X-Request-ID") || requestId;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `HTTP ${response.status}`);
    error.receiverRequestId = receiverRequestId;
    throw error;
  }
  return { ...body, receiver_request_id: receiverRequestId };
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
      await setState({
        online: true,
        captured: true,
        last_capture: result.captureResult || result,
        archive_state: result.archiveState || null,
        last_receiver_request_id:
          result.captureResult?.receiver_request_id || result.archiveState?.receiver_request_id || null
      });
    }
    return result;
  } catch (error) {
    return { ok: false, error: String(error.message || error) };
  }
}

async function captureSupportedTabs(reason) {
  if (!chrome.tabs?.query) return;
  const tabs = await chrome.tabs.query({});
  await Promise.allSettled(tabs.map((tab) => captureTab(tab, reason)));
}

chrome.runtime.onInstalled?.addListener(() => {
  void captureSupportedTabs("extension_installed_or_updated");
});

chrome.runtime.onStartup?.addListener(() => {
  void captureSupportedTabs("browser_started");
});

chrome.tabs?.onUpdated?.addListener((_tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    void captureTab(tab, "tab_updated");
  }
});

chrome.tabs?.onActivated?.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  if (tab) void captureTab(tab, "tab_activated");
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
      await setState({
        online: true,
        captured: true,
        last_capture: result,
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
  })().catch(async (error) => {
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
