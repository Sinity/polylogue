const DEFAULT_RECEIVER = "http://127.0.0.1:8765";

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

async function requestHeaders({ hasBody = false } = {}) {
  const settings = await receiverSettings();
  const headers = {};
  if (hasBody) headers["Content-Type"] = "application/json";
  if (settings.authToken) headers.Authorization = `Bearer ${settings.authToken}`;
  return headers;
}

async function postJson(path, payload) {
  const settings = await receiverSettings();
  const response = await fetch(`${settings.baseUrl}${path}`, {
    method: "POST",
    headers: await requestHeaders({ hasBody: true }),
    body: JSON.stringify(payload)
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

async function getJson(path) {
  const settings = await receiverSettings();
  const response = await fetch(`${settings.baseUrl}${path}`, {
    headers: await requestHeaders(),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message.type === "polylogue.configureReceiver") {
      const settings = await saveReceiverSettings(message.receiverBaseUrl || DEFAULT_RECEIVER, message.receiverAuthToken || "");
      sendResponse({ ok: true, receiverBaseUrl: settings.baseUrl, authConfigured: Boolean(settings.authToken) });
      return;
    }
    if (message.type === "polylogue.capture") {
      const result = await postJson("/v1/browser-captures", message.envelope);
      await setState({ online: true, captured: true, last_capture: result });
      sendResponse(result);
      return;
    }
    if (message.type === "polylogue.archiveState") {
      const query = new URLSearchParams({
        provider: message.provider,
        provider_session_id: message.provider_session_id
      });
      const state = await getJson(`/v1/archive-state?${query.toString()}`);
      await setState({ online: true, captured: Boolean(state.captured), archive_state: state });
      sendResponse(state);
      return;
    }
    if (message.type === "polylogue.status") {
      const status = await getJson("/v1/status");
      await setState({ online: true, captured: false, status });
      sendResponse(status);
      return;
    }
  })().catch(async (error) => {
    await setState({ online: false, captured: false, error: String(error.message || error) });
    sendResponse({ ok: false, error: String(error.message || error) });
  });
  return true;
});
