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
