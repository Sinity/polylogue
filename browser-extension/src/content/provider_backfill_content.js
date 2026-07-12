(function () {
  const requestType = "polylogue.backfill.pageRequest";
  const responseType = "polylogue.backfill.pageResponse";
  const timeoutMs = 57000;
  const pending = new Map();

  if (window.__polylogueBackfillContentBridgeInstalled) return;
  window.__polylogueBackfillContentBridgeInstalled = true;

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== responseType || typeof data.requestId !== "string") return;
    const entry = pending.get(data.requestId);
    if (!entry) return;
    pending.delete(data.requestId);
    window.clearTimeout(entry.timeout);
    entry.resolve(data);
  });

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.type !== requestType) return false;
    if (sender?.id && sender.id !== chrome.runtime.id) {
      sendResponse({ ok: false, error: "backfill_bridge_sender_mismatch" });
      return false;
    }
    const requestId = globalThis.crypto?.randomUUID?.() || `backfill-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const response = new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        pending.delete(requestId);
        resolve({ error: "backfill_bridge_response_timeout" });
      }, timeoutMs);
      pending.set(requestId, { resolve, timeout });
    });
    window.postMessage({
      type: requestType,
      requestId,
      provider: message.provider,
      operation: message.operation,
      params: message.params || {},
    }, window.location.origin);
    response.then((result) => sendResponse(result.error
      ? { ok: false, error: result.error }
      : { ok: true, response: result.response }));
    return true;
  });
})();
