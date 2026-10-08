(function () {
  if (window.__polylogueClaudeCaptureInstalled === 2) return;
  window.__polylogueClaudeCaptureInstalled = 2;

  // In-page Layer 1 (polylogue-ys30): capture-status dot + save action mounted
  // next to each detected message. Reused across every capture trigger below
  // (badge click, popup, background auto-capture) so the dots always reflect
  // the most recent capture outcome for the whole session.
  const MESSAGE_CONTAINER_SELECTOR = '[data-testid*="message"], [data-message-author-role], article';
  let messageLayer = null;

  const nativeAdapterName = "claude-ai-native-v1";
  const nativeFetchRequestMessage = "polylogue.claude.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.claude.nativeFetchResponse";
  const nativeFetchResponses = new Map();
  const nativeAttemptDiagnostics = [];
  let nativeAttemptsDropped = 0;

  function rememberNativeAttempt(diagnostic) {
    nativeAttemptDiagnostics.push({
      attempted_at: new Date().toISOString(),
      ...diagnostic
    });
    if (nativeAttemptDiagnostics.length > 6) {
      const dropped = nativeAttemptDiagnostics.length - 6;
      nativeAttemptDiagnostics.splice(0, dropped);
      nativeAttemptsDropped += dropped;
    }
  }

  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    return parts[0] === "chat" && parts[1] ? parts[1] : null;
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== nativeFetchResponseMessage || !data.requestId) return;
    const pending = nativeFetchResponses.get(data.requestId);
    if (!pending) return;
    nativeFetchResponses.delete(data.requestId);
    pending.resolve({ capture: data.capture || null, error: data.error || null, requestId: data.requestId,
      failure_stage: data.error && ["admission", "provider_fetch", "staging"].includes(data.failure_stage) ? data.failure_stage : "unknown" });
  });

  async function requestNativeCaptureFromPage(conversationId, signal) {
    const requestId = `polylogue-claude-native-fetch-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      nativeFetchResponses.set(requestId, { resolve });
    });
    window.polylogueAssetStream.pageMessage(
      {
        type: nativeFetchRequestMessage,
        requestId,
        conversationId
      },
      chrome.runtime.id
    );
    const onAbort = () => window.polylogueAssetStream.pageMessage({ type: "polylogue.claude.cancelRequest", requestId }, chrome.runtime.id);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
    return responsePromise.finally(() => signal.removeEventListener("abort", onAbort));
  }

  async function performCapture(reason = null, signal) {
    let throttle;
    try { throttle = await chrome.runtime.sendMessage({ type: "polylogue.providerThrottle", provider: "claude-ai" }); }
    catch { return { ok: false, error: "provider_throttle_authority_unavailable" }; }
    if (throttle?.ok !== true) return { ok: false, error: throttle?.outcome || "provider_throttle_authority_unavailable", outcome: throttle?.outcome, retry_after_seconds: throttle?.retry_after_seconds ?? null };
    const nativeId = conversationIdFromUrl();
    if (!nativeId) return { ok: false, error: "native_capture_unavailable" };
    const response = await requestNativeCaptureFromPage(nativeId, signal);
    signal.throwIfAborted();
    const acquired = response?.capture;
    if (acquired?.status === 429 || response?.error === "rate_limited") {
      await chrome.runtime.sendMessage({ type: "polylogue.providerRateLimited", provider: "claude-ai", retry_after: acquired?.retryAfter, request_id: response.requestId, provider_response: { status: acquired?.status, url: acquired?.url } });
      return { ok: false, error: "rate_limited", outcome: "rate_limited" };
    }
    // Every explicit capture acquires the current provider revision.
    const capture = acquired?.ok && acquired.bodyRef ? acquired : null;
    rememberNativeAttempt({ stage: "page_bridge_fetch", ok: acquired?.ok ?? null,
      status: acquired?.status ?? null, accepted: Boolean(capture), error: response?.error || acquired?.error || null, failure_stage: response.failure_stage });
    if (!capture) return { ok: false, error: "native_capture_unavailable", native_attempts: nativeAttemptDiagnostics.slice(), native_attempts_dropped: nativeAttemptsDropped };
    const finalEnvelope = await window.polylogueAssetStream.nativeEnvelope({ provider: "claude-ai", capture, nativeId, signal });
    const captureResult = await window.polylogueCapture.sendCapture(finalEnvelope, reason, signal);
    if (!captureResult?.ok) {
      messageLayer?.reportOutcome({ ok: false });
      return { ok: false, envelope: finalEnvelope, captureResult, error: captureResult?.error || "capture_rejected", timelineRecorded: true };
    }
    const archiveState = await window.polylogueCapture.refreshArchiveState("claude-ai", finalEnvelope.session.provider_session_id);
    messageLayer?.reportOutcome({ ok: true, acceptedIdentities: captureResult.accepted_identities });
    return { ok: true, envelope: finalEnvelope, captureResult, archiveState };
  }

  const captureOperations = new Set();
  function capture(reason = null) {
    const controller = new AbortController();
    const operation = { controller, promise: null };
    operation.promise = performCapture(reason, controller.signal).catch((error) => {
      if (controller.signal.aborted) return { ok: false, error: "capture_cancelled", outcome: "cancelled" };
      throw error;
    }).finally(() => captureOperations.delete(operation));
    captureOperations.add(operation);
    return operation.promise;
  }
  async function cancelCapture() {
    const owned = [...captureOperations];
    for (const operation of owned) operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    await Promise.allSettled(owned.map((operation) => operation.promise));
    return { ok: true, outcome: "cancelled", drained: owned.length };
  }
  window.addEventListener("pagehide", () => { void cancelCapture(); });
  window.polylogueCapture.cancelCapture = cancelCapture;
  window.polylogueCapture.capturePage = capture;
  if (window.polylogueMessageLayer) {
    messageLayer = window.polylogueMessageLayer.mount({
      containerSelector: MESSAGE_CONTAINER_SELECTOR,
      identityForNode: (node) => window.polylogueCapture.identityObservation({
        provider: "claude-ai", conversationId: conversationIdFromUrl(),
        // Only an explicit provider-native message id is authoritative. DOM
        // order, test ids, and visible text cannot authorize a badge.
        messageId: node.getAttribute("data-message-id"),
        text: node.innerText || node.textContent || "", adapterName: nativeAdapterName,
        adapterVersion: chrome.runtime.getManifest().version,
        fidelity: node.getAttribute("data-message-id") ? "native" : "unknown",
      }),
      onSave: () => {
        capture("message_layer_save").catch(() => undefined);
      },
    });
  }
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "polylogue.cancelCapture") {
      cancelCapture().then(sendResponse);
      return true;
    }
    if (message.type !== "polylogue.capturePage") return false;
    capture(message.reason || null).then(sendResponse).catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
})();
