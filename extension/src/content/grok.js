(function () {
  if (window.__polylogueGrokCaptureInstalled === 2) return;
  window.__polylogueGrokCaptureInstalled = 2;

  const nativeCaptureMessage = "polylogue.grok.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.grok.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.grok.nativeFetchResponse";
  const assetFetchRequestMessage = "polylogue.grok.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.grok.assetFetchResponse";

  let nativeCapture = null;
  let nativeHeaderPromise = Promise.resolve();
  const nativeFetchResponses = new Map();
  const assetResponses = new Map();
  const nativeAttemptDiagnostics = [];
  let nativeAttemptsDropped = 0;

  function rememberNativeAttempt(diagnostic) {
    nativeAttemptDiagnostics.push({ attempted_at: new Date().toISOString(), ...diagnostic });
    if (nativeAttemptDiagnostics.length > 8) {
      const dropped = nativeAttemptDiagnostics.length - 8;
      nativeAttemptDiagnostics.splice(0, dropped);
      nativeAttemptsDropped += dropped;
    }
  }

  // grok.com conversation URLs are /c/<uuid> (verified live 2026-07-31).
  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    const marker = parts.indexOf("c");
    if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
    return null;
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type === nativeCaptureMessage && data.capture) {
      const capture = data.capture;
      if (capture.ok && capture.bodyRef) nativeHeaderPromise = window.polylogueAssetStream.nativeHeaders("grok", capture).then(({ cache: { headers, capture: selected } }) => {
        if (String(headers.conversationId || headers.id || "") !== conversationIdFromUrl()) return;
        if (window.polylogueCapture.cacheCaptureIsNewer(nativeCapture, selected)) nativeCapture = selected;
      }).catch(() => undefined);
      return;
    }
    if (data.type === nativeFetchResponseMessage && data.requestId) {
      const pending = nativeFetchResponses.get(data.requestId);
      if (!pending) return;
      nativeFetchResponses.delete(data.requestId);
      pending.resolve({ capture: data.capture || null, error: data.error || null, requestId: data.requestId });
      return;
    }
    if (data.type === assetFetchResponseMessage && data.requestId) {
      const pending = assetResponses.get(data.requestId);
      if (!pending) return;
      assetResponses.delete(data.requestId);
      pending.resolve(data.outcome && typeof data.outcome === "object" ? data.outcome : { status: "request_failed", detail: "bridge_response_missing" });
    }
  });

  async function requestNativeCaptureFromPage(conversationId, signal) {
    const requestId = `polylogue-grok-native-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => nativeFetchResponses.set(requestId, { resolve }));
    const onAbort = () => window.polylogueAssetStream.pageMessage({ type: "polylogue.grok.cancelRequest", requestId }, chrome.runtime.id);
    signal.addEventListener("abort", onAbort, { once: true });
    window.polylogueAssetStream.pageMessage({ type: nativeFetchRequestMessage, requestId, conversationId }, chrome.runtime.id);
    if (signal.aborted) onAbort();
    return responsePromise.finally(() => signal.removeEventListener("abort", onAbort));
  }

  function requestAssetFromPage(request, signal, acquisition) {
    const requestId = `polylogue-grok-asset-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    return window.polylogueAssetStream.request({ provider: "grok", requestId, signal, purpose: { acquisition }, start: () => {
      const responsePromise = new Promise((resolve) => assetResponses.set(requestId, { resolve }));
      const onAbort = () => window.polylogueAssetStream.pageMessage({ type: "polylogue.grok.cancelRequest", requestId }, chrome.runtime.id);
      signal.addEventListener("abort", onAbort, { once: true });
      window.polylogueAssetStream.pageMessage({ type: assetFetchRequestMessage, requestId, request }, chrome.runtime.id);
      if (signal.aborted) onAbort();
      return responsePromise.finally(() => signal.removeEventListener("abort", onAbort));
    } });
  }

  async function acquireAssets(descriptors, signal, message) {
    const outcome = {
      attempted: descriptors.length,
      acquired: 0,
      failed: [],
      status_counts: {},
    };
    const attachments = [];
    for (const descriptor of descriptors) {
      signal.throwIfAborted();
      const resolvable = descriptor.url;
      if (!resolvable) {
        outcome.failed.push({ provider_attachment_id: descriptor.provider_attachment_id, status: "no_resolvable_source" });
        outcome.status_counts.no_resolvable_source = (outcome.status_counts.no_resolvable_source || 0) + 1;
        attachments.push({ ...descriptor, provider_meta: { ...descriptor.provider_meta, byte_acquisition: "no_resolvable_source" } });
        continue;
      }
      const request = { key: descriptor.url, name: descriptor.name };
      let result;
      try { result = await requestAssetFromPage(request, signal, { raw_id: message.capture_ref, native_id: message.nativeId, record_key: message.recordKey, attachment_id: descriptor.provider_attachment_id, attachment_ordinal: message.attachmentOrdinal }); }
      catch (error) {
        signal.throwIfAborted();
        result = { status: error.outcome === "rate_limited" ? "rate_limited" : "request_failed", detail: typeof error.code === "string" ? error.code : (error.message || "request_failed") };
      }
      signal.throwIfAborted();
      const status = result.http_status === 429 ? "rate_limited" : typeof result.status === "string" ? result.status : "request_failed";
      const contentSha256 = result.asset && result.asset.sha256;
      const acquiredIsValid =
        status === "acquired" && result.asset && result.asset.staged_asset && typeof contentSha256 === "string" && /^[0-9a-f]{64}$/.test(contentSha256);
      outcome.status_counts[status] = (outcome.status_counts[status] || 0) + 1;
      if (acquiredIsValid) {
        outcome.acquired += 1;
        attachments.push({
          ...descriptor,
          mime_type: result.asset.mime_type || descriptor.mime_type,
          size_bytes: result.asset.size_bytes || descriptor.size_bytes,
          staged_asset: result.asset.staged_asset,
          provider_meta: { ...descriptor.provider_meta, content_sha256: contentSha256, byte_acquisition: "acquired" },
        });
      } else {
        outcome.failed.push({ provider_attachment_id: descriptor.provider_attachment_id, status, detail: result.detail || null });
        attachments.push({ ...descriptor, provider_meta: { ...descriptor.provider_meta, byte_acquisition: status, byte_acquisition_detail: result.detail || null } });
      }
    }
    return { attachments, outcome };
  }

  async function performCapture(reason = null, requestedConversationId = null, deferReceiver = false, signal) {
    let throttle;
    try { throttle = await chrome.runtime.sendMessage({ type: "polylogue.providerThrottle", provider: "grok" }); }
    catch { return { ok: false, error: "provider_throttle_authority_unavailable" }; }
    if (throttle?.ok !== true) return { ok: false, error: throttle?.outcome || "provider_throttle_authority_unavailable", outcome: throttle?.outcome, retry_after_seconds: throttle?.retry_after_seconds ?? null };
    const nativeId = requestedConversationId || conversationIdFromUrl();
    if (!nativeId) return { ok: false, error: "native_capture_unavailable" };
    nativeCapture = await window.polylogueAssetStream.restoreNative("grok", nativeId, signal);
    const response = await requestNativeCaptureFromPage(nativeId, signal);
    signal.throwIfAborted();
    const acquired = response?.capture;
    if (acquired?.status === 429 || response?.error === "rate_limited") {
      await chrome.runtime.sendMessage({ type: "polylogue.providerRateLimited", provider: "grok", retry_after: acquired?.retryAfter || null, request_id: response.requestId, provider_response: { status: acquired?.status, url: acquired?.url } });
      return { ok: false, error: "rate_limited", outcome: "rate_limited" };
    }
    await window.polylogueAssetStream.settleNativeHeaders(nativeHeaderPromise, signal);
    signal.throwIfAborted();
    const cached = nativeCapture?.ok && nativeCapture.bodyRef && String(nativeCapture.url || "").includes(`/conversations/${nativeId}`) ? nativeCapture : null;
    const source = acquired?.ok && acquired.bodyRef ? acquired : cached;
    rememberNativeAttempt({ stage: "page_bridge_fetch", ok: acquired?.ok ?? null, status: acquired?.status ?? null,
      accepted: Boolean(source), error: response?.error || acquired?.error || null });
    if (!source) return { ok: false, error: "native_capture_unavailable", native_attempts: nativeAttemptDiagnostics.slice(), native_attempts_dropped: nativeAttemptsDropped };
    const envelope = await window.polylogueAssetStream.nativeEnvelope({ provider: "grok", capture: source, nativeId, signal,
      attribution: { acquisition: source.acquisition || {} } });
    if (deferReceiver) return { ok: true, envelope, deferred: true };
    const captureResult = await window.polylogueCapture.sendCapture(envelope, reason, signal);
    if (!captureResult?.ok) return { ok: false, envelope, captureResult, error: captureResult?.error || "capture_rejected", timelineRecorded: true };
    const archiveState = await window.polylogueCapture.refreshArchiveState("grok", envelope.session.provider_session_id);
    return { ok: true, envelope, captureResult, archiveState };
  }

  const recordOperations = new Set();
  function acquireRecord(message) {
    const controller = new AbortController();
    const operation = { controller, captureRef: message.capture_ref, promise: null };
    operation.promise = acquireAssets(message.attachments || [], controller.signal, message)
      .then((acquisition) => ({ ok: true, acquisition }))
      .catch((error) => ({ ok: false, error: controller.signal.aborted ? "capture_cancelled" : String(error.message || error) }))
      .finally(() => recordOperations.delete(operation));
    recordOperations.add(operation);
    return operation.promise;
  }
  async function cancelRecord(captureRef) {
    const owned = [...recordOperations].filter((operation) => operation.captureRef === captureRef);
    for (const operation of owned) operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    await Promise.allSettled(owned.map((operation) => operation.promise));
    return { ok: true, outcome: "cancelled" };
  }

  const captureOperations = new Set();
  function capture(reason = null, requestedConversationId = null, deferReceiver = false) {
    const controller = new AbortController();
    const operation = { controller, promise: null };
    operation.promise = performCapture(reason, requestedConversationId, deferReceiver, controller.signal).catch((error) => {
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
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "polylogue.acquireRecordAssets") { acquireRecord(message).then(sendResponse); return true; }
    if (message.type === "polylogue.cancelRecordAssets") { cancelRecord(message.capture_ref).then(sendResponse); return true; }
    if (message.type === "polylogue.cancelCapture") { cancelCapture().then(sendResponse); return true; }
    if (message.type !== "polylogue.capturePage") return false;
    capture(message.reason || null, message.providerSessionId || null, message.deferReceiver === true)
      .then(sendResponse)
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
})();
