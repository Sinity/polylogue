(function () {
  if (window.polylogueAssetStream?.protocolVersion === 2 && window.polylogueAssetStream?.implementationRevision >= 3) return;
  const chunkMessage = "polylogue.assetChunk";
  const ackMessage = "polylogue.assetChunkAck";
  const origin = window.location.origin;
  const CHUNK_BYTES = 48 * 1024;
  const pendingAcks = new Map();
  const owned = new Map();
  const responses = new Map();
  const responseControllers = new Map();
  const headerControllers = new Set();
  const validOwner = owner => typeof owner === "string" && /^[a-p]{32}$/.test(owner);
  const isContent = typeof chrome !== "undefined" && validOwner(chrome.runtime?.id) && Boolean(chrome.runtime?.sendMessage);
  const localOwner = isContent ? chrome.runtime.id : null;
  const owners = new Map();
  function requireOwner(owner) {
    if (!validOwner(owner)) throw new Error("capture_transport_owner_unavailable");
    return owner;
  }
  function pageMessage(data, ownerId = localOwner) {
    requireOwner(ownerId);
    window.postMessage({ ...data, type: `polylogue.page.v2.${ownerId}.${data.type.slice("polylogue.".length)}` }, origin);
  }
  function readPageMessage(event, ownerId = localOwner) {
    if (event.source !== window || event.origin !== origin) return null;
    const data = event.data;
    const match = typeof data?.type === "string" && data.type.match(/^polylogue\.page\.v2\.([a-p]{32})\.(.+)$/);
    if (!match || (ownerId !== null && match[1] !== ownerId)) return null;
    return { ...data, type: `polylogue.${match[2]}`, ownerId: match[1] };
  }
  // MAIN keeps only current installation eligibility, never credentials.
  // Explicit captures do not depend on ambient enablement.
  let registration = Promise.resolve();
  let registrationActive = true;
  function registerOwner() {
    registration = registration.catch(() => undefined).then(async () => {
      const stored = await chrome.storage.local.get({ polylogueAmbientSettings: {}, polylogueReceiverPairing: null });
      const settings = stored.polylogueAmbientSettings || {};
      const pairing = stored.polylogueReceiverPairing;
      const eligible = settings.enabled !== false && settings.automatic_capture_enabled !== false
        && settings.disabled_sites?.[window.location.hostname] !== true
        && Boolean(pairing?.receiver_id) && pairing.state !== "mismatch";
      if (registrationActive) pageMessage({ type: "polylogue.ownerRegistration", eligible });
    });
    void registration.catch(() => undefined);
  }
  const registrationChanged = (changes, area) => {
    if (area === "local" && (changes.polylogueAmbientSettings || changes.polylogueReceiverPairing)) registerOwner();
  };
  if (isContent) {
    registerOwner();
    try { chrome.storage?.onChanged?.addListener(registrationChanged); } catch { /* No passive ownership is inferred. */ }
  }
  if (isContent) chrome.runtime.onMessage?.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.stagingOwner") return false;
    sendResponse({ ok: true }); return false;
  });
  async function runtimeRequest(message, signal) {
    for (;;) {
      signal.throwIfAborted();
      try {
        const result = await chrome.runtime.sendMessage(message);
        if (result?.error !== "capture_acquisition_in_progress") return result;
      } catch (error) {
        // Chrome may terminate the worker after a durable write, before its
        // response. Retrying the same nonce/chunk/seal cannot repeat traffic.
        if (!/message port closed|message channel closed|receiving end does not exist|could not establish connection|extension context invalidated/i.test(String(error.message || error))) throw error;
        if (/extension context invalidated/i.test(String(error.message || error))) throw error;
      }
      await new Promise((resolve, reject) => {
        const onAbort = () => { clearTimeout(timer); reject(signal.reason); };
        const timer = setTimeout(() => { signal.removeEventListener("abort", onAbort); resolve(); }, 100);
        signal.addEventListener("abort", onAbort, { once: true });
        if (signal.aborted) { signal.removeEventListener("abort", onAbort); onAbort(); }
      });
    }
  }
  function encode(bytes) {
    let text = "";
    for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
    return window.btoa(text);
  }
  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== origin) return;
    if (isContent && event.data?.type === "polylogue.page.v2.registrationRequested") { registerOwner(); return; }
    const data = readPageMessage(event);
    if (!data) return;
    const ownerId = data.ownerId;
    if (data.type === "polylogue.ownerRegistration") {
      if (typeof data.eligible === "boolean") owners.set(ownerId, data.eligible);
      return;
    }
    if (data.type === "polylogue.responseReady" || data.type === "polylogue.responseSealed") {
      const pending = responses.get(`${ownerId}:${data.requestId}:${data.type}`);
      if (pending) { responses.delete(`${ownerId}:${data.requestId}:${data.type}`); data.ok ? pending.resolve(data) : pending.reject(new Error(data.error)); }
    }
    if (isContent && data.type === "polylogue.nativeBundleRequest" && data.requestId) {
      try {
        const result = await chrome.runtime.sendMessage({ type: `polylogue.nativeBundle.${data.operation}`, provider: data.provider,
          native_id: data.nativeId, bundle_id: data.bundleId, bundle_ref: data.bundleRef, name: data.name, outcome: data.outcome });
        pageMessage({ type: "polylogue.nativeBundleResponse", requestId: data.requestId, ...result }, ownerId);
      } catch (error) { pageMessage({ type: "polylogue.nativeBundleResponse", requestId: data.requestId, ok: false, error: String(error.message || error) }, ownerId); }
    }
    if (data.type === "polylogue.nativeBundleResponse") {
      const pending = responses.get(`${ownerId}:${data.requestId}:${data.type}`);
      if (pending) { responses.delete(`${ownerId}:${data.requestId}:${data.type}`); data.ok ? pending.resolve(data) : pending.reject(new Error(data.error)); }
    }
    if (isContent && data.type === "polylogue.responseCancel") {
      responseControllers.get(`${ownerId}:${data.requestId}`)?.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    }
    if (isContent && data.type === "polylogue.responseStart" && data.requestId && data.provider) {
      const controller = new AbortController(); responseControllers.set(`${ownerId}:${data.requestId}`, controller);
      try {
        const result = await request({ provider: data.provider, ownerId, requestId: data.requestId, signal: controller.signal, purpose: { kind: data.kind === "provider-inventory" ? "provider-inventory" : "native-response", source_url: data.source_url || null, response_metadata: data.response_metadata || null, capture_bundle: data.capture_bundle || null, observation_only: data.observation_only === true, queue_context: data.queue_context || null, account_handle: data.account_handle || null, invocation_ref: data.invocation_ref || null }, start: async () => {
          const complete = waitResponse(data.requestId, "polylogue.responseComplete", controller.signal, ownerId);
          pageMessage({ type: "polylogue.responseReady", requestId: data.requestId, ok: true }, ownerId);
          const message = await complete;
          return message.result || { status: "acquired", response_metadata: message.response_metadata,
            asset: { streamed: true, size_bytes: message.size_bytes } };
        } });
        pageMessage({ type: "polylogue.responseSealed", requestId: data.requestId, ok: true, asset: result.asset, result }, ownerId);
      } catch (error) {
        for (const type of ["polylogue.responseReady", "polylogue.responseSealed"]) pageMessage({ type, requestId: data.requestId, ok: false, error: String(error.message || error) }, ownerId);
      } finally { responseControllers.delete(`${ownerId}:${data.requestId}`); }
    }
    if (data.type === "polylogue.responseComplete") {
      const pending = responses.get(`${ownerId}:${data.requestId}:${data.type}`);
      if (pending) { responses.delete(`${ownerId}:${data.requestId}:${data.type}`); data.ok ? pending.resolve(data) : pending.reject(new Error(data.error)); }
    }
    if (data.type === ackMessage) {
      const ack = pendingAcks.get(`${ownerId}:${data.requestId}:${data.sequence}`);
      if (ack) { pendingAcks.delete(`${ownerId}:${data.requestId}:${data.sequence}`); data.ok ? ack.resolve() : ack.reject(new Error(data.error)); }
    }
    if (!isContent || data.type !== chunkMessage) return;
    const operation = owned.get(`${ownerId}:${data.requestId}`);
    if (!operation) return;
    try {
      operation.signal.throwIfAborted();
      const result = await runtimeRequest({ type: "polylogue.asset.chunk", ref: operation.ref, sequence: data.sequence, base64: data.base64 }, operation.signal);
      if (!result?.ok) throw new Error(result?.error || "capture_staging_write_failed");
      pageMessage({ type: ackMessage, requestId: data.requestId, sequence: data.sequence, ok: true }, ownerId);
    } catch (error) {
      pageMessage({ type: ackMessage, requestId: data.requestId, sequence: data.sequence, ok: false, error: String(error.message || error) }, ownerId);
    }
  });
  function waitResponse(requestId, type, signal, ownerId = localOwner) {
    const key = `${requireOwner(ownerId)}:${requestId}:${type}`;
    let onAbort;
    const promise = new Promise((resolve, reject) => {
      onAbort = () => { responses.delete(key); reject(signal.reason); };
      responses.set(key, { resolve, reject });
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) onAbort();
    });
    return promise.finally(() => signal.removeEventListener("abort", onAbort));
  }
  async function prepareResponse(provider, signal, sourceUrl, captureBundle = null, kind = "native-response", observationOnly = false, queueContext = null, accountHandle = null, borrowedResponse = false, invocationRef = null, ownerId = localOwner) {
    signal.throwIfAborted();
    requireOwner(ownerId);
    const requestId = window.crypto.randomUUID();
    const ready = waitResponse(requestId, "polylogue.responseReady", signal, ownerId);
    // A provider request may start only after the background owner has
    // durably admitted this acquisition. The terminal response also drains
    // cancellation; it is not rejected merely because the producer aborted.
    const terminalController = new AbortController();
    const sealed = waitResponse(requestId, "polylogue.responseSealed", terminalController.signal, ownerId);
    void sealed.catch(() => undefined);
    const abort = () => pageMessage({ type: "polylogue.responseCancel", requestId }, ownerId);
    signal.addEventListener("abort", abort, { once: true });
    pageMessage({ type: "polylogue.responseStart", requestId, provider, kind, source_url: sourceUrl, capture_bundle: captureBundle, observation_only: observationOnly, queue_context: queueContext, account_handle: accountHandle, invocation_ref: invocationRef }, ownerId);
    try { await ready; }
    catch (error) {
      await sealed.catch(() => undefined);
      signal.removeEventListener("abort", abort);
      throw error;
    }
    let finished = false;
    const fail = async (error) => {
      if (!finished) {
        finished = true;
        pageMessage({ type: "polylogue.responseComplete", requestId, ok: false, error: String(error.message || error) }, ownerId);
      }
      await sealed.catch(() => undefined);
      signal.removeEventListener("abort", abort);
    };
    return { fail, consume: async (response) => {
    try {
      const metadata = { status: response.status, content_type: response.headers.get("content-type"), retry_after: response.headers.get("retry-after") };
      if (!response.ok || !String(metadata.content_type || "").includes("application/json")) {
        finished = true;
        pageMessage({ type: "polylogue.responseComplete", requestId, ok: true,
          result: { status: response.status === 429 ? "rate_limited" : "http_error", http_status: response.status,
            retry_after: metadata.retry_after, response_url: sourceUrl } }, ownerId);
        await sealed;
        const cancelled = response.body?.cancel().catch(() => undefined);
        if (!borrowedResponse) await cancelled;
        signal.removeEventListener("abort", abort);
        return null;
      }
      const result = await stream(response, requestId, signal, { borrowedResponse, ownerId });
      finished = true;
      pageMessage({ type: "polylogue.responseComplete", requestId, ok: true, size_bytes: result.size_bytes, response_metadata: metadata }, ownerId);
      const asset = (await sealed).asset.staged_asset;
      signal.removeEventListener("abort", abort);
      return asset;
    } catch (error) {
      await fail(error);
      const cancelled = response.body?.cancel().catch(() => undefined);
      if (!borrowedResponse) await cancelled;
      throw error;
    }
    } };
  }
  // Observe only a clone borrowed from an app-owned request. Explicit capture
  // fetches use prepareResponse and own their complete response lifetime.
  async function stageResponse(response, provider, signal, sourceUrl = response.url, captureBundle = null, kind = "native-response", ownerId = localOwner) {
    try {
      const prepared = await prepareResponse(provider, signal, sourceUrl, captureBundle, kind, !captureBundle, null, null, true, null, ownerId);
      return await prepared.consume(response);
    } catch (error) {
      // A refused admission never enters consume, but still owns this clone.
      // Its app-owned tee may remain unread, so request cancellation without
      // waiting on that branch or altering the original response.
      void response.body?.cancel().catch(() => undefined);
      throw error;
    }
  }
  window.addEventListener("pagehide", () => {
    for (const controller of responseControllers.values()) controller.abort("provider_page_closed");
    for (const controller of headerControllers) controller.abort("provider_page_closed");
    owners.clear();
    registrationActive = false;
    if (isContent) { chrome.storage?.onChanged?.removeListener?.(registrationChanged); }
    if (isContent) void chrome.runtime.sendMessage({ type: "polylogue.releaseNativeCache" }).catch(() => undefined);
  });
  async function sendChunk(requestId, sequence, bytes, signal, ownerId) {
    signal.throwIfAborted();
    const key = `${requireOwner(ownerId)}:${requestId}:${sequence}`;
    let reject;
    const wait = new Promise((resolve, fail) => { reject = fail; pendingAcks.set(key, { resolve, reject: fail }); });
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    pageMessage({ type: chunkMessage, requestId, sequence, base64: encode(bytes) }, ownerId);
    try { await wait; } finally { signal.removeEventListener("abort", onAbort); pendingAcks.delete(key); }
  }
  async function stream(response, requestId, signal, { borrowedResponse = false, ownerId = localOwner } = {}) {
    requireOwner(ownerId);
    if (!response.body?.getReader) throw new Error("asset_body_stream_unavailable");
    const reader = response.body.getReader();
    let sequence = 0; let size = 0;
    const abort = () => { void reader.cancel(signal.reason).catch(() => undefined); };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    try {
      for (;;) {
        signal.throwIfAborted();
        const { done, value } = await reader.read();
        signal.throwIfAborted();
        if (done) break;
        for (let offset = 0; offset < value.length; offset += CHUNK_BYTES) {
          const chunk = value.subarray(offset, offset + CHUNK_BYTES);
          await sendChunk(requestId, sequence++, chunk, signal, ownerId); size += chunk.length;
        }
      }
      return { streamed: true, size_bytes: size };
    } finally {
      signal.removeEventListener("abort", abort);
      // Cancelling a cloned response must not wait for the app-owned tee
      // branch on any refusal or cancellation. Owned provider responses still
      // await their own cancellation; borrowed app requests keep their lifetime.
      const cancelled = reader.cancel().catch(() => undefined);
      if (!borrowedResponse) await cancelled;
      reader.releaseLock();
    }
  }
  async function request({ provider, requestId, signal, start, purpose = {}, ownerId = localOwner }) {
    requireOwner(ownerId);
    let begun;
    try { begun = await runtimeRequest({ type: "polylogue.asset.begin", provider, request_id: requestId, ...purpose }, signal); }
    catch (error) {
      void chrome.runtime.sendMessage({ type: "polylogue.asset.cancelRequest", provider, request_id: requestId }).catch(() => undefined);
      throw error;
    }
    if (!begun?.ok) {
      const error = new Error(begun?.error || "capture_staging_unavailable");
      error.outcome = begun?.outcome; error.retryAfterSeconds = begun?.retry_after_seconds; throw error;
    }
    if (begun.result) return begun.result;
    const ref = begun.ref;
    owned.set(`${ownerId}:${requestId}`, { ref, signal });
    let sealed = false;
    let cancellation = null;
    const cancel = () => { cancellation = chrome.runtime.sendMessage({ type: "polylogue.asset.cancelRequest", provider, request_id: requestId }).then(
      (result) => ({ result }), (error) => ({ error }),
    ); };
    signal.addEventListener("abort", cancel, { once: true });
    try {
      signal.throwIfAborted();
      const result = await start();
      signal.throwIfAborted();
      if (result?.http_status === 429) {
        const recorded = await runtimeRequest({ type: "polylogue.providerRateLimited", provider, claim: ref,
          request_id: requestId, retry_after: result.retry_after || null,
          provider_response: { status: 429, url: result.response_url || null } }, signal);
        if (!recorded?.ok) throw new Error(recorded?.error || "provider_throttle_authority_unavailable");
      }
      if (result?.status !== "acquired" || result.asset?.streamed !== true) return result;
      const complete = await runtimeRequest({ type: "polylogue.asset.seal", ref, result }, signal);
      if (!complete?.ok) throw new Error(complete?.error || "capture_staging_seal_failed");
      sealed = true;
      return { ...result, asset: { ...result.asset, ...complete.asset } };
    } finally {
      signal.removeEventListener("abort", cancel);
      if (cancellation) await cancellation;
      owned.delete(`${ownerId}:${requestId}`);
      if (!sealed) await chrome.runtime.sendMessage({ type: "polylogue.asset.discard", ref }).catch(() => undefined);
    }
  }
  async function nativeHeaders(provider, capture) {
    const controller = new AbortController(); headerControllers.add(controller);
    const onAbort = () => { void chrome.runtime.sendMessage({ type: "polylogue.cancelNativeCapture", provider, raw_ref: capture.bodyRef }).catch(() => undefined); };
    controller.signal.addEventListener("abort", onAbort, { once: true });
    try {
      const result = await chrome.runtime.sendMessage({ type: "polylogue.nativeCaptureHeader", provider, raw_ref: capture.bodyRef, related_refs: capture.relatedRefs || {}, acquisition: capture.acquisition || null, invocation_ref: capture.invocationRef || null });
      controller.signal.throwIfAborted();
      if (!result?.ok) throw new Error(result?.error || "native_capture_header_unavailable");
      return result;
    } finally { headerControllers.delete(controller); controller.signal.removeEventListener("abort", onAbort); }
  }
  async function settleNativeHeaders(promise, signal) {
    signal.throwIfAborted();
    const abort = () => { for (const controller of headerControllers) controller.abort(signal.reason); };
    signal.addEventListener("abort", abort, { once: true });
    try { await promise; signal.throwIfAborted(); }
    finally { signal.removeEventListener("abort", abort); }
  }
  async function restoreNative(provider, nativeId, signal, invocationRef = null) {
    signal.throwIfAborted();
    const requestId = crypto.randomUUID(); let cancellation = null;
    const cancel = () => { cancellation = chrome.runtime.sendMessage({ type: "polylogue.cancelNativeRecovery", provider, request_id: requestId }); };
    signal.addEventListener("abort", cancel, { once: true });
    try {
      const result = await chrome.runtime.sendMessage({ type: "polylogue.restoreNativeCapture", provider, native_id: nativeId, request_id: requestId, invocation_ref: invocationRef });
      signal.throwIfAborted();
      if (!result?.ok) throw new Error(result?.error || "native_capture_recovery_unavailable");
      return result.capture;
    } finally { signal.removeEventListener("abort", cancel); if (cancellation) await cancellation; }
  }
  async function nativeBundle(operation, fields, signal, ownerId = localOwner) {
    signal.throwIfAborted();
    const requestId = crypto.randomUUID(); const result = waitResponse(requestId, "polylogue.nativeBundleResponse", signal, ownerId);
    let cancellation = null;
    const cancel = () => {
      if (!fields.bundleRef) return;
      const cancellationId = crypto.randomUUID();
      cancellation = waitResponse(cancellationId, "polylogue.nativeBundleResponse", new AbortController().signal, ownerId);
      void cancellation.catch(() => undefined);
      pageMessage({ type: "polylogue.nativeBundleRequest", requestId: cancellationId, operation: "cancel", provider: "grok", bundleRef: fields.bundleRef }, ownerId);
    };
    signal.addEventListener("abort", cancel, { once: true });
    pageMessage({ type: "polylogue.nativeBundleRequest", requestId, operation, provider: "grok", ...fields }, ownerId);
    try { return await result; }
    finally { signal.removeEventListener("abort", cancel); if (cancellation) await cancellation; }
  }
  async function nativeEnvelope({ provider, capture, nativeId, signal, attribution = {}, summaryOnly = false }) {
    const onAbort = () => { void chrome.runtime.sendMessage({ type: "polylogue.cancelNativeCapture", provider, raw_ref: capture.bodyRef }).catch(() => undefined); };
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      signal.throwIfAborted();
      const result = await chrome.runtime.sendMessage({ type: summaryOnly ? "polylogue.nativeCaptureSummary" : "polylogue.normalizeNativeCapture", provider, raw_ref: capture.bodyRef,
        related_refs: capture.relatedRefs || {}, acquisition: capture.acquisition || null, native_id: nativeId, attribution, invocation_ref: capture.invocationRef || null, native_request_id: capture.nativeRequestId });
      signal.throwIfAborted();
      if (!result?.ok) throw new Error(result?.error || "native_normalization_failed");
      return summaryOnly ? result.summary : result.envelope;
    } finally { signal.removeEventListener("abort", onAbort); }
  }
  window.polylogueAssetStream = { protocolVersion: 2, implementationRevision: 3, pageMessage, readPageMessage, eligibleOwners: () => [...owners].filter(([, eligible]) => eligible).map(([owner]) => owner), stream, request, prepareResponse, stageResponse, nativeHeaders, settleNativeHeaders, restoreNative, nativeEnvelope, nativeBundle };
  if (!isContent) window.postMessage({ type: "polylogue.page.v2.registrationRequested" }, origin);
})();
