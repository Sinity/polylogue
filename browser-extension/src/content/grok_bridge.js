(function () {
  // MAIN-world companion to src/content/grok.js. Grok is the only provider
  // that previously had no bridge at all -- grok.js hardcoded
  // `native_attempts: []` and scraped the DOM, which this extension's other
  // adapters have shown is lossy by roughly two orders of magnitude once a
  // conversation scrolls past the virtualized viewport. grok.com exposes a
  // clean, cookie-authenticated REST surface
  // (/rest/app-chat/conversations/<id>, .../responses, .../response-node)
  // that this bridge fetches with the page's own credentials, exactly the
  // way chatgpt_bridge.js/claude_bridge.js do for their providers.
  const nativeCaptureMessage = "polylogue.grok.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.grok.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.grok.nativeFetchResponse";
  const assetFetchRequestMessage = "polylogue.grok.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.grok.assetFetchResponse";
  const currentOrigin = window.location.origin;

  function remember(capture, ownerId) {
    window.polylogueAssetStream.pageMessage({ type: nativeCaptureMessage, capture }, ownerId);
  }

  if (window.__polylogueGrokFetchHookInstalled === 2) return;
  window.__polylogueGrokFetchHookInstalled = 2;

  const originalFetch = window.fetch;

  function conversationUrl(conversationId, suffix = "") {
    return new URL(
      `/rest/app-chat/conversations/${encodeURIComponent(String(conversationId))}${suffix}`,
      currentOrigin,
    );
  }

  async function fetchStaged(url, signal, bundleRef, name, ownerId) {
    signal.throwIfAborted();
    const prepared = await window.polylogueAssetStream.prepareResponse("grok", signal, url.href, { id: bundleRef, name }, "native-response", false, null, null, false, null, ownerId);
    let response; let bodyRef;
    try {
      response = await originalFetch.call(window, url.href, { credentials: "include", cache: "no-store", signal });
      bodyRef = await prepared.consume(response);
    } catch (error) { await prepared.fail(error); throw error; }
    const contentType = response.headers.get("content-type") || "";
    if (!bodyRef) await window.polylogueAssetStream.nativeBundle("outcome", { bundleRef, name,
      outcome: { ok: response.ok, status: response.status, retry_after: response.headers.get("retry-after") || null } }, signal, ownerId);
    return { url: url.href, ok: response.ok, status: response.status, contentType, bodyRef,
      retryAfter: response.headers.get("retry-after") || null, capturedAt: new Date().toISOString() };
  }
  async function fetchConversation(conversationId, signal, ownerId) {
    const begun = await window.polylogueAssetStream.nativeBundle("begin", { nativeId: conversationId, bundleId: crypto.randomUUID() }, signal, ownerId);
    const bundleRef = begun.bundle_ref;
    const reply = (name, suffix = "") => begun.replies[name]
      ? Promise.resolve({ url: conversationUrl(conversationId, suffix).href, ok: true, status: 200, bodyRef: begun.replies[name], capturedAt: new Date().toISOString() })
      : fetchStaged(conversationUrl(conversationId, suffix), signal, bundleRef, name, ownerId);
    const conversation = await reply("conversation");
    if (!conversation.ok || !conversation.bodyRef) return { ...conversation, error: "conversation_metadata_fetch_failed" };
    const responses = await reply("responses", "/responses");
    if (!responses.ok || !responses.bodyRef) return { ...responses, relatedRefs: { conversation: conversation.bodyRef }, error: "conversation_responses_fetch_failed" };
    let nodes = null;
    try { nodes = await reply("response_nodes", "/response-node"); }
    catch {
      signal.throwIfAborted();
      await window.polylogueAssetStream.nativeBundle("outcome", { bundleRef, name: "response_nodes", outcome: { ok: false, status: null, error: "provider_transport_failure" } }, signal, ownerId);
    }
    if (nodes?.status === 429) return { ...nodes, relatedRefs: { conversation: conversation.bodyRef, responses: responses.bodyRef } };
    const complete = await window.polylogueAssetStream.nativeBundle("finish", { bundleRef }, signal, ownerId);
    return { ...responses, relatedRefs: { conversation: conversation.bodyRef, ...(nodes?.bodyRef ? { response_nodes: nodes.bodyRef } : {}) }, acquisition: complete.acquisition };
  }

  const requestControllers = new Map();
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== "polylogue.grok.cancelRequest") return;
    const controller = requestControllers.get(`${data.ownerId}:${data.requestId}`);
    if (controller) controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    else {
      window.polylogueAssetStream.pageMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, error: "capture_cancelled" }, data.ownerId);
      window.polylogueAssetStream.pageMessage({ type: assetFetchResponseMessage, requestId: data.requestId, outcome: { status: "cancelled" } }, data.ownerId);
    }
  });
  window.addEventListener("pagehide", () => {
    for (const controller of requestControllers.values()) controller.abort();
  });

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== nativeFetchRequestMessage || !data.requestId || !data.conversationId) return;
    const controller = new AbortController();
    requestControllers.set(`${data.ownerId}:${data.requestId}`, controller);
    try {
      const capture = await fetchConversation(data.conversationId, controller.signal, data.ownerId);
      if (capture.ok && capture.bodyRef) remember(capture, data.ownerId);
      window.polylogueAssetStream.pageMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, capture }, data.ownerId);
    } catch (error) {
      window.polylogueAssetStream.pageMessage(
        {
          type: nativeFetchResponseMessage,
          requestId: data.requestId,
          error: String(error && error.message ? error.message : error),
        },
        data.ownerId,
      );
    } finally { requestControllers.delete(`${data.ownerId}:${data.requestId}`); }
  });

  // --- Attachment byte acquisition -----------------------------------
  //
  // File/image attachments never carry bytes in `/responses` -- only a
  // `fileUri`/`key` such as `users/<uid>/<assetId>/content`, served from the
  // separate `assets.grok.com` host. Verified live 2026-07-31: that host
  // requires the same grok.com session credentials (403 with
  // credentials:"omit", 200 with credentials:"include" from a grok.com
  // page), unlike ChatGPT's signed object-store URLs which must NOT receive
  // page credentials cross-origin. So, deliberately unlike
  // chatgpt_bridge.js's asset fetch, this always sends credentials.
  function assetOutcome(status, { httpStatus = null, detail = null, sizeBytes = null, asset = null } = {}) {
    const outcome = { status };
    if (httpStatus !== null) outcome.http_status = httpStatus;
    if (detail !== null) outcome.detail = detail;
    if (sizeBytes !== null) outcome.size_bytes = sizeBytes;
    if (asset !== null) outcome.asset = asset;
    return outcome;
  }

  async function fetchAssetBytes(request, signal) {
    let assetUrl;
    try {
      assetUrl = new URL(`/${String(request.key).replace(/^\/+/, "")}`, "https://assets.grok.com");
    } catch {
      return assetOutcome("invalid_request", { detail: "asset_key_invalid" });
    }
    // `origin`, not `protocol`: the asset key is provider-controlled, and the
    // URL parser resolves a backslash in a special-scheme path as a slash, so
    // a key such as `\attacker.example/x` produces an https URL on a foreign
    // host. Comparing the resolved origin pins every request to the asset host
    // whatever the key spelling.
    if (
      typeof request.key !== "string" ||
      !request.key ||
      assetUrl.origin !== "https://assets.grok.com"
    ) {
      return assetOutcome("invalid_request", { detail: "asset_key_invalid" });
    }
    signal.throwIfAborted();
    const response = await originalFetch.call(window, assetUrl.href, { credentials: "include", cache: "no-store", signal });
    if (response.status === 429) return { status: "rate_limited", http_status: 429, retry_after: response.headers.get("retry-after") || null, response_url: response.url || assetUrl.href };
    if ([401, 403, 404, 410].includes(response.status)) {
      return assetOutcome("signed_url_expired", { httpStatus: response.status, detail: `asset_http_${response.status}` });
    }
    if (!response.ok) {
      return assetOutcome("request_failed", { httpStatus: response.status, detail: `asset_http_${response.status}` });
    }
    const asset = await window.polylogueAssetStream.stream(response, request.requestId, signal, { ownerId: request.ownerId });
    return assetOutcome("acquired", { httpStatus: response.status,
      asset: { ...asset, mime_type: response.headers.get("content-type") || null, name: request.name || null } });
  }

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== assetFetchRequestMessage || !data.requestId || !data.request) return;
    const controller = new AbortController();
    requestControllers.set(`${data.ownerId}:${data.requestId}`, controller);
    try {
      const outcome = await fetchAssetBytes({ ...data.request, requestId: data.requestId, ownerId: data.ownerId }, controller.signal);
      window.polylogueAssetStream.pageMessage({ type: assetFetchResponseMessage, requestId: data.requestId, outcome }, data.ownerId);
    } catch {
      window.polylogueAssetStream.pageMessage(
        {
          type: assetFetchResponseMessage,
          requestId: data.requestId,
          outcome: assetOutcome(controller.signal.aborted ? "cancelled" : "request_failed", { detail: controller.signal.aborted ? "capture_cancelled" : "request_failed" }),
        },
        data.ownerId,
      );
    } finally { requestControllers.delete(`${data.ownerId}:${data.requestId}`); }
  });
})();
