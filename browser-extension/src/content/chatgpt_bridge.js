(function () {
  const nativeCaptureMessage = "polylogue.chatgpt.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.chatgpt.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.chatgpt.nativeFetchResponse";
  const currentOrigin = window.location.origin;

  function remember(capture, ownerId) {
    window.polylogueAssetStream.pageMessage({ type: nativeCaptureMessage, capture }, ownerId);
  }

  if (window.__polylogueFetchHookInstalled === 2) return;
  window.__polylogueFetchHookInstalled = 2;

  const originalFetch = window.fetch;
  const accessTokenCacheTtlMs = 15000;
  let cachedAccessToken = null;
  let cachedAccountId = null;
  let cachedAccessTokenUntil = 0;

  function accessTokenFromPayload(payload) {
    const candidates = [
      payload?.accessToken,
      payload?.access_token,
      payload?.session?.accessToken,
      payload?.session?.access_token
    ];
    for (const candidate of candidates) {
      if (typeof candidate === "string" && candidate) return candidate;
    }
    return null;
  }

  function accountIdFromPayload(payload) {
    const candidates = [payload?.account?.id, payload?.session?.account?.id, payload?.account_id];
    return candidates.find((candidate) => typeof candidate === "string" && candidate) || null;
  }

  function bootstrapAccessToken() {
    try {
      const raw = document.getElementById("client-bootstrap")?.textContent;
      if (!raw) return null;
      const payload = JSON.parse(raw);
      cachedAccountId = accountIdFromPayload(payload) || cachedAccountId;
      return accessTokenFromPayload(payload);
    } catch {
      return null;
    }
  }

  async function fetchSessionAccessToken(signal) {
    const sessionUrl = new URL("/api/auth/session", currentOrigin);
    const response = await fetchWithAbort(
      sessionUrl.href,
      { credentials: "include", cache: "no-store" },
      signal
    );
    if (response.status === 429) {
      const error = new Error("provider_rate_limited");
      error.outcome = "rate_limited"; error.providerResponse = { status: 429, url: response.url || sessionUrl.href };
      error.retryAfter = response.headers.get("retry-after");
      await response.body?.cancel().catch(() => undefined);
      throw error;
    }
    if (!response.ok) return null;
    try {
      const payload = await response.json();
      cachedAccountId = accountIdFromPayload(payload) || cachedAccountId;
      return accessTokenFromPayload(payload);
    } catch {
      return null;
    }
  }

  async function fetchCurrentAccessToken(signal) {
    const sessionToken = await fetchSessionAccessToken(signal).catch((error) => {
      signal.throwIfAborted();
      if (error.outcome === "rate_limited") throw error;
      return null;
    });
    return sessionToken || bootstrapAccessToken();
  }

  function resolveAccessToken(signal) {
    if (Date.now() < cachedAccessTokenUntil) return Promise.resolve(cachedAccessToken);
    return fetchCurrentAccessToken(signal)
      .then((token) => {
        cachedAccessToken = token;
        cachedAccessTokenUntil = Date.now() + accessTokenCacheTtlMs;
        return token;
      });
  }

  function bearerHeaders(accessToken) {
    const headers = { Authorization: `Bearer ${accessToken}` };
    if (cachedAccountId) headers["ChatGPT-Account-Id"] = cachedAccountId;
    return headers;
  }

  function conversationUrl(conversationId) {
    return new URL(`/backend-api/conversation/${encodeURIComponent(String(conversationId))}`, currentOrigin);
  }

  async function fetchConversation(conversationId, signal, invocationRef = null, progress = () => {}, ownerId) {
    const url = conversationUrl(conversationId);
    progress("staging", "BEGIN");
    const prepared = await window.polylogueAssetStream.prepareResponse("chatgpt", signal, url.href, null, "native-response", false, null, null, false, invocationRef, ownerId);
    progress("staging", "END");
    let response; let bodyRef;
    try {
      progress("provider_auth", "BEGIN");
      const accessToken = await resolveAccessToken(signal);
      progress("provider_auth", "END");
      progress("provider_response", "BEGIN");
      response = await originalFetch.call(window, url.href, {
        credentials: "include", cache: "no-store",
        headers: accessToken ? bearerHeaders(accessToken) : {}, signal,
      });
      progress("provider_response", "END");
      progress("body", "BEGIN");
      bodyRef = await prepared.consume(response);
      progress("body", "END");
    } catch (error) { await prepared.fail(error); throw error; }
    const contentType = response.headers.get("content-type") || "";
    return {
      url: url.href,
      status: response.status,
      ok: response.ok,
      contentType,
      retryAfter: response.headers.get("retry-after") || null,
      bodyRef,
      capturedAt: new Date().toISOString()
    };
  }

  const requestControllers = new Map();
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type === "polylogue.chatgpt.cancelRequest") {
      const controller = requestControllers.get(`${data.ownerId}:${data.requestId}`);
      if (controller) controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
      else {
        window.polylogueAssetStream.pageMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, error: "capture_cancelled" }, data.ownerId);
        window.polylogueAssetStream.pageMessage({ type: "polylogue.chatgpt.assetFetchResponse", requestId: data.requestId,
          outcome: { status: "cancelled", phase: "bridge", detail: "capture_cancelled" } }, data.ownerId);
      }
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
    const progress = (stage, state) => {
      if (!controller.signal.aborted) window.polylogueAssetStream.pageMessage({ type: nativeFetchResponseMessage, requestId: data.requestId,
        progress: { stage, state } }, data.ownerId);
    };
    try {
      const capture = await fetchConversation(data.conversationId, controller.signal, data.invocationRef || null, progress, data.ownerId);
      if (capture.ok && capture.bodyRef) remember({ ...capture, source: "polylogue_native_fetch" }, data.ownerId);
      window.polylogueAssetStream.pageMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, capture }, data.ownerId);
    } catch (error) {
      window.polylogueAssetStream.pageMessage(
        {
          type: nativeFetchResponseMessage,
          requestId: data.requestId,
          ...(error.outcome === "rate_limited" ? { capture: { ok: false, status: 429, url: error.providerResponse.url, retryAfter: error.retryAfter } } : {}),
          error: controller.signal.aborted ? "capture_cancelled" : String(error && error.message ? error.message : error)
        },
        data.ownerId
      );
    } finally {
      requestControllers.delete(`${data.ownerId}:${data.requestId}`);
    }
  });

  // Asset acquisition: fetch assistant-produced files (Code Interpreter
  // sandbox deliverables, file-service uploads/outputs) through the page's
  // own authenticated session. Both endpoints return a JSON envelope with a
  // signed download_url; the bytes are then fetched from that URL directly.
  const assetFetchRequestMessage = "polylogue.chatgpt.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.chatgpt.assetFetchResponse";
  function assetOutcome(status, { phase, httpStatus = null, detail = null, sizeBytes = null, asset = null } = {}) {
    const outcome = { status, phase };
    if (httpStatus !== null) outcome.http_status = httpStatus;
    if (detail !== null) outcome.detail = detail;
    if (sizeBytes !== null) outcome.size_bytes = sizeBytes;
    if (asset !== null) outcome.asset = asset;
    return outcome;
  }

  function metadataErrorSignal(meta, rawText) {
    const values = [
      meta?.error_code,
      meta?.code,
      meta?.detail,
      meta?.message,
      meta?.error?.code,
      meta?.error?.message,
      rawText
    ];
    return values
      .filter((value) => typeof value === "string")
      .join(" ")
      .toLowerCase();
  }

  async function readMetadataEnvelope(response) {
    let rawText = "";
    try {
      rawText = await response.clone().text();
    } catch {
      return { meta: null, rawText: "" };
    }
    try {
      return { meta: JSON.parse(rawText), rawText };
    } catch {
      return { meta: null, rawText };
    }
  }

  function metadataFailureOutcome(request, response, meta, rawText) {
    const signal = metadataErrorSignal(meta, rawText);
    if (response.status === 429) return { status: "rate_limited", phase: "metadata", http_status: 429, retry_after: response.headers.get("retry-after") || null, response_url: response.url || null };
    if (signal.includes("ace_pod_expired") || signal.includes("ace pod expired")) {
      return assetOutcome("pod_expired", {
        phase: "metadata",
        httpStatus: response.status,
        detail: "ace_pod_expired"
      });
    }
    if (signal.includes("interpreter file not found") || (request.kind === "sandbox" && response.status === 404)) {
      return assetOutcome("missing", {
        phase: "metadata",
        httpStatus: response.status,
        detail: "interpreter_file_not_found"
      });
    }
    if (response.status === 401 || response.status === 403) {
      return assetOutcome("unauthorized", {
        phase: "metadata",
        httpStatus: response.status,
        detail: `metadata_http_${response.status}`
      });
    }
    if (!response.ok) {
      return assetOutcome("request_failed", {
        phase: "metadata",
        httpStatus: response.status,
        detail: `metadata_http_${response.status}`
      });
    }
    return null;
  }

  async function fetchWithAbort(url, options, signal) {
    signal.throwIfAborted();
    return originalFetch.call(window, url, { ...options, signal });
  }

  async function fetchBytesFromResolvedUrl(signedUrl, request, fallbackName, signal) {
    const byteCredentials = signedUrl.origin === currentOrigin ? "include" : "omit";
    const fileResponse = await fetchWithAbort(
      signedUrl.href,
      { credentials: byteCredentials, cache: "no-store" },
      signal
    );
    if (fileResponse.status === 429) return { status: "rate_limited", phase: "signed_bytes", http_status: 429, retry_after: fileResponse.headers.get("retry-after") || null, response_url: fileResponse.url || signedUrl.href };
    if ([401, 403, 404, 410].includes(fileResponse.status)) {
      return assetOutcome("signed_url_expired", {
        phase: "signed_bytes",
        httpStatus: fileResponse.status,
        detail: `signed_url_http_${fileResponse.status}`
      });
    }
    if (!fileResponse.ok) {
      return assetOutcome("request_failed", {
        phase: "signed_bytes",
        httpStatus: fileResponse.status,
        detail: `signed_url_http_${fileResponse.status}`
      });
    }
    const asset = await window.polylogueAssetStream.stream(fileResponse, request.requestId, signal, { ownerId: request.ownerId });
    return assetOutcome("acquired", {
      phase: "complete", httpStatus: fileResponse.status,
      asset: { ...asset, mime_type: fileResponse.headers.get("content-type") || null, name: fallbackName },
    });
  }

  async function fetchAssetBytes(request, signal) {
    if (request.kind === "url") {
      // No metadata round trip: the caller already has a concrete
      // byte-bearing URL in hand (e.g. an `img.src`/`a.href` rendered on the
      // page) and skips straight to fetching it. No current caller in this
      // repo builds a "url"-kind request (the chatgpt-dom-v1 adapter that
      // originally needed it was removed -- native capture covers what used
      // to require a DOM scrape), but the receiver-side protocol and its
      // budget/credential-scoping guarantees stay intact for any future one.
      let directUrl;
      try {
        directUrl = new URL(String(request.url), currentOrigin);
      } catch {
        return assetOutcome("invalid_request", { phase: "request", detail: "url_invalid" });
      }
      if (directUrl.protocol !== "https:") {
        return assetOutcome("invalid_request", { phase: "request", detail: "url_not_https" });
      }
      return fetchBytesFromResolvedUrl(directUrl, request, request.name || null, signal);
    }

    let metaUrl;
    if (request.kind === "sandbox") {
      metaUrl = new URL(
        `/backend-api/conversation/${encodeURIComponent(String(request.conversationId))}/interpreter/download`,
        currentOrigin
      );
      metaUrl.searchParams.set("message_id", String(request.messageId));
      metaUrl.searchParams.set("sandbox_path", String(request.sandboxPath));
    } else if (request.kind === "file") {
      metaUrl = new URL(`/backend-api/files/${encodeURIComponent(String(request.fileId))}/download`, currentOrigin);
    } else {
      return assetOutcome("invalid_request", { phase: "request", detail: "unsupported_asset_kind" });
    }
    const accessToken = await resolveAccessToken(signal);
    if (!accessToken) {
      return assetOutcome("unauthorized", { phase: "access_token", detail: "access_token_unavailable" });
    }
    const metaResponse = await fetchWithAbort(
      metaUrl.href,
      { credentials: "include", cache: "no-store", headers: bearerHeaders(accessToken) },
      signal
    );
    const { meta, rawText } = await readMetadataEnvelope(metaResponse);
    const failure = metadataFailureOutcome(request, metaResponse, meta, rawText);
    if (failure) return failure;
    const downloadUrl = meta && (meta.download_url || meta.downloadUrl || meta.url);
    if (typeof downloadUrl !== "string" || !downloadUrl) {
      return assetOutcome("invalid_response", { phase: "metadata", detail: "download_url_missing" });
    }
    let signedUrl;
    try {
      signedUrl = new URL(downloadUrl, currentOrigin);
    } catch {
      return assetOutcome("invalid_response", { phase: "metadata", detail: "download_url_invalid" });
    }
    if (signedUrl.protocol !== "https:") {
      return assetOutcome("invalid_response", { phase: "metadata", detail: "download_url_not_https" });
    }
    // Current ChatGPT interpreter downloads may point at the authenticated
    // same-origin estuary endpoint rather than at a self-authenticating object
    // store URL. fetchBytesFromResolvedUrl keeps page cookies only for that
    // exact origin, never forwarding them (or the bearer) cross-origin.
    return fetchBytesFromResolvedUrl(signedUrl, request, (meta && (meta.file_name || meta.fileName)) || null, signal);
  }

  function assetExceptionOutcome(error) {
    if (error.outcome === "rate_limited") return { status: "rate_limited", phase: "access_token", http_status: 429,
      retry_after: error.retryAfter, response_url: error.providerResponse.url };
    return assetOutcome("request_failed", {
      phase: "bridge",
      detail: "request_failed"
    });
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
    } catch (error) {
      window.polylogueAssetStream.pageMessage(
        {
          type: assetFetchResponseMessage,
          requestId: data.requestId,
          outcome: controller.signal.aborted ? assetOutcome("cancelled", { phase: "bridge", detail: "capture_cancelled" }) : assetExceptionOutcome(error)
        },
        data.ownerId
      );
    } finally {
      requestControllers.delete(`${data.ownerId}:${data.requestId}`);
    }
  });

  window.fetch = async function polylogueFetch(input) {
    const response = await originalFetch.apply(this, arguments);
    try {
      const url = typeof input === "string" ? input : input && input.url;
      const absolute = new URL(url, window.location.href);
      const isConversation =
        absolute.origin === currentOrigin &&
        /^\/backend-api\/conversation\/[^/?#]+\/?$/.test(absolute.pathname);
      const contentType = response.headers.get("content-type") || "";
      if (isConversation && response.ok && contentType.includes("application/json")) {
        for (const ownerId of window.polylogueAssetStream.eligibleOwners()) {
          const controller = new AbortController();
          const cancel = () => controller.abort("provider_page_closed");
          window.addEventListener("pagehide", cancel, { once: true });
          void window.polylogueAssetStream.stageResponse(response.clone(), "chatgpt", controller.signal, absolute.href, null, "native-response", ownerId)
            .then((bodyRef) => remember({ url: absolute.href, status: response.status, ok: true, contentType, bodyRef, capturedAt: new Date().toISOString() }, ownerId))
            .catch(() => undefined).finally(() => window.removeEventListener("pagehide", cancel));
        }
      }
    } catch {
      // Capture must never perturb the ChatGPT page's own request path.
    }
    return response;
  };
})();
