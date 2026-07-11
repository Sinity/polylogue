(function () {
  const nativeCaptureMessage = "polylogue.chatgpt.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.chatgpt.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.chatgpt.nativeFetchResponse";
  const currentOrigin = window.location.origin;
  const nativeFetchTimeoutMs = 8000;

  window.__polylogueCapturedFetches = Array.isArray(window.__polylogueCapturedFetches)
    ? window.__polylogueCapturedFetches
    : [];

  function post(capture) {
    window.postMessage({ type: nativeCaptureMessage, capture }, currentOrigin);
  }

  function remember(capture) {
    window.__polylogueCapturedFetches.push(capture);
    if (window.__polylogueCapturedFetches.length > 8) {
      window.__polylogueCapturedFetches.splice(0, window.__polylogueCapturedFetches.length - 8);
    }
    post(capture);
  }

  const existingCaptures = window.__polylogueCapturedFetches.slice(-8);
  window.__polylogueCapturedFetches = existingCaptures;
  for (const capture of existingCaptures) post(capture);

  if (window.__polylogueFetchHookInstalled) return;
  window.__polylogueFetchHookInstalled = true;

  const originalFetch = window.fetch;
  const accessTokenCacheTtlMs = 15000;
  let cachedAccessToken = null;
  let cachedAccessTokenUntil = 0;
  let accessTokenPromise = null;

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

  function bootstrapAccessToken() {
    try {
      const raw = document.getElementById("client-bootstrap")?.textContent;
      if (!raw) return null;
      return accessTokenFromPayload(JSON.parse(raw));
    } catch {
      return null;
    }
  }

  async function fetchSessionAccessToken() {
    const sessionUrl = new URL("/api/auth/session", currentOrigin);
    const response = await fetchWithAbort(
      sessionUrl.href,
      { credentials: "include", cache: "no-store" },
      "access_token_fetch"
    );
    if (!response.ok) return null;
    try {
      return accessTokenFromPayload(await response.json());
    } catch {
      return null;
    }
  }

  async function fetchCurrentAccessToken() {
    const sessionToken = await fetchSessionAccessToken().catch(() => null);
    return sessionToken || bootstrapAccessToken();
  }

  function resolveAccessToken() {
    if (Date.now() < cachedAccessTokenUntil) return Promise.resolve(cachedAccessToken);
    if (accessTokenPromise) return accessTokenPromise;
    accessTokenPromise = fetchCurrentAccessToken()
      .then((token) => {
        cachedAccessToken = token;
        cachedAccessTokenUntil = Date.now() + accessTokenCacheTtlMs;
        return token;
      })
      .finally(() => {
        accessTokenPromise = null;
      });
    return accessTokenPromise;
  }

  function bearerHeaders(accessToken) {
    return { Authorization: `Bearer ${accessToken}` };
  }

  function conversationUrl(conversationId) {
    return new URL(`/backend-api/conversation/${encodeURIComponent(String(conversationId))}`, currentOrigin);
  }

  function timeoutError(label) {
    const error = new Error(`${label}_timeout_after_${nativeFetchTimeoutMs}ms`);
    error.name = "PolylogueTimeoutError";
    return error;
  }

  async function fetchConversation(conversationId) {
    const url = conversationUrl(conversationId);
    const accessToken = await resolveAccessToken();
    const controller = new globalThis.AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(timeoutError("page_bridge_fetch")), nativeFetchTimeoutMs);
    let response;
    try {
      response = await originalFetch.call(window, url.href, {
        credentials: "include",
        cache: "no-store",
        headers: accessToken ? bearerHeaders(accessToken) : {},
        signal: controller.signal
      });
    } finally {
      window.clearTimeout(timeoutId);
    }
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.clone().text() : "";
    return {
      url: url.href,
      status: response.status,
      ok: response.ok,
      contentType,
      body,
      capturedAt: new Date().toISOString()
    };
  }

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = event.data || {};
    if (data.type !== nativeFetchRequestMessage || !data.requestId || !data.conversationId) return;
    try {
      const capture = await fetchConversation(data.conversationId);
      if (capture.ok && capture.body) remember(capture);
      window.postMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, capture }, currentOrigin);
    } catch (error) {
      window.postMessage(
        {
          type: nativeFetchResponseMessage,
          requestId: data.requestId,
          error: String(error && error.message ? error.message : error)
        },
        currentOrigin
      );
    }
  });

  // Asset acquisition: fetch assistant-produced files (Code Interpreter
  // sandbox deliverables, file-service uploads/outputs) through the page's
  // own authenticated session. Both endpoints return a JSON envelope with a
  // signed download_url; the bytes are then fetched from that URL directly.
  const assetFetchRequestMessage = "polylogue.chatgpt.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.chatgpt.assetFetchResponse";
  const assetFetchTimeoutMs = 8000;
  const assetAbsoluteMaxBytes = 25 * 1024 * 1024;

  function assetTimeoutError(label) {
    const error = new Error(`${label}_timeout_after_${assetFetchTimeoutMs}ms`);
    error.name = "PolylogueTimeoutError";
    return error;
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + chunkSize));
    }
    return window.btoa(binary);
  }

  function bytesToHex(buffer) {
    return [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  async function sha256Hex(buffer) {
    if (!globalThis.crypto?.subtle) throw new Error("asset_sha256_unavailable");
    return bytesToHex(await globalThis.crypto.subtle.digest("SHA-256", buffer));
  }

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
      rawText = (await response.clone().text()).slice(0, 16384);
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

  function boundedMaxBytes(request) {
    const requested = Number(request.maxBytes);
    if (!Number.isFinite(requested) || requested <= 0) return assetAbsoluteMaxBytes;
    return Math.min(requested, assetAbsoluteMaxBytes);
  }

  function declaredContentLength(response) {
    const raw = response.headers.get("content-length");
    if (!raw || !/^\d+$/.test(raw)) return null;
    const parsed = Number(raw);
    return Number.isSafeInteger(parsed) ? parsed : null;
  }

  async function fetchWithAbort(url, options, label) {
    const controller = new globalThis.AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(assetTimeoutError(label)), assetFetchTimeoutMs);
    try {
      return await originalFetch.call(window, url, { ...options, signal: controller.signal });
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  async function fetchAssetBytes(request) {
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
    const accessToken = await resolveAccessToken();
    if (!accessToken) {
      return assetOutcome("unauthorized", { phase: "access_token", detail: "access_token_unavailable" });
    }
    const metaResponse = await fetchWithAbort(
      metaUrl.href,
      { credentials: "include", cache: "no-store", headers: bearerHeaders(accessToken) },
      "asset_meta_fetch"
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
    const fileResponse = await fetchWithAbort(
      signedUrl.href,
      { credentials: "omit", cache: "no-store" },
      "asset_bytes_fetch"
    );
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
    const maxBytes = boundedMaxBytes(request);
    const contentLength = declaredContentLength(fileResponse);
    if (contentLength !== null && contentLength > maxBytes) {
      return assetOutcome("too_large", {
        phase: "signed_bytes",
        httpStatus: fileResponse.status,
        detail: "content_length_over_limit",
        sizeBytes: contentLength
      });
    }
    const buffer = await fileResponse.arrayBuffer();
    if (buffer.byteLength > maxBytes) {
      return assetOutcome("too_large", {
        phase: "signed_bytes",
        httpStatus: fileResponse.status,
        detail: "downloaded_bytes_over_limit",
        sizeBytes: buffer.byteLength
      });
    }
    let contentSha256;
    try {
      contentSha256 = await sha256Hex(buffer);
    } catch {
      return assetOutcome("integrity_error", { phase: "sha256", detail: "sha256_unavailable" });
    }
    return assetOutcome("acquired", {
      phase: "complete",
      httpStatus: fileResponse.status,
      asset: {
        base64: arrayBufferToBase64(buffer),
        size_bytes: buffer.byteLength,
        sha256: contentSha256,
        mime_type: fileResponse.headers.get("content-type") || null,
        name: (meta && (meta.file_name || meta.fileName)) || null
      }
    });
  }

  function assetExceptionOutcome(error) {
    const timedOut =
      error?.name === "AbortError" ||
      error?.name === "PolylogueTimeoutError" ||
      String(error?.message || "").includes("timeout_after_");
    return assetOutcome("request_failed", {
      phase: "bridge",
      detail: timedOut ? "request_timeout" : "request_failed"
    });
  }

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = event.data || {};
    if (data.type !== assetFetchRequestMessage || !data.requestId || !data.request) return;
    try {
      const outcome = await fetchAssetBytes(data.request);
      window.postMessage({ type: assetFetchResponseMessage, requestId: data.requestId, outcome }, currentOrigin);
    } catch (error) {
      window.postMessage(
        {
          type: assetFetchResponseMessage,
          requestId: data.requestId,
          outcome: assetExceptionOutcome(error)
        },
        currentOrigin
      );
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
      if (isConversation && contentType.includes("application/json")) {
        const body = await response.clone().text();
        remember({
          url: absolute.href,
          status: response.status,
          ok: response.ok,
          contentType,
          body,
          capturedAt: new Date().toISOString()
        });
      }
    } catch {
      // Capture must never perturb the ChatGPT page's own request path.
    }
    return response;
  };
})();
