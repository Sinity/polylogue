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

  function bootstrapAccessToken() {
    try {
      const raw = document.getElementById("client-bootstrap")?.textContent;
      if (!raw) return null;
      const bootstrap = JSON.parse(raw);
      const token = bootstrap?.session?.accessToken;
      return typeof token === "string" && token ? token : null;
    } catch {
      return null;
    }
  }

  function authHeaders() {
    const token = bootstrapAccessToken();
    return token ? { authorization: `Bearer ${token}` } : {};
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
    const controller = new globalThis.AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(timeoutError("page_bridge_fetch")), nativeFetchTimeoutMs);
    let response;
    try {
      response = await originalFetch.call(window, url.href, {
        credentials: "include",
        cache: "no-store",
        headers: authHeaders(),
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
  const assetFetchTimeoutMs = 30000;

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
      throw new Error(`unsupported_asset_kind_${request.kind}`);
    }
    const metaResponse = await fetchWithAbort(
      metaUrl.href,
      { credentials: "include", cache: "no-store", headers: authHeaders() },
      "asset_meta_fetch"
    );
    if (!metaResponse.ok) throw new Error(`asset_meta_status_${metaResponse.status}`);
    const meta = await metaResponse.json();
    const downloadUrl = meta && (meta.download_url || meta.downloadUrl || meta.url);
    if (typeof downloadUrl !== "string" || !downloadUrl) throw new Error("asset_meta_missing_download_url");
    const fileResponse = await fetchWithAbort(downloadUrl, { credentials: "omit", cache: "no-store" }, "asset_bytes_fetch");
    if (!fileResponse.ok) throw new Error(`asset_bytes_status_${fileResponse.status}`);
    const buffer = await fileResponse.arrayBuffer();
    const maxBytes = Number(request.maxBytes) > 0 ? Number(request.maxBytes) : 25 * 1024 * 1024;
    if (buffer.byteLength > maxBytes) throw new Error(`asset_too_large_${buffer.byteLength}`);
    return {
      base64: arrayBufferToBase64(buffer),
      size_bytes: buffer.byteLength,
      mime_type: fileResponse.headers.get("content-type") || null,
      name: (meta && (meta.file_name || meta.fileName)) || null
    };
  }

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = event.data || {};
    if (data.type !== assetFetchRequestMessage || !data.requestId || !data.request) return;
    try {
      const asset = await fetchAssetBytes(data.request);
      window.postMessage({ type: assetFetchResponseMessage, requestId: data.requestId, asset }, currentOrigin);
    } catch (error) {
      window.postMessage(
        {
          type: assetFetchResponseMessage,
          requestId: data.requestId,
          error: String(error && error.message ? error.message : error)
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
        /\/backend-api\/conversation\/[^/?#]+/.test(absolute.pathname) &&
        !absolute.pathname.endsWith("/init");
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
