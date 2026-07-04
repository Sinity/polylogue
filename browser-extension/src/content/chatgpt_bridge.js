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
