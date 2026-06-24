(function () {
  const nativeCaptureMessage = "polylogue.claude.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.claude.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.claude.nativeFetchResponse";
  const currentOrigin = window.location.origin;

  window.__polylogueClaudeCapturedFetches = Array.isArray(window.__polylogueClaudeCapturedFetches)
    ? window.__polylogueClaudeCapturedFetches
    : [];

  function post(capture) {
    window.postMessage({ type: nativeCaptureMessage, capture }, currentOrigin);
  }

  function remember(capture) {
    window.__polylogueClaudeCapturedFetches.push(capture);
    if (window.__polylogueClaudeCapturedFetches.length > 8) {
      window.__polylogueClaudeCapturedFetches.splice(0, window.__polylogueClaudeCapturedFetches.length - 8);
    }
    post(capture);
  }

  const existingCaptures = window.__polylogueClaudeCapturedFetches.slice(-8);
  window.__polylogueClaudeCapturedFetches = existingCaptures;
  for (const capture of existingCaptures) post(capture);

  if (window.__polylogueClaudeFetchHookInstalled) return;
  window.__polylogueClaudeFetchHookInstalled = true;

  const originalFetch = window.fetch;

  function resourceUrls() {
    return window.performance.getEntriesByType("resource").map((entry) => entry.name);
  }

  function organizationIdFromLocalStorage() {
    const uuidPattern = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
    const patterns = [
      new RegExp(`^claude-mcp-has-connectors:(${uuidPattern})$`, "i"),
      new RegExp(`^LSS-model-selector-thinking:(${uuidPattern}):`, "i")
    ];
    for (const key of Object.keys(window.localStorage)) {
      for (const pattern of patterns) {
        const match = key.match(pattern);
        if (match) return match[1];
      }
    }
    return null;
  }

  function conversationApiUrlFromResources(conversationId, urls = resourceUrls()) {
    const escapedId = String(conversationId);
    const observed = urls.find((url) => {
      try {
        const parsed = new URL(url, currentOrigin);
        return (
          parsed.origin === currentOrigin &&
          parsed.pathname.includes(`/chat_conversations/${escapedId}`) &&
          /\/api\/organizations\/[^/]+\/chat_conversations\/[^/]+/.test(parsed.pathname)
        );
      } catch {
        return false;
      }
    });
    if (observed) return observed;

    for (const url of urls) {
      try {
        const parsed = new URL(url, currentOrigin);
        const match = parsed.pathname.match(/\/api\/bootstrap\/([^/]+)\/current_user_access/);
        if (parsed.origin === currentOrigin && match) {
          return new URL(
            `/api/organizations/${encodeURIComponent(match[1])}/chat_conversations/${encodeURIComponent(escapedId)}?tree=True&rendering_mode=messages&render_all_tools=true&consistency=strong`,
            currentOrigin
          ).href;
        }
      } catch {
        // Ignore malformed resource entries.
      }
    }
    const localStorageOrgId = organizationIdFromLocalStorage();
    if (localStorageOrgId) {
      return new URL(
        `/api/organizations/${encodeURIComponent(localStorageOrgId)}/chat_conversations/${encodeURIComponent(escapedId)}?tree=True&rendering_mode=messages&render_all_tools=true&consistency=strong`,
        currentOrigin
      ).href;
    }
    return null;
  }

  async function fetchConversation(conversationId) {
    const url = conversationApiUrlFromResources(conversationId);
    if (!url) {
      return {
        url: "",
        status: 0,
        ok: false,
        contentType: "",
        body: "",
        capturedAt: new Date().toISOString(),
        error: "conversation_api_url_not_found"
      };
    }
    const response = await originalFetch.call(window, url, {
      credentials: "include",
      cache: "no-store"
    });
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.clone().text() : "";
    return {
      url,
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

  window.fetch = async function polylogueClaudeFetch(input) {
    const response = await originalFetch.apply(this, arguments);
    try {
      const url = typeof input === "string" ? input : input && input.url;
      const absolute = new URL(url, window.location.href);
      const isConversation =
        absolute.origin === currentOrigin &&
        /\/api\/organizations\/[^/?#]+\/chat_conversations\/[^/?#]+/.test(absolute.pathname);
      const contentType = response.headers.get("content-type") || "";
      if (isConversation && contentType.includes("application/json")) {
        const body = await response.clone().text();
        if (body.includes('"chat_messages"')) {
          remember({
            url: absolute.href,
            status: response.status,
            ok: response.ok,
            contentType,
            body,
            capturedAt: new Date().toISOString()
          });
        }
      }
    } catch {
      // Capture must never perturb the Claude.ai page's own request path.
    }
    return response;
  };
})();
