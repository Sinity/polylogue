(function () {
  const nativeFetchRequestMessage = "polylogue.claude.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.claude.nativeFetchResponse";
  const currentOrigin = window.location.origin;
  const nativeFetchTimeoutMs = 8000;

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

  function timeoutError(label) {
    const error = new Error(`${label}_timeout_after_${nativeFetchTimeoutMs}ms`);
    error.name = "PolylogueTimeoutError";
    return error;
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
    const controller = new globalThis.AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(timeoutError("page_bridge_fetch")), nativeFetchTimeoutMs);
    let response;
    try {
      response = await originalFetch.call(window, url, {
        credentials: "include",
        cache: "no-store",
        signal: controller.signal
      });
    } finally {
      window.clearTimeout(timeoutId);
    }
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

})();
