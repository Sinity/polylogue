(function () {
  const nativeFetchRequestMessage = "polylogue.claude.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.claude.nativeFetchResponse";
  const currentOrigin = window.location.origin;

  if (window.__polylogueClaudeFetchHookInstalled === 2) return;
  window.__polylogueClaudeFetchHookInstalled = 2;

  const originalFetch = window.fetch;

  function resourceUrls() {
    return window.performance.getEntriesByType("resource").map((entry) => entry.name);
  }

  function organizationIdFromLocalStorage() {
    const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    let selector;
    try { selector = JSON.parse(window.localStorage.getItem("omelette-org-settings-cache") || "null"); }
    catch { return null; }
    return selector && uuidPattern.test(selector.orgUuid) ? selector.orgUuid : null;
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

  async function fetchConversation(conversationId, signal, beforeAwait, ownerId) {
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
    beforeAwait("admission");
    const prepared = await window.polylogueAssetStream.prepareResponse("claude-ai", signal, url, null, "native-response", false, null, null, false, null, ownerId);
    let response; let bodyRef;
    try {
      beforeAwait("provider_fetch");
      response = await originalFetch.call(window, url, { credentials: "include", cache: "no-store", signal });
      beforeAwait("staging");
      bodyRef = await prepared.consume(response);
    } catch (error) { await prepared.fail(error); throw error; }
    beforeAwait("unknown");
    const contentType = response.headers.get("content-type") || "";
    return {
      url,
      status: response.status,
      ok: response.ok,
      contentType,
      bodyRef,
      retryAfter: response.headers.get("retry-after") || null,
      capturedAt: new Date().toISOString()
    };
  }

  const requestControllers = new Map();
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== "polylogue.claude.cancelRequest") return;
    const controller = requestControllers.get(`${data.ownerId}:${data.requestId}`);
    if (controller) controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    else window.polylogueAssetStream.pageMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, error: "capture_cancelled" }, data.ownerId);
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
    let failureStage = "unknown";
    try {
      const capture = await fetchConversation(data.conversationId, controller.signal, stage => { failureStage = stage; }, data.ownerId);
      window.polylogueAssetStream.pageMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, capture }, data.ownerId);
    } catch (error) {
      window.polylogueAssetStream.pageMessage(
        {
          type: nativeFetchResponseMessage,
          requestId: data.requestId,
          error: String(error && error.message ? error.message : error),
          // Original acquisition boundary that began unwinding, not cleanup
          // progress or proof of the exception cause. No private value is added.
          failure_stage: failureStage
        },
        data.ownerId
      );
    } finally {
      requestControllers.delete(`${data.ownerId}:${data.requestId}`);
    }
  });

})();
