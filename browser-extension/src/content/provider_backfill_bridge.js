(function () {
  const requestType = "polylogue.backfill.pageRequest";
  const responseType = "polylogue.backfill.pageResponse";
  const currentOrigin = window.location.origin;
  const requestTimeoutMs = 55000;
  const maxResponseBytes = 32 * 1024 * 1024;
  const originalFetch = window.fetch;
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  let chatGptContext = null;
  let chatGptContextUntil = 0;

  if (window.__polylogueBackfillPageBridgeInstalled) return;
  window.__polylogueBackfillPageBridgeInstalled = true;

  function boundedInteger(value, name, minimum, maximum) {
    if (!Number.isInteger(value) || value < minimum || value > maximum) {
      throw new Error(`backfill_bridge_invalid_${name}`);
    }
    return value;
  }

  function nativeId(value) {
    if (typeof value !== "string" || !/^[A-Za-z0-9_-]{1,256}$/.test(value)) {
      throw new Error("backfill_bridge_invalid_native_id");
    }
    return value;
  }

  function timeoutError() {
    const error = new Error("backfill_bridge_request_timeout");
    error.name = "PolylogueTimeoutError";
    return error;
  }

  async function fetchBounded(url, options) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(timeoutError()), requestTimeoutMs);
    try {
      const response = await originalFetch.call(window, url, { ...options, signal: controller.signal });
      const declared = Number.parseInt(response.headers.get("content-length") || "", 10);
      if (Number.isFinite(declared) && declared > maxResponseBytes) throw new Error("backfill_bridge_response_too_large");
      const body = await response.text();
      if (new TextEncoder().encode(body).length > maxResponseBytes) throw new Error("backfill_bridge_response_too_large");
      return {
        ok: response.ok,
        status: response.status,
        contentType: response.headers.get("content-type") || "",
        retryAfter: response.headers.get("retry-after") || null,
        body,
      };
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function chatGptAuthContext() {
    if (chatGptContext && Date.now() < chatGptContextUntil) return chatGptContext;
    const session = await fetchBounded(new URL("/api/auth/session", currentOrigin).href, {
      credentials: "include",
      cache: "no-store",
    });
    if (!session.ok) throw new Error("backfill_bridge_auth_context_unavailable");
    let payload;
    try { payload = JSON.parse(session.body); } catch { throw new Error("backfill_bridge_auth_context_unavailable"); }
    const accessToken = payload?.accessToken || payload?.access_token || payload?.session?.accessToken || payload?.session?.access_token;
    const accountId = payload?.account?.id;
    if (typeof accessToken !== "string" || !accessToken || typeof accountId !== "string" || !accountId) {
      throw new Error("backfill_bridge_auth_context_unavailable");
    }
    chatGptContext = { accessToken, accountId };
    chatGptContextUntil = Date.now() + 15000;
    return chatGptContext;
  }

  async function chatGptRequest(operation, params) {
    let url;
    if (operation === "inventory") {
      const offset = boundedInteger(params?.offset, "offset", 0, 10_000_000);
      const limit = boundedInteger(params?.limit, "limit", 1, 100);
      url = new URL("/backend-api/conversations", currentOrigin);
      url.search = new URLSearchParams({
        offset: String(offset),
        limit: String(limit),
        order: "updated",
        is_archived: "false",
        is_starred: "false",
      });
    } else if (operation === "conversation") {
      url = new URL(`/backend-api/conversation/${encodeURIComponent(nativeId(params?.nativeId))}`, currentOrigin);
    } else {
      throw new Error("backfill_bridge_operation_not_allowed");
    }
    const context = await chatGptAuthContext();
    return fetchBounded(url.href, {
      credentials: "include",
      cache: "no-store",
      headers: {
        Authorization: `Bearer ${context.accessToken}`,
        "ChatGPT-Account-Id": context.accountId,
      },
    });
  }

  function selectedClaudeOrganizationId() {
    let selector;
    try { selector = JSON.parse(window.localStorage.getItem("omelette-org-settings-cache") || "null"); } catch { selector = null; }
    if (selector && uuidPattern.test(selector.orgUuid)) return selector.orgUuid;
    throw new Error("backfill_bridge_selected_organization_unavailable");
  }

  async function claudeRequest(operation, params) {
    const selected = selectedClaudeOrganizationId();
    if (operation === "organizations") {
      const result = await fetchBounded(new URL("/api/organizations", currentOrigin).href, {
        credentials: "include",
        cache: "no-store",
      });
      if (!result.ok) return result;
      let organizations;
      try { organizations = JSON.parse(result.body); } catch { throw new Error("backfill_bridge_organizations_contract_drift"); }
      if (!Array.isArray(organizations)) throw new Error("backfill_bridge_organizations_contract_drift");
      const selectedIndex = organizations.findIndex((organization) => organization?.uuid === selected);
      if (selectedIndex < 0) throw new Error("backfill_bridge_selected_organization_stale");
      result.body = JSON.stringify([organizations[selectedIndex], ...organizations.filter((_entry, index) => index !== selectedIndex)]);
      return result;
    }
    if (params?.organizationId !== selected) throw new Error("backfill_bridge_selected_organization_stale");
    if (operation === "inventory") {
      const offset = boundedInteger(params?.offset, "offset", 0, 10_000_000);
      const limit = boundedInteger(params?.limit, "limit", 1, 100);
      const url = new URL(`/api/organizations/${encodeURIComponent(selected)}/chat_conversations`, currentOrigin);
      url.search = new URLSearchParams({ limit: String(limit), offset: String(offset) });
      return fetchBounded(url.href, { credentials: "include", cache: "no-store" });
    }
    if (operation === "conversation") {
      const url = new URL(`/api/organizations/${encodeURIComponent(selected)}/chat_conversations/${encodeURIComponent(nativeId(params?.nativeId))}`, currentOrigin);
      url.search = new URLSearchParams({
        tree: "True",
        rendering_mode: "messages",
        render_all_tools: "true",
        consistency: "strong",
      });
      return fetchBounded(url.href, { credentials: "include", cache: "no-store" });
    }
    throw new Error("backfill_bridge_operation_not_allowed");
  }

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = event.data || {};
    if (data.type !== requestType || typeof data.requestId !== "string" || !data.requestId) return;
    try {
      const hostname = window.location.hostname;
      const expectedProvider = hostname === "chatgpt.com" || hostname.endsWith(".chatgpt.com")
        ? "chatgpt"
        : hostname === "claude.ai" || hostname.endsWith(".claude.ai") ? "claude-ai" : null;
      if (!expectedProvider || data.provider !== expectedProvider) throw new Error("backfill_bridge_provider_mismatch");
      const response = expectedProvider === "chatgpt"
        ? await chatGptRequest(data.operation, data.params)
        : await claudeRequest(data.operation, data.params);
      window.postMessage({ type: responseType, requestId: data.requestId, response }, currentOrigin);
    } catch (error) {
      window.postMessage({
        type: responseType,
        requestId: data.requestId,
        error: String(error?.message || error),
      }, currentOrigin);
    }
  });
})();
