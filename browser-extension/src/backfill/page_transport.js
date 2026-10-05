export async function executeProviderPageRequest(request) {
  const currentOrigin = window.location.origin;
  const ownerId = request.ownerId;
  if (typeof ownerId !== "string" || !/^[a-p]{32}$/.test(ownerId)) throw new Error("capture_transport_owner_unavailable");
  const currentFetch = window.fetch;
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

  function boundedInteger(value, name, minimum, maximum) {
    if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`backfill_bridge_invalid_${name}`);
    return value;
  }

  function nativeId(value) {
    if (typeof value !== "string" || !/^[A-Za-z0-9_-]{1,256}$/.test(value)) throw new Error("backfill_bridge_invalid_native_id");
    return value;
  }

  const controller = new AbortController();
  const close = () => controller.abort("provider_page_closed");
  const cancel = (event) => { if (event.detail?.requestId === request.requestId && event.detail?.ownerId === ownerId) controller.abort("backfill_cancelled"); };
  window.addEventListener("pagehide", close, { once: true });
  window.addEventListener("polylogue.providerCancel", cancel);
  async function fetchResponse(url, options, accountHandle = null) {
    controller.signal.throwIfAborted();
    const prepared = await window.polylogueAssetStream.prepareResponse(request.provider, controller.signal, url, request.capture_bundle || null,
      request.operation === "inventory" || request.operation === "organizations" ? "provider-inventory" : "native-response", false, request.queue_context || null, accountHandle, false, null, ownerId);
    let response; let bodyRef;
    try {
      response = await currentFetch.call(window, url, { ...options, signal: controller.signal });
      bodyRef = await prepared.consume(response);
    } catch (error) { await prepared.fail(error); throw error; }
    return { ok: response.ok, status: response.status, contentType: response.headers.get("content-type") || "",
      retryAfter: response.headers.get("retry-after") || null, bodyRef };
  }
  async function pageJson(url) {
    controller.signal.throwIfAborted();
    const response = await currentFetch.call(window, url, { credentials: "include", cache: "no-store", signal: controller.signal });
    if (response.status === 429) {
      const error = new Error("provider_rate_limited"); error.outcome = "rate_limited";
      error.retryAfter = response.headers.get("retry-after"); error.responseUrl = response.url || url;
      await response.body?.cancel().catch(() => undefined);
      throw error;
    }
    if (!response.ok) throw new Error("backfill_bridge_auth_context_unavailable");
    return await response.json();
  }
  async function chatGptRequest() {
    let url;
    if (request.operation === "inventory") {
      const offset = boundedInteger(request.params?.offset, "offset", 0, Number.MAX_SAFE_INTEGER);
      const limit = boundedInteger(request.params?.limit, "limit", 1, 100);
      if (typeof request.params?.archived !== "boolean" || typeof request.params?.starred !== "boolean") throw new Error("backfill_bridge_invalid_inventory_flags");
      url = new URL("/backend-api/conversations", currentOrigin);
      url.search = new URLSearchParams({ offset: String(offset), limit: String(limit), is_archived: String(request.params.archived), is_starred: String(request.params.starred) });
    } else if (request.operation === "conversation") {
      url = new URL(`/backend-api/conversation/${encodeURIComponent(nativeId(request.params?.nativeId))}`, currentOrigin);
    } else if (request.operation !== "identity") throw new Error("backfill_bridge_operation_not_allowed");
    const payload = await pageJson(new URL("/api/auth/session", currentOrigin).href);
    const token = payload?.accessToken || payload?.access_token || payload?.session?.accessToken || payload?.session?.access_token;
    const accountId = payload?.account?.id;
    if (typeof token !== "string" || !token || typeof accountId !== "string" || !accountId) throw new Error("backfill_bridge_auth_context_unavailable");
    if (request.operation === "identity") return { accountHandle: accountId };
    return fetchResponse(url.href, { credentials: "include", cache: "no-store", headers: { Authorization: `Bearer ${token}`, "ChatGPT-Account-Id": accountId } }, accountId);
  }

  function selectedClaudeOrganizationId() {
    let selector;
    try { selector = JSON.parse(window.localStorage.getItem("omelette-org-settings-cache") || "null"); } catch { selector = null; }
    if (selector && uuidPattern.test(selector.orgUuid)) return selector.orgUuid;
    throw new Error("backfill_bridge_selected_organization_unavailable");
  }

  async function claudeRequest() {
    const selected = selectedClaudeOrganizationId();
    if (request.operation === "identity") {
      const organizations = await pageJson(new URL("/api/organizations", currentOrigin).href);
      if (!Array.isArray(organizations)) throw new Error("backfill_bridge_organizations_contract_drift");
      if (!organizations.some((organization) => organization?.uuid === selected)) throw new Error("backfill_bridge_selected_organization_stale");
      return { accountHandle: selected };
    }
    if (request.operation === "organizations") {
      const response = await fetchResponse(new URL("/api/organizations", currentOrigin).href, { credentials: "include", cache: "no-store" });
      return { ...response, selectedOrganizationId: selected };
    }

    if (request.params?.organizationId !== selected) throw new Error("backfill_bridge_selected_organization_stale");
    if (request.operation === "inventory") {
      const offset = boundedInteger(request.params?.offset, "offset", 0, Number.MAX_SAFE_INTEGER);
      const limit = boundedInteger(request.params?.limit, "limit", 1, 100);
      const url = new URL(`/api/organizations/${encodeURIComponent(selected)}/chat_conversations`, currentOrigin);
      url.search = new URLSearchParams({ limit: String(limit), offset: String(offset) });
      return fetchResponse(url.href, { credentials: "include", cache: "no-store" });
    }
    if (request.operation === "conversation") {
      const url = new URL(`/api/organizations/${encodeURIComponent(selected)}/chat_conversations/${encodeURIComponent(nativeId(request.params?.nativeId))}`, currentOrigin);
      url.search = new URLSearchParams({ tree: "True", rendering_mode: "messages", render_all_tools: "true", consistency: "strong" });
      return fetchResponse(url.href, { credentials: "include", cache: "no-store" }, selected);
    }
    throw new Error("backfill_bridge_operation_not_allowed");
  }

  async function grokRequest() {
    if (request.operation === "identity") throw new Error("backfill_bridge_grok_identity_unavailable");
    const url = request.operation === "inventory"
      ? new URL("/rest/app-chat/conversations", currentOrigin)
      : new URL(`/rest/app-chat/conversations/${encodeURIComponent(nativeId(request.params?.nativeId))}${request.operation === "conversation" ? "" : `/${request.operation}`}`, currentOrigin);
    if (request.operation === "inventory") {
      url.searchParams.set("pageSize", String(boundedInteger(request.params?.pageSize, "page_size", 1, 200)));
      if (request.params?.pageToken) url.searchParams.set("pageToken", request.params.pageToken);
    } else if (!["conversation", "responses", "response-node"].includes(request.operation)) throw new Error("backfill_bridge_operation_not_allowed");
    return fetchResponse(url.href, { credentials: "include", cache: "no-store" });
  }

  try {
    const hostname = window.location.hostname;
    const expectedProvider = hostname === "chatgpt.com"
      ? "chatgpt"
      : hostname === "claude.ai" ? "claude-ai" : hostname === "grok.com" ? "grok" : null;
    if (!expectedProvider || request.provider !== expectedProvider) throw new Error("backfill_bridge_provider_mismatch");
    const response = expectedProvider === "chatgpt" ? await chatGptRequest() : expectedProvider === "grok" ? await grokRequest() : await claudeRequest();
    return { ok: true, response };
  } catch (error) {
    return { ok: false, error: String(error?.message || error),
      ...(error.outcome === "rate_limited" ? { outcome: "rate_limited", status: 429, retryAfter: error.retryAfter, responseUrl: error.responseUrl } : {}) };
  } finally {
    window.removeEventListener("pagehide", close);
    window.removeEventListener("polylogue.providerCancel", cancel);
  }
}
