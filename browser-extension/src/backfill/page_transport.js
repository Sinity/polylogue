export async function executeProviderPageRequest(request) {
  const currentOrigin = window.location.origin;
  const requestTimeoutMs = 55000;
  const absoluteMaxResponseBytes = 32 * 1024 * 1024;
  const originalFetch = window.fetch;
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  const maxResponseBytes = Number.isInteger(request?.maxResponseBytes)
    && request.maxResponseBytes > 0
    && request.maxResponseBytes <= absoluteMaxResponseBytes
    ? request.maxResponseBytes
    : absoluteMaxResponseBytes;

  function boundedInteger(value, name, minimum, maximum) {
    if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`backfill_bridge_invalid_${name}`);
    return value;
  }

  function nativeId(value) {
    if (typeof value !== "string" || !/^[A-Za-z0-9_-]{1,256}$/.test(value)) throw new Error("backfill_bridge_invalid_native_id");
    return value;
  }

  async function readBoundedBody(response) {
    const declared = Number.parseInt(response.headers.get("content-length") || "", 10);
    if (Number.isFinite(declared) && declared > maxResponseBytes) throw new Error("backfill_bridge_response_too_large");
    if (response.body?.getReader) {
      const reader = response.body.getReader();
      const decoder = new globalThis.TextDecoder();
      let size = 0;
      let body = "";
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        size += chunk.value.byteLength;
        if (size > maxResponseBytes) {
          await reader.cancel("backfill_bridge_response_too_large");
          throw new Error("backfill_bridge_response_too_large");
        }
        body += decoder.decode(chunk.value, { stream: true });
      }
      return body + decoder.decode();
    }
    const body = await response.text();
    if (new globalThis.TextEncoder().encode(body).length > maxResponseBytes) throw new Error("backfill_bridge_response_too_large");
    return body;
  }

  async function fetchBounded(url, options) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort("backfill_bridge_request_timeout"), requestTimeoutMs);
    try {
      const response = await originalFetch.call(window, url, { ...options, signal: controller.signal });
      return {
        ok: response.ok,
        status: response.status,
        contentType: response.headers.get("content-type") || "",
        retryAfter: response.headers.get("retry-after") || null,
        body: await readBoundedBody(response),
      };
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function chatGptRequest() {
    let url;
    if (request.operation === "inventory") {
      const offset = boundedInteger(request.params?.offset, "offset", 0, 10_000_000);
      const limit = boundedInteger(request.params?.limit, "limit", 1, 100);
      if (typeof request.params?.archived !== "boolean" || typeof request.params?.starred !== "boolean") {
        throw new Error("backfill_bridge_invalid_inventory_flags");
      }
      url = new URL("/backend-api/conversations", currentOrigin);
      url.search = new URLSearchParams({
        offset: String(offset),
        limit: String(limit),
        order: "updated",
        is_archived: String(request.params.archived),
        is_starred: String(request.params.starred),
      });
    } else if (request.operation === "conversation") {
      url = new URL(`/backend-api/conversation/${encodeURIComponent(nativeId(request.params?.nativeId))}`, currentOrigin);
    } else {
      throw new Error("backfill_bridge_operation_not_allowed");
    }
    const session = await fetchBounded(new URL("/api/auth/session", currentOrigin).href, { credentials: "include", cache: "no-store" });
    if (!session.ok) throw new Error("backfill_bridge_auth_context_unavailable");
    let payload;
    try { payload = JSON.parse(session.body); } catch { throw new Error("backfill_bridge_auth_context_unavailable"); }
    const accessToken = payload?.accessToken || payload?.access_token || payload?.session?.accessToken || payload?.session?.access_token;
    const accountId = payload?.account?.id;
    if (typeof accessToken !== "string" || !accessToken || typeof accountId !== "string" || !accountId) {
      throw new Error("backfill_bridge_auth_context_unavailable");
    }
    return fetchBounded(url.href, {
      credentials: "include",
      cache: "no-store",
      headers: { Authorization: `Bearer ${accessToken}`, "ChatGPT-Account-Id": accountId },
    });
  }

  function selectedClaudeOrganizationId() {
    let selector;
    try { selector = JSON.parse(window.localStorage.getItem("omelette-org-settings-cache") || "null"); } catch { selector = null; }
    if (selector && uuidPattern.test(selector.orgUuid)) return selector.orgUuid;
    throw new Error("backfill_bridge_selected_organization_unavailable");
  }

  async function claudeRequest() {
    const selected = selectedClaudeOrganizationId();
    if (request.operation === "organizations") {
      const result = await fetchBounded(new URL("/api/organizations", currentOrigin).href, { credentials: "include", cache: "no-store" });
      if (!result.ok) return result;
      let organizations;
      try { organizations = JSON.parse(result.body); } catch { throw new Error("backfill_bridge_organizations_contract_drift"); }
      if (!Array.isArray(organizations)) throw new Error("backfill_bridge_organizations_contract_drift");
      const selectedIndex = organizations.findIndex((organization) => organization?.uuid === selected);
      if (selectedIndex < 0) throw new Error("backfill_bridge_selected_organization_stale");
      result.body = JSON.stringify([organizations[selectedIndex], ...organizations.filter((_entry, index) => index !== selectedIndex)]);
      return result;
    }
    if (request.params?.organizationId !== selected) throw new Error("backfill_bridge_selected_organization_stale");
    if (request.operation === "inventory") {
      const offset = boundedInteger(request.params?.offset, "offset", 0, 10_000_000);
      const limit = boundedInteger(request.params?.limit, "limit", 1, 100);
      const url = new URL(`/api/organizations/${encodeURIComponent(selected)}/chat_conversations`, currentOrigin);
      url.search = new URLSearchParams({ limit: String(limit), offset: String(offset) });
      return fetchBounded(url.href, { credentials: "include", cache: "no-store" });
    }
    if (request.operation === "conversation") {
      const url = new URL(`/api/organizations/${encodeURIComponent(selected)}/chat_conversations/${encodeURIComponent(nativeId(request.params?.nativeId))}`, currentOrigin);
      url.search = new URLSearchParams({ tree: "True", rendering_mode: "messages", render_all_tools: "true", consistency: "strong" });
      return fetchBounded(url.href, { credentials: "include", cache: "no-store" });
    }
    throw new Error("backfill_bridge_operation_not_allowed");
  }

  try {
    const hostname = window.location.hostname;
    const expectedProvider = hostname === "chatgpt.com"
      ? "chatgpt"
      : hostname === "claude.ai" ? "claude-ai" : null;
    if (!expectedProvider || request.provider !== expectedProvider) throw new Error("backfill_bridge_provider_mismatch");
    const response = expectedProvider === "chatgpt" ? await chatGptRequest() : await claudeRequest();
    return { ok: true, response };
  } catch (error) {
    return { ok: false, error: String(error?.message || error) };
  }
}
