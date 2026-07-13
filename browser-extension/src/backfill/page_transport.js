export async function executeProviderPageRequest(request) {
  const currentOrigin = window.location.origin;
  const requestTimeoutMs = 55000;
  const absoluteMaxResponseBytes = 32 * 1024 * 1024;
  // A ChatGPT conversation is fetched in MAIN world, then reduced before the
  // scripting-result bridge. Both stages stay bounded independently: raw
  // provider bloat may use 64 MiB in page memory, but no more than 8 MiB can
  // cross into extension storage/worker code.
  const compactChatGptSourceMaxBytes = 64 * 1024 * 1024;
  const compactChatGptBridgeMaxBytes = 8 * 1024 * 1024;
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

  function sizeError(code, observedBytes, limitBytes) {
    return new Error(`${code}:observed_bytes=${observedBytes};limit_bytes=${limitBytes}`);
  }

  async function readBoundedBody(response, maxBytes = maxResponseBytes, tooLargeCode = "backfill_bridge_response_too_large") {
    const declared = Number.parseInt(response.headers.get("content-length") || "", 10);
    if (Number.isFinite(declared) && declared > maxBytes) throw sizeError(tooLargeCode, declared, maxBytes);
    if (response.body?.getReader) {
      const reader = response.body.getReader();
      const decoder = new globalThis.TextDecoder();
      let size = 0;
      let body = "";
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        size += chunk.value.byteLength;
        if (size > maxBytes) {
          await reader.cancel(tooLargeCode);
          throw sizeError(tooLargeCode, size, maxBytes);
        }
        body += decoder.decode(chunk.value, { stream: true });
      }
      return body + decoder.decode();
    }
    const body = await response.text();
    if (new globalThis.TextEncoder().encode(body).length > maxBytes) throw sizeError(tooLargeCode, new globalThis.TextEncoder().encode(body).length, maxBytes);
    return body;
  }

  async function fetchBounded(url, options, maxBytes = maxResponseBytes, tooLargeCode = "backfill_bridge_response_too_large") {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort("backfill_bridge_request_timeout"), requestTimeoutMs);
    try {
      const response = await originalFetch.call(window, url, { ...options, signal: controller.signal });
      return {
        ok: response.ok,
        status: response.status,
        contentType: response.headers.get("content-type") || "",
        retryAfter: response.headers.get("retry-after") || null,
        body: await readBoundedBody(response, maxBytes, tooLargeCode),
      };
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function compactChatGptContent(content) {
    if (!content || typeof content !== "object") return {};
    if (Array.isArray(content.parts)) {
      return {
        parts: content.parts.flatMap((part) => {
          if (typeof part === "string") return [part];
          if (part && typeof part === "object" && typeof part.text === "string") return [{ text: part.text }];
          return [];
        }),
      };
    }
    if (typeof content.text === "string") return { text: content.text };
    if (typeof content.result === "string") return { result: content.result };
    return {};
  }

  function compactChatGptConversation(body) {
    let source;
    try { source = JSON.parse(body); } catch { throw new Error("provider_contract_drift:chatgpt_conversation_not_json_object"); }
    if (!source || typeof source !== "object" || !source.mapping || typeof source.mapping !== "object") {
      throw new Error("provider_contract_drift:chatgpt_conversation.mapping_must_be_object");
    }
    const mapping = {};
    for (const [nodeId, node] of Object.entries(source.mapping)) {
      const message = node?.message;
      mapping[nodeId] = {
        id: typeof node?.id === "string" ? node.id : null,
        parent: typeof node?.parent === "string" ? node.parent : null,
        message: message && typeof message === "object"
          ? {
              id: typeof message.id === "string" ? message.id : null,
              author: { role: typeof message.author?.role === "string" ? message.author.role : null },
              content: compactChatGptContent(message.content),
              create_time: typeof message.create_time === "string" || typeof message.create_time === "number" ? message.create_time : null,
              status: typeof message.status === "string" ? message.status : null,
              metadata: { model_slug: typeof message.metadata?.model_slug === "string" ? message.metadata.model_slug : null },
            }
          : null,
      };
    }
    const projected = JSON.stringify({
      polylogue_bridge_projection: "chatgpt-native-compact-v1",
      id: typeof source.id === "string" ? source.id : null,
      title: typeof source.title === "string" ? source.title : null,
      create_time: typeof source.create_time === "string" || typeof source.create_time === "number" ? source.create_time : null,
      update_time: typeof source.update_time === "string" || typeof source.update_time === "number" ? source.update_time : null,
      mapping,
    });
    const bytes = new globalThis.TextEncoder().encode(projected).length;
    if (bytes > compactChatGptBridgeMaxBytes) {
      throw sizeError("backfill_bridge_projection_too_large", bytes, compactChatGptBridgeMaxBytes);
    }
    return projected;
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
    const response = await fetchBounded(url.href, {
      credentials: "include",
      cache: "no-store",
      headers: { Authorization: `Bearer ${accessToken}`, "ChatGPT-Account-Id": accountId },
    }, request.operation === "conversation" ? compactChatGptSourceMaxBytes : maxResponseBytes,
    request.operation === "conversation" ? "backfill_bridge_source_response_too_large" : "backfill_bridge_response_too_large");
    if (request.operation === "conversation" && response.ok) response.body = compactChatGptConversation(response.body);
    return response;
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
