export async function executeProviderPageRequest(request) {
  const currentOrigin = window.location.origin;
  const requestTimeoutMs = 55000;
  const absoluteMaxResponseBytes = 32 * 1024 * 1024;
  // ChatGPT conversation responses may contain provider metadata bloat. Read
  // up to 64 MiB in MAIN world, then send one compact projection across the
  // scripting-result bridge. The compact projection is capped at 24 MiB,
  // leaving headroom beneath Chrome's established 32 MiB result limit.
  const compactChatGptSourceMaxBytes = 64 * 1024 * 1024;
  const compactChatGptBridgeMaxBytes = 24 * 1024 * 1024;
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

  // No content-type field allowlist: ChatGPT content nodes carry their
  // payload under different keys per content_type ("parts" for
  // multimodal_text/text, "thoughts" for reasoning nodes -- an array of
  // {summary, content} the previous allowlist dropped entirely, collapsing
  // every reasoning node to `{}` -- "text"/"result" for simple nodes, plus
  // "language"/"text" for code, "aggregate_result" for execution_output,
  // and content types not yet observed). Forwarding the object as-is keeps
  // every current and future content_type's payload without hand-modeling
  // each shape, and costs nothing extra: size is bounded by chunking
  // (buildChatGptChunks), not by dropping fields.
  function compactChatGptContent(content) {
    return content && typeof content === "object" ? content : {};
  }

  // Ditto for message-level metadata: the previous allowlist kept only
  // model_slug + a re-allowlisted attachments array, discarding
  // content_references (the citation graph), citations, finish_details,
  // request_id, reasoning_status, search_result_groups, and aggregate_result
  // (sandbox tool output). polylogue/sources/parsers/chatgpt.py already
  // reads all of these keys directly off message.metadata for GDPR-export
  // parsing -- passing metadata through unfiltered here means the same
  // parser now has full fidelity for browser-captured conversations too.
  function compactChatGptMetadata(metadata) {
    return metadata && typeof metadata === "object" ? metadata : {};
  }

  function chatGptConversationHeader(source) {
    return {
      id: typeof source.id === "string" ? source.id : null,
      title: typeof source.title === "string" ? source.title : null,
      create_time: typeof source.create_time === "string" || typeof source.create_time === "number" ? source.create_time : null,
      update_time: typeof source.update_time === "string" || typeof source.update_time === "number" ? source.update_time : null,
      current_node: typeof source.current_node === "string" ? source.current_node : null,
      // polylogue.sources.parsers.chatgpt.parse() reads these three
      // top-level fields to set session_kind (is_temporary) and
      // provider_project_ref (conversation_template_id/gizmo_id) -- omitting
      // them silently downgraded every temporary/project-scoped ChatGPT
      // conversation to a standard/unscoped session the moment this
      // projection started advertising itself as native_full instead of the
      // old degraded/compact signal.
      is_temporary: typeof source.is_temporary === "boolean" ? source.is_temporary : null,
      conversation_template_id: typeof source.conversation_template_id === "string" ? source.conversation_template_id : null,
      gizmo_id: typeof source.gizmo_id === "string" ? source.gizmo_id : null,
    };
  }

  function projectChatGptNode(node) {
    if (!node || typeof node !== "object") return null;
    const message = node.message;
    return {
      id: typeof node.id === "string" ? node.id : null,
      parent: typeof node.parent === "string" ? node.parent : null,
      // polylogue.sources.parsers.chatgpt.extract_messages_from_mapping
      // derives branch_index EXCLUSIVELY from the parent node's `children`
      // array position (a sibling's index within its parent's children,
      // not anything computed from the child itself) -- dropping this
      // field silently loses branch/variant ordering forever once this
      // projection is the durable raw capture, the exact defect class this
      // fix exists to eliminate, just one level up (node topology instead
      // of message content). Preserved as an array of node-id strings,
      // same shape the provider sends.
      children: Array.isArray(node.children) ? node.children.filter((childId) => typeof childId === "string") : [],
      message: message && typeof message === "object"
        ? {
            ...message,
            content: compactChatGptContent(message.content),
            metadata: compactChatGptMetadata(message.metadata),
          }
        : null,
    };
  }

  function assertScriptingResultBound(response) {
    // executeScript returns this exact result shape. JSON encoding is a
    // conservative byte model for the browser bridge because escaped content
    // can be larger than the projected conversation string alone.
    const bytes = new globalThis.TextEncoder().encode(JSON.stringify({ ok: true, response })).length;
    if (bytes > compactChatGptBridgeMaxBytes) {
      throw sizeError("backfill_bridge_projection_too_large", bytes, compactChatGptBridgeMaxBytes);
    }
    return bytes;
  }

  function compactChatGptConversation(body) {
    let source;
    try { source = JSON.parse(body); } catch { throw new Error("provider_contract_drift:chatgpt_conversation_not_json_object"); }
    if (!source || typeof source !== "object" || !source.mapping || typeof source.mapping !== "object" || Array.isArray(source.mapping)) {
      throw new Error("provider_contract_drift:chatgpt_conversation.mapping_must_be_object");
    }
    const mapping = Object.fromEntries(Object.entries(source.mapping).map(([nodeId, node]) => [nodeId, projectChatGptNode(node)]));
    const projected = JSON.stringify({
      polylogue_bridge_projection: "chatgpt-native-compact-v1",
      ...chatGptConversationHeader(source),
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
    let requestedNativeId = null;
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
      requestedNativeId = nativeId(request.params?.nativeId);
      url = new URL(`/backend-api/conversation/${encodeURIComponent(requestedNativeId)}`, currentOrigin);
    } else if (request.operation !== "identity") {
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
    if (request.operation === "identity") return { accountHandle: accountId };
    const response = await fetchBounded(url.href, {
      credentials: "include",
      cache: "no-store",
      headers: { Authorization: `Bearer ${accessToken}`, "ChatGPT-Account-Id": accountId },
    }, request.operation === "conversation" ? compactChatGptSourceMaxBytes : maxResponseBytes,
    request.operation === "conversation" ? "backfill_bridge_source_response_too_large" : "backfill_bridge_response_too_large");
    if (request.operation === "conversation" && response.ok) {
      response.body = compactChatGptConversation(response.body);
      assertScriptingResultBound(response);
    }
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
    if (request.operation === "identity" || request.operation === "organizations") {
      const result = await fetchBounded(new URL("/api/organizations", currentOrigin).href, { credentials: "include", cache: "no-store" });
      if (request.operation === "identity" && !result.ok) {
        throw new Error("backfill_bridge_auth_context_unavailable");
      }
      if (!result.ok) return result;
      let organizations;
      try { organizations = JSON.parse(result.body); } catch { throw new Error("backfill_bridge_organizations_contract_drift"); }
      if (!Array.isArray(organizations)) throw new Error("backfill_bridge_organizations_contract_drift");
      const selectedIndex = organizations.findIndex((organization) => organization?.uuid === selected);
      if (selectedIndex < 0) throw new Error("backfill_bridge_selected_organization_stale");
      if (request.operation === "identity") return { accountHandle: selected };
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
