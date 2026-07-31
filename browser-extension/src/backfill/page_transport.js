export async function executeProviderPageRequest(request) {
  const currentOrigin = window.location.origin;
  const requestTimeoutMs = 55000;
  const absoluteMaxResponseBytes = 32 * 1024 * 1024;
  // A ChatGPT conversation is fetched in MAIN world, then projected into the
  // canonical native mapping shape (every content-type payload key and every
  // metadata key preserved -- no field allowlist) before crossing the
  // scripting-result bridge. Raw provider bloat may use up to 64 MiB in page
  // memory. No single scripting-result call may exceed 24 MiB (8 MiB of
  // headroom beneath Chrome's established 32 MiB scripting-result limit), but
  // fidelity never trades off against size: a projection that would exceed
  // the per-call bridge budget is split into ordered, node-aligned chunks
  // (never splitting a single node's JSON) and the caller re-invokes this
  // function once per chunk, reassembling the full mapping on the extension
  // side. See buildChatGptChunks() below.
  const compactChatGptSourceMaxBytes = 64 * 1024 * 1024;
  const compactChatGptBridgeMaxBytes = 24 * 1024 * 1024;
  // Target size when packing nodes into a chunk, well under the hard bridge
  // cap so JSON-escaping overhead measured by assertScriptingResultBound
  // (quotes, control characters, the extension-side wrapper) has headroom.
  const chatGptChunkPackTargetBytes = compactChatGptBridgeMaxBytes / 2;
  // A chunked conversation fetch does exactly one network round-trip: the
  // raw parsed source is cached in this MAIN-world page global (persists
  // across the sequential executeScript calls that pull each chunk) so
  // chunk N>0 does not re-fetch. Keyed by native conversation id.
  const chunkCacheTtlMs = 5 * 60 * 1000;
  const maxChatGptChunks = 64;
  window.__polylogueChatGptChunkCache = window.__polylogueChatGptChunkCache || new Map();
  const chunkCache = window.__polylogueChatGptChunkCache;
  // The cache entry for a fully-served conversation is deleted right after
  // its last chunk goes out (see compactChatGptConversation). This sweep is
  // the other half: an operator tab can live for days, and a caller that
  // starts pulling chunks then never finishes (crash, cancel, extension
  // reload mid-reassembly) would otherwise leave tens of MiB of projected
  // chunks parked in this page global indefinitely -- nothing else ever
  // revisits an abandoned nativeId to trigger the TTL check inside
  // compactChatGptConversation. Runs on every bridge call (identity checks
  // happen far more often than conversation fetches), so an abandoned entry
  // is swept within one TTL window of being abandoned, not forever.
  for (const [cachedNativeId, cached] of chunkCache) {
    if (Date.now() - cached.createdAtMs > chunkCacheTtlMs) chunkCache.delete(cachedNativeId);
  }
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

  // Builds the full-fidelity projected node set for a source conversation,
  // packing nodes into ordered, byte-bounded, node-aligned chunks (a node's
  // JSON is never split across chunks) so the caller can pull the whole
  // conversation across several bounded scripting-result calls instead of
  // one call that would either overflow Chrome's limit or force a lossy
  // projection. A conversation that fits in a single chunk is still wrapped
  // in the same {chunked, chunkIndex, totalChunks} shape for a uniform
  // caller contract; chunked === false is the common case.
  function buildChatGptChunks(source) {
    const nodeEntries = Object.entries(source.mapping).map(([nodeId, node]) => [nodeId, projectChatGptNode(node)]);
    const chunks = [];
    let current = {};
    let currentBytes = 0;
    for (const [nodeId, projectedNode] of nodeEntries) {
      const nodeBytes = new globalThis.TextEncoder().encode(JSON.stringify(projectedNode)).length;
      if (nodeBytes > chatGptChunkPackTargetBytes && currentBytes === 0) {
        // A single node's own projection already exceeds the packing
        // target. There is no lossless way to split one node's JSON across
        // chunks, so this node gets its own chunk and we accept the risk
        // that assertScriptingResultBound rejects it below -- a loud,
        // specific failure, never a silent drop of that node's content.
        chunks.push({ [nodeId]: projectedNode });
        continue;
      }
      if (currentBytes > 0 && currentBytes + nodeBytes > chatGptChunkPackTargetBytes) {
        chunks.push(current);
        current = {};
        currentBytes = 0;
      }
      current[nodeId] = projectedNode;
      currentBytes += nodeBytes;
    }
    if (currentBytes > 0 || chunks.length === 0) chunks.push(current);
    if (chunks.length > maxChatGptChunks) {
      throw sizeError("backfill_bridge_projection_too_many_chunks", chunks.length, maxChatGptChunks);
    }
    return chunks;
  }

  function compactChatGptConversation(body, chunkIndex, nativeId) {
    let projectedChunks;
    let header;
    if (chunkIndex > 0) {
      const cached = chunkCache.get(nativeId);
      if (!cached || Date.now() - cached.createdAtMs > chunkCacheTtlMs) {
        throw new Error("backfill_bridge_chunk_cache_miss");
      }
      if (chunkIndex >= cached.chunks.length) {
        throw new Error("backfill_bridge_chunk_index_out_of_range");
      }
      projectedChunks = cached.chunks;
      header = cached.header;
    } else {
      let source;
      try { source = JSON.parse(body); } catch { throw new Error("provider_contract_drift:chatgpt_conversation_not_json_object"); }
      if (!source || typeof source !== "object" || !source.mapping || typeof source.mapping !== "object" || Array.isArray(source.mapping)) {
        throw new Error("provider_contract_drift:chatgpt_conversation.mapping_must_be_object");
      }
      header = chatGptConversationHeader(source);
      projectedChunks = buildChatGptChunks(source);
      if (projectedChunks.length > 1) {
        chunkCache.set(nativeId, { chunks: projectedChunks, header, createdAtMs: Date.now() });
      } else {
        chunkCache.delete(nativeId);
      }
    }
    const projected = JSON.stringify({
      polylogue_bridge_projection: "chatgpt-native-bridge-v1",
      ...header,
      chunked: projectedChunks.length > 1,
      chunkIndex,
      totalChunks: projectedChunks.length,
      mapping: projectedChunks[chunkIndex],
    });
    // The last chunk has now been built and is about to be returned to the
    // caller: free the cached projection immediately rather than waiting
    // for the TTL sweep above or a future call that may never come. Safe to
    // do before the size check below -- if that check throws, the caller
    // never got a successful last-chunk response and will restart the whole
    // fetch from chunkIndex 0 on retry, which is correct (a partial/rejected
    // last chunk is not a completed transfer either way).
    if (chunkIndex === projectedChunks.length - 1) chunkCache.delete(nativeId);
    const bytes = new globalThis.TextEncoder().encode(projected).length;
    if (bytes > compactChatGptBridgeMaxBytes) {
      throw sizeError("backfill_bridge_projection_too_large", bytes, compactChatGptBridgeMaxBytes);
    }
    return projected;
  }

  async function chatGptRequest() {
    let url;
    let requestedChunkIndex = 0;
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
      requestedChunkIndex = boundedInteger(request.params?.chunkIndex ?? 0, "chunk_index", 0, maxChatGptChunks - 1);
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
    if (request.operation === "conversation" && requestedChunkIndex > 0) {
      // Chunk N>0 reuses the MAIN-world cache the chunk-0 fetch populated
      // (see buildChatGptChunks/compactChatGptConversation) -- no repeat
      // network round-trip against the provider's conversation endpoint.
      const response = { ok: true, status: 200, contentType: "application/json", retryAfter: null, body: null };
      response.body = compactChatGptConversation(null, requestedChunkIndex, requestedNativeId);
      assertScriptingResultBound(response);
      return response;
    }
    const response = await fetchBounded(url.href, {
      credentials: "include",
      cache: "no-store",
      headers: { Authorization: `Bearer ${accessToken}`, "ChatGPT-Account-Id": accountId },
    }, request.operation === "conversation" ? compactChatGptSourceMaxBytes : maxResponseBytes,
    request.operation === "conversation" ? "backfill_bridge_source_response_too_large" : "backfill_bridge_response_too_large");
    if (request.operation === "conversation" && response.ok) {
      response.body = compactChatGptConversation(response.body, requestedChunkIndex, requestedNativeId);
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
