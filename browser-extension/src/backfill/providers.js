function requireArray(value, path) {
  if (!Array.isArray(value)) throw new Error(`provider_contract_drift:${path}_must_be_array`);
  return value;
}

function requireString(value, path) {
  if (typeof value !== "string" || !value) throw new Error(`provider_contract_drift:${path}_must_be_string`);
  return value;
}

function responseClass(response) {
  if (response.ok) return "success";
  if (response.status === 429) return "rate_limited";
  if (response.status === 401 || response.status === 403) return "auth_or_challenge";
  if (response.status >= 500) return "transport";
  return "fatal";
}

async function jsonResponse(response, label) {
  const body = await response.json().catch(() => null);
  if (!body || typeof body !== "object") throw new Error(`provider_contract_drift:${label}_not_json_object`);
  return body;
}

function isoTimestamp(value) {
  if (typeof value === "number") return new Date(value < 10_000_000_000 ? value * 1000 : value).toISOString();
  return typeof value === "string" && value ? value : null;
}

const REQUEST_OPTIONS = Object.freeze({ credentials: "include", cache: "no-store" });

async function providerRequest(fetchImpl, url) {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort("provider_request_timeout"), PROVIDER_REQUEST_TIMEOUT_MS);
  try {
    return await fetchImpl(url, { ...REQUEST_OPTIONS, signal: controller.signal });
  } finally {
    globalThis.clearTimeout(timeout);
  }
}

function chatGptText(content) {
  if (Array.isArray(content?.parts)) {
    const parts = content.parts.flatMap((part) => {
      if (typeof part === "string" && part) return [part];
      if (part && typeof part === "object" && typeof part.text === "string" && part.text) return [part.text];
      return [];
    });
    if (parts.length) return parts.join("\n");
  }
  if (typeof content?.text === "string" && content.text) return content.text;
  if (typeof content?.result === "string" && content.result) return content.result;
  // A "thoughts" (reasoning) node keeps its payload as an array of
  // {summary, content} entries rather than parts/text/result. Without this,
  // a turn whose ONLY content is a thoughts node produces an empty text ->
  // the caller's `if (!text) return []` in normalizeCapture drops the turn
  // silently, which can misclassify a genuine (if reasoning-only) exchange
  // as `no_turns` and skip the whole conversation. This session-summary
  // text is a dedup/no_turns signal, not the archival record (the fixed
  // full-fidelity raw_provider_payload.mapping is), but it should not lie.
  if (Array.isArray(content?.thoughts)) {
    const thoughts = content.thoughts.flatMap((thought) => {
      if (typeof thought?.content === "string" && thought.content) return [thought.content];
      if (typeof thought?.summary === "string" && thought.summary) return [thought.summary];
      return [];
    });
    if (thoughts.length) return thoughts.join("\n");
  }
  return "";
}

function normalizedRole(raw) {
  if (["user", "assistant", "system", "tool"].includes(raw)) return raw;
  if (["function", "tool_use", "tool_result"].includes(raw)) return "tool";
  if (raw === "human") return "user";
  if (raw === "claude") return "assistant";
  return "unknown";
}

// polylogue-ah21: project a ChatGPT mapping node's own content_type/recipient
// evidence into the typed BrowserCaptureBlock wire shape instead of leaving
// tool call/result turns as opaque prose. This runs for every turn built
// here (compact and full/non-compact alike), because that is exactly the
// turn array the parser falls back to whenever it cannot -- or chooses not
// to -- trust body.mapping as a full native payload (see
// polylogue/sources/parsers/browser_capture.py:_has_chatgpt_native_payload,
// which explicitly excludes the size-bounded compact projection). tool_id
// pairing is constructed, not guessed: a call node's own id is its tool_id,
// and its result node's tool_id is the call node's id (the result's parent
// in the mapping tree), so a call and its result always pair 1:1.
function chatGptTurnBlocks({ contentType, recipient, text, ownId, parentId }) {
  if (recipient && text) {
    let parsedInput = null;
    try {
      const candidate = JSON.parse(text);
      if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) parsedInput = candidate;
    } catch {
      parsedInput = null;
    }
    if (parsedInput) {
      return [{ type: "tool_use", tool_name: recipient, tool_id: ownId, tool_input: parsedInput, metadata: { content_type: contentType } }];
    }
  }
  if (contentType === "code") {
    return [{ type: "tool_use", tool_name: "code_interpreter", tool_id: ownId, tool_input: { code: text }, metadata: { content_type: contentType } }];
  }
  if (contentType === "execution_output") {
    return [{ type: "tool_result", tool_id: parentId, text, metadata: { content_type: contentType } }];
  }
  if (contentType === "thoughts" || contentType === "reasoning_recap") {
    return [{ type: "thinking", text, metadata: { content_type: contentType } }];
  }
  return [];
}

function claudeText(message) {
  if (typeof message?.text === "string" && message.text) return message.text;
  if (typeof message?.content === "string" && message.content) return message.content;
  if (!Array.isArray(message?.content)) return "";
  return message.content.flatMap((part) => {
    if (typeof part === "string") return [part];
    if (part && typeof part === "object" && typeof part.text === "string") return [part.text];
    // Claude's extended-thinking content blocks carry their payload under
    // `thinking`, not `text`. Without this, a turn whose ONLY content is a
    // thinking block yields an empty summary text and is dropped by
    // normalizeCapture's `if (!text) return []` -- same class of gap as the
    // ChatGPT `thoughts` fix above, for the same reason (this session-
    // summary text is a dedup/no_turns signal; the archival record is the
    // unmodified rawPayload body, already full-fidelity for Claude).
    if (part && typeof part === "object" && typeof part.thinking === "string") return [part.thinking];
    return [];
  }).filter(Boolean).join("\n");
}

function envelope({ provider, nativeId, title, createdAt, updatedAt, turns, rawPayload, adapterName, sourceUrl, attribution, captureFidelity = "native_full" }) {
  return {
    polylogue_capture_kind: "browser_llm_session",
    schema_version: 1,
    capture_id: `${provider}:${nativeId}`,
    source: "browser-extension",
    provenance: {
      source_url: sourceUrl,
      page_title: title || null,
      captured_at: new Date().toISOString(),
      extension_id: globalThis.chrome?.runtime?.id || null,
      adapter_name: adapterName,
      adapter_version: globalThis.chrome?.runtime?.getManifest?.().version || null,
      capture_mode: "snapshot",
      provider_meta: { backfill: attribution },
    },
    session: {
      provider,
      provider_session_id: nativeId,
      title: title || nativeId,
      created_at: createdAt,
      updated_at: updatedAt,
      provider_meta: { capture_fidelity: captureFidelity, backfill: attribution },
      turns: turns.map((turn, ordinal) => ({ ...turn, ordinal })),
    },
    provider_meta: { capture_fidelity: captureFidelity, backfill: attribution },
    raw_provider_payload: rawPayload,
  };
}

const CHATGPT_INVENTORY_PARTITIONS = Object.freeze([
  { archived: false, starred: false },
  { archived: false, starred: true },
  { archived: true, starred: false },
  { archived: true, starred: true },
]);

function chatGptInventoryCursor(cursor) {
  const match = String(cursor || "0").match(/^(?:(\d+):)?(\d+)$/);
  if (!match) throw new Error("provider_contract_drift:chatgpt_inventory.cursor_invalid");
  const partition = Number.parseInt(match[1] || "0", 10);
  const offset = Number.parseInt(match[2], 10);
  if (partition < 0 || partition >= CHATGPT_INVENTORY_PARTITIONS.length) {
    throw new Error("provider_contract_drift:chatgpt_inventory.cursor_invalid");
  }
  return { partition, offset };
}

export class ChatGptBackfillAdapter {
  constructor(fetchImpl = globalThis.fetch, options = {}) {
    this.fetchImpl = fetchImpl;
    this.requirePageContext = Boolean(options.requirePageContext);
    this.provider = "chatgpt";
  }
  configure() {}
  requestCost() { return 2; }
  async enumerate(cursor = "0", cutoff = null) {
    const { partition, offset } = chatGptInventoryCursor(cursor);
    const flags = CHATGPT_INVENTORY_PARTITIONS[partition];
    const response = await providerRequest(this.fetchImpl, `https://chatgpt.com/backend-api/conversations?offset=${offset}&limit=28&order=updated&is_archived=${flags.archived}&is_starred=${flags.starred}`);
    if (this.requirePageContext && response.polyloguePageContext !== true) {
      return { response, classification: "auth_or_challenge", items: [], next_cursor: cursor, done: false, request_count: 1 };
    }
    if (!response.ok) return { response, classification: responseClass(response), items: [], next_cursor: cursor, done: false, request_count: 1 };
    const body = await jsonResponse(response, "chatgpt_inventory");
    const records = requireArray(body.items, "chatgpt_inventory.items");
    const projected = records.map((item, index) => ({
      native_id: requireString(item.id, `chatgpt_inventory.items[${index}].id`),
      title: typeof item.title === "string" ? item.title : null,
      updated_at: isoTimestamp(item.update_time),
    }));
    const items = projected.filter((item) => !cutoff || !item.updated_at || item.updated_at >= cutoff);
    const crossedCutoff = Boolean(cutoff && projected.some((item) => item.updated_at && item.updated_at < cutoff));
    const total = Number.isFinite(body.total) ? body.total : offset + records.length;
    const nextOffset = offset + records.length;
    const partitionDone = nextOffset >= total || crossedCutoff;
    const finalPartition = partition === CHATGPT_INVENTORY_PARTITIONS.length - 1;
    const nextCursor = partitionDone && !finalPartition ? `${partition + 1}:0` : `${partition}:${nextOffset}`;
    return { response, classification: "success", items, next_cursor: nextCursor, done: partitionDone && finalPartition, request_count: 1 };
  }
  async fetchNative(nativeId) { return providerRequest(this.fetchImpl, `https://chatgpt.com/backend-api/conversation/${encodeURIComponent(nativeId)}`); }
  classifyResponse(response) {
    if (this.requirePageContext && response.polyloguePageContext !== true) return "auth_or_challenge";
    return responseClass(response);
  }
  async normalizeCapture(response, item, attribution) {
    const body = await jsonResponse(response, "chatgpt_conversation");
    const mapping = body.mapping && typeof body.mapping === "object" ? Object.entries(body.mapping) : null;
    if (!mapping) throw new Error("provider_contract_drift:chatgpt_conversation.mapping_must_be_object");
    const turns = mapping.flatMap(([nodeId, node]) => {
      const message = node?.message;
      if (!message || !message.author || !message.content) return [];
      const text = chatGptText(message.content).trim();
      if (!text) return [];
      const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
      const providerTurnId = requireString(message.id || node.id || nodeId, "chatgpt_conversation.message.id");
      const contentType = message.content.content_type || "text";
      return [{
        provider_turn_id: providerTurnId,
        role: normalizedRole(message.author.role),
        text,
        timestamp: isoTimestamp(message.create_time),
        parent_turn_id: node.parent || null,
        blocks: chatGptTurnBlocks({ contentType, recipient: message.recipient || null, text, ownId: providerTurnId, parentId: node.parent || null }),
        provider_meta: {
          node_id: nodeId,
          content_type: contentType,
          status: message.status || null,
          model_slug: metadata.model_slug || null,
          capture_source: "chatgpt_backend_api",
        },
      }];
    });
    // page_transport.js's ChatGPT bridge projection is full-fidelity (every
    // content-type payload key and every metadata key preserved, chunked
    // across bounded scripting-result calls rather than field-dropped -- see
    // polylogue-thoughts-fidelity), so a backfilled ChatGPT capture is always
    // native_full now; the lossy "compact" projection tag/capture_fidelity
    // value no longer exists on the emitting side (the Python parser retains
    // read support for historical archive rows tagged native_compact).
    return envelope({ provider: "chatgpt", nativeId: item.native_id, title: body.title || item.title, createdAt: isoTimestamp(body.create_time), updatedAt: isoTimestamp(body.update_time) || item.updated_at, turns, rawPayload: body, adapterName: "chatgpt-backfill-native-v1", sourceUrl: `https://chatgpt.com/c/${item.native_id}`, attribution, captureFidelity: "native_full" });
  }
}

export class ClaudeBackfillAdapter {
  constructor(fetchImpl = globalThis.fetch, organizationId = null, options = {}) {
    this.fetchImpl = fetchImpl;
    this.organizationId = organizationId;
    this.requirePageContext = Boolean(options.requirePageContext);
    this.provider = "claude-ai";
  }
  configure(options = {}) {
    if (options.claudeOrganizationId) this.organizationId = options.claudeOrganizationId;
  }
  requestCost(operation) {
    return operation === "enumerate" && !this.organizationId ? 2 : 1;
  }
  async organization() {
    if (this.organizationId) return { id: this.organizationId, request_count: 0 };
    const response = await providerRequest(this.fetchImpl, "https://claude.ai/api/organizations");
    if (this.requirePageContext && response.polyloguePageContext !== true) {
      return { response, classification: "auth_or_challenge", request_count: 1 };
    }
    if (!response.ok) return { response, classification: responseClass(response), request_count: 1 };
    const organizations = requireArray(await response.json(), "claude_organizations");
    this.organizationId = requireString(organizations[0]?.uuid, "claude_organizations[0].uuid");
    return { id: this.organizationId, request_count: 1 };
  }
  async enumerate(cursor = "0", cutoff = null) {
    const organization = await this.organization();
    if (!organization.id) return { ...organization, items: [], next_cursor: cursor, done: false };
    const offset = Number.parseInt(cursor || "0", 10) || 0;
    const response = await providerRequest(this.fetchImpl, `https://claude.ai/api/organizations/${encodeURIComponent(organization.id)}/chat_conversations?limit=100&offset=${offset}`);
    const requestCount = organization.request_count + 1;
    if (!response.ok) return { response, classification: responseClass(response), items: [], next_cursor: cursor, done: false, request_count: requestCount, provider_options: { claudeOrganizationId: organization.id } };
    const body = await response.json();
    const records = requireArray(body, "claude_inventory");
    const projected = records.map((item, index) => ({
      native_id: requireString(item.uuid, `claude_inventory[${index}].uuid`),
      title: typeof item.name === "string" ? item.name : null,
      updated_at: isoTimestamp(item.updated_at),
    }));
    const items = projected.filter((item) => !cutoff || !item.updated_at || item.updated_at >= cutoff);
    return { response, classification: "success", items, next_cursor: String(offset + records.length), done: records.length < 100, request_count: requestCount, provider_options: { claudeOrganizationId: organization.id } };
  }
  async fetchNative(nativeId) {
    const organization = await this.organization();
    if (!organization.id) return organization.response;
    const query = new URLSearchParams({
      tree: "True",
      rendering_mode: "messages",
      render_all_tools: "true",
      consistency: "strong",
    });
    return providerRequest(
      this.fetchImpl,
      `https://claude.ai/api/organizations/${encodeURIComponent(organization.id)}/chat_conversations/${encodeURIComponent(nativeId)}?${query}`,
    );
  }
  classifyResponse(response) {
    if (this.requirePageContext && response.polyloguePageContext !== true) return "auth_or_challenge";
    return responseClass(response);
  }
  async normalizeCapture(response, item, attribution) {
    const body = await jsonResponse(response, "claude_conversation");
    const messages = requireArray(body.chat_messages, "claude_conversation.chat_messages");
    const turns = messages.flatMap((message, index) => {
      const text = claudeText(message).trim();
      if (!text) return [];
      return [{
        provider_turn_id: requireString(message.uuid || message.id, `claude_conversation.chat_messages[${index}].uuid`),
        role: normalizedRole(message.sender || message.role || message.author),
        text,
        timestamp: isoTimestamp(message.created_at),
        parent_turn_id: message.parent_message_uuid || message.parent_uuid || null,
        provider_meta: {
          model: message.model || null,
          sender: message.sender || message.role || null,
          capture_source: "claude_chat_conversations_api",
        },
      }];
    });
    return envelope({ provider: "claude-ai", nativeId: item.native_id, title: body.name || item.title, createdAt: isoTimestamp(body.created_at), updatedAt: isoTimestamp(body.updated_at) || item.updated_at, turns, rawPayload: body, adapterName: "claude-ai-backfill-native-v1", sourceUrl: `https://claude.ai/chat/${item.native_id}`, attribution });
  }
}

export function providerAdapters(fetchImpl = globalThis.fetch, options = {}) {
  return {
    chatgpt: new ChatGptBackfillAdapter(fetchImpl, { requirePageContext: options.requirePageContext }),
    "claude-ai": new ClaudeBackfillAdapter(fetchImpl, options.claudeOrganizationId || null, { requirePageContext: options.requirePageContext }),
  };
}
import { PROVIDER_REQUEST_TIMEOUT_MS } from "./models.js";
