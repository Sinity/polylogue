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
  return "";
}

function normalizedRole(raw) {
  if (["user", "assistant", "system", "tool"].includes(raw)) return raw;
  if (["function", "tool_use", "tool_result"].includes(raw)) return "tool";
  if (raw === "human") return "user";
  if (raw === "claude") return "assistant";
  return "unknown";
}

function claudeText(message) {
  if (typeof message?.text === "string" && message.text) return message.text;
  if (typeof message?.content === "string" && message.content) return message.content;
  if (!Array.isArray(message?.content)) return "";
  return message.content.flatMap((part) => {
    if (typeof part === "string") return [part];
    if (part && typeof part === "object" && typeof part.text === "string") return [part.text];
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
      return [{
        provider_turn_id: requireString(message.id || node.id || nodeId, "chatgpt_conversation.message.id"),
        role: normalizedRole(message.author.role),
        text,
        timestamp: isoTimestamp(message.create_time),
        parent_turn_id: node.parent || null,
        provider_meta: {
          node_id: nodeId,
          content_type: message.content.content_type || "text",
          status: message.status || null,
          model_slug: metadata.model_slug || null,
          capture_source: "chatgpt_backend_api",
        },
      }];
    });
    const compact = body.polylogue_bridge_projection === "chatgpt-native-compact-v1";
    return envelope({ provider: "chatgpt", nativeId: item.native_id, title: body.title || item.title, createdAt: isoTimestamp(body.create_time), updatedAt: isoTimestamp(body.update_time) || item.updated_at, turns, rawPayload: body, adapterName: compact ? "chatgpt-backfill-compact-v1" : "chatgpt-backfill-native-v1", sourceUrl: `https://chatgpt.com/c/${item.native_id}`, attribution, captureFidelity: compact ? "native_compact" : "native_full" });
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
