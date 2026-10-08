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

async function providerRequest(fetchImpl, url, signal, captureBundle = null, queueContext = null) {
  signal?.throwIfAborted();
  return fetchImpl(url, { ...REQUEST_OPTIONS, signal, ...(captureBundle ? { captureBundle } : {}), ...(queueContext ? { queueContext } : {}) });
}

async function normalizeNative(response, item, attribution, signal, context = null) {
  if (typeof response.normalizeCapture !== "function") throw new Error("native_capture_transport_unavailable");
  return response.normalizeCapture(item, attribution, {}, signal, context);
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
  async enumerate(cursor = "0", cutoff = null, signal) {
    const { partition, offset } = chatGptInventoryCursor(cursor);
    const flags = CHATGPT_INVENTORY_PARTITIONS[partition];
    const response = await providerRequest(this.fetchImpl, `https://chatgpt.com/backend-api/conversations?offset=${offset}&limit=28&order=updated&is_archived=${flags.archived}&is_starred=${flags.starred}`, signal);
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
  async fetchNative(nativeId, signal, context = null) { return providerRequest(this.fetchImpl, `https://chatgpt.com/backend-api/conversation/${encodeURIComponent(nativeId)}`, signal, null, context); }
  classifyResponse(response) {
    if (this.requirePageContext && response.polyloguePageContext !== true) return "auth_or_challenge";
    return responseClass(response);
  }
  async normalizeCapture(response, item, attribution, signal, context = null) { return normalizeNative(response, item, attribution, signal, context); }

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
  async organization(signal) {
    if (this.organizationId) return { id: this.organizationId, request_count: 0 };
    const response = await providerRequest(this.fetchImpl, "https://claude.ai/api/organizations", signal);
    if (this.requirePageContext && response.polyloguePageContext !== true) {
      return { response, classification: "auth_or_challenge", request_count: 1 };
    }
    if (!response.ok) return { response, classification: responseClass(response), request_count: 1 };
    const organizations = requireArray(await response.json(), "claude_organizations");
    const selected = response.polylogueSelectedOrganizationId;
    if (!selected || !organizations.some((organization) => organization?.uuid === selected)) throw new Error("provider_contract_drift:claude_selected_organization_unavailable");
    this.organizationId = selected;
    return { id: this.organizationId, request_count: 1 };
  }
  async enumerate(cursor = "0", cutoff = null, signal) {
    const organization = await this.organization(signal);
    if (!organization.id) return { ...organization, items: [], next_cursor: cursor, done: false };
    const offset = Number.parseInt(cursor || "0", 10) || 0;
    const response = await providerRequest(this.fetchImpl, `https://claude.ai/api/organizations/${encodeURIComponent(organization.id)}/chat_conversations?limit=100&offset=${offset}`, signal);
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
  async fetchNative(nativeId, signal, context = null) {
    const organization = await this.organization(signal);
    if (!organization.id) return organization.response;
    const query = new URLSearchParams({
      tree: "True",
      rendering_mode: "messages",
      render_all_tools: "true",
      consistency: "strong",
    });
    return providerRequest(
      this.fetchImpl,
      `https://claude.ai/api/organizations/${encodeURIComponent(organization.id)}/chat_conversations/${encodeURIComponent(nativeId)}?${query}`, signal, null, context,
    );
  }
  classifyResponse(response) {
    if (this.requirePageContext && response.polyloguePageContext !== true) return "auth_or_challenge";
    return responseClass(response);
  }
  async normalizeCapture(response, item, attribution, signal, context = null) { return normalizeNative(response, item, attribution, signal, context); }

}

// Grok keeps the original conversation and response reply files separately;
// the native normalizer serializes their named acquisition bundle.
const GROK_PAGE_SIZE = 60;

export class GrokBackfillAdapter {
  constructor(fetchImpl = globalThis.fetch, options = {}) {
    this.fetchImpl = fetchImpl;
    this.requirePageContext = Boolean(options.requirePageContext);
    this.provider = "grok";
    this.bundleOwner = options.nativeBundleOwner;
  }
  configure() {}
  // Every native fetch costs two provider requests (conversation metadata +
  // responses), same accounting ChatGPT uses for its own two-stage fetch.
  requestCost(operation = "fetch", item = null) {
    if (operation === "enumerate") return 1;
    return ["conversation", "responses"].filter((name) => !item?.capture_bundle_replies?.includes(name)).length;
  }
  async enumerate(cursor = "0", cutoff = null, signal) {
    const pageToken = cursor && cursor !== "0" ? cursor : null;
    const url = new URL("https://grok.com/rest/app-chat/conversations");
    url.searchParams.set("pageSize", String(GROK_PAGE_SIZE));
    if (pageToken) url.searchParams.set("pageToken", pageToken);
    const response = await providerRequest(this.fetchImpl, url.href, signal);
    if (this.requirePageContext && response.polyloguePageContext !== true) {
      return { response, classification: "auth_or_challenge", items: [], next_cursor: cursor, done: false, request_count: 1 };
    }
    if (!response.ok) return { response, classification: responseClass(response), items: [], next_cursor: cursor, done: false, request_count: 1 };
    const body = await jsonResponse(response, "grok_inventory");
    const records = requireArray(body.conversations, "grok_inventory.conversations");
    const projected = records.map((item, index) => ({
      native_id: requireString(item.conversationId, `grok_inventory.conversations[${index}].conversationId`),
      title: typeof item.title === "string" ? item.title : null,
      updated_at: isoTimestamp(item.modifyTime),
    }));
    const items = projected.filter((item) => !cutoff || !item.updated_at || item.updated_at >= cutoff);
    const crossedCutoff = Boolean(cutoff && projected.some((item) => item.updated_at && item.updated_at < cutoff));
    const nextPageToken = typeof body.nextPageToken === "string" && body.nextPageToken ? body.nextPageToken : null;
    const done = !nextPageToken || crossedCutoff;
    return { response, classification: "success", items, next_cursor: nextPageToken || cursor, done, request_count: 1 };
  }
  async fetchNative(nativeId, signal, context = null) {
    if (!this.bundleOwner) throw new Error("native_bundle_owner_unavailable");
    const bundle = await this.bundleOwner.begin(nativeId, signal, context);
    const fetchReply = async (name, suffix = "") => {
      if (bundle.replies[name]) return this.bundleOwner.restoreReply(bundle.replies[name], signal);
      const response = await providerRequest(this.fetchImpl, `https://grok.com/rest/app-chat/conversations/${encodeURIComponent(nativeId)}${suffix}`, signal, { id: bundle.id, name });
      await this.bundleOwner.response(bundle.id, name, response, signal);
      return response;
    };
    const conversationResponse = await fetchReply("conversation");
    if (!conversationResponse.ok) return conversationResponse;
    const responsesResponse = await fetchReply("responses", "/responses");
    if (!responsesResponse.ok) return responsesResponse;
    await this.bundleOwner.finish(bundle.id, signal);
    return { ...responsesResponse, relatedResponses: { conversation: conversationResponse },
      normalizeCapture: (item, attribution, _related, normalizationSignal, queueContext = null) => responsesResponse.normalizeCapture(item, attribution, { conversation: conversationResponse }, normalizationSignal, queueContext) };
  }

  classifyResponse(response) {
    if (this.requirePageContext && response.polyloguePageContext !== true) return "auth_or_challenge";
    return responseClass(response);
  }
  async normalizeCapture(response, item, attribution, signal, context = null) { return normalizeNative(response, item, attribution, signal, context); }

}

export function providerAdapters(fetchImpl = globalThis.fetch, options = {}) {
  return {
    chatgpt: new ChatGptBackfillAdapter(fetchImpl, { requirePageContext: options.requirePageContext }),
    "claude-ai": new ClaudeBackfillAdapter(fetchImpl, options.claudeOrganizationId || null, { requirePageContext: options.requirePageContext }),
    grok: new GrokBackfillAdapter(fetchImpl, { requirePageContext: options.requirePageContext, nativeBundleOwner: options.nativeBundleOwner }),
  };
}
