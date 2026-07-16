import { BackfillCoordinator } from "./backfill/coordinator.js";
import { BACKFILL_ALARM, DURABLE_RECEIVER_ACK_FIELDS, PROVIDER_REQUEST_TIMEOUT_MS } from "./backfill/models.js";
import { providerAdapters } from "./backfill/providers.js";
import { executeProviderPageRequest } from "./backfill/page_transport.js";
import { IndexedDbBackfillStore } from "./backfill/storage.js";
import {
  classifyLaunchFailure,
  executeChatGptLaunchInPage,
  inspectChatGptLaunchPage,
} from "./launch/chatgpt_launch.js";

const DEFAULT_RECEIVER = "http://127.0.0.1:8765";
const BACKGROUND_CAPTURE_MIN_INTERVAL_MS = 30000;
const ACTIVE_TAB_STATE_MIN_INTERVAL_MS = 4000;
const CAPTURE_LOG_LIMIT = 80;
const DEBUG_LOG_LIMIT = 160;
const CONVERSATION_TIMELINE_KEY = "polylogueConversationTimeline";
const CONVERSATION_TIMELINE_EVENT_LIMIT = 24;
const BACKFILL_RECOVERY_CHECKPOINT_KEY = "polylogueBackfillRecoveryCheckpoint";
const BACKFILL_WORKER_EPOCH = globalThis.crypto?.randomUUID?.() || `worker-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const CONVERSATION_TIMELINE_CONVERSATION_LIMIT = 80;
const POST_POLL_INTERVAL_MS = 5000;
const LAUNCH_ALARM = "polylogueLaunchWake";
const LAUNCH_MAX_EXTENSION_TRANSPORT_BYTES = 16 * 1024 * 1024;
const CAPTURE_MESSAGE_TIMEOUT_MS = 15000;
const BACKFILL_PAGE_REQUEST_TIMEOUT_MS = 58000;
const BACKFILL_TRANSPORT_TAB_TTL_MS = 5 * 60 * 1000;
const BACKFILL_TRANSPORT_CLEANUP_PREFIX = "polylogueBackfillTransportCleanup";
const recentBackgroundCaptures = new Map();
const recentActiveTabStateChecks = new Map();
// command_id -> true once dispatched to a content script this SW lifetime, so a
// fast poll cannot deliver the same command twice before its ack lands.
const inFlightPostCommands = new Set();
const pendingPostCommandAcks = new Map();
let postPollTimer = 0;
let backfillCoordinatorPromise = null;
let extensionInstanceIdPromise = null;
let launchExecutorIdPromise = null;
let launchPollPromise = null;
let storageMutationQueue = Promise.resolve();
let captureQueueMutationQueue = Promise.resolve();

function serializeStorageMutation(mutation) {
  const result = storageMutationQueue.then(mutation, mutation);
  storageMutationQueue = result.then(() => undefined, () => undefined);
  return result;
}

function serializeCaptureQueueMutation(mutation) {
  const result = captureQueueMutationQueue.then(mutation, mutation);
  captureQueueMutationQueue = result.then(() => undefined, () => undefined);
  return result;
}

function extensionInstanceId() {
  if (!extensionInstanceIdPromise) {
    const key = "polylogueExtensionInstanceId";
    const candidate = (async () => {
      const stored = await chrome.storage.local.get({ [key]: "" });
      if (stored[key]) return stored[key];
      const created = globalThis.crypto?.randomUUID?.() || `instance-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      await chrome.storage.local.set({ [key]: created });
      return created;
    })();
    extensionInstanceIdPromise = candidate;
    void candidate.catch(() => {
      if (extensionInstanceIdPromise === candidate) extensionInstanceIdPromise = null;
    });
  }
  return extensionInstanceIdPromise;
}

function launchExecutorId() {
  if (!launchExecutorIdPromise) {
    const key = "polylogueLaunchExecutorId";
    const candidate = (async () => {
      const session = chrome.storage.session;
      if (!session?.get || !session?.set) return `launch-${BACKFILL_WORKER_EPOCH}`;
      const stored = await session.get({ [key]: "" });
      if (stored[key]) return stored[key];
      const created = `launch-${globalThis.crypto?.randomUUID?.() || BACKFILL_WORKER_EPOCH}`;
      await session.set({ [key]: created });
      return created;
    })();
    launchExecutorIdPromise = candidate;
    void candidate.catch(() => {
      if (launchExecutorIdPromise === candidate) launchExecutorIdPromise = null;
    });
  }
  return launchExecutorIdPromise;
}

async function withExtensionInstanceAttribution(envelope) {
  const instanceId = await extensionInstanceId();
  return {
    ...envelope,
    provenance: {
      ...(envelope?.provenance || {}),
      // The service worker owns this persistent identity. Do not trust an
      // independently reloadable content script to choose the attribution.
      extension_instance_id: instanceId,
    },
  };
}

// polylogue-06zm: best-effort mirror of the backfill-ledger checkpoint to the
// local receiver. IndexedDB is always the fast primary source and the local
// chrome.storage.local copy (BACKFILL_RECOVERY_CHECKPOINT_KEY, jlme.4) is the
// first fallback; this receiver mirror only matters when BOTH of those are
// gone (a browser profile that was fully destroyed/reinstalled, not merely
// restarted). A mirror failure must never surface as a checkpoint error --
// see BackfillCoordinator.persistCheckpoint(), which awaits only the
// checkpoint() callback itself and already tolerates that throwing.
function mirrorBackfillCheckpointToReceiver(instanceId, checkpoint) {
  postJson(
    "/v1/backfill-checkpoint",
    { extension_instance_id: instanceId, checkpoint },
    null,
    PROVIDER_REQUEST_TIMEOUT_MS,
  ).catch(() => undefined);
}

async function restoreBackfillCheckpointFromReceiver(store, instanceId) {
  try {
    const remote = await getJson(
      `/v1/backfill-checkpoint?extension_instance_id=${encodeURIComponent(instanceId)}`,
      PROVIDER_REQUEST_TIMEOUT_MS,
    );
    if (remote?.checkpoint) return store.restoreRecoveryCheckpoint(remote.checkpoint);
  } catch {
    // No receiver-mirrored checkpoint reachable or available -- fall through
    // to whatever local state already exists (typically none, the same
    // empty-ledger outcome as before this fallback existed).
  }
  return { restored: 0, reason: "checkpoint_unavailable" };
}

async function backfillCoordinator() {
  if (!backfillCoordinatorPromise) {
    const candidate = (async () => {
      const store = new IndexedDbBackfillStore();
      const instanceId = await extensionInstanceId();
      const stored = await chrome.storage.local.get({ [BACKFILL_RECOVERY_CHECKPOINT_KEY]: null });
      const localRestore = await store.restoreRecoveryCheckpoint(stored[BACKFILL_RECOVERY_CHECKPOINT_KEY]);
      if (localRestore.reason === "checkpoint_unavailable") {
        // Neither IndexedDB nor the local chrome.storage.local mirror had a
        // usable checkpoint (restoreRecoveryCheckpoint already refuses to
        // touch a non-empty IndexedDB, so this only ever runs when there is
        // truly nothing local to lose).
        await restoreBackfillCheckpointFromReceiver(store, instanceId);
      }
      return new BackfillCoordinator({
        store,
        adapters: providerAdapters(providerPageFetch, { requirePageContext: true }),
        receiver: (envelope, serialized) => postJson(
          "/v1/browser-captures",
          envelope,
          serialized,
          PROVIDER_REQUEST_TIMEOUT_MS,
          true,
        ),
        receiverPreflight: backfillReceiverPreflight,
        checkpoint: async (checkpoint) => {
          await chrome.storage.local.set({ [BACKFILL_RECOVERY_CHECKPOINT_KEY]: checkpoint });
          mirrorBackfillCheckpointToReceiver(instanceId, checkpoint);
        },
        alarms: chrome.alarms,
        instanceId,
        receiverContractEpoch: BACKFILL_WORKER_EPOCH,
      });
    })();
    backfillCoordinatorPromise = candidate;
    void candidate.catch(() => {
      if (backfillCoordinatorPromise === candidate) backfillCoordinatorPromise = null;
    });
  }
  return backfillCoordinatorPromise;
}

// ---- Capture retry queue --------------------------------------------------
//
// When the receiver is unreachable or returns a 5xx/429, a capture envelope
// is durable-queued in chrome.storage.local instead of being dropped. A
// background alarm drains the queue with per-entry exponential backoff. The
// queue is intentionally bounded (both by entry count and serialized byte
// size) — a wedged/offline receiver must not let this grow unbounded; the
// oldest entries are evicted first and counted in `dropped_count`.
const CAPTURE_QUEUE_KEY = "polylogueCaptureQueue";
const CAPTURE_QUEUE_MAX_ENTRIES = 20;
const CAPTURE_QUEUE_MAX_BYTES = 40 * 1024 * 1024;
const CAPTURE_QUEUE_EMPTY = Object.freeze({ entries: [], dropped_count: 0 });
const CAPTURE_RETRY_ALARM = "polylogueCaptureRetry";
const CAPTURE_RETRY_BASE_DELAY_MS = 30000;
const CAPTURE_RETRY_MAX_DELAY_MS = 30 * 60 * 1000;
const CAPTURE_RETRY_ALARM_PERIOD_MINUTES = 1;
// In-memory mirror of the queue length so badge rendering never needs an
// extra storage round trip; reloaded from storage once at SW startup.
let cachedQueueLength = 0;

function timeoutError(label, timeoutMs) {
  const error = new Error(`${label}_timeout_after_${timeoutMs}ms`);
  error.name = "PolylogueTimeoutError";
  return error;
}

function withTimeout(promise, timeoutMs, label) {
  let timer = 0;
  const timeout = new Promise((_resolve, reject) => {
    timer = globalThis.setTimeout(() => reject(timeoutError(label, timeoutMs)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) globalThis.clearTimeout(timer);
  });
}

function injectionPlanForUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) {
      return [
        { files: ["src/content/chatgpt_bridge.js"], world: "MAIN" },
        { files: ["src/common.js", "src/content/message_layer.js", "src/content/chatgpt.js"] },
      ];
    }
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) {
      return [
        { files: ["src/content/claude_bridge.js"], world: "MAIN" },
        { files: ["src/common.js", "src/content/message_layer.js", "src/content/claude.js"] },
      ];
    }
    if (
      parsed.hostname === "grok.com" ||
      parsed.hostname.endsWith(".grok.com") ||
      parsed.hostname === "x.com" ||
      parsed.hostname.endsWith(".x.com") ||
      parsed.hostname === "twitter.com" ||
      parsed.hostname.endsWith(".twitter.com")
    ) {
      return [{ files: ["src/common.js", "src/content/grok.js"] }];
    }
  } catch {
    return [];
  }
  return [];
}

async function receiverSettings() {
  const stored = await chrome.storage.local.get({
    receiverAuthToken: "",
    receiverBaseUrl: DEFAULT_RECEIVER,
  });
  return {
    authToken: String(stored.receiverAuthToken || ""),
    baseUrl: String(stored.receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, ""),
  };
}

async function saveReceiverSettings(receiverBaseUrl, receiverAuthToken = "") {
  await chrome.storage.local.set({
    receiverAuthToken: String(receiverAuthToken || ""),
    receiverBaseUrl: String(receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, "") || DEFAULT_RECEIVER,
  });
  return receiverSettings();
}

function sessionKey(provider, providerSessionId) {
  return `${provider || "unknown"}:${providerSessionId || "unknown"}`;
}

function retryDelayForAttempt(attempts) {
  const delay = CAPTURE_RETRY_BASE_DELAY_MS * 2 ** Math.max(0, attempts);
  return Math.min(delay, CAPTURE_RETRY_MAX_DELAY_MS);
}

function byteLength(value) {
  return new TextEncoder().encode(JSON.stringify(value)).length;
}

function envelopeSessionSummary(envelope) {
  const session = envelope?.session || {};
  return {
    provider: session.provider || null,
    providerSessionId: session.provider_session_id || null,
    captureMode: session.provider_meta?.capture_fidelity || null,
    assetAcquisition: session.provider_meta?.asset_acquisition || null,
    turnCount: Array.isArray(session.turns) ? session.turns.length : null,
    attachmentCount: Array.isArray(session.turns)
      ? session.turns.reduce((count, turn) => count + (Array.isArray(turn.attachments) ? turn.attachments.length : 0), 0)
      : null,
  };
}

async function getCaptureQueue() {
  const stored = await chrome.storage.local.get({ [CAPTURE_QUEUE_KEY]: CAPTURE_QUEUE_EMPTY });
  const queue = stored[CAPTURE_QUEUE_KEY];
  if (!queue || !Array.isArray(queue.entries)) return { entries: [], dropped_count: 0 };
  return { entries: queue.entries, dropped_count: queue.dropped_count || 0 };
}

async function refreshQueueBadge() {
  const stored = await chrome.storage.local.get({ polylogueState: {} });
  const badge = badgeForState(stored.polylogueState || {});
  await chrome.action.setBadgeText({ text: badge.text });
  await chrome.action.setBadgeBackgroundColor({ color: badge.color });
}

async function saveCaptureQueue(queue) {
  await chrome.storage.local.set({ [CAPTURE_QUEUE_KEY]: queue });
  cachedQueueLength = queue.entries.length;
  await refreshQueueBadge();
  return queue;
}

async function ensureRetryAlarm() {
  if (!chrome.alarms?.create) return;
  await chrome.alarms.create(CAPTURE_RETRY_ALARM, {
    delayInMinutes: CAPTURE_RETRY_ALARM_PERIOD_MINUTES,
    periodInMinutes: CAPTURE_RETRY_ALARM_PERIOD_MINUTES,
  });
}

async function clearRetryAlarm() {
  if (!chrome.alarms?.clear) return;
  await chrome.alarms.clear(CAPTURE_RETRY_ALARM);
}

function isRetryableCaptureError(error) {
  if (!error) return false;
  if (typeof error.status === "number") return error.status >= 500 || error.status === 429;
  // No HTTP status means fetch itself rejected (offline, DNS failure, refused
  // connection, CORS) rather than the receiver answering with an error body.
  return true;
}

async function enqueueCaptureForRetry({ envelope, reason, error, tab = null }) {
  return serializeCaptureQueueMutation(async () => {
  const entry = {
    id: buildReceiverRequestId(),
    envelope,
    reason: reason || "content_script_capture",
    tab_id: tab?.id || null,
    tab_url: tab?.url || tab?.pendingUrl || null,
    enqueued_at: new Date().toISOString(),
    attempts: 0,
    next_attempt_at: new Date(Date.now() + retryDelayForAttempt(0)).toISOString(),
    last_error: String(error?.message || error || "unknown"),
  };
  if (byteLength(entry) > CAPTURE_QUEUE_MAX_BYTES) {
    // A single envelope over budget can never fit; queueing it would only
    // evict every other pending retry to make room for one that still won't
    // fit. Surface it as an immediate drop instead.
    const queue = await getCaptureQueue();
    await saveCaptureQueue({ entries: queue.entries, dropped_count: (queue.dropped_count || 0) + 1 });
    await appendCaptureLog({
      ok: false,
      reason: "capture_queue_entry_over_budget",
      error: entry.last_error,
    });
    return { queue: await getCaptureQueue(), accepted: false, evicted: [entry] };
  }
  const queue = await getCaptureQueue();
  let entries = [...queue.entries, entry];
  let droppedCount = queue.dropped_count || 0;
  const evicted = [];
  while (entries.length > CAPTURE_QUEUE_MAX_ENTRIES || byteLength(entries) > CAPTURE_QUEUE_MAX_BYTES) {
    evicted.push(entries[0]);
    entries = entries.slice(1);
    droppedCount += 1;
  }
  const nextQueue = { entries, dropped_count: droppedCount };
  await saveCaptureQueue(nextQueue);
  await ensureRetryAlarm();
  await appendCaptureLog({
    ok: false,
    reason: "capture_queued_for_retry",
    queued_id: entry.id,
    error: entry.last_error,
    queue_length: entries.length,
  });
  return { queue: nextQueue, accepted: true, evicted };
  });
}

async function drainCaptureQueue(trigger = "alarm") {
  return serializeCaptureQueueMutation(async () => {
  const queue = await getCaptureQueue();
  if (!queue.entries.length) {
    await clearRetryAlarm();
    return { drained: 0, remaining: 0 };
  }
  const now = Date.now();
  const remaining = [];
  let drained = 0;
  for (const entry of queue.entries) {
    const dueAt = Date.parse(entry.next_attempt_at || "") || 0;
    if (dueAt > now) {
      remaining.push(entry);
      continue;
    }
    const envelope = await withExtensionInstanceAttribution(entry.envelope);
    const summary = envelopeSessionSummary(envelope);
    try {
      const result = await postJson("/v1/browser-captures", envelope);
      drained += 1;
      const archiveState = { state: result.state || "spooled_only" };
      await updateSessionLedger({
        provider: summary.provider || result.provider,
        providerSessionId: summary.providerSessionId || result.provider_session_id,
        patch: {
          capture_mode: summary.captureMode,
          asset_acquisition: summary.assetAcquisition,
          turn_count: summary.turnCount,
          attachment_count: summary.attachmentCount,
          receiver_request_id: result.receiver_request_id || null,
          artifact_ref: result.artifact_ref || null,
          extension_instance_id: result.capture_instance_id || null,
          deduplicated: Boolean(result.deduplicated),
          archive_state: archiveState,
          last_error: null,
        },
      });
      await appendCaptureLog({
        ok: true,
        reason: "capture_retry_drained",
        provider: summary.provider || result.provider,
        provider_session_id: summary.providerSessionId || result.provider_session_id,
        capture_mode: summary.captureMode,
        receiver_request_id: result.receiver_request_id || null,
        artifact_ref: result.artifact_ref || null,
        queued_id: entry.id,
        attempts: entry.attempts,
      });
      await appendConversationTimeline({
        provider: summary.provider || result.provider,
        providerSessionId: summary.providerSessionId || result.provider_session_id,
        event: "captured",
        reason: "capture_retry_drained",
        detail: archiveState.state,
      });
      if (entry.tab_id) await setStateForTab(entry.tab_id, {
        online: true,
        captured: true,
        last_capture: result,
        archive_state: archiveState,
        provider: summary.provider || result.provider,
        provider_session_id: summary.providerSessionId || result.provider_session_id,
        capture_mode: summary.captureMode,
        asset_acquisition: summary.assetAcquisition,
        turn_count: summary.turnCount,
        attachment_count: summary.attachmentCount,
        extension_instance_id: result.capture_instance_id || null,
        deduplicated: Boolean(result.deduplicated),
        last_receiver_request_id: result.receiver_request_id || null,
      }, entry.tab_url);
    } catch (error) {
      if (!isRetryableCaptureError(error)) {
        await updateSessionLedger({
          provider: summary.provider,
          providerSessionId: summary.providerSessionId,
          patch: { last_error: String(error.message || error) },
        });
        await appendCaptureLog({
          ok: false,
          reason: "capture_retry_rejected",
          queued_id: entry.id,
          attempts: entry.attempts,
          error: String(error.message || error),
        });
        await appendConversationTimeline({
          provider: summary.provider,
          providerSessionId: summary.providerSessionId,
          event: "held_with_reason",
          reason: "capture_retry_drained",
          detail: "capture_rejected",
          tabId: entry.tab_id || null,
        });
        continue;
      }
      const attempts = entry.attempts + 1;
      remaining.push({
        ...entry,
        attempts,
        last_error: String(error.message || error),
        next_attempt_at: new Date(now + retryDelayForAttempt(attempts)).toISOString(),
      });
      await appendCaptureLog({
        ok: false,
        reason: "capture_retry_failed",
        queued_id: entry.id,
        attempts,
        error: String(error.message || error),
      });
    }
  }
  const nextQueue = { entries: remaining, dropped_count: queue.dropped_count || 0 };
  await saveCaptureQueue(nextQueue);
  if (!remaining.length) {
    await clearRetryAlarm();
  } else if (trigger === "alarm") {
    await appendDebugLog({ stage: "capture_retry_drain", drained, remaining: remaining.length });
  }
  return { drained, remaining: remaining.length };
  });
}

async function loadCaptureQueueIntoCache() {
  const queue = await getCaptureQueue();
  cachedQueueLength = queue.entries.length;
  if (queue.entries.length) await ensureRetryAlarm();
  return queue;
}

async function checkReceiverHealth() {
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  try {
    const response = await fetch(`${settings.baseUrl}/v1/status`, {
      headers: await requestHeaders({ requestId }),
    });
    const body = await response.json().catch(() => null);
    if (!body || typeof body !== "object") {
      return { ok: false, status: "unreachable", detail: "non_json_response" };
    }
    if (body.error === "unauthorized") {
      return { ok: true, status: "unauthorized", detail: body.error };
    }
    if (body.ok === true) {
      return { ok: true, status: "ok", detail: null };
    }
    // Any other well-formed JSON body still proves the receiver answered —
    // report it as reachable with whatever error detail it supplied.
    return { ok: true, status: "error", detail: body.error || `http_${response.status}` };
  } catch (error) {
    return { ok: false, status: "unreachable", detail: String(error.message || error) };
  }
}

async function appendCaptureLog(entry) {
  const stored = await chrome.storage.local.get({ polylogueCaptureLog: [] });
  const prior = Array.isArray(stored.polylogueCaptureLog) ? stored.polylogueCaptureLog : [];
  const next = [
    {
      at: new Date().toISOString(),
      ...entry,
    },
    ...prior,
  ].slice(0, CAPTURE_LOG_LIMIT);
  await chrome.storage.local.set({ polylogueCaptureLog: next });
  return next;
}

function sanitizeDebugDetails(value, depth = 0) {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") return value.length > 160 ? `${value.slice(0, 157)}...` : value;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) return { count: value.length };
  if (typeof value !== "object" || depth > 2) return String(value);
  const redactedKeys = new Set(["body", "envelope", "raw_provider_payload", "text", "turns", "messages", "content"]);
  const out = {};
  for (const [key, item] of Object.entries(value)) {
    if (redactedKeys.has(key)) {
      out[key] = "[redacted]";
      continue;
    }
    out[key] = sanitizeDebugDetails(item, depth + 1);
  }
  return out;
}

async function appendDebugLog(entry) {
  const stored = await chrome.storage.local.get({ polylogueDebugLog: [] });
  const prior = Array.isArray(stored.polylogueDebugLog) ? stored.polylogueDebugLog : [];
  const next = [
    {
      at: new Date().toISOString(),
      ...sanitizeDebugDetails(entry),
    },
    ...prior,
  ].slice(0, DEBUG_LOG_LIMIT);
  await chrome.storage.local.set({ polylogueDebugLog: next });
  return next;
}

async function updateSessionLedger({ provider, providerSessionId, patch }) {
  if (!provider || !providerSessionId) return null;
  return serializeStorageMutation(async () => {
    const stored = await chrome.storage.local.get({ polylogueSessionLedger: {} });
    const ledger =
      stored.polylogueSessionLedger && typeof stored.polylogueSessionLedger === "object"
        ? stored.polylogueSessionLedger
        : {};
    const key = sessionKey(provider, providerSessionId);
    const next = {
      ...(ledger[key] || {}),
      provider,
      provider_session_id: providerSessionId,
      updated_at: new Date().toISOString(),
      ...patch,
    };
    await chrome.storage.local.set({ polylogueSessionLedger: { ...ledger, [key]: next } });
    return next;
  });
}

async function appendConversationTimeline({ provider, providerSessionId, event, reason = null, detail = null, tabId = null, onlyIfEmpty = false }) {
  if (!provider || !providerSessionId) return null;
  return serializeStorageMutation(async () => {
    const stored = await chrome.storage.local.get({ [CONVERSATION_TIMELINE_KEY]: {} });
    const timelines = stored[CONVERSATION_TIMELINE_KEY] && typeof stored[CONVERSATION_TIMELINE_KEY] === "object"
      ? stored[CONVERSATION_TIMELINE_KEY]
      : {};
    const key = sessionKey(provider, providerSessionId);
    if (onlyIfEmpty && Array.isArray(timelines[key]) && timelines[key].length) return null;
    const entry = {
      at: new Date().toISOString(),
      event,
      reason,
      detail,
      tab_id: tabId,
    };
    const next = {
      ...timelines,
      [key]: [entry, ...(Array.isArray(timelines[key]) ? timelines[key] : [])].slice(0, CONVERSATION_TIMELINE_EVENT_LIMIT),
    };
    const keys = Object.keys(next);
    if (keys.length > CONVERSATION_TIMELINE_CONVERSATION_LIMIT) {
      keys
        .sort((left, right) => Date.parse(next[left]?.[0]?.at || "") - Date.parse(next[right]?.[0]?.at || ""))
        .slice(0, keys.length - CONVERSATION_TIMELINE_CONVERSATION_LIMIT)
        .forEach((oldKey) => delete next[oldKey]);
    }
    await chrome.storage.local.set({ [CONVERSATION_TIMELINE_KEY]: next });
    return entry;
  });
}

function badgeForState(state) {
  if (cachedQueueLength > 0) {
    return { text: cachedQueueLength > 99 ? "99+" : String(cachedQueueLength), color: "#9a5b00" };
  }
  if (!state.online) return { text: "off", color: "#9b2c2c" };
  const archiveState = state.archive_state?.state;
  if (archiveState === "failed" || state.error) return { text: "err", color: "#ad2f2f" };
  if (["spooled_only", "ingest_pending", "stale"].includes(archiveState)) return { text: "…", color: "#9a5b00" };
  if (archiveState === "missing") return { text: "on", color: "#325d8f" };
  if (archiveState === "archived" || state.captured) return { text: "ok", color: "#14764e" };
  if (state.capture_mode === "dom_degraded") return { text: "dom", color: "#8a5a00" };
  return { text: "on", color: "#325d8f" };
}

async function setState(state) {
  const nextState = { ...state, updated_at: new Date().toISOString() };
  await chrome.storage.local.set({ polylogueState: nextState });
  const badge = badgeForState(nextState);
  await chrome.action.setBadgeText({ text: badge.text });
  await chrome.action.setBadgeBackgroundColor({ color: badge.color });
}

async function setStateForTab(tabId, state, expectedTabUrl = null) {
  if (!tabId || !chrome.tabs?.query) return setState(state);
  const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!activeTab || activeTab.id !== tabId) return null;
  const activeUrl = activeTab.url || activeTab.pendingUrl || "";
  if (expectedTabUrl && activeUrl && activeUrl !== expectedTabUrl) return null;
  const expectedProvider = state.provider || null;
  const expectedSessionId = state.provider_session_id || null;
  const activeProvider = archiveProviderForUrl(activeUrl);
  const activeSessionId = conversationIdForUrl(activeUrl);
  if (
    expectedProvider
    && expectedSessionId
    && activeSessionId
    && (
      activeProvider !== expectedProvider
      || activeSessionId !== expectedSessionId
    )
  ) return null;
  return setState(state);
}

function buildReceiverRequestId() {
  const random = Math.random().toString(36).slice(2, 10);
  return `polylogue-ext-${Date.now().toString(36)}-${random}`;
}

async function requestHeaders({ hasBody = false, requestId = "" } = {}) {
  const settings = await receiverSettings();
  const headers = {};
  if (hasBody) headers["Content-Type"] = "application/json";
  if (settings.authToken) headers.Authorization = `Bearer ${settings.authToken}`;
  if (requestId) headers["X-Request-ID"] = requestId;
  return headers;
}

async function postJson(path, payload, serializedBody = null, timeoutMs = null, requireReceiverRequestId = false) {
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  await appendDebugLog({ stage: "receiver_request", method: "POST", path, request_id: requestId, has_body: true });
  const controller = timeoutMs ? new AbortController() : null;
  const timeout = timeoutMs ? globalThis.setTimeout(() => controller.abort("receiver_request_timeout"), timeoutMs) : 0;
  try {
    const response = await fetch(`${settings.baseUrl}${path}`, {
      method: "POST",
      headers: await requestHeaders({ hasBody: true, requestId }),
      body: serializedBody || JSON.stringify(payload),
      signal: controller?.signal,
    });
    const acknowledgedRequestId = response.headers.get("X-Request-ID");
    const receiverRequestId = acknowledgedRequestId || requestId;
    const body = await response.json().catch(() => ({}));
    await appendDebugLog({
      stage: "receiver_response",
      method: "POST",
      path,
      request_id: requestId,
      receiver_request_id: receiverRequestId,
      ok: response.ok,
      status: response.status,
      provider: body.provider || payload?.session?.provider || null,
      provider_session_id: body.provider_session_id || payload?.session?.provider_session_id || null,
      archive_state: body.state || null,
      artifact_ref: body.artifact_ref || null,
    });
    if (!response.ok) {
      const error = new Error(body.error || `HTTP ${response.status}`);
      error.receiverRequestId = receiverRequestId;
      error.status = response.status;
      throw error;
    }
    if (requireReceiverRequestId && !acknowledgedRequestId) {
      const error = new Error("receiver_contract_incompatible:missing_receiver_request_id");
      error.receiverRequestId = null;
      error.status = response.status;
      throw error;
    }
    return { ...body, receiver_request_id: receiverRequestId };
  } catch (error) {
    await appendDebugLog({
      stage: "receiver_error",
      method: "POST",
      path,
      request_id: requestId,
      receiver_request_id: error.receiverRequestId || null,
      error: String(error.message || error),
    });
    throw error;
  } finally {
    if (timeout) globalThis.clearTimeout(timeout);
  }
}

async function getJson(path, timeoutMs = null) {
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  await appendDebugLog({ stage: "receiver_request", method: "GET", path, request_id: requestId });
  const controller = timeoutMs ? new AbortController() : null;
  const timeout = timeoutMs ? globalThis.setTimeout(() => controller.abort("receiver_request_timeout"), timeoutMs) : 0;
  try {
    const response = await fetch(`${settings.baseUrl}${path}`, {
      headers: await requestHeaders({ requestId }),
      signal: controller?.signal,
    });
    const receiverRequestId = response.headers.get("X-Request-ID") || requestId;
    const body = await response.json().catch(() => ({}));
    await appendDebugLog({
      stage: "receiver_response",
      method: "GET",
      path,
      request_id: requestId,
      receiver_request_id: receiverRequestId,
      ok: response.ok,
      status: response.status,
      provider: body.provider || null,
      provider_session_id: body.provider_session_id || null,
      archive_state: body.state || null,
    });
    if (!response.ok) {
      const error = new Error(body.error || `HTTP ${response.status}`);
      error.receiverRequestId = receiverRequestId;
      error.status = response.status;
      throw error;
    }
    return { ...body, receiver_request_id: receiverRequestId };
  } catch (error) {
    await appendDebugLog({
      stage: "receiver_error",
      method: "GET",
      path,
      request_id: requestId,
      receiver_request_id: error.receiverRequestId || null,
      error: String(error.message || error),
    });
    throw error;
  } finally {
    if (timeout) globalThis.clearTimeout(timeout);
  }
}

async function backfillReceiverPreflight() {
  let capability;
  try {
    capability = await getJson("/v1/browser-captures/capabilities", PROVIDER_REQUEST_TIMEOUT_MS);
  } catch (error) {
    if (error?.status === 404) throw new Error("receiver_contract_incompatible:capability_endpoint_missing");
    throw error;
  }
  const fields = capability?.durable_ack_fields;
  if (!Array.isArray(fields) || DURABLE_RECEIVER_ACK_FIELDS.some((field) => !fields.includes(field))) {
    throw new Error("receiver_contract_incompatible:durable_ack_fields_missing");
  }
  return capability;
}

async function refreshReceiverState() {
  try {
    const status = await getJson("/v1/status");
    await setState({
      online: true,
      captured: false,
      status,
      last_receiver_request_id: status.receiver_request_id || null,
    });
  } catch (error) {
    await setState({
      online: false,
      captured: false,
      error: String(error.message || error),
    });
  }
}

async function ensureCaptureScripts(tab) {
  const plan = injectionPlanForUrl(tab?.url || tab?.pendingUrl || "");
  if (!tab?.id || !plan.length || !chrome.scripting?.executeScript) return false;
  for (const step of plan) {
    const details = { target: { tabId: tab.id }, files: step.files };
    if (step.world) details.world = step.world;
    await chrome.scripting.executeScript(details);
  }
  return true;
}

function providerForUrl(url) {
  try {
    const hostname = new URL(url || "").hostname;
    if (hostname === "chatgpt.com") return "chatgpt";
    if (hostname === "claude.ai") return "claude-ai";
  } catch {
    return null;
  }
  return null;
}

function providerRequestFromUrl(urlValue) {
  const url = new URL(urlValue);
  if (url.hostname === "chatgpt.com") {
    if (url.pathname === "/backend-api/conversations") {
      const archived = url.searchParams.get("is_archived");
      const starred = url.searchParams.get("is_starred");
      if (!["true", "false"].includes(archived) || !["true", "false"].includes(starred)) {
        throw new Error("backfill_provider_inventory_flags_not_allowed");
      }
      return { provider: "chatgpt", operation: "inventory", params: {
        offset: Number.parseInt(url.searchParams.get("offset") || "0", 10),
        limit: Number.parseInt(url.searchParams.get("limit") || "28", 10),
        archived: archived === "true",
        starred: starred === "true",
      } };
    }
    const conversation = url.pathname.match(/^\/backend-api\/conversation\/([A-Za-z0-9_-]+)$/);
    if (conversation) return { provider: "chatgpt", operation: "conversation", params: { nativeId: decodeURIComponent(conversation[1]) } };
  }
  if (url.hostname === "claude.ai") {
    if (url.pathname === "/api/organizations") return { provider: "claude-ai", operation: "organizations", params: {} };
    const inventory = url.pathname.match(/^\/api\/organizations\/([0-9a-f-]{36})\/chat_conversations$/i);
    if (inventory) return { provider: "claude-ai", operation: "inventory", params: {
      organizationId: inventory[1],
      offset: Number.parseInt(url.searchParams.get("offset") || "0", 10),
      limit: Number.parseInt(url.searchParams.get("limit") || "100", 10),
    } };
    const conversation = url.pathname.match(/^\/api\/organizations\/([0-9a-f-]{36})\/chat_conversations\/([A-Za-z0-9_-]+)$/i);
    if (conversation) return { provider: "claude-ai", operation: "conversation", params: {
      organizationId: conversation[1],
      nativeId: decodeURIComponent(conversation[2]),
    } };
  }
  throw new Error("backfill_provider_url_not_allowed");
}

async function waitForProviderTab(tabId, provider) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    const tab = await chrome.tabs.get(tabId);
    if (providerForUrl(tab?.url || tab?.pendingUrl) === provider && tab?.status === "complete") return tab;
    await new Promise((resolve) => globalThis.setTimeout(resolve, 250));
  }
  throw new Error("backfill_provider_tab_load_timeout");
}

async function providerTab(provider) {
  const tabs = await chrome.tabs.query({});
  const existing = tabs.find((tab) => providerForUrl(tab.url || tab.pendingUrl) === provider);
  if (existing) return { tab: existing, owned: false, cleanupAlarm: null };
  const url = provider === "chatgpt" ? "https://chatgpt.com/" : "https://claude.ai/";
  const created = await chrome.tabs.create({ url, active: false });
  if (!created?.id) throw new Error("backfill_provider_tab_create_failed");
  const cleanupAlarm = `${BACKFILL_TRANSPORT_CLEANUP_PREFIX}:${provider}:${created.id}`;
  await chrome.alarms.create(cleanupAlarm, {
    when: Date.now() + BACKFILL_TRANSPORT_TAB_TTL_MS,
  });
  try {
    const ready = created.status === "complete" ? created : await waitForProviderTab(created.id, provider);
    return { tab: ready, owned: true, cleanupAlarm };
  } catch (error) {
    await chrome.tabs.remove(created.id).catch(() => undefined);
    await chrome.alarms.clear(cleanupAlarm);
    throw error;
  }
}

function pageContextResponse(response) {
  const body = typeof response?.body === "string" ? response.body : "";
  return {
    ok: Boolean(response?.ok),
    status: Number(response?.status || 0),
    polyloguePageContext: true,
    polylogueAuthReason: response?.authReason || null,
    headers: { get: (name) => {
      const normalized = name.toLowerCase();
      if (normalized === "content-type") return response?.contentType || "";
      if (normalized === "retry-after") return response?.retryAfter || null;
      return null;
    } },
    async json() { return JSON.parse(body); },
  };
}

function scriptingResultTooLarge(error) {
  const message = String(error?.message || error).toLowerCase();
  return /(?:result|response|message|script).{0,100}(?:too large|exceed(?:s|ed)?|maximum).{0,100}(?:size|limit|length)/.test(message);
}

async function providerPageFetch(url, options = {}) {
  if (options.method && options.method !== "GET") throw new Error("backfill_provider_method_not_allowed");
  const request = providerRequestFromUrl(url);
  request.maxResponseBytes = 32 * 1024 * 1024;
  const transport = await providerTab(request.provider);
  let result;
  try {
    const executions = await withTimeout(
      chrome.scripting.executeScript({
        target: { tabId: transport.tab.id },
        world: "MAIN",
        func: executeProviderPageRequest,
        args: [request],
      }),
      BACKFILL_PAGE_REQUEST_TIMEOUT_MS,
      "backfill_page_request",
    );
    result = executions?.[0]?.result;
  } catch (error) {
    if (transport.owned) {
      await chrome.tabs.remove(transport.tab.id).catch(() => undefined);
      if (transport.cleanupAlarm) await chrome.alarms.clear(transport.cleanupAlarm);
    }
    if (scriptingResultTooLarge(error)) {
      throw new Error("backfill_bridge_projection_too_large:observed_bytes=unavailable;limit_bytes=25165824");
    }
    throw error;
  }
  if (!result?.ok) {
    const error = String(result?.error || "backfill_page_request_failed");
    if (error.includes("auth_context") || error.includes("selected_organization")) {
      return pageContextResponse({ ok: false, status: 401, contentType: "application/json", authReason: error, body: JSON.stringify({ error }) });
    }
    throw new Error(error);
  }
  return pageContextResponse(result.response);
}

async function cleanupBackfillTransportTab(alarmName) {
  const parts = alarmName.split(":");
  const provider = parts[1];
  const tabId = Number.parseInt(parts[2] || "", 10);
  if (!provider || !Number.isInteger(tabId)) return;
  try {
    const tab = await chrome.tabs.get(tabId);
    if (providerForUrl(tab?.url || tab?.pendingUrl) === provider) await chrome.tabs.remove(tabId);
  } catch {
    // The operator or browser already closed the inactive transport tab.
  }
}

async function captureTab(tab, reason = "background", expectedConversation = null) {
  if (expectedConversation && tab?.id && chrome.tabs?.get) {
    const currentTab = await chrome.tabs.get(tab.id);
    const currentUrl = currentTab?.url || currentTab?.pendingUrl || "";
    if (
      !currentTab
      || currentUrl !== expectedConversation.url
      || archiveProviderForUrl(currentUrl) !== expectedConversation.provider
      || conversationIdForUrl(currentUrl) !== expectedConversation.providerSessionId
    ) return { ok: false, skipped: true, reason: "tab_navigation_changed" };
    tab = currentTab;
  }
  if (!tab?.id || !injectionPlanForUrl(tab.url || tab.pendingUrl || "").length) return null;
  const now = Date.now();
  const lastCaptureAt = recentBackgroundCaptures.get(tab.id) || 0;
  if (
    reason !== "extension_installed_or_updated"
    && !reason.startsWith("launch_job_")
    && now - lastCaptureAt < BACKGROUND_CAPTURE_MIN_INTERVAL_MS
  ) {
    return { ok: false, skipped: true, reason: "background_capture_throttled" };
  }
  recentBackgroundCaptures.set(tab.id, now);
  await ensureCaptureScripts(tab);
  try {
    const captureMessage = {
      type: "polylogue.capturePage",
      reason,
    };
    const resultWithTimeout = await withTimeout(
      chrome.tabs.sendMessage(tab.id, captureMessage),
      CAPTURE_MESSAGE_TIMEOUT_MS,
      "capture_message",
    );
    if (resultWithTimeout?.ok) {
      const envelopeSession = resultWithTimeout.envelope?.session || {};
      const provider = resultWithTimeout.captureResult?.provider || envelopeSession.provider;
      const providerSessionId = resultWithTimeout.captureResult?.provider_session_id || envelopeSession.provider_session_id;
      const pageProvider = archiveProviderForUrl(tab.url || tab.pendingUrl || "") || provider;
      const pageSessionId = conversationIdForUrl(tab.url || tab.pendingUrl || "") || providerSessionId;
      await updateSessionLedger({
        provider,
        providerSessionId,
        patch: {
          reason,
          tab_id: tab.id,
          tab_url: tab.url || tab.pendingUrl || null,
          capture_mode: envelopeSession.provider_meta?.capture_fidelity || null,
          turn_count: Array.isArray(envelopeSession.turns) ? envelopeSession.turns.length : null,
          attachment_count: Array.isArray(envelopeSession.turns)
            ? envelopeSession.turns.reduce((count, turn) => count + (Array.isArray(turn.attachments) ? turn.attachments.length : 0), 0)
            : null,
          archive_state: resultWithTimeout.archiveState || null,
          receiver_request_id: resultWithTimeout.captureResult?.receiver_request_id || resultWithTimeout.archiveState?.receiver_request_id || null,
          last_error: null,
        },
      });
      await appendCaptureLog({
        ok: true,
        reason,
        provider: pageProvider,
        provider_session_id: pageSessionId,
        tab_id: tab.id,
        archive_state: resultWithTimeout.archiveState?.state || null,
        receiver_request_id: resultWithTimeout.captureResult?.receiver_request_id || resultWithTimeout.archiveState?.receiver_request_id || null,
      });
      await setStateForTab(tab?.id || null, {
        online: true,
        captured: true,
        active_page_state: "conversation",
        active_tab_id: tab.id,
        passive_reason: reason,
        last_capture: resultWithTimeout.captureResult || resultWithTimeout,
        archive_state: resultWithTimeout.archiveState || null,
        provider: pageProvider,
        provider_session_id: pageSessionId,
        capture_mode: envelopeSession.provider_meta?.capture_fidelity || null,
        asset_acquisition: envelopeSession.provider_meta?.asset_acquisition || null,
        turn_count: Array.isArray(envelopeSession.turns) ? envelopeSession.turns.length : null,
        last_receiver_request_id:
          resultWithTimeout.captureResult?.receiver_request_id || resultWithTimeout.archiveState?.receiver_request_id || null
      }, tab.url || tab.pendingUrl || null);
      await appendDebugLog({
        stage: "capture_result",
        ok: true,
        reason,
        provider,
        provider_session_id: providerSessionId,
        capture_mode: envelopeSession.provider_meta?.capture_fidelity || null,
        archive_state: resultWithTimeout.archiveState?.state || null,
        receiver_request_id: resultWithTimeout.captureResult?.receiver_request_id || resultWithTimeout.archiveState?.receiver_request_id || null,
      });
    } else if (!resultWithTimeout?.timelineRecorded) {
      const provider = archiveProviderForUrl(tab.url || tab.pendingUrl || "");
      const providerSessionId = conversationIdForUrl(tab.url || tab.pendingUrl || "");
      await appendConversationTimeline({
        provider,
        providerSessionId,
        event: "held_with_reason",
        reason,
        detail: "capture_not_confirmed",
        tabId: tab.id,
      });
    }
    return resultWithTimeout;
  } catch (error) {
    await appendCaptureLog({
      ok: false,
      reason,
      tab_id: tab.id,
      tab_url: tab.url || tab.pendingUrl || null,
      error: String(error.message || error),
    });
    await appendDebugLog({
      stage: "capture_result",
      ok: false,
      reason,
      tab_id: tab.id,
      error: String(error.message || error),
    });
    await appendConversationTimeline({
      provider: archiveProviderForUrl(tab.url || tab.pendingUrl || ""),
      providerSessionId: conversationIdForUrl(tab.url || tab.pendingUrl || ""),
      event: "held_with_reason",
      reason,
      detail: String(error.message || error),
      tabId: tab.id,
    });
    return { ok: false, error: String(error.message || error) };
  }
}

async function captureSupportedTabs(reason) {
  if (!chrome.tabs?.query) return;
  const tabs = await chrome.tabs.query({});
  await Promise.allSettled(tabs.map((tab) => captureTab(tab, reason)));
}

// ---- GPT-5.6 Sol Pro Chat launch queue ----------------------------------

async function launchSettings() {
  const stored = await chrome.storage.local.get({ launchEnabled: false });
  return { launchEnabled: Boolean(stored.launchEnabled) };
}

async function saveLaunchSettings(launchEnabled) {
  await chrome.storage.local.set({ launchEnabled: Boolean(launchEnabled) });
  if (launchEnabled) {
    await ensureLaunchAlarm();
    void pollLaunchJobs();
  } else {
    await chrome.alarms?.clear?.(LAUNCH_ALARM);
  }
  return launchSettings();
}

async function ensureLaunchAlarm() {
  await chrome.alarms?.create?.(LAUNCH_ALARM, { delayInMinutes: 0.1, periodInMinutes: 1 });
}

async function updateLaunchJob(jobId, ownerInstanceId, patch) {
  return postJson(`/v1/launch-jobs/${encodeURIComponent(jobId)}/events`, {
    owner_instance_id: ownerInstanceId,
    ...patch,
  });
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function loadLaunchAttachments(job) {
  const settings = await receiverSettings();
  const attachments = [];
  let total = 0;
  for (const item of job.attachments || []) {
    total += Number(item.size_bytes || 0);
    if (total > LAUNCH_MAX_EXTENSION_TRANSPORT_BYTES) {
      throw new Error(`protocol_attachment_transport_limit:${total}`);
    }
    const requestId = buildReceiverRequestId();
    const response = await fetch(
      `${settings.baseUrl}/v1/launch-jobs/${encodeURIComponent(job.job_id)}/attachments/${encodeURIComponent(item.attachment_id)}`,
      { headers: await requestHeaders({ requestId }) },
    );
    if (!response.ok) {
      const retryAfter = Number.parseInt(response.headers.get("Retry-After") || "", 10) || null;
      const error = new Error(`launch_attachment_http_${response.status}`);
      error.retryAfterSeconds = retryAfter;
      throw error;
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== item.size_bytes) throw new Error(`protocol_attachment_size_mismatch:${item.attachment_id}`);
    attachments.push({
      attachment_id: item.attachment_id,
      name: item.name,
      mime_type: item.mime_type,
      content_base64: bytesToBase64(bytes),
    });
  }
  return attachments;
}

async function waitForTabComplete(tabId, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const tab = await chrome.tabs.get(tabId);
    if (tab.status === "complete") return tab;
    await new Promise((resolve) => globalThis.setTimeout(resolve, 100));
  }
  throw new Error("protocol_chatgpt_tab_load_timeout");
}

function exactChatGptConversation(url) {
  try {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (parsed.protocol !== "https:" || parsed.hostname !== "chatgpt.com" || parts.length !== 2 || parts[0] !== "c") {
      return null;
    }
    return { conversation_id: parts[1], conversation_url: parsed.href };
  } catch {
    return null;
  }
}

async function monitoringTabForLaunch(job, ownerInstanceId) {
  const expected = exactChatGptConversation(job.conversation_url || "");
  if (!expected || (job.conversation_id && expected.conversation_id !== job.conversation_id)) {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "protocol_mismatch",
      phase: "monitoring",
      detail: "submitted launch has no exact recoverable ChatGPT conversation URL",
      tab_id: job.tab_id || null,
      conversation_id: job.conversation_id || null,
      conversation_url: job.conversation_url || null,
    });
    return null;
  }

  let currentTab = null;
  if (job.tab_id) {
    try {
      currentTab = await chrome.tabs.get(job.tab_id);
    } catch {
      currentTab = null;
    }
  }
  const currentConversation = exactChatGptConversation(currentTab?.url || "");
  if (currentTab?.id && currentConversation?.conversation_id === expected.conversation_id) {
    return currentTab;
  }

  const existingTabs = await chrome.tabs.query({});
  const existingTab = existingTabs.find((tab) =>
    exactChatGptConversation(tab.url || "")?.conversation_id === expected.conversation_id
  );
  if (existingTab?.id) {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "progress",
      phase: "monitoring_recovered",
      detail: "re-associated submitted launch with an existing background conversation tab",
      tab_id: existingTab.id,
      conversation_id: expected.conversation_id,
      conversation_url: expected.conversation_url,
    });
    return existingTab;
  }

  // The submission boundary is already behind us. Reopen only the known
  // conversation URL; never return to the composer or resubmit the prompt.
  const reopenedTab = await chrome.tabs.create({ url: expected.conversation_url, active: false });
  if (!reopenedTab?.id) throw new Error("protocol_background_tab_create_failed");
  const loadedTab = await waitForTabComplete(reopenedTab.id);
  await updateLaunchJob(job.job_id, ownerInstanceId, {
    outcome: "progress",
    phase: "monitoring_recovered",
    detail: "reopened submitted conversation in a background tab after the original tab disappeared",
    tab_id: loadedTab.id,
    conversation_id: expected.conversation_id,
    conversation_url: expected.conversation_url,
  });
  return loadedTab;
}

async function monitorSubmittedLaunch(job, ownerInstanceId) {
  let monitoringTab;
  try {
    monitoringTab = await monitoringTabForLaunch(job, ownerInstanceId);
  } catch (error) {
    const classified = classifyLaunchFailure(error);
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      ...classified,
      phase: "monitoring",
      tab_id: job.tab_id || null,
      conversation_id: job.conversation_id || null,
      conversation_url: job.conversation_url || null,
    });
    return;
  }
  if (!monitoringTab?.id) return;
  job = { ...job, tab_id: monitoringTab.id };
  let inspection;
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: job.tab_id },
      world: "MAIN",
      func: inspectChatGptLaunchPage,
    });
    inspection = result;
  } catch (error) {
    const classified = classifyLaunchFailure(error);
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      ...classified,
      phase: "monitoring",
      tab_id: job.tab_id,
      conversation_id: job.conversation_id,
      conversation_url: job.conversation_url,
    });
    return;
  }
  if (inspection?.soft_warning) {
    if (job.phase !== "provider_soft_warning") {
      await updateLaunchJob(job.job_id, ownerInstanceId, {
        outcome: "soft_warning",
        phase: "provider_soft_warning",
        detail: "provider conversation-access warning visible; submitted chat may still continue",
        tab_id: job.tab_id,
        conversation_id: inspection.conversation_id,
        conversation_url: inspection.conversation_url,
      });
    }
  }
  if (inspection?.rate_limited) {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "rate_limited",
      phase: "provider_rate_limit",
      detail: "provider rate limit visible in conversation",
      tab_id: job.tab_id,
      conversation_id: inspection.conversation_id,
      conversation_url: inspection.conversation_url,
    });
    return;
  }
  if (inspection?.safety_lock) {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "safety_locked",
      phase: "provider_safety_lock",
      detail: "provider safety lock visible in conversation",
      tab_id: job.tab_id,
      conversation_id: inspection.conversation_id,
      conversation_url: inspection.conversation_url,
    });
    return;
  }
  if (!inspection?.soft_warning && (inspection?.busy || inspection?.assistant_turns) && job.phase !== "provider_running") {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "progress",
      phase: "provider_running",
      detail: "provider accepted the submitted chat without a visible circuit",
      tab_id: job.tab_id,
      conversation_id: inspection.conversation_id,
      conversation_url: inspection.conversation_url,
    });
  }
  if (inspection?.busy || !inspection?.assistant_turns) return;
  if (!inspection.handoff_name) {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "protocol_mismatch",
      phase: "handoff_validation",
      detail: "assistant completed without required cohesive handoff archive",
      tab_id: job.tab_id,
      conversation_id: inspection.conversation_id,
      conversation_url: inspection.conversation_url,
    });
    return;
  }
  try {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "progress",
      phase: "handoff_visible_pending_acquisition",
      detail: "required cohesive handoff archive is visible; acquiring bytes for receiver validation",
      tab_id: job.tab_id,
      conversation_id: inspection.conversation_id,
      conversation_url: inspection.conversation_url,
      handoff_attachment_id: inspection.handoff_name,
    });
    const captured = await captureTab(monitoringTab, "launch_job_handoff");
    const handoff = captured?.envelope?.session?.attachments?.find(
      (attachment) => attachment.name === inspection.handoff_name && attachment.inline_base64,
    );
    if (!handoff) {
      await updateLaunchJob(job.job_id, ownerInstanceId, {
        outcome: "progress",
        phase: "handoff_asset_pending",
        detail: "handoff link is visible but authenticated capture has not acquired its bytes yet",
        tab_id: job.tab_id,
        conversation_id: inspection.conversation_id,
        conversation_url: inspection.conversation_url,
      });
      return;
    }
    // captureTab already sent these exact bytes through the ordinary capture
    // envelope used for every user- or queue-created conversation. The
    // receiver correlates that canonical artifact to the launch job; never
    // repost or store assistant output through a launch-only side channel.
    await appendCaptureLog({
      ok: true,
      reason: "sol_pro_handoff_captured",
      job_id: job.job_id,
      provider_attachment_id: handoff.provider_attachment_id || null,
      artifact_ref: captured?.captureResult?.artifact_ref || null,
    });
  } catch (error) {
    const classified = classifyLaunchFailure(error, error?.retryAfterSeconds || null);
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      ...classified,
      phase: "handoff_acquisition",
      tab_id: job.tab_id,
      conversation_id: inspection.conversation_id,
      conversation_url: inspection.conversation_url,
    });
  }
}

async function recoverSubmittingLaunch(job, ownerInstanceId) {
  if (!job.tab_id) {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "submission_unknown",
      phase: "unknown_submit_outcome",
      detail: "submit intent was recorded without a recoverable tab; operator inspection required",
    });
    return;
  }
  try {
    await chrome.tabs.get(job.tab_id);
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: job.tab_id },
      world: "MAIN",
      func: inspectChatGptLaunchPage,
    });
    if (!result?.conversation_id) return;
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "submitted",
      phase: "submitted_recovered",
      detail: "recovered the conversation after an unknown submit acknowledgement outcome",
      tab_id: job.tab_id,
      conversation_id: result.conversation_id,
      conversation_url: result.conversation_url,
    });
  } catch (error) {
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "submission_unknown",
      phase: "unknown_submit_outcome",
      detail: `submit intent cannot be reconciled automatically: ${String(error.message || error)}`,
      tab_id: job.tab_id,
    });
  }
}

async function dispatchLaunchJob(job, ownerInstanceId) {
  let tab = null;
  let pageExecutionStarted = false;
  try {
    const attachments = await loadLaunchAttachments(job);
    tab = await chrome.tabs.create({ url: "https://chatgpt.com/", active: false });
    if (!tab?.id) throw new Error("protocol_background_tab_create_failed");
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "progress",
      phase: "uploading",
      detail: "background Chat tab created without activation",
      tab_id: tab.id,
    });
    await waitForTabComplete(tab.id);
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "progress",
      phase: "submit_intent",
      detail: "durable intent recorded before the single submit boundary",
      tab_id: tab.id,
    });
    pageExecutionStarted = true;
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      world: "MAIN",
      func: executeChatGptLaunchInPage,
      args: [job, attachments],
    });
    if (!result?.ok) {
      const error = new Error(result?.detail || "protocol_launch_result_missing");
      error.submissionMayHaveOccurred = Boolean(result?.submission_may_have_occurred);
      throw error;
    }
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      outcome: "submitted",
      phase: "submitted",
      detail: "Chat / GPT-5.6 Sol / Pro preflight passed and frontend submitted",
      tab_id: tab.id,
      conversation_id: result.conversation_id,
      conversation_url: result.conversation_url,
    });
    await appendCaptureLog({
      ok: true,
      reason: "sol_pro_launch_submitted",
      job_id: job.job_id,
      conversation_id: result.conversation_id,
    });
  } catch (error) {
    const ambiguous = error?.submissionMayHaveOccurred === true
      || (pageExecutionStarted && error?.submissionMayHaveOccurred !== false);
    const classified = ambiguous
      ? {
        outcome: "submission_unknown",
        retry_after_seconds: null,
        detail: `submit execution channel ended without a durable acknowledgement: ${String(error.message || error)}`,
      }
      : classifyLaunchFailure(error, error?.retryAfterSeconds || null);
    await updateLaunchJob(job.job_id, ownerInstanceId, {
      ...classified,
      phase: "launch",
      tab_id: tab?.id || null,
    }).catch(() => undefined);
    await appendCaptureLog({
      ok: false,
      reason: "sol_pro_launch_failed",
      job_id: job.job_id,
      error: String(error.message || error),
    });
  }
}

async function pollLaunchJobsOnce() {
  const { launchEnabled } = await launchSettings();
  if (!launchEnabled) return { enabled: false, jobs: [] };
  const ownerInstanceId = await launchExecutorId();
  const status = await getJson("/v1/launch-jobs").catch(() => ({ jobs: [] }));
  for (const job of status.jobs || []) {
    if (job.status === "cancelled" && job.executor_instance_id === ownerInstanceId && job.tab_id) {
      await chrome.tabs.remove(job.tab_id).catch(() => undefined);
    } else if (
      job.status === "submitted" &&
      job.lease_owner === ownerInstanceId
    ) {
      await monitorSubmittedLaunch(job, ownerInstanceId);
    } else if (job.status === "submitting" && job.lease_owner === ownerInstanceId) {
      await recoverSubmittingLaunch(job, ownerInstanceId);
    }
  }
  const claim = await getJson(`/v1/launch-jobs?claim_by=${encodeURIComponent(ownerInstanceId)}`);
  const [job] = claim.jobs || [];
  if (job?.status === "submitted") {
    await monitorSubmittedLaunch(job, ownerInstanceId);
  } else if (job?.status === "submitting") {
    await recoverSubmittingLaunch(job, ownerInstanceId);
  } else if (job?.status === "leased") {
    await dispatchLaunchJob(job, ownerInstanceId);
  }
  return { enabled: true, jobs: status.jobs || [], claimed: job || null };
}

function pollLaunchJobs() {
  if (launchPollPromise) return launchPollPromise;
  const candidate = pollLaunchJobsOnce();
  const tracked = candidate.finally(() => {
    if (launchPollPromise === tracked) launchPollPromise = null;
  });
  launchPollPromise = tracked;
  return launchPollPromise;
}

// ---- Outbound posting (reverse channel) ---------------------------------
//
// Disabled by default. The local receiver only serves post commands when its
// own POLYLOGUE_BROWSER_POST_ENABLED=1 guard is set; the extension adds a second
// independent guard (`postingEnabled`, default false) so a misconfigured
// receiver still cannot drive the page without an explicit opt-in here.

async function postingSettings() {
  const stored = await chrome.storage.local.get({ postingEnabled: false });
  return { postingEnabled: Boolean(stored.postingEnabled) };
}

async function savePostingSettings(postingEnabled) {
  await chrome.storage.local.set({ postingEnabled: Boolean(postingEnabled) });
  return postingSettings();
}

function providerTokenForUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) return "chatgpt";
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) return "claude";
  } catch {
    return null;
  }
  return null;
}

function archiveProviderForUrl(url) {
  try {
    const parsed = new URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) return "chatgpt";
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) return "claude-ai";
    if (
      parsed.hostname === "grok.com" ||
      parsed.hostname.endsWith(".grok.com") ||
      parsed.hostname === "x.com" ||
      parsed.hostname.endsWith(".x.com") ||
      parsed.hostname === "twitter.com" ||
      parsed.hostname.endsWith(".twitter.com")
    ) {
      return "grok";
    }
  } catch {
    return null;
  }
  return null;
}

function conversationIdForUrl(url) {
  try {
    const parsed = new URL(url || "");
    const parts = parsed.pathname.split("/").filter(Boolean);
    const provider = archiveProviderForUrl(url);
    if (provider === "chatgpt") {
      const marker = parts.indexOf("c");
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
      if (parsed.searchParams.get("temporary-chat") === "true") return null;
      return null;
    }
    if (provider === "claude-ai") {
      return parts[0] === "chat" && parts[1] ? parts[1] : null;
    }
    if (provider === "grok") {
      const pathId = parts.find((part, index) => parts[index - 1] === "chat" || parts[index - 1] === "grok");
      if (pathId) return pathId;
      const queryId = parsed.searchParams.get("conversation") || parsed.searchParams.get("conversationId");
      if (queryId) return queryId;
      if (!(parts[0] === "i" && parts[1] === "grok")) return null;
      let hash = 0x811c9dc5;
      for (const char of `${parsed.origin}${parsed.pathname}${parsed.search}`) {
        hash ^= char.charCodeAt(0);
        hash = Math.imul(hash, 0x01000193);
      }
      return `dom:${(hash >>> 0).toString(16).padStart(8, "0")}`;
    }
  } catch {
    return null;
  }
  return null;
}

async function refreshActiveTabArchiveState(tab, reason = "tab_state") {
  const url = tab?.url || tab?.pendingUrl || "";
  const provider = archiveProviderForUrl(url);
  const providerSessionId = conversationIdForUrl(url);
  const throttleKey = `${tab?.id || "active"}:${provider || "unsupported"}:${providerSessionId || "none"}`;
  const now = Date.now();
  const lastCheckedAt = recentActiveTabStateChecks.get(throttleKey) || 0;
  if (now - lastCheckedAt < ACTIVE_TAB_STATE_MIN_INTERVAL_MS) return;
  recentActiveTabStateChecks.set(throttleKey, now);

  try {
    if (provider && providerSessionId) {
      const query = new URLSearchParams({ provider, provider_session_id: providerSessionId });
      const state = await getJson(`/v1/archive-state?${query.toString()}`);
      await appendConversationTimeline({
        provider,
        providerSessionId,
        event: "first_seen",
        reason,
        detail: "archive_state_checked",
        tabId: tab?.id || null,
        onlyIfEmpty: true,
      });
      await setStateForTab(tab?.id || null, {
        online: true,
        captured: Boolean(state.captured),
        archive_state: state,
        provider,
        provider_session_id: providerSessionId,
        active_page_state: "conversation",
        active_tab_id: tab?.id || null,
        passive_reason: reason,
        last_receiver_request_id: state.receiver_request_id || null,
      }, url);
      await updateSessionLedger({
        provider,
        providerSessionId,
        patch: {
          archive_state: state,
          tab_id: tab?.id || null,
          tab_url: url || null,
          last_error: null,
        },
      });
      if (state.state === "missing") {
        await appendConversationTimeline({
          provider,
          providerSessionId,
          event: "detected_new",
          reason,
          detail: "archive_state_missing",
          tabId: tab?.id || null,
        });
        const captureResult = await captureTab(tab, "auto_capture_missing", {
          provider,
          providerSessionId,
          url,
        });
        if (!captureResult || captureResult.skipped) {
          await appendConversationTimeline({
            provider,
            providerSessionId,
            event: "held_with_reason",
            reason: "auto_capture_missing",
            detail: captureResult?.reason || "capture_not_available",
            tabId: tab?.id || null,
          });
        }
      }
      return;
    }

    const status = await getJson("/v1/status");
    await setStateForTab(tab?.id || null, {
      online: true,
      captured: false,
      status,
      provider,
      provider_session_id: null,
      active_page_state: provider ? "supported_no_session" : "unsupported",
      active_tab_id: tab?.id || null,
      passive_reason: reason,
      last_receiver_request_id: status.receiver_request_id || null,
    }, url);
  } catch (error) {
    await appendConversationTimeline({
      provider,
      providerSessionId,
      event: "held_with_reason",
      reason,
      detail: "archive_state_check_failed",
      tabId: tab?.id || null,
    });
    await setStateForTab(tab?.id || null, {
      online: false,
      captured: false,
      provider,
      provider_session_id: providerSessionId,
      active_page_state: provider ? "receiver_error" : "unsupported",
      active_tab_id: tab?.id || null,
      passive_reason: reason,
      error: String(error.message || error),
      last_receiver_request_id: error.receiverRequestId || null,
    }, url);
  }
}

async function refreshCurrentActiveTab(reason = "active_tab") {
  if (!chrome.tabs?.query) {
    await refreshReceiverState();
    return;
  }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    await refreshReceiverState();
    return;
  }
  await refreshActiveTabArchiveState(tab, reason);
}

async function ackPostCommand(commandId, result) {
  try {
    await postJson(`/v1/post-commands/${encodeURIComponent(commandId)}/ack`, result);
    pendingPostCommandAcks.delete(commandId);
    inFlightPostCommands.delete(commandId);
    return true;
  } catch (error) {
    await appendCaptureLog({ ok: false, reason: "post_ack", command_id: commandId, error: String(error.message || error) });
    pendingPostCommandAcks.set(commandId, result);
    return false;
  }
}

async function retryPendingPostCommandAcks() {
  for (const [commandId, result] of [...pendingPostCommandAcks.entries()]) {
    await ackPostCommand(commandId, result);
  }
}

async function findTabForCommand(command) {
  if (!chrome.tabs?.query) return null;
  const tabs = await chrome.tabs.query({});
  const provider = command.provider;
  const target = command.target || {};
  const wantNew = !target.conversation_id || target.conversation_id === "new";
  let fallback = null;
  for (const tab of tabs) {
    const url = tab.url || tab.pendingUrl || "";
    if (providerTokenForUrl(url) !== provider) continue;
    if (wantNew) {
      if (conversationIdForUrl(url)) continue;
      if (tab.active) return tab;
      fallback = fallback || tab;
      continue;
    }
    if (conversationIdForUrl(url) === target.conversation_id) return tab;
  }
  return wantNew ? fallback : null;
}

async function dispatchPostCommand(command) {
  if (!command || !command.command_id || inFlightPostCommands.has(command.command_id)) return;
  inFlightPostCommands.add(command.command_id);
  let terminalAckRecorded = false;
  try {
    const tab = await findTabForCommand(command);
    if (!tab?.id) {
      terminalAckRecorded = await ackPostCommand(command.command_id, { status: "failed", detail: "no_matching_tab" });
      return;
    }
    await ensureCaptureScripts(tab);
    let result;
    try {
      result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.postReply", command });
    } catch (error) {
      terminalAckRecorded = await ackPostCommand(command.command_id, {
        status: "failed",
        detail: String(error.message || error),
        observed_url: tab.url || null,
      });
      return;
    }
    terminalAckRecorded = await ackPostCommand(command.command_id, {
      status: result?.status === "submitted" ? "submitted" : "failed",
      detail: result?.detail || null,
      observed_url: result?.observed_url || tab.url || null,
    });
  } finally {
    if (terminalAckRecorded) inFlightPostCommands.delete(command.command_id);
  }
}

async function pollPostCommands() {
  const { postingEnabled } = await postingSettings();
  if (!postingEnabled) return;
  await retryPendingPostCommandAcks();
  for (const provider of ["chatgpt", "claude"]) {
    let body;
    try {
      body = await getJson(`/v1/post-commands?provider=${provider}`);
    } catch {
      continue;
    }
    if (!body?.post_enabled || !Array.isArray(body.commands)) continue;
    for (const command of body.commands) {
      await dispatchPostCommand(command);
    }
  }
}

async function startPostPolling() {
  const { postingEnabled } = await postingSettings();
  if (!postingEnabled) {
    stopPostPolling();
    return;
  }
  if (postPollTimer) return;
  postPollTimer = globalThis.setInterval(() => {
    void pollPostCommands();
  }, POST_POLL_INTERVAL_MS);
  void pollPostCommands();
}

function stopPostPolling() {
  if (postPollTimer) {
    globalThis.clearInterval(postPollTimer);
    postPollTimer = 0;
  }
}

void startPostPolling();
void loadCaptureQueueIntoCache();
void launchSettings().then(({ launchEnabled }) => {
  if (launchEnabled) {
    void ensureLaunchAlarm();
    void pollLaunchJobs();
  }
});

chrome.alarms?.onAlarm?.addListener((alarm) => {
  if (alarm?.name === LAUNCH_ALARM) {
    void pollLaunchJobs();
    return;
  }
  if (alarm?.name?.startsWith(`${BACKFILL_TRANSPORT_CLEANUP_PREFIX}:`)) {
    void cleanupBackfillTransportTab(alarm.name);
    return;
  }
  if (alarm?.name === CAPTURE_RETRY_ALARM) {
    void drainCaptureQueue("alarm");
  }
  if (alarm?.name?.startsWith(`${BACKFILL_ALARM}:`)) {
    const jobId = alarm.name.slice(BACKFILL_ALARM.length + 1);
    void backfillCoordinator().then((coordinator) => coordinator.wake(jobId));
  }
});

chrome.runtime.onInstalled?.addListener(() => {
  void refreshCurrentActiveTab("extension_installed");
});

chrome.runtime.onStartup?.addListener(() => {
  void refreshCurrentActiveTab("browser_startup");
  void backfillCoordinator().then((coordinator) => coordinator.wake());
  void pollLaunchJobs();
});

chrome.tabs?.onActivated?.addListener((activeInfo) => {
  void (async () => {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    await refreshActiveTabArchiveState(tab, "tab_activated");
  })();
});

chrome.tabs?.onUpdated?.addListener((tabId, changeInfo, tab) => {
  if (changeInfo?.status !== "complete" && !changeInfo?.url) return;
  void (async () => {
    await refreshActiveTabArchiveState(tab?.id ? tab : await chrome.tabs.get(tabId), "tab_updated");
  })();
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    if (message.type === "polylogue.configureReceiver") {
      const settings = await saveReceiverSettings(message.receiverBaseUrl || DEFAULT_RECEIVER, message.receiverAuthToken || "");
      sendResponse({ ok: true, receiverBaseUrl: settings.baseUrl, authConfigured: Boolean(settings.authToken) });
      return;
    }
    if (message.type === "polylogue.backfill.start") {
      const coordinator = await backfillCoordinator();
      const job = await coordinator.start({
        provider: message.provider,
        cutoff: message.cutoff,
        policy: message.policy || {},
        provider_options: message.provider_options || {},
      });
      void coordinator.wake(job.id);
      sendResponse({ ok: true, job });
      return;
    }
    if (message.type === "polylogue.backfill.control") {
      const coordinator = await backfillCoordinator();
      sendResponse({ ok: true, job: await coordinator.control(message.job_id, message.action) });
      return;
    }
    if (message.type === "polylogue.backfill.status") {
      const coordinator = await backfillCoordinator();
      sendResponse({ ok: true, jobs: await coordinator.listStatus() });
      return;
    }
    if (message.type === "polylogue.backfill.export") {
      const coordinator = await backfillCoordinator();
      sendResponse({ ok: true, ledger: await coordinator.exportLedger(message.job_id) });
      return;
    }
    if (message.type === "polylogue.capture") {
      const envelope = await withExtensionInstanceAttribution(message.envelope);
      const summary = envelopeSessionSummary(envelope);
      let result;
      try {
        result = await postJson("/v1/browser-captures", envelope);
      } catch (error) {
        if (isRetryableCaptureError(error)) {
          const queued = await enqueueCaptureForRetry({ envelope, reason: message.reason, error, tab: sender.tab });
          for (const evicted of queued.accepted ? queued.evicted : []) {
            const evictedSummary = envelopeSessionSummary(evicted.envelope);
            await appendConversationTimeline({
              provider: evictedSummary.provider,
              providerSessionId: evictedSummary.providerSessionId,
              event: "held_with_reason",
              reason: evicted.reason,
              detail: queued.accepted ? "capture_queue_evicted" : "capture_queue_entry_over_budget",
              tabId: evicted.tab_id || null,
            });
          }
          await appendConversationTimeline({
            provider: summary.provider,
            providerSessionId: summary.providerSessionId,
            event: "held_with_reason",
            reason: message.reason || "content_script_capture",
            detail: queued.accepted ? "capture_queued_for_retry" : "capture_queue_entry_over_budget",
            tabId: sender.tab?.id || null,
          });
          await setStateForTab(sender.tab?.id || null, {
            online: false,
            captured: false,
            provider: summary.provider,
            provider_session_id: summary.providerSessionId,
            error: String(error.message || error),
            last_receiver_request_id: error.receiverRequestId || null,
          }, sender.tab?.url || sender.tab?.pendingUrl || null);
          sendResponse({
            ok: false,
            queued: queued.accepted,
            error: String(error.message || error),
            receiver_request_id: error.receiverRequestId || null,
          });
          return;
        }
        await updateSessionLedger({
          provider: summary.provider,
          providerSessionId: summary.providerSessionId,
          patch: { last_error: String(error.message || error) },
        });
        await appendConversationTimeline({
          provider: summary.provider,
          providerSessionId: summary.providerSessionId,
          event: "held_with_reason",
          reason: message.reason || "content_script_capture",
          detail: "capture_rejected",
          tabId: sender.tab?.id || null,
        });
        throw error;
      }
      const archiveState = { state: result.state || "spooled_only" };
      await updateSessionLedger({
        provider: summary.provider || result.provider,
        providerSessionId: summary.providerSessionId || result.provider_session_id,
        patch: {
          capture_mode: summary.captureMode,
          asset_acquisition: summary.assetAcquisition,
          turn_count: summary.turnCount,
          attachment_count: summary.attachmentCount,
          receiver_request_id: result.receiver_request_id || null,
          artifact_ref: result.artifact_ref || null,
          extension_instance_id: result.capture_instance_id || null,
          deduplicated: Boolean(result.deduplicated),
          archive_state: archiveState,
          last_error: null,
        },
      });
      await appendCaptureLog({
        ok: true,
        reason: message.reason || "content_script_capture",
        provider: summary.provider || result.provider,
        provider_session_id: summary.providerSessionId || result.provider_session_id,
        capture_mode: summary.captureMode,
        receiver_request_id: result.receiver_request_id || null,
        artifact_ref: result.artifact_ref || null,
      });
      await appendConversationTimeline({
        provider: summary.provider || result.provider,
        providerSessionId: summary.providerSessionId || result.provider_session_id,
        event: "captured",
        reason: message.reason || "content_script_capture",
        detail: archiveState.state,
        tabId: sender.tab?.id || null,
      });
      await setStateForTab(sender.tab?.id || null, {
        online: true,
        captured: true,
        last_capture: result,
        archive_state: archiveState,
        provider: summary.provider || result.provider,
        provider_session_id: summary.providerSessionId || result.provider_session_id,
        capture_mode: summary.captureMode,
        asset_acquisition: summary.assetAcquisition,
        turn_count: summary.turnCount,
        attachment_count: summary.attachmentCount,
        extension_instance_id: result.capture_instance_id || null,
        deduplicated: Boolean(result.deduplicated),
        last_receiver_request_id: result.receiver_request_id || null
      }, sender.tab?.url || sender.tab?.pendingUrl || null);
      // Receiver just proved reachable — flush anything queued from earlier
      // outages before returning this capture's result.
      void drainCaptureQueue("post_success");
      sendResponse({ ok: true, ...result });
      return;
    }
    if (message.type === "polylogue.getCaptureQueue") {
      const queue = await getCaptureQueue();
      sendResponse({
        ok: true,
        dropped_count: queue.dropped_count,
        entries: queue.entries.map((entry) => ({
          id: entry.id,
          reason: entry.reason,
          enqueued_at: entry.enqueued_at,
          attempts: entry.attempts,
          next_attempt_at: entry.next_attempt_at,
          last_error: entry.last_error,
          provider: entry.envelope?.session?.provider || null,
          provider_session_id: entry.envelope?.session?.provider_session_id || null,
        })),
      });
      return;
    }
    if (message.type === "polylogue.retryCaptureQueue") {
      const outcome = await drainCaptureQueue("manual");
      sendResponse({ ok: true, ...outcome });
      return;
    }
    if (message.type === "polylogue.checkReceiverHealth") {
      const health = await checkReceiverHealth();
      sendResponse(health);
      return;
    }
    if (message.type === "polylogue.archiveState") {
      const query = new URLSearchParams({
        provider: message.provider,
        provider_session_id: message.provider_session_id
      });
      const state = await getJson(`/v1/archive-state?${query.toString()}`);
      const stored = await chrome.storage.local.get({ polylogueState: null });
      const previous = stored.polylogueState;
      const sameSession = previous?.provider === message.provider
        && previous?.provider_session_id === message.provider_session_id;
      const preservedCaptureMetadata = sameSession ? {
        last_capture: previous.last_capture,
        capture_mode: previous.capture_mode,
        asset_acquisition: previous.asset_acquisition,
        turn_count: previous.turn_count,
        attachment_count: previous.attachment_count,
      } : {};
      await setStateForTab(sender.tab?.id || null, {
        ...preservedCaptureMetadata,
        online: true,
        captured: Boolean(state.captured),
        archive_state: state,
        provider: message.provider,
        provider_session_id: message.provider_session_id,
        last_receiver_request_id: state.receiver_request_id || null
      }, sender.tab?.url || sender.tab?.pendingUrl || null);
      await updateSessionLedger({
        provider: message.provider,
        providerSessionId: message.provider_session_id,
        patch: {
          archive_state: state,
          tab_id: sender.tab?.id || null,
          tab_url: sender.tab?.url || null,
          last_error: null,
        },
      });
      sendResponse(state);
      return;
    }
    if (message.type === "polylogue.status") {
      await refreshCurrentActiveTab(message.reason || "status");
      const stored = await chrome.storage.local.get({ polylogueState: null });
      const state = stored.polylogueState || {};
      if (!state.online) {
        sendResponse({
          ok: false,
          error: state.error || "receiver_unavailable",
          receiver_request_id: state.last_receiver_request_id || null,
        });
        return;
      }
      sendResponse(state.archive_state || state.status || state);
      return;
    }
    if (message.type === "polylogue.capturePageFailed") {
      const tab = sender.tab || (message.tab_url ? { id: message.tab_id || null, url: message.tab_url } : null);
      const url = tab?.url || tab?.pendingUrl || "";
      const provider = archiveProviderForUrl(url);
      const providerSessionId = conversationIdForUrl(url);
      await appendConversationTimeline({
        provider,
        providerSessionId,
        event: "held_with_reason",
        reason: "popup_capture",
        detail: "content_capture_failed",
        tabId: tab?.id || null,
      });
      await setStateForTab(tab?.id || null, {
        online: true,
        captured: false,
        provider,
        provider_session_id: providerSessionId,
        active_page_state: providerSessionId ? "conversation" : "supported_no_session",
        error: message.error || "capture_page_failed",
      }, url || null);
      sendResponse({ ok: false });
      return;
    }
    if (message.type === "polylogue.captureSupportedTabs") {
      await captureSupportedTabs(message.reason || "popup_sync_open_tabs");
      sendResponse({ ok: true });
      return;
    }
    if (message.type === "polylogue.configurePosting") {
      const settings = await savePostingSettings(message.postingEnabled);
      await startPostPolling();
      sendResponse({ ok: true, postingEnabled: settings.postingEnabled });
      return;
    }
    if (message.type === "polylogue.launch.configure") {
      const settings = await saveLaunchSettings(message.launchEnabled);
      sendResponse({ ok: true, ...settings });
      return;
    }
    if (message.type === "polylogue.launch.status") {
      const [settings, status, ownerInstanceId] = await Promise.all([
        launchSettings(),
        getJson("/v1/launch-jobs"),
        launchExecutorId(),
      ]);
      sendResponse({ ok: true, ...settings, ownerInstanceId, jobs: status.jobs || [] });
      return;
    }
    if (message.type === "polylogue.launch.control") {
      const result = await postJson(`/v1/launch-jobs/${encodeURIComponent(message.job_id)}/control`, {
        action: message.action,
        inspection_receipt: message.inspection_receipt || null,
        conversation_id: message.conversation_id || null,
        conversation_url: message.conversation_url || null,
      });
      if (["resume", "retry", "launch_now", "confirm_no_conversation", "confirm_existing_conversation"].includes(message.action)) {
        void pollLaunchJobs();
      }
      sendResponse({ ok: true, job: result.job });
      return;
    }
    if (message.type === "polylogue.launch.poll") {
      sendResponse({ ok: true, ...(await pollLaunchJobs()) });
      return;
    }
    if (message.type === "polylogue.pollPostCommands") {
      await pollPostCommands();
      sendResponse({ ok: true });
      return;
    }
  })().catch(async (error) => {
    await appendCaptureLog({
      ok: false,
      reason: message.type || "runtime_message",
      error: String(error.message || error),
      receiver_request_id: error.receiverRequestId || null,
    });
    await appendDebugLog({
      stage: "runtime_message_error",
      message_type: message.type || "runtime_message",
      receiver_request_id: error.receiverRequestId || null,
      error: String(error.message || error),
    });
    const captureSummary = message.type === "polylogue.capture"
      ? envelopeSessionSummary(message.envelope)
      : null;
    await setStateForTab(sender.tab?.id || null, {
      online: false,
      captured: false,
      error: String(error.message || error),
      provider: captureSummary?.provider || null,
      provider_session_id: captureSummary?.providerSessionId || null,
      last_receiver_request_id: error.receiverRequestId || null,
    }, sender.tab?.url || sender.tab?.pendingUrl || null);
    sendResponse({
      ok: false,
      error: String(error.message || error),
      receiver_request_id: error.receiverRequestId || null
    });
  });
  return true;
});
