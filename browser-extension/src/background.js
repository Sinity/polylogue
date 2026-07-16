import { BackfillCoordinator } from "./backfill/coordinator.js";
import { BACKFILL_ALARM, DURABLE_RECEIVER_ACK_FIELDS, PROVIDER_REQUEST_TIMEOUT_MS } from "./backfill/models.js";
import { providerAdapters } from "./backfill/providers.js";
import { executeProviderPageRequest } from "./backfill/page_transport.js";
import { IndexedDbBackfillStore } from "./backfill/storage.js";
import {
  classifyBrowserActionFailure,
  executeChatGptBrowserActionInPage,
} from "./actions/chatgpt.js";
import {
  chatGptCaptureNeedsFollowUp,
  claimDueFreshness,
  completeFreshnessClaim,
  failureRetryDelayMs,
  normalizeFreshnessQueue,
  runningPollDelayMs,
  scheduleFreshnessHint,
} from "./capture/freshness.js";

const DEFAULT_RECEIVER = "http://127.0.0.1:8765";
const EXTENSION_CONTRACT_EPOCH = "canonical-capture-mission-control-v1";
const RECEIVER_API_SCHEMA = "polylogue-browser-capture/v1";
const RECEIVER_PAIRING_KEY = "polylogueReceiverPairing";
const RECEIVER_HEALTH_TIMEOUT_MS = 5000;
const RECEIVER_TRUST_CACHE_MS = 10000;
const AMBIENT_SETTINGS_KEY = "polylogueAmbientSettings";
const BACKGROUND_CAPTURE_MIN_INTERVAL_MS = 30000;
const ACTIVE_TAB_STATE_MIN_INTERVAL_MS = 4000;
const CAPTURE_LOG_LIMIT = 80;
const DEBUG_LOG_LIMIT = 160;
const CONVERSATION_TIMELINE_KEY = "polylogueConversationTimeline";
const CONVERSATION_TIMELINE_EVENT_LIMIT = 24;
const BACKFILL_RECOVERY_CHECKPOINT_KEY = "polylogueBackfillRecoveryCheckpoint";
const BACKFILL_WORKER_EPOCH = globalThis.crypto?.randomUUID?.() || `worker-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const CONVERSATION_TIMELINE_CONVERSATION_LIMIT = 80;
const BROWSER_ACTION_ALARM = "polylogueBrowserActionWake";
const CAPTURE_FRESHNESS_ALARM = "polylogueCaptureFreshnessWake";
const CAPTURE_FRESHNESS_SWEEP_ALARM = "polylogueCaptureFreshnessSweep";
const CAPTURE_FRESHNESS_QUEUE_KEY = "polylogueCaptureFreshnessQueue";
const CAPTURE_FRESHNESS_LEASE_MS = 2 * 60 * 1000;
const CAPTURE_FRESHNESS_SWEEP_MINUTES = 15;
const CAPTURE_FRESHNESS_SWEEP_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const BROWSER_ACTION_MAX_EXTENSION_TRANSPORT_BYTES = 16 * 1024 * 1024;
const CAPTURE_MESSAGE_TIMEOUT_MS = 35000;
const BACKFILL_PAGE_REQUEST_TIMEOUT_MS = 58000;
const BACKFILL_TRANSPORT_TAB_TTL_MS = 5 * 60 * 1000;
const BACKFILL_TRANSPORT_CLEANUP_PREFIX = "polylogueBackfillTransportCleanup";
const PROVIDER_TRANSPORT_SESSION_PREFIX = "polylogueProviderTransportTab";
const recentBackgroundCaptures = new Map();
const recentActiveTabStateChecks = new Map();
let backfillCoordinatorPromise = null;
let extensionInstanceIdPromise = null;
let browserActionExecutorIdPromise = null;
const providerTransportPromises = new Map();
const providerTransportOperations = new Map();
let browserActionPollPromise = null;
let captureFreshnessPollPromise = null;
let storageMutationQueue = Promise.resolve();
let captureQueueMutationQueue = Promise.resolve();
let trustedReceiverHealthCache = null;

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

function browserActionExecutorId() {
  if (!browserActionExecutorIdPromise) {
    const key = "polylogueBrowserActionExecutorId";
    const candidate = (async () => {
      const session = chrome.storage.session;
      if (!session?.get || !session?.set) return `browser-action-${BACKFILL_WORKER_EPOCH}`;
      const stored = await session.get({ [key]: "" });
      if (stored[key]) return stored[key];
      const created = `browser-action-${globalThis.crypto?.randomUUID?.() || BACKFILL_WORKER_EPOCH}`;
      await session.set({ [key]: created });
      return created;
    })();
    browserActionExecutorIdPromise = candidate;
    void candidate.catch(() => {
      if (browserActionExecutorIdPromise === candidate) browserActionExecutorIdPromise = null;
    });
  }
  return browserActionExecutorIdPromise;
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
        captureOverride: async ({ provider, nativeId, response }) => {
          if (provider !== "chatgpt") return null;
          const nativePayload = await response.json();
          const captured = await captureProviderConversation(
            provider,
            nativeId,
            "backfill_exact_capture",
            { deferReceiver: true, nativePayload },
          );
          return captured.envelope;
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
        {
          files: [
            "src/common.js",
            "src/operator_status.js",
            "src/content/message_layer.js",
            "src/content/ambient_surface.js",
            "src/content/chatgpt.js",
          ],
        },
      ];
    }
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) {
      return [
        { files: ["src/content/claude_bridge.js"], world: "MAIN" },
        {
          files: [
            "src/common.js",
            "src/operator_status.js",
            "src/content/message_layer.js",
            "src/content/ambient_surface.js",
            "src/content/claude.js",
          ],
        },
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
  trustedReceiverHealthCache = null;
  await chrome.storage.local.set({
    receiverAuthToken: String(receiverAuthToken || ""),
    receiverBaseUrl: String(receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, "") || DEFAULT_RECEIVER,
  });
  return receiverSettings();
}

async function storedReceiverPairing() {
  const stored = await chrome.storage.local.get({ [RECEIVER_PAIRING_KEY]: null });
  const pairing = stored[RECEIVER_PAIRING_KEY];
  return pairing && typeof pairing === "object" ? pairing : null;
}

async function persistReceiverPairing(pairing) {
  await chrome.storage.local.set({ [RECEIVER_PAIRING_KEY]: pairing });
  return pairing;
}

async function markReceiverPairingUnavailable(detail) {
  trustedReceiverHealthCache = null;
  const pairing = await storedReceiverPairing();
  if (!pairing) return null;
  return persistReceiverPairing({
    ...pairing,
    state: "offline",
    last_error: String(detail || "receiver_unavailable"),
    checked_at: new Date().toISOString(),
  });
}

async function observeReceiverIdentity(status, endpoint) {
  const now = new Date().toISOString();
  const prior = await storedReceiverPairing();
  const receiverId = typeof status?.receiver_id === "string" ? status.receiver_id : null;
  const apiSchema = typeof status?.api_schema === "string" ? status.api_schema : null;

  if (!receiverId || !apiSchema) {
    if (prior?.receiver_id) {
      trustedReceiverHealthCache = null;
      return persistReceiverPairing({
        ...prior,
        state: "mismatch",
        observed_endpoint: endpoint,
        observed_receiver_id: receiverId,
        observed_api_schema: apiSchema || "legacy",
        checked_at: now,
        last_error: "receiver_pairing_metadata_missing",
      });
    }
    return persistReceiverPairing({
      state: "legacy",
      endpoint,
      last_seen_at: now,
      checked_at: now,
      last_error: null,
    });
  }

  if (apiSchema !== RECEIVER_API_SCHEMA) {
    trustedReceiverHealthCache = null;
    return persistReceiverPairing({
      ...(prior || {}),
      state: "mismatch",
      endpoint: prior?.endpoint || endpoint,
      observed_endpoint: endpoint,
      observed_receiver_id: receiverId,
      observed_api_schema: apiSchema,
      checked_at: now,
      last_error: "receiver_api_schema_mismatch",
    });
  }

  if (prior?.receiver_id && prior.receiver_id !== receiverId) {
    trustedReceiverHealthCache = null;
    return persistReceiverPairing({
      ...prior,
      state: "mismatch",
      observed_endpoint: endpoint,
      observed_receiver_id: receiverId,
      observed_api_schema: apiSchema,
      checked_at: now,
      last_error: "receiver_identity_mismatch",
    });
  }

  return persistReceiverPairing({
    receiver_id: receiverId,
    api_schema: apiSchema,
    endpoint,
    paired_at: prior?.paired_at || now,
    last_seen_at: now,
    checked_at: now,
    state: "online",
    last_error: null,
  });
}

async function clearReceiverPairing() {
  trustedReceiverHealthCache = null;
  await chrome.storage.local.remove?.(RECEIVER_PAIRING_KEY);
  // Test doubles and older browser shims may not expose remove(). Setting null
  // is equivalent for all readers and keeps reset bounded to this one key.
  if (!chrome.storage.local.remove) await chrome.storage.local.set({ [RECEIVER_PAIRING_KEY]: null });
}

function hostnameForUrl(url) {
  try {
    return new URL(url || "").hostname;
  } catch {
    return "";
  }
}

async function ambientSettings(hostname = "") {
  const stored = await chrome.storage.local.get({
    [AMBIENT_SETTINGS_KEY]: { enabled: true, disabled_sites: {} },
  });
  const raw = stored[AMBIENT_SETTINGS_KEY] && typeof stored[AMBIENT_SETTINGS_KEY] === "object"
    ? stored[AMBIENT_SETTINGS_KEY]
    : {};
  const disabledSites = raw.disabled_sites && typeof raw.disabled_sites === "object" ? raw.disabled_sites : {};
  return {
    enabled: raw.enabled !== false,
    disabled_sites: disabledSites,
    site: hostname || null,
    site_enabled: hostname ? disabledSites[hostname] !== true : true,
  };
}

async function saveAmbientSettings({ enabled = null, hostname = "", siteEnabled = null } = {}) {
  const current = await ambientSettings(hostname);
  const disabledSites = { ...current.disabled_sites };
  if (hostname && siteEnabled !== null) {
    if (siteEnabled) delete disabledSites[hostname];
    else disabledSites[hostname] = true;
  }
  const next = {
    enabled: enabled === null ? current.enabled : Boolean(enabled),
    disabled_sites: disabledSites,
  };
  await chrome.storage.local.set({ [AMBIENT_SETTINGS_KEY]: next });
  return ambientSettings(hostname);
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

async function probeReceiverStatus(baseUrl, authToken = "") {
  const requestId = buildReceiverRequestId();
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort("receiver_health_timeout"), RECEIVER_HEALTH_TIMEOUT_MS);
  await appendDebugLog({ stage: "receiver_request", method: "GET", path: "/v1/status", endpoint: baseUrl, request_id: requestId });
  try {
    const headers = { "X-Request-ID": requestId };
    if (authToken) headers.Authorization = `Bearer ${authToken}`;
    const response = await fetch(`${baseUrl}/v1/status`, { headers, signal: controller.signal });
    const body = await response.json().catch(() => null);
    const receiverRequestId = response.headers?.get?.("X-Request-ID") || requestId;
    await appendDebugLog({
      stage: "receiver_response",
      method: "GET",
      path: "/v1/status",
      endpoint: baseUrl,
      request_id: requestId,
      receiver_request_id: receiverRequestId,
      ok: response.ok,
      status: response.status,
      receiver_id: body?.receiver_id || null,
      api_schema: body?.api_schema || null,
    });
    return { response, body, receiverRequestId };
  } catch (error) {
    await appendDebugLog({
      stage: "receiver_error",
      method: "GET",
      path: "/v1/status",
      endpoint: baseUrl,
      request_id: requestId,
      error: String(error.message || error),
    });
    throw error;
  } finally {
    globalThis.clearTimeout(timeout);
  }
}

async function checkReceiverHealth({ allowCanonicalRecovery = true } = {}) {
  const settings = await receiverSettings();
  const pairingBefore = await storedReceiverPairing();

  async function classifyProbe(endpoint, probe, recoveredFrom = null) {
    const body = probe.body;
    if (!body || typeof body !== "object") {
      return {
        ok: false,
        status: "unreachable",
        detail: "non_json_response",
        endpoint,
        receiver_request_id: probe.receiverRequestId || null,
        pairing: pairingBefore,
      };
    }
    if (body.error === "unauthorized" || probe.response?.status === 401) {
      const pairing = pairingBefore
        ? await persistReceiverPairing({
          ...pairingBefore,
          state: "unauthorized",
          checked_at: new Date().toISOString(),
          last_error: "unauthorized",
        })
        : null;
      return {
        ok: true,
        status: "unauthorized",
        detail: "unauthorized",
        endpoint,
        receiver_request_id: probe.receiverRequestId || null,
        pairing,
      };
    }
    if (body.ok === true && probe.response?.ok !== false) {
      const pairing = await observeReceiverIdentity(body, endpoint);
      if (pairing?.state === "mismatch") {
        return {
          ok: true,
          status: "pairing_mismatch",
          detail: pairing.last_error || "receiver_pairing_mismatch",
          endpoint,
          receiver_status: body,
          receiver_request_id: probe.receiverRequestId || null,
          pairing,
        };
      }
      const result = {
        ok: true,
        status: recoveredFrom ? "recovered" : "ok",
        detail: null,
        endpoint,
        recovered_from: recoveredFrom,
        receiver_status: body,
        receiver_request_id: probe.receiverRequestId || null,
        pairing,
      };
      if (pairing?.receiver_id && pairing.state === "online") {
        trustedReceiverHealthCache = {
          checkedAt: Date.now(),
          endpoint,
          receiverId: pairing.receiver_id,
          apiSchema: pairing.api_schema,
          health: result,
        };
      }
      return result;
    }
    return {
      ok: true,
      status: "error",
      detail: body.error || `http_${probe.response?.status || 0}`,
      endpoint,
      receiver_status: body,
      receiver_request_id: probe.receiverRequestId || null,
      pairing: pairingBefore,
    };
  }

  let primaryFailure = null;
  try {
    const primary = await probeReceiverStatus(settings.baseUrl, settings.authToken);
    const classified = await classifyProbe(settings.baseUrl, primary);
    if (classified.status !== "unreachable") return classified;
    primaryFailure = classified.detail || "receiver_unavailable";
  } catch (error) {
    primaryFailure = String(error.message || error);
  }

  if (
    allowCanonicalRecovery
    && settings.baseUrl !== DEFAULT_RECEIVER
    && pairingBefore?.receiver_id
  ) {
    try {
      const canonical = await probeReceiverStatus(DEFAULT_RECEIVER, settings.authToken);
      const body = canonical.body;
      if (
        body?.ok === true
        && canonical.response?.ok !== false
        && body.receiver_id === pairingBefore.receiver_id
        && body.api_schema === RECEIVER_API_SCHEMA
      ) {
        await chrome.storage.local.set({ receiverBaseUrl: DEFAULT_RECEIVER });
        return classifyProbe(DEFAULT_RECEIVER, canonical, settings.baseUrl);
      }
    } catch {
      // Recovery is intentionally bounded to one canonical endpoint. The
      // original failure remains the operator-facing result.
    }
  }

  const pairing = await markReceiverPairingUnavailable(primaryFailure);
  return {
    ok: false,
    status: "unreachable",
    detail: primaryFailure,
    endpoint: settings.baseUrl,
    pairing,
  };
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

async function appendConversationTimeline({ provider, providerSessionId, event, reason = null, detail = null, tabId = null, onlyIfEmpty = false, dedupeWindowMs = 0 }) {
  if (!provider || !providerSessionId) return null;
  return serializeStorageMutation(async () => {
    const stored = await chrome.storage.local.get({ [CONVERSATION_TIMELINE_KEY]: {} });
    const timelines = stored[CONVERSATION_TIMELINE_KEY] && typeof stored[CONVERSATION_TIMELINE_KEY] === "object"
      ? stored[CONVERSATION_TIMELINE_KEY]
      : {};
    const key = sessionKey(provider, providerSessionId);
    const existing = Array.isArray(timelines[key]) ? timelines[key] : [];
    if (onlyIfEmpty && existing.length) return null;
    if (dedupeWindowMs > 0) {
      const latest = existing[0];
      const latestAt = Date.parse(latest?.at || "");
      if (
        latest?.event === event
        && latest?.reason === reason
        && latest?.detail === detail
        && Number.isFinite(latestAt)
        && Date.now() - latestAt < dedupeWindowMs
      ) return null;
    }
    const entry = {
      at: new Date().toISOString(),
      event,
      reason,
      detail,
      tab_id: tabId,
    };
    const next = {
      ...timelines,
      [key]: [entry, ...existing].slice(0, CONVERSATION_TIMELINE_EVENT_LIMIT),
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
  headers["X-Polylogue-Extension-Contract"] = EXTENSION_CONTRACT_EPOCH;
  return headers;
}

async function ensureTrustedReceiver() {
  const pairing = await storedReceiverPairing();
  if (!pairing?.receiver_id) return null;
  const settings = await receiverSettings();
  const cached = trustedReceiverHealthCache;
  if (
    cached
    && Date.now() - cached.checkedAt < RECEIVER_TRUST_CACHE_MS
    && cached.endpoint === settings.baseUrl
    && cached.receiverId === pairing.receiver_id
    && cached.apiSchema === pairing.api_schema
  ) return cached.health;

  const health = await checkReceiverHealth({ allowCanonicalRecovery: true });
  if (!["ok", "recovered"].includes(health.status)) {
    const code = health.status === "pairing_mismatch"
      ? "receiver_pairing_mismatch"
      : health.status === "unauthorized"
        ? "unauthorized"
        : "receiver_unavailable";
    const error = new Error(code === "receiver_unavailable" ? health.detail || code : code);
    error.code = code;
    error.status = code === "receiver_pairing_mismatch" ? 409 : code === "unauthorized" ? 401 : 503;
    error.receiverRequestId = health.receiver_request_id || null;
    error.receiverHealth = health;
    throw error;
  }
  return health;
}

async function postJson(path, payload, serializedBody = null, timeoutMs = null, requireReceiverRequestId = false) {
  await ensureTrustedReceiver();
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
  await ensureTrustedReceiver();
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
  const health = await checkReceiverHealth();
  const online = ["ok", "recovered"].includes(health.status);
  await setState({
    online,
    captured: false,
    status: health.receiver_status || null,
    receiver_pairing: health.pairing || null,
    receiver_health: health,
    error: health.status === "unauthorized"
      ? "unauthorized"
      : health.status === "pairing_mismatch"
        ? "receiver_pairing_mismatch"
        : online
          ? null
          : health.detail || "receiver_unavailable",
    last_receiver_request_id: health.receiver_request_id || health.receiver_status?.receiver_request_id || null,
  });
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

function providerTransportSessionKey(provider) {
  return `${PROVIDER_TRANSPORT_SESSION_PREFIX}:${provider}`;
}

async function forgetProviderTransport(provider, tabId = null) {
  const key = providerTransportSessionKey(provider);
  const stored = await chrome.storage.session.get({ [key]: null });
  if (tabId === null || stored[key] === tabId) await chrome.storage.session.remove(key);
}

async function acquireProviderTab(provider) {
  const key = providerTransportSessionKey(provider);
  const stored = await chrome.storage.session.get({ [key]: null });
  const storedTabId = stored[key];
  if (Number.isInteger(storedTabId)) {
    const existing = await chrome.tabs.get(storedTabId).catch(() => null);
    if (
      existing
      && existing.active !== true
      && providerForUrl(existing.url || existing.pendingUrl) === provider
    ) {
      return {
        tab: existing,
        owned: true,
        cleanupAlarm: `${BACKFILL_TRANSPORT_CLEANUP_PREFIX}:${provider}:${storedTabId}`,
      };
    }
    await forgetProviderTransport(provider, storedTabId);
  }
  const url = provider === "chatgpt" ? "https://chatgpt.com/" : "https://claude.ai/";
  const created = await chrome.tabs.create({ url, active: false });
  if (!created?.id) throw new Error("backfill_provider_tab_create_failed");
  await chrome.storage.session.set({ [key]: created.id });
  const cleanupAlarm = `${BACKFILL_TRANSPORT_CLEANUP_PREFIX}:${provider}:${created.id}`;
  await chrome.alarms.create(cleanupAlarm, { when: Date.now() + BACKFILL_TRANSPORT_TAB_TTL_MS });
  try {
    const ready = created.status === "complete" ? created : await waitForProviderTab(created.id, provider);
    return { tab: ready, owned: true, cleanupAlarm };
  } catch (error) {
    await chrome.tabs.remove(created.id).catch(() => undefined);
    await chrome.alarms.clear(cleanupAlarm);
    await forgetProviderTransport(provider, created.id);
    throw error;
  }
}

function providerTab(provider) {
  const inFlight = providerTransportPromises.get(provider);
  if (inFlight) return inFlight;
  const candidate = acquireProviderTab(provider);
  const tracked = candidate.finally(() => {
    if (providerTransportPromises.get(provider) === tracked) providerTransportPromises.delete(provider);
  });
  providerTransportPromises.set(provider, tracked);
  return tracked;
}

function withProviderTransportOperation(provider, operation) {
  const prior = providerTransportOperations.get(provider) || Promise.resolve();
  const result = prior.catch(() => undefined).then(operation);
  const tracked = result.finally(() => {
    if (providerTransportOperations.get(provider) === tracked) providerTransportOperations.delete(provider);
  });
  providerTransportOperations.set(provider, tracked);
  return tracked;
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
  return withProviderTransportOperation(request.provider, async () => {
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
        await forgetProviderTransport(request.provider, transport.tab.id);
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
  });
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
  await chrome.alarms.clear(alarmName);
  await forgetProviderTransport(provider, tabId);
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
  const conversationUrl = tab?.url || tab?.pendingUrl || "";
  if (
    !tab?.id
    || !archiveProviderForUrl(conversationUrl)
    || !conversationIdForUrl(conversationUrl)
    || !injectionPlanForUrl(conversationUrl).length
  ) return null;
  const now = Date.now();
  const lastCaptureAt = recentBackgroundCaptures.get(tab.id) || 0;
  if (
    reason !== "extension_installed_or_updated"
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

async function captureProviderConversation(
  provider,
  providerSessionId,
  reason,
  { deferReceiver = false, nativePayload = null } = {},
) {
  if (provider !== "chatgpt") throw new Error(`exact_provider_capture_unsupported:${provider}`);
  if (!/^[A-Za-z0-9_-]{1,256}$/.test(String(providerSessionId || ""))) {
    throw new Error("exact_provider_capture_invalid_session_id");
  }
  return withProviderTransportOperation(provider, async () => {
    const transport = await providerTab(provider);
    await ensureCaptureScripts(transport.tab);
    const result = await withTimeout(
      chrome.tabs.sendMessage(transport.tab.id, {
        type: "polylogue.capturePage",
        reason,
        providerSessionId,
        deferReceiver,
        nativePayload,
      }),
      CAPTURE_MESSAGE_TIMEOUT_MS,
      "capture_message",
    );
    if (!result?.ok) throw new Error(result?.error || "exact_provider_capture_failed");
    const acceptedId = result.envelope?.session?.provider_session_id;
    if (acceptedId !== providerSessionId) throw new Error("exact_provider_capture_identity_mismatch");
    return result;
  });
}

async function storedCaptureFreshnessQueue() {
  const stored = await chrome.storage.local.get({ [CAPTURE_FRESHNESS_QUEUE_KEY]: null });
  return normalizeFreshnessQueue(stored[CAPTURE_FRESHNESS_QUEUE_KEY]);
}

async function persistCaptureFreshnessQueue(queue) {
  await chrome.storage.local.set({ [CAPTURE_FRESHNESS_QUEUE_KEY]: queue });
  return queue;
}

async function scheduleCaptureFreshness({
  provider,
  nativeId,
  reason,
  delayMs = 0,
  providerUpdatedAt = null,
}) {
  if (provider !== "chatgpt" || !/^[A-Za-z0-9_-]{1,256}$/.test(String(nativeId || ""))) {
    return { scheduled: false, reason: "unsupported_or_invalid_identity" };
  }
  const queue = await serializeStorageMutation(async () => {
    const current = await storedCaptureFreshnessQueue();
    return persistCaptureFreshnessQueue(scheduleFreshnessHint(current, {
      provider,
      nativeId,
      reason,
      nowMs: Date.now(),
      delayMs,
      providerUpdatedAt,
    }));
  });
  const entry = queue.entries[`${provider}:${nativeId}`];
  await chrome.alarms?.create?.(CAPTURE_FRESHNESS_ALARM, {
    when: Math.max(Date.now() + 1_000, entry.next_attempt_at_ms),
  });
  return { scheduled: true, entry };
}

async function scheduleNextCaptureFreshnessWake(queueValue = null) {
  const queue = queueValue || await storedCaptureFreshnessQueue();
  const deadlines = Object.values(queue.entries).map((entry) => (
    entry.lease_owner ? entry.lease_expires_at_ms : entry.next_attempt_at_ms
  )).filter(Number.isFinite);
  if (!deadlines.length) {
    await chrome.alarms?.clear?.(CAPTURE_FRESHNESS_ALARM);
    return;
  }
  await chrome.alarms?.create?.(CAPTURE_FRESHNESS_ALARM, {
    when: Math.max(Date.now() + 1_000, Math.min(...deadlines)),
  });
}

async function processCaptureFreshnessQueueOnce() {
  const owner = await extensionInstanceId();
  const now = Date.now();
  const { queue, claim } = await serializeStorageMutation(async () => {
    const current = await storedCaptureFreshnessQueue();
    const claimed = claimDueFreshness(current, {
      nowMs: now,
      owner,
      leaseMs: CAPTURE_FRESHNESS_LEASE_MS,
    });
    if (claimed.claim) await persistCaptureFreshnessQueue(claimed.queue);
    return claimed;
  });
  if (!claim) {
    await scheduleNextCaptureFreshnessWake(queue);
    return { processed: 0, remaining: Object.keys(queue.entries).length };
  }

  let needsFollowUp = false;
  let retryDelayMs = 0;
  let failure = null;
  try {
    const result = await captureProviderConversation(
      claim.provider,
      claim.native_id,
      "freshness_convergence",
    );
    needsFollowUp = chatGptCaptureNeedsFollowUp(result.envelope);
    retryDelayMs = needsFollowUp ? runningPollDelayMs(claim.running_poll_count || 0) : 0;
    const receipt = result.captureResult || {};
    if (claim.provider_updated_at && receipt.content_hash) {
      const coordinator = await backfillCoordinator();
      await coordinator.store.putRevision({
        id: `${claim.provider}:${claim.native_id}`,
        provider: claim.provider,
        native_id: claim.native_id,
        provider_updated_at: claim.provider_updated_at,
        receiver_content_hash: receipt.content_hash,
        receiver_request_id: receipt.receiver_request_id || null,
        completed_at: new Date().toISOString(),
      });
    }
    await appendConversationTimeline({
      provider: claim.provider,
      providerSessionId: claim.native_id,
      event: needsFollowUp ? "detected_new" : "captured",
      reason: "freshness_convergence",
      detail: needsFollowUp ? "provider_still_running" : "provider_head_current",
    });
  } catch (error) {
    failure = String(error?.message || error);
    const classified = classifyBrowserActionFailure(error, error?.retryAfterSeconds || null);
    retryDelayMs = failureRetryDelayMs(
      claim.attempt_count || 0,
      classified.outcome,
      classified.retry_after_seconds,
    );
    await appendConversationTimeline({
      provider: claim.provider,
      providerSessionId: claim.native_id,
      event: "held_with_reason",
      reason: "freshness_convergence",
      detail: classified.outcome,
    });
  }

  const next = await serializeStorageMutation(async () => {
    const current = await storedCaptureFreshnessQueue();
    return persistCaptureFreshnessQueue(completeFreshnessClaim(current, claim, {
      nowMs: Date.now(),
      needsFollowUp,
      retryDelayMs,
      error: failure,
    }));
  });
  await scheduleNextCaptureFreshnessWake(next);
  return { processed: 1, remaining: Object.keys(next.entries).length, needsFollowUp, error: failure };
}

function processCaptureFreshnessQueue() {
  if (captureFreshnessPollPromise) return captureFreshnessPollPromise;
  const tracked = processCaptureFreshnessQueueOnce().finally(() => {
    if (captureFreshnessPollPromise === tracked) captureFreshnessPollPromise = null;
  });
  captureFreshnessPollPromise = tracked;
  return tracked;
}

async function runCaptureFreshnessSweep() {
  const now = Date.now();
  let queue = await storedCaptureFreshnessQueue();
  if (queue.sweep_not_before_ms > now) return { skipped: true, reason: "sweep_backoff" };
  const coordinator = await backfillCoordinator();
  const activeJobs = await coordinator.store.listJobs();
  if (activeJobs.some((job) => job.provider === "chatgpt" && job.status === "running")) {
    return { skipped: true, reason: "explicit_backfill_running" };
  }
  const partition = queue.sweep_partition % 4;
  try {
    const cutoff = new Date(now - CAPTURE_FRESHNESS_SWEEP_WINDOW_MS).toISOString();
    const adapter = coordinator.adapters.chatgpt;
    const result = await adapter.enumerate(`${partition}:0`, cutoff);
    if (result.classification !== "success") {
      const error = new Error(`provider_${result.classification}_http_${result.response?.status || 0}`);
      error.retryAfterSeconds = Number.parseInt(result.response?.headers?.get?.("Retry-After") || "", 10) || null;
      throw error;
    }
    queue = await serializeStorageMutation(async () => {
      const current = await storedCaptureFreshnessQueue();
      return persistCaptureFreshnessQueue({
        ...current,
        sweep_partition: (partition + 1) % 4,
        sweep_not_before_ms: 0,
        last_sweep_at: new Date(now).toISOString(),
        last_sweep_error: null,
      });
    });
    let scheduled = 0;
    for (const item of result.items) {
      const revision = item.updated_at
        ? await coordinator.store.getRevision("chatgpt", item.native_id)
        : null;
      if (item.updated_at && revision?.provider_updated_at === item.updated_at) continue;
      const outcome = await scheduleCaptureFreshness({
        provider: "chatgpt",
        nativeId: item.native_id,
        reason: "inventory_delta",
        delayMs: scheduled * 15_000,
        providerUpdatedAt: item.updated_at,
      });
      if (outcome.scheduled) scheduled += 1;
    }
    return { skipped: false, partition, observed: result.items.length, scheduled };
  } catch (error) {
    const classified = classifyBrowserActionFailure(error, error?.retryAfterSeconds || null);
    const retryDelay = failureRetryDelayMs(0, classified.outcome, classified.retry_after_seconds);
    await serializeStorageMutation(async () => {
      const current = await storedCaptureFreshnessQueue();
      await persistCaptureFreshnessQueue({
        ...current,
        sweep_not_before_ms: now + retryDelay,
        last_sweep_at: new Date(now).toISOString(),
        last_sweep_error: classified.outcome,
      });
    });
    return { skipped: true, reason: classified.outcome, error: String(error?.message || error) };
  }
}

async function ensureCaptureFreshnessAlarms() {
  await chrome.alarms?.create?.(CAPTURE_FRESHNESS_SWEEP_ALARM, {
    delayInMinutes: 1,
    periodInMinutes: CAPTURE_FRESHNESS_SWEEP_MINUTES,
  });
  await scheduleNextCaptureFreshnessWake();
}

async function captureSupportedTabs(reason) {
  if (!chrome.tabs?.query) return;
  const tabs = await chrome.tabs.query({});
  await Promise.allSettled(tabs.map((tab) => captureTab(tab, reason)));
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

// ---- Provider-neutral browser actions ----------------------------------

async function updateBrowserAction(actionId, ownerInstanceId, patch) {
  return postJson(`/v1/browser-actions/${encodeURIComponent(actionId)}/events`, {
    owner_instance_id: ownerInstanceId,
    ...patch,
  });
}

async function browserActionAttachmentBytes(action) {
  const settings = await receiverSettings();
  const attachments = [];
  let total = 0;
  for (const item of action.attachments || []) {
    total += Number(item.size_bytes || 0);
    if (total > BROWSER_ACTION_MAX_EXTENSION_TRANSPORT_BYTES) {
      throw new Error(`protocol_attachment_transport_limit:${total}`);
    }
    const requestId = buildReceiverRequestId();
    const response = await fetch(
      `${settings.baseUrl}/v1/browser-actions/${encodeURIComponent(action.action_id)}/attachments/${encodeURIComponent(item.attachment_id)}`,
      { headers: await requestHeaders({ requestId }) },
    );
    if (!response.ok) {
      const error = new Error(`browser_action_attachment_http_${response.status}`);
      error.retryAfterSeconds = Number.parseInt(response.headers.get("Retry-After") || "", 10) || null;
      throw error;
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== item.size_bytes) throw new Error(`protocol_attachment_size_mismatch:${item.attachment_id}`);
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
      .map((value) => value.toString(16).padStart(2, "0"))
      .join("");
    if (digest !== item.sha256) throw new Error(`protocol_attachment_hash_mismatch:${item.attachment_id}`);
    attachments.push({
      attachment_id: item.attachment_id,
      name: item.name,
      mime_type: item.mime_type,
      content_base64: bytesToBase64(bytes),
    });
  }
  return attachments;
}

function browserActionTargetUrl(action) {
  if (action.target?.conversation_url) return action.target.conversation_url;
  if (action.operation === "conversation.reply") {
    return `https://chatgpt.com/c/${encodeURIComponent(action.target.conversation_id)}`;
  }
  if (action.target?.project_ref) {
    const project = String(action.target.project_ref).replace(/^https:\/\/chatgpt\.com\/g\//, "").replace(/^\/+|\/+$/g, "");
    return `https://chatgpt.com/g/${project}/project?tab=chats`;
  }
  return "https://chatgpt.com/";
}

async function prepareBrowserActionTransport(action) {
  if (action.provider !== "chatgpt") throw new Error(`unsupported_browser_action_provider:${action.provider}`);
  const transport = await providerTab("chatgpt");
  const targetUrl = browserActionTargetUrl(action);
  const currentUrl = transport.tab.url || transport.tab.pendingUrl || "";
  if (currentUrl !== targetUrl) {
    await chrome.tabs.update(transport.tab.id, { url: targetUrl, active: false });
    await waitForProviderTab(transport.tab.id, "chatgpt");
  }
  return transport;
}

async function dispatchBrowserAction(action, ownerInstanceId) {
  let submitIntentRecorded = false;
  let pageExecutionStarted = false;
  let actionTransport = null;
  try {
    const attachments = await browserActionAttachmentBytes(action);
    const result = await withProviderTransportOperation(action.provider, async () => {
      const transport = await prepareBrowserActionTransport(action);
      actionTransport = transport;
      await updateBrowserAction(action.action_id, ownerInstanceId, {
        outcome: "progress",
        phase: action.submit_policy === "submit_once" ? "submit_intent" : "preparing",
        detail: action.submit_policy === "submit_once"
          ? "durable submit intent recorded before the single provider submit boundary"
          : "owned inactive provider target prepared for a staged draft",
      });
      submitIntentRecorded = action.submit_policy === "submit_once";
      pageExecutionStarted = true;
      const [execution] = await chrome.scripting.executeScript({
        target: { tabId: transport.tab.id },
        world: "MAIN",
        func: executeChatGptBrowserActionInPage,
        args: [action, attachments],
      });
      return execution?.result;
    });
    if (!result?.ok) {
      const error = new Error(result?.detail || "protocol_browser_action_result_missing");
      error.submissionMayHaveOccurred = Boolean(result?.submission_may_have_occurred);
      throw error;
    }
    const receipt = {
      action_id: action.action_id,
      receiver_id: action.receiver_id,
      extension_instance_id: ownerInstanceId,
      provider_conversation_id: result.provider_conversation_id || null,
      provider_conversation_url: result.provider_conversation_url || null,
      provider_turn_id: result.provider_turn_id || null,
      observed_surface: result.observed_surface || null,
      observed_model: result.observed_model || null,
      observed_effort: result.observed_effort || null,
      observed_project_ref: result.observed_project_ref || null,
      provider_evidence: result.provider_evidence || {},
      observed_at: new Date().toISOString(),
    };
    await updateBrowserAction(action.action_id, ownerInstanceId, {
      outcome: result.outcome,
      phase: result.outcome,
      detail: result.outcome === "submitted"
        ? "provider returned an exact conversation and user-turn receipt"
        : "provider composer was staged without submitting",
      receipt,
    });
    if (result.outcome === "submitted" && result.provider_conversation_id) {
      await scheduleCaptureFreshness({
        provider: action.provider,
        nativeId: result.provider_conversation_id,
        reason: "provider_turn_submitted",
        delayMs: 30_000,
      });
    }
    if (result.outcome === "submitted" && actionTransport?.cleanupAlarm) {
      await cleanupBackfillTransportTab(actionTransport.cleanupAlarm);
    } else if (result.outcome === "drafted" && actionTransport?.tab?.id) {
      // A staged draft is operator-visible provider state, not a reusable
      // transport. Keep its inactive tab and TTL cleanup, but relinquish
      // ownership so later captures/actions cannot inherit draft text or
      // attachments from it.
      await forgetProviderTransport(action.provider, actionTransport.tab.id);
    }
    await appendCaptureLog({
      ok: true,
      reason: `browser_action_${result.outcome}`,
      action_id: action.action_id,
      provider: action.provider,
      provider_session_id: result.provider_conversation_id || null,
      provider_turn_id: result.provider_turn_id || null,
    });
  } catch (error) {
    const ambiguous = submitIntentRecorded
      && (error?.submissionMayHaveOccurred === true || (pageExecutionStarted && error?.submissionMayHaveOccurred !== false));
    const classified = ambiguous
      ? {
        outcome: "outcome_unknown",
        retry_after_seconds: null,
        detail: `submit execution ended without an exact provider receipt: ${String(error.message || error)}`,
      }
      : classifyBrowserActionFailure(error, error?.retryAfterSeconds || null);
    await updateBrowserAction(action.action_id, ownerInstanceId, {
      ...classified,
      phase: ambiguous ? "outcome_unknown" : "provider_action_failed",
    }).catch(() => undefined);
    await appendCaptureLog({
      ok: false,
      reason: "browser_action_failed",
      action_id: action.action_id,
      error: String(error.message || error),
    });
  }
}

async function pollBrowserActionsOnce() {
  const ownerInstanceId = await browserActionExecutorId();
  const claimed = await getJson(`/v1/browser-actions?claim_by=${encodeURIComponent(ownerInstanceId)}`)
    .catch(() => ({ actions: [] }));
  const action = Array.isArray(claimed.actions) ? claimed.actions[0] : null;
  if (action) await dispatchBrowserAction(action, ownerInstanceId);
  return claimed;
}

function pollBrowserActions() {
  if (browserActionPollPromise) return browserActionPollPromise;
  const tracked = pollBrowserActionsOnce().finally(() => {
    if (browserActionPollPromise === tracked) browserActionPollPromise = null;
  });
  browserActionPollPromise = tracked;
  return browserActionPollPromise;
}

async function ensureBrowserActionAlarm() {
  await chrome.alarms?.create?.(BROWSER_ACTION_ALARM, { delayInMinutes: 0.1, periodInMinutes: 1 });
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

async function refreshActiveTabArchiveState(tab, reason = "tab_state", allowRecovery = true) {
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
      const pairing = await storedReceiverPairing();
      await setStateForTab(tab?.id || null, {
        online: true,
        captured: Boolean(state.captured),
        receiver_pairing: pairing,
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
      } else {
        await appendConversationTimeline({
          provider,
          providerSessionId,
          event: "observed_no_action",
          reason,
          detail: state.state === "archived" ? "already_safe" : "receiver_already_processing",
          tabId: tab?.id || null,
          dedupeWindowMs: 5 * 60 * 1000,
        });
      }
      return state.state || "unknown";
    }

    const health = await checkReceiverHealth();
    const online = ["ok", "recovered"].includes(health.status);
    await setStateForTab(tab?.id || null, {
      online,
      captured: false,
      status: health.receiver_status || null,
      receiver_pairing: health.pairing || null,
      receiver_health: health,
      error: health.status === "unauthorized"
        ? "unauthorized"
        : health.status === "pairing_mismatch"
          ? "receiver_pairing_mismatch"
          : online
            ? null
            : health.detail || "receiver_unavailable",
      provider,
      provider_session_id: null,
      active_page_state: provider ? "supported_no_session" : "unsupported",
      active_tab_id: tab?.id || null,
      passive_reason: reason,
      last_receiver_request_id: health.receiver_request_id || health.receiver_status?.receiver_request_id || null,
    }, url);
    return "not_conversation";
  } catch (error) {
    if (allowRecovery) {
      const health = await checkReceiverHealth({ allowCanonicalRecovery: true });
      if (health.status === "recovered") {
        recentActiveTabStateChecks.delete(throttleKey);
        await appendConversationTimeline({
          provider,
          providerSessionId,
          event: "receiver_recovered",
          reason,
          detail: `${health.recovered_from} -> ${health.endpoint}`,
          tabId: tab?.id || null,
        });
        return refreshActiveTabArchiveState(tab, `${reason}_receiver_recovered`, false);
      }
    }
    const pairing = await storedReceiverPairing();
    await appendConversationTimeline({
      provider,
      providerSessionId,
      event: "held_with_reason",
      reason,
      detail: error.code || "archive_state_check_failed",
      tabId: tab?.id || null,
    });
    await setStateForTab(tab?.id || null, {
      online: false,
      captured: false,
      receiver_pairing: pairing,
      provider,
      provider_session_id: providerSessionId,
      active_page_state: provider ? "receiver_error" : "unsupported",
      active_tab_id: tab?.id || null,
      passive_reason: reason,
      error: String(error.message || error),
      last_receiver_request_id: error.receiverRequestId || null,
    }, url);
    return "receiver_error";
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

function stateSnapshotForTab(tab, globalState, ledger, pairing, health) {
  const url = tab?.url || tab?.pendingUrl || "";
  const provider = archiveProviderForUrl(url);
  const providerSessionId = conversationIdForUrl(url);
  const sameGlobalSession = globalState?.provider === provider
    && globalState?.provider_session_id === providerSessionId;
  const ledgerItem = provider && providerSessionId
    ? ledger?.[sessionKey(provider, providerSessionId)] || {}
    : {};
  const receiverOnline = ["ok", "recovered"].includes(health?.status);
  const receiverError = health?.status === "unauthorized"
    ? "unauthorized"
    : health?.status === "pairing_mismatch"
      ? "receiver_pairing_mismatch"
      : receiverOnline
        ? null
        : health?.detail || "receiver_unavailable";

  if (sameGlobalSession) {
    return {
      ...globalState,
      online: receiverOnline,
      error: receiverError || (receiverOnline ? globalState?.error || null : receiverError),
      receiver_pairing: pairing,
      receiver_health: health,
    };
  }

  return {
    online: receiverOnline,
    error: receiverError,
    captured: ledgerItem.archive_state?.state === "archived" || Boolean(ledgerItem.receiver_request_id),
    provider,
    provider_session_id: providerSessionId,
    active_page_state: providerSessionId ? "conversation" : provider ? "supported_no_session" : "unsupported",
    archive_state: ledgerItem.archive_state || null,
    capture_mode: ledgerItem.capture_mode || null,
    asset_acquisition: ledgerItem.asset_acquisition || null,
    turn_count: ledgerItem.turn_count ?? null,
    attachment_count: ledgerItem.attachment_count ?? null,
    last_receiver_request_id: ledgerItem.receiver_request_id || null,
    updated_at: ledgerItem.updated_at || null,
    receiver_pairing: pairing,
    receiver_health: health,
  };
}

async function missionControlSnapshot(tab = null, { refresh = true } = {}) {
  const resolvedTab = tab || (chrome.tabs?.query
    ? (await chrome.tabs.query({ active: true, currentWindow: true }))[0]
    : null);
  const tabUrl = resolvedTab?.url || resolvedTab?.pendingUrl || "";
  const health = await checkReceiverHealth();
  const receiverOnline = ["ok", "recovered"].includes(health.status);

  // Do not touch archive or queue routes after a pairing mismatch,
  // authorization failure, or offline result. This keeps the mission-control
  // surface read-only and fail-closed until receiver identity is trustworthy.
  if (refresh && resolvedTab && receiverOnline) {
    await refreshActiveTabArchiveState(resolvedTab, "mission_control_snapshot").catch(() => undefined);
  }

  const [coordinator, captureInstanceId] = await Promise.all([
    backfillCoordinator().catch(() => null),
    extensionInstanceId().catch(() => null),
  ]);
  const backfillStatusPromise = coordinator
    ? coordinator.listStatus().catch(() => [])
    : Promise.resolve([]);
  const [stored, backfillJobs, ambient] = await Promise.all([
    chrome.storage.local.get({
      polylogueState: null,
      polylogueSessionLedger: {},
      [CONVERSATION_TIMELINE_KEY]: {},
      [CAPTURE_QUEUE_KEY]: { entries: [], dropped_count: 0 },
      [CAPTURE_FRESHNESS_QUEUE_KEY]: null,
      [RECEIVER_PAIRING_KEY]: null,
    }),
    backfillStatusPromise,
    ambientSettings(hostnameForUrl(tabUrl)),
  ]);
  const pairing = health.pairing || stored[RECEIVER_PAIRING_KEY] || null;
  const baseState = stateSnapshotForTab(
    resolvedTab,
    stored.polylogueState,
    stored.polylogueSessionLedger || {},
    pairing,
    health,
  );
  const freshnessQueue = normalizeFreshnessQueue(stored[CAPTURE_FRESHNESS_QUEUE_KEY]);
  const freshnessEntry = baseState.provider && baseState.provider_session_id
    ? freshnessQueue.entries[sessionKey(baseState.provider, baseState.provider_session_id)] || null
    : null;
  const state = { ...baseState, capture_freshness: freshnessEntry };
  const timelineKey = sessionKey(state.provider, state.provider_session_id);
  const timeline = state.provider && state.provider_session_id
    ? stored[CONVERSATION_TIMELINE_KEY]?.[timelineKey] || []
    : [];
  const settings = await receiverSettings();

  return {
    ok: true,
    generated_at: new Date().toISOString(),
    extension: {
      contract_epoch: EXTENSION_CONTRACT_EPOCH,
      manifest_version: chrome.runtime.getManifest?.().version || null,
      extension_id: chrome.runtime.id || null,
      instance_id: captureInstanceId,
    },
    tab: resolvedTab ? {
      id: resolvedTab.id || null,
      title: resolvedTab.title || null,
      url: tabUrl || null,
    } : null,
    state,
    timeline,
    receiver: {
      health,
      pairing,
      configured_url: settings.baseUrl,
    },
    work: {
      capture_queue: stored[CAPTURE_QUEUE_KEY] || { entries: [], dropped_count: 0 },
      freshness_queue: freshnessQueue,
      backfill_jobs: backfillJobs || [],
    },
    ambient,
    assertions: {
      selection_candidate_supported: true,
      persistence_supported: false,
      reason: "receiver_assertion_route_not_advertised",
    },
  };
}

void loadCaptureQueueIntoCache();
void ensureBrowserActionAlarm();
void ensureCaptureFreshnessAlarms();

chrome.alarms?.onAlarm?.addListener((alarm) => {
  if (alarm?.name === BROWSER_ACTION_ALARM) {
    void pollBrowserActions();
    return;
  }
  if (alarm?.name === CAPTURE_FRESHNESS_ALARM) {
    void processCaptureFreshnessQueue();
    return;
  }
  if (alarm?.name === CAPTURE_FRESHNESS_SWEEP_ALARM) {
    void runCaptureFreshnessSweep();
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
  void captureSupportedTabs("extension_installed_or_updated");
});

chrome.runtime.onStartup?.addListener(() => {
  void captureSupportedTabs("browser_startup");
  void backfillCoordinator().then((coordinator) => coordinator.wake());
  void ensureCaptureFreshnessAlarms();
});

chrome.tabs?.onActivated?.addListener((activeInfo) => {
  void (async () => {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    const archiveState = await refreshActiveTabArchiveState(tab, "tab_activated");
    if (!["missing", "receiver_error", "not_conversation"].includes(archiveState)) {
      await captureTab(tab, "tab_activated");
    }
  })();
});

chrome.tabs?.onUpdated?.addListener((tabId, changeInfo, tab) => {
  if (changeInfo?.status !== "complete" && !changeInfo?.url) return;
  void (async () => {
    const resolvedTab = tab?.id ? tab : await chrome.tabs.get(tabId);
    const archiveState = await refreshActiveTabArchiveState(resolvedTab, "tab_updated");
    if (!["missing", "receiver_error", "not_conversation"].includes(archiveState)) {
      await captureTab(resolvedTab, "tab_updated");
    }
  })();
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    if (message.type === "polylogue.missionControl.status") {
      sendResponse(await missionControlSnapshot(sender.tab || null, { refresh: message.refresh !== false }));
      return;
    }
    if (message.type === "polylogue.receiverPairing.status") {
      const health = await checkReceiverHealth();
      sendResponse({ ok: true, health, pairing: health.pairing || await storedReceiverPairing() });
      return;
    }
    if (message.type === "polylogue.receiverPairing.reset") {
      await clearReceiverPairing();
      const health = await checkReceiverHealth({ allowCanonicalRecovery: false });
      sendResponse({ ok: true, health, pairing: health.pairing || await storedReceiverPairing() });
      return;
    }
    if (message.type === "polylogue.ambient.configure") {
      const url = sender.tab?.url || sender.tab?.pendingUrl || message.url || "";
      const settings = await saveAmbientSettings({
        enabled: message.enabled ?? null,
        hostname: message.hostname || hostnameForUrl(url),
        siteEnabled: message.site_enabled ?? null,
      });
      sendResponse({ ok: true, ambient: settings });
      return;
    }
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
          detail: error.code || "capture_rejected",
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
    if (message.type === "polylogue.captureFreshnessHint") {
      const senderUrl = sender.tab?.url || sender.tab?.pendingUrl || "";
      const senderProvider = archiveProviderForUrl(senderUrl);
      const senderSessionId = conversationIdForUrl(senderUrl);
      const provider = message.provider || senderProvider;
      const nativeId = message.provider_session_id || senderSessionId;
      if (sender.tab && (provider !== senderProvider || (senderSessionId && nativeId !== senderSessionId))) {
        throw new Error("freshness_hint_sender_identity_mismatch");
      }
      sendResponse({
        ok: true,
        ...(await scheduleCaptureFreshness({
          provider,
          nativeId,
          reason: message.reason || "provider_page_hint",
          delayMs: Math.max(0, Math.min(5 * 60_000, Number(message.delay_ms) || 5_000)),
          providerUpdatedAt: message.provider_updated_at || null,
        })),
      });
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
      const health = await checkReceiverHealth();
      if (["ok", "recovered"].includes(health.status)) {
        await refreshCurrentActiveTab(message.reason || "status");
      } else {
        const storedBefore = await chrome.storage.local.get({ polylogueState: null });
        const previous = storedBefore.polylogueState || {};
        await setState({
          ...previous,
          online: false,
          receiver_pairing: health.pairing || previous.receiver_pairing || null,
          receiver_health: health,
          last_receiver_request_id: health.receiver_request_id || previous.last_receiver_request_id || null,
          error: health.status === "unauthorized"
            ? "unauthorized"
            : health.status === "pairing_mismatch"
              ? "receiver_pairing_mismatch"
              : health.detail || "receiver_unavailable",
        });
      }
      const stored = await chrome.storage.local.get({ polylogueState: null });
      const state = stored.polylogueState || {};
      if (!state.online) {
        sendResponse({
          ok: false,
          error: state.error || "receiver_unavailable",
          receiver_request_id: state.last_receiver_request_id || null,
          receiver_pairing: state.receiver_pairing || null,
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
    if (message.type === "polylogue.browserActions.status") {
      const [status, ownerInstanceId] = await Promise.all([
        getJson("/v1/browser-actions"),
        browserActionExecutorId(),
      ]);
      sendResponse({ ok: true, ownerInstanceId, actions: status.actions || [] });
      return;
    }
    if (message.type === "polylogue.browserActions.poll") {
      sendResponse({ ok: true, ...(await pollBrowserActions()) });
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
