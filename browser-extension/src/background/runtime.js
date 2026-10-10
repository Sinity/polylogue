import { AttachmentSha256 } from "../actions/sha256.js";
import { NativeCaptureNormalizer } from "../capture/native.js";
import { CaptureStaging } from "../capture/staging.js";
import { BackfillCoordinator } from "../backfill/coordinator.js";
import { BACKFILL_RECOVERY_CHECKPOINT_VERSION, DURABLE_RECEIVER_ACK_FIELDS, retryAfterMs, receiverAckContractError } from "../backfill/models.js";
import { providerAdapters } from "../backfill/providers.js";
import { executeProviderPageRequest } from "../backfill/page_transport.js";
import { IndexedDbBackfillStore, nativeCacheOrder } from "../backfill/storage.js";
import { CaptureJobClient, deriveAccountScope } from "../backfill/capture_jobs.js";
import {
  classifyBrowserActionFailure,
  executeChatGptBrowserActionInPage,
  transferBrowserActionAttachmentInPage,
} from "../actions/chatgpt.js";
import {
  chatGptCaptureNeedsFollowUp,
  claimDueFreshness,
  completeFreshnessClaim,
  extendProviderCooldown,
  failureRetryDelayMs,
  normalizeFreshnessQueue,
  runningPollDelayMs,
  scheduleFreshnessHint,
} from "../capture/freshness.js";
import { requireProviderCooldownMs } from "../capture/provider_cooldown.js";
import { BACKGROUND_ALARMS } from "./adapters.js";
import { registerBackgroundEvents } from "./events.js";

// These bindings are supplied by the composition root. Keeping them here
// makes every domain function use the same seams in tests and in the worker;
// no controller discovers a second Chrome or network client.
let runtimeChrome = null;
let runtimeNetwork = null;
let captureStaging = null;
let captureStore = null;
let nativeNormalizer = null;
let runtimeWorkerId = null;
const nativeNormalizations = new Map();
function trackNativeOperation(operation) {
  let drained;
  operation.drained = new Promise((resolve) => { drained = resolve; });
  operation.finish = () => { nativeNormalizations.delete(operation.controller.signal); drained(); };
  nativeNormalizations.set(operation.controller.signal, operation);
}
const captureDeliveries = new Map();
const foregroundDeliveryCompletions = new Map();

// Must match src/common.js's TEMPORARY_CHAT_ID_KEY-adjacent sentinel exactly
// (sessionIdFromUrl's `__polylogue_temporary_chat__` return value) -- this
// module has no access to the content script's per-tab sessionStorage, so it
// cannot know the eventual real `temporary:<hex>` session id in advance. It
// only needs a stable, regex-valid ([A-Za-z0-9_-]{1,256}) truthy signal that
// a temporary-chat URL is a real, capturable conversation.
const TEMPORARY_CHAT_SENTINEL = "__polylogue_temporary_chat__";
const DEFAULT_RECEIVER = "http://127.0.0.1:8765";
const EXTENSION_CONTRACT_EPOCH = "canonical-capture-mission-control-v1";
const RECEIVER_API_SCHEMA = "polylogue-browser-capture/v1";
const RECEIVER_PAIRING_KEY = "polylogueReceiverPairing";
const RECEIVER_HEALTH_TIMEOUT_MS = 5000;
const MISSION_INTELLIGENCE_TIMEOUT_MS = 7000;
const RECEIVER_TRUST_CACHE_MS = 10000;
const AMBIENT_SETTINGS_KEY = "polylogueAmbientSettings";
const BACKGROUND_CAPTURE_MIN_INTERVAL_MS = 30000;
const ACTIVE_TAB_STATE_MIN_INTERVAL_MS = 4000;
const CAPTURE_LOG_LIMIT = 80;
const DEBUG_LOG_LIMIT = 160;
const CONVERSATION_TIMELINE_KEY = "polylogueConversationTimeline";
const ACCEPTED_MESSAGE_IDENTITIES_KEY = "polylogueAcceptedMessageIdentities";
// Version 2 keys each session's accepted identities by message ref.
const ACCEPTED_MESSAGE_IDENTITIES_VERSION_KEY = "polylogueAcceptedMessageIdentitiesVersion";
const CONVERSATION_TIMELINE_EVENT_LIMIT = 24;
const BACKFILL_RECOVERY_CHECKPOINT_KEY = "polylogueBackfillRecoveryCheckpoint";
const BACKFILL_WORKER_EPOCH = globalThis.crypto?.randomUUID?.() || `worker-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const CONVERSATION_TIMELINE_CONVERSATION_LIMIT = 80;
const BROWSER_ACTION_ALARM = BACKGROUND_ALARMS.browserActions;
const CAPTURE_FRESHNESS_ALARM = BACKGROUND_ALARMS.captureFreshness;
const CAPTURE_FRESHNESS_SWEEP_ALARM = BACKGROUND_ALARMS.captureFreshnessSweep;
const CAPTURE_FRESHNESS_QUEUE_KEY = "polylogueCaptureFreshnessQueue";
const CAPTURE_FRESHNESS_LEASE_MS = 2 * 60 * 1000;
const CAPTURE_FRESHNESS_SWEEP_MINUTES = 15;
const CAPTURE_FRESHNESS_SWEEP_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const BROWSER_ACTION_ATTACHMENT_CHUNK_BYTES = 64 * 1024;
const BACKFILL_TRANSPORT_TAB_TTL_MS = 5 * 60 * 1000;
const BACKFILL_TRANSPORT_CLEANUP_PREFIX = BACKGROUND_ALARMS.backfillTransportCleanup;
const PROVIDER_TRANSPORT_SESSION_PREFIX = "polylogueProviderTransportTab";
const PROVIDER_TRANSPORT_OPERATOR_TAKEN_SESSION_PREFIX = "polylogueProviderTransportOperatorTaken";
const recentBackgroundCaptures = new Map();
const recentActiveTabStateChecks = new Map();
let backfillCoordinatorPromise = null;
const backfillStartPromises = new Map();
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

function replaceLegacyAcceptedMessageIdentities() {
  // The accepted-identity cache is derived from receiver responses. A cache
  // written in the version 1 scalar shape is dropped rather than read; the
  // capture the extension runs on install or update rewrites it keyed by
  // message ref.
  return serializeStorageMutation(async () => {
    const current = await runtimeChrome.storage.local.get({ [ACCEPTED_MESSAGE_IDENTITIES_VERSION_KEY]: 1 });
    if (Number(current[ACCEPTED_MESSAGE_IDENTITIES_VERSION_KEY]) >= 2) return;
    await runtimeChrome.storage.local.set({
      [ACCEPTED_MESSAGE_IDENTITIES_KEY]: {},
      [ACCEPTED_MESSAGE_IDENTITIES_VERSION_KEY]: 2,
    });
  });
}

function serializeCaptureQueueMutation(mutation, signal = null) {
  let started = false;
  const run = () => {
    signal?.throwIfAborted();
    started = true;
    return mutation();
  };
  const result = captureQueueMutationQueue.then(run, run);
  captureQueueMutationQueue = result.then(() => undefined, () => undefined);
  if (!signal) return result;
  // Cancellation before ownership starts must not wait for another delivery's
  // physical upload. The queued callback checks the same signal before writing.
  return new Promise((resolve, reject) => {
    const abort = () => { if (!started) reject(signal.reason); };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    result.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
  });
}

function extensionInstanceId() {
  if (!extensionInstanceIdPromise) {
    const key = "polylogueExtensionInstanceId";
    const candidate = (async () => {
      const stored = await runtimeChrome.storage.local.get({ [key]: "" });
      if (stored[key]) return stored[key];
      const created = globalThis.crypto?.randomUUID?.() || `instance-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      await runtimeChrome.storage.local.set({ [key]: created });
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
      const session = runtimeChrome.storage.session;
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

async function withExtensionInstanceAttribution(envelope, sender = null) {
  if (envelope.receiver_native) {
    const witness = await captureStore.getCapture(envelope.capture_record_ref);
    if (!witness || witness.state !== "ready" || witness.receiver_native.sha256 !== envelope.receiver_native.sha256 ||
        (sender && (witness.owner.tab_id !== sender.tab?.id || witness.owner.document_id !== sender.documentId))) throw new Error("native_preparation_owner_mismatch");
    return envelope;
  }
  const instanceId = await extensionInstanceId();
  const { capture_observation_ref: observationRef, ...body } = envelope;
  const witness = envelope.capture_record_ref
    ? await captureStore.getCapture(envelope.capture_record_ref)
    : observationRef ? await captureStore.getCapture(observationRef.id) : null;
  if (envelope.capture_record_ref && (!witness || witness.state !== "ready" || typeof witness.raw_ref !== "string" ||
      typeof witness.raw_revision_sha256 !== "string" || !Number.isSafeInteger(witness.record_count))) {
    throw new Error("capture_observation_record_unavailable");
  }
  if (witness && (witness.provider !== envelope.session.provider || witness.native_id !== envelope.session.provider_session_id)) throw new Error("capture_observation_identity_mismatch");
  if (witness && sender && (witness.owner.tab_id !== sender.tab?.id || witness.owner.document_id !== (sender.documentId || null))) throw new Error("capture_observation_owner_mismatch");
  if (observationRef && (!witness || witness.kind !== "dom-observation" || witness.token !== observationRef.token)) throw new Error("capture_observation_owner_mismatch");
  const provenance = { ...(envelope?.provenance || {}) };
  delete provenance.acquisition_sequence;
  return {
    ...body,
    provenance: {
      ...provenance,
      // The service worker owns this persistent identity. Do not trust an
      // independently reloadable content script to choose the attribution.
      extension_instance_id: instanceId,
      ...(witness ? { captured_at: witness.observed_at } : {}),
      ...(Number.isSafeInteger(witness?.acquisition_sequence) && witness.acquisition_sequence > 0
        ? { acquisition_sequence: witness.acquisition_sequence } : {}),
    },
    ...(observationRef ? { capture_observation_ref: observationRef } : {}),
  };
}

async function* commitCaptureJobsToReceiver(instanceId) {
  const settings = await receiverSettings();
  const client = new CaptureJobClient({ baseUrl: settings.baseUrl, cache: runtimeChrome.storage.local, fetchImpl: runtimeNetwork });
  for await (const job of captureStore.jobs()) {
    try {
      if (!/^h1:[A-Za-z0-9_-]{43}$/.test(job.account_scope || "")) throw new Error("capture_job_account_scope_unresolved");
      const handle = await providerAccountHandle(job.provider);
      if (!handle) throw new Error(`capture_job_identity_unavailable:${job.provider}`);
      let adopted = await client.recoverOrCreate({ provider: job.provider, accountHandle: handle,
        locator: { kind: "backfill", provider: job.provider, cutoff: job.cutoff },
        intentPayload: { provider: job.provider, cutoff: job.cutoff }, sessionId: instanceId });
      if (adopted.scope.kind !== "account" || adopted.scope.key !== job.account_scope) throw new Error("capture_job_account_scope_mismatch");
      for (;;) {
        let snapshot = await captureStore.pendingRecoverySnapshot(job.id);
        const recovered = Boolean(snapshot);
        if (snapshot) {
          const meta = await captureStaging.metadata(snapshot.artifact_id).catch((error) => {
            if (error?.name === "NotFoundError") return null;
            throw error;
          });
          if (meta?.state === "checkpoint-acknowledged") {
            await captureStaging.releaseCheckpoint(snapshot, meta.receiver_receipt);
            continue;
          }
        } else {
          const snapshotId = `checkpoint:${globalThis.crypto.randomUUID()}`;
          const artifactId = captureStaging.producerStageId({ checkpoint_snapshot_id: snapshotId }, snapshotId);
          snapshot = await captureStore.createRecoverySnapshot(job.id, snapshotId, artifactId);
        }
        const prepared = await captureStaging.prepareCheckpoint(snapshot);
        adopted = await client.update(adopted, captureJobRetryState(snapshot.job));
        const result = await client.checkpoint(adopted, prepared);
        const unchanged = adopted.job.checkpoint_digest === prepared.digest;
        adopted = { ...adopted, job: result.job };
        await captureStaging.releaseCheckpoint(snapshot, result.receipt || result.job.latest_receipt);
        // A fresh consistent snapshot after ACK detects mutations that occurred
        // during the upload. Stable canonical bytes terminate without HTTP.
        if (unchanged && !recovered) break;
      }
      yield { job_id: job.id, error: null, outcome: "committed" };
    } catch (error) {
      yield { job_id: job.id, error: String(error?.message || error), outcome: error?.outcome || null,
        retry_after_ms: Number.isFinite(error?.retryAfterMs) ? error.retryAfterMs : null,
        retry_until_ms: Number.isFinite(error?.retryUntilMs) ? error.retryUntilMs : null };
    }
  }
}

function captureJobRetryState(job) {
  const attempt = Number.isSafeInteger(job.transport_failures) && job.transport_failures >= 0
    ? job.transport_failures
    : 0;
  const nextEligible = Number.isFinite(job.cooldown_until_ms)
    ? new Date(job.cooldown_until_ms).toISOString()
    : null;
  let state = "ready";
  if (job.status === "complete") state = "completed";
  else if (job.status === "cancelled") state = "abandoned";
  else if (job.status === "paused") state = "held";
  else if (nextEligible) state = "retry_wait";
  return {
    state,
    attempt,
    reason: job.cooldown_reason || job.last_error || null,
    next_eligible_at: state === "retry_wait" ? nextEligible : null,
  };
}

async function loadBackfillCheckpointFromCaptureJobs(instanceId, providers) {
  const settings = await receiverSettings();
  const client = new CaptureJobClient({ baseUrl: settings.baseUrl, cache: runtimeChrome.storage.local, fetchImpl: runtimeNetwork });
  const successfulProviders = [];
  const unavailableProviders = [];
  for (const provider of providers) {
    try {
      const accountHandle = await providerAccountHandle(provider);
      for await (const entry of client.discoverRecovery(provider, accountHandle, instanceId)) {
        if (!entry.lease) throw new Error("capture_job_recovery_lease_held");
        await client.restoreCheckpoint(entry, captureStore, captureStaging);
      }
      successfulProviders.push(provider);
    } catch (error) {
      unavailableProviders.push(provider);
      await appendDebugLog({ stage: "capture_job_recovery_unavailable", provider, error: String(error.message || error) });
    }
  }
  return { successfulProviders, unavailableProviders };
}

async function convertAcquiredBackfillRow(store, original) {
  if (await store.localCheckpointRecordPublished("queue", original)) return;
  const envelope = globalThis.structuredClone(original.envelope);
  const prepared = await captureStaging.prepare(envelope, captureStaging.conversionId(`checkpoint:${original.id}`),
    { delivery_kind: "backfill", id: original.id, job_id: original.job_id });
  await captureStaging.file(prepared.ref);
  const replacement = { ...original, body_ref: prepared.ref, envelope: {
    capture_body_ref: prepared.ref,
    session: { provider: original.provider, provider_session_id: original.native_id },
    provider_meta: { capture_fidelity: original.envelope.provider_meta?.capture_fidelity || "native_full" },
  } };
  await store.convertLocalCheckpointRecord("queue", original, replacement);
}

async function backfillCoordinator() {
  const initializing = !backfillCoordinatorPromise;
  if (initializing) {
    const candidate = (async () => {
      const store = captureStore;
      const instanceId = await extensionInstanceId();
      const stored = await runtimeChrome.storage.local.get({ [BACKFILL_RECOVERY_CHECKPOINT_KEY]: null });
      const predecessor = stored[BACKFILL_RECOVERY_CHECKPOINT_KEY];
      if (predecessor !== null) {
        if (predecessor.version !== BACKFILL_RECOVERY_CHECKPOINT_VERSION ||
            !["jobs", "queue", "revisions"].every((key) => Array.isArray(predecessor[key]))) {
          throw new Error("checkpoint_conversion_shape_invalid");
        }
        // This is the sole read of the shipped whole-ledger cache. Keep the
        // original until each record and acquired body has durable custody.
        for (const collection of ["jobs", "queue", "revisions"]) {
          for (const original of predecessor[collection]) {
            if (await store.localCheckpointRecordPublished(collection, original)) continue;
            let replacement = original;
            if (collection === "queue" && original.envelope) {
              await convertAcquiredBackfillRow(store, original);
              continue;
            }
            if (collection === "queue" && replacement.body_ref) await captureStaging.file(replacement.body_ref);
            await store.convertLocalCheckpointRecord(collection, original, replacement);
          }
        }
        await runtimeChrome.storage.local.remove(BACKFILL_RECOVERY_CHECKPOINT_KEY);
      }
      for await (const original of store.unconvertedBackfillBodies()) await convertAcquiredBackfillRow(store, original);
      const providers = ["chatgpt", "claude-ai", "grok"];
      const recovery = await loadBackfillCheckpointFromCaptureJobs(instanceId, providers);
      const adapters = providerAdapters(providerPageFetch, { requirePageContext: true, nativeBundleOwner: backfillNativeBundleOwner() });
      const coordinator = new BackfillCoordinator({
        prepareCapture: (envelope, item, signal) => envelope.receiver_native ? { contentHash: envelope.receiver_native.sha256 } : captureStaging.prepare(envelope, captureStaging.conversionId(`backfill:${item.id}`), { delivery_kind: "backfill", id: item.id, job_id: item.job_id }, signal),
        receiverAcked: (envelope) => envelope.receiver_native ? captureStaging.acknowledgeNative(envelope.capture_record_ref) : captureStaging.acknowledge(envelope.capture_body_ref),
        store,
        adapters,
        receiver: (envelope, serialized, signal) => postJson(
          "/v1/browser-captures",
          envelope,
          serialized,
          true,
          signal,
        ),
        receiverPreflight: backfillReceiverPreflight,
        checkpoint: () => commitCaptureJobsToReceiver(instanceId),
        alarms: runtimeChrome.alarms,
        instanceId,
        receiverContractEpoch: BACKFILL_WORKER_EPOCH,
      });
      coordinator.unavailableRecoveryProviders = recovery.unavailableProviders;
      coordinator.recoveryInstanceId = instanceId;
      return coordinator;
    })();
    backfillCoordinatorPromise = candidate;
    void candidate.catch(() => {
      if (backfillCoordinatorPromise === candidate) backfillCoordinatorPromise = null;
    });
  }
  const coordinator = await backfillCoordinatorPromise;
  if (!initializing && coordinator.unavailableRecoveryProviders.length) {
    const tabs = await runtimeChrome.tabs.query({});
    const providers = coordinator.unavailableRecoveryProviders.filter(
      (provider) => tabs.some((tab) => archiveProviderForUrl(tab.url || tab.pendingUrl || "") === provider),
    );
    if (!providers.length) return coordinator;
    if (!coordinator.recoveryPromise) {
      coordinator.recoveryPromise = (async () => {
        const recovery = await loadBackfillCheckpointFromCaptureJobs(
          coordinator.recoveryInstanceId, providers,
        );
        coordinator.unavailableRecoveryProviders = [
          ...coordinator.unavailableRecoveryProviders.filter((provider) => !providers.includes(provider)),
          ...recovery.unavailableProviders,
        ];
      })().finally(() => { coordinator.recoveryPromise = null; });
    }
    await coordinator.recoveryPromise;
  }
  return coordinator;
}

async function startBackfill(request) {
  const provider = String(request.provider || "");
  const stableRequestKey = (value) => {
    if (Array.isArray(value)) return value.map(stableRequestKey);
    if (value && typeof value === "object") {
      return Object.fromEntries(
        Object.entries(value)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, child]) => [key, stableRequestKey(child)]),
      );
    }
    return value;
  };
  const requestKey = JSON.stringify(stableRequestKey({
    provider,
    cutoff: request.cutoff ?? null,
    policy: request.policy || {},
    provider_options: request.provider_options || {},
  }));
  const inFlight = backfillStartPromises.get(requestKey);
  if (inFlight) return inFlight;
  const pending = (async () => {
    const coordinator = await backfillCoordinator();
    const settings = await receiverSettings();
    const client = new CaptureJobClient({ baseUrl: settings.baseUrl, cache: runtimeChrome.storage.local, fetchImpl: runtimeNetwork });
    const handle = await providerAccountHandle(provider);
    if (!handle) throw new Error(`capture_job_identity_unavailable:${provider}`);
    const accountScope = await deriveAccountScope(await client.scopeNamespace(), provider, handle);
    const job = await coordinator.start({
      provider,
      account_scope: accountScope,
      cutoff: request.cutoff,
      policy: request.policy || {},
      provider_options: request.provider_options || {},
    });
    void coordinator.wake(job.id);
    return job;
  })();
  backfillStartPromises.set(requestKey, pending);
  try {
    return await pending;
  } finally {
    if (backfillStartPromises.get(requestKey) === pending) backfillStartPromises.delete(requestKey);
  }
}

// ---- Capture retry queue --------------------------------------------------
//
// Acquired bodies and delivery references remain durable in OPFS and the
// indexed queue until an actual receiver ACK or an explicit terminal refusal.
// Chrome storage holds only the completed conversion marker; delivery cursors
// are authoritative for counts, ordering, and retries.
const CAPTURE_QUEUE_KEY = "polylogueCaptureQueue";
const CAPTURE_QUEUE_EMPTY = Object.freeze({ version: 3, dropped_count: 0 });
let queueConversionPromise = null;
const CAPTURE_RETRY_ALARM = BACKGROUND_ALARMS.captureRetry;
const CAPTURE_RETRY_BASE_DELAY_MS = 30000;
const CAPTURE_RETRY_MAX_DELAY_MS = 30 * 60 * 1000;
const CAPTURE_RETRY_ALARM_PERIOD_MINUTES = 1;
// In-memory mirror of the queue length so badge rendering never needs an
// extra storage round trip; reloaded from storage once at SW startup.
let cachedQueueLength = 0;

function injectionPlanForUrl(url) {
  try {
    const parsed = new globalThis.URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) {
      return [
        { files: ["src/content/asset_stream.js", "src/content/chatgpt_bridge.js"], world: "MAIN" },
        {
          files: [
            "src/common.js",
            "src/operator_status.js",
            "src/content/message_layer.js",
            "src/content/ambient_surface.js",
            "src/content/asset_stream.js", "src/content/chatgpt.js",
          ],
        },
      ];
    }
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) {
      return [
        { files: ["src/content/asset_stream.js", "src/content/claude_bridge.js"], world: "MAIN" },
        {
          files: [
            "src/common.js",
            "src/operator_status.js",
            "src/content/message_layer.js",
            "src/content/ambient_surface.js",
            "src/content/asset_stream.js", "src/content/claude.js",
          ],
        },
      ];
    }
    if (parsed.hostname === "grok.com" || parsed.hostname.endsWith(".grok.com")) {
      // grok.com now has a MAIN-world bridge (native REST capture,
      // polylogue Grok native-capture upgrade 2026-07-31) mirroring
      // chatgpt/claude above. x.com/twitter.com were dropped here: Grok's
      // embedded surface on X is served through X's own API, not
      // grok.com's /rest/app-chat/* REST surface this bridge calls, and
      // this upgrade deleted the DOM-only capture path that used to be the
      // (lossy) fallback for those origins. See manifest.json's matching
      // content_scripts change and this repo's Grok-on-X follow-up bead.
      return [
        { files: ["src/content/asset_stream.js", "src/content/grok_bridge.js"], world: "MAIN" },
        { files: ["src/content/asset_stream.js", "src/common.js", "src/content/grok.js"] },
      ];
    }
    if (parsed.hostname === "gemini.google.com" || parsed.hostname.endsWith(".gemini.google.com")) {
      return [{ files: ["src/common.js", "src/operator_status.js", "src/content/message_layer.js", "src/content/ambient_surface.js", "src/content/gemini.js"] }];
    }
  } catch {
    return [];
  }
  return [];
}

async function receiverSettings() {
  const stored = await runtimeChrome.storage.local.get({
    receiverBaseUrl: DEFAULT_RECEIVER,
  });
  return {
    baseUrl: String(stored.receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, ""),
  };
}

// The manifest grants http://127.0.0.1:8765/* at install time and nothing
// wider. The unscoped loopback grant it used to carry also covered the
// unauthenticated archive API on 8766, and extension fetches are not subject
// to CORS (2026-07-31 leak audit L7, polylogue-tztk), so a non-default
// receiver port is now an optional permission the operator grants from the
// popup. This is the enforcement point: settings that name an origin the
// extension does not hold are refused rather than stored, so a wedged or
// hostile configure message cannot point the capture stream at another
// loopback service.
async function loopbackOriginIsGranted(baseUrl) {
  let origin;
  try {
    origin = `${new globalThis.URL(baseUrl).origin}/*`;
  } catch {
    return false;
  }
  if (!runtimeChrome.permissions?.contains) return false;
  try {
    return Boolean(await runtimeChrome.permissions.contains({ origins: [origin] }));
  } catch {
    return false;
  }
}

// All settings and pairing writes share this owner; slow status probes run
// outside it and may publish only while their captured configuration remains.
let receiverConfigurationRevision = 0;
async function receiverScopeIsCurrent(scope) {
  if (scope.revision !== receiverConfigurationRevision) return false;
  const current = await receiverSettings();
  return scope.revision === receiverConfigurationRevision
    && current.baseUrl === scope.settings.baseUrl;
}
async function receiverHealthScope() {
  return serializeStorageMutation(async () => ({
    settings: await receiverSettings(),
    pairing: await storedReceiverPairing(),
    revision: receiverConfigurationRevision,
  }));
}
async function restoreReceiverSettings(previous, owned) {
  return serializeStorageMutation(async () => {
    const keys = ["receiverBaseUrl", RECEIVER_PAIRING_KEY];
    if (!previous || typeof previous !== "object" || Array.isArray(previous)
      || !owned || typeof owned.baseUrl !== "string"
      || !(owned.receiverId === null || typeof owned.receiverId === "string")
      || !(owned.revision === null || (Number.isSafeInteger(owned.revision) && owned.revision >= 0))) {
      throw new Error("proof_receiver_configuration_changed");
    }
    if ((Object.hasOwn(previous, "receiverBaseUrl") && typeof previous.receiverBaseUrl !== "string")
      || (Object.hasOwn(previous, RECEIVER_PAIRING_KEY) && previous[RECEIVER_PAIRING_KEY] !== null
        && (typeof previous[RECEIVER_PAIRING_KEY] !== "object" || Array.isArray(previous[RECEIVER_PAIRING_KEY])))) {
      throw new Error("proof_receiver_configuration_changed");
    }
    if (previous.receiverBaseUrl && previous.receiverBaseUrl !== DEFAULT_RECEIVER
      && !(await loopbackOriginIsGranted(previous.receiverBaseUrl))) {
      throw new Error("receiver_origin_not_permitted");
    }
    const current = await runtimeChrome.storage.local.get(keys);
    const unchanged = keys.every(key => Object.hasOwn(current, key) === Object.hasOwn(previous, key)
      && JSON.stringify(current[key]) === JSON.stringify(previous[key]));
    if (unchanged) return { ...await receiverSettings(), configurationRevision: receiverConfigurationRevision };
    if (owned.revision !== receiverConfigurationRevision) throw new Error("proof_receiver_configuration_changed");
    if (current.receiverBaseUrl !== owned.baseUrl
      || (owned.receiverId !== null && current[RECEIVER_PAIRING_KEY]?.receiver_id !== owned.receiverId)) {
      throw new Error("proof_receiver_configuration_changed");
    }
    receiverConfigurationRevision += 1;
    trustedReceiverHealthCache = null;
    const values = {}; const missing = [];
    for (const key of keys) {
      if (Object.hasOwn(previous, key)) values[key] = previous[key]; else missing.push(key);
    }
    await runtimeChrome.storage.local.set(values);
    if (missing.length) await runtimeChrome.storage.local.remove(missing);
    return { ...await receiverSettings(), configurationRevision: receiverConfigurationRevision };
  });
}

async function saveReceiverSettings(receiverBaseUrl) {
  return serializeStorageMutation(async () => {
    trustedReceiverHealthCache = null;
    const normalizedBaseUrl = String(receiverBaseUrl || DEFAULT_RECEIVER).replace(/\/+$/, "") || DEFAULT_RECEIVER;
    if (normalizedBaseUrl !== DEFAULT_RECEIVER && !(await loopbackOriginIsGranted(normalizedBaseUrl))) {
      const error = new Error("receiver_origin_not_permitted");
      error.origin = normalizedBaseUrl;
      throw error;
    }
    receiverConfigurationRevision += 1;
    await runtimeChrome.storage.local.set({
      receiverBaseUrl: normalizedBaseUrl,
    });
    // An operator who explicitly types a non-canonical endpoint into settings
    // is declaring a deliberate development pairing, not drifting there by
    // accident. Record that intent on the pairing itself so a later stale
    // probe reports loudly instead of silently self-healing back to the
    // canonical endpoint out from under an intentional dev-loop session
    // (polylogue-jlme.5). Pointing settings back at the canonical endpoint is
    // an equally explicit act and clears the flag.
    const prior = await storedReceiverPairing();
    const isDevOverride = normalizedBaseUrl !== DEFAULT_RECEIVER;
    if (prior && Boolean(prior.dev_override) !== isDevOverride) {
      await persistReceiverPairing({ ...prior, dev_override: isDevOverride });
    } else if (!prior && isDevOverride) {
      await persistReceiverPairing({
        state: "legacy",
        endpoint: normalizedBaseUrl,
        dev_override: true,
        last_seen_at: null,
        checked_at: new Date().toISOString(),
        last_error: null,
      });
    }
    const settings = await receiverSettings();
    return { ...settings, configurationRevision: receiverConfigurationRevision };
  });
}

async function storedReceiverPairing() {
  const stored = await runtimeChrome.storage.local.get({ [RECEIVER_PAIRING_KEY]: null });
  const pairing = stored[RECEIVER_PAIRING_KEY];
  return pairing && typeof pairing === "object" ? pairing : null;
}

async function persistReceiverPairing(pairing) {
  await runtimeChrome.storage.local.set({ [RECEIVER_PAIRING_KEY]: pairing });
  return pairing;
}

async function markReceiverPairingUnavailable(detail, scope) {
  return serializeStorageMutation(async () => {
    if (!await receiverScopeIsCurrent(scope)) return null;
    trustedReceiverHealthCache = null;
    const pairing = await storedReceiverPairing();
    if (!pairing) return null;
    // A deliberately dev-overridden pairing going stale is not ordinary
    // "receiver asleep" (calm, expected, self-heals) -- it is the operator's
    // chosen endpoint disappearing, and canonical failover is intentionally
    // suppressed for it (see checkReceiverHealth). Give it a distinct, loud
    // state so the popup does not present it as routine.
    return persistReceiverPairing({
      ...pairing,
      state: pairing.dev_override ? "dev_override_stale" : "offline",
      last_error: String(detail || "receiver_unavailable"),
      checked_at: new Date().toISOString(),
    });
  });
}

async function observeReceiverIdentity(status, endpoint, scope) {
  return serializeStorageMutation(async () => {
    if (!await receiverScopeIsCurrent(scope)) return null;
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
        dev_override: Boolean(prior?.dev_override),
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
      dev_override: Boolean(prior?.dev_override),
      paired_at: prior?.paired_at || now,
      last_seen_at: now,
      checked_at: now,
      state: "online",
      last_error: null,
    });
  });
}

async function clearReceiverPairing(expectedRevision = null) {
  return serializeStorageMutation(async () => {
    if (expectedRevision !== null && (!Number.isSafeInteger(expectedRevision) || expectedRevision !== receiverConfigurationRevision)) {
      throw new Error("receiver_configuration_changed");
    }
    receiverConfigurationRevision += 1;
    trustedReceiverHealthCache = null;
    await runtimeChrome.storage.local.remove?.(RECEIVER_PAIRING_KEY);
    // Test doubles and older browser shims may not expose remove(). Setting null
    // is equivalent for all readers and keeps reset bounded to this one key.
    if (!runtimeChrome.storage.local.remove) await runtimeChrome.storage.local.set({ [RECEIVER_PAIRING_KEY]: null });
    return { settings: await receiverSettings(), revision: receiverConfigurationRevision };
  });
}

function hostnameForUrl(url) {
  try {
    return new globalThis.URL(url || "").hostname;
  } catch {
    return "";
  }
}

async function ambientSettings(hostname = "") {
  const stored = await runtimeChrome.storage.local.get({
    [AMBIENT_SETTINGS_KEY]: { enabled: true, automatic_capture_enabled: true, disabled_sites: {} },
  });
  const raw = stored[AMBIENT_SETTINGS_KEY] && typeof stored[AMBIENT_SETTINGS_KEY] === "object"
    ? stored[AMBIENT_SETTINGS_KEY]
    : {};
  const disabledSites = raw.disabled_sites && typeof raw.disabled_sites === "object" ? raw.disabled_sites : {};
  return {
    enabled: raw.enabled !== false,
    automatic_capture_enabled: raw.automatic_capture_enabled !== false,
    disabled_sites: disabledSites,
    site: hostname || null,
    site_enabled: hostname ? disabledSites[hostname] !== true : true,
  };
}

async function saveAmbientSettings({ enabled = null, automaticCaptureEnabled = null, hostname = "", siteEnabled = null } = {}) {
  const current = await ambientSettings(hostname);
  const disabledSites = { ...current.disabled_sites };
  if (hostname && siteEnabled !== null) {
    if (siteEnabled) delete disabledSites[hostname];
    else disabledSites[hostname] = true;
  }
  const next = {
    enabled: enabled === null ? current.enabled : Boolean(enabled),
    automatic_capture_enabled: automaticCaptureEnabled === null
      ? current.automatic_capture_enabled
      : Boolean(automaticCaptureEnabled),
    disabled_sites: disabledSites,
  };
  await runtimeChrome.storage.local.set({ [AMBIENT_SETTINGS_KEY]: next });
  return ambientSettings(hostname);
}

async function automaticCaptureEnabled() {
  return (await ambientSettings()).automatic_capture_enabled;
}

function sessionKey(provider, providerSessionId) {
  return `${provider || "unknown"}:${providerSessionId || "unknown"}`;
}

function retryDelayForAttempt(attempts) {
  const delay = CAPTURE_RETRY_BASE_DELAY_MS * 2 ** Math.max(0, attempts);
  return Math.min(delay, CAPTURE_RETRY_MAX_DELAY_MS);
}

function envelopeSessionSummary(envelope) {
  if (envelope.capture_summary) return envelope.capture_summary;
  const session = envelope?.session || {};
  return {
    title: session.title || null,
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

const captureRetryStore = new IndexedDbBackfillStore();
let captureRetryInitialization = null;

function captureRetryMetadata(entry) {
  const { envelope, ...metadata } = entry;
  if (!envelope) return metadata;
  const summary = envelopeSessionSummary(envelope);
  return { ...metadata, provider: summary.provider, provider_session_id: summary.providerSessionId,
    title: envelope.session?.title || null, capture_fidelity: summary.captureMode,
    turn_count: summary.turnCount, attachment_count: summary.attachmentCount };
}

async function cacheCaptureQueueState(queue) {
  // The atomic IDB import marker and delivery roots own acquired bytes. A
  // failed derived Chrome cache must not prevent their retry admission.
  try { await runtimeChrome.storage.local.set({ [CAPTURE_QUEUE_KEY]: queue }); }
  catch (error) {
    await appendDebugLog({ stage: "capture_queue_cache_failed", error: String(error.message || error) }).catch(() => undefined);
  }
}

async function initializeCaptureRetries() {
  if (!captureRetryInitialization) {
    captureRetryInitialization = (async () => {
      const stored = await runtimeChrome.storage.local.get({ [CAPTURE_QUEUE_KEY]: CAPTURE_QUEUE_EMPTY });
      const previous = stored[CAPTURE_QUEUE_KEY];
      // Transfer original retry inputs once, atomically with the completion
      // marker. The body-free local value is only a derived status cache.
      await captureRetryStore.importCaptureRetries((previous?.entries || []).map((entry) => ({
        metadata: captureRetryMetadata(entry), envelope: entry.envelope,
      })));
      const entries = await captureRetryStore.listCaptureRetries();
      const queue = { entries, dropped_count: previous?.dropped_count || 0 };
      await cacheCaptureQueueState(queue);
      return queue.dropped_count;
    })().catch((error) => { captureRetryInitialization = null; throw error; });
  }
  return captureRetryInitialization;
}

async function getCaptureQueue() {
  if (!queueConversionPromise) queueConversionPromise = (async () => {
    // Import the original local inputs once under5918's existing atomic marker.
    // A metadata-only cache is never the authority for their acquired bodies.
    const dropped = await initializeCaptureRetries();
    for await (const metadata of captureRetryStore.captureRetryInputs()) {
      const envelope = await captureRetryStore.getCaptureRetryEnvelope(metadata.id);
      const delivery = { ...metadata, delivery_kind: "foreground", source_retry_id: metadata.id, summary: envelopeSessionSummary(envelope) };
      const prepared = await captureStaging.prepare(envelope, captureStaging.conversionId(metadata.id), delivery);
      const converted = { ...delivery, body_ref: prepared.ref };
      await captureStaging.file(converted.body_ref);
      const existing = await captureStore.getDelivery(converted.id);
      if (existing && existing.body_ref !== converted.body_ref) throw new Error("capture_queue_conversion_conflict");
      if (prepared.acknowledgedReceipt) {
        // Original exact body receipt prevents a restart from resurrecting a
        // delivery settled before source retirement completed.
        await captureStore.deleteDelivery(converted.id);
      } else {
        if (!existing) await captureStore.putDelivery(converted);
        const published = await captureStore.getDelivery(converted.id);
        if (published?.body_ref !== converted.body_ref) throw new Error("capture_queue_conversion_unpublished");
      }
      // Published custody precedes atomic source retirement. Any failure keeps
      // either the original input or its exact idempotently published delivery.
      await captureRetryStore.retireCaptureRetry(metadata.id);
      if (prepared.acknowledgedReceipt) await captureStaging.acknowledge(converted.body_ref);
    }
    await cacheCaptureQueueState({ version: 3, dropped_count: dropped });
    return { version: 3, total: await captureStore.deliveryCount(), dropped_count: dropped };
  })();
  try { return await queueConversionPromise; } finally { queueConversionPromise = null; }
}

async function recoverStagedDeliveries() {
  await getCaptureQueue();
  for await (const { entry } of captureStore.deliveries()) {
    if (!entry.preparing || entry.held) continue;
    try { await captureStaging.resumeDelivery(entry); }
    catch (error) {
      await recordCaptureDeliveryFailure(entry, error);
      await appendDebugLog({ stage: "capture_body_recovery_pending", queued_id: entry.id, error: String(typeof error.code === "string" ? error.code : (error.message || error)) });
    }
  }
  for await (let meta of captureStaging.metadataEntries()) {
    if (meta.state === "metadata-interrupted") {
      await appendDebugLog({ stage: "capture_staging_metadata_interrupted", ref: meta.id }); continue;
    }
    if (meta.state === "receiver-preparing") {
      try { meta = await captureStaging.completePreparedBody(meta.id); }
      catch (error) { await appendDebugLog({ stage: "capture_staging_publication_pending", ref: meta.id, error: String(typeof error.code === "string" ? error.code : (error.message || error)) }); continue; }
    }
    if (meta.delivery?.delivery_kind !== "foreground" || !["receiver-ready", "receiver-acknowledged"].includes(meta.state)) continue;
    await captureStaging.file(meta.id);
    const existing = await captureStore.getDelivery(meta.delivery.id);
    if (existing && existing.body_ref !== meta.id) throw new Error("capture_delivery_reference_conflict");
    if (meta.state === "receiver-acknowledged") {
      await captureStore.deleteDelivery(meta.delivery.id);
      await captureStaging.acknowledge(meta.id); continue;
    }
    if (!existing) await captureStore.putDelivery({ ...meta.delivery, body_ref: meta.id });
  }
}

async function tabIfPresent(tabId) {
  try { return await runtimeChrome.tabs.get(tabId); }
  catch (error) {
    // Only Chrome's positive absence result permits custody retirement or
    // forgetting an owned transport. Read/extension faults preserve both.
    const absent = String(error?.message || error).match(/^No tab with id:\s*(\d+)\.?$/i);
    if (absent && Number(absent[1]) === tabId) return null;
    throw error;
  }
}

async function reconcileCaptureRoots() {
  await recoverStagedDeliveries();
  for await (const capture of captureStore.captures()) {
    if (capture.kind === "native-invocation" && capture.worker_owner !== runtimeWorkerId) await settleNativeInvocation(capture)
      .catch((error) => appendDebugLog({ stage: "native_invocation_settlement_pending", ref: capture.id, error: String(error.message || error) }));
    if (capture.delivery_kind === "foreground" && capture.state === "ready" && capture.raw_ref) {
      await retainCaptureForDelivery({ envelope: nativeNormalizer.envelope(capture), reason: "native_publication_recovery" })
        .catch((error) => appendDebugLog({ stage: "native_delivery_recovery_pending", ref: capture.id, error: String(typeof error.code === "string" ? error.code : (error.message || error)) }));
    }
    if (capture.state === "ready" && capture.raw_ref) {
      await captureStaging.retireFailedNormalizations(capture.raw_ref, capture.id)
        .catch((error) => appendDebugLog({ stage: "native_normalization_cleanup_pending", ref: capture.id, error: String(typeof error.code === "string" ? error.code : (error.message || error)) }));
    }
    if (!["native-cache", "dom-observation"].includes(capture.kind)) continue;
    const owner = capture.owner;
    const tab = await tabIfPresent(owner.tab_id);
    let gone = !tab;
    if (!gone && owner.document_id) {
      try {
        await runtimeChrome.tabs.sendMessage(owner.tab_id, { type: "polylogue.stagingOwner" }, { documentId: owner.document_id });
      } catch (error) {
        // A transient extension transport failure does not establish document
        // loss. Chrome's targeted document lookup does.
        gone = /no document with id/i.test(String(error.message || error));
      }
    }
    if (gone) {
      for await (const ref of captureStore.releaseNativeCaches(owner)) await captureStaging.discardUnreferenced(ref?.id || ref);
    }
  }
  for await (const meta of captureStaging.metadataEntries()) {
    if (meta.state === "checkpoint-acknowledged" && meta.owner?.checkpoint_snapshot_id) {
      const snapshot = await captureStore.getCapture(meta.owner.checkpoint_snapshot_id);
      if (snapshot?.kind === "checkpoint-export") {
        await captureStore.acknowledgeExportSnapshot(snapshot, meta.receiver_receipt);
        await captureStaging.discardUnreferenced(meta.id);
      }
    }
    if (meta.kind === "native-response" && !meta.capture_bundle && ["acquiring", "sealed"].includes(meta.state)) {
      await captureStaging.reconcileNativeAcquisition(meta.id);
    }
    if (meta.state === "receiver-acknowledged") {
      const delivery = meta.delivery;
      if (delivery?.delivery_kind === "foreground" && !await captureStore.getDelivery(delivery.id)) await captureStaging.acknowledge(meta.id);
      if (delivery?.delivery_kind === "backfill") {
        const item = await captureStore.getQueue(delivery.id);
        if (item && ["complete", "superseded", "cancelled"].includes(item.state)) await captureStaging.acknowledge(meta.id);
      }
    }
    // Unpublished acquisition remains owned while its document exists, and
    // failed normalization rows retain their evidence for a visible retry.
    if ((meta.acquisition && meta.acquisition_result) || (meta.capture_bundle && meta.state === "sealed")) {
      try { await captureStaging.seal({ id: meta.id, token: meta.token }, meta.owner); }
      catch (error) { await appendDebugLog({ stage: "capture_acquisition_recovery_pending", ref: meta.id, error: String(typeof error.code === "string" ? error.code : (error.message || error)) }); }
    }
    if (meta.retired) { await captureStaging.discardUnreferenced(meta.id); continue; }
    if (!meta.owner?.tab_id || await captureStaging.referenced(meta.id)) continue;
    const tab = await tabIfPresent(meta.owner.tab_id);
    if (!tab) await captureStaging.discardUnreferenced(meta.id);
  }
}

async function captureQueuePage({ cursor: after = null, pageSize = 50 } = {}) {
  const queue = await getCaptureQueue();
  if (!Number.isSafeInteger(pageSize) || pageSize <= 0) throw new Error("capture_queue_page_size_invalid");
  const entries = [];
  let pendingAcquisitions = 0;
  for await (const capture of captureStore.captures()) {
    if (["native-acquisition", "native-bundle"].includes(capture.kind) && !capture.queue_context) pendingAcquisitions += 1;
  }
  let nextCursor = null;
  for await (const { entry, cursor } of captureStore.deliveries({ after })) {
    if (entries.length === pageSize) { nextCursor = entries.at(-1).cursor; break; }
    let tabOrigin = null;
    try { tabOrigin = entry.tab_url ? new globalThis.URL(entry.tab_url).origin : null; } catch { /* No origin for an invalid retained locator. */ }
    entries.push({ id: entry.id, reason: entry.reason, enqueued_at: entry.enqueued_at,
      attempts: entry.attempts, next_attempt_at: entry.next_attempt_at, last_error: entry.last_error,
      tab_origin: tabOrigin,
      summary: entry.summary, provider: entry.summary?.provider || null,
      provider_session_id: entry.summary?.providerSessionId || null, cursor });
  }
  return { total: queue.total, dropped_count: queue.dropped_count, pending_acquisition_count: pendingAcquisitions, entries, next_cursor: nextCursor };
}

async function refreshQueueBadge() {
  const stored = await runtimeChrome.storage.local.get({ polylogueState: {} });
  const badge = badgeForState(stored.polylogueState || {});
  await runtimeChrome.action.setBadgeText({ text: badge.text });
  await runtimeChrome.action.setBadgeBackgroundColor({ color: badge.color });
}

async function refreshCaptureQueueCount() {
  const queue = await getCaptureQueue();
  cachedQueueLength = queue.total;
  await refreshQueueBadge();
  return queue.total;
}

async function ensureRetryAlarm() {
  if (!runtimeChrome.alarms?.create) return;
  await runtimeChrome.alarms.create(CAPTURE_RETRY_ALARM, {
    delayInMinutes: CAPTURE_RETRY_ALARM_PERIOD_MINUTES,
    periodInMinutes: CAPTURE_RETRY_ALARM_PERIOD_MINUTES,
  });
}

async function clearRetryAlarm() {
  if (!runtimeChrome.alarms?.clear) return;
  await runtimeChrome.alarms.clear(CAPTURE_RETRY_ALARM);
}

function captureDeliveryOwner(sender) {
  return { extension_id: sender.id || runtimeChrome.runtime.id,
    tab_id: sender.tab?.id ?? null, document_id: sender.documentId || null };
}

function isRetryableCaptureError(error) {
  if (!error) return false;
  if (["capture_delivery_preparation_missing", "capture_staging_asset_unsealed", "capture_staging_invalid_ref", "capture_staging_owner_mismatch", "capture_staging_sequence_mismatch"].includes(error.code) || error.name === "NotFoundError") return false;
  if (error.code === "receiver_contract_incompatible" || error.code === "capture_cancelled" || error.name === "AbortError" || error.name === "QuotaExceededError") return false;
  if (typeof error.status === "number") return error.status >= 500 || error.status === 429;
  // No HTTP status means fetch itself rejected (offline, DNS failure, refused
  // connection, CORS) rather than the receiver answering with an error body.
  return true;
}

async function retainCaptureForDelivery({ envelope, reason, tab = null, signal = null, operation = null }) {
  return serializeCaptureQueueMutation(async () => {
  envelope = await withExtensionInstanceAttribution(envelope);
  if (envelope.capture_record_ref) {
    const root = await captureStore.foregroundDeliveryRoot(envelope.capture_record_ref);
    if (root?.delivery) {
      if (operation) operation.deliveryId = root.delivery.id;
      return root.delivery;
    }
    if (root?.body) {
      let meta = await captureStaging.metadata(root.body_ref);
      if (meta.state === "receiver-preparing") meta = await captureStaging.completePreparedBody(meta.id, signal);
      if (!meta.delivery || !["receiver-ready", "receiver-acknowledged"].includes(meta.state)) throw new Error("capture_delivery_preparation_missing");
      if (meta.state === "receiver-acknowledged") throw new Error("capture_delivery_already_acknowledged");
      const retained = { ...meta.delivery, body_ref: meta.id, capture_record_ref: envelope.capture_record_ref };
      if (operation) operation.deliveryId = retained.id;
      await captureStore.putDelivery(retained); await refreshCaptureQueueCount(); await ensureRetryAlarm(); return retained;
    }
  }
  const entry = {
    id: buildReceiverRequestId(), delivery_kind: "foreground",
    summary: envelopeSessionSummary(envelope),
    reason: reason || "content_script_capture",
    tab_id: tab?.id || null,
    tab_url: tab?.url || tab?.pendingUrl || null,
    enqueued_at: new Date().toISOString(),
    attempts: 0,
    next_attempt_at: new Date(Date.now() + retryDelayForAttempt(0)).toISOString(),
    last_error: null,
    capture_record_ref: envelope.capture_record_ref || null,
    observation_ref: envelope.capture_observation_ref || null,
  };
  if (operation) operation.deliveryId = entry.id;
  if (envelope.receiver_native) {
    entry.receiver_native = envelope.receiver_native;
    await captureStore.putDelivery(entry); await refreshCaptureQueueCount(); await ensureRetryAlarm(); return entry;
  }
  let prepared;
  try { prepared = await captureStaging.prepare(envelope, null, entry, signal); }
  catch (error) { error.captureDeliveryId = entry.id; await refreshCaptureQueueCount(); throw error; }
  entry.body_ref = prepared.ref;
  await getCaptureQueue();
  await captureStore.putDelivery(entry);
  await refreshCaptureQueueCount();
  await ensureRetryAlarm();
  return entry;
  }, signal);
}

async function drainOwnedNativeBundles(owner, nativeId, beforeSequence, signal) {
  for await (const bundle of captureStore.captures()) {
    signal.throwIfAborted();
    if (bundle.kind !== "native-bundle" || bundle.queue_context || bundle.state !== "ready" ||
        bundle.native_id !== nativeId || JSON.stringify(bundle.owner) !== JSON.stringify(owner) ||
        bundle.acquisition_sequence >= beforeSequence) continue;
    try {
      const completed = await nativeNormalizer.finishBundle(bundle.id, owner, { signal });
      const envelope = await nativeNormalizer.normalize({ provider: bundle.provider, rawRef: completed.rawRef,
        relatedRefs: completed.relatedRefs, acquisition: completed.acquisition, nativeId,
        extensionVersion: runtimeChrome.runtime.getManifest().version, instanceId: await extensionInstanceId(), signal });
      await retainCaptureForDelivery({ envelope, reason: "owned_acquisition_recovery", signal });
    } catch (error) {
      await appendDebugLog({ stage: "native_bundle_normalization_pending", ref: bundle.id, error: String(typeof error.code === "string" ? error.code : (error.message || error)) });
      throw error;
    }
  }
}

async function recordCaptureDeliveryFailure(entry, error) {
  return serializeCaptureQueueMutation(async () => {
    await getCaptureQueue();
    const failure = String(error.message || error);
    const current = await captureStore.getDelivery(entry.id);
    if (!current) throw new Error("capture_delivery_reference_missing");
    const retryable = isRetryableCaptureError(error);
    await captureStore.putDelivery({ ...current, held: !retryable,
      next_attempt_at: retryable ? current.next_attempt_at : null, last_error: failure });
    const total = await refreshCaptureQueueCount();
    await appendCaptureLog({ ok: false, reason: "capture_queued_for_retry", queued_id: entry.id, error: failure, queue_length: total });
  });
}

async function retireCaptureDelivery(entry) {
  return serializeCaptureQueueMutation(async () => {
    await getCaptureQueue();
    // ACK publication precedes retirement of the delivery root.
    await captureStore.deleteDelivery(entry.id);
    await refreshCaptureQueueCount();
    if (entry.receiver_native) return captureStaging.acknowledgeNative(entry.capture_record_ref);
    try { await captureStaging.metadata(entry.body_ref); }
    catch (error) { if (error.name === "NotFoundError") return; throw error; }
    await captureStaging.acknowledge(entry.body_ref);
  });
}

async function completeForegroundDelivery(delivery, signal) {
  signal.throwIfAborted();
  let completion = foregroundDeliveryCompletions.get(delivery.id);
  if (completion?.controller.signal.aborted && !completion.settled) {
    // Closing admission belongs to the cancelled physical attempt. A new
    // invocation waits its settlement without becoming an owner of that abort.
    const receipt = await new Promise((resolve, reject) => {
      const abort = () => reject(signal.reason);
      signal.addEventListener("abort", abort, { once: true });
      if (signal.aborted) abort();
      completion.promise.then(resolve, () => resolve(null))
        .finally(() => signal.removeEventListener("abort", abort));
    });
    signal.throwIfAborted();
    if (receipt) return receipt;
    if (foregroundDeliveryCompletions.get(delivery.id) === completion) foregroundDeliveryCompletions.delete(delivery.id);
    return completeForegroundDelivery(delivery, signal);
  }
  if (!completion) {
    completion = { controller: new globalThis.AbortController(), participants: new Set(), settled: false, promise: null };
    foregroundDeliveryCompletions.set(delivery.id, completion);
    completion.promise = Promise.resolve().then(async () => {
      const receipt = await postJson("/v1/browser-captures", delivery.receiver_native ? { receiver_native: delivery.receiver_native, capture_record_ref: delivery.capture_record_ref } : { capture_body_ref: delivery.body_ref }, null, true, completion.controller.signal);
      // ACK is durable already. Retirement may be queued behind an unrelated
      // upload and must not make this invocation own that physical settlement.
      void retireCaptureDelivery(delivery).catch((error) => appendDebugLog({
        stage: "capture_ack_cleanup_pending", error: String(error.message || error),
      }).catch(() => undefined));
      return receipt;
    }).finally(() => {
      completion.settled = true;
      if (!completion.participants.size && foregroundDeliveryCompletions.get(delivery.id) === completion) foregroundDeliveryCompletions.delete(delivery.id);
    });
  }
  const participant = {};
  completion.participants.add(participant);
  try {
    return await new Promise((resolve, reject) => {
      const abort = () => {
        completion.participants.delete(participant);
        if (completion.participants.size) {
          // This invocation owns no other caller's progressing upload.
          const error = new Error("capture_cancelled"); error.name = "AbortError"; error.code = "capture_cancelled";
          error.sharedDeliveryContinues = true;
          reject(error);
        } else {
          completion.controller.abort(signal.reason);
          // The final owner drains the physical request. A validated ACK wins.
          completion.promise.then(resolve, () => reject(signal.reason));
        }
      };
      signal.addEventListener("abort", abort, { once: true });
      if (signal.aborted) abort();
      completion.promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
    });
  } finally {
    completion.participants.delete(participant);
    if (completion.settled && !completion.participants.size && foregroundDeliveryCompletions.get(delivery.id) === completion) foregroundDeliveryCompletions.delete(delivery.id);
  }
}

async function recordSupersededCapture(summary, receipt, reason) {
  await updateSessionLedger({
    provider: summary.provider, providerSessionId: summary.providerSessionId,
    patch: { receiver_request_id: receipt.receiver_request_id || null,
      artifact_ref: receipt.artifact_ref || null, last_error: "receiver_superseded" },
  });
  await appendConversationTimeline({
    provider: summary.provider, providerSessionId: summary.providerSessionId,
    event: "held_with_reason", reason, detail: "receiver_superseded",
  });
}

async function drainCaptureQueue(trigger = "alarm") {
  return serializeCaptureQueueMutation(async () => {
  const queue = await getCaptureQueue();
  if (!queue.total) {
    await clearRetryAlarm();
    return { drained: 0, remaining: 0 };
  }
  const now = Date.now();
  let drained = 0;
  for await (const { entry } of captureStore.deliveries({ dueAt: now })) {
    let activelyOwned = false;
    for (const operation of captureDeliveries.values()) {
      if (operation.deliveryId === entry.id) { activelyOwned = true; break; }
    }
    if (activelyOwned) continue;
    if (entry.held && trigger !== "manual") continue;
    const dueAt = Date.parse(entry.next_attempt_at || "") || 0;
    if (dueAt > now) {
      continue;
    }
    const envelope = entry.receiver_native ? { receiver_native: entry.receiver_native, capture_record_ref: entry.capture_record_ref } : { capture_body_ref: entry.body_ref };
    const summary = entry.summary;
    let acknowledged = false;
    try {
      await captureStaging.resumeDelivery(entry);
      const result = await postJson("/v1/browser-captures", envelope);
      acknowledged = true;
      drained += 1;
      await captureStore.deleteDelivery(entry.id);
      if (entry.receiver_native) await captureStaging.acknowledgeNative(entry.capture_record_ref);
      else await captureStaging.acknowledge(entry.body_ref);
      if (result.outcome === "superseded") {
        await recordSupersededCapture(summary, result, "capture_retry_superseded");
        continue;
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
      if (acknowledged) {
        // A post-ACK cleanup or telemetry failure cannot recreate delivery.
        // The durable receipt lets startup finish retirement without another POST.
        await appendDebugLog({ event: "capture_ack_cleanup_pending", stage_id: entry.body_ref, error: String(error.message || error) });
        continue;
      }
      if (!isRetryableCaptureError(error)) {
        await captureStore.putDelivery({ ...entry, next_attempt_at: null, held: true, last_error: String(error.message || error) });
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
      await captureStore.putDelivery({
        ...entry,
        attempts,
        held: false,
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
  const remaining = await refreshCaptureQueueCount();
  if (!remaining) {
    await clearRetryAlarm();
  } else if (trigger === "alarm") {
    await appendDebugLog({ stage: "capture_retry_drain", drained, remaining });
  }
  return { drained, remaining };
  });
}

async function loadCaptureQueueIntoCache() {
  const queue = await getCaptureQueue();
  cachedQueueLength = queue.total;
  if (queue.total) await ensureRetryAlarm();
  return queue;
}

async function probeReceiverStatus(baseUrl) {
  const requestId = buildReceiverRequestId();
  const controller = new globalThis.AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort("receiver_health_timeout"), RECEIVER_HEALTH_TIMEOUT_MS);
  await appendDebugLog({ stage: "receiver_request", method: "GET", path: "/v1/status", endpoint: baseUrl, request_id: requestId });
  try {
    const headers = { "X-Request-ID": requestId };
    const response = await runtimeNetwork(`${baseUrl}/v1/status`, { headers, signal: controller.signal });
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

async function checkReceiverHealth({ allowCanonicalRecovery = true, expectedScope = null } = {}) {
  let scope = await receiverHealthScope();
  if (expectedScope && !await receiverScopeIsCurrent(expectedScope)) return { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint: scope.settings.baseUrl, pairing: null };
  const settings = scope.settings;
  const pairingBefore = scope.pairing;

  async function classifyProbe(endpoint, probe, recoveredFrom = null) {
    if (!await receiverScopeIsCurrent(scope)) return { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint, pairing: null };
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
      if (!await receiverScopeIsCurrent(scope)) return { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint, pairing: null };
      const pairing = pairingBefore
        ? await serializeStorageMutation(async () => await receiverScopeIsCurrent(scope) ? persistReceiverPairing({
          ...pairingBefore,
          state: "unauthorized",
          checked_at: new Date().toISOString(),
          last_error: "unauthorized",
        }) : null)
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
      const pairing = await observeReceiverIdentity(body, endpoint, scope);
      if (!pairing || !await receiverScopeIsCurrent(scope)) return { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint, pairing: null };
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
      const published = await serializeStorageMutation(async () => {
        if (!await receiverScopeIsCurrent(scope)) return false;
        if (pairing?.receiver_id && pairing.state === "online") {
          trustedReceiverHealthCache = {
            checkedAt: Date.now(),
            endpoint,
            receiverId: pairing.receiver_id,
            apiSchema: pairing.api_schema,
            health: result,
          };
        }
        return true;
      });
      return published ? result : { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint, pairing: null };
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
    const primary = await probeReceiverStatus(settings.baseUrl);
    const classified = await classifyProbe(settings.baseUrl, primary);
    if (classified.status !== "unreachable") return classified;
    primaryFailure = classified.detail || "receiver_unavailable";
  } catch (error) {
    primaryFailure = String(error.message || error);
    if (["receiver_identity_mismatch", "receiver_authentication_failed"].includes(primaryFailure)) {
      const mismatch = primaryFailure === "receiver_identity_mismatch";
      const pairing = await serializeStorageMutation(async () => {
        if (!await receiverScopeIsCurrent(scope)) return null;
        trustedReceiverHealthCache = null;
        return persistReceiverPairing({ ...pairingBefore, endpoint: settings.baseUrl,
          state: mismatch ? "mismatch" : "unauthorized", last_error: primaryFailure,
          checked_at: new Date().toISOString() });
      });
      if (!await receiverScopeIsCurrent(scope)) return { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint: settings.baseUrl, pairing: null };
      return { ok: false, status: mismatch ? "pairing_mismatch" : "unauthorized", detail: primaryFailure, endpoint: settings.baseUrl, pairing };
    }
  }

  if (!await receiverScopeIsCurrent(scope)) return { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint: settings.baseUrl, pairing: null };

  if (
    allowCanonicalRecovery
    && settings.baseUrl !== DEFAULT_RECEIVER
    && pairingBefore?.receiver_id
    // A deliberate dev-override pairing must not silently self-heal to the
    // canonical endpoint: that is precisely the "quiet drift" this bounded
    // recovery exists to fix for an *accidental* stale pairing, but here the
    // non-canonical endpoint is the operator's explicit choice. Report the
    // failure loudly instead (see markReceiverPairingUnavailable) and
    // require an explicit reset/settings change, same as an identity
    // mismatch.
    && !pairingBefore?.dev_override
  ) {
    try {
      const canonical = await probeReceiverStatus(DEFAULT_RECEIVER);
      const body = canonical.body;
      if (
        body?.ok === true
        && canonical.response?.ok !== false
        && body.receiver_id === pairingBefore.receiver_id
        && body.api_schema === RECEIVER_API_SCHEMA
      ) {
        const recovered = await serializeStorageMutation(async () => {
          if (!await receiverScopeIsCurrent(scope)) return null;
          receiverConfigurationRevision += 1;
          trustedReceiverHealthCache = null;
          await runtimeChrome.storage.local.set({ receiverBaseUrl: DEFAULT_RECEIVER });
          return { settings: await receiverSettings(), revision: receiverConfigurationRevision };
        });
        if (recovered) { scope = recovered; return classifyProbe(DEFAULT_RECEIVER, canonical, settings.baseUrl); }
      }
    } catch {
      // Recovery is intentionally bounded to one canonical endpoint. The
      // original failure remains the operator-facing result.
    }
  }

  const pairing = await markReceiverPairingUnavailable(primaryFailure, scope);
  if (!await receiverScopeIsCurrent(scope)) return { ok: false, status: "error", detail: "receiver_configuration_changed", endpoint: settings.baseUrl, pairing: null };
  return {
    ok: false,
    status: pairing?.state === "dev_override_stale" ? "dev_override_stale" : "unreachable",
    detail: primaryFailure,
    endpoint: settings.baseUrl,
    pairing,
  };
}

async function appendCaptureLog(entry) {
  const stored = await runtimeChrome.storage.local.get({ polylogueCaptureLog: [] });
  const prior = Array.isArray(stored.polylogueCaptureLog) ? stored.polylogueCaptureLog : [];
  const next = [
    {
      at: new Date().toISOString(),
      ...entry,
    },
    ...prior,
  ].slice(0, CAPTURE_LOG_LIMIT);
  await runtimeChrome.storage.local.set({ polylogueCaptureLog: next });
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
  const stored = await runtimeChrome.storage.local.get({ polylogueDebugLog: [] });
  const prior = Array.isArray(stored.polylogueDebugLog) ? stored.polylogueDebugLog : [];
  const next = [
    {
      at: new Date().toISOString(),
      ...sanitizeDebugDetails(entry),
    },
    ...prior,
  ].slice(0, DEBUG_LOG_LIMIT);
  await runtimeChrome.storage.local.set({ polylogueDebugLog: next });
  return next;
}

// Best-effort private tracing only: no receipt, identity or effect authority.
function nativePreparationProgress(context, phase, state) {
  if (!context || typeof context.native_request_id !== "string" || !/^polylogue-native-fetch-\d+-[a-z0-9]+$/.test(context.native_request_id) || !["normalize_admission", "native_prepare", "native_assets", "native_finalize"].includes(phase) || !["BEGIN", "END"].includes(state)) return;
  void appendDebugLog({ stage: "native_preparation_progress", ...context, phase, state }).catch(() => undefined);
}

async function updateSessionLedger({ provider, providerSessionId, patch }) {
  if (!provider || !providerSessionId) return null;
  return serializeStorageMutation(async () => {
    const stored = await runtimeChrome.storage.local.get({ polylogueSessionLedger: {} });
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
    await runtimeChrome.storage.local.set({ polylogueSessionLedger: { ...ledger, [key]: next } });
    return next;
  });
}

async function appendConversationTimeline({ provider, providerSessionId, event, reason = null, detail = null, tabId = null, onlyIfEmpty = false, dedupeWindowMs = 0 }) {
  if (!provider || !providerSessionId) return null;
  return serializeStorageMutation(async () => {
    const stored = await runtimeChrome.storage.local.get({ [CONVERSATION_TIMELINE_KEY]: {} });
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
    await runtimeChrome.storage.local.set({ [CONVERSATION_TIMELINE_KEY]: next });
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
  await runtimeChrome.storage.local.set({ polylogueState: nextState });
  const badge = badgeForState(nextState);
  await runtimeChrome.action.setBadgeText({ text: badge.text });
  await runtimeChrome.action.setBadgeBackgroundColor({ color: badge.color });
}

async function setStateForTab(tabId, state, expectedTabUrl = null) {
  if (!tabId || !runtimeChrome.tabs?.query) return setState(state);
  const [activeTab] = await runtimeChrome.tabs.query({ active: true, currentWindow: true });
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
    // TEMPORARY_CHAT_SENTINEL is a "this tab has a real, capturable
    // conversation" signal, not the conversation's real identity -- it can
    // never equal a captured state's true provider_session_id (the
    // ephemeral id from the native payload), so this staleness guard would
    // reject every state write for a temporary chat tab, including the
    // capture that just succeeded on it.
    && activeSessionId !== TEMPORARY_CHAT_SENTINEL
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
  const headers = {};
  if (hasBody) headers["Content-Type"] = "application/json";
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

// A page capture reads authenticated provider data before its envelope can be
// submitted to the receiver.  Do not begin that irreversible read merely to
// discover later that this browser has never established a receiver identity.
// `checkReceiverHealth` remains available for pairing diagnostics; this guard
// is specifically the admission boundary for automatic provider-native work.
async function requirePairedTrustedReceiver() {
  const pairing = await storedReceiverPairing();
  if (!pairing?.receiver_id || !pairing?.api_schema) {
    const error = new Error("receiver_unpaired");
    error.code = "receiver_unpaired";
    error.status = 401;
    throw error;
  }
  return ensureTrustedReceiver();
}

async function prepareNativeCapture(capture, { rawRef, relatedRefs, queueContext, signal, summaryOnly = false, onProgress = null }) {
  const progress = (phase, state) => { try { onProgress?.(phase, state); } catch { /* Diagnostics cannot change capture. */ } };
  progress("native_prepare", "BEGIN");
  await ensureTrustedReceiver();
  const settings = await receiverSettings();
  const client = new CaptureJobClient({ baseUrl: settings.baseUrl, cache: runtimeChrome.storage.local, fetchImpl: runtimeNetwork });
  client.nativeOwnerChanged = async (adopted) => {
    const retained = await captureStore.getCapture(capture.id);
    if (!retained) throw new Error("native_preparation_owner_missing");
    await captureStore.putCapture({ ...retained, receiver_native: { ...retained.receiver_native, adopted,
      acquisition_id: capture.acquisition_id, preparation_instance_id: capture.preparation_instance_id } });
  };
  const refs = capture.provider === "grok" ? { responses: rawRef, ...relatedRefs } : { conversation: rawRef };
  const members = {};
  for (const [name, ref] of Object.entries(refs)) {
    const meta = await captureStaging.metadata(ref.id);
    captureStaging.requireOwner(meta, ref, capture.owner);
    if (meta.state !== "sealed") throw new Error("capture_staging_asset_unsealed");
    members[name] = { sha256: meta.sha256, size_bytes: meta.bytes };
  }
  const rawRevision = await client.nativeRevision(capture.provider, capture.native_id, members);
  const binding = { preparation_instance_id: capture.preparation_instance_id,
    extension_instance_id: capture.extension_instance_id, acquisition_sequence: capture.acquisition_sequence,
    invocation_id: capture.invocation_id, raw_revision: rawRevision, native_id: capture.native_id,
    source_url: capture.source_url, document_id: capture.owner.document_id };
  let adopted;
  if (capture.receiver_native?.adopted) adopted = await client.refreshNativeOwner(capture.receiver_native.adopted, capture.preparation_instance_id, signal);
  else if (queueContext) {
    const job = await captureStore.assertJobExecution(queueContext.jobId, queueContext.owner, queueContext.generation);
    const accountHandle = await providerAccountHandle(capture.provider);
    if (!accountHandle) throw new Error("capture_job_account_scope_unresolved");
    adopted = await client.recoverOrCreate({ provider: capture.provider, accountHandle,
      locator: { kind: "backfill", provider: capture.provider, cutoff: job.cutoff },
      intentPayload: { provider: capture.provider, cutoff: job.cutoff }, sessionId: capture.preparation_instance_id });
    if (adopted.scope.key !== job.account_scope) throw new Error("capture_job_account_scope_mismatch");
  } else adopted = await client.recoverInvocation({ provider: capture.provider, creationToken: capture.preparation_token,
    binding, sessionId: capture.preparation_instance_id });
  const reference = { adopted, acquisition_id: capture.acquisition_id, preparation_instance_id: capture.preparation_instance_id };
  capture = { ...capture, receiver_native: reference, raw_revision_sha256: rawRevision };
  await captureStore.putCapture(capture);
  await client.beginNative(adopted, capture.acquisition_id, binding, Object.keys(refs), signal);
  for (const [name, ref] of Object.entries(refs)) {
    adopted = await client.refreshNativeOwner(adopted, capture.preparation_instance_id, signal);
    const meta = await captureStaging.metadata(ref.id);
    await client.nativeMember(adopted, capture.acquisition_id, name, await captureStaging.file(ref.id), meta.response_metadata || {}, meta.sha256, signal);
  }
  adopted = await client.refreshNativeOwner(adopted, capture.preparation_instance_id, signal);
  const prepared = await client.prepareNative(adopted, capture.acquisition_id, {
    captured_at: capture.observed_at, extension_instance_id: capture.extension_instance_id,
    acquisition_sequence: capture.acquisition_sequence, source_url: capture.source_url,
    extension_id: runtimeChrome.runtime.id || null, adapter_name: `${capture.provider}-native-v1`,
    adapter_version: capture.extension_version, capture_mode: "snapshot", provider_meta: capture.attribution },
    { capture_fidelity: "native_full", ...capture.attribution }, signal);
  const preparedRoot = await captureStore.getCapture(capture.id);
  await captureStore.putCapture({ ...preparedRoot, receiver_summary: { raw_revision: rawRevision, plan_digest: prepared.plan_digest, summary: prepared.summary } });
  progress("native_prepare", "END");
  if (summaryOnly) return { rawRevision, summary: prepared.summary, reference: { ...reference, adopted, plan_digest: prepared.plan_digest } };
  progress("native_assets", "BEGIN");
  for await (const asset of client.nativePlan(adopted, capture.acquisition_id, capture.preparation_instance_id, signal)) {
    if (asset.receipt) continue;
    const descriptor = asset.descriptor;
    let outcome = { status: "no_resolvable_source" }; let file = null; let sha256 = null;
    if (descriptor.provider_meta.native_inline_sha256) {
      outcome = { status: "retained_native_bytes", sha256: descriptor.provider_meta.native_inline_sha256, size_bytes: descriptor.provider_meta.native_inline_size_bytes };
    } else if (capture.provider !== "claude-ai") {
      signal?.throwIfAborted();
      const cancel = () => { void runtimeChrome.tabs.sendMessage(capture.owner.tab_id, { type: "polylogue.cancelRecordAssets", capture_ref: rawRef.id }, { documentId: capture.owner.document_id }).catch(() => undefined); };
      signal?.addEventListener("abort", cancel, { once: true });
      try {
        const result = await runtimeChrome.tabs.sendMessage(capture.owner.tab_id, { type: "polylogue.acquireRecordAssets",
          provider: capture.provider, nativeId: capture.native_id, capture_ref: rawRef.id,
          recordKey: descriptor.original_record_key ?? String(descriptor.original_record_ordinal),
          attachmentOrdinal: asset.ordinal, attachments: [descriptor] }, { documentId: capture.owner.document_id });
        signal?.throwIfAborted();
        if (!result?.ok) throw new Error(result?.error || "native_asset_acquisition_failed");
        const acquired = result.acquisition.attachments?.[0];
        if (acquired?.staged_asset) {
          const meta = await captureStaging.metadata(acquired.staged_asset.id);
          captureStaging.requireOwner(meta, acquired.staged_asset, capture.owner);
          file = await captureStaging.file(meta.id); sha256 = meta.sha256; outcome = { status: "acquired" };
        } else outcome = { status: result.acquisition.outcome.failed?.[0]?.status || "no_resolvable_source" };
      } finally { signal?.removeEventListener("abort", cancel); }
    }
    adopted = await client.refreshNativeOwner(adopted, capture.preparation_instance_id, signal);
    await client.nativeAsset(adopted, capture.acquisition_id, asset, outcome, file, sha256, signal);
  }
  progress("native_assets", "END");
  progress("native_finalize", "BEGIN");
  adopted = await client.refreshNativeOwner(adopted, capture.preparation_instance_id, signal);
  const final = await client.finalizeNative(adopted, capture.acquisition_id, prepared.plan_digest, signal);
  progress("native_finalize", "END");
  return { rawRevision, summary: prepared.summary, reference: { ...reference, adopted,
    plan_digest: prepared.plan_digest, sha256: final.sha256, size_bytes: final.size_bytes } };
}

async function publishNativeCapture(reference, recordRef, signal) {
  await ensureTrustedReceiver();
  const capture = await captureStore.getCapture(recordRef);
  if (!capture || capture.state !== "ready" || capture.receiver_native.sha256 !== reference.sha256) throw new Error("native_final_artifact_conflict");
  if (capture.receiver_receipt) return capture.receiver_receipt;
  const settings = await receiverSettings();
  const client = new CaptureJobClient({ baseUrl: settings.baseUrl, fetchImpl: runtimeNetwork });
  client.nativeOwnerChanged = async (adopted) => {
    const retained = await captureStore.getCapture(recordRef);
    if (!retained) throw new Error("native_preparation_owner_missing");
    await captureStore.putCapture({ ...retained, receiver_native: { ...reference, adopted } });
  };
  const adopted = await client.refreshNativeOwner(reference.adopted, reference.preparation_instance_id, signal);
  const receipt = await client.publishNative(adopted, reference.acquisition_id, reference.plan_digest, reference.sha256, signal);
  const error = receiverAckContractError(receipt, reference.sha256);
  if (error) throw error;
  await captureStore.putCapture({ ...capture, receiver_native: { ...reference, adopted }, receiver_receipt: receipt });
  return receipt;
}

async function postJson(path, payload, serializedBody = null, requireReceiverRequestId = false, signal = null) {
  if (path === "/v1/browser-captures" && payload.receiver_native) return publishNativeCapture(payload.receiver_native, payload.capture_record_ref, signal);
  await ensureTrustedReceiver();
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  await appendDebugLog({ stage: "receiver_request", method: "POST", path, request_id: requestId, has_body: true });
  let captureBody = null;
  if (path === "/v1/browser-captures") {
    const prepared = await captureStaging.prepare(payload, null, null, signal);
    captureBody = prepared;
    if (prepared.acknowledgedReceipt) return prepared.acknowledgedReceipt;
    serializedBody = prepared.body;
  }
  signal?.throwIfAborted();
  try {
    const response = await runtimeNetwork(`${settings.baseUrl}${path}`, {
      method: "POST",
      headers: await requestHeaders({ hasBody: true, requestId }),
      body: serializedBody || JSON.stringify(payload),
      signal,
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
    const receipt = { ...body, receiver_request_id: receiverRequestId };
    if (captureBody) {
      const contractError = receiverAckContractError(receipt, captureBody.contentHash);
      if (contractError || !acknowledgedRequestId) throw contractError || new Error("receiver_contract_incompatible:missing_receiver_request_id");
      await captureStaging.markAcknowledged(captureBody.ref, receipt);
    }
    return receipt;
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
  }
}

async function reportCaptureHealth(event, payload = {}) {
  try {
    await postJson("/v1/capture-health", {
      event,
      provider: payload.provider || null,
      provider_session_id: payload.provider_session_id || null,
      extension_instance_id: await extensionInstanceId(),
      visible_count: Number.isInteger(payload.visible_count) ? payload.visible_count : null,
      captured_count: Number.isInteger(payload.captured_count) ? payload.captured_count : null,
      reason: payload.reason || null,
      detail: payload.detail && typeof payload.detail === "object" ? payload.detail : {},
    });
  } catch {
    // Health reporting is best effort. A receiver outage must remain visible
    // through the local retry queue and popup state, not recursively create a
    // second failure path.
  }
}

async function getJson(path, timeoutMs = null) {
  await ensureTrustedReceiver();
  const settings = await receiverSettings();
  const requestId = buildReceiverRequestId();
  await appendDebugLog({ stage: "receiver_request", method: "GET", path, request_id: requestId });
  const controller = timeoutMs ? new globalThis.AbortController() : null;
  const timeout = timeoutMs ? globalThis.setTimeout(() => controller.abort("receiver_request_timeout"), timeoutMs) : 0;
  try {
    const response = await runtimeNetwork(`${settings.baseUrl}${path}`, {
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

// Layer 2 is a read-only projection. It deliberately uses the daemon's
// canonical session id and typed read routes; provider DOM text and local
// storage ledgers are not authority for cost, assertions, or archive links.
async function missionIntelligenceProjection(state, configuredUrl) {
  const indexedSessionId = state?.archive_state?.indexed_session_id || null;
  const base = String(configuredUrl || "").replace(/\/+$/, "");
  const unavailable = (status, reason) => ({
    status,
    reason,
    archive: { status: indexedSessionId ? "available" : "uncaptured", session_id: indexedSessionId },
    cost: { status: "unknown", total_usd: null, provenance: [] },
    assertions: { status: "unknown", items: [] },
  });
  if (state?.error === "unauthorized") return unavailable("unauthorized", "receiver_authorization_required");
  if (state?.online === false) return unavailable("offline", "receiver_unavailable");
  if (state?.archive_state?.state === "failed") return unavailable("failed", state.archive_state.reason || state.archive_state.error || "archive_ingest_failed");
  if (!indexedSessionId) return unavailable("uncaptured", "canonical_session_not_indexed");

  const encodedProvider = encodeURIComponent(state.provider || "");
  const encodedSession = encodeURIComponent(state.provider_session_id || "");
  let projection;
  try {
    projection = await getJson(
      `/v1/mission-control?provider=${encodedProvider}&provider_session_id=${encodedSession}`,
      MISSION_INTELLIGENCE_TIMEOUT_MS,
    );
  } catch (error) {
    const status = error?.status === 401 ? "unauthorized" : error?.status === 404 ? "incompatible" : error?.status ? "receiver_error" : "offline";
    return unavailable(status, error?.message || "projection_unavailable");
  }
  const archiveUrl = new globalThis.URL(base);
  // The receiver (8765) does not serve archive pages. The daemon's canonical
  // reader is the separate web endpoint (8766) and uses /s/:session_id.
  if (archiveUrl.port === "8765") archiveUrl.port = "8766";
  archiveUrl.pathname = `/s/${encodeURIComponent(indexedSessionId)}`;
  archiveUrl.search = "";
  archiveUrl.hash = "";
  return {
    ...projection,
    archive: { ...projection.archive, url: archiveUrl.toString() },
    cost: {
      ...(projection.cost || {}),
      status: projection.cost?.status === "unavailable" ? "unknown" : (projection.cost?.status || "unknown"),
      total_usd: projection.cost?.status === "unavailable" ? null : (projection.cost?.total_usd ?? null),
      provenance: Array.isArray(projection.cost?.provenance) ? projection.cost.provenance : [],
    },
    assertions: projection.assertions || { status: "unknown", items: [] },
  };
}

async function backfillReceiverPreflight() {
  let capability;
  try {
    capability = await getJson("/v1/browser-captures/capabilities");
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
  if (!tab?.id || !plan.length || !runtimeChrome.scripting?.executeScript) return false;
  for (const step of plan) {
    const details = { target: { tabId: tab.id }, files: step.files };
    if (step.world) details.world = step.world;
    await runtimeChrome.scripting.executeScript(details);
  }
  return true;
}

function providerForUrl(url) {
  try {
    const hostname = new globalThis.URL(url || "").hostname;
    if (hostname === "chatgpt.com") return "chatgpt";
    if (hostname === "claude.ai") return "claude-ai";
    if (hostname === "gemini.google.com") return "gemini";
    if (hostname === "grok.com") return "grok";
  } catch {
    return null;
  }
  return null;
}

function providerRequestFromUrl(urlValue) {
  const url = new globalThis.URL(urlValue);
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
  if (url.hostname === "grok.com") {
    if (url.pathname === "/rest/app-chat/conversations") return { provider: "grok", operation: "inventory", params: {
      pageSize: Number.parseInt(url.searchParams.get("pageSize") || "60", 10), pageToken: url.searchParams.get("pageToken") } };
    const conversation = url.pathname.match(/^\/rest\/app-chat\/conversations\/([A-Za-z0-9_-]+)(?:\/(responses|response-node))?$/);
    if (conversation) return { provider: "grok", operation: conversation[2] || "conversation", params: { nativeId: conversation[1] } };
  }
  throw new Error("backfill_provider_url_not_allowed");
}

async function waitForProviderTab(tabId, provider) {
  for (;;) {
    const tab = await runtimeChrome.tabs.get(tabId);
    if (providerForUrl(tab?.url || tab?.pendingUrl) === provider && tab?.status === "complete") return tab;
    await new Promise((resolve) => globalThis.setTimeout(resolve, 250));
  }
}

function providerTransportSessionKey(provider) {
  return `${PROVIDER_TRANSPORT_SESSION_PREFIX}:${provider}`;
}

function providerTransportOperatorTakenSessionKey(provider) {
  return `${PROVIDER_TRANSPORT_OPERATOR_TAKEN_SESSION_PREFIX}:${provider}`;
}

async function forgetProviderTransport(provider, tabId = null) {
  const key = providerTransportSessionKey(provider);
  const operatorTakenKey = providerTransportOperatorTakenSessionKey(provider);
  const stored = await runtimeChrome.storage.session.get({ [key]: null, [operatorTakenKey]: null });
  if (tabId === null || stored[key] === tabId) {
    await runtimeChrome.storage.session.remove([key, operatorTakenKey]);
  }
}

async function markProviderTransportOperatorTaken(provider, tabId) {
  const key = providerTransportSessionKey(provider);
  const operatorTakenKey = providerTransportOperatorTakenSessionKey(provider);
  await runtimeChrome.storage.session.set({ [key]: tabId, [operatorTakenKey]: tabId });
}

function operatorTakenProviderTransportError() {
  const error = new Error("provider_transport_operator_taken");
  error.code = "provider_transport_operator_taken";
  return error;
}

function providerTransportNoSurfaceError() {
  const error = new Error("provider_transport_no_surface");
  error.code = "provider_transport_no_surface";
  return error;
}

async function observedProviderTab(provider) {
  const tabs = await runtimeChrome.tabs.query({});
  return tabs.find((tab) => providerForUrl(tab.url || tab.pendingUrl) === provider) || null;
}

async function acquireProviderTab(provider, { allowCreate = false } = {}) {
  // Passive capture and inventory use a normal provider page only when the
  // operator already has one open. They must never materialize a root tab.
  // Creating a background transport is reserved for an explicit queued
  // browser action that needs to mutate the provider UI.
  if (!allowCreate) {
    const observed = await observedProviderTab(provider);
    if (!observed) throw providerTransportNoSurfaceError();
    return { tab: observed, owned: false, cleanupAlarm: null };
  }

  const key = providerTransportSessionKey(provider);
  const operatorTakenKey = providerTransportOperatorTakenSessionKey(provider);
  const stored = await runtimeChrome.storage.session.get({ [key]: null, [operatorTakenKey]: null });
  const storedTabId = stored[key];
  if (Number.isInteger(storedTabId)) {
    const existing = await tabIfPresent(storedTabId);
    if (!existing) {
      await forgetProviderTransport(provider, storedTabId);
    } else {
      if (stored[operatorTakenKey] === storedTabId) throw operatorTakenProviderTransportError();
      if (existing.active === true) {
        // A previously background-owned tab has become an operator surface. It
        // must never be repurposed, closed, or replaced behind their back.
        await markProviderTransportOperatorTaken(provider, storedTabId);
        throw operatorTakenProviderTransportError();
      }
      if (providerForUrl(existing.url || existing.pendingUrl) === provider) {
        return {
          tab: existing,
          owned: true,
          cleanupAlarm: `${BACKFILL_TRANSPORT_CLEANUP_PREFIX}:${provider}:${storedTabId}`,
        };
      }
      await forgetProviderTransport(provider, storedTabId);
    }
  }
  const url = provider === "chatgpt" ? "https://chatgpt.com/" : provider === "grok" ? "https://grok.com/" : "https://claude.ai/";
  const created = await runtimeChrome.tabs.create({ url, active: false });
  if (!created?.id) throw new Error("backfill_provider_tab_create_failed");
  await runtimeChrome.storage.session.set({ [key]: created.id });
  const cleanupAlarm = `${BACKFILL_TRANSPORT_CLEANUP_PREFIX}:${provider}:${created.id}`;
  await runtimeChrome.alarms.create(cleanupAlarm, { when: Date.now() + BACKFILL_TRANSPORT_TAB_TTL_MS });
  try {
    const ready = created.status === "complete" ? created : await waitForProviderTab(created.id, provider);
    return { tab: ready, owned: true, cleanupAlarm };
  } catch (error) {
    await cleanupBackfillTransportTab(cleanupAlarm);
    throw error;
  }
}

function providerTab(provider, { allowCreate = false } = {}) {
  const transportKey = `${provider}:${allowCreate ? "action" : "observe"}`;
  const inFlight = providerTransportPromises.get(transportKey);
  if (inFlight) return inFlight;
  const candidate = acquireProviderTab(provider, { allowCreate });
  const tracked = candidate.finally(() => {
    if (providerTransportPromises.get(transportKey) === tracked) providerTransportPromises.delete(transportKey);
  });
  providerTransportPromises.set(transportKey, tracked);
  return tracked;
}

function finiteRetryAfterSeconds(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  requireProviderCooldownMs(Math.ceil(seconds * 1000));
  return seconds;
}

function withProviderTransportOperation(provider, operation, { checkThrottle = true, signal = null } = {}) {
  let started = false;
  const prior = providerTransportOperations.get(provider) || Promise.resolve();
  const result = prior.catch(() => undefined).then(async () => {
    signal?.throwIfAborted();
    started = true;
    if (checkThrottle) await requireProviderThrottleAvailability(provider);
    signal?.throwIfAborted();
    try {
      const value = await operation();
      // Some operations report a provider refusal as a resolved failure
      // result rather than a throw; a rate limit there must still set the
      // shared cooldown, or the next request contacts the provider during
      // its advertised Retry-After. A 429 without a Retry-After header is
      // still a rate limit; the recorder supplies the default delay.
      if (value && value.ok === false) {
        const refusal = new Error(value.detail || "browser_action_failed");
        if (value.outcome) refusal.outcome = value.outcome;
        refusal.retryAfterSeconds = finiteRetryAfterSeconds(value.retry_after_seconds);
        const classified = classifyBrowserActionFailure(refusal, refusal.retryAfterSeconds);
        if (classified.outcome === "rate_limited") await recordProviderThrottle(provider, refusal, classified);
      }
      return value;
    } catch (error) {
      const classified = classifyBrowserActionFailure(error, finiteRetryAfterSeconds(error?.retryAfterSeconds));
      if (classified.outcome === "rate_limited" && !error?.providerThrottleApplied) {
        await recordProviderThrottle(provider, error, classified);
      }
      throw error;
    }
  });
  const tracked = result.finally(() => {
    if (providerTransportOperations.get(provider) === tracked) providerTransportOperations.delete(provider);
  });
  providerTransportOperations.set(provider, tracked);
  if (!signal) return tracked;
  return new Promise((resolve, reject) => {
    const abort = () => { if (!started) reject(signal.reason); };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    tracked.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
  });
}

function providerThrottleError(deadline, nowMs) {
  const error = new Error("provider_rate_limited");
  error.outcome = "rate_limited";
  error.retryUntilMs = deadline;
  error.retryAfterMs = Math.max(0, deadline - nowMs);
  error.retryAfterSeconds = Math.ceil(error.retryAfterMs / 1000);
  error.providerThrottleApplied = true;
  return error;
}

async function requireProviderThrottleAvailability(provider) {
  const queue = await storedCaptureFreshnessQueue();
  const deadline = Number(queue.provider_cooldowns[provider]) || 0;
  const now = Date.now();
  if (deadline > now) throw providerThrottleError(deadline, now);
}

function retryDelayFromProviderError(error, classified) {
  if (error?.retryAfterMs != null) return requireProviderCooldownMs(Math.max(1_000, error.retryAfterMs));
  if (error?.retryAfter) {
    const delay = retryAfterMs({ get: (name) => (name.toLowerCase() === "retry-after" ? error.retryAfter : null) }, Date.now());
    if (delay !== null) return requireProviderCooldownMs(Math.max(1_000, delay));
  }
  return requireProviderCooldownMs(failureRetryDelayMs(0, classified.outcome, classified.retry_after_seconds));
}

async function recordProviderThrottle(provider, error, classified) {
  const now = Date.now();
  const queue = await serializeStorageMutation(async () => {
    const current = await storedCaptureFreshnessQueue();
    return persistCaptureFreshnessQueue(extendProviderCooldown(current, {
      provider,
      untilMs: now + retryDelayFromProviderError(error, classified),
      nowMs: now,
    }));
  });
  await scheduleNextCaptureFreshnessWake(queue);
}

function pageContextResponse(response) {
  const body = typeof response?.body === "string" ? response.body : "";
  return {
    ok: Boolean(response?.ok),
    status: Number(response?.status || 0),
    polyloguePageContext: true,
    polylogueAuthReason: response?.authReason || null,
    polylogueSelectedOrganizationId: response?.selectedOrganizationId || null,
    headers: { get: (name) => {
      const normalized = name.toLowerCase();
      if (normalized === "content-type") return response?.contentType || "";
      if (normalized === "retry-after") return response?.retryAfter || null;
      return null;
    } },
    async json() { return JSON.parse(body); },
  };
}

function providerPageFailure(result, fallback) {
  const error = new Error(String(result?.error || fallback));
  if (result?.outcome === "rate_limited" && result.status === 429) {
    error.outcome = "rate_limited"; error.status = 429; error.retryAfter = result.retryAfter;
  }
  return error;
}

async function runProviderPageScript(transport, request, signal = null) {
  await ensureCaptureScripts(transport.tab);
  signal?.throwIfAborted();
  const requestId = globalThis.crypto.randomUUID();
  let cancellation = Promise.resolve();
  const cancel = () => {
    cancellation = runtimeChrome.scripting.executeScript({ target: { tabId: transport.tab.id }, world: "MAIN",
      func: (id, ownerId) => globalThis.window.dispatchEvent(new globalThis.CustomEvent("polylogue.providerCancel", { detail: { requestId: id, ownerId } })), args: [requestId, runtimeChrome.runtime.id] });
    void cancellation.catch(() => undefined);
  };
  signal?.addEventListener("abort", cancel, { once: true });
  try {
    const executions = await runtimeChrome.scripting.executeScript({
      target: { tabId: transport.tab.id }, world: "MAIN",
      func: executeProviderPageRequest, args: [{ ...request, requestId, ownerId: runtimeChrome.runtime.id }],
    });
    signal?.throwIfAborted();
    return executions?.[0]?.result;
  } finally { signal?.removeEventListener("abort", cancel); await cancellation; }
}

function backfillNativeBundleOwner() {
  return {
    begin: async (nativeId, signal, context) => withProviderTransportOperation("grok", async () => {
      signal?.throwIfAborted();
      if (!context?.item) throw new Error("native_bundle_execution_missing");
      const transport = await providerTab("grok");
      await ensureCaptureScripts(transport.tab);
      const documents = await runtimeChrome.scripting.executeScript({ target: { tabId: transport.tab.id }, world: "ISOLATED", func: () => true });
      signal?.throwIfAborted();
      const owner = { tab_id: transport.tab.id, document_id: documents?.[0]?.documentId || null, provider: "grok" };
      const bundle = await captureStore.beginNativeBundle({ extensionInstanceId: await extensionInstanceId(), owner, provider: "grok", nativeId, bundleId: globalThis.crypto.randomUUID(),
        requiredReplies: ["conversation", "responses"], queueContext: { itemId: context.item.id, jobId: context.jobId, owner: context.owner, generation: context.generation } });
      Object.assign(context.item, await captureStore.getQueue(context.item.id));
      return bundle;
    }, { signal }),
    response: async (bundleId, name, response, signal) => {
      signal?.throwIfAborted();
      const bundle = await captureStore.getCapture(bundleId);
      if (!bundle) throw new Error("native_bundle_recovery_missing");
      if (!response.ok) {
        await captureStore.publishNativeBundleReply(bundleId, bundle.owner, name, null, { ok: false, status: response.status, retry_after: response.headers.get("retry-after") });
        await captureStore.finishNativeBundle(bundleId, bundle.owner);
      } else if (!bundle.replies[name] || bundle.replies[name].id !== response.captureRawRef?.id) throw new Error("native_bundle_reply_missing");
    },
    restoreReply: async (ref, signal) => {
      signal?.throwIfAborted();
      let meta;
      try { meta = await captureStaging.metadata(ref.id); await captureStaging.file(ref.id); }
      catch (error) {
        if (error?.name === "NotFoundError" || error?.code === "capture_staging_interrupted") throw new Error("native_bundle_recovery_missing");
        throw error;
      }
      if (meta.token !== ref.token || meta.state !== "sealed" || meta.owner.provider !== "grok") throw new Error("native_bundle_reply_invalid");
      return stagedProviderResponse({ ok: true, status: meta.response_metadata?.status || 200,
        contentType: meta.response_metadata?.content_type || "application/json", bodyRef: ref }, { provider: "grok" }, meta);
    },
    finish: async (bundleId, signal) => {
      const bundle = await captureStore.getCapture(bundleId);
      if (!bundle) throw new Error("native_bundle_recovery_missing");
      return nativeNormalizer.finishBundle(bundleId, bundle.owner, { signal });
    },
  };
}

function nativeSessionIdFromHeaders(provider, headers) {
  if (provider === "chatgpt") return headers.conversation_id || headers.id;
  if (provider === "claude-ai") return headers.uuid || headers.id;
  return headers.conversationId || headers.id;
}

function stagedProviderResponse(wire, request, meta) {
    const response = pageContextResponse(wire);
    response.captureRawRef = wire.bodyRef;
    // Inventory replies are provider-paged. Native conversation responses use
    // the streaming normalizer and never call this page materializer.
    response.json = async () => {
      const value = JSON.parse(await (await captureStaging.file(meta.id)).text());
      if (meta.kind === "provider-inventory") await captureStaging.discardUnreferenced(meta.id);
      return value;
    };
    response.normalizeCapture = async (item, attribution, relatedResponses = {}, signal = null, queueContext = null) => {
      const controller = new globalThis.AbortController();
      const abort = () => controller.abort(signal.reason);
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
      const operation = { rawId: meta.id, controller, promise: null };
      trackNativeOperation(operation);
      operation.promise = (async () => nativeNormalizer.normalize({ provider: request.provider, rawRef: wire.bodyRef, nativeId: item.native_id,
        extensionVersion: runtimeChrome.runtime.getManifest().version, instanceId: await extensionInstanceId(), attribution: { backfill: attribution }, signal: controller.signal,
        relatedRefs: Object.fromEntries(Object.entries(relatedResponses).map(([key, value]) => [key, value.captureRawRef])), queueContext }))();
      try {
        const envelope = await operation.promise;
        await captureStaging.retireFailedNormalizations(meta.id, envelope.capture_record_ref, controller.signal)
          .catch((error) => appendDebugLog({ stage: "native_normalization_cleanup_pending", raw_ref: meta.id, error: String(typeof error.code === "string" ? error.code : (error.message || error)) }));
        return envelope;
      } finally {
        try { if (queueContext) Object.assign(item, await captureStore.getQueue(item.id)); }
        finally { signal?.removeEventListener("abort", abort); operation.finish(); }
      }
    };
    return response;
}

async function providerPageFetch(url, options = {}) {
  if (options.method && options.method !== "GET") throw new Error("backfill_provider_method_not_allowed");
  const request = { ...providerRequestFromUrl(url), capture_bundle: options.captureBundle || null,
    queue_context: options.queueContext ? { itemId: options.queueContext.item.id, jobId: options.queueContext.jobId,
      owner: options.queueContext.owner, generation: options.queueContext.generation, nativeId: options.queueContext.item.native_id } : null };
  if (request.queue_context) {
    const context = request.queue_context;
    const job = await captureStore.assertJobExecution(context.jobId, context.owner, context.generation);
    if (!/^h1:[A-Za-z0-9_-]{43}$/.test(job.account_scope || "")) throw new Error("capture_job_account_scope_unresolved");
    const item = await captureStore.getQueue(context.itemId);
    if (item?.raw_acquisition_ref) {
      const ref = item.raw_acquisition_ref;
      const meta = await captureStaging.metadata(ref.id);
      if (meta.state !== "sealed") throw new Error("native_acquisition_recovery_pending");
      if (meta.token !== ref.token || meta.owner.provider !== request.provider || meta.source_url !== url ||
          meta.queue_context?.jobId !== context.jobId || meta.queue_context?.itemId !== context.itemId) throw new Error("native_acquisition_identity_conflict");
      const headers = await nativeNormalizer.headers(ref, options.signal);
      if (String(nativeSessionIdFromHeaders(request.provider, headers) || "") !== context.nativeId) throw new Error("native_capture_identity_mismatch");
      return stagedProviderResponse({ ok: true, status: meta.response_metadata?.status || 200,
        contentType: meta.response_metadata?.content_type || "application/json", bodyRef: ref }, request, meta);
    }
  }
  return withProviderTransportOperation(request.provider, async () => {
    const transport = await providerTab(request.provider);
    let result;
    try {
      result = await runProviderPageScript(transport, request, options.signal);
    } catch (error) {
      if (transport.owned) {
        if (transport.cleanupAlarm) await cleanupBackfillTransportTab(transport.cleanupAlarm);
      }
      throw error;
    }
    if (!result?.ok) {
      const error = String(result?.error || "backfill_page_request_failed");
      if (error.includes("auth_context") || error.includes("selected_organization")) {
        return pageContextResponse({ ok: false, status: 401, contentType: "application/json", authReason: error, body: JSON.stringify({ error }) });
      }
      throw providerPageFailure(result, "backfill_page_request_failed");
    }
    const wire = result.response;
    if (!wire?.ok) {
      const failed = pageContextResponse(wire);
      if (failed.status === 429) {
        const error = new Error("provider_rate_limited"); error.outcome = "rate_limited"; error.retryAfter = wire.retryAfter;
        await recordProviderThrottle(request.provider, error, classifyBrowserActionFailure(error));
      }
      return failed;
    }
    const meta = await captureStaging.metadata(wire.bodyRef?.id);
    if (meta.token !== wire.bodyRef?.token || meta.state !== "sealed" || meta.owner.tab_id !== transport.tab.id || meta.owner.provider !== request.provider) throw new Error("capture_staging_owner_mismatch");
    const response = stagedProviderResponse(wire, request, meta);
    if (response.status === 429) {
      const error = new Error("provider_rate_limited");
      error.outcome = "rate_limited";
      error.retryAfter = response.headers.get("retry-after");
      await recordProviderThrottle(request.provider, error, classifyBrowserActionFailure(error));
    }
    return response;
  }, { signal: options.signal });
}

async function providerAccountHandle(provider) {
  return withProviderTransportOperation(provider, async () => {
    const transport = await providerTab(provider);
    let result;
    try {
      await ensureCaptureScripts(transport.tab);
      const executions = await runtimeChrome.scripting.executeScript({
        target: { tabId: transport.tab.id }, world: "MAIN",
        func: executeProviderPageRequest, args: [{ provider, operation: "identity", params: {}, ownerId: runtimeChrome.runtime.id }],
      });
      result = executions?.[0]?.result;
      if (!result?.ok) throw providerPageFailure(result, "backfill_provider_identity_unavailable");
      const accountHandle = result.response?.accountHandle;
      if (typeof accountHandle !== "string" || !accountHandle.trim()) {
        throw new Error("backfill_provider_identity_unavailable");
      }
      return accountHandle;
    } catch (error) {
      if (transport.owned) {
        if (transport.cleanupAlarm) await cleanupBackfillTransportTab(transport.cleanupAlarm);
      }
      throw error;
    }
  });
}

async function cleanupBackfillTransportTab(alarmName) {
  const parts = alarmName.split(":");
  const provider = parts[1];
  const tabId = Number.parseInt(parts[2] || "", 10);
  if (!provider || !Number.isInteger(tabId)) return;
  try {
    const key = providerTransportSessionKey(provider);
    const operatorTakenKey = providerTransportOperatorTakenSessionKey(provider);
    const stored = await runtimeChrome.storage.session.get({ [key]: null, [operatorTakenKey]: null });
    // An old wake cannot close a tab whose ownership has been relinquished or
    // replaced. A draft and an operator-adopted tab remain user surfaces.
    if (stored[key] !== tabId || stored[operatorTakenKey] === tabId) {
      await runtimeChrome.alarms.clear(alarmName);
      return;
    }
    const tab = await tabIfPresent(tabId);
    const current = await runtimeChrome.storage.session.get({ [key]: null, [operatorTakenKey]: null });
    if (current[key] !== tabId || current[operatorTakenKey] === tabId) {
      await runtimeChrome.alarms.clear(alarmName);
      return;
    }
    if (tab?.active === true) {
      await markProviderTransportOperatorTaken(provider, tabId);
      await runtimeChrome.alarms.clear(alarmName);
      return;
    }
    if (tab && providerForUrl(tab.url || tab.pendingUrl) === provider) await runtimeChrome.tabs.remove(tabId);
    await forgetProviderTransport(provider, tabId);
    await runtimeChrome.alarms.clear(alarmName);
  } catch (error) {
    // Failed lookup/removal does not prove physical closure. Preserve the
    // original persisted owner for a later cleanup or acquisition attempt.
    await appendDebugLog({ stage: "provider_transport_cleanup_pending", provider, tab_id: tabId,
      error: String(error.message || error) }).catch(() => undefined);
  }
}

async function captureTab(tab, reason = "background", expectedConversation = null) {
  if (expectedConversation && tab?.id && runtimeChrome.tabs?.get) {
    const currentTab = await runtimeChrome.tabs.get(tab.id);
    const currentUrl = currentTab?.url || currentTab?.pendingUrl || "";
    if (
      !currentTab
      || currentUrl !== expectedConversation.url
      || archiveProviderForUrl(currentUrl) !== expectedConversation.provider
      || await capturedConversationIdForTab(currentTab) !== expectedConversation.providerSessionId
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
  if (reason !== "popup_sync_open_tabs" && !(await automaticCaptureEnabled())) {
    return { ok: false, skipped: true, reason: "automatic_capture_paused" };
  }
  if (reason !== "popup_sync_open_tabs") await requirePairedTrustedReceiver();
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
    const pageProvider = archiveProviderForUrl(conversationUrl);
    const resultWithTimeout = await withProviderTransportOperation(pageProvider, async () => {
      const result = await runtimeChrome.tabs.sendMessage(tab.id, captureMessage);
      if (!result?.ok && result?.outcome === "rate_limited") {
        const error = new Error("provider_rate_limited");
        error.outcome = "rate_limited";
        error.retryAfterSeconds = Number.isFinite(result.retry_after_seconds)
          ? result.retry_after_seconds
          : null;
        error.retryAfterMs = error.retryAfterSeconds === null ? null : error.retryAfterSeconds * 1000;
        throw error;
      }
      return result;
    });
    if (resultWithTimeout?.ok && resultWithTimeout.captureResult?.outcome === "superseded") return resultWithTimeout;
    if (resultWithTimeout?.ok) {
      const envelopeSession = resultWithTimeout.envelope?.session || {};
      const summary = envelopeSessionSummary(resultWithTimeout.envelope || {});
      const provider = resultWithTimeout.captureResult?.provider || envelopeSession.provider;
      const providerSessionId = resultWithTimeout.captureResult?.provider_session_id || envelopeSession.provider_session_id;
      const pageProvider = archiveProviderForUrl(tab.url || tab.pendingUrl || "") || provider;
      // conversationIdForUrl(tab.url) is normally the right identity to log/
      // record here (it is what every other caller in this file keys tab
      // state by). The one exception is TEMPORARY_CHAT_SENTINEL: it is a
      // "this tab has a real, capturable conversation" signal, not the
      // conversation's real ephemeral id, so preferring it over the
      // just-captured envelope's actual provider_session_id would record
      // the sentinel as this tab's "captured" identity -- self-consistent
      // locally, but never matching the real id the receiver/archive
      // actually stored it under, so the tab would show "not captured"
      // forever afterward (and setStateForTab's own staleness guard has a
      // matching sentinel exemption for the same reason).
      const urlSessionId = conversationIdForUrl(tab.url || tab.pendingUrl || "");
      const pageSessionId = urlSessionId === TEMPORARY_CHAT_SENTINEL ? (providerSessionId || urlSessionId) : (urlSessionId || providerSessionId);
      await updateSessionLedger({
        provider,
        providerSessionId,
        patch: {
          reason,
          tab_id: tab.id,
          tab_url: tab.url || tab.pendingUrl || null,
          capture_mode: envelopeSession.provider_meta?.capture_fidelity || null,
          turn_count: summary.turnCount,
          attachment_count: summary.attachmentCount,
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
        asset_acquisition: summary.assetAcquisition,
        turn_count: summary.turnCount,
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
    return {
      ok: false,
      error: String(error.message || error),
      outcome: error?.outcome || null,
      retry_after_seconds: Number.isFinite(error?.retryAfterSeconds) ? error.retryAfterSeconds : null,
    };
  }
}

async function settleNativeInvocation(row) {
  const ref = { id: row.id, token: row.token };
  await captureStore.closeNativeInvocation(ref, false);
  try {
    const result = await runtimeChrome.tabs.sendMessage(row.owner.tab_id,
      { type: "polylogue.cancelCapture", invocationRef: ref }, row.owner.document_id ? { documentId: row.owner.document_id } : undefined);
    if (!result?.ok) throw new Error("native_invocation_settlement_pending");
    await captureStore.closeNativeInvocation(ref);
  } catch (error) {
    const tab = await tabIfPresent(row.owner.tab_id);
    if (!tab || (row.owner.document_id && /no document with id/i.test(String(error.message || error)))) {
      await captureStore.closeNativeInvocation(ref); return;
    }
    throw error;
  }
}

async function requireNativeInvocation(ref, owner, nativeId) {
  const row = ref ? await captureStore.getCapture(ref.id) : null;
  if (!row || row.kind !== "native-invocation" || row.token !== ref.token || row.worker_owner !== runtimeWorkerId ||
      JSON.stringify(row.owner) !== JSON.stringify(owner) || row.state !== "open") {
    throw new Error("native_invocation_owner_mismatch");
  }
  if (row.native_id !== nativeId) throw new Error("native_capture_identity_mismatch");
  const tab = await runtimeChrome.tabs.get(owner.tab_id);
  const visibleId = conversationIdForUrl(tab?.url || "");
  if (archiveProviderForUrl(tab?.url || "") !== owner.provider ||
      (visibleId && visibleId !== nativeId && visibleId !== TEMPORARY_CHAT_SENTINEL)) throw new Error("native_invocation_document_changed");
  if (owner.document_id) {
    const document = await runtimeChrome.tabs.sendMessage(owner.tab_id, { type: "polylogue.stagingOwner" }, { documentId: owner.document_id });
    if (!document?.ok) throw new Error("native_invocation_document_changed");
  }
  const current = await captureStore.getCapture(ref.id);
  if (!current || current.token !== row.token || current.worker_owner !== runtimeWorkerId ||
      JSON.stringify(current.owner) !== JSON.stringify(owner) || current.state !== "open") {
    throw new Error("native_invocation_owner_mismatch");
  }
  return current;
}

async function captureProviderConversation(
  provider,
  providerSessionId,
  reason,
  {
    deferReceiver = false,
    generationObservations = [],
    providerUpdatedAt = null,
  } = {},
) {
  if (provider !== "chatgpt") throw new Error(`exact_provider_capture_unsupported:${provider}`);
  if (!/^[A-Za-z0-9_-]{1,256}$/.test(String(providerSessionId || ""))) {
    throw new Error("exact_provider_capture_invalid_session_id");
  }
  await requirePairedTrustedReceiver();
  return withProviderTransportOperation(provider, async () => {
    const transport = await providerTab(provider);
    await ensureCaptureScripts(transport.tab);
    const documents = await runtimeChrome.scripting.executeScript({ target: { tabId: transport.tab.id }, world: "ISOLATED", func: () => true });
    const owner = { tab_id: transport.tab.id, document_id: documents?.[0]?.documentId || null, provider };
    for await (const prior of captureStore.captures()) {
      if (prior.kind === "native-invocation" && prior.owner.tab_id === owner.tab_id) await settleNativeInvocation(prior);
    }
    const invocationRef = await captureStore.reserveCaptureObservation(owner, providerSessionId, "native-invocation", runtimeWorkerId, await extensionInstanceId());
    let physicallySettled = false;
    try {
    const result = await runtimeChrome.tabs.sendMessage(transport.tab.id, {
        type: "polylogue.capturePage",
        reason,
        providerSessionId,
        invocationRef,
        deferReceiver,
        generationObservations,
        providerUpdatedAt,
      }, owner.document_id ? { documentId: owner.document_id } : undefined);
    physicallySettled = true;
    if (!result?.ok) {
      const error = new Error(result?.error || "exact_provider_capture_failed");
      error.outcome = result?.outcome || null;
      error.retryAfterSeconds = Number.isFinite(result?.retry_after_seconds)
        ? result.retry_after_seconds
        : null;
      error.retryAfterMs = error.retryAfterSeconds === null ? null : error.retryAfterSeconds * 1000;
      throw error;
    }
    const acceptedId = result.envelope?.session?.provider_session_id;
    if (acceptedId !== providerSessionId) throw new Error("exact_provider_capture_identity_mismatch");
    return result;
    } finally {
      if (physicallySettled) await captureStore.closeNativeInvocation(invocationRef);
      else await settleNativeInvocation({ owner, id: invocationRef.id, token: invocationRef.token });
    }
  });
}

async function storedCaptureFreshnessQueue() {
  const stored = await runtimeChrome.storage.local.get({ [CAPTURE_FRESHNESS_QUEUE_KEY]: null });
  return normalizeFreshnessQueue(stored[CAPTURE_FRESHNESS_QUEUE_KEY]);
}

async function persistCaptureFreshnessQueue(queue) {
  await runtimeChrome.storage.local.set({ [CAPTURE_FRESHNESS_QUEUE_KEY]: queue });
  return queue;
}

async function scheduleCaptureFreshness({
  provider,
  nativeId,
  reason,
  delayMs = 0,
  providerUpdatedAt = null,
  generationObservations = [],
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
      generationObservations,
    }));
  });
  const entry = queue.entries[`${provider}:${nativeId}`];
  await scheduleNextCaptureFreshnessWake(queue);
  return { scheduled: true, entry };
}

async function scheduleNextCaptureFreshnessWake(queueValue = null) {
  const queue = queueValue || await storedCaptureFreshnessQueue();
  const deadlines = Object.values(queue.entries).map((entry) => (
    entry.lease_owner
      ? entry.lease_expires_at_ms
      : Math.max(entry.next_attempt_at_ms || 0, queue.provider_cooldowns[entry.provider] || 0)
  )).filter(Number.isFinite);
  if (!deadlines.length) {
    await runtimeChrome.alarms?.clear?.(CAPTURE_FRESHNESS_ALARM);
    return;
  }
  await runtimeChrome.alarms?.create?.(CAPTURE_FRESHNESS_ALARM, {
    when: Math.max(Date.now() + 1_000, Math.min(...deadlines)),
  });
}

async function processCaptureFreshnessQueueOnce() {
  if (!(await automaticCaptureEnabled())) {
    await runtimeChrome.alarms?.clear?.(CAPTURE_FRESHNESS_ALARM);
    return { processed: 0, paused: true };
  }
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
      {
        generationObservations: claim.generation_observations || [],
        providerUpdatedAt: claim.provider_updated_at || null,
      },
    );
    needsFollowUp = result.captureResult?.outcome !== "superseded" && chatGptCaptureNeedsFollowUp(result.envelope);
    retryDelayMs = needsFollowUp ? runningPollDelayMs(claim.running_poll_count || 0) : 0;
    const receipt = result.captureResult || {};
    if (claim.provider_updated_at && receipt.content_hash && receipt.outcome !== "superseded") {
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
      event: receipt.outcome === "superseded" ? "held_with_reason" : (needsFollowUp ? "detected_new" : "captured"),
      reason: "freshness_convergence",
      detail: receipt.outcome === "superseded" ? "receiver_superseded" : (needsFollowUp ? "provider_still_running" : "provider_head_current"),
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
  if (!(await automaticCaptureEnabled())) {
    return { skipped: true, reason: "automatic_capture_paused" };
  }
  try {
    await requirePairedTrustedReceiver();
  } catch (error) {
    return { skipped: true, reason: "receiver_not_paired", error: String(error?.message || error) };
  }
  const now = Date.now();
  let queue = await storedCaptureFreshnessQueue();
  if (queue.sweep_not_before_ms > now) return { skipped: true, reason: "sweep_backoff" };
  const coordinator = await backfillCoordinator();
  for await (const job of coordinator.store.jobRecords()) {
    if (job.provider === "chatgpt" && job.status === "running") return { skipped: true, reason: "explicit_backfill_running" };
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
  await runtimeChrome.alarms?.create?.(CAPTURE_FRESHNESS_SWEEP_ALARM, {
    delayInMinutes: 1,
    periodInMinutes: CAPTURE_FRESHNESS_SWEEP_MINUTES,
  });
  await scheduleNextCaptureFreshnessWake();
}

async function captureSupportedTabs(reason) {
  if (!runtimeChrome.tabs?.query) return;
  const tabs = await runtimeChrome.tabs.query({});
  // Explicit operator sync may recapture an open conversation.  Automatic
  // lifecycle events must reconcile with the receiver first: `spooled_only`
  // means the receiver already accepted a capture and is converging it, not
  // that the extension should re-fetch authenticated provider data.  This is
  // deliberately durable across service-worker restarts, unlike the in-memory
  // background-capture throttle.
  if (reason === "popup_sync_open_tabs") {
    await Promise.allSettled(tabs.map((tab) => captureTab(tab, reason)));
    return;
  }
  await Promise.allSettled(tabs.map(async (tab) => {
    if (injectionPlanForUrl(tab?.url || tab?.pendingUrl || "").length) await ensureCaptureScripts(tab);
    await refreshActiveTabArchiveState(tab, reason);
  }));
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return globalThis.btoa(binary);
}

// ---- Provider-neutral browser actions ----------------------------------

async function updateBrowserAction(actionId, ownerInstanceId, patch) {
  return postJson(`/v1/browser-actions/${encodeURIComponent(actionId)}/events`, {
    owner_instance_id: ownerInstanceId,
    ...patch,
  });
}

// The operator's explicit approve/decline decision on an action the receiver
// is holding at "awaiting_approval" (polylogue-yyvg.7). Unlike
// updateBrowserAction this carries no lease -- an awaiting_approval action
// has never been leased, and approving it is what makes it claimable for the
// first time.
async function decideBrowserActionApproval(actionId, decision) {
  const ownerInstanceId = await browserActionExecutorId();
  return postJson(`/v1/browser-actions/${encodeURIComponent(actionId)}/approval`, {
    extension_instance_id: ownerInstanceId,
    decision,
  });
}

async function transferBrowserActionAttachments(action, ownerInstanceId, transport) {
  const settings = await receiverSettings();
  const transfer = async (command, item = null, offset = 0, encoded = "") => {
    const [result] = await runtimeChrome.scripting.executeScript({
      target: { tabId: transport.tab.id }, world: "MAIN",
      func: transferBrowserActionAttachmentInPage,
      args: [action.action_id, ownerInstanceId, command, item, offset, encoded],
    });
    if (!result?.result?.ok) throw new Error("protocol_attachment_transfer_response_missing");
  };
  try {
    await transfer("begin");
    for (const item of action.attachments || []) {
      if (!Number.isSafeInteger(item.size_bytes) || item.size_bytes < 0) {
        throw new Error("protocol_attachment_size_unrepresentable");
      }
      await transfer("start", item);
      const requestId = buildReceiverRequestId();
      const response = await runtimeNetwork(
        `${settings.baseUrl}/v1/browser-actions/${encodeURIComponent(action.action_id)}/attachments/${encodeURIComponent(item.attachment_id)}`,
        { headers: await requestHeaders({ requestId }) },
      );
      if (!response.ok) {
        const error = new Error(`browser_action_attachment_http_${response.status}`);
        error.retryAfterSeconds = Number.parseInt(response.headers.get("Retry-After") || "", 10) || null;
        if (response.body?.cancel) await response.body.cancel();
        throw error;
      }
      if (!response.body?.getReader) throw new Error("protocol_attachment_stream_unavailable");
      const reader = response.body.getReader();
      const hasher = new AttachmentSha256();
      let downloaded = 0;
      let complete = false;
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) { complete = true; break; }
          if (downloaded + value.byteLength > item.size_bytes) throw new Error("protocol_attachment_size_mismatch");
          for (let at = 0; at < value.byteLength; at += BROWSER_ACTION_ATTACHMENT_CHUNK_BYTES) {
            const chunk = value.subarray(at, at + BROWSER_ACTION_ATTACHMENT_CHUNK_BYTES);
            hasher.update(chunk);
            await transfer("append", item, downloaded, bytesToBase64(chunk));
            downloaded += chunk.byteLength;
          }
        }
        if (downloaded !== item.size_bytes) throw new Error("protocol_attachment_size_mismatch");
        if (hasher.digestHex() !== item.sha256) throw new Error("protocol_attachment_hash_mismatch");
        await transfer("finish", item);
      } finally {
        try { if (!complete) await reader.cancel(); } finally { reader.releaseLock(); }
      }
    }
  } catch (error) {
    // Settle owned page parts before allowing the transport to be reused.
    await transfer("discard");
    throw error;
  }
}

async function discardBrowserActionAttachments(action, ownerInstanceId, transport) {
  const [result] = await runtimeChrome.scripting.executeScript({
    target: { tabId: transport.tab.id }, world: "MAIN", func: transferBrowserActionAttachmentInPage,
    args: [action.action_id, ownerInstanceId, "discard"],
  });
  if (!result?.result?.ok) throw new Error("protocol_attachment_cleanup_response_missing");
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
  const transport = await providerTab("chatgpt", { allowCreate: true });
  const targetUrl = browserActionTargetUrl(action);
  const currentUrl = transport.tab.url || transport.tab.pendingUrl || "";
  if (currentUrl !== targetUrl) {
    await runtimeChrome.tabs.update(transport.tab.id, { url: targetUrl, active: false });
    await waitForProviderTab(transport.tab.id, "chatgpt");
  }
  return transport;
}

function startBrowserActionLeaseHeartbeat(action, ownerInstanceId, phase) {
  let stopped = false;
  let failure = null;
  let inFlight = Promise.resolve();
  const renew = () => {
    inFlight = inFlight.then(async () => {
      if (stopped) return;
      await updateBrowserAction(action.action_id, ownerInstanceId, {
        outcome: "progress",
        phase: typeof phase === "function" ? phase() : phase,
        detail: "renewed browser action lease during provider execution",
      });
    }).catch((error) => {
      failure ||= error;
    });
    return inFlight;
  };
  const timer = globalThis.setInterval(() => {
    void renew();
  }, 60_000);
  return async () => {
    stopped = true;
    globalThis.clearInterval(timer);
    await inFlight;
    if (failure) throw failure;
  };
}

async function dispatchBrowserAction(action, ownerInstanceId) {
  let submitIntentRecorded = false;
  let pageExecutionStarted = false;
  let actionTransport = null;
  try {
    const result = await withProviderTransportOperation(action.provider, async () => {
      const transport = await prepareBrowserActionTransport(action);
      actionTransport = transport;
      const stopHeartbeat = startBrowserActionLeaseHeartbeat(
        action,
        ownerInstanceId,
        () => submitIntentRecorded ? "submit_intent" : "preparing",
      );
      try {
        if (action.attachments?.length) await transferBrowserActionAttachments(action, ownerInstanceId, transport);
        await updateBrowserAction(action.action_id, ownerInstanceId, {
          outcome: "progress",
          phase: action.submit_policy === "submit_once" ? "submit_intent" : "preparing",
          detail: action.submit_policy === "submit_once"
            ? "durable submit intent recorded before the single provider submit boundary"
            : "owned inactive provider target prepared for a staged draft",
        });
        submitIntentRecorded = action.submit_policy === "submit_once";
        pageExecutionStarted = true;
        const [execution] = await runtimeChrome.scripting.executeScript({
          target: { tabId: transport.tab.id },
          world: "MAIN",
          func: executeChatGptBrowserActionInPage,
          args: [action, ownerInstanceId],
        });
        return execution?.result;
      } finally {
        try { if (action.attachments?.length) await discardBrowserActionAttachments(action, ownerInstanceId, transport); }
        finally { await stopHeartbeat(); }
      }
    });
    if (!result?.ok) {
      const error = new Error(result?.detail || "protocol_browser_action_result_missing");
      error.submissionMayHaveOccurred = Boolean(result?.submission_may_have_occurred);
      error.retryAfterSeconds = result?.retry_after_seconds || null;
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
      // transport. Keep its inactive tab, but relinquish ownership so later
      // captures/actions cannot inherit draft text or attachments from it.
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
  // Browser-action polling is autonomous work.  An unpaired profile has no
  // trusted receiver identity, so it must remain entirely local instead of
  // turning the wake alarm into recurring unauthenticated receiver traffic.
  if (!(await storedReceiverPairing())?.receiver_id) {
    return { actions: [], skipped: true, reason: "receiver_unpaired" };
  }
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
  await runtimeChrome.alarms?.create?.(BROWSER_ACTION_ALARM, { delayInMinutes: 0.1, periodInMinutes: 1 });
}

function archiveProviderForUrl(url) {
  try {
    const parsed = new globalThis.URL(url || "");
    if (parsed.hostname === "chatgpt.com" || parsed.hostname.endsWith(".chatgpt.com")) return "chatgpt";
    if (parsed.hostname === "claude.ai" || parsed.hostname.endsWith(".claude.ai")) return "claude-ai";
    if (parsed.hostname === "gemini.google.com" || parsed.hostname.endsWith(".gemini.google.com")) return "gemini";
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

// ChatGPT emits durable freshness hints and has a bounded sweep queue, so an
// already-archived ChatGPT conversation can remain receiver-owned until that
// path reports a change.  Claude and Grok have no equivalent convergence
// signal yet: suppressing their ordinary lifecycle capture would leave later
// turns permanently stale.  Keep that distinction explicit rather than
// treating every provider's `archived` state as equally terminal.
function hasReceiverFreshnessConvergence(provider) {
  return provider === "chatgpt";
}

function conversationIdForUrl(url) {
  try {
    const parsed = new globalThis.URL(url || "");
    const parts = parsed.pathname.split("/").filter(Boolean);
    const provider = archiveProviderForUrl(url);
    if (provider === "chatgpt") {
      const marker = parts.indexOf("c");
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
      // Mirror src/common.js:sessionIdFromUrl exactly. A temporary chat has
      // no /c/<id> path (ChatGPT never persists one), but it is still a
      // real, capturable conversation -- returning null here made every
      // gate downstream that checks `conversationIdForUrl(...)` truthy
      // (captureTab's automatic-capture gate chief among them) treat every
      // temporary chat tab as "no session" and silently never capture it,
      // even though the content-script capture path (common.js) has always
      // been ready to build a per-tab temporary session id. That asymmetry
      // is why zero temporary chats have ever landed in the archive.
      if (parsed.searchParams.get("temporary-chat") === "true") return TEMPORARY_CHAT_SENTINEL;
      return null;
    }
    if (provider === "claude-ai") {
      return parts[0] === "chat" && parts[1] ? parts[1] : null;
    }
    if (provider === "gemini") {
      // Mirror src/content/gemini.js:conversationIdFromUrl exactly, so a
      // Gemini freshness hint routed through captureTab reaches the content script.
      return parsed.searchParams.get("conversation") || parsed.searchParams.get("id") ||
        parsed.pathname.match(/\/app\/([A-Za-z0-9_-]+)/)?.[1] || null;
    }
    if (provider === "grok") {
      // grok.com's own conversation URLs are /c/<uuid> (verified live,
      // 2026-07-31, same convention as ChatGPT/Claude above). The /chat/
      // and /grok/ segment guesses below predate that verification.
      const marker = parts.indexOf("c");
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
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

async function capturedConversationIdForTab(tab) {
  const url = tab?.url || tab?.pendingUrl || "";
  const id = conversationIdForUrl(url);
  if (id !== TEMPORARY_CHAT_SENTINEL) return id;
  // Ask the original document's existing native capture reader. The URL
  // sentinel admits capture, but cannot name an archived conversation.
  await ensureCaptureScripts(tab);
  const identity = await runtimeChrome.tabs.sendMessage(tab.id, { type: "polylogue.captureIdentity", expectedUrl: url });
  const capturedId = identity?.provider_session_id;
  return typeof capturedId === "string" && /^[A-Za-z0-9_-]{1,256}$/.test(capturedId)
    && capturedId !== TEMPORARY_CHAT_SENTINEL ? capturedId : null;
}

async function refreshActiveTabArchiveState(tab, reason = "tab_state", allowRecovery = true) {
  const url = tab?.url || tab?.pendingUrl || "";
  const provider = archiveProviderForUrl(url);
  let providerSessionId = null;
  let throttleKey = null;

  try {
    providerSessionId = await capturedConversationIdForTab(tab);
    throttleKey = `${tab?.id || "active"}:${provider || "unsupported"}:${providerSessionId || "none"}`;
    const now = Date.now();
    const lastCheckedAt = recentActiveTabStateChecks.get(throttleKey) || 0;
    if (now - lastCheckedAt < ACTIVE_TAB_STATE_MIN_INTERVAL_MS) return;
    recentActiveTabStateChecks.set(throttleKey, now);
    if (provider && !providerSessionId && conversationIdForUrl(url) === TEMPORARY_CHAT_SENTINEL) {
      return captureTab(tab, "auto_capture_missing");
    }
    if (provider && providerSessionId) {
      const query = new globalThis.URLSearchParams({ provider, provider_session_id: providerSessionId });
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
      } else if (state.state === "archived" && !hasReceiverFreshnessConvergence(provider)) {
        const captureResult = await captureTab(tab, "auto_capture_unconverged_provider", {
          provider,
          providerSessionId,
          url,
        });
        if (!captureResult || captureResult.skipped) {
          await appendConversationTimeline({
            provider,
            providerSessionId,
            event: "held_with_reason",
            reason: "auto_capture_unconverged_provider",
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
      detail: typeof error.code === "string" ? error.code : "archive_state_check_failed",
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
  if (!runtimeChrome.tabs?.query) {
    await refreshReceiverState();
    return;
  }
  const [tab] = await runtimeChrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    await refreshReceiverState();
    return;
  }
  await refreshActiveTabArchiveState(tab, reason);
}

function stateSnapshotForTab(tab, globalState, ledger, pairing, health, capturedSessionId) {
  const url = tab?.url || tab?.pendingUrl || "";
  const provider = archiveProviderForUrl(url);
  const providerSessionId = capturedSessionId;
  const sameGlobalSession = Boolean(providerSessionId) && globalState?.provider === provider
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

async function missionControlSnapshot(tab = null, { refresh = true, includeIntelligence = false } = {}) {
  const resolvedTab = tab || (runtimeChrome.tabs?.query
    ? (await runtimeChrome.tabs.query({ active: true, currentWindow: true }))[0]
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
    ? coordinator.listStatus({ activeOnly: true }).catch(() => ({ jobs: [], unavailable: true }))
    : Promise.resolve({ jobs: [], unavailable: true });
  const [stored, backfillJobs, ambient] = await Promise.all([
    runtimeChrome.storage.local.get({
      polylogueState: null,
      polylogueSessionLedger: {},
      [CONVERSATION_TIMELINE_KEY]: {},
      [CAPTURE_QUEUE_KEY]: CAPTURE_QUEUE_EMPTY,
      [CAPTURE_FRESHNESS_QUEUE_KEY]: null,
      [RECEIVER_PAIRING_KEY]: null,
    }),
    backfillStatusPromise,
    ambientSettings(hostnameForUrl(tabUrl)),
  ]);
  const pairing = health.pairing || stored[RECEIVER_PAIRING_KEY] || null;
  const capturedSessionId = await capturedConversationIdForTab(resolvedTab).catch(() => null);
  const baseState = stateSnapshotForTab(
    resolvedTab,
    stored.polylogueState,
    stored.polylogueSessionLedger || {},
    pairing,
    health,
    capturedSessionId,
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
  const intelligence = includeIntelligence ? await missionIntelligenceProjection(state, settings.baseUrl) : null;
  let assertionCapability = false;
  if (includeIntelligence && receiverOnline) {
    try {
      const capabilities = await getJson("/v1/browser-captures/capabilities");
      // Only the declared top-level field is authoritative; any other shape
      // fails closed and leaves Save unavailable.
      assertionCapability = capabilities?.assertion_candidates === true;
    } catch { /* An unreachable capability probe fails closed. */ }
  }
  // Queued behind the startup migration and any capture's identity write.
  const acceptedIdentityMap = await serializeStorageMutation(
    () => runtimeChrome.storage.local.get({ [ACCEPTED_MESSAGE_IDENTITIES_KEY]: {} }),
  );
  const acceptedIdentities = state.provider && state.provider_session_id
    ? acceptedIdentityMap[ACCEPTED_MESSAGE_IDENTITIES_KEY]?.[sessionKey(state.provider, state.provider_session_id)] || {}
    : null;

  return {
    ok: true,
    generated_at: new Date().toISOString(),
    extension: {
      contract_epoch: EXTENSION_CONTRACT_EPOCH,
      manifest_version: runtimeChrome.runtime.getManifest?.().version || null,
      extension_id: runtimeChrome.runtime.id || null,
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
      capture_queue: await captureQueuePage(),
      freshness_queue: freshnessQueue,
      backfill_jobs: backfillJobs.jobs,
      backfill_job_page: { total: backfillJobs.total, cursor: backfillJobs.cursor, has_more: backfillJobs.has_more, unavailable: backfillJobs.unavailable || false },
    },
    ambient,
    assertions: {
      selection_candidate_supported: true,
      persistence_supported: assertionCapability,
      accepted_identities: acceptedIdentities,
      reason: assertionCapability ? "candidate_assertion_route" : "receiver_capability_unavailable",
    },
    ...(includeIntelligence ? { intelligence } : {}),
  };
}

export function startBackgroundRuntime(adapters) {
  // A test harness (and a browser profile that reloads an unpacked worker)
  // can evaluate the composition root more than once while this module stays
  // cached. Recreate process-local coordination so durable storage remains
  // the only recovery authority.
  recentBackgroundCaptures.clear();
  recentActiveTabStateChecks.clear();
  providerTransportPromises.clear();
  providerTransportOperations.clear();
  backfillCoordinatorPromise = null;
  backfillStartPromises.clear();
  extensionInstanceIdPromise = null;
  browserActionExecutorIdPromise = null;
  browserActionPollPromise = null;
  captureFreshnessPollPromise = null;
  storageMutationQueue = Promise.resolve();
  captureQueueMutationQueue = Promise.resolve();
  trustedReceiverHealthCache = null;
  cachedQueueLength = 0;
  runtimeChrome = adapters;
  // Retire stored credentials without reading or transferring their value.
  void runtimeChrome.storage.local.remove("receiverAuthToken");
  runtimeNetwork = async (input, init = {}) => adapters.network(input, {
    ...init, receiverId: (await storedReceiverPairing())?.receiver_id || null,
  });
  captureStore = captureRetryStore;
  captureStaging = adapters.captureStaging || new CaptureStaging(globalThis.navigator?.storage, captureStore);
  runtimeWorkerId = globalThis.crypto.randomUUID();
  nativeNormalizer = new NativeCaptureNormalizer({ staging: captureStaging, store: captureStore, prepareNative: prepareNativeCapture });
void reconcileCaptureRoots().then(loadCaptureQueueIntoCache).catch((error) => appendDebugLog({ stage: "capture_staging_recovery_failed", error: String(typeof error.code === "string" ? error.code : (error.message || error)) }));
void replaceLegacyAcceptedMessageIdentities();
void ensureBrowserActionAlarm();
void ensureCaptureFreshnessAlarms();

  registerBackgroundEvents(runtimeChrome, {
    browserActions: () => pollBrowserActions(),
    captureFreshness: () => processCaptureFreshnessQueue(),
    captureFreshnessSweep: () => runCaptureFreshnessSweep(),
    providerTransportCleanup: (alarmName) => cleanupBackfillTransportTab(alarmName),
    captureRetry: () => drainCaptureQueue("alarm"),
    backfill: (jobId) => backfillCoordinator().then((coordinator) => coordinator.wake(jobId)),
    installed: () => captureSupportedTabs("extension_installed_or_updated"),
    startup: () => Promise.all([
      captureSupportedTabs("browser_startup"),
      backfillCoordinator().then((coordinator) => coordinator.wake()),
      ensureCaptureFreshnessAlarms(),
    ]),
    activated: async (activeInfo) => {
      const tab = await runtimeChrome.tabs.get(activeInfo.tabId);
      await refreshActiveTabArchiveState(tab, "tab_activated");
    },
    updated: async (tabId, tab) => {
      const resolvedTab = tab?.id ? tab : await runtimeChrome.tabs.get(tabId);
      await refreshActiveTabArchiveState(resolvedTab, "tab_updated");
    },
    removed: (tabId) => Promise.all([
      forgetProviderTransport("chatgpt", tabId),
      forgetProviderTransport("claude-ai", tabId),
    ]),
  });

runtimeChrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    if (message.type === "polylogue.reserveCaptureObservation" || message.type === "polylogue.cancelCaptureObservation") {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      const nativeId = conversationIdForUrl(sender.tab?.url || "");
      const owner = { tab_id: sender.tab?.id, document_id: sender.documentId || null, provider };
      if (provider !== "gemini" || provider !== message.provider) throw new Error("capture_observation_identity_mismatch");
      if (message.type === "polylogue.cancelCaptureObservation") {
        const row = await captureStore.getCapture(message.observation_ref?.id);
        if (row && (row.kind !== "dom-observation" || row.native_id !== message.native_id || row.token !== message.observation_ref.token || JSON.stringify(row.owner) !== JSON.stringify(owner))) throw new Error("capture_observation_owner_mismatch");
        if (row) await captureStore.discardCapture(row.id);
        sendResponse({ ok: true, outcome: "cancelled" }); return;
      }
      if (nativeId !== message.native_id) throw new Error("capture_observation_identity_mismatch");
      await requirePairedTrustedReceiver();
      const ref = await captureStore.reserveCaptureObservation(owner, nativeId);
      sendResponse({ ok: true, observation_ref: ref }); return;
    }
    if (message.type === "polylogue.missionControl.status") {
      sendResponse(await missionControlSnapshot(sender.tab || null, { refresh: message.refresh !== false, includeIntelligence: message.include_intelligence === true }));
      return;
    }
    if (message.type === "polylogue.providerThrottle") {
      const queue = await storedCaptureFreshnessQueue();
      const deadline = Number(queue.provider_cooldowns[message.provider]) || 0;
      const now = Date.now();
      sendResponse(deadline > now
        ? {
          ok: false,
          outcome: "rate_limited",
          retry_after_seconds: Math.ceil((deadline - now) / 1000),
        }
        : { ok: true });
      return;
    }
    if (message.type === "polylogue.providerRateLimited") {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      if (!provider || provider !== message.provider) throw new Error("provider_rate_limit_sender_invalid");
      if (message.provider_response?.status !== 429 || typeof message.request_id !== "string" || !message.request_id) throw new Error("provider_rate_limit_response_invalid");
      if (message.claim) {
        const meta = await captureStaging.metadata(message.claim.id);
        const owner = { tab_id: sender.tab.id, document_id: sender.documentId || null, provider };
        captureStaging.requireOwner(meta, message.claim, owner);
        const responseClaim = ["native-response", "provider-inventory"].includes(meta.kind) &&
          meta.state === "acquiring" && meta.id === captureStaging.producerStageId(owner, message.request_id) &&
          meta.source_url === message.provider_response.url && archiveProviderForUrl(meta.source_url || "") === provider;
        if (!meta.acquisition && !responseClaim) throw new Error("provider_rate_limit_claim_invalid");
      } else if (archiveProviderForUrl(message.provider_response.url || "") !== provider) throw new Error("provider_rate_limit_response_invalid");
      // This binds the report to its provider document. A MAIN-world response
      // nonce correlates work; it is visible to page scripts, not authentication.
      const error = new Error("provider_rate_limited");
      error.outcome = "rate_limited";
      error.retryAfterSeconds = Number.isFinite(message.retry_after_seconds)
        ? message.retry_after_seconds
        : null;
      error.retryAfterMs = typeof message.retry_after === "string"
        ? retryAfterMs({ get: () => message.retry_after }, Date.now())
        : error.retryAfterSeconds === null ? null : error.retryAfterSeconds * 1000;
      if (error.retryAfterSeconds === null && error.retryAfterMs !== null) error.retryAfterSeconds = error.retryAfterMs / 1000;
      await recordProviderThrottle(message.provider, error, classifyBrowserActionFailure(error));
      sendResponse({ ok: true });
      return;
    }
    if (message.type === "polylogue.receiverPairing.status") {
      const health = await checkReceiverHealth();
      sendResponse({ ok: true, health, pairing: health.pairing || await storedReceiverPairing() });
      return;
    }
    if (message.type === "polylogue.receiverPairing.reset") {
      let reset;
      try {
        if (Object.hasOwn(message, "expectedConfigurationRevision") && !Number.isSafeInteger(message.expectedConfigurationRevision)) throw new Error("receiver_configuration_changed");
        reset = await clearReceiverPairing(message.expectedConfigurationRevision ?? null);
        const health = await checkReceiverHealth({ allowCanonicalRecovery: false, expectedScope: reset });
        sendResponse({ ok: true, health, pairing: health.pairing, configurationRevision: reset.revision });
      } catch (error) {
        sendResponse({ ok: false, error: error?.message || "receiver_pairing_reset_failed", configurationRevision: reset?.revision ?? null });
      }
      return;
    }
    if (message.type === "polylogue.ambient.configure") {
      const url = sender.tab?.url || sender.tab?.pendingUrl || message.url || "";
      const settings = await saveAmbientSettings({
        enabled: message.enabled ?? null,
        automaticCaptureEnabled: message.automatic_capture_enabled ?? null,
        hostname: message.hostname || hostnameForUrl(url),
        siteEnabled: message.site_enabled ?? null,
      });
      if (settings.automatic_capture_enabled) await ensureCaptureFreshnessAlarms();
      else {
        await runtimeChrome.alarms?.clear?.(CAPTURE_FRESHNESS_ALARM);
        const tabs = await runtimeChrome.tabs.query({});
        await Promise.allSettled(tabs.filter((tab) => ["chatgpt", "claude-ai", "grok", "gemini"].includes(archiveProviderForUrl(tab.url || tab.pendingUrl || "")))
          .map((tab) => runtimeChrome.tabs.sendMessage(tab.id, { type: "polylogue.cancelCapture" })));
      }
      sendResponse({ ok: true, ambient: settings });
      return;
    }
    if (message.type === "polylogue.assertion.capture") {
      const candidate = message.candidate;
      if (!candidate || candidate.context_policy?.inject !== false) {
        sendResponse({ ok: false, error: "assertion_candidate_policy_required" });
        return;
      }
      if (!candidate.evidence_ref || !candidate.message_ref || !candidate.source_observation) {
        sendResponse({ ok: false, error: "exact_message_evidence_required" });
        return;
      }
      try {
        await requirePairedTrustedReceiver();
        const result = await postJson("/v1/assertion-candidates", {
          body_text: candidate.body,
          kind: candidate.kind,
          evidence_refs: [candidate.evidence_ref],
          target_ref: candidate.message_ref,
          source_observation: candidate.source_observation,
          author_ref: "user:browser-extension",
          author_kind: "user",
          idempotency_key: candidate.idempotency_key,
          context_policy: { inject: false },
        });
        sendResponse({ ok: true, idempotent: result.status === "already_satisfied", candidate: result });
      } catch (error) {
        sendResponse({ ok: false, error: String(error?.message || error) });
      }
      return;
    }
    if (message.type === "polylogue.configureReceiver") {
      let settings;
      try {
        const restoring = Object.hasOwn(message, "restore");
        if (restoring && (!message.restore || typeof message.restore !== "object" || Array.isArray(message.restore))) {
          throw new Error("proof_receiver_configuration_changed");
        }
        settings = restoring
          ? await restoreReceiverSettings(message.restore.previous, message.restore.owned)
          : await saveReceiverSettings(message.receiverBaseUrl || DEFAULT_RECEIVER);
      } catch (error) {
        sendResponse({ ok: false, error: error?.message || "configure_receiver_failed" });
        return;
      }
      sendResponse({ ok: true, receiverBaseUrl: settings.baseUrl, configurationRevision: settings.configurationRevision });
      return;
    }
    if (message.type === "polylogue.backfill.start") {
      const job = await startBackfill(message);
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
      sendResponse({ ok: true, ...await coordinator.listStatus({ cursor: message.cursor || null, pageSize: message.pageSize || 25, activeOnly: message.activeOnly === true }) });
      return;
    }
    if (["polylogue.backfill.export", "polylogue.backfill.exportAck"].includes(message.type)) {
      if (sender.url !== `chrome-extension://${runtimeChrome.runtime.id}/src/popup.html`) throw new Error("checkpoint_export_sender_invalid");
      if (message.type === "polylogue.backfill.exportAck") {
        const snapshot = await captureStore.getCapture(message.snapshot_id);
        if (snapshot?.kind !== "checkpoint-export" || snapshot.token !== message.token) throw new Error("checkpoint_export_owner_mismatch");
        const receipt = { digest: message.digest, size_bytes: message.size_bytes, outcome: "exported" };
        if (snapshot.state === "acknowledged") {
          if (snapshot.export_receipt.digest !== receipt.digest || snapshot.export_receipt.size_bytes !== receipt.size_bytes) throw new Error("checkpoint_export_receipt_conflict");
        } else {
          const meta = await captureStaging.metadata(snapshot.artifact_id);
          if (`sha256:${meta.sha256}` !== receipt.digest || meta.bytes !== receipt.size_bytes || !["sealed", "checkpoint-acknowledged"].includes(meta.state)) throw new Error("checkpoint_export_receipt_conflict");
        }
        await captureStaging.releaseCheckpoint(snapshot, receipt);
        sendResponse({ ok: true, outcome: "exported" }); return;
      }
      let snapshot = await captureStore.pendingRecoverySnapshot(message.job_id, "checkpoint-export");
      if (!snapshot) {
        const id = `export:${globalThis.crypto.randomUUID()}`;
        const artifactId = captureStaging.producerStageId({ checkpoint_snapshot_id: id }, id);
        snapshot = await captureStore.createRecoverySnapshot(message.job_id, id, artifactId, { kind: "checkpoint-export", token: globalThis.crypto.randomUUID() });
      }
      const prepared = await captureStaging.prepareCheckpoint(snapshot);
      sendResponse({ ok: true, snapshot_id: snapshot.id, token: snapshot.token, artifact_ref: prepared.ref, digest: prepared.digest, size_bytes: prepared.sizeBytes });
      return;
    }
    if (message.type === "polylogue.releaseNativeCache") {
      if (!sender.tab?.id) throw new Error("capture_staging_sender_invalid");
      for await (const ref of captureStore.releaseNativeCaches({ tab_id: sender.tab.id, document_id: sender.documentId || null })) {
        await captureStaging.discardUnreferenced(ref?.id || ref);
      }
      sendResponse({ ok: true }); return;
    }
    if (message.type?.startsWith("polylogue.nativeBundle.")) {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      if (provider !== "grok" || message.provider !== provider) throw new Error("native_bundle_owner_mismatch");
      const owner = { tab_id: sender.tab.id, document_id: sender.documentId || null, provider };
      if (message.type === "polylogue.nativeBundle.begin") {
        await requirePairedTrustedReceiver(); await requireProviderThrottleAvailability(provider);
        if (conversationIdForUrl(sender.tab.url) !== message.native_id) throw new Error("native_capture_identity_mismatch");
        const bundle = await captureStore.beginNativeBundle({ extensionInstanceId: await extensionInstanceId(), owner, provider, nativeId: message.native_id, bundleId: message.bundle_id,
          requiredReplies: ["conversation", "responses"] });
        sendResponse({ ok: true, bundle_ref: bundle.id, replies: bundle.replies }); return;
      }
      const bundle = await captureStore.getCapture(message.bundle_ref);
      if (message.type === "polylogue.nativeBundle.cancel") {
        for (const operation of nativeNormalizations.values()) {
          if (operation.bundleId !== message.bundle_ref) continue;
          if (JSON.stringify(operation.owner) !== JSON.stringify(owner)) throw new Error("native_bundle_owner_mismatch");
          operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
          await operation.drained;
        }
        if (bundle && JSON.stringify(bundle.owner) !== JSON.stringify(owner)) throw new Error("native_bundle_owner_mismatch");
        sendResponse({ ok: true }); return;
      }
      if (bundle?.kind !== "native-bundle" || JSON.stringify(bundle.owner) !== JSON.stringify(owner)) throw new Error("native_bundle_owner_mismatch");
      if (message.type === "polylogue.nativeBundle.outcome") {
        await captureStore.publishNativeBundleReply(bundle.id, owner, message.name, null, message.outcome);
        await captureStore.finishNativeBundle(bundle.id, owner);
        sendResponse({ ok: true }); return;
      }
      if (message.type !== "polylogue.nativeBundle.finish") throw new Error("native_bundle_operation_invalid");
      if (conversationIdForUrl(sender.tab.url) !== bundle.native_id) throw new Error("native_capture_identity_mismatch");
      const controller = new globalThis.AbortController();
      const operation = { rawId: bundle.replies.conversation?.id, bundleId: bundle.id, owner, controller,
        promise: nativeNormalizer.finishBundle(bundle.id, owner, { pin: true, signal: controller.signal }) };
      trackNativeOperation(operation);
      let result;
      try { result = await operation.promise; controller.signal.throwIfAborted(); }
      finally { operation.finish(); }
      sendResponse({ ok: true, acquisition: result.acquisition }); return;
    }
    if (message.type === "polylogue.cancelNativeRecovery") {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      const owner = { tab_id: sender.tab?.id, document_id: sender.documentId || null, provider };
      if (!provider || provider !== message.provider) throw new Error("capture_staging_sender_invalid");
      let operation = null;
      for (const value of nativeNormalizations.values()) {
        if (value.recoveryId === message.request_id) { operation = value; break; }
      }
      if (operation) {
        if (JSON.stringify(operation.owner) !== JSON.stringify(owner)) throw new Error("capture_staging_owner_mismatch");
        operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError")); await operation.drained;
      }
      sendResponse({ ok: true, outcome: "cancelled" }); return;
    }
    if (message.type === "polylogue.closeNativeInvocation") {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      if (!provider) throw new Error("capture_staging_sender_invalid");
      const owner = { tab_id: sender.tab.id, document_id: sender.documentId || null, provider };
      const row = await captureStore.getCapture(message.invocation_ref?.id);
      if (!row || row.kind !== "native-invocation" || row.token !== message.invocation_ref.token ||
          JSON.stringify(row.owner) !== JSON.stringify(owner)) throw new Error("native_invocation_owner_mismatch");
      await captureStore.closeNativeInvocation(message.invocation_ref, false);
      for (const operation of nativeNormalizations.values()) {
        if (operation.invocationRef?.id === row.id && operation.invocationRef.token === row.token) {
          operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
        }
      }
      sendResponse({ ok: true }); return;
    }
    if (message.type === "polylogue.restoreNativeCapture") {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      const pageId = conversationIdForUrl(sender.tab?.url || "");
      if (!provider || provider !== message.provider) throw new Error("native_capture_identity_mismatch");
      const owner = { tab_id: sender.tab.id, document_id: sender.documentId || null, provider };
      if (!message.invocation_ref && pageId !== TEMPORARY_CHAT_SENTINEL && pageId !== message.native_id) throw new Error("native_capture_identity_mismatch");
      if (!message.request_id) throw new Error("capture_staging_request_invalid");
      for (const value of nativeNormalizations.values()) {
        if (value.recoveryId === message.request_id) throw new Error("capture_staging_request_conflict");
      }
      const controller = new globalThis.AbortController();
      const operation = { recoveryId: message.request_id, owner, controller, invocationRef: message.invocation_ref || null };
      // Register before the first await: the following cancellation message
      // must see this owner even while the first durable cursor is blocked.
      trackNativeOperation(operation);
      try {
      if (message.invocation_ref) await requireNativeInvocation(message.invocation_ref, owner, message.native_id);
      controller.signal.throwIfAborted();
      let selected = null;
      for await (const row of captureStore.captures()) {
        controller.signal.throwIfAborted();
        if (row.provider !== provider || row.native_id !== message.native_id || JSON.stringify(row.owner) !== JSON.stringify(owner)) continue;
        if (row.kind === "native-bundle") {
          if (row.queue_context) continue;
          // Incomplete evidence belongs to this exact operation. It cannot be
          // silently replaced by another provider acquisition after restart.
          if (!row.replies.conversation || !row.replies.responses) continue;
          if (Object.values(row.outcomes).some((outcome) => outcome.status === 429)) continue;
          const result = await nativeNormalizer.finishBundle(row.id, owner, { pin: true, signal: controller.signal });
          const raw = await captureStaging.metadata(result.rawRef.id);
          const candidate = { ...result, source_url: raw.source_url, observed_at: raw.observed_at };
          if (!selected || nativeCacheOrder(selected, candidate) > 0) selected = candidate;
        } else if (row.kind === "native-cache" && (!selected || nativeCacheOrder(selected, row) > 0)) {
          selected = { headers: row.headers, rawRef: row.raw_ref, relatedRefs: row.related_refs, acquisition: row.acquisition,
            acquisition_sequence: row.acquisition_sequence, observed_at: row.observed_at };
        }
      }
      if (provider !== "grok") {
        for await (const meta of captureStaging.metadataEntries()) {
          controller.signal.throwIfAborted();
          if (meta.kind !== "native-response" || meta.queue_context || meta.state !== "sealed" || JSON.stringify(meta.owner) !== JSON.stringify(owner)) continue;
          const ref = { id: meta.id, token: meta.token }; const headers = await nativeNormalizer.headers(ref, controller.signal);
          const id = nativeSessionIdFromHeaders(provider, headers);
          if (String(id || "") !== message.native_id) continue;
          const candidate = { headers, rawRef: ref, relatedRefs: {}, acquisition: null,
            acquisition_sequence: meta.acquisition_sequence, source_url: meta.source_url, observed_at: meta.observed_at };
          if (!selected || nativeCacheOrder(selected, candidate) > 0) selected = candidate;
        }
        controller.signal.throwIfAborted();
        if (selected) await captureStore.pinNativeCache({ owner, provider, nativeId: message.native_id, rawRef: selected.rawRef,
          headers: selected.headers, observedAt: selected.observed_at, acquisitionSequence: selected.acquisition_sequence });
      }
      if (selected) {
        if (pageId === TEMPORARY_CHAT_SENTINEL && selected.headers.is_temporary !== true) throw new Error("native_temporary_identity_mismatch");
        const meta = await captureStaging.metadata(selected.rawRef.id); captureStaging.requireOwner(meta, selected.rawRef, owner);
        await captureStaging.file(meta.id);
        selected = { ok: true, bodyRef: selected.rawRef, relatedRefs: selected.relatedRefs, acquisition: selected.acquisition,
          url: selected.source_url || meta.source_url, capturedAt: selected.observed_at, headers: selected.headers, acquisitionSequence: selected.acquisition_sequence };
      }
      controller.signal.throwIfAborted(); sendResponse({ ok: true, capture: selected });
      } finally { operation.finish(); }
      return;
    }
    if (message.type === "polylogue.normalizeNativeCapture" || message.type === "polylogue.nativeCaptureSummary" || message.type === "polylogue.cancelNativeCapture" || message.type === "polylogue.nativeCaptureHeader") {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      if (!provider || provider !== message.provider) throw new Error("capture_staging_sender_invalid");
      const owner = { tab_id: sender.tab.id, document_id: sender.documentId || null, provider };
      const progressContext = message.type === "polylogue.normalizeNativeCapture" ? { acquisition_ref: message.raw_ref?.id, native_request_id: message.native_request_id } : null;
      const meta = await captureStaging.metadata(message.raw_ref?.id);
      captureStaging.requireOwner(meta, message.raw_ref, owner);
      for (const ref of Object.values(message.related_refs || {})) captureStaging.requireOwner(await captureStaging.metadata(ref.id), ref, owner);
      if (message.type === "polylogue.nativeCaptureHeader") {
        const controller = new globalThis.AbortController();
        const operation = { rawId: meta.id, controller, invocationRef: message.invocation_ref || null, promise: nativeNormalizer.headers(message.related_refs?.conversation || message.raw_ref, controller.signal) };
        trackNativeOperation(operation);
        try {
          const headers = await operation.promise; controller.signal.throwIfAborted();
          const nativeId = nativeSessionIdFromHeaders(provider, headers);
          if (!nativeId) throw new Error("native_capture_identity_missing");
          const pageId = conversationIdForUrl(sender.tab.url);
          if (message.invocation_ref) {
            await requireNativeInvocation(message.invocation_ref, owner, String(nativeId));
            const source = providerRequestFromUrl(meta.source_url || "");
            if (source.provider !== provider || source.operation !== "conversation" || source.params.nativeId !== String(nativeId) ||
                (meta.invocation_native_id && meta.invocation_native_id !== String(nativeId))) throw new Error("native_capture_identity_mismatch");
          } else if (pageId === TEMPORARY_CHAT_SENTINEL ? headers.is_temporary !== true : pageId !== String(nativeId)) throw new Error("native_capture_identity_mismatch");
          const pinned = await captureStore.pinNativeCache({ owner, provider, nativeId: String(nativeId), rawRef: message.raw_ref,
            relatedRefs: message.related_refs || {}, acquisition: message.acquisition || null, headers, observedAt: meta.observed_at, acquisitionSequence: meta.acquisition_sequence });
          for (const retired of pinned.retired) await captureStaging.discardUnreferenced(retired?.id || retired);
          const current = pinned.current; const selectedMeta = await captureStaging.metadata(current.raw_ref.id);
          controller.signal.throwIfAborted();
          const describe = (ref, relatedRefs, acquisition, revisionHeaders, revisionMeta) => ({
            ok: true, bodyRef: ref, relatedRefs, acquisition, nativeId: String(nativeId),
            providerUpdatedAt: revisionHeaders.update_time ?? revisionHeaders.updated_at ?? revisionHeaders.updatedAt ?? revisionHeaders.modifyTime ?? null,
            url: revisionMeta.source_url, capturedAt: revisionMeta.created_at, acquisitionSequence: revisionMeta.acquisition_sequence,
          });
          sendResponse({ ok: true, headers, content_sha256: meta.sha256,
            capture: describe(message.raw_ref, message.related_refs || {}, message.acquisition || null, headers, meta),
            cache: { headers: current.headers, capture: describe(current.raw_ref, current.related_refs, current.acquisition, current.headers, selectedMeta) } });
        } finally { operation.finish(); }
        return;
      }
      const urlId = conversationIdForUrl(sender.tab.url);
      nativePreparationProgress(progressContext, "normalize_admission", "BEGIN");
      if (message.type === "polylogue.normalizeNativeCapture" || message.type === "polylogue.nativeCaptureSummary") {
        if (message.invocation_ref) {
          await requireNativeInvocation(message.invocation_ref, owner, message.native_id);
          const source = providerRequestFromUrl(meta.source_url || "");
          if (source.provider !== provider || source.operation !== "conversation" || source.params.nativeId !== message.native_id ||
              (meta.invocation_native_id && meta.invocation_native_id !== message.native_id)) throw new Error("native_capture_identity_mismatch");
        } else if (urlId !== TEMPORARY_CHAT_SENTINEL && urlId !== message.native_id) throw new Error("native_capture_identity_mismatch");
      }
      if (message.type === "polylogue.cancelNativeCapture") {
        // One raw revision can have a header read and normalization in flight.
        // Abort every reader first, then drain them all before acknowledging.
        const drains = [];
        for (const operation of nativeNormalizations.values()) {
          if (operation.rawId !== meta.id) continue;
          operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
          drains.push(operation.drained);
        }
        await Promise.all(drains);
        sendResponse({ ok: true, outcome: "cancelled" }); return;
      }
      await requirePairedTrustedReceiver(); await requireProviderThrottleAvailability(provider);
      if (message.invocation_ref) await requireNativeInvocation(message.invocation_ref, owner, message.native_id);
      nativePreparationProgress(progressContext, "normalize_admission", "END");
      const controller = new globalThis.AbortController();
      const onProgress = (phase, state) => { if (!controller.signal.aborted) nativePreparationProgress(progressContext, phase, state); };
      const operation = { rawId: meta.id, controller, invocationRef: message.invocation_ref || null, promise: null };
      trackNativeOperation(operation);
      operation.promise = (async () => {
      if (message.type !== "polylogue.nativeCaptureSummary") await drainOwnedNativeBundles(owner, message.native_id, meta.acquisition_sequence, controller.signal);
      return nativeNormalizer.normalize({ provider, rawRef: message.raw_ref, nativeId: message.native_id,
        extensionVersion: runtimeChrome.runtime.getManifest().version, instanceId: await extensionInstanceId(), attribution: message.attribution || {}, signal: controller.signal, relatedRefs: message.related_refs || {}, acquisition: message.acquisition || null, requireTemporary: urlId === TEMPORARY_CHAT_SENTINEL, summaryOnly: message.type === "polylogue.nativeCaptureSummary", onProgress });
      })();
      try {
        const envelope = await operation.promise;
        if (message.type === "polylogue.nativeCaptureSummary") {
          controller.signal.throwIfAborted();
          sendResponse({ ok: true, summary: envelope.summary, raw_revision: envelope.rawRevision, content_sha256: meta.sha256 });
          return;
        }
        await captureStaging.retireFailedNormalizations(meta.id, envelope.capture_record_ref, controller.signal)
          .catch((error) => appendDebugLog({ stage: "native_normalization_cleanup_pending", ref: envelope.capture_record_ref, error: String(typeof error.code === "string" ? error.code : (error.message || error)) }));
        const headers = await nativeNormalizer.headers(message.related_refs?.conversation || message.raw_ref, controller.signal);
        const pinned = await captureStore.pinNativeCache({ owner, provider, nativeId: envelope.session.provider_session_id, rawRef: message.raw_ref,
          relatedRefs: message.related_refs || {}, acquisition: message.acquisition || null, headers, observedAt: meta.created_at, acquisitionSequence: meta.acquisition_sequence });
        for (const retired of pinned.retired) await captureStaging.discardUnreferenced(retired?.id || retired);
        controller.signal.throwIfAborted();
        sendResponse({ ok: true, envelope });
      }
      catch (error) { sendResponse({ ok: false, error: typeof error.code === "string" ? error.code : String(error.message || error), outcome: controller.signal.aborted ? "cancelled" : "failed" }); }
      finally { operation.finish(); }
      return;
    }
    if (message.type.startsWith("polylogue.asset.")) {
      const provider = archiveProviderForUrl(sender.tab?.url || "");
      if (!["chatgpt", "claude-ai", "grok"].includes(provider) || (message.provider && message.provider !== provider)) throw new Error("capture_staging_sender_invalid");
      const owner = { tab_id: sender.tab.id, document_id: sender.documentId || null, provider };
      if (message.type === "polylogue.asset.begin") {
        await requirePairedTrustedReceiver();
        await requireProviderThrottleAvailability(provider);
        if (typeof message.request_id !== "string" || !message.request_id) throw new Error("capture_staging_request_invalid");
        if (message.invocation_ref) {
          const requested = providerRequestFromUrl(message.source_url || "");
          if (message.kind !== "native-response" || message.observation_only || message.acquisition || message.capture_bundle || message.queue_context ||
              requested.provider !== provider || requested.operation !== "conversation") throw new Error("native_invocation_request_invalid");
          await requireNativeInvocation(message.invocation_ref, owner, requested.params.nativeId);
        }
        let identity = null;
        if (message.acquisition) {
          const acquisition = message.acquisition;
          const raw = await captureStaging.metadata(acquisition.raw_id);
          captureStaging.requireOwner(raw, { id: raw.id, token: raw.token }, owner);
          if (raw.state !== "sealed" || !acquisition.native_id || acquisition.record_key == null || !acquisition.attachment_id) throw new Error("capture_acquisition_identity_invalid");
          if (!Number.isSafeInteger(acquisition.attachment_ordinal) || acquisition.attachment_ordinal < 0) throw new Error("capture_acquisition_identity_invalid");
          identity = [provider, raw.id, acquisition.native_id, acquisition.record_key, acquisition.attachment_id, acquisition.attachment_ordinal];
        }
        if (message.capture_bundle) {
          const bundle = await captureStore.getCapture(message.capture_bundle.id);
          if (bundle?.kind !== "native-bundle" || bundle.provider !== provider || bundle.owner.tab_id !== owner.tab_id ||
              (bundle.owner.document_id && bundle.owner.document_id !== owner.document_id) || !["conversation", "responses", "response_nodes"].includes(message.capture_bundle.name)) throw new Error("native_bundle_owner_mismatch");
        }
        if (message.queue_context) {
          const requested = providerRequestFromUrl(message.source_url || "");
          if (message.kind !== "native-response" || message.observation_only || requested.provider !== provider ||
              requested.operation !== "conversation" || requested.params.nativeId !== message.queue_context.nativeId) throw new Error("native_acquisition_claim_invalid");
          const job = await captureStore.assertJobExecution(message.queue_context.jobId, message.queue_context.owner, message.queue_context.generation);
          if (typeof message.account_handle !== "string" || !message.account_handle) throw new Error("capture_job_account_scope_unresolved");
          const settings = await receiverSettings();
          const client = new CaptureJobClient({ baseUrl: settings.baseUrl, cache: runtimeChrome.storage.local, fetchImpl: runtimeNetwork });
          const observedScope = await deriveAccountScope(await client.scopeNamespace(), provider, message.account_handle);
          if (observedScope !== job.account_scope) throw new Error("capture_job_account_scope_mismatch");
        }
        const ref = await captureStaging.begin(owner, { kind: message.kind, source_url: message.source_url, response_metadata: message.response_metadata,
          observation_only: message.observation_only === true, queue_context: message.queue_context || null,
          acquisition: identity, capture_bundle: message.capture_bundle, invocation_ref: message.invocation_ref || null, extension_instance_id: await extensionInstanceId() }, message.request_id);
        if (message.invocation_ref) {
          const requested = providerRequestFromUrl(message.source_url);
          await requireNativeInvocation(message.invocation_ref, owner, requested.params.nativeId);
        }
        if (message.queue_context) await captureStore.bindNativeAcquisition(ref, owner, message.queue_context);
        if (identity) {
          // Publish a single acquisition owner before granting provider traffic.
          // A retransmitted begin recovers that owner's immutable staged bytes.
          const claim = await captureStore.reserveAcquisition(identity, ref, message.request_id);
          if (claim.ref.id !== ref.id) await captureStaging.discard(ref.id);
          const asset = await captureStaging.metadata(claim.ref.id);
          if (asset.state === "sealed" || asset.acquisition_result) {
            await captureStaging.seal(claim.ref, owner);
            const published = await captureStore.getCapture(claim.id);
            sendResponse({ ok: true, result: published.result }); return;
          }
          if (claim.producer_id !== message.request_id) {
            sendResponse({ ok: false, error: "capture_acquisition_in_progress" }); return;
          }
          if (asset.sequence !== 0) {
            // A surviving producer retries chunks against its ref, never begins
            // a second provider read to replace an interrupted acquisition.
            sendResponse({ ok: false, error: "capture_acquisition_interrupted" }); return;
          }
        }
        sendResponse({ ok: true, ref });
      } else if (message.type === "polylogue.asset.cancelRequest") {
        if (typeof message.request_id !== "string" || !message.request_id) throw new Error("capture_staging_request_invalid");
        const stageId = captureStaging.producerStageId(owner, message.request_id);
        try {
          const meta = await captureStaging.metadata(stageId);
          await captureStaging.cancel({ id: meta.id, token: meta.token }, owner);
        } catch (error) { if (error.code !== "capture_staging_interrupted") throw error; }
        sendResponse({ ok: true });
      } else if (message.type === "polylogue.asset.chunk") {
        sendResponse({ ok: true, ...await captureStaging.append(message.ref, owner, message.sequence, message.base64) });
      } else if (message.type === "polylogue.asset.seal") {
        const asset = await captureStaging.seal(message.ref, owner, message.result);
        sendResponse({ ok: true, asset });
      } else if (message.type === "polylogue.asset.discard") {
        await captureStaging.cancel(message.ref, owner);
        sendResponse({ ok: true });
      } else throw new Error("capture_staging_operation_invalid");
      return;
    }
    if (message.type === "polylogue.cancelCaptureDelivery") {
      const operation = captureDeliveries.get(message.request_id);
      if (operation) {
        const owner = captureDeliveryOwner(sender);
        if (JSON.stringify(owner) !== JSON.stringify(operation.owner)) throw new Error("capture_delivery_owner_mismatch");
        const error = new Error("capture_cancelled"); error.code = "capture_cancelled"; error.name = "AbortError";
        operation.controller.abort(error);
        await operation.promise.catch(() => undefined);
      }
      sendResponse({ ok: true }); return;
    }
    if (message.type === "polylogue.capture") {
      const requestId = message.request_id || buildReceiverRequestId();
      if (captureDeliveries.has(requestId)) throw new Error("capture_delivery_request_conflict");
      const operation = { owner: captureDeliveryOwner(sender), controller: new globalThis.AbortController(), promise: null };
      captureDeliveries.set(requestId, operation);
      operation.promise = (async () => {
      const envelope = await withExtensionInstanceAttribution(message.envelope, sender);
      const summary = envelopeSessionSummary(envelope);
      let result;
      let delivery = null;
      try {
        // Content scripts in an existing provider tab may outlive an extension
        // reload. Do not let one of those stale producers turn an unpaired
        // receiver into unauthenticated receiver traffic.
        if (sender.tab) await requirePairedTrustedReceiver();
        delivery = await retainCaptureForDelivery({ envelope, reason: message.reason, tab: sender.tab, signal: operation.controller.signal, operation });
        result = await completeForegroundDelivery(delivery, operation.controller.signal);
        if (Array.isArray(result?.accepted_identities)) {
          const key = sessionKey(summary.provider, summary.providerSessionId);
          const identities = Object.fromEntries(
            result.accepted_identities
              .filter((item) => item?.fidelity === "native" && typeof item?.message_ref === "string" && item.message_ref)
              .map((item) => [item.message_ref, item]),
          );
          await serializeStorageMutation(async () => {
            const current = await runtimeChrome.storage.local.get({ [ACCEPTED_MESSAGE_IDENTITIES_KEY]: {} });
            await runtimeChrome.storage.local.set({
              [ACCEPTED_MESSAGE_IDENTITIES_KEY]: { ...current[ACCEPTED_MESSAGE_IDENTITIES_KEY], [key]: identities },
            });
          });
        }
      } catch (error) {
        if (result) {
          await appendDebugLog({ stage: "capture_ack_identity_cache_pending", error: String(error.message || error) }).catch(() => undefined);
        } else {
          if (!delivery && error.captureDeliveryId) delivery = await captureStore.getDelivery(error.captureDeliveryId);
          if (delivery && !error.sharedDeliveryContinues && !foregroundDeliveryCompletions.get(delivery.id)?.participants.size) {
            await recordCaptureDeliveryFailure(delivery, error);
          }
          if (delivery && isRetryableCaptureError(error)) {
            await appendConversationTimeline({
              provider: summary.provider,
              providerSessionId: summary.providerSessionId,
              event: "held_with_reason",
              reason: message.reason || "content_script_capture",
              detail: "capture_queued_for_retry",
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
              queued: true,
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
            detail: typeof error.code === "string" ? error.code : "capture_rejected",
            tabId: sender.tab?.id || null,
          });
          throw error;
        }
      }
      if (result.outcome === "superseded") {
        await recordSupersededCapture(summary, result, message.reason || "content_script_capture");
        await setStateForTab(sender.tab?.id || null, {
          online: true, captured: false, last_capture: result,
          provider: summary.provider, provider_session_id: summary.providerSessionId,
          error: "receiver_superseded", last_receiver_request_id: result.receiver_request_id || null,
        }, sender.tab?.url || sender.tab?.pendingUrl || null);
        sendResponse({ ok: true, ...result, captured: false });
        return;
      }
      try {
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
      } catch (error) {
        await appendDebugLog({ stage: "capture_ack_telemetry_pending", error: String(error.message || error) }).catch(() => undefined);
      }
      // Receiver just proved reachable; flush anything queued from earlier
      // outages before returning this capture's result.
      void drainCaptureQueue("post_success");
      sendResponse({ ok: true, ...result });
      return;
      })();
      try { await operation.promise; } finally { captureDeliveries.delete(requestId); }
      return;
    }
    if (message.type === "polylogue.captureFreshnessHint") {
      const senderUrl = sender.tab?.url || sender.tab?.pendingUrl || "";
      const senderProvider = archiveProviderForUrl(senderUrl);
      const senderSessionId = conversationIdForUrl(senderUrl);
      const provider = message.provider || senderProvider;
      const nativeId = message.provider_session_id || senderSessionId;
      // TEMPORARY_CHAT_SENTINEL is a "this tab has a real, capturable
      // conversation" signal, not a real provider identity -- it cannot
      // equal chatgpt.js's freshness hints, which always carry the
      // conversation's true ephemeral id (nativeCaptureIdentity, read from
      // the intercepted native payload's own conversation_id/id). Enforcing
      // strict equality against the sentinel here rejected every freshness
      // hint a temporary chat ever sent after its first capture, silently
      // stopping later turns from ever being re-captured.
      if (
        sender.tab
        && (provider !== senderProvider
          || (senderSessionId && senderSessionId !== TEMPORARY_CHAT_SENTINEL && nativeId !== senderSessionId))
      ) {
        throw new Error("freshness_hint_sender_identity_mismatch");
      }
      if (provider !== "chatgpt" && sender.tab) {
        // The freshness queue only knows how to refetch ChatGPT. Other
        // providers recapture their own tab through captureTab, which applies
        // the same automatic-capture policy and receiver pairing checks.
        const capture = await captureTab(sender.tab, message.reason || "provider_page_hint");
        // captureTab returns null when it never reached the content script
        // (no session identity, paused policy, unpaired receiver); say so.
        sendResponse(capture ? { ok: true, scheduled: false, capture } : { ok: false, scheduled: false, error: "capture_not_started" });
        return;
      }
      sendResponse({
        ok: true,
        ...(await scheduleCaptureFreshness({
          provider,
          nativeId,
          reason: message.reason || "provider_page_hint",
          delayMs: Math.max(
            0,
            Math.min(5 * 60_000, Number.isFinite(Number(message.delay_ms)) ? Number(message.delay_ms) : 5_000),
          ),
          providerUpdatedAt: message.provider_updated_at || null,
          generationObservations: Array.isArray(message.generation_observations)
            ? message.generation_observations
            : [],
        })),
      });
      return;
    }
    if (message.type === "polylogue.getCaptureQueue") {
      sendResponse({ ok: true, ...await captureQueuePage({ cursor: message.cursor, pageSize: message.pageSize }) });
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
      const query = new globalThis.URLSearchParams({
        provider: message.provider,
        provider_session_id: message.provider_session_id
      });
      const state = await getJson(`/v1/archive-state?${query.toString()}`);
      const stored = await runtimeChrome.storage.local.get({ polylogueState: null });
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
        const storedBefore = await runtimeChrome.storage.local.get({ polylogueState: null });
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
      const stored = await runtimeChrome.storage.local.get({ polylogueState: null });
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
      const statusPayload = state.archive_state || state.status || state;
      sendResponse({
        ...statusPayload,
        receiver_request_id:
          statusPayload.receiver_request_id || state.last_receiver_request_id || health.receiver_request_id || null,
      });
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
      void reportCaptureHealth("capture_error", {
        provider,
        provider_session_id: providerSessionId,
        reason: message.error || "capture_page_failed",
        detail: { tab_id: tab?.id || null },
      });
      sendResponse({ ok: false });
      return;
    }
    if (message.type === "polylogue.captureHealth") {
      await reportCaptureHealth(message.event, message);
      sendResponse({ ok: true });
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
    if (message.type === "polylogue.browserActions.approval") {
      const result = await decideBrowserActionApproval(message.actionId, message.decision);
      if (message.decision === "approve") void pollBrowserActions();
      sendResponse({ ok: true, action: result.action });
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
      error: error.name === "AbortError" ? "capture_cancelled" : (typeof error.code === "string" ? error.code : (error.name === "QuotaExceededError" ? "capture_staging_quota_exceeded" : String(error.message || error))),
      outcome: error.outcome || null,
      retry_after_seconds: error.retryAfterSeconds ?? null,
      receiver_request_id: error.receiverRequestId || null
    });
  });
  return true;
});
}
