import {
  DEFAULT_BACKFILL_POLICY,
  PROVIDER_REQUEST_TIMEOUT_MS,
  backfillAlarmName,
  dayKey,
  fullJitterDelay,
  retryAfterMs,
  serializedContentHash,
  serializedJson,
  receiverAckContractError,
} from "./models.js";
import { jobFinished, progressBuckets } from "./storage.js";

function nowIso(nowMs) { return new Date(nowMs).toISOString(); }

function queueId(jobId, provider, nativeId) { return `${jobId}:${provider}:${nativeId}`; }

function bridgeOversizeError(error) {
  return String(error?.message || error).startsWith("backfill_bridge_source_response_too_large:")
    || String(error?.message || error).startsWith("backfill_bridge_projection_too_large:");
}

function providerContractDriftError(error) {
  return String(error?.message || error).startsWith("provider_contract_drift:");
}

function mergePolicy(patch = {}) {
  const policy = { ...DEFAULT_BACKFILL_POLICY, ...patch };
  if (policy.leaseMs <= PROVIDER_REQUEST_TIMEOUT_MS * 2) throw new Error("backfill_lease_must_exceed_request_timeout");
  return policy;
}

export class BackfillCoordinator {
  constructor({ store, adapters, receiver, receiverPreflight = null, checkpoint = null, alarms, clock = () => Date.now(), random = Math.random, instanceId = "extension-instance", receiverContractEpoch = instanceId }) {
    this.store = store;
    this.adapters = adapters;
    this.receiver = receiver;
    this.receiverPreflight = receiverPreflight;
    this.checkpoint = checkpoint;
    this.alarms = alarms;
    this.clock = clock;
    this.random = random;
    this.instanceId = instanceId;
    this.receiverContractEpoch = receiverContractEpoch;
    this.controlChains = new Map();
  }

  async start({ provider, cutoff, policy = {}, provider_options = {} }) {
    if (!this.adapters[provider]) throw new Error(`unsupported_backfill_provider:${provider}`);
    const now = this.clock();
    const resolvedPolicy = mergePolicy(policy);
    const id = `backfill-${provider}-${now}-${Math.floor(this.random() * 1e9).toString(36)}`;
    const job = {
      id,
      provider,
      cutoff,
      provider_options,
      status: "running",
      inventory_cursor: "0",
      inventory_complete: false,
      policy: resolvedPolicy,
      learned_cadence_ms: resolvedPolicy.baseCadenceMs,
      next_request_at_ms: now,
      cooldown_until_ms: null,
      cooldown_reason: null,
      throttle_count: 0,
      transport_failures: 0,
      daily_key: dayKey(now),
      daily_requests: 0,
      last_error: null,
      last_ack: null,
      created_at: nowIso(now),
      updated_at: nowIso(now),
      execution_generation: 0,
      execution_owner: null,
      execution_expires_at_ms: null,
    };
    await this.store.createJob(job);
    const checked = await this.ensureReceiverContract(job, now, true);
    if (checked.status === "running") await this.schedule(id, now);
    return this.status(id);
  }

  async control(jobId, action) {
    // A preflight is asynchronous. Serialize operator actions per job so a
    // delayed resume cannot overwrite a later cancel (or pause) after its
    // receiver check returns.
    const previous = this.controlChains.get(jobId) || Promise.resolve();
    const next = previous.catch(() => undefined).then(() => this.performControl(jobId, action));
    this.controlChains.set(jobId, next);
    void next.then(
      () => { if (this.controlChains.get(jobId) === next) this.controlChains.delete(jobId); },
      () => { if (this.controlChains.get(jobId) === next) this.controlChains.delete(jobId); },
    );
    return next;
  }

  async performControl(jobId, action) {
    const now = this.clock();
    const status = action === "start" || action === "resume" ? "running" : action === "pause" ? "paused" : action === "cancel" ? "cancelled" : null;
    if (!status) throw new Error(`unknown_backfill_action:${action}`);
    if (action === "resume" && (await this.store.listQueue(jobId)).some((item) => item.state === "recovery_required")) {
      throw new Error("browser_profile_recovery_required");
    }
    // Preflight while the job is still paused.  Marking it running first would
    // let an alarm acquire its next execution generation while an older worker
    // is still deciding whether this receiver is safe to use.
    const contractError = status === "running" ? await this.preflightReceiverContract() : null;
    if (contractError) {
      await this.store.controlJob(jobId, "paused", nowIso(now), this.receiverContractFailurePatch(contractError));
      return this.status(jobId);
    }
    const resumed = await this.store.controlJob(jobId, status, nowIso(now), status === "running" ? {
      cooldown_until_ms: null,
      cooldown_reason: null,
      throttle_count: 0,
      receiver_contract_epoch: this.receiverContractEpoch,
      receiver_contract_checked_at: nowIso(now),
      last_error: null,
    } : {}, action === "resume" ? now : null);
    if (status === "running") {
      await this.schedule(resumed.id, now);
    }
    return this.status(jobId);
  }

  async status(jobId) {
    const job = await this.requireJob(jobId);
    const queue = await this.store.listQueue(jobId);
    const checkpointError = await this.persistCheckpoint();
    return {
      ...job,
      progress: progressBuckets(queue),
      recovery_checkpoint_error: checkpointError,
    };
  }

  async listStatus() {
    return Promise.all((await this.store.listJobs()).map((job) => this.status(job.id)));
  }

  async exportLedger(jobId) {
    return { job: await this.requireJob(jobId), queue: await this.store.listQueue(jobId) };
  }

  async wake(jobId = null) {
    const jobs = jobId ? [await this.requireJob(jobId)] : await this.store.listJobs();
    for (const job of jobs.filter((candidate) => candidate.status === "running")) await this.runJob(job.id);
  }

  async runJob(jobId) {
    const now = this.clock();
    const current = await this.requireJob(jobId);
    const job = await this.store.acquireJobExecution(jobId, this.instanceId, now, current.policy.leaseMs);
    if (!job) return this.status(jobId);
    const generation = job.execution_generation;
    try {
      return await this.runLeasedJob(job, now);
    } catch (error) {
      if (String(error?.message || error).startsWith("stale_backfill_execution:")) return this.status(jobId);
      throw error;
    } finally {
      await this.store.releaseJobExecution(jobId, this.instanceId, generation);
    }
  }

  async runLeasedJob(initialJob, now) {
    let job = initialJob;
    const jobId = job.id;
    job = await this.ensureReceiverContract(job, now);
    if (job.status !== "running") return this.status(jobId);
    const persistedBridgeHold = (await this.store.listQueue(jobId)).find((item) => item.state === "bridge_oversize");
    if (persistedBridgeHold) {
      // A worker can die after durable queue persistence but before the job
      // pause. Repair that crash window before any alarm can fetch another
      // item or mark the all-terminal queue complete.
      return this.pauseJob(
        { ...job, last_error: persistedBridgeHold.last_error || "backfill_bridge_response_too_large" },
        "backfill_bridge_response_too_large",
        now,
      );
    }
    await this.store.recoverExpiredLeases(jobId, now);
    await this.schedule(jobId, now + job.policy.leaseMs);
    const receiverItem = await this.store.acquireNextLease(jobId, this.instanceId, now, job.policy.leaseMs, true);
    if (receiverItem) {
      job = await this.submitReceiver(job, receiverItem, receiverItem.envelope, now);
    }
    if (job.cooldown_until_ms && now < job.cooldown_until_ms) {
      await this.schedule(jobId, job.cooldown_until_ms);
      return this.status(jobId);
    }
    job = await this.resetDailyBudget(job, now);
    if (job.daily_requests >= job.policy.maxDailyRequests) {
      job = await this.pauseJob(job, "daily_request_budget_exhausted", now);
      return this.status(jobId);
    }
    if (job.next_request_at_ms > now) {
      await this.schedule(jobId, job.next_request_at_ms);
      return this.status(jobId);
    }
    if (!job.inventory_complete) {
      job = await this.enumerate(job, now);
      if (job.status === "running") await this.schedule(jobId, job.next_request_at_ms);
      return this.status(jobId);
    }
    let processed = 0;
    while (processed < job.policy.maxCapturesPerWake) {
      const currentNow = this.clock();
      job = await this.store.assertJobExecution(jobId, this.instanceId, job.execution_generation);
      if (job.next_request_at_ms > currentNow || job.daily_requests >= job.policy.maxDailyRequests) break;
      const adapter = this.adapters[job.provider];
      adapter.configure?.(job.provider_options || {});
      const requestCost = adapter.requestCost?.("fetch") || 1;
      if (job.daily_requests + requestCost > job.policy.maxDailyRequests) {
        job = await this.pauseJob(job, "daily_request_budget_exhausted", currentNow);
        break;
      }
      const item = await this.store.acquireNextLease(jobId, this.instanceId, currentNow, job.policy.leaseMs, false);
      if (!item) break;
      job = await this.processItem(job, item, currentNow, requestCost);
      processed += 1;
      if (job.status !== "running" || (job.cooldown_until_ms && currentNow < job.cooldown_until_ms)) break;
    }
    const queue = await this.store.listQueue(jobId);
    if (job.status === "running" && job.inventory_complete && jobFinished(queue)) {
      await this.saveJob({ ...job, status: "complete", updated_at: nowIso(this.clock()) });
    } else if (job.status === "running") {
      const receiverDue = Math.min(
        ...queue.filter((item) => item.state === "captured_waiting_receiver").map((item) => item.next_eligible_at_ms || this.clock()),
        Infinity,
      );
      const providerEligible = queue.filter((item) => ["eligible", "retry_wait", "leased"].includes(item.state));
      const providerDue = providerEligible.length
        ? Math.max(job.next_request_at_ms || 0, Math.min(...providerEligible.map((item) => item.next_eligible_at_ms || 0)))
        : Infinity;
      const next = Math.min(receiverDue, providerDue);
      await this.schedule(jobId, Number.isFinite(next) ? Math.max(this.clock() + 1000, next) : this.clock() + 60000);
    }
    return this.status(jobId);
  }

  async enumerate(job, now) {
    const queue = await this.store.listQueue(job.id);
    if (queue.length >= job.policy.maxQueueSize) return this.pauseJob(job, "queue_budget_exhausted", now);
    let result;
    const adapter = this.adapters[job.provider];
    adapter.configure?.(job.provider_options || {});
    const requestCost = adapter.requestCost?.("enumerate") || 1;
    const reserved = await this.reserveProviderRequests(job, now, requestCost);
    if (!reserved) return this.pauseJob(job, "daily_request_budget_exhausted", now);
    job = reserved;
    try {
      result = await adapter.enumerate(job.inventory_cursor, job.cutoff);
    } catch (error) {
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
      return this.handleJobTransport(job, error, now);
    }
    job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
    if (result.classification !== "success") return this.handleProviderBlock(job, result.response, result.classification, now);
    const room = job.policy.maxQueueSize - queue.length;
    if (result.items.length > room) return this.pauseJob(job, "queue_budget_exhausted", now);
    for (const item of result.items) {
      const revision = item.updated_at ? await this.store.getRevision(job.provider, item.native_id) : null;
      const unchanged = revision?.provider_updated_at === item.updated_at;
      await this.store.upsertDiscoveredCas(job.id, this.instanceId, job.execution_generation, {
        id: queueId(job.id, job.provider, item.native_id),
        job_id: job.id,
        provider: job.provider,
        native_id: item.native_id,
        title: item.title || null,
        provider_updated_at: item.updated_at || null,
        state: unchanged ? "unchanged" : "eligible",
        attempt_count: 0,
        next_eligible_at_ms: now,
        lease_owner: null,
        lease_expires_at_ms: null,
        last_response_class: unchanged ? "known_revision" : "discovered",
        capture_fidelity: null,
        receiver_receipt: null,
        content_hash: null,
      });
    }
    const next = {
      ...job,
      provider_options: { ...job.provider_options, ...(result.provider_options || {}) },
      inventory_cursor: result.next_cursor,
      inventory_complete: Boolean(result.done),
      updated_at: nowIso(now),
    };
    await this.saveJob(next);
    return next;
  }

  async processItem(job, item, now, requestCost = 1) {
    if (item.resume_state === "captured_waiting_receiver" && item.envelope) return this.submitReceiver(job, item, item.envelope, now);
    const reserved = await this.reserveProviderRequests(job, now, requestCost);
    if (!reserved) return this.pauseJob(job, "daily_request_budget_exhausted", now);
    job = reserved;
    let response;
    try {
      response = await this.adapters[job.provider].fetchNative(item.native_id);
    } catch (error) {
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
      if (bridgeOversizeError(error)) return this.holdBridgeOversize(job, item, error, now);
      if (providerContractDriftError(error)) {
        await this.saveQueue(job, {
          ...item,
          state: "failed",
          lease_owner: null,
          lease_expires_at_ms: null,
          last_response_class: "contract_drift",
          last_error: String(error?.message || error),
        });
        return job;
      }
      return this.retryTransport(job, item, error, now);
    }
    job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
    const classification = this.adapters[job.provider].classifyResponse(response);
    if (classification !== "success") {
      if (classification === "rate_limited" || classification === "auth_or_challenge") {
        await this.saveQueue(job, { ...item, state: classification === "auth_or_challenge" ? "auth_required" : "retry_wait", lease_owner: null, lease_expires_at_ms: null, last_response_class: classification, next_eligible_at_ms: now });
        return this.handleProviderBlock(job, response, classification, now);
      }
      if (classification === "transport") return this.retryTransport(job, item, new Error(`provider_http_${response.status}`), now);
      await this.saveQueue(job, { ...item, state: "failed", lease_owner: null, lease_expires_at_ms: null, last_response_class: classification, last_error: `provider_http_${response.status}` });
      return job;
    }
    let capture;
    try {
      capture = await this.adapters[job.provider].normalizeCapture(response, item, { job_id: job.id, queue_id: item.id, instance_id: this.instanceId });
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
    } catch (error) {
      await this.saveQueue(job, { ...item, state: "failed", lease_owner: null, lease_expires_at_ms: null, last_response_class: "contract_drift", last_error: String(error.message || error) });
      return job;
    }
    const captureFidelity = capture.provider_meta?.capture_fidelity || "native_full";
    if (!capture.session?.turns?.length) {
      await this.saveQueue(job, { ...item, state: "no_turns", lease_owner: null, lease_expires_at_ms: null, last_response_class: "native_empty", capture_fidelity: captureFidelity });
      return job;
    }
    return this.submitReceiver(job, item, capture, now);
  }

  async submitReceiver(job, item, capture, now) {
    const captureFidelity = capture.provider_meta?.capture_fidelity || "native_full";
    const serialized = serializedJson(capture);
    const hash = await serializedContentHash(serialized);
    const firstPersistence = item.resume_state !== "captured_waiting_receiver" || !item.envelope;
    if (firstPersistence) {
      const projectedBytes = await this.store.storedBytes(job.id) + new TextEncoder().encode(serialized).length;
      if (projectedBytes > job.policy.maxStoredBytes) {
        await this.saveQueue(job, { ...item, state: "failed", envelope: null, lease_owner: null, lease_expires_at_ms: null, last_response_class: "storage_budget_exhausted", last_error: "storage_budget_exhausted" });
        return this.pauseJob(job, "storage_budget_exhausted", now);
      }
      try {
        await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash, capture_fidelity: capture.provider_meta?.capture_fidelity || "native_full", lease_owner: this.instanceId, lease_expires_at_ms: now + job.policy.leaseMs, last_response_class: "captured" });
      } catch (error) {
        if (error?.name !== "QuotaExceededError") throw error;
        return this.pauseJob({ ...job, last_error: "indexeddb_quota_exceeded" }, "indexeddb_quota_exceeded", now);
      }
    }
    try {
      const receipt = await this.receiver(capture, serialized);
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
      const contractError = receiverAckContractError(receipt, hash);
      if (contractError) throw contractError;
      const completeItem = { ...item, state: "complete", envelope: null, content_hash: hash, capture_fidelity: captureFidelity, receiver_receipt: receipt, lease_owner: null, lease_expires_at_ms: null, last_response_class: "receiver_acked", completed_at: nowIso(now) };
      const revision = item.provider_updated_at
        ? {
          id: `${item.provider}:${item.native_id}`,
          provider: item.provider,
          native_id: item.native_id,
          provider_updated_at: item.provider_updated_at,
          receiver_content_hash: hash,
          receiver_request_id: receipt.receiver_request_id,
          completed_at: nowIso(now),
        }
        : null;
      const lastAck = { receiver_request_id: receipt.receiver_request_id, content_hash: hash, at: nowIso(now) };
      const next = await this.store.finalizeCaptureCas(
        job,
        this.instanceId,
        job.execution_generation,
        completeItem,
        revision,
        lastAck,
      );
      return next;
    } catch (error) {
      if (String(error?.message || error).startsWith("stale_backfill_execution:")) throw error;
      if (error?.code === "receiver_contract_incompatible" || String(error?.message || error).startsWith("receiver_contract_incompatible:")) {
        await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash, capture_fidelity: captureFidelity, lease_owner: null, lease_expires_at_ms: null, last_response_class: "receiver_contract_incompatible", last_error: String(error.message || error) });
        return this.pauseJob({ ...job, last_error: String(error.message || error) }, "receiver_contract_incompatible", now);
      }
      const attempt = (item.attempt_count || 0) + 1;
      await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash, capture_fidelity: captureFidelity, attempt_count: attempt, next_eligible_at_ms: now + fullJitterDelay(attempt, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random), lease_owner: null, lease_expires_at_ms: null, last_response_class: "receiver_down", last_error: String(error.message || error) });
      const exhausted = attempt >= job.policy.maxReceiverAttempts;
      const next = {
        ...job,
        status: exhausted ? "paused" : job.status,
        cooldown_reason: exhausted ? "receiver_retry_budget_exhausted" : job.cooldown_reason,
        last_error: String(error.message || error),
        updated_at: nowIso(now),
      };
      await this.saveJob(next);
      return next;
    }
  }

  async retryTransport(job, item, error, now) {
    const attempt = (item.attempt_count || 0) + 1;
    const terminal = attempt >= job.policy.maxTransportAttempts;
    await this.saveQueue(job, { ...item, state: terminal ? "failed" : "retry_wait", attempt_count: attempt, next_eligible_at_ms: now + fullJitterDelay(attempt, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random), lease_owner: null, lease_expires_at_ms: null, last_response_class: "transport", last_error: String(error.message || error) });
    const failures = (job.transport_failures || 0) + 1;
    const next = { ...job, transport_failures: failures, last_error: String(error.message || error) };
    if (failures >= job.policy.breakerThreshold) return this.pauseJob(next, "repeated_transport_failures", now);
    await this.saveJob(next);
    return next;
  }

  async holdBridgeOversize(job, item, error, now) {
    const message = String(error?.message || error);
    await this.saveQueue(job, {
      ...item,
      state: "bridge_oversize",
      attempt_count: item.attempt_count || 0,
      next_eligible_at_ms: null,
      lease_owner: null,
      lease_expires_at_ms: null,
      last_response_class: "bridge_response_too_large",
      last_error: message,
    });
    return this.pauseJob({ ...job, last_error: message }, "backfill_bridge_response_too_large", now);
  }

  async handleProviderBlock(job, response, classification, now) {
    if (classification === "auth_or_challenge") {
      const reason = response?.polylogueAuthReason || "provider_auth_or_challenge";
      return this.pauseJob({ ...job, last_error: reason }, reason, now);
    }
    if (classification !== "rate_limited") return this.handleJobTransport(job, new Error(`provider_${classification}`), now);
    const count = (job.throttle_count || 0) + 1;
    const learned = Math.min(job.policy.maxCadenceMs, Math.max(job.learned_cadence_ms * 2, job.policy.baseCadenceMs));
    const delay = Math.max(retryAfterMs(response?.headers, now) || 0, fullJitterDelay(count, learned, job.policy.maxCadenceMs, this.random));
    const next = { ...job, throttle_count: count, learned_cadence_ms: learned, cooldown_until_ms: now + delay, cooldown_reason: "provider_rate_limited", last_error: `provider_http_${response?.status || 429}`, updated_at: nowIso(now) };
    if (count >= job.policy.breakerThreshold) next.status = "paused";
    await this.saveJob(next);
    await this.schedule(job.id, next.cooldown_until_ms);
    return next;
  }

  async handleJobTransport(job, error, now) {
    if (bridgeOversizeError(error)) {
      return this.pauseJob(
        { ...job, last_error: String(error?.message || error) },
        "backfill_bridge_response_too_large",
        now,
      );
    }
    const failures = (job.transport_failures || 0) + 1;
    const next = { ...job, transport_failures: failures, last_error: String(error.message || error) };
    if (failures >= job.policy.breakerThreshold) return this.pauseJob(next, "repeated_transport_failures", now);
    const deadline = now + fullJitterDelay(failures, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random);
    await this.saveJob({ ...next, cooldown_until_ms: deadline, cooldown_reason: "transport_backoff", updated_at: nowIso(now) });
    await this.schedule(job.id, deadline);
    return { ...next, cooldown_until_ms: deadline, cooldown_reason: "transport_backoff" };
  }

  async reserveProviderRequests(job, now, count) {
    const currentDay = dayKey(now);
    const jitter = Math.floor(this.random() * Math.max(1, job.learned_cadence_ms / 4));
    return this.store.reserveProviderRequests(
      job.id,
      this.instanceId,
      job.execution_generation,
      count,
      currentDay,
      now + job.learned_cadence_ms + jitter,
    );
  }

  async resetDailyBudget(job, now) {
    const key = dayKey(now);
    if (job.daily_key === key) return job;
    const next = { ...job, daily_key: key, daily_requests: 0 };
    await this.saveJob(next);
    return next;
  }

  async pauseJob(job, reason, now) {
    const next = { ...job, status: "paused", cooldown_reason: reason, last_error: job.last_error || reason, updated_at: nowIso(now) };
    await this.saveJob(next);
    return next;
  }

  async ensureReceiverContract(job, now, force = false) {
    if (!this.receiverPreflight) return job;
    if (!force && job.receiver_contract_epoch === this.receiverContractEpoch) return job;
    const contractError = await this.preflightReceiverContract();
    if (contractError) {
      const next = { ...job, ...this.receiverContractFailurePatch(contractError), updated_at: nowIso(now) };
      if (job.execution_owner === this.instanceId) await this.saveJob(next);
      else await this.store.putJob(next);
      return next;
    }
    const next = { ...job, receiver_contract_epoch: this.receiverContractEpoch, receiver_contract_checked_at: nowIso(now), last_error: null };
    if (job.execution_owner === this.instanceId) await this.saveJob(next);
    else await this.store.putJob(next);
    return next;
  }

  async preflightReceiverContract() {
    if (!this.receiverPreflight) return null;
    try {
      await this.receiverPreflight();
      return null;
    } catch (error) {
      return String(error?.message || error);
    }
  }

  receiverContractFailurePatch(message) {
    return {
      status: "paused",
      cooldown_reason: message.startsWith("receiver_contract_incompatible:")
        ? "receiver_contract_incompatible"
        : "receiver_preflight_unavailable",
      last_error: message,
    };
  }

  async persistCheckpoint() {
    if (!this.checkpoint) return null;
    try {
      await this.checkpoint(await this.store.exportRecoveryCheckpoint());
      return null;
    } catch (error) {
      // IndexedDB remains authoritative. A best-effort profile-recovery copy
      // must never turn an otherwise durable backfill action into a failure.
      return String(error?.message || error);
    }
  }

  async saveJob(job) {
    return this.store.putJobCas(job, this.instanceId, job.execution_generation);
  }

  async saveQueue(job, item) {
    return this.store.putQueueCas(job.id, this.instanceId, job.execution_generation, item);
  }

  async requireJob(jobId) {
    const job = await this.store.getJob(jobId);
    if (!job) throw new Error(`backfill_job_not_found:${jobId}`);
    return job;
  }

  async schedule(jobId, whenMs) {
    if (!this.alarms?.create) return;
    await this.alarms.create(backfillAlarmName(jobId), { when: Math.max(this.clock() + 1000, whenMs) });
  }
}
