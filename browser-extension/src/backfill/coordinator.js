import {
  DEFAULT_BACKFILL_POLICY,
  backfillAlarmName,
  dayKey,
  fullJitterDelay,
  retryAfterMs,
  receiverAckContractError,
} from "./models.js";

function nowIso(nowMs) { return new Date(nowMs).toISOString(); }

function queueId(jobId, provider, nativeId) { return `${jobId}:${provider}:${nativeId}`; }

function providerContractDriftError(error) {
  return String(error?.message || error).startsWith("provider_contract_drift:");
}

function mergePolicy(patch = {}) {
  const policy = { ...DEFAULT_BACKFILL_POLICY, ...patch };
  if (!Number.isFinite(policy.leaseMs) || policy.leaseMs <= 0) throw new Error("backfill_lease_invalid");
  return policy;
}

export class BackfillCoordinator {
  constructor({ store, adapters, receiver, receiverPreflight = null, checkpoint = null, prepareCapture, receiverAcked = null, alarms, clock = () => Date.now(), random = Math.random, instanceId = "extension-instance", receiverContractEpoch = instanceId }) {
    if (typeof prepareCapture !== "function") throw new Error("capture_body_owner_required");
    this.store = store;
    this.adapters = adapters;
    this.receiver = receiver;
    this.receiverPreflight = receiverPreflight;
    this.checkpoint = checkpoint;
    this.prepareCapture = prepareCapture;
    this.receiverAcked = receiverAcked;
    this.alarms = alarms;
    this.clock = clock;
    this.random = random;
    this.instanceId = instanceId;
    this.receiverContractEpoch = receiverContractEpoch;
    this.controlChains = new Map();
    this.executions = new Map();
    this.checkpointPromise = null;
  }

  async start({ provider, cutoff, policy = {}, provider_options = {}, account_scope = null }) {
    if (!this.adapters[provider]) throw new Error(`unsupported_backfill_provider:${provider}`);
    const now = this.clock();
    const resolvedPolicy = mergePolicy(policy);
    const id = `backfill-${provider}-${now}-${Math.floor(this.random() * 1e9).toString(36)}`;
    const job = {
      id,
      provider,
      account_scope,
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
    this.checkpointDirty = true;
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
    if (action === "resume" && (await this.store.queueSummary(jobId)).recoveryRequired) {
      throw new Error("browser_profile_recovery_required");
    }
    if (status === "running") {
      await this.persistCheckpoint();
      const authorityError = await this.checkpointErrorForJob(jobId);
      if (authorityError) return this.snapshotStatus(jobId, authorityError);
    }
    // Preflight while the job is still paused.  Marking it running first would
    // let an alarm acquire its next execution generation while an older worker
    // is still deciding whether this receiver is safe to use.
    const contractError = status === "running" ? await this.preflightReceiverContract() : null;
    if (contractError) {
      await this.store.controlJob(jobId, "paused", nowIso(now), this.receiverContractFailurePatch(contractError));
      this.checkpointDirty = true;
      return this.status(jobId);
    }
    if (status !== "running") this.executions.get(jobId)?.controller.abort("backfill_controlled");
    const resumed = await this.store.controlJob(jobId, status, nowIso(now), status === "running" ? {
      cooldown_until_ms: null,
      cooldown_reason: null,
      throttle_count: 0,
      receiver_contract_epoch: this.receiverContractEpoch,
      receiver_contract_checked_at: nowIso(now),
      last_error: null,
    } : {}, action === "resume" ? now : null);
    this.checkpointDirty = true;
    if (status === "running") {
      await this.schedule(resumed.id, now);
    } else {
      await this.executions.get(jobId)?.promise?.catch(() => undefined);
    }
    return this.status(jobId);
  }

  async status(jobId) {
    await this.persistCheckpoint();
    return this.snapshotStatus(jobId);
  }

  async snapshotStatus(jobId, checkpointError = undefined) {
    const job = await this.requireJob(jobId);
    if (checkpointError === undefined) checkpointError = job.recovery_checkpoint_outcome?.error || null;
    const summary = await this.store.queueSummary(jobId);
    return {
      ...job,
      progress: summary.progress,
      recovery_checkpoint_error: checkpointError,
      export_pending: Boolean(await this.store.pendingRecoverySnapshot?.(jobId, "checkpoint-export")),
    };
  }

  async listStatus(options = {}) {
    await this.persistCheckpoint();
    const page = await this.store.jobPage(options);
    const jobs = [];
    for (const job of page.jobs) jobs.push(await this.snapshotStatus(job.id));
    return { ...page, jobs };
  }

  async wake(jobId = null) {
    if (jobId) {
      const job = await this.requireJob(jobId);
      if (job.status === "running") await this.runJob(job.id);
      return;
    }
    for await (const job of this.store.jobRecords()) if (job.status === "running") await this.runJob(job.id);
  }

  async runJob(jobId) {
    const now = this.clock();
    await this.persistCheckpoint();
    const authorityError = await this.checkpointErrorForJob(jobId);
    if (authorityError) return this.snapshotStatus(jobId, authorityError);
    const current = await this.requireJob(jobId);
    const job = await this.store.acquireJobExecution(jobId, this.instanceId, now, current.policy.leaseMs);
    if (!job) return this.status(jobId);
    const generation = job.execution_generation;
    const controller = new AbortController();
    let renewal = Promise.resolve();
    const timer = globalThis.setInterval(() => {
      renewal = renewal.then(() => this.store.renewJobExecution(jobId, this.instanceId, generation, this.clock(), job.policy.leaseMs))
        .catch((error) => controller.abort(error));
    }, Math.max(1, Math.floor(job.policy.leaseMs / 3)));
    const execution = { controller, promise: null };
    this.executions.set(jobId, execution);
    try {
      execution.promise = this.runLeasedJob(job, now);
      return await execution.promise;
    } catch (error) {
      if (String(error?.message || error).startsWith("stale_backfill_execution:")) return this.status(jobId);
      throw error;
    } finally {
      globalThis.clearInterval(timer); await renewal;
      this.executions.delete(jobId);
      await this.store.releaseJobExecution(jobId, this.instanceId, generation);
    }
  }

  executionSignal(job) { return this.executions.get(job.id)?.controller.signal; }

  async runLeasedJob(initialJob, now) {
    let job = initialJob;
    const jobId = job.id;
    job = await this.ensureReceiverContract(job, now);
    if (job.status !== "running") return this.status(jobId);
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
      const item = await this.store.acquireNextLease(jobId, this.instanceId, currentNow, job.policy.leaseMs, false);
      if (!item) break;
      const requestCost = adapter.requestCost?.("fetch", item) ?? 1;
      if (job.daily_requests + requestCost > job.policy.maxDailyRequests) {
        await this.saveQueue(job, { ...item, state: item.resume_state || "eligible", lease_owner: null, lease_expires_at_ms: null });
        job = await this.pauseJob(job, "daily_request_budget_exhausted", currentNow);
        break;
      }
      job = await this.processItem(job, item, currentNow, requestCost);
      processed += 1;
      if (job.status !== "running" || (job.cooldown_until_ms && currentNow < job.cooldown_until_ms)) break;
    }
    const summary = await this.store.queueSummary(jobId, this.clock());
    if (job.status === "running" && job.inventory_complete && summary.finished) {
      await this.saveJob({ ...job, status: "complete", updated_at: nowIso(this.clock()) });
    } else if (job.status === "running") {
      const receiverDue = summary.receiverDue;
      const providerDue = Number.isFinite(summary.providerDue)
        ? Math.max(job.next_request_at_ms || 0, summary.providerDue) : Infinity;
      const next = Math.min(receiverDue, providerDue);
      await this.schedule(jobId, Number.isFinite(next) ? Math.max(this.clock() + 1000, next) : this.clock() + 60000);
    }
    return this.status(jobId);
  }

  async enumerate(job, now) {
    let result;
    const adapter = this.adapters[job.provider];
    adapter.configure?.(job.provider_options || {});
    const requestCost = adapter.requestCost?.("enumerate") || 1;
    const reserved = await this.reserveProviderRequests(job, now, requestCost);
    if (!reserved) return this.pauseJob(job, "daily_request_budget_exhausted", now);
    if (reserved.status !== "running") return reserved;
    job = reserved;
    try {
      result = await adapter.enumerate(job.inventory_cursor, job.cutoff, this.executionSignal(job));
    } catch (error) {
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
      // A shared provider cooldown refuses the request before it is sent: that
      // is the provider's rate limit, not a transport failure.
      if (sharedRateLimitError(error)) return this.handleProviderBlock(job, sharedRateLimitResponse(error), "rate_limited", now);
      return this.handleJobTransport(job, error, now);
    }
    job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
    if (result.classification !== "success") return this.handleProviderBlock(job, result.response, result.classification, now);
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
      this.checkpointDirty = true;
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
    if (item.envelope && (item.resume_state === "captured_waiting_receiver" || item.capture_record_ref)) return this.submitReceiver(job, item, item.envelope, now);
    const reserved = await this.reserveProviderRequests(job, now, requestCost);
    if (!reserved) return this.pauseJob(job, "daily_request_budget_exhausted", now);
    if (reserved.status !== "running") return reserved;
    job = reserved;
    let response;
    try {
      response = await this.adapters[job.provider].fetchNative(item.native_id, this.executionSignal(job), { item, jobId: job.id, owner: this.instanceId, generation: job.execution_generation });
    } catch (error) {
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
      Object.assign(item, await this.store.getQueue(item.id));
      if (["native_bundle_recovery_missing", "native_bundle_identity_conflict", "native_bundle_owner_mismatch", "native_bundle_reply_invalid",
        "native_acquisition_recovery_pending", "native_acquisition_identity_conflict", "capture_job_account_scope_unresolved", "capture_job_account_scope_mismatch", "capture_staging_missing"].includes(error?.code || error?.message)) {
        await this.saveQueue(job, { ...item, state: "recovery_required", lease_owner: null, lease_expires_at_ms: null,
          last_response_class: "browser_profile_recovery_required", last_error: String(typeof error.code === "string" ? error.code : (error.message || error)) });
        return this.pauseJob(job, "browser_profile_recovery_required", now);
      }
      if (sharedRateLimitError(error)) {
        await this.saveQueue(job, { ...item, state: "retry_wait", lease_owner: null, lease_expires_at_ms: null, last_response_class: "rate_limited", next_eligible_at_ms: now });
        return this.handleProviderBlock(job, sharedRateLimitResponse(error), "rate_limited", now);
      }
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
    Object.assign(item, await this.store.getQueue(item.id));
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
      const attribution = { job_id: job.id, queue_id: item.id, instance_id: this.instanceId };
      capture = await this.adapters[job.provider].normalizeCapture(response, item, attribution, this.executionSignal(job), { itemId: item.id, jobId: job.id, owner: this.instanceId, generation: job.execution_generation });
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
    } catch (error) {
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
      Object.assign(item, await this.store.getQueue(item.id));
      if (providerContractDriftError(error)) {
        await this.saveQueue(job, { ...item, state: "failed", lease_owner: null, lease_expires_at_ms: null, last_response_class: "contract_drift", last_error: String(error.message || error) });
        return job;
      }
      return this.retryTransport(job, item, error, now);
    }
    const captureFidelity = capture.provider_meta?.capture_fidelity || "native_full";
    if (!(capture.capture_summary?.turnCount ?? capture.session?.turns?.length) && capture.provider_meta?.capture_fidelity !== "native_full") {
      await this.saveQueue(job, { ...item, state: "no_turns", lease_owner: null, lease_expires_at_ms: null, last_response_class: "native_empty", capture_fidelity: captureFidelity });
      return job;
    }
    return this.submitReceiver(job, item, capture, now);
  }

  async submitReceiver(job, item, capture, now) {
    const captureFidelity = capture.provider_meta?.capture_fidelity || "native_full";
    let serialized = null; let hash = item.content_hash || null; let acknowledged = false;
    const firstPersistence = item.resume_state !== "captured_waiting_receiver" || !item.envelope;
    if (firstPersistence) {
      try {
        // Retain acquired records before serializing the receiver body. Disk
        // quota or worker loss here must not require another provider read.
        await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash,
          capture_fidelity: captureFidelity, lease_owner: this.instanceId, lease_expires_at_ms: this.clock() + job.policy.leaseMs, last_response_class: "captured" });
      } catch (error) {
        if (error?.name !== "QuotaExceededError") throw error;
        return this.pauseJob({ ...job, last_error: "indexeddb_quota_exceeded" }, "indexeddb_quota_exceeded", now);
      }
    }
    try {
      const prepared = await this.prepareCapture(capture, item, this.executionSignal(job));
      serialized = prepared.body;
      hash = prepared.contentHash;
      await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash,
        capture_fidelity: captureFidelity, lease_owner: this.instanceId, lease_expires_at_ms: this.clock() + job.policy.leaseMs, last_response_class: "prepared" });
      const receipt = await this.receiver(capture, serialized, this.executionSignal(job));
      const contractError = receiverAckContractError(receipt, hash);
      if (contractError) throw contractError;
      acknowledged = true;
      job = await this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
      const superseded = receipt.outcome === "superseded";
      const completeItem = { ...item, state: superseded ? "superseded" : "complete", envelope: null, body_ref: null, raw_acquisition_ref: null, capture_record_ref: null, capture_source_refs: [], capture_bundle_ref: null, capture_bundle_replies: null, record_ref: null, source_refs: [], content_hash: superseded ? null : receipt.content_hash, capture_fidelity: captureFidelity, receiver_receipt: receipt, lease_owner: null, lease_expires_at_ms: null, last_response_class: superseded ? "receiver_superseded" : "receiver_acked", completed_at: nowIso(now) };
      const revision = !superseded && item.provider_updated_at
        ? {
          id: `${item.provider}:${item.native_id}`,
          provider: item.provider,
          native_id: item.native_id,
          provider_updated_at: item.provider_updated_at,
          receiver_content_hash: receipt.content_hash,
          receiver_request_id: receipt.receiver_request_id,
          completed_at: nowIso(now),
        }
        : null;
      const lastAck = { receiver_request_id: receipt.receiver_request_id, content_hash: receipt.content_hash, outcome: receipt.outcome, at: nowIso(now) };
      const next = await this.store.finalizeCaptureCas(
        job,
        this.instanceId,
        job.execution_generation,
        completeItem,
        revision,
        lastAck,
      );
      this.checkpointDirty = true;
      if (this.receiverAcked) {
        try { await this.receiverAcked(capture); }
        catch (error) {
          await this.saveJob({ ...next, last_error: `capture_staging_cleanup_pending:${String(error.message || error)}` });
        }
      }
      return next;
    } catch (error) {
      if (String(error?.message || error).startsWith("stale_backfill_execution:")) throw error;
      if (acknowledged) {
        const current = await this.store.getQueue(item.id);
        if (["complete", "superseded"].includes(current?.state)) return this.store.getJob(job.id);
        return this.pauseJob({ ...job, last_error: `capture_ack_finalization_pending:${String(error.message || error)}` }, "capture_ack_finalization_pending", now);
      }
      if (error?.name === "QuotaExceededError") {
        await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash,
          capture_fidelity: captureFidelity, lease_owner: null, lease_expires_at_ms: null, last_response_class: "storage_quota", last_error: "capture_storage_quota_exceeded" });
        return this.pauseJob({ ...job, last_error: "capture_storage_quota_exceeded" }, "capture_storage_quota_exceeded", now);
      }
      if (error?.code === "receiver_contract_incompatible" || String(error?.message || error).startsWith("receiver_contract_incompatible:")) {
        await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash, capture_fidelity: captureFidelity, lease_owner: null, lease_expires_at_ms: null, last_response_class: "receiver_contract_incompatible", last_error: String(error.message || error) });
        return this.pauseJob({ ...job, last_error: String(error.message || error) }, "receiver_contract_incompatible", now);
      }
      const attempt = (item.attempt_count || 0) + 1;
      await this.saveQueue(job, { ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash, capture_fidelity: captureFidelity, attempt_count: attempt, next_eligible_at_ms: now + fullJitterDelay(attempt, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random), lease_owner: null, lease_expires_at_ms: null, last_response_class: "receiver_down", last_error: String(error.message || error) });
      const next = {
        ...job,
        status: job.status,
        cooldown_reason: job.cooldown_reason,
        last_error: String(error.message || error),
        updated_at: nowIso(now),
      };
      await this.saveJob(next);
      return next;
    }
  }

  async retryTransport(job, item, error, now) {
    const attempt = (item.attempt_count || 0) + 1;
    await this.saveQueue(job, { ...item, state: "retry_wait", attempt_count: attempt, next_eligible_at_ms: now + fullJitterDelay(attempt, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random), lease_owner: null, lease_expires_at_ms: null, last_response_class: "transport", last_error: String(error.message || error) });
    const failures = (job.transport_failures || 0) + 1;
    const next = { ...job, transport_failures: failures, last_error: String(error.message || error) };
    await this.saveJob(next);
    return next;
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
    await this.saveJob(next);
    await this.schedule(job.id, next.cooldown_until_ms);
    return next;
  }

  async handleJobTransport(job, error, now) {
    const failures = (job.transport_failures || 0) + 1;
    const next = { ...job, transport_failures: failures, last_error: String(error.message || error) };
    const deadline = now + fullJitterDelay(failures, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random);
    await this.saveJob({ ...next, cooldown_until_ms: deadline, cooldown_reason: "transport_backoff", updated_at: nowIso(now) });
    await this.schedule(job.id, deadline);
    return { ...next, cooldown_until_ms: deadline, cooldown_reason: "transport_backoff" };
  }

  async reserveProviderRequests(job, now, count) {
    if (count === 0) return this.store.assertJobExecution(job.id, this.instanceId, job.execution_generation);
    const currentDay = dayKey(now);
    const jitter = Math.floor(this.random() * Math.max(1, job.learned_cadence_ms / 4));
    const reserved = await this.store.reserveProviderRequests(
      job.id,
      this.instanceId,
      job.execution_generation,
      count,
      currentDay,
      now + job.learned_cadence_ms + jitter,
    );
    if (!reserved) return null;
    this.checkpointDirty = true;
    await this.persistCheckpoint();
    const authorityError = await this.checkpointErrorForJob(job.id);
    return authorityError ? this.requireJob(job.id) : reserved;
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
      else { await this.store.putJob(next); this.checkpointDirty = true; }
      return next;
    }
    const next = { ...job, receiver_contract_epoch: this.receiverContractEpoch, receiver_contract_checked_at: nowIso(now), last_error: null };
    if (job.execution_owner === this.instanceId) await this.saveJob(next);
    else { await this.store.putJob(next); this.checkpointDirty = true; }
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
    if (this.checkpointPromise) return this.checkpointPromise;
    this.checkpointDirty = true;
    const pending = this.commitCheckpointUntilStable();
    this.checkpointPromise = pending;
    try {
      return await pending;
    } finally {
      if (this.checkpointPromise === pending) this.checkpointPromise = null;
    }
  }

  async commitCheckpointUntilStable() {
    let error;
    do {
      this.checkpointDirty = false;
      error = await this.commitCheckpoint();
    } while (this.checkpointDirty);
    return error;
  }

  async commitCheckpoint() {
    const attemptId = crypto.randomUUID();
    try {
      const results = await this.checkpoint();
      for await (const result of results) {
        if (typeof result?.job_id !== "string") throw new Error("capture_job_checkpoint_result_invalid");
        const detail = result.error === null ? null : String(result.error || "capture_job_receiver_commit_failed");
        await this.store.putCheckpointOutcome(result.job_id, { attempt_id: attemptId,
          state: detail === null ? "committed" : "failed", error: detail, outcome: result.outcome || (detail ? "unavailable" : "committed") });
        if (detail === null) continue;
        const job = await this.store.getJob(result.job_id);
        if (job?.status !== "running") continue;
        const nowMs = this.clock(); const now = nowIso(nowMs);
        const retryUntil = Number.isFinite(result.retry_until_ms) ? result.retry_until_ms
          : Number.isFinite(result.retry_after_ms) ? nowMs + Math.max(0, result.retry_after_ms) : null;
        if (result.outcome === "rate_limited" && retryUntil !== null) {
          if (job.cooldown_reason === "provider_rate_limited" && job.cooldown_until_ms >= retryUntil && job.cooldown_until_ms > nowMs) continue;
          await this.store.controlJob(job.id, "running", now, {
            cooldown_reason: "provider_rate_limited", cooldown_until_ms: retryUntil, last_error: detail,
          });
          this.checkpointDirty = true;
          await this.schedule(job.id, retryUntil);
        } else {
          await this.store.controlJob(job.id, "paused", now, {
            cooldown_reason: "receiver_capture_job_authority_unavailable", last_error: detail,
          });
          this.checkpointDirty = true;
        }
      }
    } catch (error) {
      const detail = String(error?.message || error); const now = nowIso(this.clock());
      for await (const job of this.store.jobRecords()) {
        // A transport/iterator fault after a processed prefix must mark every
        // remaining row, without replacing that prefix's actual outcomes.
        if (job.recovery_checkpoint_outcome?.attempt_id === attemptId) continue;
        await this.store.putCheckpointOutcome(job.id, { attempt_id: attemptId,
          state: "failed", error: detail, outcome: "unavailable" });
        if (job.status !== "running") continue;
        await this.store.controlJob(job.id, "paused", now, {
          cooldown_reason: "receiver_capture_job_authority_unavailable", last_error: detail,
        });
        this.checkpointDirty = true;
      }
    }
  }

  async checkpointErrorForJob(jobId) {
    return (await this.requireJob(jobId)).recovery_checkpoint_outcome?.error || null;
  }

  async saveJob(job) {
    const saved = await this.store.putJobCas(job, this.instanceId, job.execution_generation);
    this.checkpointDirty = true;
    return saved;
  }

  async saveQueue(job, item) {
    const saved = await this.store.putQueueCas(job.id, this.instanceId, job.execution_generation, item);
    this.checkpointDirty = true;
    return saved;
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

// The runtime refuses a provider request while a shared cooldown is active and
// throws a `rate_limited` error without contacting the provider. Treat it as
// the rate limit it is, carrying the remaining cooldown as Retry-After.
function sharedRateLimitError(error) {
  return error?.outcome === "rate_limited";
}

function sharedRateLimitResponse(error) {
  const seconds = Number(error?.retryAfterSeconds);
  const value = Number.isFinite(seconds) && seconds >= 0 ? String(seconds) : null;
  return { status: 429, headers: { get: (name) => (String(name).toLowerCase() === "retry-after" ? value : null) } };
}
