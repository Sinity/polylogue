import {
  DEFAULT_BACKFILL_POLICY,
  backfillAlarmName,
  dayKey,
  fullJitterDelay,
  retryAfterMs,
  serializedContentHash,
  serializedJson,
} from "./models.js";
import { jobFinished, progressBuckets } from "./storage.js";

function nowIso(nowMs) { return new Date(nowMs).toISOString(); }

function queueId(jobId, provider, nativeId) { return `${jobId}:${provider}:${nativeId}`; }

function mergePolicy(patch = {}) { return { ...DEFAULT_BACKFILL_POLICY, ...patch }; }

export class BackfillCoordinator {
  constructor({ store, adapters, receiver, alarms, clock = () => Date.now(), random = Math.random, instanceId = "extension-instance" }) {
    this.store = store;
    this.adapters = adapters;
    this.receiver = receiver;
    this.alarms = alarms;
    this.clock = clock;
    this.random = random;
    this.instanceId = instanceId;
  }

  async start({ provider, cutoff, policy = {}, provider_options = {} }) {
    if (!this.adapters[provider]) throw new Error(`unsupported_backfill_provider:${provider}`);
    const existing = (await this.store.listJobs()).find(
      (job) => job.provider === provider && ["running", "paused"].includes(job.status),
    );
    if (existing) throw new Error(`backfill_job_already_active:${provider}:${existing.id}`);
    const now = this.clock();
    const id = `backfill-${provider}-${now}-${Math.floor(this.random() * 1e9).toString(36)}`;
    const job = {
      id,
      provider,
      cutoff,
      provider_options,
      status: "running",
      inventory_cursor: "0",
      inventory_complete: false,
      policy: mergePolicy(policy),
      learned_cadence_ms: mergePolicy(policy).baseCadenceMs,
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
    };
    await this.store.putJob(job);
    await this.schedule(id, now);
    return job;
  }

  async control(jobId, action) {
    const job = await this.requireJob(jobId);
    const now = this.clock();
    const status = action === "start" || action === "resume" ? "running" : action === "pause" ? "paused" : action === "cancel" ? "cancelled" : null;
    if (!status) throw new Error(`unknown_backfill_action:${action}`);
    const next = {
      ...job,
      status,
      cooldown_until_ms: action === "resume" ? null : job.cooldown_until_ms,
      cooldown_reason: action === "resume" ? null : job.cooldown_reason,
      throttle_count: action === "resume" ? 0 : job.throttle_count,
      updated_at: nowIso(now),
    };
    await this.store.putJob(next);
    if (status === "cancelled") {
      for (const item of await this.store.listQueue(jobId)) {
        if (!item.state || !["complete", "no_turns"].includes(item.state)) await this.store.putQueue({ ...item, state: "cancelled" });
      }
    } else if (status === "running") {
      await this.schedule(jobId, now);
    }
    return this.status(jobId);
  }

  async status(jobId) {
    const job = await this.requireJob(jobId);
    const queue = await this.store.listQueue(jobId);
    return { ...job, progress: progressBuckets(queue) };
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
    let job = await this.requireJob(jobId);
    const now = this.clock();
    await this.store.recoverExpiredLeases(jobId, now);
    if (job.status !== "running") return this.status(jobId);
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
      job = await this.requireJob(jobId);
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
    if (job.inventory_complete && jobFinished(queue)) {
      await this.store.putJob({ ...job, status: "complete", updated_at: nowIso(this.clock()) });
    } else if (job.status === "running") {
      const candidates = queue.filter((item) => ["eligible", "retry_wait", "captured_waiting_receiver", "leased"].includes(item.state));
      const next = Math.max(job.next_request_at_ms || 0, Math.min(...candidates.map((item) => item.next_eligible_at_ms || 0).filter(Boolean), Infinity));
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
    if (job.daily_requests + requestCost > job.policy.maxDailyRequests) {
      return this.pauseJob(job, "daily_request_budget_exhausted", now);
    }
    try {
      result = await adapter.enumerate(job.inventory_cursor, job.cutoff);
    } catch (error) {
      const accounted = await this.recordProviderRequests(job, now, requestCost);
      return this.handleJobTransport(accounted, error, now);
    }
    const accounted = await this.recordProviderRequests(job, now, requestCost);
    if (result.classification !== "success") return this.handleProviderBlock(accounted, result.response, result.classification, now);
    const room = job.policy.maxQueueSize - queue.length;
    if (result.items.length > room) return this.pauseJob(accounted, "queue_budget_exhausted", now);
    for (const item of result.items) {
      await this.store.upsertDiscovered({
        id: queueId(job.id, job.provider, item.native_id),
        job_id: job.id,
        provider: job.provider,
        native_id: item.native_id,
        title: item.title || null,
        provider_updated_at: item.updated_at || null,
        state: "eligible",
        attempt_count: 0,
        next_eligible_at_ms: now,
        lease_owner: null,
        lease_expires_at_ms: null,
        last_response_class: "discovered",
        capture_fidelity: null,
        receiver_receipt: null,
        content_hash: null,
      });
    }
    const next = {
      ...accounted,
      provider_options: { ...accounted.provider_options, ...(result.provider_options || {}) },
      inventory_cursor: result.next_cursor,
      inventory_complete: Boolean(result.done),
      updated_at: nowIso(now),
    };
    await this.store.putJob(next);
    return next;
  }

  async processItem(job, item, now, requestCost = 1) {
    if (item.resume_state === "captured_waiting_receiver" && item.envelope) return this.submitReceiver(job, item, item.envelope, now);
    let response;
    try {
      response = await this.adapters[job.provider].fetchNative(item.native_id);
    } catch (error) {
      const accounted = await this.recordProviderRequests(job, now, requestCost);
      return this.retryTransport(accounted, item, error, now);
    }
    job = await this.recordProviderRequests(job, now, requestCost);
    const classification = this.adapters[job.provider].classifyResponse(response);
    if (classification !== "success") {
      if (classification === "rate_limited" || classification === "auth_or_challenge") {
        await this.store.putQueue({ ...item, state: classification === "auth_or_challenge" ? "auth_required" : "retry_wait", lease_owner: null, lease_expires_at_ms: null, last_response_class: classification, next_eligible_at_ms: now });
        return this.handleProviderBlock(job, response, classification, now);
      }
      if (classification === "transport") return this.retryTransport(job, item, new Error(`provider_http_${response.status}`), now);
      await this.store.putQueue({ ...item, state: "failed", lease_owner: null, lease_expires_at_ms: null, last_response_class: classification, last_error: `provider_http_${response.status}` });
      return job;
    }
    let capture;
    try {
      capture = await this.adapters[job.provider].normalizeCapture(response, item, { job_id: job.id, queue_id: item.id, instance_id: this.instanceId });
    } catch (error) {
      await this.store.putQueue({ ...item, state: "failed", lease_owner: null, lease_expires_at_ms: null, last_response_class: "contract_drift", last_error: String(error.message || error) });
      return job;
    }
    if (!capture.session?.turns?.length) {
      await this.store.putQueue({ ...item, state: "no_turns", lease_owner: null, lease_expires_at_ms: null, last_response_class: "native_empty", capture_fidelity: "native_full" });
      return job;
    }
    return this.submitReceiver(job, item, capture, now);
  }

  async submitReceiver(job, item, capture, now) {
    const serialized = serializedJson(capture);
    const hash = await serializedContentHash(serialized);
    await this.store.putQueue({ ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash, capture_fidelity: "native_full", lease_owner: null, lease_expires_at_ms: null, last_response_class: "captured" });
    try {
      const receipt = await this.receiver(capture, serialized);
      if (!receipt?.receiver_request_id || receipt.content_hash !== hash) throw new Error("receiver_ack_hash_mismatch");
      await this.store.putQueue({ ...item, state: "complete", envelope: null, content_hash: hash, capture_fidelity: "native_full", receiver_receipt: receipt, lease_owner: null, lease_expires_at_ms: null, last_response_class: "receiver_acked", completed_at: nowIso(now) });
      const next = { ...job, last_ack: { receiver_request_id: receipt.receiver_request_id, content_hash: hash, at: nowIso(now) }, last_error: null };
      await this.store.putJob(next);
      return next;
    } catch (error) {
      const attempt = (item.attempt_count || 0) + 1;
      await this.store.putQueue({ ...item, state: "captured_waiting_receiver", envelope: capture, content_hash: hash, capture_fidelity: "native_full", attempt_count: attempt, next_eligible_at_ms: now + fullJitterDelay(attempt, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random), lease_owner: null, lease_expires_at_ms: null, last_response_class: "receiver_down", last_error: String(error.message || error) });
      const next = { ...job, last_error: String(error.message || error), updated_at: nowIso(now) };
      await this.store.putJob(next);
      return next;
    }
  }

  async retryTransport(job, item, error, now) {
    const attempt = (item.attempt_count || 0) + 1;
    const terminal = attempt >= job.policy.maxTransportAttempts;
    await this.store.putQueue({ ...item, state: terminal ? "failed" : "retry_wait", attempt_count: attempt, next_eligible_at_ms: now + fullJitterDelay(attempt, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random), lease_owner: null, lease_expires_at_ms: null, last_response_class: "transport", last_error: String(error.message || error) });
    const failures = (job.transport_failures || 0) + 1;
    const next = { ...job, transport_failures: failures, last_error: String(error.message || error) };
    if (failures >= job.policy.breakerThreshold) return this.pauseJob(next, "repeated_transport_failures", now);
    await this.store.putJob(next);
    return next;
  }

  async handleProviderBlock(job, response, classification, now) {
    if (classification === "auth_or_challenge") return this.pauseJob({ ...job, last_error: "provider_auth_or_challenge" }, "provider_auth_or_challenge", now);
    if (classification !== "rate_limited") return this.handleJobTransport(job, new Error(`provider_${classification}`), now);
    const count = (job.throttle_count || 0) + 1;
    const learned = Math.min(job.policy.maxCadenceMs, Math.max(job.learned_cadence_ms * 2, job.policy.baseCadenceMs));
    const delay = Math.max(retryAfterMs(response?.headers, now) || 0, fullJitterDelay(count, learned, job.policy.maxCadenceMs, this.random));
    const next = { ...job, throttle_count: count, learned_cadence_ms: learned, cooldown_until_ms: now + delay, cooldown_reason: "provider_rate_limited", last_error: `provider_http_${response?.status || 429}`, updated_at: nowIso(now) };
    if (count >= job.policy.breakerThreshold) next.status = "paused";
    await this.store.putJob(next);
    await this.schedule(job.id, next.cooldown_until_ms);
    return next;
  }

  async handleJobTransport(job, error, now) {
    const failures = (job.transport_failures || 0) + 1;
    const next = { ...job, transport_failures: failures, last_error: String(error.message || error) };
    if (failures >= job.policy.breakerThreshold) return this.pauseJob(next, "repeated_transport_failures", now);
    const deadline = now + fullJitterDelay(failures, job.policy.baseCadenceMs, job.policy.maxCadenceMs, this.random);
    await this.store.putJob({ ...next, cooldown_until_ms: deadline, cooldown_reason: "transport_backoff", updated_at: nowIso(now) });
    await this.schedule(job.id, deadline);
    return { ...next, cooldown_until_ms: deadline, cooldown_reason: "transport_backoff" };
  }

  async recordProviderRequests(job, now, count) {
    const currentDay = dayKey(now);
    const requests = job.daily_key === currentDay ? (job.daily_requests || 0) + count : count;
    const jitter = Math.floor(this.random() * Math.max(1, job.learned_cadence_ms / 4));
    const next = { ...job, daily_key: currentDay, daily_requests: requests, next_request_at_ms: now + job.learned_cadence_ms + jitter, updated_at: nowIso(now) };
    await this.store.putJob(next);
    return next;
  }

  async resetDailyBudget(job, now) {
    const key = dayKey(now);
    if (job.daily_key === key) return job;
    const next = { ...job, daily_key: key, daily_requests: 0 };
    await this.store.putJob(next);
    return next;
  }

  async pauseJob(job, reason, now) {
    const next = { ...job, status: "paused", cooldown_reason: reason, last_error: job.last_error || reason, updated_at: nowIso(now) };
    await this.store.putJob(next);
    return next;
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
