import { BACKFILL_DB_NAME, BACKFILL_DB_VERSION, BACKFILL_RECOVERY_CHECKPOINT_VERSION, TERMINAL_QUEUE_STATES } from "./models.js";

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("indexeddb_request_failed"));
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error || new Error("indexeddb_transaction_aborted"));
    transaction.onerror = () => reject(transaction.error || new Error("indexeddb_transaction_failed"));
  });
}

function checkpointJob(job) {
  const safe = { ...job };
  delete safe.provider_options;
  delete safe.execution_owner;
  delete safe.execution_expires_at_ms;
  return structuredClone(safe);
}

function checkpointQueueItem(item) {
  const safe = { ...item };
  delete safe.envelope;
  delete safe.receiver_receipt;
  delete safe.lease_owner;
  delete safe.lease_expires_at_ms;
  if (safe.state === "leased") safe.state = safe.resume_state || "recovery_required";
  return structuredClone(recoveryRequiredItem(safe));
}

function checkpointRevision(revision) {
  return structuredClone(revision);
}

function recoveryRequiredItem(item) {
  if (item.state === "leased") item = { ...item, state: item.resume_state || "recovery_required" };
  if (item.state !== "captured_waiting_receiver") return item;
  return {
    ...item,
    state: "recovery_required",
    resume_state: "recovery_required",
    last_response_class: "browser_profile_recovery_required",
    last_error: "captured_envelope_not_present_in_recovery_checkpoint",
    lease_owner: null,
    lease_expires_at_ms: null,
  };
}

function recoveryCheckpointJob(job, hasRecoveryRequiredQueueItem = false) {
  const wasRunning = job.status === "running";
  const needsProfileRecovery = wasRunning || (job.status === "paused" && hasRecoveryRequiredQueueItem);
  return {
    ...structuredClone(job),
    status: needsProfileRecovery ? "paused" : job.status,
    cooldown_reason: needsProfileRecovery ? "browser_profile_recovery_required" : job.cooldown_reason,
    last_error: needsProfileRecovery ? "browser_profile_recovery_required" : job.last_error,
    execution_owner: null,
    execution_expires_at_ms: null,
  };
}

export class IndexedDbBackfillStore {
  constructor(indexedDb = globalThis.indexedDB, databaseName = BACKFILL_DB_NAME) {
    this.indexedDb = indexedDb;
    this.databaseName = databaseName;
    this.databasePromise = null;
  }

  async database() {
    if (!this.indexedDb) throw new Error("indexeddb_unavailable");
    if (!this.databasePromise) {
      this.databasePromise = new Promise((resolve, reject) => {
        const request = this.indexedDb.open(this.databaseName, BACKFILL_DB_VERSION);
        request.onupgradeneeded = () => {
          const database = request.result;
          if (!database.objectStoreNames.contains("jobs")) database.createObjectStore("jobs", { keyPath: "id" });
          if (!database.objectStoreNames.contains("revisions")) database.createObjectStore("revisions", { keyPath: "id" });
          if (!database.objectStoreNames.contains("queue")) {
            const queue = database.createObjectStore("queue", { keyPath: "id" });
            queue.createIndex("job_state_next", ["job_id", "state", "next_eligible_at_ms"], { unique: false });
            queue.createIndex("job_native", ["job_id", "provider", "native_id"], { unique: true });
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("indexeddb_open_failed"));
      });
    }
    return this.databasePromise;
  }

  async getJob(id) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readonly");
    return requestResult(tx.objectStore("jobs").get(id));
  }

  async putJob(job) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    tx.objectStore("jobs").put(structuredClone(job));
    await transactionDone(tx);
    return job;
  }

  async createJob(job) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    const store = tx.objectStore("jobs");
    const jobs = await requestResult(store.getAll());
    const existing = jobs.find(
      (candidate) => candidate.provider === job.provider && ["running", "paused"].includes(candidate.status),
    );
    if (existing) {
      tx.abort();
      throw new Error(`backfill_job_already_active:${job.provider}:${existing.id}`);
    }
    store.add(structuredClone(job));
    await transactionDone(tx);
    return job;
  }

  async acquireJobExecution(jobId, owner, nowMs, leaseMs) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    const store = tx.objectStore("jobs");
    const job = await requestResult(store.get(jobId));
    if (!job || job.status !== "running" || (job.execution_owner && job.execution_expires_at_ms > nowMs)) {
      await transactionDone(tx);
      return null;
    }
    const leased = {
      ...job,
      execution_owner: owner,
      execution_expires_at_ms: nowMs + leaseMs,
      execution_generation: (job.execution_generation || 0) + 1,
    };
    store.put(leased);
    await transactionDone(tx);
    return leased;
  }

  async assertJobExecution(jobId, owner, generation) {
    const job = await this.getJob(jobId);
    if (!job || job.execution_owner !== owner || job.execution_generation !== generation || job.status !== "running") {
      throw new Error(`stale_backfill_execution:${jobId}`);
    }
    return job;
  }

  async putJobCas(job, owner, generation) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    const store = tx.objectStore("jobs");
    const current = await requestResult(store.get(job.id));
    if (!current || current.execution_owner !== owner || current.execution_generation !== generation) {
      tx.abort();
      throw new Error(`stale_backfill_execution:${job.id}`);
    }
    const next = { ...job, execution_owner: owner, execution_expires_at_ms: current.execution_expires_at_ms, execution_generation: generation };
    store.put(structuredClone(next));
    await transactionDone(tx);
    return next;
  }

  async reserveProviderRequests(jobId, owner, generation, count, dailyKey, nextRequestAtMs) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    const store = tx.objectStore("jobs");
    const current = await requestResult(store.get(jobId));
    if (!current || current.execution_owner !== owner || current.execution_generation !== generation || current.status !== "running") {
      tx.abort();
      throw new Error(`stale_backfill_execution:${jobId}`);
    }
    const used = current.daily_key === dailyKey ? current.daily_requests || 0 : 0;
    if (used + count > current.policy.maxDailyRequests) {
      await transactionDone(tx);
      return null;
    }
    const next = { ...current, daily_key: dailyKey, daily_requests: used + count, next_request_at_ms: nextRequestAtMs };
    store.put(next);
    await transactionDone(tx);
    return next;
  }

  async releaseJobExecution(jobId, owner, generation) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    const store = tx.objectStore("jobs");
    const current = await requestResult(store.get(jobId));
    if (current?.execution_owner === owner && current.execution_generation === generation) {
      store.put({ ...current, execution_owner: null, execution_expires_at_ms: null });
    }
    await transactionDone(tx);
  }

  async controlJob(jobId, status, nowIsoValue, patch = {}, resumeQueueAtMs = null) {
    const db = await this.database();
    const tx = db.transaction(["jobs", "queue"], "readwrite");
    const store = tx.objectStore("jobs");
    const current = await requestResult(store.get(jobId));
    if (!current) {
      tx.abort();
      throw new Error(`backfill_job_not_found:${jobId}`);
    }
    const next = {
      ...current,
      ...patch,
      status,
      execution_owner: null,
      execution_expires_at_ms: null,
      execution_generation: (current.execution_generation || 0) + 1,
      updated_at: nowIsoValue,
    };
    store.put(next);
    const queueStore = tx.objectStore("queue");
    if (status === "running" && resumeQueueAtMs !== null) {
      const items = await requestResult(queueStore.getAll());
      for (const item of items) {
        if (item.job_id === jobId && item.state === "auth_required") {
          queueStore.put({
            ...item,
            state: "eligible",
            resume_state: "eligible",
            lease_owner: null,
            lease_expires_at_ms: null,
            next_eligible_at_ms: resumeQueueAtMs,
            last_response_class: null,
            last_error: null,
          });
        }
      }
    }
    if (status === "cancelled") {
      const items = await requestResult(queueStore.getAll());
      for (const item of items) {
        if (item.job_id === jobId && !["complete", "unchanged", "no_turns"].includes(item.state)) {
          queueStore.put({ ...item, state: "cancelled", lease_owner: null, lease_expires_at_ms: null });
        }
      }
    }
    await transactionDone(tx);
    return next;
  }

  async listJobs() {
    const db = await this.database();
    const tx = db.transaction("jobs", "readonly");
    return requestResult(tx.objectStore("jobs").getAll());
  }

  async putQueue(item) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    tx.objectStore("queue").put(structuredClone(item));
    await transactionDone(tx);
    return item;
  }

  async putQueueCas(jobId, owner, generation, item) {
    const db = await this.database();
    const tx = db.transaction(["jobs", "queue"], "readwrite");
    const job = await requestResult(tx.objectStore("jobs").get(jobId));
    if (!job || job.execution_owner !== owner || job.execution_generation !== generation || job.status !== "running") {
      tx.abort();
      throw new Error(`stale_backfill_execution:${jobId}`);
    }
    tx.objectStore("queue").put(structuredClone(item));
    await transactionDone(tx);
    return item;
  }

  async upsertDiscoveredCas(jobId, owner, generation, item) {
    const db = await this.database();
    const tx = db.transaction(["jobs", "queue"], "readwrite");
    const job = await requestResult(tx.objectStore("jobs").get(jobId));
    if (!job || job.execution_owner !== owner || job.execution_generation !== generation || job.status !== "running") {
      tx.abort();
      throw new Error(`stale_backfill_execution:${jobId}`);
    }
    const queue = tx.objectStore("queue");
    const existing = await requestResult(queue.index("job_native").get([item.job_id, item.provider, item.native_id]));
    if (!existing) queue.add(structuredClone(item));
    await transactionDone(tx);
    return existing || item;
  }

  async finalizeCaptureCas(job, owner, generation, item, revision, lastAck) {
    const db = await this.database();
    const tx = db.transaction(["jobs", "queue", "revisions"], "readwrite");
    const jobs = tx.objectStore("jobs");
    const current = await requestResult(jobs.get(job.id));
    if (!current || current.execution_owner !== owner || current.execution_generation !== generation || current.status !== "running") {
      tx.abort();
      throw new Error(`stale_backfill_execution:${job.id}`);
    }
    tx.objectStore("queue").put(structuredClone(item));
    if (revision) tx.objectStore("revisions").put(structuredClone(revision));
    const next = { ...current, last_ack: structuredClone(lastAck), last_error: null };
    jobs.put(next);
    await transactionDone(tx);
    return next;
  }

  async upsertDiscovered(item) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    const store = tx.objectStore("queue");
    const index = store.index("job_native");
    const existing = await requestResult(index.get([item.job_id, item.provider, item.native_id]));
    if (!existing) store.add(structuredClone(item));
    await transactionDone(tx);
    return existing || item;
  }

  async listQueue(jobId) {
    const db = await this.database();
    const tx = db.transaction("queue", "readonly");
    const all = await requestResult(tx.objectStore("queue").getAll());
    return all.filter((item) => item.job_id === jobId);
  }

  async getRevision(provider, nativeId) {
    const db = await this.database();
    const tx = db.transaction("revisions", "readonly");
    return requestResult(tx.objectStore("revisions").get(`${provider}:${nativeId}`));
  }

  async putRevision(revision) {
    const db = await this.database();
    const tx = db.transaction("revisions", "readwrite");
    tx.objectStore("revisions").put(structuredClone(revision));
    await transactionDone(tx);
    return revision;
  }

  async storedBytes(jobId) {
    const items = await this.listQueue(jobId);
    return new TextEncoder().encode(JSON.stringify(items)).length;
  }

  async exportRecoveryCheckpoint() {
    const db = await this.database();
    const tx = db.transaction(["jobs", "queue", "revisions"], "readonly");
    const [jobs, queue, revisions] = await Promise.all([
      requestResult(tx.objectStore("jobs").getAll()),
      requestResult(tx.objectStore("queue").getAll()),
      requestResult(tx.objectStore("revisions").getAll()),
    ]);
    return {
      version: BACKFILL_RECOVERY_CHECKPOINT_VERSION,
      jobs: jobs.map(checkpointJob),
      queue: queue.map(checkpointQueueItem),
      revisions: revisions.map(checkpointRevision),
    };
  }

  async restoreRecoveryCheckpoint(checkpoint) {
    if (!checkpoint || checkpoint.version !== BACKFILL_RECOVERY_CHECKPOINT_VERSION) return { restored: 0, reason: "checkpoint_unavailable" };
    const db = await this.database();
    const tx = db.transaction(["jobs", "queue", "revisions"], "readwrite");
    const jobs = tx.objectStore("jobs");
    if ((await requestResult(jobs.count())) > 0) {
      await transactionDone(tx);
      return { restored: 0, reason: "indexeddb_present" };
    }
    const queue = tx.objectStore("queue");
    const restoredQueue = (checkpoint.queue || []).map((item) => recoveryRequiredItem(structuredClone(item)));
    const recoveryRequiredJobs = new Set(restoredQueue.filter((item) => item.state === "recovery_required").map((item) => item.job_id));
    for (const job of checkpoint.jobs || []) jobs.put(recoveryCheckpointJob(job, recoveryRequiredJobs.has(job.id)));
    for (const item of restoredQueue) queue.put(item);
    const revisions = tx.objectStore("revisions");
    for (const revision of checkpoint.revisions || []) revisions.put(structuredClone(revision));
    await transactionDone(tx);
    return { restored: (checkpoint.jobs || []).length, reason: "browser_profile_recovery_required" };
  }

  async recoverExpiredLeases(jobId, nowMs) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    const store = tx.objectStore("queue");
    const all = await requestResult(store.getAll());
    let recovered = 0;
    for (const item of all) {
      if (item.job_id === jobId && item.state === "leased" && item.lease_expires_at_ms <= nowMs) {
        store.put({ ...item, state: item.resume_state || "eligible", lease_owner: null, lease_expires_at_ms: null });
        recovered += 1;
      }
    }
    await transactionDone(tx);
    return recovered;
  }

  async acquireNextLease(jobId, owner, nowMs, leaseMs, receiverOnly = false) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    const store = tx.objectStore("queue");
    const all = await requestResult(store.getAll());
    const targetProvider = all.find((item) => item.job_id === jobId)?.provider;
    const providerBusy = targetProvider && all.some(
      (item) => item.provider === targetProvider && item.state === "leased" && item.lease_expires_at_ms > nowMs,
    );
    if (providerBusy) {
      await transactionDone(tx);
      return null;
    }
    const candidate = all
      .filter((item) => item.job_id === jobId && (
        receiverOnly ? item.state === "captured_waiting_receiver" : ["eligible", "retry_wait"].includes(item.state)
      ))
      .filter((item) => !item.lease_owner || item.lease_expires_at_ms <= nowMs)
      .filter((item) => (item.next_eligible_at_ms || 0) <= nowMs)
      .sort((left, right) => (left.next_eligible_at_ms || 0) - (right.next_eligible_at_ms || 0))[0];
    if (candidate) {
      const leased = {
        ...candidate,
        resume_state: candidate.state,
        state: "leased",
        lease_owner: owner,
        lease_expires_at_ms: nowMs + leaseMs,
      };
      store.put(leased);
      await transactionDone(tx);
      return leased;
    }
    await transactionDone(tx);
    return null;
  }
}

export class MemoryBackfillStore {
  constructor() {
    this.jobs = new Map();
    this.queue = new Map();
    this.revisions = new Map();
  }

  async getJob(id) { return structuredClone(this.jobs.get(id)); }
  async putJob(job) { this.jobs.set(job.id, structuredClone(job)); return job; }
  async createJob(job) {
    const existing = [...this.jobs.values()].find(
      (candidate) => candidate.provider === job.provider && ["running", "paused"].includes(candidate.status),
    );
    if (existing) throw new Error(`backfill_job_already_active:${job.provider}:${existing.id}`);
    this.jobs.set(job.id, structuredClone(job));
    return job;
  }
  async acquireJobExecution(jobId, owner, nowMs, leaseMs) {
    const job = this.jobs.get(jobId);
    if (!job || job.status !== "running" || (job.execution_owner && job.execution_expires_at_ms > nowMs)) return null;
    const leased = { ...job, execution_owner: owner, execution_expires_at_ms: nowMs + leaseMs, execution_generation: (job.execution_generation || 0) + 1 };
    this.jobs.set(jobId, structuredClone(leased));
    return structuredClone(leased);
  }
  async assertJobExecution(jobId, owner, generation) {
    const job = this.jobs.get(jobId);
    if (!job || job.execution_owner !== owner || job.execution_generation !== generation || job.status !== "running") throw new Error(`stale_backfill_execution:${jobId}`);
    return structuredClone(job);
  }
  async putJobCas(job, owner, generation) {
    const current = this.jobs.get(job.id);
    if (!current || current.execution_owner !== owner || current.execution_generation !== generation) throw new Error(`stale_backfill_execution:${job.id}`);
    const next = { ...job, execution_owner: owner, execution_expires_at_ms: current.execution_expires_at_ms, execution_generation: generation };
    this.jobs.set(job.id, structuredClone(next));
    return next;
  }
  async reserveProviderRequests(jobId, owner, generation, count, dailyKeyValue, nextRequestAtMs) {
    const current = await this.assertJobExecution(jobId, owner, generation);
    const used = current.daily_key === dailyKeyValue ? current.daily_requests || 0 : 0;
    if (used + count > current.policy.maxDailyRequests) return null;
    const next = { ...current, daily_key: dailyKeyValue, daily_requests: used + count, next_request_at_ms: nextRequestAtMs };
    this.jobs.set(jobId, structuredClone(next));
    return next;
  }
  async releaseJobExecution(jobId, owner, generation) {
    const current = this.jobs.get(jobId);
    if (current?.execution_owner === owner && current.execution_generation === generation) this.jobs.set(jobId, { ...current, execution_owner: null, execution_expires_at_ms: null });
  }
  async controlJob(jobId, status, nowIsoValue, patch = {}, resumeQueueAtMs = null) {
    const current = this.jobs.get(jobId);
    if (!current) throw new Error(`backfill_job_not_found:${jobId}`);
    const next = { ...current, ...patch, status, execution_owner: null, execution_expires_at_ms: null, execution_generation: (current.execution_generation || 0) + 1, updated_at: nowIsoValue };
    this.jobs.set(jobId, structuredClone(next));
    if (status === "running" && resumeQueueAtMs !== null) {
      for (const item of this.queue.values()) {
        if (item.job_id === jobId && item.state === "auth_required") {
          this.queue.set(item.id, {
            ...item,
            state: "eligible",
            resume_state: "eligible",
            lease_owner: null,
            lease_expires_at_ms: null,
            next_eligible_at_ms: resumeQueueAtMs,
            last_response_class: null,
            last_error: null,
          });
        }
      }
    }
    if (status === "cancelled") {
      for (const item of this.queue.values()) {
        if (item.job_id === jobId && !["complete", "unchanged", "no_turns"].includes(item.state)) {
          this.queue.set(item.id, { ...item, state: "cancelled", lease_owner: null, lease_expires_at_ms: null });
        }
      }
    }
    return next;
  }
  async listJobs() { return [...this.jobs.values()].map((job) => structuredClone(job)); }
  async putQueue(item) { this.queue.set(item.id, structuredClone(item)); return item; }
  async putQueueCas(jobId, owner, generation, item) {
    await this.assertJobExecution(jobId, owner, generation);
    this.queue.set(item.id, structuredClone(item));
    return item;
  }
  async upsertDiscoveredCas(jobId, owner, generation, item) {
    await this.assertJobExecution(jobId, owner, generation);
    return this.upsertDiscovered(item);
  }
  async finalizeCaptureCas(job, owner, generation, item, revision, lastAck) {
    await this.assertJobExecution(job.id, owner, generation);
    this.queue.set(item.id, structuredClone(item));
    if (revision) this.revisions.set(revision.id, structuredClone(revision));
    const current = this.jobs.get(job.id);
    const next = { ...current, last_ack: structuredClone(lastAck), last_error: null };
    this.jobs.set(job.id, next);
    return structuredClone(next);
  }
  async upsertDiscovered(item) {
    const found = [...this.queue.values()].find(
      (candidate) => candidate.job_id === item.job_id && candidate.provider === item.provider && candidate.native_id === item.native_id,
    );
    if (!found) this.queue.set(item.id, structuredClone(item));
    return structuredClone(found || item);
  }
  async listQueue(jobId) { return [...this.queue.values()].filter((item) => item.job_id === jobId).map((item) => structuredClone(item)); }
  async getRevision(provider, nativeId) { return structuredClone(this.revisions.get(`${provider}:${nativeId}`)); }
  async putRevision(revision) { this.revisions.set(revision.id, structuredClone(revision)); return revision; }
  async storedBytes(jobId) { return new TextEncoder().encode(JSON.stringify(await this.listQueue(jobId))).length; }
  async exportRecoveryCheckpoint() {
    return {
      version: BACKFILL_RECOVERY_CHECKPOINT_VERSION,
      jobs: [...this.jobs.values()].map(checkpointJob),
      queue: [...this.queue.values()].map(checkpointQueueItem),
      revisions: [...this.revisions.values()].map(checkpointRevision),
    };
  }
  async restoreRecoveryCheckpoint(checkpoint) {
    if (!checkpoint || checkpoint.version !== BACKFILL_RECOVERY_CHECKPOINT_VERSION) return { restored: 0, reason: "checkpoint_unavailable" };
    if (this.jobs.size) return { restored: 0, reason: "indexeddb_present" };
    const restoredQueue = (checkpoint.queue || []).map((item) => recoveryRequiredItem(structuredClone(item)));
    const recoveryRequiredJobs = new Set(restoredQueue.filter((item) => item.state === "recovery_required").map((item) => item.job_id));
    for (const job of checkpoint.jobs || []) this.jobs.set(job.id, recoveryCheckpointJob(job, recoveryRequiredJobs.has(job.id)));
    for (const item of restoredQueue) this.queue.set(item.id, structuredClone(item));
    for (const revision of checkpoint.revisions || []) this.revisions.set(revision.id, structuredClone(revision));
    return { restored: (checkpoint.jobs || []).length, reason: "browser_profile_recovery_required" };
  }
  async recoverExpiredLeases(jobId, nowMs) {
    let recovered = 0;
    for (const item of this.queue.values()) {
      if (item.job_id === jobId && item.state === "leased" && item.lease_expires_at_ms <= nowMs) {
        this.queue.set(item.id, { ...item, state: item.resume_state || "eligible", lease_owner: null, lease_expires_at_ms: null });
        recovered += 1;
      }
    }
    return recovered;
  }
  async acquireNextLease(jobId, owner, nowMs, leaseMs, receiverOnly = false) {
    const targetProvider = [...this.queue.values()].find((item) => item.job_id === jobId)?.provider;
    if (targetProvider && [...this.queue.values()].some(
      (item) => item.provider === targetProvider && item.state === "leased" && item.lease_expires_at_ms > nowMs,
    )) return null;
    const candidate = [...this.queue.values()]
      .filter((item) => item.job_id === jobId && (
        receiverOnly ? item.state === "captured_waiting_receiver" : ["eligible", "retry_wait"].includes(item.state)
      ))
      .filter((item) => !item.lease_owner || item.lease_expires_at_ms <= nowMs)
      .filter((item) => (item.next_eligible_at_ms || 0) <= nowMs)
      .sort((left, right) => (left.next_eligible_at_ms || 0) - (right.next_eligible_at_ms || 0))[0];
    if (!candidate) return null;
    const leased = { ...candidate, resume_state: candidate.state, state: "leased", lease_owner: owner, lease_expires_at_ms: nowMs + leaseMs };
    this.queue.set(leased.id, structuredClone(leased));
    return structuredClone(leased);
  }
}

export function progressBuckets(items) {
  const buckets = { total: items.length, eligible: 0, complete: 0, no_turns: 0, retry: 0, error: 0, operator_action: 0 };
  for (const item of items) {
    if (["discovered", "eligible", "leased"].includes(item.state)) buckets.eligible += 1;
    if (["complete", "unchanged"].includes(item.state)) buckets.complete += 1;
    if (item.state === "no_turns") buckets.no_turns += 1;
    if (["retry_wait", "captured_waiting_receiver"].includes(item.state)) buckets.retry += 1;
    if (["auth_required", "recovery_required"].includes(item.state)) buckets.operator_action += 1;
    if (item.state === "failed") buckets.error += 1;
  }
  return buckets;
}

export function jobFinished(items) {
  return items.every((item) => TERMINAL_QUEUE_STATES.has(item.state));
}
