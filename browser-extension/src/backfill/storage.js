import { BACKFILL_DB_NAME, BACKFILL_DB_VERSION, TERMINAL_QUEUE_STATES } from "./models.js";

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
  }

  async getJob(id) { return structuredClone(this.jobs.get(id)); }
  async putJob(job) { this.jobs.set(job.id, structuredClone(job)); return job; }
  async listJobs() { return [...this.jobs.values()].map((job) => structuredClone(job)); }
  async putQueue(item) { this.queue.set(item.id, structuredClone(item)); return item; }
  async upsertDiscovered(item) {
    const found = [...this.queue.values()].find(
      (candidate) => candidate.job_id === item.job_id && candidate.provider === item.provider && candidate.native_id === item.native_id,
    );
    if (!found) this.queue.set(item.id, structuredClone(item));
    return structuredClone(found || item);
  }
  async listQueue(jobId) { return [...this.queue.values()].filter((item) => item.job_id === jobId).map((item) => structuredClone(item)); }
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
    if (item.state === "complete") buckets.complete += 1;
    if (item.state === "no_turns") buckets.no_turns += 1;
    if (["retry_wait", "captured_waiting_receiver"].includes(item.state)) buckets.retry += 1;
    if (item.state === "auth_required") buckets.operator_action += 1;
    if (item.state === "failed") buckets.error += 1;
  }
  return buckets;
}

export function jobFinished(items) {
  return items.every((item) => TERMINAL_QUEUE_STATES.has(item.state));
}
