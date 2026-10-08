import { canonicalJson } from "./capture_jobs.js";
import { createSha256 } from "../vendor/sha256.js";
import { BACKFILL_DB_NAME, BACKFILL_DB_VERSION, BACKFILL_RECOVERY_CHECKPOINT_VERSION, TERMINAL_QUEUE_STATES } from "./models.js";

function captureRecordRow(record) {
  return { ...record, record_kind: record.record_kind || "turn", asset_refs: [
    ...(record.asset_refs || []),
    ...[...(record.turn?.attachments || []), ...(record.session_attachment ? [record.session_attachment] : [])]
      .flatMap((attachment) => attachment.staged_asset?.id ? [attachment.staged_asset.id] : []),
  ] };
}


function jobRow(job) { return { ...job, active_page: ["complete", "completed", "cancelled", "failed"].includes(job.status) ? 0 : 1 }; }

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("indexeddb_request_failed"));
  });
}

function visitCursor(request, visit) {
  return new Promise((resolve, reject) => {
    request.onerror = () => reject(request.error || new Error("indexeddb_request_failed"));
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) { resolve(); return; }
      const fail = (error) => {
        request.transaction?.abort();
        reject(error);
      };
      try {
        const result = visit(cursor);
        if (result?.then) result.then(() => cursor.continue(), fail);
        else cursor.continue();
      } catch (error) { fail(error); }
    };
  });
}

function transactionDone(transaction) {
  const done = new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error || new Error("indexeddb_transaction_aborted"));
    transaction.onerror = () => reject(transaction.error || new Error("indexeddb_transaction_failed"));
  });
  // A request may reject before its caller reaches the final transaction
  // await. Observe that rejection now; callers still await the original result.
  done.catch(() => undefined);
  return done;
}

function queueRow(item) {
  return { ...item, body_ref: item.body_ref || item.envelope?.capture_body_ref || null,
    record_ref: item.capture_record_ref || item.envelope?.capture_record_ref || null,
    source_refs: [...new Set([...(item.source_refs || []), ...(item.capture_source_refs || item.envelope?.capture_source_refs || []),
      ...(item.raw_acquisition_ref?.id ? [item.raw_acquisition_ref.id] : [])])] };
}

function checkpointJob(job) {
  const safe = { ...job };
  delete safe.recovery_source_checkpoint;
  delete safe.recovery_checkpoint_outcome;
  delete safe.provider_options;
  delete safe.active_page;
  delete safe.execution_owner;
  delete safe.execution_expires_at_ms;
  return globalThis.structuredClone(safe);
}

function checkpointQueueItem(item) {
  const safe = { ...item };
  delete safe.recovery_source_checkpoint;
  delete safe.envelope;
  delete safe.receiver_receipt;
  delete safe.lease_owner;
  delete safe.lease_expires_at_ms;
  if (safe.state === "leased") safe.state = safe.resume_state || "recovery_required";
  return globalThis.structuredClone(recoveryRequiredItem(safe));
}

function checkpointRevision(revision) {
  const safe = { ...revision };
  delete safe.recovery_source_checkpoint;
  return globalThis.structuredClone(safe);
}

function recoveryValueDigest(value) {
  const safe = { ...value };
  delete safe.recovery_source_checkpoint;
  delete safe.recovery_checkpoint_outcome;
  return createSha256().update(canonicalJson(safe)).hex();
}

function canAdvanceRecovery(current, root) {
  if (!current) return true;
  const proof = current.recovery_source_checkpoint;
  // An operator mutation or locally acquired custody is never replaced by a
  // metadata checkpoint. The proof hashes the exact installed row, including
  // its paused state, rather than a lossy checkpoint projection.
  if (!proof || proof.value_digest !== recoveryValueDigest(current)) return false;
  if (proof.account_scope !== root.account_scope) throw new Error("checkpoint_recovery_scope_conflict");
  if (proof.remote_job_id === root.remote_job_id) {
    if (proof.sequence === root.sequence && proof.digest !== root.digest) throw new Error("checkpoint_recovery_source_conflict");
    return root.sequence > proof.sequence;
  }
  const prior = Date.parse(proof.acknowledged_at || "");
  const next = Date.parse(root.acknowledged_at || "");
  if (!Number.isFinite(prior) || !Number.isFinite(next) || prior === next) throw new Error("checkpoint_recovery_source_conflict");
  return next > prior;
}

function recoveredRow(value, root) {
  return { ...value, recovery_source_checkpoint: {
    account_scope: root.account_scope, remote_job_id: root.remote_job_id,
    digest: root.digest, sequence: root.sequence, acknowledged_at: root.acknowledged_at,
    value_digest: recoveryValueDigest(value),
  } };
}

export function nativeCacheOrder(current, incoming) {
  const revision = (row) => {
    const value = row.headers?.update_time ?? row.headers?.updated_at ?? row.headers?.updatedAt ?? row.headers?.modifyTime;
    return typeof value === "number" ? value * (value < 10_000_000_000 ? 1000 : 1) : Date.parse(value || "");
  };
  const priorRevision = revision(current); const nextRevision = revision(incoming);
  if (Number.isFinite(priorRevision) && Number.isFinite(nextRevision) && priorRevision !== nextRevision) return nextRevision - priorRevision;
  if (Number.isSafeInteger(current.acquisition_sequence) && Number.isSafeInteger(incoming.acquisition_sequence)) {
    return incoming.acquisition_sequence - current.acquisition_sequence;
  }
  // Passive app responses have no pre-request witness. Keep their honest
  // background observation time; never turn publication order into a counter.
  const priorObserved = Date.parse(current.observed_at || ""); const nextObserved = Date.parse(incoming.observed_at || "");
  return Number.isFinite(priorObserved) && Number.isFinite(nextObserved) ? nextObserved - priorObserved : 0;
}

function recoveryRequiredItem(item) {
  if (item.state === "leased") item = { ...item, state: item.resume_state || "recovery_required" };
  if (item.state !== "captured_waiting_receiver" && !item.capture_bundle_ref && !item.capture_record_ref && !item.body_ref) return item;
  if (item.state === "cancelled") return { ...item, last_response_class: "browser_profile_recovery_required", last_error: "acquired_custody_not_present_in_recovery_checkpoint" };
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

function recoveryCheckpointJob(job, hasRecoveryRequiredQueueItem = false, preservePausedState = false) {
  const wasRunning = job.status === "running";
  const needsProfileRecovery = wasRunning || (job.status === "paused" && hasRecoveryRequiredQueueItem && !preservePausedState && !job.cooldown_reason);
  return {
    ...globalThis.structuredClone(job),
    ...(needsProfileRecovery ? { status: "paused", cooldown_reason: "browser_profile_recovery_required",
      last_error: "browser_profile_recovery_required" } : {}),
    execution_owner: null,
    execution_expires_at_ms: null,
  };
}

// Conversion of disposable pre-streaming queue state. This is invoked only
// while publishing an old database/checkpoint into the current queue format.
function convertAcquisitionRefusal(item) {
  if (item.state !== "bridge_oversize") return item;
  const acquired = Boolean(item.envelope || item.body_ref || item.receiver_receipt || item.capture_ref);
  return { ...item, state: acquired ? "recovery_required" : "eligible",
    resume_state: acquired ? "recovery_required" : null,
    next_eligible_at_ms: 0, lease_owner: null, lease_expires_at_ms: null,
    last_response_class: acquired ? "browser_profile_recovery_required" : "discovered",
    last_error: acquired ? "capture_refusal_retains_acquired_evidence" : null };
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
        request.onupgradeneeded = (event) => {
          const database = request.result;
          if (!database.objectStoreNames.contains("captures")) database.createObjectStore("captures", { keyPath: "id" });
          if (!database.objectStoreNames.contains("capture_records")) {
            const records = database.createObjectStore("capture_records", { keyPath: ["capture_id", "key"] });
            records.createIndex("semantic_occurrences", ["capture_id", "semantic_digest"], { unique: false });
            records.createIndex("capture_order", ["capture_id", "record_kind", "order_time", "order_id", "occurrence"], { unique: true });
          }
          if (!database.objectStoreNames.contains("jobs")) database.createObjectStore("jobs", { keyPath: "id" });
          if (!database.objectStoreNames.contains("revisions")) database.createObjectStore("revisions", { keyPath: "id" });
          if (!database.objectStoreNames.contains("capture_retry_metadata")) {
            const retries = database.createObjectStore("capture_retry_metadata", { keyPath: "queue_order", autoIncrement: true });
            retries.createIndex("id", "id", { unique: true });
            database.createObjectStore("capture_retry_bodies", { keyPath: "id" });
            database.createObjectStore("capture_retry_state");
          }
          if (!database.objectStoreNames.contains("queue")) {
            const queue = database.createObjectStore("queue", { keyPath: "id" });
            queue.createIndex("job_state_next", ["job_id", "state", "next_eligible_at_ms"], { unique: false });
            queue.createIndex("job_native", ["job_id", "provider", "native_id"], { unique: true });
          }
          const jobs = request.transaction.objectStore("jobs");
          if (!jobs.indexNames.contains("provider_status")) jobs.createIndex("provider_status", ["provider", "status"], { unique: false });
          if (!jobs.indexNames.contains("active_page")) {
            jobs.createIndex("active_page", ["active_page", "id"], { unique: true });
            const rows = jobs.openCursor();
            rows.onsuccess = () => { const cursor = rows.result; if (cursor) { cursor.update(jobRow(cursor.value)); cursor.continue(); } };
          }
          const captures = request.transaction.objectStore("captures");
          if (!captures.indexNames.contains("source_refs")) captures.createIndex("source_refs", "source_refs", { multiEntry: true });
          if (!captures.indexNames.contains("record_roots")) captures.createIndex("record_roots", "record_ref", { unique: false });
          if (!captures.indexNames.contains("checkpoint_job")) captures.createIndex("checkpoint_job", ["kind", "job_id", "id"], { unique: true });
          if (!captures.indexNames.contains("native_bundle_scope")) captures.createIndex("native_bundle_scope", ["kind", "owner.tab_id", "provider", "native_id"], { unique: false });
          if (!captures.indexNames.contains("native_revision")) captures.createIndex("native_revision", ["provider", "native_id", "raw_revision_sha256"], { unique: false });
          if (!captures.indexNames.contains("acquisition_order")) captures.createIndex("acquisition_order", ["acquisition_raw_id", "id"], { unique: true });
          const records = request.transaction.objectStore("capture_records");
          if (!records.indexNames.contains("asset_refs")) records.createIndex("asset_refs", "asset_refs", { multiEntry: true });
          if (!records.indexNames.contains("record_roots")) records.createIndex("record_roots", "record_ref", { unique: false });
          const queue = request.transaction.objectStore("queue");
          if (!queue.indexNames.contains("job_id")) queue.createIndex("job_id", "job_id", { unique: false });
          if (!queue.indexNames.contains("provider_state")) queue.createIndex("provider_state", ["provider", "state"], { unique: false });
          if (!queue.indexNames.contains("delivery_order")) queue.createIndex("delivery_order", ["delivery_kind", "delivery_sequence", "id"], { unique: true });
          if (!queue.indexNames.contains("delivery_due")) queue.createIndex("delivery_due", ["delivery_kind", "next_attempt_at_ms", "id"], { unique: true });
          if (!queue.indexNames.contains("body_ref")) queue.createIndex("body_ref", "body_ref", { unique: false });
          if (!queue.indexNames.contains("record_ref")) queue.createIndex("record_ref", "record_ref", { unique: false });
          if (!queue.indexNames.contains("source_refs")) queue.createIndex("source_refs", "source_refs", { multiEntry: true });
          if (event.oldVersion > 0 && event.oldVersion < 4) {
            const conversion = queue.openCursor();
            conversion.onsuccess = () => {
              const cursor = conversion.result;
              if (!cursor) return;
              const original = cursor.value;
              const converted = queueRow(convertAcquisitionRefusal(original));
              cursor.update(converted);
              cursor.continue();
            };
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("indexeddb_open_failed"));
      });
    }
    return this.databasePromise;
  }

  async importCaptureRetries(entries) {
    const db = await this.database();
    const tx = db.transaction(["capture_retry_metadata", "capture_retry_bodies", "capture_retry_state"], "readwrite");
    const settled = transactionDone(tx);
    settled.catch(() => undefined);
    try {
      const state = tx.objectStore("capture_retry_state");
      if (!await requestResult(state.get("local_storage_transferred"))) {
        const records = tx.objectStore("capture_retry_metadata");
        for (const { metadata, envelope } of entries) {
          const existing = await requestResult(records.index("id").get(metadata.id));
          if (existing) continue;
          records.put(metadata);
          tx.objectStore("capture_retry_bodies").put({ id: metadata.id, envelope });
        }
        // Cache publication may fail. This same transaction prevents a stale
        // old cache from resurrecting an already delivered body on restart.
        state.put(true, "local_storage_transferred");
      }
      await settled;
    } catch (error) {
      try { tx.abort(); } catch { /* Already physically settled. */ }
      await settled.catch(() => undefined);
      throw error;
    }
  }

  // Separate stores keep queue/status reads independent of retained capture
  // size. One transaction admits both the original body and its retry owner.
  async putCaptureRetry(metadata, envelope) {
    const db = await this.database();
    const tx = db.transaction(["capture_retry_metadata", "capture_retry_bodies"], "readwrite");
    const settled = transactionDone(tx);
    settled.catch(() => undefined);
    try {
      const records = tx.objectStore("capture_retry_metadata");
      const existing = await requestResult(records.index("id").get(metadata.id));
      const entry = { ...metadata, ...(existing ? { queue_order: existing.queue_order } : {}) };
      records.put(entry);
      tx.objectStore("capture_retry_bodies").put({ id: metadata.id, envelope });
      await settled;
    } catch (error) {
      try { tx.abort(); } catch { /* Already physically settled. */ }
      await settled.catch(() => undefined);
      throw error;
    }
  }

  async listCaptureRetries() {
    const db = await this.database();
    return requestResult(db.transaction("capture_retry_metadata", "readonly").objectStore("capture_retry_metadata").getAll());
  }

  async *captureRetryInputs() {
    const db = await this.database();
    let after = null;
    for (;;) {
      const tx = db.transaction("capture_retry_metadata", "readonly");
      const settled = transactionDone(tx);
      const row = await requestResult(tx.objectStore("capture_retry_metadata").openCursor(
        after === null ? undefined : globalThis.IDBKeyRange.lowerBound(after, true),
      ));
      const metadata = row ? globalThis.structuredClone(row.value) : null;
      await settled;
      if (!metadata) return;
      if (!Number.isSafeInteger(metadata.queue_order) || metadata.queue_order < 1) throw new Error("capture_retry_metadata_invalid");
      after = metadata.queue_order;
      yield metadata;
    }
  }

  async retireCaptureRetry(id) {
    const db = await this.database();
    const tx = db.transaction(["capture_retry_metadata", "capture_retry_bodies"], "readwrite");
    const settled = transactionDone(tx);
    settled.catch(() => undefined);
    try {
      const records = tx.objectStore("capture_retry_metadata");
      const metadata = await requestResult(records.index("id").get(id));
      if (metadata) records.delete(metadata.queue_order);
      tx.objectStore("capture_retry_bodies").delete(id);
      await settled;
    } catch (error) {
      try { tx.abort(); } catch { /* Already physically settled. */ }
      await settled.catch(() => undefined);
      throw error;
    }
  }

  async hasCaptureRetry(id) {
    const db = await this.database();
    return Boolean(await requestResult(db.transaction("capture_retry_metadata", "readonly").objectStore("capture_retry_metadata").index("id").getKey(id)));
  }

  async getCaptureRetryEnvelope(id) {
    const db = await this.database();
    const record = await requestResult(db.transaction("capture_retry_bodies", "readonly").objectStore("capture_retry_bodies").get(id));
    if (!record?.envelope) throw new Error("capture_retry_body_missing");
    return record.envelope;
  }

  async replaceCaptureRetryMetadata(entries) {
    const db = await this.database();
    const tx = db.transaction(["capture_retry_metadata", "capture_retry_bodies"], "readwrite");
    const settled = transactionDone(tx);
    settled.catch(() => undefined);
    try {
      const records = tx.objectStore("capture_retry_metadata");
      const retained = new Set(entries.map((entry) => entry.id));
      const existing = await requestResult(records.getAll());
      for (const entry of existing) {
        if (!retained.has(entry.id)) {
          records.delete(entry.queue_order);
          tx.objectStore("capture_retry_bodies").delete(entry.id);
        }
      }
      for (const entry of entries) records.put(entry);
      await settled;
    } catch (error) {
      try { tx.abort(); } catch { /* Already physically settled. */ }
      await settled.catch(() => undefined);
      throw error;
    }
  }

  async putDelivery(entry, bodyRoot = null) {
    const db = await this.database(); const tx = db.transaction(["captures", "queue"], "readwrite");
    const done = transactionDone(tx);
    try { const row = await this.publishDelivery(tx, entry, bodyRoot); await done; return row; }
    catch (error) { try { tx.abort(); } catch { /* A completed transaction cannot be aborted again. */ } await done.catch(() => undefined); throw error; }
  }

  async publishDelivery(tx, entry, bodyRoot = null) {
    const queue = tx.objectStore("queue"); const captures = tx.objectStore("captures");
    if (bodyRoot && bodyRoot.id !== `body:${entry.body_ref}`) { tx.abort(); throw new Error("capture_delivery_body_root_mismatch"); }
    const previous = await requestResult(queue.get(entry.id));
    if (previous && previous.delivery_kind !== "foreground") { tx.abort(); throw new Error("capture_delivery_identity_conflict"); }
    let sequence = previous?.delivery_sequence;
    if (sequence === undefined) {
      const counter = await requestResult(captures.get("foreground-delivery-sequence"));
      sequence = (counter?.sequence || 0) + 1;
      if (!Number.isSafeInteger(sequence)) { tx.abort(); throw new Error("capture_delivery_sequence_unrepresentable"); }
      captures.put({ id: "foreground-delivery-sequence", kind: "delivery-sequence", sequence });
    }
    const row = { ...entry, delivery_kind: "foreground", delivery_sequence: sequence,
      next_attempt_at_ms: Date.parse(entry.next_attempt_at || "") || 0 };
    if (bodyRoot) captures.put(bodyRoot);
    if (entry.observation_ref) {
      const observation = await requestResult(captures.get(entry.observation_ref.id));
      if (observation && (observation.kind !== "dom-observation" || observation.token !== entry.observation_ref.token)) {
        tx.abort(); throw new Error("capture_observation_owner_mismatch");
      }
      captures.delete(entry.observation_ref.id);
    }
    queue.put(queueRow(row)); return row;
  }

  async prepareTypedBody({ captureId, createCapture, turns, attachments, bodyRoot, delivery, signal }) {
    const db = await this.database();
    const tx = db.transaction(["captures", "capture_records", "queue"], "readwrite");
    const done = transactionDone(tx);
    try {
      signal?.throwIfAborted();
      const records = tx.objectStore("capture_records");
      if (createCapture) tx.objectStore("captures").put({ id: captureId, kind: "typed-capture", state: "ready" });
      for (let ordinal = 0; ordinal < (turns?.length || 0); ordinal++) {
        signal?.throwIfAborted();
        await requestResult(records.put(captureRecordRow({ capture_id: captureId, key: String(ordinal),
          occurrence: ordinal, order_time: 0, order_id: "", turn: turns[ordinal], asset_failures: [] })));
      }
      for (let ordinal = 0; ordinal < (attachments?.length || 0); ordinal++) {
        signal?.throwIfAborted();
        await requestResult(records.put(captureRecordRow({ capture_id: captureId, key: `session-attachment:${ordinal}`,
          record_kind: "session-attachment", occurrence: ordinal, order_time: 0, order_id: "",
          session_attachment: attachments[ordinal], turn: null, asset_failures: [] })));
      }
      signal?.throwIfAborted();
      if (delivery) await this.publishDelivery(tx, delivery, bodyRoot);
      else tx.objectStore("captures").put(bodyRoot);
      signal?.throwIfAborted();
      await done;
    } catch (error) {
      try { tx.abort(); } catch { /* A completed transaction cannot be aborted again. */ }
      await done.catch(() => undefined); throw error;
    }
  }

  async getDelivery(id) {
    const db = await this.database();
    const row = await requestResult(db.transaction("queue", "readonly").objectStore("queue").get(id));
    return row?.delivery_kind === "foreground" ? row : null;
  }

  async deleteDelivery(id) {
    const db = await this.database(); const tx = db.transaction("queue", "readwrite");
    const store = tx.objectStore("queue");
    const row = await requestResult(store.get(id));
    if (row?.delivery_kind === "foreground") store.delete(id);
    await transactionDone(tx);
  }

  async deliveryCount() {
    const db = await this.database();
    return requestResult(db.transaction("queue", "readonly").objectStore("queue").index("delivery_order").count(
      globalThis.IDBKeyRange.bound(["foreground"], ["foreground", []], false, true),
    ));
  }

  async *deliveries({ dueAt = null, after = null } = {}) {
    const db = await this.database();
    const indexName = dueAt === null ? "delivery_order" : "delivery_due";
    const upper = dueAt === null ? ["foreground", []] : ["foreground", dueAt, []];
    for (;;) {
      const range = globalThis.IDBKeyRange.bound(after || ["foreground"], upper, Boolean(after), true);
      const cursor = await requestResult(db.transaction("queue", "readonly").objectStore("queue").index(indexName).openCursor(range));
      if (!cursor) return;
      after = cursor.key;
      yield { entry: cursor.value, cursor: after };
    }
  }

  async *captures() {
    const db = await this.database(); let after = null;
    for (;;) {
      const range = after === null ? null : globalThis.IDBKeyRange.lowerBound(after, true);
      const cursor = await requestResult(db.transaction("captures", "readonly").objectStore("captures").openCursor(range));
      if (!cursor) return;
      after = cursor.key;
      yield cursor.value;
    }
  }

  async beginNativeBundle({ owner, provider, nativeId, bundleId, requiredReplies, queueContext = null, extensionInstanceId = null }) {
    if (!bundleId || !nativeId || !owner?.tab_id || provider !== "grok" || owner.provider !== provider || !Array.isArray(requiredReplies) ||
        !requiredReplies.includes("conversation") || !requiredReplies.includes("responses") ||
        new Set(requiredReplies).size !== requiredReplies.length || requiredReplies.some((name) => !["conversation", "responses", "response_nodes"].includes(name))) throw new Error("native_bundle_identity_invalid");
    let id = `bundle:${JSON.stringify([owner.tab_id, provider, nativeId, bundleId])}`;
    const db = await this.database(); const tx = db.transaction(queueContext ? ["captures", "queue", "jobs"] : "captures", "readwrite");
    let item = null;
    if (queueContext) {
      const job = await requestResult(tx.objectStore("jobs").get(queueContext.jobId));
      item = await requestResult(tx.objectStore("queue").get(queueContext.itemId));
      if (!job || job.status !== "running" || job.execution_owner !== queueContext.owner || job.execution_generation !== queueContext.generation ||
          !item || item.job_id !== job.id || item.native_id !== nativeId || item.provider !== provider || item.lease_owner !== queueContext.owner) {
        tx.abort(); throw new Error(`stale_backfill_execution:${queueContext.jobId}`);
      }
      id = item.capture_bundle_ref || id;
    }
    const captures = tx.objectStore("captures");
    if (!queueContext) {
      let pending = null; let conflict = false;
      await visitCursor(captures.index("native_bundle_scope").openCursor(globalThis.IDBKeyRange.only(["native-bundle", owner.tab_id, provider, nativeId])), (cursor) => {
        const candidate = cursor.value;
        if (candidate.kind !== "native-bundle" || candidate.queue_context || candidate.provider !== provider || candidate.native_id !== nativeId ||
            JSON.stringify(candidate.owner) !== JSON.stringify(owner) || candidate.state === "ready") return;
        if (pending && pending !== candidate.id) conflict = true;
        pending = candidate.id;
      });
      if (conflict) { tx.abort(); throw new Error("native_bundle_pending_identity_conflict"); }
      if (pending) id = pending;
    }
    const previous = await requestResult(captures.get(id));
    if (previous) {
      if (previous.owner.tab_id !== owner.tab_id || previous.owner.provider !== owner.provider ||
          (owner.document_id && previous.owner.document_id !== owner.document_id) || JSON.stringify(previous.required_replies) !== JSON.stringify(requiredReplies)) {
        tx.abort(); throw new Error("native_bundle_identity_conflict");
      }
      const resumed = { ...previous, queue_context: queueContext || previous.queue_context };
      captures.put(resumed);
      await transactionDone(tx); return resumed;
    }
    if (item?.capture_bundle_ref) { tx.abort(); throw new Error("native_bundle_recovery_missing"); }
    const counter = await requestResult(captures.get("native-acquisition-sequence"));
    const sequence = (counter?.sequence || 0) + 1;
    if (!Number.isSafeInteger(sequence)) { tx.abort(); throw new Error("native_acquisition_sequence_unrepresentable"); }
    const row = { id, kind: "native-bundle", owner, provider, native_id: nativeId, bundle_id: bundleId,
      extension_instance_id: extensionInstanceId, acquisition_sequence: sequence, observed_at: new Date().toISOString(), state: "acquiring", required_replies: requiredReplies, replies: {}, outcomes: {}, source_refs: [], queue_context: queueContext };
    captures.put({ id: "native-acquisition-sequence", kind: "acquisition-sequence", sequence });
    captures.put(row);
    if (item) tx.objectStore("queue").put(queueRow({ ...item, capture_bundle_ref: row.id }));
    await transactionDone(tx); return row;
  }

  async publishNativeBundleReply(bundleId, owner, name, ref, outcome) {
    if (!["conversation", "responses", "response_nodes"].includes(name)) throw new Error("native_bundle_reply_invalid");
    const db = await this.database(); const tx = db.transaction(["captures", "queue", "jobs"], "readwrite");
    const captures = tx.objectStore("captures"); const row = await requestResult(captures.get(bundleId));
    if (!row || row.kind !== "native-bundle" || row.owner.tab_id !== owner.tab_id || row.provider !== owner.provider ||
        (row.owner.document_id && row.owner.document_id !== owner.document_id)) { tx.abort(); throw new Error("native_bundle_owner_mismatch"); }
    if (row.replies[name] && (row.replies[name].id !== ref?.id || row.replies[name].token !== ref?.token)) {
      tx.abort(); throw new Error("native_bundle_reply_conflict");
    }
    const replies = ref ? { ...row.replies, [name]: ref } : row.replies;
    const next = { ...row, owner, replies, outcomes: { ...row.outcomes, [name]: outcome }, source_refs: Object.values(replies).map((reply) => reply.id) };
    captures.put(next);
    if (row.queue_context) {
      const queue = tx.objectStore("queue"); const item = await requestResult(queue.get(row.queue_context.itemId));
      const job = await requestResult(tx.objectStore("jobs").get(row.queue_context.jobId));
      // A sealed reply remains evidence after lease loss. Only the current
      // execution may publish its progress into the queue item.
      if (item?.capture_bundle_ref === row.id && item.lease_owner === row.queue_context.owner &&
          job?.status === "running" && job.execution_owner === row.queue_context.owner &&
          job.execution_generation === row.queue_context.generation) {
        queue.put(queueRow({ ...item, capture_bundle_replies: Object.keys(replies) }));
      }
    }
    await transactionDone(tx); return next;
  }

  async finishNativeBundle(bundleId, owner) {
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const captures = tx.objectStore("captures"); const row = await requestResult(captures.get(bundleId));
    if (!row || row.kind !== "native-bundle" || JSON.stringify(row.owner) !== JSON.stringify(owner)) { tx.abort(); throw new Error("native_bundle_owner_mismatch"); }
    const throttled = Object.values(row.outcomes).some((outcome) => outcome.status === 429);
    const complete = !throttled && row.required_replies.every((name) => row.replies[name] && row.outcomes[name]?.ok === true);
    const next = { ...row, state: complete ? "ready" : "pending", error: throttled ? "provider_rate_limited" : complete ? null : "native_bundle_incomplete" };
    captures.put(next); await transactionDone(tx); return next;
  }

  async nextNativeAcquisitionSequence() {
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const store = tx.objectStore("captures"); const counter = await requestResult(store.get("native-acquisition-sequence"));
    const sequence = (counter?.sequence || 0) + 1;
    if (!Number.isSafeInteger(sequence)) { tx.abort(); throw new Error("native_acquisition_sequence_unrepresentable"); }
    store.put({ id: "native-acquisition-sequence", kind: "acquisition-sequence", sequence });
    await transactionDone(tx); return sequence;
  }

  async bindNativeAcquisition(ref, owner, context) {
    const db = await this.database(); const tx = db.transaction(["captures", "jobs", "queue"], "readwrite");
    const done = transactionDone(tx);
    const job = await requestResult(tx.objectStore("jobs").get(context.jobId));
    const queue = tx.objectStore("queue"); const item = await requestResult(queue.get(context.itemId));
    const raw = await requestResult(tx.objectStore("captures").get(`raw:${ref.id}`));
    if (!job || !item || !raw || raw.kind !== "native-acquisition" || job.status !== "running" ||
        job.execution_owner !== context.owner || job.execution_generation !== context.generation ||
        !(job.execution_expires_at_ms > Date.now()) || !(item.lease_expires_at_ms > Date.now()) ||
        item.job_id !== job.id || item.lease_owner !== context.owner || item.state !== "leased" ||
        item.native_id !== context.nativeId || job.provider !== owner.provider || item.provider !== owner.provider ||
        JSON.stringify(raw.owner) !== JSON.stringify(owner) || raw.raw_ref.token !== ref.token ||
        !/^h1:[A-Za-z0-9_-]{43}$/.test(job.account_scope || "")) {
      tx.abort(); await done.catch(() => undefined); throw new Error("native_acquisition_claim_replaced");
    }
    if (item.raw_acquisition_ref && item.raw_acquisition_ref.id !== ref.id) {
      tx.abort(); await done.catch(() => undefined); throw new Error("native_acquisition_recovery_pending");
    }
    queue.put(queueRow({ ...item, raw_acquisition_ref: ref,
      source_refs: [...new Set([...(item.source_refs || []), ref.id])] }));
    tx.objectStore("captures").put({ ...raw, queue_context: context });
    await done;
  }

  async retireNativeAcquisition(rawId) {
    const db = await this.database(); const tx = db.transaction(["captures", "queue"], "readwrite");
    const done = transactionDone(tx); const queue = tx.objectStore("queue");
    await visitCursor(queue.index("source_refs").openCursor(globalThis.IDBKeyRange.only(rawId)), (cursor) => {
      const item = cursor.value;
      if (item.raw_acquisition_ref?.id !== rawId) return;
      cursor.update({ ...item, raw_acquisition_ref: null, source_refs: (item.source_refs || []).filter((ref) => ref !== rawId) });
    });
    tx.objectStore("captures").delete(`raw:${rawId}`);
    await done;
  }

  async reserveCaptureObservation(owner, nativeId, kind = "dom-observation", workerOwner = null, extensionInstanceId = null) {
    if (!["dom-observation", "native-invocation"].includes(kind)) throw new Error("capture_observation_kind_invalid");
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const captures = tx.objectStore("captures");
    const closed = await requestResult(captures.get(`closed:${JSON.stringify([owner.tab_id, owner.document_id])}`));
    if (closed) { tx.abort(); throw new Error("capture_document_lost"); }
    const counter = await requestResult(captures.get("native-acquisition-sequence"));
    const sequence = (counter?.sequence || 0) + 1;
    if (!Number.isSafeInteger(sequence)) { tx.abort(); throw new Error("native_acquisition_sequence_unrepresentable"); }
    const row = { id: globalThis.crypto.randomUUID(), token: globalThis.crypto.randomUUID(), kind, state: "open",
      owner, provider: owner.provider, native_id: nativeId, extension_instance_id: extensionInstanceId, ...(kind === "native-invocation" ? { worker_owner: workerOwner } : {}), acquisition_sequence: sequence, observed_at: new Date().toISOString() };
    captures.put({ id: "native-acquisition-sequence", kind: "acquisition-sequence", sequence }); captures.put(row);
    await transactionDone(tx); return { id: row.id, token: row.token };
  }

  async closeNativeInvocation(ref, settled = true) {
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const done = transactionDone(tx); const records = tx.objectStore("captures");
    const row = await requestResult(records.get(ref.id));
    if (row && (row.kind !== "native-invocation" || row.token !== ref.token)) {
      tx.abort(); await done.catch(() => undefined); throw new Error("native_invocation_owner_mismatch");
    }
    if (row) { if (settled) records.delete(row.id); else records.put({ ...row, state: "closing" }); }
    await done;
  }

  async pinNativeCache({ owner, provider, nativeId, rawRef, relatedRefs = {}, acquisition = null, headers, observedAt, acquisitionSequence }) {
    if (acquisitionSequence !== null && acquisitionSequence !== undefined &&
        (!Number.isSafeInteger(acquisitionSequence) || acquisitionSequence <= 0)) throw new Error("native_acquisition_sequence_invalid");
    const id = `cache:${JSON.stringify([owner.tab_id, owner.document_id, provider, nativeId])}`;
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const store = tx.objectStore("captures");
    const closed = await requestResult(store.get(`closed:${JSON.stringify([owner.tab_id, owner.document_id])}`));
    if (closed) { await transactionDone(tx); throw new Error("capture_document_lost"); }
    const previous = await requestResult(store.get(id));
    const row = { id, kind: "native-cache", owner, provider, native_id: nativeId,
      raw_ref: rawRef, related_refs: relatedRefs, acquisition, headers, observed_at: observedAt, acquisition_sequence: acquisitionSequence,
      source_refs: [rawRef?.id || rawRef, ...Object.values(relatedRefs).map((ref) => ref?.id || ref)] };
    // Compare only this page/provider identity, with provider revision first.
    if (previous && nativeCacheOrder(previous, row) <= 0 && previous.raw_ref?.id !== rawRef?.id) {
      await transactionDone(tx); return { current: previous, retired: [] };
    }
    // This key has exactly one current-page owner. Do not enumerate unrelated
    // captures to replace the row we just read in this transaction.
    const retired = previous ? [previous.raw_ref, ...Object.values(previous.related_refs || {})] : [];
    store.put(row); await transactionDone(tx); return { current: row, retired };
  }

  async *releaseNativeCaches(owner) {
    const db = await this.database();
    const closing = db.transaction("captures", "readwrite");
    closing.objectStore("captures").put({ id: `closed:${JSON.stringify([owner.tab_id, owner.document_id])}`, kind: "document-closed", owner });
    await transactionDone(closing);
    let after = null;
    for (;;) {
      const tx = db.transaction("captures", "readwrite"); const done = transactionDone(tx);
      const range = after === null ? null : globalThis.IDBKeyRange.lowerBound(after, true);
      const cursor = await requestResult(tx.objectStore("captures").openCursor(range));
      if (!cursor) { await done; return; }
      after = cursor.key; const row = cursor.value;
      const released = (row.kind === "native-cache" || row.kind === "dom-observation") &&
        row.owner.tab_id === owner.tab_id && row.owner.document_id === owner.document_id;
      if (released) cursor.delete();
      await done;
      if (released && row.kind === "native-cache") {
        yield row.raw_ref;
        for (const ref of Object.values(row.related_refs || {})) yield ref;
      }
    }
  }

  async captureReferences(stageId) {
    const db = await this.database();
    if (await requestResult(db.transaction("queue", "readonly").objectStore("queue").index("body_ref").count(stageId))) return true;
    if (await requestResult(db.transaction("queue", "readonly").objectStore("queue").index("source_refs").count(stageId))) return true;
    if (await requestResult(db.transaction("capture_records", "readonly").objectStore("capture_records").index("asset_refs").count(stageId))) return true;
    let after = null;
    for (;;) {
      const tx = db.transaction("captures", "readonly");
      const index = tx.objectStore("captures").index("source_refs");
      const request = index.openCursor(globalThis.IDBKeyRange.only(stageId));
      let sought = false;
      const cursor = await new Promise((resolve, reject) => {
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const value = request.result;
          if (value && after !== null && this.indexedDb.cmp(value.primaryKey, after) < 0 && !sought) {
            sought = true; value.continuePrimaryKey(stageId, after); return;
          }
          if (value && after !== null && this.indexedDb.cmp(value.primaryKey, after) === 0) { value.continue(); return; }
          resolve(value);
        };
      });
      if (!cursor) return false;
      after = cursor.primaryKey;
      const row = cursor.value;
      if (row.kind !== "asset-acquisition") return true;
      // Acquisition evidence follows its raw revision's independent roots;
      // the acquisition row itself never pins the raw revision.
      if (await requestResult(db.transaction("captures", "readonly").objectStore("captures").index("source_refs").count(row.acquisition_raw_id))) return true;
    }
  }

  async reserveAcquisition(identity, ref, producerId) {
    const id = `asset:${JSON.stringify(identity)}`;
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const store = tx.objectStore("captures");
    let row = await requestResult(store.get(id));
    if (!row) {
      row = { id, kind: "asset-acquisition", acquisition_raw_id: identity[1], identity, ref,
        producer_id: producerId, source_refs: [ref.id] };
      store.add(row);
    }
    await transactionDone(tx);
    return row;
  }

  async releaseAcquisition(identity, stageId) {
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const store = tx.objectStore("captures"); const key = `asset:${JSON.stringify(identity)}`;
    const previous = await requestResult(store.get(key));
    if (previous?.ref.id === stageId) store.delete(key);
    await transactionDone(tx);
  }

  async publishAcquisition(identity, result) {
    const id = `asset:${JSON.stringify(identity)}`;
    const previous = await this.getCapture(id);
    if (!previous || previous.ref.id !== result.asset.staged_asset.id) throw new Error("capture_acquisition_owner_mismatch");
    await this.putCapture({ ...previous, result });
  }

  async *acquisitions(rawId) {
    const db = await this.database(); let after = null;
    for (;;) {
      const range = globalThis.IDBKeyRange.bound(after || [rawId], [rawId, []], Boolean(after), true);
      const cursor = await requestResult(db.transaction("captures", "readonly").objectStore("captures").index("acquisition_order").openCursor(range));
      if (!cursor) return;
      after = cursor.key;
      yield cursor.value;
    }
  }

  async recordRootCount(captureId) {
    const db = await this.database();
    const tx = db.transaction(["captures", "queue", "capture_records"], "readonly");
    const [captures, queue, snapshots] = await Promise.all([
      requestResult(tx.objectStore("captures").index("record_roots").count(captureId)),
      requestResult(tx.objectStore("queue").index("record_ref").count(captureId)),
      requestResult(tx.objectStore("capture_records").index("record_roots").count(captureId)),
    ]);
    return captures + queue + snapshots;
  }

  async foregroundDeliveryRoot(captureId) {
    const db = await this.database(); const tx = db.transaction(["captures", "queue"], "readonly");
    const delivery = await requestResult(tx.objectStore("queue").index("record_ref").get(captureId));
    if (delivery?.delivery_kind === "foreground") return { delivery };
    const body = await requestResult(tx.objectStore("captures").index("record_roots").get(captureId));
    return body?.kind === "receiver-body" ? { body, body_ref: body.id.slice("body:".length) } : null;
  }

  async hasNormalizedNativeSource(stageId) {
    const db = await this.database(); const tx = db.transaction("captures", "readonly");
    let retained = false;
    await visitCursor(tx.objectStore("captures").index("source_refs").openCursor(globalThis.IDBKeyRange.only(stageId)), (cursor) => {
      const row = cursor.value;
      if (row.state === "ready" && row.raw_ref === stageId) retained = true;
    });
    return retained;
  }

  async publishForegroundNativeCapture(capture, bundleId = null) {
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const captures = tx.objectStore("captures");
    if (bundleId) {
      const bundle = await requestResult(captures.get(bundleId));
      if (bundle && (bundle.kind !== "native-bundle" || bundle.queue_context || bundle.state !== "ready" ||
          bundle.provider !== capture.provider || bundle.native_id !== capture.native_id ||
          JSON.stringify(bundle.owner) !== JSON.stringify(capture.owner) ||
          !bundle.source_refs.every((ref) => capture.source_refs.includes(ref)))) {
        tx.abort(); throw new Error("native_bundle_custody_transfer_invalid");
      }
      // A resumed normalization may follow a completed publication of the
      // same immutable raw input. Its source roots remain independently owned.
      if (!bundle) {
        let retained = false;
        await visitCursor(captures.index("source_refs").openCursor(globalThis.IDBKeyRange.only(capture.raw_ref)), (cursor) => {
          const row = cursor.value;
          if ((row.kind === "native-cache" || row.state === "ready") && row.provider === capture.provider &&
              row.native_id === capture.native_id && JSON.stringify(row.owner) === JSON.stringify(capture.owner) &&
              capture.source_refs.every((ref) => row.source_refs?.includes(ref))) retained = true;
        });
        if (!retained) { tx.abort(); throw new Error("native_bundle_custody_transfer_missing"); }
      }
      captures.delete(bundleId);
    }
    const rawAcquisition = await requestResult(captures.get(`raw:${capture.raw_ref}`));
    if (rawAcquisition && (JSON.stringify(rawAcquisition.owner) !== JSON.stringify(capture.owner) || rawAcquisition.state !== "pending-normalization")) {
      tx.abort(); throw new Error("native_acquisition_custody_transfer_invalid");
    }
    captures.put({ ...capture, delivery_kind: "foreground" });
    captures.delete(`raw:${capture.raw_ref}`);
    await transactionDone(tx);
  }

  async publishNativeCapture(capture, envelope, context) {
    const db = await this.database(); const tx = db.transaction(["captures", "queue", "jobs"], "readwrite");
    const job = await requestResult(tx.objectStore("jobs").get(context.jobId));
    const queue = tx.objectStore("queue"); const item = await requestResult(queue.get(context.itemId));
    if (!job || job.status !== "running" || job.execution_owner !== context.owner || job.execution_generation !== context.generation ||
        !item || item.job_id !== job.id || item.native_id !== capture.native_id || item.provider !== capture.provider || item.lease_owner !== context.owner) {
      tx.abort(); throw new Error(`stale_backfill_execution:${context.jobId}`);
    }
    const captures = tx.objectStore("captures");
    if (item.capture_bundle_ref) {
      const bundle = await requestResult(captures.get(item.capture_bundle_ref));
      if (!bundle || bundle.kind !== "native-bundle" || bundle.native_id !== capture.native_id || bundle.provider !== capture.provider ||
          !bundle.source_refs.every((ref) => capture.source_refs.includes(ref))) {
        tx.abort(); throw new Error("native_bundle_custody_transfer_invalid");
      }
    }
    const rawAcquisition = await requestResult(captures.get(`raw:${capture.raw_ref}`));
    if (rawAcquisition && (JSON.stringify(rawAcquisition.owner) !== JSON.stringify(capture.owner) || rawAcquisition.state !== "pending-normalization")) {
      tx.abort(); throw new Error("native_acquisition_custody_transfer_invalid");
    }
    captures.put(capture);
    captures.delete(`raw:${capture.raw_ref}`);
    queue.put(queueRow({ ...item, envelope, capture_record_ref: capture.id, capture_source_refs: capture.source_refs,
      capture_bundle_ref: null, capture_bundle_replies: null, resume_state: "captured_waiting_receiver" }));
    if (item.capture_bundle_ref) captures.delete(item.capture_bundle_ref);
    await transactionDone(tx);
  }

  async deleteCaptureRecord(captureId, key) {
    const db = await this.database(); const tx = db.transaction("capture_records", "readwrite");
    tx.objectStore("capture_records").delete([captureId, key]); await transactionDone(tx);
  }

  async getCapture(id) {
    const db = await this.database();
    return requestResult(db.transaction("captures", "readonly").objectStore("captures").get(id));
  }
  async *failedNativeCaptures(current) {
    const db = await this.database(); let after = null;
    for (;;) {
      const revisionKey = [current.provider, current.native_id, current.raw_revision_sha256];
      const request = db.transaction("captures", "readonly").objectStore("captures").index("native_revision").openCursor(globalThis.IDBKeyRange.only(revisionKey));
      let sought = false;
      const cursor = await new Promise((resolve, reject) => {
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const value = request.result;
          if (value && after !== null && this.indexedDb.cmp(value.primaryKey, after) < 0 && !sought) {
            sought = true; value.continuePrimaryKey(revisionKey, after); return;
          }
          if (value && after !== null && this.indexedDb.cmp(value.primaryKey, after) === 0) { value.continue(); return; }
          resolve(value);
        };
      });
      if (!cursor) return;
      after = cursor.primaryKey;
      if (cursor.value.id !== current.id && cursor.value.state === "failed") yield cursor.value;
    }
  }
  async putCapture(capture) {
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const done = transactionDone(tx); const records = tx.objectStore("captures");
    if (capture.kind === "native-acquisition" && capture.state === "acquiring") {
      const current = await requestResult(records.get(capture.id));
      if (!current && capture.invocation_ref) {
        const invocation = await requestResult(records.get(capture.invocation_ref.id));
        if (!invocation || invocation.kind !== "native-invocation" || invocation.state !== "open" ||
            invocation.token !== capture.invocation_ref.token || invocation.native_id !== capture.invocation_native_id ||
            JSON.stringify(invocation.owner) !== JSON.stringify(capture.owner)) {
          tx.abort(); await done.catch(() => undefined); throw new Error("native_invocation_owner_mismatch");
        }
      }
      // Startup may have read acquiring metadata before the producer sealed it.
      // Its delayed publication cannot revoke already committed raw custody.
      if (current?.kind === "native-acquisition" && current.state === "pending-normalization") {
        if (JSON.stringify(current.owner) !== JSON.stringify(capture.owner) ||
            current.raw_ref?.id !== capture.raw_ref?.id || current.raw_ref?.token !== capture.raw_ref?.token) {
          tx.abort(); await done.catch(() => undefined); throw new Error("native_acquisition_identity_conflict");
        }
        await done; return current;
      }
    }
    records.put({ ...capture, source_refs: [...(capture.source_refs || []), capture.raw_ref?.id || capture.raw_ref, ...Object.values(capture.related_refs || {}).map((ref) => ref?.id || ref)].filter(Boolean) }); await done; return capture;
  }
  async putCaptureRecord(record) {
    const db = await this.database(); const tx = db.transaction("capture_records", "readwrite");
    tx.objectStore("capture_records").put(captureRecordRow(record)); await transactionDone(tx);
  }
  async semanticOccurrences(captureId, digest) {
    const db = await this.database();
    return requestResult(db.transaction("capture_records", "readonly").objectStore("capture_records").index("semantic_occurrences").count([captureId, digest]));
  }
  async getCaptureRecord(captureId, key) {
    const db = await this.database();
    return requestResult(db.transaction("capture_records", "readonly").objectStore("capture_records").get([captureId, key]));
  }
  async *captureRecords(captureId) {
    const db = await this.database(); let after = null;
    for (;;) {
      const tx = db.transaction("capture_records", "readonly");
      const range = after
        ? globalThis.IDBKeyRange.bound(after, [captureId, []], true, true)
        : globalThis.IDBKeyRange.bound([captureId], [captureId, []], false, true);
      const cursor = await requestResult(tx.objectStore("capture_records").index("capture_order").openCursor(range));
      if (!cursor) return;
      after = cursor.key;
      yield cursor.value;
    }
  }
  async discardCapture(captureId) {
    const db = await this.database(); const tx = db.transaction(["captures", "capture_records"], "readwrite");
    tx.objectStore("captures").delete(captureId);
    tx.objectStore("capture_records").delete(globalThis.IDBKeyRange.bound([captureId], [captureId, []], false, true));
    await transactionDone(tx);
  }

  async getJob(id) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readonly");
    return requestResult(tx.objectStore("jobs").get(id));
  }

  async putJob(job) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    tx.objectStore("jobs").put(jobRow(globalThis.structuredClone(job)));
    await transactionDone(tx);
    return job;
  }

  async putCheckpointOutcome(jobId, outcome) {
    const db = await this.database(); const tx = db.transaction("jobs", "readwrite");
    const done = transactionDone(tx); const store = tx.objectStore("jobs");
    const current = await requestResult(store.get(jobId));
    if (!current) { tx.abort(); await done.catch(() => undefined); throw new Error(`backfill_job_not_found:${jobId}`); }
    // Derived checkpoint observation updates only this field. It cannot
    // overwrite concurrent operator status, execution lease or acquired custody.
    store.put({ ...current, recovery_checkpoint_outcome: outcome });
    await done;
  }

  async createJob(job) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    const store = tx.objectStore("jobs");
    const index = store.index("provider_status");
    const candidates = await Promise.all(["running", "paused"].map((status) => requestResult(index.get([job.provider, status]))));
    const existing = candidates.find(Boolean);
    if (existing) {
      tx.abort();
      throw new Error(`backfill_job_already_active:${job.provider}:${existing.id}`);
    }
    store.add(jobRow(globalThis.structuredClone(job)));
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
    store.put(jobRow(leased));
    await transactionDone(tx);
    return leased;
  }

  async renewJobExecution(jobId, owner, generation, nowMs, leaseMs) {
    const db = await this.database(); const tx = db.transaction(["jobs", "queue"], "readwrite");
    const jobs = tx.objectStore("jobs"); const current = await requestResult(jobs.get(jobId));
    if (!current || current.status !== "running" || current.execution_owner !== owner || current.execution_generation !== generation) {
      tx.abort(); throw new Error(`stale_backfill_execution:${jobId}`);
    }
    jobs.put(jobRow({ ...current, execution_expires_at_ms: nowMs + leaseMs }));
    await visitCursor(tx.objectStore("queue").index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)), (cursor) => {
      if (cursor.value.state === "leased" && cursor.value.lease_owner === owner) cursor.update({ ...cursor.value, lease_expires_at_ms: nowMs + leaseMs });
    });
    await transactionDone(tx);
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
    store.put(jobRow(globalThis.structuredClone(next)));
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
    store.put(jobRow(next));
    await transactionDone(tx);
    return next;
  }

  async releaseJobExecution(jobId, owner, generation) {
    const db = await this.database();
    const tx = db.transaction("jobs", "readwrite");
    const store = tx.objectStore("jobs");
    const current = await requestResult(store.get(jobId));
    if (current?.execution_owner === owner && current.execution_generation === generation) {
      store.put(jobRow({ ...current, execution_owner: null, execution_expires_at_ms: null }));
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
    store.put(jobRow(next));
    const queueStore = tx.objectStore("queue");
    if (status === "running" && resumeQueueAtMs !== null) {
      await visitCursor(queueStore.index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)), (cursor) => {
        const item = cursor.value;
        if (item.job_id === jobId && ["auth_required"].includes(item.state)) {
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
      });
    }
    if (status === "cancelled") {
      await visitCursor(queueStore.index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)), (cursor) => {
        const item = cursor.value;
        if (item.job_id === jobId && !["complete", "unchanged", "superseded", "no_turns"].includes(item.state)) {
          queueStore.put({ ...item, state: "cancelled", lease_owner: null, lease_expires_at_ms: null });
        }
      });
    }
    await transactionDone(tx);
    return next;
  }

  async jobPage({ cursor = null, pageSize = 25, activeOnly = false } = {}) {
    if (!Number.isSafeInteger(pageSize) || pageSize < 1 || (cursor !== null && typeof cursor !== "string")) throw new Error("backfill_page_invalid");
    const db = await this.database(); const tx = db.transaction("jobs", "readonly"); const done = transactionDone(tx);
    const jobs = []; const source = activeOnly ? tx.objectStore("jobs").index("active_page") : tx.objectStore("jobs");
    const countRange = activeOnly ? globalThis.IDBKeyRange.bound([1], [1, []], false, true) : null;
    const pageRange = activeOnly ? globalThis.IDBKeyRange.bound(cursor === null ? [1] : [1, cursor], [1, []], cursor !== null, true)
      : cursor === null ? null : globalThis.IDBKeyRange.lowerBound(cursor, true);
    const totalPromise = requestResult(source.count(countRange)); const request = source.openCursor(pageRange);
    let hasMore = false;
    await new Promise((resolve, reject) => { request.onerror = () => reject(request.error); request.onsuccess = () => {
      const entry = request.result; if (!entry) { resolve(); return; }
      if (jobs.length === pageSize) { hasMore = true; resolve(); return; }
      jobs.push(entry.value); entry.continue();
    }; });
    const total = await totalPromise; await done;
    return { jobs, total, cursor: hasMore ? jobs.at(-1).id : null, has_more: hasMore };
  }

  async *jobs() {
    const db = await this.database(); let after = null;
    for (;;) {
      const range = after === null ? null : globalThis.IDBKeyRange.lowerBound(after, true);
      const cursor = await requestResult(db.transaction("jobs", "readonly").objectStore("jobs").openCursor(range));
      if (!cursor) return;
      after = cursor.key;
      yield cursor.value;
    }
  }

  async getQueue(id) {
    const db = await this.database();
    return requestResult(db.transaction("queue", "readonly").objectStore("queue").get(id));
  }

  async putQueue(item) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    tx.objectStore("queue").put(queueRow(globalThis.structuredClone(item)));
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
    tx.objectStore("queue").put(queueRow(globalThis.structuredClone(item)));
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
    if (!existing) queue.add(globalThis.structuredClone(item));
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
    tx.objectStore("queue").put(queueRow(globalThis.structuredClone(item)));
    if (revision) tx.objectStore("revisions").put(globalThis.structuredClone(revision));
    const next = { ...current, last_ack: globalThis.structuredClone(lastAck), last_error: null };
    jobs.put(jobRow(next));
    await transactionDone(tx);
    return next;
  }

  async upsertDiscovered(item) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    const store = tx.objectStore("queue");
    const index = store.index("job_native");
    const existing = await requestResult(index.get([item.job_id, item.provider, item.native_id]));
    if (!existing) store.add(globalThis.structuredClone(item));
    await transactionDone(tx);
    return existing || item;
  }

  async queueSummary(jobId, nowMs = 0) {
    const db = await this.database(); const tx = db.transaction("queue", "readonly");
    const done = transactionDone(tx); const summary = emptyQueueSummary();
    await visitCursor(tx.objectStore("queue").index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)),
      (cursor) => includeQueueSummary(summary, cursor.value, nowMs));
    await done; return summary;
  }

  async *jobRecords() { yield* this.jobs(); }

  async queuePage(jobId, { cursor = null, pageSize = 50 } = {}) {
    if (!Number.isSafeInteger(pageSize) || pageSize < 1 || (cursor !== null && typeof cursor !== "string")) throw new Error("backfill_page_invalid");
    const db = await this.database(); const tx = db.transaction("queue", "readonly"); const done = transactionDone(tx);
    const source = tx.objectStore("queue"); const items = [];
    const totalPromise = requestResult(source.index("job_id").count(globalThis.IDBKeyRange.only(jobId)));
    const request = source.index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)); let hasMore = false;
    await new Promise((resolve, reject) => { request.onerror = () => reject(request.error); request.onsuccess = () => {
      const entry = request.result; if (!entry) { resolve(); return; }
      if (cursor !== null && entry.primaryKey < cursor) { entry.continuePrimaryKey(jobId, cursor); return; }
      if (cursor !== null && entry.primaryKey === cursor) { entry.continue(); return; }
      if (items.length === pageSize) { hasMore = true; resolve(); return; }
      items.push(entry.value); entry.continue();
    }; });
    const total = await totalPromise; await done;
    return { items, total, cursor: hasMore ? items.at(-1).id : null, has_more: hasMore };
  }

  async getRevision(provider, nativeId) {
    const db = await this.database();
    const tx = db.transaction("revisions", "readonly");
    return requestResult(tx.objectStore("revisions").get(`${provider}:${nativeId}`));
  }

  async putRevision(revision) {
    const db = await this.database();
    const tx = db.transaction("revisions", "readwrite");
    tx.objectStore("revisions").put(globalThis.structuredClone(revision));
    await transactionDone(tx);
    return revision;
  }

  async createRecoverySnapshot(jobId, snapshotId, artifactId, { kind = "checkpoint-snapshot", token = null } = {}) {
    if (!["checkpoint-snapshot", "checkpoint-export"].includes(kind)) throw new Error("checkpoint_snapshot_owner_mismatch");
    const db = await this.database();
    // Copy under one transaction; no filesystem or network await occurs
    // before these source and custody records commit together.
    const tx = db.transaction(["jobs", "queue", "revisions", "captures", "capture_records"], "readwrite");
    const done = transactionDone(tx);
    const captures = tx.objectStore("captures");
    const prior = await requestResult(captures.get(snapshotId));
    if (prior) {
      if (prior.kind !== kind || prior.token !== token || prior.job_id !== jobId || prior.artifact_id !== artifactId) {
        tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_snapshot_owner_mismatch");
      }
      await done; return prior;
    }
    const job = await requestResult(tx.objectStore("jobs").get(jobId));
    if (!job) { tx.abort(); await done.catch(() => undefined); throw new Error(`unknown_backfill_job:${jobId}`); }
    const records = tx.objectStore("capture_records");
    let occurrence = 0;
    const publish = (collection, key, value, original = null, exportCapture = null) => {
      records.put({ capture_id: snapshotId, key: `${collection}:${key}`, record_kind: `checkpoint:${collection}`,
        order_time: 0, order_id: "", occurrence: occurrence++, checkpoint_value: value,
        export_capture: exportCapture,
        record_ref: original?.record_ref || original?.capture_record_ref || original?.capture_bundle_ref || null,
        asset_refs: original ? [original.body_ref, ...(original.source_refs || original.capture_source_refs || []),
          ...(exportCapture?.source_refs || [])].filter(Boolean) : [] });
    };
    publish("jobs", job.id, checkpointJob(job));
    await visitCursor(tx.objectStore("queue").index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)), async (cursor) => {
      const original = cursor.value;
      const captureId = original.record_ref || original.capture_record_ref || original.capture_bundle_ref;
      const exportCapture = kind === "checkpoint-export" && captureId ? await requestResult(captures.get(captureId)) : null;
      publish("queue", original.id, kind === "checkpoint-export" ? globalThis.structuredClone(original) : checkpointQueueItem(original), original, exportCapture);
    });
    await visitCursor(tx.objectStore("revisions").openCursor(), (cursor) => {
      if (cursor.value.provider === job.provider) publish("revisions", cursor.value.id, checkpointRevision(cursor.value));
    });
    const snapshot = { id: snapshotId, kind, token, state: "ready", job_id: jobId,
      provider: job.provider, job: checkpointJob(job), version: BACKFILL_RECOVERY_CHECKPOINT_VERSION,
      artifact_id: artifactId, source_refs: [artifactId], record_count: occurrence };
    captures.add(snapshot);
    await done;
    return snapshot;
  }

  async pendingRecoverySnapshot(jobId, kind = "checkpoint-snapshot") {
    const db = await this.database();
    const range = globalThis.IDBKeyRange.bound([kind, jobId], [kind, jobId, []], false, true);
    const request = db.transaction("captures", "readonly").objectStore("captures").index("checkpoint_job").openCursor(range);
    return new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor && cursor.value.state !== "ready") { cursor.continue(); return; }
        resolve(cursor?.value || null);
      };
    });
  }

  async acknowledgeExportSnapshot(snapshot, receipt) {
    const db = await this.database(); const tx = db.transaction(["captures", "capture_records"], "readwrite");
    const done = transactionDone(tx); const captures = tx.objectStore("captures");
    const root = await requestResult(captures.get(snapshot.id));
    if (root?.kind !== "checkpoint-export" || root.token !== snapshot.token || root.job_id !== snapshot.job_id) {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_export_owner_mismatch");
    }
    if (root.state === "acknowledged") {
      if (root.export_receipt.digest !== receipt.digest || root.export_receipt.size_bytes !== receipt.size_bytes) {
        tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_export_receipt_conflict");
      }
    } else {
      captures.put({ id: root.id, kind: root.kind, token: root.token, job_id: root.job_id, provider: root.provider,
        state: "acknowledged", export_receipt: receipt, artifact_id: root.artifact_id, source_refs: [], record_count: 0 });
      const request = tx.objectStore("capture_records").openCursor(globalThis.IDBKeyRange.bound([root.id], [root.id, []], false, true));
      await new Promise((resolve, reject) => {
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const cursor = request.result;
          if (!cursor) return resolve();
          // Keep exact copy references until the staging owner retires each
          // file. Restart cleanup must not rediscover a whole directory.
          if (cursor.value.record_kind !== "checkpoint:evidence") cursor.delete();
          cursor.continue();
        };
      });
    }
    await done;
  }

  async beginCheckpointRecovery(root) {
    if (!Number.isSafeInteger(root.sequence) || root.sequence < 0 ||
        typeof root.account_scope !== "string" || !root.account_scope ||
        typeof root.digest !== "string" || !/^sha256:[a-f0-9]{64}$/.test(root.digest)) {
      throw new Error("checkpoint_recovery_source_invalid");
    }
    const db = await this.database(); const tx = db.transaction("captures", "readwrite");
    const done = transactionDone(tx); const captures = tx.objectStore("captures");
    const prior = await requestResult(captures.get(root.id));
    if (prior && (prior.kind !== "checkpoint-recovery" || prior.remote_job_id !== root.remote_job_id ||
        prior.account_scope !== root.account_scope || prior.digest !== root.digest)) {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_recovery_owner_mismatch");
    }
    let selected = prior;
    if (!prior) captures.add({ ...root, kind: "checkpoint-recovery", state: "receiving", record_count: 0 });
    else if (root.sequence > prior.sequence) {
      selected = { ...prior, sequence: root.sequence, acknowledged_at: root.acknowledged_at };
      captures.put(selected);
    }
    await done; return selected || root;
  }

  async retireCheckpointRecovery(rootId) {
    const db = await this.database(); const tx = db.transaction(["captures", "capture_records"], "readwrite");
    const done = transactionDone(tx); const captures = tx.objectStore("captures");
    const root = await requestResult(captures.get(rootId));
    if (root?.kind !== "checkpoint-recovery" || root.state !== "published") {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_recovery_not_published");
    }
    // The exact publication witness prevents a replay from resurrecting local
    // records retired after recovery. It owns no delivery or artifact bytes.
    captures.put({ ...root, source_refs: [], artifact_id: null, record_count: 0 });
    tx.objectStore("capture_records").delete(globalThis.IDBKeyRange.bound([rootId], [rootId, []], false, true));
    await done;
  }

  async appendCheckpointRecoveryRecord(rootId, collection, value) {
    if (!["jobs", "queue", "revisions"].includes(collection) || !value || typeof value !== "object" ||
        Array.isArray(value) || typeof value.id !== "string" || !value.id) throw new Error("checkpoint_recovery_record_invalid");
    const db = await this.database(); const tx = db.transaction(["captures", "capture_records"], "readwrite");
    const done = transactionDone(tx); const captures = tx.objectStore("captures");
    const root = await requestResult(captures.get(rootId));
    if (root?.kind !== "checkpoint-recovery" || root.state !== "receiving" || value.provider !== root.provider) {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_recovery_owner_mismatch");
    }
    const records = tx.objectStore("capture_records");
    const key = `${collection}:${value.id}`;
    const prior = await requestResult(records.get([rootId, key]));
    if (prior && JSON.stringify(prior.checkpoint_value) !== JSON.stringify(value)) {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_recovery_record_conflict");
    }
    if (!prior) {
      records.add({ capture_id: rootId, key, record_kind: `checkpoint:${collection}`, order_time: 0,
        order_id: "", occurrence: root.record_count++, checkpoint_value: value });
      captures.put(root);
    }
    await done;
  }

  async finishCheckpointRecovery(rootId, shape) {
    if (shape.version !== BACKFILL_RECOVERY_CHECKPOINT_VERSION ||
        !["jobs", "queue", "revisions"].every((name) => shape[name] === "array")) throw new Error("checkpoint_recovery_shape_invalid");
    const db = await this.database(); const tx = db.transaction(["captures", "capture_records"], "readwrite");
    const done = transactionDone(tx); const captures = tx.objectStore("captures");
    const root = await requestResult(captures.get(rootId));
    if (root?.kind !== "checkpoint-recovery") { tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_recovery_owner_mismatch"); }
    let job = null;
    await visitCursor(tx.objectStore("capture_records").openCursor(globalThis.IDBKeyRange.bound([rootId], [rootId, []], false, true)), (cursor) => {
      if (cursor.value.record_kind !== "checkpoint:jobs") return;
      if (job) throw new Error("checkpoint_recovery_job_conflict");
      job = cursor.value.checkpoint_value;
    });
    if (!job || job.cutoff !== root.cutoff || (job.account_scope && job.account_scope !== root.account_scope)) {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_recovery_intent_mismatch");
    }
    captures.put({ ...root, state: "ready", job_id: job.id });
    await done;
  }

  async publishCheckpointRecovery(rootId) {
    const db = await this.database(); const tx = db.transaction(["captures", "capture_records", "jobs", "queue", "revisions"], "readwrite");
    const done = transactionDone(tx); const captures = tx.objectStore("captures");
    const root = await requestResult(captures.get(rootId));
    if (root?.kind !== "checkpoint-recovery" || root.state !== "ready") {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_recovery_not_ready");
    }
    let restored = 0;
    await visitCursor(tx.objectStore("capture_records").openCursor(globalThis.IDBKeyRange.bound([rootId], [rootId, []], false, true)), async (cursor) => {
      const record = cursor.value; const collection = record.record_kind.slice("checkpoint:".length);
      const value = record.checkpoint_value; const target = tx.objectStore(collection);
      const current = await requestResult(target.get(value.id));
      if (current?.provider && current.provider !== root.provider) throw new Error("checkpoint_recovery_provider_conflict");
      if (collection === "jobs") {
        if (current && current.account_scope !== root.account_scope) throw new Error("checkpoint_recovery_scope_unresolved");
        if (canAdvanceRecovery(current, root)) {
          target.put(recoveredRow(jobRow({ ...recoveryCheckpointJob(value, true), account_scope: root.account_scope, receiver_job_id: root.remote_job_id }), root));
          restored += 1;
        }
      } else if (collection === "queue") {
        if (value.job_id !== root.job_id) throw new Error("checkpoint_recovery_job_conflict");
        // Locally acquired custody is stronger than a metadata-only remote
        // snapshot. Never replace its real body or resume operator work here.
        const acquired = current && (current.body_ref || current.record_ref || current.raw_acquisition_ref || current.capture_bundle_ref || current.envelope);
        if (!acquired && canAdvanceRecovery(current, root)) target.put(recoveredRow(queueRow(recoveryRequiredItem(convertAcquisitionRefusal(value))), root));
      } else if (collection === "revisions") {
        if (canAdvanceRecovery(current, root)) target.put(recoveredRow(value, root));
      } else throw new Error("checkpoint_recovery_collection_invalid");
    });
    captures.put({ ...root, state: "published" });
    await done; return { restored, reason: "receiver_authority_reconciled" };
  }

  async *unconvertedBackfillBodies() {
    const db = await this.database(); let after = null;
    for (;;) {
      const cursor = await requestResult(db.transaction("queue", "readonly").objectStore("queue")
        .openCursor(after === null ? null : globalThis.IDBKeyRange.lowerBound(after, true)));
      if (!cursor) return;
      after = cursor.key;
      if (cursor.value.delivery_kind !== "foreground" && cursor.value.envelope && !cursor.value.envelope.capture_body_ref) yield cursor.value;
    }
  }

  async localCheckpointRecordPublished(collection, original) {
    const marker = await this.getCapture(`checkpoint-conversion:${collection}:${original.id}:${recoveryValueDigest(original)}`);
    if (!marker) return false;
    if (marker.original_digest !== recoveryValueDigest(original)) throw new Error("checkpoint_conversion_source_conflict");
    return marker.state === "published";
  }

  async convertLocalCheckpointRecord(collection, original, replacement = original) {
    if (!["jobs", "queue", "revisions"].includes(collection) || !original || typeof original.id !== "string" || !original.id ||
        replacement.id !== original.id || replacement.provider !== original.provider) {
      throw new Error("checkpoint_conversion_record_invalid");
    }
    const db = await this.database();
    const tx = db.transaction([collection, "captures"], "readwrite"); const done = transactionDone(tx);
    const markerId = `checkpoint-conversion:${collection}:${original.id}:${recoveryValueDigest(original)}`;
    const captures = tx.objectStore("captures"); const target = tx.objectStore(collection);
    const marker = await requestResult(captures.get(markerId));
    const originalDigest = recoveryValueDigest(original);
    if (marker) {
      if (marker.original_digest !== originalDigest) {
        tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_conversion_source_conflict");
      }
      await done; return;
    }
    const current = await requestResult(target.get(original.id));
    const replaceOriginal = collection === "queue" && current && replacement.body_ref && recoveryValueDigest(queueRow(original)) === recoveryValueDigest(current);
    if (collection === "queue" && current && replacement.body_ref && current.body_ref !== replacement.body_ref && !replaceOriginal) {
      tx.abort(); await done.catch(() => undefined); throw new Error("checkpoint_conversion_custody_conflict");
    }
    if (!current || replaceOriginal) {
      const converted = convertAcquisitionRefusal(replacement);
      const value = collection === "jobs" ? recoveryCheckpointJob(checkpointJob(replacement), false, true)
        : collection === "queue" ? queueRow(converted.body_ref
          ? { ...converted, state: converted.state === "leased" ? converted.resume_state || "captured_waiting_receiver" : converted.state,
            lease_owner: null, lease_expires_at_ms: null }
          : recoveryRequiredItem(converted)) : replacement;
      target.put(collection === "jobs" ? jobRow(value) : value);
    }
    captures.put({ id: markerId, kind: "checkpoint-conversion", original_digest: originalDigest, state: "published" });
    await done;
  }

  async recoverExpiredLeases(jobId, nowMs) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    const store = tx.objectStore("queue");
    let recovered = 0;
    await visitCursor(store.index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)), (cursor) => {
      const item = cursor.value;
      if (item.job_id === jobId && item.state === "leased" && item.lease_expires_at_ms <= nowMs) {
        store.put({ ...item, state: item.resume_state || "eligible", lease_owner: null, lease_expires_at_ms: null });
        recovered += 1;
      }
    });
    await transactionDone(tx);
    return recovered;
  }

  async acquireNextLease(jobId, owner, nowMs, leaseMs, receiverOnly = false) {
    const db = await this.database();
    const tx = db.transaction("queue", "readwrite");
    const store = tx.objectStore("queue");
    let targetProvider = null; let candidate = null;
    await visitCursor(store.index("job_id").openCursor(globalThis.IDBKeyRange.only(jobId)), (cursor) => {
      const item = cursor.value;
      targetProvider ||= item.provider;
      const eligible = receiverOnly ? item.state === "captured_waiting_receiver" : ["eligible", "retry_wait"].includes(item.state);
      if (eligible && (!item.lease_owner || item.lease_expires_at_ms <= nowMs) && (item.next_eligible_at_ms || 0) <= nowMs &&
          (!candidate || (item.next_eligible_at_ms || 0) < (candidate.next_eligible_at_ms || 0))) candidate = item;
    });
    let providerBusy = false;
    if (targetProvider) await visitCursor(store.index("provider_state").openCursor(globalThis.IDBKeyRange.only([targetProvider, "leased"])), (cursor) => {
      if (cursor.value.lease_expires_at_ms > nowMs) providerBusy = true;
    });
    if (providerBusy) {
      await transactionDone(tx);
      return null;
    }
    if (candidate) {
      const leased = {
        ...candidate,
        resume_state: candidate.state,
        state: "leased",
        lease_owner: owner,
        lease_expires_at_ms: nowMs + leaseMs,
      };
      store.put(queueRow(leased));
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

  async getJob(id) { return globalThis.structuredClone(this.jobs.get(id)); }
  async putJob(job) { this.jobs.set(job.id, globalThis.structuredClone(job)); return job; }
  async putCheckpointOutcome(jobId, outcome) {
    const current = await this.getJob(jobId);
    if (!current) throw new Error(`backfill_job_not_found:${jobId}`);
    this.jobs.set(jobId, { ...current, recovery_checkpoint_outcome: globalThis.structuredClone(outcome) });
  }
  async createJob(job) {
    const existing = [...this.jobs.values()].find(
      (candidate) => candidate.provider === job.provider && ["running", "paused"].includes(candidate.status),
    );
    if (existing) throw new Error(`backfill_job_already_active:${job.provider}:${existing.id}`);
    this.jobs.set(job.id, globalThis.structuredClone(job));
    return job;
  }
  async acquireJobExecution(jobId, owner, nowMs, leaseMs) {
    const job = this.jobs.get(jobId);
    if (!job || job.status !== "running" || (job.execution_owner && job.execution_expires_at_ms > nowMs)) return null;
    const leased = { ...job, execution_owner: owner, execution_expires_at_ms: nowMs + leaseMs, execution_generation: (job.execution_generation || 0) + 1 };
    this.jobs.set(jobId, globalThis.structuredClone(leased));
    return globalThis.structuredClone(leased);
  }
  async renewJobExecution(jobId, owner, generation, nowMs, leaseMs) {
    const current = await this.assertJobExecution(jobId, owner, generation);
    this.jobs.set(jobId, { ...current, execution_expires_at_ms: nowMs + leaseMs });
    for (const item of this.queue.values()) if (item.job_id === jobId && item.state === "leased" && item.lease_owner === owner) {
      this.queue.set(item.id, { ...item, lease_expires_at_ms: nowMs + leaseMs });
    }
  }
  async assertJobExecution(jobId, owner, generation) {
    const job = this.jobs.get(jobId);
    if (!job || job.execution_owner !== owner || job.execution_generation !== generation || job.status !== "running") throw new Error(`stale_backfill_execution:${jobId}`);
    return globalThis.structuredClone(job);
  }
  async putJobCas(job, owner, generation) {
    const current = this.jobs.get(job.id);
    if (!current || current.execution_owner !== owner || current.execution_generation !== generation) throw new Error(`stale_backfill_execution:${job.id}`);
    const next = { ...job, execution_owner: owner, execution_expires_at_ms: current.execution_expires_at_ms, execution_generation: generation };
    this.jobs.set(job.id, globalThis.structuredClone(next));
    return next;
  }
  async reserveProviderRequests(jobId, owner, generation, count, dailyKeyValue, nextRequestAtMs) {
    const current = await this.assertJobExecution(jobId, owner, generation);
    const used = current.daily_key === dailyKeyValue ? current.daily_requests || 0 : 0;
    if (used + count > current.policy.maxDailyRequests) return null;
    const next = { ...current, daily_key: dailyKeyValue, daily_requests: used + count, next_request_at_ms: nextRequestAtMs };
    this.jobs.set(jobId, globalThis.structuredClone(next));
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
    this.jobs.set(jobId, globalThis.structuredClone(next));
    if (status === "running" && resumeQueueAtMs !== null) {
      for (const item of this.queue.values()) {
        if (item.job_id === jobId && ["auth_required"].includes(item.state)) {
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
        if (item.job_id === jobId && !["complete", "unchanged", "superseded", "no_turns"].includes(item.state)) {
          this.queue.set(item.id, { ...item, state: "cancelled", lease_owner: null, lease_expires_at_ms: null });
        }
      }
    }
    return next;
  }
  async jobPage({ cursor = null, pageSize = 25, activeOnly = false } = {}) {
    if (!Number.isSafeInteger(pageSize) || pageSize < 1 || (cursor !== null && typeof cursor !== "string")) throw new Error("backfill_page_invalid");
    const jobs = []; let total = 0;
    for (const job of this.jobs.values()) if (!activeOnly || !["complete", "completed", "cancelled", "failed"].includes(job.status)) total += 1;
    let after = cursor;
    for (let index = 0; index <= pageSize; index++) {
      let selected = null;
      for (const job of this.jobs.values()) {
        if (activeOnly && ["complete", "completed", "cancelled", "failed"].includes(job.status)) continue;
        if ((after === null || job.id > after) && (!selected || job.id < selected.id)) selected = job;
      }
      if (!selected) return { jobs, total, cursor: null, has_more: false };
      if (index === pageSize) return { jobs, total, cursor: after, has_more: true };
      jobs.push(globalThis.structuredClone(selected)); after = selected.id;
    }
  }
  async getQueue(id) { return globalThis.structuredClone(this.queue.get(id)); }
  async putQueue(item) { this.queue.set(item.id, globalThis.structuredClone(item)); return item; }
  async putQueueCas(jobId, owner, generation, item) {
    await this.assertJobExecution(jobId, owner, generation);
    this.queue.set(item.id, globalThis.structuredClone(item));
    return item;
  }
  async upsertDiscoveredCas(jobId, owner, generation, item) {
    await this.assertJobExecution(jobId, owner, generation);
    return this.upsertDiscovered(item);
  }
  async finalizeCaptureCas(job, owner, generation, item, revision, lastAck) {
    await this.assertJobExecution(job.id, owner, generation);
    this.queue.set(item.id, globalThis.structuredClone(item));
    if (revision) this.revisions.set(revision.id, globalThis.structuredClone(revision));
    const current = this.jobs.get(job.id);
    const next = { ...current, last_ack: globalThis.structuredClone(lastAck), last_error: null };
    this.jobs.set(job.id, next);
    return globalThis.structuredClone(next);
  }
  async upsertDiscovered(item) {
    const found = [...this.queue.values()].find(
      (candidate) => candidate.job_id === item.job_id && candidate.provider === item.provider && candidate.native_id === item.native_id,
    );
    if (!found) this.queue.set(item.id, globalThis.structuredClone(item));
    return globalThis.structuredClone(found || item);
  }
  async queueSummary(jobId, nowMs = 0) {
    const summary = emptyQueueSummary();
    for (const item of this.queue.values()) if (item.job_id === jobId) includeQueueSummary(summary, item, nowMs);
    return summary;
  }
  async *jobRecords() { for (const job of this.jobs.values()) yield globalThis.structuredClone(job); }
  async queuePage(jobId, { cursor = null, pageSize = 50 } = {}) {
    if (!Number.isSafeInteger(pageSize) || pageSize < 1 || (cursor !== null && typeof cursor !== "string")) throw new Error("backfill_page_invalid");
    const items = []; let total = 0;
    for (const item of this.queue.values()) if (item.job_id === jobId) total += 1;
    let after = cursor;
    for (let index = 0; index <= pageSize; index++) {
      let selected = null;
      for (const item of this.queue.values()) if (item.job_id === jobId && (after === null || item.id > after) && (!selected || item.id < selected.id)) selected = item;
      if (!selected) return { items, total, cursor: null, has_more: false };
      if (index === pageSize) return { items, total, cursor: after, has_more: true };
      items.push(globalThis.structuredClone(selected)); after = selected.id;
    }
  }
  async getRevision(provider, nativeId) { return globalThis.structuredClone(this.revisions.get(`${provider}:${nativeId}`)); }
  async putRevision(revision) { this.revisions.set(revision.id, globalThis.structuredClone(revision)); return revision; }
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
    this.queue.set(leased.id, globalThis.structuredClone(leased));
    return globalThis.structuredClone(leased);
  }
}

function emptyQueueSummary() {
  return { progress: { total: 0, eligible: 0, complete: 0, superseded: 0, no_turns: 0, retry: 0, error: 0, operator_action: 0 },
    finished: true, recoveryRequired: false, receiverDue: Infinity, providerDue: Infinity };
}
function includeQueueSummary(summary, item, nowMs) {
  const buckets = summary.progress; buckets.total += 1;
  if (["discovered", "eligible", "leased"].includes(item.state)) buckets.eligible += 1;
  if (["complete", "unchanged"].includes(item.state)) buckets.complete += 1;
  if (item.state === "superseded") buckets.superseded += 1;
  if (item.state === "no_turns") buckets.no_turns += 1;
  if (["retry_wait", "captured_waiting_receiver"].includes(item.state)) buckets.retry += 1;
  if (["auth_required", "recovery_required"].includes(item.state)) buckets.operator_action += 1;
  if (item.state === "failed") buckets.error += 1;
  if (!TERMINAL_QUEUE_STATES.has(item.state)) summary.finished = false;
  if (item.state === "recovery_required") summary.recoveryRequired = true;
  if (item.state === "captured_waiting_receiver") summary.receiverDue = Math.min(summary.receiverDue, item.next_eligible_at_ms || nowMs);
  if (["eligible", "retry_wait", "leased"].includes(item.state)) summary.providerDue = Math.min(summary.providerDue, item.next_eligible_at_ms || 0);
}
