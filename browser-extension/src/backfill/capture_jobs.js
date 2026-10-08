// Receiver-authoritative CaptureJob client.  chrome.storage is a cache of
// opaque ids only; account handles are reduced locally before any request.
import JSONParser from "../vendor/streamparser-json/jsonparser.js";

export const CAPTURE_JOB_PROTOCOL = 2;

function compareCanonicalKeys(left, right) {
  // Python's CAPTURE encoder orders Unicode scalar values, whereas JavaScript
  // string comparison orders UTF-16 code units. Keep supplementary keys equal
  // to the receiver's canonical byte contract.
  let leftIndex = 0; let rightIndex = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    const leftPoint = left.codePointAt(leftIndex); const rightPoint = right.codePointAt(rightIndex);
    if (leftPoint !== rightPoint) return leftPoint - rightPoint;
    leftIndex += leftPoint > 0xFFFF ? 2 : 1; rightIndex += rightPoint > 0xFFFF ? 2 : 1;
  }
  return leftIndex < left.length ? 1 : rightIndex < right.length ? -1 : 0;
}

export function canonicalJson(value) {
  if (value === null || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value.normalize("NFC"));
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new Error("capture_job_non_canonical_number");
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (!value || typeof value !== "object") throw new Error("capture_job_non_canonical_json");
  const entries = Object.keys(value)
    .map((key) => [key.normalize("NFC"), value[key]])
    .sort(([left], [right]) => compareCanonicalKeys(left, right));
  if (entries.some(([key], index) => index > 0 && entries[index - 1][0] === key)) {
    throw new Error("capture_job_non_canonical_key_collision");
  }
  return `{${entries.map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`).join(",")}}`;
}

async function digest(value) {
  const bytes = new TextEncoder().encode(canonicalJson(value));
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return `sha256:${[...hash].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

async function hmac(token, message) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(token), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const bytes = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message)));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export async function deriveAccountScope(scopeNamespace, provider, accountHandle) {
  if (!/^[a-z][a-z0-9_.-]{0,63}$/.test(provider) || !String(accountHandle).trim()) throw new Error("capture_job_invalid_scope_input");
  return `h1:${await hmac(scopeNamespace, `polylogue:account-scope:v1\0${provider}\0${String(accountHandle).normalize("NFKC").trim()}`)}`;
}

async function intentKey(scopeNamespace, provider, accountScope, locator) {
  return `i1:${await hmac(scopeNamespace, `polylogue:capture-intent:v1\0${provider}\0${accountScope}\0${canonicalJson(locator)}`)}`;
}

export class CaptureJobClient {
  constructor({ baseUrl, token, cache, fetchImpl = globalThis.fetch?.bind(globalThis) }) {
    this.baseUrl = baseUrl;
    this.token = token;
    this.cache = cache;
    this.fetchImpl = fetchImpl;
    this.scopeNamespacePromise = null;
  }

  async request(method, path, body, { headers = {}, signal = null, raw = false, stream = false, receipt = false } = {}) {
      const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.token}`, "X-Polylogue-Client-Protocol": String(CAPTURE_JOB_PROTOCOL), ...headers },
        ...(method === "GET" ? {} : { body: raw ? body : JSON.stringify(body) }),
        cache: "no-store",
        ...(signal ? { signal } : {}),
      });
      if (stream && response.ok) return response;
      const payload = await response.json();
      if (!response.ok) {
        const error = new Error(payload?.error?.code || "capture_job_request_failed");
        error.code = payload?.error?.code;
        error.status = response.status;
        throw error;
      }
      if (receipt) return { ...payload, receiver_request_id: response.headers.get("X-Request-ID") };
      return payload;
  }

  async scopeNamespace() {
    if (!this.scopeNamespacePromise) {
      this.scopeNamespacePromise = this.request("GET", "/v1/capture-jobs/capabilities").then((payload) => {
        if (payload?.schema !== "polylogue.capture-jobs.capabilities.v1"
          || payload.checkpoint_transport !== "canonical-artifact-v1"
          || typeof payload.scope_namespace !== "string"
          || !payload.scope_namespace.startsWith("cjs1:")) {
          throw new Error("capture_job_capabilities_invalid");
        }
        return payload.scope_namespace;
      });
    }
    return this.scopeNamespacePromise;
  }

  async recoverOrCreate({ provider, accountHandle, locator, intentPayload, sessionId }) {
    const scopeNamespace = await this.scopeNamespace();
    const scope = { kind: "account", key: await deriveAccountScope(scopeNamespace, provider, accountHandle) };
    const intent_key = await intentKey(scopeNamespace, provider, scope.key, locator);
    const found = await this.request("POST", "/v1/capture-jobs/discover", { provider, scope, intent_key });
    const intent = { schema_version: 1, version: 1, intent_key, kind: "backfill-ledger", payload: intentPayload, digest: await digest(intentPayload) };
    const job = found.jobs.length ? found.jobs[0] : (await this.request("POST", "/v1/capture-jobs", { request_id: crypto.randomUUID(), provider, scope, intent })).job;
    if (found.jobs.length > 1) throw new Error("capture_job_ambiguous_adoption");
    return this.adoptExisting(job, scope, sessionId);
  }

  async recoverInvocation({ provider, creationToken, binding, sessionId }) {
    const namespace = await this.scopeNamespace();
    const intent_key = await intentKey(namespace, provider, `invocation:${creationToken}`, { acquisition: creationToken });
    const intent = { schema_version: 1, version: 1, intent_key, kind: "native-acquisition",
      payload: binding, digest: await digest(binding) };
    const created = await this.request("POST", "/v1/capture-jobs", { provider,
      scope: { kind: "invocation", creation_token: creationToken, binding }, intent });
    return this.adoptExisting(created.job, created.scope, sessionId);
  }

  nativeDescriptor(adopted, values) {
    return { provider: adopted.job.provider, scope: adopted.scope, client_protocol: CAPTURE_JOB_PROTOCOL,
      request_id: crypto.randomUUID(), expected_revision: adopted.job.revision,
      lease_id: adopted.lease.lease_id, generation: adopted.lease.generation, proof: adopted.lease.proof, ...values };
  }

  async beginNative(adopted, acquisitionId, binding, memberNames, signal) {
    return this.request("POST", `/v1/capture-jobs/${adopted.job.job_id}/native/begin`,
      this.nativeDescriptor(adopted, { acquisition_id: acquisitionId, binding, member_names: memberNames }), { signal });
  }

  async nativeMember(adopted, acquisitionId, memberName, file, metadata, sha256, signal) {
    const descriptor = this.nativeDescriptor(adopted, { acquisition_id: acquisitionId, member_name: memberName,
      sha256, size_bytes: file.size, metadata });
    return this.request("PUT", `/v1/capture-jobs/${adopted.job.job_id}/native/member`, file,
      { raw: true, signal, headers: { "X-Polylogue-Native": JSON.stringify(descriptor) } });
  }

  async nativeRevision(provider, nativeId, members) {
    return digest({ provider, native_id: nativeId, members });
  }

  async prepareNative(adopted, acquisitionId, provenance, providerMeta, signal) {
    return this.request("POST", `/v1/capture-jobs/${adopted.job.job_id}/native/prepare`,
      this.nativeDescriptor(adopted, { acquisition_id: acquisitionId, provenance, provider_meta: providerMeta }), { signal });
  }

  async *nativePlan(adopted, acquisitionId, sessionId, signal) {
    let after = -1;
    do {
      await this.refreshNativeOwner(adopted, sessionId, signal);
      const page = await this.request("POST", `/v1/capture-jobs/${adopted.job.job_id}/native/plan`,
        this.nativeDescriptor(adopted, { acquisition_id: acquisitionId, after }), { signal });
      for (const asset of page.assets) yield { ...asset, plan_digest: page.plan_digest };
      after = page.after;
    } while (after !== null);
  }

  async nativeAsset(adopted, acquisitionId, asset, outcome, file, sha256, signal) {
    const descriptor = this.nativeDescriptor(adopted, { acquisition_id: acquisitionId,
      ordinal: asset.ordinal, descriptor_digest: asset.descriptor_digest, plan_digest: asset.plan_digest,
      outcome, ...(file ? { sha256, size_bytes: file.size } : {}) });
    return this.request(file ? "PUT" : "POST", `/v1/capture-jobs/${adopted.job.job_id}/native/asset`,
      file || descriptor, { signal, ...(file ? { raw: true, headers: { "X-Polylogue-Native": JSON.stringify(descriptor) } } : {}) });
  }

  async finalizeNative(adopted, acquisitionId, planDigest, signal) {
    return this.request("POST", `/v1/capture-jobs/${adopted.job.job_id}/native/finalize`,
      this.nativeDescriptor(adopted, { acquisition_id: acquisitionId, plan_digest: planDigest }), { signal });
  }

  async publishNative(adopted, acquisitionId, planDigest, sha256, signal) {
    return this.request("POST", `/v1/capture-jobs/${adopted.job.job_id}/native/publish`,
      this.nativeDescriptor(adopted, { acquisition_id: acquisitionId, plan_digest: planDigest, sha256 }), { signal, receipt: true });
  }

  async refreshNativeOwner(adopted, sessionId, signal) {
    const query = new URLSearchParams({ provider: adopted.job.provider, scope: JSON.stringify(adopted.scope), client_protocol: String(CAPTURE_JOB_PROTOCOL) });
    const current = await this.request("GET", `/v1/capture-jobs/${adopted.job.job_id}?${query}`, null, { signal });
    if (current.job.lease_generation !== adopted.lease.generation) throw new Error("lease_replaced");
    Object.assign(adopted, await this.adoptExisting(current.job, adopted.scope, sessionId));
    await this.nativeOwnerChanged?.(adopted);
    return adopted;
  }

  async adoptExisting(job, scope, sessionId) {
    const key = `capture-job-cache:v1:${job.intent_key}`;
    const request_id = await digest({ kind: "capture-job-adoption", job_id: job.job_id, session_id: sessionId });
    const adopted = await this.request("POST", `/v1/capture-jobs/${job.job_id}/adopt`, {
      provider: job.provider, scope, request_id, session_id: sessionId, expected_revision: job.revision,
      expected_lease_generation: job.lease_generation, lease_ttl_seconds: 120,
    });
    try {
      await this.cache?.set({ [key]: { job_id: job.job_id, request_id } });
    } catch {
      // Receiver adoption is authoritative. This opaque convenience cache may
      // disappear or reject writes without turning a valid lease into failure.
    }
    return { ...adopted, scope, intent_key: job.intent_key };
  }

  async *discoverRecovery(provider, accountHandle, sessionId) {
    const scope = { kind: "account", key: await deriveAccountScope(await this.scopeNamespace(), provider, accountHandle) };
    let cursor = null;
    do {
      const result = await this.request("POST", "/v1/capture-jobs/discover", { provider, scope, ...(cursor ? { cursor } : {}) });
      for (const job of result.jobs) {
        if (!job.checkpoint?.artifact_ref) continue;
        try {
          yield { ...await this.adoptExisting(job, scope, sessionId), recovery_updated_at: job.checkpoint_updated_at || job.updated_at };
        } catch (error) {
          // A destroyed profile cannot prove its still-live lease. Keep the
          // exact scoped checkpoint pending until ordinary lease expiry.
          if (error?.code !== "lease_held") throw error;
          yield { job, scope, intent_key: job.intent_key, recovery_state: "lease_held",
            recovery_updated_at: job.checkpoint_updated_at || job.updated_at };
        }
      }
      cursor = result.cursor || null;
    } while (cursor);
  }

  async update(adopted, retry, leaseTtlSeconds = 120) {
    const result = await this.request("POST", `/v1/capture-jobs/${adopted.job.job_id}/update`, {
      provider: adopted.job.provider,
      scope: adopted.scope,
      request_id: crypto.randomUUID(),
      expected_revision: adopted.job.revision,
      lease_id: adopted.lease.lease_id,
      generation: adopted.lease.generation,
      proof: adopted.lease.proof,
      retry,
      lease_ttl_seconds: leaseTtlSeconds,
    });
    return {
      ...adopted,
      job: result.job,
      lease: { ...adopted.lease, expires_at: result.job.lease_expires_at },
      update_receipt: result.receipt,
    };
  }

  checkpointDescriptor(adopted, sequence, checkpointDigest) {
    return {
      provider: adopted.job.provider, scope: adopted.scope, client_protocol: CAPTURE_JOB_PROTOCOL,
      request_id: crypto.randomUUID(), expected_revision: adopted.job.revision,
      lease_id: adopted.lease.lease_id, generation: adopted.lease.generation, proof: adopted.lease.proof,
      sequence, digest: checkpointDigest,
    };
  }

  async checkpoint(adopted, prepared, signal = null) {
    const checkpointDigest = prepared.digest;
    if (adopted.job.checkpoint_digest === checkpointDigest) {
      return { job: adopted.job, receipt: null, duplicate: true };
    }
    const sequence = Number.isSafeInteger(adopted.job.checkpoint_sequence)
      ? adopted.job.checkpoint_sequence + 1
      : 0;
    const descriptor = this.checkpointDescriptor(adopted, sequence, checkpointDigest);
    return this.request("PUT", `/v1/capture-jobs/${adopted.job.job_id}/checkpoint`, prepared.body,
      { raw: true, signal, headers: { "X-Polylogue-Checkpoint": JSON.stringify(descriptor) } });
  }

  async checkpointArtifact(adopted, signal = null) {
    const checkpoint = adopted.job.checkpoint;
    if (!checkpoint) return null;
    const descriptor = this.checkpointDescriptor(adopted, checkpoint.sequence, checkpoint.digest);
    return this.request("GET", `/v1/capture-jobs/${adopted.job.job_id}/checkpoint-artifacts/${checkpoint.artifact_ref}`,
      null, { stream: true, signal, headers: { "X-Polylogue-Checkpoint": JSON.stringify(descriptor) } });
  }

  async restoreCheckpoint(adopted, store, staging, signal = null) {
    if (!adopted.lease) return { restored: 0, reason: "lease_held" };
    const checkpoint = adopted.job.checkpoint;
    if (!checkpoint) return { restored: 0, reason: "checkpoint_unavailable" };
    const rootId = `checkpoint-recovery:${await digest([adopted.scope, adopted.job.job_id, checkpoint.sequence, checkpoint.digest])}`;
    const artifactId = staging.producerStageId({ checkpoint_recovery_id: rootId }, rootId);
    const root = await store.beginCheckpointRecovery({ id: rootId, provider: adopted.job.provider,
      account_scope: adopted.scope.key, remote_job_id: adopted.job.job_id, digest: checkpoint.digest,
      size_bytes: checkpoint.size_bytes, sequence: checkpoint.sequence,
      acknowledged_at: adopted.job.checkpoint_updated_at, cutoff: adopted.job.intent.payload.cutoff,
      artifact_id: artifactId, source_refs: [artifactId] });
    if (root.state === "published") {
      await store.retireCheckpointRecovery(rootId); await staging.discardUnreferenced(artifactId);
      return { restored: 0, reason: "already_published" };
    }
    if (root.state !== "ready") {
      let file;
      const meta = await staging.metadata(artifactId).catch((error) => {
        if (error?.name === "NotFoundError") return null;
        throw error;
      });
      if (meta?.state === "sealed") {
        if (`sha256:${meta.sha256}` !== checkpoint.digest || meta.bytes !== checkpoint.size_bytes) throw new Error("checkpoint_digest_mismatch");
        file = await staging.file(artifactId);
      } else {
        const response = await this.checkpointArtifact(adopted, signal);
        file = await staging.receiveCheckpoint({ ...root, id: rootId, artifact_id: artifactId }, response, signal);
      }
      const shape = {}; const pending = [];
      const parser = new JSONParser({ paths: ["$.version", "$.jobs", "$.jobs.*", "$.queue", "$.queue.*", "$.revisions", "$.revisions.*"],
        keepStack: false, stringBufferSize: 64 * 1024 });
      parser.onValue = ({ value, key, parent, stack }) => {
        if (stack.length === 1) shape[key] = key === "version" ? value : Array.isArray(value) ? "array" : "invalid";
        else if (stack.length === 2) {
          pending.push({ collection: stack[1].key, value });
          if (Array.isArray(parent)) parent.length = 0;
          else throw new Error("checkpoint_recovery_shape_invalid");
        }
      };
      const reader = file.stream().getReader();
      const abort = () => { void reader.cancel(signal.reason).catch(() => undefined); };
      signal?.addEventListener("abort", abort, { once: true });
      try {
        for (;;) {
          signal?.throwIfAborted();
          const { value, done } = await reader.read();
          if (done) break;
          for (let offset = 0; offset < value.length; offset += 48 * 1024) {
            parser.write(value.subarray(offset, offset + 48 * 1024));
            for (const record of pending.splice(0)) await store.appendCheckpointRecoveryRecord(rootId, record.collection, record.value);
          }
        }
        if (!parser.isEnded) parser.end();
        for (const record of pending.splice(0)) await store.appendCheckpointRecoveryRecord(rootId, record.collection, record.value);
        signal?.throwIfAborted();
        await store.finishCheckpointRecovery(rootId, shape);
      } catch (error) { await reader.cancel(error).catch(() => undefined); throw error; }
      finally { signal?.removeEventListener("abort", abort); reader.releaseLock(); }
    }
    const result = await store.publishCheckpointRecovery(rootId);
    await store.retireCheckpointRecovery(rootId); await staging.discardUnreferenced(artifactId);
    return result;
  }
}
