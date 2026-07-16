// Receiver-authoritative CaptureJob client.  chrome.storage is a cache of
// opaque ids only; account handles are reduced locally before any request.

export const CAPTURE_JOB_PROTOCOL = 1;
export const CAPTURE_JOB_REQUEST_TIMEOUT_MS = 20_000;

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
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0));
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
  constructor({ baseUrl, token, cache, fetchImpl = fetch, requestTimeoutMs = CAPTURE_JOB_REQUEST_TIMEOUT_MS }) {
    this.baseUrl = baseUrl;
    this.token = token;
    this.cache = cache;
    this.fetchImpl = fetchImpl;
    this.requestTimeoutMs = requestTimeoutMs;
    this.scopeNamespacePromise = null;
  }

  async request(method, path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.requestTimeoutMs);
    let response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.token}`, "X-Polylogue-Client-Protocol": String(CAPTURE_JOB_PROTOCOL) },
        ...(method === "GET" ? {} : { body: JSON.stringify(body) }),
        cache: "no-store",
        signal: controller.signal,
      });
    } catch (error) {
      if (controller.signal.aborted) throw new Error("capture_job_request_timeout");
      throw error;
    } finally {
      clearTimeout(timeout);
    }
    const payload = await response.json();
    if (!response.ok) {
      const error = new Error(payload?.error?.code || "capture_job_request_failed");
      error.code = payload?.error?.code;
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  async scopeNamespace() {
    if (!this.scopeNamespacePromise) {
      this.scopeNamespacePromise = this.request("GET", "/v1/capture-jobs/capabilities").then((payload) => {
        if (payload?.schema !== "polylogue.capture-jobs.capabilities.v1"
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
    const account_scope = await deriveAccountScope(scopeNamespace, provider, accountHandle);
    const intent_key = await intentKey(scopeNamespace, provider, account_scope, locator);
    const found = await this.request("POST", "/v1/capture-jobs/discover", { provider, account_scope, intent_key });
    const intent = { schema_version: 1, version: 1, intent_key, kind: "backfill-ledger", payload: intentPayload, digest: await digest(intentPayload) };
    const job = found.jobs.length ? found.jobs[0] : (await this.request("POST", "/v1/capture-jobs", { request_id: crypto.randomUUID(), provider, account_scope, intent })).job;
    if (found.jobs.length > 1) throw new Error("capture_job_ambiguous_adoption");
    return this.adoptExisting(job, account_scope, sessionId);
  }

  async adoptExisting(job, accountScope, sessionId) {
    const key = `capture-job-cache:v1:${job.intent_key}`;
    const cached = (await this.cache.get({ [key]: null }))[key];
    const request_id = cached?.job_id === job.job_id ? cached.request_id : crypto.randomUUID();
    const adopted = await this.request("POST", `/v1/capture-jobs/${job.job_id}/adopt`, {
      provider: job.provider, account_scope: accountScope, request_id, session_id: sessionId, expected_revision: job.revision,
      expected_lease_generation: job.lease_generation, lease_ttl_seconds: 120,
    });
    await this.cache.set({ [key]: { job_id: job.job_id, request_id } });
    return { ...adopted, account_scope: accountScope, intent_key: job.intent_key };
  }

  async discoverRecovery(provider, accountHandle, sessionId) {
    const account_scope = await deriveAccountScope(await this.scopeNamespace(), provider, accountHandle);
    const result = await this.request("POST", "/v1/capture-jobs/discover", { provider, account_scope });
    const jobs = result.jobs
      .filter((job) => job.checkpoint?.payload)
      .sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
    return Promise.all(jobs.map(async (job) => {
      try {
        return {
          ...await this.adoptExisting(job, account_scope, sessionId),
          recovery_updated_at: job.checkpoint_updated_at || job.updated_at,
        };
      } catch (error) {
        // A wiped profile cannot prove ownership of the destroyed profile's
        // still-live lease. Retain its scoped checkpoint for visible recovery;
        // later status/checkpoint attempts retry adoption after lease expiry.
        if (error?.code === "lease_held") {
          return {
            job, account_scope, intent_key: job.intent_key,
            recovery_state: "lease_held", recovery_updated_at: job.checkpoint_updated_at || job.updated_at,
          };
        }
        throw error;
      }
    }));
  }

  async update(adopted, retry, leaseTtlSeconds = 120) {
    const result = await this.request("POST", `/v1/capture-jobs/${adopted.job.job_id}/update`, {
      provider: adopted.job.provider,
      account_scope: adopted.account_scope,
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

  async checkpoint(adopted, checkpointPayload) {
    const checkpointDigest = await digest(checkpointPayload);
    if (adopted.job.checkpoint_digest === checkpointDigest) {
      return { job: adopted.job, receipt: null, duplicate: true };
    }
    const sequence = Number.isSafeInteger(adopted.job.checkpoint_sequence)
      ? adopted.job.checkpoint_sequence + 1
      : 0;
    return this.request("PUT", `/v1/capture-jobs/${adopted.job.job_id}/checkpoint`, {
      provider: adopted.job.provider, account_scope: adopted.account_scope, request_id: crypto.randomUUID(),
      expected_revision: adopted.job.revision, lease_id: adopted.lease.lease_id, generation: adopted.lease.generation,
      proof: adopted.lease.proof, checkpoint: { sequence, payload: checkpointPayload, digest: checkpointDigest },
    });
  }
}
