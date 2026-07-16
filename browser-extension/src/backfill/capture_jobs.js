// Receiver-authoritative CaptureJob client.  chrome.storage is a cache of
// opaque ids only; account handles are reduced locally before any request.

export const CAPTURE_JOB_PROTOCOL = 1;

export function canonicalJson(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new Error("capture_job_non_canonical_number");
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (!value || typeof value !== "object") throw new Error("capture_job_non_canonical_json");
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
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

export async function deriveAccountScope(token, provider, accountHandle) {
  if (!/^[a-z][a-z0-9_.-]{0,63}$/.test(provider) || !String(accountHandle).trim()) throw new Error("capture_job_invalid_scope_input");
  return `h1:${await hmac(token, `polylogue:account-scope:v1\0${provider}\0${String(accountHandle).normalize("NFKC").trim()}`)}`;
}

async function intentKey(token, provider, accountScope, locator) {
  return `i1:${await hmac(token, `polylogue:capture-intent:v1\0${provider}\0${accountScope}\0${canonicalJson(locator)}`)}`;
}

export class CaptureJobClient {
  constructor({ baseUrl, token, cache, fetchImpl = fetch }) {
    this.baseUrl = baseUrl;
    this.token = token;
    this.cache = cache;
    this.fetchImpl = fetchImpl;
  }

  async request(method, path, body) {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.token}`, "X-Polylogue-Client-Protocol": String(CAPTURE_JOB_PROTOCOL) },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const payload = await response.json();
    if (!response.ok) {
      const error = new Error(payload?.error?.code || "capture_job_request_failed");
      error.code = payload?.error?.code;
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  async recoverOrCreate({ provider, accountHandle, locator, intentPayload, sessionId }) {
    const account_scope = await deriveAccountScope(this.token, provider, accountHandle);
    const intent_key = await intentKey(this.token, provider, account_scope, locator);
    const found = await this.request("POST", "/v1/capture-jobs/discover", { provider, account_scope, intent_key });
    const intent = { schema_version: 1, version: 1, intent_key, kind: "backfill-ledger", payload: intentPayload, digest: await digest(intentPayload) };
    const job = found.jobs.length ? found.jobs[0] : (await this.request("POST", "/v1/capture-jobs", { request_id: crypto.randomUUID(), provider, account_scope, intent })).job;
    if (found.jobs.length > 1) throw new Error("capture_job_ambiguous_adoption");
    const key = `capture-job-cache:v1:${intent_key}`;
    const cached = (await this.cache.get({ [key]: null }))[key];
    const request_id = cached?.job_id === job.job_id ? cached.request_id : crypto.randomUUID();
    const adopted = await this.request("POST", `/v1/capture-jobs/${job.job_id}/adopt`, {
      provider, account_scope, request_id, session_id: sessionId, expected_revision: job.revision,
      expected_lease_generation: job.lease_generation, lease_ttl_seconds: 120,
    });
    await this.cache.set({ [key]: { job_id: job.job_id, request_id } });
    return { ...adopted, account_scope, intent_key };
  }

  async discoverRecovery(provider, accountHandle) {
    const account_scope = await deriveAccountScope(this.token, provider, accountHandle);
    const result = await this.request("POST", "/v1/capture-jobs/discover", { provider, account_scope });
    return result.jobs.filter((job) => job.checkpoint?.payload).sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
  }

  async checkpoint(adopted, checkpointPayload, sequence) {
    return this.request("PUT", `/v1/capture-jobs/${adopted.job.job_id}/checkpoint`, {
      provider: adopted.job.provider, account_scope: adopted.account_scope, request_id: crypto.randomUUID(),
      expected_revision: adopted.job.revision, lease_id: adopted.lease.lease_id, generation: adopted.lease.generation,
      proof: adopted.lease.proof, checkpoint: { sequence, payload: checkpointPayload, digest: await digest(checkpointPayload) },
    });
  }
}
