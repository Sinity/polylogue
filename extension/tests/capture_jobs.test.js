import { Blob as NativeBlob } from "node:buffer";
import { describe, expect, it, vi } from "vitest";
import { CaptureJobClient, canonicalJson, deriveAccountScope } from "../src/backfill/capture_jobs.js";

describe("CaptureJob extension recovery", () => {
  it("writes CAPTURE key order across supplementary Unicode keys", () => {
    expect(canonicalJson({ "\u{10000}": 2, "\uE000": 1 })).toBe('{"\uE000":1,"\u{10000}":2}');
  });
  it("binds the default fetch to the worker global", async () => {
    const previousFetch = globalThis.fetch;
    const fetchImpl = vi.fn(function fetchWithReceiver() {
      expect(this).toBe(globalThis);
      return Promise.resolve({
        ok: true,
        json: async () => ({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
          protocol_min: 2,
          protocol_max: 2,
          scope_namespace: "cjs1:worker-global",
        }),
      });
    });
    globalThis.fetch = fetchImpl;
    try {
      const client = new CaptureJobClient({
        baseUrl: "http://receiver",
        token: "receiver-token",
        cache: { get: vi.fn(), set: vi.fn() },
      });
      await expect(client.scopeNamespace()).resolves.toBe("cjs1:worker-global");
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expect(fetchImpl.mock.calls[0][1].headers.Authorization).toBe("Bearer receiver-token");
    } finally {
      globalThis.fetch = previousFetch;
    }
  });

  it("omits Authorization for an injected native transport", async () => {
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => ({ ok: true }) }));
    const client = new CaptureJobClient({ baseUrl: "http://receiver", fetchImpl });
    await client.request("POST", "/v1/capture-jobs", { neutral: true });
    expect(fetchImpl.mock.calls[0][1].headers.Authorization).toBeUndefined();
  });

  it("derives opaque scopes and rehydrates after chrome.storage cache loss", async () => {
    const cache = { values: {}, get: vi.fn(async (keys) => Object.fromEntries(Object.keys(keys).map((key) => [key, cache.values[key] ?? null]))), set: vi.fn(async (patch) => Object.assign(cache.values, patch)) };
    const scope = await deriveAccountScope("receiver-token", "chatgpt", "account@example.test");
    expect(scope).toMatch(/^h1:/);
    expect(scope).not.toContain("account");
    const responses = [
      { schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1", protocol_min: 2, protocol_max: 2, scope_namespace: "cjs1:receiver-namespace" },
      { jobs: [] },
      { job: { job_id: "receiver-job", provider: "chatgpt", intent_key: "intent", revision: 0, lease_generation: 0 } },
      { job: { job_id: "receiver-job", provider: "chatgpt", intent_key: "intent", revision: 1, lease_generation: 1 }, lease: { lease_id: "lease", generation: 1, proof: "proof" } },
    ];
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => responses.shift() }));
    const client = new CaptureJobClient({ baseUrl: "http://receiver", token: "receiver-token", cache, fetchImpl });
    const adopted = await client.recoverOrCreate({ provider: "chatgpt", accountHandle: "account@example.test", locator: { kind: "backfill", cutoff: "now" }, intentPayload: { cutoff: "now" }, sessionId: "new-profile" });
    expect(adopted.job.job_id).toBe("receiver-job");
    expect(cache.set).toHaveBeenCalled();
    expect(canonicalJson({ b: 2, a: 1 })).toBe('{"a":1,"b":2}');
    expect(canonicalJson({ "e\u0301": "e\u0301" })).toBe(canonicalJson({ "é": "é" }));
    const serializedRequests = fetchImpl.mock.calls.map(([, options]) => options.body).join("\n");
    expect(serializedRequests).not.toContain("account@example.test");
    expect(JSON.stringify(cache.values)).not.toContain("account@example.test");
  });

  it("keeps a receiver adoption valid when its opaque chrome.storage cache write fails", async () => {
    const requestIds = [];
    const cache = { set: vi.fn(async () => { throw new Error("storage_local_quota"); }) };
    const job = {
      job_id: "receiver-job",
      provider: "chatgpt",
      intent_key: "intent",
      revision: 0,
      lease_generation: 0,
    };
    const fetchImpl = vi.fn(async (_url, options) => {
      const body = JSON.parse(options.body);
      requestIds.push(body.request_id);
      return {
        ok: true,
        json: async () => ({
          job: { ...job, revision: 1, lease_generation: 1 },
          lease: { lease_id: "lease", generation: 1, proof: "proof" },
        }),
      };
    });
    const client = new CaptureJobClient({ baseUrl: "http://receiver", token: "receiver-token", cache, fetchImpl });

    const first = await client.adoptExisting(job, { kind: "account", key: "h1:" + "A".repeat(43) }, "replacement-profile");
    const second = await client.adoptExisting(job, { kind: "account", key: "h1:" + "A".repeat(43) }, "replacement-profile");

    expect(first.job.revision).toBe(1);
    expect(second.job.revision).toBe(1);
    expect(cache.set).toHaveBeenCalledTimes(2);
    expect(requestIds[0]).toBe(requestIds[1]);
  });

  it("consumes discovery pages and adopts one checkpoint at a time without dropping held leases", async () => {
    const events = []; const cursor = { created_at: "2026-01-01T00:00:00Z", job_id: "b" };
    const client = new CaptureJobClient({ baseUrl: "http://receiver", token: "synthetic-token", cache: {}, fetchImpl: async (url, options) => {
      const path = new globalThis.URL(url).pathname;
      if (path.endsWith("capabilities")) return { ok: true, json: async () => ({ schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1", scope_namespace: "cjs1:synthetic" }) };
      if (path.endsWith("discover")) {
        const body = JSON.parse(options.body); events.push(body.cursor ? "page-2" : "page-1");
        const ids = body.cursor ? ["c"] : ["a", "b"];
        if (body.cursor) expect(body.cursor).toEqual(cursor);
        return { ok: true, json: async () => ({ jobs: ids.map((id) => ({ job_id: id, provider: "chatgpt", intent_key: id, checkpoint: { artifact_ref: id } })), cursor: body.cursor ? null : cursor }) };
      }
      const id = path.split("/").at(-2); events.push(`adopt-${id}`);
      if (id === "b") return { ok: false, status: 409, json: async () => ({ error: { code: "lease_held" } }) };
      return { ok: true, json: async () => ({ job: { job_id: id }, lease: { lease_id: id } }) };
    } });
    const iterator = client.discoverRecovery("chatgpt", "synthetic-account", "synthetic-session");
    expect(events).toEqual([]);
    expect((await iterator.next()).value.job.job_id).toBe("a");
    expect(events).toEqual(["page-1", "adopt-a"]);
    expect((await iterator.next()).value).toMatchObject({ job: { job_id: "b" }, recovery_state: "lease_held" });
    expect(events).toEqual(["page-1", "adopt-a", "adopt-b"]);
    expect((await iterator.next()).value.job.job_id).toBe("c");
    expect((await iterator.next()).done).toBe(true);
    expect(events).toEqual(["page-1", "adopt-a", "adopt-b", "page-2", "adopt-c"]);
  });

  it("keeps discovery scope stable when the receiver bearer rotates", async () => {
    const discovered = [];
    const cache = { get: vi.fn(async () => ({})), set: vi.fn(async () => undefined) };
    for (const token of ["old-bearer", "rotated-bearer"]) {
      const fetchImpl = vi.fn(async (_url, options) => {
        if (options.method === "GET") {
          return {
            ok: true,
            json: async () => ({
              schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
              protocol_min: 2,
              protocol_max: 2,
              scope_namespace: "cjs1:stable-receiver-namespace",
            }),
          };
        }
        discovered.push({ token: options.headers.Authorization, body: JSON.parse(options.body) });
        return { ok: true, json: async () => ({ jobs: [] }) };
      });
      const client = new CaptureJobClient({ baseUrl: "http://receiver", token, cache, fetchImpl });
      for await (const entry of client.discoverRecovery("chatgpt", "same-account", "replacement-profile")) expect(entry).toBeDefined();
    }

    expect(discovered.map((request) => request.token)).toEqual(["Bearer old-bearer", "Bearer rotated-bearer"]);
    expect(discovered[0].body.scope.key).toBe(discovered[1].body.scope.key);
  });

  it("allows a slow progressing CaptureJob response without a deadline", async () => {
    vi.useFakeTimers();
    try {
      const fetchImpl = vi.fn(async () => ({ ok: true,
        json: async () => new Promise((resolve) => globalThis.setTimeout(() => resolve({ completed: true }), 48_000)),
      }));
      const client = new CaptureJobClient({ baseUrl: "http://receiver", token: "receiver-token", fetchImpl });
      const pending = client.request("GET", "/v1/capture-jobs/capabilities");
      await vi.advanceTimersByTimeAsync(48_000);
      await expect(pending).resolves.toEqual({ completed: true });
    } finally { vi.useRealTimers(); }
  });

  it("propagates explicit cancellation through a CaptureJob request", async () => {
    const controller = new globalThis.AbortController();
    const fetchImpl = vi.fn(async (_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true });
    }));
    const client = new CaptureJobClient({ baseUrl: "http://receiver", token: "receiver-token", fetchImpl });
    const pending = client.request("GET", "/v1/capture-jobs/capabilities", null, { signal: controller.signal });
    const cancelled = new Error("operator_cancelled");
    const result = expect(pending).rejects.toBe(cancelled);
    controller.abort(cancelled);
    await result;
  });

  it("renews the proven lease before checkpointing the returned revision", async () => {
    const cache = {
      values: {},
      get: vi.fn(async (keys) => Object.fromEntries(Object.keys(keys).map((key) => [key, null]))),
      set: vi.fn(async (patch) => Object.assign(cache.values, patch)),
    };
    const responses = [
      { schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1", protocol_min: 2, protocol_max: 2, scope_namespace: "cjs1:receiver-namespace" },
      { jobs: [] },
      { job: { job_id: "receiver-job", provider: "chatgpt", intent_key: "intent", revision: 0, lease_generation: 0 } },
      {
        job: { job_id: "receiver-job", provider: "chatgpt", intent_key: "intent", revision: 1, lease_generation: 1 },
        lease: { lease_id: "lease", generation: 1, proof: "proof", expires_at: "old" },
      },
      {
        job: {
          job_id: "receiver-job", provider: "chatgpt", revision: 2, lease_generation: 1,
          lease_expires_at: "renewed", checkpoint_sequence: null,
        },
        receipt: { kind: "capture_job_update", revision: 2 },
      },
      { job: { job_id: "receiver-job", revision: 3, checkpoint_sequence: 0 }, receipt: { revision: 3 } },
    ];
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => responses.shift() }));
    const client = new CaptureJobClient({ baseUrl: "http://receiver", token: "receiver-token", cache, fetchImpl });
    const adopted = await client.recoverOrCreate({
      provider: "chatgpt",
      accountHandle: "account-id",
      locator: { kind: "backfill", cutoff: "now" },
      intentPayload: { cutoff: "now" },
      sessionId: "profile",
    });
    const renewed = await client.update(adopted, {
      state: "held", attempt: 2, reason: "provider_safety_interstitial", next_eligible_at: null,
    });
    const body = new NativeBlob([canonicalJson({ version: 1, jobs: [], queue: [], revisions: [] })]);
    await client.checkpoint(renewed, { body, digest: `sha256:${"a".repeat(64)}` });

    const updateBody = JSON.parse(fetchImpl.mock.calls[4][1].body);
    const checkpointBody = JSON.parse(fetchImpl.mock.calls[5][1].headers["X-Polylogue-Checkpoint"]);
    expect(updateBody).toMatchObject({ expected_revision: 1, lease_id: "lease", proof: "proof" });
    expect(checkpointBody).toMatchObject({ expected_revision: 2, lease_id: "lease", proof: "proof" });
    expect(checkpointBody.sequence).toBe(0);
    expect(fetchImpl.mock.calls[5][1].body).toBe(body);
    expect(await body.text()).toBe('{"jobs":[],"queue":[],"revisions":[],"version":1}');
  });
});
