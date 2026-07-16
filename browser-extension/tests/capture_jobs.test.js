import { describe, expect, it, vi } from "vitest";
import { CaptureJobClient, canonicalJson, deriveAccountScope } from "../src/backfill/capture_jobs.js";

describe("CaptureJob extension recovery", () => {
  it("derives opaque scopes and rehydrates after chrome.storage cache loss", async () => {
    const cache = { values: {}, get: vi.fn(async (keys) => Object.fromEntries(Object.keys(keys).map((key) => [key, cache.values[key] ?? null]))), set: vi.fn(async (patch) => Object.assign(cache.values, patch)) };
    const scope = await deriveAccountScope("receiver-token", "chatgpt", "account@example.test");
    expect(scope).toMatch(/^h1:/);
    expect(scope).not.toContain("account");
    const responses = [
      { schema: "polylogue.capture-jobs.capabilities.v1", protocol_min: 1, protocol_max: 1, scope_namespace: "cjs1:receiver-namespace" },
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
    const serializedRequests = fetchImpl.mock.calls.map(([, options]) => options.body).join("\n");
    expect(serializedRequests).not.toContain("account@example.test");
    expect(JSON.stringify(cache.values)).not.toContain("account@example.test");
  });

  it("renews the proven lease before checkpointing the returned revision", async () => {
    const cache = {
      values: {},
      get: vi.fn(async (keys) => Object.fromEntries(Object.keys(keys).map((key) => [key, null]))),
      set: vi.fn(async (patch) => Object.assign(cache.values, patch)),
    };
    const responses = [
      { schema: "polylogue.capture-jobs.capabilities.v1", protocol_min: 1, protocol_max: 1, scope_namespace: "cjs1:receiver-namespace" },
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
    await client.checkpoint(renewed, { version: 1, jobs: [], queue: [], revisions: [] });

    const updateBody = JSON.parse(fetchImpl.mock.calls[4][1].body);
    const checkpointBody = JSON.parse(fetchImpl.mock.calls[5][1].body);
    expect(updateBody).toMatchObject({ expected_revision: 1, lease_id: "lease", proof: "proof" });
    expect(checkpointBody).toMatchObject({ expected_revision: 2, lease_id: "lease", proof: "proof" });
    expect(checkpointBody.checkpoint.sequence).toBe(0);
  });
});
