import { describe, expect, it, vi } from "vitest";
import { CaptureJobClient, canonicalJson, deriveAccountScope } from "../src/backfill/capture_jobs.js";

describe("CaptureJob extension recovery", () => {
  it("derives opaque scopes and rehydrates after chrome.storage cache loss", async () => {
    const cache = { values: {}, get: vi.fn(async (keys) => Object.fromEntries(Object.keys(keys).map((key) => [key, cache.values[key] ?? null]))), set: vi.fn(async (patch) => Object.assign(cache.values, patch)) };
    const scope = await deriveAccountScope("receiver-token", "chatgpt", "account@example.test");
    expect(scope).toMatch(/^h1:/);
    expect(scope).not.toContain("account");
    const responses = [
      { jobs: [] },
      { job: { job_id: "receiver-job", provider: "chatgpt", revision: 0, lease_generation: 0 } },
      { job: { job_id: "receiver-job", provider: "chatgpt", revision: 1, lease_generation: 1 }, lease: { lease_id: "lease", generation: 1, proof: "proof" } },
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
});
