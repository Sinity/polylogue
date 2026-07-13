import { createHash } from "node:crypto";

import { indexedDB } from "fake-indexeddb";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { BackfillCoordinator } from "../src/backfill/coordinator.js";
import { backfillAlarmName, serializedContentHash, serializedJson } from "../src/backfill/models.js";
import { ChatGptBackfillAdapter, ClaudeBackfillAdapter } from "../src/backfill/providers.js";
import { IndexedDbBackfillStore, MemoryBackfillStore, progressBuckets } from "../src/backfill/storage.js";

function response(body, { status = 200, retryAfter = null } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => (name === "Retry-After" ? retryAfter : null) },
    json: vi.fn(async () => structuredClone(body)),
  };
}

function chatGptNative(id, turns = true) {
  return {
    id,
    title: `Conversation ${id}`,
    create_time: 1710000000,
    update_time: 1710000100,
    mapping: turns
      ? {
          first: { parent: null, message: { id: `${id}-u`, author: { role: "user" }, content: { parts: [{ text: "hello" }] }, create_time: 1710000000 } },
          second: { parent: "first", message: { id: `${id}-a`, author: { role: "function" }, content: { result: "world", content_type: "tool_result" }, create_time: 1710000001, metadata: { model_slug: "tool-model" } } },
        }
      : {},
  };
}

class FixtureAdapter {
  constructor(ids = ["one", "two"]) {
    this.ids = ids;
    this.fetchCalls = [];
    this.responses = [];
    this.enumerateCalls = 0;
  }
  async enumerate() {
    this.enumerateCalls += 1;
    return { classification: "success", items: this.ids.map((native_id) => ({ native_id, updated_at: "2026-07-01T00:00:00Z" })), next_cursor: String(this.ids.length), done: true, request_count: 1 };
  }
  async fetchNative(nativeId) {
    this.fetchCalls.push(nativeId);
    return this.responses.shift() || response(chatGptNative(nativeId));
  }
  classifyResponse(result) {
    if (result.ok) return "success";
    if (result.status === 429) return "rate_limited";
    if (result.status === 403) return "auth_or_challenge";
    if (result.status >= 500) return "transport";
    return "fatal";
  }
  async normalizeCapture(result, item, attribution) {
    return new ChatGptBackfillAdapter().normalizeCapture(result, item, attribution);
  }
}

function harness({ adapter = new FixtureAdapter(), receiver = null, receiverPreflight = null, start = 100000, instanceId = "instance-a", policy = {}, store = new MemoryBackfillStore() } = {}) {
  let now = start;
  const alarms = { create: vi.fn(async () => undefined) };
  const durableReceiver = receiver || vi.fn(async (envelope, serialized) => ({ receiver_request_id: `ack-${envelope.session.provider_session_id}`, content_hash: await serializedContentHash(serialized) }));
  const coordinator = new BackfillCoordinator({
    store,
    adapters: { chatgpt: adapter },
    receiver: durableReceiver,
    receiverPreflight,
    alarms,
    clock: () => now,
    random: () => 0,
    instanceId,
  });
  return { adapter, store, alarms, receiver: durableReceiver, coordinator, now: () => now, advance: (ms) => { now += ms; }, policy: { baseCadenceMs: 1000, ...policy } };
}

async function startJob(h, patch = {}) {
  return h.coordinator.start({ provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", policy: { ...h.policy, ...patch } });
}

async function enumerateThenAdvance(h, job) {
  await h.coordinator.wake(job.id);
  h.advance(h.policy.baseCadenceMs);
}

describe("background backfill coordinator", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("survives a service-worker restart and completes each native capture once", async () => {
    const h = harness();
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    expect(progressBuckets(await h.store.listQueue(job.id)).complete).toBe(1);

    h.advance(1000);
    const restarted = new BackfillCoordinator({ store: h.store, adapters: { chatgpt: h.adapter }, receiver: h.receiver, alarms: h.alarms, clock: h.now, random: () => 0, instanceId: "instance-b" });
    await restarted.wake(job.id);

    const status = await restarted.status(job.id);
    expect(status.status).toBe("complete");
    expect(status.progress.complete).toBe(2);
    expect(h.adapter.fetchCalls).toEqual(["one", "two"]);
    expect(h.receiver).toHaveBeenCalledTimes(2);
  });

  it("recovers the durable job, queue, revision, and ACK ledgers from real IndexedDB after a worker restart", async () => {
    const databaseName = `polylogue-restart-${globalThis.crypto.randomUUID()}`;
    const adapter = new FixtureAdapter(["one", "two"]);
    const store = new IndexedDbBackfillStore(indexedDB, databaseName);
    const h = harness({ adapter, store });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    const before = await h.coordinator.status(job.id);
    const restarted = new BackfillCoordinator({
      store: new IndexedDbBackfillStore(indexedDB, databaseName),
      adapters: { chatgpt: adapter }, receiver: h.receiver, alarms: h.alarms,
      clock: h.now, random: () => 0, instanceId: "instance-after-restart",
    });
    h.advance(h.policy.baseCadenceMs);
    await restarted.wake(job.id);
    const after = await restarted.status(job.id);
    expect(after).toMatchObject({ id: job.id, inventory_cursor: before.inventory_cursor, last_ack: expect.objectContaining({ content_hash: expect.any(String) }) });
    expect(after.progress.complete).toBe(2);
    expect(adapter.fetchCalls).toEqual(["one", "two"]);
    expect(h.receiver).toHaveBeenCalledTimes(2);
    indexedDB.deleteDatabase(databaseName);
  });

  it("rechecks the receiver contract after a worker restart even when its durable extension identity is unchanged", async () => {
    const receiverPreflight = vi.fn(async () => undefined);
    const h = harness({ receiverPreflight, instanceId: "stable-extension-id" });
    const job = await startJob(h);
    const restarted = new BackfillCoordinator({
      store: h.store, adapters: { chatgpt: h.adapter }, receiver: h.receiver, receiverPreflight,
      alarms: h.alarms, clock: h.now, random: () => 0, instanceId: "stable-extension-id", receiverContractEpoch: "new-worker-epoch",
    });
    receiverPreflight.mockRejectedValueOnce(new Error("receiver_contract_incompatible:durable_ack_fields_missing"));
    await restarted.wake(job.id);
    expect(await restarted.status(job.id)).toMatchObject({ status: "paused", cooldown_reason: "receiver_contract_incompatible" });
    expect(receiverPreflight).toHaveBeenCalledTimes(2);
    expect(h.adapter.enumerateCalls).toBe(0);
  });

  it("keeps a receiver-down capture durable and retries the ACK without refetching provider data", async () => {
    let calls = 0;
    const receiver = vi.fn(async (_envelope, serialized) => {
      calls += 1;
      if (calls === 1) throw new Error("receiver_down");
      return { receiver_request_id: "ack-recovered", content_hash: await serializedContentHash(serialized) };
    });
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    expect((await h.store.listQueue(job.id))[0].state).toBe("captured_waiting_receiver");
    expect(h.adapter.fetchCalls).toHaveLength(1);
    const providerRequests = (await h.coordinator.status(job.id)).daily_requests;

    h.advance(1000);
    await h.coordinator.wake(job.id);
    const item = (await h.store.listQueue(job.id))[0];
    expect(item.state).toBe("complete");
    expect(item.receiver_receipt.content_hash).toBe(item.content_hash);
    expect(h.adapter.fetchCalls).toHaveLength(1);
    expect((await h.coordinator.status(job.id)).daily_requests).toBe(providerRequests);
  });

  it("preflights the receiver contract before any provider request", async () => {
    const receiverPreflight = vi.fn(async () => { throw new Error("receiver_contract_incompatible:durable_ack_fields_missing"); });
    const h = harness({ receiverPreflight });
    const job = await startJob(h);
    const status = await h.coordinator.status(job.id);
    expect(status).toMatchObject({ status: "paused", cooldown_reason: "receiver_contract_incompatible" });
    expect(h.adapter.enumerateCalls).toBe(0);
    expect(h.receiver).not.toHaveBeenCalled();
  });

  it("pauses exactly once on a 202-shaped ACK missing durable fields, then explicitly drains its stored envelope", async () => {
    let compatible = false;
    const receiver = vi.fn(async (_envelope, serialized) => compatible
      ? { receiver_request_id: "ack-after-upgrade", content_hash: await serializedContentHash(serialized) }
      : { receiver_request_id: "accepted-but-stale" });
    const receiverPreflight = vi.fn(async () => undefined);
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver, receiverPreflight });
    const job = await startJob(h);
    await h.coordinator.wake(job.id);
    h.advance(h.policy.baseCadenceMs);
    await h.coordinator.wake(job.id);
    expect((await h.coordinator.status(job.id)).cooldown_reason).toBe("receiver_contract_incompatible");
    expect(h.receiver).toHaveBeenCalledTimes(1);
    expect(h.adapter.fetchCalls).toEqual(["one"]);

    h.advance(60000);
    await h.coordinator.wake(job.id);
    expect(h.receiver).toHaveBeenCalledTimes(1);
    expect(h.adapter.fetchCalls).toEqual(["one"]);

    compatible = true;
    await h.coordinator.control(job.id, "resume");
    await h.coordinator.wake(job.id);
    const resumed = await h.coordinator.status(job.id);
    expect(resumed.progress.complete).toBe(1);
    expect(h.adapter.fetchCalls).toEqual(["one"]);
    expect(h.receiver).toHaveBeenCalledTimes(2);
  });

  it("turns a stale accepted ACK into a receiver-contract pause without retrying or refetching", async () => {
    const receiver = vi.fn(async () => ({ receiver_request_id: "accepted-but-no-hash" }));
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    const first = await h.coordinator.status(job.id);
    const item = (await h.store.listQueue(job.id))[0];
    expect(first).toMatchObject({ status: "paused", cooldown_reason: "receiver_contract_incompatible" });
    expect(item).toMatchObject({ state: "captured_waiting_receiver", attempt_count: 0, last_response_class: "receiver_contract_incompatible" });
    expect(h.adapter.fetchCalls).toEqual(["one"]);
    expect(receiver).toHaveBeenCalledTimes(1);

    h.advance(60000);
    await h.coordinator.wake(job.id);
    expect(h.adapter.fetchCalls).toEqual(["one"]);
    expect(receiver).toHaveBeenCalledTimes(1);
  });

  it("exports no credentials and restores profile-loss evidence without replaying a missing envelope", async () => {
    const store = new MemoryBackfillStore();
    await store.createJob({
      id: "checkpoint-job", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", provider_options: { claudeOrganizationId: "account-should-not-survive" },
      status: "running", policy: { maxDailyRequests: 10 }, learned_cadence_ms: 40000, daily_requests: 7,
      last_ack: { receiver_request_id: "ack-1", content_hash: "abc" }, execution_generation: 0,
    });
    await store.putQueue({
      id: "checkpoint-q", job_id: "checkpoint-job", provider: "chatgpt", native_id: "conversation-1", state: "captured_waiting_receiver",
      envelope: { raw_provider_payload: { authorization: "Bearer secret", account_id: "account-should-not-survive" } },
      receiver_receipt: { cookie: "secret" }, content_hash: "abc", attempt_count: 2,
    });
    const checkpoint = await store.exportRecoveryCheckpoint();
    const encoded = JSON.stringify(checkpoint);
    expect(encoded).not.toContain("Bearer secret");
    expect(encoded).not.toContain("account-should-not-survive");
    expect(encoded).not.toContain("cookie");
    expect(checkpoint.jobs[0]).toMatchObject({ learned_cadence_ms: 40000, daily_requests: 7, last_ack: { receiver_request_id: "ack-1" } });

    const restoredStore = new MemoryBackfillStore();
    expect(await restoredStore.restoreRecoveryCheckpoint(checkpoint)).toEqual({ restored: 1, reason: "browser_profile_recovery_required" });
    const restored = await restoredStore.getJob("checkpoint-job");
    expect(restored).toMatchObject({ status: "paused", cooldown_reason: "browser_profile_recovery_required", last_ack: { receiver_request_id: "ack-1" } });
    const restoredQueue = await restoredStore.listQueue("checkpoint-job");
    expect(restoredQueue).toMatchObject([{ state: "recovery_required" }]);
    expect(restoredQueue[0]).not.toHaveProperty("envelope");
    const receiver = vi.fn();
    const coordinator = new BackfillCoordinator({ store: restoredStore, adapters: { chatgpt: new FixtureAdapter(["one"]) }, receiver, alarms: { create: vi.fn() }, clock: () => 200000 });
    await coordinator.wake("checkpoint-job");
    expect(receiver).not.toHaveBeenCalled();
  });

  it("honors Retry-After exactly and opens a circuit after repeated 429s", async () => {
    const adapter = new FixtureAdapter(["one"]);
    adapter.responses = [response({}, { status: 429, retryAfter: "60" }), response({}, { status: 429, retryAfter: "60" })];
    const h = harness({ adapter, policy: { breakerThreshold: 2 } });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    let status = await h.coordinator.status(job.id);
    expect(status.cooldown_reason).toBe("provider_rate_limited");
    expect(status.cooldown_until_ms).toBe(h.now() + 60000);

    h.advance(59999);
    await h.coordinator.wake(job.id);
    expect(adapter.fetchCalls).toHaveLength(1);
    h.advance(1);
    await h.coordinator.wake(job.id);
    status = await h.coordinator.status(job.id);
    expect(adapter.fetchCalls).toHaveLength(2);
    expect(status.status).toBe("paused");
    expect(status.learned_cadence_ms).toBeGreaterThan(status.policy.baseCadenceMs);
    expect(status.daily_requests).toBe(3);
  });

  it("persists auth, native-empty, bounded transport, and durable ACK as distinct outcomes", async () => {
    const cases = [
      { result: response({}, { status: 403 }), expected: "auth_required" },
      { result: response(chatGptNative("one", false)), expected: "no_turns" },
      { result: response({}, { status: 503 }), expected: "failed", policy: { maxTransportAttempts: 1, breakerThreshold: 5 } },
      { result: response({}, { status: 400 }), expected: "failed" },
      { result: response(chatGptNative("one")), expected: "complete" },
    ];
    for (const scenario of cases) {
      const adapter = new FixtureAdapter(["one"]);
      adapter.responses = [scenario.result];
      const h = harness({ adapter, policy: scenario.policy });
      const job = await startJob(h);
      await enumerateThenAdvance(h, job);
      await h.coordinator.wake(job.id);
      expect((await h.store.listQueue(job.id))[0].state).toBe(scenario.expected);
      expect((await h.coordinator.status(job.id)).daily_requests).toBe(2);
    }
  });

  it.each([
    ["memory storage", () => new MemoryBackfillStore()],
    ["IndexedDB", () => new IndexedDbBackfillStore(indexedDB, `polylogue-test-${globalThis.crypto.randomUUID()}`)],
  ])("atomically requeues auth-required work when resuming with %s", async (_label, makeStore) => {
    const adapter = new FixtureAdapter(["one"]);
    adapter.responses = [response({}, { status: 403 }), response(chatGptNative("one"))];
    const store = makeStore();
    const h = harness({ adapter, store });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);

    expect((await h.coordinator.status(job.id)).status).toBe("paused");
    expect((await store.listQueue(job.id))[0].state).toBe("auth_required");

    await h.coordinator.control(job.id, "resume");
    const resumed = (await store.listQueue(job.id))[0];
    expect(resumed).toMatchObject({
      state: "eligible",
      resume_state: "eligible",
      lease_owner: null,
      lease_expires_at_ms: null,
      next_eligible_at_ms: h.now(),
      last_response_class: null,
      last_error: null,
    });

    h.advance(h.policy.baseCadenceMs);
    await h.coordinator.wake(job.id);
    const status = await h.coordinator.status(job.id);
    expect(status.status).toBe("complete");
    expect(status.progress).toMatchObject({ complete: 1, operator_action: 0 });
    expect(adapter.fetchCalls).toEqual(["one", "one"]);
    expect(h.receiver).toHaveBeenCalledTimes(1);
  });

  it("grants only one lease across simultaneous extension instances", async () => {
    const store = new MemoryBackfillStore();
    await store.putQueue({ id: "q1", job_id: "j1", provider: "chatgpt", native_id: "one", state: "eligible", next_eligible_at_ms: 0 });
    const [left, right] = await Promise.all([
      store.acquireNextLease("j1", "instance-left", 100, 1000),
      store.acquireNextLease("j1", "instance-right", 100, 1000),
    ]);
    expect([left, right].filter(Boolean)).toHaveLength(1);
    expect((await store.listQueue("j1"))[0].state).toBe("leased");
  });

  it("serializes leases across separate jobs for the same provider", async () => {
    const store = new MemoryBackfillStore();
    await store.putQueue({ id: "q1", job_id: "j1", provider: "chatgpt", native_id: "one", state: "eligible", next_eligible_at_ms: 0 });
    await store.putQueue({ id: "q2", job_id: "j2", provider: "chatgpt", native_id: "two", state: "eligible", next_eligible_at_ms: 0 });
    expect(await store.acquireNextLease("j1", "instance-left", 100, 1000)).not.toBeNull();
    expect(await store.acquireNextLease("j2", "instance-right", 100, 1000)).toBeNull();
  });

  it("invalidates an expired execution even when the stable instance id reacquires it", async () => {
    const store = new MemoryBackfillStore();
    await store.createJob({ id: "j1", provider: "chatgpt", status: "running", policy: { maxDailyRequests: 10 }, execution_generation: 0 });
    const first = await store.acquireJobExecution("j1", "stable-instance", 0, 10);
    const second = await store.acquireJobExecution("j1", "stable-instance", 11, 10);
    expect(second.execution_generation).toBe(first.execution_generation + 1);
    await expect(store.putQueueCas("j1", "stable-instance", first.execution_generation, { id: "q", job_id: "j1" })).rejects.toThrow("stale_backfill_execution:j1");
  });

  it("accounts inventory cadence and budget before any native fetch", async () => {
    const h = harness({ adapter: new FixtureAdapter(["one"]), policy: { maxDailyRequests: 1 } });
    const job = await startJob(h);
    await h.coordinator.wake(job.id);
    expect(h.adapter.enumerateCalls).toBe(1);
    expect((await h.coordinator.status(job.id)).daily_requests).toBe(1);

    await h.coordinator.wake(job.id);
    expect(h.adapter.fetchCalls).toHaveLength(0);
    h.advance(1000);
    await h.coordinator.wake(job.id);
    expect((await h.coordinator.status(job.id)).status).toBe("paused");
    expect(h.adapter.fetchCalls).toHaveLength(0);
  });

  it("uses per-job alarms so a later deadline cannot replace earlier work", async () => {
    const h = harness();
    const first = await startJob(h);
    await h.store.putJob({ ...(await h.store.getJob(first.id)), status: "complete" });
    h.advance(10);
    const second = await startJob(h);
    const names = h.alarms.create.mock.calls.map(([name]) => name);
    expect(names).toContain(backfillAlarmName(first.id));
    expect(names).toContain(backfillAlarmName(second.id));
    expect(new Set(names).size).toBeGreaterThanOrEqual(2);
  });

  it("rejects a second active crawl for the same provider", async () => {
    const h = harness();
    await startJob(h);
    await expect(startJob(h)).rejects.toThrow("backfill_job_already_active:chatgpt");
  });

  it("serializes concurrent wakes and request reservation in real IndexedDB", async () => {
    const databaseName = `polylogue-test-${globalThis.crypto.randomUUID()}`;
    const store = new IndexedDbBackfillStore(indexedDB, databaseName);
    let releaseInventory;
    const inventoryGate = new Promise((resolve) => { releaseInventory = resolve; });
    const adapter = new FixtureAdapter(["one"]);
    adapter.enumerate = vi.fn(async () => {
      await inventoryGate;
      return { classification: "success", items: [{ native_id: "one", updated_at: "2026-07-01T00:00:00Z" }], next_cursor: "1", done: true, request_count: 1 };
    });
    const coordinator = new BackfillCoordinator({ store, adapters: { chatgpt: adapter }, receiver: vi.fn(), alarms: { create: vi.fn() }, clock: () => 1000, random: () => 0, instanceId: "idb-instance" });
    const job = await coordinator.start({ provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    const wakes = [coordinator.wake(job.id), coordinator.wake(job.id)];
    await vi.waitFor(() => expect(adapter.enumerate).toHaveBeenCalledTimes(1));
    releaseInventory();
    await Promise.all(wakes);
    expect(adapter.enumerate).toHaveBeenCalledTimes(1);
    expect((await coordinator.status(job.id)).daily_requests).toBe(1);
    indexedDB.deleteDatabase(databaseName);
  });

  it("invalidates an in-flight fetch on cancel without resurrecting queue state", async () => {
    const adapter = new FixtureAdapter(["one"]);
    let releaseFetch;
    const fetchGate = new Promise((resolve) => { releaseFetch = resolve; });
    adapter.fetchNative = vi.fn(async () => fetchGate);
    const h = harness({ adapter });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    const wake = h.coordinator.wake(job.id);
    await vi.waitFor(() => expect(adapter.fetchNative).toHaveBeenCalledTimes(1));
    await h.coordinator.control(job.id, "cancel");
    releaseFetch(response(chatGptNative("one")));
    await wake;
    expect((await h.coordinator.status(job.id)).status).toBe("cancelled");
    expect((await h.store.listQueue(job.id))[0].state).toBe("cancelled");
    expect(h.receiver).not.toHaveBeenCalled();
  });

  it("skips an unchanged native revision in a later job", async () => {
    const adapter = new FixtureAdapter(["one"]);
    const h = harness({ adapter });
    const first = await startJob(h);
    await enumerateThenAdvance(h, first);
    await h.coordinator.wake(first.id);
    expect(h.receiver).toHaveBeenCalledTimes(1);
    h.advance(1000);
    const second = await startJob(h);
    await enumerateThenAdvance(h, second);
    await h.coordinator.wake(second.id);
    expect((await h.store.listQueue(second.id))[0].state).toBe("unchanged");
    expect(adapter.fetchCalls).toEqual(["one"]);
    expect(h.receiver).toHaveBeenCalledTimes(1);
  });

  it("fail-pauses receiver retries and stored envelope bytes at configured bounds", async () => {
    const receiverDown = vi.fn(async () => { throw new Error("receiver_down"); });
    const receiverHarness = harness({ adapter: new FixtureAdapter(["one"]), receiver: receiverDown, policy: { maxReceiverAttempts: 1 } });
    const receiverJob = await startJob(receiverHarness);
    await enumerateThenAdvance(receiverHarness, receiverJob);
    await receiverHarness.coordinator.wake(receiverJob.id);
    expect((await receiverHarness.coordinator.status(receiverJob.id)).cooldown_reason).toBe("receiver_retry_budget_exhausted");
    expect((await receiverHarness.store.listQueue(receiverJob.id))[0].state).toBe("captured_waiting_receiver");

    const byteHarness = harness({ adapter: new FixtureAdapter(["one"]), policy: { maxStoredBytes: 64 } });
    const byteJob = await startJob(byteHarness);
    await enumerateThenAdvance(byteHarness, byteJob);
    await byteHarness.coordinator.wake(byteJob.id);
    expect((await byteHarness.coordinator.status(byteJob.id)).cooldown_reason).toBe("storage_budget_exhausted");
    expect((await byteHarness.store.listQueue(byteJob.id))[0].last_response_class).toBe("storage_budget_exhausted");
    expect(byteHarness.receiver).not.toHaveBeenCalled();
  });

  it("pins Claude organization identity across restart and hard-reserves its request budget", async () => {
    let now = 1000;
    const store = new MemoryBackfillStore();
    const alarms = { create: vi.fn(async () => undefined) };
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response([{ uuid: "org-1" }]))
      .mockResolvedValueOnce(response([{ uuid: "claude-1", updated_at: "2026-01-02T00:00:00Z" }]))
      .mockResolvedValueOnce(response({ uuid: "claude-1", chat_messages: [{ uuid: "m1", sender: "human", text: "hello" }] }));
    const receiver = vi.fn(async (_envelope, serialized) => ({ receiver_request_id: "ack", content_hash: await serializedContentHash(serialized) }));
    const first = new BackfillCoordinator({ store, adapters: { "claude-ai": new ClaudeBackfillAdapter(fetchImpl) }, receiver, alarms, clock: () => now, random: () => 0 });
    const job = await first.start({ provider: "claude-ai", cutoff: "2026-01-01T00:00:00Z", policy: { baseCadenceMs: 1000, maxDailyRequests: 3 } });
    await first.wake(job.id);
    expect((await first.status(job.id)).daily_requests).toBe(2);
    expect((await first.status(job.id)).provider_options.claudeOrganizationId).toBe("org-1");

    now += 1000;
    const restarted = new BackfillCoordinator({ store, adapters: { "claude-ai": new ClaudeBackfillAdapter(fetchImpl) }, receiver, alarms, clock: () => now, random: () => 0 });
    await restarted.wake(job.id);
    expect((await restarted.status(job.id)).daily_requests).toBe(3);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    expect(fetchImpl.mock.calls[2][0]).toContain("/organizations/org-1/chat_conversations/claude-1");

    const cappedFetch = vi.fn();
    const capped = new BackfillCoordinator({ store: new MemoryBackfillStore(), adapters: { "claude-ai": new ClaudeBackfillAdapter(cappedFetch) }, receiver, alarms, clock: () => now, random: () => 0 });
    const cappedJob = await capped.start({ provider: "claude-ai", cutoff: "2026-01-01T00:00:00Z", policy: { maxDailyRequests: 1 } });
    await capped.wake(cappedJob.id);
    expect((await capped.status(cappedJob.id)).status).toBe("paused");
    expect(cappedFetch).not.toHaveBeenCalled();
  });
});

describe("provider adapter contracts", () => {
  it("hashes the exact UTF-8 JSON request bytes used by the receiver contract", async () => {
    const payload = { z: "żółć 😀", number: 1.25, nested: { b: 2, a: 1 } };
    const serialized = serializedJson(payload);
    expect(await serializedContentHash(serialized)).toBe(createHash("sha256").update(serialized, "utf8").digest("hex"));
  });

  it("normalizes ChatGPT and rejects inventory drift loudly", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response({ items: [{ id: "gpt-1", title: "GPT", update_time: 1710000100 }], total: 1 }))
      .mockResolvedValueOnce(response(chatGptNative("gpt-1")));
    const adapter = new ChatGptBackfillAdapter(fetchImpl);
    const inventory = await adapter.enumerate("0", "2020-01-01T00:00:00Z");
    const capture = await adapter.normalizeCapture(await adapter.fetchNative("gpt-1"), inventory.items[0], { job_id: "j", queue_id: "q", instance_id: "i" });
    expect(capture.session.turns).toHaveLength(2);
    expect(capture.session.turns[0].text).toBe("hello");
    expect(capture.session.turns[1].role).toBe("tool");
    expect(capture.session.turns[1].provider_meta.model_slug).toBe("tool-model");
    expect(capture.session.provider_meta.backfill.instance_id).toBe("i");

    const drifted = new ChatGptBackfillAdapter(vi.fn(async () => response({ conversations: [] })));
    await expect(drifted.enumerate()).rejects.toThrow("provider_contract_drift:chatgpt_inventory.items_must_be_array");
  });

  it("refuses a false-empty ChatGPT inventory without proven page auth context", async () => {
    const adapter = new ChatGptBackfillAdapter(
      vi.fn(async () => response({ items: [], total: 0 })),
      { requirePageContext: true },
    );

    await expect(adapter.enumerate("0", "2026-01-01T00:00:00Z")).resolves.toMatchObject({
      classification: "auth_or_challenge",
      done: false,
      items: [],
    });
  });

  it.each([
    ["memory storage", () => new MemoryBackfillStore()],
    ["IndexedDB", () => new IndexedDbBackfillStore(indexedDB, `polylogue-test-${globalThis.crypto.randomUUID()}`)],
  ])("keeps an unproven 200-empty inventory paused, then captures once with %s", async (_label, makeStore) => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response({ items: [], total: 0 }))
      .mockResolvedValueOnce(Object.assign(response({ items: [{ id: "one", update_time: 1780000000 }], total: 1 }), { polyloguePageContext: true }))
      .mockResolvedValueOnce(Object.assign(response({ items: [], total: 0 }), { polyloguePageContext: true }))
      .mockResolvedValueOnce(Object.assign(response({ items: [], total: 0 }), { polyloguePageContext: true }))
      .mockResolvedValueOnce(Object.assign(response({ items: [], total: 0 }), { polyloguePageContext: true }))
      .mockResolvedValueOnce(Object.assign(response(chatGptNative("one")), { polyloguePageContext: true }));
    const adapter = new ChatGptBackfillAdapter(fetchImpl, { requirePageContext: true });
    const h = harness({ adapter, store: makeStore() });
    const job = await startJob(h);

    await h.coordinator.wake(job.id);
    let status = await h.coordinator.status(job.id);
    expect(status).toMatchObject({ status: "paused", inventory_complete: false, cooldown_reason: "provider_auth_or_challenge" });

    await h.coordinator.control(job.id, "resume");
    for (let wake = 0; wake < 5; wake += 1) {
      h.advance(h.policy.baseCadenceMs);
      await h.coordinator.wake(job.id);
    }
    status = await h.coordinator.status(job.id);
    expect(status).toMatchObject({ status: "complete", inventory_complete: true, progress: { complete: 1 } });
    expect(fetchImpl).toHaveBeenCalledTimes(6);
    expect(h.receiver).toHaveBeenCalledTimes(1);
  });

  it("stops descending inventory pagination when a page crosses the cutoff", async () => {
    const adapter = new ChatGptBackfillAdapter(vi.fn(async () => response({
      items: [
        { id: "new", update_time: 1780000000 },
        { id: "old", update_time: 1600000000 },
      ],
      total: 5000,
    })));
    const inventory = await adapter.enumerate("3:0", "2026-01-01T00:00:00Z");
    expect(inventory.items.map((item) => item.native_id)).toEqual(["new"]);
    expect(inventory.done).toBe(true);
  });

  it("treats ChatGPT total as a page sentinel rather than a global corpus count", async () => {
    const firstPage = Array.from({ length: 28 }, (_value, index) => ({
      id: `conversation-${index}`,
      update_time: 1780000000 - index,
    }));
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response({ items: firstPage, total: 29 }))
      .mockResolvedValueOnce(response({ items: [{ id: "conversation-28", update_time: 1779990000 }], total: 29 }));
    const adapter = new ChatGptBackfillAdapter(fetchImpl);

    const first = await adapter.enumerate("0", null);
    const second = await adapter.enumerate(first.next_cursor, null);

    expect(first).toMatchObject({ next_cursor: "0:28", done: false });
    expect(second).toMatchObject({ next_cursor: "1:0", done: false });
    expect(fetchImpl.mock.calls[0][0]).toContain("limit=28");
    expect(fetchImpl.mock.calls[1][0]).toContain("offset=28");
  });

  it("enumerates every active, starred, and archived ChatGPT partition", async () => {
    const fetchImpl = vi.fn(async () => response({ items: [], total: 0 }));
    const adapter = new ChatGptBackfillAdapter(fetchImpl);
    let cursor = "0";
    let result;
    for (let partition = 0; partition < 4; partition += 1) {
      result = await adapter.enumerate(cursor, null);
      cursor = result.next_cursor;
    }

    expect(result.done).toBe(true);
    expect(fetchImpl.mock.calls.map(([url]) => {
      const parsed = new URL(url);
      return [parsed.searchParams.get("is_archived"), parsed.searchParams.get("is_starred")];
    })).toEqual([
      ["false", "false"],
      ["false", "true"],
      ["true", "false"],
      ["true", "true"],
    ]);
  });

  it("does not assume Claude inventory order when filtering a full page", async () => {
    const records = Array.from({ length: 100 }, (_, index) => ({
      uuid: `claude-${index}`,
      updated_at: index === 1 ? "2020-01-01T00:00:00Z" : "2026-02-01T00:00:00Z",
    }));
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response([{ uuid: "org-1" }]))
      .mockResolvedValueOnce(response(records));
    const adapter = new ClaudeBackfillAdapter(fetchImpl);
    const inventory = await adapter.enumerate("0", "2026-01-01T00:00:00Z");
    expect(inventory.done).toBe(false);
    expect(inventory.items.at(-1).native_id).toBe("claude-99");
  });

  it("normalizes Claude inventory/native fixtures and rejects message drift", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response([{ uuid: "org-1" }]))
      .mockResolvedValueOnce(response([{ uuid: "claude-1", name: "Claude", updated_at: "2026-01-02T00:00:00Z" }]))
      .mockResolvedValueOnce(response({ uuid: "claude-1", name: "Claude", chat_messages: [{ uuid: "m1", sender: "claude", content: [{ type: "text", text: "hello" }], parent_message_uuid: "parent", model: "claude-opus" }] }));
    const adapter = new ClaudeBackfillAdapter(fetchImpl);
    const inventory = await adapter.enumerate("0", "2026-01-01T00:00:00Z");
    const capture = await adapter.normalizeCapture(await adapter.fetchNative("claude-1"), inventory.items[0], { job_id: "j" });
    expect(capture.session.provider).toBe("claude-ai");
    expect(capture.session.turns[0].role).toBe("assistant");
    expect(capture.session.turns[0].parent_turn_id).toBe("parent");
    expect(capture.session.turns[0].provider_meta.model).toBe("claude-opus");
    expect(fetchImpl.mock.calls[2][0]).toContain("tree=True");
    expect(fetchImpl.mock.calls[2][0]).toContain("render_all_tools=true");
    expect(fetchImpl.mock.calls[2][0]).toContain("consistency=strong");

    await expect(adapter.normalizeCapture(response({ messages: [] }), inventory.items[0], {})).rejects.toThrow("provider_contract_drift:claude_conversation.chat_messages_must_be_array");
  });
});
