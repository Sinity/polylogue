import { createHash } from "node:crypto";

import { beforeEach, describe, expect, it, vi } from "vitest";

import { BackfillCoordinator } from "../src/backfill/coordinator.js";
import { backfillAlarmName, serializedContentHash, serializedJson } from "../src/backfill/models.js";
import { ChatGptBackfillAdapter, ClaudeBackfillAdapter } from "../src/backfill/providers.js";
import { MemoryBackfillStore, progressBuckets } from "../src/backfill/storage.js";

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

function harness({ adapter = new FixtureAdapter(), receiver = null, start = 100000, instanceId = "instance-a", policy = {} } = {}) {
  let now = start;
  const store = new MemoryBackfillStore();
  const alarms = { create: vi.fn(async () => undefined) };
  const durableReceiver = receiver || vi.fn(async (envelope, serialized) => ({ receiver_request_id: `ack-${envelope.session.provider_session_id}`, content_hash: await serializedContentHash(serialized) }));
  const coordinator = new BackfillCoordinator({
    store,
    adapters: { chatgpt: adapter },
    receiver: durableReceiver,
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

  it("persists Claude organization identity across restart and hard-reserves its request budget", async () => {
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

    await expect(adapter.normalizeCapture(response({ messages: [] }), inventory.items[0], {})).rejects.toThrow("provider_contract_drift:claude_conversation.chat_messages_must_be_array");
  });
});
