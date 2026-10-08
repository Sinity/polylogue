import { createHash } from "node:crypto";
import { Buffer } from "node:buffer";

import { indexedDB, IDBKeyRange } from "fake-indexeddb";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NativeCaptureNormalizer } from "../src/capture/native.js";
import { receiverContractPreparation, stagingRuntime } from "./infra/capture-staging.js";

import { BackfillCoordinator } from "../src/backfill/coordinator.js";
import { backfillAlarmName } from "../src/backfill/models.js";
import { ChatGptBackfillAdapter, ClaudeBackfillAdapter, GrokBackfillAdapter } from "../src/backfill/providers.js";
import { IndexedDbBackfillStore, MemoryBackfillStore } from "../src/backfill/storage.js";

const captureOwners = new Map();
async function serializedContentHash(file) { return createHash("sha256").update(Buffer.from(await file.arrayBuffer())).digest("hex"); }
async function captureContentHash(envelope, file) { return envelope.receiver_native?.sha256 || serializedContentHash(file); }
async function* checkpointResults(store, failures = []) {
  for await (const job of store.jobRecords()) {
    yield failures.find((failure) => failure.job_id === job.id) || { job_id: job.id, error: null, outcome: "committed" };
  }
}
function coordinatorFixture(options) {
  const key = options.store.databaseName || options.store;
  if (!captureOwners.has(key)) captureOwners.set(key, stagingRuntime());
  const { staging } = captureOwners.get(key);
  return new BackfillCoordinator({
    prepareCapture: (envelope, item, signal) => envelope.receiver_native
      ? { contentHash: envelope.receiver_native.sha256 }
      : staging.prepare(envelope, staging.conversionId(`backfill:${item.id}`), { delivery_kind: "backfill", id: item.id, job_id: item.job_id }, signal),
    ...options,
  });
}

async function retainedNativeCapture(envelope) {
  return captureOwners.get(envelope.capture_record_ref).retainedNativeReplies(envelope);
}

function response(body, { status = 200, retryAfter = null, provider: declaredProvider = null, refusal = null } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    polylogueSelectedOrganizationId: Array.isArray(body) ? body[0]?.uuid : null,
    headers: { get: (name) => (name.toLowerCase() === "retry-after" ? retryAfter : null) },
    json: vi.fn(async () => globalThis.structuredClone(body)),
    async normalizeCapture(item, attribution, relatedResponses = {}, signal = new globalThis.AbortController().signal) {
      const runtime = stagingRuntime(); const { staging, store } = runtime;
      const provider = declaredProvider || (body.mapping ? "chatgpt" : body.chat_messages ? "claude-ai" : "grok");
      const owner = { provider, tab_id: 42, document_id: "backfill-fixture-document" };
      const raw = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
      await staging.append(raw, owner, 0, Buffer.from(JSON.stringify(body)).toString("base64"));
      await staging.seal(raw, owner);
      const relatedRefs = {};
      for (const [key, response] of Object.entries(relatedResponses)) {
        const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
        await staging.append(ref, owner, 0, Buffer.from(JSON.stringify(await response.json())).toString("base64"));
        await staging.seal(ref, owner); relatedRefs[key] = ref;
      }
      const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store, {
        onPrepare: refusal ? () => { const error = new Error(refusal); error.code = refusal; throw error; } : undefined,
      }) });
      const envelope = await normalizer.normalize({ provider, rawRef: raw, nativeId: item.native_id, extensionVersion: "0.1.0", instanceId: "backfill-preparation-instance", attribution: { backfill: attribution }, signal, relatedRefs });
      captureOwners.set(envelope.capture_record_ref, runtime);
      return envelope;
    },
  };
}

function stagedGrokAdapter(fetchImpl, summary = {}) {
  const runtime = stagingRuntime(); const { staging, store } = runtime;
  const owner = { tab_id: 42, document_id: "grok-adapter-document", provider: "grok" };
  const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store, { summary }) });
  const responses = new Map();
  return new GrokBackfillAdapter(async (url, options) => {
    const result = await fetchImpl(url, options);
    if (!result.ok) return result;
    const body = await result.json();
    const ref = await staging.begin(owner, { kind: "native-response", source_url: url, capture_bundle: options.captureBundle });
    await staging.append(ref, owner, 0, Buffer.from(JSON.stringify(body)).toString("base64"));
    await staging.seal(ref, owner);
    const staged = { ...result, captureRawRef: ref, normalizeCapture: async (item, attribution, related, signal = new globalThis.AbortController().signal) => {
      const envelope = await normalizer.normalize({ provider: "grok", nativeId: item.native_id, rawRef: ref,
        relatedRefs: Object.fromEntries(Object.entries(related).map(([name, value]) => [name, value.captureRawRef])),
        acquisition: { kind: "grok-endpoint-bundle" }, extensionVersion: "0.1.0", instanceId: "backfill-preparation-instance", attribution, signal });
      captureOwners.set(envelope.capture_record_ref, runtime);
      return envelope;
    } };
    responses.set(ref.id, staged); return staged;
  }, { nativeBundleOwner: {
    begin: (nativeId) => store.beginNativeBundle({ owner, provider: "grok", nativeId, bundleId: globalThis.crypto.randomUUID(), requiredReplies: ["conversation", "responses"] }),
    response: async (bundleId, name, result) => {
      if (!result.ok) await store.publishNativeBundleReply(bundleId, owner, name, null, { ok: false, status: result.status });
    },
    restoreReply: (ref) => responses.get(ref.id),
    finish: (bundleId, signal) => normalizer.finishBundle(bundleId, owner, { signal }),
  } });
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
    if (this.fetchError) throw this.fetchError;
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

function harness({ adapter = new FixtureAdapter(), receiver = null, receiverPreflight = null, checkpoint = null, start = 100000, instanceId = "instance-a", policy = {}, store = new MemoryBackfillStore() } = {}) {
  let now = start;
  const alarms = { create: vi.fn(async () => undefined) };
  const durableReceiver = receiver || vi.fn(async (envelope, serialized) => ({ receiver_request_id: `ack-${envelope.session.provider_session_id}`, outcome: "accepted", submitted_content_hash: await captureContentHash(envelope, serialized), content_hash: await captureContentHash(envelope, serialized) }));
  const coordinator = coordinatorFixture({
    store,
    adapters: { chatgpt: adapter },
    receiver: durableReceiver,
    receiverPreflight,
    checkpoint,
    alarms,
    clock: () => now,
    random: () => 0,
    instanceId,
  });
  return { adapter, store, alarms, receiver: durableReceiver, coordinator, now: () => now, advance: (ms) => { now += ms; }, policy: { baseCadenceMs: 1000, ...policy } };
}

async function snapshotJobs(store) {
  const jobs = [];
  for await (const job of store.jobRecords()) jobs.push(job);
  return jobs;
}

async function startJob(h, patch = {}) {
  return h.coordinator.start({ provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", policy: { ...h.policy, ...patch } });
}

async function enumerateThenAdvance(h, job) {
  await h.coordinator.wake(job.id);
  h.advance(h.policy.baseCadenceMs);
}

describe("background backfill coordinator", () => {
  beforeEach(() => { vi.restoreAllMocks(); captureOwners.clear(); globalThis.IDBKeyRange = IDBKeyRange; });

  it.each([
    ["memory", () => new MemoryBackfillStore()],
    ["IndexedDB", () => new IndexedDbBackfillStore(indexedDB, `slow-${globalThis.crypto.randomUUID()}`)],
  ])("renews a progressing capture past multiple recovery leases with %s", async (_label, makeStore) => {
    // Leave IndexedDB scheduling on real immediates while controlling lease timers.
    vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    try {
      const adapter = new FixtureAdapter(["one"]); const h = harness({ adapter, store: makeStore(), policy: { leaseMs: 900 } });
      const job = await startJob(h); await enumerateThenAdvance(h, job);
      let release; let started;
      const began = new Promise((resolve) => { started = resolve; });
      const pending = new Promise((resolve) => { release = resolve; });
      adapter.fetchNative = async (id, signal) => { adapter.fetchCalls.push(id); started(); await pending; signal.throwIfAborted(); return response(chatGptNative(id)); };
      const execution = h.coordinator.wake(job.id); await began;
      for (let pass = 0; pass < 12; pass++) { h.advance(300); await vi.advanceTimersByTimeAsync(300); }
      expect((await h.store.getJob(job.id)).execution_expires_at_ms).toBeGreaterThan(h.now());
      expect(await h.store.acquireJobExecution(job.id, "second-worker", h.now(), 900)).toBeNull();
      release(); await execution;
      expect(adapter.fetchCalls).toEqual(["one"]);
      expect(((await h.store.queuePage(job.id)).items)[0].state).toBe("complete");
    } finally { vi.useRealTimers(); }
  });

  it("cancels and drains an active provider read before returning control", async () => {
    const adapter = new FixtureAdapter(["one"]); const h = harness({ adapter });
    const job = await startJob(h); await enumerateThenAdvance(h, job);
    let started; let drained = false;
    const began = new Promise((resolve) => { started = resolve; });
    adapter.fetchNative = async (_id, signal) => {
      started();
      try { await new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true })); }
      finally { drained = true; }
    };
    const execution = h.coordinator.wake(job.id); await began;
    const controlled = await h.coordinator.control(job.id, "cancel");
    expect(drained).toBe(true); expect(controlled.status).toBe("cancelled");
    await execution; expect(h.receiver).not.toHaveBeenCalled();
  });

  it.each(["accepted", "noop", "superseded"])("keeps retained ACK truth for %s", async (outcome) => {
    const receiver = vi.fn(async (envelope, serialized) => ({
      receiver_request_id: "retained-ack", outcome,
      submitted_content_hash: await captureContentHash(envelope, serialized),
      content_hash: outcome === "accepted" ? await captureContentHash(envelope, serialized) : "resident-hash",
    }));
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    h.advance(1000);
    await h.coordinator.wake(job.id);
    const item = (await h.store.queuePage(job.id)).items[0];
    expect(item.envelope).toBeNull();
    expect(receiver).toHaveBeenCalledTimes(1);
    expect((await h.coordinator.status(job.id)).status).toBe("complete");
    if (outcome === "superseded") {
      expect(item).toMatchObject({ state: "superseded", content_hash: null, last_response_class: "receiver_superseded" });
      expect(await h.store.getRevision("chatgpt", "one")).toBeUndefined();
    } else {
      expect(item.state).toBe("complete");
      expect(item.content_hash).toBe(item.receiver_receipt.content_hash);
      expect((await h.store.getRevision("chatgpt", "one")).receiver_content_hash).toBe(item.content_hash);
    }
  });

  it.each(["accepted", "superseded"])("keeps committed %s truth after finalization response loss", async (outcome) => {
    const receiver = vi.fn(async (envelope, serialized) => ({
      receiver_request_id: "neutral-finalization-ack", outcome,
      submitted_content_hash: await captureContentHash(envelope, serialized),
      content_hash: outcome === "accepted" ? await captureContentHash(envelope, serialized) : "resident-hash",
    }));
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver });
    const original = h.store.finalizeCaptureCas.bind(h.store);
    vi.spyOn(h.store, "finalizeCaptureCas").mockImplementation(async (...args) => {
      await original(...args);
      throw new Error("neutral post-commit response loss");
    });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    const item = (await h.store.queuePage(job.id)).items[0];
    expect(item).toMatchObject({ state: outcome === "superseded" ? "superseded" : "complete",
      envelope: null, receiver_receipt: { outcome } });
    expect((await h.coordinator.status(job.id)).status).toBe("complete");
    expect(receiver).toHaveBeenCalledTimes(1);
    expect(h.adapter.fetchCalls).toEqual(["one"]);
    if (outcome === "superseded") expect(await h.store.getRevision("chatgpt", "one")).toBeUndefined();
  });

  it.each(["foreign_submission", "unknown_outcome", "accepted_wrong_hash"])("refuses %s ACK without dropping retained input", async (fault) => {
    const receiver = vi.fn(async (envelope, serialized) => ({
      receiver_request_id: "ack", outcome: fault === "unknown_outcome" ? "unknown" : "accepted",
      submitted_content_hash: fault === "foreign_submission" ? "foreign" : await captureContentHash(envelope, serialized),
      content_hash: fault === "accepted_wrong_hash" ? "wrong" : await captureContentHash(envelope, serialized),
    }));
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    expect((await h.store.queuePage(job.id)).items[0]).toMatchObject({ state: "captured_waiting_receiver", envelope: expect.any(Object) });
    expect((await h.coordinator.status(job.id)).cooldown_reason).toBe("receiver_contract_incompatible");
  });

  it("survives a service-worker restart and completes each native capture once", async () => {
    const h = harness();
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    expect((await h.store.queueSummary(job.id)).progress.complete).toBe(1);

    h.advance(1000);
    const restarted = coordinatorFixture({ store: h.store, adapters: { chatgpt: h.adapter }, receiver: h.receiver, alarms: h.alarms, clock: h.now, random: () => 0, instanceId: "instance-b" });
    await restarted.wake(job.id);

    const status = await restarted.status(job.id);
    expect(status.status).toBe("complete");
    expect(status.progress.complete).toBe(2);
    expect(h.adapter.fetchCalls).toEqual(["one", "two"]);
    expect(h.receiver).toHaveBeenCalledTimes(2);
    expect((await retainedNativeCapture(h.receiver.mock.calls[0][0])).id).toBe("one");
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
    const restarted = coordinatorFixture({
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
    const restarted = coordinatorFixture({
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
    const receiver = vi.fn(async (envelope, serialized) => {
      calls += 1;
      if (calls === 1) throw new Error("receiver_down");
      return { receiver_request_id: "ack-recovered", outcome: "accepted", submitted_content_hash: await captureContentHash(envelope, serialized), content_hash: await captureContentHash(envelope, serialized) };
    });
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);
    expect(((await h.store.queuePage(job.id)).items)[0].state).toBe("captured_waiting_receiver");
    expect(h.adapter.fetchCalls).toHaveLength(1);
    const providerRequests = (await h.coordinator.status(job.id)).daily_requests;

    h.advance(1000);
    await h.coordinator.wake(job.id);
    const item = ((await h.store.queuePage(job.id)).items)[0];
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

  it("keeps a paused job non-runnable until its resume preflight succeeds", async () => {
    let releasePreflight;
    const receiverPreflight = vi.fn(async () => new Promise((resolve) => { releasePreflight = resolve; }));
    receiverPreflight.mockResolvedValueOnce(undefined);
    const h = harness({ receiverPreflight });
    const job = await startJob(h);
    await h.coordinator.control(job.id, "pause");

    const resuming = h.coordinator.control(job.id, "resume");
    await vi.waitFor(() => expect(receiverPreflight).toHaveBeenCalledTimes(2));
    expect((await h.store.getJob(job.id)).status).toBe("paused");
    await h.coordinator.wake(job.id);
    expect(h.adapter.enumerateCalls).toBe(0);

    const cancelling = h.coordinator.control(job.id, "cancel");
    releasePreflight();
    await Promise.all([resuming, cancelling]);
    expect(await h.store.getJob(job.id)).toMatchObject({ status: "cancelled", receiver_contract_epoch: "instance-a" });
    await h.coordinator.wake(job.id);
    expect(h.adapter.enumerateCalls).toBe(0);
  });

  it("holds provider work when the receiver-authority checkpoint cannot commit", async () => {
    const checkpoint = vi.fn(async () => { throw new Error("storage_local_quota"); });
    const h = harness({ checkpoint });
    const job = await startJob(h);
    expect(job).toMatchObject({
      status: "paused",
      cooldown_reason: "receiver_capture_job_authority_unavailable",
      recovery_checkpoint_error: "storage_local_quota",
    });
    await h.coordinator.wake(job.id);
    expect(h.adapter.enumerateCalls).toBe(0);
    expect((await h.coordinator.status(job.id)).recovery_checkpoint_error).toBe("storage_local_quota");
  });

  it("isolates a typed receiver checkpoint failure to its owning job", async () => {
    const h = harness();
    const failed = await startJob(h);
    const healthy = {
      ...await h.store.getJob(failed.id),
      id: "healthy-claude-job",
      provider: "claude-ai",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    await h.store.createJob(healthy);
    h.coordinator.checkpoint = vi.fn(() => checkpointResults(h.store, [{ job_id: failed.id, error: "capture_job_receiver_unavailable" }]));

    const status = await h.coordinator.status(failed.id);

    expect(status).toMatchObject({
      status: "paused",
      recovery_checkpoint_error: "capture_job_receiver_unavailable",
    });
    expect(await h.store.getJob(healthy.id)).toMatchObject({ status: "running" });
  });

  it("streams many paused scope refusals and coalesces a concurrent semantic control without re-dirtying derived outcomes", async () => {
    const store = new IndexedDbBackfillStore(indexedDB, `checkpoint-stream-${globalThis.crypto.randomUUID()}`);
    const h = harness({ store }); const seed = await startJob(h);
    await h.coordinator.control(seed.id, "pause");
    await store.putJob({ ...await store.getJob(seed.id), account_scope: null });
    for (let index = 0; index < 256; index++) await store.putJob({ ...await store.getJob(seed.id), id: `scope-${String(index).padStart(4, "0")}` });
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    let passes = 0; let observed = 0;
    h.coordinator.checkpoint = vi.fn(async function* () {
      const pass = ++passes; let first = true;
      for await (const job of store.jobRecords()) {
        yield { job_id: job.id, error: "capture_job_account_scope_unresolved", outcome: "scope_unresolved" };
        // An accumulator that consumes the whole iterator before applying
        // outcomes fails here; each row must publish before requesting another.
        expect((await store.getJob(job.id)).recovery_checkpoint_outcome).toMatchObject({ state: "failed", error: "capture_job_account_scope_unresolved" });
        observed++;
        if (first && pass === 1) { first = false; await gate; }
      }
    });
    const reading = h.coordinator.status(seed.id);
    await vi.waitFor(() => expect(observed).toBe(1));
    const controlling = h.coordinator.control("scope-0255", "cancel");
    await vi.waitFor(async () => expect((await store.getJob("scope-0255")).status).toBe("cancelled"));
    release();
    const [status] = await Promise.all([reading, controlling]);
    expect(status.recovery_checkpoint_error).toBe("capture_job_account_scope_unresolved");
    expect(passes).toBe(2); expect(observed).toBe(514);
    expect((await store.getJob("scope-0255")).status).toBe("cancelled");
    expect(h.adapter.enumerateCalls).toBe(0);
  });

  it("preserves a completed checkpoint prefix and marks every unattempted job after transport loss, then clears failures on success", async () => {
    const h = harness(); const first = await startJob(h);
    const other = { ...await h.store.getJob(first.id), id: "unattempted-job", provider: "claude-ai" };
    await h.store.createJob(other);
    h.coordinator.checkpoint = vi.fn(async function* () {
      yield { job_id: first.id, error: null, outcome: "committed" };
      throw new Error("receiver_transport_lost");
    });
    expect((await h.coordinator.status(first.id)).recovery_checkpoint_error).toBeNull();
    expect(await h.store.getJob(other.id)).toMatchObject({ status: "paused",
      recovery_checkpoint_outcome: { state: "failed", error: "receiver_transport_lost" } });
    expect((await h.store.getJob(first.id)).status).toBe("running");
    h.coordinator.checkpoint = vi.fn(() => checkpointResults(h.store));
    expect((await h.coordinator.status(other.id)).recovery_checkpoint_error).toBeNull();
    expect((await h.store.getJob(other.id)).status).toBe("paused");
    expect(h.adapter.enumerateCalls).toBe(0);
  });

  it("keeps a rate-limited checkpoint failure running until its retry deadline", async () => {
    // Anti-vacuity: pausing here leaves the job non-runnable, so the alarm at
    // the deadline could never resume it without operator action.
    const h = harness();
    const job = await startJob(h);
    const retryUntil = h.now() + 60_000;
    h.coordinator.checkpoint = vi.fn(() => checkpointResults(h.store, [{ job_id: job.id, error: "provider_throttled", outcome: "rate_limited", retry_until_ms: retryUntil }]));
    h.alarms.create.mockClear();

    await h.coordinator.status(job.id);

    expect(await h.store.getJob(job.id)).toMatchObject({
      status: "running",
      cooldown_reason: "provider_rate_limited",
      cooldown_until_ms: retryUntil,
    });
    expect(h.alarms.create).toHaveBeenCalledWith(expect.any(String), { when: retryUntil });
  });

  it("extends an existing checkpoint cooldown for a later Retry-After without shortening it on later observations", async () => {
    const h = harness();
    const job = await startJob(h);
    let retryUntil = h.now() + 60_000;
    h.coordinator.checkpoint = vi.fn(() => checkpointResults(h.store, [{
      job_id: job.id, error: "provider_throttled", outcome: "rate_limited", retry_until_ms: retryUntil,
    }]));
    await h.coordinator.status(job.id);
    expect((await h.store.getJob(job.id)).cooldown_until_ms).toBe(retryUntil);

    retryUntil = h.now() + 48 * 60 * 60 * 1000;
    const extended = retryUntil;
    h.alarms.create.mockClear();
    await h.coordinator.status(job.id);
    expect(await h.store.getJob(job.id)).toMatchObject({ status: "running", cooldown_until_ms: extended });
    expect(h.alarms.create).toHaveBeenCalledWith(expect.any(String), { when: extended });

    retryUntil = h.now() + 30_000;
    await h.coordinator.status(job.id);
    expect((await h.store.getJob(job.id)).cooldown_until_ms).toBe(extended);
    expect(h.adapter.enumerateCalls).toBe(0);
  });

  it("coalesces concurrent receiver-authority checkpoints across status readers", async () => {
    let releaseCheckpoint;
    const checkpoint = vi.fn(async () => {
      await new Promise((resolve) => { releaseCheckpoint = resolve; });
      return checkpointResults(h.store);
    });
    const h = harness({ checkpoint });
    const starting = h.coordinator.start({ provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    await vi.waitFor(() => expect(checkpoint).toHaveBeenCalledTimes(1));
    releaseCheckpoint();
    const job = await starting;
    checkpoint.mockClear();

    const first = h.coordinator.status(job.id);
    const second = h.coordinator.status(job.id);
    await vi.waitFor(() => expect(checkpoint).toHaveBeenCalledTimes(1));
    releaseCheckpoint();
    await Promise.all([first, second]);

    expect(checkpoint).toHaveBeenCalledTimes(1);
  });

  it("commits state mutated during an in-flight receiver checkpoint before returning", async () => {
    let releaseCheckpoint;
    const checkpointGate = new Promise((resolve) => { releaseCheckpoint = resolve; });
    const writes = [];
    const checkpoint = vi.fn(async () => {
      writes.push({ jobs: await snapshotJobs(h.store) });
      if (writes.length === 1) await checkpointGate;
      return checkpointResults(h.store);
    });
    const h = harness();
    const job = await startJob(h);
    h.coordinator.checkpoint = checkpoint;

    const reading = h.coordinator.status(job.id);
    await vi.waitFor(() => expect(checkpoint).toHaveBeenCalledTimes(1));
    const pausing = h.coordinator.control(job.id, "pause");
    await vi.waitFor(async () => expect((await h.store.getJob(job.id)).status).toBe("paused"));
    releaseCheckpoint();
    await Promise.all([reading, pausing]);

    expect(checkpoint).toHaveBeenCalledTimes(2);
    expect(writes[0].jobs[0].status).toBe("running");
    expect(writes[1].jobs[0].status).toBe("paused");
  });

  it("commits a healthy job mutation after a concurrent job checkpoint fails", async () => {
    let releaseCheckpoint;
    const checkpointGate = new Promise((resolve) => { releaseCheckpoint = resolve; });
    const writes = [];
    const h = harness();
    const failed = await startJob(h);
    const healthy = {
      ...await h.store.getJob(failed.id),
      id: "healthy-concurrent-job",
      provider: "claude-ai",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    await h.store.createJob(healthy);
    h.coordinator.checkpoint = vi.fn(async () => {
      writes.push({ jobs: await snapshotJobs(h.store) });
      if (writes.length === 1) await checkpointGate;
      return checkpointResults(h.store, [{ job_id: failed.id, error: "capture_job_receiver_unavailable" }]);
    });

    const readingFailed = h.coordinator.status(failed.id);
    await vi.waitFor(() => expect(h.coordinator.checkpoint).toHaveBeenCalledTimes(1));
    const pausingHealthy = h.coordinator.control(healthy.id, "pause");
    await vi.waitFor(async () => expect((await h.store.getJob(healthy.id)).status).toBe("paused"));
    releaseCheckpoint();
    const [failedStatus, healthyStatus] = await Promise.all([readingFailed, pausingHealthy]);

    expect(h.coordinator.checkpoint).toHaveBeenCalledTimes(2);
    expect(writes[0].jobs.find((job) => job.id === healthy.id).status).toBe("running");
    expect(writes[1].jobs.find((job) => job.id === healthy.id).status).toBe("paused");
    expect(failedStatus.recovery_checkpoint_error).toBe("capture_job_receiver_unavailable");
    expect(healthyStatus.recovery_checkpoint_error).toBeNull();
  });

  it("pauses exactly once on a 202-shaped ACK missing durable fields, then explicitly drains its stored envelope", async () => {
    let compatible = false;
    const receiver = vi.fn(async (envelope, serialized) => compatible
      ? { receiver_request_id: "ack-after-upgrade", outcome: "accepted", submitted_content_hash: await captureContentHash(envelope, serialized), content_hash: await captureContentHash(envelope, serialized) }
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
    const item = ((await h.store.queuePage(job.id)).items)[0];
    expect(first).toMatchObject({ status: "paused", cooldown_reason: "receiver_contract_incompatible" });
    expect(item).toMatchObject({ state: "captured_waiting_receiver", attempt_count: 0, last_response_class: "receiver_contract_incompatible" });
    expect(h.adapter.fetchCalls).toEqual(["one"]);
    expect(receiver).toHaveBeenCalledTimes(1);

    h.advance(60000);
    await h.coordinator.wake(job.id);
    expect(h.adapter.fetchCalls).toEqual(["one"]);
    expect(receiver).toHaveBeenCalledTimes(1);
  });

  it("retains the full native mapping and current node while publishing the canonical receiver reference", async () => {
    const native = chatGptNative("branch-session");
    native.mapping.first.children = ["second", "sibling"];
    native.mapping.sibling = { id: "sibling", parent: "first", children: [], message: { id: "sibling-message", author: { role: "assistant" }, content: { parts: ["alternate"] } } };
    native.current_node = "second";
    const adapter = new ChatGptBackfillAdapter();
    const capture = await adapter.normalizeCapture(response(native), { native_id: native.id }, { job_id: "job" });
    expect(await retainedNativeCapture(capture)).toEqual(native);
    expect(capture.provider_meta.capture_fidelity).toBe("native_full");
    expect(capture.session.turns).toEqual([]);
    expect(capture.receiver_native).toBeDefined();
    const fixture = new FixtureAdapter([native.id]); fixture.responses = [response(native)];
    const h = harness({ adapter: fixture }); const job = await startJob(h);
    await enumerateThenAdvance(h, job); await h.coordinator.wake(job.id);
    expect(await retainedNativeCapture(h.receiver.mock.calls[0][0])).toEqual(native);
    expect((await h.store.queuePage(job.id)).items).toMatchObject([{ state: "complete", capture_fidelity: "native_full" }]);
  });

  it("submits acquired assets through the authoritative adapter through the ordinary receiver path", async () => {
    const exactEnvelope = {
      polylogue_capture_kind: "browser_llm_session",
      provider_meta: { capture_fidelity: "native_full" },
      session: {
        provider: "chatgpt",
        provider_session_id: "one",
        turns: [{ role: "assistant", text: "complete output" }],
        attachments: [{
          name: "assistant-output.zip",
          inline_base64: globalThis.btoa("PK exact output"),
        }],
      },
    };
    const adapter = new FixtureAdapter(["one"]);
    adapter.normalizeCapture = vi.fn(async () => exactEnvelope);
    const h = harness({ adapter });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);

    expect(adapter.normalizeCapture).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ native_id: "one" }), expect.objectContaining({ job_id: job.id, instance_id: "instance-a" }), expect.any(globalThis.AbortSignal), expect.objectContaining({ itemId: expect.any(String), jobId: job.id, owner: "instance-a", generation: expect.any(Number) }));
    expect(h.receiver).toHaveBeenCalledWith(exactEnvelope, expect.objectContaining({ size: expect.any(Number) }), expect.any(globalThis.AbortSignal));
    expect(JSON.parse(await h.receiver.mock.calls[0][1].text())).toMatchObject({ session: { ...exactEnvelope.session, turns: exactEnvelope.session.turns.map((turn, ordinal) => ({ ...turn, ordinal })) } });
    expect((await h.store.queuePage(job.id)).items).toMatchObject([{
      state: "complete",
      capture_fidelity: "native_full",
    }]);
  });

  it("retains retryable acquisition failures without delivering incomplete assets", async () => {
    const adapter = new FixtureAdapter(["one"]);
    adapter.normalizeCapture = vi.fn(async () => { throw new Error("exact_asset_acquisition_pending"); });
    const h = harness({ adapter });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);

    expect(adapter.normalizeCapture).toHaveBeenCalledTimes(1);
    expect(h.receiver).not.toHaveBeenCalled();
    expect((await h.store.queuePage(job.id)).items).toMatchObject([{
      state: "retry_wait",
      last_response_class: "transport",
      last_error: "exact_asset_acquisition_pending",
    }]);
  });

  it("treats a shared provider cooldown as a rate limit, not a transport failure", async () => {
    // Anti-vacuity: route the shared-cooldown refusal back through
    // retryTransport and the job counts transport failures and pauses as
    // repeated_transport_failures although no provider request was made.
    const adapter = new FixtureAdapter(["one"]);
    const cooldown = Object.assign(new Error("provider_rate_limited"), { outcome: "rate_limited", retryAfterSeconds: 30 });
    const h = harness({ adapter, policy: {} });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    adapter.fetchError = cooldown;
    await h.coordinator.wake(job.id);
    const status = await h.coordinator.status(job.id);
    expect(status.cooldown_reason).toBe("provider_rate_limited");
    expect(status.cooldown_until_ms).toBe(h.now() + 30000);
    expect(status.transport_failures || 0).toBe(0);
    expect(status.status).not.toBe("paused");
  });

  it("honors Retry-After on repeated 429s and resumes without a permanent count cut", async () => {
    const adapter = new FixtureAdapter(["one"]);
    adapter.responses = [response({}, { status: 429, retryAfter: "60" }), response({}, { status: 429, retryAfter: "60" })];
    const h = harness({ adapter, policy: {} });
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
    expect(status.status).toBe("running");
    expect(status.learned_cadence_ms).toBeGreaterThan(status.policy.baseCadenceMs);
    expect(status.daily_requests).toBe(3);
  });

  it("retains empty native evidence and distinguishes auth, transport, refusal, and ACK", async () => {
    const cases = [
      { result: response({}, { status: 403 }), expected: "auth_required" },
      { result: response(chatGptNative("one", false)), expected: "complete" },
      { result: response({}, { status: 503 }), expected: "retry_wait" },
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
      expect(((await h.store.queuePage(job.id)).items)[0].state).toBe(scenario.expected);
      expect((await h.coordinator.status(job.id)).daily_requests).toBe(2);
    }
  });

  it("fails native provider-contract drift per item and continues the job", async () => {
    const adapter = new FixtureAdapter(["bad", "good"]);
    adapter.fetchNative = async (nativeId) => {
      adapter.fetchCalls.push(nativeId);
      if (nativeId === "bad") throw new Error("provider_contract_drift:chatgpt_conversation.mapping_must_be_object");
      return response(chatGptNative(nativeId));
    };
    const h = harness({ adapter });
    const job = await startJob(h);
    await enumerateThenAdvance(h, job);
    await h.coordinator.wake(job.id);

    expect(((await h.store.queuePage(job.id)).items).find((item) => item.native_id === "bad")).toMatchObject({
      native_id: "bad",
      state: "failed",
      attempt_count: 0,
      last_response_class: "contract_drift",
    });
    expect((await h.coordinator.status(job.id)).transport_failures).toBe(0);

    h.advance(h.policy.baseCadenceMs);
    await h.coordinator.wake(job.id);
    expect(await h.coordinator.status(job.id)).toMatchObject({ status: "complete" });
    expect(h.adapter.fetchCalls).toEqual(["bad", "good"]);
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
    expect(((await store.queuePage(job.id)).items)[0].state).toBe("auth_required");

    await h.coordinator.control(job.id, "resume");
    const resumed = ((await store.queuePage(job.id)).items)[0];
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

  it.each([
    ["memory storage", () => new MemoryBackfillStore()],
    ["IndexedDB", () => new IndexedDbBackfillStore(indexedDB, `polylogue-test-${globalThis.crypto.randomUUID()}`)],
  ])("atomically requeues authenticated work when resuming with %s", async (_label, makeStore) => {
    const store = makeStore();
    const h = harness({ store });
    const job = await startJob(h);
    await store.putQueue({ id: "auth-held", job_id: job.id, provider: "chatgpt", native_id: "one", state: "auth_required", attempt_count: 0, next_eligible_at_ms: null });

    await h.coordinator.control(job.id, "resume");

    expect((await store.queuePage(job.id)).items).toMatchObject([{ state: "eligible", resume_state: "eligible", next_eligible_at_ms: h.now(), last_error: null }]);
  });

  it("grants only one lease across simultaneous extension instances", async () => {
    const store = new MemoryBackfillStore();
    await store.putQueue({ id: "q1", job_id: "j1", provider: "chatgpt", native_id: "one", state: "eligible", next_eligible_at_ms: 0 });
    const [left, right] = await Promise.all([
      store.acquireNextLease("j1", "instance-left", 100, 1000),
      store.acquireNextLease("j1", "instance-right", 100, 1000),
    ]);
    expect([left, right].filter(Boolean)).toHaveLength(1);
    expect(((await store.queuePage("j1")).items)[0].state).toBe("leased");
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
    const coordinator = coordinatorFixture({ store, adapters: { chatgpt: adapter }, receiver: vi.fn(), alarms: { create: vi.fn() }, clock: () => 1000, random: () => 0, instanceId: "idb-instance" });
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
    const cancellation = h.coordinator.control(job.id, "cancel");
    await vi.waitFor(async () => expect((await h.store.getJob(job.id)).status).toBe("cancelled"));
    releaseFetch(response(chatGptNative("one")));
    await cancellation;
    await wake;
    expect((await h.coordinator.status(job.id)).status).toBe("cancelled");
    expect(((await h.store.queuePage(job.id)).items)[0].state).toBe("cancelled");
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
    expect(((await h.store.queuePage(second.id)).items)[0].state).toBe("unchanged");
    expect(adapter.fetchCalls).toEqual(["one"]);
    expect(h.receiver).toHaveBeenCalledTimes(1);
  });

  it("retains acquired receiver evidence across repeated failures without refetching or exhausting attempts", async () => {
    const receiver = vi.fn(async () => { throw new Error("receiver_down"); });
    const h = harness({ adapter: new FixtureAdapter(["one"]), receiver });
    const job = await startJob(h); await enumerateThenAdvance(h, job);
    for (let attempt = 0; attempt < 20; attempt++) {
      await h.coordinator.wake(job.id); h.advance(60000);
    }
    expect((await h.coordinator.status(job.id)).status).toBe("running");
    expect((await h.store.queuePage(job.id)).items).toMatchObject([{ state: "captured_waiting_receiver", attempt_count: 20 }]);
    expect(h.adapter.fetchCalls).toEqual(["one"]);
    expect(receiver).toHaveBeenCalledTimes(20);
  });

  it("pins Claude organization identity across restart and hard-reserves its request budget", async () => {
    let now = 1000;
    const store = new MemoryBackfillStore();
    const alarms = { create: vi.fn(async () => undefined) };
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response([{ uuid: "org-1" }]))
      .mockResolvedValueOnce(response([{ uuid: "claude-1", updated_at: "2026-01-02T00:00:00Z" }]))
      .mockResolvedValueOnce(response({ uuid: "claude-1", chat_messages: [{ uuid: "m1", sender: "human", text: "hello" }] }));
    const receiver = vi.fn(async (envelope, serialized) => ({ receiver_request_id: "ack", outcome: "accepted", submitted_content_hash: await captureContentHash(envelope, serialized), content_hash: await captureContentHash(envelope, serialized) }));
    const first = coordinatorFixture({ store, adapters: { "claude-ai": new ClaudeBackfillAdapter(fetchImpl) }, receiver, alarms, clock: () => now, random: () => 0 });
    const job = await first.start({ provider: "claude-ai", cutoff: "2026-01-01T00:00:00Z", policy: { baseCadenceMs: 1000, maxDailyRequests: 3 } });
    await first.wake(job.id);
    expect((await first.status(job.id)).daily_requests).toBe(2);
    expect((await first.status(job.id)).provider_options.claudeOrganizationId).toBe("org-1");

    now += 1000;
    const restarted = coordinatorFixture({ store, adapters: { "claude-ai": new ClaudeBackfillAdapter(fetchImpl) }, receiver, alarms, clock: () => now, random: () => 0 });
    await restarted.wake(job.id);
    expect((await restarted.status(job.id)).daily_requests).toBe(3);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    expect(fetchImpl.mock.calls[2][0]).toContain("/organizations/org-1/chat_conversations/claude-1");

    const cappedFetch = vi.fn();
    const capped = coordinatorFixture({ store: new MemoryBackfillStore(), adapters: { "claude-ai": new ClaudeBackfillAdapter(cappedFetch) }, receiver, alarms, clock: () => now, random: () => 0 });
    const cappedJob = await capped.start({ provider: "claude-ai", cutoff: "2026-01-01T00:00:00Z", policy: { maxDailyRequests: 1 } });
    await capped.wake(cappedJob.id);
    expect((await capped.status(cappedJob.id)).status).toBe("paused");
    expect(cappedFetch).not.toHaveBeenCalled();
  });
});

describe("provider adapter contracts", () => {

  it("normalizes ChatGPT and rejects inventory drift loudly", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response({ items: [{ id: "gpt-1", title: "GPT", update_time: 1710000100 }], total: 1 }))
      .mockResolvedValueOnce(response(chatGptNative("gpt-1")));
    const adapter = new ChatGptBackfillAdapter(fetchImpl);
    const inventory = await adapter.enumerate("0", "2020-01-01T00:00:00Z");
    const capture = await adapter.normalizeCapture(await adapter.fetchNative("gpt-1"), inventory.items[0], { job_id: "j", queue_id: "q", instance_id: "i" });
    expect(await retainedNativeCapture(capture)).toEqual(chatGptNative("gpt-1"));
    expect(capture.session.turns).toEqual([]);
    expect(capture.receiver_native).toBeDefined();
    expect(capture.session.provider_meta.backfill.instance_id).toBe("i");

    const drifted = new ChatGptBackfillAdapter(vi.fn(async () => response({ conversations: [] })));
    await expect(drifted.enumerate()).rejects.toThrow("provider_contract_drift:chatgpt_inventory.items_must_be_array");
  });

  // polylogue-ah21: a code-interpreter call/result pair must land as a
  // matched tool_use/tool_result block pair, not two independent prose
  // turns. This is the exact shape (content_type "code" then
  // "execution_output") that used to be indistinguishable from ordinary
  // assistant text once BrowserCaptureTurn had no blocks channel.
  it("retains a ChatGPT code-interpreter call and output for canonical receiver pairing", async () => {
    const native = {
      id: "gpt-ci",
      title: "Code interpreter run",
      mapping: {
        u1: { parent: null, message: { id: "u1", author: { role: "user" }, content: { parts: ["compute 6 * 7"] }, create_time: 1 } },
        "call-1": { parent: "u1", message: { id: "call-1", author: { role: "assistant" }, content: { content_type: "code", text: "6 * 7" }, create_time: 2 } },
        "result-1": { parent: "call-1", message: { id: "result-1", author: { role: "tool" }, content: { content_type: "execution_output", text: "42" }, create_time: 3 } },
      },
    };
    const adapter = new ChatGptBackfillAdapter(vi.fn());
    const capture = await adapter.normalizeCapture(response(native), { native_id: "gpt-ci", title: "Code interpreter run" }, {});

    expect(await retainedNativeCapture(capture)).toEqual(native);
    expect(capture.receiver_native).toBeDefined();
    // Canonical pairing: native-rich-blocks-v1.json through actual receiver
    // parity and test_code_interpreter_call_and_result_share_a_tool_id.

  });

  it("does not misclassify a thoughts-only ChatGPT turn as no_turns", async () => {
    // Canonical block structure is exercised by native-rich-blocks-v1.json
    // through the actual receiver. The browser retains the original reply
    // and the canonical nonempty summary instead of projecting prose.
    const native = {
      id: "gpt-reasoning-only",
      title: "Reasoning only",
      mapping: {
        reasoning: {
          parent: null,
          message: {
            id: "reasoning-1",
            author: { role: "assistant" },
            content: {
              content_type: "thoughts",
              thoughts: [{ summary: "Weighing options", content: "First I considered X, then Y." }],
            },
            create_time: 1,
          },
        },
      },
    };
    const adapter = new ChatGptBackfillAdapter(vi.fn());
    const capture = await adapter.normalizeCapture(response(native), { native_id: "gpt-reasoning-only", title: "Reasoning only" }, {});

    expect(await retainedNativeCapture(capture)).toEqual(native);
    expect(capture.capture_summary.turnCount).toBeGreaterThan(0);
    expect(capture.receiver_native).toBeDefined();
  });

  it("does not misclassify a thinking-only Claude turn as no_turns", async () => {
    const body = {
      uuid: "claude-reasoning-only",
      name: "Reasoning only",
      chat_messages: [
        {
          uuid: "message-1",
          sender: "assistant",
          content: [{ type: "thinking", thinking: "Considering the tradeoffs before answering." }],
          created_at: "2026-01-01T00:00:00Z",
        },
      ],
    };
    const adapter = new ClaudeBackfillAdapter(vi.fn(), "org-1");
    const capture = await adapter.normalizeCapture(response(body), { native_id: "claude-reasoning-only", title: "Reasoning only" }, {});

    expect(await retainedNativeCapture(capture)).toEqual(body);
    expect(capture.capture_summary.turnCount).toBeGreaterThan(0);
    expect(capture.receiver_native).toBeDefined();
  });

  it("retains a recipient-addressed JSON tool call for canonical receiver classification", async () => {
    const native = {
      id: "gpt-tool",
      title: "Web search",
      mapping: {
        call: { parent: null, message: { id: "call-1", author: { role: "assistant" }, recipient: "web", content: { content_type: "text", parts: [JSON.stringify({ search_query: "weather" })] }, create_time: 1 } },
      },
    };
    const adapter = new ChatGptBackfillAdapter(vi.fn());
    const capture = await adapter.normalizeCapture(response(native), { native_id: "gpt-tool", title: "Web search" }, {});

    expect(await retainedNativeCapture(capture)).toEqual(native);
    expect(capture.receiver_native).toBeDefined();
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
      const parsed = new globalThis.URL(url);
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
    expect((await retainedNativeCapture(capture)).chat_messages[0]).toMatchObject({
      sender: "claude", parent_message_uuid: "parent", model: "claude-opus",
    });
    expect(capture.receiver_native).toBeDefined();
    expect(fetchImpl.mock.calls[2][0]).toContain("tree=True");
    expect(fetchImpl.mock.calls[2][0]).toContain("render_all_tools=true");
    expect(fetchImpl.mock.calls[2][0]).toContain("consistency=strong");

    await expect(adapter.normalizeCapture(response({ messages: [] }, { provider: "claude-ai", refusal: "invalid_native_preparation" }), inventory.items[0], {})).rejects.toThrow("invalid_native_preparation");
  });

  // Grok's own /rest/app-chat/conversations REST surface, verified live
  // 2026-07-31 (see src/content/grok_bridge.js): pageToken-cursored
  // enumeration, and a two-request native fetch (conversation metadata +
  // /responses) retained as separate original replies for normalizeCapture.
  it("enumerates Grok conversations by pageToken and stops when nextPageToken is absent", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response({
        conversations: [
          { conversationId: "g-1", title: "First", modifyTime: "2026-07-01T00:00:00Z" },
          { conversationId: "g-2", title: "Second", modifyTime: "2026-06-30T00:00:00Z" },
        ],
        nextPageToken: "g-2",
      }))
      .mockResolvedValueOnce(response({
        conversations: [{ conversationId: "g-3", title: "Third", modifyTime: "2026-06-01T00:00:00Z" }],
      }));
    const adapter = new GrokBackfillAdapter(fetchImpl);

    const first = await adapter.enumerate("0", null);
    expect(first).toMatchObject({ classification: "success", done: false, next_cursor: "g-2" });
    expect(first.items.map((item) => item.native_id)).toEqual(["g-1", "g-2"]);
    expect(fetchImpl.mock.calls[0][0]).not.toContain("pageToken");

    const second = await adapter.enumerate(first.next_cursor, null);
    expect(second).toMatchObject({ classification: "success", done: true });
    expect(second.items.map((item) => item.native_id)).toEqual(["g-3"]);
    expect(fetchImpl.mock.calls[1][0]).toContain("pageToken=g-2");
  });

  it("stops Grok pagination once a page crosses the cutoff", async () => {
    const adapter = new GrokBackfillAdapter(vi.fn(async () => response({
      conversations: [
        { conversationId: "new", modifyTime: "2026-07-01T00:00:00Z" },
        { conversationId: "old", modifyTime: "2020-01-01T00:00:00Z" },
      ],
      nextPageToken: "old",
    })));
    const inventory = await adapter.enumerate("0", "2026-01-01T00:00:00Z");
    expect(inventory.items.map((item) => item.native_id)).toEqual(["new"]);
    expect(inventory.done).toBe(true);
  });

  it("rejects a non-array Grok inventory loudly", async () => {
    const adapter = new GrokBackfillAdapter(vi.fn(async () => response({ conversations: "not-an-array" })));
    await expect(adapter.enumerate()).rejects.toThrow("provider_contract_drift:grok_inventory.conversations_must_be_array");
  });

  it("retains original Grok endpoint replies while normalizing temporary sessions", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response({ conversationId: "g-1", title: "Fixture", temporary: true, createTime: "2026-06-27T12:39:08Z", modifyTime: "2026-06-27T13:46:12Z" }))
      .mockResolvedValueOnce(response({
        responses: [
          { responseId: "r-1", sender: "human", message: "hello", createTime: "2026-06-27T12:39:09Z" },
          { responseId: "r-2", sender: "ASSISTANT", parentResponseId: "r-1", message: "hi there", createTime: "2026-06-27T12:39:15Z", model: "grok-3" },
        ],
      }));
    const adapter = stagedGrokAdapter(fetchImpl, { session_kind: "temporary" });
    const capture = await adapter.normalizeCapture(await adapter.fetchNative("g-1"), { native_id: "g-1", title: "Fixture" }, { job_id: "j" });

    expect(capture.session.provider).toBe("grok");
    expect(capture.session.session_kind).toBe("temporary");
    expect(capture.session.turns).toEqual([]);
    expect(capture.receiver_native).toBeDefined();
    expect(fetchImpl.mock.calls[1][0]).toContain("/responses");
    const retained = await retainedNativeCapture(capture);
    expect(retained.conversation).toMatchObject({ conversationId: "g-1", temporary: true });
    expect(retained.responses.responses).toHaveLength(2);
    expect(retained.responses.responses[1]).toMatchObject({ sender: "ASSISTANT", message: "hi there", parentResponseId: "r-1", model: "grok-3" });

    await expect(
      adapter.normalizeCapture(response({ responses: "not-an-array" }, { refusal: "invalid_native_preparation" }), { native_id: "g-1" }, {}),
    ).rejects.toThrow("invalid_native_preparation");
  });

  it("surfaces a failed Grok /responses fetch as the fetchNative result without a second request", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response({ conversationId: "g-1" }))
      .mockResolvedValueOnce(response({ code: 5 }, { status: 404 }));
    const adapter = stagedGrokAdapter(fetchImpl);

    const result = await adapter.fetchNative("g-1");
    expect(result.ok).toBe(false);
    expect(result.status).toBe(404);
  });
});

describe("disposable pre-streaming refusal conversion", () => {
  const refusal = { id: "old-refusal", job_id: "paused-job", provider: "chatgpt", native_id: "native-old", state: "bridge_oversize", lease_owner: null };
  const paused = { id: "paused-job", provider: "chatgpt", status: "paused", cooldown_reason: "operator_pause" };

  it.each([2, 3])("atomically converts v%s after an aborted upgrade and preserves paused/acquired custody", async (oldVersion) => {
    const name = `conversion-${globalThis.crypto.randomUUID()}`;
    const old = await new Promise((resolve, reject) => {
      const request = indexedDB.open(name, oldVersion);
      request.onupgradeneeded = () => {
        for (const store of ["queue", "jobs", "revisions"]) request.result.createObjectStore(store, { keyPath: "id" });
        if (oldVersion === 3) {
          const metadata = request.result.createObjectStore("capture_retry_metadata", { keyPath: "queue_order", autoIncrement: true });
          metadata.createIndex("id", "id", { unique: true });
          request.result.createObjectStore("capture_retry_bodies", { keyPath: "id" });
          request.result.createObjectStore("capture_retry_state");
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    await new Promise((resolve, reject) => {
      const tx = old.transaction(["queue", "jobs"], "readwrite");
      tx.objectStore("queue").put(refusal); tx.objectStore("jobs").put(paused);
      tx.objectStore("queue").put({ ...refusal, id: "acquired-old-refusal", envelope: { retained: "original bytes" } });
      tx.oncomplete = resolve; tx.onerror = () => reject(tx.error);
    });
    old.close();
    await new Promise((resolve, reject) => {
      const request = indexedDB.open(name, 4);
      request.onupgradeneeded = () => request.transaction.abort();
      request.onerror = () => resolve();
      request.onsuccess = () => { request.result.close(); reject(new Error("aborted_upgrade_published")); };
    });
    const owner = new IndexedDbBackfillStore(indexedDB, name);
    expect(await owner.getQueue(refusal.id)).toMatchObject({ state: "eligible", next_eligible_at_ms: 0, lease_owner: null });
    expect(await owner.getJob(paused.id)).toMatchObject(paused);
    expect(await owner.getQueue("acquired-old-refusal")).toMatchObject({
      state: "recovery_required", envelope: { retained: "original bytes" },
      last_error: "capture_refusal_retains_acquired_evidence",
    });
    (await owner.database()).close();
    const restarted = new IndexedDbBackfillStore(indexedDB, name);
    expect(await restarted.getQueue(refusal.id)).toMatchObject({ state: "eligible" });
    expect(await restarted.acquireJobExecution(paused.id, "other-owner", 1000, 1000)).toBeNull();
  });

  it("converts checkpoint refusal once and preserves acquired custody as an explicit hold", async () => {
    const owner = new IndexedDbBackfillStore(indexedDB, `checkpoint-conversion-${globalThis.crypto.randomUUID()}`);
    const acquired = { ...refusal, id: "acquired-refusal", native_id: "native-acquired", body_ref: "retained-body" };
    const checkpoint = { version: 1, jobs: [paused], queue: [refusal, acquired], revisions: [] };
    for (let restart = 0; restart < 2; restart++) {
      for (const job of checkpoint.jobs) await owner.convertLocalCheckpointRecord("jobs", job);
      for (const item of checkpoint.queue) await owner.convertLocalCheckpointRecord("queue", item);
    }
    expect(await owner.getQueue(refusal.id)).toMatchObject({ state: "eligible" });
    expect(await owner.getQueue(acquired.id)).toMatchObject({ state: "recovery_required", body_ref: "retained-body", last_error: "capture_refusal_retains_acquired_evidence" });
    expect(await owner.getJob(paused.id)).toMatchObject({ status: "paused", cooldown_reason: "operator_pause" });
    expect(((await owner.queuePage(paused.id)).items).every((item) => item.state !== "bridge_oversize")).toBe(true);
  });
});
