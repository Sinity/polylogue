// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import { IDBFactory, IDBKeyRange } from "fake-indexeddb";
import { memoryOriginStorage } from "./infra/capture-staging.js";

afterEach(() => vi.restoreAllMocks());

async function setup() {
  vi.resetModules();
  globalThis.indexedDB = new IDBFactory();
  globalThis.IDBKeyRange = IDBKeyRange;
  const { IndexedDbBackfillStore } = await import("../src/backfill/storage.js");
  const { CaptureStaging } = await import("../src/capture/staging.js");
  const store = new IndexedDbBackfillStore(globalThis.indexedDB);
  const storage = memoryOriginStorage();
  let local = { polylogueAmbientSettings: { enabled: false, automaticCaptureEnabled: false } };
  let listener;
  const area = {
    get: async (defaults) => ({ ...defaults, ...local }),
    set: async (values) => { local = { ...local, ...values }; },
  };
  const network = vi.fn(async () => { throw new Error("unexpected network"); });
  const adapters = {
    storage: { local: area, session: area }, network,
    runtime: { onMessage: { addListener: (fn) => { listener = fn; } } },
    action: { setBadgeText: async () => {}, setBadgeBackgroundColor: async () => {} },
    alarms: { create: async () => {}, clear: async () => {} },
    tabs: {},
  };
  async function start({ cold = false } = {}) {
    if (cold) vi.resetModules();
    const staging = new CaptureStaging(storage, store);
    const { startBackgroundRuntime } = await import("../src/background/runtime.js");
    startBackgroundRuntime({ ...adapters, captureStaging: staging });
    const page = () => new Promise((resolve) => listener({ type: "polylogue.getCaptureQueue", pageSize: 50 }, {}, resolve));
    return { staging, page };
  }
  return { store, storage, network, start, local: () => local };
}

const input = (text) => ({ session: { provider: "chatgpt", provider_session_id: "neutral-session",
  provider_meta: { capture_fidelity: "native_full" }, turns: [{ text, metadata: { absent: null } }] },
  provenance: { extension_instance_id: null, acquisition_sequence: null } });

describe("original retry input conversion through the background worker", () => {
  it("publishes each body before retiring source and preserves absent historical identity", async () => {
    const { store, storage, network, start } = await setup();
    await store.importCaptureRetries([
      { metadata: { id: "z-first" }, envelope: input("first") },
      { metadata: { id: "a-second" }, envelope: input("second") },
    ]);
    const retire = store.retireCaptureRetry.bind(store);
    const seen = [];
    vi.spyOn(store.constructor.prototype, "retireCaptureRetry").mockImplementation(async function (id) {
      const delivery = await this.getDelivery(id);
      const file = storage.files.get(`${delivery.body_ref}.bytes`);
      const envelope = JSON.parse(new globalThis.TextDecoder().decode(file));
      expect(envelope.provenance).toEqual({ extension_instance_id: null, acquisition_sequence: null });
      expect(await this.getCaptureRetryEnvelope(id)).toEqual(input(id === "z-first" ? "first" : "second"));
      seen.push(id);
      return retire(id);
    });
    const { page } = await start();
    expect(await page()).toMatchObject({ ok: true, total: 2 });
    expect(seen).toEqual(["z-first", "a-second"]);
    expect(await store.listCaptureRetries()).toEqual([]);
    expect(network).not.toHaveBeenCalled();
  });

  it("retains originals after a body close fault and retries the same deterministic publication", async () => {
    const { store, storage, start } = await setup();
    await store.importCaptureRetries([{ metadata: { id: "kept" }, envelope: input("exact original") }]);
    storage.directory.failClose = (name) => name.endsWith(".bytes");
    const { page } = await start();
    expect(await page()).toMatchObject({ ok: false, error: "capture_staging_quota_exceeded" });
    expect(await store.getCaptureRetryEnvelope("kept")).toEqual(input("exact original"));
    storage.directory.failClose = null;
    expect(await page()).toMatchObject({ ok: true, total: 1 });
    expect(await store.listCaptureRetries()).toEqual([]);
    const deliveries = [];
    for await (const { entry } of store.deliveries()) deliveries.push(entry);
    expect(deliveries).toHaveLength(1);
    expect(deliveries[0].body_ref).toBe((await start()).staging.conversionId("kept"));
  });

  it("restarts after source retirement failure without rewriting or duplicating published bytes", async () => {
    const { store, storage, start } = await setup();
    await store.importCaptureRetries([{ metadata: { id: "kept" }, envelope: input("exact original") }]);
    const fault = vi.spyOn(store.constructor.prototype, "retireCaptureRetry").mockRejectedValue(new Error("neutral retirement fault"));
    const first = await start();
    expect(await first.page()).toMatchObject({ ok: false });
    const before = await store.getDelivery("kept");
    const bytes = storage.files.get(`${before.body_ref}.bytes`).slice();
    expect(await store.getCaptureRetryEnvelope("kept")).toEqual(input("exact original"));
    fault.mockRestore();
    const reopened = await start({ cold: true });
    expect(await reopened.page()).toMatchObject({ ok: true, total: 1 });
    expect(await store.getDelivery("kept")).toMatchObject({ body_ref: before.body_ref, delivery_sequence: before.delivery_sequence });
    expect(storage.files.get(`${before.body_ref}.bytes`)).toEqual(bytes);
    expect(await store.listCaptureRetries()).toEqual([]);
  });

  it.each(["accepted", "noop", "superseded"])("retains the %s receipt until source retirement settles, then removes only owned bytes", async (outcome) => {
    const { store, storage, network, start } = await setup();
    await store.importCaptureRetries([{ metadata: { id: "kept" }, envelope: input("exact original") }]);
    const fault = vi.spyOn(store.constructor.prototype, "retireCaptureRetry").mockRejectedValue(new Error("neutral retirement fault"));
    const first = await start();
    expect(await first.page()).toMatchObject({ ok: false });
    const delivery = await store.getDelivery("kept");
    const meta = await first.staging.metadata(delivery.body_ref);
    const receipt = { outcome, receiver_request_id: "neutral-receiver-request", submitted_content_hash: meta.sha256,
      content_hash: outcome === "accepted" ? meta.sha256 : "a".repeat(64) };
    await first.staging.markAcknowledged(delivery.body_ref, receipt);
    await store.deleteDelivery("kept");
    await first.staging.acknowledge(delivery.body_ref);
    expect(storage.files.has(`${delivery.body_ref}.bytes`)).toBe(true);
    expect((await first.staging.metadata(delivery.body_ref)).receiver_receipt).toEqual(receipt);
    fault.mockRestore();
    const restarted = await start({ cold: true });
    expect(await restarted.page()).toMatchObject({ ok: true, total: 0 });
    expect(await store.listCaptureRetries()).toEqual([]);
    expect(await store.getDelivery("kept")).toBeNull();
    expect(storage.files.has(`${delivery.body_ref}.bytes`)).toBe(false);
    expect(network).not.toHaveBeenCalled();
  });
});
