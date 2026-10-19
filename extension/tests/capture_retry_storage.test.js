import { beforeEach, describe, expect, it } from "vitest";
import { IDBFactory, IDBKeyRange } from "fake-indexeddb";
import { BACKFILL_DB_NAME } from "../src/backfill/models.js";
import { IndexedDbBackfillStore } from "../src/backfill/storage.js";

describe("retained automatic capture bodies", () => {
  beforeEach(() => { globalThis.IDBKeyRange = IDBKeyRange; });
  it("adds retry stores without changing existing backfill jobs and queue inputs", async () => {
    const factory = new IDBFactory();
    await new Promise((resolve, reject) => {
      const opening = factory.open(BACKFILL_DB_NAME, 2);
      opening.onerror = () => reject(opening.error);
      opening.onupgradeneeded = () => {
        const db = opening.result;
        db.createObjectStore("jobs", { keyPath: "id" }).put({ id: "existing-job", status: "paused" });
        db.createObjectStore("revisions", { keyPath: "id" });
        const queue = db.createObjectStore("queue", { keyPath: "id" });
        queue.createIndex("job_state_next", ["job_id", "state", "next_eligible_at_ms"]);
        queue.createIndex("job_native", ["job_id", "provider", "native_id"], { unique: true });
        queue.put({ id: "existing-input", job_id: "existing-job", provider: "grok", native_id: "neutral", state: "captured_waiting_receiver", envelope: { session: { turns: [] } } });
      };
      opening.onsuccess = () => { opening.result.close(); resolve(); };
    });
    const store = new IndexedDbBackfillStore(factory);
    // Database admission adds the declared active-page index key to paused jobs.
    expect(await store.getJob("existing-job")).toEqual({ id: "existing-job", status: "paused", active_page: 1 });
    expect((await store.queuePage("existing-job")).items[0]).toMatchObject({ state: "captured_waiting_receiver", envelope: { session: { turns: [] } } });
    await store.putCaptureRetry({ id: "automatic-input" }, { session: { turns: [] } });
    expect((await store.listCaptureRetries()).map((entry) => entry.id)).toEqual(["automatic-input"]);
  });

  it("reopens exact bodies separately from ordered metadata and retires both atomically", async () => {
    const factory = new IDBFactory();
    const original = new IndexedDbBackfillStore(factory);
    const envelope = { session: { provider: "grok", provider_session_id: "neutral",
      turns: [{ text: "body", attachments: [{ inline_base64: "AA==", metadata: { exact: null } }] }] } };
    await original.putCaptureRetry({ id: "z-first", attempts: 0 }, envelope);
    await original.putCaptureRetry({ id: "a-second", attempts: 0 }, envelope);
    // Interrupted old-cache transfer uses the original id, not another owner.
    await original.putCaptureRetry({ id: "z-first", attempts: 1 }, envelope);
    const reopened = new IndexedDbBackfillStore(factory);
    const metadata = await reopened.listCaptureRetries();
    expect(metadata.map((item) => item.id)).toEqual(["z-first", "a-second"]);
    expect(metadata.every((item) => !Object.hasOwn(item, "envelope"))).toBe(true);
    expect(await reopened.getCaptureRetryEnvelope("z-first")).toEqual(envelope);
    await reopened.replaceCaptureRetryMetadata([metadata[1]]);
    expect(await reopened.listCaptureRetries()).toEqual([metadata[1]]);
    await expect(reopened.getCaptureRetryEnvelope("z-first")).rejects.toThrow("capture_retry_body_missing");
    expect(await reopened.getCaptureRetryEnvelope("a-second")).toEqual(envelope);
  });

  it("does not resurrect delivered bodies from an unrefreshed old cache", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbBackfillStore(factory);
    const input = [{ metadata: { id: "old-input" }, envelope: { session: { turns: [] } } }];
    await store.importCaptureRetries(input);
    await store.replaceCaptureRetryMetadata([]);
    const restarted = new IndexedDbBackfillStore(factory);
    await restarted.importCaptureRetries(input);
    expect(await restarted.listCaptureRetries()).toEqual([]);
    await expect(restarted.getCaptureRetryEnvelope("old-input")).rejects.toThrow("capture_retry_body_missing");
  });

  it("settles failed retirement without deleting any original body", async () => {
    const store = new IndexedDbBackfillStore(new IDBFactory());
    const envelope = { session: { turns: [{ text: "retained" }] } };
    await store.putCaptureRetry({ id: "first" }, envelope);
    await store.putCaptureRetry({ id: "second" }, envelope);
    const metadata = await store.listCaptureRetries();
    await expect(store.replaceCaptureRetryMetadata([{ ...metadata[1], uncloneable: () => {} }])).rejects.toThrow();
    expect(await store.listCaptureRetries()).toEqual(metadata);
    expect(await store.getCaptureRetryEnvelope("first")).toEqual(envelope);
    expect(await store.getCaptureRetryEnvelope("second")).toEqual(envelope);
  });

  it("rolls back old-input transfer and its marker before retrying", async () => {
    const store = new IndexedDbBackfillStore(new IDBFactory());
    await expect(store.importCaptureRetries([
      { metadata: { id: "first" }, envelope: { session: { turns: [] } } },
      { metadata: { id: "failed" }, envelope: { uncloneable: () => {} } },
    ])).rejects.toThrow();
    expect(await store.listCaptureRetries()).toEqual([]);
    await store.importCaptureRetries([{ metadata: { id: "first" }, envelope: { session: { turns: [] } } }]);
    expect((await store.listCaptureRetries()).map((entry) => entry.id)).toEqual(["first"]);
  });

  it("rolls back metadata when retaining the original body fails", async () => {
    const store = new IndexedDbBackfillStore(new IDBFactory());
    await store.putCaptureRetry({ id: "kept" }, { session: { turns: [] } });
    await expect(store.putCaptureRetry({ id: "failed" }, { uncloneable: () => {} })).rejects.toThrow();
    expect((await store.listCaptureRetries()).map((item) => item.id)).toEqual(["kept"]);
    await expect(store.getCaptureRetryEnvelope("failed")).rejects.toThrow("capture_retry_body_missing");
  });
});
