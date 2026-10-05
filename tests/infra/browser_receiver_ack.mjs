import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { BackfillCoordinator } from "../../browser-extension/src/backfill/coordinator.js";
import { MemoryBackfillStore } from "../../browser-extension/src/backfill/storage.js";

const { endpoint, capture } = JSON.parse(readFileSync(0, "utf8"));
const store = new MemoryBackfillStore();
let now = 100000;
let submissions = 0;
// The capture body the coordinator prepares stays staged until the receiver
// ACK hook retires it, exactly as the extension runtime's capture staging does.
const staged = new Map();
const adapter = {
  enumerate: async () => ({ classification: "success", items: [{ native_id: capture.session.provider_session_id,
    updated_at: capture.session.updated_at }], done: true, request_count: 1 }),
  fetchNative: async () => ({ ok: true, status: 200 }),
  classifyResponse: () => "success",
  normalizeCapture: async () => capture,
};
const coordinator = new BackfillCoordinator({
  store, adapters: { chatgpt: adapter },
  prepareCapture: async (envelope) => {
    const body = JSON.stringify(envelope);
    staged.set(envelope.capture_id, body);
    return { body, contentHash: createHash("sha256").update(body).digest("hex") };
  },
  receiverAcked: async (envelope) => { staged.delete(envelope.capture_id); },
  receiver: async (_capture, serialized) => {
    submissions += 1;
    const response = await fetch(endpoint, { method: "POST", headers: {
      "Content-Type": "application/json", Origin: "chrome-extension://polylogue-test",
    }, body: serialized });
    if (response.status !== 202) throw new Error(`receiver_http_${response.status}`);
    return { ...await response.json(), receiver_request_id: response.headers.get("X-Request-ID") };
  },
  alarms: { create: async () => {} }, clock: () => now, random: () => 0,
  instanceId: "neutral-ack-owner",
});
const job = await coordinator.start({ provider: "chatgpt", policy: { baseCadenceMs: 1000 } });
await coordinator.wake(job.id);
now += 1000;
await coordinator.wake(job.id);
now += 1000;
await coordinator.wake(job.id);
const page = await store.queuePage(job.id, { pageSize: 1 });
if (page.total !== 1 || page.has_more) throw new Error(`expected one queued item, got ${JSON.stringify(page)}`);
const item = page.items[0];
console.log(JSON.stringify({ item, revision: await store.getRevision(item.provider, item.native_id) || null,
  job: await coordinator.status(job.id), submissions, staged_files: staged.size }));
