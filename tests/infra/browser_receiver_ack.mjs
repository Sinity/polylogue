import { readFileSync } from "node:fs";
import { BackfillCoordinator } from "../../browser-extension/src/backfill/coordinator.js";
import { MemoryBackfillStore } from "../../browser-extension/src/backfill/storage.js";

const { endpoint, capture } = JSON.parse(readFileSync(0, "utf8"));
const store = new MemoryBackfillStore();
let now = 100000;
let submissions = 0;
const adapter = {
  enumerate: async () => ({ classification: "success", items: [{ native_id: capture.session.provider_session_id,
    updated_at: capture.session.updated_at }], done: true, request_count: 1 }),
  fetchNative: async () => ({ ok: true, status: 200 }),
  classifyResponse: () => "success",
};
const coordinator = new BackfillCoordinator({
  store, adapters: { chatgpt: adapter }, captureOverride: async () => capture,
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
const item = (await store.listQueue(job.id))[0];
console.log(JSON.stringify({ item, revision: await store.getRevision(item.provider, item.native_id) || null,
  job: await coordinator.status(job.id), submissions }));
