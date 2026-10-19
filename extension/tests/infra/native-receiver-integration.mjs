// Driven by the managed Python integration selection, never by Vitest.
// Browser APIs are synthetic; the background owner and HTTP receiver are real.
import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { readFile } from "node:fs/promises";
import process from "node:process";
import { setImmediate } from "node:timers/promises";
import { URL } from "node:url";
import { IDBFactory, IDBKeyRange } from "fake-indexeddb";
import { IndexedDbBackfillStore } from "../../src/backfill/storage.js";
import { CaptureStaging } from "../../src/capture/staging.js";
import { memoryOriginStorage } from "./capture-staging.js";
import { nativeRuntime } from "./native-port.mjs";
import { nativeFetch } from "../../src/background/native_fetch.js";

const [baseUrl, receiverId, commandJson, fixture, provider, nativeId] = process.argv.slice(2);
const transport = nativeRuntime(JSON.parse(commandJson));
globalThis.IDBKeyRange = IDBKeyRange;
globalThis.indexedDB = new IDBFactory();
const origin = memoryOriginStorage();
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { storage: origin } });
// The original runtime constructs its durable store when this module loads.
const { startBackgroundRuntime } = await import("../../src/background/runtime.js");
const status = await (await nativeFetch(transport, `${baseUrl}/v1/status`, { receiverId })).json();
const values = { receiverBaseUrl: baseUrl,
  polylogueReceiverPairing: { state: "online", receiver_id: status.receiver_id, api_schema: status.api_schema, endpoint: baseUrl } };
const storageArea = (values) => ({
  async get(defaults) { return { ...defaults, ...values }; },
  async set(patch) { Object.assign(values, patch); },
  async remove(keys) { for (const key of Array.isArray(keys) ? keys : [keys]) delete values[key]; },
});
const host = provider === "chatgpt" ? "chatgpt.com" : provider === "claude-ai" ? "claude.ai" : "grok.com";
const page = provider === "chatgpt" ? "c" : "chat";
const sender = { tab: { id: 42, url: `https://${host}/${page}/${nativeId}` }, documentId: "original-synthetic-document" };
let listener;
const event = { addListener() {} };
const calls = [];
let assetRequests = 0;
const adapters = {
  storage: { local: storageArea(values), session: storageArea({}) },
  alarms: { async create() {}, async clear() {}, onAlarm: event },
  runtime: { id: "synthetic-integration-extension", getManifest: () => ({ version: "0.3.0" }),
    onMessage: { addListener(fn) { listener = fn; } }, onInstalled: event, onStartup: event },
  action: { async setBadgeText() {}, async setBadgeBackgroundColor() {} },
  permissions: { async contains() { return true; } },
  tabs: { onActivated: event, onUpdated: event, onRemoved: event,
    async get() { return sender.tab; }, async query() { return []; },
    async sendMessage(tabId, message, options) {
      assert.equal(tabId, sender.tab.id);
      assert.equal(options.documentId, sender.documentId);
      assert.equal(message.type, "polylogue.acquireRecordAssets");
      assetRequests += 1;
      return { ok: true, acquisition: { attachments: [], outcome: { failed: [{ status: "no_resolvable_source" }] } } };
    } },
  network: async (url, options) => { calls.push({ path: new URL(url).pathname, method: options?.method || "GET" }); return nativeFetch(transport, url, options); },
  now: () => Date.now(), log() {},
};
startBackgroundRuntime(adapters);
// Settle startup reconciliation before admitting a new synthetic observation.
for (let turn = 0; turn < 20; turn++) await setImmediate();
const store = new IndexedDbBackfillStore();
const staging = new CaptureStaging(origin, store);
const raw = await readFile(fixture);
const payload = JSON.parse(raw);
const members = provider === "grok" ? Object.fromEntries(Object.entries(payload).map(([name, body]) => [name, Buffer.from(JSON.stringify(body))])) : { conversation: raw };
const owner = { tab_id: sender.tab.id, document_id: sender.documentId, provider };
const refs = {};
for (const [name, bytes] of Object.entries(members)) {
  const ref = await staging.begin(owner, { kind: "native-response", source_url: `https://${host}/synthetic-original/${nativeId}/${name}` });
  await staging.append(ref, owner, 0, bytes.toString("base64"));
  await staging.seal(ref, owner);
  refs[name] = ref;
}
const send = (message) => new Promise((resolve) => listener(message, sender, resolve));
const summarized = await send({ type: "polylogue.nativeCaptureSummary", provider, native_id: nativeId,
  raw_ref: refs.responses || refs.conversation,
  related_refs: provider === "grok" ? { conversation: refs.conversation, ...(refs.response_nodes ? { response_nodes: refs.response_nodes } : {}) } : {} });
assert.equal(summarized.ok, true, JSON.stringify(summarized));
assert.equal(typeof summarized.summary.needs_follow_up, "boolean");
assert.equal(assetRequests, 0);
const prefix = await store.getCapture(`native-preparation:${(refs.responses || refs.conversation).id}`);
assert.equal(prefix.state, "normalizing");
assert.equal(prefix.receiver_summary.raw_revision, summarized.raw_revision);
assert.equal(prefix.receiver_summary.summary.needs_follow_up, summarized.summary.needs_follow_up);
assert.ok(!calls.some((call) => call.path.endsWith("/native/asset") || call.path.endsWith("/native/finalize") || call.path.endsWith("/native/publish")));
const normalized = await send({ type: "polylogue.normalizeNativeCapture", provider, native_id: nativeId,
  raw_ref: refs.responses || refs.conversation,
  related_refs: provider === "grok" ? { conversation: refs.conversation, ...(refs.response_nodes ? { response_nodes: refs.response_nodes } : {}) } : {} });
assert.equal(normalized.ok, true, JSON.stringify(normalized));
assert.equal(normalized.envelope.provenance.extension_instance_id, null);
assert.equal(normalized.envelope.provenance.acquisition_sequence, null);
const root = await store.getCapture(normalized.envelope.capture_record_ref);
assert.equal(root.owner.document_id, sender.documentId);
const published = await send({ type: "polylogue.capture", envelope: normalized.envelope, request_id: "synthetic-integration-delivery" });
assert.equal(published.ok, true, JSON.stringify(published));
assert.equal(published.content_hash, normalized.envelope.receiver_native.sha256);
assert.ok(published.receiver_request_id);
assert.ok(calls.some((call) => call.path.endsWith("/native/member") && call.method === "PUT"));
assert.ok(calls.some((call) => call.path.endsWith("/native/publish") && call.method === "POST"));
assert.ok(!calls.some((call) => call.path === "/v1/browser-captures" && call.method === "POST"));
process.stdout.write(JSON.stringify({ sha256: published.content_hash, receiver_request_id: published.receiver_request_id, summary: summarized.summary, calls }));
